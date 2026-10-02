# ===================================================================
# custom_attention_blocks.py — النسخة المحسَّنة النهائية
#
# المشاكل في النسخة القديمة:
#   - Global Attention: O(N²) حيث N=H×W → بطيء لـ patches كبيرة
#   - OverlapPatchEmbed: conv3×3 إضافية بلا فائدة
#   - لا يوجد local inductive bias مناسب لـ SR
#
# الحل — SWDA (Shifted-Window Dual Aggregation):
#   1. Window Attention (ws=8)  → O(N·ws²) بدل O(N²)
#   2. Shifted Window → تغطية سياق أوسع بلا params إضافية
#   3. ConvFFN (depthwise) بدل Linear MLP → inductive bias محلي
#   4. حذف OverlapPatchEmbed الزائدة
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# LayerNorm2d
# ─────────────────────────────────────────────────────────────────────
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────
# Window helpers
# ─────────────────────────────────────────────────────────────────────
def _pad_to_window(x, ws):
    _, _, H, W = x.shape
    ph = (ws - H % ws) % ws
    pw = (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, H, W, ph, pw


def window_partition(x, ws):
    """(B,C,H,W) → (B*nH*nW, ws*ws, C)"""
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()           # (B,H,W,C)
    x = x.view(B, H//ws, ws, W//ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B * (H//ws) * (W//ws), ws*ws, C)


def window_reverse(x, ws, H, W, B):
    """(B*nH*nW, ws*ws, C) → (B,C,H,W)"""
    C  = x.shape[-1]
    nH, nW = H//ws, W//ws
    x = x.view(B, nH, nW, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x.permute(0, 3, 1, 2)


# ─────────────────────────────────────────────────────────────────────
# Relative Position Bias index (يُحسَب مرة واحدة)
# ─────────────────────────────────────────────────────────────────────
def _make_rpb_index(ws):
    coords = torch.stack(torch.meshgrid(
        torch.arange(ws), torch.arange(ws), indexing='ij'))   # (2,ws,ws)
    coords_flat = coords.flatten(1)                           # (2,ws²)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2,ws²,ws²)
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += ws - 1
    rel[:, :, 1] += ws - 1
    rel[:, :, 0] *= 2*ws - 1
    return rel.sum(-1)                                        # (ws²,ws²)


# ─────────────────────────────────────────────────────────────────────
# WindowAttention
# ─────────────────────────────────────────────────────────────────────
class WindowAttention(nn.Module):
    def __init__(self, dim, ws=8, num_heads=3):
        super().__init__()
        assert dim % num_heads == 0
        self.ws        = ws
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.rpb_table = nn.Parameter(
            torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        self.register_buffer('rpb_index', _make_rpb_index(ws))

    def _rpb(self):
        ws2 = self.ws * self.ws
        b   = self.rpb_table[self.rpb_index.reshape(-1)].view(ws2, ws2, self.num_heads)
        return b.permute(2, 0, 1).unsqueeze(0)               # (1, nh, ws², ws²)

    def forward(self, x, mask=None):
        """
        x:    (BW, ws², C)   BW = B * nH * nW
        mask: (nH*nW, 1, ws², ws²)  أو None
        """
        BW, N, C = x.shape
        h = self.num_heads
        d = self.head_dim

        qkv = self.qkv(x).view(BW, N, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                              # (BW, h, N, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._rpb()                            # broadcast (1,h,N,N)

        if mask is not None:
            # mask: (nW, 1, N, N)
            # attn: (BW, h, N, N)  حيث BW = B * nW
            nW   = mask.shape[0]
            B_   = BW // nW
            attn = attn.view(B_, nW, h, N, N)
            # mask (nW,1,N,N) → unsqueeze(0) → (1,nW,1,N,N)  ✅ broadcast مع (B_,nW,h,N,N)
            attn = attn + mask.unsqueeze(0)
            attn = attn.view(BW, h, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(BW, N, C)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────
# ConvFFN — Depthwise Feed-Forward
# ─────────────────────────────────────────────────────────────────────
class ConvFFN(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        hid = dim * expand
        self.pw1 = nn.Conv2d(dim, hid, 1)
        self.dw  = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.pw2 = nn.Conv2d(hid, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.pw2(self.act(self.dw(self.pw1(x))))


# ─────────────────────────────────────────────────────────────────────
# Shift Mask  (يُحسَب ويُخزَّن في cache حسب (Hp, Wp))
# ─────────────────────────────────────────────────────────────────────
def _make_shift_mask(Hp, Wp, ws, shift, device):
    """يُرجع mask شكله (nH*nW, 1, ws², ws²)"""
    img  = torch.zeros(Hp, Wp, device=device)
    # تعبئة المناطق بأرقام مختلفة
    regions = [
        ((0, Hp-shift),  (0, Wp-shift)),   # 0
        ((0, Hp-shift),  (Wp-shift, Wp)),   # 1
        ((Hp-shift, Hp), (0, Wp-shift)),    # 2
        ((Hp-shift, Hp), (Wp-shift, Wp)),   # 3
    ]
    for idx, (hr, wr) in enumerate(regions):
        img[hr[0]:hr[1], wr[0]:wr[1]] = idx

    nH, nW = Hp//ws, Wp//ws
    img = img.view(nH, ws, nW, ws).permute(0, 2, 1, 3).contiguous()
    img = img.view(nH*nW, ws*ws)                             # (nW_total, ws²)

    mask = img.unsqueeze(1) - img.unsqueeze(2)               # (nW_total, ws², ws²)
    mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
    return mask.unsqueeze(1)                                  # (nW_total, 1, ws², ws²)


# ─────────────────────────────────────────────────────────────────────
# SWDABlock — Shifted-Window Dual Aggregation Block
# ─────────────────────────────────────────────────────────────────────
class SWDABlock(nn.Module):
    def __init__(self, dim, ws=8, num_heads=3, ffn_expand=2):
        super().__init__()
        self.ws    = ws
        self.shift = ws // 2

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.norm3 = LayerNorm2d(dim)

        self.w_attn  = WindowAttention(dim, ws, num_heads)   # بدون shift
        self.sw_attn = WindowAttention(dim, ws, num_heads)   # مع shift

        self.ffn = ConvFFN(dim, ffn_expand)

        self._mask_cache = {}

    def _get_mask(self, Hp, Wp, device):
        key = (Hp, Wp)
        if key not in self._mask_cache:
            m = _make_shift_mask(Hp, Wp, self.ws, self.shift, device)
            self._mask_cache[key] = m
        return self._mask_cache[key].to(device)

    def _apply_wattn(self, x, attn_mod, shift):
        B, C, H, W = x.shape
        ws = self.ws

        # Pad
        x, H_orig, W_orig, ph, pw = _pad_to_window(x, ws)
        Hp, Wp = x.shape[2], x.shape[3]

        # Shift
        if shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(2, 3))
            mask = self._get_mask(Hp, Wp, x.device)
        else:
            mask = None

        # Partition → Attend → Reverse
        wins = window_partition(x, ws)                       # (B*nH*nW, ws², C)
        wins = attn_mod(wins, mask)
        x    = window_reverse(wins, ws, Hp, Wp, B)          # (B, C, Hp, Wp)

        # Unshift
        if shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(2, 3))

        # Unpad
        return x[:, :, :H_orig, :W_orig]

    def forward(self, x):
        x = x + self._apply_wattn(self.norm1(x), self.w_attn,  shift=False)
        x = x + self._apply_wattn(self.norm2(x), self.sw_attn, shift=True)
        x = x + self.ffn(self.norm3(x))
        return x


# ─────────────────────────────────────────────────────────────────────
# DAT — Dual Aggregation Transformer (محسَّن)
#
# مقارنة مع النسخة القديمة (dim=90, patch 48×48):
# ┌──────────────┬────────────────┬─────────────────┐
# │ المعامل      │   قديم         │   جديد          │
# ├──────────────┼────────────────┼─────────────────┤
# │ Attention    │ Global O(N²)   │ Window O(N·ws²) │
# │ num_heads    │ 6              │ 3               │
# │ FFN          │ Linear 2×      │ ConvFFN dw 2×   │
# │ patch_embed  │ Conv3×3 زائدة  │ محذوف           │
# └──────────────┴────────────────┴─────────────────┘
# ─────────────────────────────────────────────────────────────────────
class DAT(nn.Module):
    def __init__(self, dim=90, num_heads=3, ws=8, num_blocks=1):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim={dim} يجب أن يقبل القسمة على num_heads={num_heads}"
        self.blocks = nn.ModuleList([
            SWDABlock(dim, ws=ws, num_heads=num_heads, ffn_expand=2)
            for _ in range(num_blocks)
        ])
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)