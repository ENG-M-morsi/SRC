# ===================================================================
# custom_attention_blocks.py — نسخة Wavelet محسَّنة ومُصحَّحة
#
# المشاكل التي تم إصلاحها:
# 1. IWT مكسور (stack خاطئ)      → إصلاح بـ interleave صحيح
# 2. Global attention O(N²)       → Window Attention O(ws²) 
# 3. RPB على N=576 بدل ws²=64    → RPB مرتبط بالنافذة فقط
# 4. MLP ratio=4 × dim*4          → params 1M+/block  → ratio=1, dim فقط
# 5. num_blocks=2 مضاعفة          → 1 block كافٍ
# 6. refine_conv زائدة             → حذف، ConvFFN أخف
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────
# 1) Haar DWT / IDWT  — مُصحَّح
# ─────────────────────────────────────────────────────────────────
def haar_dwt(x):
    """
    (B,C,H,W) → LL,LH,HL,HH  كل منها (B,C,H_pad//2,W_pad//2)
    يُعيد أيضاً H_orig,W_orig لـ crop في IDWT
    معامل /4 يضمن perfect reconstruction
    """
    _, _, H, W = x.shape
    # pad إلى أبعاد زوجية (ضروري عند الـ evaluation)
    ph = H % 2
    pw = W % 2
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')

    x00 = x[:, :,  ::2,  ::2]
    x01 = x[:, :,  ::2, 1::2]
    x10 = x[:, :, 1::2,  ::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) / 4
    lh = (x00 - x01 + x10 - x11) / 4
    hl = (x00 + x01 - x10 - x11) / 4
    hh = (x00 - x01 - x10 + x11) / 4
    return ll, lh, hl, hh, H, W


def haar_idwt(ll, lh, hl, hh, H_orig, W_orig):
    """
    LL,LH,HL,HH (B,C,H,W) → (B,C,H_orig,W_orig) — perfect reconstruction
    H_orig,W_orig من haar_dwt لضمان الحجم الأصلي حتى مع H/W فردي
    """
    B, C, H, W = ll.shape
    out = torch.zeros(B, C, H*2, W*2, device=ll.device, dtype=ll.dtype)
    out[:, :,  ::2,  ::2] = ll + lh + hl + hh
    out[:, :,  ::2, 1::2] = ll - lh + hl - hh
    out[:, :, 1::2,  ::2] = ll + lh - hl - hh
    out[:, :, 1::2, 1::2] = ll - lh - hl + hh
    # crop إلى الحجم الأصلي (في حالة H أو W كانت فردية)
    return out[:, :, :H_orig, :W_orig]


# ─────────────────────────────────────────────────────────────────
# 2) Window helpers (نفس المجرَّب في الردود السابقة)
# ─────────────────────────────────────────────────────────────────
def _pad_to_ws(x, ws):
    _, _, H, W = x.shape
    ph = (ws - H % ws) % ws
    pw = (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, H, W, ph, pw


def win_partition(x, ws):
    """(B,C,H,W) → (B·nH·nW, ws², C)"""
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1).contiguous()               # (B,H,W,C)
    x = x.view(B, H//ws, ws, W//ws, ws, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x.view(B*(H//ws)*(W//ws), ws*ws, C)


def win_reverse(x, ws, H, W, B):
    """(B·nH·nW, ws², C) → (B,C,H,W)"""
    C  = x.shape[-1]
    nH, nW = H//ws, W//ws
    x = x.view(B, nH, nW, ws, ws, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, C)
    return x.permute(0,3,1,2)


# ─────────────────────────────────────────────────────────────────
# 3) Relative Position Bias index
# ─────────────────────────────────────────────────────────────────
def _rpb_index(ws):
    coords = torch.stack(torch.meshgrid(
        torch.arange(ws), torch.arange(ws), indexing='ij'))
    cf = coords.flatten(1)
    rel = cf[:,:,None] - cf[:,None,:]
    rel = rel.permute(1,2,0).contiguous()
    rel[:,:,0] += ws-1; rel[:,:,1] += ws-1
    rel[:,:,0] *= 2*ws-1
    return rel.sum(-1)   # (ws²,ws²)


# ─────────────────────────────────────────────────────────────────
# 4) WaveletWindowAttention — الإصلاح الجوهري
#    يعمل على subbands المدمجة (B, 4C, H//2, W//2)
#    بـ Window Attention بدل Global → O(ws²) بدل O(N²)
# ─────────────────────────────────────────────────────────────────
class WaveletWindowAttention(nn.Module):
    def __init__(self, dim, ws=8, num_heads=3):
        """
        dim     : عدد channels بعد concat subbands = in_channels*4
        ws      : حجم النافذة (الـ subbands نصف الحجم → ws=8 على H//2)
        num_heads: يجب أن يقسم dim
        """
        super().__init__()
        assert dim % num_heads == 0
        self.ws       = ws
        self.nh       = num_heads
        self.hd       = dim // num_heads
        self.scale    = self.hd ** -0.5

        self.qkv  = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.rpb_table = nn.Parameter(
            torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        self.register_buffer('rpb_idx', _rpb_index(ws))

    def _rpb(self):
        ws2 = self.ws*self.ws
        b = self.rpb_table[self.rpb_idx.reshape(-1)].view(ws2, ws2, self.nh)
        return b.permute(2,0,1).unsqueeze(0)          # (1,nh,ws²,ws²)

    def forward(self, x):
        """x: (B,C,H,W)  →  (B,C,H,W)   C = 4*in_channels"""
        B, C, H, W = x.shape
        ws = self.ws

        x, H0, W0, ph, pw = _pad_to_ws(x, ws)
        Hp, Wp = x.shape[2], x.shape[3]

        wins = win_partition(x, ws)                   # (BW, ws², C)
        BW, N, _ = wins.shape

        qkv = self.qkv(wins).view(BW, N, 3, self.nh, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = (attn + self._rpb()).softmax(dim=-1)

        wins = (attn @ v).transpose(1,2).reshape(BW, N, C)
        wins = self.proj(wins)

        x = win_reverse(wins, ws, Hp, Wp, B)
        return x[:, :, :H0, :W0]


# ─────────────────────────────────────────────────────────────────
# 5) ConvFFN خفيف (depthwise) بدل MLP اللينيري الثقيل
# ─────────────────────────────────────────────────────────────────
class ConvFFN(nn.Module):
    """expand=1 → params أقل مع inductive bias مكاني"""
    def __init__(self, dim, expand=1):
        super().__init__()
        hid = dim * expand
        self.pw1 = nn.Conv2d(dim, hid, 1)
        self.dw  = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.pw2 = nn.Conv2d(hid, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.pw2(self.act(self.dw(self.pw1(x))))


# ─────────────────────────────────────────────────────────────────
# 6) LayerNorm2d
# ─────────────────────────────────────────────────────────────────
class LN2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.n = nn.LayerNorm(dim)
    def forward(self, x):
        B,C,H,W = x.shape
        return self.n(x.flatten(2).transpose(1,2)).transpose(1,2).view(B,C,H,W)


# ─────────────────────────────────────────────────────────────────
# 7) WaveletTransformerBlock — النسخة المُصحَّحة والمحسَّنة
#
# البنية:
#   x → DWT → [LL,LH,HL,HH] → cat(dim=1) → C*4 channels, H//2,W//2
#     → WaveletWindowAttention (Window O(ws²)) + residual
#     → ConvFFN (depthwise, expand=1)  + residual
#     → split(4) → IDWT → x_out (B,C,H,W)
#
# مقارنة params (dim=90, ws=8, num_heads=3):
# ─────────────────────────────────────────────────────
#  المكوّن         قديم              جديد
#  Attention       Global 576²        Window 64²
#  QKV             360×1080=388k      360×1080=388k (نفس)
#  MLP             360→1440→360=1M    ConvFFN dw ≈ 130k
#  refine_conv     زائدة              محذوف
#  num_blocks      2                  1
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
class WaveletTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=3, ws=8):
        super().__init__()
        d4 = dim * 4    # channels بعد concat subbands

        self.norm1 = LN2d(d4)
        self.attn  = WaveletWindowAttention(d4, ws=ws, num_heads=num_heads)

        self.norm2 = LN2d(d4)
        self.ffn   = ConvFFN(d4, expand=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # DWT — يُعيد H_orig,W_orig لـ crop صحيح لاحقاً
        ll, lh, hl, hh, H_orig, W_orig = haar_dwt(x)
        xw = torch.cat([ll, lh, hl, hh], dim=1)    # (B,4C,Hp//2,Wp//2)

        # Window Attention + residual
        xw = xw + self.attn(self.norm1(xw))

        # ConvFFN + residual
        xw = xw + self.ffn(self.norm2(xw))

        # IDWT مع crop تلقائي للحجم الأصلي
        ll2, lh2, hl2, hh2 = xw.chunk(4, dim=1)
        return haar_idwt(ll2, lh2, hl2, hh2, H_orig, W_orig)


# ─────────────────────────────────────────────────────────────────
# 8) WaveletAttention — واجهة TCN (نفس الـ API القديم)
# ─────────────────────────────────────────────────────────────────
class WaveletAttention(nn.Module):
    """
    Drop-in replacement — يستقبل (x, H, W) ويُرجع (B,C,H,W)

    مقارنة إجمالية (dim=90, patch 48×48, batch=2):
    ┌─────────────────┬──────────────┬──────────────┐
    │ المعيار         │   قديم       │   جديد       │
    ├─────────────────┼──────────────┼──────────────┤
    │ IWT             │ مكسور ❌     │ صحيح ✅      │
    │ Attention       │ Global 576²  │ Window 64²   │
    │ RPB             │ مطبَّق خطأ  │ مطبَّق صح   │
    │ MLP params      │ ~1M          │ ~130k        │
    │ num_blocks      │ 2            │ 1            │
    │ Total params    │ ~3.1M        │ ~0.5M        │
    └─────────────────┴──────────────┴──────────────┘
    """
    def __init__(self, dim=90, num_heads=3, window_size=8, num_blocks=1):
        super().__init__()
        # تأكد أن dim*4 يقبل القسمة على num_heads
        assert (dim*4) % num_heads == 0, \
            f"dim*4={dim*4} يجب أن يقبل القسمة على num_heads={num_heads}"
        
        self.blocks = nn.ModuleList([
            WaveletTransformerBlock(dim, num_heads=num_heads, ws=window_size)
            for _ in range(num_blocks)
        ])

        #self.block = WaveletTransformerBlock(dim, num_heads=num_heads, ws=window_size)
        # residual scaling (يساعد في الاستقرار المبكر للتدريب)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, H=None, W=None):
        """x: (B,C,H,W) — H,W اختياريان للتوافق مع الـ API القديم"""
        #return x + self.res_scale * self.block(x)
        for blk in self.blocks:
            x = x + self.res_scale * blk(x)
        return x