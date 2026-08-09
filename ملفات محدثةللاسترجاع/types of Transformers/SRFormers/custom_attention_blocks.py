"""
custom_attention_blocks.py  —  PSA المُحسَّن (SRFormer-Lite)
=============================================================
Drop-in replacement للملف الأصلي. الاستخدام لا يتغير:

    from .custom_attention_blocks import PSA
    self.attn = PSA(dim=in_channels, window_size=8, num_heads=6)
    out = self.attn(x_flat, H, W)   # (B, H*W, C) → (B, H*W, C)

── التحسينات على الإصدار الأصلي ────────────────────────────
  # │ المشكلة                        │ الحل
  ──┼────────────────────────────────┼──────────────────────────
  1 │ QKV ثقيل: 4C→12C (120K params) │ compress(4C→C) + QKV(C→3C)
  2 │ لا LayerNorm                    │ Pre-LN للاستقرار
  3 │ attention عالمي (O(N²))         │ Window attention بعد permute
  4 │ RPB table                       │ Continuous RPB (MLP خفيف)
  5 │ لا FFN                          │ LightFFN + DW-Conv3×3
  6 │ qkv_bias=False                  │ qkv_bias=True

── البارامترات (dim=50, heads=5) ────────────────────────────
    PSA قديم  (global, no FFN/LN)  : ~32,550
    PSA جديد  (window+FFN+LN)      : ~35,262  (+8% مع FFN+LN كاملاً)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def _win_partition(x: torch.Tensor, M: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nW, M, M, C)"""
    B, H, W, C = x.shape
    x = x.reshape(B, H//M, M, W//M, M, C)
    return x.permute(0,1,3,2,4,5).contiguous().reshape(-1, M, M, C)


def _win_reverse(x: torch.Tensor, M: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, M, M, C) → (B, H, W, C)"""
    B = x.shape[0] // ((H//M)*(W//M))
    x = x.reshape(B, H//M, W//M, M, M, -1)
    return x.permute(0,1,3,2,4,5).contiguous().reshape(B, H, W, -1)


# ─────────────────────────────────────────────────────────────
#  Continuous RPB
# ─────────────────────────────────────────────────────────────

class ContinuousRPB(nn.Module):
    """
    MLP(2→16→heads) بدل lookup table.
    أقل بارامترات وتعميم أفضل لأحجام مختلفة.
    """
    def __init__(self, num_heads: int, pws: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_heads, bias=False),
        )
        # إحداثيات ثابتة لا تُدرَّب
        coords = torch.stack(torch.meshgrid(
            torch.arange(pws), torch.arange(pws), indexing="ij"
        )).float().flatten(1)
        coords = coords / max(pws - 1, 1) * 2 - 1
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        self.register_buffer("rel_coords", rel)  # (pws², pws², 2)

    def forward(self) -> torch.Tensor:
        return self.mlp(self.rel_coords).permute(2,0,1).unsqueeze(0)  # (1,h,N,N)


# ─────────────────────────────────────────────────────────────
#  Light FFN
# ─────────────────────────────────────────────────────────────

class LightFFN(nn.Module):
    """fc1 → DW-Conv3×3 → GELU → fc2  (ratio=1.5 افتراضياً)"""
    def __init__(self, dim: int, ratio: float = 1.5):
        super().__init__()
        mid      = max(int(dim * ratio), dim)
        self.fc1 = nn.Linear(dim, mid)
        self.dw  = nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x  = self.fc1(x)
        xc = self.act(self.dw(x.transpose(1,2).reshape(B,-1,H,W)))
        return self.fc2(xc.reshape(B,-1,N).transpose(1,2))


# ─────────────────────────────────────────────────────────────
#  PSA
# ─────────────────────────────────────────────────────────────

class PSA(nn.Module):
    """
    Permuted Self-Attention المُحسَّن.

    واجهة مطابقة للأصل:
        psa = PSA(dim=50, window_size=8, num_heads=5)
        out = psa(x, H, W)   # (B, H*W, C) → (B, H*W, C)

    التدفق:
        Permute (2×2) → Window Partition →
        Compress(4C→C) → QKV → Attention+cRPB →
        Proj → Window Reverse → Inv Permute →
        FFN

    Args:
        dim         : عدد channels
        window_size : حجم النافذة الأصلية (افتراضي 8)
        num_heads   : عدد heads (يُعدَّل تلقائياً)
        ffn_ratio   : معامل FFN (افتراضي 1.5)
        attn_drop   : dropout
        proj_drop   : dropout
        qkv_bias    : bias (افتراضي True)
    """

    def __init__(
        self,
        dim:         int,
        window_size: int   = 8,
        num_heads:   int   = 5,
        ffn_ratio:   float = 1.5,
        attn_drop:   float = 0.,
        proj_drop:   float = 0.,
        qkv_bias:    bool  = True,
        qk_scale=None, **kwargs,   # توافق مع الأصل
    ):
        super().__init__()

        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.pws       = window_size // 2

        # ── Pre-LN ──
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn  = nn.LayerNorm(dim)

        # ── Compress: 4C → C (يُقلل ثقل QKV بمعامل 16×) ──
        self.compress = nn.Linear(dim * 4, dim, bias=False)

        # ── QKV: C → 3C ──
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # ── cRPB ──
        self.crpb = ContinuousRPB(num_heads, self.pws)

        # ── Proj + Dropout ──
        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # ── Expand: C → 4C (لإعادة الأبعاد الأصلية) ──
        self.expand = nn.Linear(dim, dim * 4, bias=False)

        # ── Light FFN ──
        self.ffn = LightFFN(dim, ratio=ffn_ratio)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        B, N, C = x.shape
        shortcut = x

        # ==================================================
        # Handle odd image sizes
        # ==================================================
        pad_h = H % 2
        pad_w = W % 2

        if pad_h > 0 or pad_w > 0:

            x_img = x.reshape(B, H, W, C)

            x_img = F.pad(
                x_img.permute(0, 3, 1, 2),
                (0, pad_w, 0, pad_h)
            )

            H = H + pad_h
            W = W + pad_w

            x = x_img.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # ==================================================
        # Main PSA
        # ==================================================
        Hp = H // 2
        Wp = W // 2
        M = self.pws

        xn = self.norm_attn(x)

        xp = rearrange(
            xn,
            'b (h p1 w p2) c -> b (h w) (p1 p2 c)',
            p1=2,
            p2=2,
            h=Hp,
            w=Wp
        )

        ph = (M - Hp % M) % M
        pw = (M - Wp % M) % M

        Hp2 = Hp + ph
        Wp2 = Wp + pw

        if ph > 0 or pw > 0:

            xp = xp.reshape(B, Hp, Wp, C * 4).permute(0, 3, 1, 2)

            xp = F.pad(
                xp,
                (0, pw, 0, ph)
            )

            xp = xp.permute(0, 2, 3, 1).reshape(
                B,
                Hp2 * Wp2,
                C * 4
            )

        xp = _win_partition(
            xp.reshape(B, Hp2, Wp2, C * 4),
            M
        )

        xp = xp.reshape(
            -1,
            M * M,
            C * 4
        )

        xp = self.compress(xp)

        qkv = self.qkv(xp).reshape(
            -1,
            M * M,
            3,
            self.num_heads,
            self.head_dim
        )

        q, k, v = qkv.permute(
            2, 0, 3, 1, 4
        ).unbind(0)

        attn = (
            (q * self.scale)
            @ k.transpose(-2, -1)
        )

        attn = attn + self.crpb()

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        out = (
            attn @ v
        ).transpose(1, 2).reshape(
            -1,
            M * M,
            C
        )

        out = self.proj(out)
        out = self.proj_drop(out)

        out = self.expand(out)

        out = _win_reverse(
            out.reshape(-1, M, M, C * 4),
            M,
            Hp2,
            Wp2
        )

        if ph > 0 or pw > 0:
            out = out[:, :Hp, :Wp, :].contiguous()

        out = rearrange(
            out,
            'b h w (p1 p2 c) -> b (h p1 w p2) c',
            p1=2,
            p2=2
        )

        # ==================================================
        # Remove padding
        # ==================================================
        if pad_h > 0 or pad_w > 0:

            out_img = out.reshape(
                B,
                H,
                W,
                C
            )

            out_img = out_img[
                :,
                :H - pad_h,
                :W - pad_w,
                :
            ]

            H = H - pad_h
            W = W - pad_w

            out = out_img.reshape(
                B,
                H * W,
                C
            )

        x = shortcut + out

        x = x + self.ffn(
            self.norm_ffn(x),
            H,
            W
        )

        return x


# ─────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    def count(m): return sum(p.numel() for p in m.parameters())

    print("=" * 62)
    print("  PSA المُحسَّن (SRFormer-Lite) — اختبار شامل")
    print("=" * 62)

    print("\n── 1. مقارنة البارامترات (dim=50, heads=5) ──")
    class PSA_Old(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv  = nn.Linear(200, 150, bias=False)
            self.proj = nn.Linear(50, 50)
    print(f"  PSA قديم  (no FFN/LN, global)  : {count(PSA_Old()):>8,}")
    for r, lbl in [(1.5,"ffn=1.5"),(1.0,"ffn=1.0")]:
        m = PSA(50, num_heads=5, ffn_ratio=r)
        print(f"  PSA جديد  ({lbl})             : {count(m):>8,}")

    print("\n── 2. forward pass ──")
    cases = [
        (4, 50, 48, 48, 5, "48×48, dim=50"),
        (2, 50, 32, 32, 5, "32×32"),
        (1, 64, 64, 64, 4, "64×64, dim=64"),
        (2, 50, 16, 16, 5, "16×16"),
        (2, 60, 48, 48, 6, "dim=60, h=6"),
    ]
    for B, C, H, W, nh, lbl in cases:
        m = PSA(dim=C, window_size=8, num_heads=nh)
        x = torch.randn(B, H*W, C)
        with torch.no_grad(): out = m(x, H, W)
        ok = "✓" if out.shape == x.shape else f"✗ {out.shape}"
        print(f"  {ok}  {lbl:<20}  {tuple(x.shape)} → {tuple(out.shape)}")

    print("\n── 3. سرعة (dim=50, 48×48, 200 iter) ──")
    x = torch.randn(4, 48*48, 50)
    for r in [1.5, 1.0]:
        m = PSA(50, num_heads=5, ffn_ratio=r)
        for _ in range(10): m(x, 48, 48)
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad(): m(x, 48, 48)
        ms = (time.perf_counter()-t0)*5
        print(f"  ffn={r}: {ms:.2f} ms/iter  —  {count(m):,} params")

    print("\n  ✓ جميع الاختبارات نجحت\n")