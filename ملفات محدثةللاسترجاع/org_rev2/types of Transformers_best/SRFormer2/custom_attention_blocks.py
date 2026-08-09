# ===================================================================
# custom_attention_blocks.py — PSA المُحسَّن مع دعم num_blocks
# ===================================================================

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
    B, H, W, C = x.shape
    x = x.reshape(B, H//M, M, W//M, M, C)
    return x.permute(0,1,3,2,4,5).contiguous().reshape(-1, M, M, C)

def _win_reverse(x: torch.Tensor, M: int, H: int, W: int) -> torch.Tensor:
    B = x.shape[0] // ((H//M)*(W//M))
    x = x.reshape(B, H//M, W//M, M, M, -1)
    return x.permute(0,1,3,2,4,5).contiguous().reshape(B, H, W, -1)

# ─────────────────────────────────────────────────────────────
#  Continuous RPB
# ─────────────────────────────────────────────────────────────

class ContinuousRPB(nn.Module):
    def __init__(self, num_heads: int, pws: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_heads, bias=False),
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(pws), torch.arange(pws), indexing="ij"
        )).float().flatten(1)
        coords = coords / max(pws - 1, 1) * 2 - 1
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        self.register_buffer("rel_coords", rel)

    def forward(self) -> torch.Tensor:
        return self.mlp(self.rel_coords).permute(2,0,1).unsqueeze(0)

# ─────────────────────────────────────────────────────────────
#  Light FFN
# ─────────────────────────────────────────────────────────────

class LightFFN(nn.Module):
    def __init__(self, dim: int, ratio: float = 1.5):
        super().__init__()
        mid = max(int(dim * ratio), dim)
        self.fc1 = nn.Linear(dim, mid)
        self.dw  = nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        xc = self.act(self.dw(x.transpose(1,2).reshape(B,-1,H,W)))
        return self.fc2(xc.reshape(B,-1,N).transpose(1,2))

# ─────────────────────────────────────────────────────────────
#  PSA_Block — بلوك PSA واحد (يُستخدم داخل PSA)
# ─────────────────────────────────────────────────────────────

class PSA_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 5,
        ffn_ratio: float = 1.5,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        qkv_bias: bool = True,
    ):
        super().__init__()
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pws = window_size // 2

        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        self.compress = nn.Linear(dim * 4, dim, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.crpb = ContinuousRPB(num_heads, self.pws)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.expand = nn.Linear(dim, dim * 4, bias=False)
        self.ffn = LightFFN(dim, ratio=ffn_ratio)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        shortcut = x

        # Handle odd sizes
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x_img = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x_img = F.pad(x_img, (0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
            x = x_img.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Permute
        Hp, Wp = H // 2, W // 2
        M = self.pws
        xn = self.norm_attn(x)
        xp = rearrange(xn, 'b (h p1 w p2) c -> b (h w) (p1 p2 c)', p1=2, p2=2, h=Hp, w=Wp)

        # Pad windows
        ph = (M - Hp % M) % M
        pw = (M - Wp % M) % M
        Hp2, Wp2 = Hp + ph, Wp + pw
        if ph > 0 or pw > 0:
            xp = xp.reshape(B, Hp, Wp, C * 4).permute(0, 3, 1, 2)
            xp = F.pad(xp, (0, pw, 0, ph))
            xp = xp.permute(0, 2, 3, 1).reshape(B, Hp2 * Wp2, C * 4)

        # Window attention
        xp = _win_partition(xp.reshape(B, Hp2, Wp2, C * 4), M)
        xp = xp.reshape(-1, M * M, C * 4)
        xp = self.compress(xp)

        qkv = self.qkv(xp).reshape(-1, M * M, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.crpb()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(-1, M * M, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = self.expand(out)

        out = _win_reverse(out.reshape(-1, M, M, C * 4), M, Hp2, Wp2)
        if ph > 0 or pw > 0:
            out = out[:, :Hp, :Wp, :].contiguous()

        out = rearrange(out, 'b h w (p1 p2 c) -> b (h p1 w p2) c', p1=2, p2=2)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out_img = out.reshape(B, H, W, C)[:, :H - pad_h, :W - pad_w, :]
            H, W = H - pad_h, W - pad_w
            out = out_img.reshape(B, H * W, C)

        x = shortcut + out
        x = x + self.ffn(self.norm_ffn(x), H, W)
        return x

# ─────────────────────────────────────────────────────────────
#  PSA — النسخة الرئيسية (تدعم num_blocks)
# ─────────────────────────────────────────────────────────────

class PSA(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 5,
        num_blocks: int = 1,
        ffn_ratio: float = 1.5,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            PSA_Block(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qkv_bias=qkv_bias,
            )
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, H, W)
        return self.norm(x)

# ===================================================================
#  اختبار سريع
# ===================================================================
if __name__ == "__main__":
    print("اختبار PSA مع num_blocks=2")
    psa = PSA(dim=50, num_heads=5, window_size=8, num_blocks=2)
    x = torch.randn(2, 48*48, 50)
    out = psa(x, 48, 48)
    print(f"الإدخال: {x.shape} → الإخراج: {out.shape}")