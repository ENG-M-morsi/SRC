# ===================================================================
# model/dhtcu_block_DTCF.py
# DTCF-SR: Dual-Transformer Cross-Fusion Block
#
# البنية:
#   TESA_in → [DAT ‖ ELAN] → CrossFusion → HFE → Conv1×1 → TESA_out → +x
# ===================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .custom_attention_blocks_DAT  import DAT
from .custom_attention_blocks_ELAN import ELAN


# ─────────────────────────────────────────────────────────────────
# دوال مساعدة (مستقلة عن الملفات الأخرى لتفادي أي تعارض)
# ─────────────────────────────────────────────────────────────────
def conv_layer(in_channels, out_channels, kernel_size, stride=1,
               dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding=padding, bias=True,
                     dilation=dilation, groups=groups)


def conv_block(in_nc, out_nc, kernel_size, act_type='lrelu'):
    c = nn.Conv2d(in_nc, out_nc, kernel_size,
                  padding=(kernel_size - 1) // 2, bias=True)
    a = nn.LeakyReLU(0.05, True) if act_type == 'lrelu' else None
    return nn.Sequential(c, a) if a else c


# ─────────────────────────────────────────────────────────────────
# ESA — Enhanced Spatial Attention (نفس بنية dhtcu_block_ELAN)
# ─────────────────────────────────────────────────────────────────
class ESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1   = conv(n_feats, f, kernel_size=1)
        self.conv2   = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3   = conv(f, f, kernel_size=3, padding=1)
        self.conv4   = conv(f, n_feats, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_   = self.conv1(x)
        c1    = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3    = self.relu(self.conv3(v_max))
        c3    = F.interpolate(c3, (x.size(2), x.size(3)),
                              mode='bilinear', align_corners=False)
        c4    = self.conv4(c3 + c1_)
        m     = self.sigmoid(c4)
        return x * m


class TESA(nn.Module):
    """Triple ESA — Eq.7"""
    def __init__(self, in_channels):
        super(TESA, self).__init__()
        self.esa1 = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        self.esa3 = ESA(in_channels, nn.Conv2d)

    def forward(self, x):
        return self.esa3(self.esa2(self.esa1(x)))


# ─────────────────────────────────────────────────────────────────
# CrossFusion — قلب DTCF
# ─────────────────────────────────────────────────────────────────
class CrossFusion(nn.Module):
    """
    Windowed Cross-Attention Fusion بين فرعي DAT و ELAN.

    التعقيد:
      Global Attention:   O((H·W)²)
      Windowed Attention: O(H·W · ws²)   ← ws² = 64

    مثال (Set5 image: 339×510):
      Global:   172890² × 2 heads × 4 bytes ≈ 239 GB per direction  ❌
      Windowed: 2700 windows × 64² × 2 heads × 4 bytes ≈ 45 MB      ✅
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, window_size=8):
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.ws        = window_size

        # QKV بـ Conv1×1 (أسرع من Linear مع (B,C,H,W))
        self.q_A2B = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_A2B = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_A2B = nn.Conv2d(dim, dim, 1, bias=True)
        self.proj_A2B = nn.Conv2d(dim, dim, 1)

        self.q_B2A = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_B2A = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_B2A = nn.Conv2d(dim, dim, 1, bias=True)
        self.proj_B2A = nn.Conv2d(dim, dim, 1)

        # Adaptive Gate
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )
        # Projection نهائي
        self.proj = nn.Conv2d(dim * 2, dim, 1)

        # LayerNorm (على القنوات)
        self.norm_A = nn.GroupNorm(1, dim)   # = LayerNorm over C for (B,C,H,W)
        self.norm_B = nn.GroupNorm(1, dim)

        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ─── Window helpers ───
    def _pad_to_window(self, x):
        _, _, H, W = x.shape
        ph = (self.ws - H % self.ws) % self.ws
        pw = (self.ws - W % self.ws) % self.ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, H, W

    def _partition(self, x):
        """(B,C,H,W) → (B*nH*nW, ws², C)"""
        B, C, H, W = x.shape
        ws = self.ws
        x = x.permute(0, 2, 3, 1).contiguous()          # (B,H,W,C)
        x = x.view(B, H//ws, ws, W//ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B * (H//ws) * (W//ws), ws*ws, C)

    def _reverse(self, x, H, W, B):
        """(B*nH*nW, ws², C) → (B,C,H,W)"""
        ws = self.ws
        C  = x.shape[-1]
        nH, nW = H//ws, W//ws
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x.permute(0, 3, 1, 2)                    # (B,C,H,W)

    def _windowed_cross_attn(self, q_img, k_img, v_img):
        """Cross-attention داخل النوافذ فقط."""
        B, C, H, W = q_img.shape

        # Pad الكل بنفس الأبعاد
        q_img, H_o, W_o = self._pad_to_window(q_img)
        k_img, _, _     = self._pad_to_window(k_img)
        v_img, _, _     = self._pad_to_window(v_img)
        Hp, Wp = q_img.shape[2], q_img.shape[3]

        # Partition إلى نوافذ
        q = self._partition(q_img)      # (BW, ws², C)
        k = self._partition(k_img)
        v = self._partition(v_img)

        BW, N, _ = q.shape               # N = ws² = 64
        h = self.num_heads
        d = self.head_dim

        # Multi-head reshape
        q = q.view(BW, N, h, d).transpose(1, 2)   # (BW, h, N, d)
        k = k.view(BW, N, h, d).transpose(1, 2)
        v = v.view(BW, N, h, d).transpose(1, 2)

        # Attention (N=64 → matrix 64×64 فقط!)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (BW, h, 64, 64)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(BW, N, C)

        # إعادة التجميع
        out = self._reverse(out, Hp, Wp, B)
        return out[:, :, :H_o, :W_o]

    def forward(self, F_A, F_B):
        B, C, H, W = F_A.shape

        # Normalize
        A_n = self.norm_A(F_A)
        B_n = self.norm_B(F_B)

        # QKV
        q_A = self.q_A2B(A_n)
        k_B = self.k_A2B(B_n)
        v_B = self.v_A2B(B_n)
        A2B = self._windowed_cross_attn(q_A, k_B, v_B)
        A2B = self.proj_A2B(A2B)

        q_B = self.q_B2A(B_n)
        k_A = self.k_B2A(A_n)
        v_A = self.v_B2A(A_n)
        B2A = self._windowed_cross_attn(q_B, k_A, v_A)
        B2A = self.proj_B2A(B2A)

        # Adaptive Gate
        gate  = self.gate(torch.cat([F_A, F_B], dim=1))
        fused = gate * A2B + (1.0 - gate) * B2A

        # Final projection
        out = self.proj(torch.cat([fused, F_A + F_B], dim=1))
        return out

# ─────────────────────────────────────────────────────────────────
# HighFrequencyEnhancement — تعزيز الترددات العالية
# ─────────────────────────────────────────────────────────────────
class HighFrequencyEnhancement(nn.Module):
    """
    Depthwise conv كاشف للحواف + Conv1×1 لتعزيز التفاصيل الدقيقة.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hid = max(dim // reduction, 16)
        self.dw  = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw1 = nn.Conv2d(dim, hid, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hid, dim, 1)

    def forward(self, x):
        edge = self.dw(x)
        out = self.pw2(self.act(self.pw1(edge)))
        return x + out


# ─────────────────────────────────────────────────────────────────
# DualBranchFusionBlock — الكتلة الرئيسية
# ─────────────────────────────────────────────────────────────────
class DualBranchFusionBlock(nn.Module):
    """
    بديل P_HTCB.

    forward:
        h      = TESA_in(x)
        h_dat  = DAT_branch(h)            ← (B,C,H,W)
        h_elan = ELAN_branch(h, H, W)     ← (B,C,H,W)
        h_fuse = CrossFusion(h_dat, h_elan)
        h_hfe  = HFE(h_fuse)
        h_conv = Conv1×1(h_hfe)
        out    = TESA_out(h_conv)
        return out + x
    """
    def __init__(self, in_channels,
                 # DAT params
                 num_heads_dat=4, ws_dat=8, num_blocks_dat=2,
                 # ELAN params
                 num_heads_elan=2, ws_elan=12, num_blocks_elan=3,
                 # Fusion params
                 fusion_heads=4, fusion_dropout=0.1,
                 # HFE params
                 hfe_reduction=4):
        super().__init__()

        assert in_channels % num_heads_dat  == 0
        assert in_channels % num_heads_elan == 0
        assert in_channels % fusion_heads   == 0

        # TESA in
        self.tesa_in = TESA(in_channels)

        # DAT Branch
        self.dat_branch = DAT(
            dim=in_channels,
            num_heads=num_heads_dat,
            ws=ws_dat,
            num_blocks=num_blocks_dat,
        )
        self.dat_conv = conv_block(in_channels, in_channels, 1, 'lrelu')

        # ELAN Branch
        self.elan_branch = ELAN(
            dim=in_channels,
            num_heads=num_heads_elan,
            window_size=ws_elan,
            num_blocks=num_blocks_elan,
        )
        self.elan_conv = conv_block(in_channels, in_channels, 1, 'lrelu')

        # Cross-Fusion
        self.cross_fusion = CrossFusion(
            dim=in_channels,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )

        # High-Frequency Enhancement
        self.hfe = HighFrequencyEnhancement(in_channels, reduction=hfe_reduction)

        # Conv1×1 بعد الدمج
        self.c = conv_block(in_channels, in_channels, 1, 'lrelu')

        # TESA out
        self.tesa_out = TESA(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.tesa_in(x)

        # فرع DAT (يعمل مباشرة على (B,C,H,W))
        h_dat = self.dat_branch(h)
        h_dat = self.dat_conv(h_dat)

        # فرع ELAN (يحتاج H, W)
        h_elan = self.elan_branch(h, H, W)
        h_elan = self.elan_conv(h_elan)

        # دمج متقاطع
        h_fused = self.cross_fusion(h_dat, h_elan)

        # تعزيز الترددات
        h_hfe = self.hfe(h_fused)

        # Conv1×1
        h_conv = self.c(h_hfe)

        # TESA out
        out = self.tesa_out(h_conv)

        return out + x