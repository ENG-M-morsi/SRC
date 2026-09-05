# ===================================================================
# dhtcu_block.py — النسخة المصححة (مع معاملات منفصلة)
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .custom_attention_blocks import DAT
from .custom_attention_blocks_ELAN import ELAN

# ... (دوال conv_layer, norm, pad, ... تبقى كما هي) ...

# -------------------------------------------------------------------
# TCN — Transformer CNN Block (يستخدم DAT)
# -------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, in_channels, num_heads=3, ws=8, num_blocks=1):
        super(TCN, self).__init__()
        assert in_channels % num_heads == 0, \
            f"in_channels={in_channels} يجب أن يقبل القسمة على num_heads={num_heads}"
        self.dat = DAT(dim=in_channels, num_heads=num_heads, ws=ws, num_blocks=num_blocks)
        self.conv3 = conv_layer(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        return self.conv3(self.dat(x))


# -------------------------------------------------------------------
# TECN — Transformer CNN Block (يستخدم ELAN)
# -------------------------------------------------------------------
class TECN(nn.Module):
    def __init__(self, in_channels, num_heads=2, window_size=12, num_blocks=3):
        super(TECN, self).__init__()
        assert in_channels % num_heads == 0, \
            f"in_channels={in_channels} يجب أن يقبل القسمة على num_heads={num_heads}"
        self.elan = ELAN(dim=in_channels, num_heads=num_heads, window_size=window_size, num_blocks=num_blocks)
        self.conv3 = conv_layer(in_channels, in_channels, kernel_size=3)

    def forward(self, x, H, W):
        return self.conv3(self.elan(x, H, W))


# -------------------------------------------------------------------
# P_HTCB — Parallel Hybrid Transformer CNN Block (مع معاملات منفصلة)
# -------------------------------------------------------------------
class P_HTCB(nn.Module):
    def __init__(self, in_channels,
                 # معاملات DAT
                 num_heads_dat=3, ws_dat=8, num_blocks_dat=1,
                 # معاملات ELAN
                 num_heads_elan=2, ws_elan=12, num_blocks_elan=3):
        super(P_HTCB, self).__init__()

        self.tesa_in = TESA(in_channels)

        # TCN1 يستخدم DAT بمعاملاته الخاصة
        self.tcn1 = TCN(in_channels, num_heads=num_heads_dat, ws=ws_dat, num_blocks=num_blocks_dat)

        # TECN2 يستخدم ELAN بمعاملاته الخاصة
        self.tecn2 = TECN(in_channels, num_heads=num_heads_elan, window_size=ws_elan, num_blocks=num_blocks_elan)

        self.c = conv_block(in_channels, in_channels, kernel_size=1, act_type='lrelu')
        self.tesa_out = TESA(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h_tesa = self.tesa_in(x)

        h_tcn1 = self.tcn1(h_tesa)                 # DAT لا يحتاج H,W
        h_tecn2 = self.tecn2(h_tesa, H, W)         # ELAN يحتاج H,W

        h_add = h_tcn1 + h_tecn2
        h_conv = self.c(h_add)
        out = self.tesa_out(h_conv)
        return out + x