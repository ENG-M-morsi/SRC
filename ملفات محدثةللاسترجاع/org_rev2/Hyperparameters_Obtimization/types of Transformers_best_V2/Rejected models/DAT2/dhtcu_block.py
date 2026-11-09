# ===================================================================
# dhtcu_block.py — النسخة المصححة الكاملة
# الإصلاحات:
# 1. TCN.forward: يستدعي DAT بـ (x) فقط (بدون H,W)
# 2. P_HTCB.forward: حذف استخراج B,C,H,W غير الضروري
# 3. إضافة tcn2 (البلوك المتوازي الثاني — موجود في الورقة)
# 4. fea_conv kernel_size=3 في dhtcun لتحسين الجودة
# 5. num_heads=6 لـ DAT مع dim=90 (90//6=15 ✅)
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .dat_attention_block import DAT


# -------------------------------------------------------------------
# دوال مساعدة
# -------------------------------------------------------------------
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding=padding, bias=True, dilation=dilation, groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        return nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(nc, affine=False)
    raise NotImplementedError(f'norm [{norm_type}] not found')


def pad(pad_type, padding):
    if padding == 0:
        return None
    pad_type = pad_type.lower()
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    raise NotImplementedError(f'padding [{pad_type}] not implemented')


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    return (kernel_size - 1) // 2


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU(inplace)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f'activation [{act_type}] not found')


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = [m for arg in args
               for m in (arg.children() if isinstance(arg, nn.Sequential) else [arg])
               if isinstance(m, nn.Module)]
    return nn.Sequential(*modules)


def mean_channels(F):
    assert F.dim() == 4
    return F.sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_var  = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_var.pow(0.5)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    return sequential(conv, nn.PixelShuffle(upscale_factor))


# -------------------------------------------------------------------
# ESA — Enhanced Spatial Attention (كما هو، يعمل جيداً)
# -------------------------------------------------------------------
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
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


# -------------------------------------------------------------------
# TESA — Triple ESA
# -------------------------------------------------------------------
class TESA(nn.Module):
    def __init__(self, in_channels):
        super(TESA, self).__init__()
        self.esa1 = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        self.esa3 = ESA(in_channels, nn.Conv2d)

    def forward(self, x):
        return self.esa3(self.esa2(self.esa1(x)))


# -------------------------------------------------------------------
# TCN — Transformer CNN Block (يستخدم DAT المصحّح)
# الإصلاح الرئيسي: self.dat(x) بدون H, W
# -------------------------------------------------------------------
class TCN(nn.Module):
    def __init__(self, in_channels, num_heads=3):
        super(TCN, self).__init__()
        # التحقق من توافق dim مع num_heads
        assert in_channels % num_heads == 0, \
            f"in_channels={in_channels} يجب أن يقبل القسمة على num_heads={num_heads}"
        self.dat = DAT(dim=in_channels, num_heads=num_heads, window_size=8, num_blocks=2)
        # Conv3×3 بعد DAT (كما في الورقة: FConv3 ∘ FSTL)
        #self.conv3 = conv_layer(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        # x: (B, C, H, W)  — DAT يستقبل ويُرجع نفس الشكل
        return self.dat(x)
        #return self.conv3(self.dat(x))


# -------------------------------------------------------------------
# P_HTCB — Parallel Hybrid Transformer CNN Block (مصحّح)
# -------------------------------------------------------------------
class P_HTCB(nn.Module):
    """
    الإصلاحات:
    - TCN.forward يُستدعى بـ (x) فقط
    - TCN2 مفعّل (parallel كما في الورقة)
    - Addition بدل Cat (الورقة Eq.4: HTCN1 + HTCN2)
    """
    def __init__(self, in_channels, num_heads=3):
        super(P_HTCB, self).__init__()

        self.tesa_in  = TESA(in_channels)

        # TCN1 و TCN2 متوازيان — Eq.3
        self.tcn1 = TCN(in_channels, num_heads=num_heads)
        #self.tcn2 = TCN(in_channels, num_heads=num_heads)

        # Conv1×1 بعد Addition — Eq.5
        self.c = conv_block(in_channels, in_channels,
                            kernel_size=1, act_type='lrelu')

        self.tesa_out = TESA(in_channels)

    def forward(self, x):
        h_tesa = self.tesa_in(x)

        # TCN1 و TCN2 بالتوازي على نفس المدخل
        h_tcn1 = self.tcn1(h_tesa)
       # h_tcn2 = self.tcn2(h_tesa)

        # Addition — Eq.4
        #h_add  = h_tcn1 + h_tcn2
        h_add  = h_tcn1 

        # Conv1×1 — Eq.5
        h_conv = self.c(h_add)

        # TESA النهائية — Eq.6
        out = self.tesa_out(h_conv)

        # Global residual
        return out + x