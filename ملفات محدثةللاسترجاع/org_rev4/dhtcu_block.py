import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F
from . import SwinT


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding=padding, bias=True, dilation=dilation, groups=groups)


def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    """Bottleneck-style conv block for efficiency"""
    return nn.Sequential(
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1), bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# ============================================================
# Enhanced Spatial Attention (ESA) - As per DHTCUN paper
# ============================================================
class ESA(nn.Module):
    """
    Enhanced Spatial Attention module as described in DHTCUN paper.
    Uses a series of convolutions with downsampling and upsampling
    to produce a spatial attention map.
    """
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv4 = conv(f, n_feats, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.relu(self.conv3(v_max))
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m


# ============================================================
# Triple Enhanced Spatial Attention (TESA) - Eq.2 in DHTCUN
# ============================================================
class TESA(nn.Module):
    """
    Triple Enhanced Spatial Attention - Eq.2 in DHTCUN paper.
    Applies ESA three times sequentially: HTESA = ESA(ESA(ESA(H)))
    """
    def __init__(self, in_channels):
        super(TESA, self).__init__()
        self.esa1 = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        self.esa3 = ESA(in_channels, nn.Conv2d)

    def forward(self, x):
        return self.esa3(self.esa2(self.esa1(x)))


# ============================================================
# Transformer CNN Block (TCN) - Eq.3 in DHTCUN
# ============================================================
class TCN(nn.Module):
    """
    Transformer CNN Block - Eq.3 in DHTCUN paper.
    HTCN = FSTL(FConv3(HTESA))
    Order: Conv3x3 first, then Swin Transformer Layer (STL)
    """
    def __init__(self, in_channels):
        super(TCN, self).__init__()
        self.conv3 = conv_layer(in_channels, in_channels, kernel_size=3)
        self.swinT = SwinT.SwinT(n_feats=in_channels)

    def forward(self, x):
        return self.swinT(self.conv3(x))


# ============================================================
# Parallel Hybrid Transformer CNN Block (P_HTCB) - DHTCUN paper
# ============================================================
class P_HTCB(nn.Module):
    """
    Parallel Hybrid Transformer CNN Block - DHTCUN paper.

    Architecture:
    - TESA input (Eq.2): Triple ESA on input
    - Two TCN branches in parallel (Eq.3): Each = SwinT(Conv3(x))
    - Concatenation (Eq.4): HCat = cat([HTCN1, HTCN2])
    - Conv1x1 fusion (Eq.5): HConv = Conv1x1(HCat)  [nf*2 -> nf]
    - TESA output (Eq.6): Final triple ESA
    - Residual connection: + input
    """
    def __init__(self, in_channels):
        super(P_HTCB, self).__init__()

        # TESA input - Eq.2
        self.tesa_in = TESA(in_channels)

        # TCN1 and TCN2 in parallel - Eq.3
        self.tcn1 = TCN(in_channels)
        self.tcn2 = TCN(in_channels)

        # Conv1x1 after Concatenation - Eq.4+5
        # Input: nf*2 (from cat), Output: nf
        self.c = conv_block(in_channels * 2, in_channels,
                            kernel_size=1, act_type='lrelu')

        # TESA output - Eq.6
        self.tesa_out = TESA(in_channels)

    def forward(self, x):
        # Eq.2: Input TESA
        h_tesa = self.tesa_in(x)

        # Eq.3: Two TCN branches in parallel
        h_tcn1 = self.tcn1(h_tesa)
        h_tcn2 = self.tcn2(h_tesa)

        # Eq.4: Concatenation along channel dimension
        h_cat = torch.cat([h_tcn1, h_tcn2], dim=1)

        # Eq.5: Conv1x1 to fuse concatenated features
        h_conv = self.c(h_cat)

        # Eq.6: Output TESA
        out = self.tesa_out(h_conv)

        # Residual connection as per Figure 3 in paper
        return out + x


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)