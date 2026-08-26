import torch.nn as nn
from collections import OrderedDict  # ✅ أضف هذا السطر
import torch
import torch.nn.functional as F
#from . import SGBlock,FNet,Spartial_Attention,SwinT
from .custom_attention_blocks import WaveletAttention

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
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


"""class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m"""

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1    = conv(n_feats, f, kernel_size=1)          # Conv1×1 أول
        self.conv2    = conv(f, f, kernel_size=3, stride=2, padding=0)  # Conv3×3 + stride
        # MaxPool في forward
        self.conv3    = conv(f, f, kernel_size=3, padding=1)     # Conv3×3 بعد Pool
        # Upsample في forward
        self.conv_f   = conv(f, f, kernel_size=1)                # Conv1×1 المسار المواز
        self.conv4    = conv(f, n_feats, kernel_size=3, padding=1)  # Conv3×3 قبل Sigmoid ✅
        self.sigmoid  = nn.Sigmoid()
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_  = self.conv1(x)
        c1   = self.conv2(c1_)                                   # stride=2 تقليل الحجم
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)        # Pooling
        c3   = self.relu(self.conv3(v_max))                      # Conv3×3 بعد Pool
        c3   = F.interpolate(c3, (x.size(2), x.size(3)),
                             mode='bilinear', align_corners=False)  # Upsample
        #cf   = self.conv_f(c1_)                                  # HE2
        c4   = self.conv4(c3 + c1_)                               # Conv3×3(HE1+HE2)
        m    = self.sigmoid(c4)
        return x * m                                             # ✅ x وليس HE1


"""class P_HTCB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(P_HTCB, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.esa = ESA(in_channels, nn.Conv2d)
        #self.esa2 = ESA(in_channels, nn.Conv2d)
        # self.sparatt = Spartial_Attention.Spartial_Attention()
        self.swinT = SwinT.SwinT()

    def forward(self, input):
        input0 = self.esa(input)
        input1 = self.swinT(input0)
        input2 = self.esa(input)
        input3 = self.swinT(input2)
        input_esa = self.c1_r(input1) + input3
        out_fused = self.esa(self.c1_r(input_esa))
        return out_fused"""
        
"""class P_HTCB(nn.Module):
    def __init__(self, in_channels):
        super(P_HTCB, self).__init__()
        self.c1_r = conv_layer(in_channels, in_channels, 3)
        self.c1_r2 = conv_layer(in_channels, in_channels, 3)
        # TESA الأولى - 3 instances مستقلة ✅
        self.esa1_1 = ESA(in_channels, nn.Conv2d)
        self.esa1_2 = ESA(in_channels, nn.Conv2d)
        self.esa1_3 = ESA(in_channels, nn.Conv2d)

        # TESA الأخيرة - 3 instances مستقلة ✅
        self.esa2_1 = ESA(in_channels, nn.Conv2d)
        self.esa2_2 = ESA(in_channels, nn.Conv2d)
        self.esa2_3 = ESA(in_channels, nn.Conv2d)

        # SwinT يستقبل in_channels ✅
        #self.swinT = SwinT.SwinT()
        #self.swinT1 = SwinT.SwinT(n_feats=in_channels)
        #self.swinT2 = SwinT.SwinT(n_feats=in_channels)
        self.swinT = SwinT.SwinT(n_feats=in_channels)

    def forward(self, input):
        # TESA الأولى
        x = self.esa1_1(input)
        x = self.esa1_2(x)
        x = self.esa1_3(x)

        # TCN1 و TCN2 parallel
        swint_out = self.swinT(x)          # ✅ مرة واحدة
        tcn1 = self.c1_r(swint_out)        # conv1
        tcn2 = self.c1_r2(swint_out)       # conv2 مختلفة        
        
        #tcn1 = self.c1_r(self.swinT1(x))
        #tcn2 = self.c1_r(self.swinT2(x))

        # CONV + residual connection
        fused = self.c1_r(tcn1 + tcn2) + input

        # TESA الأخيرة
        out = self.esa2_1(fused)
        out = self.esa2_2(out)
        out = self.esa2_3(out)

        return out"""
class TESA(nn.Module):
    """Triple Enhanced Spatial Attention — Eq.7"""
    def __init__(self, in_channels):
        super(TESA, self).__init__()
        self.esa1 = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        self.esa3 = ESA(in_channels, nn.Conv2d)

    def forward(self, x):
        # HTESA = FESA(FESA(FESA(Hi/p)))
        return self.esa3(self.esa2(self.esa1(x)))


class TCN(nn.Module):
    def __init__(self, in_channels, num_heads=8, window_size=12, num_blocks=3):
        super(TCN, self).__init__()
        self.wavelet_attn = WaveletAttention(
            dim=in_channels,
            num_heads=num_heads,
            window_size=window_size,
            num_blocks=num_blocks
        )
        self.conv3 = conv_layer(in_channels, in_channels, kernel_size=3)

    def forward(self, x, H, W):
        #return self.wavelet_attn(x, H, W)
        return self.conv3(self.wavelet_attn(x, H, W))


class P_HTCB(nn.Module):
    def __init__(self, in_channels, num_heads=8, window_size=12, num_blocks=1):
        super(P_HTCB, self).__init__()
        self.tesa_in = TESA(in_channels)
        self.tcn1 = TCN(in_channels, num_heads, window_size, num_blocks)
        self.tcn2 = TCN(in_channels, num_heads, window_size, num_blocks)
        self.c = conv_block(in_channels, in_channels, kernel_size=1, act_type='lrelu')
        self.tesa_out = TESA(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h_tesa = self.tesa_in(x)
        h_tcn1 = self.tcn1(h_tesa, H, W)
        h_tcn2 = self.tcn2(h_tesa, H, W)
        h_add = h_tcn1 + h_tcn2
        h_conv = self.c(h_add)
        out = self.tesa_out(h_conv)
        return out + x

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)