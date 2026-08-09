import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

def to_2tuple(x):
    return (x, x)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HAB(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*2), act_layer=nn.GELU, drop=drop)
        reduction = 4
        self.conv_du = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, H=None, W=None):
        B, C, H, W = x.shape
        shortcut = x

        # W-MSA path
        x_flat = x.flatten(2).transpose(1, 2)  # B, H*W, C
        x_flat = self.norm1(x_flat)
        x_flat = x_flat.view(B, H, W, C)

        # Pad
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        if pad_h > 0 or pad_w > 0:
            x_flat = F.pad(x_flat, (0, 0, 0, pad_w, 0, pad_h), mode='reflect')
            _, H_pad, W_pad, _ = x_flat.shape
        else:
            H_pad, W_pad = H, W

        x_windows = window_partition(x_flat, self.window_size[0])
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x_flat = window_reverse(attn_windows, self.window_size[0], H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            x_flat = x_flat[:, :H, :W, :]
        # ✅ استخدم reshape بدلاً من view
        x_flat = x_flat.reshape(B, H*W, C)

        # Residual + CAB path
        x_flat = shortcut.flatten(2).transpose(1, 2) + self.drop_path(x_flat)
        x_ca = shortcut
        y = x_ca.mean((2, 3), keepdim=True)
        y = self.conv_du(y)
        x_ca = x_ca * y
        x_ca = x_ca.flatten(2).transpose(1, 2)
        x_flat = x_flat + x_ca

        # MLP
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x

class OCAB(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*2), act_layer=nn.GELU, drop=drop)

    def forward(self, x, H=None, W=None):
        B, C, H, W = x.shape
        shortcut = x.flatten(2).transpose(1, 2)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)
        x_flat = x_flat.view(B, H, W, C)

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        if pad_h > 0 or pad_w > 0:
            x_flat = F.pad(x_flat, (0, 0, 0, pad_w, 0, pad_h), mode='reflect')
            _, H_pad, W_pad, _ = x_flat.shape
        else:
            H_pad, W_pad = H, W

        x_windows = window_partition(x_flat, self.window_size[0])
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x_flat = window_reverse(attn_windows, self.window_size[0], H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            x_flat = x_flat[:, :H, :W, :]
        x_flat = x_flat.reshape(B, H*W, C)

        x_flat = shortcut + self.drop_path(x_flat)
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x

class HAT_Group(nn.Module):
    def __init__(self, dim, num_blocks=2, window_size=8, num_heads=4, drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(HAB(dim, window_size, num_heads, drop_path=drop_path))
        self.ocab = OCAB(dim, window_size, num_heads, drop_path=drop_path)

    def forward(self, x, H=None, W=None):
        for blk in self.blocks:
            x = blk(x)
        x = self.ocab(x)
        return x

class HAT(nn.Module):
    def __init__(self, dim=90, num_heads=6, window_size=8, num_blocks=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.hat_group = HAT_Group(dim, num_blocks, window_size, num_heads)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, H=None, W=None):
        identity = x
        x = self.hat_group(x)
        x = self.conv(x) + identity
        return x