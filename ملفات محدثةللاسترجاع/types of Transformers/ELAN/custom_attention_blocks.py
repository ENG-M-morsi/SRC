import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import warnings

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias
    )

class ShiftConv(nn.Module):
    """
    Shift Convolution module as described in ELAN paper.
    It performs shift operations (up, down, left, right) and then a 1x1 convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(ShiftConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 1x1 convolution for fusion
        self.conv = nn.Conv2d(in_channels * 5, out_channels, 1, padding=0, bias=True)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # Original feature map
        orig = x

        # Shift operations: up, down, left, right
        up = torch.roll(x, shifts=-1, dims=2)   # Shift up
        down = torch.roll(x, shifts=1, dims=2)  # Shift down
        left = torch.roll(x, shifts=-1, dims=3)  # Shift left
        right = torch.roll(x, shifts=1, dims=3)  # Shift right

        # Concatenate along channel dimension
        shifted = torch.cat([orig, up, down, left, right], dim=1)  # (B, C*5, H, W)

        # Apply 1x1 convolution to fuse information
        out = self.conv(shifted)  # (B, out_channels, H, W)

        return out

class GMSA(nn.Module):
    """
    Group-wise Multi-scale Self-Attention (GMSA) module.
    It divides features into groups and applies window-based attention with different window sizes.
    """
    def __init__(self, dim, num_heads=8, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Multi-scale window sizes
        self.window_sizes = [window_size // 2, window_size, window_size * 2]
        self.window_sizes = [ws for ws in self.window_sizes if ws > 0]

        # Projections for Q, K, V for each window size
        self.qkv_list = nn.ModuleList()
        self.attn_drop_list = nn.ModuleList()
        self.proj_list = nn.ModuleList()

        for _ in self.window_sizes:
            self.qkv_list.append(nn.Linear(dim, dim * 3, bias=qkv_bias))
            self.attn_drop_list.append(nn.Dropout(attn_drop))
            self.proj_list.append(nn.Linear(dim, dim))

        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        self.relative_position_bias_tables = nn.ParameterList()
        for ws in self.window_sizes:
            # Relative position bias table for each window size
            bias_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads))
            trunc_normal_(bias_table, std=.02)
            self.relative_position_bias_tables.append(bias_table)

        # Pre-compute indices for relative position bias
        self.relative_position_indices = []
        for ws in self.window_sizes:
            coords_h = torch.arange(ws)
            coords_w = torch.arange(ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += ws - 1
            relative_coords[:, :, 1] += ws - 1
            relative_coords[:, :, 0] *= 2 * ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer(f'relative_position_index_{ws}', relative_position_index)
            self.relative_position_indices.append(relative_position_index)

    def forward(self, x, H, W):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        out = 0
        for idx, ws in enumerate(self.window_sizes):
            # Pad to ensure divisibility
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                B, C, H_pad, W_pad = x_padded.shape
            else:
                x_padded = x
                H_pad, W_pad = H, W

            # Window partition
            x_windows = rearrange(x_padded, 'b c (h w1) (w w2) -> (b h w) (w1 w2) c', w1=ws, w2=ws)
            B_, N, C = x_windows.shape

            # QKV projection
            qkv = self.qkv_list[idx](x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            # Relative position bias
            relative_position_bias = self.relative_position_bias_tables[idx][self.relative_position_indices[idx].view(-1)].view(
                ws * ws, ws * ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop_list[idx](attn)

            x_windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x_windows = self.proj_list[idx](x_windows)

            # Window reverse
            x_out = rearrange(x_windows, '(b h w) (w1 w2) c -> b c (h w1) (w w2)', b=B, h=H_pad // ws, w=W_pad // ws, w1=ws, w2=ws)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                x_out = x_out[:, :, :H, :W]

            out += x_out

        out = self.proj_drop(self.norm(out.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W))
        return out

class ELAB(nn.Module):
    """
    Efficient Long-Range Attention Block (ELAB) as described in ELAN paper.
    It cascades two ShiftConv layers with a GMSA module.
    """
    def __init__(self, dim, num_heads=8, window_size=8, shift_conv=True):
        super(ELAB, self).__init__()
        self.shift_conv1 = ShiftConv(dim, dim) if shift_conv else default_conv(dim, dim, 3)
        self.shift_conv2 = ShiftConv(dim, dim) if shift_conv else default_conv(dim, dim, 3)
        self.gmsa = GMSA(dim, num_heads, window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.)
        )
        self.drop_path = nn.Identity()

    def forward(self, x, H, W):
        B, C, H, W = x.shape
        shortcut = x

        # First shift convolution
        x = self.shift_conv1(x)
        x = F.gelu(x)

        # GMSA with residual connection
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)
        x_attn = self.gmsa(x, H, W)
        x_attn_flat = x_attn.flatten(2).transpose(1, 2)
        x_attn_flat = shortcut.flatten(2).transpose(1, 2) + self.drop_path(x_attn_flat)
        x_attn = x_attn_flat.transpose(1, 2).view(B, C, H, W)

        # Second shift convolution
        x = self.shift_conv2(x_attn)
        x = F.gelu(x)

        # MLP
        x_flat = x.flatten(2).transpose(1, 2)
        x_mlp = self.norm2(x_flat)
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_flat + self.drop_path(x_mlp)
        x_out = x_mlp.transpose(1, 2).view(B, C, H, W)

        return x_out

class ELAN(nn.Module):
    """
    Efficient Long-Range Attention Network (ELAN) for image super-resolution.
    """
    def __init__(self, dim=60, num_heads=6, window_size=8, num_blocks=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_blocks = num_blocks

        # Patch embedding
        self.patch_embed = default_conv(dim, dim, 3)

        # ELAB blocks
        self.blocks = nn.ModuleList([
            ELAB(dim, num_heads, window_size, shift_conv=True)
            for _ in range(num_blocks)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        # Input x shape: (B, C, H, W)
        x = self.patch_embed(x)  # (B, dim, H, W)

        for blk in self.blocks:
            x = blk(x, H, W)

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x = self.norm(x_flat).transpose(1, 2).view(B, C, H, W)
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
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