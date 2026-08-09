import torch
import torch.nn as nn
from . import dhtcu_block as B


def make_model(args, parent=False):
    """Factory function to create DHTCUN model."""
    model = HUTCN(upscale=args.scale[0], nf=args.n_feats)
    return model


class HUTCN(nn.Module):
    """
    DHTCUN: Deep Hybrid Transformer CNN U Network for Single-Image Super-Resolution
    
    Architecture:
    1. Shallow feature extraction: Conv1x1 (3 -> nf)
    2. U-Net structure with 5 P_HTCB blocks:
       - Encoder: B1 -> B2 -> B3
       - Decoder: B4 (with skip from B2) -> B5 (with skip from B1)
    3. Post U-Net: ESA + Conv1x1 + residual connection
    4. Reconstruction: Conv3x3 -> Conv3x3 -> PixelShuffle
    """
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=3):
        super(HUTCN, self).__init__()

        # ---- Shallow Feature Extraction ----
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)

        # ---- Post U-Net: ESA + Conv1x1 (as per paper) ----
        self.post_unet_esa = B.ESA(nf, nn.Conv2d)
        self.post_unet_conv = B.conv_layer(nf, nf, kernel_size=1)

        # ---- P_HTCB Blocks: U-Net Structure ----
        self.B1 = B.P_HTCB(in_channels=nf)
        self.B2 = B.P_HTCB(in_channels=nf)
        self.B3 = B.P_HTCB(in_channels=nf)
        self.B4 = B.P_HTCB(in_channels=nf)
        self.B5 = B.P_HTCB(in_channels=nf)

        # Skip connection convolutions (1x1 to match channels if needed)
        self.LR_conv1 = B.conv_layer(nf, nf, kernel_size=1)
        self.LR_conv2 = B.conv_layer(nf, nf, kernel_size=1)

        # ---- Reconstruction Module ----
        # Paper: "Pixel Shuffle followed by two layers of the 3x3 convolution"
        # Order: Conv3x3 -> Conv3x3(channels upscale^2) -> PixelShuffle
        self.recon_conv1 = B.conv_layer(nf, nf, kernel_size=3)
        self.recon_conv2 = B.conv_layer(nf, out_nc * (upscale ** 2), kernel_size=3)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # ---- Initialize weights ----
        self._initialize_weights()

        self.scale_idx = 0

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # ---- Shallow features ----
        out_fea = self.fea_conv(input)

        # ---- U-Net Encoder ----
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)

        # ---- U-Net Decoder with Skip Connections ----
        # Skip connection from B2 to B4
        out_B4 = self.B4(out_B3)
        out_B4 = self.LR_conv1(out_B4) + out_B2

        # Skip connection from B1 to B5
        out_B5 = self.B5(out_B4)
        out_B5 = self.LR_conv2(out_B5) + out_B1

        # ---- Post U-Net: ESA + Conv1x1 + Residual ----
        out_lr = self.post_unet_conv(self.post_unet_esa(out_B5)) + out_fea

        # ---- Reconstruction: Conv3x3 -> Conv3x3 -> PixelShuffle ----
        out_r1 = self.recon_conv1(out_lr)
        out_r2 = self.recon_conv2(out_r1)
        output = self.pixel_shuffle(out_r2)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx