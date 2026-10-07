# model/dhtcun.py — DTCF-SR
# يستخدم DualBranchFusionBlock بدل التسلسل BD + BE
# ===================================================================
import torch
import torch.nn as nn
from . import dhtcu_block_DTCF as B


def make_model(args, parent=False):
    model = HUTCN(
        upscale=args.scale[0],
        nf=args.n_feats,
        num_heads_dat=getattr(args, 'num_heads_dat', 4),
        ws_dat=getattr(args, 'ws_dat', 8),
        num_blocks_dat=getattr(args, 'num_blocks_dat', 2),
        num_heads_elan=getattr(args, 'num_heads_elan', 2),
        ws_elan=getattr(args, 'ws_elan', 12),
        num_blocks_elan=getattr(args, 'num_blocks_elan', 3),
        fusion_heads=getattr(args, 'fusion_heads', 4),
        fusion_dropout=getattr(args, 'fusion_dropout', 0.1),
        hfe_reduction=getattr(args, 'hfe_reduction', 4),
    )
    return model


class HUTCN(nn.Module):
    def __init__(self, in_nc=3, nf=88, out_nc=3, upscale=4,
                 num_heads_dat=4, ws_dat=8, num_blocks_dat=2,
                 num_heads_elan=2, ws_elan=12, num_blocks_elan=3,
                 fusion_heads=4, fusion_dropout=0.1,
                 hfe_reduction=4):
        super(HUTCN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.post_unet_esa = B.ESA(nf, nn.Conv2d)
        self.post_unet_conv = B.conv_layer(nf, nf, kernel_size=1)

        # الكتلة الموحدة الجديدة
        self.B1 = B.DualBranchFusionBlock(
            in_channels=nf,
            num_heads_dat=num_heads_dat,   ws_dat=ws_dat,
            num_blocks_dat=num_blocks_dat,
            num_heads_elan=num_heads_elan, ws_elan=ws_elan,
            num_blocks_elan=num_blocks_elan,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            hfe_reduction=hfe_reduction,
        )

        self.LR_conv1 = B.conv_layer(nf, nf, kernel_size=1)
        self.LR_conv2 = B.conv_layer(nf, nf, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.recon_conv1 = B.conv_layer(nf, nf, kernel_size=3)
        self.recon_conv2 = B.conv_layer(nf, out_nc * (upscale ** 2), kernel_size=3)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_lr = self.post_unet_conv(self.post_unet_esa(out_B1)) + out_fea
        out_r1 = self.recon_conv1(out_lr)
        out_r2 = self.recon_conv2(out_r1)
        return self.pixel_shuffle(out_r2)

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx