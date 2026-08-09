import torch
import torch.nn as nn
from . import dhtcu_block as B
def make_model(args, parent=False):
    #model = HUTCN()
    #model = HUTCN(upscale = args.scale[0])
    model = HUTCN(upscale=args.scale[0],nf=args.n_feats)
    return model


"""class Cascade(nn.Module):
    def __init__(self, ):
        super(Cascade, self).__init__()
        self.conv1 = B.conv_layer(50, 50, kernel_size=1)
        self.conv3 = B.conv_layer(50, 50, kernel_size=3)
        self.conv5 = B.conv_layer(50, 50, kernel_size=5)
        self.c = B.conv_block(50 * 4, 50, kernel_size=1, act_type='lrelu')

    def forward(self, x):
        conv5 = self.conv5(x)
        extra = x+conv5
        conv3 = self.conv3(extra)
        extra = x + conv3
        conv1 = self.conv1(extra)
        cat = torch.cat([conv5, conv3, conv1, x], dim=1)
        input = self.c(cat)
        return input"""


class HUTCN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(HUTCN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)

        # ─── Pre‑Upsampling ───
        self.pre_shuffle = B.conv_layer(nf, nf * (upscale ** 2), kernel_size=3)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # ─── P_HTCB (الآن يقبل upscale) ───
        self.B1 = B.P_HTCB(in_channels=nf, upscale=upscale)

        # ─── ESA + Conv بعد UNet ───
        self.post_unet_esa  = B.ESA(nf, nn.Conv2d)
        self.post_unet_conv = B.conv_layer(nf, nf, kernel_size=1)

        # ─── طبقة إعادة البناء النهائية (بدون PixelShuffle) ───
        self.recon_conv = B.conv_layer(nf, out_nc, kernel_size=3)

        self.scale_idx = 0

    def forward(self, input):
        # 1. استخلاص المميزات
        out_fea = self.fea_conv(input)                     # (B, nf, H, W)

        # 2. رفع الدقة (Pre‑Upsampling)
        out_fea_up = self.pixel_shuffle(self.pre_shuffle(out_fea))  # (B, nf, H*scale, W*scale)

        # 3. تمرير على P_HTCB
        out_B1 = self.B1(out_fea_up)                       # (B, nf, H*scale, W*scale)

        # 4. ESA + Conv + Skip Connection (باستخدام out_fea_up)
        out_lr = self.post_unet_conv(self.post_unet_esa(out_B1)) + out_fea_up

        # 5. توليد الصورة النهائية
        output = self.recon_conv(out_lr)                   # (B, 3, H*scale, W*scale)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

