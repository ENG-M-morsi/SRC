# ===================================================================
# loss/edge_loss.py — Edge-Aware Loss للصور
#
# الاستخدام في args.loss:
#   --loss "1*L1+0.05*VGG22+0.05*FFT+0.02*EDGE"
#
# ملاحظة: لتشغيله، أضف 6 أسطر فقط في loss/__init__.py
# (موضّحة في نهاية الملف) — بدون أي تعديل على vgg/adversarial/discriminator.
# ===================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """
    Sobel-based Edge Loss.

    الفكرة:
    - حساب خريطة الحواف (edge map) باستخدام Sobel filter للصورتين
    - مقارنة الخريطتين بـ L1
    - يعاقب النموذج على فقدان الحواف الحادة (sharpeness)

    لماذا Sobel وليس Laplacian؟
    - Sobel يعطي حساسية اتجاهية (اتجاه x و y) → أفضل للتفاصيل
    - Laplacian أكثر حساسية للضجيج
    """
    def __init__(self, rgb_range=1.0):
        super(EdgeLoss, self).__init__()
        self.rgb_range = rgb_range

        # Sobel kernels
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _edge_map(self, x):
        # grayscale
        gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # normalize to [-1,1] if rgb_range=255
        if self.rgb_range > 1.5:
            gray = gray / self.rgb_range
            gray = (gray - 0.5) * 2.0

        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, sr, hr):
        sr_edge = self._edge_map(sr)
        hr_edge = self._edge_map(hr)
        return F.l1_loss(sr_edge, hr_edge)