# ===================================================================
# eval_protocols.py
# يقيّم نفس الأوزان المدرَّبة بعدة بروتوكولات PSNR لعزل سبب الفرق
# بين قيم Optuna وقيم التدريب الحقيقي.
#
# ضعه في مجلد src وشغّله:  python eval_protocols.py
# يكتشف تلقائياً هل model/dhtcun.py الحالي DAT أم HAT ويختار الـ checkpoint المناسب.
# ===================================================================
import os
import sys
import copy
import math
import inspect
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data
from option import args as base_args
from model.dhtcun import HUTCN

# ───────────── الإعدادات (عدّل المسارات إن لزم) ─────────────
DIR_DATA = r'D:\Mohamed Morsi\DATA'
SCALE = 4
CKPT = {
    'DAT': r'D:\Mohamed Morsi\src\experiment\DAT_best-CNNx4\model\model_best.pt',
    'HAT': r'D:\Mohamed Morsi\src\experiment\HAT_best-CNNx4\model\model_best.pt',
}
# أفضل معاملات كل موديل + حجم crop التقييم الذي استُخدم في Optuna
CFG = {
    'DAT': dict(num_heads=2, ws=8, num_blocks=4,
                num_cnn_stages=4, act_fn='prelu', crop=128),
    'HAT': dict(num_heads=4, window_size=12, num_blocks=4,
                num_cnn_stages=6, act_fn='lrelu', crop=120),
}
# ───────────────────────────────────────────────────────────


def quantize(img):
    return img.mul(255).clamp(0, 255).round().div(255)


def psnr(diff, shave, use_y):
    """diff = sr - hr بمدى [0,1] وشكل (1,3,H,W)"""
    if use_y:
        coef = diff.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
        diff = (diff * coef).sum(dim=1)
    if shave > 0:
        diff = diff[..., shave:-shave, shave:-shave]
    return -10 * math.log10(diff.pow(2).mean().item())


def center_crop_pair(lr, hr, scale, size):
    _, _, h, w = lr.shape
    ch, cw = min(size, h), min(size, w)
    top, left = (h - ch) // 2, (w - cw) // 2
    lr_c = lr[:, :, top:top + ch, left:left + cw]
    hr_c = hr[:, :, top * scale:(top + ch) * scale,
              left * scale:(left + cw) * scale]
    return lr_c, hr_c


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    arch = 'HAT' if 'window_size' in inspect.signature(HUTCN.__init__).parameters else 'DAT'
    cfg = dict(CFG[arch])
    crop = cfg.pop('crop')
    print(f'▶ الموديل المكتشف في model/dhtcun.py: {arch}')
    print(f'▶ الـ checkpoint: {CKPT[arch]}')

    model = HUTCN(in_nc=3, nf=128, num_modules=1, out_nc=3,
                  upscale=SCALE, **cfg)
    state = torch.load(CKPT[arch], map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    args = copy.deepcopy(base_args)
    args.data_train = ['DIV2K']
    args.data_test = ['DIV2K']
    args.data_range = '1-800/896-900'
    args.scale = [SCALE]
    args.dir_data = DIR_DATA
    args.rgb_range = 1
    args.test_only = True        # لا يبني train loader
    args.n_threads = 0
    args.batch_size = 1
    val = data.Data(args).loader_test[0]
    val.dataset.set_scale(0)

    names = ['P1 Trainer: كاملة|Y|shave4|quantize',
             'P2 كاملة|RGB|shave4|quantize',
             'P3 كاملة|RGB|بلا shave|بلا quantize',
             f'P4 crop{crop}|RGB|بلا shave (fp32)',
             f'P5 crop{crop}|RGB|بلا shave (fp16)= Optuna',
             f'P6 crop{crop}|Y|shave4|quantize']
    rows = []

    with torch.no_grad():
        for lr, hr, fname in val:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)
            h = min(sr.shape[-2], hr.shape[-2])
            w = min(sr.shape[-1], hr.shape[-1])
            sr_f, hr_f = sr[..., :h, :w], hr[..., :h, :w]

            p1 = psnr(quantize(sr_f) - hr_f, SCALE, True)
            p2 = psnr(quantize(sr_f) - hr_f, SCALE, False)
            p3 = psnr(sr_f - hr_f, 0, False)

            lr_c, hr_c = center_crop_pair(lr, hr, SCALE, crop)
            sr_c = model(lr_c)
            p4 = psnr(sr_c - hr_c, 0, False)
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    sr_c16 = model(lr_c)
                p5 = psnr(sr_c16.float() - hr_c, 0, False)
            else:
                p5 = float('nan')
            p6 = psnr(quantize(sr_c) - hr_c, SCALE, True)

            rows.append((fname[0], [p1, p2, p3, p4, p5, p6]))

    print('\n' + '=' * 100)
    print('PSNR لكل صورة (dB):')
    print('=' * 100)
    print(f"{'image':<10}" + ''.join(f'  P{i + 1:<6}' for i in range(6)))
    for n, v in rows:
        print(f'{n:<10}' + ''.join(f'  {x:<7.3f}' for x in v))
    mean = [sum(v[i] for _, v in rows) / len(rows) for i in range(6)]
    print(f"{'AVERAGE':<10}" + ''.join(f'  {x:<7.3f}' for x in mean))
    print('\nالمفتاح:')
    for i, n in enumerate(names):
        print(f'  {n}')

    print('\n' + '=' * 100)
    print('تفكيك الفرق (متوسط 5 صور):')
    print('=' * 100)
    print(f'  تأثير قناة Y مقابل RGB     (P1-P2): {mean[0] - mean[1]:+.3f} dB')
    print(f'  تأثير shave+quantize       (P2-P3): {mean[1] - mean[2]:+.3f} dB')
    print(f'  تأثير crop الصغير مقابل الكاملة (P3-P4): {mean[2] - mean[3]:+.3f} dB')
    print(f'  تأثير AMP fp16             (P5-P4): {mean[4] - mean[3]:+.3f} dB')
    print(f'  الفرق الكلي Trainer مقابل Optuna (P1-P5): {mean[0] - mean[4]:+.3f} dB')
    print('\nتحقق: P1 يجب أن يطابق أفضل PSNR في log.txt '
          '(DAT ≈ 32.79 ، HAT ≈ 32.60).')


if __name__ == '__main__':
    main()
