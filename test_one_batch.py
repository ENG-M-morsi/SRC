# test_fft_loss.py
import sys
sys.path.append('.')
import torch
from loss import FFTLoss

device = 'cuda'

# إنشاء FFTLoss
l = FFTLoss().to(device)
print('FFTLoss built ✅')

# 1) صورة HR اصطناعية
hr = torch.randn(2, 3, 96, 96).cuda() * 0.1 + 0.5

# 2) SR مطابق تماماً (متوقع: FFT = 0)
v_identical = l(hr, hr)
print(f'FFT loss (identical): {v_identical.item():.8f}')

# 3) SR قريب (متوقع: FFT > 0 صغير)
sr_close = hr + 0.01 * torch.randn_like(hr)
v_close = l(sr_close, hr)
print(f'FFT loss (close):     {v_close.item():.8f}')

# 4) SR بعيد (متوقع: FFT أكبر)
sr_far = hr + 0.1 * torch.randn_like(hr)
v_far = l(sr_far, hr)
print(f'FFT loss (far):       {v_far.item():.8f}')

# 5) صورة عشوائية تماماً (متوقع: FFT كبير)
sr_random = torch.randn_like(hr)
v_random = l(sr_random, hr)
print(f'FFT loss (random):    {v_random.item():.8f}')

# التحليل
print()
print('=' * 50)
print('Analysis:')
print(f'  identical: {v_identical.item():.2e}  (يجب ≈ 0)')
print(f'  close:     {v_close.item():.2e}  (يجب صغير)')
print(f'  far:       {v_far.item():.2e}  (يجب أكبر من close)')
print(f'  random:    {v_random.item():.2e}  (يجب الأكبر)')
print('=' * 50)

# القرار
if v_close.item() < 1e-6:
    print('⚠️  FFT loss صغير جداً — قد يحتاج وزن أكبر')
    print(f'   القيمة الحالية × weight={0.0019062374828215077}')
    print(f'   = {v_close.item() * 0.0019062374828215077:.2e}')
    print(f'   → يظهر كـ 0.0000 في السجل')
    print()
    print('   💡 الحل: زيادة weight إلى 0.02')
elif v_close.item() > 1e-3:
    print('✅ FFT loss يعمل بشكل جيد')
else:
    print('✅ FFT loss يعمل (قيم صغيرة طبيعية)')