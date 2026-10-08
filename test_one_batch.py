import sys, time
sys.path.append('.')
import torch
from types import SimpleNamespace
from torch.cuda.amp import autocast, GradScaler
from model.dhtcun import HUTCN
from loss import Loss
from optuna_tuner_loss import build_loaders, make_loss_args

device = 'cuda'

# 1. DataLoader
t0 = time.time()
tr, _ = build_loaders('D:/Mohamed Morsi/DATA', batch_size=8)
print(f'DataLoader built: {time.time()-t0:.1f}s, batches: {len(tr)}')

# 2. النموذج
model = HUTCN(in_nc=3, nf=88, out_nc=3, upscale=4,
              num_heads_dat=2, ws_dat=4, num_blocks_dat=3,
              num_heads_elan=2, ws_elan=8, num_blocks_elan=1,
              fusion_heads=8, fusion_dropout=0.1, hfe_reduction=2).to(device)
print(f'Model built: {time.time()-t0:.1f}s')

# 3. Loss
crit = Loss(make_loss_args("1*L1+0.005*VGG22+0.005*FFT+0.02*EDGE"), None).to(device)
crit.train()
print(f'Loss built: {time.time()-t0:.1f}s')

# 4. first batch forward
it = iter(tr)
batch = next(it)
lr_img, hr_img = batch[0].to(device), batch[1].to(device)
print(f'Batch loaded: LR={tuple(lr_img.shape)} HR={tuple(hr_img.shape)}')
print(f'Time so far: {time.time()-t0:.1f}s')

crit.start_log()
t1 = time.time()
with autocast():
    sr = model(lr_img)
print(f'Model forward: {time.time()-t1:.1f}s')

t1 = time.time()
loss = crit(sr, hr_img)      # ← خارج autocast (هو موجود بالفعل خارج)
print(f'Loss forward: {time.time()-t1:.1f}s, value={loss.item():.4f}')

# 5. backward
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = GradScaler()
t1 = time.time()
optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
print(f'Backward + step: {time.time()-t1:.1f}s')
print(f'TOTAL: {time.time()-t0:.1f}s')