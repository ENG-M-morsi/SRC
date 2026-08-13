# ===================================================================
# hyperopt_with_your_data.py — بحث سريع مع OmniSR و HUTCN (محسّن للوقت)
# ===================================================================
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import os
import sys
import itertools

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import data
import model as model_module
import loss as loss_module
from option import args as base_args
from model.dhtcun import HUTCN
from model import dhtcu_block as B
from model.omnisr_attention_block import OmniSR

# ===================================================================
# دوال الخسارة الإضافية
# ===================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))

class HuberLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(HuberLoss, self).__init__()
        self.delta = delta
    def forward(self, x, y):
        diff = torch.abs(x - y)
        mask = (diff < self.delta).float()
        return torch.mean(mask * (x - y)**2 + (1 - mask) * (2 * self.delta * diff - self.delta**2))

# ===================================================================
# دالة إنشاء DataLoaders
# ===================================================================
def get_loaders_from_args(trial_params, base_args):
    args = base_args
    args.patch_size = trial_params['patch_size']
    args.batch_size = trial_params['batch_size'] if 'batch_size' in trial_params else base_args.batch_size
    args.dir_data = r'D:\Mohamed Morsi\DATA'
    if not hasattr(args, 'num_workers'):
        args.num_workers = 4
    if not hasattr(args, 'pin_memory'):
        args.pin_memory = True
    loader = data.Data(args)
    return loader.loader_train, loader.loader_test

# ===================================================================
# دالة الهدف الرئيسية
# ===================================================================
def objective(trial):
    # ---------- معاملات النموذج ----------
    nf = trial.suggest_int('nf', 32, 64, step=8)
    num_heads = trial.suggest_categorical('num_heads', [2, 4])
    if nf % num_heads != 0:
        raise optuna.TrialPruned()

    upscale = 4
    patch_size = trial.suggest_categorical('patch_size', [64, 96, 128])
    lr_size = patch_size // upscale

    # ws ثابتة القيم، مع شرط الصلاحية
    ws = trial.suggest_categorical('ws', [2, 4, 6, 8])
    if ws * 4 > lr_size:
        raise optuna.TrialPruned()

    num_blocks = trial.suggest_int('num_blocks', 1, 2, step=1)

    optimizer_name = trial.suggest_categorical('optimizer', ['ADAM', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler', ['fixed', 'cosine'])
    loss_name = trial.suggest_categorical('loss', ['L1', 'Charbonnier'])
    lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    # ---------- بناء النموذج ----------
    model = HUTCN(in_nc=3, nf=nf, num_modules=1, out_nc=3, upscale=upscale)

    for module in model.modules():
        if isinstance(module, B.TCN):
            module.omnisr = OmniSR(
                dim=nf,
                num_heads=num_heads,
                ws=ws,
                num_blocks=num_blocks
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------- المُحسّن ----------
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    # ---------- جدول التوهين ----------
    EPOCHS = 10
    NUM_BATCHES_PER_EPOCH = 30
    if scheduler_name == 'fixed':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    else:
        scheduler = lrs.StepLR(optimizer, step_size=2, gamma=0.5)

    # ---------- دالة الخسارة ----------
    if loss_name == 'L1':
        criterion = nn.L1Loss()
    elif loss_name == 'Charbonnier':
        criterion = CharbonnierLoss(eps=1e-3)
    else:
        criterion = HuberLoss(delta=0.01)

    # ---------- تحميل البيانات ----------
    trial_params = {
        'patch_size': patch_size,
        'batch_size': 16
    }
    train_loader, test_loaders = get_loaders_from_args(trial_params, base_args)
    val_loader = test_loaders[0] if test_loaders else None
    if val_loader is None:
        raise optuna.TrialPruned()

    # ---------- حلقة تدريب سريعة ----------
    best_psnr = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (lr, hr, _) in enumerate(itertools.islice(train_loader, NUM_BATCHES_PER_EPOCH)):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # تقييم سريع (أول 10 دفعات من val)
        model.eval()
        psnr_sum = 0.0
        count = 0
        with torch.no_grad():
            for lr, hr, _ in itertools.islice(val_loader, 10):
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                mse = torch.mean((sr - hr) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                psnr_sum += psnr.item()
                count += 1
        avg_psnr = psnr_sum / count if count > 0 else 0.0

        trial.report(avg_psnr, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_psnr = max(best_psnr, avg_psnr)

    return best_psnr

# ===================================================================
# تشغيل البحث
# ===================================================================
if __name__ == "__main__":
    N_TRIALS = 10

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )

    print(f"🚀 بدء البحث السريع ({N_TRIALS} محاولة، كل محاولة {3} Epochs، 30 دفعة لكل Epoch)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*70)
    print("🏆 أفضل المعاملات:")
    print("="*70)
    for key, value in study.best_params.items():
        print(f"  {key:>20} : {value}")
    print(f"\n📈 أفضل PSNR: {study.best_value:.3f} dB")
    print("="*70)

    import json
    with open('best_params_optimized.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ تم حفظ أفضل المعاملات في best_params_optimized.json")