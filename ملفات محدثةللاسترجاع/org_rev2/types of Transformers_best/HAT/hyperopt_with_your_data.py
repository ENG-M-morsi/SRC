# ===================================================================
# hyperopt_with_your_data.py — بحث شامل مع HAT (Hybrid Attention Transformer)
# ===================================================================
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import os
import sys
import importlib
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import data
import model as model_module
import loss as loss_module
from option import args as base_args
from model.dhtcun import HUTCN
from model import dhtcu_block as B
from model.custom_attention_blocks import HAT  # ✅ استيراد HAT
import pdb

# ===================================================================
# 1. دوال الخسارة الإضافية (Charbonnier & Huber) كما في السابق
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
# 2. دالة لإنشاء DataLoaders باستخدام الـ Data class الخاص بك
# ===================================================================
def get_loaders_from_args(trial_params, base_args):
    args = base_args
    args.patch_size = trial_params['patch_size']
    args.batch_size = trial_params['batch_size'] if 'batch_size' in trial_params else base_args.batch_size
    
    # ✅ تأكد من مسار البيانات الصحيح
    # استخدم المسار المطلق لتجنب أي مشاكل في المسار النسبي
    args.dir_data = r'D:\Mohamed Morsi\DATA'
    
    # تأكد من أن data_range صحيح
    # args.data_range = '1-800/801-900'  # تأكد من وجود هذا السطر في option.py
    
    loader = data.Data(args)
    return loader.loader_train, loader.loader_test

# ===================================================================
# 3. دالة الهدف الرئيسية (Objective) — مع HAT
# ===================================================================
def objective(trial):
    # ---------- (أ) معاملات بنية النموذج ----------
    nf = trial.suggest_int('n_feats', 64, 128, step=8)

    num_heads_options = [2, 4, 8, 16]
    num_heads = trial.suggest_categorical('num_heads', num_heads_options)
    if nf % num_heads != 0:
        raise optuna.TrialPruned()

    # عدد الكتل داخل HAT (num_blocks في HAT)
    num_blocks = trial.suggest_int('num_blocks', 2, 4, step=1)

    window_size = trial.suggest_categorical('window_size', [8, 12, 16])

    num_modules = 1  # عدد كتل P_HTCB (ثابت)

    # ---------- (ب) patch_size ----------
    #patch_size = trial.suggest_categorical('patch_size', [128, 160, 192, 224, 256])
    patch_size = 192
    if window_size > patch_size:
        raise optuna.TrialPruned()

    # ---------- (ج) معاملات التدريب ----------
    optimizer_name = trial.suggest_categorical('optimizer', ['ADAM', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler', ['fixed', 'cosine', 'step'])
    loss_name = trial.suggest_categorical('loss', ['L1', 'L2', 'Charbonnier', 'Huber'])
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ---------- (د) بناء النموذج ----------
    # HUTCN لا يأخذ num_heads
    model = HUTCN(in_nc=3, nf=nf, num_modules=num_modules, out_nc=3, upscale=4)

    # ✅ إعادة تعريف HAT داخل كل TCN بالمعاملات الجديدة
    for module in model.modules():
        if isinstance(module, B.TCN):
            module.hat = HAT(
                dim=nf,
                num_heads=num_heads,
                window_size=window_size,
                num_blocks=num_blocks
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------- (هـ) المُحسّن ----------
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    # ---------- (و) جدول توهين معدل التعلم ----------
    EPOCHS = 10
    if scheduler_name == 'fixed':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    else:
        scheduler = lrs.StepLR(optimizer, step_size=3, gamma=0.5)

    # ---------- (ز) دالة الخسارة ----------
    if loss_name == 'L1':
        criterion = nn.L1Loss()
    elif loss_name == 'L2':
        criterion = nn.MSELoss()
    elif loss_name == 'Charbonnier':
        criterion = CharbonnierLoss(eps=1e-3)
    else:
        criterion = HuberLoss(delta=0.01)

    # ---------- (ح) تحميل البيانات ----------
    trial_params = {
        'patch_size': patch_size,
        'batch_size': 8
    }
    train_loader, test_loaders = get_loaders_from_args(trial_params, base_args)
    val_loader = test_loaders[0] if test_loaders else None
    if val_loader is None:
        raise optuna.TrialPruned()

    # ---------- (ط) حلقة التدريب ----------
    best_psnr = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (lr, hr, _) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # التقييم
        model.eval()
        psnr_sum = 0.0
        with torch.no_grad():
            for lr, hr, _ in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                mse = torch.mean((sr - hr) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                psnr_sum += psnr.item()

        avg_psnr = psnr_sum / len(val_loader)
        trial.report(avg_psnr, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_psnr = max(best_psnr, avg_psnr)

    return best_psnr

# ===================================================================
# 4. تشغيل البحث الرئيسي
# ===================================================================
if __name__ == "__main__":
    N_TRIALS = 10

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    print(f"🚀 بدء البحث الشامل ({N_TRIALS} محاولة، كل محاولة {10} Epochs)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*70)
    print("🏆 أفضل المعاملات:")
    print("="*70)
    for key, value in study.best_params.items():
        print(f"  {key:>20} : {value}")
    print(f"\n📈 أفضل PSNR على مجموعة التحقق: {study.best_value:.3f} dB")
    print("="*70)

    import json
    with open('best_params_optimized.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ تم حفظ أفضل المعاملات في best_params_optimized.json")