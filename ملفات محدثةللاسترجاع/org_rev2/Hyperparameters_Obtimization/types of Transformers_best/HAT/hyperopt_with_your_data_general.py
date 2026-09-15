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
import json    ####################

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

# ═══════════════════════════════════════════════════════════
# 🆕 إضافة هذه الأسطر (لا تحذف شيئاً)
# ═══════════════════════════════════════════════════════════
N_TRIALS = 50
EPOCHS_PER_TRIAL = 50
N_STARTUP_TRIALS = 5
N_WARMUP_STEPS = 3
STUDY_NAME = "hutcn_hyperopt_study"
STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "optuna_study.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

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
# دالة إنشاء DataLoaders (مصححة)
# ===================================================================
def get_loaders_from_args(trial_params, base_args):
    args = copy.deepcopy(base_args)
    # إعدادات البيانات الصحيحة (كقوائم لتجنب خطأ data.d)
    args.data_train = ['DIV2K']
    args.data_test = ['DIV2K']
    args.data_range = '1-800/896-900'
    args.scale = [4]  # قائمة أعداد صحيحة
    args.dir_data = r'D:\Mohamed Morsi\DATA'
    args.patch_size = trial_params['patch_size']
    args.batch_size = trial_params['batch_size']
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
    patch_size = trial.suggest_categorical('patch_size', [128, 160, 192, 224, 256])
    #patch_size = 192
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
# تشغيل البحث
# ===================================================================
if __name__ == "__main__":
    
    # 1️⃣ إنشاء أو تحميل الدراسة
    study = optuna.create_study(
        study_name=STUDY_NAME,                   # ← جديد
        storage=STORAGE_URL,                      # ← جديد
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=N_STARTUP_TRIALS,
            n_warmup_steps=N_WARMUP_STEPS,
        ),
        load_if_exists=True                       # ← جديد
    )

    # 2️⃣ حساب المحاولات المتبقية
    complete_trials = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = N_TRIALS - len(complete_trials)

    if remaining <= 0:
        print(f"✅ اكتمل البحث. أفضل PSNR: {study.best_value:.4f} dB")
    else:
        print(f"🚀 بدء/استئناف البحث ({remaining} محاولة متبقية)...")
        try:
            study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        except KeyboardInterrupt:
            print("\n⚠️ تم الإيقاف. النتائج محفوظة في قاعدة البيانات.")

    # 3️⃣ حفظ النتائج
    if len(study.trials) > 0:
        with open('best_params_optimized.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)
        study.trials_dataframe().to_csv('optuna_all_trials.csv', 
                                         index=False, encoding='utf-8-sig')
        print("✅ تم حفظ النتائج")