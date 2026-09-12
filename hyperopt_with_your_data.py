# ===================================================================
# hyperopt_with_your_data.py — بحث شامل مع معاملات منفصلة
# ✅ النسخة المحسّنة: تدعم الاستئناف بعد انقطاع الكهرباء
# ===================================================================
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import os
import sys
import copy
import json
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import data
import model as model_module
import loss as loss_module
from option import args as base_args
from model.dhtcun import HUTCN

# ===================================================================
# إعدادات الاستئناف
# ===================================================================
STUDY_NAME = "hutcn_hyperopt_study"
STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "optuna_study.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

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
# دالة تحميل البيانات
# ===================================================================
def get_loaders_from_args(trial_params, base_args):
    args = copy.deepcopy(base_args)
    args.data_train = ['DIV2K']
    args.data_test = ['DIV2K']
    args.data_range = '1-800/896-900'
    args.scale = [4]
    args.dir_data = r'D:\Mohamed Morsi\DATA'
    args.patch_size = trial_params['patch_size']
    args.batch_size = trial_params['batch_size']
    loader = data.Data(args)
    return loader.loader_train, loader.loader_test

# ===================================================================
# دالة الهدف الرئيسية
# ===================================================================
def objective(trial):
    # ---------- (أ) معاملات بنية النموذج ----------
    nf = trial.suggest_int('n_feats', 32, 128, step=8)

    # --- معاملات DAT ---
    num_heads_dat = trial.suggest_categorical('num_heads_dat', [2, 4, 8])
    if nf % num_heads_dat != 0:
        raise optuna.TrialPruned()
    ws_dat = trial.suggest_categorical('ws_dat', [4, 6, 8, 12, 16])
    num_blocks_dat = trial.suggest_int('num_blocks_dat', 1, 4, step=1)

    # --- معاملات ELAN ---
    num_heads_elan = trial.suggest_categorical('num_heads_elan', [2, 4, 8])
    if nf % num_heads_elan != 0:
        raise optuna.TrialPruned()
    ws_elan = trial.suggest_categorical('ws_elan', [4, 6, 8, 12, 16])
    num_blocks_elan = trial.suggest_int('num_blocks_elan', 1, 4, step=1)

    # --- معاملات مشتركة ---
    batch_size = trial.suggest_categorical('batch_size', [4, 8])
    patch_size = trial.suggest_categorical('patch_size', [128, 160, 192, 224])
    if ws_dat > patch_size or ws_elan > patch_size:
        raise optuna.TrialPruned()

    # ---------- (ب) معاملات التدريب ----------
    optimizer_name = trial.suggest_categorical('optimizer', ['ADAM', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler', ['fixed', 'cosine', 'step'])
    loss_name = trial.suggest_categorical('loss', ['L1', 'L2', 'Charbonnier', 'Huber'])
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ---------- (ج) بناء النموذج ----------
    model = HUTCN(
        in_nc=3,
        nf=nf,
        num_modules=1,
        out_nc=3,
        upscale=4,
        num_heads_dat=num_heads_dat,
        ws_dat=ws_dat,
        num_blocks_dat=num_blocks_dat,
        num_heads_elan=num_heads_elan,
        ws_elan=ws_elan,
        num_blocks_elan=num_blocks_elan
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------- (د) المُحسّن ----------
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    # ---------- (هـ) جدول توهين ----------
    EPOCHS = 25
    if scheduler_name == 'fixed':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    else:
        scheduler = lrs.StepLR(optimizer, step_size=3, gamma=0.5)

    # ---------- (و) دالة الخسارة ----------
    if loss_name == 'L1':
        criterion = nn.L1Loss()
    elif loss_name == 'L2':
        criterion = nn.MSELoss()
    elif loss_name == 'Charbonnier':
        criterion = CharbonnierLoss(eps=1e-3)
    else:
        criterion = HuberLoss(delta=0.01)

    # ---------- (ز) تحميل البيانات ----------
    trial_params = {'patch_size': patch_size, 'batch_size': batch_size}
    train_loader, test_loaders = get_loaders_from_args(trial_params, base_args)
    val_loader = test_loaders[0] if test_loaders else None
    if val_loader is None:
        raise optuna.TrialPruned()

    # ---------- (ح) حلقة التدريب ----------
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

            if batch_idx % 50 == 0:
                print(f'Trial {trial.number} | Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

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

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

    return best_psnr

# ===================================================================
# تشغيل البحث (مع دعم الاستئناف)
# ===================================================================
if __name__ == "__main__":
    N_TRIALS = 20  # إجمالي عدد المحاولات المطلوبة

    # --------------------------------------------------------------
    # ✅ إنشاء أو تحميل الدراسة السابقة (Persistence)
    # --------------------------------------------------------------
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        load_if_exists=True   # ← يحمّل الدراسة إن كانت موجودة
    )

    # عرض ملخص للدراسة الحالية (إن كانت موجودة)
    completed_trials = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
    total_trials = len(study.trials)

    print("=" * 70)
    print(f"📚 اسم الدراسة: {STUDY_NAME}")
    print(f"📂 مسار التخزين: {STORAGE_PATH}")
    print(f"✅ محاولات مكتملة سابقاً: {completed_trials}")
    print(f"📊 إجمالي المحاولات المسجلة: {total_trials}")
    if completed_trials > 0:
        print(f"🏆 أفضل PSNR سابق: {study.best_value:.3f} dB")
        print(f"🎯 أفضل trial: {study.best_trial.number}")
    print("=" * 70)

    # حساب عدد المحاولات المتبقية
    remaining_trials = N_TRIALS - completed_trials
    if remaining_trials <= 0:
        print("✅ تم إكمال جميع المحاولات المطلوبة مسبقاً.")
    else:
        print(f"🚀 بدء/استئناف البحث ({remaining_trials} محاولة متبقية)...")

        try:
            # ----------------------------------------------------------
            # ✅ حلقة البحث الآمنة (تدعم الإيقاف بـ Ctrl+C)
            # ----------------------------------------------------------
            study.optimize(
                objective,
                n_trials=remaining_trials,
                show_progress_bar=True,
                catch=(RuntimeError,)  # تجاهل أخطاء CUDA المؤقتة
            )
        except KeyboardInterrupt:
            print("\n\n⚠️ تم إيقاف البحث يدوياً (Ctrl+C).")
            print("💾 تم حفظ جميع النتائج تلقائياً في قاعدة البيانات.")
            print("🔄 لتكملة البحث، أعد تشغيل نفس الأمر.")

    # ==============================================================
    # عرض وحفظ النتائج النهائية
    # ==============================================================
    print("\n" + "=" * 70)
    print("🏆 أفضل المعاملات حتى الآن:")
    print("=" * 70)
    for key, value in study.best_params.items():
        print(f"  {key:>20} : {value}")
    print(f"\n📈 أفضل PSNR على مجموعة التحقق: {study.best_value:.3f} dB")
    print("=" * 70)

    # حفظ النتائج في JSON
    output_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'best_params_optimized_separate.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=4, ensure_ascii=False)
    print(f"✅ تم حفظ أفضل المعاملات في: {output_json}")

    # (اختياري) حفظ جميع المحاولات في CSV للمراجعة
    try:
        df = study.trials_dataframe()
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'optuna_all_trials.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"📊 تم حفظ جميع المحاولات في: {csv_path}")
    except Exception as e:
        print(f"⚠️ تعذّر حفظ ملف CSV: {e}")