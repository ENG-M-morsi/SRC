# ===================================================================
# scripts/optuna_tuner_arch.py
# تحسين Hyperparameters المعمارية لـ DTCF-SR
#
# الاستخدام:
#   python scripts/optuna_tuner_arch.py --n_trials 50 --data_dir ./data
# ===================================================================
import os
import argparse
import optuna
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dhtcun import HUTCN

# ⚠️ عدّل هذا الاستيراد حسب مشروعك
try:
    from data.div2k import DIV2KDataset
except ImportError:
    DIV2KDataset = None


# ─────────────────────────────────────────────────────────────────
# أدوات مساعدة
# ─────────────────────────────────────────────────────────────────
def calc_psnr(sr, hr, shave=4):
    sr = sr[:, :, shave:-shave, shave:-shave]
    hr = hr[:, :, shave:-shave, shave:-shave]
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def build_loaders(data_dir, batch_size, scale=4):
    if DIV2KDataset is None:
        raise RuntimeError("DIV2KDataset غير متوفر — عدّل الاستيراد")
    train_set = DIV2KDataset(data_dir, 'train', scale=scale, patch_size=48)
    val_set   = DIV2KDataset(data_dir, 'val',   scale=scale, patch_size=48)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1,
                              shuffle=False, num_workers=2)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────
def objective(trial):
    # (أ) اقتراح المعمارية
    nf              = trial.suggest_categorical('nf', [64, 80, 88, 96])
    num_heads_dat   = trial.suggest_categorical('num_heads_dat', [2, 4, 8])
    ws_dat          = trial.suggest_categorical('ws_dat', [4, 8, 12, 16])
    num_blocks_dat  = trial.suggest_int('num_blocks_dat', 1, 3)
    num_heads_elan  = trial.suggest_categorical('num_heads_elan', [2, 4, 8])
    ws_elan         = trial.suggest_categorical('ws_elan', [4, 8, 12, 16])
    num_blocks_elan = trial.suggest_int('num_blocks_elan', 1, 3)
    fusion_heads    = trial.suggest_categorical('fusion_heads', [2, 4, 8])
    fusion_dropout  = trial.suggest_float('fusion_dropout', 0.0, 0.3, step=0.1)
    hfe_reduction   = trial.suggest_categorical('hfe_reduction', [2, 4, 8])

    # تحقق من التوافق
    for h in (num_heads_dat, num_heads_elan, fusion_heads):
        if nf % h != 0:
            raise optuna.TrialPruned()

    # (ب) اقتراح التدريب
    lr           = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (ج) بناء النموذج
    try:
        model = HUTCN(
            in_nc=3, nf=nf, out_nc=3, upscale=4,
            num_heads_dat=num_heads_dat, ws_dat=ws_dat,
            num_blocks_dat=num_blocks_dat,
            num_heads_elan=num_heads_elan, ws_elan=ws_elan,
            num_blocks_elan=num_blocks_elan,
            fusion_heads=fusion_heads, fusion_dropout=fusion_dropout,
            hfe_reduction=hfe_reduction,
        ).to(device)
    except Exception as e:
        print(f"[Trial {trial.number}] Model build failed: {e}")
        raise optuna.TrialPruned()

    # (د) البيانات
    try:
        train_loader, val_loader = build_loaders(DATA_DIR, batch_size, scale=4)
    except Exception as e:
        print(f"DataLoader error: {e}")
        raise optuna.TrialPruned()

    # (هـ) Optimizer — Loss ثابت L1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=lr * 0.01
    )
    criterion = nn.L1Loss()

    best_psnr = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for lr_img, hr_img in tqdm(train_loader,
                                   desc=f'Trial {trial.number} Ep {epoch}',
                                   leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # تقييم
        model.eval()
        psnr_sum = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr = model(lr_img)
                psnr_sum += calc_psnr(sr, hr_img)
        avg_psnr = psnr_sum / max(len(val_loader), 1)
        best_psnr = max(best_psnr, avg_psnr)

        trial.report(avg_psnr, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_psnr


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def run(data_dir, n_trials, num_epochs, out_dir):
    global DATA_DIR, NUM_EPOCHS
    DATA_DIR   = data_dir
    NUM_EPOCHS = num_epochs

    os.makedirs(out_dir, exist_ok=True)
    storage = f'sqlite:///{os.path.join(out_dir, "optuna_arch.db")}'

    study = optuna.create_study(
        study_name='dtcf_arch',
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8,
                                            multivariate=True),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   gc_after_trial=True)

    print("=" * 60)
    print("BEST ARCHITECTURE:")
    print(f"  PSNR: {study.best_value:.4f} dB")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # ✅ CSV بكل التجارب
    study.trials_dataframe().to_csv(
        os.path.join(out_dir, 'optuna_arch_results.csv'), index=False
    )

    # ✅ JSON بأفضل معاملات فقط — هذا ما نضيفه
    best_json_path = os.path.join(out_dir, 'best_arch.json')
    with open(best_json_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)

    print(f"\nSaved:")
    print(f"  DB   : {storage}")
    print(f"  CSV  : {os.path.join(out_dir, 'optuna_arch_results.csv')}")
    print(f"  JSON : {best_json_path}")
    return study

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--n_trials', type=int, default=50)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--out_dir', type=str, default='./optuna_arch_out')
    args = p.parse_args()

    run(args.data_dir, args.n_trials, args.num_epochs, args.out_dir)