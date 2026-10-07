# ===================================================================
# scripts/optuna_tuner_loss.py
# تحسين أوزان Loss فقط (المعمارية ثابتة).
#
# نستخدم loss.Loss الموجود كما هو — لا نعدّله.
# التركيبة: L1 + VGG22 + FFT  (+ EDGE اختياري)
#
# الاستخدام:
#   python scripts/optuna_tuner_loss.py --data_dir ./data \
#          --arch_params best_arch.json --n_trials 25
# ===================================================================
import os
import json
import argparse
import optuna
import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
import json          # ← أضف هذا السطر
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dhtcun import HUTCN
from loss import Loss  # ← نستخدم Loss الموجود كما هو

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
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def build_loaders(data_dir, batch_size, scale=4):
    if DIV2KDataset is None:
        raise RuntimeError("DIV2KDataset غير متوفر")
    train_set = DIV2KDataset(data_dir, 'train', scale=scale, patch_size=48)
    val_set   = DIV2KDataset(data_dir, 'val',   scale=scale, patch_size=48)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=1,
                              shuffle=False, num_workers=2)
    return train_loader, val_loader


def make_args(loss_string):
    """args بسيطة لاستخدامها مع loss.Loss الموجود."""
    return SimpleNamespace(
        loss=loss_string,
        n_GPUs=1,
        cpu=False,
        precision='single',
        rgb_range=1.0,
        weight_decay=1e-5,
        decay='step',
        gamma=0.5,
        gan_k=1,
        load='',
    )


# ─────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────
def objective(trial):
    # ─── (أ) اقتراح الأوزان — log scale ───
    lam_vgg = trial.suggest_float('lam_vgg',  1e-3, 1e-1, log=True)
    lam_fft = trial.suggest_float('lam_fft',  1e-4, 1e-2, log=True)
    lam_edge = trial.suggest_float('lam_edge', 1e-3, 1e-1, log=True)

    loss_string = (
        f"1*L1"
        f"+{lam_vgg}*VGG22"
        f"+{lam_fft}*FFT"
        f"+{lam_edge}*EDGE"
    )
    print(f"\n[Trial {trial.number}] loss = {loss_string}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── (ب) بناء النموذج بأفضل معمارية ───
    model = HUTCN(
        in_nc=3, nf=ARCH['nf'], out_nc=3, upscale=4,
        num_heads_dat=ARCH['num_heads_dat'],
        ws_dat=ARCH['ws_dat'],
        num_blocks_dat=ARCH['num_blocks_dat'],
        num_heads_elan=ARCH['num_heads_elan'],
        ws_elan=ARCH['ws_elan'],
        num_blocks_elan=ARCH['num_blocks_elan'],
        fusion_heads=ARCH['fusion_heads'],
        fusion_dropout=ARCH['fusion_dropout'],
        hfe_reduction=ARCH['hfe_reduction'],
    ).to(device)

    # ─── (ج) Loss من loss.Loss الموجود ───
    args = make_args(loss_string)
    criterion = Loss(args, ckp=None).to(device)

    # ─── (د) Optimizer ───
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=2e-6
    )

    # ─── (هـ) بيانات ───
    train_loader, val_loader = build_loaders(DATA_DIR, batch_size=16, scale=4)

    # ─── (و) تدريب ───
    WARMUP_EPOCHS = 1   # أول epoch: L1 فقط
    best_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        # Warmup: في أول epoch نستخدم L1 فقط
        if epoch < WARMUP_EPOCHS:
            warm_args = make_args("1*L1")
            criterion_warm = Loss(warm_args, ckp=None).to(device)
            cur_crit = criterion_warm
        else:
            cur_crit = criterion

        for lr_img, hr_img in tqdm(train_loader,
                                    desc=f'Trial {trial.number} Ep {epoch}',
                                    leave=False):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            sr = model(lr_img)
            loss = cur_crit(sr, hr_img)
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
def run(data_dir, arch_json, n_trials, num_epochs, out_dir):
    global DATA_DIR, ARCH, NUM_EPOCHS
    DATA_DIR   = data_dir
    NUM_EPOCHS = num_epochs

    # تحميل أفضل معمارية
    with open(arch_json, 'r') as f:
        ARCH = json.load(f)
    print("Loaded architecture:")
    for k, v in ARCH.items():
        print(f"  {k}: {v}")

    os.makedirs(out_dir, exist_ok=True)
    storage = f'sqlite:///{os.path.join(out_dir, "optuna_loss.db")}'

    study = optuna.create_study(
        study_name='dtcf_loss',
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   gc_after_trial=True)

    # ─── طباعة أفضل النتائج ───
    print("=" * 60)
    print("BEST LOSS WEIGHTS:")
    print(f"  PSNR: {study.best_value:.4f} dB")
    print(f"  lam_vgg  = {study.best_params['lam_vgg']:.4e}")
    print(f"  lam_fft  = {study.best_params['lam_fft']:.4e}")
    print(f"  lam_edge = {study.best_params['lam_edge']:.4e}")
    print("=" * 60)

    # ─── حفظ CSV بكل التجارب ───
    csv_path = os.path.join(out_dir, 'optuna_loss_results.csv')
    study.trials_dataframe().to_csv(csv_path, index=False)

    # ─── ✅ حفظ JSON بأفضل قيم فقط ───
    best_json_path = os.path.join(out_dir, 'best_loss.json')

    # نضيف أيضاً loss_string النهائي الجاهز للاستخدام المباشر
    best = study.best_params
    best['loss_string'] = (
        f"1*L1"
        f"+{best['lam_vgg']}*VGG22"
        f"+{best['lam_fft']}*FFT"
        f"+{best['lam_edge']}*EDGE"
    )
    best['best_psnr'] = study.best_value

    with open(best_json_path, 'w') as f:
        json.dump(best, f, indent=2)

    # ─── طباعة ملخص الملفات المحفوظة ───
    print(f"\nSaved:")
    print(f"  DB   : {storage}")
    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {best_json_path}")
    print(f"\nFinal loss string:")
    print(f"  --loss \"{best['loss_string']}\"")

    return study


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--arch_params', type=str, required=True,
                   help='مسار JSON فيه best_params من optuna_arch')
    p.add_argument('--n_trials', type=int, default=25)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--out_dir', type=str, default='./optuna_loss_out')
    args = p.parse_args()

    run(args.data_dir, args.arch_params, args.n_trials,
        args.num_epochs, args.out_dir)