# ===================================================================
# optuna_tuner_loss.py — تحسين أوزان Loss
#
# المعمارية ثابتة (من Fine Search)، نُحسّن فقط:
#   λ_vgg   ∈ [1e-3, 1e-1]  (log scale)
#   λ_fft   ∈ [1e-4, 1e-2]  (log scale)
#   λ_edge  ∈ [1e-3, 1e-1]  (log scale)
#
# Loss = 1*L1 + λ_vgg*VGG22 + λ_fft*FFT + λ_edge*EDGE
#
# الاستخدام:
#   python optuna_tuner_loss.py \
#     --data_dir "D:/Mohamed Morsi/DATA" \
#     --arch_params ./optuna_fine_out/best_arch_fine.json \
#     --n_trials 20 --num_epochs 10 \
#     --out_dir ./optuna_loss_out
# ===================================================================
import os
import sys
import json
import argparse
import optuna
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dhtcun import HUTCN
from loss import Loss


# ═════════════════════════════════════════════════════════════════
# إعداد args لـ data.Data
# ═════════════════════════════════════════════════════════════════
def make_data_args(data_dir, batch_size, scale=4, patch_size=96,
                   data_range='1-800/896-900'):
    return SimpleNamespace(
        dir_data=data_dir,
        data_train=['DIV2K'],
        data_test=['DIV2K'],
        data_range=data_range,
        scale=[scale],
        patch_size=patch_size,
        n_colors=3,
        rgb_range=1.0,
        ext='img',
        no_augment=False,
        test_every=1000,
        batch_size=batch_size,
        n_threads=4,
        cpu=False,
        test_only=False,
        model='dhtcun',
        reset=False,
        preprocess='none',
        dir_demo='.',
    )


def build_loaders(data_dir, batch_size, scale=4, patch_size=96):
    from data import Data
    args = make_data_args(data_dir, batch_size, scale, patch_size)
    d = Data(args)
    if d.loader_train is None or len(d.loader_test) == 0:
        raise RuntimeError('DataLoader failed')
    return d.loader_train, d.loader_test[0]


def make_loss_args(loss_string, rgb_range=1.0):
    return SimpleNamespace(
        loss=loss_string,
        n_GPUs=1,
        cpu=False,
        precision='single',
        rgb_range=rgb_range,
        weight_decay=1e-5,
        decay='step',
        gamma=0.5,
        gan_k=1,
        load='',
    )


def calc_psnr(sr, hr, shave=4):
    sr = sr[:, :, shave:-shave, shave:-shave]
    hr = hr[:, :, shave:-shave, shave:-shave]
    mse = torch.mean((sr - hr) ** 2).item()
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


# ═════════════════════════════════════════════════════════════════
# Objective
# ═════════════════════════════════════════════════════════════════
def objective(trial):
    # ── اقتراح الأوزان (log scale) ──
    lam_vgg  = trial.suggest_float('lam_vgg',  1e-3, 1e-1, log=True)
    lam_fft  = trial.suggest_float('lam_fft',  1e-4, 1e-2, log=True)
    lam_edge = trial.suggest_float('lam_edge', 1e-3, 1e-1, log=True)

    loss_string = f"1*L1+{lam_vgg}*VGG22+{lam_fft}*FFT+{lam_edge}*EDGE"
    print(f'\n[Trial {trial.number}] loss = {loss_string}')

    device = torch.device('cuda')

    # ── بناء النموذج بالمعمارية الثابتة ──
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

    # ── Loss ──
    criterion = Loss(make_loss_args(loss_string), ckp=None).to(device)
    criterion.train()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ARCH['lr'],
        weight_decay=ARCH['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=ARCH['lr'] * 0.01
    )
    scaler = GradScaler()

    # ── DataLoaders ──
    train_loader, val_loader = build_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, scale=4, patch_size=96
    )

    # ── Warmup: أول epoch L1 فقط ──
    warmup_criterion = Loss(make_loss_args("1*L1"), ckp=None).to(device)
    warmup_criterion.train()

    best_psnr = 0.0
    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            cur_crit = warmup_criterion if epoch < WARMUP_EPOCHS else criterion

            for batch in tqdm(train_loader,
                              desc=f'Trial {trial.number} Ep {epoch}',
                              leave=False):
                lr_img = batch[0].to(device)
                hr_img = batch[1].to(device)
                optimizer.zero_grad()
                with autocast():
                    sr = model(lr_img)
                    loss = cur_crit(sr, hr_img)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            # تقييم (بدون loss، فقط PSNR)
            model.eval()
            psnr_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    lr_img = batch[0].to(device)
                    hr_img = batch[1].to(device)
                    sr = model(lr_img)
                    psnr_sum += calc_psnr(sr, hr_img)
            avg_psnr = psnr_sum / max(len(val_loader), 1)
            best_psnr = max(best_psnr, avg_psnr)

            trial.report(avg_psnr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    except torch.cuda.OutOfMemoryError:
        print(f'[Trial {trial.number}] OOM — pruned')
        del model, optimizer, scheduler, criterion, warmup_criterion
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        raise optuna.TrialPruned()

    # تنظيف
    del model, optimizer, scheduler, criterion, warmup_criterion
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return best_psnr


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════
def run(data_dir, arch_json, n_trials, num_epochs, out_dir,
        batch_size=8, warmup_epochs=1):
    global DATA_DIR, ARCH, NUM_EPOCHS, BATCH_SIZE, WARMUP_EPOCHS
    DATA_DIR = data_dir
    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    WARMUP_EPOCHS = warmup_epochs

    with open(arch_json, 'r') as f:
        ARCH = json.load(f)
    print('Loaded architecture:')
    for k, v in ARCH.items():
        print(f'  {k}: {v}')

    # اختبار DataLoader
    print('\n' + '=' * 60)
    print('Testing DataLoader...')
    print('=' * 60)
    tr, va = build_loaders(data_dir, batch_size=2, scale=4, patch_size=96)
    print(f'✅ Train batches: {len(tr)}')
    print(f'✅ Val   batches: {len(va)}')
    del tr, va
    torch.cuda.empty_cache()

    # فحص EdgeLoss
    print('\n' + '=' * 60)
    print('Testing Loss...')
    print('=' * 60)
    test_loss = Loss(make_loss_args("1*L1+0.05*VGG22+0.005*FFT+0.02*EDGE"), ckp=None)
    print('✅ Loss built with L1+VGG22+FFT+EDGE')
    del test_loss
    torch.cuda.empty_cache()

    # بدء الدراسة
    os.makedirs(out_dir, exist_ok=True)
    storage = f'sqlite:///{os.path.join(out_dir, "optuna_loss.db")}'

    study = optuna.create_study(
        study_name='dtcf_loss',
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=2
        ),
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=5
        ),
    )

    def _cleanup(study, trial):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(RuntimeError,),
        callbacks=[_cleanup],
        gc_after_trial=True,
    )

    # النتائج
    print('=' * 60)
    print(f'BEST LOSS WEIGHTS: {study.best_value:.4f} dB')
    best = study.best_params
    for k, v in best.items():
        print(f'  {k}: {v:.6e}')
    loss_string = (f"1*L1+{best['lam_vgg']}*VGG22"
                   f"+{best['lam_fft']}*FFT+{best['lam_edge']}*EDGE")
    print(f'\nFinal loss string:')
    print(f'  --loss "{loss_string}"')
    print('=' * 60)

    # حفظ
    study.trials_dataframe().to_csv(
        os.path.join(out_dir, 'optuna_loss_results.csv'), index=False
    )
    best['loss_string'] = loss_string
    best['best_psnr'] = study.best_value
    with open(os.path.join(out_dir, 'best_loss.json'), 'w') as f:
        json.dump(best, f, indent=2)

    print(f'\nSaved:')
    print(f'  DB   : {storage}')
    print(f'  CSV  : {os.path.join(out_dir, "optuna_loss_results.csv")}')
    print(f'  JSON : {os.path.join(out_dir, "best_loss.json")}')
    return study


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--arch_params', type=str, required=True,
                   help='مسار best_arch_fine.json')
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='./optuna_loss_out')
    args = p.parse_args()

    run(args.data_dir, args.arch_params, args.n_trials,
        args.num_epochs, args.out_dir,
        batch_size=args.batch_size, warmup_epochs=args.warmup_epochs)