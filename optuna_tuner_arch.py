# ===================================================================
# optuna_tuner_arch.py — نسخة مُصحَّحة نهائية
#
# الإصلاحات:
#   1. catch=(RuntimeError,) → لا تتوقف الدراسة عند OOM
#   2. حذف ws_elan=16 (يسبب OOM في ELAN/GMSA)
#   3. حذف batch_size=8 (يضاعف الذاكرة)
#   4. cleanup callback → تنظيف الذاكرة بين trials
#   5. حماية مسبقة من التكوينات المعروفة بفشلها
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dhtcun import HUTCN


# ═════════════════════════════════════════════════════════════════
# بناء args المطلوبة لمشروعك
# ═════════════════════════════════════════════════════════════════
def make_data_args(data_dir, batch_size, scale=4, patch_size=96,
                   data_train=('DIV2K',), data_test=('DIV2K',),
                   data_range='1-200/896-900'):
    return SimpleNamespace(
        dir_data=data_dir,
        data_train=list(data_train),
        data_test=list(data_test),
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
    args = make_data_args(
        data_dir=data_dir, batch_size=batch_size,
        scale=scale, patch_size=patch_size,
        data_train=('DIV2K',), data_test=('DIV2K',),
        data_range='1-800/896-900',
    )
    d = Data(args)
    if d.loader_train is None:
        raise RuntimeError('loader_train == None')
    if len(d.loader_test) == 0:
        raise RuntimeError('لا يوجد loader_test')
    return d.loader_train, d.loader_test[0]


# ═════════════════════════════════════════════════════════════════
# أدوات مساعدة
# ═════════════════════════════════════════════════════════════════
def calc_psnr(sr, hr, shave=4):
    sr = sr[:, :, shave:-shave, shave:-shave]
    hr = hr[:, :, shave:-shave, shave:-shave]
    mse = torch.mean((sr - hr) ** 2).item()
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def _quick_check(nf, batch_size, ws_elan, ws_dat):
    """يتحقق مسبقاً إذا كان التكوين سينفجر في الذاكرة."""
    # ws_elan=16 → GMSA يستخدم نافذة 32×32 → انفجار
    if ws_elan >= 16:
        return False
    # ws_dat=16 → نافذة كبيرة في DAT
    if ws_dat >= 16 and batch_size >= 4:
        return False
    # nf كبير + batch كبير
    if nf >= 96 and batch_size >= 4:
        return False
    return True


# ═════════════════════════════════════════════════════════════════
# Objective
# ═════════════════════════════════════════════════════════════════
def objective(trial):
    # ── (أ) اقتراح المعمارية ──
    nf              = trial.suggest_categorical('nf', [64, 80, 88, 96])
    num_heads_dat   = trial.suggest_categorical('num_heads_dat', [2, 4, 8])
    ws_dat          = trial.suggest_categorical('ws_dat', [4, 8, 12])       # ← حذف 16
    num_blocks_dat  = trial.suggest_int('num_blocks_dat', 1, 3)
    num_heads_elan  = trial.suggest_categorical('num_heads_elan', [2, 4, 8])
    ws_elan         = trial.suggest_categorical('ws_elan', [4, 8, 12])      # ← حذف 16
    num_blocks_elan = trial.suggest_int('num_blocks_elan', 1, 3)
    fusion_heads    = trial.suggest_categorical('fusion_heads', [2, 4, 8])
    fusion_dropout  = trial.suggest_float('fusion_dropout', 0.0, 0.3, step=0.1)
    hfe_reduction   = trial.suggest_categorical('hfe_reduction', [2, 4, 8])

    # ── (ب) اقتراح التدريب ──
    lr           = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [2, 4, 8])          # ← حذف 8
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    # ── فحص مسبق ──
    if not _quick_check(nf, batch_size, ws_elan, ws_dat):
        print(f'[Trial {trial.number}] Pruned by quick_check (potential OOM)')
        raise optuna.TrialPruned()

    # ── توافق القسمة ──
    for h in (num_heads_dat, num_heads_elan, fusion_heads):
        if nf % h != 0:
            raise optuna.TrialPruned()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── (ج) بناء النموذج ──
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
        print(f'[Trial {trial.number}] Model build failed: {e}')
        raise optuna.TrialPruned()

    # ── (د) DataLoaders ──
    try:
        train_loader, val_loader = build_loaders(
            DATA_DIR, batch_size=batch_size, scale=4, patch_size=96
        )
    except Exception as e:
        print(f'[Trial {trial.number}] DataLoader failed: {e}')
        raise optuna.TrialPruned()

    # ── (هـ) Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=lr * 0.01
    )
    criterion = nn.L1Loss()

    # ── (و) التدريب ──
    best_psnr = 0.0
    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch in tqdm(train_loader,
                              desc=f'Trial {trial.number} Ep {epoch}',
                              leave=False):
                lr_img, hr_img = batch[0].to(device), batch[1].to(device)
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
                for batch in val_loader:
                    lr_img, hr_img = batch[0].to(device), batch[1].to(device)
                    sr = model(lr_img)
                    psnr_sum += calc_psnr(sr, hr_img)
            avg_psnr = psnr_sum / max(len(val_loader), 1)
            best_psnr = max(best_psnr, avg_psnr)

            trial.report(avg_psnr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    except torch.cuda.OutOfMemoryError:
        # تنظيف الذاكرة وإرجاع None (بدل رفع استثناء يوقف الدراسة)
        print(f'[Trial {trial.number}] OOM أثناء التدريب — سيُتجاهل')
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        # نُبلّغ Optuna بالفشل بهدوء
        raise optuna.TrialPruned()

    # تنظيف نهائي بين الـ trials
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return best_psnr


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════
def run(data_dir, n_trials, num_epochs, out_dir):
    global DATA_DIR, NUM_EPOCHS
    DATA_DIR   = data_dir
    NUM_EPOCHS = num_epochs

    # ── اختبار DataLoader ──
    print('=' * 60)
    print('Testing DataLoader...')
    print('=' * 60)
    try:
        tr, va = build_loaders(data_dir, batch_size=2, scale=4, patch_size=96)
        print(f'✅ Train batches: {len(tr)}')
        print(f'✅ Val   batches: {len(va)}')
        for b in tr:
            print(f'   Sample LR shape: {b[0].shape}, HR shape: {b[1].shape}')
            break
        del tr, va
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f'❌ DataLoader failed:')
        traceback.print_exc()
        sys.exit(1)

    # ── بدء الدراسة ──
    os.makedirs(out_dir, exist_ok=True)
    storage = f'sqlite:///{os.path.join(out_dir, "optuna_arch.db")}'

    study = optuna.create_study(
        study_name='dtcf_arch',
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
    )

    # ── cleanup callback ──
    def _cleanup(study, trial):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # ── تشغيل الدراسة مع catch ──
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(RuntimeError,),           # ← لا تتوقف عند OOM
        callbacks=[_cleanup],
        gc_after_trial=True,
    )

    # ── حفظ النتائج ──
    print('=' * 60)
    print('BEST ARCHITECTURE:')
    if len(study.trials) > 0 and study.best_trial is not None:
        try:
            print(f'  PSNR: {study.best_value:.4f} dB')
            for k, v in study.best_params.items():
                print(f'  {k}: {v}')
        except ValueError:
            print('  ⚠️  لا يوجد trial ناجح حتى الآن')
    else:
        print('  ⚠️  لا يوجد trials')
    print('=' * 60)

    study.trials_dataframe().to_csv(
        os.path.join(out_dir, 'optuna_arch_results.csv'), index=False
    )

    best_json_path = os.path.join(out_dir, 'best_arch.json')
    try:
        with open(best_json_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f'\n  JSON : {best_json_path}')
    except ValueError:
        print('  ⚠️  لا يوجد best_params — كل التجارب فشلت')

    print(f'  DB   : {storage}')
    print(f'  CSV  : {os.path.join(out_dir, "optuna_arch_results.csv")}')
    return study


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--n_trials', type=int, default=50)
    p.add_argument('--num_epochs', type=int, default=5)   # ← 5 بدل 10
    p.add_argument('--out_dir', type=str, default='./optuna_arch_out')
    args = p.parse_args()

    run(args.data_dir, args.n_trials, args.num_epochs, args.out_dir)