# ===================================================================
# hyperopt_hat_cnn_act.py — بحث سريع لموديل HUTCN + HAT
# المحاور المبحوثة: num_cnn_stages + act_fn
# كل شيء آخر ثابت من بحث HAT الشامل (best_params_optimized.json)
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
import gc
import time
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import data
import model as model_module
import loss as loss_module
from option import args as base_args
from model.dhtcun import HUTCN

# ═══════════════════════════════════════════════════════════════════
# 🎯 إعدادات البحث (غيّر هنا فقط)
# ═══════════════════════════════════════════════════════════════════
STAGES_CHOICES = [1, 2, 3, 4, 5, 6]
ACT_CHOICES = ['relu', 'lrelu', 'prelu', 'gelu', 'silu', 'tanh']   # يمكن إضافة 'mish'
SEARCH_SPACE = {
    'num_cnn_stages': STAGES_CHOICES,
    'act_fn': ACT_CHOICES,
}
N_TRIALS = len(STAGES_CHOICES) * len(ACT_CHOICES)   # حجم الشبكة (36)

# ── إعدادات السرعة ──
EPOCHS_PER_TRIAL = 20
MAX_ITERS_PER_EPOCH = 250    # يحدد طول الـ epoch مهما كان حجم الداتا
VAL_EVERY = 2                # تقييم كل epoch زوجي + آخر epoch
VAL_CROP_LR = 120            # مضاعف لـ window_size=12
SEARCH_PATCH_SIZE = 224      # HR → LR=36 = 3 × window_size (بلا padding)
SEARCH_BATCH = 8
USE_AMP = True               # يُعطَّل تلقائياً على CPU. إن ظهرت nan اجعله False
SEED = 42

STUDY_NAME = "hutcn_hat_search_cnn_act_fast_x4"
STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "optuna_study_hat_cnn_act_fast_x4.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

DATA_DIR = r'D:\Mohamed Morsi\DATA'
SCALE = 4

# ── القيم الثابتة من بحث HAT ──
FIXED_NF = 128
FIXED_HEADS = 4
FIXED_WINDOW = 12
FIXED_BLOCKS = 4
FIXED_LR = 0.00040181223544441785
FIXED_WD = 0.00018709724200887206
# optimizer = AdamW | scheduler = cosine | loss = L1


# ===================================================================
# أدوات مساعدة
# ===================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def center_crop_pair(lr, hr, scale, size):
    """يقص منطقة مركزية من LR وما يقابلها من HR."""
    _, _, h, w = lr.shape
    ch, cw = min(size, h), min(size, w)
    top, left = (h - ch) // 2, (w - cw) // 2
    lr_c = lr[:, :, top:top + ch, left:left + cw]
    hr_c = hr[:, :, top * scale:(top + ch) * scale,
              left * scale:(left + cw) * scale]
    return lr_c, hr_c


def evaluate(model, val_loader, device, use_amp):
    model.eval()
    psnr_sum, n = 0.0, 0
    with torch.no_grad():
        for lr_b, hr_b, _ in val_loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            lr_b, hr_b = center_crop_pair(lr_b, hr_b, SCALE, VAL_CROP_LR)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=use_amp):
                sr = model(lr_b)
            mse = torch.mean((sr.float() - hr_b) ** 2)
            psnr_sum += (10 * torch.log10(1.0 / (mse + 1e-8))).item()
            n += 1
    return psnr_sum / max(n, 1)


# ===================================================================
# تحميل البيانات
# ===================================================================
def get_loaders_from_args(patch_size, batch_size, base_args):
    args = copy.deepcopy(base_args)
    args.data_train = ['DIV2K']
    args.data_test = ['DIV2K']
    args.data_range = '1-800/896-900'
    args.scale = [SCALE]
    args.dir_data = DATA_DIR
    args.patch_size = patch_size
    args.batch_size = batch_size
    args.n_threads = 0
    loader = data.Data(args)
    return loader.loader_train, loader.loader_test


# ===================================================================
# دالة الهدف
# ===================================================================
def objective(trial):
    t0 = time.time()
    num_cnn_stages = trial.suggest_categorical('num_cnn_stages', STAGES_CHOICES)
    act_fn = trial.suggest_categorical('act_fn', ACT_CHOICES)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = USE_AMP and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True

    # نفس الـ seed لكل trial => نفس التهيئة ونفس ترتيب البيانات تقريباً
    set_seed(SEED)

    model = HUTCN(
        in_nc=3, nf=FIXED_NF, num_modules=1, out_nc=3,
        upscale=SCALE,
        num_heads=FIXED_HEADS,
        window_size=FIXED_WINDOW,      # ⚠️ في HAT الاسم window_size وليس ws
        num_blocks=FIXED_BLOCKS,
        num_cnn_stages=num_cnn_stages,
        act_fn=act_fn,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'   📊 [Trial {trial.number}] cnn_stages={num_cnn_stages}, '
          f'act={act_fn}, params={n_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=FIXED_LR,
                            weight_decay=FIXED_WD, betas=(0.9, 0.99))
    scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL,
                                      eta_min=1e-7)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_loader, test_loaders = get_loaders_from_args(
        SEARCH_PATCH_SIZE, SEARCH_BATCH, base_args)
    val_loader = test_loaders[0] if test_loaders else None
    if val_loader is None:
        raise optuna.TrialPruned()

    best_psnr = 0.0
    try:
        for epoch in range(1, EPOCHS_PER_TRIAL + 1):
            model.train()
            for it, (lr_b, hr_b, _) in enumerate(train_loader):
                if it >= MAX_ITERS_PER_EPOCH:
                    break
                lr_b = lr_b.to(device, non_blocking=True)
                hr_b = hr_b.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=use_amp):
                    sr = model(lr_b)
                loss = criterion(sr.float(), hr_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            if epoch % VAL_EVERY == 0 or epoch == EPOCHS_PER_TRIAL:
                avg_psnr = evaluate(model, val_loader, device, use_amp)
                print(f'   📈 [Trial {trial.number}] Epoch {epoch}/'
                      f'{EPOCHS_PER_TRIAL}: PSNR = {avg_psnr:.3f} dB')

                if not np.isfinite(avg_psnr):
                    print(f'   ⚠️ [Trial {trial.number}] PSNR غير صالح (nan/inf) '
                          f'→ جرّب USE_AMP = False')
                    raise optuna.TrialPruned()

                trial.report(avg_psnr, epoch)
                best_psnr = max(best_psnr, avg_psnr)

    except torch.cuda.OutOfMemoryError:
        print(f'   ⚠️ [Trial {trial.number}] CUDA OOM → pruned')
        raise optuna.TrialPruned()

    finally:
        try:
            del model, optimizer, train_loader, val_loader
        except Exception:
            pass
        cleanup_memory()

    print(f'   ⏱️ [Trial {trial.number}] الزمن: {(time.time() - t0) / 60:.1f} دقيقة')
    return best_psnr


# ===================================================================
# التشغيل
# ===================================================================
if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction='maximize',
        # GridSampler: لا تكرار للتركيبات
        sampler=optuna.samplers.GridSampler(SEARCH_SPACE, seed=SEED),
        # NopPruner: كل trial يُكمل لتبقى المقارنة عادلة
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True
    )

    TS = optuna.trial.TrialState
    complete_trials = [t for t in study.trials if t.state == TS.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TS.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TS.FAIL]
    remaining = max(0, N_TRIALS - len(complete_trials) - len(pruned_trials))

    print("=" * 70)
    print(f"📚 Study: {STUDY_NAME}")
    print(f"🎯 Grid size: {N_TRIALS} combinations")
    print("=" * 70)
    print(f"✅ COMPLETE: {len(complete_trials)}")
    print(f"✂️ PRUNED  : {len(pruned_trials)}")
    print(f"❌ FAILED  : {len(failed_trials)}  (ignored)")
    print(f"🎯 Remaining: {remaining}")
    if complete_trials:
        print(f"🏆 Best PSNR: {study.best_value:.4f} dB "
              f"(Trial #{study.best_trial.number})")
    print("=" * 70)

    if remaining <= 0:
        print("\n✅ البحث مكتمل.")
    else:
        print(f"\n🚀 بدء/استئناف البحث ({remaining} trial)...\n")
        try:
            study.optimize(
                objective,
                n_trials=remaining,
                show_progress_bar=True,
                catch=(RuntimeError, MemoryError),
            )
        except KeyboardInterrupt:
            print("\n⚠️ تم الإيقاف يدوياً. النتائج محفوظة.")

    complete_now = [t for t in study.trials if t.state == TS.COMPLETE]
    if complete_now:
        print("\n" + "=" * 70)
        print("🏆 أفضل المعاملات:")
        print("=" * 70)
        for key, value in study.best_params.items():
            print(f"  {key:>18} : {value}")
        print(f"\n📈 أفضل PSNR: {study.best_value:.4f} dB")

        print("\n🔝 أفضل 5 تركيبات:")
        for t in sorted(complete_now, key=lambda t: t.value, reverse=True)[:5]:
            print(f"   {t.value:.4f} dB | stages={t.params['num_cnn_stages']} "
                  f"| act={t.params['act_fn']}")
        print("=" * 70)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_json = os.path.join(base_dir, 'best_params_hat_cnn_act.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({
                'best_trial_number': study.best_trial.number,
                'best_psnr_db': study.best_value,
                'best_params': study.best_params,
            }, f, indent=4, ensure_ascii=False)
        print(f"\n✅ محفوظ في: {output_json}")

        try:
            csv_path = os.path.join(base_dir, 'optuna_trials_hat_cnn_act.csv')
            study.trials_dataframe().to_csv(csv_path, index=False,
                                            encoding='utf-8-sig')
            print(f"✅ CSV: {csv_path}")
        except Exception as e:
            print(f"⚠️ CSV: {e}")