# ===================================================================
# hyperopt_with_your_data_general.py
# البحث عن: num_cnn_stages + act_fn + (بقية المعاملات)
# الهدف: تحسين PSNR
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import data
import model as model_module
import loss as loss_module
from option import args as base_args
from model.dhtcun import HUTCN
from model import dhtcu_block as B

# ═══════════════════════════════════════════════════════════════════
# 🎯 إعدادات البحث
# ═══════════════════════════════════════════════════════════════════
N_TRIALS = 30
EPOCHS_PER_TRIAL = 30
N_STARTUP_TRIALS = 10
N_WARMUP_STEPS = 6

STUDY_NAME = "hutcn_search_cnn_act_fn"   # ← اسم جديد للدراسة
STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "optuna_study_cnn_act.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

DATA_DIR = r'D:\Mohamed Morsi\DATA'
SCALE = 2   # ← يتوافق مع DAT_bestx2


# ===================================================================
# دوال الخسارة الإضافية
# ===================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


class HuberLoss(nn.Module):
    def __init__(self, delta=0.01):
        super(HuberLoss, self).__init__()
        self.delta = delta
    def forward(self, x, y):
        diff = torch.abs(x - y)
        mask = (diff < self.delta).float()
        return torch.mean(mask * (x - y) ** 2 +
                          (1 - mask) * (2 * self.delta * diff - self.delta ** 2))


# ===================================================================
# تحرير الذاكرة
# ===================================================================
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===================================================================
# تحميل البيانات
# ===================================================================
def get_loaders_from_args(trial_params, base_args):
    args = copy.deepcopy(base_args)
    args.data_train = ['DIV2K']
    args.data_test = ['DIV2K']
    args.data_range = '1-800/896-900'
    args.scale = [SCALE]
    args.dir_data = DATA_DIR
    args.patch_size = trial_params['patch_size']
    args.batch_size = trial_params['batch_size']
    args.n_threads = 0   # ✅ منع multiprocessing

    loader = data.Data(args)
    return loader.loader_train, loader.loader_test


# ===================================================================
# دالة الهدف
# ===================================================================
def objective(trial):
    # ---------- معاملات النموذج ----------
    nf = trial.suggest_int('nf', 32, 128, step=8)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    if nf % num_heads != 0:
        raise optuna.TrialPruned()

    ws = trial.suggest_categorical('ws', [4, 6, 8, 12, 16])
    num_blocks = trial.suggest_int('num_blocks', 1, 4, step=1)

    # ✅ المعاملات الجديدة — محور البحث الأساسي
    num_cnn_stages = trial.suggest_int('num_cnn_stages', 1, 6, step=1)
    act_fn = trial.suggest_categorical(
        'act_fn', ['relu', 'lrelu', 'prelu', 'gelu', 'silu', 'tanh']
    )

    # ---------- معاملات التدريب ----------
    patch_size = trial.suggest_categorical('patch_size', [64, 96, 128, 160, 192])
    if ws > patch_size:
        raise optuna.TrialPruned()

    # batch_size ديناميكي
    if patch_size >= 160:
        safe_batch = 4
    elif patch_size >= 128:
        safe_batch = 6
    else:
        safe_batch = 8

    optimizer_name = trial.suggest_categorical('optimizer', ['ADAM', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler',
                                               ['fixed', 'cosine', 'step'])
    loss_name = trial.suggest_categorical('loss',
                                          ['L1', 'L2', 'Charbonnier', 'Huber'])
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ---------- بناء النموذج ----------
    model = HUTCN(
        in_nc=3, nf=nf, num_modules=1, out_nc=3,
        upscale=SCALE,
        num_heads=num_heads,
        ws=ws,
        num_blocks=num_blocks,
        num_cnn_stages=num_cnn_stages,
        act_fn=act_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'   📊 [Trial {trial.number}] nf={nf}, heads={num_heads}, '
          f'ws={ws}, blocks={num_blocks}, cnn_stages={num_cnn_stages}, '
          f'act={act_fn}, params={n_params:,}')

    # ---------- المحسّن ----------
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay, betas=(0.9, 0.99))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay, betas=(0.9, 0.99))

    EPOCHS = EPOCHS_PER_TRIAL
    if scheduler_name == 'fixed':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    else:
        scheduler = lrs.StepLR(optimizer, step_size=3, gamma=0.5)

    # ---------- دالة الخسارة ----------
    if loss_name == 'L1':
        criterion = nn.L1Loss()
    elif loss_name == 'L2':
        criterion = nn.MSELoss()
    elif loss_name == 'Charbonnier':
        criterion = CharbonnierLoss(eps=1e-3)
    else:
        criterion = HuberLoss(delta=0.01)

    # ---------- تحميل البيانات ----------
    trial_params = {'patch_size': patch_size, 'batch_size': safe_batch}
    train_loader, test_loaders = get_loaders_from_args(trial_params, base_args)
    val_loader = test_loaders[0] if test_loaders else None
    if val_loader is None:
        raise optuna.TrialPruned()

    # ---------- التدريب والتقييم ----------
    best_psnr = 0.0
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for lr_batch, hr_batch, _ in train_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                optimizer.zero_grad()
                sr = model(lr_batch)
                loss = criterion(sr, hr_batch)
                loss.backward()
                optimizer.step()
                del lr_batch, hr_batch, sr, loss

            if scheduler is not None:
                scheduler.step()

            model.eval()
            psnr_sum = 0.0
            with torch.no_grad():
                for lr_batch, hr_batch, _ in val_loader:
                    lr_batch = lr_batch.to(device)
                    hr_batch = hr_batch.to(device)
                    sr = model(lr_batch)
                    mse = torch.mean((sr - hr_batch) ** 2)
                    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                    psnr_sum += psnr.item()
                    del lr_batch, hr_batch, sr

            avg_psnr = psnr_sum / len(val_loader)
            print(f'   📈 [Trial {trial.number}] Epoch {epoch}/{EPOCHS}: '
                  f'PSNR = {avg_psnr:.3f} dB')

            trial.report(avg_psnr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

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

    return best_psnr


# ===================================================================
# التشغيل
# ===================================================================
if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=N_STARTUP_TRIALS,
            n_warmup_steps=N_WARMUP_STEPS,
        ),
        load_if_exists=True
    )

    complete_trials = [t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.FAIL]

    valid_trials = [t for t in study.trials
                    if t.state in [optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.PRUNED]]
    remaining = max(0, N_TRIALS - len(valid_trials))

    print("=" * 70)
    print(f"📚 Study: {STUDY_NAME}")
    print(f"🎯 Target: {N_TRIALS} valid trials")
    print("=" * 70)
    print(f"✅ COMPLETE: {len(complete_trials)}")
    print(f"✂️ PRUNED  : {len(pruned_trials)}")
    print(f"❌ FAILED  : {len(failed_trials)}  (ignored)")
    print(f"🎯 Remaining: {remaining}")
    if len(complete_trials) > 0:
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

    # حفظ النتائج
    complete_now = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_now) > 0:
        print("\n" + "=" * 70)
        print("🏆 أفضل المعاملات:")
        print("=" * 70)
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key:>18} : {value:.6e}")
            else:
                print(f"  {key:>18} : {value}")
        print(f"\n📈 أفضل PSNR: {study.best_value:.4f} dB")
        print("=" * 70)

        output_json = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'best_params_cnn_act.json'
        )
        best_result = {
            'best_trial_number': study.best_trial.number,
            'best_psnr_db': study.best_value,
            'best_params': study.best_params,
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=4, ensure_ascii=False)
        print(f"\n✅ محفوظ في: {output_json}")

        # حفظ CSV
        try:
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'optuna_trials_cnn_act.csv'
            )
            study.trials_dataframe().to_csv(csv_path, index=False,
                                             encoding='utf-8-sig')
            print(f"✅ CSV: {csv_path}")
        except Exception as e:
            print(f"⚠️ CSV: {e}")