# ===================================================================
# hyperopt_with_your_data.py — بحث شامل مع WaveletAttention و HUTCN
# متوافق مع ملفات dhtcu_block.py, dhtcun.py, custom_attention_blocks.py
# ===================================================================
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import os
import sys
import copy
import json    ####################
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import copy
import utility
import data
import model as model_module
import loss as loss_module
from option import args as base_args
from model.dhtcun import HUTCN
from model import dhtcu_block as B

# ═══════════════════════════════════════════════════════════
# 🆕 إضافة هذه الأسطر (لا تحذف شيئاً)
# ═══════════════════════════════════════════════════════════
N_TRIALS = 50
EPOCHS_PER_TRIAL = 50
N_STARTUP_TRIALS = 10
N_WARMUP_STEPS = 6
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
# دالة الهدف الرئيسية
# ===================================================================
def objective(trial):
    # ---------- معاملات بنية النموذج ----------
    nf = trial.suggest_int('n_feats', 64, 128, step=8)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
    # تحقق من قابلية القسمة (يجب أن يقبل nf القسمة على num_heads)
    if nf % num_heads != 0:
        raise optuna.TrialPruned()

    window_size = trial.suggest_categorical('window_size', [8, 12, 16])
    num_blocks = trial.suggest_int('num_blocks', 2, 4, step=1)

    patch_size = trial.suggest_categorical('patch_size', [128, 160, 192, 224])
    if window_size > patch_size:
        raise optuna.TrialPruned()

    # ---------- معاملات التدريب ----------
    optimizer_name = trial.suggest_categorical('optimizer', ['ADAM', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler', ['fixed', 'cosine', 'step'])
    loss_name = trial.suggest_categorical('loss', ['L1', 'L2', 'Charbonnier', 'Huber'])
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ---------- بناء النموذج ----------
    model = HUTCN(
        in_nc=3,
        nf=nf,
        num_modules=1,          # نستخدم كتلة واحدة للسرعة
        out_nc=3,
        upscale=4,
        num_heads=num_heads,
        window_size=window_size,
        num_blocks=num_blocks
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------- المُحسّن ----------
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    # ---------- جدول توهين معدل التعلم ----------
    EPOCHS = EPOCHS_PER_TRIAL  # عدد قليل للتجربة السريعة
    if scheduler_name == 'fixed':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    else:  # step
        scheduler = lrs.StepLR(optimizer, step_size=3, gamma=0.5)

    # ---------- دالة الخسارة ----------
    if loss_name == 'L1':
        criterion = nn.L1Loss()
    elif loss_name == 'L2':
        criterion = nn.MSELoss()
    elif loss_name == 'Charbonnier':
        criterion = CharbonnierLoss(eps=1e-3)
    else:  # Huber
        criterion = HuberLoss(delta=0.01)

    # ---------- تحميل البيانات ----------
    trial_params = {
        'patch_size': patch_size,
        'batch_size': 8  # يمكن جعله متغيراً أيضاً
    }
    train_loader, test_loaders = get_loaders_from_args(trial_params, base_args)
    val_loader = test_loaders[0] if test_loaders else None
    if val_loader is None:
        raise optuna.TrialPruned()

    # ---------- حلقة التدريب ----------
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
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        if scheduler is not None:
            scheduler.step()

        # ---------- التقييم ----------
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
# تشغيل البحث
# ===================================================================
if __name__ == "__main__":
    
    # 1️⃣ إنشاء أو تحميل الدراسة
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

    # 2️⃣ حساب الإحصاءات الحالية (بغض النظر عن الحالة)
    complete_trials = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials   = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials   = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.FAIL]
    running_trials  = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.RUNNING]

    total_so_far = len(study.trials)
    remaining = max(0, N_TRIALS - total_so_far)

    # 3️⃣ عرض ملخص واضح
    print("=" * 70)
    print(f"📚 اسم الدراسة: {STUDY_NAME}")
    print(f"🎯 الهدف: {N_TRIALS} trial إجمالي (COMPLETE + PRUNED + FAIL)")
    print("=" * 70)
    print(f"📊 الحالة الحالية في قاعدة البيانات:")
    print(f"   ✅ COMPLETE : {len(complete_trials)}")
    print(f"   ✂️ PRUNED   : {len(pruned_trials)}")
    print(f"   ❌ FAILED   : {len(failed_trials)}")
    print(f"   🔄 RUNNING  : {len(running_trials)}")
    print(f"   ─────────────────────")
    print(f"   📦 الإجمالي : {total_so_far} / {N_TRIALS}")
    print(f"   🎯 المتبقي  : {remaining}")
    if len(complete_trials) > 0:
        print(f"   🏆 أفضل PSNR: {study.best_value:.4f} dB "
              f"(Trial #{study.best_trial.number})")
    print("=" * 70)

    # 4️⃣ تشغيل / استئناف البحث
    if remaining <= 0:
        print("\n✅ تم إكمال جميع المحاولات المطلوبة.")
        print("   💡 لبدء بحث جديد: احذف optuna_study.db أو غيّر STUDY_NAME")
    else:
        print(f"\n🚀 بدء/استئناف البحث ({remaining} trial متبقٍ)...\n")
        try:
            study.optimize(
                objective,
                n_trials=remaining,
                show_progress_bar=True,
                catch=(RuntimeError,)
            )
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("⚠️ تم الإيقاف يدوياً (Ctrl+C).")
            print("💾 جميع النتائج محفوظة في قاعدة البيانات.")
            print("🔄 لتكملة البحث: أعد تشغيل نفس الأمر.")
            print("=" * 70)

    # 5️⃣ حفظ النتائج النهائية
    if len(complete_trials) > 0 or len(study.trials) > 0:
        complete_now = [t for t in study.trials 
                        if t.state == optuna.trial.TrialState.COMPLETE]
        
        print("\n" + "=" * 70)
        print("🏆 أفضل المعاملات النهائية:")
        print("=" * 70)
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key:>25} : {value:.6e}")
            else:
                print(f"  {key:>25} : {value}")
        print(f"\n📈 أفضل PSNR: {study.best_value:.4f} dB")
        print("=" * 70)
        
        # الإحصاءات النهائية
        print(f"\n📊 الإحصاءات النهائية:")
        print(f"   إجمالي trials : {len(study.trials)}")
        print(f"   ✅ COMPLETE   : {len(complete_now)}")
        print(f"   ✂️ PRUNED     : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"   ❌ FAILED     : {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        # حفظ JSON
        with open('best_params_optimized.json', 'w', encoding='utf-8') as f:
            json.dump(study.best_params, f, indent=4, ensure_ascii=False)
        print(f"\n✅ تم حفظ أفضل المعاملات في best_params_optimized.json")
        
        # حفظ CSV
        try:
            study.trials_dataframe().to_csv(
                'optuna_all_trials.csv', 
                index=False, 
                encoding='utf-8-sig'
            )
            print(f"✅ تم حفظ جميع المحاولات في optuna_all_trials.csv")
        except Exception as e:
            print(f"⚠️ تعذّر حفظ CSV: {e}")