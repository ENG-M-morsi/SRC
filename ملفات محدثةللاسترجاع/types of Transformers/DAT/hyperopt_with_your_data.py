# ===================================================================
# hyperopt_with_your_data.py — بحث شامل متوافق مع نظام Data الحالي
# يستخدم نفس بنية البيانات الموجودة في main.py (DIV2K, Benchmark, إلخ)
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

# إضافة المسار الحالي للـ sys.path لضمان استيراد الوحدات الخاصة بك
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# استيراد الوحدات الخاصة بك
import utility
import data  # هذا هو الـ __init__.py الذي يحتوي على class Data
import model as model_module
import loss as loss_module
from option import args as base_args  # استخدم args الأصلي كمرجع
from model.dhtcun import HUTCN
from model import dhtcu_block as B
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
    """
    تنشئ DataLoaders باستخدام class Data الموجود في __init__.py.
    نمرر لها المعاملات الجديدة (patch_size, batch_size) ونضبط args مؤقتاً.
    """
    # نسخ args الأصلي لتعديله دون التأثير على باقي الكود
    args = base_args
    # تعديل الباراميترات التي تختلف حسب المحاولة
    args.patch_size = trial_params['patch_size']
    args.batch_size = trial_params['batch_size'] if 'batch_size' in trial_params else base_args.batch_size
    # نضمن أن data_train, data_test, data_range كما هي (نأخذها من base_args)
    # ولكن يمكنك جعلها قابلة للتخصيص إذا أردت.
    
    # إنشاء كائن Data (سيقوم بتحميل DIV2K ومجموعات الاختبار)
    loader = data.Data(args)
    return loader.loader_train, loader.loader_test

# ===================================================================
# 3. دالة الهدف الرئيسية (Objective) — متوافقة مع نظامك
# ===================================================================
def objective(trial):
# ---------- (أ) معاملات بنية النموذج ----------
    # 1. عدد القنوات (n_feats)
    nf = trial.suggest_int('n_feats', 64, 128, step=8)

    # 2. عدد الرؤوس (num_heads) — قائمة ثابتة
    num_heads_options = [2, 4, 8, 16]
    num_heads = trial.suggest_categorical('num_heads', num_heads_options)

    # تحقق من قابلية القسمة (إذا لم يكن nf يقبل القسمة على num_heads، تخطى هذه المحاولة)
    if nf % num_heads != 0:
        raise optuna.TrialPruned()

    # 3. عدد طبقات DAT (num_blocks) — زوجي
    num_blocks = trial.suggest_int('num_blocks', 2, 6, step=1)

    # 4. حجم النافذة المكانية (window_size) — يجب أن يكون <= patch_size
    window_size = trial.suggest_categorical('window_size', [8, 12, 16])

    # 5. نسبة التوسع في SGFN (mlp_ratio)
    mlp_ratio = trial.suggest_float('mlp_ratio', 1.5, 2.5, step=0.25)

    # 6. عدد كتل P_HTCB (num_modules) — في HUTCN يوجد 4 افتراضياً
    #num_modules = trial.suggest_int('num_modules', 2, 6, step=2)
    num_modules = 1

    # 7. معدل التسرب (drop_path)
    drop_path = trial.suggest_float('drop_path', 0.0, 0.2, step=0.05)

    # ---------- (ب) الباراميتر الجديد: patch_size ----------
    # نطاق patch_size مناسب لـ DIV2K (حجم الصورة 192 في التدريب الأصلي)
    patch_size = trial.suggest_categorical('patch_size', [128, 160, 192, 224, 256])

    # شرط: window_size لا يتجاوز patch_size
    if window_size > patch_size:
        raise optuna.TrialPruned()

    # ---------- (ج) معاملات التدريب ----------
    optimizer_name = trial.suggest_categorical('optimizer', ['ADAM', 'AdamW'])
    scheduler_name = trial.suggest_categorical('scheduler', ['fixed', 'cosine', 'step'])
    loss_name = trial.suggest_categorical('loss', ['L1', 'L2', 'Charbonnier', 'Huber'])
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ---------- (د) بناء النموذج ----------
    # نستخدم نفس دالة بناء النموذج من ملفك (HUTCN)
    model = HUTCN(in_nc=3, nf=nf, num_modules=num_modules, out_nc=3, upscale=4, num_heads=num_heads)  # upscale=4 حسب أمرك

    # تطبيق المعاملات الجديدة على كتل DAT الداخلية (لأنها ليست parameters مباشرة في HUTCN)
    for module in model.modules():
        if isinstance(module, B.TCN):
            # نعيد تعريف الـ DAT بالمعاملات الجديدة
            module.SwinT = B.SwinT.SwinT(n_feats=nf, num_heads=num_heads, depth=num_blocks, window_size=window_size, mlp_ratio=mlp_ratio, resolution=48)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------- (هـ) اختيار المُحسّن ----------
    if optimizer_name == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    # ---------- (و) اختيار جدول توهين معدل التعلم ----------
    # 🔴 [النقطة رقم 2] — عدد Epochs لكل محاولة (غيّر هذا الرقم حسب رغبتك)
    EPOCHS = 10  # اجعلها 5-10 للتجربة السريعة، و20-30 للبحث العميق
    if scheduler_name == 'fixed':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    else:  # step
        scheduler = lrs.StepLR(optimizer, step_size=3, gamma=0.5)

    # ---------- (ز) اختيار دالة الخسارة ----------
    if loss_name == 'L1':
        criterion = nn.L1Loss()
    elif loss_name == 'L2':
        criterion = nn.MSELoss()
    elif loss_name == 'Charbonnier':
        criterion = CharbonnierLoss(eps=1e-3)
    else:  # Huber
        criterion = HuberLoss(delta=0.01)

    # ---------- (ح) تحميل البيانات باستخدام نظامك ----------
    # نمرر المعاملات المختارة (patch_size, batch_size) إلى دالة get_loaders_from_args
    trial_params = {
        'patch_size': patch_size,
        'batch_size': 8  # يمكنك جعله قابلاً للتحسين أيضاً، لكنني ثبته لتقليل التعقيد
    }
    # نستخدم base_args من option.py، لكننا سنعدل patch_size و batch_size مؤقتاً
    # للحصول على DataLoaders صحيحة
    train_loader, test_loaders = get_loaders_from_args(trial_params, base_args)

    # نأخذ أول مجموعة اختبار (DIV2K) للتقييم السريع
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
            sr = model(lr)  # HUTCN يستقبل lr فقط (بدون idx_scale)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            # اختياري: طباعة التقدم بين الحين والآخر
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # تحديث معدل التعلم
        if scheduler is not None:
            scheduler.step()

        # ---------- التقييم (حساب PSNR) ----------
        model.eval()
        psnr_sum = 0.0
        with torch.no_grad():
            for lr, hr, _ in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                # حساب PSNR (نطاق [0,1] لأن rgb_range=1)
                mse = torch.mean((sr - hr) ** 2)
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                psnr_sum += psnr.item()

        avg_psnr = psnr_sum / len(val_loader)

        # إبلاغ Optuna (للتقليم المبكر)
        trial.report(avg_psnr, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # تحديث أفضل PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

    return best_psnr

# ===================================================================
# 4. تشغيل البحث الرئيسي
# ===================================================================
if __name__ == "__main__":
    # 🔴 [النقطة رقم 1] — هنا تُحدد عدد المحاولات (Trials)
    # غيّر الرقم 20 إلى 10 (للتجربة السريعة) أو 50 (للبحث العميق)
    N_TRIALS = 20

    # إنشاء دراسة Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    print(f"🚀 بدء البحث الشامل ({N_TRIALS} محاولة، كل محاولة {10} Epochs)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # عرض النتائج
    print("\n" + "="*70)
    print("🏆 أفضل المعاملات:")
    print("="*70)
    for key, value in study.best_params.items():
        print(f"  {key:>20} : {value}")
    print(f"\n📈 أفضل PSNR على مجموعة التحقق: {study.best_value:.3f} dB")
    print("="*70)

    # حفظ النتائج
    import json
    with open('best_params_optimized.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ تم حفظ أفضل المعاملات في best_params_optimized.json")