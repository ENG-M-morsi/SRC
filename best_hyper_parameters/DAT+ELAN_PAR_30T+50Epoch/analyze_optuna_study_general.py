# ===================================================================
# analyze_optuna_study_general.py — تحليل شامل وعام لأي دراسة Optuna
# 
# الميزات:
#   1. عام 100% — يعمل مع أي موديل دون تعديل
#   2. تحليل trials المقتطعة بشكل منفصل
#   3. اختبارات إحصائية (Wilcoxon, Mann-Whitney U, Bootstrap CI)
# ===================================================================
import optuna
import optuna.visualization.matplotlib as optuna_vis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ===================================================================
# 🟠 [خاص بالموديل] — الإعدادات الأساسية
#    عدّل هذين المتغيرين فقط عند تغيير الموديل
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(SCRIPT_DIR, "optuna_study.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

# 🔵 اكتشاف تلقائي للدراسة
STUDY_NAME = None   # ← اتركه None للاكتشاف التلقائي
                    #    أو ضع الاسم يدوياً مثل: "hutcn_hyperopt_separate_v1"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "optuna_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================================================================
# 🔵 [عام] — دوال مساعدة
# ===================================================================
def save_figure(fig, name):
    """حفظ الرسم بصيغتي PDF و PNG."""
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ {name}.pdf + {name}.png")
    plt.close(fig)


def detect_parameters(df, exclude_cols=None):
    """
    🔵 [عام] يكتشف كل المعاملات تلقائياً من DataFrame
    ويعيد قائمتين: رقمية + تصنيفية
    """
    if exclude_cols is None:
        exclude_cols = ['number', 'value', 'state', 'datetime_start',
                        'datetime_complete', 'duration', 'system_attrs',
                        'user_attrs']
    
    numeric_params = []
    categorical_params = []
    
    for col in df.columns:
        if not col.startswith('params_'):
            continue
        if col in exclude_cols:
            continue
        
        # التحقق من نوع البيانات
        dtype = df[col].dtype
        if dtype in ['int64', 'int32', 'float64', 'float32']:
            # التحقق من عدد القيم الفريدة
            if df[col].nunique() <= 10:
                categorical_params.append(col)
            else:
                numeric_params.append(col)
        else:
            categorical_params.append(col)
    
    return numeric_params, categorical_params


def confidence_interval(data, confidence=0.95, n_bootstrap=10000):
    """
    🔵 [عام] حساب فاصل الثقة بطريقة Bootstrap
    """
    data = np.asarray(data)
    if len(data) < 2:
        return (np.nan, np.nan)
    
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return lower, upper

def fix_optuna_plot_size(plot_result, size=(15, 10)):
    """
    🔵 [عام] إصلاح حجم الرسوم من optuna.visualization.matplotlib
    تتعامل مع الحالات المختلفة: Axes, ndarray of Axes, Figure
    """
    fig = None
    
    # حالة 1: numpy array من Axes
    if isinstance(plot_result, np.ndarray):
        if plot_result.size > 0:
            fig = plot_result.flatten()[0].get_figure()
    
    # حالة 2: Axes واحد
    elif hasattr(plot_result, 'get_figure'):
        fig = plot_result.get_figure()
    
    # حالة 3: Axes بإصدارات أقدم
    elif hasattr(plot_result, 'figure'):
        fig = plot_result.figure
    
    # حالة 4: Figure جاهز
    elif hasattr(plot_result, 'set_size_inches'):
        fig = plot_result
    
    # ضبط الحجم
    if fig is not None:
        fig.set_size_inches(*size)
    
    return fig
# ===================================================================
# 🔵 [عام] — تحميل الدراسة (مع اكتشاف تلقائي للاسم)
# ===================================================================
print("=" * 75)
print("📊 تحليل دراسة Optuna (نسخة عامة)")
print("=" * 75)

if not os.path.exists(STORAGE_PATH):
    print(f"❌ لم يتم العثور على قاعدة البيانات: {STORAGE_PATH}")
    sys.exit(1)

# ✅ اكتشاف اسم الدراسة تلقائياً
if STUDY_NAME is None:
    try:
        summaries = optuna.study.get_all_study_summaries(storage=STORAGE_URL)
    except Exception as e:
        print(f"❌ فشل قراءة قاعدة البيانات: {e}")
        sys.exit(1)

    if len(summaries) == 0:
        print("❌ لا توجد دراسات في قاعدة البيانات!")
        sys.exit(1)
    elif len(summaries) == 1:
        STUDY_NAME = summaries[0].study_name
        print(f"✅ اكتُشفت دراسة واحدة: '{STUDY_NAME}'")
    else:
        print(f"⚠️ توجد {len(summaries)} دراسات في قاعدة البيانات:")
        for i, s in enumerate(summaries, 1):
            n_complete = sum(1 for t in optuna.load_study(
                study_name=s.study_name, storage=STORAGE_URL).trials
                if t.state == optuna.trial.TrialState.COMPLETE)
            print(f"   [{i}] '{s.study_name}' — {n_complete} trials مكتملة")
        print()
        choice = input("👉 أدخل رقم الدراسة المطلوبة: ").strip()
        try:
            idx = int(choice) - 1
            STUDY_NAME = summaries[idx].study_name
            print(f"✅ اخترت: '{STUDY_NAME}'")
        except (ValueError, IndexError):
            print("❌ اختيار غير صحيح.")
            sys.exit(1)

# ✅ تحميل الدراسة
try:
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
except KeyError:
    print(f"❌ الدراسة '{STUDY_NAME}' غير موجودة في قاعدة البيانات.")
    print("💡 الدراسات المتوفرة:")
    for s in optuna.study.get_all_study_summaries(storage=STORAGE_URL):
        print(f"   - {s.study_name}")
    sys.exit(1)
# استخراج كل trials
all_trials = study.trials
complete_trials = [t for t in all_trials 
                   if t.state == optuna.trial.TrialState.COMPLETE]
pruned_trials = [t for t in all_trials 
                 if t.state == optuna.trial.TrialState.PRUNED]
failed_trials = [t for t in all_trials 
                 if t.state == optuna.trial.TrialState.FAIL]

print(f"\n📚 اسم الدراسة: {STUDY_NAME}")
print(f"📊 إجمالي trials: {len(all_trials)}")
print(f"✅ مكتملة (COMPLETE): {len(complete_trials)}")
print(f"✂️ مقتطعة (PRUNED):   {len(pruned_trials)}")
print(f"❌ فاشلة (FAILED):    {len(failed_trials)}")

if len(complete_trials) < 2:
    print("⚠️ عدد المحاولات المكتملة قليل جداً للتحليل.")
    sys.exit(1)

print(f"\n🏆 أفضل PSNR: {study.best_value:.4f} dB (Trial #{study.best_trial.number})")

# نسبة الاقتطاع
pruning_rate = len(pruned_trials) / max(len(all_trials), 1) * 100
print(f"📉 نسبة الاقتطاع: {pruning_rate:.1f}%")
if pruning_rate > 30:
    print("⚠️ تحذير: نسبة الاقتطاع عالية (> 30%). قد يؤثر على التحليل.")
elif pruning_rate > 15:
    print("ℹ️ ملاحظة: نسبة اقتطاع متوسطة (15-30%). التحليل مقبول.")
else:
    print("✅ نسبة اقتطاع منخفضة (< 15%). التحليل موثوق.")

# تحويل إلى DataFrame
df = study.trials_dataframe()
df_complete = df[df['state'] == 'COMPLETE'].copy()
df_pruned = df[df['state'] == 'PRUNED'].copy() if len(pruned_trials) > 0 else pd.DataFrame()

# 🔵 اكتشاف المعاملات تلقائياً
numeric_params, categorical_params = detect_parameters(df_complete)
print(f"\n🔍 اكتُشفت {len(numeric_params)} معامل رقمي و {len(categorical_params)} تصنيفي")
print(f"   الرقمية: {[p.replace('params_', '') for p in numeric_params]}")
print(f"   التصنيفية: {[p.replace('params_', '') for p in categorical_params]}")


# ===================================================================
# القسم 1: الرسوم البيانية الأساسية (Optimization History)
# ===================================================================
print("\n" + "=" * 75)
print("📈 القسم 1: الرسوم البيانية الأساسية")
print("=" * 75)
print("   → History Plot...")

fig, ax = plt.subplots(figsize=(12, 6))
values = df_complete['value'].values
best_so_far = np.maximum.accumulate(values)
trial_numbers = df_complete['number'].values

ax.plot(trial_numbers, values, 'o-', color='steelblue',
        markersize=6, label='PSNR لكل trial', alpha=0.7)
ax.plot(trial_numbers, best_so_far, 'r-', linewidth=2,
        label=f'أفضل PSNR تراكمي ({best_so_far[-1]:.3f} dB)')
ax.axhline(y=study.best_value, color='green', linestyle='--',
           alpha=0.5, label=f'أفضل نتيجة = {study.best_value:.3f} dB')
ax.set_xlabel('رقم المحاولة (Trial)')
ax.set_ylabel('PSNR (dB)')
ax.set_title('تاريخ التحسين — PSNR عبر المحاولات')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
save_figure(fig, "01_optimization_history")


# ===================================================================
# القسم 2: أهمية المعاملات
# ===================================================================
print("   → Parameter Importance...")
try:
    plot_result = optuna_vis.plot_param_importances(study)
    fig = fix_optuna_plot_size(plot_result, size=(12, 7))
    if fig is not None:
        save_figure(fig, "02_param_importance")
    else:
        print("   ⚠️ لم يتم الحصول على Figure")
except Exception as e:
    print(f"   ⚠️ تعذّر: {e}")


# ===================================================================
# القسم 3: Slice + Contour (يعملان مع أي معاملات)
# ===================================================================
print("   → Slice Plot...")
try:
    plot_result = optuna_vis.plot_slice(study)
    fig = fix_optuna_plot_size(plot_result, size=(18, 12))
    if fig is not None:
        save_figure(fig, "03_slice_plot")
    else:
        print("   ⚠️ لم يتم الحصول على Figure")
except Exception as e:
    print(f"   ⚠️ تعذّر: {e}")

print("   → Contour Plot...")
try:
    plot_result = optuna_vis.plot_contour(study)
    fig = fix_optuna_plot_size(plot_result, size=(18, 12))
    if fig is not None:
        save_figure(fig, "04_contour_plot")
    else:
        print("   ⚠️ لم يتم الحصول على Figure")
except Exception as e:
    print(f"   ⚠️ تعذّر: {e}")

print("   → Parallel Coordinates...")
try:
    plot_result = optuna_vis.plot_parallel_coordinate(study)
    fig = fix_optuna_plot_size(plot_result, size=(18, 9))
    if fig is not None:
        save_figure(fig, "05_parallel_coordinates")
    else:
        print("   ⚠️ لم يتم الحصول على Figure")
except Exception as e:
    print(f"   ⚠️ تعذّر: {e}")


# ===================================================================
# القسم 4: Custom Scatter Plots (لكل معامل تلقائياً)
# ===================================================================
print("\n" + "=" * 75)
print("📊 القسم 2: Custom Scatter Plots (تلقائي)")
print("=" * 75)
print("   → Scatter لكل معامل رقمي...")

all_plottable = numeric_params + categorical_params
n_params = len(all_plottable)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]

for idx, param in enumerate(all_plottable):
    ax = axes[idx]
    x = df_complete[param].values
    y = df_complete['value'].values
    
    scatter = ax.scatter(x, y, c=y, cmap='viridis',
                        s=80, alpha=0.7, edgecolors='black')
    
    # أفضل قيمة
    best_idx = np.argmax(y)
    ax.scatter(x[best_idx], y[best_idx], color='red', s=200,
               marker='*', zorder=5, edgecolors='black',
               label=f'الأفضل: {y[best_idx]:.3f}')
    
    # خط الاتجاه (فقط للمعاملات الرقمية)
    if param in numeric_params:
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.5,
                    linewidth=1.5, label='خط الاتجاه')
        except Exception:
            pass
    
    clean_name = param.replace('params_', '')
    ax.set_xlabel(clean_name, fontsize=11, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(f'{clean_name} vs PSNR', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

for idx in range(len(all_plottable), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
save_figure(fig, "06_custom_scatter_plots")


# ===================================================================
# القسم 5: Correlation Heatmap (تلقائي)
# ===================================================================
print("   → Correlation Heatmap...")

corr_data = df_complete[numeric_params + ['value']].copy()
if len(numeric_params) > 1:
    corr_matrix = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1)
    
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black',
                    fontsize=9)
    
    clean_labels = [c.replace('params_', '') for c in corr_matrix.columns]
    ax.set_xticks(range(len(clean_labels)))
    ax.set_yticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(clean_labels, fontsize=10)
    ax.set_title('مصفوفة الارتباط', fontsize=13, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='معامل الارتباط')
    plt.tight_layout()
    save_figure(fig, "07_correlation_heatmap")


# ===================================================================
# القسم 6: تحليل Top 10 Trials (تلقائي)
# ===================================================================
print("   → Top 10 Trials...")

df_sorted = df_complete.sort_values('value', ascending=False).head(10)
display_params = numeric_params + categorical_params

fig, ax = plt.subplots(figsize=(18, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Trial', 'PSNR'] + [p.replace('params_', '') for p in display_params]
for _, row in df_sorted.iterrows():
    row_data = [f"{int(row['number'])}", f"{row['value']:.3f}"]
    for p in display_params:
        val = row[p]
        if isinstance(val, (float, np.floating)) and val < 1:
            row_data.append(f"{val:.2e}")
        elif isinstance(val, (float, np.floating)):
            row_data.append(f"{val:.2f}")
        else:
            row_data.append(str(val))
    table_data.append(row_data)

table = ax.table(cellText=table_data, colLabels=headers,
                 cellLoc='center', loc='center',
                 colColours=['#4472C4'] * len(headers))
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

for i in range(len(table_data)):
    for j in range(len(headers)):
        cell = table[(i + 1, j)]
        if j == 1:
            cell.set_facecolor('#FFF2CC')
        elif i == 0:
            cell.set_facecolor('#C6EFCE')

ax.set_title('أفضل 10 محاولات', fontsize=14, fontweight='bold', pad=20)
save_figure(fig, "08_top10_trials")


# ===================================================================
# 🆕 القسم 7: تحليل trials المقتطعة (PRUNED)
# ===================================================================
print("\n" + "=" * 75)
print("✂️ القسم 3: تحليل trials المقتطعة (PRUNED)")
print("=" * 75)

if len(df_pruned) > 0:
    print(f"   عدد trials المقتطعة: {len(df_pruned)}")
    
    # 7.1 مقارنة توزيع المعاملات بين trials المكتملة والمقتطعة
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, param in enumerate(all_plottable):
        ax = axes[idx]
        if param in df_complete.columns and param in df_pruned.columns:
            complete_vals = df_complete[param].dropna().values
            pruned_vals = df_pruned[param].dropna().values
            
            # رسم histogram مقارن
            if param in numeric_params:
                bins = np.linspace(
                    min(complete_vals.min(), pruned_vals.min() if len(pruned_vals) > 0 else complete_vals.min()),
                    max(complete_vals.max(), pruned_vals.max() if len(pruned_vals) > 0 else complete_vals.max()),
                    15
                )
                ax.hist(complete_vals, bins=bins, alpha=0.6, 
                        color='green', label=f'مكتملة ({len(complete_vals)})')
                if len(pruned_vals) > 0:
                    ax.hist(pruned_vals, bins=bins, alpha=0.6, 
                            color='red', label=f'مقتطعة ({len(pruned_vals)})')
            else:
                # معامل تصنيفي
                all_vals = sorted(set(list(complete_vals) + list(pruned_vals)))
                complete_counts = [np.sum(complete_vals == v) for v in all_vals]
                pruned_counts = [np.sum(pruned_vals == v) for v in all_vals]
                x_pos = np.arange(len(all_vals))
                width = 0.35
                ax.bar(x_pos - width/2, complete_counts, width, 
                       alpha=0.6, color='green', label='مكتملة')
                ax.bar(x_pos + width/2, pruned_counts, width, 
                       alpha=0.6, color='red', label='مقتطعة')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(v) for v in all_vals], rotation=45)
            
            clean_name = param.replace('params_', '')
            ax.set_xlabel(clean_name, fontsize=10, fontweight='bold')
            ax.set_ylabel('التكرار', fontsize=10)
            ax.set_title(f'{clean_name}: مكتملة vs مقتطعة', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(all_plottable), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_figure(fig, "12_pruned_vs_complete_distribution")
    
    # 7.2 جدول: أي قيم معاملات تُقتطع أكثر؟
    print("   → تحليل مناطق الاقتطاع...")
    
    # حساب نسبة الاقتطاع لكل قيمة معامل
    pruning_analysis = {}
    for param in all_plottable:
        if param not in df.columns:
            continue
        
        # إجمالي trials لكل قيمة
        all_counts = df[df['state'].isin(['COMPLETE', 'PRUNED'])].groupby(param).size()
        pruned_counts = df_pruned.groupby(param).size() if len(df_pruned) > 0 else pd.Series()
        
        param_analysis = []
        for val in all_counts.index:
            total = all_counts[val]
            pruned = pruned_counts.get(val, 0)
            rate = pruned / total * 100 if total > 0 else 0
            param_analysis.append((val, total, pruned, rate))
        
        pruning_analysis[param] = param_analysis
    
    # حفظ النتائج في ملف نصي
    pruning_report_path = os.path.join(OUTPUT_DIR, "pruning_analysis.txt")
    with open(pruning_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 75 + "\n")
        f.write("✂️ تحليل trials المقتطعة — أي قيم معاملات تؤدي للاقتطاع؟\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"إجمالي trials: {len(all_trials)}\n")
        f.write(f"مكتملة: {len(complete_trials)}\n")
        f.write(f"مقتطعة: {len(pruned_trials)}\n")
        f.write(f"نسبة الاقتطاع: {pruning_rate:.1f}%\n\n")
        
        for param, analysis in pruning_analysis.items():
            clean = param.replace('params_', '')
            f.write(f"\n{clean}:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'القيمة':>15} | {'إجمالي':>8} | {'مقتطعة':>8} | {'نسبة %':>8}\n")
            f.write("-" * 50 + "\n")
            
            # ترتيب حسب نسبة الاقتطاع
            analysis_sorted = sorted(analysis, key=lambda x: x[3], reverse=True)
            for val, total, pruned, rate in analysis_sorted:
                marker = "🔴" if rate > 50 else "🟡" if rate > 25 else "🟢"
                f.write(f"{str(val):>15} | {total:>8} | {pruned:>8} | {rate:>7.1f}% {marker}\n")
    
    print(f"   ✅ تم حفظ تحليل الاقتطاع في: {pruning_report_path}")
    
    # 7.3 رسم خاص بنسبة الاقتطاع لكل قيمة معامل
    for param in numeric_params[:6]:  # أول 6 معاملات رقمية
        if param not in pruning_analysis:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        analysis = pruning_analysis[param]
        values = [a[0] for a in analysis]
        rates = [a[3] for a in analysis]
        totals = [a[1] for a in analysis]
        
        # رسم بياني
        bars = ax.bar(range(len(values)), rates, 
                     color=['red' if r > 50 else 'orange' if r > 25 else 'green' 
                            for r in rates],
                     alpha=0.7, edgecolor='black')
        
        # إضافة عدد trials فوق الأعمدة
        for i, (bar, total) in enumerate(zip(bars, totals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'n={total}', ha='center', va='bottom', fontsize=9)
        
        clean_name = param.replace('params_', '')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=45)
        ax.set_xlabel(clean_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('نسبة الاقتطاع (%)', fontsize=11)
        ax.set_title(f'نسبة الاقتطاع لكل قيمة {clean_name}', 
                    fontsize=12, fontweight='bold')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='25%')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        plt.tight_layout()
        save_figure(fig, f"13_pruning_rate_{clean_name}")

else:
    print("   ℹ️ لا توجد trials مقتطعة. تخطي هذا التحليل.")


# ===================================================================
# 🆕 القسم 8: الاختبارات الإحصائية
# ===================================================================
print("\n" + "=" * 75)
print("📊 القسم 4: الاختبارات الإحصائية")
print("=" * 75)

stats_report_path = os.path.join(OUTPUT_DIR, "statistical_tests.txt")
with open(stats_report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("📊 تقرير الاختبارات الإحصائية\n")
    f.write("=" * 75 + "\n\n")
    
    # 8.1 الإحصاءات الوصفية
    print("   → إحصاءات وصفية...")
    psnr_values = df_complete['value'].values
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values, ddof=1)
    median_psnr = np.median(psnr_values)
    min_psnr = np.min(psnr_values)
    max_psnr = np.max(psnr_values)
    q25, q75 = np.percentile(psnr_values, [25, 75])
    
    # فاصل الثقة
    ci_lower, ci_upper = confidence_interval(psnr_values)
    
    f.write("1) الإحصاءات الوصفية لـ PSNR (trials المكتملة)\n")
    f.write("-" * 50 + "\n")
    f.write(f"   عدد trials: {len(psnr_values)}\n")
    f.write(f"   المتوسط: {mean_psnr:.4f} dB\n")
    f.write(f"   الانحراف المعياري: {std_psnr:.4f} dB\n")
    f.write(f"   الوسيط: {median_psnr:.4f} dB\n")
    f.write(f"   الأدنى: {min_psnr:.4f} dB\n")
    f.write(f"   الأعلى: {max_psnr:.4f} dB\n")
    f.write(f"   الربع الأول (Q1): {q25:.4f} dB\n")
    f.write(f"   الربع الثالث (Q3): {q75:.4f} dB\n")
    f.write(f"   IQR: {q75 - q25:.4f} dB\n")
    f.write(f"   فاصل الثقة 95% (Bootstrap): [{ci_lower:.4f}, {ci_upper:.4f}]\n")
    f.write(f"   الخطأ المعياري (SE): {std_psnr / np.sqrt(len(psnr_values)):.4f} dB\n\n")
    
    print(f"      المتوسط: {mean_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"      فاصل الثقة 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 8.2 اختبار Wilcoxon: أفضل 25% مقابل أسوأ 25%
    print("   → اختبار Wilcoxon (أفضل 25% vs أسوأ 25%)...")
    n_top = max(3, len(psnr_values) // 4)
    sorted_psnr = np.sort(psnr_values)
    worst_group = sorted_psnr[:n_top]
    best_group = sorted_psnr[-n_top:]
    
    try:
        # Wilcoxon signed-rank test (يقارن عينتين مرتبطتين)
        # لكن هنا العينتان مستقلتان، لذا نستخدم Mann-Whitney U
        statistic, p_value = stats.mannwhitneyu(
            best_group, worst_group, alternative='greater'
        )
        
        f.write("2) اختبار Mann-Whitney U (أفضل 25% vs أسوأ 25%)\n")
        f.write("-" * 50 + "\n")
        f.write(f"   حجم أفضل مجموعة: {n_top}\n")
        f.write(f"   حجم أسوأ مجموعة: {n_top}\n")
        f.write(f"   متوسط أفضل مجموعة: {np.mean(best_group):.4f} dB\n")
        f.write(f"   متوسط أسوأ مجموعة: {np.mean(worst_group):.4f} dB\n")
        f.write(f"   الفرق: {np.mean(best_group) - np.mean(worst_group):.4f} dB\n")
        f.write(f"   إحصائية U: {statistic:.4f}\n")
        f.write(f"   p-value: {p_value:.6e}\n")
        
        if p_value < 0.001:
            f.write("   ✅ فرق معنوي جداً (p < 0.001) — النتائج قابلة للنشر\n")
        elif p_value < 0.05:
            f.write("   ✅ فرق معنوي (p < 0.05) — النتائج مقبولة\n")
        else:
            f.write("   ⚠️ لا يوجد فرق معنوي (p >= 0.05) — قد تحتاج trials أكثر\n")
        f.write("\n")
        
        print(f"      p-value = {p_value:.6e}")
    except Exception as e:
        f.write(f"   ⚠️ تعذّر إجراء الاختبار: {e}\n\n")
    
    # 8.3 اختبار natural distribution (Shapiro-Wilk)
    print("   → اختبار Shapiro-Wilk (توزيع طبيعي؟)...")
    if len(psnr_values) <= 5000 and len(psnr_values) >= 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(psnr_values[:5000])
            f.write("3) اختبار Shapiro-Wilk (هل التوزيع طبيعي؟)\n")
            f.write("-" * 50 + "\n")
            f.write(f"   إحصائية W: {shapiro_stat:.4f}\n")
            f.write(f"   p-value: {shapiro_p:.6e}\n")
            if shapiro_p > 0.05:
                f.write("   ✅ التوزيع طبيعي (p > 0.05)\n")
                f.write("   → يمكن استخدام اختبارات Parametric (مثل t-test)\n\n")
            else:
                f.write("   ⚠️ التوزيع ليس طبيعياً (p < 0.05)\n")
                f.write("   → يُفضَّل استخدام اختبارات Non-parametric (مثل Mann-Whitney)\n\n")
        except Exception as e:
            f.write(f"   ⚠️ تعذّر: {e}\n\n")
    
    # 8.4 مقارنة trials المكتملة والمقتطعة (إن وجدت)
    if len(df_pruned) > 0:
        f.write("4) مقارنة trials المكتملة vs المقتطعة\n")
        f.write("-" * 50 + "\n")
        f.write(f"   عدد المكتملة: {len(complete_trials)}\n")
        f.write(f"   عدد المقتطعة: {len(pruned_trials)}\n")
        f.write(f"   نسبة الاقتطاع: {pruning_rate:.1f}%\n\n")
        
        # هل معاملات معينة ترتبط بالاقتطاع؟
        f.write("   المعاملات الأكثر ارتباطاً بالاقتطاع:\n")
        for param in numeric_params:
            if param not in df.columns:
                continue
            complete_vals = df_complete[param].dropna().values
            pruned_vals = df_pruned[param].dropna().values
            if len(pruned_vals) > 0 and len(complete_vals) > 0:
                try:
                    stat, p = stats.mannwhitneyu(complete_vals, pruned_vals)
                    if p < 0.05:
                        clean = param.replace('params_', '')
                        f.write(f"      🔴 {clean}: p = {p:.4f} (فرق معنوي)\n")
                except Exception:
                    pass
        f.write("\n")
    
    # 8.5 فاصل الثقة لأفضل trial
    f.write("5) موثوقية أفضل trial\n")
    f.write("-" * 50 + "\n")
    best_psnr = study.best_value
    f.write(f"   أفضل PSNR: {best_psnr:.4f} dB\n")
    f.write(f"   مقارنة بالمعدل: +{best_psnr - mean_psnr:.4f} dB\n")
    f.write(f"   عدد الانحرافات المعيارية فوق المعدل: {(best_psnr - mean_psnr) / std_psnr:.2f}σ\n")
    
    if best_psnr > mean_psnr + 2 * std_psnr:
        f.write("   ✅ أفضل trial أعلى من المتوسط بـ 2σ — نتيجة متميزة\n")
    elif best_psnr > mean_psnr + std_psnr:
        f.write("   ✅ أفضل trial أعلى من المتوسط بـ 1σ — نتيجة جيدة\n")
    else:
        f.write("   ⚠️ أفضل trial قريب من المتوسط — قد تحتاج trials أكثر\n")

print(f"   ✅ تم حفظ الاختبارات الإحصائية في: {stats_report_path}")


# ===================================================================
# القسم 9: تقرير نصي شامل
# ===================================================================
print("\n" + "=" * 75)
print("📝 القسم 5: التقرير النصي الشامل")
print("=" * 75)

report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("📊 تقرير تحليل دراسة Optuna (عام)\n")
    f.write("=" * 75 + "\n\n")
    
    f.write(f"📚 اسم الدراسة: {STUDY_NAME}\n")
    f.write(f"📂 مسار التخزين: {STORAGE_PATH}\n")
    f.write(f"📊 إجمالي trials: {len(all_trials)}\n")
    f.write(f"✅ مكتملة: {len(complete_trials)}\n")
    f.write(f"✂️ مقتطعة: {len(pruned_trials)} ({pruning_rate:.1f}%)\n")
    f.write(f"❌ فاشلة: {len(failed_trials)}\n")
    f.write(f"🏆 أفضل PSNR: {study.best_value:.4f} dB\n")
    f.write(f"🎯 أفضل Trial: #{study.best_trial.number}\n\n")
    
    f.write("=" * 75 + "\n")
    f.write("🏆 أفضل المعاملات:\n")
    f.write("=" * 75 + "\n")
    for key, value in study.best_trial.params.items():
        if isinstance(value, float):
            f.write(f"  {key:>25} : {value:.6e}\n")
        else:
            f.write(f"  {key:>25} : {value}\n")
    
    f.write("\n" + "=" * 75 + "\n")
    f.write("📈 معامل الارتباط بين كل معامل و PSNR (مرتب):\n")
    f.write("=" * 75 + "\n")
    if len(numeric_params) > 0:
        corr_data = df_complete[numeric_params + ['value']].corr()['value'].drop('value')
        corr_data = corr_data.sort_values(ascending=False)
        for param, corr in corr_data.items():
            clean = param.replace('params_', '')
            if abs(corr) > 0.5:
                marker = "🟢" if corr > 0 else "🔴"
            elif abs(corr) > 0.3:
                marker = "🟡" if corr > 0 else "🟠"
            else:
                marker = "⚪"
            f.write(f"  {marker} {clean:>25} : {corr:+.4f}\n")
    
    f.write("\n" + "=" * 75 + "\n")
    f.write("💡 توصيات لاختيار أفضل المعاملات:\n")
    f.write("=" * 75 + "\n")
    if len(numeric_params) > 0:
        for param, corr in corr_data.items():
            clean = param.replace('params_', '')
            if abs(corr) > 0.3:
                if corr > 0:
                    f.write(f"  ✅ زيادة {clean} يحسّن PSNR (r = {corr:+.3f})\n")
                else:
                    f.write(f"  ✅ تقليل {clean} يحسّن PSNR (r = {corr:+.3f})\n")

print(f"✅ تم حفظ التقرير في: {report_path}")

# حفظ DataFrame كامل
csv_path = os.path.join(OUTPUT_DIR, "all_trials_full.csv")
df_complete.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ تم حفظ كل المحاولات في: {csv_path}")


# ===================================================================
# الملخص النهائي
# ===================================================================
print("\n" + "=" * 75)
print("🎉 تم الانتهاء من التحليل الشامل!")
print("=" * 75)
print(f"📂 جميع الملفات في: {OUTPUT_DIR}")
print("\n📁 الملفات المنتجة:")
print("   01_optimization_history.pdf/png")
print("   02_param_importance.pdf/png")
print("   03_slice_plot.pdf/png")
print("   04_contour_plot.pdf/png")
print("   05_parallel_coordinates.pdf/png")
print("   06_custom_scatter_plots.pdf/png")
print("   07_correlation_heatmap.pdf/png")
print("   08_top10_trials.pdf/png")
print("   12_pruned_vs_complete_distribution.pdf/png  ← 🆕 مقارنة")
print("   13_pruning_rate_*.pdf/png                     ← 🆕 نسب الاقتطاع")
print("   analysis_report.txt                           ← تقرير شامل")
print("   pruning_analysis.txt                          ← 🆕 تحليل الاقتطاع")
print("   statistical_tests.txt                         ← 🆕 اختبارات إحصائية")
print("   all_trials_full.csv")
print("=" * 75)