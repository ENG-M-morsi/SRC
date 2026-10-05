# ===================================================================
# analyze_optuna_study.py — تحليل شامل لدراسة Optuna
# يولّد رسوماً بيانية توضح العلاقة بين كل معامل و PSNR
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

# ===================================================================
# الإعدادات
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(SCRIPT_DIR, "optuna_study.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"
STUDY_NAME = "hutcn_hyperopt_separate_v1"   # ← نفس الاسم المستخدم في البحث

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "optuna_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# دالة لحفظ الرسم بصيغتي PDF و PNG
# ===================================================================
def save_figure(fig, name):
    """حفظ الرسم بصيغتي PDF و PNG."""
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ {name}.pdf + {name}.png")
    plt.close(fig)

# ===================================================================
# 1. تحميل الدراسة
# ===================================================================
print("=" * 70)
print("📊 تحليل دراسة Optuna")
print("=" * 70)

if not os.path.exists(STORAGE_PATH):
    print(f"❌ لم يتم العثور على قاعدة البيانات: {STORAGE_PATH}")
    sys.exit(1)

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

# استخراج البيانات
trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"✅ عدد المحاولات المكتملة: {len(trials)}")
print(f"🏆 أفضل PSNR: {study.best_value:.3f} dB (Trial {study.best_trial.number})")

if len(trials) < 2:
    print("⚠️ عدد المحاولات قليل جداً للتحليل. شغّل البحث أولاً.")
    sys.exit(1)

# تحويل البيانات إلى DataFrame
df = study.trials_dataframe()
df_complete = df[df['state'] == 'COMPLETE'].copy()
print(f"📋 الأعمدة المتوفرة: {list(df_complete.columns)}")

# ===================================================================
# 2. رسم تاريخ التحسين (Optimization History)
# ===================================================================
print("\n📈 توليد الرسوم البيانية...")
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
# 3. رسم أهمية المعاملات (Parameter Importance)
# ===================================================================
print("   → Parameter Importance Plot...")

try:
    fig = optuna_vis.plot_param_importances(study)
    fig.set_size_inches(10, 6)
    save_figure(fig, "02_param_importance")
except Exception as e:
    print(f"   ⚠️ تعذّر رسم أهمية المعاملات: {e}")

# ===================================================================
# 4. رسم Slice (العلاقة بين كل معامل والـ PSNR)
# ===================================================================
print("   → Slice Plot...")

try:
    fig = optuna_vis.plot_slice(study)
    # ضبط حجم الرسم
    fig.set_size_inches(15, 10)
    save_figure(fig, "03_slice_plot")
except Exception as e:
    print(f"   ⚠️ تعذّر رسم Slice Plot: {e}")

# ===================================================================
# 5. رسم Contour (العلاقة بين معاملين والـ PSNR)
# ===================================================================
print("   → Contour Plot...")

try:
    fig = optuna_vis.plot_contour(study)
    fig.set_size_inches(15, 10)
    save_figure(fig, "04_contour_plot")
except Exception as e:
    print(f"   ⚠️ تعذّر رسم Contour Plot: {e}")

# ===================================================================
# 6. رسم Parallel Coordinates (كل المحاولات في مساحة المعاملات)
# ===================================================================
print("   → Parallel Coordinates Plot...")

try:
    fig = optuna_vis.plot_parallel_coordinate(study)
    fig.set_size_inches(15, 8)
    save_figure(fig, "05_parallel_coordinates")
except Exception as e:
    print(f"   ⚠️ تعذّر رسم Parallel Coordinates: {e}")

# ===================================================================
# 7. رسم مخصص: كل معامل مقابل PSNR (Scatter Plot)
# ===================================================================
print("   → Custom Scatter Plots لكل معامل...")

# المعاملات الرقمية (التي يمكن رسمها)
numeric_params = [
    'params_n_feats',
    'params_num_heads_dat', 'params_ws_dat', 'params_num_blocks_dat',
    'params_num_heads_elan', 'params_ws_elan', 'params_num_blocks_elan',
    'params_patch_size', 'params_batch_size', 'params_lr',
    'params_weight_decay'
]

# تصفية المعاملات الموجودة فعلاً
numeric_params = [p for p in numeric_params if p in df_complete.columns]

n_params = len(numeric_params)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]

for idx, param in enumerate(numeric_params):
    ax = axes[idx]
    x = df_complete[param].values
    y = df_complete['value'].values

    # Scatter plot مع تلوين حسب القيمة
    scatter = ax.scatter(x, y, c=y, cmap='viridis',
                        s=80, alpha=0.7, edgecolors='black')

    # رسم أفضل قيمة
    best_idx = np.argmax(y)
    ax.scatter(x[best_idx], y[best_idx], color='red', s=200,
               marker='*', zorder=5, edgecolors='black',
               label=f'الأفضل: {y[best_idx]:.3f}')

    # خط متوسط متحرك (Trend line)
    try:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.5,
                linewidth=1.5, label='خط الاتجاه')
    except Exception:
        pass

    # تنظيف اسم المعامل
    clean_name = param.replace('params_', '')
    ax.set_xlabel(clean_name, fontsize=11, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(f'{clean_name} vs PSNR', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

# إخفاء المحاور الفارغة
for idx in range(len(numeric_params), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
save_figure(fig, "06_custom_scatter_plots")

# ===================================================================
# 8. رسم Heatmap للارتباط (Correlation Heatmap)
# ===================================================================
print("   → Correlation Heatmap...")

# حساب معامل الارتباط بين كل معامل و PSNR
corr_data = df_complete[numeric_params + ['value']].copy()
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto',
               vmin=-1, vmax=1)

# إضافة الأرقام
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        text = ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                       ha='center', va='center',
                       color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black',
                       fontsize=9)

# تنظيف الأسماء
clean_labels = [c.replace('params_', '') for c in corr_matrix.columns]
ax.set_xticks(range(len(clean_labels)))
ax.set_yticks(range(len(clean_labels)))
ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(clean_labels, fontsize=10)
ax.set_title('مصفوفة الارتباط بين المعاملات و PSNR',
             fontsize=13, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='معامل الارتباط')
plt.tight_layout()
save_figure(fig, "07_correlation_heatmap")

# ===================================================================
# 9. رسم أفضل 10 Trials
# ===================================================================
print("   → Top 10 Trials...")

# ترتيب حسب PSNR تنازلياً
df_sorted = df_complete.sort_values('value', ascending=False).head(10)

# المعاملات التي نريد عرضها
display_params = [p for p in numeric_params if p in df_sorted.columns]

# رسم جدول
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# تحضير البيانات
table_data = []
headers = ['Trial', 'PSNR'] + [p.replace('params_', '') for p in display_params]
for _, row in df_sorted.iterrows():
    row_data = [f"{int(row['number'])}", f"{row['value']:.3f}"]
    for p in display_params:
        val = row[p]
        if isinstance(val, float) and val < 1:
            row_data.append(f"{val:.2e}")
        elif isinstance(val, float):
            row_data.append(f"{val:.2f}")
        else:
            row_data.append(str(val))
    table_data.append(row_data)

table = ax.table(cellText=table_data, colLabels=headers,
                 cellLoc='center', loc='center',
                 colColours=['#4472C4'] * len(headers))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# تلوين الخلايا
for i in range(len(table_data)):
    for j in range(len(headers)):
        cell = table[(i + 1, j)]
        if j == 1:  # عمود PSNR
            cell.set_facecolor('#FFF2CC')
        elif i == 0:  # الصف الأول (الأفضل)
            cell.set_facecolor('#C6EFCE')

ax.set_title('أفضل 10 محاولات (Trials) من حيث PSNR',
             fontsize=14, fontweight='bold', pad=20)
save_figure(fig, "08_top10_trials")

# ===================================================================
# 10. رسم Best Trial تفصيلي
# ===================================================================
print("   → Best Trial Details...")

best_trial = study.best_trial
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# تجهيز النص
lines = [f"🏆 أفضل محاولة: Trial #{best_trial.number}",
         f"📈 أفضل PSNR: {best_trial.value:.4f} dB",
         "",
         "المعاملات:"]

for key, value in best_trial.params.items():
    if isinstance(value, float):
        lines.append(f"   {key:>25} : {value:.6e}")
    else:
        lines.append(f"   {key:>25} : {value}")

text = "\n".join(lines)

ax.text(0.05, 0.95, text, transform=ax.transAxes,
        fontsize=14, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1',
                  facecolor='#C6EFCE',
                  edgecolor='green',
                  linewidth=2))

save_figure(fig, "09_best_trial_details")

# ===================================================================
# 11. رسم Empirical Distribution Function
# ===================================================================
print("   → EDF Plot...")

try:
    fig = optuna_vis.plot_edf(study)
    fig.set_size_inches(10, 6)
    save_figure(fig, "10_edf_plot")
except Exception as e:
    print(f"   ⚠️ تعذّر رسم EDF Plot: {e}")

# ===================================================================
# 12. رسم Terminator Improvement
# ===================================================================
print("   → Terminator Improvement Plot...")

try:
    fig = optuna_vis.plot_terminator_improvement(study)
    fig.set_size_inches(10, 6)
    save_figure(fig, "11_terminator_improvement")
except Exception as e:
    print(f"   ⚠️ تعذّر رسم Terminator Improvement: {e}")

# ===================================================================
# 13. تقرير نصي شامل
# ===================================================================
print("\n📝 توليد التقرير النصي...")

report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("📊 تقرير تحليل دراسة Optuna\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"📚 اسم الدراسة: {STUDY_NAME}\n")
    f.write(f"📂 مسار التخزين: {STORAGE_PATH}\n")
    f.write(f"✅ عدد المحاولات المكتملة: {len(trials)}\n")
    f.write(f"🏆 أفضل PSNR: {study.best_value:.4f} dB\n")
    f.write(f"🎯 أفضل Trial: #{study.best_trial.number}\n\n")

    f.write("=" * 70 + "\n")
    f.write("🏆 أفضل المعاملات:\n")
    f.write("=" * 70 + "\n")
    for key, value in study.best_trial.params.items():
        if isinstance(value, float):
            f.write(f"  {key:>25} : {value:.6e}\n")
        else:
            f.write(f"  {key:>25} : {value}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("📈 معامل الارتباط بين كل معامل و PSNR:\n")
    f.write("=" * 70 + "\n")
    correlations = corr_data.corr()['value'].drop('value').sort_values(ascending=False)
    for param, corr in correlations.items():
        clean = param.replace('params_', '')
        if abs(corr) > 0.5:
            marker = "🟢" if corr > 0 else "🔴"
        elif abs(corr) > 0.3:
            marker = "🟡" if corr > 0 else "🟠"
        else:
            marker = "⚪"
        f.write(f"  {marker} {clean:>25} : {corr:+.4f}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("📋 أفضل 5 محاولات:\n")
    f.write("=" * 70 + "\n")
    top5 = df_complete.nlargest(5, 'value')
    for _, row in top5.iterrows():
        f.write(f"\n  Trial #{int(row['number'])}: PSNR = {row['value']:.4f} dB\n")
        for p in display_params:
            val = row[p]
            if isinstance(val, float) and val < 1:
                f.write(f"     {p.replace('params_', ''):>22}: {val:.6e}\n")
            elif isinstance(val, float):
                f.write(f"     {p.replace('params_', ''):>22}: {val:.2f}\n")
            else:
                f.write(f"     {p.replace('params_', ''):>22}: {val}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("💡 توصيات لاختيار أفضل المعاملات:\n")
    f.write("=" * 70 + "\n")

    # توصيات مبنية على الارتباط
    for param, corr in correlations.items():
        clean = param.replace('params_', '')
        if abs(corr) > 0.3:
            if corr > 0:
                f.write(f"  ✅ زيادة {clean} يحسّن PSNR (ارتباط: {corr:+.3f})\n")
            else:
                f.write(f"  ✅ تقليل {clean} يحسّن PSNR (ارتباط: {corr:+.3f})\n")

print(f"✅ تم حفظ التقرير في: {report_path}")

# حفظ DataFrame كامل
csv_path = os.path.join(OUTPUT_DIR, "all_trials_full.csv")
df_complete.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ تم حفظ كل المحاولات في: {csv_path}")

# ===================================================================
# الملخص النهائي
# ===================================================================
print("\n" + "=" * 70)
print("🎉 تم الانتهاء من التحليل!")
print("=" * 70)
print(f"📂 جميع الملفات محفوظة في: {OUTPUT_DIR}")
print("\n📁 الملفات المنتجة:")
print("   01_optimization_history.pdf/png")
print("   02_param_importance.pdf/png")
print("   03_slice_plot.pdf/png")
print("   04_contour_plot.pdf/png")
print("   05_parallel_coordinates.pdf/png")
print("   06_custom_scatter_plots.pdf/png   ← لكل معامل مقابل PSNR")
print("   07_correlation_heatmap.pdf/png    ← مصفوفة الارتباط")
print("   08_top10_trials.pdf/png")
print("   09_best_trial_details.pdf/png")
print("   10_edf_plot.pdf/png")
print("   11_terminator_improvement.pdf/png")
print("   analysis_report.txt                ← تقرير نصي شامل")
print("   all_trials_full.csv                ← جميع المحاولات")
print("=" * 70)