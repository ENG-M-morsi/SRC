"""
plot_results.py
---------------
يقرأ ملف Excel (DAT_tot.xlsx) الموجود في نفس المجلد ويرسم:
  - رسم PSNR مقابل Epoch  →  مع تحديد أعلى قيمة
  - رسم Loss  مقابل Epoch  →  مع legend

الملف يجب أن يحتوي على:
  - ورقة "PSNR"  بأعمدة: Epoch  +  أي عدد من أعمدة النماذج
  - ورقة "Loss"  بأعمدة: Epoch  +  أي عدد من أعمدة النماذج  (اختيارية)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ───────────────────────────────────────────────
# 0.  مسار الملف  (نفس مجلد السكريبت)
# ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# البحث تلقائياً عن أول ملف xlsx في نفس المجلد
xlsx_files = [f for f in os.listdir(SCRIPT_DIR) if f.endswith(".xlsx")]

if not xlsx_files:
    sys.exit("[خطأ] لم يُعثر على أي ملف .xlsx في المجلد.")

if len(xlsx_files) > 1:
    print(f"[تنبيه] يوجد أكثر من ملف xlsx، سيتم استخدام: {xlsx_files[0]}")

EXCEL_FILE = os.path.join(SCRIPT_DIR, xlsx_files[0])
print(f"[معلومة] جارٍ قراءة الملف: {xlsx_files[0]}")

# ───────────────────────────────────────────────
# 1.  قراءة البيانات
# ───────────────────────────────────────────────
all_sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
sheet_names_lower = {k.strip().upper(): k for k in all_sheets}

def load_sheet(name_upper):
    """تُعيد DataFrame أو None إن لم تكن الورقة موجودة."""
    key = sheet_names_lower.get(name_upper)
    if key is None:
        return None
    df = all_sheets[key].dropna(how="all")
    # تأكّد أن العمود الأول هو Epoch
    df.columns = [str(c).strip() for c in df.columns]
    return df

df_psnr = load_sheet("PSNR")
df_loss = load_sheet("LOSS")

if df_psnr is None:
    sys.exit("[خطأ] لم يُعثر على ورقة 'PSNR' في الملف.")

# ───────────────────────────────────────────────
# 2.  إعدادات الرسم العامة
# ───────────────────────────────────────────────
COLORS  = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0",
           "#00BCD4", "#F44336", "#8BC34A"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]
MARKER_EVERY = max(1, len(df_psnr) // 20)   # علامة كل ~5% من النقاط

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.grid":    True,
    "grid.alpha":   0.35,
    "grid.linestyle": "--",
    "figure.dpi":   150,
})

fig_count = 1 + (0 if df_loss is None else 1)
fig, axes = plt.subplots(fig_count, 1,
                         figsize=(10, 5 * fig_count),
                         constrained_layout=True)

if fig_count == 1:
    axes = [axes]

# ───────────────────────────────────────────────
# 3.  رسم PSNR
# ───────────────────────────────────────────────
ax_psnr = axes[0]
epoch_col_psnr = df_psnr.columns[0]
model_cols_psnr = df_psnr.columns[1:]

for i, col in enumerate(model_cols_psnr):
    color  = COLORS[i % len(COLORS)]
    marker = MARKERS[i % len(MARKERS)]
    x = df_psnr[epoch_col_psnr].values
    y = df_psnr[col].values

    ax_psnr.plot(x, y,
                 label=col,
                 color=color,
                 linewidth=1.8,
                 zorder=3)

    # ── تحديد أعلى قيمة ──
    best_idx = int(np.argmax(y))
    bx, by   = x[best_idx], y[best_idx]

    # نقطة عظمى باللون الأحمر فقط
    ax_psnr.scatter(bx, by, color="red", s=120,
                    edgecolors="darkred", linewidths=1.4,
                    zorder=5)

    # تسمية النص باللون الأحمر
    offset_y = (ax_psnr.get_ylim()[1] - ax_psnr.get_ylim()[0]) * 0.025 \
               if ax_psnr.get_ylim()[1] != ax_psnr.get_ylim()[0] else 0.05
    ax_psnr.annotate(
        f"Max {col}\n({bx:.0f}, {by:.4f} dB)",
        xy=(bx, by),
        xytext=(bx + len(x) * 0.03, by + offset_y),
        fontsize=8.5,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="red", alpha=0.85),
        zorder=6,
    )

ax_psnr.set_title("PSNR vs. Epoch", fontsize=14, fontweight="bold", pad=10)
ax_psnr.set_xlabel("Epoch", fontsize=12)
ax_psnr.set_ylabel("PSNR (dB)", fontsize=12)
ax_psnr.xaxis.set_major_locator(ticker.AutoLocator())
ax_psnr.legend(title="Model", loc="lower right",
               framealpha=0.9, edgecolor="gray")

# ───────────────────────────────────────────────
# 4.  رسم Loss  (إن وُجدت)
# ───────────────────────────────────────────────
if df_loss is not None:
    ax_loss = axes[1]
    epoch_col_loss = df_loss.columns[0]
    model_cols_loss = df_loss.columns[1:]

    for i, col in enumerate(model_cols_loss):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        x = df_loss[epoch_col_loss].values
        y = df_loss[col].values

        ax_loss.plot(x, y,
                     label=col,
                     color=color,
                     linewidth=1.8,
                     marker=marker,
                     markevery=max(1, len(x) // 20),
                     markersize=5,
                     zorder=3)

    ax_loss.set_title("Loss vs. Epoch", fontsize=14, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.xaxis.set_major_locator(ticker.AutoLocator())
    ax_loss.legend(title="Model", loc="upper right",
                   framealpha=0.9, edgecolor="gray")
else:
    print("[تنبيه] لم يُعثر على ورقة 'Loss'  ← سيُرسم رسم PSNR فقط.")
    print("        لإضافة Loss أنشئ ورقة باسم 'Loss' بنفس تنسيق ورقة PSNR.")

# ───────────────────────────────────────────────
# 5.  حفظ الصورة
# ───────────────────────────────────────────────
OUT_PNG = os.path.join(SCRIPT_DIR, "training_curves.png")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n✅ تم حفظ الرسم في:  {OUT_PNG}")