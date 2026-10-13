# =====================================================================
# compare_all_models.py — Universal multi-model comparison
# =====================================================================
# مكان الملف: المجلد الأب الذي يحتوي مجلدات الموديلات.
#
# البنية المتوقعة:
#   parent/
#   ├── compare_all_models.py           ← هذا الملف
#   ├── DHTCUN_baseline/
#   │   ├── optuna_study.db
#   │   └── params_vs_psnr_analysis/
#   │       ├── DAT_hutcn_hyperopt_study_3cols.xlsx
#   │       └── DAT_hutcn_hyperopt_study_full.xlsx
#   ├── DHTCUN_ELAN/
#   │   └── params_vs_psnr_analysis/
#   │       └── ELAN_hutcn_hyperopt_study_3cols.xlsx
#   └── ...
#
# المخرجات (في ./comparison_output/):
#   C1_params_vs_psnr_all_models.{pdf,png,jpeg}
#   C2_best_psnr_and_fewest_params.{pdf,png,jpeg}
#   C3_delta_vs_baseline.{pdf,png,jpeg}
#   E1_best_so_far_convergence.{pdf,png,jpeg}
#   E2_params_vs_psnr_by_nfeats.{pdf,png,jpeg}
#   E3_params_vs_psnr_by_numblocks.{pdf,png,jpeg}
#   E4_psnr_vs_training_time.{pdf,png,jpeg}
#   E5_flops_vs_psnr.{pdf,png,jpeg}      (اختياري — يحتاج fvcore)
#   models_comparison_summary.xlsx
#   trial_level_all_models.xlsx
#   README.txt
# =====================================================================

import os, sys, re, json, glob, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import optuna

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ═════════════════════════════════════════════════════════════════════
HERE       = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "comparison_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 🔴 مفتاح baseline = **اسم المجلد الرئيسي** (الابن المباشر لـ HERE)
#    مثال: لو الـ baseline داخل:
#      Org_rev2_best/DHTCUN+50T+50EP+optim_purn/params_vs_psnr_analysis/
#    فالمفتاح = "Org_rev2_best"  (وليس "DHTCUN+50T+50EP+optim_purn")
BASELINE_KEY = "Org_rev2_best"           # ← غيّره لو اختلف الاسم

# 🟠 أسماء العرض — المفاتيح هي **أسماء المجلدات الرئيسية**
MODEL_LABELS = {
    "Org_rev2_best": "DHTCUN",
    "DAT":           "DAT",
    "ELAN":          "ELAN",
    "HAT":           "HAT",
    "Omni-SR":       "OmniSR",
    "SRFormer2":     "SRFormer",
    "Swint_2":       "SwinT",
    "Wavelet Trans": "Wavelet",
}

# 🟠 مجلدات تُتجاهَل تماماً
EXCLUDE_FOLDERS = {
    "Rejected models", "__pycache__", ".git", ".idea",
    "comparison_output", "params_vs_psnr_analysis",
    "_snapshot_model",
}

# 🟢 عمق البحث الأقصى — 4 يغطّي حالة: root/AAA/BBB/params_vs_psnr_analysis/
MAX_DISCOVERY_DEPTH = 4

# 🟢 FLOPs (اختياري — من ملفات الإكسل، بدون إعادة بناء)
FLOPs_ENABLED = False
INPUT_SIZE_FOR_FLOPs = (1, 3, 48, 48)

# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — Plot style + save_figure
# ═════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'figure.autolayout': False,
})


def save_figure(fig, name):
    for ext in ('pdf', 'png', 'jpeg'):
        path = OUTPUT_DIR / f"{name}.{ext}"
        kw = dict(dpi=300, bbox_inches='tight', facecolor='white')
        if ext == 'jpeg':
            kw['pil_kwargs'] = {'quality': 95, 'optimize': True}
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"   ✅ {name}.pdf/.png/.jpeg")


def _label(key):
    return MODEL_LABELS.get(key, key)


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Discovery: find all model folders with their Excel files
# ═════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Discovery (recursive, uses TOP-LEVEL folder name)
# ═════════════════════════════════════════════════════════════════════
def discover_models():
    """
    يبحث recursively عن أي مجلد يحوي params_vs_psnr_analysis/ مع ملفات إكسل.

    ✅ المفتاح = اسم **المجلد الرئيسي** (الابن المباشر لـ HERE)
    ✅ مجلد الموديل الفعلي (الأعمق) يُخزَّن في model_dir

    مثال:
        HERE = types of Transformers_best/
        اكتُشف: Org_rev2_best/DHTCUN+50T+50EP+optim_purn/params_vs_psnr_analysis/
        المفتاح = "Org_rev2_best"
        model_dir = Org_rev2_best/DHTCUN+50T+50EP+optim_purn
    """
    found = {}
    base_depth = len(HERE.parts)

    for root, dirs, files in os.walk(HERE):
        # استثنِ المجلدات الممنوعة والمخفية
        dirs[:] = [d for d in dirs
                   if d not in EXCLUDE_FOLDERS and not d.startswith('.')]

        depth = len(Path(root).parts) - base_depth
        if depth >= MAX_DISCOVERY_DEPTH:
            dirs[:] = []
            continue

        root_path = Path(root)
        ana = root_path / 'params_vs_psnr_analysis'
        if not ana.is_dir():
            continue

        xlsx_3cols = sorted(ana.glob("*_3cols.xlsx"))
        xlsx_full  = sorted(ana.glob("*_full.xlsx"))
        db         = root_path / "optuna_study.db"

        if not xlsx_3cols and not xlsx_full:
            continue

        # ═══ المفتاح = اسم المجلد الرئيسي (المستوى الأول) ═══
        rel = root_path.relative_to(HERE)
        if len(rel.parts) >= 1:
            top_name = rel.parts[0]           # الابن المباشر لـ HERE
        else:
            top_name = root_path.name

        key = top_name

        # معالجة التكرار: لو أكثر من params_vs_psnr_analysis تحت نفس الرئيسي
        if key in found:
            # أضف اسم الأب المباشر لتمييزه
            key = f"{top_name}__{root_path.name}"
            print(f"   ℹ️  Duplicate top-level folder '{top_name}' — "
                  f"using key '{key}'")

        src = xlsx_full[0] if xlsx_full else xlsx_3cols[0]

        found[key] = {
            'model_dir':  root_path,               # المسار الأعمق
            'top_dir':    HERE / top_name,         # المسار الرئيسي
            '3cols_xlsx': xlsx_3cols[0] if xlsx_3cols else None,
            'full_xlsx':  xlsx_full[0]  if xlsx_full  else None,
            'db_path':    db if db.exists() else None,
            'src_file':   src,
        }

    return found


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Load per-model data
# ═════════════════════════════════════════════════════════════════════
def load_model_tables(models):
    """
    يرجع:
      trial_df : DataFrame موحّد بكل التجارب من كل الموديلات
                 أعمدة: model, trial, n_params, psnr, (HP columns إن وُجدت)
      per_model: dict key → DataFrame لكل موديل
    """
    frames = []
    per_model = {}

    for key, info in models.items():
        # فضّل full (فيه كل HP)
        src = info['full_xlsx'] or info['3cols_xlsx']
        if src is None:
            print(f"   ⚠️  {key}: no excel — skipped")
            continue
        try:
            df = pd.read_excel(src)
        except Exception as e:
            print(f"   ⚠️  {key}: cannot read {src.name} — {e}")
            continue

        # توحيد أسماء الأعمدة
        rename = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ('trial number', 'trial', 'number'):
                rename[c] = 'trial'
            elif cl in ('number of parameters', 'n_params', 'params'):
                rename[c] = 'n_params'
            elif cl in ('psnr (db)', 'psnr', 'value'):
                rename[c] = 'psnr'
        df = df.rename(columns=rename)

        needed = {'trial', 'n_params', 'psnr'}
        if not needed.issubset(df.columns):
            print(f"   ⚠️  {key}: missing columns {needed - set(df.columns)} — skipped")
            continue

        df = df[['trial', 'n_params', 'psnr'] +
                [c for c in df.columns
                 if c not in ('trial', 'n_params', 'psnr')]].copy()
        df.insert(0, 'model', key)
        df = df.sort_values('trial').reset_index(drop=True)

        per_model[key] = df
        frames.append(df)
        print(f"   ✅ {key:<20} → {len(df):>3} trials | "
              f"n_params ∈ [{df['n_params'].min()/1e6:.2f}M, "
              f"{df['n_params'].max()/1e6:.2f}M]")

    trial_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trial_df, per_model


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Summary table (matches slides 10 & 11)
# ═════════════════════════════════════════════════════════════════════
def build_summary(per_model):
    """
    جدول ملخص لكل موديل:
      - best_psnr + params @ best
      - min_params + psnr @ min_params
      - trade-off: smallest model within 0.1 dB of best
      - Δ vs baseline (psnr & params %)
    """
    rows = []
    for key, df in per_model.items():
        i_hi = df['psnr'].idxmax()
        i_lo = df['n_params'].idxmin()

        best_psnr  = float(df.loc[i_hi, 'psnr'])
        best_par   = int(df.loc[i_hi, 'n_params'])
        min_par    = int(df.loc[i_lo, 'n_params'])
        psnr_minp  = float(df.loc[i_lo, 'psnr'])

        # Best trade-off: smallest within (best_psnr - 0.1 dB)
        tol = 0.10
        cand = df[df['psnr'] >= best_psnr - tol]
        if len(cand) > 0:
            j = cand['n_params'].idxmin()
            trade_psnr = float(cand.loc[j, 'psnr'])
            trade_par  = int(cand.loc[j, 'n_params'])
            trade_trial = int(cand.loc[j, 'trial'])
        else:
            trade_psnr, trade_par, trade_trial = np.nan, np.nan, np.nan

        rows.append({
            'model':               key,
            'label':               _label(key),
            'n_trials':            len(df),
            'best_psnr':           best_psnr,
            'best_params':         best_par,
            'best_trial':          int(df.loc[i_hi, 'trial']),
            'min_params':          min_par,
            'psnr_at_min_params':  psnr_minp,
            'min_params_trial':    int(df.loc[i_lo, 'trial']),
            'trade_psnr':          trade_psnr,
            'trade_params':        trade_par,
            'trade_trial':         trade_trial,
        })

    summ = pd.DataFrame(rows)

    # إضافة Δ vs baseline
    if BASELINE_KEY in summ['model'].values:
        base = summ[summ['model'] == BASELINE_KEY].iloc[0]
        summ['Δpsnr_vs_base']   = summ['best_psnr']   - base['best_psnr']
        summ['Δparams_vs_base'] = summ['best_params'] - base['best_params']
        summ['Δparams_pct']     = (summ['best_params'] / base['best_params'] - 1) * 100
        summ['Δmin_par_pct']    = (summ['min_params']  / base['min_params']  - 1) * 100
    else:
        summ['Δpsnr_vs_base']   = np.nan
        summ['Δparams_vs_base'] = np.nan
        summ['Δparams_pct']     = np.nan
        summ['Δmin_par_pct']    = np.nan

    # ترتيب: baseline أولاً
    summ = summ.sort_values(
        ['model'],
        key=lambda c: [0 if m == BASELINE_KEY else 1 for m in c]
    ).reset_index(drop=True)
    return summ


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Pareto front
# ═════════════════════════════════════════════════════════════════════
def pareto_front(p, v):
    order = np.argsort(p)
    best, idx = -np.inf, []
    for i in order:
        if v[i] > best:
            idx.append(i); best = v[i]
    return np.array(idx)


# ═════════════════════════════════════════════════════════════════════
# C1 — رسم مستقل لكل موديل: PSNR vs #Params
#      مع علامتين: ⭐ Best PSNR  و  💎 Fewest params
# ═════════════════════════════════════════════════════════════════════
def plot_C1(per_model):
    """
    يُنتج:
      C1_all_models.*                      ← نظرة شاملة
      C1_<model>_vs_baseline.*             ← شكل مستقل لكل موديل
    """
    if BASELINE_KEY not in per_model:
        print("   ⚠️  No baseline — skip C1.")
        return
    base_df = per_model[BASELINE_KEY]
    base_p  = base_df['n_params'].values / 1e6
    base_v  = base_df['psnr'].values
    base_pf = pareto_front(base_p, base_v)

    # ─── (أ) نظرة شاملة ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    cmap = plt.get_cmap('tab10')
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    ax.scatter(base_p, base_v, s=120, marker='s', color='black',
               alpha=0.9, edgecolors='black', linewidths=0.8,
               label=_label(BASELINE_KEY) + " (baseline)", zorder=5)
    ax.plot(base_p[base_pf], base_v[base_pf], '-', color='black',
            lw=2.2, alpha=0.85, zorder=4)

    for i, (key, df) in enumerate(per_model.items()):
        if key == BASELINE_KEY:
            continue
        p = df['n_params'].values / 1e6
        v = df['psnr'].values
        color = cmap(i)
        ax.scatter(p, v, s=70, marker=markers[i % len(markers)],
                   color=color, alpha=0.75,
                   edgecolors='black', linewidths=0.6,
                   label=_label(key), zorder=3)
        pf = pareto_front(p, v)
        ax.plot(p[pf], v[pf], '--', color=color, lw=1.2, alpha=0.75, zorder=2)

    ax.set_xlabel('Number of parameters (M)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('C1 — All models vs baseline')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    fig.tight_layout()
    save_figure(fig, "C1_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────
    for key, df in per_model.items():
        if key == BASELINE_KEY:
            continue

        p = df['n_params'].values / 1e6
        v = df['psnr'].values
        pf = pareto_front(p, v)

        fig, ax = plt.subplots(figsize=(11, 7))

        # baseline
        ax.scatter(base_p, base_v, s=110, marker='s', color='black',
                   alpha=0.85, edgecolors='black', linewidths=0.7,
                   label=f'{_label(BASELINE_KEY)} (baseline)', zorder=5)
        ax.plot(base_p[base_pf], base_v[base_pf], '-', color='black',
                lw=2.2, alpha=0.85, zorder=4, label='Baseline Pareto front')

        # الموديل الحالي
        ax.scatter(p, v, s=90, marker='o', color='steelblue',
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=_label(key), zorder=4)
        ax.plot(p[pf], v[pf], '--', color='steelblue',
                lw=1.8, alpha=0.9, zorder=3,
                label=f'{_label(key)} Pareto front')

        # ⭐ Best PSNR
        ib = int(np.argmax(v))
        ax.scatter(p[ib], v[ib], marker='*', s=420, color='gold',
                   edgecolors='black', linewidths=1.2, zorder=7,
                   label=f'⭐ Best {_label(key)} = '
                         f'{v[ib]:.3f} dB ({p[ib]:.2f}M)')

        # 💎 Fewest parameters  (جديد)
        il = int(np.argmin(p))
        if il != ib:
            ax.scatter(p[il], v[il], marker='D', s=190, color='cyan',
                       edgecolors='black', linewidths=1.3, zorder=7,
                       label=f'💎 Fewest {_label(key)} = '
                             f'{p[il]:.2f}M ({v[il]:.3f} dB)')

        # ⬛ Best baseline (مرجعي)
        ib_b = int(np.argmax(base_v))
        ax.scatter(base_p[ib_b], base_v[ib_b], marker='s', s=260,
                   facecolors='none', edgecolors='gold', linewidths=2.4,
                   zorder=6,
                   label=f'⭐ Best baseline = '
                         f'{base_v[ib_b]:.3f} dB ({base_p[ib_b]:.2f}M)')

        ax.set_xlabel('Number of parameters (M)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'C1 — {_label(key)} vs {_label(BASELINE_KEY)}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', key)
        save_figure(fig, f"C1_{safe}_vs_baseline")

# ═════════════════════════════════════════════════════════════════════
# C1_FLOPs — نفس C1 لكن المحور X = FLOPs بدل #Params
# ═════════════════════════════════════════════════════════════════════
def plot_C1_flops(per_model):
    """
    يحتاج عمود 'flops_G' في ملفات الإكسل.
    يُنتج:
      C1_FLOPs_all_models.*
      C1_FLOPs_<model>_vs_baseline.*
    """
    if BASELINE_KEY not in per_model:
        return
    base_df = per_model[BASELINE_KEY]
    if ('flops_G' not in base_df.columns
            or base_df['flops_G'].isna().all()):
        print("   ℹ️  flops_G missing in baseline — skip C1_FLOPs.")
        return

    # احذف الصفوف التي بلا flops
    def _clean(df):
        return df.dropna(subset=['flops_G']).copy()

    base = _clean(base_df)
    base_f = base['flops_G'].values
    base_v = base['psnr'].values
    base_pf = pareto_front(base_f, base_v)

    # ─── (أ) نظرة شاملة ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    cmap = plt.get_cmap('tab10')
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    ax.scatter(base_f, base_v, s=120, marker='s', color='black',
               alpha=0.9, edgecolors='black', linewidths=0.8,
               label=_label(BASELINE_KEY) + " (baseline)", zorder=5)
    ax.plot(base_f[base_pf], base_v[base_pf], '-', color='black',
            lw=2.2, alpha=0.85, zorder=4)

    for i, (key, df) in enumerate(per_model.items()):
        if key == BASELINE_KEY:
            continue
        d = _clean(df)
        if d.empty:
            continue
        f = d['flops_G'].values
        v = d['psnr'].values
        color = cmap(i)
        ax.scatter(f, v, s=70, marker=markers[i % len(markers)],
                   color=color, alpha=0.75,
                   edgecolors='black', linewidths=0.6,
                   label=_label(key), zorder=3)
        pf = pareto_front(f, v)
        ax.plot(f[pf], v[pf], '--', color=color, lw=1.2,
                alpha=0.75, zorder=2)

    ax.set_xlabel(f'FLOPs (G) @ {INPUT_SIZE_FOR_FLOPs[2]}×'
                  f'{INPUT_SIZE_FOR_FLOPs[3]} LR input')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('C1 — FLOPs vs PSNR — all models')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    fig.tight_layout()
    save_figure(fig, "C1_FLOPs_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────
    for key, df in per_model.items():
        if key == BASELINE_KEY:
            continue
        d = _clean(df)
        if d.empty:
            continue

        f = d['flops_G'].values
        v = d['psnr'].values
        pf = pareto_front(f, v)

        fig, ax = plt.subplots(figsize=(11, 7))

        ax.scatter(base_f, base_v, s=110, marker='s', color='black',
                   alpha=0.85, edgecolors='black', linewidths=0.7,
                   label=f'{_label(BASELINE_KEY)} (baseline)', zorder=5)
        ax.plot(base_f[base_pf], base_v[base_pf], '-', color='black',
                lw=2.2, alpha=0.85, zorder=4, label='Baseline Pareto front')

        ax.scatter(f, v, s=90, marker='o', color='steelblue',
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=_label(key), zorder=4)
        ax.plot(f[pf], v[pf], '--', color='steelblue',
                lw=1.8, alpha=0.9, zorder=3,
                label=f'{_label(key)} Pareto front')

        # ⭐ Best PSNR
        ib = int(np.argmax(v))
        ax.scatter(f[ib], v[ib], marker='*', s=420, color='gold',
                   edgecolors='black', linewidths=1.2, zorder=7,
                   label=f'⭐ Best {_label(key)} = '
                         f'{v[ib]:.3f} dB ({f[ib]:.2f}G)')

        # 💎 Fewest FLOPs  (جديد)
        il = int(np.argmin(f))
        if il != ib:
            ax.scatter(f[il], v[il], marker='D', s=190, color='cyan',
                       edgecolors='black', linewidths=1.3, zorder=7,
                       label=f'💎 Fewest-FLOPs {_label(key)} = '
                             f'{f[il]:.2f}G ({v[il]:.3f} dB)')

        # ⬛ Best baseline (مرجعي)
        ib_b = int(np.argmax(base_v))
        ax.scatter(base_f[ib_b], base_v[ib_b], marker='s', s=260,
                   facecolors='none', edgecolors='gold', linewidths=2.4,
                   zorder=6,
                   label=f'⭐ Best baseline = '
                         f'{base_v[ib_b]:.3f} dB ({base_f[ib_b]:.2f}G)')

        ax.set_xlabel(f'FLOPs (G) @ {INPUT_SIZE_FOR_FLOPs[2]}×'
                      f'{INPUT_SIZE_FOR_FLOPs[3]} LR input')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'C1_FLOPs — {_label(key)} vs {_label(BASELINE_KEY)}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', key)
        save_figure(fig, f"C1_FLOPs_{safe}_vs_baseline")

# ═════════════════════════════════════════════════════════════════════
# C2_FLOPs — Best PSNR & Fewest FLOPs لكل موديل + مقارنة بالـ baseline
# ═════════════════════════════════════════════════════════════════════
def plot_C2_flops(per_model, summ):
    """
    يحتاج عمود 'flops_G'.
    يُنتج:
      C2_FLOPs_all_models.*
      C2_FLOPs_<model>_vs_baseline.*
    """
    if 'flops_G' not in trial_df.columns if 'trial_df' in dir() else True:
        pass  # trial_df قد لا يكون متاحاً هنا — نعمل من per_model

    # اجمع min_flops + psnr لكل موديل
    rows = []
    for key, df in per_model.items():
        if 'flops_G' not in df.columns:
            continue
        d = df.dropna(subset=['flops_G'])
        if d.empty:
            continue
        i_hi = d['psnr'].idxmax()
        i_lo = d['flops_G'].idxmin()
        rows.append({
            'model': key,
            'label': _label(key),
            'best_psnr':  float(d.loc[i_hi, 'psnr']),
            'flops_hi':   float(d.loc[i_hi, 'flops_G']),
            'min_flops':  float(d.loc[i_lo, 'flops_G']),
            'psnr_lo':    float(d.loc[i_lo, 'psnr']),
        })
    if not rows:
        print("   ℹ️  No flops data — skip C2_FLOPs.")
        return
    fdf = pd.DataFrame(rows)

    if BASELINE_KEY not in fdf['model'].values:
        return
    base = fdf[fdf['model'] == BASELINE_KEY].iloc[0]

    # ─── (أ) نظرة شاملة ─────────────────────────────────────────
    labels   = fdf['label'].tolist()
    best_p   = fdf['best_psnr'].tolist()
    best_f   = fdf['flops_hi'].tolist()
    min_f    = fdf['min_flops'].tolist()
    psnr_mp  = fdf['psnr_lo'].tolist()
    colors   = ['#d62728' if m == BASELINE_KEY else 'steelblue'
                for m in fdf['model']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    bars = ax.bar(range(len(labels)), best_p, color=colors,
                  edgecolor='black', alpha=0.88)
    for bar, gf, p in zip(bars, best_f, best_p):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f'{p:.3f} dB\n{gf:.2f}G',
                ha='center', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('Best PSNR (dB)')
    ax.set_title('Best PSNR per model (with FLOPs annotation)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    bars = ax.bar(range(len(labels)), min_f, color=colors,
                  edgecolor='black', alpha=0.88)
    for bar, gf, p in zip(bars, min_f, psnr_mp):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(min_f)*0.02,
                f'{gf:.2f}G\n{p:.2f} dB',
                ha='center', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('Minimum FLOPs (G)')
    ax.set_title('Fewest FLOPs per model (with PSNR annotation)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('C2_FLOPs — Best PSNR & Fewest FLOPs — all models',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "C2_FLOPs_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────
    for _, r in fdf.iterrows():
        if r['model'] == BASELINE_KEY:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # يسار: Best PSNR مع FLOPs
        ax = axes[0]
        lbls = [f"{_label(BASELINE_KEY)}\n(baseline)", r['label']]
        vals = [base['best_psnr'], r['best_psnr']]
        fls  = [base['flops_hi'],  r['flops_hi']]
        bars = ax.bar(lbls, vals, color=['#d62728', 'steelblue'],
                      edgecolor='black', alpha=0.88, width=0.55)
        for bar, gf, p in zip(bars, fls, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.03,
                    f'{p:.3f} dB\n{gf:.2f}G',
                    ha='center', fontsize=10)
        delta = r['best_psnr'] - base['best_psnr']
        ax.set_title(f'Best PSNR — Δ = {delta:+.3f} dB')
        ax.set_ylabel('PSNR (dB)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(min(vals) - 0.4, max(vals) + 0.6)

        # يمين: Fewest FLOPs مع PSNR
        ax = axes[1]
        fvals = [base['min_flops'], r['min_flops']]
        pps   = [base['psnr_lo'],  r['psnr_lo']]
        bars = ax.bar(lbls, fvals, color=['#d62728', 'green'],
                      edgecolor='black', alpha=0.88, width=0.55)
        for bar, gf, p in zip(bars, fvals, pps):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(fvals)*0.02,
                    f'{gf:.2f}G\n{p:.2f} dB',
                    ha='center', fontsize=10)
        delta_f = ((r['min_flops'] - base['min_flops'])
                   / base['min_flops'] * 100)
        ax.set_title(f'Fewest FLOPs — Δ = {delta_f:+.1f}%')
        ax.set_ylabel('Minimum FLOPs (G)')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'C2_FLOPs — {r["label"]} vs {_label(BASELINE_KEY)}',
                     fontsize=13, y=1.02)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', r['model'])
        save_figure(fig, f"C2_FLOPs_{safe}_vs_baseline")

# ═════════════════════════════════════════════════════════════════════
# C2 — رسم مستقل لكل موديل: Best PSNR + Fewest params vs baseline
# ═════════════════════════════════════════════════════════════════════
def plot_C2(summ):
    """
    يُنتج:
      C2_all_models.*                     ← نظرة شاملة
      C2_<model>_vs_baseline.*            ← شكل مستقل لكل موديل
    """
    if BASELINE_KEY not in summ['model'].values:
        print("   ⚠️  No baseline — skip C2.")
        return
    base = summ[summ['model'] == BASELINE_KEY].iloc[0]

    # ─── (أ) نظرة شاملة ──────────────────────────────────────────
    labels  = summ['label'].tolist()
    best_p  = summ['best_psnr'].tolist()
    best_np = summ['best_params'].tolist()
    min_np  = summ['min_params'].tolist()
    psnr_mp = summ['psnr_at_min_params'].tolist()
    colors  = ['#d62728' if m == BASELINE_KEY else 'steelblue'
               for m in summ['model']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    bars = ax.bar(range(len(labels)), best_p, color=colors,
                  edgecolor='black', alpha=0.88)
    for bar, np_, p in zip(bars, best_np, best_p):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f'{p:.3f} dB\n{np_/1e6:.2f}M',
                ha='center', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('Best PSNR (dB)')
    ax.set_title('Best PSNR per model')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    bars = ax.bar(range(len(labels)), np.array(min_np) / 1e6,
                  color=['#d62728' if m == BASELINE_KEY else 'green'
                         for m in summ['model']],
                  edgecolor='black', alpha=0.88)
    for bar, np_, p in zip(bars, min_np, psnr_mp):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(min_np)/1e6 * 0.02,
                f'{np_/1e6:.2f}M\n{p:.2f} dB',
                ha='center', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('Minimum #Parameters (M)')
    ax.set_title('Fewest parameters per model')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('C2 — Best PSNR & Fewest Parameters — all models',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "C2_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────────
    for _, r in summ.iterrows():
        if r['model'] == BASELINE_KEY:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # يسار: Best PSNR
        ax = axes[0]
        lbls  = [f"{_label(BASELINE_KEY)}\n(baseline)", r['label']]
        vals  = [base['best_psnr'], r['best_psnr']]
        pars  = [base['best_params'], r['best_params']]
        clrs  = ['#d62728', 'steelblue']
        bars  = ax.bar(lbls, vals, color=clrs,
                       edgecolor='black', alpha=0.88, width=0.55)
        for bar, np_, p in zip(bars, pars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.03,
                    f'{p:.3f} dB\n{np_/1e6:.2f}M',
                    ha='center', fontsize=10)
        delta = r['best_psnr'] - base['best_psnr']
        ax.set_title(f'Best PSNR — Δ = {delta:+.3f} dB')
        ax.set_ylabel('PSNR (dB)')
        ax.grid(True, alpha=0.3, axis='y')
        ylo = min(vals) - 0.4
        yhi = max(vals) + 0.6
        ax.set_ylim(ylo, yhi)

        # يمين: Fewest params
        ax = axes[1]
        pvals = [base['min_params']/1e6, r['min_params']/1e6]
        pp    = [base['psnr_at_min_params'], r['psnr_at_min_params']]
        bars  = ax.bar(lbls, pvals, color=['#d62728', 'green'],
                       edgecolor='black', alpha=0.88, width=0.55)
        for bar, np_, p in zip(bars, pvals, pp):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(pvals)*0.02,
                    f'{np_:.2f}M\n{p:.2f} dB',
                    ha='center', fontsize=10)
        delta_p = (r['min_params'] - base['min_params']) / base['min_params'] * 100
        ax.set_title(f'Fewest params — Δ = {delta_p:+.1f}%')
        ax.set_ylabel('Minimum #Parameters (M)')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'C2 — {r["label"]} vs {_label(BASELINE_KEY)}',
                     fontsize=13, y=1.02)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', r['model'])
        save_figure(fig, f"C2_{safe}_vs_baseline")


# ═════════════════════════════════════════════════════════════════════
# C3 — رسم مستقل لكل موديل: ΔPSNR + ΔParams% vs baseline
# ═════════════════════════════════════════════════════════════════════
def plot_C3(summ):
    """
    يُنتج:
      C3_all_models.*                     ← نظرة شاملة
      C3_<model>_vs_baseline.*            ← شكل مستقل لكل موديل
    """
    rows = summ[summ['model'] != BASELINE_KEY].copy()
    if rows.empty:
        print("   ℹ️  No non-baseline models — skip C3.")
        return

    # ─── (أ) نظرة شاملة ──────────────────────────────────────────
    labels = rows['label'].tolist()
    d_psnr = rows['Δpsnr_vs_base'].tolist()
    d_par  = rows['Δparams_pct'].tolist()
    colors = ['green' if (a >= 0 and b <= 0) else 'red'
              for a, b in zip(d_psnr, d_par)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]
    bars = ax.barh(labels, d_psnr, color=colors,
                   edgecolor='black', alpha=0.88)
    for bar, v in zip(bars, d_psnr):
        ax.text(v + (0.005 if v >= 0 else -0.005),
                bar.get_y() + bar.get_height()/2,
                f'{v:+.3f} dB', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('ΔPSNR vs baseline (dB)')
    ax.set_title(f'ΔPSNR vs {_label(BASELINE_KEY)}')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    bars = ax.barh(labels, d_par, color=colors,
                   edgecolor='black', alpha=0.88)
    for bar, v in zip(bars, d_par):
        ax.text(v + (0.5 if v >= 0 else -0.5),
                bar.get_y() + bar.get_height()/2,
                f'{v:+.1f}%', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('ΔParameters vs baseline (%)')
    ax.set_title(f'ΔParams% vs {_label(BASELINE_KEY)}')
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('C3 — Comparison against baseline — all models',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, "C3_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────────
    for _, r in rows.iterrows():
        fig, ax = plt.subplots(figsize=(9, 6))

        metrics = ['ΔPSNR\n(dB)', 'ΔParams\n(%)']
        values  = [r['Δpsnr_vs_base'], r['Δparams_pct']]
        bar_colors = [
            'green' if r['Δpsnr_vs_base'] >= 0 else 'red',
            'green' if r['Δparams_pct'] <= 0 else 'red',
        ]
        bars = ax.bar(metrics, values, color=bar_colors,
                      edgecolor='black', alpha=0.88, width=0.5)
        for bar, v in zip(bars, values):
            y = v + (0.02 * (1 if v >= 0 else -1))
            ax.text(bar.get_x() + bar.get_width()/2, y,
                    f'{v:+.3f}' if abs(v) < 10 else f'{v:+.1f}',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', lw=1)
        ax.set_title(f'C3 — {r["label"]} vs {_label(BASELINE_KEY)}')
        ax.set_ylabel('Δ vs baseline')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', r['model'])
        save_figure(fig, f"C3_{safe}_vs_baseline")


# ═════════════════════════════════════════════════════════════════════
# SECTION 10 — E1: Best-so-far convergence per model
# ═════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════
# E1 — Best-so-far convergence — baseline + موديل واحد لكل شكل
# ═════════════════════════════════════════════════════════════════════
def _load_bsf(db_path):
    """يرجع (trials, best_so_far) لموديل واحد، أو (None, None)."""
    if db_path is None:
        return None, None
    try:
        names = [s.study_name for s in
                 optuna.study.get_all_study_summaries(
                     storage=f"sqlite:///{db_path}")]
        if not names:
            return None, None
        study = optuna.load_study(study_name=names[0],
                                  storage=f"sqlite:///{db_path}")
    except Exception:
        return None, None
    vals = [(t.number, float(t.value))
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None]
    if not vals:
        return None, None
    vals.sort()
    trials = np.array([v[0] for v in vals])
    psnr   = np.array([v[1] for v in vals])
    return trials, np.maximum.accumulate(psnr)


def plot_E1(models):
    """
    يُنتج:
      E1_all_models.*                     ← نظرة شاملة
      E1_<model>_vs_baseline.*            ← شكل مستقل لكل موديل
    """
    # اقرأ best-so-far لكل موديل أولاً
    bsf_data = {}
    for key, info in models.items():
        t, b = _load_bsf(info['db_path'])
        if t is not None:
            bsf_data[key] = (t, b)

    if not bsf_data:
        print("   ℹ️  No DB data — skip E1.")
        return

    # ─── (أ) نظرة شاملة ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap('tab10')
    for i, (key, (t, b)) in enumerate(bsf_data.items()):
        is_base = (key == BASELINE_KEY)
        ax.plot(t, b,
                lw=2.4 if is_base else 1.4,
                marker='o' if is_base else None,
                ms=4 if is_base else 0,
                color='black' if is_base else cmap(i),
                alpha=0.9 if is_base else 0.75,
                label=_label(key) + (" (baseline)" if is_base else ""))
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Best-so-far PSNR (dB)')
    ax.set_title('E1 — Best-so-far convergence — all models')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    fig.tight_layout()
    save_figure(fig, "E1_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────────
    if BASELINE_KEY not in bsf_data:
        print("   ⚠️  No baseline in DB data — skip per-model E1.")
        return
    base_t, base_b = bsf_data[BASELINE_KEY]

    for key, (t, b) in bsf_data.items():
        if key == BASELINE_KEY:
            continue

        fig, ax = plt.subplots(figsize=(11, 7))

        # baseline
        ax.plot(base_t, base_b, '-o', color='black', lw=2.4,
                ms=4, alpha=0.9,
                label=f'{_label(BASELINE_KEY)} (baseline)')

        # الموديل الحالي
        ax.plot(t, b, '-s', color='steelblue', lw=2.0,
                ms=4, alpha=0.9,
                label=_label(key))

        # ملاحظة الفرق النهائي
        diff = b[-1] - base_b[-1]
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Best-so-far PSNR (dB)')
        ax.set_title(f'E1 — {_label(key)} vs {_label(BASELINE_KEY)} '
                     f'(final Δ = {diff:+.3f} dB)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', key)
        save_figure(fig, f"E1_{safe}_vs_baseline")

# ═════════════════════════════════════════════════════════════════════
# SECTION 11 — E2/E3: Params vs PSNR colored by n_feats / num_blocks
# ═════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════
# E2/E3 — Params vs PSNR ملوّنة بـ HP — شكل لكل موديل
# ═════════════════════════════════════════════════════════════════════
def _plot_colored_by_per_model(trial_df, hp_col, fname_base):
    """لكل موديل: scatter ملوّن بالـ HP + طبقة baseline رمادية للخلفية."""
    if hp_col not in trial_df.columns:
        print(f"   ℹ️  Column '{hp_col}' missing — skip {fname_base}.")
        return

    if BASELINE_KEY not in trial_df['model'].unique():
        return
    base = trial_df[trial_df['model'] == BASELINE_KEY]

    for key, sub in trial_df.groupby('model'):
        if key == BASELINE_KEY:
            continue
        s = sub[hp_col]
        is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 6

        fig, ax = plt.subplots(figsize=(11, 7))

        # baseline رمادي خفيف للخلفية
        ax.scatter(base['n_params']/1e6, base['psnr'],
                   s=45, color='lightgray', alpha=0.7,
                   edgecolors='gray', linewidths=0.4,
                   label=_label(BASELINE_KEY) + " (bg)", zorder=2)

        if is_num:
            sc = ax.scatter(sub['n_params']/1e6, sub['psnr'],
                            c=s.astype(float), cmap='plasma',
                            s=100, alpha=0.9, edgecolors='black',
                            linewidths=0.7, zorder=4)
            plt.colorbar(sc, ax=ax, label=hp_col)
        else:
            uniq = sorted(s.dropna().unique().tolist(),
                          key=lambda x: (str(type(x)), x))
            cmap = cm.get_cmap('tab10', max(len(uniq), 2))
            for i, u in enumerate(uniq):
                m = (s == u)
                ax.scatter(sub.loc[m, 'n_params']/1e6,
                           sub.loc[m, 'psnr'],
                           color=cmap(i), s=100, alpha=0.9,
                           edgecolors='black', linewidths=0.7,
                           label=str(u), zorder=4)
            ax.legend(title=hp_col, loc='lower right', fontsize=8)

        ax.set_xlabel('Number of parameters (M)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{fname_base.replace("_", " ")} — '
                     f'{_label(key)} (colored by {hp_col})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', key)
        save_figure(fig, f"{fname_base}_{safe}")


def plot_E2_E3(trial_df):
    # استخدم أول عمود HP متوفر لكل خاصية
    for col in ('hp_n_feats', 'hp_nf'):
        _plot_colored_by_per_model(trial_df, col, "E2_params_vs_psnr_by_nfeats")
    _plot_colored_by_per_model(trial_df, 'hp_num_blocks',
                               "E3_params_vs_psnr_by_numblocks")

# ═════════════════════════════════════════════════════════════════════
# E4 — PSNR vs training time — شكل مستقل لكل موديل
# ═════════════════════════════════════════════════════════════════════
def _load_time_and_psnr(db_path):
    if db_path is None:
        return None, None
    try:
        names = [s.study_name for s in
                 optuna.study.get_all_study_summaries(
                     storage=f"sqlite:///{db_path}")]
        if not names:
            return None, None
        study = optuna.load_study(study_name=names[0],
                                  storage=f"sqlite:///{db_path}")
    except Exception:
        return None, None
    df = study.trials_dataframe()
    if 'state' not in df.columns or 'value' not in df.columns:
        return None, None
    df = df[df['state'] == 'COMPLETE'].copy()
    if 'duration' not in df.columns:
        return None, None
    dur = pd.to_timedelta(df['duration']).dt.total_seconds() / 60.0
    return dur.values, df['value'].values


def plot_E4(models):
    """
    يُنتج:
      E4_all_models.*                     ← نظرة شاملة
      E4_<model>_vs_baseline.*            ← شكل مستقل لكل موديل
    """
    data = {}
    for key, info in models.items():
        t, p = _load_time_and_psnr(info['db_path'])
        if t is not None and len(t) > 0:
            data[key] = (t, p)

    if not data:
        print("   ℹ️  No duration data — skip E4.")
        return

    # ─── (أ) نظرة شاملة ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap('tab10')
    for i, (key, (t, p)) in enumerate(data.items()):
        is_base = (key == BASELINE_KEY)
        ax.scatter(t, p,
                   s=110 if is_base else 70,
                   marker='s' if is_base else 'o',
                   color='black' if is_base else cmap(i),
                   alpha=0.9 if is_base else 0.75,
                   edgecolors='black', linewidths=0.6,
                   label=_label(key) + (" (baseline)" if is_base else ""))
    ax.set_xlabel('Training time per trial (minutes)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('E4 — PSNR vs training time — all models')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    fig.tight_layout()
    save_figure(fig, "E4_all_models")

    # ─── (ب) رسم مستقل لكل موديل ─────────────────────────────────
    if BASELINE_KEY not in data:
        return
    bt, bp = data[BASELINE_KEY]

    for key, (t, p) in data.items():
        if key == BASELINE_KEY:
            continue

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.scatter(bt, bp, s=90, marker='s', color='black',
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=f'{_label(BASELINE_KEY)} (baseline)')
        ax.scatter(t, p, s=80, marker='o', color='steelblue',
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=_label(key))
        ax.set_xlabel('Training time per trial (minutes)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'E4 — {_label(key)} vs {_label(BASELINE_KEY)}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        fig.tight_layout()

        safe = re.sub(r'[^\w\-]+', '_', key)
        save_figure(fig, f"E4_{safe}_vs_baseline")


# ═════════════════════════════════════════════════════════════════════
# SECTION 13 — E5 (اختياري): FLOPs vs PSNR
# ═════════════════════════════════════════════════════════════════════
def plot_E5(per_model, summ):
    """
    يحسب FLOPs لكل موديل بأفضل trial لديه باستخدام fvcore.
    يحتاج:
      - fvcore مثبّت:  pip install fvcore
      - توفر دالة build_model(key, params) في هذا الملف
    """
    if not FLOPs_ENABLED:
        print("   ℹ️  FLOPs disabled (FLOPs_ENABLED=False) — skip E5.")
        return

    try:
        from fvcore.nn import FlopCountAnalysis
        import torch
    except ImportError:
        print("   ⚠️  fvcore not installed — skip E5. "
              "Install: pip install fvcore")
        return

    # 🔴 يجب تعديل هذه الدالة حسب كل موديل
    def build_model(key, params):
        raise NotImplementedError(
            "Set build_model() inside plot_E5() before enabling FLOPs."
        )

    flops_rows = []
    for key, df in per_model.items():
        i_hi = df['psnr'].idxmax()
        row = df.loc[i_hi]
        try:
            model = build_model(key, row)
            model.eval()
            x = torch.randn(*INPUT_SIZE_FOR_FLOPs if False else
                            INPUT_SIZE_FOR_FLOPs)
            fca = FlopCountAnalysis(model, x)
            flops = fca.total() / 1e9        # GFLOPs
            flops_rows.append({
                'model': key,
                'label': _label(key),
                'flops_G': flops,
                'psnr': float(row['psnr']),
                'n_params': int(row['n_params']),
            })
        except Exception as e:
            print(f"   ⚠️  {key}: FLOPs failed — {e}")

    if not flops_rows:
        return

    fdf = pd.DataFrame(flops_rows)
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap('tab10')
    for i, r in fdf.iterrows():
        is_base = (r['model'] == BASELINE_KEY)
        ax.scatter(r['flops_G'], r['psnr'],
                   s=180 if is_base else 120,
                   marker='s' if is_base else 'o',
                   color='black' if is_base else cmap(i),
                   edgecolors='black', linewidths=0.8, alpha=0.9)
        ax.annotate(r['label'], (r['flops_G'], r['psnr']),
                    xytext=(6, 4), textcoords='offset points', fontsize=9)
    ax.set_xlabel('FLOPs (GFLOPs, @ 48×48 LR input)')
    ax.set_ylabel('Best PSNR (dB)')
    ax.set_title('E5 — FLOPs vs Best PSNR')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "E5_flops_vs_psnr")


# ═════════════════════════════════════════════════════════════════════
# SECTION 14 — Excel exports
# ═════════════════════════════════════════════════════════════════════
def export_excel(summ, trial_df, per_model):
    # ═══ Sheet 1: model summary (slides 10 & 11) ═══
    s1 = summ[[
        'label', 'model', 'n_trials',
        'best_psnr', 'best_params', 'best_trial',
        'min_params', 'psnr_at_min_params', 'min_params_trial',
        'trade_psnr', 'trade_params', 'trade_trial',
        'Δpsnr_vs_base', 'Δparams_vs_base',
        'Δparams_pct', 'Δmin_par_pct',
    ]].copy()
    s1.columns = [
        'Model', 'Key', 'Trials',
        'Best PSNR (dB)', 'Params @ best', 'Best trial',
        'Min params', 'PSNR @ min', 'Min params trial',
        'Trade-off PSNR', 'Trade-off params', 'Trade-off trial',
        'ΔPSNR vs base (dB)', 'ΔParams vs base',
        'ΔParams %', 'ΔMinParams %',
    ]

    # ═══ Sheet 2: raw trial-level ═══
    s2 = trial_df.copy()

    # ═══ Sheet 3: long-format for slides ═══
    long_rows = []
    for _, r in summ.iterrows():
        long_rows.append({
            'Model': r['label'],
            'Selection': 'Highest PSNR',
            'PSNR (dB)': r['best_psnr'],
            'Params (M)': r['best_params'] / 1e6,
            'Trial': r['best_trial'],
            'ΔPSNR (dB)': r['Δpsnr_vs_base'],
            'ΔParams %': r['Δparams_pct'],
        })
        long_rows.append({
            'Model': r['label'],
            'Selection': 'Best trade-off (≤ 0.1 dB)',
            'PSNR (dB)': r['trade_psnr'],
            'Params (M)': (r['trade_params'] / 1e6
                           if pd.notna(r['trade_params']) else np.nan),
            'Trial': r['trade_trial'],
            'ΔPSNR (dB)': (r['trade_psnr'] - r['best_psnr']
                           if pd.notna(r['trade_psnr']) else np.nan),
            'ΔParams %': ((r['trade_params'] / r['best_params'] - 1) * 100
                          if pd.notna(r['trade_params']) else np.nan),
        })
        long_rows.append({
            'Model': r['label'],
            'Selection': 'Fewest parameters',
            'PSNR (dB)': r['psnr_at_min_params'],
            'Params (M)': r['min_params'] / 1e6,
            'Trial': r['min_params_trial'],
            'ΔPSNR (dB)': r['psnr_at_min_params'] - r['best_psnr'],
            'ΔParams %': (r['min_params'] / r['best_params'] - 1) * 100,
        })
    s3 = pd.DataFrame(long_rows)

    # ═══ Sheet 4: FLOPs summary (اختياري) ═══
    s4 = None
    flops_rows = []
    for key, df in per_model.items():
        if 'flops_G' not in df.columns:
            continue
        d = df.dropna(subset=['flops_G'])
        if d.empty:
            continue
        i_hi = d['psnr'].idxmax()
        i_lo = d['flops_G'].idxmin()
        flops_rows.append({
            'Model':            key,
            'Label':            _label(key),
            'Best PSNR (dB)':   float(d.loc[i_hi, 'psnr']),
            'FLOPs @ best (G)': float(d.loc[i_hi, 'flops_G']),
            'Min FLOPs (G)':    float(d.loc[i_lo, 'flops_G']),
            'PSNR @ min (dB)':  float(d.loc[i_lo, 'psnr']),
        })
    if flops_rows:
        s4 = pd.DataFrame(flops_rows)

    # ═══ اكتب كل الأوراق معاً ═══
    out = OUTPUT_DIR / "models_comparison_summary.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        s1.to_excel(w, sheet_name='model_summary', index=False)
        s3.to_excel(w, sheet_name='slides_10_11',  index=False)
        s2.to_excel(w, sheet_name='trial_level',   index=False)
        if s4 is not None:
            s4.to_excel(w, sheet_name='flops_summary', index=False)
    print(f"   ✅ models_comparison_summary.xlsx "
          f"({4 if s4 is not None else 3} sheets)")

    # ملف مستقل بالـ trial-level
    trial_df.to_excel(OUTPUT_DIR / "trial_level_all_models.xlsx", index=False)
    print(f"   ✅ trial_level_all_models.xlsx")


# ═════════════════════════════════════════════════════════════════════
# SECTION 15 — README
# ═════════════════════════════════════════════════════════════════════
def write_readme(summ):
    lines = [
        "Model Comparison — Auto-Generated",
        "=" * 70, "",
        f"Baseline: {_label(BASELINE_KEY)} ({BASELINE_KEY})",
        f"Models  : {len(summ)}",
        "",
        "Files in this folder:",
        "  C1_params_vs_psnr_all_models.{pdf,png,jpeg}",
        "  C2_best_psnr_and_fewest_params.{pdf,png,jpeg}",
        "  C3_delta_vs_baseline.{pdf,png,jpeg}",
        "  E1_best_so_far_convergence.{pdf,png,jpeg}",
        "  E2_params_vs_psnr_by_nfeats.{pdf,png,jpeg}",
        "  E3_params_vs_psnr_by_numblocks.{pdf,png,jpeg}",
        "  E4_psnr_vs_training_time.{pdf,png,jpeg}",
        "  E5_flops_vs_psnr.{pdf,png,jpeg}  (optional)",
        "  models_comparison_summary.xlsx    (3 sheets)",
        "  trial_level_all_models.xlsx",
        "",
        "Summary table:",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"  {r['label']:<25} "
            f"best={r['best_psnr']:.3f} dB @ {r['best_params']/1e6:.2f}M  "
            f"min={r['min_params']/1e6:.2f}M @ {r['psnr_at_min_params']:.3f} dB  "
            f"ΔPSNR={r['Δpsnr_vs_base']:+.3f} dB  "
            f"ΔPar={r['Δparams_pct']:+.1f}%"
        )
    (OUTPUT_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"   ✅ README.txt")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    global BASELINE_KEY          # ← أضف هذا السطر
    print("=" * 78)
    print("📊 Multi-Model Comparison")
    print(f"📂 Root  : {HERE}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"🎯 Baseline: {BASELINE_KEY}")
    print("=" * 78)

    # ─── 1) اكتشاف الموديلات ───
    print("\n🔎 Discovering models (recursive, key = top-level folder)...")
    models = discover_models()
    if not models:
        print("❌ لم يُعثر على أي مجلد params_vs_psnr_analysis")
        sys.exit(1)

    print(f"   Found {len(models)} model(s):\n")
    for k, info in models.items():
        rel_top = info['top_dir'].relative_to(HERE)
        rel_deep = info['model_dir'].relative_to(HERE)
        print(f"     • key  = {k!r}")
        print(f"       top  = {rel_top}")
        if rel_top != rel_deep:
            print(f"       deep = {rel_deep}   (where DB lives)")
        print(f"       src  = {info['src_file'].name}")
    print()

    if BASELINE_KEY not in models:
        print(f"⚠️  Baseline key {BASELINE_KEY!r} not found.")
        print(f"   Available keys: {list(models.keys())}")
        print(f"   Update BASELINE_KEY in SECTION 1 to one of the above.")
        BASELINE_KEY = list(models.keys())[0]
        print(f"   → Using {BASELINE_KEY!r} as fallback baseline.\n")

    # ─── 2) تحميل البيانات ───
    print("\n📥 Loading per-model trial tables...")
    trial_df, per_model = load_model_tables(models)
    if trial_df.empty:
        print("❌ No usable data.")
        sys.exit(1)

    # ─── 3) ملخص ───
    print("\n📋 Building summary table...")
    summ = build_summary(per_model)
    print(summ[['label', 'n_trials', 'best_psnr', 'best_params',
                'min_params', 'Δpsnr_vs_base', 'Δparams_pct']]
          .to_string(index=False))

    # ─── 4) الرسومات ───
    print("\n🎨 Generating figures (params-based)...")
    plot_C1(per_model)
    plot_C2(summ)
    plot_C3(summ)
    plot_E1(models)
    plot_E2_E3(trial_df)
    plot_E4(models)
    plot_E5(per_model, summ)

    print("\n🎨 Generating figures (FLOPs-based)...")
    plot_C1_flops(per_model)
    plot_C2_flops(per_model, summ)

    # ─── 5) Excel ───
    print("\n📄 Writing Excel summaries...")
    export_excel(summ, trial_df, per_model)

    # ─── 6) README ───
    write_readme(summ)

    print("\n" + "=" * 78)
    print("🎉 Done")
    print(f"📂 {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()