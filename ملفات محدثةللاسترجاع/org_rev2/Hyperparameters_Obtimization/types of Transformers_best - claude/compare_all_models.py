# =====================================================================
# compare_all_models.py — Universal multi-model comparison   (v2 — corrected)
# =====================================================================
# مكان الملف: المجلد الأب الذي يحتوي مجلدات الموديلات.
#
# البنية المتوقعة:
#   parent/
#   ├── compare_all_models.py           ← هذا الملف
#   ├── <baseline folder>/
#   │   ├── optuna_study.db
#   │   └── params_vs_psnr_analysis/
#   │       ├── <name>_hutcn_hyperopt_study_3cols.xlsx
#   │       └── <name>_hutcn_hyperopt_study_full.xlsx
#   ├── <model B folder>/ ...
#   └── ...
#
# المخرجات (في ./comparison_output/) — كل شكل بصيغ pdf/png/jpeg:
#   C1_all_models, C1_<model>_vs_baseline              Params vs PSNR + جبهة Pareto (درجية)
#   C1_FLOPs_all_models, C1_FLOPs_<model>_vs_baseline  FLOPs  vs PSNR
#   C2_all_models, C2_<model>_vs_baseline              أعلى PSNR + أقل Params
#   C2_FLOPs_all_models, C2_FLOPs_<model>_vs_baseline  أعلى PSNR + أقل FLOPs
#   C3_all_models, C3_<model>_vs_baseline              ΔPSNR / ΔParams / ΔFLOPs مقابل baseline
#   E1_all_models, E1_<model>_vs_baseline              best-so-far (درجي)
#   E2_params_vs_psnr_by_nfeats_<model>                ملوّن بـ n_feats / nf
#   E3_params_vs_psnr_by_numblocks_<model>             ملوّن بـ num_blocks
#   E4_all_models, E4_<model>_vs_baseline              PSNR مقابل زمن التدريب
#   models_comparison_summary.xlsx, trial_level_all_models.xlsx, README.txt
#
# قواعد الاختيار (موحّدة في كل الملف):
#   • Highest PSNR      : أعلى PSNR (عند التعادل: الأقل تكلفة)
#   • Fewest params/FLOPs: أقل تكلفة (عند التعادل: الأعلى PSNR)
#   • Best trade-off    : أصغر موديل ضمن TOLERANCE_DB من أعلى PSNR لنفس الموديل
#   • Pareto front      : أعلى PSNR يمكن بلوغه بميزانية لا تتجاوز x  → رسم درجي (staircase)
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
# 🟢 اسم مجلد المخرجات. غيّره عند تشغيل المقارنة بـ baseline آخر (مثلاً موديل بنفس الهيكل)
#    حتى لا تُكتب النتائج فوق نتائج المقارنة السابقة.
OUTPUT_SUBDIR = "comparison_output"
OUTPUT_DIR = HERE / OUTPUT_SUBDIR
OUTPUT_DIR.mkdir(exist_ok=True)

# 🔴 مفتاح baseline = **اسم المجلد الرئيسي** (الابن المباشر لـ HERE)
#    مثال: لو الـ baseline داخل:
#      Org_rev2_best/DHTCUN+50T+50EP+optim_purn/params_vs_psnr_analysis/
#    فالمفتاح = "Org_rev2_best"  (وليس "DHTCUN+50T+50EP+optim_purn")
BASELINE_KEY = "Org_rev2_best"           # ← غيّره لو اختلف الاسم

# 🟢 تشغيلات المقارنة: كل عنصر = (مفتاح الـ baseline ، اسم مجلد المخرجات). تُنفَّذ بالترتيب.
#    1) DHTCUN الأصلي كما طلب المشرف.
#    2) (اختياري، مُفعَّل افتراضياً) مقارنة بنفس الهيكل: الموديل المرجعي يستخدم كتلة P_HTCB
#       واحدة ومسار TCN واحد مثل باقي الموديلات، أما DHTCUN الأصلي فـ 5 كتل بمسارين TCN.
#       احذف السطر الثاني إن لم تحتجه، أو غيّر المفتاح إن اختلف اسم مجلد موديل SwinT.
BASELINE_RUNS = [
    (BASELINE_KEY, "comparison_output"),
    ("Swint_2",    "comparison_output_vs_Swint_2"),
]

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

# 🟢 FLOPs: تُقرأ من ملفات الإكسل (عمود flops_G) بدون إعادة بناء
INPUT_SIZE_FOR_FLOPs = (1, 3, 48, 48)


# 🟢 عمود عدد البارامترات المستخدم في كل الأشكال والجداول:
#    "n_params"           = كل بارامترات النموذج (المتعارف عليه)
#    "n_params_effective" = البارامترات التي تصلها gradients فقط (يستثني الطبقات/الـ blocks الميتة)
#    (العمود الثاني يُنتجه analyze_params_vs_psnr.py v2 داخل ملف *_full.xlsx)
PARAMS_COLUMN = "n_params"

# 🟢 قاعدة "أفضل توازن": أصغر موديل ضمن هذا الفارق (dB) من أعلى PSNR لنفس الموديل.
#    (التجارب المتطابقة تختلف عادة 0.01–0.03 dB، فـ 0.10 هامش آمن.)
TOLERANCE_DB = 0.10

# 🟢 محور x في شكل E1:  "completed_index" (ترتيب التجربة بين المكتملة فقط — عادل
#    عند اختلاف عدد المقطوعة)   |   "trial_number" (رقم trial في Optuna)
E1_X_AXIS = "completed_index"

# 🟢 اسم الدراسة المفضّل عند وجود أكثر من دراسة في قاعدة واحدة
STUDY_NAME_PREFERRED = "hutcn_hyperopt_study"

# 🟢 ألوان: أزرق = أفضل من baseline ، برتقالي = أسوأ (آمنة لعمى الألوان)
COL_GOOD, COL_BAD, COL_BASE, COL_MODEL = '#0072B2', '#D55E00', 'black', 'steelblue'

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

        xlsx_3cols = sorted(ana.glob("*_3cols.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        xlsx_full  = sorted(ana.glob("*_full.xlsx"),  key=lambda p: p.stat().st_mtime, reverse=True)
        for _grp in (xlsx_3cols, xlsx_full):
            if len(_grp) > 1:
                print(f"   ⚠️  {root_path.name}: {len(_grp)} Excel files of the same kind — "
                      f"using the NEWEST: {_grp[0].name}")
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
                 أعمدة: model, trial, n_params, psnr, (flops_G), (HP columns)
      per_model: dict key → DataFrame لكل موديل
    """
    frames = []
    per_model = {}

    for key, info in models.items():
        # فضّل full (فيه كل HP و FLOPs)
        src = info['full_xlsx'] or info['3cols_xlsx']
        if src is None:
            print(f"   ⚠️  {key}: no excel — skipped")
            continue
        try:
            df = pd.read_excel(src)
        except Exception as e:
            print(f"   ⚠️  {key}: cannot read {src.name} — {e}")
            continue

        # توحيد أسماء الأعمدة (يشمل ترويسة FLOPs في ملفات 3cols)
        rename = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ('trial number', 'trial', 'number'):
                rename[c] = 'trial'
            elif cl in ('number of parameters', 'n_params', 'params'):
                rename[c] = 'n_params'
            elif cl in ('psnr (db)', 'psnr', 'value'):
                rename[c] = 'psnr'
            elif cl == 'flops_g' or cl.startswith('flops (g)'):
                rename[c] = 'flops_G'
        df = df.rename(columns=rename)

        if PARAMS_COLUMN != 'n_params':
            if PARAMS_COLUMN in df.columns and df[PARAMS_COLUMN].notna().any():
                df['n_params_declared'] = df['n_params']
                df['n_params'] = df[PARAMS_COLUMN]
            else:
                print(f"   ⚠️  {key}: column '{PARAMS_COLUMN}' not found — using n_params")

        needed = {'trial', 'n_params', 'psnr'}
        if not needed.issubset(df.columns):
            print(f"   ⚠️  {key}: missing columns {needed - set(df.columns)} — skipped")
            continue

        n0 = len(df)
        df = df.dropna(subset=['n_params', 'psnr']).copy()
        if len(df) < n0:
            print(f"   ⚠️  {key}: dropped {n0 - len(df)} row(s) with empty n_params/PSNR")
        if df['trial'].duplicated().any():
            print(f"   ⚠️  {key}: duplicated trial numbers in {src.name}")

        df = df[['trial', 'n_params', 'psnr'] +
                [c for c in df.columns
                 if c not in ('trial', 'n_params', 'psnr')]].copy()
        df.insert(0, 'model', key)
        df = df.sort_values('trial').reset_index(drop=True)

        per_model[key] = df
        frames.append(df)
        has_f = 'flops_G' in df.columns and df['flops_G'].notna().any()
        print(f"   ✅ {key:<20} → {len(df):>3} trials | "
              f"n_params ∈ [{df['n_params'].min()/1e6:.2f}M, "
              f"{df['n_params'].max()/1e6:.2f}M] | FLOPs: {'yes' if has_f else 'no'}")

    trial_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trial_df, per_model


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Summary table (matches slides 10 & 11)
# ═════════════════════════════════════════════════════════════════════
def build_summary(per_model):
    """
    جدول ملخص لكل موديل (كل Δ معنونة بمرجعها صراحةً):
      - best_psnr + params @ best
      - min_params + psnr @ min_params   (عند التعادل: الأعلى PSNR)
      - trade-off: أصغر موديل ضمن TOLERANCE_DB من أعلى PSNR لنفس الموديل
      - Δ مقابل baseline  (…_vs_base)  و Δ مقابل أعلى PSNR للموديل نفسه (…_vs_own)
    """
    rows = []
    for key, df in per_model.items():
        hi = df.loc[pick_best(df)]
        lo = df.loc[pick_cheapest(df)]
        cand = df[df['psnr'] >= float(hi['psnr']) - TOLERANCE_DB]
        tr = df.loc[pick_cheapest(cand)]

        row = {
            'model': key, 'label': _label(key), 'n_trials': len(df),
            'best_psnr': float(hi['psnr']), 'best_params': int(hi['n_params']),
            'best_trial': int(hi['trial']),
            'min_params': int(lo['n_params']), 'psnr_at_min_params': float(lo['psnr']),
            'min_params_trial': int(lo['trial']),
            'trade_psnr': float(tr['psnr']), 'trade_params': int(tr['n_params']),
            'trade_trial': int(tr['trial']),
        }
        if 'flops_G' in df.columns and df['flops_G'].notna().any():
            row['best_flops'] = float(hi['flops_G']) if pd.notna(hi['flops_G']) else np.nan
        else:
            row['best_flops'] = np.nan
        rows.append(row)

    summ = pd.DataFrame(rows)

    # Δ مقابل أعلى PSNR للموديل نفسه
    summ['Δtrade_psnr_vs_own'] = summ['trade_psnr'] - summ['best_psnr']
    summ['Δtrade_params_pct_vs_own'] = (summ['trade_params'] / summ['best_params'] - 1) * 100
    summ['Δmin_psnr_vs_own'] = summ['psnr_at_min_params'] - summ['best_psnr']
    summ['Δmin_params_pct_vs_own'] = (summ['min_params'] / summ['best_params'] - 1) * 100

    # Δ مقابل baseline (مرجع: أعلى PSNR للـ baseline، وأقل params للـ baseline)
    cols_base = ['Δpsnr_vs_base', 'Δparams_vs_base', 'Δparams_pct', 'Δmin_par_pct',
                 'Δtrade_psnr_vs_base', 'Δtrade_params_pct_vs_base',
                 'Δmin_psnr_vs_base_min', 'Δbest_flops_pct']
    if BASELINE_KEY in summ['model'].values:
        base = summ[summ['model'] == BASELINE_KEY].iloc[0]
        summ['Δpsnr_vs_base'] = summ['best_psnr'] - base['best_psnr']
        summ['Δparams_vs_base'] = summ['best_params'] - base['best_params']
        summ['Δparams_pct'] = (summ['best_params'] / base['best_params'] - 1) * 100
        summ['Δmin_par_pct'] = (summ['min_params'] / base['min_params'] - 1) * 100
        summ['Δtrade_psnr_vs_base'] = summ['trade_psnr'] - base['best_psnr']
        summ['Δtrade_params_pct_vs_base'] = (summ['trade_params'] / base['best_params'] - 1) * 100
        summ['Δmin_psnr_vs_base_min'] = summ['psnr_at_min_params'] - base['psnr_at_min_params']
        summ['Δbest_flops_pct'] = (summ['best_flops'] / base['best_flops'] - 1) * 100
    else:
        for c in cols_base:
            summ[c] = np.nan

    # ترتيب: baseline أولاً ثم الباقي حسب أعلى PSNR (تنازلياً)
    summ['_o'] = [0 if m == BASELINE_KEY else 1 for m in summ['model']]
    summ = summ.sort_values(['_o', 'best_psnr'], ascending=[True, False]) \
               .drop(columns='_o').reset_index(drop=True)
    return summ


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Pareto front
# ═════════════════════════════════════════════════════════════════════
def pareto_front(p, v):
    """Indices of the Pareto front (highest PSNR for the lowest cost).
    When two trials have the same cost, the one with the higher PSNR is
    visited first, so a dominated trial is never added to the front."""
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    order = np.lexsort((-v, p))          # primary key: p ascending, then v descending
    best, idx = -np.inf, []
    for i in order:
        if v[i] > best:
            idx.append(i); best = v[i]
    return np.array(idx)


def plot_front(ax, x, y, idx, **kw):
    """Draw the front as a STAIRCASE: the best PSNR reachable with a cost
    budget not exceeding x. A straight line would invent values between trials."""
    ax.step(x[idx], y[idx], where='post', **kw)


def pick_best(df, cost_col='n_params'):
    """Index of the highest PSNR (tie: the cheaper trial)."""
    return df.sort_values(['psnr', cost_col], ascending=[False, True]).index[0]


def pick_cheapest(df, cost_col='n_params'):
    """Index of the lowest cost (tie: the higher PSNR)."""
    return df.sort_values([cost_col, 'psnr'], ascending=[True, False]).index[0]


# ═════════════════════════════════════════════════════════════════════
# C1 — رسم مستقل لكل موديل: PSNR vs #Params
#      مع علامتين: Best PSNR  و  Fewest params
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
               label=f"{_label(BASELINE_KEY)} (baseline, n={len(base_df)})", zorder=5)
    plot_front(ax, base_p, base_v, base_pf, color='black',
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
                   label=f"{_label(key)} (n={len(df)})", zorder=3)
        pf = pareto_front(p, v)
        plot_front(ax, p, v, pf, ls='--', color=color, lw=1.2, alpha=0.75, zorder=2)

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
        plot_front(ax, base_p, base_v, base_pf, color='black',
                lw=2.2, alpha=0.85, zorder=4, label='Baseline Pareto front')

        # الموديل الحالي
        ax.scatter(p, v, s=90, marker='o', color='steelblue',
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=_label(key), zorder=4)
        plot_front(ax, p, v, pf, ls='--', color='steelblue',
                lw=1.8, alpha=0.9, zorder=3,
                label=f'{_label(key)} Pareto front')

        # Best PSNR
        ib = int(np.argmax(v))
        ax.scatter(p[ib], v[ib], marker='*', s=420, color='gold',
                   edgecolors='black', linewidths=1.2, zorder=7,
                   label=f'Best {_label(key)} = '
                         f'{v[ib]:.3f} dB ({p[ib]:.2f}M)')

        # Fewest parameters  (جديد)
        il = int(np.lexsort((-v, p))[0])
        if il != ib:
            ax.scatter(p[il], v[il], marker='D', s=190, color='cyan',
                       edgecolors='black', linewidths=1.3, zorder=7,
                       label=f'Fewest {_label(key)} = '
                             f'{p[il]:.2f}M ({v[il]:.3f} dB)')

        # ⬛ Best baseline (مرجعي)
        ib_b = int(np.argmax(base_v))
        ax.scatter(base_p[ib_b], base_v[ib_b], marker='s', s=260,
                   facecolors='none', edgecolors='gold', linewidths=2.4,
                   zorder=6,
                   label=f'Best baseline = '
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
        # موديل بلا عمود FLOPs يُتخطى بدل أن ينهار السكربت
        if 'flops_G' not in df.columns:
            return df.iloc[0:0].copy()
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
               label=f"{_label(BASELINE_KEY)} (baseline, n={len(base_df)})", zorder=5)
    plot_front(ax, base_f, base_v, base_pf, color='black',
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
                   label=f"{_label(key)} (n={len(df)})", zorder=3)
        pf = pareto_front(f, v)
        plot_front(ax, f, v, pf, ls='--', color=color, lw=1.2,
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
        plot_front(ax, base_f, base_v, base_pf, color='black',
                lw=2.2, alpha=0.85, zorder=4, label='Baseline Pareto front')

        ax.scatter(f, v, s=90, marker='o', color='steelblue',
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=_label(key), zorder=4)
        plot_front(ax, f, v, pf, ls='--', color='steelblue',
                lw=1.8, alpha=0.9, zorder=3,
                label=f'{_label(key)} Pareto front')

        # Best PSNR
        ib = int(np.argmax(v))
        ax.scatter(f[ib], v[ib], marker='*', s=420, color='gold',
                   edgecolors='black', linewidths=1.2, zorder=7,
                   label=f'Best {_label(key)} = '
                         f'{v[ib]:.3f} dB ({f[ib]:.2f}G)')

        # Fewest FLOPs  (جديد)
        il = int(np.lexsort((-v, f))[0])
        if il != ib:
            ax.scatter(f[il], v[il], marker='D', s=190, color='cyan',
                       edgecolors='black', linewidths=1.3, zorder=7,
                       label=f'Fewest-FLOPs {_label(key)} = '
                             f'{f[il]:.2f}G ({v[il]:.3f} dB)')

        # ⬛ Best baseline (مرجعي)
        ib_b = int(np.argmax(base_v))
        ax.scatter(base_f[ib_b], base_v[ib_b], marker='s', s=260,
                   facecolors='none', edgecolors='gold', linewidths=2.4,
                   zorder=6,
                   label=f'Best baseline = '
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
# C2 / C2_FLOPs — أعلى PSNR وأقل تكلفة (دالة واحدة لكليهما)
# ═════════════════════════════════════════════════════════════════════
def _best_and_cheapest(per_model, cost_col, scale):
    """جدول: أعلى PSNR وأقل تكلفة لكل موديل (cost_col = 'n_params' أو 'flops_G')."""
    rows = []
    for key, df in per_model.items():
        if cost_col not in df.columns:
            continue
        d = df.dropna(subset=[cost_col])
        if d.empty:
            continue
        hi = d.loc[pick_best(d, cost_col)]
        lo = d.loc[pick_cheapest(d, cost_col)]
        rows.append({
            'model': key, 'label': _label(key), 'n': len(d),
            'best_psnr': float(hi['psnr']), 'cost_hi': float(hi[cost_col]) / scale,
            'trial_hi': int(hi['trial']),
            'min_cost': float(lo[cost_col]) / scale, 'psnr_lo': float(lo['psnr']),
            'trial_lo': int(lo['trial']),
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab['_o'] = [0 if m == BASELINE_KEY else 1 for m in tab['model']]
    return tab.sort_values(['_o', 'best_psnr'], ascending=[True, False]) \
              .drop(columns='_o').reset_index(drop=True)


def _plot_C2_generic(tab, unit, fmt, prefix, title_cost, title_main):
    """
    يكتب:  <prefix>_all_models.*  و  <prefix>_<model>_vs_baseline.*
    - أعلى PSNR: نقاط (lollipop) بمحور مقتطع مُعلَن (الأعمدة من الصفر تُخفي الفروق).
    - أقل تكلفة: أعمدة من الصفر + PSNR عند ذلك الـ trial وفرقه عن أعلى PSNR للموديل.
    """
    if tab.empty or BASELINE_KEY not in tab['model'].values:
        print(f"   ⚠️  {prefix}: no data/baseline — skipped.")
        return
    base = tab[tab['model'] == BASELINE_KEY].iloc[0]

    # ─── (أ) نظرة شاملة ───
    n = len(tab)
    y = np.arange(n)[::-1]
    colors = [COL_BASE if m == BASELINE_KEY else COL_MODEL for m in tab['model']]
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.75 * n + 3.6), sharey=True)

    ax = axes[0]
    lo_axis = tab['best_psnr'].min() - 0.35
    ax.hlines(y, lo_axis, tab['best_psnr'], colors=colors, lw=2.2, alpha=0.8)
    ax.scatter(tab['best_psnr'], y, s=170, c=colors, edgecolors='black', zorder=3)
    ax.axvline(base['best_psnr'], color='gray', ls='--', lw=1)
    for yi, r in zip(y, tab.itertuples()):
        ax.text(r.best_psnr + 0.03, yi, f"{r.best_psnr:.3f} dB · {r.cost_hi:{fmt}} {unit} · n={r.n}",
                va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(tab['label'])
    ax.set_xlim(lo_axis, tab['best_psnr'].max() + 0.95)
    ax.set_xlabel(f'Highest PSNR (dB) — axis starts at {lo_axis:.1f}')
    ax.set_title('Highest PSNR per model  (dashed = baseline)')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    ax.barh(y, tab['min_cost'], color=colors, edgecolor='black', alpha=0.88)
    for yi, r in zip(y, tab.itertuples()):
        d = r.psnr_lo - r.best_psnr
        ax.text(r.min_cost + tab['min_cost'].max() * 0.01, yi,
                f"{r.min_cost:{fmt}} {unit} · {r.psnr_lo:.2f} dB ({d:+.2f} vs own best)",
                va='center', fontsize=9)
    ax.set_xlim(0, tab['min_cost'].max() * 1.75)
    ax.set_xlabel(f'Minimum {title_cost} ({unit})')
    ax.set_title(f'Fewest {title_cost.lower()} per model')
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'{title_main} — all models', fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_all_models")

    # ─── (ب) شكل لكل موديل مقابل baseline ───
    for r in tab.itertuples():
        if r.model == BASELINE_KEY:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        lbls = [f"{_label(BASELINE_KEY)}\n(baseline)", r.label]
        cols = [COL_BASE, COL_MODEL]

        ax = axes[0]
        vals = [base['best_psnr'], r.best_psnr]
        costs = [base['cost_hi'], r.cost_hi]
        lo_a = min(vals) - 0.5
        ax.hlines([0, 1], lo_a, vals, colors=cols, lw=3)
        ax.scatter(vals, [0, 1], s=260, c=cols, edgecolors='black', zorder=3)
        for yi, v, c in zip([0, 1], vals, costs):
            ax.text(v + 0.04, yi, f"{v:.3f} dB\n{c:{fmt}} {unit}", va='center', fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(lbls)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(lo_a, max(vals) + 0.7)
        ax.set_xlabel(f'Highest PSNR (dB) — axis starts at {lo_a:.1f}')
        ax.set_title(f'Highest PSNR — Δ = {r.best_psnr - base["best_psnr"]:+.3f} dB')
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[1]
        cvals = [base['min_cost'], r.min_cost]
        pps = [base['psnr_lo'], r.psnr_lo]
        own = [base['best_psnr'], r.best_psnr]
        ax.bar(range(2), cvals, color=cols, edgecolor='black', alpha=0.88, width=0.55)
        for xi, c, p, o in zip(range(2), cvals, pps, own):
            ax.text(xi, c + max(cvals) * 0.02,
                    f"{c:{fmt}} {unit}\n{p:.2f} dB ({p - o:+.2f} vs own best)",
                    ha='center', fontsize=10)
        ax.set_xticks(range(2))
        ax.set_xticklabels(lbls)
        ax.set_ylim(0, max(cvals) * 1.3)
        ax.set_ylabel(f'Minimum {title_cost} ({unit})')
        dpct = (r.min_cost - base['min_cost']) / base['min_cost'] * 100
        ax.set_title(f'Fewest {title_cost.lower()} — Δ = {dpct:+.1f}%  '
                     f'(PSNR there: {r.psnr_lo - base["psnr_lo"]:+.2f} dB vs baseline)')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'{prefix.split("_")[0]} — {r.label} vs {_label(BASELINE_KEY)}',
                     fontsize=13, y=1.02)
        fig.tight_layout()
        safe = re.sub(r'[^\w\-]+', '_', r.model)
        save_figure(fig, f"{prefix}_{safe}_vs_baseline")


def plot_C2_flops(per_model, summ):
    """C2_FLOPs_*  — أعلى PSNR وأقل FLOPs (يحتاج عمود flops_G)."""
    tab = _best_and_cheapest(per_model, 'flops_G', 1.0)
    if tab.empty:
        print("   ℹ️  No flops data — skip C2_FLOPs.")
        return
    _plot_C2_generic(tab, 'G', '.2f', 'C2_FLOPs', 'FLOPs', 'C2_FLOPs — Highest PSNR & Fewest FLOPs')


def plot_C2(summ, per_model):
    """C2_*  — أعلى PSNR وأقل Parameters."""
    tab = _best_and_cheapest(per_model, 'n_params', 1e6)
    _plot_C2_generic(tab, 'M', '.2f', 'C2', 'Parameters', 'C2 — Highest PSNR & Fewest Parameters')


# ═════════════════════════════════════════════════════════════════════
# C3 — رسم مستقل لكل موديل: ΔPSNR + ΔParams% vs baseline
# ═════════════════════════════════════════════════════════════════════
def _c3_metrics(summ, flops_available):
    """قائمة (اسم العمود، عنوان، وحدة، هل الأقل أفضل) للمقاييس المتاحة."""
    m = [('Δpsnr_vs_base', 'ΔPSNR (dB)', 'dB', False),
         ('Δparams_pct', 'ΔParameters (%)', '%', True)]
    if flops_available:
        m.append(('Δbest_flops_pct', 'ΔFLOPs (%)', '%', True))
    return m


def plot_C3(summ):
    """
    ΔPSNR و ΔParams (و ΔFLOPs) لأعلى-PSNR trial مقابل أعلى-PSNR للـ baseline.
    كل مقياس في لوحة مستقلة بمحوره ووحدته، ولونه حسب إشارته هو (أزرق = أفضل).
    """
    rows = summ[summ['model'] != BASELINE_KEY].copy()
    if rows.empty:
        print("   ℹ️  No non-baseline models — skip C3.")
        return
    metrics = _c3_metrics(summ, rows['Δbest_flops_pct'].notna().any())

    def _color(v, lower_better):
        return COL_GOOD if ((v <= 0) if lower_better else (v >= 0)) else COL_BAD

    # ─── (أ) نظرة شاملة ───
    labels = rows['label'].tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.8 * len(metrics), 0.6 * len(rows) + 3.2),
                             sharey=True)
    for ax, (col, ttl, unit, lower_better) in zip(np.atleast_1d(axes), metrics):
        vals = rows[col].tolist()
        bars = ax.barh(labels, vals, color=[_color(v, lower_better) for v in vals],
                       edgecolor='black', alpha=0.9)
        span = max(abs(v) for v in vals) or 1
        for bar, v in zip(bars, vals):
            ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:+.3f}' if unit == 'dB' else f'{v:+.1f}%',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        ax.axvline(0, color='black', lw=1)
        ax.margins(x=0.22)
        ax.set_xlabel(f'{ttl} vs {_label(BASELINE_KEY)}')
        ax.set_title(ttl + ('  (negative = smaller)' if lower_better else '  (positive = better)'))
        ax.grid(True, alpha=0.3, axis='x')
    fig.suptitle('C3 — Highest-PSNR trial of each model vs the baseline — blue = better, orange = worse',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, "C3_all_models")

    # ─── (ب) شكل لكل موديل ───
    for _, r in rows.iterrows():
        mets = [m for m in metrics if pd.notna(r[m[0]])]
        fig, axes = plt.subplots(1, len(mets), figsize=(4.4 * len(mets), 5.6))
        for ax, (col, ttl, unit, lower_better) in zip(np.atleast_1d(axes), mets):
            v = float(r[col])
            ax.bar([ttl], [v], color=_color(v, lower_better), edgecolor='black', alpha=0.9, width=0.5)
            ax.axhline(0, color='black', lw=1)
            ax.text(0, v + (abs(v) * 0.04 + 1e-9) * (1 if v >= 0 else -1),
                    f'{v:+.3f} dB' if unit == 'dB' else f'{v:+.1f}%',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=13, fontweight='bold')
            ax.margins(y=0.25)
            ax.set_ylabel(unit if unit == 'dB' else 'percent')
            ax.set_title(ttl)
            ax.grid(True, alpha=0.3, axis='y')
        fig.suptitle(f'C3 — {r["label"]} vs {_label(BASELINE_KEY)} (highest-PSNR trials)',
                     fontsize=13, y=1.03)
        fig.tight_layout()
        safe = re.sub(r'[^\w\-]+', '_', r['model'])
        save_figure(fig, f"C3_{safe}_vs_baseline")


# ═════════════════════════════════════════════════════════════════════
# SECTION 10 — E1: Best-so-far convergence per model
# ═════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════
# E1 — Best-so-far convergence — baseline + موديل واحد لكل شكل
# ═════════════════════════════════════════════════════════════════════
def _pick_study(db_path):
    """يحمّل دراسة واحدة من القاعدة: STUDY_NAME_PREFERRED إن وُجدت وإلا أول دراسة."""
    url = f"sqlite:///{db_path}"
    try:
        names = [s.study_name for s in optuna.study.get_all_study_summaries(storage=url)]
        if not names:
            return None
        name = STUDY_NAME_PREFERRED if STUDY_NAME_PREFERRED in names else names[0]
        if len(names) > 1:
            print(f"   ℹ️  {Path(db_path).parent.name}: {len(names)} studies in DB — using '{name}'")
        return optuna.load_study(study_name=name, storage=url)
    except Exception:
        return None


def _load_bsf(db_path):
    """يرجع (x, best_so_far, n_total, n_completed) لموديل واحد، أو None."""
    if db_path is None:
        return None
    study = _pick_study(db_path)
    if study is None:
        return None
    comp = sorted((t.number, float(t.value)) for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None)
    if not comp:
        return None
    nums = np.array([c[0] for c in comp])
    psnr = np.array([c[1] for c in comp])
    x = nums if E1_X_AXIS == "trial_number" else np.arange(1, len(comp) + 1)
    return x, np.maximum.accumulate(psnr), len(study.trials), len(comp)


def plot_E1(models):
    """E1 — best-so-far (درجي: القيمة تتغير فقط عند ظهور trial أفضل)."""
    bsf = {}
    for key, info in models.items():
        r = _load_bsf(info['db_path'])
        if r is not None:
            bsf[key] = r
    if not bsf:
        print("   ℹ️  No DB data — skip E1.")
        return
    xlabel = ('Completed-trial index (pruned / invalid trials excluded)'
              if E1_X_AXIS != "trial_number" else 'Trial number')

    # ─── (أ) نظرة شاملة ───
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap('tab10')
    for i, (key, (x, b, n_all, n_c)) in enumerate(bsf.items()):
        is_base = (key == BASELINE_KEY)
        ax.step(x, b, where='post', lw=2.6 if is_base else 1.5,
                color=COL_BASE if is_base else cmap(i), alpha=0.95 if is_base else 0.8,
                label=f"{_label(key)}{' (baseline)' if is_base else ''} (n={n_c})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Best-so-far PSNR (dB)')
    ax.set_title('E1 — Best-so-far convergence — all models')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    fig.tight_layout()
    save_figure(fig, "E1_all_models")

    # ─── (ب) شكل لكل موديل ───
    if BASELINE_KEY not in bsf:
        print("   ⚠️  No baseline in DB data — skip per-model E1.")
        return
    bx, bb, _, bn = bsf[BASELINE_KEY]
    for key, (x, b, n_all, n_c) in bsf.items():
        if key == BASELINE_KEY:
            continue
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.step(bx, bb, where='post', color=COL_BASE, lw=2.6,
                label=f'{_label(BASELINE_KEY)} (baseline, n={bn})')
        ax.step(x, b, where='post', color=COL_MODEL, lw=2.2, label=f'{_label(key)} (n={n_c})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Best-so-far PSNR (dB)')
        ax.set_title(f'E1 — {_label(key)} vs {_label(BASELINE_KEY)} '
                     f'(final Δ = {b[-1] - bb[-1]:+.3f} dB)')
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
def _plot_colored_by_per_model(trial_df, hp_cols, fname_base):
    """لكل موديل: scatter ملوّن بالـ HP + طبقة baseline رمادية للخلفية.
    hp_cols: اسم عمود أو tuple من الأعمدة المرشّحة؛ يُستخدم لكل موديل أول عمود
    فيه قيم فعلاً (حتى لا يُكتب الملف نفسه مرتين ولا يُرسم شكل فارغ)."""
    if isinstance(hp_cols, str):
        hp_cols = (hp_cols,)
    if not any(c in trial_df.columns for c in hp_cols):
        print(f"   ℹ️  None of {hp_cols} exists — skip {fname_base}.")
        return

    if BASELINE_KEY not in trial_df['model'].unique():
        return
    base = trial_df[trial_df['model'] == BASELINE_KEY]

    for key, sub in trial_df.groupby('model'):
        if key == BASELINE_KEY:
            continue
        hp_col = next((c for c in hp_cols
                       if c in sub.columns and sub[c].notna().any()), None)
        if hp_col is None:
            print(f"   ℹ️  {key}: no data in {hp_cols} — skip {fname_base}.")
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
                lbl = (str(int(u)) if isinstance(u, (float, np.floating))
                       and float(u).is_integer() else str(u))
                ax.scatter(sub.loc[m, 'n_params']/1e6,
                           sub.loc[m, 'psnr'],
                           color=cmap(i), s=100, alpha=0.9,
                           edgecolors='black', linewidths=0.7,
                           label=lbl, zorder=4)
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
    _plot_colored_by_per_model(trial_df, ('hp_n_feats', 'hp_nf'),
                               "E2_params_vs_psnr_by_nfeats")
    _plot_colored_by_per_model(trial_df, 'hp_num_blocks',
                               "E3_params_vs_psnr_by_numblocks")

# ═════════════════════════════════════════════════════════════════════
# E4 — PSNR vs training time — شكل مستقل لكل موديل
# ═════════════════════════════════════════════════════════════════════
def _load_time_and_psnr(db_path):
    if db_path is None:
        return None, None
    study = _pick_study(db_path)
    if study is None:
        return None, None
    df = study.trials_dataframe()
    if 'state' not in df.columns or 'value' not in df.columns or 'duration' not in df.columns:
        return None, None
    df = df[df['state'] == 'COMPLETE'].copy()
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
    ax.set_title('E4 — PSNR vs training time — all models (indicative: GPU load / search space differ)')
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
# SECTION 14 — Excel exports
# ═════════════════════════════════════════════════════════════════════
def _style_sheet(ws, df, number_formats=None):
    """رأس عريض + تجميد الصف الأول + عرض أعمدة + تنسيق أرقام."""
    from openpyxl.styles import Font
    number_formats = number_formats or {}
    for j, col in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=j)
        cell.font = Font(bold=True)
        ws.column_dimensions[cell.column_letter].width = max(12, min(46, len(str(col)) + 4))
        if col in number_formats:
            for r in range(2, len(df) + 2):
                ws.cell(row=r, column=j).number_format = number_formats[col]
    ws.freeze_panes = 'B2'


def export_excel(summ, trial_df, per_model):
    # ═══ Sheet 1: model summary ═══
    s1 = summ[[
        'label', 'model', 'n_trials',
        'best_psnr', 'best_params', 'best_trial', 'best_flops',
        'min_params', 'psnr_at_min_params', 'min_params_trial',
        'trade_psnr', 'trade_params', 'trade_trial',
        'Δpsnr_vs_base', 'Δparams_vs_base', 'Δparams_pct', 'Δmin_par_pct',
        'Δtrade_psnr_vs_base', 'Δtrade_params_pct_vs_base',
        'Δtrade_psnr_vs_own', 'Δtrade_params_pct_vs_own',
        'Δmin_psnr_vs_own', 'Δmin_params_pct_vs_own',
    ]].copy()
    s1.columns = [
        'Model', 'Key', 'Completed trials',
        'Best PSNR (dB)', 'Params @ best', 'Best trial', 'FLOPs @ best (G)',
        'Min params', 'PSNR @ min params (dB)', 'Min-params trial',
        f'Trade-off PSNR (dB) (≤{TOLERANCE_DB:g} dB below best)', 'Trade-off params', 'Trade-off trial',
        'ΔPSNR best vs BASELINE best (dB)', 'ΔParams best vs BASELINE best',
        'ΔParams best vs BASELINE best (%)', 'ΔMinParams vs BASELINE min (%)',
        'ΔPSNR trade-off vs BASELINE best (dB)', 'ΔParams trade-off vs BASELINE best (%)',
        'ΔPSNR trade-off vs OWN best (dB)', 'ΔParams trade-off vs OWN best (%)',
        'ΔPSNR fewest vs OWN best (dB)', 'ΔParams fewest vs OWN best (%)',
    ]

    # ═══ Sheet 2: raw trial-level ═══
    s2 = trial_df.copy()

    # ═══ Sheet 3: long format — كل Δ بمرجع صريح في اسم العمود ═══
    def _flops_at(key, trial):
        d = per_model[key]
        if 'flops_G' not in d.columns:
            return np.nan
        v = d.loc[d['trial'] == trial, 'flops_G']
        return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else np.nan

    base = summ[summ['model'] == BASELINE_KEY]
    base = base.iloc[0] if len(base) else None
    long_rows = []
    for _, r in summ.iterrows():
        for sel, p_psnr, p_par, p_tr in (
            ('Highest PSNR', r['best_psnr'], r['best_params'], r['best_trial']),
            (f'Best trade-off (≤ {TOLERANCE_DB:g} dB)', r['trade_psnr'], r['trade_params'], r['trade_trial']),
            ('Fewest parameters', r['psnr_at_min_params'], r['min_params'], r['min_params_trial']),
        ):
            long_rows.append({
                'Model': r['label'], 'Selection': sel, 'Trial': int(p_tr),
                'PSNR (dB)': p_psnr, 'Params (M)': p_par / 1e6,
                'FLOPs (G)': _flops_at(r['model'], p_tr),
                'ΔPSNR vs OWN highest (dB)': p_psnr - r['best_psnr'],
                'ΔParams vs OWN highest (%)': (p_par / r['best_params'] - 1) * 100,
                'ΔPSNR vs BASELINE highest (dB)': (p_psnr - base['best_psnr']) if base is not None else np.nan,
                'ΔParams vs BASELINE highest (%)': ((p_par / base['best_params'] - 1) * 100) if base is not None else np.nan,
                'Completed trials in study': r['n_trials'],
            })
    s3 = pd.DataFrame(long_rows)

    # ═══ Sheet 4: FLOPs summary (اختياري) ═══
    flops_tab = _best_and_cheapest(per_model, 'flops_G', 1.0)
    s4 = None
    if not flops_tab.empty:
        s4 = flops_tab.rename(columns={
            'model': 'Model', 'label': 'Label', 'n': 'Completed trials',
            'best_psnr': 'Best PSNR (dB)', 'cost_hi': 'FLOPs @ best (G)', 'trial_hi': 'Best trial',
            'min_cost': 'Min FLOPs (G)', 'psnr_lo': 'PSNR @ min FLOPs (dB)', 'trial_lo': 'Min-FLOPs trial'})

    # ═══ Sheet 5: definitions ═══
    s5 = pd.DataFrame({'Item': [
        'Highest PSNR', 'Fewest parameters / FLOPs', 'Best trade-off', 'Pareto front',
        'Tie rule', 'ΔPSNR / ΔParams columns', 'FLOPs', 'Number of trials'],
        'Definition': [
        'Trial with the highest PSNR of the model.',
        'Trial with the lowest cost; if several trials share the same cost, the one with the higher PSNR.',
        f'Smallest model whose PSNR is within {TOLERANCE_DB:g} dB of the highest PSNR of the same model.',
        'Trials for which no other trial has higher PSNR with a lower or equal cost (drawn as a staircase).',
        'Ties are resolved identically everywhere (see pick_best / pick_cheapest).',
        'The reference is written in every column name: OWN = the same model\'s highest PSNR; BASELINE = the baseline\'s highest PSNR.',
        'As stored in the study files (fvcore: 1 multiply-add = 1 FLOP, unsupported ops ignored), input size per file header.',
        'Models have different numbers of completed trials; a larger search tends to find a higher maximum.']})

    out = OUTPUT_DIR / "models_comparison_summary.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        s1.to_excel(w, sheet_name='model_summary', index=False)
        s3.to_excel(w, sheet_name='slides_10_11', index=False)
        s2.to_excel(w, sheet_name='trial_level', index=False)
        n_sheets = 4
        if s4 is not None:
            s4.to_excel(w, sheet_name='flops_summary', index=False)
            n_sheets += 1
        s5.to_excel(w, sheet_name='definitions', index=False)
        for name, frame in (('model_summary', s1), ('slides_10_11', s3), ('trial_level', s2)):
            _style_sheet(w.sheets[name], frame)
        w.sheets['definitions'].column_dimensions['A'].width = 30
        w.sheets['definitions'].column_dimensions['B'].width = 120
    print(f"   ✅ models_comparison_summary.xlsx ({n_sheets + 1} sheets)")

    trial_df.to_excel(OUTPUT_DIR / "trial_level_all_models.xlsx", index=False)
    print(f"   ✅ trial_level_all_models.xlsx")


# ═════════════════════════════════════════════════════════════════════
# SECTION 15 — README
# ═════════════════════════════════════════════════════════════════════
def write_readme(summ):
    files = sorted({p.stem for p in OUTPUT_DIR.glob('*.png')})
    lines = [
        "Model Comparison — Auto-Generated (v2)",
        "=" * 70, "",
        f"Baseline: {_label(BASELINE_KEY)} ({BASELINE_KEY})",
        f"Models  : {len(summ)}",
        f"Best trade-off tolerance: {TOLERANCE_DB:g} dB",
        "",
        "Selection rules: Highest PSNR (tie: cheaper) | Fewest cost (tie: higher PSNR) |",
        "Best trade-off = smallest model within the tolerance of the highest PSNR |",
        "Pareto front drawn as a staircase (best PSNR reachable within a budget).",
        "Caveats: models have different numbers of completed trials (see 'Trials');",
        "FLOPs come from the study files (fvcore: 1 multiply-add = 1 FLOP).",
        "",
        "Figures in this folder (each as .pdf/.png/.jpeg):",
    ]
    lines += [f"  {f}" for f in files]
    lines += [
        "",
        "Excel: models_comparison_summary.xlsx (sheets: model_summary, slides_10_11, trial_level,",
        "       flops_summary [if FLOPs], definitions) and trial_level_all_models.xlsx",
        "",
        "Summary table:",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"  {r['label']:<22} n={int(r['n_trials']):>2}  "
            f"best={r['best_psnr']:.3f} dB @ {r['best_params']/1e6:.2f}M  "
            f"fewest={r['min_params']/1e6:.2f}M @ {r['psnr_at_min_params']:.3f} dB  "
            f"ΔPSNR={r['Δpsnr_vs_base']:+.3f} dB  ΔPar={r['Δparams_pct']:+.1f}%"
        )
    (OUTPUT_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"   ✅ README.txt")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
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
        print(f"   Update BASELINE_RUNS in SECTION 1 to one of the above.")
        print(f"   → this run is skipped (no silent fallback to another model).\n")
        return False

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
    plot_C2(summ, per_model)
    plot_C3(summ)
    plot_E1(models)
    plot_E2_E3(trial_df)
    plot_E4(models)

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
    return True


if __name__ == "__main__":
    for _key, _sub in BASELINE_RUNS:
        BASELINE_KEY = _key
        OUTPUT_SUBDIR = _sub
        OUTPUT_DIR = HERE / OUTPUT_SUBDIR
        OUTPUT_DIR.mkdir(exist_ok=True)
        print("\n" + "#" * 78 + f"\n# RUN: baseline = {_key!r}  →  {OUTPUT_SUBDIR}\n" + "#" * 78)
        main()