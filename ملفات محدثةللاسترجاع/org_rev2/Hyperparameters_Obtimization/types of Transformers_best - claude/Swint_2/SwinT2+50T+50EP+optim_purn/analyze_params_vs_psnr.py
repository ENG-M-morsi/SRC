# =====================================================================
# analyze_params_vs_psnr.py  —  v2 (corrected, model-agnostic)
# =====================================================================
# يحسب لكل trial مكتمل:  n_params (من النموذج المبني فعلاً)، FLOPs (fvcore)،
# وعدد البارامترات الفعّالة (التي تصلها gradients) ويُنتج Excel + الأشكال 20/21/22/23.
#
# ما تغيّر عن النسخة السابقة (الأخطاء المصححة):
#   ✅ n_params يُحسب من الموديل المبني بهايبربارامترات الـ trial نفسها (لا صيغة يدوية)؛
#      والصيغة التحليلية صارت اختيارية للمقارنة فقط (ANALYTIC_NPARAMS_FN).
#   ✅ لا قيم افتراضية صامتة: أي هايبربارامتر لا يقبله HUTCN يُعلَن "IGNORED" بوضوح،
#      وأي هايبربارامتر غير مُصنَّف يُعلَن، وأي مفتاح ناقص يُعلَن.
#   ✅ مفتاح الـ cache يتضمن hash لكود الموديل → تعديل الموديل يُبطل القيم القديمة.
#   ✅ البارامترات الفعّالة (n_params_effective) تكشف الطبقات/الـ blocks غير المستخدمة في forward.
#   ✅ جبهة Pareto درجية + معالجة التعادل؛ شكل 22 مرتّب رقمياً ولا يبدأ من الصفر.
#   ✅ شكل 21: مقياس لوغاريتمي للألوان عند البارامترات log-sampled (lr...).
#   ✅ تسمية FLOPs: قيمة fvcore (1 multiply-add = 1 FLOP، أي MACs).
# =====================================================================

import os, sys, re, json, hashlib, time, shutil, warnings, importlib, importlib.util, inspect
from pathlib import Path
warnings.filterwarnings('ignore')

import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ═════════════════════════════════════════════════════════════════════
HERE         = Path(__file__).resolve().parent
# 🟠 اسم الموديل في الأشكال والملفات. الافتراضي: أول جزء من اسم المجلد قبل أول "+" أو مسافة
#    (مثال: "DAT+50T+50EP+optim_purn" ← "DAT"). اكتبه يدوياً إن أردت اسماً آخر.
MODEL_DISPLAY_NAME = re.split(r"[+\s]", HERE.name)[0] or HERE.name
STORAGE_PATH = HERE / "optuna_study.db"
OUTPUT_DIR   = HERE / "params_vs_psnr_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE   = OUTPUT_DIR / "n_params_cache.json"       # يحوي n_params + effective + FLOPs

STUDY_NAME = "hutcn_hyperopt_study"       # يُفضَّل هذا الاسم إن وُجد، وإلا أول دراسة

# ═══ Model source ═══
SNAPSHOT_DIR      = HERE / "_snapshot_model"
SHARED_MAX_LEVELS = 12

# ═══ 🟠 ربط هايبربارامترات Optuna بوسائط HUTCN (يعمل لأي موديل بنفس الأسلوب) ═══
FIXED_MODEL_KWARGS = {'in_nc': 3, 'num_modules': 1, 'out_nc': 3, 'upscale': 4}
#   (تُمرَّر فقط الوسائط التي يقبلها HUTCN فعلاً)
DEFAULT_PARAM_TO_KWARG = {                 # اسم الهايبربارامتر ← وسائط مرشّحة (يُستخدم أول ما يقبله HUTCN)
    'n_feats': ('nf',), 'nf': ('nf',),
    'num_heads': ('num_heads',),
    'window_size': ('window_size', 'ws'),
    'num_blocks': ('num_blocks',),
    'depth': ('depth',), 'mlp_ratio': ('mlp_ratio',), 'ffn_ratio': ('ffn_ratio',),
}
EXTRA_PARAM_TO_KWARG = {}                  # 🟠 تخصيص لكل موديل، مثال لـ SwinT:  {'num_blocks': ('depth',)}
TRAINING_ONLY_PARAMS = {                   # لا تغيّر بنية النموذج ← لا تُمرَّر
    'lr', 'weight_decay', 'optimizer', 'scheduler', 'loss', 'patch_size', 'batch_size',
    'momentum', 'beta1', 'beta2', 'warmup_epochs', 'epochs', 'grad_clip',
}

# ═══ الأعداد ═══
PARAMS_FOR_PLOTS = "declared"      # "declared" (كل بارامترات النموذج) | "effective" (التي تصلها gradients)
COMPUTE_EFFECTIVE_PARAMS = True    # forward+backward تجريبي لكشف البارامترات غير المستخدمة
ANALYTIC_NPARAMS_FN = None         # 🟠 اختياري: دالة (params_dict) → int للمقارنة مع العدد المبني

# ═══ FLOPs (fvcore: 1 multiply-add = 1 FLOP) ═══
COMPUTE_FLOPS    = True
FLOPS_IMG_H      = 48              # 🟠 ارتفاع صورة الإدخال (LR)
FLOPS_IMG_W      = 48              # 🟠 عرض  صورة الإدخال (LR)
FLOPS_IN_CH      = 3
FLOPS_LABEL      = "FLOPs (G, fvcore: 1 MAC = 1 FLOP)"

SCRIPT_VERSION = "v2.1"

# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — Plot style
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


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — Model loading (snapshot-first) + architecture mapping
# ═════════════════════════════════════════════════════════════════════
_MODEL_MODULE = None
_MODEL_DIR = None
_HUTCN_CLASS = None
_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg)


def _resolve_model_module():
    """يرجع اسم الوحدة التي تحوي HUTCN ويحدد مجلد الموديل."""
    global _MODEL_MODULE, _MODEL_DIR
    if _MODEL_MODULE is not None:
        return _MODEL_MODULE
    if SNAPSHOT_DIR.is_dir() and (SNAPSHOT_DIR / "dhtcun.py").exists():
        init = SNAPSHOT_DIR / "__init__.py"
        if not init.exists():
            init.write_text("# auto-generated by analyze_params_vs_psnr.py\n")
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        _MODEL_MODULE, _MODEL_DIR = "_snapshot_model.dhtcun", SNAPSHOT_DIR
        print(f"   ℹ️  using SNAPSHOT: {SNAPSHOT_DIR.name}/")
        return _MODEL_MODULE
    cur = HERE
    for _ in range(SHARED_MAX_LEVELS):
        if (cur / "model").is_dir():
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            _MODEL_MODULE, _MODEL_DIR = "model.dhtcun", cur / "model"
            print(f"   ℹ️  using SHARED model/ at: {cur}")
            return _MODEL_MODULE
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(
        "model code not found.\n"
        f"   → create the folder:  {SNAPSHOT_DIR}\n"
        "   → copy into it the model files used by THIS study: dhtcun.py, dhtcu_block.py, "
        "custom_attention_blocks.py (and any other file they import, e.g. SwinT.py)\n"
        "   → then run the script again.")


def _get_HUTCN():
    global _HUTCN_CLASS
    if _HUTCN_CLASS is None:
        mod = importlib.import_module(_resolve_model_module())
        _HUTCN_CLASS = mod.HUTCN
        sig = inspect.signature(_HUTCN_CLASS.__init__)
        print(f"   ℹ️  HUTCN signature: {sorted(k for k in sig.parameters if k != 'self')}")
    return _HUTCN_CLASS


def _model_source_hash():
    """hash لكل ملفات .py في مجلد الموديل — يُبطل الـ cache عند أي تعديل في الموديل."""
    _resolve_model_module()
    h = hashlib.md5()
    for p in sorted(Path(_MODEL_DIR).glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _as_py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def build_arch_kwargs(params):
    """
    يحوّل هايبربارامترات trial إلى وسائط HUTCN بلا أي قيمة افتراضية صامتة.
    يرجع (kwargs, notes) حيث notes قائمة (نوع, اسم) للتقرير.
    """
    HUTCN = _get_HUTCN()
    valid = {k for k in inspect.signature(HUTCN.__init__).parameters if k != 'self'}
    kwargs = {k: v for k, v in FIXED_MODEL_KWARGS.items() if k in valid}
    notes = []
    mapping = dict(DEFAULT_PARAM_TO_KWARG)
    mapping.update(EXTRA_PARAM_TO_KWARG)

    for name, value in params.items():
        if name in TRAINING_ONLY_PARAMS:
            continue
        cands = mapping.get(name)
        if cands is None:
            notes.append(('UNMAPPED', name))          # ليس تدريبياً معروفاً ولا مربوطاً بالنموذج
            continue
        hit = next((c for c in cands if c in valid), None)
        if hit is None:
            notes.append(('IGNORED', name))           # HUTCN لا يقبله ← لا أثر له على النموذج
            continue
        kwargs[hit] = _as_py(value)
        notes.append(('USED', f"{name} → {hit}"))
    for name in valid - set(kwargs) - {'in_nc', 'out_nc', 'upscale', 'num_modules'}:
        pass                                           # وسائط HUTCN غير مُمرَّرة تأخذ قيمة الموديل الافتراضية
    return kwargs, notes


def report_mapping(sample_params):
    """يطبع مرة واحدة كيف رُبطت هايبربارامترات الدراسة بوسائط الموديل."""
    kwargs, notes = build_arch_kwargs(sample_params)
    HUTCN = _get_HUTCN()
    sig = inspect.signature(HUTCN.__init__)
    print("\n🔗 Hyperparameter → model mapping:")
    for kind, what in notes:
        icon = {'USED': '✅', 'IGNORED': '⛔', 'UNMAPPED': '❓'}[kind]
        extra = {'IGNORED': '   ← HUTCN does NOT accept it: it has NO effect on the built model',
                 'UNMAPPED': '   ← not mapped: add it to EXTRA_PARAM_TO_KWARG or TRAINING_ONLY_PARAMS'}.get(kind, '')
        print(f"   {icon} {kind:<8} {what}{extra}")
    not_set = [k for k in sig.parameters if k != 'self' and k not in kwargs]
    if not_set:
        print(f"   ℹ️  model arguments left at the model's own defaults: "
              f"{ {k: sig.parameters[k].default for k in not_set} }")
    print(f"   → constructor kwargs: {kwargs}\n")
    return notes


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — Counting parameters / effective parameters / FLOPs (per architecture)
# ═════════════════════════════════════════════════════════════════════
def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=1))


def _arch_key(kwargs, src_hash):
    data = dict(kwargs)
    data.update(_src=src_hash, _v=SCRIPT_VERSION, _H=FLOPS_IMG_H, _W=FLOPS_IMG_W, _C=FLOPS_IN_CH,
                _flops=COMPUTE_FLOPS, _eff=COMPUTE_EFFECTIVE_PARAMS)
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class _SkipFlops(Exception):
    pass


def summarize_unused(names, max_items=12):
    """أسماء الوحدات (وليس كل بارامتر) التي لم تصلها gradients، مثال: ['LR_conv1', 'ps_conv2']."""
    mods = sorted({n.rsplit('.', 1)[0] if '.' in n else n for n in names})
    return mods[:max_items] + (['...'] if len(mods) > max_items else [])


def measure_architecture(kwargs):
    """يبني النموذج ويرجع dict: n_params, n_effective, flops_G (قيم None إن تعذّر القياس)."""
    HUTCN = _get_HUTCN()
    model = HUTCN(**kwargs)
    out = {'n_params': int(sum(p.numel() for p in model.parameters())),
           'n_effective': None, 'flops_G': None}
    try:
        import torch
    except ImportError:
        _warn_once('notorch', "   ⚠️  torch not importable — effective params and FLOPs skipped.")
        return out

    if COMPUTE_EFFECTIVE_PARAMS:
        try:
            model.train()
            model.zero_grad(set_to_none=True)
            x = torch.randn(1, FLOPS_IN_CH, FLOPS_IMG_H, FLOPS_IMG_W)
            y = model(x)
            y.float().mean().backward()
            out['n_effective'] = int(sum(p.numel() for p in model.parameters() if p.grad is not None))
            out['unused_modules'] = summarize_unused(
                [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None])
        except Exception as e:
            _warn_once('eff', f"   ⚠️  effective-params probe failed ({type(e).__name__}: {e}) — skipped.")
    if COMPUTE_FLOPS:
        try:
            try:
                from fvcore.nn import FlopCountAnalysis
            except ImportError:
                _warn_once('nofv', "   ⚠️  fvcore is not installed → FLOPs skipped.  Install it in this environment:  pip install fvcore")
                raise _SkipFlops()
            model.eval()
            x = torch.randn(1, FLOPS_IN_CH, FLOPS_IMG_H, FLOPS_IMG_W)
            with torch.no_grad():
                fca = FlopCountAnalysis(model, x)
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
                out['flops_G'] = float(fca.total() / 1e9)
        except _SkipFlops:
            pass
        except Exception as e:
            if 'flops' not in _WARNED:
                import traceback
                print("\n" + "=" * 78 + "\n   ❌ FLOPs ERROR (first failure — full detail)\n" + "=" * 78)
                traceback.print_exc()
                print(f"   kwargs at failure: {kwargs}\n" + "=" * 78 + "\n")
                _WARNED.add('flops')
    del model
    return out


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Study loading + table
# ═════════════════════════════════════════════════════════════════════
def load_study(db_path):
    url = f"sqlite:///{db_path}"
    try:
        names = [s.study_name for s in optuna.study.get_all_study_summaries(storage=url)]
    except Exception as e:
        print(f"   ⚠️  cannot read {db_path}: {e}")
        return None, None
    if not names:
        return None, None
    name = STUDY_NAME if STUDY_NAME in names else names[0]
    if len(names) > 1:
        print(f"   ℹ️  {len(names)} studies in the DB {names} — using '{name}'")
    return optuna.load_study(study_name=name, storage=url), name


def build_table(study):
    cache = load_cache()
    src_hash = _model_source_hash()
    rows, per_arch = [], {}
    comp = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    for t in comp:
        kwargs, _ = build_arch_kwargs(t.params)
        key = _arch_key(kwargs, src_hash)
        if key not in cache:
            try:
                cache[key] = measure_architecture(kwargs)
            except Exception as e:
                print(f"   ⚠️  trial #{t.number}: cannot build model ({type(e).__name__}: {e}) — skipped")
                continue
        m = cache[key]
        n_built = m['n_params']
        n_attr = t.user_attrs.get('n_params')
        if n_attr is not None and int(n_attr) != n_built:
            _warn_once(f'attr{t.number}', f"   ⚠️  trial #{t.number}: user_attr n_params={int(n_attr):,} ≠ built model {n_built:,}")
        if ANALYTIC_NPARAMS_FN is not None:
            try:
                a = int(ANALYTIC_NPARAMS_FN(t.params))
                if a != n_built:
                    _warn_once(f'an{t.number}', f"   ⚠️  trial #{t.number}: analytic formula {a:,} ≠ built model {n_built:,}")
            except Exception as e:
                _warn_once('anfail', f"   ⚠️  analytic formula failed: {e}")
        eff = m.get('n_effective')
        row = {'trial': t.number, 'n_params': n_built, 'n_params_effective': eff,
               'n_unused': (n_built - eff) if eff is not None else None,
               'unused_modules': ', '.join(m.get('unused_modules') or []),
               'psnr': float(t.value)}
        if COMPUTE_FLOPS:
            row['flops_G'] = m.get('flops_G')
        for k, v in t.params.items():
            row[f"hp_{k}"] = v
        rows.append(row)
        per_arch[key] = True
    save_cache(cache)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # العمود المستخدم في الأشكال
    use_eff = (PARAMS_FOR_PLOTS == "effective" and df['n_params_effective'].notna().all())
    if PARAMS_FOR_PLOTS == "effective" and not use_eff:
        print("   ⚠️  PARAMS_FOR_PLOTS='effective' but effective counts are missing — using declared.")
    df['n_plot'] = df['n_params_effective'] if use_eff else df['n_params']
    return df


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Pareto + plots
# ═════════════════════════════════════════════════════════════════════
def pareto_front(p, v):
    """Pareto: أعلى PSNR لأقل تكلفة. عند تعادل التكلفة يُزار الأعلى PSNR أولاً."""
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    order = np.lexsort((-v, p))
    best, idx = -np.inf, []
    for i in order:
        if v[i] > best:
            idx.append(i); best = v[i]
    return np.array(idx)


def _is_log_param(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return len(s) > 2 and (s > 0).all() and s.max() / s.min() > 20


def plot_single_study(df, study_name):
    label = MODEL_DISPLAY_NAME
    flabel = re.sub(r'[^\w\-]+', '_', label)       # safe text for FILE names
    unit_label = 'effective parameters' if PARAMS_FOR_PLOTS == 'effective' and (df['n_plot'] != df['n_params']).any() \
        else 'parameters'
    p = df['n_plot'].values.astype(float) / 1e6
    v = df['psnr'].values.astype(float)
    pf = pareto_front(p, v)
    tr = df['trial'].values
    ib = int(np.argmax(v))
    il = int(np.lexsort((-v, p))[0])

    # ── 20 ──
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(p, v, c=tr, cmap='viridis', s=90, alpha=0.85, edgecolors='black', linewidths=0.6)
    ax.step(p[pf], v[pf], where='post', color='red', lw=2, label='Pareto front (staircase)', zorder=5)
    ax.scatter(p[pf], v[pf], facecolors='none', edgecolors='red', s=200, linewidths=1.6, zorder=5)
    ax.scatter(p[ib], v[ib], marker='*', s=420, color='gold', edgecolors='black', zorder=7,
               label=f'Highest PSNR #{int(tr[ib])}: {v[ib]:.3f} dB ({p[ib]:.2f} M)')
    if il != ib:
        ax.scatter(p[il], v[il], marker='D', s=190, color='cyan', edgecolors='black', zorder=7,
                   label=f'Fewest {unit_label} #{int(tr[il])}: {p[il]:.3f} M ({v[il]:.3f} dB)')
    ax.set_xlabel(f'Number of {unit_label} (millions)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'PSNR vs #Parameters — {label}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.colorbar(sc, ax=ax, label='Trial number')
    fig.tight_layout()
    save_figure(fig, f"20_{flabel}_psnr_vs_params")

    hp_cols = [c for c in df.columns if c.startswith('hp_')]

    # ── 21 ──
    for hp in hp_cols:
        clean = hp.replace('hp_', '')
        s = df[hp]
        is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 6
        fig, ax = plt.subplots(figsize=(11, 7))
        if is_num:
            norm = mcolors.LogNorm(s.min(), s.max()) if _is_log_param(s) else None
            sc = ax.scatter(p, v, c=s.astype(float), cmap='plasma', norm=norm, s=90, alpha=0.85,
                            edgecolors='black', linewidths=0.6)
            plt.colorbar(sc, ax=ax, label=clean + (' (log scale)' if norm else ''))
        else:
            uniq = sorted(s.dropna().unique().tolist(), key=lambda x: (str(type(x)), x))
            cmap = plt.get_cmap('tab10', max(len(uniq), 2))
            for i, u in enumerate(uniq):
                m = (s == u)
                lbl = str(int(u)) if isinstance(u, (float, np.floating)) and float(u).is_integer() else str(u)
                ax.scatter(p[m], v[m], color=cmap(i), s=90, alpha=0.85,
                           edgecolors='black', linewidths=0.6, label=lbl)
            ax.legend(title=clean, loc='lower right', fontsize=8)
        ax.step(p[pf], v[pf], where='post', color='k', ls='--', lw=1.5, alpha=0.7)
        ax.set_xlabel(f'Number of {unit_label} (millions)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'PSNR vs #Params — {label} — by {clean}  (dashed = Pareto staircase)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, f"21_{flabel}_by_{clean}")

    # ── 22 ── (مرتّب رقمياً، نقاط ± std بلا أعمدة من الصفر، bins لوغاريتمية عند الحاجة)
    for hp in hp_cols:
        clean = hp.replace('hp_', '')
        s = df[hp]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            try:
                edges = (np.logspace(np.log10(s.min()), np.log10(s.max()), 9) if _is_log_param(s)
                         else np.linspace(s.min(), s.max(), 9))
                groups = pd.cut(s, bins=edges, include_lowest=True)
            except Exception:
                continue
            order_key = lambda g: g.left
            lab = lambda g: f"[{g.left:.3g}, {g.right:.3g}]"
        else:
            groups = s
            order_key = lambda g: g
            lab = lambda g: (str(int(g)) if isinstance(g, (float, np.floating)) and float(g).is_integer() else str(g))
        gb = df.groupby(groups, observed=True)['psnr'].agg(['mean', 'std', 'count'])
        if len(gb) < 2:
            continue
        gb = gb.sort_index(key=lambda idx: [order_key(g) for g in idx])
        fig, ax = plt.subplots(figsize=(max(8, len(gb) * 0.9), 6))
        x = np.arange(len(gb))
        ax.errorbar(x, gb['mean'], yerr=gb['std'].fillna(0), fmt='o', color='steelblue',
                    ecolor='black', capsize=4, ms=9, mec='black')
        for xi, (m, sd, n) in zip(x, gb[['mean', 'std', 'count']].values):
            ax.text(xi, m + (sd if np.isfinite(sd) else 0) + 0.05, f'n={int(n)}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([lab(g) for g in gb.index], rotation=40, ha='right', fontsize=9)
        ax.set_xlabel(clean)
        ax.set_ylabel('Mean PSNR ± std (dB)  — axis not forced to zero')
        ax.set_title(f'Mean PSNR per value of {clean} — {label}  (std not shown when n=1)')
        ax.margins(y=0.15)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        save_figure(fig, f"22_{flabel}_mean_psnr_by_{clean}")

    # ── 23 — FLOPs vs PSNR ──
    if 'flops_G' in df.columns and df['flops_G'].notna().any():
        valid = df.dropna(subset=['flops_G'])
        f = valid['flops_G'].values.astype(float)
        vf = valid['psnr'].values.astype(float)
        trf = valid['trial'].values
        pf_f = pareto_front(f, vf)
        ibf = int(np.argmax(vf))
        ilf = int(np.lexsort((-vf, f))[0])
        fig, ax = plt.subplots(figsize=(11, 7))
        sc = ax.scatter(f, vf, c=trf, cmap='viridis', s=90, alpha=0.85, edgecolors='black', linewidths=0.6)
        ax.step(f[pf_f], vf[pf_f], where='post', color='red', lw=2, label='Pareto front (staircase)', zorder=5)
        ax.scatter(f[pf_f], vf[pf_f], facecolors='none', edgecolors='red', s=200, linewidths=1.6, zorder=5)
        ax.scatter(f[ibf], vf[ibf], marker='*', s=420, color='gold', edgecolors='black', zorder=7,
                   label=f'Highest PSNR #{int(trf[ibf])}: {vf[ibf]:.3f} dB ({f[ibf]:.2f} G)')
        if ilf != ibf:
            ax.scatter(f[ilf], vf[ilf], marker='D', s=190, color='cyan', edgecolors='black', zorder=7,
                       label=f'Fewest FLOPs #{int(trf[ilf])}: {f[ilf]:.2f} G ({vf[ilf]:.3f} dB)')
        ax.set_xlabel(f'{FLOPS_LABEL} — input {FLOPS_IMG_H}×{FLOPS_IMG_W}')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'PSNR vs FLOPs — {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        plt.colorbar(sc, ax=ax, label='Trial number')
        fig.tight_layout()
        save_figure(fig, f"23_{flabel}_psnr_vs_flops")
    else:
        print("   ℹ️  no FLOPs — figure 23 skipped.")


# ═════════════════════════════════════════════════════════════════════
# SECTION 7 — Excel
# ═════════════════════════════════════════════════════════════════════
def _write_excel(frame, path, formats):
    from openpyxl.styles import Font
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        frame.to_excel(w, sheet_name='data', index=False)
        ws = w.sheets['data']
        for j, col in enumerate(frame.columns, start=1):
            c = ws.cell(row=1, column=j)
            c.font = Font(bold=True)
            ws.column_dimensions[c.column_letter].width = max(12, min(40, len(str(col)) + 4))
            if col in formats:
                for r in range(2, len(frame) + 2):
                    ws.cell(row=r, column=j).number_format = formats[col]
        ws.freeze_panes = 'A2'



# ═════════════════════════════════════════════════════════════════════
# SECTION 7b — Check against the PREVIOUS Excel files + safe backup
# ═════════════════════════════════════════════════════════════════════
def check_against_previous(df, target_names):
    """
    يقارن n_params المبني الآن بأي ملف *_full.xlsx سابق في مجلد المخرجات (من النسخة القديمة)،
    ويحفظ نسخة احتياطية من الملفات التي ستُستبدل. مفيد للتأكد أن الأعداد القديمة صحيحة.
    """
    prevs = [q for q in sorted(OUTPUT_DIR.glob("*_full.xlsx"), key=lambda q: q.stat().st_mtime, reverse=True)]
    if prevs:
        q = prevs[0]
        try:
            old = pd.read_excel(q)
            if {'trial', 'n_params'}.issubset(old.columns):
                m = df[['trial', 'n_params']].merge(old[['trial', 'n_params']], on='trial',
                                                    suffixes=('_new', '_previous'))
                bad = m[m['n_params_new'] != m['n_params_previous']].copy()
                print(f"\n🔎 Check against previous file {q.name}: {len(m)} common trials, "
                      f"{len(bad)} differ.")
                if len(bad):
                    bad['diff'] = bad['n_params_new'] - bad['n_params_previous']
                    bad['diff_%'] = bad['diff'] / bad['n_params_previous'] * 100
                    bad.to_csv(OUTPUT_DIR / "n_params_check_vs_previous.csv", index=False)
                    w = bad.loc[bad['diff_%'].abs().idxmax()]
                    print(f"   ⚠️  largest difference: trial #{int(w['trial'])}: built {int(w['n_params_new']):,} "
                          f"vs previous {int(w['n_params_previous']):,} ({w['diff_%']:+.2f}%).")
                    print("   → the previous numbers came from a formula or another model version; "
                          "the built-model numbers are the ones that match THIS model code.")
                    print("   → details: n_params_check_vs_previous.csv")
        except Exception as e:
            print(f"   ⚠️  could not compare with {q.name}: {e}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name in target_names:
        f = OUTPUT_DIR / name
        if f.exists():
            shutil.copy2(f, OUTPUT_DIR / f"{f.stem}_PREVIOUS_{stamp}{f.suffix}")
            print(f"   💾 previous {name} kept as {f.stem}_PREVIOUS_{stamp}{f.suffix}")

# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print(f"📂 Folder: {HERE}\n🗄️  DB    : {STORAGE_PATH}\n🧠 Model : {MODEL_DISPLAY_NAME}   ({SCRIPT_VERSION})")
    print("=" * 78)
    if not STORAGE_PATH.exists():
        print(f"❌ لم يُعثر على {STORAGE_PATH}")
        sys.exit(1)
    os.chdir(HERE)

    if COMPUTE_FLOPS and importlib.util.find_spec("fvcore") is None:
        print("⚠️  fvcore is not installed: FLOPs will be skipped.  Install it:  pip install fvcore\n")

    study, study_name = load_study(STORAGE_PATH)
    if study is None:
        print("❌ لا توجد دراسة قابلة للقراءة.")
        sys.exit(1)
    comp = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    print(f"\n✅ Study: '{study_name}' — {len(study.trials)} trials, {len(comp)} completed")
    if not comp:
        print("❌ لا توجد trials مكتملة.")
        sys.exit(1)

    # تقرير الربط + اختبار بناء مبكر على أول trial مكتمل (وليس trials[0] الذي قد يكون مقطوعاً)
    try:
        report_mapping(comp[0].params)
        kw, _ = build_arch_kwargs(comp[0].params)
        _get_HUTCN()(**kw)
        print("   ✅ Model builds OK")
    except Exception as e:
        print(f"   ❌ فشل بناء النموذج: {type(e).__name__}: {e}")
        print("   → راجع _snapshot_model/ أو model/ وربط EXTRA_PARAM_TO_KWARG قبل المتابعة")
        sys.exit(1)

    print("⏳ Measuring architectures (parameters / effective parameters / FLOPs)...")
    t0 = time.time()
    df = build_table(study)
    print(f"✅ Done in {time.time() - t0:.1f}s | {len(df)} usable trials")
    if df.empty:
        print("❌ لا بيانات صالحة.")
        sys.exit(1)

    if COMPUTE_FLOPS and 'flops_G' in df.columns:
        print(f"📊 FLOPs: {int(df['flops_G'].notna().sum())}/{len(df)} rows filled")
    if df['n_params_effective'].notna().any():
        un = df['n_unused'].fillna(0)
        if (un > 0).any():
            top = df.loc[un.idxmax()]
            print(f"\n⚠️  UNUSED PARAMETERS detected (they get no gradient in forward): "
                  f"{int((un > 0).sum())}/{len(df)} trials; largest: trial #{int(top['trial'])} has "
                  f"{int(top['n_unused']):,} of {int(top['n_params']):,} ({top['n_unused'] / top['n_params'] * 100:.1f}%).")
            names = df.loc[un.idxmax(), 'unused_modules']
            if names:
                print(f"   → unused layers/blocks (example from trial #{int(top['trial'])}): {names}")
            print("   → declared n_params overstates the model that is actually used. "
                  "Set PARAMS_FOR_PLOTS = 'effective' to plot the effective count, and review the model code.\n")

    print(f"📊 Unique n_params: {df['n_params'].nunique()}  range: [{df['n_params'].min():,}, {df['n_params'].max():,}]")
    ib = df['psnr'].idxmax()
    print(f"   best trial #{int(df.loc[ib, 'trial'])}: PSNR = {df.loc[ib, 'psnr']:.4f} dB | "
          f"#Params = {df.loc[ib, 'n_params']:,}")

    safe = re.sub(r'[^\w\-]+', '_', f"{MODEL_DISPLAY_NAME}_{study_name}")
    check_against_previous(df, [f"{safe}_3cols.xlsx", f"{safe}_3cols.csv", f"{safe}_full.xlsx"])
    cols = ['trial', 'n_params'] + (['flops_G'] if 'flops_G' in df.columns else []) + ['psnr']
    three = df[cols].rename(columns={
        'trial': 'Trial Number', 'n_params': 'Number of Parameters',
        'flops_G': f'FLOPs (G) @ {FLOPS_IMG_H}x{FLOPS_IMG_W}', 'psnr': 'PSNR (dB)'})
    fmt3 = {'Number of Parameters': '#,##0', 'PSNR (dB)': '0.0000',
            f'FLOPs (G) @ {FLOPS_IMG_H}x{FLOPS_IMG_W}': '0.000'}
    _write_excel(three, OUTPUT_DIR / f"{safe}_3cols.xlsx", fmt3)
    three.to_csv(OUTPUT_DIR / f"{safe}_3cols.csv", index=False, encoding='utf-8-sig')
    full = df.drop(columns=['n_plot'])
    _write_excel(full, OUTPUT_DIR / f"{safe}_full.xlsx",
                 {'n_params': '#,##0', 'n_params_effective': '#,##0', 'n_unused': '#,##0',
                  'psnr': '0.0000', 'flops_G': '0.000'})
    print(f"\n📄 Excel: {safe}_3cols.xlsx + {safe}_full.xlsx")

    print("\n🎨 Plots:")
    plot_single_study(df, study_name)

    print("\n" + "=" * 78 + f"\n🎉 Analysis complete — {MODEL_DISPLAY_NAME}\n📂 Output: {OUTPUT_DIR}\n" + "=" * 78)


if __name__ == "__main__":
    main()