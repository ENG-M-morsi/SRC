# =====================================================================
# analyze_params_vs_psnr.py  —  SRFormer (PSA)
# =====================================================================
# نموذج: HUTCN + PSA (Pyramid Structure Attention)
#
# ✅ snapshot-first: يستورد النموذج من ./_snapshot_model/
# ✅ fallback: shared model/ (إن لم يوجد snapshot)
# ✅ فحص صريح لتوقيع HUTCN قبل البدء
# ✅ FLOPs عبر fvcore من نفس بنية الموديل
# ✅ Excel (3cols + full) + رسومات 20/21/22/23 × 3 امتدادات
#
# ملاحظة: SRFormer يستخدم HP إضافي — ffn_ratio —
#         يُمرَّر إلى HUTCN كما هو مطلوب في بنيته.
# =====================================================================

import os, sys, re, json, hashlib, time, warnings, importlib, inspect
from pathlib import Path
warnings.filterwarnings('ignore')

import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ═════════════════════════════════════════════════════════════════════
MODEL_DISPLAY_NAME = "SRFormer"

HERE         = Path(__file__).resolve().parent
STORAGE_PATH = HERE / "optuna_study.db"
OUTPUT_DIR   = HERE / "params_vs_psnr_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE   = OUTPUT_DIR / "n_params_cache.json"

STUDY_NAME         = "hutcn_hyperopt_study"
AUTO_SCAN_SIBLINGS = False

# ═══ Model source ═══
SNAPSHOT_DIR      = HERE / "_snapshot_model"
SHARED_MAX_LEVELS = 12

# ═══ FLOPs ═══
COMPUTE_FLOPS    = True
FLOPS_IMG_H      = 48           # ← 🟠 ارتفاع صورة الإدخال (LR) — عدّل هنا
FLOPS_IMG_W      = 48           # ← 🟠 عرض  صورة الإدخال (LR) — عدّل هنا
FLOPS_IN_CH      = 3
FLOPS_CACHE_FILE = OUTPUT_DIR / "flops_cache.json"

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
# SECTION 3 — ANALYTIC FORMULA (verified: trial #31 → 1,309,518)
# ═════════════════════════════════════════════════════════════════════
def srformer_hutcn_nparams(nf, num_heads, ws, num_blocks, ffn_ratio,
                           out_nc=3, upscale=4):
    """
    صيغة تحليلية دقيقة لـ HUTCN + PSA (SRFormer).

    المشتقة من:
      dhtcun.py               → HUTCN
      dhtcu_block.py          → ESA, TESA, TCN, P_HTCB
      custom_attention_blocks → ContinuousRPB, LightFFN, PSA_Block, PSA
    """
    f   = nf // 4
    up2 = upscale * upscale
    mid = max(int(nf * ffn_ratio), nf)     # LightFFN hidden

    # ─── ESA (بدون conv_f) ─────────────────────────────────────
    # conv1: nf→f  k=1  → nf·f + f
    # conv2: f→f   k=3  → 9·f² + f
    # conv3: f→f   k=3  → 9·f² + f
    # conv4: f→nf  k=3  → 9·f·nf + nf
    esa = 10 * f * nf + 18 * f * f + 3 * f + nf

    # ─── ContinuousRPB ────────────────────────────────────────
    # Linear(2,16,bias=True)     → 2·16 + 16 = 48
    # Linear(16,nh,bias=False)   → 16·nh
    crpb = 48 + 16 * num_heads

    # ─── PSA_Block (per block) ────────────────────────────────
    # norm_attn : LayerNorm(nf)                    → 2·nf
    # norm_ffn  : LayerNorm(nf)                    → 2·nf
    # compress  : Linear(4nf, nf, bias=False)      → 4·nf²
    # qkv       : Linear(nf, 3nf, bias=True)       → 3·nf² + 3·nf
    # crpb      : ContinuousRPB                    → 48 + 16·nh
    # proj      : Linear(nf, nf, bias=True)        → nf² + nf
    # expand    : Linear(nf, 4nf, bias=False)      → 4·nf²
    # ffn.fc1   : Linear(nf, mid, bias=True)       → nf·mid + mid
    # ffn.dw    : Conv2d(mid, mid, 3, groups=mid)  → 9·mid
    # ffn.fc2   : Linear(mid, nf, bias=True)       → mid·nf + nf
    psa_block = (12 * nf * nf
                 + 10 * nf
                 + crpb
                 + 2 * nf * mid
                 + 10 * mid)

    # ─── PSA = num_blocks × PSA_Block + final LayerNorm ──────
    psa = num_blocks * psa_block + 2 * nf

    # ─── TCN = PSA ────────────────────────────────────────────
    tcn = psa

    # ─── P_HTCB ───────────────────────────────────────────────
    ptcb = 6 * esa + tcn + (nf * nf + nf)

    # ─── HUTCN ────────────────────────────────────────────────
    total  = 7 * esa                    # 1 (post) + 3 (tesa_in) + 3 (tesa_out)
    total += psa
    total += 13 * nf * nf               # post_conv + c + LR1 + LR2 + recon1(9×)
    total += 9 * nf                     # biases: fea(4) + post(1) + c(1) + LR1(1) + LR2(1) + recon1(1)
    total += 9 * nf * out_nc * up2      # recon_conv2
    total += out_nc * up2               # recon_conv2 bias

    return int(total)


# ═════════════════════════════════════════════════════════════════════
# SECTION 3.5 — Model loading (snapshot-first) + FLOPs
# ═════════════════════════════════════════════════════════════════════
_MODEL_PATH_RESOLVED = None
_HUTCN_CLASS = None


def _resolve_model_path():
    global _MODEL_PATH_RESOLVED
    if _MODEL_PATH_RESOLVED is not None:
        return _MODEL_PATH_RESOLVED

    # 1) snapshot محلي
    if SNAPSHOT_DIR.is_dir() and (SNAPSHOT_DIR / "dhtcun.py").exists():
        init = SNAPSHOT_DIR / "__init__.py"
        if not init.exists():
            init.write_text("# auto-generated\n")
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        _MODEL_PATH_RESOLVED = "_snapshot_model.dhtcun"
        print(f"   ℹ️  using SNAPSHOT: {SNAPSHOT_DIR.name}/")
        return _MODEL_PATH_RESOLVED

    # 2) shared model/ صعوداً
    cur = HERE
    for _ in range(SHARED_MAX_LEVELS):
        if (cur / "model").is_dir():
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            _MODEL_PATH_RESOLVED = "model.dhtcun"
            print(f"   ℹ️  using SHARED model/ at: {cur}")
            return _MODEL_PATH_RESOLVED
        if cur.parent == cur:
            break
        cur = cur.parent

    raise RuntimeError(
        "لا يوجد model/ في _snapshot_model/ ولا في أي مجلد أب."
    )


def _get_HUTCN():
    global _HUTCN_CLASS
    if _HUTCN_CLASS is not None:
        return _HUTCN_CLASS
    mod_name = _resolve_model_path()
    mod = importlib.import_module(mod_name)
    _HUTCN_CLASS = mod.HUTCN
    try:
        sig = inspect.signature(_HUTCN_CLASS.__init__)
        names = sorted(k for k in sig.parameters if k != 'self')
        print(f"   ℹ️  HUTCN signature: {names}")
    except Exception:
        pass
    return _HUTCN_CLASS


def build_model_for_flops(params):
    HUTCN = _get_HUTCN()
    p = dict(params)
    nf        = p.get('nf', p.get('n_feats', 64))
    num_heads = p.get('num_heads', 2)
    ws        = p.get('window_size', 8)
    nb        = p.get('num_blocks', 1)
    fr        = p.get('ffn_ratio', 1.5)

    kwargs = dict(in_nc=3, nf=nf, num_modules=1, out_nc=3, upscale=4,
                  num_heads=num_heads, window_size=ws,
                  num_blocks=nb, ffn_ratio=fr)

    try:
        sig = inspect.signature(HUTCN.__init__)
        valid = {k for k in sig.parameters if k != 'self'}
        missing = set(kwargs) - valid
        if missing:
            raise TypeError(
                f"HUTCN لا يقبل المفاتيح: {missing}. "
                f"التوقيع الحالي: {sorted(valid)}. "
                f"تأكد أن _snapshot_model/ يحتوي نسخة SRFormer الصحيحة."
            )
    except (ValueError, TypeError) as e:
        if isinstance(e, TypeError):
            raise

    return HUTCN(**kwargs)


def load_flops_cache():
    if FLOPS_CACHE_FILE.exists():
        try:
            return json.loads(FLOPS_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_flops_cache(cache):
    FLOPS_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _flops_key(params):
    data = dict(params)
    data['_H'] = FLOPS_IMG_H
    data['_W'] = FLOPS_IMG_W
    data['_C'] = FLOPS_IN_CH
    return hashlib.md5(
        (MODEL_DISPLAY_NAME + "|flops|"
         + json.dumps(data, sort_keys=True, default=str)).encode()
    ).hexdigest()


_FLOP_ERROR_SHOWN = False


def compute_flops(params, cache):
    global _FLOP_ERROR_SHOWN
    if not COMPUTE_FLOPS:
        return None
    key = _flops_key(params)
    if key in cache:
        return cache[key]
    try:
        import torch
        from fvcore.nn import FlopCountAnalysis
        model = build_model_for_flops(params)
        model.eval()
        x = torch.randn(1, FLOPS_IN_CH, FLOPS_IMG_H, FLOPS_IMG_W)
        with torch.no_grad():
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            total = fca.total()
        del model, x
        gflops = total / 1e9
        cache[key] = float(gflops)
        return float(gflops)
    except Exception as e:
        if not _FLOP_ERROR_SHOWN:
            import traceback
            print("\n" + "=" * 78)
            print("   ❌ FLOPs ERROR (أول فشل — التفصيل الكامل)")
            print("=" * 78)
            traceback.print_exc()
            print(f"   Params at failure: {dict(params)}")
            print("=" * 78 + "\n")
            _FLOP_ERROR_SHOWN = True
        return None


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — n_params (formula + cache)
# ═════════════════════════════════════════════════════════════════════
def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _cache_key(params):
    return hashlib.md5(
        (MODEL_DISPLAY_NAME + "|"
         + json.dumps(params, sort_keys=True, default=str)).encode()
    ).hexdigest()


def compute_nparams(params, cache, user_attr=None):
    if user_attr is not None:
        return int(user_attr)
    key = _cache_key(params)
    if key in cache:
        return cache[key]
    p  = dict(params)
    nf        = p.get('nf', p.get('n_feats', 64))
    num_heads = p.get('num_heads', 2)
    ws        = p.get('window_size', 8)
    nb        = p.get('num_blocks', 1)
    fr        = p.get('ffn_ratio', 1.5)
    try:
        n = srformer_hutcn_nparams(nf, num_heads, ws, nb, fr)
        cache[key] = int(n)
        return int(n)
    except Exception as e:
        print(f"      ⚠️  formula failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════
# SECTION 5 — Study loading
# ═════════════════════════════════════════════════════════════════════
def list_studies(db_path):
    url = f"sqlite:///{db_path}"
    try:
        return [s.study_name
                for s in optuna.study.get_all_study_summaries(storage=url)]
    except Exception as e:
        print(f"   ⚠️  cannot read {db_path}: {e}")
        return []


def load_first_study(db_path):
    names = list_studies(db_path)
    if not names:
        return None, None
    name = STUDY_NAME if STUDY_NAME in names else names[0]
    try:
        study = optuna.load_study(study_name=name,
                                  storage=f"sqlite:///{db_path}")
        return study, name
    except Exception as e:
        print(f"   ⚠️  cannot load '{name}': {e}")
        return None, None


def build_table(study, cache):
    flops_cache = load_flops_cache() if COMPUTE_FLOPS else {}
    rows = []
    n_total = sum(1 for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE
                  and t.value is not None)
    done = 0
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        n = compute_nparams(t.params, cache,
                            user_attr=t.user_attrs.get('n_params'))
        if n is None:
            continue
        row = {'trial': t.number, 'n_params': int(n), 'psnr': float(t.value)}
        if COMPUTE_FLOPS:
            row['flops_G'] = compute_flops(t.params, flops_cache)
        for k, v in t.params.items():
            row[f"hp_{k}"] = v
        rows.append(row)
        done += 1
        if COMPUTE_FLOPS and (done % 5 == 0 or done == n_total):
            print(f"      ... FLOPs {done}/{n_total}")
    if COMPUTE_FLOPS:
        save_flops_cache(flops_cache)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════
# SECTION 6 — Pareto + plots
# ═════════════════════════════════════════════════════════════════════
def pareto_front(p, v):
    order = np.argsort(p)
    best, idx = -np.inf, []
    for i in order:
        if v[i] > best:
            idx.append(i); best = v[i]
    return np.array(idx)


def plot_single_study(df, study_name):
    label = MODEL_DISPLAY_NAME
    p = df['n_params'].values.astype(float)
    v = df['psnr'].values.astype(float)
    pf = pareto_front(p, v)

    # ── 20 ──
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(p, v, c=v, cmap='viridis', s=90, alpha=0.85,
                    edgecolors='black', linewidths=0.6)
    ax.plot(p[pf], v[pf], 'r--o', lw=2, ms=8,
            label='Pareto front', zorder=5)
    ib = int(np.argmax(v))
    ax.scatter(p[ib], v[ib], marker='*', s=380, color='gold',
               edgecolors='black', zorder=6,
               label=f'Best PSNR = {v[ib]:.3f} dB ({p[ib]/1e6:.2f}M)')
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'PSNR vs #Parameters — {label}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.colorbar(sc, ax=ax, label='PSNR (dB)')
    fig.tight_layout()
    save_figure(fig, f"20_{label}_psnr_vs_params")

    # ── 21 ──
    for hp in [c for c in df.columns if c.startswith('hp_')]:
        clean = hp.replace('hp_', '')
        s = df[hp]
        is_num = pd.api.types.is_numeric_dtype(s) and s.nunique() > 6
        fig, ax = plt.subplots(figsize=(11, 7))
        if is_num:
            sc = ax.scatter(p, v, c=s.astype(float), cmap='plasma',
                            s=90, alpha=0.85, edgecolors='black',
                            linewidths=0.6)
            plt.colorbar(sc, ax=ax, label=clean)
        else:
            uniq = sorted(s.dropna().unique().tolist(),
                          key=lambda x: (str(type(x)), x))
            cmap = cm.get_cmap('tab10', max(len(uniq), 2))
            for i, u in enumerate(uniq):
                m = (s == u)
                ax.scatter(p[m], v[m], color=cmap(i), s=90, alpha=0.85,
                           edgecolors='black', linewidths=0.6, label=str(u))
            ax.legend(title=clean, loc='lower right', fontsize=8)
        ax.plot(p[pf], v[pf], 'k--', lw=1.5, alpha=0.7, label='Pareto front')
        ax.set_xlabel('Number of parameters')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'PSNR vs #Params — {label} — by {clean}')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, f"21_{label}_by_{clean}")

    # ── 22 ──
    for hp in [c for c in df.columns if c.startswith('hp_')]:
        clean = hp.replace('hp_', '')
        s = df[hp]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            try:
                groups = pd.cut(s, bins=8).astype(str)
            except Exception:
                continue
        else:
            groups = s.astype(str)
        grp = df.groupby(groups)['psnr'].agg(['mean', 'std', 'count'])
        grp = grp.sort_index()
        if len(grp) < 2:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(grp) * 0.8), 6))
        bars = ax.bar(range(len(grp)), grp['mean'].fillna(0),
                      yerr=grp['std'].fillna(0), capsize=4,
                      color='steelblue', edgecolor='black', alpha=0.85)
        for i, (bar, cnt) in enumerate(zip(bars, grp['count'])):
            h = bar.get_height()
            if not np.isfinite(h):
                continue
            std_i = grp['std'].iloc[i]
            y_off = (std_i if np.isfinite(std_i) else 0) + 0.05
            ax.text(bar.get_x() + bar.get_width() / 2, h + y_off,
                    f'n={cnt}', ha='center', fontsize=8)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index, rotation=40, ha='right', fontsize=9)
        ax.set_xlabel(clean)
        ax.set_ylabel('Mean PSNR (dB)')
        ax.set_title(f'Mean PSNR per value of {clean} — {label}')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        save_figure(fig, f"22_{label}_mean_psnr_by_{clean}")

    # ── 23 — FLOPs vs PSNR ──
    if 'flops_G' in df.columns and df['flops_G'].notna().any():
        valid = df.dropna(subset=['flops_G'])
        f  = valid['flops_G'].values.astype(float)
        vf = valid['psnr'].values.astype(float)
        pf_f = pareto_front(f, vf)

        fig, ax = plt.subplots(figsize=(11, 7))
        sc = ax.scatter(f, vf, c=vf, cmap='viridis', s=90, alpha=0.85,
                        edgecolors='black', linewidths=0.6)
        ax.plot(f[pf_f], vf[pf_f], 'r--o', lw=2, ms=8,
                label='Pareto front', zorder=5)
        ib = int(np.argmax(vf))
        ax.scatter(f[ib], vf[ib], marker='*', s=380, color='gold',
                   edgecolors='black', zorder=6,
                   label=f'Best PSNR = {vf[ib]:.3f} dB '
                         f'({f[ib]:.2f} GFLOPs)')
        ax.set_xlabel(f'FLOPs (G) — input {FLOPS_IMG_H}×{FLOPS_IMG_W}')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'PSNR vs FLOPs — {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        plt.colorbar(sc, ax=ax, label='PSNR (dB)')
        fig.tight_layout()
        save_figure(fig, f"23_{label}_psnr_vs_flops")
    else:
        print("   ℹ️  لا FLOPs — تخطي شكل 23.")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print(f"📂 Folder: {HERE}")
    print(f"🗄️  DB    : {STORAGE_PATH}")
    print(f"🧠 Model : {MODEL_DISPLAY_NAME}")
    print("=" * 78)

    if not STORAGE_PATH.exists():
        print(f"❌ لم يُعثر على {STORAGE_PATH}")
        sys.exit(1)

    os.chdir(HERE)

    study, study_name = load_first_study(STORAGE_PATH)
    if study is None:
        print("❌ لا توجد دراسة قابلة للقراءة.")
        sys.exit(1)
    print(f"\n✅ Study: '{study_name}' — {len(study.trials)} trials")

    if COMPUTE_FLOPS:
        print("🔎 Pre-checking model build...")
        try:
            sample = dict(study.trials[0].params) if study.trials else {}
            _ = build_model_for_flops(sample)
            print("   ✅ Model builds OK — FLOPs ستعمل")
        except Exception as e:
            print(f"   ❌ فشل بناء النموذج: {e}")
            if SNAPSHOT_DIR.is_dir():
                print(f"   → snapshot موجود: {SNAPSHOT_DIR}")
            sys.exit(1)

    cache = load_cache()
    print(f"\n🗄️  Cache: {len(cache)} entries")
    print("⏳ Computing n_params + FLOPs...")
    t0 = time.time()
    df = build_table(study, cache)
    save_cache(cache)
    print(f"✅ Done in {time.time()-t0:.3f}s | {len(df)} usable trials")
    if df.empty:
        print("❌ لا بيانات صالحة.")
        sys.exit(1)

    if COMPUTE_FLOPS and 'flops_G' in df.columns:
        n_valid = df['flops_G'].notna().sum()
        print(f"📊 FLOPs: {n_valid}/{len(df)} صف ممتلئ")

    print(f"\n📊 Unique n_params: {df['n_params'].nunique()}")
    print(f"   range: [{df['n_params'].min():,}, {df['n_params'].max():,}]")
    i_best = df['psnr'].idxmax()
    print(f"   best trial #{int(df.loc[i_best,'trial'])}: "
          f"PSNR = {df.loc[i_best,'psnr']:.4f} dB | "
          f"#Params = {df.loc[i_best,'n_params']:,}")

    safe = re.sub(r'[^\w\-]+', '_', f"{MODEL_DISPLAY_NAME}_{study_name}")
    cols = ['trial', 'n_params']
    if 'flops_G' in df.columns:
        cols.append('flops_G')
    cols.append('psnr')
    three = df[cols].copy()
    header = {
        'trial':    'Trial Number',
        'n_params': 'Number of Parameters',
        'flops_G':  f'FLOPs (G) @ {FLOPS_IMG_H}x{FLOPS_IMG_W}',
        'psnr':     'PSNR (dB)',
    }
    three = three.rename(columns=header)
    three.to_excel(OUTPUT_DIR / f"{safe}_3cols.xlsx", index=False)
    three.to_csv(OUTPUT_DIR / f"{safe}_3cols.csv",
                 index=False, encoding='utf-8-sig')
    df.to_excel(OUTPUT_DIR / f"{safe}_full.xlsx", index=False)
    print(f"\n📄 Excel: {safe}_3cols.xlsx + {safe}_full.xlsx")

    print("\n🎨 Plots:")
    plot_single_study(df, study_name)

    print("\n" + "=" * 78)
    print(f"🎉 Analysis complete — {MODEL_DISPLAY_NAME}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()