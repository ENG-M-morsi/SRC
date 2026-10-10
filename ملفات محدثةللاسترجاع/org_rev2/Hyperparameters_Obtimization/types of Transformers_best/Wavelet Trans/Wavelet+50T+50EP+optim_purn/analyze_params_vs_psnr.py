# =====================================================================
# analyze_params_vs_psnr.py  —  WaveletAttention
# =====================================================================
# نموذج: HUTCN + WaveletAttention (DWT/IDWT + Window Attention + ConvFFN)
# يحسب #Params لكل trial بصيغة تحليلية دقيقة — بدون بناء النموذج.
# يُخرج: Excel (3 أعمدة + كامل) + رسومات 20/21/22 بثلاث امتدادات.
# =====================================================================

import os, sys, re, json, hashlib, time, warnings
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
MODEL_DISPLAY_NAME = "WaveletAttention"

HERE         = Path(__file__).resolve().parent
STORAGE_PATH = HERE / "optuna_study.db"
OUTPUT_DIR   = HERE / "params_vs_psnr_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_FILE   = OUTPUT_DIR / "n_params_cache.json"

STUDY_NAME         = "hutcn_hyperopt_study"
AUTO_SCAN_SIBLINGS = False

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
# SECTION 3 — ANALYTIC FORMULA (verified: trial #26 → 5,398,297)
# ═════════════════════════════════════════════════════════════════════
def wavelet_hutcn_nparams(nf, num_heads, ws, num_blocks,
                          out_nc=3, upscale=4):
    """
    صيغة تحليلية دقيقة لـ HUTCN + WaveletAttention.

    المشتقة من:
      dhtcun.py               → HUTCN
      dhtcu_block.py          → ESA, TESA, TCN, P_HTCB
      custom_attention_blocks → haar_dwt/idwt, WaveletWindowAttention,
                                ConvFFN, LN2d, WaveletTransformerBlock,
                                WaveletAttention
    """
    f   = nf // 4
    up2 = upscale * upscale
    d4  = 4 * nf                    # channels بعد concat subbands

    # ─── ESA (dhtcu_block.py — مع conv_f معرّف) ─────────────────
    # conv1:  nf→f  k=1  → nf·f + f
    # conv2:  f→f   k=3  → 9·f² + f
    # conv3:  f→f   k=3  → 9·f² + f
    # conv_f: f→f   k=1  → f² + f    (معرّف — غير مستخدم في forward)
    # conv4:  f→nf  k=3  → 9·f·nf + nf
    esa = 10 * f * nf + 19 * f * f + 4 * f + nf

    # ─── WaveletWindowAttention ─────────────────────────────────
    # qkv       : Linear(d4, 3·d4, bias=True) → 3·d4² + 3·d4
    # proj      : Linear(d4, d4, bias=True)   → d4² + d4
    # rpb_table : (2ws−1)² · nh
    # ↳ = 4·d4² + 4·d4 + (2ws−1)²·nh
    # with d4 = 4·nf:  64·nf² + 16·nf + (2ws−1)²·nh
    wwa = 64 * nf * nf + 16 * nf + (2 * ws - 1) ** 2 * num_heads

    # ─── ConvFFN (expand=1, so hid = d4) ────────────────────────
    # pw1 : Conv2d(d4, d4, 1)  → d4² + d4
    # dw  : Conv2d(d4, d4, 3, groups=d4) → 9·d4 + d4 = 10·d4
    # pw2 : Conv2d(d4, d4, 1)  → d4² + d4
    # ↳ = 2·d4² + 12·d4
    # with d4 = 4·nf:  32·nf² + 48·nf
    convffn = 32 * nf * nf + 48 * nf

    # ─── LN2d ────────────────────────────────────────────────────
    # LayerNorm(d4) → 2·d4 = 8·nf
    ln2d = 2 * d4

    # ─── WaveletTransformerBlock ────────────────────────────────
    # LN2d + WWA + LN2d + ConvFFN = 16·nf + WWA + ConvFFN
    wtb = 2 * ln2d + wwa + convffn

    # ─── WaveletAttention ───────────────────────────────────────
    # num_blocks × WTB + res_scale (1)
    wattn = num_blocks * wtb + 1

    # ─── TCN = WaveletAttention ─────────────────────────────────
    tcn = wattn

    # ─── P_HTCB ─────────────────────────────────────────────────
    # tesa_in (TESA=3·ESA) + tcn1 + c (nf²+nf) + tesa_out (TESA=3·ESA)
    ptcb = 6 * esa + tcn + (nf * nf + nf)

    # ─── HUTCN ──────────────────────────────────────────────────
    # fea_conv (3→nf, k=1)          → 3·nf + nf = 4·nf
    # post_unet_esa                 → ESA
    # post_unet_conv (nf→nf, k=1)   → nf² + nf
    # B1                            → P_HTCB
    # LR_conv1 (nf→nf, k=1)         → nf² + nf   (معرّف — غير مستخدم)
    # LR_conv2 (nf→nf, k=1)         → nf² + nf   (معرّف — غير مستخدم)
    # recon_conv1 (nf→nf, k=3)      → 9·nf² + nf
    # recon_conv2 (nf→3·up², k=3)   → 9·nf·3·up² + 3·up²
    total  = 7 * esa                    # 1 (post) + 6 (P_HTCB)
    total += wattn
    total += 13 * nf * nf               # post_conv + c + LR1 + LR2 + recon1(9×)
    total += 441 * nf                   # fea(4) + post(1) + c(1) + LR1(1) + LR2(1) + recon1(1) + recon2(432)
    total += 48                         # recon2 bias = 3·up² = 48

    return int(total)


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
    prefix = MODEL_DISPLAY_NAME + "|"
    return hashlib.md5(
        (prefix + json.dumps(params, sort_keys=True, default=str)).encode()
    ).hexdigest()


def compute_nparams(params, cache, user_attr=None):
    if user_attr is not None:
        return int(user_attr)

    key = _cache_key(params)
    if key in cache:
        return cache[key]

    p = dict(params)
    nf        = p.get('n_feats', p.get('nf', 64))
    num_heads = p.get('num_heads', 2)
    ws        = p.get('window_size', 16)
    nb        = p.get('num_blocks', 1)

    try:
        n = wavelet_hutcn_nparams(nf, num_heads, ws, nb)
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
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        n = compute_nparams(t.params, cache,
                            user_attr=t.user_attrs.get('n_params'))
        if n is None:
            continue
        row = {'trial': t.number, 'n_params': int(n), 'psnr': float(t.value)}
        for k, v in t.params.items():
            row[f"hp_{k}"] = v
        rows.append(row)
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

    # ── 20 — PSNR vs #Params ─────────────────────────────────────
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

    # ── 21 — ملوّن بكل HP ────────────────────────────────────────
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

    # ── 22 — Mean PSNR bar per HP value ──────────────────────────
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

    cache = load_cache()
    print(f"🗄️  Cache: {len(cache)} entries")
    print("⏳ Computing n_params (analytic formula — instant)...")
    t0 = time.time()
    df = build_table(study, cache)
    save_cache(cache)
    print(f"✅ Done in {time.time()-t0:.3f}s | {len(df)} usable trials")
    if df.empty:
        print("❌ لا بيانات صالحة.")
        sys.exit(1)

    print(f"\n📊 Unique n_params: {df['n_params'].nunique()}")
    print(f"   range: [{df['n_params'].min():,}, {df['n_params'].max():,}]")
    i_best = df['psnr'].idxmax()
    print(f"   best trial #{int(df.loc[i_best,'trial'])}: "
          f"PSNR = {df.loc[i_best,'psnr']:.4f} dB | "
          f"#Params = {df.loc[i_best,'n_params']:,}")

    safe = re.sub(r'[^\w\-]+', '_', f"{MODEL_DISPLAY_NAME}_{study_name}")
    three = df[['trial', 'n_params', 'psnr']].copy()
    three.columns = ['Trial Number', 'Number of Parameters', 'PSNR (dB)']
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