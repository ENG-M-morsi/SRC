# ===================================================================
# analyze_optuna_study_general.py — General-purpose Optuna study analysis  (v2, corrected)
#
# Features:
#   1. 100% generic — works with any model without modification
#   2. Pruned trials split into PERFORMANCE prunes (the Pruner decided) and
#      INVALID-CONFIG prunes (the objective stopped them before training)
#   3. Statistical analysis that is valid for adaptive (TPE) sampling:
#      Spearman / Kruskal-Wallis with BH correction, convergence, run-to-run noise
#      from duplicated configurations, search-space boundary check
#   4. All figures in English, publication-ready (no overlapping labels)
#
# v2 fixes: parameter type taken from Optuna distributions (not from "<=10 unique values");
#           seeded fANOVA computed once; honest pruning rate; circular tests removed;
#           best-so-far drawn as a staircase; log-scale handling; all_trials_full.csv really
#           contains all trials.
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
# 🟠 [Model-specific] — Base settings
#    Only edit these two variables when changing the model
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(SCRIPT_DIR, "optuna_study.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

# 🔵 Automatic study detection
STUDY_NAME = None   # ← leave as None for auto-detection
                    #    or set the name manually, e.g.: "hutcn_hyperopt_separate_v1"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "optuna_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================================================
# 🔵 [Generic] — Global publication-quality plot style
# ===================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.titlepad': 12,
    'axes.labelpad': 8,
    'savefig.dpi': 300,
    'figure.autolayout': False,
})


# ===================================================================
# 🔵 [Generic] — Helper functions
# ===================================================================
def save_figure(fig, name):
    """Save the figure as PDF, PNG, and JPEG (all 300 dpi, tight bounding box).

    JPEG does not support transparency, so the figure/axes background is
    forced to opaque white before export to avoid black backgrounds on
    figures that use a transparent canvas.
    """
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    jpeg_path = os.path.join(OUTPUT_DIR, f"{name}.jpeg")

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    # JPEG: force white background explicitly + high quality (no visible
    # compression artifacts) via pil_kwargs.
    fig.savefig(jpeg_path, dpi=300, bbox_inches='tight', facecolor='white',
                format='jpeg', pil_kwargs={'quality': 95, 'optimize': True})

    print(f"   ✅ {name}.pdf + {name}.png + {name}.jpeg")
    plt.close(fig)


def clear_all_titles(fig):
    """
    🔵 [Generic] Remove every title Optuna may have set on a figure,
    including axes titles set with a non-default loc ('left'/'right'),
    and the figure-level suptitle, so a custom English title can replace
    them without any leftover duplicate text.
    """
    for ax in fig.get_axes():
        for loc in ('left', 'center', 'right'):
            try:
                ax.set_title('', loc=loc)
            except Exception:
                pass
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.set_text('')


def clean_label(name):
    """Turn 'params_xxx' into a readable English-friendly label."""
    return name.replace('params_', '').replace('_', ' ')


def detect_parameters(df, study=None):
    """
    🔵 [Generic] Classify hyperparameters using Optuna's own distributions:
        FloatDistribution / IntDistribution  -> numeric
        CategoricalDistribution              -> categorical
    Integer parameters with <= 12 distinct values are also listed in `discrete`
    (they stay numeric for correlation, but are drawn as discrete values in bar plots).
    Returns (numeric, categorical, discrete, log_params).
    Falls back to the old heuristic only if no distribution is available.
    """
    dists = {}
    if study is not None:
        for t in study.trials:
            for k, d in t.distributions.items():
                dists.setdefault(k, d)

    numeric, categorical, discrete, log_params = [], [], [], set()
    for col in df.columns:
        if not col.startswith('params_'):
            continue
        name = col[len('params_'):]
        d = dists.get(name)
        if isinstance(d, optuna.distributions.CategoricalDistribution):
            categorical.append(col)
        elif isinstance(d, (optuna.distributions.FloatDistribution,
                            optuna.distributions.IntDistribution)):
            numeric.append(col)
            if getattr(d, 'log', False):
                log_params.add(col)
            if isinstance(d, optuna.distributions.IntDistribution) and df[col].nunique() <= 12:
                discrete.append(col)
        else:                                   # fallback (no distribution info)
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                numeric.append(col)
            else:
                categorical.append(col)
    return numeric, categorical, discrete, log_params


def fix_optuna_plot_size(plot_result, size=(15, 10)):
    """
    🔵 [Generic] Fix the figure size returned by optuna.visualization.matplotlib
    and translate any default Optuna text/labels into a clean English style.
    Handles different return types: Axes, ndarray of Axes, Figure
    """
    fig = None
    axes_list = []

    if isinstance(plot_result, np.ndarray):
        if plot_result.size > 0:
            axes_list = list(plot_result.flatten())
            fig = axes_list[0].get_figure()
    elif hasattr(plot_result, 'get_figure'):
        axes_list = [plot_result]
        fig = plot_result.get_figure()
    elif hasattr(plot_result, 'figure'):
        axes_list = [plot_result]
        fig = plot_result.figure
    elif hasattr(plot_result, 'set_size_inches'):
        fig = plot_result
        axes_list = fig.get_axes()

    if fig is not None:
        fig.set_size_inches(*size)
        n_axes = len(axes_list)
        # Denser grids (slice/contour with many parameters) need smaller
        # fonts and more spacing between subplots to avoid any overlap
        # with the colorbar or with neighbouring axis labels.
        tick_size = 8 if n_axes <= 6 else 6
        label_size = 9 if n_axes <= 6 else 7
        for ax in axes_list:
            try:
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_ha('right')
                ax.tick_params(axis='both', labelsize=tick_size)
                ax.title.set_fontsize(11)
                ax.xaxis.label.set_fontsize(label_size)
                ax.yaxis.label.set_fontsize(label_size)
                ax.xaxis.labelpad = 4
                ax.yaxis.labelpad = 4
            except Exception:
                pass
        # NOTE: fig.tight_layout() is intentionally NOT called here.
        # Optuna's contour/parallel-coordinate figures embed a shared
        # colorbar axis inside the same GridSpec; tight_layout() does not
        # know about that extra axis and pushes/overlaps it with the last
        # column's tick labels. Instead we widen the subplot spacing
        # manually (safe for any number of axes, with or without a
        # colorbar) and let bbox_inches='tight' at save time trim the
        # final whitespace without disturbing internal spacing.
        # Deliberately do NOT call subplots_adjust/tight_layout here.
        # Optuna's contour/parallel-coordinate figures place a shared
        # colorbar axis at a fixed GridSpec position computed for the
        # figure's *original* size; any subplots_adjust call after
        # set_size_inches() recomputes subplot positions independently of
        # that colorbar axis and pushes them into each other. The figure's
        # default internal spacing (computed by Optuna itself) is already
        # correct — we only resize the canvas and rely on
        # bbox_inches='tight' at save time to trim outer whitespace.

    return fig


# ===================================================================
# 🔵 [Generic] — Load the study (with automatic name detection)
# ===================================================================
print("=" * 75)
print("📊 Optuna Study Analysis (generic version)")
print("=" * 75)

if not os.path.exists(STORAGE_PATH):
    print(f"❌ Database not found: {STORAGE_PATH}")
    sys.exit(1)

# ✅ Auto-detect the study name
if STUDY_NAME is None:
    try:
        summaries = optuna.study.get_all_study_summaries(storage=STORAGE_URL)
    except Exception as e:
        print(f"❌ Failed to read the database: {e}")
        sys.exit(1)

    if len(summaries) == 0:
        print("❌ No studies found in the database!")
        sys.exit(1)
    elif len(summaries) == 1:
        STUDY_NAME = summaries[0].study_name
        print(f"✅ Detected a single study: '{STUDY_NAME}'")
    else:
        print(f"⚠️ Found {len(summaries)} studies in the database:")
        for i, s in enumerate(summaries, 1):
            n_complete = sum(1 for t in optuna.load_study(
                study_name=s.study_name, storage=STORAGE_URL).trials
                if t.state == optuna.trial.TrialState.COMPLETE)
            print(f"   [{i}] '{s.study_name}' — {n_complete} completed trials")
        print()
        choice = input("👉 Enter the number of the desired study: ").strip()
        try:
            idx = int(choice) - 1
            STUDY_NAME = summaries[idx].study_name
            print(f"✅ Selected: '{STUDY_NAME}'")
        except (ValueError, IndexError):
            print("❌ Invalid selection.")
            sys.exit(1)

# ✅ Load the study
try:
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)
except KeyError:
    print(f"❌ Study '{STUDY_NAME}' was not found in the database.")
    print("💡 Available studies:")
    for s in optuna.study.get_all_study_summaries(storage=STORAGE_URL):
        print(f"   - {s.study_name}")
    sys.exit(1)

# Extract all trials
all_trials = study.trials
complete_trials = [t for t in all_trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
pruned_trials = [t for t in all_trials
                 if t.state == optuna.trial.TrialState.PRUNED]
failed_trials = [t for t in all_trials
                 if t.state == optuna.trial.TrialState.FAIL]
running_trials = [t for t in all_trials
                  if t.state == optuna.trial.TrialState.RUNNING]

# A pruned trial that never reported an intermediate value was stopped by the objective
# itself (e.g. `raise optuna.TrialPruned()` for an invalid architecture before training).
# It says nothing about how the Pruner judges a configuration, so it is analysed separately.
pruned_invalid = [t for t in pruned_trials if not t.intermediate_values]
pruned_perf = [t for t in pruned_trials if t.intermediate_values]
invalid_numbers = {t.number for t in pruned_invalid}
perf_numbers = {t.number for t in pruned_perf}

print(f"\n📚 Study name: {STUDY_NAME}")
print(f"📊 Total trials recorded: {len(all_trials)}")
print(f"✅ Completed:                {len(complete_trials)}")
print(f"✂️ Pruned by the Pruner:     {len(pruned_perf)}   (performance prunes)")
print(f"⛔ Stopped as invalid config: {len(pruned_invalid)}   (pruned before any training)")
print(f"🔄 Still RUNNING (stale):    {len(running_trials)}")
print(f"❌ Failed:                   {len(failed_trials)}")
if running_trials:
    print(f"⚠️ {len(running_trials)} trial(s) are still RUNNING — usually interrupted runs; "
          f"they are ignored in every rate below: {[t.number for t in running_trials]}")

if len(complete_trials) < 2:
    print("⚠️ Too few completed trials for analysis.")
    sys.exit(1)

print(f"\n🏆 Best PSNR: {study.best_value:.4f} dB (Trial #{study.best_trial.number})")

# Pruning rate among trials that actually ran (invalid configs and RUNNING excluded)
n_ran = len(complete_trials) + len(pruned_perf)
pruning_rate = len(pruned_perf) / max(n_ran, 1) * 100
pruning_rate_raw = len(pruned_trials) / max(len(all_trials), 1) * 100
print(f"📉 Pruning rate (Pruner decisions / trials that ran): {pruning_rate:.1f}%   "
      f"[raw incl. invalid configs: {pruning_rate_raw:.1f}%]")
if pruning_rate > 40:
    print("⚠️ Warning: high pruning rate (> 40%). The Pruner may be too aggressive.")
elif pruning_rate > 20:
    print("ℹ️ Note: moderate pruning rate (20-40%). Analysis is acceptable.")
else:
    print("✅ Low pruning rate (< 20%).")

# Convert to DataFrame
df = study.trials_dataframe()
df['pruned_kind'] = ''
df.loc[df['number'].isin(invalid_numbers), 'pruned_kind'] = 'invalid_config'
df.loc[df['number'].isin(perf_numbers), 'pruned_kind'] = 'performance'
df_complete = df[df['state'] == 'COMPLETE'].copy()
df_pruned = df[df['number'].isin(perf_numbers)].copy() if perf_numbers else pd.DataFrame()
df_invalid = df[df['number'].isin(invalid_numbers)].copy() if invalid_numbers else pd.DataFrame()

# 🔵 Auto-detect parameters (from Optuna distributions)
numeric_params, categorical_params, discrete_params, log_params = detect_parameters(df_complete, study)
print(f"\n🔍 Detected {len(numeric_params)} numeric parameter(s) and {len(categorical_params)} categorical parameter(s)")
print(f"   Numeric: {[clean_label(p) for p in numeric_params]}")
print(f"   Categorical: {[clean_label(p) for p in categorical_params]}")
print(f"   Log-sampled: {[clean_label(p) for p in sorted(log_params)]}")

# 🔵 Parameter importance — computed ONCE with a fixed seed (fANOVA is random otherwise)
IMPORTANCES = None
IMPORTANCE_EVALUATOR = None
try:
    from optuna.importance import FanovaImportanceEvaluator
    IMPORTANCE_EVALUATOR = FanovaImportanceEvaluator(seed=42)
    IMPORTANCES = optuna.importance.get_param_importances(study, evaluator=IMPORTANCE_EVALUATOR)
except Exception as e:
    print(f"   ⚠️ Seeded fANOVA unavailable ({e}); falling back to Optuna's default evaluator.")
    try:
        IMPORTANCES = optuna.importance.get_param_importances(study)
    except Exception:
        IMPORTANCES = None


# ===================================================================
# Section 1: Basic plots (Optimization History)
# ===================================================================
print("\n" + "=" * 75)
print("📈 Section 1: Basic plots")
print("=" * 75)
print("   → History Plot...")

fig, ax = plt.subplots(figsize=(12, 6))
values = df_complete['value'].values
best_so_far = np.maximum.accumulate(values)
trial_numbers = df_complete['number'].values

# Trials are independent samples: draw them as points (a line would suggest a trend),
# and the best-so-far as a staircase (it only changes when a better trial appears).
ax.scatter(trial_numbers, values, color='steelblue', s=45, alpha=0.75,
           label=f'PSNR per completed trial (n={len(values)})', zorder=3)
ax.step(trial_numbers, best_so_far, where='post', color='red', linewidth=2,
        label=f'Best cumulative PSNR ({best_so_far[-1]:.3f} dB)', zorder=4)
ax.axhline(y=study.best_value, color='green', linestyle='--',
           alpha=0.5, label=f'Best value = {study.best_value:.3f} dB')
ax.set_xlabel('Trial number (pruned / invalid trials are not shown)')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Optimization History — PSNR across trials')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', framealpha=0.9)
fig.tight_layout()
save_figure(fig, "01_optimization_history")


# ===================================================================
# Section 2: Parameter importance
# ===================================================================
print("   → Parameter Importance...")
try:
    if IMPORTANCE_EVALUATOR is not None:
        plot_result = optuna_vis.plot_param_importances(study, evaluator=IMPORTANCE_EVALUATOR)
    else:
        plot_result = optuna_vis.plot_param_importances(study)
    fig = fix_optuna_plot_size(plot_result, size=(12, 7))
    if fig is not None:
        ax = fig.get_axes()[0]
        clear_all_titles(fig)
        fig.suptitle('Hyperparameter Importance', fontsize=13, fontweight='bold', y=1.02)
        ax.set_xlabel('Importance (relative)', fontsize=11)
        ax.set_ylabel('Hyperparameter', fontsize=11)
        # Clean up default Optuna param_ prefixes on y tick labels if present
        new_labels = [clean_label(t.get_text()) for t in ax.get_yticklabels()]
        ax.set_yticklabels(new_labels)
        fig.tight_layout()
        save_figure(fig, "02_param_importance")
    else:
        print("   ⚠️ Could not obtain a Figure")
except Exception as e:
    print(f"   ⚠️ Failed: {e}")


# ===================================================================
# Section 3: Slice + Contour (work with any parameters)
# ===================================================================
def top_importance_params(study, max_params=4):
    """
    🔵 [Generic] Return the top-N most important parameter names, using the SAME seeded
    importance values that Figure 02 shows (computed once at start-up), so the slice /
    contour grids always agree with the importance figure.
    """
    if not IMPORTANCES:
        return None
    top = list(IMPORTANCES.keys())[:max_params]
    return top if len(top) >= 2 else None


print("   → Slice Plot...")
try:
    top_params_for_grid = top_importance_params(study, max_params=4)
    if top_params_for_grid:
        plot_result = optuna_vis.plot_slice(study, params=top_params_for_grid)
    else:
        plot_result = optuna_vis.plot_slice(study)
    fig = fix_optuna_plot_size(plot_result, size=(18, 12))
    if fig is not None:
        for ax in fig.get_axes():
            xl = ax.get_xlabel()
            if xl:
                ax.set_xlabel(clean_label(xl))
        if fig._suptitle is not None:
            fig._suptitle.set_text('')
        subtitle = ('Slice Plot — Parameter Value vs. Objective' if not top_params_for_grid
                    else f'Slice Plot — Top {len(top_params_for_grid)} Most Important Parameters')
        fig.suptitle(subtitle, fontsize=14, y=1.02)
        save_figure(fig, "03_slice_plot")
    else:
        print("   ⚠️ Could not obtain a Figure")
except Exception as e:
    print(f"   ⚠️ Failed: {e}")

print("   → Contour Plot...")
try:
    if top_params_for_grid:
        plot_result = optuna_vis.plot_contour(study, params=top_params_for_grid)
    else:
        plot_result = optuna_vis.plot_contour(study)
    fig = fix_optuna_plot_size(plot_result, size=(18, 12))
    if fig is not None:
        for ax in fig.get_axes():
            xl, yl = ax.get_xlabel(), ax.get_ylabel()
            if xl:
                ax.set_xlabel(clean_label(xl))
            if yl:
                ax.set_ylabel(clean_label(yl))
        if fig._suptitle is not None:
            fig._suptitle.set_text('')
        subtitle = ('Contour Plot — Parameter Interactions' if not top_params_for_grid
                    else f'Contour Plot — Top {len(top_params_for_grid)} Most Important Parameters')
        fig.suptitle(subtitle, fontsize=14, y=1.02)
        save_figure(fig, "04_contour_plot")
    else:
        print("   ⚠️ Could not obtain a Figure")
except Exception as e:
    print(f"   ⚠️ Failed: {e}")

print("   → Parallel Coordinates...")
try:
    plot_result = optuna_vis.plot_parallel_coordinate(study)
    fig = fix_optuna_plot_size(plot_result, size=(18, 9))
    if fig is not None:
        for ax in fig.get_axes():
            ax.set_title('')  # clear Optuna's default title to avoid a duplicate
            # Clean up underscore-style tick labels (e.g. 'batch_size' ->
            # 'batch size') for visual consistency with every other figure.
            new_xticklabels = [clean_label(t.get_text()) if t.get_text() != 'Objective Value'
                                else t.get_text() for t in ax.get_xticklabels()]
            if new_xticklabels:
                ax.set_xticklabels(new_xticklabels)
        if fig._suptitle is not None:
            fig._suptitle.set_text('')
        fig.suptitle('Parallel Coordinate Plot', fontsize=14, y=0.98)
        save_figure(fig, "05_parallel_coordinates")
    else:
        print("   ⚠️ Could not obtain a Figure")
except Exception as e:
    print(f"   ⚠️ Failed: {e}")


# ===================================================================
# Section 4: Custom scatter plots (automatic, per parameter)
# ===================================================================
print("\n" + "=" * 75)
print("📊 Section 2: Custom Scatter Plots (automatic)")
print("=" * 75)
print("   → Scatter plot for each parameter...")

all_plottable = numeric_params + categorical_params
n_params = len(all_plottable)
n_cols = 3
n_rows = (n_params + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.2 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]

for idx, param in enumerate(all_plottable):
    ax = axes[idx]
    x = df_complete[param].values
    y = df_complete['value'].values

    ax.scatter(x, y, c=y, cmap='viridis',
               s=80, alpha=0.7, edgecolors='black', linewidths=0.5)

    # Best value
    best_idx = np.argmax(y)
    ax.scatter(x[best_idx], y[best_idx], color='red', s=200,
               marker='*', zorder=5, edgecolors='black',
               label=f'Best: {y[best_idx]:.3f}')

    # Numeric parameters: Spearman rank correlation instead of a linear trend line
    # (monotonic, robust, and valid for log-sampled parameters); log axis when log-sampled.
    rho_txt = ''
    if param in numeric_params:
        if param in log_params:
            ax.set_xscale('log')
        try:
            rho, rho_p = stats.spearmanr(x, y)
            rho_txt = f'  (Spearman ρ={rho:+.2f}, p={rho_p:.3f})'
        except Exception:
            pass

    clean_name = clean_label(param)
    ax.set_xlabel(clean_name, fontsize=10, fontweight='bold', labelpad=6)
    ax.set_ylabel('PSNR (dB)', fontsize=10, labelpad=6)
    ax.set_title(f'{clean_name} vs PSNR{rho_txt}', fontsize=9.5, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha('right')

for idx in range(len(all_plottable), len(axes)):
    axes[idx].axis('off')

fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
save_figure(fig, "06_custom_scatter_plots")


# ===================================================================
# Section 5: Correlation heatmap (automatic)
# ===================================================================
print("   → Correlation Heatmap...")

corr_data = df_complete[numeric_params + ['value']].copy()
if len(numeric_params) > 1:
    corr_matrix = corr_data.corr(method='spearman')   # rank correlation (monotonic, outlier-robust)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1)
    ax.grid(False)   # the style's grid lines were drawn over the cells and hid the minus signs

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr_matrix.values[i, j]) > 0.5 else 'black',
                    fontsize=9)

    clean_labels = [clean_label(c) if c != 'value' else 'PSNR' for c in corr_matrix.columns]
    ax.set_xticks(range(len(clean_labels)))
    ax.set_yticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_title('Spearman Rank Correlation Matrix', fontsize=13, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman correlation coefficient', fontsize=10)
    fig.tight_layout()
    save_figure(fig, "07_correlation_heatmap")


# ===================================================================
# Section 6: Top 10 trials analysis (automatic)
# ===================================================================
print("   → Top 10 Trials...")

df_sorted = df_complete.sort_values('value', ascending=False).head(10)
display_params = numeric_params + categorical_params

fig, ax = plt.subplots(figsize=(18, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Trial', 'PSNR'] + [clean_label(p) for p in display_params]
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

for j in range(len(headers)):
    table[(0, j)].set_text_props(fontweight='bold', color='white')

ax.set_title('Top 10 Trials', fontsize=14, fontweight='bold', pad=20)
save_figure(fig, "08_top10_trials")


# ===================================================================
# 🆕 Section 7: Pruned trials analysis (performance prunes only)
# ===================================================================
print("\n" + "=" * 75)
print("✂️ Section 3: Pruned trials analysis")
print("=" * 75)

N_PRUNING_BINS = 8  # number of bins used for continuous numeric parameters


def _is_continuous(param):
    return param in numeric_params and param not in discrete_params


def _bin_edges(series, param):
    lo, hi = float(series.min()), float(series.max())
    if param in log_params and lo > 0:
        return np.logspace(np.log10(lo), np.log10(hi), N_PRUNING_BINS + 1)
    return np.linspace(lo, hi, N_PRUNING_BINS + 1)


def _fmt_val(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


if len(df_pruned) > 0:
    print(f"   Performance-pruned trials: {len(df_pruned)}   |   invalid-config trials: {len(df_invalid)}")

    # 7.1 Distribution of each parameter: completed vs pruned — as SHARE of each group (%),
    #     so the two groups are comparable although they have different sizes.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.2 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, param in enumerate(all_plottable):
        ax = axes[idx]
        if param in df_complete.columns and param in df_pruned.columns:
            complete_vals = df_complete[param].dropna().values
            pruned_vals = df_pruned[param].dropna().values
            if len(pruned_vals) == 0:
                ax.axis('off')
                continue

            if _is_continuous(param):
                allv = pd.Series(np.concatenate([complete_vals, pruned_vals]).astype(float))
                bins = _bin_edges(allv, param)
                ax.hist(complete_vals, bins=bins, alpha=0.6, color='green',
                        weights=np.full(len(complete_vals), 100 / len(complete_vals)),
                        label=f'Completed (n={len(complete_vals)})')
                ax.hist(pruned_vals, bins=bins, alpha=0.6, color='red',
                        weights=np.full(len(pruned_vals), 100 / len(pruned_vals)),
                        label=f'Pruned (n={len(pruned_vals)})')
                if param in log_params:
                    ax.set_xscale('log')
            else:
                all_vals = sorted(set(list(complete_vals) + list(pruned_vals)))
                complete_pct = [np.mean(complete_vals == v) * 100 for v in all_vals]
                pruned_pct = [np.mean(pruned_vals == v) * 100 for v in all_vals]
                x_pos = np.arange(len(all_vals))
                width = 0.35
                ax.bar(x_pos - width / 2, complete_pct, width, alpha=0.6, color='green',
                       label=f'Completed (n={len(complete_vals)})')
                ax.bar(x_pos + width / 2, pruned_pct, width, alpha=0.6, color='red',
                       label=f'Pruned (n={len(pruned_vals)})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([_fmt_val(v) for v in all_vals], rotation=30, ha='right')

            clean_name = clean_label(param)
            ax.set_xlabel(clean_name, fontsize=10, fontweight='bold', labelpad=6)
            ax.set_ylabel('Share of group (%)', fontsize=10, labelpad=6)
            ax.set_title(f'{clean_name}: Completed vs Pruned',
                         fontsize=10, fontweight='bold', pad=10)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=8)

    for idx in range(len(all_plottable), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Performance-pruned trials only (n={len(df_pruned)}); '
                 f'{len(df_invalid)} invalid-config trials excluded', fontsize=12, y=1.0)
    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    save_figure(fig, "12_pruned_vs_complete_distribution")
else:
    print("   ℹ️ No performance-pruned trials (the Pruner never stopped a trial).")

# 7.2 Table: which parameter values get pruned more often?  (denominator = trials that ran)
print("   → Analyzing pruning regions...")
pruning_analysis = {}
df_relevant = df[(df['state'] == 'COMPLETE') | (df['number'].isin(perf_numbers))]

for param in all_plottable:
    if param not in df.columns or len(df_pruned) == 0:
        continue
    if _is_continuous(param):
        edges = _bin_edges(df_relevant[param].dropna(), param)
        binned = pd.cut(df_relevant[param], bins=edges, include_lowest=True)
        all_counts = df_relevant.groupby(binned, observed=True).size()
        pruned_binned = pd.cut(df_pruned[param], bins=binned.cat.categories)
        pruned_counts = df_pruned.groupby(pruned_binned, observed=True).size()
        label_fn = lambda interval: f"[{interval.left:.3g}, {interval.right:.3g}]"
    else:
        all_counts = df_relevant.groupby(param).size()
        pruned_counts = df_pruned.groupby(param).size()
        label_fn = _fmt_val

    param_analysis = []
    for val in all_counts.index:
        total = all_counts[val]
        if total == 0:
            continue
        pruned = pruned_counts.get(val, 0)
        param_analysis.append((label_fn(val), total, pruned, pruned / total * 100))
    pruning_analysis[param] = param_analysis

pruning_report_path = os.path.join(OUTPUT_DIR, "pruning_analysis.txt")
with open(pruning_report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("Pruned Trials Analysis — Which parameter values lead to pruning?\n")
    f.write("=" * 75 + "\n\n")
    f.write(f"Total trials recorded: {len(all_trials)}\n")
    f.write(f"Completed: {len(complete_trials)}\n")
    f.write(f"Pruned by the Pruner (performance): {len(pruned_perf)}\n")
    f.write(f"Stopped as invalid config (before training): {len(pruned_invalid)}\n")
    f.write(f"Still RUNNING (stale): {len(running_trials)}\n")
    f.write(f"Pruning rate (Pruner decisions / trials that ran): {pruning_rate:.1f}%\n")
    f.write(f"Raw rate including invalid configs: {pruning_rate_raw:.1f}%\n")
    f.write("Rates per value below use only completed + performance-pruned trials.\n")

    for param, analysis in pruning_analysis.items():
        clean = clean_label(param)
        f.write(f"\n{clean}:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Value':>15} | {'Total':>8} | {'Pruned':>8} | {'Rate %':>8}\n")
        f.write("-" * 50 + "\n")
        for val, total, pruned, rate in sorted(analysis, key=lambda x: x[3], reverse=True):
            marker = "HIGH" if rate > 50 else "MED" if rate > 25 else "LOW"
            f.write(f"{str(val):>15} | {total:>8} | {pruned:>8} | {rate:>7.1f}% [{marker}]  (n={total}"
                    f"{', low n' if total < 5 else ''})\n")

    if len(df_invalid) > 0:
        f.write("\n" + "=" * 75 + "\n")
        f.write("Invalid-configuration trials (stopped by the objective, NOT by the Pruner)\n")
        f.write("=" * 75 + "\n")
        for param in all_plottable:
            if param in df_invalid.columns and df_invalid[param].notna().any():
                vc = df_invalid[param].dropna().map(_fmt_val).value_counts()
                f.write(f"  {clean_label(param):>16}: " + ", ".join(f"{k} (x{v})" for k, v in vc.items()) + "\n")
        f.write("  → Consider sampling only valid combinations (e.g. choose num_heads from the divisors\n"
                "    of n_feats) so these trials are not spent.\n")

print(f"   ✅ Pruning analysis saved to: {pruning_report_path}")

# 7.3 Pruning-rate plot for each numeric parameter
for param in numeric_params[:6]:
    if param not in pruning_analysis:
        continue

    fig, ax = plt.subplots(figsize=(10, 6.5))
    analysis = pruning_analysis[param]
    values = [a[0] for a in analysis]
    rates = [a[3] for a in analysis]
    totals = [a[1] for a in analysis]

    bars = ax.bar(range(len(values)), rates,
                  color=['red' if r > 50 else 'orange' if r > 25 else 'green' for r in rates],
                  alpha=0.7, edgecolor='black')
    for i, (bar, total) in enumerate(zip(bars, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'n={total}', ha='center', va='bottom', fontsize=8)

    clean_name = clean_label(param)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values], rotation=40, ha='right', fontsize=8)
    ax.set_xlabel(clean_name, fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Pruning rate (%)', fontsize=11, labelpad=8)
    ax.set_title(f'Pruning rate per value of {clean_name} (Pruner decisions only)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='25%')
    ax.set_ylim(0, max(100, max(rates) * 1.15))
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, f"13_pruning_rate_{clean_name.replace(' ', '_')}")


# ===================================================================
# 🆕 Section 8: Statistical analysis (valid for adaptive / TPE sampling)
# ===================================================================
print("\n" + "=" * 75)
print("📊 Section 4: Statistical analysis")
print("=" * 75)


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def _bound_position(value, dist):
    lo, hi = dist.low, dist.high
    if getattr(dist, 'log', False) and lo > 0:
        return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (value - lo) / (hi - lo) if hi > lo else 0.5


boundary_flags = []      # (param, position, low, high, best value)
stats_report_path = os.path.join(OUTPUT_DIR, "statistical_tests.txt")
with open(stats_report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("Statistical Report\n")
    f.write("=" * 75 + "\n")
    f.write("Note: TPE samples ADAPTIVELY (later trials concentrate near good regions), so trials are\n")
    f.write("not i.i.d. draws; p-values below are descriptive evidence, not formal significance.\n\n")

    # 8.1 Descriptive statistics
    print("   → Descriptive statistics...")
    psnr_values = df_complete['value'].values
    q25, q75 = np.percentile(psnr_values, [25, 75])
    f.write("1) Descriptive statistics for PSNR (completed trials)\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Number of trials: {len(psnr_values)}\n")
    f.write(f"   Median: {np.median(psnr_values):.4f} dB   IQR: [{q25:.4f}, {q75:.4f}]\n")
    f.write(f"   Minimum: {psnr_values.min():.4f} dB   Maximum: {psnr_values.max():.4f} dB\n")
    f.write("   (mean / SD / bootstrap CI are intentionally not reported: the distribution is shaped\n"
            "    by the sampler, mixing many poor early trials with many near-optimal late trials)\n\n")

    # 8.2 Convergence
    print("   → Convergence...")
    order_idx = np.argsort(df_complete['number'].values)
    ordered = psnr_values[order_idx]
    bsf = np.maximum.accumulate(ordered)
    n_c = len(ordered)
    f.write("2) Convergence of the best-so-far PSNR\n")
    f.write("-" * 50 + "\n")
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(round(n_c * frac)))
        f.write(f"   after {k:>3} completed trials ({frac * 100:>3.0f}%): {bsf[k - 1]:.4f} dB\n")
    last_imp = int(np.where(np.diff(bsf, prepend=-np.inf) > 0)[0][-1])
    f.write(f"   last improvement at completed-trial #{last_imp + 1} of {n_c}; "
            f"{n_c - last_imp - 1} trial(s) since.\n")
    half = bsf[-1] - bsf[n_c // 2 - 1] if n_c >= 4 else float('nan')
    f.write(f"   gain over the second half of the search: {half:+.4f} dB\n\n")

    # 8.3 Run-to-run noise from duplicated configurations
    print("   → Noise from duplicated configurations...")
    param_cols = [c for c in df_complete.columns if c.startswith('params_')]
    f.write("3) Run-to-run noise (completed trials with IDENTICAL hyperparameters)\n")
    f.write("-" * 50 + "\n")
    noise = []
    for _, g in df_complete.groupby(param_cols):
        if len(g) > 1:
            noise.append(g['value'].max() - g['value'].min())
            f.write(f"   trials {sorted(g['number'].astype(int).tolist())}: "
                    f"PSNR {', '.join(f'{v:.4f}' for v in g['value'])}  (spread {noise[-1]:.4f} dB)\n")
    if noise:
        f.write(f"   → typical noise ≈ {np.mean(noise):.3f} dB (max {np.max(noise):.3f} dB); "
                f"differences below this are not reliable.\n")
    else:
        f.write("   no duplicated configuration — noise cannot be estimated; repeat the best trial "
                "with several seeds.\n")
    f.write("\n")

    # 8.4 Hyperparameter effects (Spearman for numeric, Kruskal-Wallis for categorical/discrete)
    print("   → Hyperparameter effects...")
    f.write("4) Hyperparameter effects on PSNR (BH-adjusted p-values)\n")
    f.write("-" * 50 + "\n")
    eff_rows = []
    for param in numeric_params:
        x = df_complete[param].values.astype(float)
        if len(np.unique(x)) < 3 or len(x) < 8:
            continue
        rho, p = stats.spearmanr(x, df_complete['value'].values)
        eff_rows.append((clean_label(param), 'Spearman', rho, p, f'rho={rho:+.3f}'))
    for param in categorical_params + discrete_params:
        groups = [g['value'].values for _, g in df_complete.groupby(param) if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            H, p = stats.kruskal(*groups)
            n_tot = sum(len(g) for g in groups)
            eps2 = (H - len(groups) + 1) / max(n_tot - len(groups), 1)
            eff_rows.append((clean_label(param), 'Kruskal-Wallis', H, p, f'eps2={eps2:.3f} ({len(groups)} groups)'))
        except Exception:
            pass
    if eff_rows:
        adj = bh_adjust([r[3] for r in eff_rows])
        for (name, test, stat, p, extra), pa in sorted(zip(eff_rows, adj), key=lambda z: z[0][3]):
            f.write(f"   {name:>18} | {test:<14} | {extra:<28} | p={p:.4f} | p_BH={pa:.4f}\n")
    else:
        f.write("   not enough data\n")
    f.write("\n")

    # 8.5 Completed vs performance-pruned (numeric parameters)
    f.write("5) Completed vs performance-pruned trials\n")
    f.write("-" * 50 + "\n")
    f.write(f"   completed: {len(complete_trials)}   pruned by the Pruner: {len(pruned_perf)}   "
            f"invalid-config (excluded): {len(pruned_invalid)}\n")
    found = False
    if len(df_pruned) > 0:
        for param in numeric_params:
            cv = df_complete[param].dropna().values
            pv = df_pruned[param].dropna().values
            if len(pv) >= 3 and len(cv) >= 3:
                try:
                    s_, p = stats.mannwhitneyu(cv, pv)
                    if p < 0.05:
                        f.write(f"      [DIFFERENT] {clean_label(param)}: p = {p:.4f}\n")
                        found = True
                except Exception:
                    pass
    if not found:
        f.write("      no numeric parameter differs significantly (or too few pruned trials)\n")
    f.write("\n")

    # 8.6 Search-space boundary check
    print("   → Search-space boundary check...")
    f.write("6) Search-space boundary check (is the best value at the edge of its range?)\n")
    f.write("-" * 50 + "\n")
    best_t = study.best_trial
    for param in numeric_params:
        name = param[len('params_'):]
        d = best_t.distributions.get(name)
        if d is None or name not in best_t.params:
            continue
        pos = _bound_position(best_t.params[name], d)
        top5 = df_complete.sort_values('value', ascending=False).head(5)[param].values.astype(float)
        pos5 = float(np.median([_bound_position(v, d) for v in top5]))
        flag = ''
        if pos < 0.05 or pos > 0.95:
            flag = '  ⚠️ AT THE EDGE — widen the range'
            boundary_flags.append((param, pos, d.low, d.high, best_t.params[name]))
        f.write(f"   {clean_label(param):>18}: best={best_t.params[name]:.4g} in [{d.low:g}, {d.high:g}]"
                f"{' (log)' if getattr(d, 'log', False) else ''}  position={pos:.2f}; "
                f"median position of top-5 = {pos5:.2f}{flag}\n")
    f.write("\n")

    # 8.7 How well separated is the best trial?
    f.write("7) How well separated is the best trial?\n")
    f.write("-" * 50 + "\n")
    best_psnr = study.best_value
    top5 = df_complete.sort_values('value', ascending=False).head(5)
    f.write("   top-5: " + ", ".join(f"#{int(n)} ({v:.3f})" for n, v in zip(top5['number'], top5['value'])) + "\n")
    for tol in (0.05, 0.10):
        k = int((psnr_values >= best_psnr - tol).sum())
        f.write(f"   completed trials within {tol:.2f} dB of the best: {k}\n")
    f.write("   → if several trials are within the run-to-run noise (section 3), treat them as tied.\n")

print(f"   ✅ Statistical analysis saved to: {stats_report_path}")


# ===================================================================
# Section 9: Comprehensive text report
# ===================================================================
print("\n" + "=" * 75)
print("📝 Section 5: Comprehensive text report")
print("=" * 75)

report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("Optuna Study Analysis Report (generic, v2)\n")
    f.write("=" * 75 + "\n\n")

    f.write(f"Study name: {STUDY_NAME}\n")
    f.write(f"Storage path: {STORAGE_PATH}\n")
    f.write(f"Total trials recorded: {len(all_trials)}\n")
    f.write(f"Completed: {len(complete_trials)}\n")
    f.write(f"Pruned by the Pruner: {len(pruned_perf)} ({pruning_rate:.1f}% of trials that ran)\n")
    f.write(f"Stopped as invalid config: {len(pruned_invalid)}\n")
    f.write(f"Still RUNNING (stale): {len(running_trials)}\n")
    f.write(f"Failed: {len(failed_trials)}\n")
    f.write(f"Best PSNR: {study.best_value:.4f} dB\n")
    f.write(f"Best trial: #{study.best_trial.number}\n\n")

    f.write("=" * 75 + "\n")
    f.write("Best parameters:\n")
    f.write("=" * 75 + "\n")
    for key, value in study.best_trial.params.items():
        if isinstance(value, float):
            f.write(f"  {key:>25} : {value:.6e}\n")
        else:
            f.write(f"  {key:>25} : {value}\n")

    f.write("\n" + "=" * 75 + "\n")
    f.write("Spearman rank correlation of each numeric parameter with PSNR (sorted):\n")
    f.write("=" * 75 + "\n")
    corr_series = None
    if len(numeric_params) > 0:
        corr_series = df_complete[numeric_params + ['value']].corr(method='spearman')['value'].drop('value')
        corr_series = corr_series.sort_values(ascending=False)
        for param, corr in corr_series.items():
            clean = clean_label(param)
            if abs(corr) > 0.5:
                marker = "[STRONG+]" if corr > 0 else "[STRONG-]"
            elif abs(corr) > 0.3:
                marker = "[MOD+]" if corr > 0 else "[MOD-]"
            else:
                marker = "[WEAK]"
            f.write(f"  {marker} {clean:>25} : {corr:+.4f}\n")

    f.write("\n" + "=" * 75 + "\n")
    f.write("Notes for choosing parameters (within the searched ranges only):\n")
    f.write("=" * 75 + "\n")
    if corr_series is not None:
        for param, corr in corr_series.items():
            clean = clean_label(param)
            if abs(corr) > 0.3:
                direction = "higher" if corr > 0 else "lower"
                f.write(f"  • {direction} {clean} goes with better PSNR (rho = {corr:+.3f}) — "
                        f"association, not proof of causation.\n")
    for param, pos, lo, hi, val in boundary_flags:
        f.write(f"  ⚠️ {clean_label(param)}: the best value {val:.4g} is at the edge of its range "
                f"[{lo:g}, {hi:g}] — extend the range and re-run.\n")

print(f"✅ Report saved to: {report_path}")

# Save the DataFrames: ALL trials (every state, with pruned_kind) and the completed ones
csv_all = os.path.join(OUTPUT_DIR, "all_trials_full.csv")
df.to_csv(csv_all, index=False, encoding='utf-8-sig')
csv_completed = os.path.join(OUTPUT_DIR, "completed_trials.csv")
df_complete.to_csv(csv_completed, index=False, encoding='utf-8-sig')
print(f"✅ All trials (every state) saved to: {csv_all}")
print(f"✅ Completed trials saved to: {csv_completed}")


# ===================================================================
# Final summary
# ===================================================================
print("\n" + "=" * 75)
print("🎉 Comprehensive analysis complete!")
print("=" * 75)
print(f"📂 All files are in: {OUTPUT_DIR}")
print("\n📁 Generated files (each figure as .pdf / .png / .jpeg):")
print("   01_optimization_history.pdf/png/jpeg")
print("   02_param_importance.pdf/png/jpeg")
print("   03_slice_plot.pdf/png/jpeg")
print("   04_contour_plot.pdf/png/jpeg")
print("   05_parallel_coordinates.pdf/png/jpeg")
print("   06_custom_scatter_plots.pdf/png/jpeg")
print("   07_correlation_heatmap.pdf/png/jpeg")
print("   08_top10_trials.pdf/png/jpeg")
print("   12_pruned_vs_complete_distribution.pdf/png/jpeg   (only if the Pruner pruned trials)")
print("   13_pruning_rate_*.pdf/png/jpeg")
print("   analysis_report.txt")
print("   pruning_analysis.txt")
print("   statistical_tests.txt")
print("   all_trials_full.csv  (every state, with pruned_kind)")
print("   completed_trials.csv")
print("=" * 75)