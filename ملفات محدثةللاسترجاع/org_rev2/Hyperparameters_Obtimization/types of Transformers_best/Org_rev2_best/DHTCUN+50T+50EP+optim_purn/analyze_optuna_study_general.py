# ===================================================================
# analyze_optuna_study_general.py — General-purpose Optuna study analysis
#
# Features:
#   1. 100% generic — works with any model without modification
#   2. Separate analysis of pruned trials
#   3. Statistical tests (Wilcoxon, Mann-Whitney U, Bootstrap CI)
#   4. All figures in English, publication-ready (no overlapping labels)
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


def detect_parameters(df, exclude_cols=None):
    """
    🔵 [Generic] Automatically detects all parameters from the DataFrame
    and returns two lists: numeric + categorical
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

        dtype = df[col].dtype
        if dtype in ['int64', 'int32', 'float64', 'float32']:
            if df[col].nunique() <= 10:
                categorical_params.append(col)
            else:
                numeric_params.append(col)
        else:
            categorical_params.append(col)

    return numeric_params, categorical_params


def confidence_interval(data, confidence=0.95, n_bootstrap=10000):
    """
    🔵 [Generic] Compute the confidence interval using Bootstrap
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

print(f"\n📚 Study name: {STUDY_NAME}")
print(f"📊 Total trials: {len(all_trials)}")
print(f"✅ Completed: {len(complete_trials)}")
print(f"✂️ Pruned:    {len(pruned_trials)}")
print(f"❌ Failed:    {len(failed_trials)}")

if len(complete_trials) < 2:
    print("⚠️ Too few completed trials for analysis.")
    sys.exit(1)

print(f"\n🏆 Best PSNR: {study.best_value:.4f} dB (Trial #{study.best_trial.number})")

# Pruning rate
pruning_rate = len(pruned_trials) / max(len(all_trials), 1) * 100
print(f"📉 Pruning rate: {pruning_rate:.1f}%")
if pruning_rate > 30:
    print("⚠️ Warning: high pruning rate (> 30%). May affect the analysis.")
elif pruning_rate > 15:
    print("ℹ️ Note: moderate pruning rate (15-30%). Analysis is acceptable.")
else:
    print("✅ Low pruning rate (< 15%). Analysis is reliable.")

# Convert to DataFrame
df = study.trials_dataframe()
df_complete = df[df['state'] == 'COMPLETE'].copy()
df_pruned = df[df['state'] == 'PRUNED'].copy() if len(pruned_trials) > 0 else pd.DataFrame()

# 🔵 Auto-detect parameters
numeric_params, categorical_params = detect_parameters(df_complete)
print(f"\n🔍 Detected {len(numeric_params)} numeric parameter(s) and {len(categorical_params)} categorical parameter(s)")
print(f"   Numeric: {[clean_label(p) for p in numeric_params]}")
print(f"   Categorical: {[clean_label(p) for p in categorical_params]}")


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

ax.plot(trial_numbers, values, 'o-', color='steelblue',
        markersize=6, label='PSNR per trial', alpha=0.7)
ax.plot(trial_numbers, best_so_far, 'r-', linewidth=2,
        label=f'Best cumulative PSNR ({best_so_far[-1]:.3f} dB)')
ax.axhline(y=study.best_value, color='green', linestyle='--',
           alpha=0.5, label=f'Best value = {study.best_value:.3f} dB')
ax.set_xlabel('Trial number')
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
    🔵 [Generic] Return the top-N most important parameter names.
    Optuna's matplotlib grid renderers (plot_slice / plot_contour) start
    overlapping their shared colorbar with the last column once more than
    ~4 parameters are shown — this is a limitation of the Optuna renderer
    itself. Capping to the most important parameters keeps every subplot
    readable and avoids that overlap while still showing the relationships
    that matter most.
    """
    try:
        importances = optuna.importance.get_param_importances(study)
        top = list(importances.keys())[:max_params]
        return top if len(top) >= 2 else None
    except Exception:
        return None


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

    # Trend line (numeric parameters only)
    if param in numeric_params:
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.5,
                    linewidth=1.5, label='Trend line')
        except Exception:
            pass

    clean_name = clean_label(param)
    ax.set_xlabel(clean_name, fontsize=10, fontweight='bold', labelpad=6)
    ax.set_ylabel('PSNR (dB)', fontsize=10, labelpad=6)
    ax.set_title(f'{clean_name} vs PSNR', fontsize=11, fontweight='bold', pad=10)
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

    clean_labels = [clean_label(c) if c != 'value' else 'PSNR' for c in corr_matrix.columns]
    ax.set_xticks(range(len(clean_labels)))
    ax.set_yticks(range(len(clean_labels)))
    ax.set_xticklabels(clean_labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation coefficient', fontsize=10)
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
# 🆕 Section 7: Pruned trials analysis
# ===================================================================
print("\n" + "=" * 75)
print("✂️ Section 3: Pruned trials analysis")
print("=" * 75)

if len(df_pruned) > 0:
    print(f"   Number of pruned trials: {len(df_pruned)}")

    # 7.1 Compare parameter distributions between completed and pruned trials
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.2 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, param in enumerate(all_plottable):
        ax = axes[idx]
        if param in df_complete.columns and param in df_pruned.columns:
            complete_vals = df_complete[param].dropna().values
            pruned_vals = df_pruned[param].dropna().values

            if param in numeric_params:
                bins = np.linspace(
                    min(complete_vals.min(), pruned_vals.min() if len(pruned_vals) > 0 else complete_vals.min()),
                    max(complete_vals.max(), pruned_vals.max() if len(pruned_vals) > 0 else complete_vals.max()),
                    15
                )
                ax.hist(complete_vals, bins=bins, alpha=0.6,
                        color='green', label=f'Completed ({len(complete_vals)})')
                if len(pruned_vals) > 0:
                    ax.hist(pruned_vals, bins=bins, alpha=0.6,
                            color='red', label=f'Pruned ({len(pruned_vals)})')
            else:
                all_vals = sorted(set(list(complete_vals) + list(pruned_vals)))
                complete_counts = [np.sum(complete_vals == v) for v in all_vals]
                pruned_counts = [np.sum(pruned_vals == v) for v in all_vals]
                x_pos = np.arange(len(all_vals))
                width = 0.35
                ax.bar(x_pos - width/2, complete_counts, width,
                       alpha=0.6, color='green', label='Completed')
                ax.bar(x_pos + width/2, pruned_counts, width,
                       alpha=0.6, color='red', label='Pruned')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(v) for v in all_vals], rotation=30, ha='right')

            clean_name = clean_label(param)
            ax.set_xlabel(clean_name, fontsize=10, fontweight='bold', labelpad=6)
            ax.set_ylabel('Count', fontsize=10, labelpad=6)
            ax.set_title(f'{clean_name}: Completed vs Pruned',
                        fontsize=10, fontweight='bold', pad=10)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=8)

    for idx in range(len(all_plottable), len(axes)):
        axes[idx].axis('off')

    fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    save_figure(fig, "12_pruned_vs_complete_distribution")

    # 7.2 Table: which parameter values get pruned more often?
    print("   → Analyzing pruning regions...")

    N_PRUNING_BINS = 8  # number of bins used for continuous numeric parameters

    pruning_analysis = {}
    df_relevant = df[df['state'].isin(['COMPLETE', 'PRUNED'])]

    for param in all_plottable:
        if param not in df.columns:
            continue

        if param in numeric_params:
            # Continuous parameter: group by value RANGE, not exact float
            # value. Grouping by exact float would put almost every trial
            # in its own singleton bucket (n=1), which is meaningless and
            # produces an unreadable chart.
            try:
                binned = pd.cut(df_relevant[param], bins=N_PRUNING_BINS)
                all_counts = df_relevant.groupby(binned, observed=True).size()
                if len(df_pruned) > 0:
                    pruned_binned = pd.cut(df_pruned[param], bins=binned.cat.categories)
                    pruned_counts = df_pruned.groupby(pruned_binned, observed=True).size()
                else:
                    pruned_counts = pd.Series(dtype=float)
                label_fn = lambda interval: f"[{interval.left:.3g}, {interval.right:.3g}]"
            except Exception:
                # Fallback: exact grouping if binning fails for any reason
                all_counts = df_relevant.groupby(param).size()
                pruned_counts = df_pruned.groupby(param).size() if len(df_pruned) > 0 else pd.Series()
                label_fn = lambda v: str(v)
        else:
            # Categorical / low-cardinality parameter: exact grouping is fine
            all_counts = df_relevant.groupby(param).size()
            pruned_counts = df_pruned.groupby(param).size() if len(df_pruned) > 0 else pd.Series()
            label_fn = lambda v: str(v)

        param_analysis = []
        for val in all_counts.index:
            total = all_counts[val]
            if total == 0:
                continue
            pruned = pruned_counts.get(val, 0)
            rate = pruned / total * 100 if total > 0 else 0
            param_analysis.append((label_fn(val), total, pruned, rate))

        pruning_analysis[param] = param_analysis

    pruning_report_path = os.path.join(OUTPUT_DIR, "pruning_analysis.txt")
    with open(pruning_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 75 + "\n")
        f.write("Pruned Trials Analysis — Which parameter values lead to pruning?\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Total trials: {len(all_trials)}\n")
        f.write(f"Completed: {len(complete_trials)}\n")
        f.write(f"Pruned: {len(pruned_trials)}\n")
        f.write(f"Pruning rate: {pruning_rate:.1f}%\n\n")

        for param, analysis in pruning_analysis.items():
            clean = clean_label(param)
            f.write(f"\n{clean}:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Value':>15} | {'Total':>8} | {'Pruned':>8} | {'Rate %':>8}\n")
            f.write("-" * 50 + "\n")

            analysis_sorted = sorted(analysis, key=lambda x: x[3], reverse=True)
            for val, total, pruned, rate in analysis_sorted:
                marker = "HIGH" if rate > 50 else "MED" if rate > 25 else "LOW"
                f.write(f"{str(val):>15} | {total:>8} | {pruned:>8} | {rate:>7.1f}% [{marker}]\n")

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
                     color=['red' if r > 50 else 'orange' if r > 25 else 'green'
                            for r in rates],
                     alpha=0.7, edgecolor='black')

        for i, (bar, total) in enumerate(zip(bars, totals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'n={total}', ha='center', va='bottom', fontsize=8)

        clean_name = clean_label(param)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=40, ha='right', fontsize=8)
        ax.set_xlabel(clean_name, fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('Pruning rate (%)', fontsize=11, labelpad=8)
        ax.set_title(f'Pruning rate per value of {clean_name}',
                    fontsize=12, fontweight='bold', pad=14)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='25%')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        save_figure(fig, f"13_pruning_rate_{clean_name.replace(' ', '_')}")

else:
    print("   ℹ️ No pruned trials found. Skipping this analysis.")


# ===================================================================
# 🆕 Section 8: Statistical tests
# ===================================================================
print("\n" + "=" * 75)
print("📊 Section 4: Statistical tests")
print("=" * 75)

stats_report_path = os.path.join(OUTPUT_DIR, "statistical_tests.txt")
with open(stats_report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("Statistical Tests Report\n")
    f.write("=" * 75 + "\n\n")

    # 8.1 Descriptive statistics
    print("   → Descriptive statistics...")
    psnr_values = df_complete['value'].values
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values, ddof=1)
    median_psnr = np.median(psnr_values)
    min_psnr = np.min(psnr_values)
    max_psnr = np.max(psnr_values)
    q25, q75 = np.percentile(psnr_values, [25, 75])

    ci_lower, ci_upper = confidence_interval(psnr_values)

    f.write("1) Descriptive statistics for PSNR (completed trials)\n")
    f.write("-" * 50 + "\n")
    f.write(f"   Number of trials: {len(psnr_values)}\n")
    f.write(f"   Mean: {mean_psnr:.4f} dB\n")
    f.write(f"   Standard deviation: {std_psnr:.4f} dB\n")
    f.write(f"   Median: {median_psnr:.4f} dB\n")
    f.write(f"   Minimum: {min_psnr:.4f} dB\n")
    f.write(f"   Maximum: {max_psnr:.4f} dB\n")
    f.write(f"   1st quartile (Q1): {q25:.4f} dB\n")
    f.write(f"   3rd quartile (Q3): {q75:.4f} dB\n")
    f.write(f"   IQR: {q75 - q25:.4f} dB\n")
    f.write(f"   95% confidence interval (Bootstrap): [{ci_lower:.4f}, {ci_upper:.4f}]\n")
    f.write(f"   Standard error (SE): {std_psnr / np.sqrt(len(psnr_values)):.4f} dB\n\n")

    print(f"      Mean: {mean_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"      95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 8.2 Wilcoxon/Mann-Whitney test: best 25% vs worst 25%
    print("   → Mann-Whitney U test (best 25% vs worst 25%)...")
    n_top = max(3, len(psnr_values) // 4)
    sorted_psnr = np.sort(psnr_values)
    worst_group = sorted_psnr[:n_top]
    best_group = sorted_psnr[-n_top:]

    try:
        statistic, p_value = stats.mannwhitneyu(
            best_group, worst_group, alternative='greater'
        )

        f.write("2) Mann-Whitney U test (best 25% vs worst 25%)\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Best-group size: {n_top}\n")
        f.write(f"   Worst-group size: {n_top}\n")
        f.write(f"   Best-group mean: {np.mean(best_group):.4f} dB\n")
        f.write(f"   Worst-group mean: {np.mean(worst_group):.4f} dB\n")
        f.write(f"   Difference: {np.mean(best_group) - np.mean(worst_group):.4f} dB\n")
        f.write(f"   U statistic: {statistic:.4f}\n")
        f.write(f"   p-value: {p_value:.6e}\n")

        if p_value < 0.001:
            f.write("   ✅ Highly significant difference (p < 0.001) — results are publishable\n")
        elif p_value < 0.05:
            f.write("   ✅ Significant difference (p < 0.05) — results are acceptable\n")
        else:
            f.write("   ⚠️ No significant difference (p >= 0.05) — more trials may be needed\n")
        f.write("\n")

        print(f"      p-value = {p_value:.6e}")
    except Exception as e:
        f.write(f"   ⚠️ Could not run the test: {e}\n\n")

    # 8.3 Normality test (Shapiro-Wilk)
    print("   → Shapiro-Wilk test (is the distribution normal?)...")
    if len(psnr_values) <= 5000 and len(psnr_values) >= 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(psnr_values[:5000])
            f.write("3) Shapiro-Wilk test (is the distribution normal?)\n")
            f.write("-" * 50 + "\n")
            f.write(f"   W statistic: {shapiro_stat:.4f}\n")
            f.write(f"   p-value: {shapiro_p:.6e}\n")
            if shapiro_p > 0.05:
                f.write("   ✅ The distribution is normal (p > 0.05)\n")
                f.write("   → Parametric tests (e.g., t-test) can be used\n\n")
            else:
                f.write("   ⚠️ The distribution is not normal (p < 0.05)\n")
                f.write("   → Non-parametric tests (e.g., Mann-Whitney) are preferred\n\n")
        except Exception as e:
            f.write(f"   ⚠️ Failed: {e}\n\n")

    # 8.4 Compare completed vs pruned trials (if any)
    if len(df_pruned) > 0:
        f.write("4) Comparison of completed vs pruned trials\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Number completed: {len(complete_trials)}\n")
        f.write(f"   Number pruned: {len(pruned_trials)}\n")
        f.write(f"   Pruning rate: {pruning_rate:.1f}%\n\n")

        f.write("   Parameters most associated with pruning:\n")
        for param in numeric_params:
            if param not in df.columns:
                continue
            complete_vals = df_complete[param].dropna().values
            pruned_vals = df_pruned[param].dropna().values
            if len(pruned_vals) > 0 and len(complete_vals) > 0:
                try:
                    stat, p = stats.mannwhitneyu(complete_vals, pruned_vals)
                    if p < 0.05:
                        clean = clean_label(param)
                        f.write(f"      [SIGNIFICANT] {clean}: p = {p:.4f}\n")
                except Exception:
                    pass
        f.write("\n")

    # 8.5 Reliability of the best trial
    f.write("5) Reliability of the best trial\n")
    f.write("-" * 50 + "\n")
    best_psnr = study.best_value
    f.write(f"   Best PSNR: {best_psnr:.4f} dB\n")
    f.write(f"   Difference from mean: +{best_psnr - mean_psnr:.4f} dB\n")
    f.write(f"   Standard deviations above mean: {(best_psnr - mean_psnr) / std_psnr:.2f} sigma\n")

    if best_psnr > mean_psnr + 2 * std_psnr:
        f.write("   ✅ Best trial is more than 2 sigma above the mean — outstanding result\n")
    elif best_psnr > mean_psnr + std_psnr:
        f.write("   ✅ Best trial is more than 1 sigma above the mean — good result\n")
    else:
        f.write("   ⚠️ Best trial is close to the mean — more trials may be needed\n")

print(f"   ✅ Statistical tests saved to: {stats_report_path}")


# ===================================================================
# Section 9: Comprehensive text report
# ===================================================================
print("\n" + "=" * 75)
print("📝 Section 5: Comprehensive text report")
print("=" * 75)

report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("Optuna Study Analysis Report (generic)\n")
    f.write("=" * 75 + "\n\n")

    f.write(f"Study name: {STUDY_NAME}\n")
    f.write(f"Storage path: {STORAGE_PATH}\n")
    f.write(f"Total trials: {len(all_trials)}\n")
    f.write(f"Completed: {len(complete_trials)}\n")
    f.write(f"Pruned: {len(pruned_trials)} ({pruning_rate:.1f}%)\n")
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
    f.write("Correlation of each parameter with PSNR (sorted):\n")
    f.write("=" * 75 + "\n")
    if len(numeric_params) > 0:
        corr_data = df_complete[numeric_params + ['value']].corr()['value'].drop('value')
        corr_data = corr_data.sort_values(ascending=False)
        for param, corr in corr_data.items():
            clean = clean_label(param)
            if abs(corr) > 0.5:
                marker = "[STRONG+]" if corr > 0 else "[STRONG-]"
            elif abs(corr) > 0.3:
                marker = "[MOD+]" if corr > 0 else "[MOD-]"
            else:
                marker = "[WEAK]"
            f.write(f"  {marker} {clean:>25} : {corr:+.4f}\n")

    f.write("\n" + "=" * 75 + "\n")
    f.write("Recommendations for choosing the best parameters:\n")
    f.write("=" * 75 + "\n")
    if len(numeric_params) > 0:
        for param, corr in corr_data.items():
            clean = clean_label(param)
            if abs(corr) > 0.3:
                if corr > 0:
                    f.write(f"  ✅ Increasing {clean} improves PSNR (r = {corr:+.3f})\n")
                else:
                    f.write(f"  ✅ Decreasing {clean} improves PSNR (r = {corr:+.3f})\n")

print(f"✅ Report saved to: {report_path}")

# Save the full DataFrame
csv_path = os.path.join(OUTPUT_DIR, "all_trials_full.csv")
df_complete.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ All trials saved to: {csv_path}")


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
print("   12_pruned_vs_complete_distribution.pdf/png/jpeg")
print("   13_pruning_rate_*.pdf/png/jpeg")
print("   analysis_report.txt")
print("   pruning_analysis.txt")
print("   statistical_tests.txt")
print("   all_trials_full.csv")
print("=" * 75)