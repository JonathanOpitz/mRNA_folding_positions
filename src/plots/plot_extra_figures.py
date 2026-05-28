#!/usr/bin/env python3
"""
make_extra_plots.py

Generates two additional figures for the Results section.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# adjustText fuer kollisionsfreie Gen-Labels (pip install adjustText)
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("WARNING: adjustText nicht installiert -> Fallback auf festen Offset.")
    print("         Installiere mit: pip install adjustText")

# ─── Font Settings ────────────────────────────────────────────────────────────
FONT_BASE     = 16   
FONT_LABEL    = 16   
FONT_TITLE    = 17   
FONT_SUPTITLE = 19   
FONT_ANNOT    = 11   

mpl.rcParams.update({
    'font.size':        FONT_BASE,
    'axes.titlesize':   FONT_TITLE,
    'axes.labelsize':   FONT_LABEL,
    'xtick.labelsize':  FONT_BASE,
    'ytick.labelsize':  FONT_BASE,
    'legend.fontsize':  FONT_BASE,
    'figure.titlesize': FONT_SUPTITLE,
})

# ─── Configuration ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]

EVAL_05 = BASE / "data/analysis/sweep_evaluation/per_gene_metrics.csv"
EVAL_03 = BASE / "data/analysis/sweep_evaluation_g03/per_gene_metrics.csv"

OUT_DIR = BASE / "data/analysis/sweep_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLORS = {
    'housekeeping': '#1f77b4',
    'multi_domain': '#d62728',
    'therapeutic':  '#2ca02c',
}

# ─── Load data ────────────────────────────────────────────────────────────────
print(f"Loading γ=0.5 data from: {EVAL_05}")
df_05 = pd.read_csv(EVAL_05)
df_05['gamma'] = 0.5
print(f"  Loaded {len(df_05)} genes")

print(f"\nLoading γ=0.3 data from: {EVAL_03}")
if EVAL_03.exists():
    df_03 = pd.read_csv(EVAL_03)
    df_03['gamma'] = 0.3
    print(f"  Loaded {len(df_03)} genes")
else:
    print(f"  WARNING: γ=0.3 CSV not found")
    df_03 = None


def delta_rho(df, low_col, high_col):
    return df[high_col].values - df[low_col].values


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Dose-response
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding dose_response.png ...")

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)

metrics = [
    ("rho_low_hydro_z",    "rho_high_hydro_z",    r"$\rho(\tau_z, \mathrm{KD})$"),
    ("rho_low_contacts_z", "rho_high_contacts_z", r"$\rho(\tau_z, \mathrm{CD})$"),
    ("rho_low_fd_z",       "rho_high_fd_z",       r"$\rho(\tau_z, \mathrm{fd})$"),
]

gammas = [0.0, 0.3, 0.5]

for ax, (lo, hi, label) in zip(axes, metrics):
    medians = [0.0]
    q1 = [0.0]
    q3 = [0.0]

    if df_03 is not None:
        d03 = delta_rho(df_03, lo, hi)
        d03 = d03[~np.isnan(d03)]
        medians.append(np.median(d03))
        q1.append(np.percentile(d03, 25))
        q3.append(np.percentile(d03, 75))
    else:
        medians.append(np.nan)
        q1.append(np.nan)
        q3.append(np.nan)

    d05 = delta_rho(df_05, lo, hi)
    d05 = d05[~np.isnan(d05)]
    medians.append(np.median(d05))
    q1.append(np.percentile(d05, 25))
    q3.append(np.percentile(d05, 75))

    medians = np.array(medians)
    q1 = np.array(q1)
    q3 = np.array(q3)

    ax.errorbar(gammas, medians, yerr=[medians - q1, q3 - medians],
                fmt='o-', color='#1f77b4', lw=2, ms=8, capsize=4, capthick=1.5)

    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.set_xlabel(r"$\gamma$ (fold penalty weight)")
    ax.set_title(label)
    ax.set_xticks(gammas)
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r"$\Delta\rho = \rho_\gamma - \rho_0$")
fig.suptitle("Dose-response of positional correlation effects\n"
             "(median across genes, error bars = IQR)")

plt.tight_layout()
plt.savefig(OUT_DIR / "dose_response.png", dpi=160, bbox_inches='tight')
plt.close()
print(f"[Saved] {OUT_DIR / 'dose_response.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Effect vs Density – VERBESSERT
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding effect_vs_density.png ...")

# Axis margin fraction — larger values add padding so points/labels don't touch the border
X_MARGIN = 0.15
Y_MARGIN = 0.15

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=False)


def plot_panel(ax, df, gamma_label):
    d = df['rho_high_fd_z'] - df['rho_low_fd_z']
    xs = df['mean_contact_density'].values
    ys = d.values
    mask = ~np.isnan(xs) & ~np.isnan(ys)

    # Draw points
    for cat, color in CAT_COLORS.items():
        sub_mask = mask & (df['category'] == cat)
        ax.scatter(xs[sub_mask], ys[sub_mask],
                   c=color, s=120, alpha=0.85,
                   edgecolors='black', linewidths=0.7)

    # Set axis limits before adjustText so labels stay within bounds
    x_min, x_max = xs[mask].min(), xs[mask].max()
    y_min, y_max = ys[mask].min(), ys[mask].max()
    xr = (x_max - x_min) or 1.0
    yr = (y_max - y_min) or 1.0
    ax.set_xlim(x_min - X_MARGIN * xr, x_max + X_MARGIN * xr)
    ax.set_ylim(y_min - Y_MARGIN * yr, y_max + Y_MARGIN * yr)

    # Gen-Labels
    genes = df['gene'].values
    if HAS_ADJUSTTEXT:
        texts = [
            ax.text(xs[i], ys[i], genes[i], fontsize=FONT_ANNOT, alpha=0.9)
            for i in range(len(genes)) if mask[i]
        ]
        adjust_text(
            texts,
            x=xs[mask], y=ys[mask],   # repel away from these points
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            expand_points=(1.4, 1.4),
            expand_text=(1.2, 1.4),
            force_text=(0.4, 0.6),
            only_move={'points': 'xy', 'text': 'xy'},
        )
    else:
        # Fallback ohne adjustText: fester Offset (kann ueberlappen)
        for i in range(len(genes)):
            if mask[i]:
                ax.annotate(genes[i], (xs[i], ys[i]),
                            fontsize=FONT_ANNOT,
                            xytext=(6, 6), textcoords='offset points',
                            alpha=0.85)

    # Statistik
    if mask.sum() >= 3:
        rho, p = spearmanr(xs[mask], ys[mask])
        title = rf"$\gamma = {gamma_label}$" + f"\nSpearman ρ = {rho:+.2f} (p = {p:.3f})"
    else:
        title = rf"$\gamma = {gamma_label}$"

    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.set_xlabel("Mean contact density", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$\Delta\rho(\tau_z, \mathrm{fd})$", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, pad=12)
    ax.grid(alpha=0.3)


# Panels zeichnen
if df_03 is not None:
    plot_panel(axes[0], df_03, "0.3")
else:
    axes[0].text(0.5, 0.5, "γ=0.3 data not available", ha='center', va='center',
                 transform=axes[0].transAxes, fontsize=12)

plot_panel(axes[1], df_05, "0.5")

# Manual legend — always show all three categories, horizontal
legend_handles = [
    Line2D([0], [0], marker='o', linestyle='none', label=cat,
           markerfacecolor=color, markeredgecolor='black', markersize=11)
    for cat, color in CAT_COLORS.items()
]
fig.legend(legend_handles, list(CAT_COLORS.keys()),
           loc='lower center',
           bbox_to_anchor=(0.5, -0.01),   # naeher am Plot
           ncol=3,                         # horizontal
           fontsize=FONT_BASE - 1,
           frameon=True,
           edgecolor='black')

fig.suptitle("Effect size vs. contact density per gene",
             fontsize=FONT_SUPTITLE, y=0.98)

# Reserve space: bottom for legend, top for suptitle
fig.tight_layout(rect=[0, 0.07, 1, 0.94])

out_path = OUT_DIR / "effect_vs_density.png"
plt.savefig(out_path, dpi=160, bbox_inches='tight')
plt.close()

print(f"[Saved] {out_path}")
print("\nDone.")