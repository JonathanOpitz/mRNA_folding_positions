#!/usr/bin/env python3
"""
Analyze GNN prediction patterns:
  1. Why does the model never predict > 0.7?
  2. Where are the 0-value targets and predictions?
  3. Target distribution analysis
  4. Error analysis by folding_demand range
  5. Suggestions for improvement
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from scipy import stats

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/jonathanopitz/Desktop/Master")
DATA_DIR = BASE_DIR / "data/ribo_counts"
OUT_DIR  = BASE_DIR / "data/results"

# ─── Load all folding_demand data ──────────────────────────────────────────────

def load_all_fd():
    files = sorted(DATA_DIR.glob("*_with_folddemand.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        gene = f.stem.split("_ribosome")[0].split("_with")[0].upper()
        df['gene'] = gene
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    print("=" * 60)
    print("  Prediction & Target Distribution Analysis")
    print("=" * 60)

    all_data = load_all_fd()

    # Separate regions
    cds = all_data[all_data['region'] == 'CDS'].copy()
    utr5 = all_data[all_data['region'] == '5UTR'].copy()
    utr3 = all_data[all_data['region'] == '3UTR'].copy()

    fd_cds = cds['folding_demand'].dropna()

    # ════════════════════════════════════════════════════════════════
    print("\n1. TARGET DISTRIBUTION (folding_demand)")
    print("=" * 60)

    print(f"\n   Total codons:  {len(all_data)}")
    print(f"   CDS codons:    {len(cds)} ({len(cds)/len(all_data)*100:.0f}%)")
    print(f"   5'UTR codons:  {len(utr5)}")
    print(f"   3'UTR codons:  {len(utr3)}")

    print(f"\n   folding_demand (CDS only):")
    print(f"     min:    {fd_cds.min():.4f}")
    print(f"     25%:    {fd_cds.quantile(0.25):.4f}")
    print(f"     50%:    {fd_cds.median():.4f}")
    print(f"     75%:    {fd_cds.quantile(0.75):.4f}")
    print(f"     90%:    {fd_cds.quantile(0.90):.4f}")
    print(f"     95%:    {fd_cds.quantile(0.95):.4f}")
    print(f"     max:    {fd_cds.max():.4f}")
    print(f"     mean:   {fd_cds.mean():.4f}")
    print(f"     std:    {fd_cds.std():.4f}")

    # ════════════════════════════════════════════════════════════════
    print("\n2. WHY THE MODEL CAPS AT ~0.7")
    print("=" * 60)

    n_above_07 = (fd_cds > 0.7).sum()
    n_above_05 = (fd_cds > 0.5).sum()
    n_above_03 = (fd_cds > 0.3).sum()
    print(f"\n   Targets > 0.7: {n_above_07} / {len(fd_cds)} ({n_above_07/len(fd_cds)*100:.1f}%)")
    print(f"   Targets > 0.5: {n_above_05} / {len(fd_cds)} ({n_above_05/len(fd_cds)*100:.1f}%)")
    print(f"   Targets > 0.3: {n_above_03} / {len(fd_cds)} ({n_above_03/len(fd_cds)*100:.1f}%)")

    print(f"\n   → If very few targets are > 0.7, the model has no incentive")
    print(f"     to predict high values. MSE loss penalizes overshooting on")
    print(f"     the 95% of points that are < 0.5 more than it rewards")
    print(f"     correctly predicting the rare 5% that are > 0.7.")

    # ════════════════════════════════════════════════════════════════
    print("\n3. THE ZERO PROBLEM")
    print("=" * 60)

    n_zero = (fd_cds == 0).sum()
    n_near_zero = (fd_cds < 0.01).sum()
    print(f"\n   fd == 0:     {n_zero} / {len(fd_cds)} ({n_zero/len(fd_cds)*100:.1f}%)")
    print(f"   fd < 0.01:   {n_near_zero} / {len(fd_cds)} ({n_near_zero/len(fd_cds)*100:.1f}%)")

    # Why are they zero? Check the components
    zero_mask = fd_cds < 0.01
    zero_codons = cds[cds['folding_demand'] < 0.01]

    if 'fd_protein_need' in zero_codons.columns:
        mean_pn = zero_codons['fd_protein_need'].mean()
        mean_rc = zero_codons['fd_ribo_confirmation'].mean()
        print(f"\n   At zero-fd positions:")
        print(f"     mean protein_need:      {mean_pn:.4f}")
        print(f"     mean ribo_confirmation: {mean_rc:.4f}")

        # Which component is zeroing it out?
        low_pn = (zero_codons['fd_protein_need'] < 0.01).sum()
        low_rc = (zero_codons['fd_ribo_confirmation'] < 0.01).sum()
        both_low = ((zero_codons['fd_protein_need'] < 0.01) &
                    (zero_codons['fd_ribo_confirmation'] < 0.01)).sum()
        print(f"\n     protein_need < 0.01:      {low_pn} ({low_pn/len(zero_codons)*100:.0f}%)")
        print(f"     ribo_confirmation < 0.01: {low_rc} ({low_rc/len(zero_codons)*100:.0f}%)")
        print(f"     both < 0.01:              {both_low} ({both_low/len(zero_codons)*100:.0f}%)")

    # Check pLDDT at zero positions
    if 'plddt' in zero_codons.columns:
        mean_plddt_zero = zero_codons['plddt'].mean()
        mean_plddt_nonzero = cds[cds['folding_demand'] >= 0.01]['plddt'].mean()
        print(f"\n     pLDDT at zero-fd:    {mean_plddt_zero:.1f}")
        print(f"     pLDDT at nonzero-fd: {mean_plddt_nonzero:.1f}")
        print(f"     → Low pLDDT (disordered) → low plddt_weight → fd ≈ 0")

    # Check occupancy at zero positions
    if 'rel_occupancy' in zero_codons.columns:
        mean_occ_zero = zero_codons['rel_occupancy'].mean()
        mean_occ_nonzero = cds[cds['folding_demand'] >= 0.01]['rel_occupancy'].mean()
        print(f"\n     Occupancy at zero-fd:    {mean_occ_zero:.6f}")
        print(f"     Occupancy at nonzero-fd: {mean_occ_nonzero:.6f}")

    # ════════════════════════════════════════════════════════════════
    print("\n4. ERROR ANALYSIS BY RANGE")
    print("=" * 60)

    # Simulate what the model error looks like
    # If model predicts mean everywhere, what's the MSE per range?
    global_mean = fd_cds.mean()
    ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]

    print(f"\n   If model predicted mean ({global_mean:.3f}) everywhere:")
    for lo, hi in ranges:
        mask = (fd_cds >= lo) & (fd_cds < hi)
        n = mask.sum()
        if n > 0:
            vals = fd_cds[mask].values
            naive_mse = np.mean((vals - global_mean) ** 2)
            print(f"     [{lo:.1f}, {hi:.1f}): n={n:>6} ({n/len(fd_cds)*100:>5.1f}%)  "
                  f"naive MSE={naive_mse:.5f}")

    # ════════════════════════════════════════════════════════════════
    print("\n5. RECOMMENDATIONS")
    print("=" * 60)

    pct_below_01 = n_near_zero / len(fd_cds) * 100
    pct_above_05 = n_above_05 / len(fd_cds) * 100

    print(f"""
   The target distribution is heavily right-skewed:
     {pct_below_01:.0f}% of values are < 0.01 (near zero)
     {pct_above_05:.0f}% of values are > 0.5 (high demand)

   This explains both problems:
   a) Model caps at ~0.7 because too few training examples are > 0.7
   b) Biggest errors at 0 because the model can't distinguish
      "zero because disordered" from "zero because low occupancy"

   SOLUTIONS TO TRY:
   1. WEIGHTED LOSS: weight high-fd samples more
      weight_i = 1 + α * fd_i  (e.g. α=3)
      This tells the model: getting high-fd positions right matters more.

   2. LOG TRANSFORM TARGET: fd_transformed = log(fd + epsilon)
      Spreads out the near-zero values, compresses the high end.
      BUT: changes interpretation of the score.

   3. QUANTILE TRANSFORM: rank-transform fd to uniform [0,1]
      Every range gets equal representation in the loss.
      Downside: loses the absolute scale.

   4. FOCAL LOSS VARIANT: reduce loss for easy (near-zero) samples
      loss_i = (1 - |pred_i - true_i|)^γ * MSE_i
      Model focuses on hard examples (medium-to-high fd).

   5. TWO-STAGE: First classify (zero vs nonzero), then regress.
      Stage 1: Binary — is folding_demand > 0.05? (easy to learn)
      Stage 2: Regression — how high? (only on nonzero positions)

   RECOMMENDED: Start with weighted loss (simplest, no target change).
""")

    # ════════════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════════════

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)

    # A: Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(fd_cds.values, bins=100, color='steelblue', edgecolor='white',
             linewidth=0.3, alpha=0.8)
    ax1.axvline(fd_cds.median(), color='red', linewidth=1, label=f'median={fd_cds.median():.3f}')
    ax1.set_xlabel('folding_demand')
    ax1.set_ylabel('Count')
    ax1.set_title('A. Target distribution (CDS)')
    ax1.legend(fontsize=8)

    # B: Log-scale target distribution
    ax2 = fig.add_subplot(gs[0, 1])
    fd_nonzero = fd_cds[fd_cds > 0]
    ax2.hist(np.log10(fd_nonzero.values + 1e-8), bins=100,
             color='steelblue', edgecolor='white', linewidth=0.3, alpha=0.8)
    ax2.set_xlabel('log10(folding_demand)')
    ax2.set_ylabel('Count')
    ax2.set_title('B. Log-scale distribution')

    # C: Cumulative distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_fd = np.sort(fd_cds.values)
    cumulative = np.arange(1, len(sorted_fd) + 1) / len(sorted_fd)
    ax3.plot(sorted_fd, cumulative, color='steelblue', linewidth=1.5)
    ax3.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
    ax3.axhline(0.9, color='gray', linewidth=0.5, linestyle='--')
    ax3.axhline(0.95, color='gray', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('folding_demand')
    ax3.set_ylabel('Cumulative fraction')
    ax3.set_title('C. CDF — where is the mass?')

    # D: protein_need distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if 'fd_protein_need' in cds.columns:
        pn = cds['fd_protein_need'].dropna()
        ax4.hist(pn.values, bins=100, color='#D85A30', edgecolor='white',
                 linewidth=0.3, alpha=0.8)
        ax4.set_xlabel('protein_need')
        ax4.set_title(f'D. protein_need (mean={pn.mean():.3f})')

    # E: ribo_confirmation distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if 'fd_ribo_confirmation' in cds.columns:
        rc = cds['fd_ribo_confirmation'].dropna()
        ax5.hist(rc.values, bins=100, color='#5DCAA5', edgecolor='white',
                 linewidth=0.3, alpha=0.8)
        ax5.set_xlabel('ribo_confirmation')
        ax5.set_title(f'E. ribo_confirmation (mean={rc.mean():.3f})')

    # F: protein_need vs ribo_confirmation scatter
    ax6 = fig.add_subplot(gs[1, 2])
    if 'fd_protein_need' in cds.columns and 'fd_ribo_confirmation' in cds.columns:
        pn = cds['fd_protein_need'].fillna(0).values
        rc = cds['fd_ribo_confirmation'].fillna(0).values
        ax6.scatter(pn, rc, alpha=0.02, s=2, c='steelblue')
        ax6.set_xlabel('protein_need')
        ax6.set_ylabel('ribo_confirmation')
        ax6.set_title('F. protein_need × ribo_confirmation')

    # G: folding_demand by pLDDT range
    ax7 = fig.add_subplot(gs[2, 0])
    plddt_bins = [(0, 50), (50, 70), (70, 90), (90, 100)]
    means = []
    labels = []
    for lo, hi in plddt_bins:
        mask = (cds['plddt'] >= lo) & (cds['plddt'] < hi)
        if mask.sum() > 0:
            means.append(cds.loc[mask, 'folding_demand'].mean())
            labels.append(f'{lo}-{hi}')
    ax7.bar(range(len(means)), means, color='steelblue', edgecolor='white')
    ax7.set_xticks(range(len(labels)))
    ax7.set_xticklabels(labels)
    ax7.set_xlabel('pLDDT range')
    ax7.set_ylabel('mean folding_demand')
    ax7.set_title('G. fd by pLDDT confidence')

    # H: folding_demand by region
    ax8 = fig.add_subplot(gs[2, 1])
    regions = ['5UTR', 'CDS', '3UTR']
    region_fds = []
    for r in regions:
        vals = all_data[all_data['region'] == r]['folding_demand'].dropna()
        region_fds.append(vals.values if len(vals) > 0 else np.array([0]))
    bp = ax8.boxplot(region_fds, labels=regions, patch_artist=True,
                     showfliers=False)
    for patch, color in zip(bp['boxes'], ['#85B7EB', '#5DCAA5', '#EF9F27']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax8.set_ylabel('folding_demand')
    ax8.set_title('H. fd by transcript region')

    # I: Per-gene mean fd distribution
    ax9 = fig.add_subplot(gs[2, 2])
    gene_means = cds.groupby('gene')['folding_demand'].mean().dropna()
    ax9.hist(gene_means.values, bins=30, color='steelblue', edgecolor='white',
             linewidth=0.5, alpha=0.8)
    ax9.set_xlabel('Mean folding_demand per gene')
    ax9.set_ylabel('Number of genes')
    ax9.set_title('I. Gene-level fd variation')

    plt.savefig(OUT_DIR / 'fd_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {OUT_DIR / 'fd_distribution_analysis.png'}")
    plt.close()


if __name__ == '__main__':
    main()