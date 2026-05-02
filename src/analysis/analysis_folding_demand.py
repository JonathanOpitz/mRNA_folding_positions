#!/usr/bin/env python3
"""
Validate the folding_demand score against actual ribosome occupancy data.

Key questions:
  1. Does folding_demand correlate with rel_occupancy? (Spearman + Pearson)
  2. Do high-folding-demand regions have higher ribosome occupancy?
  3. Which sub-components contribute most to the correlation?
  4. Per-gene breakdown: does it work consistently or only for some genes?
  5. Visual: overlay folding_demand and occupancy for example genes

Output:
  - Console statistics
  - data/analysis/folding_demand_validation.png (multi-panel figure)
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/jonathanopitz/Desktop/Master")
RESULTS_DIR = BASE_DIR / "data/ribo_counts"
OUT_DIR     = BASE_DIR / "data/analysis"

# ─── Load all data ─────────────────────────────────────────────────────────────

def load_all_genes() -> pd.DataFrame:
    """Load all *_with_folddemand.csv files into one DataFrame."""
    files = sorted(RESULTS_DIR.glob("*_with_folddemand.csv"))
    if not files:
        # Fallback: try rnaplfold files if folddemand not yet computed
        files = sorted(RESULTS_DIR.glob("*_with_rnaplfold.csv"))
        if not files:
            print("No processed CSV files found!")
            sys.exit(1)

    dfs = []
    for f in files:
        gene = re.split(r'_ribosome|_with_|_counts', f.stem)[0].upper()
        df = pd.read_csv(f)
        df['gene'] = gene
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} genes, {len(combined)} total codons")
    return combined


# ─── Analysis functions ────────────────────────────────────────────────────────

def global_correlation(cds: pd.DataFrame):
    """Overall correlation between folding_demand and ribosome occupancy."""
    print("\n" + "=" * 60)
    print("1. GLOBAL CORRELATION: folding_demand vs rel_occupancy")
    print("=" * 60)

    fd = cds['folding_demand'].values
    occ = cds['rel_occupancy'].values

    r_pearson, p_pearson = stats.pearsonr(fd, occ)
    r_spearman, p_spearman = stats.spearmanr(fd, occ)

    print(f"   Pearson:  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")
    print(f"   Spearman: r = {r_spearman:.4f}  (p = {p_spearman:.2e})")

    # Log-transformed occupancy (might be better for skewed data)
    log_occ = np.log(occ + 1e-6)
    r_sp_log, p_sp_log = stats.spearmanr(fd, log_occ)
    print(f"   Spearman (log occ): r = {r_sp_log:.4f}  (p = {p_sp_log:.2e})")

    return r_spearman, p_spearman


def component_correlations(cds: pd.DataFrame):
    """Which sub-components of folding_demand correlate best with occupancy?"""
    print("\n" + "=" * 60)
    print("2. SUB-COMPONENT CORRELATIONS vs rel_occupancy")
    print("=" * 60)

    components = [
        ('fd_contact_complexity', 'Contact complexity (alpha=0.40)'),
        ('fd_contact_gradient',   'Contact gradient (beta=0.25)'),
        ('fd_domain_transition',  'Domain transition (gamma=0.20)'),
        ('fd_ss_transition',      'SS transition (delta=0.15)'),
        ('fd_plddt_weight',       'pLDDT weight'),
        ('folding_demand',        'COMBINED folding_demand'),
    ]

    occ = cds['rel_occupancy'].values
    results = []

    for col, label in components:
        if col not in cds.columns:
            continue
        vals = cds[col].values
        r, p = stats.spearmanr(vals, occ)
        results.append((col, label, r, p))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"   {label:<40} r = {r:+.4f}  {sig}")

    return results


def binned_analysis(cds: pd.DataFrame):
    """Split folding_demand into bins and compare mean occupancy."""
    print("\n" + "=" * 60)
    print("3. BINNED ANALYSIS: mean occupancy per folding_demand tertile")
    print("=" * 60)

    cds = cds.copy()
    cds['fd_bin'] = pd.qcut(cds['folding_demand'], q=3, labels=['low', 'mid', 'high'])

    for bin_name in ['low', 'mid', 'high']:
        subset = cds[cds['fd_bin'] == bin_name]
        mean_occ = subset['rel_occupancy'].mean()
        median_occ = subset['rel_occupancy'].median()
        mean_fd = subset['folding_demand'].mean()
        print(f"   {bin_name:>4}: fd={mean_fd:.3f}  "
              f"mean_occ={mean_occ:.6f}  median_occ={median_occ:.6f}  "
              f"n={len(subset)}")

    # Mann-Whitney: high vs low
    high = cds[cds['fd_bin'] == 'high']['rel_occupancy'].values
    low = cds[cds['fd_bin'] == 'low']['rel_occupancy'].values
    u_stat, u_p = stats.mannwhitneyu(high, low, alternative='greater')
    print(f"\n   Mann-Whitney U (high > low): U={u_stat:.0f}, p={u_p:.2e}")

    return cds


def per_gene_correlation(cds: pd.DataFrame):
    """Per-gene Spearman correlation."""
    print("\n" + "=" * 60)
    print("4. PER-GENE CORRELATION (Spearman)")
    print("=" * 60)

    results = []
    for gene, group in cds.groupby('gene'):
        if len(group) < 20:
            continue
        fd = group['folding_demand'].values
        occ = group['rel_occupancy'].values
        # Skip genes with zero variance
        if np.std(fd) < 1e-8 or np.std(occ) < 1e-8:
            continue
        r, p = stats.spearmanr(fd, occ)
        results.append({'gene': gene, 'spearman_r': r, 'p_value': p, 'n_codons': len(group)})

    results_df = pd.DataFrame(results).sort_values('spearman_r', ascending=False)

    n_pos = (results_df['spearman_r'] > 0).sum()
    n_neg = (results_df['spearman_r'] < 0).sum()
    n_sig = (results_df['p_value'] < 0.05).sum()
    mean_r = results_df['spearman_r'].mean()
    median_r = results_df['spearman_r'].median()

    print(f"   {len(results_df)} genes analyzed")
    print(f"   Positive correlation: {n_pos}/{len(results_df)} genes")
    print(f"   Significant (p<0.05): {n_sig}/{len(results_df)} genes")
    print(f"   Mean Spearman r:   {mean_r:.4f}")
    print(f"   Median Spearman r: {median_r:.4f}")
    print(f"\n   Top 5 genes:")
    for _, row in results_df.head(5).iterrows():
        print(f"     {row['gene']:>10}: r={row['spearman_r']:+.4f} "
              f"(p={row['p_value']:.2e}, n={row['n_codons']})")
    print(f"\n   Bottom 5 genes:")
    for _, row in results_df.tail(5).iterrows():
        print(f"     {row['gene']:>10}: r={row['spearman_r']:+.4f} "
              f"(p={row['p_value']:.2e}, n={row['n_codons']})")

    return results_df


def structural_context(cds: pd.DataFrame):
    """Compare folding_demand at different structural features."""
    print("\n" + "=" * 60)
    print("5. STRUCTURAL CONTEXT")
    print("=" * 60)

    # Domain boundaries vs non-boundaries
    if 'domain_boundary' in cds.columns:
        boundary = cds[cds['domain_boundary'] > 0]
        non_boundary = cds[cds['domain_boundary'] == 0]
        print(f"   Domain boundaries (n={len(boundary)}):")
        print(f"     mean fd={boundary['folding_demand'].mean():.4f}  "
              f"mean occ={boundary['rel_occupancy'].mean():.6f}")
        print(f"   Non-boundary (n={len(non_boundary)}):")
        print(f"     mean fd={non_boundary['folding_demand'].mean():.4f}  "
              f"mean occ={non_boundary['rel_occupancy'].mean():.6f}")

    # SS class comparison
    if 'rna_local_ss_class' in cds.columns:
        print(f"\n   RNA structure class:")
        for cls in ['stem', 'weak', 'open_loop']:
            subset = cds[cds['rna_local_ss_class'] == cls]
            if len(subset) > 0:
                print(f"     {cls:>10}: mean fd={subset['folding_demand'].mean():.4f}  "
                      f"mean occ={subset['rel_occupancy'].mean():.6f}  n={len(subset)}")


# ─── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(cds: pd.DataFrame, gene_results: pd.DataFrame):
    """Generate multi-panel validation figure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # ── Panel A: Scatter folding_demand vs log(rel_occupancy) ──
    ax1 = fig.add_subplot(gs[0, 0])
    log_occ = np.log10(cds['rel_occupancy'].values + 1e-8)
    ax1.scatter(cds['folding_demand'], log_occ, alpha=0.05, s=2, c='steelblue')
    # Binned means
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_mean = []
    for i in range(len(bins) - 1):
        mask = (cds['folding_demand'] >= bins[i]) & (cds['folding_demand'] < bins[i+1])
        if mask.sum() > 10:
            binned_mean.append(np.mean(log_occ[mask]))
        else:
            binned_mean.append(np.nan)
    ax1.plot(bin_centers, binned_mean, 'r-o', markersize=4, linewidth=2, label='binned mean')
    ax1.set_xlabel('folding_demand')
    ax1.set_ylabel('log10(rel_occupancy)')
    ax1.set_title('A. Overall relationship')
    ax1.legend(fontsize=8)

    # ── Panel B: Sub-component correlations bar chart ──
    ax2 = fig.add_subplot(gs[0, 1])
    components = ['fd_contact_complexity', 'fd_contact_gradient',
                  'fd_domain_transition', 'fd_ss_transition', 'folding_demand']
    labels = ['Contact\ncomplexity', 'Contact\ngradient', 'Domain\ntransition',
              'SS\ntransition', 'Combined']
    colors = ['#5DCAA5', '#7F77DD', '#D85A30', '#EF9F27', '#E24B4A']
    occ = cds['rel_occupancy'].values
    rs = []
    for col in components:
        if col in cds.columns:
            r, _ = stats.spearmanr(cds[col].values, occ)
            rs.append(r)
        else:
            rs.append(0)
    bars = ax2.bar(range(len(rs)), rs, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Spearman r')
    ax2.set_title('B. Component correlations')
    ax2.axhline(0, color='gray', linewidth=0.5)
    for bar, r in zip(bars, rs):
        ax2.text(bar.get_x() + bar.get_width()/2, r + 0.005 * np.sign(r),
                 f'{r:.3f}', ha='center', va='bottom' if r >= 0 else 'top', fontsize=8)

    # ── Panel C: Distribution of per-gene correlations ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(gene_results['spearman_r'], bins=25, color='steelblue', edgecolor='white',
             linewidth=0.5, alpha=0.8)
    ax3.axvline(0, color='gray', linewidth=1, linestyle='--')
    ax3.axvline(gene_results['spearman_r'].mean(), color='red', linewidth=2,
                label=f"mean={gene_results['spearman_r'].mean():.3f}")
    ax3.set_xlabel('Spearman r')
    ax3.set_ylabel('Number of genes')
    ax3.set_title('C. Per-gene correlations')
    ax3.legend(fontsize=8)

    # ── Panels D-F: Example gene profiles (best, median, worst correlation) ──
    sorted_genes = gene_results.sort_values('spearman_r')
    examples = [
        ('D. Best correlation', sorted_genes.iloc[-1]),
        ('E. Median correlation', sorted_genes.iloc[len(sorted_genes)//2]),
        ('F. Worst correlation', sorted_genes.iloc[0]),
    ]

    for idx, (title, row) in enumerate(examples):
        ax = fig.add_subplot(gs[1, idx])
        gene_data = cds[cds['gene'] == row['gene']].copy().reset_index(drop=True)

        # Normalize both to 0-1 for visual comparison
        fd = gene_data['folding_demand'].values
        occ_raw = gene_data['rel_occupancy'].values
        occ_norm = occ_raw / (occ_raw.max() + 1e-8)
        fd_norm = fd / (fd.max() + 1e-8)

        x = range(len(gene_data))
        ax.fill_between(x, occ_norm, alpha=0.3, color='steelblue', label='occupancy')
        ax.plot(x, fd_norm, color='red', linewidth=1.2, alpha=0.8, label='folding_demand')
        ax.set_xlabel('Codon position (CDS)')
        ax.set_ylabel('Normalized score')
        ax.set_title(f"{title}\n{row['gene']} (r={row['spearman_r']:.3f})", fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(-0.05, 1.1)

    # ── Panel G: Boxplot of occupancy per folding_demand tertile ──
    ax7 = fig.add_subplot(gs[2, 0])
    cds_binned = cds.copy()
    cds_binned['fd_tertile'] = pd.qcut(cds_binned['folding_demand'], q=3,
                                        labels=['Low', 'Mid', 'High'])
    data_for_box = [
        np.log10(cds_binned[cds_binned['fd_tertile'] == t]['rel_occupancy'].values + 1e-8)
        for t in ['Low', 'Mid', 'High']
    ]
    bp = ax7.boxplot(data_for_box, labels=['Low', 'Mid', 'High'], patch_artist=True,
                     showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#85B7EB', '#EF9F27', '#E24B4A']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax7.set_xlabel('folding_demand tertile')
    ax7.set_ylabel('log10(rel_occupancy)')
    ax7.set_title('G. Occupancy by fd tertile')

    # ── Panel H: Folding demand vs opening energy ──
    ax8 = fig.add_subplot(gs[2, 1])
    if 'rna_opening_energy_5nt' in cds.columns:
        ax8.scatter(cds['folding_demand'], cds['rna_opening_energy_5nt'],
                    alpha=0.05, s=2, c='steelblue')
        r, p = stats.spearmanr(cds['folding_demand'], cds['rna_opening_energy_5nt'])
        ax8.set_xlabel('folding_demand')
        ax8.set_ylabel('RNA opening energy (5nt)')
        ax8.set_title(f'H. fd vs RNA barrier (r={r:.3f})')

    # ── Panel I: Weight sensitivity ──
    ax9 = fig.add_subplot(gs[2, 2])
    # Test different weight combinations
    weight_sets = [
        ('Current', 0.40, 0.25, 0.20, 0.15),
        ('Contact only', 1.0, 0.0, 0.0, 0.0),
        ('Gradient only', 0.0, 1.0, 0.0, 0.0),
        ('Domain only', 0.0, 0.0, 1.0, 0.0),
        ('SS only', 0.0, 0.0, 0.0, 1.0),
        ('Equal', 0.25, 0.25, 0.25, 0.25),
    ]

    w_labels, w_rs = [], []
    for name, a, b, g, d in weight_sets:
        raw = (a * cds['fd_contact_complexity'].fillna(0) +
               b * cds['fd_contact_gradient'].fillna(0) +
               g * cds['fd_domain_transition'].fillna(0) +
               d * cds['fd_ss_transition'].fillna(0))
        r, _ = stats.spearmanr(raw.values, occ)
        w_labels.append(name)
        w_rs.append(r)

    bars = ax9.barh(range(len(w_labels)), w_rs, color='steelblue', edgecolor='white')
    ax9.set_yticks(range(len(w_labels)))
    ax9.set_yticklabels(w_labels, fontsize=9)
    ax9.set_xlabel('Spearman r with rel_occupancy')
    ax9.set_title('I. Weight sensitivity')
    ax9.axvline(0, color='gray', linewidth=0.5)
    for bar, r in zip(bars, w_rs):
        ax9.text(r + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{r:.3f}', va='center', fontsize=8)

    plt.savefig(OUT_DIR / 'folding_demand_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {OUT_DIR / 'folding_demand_validation.png'}")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Folding Demand Validation")
    print("=" * 60)

    all_data = load_all_genes()

    # Filter to CDS codons with valid folding_demand
    cds = all_data[
        (all_data['region'] == 'CDS') &
        (all_data['folding_demand'].notna()) &
        (all_data['rel_occupancy'].notna())
    ].copy()

    print(f"CDS codons with valid folding_demand: {len(cds)}")
    print(f"Unique genes: {cds['gene'].nunique()}")

    # Run analyses
    global_correlation(cds)
    component_correlations(cds)
    binned_analysis(cds)
    gene_results = per_gene_correlation(cds)
    structural_context(cds)

    # Generate plots
    make_plots(cds, gene_results)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    r_global, _ = stats.spearmanr(cds['folding_demand'], cds['rel_occupancy'])
    n_pos = (gene_results['spearman_r'] > 0).sum()
    n_total = len(gene_results)

    if r_global > 0.1:
        print(f"  Positive global correlation (r={r_global:.3f})")
        print(f"  The score captures SOME signal.")
    elif r_global > 0:
        print(f"  Weak positive global correlation (r={r_global:.3f})")
        print(f"  Signal is there but noisy. This is expected for a")
        print(f"  heuristic score -- the GNN should improve on this.")
    else:
        print(f"  No positive global correlation (r={r_global:.3f})")
        print(f"  The current weight combination may need adjustment.")
        print(f"  Check Panel I (weight sensitivity) for better options.")

    print(f"\n  {n_pos}/{n_total} genes show positive per-gene correlation")
    print(f"\n  Check: {OUT_DIR / 'folding_demand_validation.png'}")


if __name__ == '__main__':
    main()