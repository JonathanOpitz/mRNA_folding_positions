#!/usr/bin/env python3
"""
Compare two RiboDecode+fold_penalty runs (typically γ=0 baseline vs γ>0).

Inputs:
    Two *_results.json files from data/optimized/, produced by
    ribodecode_with_fold_penalty.py.

What it measures
════════════════

1. CODON CHOICE BY FOLDING DEMAND (the main mechanistic test)
   For each codon position we have:
     - fd_pred  (from the GNN, fixed — depends only on protein)
     - TAI      (derived from the chosen codon)

   We bucket positions by fd_pred (<0.2 / 0.2-0.5 / >0.5) and compare
   mean TAI per bucket between the two runs. The fold_penalty should
   REDUCE mean TAI in the high-fd bucket (slower codons at structural
   hotspots) while leaving the low-fd bucket ~unchanged.

2. PAUSING / FD CORRELATION
   Pausing proxy = 1/TAI per position. A run that respects folding
   should produce a POSITIVE Spearman between pausing and fd_pred
   across the CDS. Baseline (γ=0) should be ~0.

3. GLOBAL METRICS
   Final predicted RPF, tool MFE, mean TAI, protein identity check.

Usage
═════
    python compare_runs.py \
        data/optimized/ACTB_fold000_mfe000_HEK293T_ep1_results.json \
        data/optimized/ACTB_fold030_mfe000_HEK293T_ep1_results.json

Output
══════
    data/optimized/compare_ACTB_fold000_vs_fold030.png   (4-panel plot)
    prints a summary table to stdout
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ─── Codon tables ────────────────────────────────────────────────────────────

CODON_TAI = {
    'TTT':0.42,'TTC':1.00,'TTA':0.08,'TTG':0.42,
    'CTT':0.42,'CTC':0.58,'CTA':0.08,'CTG':1.00,
    'ATT':0.58,'ATC':1.00,'ATA':0.08,'ATG':1.00,
    'GTT':0.42,'GTC':0.58,'GTA':0.08,'GTG':1.00,
    'TCT':0.58,'TCC':0.75,'TCA':0.25,'TCG':0.17,
    'CCT':0.58,'CCC':0.75,'CCA':0.42,'CCG':0.17,
    'ACT':0.58,'ACC':1.00,'ACA':0.42,'ACG':0.17,
    'GCT':0.75,'GCC':1.00,'GCA':0.42,'GCG':0.17,
    'TAT':0.42,'TAC':1.00,'TAA':0.00,'TAG':0.00,
    'CAT':0.42,'CAC':1.00,'CAA':0.42,'CAG':1.00,
    'AAT':0.42,'AAC':1.00,'AAA':0.42,'AAG':1.00,
    'GAT':0.42,'GAC':1.00,'GAA':0.42,'GAG':1.00,
    'TGT':0.42,'TGC':1.00,'TGA':0.00,'TGG':1.00,
    'CGT':0.42,'CGC':0.75,'CGA':0.08,'CGG':0.17,
    'AGT':0.25,'AGC':0.75,'AGA':0.42,'AGG':0.25,
    'GGT':0.42,'GGC':1.00,'GGA':0.25,'GGG':0.17,
}

AA_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def seq_to_codons(seq: str):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def translate(seq: str):
    return ''.join(AA_TABLE.get(c, '?') for c in seq_to_codons(seq))


def tai_per_position(seq: str):
    """TAI value per codon position (0.5 default for missing)."""
    return np.array([CODON_TAI.get(c, 0.5) for c in seq_to_codons(seq)])


def load_fd_pred_from_csv(gene: str, wt_dir: Path, n_codons_expected: int):
    """
    Load folding_demand from the WT CSV — this is the ground-truth signal
    the GNN was trained on. Works as long as WT CSV is present; if you
    have GNN-predicted fd_pred stored somewhere else, adapt this.
    """
    gene_u = gene.upper()
    candidates = sorted(p for p in wt_dir.glob("*_with_folddemand.csv")
                        if gene_u in p.name.upper())
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    if 'folding_demand' not in cds.columns:
        return None
    fd = pd.to_numeric(cds['folding_demand'], errors='coerce').fillna(0.0).values
    if len(fd) != n_codons_expected:
        # length mismatch; truncate or pad with zeros — better than crashing
        if len(fd) > n_codons_expected:
            fd = fd[:n_codons_expected]
        else:
            fd = np.concatenate([fd, np.zeros(n_codons_expected - len(fd))])
    return fd


def compute_real_mfe(seq: str):
    """Compute MFE using ViennaRNA if available; return None otherwise."""
    try:
        import RNA
        _, mfe = RNA.fold(seq.replace('T', 'U'))
        return float(mfe)
    except Exception:
        return None


def fd_bucket_stats(tai: np.ndarray, fd: np.ndarray):
    """Mean TAI per fd_pred bucket."""
    buckets = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.01)]
    out = {}
    for lo, hi in buckets:
        mask = (fd >= lo) & (fd < hi)
        n = int(mask.sum())
        mean_tai = float(tai[mask].mean()) if n else float('nan')
        std_tai  = float(tai[mask].std())  if n else float('nan')
        out[f"{lo:.1f}-{hi:.2f}"] = dict(n=n, mean_tai=mean_tai, std_tai=std_tai)
    return out


def pausing_fd_corr(tai: np.ndarray, fd: np.ndarray):
    """Spearman correlation between pausing (1/TAI) and fd_pred."""
    # Avoid div by zero: stop codons have TAI=0
    pausing = np.where(tai > 0, 1.0 / tai, 0.0)
    if fd.std() == 0 or pausing.std() == 0:
        return float('nan'), float('nan')
    r, p = stats.spearmanr(pausing, fd)
    return float(r), float(p)


# ─── main comparison ─────────────────────────────────────────────────────────

def load_run(path: Path):
    with open(path) as f:
        data = json.load(f)
    seq      = data['sequence']['cds_only']
    gene     = data['gene']
    gamma    = data['config']['fold_gamma']
    mfe_w    = data['config']['mfe_weight']
    rpf      = data['final']['rpf_model']
    mfe_tool = data['final']['mfe_tool']
    n_ep     = data['config']['n_epochs']
    return dict(path=path, data=data, seq=seq, gene=gene, gamma=gamma,
                mfe_w=mfe_w, rpf=rpf, mfe_tool=mfe_tool, n_epochs=n_ep)


def analyze(run, fd):
    """Compute all metrics for one run given the gene's fd array."""
    tai = tai_per_position(run['seq'])
    if len(tai) != len(fd):
        # trim to min length
        n = min(len(tai), len(fd))
        tai, fd = tai[:n], fd[:n]

    prot_gen = translate(run['seq'])

    res = dict(
        run_label  = f"γ={run['gamma']}",
        n_codons   = len(tai),
        mean_tai   = float(tai.mean()),
        median_tai = float(np.median(tai)),
        rpf_model  = run['rpf'],
        mfe_tool   = run['mfe_tool'],
        mfe_recomputed = compute_real_mfe(run['seq']),
        tai_per_pos = tai,
        fd_per_pos  = fd,
        buckets    = fd_bucket_stats(tai, fd),
    )
    r, p = pausing_fd_corr(tai, fd)
    res['pausing_fd_spearman']   = r
    res['pausing_fd_pvalue']     = p
    return res


def print_summary(runA, runB, anA, anB):
    print()
    print("═" * 78)
    print(f"  Comparison: {runA['gene']}  "
          f"[{runA['path'].name}  vs  {runB['path'].name}]")
    print("═" * 78)

    print(f"\n  Config:")
    print(f"    Run A: fold_gamma={runA['gamma']}  mfe_weight={runA['mfe_w']}  epochs={runA['n_epochs']}")
    print(f"    Run B: fold_gamma={runB['gamma']}  mfe_weight={runB['mfe_w']}  epochs={runB['n_epochs']}")

    print(f"\n  Global metrics:")
    fmt = "    {:<28} {:>12}  {:>12}  {:>12}"
    print(fmt.format("", "γ="+str(runA['gamma']), "γ="+str(runB['gamma']), "Δ (B-A)"))
    print(fmt.format("-"*28, "-"*12, "-"*12, "-"*12))
    print(fmt.format("RPF (model)",
                     f"{anA['rpf_model']:.3f}",
                     f"{anB['rpf_model']:.3f}",
                     f"{anB['rpf_model']-anA['rpf_model']:+.3f}"))
    print(fmt.format("MFE (tool, logged)",
                     f"{anA['mfe_tool']:.1f}",
                     f"{anB['mfe_tool']:.1f}",
                     f"{anB['mfe_tool']-anA['mfe_tool']:+.1f}"))
    if anA['mfe_recomputed'] is not None and anB['mfe_recomputed'] is not None:
        print(fmt.format("MFE (recomputed now)",
                         f"{anA['mfe_recomputed']:.1f}",
                         f"{anB['mfe_recomputed']:.1f}",
                         f"{anB['mfe_recomputed']-anA['mfe_recomputed']:+.1f}"))
    print(fmt.format("Mean TAI",
                     f"{anA['mean_tai']:.3f}",
                     f"{anB['mean_tai']:.3f}",
                     f"{anB['mean_tai']-anA['mean_tai']:+.3f}"))
    print(fmt.format("Pausing↔FD Spearman",
                     f"{anA['pausing_fd_spearman']:+.3f}",
                     f"{anB['pausing_fd_spearman']:+.3f}",
                     f"{anB['pausing_fd_spearman']-anA['pausing_fd_spearman']:+.3f}"))

    print(f"\n  TAI by folding-demand bucket (lower TAI = slower codon = desired at high FD):")
    fmt2 = "    {:<14} {:>6}    {:>12}    {:>12}    {:>12}"
    print(fmt2.format("FD range", "n", "mean TAI γ=A", "mean TAI γ=B", "Δ"))
    print(fmt2.format("-"*14, "-"*6, "-"*12, "-"*12, "-"*12))
    for key in anA['buckets']:
        a = anA['buckets'][key]; b = anB['buckets'][key]
        delta = b['mean_tai'] - a['mean_tai']
        print(fmt2.format(key, a['n'], f"{a['mean_tai']:.3f}", f"{b['mean_tai']:.3f}",
                          f"{delta:+.3f}"))

    # Sanity check: same protein encoded?
    protA = translate(runA['seq'])
    protB = translate(runB['seq'])
    if protA == protB:
        print(f"\n  ✓ Both runs encode the same protein ({len(protA)} aa, identity=100%)")
    else:
        nm = min(len(protA), len(protB))
        ident = sum(a == b for a, b in zip(protA, protB)) / nm
        print(f"\n  ⚠  Proteins DIFFER: identity={100*ident:.1f}%  "
              f"(lenA={len(protA)}, lenB={len(protB)})")


def plot_comparison(runA, runB, anA, anB, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: TAI by FD bucket
    ax = axes[0, 0]
    keys = list(anA['buckets'].keys())
    x = np.arange(len(keys))
    w = 0.35
    a_means = [anA['buckets'][k]['mean_tai'] for k in keys]
    b_means = [anB['buckets'][k]['mean_tai'] for k in keys]
    a_stds  = [anA['buckets'][k]['std_tai']  for k in keys]
    b_stds  = [anB['buckets'][k]['std_tai']  for k in keys]
    ax.bar(x - w/2, a_means, w, yerr=a_stds, capsize=4,
           label=f"γ={runA['gamma']}", color='steelblue')
    ax.bar(x + w/2, b_means, w, yerr=b_stds, capsize=4,
           label=f"γ={runB['gamma']}", color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_xlabel('folding_demand bucket')
    ax.set_ylabel('mean TAI')
    ax.set_title('TAI per FD bucket\n(lower = slower codons at high FD = desired)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Pausing vs FD scatter (both runs overlaid)
    ax = axes[0, 1]
    pausingA = np.where(anA['tai_per_pos'] > 0, 1.0/anA['tai_per_pos'], 0.0)
    pausingB = np.where(anB['tai_per_pos'] > 0, 1.0/anB['tai_per_pos'], 0.0)
    ax.scatter(anA['fd_per_pos'], pausingA, alpha=0.4, s=15,
               label=f"γ={runA['gamma']} (r={anA['pausing_fd_spearman']:+.2f})",
               color='steelblue')
    ax.scatter(anB['fd_per_pos'], pausingB, alpha=0.4, s=15,
               label=f"γ={runB['gamma']} (r={anB['pausing_fd_spearman']:+.2f})",
               color='coral')
    ax.set_xlabel('folding_demand (per position)')
    ax.set_ylabel('pausing proxy = 1 / TAI')
    ax.set_title('Pausing vs FD (positive corr = fold-aware)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: TAI profile along CDS
    ax = axes[1, 0]
    n = min(len(anA['tai_per_pos']), len(anB['tai_per_pos']))
    x = np.arange(n)
    # Smooth with running mean (window 15) for readability
    w = 15
    def rmean(a, w):
        pad = np.pad(a, (w//2, w - 1 - w//2), mode='edge')
        return np.convolve(pad, np.ones(w)/w, mode='valid')
    ax.plot(x, rmean(anA['tai_per_pos'][:n], w), label=f"TAI γ={runA['gamma']}",
            color='steelblue', alpha=0.8)
    ax.plot(x, rmean(anB['tai_per_pos'][:n], w), label=f"TAI γ={runB['gamma']}",
            color='coral', alpha=0.8)
    # FD overlay on secondary axis
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, rmean(anA['fd_per_pos'][:n], w),
                     color='gray', alpha=0.2, label='FD (target)')
    ax2.set_ylabel('folding_demand', color='gray')
    ax.set_xlabel('codon position')
    ax.set_ylabel('TAI (smoothed, w=15)')
    ax.set_title('TAI profile along CDS (smoothed)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        f"Gene: {runA['gene']}",
        f"",
        f"Run A: γ={runA['gamma']}, mfe_w={runA['mfe_w']}",
        f"Run B: γ={runB['gamma']}, mfe_w={runB['mfe_w']}",
        f"Epochs: A={runA['n_epochs']}  B={runB['n_epochs']}",
        f"",
        f"────────── Global ──────────",
        f"RPF_model:     A={anA['rpf_model']:.2f}  B={anB['rpf_model']:.2f}  Δ={anB['rpf_model']-anA['rpf_model']:+.2f}",
        f"MFE (tool):    A={anA['mfe_tool']:.1f}   B={anB['mfe_tool']:.1f}   Δ={anB['mfe_tool']-anA['mfe_tool']:+.1f}",
        f"Mean TAI:      A={anA['mean_tai']:.3f}  B={anB['mean_tai']:.3f}  Δ={anB['mean_tai']-anA['mean_tai']:+.3f}",
        f"",
        f"─────── Fold-compliance ───────",
        f"Pausing↔FD Spearman:",
        f"  A: {anA['pausing_fd_spearman']:+.3f}  (p={anA['pausing_fd_pvalue']:.2e})",
        f"  B: {anB['pausing_fd_spearman']:+.3f}  (p={anB['pausing_fd_pvalue']:.2e})",
        f"",
        f"Interpretation:",
        f"  Positive corr in B but not A →",
        f"  fold_penalty works as intended.",
    ]
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
            family='monospace', fontsize=9, verticalalignment='top')

    fig.suptitle(f"RiboDecode fold_penalty comparison: {runA['gene']}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\n  [plot] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path, help="First JSON (typically γ=0 baseline)")
    parser.add_argument("run_b", type=Path, help="Second JSON (typically γ>0)")
    parser.add_argument("--wt_dir",
                        default="/Users/jonathanopitz/Desktop/Master/data/ribo_counts",
                        type=Path,
                        help="Dir containing *_with_folddemand.csv for fd_pred loading")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: next to run_b)")
    args = parser.parse_args()

    runA = load_run(args.run_a)
    runB = load_run(args.run_b)

    if runA['gene'] != runB['gene']:
        sys.exit(f"Gene mismatch: {runA['gene']} vs {runB['gene']}")

    n = len(seq_to_codons(runA['seq']))
    fd = load_fd_pred_from_csv(runA['gene'], args.wt_dir, n)
    if fd is None:
        sys.exit(f"No folding_demand CSV found for {runA['gene']} in {args.wt_dir}")

    anA = analyze(runA, fd)
    anB = analyze(runB, fd)

    print_summary(runA, runB, anA, anB)

    out_path = args.out
    if out_path is None:
        stem = f"compare_{runA['gene']}_gamma{runA['gamma']}_vs_gamma{runB['gamma']}.png"
        out_path = args.run_b.parent / stem.replace('.', '', 1).replace('compare_', 'compare_', 1)
        # Re-do without the first replace, just use the stem as-is:
        out_path = args.run_b.parent / f"compare_{runA['gene']}_g{runA['gamma']}_vs_g{runB['gamma']}.png"

    plot_comparison(runA, runB, anA, anB, out_path)


if __name__ == '__main__':
    main()
