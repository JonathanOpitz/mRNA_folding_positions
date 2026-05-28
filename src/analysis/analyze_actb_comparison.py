#!/usr/bin/env python3
"""
analyze_actb_comparison.py

Compare two ACTB optimization runs:
  - γ=0.0 (no fold penalty)        : MFE-only optimization (RiboDecode default)
  - γ=0.5 (medium fold penalty)    : Adds fold_demand-aware codon selection

Outputs:
  1. Codon-level diff: which codons changed, where
  2. TAI profile per position (running mean)
  3. Hydrophobicity correlation: do slow codons cluster near hydrophilic stretches?
  4. AlphaFold-based comparison: pLDDT, domain boundaries, contact density vs codon speed
  5. Aggregation prediction (TANGO-like) vs codon usage
  6. Multi-panel summary figure

Usage:
  python analyze_actb_comparison.py
  
Outputs saved to: data/analysis/ACTB_g0_vs_g05/
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from collections import Counter
from scipy.stats import spearmanr, mannwhitneyu

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GENE = "ATCB"
BASE = Path(__file__).resolve().parents[2]
JSON_G0   = BASE / "data/optimized/ACTB_fold000_mfe070_HEK293T_ep5_results.json"
JSON_G05  = BASE / "data/optimized/ACTB_fold030_mfe070_HEK293T_ep5_results.json"
PDB_DIR   = BASE / "data/alphafold_results/ATCB"
WT_CSV    = BASE / "data/ribo_counts/ATCB_ribosome_counts_with_structure_with_rnaplfold_with_folddemand.csv"
OUT_DIR   = BASE / "data/analysis/ATCB_g0_vs_g05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Codon → AA
CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L', 'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M', 'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S', 'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T', 'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*', 'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K', 'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W', 'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R', 'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

# TAI = relative codon translation speed (HIGH=fast, LOW=slow)
TAI = {
    'TTT':0.42,'TTC':1.00,'TTA':0.08,'TTG':0.42, 'CTT':0.42,'CTC':0.58,'CTA':0.08,'CTG':1.00,
    'ATT':0.58,'ATC':1.00,'ATA':0.08,'ATG':1.00, 'GTT':0.42,'GTC':0.58,'GTA':0.08,'GTG':1.00,
    'TCT':0.58,'TCC':0.75,'TCA':0.25,'TCG':0.17, 'CCT':0.58,'CCC':0.75,'CCA':0.42,'CCG':0.17,
    'ACT':0.58,'ACC':1.00,'ACA':0.42,'ACG':0.17, 'GCT':0.75,'GCC':1.00,'GCA':0.42,'GCG':0.17,
    'TAT':0.42,'TAC':1.00,'TAA':0.00,'TAG':0.00, 'CAT':0.42,'CAC':1.00,'CAA':0.42,'CAG':1.00,
    'AAT':0.42,'AAC':1.00,'AAA':0.42,'AAG':1.00, 'GAT':0.42,'GAC':1.00,'GAA':0.42,'GAG':1.00,
    'TGT':0.42,'TGC':1.00,'TGA':0.00,'TGG':1.00, 'CGT':0.42,'CGC':0.75,'CGA':0.08,'CGG':0.17,
    'AGT':0.25,'AGC':0.75,'AGA':0.42,'AGG':0.25, 'GGT':0.42,'GGC':1.00,'GGA':0.25,'GGG':0.17,
}

# Kyte-Doolittle hydrophobicity (positive = hydrophobic, negative = hydrophilic)
KD = {
    'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
    'H':-3.2,'I': 4.5,'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2,'*': 0.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_run(path):
    """Load a results.json and return the final-epoch CDS sequence + metrics."""
    with open(path) as f:
        d = json.load(f)
    cds = d["sequence"]["cds_only"]
    n_ep = d["config"]["n_epochs"]
    best_per_ep = d["sequence"].get("best_seq", {}) or d["metrics_per_epoch"].get("best_seq", {})
    final_seq = best_per_ep.get(str(n_ep), cds)
    return {
        "name": path.stem,
        "fold_gamma": d["config"]["fold_gamma"],
        "mfe_weight": d["config"]["mfe_weight"],
        "n_epochs": n_ep,
        "cds_orig": cds,            # input WT sequence
        "seq": final_seq,            # OPTIMIZED sequence (final epoch)
        "rpf": d["final"]["rpf_model"],
        "mfe_model": d["final"]["mfe_model"],
        "mfe_tool": d["final"]["mfe_tool"],
        "metrics_per_epoch": d["metrics_per_epoch"],
    }


def codons_of(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def tai_profile(codons):
    return np.array([TAI.get(c, 0.5) for c in codons])


def aa_seq(codons):
    return "".join(CODON_TABLE.get(c, "?") for c in codons)


def hydrophobicity_profile(aa, window=9):
    """KD running mean. Negative = hydrophilic stretch."""
    kd = np.array([KD.get(a, 0.0) for a in aa])
    if len(kd) < window:
        return kd
    half = window // 2
    smooth = np.convolve(kd, np.ones(window)/window, mode='same')
    return smooth


def parse_pdb_plddt_and_ca(pdb_path):
    """Return (plddt_per_residue, ca_coords_per_residue) as dicts keyed by residue idx (0-based)."""
    plddt, ca = {}, {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            res_idx = int(line[22:26].strip()) - 1   # 1-based to 0-based
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            bfactor = float(line[60:66])              # AlphaFold encodes pLDDT here
            plddt[res_idx] = bfactor
            ca[res_idx] = np.array([x, y, z])
    return plddt, ca


def find_pdb(gene_dir):
    """Find best AlphaFold PDB (rank_001 preferred)."""
    if not gene_dir.exists():
        return None
    pdbs = list(gene_dir.glob("*.pdb"))
    if not pdbs:
        return None
    # Prefer rank_001
    rank1 = [p for p in pdbs if "rank_001" in p.name or "rank_1" in p.name]
    if rank1:
        return rank1[0]
    # Else prefer model_1
    m1 = [p for p in pdbs if "model_1" in p.name]
    if m1:
        return m1[0]
    return pdbs[0]


def contact_density(ca_coords, n_res, threshold=8.0, sep=6):
    """For each residue, count Cα contacts within threshold Å (skipping nearby in sequence)."""
    cd = np.zeros(n_res)
    res_indices = sorted(ca_coords.keys())
    for i, ri in enumerate(res_indices):
        if ri >= n_res:
            continue
        for rj in res_indices[i+1:]:
            if rj >= n_res:
                continue
            if abs(ri - rj) < sep:
                continue
            d = np.linalg.norm(ca_coords[ri] - ca_coords[rj])
            if d < threshold:
                cd[ri] += 1
                cd[rj] += 1
    return cd


def hydrophilic_stretches(hydro, threshold=-1.0, min_len=5):
    """Find regions where hydrophobicity smoothed value is below threshold."""
    in_stretch = False
    start = 0
    stretches = []
    for i, v in enumerate(hydro):
        if v < threshold and not in_stretch:
            in_stretch = True
            start = i
        elif v >= threshold and in_stretch:
            in_stretch = False
            if i - start >= min_len:
                stretches.append((start, i))
    if in_stretch and len(hydro) - start >= min_len:
        stretches.append((start, len(hydro)))
    return stretches


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  ACTB Optimization Comparison: γ=0.0 vs γ=0.5")
print(f"{'='*70}\n")

run_g0 = load_run(JSON_G0)
run_g05 = load_run(JSON_G05)

c_g0  = codons_of(run_g0["seq"])
c_g05 = codons_of(run_g05["seq"])
c_wt  = codons_of(run_g0["cds_orig"])      # same WT for both runs

aa_g0  = aa_seq(c_g0)
aa_g05 = aa_seq(c_g05)
aa_wt  = aa_seq(c_wt)

n_codons = len(c_g0)
print(f"Sequences: {n_codons} codons")
print(f"  WT      → AA[:30]:  {aa_wt[:30]}")
print(f"  γ=0.0   → AA[:30]:  {aa_g0[:30]}")
print(f"  γ=0.5   → AA[:30]:  {aa_g05[:30]}")

# Sanity check: AA sequences should be identical (synonymous codon changes only)
mismatches_g0_wt  = sum(a != b for a, b in zip(aa_g0, aa_wt))
mismatches_g05_wt = sum(a != b for a, b in zip(aa_g05, aa_wt))
print(f"\nAA mismatches vs WT:")
print(f"  γ=0.0   vs WT:  {mismatches_g0_wt}")
print(f"  γ=0.5   vs WT:  {mismatches_g05_wt}")
if mismatches_g0_wt > 0 or mismatches_g05_wt > 0:
    print("  WARNING: nonsynonymous changes detected (should be zero)")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CODON-LEVEL DIFFERENCES
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*70}\n1. CODON-LEVEL DIFFERENCES\n{'─'*70}")

n_diff = sum(1 for a, b in zip(c_g0, c_g05) if a != b)
print(f"\nCodons different between γ=0 and γ=0.5: {n_diff}/{n_codons} ({100*n_diff/n_codons:.1f}%)")

# Where do they differ?
diff_positions = [i for i, (a, b) in enumerate(zip(c_g0, c_g05)) if a != b]
print(f"Positions of differences (first 20): {diff_positions[:20]}")

# Codon usage frequency
def codon_freq(codons):
    n = len(codons)
    return {c: codons.count(c)/n for c in set(codons)}

freq_g0  = Counter(c_g0)
freq_g05 = Counter(c_g05)
freq_wt  = Counter(c_wt)

# Top codons that γ=0.5 uses more than γ=0
diff_freq = []
for codon in CODON_TABLE:
    f_g0 = freq_g0.get(codon, 0) / n_codons
    f_g05 = freq_g05.get(codon, 0) / n_codons
    f_wt = freq_wt.get(codon, 0) / n_codons
    diff_freq.append({
        "codon": codon, "aa": CODON_TABLE[codon], "tai": TAI.get(codon, 0.5),
        "freq_wt": f_wt, "freq_g0": f_g0, "freq_g05": f_g05,
        "delta_g05_minus_g0": f_g05 - f_g0,
    })
df_freq = pd.DataFrame(diff_freq).sort_values("delta_g05_minus_g0")

# Save full table
df_freq.to_csv(OUT_DIR / "codon_usage_comparison.csv", index=False)
print(f"\n[Saved] {OUT_DIR / 'codon_usage_comparison.csv'}")

print(f"\nTop-5 codons more used in γ=0.5 vs γ=0 (potentially recruited slow codons):")
for _, row in df_freq.tail(5)[::-1].iterrows():
    print(f"  {row['codon']} ({row['aa']})  TAI={row['tai']:.2f}  Δ={row['delta_g05_minus_g0']:+.3f}")

print(f"\nTop-5 codons less used in γ=0.5 vs γ=0 (potentially abandoned fast codons):")
for _, row in df_freq.head(5).iterrows():
    print(f"  {row['codon']} ({row['aa']})  TAI={row['tai']:.2f}  Δ={row['delta_g05_minus_g0']:+.3f}")

# Mean TAI of changes
tai_g0 = tai_profile(c_g0)
tai_g05 = tai_profile(c_g05)
tai_wt = tai_profile(c_wt)
print(f"\nMean TAI:")
print(f"  WT:     {tai_wt.mean():.3f}")
print(f"  γ=0.0:  {tai_g0.mean():.3f}")
print(f"  γ=0.5:  {tai_g05.mean():.3f}")
print(f"  Δ(γ=0.5 − γ=0): {tai_g05.mean() - tai_g0.mean():+.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ALPHAFOLD STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*70}\n2. ALPHAFOLD STRUCTURE\n{'─'*70}")

pdb_path = find_pdb(PDB_DIR)
plddt_arr = None
contact_arr = None
if pdb_path:
    print(f"PDB: {pdb_path.name}")
    plddt, ca = parse_pdb_plddt_and_ca(pdb_path)
    plddt_arr = np.zeros(n_codons)
    for i in range(n_codons):
        plddt_arr[i] = plddt.get(i, 0.0)
    contact_arr = contact_density(ca, n_codons)
    print(f"Mean pLDDT: {plddt_arr[plddt_arr > 0].mean():.1f}")
    print(f"Mean contact density: {contact_arr.mean():.2f}")
else:
    print(f"No PDB found in {PDB_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HYDROPHOBICITY & CORRELATION WITH CODON SPEED
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*70}\n3. HYDROPHOBICITY ANALYSIS\n{'─'*70}")

hydro = hydrophobicity_profile(aa_wt, window=9)   # KD smoothed (window-9)
print(f"Mean hydrophobicity (KD): {hydro.mean():+.2f}")
print(f"Hydrophilic stretches (KD < -1.0, length ≥ 5): {len(hydrophilic_stretches(hydro))}")

# Correlation: TAI vs hydrophobicity
# Theory: γ=0.5 should put slow codons in hydrophilic regions (negative correlation)
rho_g0,  p_g0  = spearmanr(tai_g0,  hydro)
rho_g05, p_g05 = spearmanr(tai_g05, hydro)
print(f"\nSpearman ρ(TAI, hydrophobicity):")
print(f"  γ=0.0:  ρ={rho_g0:+.3f}  p={p_g0:.2e}")
print(f"  γ=0.5:  ρ={rho_g05:+.3f}  p={p_g05:.2e}")
print(f"  → If γ=0.5 puts slow codons in hydrophilic regions, ρ should be more positive than γ=0")
print(f"     (because positive ρ between TAI and hydrophobicity means hydrophobic→fast, hydrophilic→slow)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONTACT DENSITY CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

if contact_arr is not None:
    print(f"\n{'─'*70}\n4. PROTEIN CONTACT DENSITY\n{'─'*70}")
    rho_g0_cd,  p_g0_cd  = spearmanr(tai_g0,  contact_arr)
    rho_g05_cd, p_g05_cd = spearmanr(tai_g05, contact_arr)
    print(f"\nSpearman ρ(TAI, contact_density):")
    print(f"  γ=0.0:  ρ={rho_g0_cd:+.3f}  p={p_g0_cd:.2e}")
    print(f"  γ=0.5:  ρ={rho_g05_cd:+.3f}  p={p_g05_cd:.2e}")
    print(f"  → If γ=0.5 slows down at structurally dense regions, ρ should be MORE NEGATIVE")
    print(f"     (high contact density → low TAI = slow codon)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. WT CSV: folding_demand profile (for reference)
# ═══════════════════════════════════════════════════════════════════════════════

fd_wt = None
if WT_CSV.exists():
    df_wt = pd.read_csv(WT_CSV)
    cds_df = df_wt[df_wt['region'] == 'CDS'].reset_index(drop=True)
    if 'folding_demand' in cds_df.columns:
        fd_wt = cds_df['folding_demand'].values[:n_codons]
        print(f"\n{'─'*70}\n5. GROUND-TRUTH FOLDING_DEMAND (from WT CSV)\n{'─'*70}")
        print(f"Mean folding_demand: {fd_wt.mean():.3f}")
        print(f"Range: {fd_wt.min():.3f} → {fd_wt.max():.3f}")
        # Correlation with TAI
        rho_g0_fd,  _ = spearmanr(tai_g0,  fd_wt)
        rho_g05_fd, _ = spearmanr(tai_g05, fd_wt)
        print(f"\nSpearman ρ(TAI, folding_demand):")
        print(f"  γ=0.0:  ρ={rho_g0_fd:+.3f}")
        print(f"  γ=0.5:  ρ={rho_g05_fd:+.3f}")
        print(f"  → Goal: γ=0.5 should be MORE NEGATIVE (high fold_demand → slow codon)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. AGGREGATION-PRONE REGIONS (simple proxy: hydrophobic + structured)
# ═══════════════════════════════════════════════════════════════════════════════

# Crude aggregation proxy: high hydrophobicity AND high structure (β-sheet propensity)
# Better: use TANGO output if available. Here a Lobanov-Galzitskaya-style proxy.
def aggregation_proxy(aa, window=7):
    """Score = mean(hydrophobicity * β-propensity) over window."""
    BETA_PROP = {  # Chou-Fasman β-sheet propensity
        'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'E':0.37,'Q':1.10,'G':0.75,
        'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,
        'T':1.19,'W':1.37,'Y':1.47,'V':1.70,'*':0.0,
    }
    h = np.array([KD.get(a, 0.0) for a in aa])
    b = np.array([BETA_PROP.get(a, 1.0) for a in aa])
    score = np.maximum(h, 0) * b   # only count hydrophobic positions
    half = window // 2
    smooth = np.convolve(score, np.ones(window)/window, mode='same')
    return smooth

agg_score = aggregation_proxy(aa_wt)
agg_threshold = np.percentile(agg_score, 85)
agg_hot_positions = np.where(agg_score > agg_threshold)[0]

print(f"\n{'─'*70}\n6. AGGREGATION-PRONE REGIONS (proxy)\n{'─'*70}")
print(f"Top 15% positions: {len(agg_hot_positions)} codons")
print(f"Threshold: {agg_threshold:.2f}")

# Mean TAI before vs at aggregation hotspots
window_before = 3   # 3 codons upstream
mean_tai_before = []
for pos in agg_hot_positions:
    if pos >= window_before:
        for which, tai in [("g0", tai_g0), ("g05", tai_g05)]:
            mean_tai_before.append({
                "pos": pos,
                "which": which,
                "mean_tai_before": tai[pos-window_before:pos].mean(),
                "mean_tai_at": tai[pos],
            })
df_agg = pd.DataFrame(mean_tai_before)
if len(df_agg) > 0:
    g0_before = df_agg[df_agg.which == "g0"]["mean_tai_before"].mean()
    g05_before = df_agg[df_agg.which == "g05"]["mean_tai_before"].mean()
    print(f"\nMean TAI just BEFORE aggregation hotspots ({window_before} codons upstream):")
    print(f"  γ=0.0:  {g0_before:.3f}")
    print(f"  γ=0.5:  {g05_before:.3f}")
    print(f"  → If γ=0.5 slows down BEFORE hotspots, value should be LOWER")
    if g05_before < g0_before:
        print(f"  ✓ γ=0.5 slows down before aggregation hotspots (Δ={g05_before-g0_before:+.3f})")
    else:
        print(f"  ✗ γ=0.5 does NOT slow down before hotspots (Δ={g05_before-g0_before:+.3f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. METRICS TRAJECTORIES OVER EPOCHS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*70}\n7. EPOCH TRAJECTORIES\n{'─'*70}")

eps = list(range(1, run_g0["n_epochs"] + 1))
rpf_g0  = [run_g0["metrics_per_epoch"]["rpf_model"][str(e)]  for e in eps]
rpf_g05 = [run_g05["metrics_per_epoch"]["rpf_model"][str(e)] for e in eps]
mfe_g0  = [run_g0["metrics_per_epoch"]["mfe_tool"][str(e)]   for e in eps]
mfe_g05 = [run_g05["metrics_per_epoch"]["mfe_tool"][str(e)]  for e in eps]

print(f"\nFinal RPF (model):")
print(f"  γ=0.0:  {rpf_g0[-1]:.2f}")
print(f"  γ=0.5:  {rpf_g05[-1]:.2f}")
print(f"\nFinal MFE (RNAfold):")
print(f"  γ=0.0:  {mfe_g0[-1]:.2f}")
print(f"  γ=0.5:  {mfe_g05[-1]:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. MULTI-PANEL FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*70}\n8. PLOTTING\n{'─'*70}")

fig = plt.figure(figsize=(16, 14))
gs = GridSpec(5, 2, height_ratios=[1.2, 1, 1, 1, 1], hspace=0.55, wspace=0.25)

# Panel 1: Codon usage delta (top, full-width)
ax1 = fig.add_subplot(gs[0, :])
df_top = df_freq.reindex(df_freq["delta_g05_minus_g0"].abs().sort_values(ascending=False).index).head(20)
df_top = df_top.sort_values("delta_g05_minus_g0")
colors = ['#d62728' if d < 0 else '#2ca02c' for d in df_top["delta_g05_minus_g0"]]
bar_labels = [f"{c}\n({a}, TAI={t:.2f})" for c, a, t in zip(df_top["codon"], df_top["aa"], df_top["tai"])]
ax1.barh(range(len(df_top)), df_top["delta_g05_minus_g0"], color=colors, alpha=0.85)
ax1.set_yticks(range(len(df_top)))
ax1.set_yticklabels(bar_labels, fontsize=8)
ax1.axvline(0, color='black', lw=0.5)
ax1.set_xlabel("Δ frequency (γ=0.5 − γ=0.0)")
ax1.set_title(f"{GENE}: Top 20 codons with biggest usage shift between γ=0.5 and γ=0.0\n"
              f"Green = γ=0.5 uses MORE   |   Red = γ=0.0 uses MORE", fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Panel 2: TAI profile along sequence
ax2 = fig.add_subplot(gs[1, :])
positions = np.arange(n_codons)
ax2.plot(positions, np.convolve(tai_g0, np.ones(15)/15, mode='same'),
         label=f"γ=0.0 (mean={tai_g0.mean():.2f})", color='#1f77b4', lw=1.2)
ax2.plot(positions, np.convolve(tai_g05, np.ones(15)/15, mode='same'),
         label=f"γ=0.5 (mean={tai_g05.mean():.2f})", color='#d62728', lw=1.2)
# Mark aggregation hotspots
for pos in agg_hot_positions:
    ax2.axvspan(pos - 0.4, pos + 0.4, color='orange', alpha=0.15)
ax2.set_xlabel("Codon position")
ax2.set_ylabel("TAI (15-codon running mean)")
ax2.set_title("TAI profile along CDS  (orange shading = aggregation hotspots)", fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, n_codons)

# Panel 3: Hydrophobicity overlay
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(positions, hydro, color='gray', lw=1, label='Hydrophobicity (KD-9)')
ax3.fill_between(positions, hydro, 0, where=(hydro < -1.0), color='blue', alpha=0.25, label='hydrophilic stretch')
ax3.fill_between(positions, hydro, 0, where=(hydro > 1.5),  color='orange', alpha=0.25, label='hydrophobic stretch')
ax3.axhline(0, color='black', lw=0.4)
ax3.set_xlabel("Codon position")
ax3.set_ylabel("KD hydrophobicity")
ax3.set_title("Hydrophobicity profile (negative = hydrophilic)", fontsize=10)
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(alpha=0.3)
ax3.set_xlim(0, n_codons)

# Panel 4: AlphaFold pLDDT (left) and contact density (right)
if plddt_arr is not None:
    ax4a = fig.add_subplot(gs[3, 0])
    ax4a.plot(positions, plddt_arr, color='purple', lw=1)
    ax4a.fill_between(positions, plddt_arr, 0, color='purple', alpha=0.2)
    ax4a.axhline(70, color='red', ls='--', lw=0.6, label='pLDDT=70 (confident)')
    ax4a.set_xlabel("Codon position")
    ax4a.set_ylabel("AlphaFold pLDDT")
    ax4a.set_title("Per-residue confidence", fontsize=10)
    ax4a.legend(fontsize=8)
    ax4a.grid(alpha=0.3)
    ax4a.set_xlim(0, n_codons)

    ax4b = fig.add_subplot(gs[3, 1])
    ax4b.plot(positions, contact_arr, color='teal', lw=1)
    ax4b.fill_between(positions, contact_arr, 0, color='teal', alpha=0.25)
    ax4b.set_xlabel("Codon position")
    ax4b.set_ylabel("# Cα contacts (8Å, sep≥6)")
    ax4b.set_title("Protein contact density (structural compactness)", fontsize=10)
    ax4b.grid(alpha=0.3)
    ax4b.set_xlim(0, n_codons)

# Panel 5: TAI difference per position + folding_demand (if available)
ax5 = fig.add_subplot(gs[4, :])
tai_diff = tai_g05 - tai_g0
smoothed = np.convolve(tai_diff, np.ones(11)/11, mode='same')
ax5.plot(positions, smoothed, color='#9467bd', lw=1.2, label='TAI(γ=0.5) − TAI(γ=0.0)')
ax5.axhline(0, color='black', lw=0.4)
ax5.fill_between(positions, smoothed, 0, where=(smoothed < 0), color='#d62728', alpha=0.25, label='γ=0.5 SLOWER here')
ax5.fill_between(positions, smoothed, 0, where=(smoothed > 0), color='#2ca02c', alpha=0.25, label='γ=0.5 FASTER here')
if fd_wt is not None:
    ax5b = ax5.twinx()
    ax5b.plot(positions, fd_wt, color='black', lw=0.7, alpha=0.5, label='folding_demand (WT)')
    ax5b.set_ylabel("folding_demand (WT)", color='black')
    ax5b.tick_params(axis='y', labelcolor='black')
    ax5b.legend(loc='upper right', fontsize=8)
ax5.set_xlabel("Codon position")
ax5.set_ylabel("Δ TAI (smoothed)")
ax5.set_title("Where γ=0.5 differs from γ=0.0  (red = γ=0.5 is slower, expected to align with high folding_demand)", fontsize=10)
ax5.legend(loc='upper left', fontsize=8)
ax5.grid(alpha=0.3)
ax5.set_xlim(0, n_codons)

# Title
fig.suptitle(f"{GENE} — Optimization Comparison: γ=0.0 vs γ=0.5 (mfe_weight=0.7, 5 epochs)",
             fontsize=13, y=0.995)

# Save
out_fig = OUT_DIR / f"{GENE}_g0_vs_g05_comparison.png"
plt.savefig(out_fig, dpi=160, bbox_inches='tight')
plt.close()
print(f"[Saved] {out_fig}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. SECOND FIGURE: TAI scatter at aggregation/hydrophilic regions
# ═══════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) TAI vs hydrophobicity
ax = axes[0]
ax.scatter(hydro, tai_g0,  alpha=0.3, s=10, color='#1f77b4', label=f'γ=0.0 (ρ={rho_g0:+.2f})')
ax.scatter(hydro, tai_g05, alpha=0.3, s=10, color='#d62728', label=f'γ=0.5 (ρ={rho_g05:+.2f})')
ax.axvline(0, color='black', lw=0.4)
ax.axvline(-1, color='blue', ls='--', lw=0.4, alpha=0.5)
ax.set_xlabel("Hydrophobicity (KD-9)")
ax.set_ylabel("TAI")
ax.set_title("TAI vs Hydrophobicity")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) TAI vs contact density
if contact_arr is not None:
    ax = axes[1]
    ax.scatter(contact_arr, tai_g0,  alpha=0.3, s=10, color='#1f77b4', label=f'γ=0.0 (ρ={rho_g0_cd:+.2f})')
    ax.scatter(contact_arr, tai_g05, alpha=0.3, s=10, color='#d62728', label=f'γ=0.5 (ρ={rho_g05_cd:+.2f})')
    ax.set_xlabel("# Cα contacts (8Å, sep≥6)")
    ax.set_ylabel("TAI")
    ax.set_title("TAI vs contact density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# (c) Epoch trajectories
ax = axes[2]
ax.plot(eps, rpf_g0,  marker='o', color='#1f77b4', label='RPF γ=0.0')
ax.plot(eps, rpf_g05, marker='o', color='#d62728', label='RPF γ=0.5')
ax.set_xlabel("Epoch")
ax.set_ylabel("RPF (model)", color='black')
ax.tick_params(axis='y', labelcolor='black')
ax.legend(loc='upper left', fontsize=8)
ax2 = ax.twinx()
ax2.plot(eps, mfe_g0,  marker='s', color='#1f77b4', linestyle=':', alpha=0.6, label='MFE γ=0.0')
ax2.plot(eps, mfe_g05, marker='s', color='#d62728', linestyle=':', alpha=0.6, label='MFE γ=0.5')
ax2.set_ylabel("MFE (RNAfold tool)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.legend(loc='lower right', fontsize=8)
ax.set_title("Optimization trajectory")
ax.grid(alpha=0.3)

fig2.suptitle(f"{GENE} — Codon usage vs structural context", fontsize=12)
plt.tight_layout()
out_fig2 = OUT_DIR / f"{GENE}_g0_vs_g05_correlations.png"
plt.savefig(out_fig2, dpi=160, bbox_inches='tight')
plt.close()
print(f"[Saved] {out_fig2}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY TEXT FILE
# ═══════════════════════════════════════════════════════════════════════════════

summary_path = OUT_DIR / f"{GENE}_summary.txt"
with open(summary_path, 'w') as f:
    f.write(f"ACTB Optimization Comparison — Summary\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"γ=0.0 (no fold penalty) vs γ=0.5 (medium fold penalty)\n")
    f.write(f"Both with mfe_weight=0.7, 5 epochs, HEK293T\n\n")
    f.write(f"Sequence stats:\n")
    f.write(f"  Length: {n_codons} codons\n")
    f.write(f"  Codon differences (γ=0.5 vs γ=0.0): {n_diff} ({100*n_diff/n_codons:.1f}%)\n\n")
    f.write(f"Mean TAI:\n")
    f.write(f"  WT:    {tai_wt.mean():.3f}\n")
    f.write(f"  γ=0.0: {tai_g0.mean():.3f}\n")
    f.write(f"  γ=0.5: {tai_g05.mean():.3f}\n\n")
    f.write(f"Final RPF (higher = more translation):\n")
    f.write(f"  γ=0.0: {run_g0['rpf']:.2f}\n")
    f.write(f"  γ=0.5: {run_g05['rpf']:.2f}\n\n")
    f.write(f"Final MFE (RNAfold, more negative = more structured):\n")
    f.write(f"  γ=0.0: {run_g0['mfe_tool']:.2f}\n")
    f.write(f"  γ=0.5: {run_g05['mfe_tool']:.2f}\n\n")
    f.write(f"Spearman correlations (γ=0 vs γ=0.5):\n")
    f.write(f"  ρ(TAI, hydrophobicity):     γ=0={rho_g0:+.3f}  γ=0.5={rho_g05:+.3f}\n")
    if contact_arr is not None:
        f.write(f"  ρ(TAI, contact_density):    γ=0={rho_g0_cd:+.3f}  γ=0.5={rho_g05_cd:+.3f}\n")
    if fd_wt is not None:
        f.write(f"  ρ(TAI, folding_demand):     γ=0={rho_g0_fd:+.3f}  γ=0.5={rho_g05_fd:+.3f}\n")
    f.write(f"\nInterpretation:\n")
    f.write(f"  - If γ=0.5 places slow codons at high-folding-demand positions,\n")
    f.write(f"    ρ(TAI, folding_demand) should be MORE NEGATIVE for γ=0.5.\n")
    f.write(f"  - Same logic for contact density (compact regions should slow translation).\n")
    f.write(f"  - For hydrophobicity, slow codons before hydrophobic stretches help SRP\n")
    f.write(f"    targeting (positive ρ between TAI and hydrophobicity expected).\n")

print(f"[Saved] {summary_path}")
print(f"\n{'='*70}")
print(f"All outputs in: {OUT_DIR}")
print(f"{'='*70}")