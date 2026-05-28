#!/usr/bin/env python3
"""
evaluate_sweep.py

Full evaluation pipeline implementing the methodology described in
Section "Evaluation" of the thesis.

Per-gene metrics:
  1. Spearman ρ(τ_z, feature) for hydropathy, contact density, folding_demand
  2. Permutation test AT functional regions (3 categories)
  3. Permutation test BEFORE functional regions (5-codon window)

Cross-gene aggregation:
  4. Wilcoxon signed-rank on Δρ
  5. Hodges-Lehmann pseudo-medians
  6. Sign tests
  7. Benjamini-Hochberg FDR correction

Stratification:
  - By functional category (housekeeping / multi-domain / therapeutic)
  - By contact density (low / high) - controls for the hypothesis that
    contact density saturates folding_demand and just slows everything

Functional regions:
  - Hydrophilic stretches (Kyte-Doolittle smoothed window=9, < -1.0, ≥5 codons)
  - Aggregation hotspots (TANGO-like: max(KD,0) * β-propensity, top 15%)
  - Domain boundaries (max of domain_boundary_pae, _interpro from CSV)

Usage:
    python evaluate_sweep.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, wilcoxon, binomtest, mannwhitneyu
from itertools import combinations
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL FONT SIZES (thesis-ready)
# ═══════════════════════════════════════════════════════════════════════════════

FONT_BASE   = 16   # axis tick labels, legend, annotations
FONT_LABEL  = 16   # x/y axis labels
FONT_TITLE  = 17   # subplot titles
FONT_SUPTITLE = 19 # figure-level super-title
FONT_ANNOT  = 11   # per-gene text annotations in scatter plots

mpl.rcParams.update({
    'font.size':        FONT_BASE,
    'axes.titlesize':   FONT_TITLE,
    'axes.labelsize':   FONT_LABEL,
    'xtick.labelsize':  FONT_BASE,
    'ytick.labelsize':  FONT_BASE,
    'legend.fontsize':  FONT_BASE,
    'figure.titlesize': FONT_SUPTITLE,
})

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[2]
OPT_DIR = BASE / "data/optimized"
PDB_DIR = BASE / "data/alphafold_results"
CSV_DIR = BASE / "data/ribo_counts"
OUT_DIR = BASE / "data/analysis/sweep_evaluation_before_15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Comparison: γ=0.0 (baseline) vs γ=0.5 (with fold penalty)
GAMMA_LOW  = "fold000"
GAMMA_HIGH = "fold050"
MFE_TAG    = "mfe070"
EP_TAG     = "ep5"

# Permutation test settings
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

# Hotspot definitions
HYDROPHILIC_THRESHOLD = -1
HYDROPHILIC_MIN_LEN   = 5
AGGREGATION_PERCENTILE = 85
DOMAIN_BOUNDARY_THRESHOLD = 0.5
BEFORE_WINDOW = 15

# ═══════════════════════════════════════════════════════════════════════════════
# GENE CATEGORIZATION
# ═══════════════════════════════════════════════════════════════════════════════

GENE_CATEGORIES = {
    'housekeeping': ['ACTB', 'CCT3', 'HSPD1', 'PFKM', 'PKM',
                     'LMNA', 'LMNB1', 'MAPK1', 'MAPK3', 'NSF'],
    'multi_domain': ['EGFR', 'CASP7', 'CTSB', 'IDH2'],
    'therapeutic':  ['IFNB1', 'FGA', 'HBB', 'PROC', 'TTR'],
}
# Reverse lookup
CATEGORY_OF = {gene: cat for cat, genes in GENE_CATEGORIES.items() for gene in genes}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — codon tables
# ═══════════════════════════════════════════════════════════════════════════════

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
TAI = {
    'TTT':0.42,'TTC':1.00,'TTA':0.08,'TTG':0.42,'CTT':0.42,'CTC':0.58,'CTA':0.08,'CTG':1.00,
    'ATT':0.58,'ATC':1.00,'ATA':0.08,'ATG':1.00,'GTT':0.42,'GTC':0.58,'GTA':0.08,'GTG':1.00,
    'TCT':0.58,'TCC':0.75,'TCA':0.25,'TCG':0.17,'CCT':0.58,'CCC':0.75,'CCA':0.42,'CCG':0.17,
    'ACT':0.58,'ACC':1.00,'ACA':0.42,'ACG':0.17,'GCT':0.75,'GCC':1.00,'GCA':0.42,'GCG':0.17,
    'TAT':0.42,'TAC':1.00,'TAA':0.00,'TAG':0.00,'CAT':0.42,'CAC':1.00,'CAA':0.42,'CAG':1.00,
    'AAT':0.42,'AAC':1.00,'AAA':0.42,'AAG':1.00,'GAT':0.42,'GAC':1.00,'GAA':0.42,'GAG':1.00,
    'TGT':0.42,'TGC':1.00,'TGA':0.00,'TGG':1.00,'CGT':0.42,'CGC':0.75,'CGA':0.08,'CGG':0.17,
    'AGT':0.25,'AGC':0.75,'AGA':0.42,'AGG':0.25,'GGT':0.42,'GGC':1.00,'GGA':0.25,'GGG':0.17,
}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'*':0.0}
# Chou-Fasman β-sheet propensity
BETA_PROP = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'E':0.37,'Q':1.10,'G':0.75,
             'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,
             'T':1.19,'W':1.37,'Y':1.47,'V':1.70,'*':0.0}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def codons_of(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def aa_seq(codons):
    return "".join(CODON_TABLE.get(c, "?") for c in codons)

def tai_profile(codons):
    return np.array([TAI.get(c, 0.5) for c in codons])

def zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 0 else x - np.mean(x)

def hydro_profile(aa, window=9):
    kd = np.array([KD.get(a, 0.0) for a in aa])
    if len(kd) < window:
        return kd
    return np.convolve(kd, np.ones(window)/window, mode='same')

def aggregation_profile(aa, window=7):
    h = np.array([KD.get(a, 0.0) for a in aa])
    b = np.array([BETA_PROP.get(a, 1.0) for a in aa])
    score = np.maximum(h, 0) * b
    return np.convolve(score, np.ones(window)/window, mode='same')

# ─── PDB handling ─────────────────────────────────────────────────────────────

def parse_pdb(pdb_path):
    plddt, ca = {}, {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            r = int(line[22:26].strip()) - 1
            plddt[r] = float(line[60:66])
            ca[r] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return plddt, ca

def find_pdb(gene):
    g_dir = PDB_DIR / gene
    if not g_dir.exists():
        return None
    pdbs = list(g_dir.glob("*.pdb"))
    if not pdbs:
        return None
    rank1 = [p for p in pdbs if "rank_001" in p.name or "rank_1" in p.name]
    if rank1:
        return rank1[0]
    m1 = [p for p in pdbs if "model_1" in p.name]
    return m1[0] if m1 else pdbs[0]

def contact_density(ca, n_res, threshold=8.0, sep=6):
    cd = np.zeros(n_res)
    keys = sorted(k for k in ca.keys() if k < n_res)
    for i, ri in enumerate(keys):
        for rj in keys[i+1:]:
            if abs(ri - rj) < sep:
                continue
            d = np.linalg.norm(ca[ri] - ca[rj])
            if d < threshold:
                cd[ri] += 1
                cd[rj] += 1
    return cd

# ─── WT CSV ───────────────────────────────────────────────────────────────────

def find_wt_csv(gene):
    g_lower = gene.lower()
    for p in CSV_DIR.glob("*_with_folddemand.csv"):
        base = p.name.split("_")[0]
        if base.lower() == g_lower:
            return p
    return None

def load_wt_features(gene, n_codons):
    csv = find_wt_csv(gene)
    if not csv:
        return None, None
    df = pd.read_csv(csv)
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    fd = None
    if 'folding_demand' in cds.columns:
        fd = cds['folding_demand'].values[:n_codons]
        if len(fd) < n_codons:
            fd = np.pad(fd, (0, n_codons - len(fd)), constant_values=np.nan)
    boundary = None
    cols = ['domain_boundary', 'domain_boundary_interpro', 'domain_boundary_pae']
    avail = [c for c in cols if c in cds.columns]
    if avail:
        b = np.zeros(n_codons)
        for col in avail:
            v = cds[col].fillna(0).values[:n_codons]
            if len(v) < n_codons:
                v = np.pad(v, (0, n_codons - len(v)), constant_values=0)
            b = np.maximum(b, v)
        boundary = b
    return fd, boundary

# ─── Hotspot detection ────────────────────────────────────────────────────────

def hydrophilic_positions(hydro, threshold=HYDROPHILIC_THRESHOLD, min_len=HYDROPHILIC_MIN_LEN):
    in_s = False; start = 0; pos = []
    for i, v in enumerate(hydro):
        if v < threshold and not in_s:
            in_s = True; start = i
        elif v >= threshold and in_s:
            in_s = False
            if i - start >= min_len:
                pos.extend(range(start, i))
    if in_s and len(hydro) - start >= min_len:
        pos.extend(range(start, len(hydro)))
    return np.array(pos, dtype=int)

def aggregation_hotspot_positions(agg_score, percentile=AGGREGATION_PERCENTILE):
    threshold = np.percentile(agg_score, percentile)
    return np.where(agg_score > threshold)[0]

def domain_boundary_positions(boundary, threshold=DOMAIN_BOUNDARY_THRESHOLD):
    if boundary is None:
        return np.array([], dtype=int)
    return np.where(boundary > threshold)[0]

# ─── Permutation tests ────────────────────────────────────────────────────────

def perm_test_at(values, hot_pos, n_perm=N_PERMUTATIONS, rng=None, alternative='less'):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n = len(values)
    if len(hot_pos) == 0 or len(hot_pos) == n:
        return np.nan, np.nan
    hot_mask = np.zeros(n, dtype=bool); hot_mask[hot_pos] = True
    obs = values[hot_mask].mean() - values[~hot_mask].mean()
    n_hot = hot_mask.sum()
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(n, n_hot, replace=False)
        m = np.zeros(n, dtype=bool); m[idx] = True
        null[i] = values[m].mean() - values[~m].mean()
    if alternative == 'less':
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    elif alternative == 'greater':
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)
    return obs, p

def before_window_mean(values, hot_pos, window=BEFORE_WINDOW):
    if len(hot_pos) == 0:
        return np.nan
    before = set()
    for p in hot_pos:
        for d in range(1, window+1):
            if p - d >= 0:
                before.add(p - d)
    if not before:
        return np.nan
    return values[sorted(before)].mean()

def perm_test_before(values, hot_pos, window=BEFORE_WINDOW,
                     n_perm=N_PERMUTATIONS, rng=None, alternative='less'):
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n = len(values)
    if len(hot_pos) == 0:
        return np.nan, np.nan
    obs = before_window_mean(values, hot_pos, window)
    if np.isnan(obs):
        return np.nan, np.nan
    n_hot = len(hot_pos)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(n, n_hot, replace=False)
        null[i] = before_window_mean(values, idx, window)
    null = null[~np.isnan(null)]
    if len(null) == 0:
        return obs, np.nan
    if alternative == 'less':
        p = (np.sum(null <= obs) + 1) / (len(null) + 1)
    else:
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    return obs, p

# ─── Aggregation tests ────────────────────────────────────────────────────────

def hodges_lehmann(diffs):
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return np.nanmedian(diffs) if len(diffs) > 0 else np.nan
    pairs = [(a + b) / 2 for a, b in combinations(diffs, 2)]
    return np.median(pairs)

def cohen_r_wilcoxon(W, n):
    """Approximate Cohen's r effect size from Wilcoxon W statistic."""
    if n < 4:
        return np.nan
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2*n + 1) / 24
    if var_W == 0:
        return np.nan
    Z = (W - mean_W) / np.sqrt(var_W)
    return abs(Z) / np.sqrt(n)

def aggregate(diffs, label, alternative='less'):
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return {"label": label, "n": n, "median": np.nan, "hl": np.nan,
                "wilcoxon_p": np.nan, "cohen_r": np.nan,
                "n_neg": np.nan, "n_pos": np.nan, "sign_p": np.nan}
    n_neg = int((d < 0).sum())
    n_pos = int((d > 0).sum())
    try:
        W, p_w = wilcoxon(d, alternative=alternative)
    except Exception:
        W, p_w = np.nan, np.nan
    r = cohen_r_wilcoxon(W, n) if not np.isnan(W) else np.nan
    # Sign test (one-sided: more negative than expected)
    if alternative == 'less':
        sign_p = binomtest(n_neg, n_neg + n_pos, p=0.5, alternative='greater').pvalue if (n_neg + n_pos) > 0 else np.nan
    else:
        sign_p = binomtest(n_pos, n_neg + n_pos, p=0.5, alternative='greater').pvalue if (n_neg + n_pos) > 0 else np.nan
    return {
        "label": label, "n": n,
        "median": float(np.median(d)),
        "hl": float(hodges_lehmann(d)),
        "wilcoxon_p": float(p_w) if not np.isnan(p_w) else np.nan,
        "cohen_r": float(r) if not np.isnan(r) else np.nan,
        "n_neg": n_neg, "n_pos": n_pos,
        "sign_p": float(sign_p) if not np.isnan(sign_p) else np.nan,
    }

def fdr_bh(pvalues):
    """Benjamini-Hochberg FDR correction."""
    p = np.array([pv if not np.isnan(pv) else 1.0 for pv in pvalues])
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n+1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# PER-GENE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_gene(gene, json_low, json_high):
    with open(json_low) as f:  d_low = json.load(f)
    with open(json_high) as f: d_high = json.load(f)
    n_ep = d_low["config"]["n_epochs"]
    
    def get_seq(d):
        n = d["config"]["n_epochs"]
        try:
            return d["sequence"]["best_seq"][str(n)]
        except (KeyError, TypeError):
            try:
                return d["metrics_per_epoch"]["best_seq"][str(n)]
            except (KeyError, TypeError):
                return d["sequence"]["cds_only"]
    
    seq_low, seq_high = get_seq(d_low), get_seq(d_high)
    c_low, c_high = codons_of(seq_low), codons_of(seq_high)
    if len(c_low) != len(c_high):
        return None
    n = len(c_low)
    aa = aa_seq(c_low)
    
    tai_low = tai_profile(c_low)
    tai_high = tai_profile(c_high)
    tai_low_z = zscore(tai_low)
    tai_high_z = zscore(tai_high)
    
    hydro = hydro_profile(aa, window=9)
    agg = aggregation_profile(aa, window=7)
    fd, boundary = load_wt_features(gene, n)
    
    pdb = find_pdb(gene)
    contacts = None; plddt = None
    if pdb:
        plddt_d, ca_d = parse_pdb(pdb)
        plddt = np.array([plddt_d.get(i, 0.0) for i in range(n)])
        contacts = contact_density(ca_d, n)
    
    out = {
        "gene": gene,
        "category": CATEGORY_OF.get(gene, 'unknown'),
        "n_codons": n,
        "n_codons_diff": int(sum(1 for a, b in zip(c_low, c_high) if a != b)),
        "tai_mean_low": float(tai_low.mean()),
        "tai_mean_high": float(tai_high.mean()),
        "tai_mean_diff": float(tai_high.mean() - tai_low.mean()),
        "rpf_low": float(d_low["final"]["rpf_model"]),
        "rpf_high": float(d_high["final"]["rpf_model"]),
        "mfe_low": float(d_low["final"]["mfe_tool"]),
        "mfe_high": float(d_high["final"]["mfe_tool"]),
        "mean_contact_density": float(contacts.mean()) if contacts is not None else np.nan,
        "mean_plddt": float(plddt[plddt > 0].mean()) if plddt is not None and (plddt > 0).any() else np.nan,
        "mean_folding_demand": float(np.nanmean(fd)) if fd is not None else np.nan,
    }
    
    def corr(x, y):
        if y is None or np.all(np.isnan(y)):
            return np.nan
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            return np.nan
        rho, _ = spearmanr(x[mask], y[mask])
        return rho
    
    # Correlations on z-scored TAI (controls for global shift)
    out["rho_low_hydro_z"]    = corr(tai_low_z, hydro)
    out["rho_high_hydro_z"]   = corr(tai_high_z, hydro)
    out["rho_low_contacts_z"] = corr(tai_low_z, contacts) if contacts is not None else np.nan
    out["rho_high_contacts_z"]= corr(tai_high_z, contacts) if contacts is not None else np.nan
    out["rho_low_fd_z"]       = corr(tai_low_z, fd)
    out["rho_high_fd_z"]      = corr(tai_high_z, fd)
    
    # Permutation tests at functional regions (use z-scored TAI)
    hyd_pos = hydrophilic_positions(hydro)
    agg_pos = aggregation_hotspot_positions(agg)
    bnd_pos = domain_boundary_positions(boundary)
    
    for region_name, region_pos in [("hydrophilic", hyd_pos),
                                      ("aggregation", agg_pos),
                                      ("boundary", bnd_pos)]:
        out[f"{region_name}_n"] = len(region_pos)
        if len(region_pos) == 0:
            for kind in ["at_low", "at_high", "before_low", "before_high"]:
                out[f"{region_name}_{kind}_delta"] = np.nan
                out[f"{region_name}_{kind}_p"] = np.nan
            continue
        rng_seed = RANDOM_SEED + hash(gene + region_name) % 1000
        d_at_lo, p_at_lo = perm_test_at(tai_low_z, region_pos, rng=np.random.default_rng(rng_seed))
        d_at_hi, p_at_hi = perm_test_at(tai_high_z, region_pos, rng=np.random.default_rng(rng_seed))
        d_be_lo, p_be_lo = perm_test_before(tai_low_z, region_pos, rng=np.random.default_rng(rng_seed))
        d_be_hi, p_be_hi = perm_test_before(tai_high_z, region_pos, rng=np.random.default_rng(rng_seed))
        out[f"{region_name}_at_low_delta"]   = d_at_lo
        out[f"{region_name}_at_low_p"]       = p_at_lo
        out[f"{region_name}_at_high_delta"]  = d_at_hi
        out[f"{region_name}_at_high_p"]      = p_at_hi
        out[f"{region_name}_before_low_delta"]  = d_be_lo
        out[f"{region_name}_before_low_p"]      = p_be_lo
        out[f"{region_name}_before_high_delta"] = d_be_hi
        out[f"{region_name}_before_high_p"]     = p_be_hi
    
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION ACROSS GENES (with stratification)
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_aggregation(df, group_col, group_values, value_col_low, value_col_high, label, alternative='less'):
    """Run aggregate test for each subgroup."""
    results = []
    for g in group_values:
        subset = df[df[group_col] == g]
        diffs = subset[value_col_high].values - subset[value_col_low].values
        r = aggregate(diffs, f"{label} [{g}]", alternative=alternative)
        results.append(r)
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Find paired runs
    all_jsons = list(OPT_DIR.glob(f"*_{MFE_TAG}_HEK293T_{EP_TAG}_results.json"))
    by_gene_low = {}
    by_gene_high = {}
    for p in all_jsons:
        gene = p.stem.split("_")[0]
        if f"_{GAMMA_LOW}_" in p.name:
            by_gene_low[gene] = p
        elif f"_{GAMMA_HIGH}_" in p.name:
            by_gene_high[gene] = p
    
    common = sorted(set(by_gene_low.keys()) & set(by_gene_high.keys()))
    print(f"\n{'='*72}")
    print(f"  Sweep Evaluation: γ={GAMMA_LOW} vs γ={GAMMA_HIGH}")
    print(f"  ({MFE_TAG}, {EP_TAG})")
    print(f"{'='*72}\n")
    print(f"Genes with both runs: {len(common)}")
    print(f"  {', '.join(common)}\n")
    
    if len(common) < 3:
        print("Not enough gene pairs for cross-gene tests.")
        return
    
    # 2. Per-gene analysis
    results = []
    print("Per-gene analysis:")
    for gene in common:
        try:
            r = analyze_gene(gene, by_gene_low[gene], by_gene_high[gene])
            if r is not None:
                results.append(r)
                cat = r['category']
                print(f"  {gene:8s} [{cat:12s}]  n={r['n_codons']}  "
                      f"diff={r['n_codons_diff']} ({100*r['n_codons_diff']/r['n_codons']:.0f}%)  "
                      f"ΔTAI={r['tai_mean_diff']:+.3f}")
        except Exception as e:
            print(f"  {gene}: ERROR — {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "per_gene_metrics.csv", index=False)
    print(f"\n[Saved] {OUT_DIR / 'per_gene_metrics.csv'}\n")
    
    # 3. Density stratification: split by mean_contact_density (median)
    if 'mean_contact_density' in df.columns and df['mean_contact_density'].notna().sum() > 3:
        median_cd = df['mean_contact_density'].median()
        df['density_class'] = df['mean_contact_density'].apply(
            lambda x: 'low_density' if x < median_cd else 'high_density'
        )
        print(f"Density split: median contact_density = {median_cd:.2f}")
        print(f"  Low density genes:  {df[df.density_class=='low_density']['gene'].tolist()}")
        print(f"  High density genes: {df[df.density_class=='high_density']['gene'].tolist()}\n")
    
    # 4. Aggregate analysis
    print(f"{'─'*72}")
    print(f"  CROSS-GENE AGGREGATE TESTS (overall, N={len(df)})")
    print(f"{'─'*72}\n")
    
    correlation_metrics = [
        ("rho_low_hydro_z",    "rho_high_hydro_z",    "ρ(τ_z, hydropathy)"),
        ("rho_low_contacts_z", "rho_high_contacts_z", "ρ(τ_z, contact_density)"),
        ("rho_low_fd_z",       "rho_high_fd_z",       "ρ(τ_z, folding_demand)"),
    ]
    region_metrics = []
    for region in ["hydrophilic", "aggregation", "boundary"]:
        for kind, ls in [("at", "AT"), ("before", "BEFORE")]:
            region_metrics.append((
                f"{region}_{kind}_low_delta",
                f"{region}_{kind}_high_delta",
                f"τ_z {ls} {region}",
            ))
    
    overall_results = []
    for low_col, high_col, label in correlation_metrics:
        diffs = df[high_col].values - df[low_col].values
        r = aggregate(diffs, label, alternative='less')
        overall_results.append(r)
        print(f"  {label:32s}  N={r['n']:2d}  "
              f"HL={r['hl']:+.3f}  W-p={r['wilcoxon_p']:.4f}  "
              f"signs={r['n_neg']}/{r['n_pos']}  sign-p={r['sign_p']:.4f}  "
              f"r={r['cohen_r']:.2f}")
    
    print()
    for low_col, high_col, label in region_metrics:
        if low_col not in df.columns:
            continue
        diffs = df[high_col].values - df[low_col].values
        r = aggregate(diffs, label, alternative='less')
        overall_results.append(r)
        print(f"  {label:32s}  N={r['n']:2d}  "
              f"HL={r['hl']:+.3f}  W-p={r['wilcoxon_p']:.4f}  "
              f"signs={r['n_neg']}/{r['n_pos']}  sign-p={r['sign_p']:.4f}  "
              f"r={r['cohen_r']:.2f}")
    
    overall_df = pd.DataFrame(overall_results)
    # FDR correction across the cross-gene tests
    overall_df['wilcoxon_p_fdr'] = fdr_bh(overall_df['wilcoxon_p'].values)
    overall_df.to_csv(OUT_DIR / "aggregate_overall.csv", index=False)
    print(f"\n[Saved] {OUT_DIR / 'aggregate_overall.csv'}")
    
    # 5. Stratified by category
    print(f"\n{'─'*72}")
    print(f"  STRATIFIED BY FUNCTIONAL CATEGORY")
    print(f"{'─'*72}\n")
    
    cat_rows = []
    categories = sorted(df['category'].unique())
    for cat in categories:
        subset = df[df['category'] == cat]
        if len(subset) < 2:
            continue
        print(f"  ── {cat} (N={len(subset)}, genes: {', '.join(subset['gene'].tolist())}) ──")
        for low_col, high_col, label in correlation_metrics + region_metrics:
            if low_col not in subset.columns:
                continue
            diffs = subset[high_col].values - subset[low_col].values
            r = aggregate(diffs, f"{label} [{cat}]", alternative='less')
            cat_rows.append({**r, "category": cat})
            print(f"    {label:30s}  N={r['n']:2d}  HL={r['hl']:+.3f}  "
                  f"W-p={r['wilcoxon_p']:.3f}  signs={r['n_neg']}/{r['n_pos']}")
        print()
    
    cat_df = pd.DataFrame(cat_rows)
    if len(cat_df) > 0:
        cat_df.to_csv(OUT_DIR / "aggregate_by_category.csv", index=False)
        print(f"[Saved] {OUT_DIR / 'aggregate_by_category.csv'}")
    
    # 6. Stratified by density
    if 'density_class' in df.columns:
        print(f"\n{'─'*72}")
        print(f"  STRATIFIED BY CONTACT DENSITY")
        print(f"{'─'*72}\n")
        
        dens_rows = []
        for dens in ['low_density', 'high_density']:
            subset = df[df['density_class'] == dens]
            if len(subset) < 2:
                continue
            print(f"  ── {dens} (N={len(subset)}, mean CD={subset['mean_contact_density'].mean():.2f}) ──")
            for low_col, high_col, label in correlation_metrics + region_metrics:
                if low_col not in subset.columns:
                    continue
                diffs = subset[high_col].values - subset[low_col].values
                r = aggregate(diffs, f"{label} [{dens}]", alternative='less')
                dens_rows.append({**r, "density_class": dens})
                print(f"    {label:30s}  N={r['n']:2d}  HL={r['hl']:+.3f}  "
                      f"W-p={r['wilcoxon_p']:.3f}  signs={r['n_neg']}/{r['n_pos']}")
            print()
        
        dens_df = pd.DataFrame(dens_rows)
        if len(dens_df) > 0:
            dens_df.to_csv(OUT_DIR / "aggregate_by_density.csv", index=False)
            print(f"[Saved] {OUT_DIR / 'aggregate_by_density.csv'}")
    
    # 7. Plots
    print(f"\n{'─'*72}\n  PLOTTING\n{'─'*72}\n")
    
    # Plot 1: Per-gene Δρ forest plot (color by category)
    fig, axes = plt.subplots(1, 3, figsize=(20, max(7, 0.5*len(df))), sharey=True)
    cat_colors = {'housekeeping': '#1f77b4', 'multi_domain': '#d62728', 'therapeutic': '#2ca02c', 'unknown': '#7f7f7f'}
    for ax, (low_col, high_col, label) in zip(axes, correlation_metrics):
        deltas = df[high_col].values - df[low_col].values
        order = np.argsort(deltas)
        deltas_o = deltas[order]
        genes_o = df["gene"].values[order]
        cats_o = df["category"].values[order]
        colors = [cat_colors[c] for c in cats_o]
        ax.barh(range(len(deltas_o)), deltas_o, color=colors, alpha=0.85)
        ax.set_yticks(range(len(deltas_o)))
        ax.set_yticklabels(genes_o, fontsize=FONT_BASE)
        ax.axvline(0, color='black', lw=0.8)
        med = np.nanmedian(deltas)
        ax.axvline(med, color='blue', lw=1.5, ls='--', label=f"median={med:+.3f}")
        ax.set_xlabel(f"Δρ  (γ={GAMMA_HIGH} − γ={GAMMA_LOW})", fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.legend(loc='lower right', fontsize=FONT_BASE)
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=FONT_BASE)
    axes[0].set_ylabel("Gene", fontsize=FONT_LABEL)
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in cat_colors.values()]
    fig.legend(handles, list(cat_colors.keys()),
               loc='upper center', ncol=4, fontsize=FONT_BASE,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Per-gene Δρ forest plot — stratified by functional category",
                 fontsize=FONT_SUPTITLE, y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forest_per_gene_by_category.png", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Saved] forest_per_gene_by_category.png")
    
    # Plot 2: Category comparison boxplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plot_specs = correlation_metrics + region_metrics[:3]   # 3 corr + 3 region (AT)
    for ax, (low_col, high_col, label) in zip(axes.flat, plot_specs):
        if low_col not in df.columns:
            continue
        diffs_per_cat = []
        cat_labels = []
        for cat in categories:
            sub = df[df['category'] == cat]
            d = sub[high_col].values - sub[low_col].values
            d = d[~np.isnan(d)]
            if len(d) > 0:
                diffs_per_cat.append(d)
                cat_labels.append(f"{cat}\nN={len(d)}")
        if not diffs_per_cat:
            continue
        bp = ax.boxplot(diffs_per_cat, labels=cat_labels, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(cat_colors[categories[i]])
            patch.set_alpha(0.7)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_ylabel("Δ value", fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=FONT_BASE)
        ax.tick_params(axis='y', labelsize=FONT_BASE)
    fig.suptitle("Effect by functional category", fontsize=FONT_SUPTITLE)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplots_by_category.png", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Saved] boxplots_by_category.png")
    
    # Plot 3: Density comparison
    if 'density_class' in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        density_colors = {'low_density': '#2ca02c', 'high_density': '#d62728'}
        for ax, (low_col, high_col, label) in zip(axes.flat, plot_specs):
            if low_col not in df.columns:
                continue
            diffs_per_d = []
            d_labels = []
            for dens in ['low_density', 'high_density']:
                sub = df[df['density_class'] == dens]
                d = sub[high_col].values - sub[low_col].values
                d = d[~np.isnan(d)]
                if len(d) > 0:
                    diffs_per_d.append(d)
                    d_labels.append(f"{dens}\nN={len(d)}")
            if not diffs_per_d:
                continue
            bp = ax.boxplot(diffs_per_d, labels=d_labels, patch_artist=True, widths=0.6)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(density_colors[list(density_colors.keys())[i]])
                patch.set_alpha(0.7)
            ax.axhline(0, color='black', lw=0.8, ls='--')
            ax.set_ylabel("Δ value", fontsize=FONT_LABEL)
            ax.set_title(label, fontsize=FONT_TITLE)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', labelsize=FONT_BASE)
            ax.tick_params(axis='y', labelsize=FONT_BASE)
        fig.suptitle("Effect by contact density tier (median split)", fontsize=FONT_SUPTITLE)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "boxplots_by_density.png", dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[Saved] boxplots_by_density.png")
    
    # Plot 4: Density vs effect size (scatter)
    if 'mean_contact_density' in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        for ax, (low_col, high_col, label) in zip(axes.flat, plot_specs):
            if low_col not in df.columns:
                continue
            xs = df['mean_contact_density'].values
            ys = df[high_col].values - df[low_col].values
            mask = ~np.isnan(xs) & ~np.isnan(ys)
            if mask.sum() < 3:
                continue
            colors = [cat_colors[c] for c in df['category'].values[mask]]
            ax.scatter(xs[mask], ys[mask], c=colors, s=100, alpha=0.85,
                       edgecolors='black', linewidths=0.6)
            for i, gene in enumerate(df['gene'].values[mask]):
                ax.annotate(gene, (xs[mask][i], ys[mask][i]),
                            fontsize=FONT_ANNOT, alpha=0.8,
                            xytext=(4, 4), textcoords='offset points')
            ax.axhline(0, color='black', lw=0.8, ls='--')
            # Spearman ρ between density and effect
            rho, p = spearmanr(xs[mask], ys[mask])
            ax.set_title(f"{label}\nρ(density, effect) = {rho:+.2f}  (p = {p:.3f})",
                         fontsize=FONT_TITLE)
            ax.set_xlabel("Mean contact density", fontsize=FONT_LABEL)
            ax.set_ylabel("Δ value", fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_BASE)
            ax.grid(alpha=0.3)
        fig.suptitle(
            "Effect size vs contact density per gene\n"
            "(does fold_penalty effect depend on protein compactness?)",
            fontsize=FONT_SUPTITLE,
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "effect_vs_density.png", dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[Saved] effect_vs_density.png")
    
    # Plot 5: Global TAI shift verification
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (col_low, col_high, title) in zip(axes, [
        ("tai_mean_low", "tai_mean_high", "Global mean TAI"),
        ("rpf_low", "rpf_high", "Final RPF (model)"),
        ("mfe_low", "mfe_high", "Final MFE (RNAfold)"),
    ]):
        for cat in categories:
            sub = df[df['category'] == cat]
            ax.scatter(sub[col_low], sub[col_high], color=cat_colors[cat], s=100, alpha=0.85,
                      edgecolors='black', linewidths=0.6, label=cat)
            for _, row in sub.iterrows():
                ax.annotate(row['gene'], (row[col_low], row[col_high]),
                           fontsize=FONT_ANNOT, alpha=0.8,
                           xytext=(4, 4), textcoords='offset points')
        lim = [df[[col_low, col_high]].min().min(), df[[col_low, col_high]].max().max()]
        pad = (lim[1] - lim[0]) * 0.05
        lim = [lim[0]-pad, lim[1]+pad]
        ax.plot(lim, lim, 'k--', lw=0.9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"γ = {GAMMA_LOW}", fontsize=FONT_LABEL)
        ax.set_ylabel(f"γ = {GAMMA_HIGH}", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.legend(fontsize=FONT_BASE, loc='best')
        ax.tick_params(labelsize=FONT_BASE)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_shifts.png", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[Saved] global_shifts.png")
    
    # 8. Summary text
    summary = OUT_DIR / "evaluation_summary.txt"
    with open(summary, 'w') as f:
        f.write(f"Sweep evaluation summary\n{'='*72}\n\n")
        f.write(f"Comparison: γ={GAMMA_LOW} vs γ={GAMMA_HIGH}, {MFE_TAG}, {EP_TAG}\n")
        f.write(f"N genes: {len(df)}\n\n")
        f.write(f"By category:\n")
        for cat in categories:
            sub = df[df['category'] == cat]
            f.write(f"  {cat:14s}  N={len(sub)}  genes: {', '.join(sub['gene'].tolist())}\n")
        f.write(f"\n{'─'*72}\nGLOBAL SHIFTS (mean across all genes)\n{'─'*72}\n")
        f.write(f"  Mean TAI:  {df['tai_mean_low'].mean():.3f} → {df['tai_mean_high'].mean():.3f} "
                f"(Δ={df['tai_mean_diff'].mean():+.3f})\n")
        f.write(f"  Mean RPF:  {df['rpf_low'].mean():.1f} → {df['rpf_high'].mean():.1f}\n")
        f.write(f"  Mean MFE:  {df['mfe_low'].mean():.1f} → {df['mfe_high'].mean():.1f}\n\n")
        f.write(f"{'─'*72}\nOVERALL CROSS-GENE TESTS (Wilcoxon, alt='less')\n{'─'*72}\n")
        f.write(f"{'metric':32s} {'N':>3s} {'HL':>8s} {'W-p':>8s} {'p-FDR':>8s} {'signs':>10s} {'r':>6s}\n")
        for row in overall_df.itertuples():
            f.write(f"{row.label:32s} {row.n:>3d} {row.hl:>+8.3f} "
                    f"{row.wilcoxon_p:>8.4f} {row.wilcoxon_p_fdr:>8.4f} "
                    f"{row.n_neg}/{row.n_pos:<6d} {row.cohen_r:>6.2f}\n")
        f.write(f"\nMethodology notes:\n")
        f.write(f"  - All correlations on z-scored TAI (controls global shift)\n")
        f.write(f"  - Hydrophilic: KD-window9 < {HYDROPHILIC_THRESHOLD}, ≥{HYDROPHILIC_MIN_LEN} codons\n")
        f.write(f"  - Aggregation: top {100-AGGREGATION_PERCENTILE}% of (max(KD,0) * β-prop)\n")
        f.write(f"  - Domain boundary: max of CSV columns > {DOMAIN_BOUNDARY_THRESHOLD}\n")
        f.write(f"  - Permutations: {N_PERMUTATIONS}\n")
        f.write(f"  - BEFORE window: {BEFORE_WINDOW} codons upstream\n")
        f.write(f"  - One-sided alternative='less' (high γ should have more negative correlations)\n")
        f.write(f"  - FDR: Benjamini-Hochberg across the {len(overall_df)} cross-gene tests\n")
    print(f"\n[Saved] {summary}")
    print(f"\n{'='*72}\nDone. All outputs in: {OUT_DIR}\n{'='*72}")


if __name__ == "__main__":
    main()