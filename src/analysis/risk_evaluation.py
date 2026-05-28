#!/usr/bin/env python3
"""
risk_evaluation.py

Demonstrates that the 18 genes in the test set contain biologically
meaningful misfolding-risk regions, and that these regions co-locate with
the functional regions (hydrophilic, aggregation, boundary) used in the
optimisation evaluation.

This is a DESCRIPTIVE analysis — it shows that:
  (a) every gene has positions of elevated misfolding risk
  (b) these positions are non-uniformly distributed
  (c) they overlap meaningfully with the functional regions tested for τ_z slowdown
  (d) the wild-type codon usage already accommodates these regions

Five independent risk proxies (mostly orthogonal):

  1. Cysteine-density risk: clusters of cysteine residues
     -> wrong disulfide bonds are a major source of misfolding (Anfinsen 1973)
     -> >2 Cys within 30 aa = high mis-pairing risk

  2. Charge-cluster aggregation risk: alternating charged/hydrophobic patches
     -> classical signature of amyloidogenic stretches (Lobanov-Galzitskaya 2010)

  3. Domain-adjacency risk: residues at the junction between two predicted
     AlphaFold domains
     -> domain swapping if the linker is mis-paced (Bennett 1995)

  4. pLDDT-confidence-drop risk: regions where the AlphaFold confidence
     transitions sharply
     -> structurally ambiguous regions

  5. WT codon-conservation risk: positions where the natural (wild-type)
     sequence already uses slow codons
     -> evolutionary signal of folding-critical positions

Outputs per gene:
  - Position-resolved risk scores (5 channels)
  - Composite risk index (mean of normalised channels)
  - Risk-region count
  - Overlap statistics with hydrophilic / aggregation / boundary regions

Aggregate plots:
  - Per-gene risk profile heatmap
  - Distribution of composite risk scores
  - Overlap matrix (risk vs functional region)
  - Correlation: composite risk vs WT TAI (should be negative if WT
    codons already pause at risk sites)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[2]
OPT_DIR = BASE / "data/optimized"
PDB_DIR = BASE / "data/alphafold_results"
CSV_DIR = BASE / "data/ribo_counts"
OUT_DIR = BASE / "data/analysis/risk_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MFE_TAG = "mfe070"
EP_TAG = "ep5"
GAMMA_LOW = "fold000"   # baseline run used to extract WT sequences

# Risk thresholds
CYS_WINDOW = 30
CYS_MIN = 2
CHARGE_WINDOW = 9
PLDDT_TRANSITION_THRESHOLD = 15
RISK_PERCENTILE = 85    # top 15% of composite risk = "risk region"

GENE_CATEGORIES = {
    'housekeeping': ['ACTB', 'CCT3', 'HSPD1', 'PFKM', 'PKM',
                     'LMNA', 'LMNB1', 'MAPK1', 'MAPK3', 'NSF'],
    'multi_domain': ['EGFR', 'CASP7', 'CTSB', 'IDH2'],
    'therapeutic':  ['IFNB1', 'FGA', 'HBB', 'PROC', 'TTR'],
}
CATEGORY_OF = {g: c for c, gs in GENE_CATEGORIES.items() for g in gs}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
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
CHARGE = {'D':-1,'E':-1,'K':+1,'R':+1,'H':+0.5}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def codons_of(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def aa_seq(codons):
    return "".join(CODON_TABLE.get(c, "?") for c in codons)

def tai_profile(codons):
    return np.array([TAI.get(c, 0.5) for c in codons])

def normalize_01(x):
    """Rescale to [0,1] with NaN-safety."""
    x = np.array(x, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) == 0 or finite.max() == finite.min():
        return np.zeros_like(x)
    return (x - finite.min()) / (finite.max() - finite.min())

# ─── PDB ──────────────────────────────────────────────────────────────────────

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

# ─── WT CSV ───────────────────────────────────────────────────────────────────

def find_wt_csv(gene):
    g_lower = gene.lower()
    for p in CSV_DIR.glob("*_with_folddemand.csv"):
        base = p.name.split("_")[0]
        if base.lower() == g_lower:
            return p
    return None

def load_csv_features(gene, n_codons):
    """Return: domain_boundary annotation."""
    csv = find_wt_csv(gene)
    if not csv:
        return None
    df = pd.read_csv(csv)
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    cols = ['domain_boundary', 'domain_boundary_interpro', 'domain_boundary_pae']
    avail = [c for c in cols if c in cds.columns]
    if not avail:
        return None
    b = np.zeros(n_codons)
    for col in avail:
        v = cds[col].fillna(0).values[:n_codons]
        if len(v) < n_codons:
            v = np.pad(v, (0, n_codons - len(v)), constant_values=0)
        b = np.maximum(b, v)
    return b

# ═══════════════════════════════════════════════════════════════════════════════
# RISK PROXY 1: Cysteine cluster
# ═══════════════════════════════════════════════════════════════════════════════

def cysteine_risk(aa, window=CYS_WINDOW, min_count=CYS_MIN):
    """
    Cysteines in close sequence proximity have elevated disulfide
    mis-pairing risk. Score = number of additional Cys within ±window/2
    of each Cys (zero elsewhere).
    """
    n = len(aa)
    score = np.zeros(n)
    cys_pos = [i for i, a in enumerate(aa) if a == 'C']
    if len(cys_pos) < min_count:
        return score
    cys_set = set(cys_pos)
    half = window // 2
    for i in range(n):
        if aa[i] != 'C':
            continue
        # Count other Cys within window/2
        nearby = sum(1 for j in cys_pos if j != i and abs(j - i) <= half)
        if nearby >= 1:
            score[i] = nearby
        # Smear contribution to flanking residues
        for d in range(-3, 4):
            if 0 <= i + d < n:
                score[i + d] = max(score[i + d], nearby * 0.6)
    return score

# ═══════════════════════════════════════════════════════════════════════════════
# RISK PROXY 2: Charge + hydrophobic cluster (aggregation-prone, Lobanov-style)
# ═══════════════════════════════════════════════════════════════════════════════

def charge_aggregation_risk(aa, window=CHARGE_WINDOW):
    """
    Score = local |net charge| * local hydrophobicity * |hydrophobic gradient|
    Captures the classical 'beta-aggregation prone' signature: hydrophobic
    stretch flanked by/preceded by charged cluster.
    """
    n = len(aa)
    charges = np.array([CHARGE.get(a, 0) for a in aa])
    kd = np.array([KD.get(a, 0) for a in aa])

    # Smoothed local net charge density (magnitude)
    abs_charge_smooth = np.convolve(np.abs(charges), np.ones(window)/window, mode='same')
    # Smoothed local hydrophobicity (positive parts)
    hydro_smooth = np.convolve(np.maximum(kd, 0), np.ones(window)/window, mode='same')
    # Gradient of hydrophobicity (transitions from polar to hydrophobic)
    hydro_grad = np.abs(np.gradient(hydro_smooth))

    risk = abs_charge_smooth * hydro_smooth * (1 + hydro_grad)
    return risk

# ═══════════════════════════════════════════════════════════════════════════════
# RISK PROXY 3: Domain-adjacency
# ═══════════════════════════════════════════════════════════════════════════════

def domain_adjacency_risk(boundary_arr, spread=5):
    """
    Boundary positions and their immediate flanks.
    Score = max boundary value within spread.
    """
    if boundary_arr is None:
        return None
    n = len(boundary_arr)
    risk = np.zeros(n)
    for i in range(n):
        lo = max(0, i - spread)
        hi = min(n, i + spread + 1)
        risk[i] = boundary_arr[lo:hi].max()
    return risk

# ═══════════════════════════════════════════════════════════════════════════════
# RISK PROXY 4: pLDDT transitions
# ═══════════════════════════════════════════════════════════════════════════════

def plddt_transition_risk(plddt_arr, threshold=PLDDT_TRANSITION_THRESHOLD, window=5):
    """
    Sharp drops in pLDDT confidence between adjacent structured regions.
    Risk = magnitude of local pLDDT change.
    """
    if plddt_arr is None or len(plddt_arr) == 0:
        return None
    smoothed = np.convolve(plddt_arr, np.ones(window)/window, mode='same')
    grad = np.abs(np.gradient(smoothed))
    # Only score positions adjacent to high-confidence regions
    risk = grad * (smoothed > 50).astype(float)
    return risk

# ═══════════════════════════════════════════════════════════════════════════════
# RISK PROXY 5: WT codon-pausing signal
# ═══════════════════════════════════════════════════════════════════════════════

def wt_codon_risk(wt_codons, window=9):
    """
    Smoothed inverse TAI of the wild-type sequence.
    Reflects evolutionary 'where ribosome already slowed down'.
    """
    tai_w = tai_profile(wt_codons)
    smoothed = np.convolve(tai_w, np.ones(window)/window, mode='same')
    risk = 1 - smoothed   # invert so high risk = slow
    return risk

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL REGION RE-DETECTION (same as evaluate_sweep.py)
# ═══════════════════════════════════════════════════════════════════════════════

def hydro_profile(aa, window=9):
    kd = np.array([KD.get(a, 0.0) for a in aa])
    if len(kd) < window:
        return kd
    return np.convolve(kd, np.ones(window)/window, mode='same')

def aggregation_profile(aa, window=7):
    BETA = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'E':0.37,'Q':1.10,'G':0.75,
            'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,
            'T':1.19,'W':1.37,'Y':1.47,'V':1.70,'*':0.0}
    h = np.array([KD.get(a, 0.0) for a in aa])
    b = np.array([BETA.get(a, 1.0) for a in aa])
    score = np.maximum(h, 0) * b
    return np.convolve(score, np.ones(window)/window, mode='same')

def hydrophilic_positions(hydro, threshold=-1.0, min_len=5):
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

def aggregation_hotspot_positions(score, percentile=85):
    return np.where(score > np.percentile(score, percentile))[0]

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_gene(gene):
    """Compute all five risk channels for one gene."""
    # Find baseline JSON to get sequence
    pattern = f"{gene}_{GAMMA_LOW}_{MFE_TAG}_HEK293T_{EP_TAG}_results.json"
    json_path = OPT_DIR / pattern
    if not json_path.exists():
        # Fallback: any results file for this gene
        candidates = list(OPT_DIR.glob(f"{gene}_*_results.json"))
        if not candidates:
            return None
        json_path = candidates[0]

    with open(json_path) as f:
        d = json.load(f)
    
    # Use the WT (input) sequence — that's what we evaluate risk on
    # The cds_only field has the WT CDS (input to optimizer)
    wt_seq = d["sequence"]["cds_only"]
    wt_codons = codons_of(wt_seq)
    aa = aa_seq(wt_codons)
    n = len(wt_codons)
    
    # Load structural features
    pdb = find_pdb(gene)
    plddt = None
    if pdb:
        plddt_d, _ = parse_pdb(pdb)
        plddt = np.array([plddt_d.get(i, 0.0) for i in range(n)])
    
    boundary = load_csv_features(gene, n)
    
    # Compute five risk channels
    r_cys = cysteine_risk(aa)
    r_chg = charge_aggregation_risk(aa)
    r_dom = domain_adjacency_risk(boundary) if boundary is not None else np.zeros(n)
    r_pld = plddt_transition_risk(plddt) if plddt is not None else np.zeros(n)
    r_wt  = wt_codon_risk(wt_codons)
    
    # Normalize each channel to [0,1]
    r_cys_n = normalize_01(r_cys)
    r_chg_n = normalize_01(r_chg)
    r_dom_n = normalize_01(r_dom) if r_dom is not None else np.zeros(n)
    r_pld_n = normalize_01(r_pld) if r_pld is not None else np.zeros(n)
    r_wt_n  = normalize_01(r_wt)
    
    # Composite risk = mean of available channels
    composite = np.mean([r_cys_n, r_chg_n, r_dom_n, r_pld_n, r_wt_n], axis=0)
    
    # Identify risk regions
    risk_threshold = np.percentile(composite, RISK_PERCENTILE)
    risk_positions = np.where(composite > risk_threshold)[0]
    
    # Functional regions (for overlap analysis)
    hydro = hydro_profile(aa)
    agg = aggregation_profile(aa)
    hyd_pos = hydrophilic_positions(hydro)
    agg_pos = aggregation_hotspot_positions(agg)
    bnd_pos = np.where(boundary > 0.5)[0] if boundary is not None else np.array([])
    
    def overlap(a, b, n_total):
        """Fraction of b covered by a, and enrichment over random."""
        if len(a) == 0 or len(b) == 0:
            return 0.0, 0.0
        a_set = set(a.tolist()); b_set = set(b.tolist())
        intersection = len(a_set & b_set)
        # Fraction of b that's also in a
        frac = intersection / len(b_set) if b_set else 0.0
        # Enrichment over random expectation
        expected = (len(a_set) / n_total) * len(b_set)
        enrichment = intersection / expected if expected > 0 else 0.0
        return frac, enrichment
    
    out = {
        "gene": gene,
        "category": CATEGORY_OF.get(gene, 'unknown'),
        "n_codons": n,
        "n_cys": int(sum(1 for a in aa if a == 'C')),
        "n_risk_positions": len(risk_positions),
        "frac_risk": len(risk_positions) / n,
        "mean_composite_risk": float(composite.mean()),
        "max_composite_risk": float(composite.max()),
        # Per-channel summary
        "mean_cys_risk": float(r_cys_n.mean()),
        "mean_charge_risk": float(r_chg_n.mean()),
        "mean_domain_risk": float(r_dom_n.mean()),
        "mean_plddt_risk": float(r_pld_n.mean()),
        "mean_wt_risk": float(r_wt_n.mean()),
    }
    
    # Overlap with functional regions (does risk co-locate?)
    for region_name, region_pos in [("hydrophilic", hyd_pos),
                                      ("aggregation", agg_pos),
                                      ("boundary", bnd_pos)]:
        frac, enrich = overlap(risk_positions, region_pos, n)
        out[f"risk_overlap_{region_name}_frac"] = frac
        out[f"risk_overlap_{region_name}_enrichment"] = enrich
    
    # Correlation: composite risk vs WT TAI (negative = WT already slow at risk)
    tai_w = tai_profile(wt_codons)
    rho_risk_tai, p_risk_tai = spearmanr(composite, tai_w)
    out["spearman_risk_wt_tai"] = float(rho_risk_tai)
    out["spearman_risk_wt_tai_p"] = float(p_risk_tai)
    
    return {
        **out,
        "_arrays": {
            "composite": composite,
            "channels": {
                "cysteine": r_cys_n,
                "charge_agg": r_chg_n,
                "domain_adj": r_dom_n,
                "plddt_drop": r_pld_n,
                "wt_pausing": r_wt_n,
            },
            "wt_tai": tai_w,
            "hydro": hydro,
            "agg": agg,
            "risk_positions": risk_positions,
        }
    }


def main():
    # Find all genes with at least one results.json
    all_genes = set()
    for p in OPT_DIR.glob(f"*_{MFE_TAG}_HEK293T_{EP_TAG}_results.json"):
        all_genes.add(p.stem.split("_")[0])
    all_genes = sorted(all_genes)
    
    print(f"\n{'='*72}\n  Risk Evaluation\n{'='*72}\n")
    print(f"Genes found: {len(all_genes)}")
    print(f"  {', '.join(all_genes)}\n")
    
    results = []
    arrays = {}
    for gene in all_genes:
        try:
            r = analyze_gene(gene)
            if r is not None:
                arrays[gene] = r.pop("_arrays")
                results.append(r)
                print(f"  {gene:8s} [{r['category']:12s}]  n={r['n_codons']:4d}  "
                      f"n_Cys={r['n_cys']:2d}  "
                      f"risk_positions={r['n_risk_positions']:3d}  "
                      f"ρ(risk,WT_TAI)={r['spearman_risk_wt_tai']:+.3f}")
        except Exception as e:
            print(f"  {gene}: ERROR — {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "per_gene_risk.csv", index=False)
    print(f"\n[Saved] {OUT_DIR / 'per_gene_risk.csv'}")
    
    # ─── AGGREGATE STATISTICS ─────────────────────────────────────────────────
    print(f"\n{'─'*72}\n  Aggregate risk-region overlap analysis\n{'─'*72}\n")
    
    print(f"Mean fraction of risk regions overlapping each functional category:")
    for cat in ['hydrophilic', 'aggregation', 'boundary']:
        col = f"risk_overlap_{cat}_enrichment"
        if col in df.columns:
            mean_enrich = df[col].mean()
            n_enriched = (df[col] > 1.0).sum()
            print(f"  {cat:12s}: mean enrichment = {mean_enrich:.2f}x  ({n_enriched}/{len(df)} genes show enrichment > 1)")
    
    print(f"\nWT TAI correlation with composite risk:")
    print(f"  Mean Spearman ρ = {df['spearman_risk_wt_tai'].mean():+.3f}")
    print(f"  Median: {df['spearman_risk_wt_tai'].median():+.3f}")
    print(f"  Genes with significant negative correlation (p<0.05): "
          f"{((df['spearman_risk_wt_tai'] < 0) & (df['spearman_risk_wt_tai_p'] < 0.05)).sum()}/{len(df)}")
    print(f"  Interpretation: negative correlation means WT codon usage is slower at high-risk positions")
    
    # ─── PLOTS ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}\n  Plotting\n{'─'*72}\n")
    
    cat_colors = {'housekeeping': '#1f77b4', 'multi_domain': '#d62728',
                  'therapeutic': '#2ca02c', 'unknown': '#7f7f7f'}
    
    # Figure 1: Risk heatmap across all genes
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35*len(arrays))))
    sorted_genes = sorted(arrays.keys(), key=lambda g: -arrays[g]['composite'].mean())
    max_n = max(len(arrays[g]['composite']) for g in sorted_genes)
    heatmap = np.full((len(sorted_genes), max_n), np.nan)
    for i, g in enumerate(sorted_genes):
        c = arrays[g]['composite']
        heatmap[i, :len(c)] = c
    
    cmap = LinearSegmentedColormap.from_list('risk', ['white', 'orange', 'red', 'darkred'])
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_yticks(range(len(sorted_genes)))
    ax.set_yticklabels(sorted_genes, fontsize=9)
    ax.set_xlabel("Codon position")
    ax.set_title("Composite misfolding-risk profile per gene\n"
                 "(mean of cysteine, charge, domain, pLDDT-drop, WT-pausing channels)")
    plt.colorbar(im, ax=ax, label="Composite risk (0-1)")
    # Color the gene labels by category
    for tick, g in zip(ax.get_yticklabels(), sorted_genes):
        cat = CATEGORY_OF.get(g, 'unknown')
        tick.set_color(cat_colors[cat])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "risk_heatmap.png", dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[Saved] risk_heatmap.png")
    
    # Figure 2: Per-channel breakdown for a single gene (example: ACTB if present)
    example = "ACTB" if "ACTB" in arrays else sorted_genes[0]
    arr = arrays[example]
    n = len(arr['composite'])
    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    positions = np.arange(n)
    channels = arr['channels']
    titles = ['Cysteine cluster',
              'Charge/aggregation',
              'Domain adjacency',
              'pLDDT transitions',
              'WT codon pausing']
    keys = ['cysteine', 'charge_agg', 'domain_adj', 'plddt_drop', 'wt_pausing']
    
    for ax, key, title in zip(axes[:5], keys, titles):
        ax.fill_between(positions, channels[key], color='steelblue', alpha=0.6)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    
    axes[5].fill_between(positions, arr['composite'], color='darkred', alpha=0.7,
                          label='Composite risk')
    threshold = np.percentile(arr['composite'], RISK_PERCENTILE)
    axes[5].axhline(threshold, color='black', ls='--', lw=0.7, label=f'Risk threshold ({RISK_PERCENTILE}th pct)')
    axes[5].set_ylabel('Composite', fontsize=9)
    axes[5].set_xlabel('Codon position')
    axes[5].legend(loc='upper right', fontsize=8)
    axes[5].grid(alpha=0.3)
    
    fig.suptitle(f"Misfolding-risk profile: {example}", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"risk_breakdown_{example}.png", dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[Saved] risk_breakdown_{example}.png")
    
    # Figure 3: Overlap with functional regions (boxplots by category)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, region in zip(axes, ['hydrophilic', 'aggregation', 'boundary']):
        col = f"risk_overlap_{region}_enrichment"
        if col not in df.columns:
            continue
        data = []
        labels = []
        for cat in ['housekeeping', 'multi_domain', 'therapeutic']:
            sub = df[df['category'] == cat][col].dropna().values
            if len(sub) > 0:
                data.append(sub)
                labels.append(f"{cat}\nN={len(sub)}")
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, cat in zip(bp['boxes'], ['housekeeping', 'multi_domain', 'therapeutic']):
            patch.set_facecolor(cat_colors[cat])
            patch.set_alpha(0.7)
        ax.axhline(1.0, color='black', ls='--', lw=0.7, label='Random expectation')
        ax.set_ylabel("Enrichment over random")
        ax.set_title(f"Risk regions ∩ {region} regions")
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle("Co-localisation of misfolding-risk regions with functional regions", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "risk_overlap.png", dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[Saved] risk_overlap.png")
    
    # Figure 4: WT TAI vs composite risk correlation
    fig, ax = plt.subplots(figsize=(10, 6))
    rhos = df['spearman_risk_wt_tai'].values
    genes = df['gene'].values
    cats = df['category'].values
    order = np.argsort(rhos)
    colors = [cat_colors[cats[i]] for i in order]
    ax.barh(range(len(rhos)), rhos[order], color=colors, alpha=0.85)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels(genes[order], fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(rhos.mean(), color='blue', lw=1, ls='--', label=f'mean={rhos.mean():+.3f}')
    ax.set_xlabel("Spearman ρ (composite risk, WT TAI)")
    ax.set_title("Wild-type codon pausing at misfolding-risk positions\n"
                 "(negative ρ = WT already uses slower codons at high-risk positions)")
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    handles = [plt.Rectangle((0,0),1,1, color=cat_colors[c], alpha=0.85)
               for c in ['housekeeping', 'multi_domain', 'therapeutic']]
    ax.legend(handles + [ax.lines[1]],
              ['housekeeping', 'multi_domain', 'therapeutic', f'mean={rhos.mean():+.3f}'],
              fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "wt_codon_at_risk.png", dpi=160, bbox_inches='tight')
    plt.close()
    print(f"[Saved] wt_codon_at_risk.png")
    
    # ─── SUMMARY ──────────────────────────────────────────────────────────────
    summary = OUT_DIR / "risk_summary.txt"
    with open(summary, 'w') as f:
        f.write(f"Misfolding-Risk Evaluation Summary\n{'='*72}\n\n")
        f.write(f"Number of genes: {len(df)}\n")
        f.write(f"Categories: {df['category'].value_counts().to_dict()}\n\n")
        f.write(f"Per-gene risk-region counts (top 15% composite risk):\n")
        for _, r in df.iterrows():
            f.write(f"  {r['gene']:8s} [{r['category']:12s}]  "
                    f"n_codons={r['n_codons']:4d}  "
                    f"risk_positions={r['n_risk_positions']:3d} "
                    f"({100*r['frac_risk']:.0f}%)  "
                    f"n_Cys={r['n_cys']:2d}\n")
        f.write(f"\nMean enrichment of risk regions in functional regions:\n")
        for region in ['hydrophilic', 'aggregation', 'boundary']:
            col = f"risk_overlap_{region}_enrichment"
            if col in df.columns:
                f.write(f"  {region:12s}: {df[col].mean():.2f}x  "
                        f"(enrichment > 1 in {(df[col] > 1.0).sum()}/{len(df)} genes)\n")
        f.write(f"\nWT codon-pausing at risk positions:\n")
        f.write(f"  Mean ρ(composite risk, WT TAI) = {df['spearman_risk_wt_tai'].mean():+.3f}\n")
        f.write(f"  Negative correlation = wild-type sequence already uses slow codons\n")
        f.write(f"  at high-risk positions (evolutionary support for risk locations)\n")
    print(f"\n[Saved] {summary}")
    print(f"\n{'='*72}\nDone. Outputs in {OUT_DIR}\n{'='*72}")


if __name__ == "__main__":
    main()