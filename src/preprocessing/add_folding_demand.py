#!/usr/bin/env python3
"""
Folding Demand — VERSION 8

CRITICAL FIX vs v7: GLOBAL normalization (not per-gene!)

Problem in v7:
  Each gene got its OWN percentile stretch → every gene's max = 1.0
  This was biologically nonsensical:
    - A simple 50-aa peptide with no folding challenges got max=1.0
    - ACTB (complex multi-domain) got max=1.0
    - → No way to express that some proteins are INTRINSICALLY harder to fold
    - → GNN trained on this learned false equivalence between genes
    - → For the optimizer, a high-fd position in a simple protein was
      treated the same as a high-fd position in a complex protein.

Fix in v8: TWO-PASS algorithm
  Pass 1: Compute fd_raw for ALL genes, accumulate ALL raw values
  Pass 2: Compute GLOBAL p5, p95 across all genes combined
          Apply the SAME global stretch to every gene's fd_raw

Result: absolute values now comparable across genes
  - Hard-to-fold proteins (lots of contacts, domains): many positions >0.7
  - Simple proteins: few or no positions >0.7
  - This is what's biologically realistic and what the GNN should learn.

Other changes from v7: none. Score formulas unchanged.

Processes both data/ribo_counts/ (WT) and data/ribo_counts_simulated/ (GEMORNA).
"""

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DIRS_TO_PROCESS = [
    BASE_DIR / "data/ribo_counts",
    BASE_DIR / "data/ribo_counts_simulated",
]

IN_SUFFIX  = "_with_rnaplfold.csv"
OUT_SUFFIX = "_with_folddemand.csv"

# ── Protein need weights (unchanged) ──────────────────────────────────────────
W_CONTACT     = 0.50
W_DOMAIN      = 0.25
W_SS_TRANS    = 0.25
CONTACT_BOOST = 1.25
PLDDT_POWER   = 1.40
SIGMA_DOMAIN  = 3
SIGMA_SS      = 2
DOMAIN_SHIFT  = 4

# ── Ribo confirmation ─────────────────────────────────────────────────────────
RIBO_SMOOTH    = 5
RIBO_THRESHOLD = 3.0

# ── v7/v8: Score-spreading + reduced ribo influence ──────────────────────────
PN_POWER    = 0.65   # pn^0.65 opens up top end
RIBO_GAMMA  = 0.15   # rc modulates pn by ±7.5% before stretch

# ── v8 NEW: Global stretch parameters ─────────────────────────────────────────
STRETCH_LO  = 5.0    # Global 5th percentile across ALL genes → 0.0
STRETCH_HI  = 95.0   # Global 95th percentile across ALL genes → 1.0


# ─── Core feature computation (no stretch here) ───────────────────────────────

def compute_fd_raw(df: pd.DataFrame) -> dict:
    """
    Compute all fd sub-scores AND fd_raw for CDS codons.
    Does NOT stretch to [0,1] — that happens globally in pass 2.

    Returns dict with arrays and the CDS mask, so we can update df later.
    """
    cds_mask = df['region'] == 'CDS'
    cds = df.loc[cds_mask].copy()
    if len(cds) == 0:
        return None

    n = len(cds)

    # 1. Contact density (non-linear boost)
    cd = cds['contact_density'].fillna(0).values.astype(float)
    cd_max = cd.max() if cd.max() > 0 else 1.0
    cd_boosted = (cd / cd_max) ** CONTACT_BOOST
    cd_boosted = cd_boosted / (cd_boosted.max() or 1.0)

    # 2. Domain boundaries (upstream shift for ribosome tunnel)
    domain = cds['domain_boundary'].fillna(0).values.astype(float)
    domain_smooth = gaussian_filter1d(domain, sigma=SIGMA_DOMAIN)
    domain_shifted = np.zeros(n)
    if n > DOMAIN_SHIFT:
        domain_shifted[:n - DOMAIN_SHIFT] = domain_smooth[DOMAIN_SHIFT:]
    domain_shifted_norm = domain_shifted / (domain_shifted.max() or 1.0)

    # 3. Secondary structure transitions
    ss_cols = cds[['ss_H', 'ss_E', 'ss_C']].fillna(0).values.astype(float)
    ss_type = np.argmax(ss_cols, axis=1)
    ss_trans = np.zeros(n)
    ss_trans[1:] = (ss_type[1:] != ss_type[:-1]).astype(float)
    ss_trans_norm = gaussian_filter1d(ss_trans, sigma=SIGMA_SS)
    ss_trans_norm = ss_trans_norm / (ss_trans_norm.max() or 1.0)

    # 4. pLDDT confidence weight
    plddt_weight = cds['plddt'].fillna(0).values.astype(float) / 100.0

    # 5. Protein need (power transform)
    raw = W_CONTACT * cd_boosted + W_DOMAIN * domain_shifted_norm + W_SS_TRANS * ss_trans_norm
    protein_need = (raw ** PN_POWER) * (plddt_weight ** PLDDT_POWER)

    # 6. Ribo confirmation (global threshold)
    rel_occ = cds['rel_occupancy'].fillna(0).values.astype(float)
    ribo_confirmation = np.clip(
        gaussian_filter1d(rel_occ, sigma=RIBO_SMOOTH) / RIBO_THRESHOLD,
        0.0, 1.0,
    )

    # 7. Gentle ribo modulation (additive, ±7.5%)
    fd_raw = protein_need * (1.0 + RIBO_GAMMA * (ribo_confirmation - 0.5))

    return dict(
        cds_mask=cds_mask,
        fd_raw=fd_raw,
        protein_need=protein_need,
        ribo_confirmation=ribo_confirmation,
        cd_boosted=cd_boosted,
        domain_shifted_norm=domain_shifted_norm,
        ss_trans_norm=ss_trans_norm,
        plddt_weight=plddt_weight,
    )


def apply_stretch_and_save(df: pd.DataFrame, data: dict,
                            global_lo: float, global_hi: float,
                            out_path: Path) -> tuple:
    """Apply global stretch to fd_raw, write assignments to df, save CSV."""
    out_cols = [
        "fd_protein_need", "fd_ribo_confirmation", "fd_raw",
        "fd_contact_complexity", "fd_domain_transition",
        "fd_ss_transition", "fd_plddt_weight",
        "folding_demand",
    ]
    for col in out_cols:
        df[col] = np.nan

    if global_hi > global_lo:
        folding_demand = np.clip(
            (data['fd_raw'] - global_lo) / (global_hi - global_lo),
            0.0, 1.0,
        )
    else:
        folding_demand = np.clip(data['fd_raw'], 0.0, 1.0)

    cds_mask = data['cds_mask']
    df.loc[cds_mask, 'fd_protein_need']       = np.round(data['protein_need'], 4)
    df.loc[cds_mask, 'fd_ribo_confirmation']  = np.round(data['ribo_confirmation'], 4)
    df.loc[cds_mask, 'fd_raw']                = np.round(data['fd_raw'], 4)
    df.loc[cds_mask, 'fd_contact_complexity'] = np.round(data['cd_boosted'], 4)
    df.loc[cds_mask, 'fd_domain_transition']  = np.round(data['domain_shifted_norm'], 4)
    df.loc[cds_mask, 'fd_ss_transition']      = np.round(data['ss_trans_norm'], 4)
    df.loc[cds_mask, 'fd_plddt_weight']       = np.round(data['plddt_weight'], 4)
    df.loc[cds_mask, 'folding_demand']        = np.round(folding_demand, 4)

    df.to_csv(out_path, index=False)

    pct_high = (folding_demand > 0.7).mean() * 100
    pct_top  = (folding_demand > 0.9).mean() * 100
    return (folding_demand.mean(), folding_demand.max(), folding_demand.std(),
            pct_high, pct_top)


# ─── File discovery ───────────────────────────────────────────────────────────

def discover_inputs() -> list:
    """Return list of (dir_name, gene_name, csv_path) tuples to process."""
    found = []
    for results_dir in DIRS_TO_PROCESS:
        if not results_dir.exists():
            continue
        all_csv = sorted(results_dir.glob(f"*{IN_SUFFIX}"))
        all_csv = [f for f in all_csv if not f.stem.endswith("_with_folddemand")]
        for csv_path in all_csv:
            stem = csv_path.stem
            gene = re.split(r'_ribosome|_with_|_counts|\.', stem)[0].upper()
            found.append((results_dir.name, gene, csv_path))
    return found


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info(
        "Folding Demand v8 (GLOBAL stretch) | fd = stretch(pn^%.2f × (1 + %.2f × (rc-0.5)), global_p%d-p%d)",
        PN_POWER, RIBO_GAMMA, int(STRETCH_LO), int(STRETCH_HI),
    )
    log.info(
        "  pn built from: W_contact=%.2f, W_domain=%.2f, W_ss=%.2f | plddt^%.1f",
        W_CONTACT, W_DOMAIN, W_SS_TRANS, PLDDT_POWER,
    )
    log.info(
        "  rc influence: ±%.1f%% of pn (structure dominates)",
        RIBO_GAMMA * 50,
    )
    log.info("  → Stretch percentiles computed across ALL genes together.")

    t_start = time.time()
    inputs = discover_inputs()
    log.info("\nFound %d files to process across %d directories.",
             len(inputs), len(DIRS_TO_PROCESS))

    # ── PASS 1: Compute fd_raw for every gene, collect all values ─────────────
    log.info("\n%s", "═" * 70)
    log.info("PASS 1: computing fd_raw per gene (no stretch yet)")
    log.info("%s", "═" * 70)

    all_genes = []       # [(dir_name, gene, csv_path, df, data)]
    all_fd_raw_values = []
    pass1_errors = []

    for idx, (dir_name, gene, csv_path) in enumerate(inputs, 1):
        log.info("[%d/%d] %s / %s", idx, len(inputs), dir_name, gene)
        try:
            df = pd.read_csv(csv_path)
            required = ['region', 'contact_density', 'domain_boundary',
                        'ss_H', 'ss_E', 'ss_C', 'plddt', 'rel_occupancy']
            missing = [c for c in required if c not in df.columns]
            if missing:
                log.error("  Missing columns: %s", missing)
                pass1_errors.append((dir_name, gene))
                continue

            data = compute_fd_raw(df)
            if data is None:
                log.error("  No CDS rows")
                pass1_errors.append((dir_name, gene))
                continue

            all_genes.append((dir_name, gene, csv_path, df, data))
            all_fd_raw_values.append(data['fd_raw'])

            # Quick stats for this gene (pre-stretch)
            raw = data['fd_raw']
            log.info("  fd_raw: mean=%.3f max=%.3f std=%.3f  n_codons=%d",
                     raw.mean(), raw.max(), raw.std(), len(raw))
        except Exception as e:
            log.error("  ERROR: %s", e)
            pass1_errors.append((dir_name, gene))

    if not all_fd_raw_values:
        log.error("No valid genes processed. Aborting.")
        return

    # ── Compute GLOBAL percentiles across all genes ───────────────────────────
    global_pool = np.concatenate(all_fd_raw_values)
    global_lo = np.percentile(global_pool, STRETCH_LO)
    global_hi = np.percentile(global_pool, STRETCH_HI)

    log.info("\n%s", "═" * 70)
    log.info("GLOBAL STATISTICS across %d genes, %d codons total",
             len(all_fd_raw_values), len(global_pool))
    log.info("%s", "═" * 70)
    log.info("  fd_raw distribution: mean=%.3f median=%.3f std=%.3f",
             global_pool.mean(), np.median(global_pool), global_pool.std())
    log.info("  percentiles: p5=%.3f p25=%.3f p50=%.3f p75=%.3f p95=%.3f p99=%.3f",
             np.percentile(global_pool, 5),
             np.percentile(global_pool, 25),
             np.percentile(global_pool, 50),
             np.percentile(global_pool, 75),
             np.percentile(global_pool, 95),
             np.percentile(global_pool, 99))
    log.info("  GLOBAL stretch: [%.3f, %.3f] → [0, 1]", global_lo, global_hi)

    # ── PASS 2: Apply global stretch, save each gene ──────────────────────────
    log.info("\n%s", "═" * 70)
    log.info("PASS 2: applying global stretch and saving")
    log.info("%s", "═" * 70)

    pass2_ok = 0
    pass2_errors = []
    # Track cross-gene variation in final folding_demand
    gene_maxes = []
    gene_means = []

    for (dir_name, gene, csv_path, df, data) in all_genes:
        try:
            out_path = csv_path.parent / f"{csv_path.stem}{OUT_SUFFIX}"
            if out_path.exists():
                out_path.unlink()

            mean_v, max_v, std_v, pct_high, pct_top = apply_stretch_and_save(
                df, data, global_lo, global_hi, out_path
            )
            gene_maxes.append(max_v)
            gene_means.append(mean_v)
            log.info("  %s / %s: mean=%.3f max=%.3f std=%.3f | >0.7:%.1f%% >0.9:%.1f%%",
                     dir_name, gene, mean_v, max_v, std_v, pct_high, pct_top)
            pass2_ok += 1
        except Exception as e:
            log.error("  %s / %s: ERROR during save: %s", dir_name, gene, e)
            pass2_errors.append((dir_name, gene))

    elapsed = time.time() - t_start
    log.info("\n%s", "═" * 70)
    log.info("Done in %.1fs", elapsed)
    log.info("%s", "═" * 70)
    log.info("  Pass 1 errors: %d", len(pass1_errors))
    log.info("  Pass 2 OK:     %d", pass2_ok)
    log.info("  Pass 2 errors: %d", len(pass2_errors))

    if gene_maxes:
        gm = np.array(gene_maxes); gn = np.array(gene_means)
        log.info("\n  CROSS-GENE VARIATION (this is what you want to see):")
        log.info("    Per-gene fd max:  min=%.3f  mean=%.3f  max=%.3f  std=%.3f",
                 gm.min(), gm.mean(), gm.max(), gm.std())
        log.info("    Per-gene fd mean: min=%.3f  mean=%.3f  max=%.3f  std=%.3f",
                 gn.min(), gn.mean(), gn.max(), gn.std())
        log.info("    (std > 0 means: genes DIFFER in how hard they are to fold) ✓")


if __name__ == '__main__':
    main()