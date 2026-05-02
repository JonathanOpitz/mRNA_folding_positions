#!/usr/bin/env python3
"""
Add RNAplfold RNA structure features to simulated per-codon CSV files.

For GEMORNA-generated sequences the protein is unchanged (so AlphaFold structure
features are reused from WT), but the mRNA sequence differs — so RNAplfold must
be re-run on the synthetic coding sequence extracted from the CSV.
"""

import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

from utils.rnaplfold_utils import run_rnaplfold, aggregate_codon_features

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/Users/jonathanopitz/Desktop/Master")
SIMULATED_DIR = BASE_DIR / "data/ribo_counts_simulated"
OUT_SUFFIX    = "_with_rnaplfold.csv"


# ─── Sequence extraction ──────────────────────────────────────────────────────

def get_full_mrna_from_simulated_csv(csv_path: Path) -> str:
    """Reconstruct the full mRNA sequence from codon rows in a simulated CSV."""
    df = pd.read_csv(csv_path)
    seq_col = next(
        (col for col in ['sequence', 'codon', 'nt_seq', 'nt_sequence', 'seq'] if col in df.columns),
        None,
    )
    if not seq_col:
        raise ValueError(f"No sequence column found in {csv_path.name}")
    df_sorted = df.sort_values(by='nt_start').copy()
    full_seq = "".join(
        df_sorted[seq_col].astype(str).str.upper().str.replace("T", "U")
    )
    log.info("  Extracted %s nt from column '%s'", f"{len(full_seq):,}", seq_col)
    return full_seq


# ─── Per-file processing ──────────────────────────────────────────────────────

def process_one_file(csv_path: Path) -> str:
    """
    Add RNAplfold features to a single simulated *_with_structure.csv file.

    Returns 'ok', 'skipped', or 'error'.
    """
    stem = csv_path.stem
    gene_raw = re.split(r'_ribosome|_with_structure|_counts|\.', stem)[0]
    gene = gene_raw.strip().upper()
    log.info("\n%s\nProcessing: %s", "═" * 80, gene)

    out_path = csv_path.parent / f"{stem}{OUT_SUFFIX}"
    if out_path.is_file():
        log.info("  Skipping — output already exists")
        return "skipped"

    try:
        seq = get_full_mrna_from_simulated_csv(csv_path)
    except Exception as e:
        log.error("  Sequence extraction failed: %s", e)
        return "error"

    tmp_dir = SIMULATED_DIR / f"{gene}_plfold_tmp"
    nt_features = run_rnaplfold(seq, gene, tmp_dir)

    df = pd.read_csv(csv_path)

    def map_codon(row):
        """Map per-nucleotide features to a codon row."""
        if pd.isna(row.get('nt_start')) or pd.isna(row.get('nt_end')):
            return pd.Series(aggregate_codon_features({}, 0, 0))
        return pd.Series(aggregate_codon_features(nt_features, int(row['nt_start']), int(row['nt_end'])))

    new_cols = df.apply(map_codon, axis=1)
    pd.concat([df, new_cols], axis=1).to_csv(out_path, index=False)
    log.info("  Saved: %s", out_path.name)
    return "ok"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    csv_files = sorted(SIMULATED_DIR.glob("*_with_structure.csv"))
    if not csv_files:
        log.error("No *_with_structure.csv files found in %s", SIMULATED_DIR)
        sys.exit(1)

    log.info("RNAplfold pipeline (simulated) — %d files", len(csv_files))

    results = {"ok": [], "skipped": [], "error": []}
    t_start = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        log.info("[%d/%d] %s", idx, len(csv_files), csv_path.name)
        try:
            status = process_one_file(csv_path)
        except Exception as e:
            log.error("  FAILED: %s", e)
            status = "error"
        results[status].append(csv_path.stem)

    elapsed = time.time() - t_start
    log.info("\nDone in %.0fs", elapsed)
    log.info("  OK:      %d", len(results['ok']))
    log.info("  Skipped: %d", len(results['skipped']))
    log.info("  Errors:  %d", len(results['error']))


if __name__ == '__main__':
    main()
