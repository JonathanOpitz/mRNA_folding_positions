#!/usr/bin/env python3
"""
Add local RNA structure features from RNAplfold (ViennaRNA) to per-codon CSV files.
Uses isoform_selection.json to find the correct FASTA file.

Features added per codon:
  - rna_unpaired_{1-5}nt          unpaired probabilities from plfold_lunp
  - rna_local_paired_prob         1 - unpaired_1nt
  - rna_opening_energy_{1,3,5}nt  unfolding cost from plfold_openen
  - rna_paired_prob_window5cod    sliding-window mean ±15 nt
  - rna_paired_prob_window10cod   sliding-window mean ±30 nt
  - rna_struct_change             local minus regional paired_prob
  - rna_local_ss_class            stem / weak / open_loop
"""

import logging
import re
import json
import sys
import time
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from utils.rnaplfold_utils import run_rnaplfold, aggregate_codon_features

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/Users/jonathanopitz/Desktop/Master")
RESULTS_DIR  = BASE_DIR / "data/ribo_counts"
GENES_DIR    = BASE_DIR / "data/genes"
ISOFORM_JSON = BASE_DIR / "isoform_selection.json"
OUT_SUFFIX   = "_with_rnaplfold.csv"

# ─── Load isoforms ────────────────────────────────────────────────────────────
if not ISOFORM_JSON.exists():
    log.error("isoform_selection.json not found: %s", ISOFORM_JSON)
    sys.exit(1)

with open(ISOFORM_JSON) as f:
    isoforms = json.load(f)


# ─── Sequence loading ─────────────────────────────────────────────────────────

def get_full_sequence(fasta_path: Path) -> str:
    """Read the first record from a FASTA file and return the RNA sequence (T→U)."""
    record = next(SeqIO.parse(fasta_path, "fasta"))
    return str(record.seq).upper().replace("T", "U")


def find_best_fasta(gene: str) -> Path | None:
    """Look up the canonical isoform FASTA for gene from isoform_selection.json."""
    gene_upper = gene.upper()
    if gene_upper not in isoforms:
        log.warning("  No entry in isoform_selection.json for %s", gene_upper)
        return None

    info = isoforms[gene_upper]
    if info.get("status") != "ok":
        log.warning("  Status is not 'ok' for %s", gene_upper)
        return None

    best_enst = info.get("best_isoform")
    best_full = info.get("best_isoform_full")

    patterns = [
        f"{gene_upper}_{best_full}.fasta",
        f"{gene_upper}_{best_enst}.fasta",
        f"{gene_upper}_{best_enst}.*.fasta",
        f"*{gene_upper}*{best_enst}*.fasta",
        f"*{gene_upper}*{best_full}*.fasta",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(GENES_DIR.glob(pat))

    if not candidates:
        log.warning("  No FASTA found for %s / %s or %s", gene_upper, best_enst, best_full)
        return None

    fasta_path = max(candidates, key=lambda p: (p.stat().st_size, p.stat().st_mtime))
    log.info("  FASTA: %s  (%s bytes)", fasta_path.name, f"{fasta_path.stat().st_size:,}")
    return fasta_path


# ─── Per-file processing ──────────────────────────────────────────────────────

def process_one_file(csv_path: Path) -> str:
    """
    Add RNAplfold features to a single *_with_structure.csv file.

    Returns 'ok', 'skipped', or 'error'.
    """
    stem = csv_path.stem
    gene_raw = re.split(r'_ribosome|_with_structure|_counts|\.', stem)[0]
    gene = gene_raw.strip().upper()
    log.info("\n%s\nProcessing: %s  (file: %s)", "═" * 70, gene, stem)

    out_path = csv_path.parent / f"{csv_path.stem}{OUT_SUFFIX}"
    if out_path.is_file():
        log.info("  Skipping — output already exists: %s", out_path.name)
        return "skipped"

    df = pd.read_csv(csv_path)

    fasta_path = find_best_fasta(gene)
    if not fasta_path:
        return "error"

    seq = get_full_sequence(fasta_path)
    log.info("  Sequence length: %s nt", f"{len(seq):,}")

    tmp_dir = RESULTS_DIR / f"{gene}_plfold_tmp"
    nt_features = run_rnaplfold(seq, gene, tmp_dir)

    def map_codon(row):
        """Map per-nucleotide features to a codon row."""
        if pd.isna(row.get('nt_start')) or pd.isna(row.get('nt_end')):
            return pd.Series(aggregate_codon_features({}, 0, 0))
        return pd.Series(aggregate_codon_features(nt_features, int(row['nt_start']), int(row['nt_end'])))

    new_cols = df.apply(map_codon, axis=1)
    pd.concat([df, new_cols], axis=1).to_csv(out_path, index=False)
    log.info("  Saved: %s", out_path.name)
    log.info("  New columns: %s", ", ".join(new_cols.columns))
    return "ok"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_csv = sorted(RESULTS_DIR.glob("*_with_structure.csv"))
    all_csv = [f for f in all_csv if "_with_rnaplfold" not in f.stem]

    if not all_csv:
        log.error("No *_with_structure.csv files found in %s", RESULTS_DIR)
        sys.exit(1)

    log.info("RNAplfold feature pipeline — %d genes", len(all_csv))

    results = {"ok": [], "skipped": [], "error": []}
    t_start = time.time()

    for idx, csv_path in enumerate(all_csv, 1):
        gene_raw = re.split(r'_ribosome|_with_structure|_counts|\.', csv_path.stem)[0].upper()
        elapsed = time.time() - t_start
        log.info("\n[%d/%d]  %s  (elapsed: %.0fs)", idx, len(all_csv), gene_raw, elapsed)

        try:
            status = process_one_file(csv_path)
        except Exception as e:
            log.error("  FAILED: %s", e)
            status = "error"

        results[status].append(gene_raw)

    elapsed_total = time.time() - t_start
    log.info("\n%s", "═" * 70)
    log.info("Done in %.0fs (%.1f min)", elapsed_total, elapsed_total / 60)
    log.info("  OK:      %d", len(results['ok']))
    log.info("  Skipped: %d", len(results['skipped']))
    log.info("  Errors:  %d", len(results['error']))

    if results['error']:
        log.warning("Failed genes: %s", ", ".join(results['error']))


if __name__ == '__main__':
    main()
