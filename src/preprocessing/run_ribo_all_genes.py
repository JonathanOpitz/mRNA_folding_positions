#!/usr/bin/env python3
"""
Runs fast_ribo_analysis.py for all genes in isoform_selection.json.
Uses EXACTLY the best isoform FASTA per gene – NO fallback.
Skips genes with no_clean_isoform or other problem statuses.
"""

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parents[2]
ISOFORM_JSON = BASE / "isoform_selection.json"
FASTQ        = BASE / "data/ribo_fastq/SRR10072555.fastq"
GENES_DIR    = BASE / "data/genes"
SCRIPT       = BASE / "src/preprocessing/fast_ribo_analysis.py"
OUTPUT_DIR   = BASE / "data/ribo_counts"
PYTHON       = BASE / "folding/bin/python"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ISOFORM_JSON) as f:
    isoforms = json.load(f)

genes_ok      = []
genes_skipped = []

log.info("Reading isoform_selection.json ...")

for gene, info in isoforms.items():
    status = info.get("status", "")

    if status != "ok":
        log.info("  SKIP %-14s (status: %s)", gene, status)
        genes_skipped.append((gene, status))
        continue

    best_enst = info.get("best_isoform", "")
    if not best_enst:
        log.info("  SKIP %-14s (no best_isoform in JSON)", gene)
        genes_skipped.append((gene, "no_best_isoform"))
        continue

    matches = list(GENES_DIR.glob(f"{gene}_{best_enst}.fasta"))
    if not matches:
        matches = list(GENES_DIR.glob(f"{gene}_{best_enst}*.fasta"))

    if not matches:
        log.warning("  MISS %-14s FASTA not found for %s", gene, best_enst)
        genes_skipped.append((gene, f"fasta_missing:{best_enst}"))
        continue

    fasta    = matches[0]
    prot_len = info.get("protein_length", "?")
    tpm      = info.get("tpm", 0)
    log.info("  OK   %-14s %s  %6s AA  TPM:%.1f  → %s", gene, best_enst, prot_len, tpm, fasta.name)
    genes_ok.append((gene, fasta, best_enst))

# ── Pre-run summary ───────────────────────────────────────────────────────────
log.info("\nGenes to run  : %d", len(genes_ok))
log.info("Genes skipped : %d", len(genes_skipped))
if genes_skipped:
    for g, reason in genes_skipped:
        log.info("  - %-14s %s", g, reason)
log.info("Output dir    : %s", OUTPUT_DIR)

if not genes_ok:
    log.error("Nothing to run. Check isoform_selection.json and GENES_DIR.")
    exit(1)

input("Press Enter to start, Ctrl+C to abort ...")
print()

# ── Run ───────────────────────────────────────────────────────────────────────
failed = []

for i, (gene, fasta, enst) in enumerate(genes_ok, 1):
    out_csv = OUTPUT_DIR / f"{gene.lower()}_ribosome_counts.csv"

    log.info("[%2d/%d] %s", i, len(genes_ok), gene)
    log.info("        FASTA  : %s", fasta.name)
    log.info("        Output : %s", out_csv.name)

    result = subprocess.run(
        [str(PYTHON), str(SCRIPT), str(FASTQ), gene, str(fasta)],
        cwd=str(OUTPUT_DIR),
    )

    if result.returncode != 0:
        log.error("        FAILED (exit %d)", result.returncode)
        failed.append(gene)
    elif out_csv.is_file():
        log.info("        Done")
    else:
        log.warning("        Process exited OK but CSV not found")
        failed.append(gene)

# ── Final summary ─────────────────────────────────────────────────────────────
log.info("\nFinished : %d/%d successful", len(genes_ok) - len(failed), len(genes_ok))
if failed:
    log.warning("Failed   : %s", ", ".join(failed))
log.info("CSVs in  : %s", OUTPUT_DIR)