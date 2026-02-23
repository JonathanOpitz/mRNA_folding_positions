#!/usr/bin/env python3
"""
Batch: Select best isoform per gene (based on Salmon TPM) and run ribosome profiling alignment.
Saves all CSVs directly to data/results/
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from Bio import SeqIO
import subprocess
import tempfile
import shutil

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

BASE_DIR = Path("/Users/jonathanopitz/Desktop/Master")
GENCODE_FA = BASE_DIR / "data/genes/gencode/gencode.v48.pc_transcripts.fa"
QUANT_FILE = BASE_DIR / "salmon_quant/quant.sf"
ALIGN_SCRIPT = BASE_DIR / "src/preprocessing/fast_ribo_analysis.py"
FASTQ = BASE_DIR / "data/ribo_fastq/SRR10072555.fastq"
RESULTS_DIR = BASE_DIR / "data/results"

MIN_TPM_ABS = 5.0       # Minimum absolute TPM
MIN_TPM_PCT = 10.0      # Minimum percentage of gene's total TPM

GENES = [
    "GAPDH", "LDHA", "LDHB", "ENO1", "PKM", "ALDOA", "PGK1", "IDH1", "IDH2", "MDH2",
    "G6PD", "PFKL", "PFKM", "ACO2", "CS", "SDHA", "FH", "GOT2", "ACTB", "ACTG1",
    "TUBB", "TUBA1B", "TUBA1A", "VIM", "LMNA", "LMNB1", "EEF1A1", "EEF2", "EIF4A1",
    "EIF3A", "EIF2S1", "EEF1G", "HSP90AA1", "HSP90AB1", "HSPA8", "HSPA5", "HSPD1",
    "CCT2", "CCT3", "CCT4", "DNAJA1", "DNAJB1", "MAPK1", "MAPK3", "AKT1", "AKT2",
    "PRKACA", "GSK3B", "SRC", "EGFR", "PIK3CA", "HNRNPA1", "HNRNPC", "HNRNPK",
    "RFC1", "TOP1", "PARP1", "JUN", "FOS", "MYC", "TP53", "STAT3", "NFKB1",
    "CCNB1", "CCNE1", "CDC20", "PLK1", "AURKB", "CASP7", "CTSB", "CTSD", "PSMD1",
    "PSMD2", "CAT", "TXNRD1", "VCP", "NSF", "ALB", "SERPINA1", "F9", "INSR",
    "NCL", "RPLP0"
]

def get_best_isoform(gene_name: str):
    """Find best transcript by TPM"""
    if not QUANT_FILE.is_file():
        print("Error: quant.sf not found – run salmon quant first!")
        return None, 0, 0

    df = pd.read_csv(QUANT_FILE, sep='\t')
    gene_df = df[df['Name'].str.contains(gene_name, case=False, na=False)]

    if gene_df.empty:
        print(f"No transcripts quantified for {gene_name}")
        return None, 0, 0

    best = gene_df.loc[gene_df['TPM'].idxmax()]
    best_enst = best['Name'].split('|')[0]  # ENST000...
    best_tpm = best['TPM']

    total_tpm = gene_df['TPM'].sum()
    pct = (best_tpm / total_tpm) * 100 if total_tpm > 0 else 0

    if best_tpm < MIN_TPM_ABS or pct < MIN_TPM_PCT:
        print(f"{gene_name}: Best TPM {best_tpm:.1f} ({pct:.1f}%) too low → skipped")
        return None, best_tpm, pct

    print(f"{gene_name}: Selected {best_enst} (TPM {best_tpm:.1f}, {pct:.1f}% of gene)")
    return best_enst, best_tpm, pct

def extract_sequence(enst_id: str) -> str | None:
    """Get sequence from GENCODE FASTA"""
    for record in SeqIO.parse(GENCODE_FA, "fasta"):
        if enst_id in record.id:
            return str(record.seq)
    print(f"Sequence not found for {enst_id}")
    return None

def run_alignment(gene_name: str, ref_seq: str):
    """Run your alignment script with temp FASTA"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp.write(f">{gene_name}\n")
        for i in range(0, len(ref_seq), 60):
            tmp.write(ref_seq[i:i+60] + '\n')
        tmp_path = Path(tmp.name)

    cmd = [
        "python3", str(ALIGN_SCRIPT),
        str(FASTQ),
        gene_name.upper(),
        str(tmp_path)
    ]

    print(f"\n=== Running alignment for {gene_name} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout.strip())
    if result.stderr.strip():
        print("Errors / Warnings:")
        print(result.stderr.strip())

    # Move CSV to results folder
    expected_csv = Path(f"{gene_name.lower()}_ribosome_counts.csv")
    if expected_csv.is_file():
        target = RESULTS_DIR / expected_csv.name
        shutil.move(expected_csv, target)
        print(f"CSV saved: {target}")
    else:
        print(f"Warning: CSV not created for {gene_name}")

    tmp_path.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Batch Ribo-seq with best isoform selection')
    parser.add_argument('--gene', help='Run only one gene (for testing)')
    parser.add_argument('--force', action='store_true', help='Run even if TPM low')

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    genes_to_run = [args.gene.upper()] if args.gene else GENES

    print(f"Processing {len(genes_to_run)} genes → results saved to {RESULTS_DIR}\n")

    for gene in genes_to_run:
        enst, tpm, pct = get_best_isoform(gene)
        if enst is None:
            if not args.force:
                continue
            print(f"Forcing {gene} – using fallback if available (currently skipped)")

        seq = extract_sequence(enst)
        if seq is None:
            continue

        run_alignment(gene, seq)

    print("\nAll genes processed. Check data/results/ for CSVs.")

if __name__ == '__main__':
    main()