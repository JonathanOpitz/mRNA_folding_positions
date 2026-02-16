#!/usr/bin/env python3
"""
Batch process all genes - ribosome profiling
Processes .fa and .fasta files in data/genes/

Usage: python3 batch_process_genes.py
"""

import os
import sys
import subprocess
from pathlib import Path


# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────

BASE_DIR   = "/Users/jonathanopitz/Desktop/Master"
FASTQ      = f"{BASE_DIR}/data/SRR10665827.fastq"
GENES_DIR  = f"{BASE_DIR}/data/genes"
SCRIPT     = f"{BASE_DIR}/src/preprocessing/fast_ribo_analysis.py"
OUTPUT_DIR = f"{BASE_DIR}/data/results/gene_counts"


def find_gene_files():
    """Find all .fa / .fasta files (case insensitive)"""
    patterns = ["*.fa", "*.fasta", "*.FA", "*.FASTA"]
    files = []
    for pat in patterns:
        files.extend(Path(GENES_DIR).glob(pat))
    return sorted(set(files))  # remove possible duplicates


def check_requirements():
    if not Path(FASTQ).is_file():
        print(f"ERROR: FASTQ missing\n  {FASTQ}")
        return False

    if not Path(GENES_DIR).is_dir():
        print(f"ERROR: Genes directory missing\n  {GENES_DIR}")
        return False

    if not Path(SCRIPT).is_file():
        print(f"ERROR: Analysis script missing\n  {SCRIPT}")
        return False

    gene_files = find_gene_files()
    if not gene_files:
        print(f"ERROR: No .fa / .fasta files found in\n  {GENES_DIR}")
        print("Run this to see what's there:")
        print(f"  ls -l '{GENES_DIR}'")
        return False

    print(f"FASTQ  : {FASTQ}")
    print(f"Genes  : {len(gene_files)} files found")
    print(f"Script : {SCRIPT}")
    print(f"Output : {OUTPUT_DIR}")
    return True


def main():
    print("BATCH RIBOSOME PROFILING")
    print("-" * 50)

    if not check_requirements():
        sys.exit(1)

    gene_files = find_gene_files()
    print(f"Will process {len(gene_files)} genes\n")

    input("Press ENTER to start (Ctrl+C = cancel) ... ")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = []

    for i, gene_path in enumerate(gene_files, 1):
        gene_name = gene_path.stem
        print(f"[{i:3d}/{len(gene_files):3d}] {gene_name}")

        cmd = [
            "python3", SCRIPT,
            FASTQ,
            gene_name.upper(),
            str(gene_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600          # 10 Minuten pro Gen – bei Bedarf erhöhen
            )

            # Das Skript erzeugt Datei im aktuellen Verzeichnis
            csv_name = f"{gene_name.lower()}_ribosome_counts.csv"
            src_path = Path(csv_name)

            if src_path.is_file():
                dst_path = Path(OUTPUT_DIR) / csv_name
                src_path.rename(dst_path)
                success += 1
                print("  → OK")
            else:
                print("  → FAILED: output file not created")
                if result.stderr.strip():
                    print("  Error output was:")
                    print(result.stderr.strip()[:300])
                failed.append(gene_name)

        except subprocess.TimeoutExpired:
            print("  → FAILED: timeout (>10 min)")
            failed.append(gene_name)
        except Exception as e:
            print(f"  → FAILED: {e}")
            failed.append(gene_name)

    # ── Summary ─────────────────────────────────────
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total genes   : {len(gene_files)}")
    print(f"Successful    : {success}")
    print(f"Failed        : {len(failed)}")

    if failed:
        print("\nFailed genes:")
        for g in failed:
            print(f"  - {g}")

    print("\nResults in:")
    print(f"  {OUTPUT_DIR}")
    print(f"  → {len(list(Path(OUTPUT_DIR).glob('*.csv')))} CSV files")

    print("\nDONE.")


if __name__ == '__main__':
    main()