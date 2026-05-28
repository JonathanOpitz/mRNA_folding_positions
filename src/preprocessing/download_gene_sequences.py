#!/usr/bin/env python3
"""
Filter cDNA sequences from the Ensembl FASTA for a list of gene symbols.
Writes one .fasta file per gene + transcript.
"""

import logging
import re
from pathlib import Path
import argparse
import gzip

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Target genes — edit this list before running
GENES = [
    "BRCA1", "MTOR", "JAK2", "INS", "EPO", "IFNB1", "IL2", "GH1", "F8", "PROC",
    "HBB", "APOA1", "TTR", "VEGFA", "TGFB1", "FGA",]



GENES_SET = {g.upper() for g in GENES}  # case-insensitive matching

def parse_header(header: str) -> tuple[str, str, str]:
    """Extract gene_symbol, transcript_id, and full description from an Ensembl FASTA header."""
    m = re.search(r'gene_symbol:(\S+)', header)
    gene = m.group(1) if m else None

    m_trans = re.search(r'^>(\S+)', header)
    trans_id = m_trans.group(1) if m_trans else "unknown"

    desc = header[1:].strip()
    return gene, trans_id, desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_file", help="Path to the unzipped Ensembl cdna.all.fa file")
    parser.add_argument("--output-dir", default="data/genes", help="Output directory")
    args = parser.parse_args()

    fasta_path = Path(args.fasta_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_gene = None
    current_trans = None
    current_seq = []
    written = set()

    log.info("Filtering %d genes from %s ...", len(GENES), fasta_path)

    with fasta_path.open("rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Flush the previous entry
                if current_gene and current_gene.upper() in GENES_SET:
                    seq = "".join(current_seq)
                    if len(seq) > 20:
                        filename = f"{current_gene}_{current_trans}.fasta"
                        filepath = out_dir / filename
                        with filepath.open("w") as out:
                            out.write(f">{current_gene}|{current_trans} {current_desc}\n")
                            for i in range(0, len(seq), 60):
                                out.write(seq[i:i+60] + "\n")
                        log.info("  → %s  (%d bp)", filename, len(seq))
                        written.add(current_gene)

                current_gene, current_trans, current_desc = parse_header(line)
                current_seq = []
            else:
                if current_gene and current_gene.upper() in GENES_SET:
                    current_seq.append(line)

    # Letzten Eintrag nicht vergessen
    if current_gene and current_gene.upper() in GENES_SET:
        seq = "".join(current_seq)
        if len(seq) > 20:
            filename = f"{current_gene}_{current_trans}.fasta"
            filepath = out_dir / filename
            with filepath.open("w") as out:
                out.write(f">{current_gene}|{current_trans} {current_desc}\n")
                for i in range(0, len(seq), 60):
                    out.write(seq[i:i+60] + "\n")
            log.info("  → %s  (%d bp)", filename, len(seq))
            written.add(current_gene)

    log.info("\nDone. Found transcripts for %d / %d genes.", len(written), len(GENES))
    missing = sorted(set(GENES) - {g for g in written})
    if missing:
        log.warning("Not found: %s", ", ".join(missing))


if __name__ == "__main__":
    main()