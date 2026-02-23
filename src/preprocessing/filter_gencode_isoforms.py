#!/usr/bin/env python3
"""
Filter GENCODE pc_transcripts.fa for your genes → one .fasta per transcript isoform
Improved header parsing for gene symbols.
"""

import argparse
from pathlib import Path
from Bio import SeqIO
import re

GENES = {
    g.upper() for g in [
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
}

def main():
    parser = argparse.ArgumentParser(description='Filter GENCODE transcripts for specific genes')
    parser.add_argument('--gencode_fa', required=True, help='Path to gencode.vXX.pc_transcripts.fa')
    parser.add_argument('--output_dir', default="data/genes", help='Output directory for per-isoform fastas')

    args = parser.parse_args()
    gencode_path = Path(args.gencode_fa).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not gencode_path.is_file():
        print(f"ERROR: GENCODE FASTA not found: {gencode_path}")
        return 1

    print(f"Parsing {gencode_path} ...")
    print("Header format example (first few):")

    gene_to_trans = {g: [] for g in GENES}
    total_parsed = 0
    matched = 0
    debug_headers = 0

    for record in SeqIO.parse(gencode_path, "fasta"):
        total_parsed += 1
        header = record.id
        desc = record.description

        # Debug: Show first 10 headers
        if debug_headers < 10:
            print(f"  Header: {desc}")
            debug_headers += 1

        # Search for gene symbol in description 
        desc_upper = desc.upper()
        for gene in GENES:
            # Match whole word or with suffix like -201
            if re.search(rf'\b{gene}\b|\b{gene}-\d{{3}}\b', desc_upper):
                gene_to_trans[gene].append((header, str(record.seq)))
                matched += 1
                if matched % 100 == 0:
                    print(f"  Found {header} for {gene}")
                break

        if total_parsed % 10000 == 0:
            print(f"  Parsed {total_parsed:,} transcripts...")

    print(f"\nDone. Parsed {total_parsed:,} transcripts, matched {matched:,}")

    # Write files
    for gene, transcripts in gene_to_trans.items():
        if not transcripts:
            print(f"No transcripts found for {gene}")
            continue

        print(f"{gene}: {len(transcripts)} transcripts")
        for trans_id, seq in transcripts:
            clean_id = trans_id.split('.')[0]  
            filename = f"{gene}_{clean_id}.fasta"
            filepath = out_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{gene}|{trans_id}\n")
                for j in range(0, len(seq), 60):
                    f.write(seq[j:j+60] + '\n')
            print(f"  Wrote {filename} ({len(seq)} bp)")

    print("\nAll done. Ready for isoform selection / alignment.")

if __name__ == '__main__':
    main()