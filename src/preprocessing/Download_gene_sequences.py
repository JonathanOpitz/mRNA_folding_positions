#!/usr/bin/env python3
"""
Download all protein-coding transcript isoforms (cDNA with UTRs) from Ensembl/GENCODE
for a list of human genes using the REST API.

Output: data/genes/GENE_ENSTxxxxxxxx.fasta (one file per transcript)
"""

import time
import requests
import argparse
import os
import json
from pathlib import Path

# Gene list
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

ENSEMBL_SERVER = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json"}

def lookup_gene_id(gene_symbol):
    """Find Ensembl Gene ID (ENSG...) for a human gene symbol."""
    ext = f"/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
    try:
        r = requests.get(ENSEMBL_SERVER + ext, headers=HEADERS, timeout=10)
        if not r.ok:
            r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data = data[0]
        return data.get('id')
    except Exception as e:
        print(f"  Error looking up {gene_symbol}: {e}")
        return None

def get_transcripts_for_gene(gene_id):
    """Fetch all protein_coding transcripts (ENST...) for a gene ID."""
    ext = f"/overlap/id/{gene_id}?feature=transcript;biotype=protein_coding;content-type=application/json"
    try:
        r = requests.get(ENSEMBL_SERVER + ext, headers=HEADERS, timeout=15)
        if not r.ok:
            r.raise_for_status()
        transcripts = r.json()
        return [t['transcript_id'] for t in transcripts if 'transcript_id' in t]
    except Exception as e:
        print(f"  Error fetching transcripts for {gene_id}: {e}")
        return []

def download_transcript_sequence(transcript_id):
    """Download cDNA sequence (including UTRs) for a transcript (ENST...)."""
    ext = f"/sequence/id/{transcript_id}?type=cdna;content-type=text/x-fasta"
    try:
        r = requests.get(ENSEMBL_SERVER + ext, headers=HEADERS, timeout=10)
        if not r.ok:
            r.raise_for_status()
        data = r.text.strip()
        if not data.startswith('>'):
            return None

        # Remove header and extract sequence
        lines = data.split('\n')
        seq = ''.join(lines[1:]).replace('\n', '')
        return seq
    except Exception as e:
        print(f"  Error downloading {transcript_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Download all protein-coding transcript isoforms from Ensembl/GENCODE'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.35,
        help='Delay between requests (NCBI/Ensembl rate limit)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="data/genes",
        help='Output directory'
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_transcripts = 0
    failed_genes = []

    for i, gene_name in enumerate(GENES, 1):
        print(f"[{i}/{len(GENES)}] {gene_name} ... ", end="")

        gene_id = lookup_gene_id(gene_name)
        if not gene_id:
            print("✗ Gene ID not found")
            failed_genes.append(gene_name)
            time.sleep(args.delay)
            continue

        transcripts = get_transcripts_for_gene(gene_id)
        if not transcripts:
            print("✗ No protein-coding transcripts found")
            failed_genes.append(gene_name)
            time.sleep(args.delay)
            continue

        print(f"✓ {len(transcripts)} transcripts found")

        gene_success = 0
        for trans_id in transcripts:
            seq = download_transcript_sequence(trans_id)
            if seq and len(seq) > 20:
                filename = f"{gene_name}_{trans_id}.fasta"
                filepath = out_dir / filename
                with open(filepath, 'w') as f:
                    f.write(f">{gene_name}|{trans_id}\n")
                    for j in range(0, len(seq), 60):
                        f.write(seq[j:j+60] + '\n')

                print(f"    → {filename} ({len(seq)} bp)")
                gene_success += 1
                total_transcripts += 1
            else:
                print(f"    ✗ Failed: {trans_id}")

            time.sleep(args.delay)

        if gene_success > 0:
            success_count += 1
        else:
            failed_genes.append(gene_name)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Genes processed:     {len(GENES)}")
    print(f"Genes with success:  {success_count}")
    print(f"Total transcripts:   {total_transcripts}")
    print(f"Failed genes:        {len(failed_genes)}")

    if failed_genes:
        print("\nFailed Genes:")
        for g in failed_genes:
            print(f"  - {g}")

if __name__ == '__main__':
    main()