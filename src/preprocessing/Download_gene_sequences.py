"""
Download gene CDS sequences from NCBI RefSeq
"""

import time
import urllib.request
import argparse
import os
from pathlib import Path

GENES = GENES = [
    ("GAPDH", "NM_002046"),
    ("LDHA", "NM_005566"),
    ("LDHB", "NM_002300"),
    ("ENO1", "NM_001428"),
    ("PKM", "NM_002654"),
    ("ALDOA", "NM_184041"),
    ("PGK1", "NM_000291"),
    ("IDH1", "NM_005896"),
    ("IDH2", "NM_002168"),
    ("MDH2", "NM_005918"),
    ("G6PD", "NM_000402"),
    ("PFKL", "NM_002626"),
    ("PFKM", "NM_000289"),
    ("ACO2", "NM_001098"),
    ("CS", "NM_004077"),
    ("SDHA", "NM_004168"),
    ("FH", "NM_000143"),
    ("GOT2", "NM_002080"),
    ("ACTB", "NM_001101"),
    ("ACTG1", "NM_001614"),
    ("TUBB", "NM_178014"),
    ("TUBA1B", "NM_006082"),
    ("TUBA1A", "NM_006009"),
    ("VIM", "NM_003380"),
    ("LMNA", "NM_170707"),
    ("LMNB1", "NM_005573"),
    ("EEF1A1", "NM_001402"),
    ("EEF2", "NM_001961"),
    ("EIF4A1", "NM_001416"),
    ("EIF3A", "NM_003750"),
    ("EIF2S1", "NM_004094"),
    ("EEF1G", "NM_001404"),
    ("HSP90AA1", "NM_005348"),
    ("HSP90AB1", "NM_007355"),
    ("HSPA8", "NM_006597"),
    ("HSPA5", "NM_005347"),
    ("HSPD1", "NM_002156"),
    ("CCT2", "NM_006431"),
    ("CCT3", "NM_005998"),
    ("CCT4", "NM_006430"),
    ("DNAJA1", "NM_001539"),
    ("DNAJB1", "NM_006145"),
    ("MAPK1", "NM_002745"),
    ("MAPK3", "NM_002746"),
    ("AKT1", "NM_001014431"),
    ("AKT2", "NM_001626"),
    ("PRKACA", "NM_002730"),
    ("GSK3B", "NM_002093"),
    ("SRC", "NM_005417"),
    ("EGFR", "NM_005228"),
    ("PIK3CA", "NM_006218"),
    ("HNRNPA1", "NM_002136"),
    ("HNRNPC", "NM_004500"),
    ("HNRNPK", "NM_002140"),
    ("RFC1", "NM_002913"),
    ("TOP1", "NM_003286"),
    ("PARP1", "NM_001618"),
    ("JUN", "NM_002228"),
    ("FOS", "NM_005252"),
    ("MYC", "NM_002467"),
    ("TP53", "NM_000546"),
    ("STAT3", "NM_003150"),
    ("NFKB1", "NM_003998"),
    ("CCNB1", "NM_031966"),
    ("CCNE1", "NM_001238"),
    ("CDC20", "NM_001255"),
    ("PLK1", "NM_005030"),
    ("AURKB", "NM_004217"),
    ("CASP7", "NM_001227"),
    ("CTSB", "NM_001908"),
    ("CTSD", "NM_001909"),
    ("PSMD1", "NM_002807"),
    ("PSMD2", "NM_002808"),
    ("CAT", "NM_001752"),
    ("TXNRD1", "NM_003330"),
    ("VCP", "NM_007126"),
    ("NSF", "NM_006178"),
    ("ALB", "NM_000477"),
    ("SERPINA1", "NM_000295"),
    ("F9", "NM_000133"),
    ("INSR", "NM_000208"),
    ("NCL", "NM_005381"),
    ("RPLP0", "NM_001002"),
]


def download_sequence(refseq_id: str) -> str | None:
    """Download CDS sequence from NCBI (nur die Sequenz, ohne Header)"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={refseq_id}&rettype=fasta_cds_na&retmode=text"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8').strip()
            if not data.startswith('>'):
                return None
            # remove header
            lines = data.split('\n')
            sequence = ''.join(lines[1:])
            return sequence
    except Exception as e:
        print(f"  Error downloading {refseq_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download individual gene CDS sequences from NCBI')
    parser.add_argument('--delay', type=float, default=0.4, 
                        help='Delay between requests in seconds (NCBI rate limit)')
    parser.add_argument('--output-dir', type=str, default="data/genes",
                        help='Base output directory (will create subfolder "genes")')

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed = []

    for i, (gene_name, refseq_id) in enumerate(GENES, 1):
        print(f"[{i}/{len(GENES)}] {gene_name} ({refseq_id}) ... ", end="")

        seq = download_sequence(refseq_id)

        if seq and len(seq) > 20: 
            filename = f"{gene_name}_{refseq_id}.fasta"
            filepath = out_dir / filename

            with open(filepath, 'w') as f:
                f.write(f">{gene_name}|{refseq_id}\n")
                for j in range(0, len(seq), 60):
                    f.write(seq[j:j+60] + '\n')

            print(f"✓  ({len(seq)} bp) → {filename}")
            success_count += 1
        else:
            print("✗ Failed")
            failed.append((gene_name, refseq_id))

        time.sleep(args.delay)

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Genes total:     {len(GENES)}")
    print(f"Successful:     {success_count}")
    print(f"Failed:  {len(failed)}")

    if failed:
        print("\nFailed Genes:")
        for g, acc in failed:
            print(f"  {g:<12} {acc}")


if __name__ == '__main__':
    main()