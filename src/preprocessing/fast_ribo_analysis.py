#!/usr/bin/env python3
"""
FAST Ribosome Profiling - k-mer based gene-specific read extraction

Usage:
    python3 fast_ribo_analysis.py <fastq_file> <gene_name> <reference_seq_file>

Example:
    python3 fast_ribo_analysis.py SRR10513636.fastq GAPDH GAPDH.fa
"""

import sys
from collections import Counter, defaultdict
from Bio.Seq import Seq

P_SITE_OFFSET = 13
MIN_LENGTH = 26
MAX_LENGTH = 34
KMER_SIZE = 15
MAX_MISMATCHES = 3

ADAPTERS = [
    "AGATCGGAAGAGCACACGTCT",
    "AGATCGGAAGAGCGTCGTGTA",
]


def load_reference_sequence(ref_path, gene_name):
    seq_lines = []
    with open(ref_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            seq_lines.append(line)

    seq = ''.join(seq_lines).upper().replace('U', 'T')

    if not seq:
        sys.exit(f"Error: no sequence found in {ref_path}")

    print(f"Reference: {gene_name}")
    print(f"  File:    {ref_path}")
    print(f"  Length:  {len(seq)} nt")
    print(f"  Codons:  {len(seq)//3}\n")

    return seq


def build_kmer_index(seq, k=KMER_SIZE):
    print(f"Building {k}-mer index...")

    index = defaultdict(list)

    # forward
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        index[kmer].append(('f', i))

    # reverse complement
    rc = str(Seq(seq).reverse_complement())
    for i in range(len(rc) - k + 1):
        kmer = rc[i:i+k]
        index[kmer].append(('r', i))

    print(f"  {len(index)} unique kmers")
    print(f"  avg {sum(len(v) for v in index.values()) / len(index):.1f} positions per kmer\n")
    return index


def trim_adapter(read):
    for adapter in ADAPTERS:
        pos = read.find(adapter)
        if pos != -1:
            return read[:pos]
    return read


def fast_align(read, ref_seq, kmer_index, k=KMER_SIZE):
    if len(read) < k:
        return None, None, None

    read_kmers = []
    for off in [0, len(read)//4, len(read)//2, 3*len(read)//4]:
        if off + k <= len(read):
            read_kmers.append((off, read[off:off+k]))

    candidates = []
    for off, kmer in read_kmers:
        if kmer in kmer_index:
            for strand, pos in kmer_index[kmer]:
                candidates.append((pos - off, strand))

    if not candidates:
        return None, None, None

    best_pos = None
    best_mm = MAX_MISMATCHES + 1
    best_strand = None

    for pos, strand in set(candidates):
        if strand == 'f':
            if pos < 0 or pos + len(read) > len(ref_seq):
                continue
            window = ref_seq[pos:pos+len(read)]
            mm = sum(a != b for a, b in zip(read, window))
        else:
            read_rc = str(Seq(read).reverse_complement())
            if pos < 0 or pos + len(read_rc) > len(ref_seq):
                continue
            window = ref_seq[pos:pos+len(read_rc)]
            mm = sum(a != b for a, b in zip(read_rc, window))

        if mm < best_mm:
            best_mm = mm
            best_pos = pos
            best_strand = '+' if strand == 'f' else '-'

    if best_mm <= MAX_MISMATCHES:
        return best_pos, best_mm, best_strand

    return None, None, None


def process_fastq(fastq_path, ref_seq, kmer_index, gene_name):
    print(f"Processing: {fastq_path}\n")

    codon_counts = Counter()
    total = trimmed = passed_len = aligned = 0

    with open(fastq_path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # +
            f.readline()  # qual

            total += 1

            seq = trim_adapter(seq)
            if len(seq) < len(seq):
                trimmed += 1

            if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH):
                continue
            passed_len += 1

            pos, mm, strand = fast_align(seq, ref_seq, kmer_index)
            if pos is not None:
                aligned += 1

                if strand == '+':
                    ppos = pos + P_SITE_OFFSET
                else:
                    ppos = pos + len(seq) - P_SITE_OFFSET

                codon_idx = ppos // 3
                if 0 <= codon_idx < (len(ref_seq)//3):
                    codon_counts[codon_idx] += 1

            if total % 100_000 == 0:
                print(f"{total:,} reads | trim:{trimmed:,} | len-ok:{passed_len:,} | gene:{aligned:,}", end='\r')

    print(f"\n\nStatistics:")
    print(f"  total reads         : {total:,}")
    print(f"  adapter trimmed     : {trimmed:,} ({trimmed/total*100:.2f}%)")
    print(f"  length passed       : {passed_len:,} ({passed_len/total*100:.2f}%)")
    print(f"  aligned to {gene_name:<10}: {aligned:,} ({aligned/total*100:.4f}%)")
    if passed_len > 0:
        print(f"    → of length-passed: {aligned/passed_len*100:.2f}%")
    print(f"  codons with ≥1 read : {len(codon_counts)}")
    print("-"*60 + "\n")

    return codon_counts


def get_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq)-len(seq)%3, 3)]


def save_results(codon_counts, ref_seq, gene_name, outfile):
    print(f"Writing: {outfile}")

    codons = get_codons(ref_seq)
    total_mapped = sum(codon_counts.values())
    n_codons = len(codons)
    mean = total_mapped / n_codons if n_codons > 0 else 0

    with open(outfile, 'w') as f:
        f.write("codon_pos,codon,nt_start,nt_end,count,rel_occupancy,density,rpm\n")
        for i, codon in enumerate(codons):
            cnt = codon_counts.get(i, 0)
            start = i * 3
            rel = cnt / total_mapped if total_mapped > 0 else 0
            dens = cnt / mean if mean > 0 else 0
            rpm = (cnt / total_mapped) * 1e6 if total_mapped > 0 else 0
            f.write(f"{i+1},{codon},{start},{start+2},{cnt},{rel:.6f},{dens:.2f},{rpm:.2f}\n")

    print(f"  codons total     : {n_codons}")
    print(f"  codons with reads: {sum(1 for v in codon_counts.values() if v > 0)}")

    if codon_counts:
        print("\nTop 5 covered codons:")
        for idx, cnt in codon_counts.most_common(5):
            print(f"  {idx+1:4d}  {codons[idx]}  {cnt:6d} reads")


def main():
    if len(sys.argv) != 4:
        print(__doc__.strip())
        sys.exit(1)

    fastq = sys.argv[1]
    gene  = sys.argv[2].upper()
    ref   = sys.argv[3]

    outfile = f"{gene.lower()}_ribosome_counts.csv"

    print(f"Input FASTQ : {fastq}")
    print(f"Gene        : {gene}")
    print(f"Reference   : {ref}")
    print(f"Output      : {outfile}")
    print("-"*60 + "\n")

    ref_seq = load_reference_sequence(ref, gene)
    index = build_kmer_index(ref_seq)
    counts = process_fastq(fastq, ref_seq, index, gene)
    save_results(counts, ref_seq, gene, outfile)

    print("Done.")


if __name__ == '__main__':
    main()