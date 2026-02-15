#!/usr/bin/env python3
"""
FAST Ribosome Profiling - using k-mer hashing
Extracts only reads matching a specific gene!

Usage:
    python3 fast_ribo_analysis.py <fastq_file> <gene_name>

Example:
    python3 fast_ribo_analysis.py SRR10513636.fastq GAPDH
"""

import sys
from collections import Counter, defaultdict
from Bio.Seq import Seq

# Parameters
P_SITE_OFFSET = 12
MIN_LENGTH = 20
MAX_LENGTH = 32
KMER_SIZE = 15  # Länge der k-mere für hashing
MAX_MISMATCHES = 2

# Adapters
ADAPTERS = [
    "AGATCGGAAGAGCACACGTCT",
    "AGATCGGAAGAGCGTCGTGTA",
]

# Gene sequences
GENE_SEQUENCES = {
    'GAPDH': """
        ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACC
        AGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATT
        GACCTCAACTACATGGTTTACATGTTCCAATATGATTCCACCCATGGCAAATTCCAT
        GGCACCGTCAAGGCTGAGAACGGGAAGCTTGTCATCAATGGAAATCCCATCACCATC
        TTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAGTACGTC
        GTGGAGTCCACTGGCGTCTTCACCACCATGGAGAAGGCTGGGGCTCATTTGCAGGGG
        GGAGCCAAAAGGGTCATCATCTCTGCCCCCTCTGCTGATGCCCCCATGTTCGTCATG
        GGTGTGAACCATGAGAAGTATGACAACAGCCTCAAGATCATCAGCAATGCCTCCTGC
        ACCACCAACTGCTTAGCACCCCTGGCCAAGGTCATCCATGACAACTTTGGTATCGTG
        GAAGGACTCATGACCACAGTCCATGCCATCACTGCCACCCAGAAGACTGTGGATGGC
        CCCTCCGGGAAACTGTGGCGTGATGGCCGCGGGGCTCTCCAGAACATCATCCCTGCC
        TCTACTGGCGCTGCCAAGGCTGTGGGCAAGGTCATCCCTGAGCTGAACGGGAAGCTC
        ACTGGCATGGCCTTCCGTGTCCCCACTGCCAACGTGTCAGTGGTGGACCTGACCTGC
        CGTCTAGAAAAACCTGCCAAATATGATGACATCAAGAAGGTGGTGAAGCAGGCGTCG
        GAGGGCCCCCTCAAGGGCATCCTGGGCTACACTGAGCACCAGGTGGTCTCCTCTGAC
        TTCAACAGCGACACCCACTCCTCCACCTTTGACGCTGGGGCTGGCATTGCCCTCAAC
        GACCACTTTGTCAAGCTCATTTCCTGGTATGACAACGAATTTGGCTACAGCAACAGG
        GTGGTGGACCTCATGGCCCACATGGCCTCCAAGGAGTAA
    """,
    'ACTB': """
        ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAGGCC
        GGCTTCGCGGGCGACGATGCCCCCCGGGCCGTCTTCCCCTCCATCGTGGGGCGCCCC
        AGGCACCAGGGCGTGATGGTGGGCATGGGTCAGAAGGATTCCTATGTGGGCGACGAG
        GCCCAGAGCAAGAGAGGCATCCTCACCCTGAAGTACCCCATCGAGCACGGCATCGTC
        ACCAACTGGGACGACATGGAGAAAATCTGGCACCACACCTTCTACAATGAGCTGCGT
        GTGGCTCCCGAGGAGCACCCCGTGCTGCTGACCGAGGCCCCCCTGAACCCCAAGGCC
        AACCGCGAGAAGATGACCCAGATCATGTTTGAGACCTTCAACACCCCAGCCATGTAC
        GTTGCTATCCAGGCTGTGCTATCCCTGTACGCCTCTGGCCGTACCACTGGCATCGTG
        ATGGACTCCGGTGACGGGGTCACCCACACTGTGCCCATCTACGAGGGGTATGCCCTG
        CCCACACACATGCCACACCCAGCCATGTATGTTGCTATCCAGGCTGTGCTATCCCTG
        TACGCCTCTGGCCGTACCACTGGCATCGTGATGGACTCCGGTGACGGGGTCACCCAC
        ACTGTGCCCATCTACGAGGGGTATGCCCTGCCCACATAA
    """,
    'F9': """
        ATGCAGCGCGTGAACATGATCATGGCAGAATCAACCAACTTTGTCCTCTGCCTGGTG
        ATTGCCATTCTCTTGATGGCCAGCTTTACCTTGAAGAAATGGTCAGTCGCCAAGGTG
        AAGGATGATGAGAGGCTGTGTTGCCTTGAAGGAAGTGGACACCGGAACTACTTTCCT
        GACCTTATGGAATTTCAAGGACAAGGAGACTCCTGATGGCATCATGTTGACCACCAA
        CCTGGGCAAGAACTTCATCGGCAGCACCTACGTGACCAGCTTCAAGGAGTGCAACAA
        GATCCTCAAGGGCTCCTTGAAATGCACCAAGTACCCCTTGTCCAGCTGTGGCTTCAC
        GGTGTTCAACACCAACTTCTCCGTGGAGCACCGCTTCAAGTTCAAGAACAACAACTT
        CACCATCCCCGAGCTGGCCTGGGACGTGACGGATGACTTCCGCGTGCTGCACTTCAG
        CTTGCCGCCCGAGACCTTCTGGGACCAGGTCATCCAGGCCAGCCAGACCATCACCTT
        CGGACCTGTCACCGCCATCAACGCCTACATCGTGGCCAACCTGCAGTGCAACGGCTG
        CACCAACCTGTTCAACATCAACTCCCTGGTGCCGGAGGTGTCCCACAACAACATCTT
        CGTGAAGGGCAACTTCTCCCAGGCCAACTGGACAGTGACGGGCAACACGTGCGAGTA
        TAGCGGCTACAACGTGCCTTTCTCCCGGAATTATAAGGCTCAGCGCGCCATCCTCGT
        GCACCGGGGCATGAACTGGACCGGCAACTACGGCTACTTCACCTTCAGCCACAAGAA
        GTGCAACCGGGGCACCTTCTCCTACAAGACCGGCACGGGCTCCAACTTCACCTACCA
        GAACGGCATCATCCAGTTTCTGATCAACAAGACCACCGGCAAGCCCTTCACCTTCCA
        GGTCATGGGCTCCCGGACCCTGTACAGGGTGTCCCTGAACCGCACGGTGTTCACGCT
        GGGGAACGCCTGCTTCGAGAACAACTGGACCTGCTACGAGACCAACAACACCCCGGA
        GCTGACAGGCCGAGACAAGAACACCGAGATCTAA
    """,
    
    'F9_CO': """
        ATGCAGCGGGTCAATATGATCATGGCCGAGTCCACAAACTTCGTGCTGTGCCTGGTG
        ATCGCCATCCTGCTGATGGCCTCCTTCACCCTGAAGAAGTGGTCCGTGGCCAAGGTG
        AAGGACGACGAGCGGCTGTGCTGCCTGGAAGGAAGTGGACACAGGCACAACCTTCCT
        GACCCTGTGGAACTTCAAGGACAAGGAAACACCAGATGGCATCATGCTGACCACAAA
        CCTGGGCAAGAACTTCATCGGCTCCACATACGTGACATCCTTCAAGGAGTGCAACAA
        GATCCTGAAGGGCTCCCTGAAGTGCACCAAGTACCCACTGTCCAGCTGCGGATTCAC
        AGTGTTCAACACCAACTTCTCCGTGGAACACAGATTCAAGTTCAAGAACAACAACTT
        CACCATCCCAGAACTGGCCTGGGACGTGACAGACGACTTCAGAGTGCTGCACTTCTC
        CCTGCCACCAGAGACATTCTGGGACCAGGTGATCCAGGCCAGCCAGACCATCACCTT
        CGGACCAGTGACCGCCATCAACGCCTACATCGTGGCCAACCTGCAGTGCAACGGCTG
        CACCAACCTGTTCAACATCAACTCCCTGGTGCCAGAGGTGTCCCACAACAACATCTT
        CGTGAAGGGCAACTTCTCCCAGGCCAACTGGACAGTGACCGGCAACACATGCGAGTA
        CTCCGGCTACAACGTGCCATTCTCCAGGAACTATAAGGCCCAGAGAGCCATCCTGGT
        GCACAGGGGCATGAACTGGACAGGCAACTACGGCTACTTCACCTTCTCCCACAAGAA
        GTGCAACAGGGGCACCTTCTCCTACAAGACAGGCACAGGCTCCAACTTCACCTACCA
        GAACGGCATCATCCAGTTCCTGATCAACAAGACCACAGGCAAGCCATTCACCTTCCA
        GGTGATGGGCTCCAGGACCCTGTACAGGGTGTCCCTGAACAGAACAGTGTTCACACT
        GGGCAACGCCTGCTTCGAGAACAACTGGACCTGCTACGAGACCAACAACACCCCAGA
        ACTGACAGGCAGAGACAAGAACACCGAGATCTAA
    """
}


def load_gene_sequence(gene_name):
    """Loads gene sequence"""
    if gene_name.upper() not in GENE_SEQUENCES:
        raise ValueError(f"Gene {gene_name} not available")
    
    seq = GENE_SEQUENCES[gene_name.upper()]
    seq = ''.join(seq.split()).upper().replace('U', 'T')
    
    print(f"Gene: {gene_name}")
    print(f"Sequence length: {len(seq)} nt")
    print(f"Codons: {len(seq)//3}")
    
    return seq


def build_kmer_index(gene_seq, k=KMER_SIZE):
    """
    Builds k-mer hash index for fast lookup
    Returns: dict {kmer: [positions]}
    """
    print(f"Building k-mer index (k={k})...")
    
    kmer_index = defaultdict(list)
    
    # Forward strand
    for i in range(len(gene_seq) - k + 1):
        kmer = gene_seq[i:i+k]
        kmer_index[kmer].append(('forward', i))
    
    # Reverse complement strand
    gene_rc = str(Seq(gene_seq).reverse_complement())
    for i in range(len(gene_rc) - k + 1):
        kmer = gene_rc[i:i+k]
        kmer_index[kmer].append(('reverse', i))
    
    print(f"  Index contains {len(kmer_index)} unique k-mers")
    print(f"  Average {sum(len(v) for v in kmer_index.values())/len(kmer_index):.1f} positions per k-mer")
    
    return kmer_index


def trim_adapter(seq):
    """Removes Illumina adapter from read"""
    for adapter in ADAPTERS:
        pos = seq.find(adapter)
        if pos != -1:
            return seq[:pos]
    return seq


def fast_align(read_seq, gene_seq, kmer_index, k=KMER_SIZE):
    """
    Fast alignment using k-mer lookup
    Returns: (position, mismatches, strand) or (None, None, None)
    """
    # Extract k-mer from read (use middle k-mer for best accuracy)
    if len(read_seq) < k:
        return None, None, None
    
    # Try multiple k-mers from the read
    read_kmers = []
    for offset in [0, len(read_seq)//4, len(read_seq)//2, 3*len(read_seq)//4]:
        if offset + k <= len(read_seq):
            read_kmers.append((offset, read_seq[offset:offset+k]))
    
    candidates = []
    
    # Search k-mers in index
    for kmer_offset, kmer in read_kmers:
        if kmer in kmer_index:
            for strand, gene_pos in kmer_index[kmer]:
                # Calculate where read would start
                read_start = gene_pos - kmer_offset
                candidates.append((read_start, strand))
    
    if not candidates:
        return None, None, None
    
    # Verify candidates (full comparison with mismatch check)
    best_pos = None
    best_mismatches = MAX_MISMATCHES + 1
    best_strand = None
    
    for pos, strand in set(candidates):  # unique candidates
        if strand == 'forward':
            if pos < 0 or pos + len(read_seq) > len(gene_seq):
                continue
            gene_window = gene_seq[pos:pos+len(read_seq)]
            mismatches = sum(1 for a, b in zip(read_seq, gene_window) if a != b)
        else:
            # Reverse complement
            read_rc = str(Seq(read_seq).reverse_complement())
            if pos < 0 or pos + len(read_rc) > len(gene_seq):
                continue
            gene_window = gene_seq[pos:pos+len(read_rc)]
            mismatches = sum(1 for a, b in zip(read_rc, gene_window) if a != b)
        
        if mismatches < best_mismatches:
            best_mismatches = mismatches
            best_pos = pos
            best_strand = '+' if strand == 'forward' else '-'
    
    if best_mismatches <= MAX_MISMATCHES:
        return best_pos, best_mismatches, best_strand
    
    return None, None, None


def process_fastq_fast(fastq_file, gene_seq, kmer_index, gene_name):
    """Processes FASTQ with fast k-mer alignment"""
    
    print(f"\n{'='*70}")
    print(f"Analyzing FASTQ: {fastq_file}")
    print(f"{'='*70}\n")
    
    codon_counts = Counter()
    total_reads = 0
    trimmed_reads = 0
    length_filtered = 0
    aligned_reads = 0
    
    with open(fastq_file, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break
            
            seq = f.readline().strip()
            plus = f.readline()
            qual = f.readline()
            
            total_reads += 1
            
            # Trim adapter
            seq_trimmed = trim_adapter(seq)
            if len(seq_trimmed) < len(seq):
                trimmed_reads += 1
            
            # Filter length
            if len(seq_trimmed) < MIN_LENGTH or len(seq_trimmed) > MAX_LENGTH:
                continue
            
            length_filtered += 1
            
            # Fast alignment with k-mer index
            pos, mismatches, strand = fast_align(seq_trimmed, gene_seq, kmer_index)
            
            if pos is not None:
                aligned_reads += 1
                
                # P-site position
                if strand == '+':
                    p_site_nt = pos + P_SITE_OFFSET
                else:
                    p_site_nt = pos + len(seq_trimmed) - P_SITE_OFFSET
                
                codon_num = p_site_nt // 3
                
                if 0 <= codon_num < len(gene_seq)//3:
                    codon_counts[codon_num] += 1
            
            # Progress
            if total_reads % 100000 == 0:
                print(f"Processed: {total_reads:,} | Trimmed: {trimmed_reads:,} | "
                      f"Length OK: {length_filtered:,} | Aligned: {aligned_reads:,}", end='\r')
    
    print(f"\n\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Total reads: {total_reads:,}")
    print(f"Adapter trimmed: {trimmed_reads:,} ({trimmed_reads/total_reads*100:.2f}%)")
    print(f"Correct length: {length_filtered:,} ({length_filtered/total_reads*100:.2f}%)")
    print(f"Aligned to {gene_name}: {aligned_reads:,} ({aligned_reads/total_reads*100:.4f}%)")
    if length_filtered > 0:
        print(f"  Alignment rate: {aligned_reads/length_filtered*100:.2f}% of length-filtered")
    print(f"Unique codons with reads: {len(codon_counts)}")
    print(f"{'='*70}\n")
    
    return codon_counts


def extract_codons(sequence):
    """Extracts codons from sequence"""
    codons = []
    for i in range(0, len(sequence) - len(sequence) % 3, 3):
        codons.append(sequence[i:i+3])
    return codons


def write_output(codon_counts, gene_seq, gene_name, output_file):
    """Writes CSV output"""
    
    print(f"Writing output: {output_file}")
    
    codons = extract_codons(gene_seq)
    total_mapped = sum(codon_counts.values())
    num_codons = len(codons)
    mean_count = total_mapped / num_codons if num_codons > 0 else 0
    
    with open(output_file, 'w') as f:
        # Header with better metrics
        f.write("codon_position,codon,nucleotide_start,nucleotide_end,p_site_count,")
        f.write("relative_occupancy,density,rpm\n")
        
        for i, codon in enumerate(codons):
            count = codon_counts.get(i, 0)
            nt_start = i * 3
            nt_end = nt_start + 2
            
            # Relative occupancy (0.0 - 1.0)
            rel_occ = count / total_mapped if total_mapped > 0 else 0
            
            # Density (normalized to mean, 1.0 = average)
            density = count / mean_count if mean_count > 0 else 0
            
            # RPM (Reads Per Million mapped to this gene)
            rpm = (count / total_mapped) * 1e6 if total_mapped > 0 else 0
            
            f.write(f"{i+1},{codon},{nt_start},{nt_end},{count},")
            f.write(f"{rel_occ:.6f},{density:.2f},{rpm:.2f}\n")
    
    print(f"✓ Output: {output_file}")
    print(f"  Total codons: {len(codons)}")
    print(f"  Codons with reads: {len([c for c in codon_counts.values() if c > 0])}")
    
    if codon_counts:
        print(f"\nTop 5 slowest codons:")
        for codon_num, count in codon_counts.most_common(5):
            print(f"  Position {codon_num+1}: {codons[codon_num]} → {count} reads")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nAvailable genes: GAPDH, ACTB, F9, F9_CO")
        sys.exit(1)
    
    fastq_file = sys.argv[1]
    gene_name = sys.argv[2].upper()
    output_file = f"{gene_name.lower()}_ribosome_counts.csv"
    
    print("="*70)
    print("FAST RIBOSOME PROFILING (k-mer hashing)")
    print("="*70)
    print(f"FASTQ: {fastq_file}")
    print(f"Gene: {gene_name}")
    print(f"Output: {output_file}")
    print("="*70)
    print()
    
    # Load gene
    gene_seq = load_gene_sequence(gene_name)
    print()
    
    # Build k-mer index (one-time, fast!)
    kmer_index = build_kmer_index(gene_seq)
    print()
    
    # Process FASTQ (much faster now!)
    codon_counts = process_fastq_fast(fastq_file, gene_seq, kmer_index, gene_name)
    
    # Write output
    print()
    write_output(codon_counts, gene_seq, gene_name, output_file)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == '__main__':
    main()