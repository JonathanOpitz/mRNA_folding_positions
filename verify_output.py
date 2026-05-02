"""Verify that the best generated sequence preserves the ACTB protein."""
import glob, sys

FA = '/Users/jonathanopitz/Desktop/Master/data/genes/ACTB_ENST00000493945.fasta'
RESULTS_GLOB = '/Users/jonathanopitz/Desktop/Master/results_*/HEK293T*/*/samples/optim_results.txt'

# WT CDS
seq = ''.join(l.strip() for l in open(FA) if not l.startswith('>')).upper().replace('U','T')
wt_cds = seq[155:1283]   # from GENCODE header

codon_table = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
def tr(s): return ''.join(codon_table.get(s[i:i+3], '?') for i in range(0, len(s)-2, 3))

wt_prot = tr(wt_cds)
print(f"WT protein: {len(wt_prot)} aa  starts={wt_prot[:15]}  ends={wt_prot[-5:]}")

for results in sorted(glob.glob(RESULTS_GLOB)):
    print(f"\n── {results}")
    with open(results) as f:
        lines = [l.strip().split('\t') for l in f if l.strip()]
    # Best by rpf
    lines_sorted = sorted(lines, key=lambda l: float(l[1]), reverse=True)
    best_seq, best_rpf, _ = lines_sorted[0]
    print(f"  Best seq: len={len(best_seq)}  rpf={best_rpf}")
    gen_prot = tr(best_seq)
    print(f"  Generated protein: {len(gen_prot)} aa  starts={gen_prot[:15]}  ends={gen_prot[-5:]}")
    if len(wt_prot) == len(gen_prot):
        identity = sum(a==b for a,b in zip(wt_prot, gen_prot)) / len(wt_prot)
        print(f"  Protein identity: {100*identity:.2f}%  {'✓ PASS' if identity > 0.99 else '✗ FAIL'}")
    else:
        print(f"  ✗ FAIL: length mismatch")
