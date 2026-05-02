#!/usr/bin/env python3
"""
FAST Ribosome Profiling - k-mer based gene-specific read extraction
With UTR/CDS annotation per codon position.
CDS boundaries read from GENCODE header (CDS:start-end) – NOT from first ATG.
Skips genes flagged as no_clean_isoform in isoform_selection.json.

Improved:
  • P-site offset is now ONLY applied to CDS-mapped reads
  • UTR reads use raw 5'-end position → no artificial shift into CDS
  • Codon grid is ANCHORED at cds_start → first CDS codon is always in-frame
    (this fixes UTR5-lengths that are not multiples of 3, e.g. ACTB UTR5=155 nt)

Usage:
    python3 fast_ribo_analysis.py <fastq_file> <gene_name> <reference_seq_file>

Example:
    python3 fast_ribo_analysis.py SRR10072555.fastq ACO2 data/genes/ACO2_ENST00000466237.fasta
"""

import logging
import sys
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from Bio.Seq import Seq

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

P_SITE_OFFSET  = 13
MIN_LENGTH     = 26
MAX_LENGTH     = 34
KMER_SIZE      = 15
MAX_MISMATCHES = 3

ADAPTERS = [
    "AGATCGGAAGAGCACACGTCT",
    "AGATCGGAAGAGCGTCGTGTA",
]

ISOFORM_JSON = Path("/Users/jonathanopitz/Desktop/Master/isoform_selection.json")


# ─── CDS boundary detection ───────────────────────────────────────────────────

def find_cds_boundaries_from_header(header: str, seq: str) -> tuple[int, int]:
    """
    Reads CDS coordinates from GENCODE header annotation:
      e.g.  UTR5:1-30|CDS:31-1299|UTR3:1300-1460
      →  cds_start = 30  (0-based, inclusive)
      →  cds_end   = 1299 (0-based, exclusive)

    Falls back to first in-frame ATG detection if no CDS annotation.
    """
    m = re.search(r'CDS:(\d+)-(\d+)', header)
    if m:
        cds_start = int(m.group(1)) - 1   # GENCODE 1-based inclusive → 0-based
        cds_end   = int(m.group(2))       # end inclusive in header → exclusive in python
        return cds_start, cds_end

    # Fallback: first ATG followed by in-frame stop
    seq_up = seq.upper()
    for atg_pos in range(len(seq_up) - 2):
        if seq_up[atg_pos:atg_pos+3] != "ATG":
            continue
        from_atg  = seq_up[atg_pos:]
        from_atg  = from_atg[:len(from_atg)//3 * 3]
        prot_full = str(Seq(from_atg).translate())
        first_stop = prot_full.find('*')
        if first_stop == -1:
            continue
        cds_end = atg_pos + (first_stop + 1) * 3   # include stop codon
        return atg_pos, cds_end

    # absolute fallback: treat entire sequence as CDS
    return 0, len(seq) - len(seq) % 3


def codon_region(nt_pos: int, cds_start: int, cds_end: int) -> str:
    """Returns '5UTR', 'CDS', or '3UTR' for a nucleotide position."""
    if nt_pos < cds_start:
        return "5UTR"
    elif nt_pos >= cds_end:
        return "3UTR"
    else:
        return "CDS"


def validate_cds(ref_seq: str, cds_start: int, cds_end: int, gene: str) -> bool:
    """Sanity-check: CDS must start with ATG, end with stop, be multiple of 3."""
    cds = ref_seq[cds_start:cds_end]
    ok = True
    if len(cds) % 3 != 0:
        log.warning("  ⚠  CDS length %d nt is NOT a multiple of 3 for %s", len(cds), gene)
        ok = False
    if cds[:3] != "ATG":
        log.warning("  ⚠  CDS does NOT start with ATG for %s (got %s)", gene, cds[:3])
        ok = False
    if cds[-3:] not in ("TAA", "TAG", "TGA"):
        log.warning("  ⚠  CDS does NOT end with stop codon for %s (got %s)", gene, cds[-3:])
        ok = False
    return ok


# ─── Gene exclusion check ─────────────────────────────────────────────────────

def load_excluded_genes() -> set[str]:
    """Load genes flagged as no_clean_isoform or not_found from isoform_selection.json."""
    excluded = set()
    if not ISOFORM_JSON.is_file():
        log.warning("isoform_selection.json not found — no gene exclusion applied")
        return excluded

    with open(ISOFORM_JSON) as f:
        data = json.load(f)

    for gene, info in data.items():
        if isinstance(info, dict) and info.get("status") in ("no_clean_isoform", "not_found"):
            excluded.add(gene.upper())

    if excluded:
        log.info("Excluding genes: %s", ", ".join(sorted(excluded)))

    return excluded


# ─── Reference loading ────────────────────────────────────────────────────────

def load_reference_sequence(ref_path: str, gene_name: str) -> tuple[str, int, int]:
    seq_lines   = []
    header_line = ""

    with open(ref_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                header_line = line[1:]
            else:
                seq_lines.append(line)

    seq = ''.join(seq_lines).upper().replace('U', 'T')
    if not seq:
        sys.exit(f"Error: empty sequence in {ref_path}")

    cds_start, cds_end = find_cds_boundaries_from_header(header_line, seq)

    utr5_len = cds_start
    cds_len  = cds_end - cds_start
    utr3_len = len(seq) - cds_end
    source   = "GENCODE header" if re.search(r'CDS:\d+-\d+', header_line) else "ATG fallback"

    log.info("Reference : %s", gene_name)
    log.info("  File      : %s", ref_path)
    log.info("  Length    : %d nt  [%s]", len(seq), source)
    log.info("  5'UTR     : 0–%d (%d nt)", cds_start - 1 if cds_start > 0 else -1, utr5_len)
    log.info("  CDS       : %d–%d (%d nt, %d codons)", cds_start, cds_end - 1, cds_len, cds_len // 3)
    log.info("  3'UTR     : %d–%d (%d nt)", cds_end, len(seq) - 1, utr3_len)

    validate_cds(seq, cds_start, cds_end, gene_name)

    return seq, cds_start, cds_end


# ─── k-mer index ─────────────────────────────────────────────────────────────

def build_kmer_index(seq: str, k: int = KMER_SIZE) -> dict:
    """Build a k-mer index over both strands of seq for seed-and-extend alignment."""
    log.info("Building %d-mer index...", k)
    index = defaultdict(list)

    for i in range(len(seq) - k + 1):
        index[seq[i:i+k]].append(('f', i))

    rc = str(Seq(seq).reverse_complement())
    for i in range(len(rc) - k + 1):
        index[rc[i:i+k]].append(('r', i))

    log.info("  %d unique %d-mers", len(index), k)
    return index


# ─── Adapter trimming ─────────────────────────────────────────────────────────

def trim_adapter(read: str) -> str:
    """Trim the first matching adapter sequence from the 3' end of read."""
    for adapter in ADAPTERS:
        pos = read.find(adapter)
        if pos != -1:
            return read[:pos]
    return read


# ─── Fast seed-and-extend alignment ───────────────────────────────────────────

def fast_align(read: str, ref_seq: str, kmer_index: dict, k: int = KMER_SIZE):
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

    best_pos, best_mm, best_strand = None, MAX_MISMATCHES + 1, None

    for pos, strand in set(candidates):
        if strand == 'f':
            if pos < 0 or pos + len(read) > len(ref_seq):
                continue
            mm = sum(a != b for a, b in zip(read, ref_seq[pos:pos+len(read)]))
        else:
            read_rc = str(Seq(read).reverse_complement())
            if pos < 0 or pos + len(read_rc) > len(ref_seq):
                continue
            mm = sum(a != b for a, b in zip(read_rc, ref_seq[pos:pos+len(read_rc)]))

        if mm < best_mm:
            best_mm, best_pos, best_strand = mm, pos, ('+' if strand == 'f' else '-')

    return (best_pos, best_mm, best_strand) if best_mm <= MAX_MISMATCHES else (None, None, None)


# ─── Codon-grid helpers (CDS-ANCHORED) ────────────────────────────────────────

def build_codon_grid(ref_seq: str, cds_start: int, cds_end: int) -> list[tuple[int, str, str]]:
    """
    Build a list of codon tuples (nt_start, codon_seq, region) anchored at cds_start.

    CDS is laid out in-frame starting exactly at cds_start. UTR5 codons are built
    backwards from cds_start (so that if UTR5 length is not a multiple of 3, the
    leftmost fragment is dropped rather than shifting the CDS frame). UTR3 codons
    are built forward from cds_end.
    """
    codons: list[tuple[int, str, str]] = []

    # ── 5'UTR codons: backward from cds_start, in multiples of 3 ─────────────
    utr5_len        = cds_start
    n_utr5_codons   = utr5_len // 3
    utr5_codon_start = cds_start - n_utr5_codons * 3   # drop the leading fragment if utr5_len % 3 != 0
    for i in range(n_utr5_codons):
        nt = utr5_codon_start + i * 3
        codons.append((nt, ref_seq[nt:nt+3], "5UTR"))

    # ── CDS codons: forward from cds_start ───────────────────────────────────
    cds_len      = cds_end - cds_start
    n_cds_codons = cds_len // 3
    for i in range(n_cds_codons):
        nt = cds_start + i * 3
        codons.append((nt, ref_seq[nt:nt+3], "CDS"))

    # ── 3'UTR codons: forward from cds_end, drop trailing fragment ───────────
    utr3_remaining = len(ref_seq) - cds_end
    n_utr3_codons  = utr3_remaining // 3
    for i in range(n_utr3_codons):
        nt = cds_end + i * 3
        codons.append((nt, ref_seq[nt:nt+3], "3UTR"))

    return codons


def nt_to_codon_index(nt_pos: int, codon_grid: list[tuple[int, str, str]]) -> int | None:
    """
    Map an absolute nucleotide position to its codon index in the CDS-anchored grid.
    Returns None if the position falls outside any grid codon (i.e. dropped UTR fragment).

    Uses binary search on the (sorted) nt_start array.
    """
    if not codon_grid:
        return None

    # Binary search for largest nt_start ≤ nt_pos
    lo, hi = 0, len(codon_grid) - 1
    best = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if codon_grid[mid][0] <= nt_pos:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best == -1:
        return None
    nt_start = codon_grid[best][0]
    if nt_pos < nt_start or nt_pos >= nt_start + 3:
        return None
    return best


# ─── FASTQ processing ─────────────────────────────────────────────────────────

def process_fastq(
    fastq_path: str,
    ref_seq: str,
    kmer_index: dict,
    gene_name: str,
    cds_start: int,
    cds_end: int,
    codon_grid: list[tuple[int, str, str]],
) -> Counter:
    """
    Aligns ribosome footprints + improved region assignment:
      • CDS  → use P-site offset
      • UTR  → use raw 5'-end position

    Codon indexing uses the CDS-anchored grid (so codon 0 is never split across
    a frame boundary).
    """
    log.info("Processing: %s", fastq_path)

    codon_counts = Counter()
    seen: set[tuple[str, int, str]] = set()   # (seq, pos, strand)

    total = trimmed = passed_len = aligned = duplicates = 0

    with open(fastq_path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq  = f.readline().strip()
            f.readline()   # +
            f.readline()   # qual

            total += 1
            trimmed_seq = trim_adapter(seq)
            if len(trimmed_seq) < len(seq):
                trimmed += 1
            seq = trimmed_seq

            if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH):
                continue
            passed_len += 1

            pos, mm, strand = fast_align(seq, ref_seq, kmer_index)
            if pos is None:
                continue

            aligned += 1

            # ── PCR deduplication ────────────────────────────────────────
            dedup_key = (seq, pos, strand)
            if dedup_key in seen:
                duplicates += 1
                continue
            seen.add(dedup_key)

            # ── Improved position assignment ─────────────────────────────
            raw_nt = pos if strand == '+' else pos + len(seq) - 1
            raw_region = codon_region(raw_nt, cds_start, cds_end)

            if raw_region == "CDS":
                ppos = pos + P_SITE_OFFSET if strand == '+' else pos + len(seq) - P_SITE_OFFSET
            else:
                ppos = raw_nt

            codon_idx = nt_to_codon_index(ppos, codon_grid)
            if codon_idx is not None:
                codon_counts[codon_idx] += 1

            if total % 100_000 == 0:
                print(
                    f"  {total:,} reads | trim:{trimmed:,} | len-ok:{passed_len:,} | "
                    f"align:{aligned:,} | dupl:{duplicates:,}",
                    end='\r'
                )

    dedup_rate = duplicates / aligned * 100 if aligned > 0 else 0

    log.info("\nStatistics:")
    log.info("  Total reads            : %s", f"{total:,}")
    log.info("  Adapter trimmed        : %s (%.2f%%)", f"{trimmed:,}", trimmed / total * 100)
    log.info("  Length filter passed   : %s (%.2f%%)", f"{passed_len:,}", passed_len / total * 100)
    log.info("  Aligned to %-12s : %s (%.4f%%)", gene_name, f"{aligned:,}", aligned / total * 100)
    if passed_len > 0:
        log.info("    → of length-passed   : %.2f%%", aligned / passed_len * 100)
    log.info("  PCR duplicates removed : %s (%.1f%% of aligned)", f"{duplicates:,}", dedup_rate)
    log.info("  Unique reads counted   : %s", f"{aligned - duplicates:,}")
    log.info("  Codons with ≥1 read    : %d", len(codon_counts))

    return codon_counts


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(
    codon_counts: Counter,
    codon_grid: list[tuple[int, str, str]],
    gene_name: str,
    outfile: str,
) -> None:
    """Write per-codon ribosome occupancy CSV with region annotations."""
    log.info("Writing: %s", outfile)

    total_mapped = sum(codon_counts.values())
    n_codons     = len(codon_grid)
    mean         = total_mapped / n_codons if n_codons > 0 else 0

    with open(outfile, 'w') as f:
        f.write("codon_pos,codon,nt_start,nt_end,region,count,rel_occupancy,density,rpm\n")
        for i, (nt_start, codon, region) in enumerate(codon_grid):
            cnt  = codon_counts.get(i, 0)
            rel  = cnt / total_mapped if total_mapped > 0 else 0
            dens = cnt / mean         if mean > 0         else 0
            rpm  = rel * 1e6
            f.write(f"{i+1},{codon},{nt_start},{nt_start+2},{region},{cnt},{rel:.6f},{dens:.2f},{rpm:.2f}\n")

    cds_reads  = sum(cnt for i, cnt in codon_counts.items() if codon_grid[i][2] == "CDS")
    utr5_reads = sum(cnt for i, cnt in codon_counts.items() if codon_grid[i][2] == "5UTR")
    utr3_reads = sum(cnt for i, cnt in codon_counts.items() if codon_grid[i][2] == "3UTR")

    log.info("  Codons total        : %d", n_codons)
    log.info("  Reads in 5'UTR      : %s", f"{utr5_reads:,}")
    log.info("  Reads in CDS        : %s", f"{cds_reads:,}")
    log.info("  Reads in 3'UTR      : %s", f"{utr3_reads:,}")

    # Verify CDS integrity in output
    cds_rows = [(i, nt, codon) for i, (nt, codon, r) in enumerate(codon_grid) if r == "CDS"]
    if cds_rows:
        first_idx, first_nt, first_codon = cds_rows[0]
        last_idx,  last_nt,  last_codon  = cds_rows[-1]
        log.info("  First CDS codon     : pos=%d  nt=%d  codon=%s  (expected ATG)",
                 first_idx + 1, first_nt, first_codon)
        log.info("  Last  CDS codon     : pos=%d  nt=%d  codon=%s  (expected TAA/TAG/TGA)",
                 last_idx + 1, last_nt, last_codon)
        if first_codon != "ATG":
            log.warning("  ⚠  First CDS codon is NOT ATG — frame may be wrong!")
        if last_codon not in ("TAA", "TAG", "TGA"):
            log.warning("  ⚠  Last CDS codon is NOT a stop — frame may be wrong!")

    if codon_counts:
        log.info("Top 5 covered codons:")
        for idx, cnt in codon_counts.most_common(5):
            nt, codon, r = codon_grid[idx]
            log.info("  %4d  %s  %-5s  %6d reads", idx + 1, codon, r, cnt)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 4:
        print(__doc__.strip())
        sys.exit(1)

    fastq    = sys.argv[1]
    gene     = sys.argv[2].upper()
    ref_path = sys.argv[3]
    outfile  = f"{gene.lower()}_ribosome_counts.csv"

    log.info("Input FASTQ : %s", fastq)
    log.info("Gene        : %s", gene)
    log.info("Reference   : %s", ref_path)
    log.info("Output      : %s", outfile)

    excluded = load_excluded_genes()
    if gene in excluded:
        log.info("%s excluded (no clean isoform / stop codons in CDS)", gene)
        sys.exit(0)

    ref_seq, cds_start, cds_end = load_reference_sequence(ref_path, gene)
    codon_grid = build_codon_grid(ref_seq, cds_start, cds_end)
    index = build_kmer_index(ref_seq)

    counts = process_fastq(fastq, ref_seq, index, gene, cds_start, cds_end, codon_grid)

    save_results(counts, codon_grid, gene, outfile)
    log.info("Done.")


if __name__ == '__main__':
    main()