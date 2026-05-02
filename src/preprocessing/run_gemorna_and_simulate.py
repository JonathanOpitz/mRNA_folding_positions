#!/usr/bin/env python3
"""
Batch GEMORNA + sTASEP Pipeline
================================
1. Reads isoform_selection.json
2. Runs GEMORNA to generate optimized CDS + 5'UTR + 3'UTR per gene
3. Rebuilds full optimized FASTA
4. Runs sTASEP simulation → same CSV format as fast_ribo_analysis.py

Usage:
    conda activate gemorna
    python src/preprocessing/run_gemorna_and_simulate.py
    python src/preprocessing/run_gemorna_and_simulate.py --genes GAPDH ACTB
    python src/preprocessing/run_gemorna_and_simulate.py --no-simulate
"""

import re
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.Seq import Seq

# ── Paths ──────────────────────────────────────────────────────────
BASE         = Path("/Users/jonathanopitz/Desktop/Master")
ISOFORM_JSON = BASE / "isoform_selection.json"
GENES_DIR    = BASE / "data/genes"
OPT_DIR      = BASE / "data/genes_gemorna"
SIM_DIR      = BASE / "data/ribo_counts_simulated"
GEMORNA_DIR  = BASE / "GEMORNA"

# ── sTASEP parameters ──────────────────────────────────────────────
RIBOSOME_FOOTPRINT = 10
N_STEPS            = 500_000
WARMUP             = 100_000
INITIATION_RATE    = 0.08
STRUCTURE_WEIGHT   = 0.25
SPARSITY           = 0.35

HUMAN_TAI = {
    "TTT": 0.232, "TTC": 1.000, "TTA": 0.056, "TTG": 0.231,
    "CTT": 0.232, "CTC": 0.530, "CTA": 0.098, "CTG": 1.000,
    "ATT": 0.522, "ATC": 1.000, "ATA": 0.089, "ATG": 1.000,
    "GTT": 0.376, "GTC": 0.627, "GTA": 0.140, "GTG": 1.000,
    "TCT": 0.627, "TCC": 0.887, "TCA": 0.298, "TCG": 0.067,
    "AGT": 0.232, "AGC": 1.000,
    "CCT": 0.522, "CCC": 0.743, "CCA": 0.522, "CCG": 0.067,
    "ACT": 0.522, "ACC": 1.000, "ACA": 0.376, "ACG": 0.067,
    "GCT": 0.627, "GCC": 1.000, "GCA": 0.376, "GCG": 0.089,
    "TAT": 0.232, "TAC": 1.000,
    "TAA": 0.000, "TAG": 0.000, "TGA": 0.000,
    "CAT": 0.298, "CAC": 1.000, "CAA": 0.298, "CAG": 1.000,
    "AAT": 0.232, "AAC": 1.000, "AAA": 0.627, "AAG": 1.000,
    "GAT": 0.376, "GAC": 1.000, "GAA": 0.627, "GAG": 1.000,
    "TGT": 0.298, "TGC": 1.000, "TGG": 1.000,
    "CGT": 0.627, "CGC": 0.887, "CGA": 0.140, "CGG": 0.376,
    "AGA": 0.627, "AGG": 0.522,
    "GGT": 0.627, "GGC": 1.000, "GGA": 0.376, "GGG": 0.376,
}


# ════════════════════════════════════════════════════════════════════
# FASTA utilities
# ════════════════════════════════════════════════════════════════════

def load_fasta(path: Path) -> tuple:
    lines, header = [], ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
            elif line:
                lines.append(line)
    return header, ''.join(lines).upper().replace('U', 'T')


def get_cds_boundaries(header: str, seq: str) -> tuple:
    m = re.search(r'CDS:(\d+)-(\d+)', header)
    if m:
        return int(m.group(1)) - 1, int(m.group(2))
    atg = seq.find('ATG')
    if atg == -1:
        return 0, len(seq)
    sub  = seq[atg:][:len(seq[atg:]) // 3 * 3]
    prot = str(Seq(sub).translate())
    stop = prot.find('*')
    return atg, atg + (stop+1)*3 if stop != -1 else atg + len(sub)


def codon_region(nt: int, cs: int, ce: int) -> str:
    if nt < cs:  return '5UTR'
    if nt >= ce: return '3UTR'
    return 'CDS'


def extract_protein(seq: str, cs: int, ce: int) -> str:
    cds  = seq[cs:ce][:len(seq[cs:ce]) // 3 * 3]
    prot = str(Seq(cds).translate())
    return prot.rstrip('*')


def utr_length_category(utr_len: int) -> str:
    if utr_len < 80:   return 'short'
    if utr_len < 200:  return 'medium'
    return 'long'


# ════════════════════════════════════════════════════════════════════
# GEMORNA output parser — FIX
# ════════════════════════════════════════════════════════════════════

def parse_gemorna_output(stdout: str) -> str | None:
    """
    GEMORNA output format: SEQUENCE SCORE
    e.g.: ATGGCGAAAGTT...GCTTAA 0.61

    Strategy: find the line where the second-to-last token is a valid
    nucleotide sequence and the last token is a float score.
    """
    VALID_NTS = set('ATCGUatcgu')

    for line in reversed(stdout.strip().split('\n')):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Format: SEQUENCE SCORE  (2 tokens)
        if len(parts) == 2:
            seq_cand, score_cand = parts
            try:
                float(score_cand)   # validate score
                if all(c in VALID_NTS for c in seq_cand) and len(seq_cand) > 30:
                    return seq_cand.upper().replace('U', 'T')
            except ValueError:
                pass

        # Format: just SEQUENCE (1 token, no score)
        if len(parts) == 1:
            seq_cand = parts[0]
            if all(c in VALID_NTS for c in seq_cand) and len(seq_cand) > 30:
                return seq_cand.upper().replace('U', 'T')

        # Format: longer line — find first token that is a nucleotide sequence
        for token in parts:
            if len(token) > 30 and all(c in VALID_NTS for c in token):
                return token.upper().replace('U', 'T')

    return None


# ════════════════════════════════════════════════════════════════════
# GEMORNA runners
# ════════════════════════════════════════════════════════════════════

def run_gemorna_cds(protein_seq: str, gene: str,
                   cache_dir: Path) -> str | None:
    cache_file = cache_dir / f"{gene}_gemorna_cds.txt"
    if cache_file.exists():
        seq = cache_file.read_text().strip()
        if seq:
            return seq.upper().replace('U', 'T')

    result = subprocess.run(
        [sys.executable,
         str(GEMORNA_DIR / 'src' / 'generate.py'),
         '--mode', 'cds',
         '--ckpt_path', str(GEMORNA_DIR / 'checkpoints' / 'gemorna_cds.pt'),
         '--protein_seq', protein_seq],
        capture_output=True, text=True, timeout=180,
        cwd=str(GEMORNA_DIR)
    )

    if result.returncode != 0:
        print(f'\n    ✗ CDS error: {result.stderr[-150:]}')
        return None

    seq = parse_gemorna_output(result.stdout)
    if seq is None:
        # Show raw output for debugging
        print(f'\n    ✗ Parse failed. GEMORNA stdout:')
        for line in result.stdout.strip().split('\n')[-5:]:
            print(f'      | {line}')
        return None

    cache_file.write_text(seq)
    return seq


def run_gemorna_utr(mode: str, length_cat: str,
                   gene: str, cache_dir: Path) -> str | None:
    cache_file = cache_dir / f"{gene}_gemorna_{mode}.txt"
    if cache_file.exists():
        seq = cache_file.read_text().strip()
        if seq:
            return seq.upper().replace('U', 'T')

    ckpt = 'gemorna_5utr.pt' if mode == '5utr' else 'gemorna_3utr.pt'
    result = subprocess.run(
        [sys.executable,
         str(GEMORNA_DIR / 'src' / 'generate.py'),
         '--mode', mode,
         '--ckpt_path', str(GEMORNA_DIR / 'checkpoints' / ckpt),
         '--utr_length', length_cat],
        capture_output=True, text=True, timeout=120,
        cwd=str(GEMORNA_DIR)
    )

    if result.returncode != 0:
        return None

    seq = parse_gemorna_output(result.stdout)
    if seq is None:
        # UTR might be plain sequence without score
        for line in reversed(result.stdout.strip().split('\n')):
            line = line.strip()
            if line and all(c in 'ATCGUatcgu' for c in line) and len(line) > 5:
                seq = line.upper().replace('U', 'T')
                break

    if seq:
        cache_file.write_text(seq)
    return seq


def build_full_fasta(gene: str, enst: str,
                     utr5: str, cds: str, utr3: str,
                     output_path: Path) -> None:
    full     = utr5 + cds + utr3
    cs_1b    = len(utr5) + 1
    ce_1b    = len(utr5) + len(cds)
    header   = (f"{gene}|{enst}|gemorna_optimized"
                f"|UTR5:1-{len(utr5)}"
                f"|CDS:{cs_1b}-{ce_1b}"
                f"|UTR3:{ce_1b+1}-{len(full)}")
    with open(output_path, 'w') as f:
        f.write(f'>{header}\n')
        for i in range(0, len(full), 60):
            f.write(full[i:i+60] + '\n')


# ════════════════════════════════════════════════════════════════════
# sTASEP simulation
# ════════════════════════════════════════════════════════════════════

def compute_local_structure(seq: str, window: int = 30) -> np.ndarray:
    n, dg = len(seq), np.zeros(len(seq))
    try:
        import RNA
        half = window // 2
        for i in range(n):
            sub    = seq[max(0,i-half):min(n,i+half)].replace('T','U')
            _, mfe = RNA.fold_compound(sub).mfe()
            dg[i]  = mfe
    except ImportError:
        half = window // 2
        for i in range(n):
            sub   = seq[max(0,i-half):min(n,i+half)]
            gc    = (sub.count('G') + sub.count('C')) / max(len(sub), 1)
            dg[i] = -30.0 * gc
    return dg


def compute_elongation_rates(codons: list, dg: np.ndarray,
                              cds_start: int) -> np.ndarray:
    n   = len(codons)
    dgc = np.array([
        dg[cds_start + i*3] if cds_start + i*3 < len(dg) else 0.0
        for i in range(n)
    ])
    mn, mx = dgc.min(), dgc.max()
    sp = (dgc - mx) / (mn - mx) if mx != mn else np.zeros(n)
    rates = np.zeros(n)
    for i, cod in enumerate(codons):
        tai      = HUMAN_TAI.get(cod, 0.3)
        rates[i] = max(tai * (1.0 - STRUCTURE_WEIGHT * sp[i]), 0.01)
    return rates


def run_tasep(rates: np.ndarray, initiation_rate: float,
              seed: int = 42) -> np.ndarray:
    n, fp = len(rates), RIBOSOME_FOOTPRINT
    max_r = max(1, n // fp)
    n_use = max(1, max_r // 3)
    start = max(1, n_use // 4)

    positions = np.linspace(0, n - fp - 1, start, dtype=int)
    occupied  = set(positions.tolist())
    occupancy = np.zeros(n, dtype=np.float64)
    rng       = np.random.default_rng(seed)
    rec       = 0

    for step in range(N_STEPS):
        if len(occupied) < n_use:
            if all(p >= fp for p in occupied) or not occupied:
                if rng.random() < initiation_rate:
                    occupied.add(0)
        if not occupied:
            continue

        pos      = list(occupied)[rng.integers(0, len(occupied))]
        next_pos = pos + 1
        if next_pos + fp - 1 >= n:
            occupied.discard(pos)
            continue

        blocked = any(
            next_pos <= p < next_pos + fp or p <= next_pos < p + fp
            for p in occupied if p != pos
        )
        if blocked:
            continue

        if rng.random() < rates[next_pos]:
            occupied.discard(pos)
            occupied.add(next_pos)

        if step >= WARMUP and step % 5 == 0:
            for p in occupied:
                occupancy[p] += 1
            rec += 1

    return occupancy / max(rec, 1)


def add_sparsity(occ: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = occ.copy()
    pos = out[out > 0]
    if len(pos) == 0:
        return out
    med  = np.median(pos)
    cand = np.where((out > 0) & (out < med))[0]
    n_z  = int(len(cand) * SPARSITY)
    if n_z > 0:
        out[rng.choice(cand, size=n_z, replace=False)] = 0
    return out


def simulate(fasta_path: Path, scale: int,
             initiation_rate: float) -> pd.DataFrame | None:
    header, seq = load_fasta(fasta_path)
    cs, ce      = get_cds_boundaries(header, seq)
    cds_seq     = seq[cs:ce]
    codons      = [cds_seq[i:i+3] for i in range(0, len(cds_seq)//3*3, 3)]

    dg    = compute_local_structure(seq)
    rates = compute_elongation_rates(codons, dg, cs)
    occ   = run_tasep(rates, initiation_rate)
    occ   = add_sparsity(occ, np.random.default_rng(99))

    n_full   = len(seq) // 3
    full_occ = np.zeros(n_full)
    for i, v in enumerate(occ):
        idx = cs // 3 + i
        if idx < n_full:
            full_occ[idx] = v

    total = full_occ.sum()
    if total == 0:
        return None
    norm   = full_occ / total
    counts = np.round(norm * scale).astype(int)
    tot_c  = counts.sum()
    mean_c = counts[counts > 0].mean() if (counts > 0).any() else 1.0

    all_codons = [seq[i*3:i*3+3] for i in range(n_full)]
    rows = []
    for i, codon in enumerate(all_codons):
        nt  = i * 3
        cnt = int(counts[i])
        reg = codon_region(nt, cs, ce)
        rel = cnt / tot_c  if tot_c  > 0 else 0.0
        dns = cnt / mean_c if mean_c > 0 else 0.0
        rows.append({
            'codon_pos':     i + 1,
            'codon':         codon,
            'nt_start':      nt,
            'nt_end':        nt + 2,
            'region':        reg,
            'count':         cnt,
            'rel_occupancy': round(rel, 6),
            'density':       round(dns, 2),
            'rpm':           round(rel * 1e6, 2),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genes',       nargs='+', default=None)
    parser.add_argument('--no-simulate', action='store_true')
    parser.add_argument('--scale',       type=int,   default=3000)
    parser.add_argument('--initiation',  type=float, default=INITIATION_RATE)
    args = parser.parse_args()

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)

    with open(ISOFORM_JSON) as f:
        isoforms = json.load(f)

    genes_ok = [
        (gene, info) for gene, info in isoforms.items()
        if info.get('status') == 'ok'
        and (args.genes is None or gene in args.genes)
    ]

    print(f'{"="*65}')
    print(f'GEMORNA + sTASEP Batch Pipeline')
    print(f'{"="*65}')
    print(f'Genes     : {len(genes_ok)}')
    print(f'Scale     : {args.scale}')
    print(f'Simulate  : {not args.no_simulate}\n')

    success, failed = [], []

    for i, (gene, info) in enumerate(genes_ok, 1):
        best_enst = info['best_isoform']
        print(f'[{i:2d}/{len(genes_ok)}] {gene:<14} {best_enst}')

        # ── Find wildtype FASTA ───────────────────────────────────
        matches = list(GENES_DIR.glob(f'{gene}_{best_enst}*.fasta'))
        if not matches:
            print(f'  ✗ FASTA nicht gefunden')
            failed.append((gene, 'fasta_missing'))
            continue
        fasta_path = matches[0]

        # ── Load + extract components ─────────────────────────────
        try:
            header, seq = load_fasta(fasta_path)
            cs, ce      = get_cds_boundaries(header, seq)
            utr5_seq    = seq[:cs]
            cds_seq     = seq[cs:ce]
            utr3_seq    = seq[ce:]
            protein     = extract_protein(seq, cs, ce)
        except Exception as e:
            print(f'  ✗ Ladefehler: {e}')
            failed.append((gene, str(e)))
            continue

        print(f'  WT: 5UTR={len(utr5_seq)}nt  CDS={len(cds_seq)}nt  '
              f'3UTR={len(utr3_seq)}nt  Prot={len(protein)}AA')

        # ── GEMORNA (with caching) ────────────────────────────────
        opt_fasta = OPT_DIR / f'{gene}_{best_enst}_gemorna.fasta'
        cache_dir = OPT_DIR / gene
        cache_dir.mkdir(parents=True, exist_ok=True)

        if opt_fasta.exists():
            print(f'  GEMORNA: SKIP (bereits vorhanden)')
        else:
            # CDS
            print(f'  GEMORNA CDS    ...', end=' ', flush=True)
            opt_cds = run_gemorna_cds(protein, gene, cache_dir)
            if opt_cds is None:
                failed.append((gene, 'gemorna_cds_failed'))
                continue
            gc = sum(c in 'GC' for c in opt_cds) / len(opt_cds) * 100
            print(f'OK  ({len(opt_cds)}nt, GC={gc:.1f}%)')

            # 5'UTR
            utr5_cat = utr_length_category(len(utr5_seq))
            print(f'  GEMORNA 5UTR ({utr5_cat}) ...', end=' ', flush=True)
            opt_utr5 = run_gemorna_utr('5utr', utr5_cat, gene, cache_dir)
            print(f'OK ({len(opt_utr5)}nt)' if opt_utr5
                  else 'fallback → original')
            if not opt_utr5:
                opt_utr5 = utr5_seq

            # 3'UTR
            utr3_cat = utr_length_category(len(utr3_seq))
            print(f'  GEMORNA 3UTR ({utr3_cat}) ...', end=' ', flush=True)
            opt_utr3 = run_gemorna_utr('3utr', utr3_cat, gene, cache_dir)
            print(f'OK ({len(opt_utr3)}nt)' if opt_utr3
                  else 'fallback → original')
            if not opt_utr3:
                opt_utr3 = utr3_seq

            build_full_fasta(gene, best_enst, opt_utr5, opt_cds,
                             opt_utr3, opt_fasta)
            total_len = len(opt_utr5) + len(opt_cds) + len(opt_utr3)
            print(f'  → {opt_fasta.name} ({total_len}nt total)')

        # ── sTASEP simulation ─────────────────────────────────────
        if not args.no_simulate:
            out_csv = SIM_DIR / f'{gene.lower()}_gemorna_simulated_ribo.csv'
            if out_csv.exists():
                print(f'  sTASEP: SKIP (bereits vorhanden)')
            else:
                print(f'  sTASEP: simuliere ...', end=' ', flush=True)
                try:
                    df = simulate(opt_fasta, args.scale, args.initiation)
                    if df is None:
                        print('FEHLER (0 counts)')
                        failed.append((gene, 'tasep_zero'))
                        continue
                    df.to_csv(out_csv, index=False)
                    cds_r = df[df['region']=='CDS']['count'].sum()
                    cov   = (df['count']>0).sum() / len(df) * 100
                    print(f'OK  (CDS: {cds_r:,} reads, cov: {cov:.0f}%)')
                except Exception as e:
                    print(f'FEHLER: {e}')
                    failed.append((gene, str(e)))
                    continue

        success.append(gene)
        print()

    print(f'{"="*65}')
    print(f'Fertig: {len(success)}/{len(genes_ok)} erfolgreich')
    if failed:
        print('Fehlgeschlagen:')
        for g, r in failed:
            print(f'  {g:<14} {r}')


if __name__ == '__main__':
    main()