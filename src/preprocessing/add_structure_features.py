#!/usr/bin/env python3
"""
Run local ColabFold (AlphaFold) on CDS-only protein sequence,
then extract structural features per codon position.

Secondary Structure: DSSP-style H-bond algorithm (Kabsch & Sander 1983)
Domain Boundaries: InterPro + PAE-based detection
"""

import json
import os
import re
import time
import subprocess
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/Users/jonathanopitz/Desktop/Master")
RESULTS_DIR   = BASE_DIR / "data/ribo_counts"
GENES_DIR     = BASE_DIR / "data/genes"
ISOFORM_JSON  = BASE_DIR / "isoform_selection.json"
AF_OUT_DIR    = BASE_DIR / "data/alphafold_results"
COLABFOLD_BIN = BASE_DIR / "localcolabfold/.pixi/envs/default/bin/colabfold_batch"

INTERPRO_URL   = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot}/?format=json"
ENSEMBL_SERVER = "https://rest.ensembl.org"
MIN_PROTEIN_AA = 100

# ─── FIX 1: ColabFold subprocess timeout ──────────────────────────────────────
# ColabFold on CPU can take ~2h per protein.
# Set generous timeout (6h), but also handle post-write latency gracefully.
COLABFOLD_TIMEOUT_SEC = 55 * 3600   # 6 hours hard limit per gene

# DSSP H-bond energy threshold (Kabsch & Sander 1983)
HBOND_ENERGY_CUTOFF = -0.5   # kcal/mol

# PAE domain boundary detection parameters
PAE_WINDOW      = 10
PAE_THRESHOLD   = 15.0
PAE_MIN_GAP     = 15

# ─── FIX 2: Contact density parameters ───────────────────────────────────────
# Problem: PAE < 10 is too permissive – every local residue passes → always 21.
# Solution:
#   • Use PAE < 5 Å (well-determined, tight contacts only)
#   • Normalize by window size  → value in [0, 1]
#   • Subtract self (residue i vs i) → only count *neighbours*
CONTACT_PAE_CUTOFF = 5.0    # Å  (was 10.0 – far too loose)
CONTACT_WINDOW     = 10     # ±10 residues


def _aa(p) -> int | None:
    if p is None:
        return None
    try:
        if np.isnan(float(p)):
            return None
    except (TypeError, ValueError):
        return None
    return int(p)


# ─── Find ColabFold output files ──────────────────────────────────────────────

def find_struct_file(gene_dir: Path) -> Path | None:
    patterns = [
        "*rank_001*.cif", "*rank_001*.pdb",
        "*model_1_seed_000*.pdb", "*model_1_seed_000*.cif",
        "*unrelaxed*.pdb", "*unrelaxed*.cif",
        "*.pdb", "*.cif",
    ]
    for pattern in patterns:
        found = [
            f for f in sorted(gene_dir.glob(pattern))
            if f.suffix in ('.pdb', '.cif') and 'input' not in f.name
        ]
        if found:
            return found[0]
    return None


def find_scores_file(gene_dir: Path) -> Path | None:
    patterns = [
        "*scores_rank_001*.json",
        "*scores_alphafold2*model_1*.json",
        "*scores*model_1_seed_000*.json",
        "*scores*.json",
    ]
    for pattern in patterns:
        found = [f for f in sorted(gene_dir.glob(pattern)) if 'input' not in f.name]
        if found:
            return found[0]
    return None

def find_best_fasta(genes_dir: Path, gene: str, enst_base: str) -> Path | None:
    """
    Sucht FASTA-Datei mit Priorität:
      1. Exakter Match ohne Version: GENE_ENST0000XXXX.fasta     ← GENCODE
      2. Match mit Version:          GENE_ENST0000XXXX.8.fasta   ← Ensembl
      3. Beliebiger Match:           GENE_ENST0000XXXX*.fasta
 
    GENCODE-Dateien haben CDS:-Koordinaten im Header → bevorzugen.
    """
    # Priorität 1: exakter Match (GENCODE)
    exact = genes_dir / f"{gene}_{enst_base}.fasta"
    if exact.exists():
        return exact
 
    # Priorität 2: mit Versionssuffix (Ensembl REST)
    versioned = sorted(genes_dir.glob(f"{gene}_{enst_base}.*.fasta"))
    if versioned:
        return versioned[0]
 
    # Priorität 3: alles was passt
    any_match = sorted(genes_dir.glob(f"{gene}_{enst_base}*.fasta"))
    if any_match:
        return any_match[0]
 
    return None
 
 
# ════════════════════════════════════════════════════════════════════
# Fix 2: CDS-Erkennung mit Ensembl-Fallback
# ════════════════════════════════════════════════════════════════════
 
def get_cds_from_ensembl(enst_id: str) -> tuple[int, int] | None:
    """
    Holt CDS-Koordinaten (0-based start, exclusive end) von Ensembl REST API.
    enst_id: z.B. 'ENST00000226730' oder 'ENST00000226730.5'
    """
    enst_base = enst_id.split('.')[0]
    url = f"{ENSEMBL_SERVER}/lookup/id/{enst_base}?content-type=application/json;expand=1"
    try:
        r = requests.get(url, timeout=15)
        if not r.ok:
            return None
        data = r.json()
        # Translation object has CDS start/end in genomic coords — not useful
        # Instead use the UTR information to derive CDS position in transcript
        utr5_len = 0
        utr3_len = 0
        seq_len  = data.get('length', 0)
        for utr in data.get('Utr', []):
            utr_type = utr.get('object_type', '')
            utr_len  = abs(int(utr.get('end', 0)) - int(utr.get('start', 0))) + 1
            if 'five' in utr_type.lower() or '5' in utr_type:
                utr5_len += utr_len
            elif 'three' in utr_type.lower() or '3' in utr_type:
                utr3_len += utr_len
 
        if utr5_len > 0 or utr3_len > 0:
            cds_start = utr5_len
            cds_end   = seq_len - utr3_len
            return cds_start, cds_end
 
    except Exception as e:
        print(f"  [Ensembl CDS] {enst_base}: {e}")
    return None
 
 
def get_cds_protein_v2(fasta_path: Path) -> tuple[str, int, int]:
    """
    Improved version of get_cds_protein().
 
    Priority:
      1. CDS: coordinates in GENCODE header  ← most reliable
      2. Ensembl REST API lookup             ← for Ensembl-downloaded files
      3. ATG scan with longest ORF           ← last resort, better than first ATG
    """
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq    = str(record.seq).upper().replace("U", "T")
    header = record.description
 
    # ── Priority 1: GENCODE header ────────────────────────────────────────────
    m = re.search(r'CDS:(\d+)-(\d+)', header)
    if m:
        cds_start = int(m.group(1)) - 1
        cds_end   = int(m.group(2))
        cds_seq   = seq[cds_start:cds_end]
        cds_trim  = cds_seq[:len(cds_seq) // 3 * 3]
        protein   = str(Seq(cds_trim).translate()).rstrip('*')
        print(f"  [CDS] GENCODE header → nt {cds_start}–{cds_end} "
              f"({cds_end - cds_start} nt) → {len(protein)} AA")
        return protein, cds_start, cds_end
 
    # ── Priority 2: Ensembl REST API ─────────────────────────────────────────
    # Extract ENST ID from header
    enst_match = re.search(r'(ENST\d+)', header)
    if enst_match:
        enst_id = enst_match.group(1)
        print(f"  [CDS] No GENCODE header for {enst_id}, querying Ensembl REST ...")
        time.sleep(0.5)  # be polite
        coords = get_cds_from_ensembl(enst_id)
        if coords:
            cds_start, cds_end = coords
            # Sanity check
            if 0 <= cds_start < cds_end <= len(seq):
                cds_seq  = seq[cds_start:cds_end]
                cds_trim = cds_seq[:len(cds_seq) // 3 * 3]
                protein  = str(Seq(cds_trim).translate()).rstrip('*')
                print(f"  [CDS] Ensembl REST → nt {cds_start}–{cds_end} "
                      f"({cds_end - cds_start} nt) → {len(protein)} AA")
                if len(protein) > 50:
                    return protein, cds_start, cds_end
                else:
                    print(f"  [CDS] Ensembl coords gave short protein ({len(protein)} AA) "
                          f"— trying ORF scan")
 
    # ── Priority 3: Longest ORF scan (better than first-ATG) ─────────────────
    print(f"  [CDS] Fallback: scanning for longest ORF ...")
    best_protein = ""
    best_start   = 0
    best_end     = 0
 
    for frame_start in range(len(seq)):
        if seq[frame_start:frame_start+3] != 'ATG':
            continue
        sub  = seq[frame_start:]
        sub  = sub[:len(sub) // 3 * 3]
        prot = str(Seq(sub).translate())
        stop = prot.find('*')
        if stop == -1:
            continue
        orf_prot = prot[:stop]
        if len(orf_prot) > len(best_protein):
            best_protein = orf_prot
            best_start   = frame_start
            best_end     = frame_start + (stop + 1) * 3
 
    if len(best_protein) < 50:
        raise ValueError(f"No ORF ≥ 50 AA found in {fasta_path.name}")
 
    print(f"  [CDS] Longest ORF → nt {best_start}–{best_end} "
          f"({best_end - best_start} nt) → {len(best_protein)} AA")
    return best_protein, best_start, best_end

# ─── CDS extraction ───────────────────────────────────────────────────────────

def get_cds_protein(fasta_path: Path) -> tuple[str, int, int]:
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq    = str(record.seq).upper().replace("U", "T")
    header = record.description

    m = re.search(r'CDS:(\d+)-(\d+)', header)
    if m:
        cds_start = int(m.group(1)) - 1
        cds_end   = int(m.group(2))
        cds_seq   = seq[cds_start:cds_end]
        source    = "GENCODE header"
    else:
        atg = seq.find("ATG")
        if atg == -1:
            raise ValueError(f"No CDS annotation and no ATG in {fasta_path.name}")
        cds_start = atg
        from_atg  = seq[atg:]
        from_atg  = from_atg[:len(from_atg) // 3 * 3]
        prot_tmp  = str(Seq(from_atg).translate())
        stop_idx  = prot_tmp.find('*')
        cds_end   = atg + (stop_idx + 1) * 3 if stop_idx != -1 else atg + len(from_atg)
        cds_seq   = seq[cds_start:cds_end]
        source    = "ATG fallback"

    cds_trim = cds_seq[:len(cds_seq) // 3 * 3]
    protein  = str(Seq(cds_trim).translate()).rstrip('*')
    print(f"  [CDS] {source} → nt {cds_start}–{cds_end} ({cds_end - cds_start} nt) → {len(protein)} AA")
    return protein, cds_start, cds_end


# ─── ColabFold ────────────────────────────────────────────────────────────────

def run_colabfold(gene: str, protein_seq: str, out_dir: Path) -> Path | None:
    gene_dir = out_dir / gene
    gene_dir.mkdir(parents=True, exist_ok=True)

    existing = find_struct_file(gene_dir)
    if existing:
        print(f"  [AF] {gene}: Reusing → {existing.name}")
        return existing

    fasta_in = gene_dir / f"{gene}_input.fasta"
    fasta_in.write_text(f">{gene}\n{protein_seq}\n")
    print(f"  [AF] {gene}: Running ColabFold ({len(protein_seq)} AA) ...")
    print(f"  [AF] Timeout set to {COLABFOLD_TIMEOUT_SEC // 3600}h")

    log_path = gene_dir / "colabfold_run.log"

    # ─── FIX 1a: Use timeout= so the process can't hang forever ──────────────
    # Also add --num-models 1 to skip running models 2-5 (saves 4x time on CPU)
    cmd = [
        str(COLABFOLD_BIN), str(fasta_in), str(gene_dir),
        "--num-models", "1",          # only best model – huge CPU speedup
        "--num-recycle", "1",         # default 3 is fine
        "--model-type", "alphafold2_ptm",
    ]

    try:
        with open(log_path, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=log_f,
                timeout=COLABFOLD_TIMEOUT_SEC,   # ← FIX: hard timeout
            )
    except subprocess.TimeoutExpired:
        print(f"  [AF] {gene}: TIMEOUT after {COLABFOLD_TIMEOUT_SEC//3600}h – skipping")
        return None
    except Exception as e:
        print(f"  [AF] {gene}: subprocess error: {e}")
        return None

    if result.returncode != 0:
        print(f"  [AF] {gene}: FAILED (exit {result.returncode}) – see {log_path}")
        return None

    # ─── FIX 1b: Wait for file-system flush after ColabFold exits ────────────
    # ColabFold may still be flushing output after subprocess returns.
    # Poll up to 120s (was 60s) with exponential backoff.
    print("  [AF] ColabFold exited cleanly. Waiting for output files ...")
    for attempt, wait in enumerate([5, 5, 10, 10, 15, 15, 20, 20, 20], 1):
        time.sleep(wait)
        found = find_struct_file(gene_dir)
        if found:
            print(f"  [AF] Found after attempt {attempt}: {found.name}")
            return found
        print(f"  [AF] Not yet ({attempt}/9) ...")

    # Still nothing – list directory for debugging
    print(f"  [AF] {gene}: No structure file found. Directory contents:")
    for f in sorted(gene_dir.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size:,} bytes)")
    return None


# ─── PDB/CIF backbone parsing ─────────────────────────────────────────────────

def parse_backbone(struct_path: Path) -> dict[int, dict]:
    backbone: dict[int, dict] = {}

    if struct_path.suffix == '.pdb':
        with open(struct_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name not in ('N', 'CA', 'C', 'O'):
                    continue
                try:
                    res_num = int(line[22:26].strip())
                    x       = float(line[30:38].strip())
                    y       = float(line[38:46].strip())
                    z       = float(line[46:54].strip())
                    plddt   = float(line[60:66].strip())
                except (ValueError, IndexError):
                    continue
                if res_num not in backbone:
                    backbone[res_num] = {}
                backbone[res_num][atom_name] = np.array([x, y, z])
                if atom_name == 'CA':
                    backbone[res_num]['plddt'] = plddt

    elif struct_path.suffix == '.cif':
        with open(struct_path) as f:
            content = f.read()
        lines        = content.split('\n')
        atom_cols: dict[str, int] = {}
        col_idx      = 0
        in_atom_site = False

        for line in lines:
            s = line.strip()
            if s == 'loop_':
                atom_cols = {}; col_idx = 0; in_atom_site = False
                continue
            if s.startswith('_atom_site.'):
                atom_cols[s.split('.', 1)[1]] = col_idx
                col_idx += 1
                in_atom_site = True
                continue
            if in_atom_site and atom_cols and s and not s.startswith('_') and not s.startswith('#'):
                needed = ['label_atom_id', 'auth_seq_id', 'B_iso_or_equiv',
                          'Cartn_x', 'Cartn_y', 'Cartn_z']
                if not all(k in atom_cols for k in needed):
                    continue
                parts = s.split()
                if len(parts) <= max(atom_cols[k] for k in needed):
                    continue
                try:
                    atom_name = parts[atom_cols['label_atom_id']]
                    if atom_name not in ('N', 'CA', 'C', 'O'):
                        continue
                    res_num = int(parts[atom_cols['auth_seq_id']])
                    x       = float(parts[atom_cols['Cartn_x']])
                    y       = float(parts[atom_cols['Cartn_y']])
                    z       = float(parts[atom_cols['Cartn_z']])
                    plddt   = float(parts[atom_cols['B_iso_or_equiv']])
                except (ValueError, IndexError):
                    continue
                if res_num not in backbone:
                    backbone[res_num] = {}
                backbone[res_num][atom_name] = np.array([x, y, z])
                if atom_name == 'CA':
                    backbone[res_num]['plddt'] = plddt

    return backbone


# ─── DSSP H-bond based Secondary Structure ───────────────────────────────────

def _hbond_energy(donor_N, donor_H, acceptor_O, acceptor_C) -> float:
    def dist(a, b):
        return max(np.linalg.norm(a - b), 0.01)
    r_ON = dist(acceptor_O, donor_N)
    r_CH = dist(acceptor_C, donor_H)
    r_OH = dist(acceptor_O, donor_H)
    r_CN = dist(acceptor_C, donor_N)
    return 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332


def _estimate_H_positions(backbone, res_nums):
    H_pos = {}
    for idx, res in enumerate(res_nums):
        if idx == 0:
            continue
        prev = res_nums[idx - 1]
        if not all(k in backbone[res] for k in ('N', 'CA')) or 'C' not in backbone[prev]:
            continue
        N      = backbone[res]['N']
        C_prev = backbone[prev]['C']
        CA     = backbone[res]['CA']
        v1 = C_prev - N
        v2 = CA - N
        v1n = v1 / (np.linalg.norm(v1) + 1e-10)
        v2n = v2 / (np.linalg.norm(v2) + 1e-10)
        h_dir = -(v1n + v2n)
        norm  = np.linalg.norm(h_dir)
        if norm > 1e-6:
            H_pos[res] = N + h_dir / norm * 1.0
    return H_pos


def compute_dssp_ss(backbone):
    if not backbone:
        return {}
    res_nums = sorted(backbone.keys())
    n        = len(res_nums)
    H_pos    = _estimate_H_positions(backbone, res_nums)

    hbonds = [[False] * n for _ in range(n)]
    for i_idx, donor_res in enumerate(res_nums):
        if donor_res not in H_pos:
            continue
        donor_N = backbone[donor_res]['N']
        donor_H = H_pos[donor_res]
        for j_idx, acc_res in enumerate(res_nums):
            if abs(i_idx - j_idx) < 2:
                continue
            if 'O' not in backbone[acc_res] or 'C' not in backbone[acc_res]:
                continue
            if np.linalg.norm(donor_N - backbone[acc_res]['O']) > 5.5:
                continue
            e = _hbond_energy(donor_N, donor_H, backbone[acc_res]['O'], backbone[acc_res]['C'])
            if e < HBOND_ENERGY_CUTOFF:
                hbonds[i_idx][j_idx] = True

    ss = ['C'] * n
    helix4 = set()
    for i in range(n - 4):
        if hbonds[i + 4][i]:
            helix4.add(i)
    for i in range(n):
        if i in helix4 and (i + 1) in helix4 and (i + 2) in helix4:
            for k in range(i, min(i + 5, n)):
                ss[k] = 'H'

    helix3 = set()
    for i in range(n - 3):
        if hbonds[i + 3][i]:
            helix3.add(i)
    for i in range(n):
        if ss[i] == 'C' and i in helix3 and (i + 1) in helix3:
            for k in range(i, min(i + 4, n)):
                if ss[k] == 'C':
                    ss[k] = 'H'

    strand_candidates = set()
    for i in range(n):
        for j in range(i + 4, n):
            if hbonds[i][j] and hbonds[j][i]:
                strand_candidates.add(i); strand_candidates.add(j)
            if (j + 1 < n and i + 1 < n and hbonds[i][j] and hbonds[j + 1][i + 1]):
                strand_candidates.add(i); strand_candidates.add(j)
    for i in sorted(strand_candidates):
        if ss[i] == 'C':
            ss[i] = 'E'

    ss = _smooth_ss(ss)
    return {res: ss[i] for i, res in enumerate(res_nums)}


def _smooth_ss(ss_list, min_helix=3, min_strand=2):
    result = list(ss_list)
    n = len(result)
    i = 0
    while i < n:
        if result[i] == 'C':
            i += 1; continue
        j = i
        while j < n and result[j] == result[i]:
            j += 1
        run_len = j - i
        min_len = min_helix if result[i] == 'H' else min_strand
        if run_len < min_len:
            for k in range(i, j):
                result[k] = 'C'
        i = j
    return result


def parse_plddt_and_ss(struct_path: Path) -> dict[int, dict]:
    backbone  = parse_backbone(struct_path)
    if not backbone:
        print(f"  [struct] No backbone atoms parsed from {struct_path.name}")
        return {}
    ss_by_res = compute_dssp_ss(backbone)
    features  = {
        res: {'plddt': backbone[res].get('plddt', 50.0), 'ss': ss_by_res.get(res, 'C')}
        for res in backbone
    }
    total = len(features)
    n_H = sum(1 for v in features.values() if v['ss'] == 'H')
    n_E = sum(1 for v in features.values() if v['ss'] == 'E')
    n_C = sum(1 for v in features.values() if v['ss'] == 'C')
    print(f"  [struct] {total} residues | "
          f"H:{n_H} ({n_H/total*100:.0f}%)  "
          f"E:{n_E} ({n_E/total*100:.0f}%)  "
          f"C:{n_C} ({n_C/total*100:.0f}%)")
    return features


# ─── Contact density from PAE ─────────────────────────────────────────────────

def parse_pae_matrix(gene_dir: Path, protein_len: int) -> np.ndarray | None:
    scores_file = find_scores_file(gene_dir)
    if not scores_file:
        return None
    with open(scores_file) as f:
        data = json.load(f)
    if "pae" in data:
        pae = np.array(data["pae"])
    elif "predicted_aligned_error" in data:
        pae = np.array(data["predicted_aligned_error"])
    else:
        return None
    if pae.shape[0] != protein_len:
        print(f"  [PAE] Matrix {pae.shape[0]} ≠ protein {protein_len}")
        return None
    return pae


def compute_contact_density(pae: np.ndarray | None, protein_len: int,
                             window: int = CONTACT_WINDOW,
                             cutoff: float = CONTACT_PAE_CUTOFF) -> list[float]:
    """
    FIX 2: Normalized contact density in [0, 1].

    For each residue i, count how many neighbours within ±window have
    PAE < cutoff (5 Å, not 10 Å), then normalize by window size.

    Using PAE < 5 Å means only genuinely tight, well-determined contacts
    are counted. Normalization removes the boundary artefact where edge
    residues trivially score lower.

    Old behaviour: raw count with PAE < 10  →  almost always 21 (saturated)
    New behaviour: fraction in [0,1], ~0.3–0.9 for typical proteins,
                   low values at disordered loops / domain linkers.
    """
    if pae is None:
        return [float('nan')] * protein_len

    n = pae.shape[0]
    result = []
    for i in range(n):
        lo  = max(0, i - window)
        hi  = min(n, i + window + 1)
        # Exclude self (pae[i,i] = 0 by definition → always passes any cutoff)
        neighbours = np.concatenate([pae[i, lo:i], pae[i, i+1:hi]])
        n_neighbours = len(neighbours)
        if n_neighbours == 0:
            result.append(float('nan'))
        else:
            count = float(np.sum(neighbours < cutoff))
            result.append(count / n_neighbours)   # normalized [0, 1]
    return result


# ─── PAE-based Domain Boundary Detection ─────────────────────────────────────

def detect_pae_domain_boundaries(pae, window=PAE_WINDOW,
                                  threshold=PAE_THRESHOLD,
                                  min_gap=PAE_MIN_GAP):
    n = pae.shape[0]
    if n < 2 * window + min_gap:
        return []
    scores = np.zeros(n)
    for b in range(window, n - window):
        left  = pae[b - window:b, b - window:b]
        right = pae[b:b + window, b:b + window]
        inter_lr = pae[b - window:b, b:b + window]
        inter_rl = pae[b:b + window, b - window:b]
        intra    = (np.mean(left) + np.mean(right)) / 2
        inter    = (np.mean(inter_lr) + np.mean(inter_rl)) / 2
        scores[b] = inter - intra

    candidates = sorted(
        [(scores[b], b) for b in range(window, n - window) if scores[b] > threshold],
        reverse=True,
    )
    selected = set()
    for score, b in candidates:
        if all(abs(b - s) >= min_gap for s in selected):
            selected.add(b)

    boundaries = sorted(selected)
    if boundaries:
        print(f"  [PAE] {len(boundaries)} domain boundaries at AA: {[b+1 for b in boundaries]}")
    return boundaries


# ─── UniProt + InterPro ───────────────────────────────────────────────────────

def enst_to_uniprot(enst_base: str) -> str | None:
    try:
        r = requests.get(
            f"{ENSEMBL_SERVER}/xrefs/id/{enst_base}"
            f"?content-type=application/json;all_levels=1",
            timeout=15,
        )
        if not r.ok:
            return None
        for xref in r.json():
            db = xref.get("dbname", "").lower()
            if "uniprot" in db or "swiss" in db:
                uid = xref.get("primary_id")
                if uid:
                    return uid
    except Exception as e:
        print(f"  [UniProt] {enst_base}: {e}")
    return None


def get_interpro_domains(uniprot: str) -> list[tuple[int, int]]:
    try:
        r = requests.get(INTERPRO_URL.format(uniprot=uniprot), timeout=15)
        if not r.ok:
            return []
        domains = []
        for entry in r.json().get("results", []):
            for protein in entry.get("proteins", []):
                for loc in protein.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        s, e = frag.get("start"), frag.get("end")
                        if s and e:
                            domains.append((int(s), int(e)))
        return domains
    except Exception as e:
        print(f"  [InterPro] {uniprot}: {e}")
        return []


def is_interpro_boundary(res_num, domains, tol=1):
    return int(any(abs(res_num - s) <= tol or abs(res_num - e) <= tol for s, e in domains))


def is_pae_boundary(aa_pos, pae_boundaries, tol=2):
    res_0based = aa_pos - 1
    return int(any(abs(res_0based - b) <= tol for b in pae_boundaries))


# ─── Per-gene pipeline ────────────────────────────────────────────────────────

def process_gene(gene: str, enst_base: str) -> bool:
    csv_path = RESULTS_DIR / f"{gene.lower()}_ribosome_counts.csv"

    # ─── Skip if already done ────────────────────────────────────────────────
    out_path = csv_path.with_name(f"{csv_path.stem}_with_structure.csv")
    if out_path.is_file():
        print(f"  [SKIP] {out_path.name} already exists")
        return True

    if not csv_path.is_file():
        print(f"  No CSV: {csv_path.name} – skipping")
        return False

    fasta_path = find_best_fasta(GENES_DIR, gene, enst_base)

    if not fasta_path:
        print(f"  No FASTA for {gene}/{enst_base} – skipping")
        return False

    print(f"\n{'─'*60}")
    print(f"Gene: {gene}  |  ENST: {enst_base}  |  FASTA: {fasta_path.name}")

    try:
        protein, cds_start, cds_end = get_cds_protein(fasta_path)
    except Exception as e:
        print(f"  CDS extraction failed: {e}")
        return False

    if len(protein) < MIN_PROTEIN_AA:
        print(f"  Protein too short ({len(protein)} AA) – skipping")
        return False

    struct_file = run_colabfold(gene, protein, AF_OUT_DIR)
    if not struct_file:
        return False

    features = parse_plddt_and_ss(struct_file)
    if not features:
        print(f"  No structural features – skipping")
        return False

    gene_dir = AF_OUT_DIR / gene
    pae      = parse_pae_matrix(gene_dir, len(protein))

    contact_density = compute_contact_density(pae, len(protein))
    pae_boundaries  = detect_pae_domain_boundaries(pae) if pae is not None else []

    uniprot = enst_to_uniprot(enst_base)
    domains = get_interpro_domains(uniprot) if uniprot else []
    print(f"  UniProt: {uniprot or '–'}  |  InterPro domains: {len(domains)}")

    df = pd.read_csv(csv_path)

    def codon_to_aa_pos(row) -> int | None:
        if row.get('region') != 'CDS':
            return None
        nt = int(row['nt_start'])
        if nt < cds_start or nt >= cds_end:
            return None
        return (nt - cds_start) // 3 + 1

    df['_aa_pos'] = df.apply(codon_to_aa_pos, axis=1)

    df['plddt'] = df['_aa_pos'].apply(
        lambda p: features[_aa(p)]['plddt']
        if (_aa(p) is not None and _aa(p) in features) else np.nan
    )

    _ss = df['_aa_pos'].apply(
        lambda p: features[_aa(p)]['ss']
        if (_aa(p) is not None and _aa(p) in features) else None
    )
    df['ss_H'] = _ss.apply(lambda s: 1 if s == 'H' else (0 if s is not None else np.nan))
    df['ss_E'] = _ss.apply(lambda s: 1 if s == 'E' else (0 if s is not None else np.nan))
    df['ss_C'] = _ss.apply(lambda s: 1 if s == 'C' else (0 if s is not None else np.nan))

    # Normalized contact density [0, 1]
    df['contact_density'] = df['_aa_pos'].apply(
        lambda p: contact_density[_aa(p) - 1]
        if (_aa(p) is not None and 1 <= _aa(p) <= len(contact_density)) else np.nan
    )

    df['domain_boundary_interpro'] = df['_aa_pos'].apply(
        lambda p: is_interpro_boundary(_aa(p), domains) if _aa(p) is not None else np.nan
    )
    df['domain_boundary_pae'] = df['_aa_pos'].apply(
        lambda p: is_pae_boundary(_aa(p), pae_boundaries) if _aa(p) is not None else np.nan
    )
    df['domain_boundary'] = df['_aa_pos'].apply(
        lambda p: int(
            is_interpro_boundary(_aa(p), domains) == 1 or
            is_pae_boundary(_aa(p), pae_boundaries) == 1
        ) if _aa(p) is not None else np.nan
    )

    df['region_5UTR'] = (df['region'] == '5UTR').astype(int)
    df['region_CDS']  = (df['region'] == 'CDS').astype(int)
    df['region_3UTR'] = (df['region'] == '3UTR').astype(int)

    df.drop(columns=['_aa_pos'], inplace=True)
    df.to_csv(out_path, index=False)

    cds_rows = df[df['region_CDS'] == 1]
    print(f"  Saved: {out_path.name}")
    print(f"  CDS rows: {len(cds_rows)} | with pLDDT: {cds_rows['plddt'].notna().sum()}")
    print(f"  Contact density stats (CDS): "
          f"mean={cds_rows['contact_density'].mean():.3f}  "
          f"std={cds_rows['contact_density'].std():.3f}  "
          f"min={cds_rows['contact_density'].min():.3f}  "
          f"max={cds_rows['contact_density'].max():.3f}")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    AF_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ISOFORM_JSON.is_file():
        print(f"ERROR: {ISOFORM_JSON} not found")
        return

    with open(ISOFORM_JSON) as f:
        isoforms = json.load(f)

    success, failed, skipped = [], [], []

    for gene, info in isoforms.items():
        if info.get("status") != "ok":
            print(f"Skipping {gene} ({info.get('status')})")
            skipped.append(gene)
            continue
        enst_base = info.get("best_isoform", "")
        if not enst_base:
            skipped.append(gene)
            continue
        try:
            ok = process_gene(gene.upper(), enst_base)
        except Exception as e:
            print(f"  UNHANDLED ERROR in {gene}: {e}")
            ok = False
        (success if ok else failed).append(gene)

    print(f"\n{'═'*60}")
    print(f"Success : {len(success)}")
    print(f"Failed  : {len(failed)}")
    print(f"Skipped : {len(skipped)}")
    if failed:
        print(f"Failed genes: {', '.join(failed)}")
    print(f"Results : {RESULTS_DIR}/*_with_structure.csv")


if __name__ == '__main__':
    main()