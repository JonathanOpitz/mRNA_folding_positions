#!/usr/bin/env python3
"""
Add structural features (AlphaFold / ColabFold) to de novo / simulated ribosome count CSVs.
Proteine sind identisch mit wildtype → Strukturen werden wiederverwendet.
MRNA-Sequenz ist anders → CDS-Koordinaten werden direkt aus der simulated CSV extrahiert.
"""

import json
import os
import re
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parents[2]
SIMULATED_DIR     = BASE_DIR / "data/ribo_counts_simulated"
AF_OUT_DIR        = BASE_DIR / "data/alphafold_results"
ISOFORM_JSON      = BASE_DIR / "isoform_selection.json"

INTERPRO_URL      = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot}/?format=json"
ENSEMBL_SERVER    = "https://rest.ensembl.org"

# PAE / Contact parameters (identisch mit Original-Script)
PAE_WINDOW        = 10
PAE_THRESHOLD     = 15.0
PAE_MIN_GAP       = 15
CONTACT_PAE_CUTOFF = 5.0
CONTACT_WINDOW    = 10

HBOND_ENERGY_CUTOFF = -0.5   # kcal/mol (Kabsch & Sander 1983)

# ─── HELPER FUNCTIONS (alle aus dem Original-Script übernommen) ──────────────

def _aa(p) -> int | None:
    if p is None:
        return None
    try:
        if np.isnan(float(p)):
            return None
    except (TypeError, ValueError):
        return None
    return int(p)


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
    if pae is None:
        return [float('nan')] * protein_len

    n = pae.shape[0]
    result = []
    for i in range(n):
        lo  = max(0, i - window)
        hi  = min(n, i + window + 1)
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


# ─── CDS-EXTRAKTION AUS SIMULATED CSV ────────────────────────────────────────
def get_cds_coords_from_simulated_csv(csv_path: Path) -> tuple[int, int]:
    """Extrahiert CDS-Start/Ende direkt aus der simulated CSV."""
    df = pd.read_csv(csv_path)

    if 'region' not in df.columns or 'nt_start' not in df.columns:
        raise ValueError(f"CSV {csv_path.name} hat keine 'region' oder 'nt_start' Spalten.")

    cds_df = df[df['region'].astype(str).str.strip().str.upper() == 'CDS'].copy()
    if cds_df.empty:
        raise ValueError(f"Keine CDS-Zeilen in {csv_path.name}")

    nt_starts = pd.to_numeric(cds_df['nt_start'], errors='coerce')
    cds_start = int(nt_starts.min())
    num_codons = len(cds_df)
    cds_end = cds_start + num_codons * 3

    print(f"  [CDS simulated] {num_codons} Codons → nt {cds_start}–{cds_end} (extrahiert aus CSV)")
    return cds_start, cds_end


# ─── FEATURE-PIPELINE FÜR EINE DATEI ─────────────────────────────────────────
def process_simulated_file(csv_path: Path, isoforms: dict) -> bool:
    stem = csv_path.stem
    out_path = csv_path.with_name(f"{stem}_with_structure.csv")

    if out_path.is_file():
        print(f"  [SKIP] {out_path.name} existiert bereits")
        return True

    # Gene aus Dateinamen extrahieren (z. B. aco2_gemorna_simulated_ribo → ACO2)
    gene_match = re.match(r'^([A-Za-z0-9]+)', stem)
    gene = gene_match.group(1).upper() if gene_match else None
    if not gene:
        print(f"  [ERROR] Konnte Gene-Namen nicht aus {csv_path.name} extrahieren")
        return False

    print(f"\n{'─'*70}")
    print(f"Datei: {csv_path.name}  →  Gene: {gene}")

    # CDS-Koordinaten aus CSV
    try:
        cds_start, cds_end = get_cds_coords_from_simulated_csv(csv_path)
    except Exception as e:
        print(f"  [CDS] Fehler: {e}")
        return False

    # AlphaFold-Struktur wiederverwenden
    gene_dir = AF_OUT_DIR / gene
    struct_file = find_struct_file(gene_dir)
    if not struct_file:
        print(f"  [AF] Keine Struktur für {gene} gefunden (zuerst wildtype-Script ausführen!)")
        return False

    print(f"  [AF] Wiederverwende → {struct_file.name}")

    features = parse_plddt_and_ss(struct_file)
    if not features:
        print("  [struct] Konnte Features nicht parsen")
        return False

    protein_len = max(features.keys()) if features else 0
    pae = parse_pae_matrix(gene_dir, protein_len)
    contact_density = compute_contact_density(pae, protein_len)
    pae_boundaries = detect_pae_domain_boundaries(pae) if pae is not None else []

    # UniProt + InterPro (Protein identisch)
    enst_base = isoforms.get(gene.lower(), {}).get("best_isoform", "")
    uniprot = enst_to_uniprot(enst_base) if enst_base else None
    domains = get_interpro_domains(uniprot) if uniprot else []
    print(f"  [InterPro] UniProt: {uniprot or '–'}  |  Domains: {len(domains)}")

    # CSV laden
    df = pd.read_csv(csv_path)

    def codon_to_aa_pos(row) -> int | None:
        if str(row.get('region', '')).upper() != 'CDS':
            return None
        try:
            nt = int(row['nt_start'])
        except (ValueError, TypeError):
            return None
        if nt < cds_start or nt >= cds_end:
            return None
        return (nt - cds_start) // 3 + 1

    df['_aa_pos'] = df.apply(codon_to_aa_pos, axis=1)

    # Struktur-Features hinzufügen (exakt wie im Original)
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
            (is_interpro_boundary(_aa(p), domains) == 1) or
            (is_pae_boundary(_aa(p), pae_boundaries) == 1)
        ) if _aa(p) is not None else np.nan
    )

    df['region_5UTR'] = (df['region'].astype(str).str.upper() == '5UTR').astype(int)
    df['region_CDS']  = (df['region'].astype(str).str.upper() == 'CDS').astype(int)
    df['region_3UTR'] = (df['region'].astype(str).str.upper() == '3UTR').astype(int)

    df.drop(columns=['_aa_pos'], errors='ignore', inplace=True)
    df.to_csv(out_path, index=False)

    cds_rows = df[df['region_CDS'] == 1]
    print(f"  [FERTIG] {out_path.name} gespeichert")
    print(f"  CDS-Zeilen: {len(cds_rows)} | mit pLDDT: {cds_rows['plddt'].notna().sum()}")
    return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)

    # Isoform-Info für InterPro
    if ISOFORM_JSON.is_file():
        with open(ISOFORM_JSON) as f:
            isoforms = json.load(f)
    else:
        isoforms = {}
        print(f"[WARN] {ISOFORM_JSON} nicht gefunden → InterPro deaktiviert")

    csv_files = sorted(SIMULATED_DIR.glob("*_simulated_ribo.csv"))
    print(f"→ {len(csv_files)} simulated Dateien gefunden\n")

    success = 0
    for csv_path in csv_files:
        if process_simulated_file(csv_path, isoforms):
            success += 1

    print(f"\n{'═'*70}")
    print(f"Fertig! {success}/{len(csv_files)} Dateien mit Struktur-Features angereichert.")
    print(f"Ergebnisse liegen als *_with_structure.csv im Ordner:")
    print(f"   {SIMULATED_DIR}")


if __name__ == '__main__':
    main()