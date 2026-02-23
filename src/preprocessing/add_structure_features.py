#!/usr/bin/env python3
"""
Add structural features from AlphaFold to ribosome profiling CSVs
Uses isoform_mapping.json for automatic isoform selection
Includes one-hot encoding for secondary structure
"""

import json
import pandas as pd
from pathlib import Path
import requests
import subprocess
import tempfile
import numpy as np
from Bio import SeqIO

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

BASE_DIR = Path("/Users/jonathanopitz/Desktop/Master")
RESULTS_DIR = BASE_DIR / "data/results"
MAPPING_FILE = BASE_DIR / "isoform_mapping.json"
GENCODE_FA = BASE_DIR / "data/genes/gencode/gencode.v48.pc_transcripts.fa"

ENSEMBL_SERVER = "https://rest.ensembl.org"
ALPHAFOLD_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.cif"
ALPHAFOLD_PAE_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-predicted_aligned_error_v4.json"
INTERPRO_URL = "https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry?accession={uniprot}"

def enst_to_uniprot(enst: str) -> str | None:
    """ENST → UniProt accession via Ensembl REST API"""
    ext = f"/map/id/{enst}?content-type=application/json"
    try:
        r = requests.get(ENSEMBL_SERVER + ext, headers={"Content-Type": "application/json"})
        if r.ok:
            data = r.json()
            for m in data.get("mappings", []):
                if m.get("type") == "UniProt":
                    return m["id"]
    except Exception as e:
        print(f"ENST → UniProt failed for {enst}: {e}")
    return None

def get_interpro_domains(uniprot: str) -> list[tuple[int, int]]:
    """Fetch domain start/end from InterPro API"""
    url = INTERPRO_URL.format(uniprot=uniprot)
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return []
        data = r.json()
        domains = []
        for entry in data.get("results", []):
            for loc in entry.get("entry_locations", []):
                for frag in loc.get("fragments", []):
                    start = frag.get("start", 0)
                    end = frag.get("end", 0)
                    if start > 0 and end > 0:
                        domains.append((start, end))
        return domains
    except Exception as e:
        print(f"InterPro API failed for {uniprot}: {e}")
        return []

def is_domain_boundary(res_num: int, domains: list[tuple[int, int]], tolerance: int = 1) -> int:
    """Flag 1 if residue is at/near a domain boundary"""
    for start, end in domains:
        if abs(res_num - start) <= tolerance or abs(res_num - end) <= tolerance:
            return 1
    return 0

def download_file(url: str, suffix: str) -> Path | None:
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(r.content)
            return Path(tmp.name)
    except Exception as e:
        print(f"Download failed: {url} → {e}")
        return None

def parse_plddt_and_ss(cif_path: Path) -> dict:
    """Parse pLDDT and secondary structure via DSSP"""
    # Convert CIF to PDB for DSSP
    pdb_path = cif_path.with_suffix('.pdb')
    subprocess.run(["obabel", "-icif", str(cif_path), "-opdb", "-O", str(pdb_path)], check=False)

    if not pdb_path.is_file():
        print("CIF → PDB conversion failed")
        return {}

    dssp_out = cif_path.with_suffix('.dssp')
    subprocess.run(["mkdssp", "-i", str(pdb_path), "-o", str(dssp_out)], check=False)

    features = {}
    if dssp_out.is_file():
        with open(dssp_out) as f:
            in_table = False
            for line in f:
                if line.startswith("  #  RESIDUE"):
                    in_table = True
                    continue
                if in_table and len(line) > 100:
                    try:
                        res_num = int(line[5:10].strip())
                        ss = line[16].strip() or 'C'
                        plddt_str = line[100:105].strip()
                        plddt = float(plddt_str) if plddt_str.replace('.', '').isdigit() else 50.0
                        features[res_num] = {'ss': ss, 'plddt': plddt}
                    except:
                        pass

    pdb_path.unlink(missing_ok=True)
    dssp_out.unlink(missing_ok=True)
    return features

def parse_pae_contact_density(pae_path: Path, length: int) -> list:
    """Simple per-residue contact density from PAE matrix"""
    if not pae_path.is_file():
        return [0.0] * length

    with open(pae_path) as f:
        pae_data = json.load(f)

    pae_matrix = np.array(pae_data["predicted_aligned_error"])
    if pae_matrix.shape[0] != length:
        print("PAE matrix size mismatch")
        return [0.0] * length

    density = []
    for i in range(length):
        neighbors = pae_matrix[i, max(0, i-10):min(length, i+11)]
        contacts = np.sum(neighbors < 10)
        density.append(float(contacts))

    return density

def add_features_to_csv(csv_path: Path, enst: str):
    """Main function: add all features + one-hot encode SS"""
    df = pd.read_csv(csv_path)

    uniprot = enst_to_uniprot(enst)
    if not uniprot:
        print(f"No UniProt mapping for {enst}")
        return

    print(f"Adding structural features for {uniprot} (ENST: {enst})")

    cif_path = download_file(ALPHAFOLD_CIF_URL.format(uniprot=uniprot), '.cif')
    pae_path = download_file(ALPHAFOLD_PAE_URL.format(uniprot=uniprot), '.json')

    if not cif_path:
        print("No AlphaFold model available")
        return

    # pLDDT + secondary structure
    features = parse_plddt_and_ss(cif_path)

    # Contact density
    contact_dens = parse_pae_contact_density(pae_path, len(features))

    # Domain boundaries via InterPro
    domains = get_interpro_domains(uniprot)
    domain_boundaries = [is_domain_boundary(p, domains) for p in range(1, len(features)+1)]

    # Add columns (1-based codon_pos → AA index)
    df['plddt'] = df['codon_pos'].apply(lambda p: features.get(p, {}).get('plddt', 50.0))
    df['secondary_structure'] = df['codon_pos'].apply(lambda p: features.get(p, {}).get('ss', 'C'))
    df['contact_density'] = df['codon_pos'].apply(lambda p: contact_dens[p-1] if p <= len(contact_dens) else 0.0)
    df['domain_boundary'] = df['codon_pos'].apply(lambda p: domain_boundaries[p-1] if p <= len(domain_boundaries) else 0)

    # One-hot encode secondary structure
    ss_dummies = pd.get_dummies(df['secondary_structure'], prefix='ss')
    df = pd.concat([df, ss_dummies], axis=1)

    # Optional: Drop original categorical column
    # df = df.drop(columns=['secondary_structure'])

    # Save updated CSV
    new_path = csv_path.with_name(f"{csv_path.stem}_with_structure{csv_path.suffix}")
    df.to_csv(new_path, index=False)
    print(f"Updated CSV saved: {new_path}")

    # Cleanup
    if cif_path: cif_path.unlink(missing_ok=True)
    if pae_path: pae_path.unlink(missing_ok=True)

def main():
    if not MAPPING_FILE.is_file():
        print("Missing isoform_mapping.json – run the batch script first")
        return

    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    for gene, info in mapping.items():
        csv_path = RESULTS_DIR / f"{gene.lower()}_ribosome_counts.csv"
        if not csv_path.is_file():
            print(f"No CSV for {gene} – skipping")
            continue

        enst = info.get("enst")
        if not enst:
            print(f"No ENST for {gene}")
            continue

        add_features_to_csv(csv_path, enst)

    print("\nAll structural features added. Check files ending with _with_structure.csv")

if __name__ == '__main__':
    main()