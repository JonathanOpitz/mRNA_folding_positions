#!/usr/bin/env python3
"""
Add AlphaFold-derived structural features to existing ribosome counts CSVs
Uses saved isoform_mapping.json to select the correct isoform/model automatically
Adds columns: plddt, secondary_structure, domain_boundary, contact_density
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

def enst_to_uniprot(enst: str) -> str | None:
    """ENST → UniProt accession via Ensembl REST"""
    ext = f"/map/id/{enst}?content-type=application/json"
    try:
        r = requests.get(ENSEMBL_SERVER + ext, headers={"Content-Type": "application/json"})
        if not r.ok:
            return None
        data = r.json()
        for m in data.get("mappings", []):
            if m.get("type") == "UniProt":
                return m["id"]
    except Exception as e:
        print(f"ENST → UniProt failed for {enst}: {e}")
    return None

def download_file(url: str, suffix: str) -> Path | None:
    """Download AlphaFold CIF or JSON"""
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
    """Very simple parser: residue → pLDDT + secondary structure (from DSSP)"""
    # Convert CIF → PDB for DSSP
    pdb_path = cif_path.with_suffix('.pdb')
    subprocess.run(["obabel", "-icif", str(cif_path), "-opdb", "-O", str(pdb_path)], check=False)

    if not pdb_path.is_file():
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
                        ss = line[16].strip() or 'C'  # H,E,B,T,S,G,I → else coil
                        plddt_str = line[100:105].strip()
                        plddt = float(plddt_str) if plddt_str.replace('.', '').isdigit() else 50.0
                        features[res_num] = {'ss': ss, 'plddt': plddt}
                    except:
                        pass

    # Cleanup temp files
    pdb_path.unlink(missing_ok=True)
    dssp_out.unlink(missing_ok=True)

    return features

def parse_pae_and_contact_density(pae_path: Path, length: int) -> list:
    """Compute simple per-residue contact density from PAE matrix"""
    if not pae_path.is_file():
        return [0.0] * length

    import json
    with open(pae_path) as f:
        pae_data = json.load(f)

    pae_matrix = np.array(pae_data["predicted_aligned_error"])
    if pae_matrix.shape[0] != length:
        print("PAE matrix size mismatch")
        return [0.0] * length

    # Contact density = number of residues with PAE < 10 Å within ±10 residues
    density = []
    for i in range(length):
        contacts = np.sum(pae_matrix[i, max(0, i-10):min(length, i+11)] < 10)
        density.append(float(contacts))

    return density

def add_features_to_csv(csv_path: Path, enst: str):
    df = pd.read_csv(csv_path)

    uniprot = enst_to_uniprot(enst)
    if not uniprot:
        print(f"Could not map {enst} to UniProt")
        return

    print(f"Using AlphaFold model for UniProt {uniprot}")

    cif_path = download_file(ALPHAFOLD_CIF_URL.format(uniprot=uniprot), '.cif')
    pae_path = download_file(ALPHAFOLD_PAE_URL.format(uniprot=uniprot), '.json')

    if not cif_path:
        print("No AlphaFold model available")
        return

    # Get pLDDT + SS
    features = parse_plddt_and_ss(cif_path)

    # Contact density
    contact_dens = parse_pae_and_contact_density(pae_path, len(features))

    # Domain boundary (simple placeholder: could use Pfam later)
    domain_boundary = [0] * len(features)  # 0 = no boundary, 1 = boundary
    # TODO: Add real Pfam/InterPro scan here

    # Map to codons (assume 1 AA = 1 codon for now; adjust if UTRs present)
    df['plddt'] = df['codon_pos'].apply(lambda p: features.get(p, {}).get('plddt', 50.0))
    df['secondary_structure'] = df['codon_pos'].apply(lambda p: features.get(p, {}).get('ss', 'C'))
    df['contact_density'] = df['codon_pos'].apply(lambda p: contact_dens[p-1] if p <= len(contact_dens) else 0.0)
    df['domain_boundary'] = df['codon_pos'].apply(lambda p: domain_boundary[p-1] if p <= len(domain_boundary) else 0)

    # Save updated CSV
    new_path = csv_path.with_name(f"{csv_path.stem}_with_structure{csv_path.suffix}")
    df.to_csv(new_path, index=False)
    print(f"Updated file saved: {new_path}")

    # Cleanup
    if cif_path: cif_path.unlink(missing_ok=True)
    if pae_path: pae_path.unlink(missing_ok=True)

def main():
    if not MAPPING_FILE.is_file():
        print("Run the batch script first to create isoform_mapping.json")
        return

    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    for gene, info in mapping.items():
        csv_path = RESULTS_DIR / f"{gene.lower()}_ribosome_counts.csv"
        if not csv_path.is_file():
            print(f"No CSV found for {gene} – skipping")
            continue

        enst = info.get("enst")
        if not enst:
            print(f"No ENST found for {gene}")
            continue

        add_features_to_csv(csv_path, enst)

    print("\nAll structural features added. Check files ending with _with_structure.csv")

if __name__ == '__main__':
    main()