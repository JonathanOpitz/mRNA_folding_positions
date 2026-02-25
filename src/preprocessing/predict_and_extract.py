#!/usr/bin/env python3
"""
Local AlphaFold prediction pipeline using isoform_mapping.json
Runs a NEW prediction for every sequence
Adds pLDDT, one-hot DSSP SS, PAE contact density, domain boundary flag
Preserves all original data in CSVs

"""

import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, DSSP
import warnings
import shutil

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

BASE_DIR = Path("/Users/jonathanopitz/Desktop/Master")
FASTA_DIR = BASE_DIR / "data" / "genes"
PRED_DIR  = BASE_DIR / "predictions"
JSON_PATH = BASE_DIR / "isoform_mapping.json"
RESULTS_DIR = BASE_DIR / "data" / "results"

COLABFOLD_ARGS = [
    "--amber",
    "--num-recycle", "3",
    "--num-models", "1",
    "--random-seed", "42",
]

LOW_PLDDT = 50
HIGH_PAE_NEIGHBOR = 15

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def find_fasta_for_enst(gene: str, enst: str) -> Path | None:
    pattern = f"{gene}_{enst.split('.')[0]}"
    for f in FASTA_DIR.glob(f"{pattern}*.fasta"):
        return f
    print(f"No FASTA for {gene} ({enst})")
    return None


def translate_to_protein(fasta_path: Path) -> Path:
    record = SeqIO.read(fasta_path, "fasta")
    seq = record.seq[:len(record.seq) // 3 * 3]
    protein_seq = seq.translate(to_stop=True)
    protein_id = f"{record.id}_protein"
    protein_record = SeqRecord(protein_seq, id=protein_id, description="Translated")

    protein_fasta = FASTA_DIR / f"{protein_id}.fasta"
    SeqIO.write(protein_record, protein_fasta, "fasta")
    return protein_fasta


def run_fresh_prediction(protein_fasta: Path) -> Path:
    out_dir = PRED_DIR / protein_fasta.stem

    if out_dir.exists():
        print(f"Deleting old prediction: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["colabfold_batch", str(protein_fasta), str(out_dir)] + COLABFOLD_ARGS
    print(f"Starting fresh prediction:\n{' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:")
        print(result.stderr)
        raise RuntimeError("Prediction failed")

    print("Fresh prediction done.")
    return out_dir


def extract_features(pred_dir: Path, seq_len: int) -> dict:
    features = {
        "plddt": [np.nan] * seq_len,
        "ss": ["C"] * seq_len,
        "contact_density": [0.0] * seq_len,
        "domain_boundary": [0] * seq_len,
        "error": None
    }

    pdb_files = list(pred_dir.glob("*rank_001*.pdb"))
    if not pdb_files:
        features["error"] = "No PDB"
        return features
    pdb_path = pdb_files[0]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)[0]  # [0] = first Model

    # pLDDT — structure is already a Model, so iterate chains → residues directly
    plddt_list = []
    for chain in structure:
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue
            if residue.has_id("CA"):
                plddt_list.append(residue["CA"].bfactor)

    if plddt_list:
        features["plddt"] = plddt_list[:seq_len] + [np.nan] * (seq_len - len(plddt_list))

    # DSSP — explicitly use mkdssp binary so Biopython doesn't try to parse as mmCIF
    try:
        dssp = DSSP(structure, str(pdb_path), dssp="mkdssp")
        ss_list = ["C"] * seq_len
        for key in dssp.keys():
            res_id = key[1][1] - 1
            if 0 <= res_id < seq_len:
                ss_list[res_id] = dssp[key][2]
        features["ss"] = ss_list
        print(f"  DSSP OK — {len(dssp)} residues assigned")
    except Exception as e:
        print(f"  DSSP failed (ss will be all 'C'): {e}")
        features["ss_error"] = str(e)

    # PAE
    pae_files = list(pred_dir.glob("*predicted_aligned_error*.json"))
    if pae_files:
        with open(pae_files[0]) as f:
            pae_data = json.load(f)
        pae = np.array(pae_data["predicted_aligned_error"])
        if pae.shape[0] >= seq_len:
            for i in range(seq_len):
                neighbors = pae[i, :seq_len]
                features["contact_density"][i] = float(np.sum(neighbors < 10))

    # Domain boundary
    plddt_arr = np.array(features["plddt"])
    for i in range(seq_len):
        low_plddt = not np.isnan(plddt_arr[i]) and plddt_arr[i] < LOW_PLDDT
        loose_contact = features["contact_density"][i] < 5
        features["domain_boundary"][i] = 1 if low_plddt or loose_contact else 0

    return features


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH) as f:
        isoform_map = json.load(f)

    success = 0
    failed = 0

    for gene, data in isoform_map.items():
        enst = data.get("enst")
        if not enst:
            print(f"No ENST → skipping {gene}")
            failed += 1
            continue

        csv_path = RESULTS_DIR / f"{gene.lower()}_ribosome_counts.csv"
        if not csv_path.is_file():
            print(f"No CSV → skipping {gene}")
            failed += 1
            continue

        print(f"\n=== {gene} ({enst}) ===")

        fasta_path = find_fasta_for_enst(gene, enst)
        if not fasta_path:
            failed += 1
            continue

        protein_fasta = translate_to_protein(fasta_path)
        seq_len = len(SeqIO.read(protein_fasta, "fasta").seq)

        pred_dir = run_fresh_prediction(protein_fasta)

        features = extract_features(pred_dir, seq_len)
        if features.get("error"):
            print(f"Extraction failed: {features['error']}")
            failed += 1
            continue

        df = pd.read_csv(csv_path)

        pos_col = "codon_pos" if "codon_pos" in df.columns else "residue_pos"
        if pos_col not in df.columns:
            print(f"No position column → skipping {gene}")
            failed += 1
            continue

        df["plddt"]          = df[pos_col].apply(lambda p: features["plddt"][int(p)-1]          if 1 <= int(p) <= seq_len else np.nan)
        df["ss"]             = df[pos_col].apply(lambda p: features["ss"][int(p)-1]             if 1 <= int(p) <= seq_len else "C")
        df["contact_density"]= df[pos_col].apply(lambda p: features["contact_density"][int(p)-1] if 1 <= int(p) <= seq_len else 0.0)
        df["domain_boundary"]= df[pos_col].apply(lambda p: features["domain_boundary"][int(p)-1] if 1 <= int(p) <= seq_len else 0)

        ss_onehot = pd.get_dummies(df["ss"], prefix="ss")
        df = pd.concat([df, ss_onehot], axis=1)

        new_path = csv_path.with_name(f"{csv_path.stem}_with_structure{csv_path.suffix}")
        df.to_csv(new_path, index=False)
        print(f"Saved: {new_path}")

        success += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total genes:     {len(isoform_map)}")
    print(f"Successful:      {success}")
    print(f"Failed/skipped:  {failed}")
    print(f"Look for *_with_structure.csv in {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()