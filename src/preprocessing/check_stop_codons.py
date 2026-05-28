#!/usr/bin/env python3
"""
Check all genes for stop codons within the CDS.
Strategy: find the first ATG → translate(to_stop=True) → that is the true CDS.
Everything after the first stop is treated as 3'UTR and ignored.
"""

from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
MANE_FILE = BASE / "data/genes/MANE.GRCh38.v1.5.refseq_rna.fna"
OUTPUT_DIR = BASE / "data/proteins"
JSON_OUT   = BASE / "data/stop_codon_check_all.json"

GENES = [
    "GAPDH", "LDHA", "LDHB", "ENO1", "PKM", "ALDOA", "PGK1", "IDH1", "IDH2", "MDH2",
    "G6PD", "PFKL", "PFKM", "ACO2", "CS", "SDHA", "FH", "GOT2", "ACTB", "ACTG1",
    "TUBB", "TUBA1B", "TUBA1A", "VIM", "LMNA", "LMNB1", "EEF1A1", "EEF2", "EIF4A1",
    "EIF3A", "EIF2S1", "EEF1G", "HSP90AA1", "HSP90AB1", "HSPA8", "HSPA5", "HSPD1",
    "CCT2", "CCT3", "CCT4", "DNAJA1", "DNAJB1", "MAPK1", "MAPK3", "AKT1", "AKT2",
    "PRKACA", "GSK3B", "SRC", "EGFR", "PIK3CA", "HNRNPA1", "HNRNPC", "HNRNPK",
    "RFC1", "TOP1", "PARP1", "JUN", "FOS", "MYC", "TP53", "STAT3", "NFKB1",
    "CCNB1", "CCNE1", "CDC20", "PLK1", "AURKB", "CASP7", "CTSB", "CTSD", "PSMD1",
    "PSMD2", "CAT", "TXNRD1", "VCP", "NSF", "ALB", "SERPINA1", "F9", "INSR",
    "NCL", "RPLP0",
]
# ─────────────────────────────────────────────────────────────────────────────


def gene_matches(description: str, gene: str) -> bool:
    """
    Check whether a gene symbol appears in the FASTA header.
    Only matches whole words, e.g. 'LDHA' does not match 'LDHB'.
    """
    desc = description.upper()
    gene = gene.upper()
    # Suche nach (GENE) oder [GENE] oder Leerzeichen-begrenzt
    for pattern in [f"({gene})", f"[{gene}]", f" {gene},", f" {gene} ", f" {gene})"]:
        if pattern in desc:
            return True
    return False


def extract_protein(seq: str) -> dict:
    seq = seq.upper()
    atg_pos = seq.find("ATG")
    if atg_pos == -1:
        return {"status": "NO_ATG", "protein": None, "protein_length": 0, "atg_position": None}

    # Trim to a multiple of 3 starting from ATG
    from_atg = seq[atg_pos:]
    from_atg = from_atg[:len(from_atg) // 3 * 3]

    # Without to_stop — full protein including all '*'
    protein_full = str(Seq(from_atg).translate())

    # Find the first stop — that marks the end of the CDS
    first_stop_idx = protein_full.find('*')

    if first_stop_idx == -1:
        # No stop codon found — missing terminal stop is a problem
        return {
            "status": "PROBLEM",
            "atg_position": atg_pos + 1,
            "protein_length": len(protein_full),
            "terminal_stop_codon": "",
            "has_terminal_stop": False,
            "has_internal_stop": False,
            "internal_stop_positions": [],
            "note": "No stop codon found",
            "protein": protein_full,
        }

    # CDS-Protein = alles VOR dem ersten Stop
    protein_cds = protein_full[:first_stop_idx]

    # Terminal Stop Codon bestimmen
    terminal_stop_nt_pos = atg_pos + first_stop_idx * 3
    terminal_stop_codon = seq[terminal_stop_nt_pos:terminal_stop_nt_pos + 3]

    # Check for '*' within the CDS (before the first stop). By definition of the
    # first stop this cannot happen, but a very early stop (<50 AA) suggests the
    # first ATG is not the true start codon.
    internal_stops = [i for i, aa in enumerate(protein_full[:first_stop_idx]) if aa == '*']

    # Flag suspiciously early stops — likely a wrong start ATG
    suspiciously_short = first_stop_idx < 50

    status = "OK"
    if internal_stops:
        status = "PROBLEM"
    elif suspiciously_short:
        status = "PROBLEM"
    elif not terminal_stop_codon in {"TAA", "TAG", "TGA"}:
        status = "PROBLEM"

    return {
        "status": status,
        "atg_position": atg_pos + 1,  # 1-based
        "protein_length": len(protein_cds),
        "terminal_stop_codon": terminal_stop_codon,
        "has_terminal_stop": terminal_stop_codon in {"TAA", "TAG", "TGA"},
        "has_internal_stop": len(internal_stops) > 0,
        "internal_stop_positions": [i + 1 for i in internal_stops],  # 1-based AA-Position
        "suspiciously_short": suspiciously_short,
        "protein": protein_cds,
    }


def main():
    if not MANE_FILE.is_file():
        print(f"ERROR: MANE file not found:\n  {MANE_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read all relevant entries from the MANE file ──
    print(f"Reading {MANE_FILE.name} ...")
    gene_records: dict[str, list] = {g: [] for g in GENES}

    for record in SeqIO.parse(MANE_FILE, "fasta"):
        for gene in GENES:
            if gene_matches(record.description, gene):
                gene_records[gene].append(record)

    # ── Analysis ──
    results = {}
    clean_genes = []
    problem_genes = []
    not_found_genes = []

    print(f"\n{'Gene':<12} {'Accession':<16} {'ATG':<8} {'Protein AA':<12} {'Term.Stop':<12} {'Status'}")
    print("─" * 72)

    for gene in GENES:
        records = gene_records[gene]

        if not records:
            not_found_genes.append(gene)
            results[gene] = {"status": "NOT_FOUND"}
            print(f"{gene:<12} {'–':<16} {'–':<8} {'–':<12} {'–':<12} ❓ not found")
            continue

        # Multiple hits: prefer NM_ accessions (MANE Select)
        record = next((r for r in records if r.id.startswith("NM_")), records[0])
        result = extract_protein(str(record.seq))
        result["accession"] = record.id
        result["header"] = record.description
        results[gene] = result

        status_icon = "✅ CLEAN" if result["status"] == "OK" else "❌ PROBLEM"
        print(f"{gene:<12} {record.id:<16} {str(result['atg_position']):<8} "
              f"{str(result['protein_length']):<12} {str(result['terminal_stop_codon']):<12} {status_icon}")

        if result["status"] == "OK":
            clean_genes.append(gene)
            # Save FASTA for AlphaFold
            fasta_out = OUTPUT_DIR / f"{gene}_alphafold.fasta"
            fasta_out.write_text(
                f">{gene} | {record.id} | AlphaFold-ready | {result['protein_length']} AA\n"
                f"{result['protein']}\n"
            )
        else:
            problem_genes.append(gene)

    # ── Summary ──
    print("\n" + "═" * 72)
    print(f"TOTAL:           {len(GENES)} genes")
    print(f"✅ Clean:        {len(clean_genes)}")
    print(f"❌ Problems:     {len(problem_genes)}")
    print(f"❓ Not found:    {len(not_found_genes)}")

    if problem_genes:
        print(f"\nProblem genes: {', '.join(problem_genes)}")

    if not_found_genes:
        print(f"Not found: {', '.join(not_found_genes)}")

    if clean_genes:
        print(f"\n✅ AlphaFold FASTA files saved to:\n   {OUTPUT_DIR}")

    # ── Save JSON (excluding protein sequences for readability) ──
    json_out_data = {
        g: {k: v for k, v in d.items() if k != "protein"}
        for g, d in results.items()
    }
    JSON_OUT.write_text(json.dumps(json_out_data, indent=2))
    print(f"\nJSON report: {JSON_OUT}")


if __name__ == "__main__":
    main()