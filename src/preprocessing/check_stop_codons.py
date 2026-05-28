#!/usr/bin/env python3
"""
Prüft für alle Gene ob Stop-Codons in der CDS vorkommen.
Strategie: Erstes ATG finden → translate(to_stop=True) → das ist die echte CDS.
Alles nach dem ersten Stop ist 3'UTR und wird ignoriert.
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
    Prüft ob ein Gen-Symbol im FASTA-Header vorkommt.
    Matcht nur auf ganzen Wörtern, z.B. 'LDHA' matcht nicht auf 'LDHB'.
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

    # Trim auf Vielfaches von 3 ab ATG
    from_atg = seq[atg_pos:]
    from_atg = from_atg[:len(from_atg) // 3 * 3]

    # OHNE to_stop → volles Protein inkl. aller '*'
    protein_full = str(Seq(from_atg).translate())

    # Ersten Stop finden → das ist das Ende der CDS
    first_stop_idx = protein_full.find('*')

    if first_stop_idx == -1:
        # Kein Stop-Codon gefunden (kein terminaler Stop → Problem)
        return {
            "status": "PROBLEM",
            "atg_position": atg_pos + 1,
            "protein_length": len(protein_full),
            "terminal_stop_codon": "",
            "has_terminal_stop": False,
            "has_internal_stop": False,
            "internal_stop_positions": [],
            "note": "Kein Stop-Codon gefunden",
            "protein": protein_full,
        }

    # CDS-Protein = alles VOR dem ersten Stop
    protein_cds = protein_full[:first_stop_idx]

    # Terminal Stop Codon bestimmen
    terminal_stop_nt_pos = atg_pos + first_stop_idx * 3
    terminal_stop_codon = seq[terminal_stop_nt_pos:terminal_stop_nt_pos + 3]

    # ECHTER CHECK: Gibt es '*' INNERHALB der CDS (vor dem ersten Stop)?
    # → Nach Definition des ersten Stop unmöglich, ABER:
    # Wir prüfen ob der erste ATG wirklich der richtige Start ist,
    # indem wir schauen ob das Protein plausibel lang ist (>50 AA)
    # und ob nach dem terminalen Stop noch weitere Stops in 3'UTR kommen (normal).
    internal_stops = [i for i, aa in enumerate(protein_full[:first_stop_idx]) if aa == '*']

    # Zusätzlich: prüfe ob Stop SEHR früh kommt (< 50 AA → wahrscheinlich falsches ATG)
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
        print(f"ERROR: MANE-Datei nicht gefunden:\n  {MANE_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Alle relevanten Einträge aus der MANE-Datei lesen ──
    print(f"Lese {MANE_FILE.name} ...")
    gene_records: dict[str, list] = {g: [] for g in GENES}

    for record in SeqIO.parse(MANE_FILE, "fasta"):
        for gene in GENES:
            if gene_matches(record.description, gene):
                gene_records[gene].append(record)

    # ── Analyse ──
    results = {}
    clean_genes = []
    problem_genes = []
    not_found_genes = []

    print(f"\n{'Gen':<12} {'Accession':<16} {'ATG':<8} {'Protein AA':<12} {'Term.Stop':<12} {'Status'}")
    print("─" * 72)

    for gene in GENES:
        records = gene_records[gene]

        if not records:
            not_found_genes.append(gene)
            results[gene] = {"status": "NOT_FOUND"}
            print(f"{gene:<12} {'–':<16} {'–':<8} {'–':<12} {'–':<12} ❓ nicht gefunden")
            continue

        # Wenn mehrere Treffer: ersten nehmen (MANE Select bevorzugt NM_)
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
            # FASTA für AlphaFold speichern
            fasta_out = OUTPUT_DIR / f"{gene}_alphafold.fasta"
            fasta_out.write_text(
                f">{gene} | {record.id} | AlphaFold-ready | {result['protein_length']} AA\n"
                f"{result['protein']}\n"
            )
        else:
            problem_genes.append(gene)

    # ── Zusammenfassung ──
    print("\n" + "═" * 72)
    print(f"GESAMT:          {len(GENES)} Gene")
    print(f"✅ Clean:        {len(clean_genes)}")
    print(f"❌ Probleme:     {len(problem_genes)}")
    print(f"❓ Nicht gefunden: {len(not_found_genes)}")

    if problem_genes:
        print(f"\nProblem-Gene: {', '.join(problem_genes)}")

    if not_found_genes:
        print(f"Nicht gefunden: {', '.join(not_found_genes)}")

    if clean_genes:
        print(f"\n✅ AlphaFold FASTA-Dateien gespeichert in:\n   {OUTPUT_DIR}")

    # ── JSON speichern (ohne Protein-Sequenzen für Übersicht) ──
    json_out_data = {
        g: {k: v for k, v in d.items() if k != "protein"}
        for g, d in results.items()
    }
    JSON_OUT.write_text(json.dumps(json_out_data, indent=2))
    print(f"\nJSON-Report: {JSON_OUT}")


if __name__ == "__main__":
    main()