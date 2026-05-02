#!/usr/bin/env python3
"""
Filter GENCODE pc_transcripts.fa for your genes → one .fasta per transcript isoform.

Filters applied (in order):
  1. Must have 5'UTR ≥ MIN_UTR5_NT nt   ← NEU
  2. Must have 3'UTR ≥ MIN_UTR3_NT nt   ← NEU
  3. Must be CDS-clean (no internal stop codons)
  4. Must meet minimum protein length (MIN_PROTEIN_AA)

Canonical isoform selection (among passing isoforms):
  1. Highest TPM wins
  2. Longest CDS as tiebreaker
"""

import argparse
import re
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import json
import subprocess
import pandas as pd
from typing import Dict, List, Tuple

BASE_DIR = Path("/Users/jonathanopitz/Desktop/Master")

GENES = {
    g.upper() for g in [
        "GAPDH", "LDHA", "LDHB", "ENO1", "PKM", "ALDOA", "PGK1", "IDH1", "IDH2", "MDH2",
        "G6PD", "PFKL", "PFKM", "ACO2", "CS", "SDHA", "FH", "GOT2", "ACTB", "ACTG1",
        "TUBB", "TUBA1B", "TUBA1A", "VIM", "LMNA", "LMNB1", "EEF1A1", "EEF2", "EIF4A1",
        "EIF3A", "EIF2S1", "EEF1G", "HSP90AA1", "HSP90AB1", "HSPA8", "HSPA5", "HSPD1",
        "CCT2", "CCT3", "CCT4", "DNAJA1", "DNAJB1", "MAPK1", "MAPK3", "AKT1", "AKT2",
        "PRKACA", "GSK3B", "SRC", "EGFR", "PIK3CA", "HNRNPA1", "HNRNPC", "HNRNPK",
        "RFC1", "TOP1", "PARP1", "JUN", "FOS", "MYC", "TP53", "STAT3", "NFKB1",
        "CCNB1", "CCNE1", "CDC20", "PLK1", "AURKB", "CASP7", "CTSB", "CTSD", "PSMD1",
        "PSMD2", "CAT", "TXNRD1", "VCP", "NSF", "ALB", "SERPINA1", "F9", "INSR",
        "NCL", "RPLP0", "BRCA1", "MTOR", "JAK2", "INS", "EPO", "IFNB1", "IL2", "GH1", "F8", "PROC",
    "HBB", "APOA1", "TTR", "VEGFA", "TGFB1", "FGA"
    ]
}

STOP_CODONS   = {"TAA", "TAG", "TGA"}
MIN_PROTEIN_AA = 100
MIN_UTR5_NT    = 20    # 5'UTR must be ≥ 20 nt (Kozak + cap context)
MIN_UTR3_NT    = 50    # 3'UTR must be ≥ 50 nt (poly-A signal + stability elements)


# ─── UTR presence check (NEW) ─────────────────────────────────────────────────

def check_utr_presence(header: str, seq: str) -> dict:
    """
    Checks that both UTRs are present and long enough.
    Required for mRNA vaccine design — UTRs carry Kozak, poly-A signal, etc.

    Priority:
      1. Read UTR5:/UTR3: from GENCODE header if available
      2. Derive from CDS: coordinates if UTR annotations missing
      3. Reject if neither available
    """
    seq_len    = len(seq)
    utr5_match = re.search(r'UTR5:(\d+)-(\d+)', header)
    utr3_match = re.search(r'UTR3:(\d+)-(\d+)', header)
    cds_match  = re.search(r'CDS:(\d+)-(\d+)',  header)

    # ── Case 1: explicit UTR annotations in header ────────────────────────────
    if utr5_match and utr3_match:
        utr5_len = int(utr5_match.group(2)) - int(utr5_match.group(1)) + 1
        utr3_len = int(utr3_match.group(2)) - int(utr3_match.group(1)) + 1

        if utr5_len < MIN_UTR5_NT:
            return {"ok": False,
                    "reason": f"5'UTR zu kurz ({utr5_len} nt < {MIN_UTR5_NT})"}
        if utr3_len < MIN_UTR3_NT:
            return {"ok": False,
                    "reason": f"3'UTR zu kurz ({utr3_len} nt < {MIN_UTR3_NT})"}
        return {"ok": True, "utr5_len": utr5_len, "utr3_len": utr3_len}

    # ── Case 2: only CDS annotation — derive UTR lengths from position ────────
    if cds_match:
        cds_start = int(cds_match.group(1)) - 1   # 0-based
        cds_end   = int(cds_match.group(2))        # exclusive
        utr5_len  = cds_start
        utr3_len  = seq_len - cds_end

        if utr5_len < MIN_UTR5_NT:
            return {"ok": False,
                    "reason": (f"Kein/zu kurzes 5'UTR ({utr5_len} nt) — "
                               f"CDS beginnt bei Position {cds_start + 1}")}
        if utr3_len < MIN_UTR3_NT:
            return {"ok": False,
                    "reason": (f"Kein/zu kurzes 3'UTR ({utr3_len} nt) — "
                               f"CDS endet bei Position {cds_end}")}
        return {"ok": True, "utr5_len": utr5_len, "utr3_len": utr3_len}

    # ── Case 3: no header info — reject ──────────────────────────────────────
    return {"ok": False,
            "reason": "Kein CDS/UTR-Header — UTR-Präsenz nicht bestimmbar"}


# ─── CDS length from GENCODE header ──────────────────────────────────────────

def get_cds_protein_length_from_header(header: str, seq: str) -> int:
    m = re.search(r'CDS:(\d+)-(\d+)', header)
    if m:
        cds_start = int(m.group(1)) - 1
        cds_end   = int(m.group(2))
        cds_seq   = seq[cds_start:cds_end]
        cds_trim  = cds_seq[:len(cds_seq) // 3 * 3]
        protein   = str(Seq(cds_trim).translate()).rstrip('*')
        return len(protein)

    seq_up  = seq.upper()
    atg_pos = seq_up.find("ATG")
    if atg_pos == -1:
        return 0
    from_atg = seq_up[atg_pos:]
    from_atg = from_atg[:len(from_atg) // 3 * 3]
    prot     = str(Seq(from_atg).translate())
    stop_idx = prot.find('*')
    return stop_idx if stop_idx != -1 else len(prot)


# ─── CDS Stop-Codon Check ─────────────────────────────────────────────────────

def check_cds_for_stops(header: str, seq: str) -> dict:
    seq_up = seq.upper()

    m = re.search(r'CDS:(\d+)-(\d+)', header)
    if m:
        cds_start = int(m.group(1)) - 1
        cds_end   = int(m.group(2))
        cds_seq   = seq_up[cds_start:cds_end]
        source    = "header"
    else:
        atg_pos = seq_up.find("ATG")
        if atg_pos == -1:
            return {"clean": False, "reason": "Kein ATG gefunden",
                    "protein_length": 0, "internal_stops": [], "has_terminal": False}
        cds_start = atg_pos
        from_atg  = seq_up[atg_pos:]
        cds_trim  = from_atg[:len(from_atg) // 3 * 3]
        prot_tmp  = str(Seq(cds_trim).translate())
        stop_idx  = prot_tmp.find('*')
        cds_end   = atg_pos + (stop_idx + 1) * 3 if stop_idx != -1 else atg_pos + len(cds_trim)
        cds_seq   = seq_up[cds_start:cds_end]
        source    = "ATG fallback"

    cds_trim  = cds_seq[:len(cds_seq) // 3 * 3]
    prot_full = str(Seq(cds_trim).translate())
    first_stop = prot_full.find('*')

    if first_stop == -1:
        return {"clean": False, "reason": "Kein terminaler Stop gefunden",
                "protein_length": len(prot_full), "internal_stops": [],
                "has_terminal": False, "cds_source": source}

    protein_cds    = prot_full[:first_stop]
    internal_stops = [i + 1 for i, aa in enumerate(protein_cds) if aa == '*']
    term_codon     = cds_trim[first_stop * 3: first_stop * 3 + 3]
    has_terminal   = term_codon in STOP_CODONS
    prot_len       = len(protein_cds)

    if prot_len < MIN_PROTEIN_AA:
        return {"clean": False,
                "reason": f"Protein zu kurz ({prot_len} AA < {MIN_PROTEIN_AA})",
                "protein_length": prot_len, "internal_stops": [],
                "has_terminal": has_terminal, "cds_source": source}

    if internal_stops:
        return {"clean": False,
                "reason": f"Interne Stop-Codons bei AA-Position(en): {internal_stops}",
                "protein_length": prot_len, "internal_stops": internal_stops,
                "has_terminal": has_terminal, "cds_source": source}

    if not has_terminal:
        return {"clean": False,
                "reason": f"Kein valider terminaler Stop (gefunden: '{term_codon}')",
                "protein_length": prot_len, "internal_stops": [],
                "has_terminal": False, "cds_source": source}

    return {"clean": True, "reason": "OK", "protein_length": prot_len,
            "internal_stops": [], "has_terminal": True, "cds_source": source}


# ─── GENCODE Filtering ────────────────────────────────────────────────────────

def filter_gencode(gencode_path: Path, out_dir: Path) -> Tuple[Dict, Dict]:
    gene_to_clean:    Dict[str, List[dict]] = {g: [] for g in GENES}
    gene_to_rejected: Dict[str, List[dict]] = {g: [] for g in GENES}

    total_parsed = matched = rejected = 0
    utr_rejected = 0   # counter for UTR-specific rejections

    print(f"Parsing {gencode_path.name} ...")
    print(f"UTR filter: 5'UTR ≥ {MIN_UTR5_NT} nt, 3'UTR ≥ {MIN_UTR3_NT} nt\n")

    for record in SeqIO.parse(gencode_path, "fasta"):
        total_parsed += 1
        desc = record.description.upper()

        for gene in GENES:
            if not re.search(rf'\b{gene}\b|\b{gene}-\d{{3}}\b', desc):
                continue

            matched += 1
            seq    = str(record.seq)
            header = record.description

            # ── Step 1: UTR check (NEW) ───────────────────────────────────────
            utr_check = check_utr_presence(header, seq)
            if not utr_check["ok"]:
                rejected += 1
                utr_rejected += 1
                gene_to_rejected[gene].append({
                    "trans_id":       record.id,
                    "header":         header,
                    "seq":            seq,
                    "length":         len(seq),
                    "protein_length": 0,
                    "cds_check":      {
                        "clean":  False,
                        "reason": f"[UTR] {utr_check['reason']}",
                    },
                })
                break

            # ── Step 2: CDS quality check (existing logic) ────────────────────
            cds_check = check_cds_for_stops(header, seq)
            prot_len  = get_cds_protein_length_from_header(header, seq)

            entry = {
                "trans_id":       record.id,
                "header":         header,
                "seq":            seq,
                "length":         len(seq),
                "protein_length": prot_len,
                "utr5_len":       utr_check.get("utr5_len", 0),
                "utr3_len":       utr_check.get("utr3_len", 0),
                "cds_check":      cds_check,
            }

            if cds_check["clean"]:
                gene_to_clean[gene].append(entry)
            else:
                rejected += 1
                gene_to_rejected[gene].append(entry)
            break

        if total_parsed % 10000 == 0:
            print(f"  Parsed {total_parsed:,} ...")

    print(f"\nDone.")
    print(f"  Parsed   : {total_parsed:,}")
    print(f"  Matched  : {matched:,}")
    print(f"  Rejected : {rejected:,}  (davon {utr_rejected} wegen fehlender UTRs)\n")

    # Write clean FASTAs
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for gene, transcripts in gene_to_clean.items():
        if not transcripts:
            continue
        for t in transcripts:
            clean_id = t["trans_id"].split('.')[0]
            filepath = out_dir / f"{gene}_{clean_id}.fasta"
            with open(filepath, 'w') as f:
                f.write(f">{t['header']}\n")
                seq = t["seq"]
                for j in range(0, len(seq), 60):
                    f.write(seq[j:j+60] + '\n')
            written += 1

    print(f"Wrote {written} clean transcript FASTAs to {out_dir}\n")
    return gene_to_clean, gene_to_rejected


# ─── Kallisto ─────────────────────────────────────────────────────────────────

def run_kallisto(gencode_path: Path, fastq_path: Path, kallisto_out: Path) -> Path:
    index_path = kallisto_out / "gencode.idx"
    kallisto_out.mkdir(parents=True, exist_ok=True)

    if not index_path.is_file():
        print("Building Kallisto index ...")
        subprocess.run(["kallisto", "index", "-i", str(index_path),
                        str(gencode_path)], check=True)

    print("Running Kallisto quant ...")
    subprocess.run([
        "kallisto", "quant",
        "-i", str(index_path),
        "-o", str(kallisto_out),
        "-b", "100",
        "--single", "-l", "200", "-s", "30",
        str(fastq_path)
    ], check=True)

    abundance_file = kallisto_out / "abundance.tsv"
    if not abundance_file.is_file():
        raise FileNotFoundError(f"Kallisto output not found: {abundance_file}")
    return abundance_file


# ─── Isoform Selection ────────────────────────────────────────────────────────

def select_best_isoforms(
    kallisto_file: Path,
    gene_to_clean: Dict,
    gene_to_rejected: Dict,
) -> Dict[str, dict]:
    df = pd.read_csv(kallisto_file, sep='\t')
    results = {}

    print(f"\n{'Gen':<12} {'Status':<10} {'Best Isoform':<26} "
          f"{'Prot AA':>8} {'5UTR':>6} {'3UTR':>6} {'TPM':>10}  {'%TPM':>6}")
    print("─" * 84)

    no_clean_isoform = []
    not_in_kallisto  = []

    for gene in sorted(GENES):
        clean_list    = gene_to_clean.get(gene, [])
        rejected_list = gene_to_rejected.get(gene, [])

        if not clean_list and not rejected_list:
            print(f"{gene:<12} ❓ NOT FOUND")
            results[gene] = {"status": "not_found"}
            continue

        if not clean_list:
            reasons = list({t['cds_check']['reason'] for t in rejected_list})
            print(f"{gene:<12} ❌ NO CLEAN   "
                  f"(all {len(rejected_list)} isoforms rejected)")
            for t in rejected_list[:3]:   # show max 3
                print(f"             ✗ {t['trans_id'].split('.')[0]:<22} "
                      f"– {t['cds_check']['reason']}")
            results[gene] = {"status": "no_clean_isoform",
                             "rejected_count": len(rejected_list),
                             "rejection_reasons": reasons}
            no_clean_isoform.append(gene)
            continue

        clean_by_enst = {t["trans_id"].split('.')[0]: t for t in clean_list}

        def matches_clean(target_id: str):
            for p in target_id.split('|'):
                enst_base = p.split('.')[0]
                if enst_base in clean_by_enst:
                    return enst_base
            return None

        df['_enst_match'] = df['target_id'].apply(matches_clean)
        gene_df = df[df['_enst_match'].notna()].copy()

        if gene_df.empty:
            print(f"{gene:<12} ⚠ NO TPM     "
                  f"({len(clean_list)} clean isoforms, not in Kallisto)")
            results[gene] = {"status": "no_tpm_data", "clean_count": len(clean_list)}
            not_in_kallisto.append(gene)
            df.drop(columns=['_enst_match'], inplace=True)
            continue

        gene_df = gene_df.copy()
        gene_df['_prot_len'] = gene_df['_enst_match'].apply(
            lambda e: clean_by_enst[e]['protein_length'] if e in clean_by_enst else 0
        )

        best_row       = gene_df.sort_values(['tpm', '_prot_len'],
                                             ascending=[False, False]).iloc[0]
        best_enst_base = best_row['_enst_match']
        best_enst_full = next(
            (p for p in best_row['target_id'].split('|') if p.startswith('ENST')),
            best_enst_base
        )
        best_tpm  = float(best_row['tpm'])
        total_tpm = float(gene_df['tpm'].sum())
        pct       = (best_tpm / total_tpm * 100) if total_tpm > 0 else 0.0
        prot_len  = clean_by_enst[best_enst_base]['protein_length']
        utr5_len  = clean_by_enst[best_enst_base].get('utr5_len', '?')
        utr3_len  = clean_by_enst[best_enst_base].get('utr3_len', '?')

        print(f"{gene:<12} ✅ OK         {best_enst_full:<26} "
              f"{str(prot_len):>8} AA "
              f"{str(utr5_len):>5}nt "
              f"{str(utr3_len):>5}nt "
              f"{best_tpm:>10.2f}  {pct:>5.1f}%")

        results[gene] = {
            "status":                   "ok",
            "best_isoform":             best_enst_base,
            "best_isoform_full":        best_enst_full,
            "tpm":                      best_tpm,
            "pct_of_gene_tpm":          round(pct, 2),
            "protein_length":           prot_len,
            "utr5_length":              utr5_len,
            "utr3_length":              utr3_len,
            "clean_isoforms_available": len(clean_list),
            "rejected_isoforms":        len(rejected_list),
        }

        df.drop(columns=['_enst_match'], inplace=True)

    ok_count = sum(1 for v in results.values() if v.get("status") == "ok")
    print("\n" + "═" * 84)
    print(f"✅ OK               : {ok_count}")
    print(f"❌ No clean isoform : {len(no_clean_isoform)}")
    print(f"⚠  No TPM data      : {len(not_in_kallisto)}")

    if no_clean_isoform:
        print(f"\n❌ {', '.join(no_clean_isoform)}")
    if not_in_kallisto:
        print(f"\n⚠  {', '.join(not_in_kallisto)}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gencode_fa',   required=True)
    parser.add_argument('--fastq',        required=True)
    parser.add_argument('--output_dir',   default=str(BASE_DIR / "data" / "genes"))
    parser.add_argument('--kallisto_out', default=str(BASE_DIR / "kallisto_quant"))
    args = parser.parse_args()

    gencode_path = Path(args.gencode_fa).resolve()
    fastq_path   = Path(args.fastq).resolve()
    out_dir      = Path(args.output_dir).resolve()
    kallisto_out = Path(args.kallisto_out).resolve()

    if not gencode_path.is_file():
        print(f"ERROR: GENCODE FASTA not found: {gencode_path}")
        return 1

    gene_to_clean, gene_to_rejected = filter_gencode(gencode_path, out_dir)
    abundance_file = run_kallisto(gencode_path, fastq_path, kallisto_out)
    results = select_best_isoforms(abundance_file, gene_to_clean, gene_to_rejected)

    out_json = BASE_DIR / "isoform_selection.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == '__main__':
    main()