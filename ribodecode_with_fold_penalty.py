import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    import RNA
except Exception as e:
    import traceback
    traceback.print_exc()
import json
import argparse
import random
import shutil
import subprocess
import time
import multiprocessing as mp
from contextlib import suppress
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Force 'fork' start method for multiprocessing (macOS safety).
# ═══════════════════════════════════════════════════════════════════════════════
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

from RiboDecode.dataset import (
    Dataset_rna, Dataset_rna_mfe, dict_vocab_inv, my_vocab,
    read_data, read_lines, process_line, dict_codon_group,
)
from RiboDecode.models import mfe_conv_sim
from RiboDecode.score_model.inference import InferenceModel_conditon_spec as score_old
from RiboDecode.check import Check

# ═══════════════════════════════════════════════════════════════════════════════
# FOLD PENALTY MODULE — now with FULL graph (seq + protein 3D + RNA) matching
# the training script exactly.
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
from torch.nn import LayerNorm
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_MASTER   = Path("/Users/jonathanopitz/Desktop/Master")
WT_DATA_DIR   = BASE_MASTER / "data/ribo_counts"
# PDBs live under data/alphafold_results/<GENE>/ with AlphaFold naming:
# *_unrelaxed_alphafold2_ptm_model_1_seed_000.pdb (etc.)
# Also allow legacy data/pdb/ as fallback.
PDB_DIRS      = [
    BASE_MASTER / "data/alphafold_results",
    BASE_MASTER / "data/pdb",
]
OPTIMIZED_DIR = BASE_MASTER / "data/optimized"
GNN_MODEL_PATH = BASE_MASTER / "data/results/gnn_v10_seq_prot_rna.pt"
if not GNN_MODEL_PATH.exists():
    GNN_MODEL_PATH = BASE_MASTER / "data/results/gnn_v4_seq_prot_rna.pt"

# ─── GNN constants (must match training) ─────────────────────────────────────
FOLD_GAMMA         = 0.3
GNN_UPDATE_EVERY   = 1
CODON_EMBED_DIM    = 8
N_CODON_VOCAB      = 65
N_DENSE_FEATURES   = 20          # 18 struct + ss_class + is_denovo
EDGE_DIM           = 6
CA_THRESHOLD       = 8.0         # Å, protein contact cutoff
MIN_SEQ_SEP        = 6           # min codon separation for prot contacts

STRUCT_COLS = [
    'plddt', 'contact_density', 'domain_boundary',
    'domain_boundary_interpro', 'domain_boundary_pae',
    'ss_H', 'ss_E', 'ss_C',
    'rna_local_paired_prob',
    'rna_unpaired_1nt', 'rna_unpaired_3nt', 'rna_unpaired_5nt',
    'rna_opening_energy_1nt', 'rna_opening_energy_3nt', 'rna_opening_energy_5nt',
    'rna_paired_prob_window5cod', 'rna_paired_prob_window10cod',
    'rna_struct_change',
]

# ─── Codon vocabulary ─────────────────────────────────────────────────────────
CODONS_LIST  = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS_LIST)}

RD_TO_GNN = torch.zeros(65, dtype=torch.long)
for codon_str, rd_idx in my_vocab.items():
    dna = codon_str.replace('U', 'T')
    RD_TO_GNN[rd_idx] = CODON_TO_IDX.get(dna, 64)

# ─── TAI vector aligned to RiboDecode vocab ───────────────────────────────────
CODON_TAI_RAW = {
    'TTT':0.42,'TTC':1.00,'TTA':0.08,'TTG':0.42,
    'CTT':0.42,'CTC':0.58,'CTA':0.08,'CTG':1.00,
    'ATT':0.58,'ATC':1.00,'ATA':0.08,'ATG':1.00,
    'GTT':0.42,'GTC':0.58,'GTA':0.08,'GTG':1.00,
    'TCT':0.58,'TCC':0.75,'TCA':0.25,'TCG':0.17,
    'CCT':0.58,'CCC':0.75,'CCA':0.42,'CCG':0.17,
    'ACT':0.58,'ACC':1.00,'ACA':0.42,'ACG':0.17,
    'GCT':0.75,'GCC':1.00,'GCA':0.42,'GCG':0.17,
    'TAT':0.42,'TAC':1.00,'TAA':0.00,'TAG':0.00,
    'CAT':0.42,'CAC':1.00,'CAA':0.42,'CAG':1.00,
    'AAT':0.42,'AAC':1.00,'AAA':0.42,'AAG':1.00,
    'GAT':0.42,'GAC':1.00,'GAA':0.42,'GAG':1.00,
    'TGT':0.42,'TGC':1.00,'TGA':0.00,'TGG':1.00,
    'CGT':0.42,'CGC':0.75,'CGA':0.08,'CGG':0.17,
    'AGT':0.25,'AGC':0.75,'AGA':0.42,'AGG':0.25,
    'GGT':0.42,'GGC':1.00,'GGA':0.25,'GGG':0.17,
}
TAI_VECTOR_RD = torch.zeros(65)
for codon_str, rd_idx in my_vocab.items():
    TAI_VECTOR_RD[rd_idx] = CODON_TAI_RAW.get(codon_str.replace('U','T'), 0.5)


# ─── GNN model (identical to train_gnn_v5.py) ────────────────────────────────
class FoldingGATv2(torch.nn.Module):
    def __init__(self, dense_dim=N_DENSE_FEATURES, hid=32, heads=2, edge_dim=EDGE_DIM):
        super().__init__()
        self.codon_emb = torch.nn.Embedding(N_CODON_VOCAB, CODON_EMBED_DIM)
        in_dim = dense_dim + CODON_EMBED_DIM
        self.proj  = torch.nn.Linear(in_dim, hid)
        self.norm0 = LayerNorm(hid)
        self.conv1 = GATv2Conv(hid, hid, heads=heads, edge_dim=edge_dim, dropout=0.15, add_self_loops=False)
        self.norm1 = LayerNorm(hid*heads); self.drop1 = torch.nn.Dropout(0.3)
        self.res1  = torch.nn.Linear(hid, hid*heads, bias=False)
        self.conv2 = GATv2Conv(hid*heads, hid, heads=heads, edge_dim=edge_dim, dropout=0.15, add_self_loops=False)
        self.norm2 = LayerNorm(hid*heads); self.drop2 = torch.nn.Dropout(0.3)
        self.res2  = torch.nn.Linear(hid*heads, hid*heads, bias=False)
        self.conv3 = GATv2Conv(hid*heads, hid, heads=heads, edge_dim=edge_dim, dropout=0.15, add_self_loops=False)
        self.norm3 = LayerNorm(hid*heads); self.drop3 = torch.nn.Dropout(0.3)
        self.res3  = torch.nn.Linear(hid*heads, hid*heads, bias=False)
        self.head  = torch.nn.Sequential(
            torch.nn.Linear(hid*heads, 32), torch.nn.ELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1), torch.nn.Sigmoid()
        )

    def forward(self, data):
        ei, ea = data.edge_index, data.edge_attr
        x = torch.cat([data.x, self.codon_emb(data.codon_idx)], dim=-1)
        x = F.elu(self.norm0(self.proj(x))); r = x
        x = self.conv1(x,ei,ea); x = self.drop1(F.elu(self.norm1(x))); x = x+self.res1(r); r = x
        x = self.conv2(x,ei,ea); x = self.drop2(F.elu(self.norm2(x))); x = x+self.res2(r); r = x
        x = self.conv3(x,ei,ea); x = self.drop3(F.elu(self.norm3(x))); x = x+self.res3(r)
        return self.head(x).squeeze(-1)


def load_gnn(path, device):
    try:
        m = FoldingGATv2().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        print(f"[GNN] Loaded: {path.name}", flush=True)
        return m
    except Exception as e:
        print(f"[GNN] Load failed: {e}", flush=True)
        return None


def load_protein_features(gene: str, device):
    """Load AlphaFold-based structural features from WT *_with_folddemand.csv."""
    gene = gene.upper()
    candidates = sorted(
        p for p in WT_DATA_DIR.glob("*_with_folddemand.csv")
        if gene in p.name.upper()
    )
    if not candidates:
        print(f"[GNN] No WT CSV for {gene}, using zeros", flush=True)
        return None, None

    df = pd.read_csv(candidates[0])
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    n = len(cds)

    feats = []
    for col in STRUCT_COLS:
        if col in cds.columns:
            vals = pd.to_numeric(cds[col], errors='coerce').fillna(0.0).values
        else:
            vals = np.zeros(n)
        feats.append(vals)

    ss_enc = cds.get('rna_local_ss_class', pd.Series('weak', index=cds.index)) \
               .map({'stem':1.0,'weak':0.5,'open_loop':0.0}).fillna(0.0).values
    feats.append(ss_enc)
    feats.append(np.zeros(n))  # is_denovo = 0 for WT

    dense_x = torch.tensor(
        np.nan_to_num(np.stack(feats, axis=1).astype(np.float32), nan=0.0),
        dtype=torch.float32, device=device
    )
    print(f"[GNN] Protein features: {n} codons, shape={dense_x.shape}", flush=True)
    return dense_x, n


def find_cds_position(full_seq: str, expected_cds_len_nt: int):
    """Find CDS start: ATG ... in-frame STOP spanning exactly expected_cds_len_nt."""
    for start in range(len(full_seq) - expected_cds_len_nt + 1):
        if full_seq[start:start+3] != 'ATG':
            continue
        end = start + expected_cds_len_nt
        if end > len(full_seq):
            continue
        if full_seq[end-3:end] in ('TAA', 'TAG', 'TGA'):
            return start
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE BUILDERS (match train_gnn_v5.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def build_seq_edges(n):
    """Sequential edges at distances 1, 3, 5. Bidirectional, edge_dim=6."""
    src, dst, feats = [], [], []
    for dist in [1, 3, 5]:
        for i in range(n - dist):
            j = i + dist
            db = {1:0.2, 3:0.6, 5:1.0}[dist]
            feat = [1, 0, 0, dist/5.0, dist/n, db]
            src.extend([i, j]); dst.extend([j, i])
            feats.extend([feat, feat])
    return src, dst, feats


def find_pdb(gene):
    """
    Locate AlphaFold PDB for gene across multiple directories.

    Search order:
      1. data/alphafold_results/<GENE>/*.pdb           (ColabFold output layout)
      2. data/alphafold_results/**/*<GENE>*.pdb        (any subdir matching gene)
      3. data/pdb/*<GENE>*.pdb                         (legacy flat layout)

    Preference among matches:
      - model_1 > model_2 > ... (AlphaFold ranks model_1 as best)
      - otherwise largest file (usually highest-resolution / most complete)
    """
    gene_u = gene.upper()
    gene_l = gene.lower()
    candidates = []

    for pdb_dir in PDB_DIRS:
        if not pdb_dir.is_dir():
            continue

        # 1. Exact gene-named subdirectory (ColabFold output style)
        for sub in [pdb_dir / gene_u, pdb_dir / gene_l, pdb_dir / gene]:
            if sub.is_dir():
                candidates.extend(sub.glob("*.pdb"))

        # 2. Any file/subdir matching gene name (case-insensitive)
        for pat in [f"*{gene_u}*.pdb", f"*{gene_l}*.pdb", f"*{gene_u}*/*.pdb", f"*{gene_l}*/*.pdb"]:
            candidates.extend(pdb_dir.glob(pat))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c); unique.append(c)

    if not unique:
        return None

    # Prefer model_1 (AlphaFold's top-ranked), then model_2, etc.
    def rank(p):
        name = p.name.lower()
        for i in range(1, 6):
            if f"model_{i}" in name:
                return (i, -p.stat().st_size)
        return (99, -p.stat().st_size)
    return min(unique, key=rank)


def parse_ca(pdb_path):
    """Parse Cα coordinates from PDB. Returns dict {residue_idx: xyz_ndarray}."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                r = int(line[22:26].strip()) - 1  # 1-based → 0-based
                if r not in coords:
                    coords[r] = np.array([
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ])
    return coords


def build_prot_edges(gene: str, n_codons: int):
    """
    Build protein 3D contact edges from AlphaFold PDB.
    Two codons are connected if their Cα atoms are within CA_THRESHOLD Å
    AND their sequence separation is ≥ MIN_SEQ_SEP.
    This matches train_gnn_v5.py exactly.
    """
    pdb_path = find_pdb(gene)
    if pdb_path is None:
        return [], [], []
    print(f"[GNN] Using PDB: {pdb_path.relative_to(BASE_MASTER)}", flush=True)

    coords = parse_ca(pdb_path)
    residues = sorted([r for r in coords if r < n_codons])
    if len(residues) < 2:
        return [], [], []

    src, dst, feats = [], [], []
    for i, ri in enumerate(residues):
        for j in range(i + 1, len(residues)):
            rj = residues[j]
            if abs(ri - rj) < MIN_SEQ_SEP:
                continue
            d = np.linalg.norm(coords[ri] - coords[rj])
            if d < CA_THRESHOLD:
                src.extend([ri, rj]); dst.extend([rj, ri])
                feat = [0, 1, 0, d / CA_THRESHOLD, abs(ri - rj) / n_codons, 0.0]
                feats.extend([feat, feat])

    return src, dst, feats


def build_rna_edges_from_sequence(cds_seq: str, n_codons: int,
                                   utr5: str = '', utr3: str = ''):
    """
    Fold FULL mRNA with ViennaRNA, extract base pairs, map to CDS codon indices.
    Only pairs fully inside CDS are kept (UTR-crossing pairs dropped).
    """
    full_seq = utr5 + cds_seq + utr3
    rna_seq  = full_seq.replace('T', 'U')
    try:
        db, _mfe = RNA.fold(rna_seq)
    except Exception as e:
        print(f"[GNN] RNA.fold failed: {e}", flush=True)
        return [], [], []

    stack, pairs = [], []
    for i, ch in enumerate(db):
        if ch in '({[': stack.append(i)
        elif ch in ')}]' and stack: pairs.append((stack.pop(), i))

    utr5_len   = len(utr5)
    cds_len_nt = len(cds_seq)

    src, dst, feats = [], [], []
    for i_nt, j_nt in pairs:
        i_cds = i_nt - utr5_len
        j_cds = j_nt - utr5_len
        if not (0 <= i_cds < cds_len_nt and 0 <= j_cds < cds_len_nt):
            continue
        ci, cj = i_cds // 3, j_cds // 3
        if ci == cj or ci >= n_codons or cj >= n_codons or abs(ci - cj) < 3:
            continue
        feat = [0, 0, 1, 0.0, abs(ci - cj) / n_codons, 0.0]
        src.extend([ci, cj]); dst.extend([cj, ci])
        feats.extend([feat, feat])

    return src, dst, feats


def build_graph_for_sequence(dense_x, codon_ids_rd, cds_seq_dna, n_codons,
                              device, utr5='', utr3='',
                              prot_edges_cached=None):
    """
    Build full GNN graph: seq + prot (cached, FIXED) + RNA (recomputed every call).
    prot_edges_cached: (src, dst, feats) tuple to avoid re-parsing PDB each step.
    """
    s_src, s_dst, s_feats = build_seq_edges(n_codons)

    # Protein edges: FIXED across optimization → cache and reuse
    if prot_edges_cached is not None:
        p_src, p_dst, p_feats = prot_edges_cached
    else:
        p_src, p_dst, p_feats = [], [], []

    # RNA edges: change with each new codon sequence → recompute
    r_src, r_dst, r_feats = build_rna_edges_from_sequence(
        cds_seq_dna, n_codons, utr5=utr5, utr3=utr3
    )

    all_src   = s_src + p_src + r_src
    all_dst   = s_dst + p_dst + r_dst
    all_feats = s_feats + p_feats + r_feats

    if not all_src:
        all_src = list(range(n_codons)); all_dst = list(range(n_codons))
        all_feats = [[0,0,0,0,0,0]] * n_codons

    gnn_codon_idx = RD_TO_GNN.to(device)[codon_ids_rd]

    return Data(
        x=dense_x,
        codon_idx=gnn_codon_idx,
        edge_index=torch.tensor([all_src, all_dst], dtype=torch.long, device=device),
        edge_attr=torch.tensor(all_feats, dtype=torch.float32, device=device),
        num_nodes=n_codons,
    )


def ribodecode_seq_to_dna(seq_gen, n_codons, len_gene_ori):
    """Convert seq_gen (batch, 65, n_codons) → DNA string for first batch item."""
    codon_ids = torch.argmax(seq_gen[0], dim=0)
    dna = ""
    for rd_idx in codon_ids.cpu().numpy():
        codon_str = dict_vocab_inv.get(int(rd_idx), 'NNN')
        dna += codon_str.replace('U', 'T')
    return dna[:len_gene_ori], codon_ids


def compute_fold_penalty(seq_gen_new, fd_pred, tai_vector, device):
    """Differentiable fold_penalty: mean( fd_pred * E[TAI] )."""
    tai = tai_vector.to(device)
    fd  = fd_pred.to(device)
    expected_tai = torch.matmul(seq_gen_new, tai)
    return torch.mean(fd.unsqueeze(0) * expected_tai)


def r2_score_func(y_true, y_pred):
    a = np.square(y_pred - y_true); b = np.sum(a)
    c = np.mean(y_true); d = np.square(y_true - c); e = np.sum(d)
    return 1 - b / (e + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# MFE helpers (ViennaRNA)
# ═══════════════════════════════════════════════════════════════════════════════

def get_eng_for_get_mfe(seq, len_sub):
    if len(seq) <= len_sub:
        _, mfe = RNA.fold(seq)
    else:
        mfe = sum(RNA.fold(seq[i:i+len_sub])[1] for i in range(0, len(seq), len_sub))
    return [seq, mfe]


def get_mfe(list_data, len_sub=4000):
    seqs = [d if isinstance(d, str) else d[0] for d in list_data]
    with Pool(cpu_count() - 1) as p:
        return p.starmap(get_eng_for_get_mfe, [(s, len_sub) for s in seqs])


def get_eng_for_get_mfe_sim(seq):
    _, mfe = RNA.fold(seq if isinstance(seq, str) else seq[0])
    return mfe


def get_mfe_sim(list_data):
    seqs = [d if isinstance(d, str) else d[0] for d in list_data]
    with Pool(cpu_count() - 1) as p:
        return p.map(get_eng_for_get_mfe_sim, seqs)


# ═══════════════════════════════════════════════════════════════════════════════
# RiboDecode generator
# ═══════════════════════════════════════════════════════════════════════════════

class seq_codon_gen(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.seqs = nn.Parameter(
            torch.randn(model_config["optim_batchsize"], model_config["max_len"] // 3, 65),
            requires_grad=True,
        )

    def generate(self, codon_mask):
        self.seqs_codon = torch.exp(self.seqs) * codon_mask
        self.seqs_codon = self.seqs_codon / torch.sum(self.seqs_codon, dim=-1, keepdim=True)
        return self.seqs_codon.permute(0, 2, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# gen_train / mfe_train (needed when --mfe_weight > 0)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_train(args, idx_optim, model_config, data_gen_optim, device):
    len_gene_ori = model_config["len_gene_ori"]
    rna_condition = model_config["RNA_condition"]
    sampler_type  = model_config["sampler_type"]
    save_log_dir  = (
        f"./results_{model_config['initial_seq']}/{rna_condition}_optim_mfe_"
        f"{sampler_type}/{idx_optim}/{args.save_dir}"
    )
    os.makedirs(save_log_dir, exist_ok=True)

    if idx_optim == 0:
        dataset_rna = Dataset_rna(
            model_config, "train", status="train",
            data_scale=model_config["num_seqs_train_gen"], seq=args.cds_seq,
        )
    else:
        dataset_rna = Dataset_rna(
            model_config, "optim", status="train",
            data_scale=model_config["num_seqs_train_gen"],
            dist=data_gen_optim["mask_optim_dist"][idx_optim - 1],
        )
    loader_train = DataLoader(dataset_rna, batch_size=model_config["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True, pin_memory=False)

    list_seqs = []
    for data in loader_train:
        list_seqs += torch.argmax(data["seq_con"].to(device).permute(0,2,1), dim=-1).detach().cpu().numpy().tolist()

    list_lines = []
    list_seqs_random = random.choices(list_seqs, k=50000)
    with open(save_log_dir + "/train_data_generate.txt", mode="w") as w:
        for temp_seq in list_seqs_random:
            seq_str = "".join(dict_vocab_inv.get(round(i,0), dict_vocab_inv[64]) for i in temp_seq)
            list_lines.append(seq_str[:len_gene_ori])
            w.write(seq_str[:len_gene_ori] + "\n")

    if idx_optim == 0:
        temp_list_data = list_lines[:model_config["num_seqs_select_mfe"]]
    else:
        sampler_type = model_config["sampler_type"]
        if sampler_type in ("dist-optim", "random-optim"):
            n = model_config["num_seqs_select_mfe"] // 4
            temp_list_data = data_gen_optim["seqs_optim_gen"][idx_optim-1][:n] + list_lines[:n]
        else:
            temp_list_data = list_lines[:model_config["num_seqs_select_mfe"] // 2]

    list_data_mfe = get_mfe(temp_list_data, len_sub=model_config["mfe_tool_sub_len"])
    list_data_gen = []
    with open(save_log_dir + "/train_data_generate_mfe.txt", mode="w") as w:
        for seq, mfe in list_data_mfe:
            list_data_gen.append([seq, mfe])
            w.write(seq + "," + str(mfe) + "\n")

    data_gen_optim["data_gen_mfe"][idx_optim] = sorted(list_data_gen, key=lambda x: x[-1])


def mfe_train(args, idx_optim, model_config, data_gen_optim, model_mfe, device):
    print("[mfe_train] optimizing MFE prediction model...", flush=True)
    rna_condition = model_config["RNA_condition"]
    sampler_type  = model_config["sampler_type"]
    save_log_dir  = (
        f"./results_{model_config['initial_seq']}/{rna_condition}_optim_mfe_"
        f"{sampler_type}/{idx_optim}/{args.save_dir}"
    )
    os.makedirs(save_log_dir, exist_ok=True)

    list_data_mfe = data_gen_optim["data_gen_mfe"][idx_optim]
    data_scale = model_config["num_seqs_train_mfe"] * 2 if idx_optim == 0 \
                 else model_config["num_seqs_train_mfe"] // 2
    dataset_mfe = Dataset_rna_mfe(model_config, list_data_mfe, data_scale=data_scale)
    loader_train = DataLoader(dataset_mfe, batch_size=model_config["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)

    lr = model_config["lr"] if idx_optim == 0 else model_config["lr"] * 0.1
    opt_mfe   = torch.optim.AdamW(model_mfe.parameters(), lr, weight_decay=5e-4)
    loss_fun  = nn.SmoothL1Loss()
    model_mfe.train()

    class FGM:
        def __init__(self, m): self.model=m; self.backup={}
        def attack(self, epsilon=1.0, emb_name="embedding"):
            for name, param in self.model.named_parameters():
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    with suppress(Exception):
                        norm = torch.norm(param.grad)
                        if norm != 0 and not torch.isnan(norm):
                            param.data.add_(epsilon * param.grad / norm)
        def restore(self, emb_name="embedding"):
            for name, param in self.model.named_parameters():
                if param.requires_grad and emb_name in name:
                    param.data = self.backup[name]
            self.backup = {}

    fgm = FGM(model_mfe)
    for data in loader_train:
        seq = data["seq"].to(device); mfe = data["mfe"].to(device)
        opt_mfe.zero_grad()
        out = model_mfe(seq)
        loss = loss_fun(out.view(-1), mfe)
        fgm.attack(epsilon=float(random.randint(100,1000)/1000.0), emb_name="encoder")
        loss += loss_fun(model_mfe(seq).view(-1), mfe)
        fgm.restore(emb_name="encoder")
        loss.backward(); opt_mfe.step()
    model_mfe.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# optim — main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def optim(args, idx_optim, model_config, data_gen_optim,
          model_mfe, z_model, gnn_model, protein_dense_x,
          UTR5, UTR3, weight_mfe, check, list_pcscg_W, W_K2BP,
          base_dir, device, prot_edges_cached):
    batch_size   = model_config["optim_batchsize"]
    len_gene_ori = model_config["len_gene_ori"]
    n_codons     = len_gene_ori // 3
    print(f"\n══════ epoch {idx_optim+1}/{model_config['num_optim']} ══════", flush=True)
    print(f"[optim] CDS length: {len_gene_ori} nt ({n_codons} codons)", flush=True)
    print(f"[optim] 5'UTR: {len(UTR5)} nt  |  3'UTR: {len(UTR3)} nt", flush=True)

    res_k, _ = read_lines([args.cds_seq.replace("U","T")], model_config["max_len"])
    list_res_ks = []
    for temp_k in res_k[0]:
        tmp = np.array([0]*65)
        for ck in dict_codon_group[temp_k]: tmp[ck] = 1
        list_res_ks.append(tmp)
    mask_gene_codon = torch.from_numpy(np.array([list_res_ks])).to(device).float()

    rna_condition     = model_config["RNA_condition"]
    rna_condition_neg = model_config["RNA_condition_negative"]
    sampler_type      = model_config["sampler_type"]
    result_dir = (
        f"./results_{model_config['initial_seq']}/{rna_condition}_optim_mfe_"
        f"{sampler_type}/{idx_optim}/{args.result_dir}"
    )
    os.makedirs(result_dir, exist_ok=True)
    data_gen_optim["optim_iteration_logs"][idx_optim] = result_dir

    print(f"[optim] Loading score_model...", flush=True)
    score_model = score_old(args.model_config_s, args.best_model)
    if model_config["using_custom_env"]:
        score_model.prepare(batch_size, rna_condition, base_dir+"/score_model/conditions/", args.csv)
    else:
        score_model.prepare(batch_size, rna_condition, base_dir+"/score_model/conditions/")

    print(f"[optim] Building dataset...", flush=True)
    dataset_rna = Dataset_rna(
        model_config,
        dataset_name="train" if idx_optim == 0 else "optim",
        status="train",
        data_scale=model_config["optim_num"],
        seq=args.cds_seq if idx_optim == 0 else None,
    )
    loader_valid = DataLoader(dataset_rna, batch_size=batch_size, shuffle=False,
                              num_workers=0, drop_last=False)
    n_steps = len(loader_valid)
    print(f"[optim] Ready: {n_steps} steps, batch_size={batch_size}", flush=True)

    opt_seq = torch.optim.AdamW(z_model.parameters(),
                                 lr=model_config["optim_lr"], weight_decay=1e-4)
    seq_gen_pad = torch.zeros(batch_size, 4500 - model_config["max_len"], 4).to(device)
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        opt_seq, milestones=[n_steps//2, n_steps*3//4, n_steps*7//8, n_steps*17//16]
    )
    order_windows_list = [[i/100, (i+2)/100] for i in range(0, 99, 2)]

    list_loss, list_rpf, list_seqs = [], [], []
    list_similarity, list_mfe, list_cscgs = [], [], []
    list_seq_gens, list_fold_penalty = [], []

    fd_pred_cache = torch.ones(n_codons, device=device) * 0.3

    model_mfe.eval()
    epoch_start = time.time()

    for step_i, data in enumerate(loader_valid):
        step_start = time.time()
        seq = data["seq_ori"].to(device)
        b   = seq.shape[0]

        seq_gen = z_model.generate(mask_gene_codon.clone())
        list_seq_gens.append(seq_gen.detach().cpu())

        gnn_time = 0.0
        if gnn_model is not None and step_i % GNN_UPDATE_EVERY == 0:
            gnn_start = time.time()
            with torch.no_grad():
                current_dna, codon_ids_rd = ribodecode_seq_to_dna(
                    seq_gen, n_codons, len_gene_ori
                )
                graph = build_graph_for_sequence(
                    protein_dense_x, codon_ids_rd, current_dna, n_codons,
                    device, utr5=UTR5, utr3=UTR3,
                    prot_edges_cached=prot_edges_cached,
                )
                fd_pred_cache = gnn_model(graph)
            gnn_time = time.time() - gnn_start

        temp_seq_gen = F.one_hot(torch.argmax(seq_gen, 1), 65).float()
        temp_seq_gen = temp_seq_gen.permute(0,2,1) - seq_gen.detach() + seq_gen
        temp_score_mfe = model_mfe(temp_seq_gen)
        loss_mfe = torch.mean(-float(model_config["mfe_norm_index"]) / temp_score_mfe)

        l = n_codons
        list_temp_cscgs = [
            torch.matmul(
                temp_seq_gen.permute(0,2,1)[:, int(idx[0]*l):int(idx[1]*l), :],
                list_pcscg_W[n].to(device)
            ).view(b,-1)
            for n, idx in enumerate(order_windows_list)
        ]
        temp_cscgs = torch.sum(torch.concatenate(list_temp_cscgs, dim=1), dim=1)
        mean_sim   = torch.mean(torch.cosine_similarity(seq, seq_gen), dim=1)

        seq_gen_label  = torch.argmax(seq_gen, 1)
        seq_gen_onehot = F.one_hot(seq_gen_label, 65).float()
        seq_gen_t      = seq_gen.permute(0,2,1)
        seq_gen_new    = seq_gen_onehot + seq_gen_t - seq_gen_t.detach()

        temp_codon = torch.sum(seq_gen_new * mask_gene_codon) / (batch_size * n_codons)
        loss_codon = 1 - temp_codon

        seq_gen_bp     = torch.matmul(seq_gen_new, W_K2BP.to(device).view(-1,12)).view(b,-1,4)[:,:4500,:]
        seq_gen_bp_pad = torch.cat([seq_gen_bp, seq_gen_pad], dim=1)

        score_target, score_neg = score_model.predict_seq_spec_single(
            seq_gen_bp_pad.permute(0,2,1), rna_condition, rna_condition_neg
        )
        rpf_target  = 0.2 * torch.expm1(score_target)
        loss_target = torch.mean(torch.sqrt(torch.pow(model_config["rpf_target"] - rpf_target, 2)) / 10.0)
        loss_rpf    = loss_target * 0.1 if model_config["rpf_target"] == 100 else loss_target

        loss_fold = compute_fold_penalty(seq_gen_new, fd_pred_cache, TAI_VECTOR_RD, device)
        list_fold_penalty.append(loss_fold.item())

        loss = (loss_codon
                + loss_mfe   * weight_mfe
                + loss_rpf   * (1 - weight_mfe)
                + loss_fold  * FOLD_GAMMA)

        loss.backward()
        opt_seq.step(); opt_seq.zero_grad(); lr_sch.step()

        list_loss += [loss.detach().cpu().item()] * batch_size
        list_rpf  += (torch.expm1(score_target.detach().cpu()) / 5).view(-1).numpy().tolist()
        list_similarity += mean_sim.detach().cpu().tolist()
        list_mfe  += temp_score_mfe.detach().cpu().view(-1).numpy().tolist()
        list_seqs += torch.argmax(seq_gen, dim=1).detach().cpu().numpy().tolist()
        list_cscgs += temp_cscgs.detach().cpu().numpy().tolist()

        step_time = time.time() - step_start
        print(
            f"  step {step_i+1:>3}/{n_steps}  "
            f"loss={loss.item():.3f}  "
            f"codon={loss_codon.item():.3f}  "
            f"rpf={loss_rpf.item():.3f}  "
            f"mfe={loss_mfe.item():.3f}  "
            f"fold={loss_fold.item():.4f}  "
            f"fd_pred[mean/max]={fd_pred_cache.mean().item():.2f}/{fd_pred_cache.max().item():.2f}  "
            f"t={step_time:.1f}s (gnn={gnn_time:.1f}s)",
            flush=True
        )

    epoch_time = time.time() - epoch_start
    fp_mean = np.mean(list_fold_penalty)
    print(f"[epoch {idx_optim+1}] done in {epoch_time:.0f}s  |  "
          f"fold_penalty_mean={fp_mean:.4f}  gamma={FOLD_GAMMA}  "
          f"fd_mean={fd_pred_cache.mean().item():.3f}  fd_max={fd_pred_cache.max().item():.3f}",
          flush=True)

    list_seq_rpf_mfe = []
    with open(result_dir + "/optim_results.txt", mode="w") as w:
        for id_line, temp_seq in enumerate(list_seqs[:model_config["optim_num"]]):
            seq_str = "".join(dict_vocab_inv.get(round(i,0), dict_vocab_inv[64])
                              for i in temp_seq)[:len_gene_ori]
            rpf_str = str(list_rpf[id_line]); mfe_str = str(list_mfe[id_line])
            list_seq_rpf_mfe.append([seq_str, float(rpf_str), float(mfe_str)])
            w.write(seq_str + "\t" + rpf_str + "\t" + mfe_str + "\n")

    data_gen_optim["data_optim_rpf"][idx_optim]  = list_rpf
    data_gen_optim["data_optim_mfe"][idx_optim]  = list_mfe
    data_gen_optim["data_optim_cscg"][idx_optim] = list_cscgs
    if idx_optim == 0:
        data_gen_optim["data_rpf_base"] = [list_rpf[0]] * len(list_rpf)
        data_gen_optim["data_mfe_base"] = [list_mfe[0]] * len(list_mfe)

    print(f"[epoch {idx_optim+1}] unique seqs generated: {len(set(s[0] for s in list_seq_rpf_mfe))}", flush=True)
    check.calc_mean_error_rate(result_dir + "/optim_results.txt")

    temp_num = len(list_seq_rpf_mfe)
    list_seq_mfe_sort = sorted(list_seq_rpf_mfe[temp_num//4:], key=lambda x: x[-1])
    data_gen_optim["seqs_optim_gen"][idx_optim] = list_seq_mfe_sort

    temp_list_seqs = list_seq_mfe_sort[:model_config["num_optim_top"]]
    temp_list_mfes = get_mfe_sim(temp_list_seqs)

    best_idx, best_rpf = 0, 0
    for i, (mfe_v, seq_v) in enumerate(zip(temp_list_mfes, temp_list_seqs)):
        if seq_v[1] >= best_rpf:
            best_idx = i; best_rpf = seq_v[1]

    best_seq = list_seq_mfe_sort[best_idx][0]
    best_rpf_val   = list_seq_mfe_sort[best_idx][1]
    best_mfe_model = list_seq_mfe_sort[best_idx][-1]
    best_mfe_tool  = temp_list_mfes[best_idx]

    print(f"[epoch {idx_optim+1}] best seq rpf (model): {best_rpf_val:.3f}", flush=True)
    print(f"[epoch {idx_optim+1}] best seq mfe (tool) : {best_mfe_tool:.3f}", flush=True)

    data_gen_optim["results_rpf"][idx_optim] = best_rpf_val
    data_gen_optim["results_mfe"][idx_optim] = [best_mfe_model, best_mfe_tool]
    data_gen_optim["seq_optim_best"][idx_optim] = best_seq

    # Legacy single-file output (overwritten each epoch)
    with open(model_config["data_folder"] + "/optim_data.txt", mode="w") as w:
        w.write(best_seq + "\n")

    temp_optim_seqs = torch.concatenate(list_seq_gens[-5:],dim=0).permute(0,2,1).detach().cpu().numpy()
    data_gen_optim['mask_optim_dist'][idx_optim] = np.mean(temp_optim_seqs, axis=0).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Per-gene output saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_gene_results(gene_name, fold_gamma, mfe_weight, env, best_seq_per_epoch,
                       results_rpf, results_mfe, full_mrna, utr5, utr3,
                       n_codons_csv, n_epochs, fold_penalty_mean):
    """Save structured per-gene output: FASTA + JSON metadata."""
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical run tag
    fold_tag = f"fold{fold_gamma:.2f}".replace('.', '')
    mfe_tag  = f"mfe{mfe_weight:.2f}".replace('.', '')
    tag = f"{gene_name.upper()}_{fold_tag}_{mfe_tag}_{env}_ep{n_epochs}"

    # Final best sequence = from the last epoch
    final_best = best_seq_per_epoch[n_epochs - 1]

    # FASTA with full context (UTR5 + CDS + UTR3) and CDS-only
    fasta_path = OPTIMIZED_DIR / f"{tag}_best.fasta"
    with open(fasta_path, 'w') as f:
        f.write(f">{gene_name.upper()}_optimized_CDS fold_gamma={fold_gamma} "
                f"mfe_weight={mfe_weight} env={env} epochs={n_epochs}\n")
        f.write(final_best + "\n")
        f.write(f">{gene_name.upper()}_optimized_fullmRNA utr5+cds+utr3\n")
        f.write(utr5 + final_best + utr3 + "\n")

    # JSON metadata
    json_path = OPTIMIZED_DIR / f"{tag}_results.json"
    metadata = {
        "gene": gene_name.upper(),
        "config": {
            "fold_gamma":  fold_gamma,
            "mfe_weight":  mfe_weight,
            "env":         env,
            "n_epochs":    n_epochs,
            "gnn_included": fold_gamma > 0,
        },
        "sequence": {
            "cds_only":     final_best,
            "cds_len_nt":   len(final_best),
            "cds_len_codons": n_codons_csv,
            "utr5":         utr5,
            "utr3":         utr3,
            "full_mrna":    utr5 + final_best + utr3,
            "full_mrna_len": len(utr5) + len(final_best) + len(utr3),
        },
        "metrics_per_epoch": {
            "rpf_model":  {str(i+1): float(results_rpf[i]) for i in range(n_epochs)},
            "mfe_model":  {str(i+1): float(results_mfe[i][0]) for i in range(n_epochs)},
            "mfe_tool":   {str(i+1): float(results_mfe[i][1]) for i in range(n_epochs)},
            "best_seq":   {str(i+1): best_seq_per_epoch[i] for i in range(n_epochs)},
        },
        "final": {
            "rpf_model": float(results_rpf[n_epochs - 1]),
            "mfe_model": float(results_mfe[n_epochs - 1][0]),
            "mfe_tool":  float(results_mfe[n_epochs - 1][1]),
            "fold_penalty_mean": float(fold_penalty_mean),
        },
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[output] Saved: {fasta_path}", flush=True)
    print(f"[output] Saved: {json_path}", flush=True)
    return fasta_path, json_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global FOLD_GAMMA, GNN_UPDATE_EVERY

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.abspath(os.path.curdir)
    if base_dir and base_dir != cwd:
        shutil.copytree(base_dir, cwd, dirs_exist_ok=True)
        base_dir = cwd
    print(f"base dir: {base_dir}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cds",         type=str, required=True, help="Gene name")
    parser.add_argument("--cds_seq",     type=str, required=True, help="Full mRNA sequence")
    parser.add_argument("--alpha",       type=float, default=100.0, help="Target RPF")
    parser.add_argument("--beta",        type=float, default=100.0, help="MFE normalization index")
    parser.add_argument("--mfe_weight",  type=float, default=0.0,
                        help="Weight for MFE loss (0=disabled, 0.7=paper default, expensive)")
    parser.add_argument("--env",         type=str, default="HEK293T")
    parser.add_argument("--optim_epoch", type=int, default=10)
    parser.add_argument("--csv",         type=str, default=None)
    parser.add_argument("--fold_gamma",  type=float, default=0.3,
                        help="Fold penalty weight (0=baseline, 0.3=default, 0.5=strong)")
    parser.add_argument("--gnn_update_every", type=int, default=1,
                        help="Recompute GNN every N steps (higher=faster)")
    args = parser.parse_args()
    print(args, flush=True)

    FOLD_GAMMA       = args.fold_gamma
    GNN_UPDATE_EVERY = args.gnn_update_every

    os.makedirs(os.path.join(base_dir, "data", "cds"), exist_ok=True)
    args.model_config_g = base_dir + "/model_config.py"
    args.save_dir       = "./logs/"
    args.result_dir     = "./samples/"
    args.model_config_s = base_dir + "/score_model/model_config.json"
    args.best_model     = base_dir + "/score_model/best_model.p"

    data_gen_optim = {k: {} for k in [
        "data_gen_mfe", "data_optim_rpf", "data_optim_mfe", "data_optim_cscg",
        "results_rpf", "results_mfe", "results_cscg", "seqs_optim_gen",
        "mask_optim_dist", "seq_optim_best", "optim_iteration_logs",
    ]}

    model_config = json.loads(open(base_dir + "/model_config.py").read())["training"]
    model_config["data_folder"]     = base_dir + "/data/cds/"
    model_config["weight_mfe"]      = args.mfe_weight
    model_config["num_optim"]       = args.optim_epoch
    model_config["RNA_condition"]   = args.env
    model_config["rpf_target"]      = args.alpha
    model_config["mfe_norm_index"]  = args.beta
    model_config["using_custom_env"] = args.env not in ['HEK293T','BJ','A549','HeLa']

    # ── Feature matrices ──────────────────────────────────────────────────────
    dict_bp = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"T":[0,0,0,1],"N":[0,0,0,0]}
    list_key2bp = [[dict_bp[k] for k in dict_vocab_inv[i]] for i in range(65)]
    W_K2BP = torch.from_numpy(np.array(list_key2bp)).float()

    with open(base_dir + "/data/pcscg.log", mode="r") as f:
        pcscg_info = f.readlines()[1:]
    list_pcscg_info, list_codon_index = [], []
    for line in pcscg_info:
        line = line.strip().split("\t")
        list_pcscg_info.append(line[1:]); list_codon_index.append(line[0])
    pcscg_info = np.array(list_pcscg_info).astype(np.float64)
    list_pcscg_W = []
    for i in range(50):
        tmp = torch.zeros(65, 1)
        for j, v in enumerate(pcscg_info[:, i]):
            tmp[my_vocab[list_codon_index[j]]] = float(v)
        list_pcscg_W.append(tmp)

    # ── Load GNN and WT CSV features ──────────────────────────────────────────
    full_mrna = args.cds_seq.replace("U", "T").upper()
    print(f"\n[main] Input mRNA: {len(full_mrna)} nt", flush=True)

    gnn_model = load_gnn(GNN_MODEL_PATH, device)
    protein_dense_x, n_codons_csv = load_protein_features(args.cds, device)

    # ── Auto-detect CDS boundaries ────────────────────────────────────────────
    UTR5, UTR3 = '', ''
    if n_codons_csv is not None:
        expected_cds_len_nt = n_codons_csv * 3
        cds_start = find_cds_position(full_mrna, expected_cds_len_nt)
        if cds_start is not None:
            cds_only = full_mrna[cds_start:cds_start + expected_cds_len_nt]
            UTR5 = full_mrna[:cds_start]
            UTR3 = full_mrna[cds_start + expected_cds_len_nt:]
            print(f"[main] CDS auto-detected at nt position {cds_start}", flush=True)
            print(f"[main] 5'UTR: {len(UTR5)} nt  |  CDS: {len(cds_only)} nt ({n_codons_csv} codons)  |  3'UTR: {len(UTR3)} nt", flush=True)
            print(f"[main] CDS starts: {cds_only[:18]}...  ends: ...{cds_only[-18:]}", flush=True)
            args.cds_seq = cds_only
        else:
            print(f"[main] WARNING: couldn't find ORF of length {expected_cds_len_nt} — using full input as CDS", flush=True)
            args.cds_seq = full_mrna
    else:
        print(f"[main] No CSV features → treating whole input as CDS", flush=True)
        args.cds_seq = full_mrna

    orig_seq        = args.cds_seq
    len_gene_ori    = len(orig_seq)
    n_codons_global = len_gene_ori // 3
    check = Check(orig_seq)
    model_config["len_gene_ori"] = len_gene_ori
    model_config["max_len"]      = len_gene_ori
    print(f"[main] RiboDecode max_len (CDS-only): {model_config['max_len']} nt", flush=True)

    model_mfe = mfe_conv_sim(65, model_config["hidden_dim"], model_config["latent_dim"]).to(device)
    z_model   = seq_codon_gen(model_config).to(device)
    weight_mfe = model_config["weight_mfe"]

    # ── Align protein features to CDS length ──────────────────────────────────
    if protein_dense_x is None:
        print("[main] Warning: no protein features — fold_penalty will be zeros", flush=True)
        protein_dense_x = torch.zeros(n_codons_global, N_DENSE_FEATURES, device=device)

    if protein_dense_x.shape[0] != n_codons_global:
        print(f"[main] Aligning protein features {protein_dense_x.shape[0]} → {n_codons_global} codons", flush=True)
        if protein_dense_x.shape[0] > n_codons_global:
            protein_dense_x = protein_dense_x[:n_codons_global]
        else:
            pad = torch.zeros(n_codons_global - protein_dense_x.shape[0],
                              N_DENSE_FEATURES, device=device)
            protein_dense_x = torch.cat([protein_dense_x, pad], dim=0)
    else:
        print(f"[main] Protein features match CDS length perfectly: {n_codons_global} codons ✓", flush=True)

    # ── Build protein 3D contact edges (ONCE, protein doesn't change) ─────────
    print(f"\n[main] Loading protein 3D contact edges from PDB...", flush=True)
    p_src, p_dst, p_feats = build_prot_edges(args.cds, n_codons_global)
    if p_src:
        n_prot_edges = len(p_src) // 2  # bidirectional
        print(f"[main] Protein edges: {n_prot_edges} contacts "
              f"(Cα<{CA_THRESHOLD}Å, sep≥{MIN_SEQ_SEP}) ✓", flush=True)
    else:
        print(f"[main] No PDB found for {args.cds} — GNN will use seq+RNA only", flush=True)
    prot_edges_cached = (p_src, p_dst, p_feats)

    # ── Optimization loop ─────────────────────────────────────────────────────
    num_optim = model_config["num_optim"]
    print(f"\n[main] Starting optimization: {num_optim} epoch(s)", flush=True)
    print(f"[main] fold_gamma={FOLD_GAMMA}  mfe_weight={weight_mfe}  "
          f"env={args.env}  alpha={args.alpha}  beta={args.beta}", flush=True)

    for idx_optim in range(num_optim):
        time_start = time.time()
        if model_config["weight_mfe"] > 0:
            print(f"\n[main] Running gen_train + mfe_train (mfe_weight={weight_mfe})...", flush=True)
            gen_train(args, idx_optim, model_config, data_gen_optim, device)
            mfe_train(args, idx_optim, model_config, data_gen_optim, model_mfe, device)
        optim(args, idx_optim, model_config, data_gen_optim,
              model_mfe, z_model, gnn_model, protein_dense_x,
              UTR5, UTR3, weight_mfe, check, list_pcscg_W, W_K2BP,
              base_dir, device, prot_edges_cached)
        print(f"[main] loop time: {time.time()-time_start:.0f}s | epoch {idx_optim+1}/{num_optim} done", flush=True)

    # ── Final per-gene output ─────────────────────────────────────────────────
    print("\n══════ RPF summary ══════", flush=True)
    for i in range(num_optim):
        print(f"  epoch {i+1}: rpf={data_gen_optim['results_rpf'][i]:.3f}", flush=True)
    print("\n══════ MFE summary ══════", flush=True)
    for i in range(num_optim):
        mm = data_gen_optim['results_mfe'][i]
        print(f"  epoch {i+1}: mfe_model={mm[0]:.3f}  mfe_tool={mm[1]:.3f}", flush=True)

    # Final fold_penalty_mean (from last epoch, if available)
    fp_final = 0.0
    if num_optim > 0 and "data_optim_mfe" in data_gen_optim and (num_optim-1) in data_gen_optim.get("data_optim_mfe", {}):
        # we don't track fold_penalty_mean in data_gen_optim; just use 0 for now
        pass

    try:
        save_gene_results(
            gene_name         = args.cds,
            fold_gamma        = FOLD_GAMMA,
            mfe_weight        = weight_mfe,
            env               = args.env,
            best_seq_per_epoch = data_gen_optim["seq_optim_best"],
            results_rpf       = data_gen_optim["results_rpf"],
            results_mfe       = data_gen_optim["results_mfe"],
            full_mrna         = full_mrna,
            utr5              = UTR5,
            utr3              = UTR3,
            n_codons_csv      = n_codons_global,
            n_epochs          = num_optim,
            fold_penalty_mean = fp_final,
        )
    except Exception as e:
        print(f"[output] WARNING: saving per-gene output failed: {e}", flush=True)


if __name__ == '__main__':
    main()