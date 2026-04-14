#!/usr/bin/env python3
# ARCHIVED - superseded by v5
"""
GNN Training v4 — CODON + AMINO ACID FEATURES + 3 LAYERS

New features vs v3:
  codon_tai          — tRNA Adaptation Index (translation speed)
  codon_gc3          — GC at wobble position
  codon_rel_opt      — optimality WITHIN synonymous family (0=worst, 1=best)
  aa_hydrophobicity  — Kyte-Doolittle scale (drives folding)
  aa_charge          — -1/0/+1 (exit tunnel interactions)
  is_proline         — proline causes specific ribosome pausing

3 GATv2 layers: with codon features, more hops help detect
rare codon clusters and local sequence context.
"""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from scipy import stats
from torch.nn import LayerNorm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/Users/jonathanopitz/Desktop/Master")
WT_DIR       = BASE_DIR / "data/ribo_counts"
SIM_DIR      = BASE_DIR / "data/ribo_counts_simulated"
GENES_DIR    = BASE_DIR / "data/genes"
PDB_DIR      = BASE_DIR / "data/pdb"
ISOFORM_JSON = BASE_DIR / "isoform_selection.json"
OUT_DIR      = BASE_DIR / "data/results"

SEED         = 42
VAL_RATIO    = 0.2
EPOCHS       = 300
BATCH_SIZE   = 4
LR           = 5e-4
PATIENCE     = 8
HIDDEN       = 32
HEADS        = 2
EDGE_DIM     = 5
CA_THRESHOLD = 8.0
MIN_SEQ_SEP  = 6
LOSS_ALPHA   = 5.0

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

with open(ISOFORM_JSON) as f:
    isoforms = json.load(f)

# ─── Codon & amino acid lookups ───────────────────────────────────────────────

CODON_TAI = {
    'TTT': 0.42, 'TTC': 1.00, 'TTA': 0.08, 'TTG': 0.42,
    'CTT': 0.42, 'CTC': 0.58, 'CTA': 0.08, 'CTG': 1.00,
    'ATT': 0.58, 'ATC': 1.00, 'ATA': 0.08, 'ATG': 1.00,
    'GTT': 0.42, 'GTC': 0.58, 'GTA': 0.08, 'GTG': 1.00,
    'TCT': 0.58, 'TCC': 0.75, 'TCA': 0.25, 'TCG': 0.17,
    'CCT': 0.58, 'CCC': 0.75, 'CCA': 0.42, 'CCG': 0.17,
    'ACT': 0.58, 'ACC': 1.00, 'ACA': 0.42, 'ACG': 0.17,
    'GCT': 0.75, 'GCC': 1.00, 'GCA': 0.42, 'GCG': 0.17,
    'TAT': 0.42, 'TAC': 1.00, 'TAA': 0.00, 'TAG': 0.00,
    'CAT': 0.42, 'CAC': 1.00, 'CAA': 0.42, 'CAG': 1.00,
    'AAT': 0.42, 'AAC': 1.00, 'AAA': 0.42, 'AAG': 1.00,
    'GAT': 0.42, 'GAC': 1.00, 'GAA': 0.42, 'GAG': 1.00,
    'TGT': 0.42, 'TGC': 1.00, 'TGA': 0.00, 'TGG': 1.00,
    'CGT': 0.42, 'CGC': 0.75, 'CGA': 0.08, 'CGG': 0.17,
    'AGT': 0.25, 'AGC': 0.75, 'AGA': 0.42, 'AGG': 0.25,
    'GGT': 0.42, 'GGC': 1.00, 'GGA': 0.25, 'GGG': 0.17,
}

CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Kyte-Doolittle hydrophobicity, normalized to [0, 1]
AA_HYDRO_RAW = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5, '*': 0.0
}
_hmin = min(AA_HYDRO_RAW.values())
_hmax = max(AA_HYDRO_RAW.values())
AA_HYDRO = {k: (v - _hmin) / (_hmax - _hmin) for k, v in AA_HYDRO_RAW.items()}

AA_CHARGE = {
    'R': 1, 'K': 1, 'H': 0.5,  # positive
    'D': -1, 'E': -1,            # negative
}

# Pre-compute max tAI per amino acid for relative optimality
_aa_to_codons = {}
for c, aa in CODON_TO_AA.items():
    _aa_to_codons.setdefault(aa, []).append(c)
AA_MAX_TAI = {aa: max(CODON_TAI.get(c, 0) for c in codons)
              for aa, codons in _aa_to_codons.items()}


def codon_features(codon_str):
    """Return [tai, gc3, rel_optimality, hydrophobicity, charge, is_proline]."""
    if pd.isna(codon_str):
        return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    c = str(codon_str).upper().strip().replace('U', 'T')
    if len(c) < 3:
        return [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]

    tai = CODON_TAI.get(c, 0.5)
    gc3 = 1.0 if c[2] in 'GC' else 0.0
    aa = CODON_TO_AA.get(c, '*')
    max_tai = AA_MAX_TAI.get(aa, 1.0)
    rel_opt = tai / max_tai if max_tai > 0 else 0.5
    hydro = AA_HYDRO.get(aa, 0.5)
    charge = AA_CHARGE.get(aa, 0.0)
    is_pro = 1.0 if aa == 'P' else 0.0

    return [tai, gc3, rel_opt, hydro, charge, is_pro]


# ─── Node features ────────────────────────────────────────────────────────────

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

# 18 struct + 1 ss_class + 1 is_denovo + 6 codon/aa = 26
N_FEATURES = len(STRUCT_COLS) + 1 + 1 + 6


def build_node_features(cds, is_denovo=False):
    feats = []
    for col in STRUCT_COLS:
        if col in cds.columns:
            vals = pd.to_numeric(cds[col], errors='coerce').fillna(0.0).values
        else:
            vals = np.zeros(len(cds))
        feats.append(vals)

    ss_enc = (cds.get('rna_local_ss_class', pd.Series('weak', index=cds.index))
              .map({'stem': 1.0, 'weak': 0.5, 'open_loop': 0.0}).fillna(0.0).values)
    feats.append(ss_enc)
    feats.append(np.full(len(cds), 1.0 if is_denovo else 0.0))

    # Codon + amino acid features (6 per codon)
    if 'codon' in cds.columns:
        codon_feats = np.array([codon_features(c) for c in cds['codon']], dtype=np.float32)
    else:
        codon_feats = np.full((len(cds), 6), 0.5, dtype=np.float32)

    for i in range(6):
        feats.append(codon_feats[:, i])

    x = np.stack(feats, axis=1).astype(np.float32)
    return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32)


# ─── Edge builders ────────────────────────────────────────────────────────────

def build_seq_edges(n):
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    feats = [[1, 0, 0, 1.0, 1.0 / n]] * len(src)
    return src, dst, feats

def build_prot_edges(gene, n_cds):
    pdb_path = find_pdb(gene)
    if pdb_path is None: return [], [], []
    coords = parse_ca(pdb_path)
    residues = sorted([r for r in coords if r < n_cds])
    if len(residues) < 2: return [], [], []
    src, dst, feats = [], [], []
    for i, ri in enumerate(residues):
        for j in range(i + 1, len(residues)):
            rj = residues[j]
            if abs(ri - rj) < MIN_SEQ_SEP: continue
            d = np.linalg.norm(coords[ri] - coords[rj])
            if d < CA_THRESHOLD:
                src.extend([ri, rj]); dst.extend([rj, ri])
                feats.extend([[0, 1, 0, d / CA_THRESHOLD, abs(ri - rj) / n_cds]] * 2)
    return src, dst, feats

def build_rna_edges(gene, n_codons, is_denovo=False):
    fasta = find_fasta(gene, denovo=is_denovo)
    if not fasta: fasta = find_fasta(gene, denovo=False)
    if not fasta: return [], [], []
    seq = get_sequence(fasta)
    if len(seq) < 100: return [], [], []
    try:
        result = subprocess.run(["RNAfold", "--noPS"], input=seq + "\n",
                                capture_output=True, text=True, timeout=10)
        if result.returncode != 0: return [], [], []
        db = result.stdout.strip().split('\n')[-1].split()[0]
    except Exception: return [], [], []
    stack, pairs = [], []
    for i, ch in enumerate(db):
        if ch in '({[': stack.append(i)
        elif ch in ')}]' and stack: pairs.append((stack.pop(), i))
    src, dst, feats = [], [], []
    for i_nt, j_nt in pairs:
        ci, cj = i_nt // 3, j_nt // 3
        if ci == cj or ci >= n_codons or cj >= n_codons or abs(ci - cj) < 3: continue
        src.extend([ci, cj]); dst.extend([cj, ci])
        feats.extend([[0, 0, 1, 0.0, abs(ci - cj) / n_codons]] * 2)
    return src, dst, feats


# ─── Helpers ───────────────────────────────────────────────────────────────────

def find_pdb(gene):
    for pat in [f"*{gene.upper()}*.pdb", f"*{gene.lower()}*.pdb"]:
        c = list(PDB_DIR.glob(pat))
        if c: return max(c, key=lambda p: p.stat().st_size)
    return None

def parse_ca(pdb_path):
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                r = int(line[22:26].strip()) - 1
                if r not in coords:
                    coords[r] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return coords

def find_fasta(gene, denovo=False):
    g = gene.upper()
    if g not in isoforms or isoforms[g].get("status") != "ok": return None
    best = isoforms[g].get("best_isoform_full") or isoforms[g].get("best_isoform")
    if denovo:
        c = (list(GENES_DIR.glob(f"*{g}*gemorna*.fasta")) + list(GENES_DIR.glob(f"*{g}*denovo*.fasta")))
    else:
        c = list(GENES_DIR.glob(f"*{g}*{best}*.fasta"))
    if not c: c = list(GENES_DIR.glob(f"*{g}*.fasta"))
    return max(c, key=lambda p: p.stat().st_size) if c else None

def get_sequence(fp):
    return str(next(SeqIO.parse(fp, "fasta")).seq).upper().replace("T", "U")

def extract_gene_name(fp):
    stem = fp.stem.lower()
    for s in ['_with_folddemand', '_with_rnaplfold', '_with_structure',
              '_ribosome_counts', '_simulated_ribo', '_gemorna',
              '_simulated', '_denovo', '_synthetic', '_optimized']:
        stem = stem.replace(s, '')
    return stem.strip('_').upper()

def discover_files():
    files = []
    for f in sorted(WT_DIR.glob("*_with_folddemand.csv")):
        files.append({'path': f, 'gene': extract_gene_name(f), 'is_denovo': False})
    if SIM_DIR.exists():
        for f in sorted(SIM_DIR.glob("*_with_folddemand.csv")):
            files.append({'path': f, 'gene': extract_gene_name(f), 'is_denovo': True})
    return files


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph(df, gene, edge_config, is_denovo=False):
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    if len(cds) < 20 or 'folding_demand' not in cds.columns or cds['folding_demand'].isna().all():
        return None
    n = len(cds)
    x = build_node_features(cds, is_denovo=is_denovo)
    y = torch.tensor(cds['folding_demand'].fillna(0).values, dtype=torch.float32)

    all_src, all_dst, all_feats = [], [], []
    if edge_config != 'none':
        if 'seq' in edge_config:
            s, d, f = build_seq_edges(n); all_src += s; all_dst += d; all_feats += f
        if 'prot' in edge_config:
            s, d, f = build_prot_edges(gene, n); all_src += s; all_dst += d; all_feats += f
        if 'rna' in edge_config:
            s, d, f = build_rna_edges(gene, n, is_denovo); all_src += s; all_dst += d; all_feats += f
    if not all_src:
        all_src = list(range(n)); all_dst = list(range(n))
        all_feats = [[0, 0, 0, 0, 0]] * n

    return Data(
        x=x, edge_index=torch.tensor([all_src, all_dst], dtype=torch.long),
        edge_attr=torch.tensor(all_feats, dtype=torch.float32),
        y=y, num_nodes=n, gene=gene, is_denovo=is_denovo,
        n_seq=sum(1 for f in all_feats if f[0] == 1) // 2,
        n_prot=sum(1 for f in all_feats if f[1] == 1) // 2,
        n_rna=sum(1 for f in all_feats if f[2] == 1) // 2,
    )


# ─── 3-Layer GATv2 Model ──────────────────────────────────────────────────────

class FoldingGATv2(torch.nn.Module):
    """3-layer GATv2 — detects codon clusters via wider receptive field."""
    def __init__(self, in_dim=N_FEATURES, hid=HIDDEN, heads=HEADS, edge_dim=EDGE_DIM):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, hid)
        self.norm0 = LayerNorm(hid)

        self.conv1 = GATv2Conv(hid, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.15, add_self_loops=False)
        self.norm1 = LayerNorm(hid * heads)
        self.drop1 = torch.nn.Dropout(0.3)
        self.res1 = torch.nn.Linear(hid, hid * heads, bias=False)

        self.conv2 = GATv2Conv(hid * heads, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.15, add_self_loops=False)
        self.norm2 = LayerNorm(hid * heads)
        self.drop2 = torch.nn.Dropout(0.3)
        self.res2 = torch.nn.Linear(hid * heads, hid * heads, bias=False)

        self.conv3 = GATv2Conv(hid * heads, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.15, add_self_loops=False)
        self.norm3 = LayerNorm(hid * heads)
        self.drop3 = torch.nn.Dropout(0.3)
        self.res3 = torch.nn.Linear(hid * heads, hid * heads, bias=False)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hid * heads, 32),
            torch.nn.ELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.norm0(self.proj(x)))
        r = x

        x = self.conv1(x, ei, ea); x = self.drop1(F.elu(self.norm1(x)))
        x = x + self.res1(r); r = x

        x = self.conv2(x, ei, ea); x = self.drop2(F.elu(self.norm2(x)))
        x = x + self.res2(r); r = x

        x = self.conv3(x, ei, ea); x = self.drop3(F.elu(self.norm3(x)))
        x = x + self.res3(r)

        return self.head(x).squeeze(-1)


class MLPBaseline(torch.nn.Module):
    def __init__(self, in_dim=N_FEATURES):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 48), torch.nn.ELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(48, 32), torch.nn.ELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1), torch.nn.Sigmoid()
        )
    def forward(self, data):
        return self.net(data.x).squeeze(-1)


# ─── Loss ──────────────────────────────────────────────────────────────────────

def hybrid_loss(pred, target, alpha=LOSS_ALPHA):
    weights = 1.0 + alpha * target
    mse_part = (weights * (pred - target) ** 2).mean()
    bce_part = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target, reduction='mean')
    return 0.7 * mse_part + 0.3 * bce_part


# ─── Training / eval ──────────────────────────────────────────────────────────

def train_one(model, loader, opt):
    model.train()
    tot, n = 0, 0
    for b in loader:
        b = b.to(DEVICE); opt.zero_grad()
        loss = hybrid_loss(model(b), b.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * b.num_nodes; n += b.num_nodes
    return tot / n

@torch.no_grad()
def eval_one(model, loader):
    model.eval()
    tot, n = 0, 0
    ps, ts, dns = [], [], []
    for b in loader:
        b = b.to(DEVICE); p = model(b)
        tot += hybrid_loss(p, b.y).item() * b.num_nodes; n += b.num_nodes
        ps.append(p.cpu().numpy()); ts.append(b.y.cpu().numpy())
        # is_denovo is at index 19 (18 struct + 1 ss_class + [is_denovo])
        dns.append(b.x[:, 19].cpu().numpy())
    return tot / n, np.concatenate(ps), np.concatenate(ts), np.concatenate(dns)

def compute_metrics(p, t, dn=None):
    m = {
        'mse': float(np.mean((p - t) ** 2)),
        'mae': float(np.mean(np.abs(p - t))),
        'pearson': float(stats.pearsonr(p, t)[0]),
        'spearman': float(stats.spearmanr(p, t)[0]),
        'pred_min': float(p.min()), 'pred_max': float(p.max()),
    }
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        mask = (t >= lo) & (t < hi)
        if mask.sum() > 0:
            m[f'mse_{lo:.1f}_{hi:.1f}'] = float(np.mean((p[mask] - t[mask]) ** 2))
    if dn is not None:
        wt = dn < 0.5; dnm = dn >= 0.5
        if wt.sum() > 10:
            m['pearson_wt'] = float(stats.pearsonr(p[wt], t[wt])[0])
            m['mse_wt'] = float(np.mean((p[wt] - t[wt]) ** 2))
        if dnm.sum() > 10:
            m['pearson_dn'] = float(stats.pearsonr(p[dnm], t[dnm])[0])
            m['mse_dn'] = float(np.mean((p[dnm] - t[dnm]) ** 2))
    return m


def plot(tl, vl, vp, vt, dn, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0,0].plot(tl, label='train', alpha=0.8); ax[0,0].plot(vl, label='val', alpha=0.8)
    ax[0,0].set_xlabel('Epoch'); ax[0,0].set_ylabel('Loss'); ax[0,0].set_title(f'Loss ({name})'); ax[0,0].legend()

    ax[0,1].scatter(vt, vp, alpha=0.05, s=2); ax[0,1].plot([0,1],[0,1],'r--')
    r = stats.pearsonr(vp, vt)[0]
    ax[0,1].set_xlabel('True fd'); ax[0,1].set_ylabel('Pred'); ax[0,1].set_title(f'r={r:.3f}, r²={r**2:.3f}')

    res = vp - vt
    ax[0,2].hist(res, bins=50, alpha=0.7, color='steelblue'); ax[0,2].axvline(0, color='red')
    ax[0,2].set_xlabel('Residual'); ax[0,2].set_title('Residuals')

    ax[1,0].hist(vt, bins=50, alpha=0.5, color='steelblue', label='true')
    ax[1,0].hist(vp, bins=50, alpha=0.5, color='coral', label='pred'); ax[1,0].legend()
    ax[1,0].set_xlabel('fd'); ax[1,0].set_title('Distributions')

    ranges = [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.0)]
    mses = [np.mean((vp[(vt>=lo)&(vt<hi)]-vt[(vt>=lo)&(vt<hi)])**2) if ((vt>=lo)&(vt<hi)).sum()>0 else 0 for lo,hi in ranges]
    ax[1,1].bar(range(5), mses, color='steelblue')
    ax[1,1].set_xticks(range(5)); ax[1,1].set_xticklabels([f'{lo}-{hi}' for lo,hi in ranges], fontsize=9)
    ax[1,1].set_xlabel('True fd range'); ax[1,1].set_ylabel('MSE'); ax[1,1].set_title('Error by range')

    if dn is not None:
        wt = dn < 0.5; dnm = dn >= 0.5
        if wt.sum() > 10 and dnm.sum() > 10:
            ax[1,2].scatter(vt[wt], vp[wt], alpha=0.05, s=2, c='steelblue', label='WT')
            ax[1,2].scatter(vt[dnm], vp[dnm], alpha=0.05, s=2, c='coral', label='de novo')
            ax[1,2].plot([0,1],[0,1],'r--'); ax[1,2].legend(fontsize=8, markerscale=5)
            ax[1,2].set_title('WT vs de novo')

    plt.tight_layout(); plt.savefig(OUT_DIR / f'gnn_v4_{name}.png', dpi=150); plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', default='seq+prot+rna',
                        choices=['none', 'seq', 'seq+prot', 'seq+rna', 'seq+prot+rna'])
    args = parser.parse_args()
    ec = args.edges

    print(f"\n{'═'*60}")
    print(f"  GNN v4 — 3-Layer + Codon/AA Features")
    print(f"  Edges: {ec} | Device: {DEVICE}")
    print(f"  New: tai, gc3, rel_opt, hydrophobicity, charge, is_proline")
    print(f"{'═'*60}\n")

    file_info = discover_files()
    n_wt = sum(1 for f in file_info if not f['is_denovo'])
    n_dn = sum(1 for f in file_info if f['is_denovo'])
    print(f"  Files: {n_wt} WT + {n_dn} de novo")

    graphs, gene_ids = [], []
    for fi in file_info:
        g = build_graph(pd.read_csv(fi['path']), fi['gene'], ec, fi['is_denovo'])
        if g is not None:
            graphs.append(g); gene_ids.append(fi['gene'])
            tag = "dn" if fi['is_denovo'] else "wt"
            print(f"  {fi['gene']:>10} ({tag}): {g.num_nodes:>4} nodes | "
                  f"seq={g.n_seq:>4} prot={g.n_prot:>4} rna={g.n_rna:>4}")

    print(f"\n  Total: {len(graphs)} graphs, {sum(g.num_nodes for g in graphs)} nodes")
    print(f"  Features per node: {graphs[0].x.shape[1]}")

    unique_genes = sorted(set(gene_ids))
    rng = random.Random(SEED); rng.shuffle(unique_genes)
    n_val = max(1, int(len(unique_genes) * VAL_RATIO))
    val_genes = set(unique_genes[:n_val])

    train_data = [g for g, gid in zip(graphs, gene_ids) if gid not in val_genes]
    val_data = [g for g, gid in zip(graphs, gene_ids) if gid in val_genes]
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  Val genes: {sorted(val_genes)}")

    train_ld = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_data, batch_size=BATCH_SIZE)

    in_dim = graphs[0].x.shape[1]
    model = MLPBaseline(in_dim) if ec == 'none' else FoldingGATv2(in_dim)
    model = model.to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {type(model).__name__} | {n_par:,} params | 3 layers\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    best_vl, pat_ct = float('inf'), 0
    tls, vls, etimes = [], [], []
    best_vp, best_vt, best_dn = None, None, None

    for ep in range(EPOCHS):
        t0 = time.time()
        tl = train_one(model, train_ld, opt)
        vl, vp, vt, dn = eval_one(model, val_ld)
        etimes.append(time.time() - t0)
        tls.append(tl); vls.append(vl); sched.step(vl)

        if vl < best_vl:
            best_vl = vl; pat_ct = 0; best_vp, best_vt, best_dn = vp, vt, dn
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_DIR / f"gnn_v4_{ec.replace('+','_')}.pt")
        else:
            pat_ct += 1

        if ep % 10 == 0 or pat_ct >= PATIENCE:
            print(f"  Ep {ep:3d} | train={tl:.5f} val={vl:.5f} best={best_vl:.5f} "
                  f"lr={opt.param_groups[0]['lr']:.1e} | {np.mean(etimes[-10:]):.1f}s/ep")
        if pat_ct >= PATIENCE:
            print(f"  Early stop at epoch {ep}"); break

    m = compute_metrics(best_vp, best_vt, best_dn)
    tag = ec.replace('+', '_')

    print(f"\n{'═'*60}")
    print(f"  RESULTS v4 — {ec} (3-layer + codon/aa features)")
    print(f"{'═'*60}")
    print(f"  MSE:        {m['mse']:.5f}")
    print(f"  MAE:        {m['mae']:.5f}")
    print(f"  Pearson r:  {m['pearson']:.4f}  (r²={m['pearson']**2:.4f})")
    print(f"  Spearman r: {m['spearman']:.4f}")
    print(f"  Pred range: [{m['pred_min']:.3f}, {m['pred_max']:.3f}]")
    print(f"  Params:     {n_par:,}")

    if 'pearson_wt' in m: print(f"\n  WT:     r={m['pearson_wt']:.4f}  MSE={m['mse_wt']:.5f}")
    if 'pearson_dn' in m: print(f"  DeNovo: r={m['pearson_dn']:.4f}  MSE={m['mse_dn']:.5f}")

    print(f"\n  MSE by range:")
    for lo, hi in [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.0)]:
        key = f'mse_{lo:.1f}_{hi:.1f}'
        if key in m: print(f"    [{lo:.1f}, {hi:.1f}): {m[key]:.5f}")

    plot(tls, vls, best_vp, best_vt, best_dn, tag)

    # Compare with v3 (no codon, 2 layers)
    for ver, pref in [('v3', 'v3_'), ('v4_2feat', 'v4_')]:
        prev = OUT_DIR / f"metrics_{pref}{tag}.json"
        if prev.exists():
            p = json.loads(prev.read_text())
            print(f"\n  vs {ver}: r={p['pearson']:.4f}→{m['pearson']:.4f} "
                  f"({m['pearson']-p['pearson']:+.4f}) | "
                  f"MSE@0.7+ {p.get('mse_0.7_1.0',0):.5f}→{m.get('mse_0.7_1.0',0):.5f}")

    m.update({'edges': ec, 'params': n_par, 'epochs': len(tls), 'layers': 3,
              'features': in_dim, 'val_genes': sorted(val_genes),
              'avg_epoch_time': float(np.mean(etimes))})
    with open(OUT_DIR / f"metrics_v4_{tag}.json", 'w') as f:
        json.dump(m, f, indent=2)
    print(f"\n  Saved: metrics_v4_{tag}.json")


if __name__ == '__main__':
    main()