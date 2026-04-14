#!/usr/bin/env python3
# ARCHIVED - superseded by v5
"""
GNN Training — Predict folding_demand from structure features

═══════════════════════════════════════════════════════════════
PURPOSE:
  Train a GNN that predicts the folding_demand score per codon
  from protein and RNA structure features. For a NEW protein
  (no Ribo-seq available), the trained GNN predicts WHERE
  translational pauses are needed for correct folding.

ARCHITECTURE:
  GATv2Conv with edge features (edge_dim=5)
  ~15k parameters — appropriate for 180 graphs

TARGET:
  folding_demand = protein_need × ribo_confirmation
  (computed by add_folding_demand.py v4)

EDGE TYPES:
  1. Sequential  (i ↔ i+1) — mRNA backbone
  2. Protein 3D  (Cα < 8Å) — AlphaFold contacts
  3. RNA 3D      (base-pairs) — RNAfold structure

ABLATION (run with --edges):
  none          — MLP baseline
  seq           — sequential only
  seq+prot      — + protein 3D
  seq+rna       — + RNA base-pairs
  seq+prot+rna  — all three

Usage:
  python train_gnn.py --edges seq+prot+rna
  python run_ablation.py   # runs all configs
═══════════════════════════════════════════════════════════════
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
DATA_DIR     = BASE_DIR / "data/ribo_counts"
GENES_DIR    = BASE_DIR / "data/genes"
PDB_DIR      = BASE_DIR / "data/pdb"
ISOFORM_JSON = BASE_DIR / "isoform_selection.json"
OUT_DIR      = BASE_DIR / "data/results"

SEED         = 42
VAL_RATIO    = 0.2
EPOCHS       = 300
BATCH_SIZE   = 4
LR           = 5e-4
PATIENCE     = 30
HIDDEN       = 32        # smaller than before — 180 graphs
HEADS        = 2
EDGE_DIM     = 5
CA_THRESHOLD = 8.0
MIN_SEQ_SEP  = 6

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

with open(ISOFORM_JSON) as f:
    isoforms = json.load(f)

# ─── Node features ────────────────────────────────────────────────────────────
# NOTE: we do NOT include folding_demand sub-components as features,
# because those are part of the TARGET. The model must learn to predict
# folding_demand from raw structure features, not from pre-computed hints.

NODE_COLS = [
    # Protein structure (what the model needs to learn from)
    'plddt', 'contact_density', 'domain_boundary',
    'domain_boundary_interpro', 'domain_boundary_pae',
    'ss_H', 'ss_E', 'ss_C',
    # RNA structure
    'rna_local_paired_prob',
    'rna_unpaired_1nt', 'rna_unpaired_3nt', 'rna_unpaired_5nt',
    'rna_opening_energy_1nt', 'rna_opening_energy_3nt', 'rna_opening_energy_5nt',
    'rna_paired_prob_window5cod', 'rna_paired_prob_window10cod',
    'rna_struct_change',
]

N_FEATURES = len(NODE_COLS) + 1  # +1 for encoded ss_class


def build_node_features(cds: pd.DataFrame) -> torch.Tensor:
    feats = []
    for col in NODE_COLS:
        if col in cds.columns:
            vals = pd.to_numeric(cds[col], errors='coerce').fillna(0.0).values
        else:
            vals = np.zeros(len(cds))
        feats.append(vals)

    # Encode RNA ss_class
    ss_enc = (cds.get('rna_local_ss_class', pd.Series('weak', index=cds.index))
              .map({'stem': 1.0, 'weak': 0.5, 'open_loop': 0.0}).fillna(0.0).values)
    feats.append(ss_enc)

    x = np.stack(feats, axis=1).astype(np.float32)
    return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32)


# ─── Edge builders ─────────────────────────────────────────────────────────────

def build_seq_edges(n: int):
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    feats = [[1, 0, 0, 1.0, 1.0 / n]] * len(src)
    return src, dst, feats


def build_prot_edges(gene: str, n_cds: int):
    pdb_path = find_pdb(gene)
    if pdb_path is None:
        return [], [], []
    coords = parse_ca(pdb_path)
    residues = sorted([r for r in coords if r < n_cds])
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
                src.extend([ri, rj])
                dst.extend([rj, ri])
                feats.extend([[0, 1, 0, d / CA_THRESHOLD, abs(ri - rj) / n_cds]] * 2)
    return src, dst, feats


def build_rna_edges(gene: str, n_codons: int):
    fasta = find_fasta(gene)
    if not fasta:
        return [], [], []
    seq = get_sequence(fasta)
    if len(seq) < 100:
        return [], [], []

    try:
        result = subprocess.run(
            ["RNAfold", "--noPS"], input=seq + "\n",
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return [], [], []
        db = result.stdout.strip().split('\n')[-1].split()[0]
    except Exception:
        return [], [], []

    stack, pairs = [], []
    for i, ch in enumerate(db):
        if ch in '({[':
            stack.append(i)
        elif ch in ')}]' and stack:
            pairs.append((stack.pop(), i))

    src, dst, feats = [], [], []
    for i_nt, j_nt in pairs:
        ci, cj = i_nt // 3, j_nt // 3
        if ci == cj or ci >= n_codons or cj >= n_codons or abs(ci - cj) < 3:
            continue
        src.extend([ci, cj])
        dst.extend([cj, ci])
        feats.extend([[0, 0, 1, 0.0, abs(ci - cj) / n_codons]] * 2)
    return src, dst, feats


# ─── Helpers ───────────────────────────────────────────────────────────────────

def find_pdb(gene: str):
    for pat in [f"*{gene.upper()}*.pdb", f"*{gene.lower()}*.pdb"]:
        c = list(PDB_DIR.glob(pat))
        if c:
            return max(c, key=lambda p: p.stat().st_size)
    return None


def parse_ca(pdb_path: Path) -> dict:
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                r = int(line[22:26].strip()) - 1
                if r not in coords:
                    coords[r] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return coords


def find_fasta(gene: str):
    g = gene.upper()
    if g not in isoforms or isoforms[g].get("status") != "ok":
        return None
    best = isoforms[g].get("best_isoform_full") or isoforms[g].get("best_isoform")
    c = list(GENES_DIR.glob(f"*{g}*{best}*.fasta"))
    if not c:
        c = list(GENES_DIR.glob(f"*{g}*.fasta"))
    return max(c, key=lambda p: p.stat().st_size) if c else None


def get_sequence(fasta_path: Path) -> str:
    return str(next(SeqIO.parse(fasta_path, "fasta")).seq).upper().replace("T", "U")


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph(df: pd.DataFrame, gene: str, edge_config: str) -> Data | None:
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    if len(cds) < 20:
        return None

    # Check target exists
    if 'folding_demand' not in cds.columns or cds['folding_demand'].isna().all():
        return None

    n = len(cds)
    x = build_node_features(cds)
    y = torch.tensor(cds['folding_demand'].fillna(0).values, dtype=torch.float32)

    # Build edges
    all_src, all_dst, all_feats = [], [], []

    if edge_config != 'none':
        if 'seq' in edge_config:
            s, d, f = build_seq_edges(n)
            all_src += s; all_dst += d; all_feats += f
        if 'prot' in edge_config:
            s, d, f = build_prot_edges(gene, n)
            all_src += s; all_dst += d; all_feats += f
        if 'rna' in edge_config:
            s, d, f = build_rna_edges(gene, n)
            all_src += s; all_dst += d; all_feats += f

    if not all_src:
        all_src = list(range(n))
        all_dst = list(range(n))
        all_feats = [[0, 0, 0, 0, 0]] * n

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_attr = torch.tensor(all_feats, dtype=torch.float32)

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, num_nodes=n, gene=gene,
        n_seq=sum(1 for f in all_feats if f[0] == 1) // 2,
        n_prot=sum(1 for f in all_feats if f[1] == 1) // 2,
        n_rna=sum(1 for f in all_feats if f[2] == 1) // 2,
    )


# ─── Models ────────────────────────────────────────────────────────────────────

class FoldingGATv2(torch.nn.Module):
    """GATv2 with edge features, residual connections, per-node output."""

    def __init__(self, in_dim=N_FEATURES, hid=HIDDEN, heads=HEADS, edge_dim=EDGE_DIM):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, hid)
        self.norm0 = LayerNorm(hid)

        # GATv2 with edge features
        self.conv1 = GATv2Conv(hid, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.2, add_self_loops=False)
        self.norm1 = LayerNorm(hid * heads)
        self.drop1 = torch.nn.Dropout(0.4)
        self.res1 = torch.nn.Linear(hid, hid * heads, bias=False)

        self.conv2 = GATv2Conv(hid * heads, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.2, add_self_loops=False)
        self.norm2 = LayerNorm(hid * heads)
        self.drop2 = torch.nn.Dropout(0.4)
        self.res2 = torch.nn.Linear(hid * heads, hid * heads, bias=False)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hid * heads, 32),
            torch.nn.ELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr

        x = F.elu(self.norm0(self.proj(x)))
        r = x

        x = self.conv1(x, ei, ea)
        x = self.drop1(F.elu(self.norm1(x)))
        x = x + self.res1(r)
        r = x

        x = self.conv2(x, ei, ea)
        x = self.drop2(F.elu(self.norm2(x)))
        x = x + self.res2(r)

        return self.head(x).squeeze(-1)


class MLPBaseline(torch.nn.Module):
    def __init__(self, in_dim=N_FEATURES):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 48),
            torch.nn.ELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(48, 32),
            torch.nn.ELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )

    def forward(self, data):
        return self.net(data.x).squeeze(-1)


# ─── Training / eval ──────────────────────────────────────────────────────────

def train_one(model, loader, opt, crit):
    model.train()
    tot_loss, tot_n = 0, 0
    for b in loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(b), b.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item() * b.num_nodes
        tot_n += b.num_nodes
    return tot_loss / tot_n


@torch.no_grad()
def eval_one(model, loader, crit):
    model.eval()
    tot_loss, tot_n = 0, 0
    preds, trues = [], []
    for b in loader:
        b = b.to(DEVICE)
        p = model(b)
        tot_loss += crit(p, b.y).item() * b.num_nodes
        tot_n += b.num_nodes
        preds.append(p.cpu().numpy())
        trues.append(b.y.cpu().numpy())
    return tot_loss / tot_n, np.concatenate(preds), np.concatenate(trues)


def metrics(p, t):
    return {
        'mse': float(np.mean((p - t) ** 2)),
        'mae': float(np.mean(np.abs(p - t))),
        'pearson': float(stats.pearsonr(p, t)[0]),
        'spearman': float(stats.spearmanr(p, t)[0]),
    }


def plot(tl, vl, vp, vt, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(tl, label='train', alpha=0.8)
    ax[0].plot(vl, label='val', alpha=0.8)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('MSE')
    ax[0].set_title(f'Loss ({name})'); ax[0].legend()

    ax[1].scatter(vt, vp, alpha=0.05, s=2)
    lims = [min(vt.min(), vp.min()), max(vt.max(), vp.max())]
    ax[1].plot(lims, lims, 'r--')
    r = stats.pearsonr(vp, vt)[0]
    ax[1].set_xlabel('True folding_demand'); ax[1].set_ylabel('Predicted')
    ax[1].set_title(f'Pred vs True (r={r:.3f})')

    res = vp - vt
    ax[2].hist(res, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax[2].axvline(0, color='red')
    ax[2].set_xlabel('Residual'); ax[2].set_title('Residuals')

    plt.tight_layout()
    plt.savefig(OUT_DIR / f'gnn_{name}.png', dpi=150)
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', default='seq+prot+rna',
                        choices=['none', 'seq', 'seq+prot', 'seq+rna', 'seq+prot+rna'])
    args = parser.parse_args()
    ec = args.edges

    print(f"\n{'═'*60}")
    print(f"  GNN Training — edges: {ec}")
    print(f"  Target: folding_demand (protein_need × ribo_confirmation)")
    print(f"  Device: {DEVICE}")
    print(f"{'═'*60}\n")

    # Load graphs
    files = sorted(DATA_DIR.glob("*_with_folddemand.csv"))
    if not files:
        print("No *_with_folddemand.csv found. Run add_folding_demand.py first.")
        return

    graphs, names = [], []
    for f in files:
        gene = f.stem.split("_ribosome")[0].split("_with")[0].upper()
        g = build_graph(pd.read_csv(f), gene, ec)
        if g is not None:
            graphs.append(g)
            names.append(gene)
            print(f"  {gene}: {g.num_nodes} nodes | "
                  f"seq={g.n_seq} prot={g.n_prot} rna={g.n_rna}")

    print(f"\n  Total: {len(graphs)} graphs, "
          f"{sum(g.num_nodes for g in graphs)} total nodes")

    # Gene-grouped split
    uniq = list(set(names))
    random.shuffle(uniq)
    n_val = max(1, int(len(uniq) * VAL_RATIO))
    val_g = set(uniq[:n_val])

    train_data = [g for g, n in zip(graphs, names) if n not in val_g]
    val_data = [g for g, n in zip(graphs, names) if n in val_g]
    print(f"  Split: {len(train_data)} train, {len(val_data)} val\n")

    train_ld = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Model
    in_dim = graphs[0].x.shape[1]
    model = (MLPBaseline(in_dim) if ec == 'none'
             else FoldingGATv2(in_dim)).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {'MLP' if ec == 'none' else 'GATv2'} | {n_par:,} params\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = torch.nn.MSELoss()

    # Train
    best_vl, patience_ct = float('inf'), 0
    tls, vls = [], []
    best_vp, best_vt = None, None

    for ep in range(EPOCHS):
        tl = train_one(model, train_ld, opt, crit)
        vl, vp, vt = eval_one(model, val_ld, crit)
        tls.append(tl); vls.append(vl)
        sched.step(vl)

        if vl < best_vl:
            best_vl = vl
            patience_ct = 0
            best_vp, best_vt = vp, vt
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_DIR / f"gnn_{ec.replace('+','_')}.pt")
        else:
            patience_ct += 1

        if ep % 10 == 0 or patience_ct >= PATIENCE:
            print(f"  Ep {ep:3d} | train={tl:.5f} val={vl:.5f} "
                  f"best={best_vl:.5f} lr={opt.param_groups[0]['lr']:.1e}")
        if patience_ct >= PATIENCE:
            print(f"  Early stop at epoch {ep}")
            break

    # Results
    m = metrics(best_vp, best_vt)
    tag = ec.replace('+', '_')

    print(f"\n{'═'*60}")
    print(f"  RESULTS — {ec}")
    print(f"{'═'*60}")
    print(f"  MSE:      {m['mse']:.5f}")
    print(f"  MAE:      {m['mae']:.5f}")
    print(f"  Pearson:  {m['pearson']:.4f}")
    print(f"  Spearman: {m['spearman']:.4f}")
    print(f"  Params:   {n_par:,}")

    plot(tls, vls, best_vp, best_vt, tag)
    print(f"  Plot: {OUT_DIR / f'gnn_{tag}.png'}")

    m.update({'edges': ec, 'params': n_par, 'n_train': len(train_data),
              'n_val': len(val_data), 'epochs': len(tls)})
    with open(OUT_DIR / f"metrics_{tag}.json", 'w') as f:
        json.dump(m, f, indent=2)


if __name__ == '__main__':
    main()