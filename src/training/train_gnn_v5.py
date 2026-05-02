#!/usr/bin/env python3
"""
GNN Training v5 — LEARNED CODON EMBEDDINGS + EXTENDED CONTEXT

Improvements over v4:
  1. LEARNED CODON EMBEDDING (8-dim) replaces 6 handcrafted codon features
     → Model learns WHICH codons pattern together, not just "fast/slow"
     → 64 codons × 8 dims = 512 learned params (cheap)

  2. EXTENDED SEQUENCE EDGES (±1, ±3, ±5 codons)
     → Model sees rare-codon clusters across longer windows
     → Distinguishes single slow codon from cluster of slow codons

  3. Keeps v4 additions:
     - 3-layer GATv2
     - Structure + RNA + codon features
     - Protein 3D + RNA base-pair edges
     - Sigmoid output, hybrid loss (α=5)
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
PATIENCE     = 10
HIDDEN       = 32
HEADS        = 2
EDGE_DIM     = 6           
CA_THRESHOLD = 8.0
MIN_SEQ_SEP  = 6
LOSS_ALPHA   = 5.0

CODON_EMBED_DIM = 8        # learned embedding size per codon

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

with open(ISOFORM_JSON) as f:
    isoforms = json.load(f)

# ─── Codon vocabulary ─────────────────────────────────────────────────────────

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]  # 64 codons
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}
UNK_IDX = 64  # index for unknown/missing codons
N_CODON_VOCAB = 65


def codon_to_idx(codon_str):
    if pd.isna(codon_str):
        return UNK_IDX
    c = str(codon_str).upper().strip().replace('U', 'T')
    return CODON_TO_IDX.get(c, UNK_IDX)


# ─── Structure features ───────────────────────────────────────────────────────

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

# 18 struct + 1 ss_class + 1 is_denovo = 20 dense features
# + codon index (integer, looked up in embedding) = 21
N_DENSE_FEATURES = len(STRUCT_COLS) + 1 + 1


def build_node_features(cds, is_denovo=False):
    """Returns (dense_features, codon_indices)."""
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

    dense = np.stack(feats, axis=1).astype(np.float32)
    dense = np.nan_to_num(dense, nan=0.0)

    # Codon indices
    if 'codon' in cds.columns:
        cidx = np.array([codon_to_idx(c) for c in cds['codon']], dtype=np.int64)
    else:
        cidx = np.full(len(cds), UNK_IDX, dtype=np.int64)

    return (torch.tensor(dense, dtype=torch.float32),
            torch.tensor(cidx, dtype=torch.long))


# ─── Edge builders ────────────────────────────────────────────────────────────

def build_seq_edges(n):
    """Extended: edges at distances 1, 3, 5.
    Edge feature: [is_seq, is_prot, is_rna, norm_distance, seq_sep_norm, dist_bucket]
    """
    src, dst, feats = [], [], []
    for dist in [1, 3, 5]:
        for i in range(n - dist):
            j = i + dist
            # forward + backward
            src.extend([i, j])
            dst.extend([j, i])
            # dist_bucket: 1→0.2, 3→0.6, 5→1.0 (encodes which "hop class")
            db = {1: 0.2, 3: 0.6, 5: 1.0}[dist]
            feat = [1, 0, 0, dist / 5.0, dist / n, db]
            feats.extend([feat, feat])
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
                feat = [0, 1, 0, d / CA_THRESHOLD, abs(ri - rj) / n_cds, 0.0]
                feats.extend([feat, feat])
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
        feat = [0, 0, 1, 0.0, abs(ci - cj) / n_codons, 0.0]
        feats.extend([feat, feat])
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
    dense, codon_idx = build_node_features(cds, is_denovo=is_denovo)
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
        all_feats = [[0, 0, 0, 0, 0, 0]] * n

    return Data(
        x=dense,
        codon_idx=codon_idx,
        edge_index=torch.tensor([all_src, all_dst], dtype=torch.long),
        edge_attr=torch.tensor(all_feats, dtype=torch.float32),
        y=y, num_nodes=n, gene=gene, is_denovo=is_denovo,
    )


# ─── Model ─────────────────────────────────────────────────────────────────────

class FoldingGATv2(torch.nn.Module):
    """3-layer GATv2 with learned codon embeddings."""
    def __init__(self, dense_dim=N_DENSE_FEATURES, hid=HIDDEN, heads=HEADS, edge_dim=EDGE_DIM):
        super().__init__()
        self.codon_emb = torch.nn.Embedding(N_CODON_VOCAB, CODON_EMBED_DIM)
        torch.nn.init.xavier_uniform_(self.codon_emb.weight)

        in_dim = dense_dim + CODON_EMBED_DIM

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
        ei, ea = data.edge_index, data.edge_attr

        # Concatenate dense features with learned codon embedding
        codon_e = self.codon_emb(data.codon_idx)
        x = torch.cat([data.x, codon_e], dim=-1)

        x = F.elu(self.norm0(self.proj(x)))
        r = x
        x = self.conv1(x, ei, ea); x = self.drop1(F.elu(self.norm1(x)))
        x = x + self.res1(r); r = x
        x = self.conv2(x, ei, ea); x = self.drop2(F.elu(self.norm2(x)))
        x = x + self.res2(r); r = x
        x = self.conv3(x, ei, ea); x = self.drop3(F.elu(self.norm3(x)))
        x = x + self.res3(r)
        return self.head(x).squeeze(-1)


# ─── Loss & training ──────────────────────────────────────────────────────────

def hybrid_loss(pred, target, alpha=LOSS_ALPHA):
    weights = 1.0 + alpha * target
    mse = (weights * (pred - target) ** 2).mean()
    bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target, reduction='mean')
    return 0.7 * mse + 0.3 * bce

def train_one(model, loader, opt):
    model.train(); tot, n = 0, 0
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
    model.eval(); tot, n = 0, 0; ps, ts, dns = [], [], []
    for b in loader:
        b = b.to(DEVICE); p = model(b)
        tot += hybrid_loss(p, b.y).item() * b.num_nodes; n += b.num_nodes
        ps.append(p.cpu().numpy()); ts.append(b.y.cpu().numpy())
        # is_denovo is last dense feature (index 19)
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
    ax[0,0].plot(tl, label='train'); ax[0,0].plot(vl, label='val'); ax[0,0].legend()
    ax[0,0].set_title(f'Loss ({name})')
    ax[0,1].scatter(vt, vp, alpha=0.05, s=2); ax[0,1].plot([0,1],[0,1],'r--')
    r = stats.pearsonr(vp, vt)[0]
    ax[0,1].set_title(f'r={r:.3f}, r²={r**2:.3f}')
    ax[0,2].hist(vp-vt, bins=50, color='steelblue'); ax[0,2].set_title('Residuals')
    ax[1,0].hist(vt, bins=50, alpha=0.5, color='steelblue', label='true')
    ax[1,0].hist(vp, bins=50, alpha=0.5, color='coral', label='pred'); ax[1,0].legend()
    ranges = [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.0)]
    mses = [np.mean((vp[(vt>=lo)&(vt<hi)]-vt[(vt>=lo)&(vt<hi)])**2) if ((vt>=lo)&(vt<hi)).sum()>0 else 0 for lo,hi in ranges]
    ax[1,1].bar(range(5), mses); ax[1,1].set_xticks(range(5))
    ax[1,1].set_xticklabels([f'{lo}-{hi}' for lo,hi in ranges], fontsize=9)
    ax[1,1].set_title('Error by range')
    if dn is not None:
        wt = dn < 0.5; dnm = dn >= 0.5
        if wt.sum() > 10 and dnm.sum() > 10:
            ax[1,2].scatter(vt[wt], vp[wt], alpha=0.05, s=2, c='steelblue', label='WT')
            ax[1,2].scatter(vt[dnm], vp[dnm], alpha=0.05, s=2, c='coral', label='de novo')
            ax[1,2].plot([0,1],[0,1],'r--'); ax[1,2].legend(fontsize=8, markerscale=5)
    plt.tight_layout(); plt.savefig(OUT_DIR / f'gnn_v10_{name}.png', dpi=150); plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', default='seq+prot+rna',
                        choices=['none', 'seq', 'seq+prot', 'seq+rna', 'seq+prot+rna'])
    args = parser.parse_args()
    ec = args.edges

    print(f"\n{'═'*60}")
    print(f"  GNN v6 — Learned codon embeddings + extended context edges")
    print(f"  Edges: {ec} | Device: {DEVICE}")
    print(f"  Codon embed dim: {CODON_EMBED_DIM} | Seq edges at dist 1,3,5")
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

    print(f"  Total: {len(graphs)} graphs, {sum(g.num_nodes for g in graphs)} nodes")
    print(f"  Dense features: {graphs[0].x.shape[1]} | Edge dim: {graphs[0].edge_attr.shape[1]}")
    print(f"  Example edges: {graphs[0].edge_index.shape[1]}")

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

    model = FoldingGATv2().to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {type(model).__name__} | {n_par:,} params\n")

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
            best_vl = vl; pat_ct = 0
            best_vp, best_vt, best_dn = vp, vt, dn
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_DIR / f"gnn_v10_{ec.replace('+','_')}.pt")
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
    print(f"  RESULTS v9 — {ec}")
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

    v4 = OUT_DIR / f"metrics_v4_{tag}.json"
    if v4.exists():
        p = json.loads(v4.read_text())
        print(f"\n  vs v4: r={p['pearson']:.4f}→{m['pearson']:.4f} ({m['pearson']-p['pearson']:+.4f})")
        print(f"         MSE={p['mse']:.5f}→{m['mse']:.5f}")

    m.update({'edges': ec, 'params': n_par, 'epochs': len(tls),
              'val_genes': sorted(val_genes), 'avg_epoch_time': float(np.mean(etimes))})
    with open(OUT_DIR / f"metrics_v5_{tag}.json", 'w') as f:
        json.dump(m, f, indent=2)


if __name__ == '__main__':
    main()