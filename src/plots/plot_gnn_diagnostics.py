#!/usr/bin/env python3
"""
plot_gnn_v10.py — Standalone evaluation & per-plot export
Loads the best checkpoint and saves each diagnostic as its own high-res PNG.

Usage:
    python plot_gnn_v10.py [--edges seq+prot+rna]
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch.nn import LayerNorm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

# ── pull in helpers from training script ─────────────────────────────────────
import subprocess
from Bio import SeqIO

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

PALETTE = {
    'primary':   '#4C72B0',
    'secondary': '#DD8452',
    'green':     '#55A868',
    'red':       '#C44E52',
    'grey':      '#8C8C8C',
    'bg':        '#F7F7F7',
}

# ── CONFIG (mirror training) ──────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
WT_DIR       = BASE_DIR / "data/ribo_counts"
SIM_DIR      = BASE_DIR / "data/ribo_counts_simulated"
GENES_DIR    = BASE_DIR / "data/genes"
PDB_DIR      = BASE_DIR / "data/pdb"
ISOFORM_JSON = BASE_DIR / "isoform_selection.json"
OUT_DIR      = BASE_DIR / "data/results/plots_v10"

SEED         = 42
VAL_RATIO    = 0.2
HIDDEN       = 32
HEADS        = 2
EDGE_DIM     = 6
CA_THRESHOLD = 8.0
MIN_SEQ_SEP  = 6
LOSS_ALPHA   = 5.0
CODON_EMBED_DIM = 8

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

with open(ISOFORM_JSON) as f:
    isoforms = json.load(f)

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}
UNK_IDX = 64
N_CODON_VOCAB = 65

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
N_DENSE_FEATURES = len(STRUCT_COLS) + 1 + 1   # 20


def codon_to_idx(codon_str):
    if pd.isna(codon_str): return UNK_IDX
    c = str(codon_str).upper().strip().replace('U', 'T')
    return CODON_TO_IDX.get(c, UNK_IDX)


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
    dense = np.stack(feats, axis=1).astype(np.float32)
    dense = np.nan_to_num(dense, nan=0.0)
    if 'codon' in cds.columns:
        cidx = np.array([codon_to_idx(c) for c in cds['codon']], dtype=np.int64)
    else:
        cidx = np.full(len(cds), UNK_IDX, dtype=np.int64)
    return (torch.tensor(dense, dtype=torch.float32),
            torch.tensor(cidx, dtype=torch.long))


def build_seq_edges(n):
    src, dst, feats = [], [], []
    for dist in [1, 3, 5]:
        for i in range(n - dist):
            j = i + dist
            src.extend([i, j]); dst.extend([j, i])
            db = {1: 0.2, 3: 0.6, 5: 1.0}[dist]
            feat = [1, 0, 0, dist / 5.0, dist / n, db]
            feats.extend([feat, feat])
    return src, dst, feats


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
        x=dense, codon_idx=codon_idx,
        edge_index=torch.tensor([all_src, all_dst], dtype=torch.long),
        edge_attr=torch.tensor(all_feats, dtype=torch.float32),
        y=y, num_nodes=n, gene=gene, is_denovo=is_denovo,
    )


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


# ── Model (identical to training) ────────────────────────────────────────────

class FoldingGATv2(torch.nn.Module):
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
        self.res1  = torch.nn.Linear(hid, hid * heads, bias=False)
        self.conv2 = GATv2Conv(hid * heads, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.15, add_self_loops=False)
        self.norm2 = LayerNorm(hid * heads)
        self.drop2 = torch.nn.Dropout(0.3)
        self.res2  = torch.nn.Linear(hid * heads, hid * heads, bias=False)
        self.conv3 = GATv2Conv(hid * heads, hid, heads=heads, edge_dim=edge_dim,
                                dropout=0.15, add_self_loops=False)
        self.norm3 = LayerNorm(hid * heads)
        self.drop3 = torch.nn.Dropout(0.3)
        self.res3  = torch.nn.Linear(hid * heads, hid * heads, bias=False)
        self.head  = torch.nn.Sequential(
            torch.nn.Linear(hid * heads, 32), torch.nn.ELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 1), torch.nn.Sigmoid()
        )

    def forward(self, data):
        ei, ea = data.edge_index, data.edge_attr
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


def hybrid_loss(pred, target, alpha=LOSS_ALPHA):
    weights = 1.0 + alpha * target
    mse = (weights * (pred - target) ** 2).mean()
    bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target, reduction='mean')
    return 0.7 * mse + 0.3 * bce


@torch.no_grad()
def run_eval(model, loader):
    model.eval()
    ps, ts, dns, genes = [], [], [], []
    for b in loader:
        b = b.to(DEVICE)
        p = model(b)
        ps.append(p.cpu().numpy())
        ts.append(b.y.cpu().numpy())
        dns.append(b.x[:, 19].cpu().numpy())
        # replicate gene label per node
        if hasattr(b, 'gene'):
            if isinstance(b.gene, list):
                for g, cnt in zip(b.gene, b.batch.bincount().tolist()):
                    genes.extend([g] * cnt)
            else:
                genes.extend([b.gene] * b.num_nodes)
    return (np.concatenate(ps), np.concatenate(ts),
            np.concatenate(dns), genes)


# ── Individual plot functions ─────────────────────────────────────────────────

def _fig(w=7, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor(PALETTE['bg'])
    fig.patch.set_facecolor('white')
    return fig, ax


def plot_scatter(vp, vt, out):
    """True vs predicted scatter with density colouring."""
    from matplotlib.colors import LogNorm

    fig, ax = _fig(7, 6)
    h = ax.hist2d(vt, vp, bins=80, cmap='Blues', norm=LogNorm(), density=True)
    fig.colorbar(h[3], ax=ax, label='log density')

    r, p_val = stats.pearsonr(vp, vt)
    sp = stats.spearmanr(vp, vt)[0]
    ax.plot([0, 1], [0, 1], '--', color=PALETTE['red'], lw=1.5, label='y = x')

    # regression line
    m_, b_ = np.polyfit(vt, vp, 1)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs, m_ * xs + b_, '-', color=PALETTE['secondary'], lw=1.5,
            label=f'fit  (slope={m_:.2f})')

    ax.set_xlabel('True folding demand')
    ax.set_ylabel('Predicted folding demand')
    ax.set_title(f'Pearson r = {r:.4f}   |   Spearman ρ = {sp:.4f}', pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=11)

    textstr = f'r² = {r**2:.4f}\nn = {len(vt):,}'
    ax.text(0.04, 0.93, textstr, transform=ax.transAxes,
            fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(out / 'scatter_true_vs_pred.png')
    plt.close()
    print(f"  ✓  scatter_true_vs_pred.png")


def plot_residuals(vp, vt, out):
    """Residual histogram."""
    residuals = vp - vt
    fig, ax = _fig(7, 5)
    ax.hist(residuals, bins=60, color=PALETTE['primary'], edgecolor='white',
            linewidth=0.4, alpha=0.9)
    ax.axvline(0, color=PALETTE['red'], lw=1.8, ls='--', label='zero error')
    ax.axvline(np.mean(residuals), color=PALETTE['secondary'], lw=1.5, ls='-',
               label=f'mean = {np.mean(residuals):.4f}')
    ax.set_xlabel('Residual  (predicted − true)')
    ax.set_ylabel('Count')
    ax.set_title('Residual distribution')
    ax.legend(fontsize=11)

    textstr = (f'std  = {np.std(residuals):.4f}\n'
               f'MAE = {np.mean(np.abs(residuals)):.4f}')
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(out / 'residuals.png')
    plt.close()
    print(f"  ✓  residuals.png")


def plot_distributions(vp, vt, out):
    """Overlaid KDE of true and predicted distributions."""
    from scipy.stats import gaussian_kde

    fig, ax = _fig(7, 5)
    xs = np.linspace(0, 1, 300)
    for arr, label, color in [
        (vt, 'True', PALETTE['primary']),
        (vp, 'Predicted', PALETTE['secondary']),
    ]:
        kde = gaussian_kde(arr, bw_method=0.05)
        ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
        ax.plot(xs, kde(xs), color=color, lw=2.2, label=label)

    ax.set_xlabel('Folding demand')
    ax.set_ylabel('Density')
    ax.set_title('Predicted vs True distribution')
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(out / 'distributions.png')
    plt.close()
    print(f"  ✓  distributions.png")


def plot_mse_by_range(vp, vt, out):
    """Bar chart of MSE broken down by true-value bin."""
    ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    labels = [f'[{lo:.1f}, {hi:.1f})' for lo, hi in ranges]
    mses, counts = [], []
    for lo, hi in ranges:
        mask = (vt >= lo) & (vt < hi)
        if mask.sum() > 0:
            mses.append(np.mean((vp[mask] - vt[mask]) ** 2))
            counts.append(mask.sum())
        else:
            mses.append(0); counts.append(0)

    fig, ax1 = _fig(8, 5)
    bars = ax1.bar(range(5), mses, color=PALETTE['primary'],
                   edgecolor='white', linewidth=0.5, width=0.55, zorder=3)
    ax1.set_xticks(range(5)); ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel('MSE', color=PALETTE['primary'])
    ax1.set_title('Prediction error by true-value range')
    ax1.grid(axis='y', alpha=0.3, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(range(5), counts, 'o--', color=PALETTE['secondary'],
             lw=1.8, ms=7, label='# nodes')
    ax2.set_ylabel('Node count', color=PALETTE['secondary'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['secondary'])

    # annotate bars
    for rect, mse in zip(bars, mses):
        ax1.text(rect.get_x() + rect.get_width() / 2,
                 rect.get_height() + max(mses) * 0.02,
                 f'{mse:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(out / 'mse_by_range.png')
    plt.close()
    print(f"  ✓  mse_by_range.png")


def plot_wt_vs_denovo(vp, vt, dn, out):
    """Scatter split by WT vs de novo."""
    wt  = dn < 0.5
    dnm = dn >= 0.5
    if wt.sum() < 10 or dnm.sum() < 10:
        print("  ⚠  Not enough WT/de-novo nodes — skipping wt_vs_denovo.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.patch.set_facecolor('white')

    for ax, mask, label, color in [
        (axes[0], wt,  'Wild-type',  PALETTE['primary']),
        (axes[1], dnm, 'De novo',    PALETTE['secondary']),
    ]:
        ax.set_facecolor(PALETTE['bg'])
        r = stats.pearsonr(vp[mask], vt[mask])[0]
        ax.scatter(vt[mask], vp[mask], alpha=0.07, s=3, color=color, rasterized=True)
        ax.plot([0, 1], [0, 1], '--', color=PALETTE['red'], lw=1.5)
        ax.set_title(f'{label}\nr = {r:.4f}   n = {mask.sum():,}')
        ax.set_xlabel('True folding demand')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    axes[0].set_ylabel('Predicted folding demand')
    plt.suptitle('WT vs De-novo performance', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(out / 'wt_vs_denovo.png')
    plt.close()
    print(f"  ✓  wt_vs_denovo.png")


def plot_per_gene(vp, vt, genes, out):
    """Per-gene Pearson r ranked bar chart."""
    gene_arr = np.array(genes)
    unique = sorted(set(gene_arr))
    rs = []
    for g in unique:
        m = gene_arr == g
        if m.sum() < 5: continue
        r = stats.pearsonr(vp[m], vt[m])[0]
        rs.append((g, r, m.sum()))

    rs.sort(key=lambda x: x[1], reverse=True)
    names = [x[0] for x in rs]
    vals  = [x[1] for x in rs]
    ns    = [x[2] for x in rs]
    colors = [PALETTE['primary'] if v >= 0 else PALETTE['red'] for v in vals]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(PALETTE['bg'])
    bars = ax.bar(range(len(names)), vals, color=colors,
                  edgecolor='white', linewidth=0.4, zorder=3)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Pearson r')
    ax.set_title('Per-gene prediction accuracy (val set)')
    ax.set_ylim(-1, 1)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    # annotate n
    for i, (rect, n_) in enumerate(zip(bars, ns)):
        ax.text(rect.get_x() + rect.get_width() / 2,
                max(rect.get_height(), 0) + 0.02,
                str(n_), ha='center', va='bottom', fontsize=8, color='#444')
    plt.tight_layout()
    fig.savefig(out / 'per_gene_r.png')
    plt.close()
    print(f"  ✓  per_gene_r.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edges', default='seq+prot+rna',
                        choices=['none', 'seq', 'seq+prot', 'seq+rna', 'seq+prot+rna'])
    args = parser.parse_args()
    ec = args.edges
    tag = ec.replace('+', '_')

    ckpt = BASE_DIR / f"data/results/gnn_v10_{tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Loading checkpoint: {ckpt}")
    print(f"  Device: {DEVICE}\n")

    # ── Build val graphs (same split as training) ──────────────────────────
    file_info = discover_files()
    graphs, gene_ids = [], []
    for fi in file_info:
        g = build_graph(pd.read_csv(fi['path']), fi['gene'], ec, fi['is_denovo'])
        if g is not None:
            graphs.append(g); gene_ids.append(fi['gene'])

    unique_genes = sorted(set(gene_ids))
    rng = random.Random(SEED); rng.shuffle(unique_genes)
    n_val = max(1, int(len(unique_genes) * VAL_RATIO))
    val_genes = set(unique_genes[:n_val])
    val_data = [g for g, gid in zip(graphs, gene_ids) if gid in val_genes]
    print(f"  Val genes ({len(val_genes)}): {sorted(val_genes)}")
    print(f"  Val graphs: {len(val_data)}\n")

    val_ld = DataLoader(val_data, batch_size=4)

    # ── Load model ────────────────────────────────────────────────────────
    model = FoldingGATv2().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded — {n_par:,} params\n")

    vp, vt, dn, genes = run_eval(model, val_ld)

    def report_metrics(p, t, label):
        r   = stats.pearsonr(p, t)[0]
        sp  = stats.spearmanr(p, t)[0]
        mse = np.mean((p - t) ** 2)
        mae = np.mean(np.abs(p - t))
        print(f"  {'─'*40}")
        print(f"  {label}  (n = {len(p):,})")
        print(f"  {'─'*40}")
        print(f"    Pearson r  = {r:.4f}  (r² = {r**2:.4f})")
        print(f"    Spearman ρ = {sp:.4f}")
        print(f"    MSE        = {mse:.5f}")
        print(f"    MAE        = {mae:.5f}")

    print()
    report_metrics(vp, vt, "Overall")

    wt_mask  = dn < 0.5
    dn_mask  = dn >= 0.5
    if wt_mask.sum() >= 5:
        report_metrics(vp[wt_mask], vt[wt_mask], "Wild-type")
    else:
        print("  ⚠  Not enough wild-type nodes for separate metrics")

    if dn_mask.sum() >= 5:
        report_metrics(vp[dn_mask], vt[dn_mask], "De novo (GEMORNA)")
    else:
        print("  ⚠  Not enough de novo nodes for separate metrics")

    print(f"\n  Saving plots → {OUT_DIR}\n")

    plot_scatter(vp, vt, OUT_DIR)
    plot_residuals(vp, vt, OUT_DIR)
    plot_distributions(vp, vt, OUT_DIR)
    plot_mse_by_range(vp, vt, OUT_DIR)
    plot_wt_vs_denovo(vp, vt, dn, OUT_DIR)
    if genes:
        plot_per_gene(vp, vt, genes, OUT_DIR)

    print(f"\n  All plots saved to {OUT_DIR}")


if __name__ == '__main__':
    main()