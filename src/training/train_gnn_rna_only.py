#!/usr/bin/env python3
"""
GNN Training — RNA-only graph (final stable version)
- 2-head architecture with residual connections
- Stabilised training (LR scheduler without 'verbose')
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch.nn import LayerNorm
import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
from Bio import SeqIO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
_BASE        = Path(__file__).resolve().parents[2]
DATA_DIR     = _BASE / "data/ribo_counts"
GENES_DIR    = _BASE / "data/genes"
ISOFORM_JSON = _BASE / "isoform_selection.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TRAIN_RATIO = 0.8
EPOCHS = 200
BATCH_SIZE = 6
LR = 0.0001

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── Load isoforms ─────────────────────────────────────────────────────────────
with open(ISOFORM_JSON, 'r') as f:
    isoforms = json.load(f)

# ─── 25 NODE FEATURES ─────────────────────────────────────────────────────────
def create_node_features(df: pd.DataFrame) -> torch.Tensor:
    features = [
        df['plddt'].fillna(50).values / 100.0,
        df['contact_density'].fillna(0).values,
        df['domain_boundary'].fillna(0).values,
        df['domain_boundary_interpro'].fillna(0).values,
        df['domain_boundary_pae'].fillna(0).values,
        df['ss_H'].fillna(0).values,
        df['ss_E'].fillna(0).values,
        df['ss_C'].fillna(0).values,
        df['rna_local_paired_prob'].fillna(0.5).values,
        df['rna_unpaired_1nt'].fillna(0.7).values,
        df['rna_unpaired_2nt'].fillna(0.6).values,
        df['rna_unpaired_3nt'].fillna(0.5).values,
        df['rna_unpaired_4nt'].fillna(0.4).values,
        df['rna_unpaired_5nt'].fillna(0.3).values,
        df['rna_opening_energy_1nt'].fillna(0.0).values,
        df['rna_opening_energy_3nt'].fillna(0.0).values,
        df['rna_opening_energy_5nt'].fillna(0.0).values,
        df['rna_paired_prob_window5cod'].fillna(0.5).values,
        df['rna_paired_prob_window10cod'].fillna(0.5).values,
        df['rna_struct_change'].fillna(0.0).values,
        df['rna_local_ss_class'].map({'stem': 1.0, 'weak': 0.5, 'open_loop': 0.0}).fillna(0.0).values,
        df.get('fd_contact_complexity', 0).fillna(0).values,
        df.get('fd_domain_transition', 0).fillna(0).values,
        df.get('fd_ss_transition', 0).fillna(0).values,
        df.get('fd_plddt_weight', df['plddt'].fillna(50)/100).fillna(0).values,
    ]
    x = np.stack(features, axis=1)
    return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32)


# ─── FASTA Loader ─────────────────────────────────────────────────────────────
def find_best_fasta(gene: str) -> Path | None:
    gene = gene.upper()
    if gene not in isoforms or isoforms[gene].get("status") != "ok":
        return None
    info = isoforms[gene]
    best = info.get("best_isoform_full") or info.get("best_isoform")
    candidates = list(GENES_DIR.glob(f"*{gene}*{best}*.fasta")) + list(GENES_DIR.glob(f"*{gene}*.fasta"))
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def get_full_sequence(fasta_path: Path) -> str:
    record = next(SeqIO.parse(fasta_path, "fasta"))
    return str(record.seq).upper().replace("T", "U")


# ─── RNA Base Pairs via RNAfold ───────────────────────────────────────────────
def get_rna_base_pairs(nt_sequence: str):
    try:
        result = subprocess.run(["RNAfold", "--noPS"], input=nt_sequence + "\n",
                                capture_output=True, text=True, timeout=8)
        if result.returncode != 0:
            return []
        dotbracket = result.stdout.strip().split('\n')[-1].split()[0]
        stack = []
        pairs = []
        for i, char in enumerate(dotbracket):
            if char in '({[':
                stack.append(i)
            elif char in ')}]':
                if stack:
                    pairs.append((stack.pop(), i))
        return pairs
    except:
        return []


def extract_rna_edges(df: pd.DataFrame, nt_sequence: str = None):
    n = len(df)
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])

    if nt_sequence and len(nt_sequence) > 100:
        base_pairs = get_rna_base_pairs(nt_sequence)
        for i_nt, j_nt in base_pairs:
            ci = i_nt // 3
            cj = j_nt // 3
            if ci != cj and 0 <= ci < n and 0 <= cj < n and abs(ci - cj) > 2:
                edges.extend([[ci, cj], [cj, ci]])

    return torch.tensor(edges, dtype=torch.long).T


# ─── Build Graph ──────────────────────────────────────────────────────────────
def build_graph(df: pd.DataFrame, gene_name: str):
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    if len(cds) < 20:
        return None

    fasta_path = find_best_fasta(gene_name)
    nt_sequence = get_full_sequence(fasta_path) if fasta_path else None

    x = create_node_features(cds)
    edge_index = extract_rna_edges(cds, nt_sequence)

    occ = cds['rel_occupancy'].fillna(1e-8).values
    y = torch.log(torch.tensor(occ, dtype=torch.float32) + 1e-8).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, y=y, gene=gene_name)


# ─── Model – Deine exakte Architektur mit 2 Heads ─────────────────────────────
class FoldingGNN(torch.nn.Module):
    def __init__(self, in_features=25, hidden=48):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_features, hidden)
        self.input_norm = LayerNorm(hidden)

        self.conv1 = GATv2Conv(hidden, hidden, heads=2, dropout=0.2)
        self.norm1 = LayerNorm(hidden * 2)
        self.dropout1 = torch.nn.Dropout(0.4)
        self.res_proj1 = torch.nn.Linear(hidden, hidden * 2, bias=False)

        self.conv2 = GATv2Conv(hidden * 2, hidden, heads=2, dropout=0.2)
        self.norm2 = LayerNorm(hidden * 2)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.res_proj2 = torch.nn.Linear(hidden * 2, hidden * 2, bias=False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, 32),
            torch.nn.ELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        x = self.input_norm(x)
        x = F.elu(x)
        residual = x.clone()

        # Block 1
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        residual = self.res_proj1(residual)
        x = x + residual
        residual = x.clone()

        # Block 2
        x = self.conv2(x, data.edge_index)
        x = F.elu(x)
        x = self.dropout2(x)
        x = self.norm2(x)
        x = x + self.res_proj2(residual)

        x = x.mean(dim=0, keepdim=True)
        return self.mlp(x)


# ─── Training mit Scheduler ───────────────────────────────────────────────────
def main():
    files = sorted(DATA_DIR.glob("*_with_folddemand.csv"))
    graphs = []
    for f in files:
        gene = f.stem.split("_with_folddemand")[0].upper()
        df = pd.read_csv(f)
        graph = build_graph(df, gene)
        if graph is not None:
            graphs.append(graph)

    random.shuffle(graphs)
    split = int(len(graphs) * TRAIN_RATIO)
    train_graphs = graphs[:split]
    test_graphs = graphs[split:]

    print(f"Loaded {len(graphs)} genes → Train: {len(train_graphs)} | Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = FoldingGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )

    best_test_loss = float('inf')
    patience_counter = 0
    patience = 30

    print("Training gestartet (stabilisiert)...\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            target = batch.y.mean(dim=0, keepdim=True)
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                target = batch.y.mean(dim=0, keepdim=True)
                test_loss += criterion(out, target).item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), "gnn_rna_only_best.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.5f} | "
                  f"Test MSE: {test_loss:.5f} | Best: {best_test_loss:.5f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= patience:
            print(f"\nEarly Stopping nach Epoch {epoch}")
            break

    print(f"\nTraining beendet! Best Test MSE: {best_test_loss:.5f}")
    print("Modell gespeichert als: gnn_rna_only_best.pt")


if __name__ == "__main__":
    main()