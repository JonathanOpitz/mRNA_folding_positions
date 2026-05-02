#!/usr/bin/env python3
"""Run all 5 ablation configs and print comparison table."""

import subprocess, json, sys
from pathlib import Path

CONFIGS = ['none', 'seq', 'seq+prot', 'seq+rna', 'seq+prot+rna']
OUT = Path("/Users/jonathanopitz/Desktop/Master/data/results")

for i, c in enumerate(CONFIGS, 1):
    print(f"\n{'═'*60}")
    print(f"  [{i}/{len(CONFIGS)}] --edges {c}")
    print(f"{'═'*60}")
    subprocess.run([sys.executable, "train_gnn.py", "--edges", c],
                   cwd=str(Path(__file__).parent))

print(f"\n{'═'*70}")
print(f"  ABLATION COMPARISON")
print(f"{'═'*70}")
print(f"  {'Config':<18} {'MSE':>8} {'MAE':>8} {'Pearson':>8} {'Spearman':>9} {'Params':>8}")
print(f"  {'─'*63}")

for c in CONFIGS:
    f = OUT / f"metrics_{c.replace('+','_')}.json"
    if f.exists():
        m = json.loads(f.read_text())
        print(f"  {c:<18} {m['mse']:8.5f} {m['mae']:8.5f} "
              f"{m['pearson']:8.4f} {m['spearman']:9.4f} {m['params']:8,}")
    else:
        print(f"  {c:<18} {'—':>8} {'—':>8} {'—':>8} {'—':>9} {'—':>8}")

print(f"\n  ✓ = graph helps if seq > none")
print(f"  ✓ = protein structure helps if seq+prot > seq")
print(f"  ✓ = RNA structure helps if seq+rna > seq")