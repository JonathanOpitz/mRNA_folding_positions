#!/usr/bin/env python3
"""Run v5 ablation over all edge configs and compare results across model versions."""

import subprocess
import json
import sys
from pathlib import Path

CONFIGS = ['seq+prot+rna', 'seq+prot', 'seq+rna']
OUT = Path(__file__).resolve().parents[2] / "data/results"

for i, c in enumerate(CONFIGS, 1):
    print(f"\n{'═'*60}")
    print(f"  [{i}/{len(CONFIGS)}] v5 --edges {c}")
    print(f"{'═'*60}")
    subprocess.run(
        [sys.executable, "train_gnn_v5.py", "--edges", c],
        cwd=str(Path(__file__).parent),
    )

print(f"\n{'═'*90}")
print(f"  ALL VERSIONS COMPARISON")
print(f"{'═'*90}")
print(f"  {'Config':<18} {'v1 r':>6} {'v2 r':>6} {'v3 r':>6} {'v5 r':>6}  "
      f"{'v5 r²':>6} {'v5 pred_max':>11} {'v5 MSE@0.7+':>11}")
print(f"  {'─'*85}")

for c in CONFIGS:
    tag = c.replace('+', '_')
    rs = {}
    for ver, prefix in [('v1', ''), ('v2', 'v2_'), ('v3', 'v3_'), ('v5', 'v5_')]:
        f = OUT / f"metrics_{prefix}{tag}.json"
        if f.exists():
            rs[ver] = json.loads(f.read_text())

    v1r  = rs.get('v1', {}).get('pearson', 0)
    v2r  = rs.get('v2', {}).get('pearson', 0)
    v3r  = rs.get('v3', {}).get('pearson', 0)
    v5   = rs.get('v5', {})
    v5r  = v5.get('pearson', 0)
    v5pm = v5.get('pred_max', 0)
    v5mh = v5.get('mse_0.7_1.0', 0)

    print(f"  {c:<18} {v1r:>6.3f} {v2r:>6.3f} {v3r:>6.3f} {v5r:>6.3f}  "
          f"{v5r**2:>6.3f} {v5pm:>11.3f} {v5mh:>11.5f}")

print(f"\n  v1: uniform MSE, WT only")
print(f"  v2: weighted MSE (w=1+3*fd), WT only")
print(f"  v3: sigmoid + weighted MSE/BCE + WT & de novo")
print(f"  v5: learned codon embeddings + extended seq edges (±1,3,5) — current production")
