#!/usr/bin/env python3
"""
plot_tai_comparison_ACTB_CTSB.py

Two-panel comparison: TAI profiles (γ=0.0 vs γ=0.5) for ACTB and CTSB.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ─── FONT SETTINGS ───────────────────────────────────────────────────────────
FONT_BASE    = 16   
FONT_LABEL   = 16   
FONT_TITLE   = 17   
FONT_SUPTITLE = 19  
FONT_ANNOT   = 11   

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "data/analysis/tai_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENES = {
    "ACTB": {
        "g0":  BASE / "data/optimized/ACTB_fold000_mfe070_HEK293T_ep5_results.json",
        "g05": BASE / "data/optimized/ACTB_fold050_mfe070_HEK293T_ep5_results.json",
    },
    "CTSB": {
        "g0":  BASE / "data/optimized/CTSB_fold000_mfe070_HEK293T_ep5_results.json",
        "g05": BASE / "data/optimized/CTSB_fold050_mfe070_HEK293T_ep5_results.json",
    },
}

SMOOTH_WIN = 15

# ─── TAI TABLE ────────────────────────────────────────────────────────────────
TAI = {
    'TTT':0.42,'TTC':1.00,'TTA':0.08,'TTG':0.42, 'CTT':0.42,'CTC':0.58,'CTA':0.08,'CTG':1.00,
    'ATT':0.58,'ATC':1.00,'ATA':0.08,'ATG':1.00, 'GTT':0.42,'GTC':0.58,'GTA':0.08,'GTG':1.00,
    'TCT':0.58,'TCC':0.75,'TCA':0.25,'TCG':0.17, 'CCT':0.58,'CCC':0.75,'CCA':0.42,'CCG':0.17,
    'ACT':0.58,'ACC':1.00,'ACA':0.42,'ACG':0.17, 'GCT':0.75,'GCC':1.00,'GCA':0.42,'GCG':0.17,
    'TAT':0.42,'TAC':1.00,'TAA':0.00,'TAG':0.00, 'CAT':0.42,'CAC':1.00,'CAA':0.42,'CAG':1.00,
    'AAT':0.42,'AAC':1.00,'AAA':0.42,'AAG':1.00, 'GAT':0.42,'GAC':1.00,'GAA':0.42,'GAG':1.00,
    'TGT':0.42,'TGC':1.00,'TGA':0.00,'TGG':1.00, 'CGT':0.42,'CGC':0.75,'CGA':0.08,'CGG':0.17,
    'AGT':0.25,'AGC':0.75,'AGA':0.42,'AGG':0.25, 'GGT':0.42,'GGC':1.00,'GGA':0.25,'GGG':0.17,
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_optimized_seq(path):
    with open(path) as f:
        d = json.load(f)
    n_ep = d["config"]["n_epochs"]
    best = d["sequence"].get("best_seq", {}) or d["metrics_per_epoch"].get("best_seq", {})
    return best.get(str(n_ep), d["sequence"]["cds_only"])

def tai_profile(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return np.array([TAI.get(c, 0.5) for c in codons])

def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

# ─── LOAD ─────────────────────────────────────────────────────────────────────
data = {}
for gene, paths in GENES.items():
    seq_g0  = load_optimized_seq(paths["g0"])
    seq_g05 = load_optimized_seq(paths["g05"])
    data[gene] = {
        "tai_g0":  tai_profile(seq_g0),
        "tai_g05": tai_profile(seq_g05),
    }
    print(f"{gene}: {len(data[gene]['tai_g0'])} codons | "
          f"mean TAI γ=0.0={data[gene]['tai_g0'].mean():.3f}, "
          f"γ=0.5={data[gene]['tai_g05'].mean():.3f}")

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9.5), sharey=True)

for ax, (gene, d) in zip(axes, data.items()):
    n = len(d["tai_g0"])
    pos = np.arange(n)
    
    ax.plot(pos, smooth(d["tai_g0"], SMOOTH_WIN),
            label=f"γ=0.0 (mean={d['tai_g0'].mean():.2f})",
            color='#1f77b4', lw=1.5)
    
    ax.plot(pos, smooth(d["tai_g05"], SMOOTH_WIN),
            label=f"γ=0.5 (mean={d['tai_g05'].mean():.2f})",
            color='#d62728', lw=1.5)
    
    ax.set_ylabel("TAI (15-codon running mean)", fontsize=FONT_LABEL)
    ax.set_title(f"{gene} — TAI profile along CDS", fontsize=FONT_TITLE, pad=15)
    
    ax.grid(alpha=0.3)
    ax.set_xlim(0, n)
    ax.tick_params(axis='both', which='major', labelsize=FONT_BASE)
    
    # Per-panel legend below each subplot
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.18),
              fontsize=FONT_BASE - 1,
              ncol=2,
              frameon=True,
              edgecolor='black')

# Subplot spacing
plt.subplots_adjust(hspace=0.65, bottom=0.12, top=0.90)

axes[-1].set_xlabel("Codon position", fontsize=FONT_LABEL)

fig.suptitle("ACTB vs CTSB — TAI profiles (γ=0.0 vs γ=0.5)", 
             fontsize=FONT_SUPTITLE, y=0.97)

out_path = OUT_DIR / "ACTB_vs_CTSB_tai_profile.png"
plt.savefig(out_path, dpi=160, bbox_inches='tight')
plt.close()

print(f"\n[Saved] {out_path}")