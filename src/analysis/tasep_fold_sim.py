#!/usr/bin/env python3
"""
Real stochastic TASEP co-translational folding simulation.

This is a MECHANISTIC simulation of ribosome translation. Unlike the
simple "1/TAI" proxy, TASEP models the mRNA as a 1D lattice where:

  - Multiple ribosomes move simultaneously
  - Each codon has a RATE constant (derived from TAI)
  - Ribosomes are RIGID (footprint = 10 codons; no overlap)
  - Initiation happens at a fixed rate at position 0
  - → SLOW CODONS CAUSE TRAFFIC JAMS (real biology!)

At each simulation step we track three INDEPENDENT folding risks (same
biological rationale as before, but now using REAL dwell times that
include queueing effects):

  1. HYDROPHOBIC-EXPOSURE RISK
     A hydrophobic stretch just emerged from the tunnel and the next
     codon is moving FAST → stretch is exposed before it can bury.

  2. ORPHAN-CYSTEINE RISK
     Cys emerged but its partner Cys is still being synthesized →
     vulnerable to wrong disulfide bonds.

  3. DOMAIN-CROWDING RISK
     Domain boundary just passed but the previous domain hasn't had
     time to fold → domain interference.

Output
══════
  Three per-position risk trajectories for each of γ=0 vs γ=0.3.
  These are INDEPENDENT outputs, not combined — you can see which
  biological principle each sequence handles best.

References
══════════
  Original TASEP for translation:
    MacDonald, Gibbs & Pipkin (1968). Biopolymers 6: 1-25.
  Codon-level TASEP with tRNA abundances:
    Shaw et al (2003). Biophys J 85: 3512-3525.
  Co-translational folding connection:
    Liutkute et al (2020). FEBS Letters 594: 4287-4305.

Usage
═════
    python tasep_fold_sim.py \
        data/optimized/ACTB_fold000_mfe000_HEK293T_ep1_results.json \
        data/optimized/ACTB_fold030_mfe000_HEK293T_ep1_results.json \
        --n_simulations 20 --plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CODON_TO_AA = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

CODON_TAI = {
    'TTT':0.42,'TTC':1.00,'TTA':0.08,'TTG':0.42,'CTT':0.42,'CTC':0.58,'CTA':0.08,'CTG':1.00,
    'ATT':0.58,'ATC':1.00,'ATA':0.08,'ATG':1.00,'GTT':0.42,'GTC':0.58,'GTA':0.08,'GTG':1.00,
    'TCT':0.58,'TCC':0.75,'TCA':0.25,'TCG':0.17,'CCT':0.58,'CCC':0.75,'CCA':0.42,'CCG':0.17,
    'ACT':0.58,'ACC':1.00,'ACA':0.42,'ACG':0.17,'GCT':0.75,'GCC':1.00,'GCA':0.42,'GCG':0.17,
    'TAT':0.42,'TAC':1.00,'TAA':0.00,'TAG':0.00,'CAT':0.42,'CAC':1.00,'CAA':0.42,'CAG':1.00,
    'AAT':0.42,'AAC':1.00,'AAA':0.42,'AAG':1.00,'GAT':0.42,'GAC':1.00,'GAA':0.42,'GAG':1.00,
    'TGT':0.42,'TGC':1.00,'TGA':0.00,'TGG':1.00,'CGT':0.42,'CGC':0.75,'CGA':0.08,'CGG':0.17,
    'AGT':0.25,'AGC':0.75,'AGA':0.42,'AGG':0.25,'GGT':0.42,'GGC':1.00,'GGA':0.25,'GGG':0.17,
}

# Kyte-Doolittle hydrophobicity scale
HYDRO_KD = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
    '*':  0.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# TASEP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

RIBOSOME_FOOTPRINT = 10       # codons occupied by one ribosome (~30 nt, classical TASEP)
TUNNEL_LEN         = 30       # ~30 AAs in ribosome exit tunnel before emerging
K_INIT             = 0.2      # initiation rate (arbitrary units; calibrated against elongation)
K_ELONG_MAX        = 10.0     # max elongation rate (codons/unit time) when TAI=1
HYDRO_WINDOW       = 9        # AA window for hydrophobicity averaging
DOMAIN_CONSOL_WINDOW = 50     # codons after domain boundary where "still folding"
N_SIMULATIONS_DEFAULT = 10    # averages per sequence (stochastic → need reps)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def seq_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def translate(seq):
    return ''.join(CODON_TO_AA.get(c, '?') for c in seq_to_codons(seq))


def elongation_rates(seq):
    """Per-codon elongation rate = K_ELONG_MAX * TAI.
    High TAI → fast; low TAI → slow."""
    tai = np.array([CODON_TAI.get(c, 0.5) for c in seq_to_codons(seq)])
    return K_ELONG_MAX * np.maximum(tai, 0.05)  # floor to avoid stalls


def load_run(path):
    with open(path) as f:
        data = json.load(f)
    return dict(
        gene  = data['gene'],
        gamma = data['config']['fold_gamma'],
        seq   = data['sequence']['cds_only'],
    )


def load_domain_boundaries(gene, wt_dir, n_expected):
    gene_u = gene.upper()
    candidates = sorted(p for p in wt_dir.glob("*_with_folddemand.csv")
                        if gene_u in p.name.upper())
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    cds = df[df['region'] == 'CDS'].reset_index(drop=True)
    if 'domain_boundary' not in cds.columns:
        return None
    arr = pd.to_numeric(cds['domain_boundary'], errors='coerce').fillna(0).values
    if len(arr) != n_expected:
        arr = arr[:n_expected] if len(arr) > n_expected \
              else np.concatenate([arr, np.zeros(n_expected - len(arr))])
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# CORE TASEP SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_tasep(seq, sim_time=500.0, seed=42):
    """
    Stochastic TASEP simulation using the Gillespie algorithm.

    State: list of ribosome positions (integers, ordered left-to-right).
    Events possible at each step:
      - INITIATION: place a new ribosome at position 0
                    (only if position 0..FOOTPRINT is free)
      - ELONGATION: move ribosome i from p to p+1
                    (only if position p+FOOTPRINT is free)

    Gillespie algorithm picks the next event with probability proportional
    to its rate, and advances time by Δt ~ Exp(1/total_rate).

    Returns:
      dwell_time[pos]  = total time the ribosome A-site spent at position pos,
                         averaged over all ribosomes that visited that position
      n_visits[pos]    = number of ribosomes that reached position pos
      position_history = list of (time, [ribosome_positions]) snapshots
    """
    rng = np.random.default_rng(seed)
    k_elong = elongation_rates(seq)
    n_codons = len(k_elong)

    ribosomes = []              # sorted list of ribosome positions (A-site codon index)
    t = 0.0
    dwell_time = np.zeros(n_codons)
    n_visits   = np.zeros(n_codons, dtype=int)
    last_event_time_per_pos = {}  # (ribosome_id, pos) → time arrived

    # Track ribosomes by unique ID (so we can measure per-ribosome dwell time)
    next_rib_id = 0
    rib_ids_at_pos = []  # parallel to `ribosomes`, holds the rib_id at each
    rib_arrival_time = {}  # rib_id → time it arrived at current position

    while t < sim_time:
        # ── Build event list ──────────────────────────────────────────────
        events = []            # list of (rate, type, ribosome_index_or_None)
        total_rate = 0.0

        # INITIATION: only if no ribosome within RIBOSOME_FOOTPRINT of 0
        if not ribosomes or ribosomes[0] >= RIBOSOME_FOOTPRINT:
            events.append((K_INIT, 'init', None))
            total_rate += K_INIT

        # ELONGATION: for each ribosome, can it move?
        for i, pos in enumerate(ribosomes):
            if pos + 1 >= n_codons:
                # Ribosome at end → terminates; handled in event loop
                events.append((k_elong[pos], 'term', i))
                total_rate += k_elong[pos]
            else:
                # Check no ribosome ahead within footprint
                if i + 1 < len(ribosomes) and ribosomes[i+1] - pos <= RIBOSOME_FOOTPRINT:
                    continue  # blocked
                events.append((k_elong[pos], 'elong', i))
                total_rate += k_elong[pos]

        if total_rate == 0:
            break  # deadlock (shouldn't happen with init>0)

        # ── Gillespie: time to next event, which event ──────────────────
        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t >= sim_time:
            break

        # Record that all ribosomes accumulated dwell time of `dt`
        # at their current position during this Δt
        for i, pos in enumerate(ribosomes):
            dwell_time[pos] += dt

        # Pick which event fires
        u = rng.uniform(0, total_rate)
        cumulative = 0.0
        chosen = None
        for rate, typ, idx in events:
            cumulative += rate
            if u <= cumulative:
                chosen = (typ, idx)
                break

        if chosen is None:
            chosen = (events[-1][1], events[-1][2])

        typ, idx = chosen
        if typ == 'init':
            ribosomes.insert(0, 0)
            rib_ids_at_pos.insert(0, next_rib_id)
            rib_arrival_time[next_rib_id] = t
            n_visits[0] += 1
            next_rib_id += 1
        elif typ == 'elong':
            ribosomes[idx] += 1
            pos = ribosomes[idx]
            n_visits[pos] += 1
        elif typ == 'term':
            # Ribosome reaches last codon → remove
            del ribosomes[idx]
            del rib_ids_at_pos[idx]

    # Normalize dwell time per visit (so we get "time per passage", not total accumulated)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_dwell = np.where(n_visits > 0, dwell_time / np.maximum(n_visits, 1), 0)

    return dict(
        mean_dwell_time = mean_dwell,    # time a single ribosome spends at each position
        total_dwell     = dwell_time,    # summed over all ribosomes (throughput-weighted)
        n_visits        = n_visits,
    )


def run_multiple_tasep(seq, n_sims=10, sim_time=500.0, base_seed=42):
    """Run TASEP multiple times with different seeds; average dwell times."""
    all_dwell = []
    all_visits = []
    for i in range(n_sims):
        res = simulate_tasep(seq, sim_time=sim_time, seed=base_seed + i)
        all_dwell.append(res['mean_dwell_time'])
        all_visits.append(res['n_visits'])
    dwell_arr = np.stack(all_dwell, axis=0)
    visits_arr = np.stack(all_visits, axis=0)
    return dict(
        mean_dwell      = dwell_arr.mean(axis=0),
        std_dwell       = dwell_arr.std(axis=0),
        mean_visits     = visits_arr.mean(axis=0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# RISK TRAJECTORIES (from TASEP dwell times)
# ═══════════════════════════════════════════════════════════════════════════

def hydrophobicity_per_position(aa_seq, window=HYDRO_WINDOW):
    """Running mean of KD hydrophobicity over AA window ending at each pos."""
    scores = np.array([HYDRO_KD.get(a, 0) for a in aa_seq])
    out = np.zeros_like(scores)
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        out[i] = scores[start:i+1].mean()
    return out


def risk_hydrophobic(seq, mean_dwell):
    """
    Hydrophobic-exposure risk per position:
      risk[t] = max(0, hydrophobicity[t]) * (1 / dwell_time[t])

    High risk = hydrophobic stretch JUST emerged AND ribosome moving fast there.
    Zero for negative hydrophobicity (hydrophilic regions — no aggregation risk).
    """
    aa = translate(seq)
    hydro = hydrophobicity_per_position(aa)
    # Only positions after tunnel emergence contribute
    risk = np.zeros(len(hydro))
    for t in range(TUNNEL_LEN, len(hydro)):
        h = max(0, hydro[t])
        d = mean_dwell[t] if mean_dwell[t] > 1e-6 else 1e-6
        risk[t] = h * (1.0 / d)
    return risk


def risk_orphan_cys(seq, mean_dwell):
    """
    Orphan-Cys risk per position:
      risk[t] = n_orphan_Cys_at_time_t * (1 / dwell_time[t])

    Orphan = Cys has emerged from tunnel (position <= t-TUNNEL_LEN)
             but its pairing partner hasn't emerged yet.

    We use greedy Cys pairing (cys[0]↔cys[1], cys[2]↔cys[3], ...).
    This is approximate — real pairing requires PDB data.
    """
    aa = translate(seq)
    n = len(aa)
    cys_positions = [i for i, a in enumerate(aa) if a == 'C']
    # Build pairing dict
    pairs = {}
    for i in range(0, len(cys_positions) - 1, 2):
        a, b = cys_positions[i], cys_positions[i+1]
        pairs[a] = b; pairs[b] = a
    if len(cys_positions) % 2 == 1:
        pairs[cys_positions[-1]] = None

    risk = np.zeros(n)
    for t in range(TUNNEL_LEN, n):
        emerged_limit = t - TUNNEL_LEN
        n_orphan = 0
        for c in cys_positions:
            if c > emerged_limit:
                continue
            partner = pairs.get(c)
            if partner is None or partner > t:
                n_orphan += 1
        d = mean_dwell[t] if mean_dwell[t] > 1e-6 else 1e-6
        risk[t] = n_orphan * (1.0 / d)
    return risk


def risk_domain_crowding(mean_dwell, domain_arr, window=DOMAIN_CONSOL_WINDOW):
    """
    Domain-crowding risk per position:
      risk[t] = max(domain_signal[t-window:t]) * (1 / dwell_time[t])

    High risk = just passed a domain boundary AND moving fast.
    """
    n = min(len(mean_dwell), len(domain_arr))
    risk = np.zeros(n)
    for t in range(TUNNEL_LEN, n):
        start = max(0, t - window)
        recent = domain_arr[start:t+1].max() if start <= t else 0
        d = mean_dwell[t] if mean_dwell[t] > 1e-6 else 1e-6
        risk[t] = recent * (1.0 / d)
    return risk


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def summarize(risk, label):
    active = risk[risk > 0]
    return dict(
        label   = label,
        auc     = float(risk.sum()),
        peak    = float(risk.max()),
        mean_a  = float(active.mean()) if len(active) else 0.0,
    )


def print_report(gene, runA, runB, hA, hB, cA, cB, dA, dB,
                 tasA_visits, tasB_visits, n_sims):
    print()
    print("═" * 78)
    print(f"  Stochastic TASEP Co-translational Folding Simulation: {gene}")
    print(f"    Run A: γ={runA['gamma']}   Run B: γ={runB['gamma']}")
    print(f"    Averaged over {n_sims} independent simulations each")
    print("═" * 78)

    # Basic TASEP diagnostics
    print()
    print("┌─ TASEP DIAGNOSTICS ────────────────────────────────────────────────────────┐")
    print(f"  Ribosomes that reached the end:")
    print(f"    Run A: {tasA_visits[-1]:.1f} (avg)   Run B: {tasB_visits[-1]:.1f} (avg)")
    print(f"  → Throughput of protein synthesis")

    print()
    print("Interpretation:  Lower AUC = ribosome handled dangerous moments more carefully.")
    print("                 Each metric is INDEPENDENT — tells a different story.")

    print()
    print("┌─ 1. HYDROPHOBIC-EXPOSURE RISK ─────────────────────────────────────────────┐")
    sA = summarize(hA, 'A'); sB = summarize(hB, 'B')
    print(f"  Run A (γ={runA['gamma']}):  AUC={sA['auc']:.1f}   peak={sA['peak']:.2f}")
    print(f"  Run B (γ={runB['gamma']}):  AUC={sB['auc']:.1f}   peak={sB['peak']:.2f}")
    d_auc = sB['auc'] - sA['auc']
    verdict = "← B is SAFER (slower at hydrophobic blocks)" if d_auc < 0 else "← B is riskier or equal"
    print(f"  Δ AUC:        {d_auc:+.1f}   {verdict}")

    print()
    print("┌─ 2. ORPHAN-CYSTEINE RISK ──────────────────────────────────────────────────┐")
    sA = summarize(cA, 'A'); sB = summarize(cB, 'B')
    print(f"  Run A (γ={runA['gamma']}):  AUC={sA['auc']:.1f}   peak={sA['peak']:.2f}")
    print(f"  Run B (γ={runB['gamma']}):  AUC={sB['auc']:.1f}   peak={sB['peak']:.2f}")
    d_auc = sB['auc'] - sA['auc']
    verdict = "← B is SAFER (slower at orphan Cys)" if d_auc < 0 else "← B is riskier or equal"
    print(f"  Δ AUC:        {d_auc:+.1f}   {verdict}")

    print()
    print("┌─ 3. DOMAIN-CROWDING RISK ──────────────────────────────────────────────────┐")
    sA = summarize(dA, 'A'); sB = summarize(dB, 'B')
    if sA['auc'] == 0 and sB['auc'] == 0:
        print("  (No domain boundaries known for this gene.)")
    else:
        print(f"  Run A (γ={runA['gamma']}):  AUC={sA['auc']:.1f}   peak={sA['peak']:.2f}")
        print(f"  Run B (γ={runB['gamma']}):  AUC={sB['auc']:.1f}   peak={sB['peak']:.2f}")
        d_auc = sB['auc'] - sA['auc']
        verdict = "← B is SAFER (slower near domain boundaries)" if d_auc < 0 else "← B is riskier or equal"
        print(f"  Δ AUC:        {d_auc:+.1f}   {verdict}")
    print()


def plot_trajectories(gene, runA, runB, hA, hB, cA, cB, dA, dB,
                      dwellA, dwellB, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    positions = np.arange(len(hA))

    # Panel 0: Dwell time profile (the underlying TASEP output)
    axes[0].plot(positions[:len(dwellA)], dwellA, color='steelblue', alpha=0.8,
                 label=f"γ={runA['gamma']}")
    axes[0].plot(positions[:len(dwellB)], dwellB, color='coral', alpha=0.8,
                 label=f"γ={runB['gamma']}")
    axes[0].set_ylabel('Dwell time per codon\n(TASEP output)')
    axes[0].set_title(f"{gene} — Stochastic TASEP folding simulation")
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    # Panel 1: Hydrophobic risk
    axes[1].plot(positions, hA, color='steelblue', alpha=0.8)
    axes[1].plot(positions, hB, color='coral', alpha=0.8)
    axes[1].fill_between(positions, 0, hA, color='steelblue', alpha=0.15)
    axes[1].fill_between(positions, 0, hB, color='coral', alpha=0.15)
    axes[1].set_ylabel('Hydrophobic\nexposure risk')
    axes[1].grid(alpha=0.3)

    # Panel 2: Cys risk
    axes[2].plot(positions[:len(cA)], cA, color='steelblue', alpha=0.8)
    axes[2].plot(positions[:len(cB)], cB, color='coral', alpha=0.8)
    axes[2].fill_between(positions[:len(cA)], 0, cA, color='steelblue', alpha=0.15)
    axes[2].fill_between(positions[:len(cB)], 0, cB, color='coral', alpha=0.15)
    axes[2].set_ylabel('Orphan-Cys\nrisk')
    axes[2].grid(alpha=0.3)

    # Panel 3: Domain risk
    axes[3].plot(positions[:len(dA)], dA, color='steelblue', alpha=0.8)
    axes[3].plot(positions[:len(dB)], dB, color='coral', alpha=0.8)
    axes[3].fill_between(positions[:len(dA)], 0, dA, color='steelblue', alpha=0.15)
    axes[3].fill_between(positions[:len(dB)], 0, dB, color='coral', alpha=0.15)
    axes[3].set_ylabel('Domain-crowding\nrisk')
    axes[3].set_xlabel('Codon position (synthesis progress)')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path, help="JSON from γ=0 baseline")
    parser.add_argument("run_b", type=Path, help="JSON from γ>0")
    parser.add_argument("--wt_dir",
                        default="/Users/jonathanopitz/Desktop/Master/data/ribo_counts",
                        type=Path)
    parser.add_argument("--n_simulations", type=int, default=N_SIMULATIONS_DEFAULT,
                        help=f"Number of stochastic reps (default {N_SIMULATIONS_DEFAULT})")
    parser.add_argument("--sim_time", type=float, default=500.0,
                        help="TASEP simulation time (arbitrary units; default 500)")
    parser.add_argument("--plot", action='store_true',
                        help="Save 4-panel plot alongside run_b JSON")
    args = parser.parse_args()

    runA = load_run(args.run_a)
    runB = load_run(args.run_b)
    if runA['gene'] != runB['gene']:
        raise SystemExit(f"Gene mismatch: {runA['gene']} vs {runB['gene']}")

    n_codons = len(runA['seq']) // 3
    domain_arr = load_domain_boundaries(runA['gene'], args.wt_dir, n_codons)
    if domain_arr is None:
        domain_arr = np.zeros(n_codons)

    print(f"Running TASEP for Run A (γ={runA['gamma']})...  "
          f"{args.n_simulations} simulations × {args.sim_time} time units")
    resA = run_multiple_tasep(runA['seq'], n_sims=args.n_simulations,
                               sim_time=args.sim_time)
    print(f"Running TASEP for Run B (γ={runB['gamma']})...")
    resB = run_multiple_tasep(runB['seq'], n_sims=args.n_simulations,
                               sim_time=args.sim_time)

    hA = risk_hydrophobic(runA['seq'], resA['mean_dwell'])
    hB = risk_hydrophobic(runB['seq'], resB['mean_dwell'])
    cA = risk_orphan_cys(runA['seq'], resA['mean_dwell'])
    cB = risk_orphan_cys(runB['seq'], resB['mean_dwell'])
    dA = risk_domain_crowding(resA['mean_dwell'], domain_arr)
    dB = risk_domain_crowding(resB['mean_dwell'], domain_arr)

    print_report(runA['gene'], runA, runB, hA, hB, cA, cB, dA, dB,
                 resA['mean_visits'], resB['mean_visits'], args.n_simulations)

    if args.plot:
        out = args.run_b.parent / f"tasep_{runA['gene']}_g{runA['gamma']}_vs_g{runB['gamma']}.png"
        plot_trajectories(runA['gene'], runA, runB, hA, hB, cA, cB, dA, dB,
                          resA['mean_dwell'], resB['mean_dwell'], out)


if __name__ == '__main__':
    main()
