#!/usr/bin/env python3
"""
Multi-gene multi-config sweep wrapper for ribodecode_with_fold_penalty.py.

Strategy: TWO-PHASE sweep designed to fit a 1-2 week compute budget on a
MacBook Air M4 while producing publication-grade results.

  Phase 1 — BROAD COVERAGE (high priority):
    All 19 genes × {γ=0, γ=0.3} × 1 epoch × no MFE
    → 38 runs × ~55 min  ≈  35 hours
    → Establishes whether fold_penalty generalizes across genes.

  Phase 2 — γ SWEEP for selected genes:
    5 representative genes × {γ=0, 0.1, 0.3, 0.5, 1.0} × 1 epoch × no MFE
    → 25 runs × ~55 min  ≈  23 hours
    (subtract Phase 1 overlap → only 15 new runs needed)
    → Characterizes the γ trade-off curve.

  Phase 3 — DEPTH (1-2 genes, optional):
    2 genes × {γ=0, 0.3} × 3 epochs × no MFE
    → 4 runs × 3 epochs ≈ 11 hours
    → Verifies single-epoch results converge on multi-epoch.

  Phase 4 — MFE BASELINE (1-2 genes, optional):
    1 gene × {γ=0, 0.3} × {mfe_weight=0.3} × 1 epoch
    → 2 expensive runs ≈ 5-6 hours
    → Provides comparison with paper's MFE-aware baseline.

  Total: ~70-90 runs, fits in roughly one week of overnight compute.

Resume-safe: if output JSON exists, skips. Stop/restart anytime.

Usage:
  python3 sweep_runner.py --list          # show plan
  python3 sweep_runner.py --dry_run       # preview execution order
  python3 sweep_runner.py                 # run sweep
  python3 sweep_runner.py --gene ACTB     # restrict to one gene
  python3 sweep_runner.py --max_runs 5    # stop after N successful runs
  python3 sweep_runner.py --phase 1       # only run Phase 1
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).resolve().parent
# Override with PYTHON_BIN env var if running inside a custom environment.
PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", sys.executable))
SCRIPT     = BASE_DIR / "ribodecode_with_fold_penalty.py"
GENES_DIR  = BASE_DIR / "data/genes"
OUT_DIR    = BASE_DIR / "data/optimized"
LOG_DIR    = BASE_DIR / "logs"

# ═══════════════════════════════════════════════════════════════════════════
# SWEEP DEFINITION — four phases, balanced for ~1 week of overnight compute
# ═══════════════════════════════════════════════════════════════════════════

ALL_GENES = [
    'ACTB', 'CASP7', 'CCT3', 'CTSB', 'EGFR', 'FGA', 'HBB', 'HSPD1',
    'IDH2', 'IFNB1', 'LMNA', 'LMNB1', 'MAPK1', 'MAPK3', 'NSF',
    'PFKM', 'PKM', 'PROC', 'TTR',
]

# 5 representative genes for the γ-sweep, picked to span structural complexity:
#   ACTB    — cytoskeletal, well-studied, multi-domain
#   PFKM    — large enzyme, multiple domains
#   HBB     — small (~146 aa), single Hb subunit (good simple-protein control)
#   EGFR    — receptor, many disulfides, complex folding
#   HSPD1   — chaperone (interesting biological reverse case)
GAMMA_SWEEP_GENES = ['ACTB', 'PFKM', 'HBB', 'EGFR', 'HSPD1']

# 2 genes for multi-epoch convergence test
DEPTH_GENES = ['ACTB', 'PFKM']

# 1 gene for the expensive MFE-aware baseline
MFE_GENE = 'ACTB'

ENV = "HEK293T"
GNN_UPDATE_EVERY = 5

# Phase definitions: each entry is (gene_list, fold_gammas, mfe_weights,
#                                    rpf_targets, epochs, phase_id, description)
PHASES = [
    # Phase 1: broad screening across all genes
    {
        'id': 1,
        'desc': 'Phase 1: Broad screening (all genes, γ∈{0, 0.3})',
        'genes':       ALL_GENES,
        'fold_gammas': [0.0, 0.3],
        'mfe_weights': [0.0],
        'rpf_targets': [100.0],
        'epochs':      [1],
    },
    # Phase 2: γ-sweep for selected genes
    {
        'id': 2,
        'desc': 'Phase 2: γ sweep for 5 selected genes',
        'genes':       GAMMA_SWEEP_GENES,
        'fold_gammas': [0.0, 0.1, 0.3, 0.5, 1.0],
        'mfe_weights': [0.0],
        'rpf_targets': [100.0],
        'epochs':      [1],
    },
    # Phase 3: depth — multi-epoch convergence
    {
        'id': 3,
        'desc': 'Phase 3: Multi-epoch convergence (2 genes × {γ=0, 0.3})',
        'genes':       DEPTH_GENES,
        'fold_gammas': [0.0, 0.3],
        'mfe_weights': [0.0],
        'rpf_targets': [100.0],
        'epochs':      [3],
    },
    # Phase 4: MFE-aware baseline (expensive!)
    {
        'id': 4,
        'desc': 'Phase 4: MFE-aware baseline (1 gene, γ∈{0, 0.3}, mfe_weight=0.3)',
        'genes':       [MFE_GENE],
        'fold_gammas': [0.0, 0.3],
        'mfe_weights': [0.3],
        'rpf_targets': [100.0],
        'epochs':      [1],
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def find_fasta(gene: str) -> Optional[Path]:
    """Find the FASTA file for a gene in data/genes/."""
    gene_u = gene.upper()
    candidates = list(GENES_DIR.glob(f"{gene_u}_*.fasta")) + \
                 list(GENES_DIR.glob(f"{gene_u}*.fasta"))
    candidates = [c for c in candidates
                  if 'gemorna' not in c.name.lower()
                  and 'denovo' not in c.name.lower()
                  and 'simulated' not in c.name.lower()
                  and 'optimized' not in c.name.lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def load_cds_seq(fasta_path: Path) -> str:
    lines = fasta_path.read_text().split('\n')
    seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    return seq.upper().replace('U', 'T')


def output_tag(gene: str, fold_gamma: float, mfe_weight: float,
               env: str, epochs: int) -> str:
    fold_tag = f"fold{fold_gamma:.2f}".replace('.', '')
    mfe_tag  = f"mfe{mfe_weight:.2f}".replace('.', '')
    return f"{gene.upper()}_{fold_tag}_{mfe_tag}_{env}_ep{epochs}"


def output_already_exists(tag: str) -> bool:
    json_path = OUT_DIR / f"{tag}_results.json"
    return json_path.exists() and json_path.stat().st_size > 100


def estimate_runtime_minutes(mfe_weight: float, epochs: int) -> float:
    """Rough estimate: ~55 min/epoch base, ~150 min/epoch with mfe training."""
    if mfe_weight > 0:
        return 150 * epochs
    return 55 * epochs


def format_duration(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds/60:.0f} min"
    if seconds < 86400:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.1f} d"


def build_combos_for_phase(phase: dict) -> List[dict]:
    """Expand one phase into a list of individual run specs."""
    combos = []
    for gene, fg, mw, rpf, ep in itertools.product(
        phase['genes'], phase['fold_gammas'], phase['mfe_weights'],
        phase['rpf_targets'], phase['epochs']
    ):
        combos.append(dict(
            gene=gene, fold_gamma=fg, mfe_weight=mw,
            rpf_target=rpf, epochs=ep, env=ENV,
            phase_id=phase['id'],
        ))
    return combos


def deduplicate_combos(combos: List[dict]) -> List[dict]:
    """If two phases produce identical configs, keep only the lower-phase one."""
    seen = {}
    for c in combos:
        tag = output_tag(c['gene'], c['fold_gamma'], c['mfe_weight'],
                         c['env'], c['epochs'])
        # Keep the first occurrence (i.e. earlier phase wins)
        if tag not in seen:
            seen[tag] = c
    return list(seen.values())


def run_single(gene: str, cds_seq: str, fold_gamma: float, mfe_weight: float,
               rpf_target: float, epochs: int, env: str,
               log_dir: Path) -> tuple:
    """Returns (success, elapsed_seconds, log_path)."""
    tag = output_tag(gene, fold_gamma, mfe_weight, env, epochs)
    log_path = log_dir / f"sweep_{tag}.log"

    cmd = [
        str(PYTHON_BIN), str(SCRIPT),
        "--cds",          gene.upper(),
        "--cds_seq",      cds_seq,
        "--env",          env,
        "--mfe_weight",   str(mfe_weight),
        "--optim_epoch",  str(epochs),
        "--alpha",        str(rpf_target),
        "--beta",         "100",
        "--fold_gamma",   str(fold_gamma),
        "--gnn_update_every", str(GNN_UPDATE_EVERY),
    ]

    start = time.time()
    try:
        with open(log_path, 'w') as logf:
            logf.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logf.write(f"# Tag: {tag}\n")
            logf.write(f"# Command: {' '.join(cmd[:3])} ...\n\n")
            logf.flush()
            result = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR), check=False,
            )
        elapsed = time.time() - start
        success = (result.returncode == 0) and output_already_exists(tag)
        return success, elapsed, log_path
    except Exception as e:
        elapsed = time.time() - start
        with open(log_path, 'a') as logf:
            logf.write(f"\nWRAPPER ERROR: {e}\n")
        return False, elapsed, log_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry_run",  action='store_true',
                        help="Show what would run, don't execute.")
    parser.add_argument("--list",     action='store_true',
                        help="List all combinations with [DONE]/[TODO] status.")
    parser.add_argument("--max_runs", type=int, default=None,
                        help="Stop after this many new successful runs.")
    parser.add_argument("--gene",     type=str, default=None,
                        help="Only run combinations for this single gene.")
    parser.add_argument("--phase",    type=int, default=None,
                        help="Only run a specific phase (1-4).")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build combo list across all phases ──
    all_combos = []
    selected_phases = [p for p in PHASES if args.phase is None or p['id'] == args.phase]
    for phase in selected_phases:
        all_combos.extend(build_combos_for_phase(phase))
    all_combos = deduplicate_combos(all_combos)

    if args.gene:
        all_combos = [c for c in all_combos if c['gene'] == args.gene.upper()]
        if not all_combos:
            print(f"No combinations for gene {args.gene}", file=sys.stderr)
            return 1

    # ── Categorize done vs todo ──
    todo = []
    done = []
    for c in all_combos:
        tag = output_tag(c['gene'], c['fold_gamma'], c['mfe_weight'],
                         c['env'], c['epochs'])
        if output_already_exists(tag):
            done.append(c)
        else:
            todo.append(c)

    # ── Sort todo: cheaper first (epoch=1, mfe=0) ──
    def cost(c):
        return estimate_runtime_minutes(c['mfe_weight'], c['epochs'])
    todo.sort(key=lambda c: (cost(c), c['phase_id'], c['gene'], c['fold_gamma']))

    # ── Print plan ──
    print()
    print("═" * 72)
    print("  RIBODECODE SWEEP — PLAN")
    print("═" * 72)
    for phase in selected_phases:
        phase_combos = build_combos_for_phase(phase)
        phase_done = sum(1 for c in phase_combos
                         if output_already_exists(output_tag(
                             c['gene'], c['fold_gamma'], c['mfe_weight'],
                             c['env'], c['epochs'])))
        est_per_run = estimate_runtime_minutes(
            phase['mfe_weights'][0], sum(phase['epochs'])/len(phase['epochs']))
        total_min = (len(phase_combos) - phase_done) * est_per_run
        print(f"  {phase['desc']}")
        print(f"      {len(phase_combos)} runs ({phase_done} done, "
              f"{len(phase_combos)-phase_done} todo)  "
              f"~{format_duration(total_min*60)} remaining")

    print("─" * 72)
    print(f"  TOTAL: {len(all_combos)} runs   "
          f"({len(done)} done, {len(todo)} todo)")
    todo_minutes = sum(estimate_runtime_minutes(c['mfe_weight'], c['epochs'])
                       for c in todo)
    print(f"  Estimated remaining time: {format_duration(todo_minutes*60)}")
    print("═" * 72)

    # ── --list mode ──
    if args.list:
        print()
        for phase in selected_phases:
            print(f"\n── Phase {phase['id']}: {phase['desc']} ──")
            for c in build_combos_for_phase(phase):
                tag = output_tag(c['gene'], c['fold_gamma'], c['mfe_weight'],
                                 c['env'], c['epochs'])
                status = "[DONE]" if output_already_exists(tag) else "[TODO]"
                print(f"  {status} {tag}")
        return 0

    # ── --dry_run ──
    if args.dry_run:
        print("\n=== Would execute these (in order): ===")
        for c in todo[:30]:
            tag = output_tag(c['gene'], c['fold_gamma'], c['mfe_weight'],
                             c['env'], c['epochs'])
            est = estimate_runtime_minutes(c['mfe_weight'], c['epochs'])
            print(f"   [P{c['phase_id']}] {tag}  (~{est:.0f} min)")
        if len(todo) > 30:
            print(f"   ... and {len(todo) - 30} more")
        return 0

    if not todo:
        print("\nNothing to do — all combinations have output files. ✓")
        return 0

    # ── Verify FASTAs exist ──
    print("\n[setup] Verifying FASTA files...")
    seen_genes = set(c['gene'] for c in todo)
    cds_cache = {}
    missing = []
    for g in seen_genes:
        fasta = find_fasta(g)
        if fasta is None:
            missing.append(g)
            print(f"   {g}: ✗ MISSING")
        else:
            cds_cache[g] = load_cds_seq(fasta)
            print(f"   {g}: {fasta.name}  ({len(cds_cache[g])} nt)")

    if missing:
        print(f"\n⚠  Missing FASTAs for: {missing}")
        ans = input("Continue anyway, skipping these genes? [y/N]: ")
        if ans.strip().lower() != 'y':
            return 1

    # ── Execute ──
    print()
    print("═" * 72)
    print("  EXECUTING SWEEP")
    print("═" * 72)

    successful = 0; failed = 0; skipped = 0
    sweep_start = time.time()
    runtimes = []  # for adaptive ETA

    for idx, c in enumerate(todo, 1):
        if c['gene'] in missing:
            skipped += 1
            continue
        if args.max_runs is not None and successful >= args.max_runs:
            print(f"\n[wrapper] --max_runs={args.max_runs} reached. Stopping.")
            break

        tag = output_tag(c['gene'], c['fold_gamma'], c['mfe_weight'],
                         c['env'], c['epochs'])
        if output_already_exists(tag):
            skipped += 1
            continue

        avg_per_run = (sum(runtimes)/len(runtimes)/60) if runtimes else \
                      estimate_runtime_minutes(c['mfe_weight'], c['epochs'])
        eta_remaining = (len(todo) - idx + 1) * avg_per_run

        print()
        print(f"[{idx}/{len(todo)}] Phase {c['phase_id']}: {tag}")
        print(f"   γ={c['fold_gamma']}  mfe={c['mfe_weight']}  "
              f"alpha={c['rpf_target']}  epochs={c['epochs']}")
        print(f"   ETA remaining: {format_duration(eta_remaining*60)}  "
              f"(avg {avg_per_run:.0f} min/run)")

        success, elapsed, log_path = run_single(
            gene=c['gene'], cds_seq=cds_cache[c['gene']],
            fold_gamma=c['fold_gamma'], mfe_weight=c['mfe_weight'],
            rpf_target=c['rpf_target'], epochs=c['epochs'],
            env=c['env'], log_dir=LOG_DIR,
        )
        runtimes.append(elapsed)

        if success:
            successful += 1
            print(f"   ✓ DONE in {elapsed/60:.1f} min")
        else:
            failed += 1
            print(f"   ✗ FAILED in {elapsed/60:.1f} min  →  log: {log_path}")
            try:
                lines = log_path.read_text().splitlines()
                for line in lines[-3:]:
                    print(f"      {line}")
            except Exception:
                pass

    # ── Summary ──
    total_elapsed = time.time() - sweep_start
    print()
    print("═" * 72)
    print(f"  SWEEP COMPLETE")
    print(f"     successful: {successful}")
    print(f"     failed:     {failed}")
    print(f"     skipped:    {skipped}")
    print(f"     wall time:  {format_duration(total_elapsed)}")
    print("═" * 72)
    return 0 if failed == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
