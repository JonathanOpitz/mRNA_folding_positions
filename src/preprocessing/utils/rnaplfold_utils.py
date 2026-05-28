#!/usr/bin/env python3
"""
Shared RNAplfold utilities for WT and simulated RNA structure feature extraction.

Provides:
  - get_rnaplfold_path()       — locate RNAplfold binary
  - run_rnaplfold()            — run RNAplfold with retry logic, return per-nt features
  - aggregate_codon_features() — average per-nt features over a 3-nt codon
  - UNPAIRED_COLS              — standard column name list
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

UNPAIRED_COLS = [
    "rna_unpaired_1nt", "rna_unpaired_2nt", "rna_unpaired_3nt",
    "rna_unpaired_4nt", "rna_unpaired_5nt",
]

WINDOW_SMALL = 15   # ±15 nt ≈ ±5 codons
WINDOW_LARGE = 30   # ±30 nt ≈ ±10 codons

_RNAPLFOLD_PARAMS = ["-W", "80", "-L", "40", "-u", "5", "-T", "37.0", "-O"]
_RNAPLFOLD_MINIMAL = ["-W", "60", "-u", "2", "-O"]

_DEFAULT_NT: dict = {
    "paired_prob": 0.5,
    "unpaired_1nt": 0.7, "unpaired_2nt": 0.6,
    "unpaired_3nt": 0.5, "unpaired_4nt": 0.4, "unpaired_5nt": 0.3,
    "opening_energy_1nt": 0.0, "opening_energy_3nt": 0.0,
    "opening_energy_5nt": 0.0,
    "paired_prob_window5cod": 0.5, "paired_prob_window10cod": 0.5,
    "struct_change": 0.0,
}


# ─── Path discovery ───────────────────────────────────────────────────────────

def get_rnaplfold_path() -> str:
    """Locate the RNAplfold binary.

    Resolution order:
    1. RNAPLFOLD_BIN environment variable
    2. PATH via shutil.which
    3. Common install locations (homebrew, system)
    """
    env_override = os.environ.get("RNAPLFOLD_BIN")
    if env_override:
        return env_override
    candidates = [
        shutil.which("RNAplfold"),
        "/opt/homebrew/bin/RNAplfold",
        "/usr/local/bin/RNAplfold",
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError(
        "RNAplfold not found. Install ViennaRNA or set RNAPLFOLD_BIN=/path/to/RNAplfold."
    )


# ─── Parsing helpers ─────────────────────────────────────────────────────────

def _parse_tabular_file(filepath: Path, n_value_cols: int, prefix: str) -> tuple[dict, int]:
    """Parse an RNAplfold tabular output file (plfold_lunp or plfold_openen)."""
    features: dict = {}
    parse_errors = 0
    if not filepath.exists():
        return features, parse_errors
    try:
        with open(filepath) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = stripped.split()
                if len(parts) >= n_value_cols + 1:
                    try:
                        i = int(parts[0]) - 1  # 1-indexed → 0-indexed
                        for k in range(1, n_value_cols + 1):
                            val = parts[k]
                            features.setdefault(i, {})[f'{prefix}_{k}'] = (
                                float(val) if val != 'NA' else 0.0
                            )
                    except (ValueError, IndexError):
                        parse_errors += 1
    except OSError as e:
        log.warning("Cannot read %s: %s", filepath.name, e)
    return features, parse_errors


def _compute_sliding_windows(paired_arr: np.ndarray, seq_len: int) -> dict:
    """Compute sliding-window means of paired_prob at ±WINDOW_SMALL and ±WINDOW_LARGE."""
    window_features: dict = {}
    for i in range(seq_len):
        lo_s = max(0, i - WINDOW_SMALL)
        hi_s = min(seq_len, i + WINDOW_SMALL + 1)
        lo_l = max(0, i - WINDOW_LARGE)
        hi_l = min(seq_len, i + WINDOW_LARGE + 1)
        mean_small = float(np.mean(paired_arr[lo_s:hi_s]))
        mean_large = float(np.mean(paired_arr[lo_l:hi_l]))
        window_features[i] = {
            'paired_prob_window5cod':  round(mean_small, 4),
            'paired_prob_window10cod': round(mean_large, 4),
            'struct_change':           round(float(paired_arr[i]) - mean_large, 4),
        }
    return window_features


def _parse_rnaplfold_output(tmp_dir: Path, seq_len: int) -> dict:
    """Parse all RNAplfold output files and compute derived per-nucleotide features."""
    nt_features: dict = {}

    lunp_data, lunp_err = _parse_tabular_file(tmp_dir / "plfold_lunp", 5, 'unpaired')
    if lunp_err:
        log.debug("%d lines skipped in plfold_lunp", lunp_err)
    for i, feats in lunp_data.items():
        nt_features.setdefault(i, {})
        for k in range(1, 6):
            nt_features[i][f'unpaired_{k}nt'] = feats.get(f'unpaired_{k}', 0.0)
    for i in nt_features:
        nt_features[i]['paired_prob'] = 1.0 - nt_features[i].get('unpaired_1nt', 0.5)
    if lunp_data:
        log.info("  Parsed %d nucleotides from plfold_lunp", len(lunp_data))

    openen_data, openen_err = _parse_tabular_file(tmp_dir / "plfold_openen", 5, 'openen')
    if openen_err:
        log.debug("%d lines skipped in plfold_openen", openen_err)
    for i, feats in openen_data.items():
        nt_features.setdefault(i, {})
        nt_features[i]['opening_energy_5nt'] = feats.get('openen_5', 0.0)
        nt_features[i]['opening_energy_1nt'] = feats.get('openen_1', 0.0)
        nt_features[i]['opening_energy_3nt'] = feats.get('openen_3', 0.0)
    if openen_data:
        log.info("  Parsed %d nucleotides from plfold_openen", len(openen_data))

    if nt_features:
        paired_arr = np.array([
            nt_features.get(i, {}).get('paired_prob', 0.5) for i in range(seq_len)
        ])
        for i, wf in _compute_sliding_windows(paired_arr, seq_len).items():
            nt_features.setdefault(i, {}).update(wf)
        log.info("  Sliding-window features computed (±%dnt, ±%dnt)", WINDOW_SMALL, WINDOW_LARGE)

    if not nt_features:
        log.warning("No RNAplfold data parsed — using defaults for all %d positions", seq_len)
        return {i: dict(_DEFAULT_NT) for i in range(seq_len)}

    return nt_features


# ─── Main entry points ────────────────────────────────────────────────────────

def run_rnaplfold(seq: str, gene: str, tmp_dir: Path) -> dict:
    """
    Run RNAplfold on seq, writing output files to tmp_dir.

    Tries full parameters first (-W 80 -L 40 -u 5), falls back to minimal
    parameters (-W 60 -u 2) on timeout. Returns a fallback dict of defaults
    if both attempts fail.
    """
    tmp_dir.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_dir)

    try:
        exe = get_rnaplfold_path()
    except FileNotFoundError as e:
        log.error("%s", e)
        os.chdir(cwd)
        return {i: dict(_DEFAULT_NT) for i in range(len(seq))}

    rnaplfold_input = f"{seq}\n@\n"
    log.info("  Running RNAplfold (%d nt) for %s ...", len(seq), gene)

    for attempt, (extra_params, timeout) in enumerate(
        [(_RNAPLFOLD_PARAMS, 120), (_RNAPLFOLD_MINIMAL, 90)], 1
    ):
        cmd = [exe] + extra_params
        try:
            result = subprocess.run(
                cmd, input=rnaplfold_input,
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                log.info("  RNAplfold succeeded (attempt %d)", attempt)
                os.chdir(cwd)
                feats = _parse_rnaplfold_output(tmp_dir, len(seq))
                if feats:
                    return feats
                log.warning("  No features parsed on attempt %d", attempt)
            else:
                log.warning(
                    "  RNAplfold exit %d on attempt %d: %s",
                    result.returncode, attempt, result.stderr[:150],
                )
        except subprocess.TimeoutExpired:
            log.warning("  RNAplfold timed out after %ds (attempt %d)", timeout, attempt)

    log.warning("  All RNAplfold attempts failed for %s — using defaults", gene)
    os.chdir(cwd)
    return {i: dict(_DEFAULT_NT) for i in range(len(seq))}


def aggregate_codon_features(nt_features: dict, nt_start: int, nt_end: int) -> dict:
    """
    Average per-nucleotide RNAplfold features over a 3-nucleotide codon window.

    nt_end is inclusive (e.g. positions 30, 31, 32 for codon 10).
    Returns NaN-filled dict for malformed inputs.
    """
    all_cols = (
        ["rna_local_paired_prob"] + UNPAIRED_COLS
        + ["rna_opening_energy_1nt", "rna_opening_energy_3nt", "rna_opening_energy_5nt",
           "rna_paired_prob_window5cod", "rna_paired_prob_window10cod",
           "rna_struct_change", "rna_local_ss_class"]
    )
    if (nt_end - nt_start + 1) != 3 or nt_start < 0:
        return {c: float('nan') for c in all_cols}

    positions = range(nt_start, nt_end + 1)

    def _avg(key, default=0.0):
        return float(np.mean([nt_features.get(i, {}).get(key, default) for i in positions]))

    avg_paired = _avg('paired_prob', 0.0)
    cls = "stem" if avg_paired > 0.7 else "open_loop" if avg_paired < 0.3 else "weak"

    return {
        "rna_local_paired_prob":       round(avg_paired, 4),
        "rna_unpaired_1nt":            round(_avg('unpaired_1nt', 1.0), 4),
        "rna_unpaired_2nt":            round(_avg('unpaired_2nt', 1.0), 4),
        "rna_unpaired_3nt":            round(_avg('unpaired_3nt', 1.0), 4),
        "rna_unpaired_4nt":            round(_avg('unpaired_4nt', 1.0), 4),
        "rna_unpaired_5nt":            round(_avg('unpaired_5nt', 1.0), 4),
        "rna_opening_energy_1nt":      round(_avg('opening_energy_1nt', 0.0), 4),
        "rna_opening_energy_3nt":      round(_avg('opening_energy_3nt', 0.0), 4),
        "rna_opening_energy_5nt":      round(_avg('opening_energy_5nt', 0.0), 4),
        "rna_paired_prob_window5cod":  round(_avg('paired_prob_window5cod', 0.5), 4),
        "rna_paired_prob_window10cod": round(_avg('paired_prob_window10cod', 0.5), 4),
        "rna_struct_change":           round(_avg('struct_change', 0.0), 4),
        "rna_local_ss_class":          cls,
    }
