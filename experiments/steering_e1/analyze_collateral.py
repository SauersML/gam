"""E2 — the collateral-damage curve (gam#2234, the headline plot).

E1 (`run_e1.py`) sweeps explicit target shifts and fractional doses for BOTH
intervention arms (on-manifold code rotation vs matched-norm flat-direction
addition) and records, per intervention, the achieved effect (FULL-SOFTMAX
probability mass moved onto the intended next-day target token) and collateral
`KL(patched_non_target || base_non_target)`. E2 is the read-off the #2234 thesis
lives or dies on:

    "For each intervention family, sweep achieved-effect vs collateral KL. The
     on-manifold curve should sit strictly below flat steering's — curved
     features are the right control knobs."

This module consumes `e1_records.jsonl` and produces the effect→collateral
efficiency frontier per arm plus the strict-dominance verdict, with NO model in the loop
(pure numpy over the recorded points), so it re-runs cheaply on any E1 output.

The frontier is `g_arm(e) = min collateral over recorded points whose achieved
effect ≥ e` — the least collateral at which the arm can BUY at least effect `e`.
It is monotone non-decreasing in `e`, and "manifold below flat" is exactly
`g_manifold(e) < g_flat(e)` over the shared achievable-effect range. Reporting
the frontier (rather than a raw scatter) is the honest matched-effect
comparison: the two arms reach different effects at the same dose `k`, so they
can only be compared at matched EFFECT, not matched dose.

    python3 experiments/steering_e1/analyze_collateral.py \
        --e1-dir experiments/steering_e1/out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("manifold", "flat")


def log(msg: str) -> None:
    print(f"[e2] {msg}", flush=True)


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def arm_points(records: list[dict[str, Any]], arm: str) -> np.ndarray:
    """`(effect, collateral)` per recorded intervention for one arm. Effect =
    full-softmax probability mass moved onto the intended next-day target token;
    collateral = target-excluded KL(patched||base). Only
    positive-effect interventions belong on an efficiency frontier."""
    pts = []
    for record in records:
        if record["arm"] != arm or float(record["dose_fraction"]) == 0.0:
            continue
        effect = float(record["target_probability_mass_moved"])
        if effect > 0.0:
            pts.append((effect, float(record["collateral_kl_model_to_base_non_target"])))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def frontier(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """`g(e) = min collateral over points with effect ≥ e`, evaluated on `grid`.
    NaN where no recorded point reaches effect `e` (the arm cannot buy it at any
    observed dose)."""
    eff, col = points[:, 0], points[:, 1]
    out = np.full(grid.shape, np.nan)
    for i, e in enumerate(grid):
        reach = col[eff >= e - 1e-12]
        if reach.size:
            out[i] = float(reach.min())
    return out


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    pts = {arm: arm_points(records, arm) for arm in ARMS}
    for arm in ARMS:
        if pts[arm].size == 0:
            raise ValueError(f"arm '{arm}' has no non-trivial steering records")

    # Compare only over the effect range BOTH arms actually reach, so neither is
    # credited for effect the other never achieved.
    lo = max(pts[arm][:, 0].min() for arm in ARMS)
    hi = min(pts[arm][:, 0].max() for arm in ARMS)
    grid = np.linspace(lo, hi, 41) if hi > lo else np.asarray([lo])
    fr = {arm: frontier(pts[arm], grid) for arm in ARMS}

    both = np.isfinite(fr["manifold"]) & np.isfinite(fr["flat"])
    gap = fr["flat"][both] - fr["manifold"][both]  # >0 ⇒ manifold cheaper
    dominance_fraction = float(np.mean(gap > 0.0)) if gap.size else float("nan")
    mean_gap = float(np.mean(gap)) if gap.size else float("nan")

    # Aggregate collateral efficiency: total collateral spent per unit of total
    # achieved effect. Lower is better; manifold should undercut flat.
    def efficiency(p: np.ndarray) -> float:
        eff_sum = float(p[:, 0].sum())
        return float(p[:, 1].sum() / eff_sum) if eff_sum > 0 else float("nan")

    eff_ratio = {arm: efficiency(pts[arm]) for arm in ARMS}

    curve = [
        {
            "effect": float(e),
            "manifold_collateral": (None if not np.isfinite(fr["manifold"][i]) else float(fr["manifold"][i])),
            "flat_collateral": (None if not np.isfinite(fr["flat"][i]) else float(fr["flat"][i])),
        }
        for i, e in enumerate(grid)
    ]

    manifold_dominates = bool(gap.size and np.all(gap > 0.0))
    return {
        "n_points": {arm: int(pts[arm].shape[0]) for arm in ARMS},
        "effect_range": {"lo": float(lo), "hi": float(hi)},
        "collateral_efficiency": eff_ratio,  # collateral KL per unit achieved effect
        "frontier_dominance_fraction": dominance_fraction,  # fraction of effect grid where manifold cheaper
        "mean_collateral_gap": mean_gap,  # mean (flat − manifold) collateral over the shared grid
        "manifold_dominates": bool(manifold_dominates),
        "curve": curve,
    }


def write_report(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2_collateral.json").write_text(json.dumps(result, indent=2) + "\n")

    eff = result["collateral_efficiency"]
    lines = [
        "# E2 — Collateral-damage curve (gam#2234)",
        "",
        "Achieved effect = full-softmax probability mass moved onto the intended next-day target",
        "token; collateral = target-excluded `KL(patched || base)`. The frontier `g(e)` is",
        "the least collateral",
        "at which each arm buys at least effect `e` — the honest matched-EFFECT",
        "comparison (the arms reach different effects at the same dose).",
        "",
        f"- Points: manifold `{result['n_points']['manifold']}`, flat `{result['n_points']['flat']}`.",
        f"- Shared achievable-effect range: `[{result['effect_range']['lo']:.4f}, {result['effect_range']['hi']:.4f}]`.",
        f"- Collateral per unit effect (lower is better): manifold `{eff['manifold']:.4f}`, "
        f"flat `{eff['flat']:.4f}`.",
        f"- Frontier dominance fraction (manifold cheaper): `{result['frontier_dominance_fraction']:.3f}`.",
        f"- Mean collateral gap `flat − manifold`: `{result['mean_collateral_gap']:.4f}` "
        "(positive ⇒ manifold below flat).",
        "",
        f"**Verdict (#2234 E2): manifold steering {'DOMINATES' if result['manifold_dominates'] else 'does NOT dominate'} "
        "flat at matched effect.**",
        "",
        "## Efficiency frontier (collateral to reach ≥ effect e)",
        "",
        "| achieved effect e | manifold collateral | flat collateral |",
        "|---:|---:|---:|",
    ]
    for row in result["curve"]:
        m = "—" if row["manifold_collateral"] is None else f"{row['manifold_collateral']:.4f}"
        fl = "—" if row["flat_collateral"] is None else f"{row['flat_collateral']:.4f}"
        lines.append(f"| {row['effect']:.4f} | {m} | {fl} |")
    lines.append("")
    (out_dir / "e2_collateral.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out_dir / 'e2_collateral.md'}")


def run(e1_dir: Path) -> dict[str, Any]:
    records = load_records(e1_dir / "e1_records.jsonl")
    result = analyze(records)
    write_report(e1_dir, result)
    log(
        f"manifold_dominates={result['manifold_dominates']} "
        f"dominance_fraction={result['frontier_dominance_fraction']:.3f} "
        f"eff(manifold)={result['collateral_efficiency']['manifold']:.4f} "
        f"eff(flat)={result['collateral_efficiency']['flat']:.4f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-dir", default="experiments/steering_e1/out",
                    help="directory holding e1_records.jsonl")
    return ap.parse_args()


def main() -> int:
    run(Path(parse_args().e1_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
