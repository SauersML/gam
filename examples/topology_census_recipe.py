#!/usr/bin/env python
"""Run an honest unsupervised topology census with full-pipeline controls.

The runner delegates to ``gamfit.run_shape_controlled_census``, which calls one
user-supplied census function three times: once on the observed activation
matrix, once on a per-dimension shuffle, and once on a covariance-exact
randomized Hadamard control. The callable and its seed are identical across all
three runs. Only the input matrix changes, so SAE training, co-activation
grouping, projection, and shape adjudication all remain inside the controlled
path.

The pipeline callable is supplied as ``MODULE:CALLABLE`` and must have this
contract::

    def run_census(activations: np.ndarray, seed: int) -> Mapping[str, object]:
        # Construct every fitted object from scratch on each invocation.
        ...
        return {
            "n_attempted": ...,       # includes skipped/failed groups
            "n_adjudicated": ...,     # successful shape races
            "n_circular_wins": ...,   # verdict["circle_wins"] is true
            "dictionary_mean_l0": ...,
        }

``n_circular_wins`` uses the production circular-class verdict: total stacking
mass of the smooth-circle and ring-of-clusters candidates exceeds total
non-circular mass. It is not a count of individual
``reporting_winner == "circle"`` labels. A
``winner_class == "ring_clusters"`` verdict is already owned by the
circular class; ``ring_clusters_reporting_k`` names its all-data diagnostic
fit, while ``ring_clusters_fold_selected_k`` records the leakage-free
outer-fold orders. The centroid ordering test remains useful as an independent
ordering diagnostic, especially when the free ``mixture`` class wins, but it
no longer "rescues" cyclic clusters from a race that omitted their density
class.

Example::

    python examples/topology_census_recipe.py \
        --activations activations.npy \
        --pipeline my_census:run_census \
        --pipeline-seed 11 \
        --control-seed 17 \
        --output census_with_controls.json

The activation file must contain one finite, two-dimensional NumPy array.
Controls are generated and consumed one at a time so the runner never retains
both corpus-sized controls simultaneously.  The native control generator
requires contiguous ``float64`` input; conversion may therefore materialize
one copy when the source file uses another dtype or layout.

Measured caveats from the Qwen3-8B/OLMo-2 censuses still apply:

* report dictionary mean L0 next to every verdict rate (dense dictionaries
  suppressed circle verdicts entirely in the recorded runs);
* count failed adjudications as attempted-but-not-adjudicated, never as
  non-circular verdicts;
* top-two-variance PCA can lose a genuine ring masked by a linear factor at
  roughly one radius, so absence of circular wins is not absence of circles.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


Pipeline = Callable[[np.ndarray, int], Mapping[str, object]]


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"pipeline field {field!r} must be an integer; got {value!r}")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"pipeline field {field!r} must be non-negative; got {integer}")
    return integer


def _validate_pipeline_summary(payload: object, run_name: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"pipeline run {run_name!r} must return a mapping; got {type(payload).__name__}"
        )
    required = {
        "n_attempted",
        "n_adjudicated",
        "n_circular_wins",
        "dictionary_mean_l0",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"pipeline run {run_name!r} omitted required fields {missing}")

    attempted = _nonnegative_integer(payload["n_attempted"], "n_attempted")
    adjudicated = _nonnegative_integer(payload["n_adjudicated"], "n_adjudicated")
    circular_wins = _nonnegative_integer(payload["n_circular_wins"], "n_circular_wins")
    if adjudicated > attempted:
        raise ValueError(
            f"pipeline run {run_name!r} adjudicated {adjudicated} of only {attempted} attempts"
        )
    if circular_wins > adjudicated:
        raise ValueError(
            f"pipeline run {run_name!r} reported {circular_wins} circular wins "
            f"from only {adjudicated} adjudications"
        )

    mean_l0_value = payload["dictionary_mean_l0"]
    if isinstance(mean_l0_value, (bool, np.bool_)) or not isinstance(
        mean_l0_value, numbers.Real
    ):
        raise TypeError(
            f"pipeline field 'dictionary_mean_l0' must be a real number; got {mean_l0_value!r}"
        )
    mean_l0 = float(mean_l0_value)
    if not math.isfinite(mean_l0) or mean_l0 < 0.0:
        raise ValueError(
            f"pipeline run {run_name!r} returned invalid dictionary_mean_l0={mean_l0}"
        )
    return {
        "n_attempted": attempted,
        "n_adjudicated": adjudicated,
        "n_failed_or_skipped": attempted - adjudicated,
        "n_circular_wins": circular_wins,
        "circular_win_rate": circular_wins / adjudicated if adjudicated else None,
        "dictionary_mean_l0": mean_l0,
    }


def _validated_control_report(controlled_census: object) -> dict[str, object]:
    """Validate and summarize a public ``ShapeControlledCensus`` result."""
    observed = _validate_pipeline_summary(controlled_census.observed, "observed")
    controls = {
        "per_dimension_shuffle": _validate_pipeline_summary(
            controlled_census.per_dimension_shuffle,
            "per_dimension_shuffle",
        ),
        "covariance_exact_hadamard": _validate_pipeline_summary(
            controlled_census.covariance_exact_hadamard,
            "covariance_exact_hadamard",
        ),
    }
    control_adjudications = sum(int(run["n_adjudicated"]) for run in controls.values())
    control_circular_wins = sum(int(run["n_circular_wins"]) for run in controls.values())
    return {
        "observed": observed,
        "controls": controls,
        "control_false_circle_rates": {
            kind: run["circular_win_rate"] for kind, run in controls.items()
        },
        "pooled_control_false_circle_rate": (
            control_circular_wins / control_adjudications if control_adjudications else None
        ),
        "pipeline_seed": controlled_census.pipeline_seed,
        "per_dimension_shuffle_seed": controlled_census.per_dimension_shuffle_seed,
        "covariance_exact_hadamard_seed": controlled_census.covariance_exact_hadamard_seed,
    }


def _load_pipeline(specification: str) -> Pipeline:
    module_name, separator, qualified_name = specification.partition(":")
    if not separator or not module_name or not qualified_name:
        raise ValueError("--pipeline must use MODULE:CALLABLE syntax")
    target: Any = importlib.import_module(module_name)
    for attribute in qualified_name.split("."):
        if not attribute:
            raise ValueError("--pipeline callable path contains an empty component")
        target = getattr(target, attribute)
    if not callable(target):
        raise TypeError(f"pipeline target {specification!r} is not callable")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True, help="finite 2-D .npy matrix")
    parser.add_argument(
        "--pipeline",
        required=True,
        help="MODULE:CALLABLE implementing the documented census contract",
    )
    parser.add_argument("--pipeline-seed", type=int, default=11)
    parser.add_argument("--control-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None, help="JSON report path")
    args = parser.parse_args(argv)

    loaded = np.load(args.activations, mmap_mode="r", allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"--activations must name one .npy array; got {type(loaded).__name__}")
    try:
        import gamfit
    except ImportError as exc:
        raise RuntimeError(
            "the full-pipeline runner requires a gamfit build exposing "
            "run_shape_controlled_census"
        ) from exc
    controlled_census = gamfit.run_shape_controlled_census(
        loaded,
        _load_pipeline(args.pipeline),
        pipeline_seed=args.pipeline_seed,
        control_seed=args.control_seed,
    )
    report = _validated_control_report(controlled_census)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
