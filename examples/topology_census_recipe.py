#!/usr/bin/env python
"""Run an honest unsupervised topology census with full-pipeline controls.

The runner calls one user-supplied census function three times: once on the
observed activation matrix, once on a per-dimension shuffle, and once on a
covariance-matched Gaussian.  The callable and its seed are identical across
all three runs.  Only the input matrix changes, so SAE training,
co-activation grouping, projection, and shape adjudication all remain inside
the controlled path.

The pipeline callable is supplied as ``MODULE:CALLABLE`` and must have this
contract::

    def run_census(activations: np.ndarray, *, seed: int) -> Mapping[str, object]:
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
non-circular mass.  It is not a count of individual ``winner == "circle"``
labels.  A ``ring_clusters_k{k}`` winner is already owned by the circular
class.  The centroid ordering test remains useful as an independent ordering
diagnostic, especially when a free ``mixture_k{k}`` wins, but it no longer
"rescues" cyclic clusters from a shape race that omitted their density class.

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CONTROL_KINDS = ("per_dimension_shuffle", "covariance_matched_gaussian")
Pipeline = Callable[..., Mapping[str, object]]


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

    mean_l0 = float(payload["dictionary_mean_l0"])
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


def _run_pipeline(
    pipeline: Pipeline,
    activations: np.ndarray,
    *,
    run_name: str,
    seed: int,
) -> dict[str, object]:
    readonly = activations.view()
    readonly.setflags(write=False)
    payload = pipeline(readonly, seed=seed)
    return _validate_pipeline_summary(payload, run_name)


def run_full_pipeline_controls(
    activations: np.ndarray,
    pipeline: Pipeline,
    *,
    pipeline_seed: int = 11,
    control_seed: int = 17,
) -> dict[str, object]:
    """Run the same complete census on observed data and both matched controls.

    ``pipeline`` receives the same keyword arguments on every invocation and
    must create a fresh fit from the supplied matrix.  The returned report
    preserves the two control rates separately and also reports their pooled
    binomial rate; unequal control denominators are never silently averaged.
    """
    if isinstance(pipeline_seed, bool) or not isinstance(pipeline_seed, int):
        raise TypeError("pipeline_seed must be an integer")
    if isinstance(control_seed, bool) or not isinstance(control_seed, int):
        raise TypeError("control_seed must be an integer")
    if not callable(pipeline):
        raise TypeError("pipeline must be callable")

    matrix = np.asarray(activations, dtype=np.float64, order="C")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"activations must be a nonempty two-dimensional matrix; got {matrix.shape}")
    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("activations must contain only finite values")

    observed = _run_pipeline(
        pipeline,
        matrix,
        run_name="observed",
        seed=pipeline_seed,
    )

    try:
        import gamfit
    except ImportError as exc:
        raise RuntimeError(
            "full-pipeline controls require a gamfit build exposing shape_matched_control"
        ) from exc

    controls: dict[str, dict[str, object]] = {}
    for kind in CONTROL_KINDS:
        controlled = np.asarray(
            gamfit.shape_matched_control(matrix, kind=kind, seed=control_seed),
            dtype=np.float64,
            order="C",
        )
        if controlled.shape != matrix.shape or not np.isfinite(controlled).all():
            raise RuntimeError(
                f"shape_matched_control returned invalid {kind!r} matrix {controlled.shape}"
            )
        controls[kind] = _run_pipeline(
            pipeline,
            controlled,
            run_name=kind,
            seed=pipeline_seed,
        )
        del controlled

    control_adjudications = sum(int(run["n_adjudicated"]) for run in controls.values())
    control_circular_wins = sum(int(run["n_circular_wins"]) for run in controls.values())
    return {
        "observed": observed,
        "controls": controls,
        "control_false_circle_rates": {
            kind: controls[kind]["circular_win_rate"] for kind in CONTROL_KINDS
        },
        "pooled_control_false_circle_rate": (
            control_circular_wins / control_adjudications if control_adjudications else None
        ),
        "pipeline_seed": pipeline_seed,
        "control_seed": control_seed,
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
    report = run_full_pipeline_controls(
        loaded,
        _load_pipeline(args.pipeline),
        pipeline_seed=args.pipeline_seed,
        control_seed=args.control_seed,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
