"""gam#4564: a rigid-law BMS fit with ``linkwiggle()`` in both formulas does not finish.

The issue's own reproduction runs gnomon's ``shipped_probe.py`` against gnomon's
``examples/biobank`` study, which is not in this repository. Everything that decides the
question, though, is in the issue text: the model is spelled there in full, and the frame it
was observed on is described in full (3,000 rows, six independent ``N(0, 1)`` principal
components, a logistic truth with a linear score effect, so off the probit family). This probe
rebuilds that frame here and fits the same model, so timing it needs nothing but a ``gamfit``
wheel.

It is SHAPE-equivalent, not gnomon's frame: the row count, the number of components, the
truth's family and the two formulas are the ones the issue names, and those are what set the
cost. The BMS flex row program's primary width is
``r = 2 + dim(score warp) + dim(link deviation)``, and
``fit_orchestration::deviation::route_marginal_slope_deviation_blocks`` sends the SLOPE
formula's ``linkwiggle()`` to the score warp and the MEAN formula's to the link deviation, so
spelling one in each is what takes ``r`` from 2 to 20 at the ``linkwiggle()`` default of eight
internal knots. The coefficient values differ from gnomon's fit; the width, the number of
smoothing parameters and the row count do not.

Both arms run, in one process, so the comparison needs no second job:

* ``shipped``   -- the issue's model, both ``linkwiggle()`` calls, the arm that did not return;
* ``ours``      -- the same frame with no wiggle in either formula, the arm the issue reports
                   finishing in 9-16 s at 8 threads. It is the positive control: if it is also
                   slow here, the frame is not the one the issue measured and nothing below is
                   comparable.

Each arm is wrapped in its own timer and its own exception guard, so a refusal from one is
reported with its message and the other still runs.

Usage:

    python3 experiments/bms_linkwiggle_both_formulas_4564.py [--rows 3000] [--seed 7]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import gamfit

COMPONENTS = ("PC1", "PC2", "PC3", "PC4", "PC5", "PC6")

SHIPPED_FORMULA = (
    "y ~ sex + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=9) + linkwiggle()"
)
SHIPPED_SLOPE = (
    "1 + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=8) + linkwiggle()"
)
CONTROL_FORMULA = "y ~ sex + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=9)"
CONTROL_SLOPE = "1 + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=8)"


def build_frame(rows: int, seed: int) -> dict[str, np.ndarray]:
    """The frame the issue describes, built from one seeded generator.

    Six independent standard normal components, a balanced binary ``sex``, a score ``z``
    standardised on its own sample so the latent measure anchors on a unit-scale axis, and a
    LOGISTIC truth whose score effect is linear -- which is what puts the truth off the probit
    family the marginal-slope model assumes, the condition the issue's test module documents
    as sending a link wiggle's lambda to the boundary (gam#2978).
    """
    rng = np.random.default_rng(seed)
    frame: dict[str, np.ndarray] = {
        name: rng.standard_normal(rows) for name in COMPONENTS
    }
    frame["sex"] = (rng.uniform(size=rows) < 0.5).astype(float)
    raw_score = rng.standard_normal(rows)
    frame["z"] = (raw_score - raw_score.mean()) / raw_score.std(ddof=0)
    component_effect = (
        0.30 * frame["PC1"] - 0.20 * frame["PC2"] + 0.15 * frame["PC3"] * frame["PC3"]
    )
    eta = -0.7 + 0.4 * frame["sex"] + component_effect + 0.8 * frame["z"]
    probability = 1.0 / (1.0 + np.exp(-eta))
    frame["y"] = (rng.uniform(size=rows) < probability).astype(float)
    return frame


def run_arm(label: str, frame: dict[str, np.ndarray], formula: str, slope: str) -> None:
    started = time.perf_counter()
    try:
        model = gamfit.fit(
            frame,
            formula,
            family="bernoulli-marginal-slope",
            z_column="z",
            slope_formula=slope,
            config={"latent_measure": "global-empirical"},
        )
    except Exception as error:  # noqa: BLE001 - the refusal text is the result here
        elapsed = time.perf_counter() - started
        print(f"[#4564 {label}] REFUSED after {elapsed:.1f} s: {type(error).__name__}: {error}", flush=True)
        return
    elapsed = time.perf_counter() - started
    fitted = np.asarray(model.predict(frame), dtype=float)
    print(
        f"[#4564 {label}] returned in {elapsed:.1f} s; "
        f"fitted mean {fitted.mean():.6f}, observed mean {frame['y'].mean():.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000, help="rows in the frame (issue: 3000)")
    parser.add_argument("--seed", type=int, default=7, help="seed for the frame")
    parser.add_argument(
        "--arm",
        choices=("both", "shipped", "ours"),
        default="both",
        help="which arm to run; 'ours' alone is the positive control",
    )
    args = parser.parse_args()

    frame = build_frame(args.rows, args.seed)
    print(
        f"[#4564] frame: rows={args.rows} components={len(COMPONENTS)} "
        f"positives={int(frame['y'].sum())} seed={args.seed}",
        flush=True,
    )
    if args.arm in ("ours", "both"):
        run_arm("ours (no linkwiggle, positive control)", frame, CONTROL_FORMULA, CONTROL_SLOPE)
    if args.arm in ("shipped", "both"):
        run_arm("shipped (linkwiggle in both formulas)", frame, SHIPPED_FORMULA, SHIPPED_SLOPE)


if __name__ == "__main__":
    main()
