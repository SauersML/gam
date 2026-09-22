#!/usr/bin/env python3
"""Regenerate the p6d50k survival marginal-slope acceptance frame (gam#3287).

#3287's acceptance is one fit:

    gam -v fit p6d50k.csv --request p6d50k.json --out m

against the a4baeb9ba4 baseline (status=Converged, 9 outer iterations,
loglik -1.560776e4, raw criterion 1.564010e4, 273 s). Both code halves of the
fix are on main (397d0d4bdf, 403a7d005c), but the frame the run needs lived
only on MSI scratch at /scratch.global/sauer354/sp-surv-inner/runs/p6d50k, and
that tree has been purged. An acceptance whose fixture exists in one directory
on one cluster is an acceptance that expires; this generator is the fixture,
committed, so the run can be reproduced at any commit from now on.

WHAT IT REPRODUCES, and what it deliberately does not. The original frame is a
draw from the AoU-study simulator and is not reconstructible from the issue.
What the fit exercises is its SHAPE, and every number below is one the issue
states about that shape rather than a tuning choice:

  * n = 50000 rows, one per subject;
  * DELAYED ENTRY: each subject is observed from `entry_age`, not from birth,
    so the risk set is left-truncated — this is the feature the survival
    marginal-slope time block is built around;
  * 3494 events, i.e. 6.988% of rows, the rest administratively censored;
  * six continuous covariates PC1..PC6 entering through one `duchon(...)`
    smooth at 24 centers, plus a binary `sex`;
  * the same covariates again in the slope formula, which is what makes this a
    marginal-slope fit with a nontrivial psi block.

The baseline is Gompertz in age (log-hazard linear in age), the standard shape
for an adult-onset outcome, and its intercept is SOLVED for so the expected
event count is exactly the issue's 3494 under the realized covariates — a
solve, not a fitted constant. The realized count is a draw about that
expectation, so it is printed and asserted to be within three of the count's own
standard errors, `sqrt(EVENTS)` -- a change to the draw that moved the event
fraction is caught here rather than inside a four-minute fit. This draw realizes
3379, which is 1.95 of them below 3494.

Deterministic with no third-party dependency: one 64-bit LCG, seeded by the
frame's own name, and a Box-Muller pair for the normals. Running it twice
writes byte-identical files.

    python3 bench/fixtures/p6d50k.py --out <dir>

writes <dir>/p6d50k.csv and <dir>/p6d50k.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# The frame's declared shape (gam#3287's own description of the fixture).
ROWS = 50_000
EVENTS = 3_494
PC_COUNT = 6
DUCHON_CENTERS = 24

# Observation window. Entry is uniform on [ENTRY_LOW, ENTRY_HIGH) and everyone
# still at risk at ADMIN_END is censored there, so every row is left-truncated
# and right-censored, which is the risk-set geometry the time block reads.
ENTRY_LOW = 40.0
ENTRY_HIGH = 70.0
ADMIN_END = 80.0

# Gompertz baseline: log h_0(t) = LOG_BASELINE + GOMPERTZ_RATE * (t - ENTRY_LOW).
# The rate is the doubling-per-decade an adult-onset hazard is usually written
# with (ln 2 / 10); LOG_BASELINE is solved for below, not chosen.
GOMPERTZ_RATE = math.log(2.0) / 10.0

# Covariate effects on the log hazard. PC1..PC6 enter through a smooth, so they
# are given a curved contribution the fit has to recover rather than a linear
# one it could absorb into the parametric block.
SEX_LOG_HAZARD = 0.35


class Lcg:
    """One 64-bit linear congruential stream (Knuth's MMIX constants)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFF_FFFF_FFFF_FFFF

    def uniform(self) -> float:
        self.state = (
            self.state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
        ) & 0xFFFF_FFFF_FFFF_FFFF
        # The top 53 bits, half-open in (0, 1) so `log` is always finite.
        return ((self.state >> 11) + 0.5) / float(1 << 53)

    def normal(self) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def smooth_log_hazard(pcs: list[float]) -> float:
    """The six PCs' contribution to the log hazard.

    Curved in PC1 and PC2 and linear in the rest, so the `duchon(...)` smooth
    has something a parametric block cannot absorb while the tail dimensions
    still carry signal. Centred so its mean over a standard normal draw is
    about zero and the baseline solve below is not absorbing it.
    """
    curved = 0.30 * (pcs[0] * pcs[0] - 1.0) + 0.25 * math.tanh(1.5 * pcs[1])
    linear = 0.20 * pcs[2] - 0.15 * pcs[3] + 0.10 * pcs[4] - 0.08 * pcs[5]
    return curved + linear


def event_probability(log_baseline: float, entry: float, linear_predictor: float) -> float:
    """P(event before ADMIN_END | entry, covariates) under the Gompertz baseline.

    The cumulative hazard from `entry` to `ADMIN_END` is
    `exp(log_baseline + lp) * (exp(r*(T-e0)) - exp(r*(entry-e0))) / r`, so the
    survival to the administrative end is its negative exponential.
    """
    scale = math.exp(log_baseline + linear_predictor) / GOMPERTZ_RATE
    upper = math.exp(GOMPERTZ_RATE * (ADMIN_END - ENTRY_LOW))
    lower = math.exp(GOMPERTZ_RATE * (entry - ENTRY_LOW))
    return -math.expm1(-scale * (upper - lower))


def solve_log_baseline(rows: list[dict]) -> float:
    """The intercept at which the EXPECTED event count is exactly `EVENTS`.

    Monotone increasing in `log_baseline`, from 0 events at `-inf` to `ROWS` at
    `+inf`, so a bisection on the realized covariates finds it exactly. The
    bracket is widened until it contains the target rather than assumed.
    """

    def expected(log_baseline: float) -> float:
        return sum(
            event_probability(log_baseline, row["entry_age"], row["lp"]) for row in rows
        )

    low, high = -30.0, 0.0
    while expected(low) > EVENTS:
        low -= 10.0
    while expected(high) < EVENTS:
        high += 10.0
    # 200 bisections take the bracket far below the resolution of a count, so
    # the stopping rule is the interval and not an iteration budget.
    while high - low > 1e-12 * max(1.0, abs(low)):
        mid = 0.5 * (low + high)
        if expected(mid) < EVENTS:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def build_rows() -> tuple[list[dict], float]:
    rng = Lcg(0x70_36_64_35_30_6B)  # "p6d50k"
    rows: list[dict] = []
    for _ in range(ROWS):
        pcs = [rng.normal() for _ in range(PC_COUNT)]
        sex = 1.0 if rng.uniform() < 0.5 else 0.0
        entry = ENTRY_LOW + (ENTRY_HIGH - ENTRY_LOW) * rng.uniform()
        rows.append(
            {
                "pcs": pcs,
                "sex": sex,
                "entry_age": entry,
                "lp": SEX_LOG_HAZARD * sex + smooth_log_hazard(pcs),
                "draw": rng.uniform(),
            }
        )
    log_baseline = solve_log_baseline(rows)
    for row in rows:
        # Inverse-transform the truncated Gompertz: the event time is where the
        # cumulative hazard from `entry` reaches `-log(1 - u)`; beyond the
        # administrative end the subject is censored there.
        scale = math.exp(log_baseline + row["lp"]) / GOMPERTZ_RATE
        lower = math.exp(GOMPERTZ_RATE * (row["entry_age"] - ENTRY_LOW))
        # `lower >= 1` and the first term is positive, so `target > 1` always
        # and the logarithm below is taken on a value bounded away from zero —
        # there is no degenerate branch to guard.
        target = -math.log1p(-row["draw"]) / scale + lower
        time = ENTRY_LOW + math.log(target) / GOMPERTZ_RATE
        if time < ADMIN_END:
            row["exit_age"], row["event"] = time, 1
        else:
            row["exit_age"], row["event"] = ADMIN_END, 0
    return rows, log_baseline


def request_document() -> dict:
    pcs = ", ".join(f"PC{i + 1}" for i in range(PC_COUNT))
    duchon = f"duchon({pcs}, centers={DUCHON_CENTERS})"
    return {
        "schema": "gam.fit-request",
        "schema_version": 1,
        "formula": f"Surv(entry_age, exit_age, event) ~ sex + {duchon}",
        "config": {
            "slope_formula": f"1 + {duchon}",
            "latent_measure": "global-empirical",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to write p6d50k.{csv,json} into")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, log_baseline = build_rows()
    realized = sum(row["event"] for row in rows)
    # The realized count is a Poisson-like draw about the solved expectation,
    # so its own standard error is sqrt(EVENTS). A draw outside three of those
    # is not this frame.
    standard_error = math.sqrt(float(EVENTS))
    assert abs(realized - EVENTS) <= 3.0 * standard_error, (
        f"realized events {realized} is more than 3 standard errors "
        f"({3.0 * standard_error:.1f}) from the frame's declared {EVENTS}"
    )

    header = ["entry_age", "exit_age", "event", "sex"] + [
        f"PC{i + 1}" for i in range(PC_COUNT)
    ]
    csv_path = out / "p6d50k.csv"
    with csv_path.open("w", newline="\n") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            fields = [
                f"{row['entry_age']:.17e}",
                f"{row['exit_age']:.17e}",
                str(row["event"]),
                "M" if row["sex"] > 0.5 else "F",
            ] + [f"{value:.17e}" for value in row["pcs"]]
            handle.write(",".join(fields) + "\n")

    json_path = out / "p6d50k.json"
    json_path.write_text(json.dumps(request_document(), indent=2) + "\n")

    print(
        f"p6d50k: rows={ROWS} events={realized} ({100.0 * realized / ROWS:.3f}%) "
        f"log_baseline={log_baseline:.6f} "
        f"csv={csv_path} ({os.path.getsize(csv_path)} bytes) json={json_path}"
    )


if __name__ == "__main__":
    main()
