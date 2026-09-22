#!/usr/bin/env python3
"""Two equal-size event-history cohorts from one generator, for gam#2986.

Why this exists
---------------
gam#2986 measured two cohorts of the same size, from one generator at seeds 11
and 12, taking 81.6 s and over 420 s to fit. The three things it leaves open are
whether seed 12 spent that on more outer iterations, on more inner iterations, or
on costlier evaluations. `gam fit-events -v` now reports that decomposition per
ladder rung (`rung_cost`, d1b60b8e4f), so the measurement needs only the two
cohorts, and this writes them.

The generator the issue names is `_simulate(120, seed, "s")` in evh-artifact's
`tests/test_event_history_saved_predictor_parity.py`, which is not in this
repository. This is a reconstruction from the parameters the issue STATES: 120
subjects, three marks (visit recurrent, disease once-only, death terminal)
sharing one standard-normal static frailty, baseline rates 0.8, 0.25 and 0.15,
loadings 1.0, 1.2 and 0.9, a score effect of 0.3 on a standard-normal covariate
`g`, and a follow-up of 4.0. It is not claimed to reproduce evh-artifact's draws.
It does not have to: the question is seed 11 against seed 12 under ONE generator,
and only the absolute seconds are tied to which generator was used. If
evh-artifact is available on the machine, use its `_simulate` instead and this
file is unnecessary.

Why the draws are exact
-----------------------
The frailty is static and the covariate is constant over follow-up, so every
mark's intensity is constant in time: `lambda_d = rate_d * exp(0.3 g + a_d z)`.
A constant-intensity process needs no thinning — the terminal and once-only
times are single exponential draws and the recurrent mark is a Poisson process
sampled by its exponential gaps — so the cohort is an exact draw from the stated
model rather than an approximation of it, and two seeds differ only in the draw.

Determinism
-----------
One `numpy.random.default_rng(seed)` per cohort, consumed in a fixed order: the
covariate column `g`, then the frailty column `z`, then per subject in index
order the death time, the disease time, and the recurrent gaps until the first
one that lands at or past the subject's exit. The number of gap draws varies by
subject, which is why the order is stated: it is what makes a rerun reproduce
the file byte for byte.

Output
------
`<directory>/seed<N>/{subjects,events,covariates}.csv` in the three shapes
`gam fit-events` reads: subjects `id,entry,exit`; events `id,time,mark`;
covariates `id,start,g`. Events are written in time order per subject. A death
is recorded at the subject's exit when it falls inside the follow-up; a visit or
a diagnosis is recorded only strictly before the exit, because the cohort reader
takes an event after the exit as outside the record.

Usage
-----
    python3 experiments/evh2986_gen.py /scratch.global/sauer354/evh2986
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

SUBJECTS = 120
FOLLOW_UP = 4.0
SCORE_EFFECT = 0.3
MARKS = ("visit", "disease", "death")
RATES = (0.8, 0.25, 0.15)
LOADINGS = (1.0, 1.2, 0.9)
SEEDS = (11, 12)


def simulate(seed: int) -> tuple[list, list, list]:
    """One cohort at `seed`: its subject, event and covariate rows."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(SUBJECTS)
    z = rng.standard_normal(SUBJECTS)
    subjects, events, covariates = [], [], []
    for i in range(SUBJECTS):
        ident = f"s{i}"
        visit_rate, disease_rate, death_rate = (
            rate * np.exp(SCORE_EFFECT * g[i] + loading * z[i])
            for rate, loading in zip(RATES, LOADINGS)
        )
        death = rng.exponential(1.0 / death_rate)
        disease = rng.exponential(1.0 / disease_rate)
        exit_time = min(FOLLOW_UP, death)
        spell = []
        visit = rng.exponential(1.0 / visit_rate)
        while visit < exit_time:
            spell.append((visit, "visit"))
            visit += rng.exponential(1.0 / visit_rate)
        if disease < exit_time:
            spell.append((disease, "disease"))
        if death < FOLLOW_UP:
            spell.append((death, "death"))
        spell.sort()
        subjects.append((ident, 0.0, exit_time))
        events.extend((ident, time, mark) for time, mark in spell)
        covariates.append((ident, 0.0, float(g[i])))
    return subjects, events, covariates


def write(path: pathlib.Path, header: tuple[str, ...], rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(directory: str) -> None:
    root = pathlib.Path(directory)
    for seed in SEEDS:
        subjects, events, covariates = simulate(seed)
        out = root / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        write(out / "subjects.csv", ("id", "entry", "exit"), subjects)
        write(out / "events.csv", ("id", "time", "mark"), events)
        write(out / "covariates.csv", ("id", "start", "g"), covariates)
        counts = {mark: sum(1 for row in events if row[2] == mark) for mark in MARKS}
        deaths = counts["death"]
        print(
            f"seed {seed}: {len(subjects)} subjects, {len(events)} events "
            f"({', '.join(f'{mark} {counts[mark]}' for mark in MARKS)}), "
            f"{deaths} terminated inside the follow-up, written to {out}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <directory>")
    main(sys.argv[1])
