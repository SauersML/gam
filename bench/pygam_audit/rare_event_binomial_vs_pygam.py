"""Rare-event logistic GAM fits: gamfit vs pyGAM timing (pyGAM audit F18).

A synthetic analogue of the ISLR ``Default`` credit data: ~3% positives whose
log-odds rise steeply with a skewed covariate,

    balance ~ N(835, 480) truncated at 0,   income ~ N(33500, 13300),
    student ~ Bernoulli(0.3),               logit p = -10.65 + 0.0055 * balance.

The steep, rare-event response is what makes the undamped Newton step from the
prevalence-intercept seed overshoot (it pushes eta toward +11 on the
high-balance rows), so the inner P-IRLS step control sets the cost.

Reported per model and n: wall and CPU seconds for gamfit (REML-selected
smoothing, certified fit) and for pyGAM (``LogisticGAM`` with its default
fixed ``lam``, a single P-IRLS fit), plus gamfit's convergence certificate and
total edf. pyGAM does no smoothing-parameter selection here, so the ratio
overstates the like-for-like gap; it is the ratio the audit reported.

Run: ``python bench/pygam_audit/rare_event_binomial_vs_pygam.py [--n 2000 10000]``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import gamfit

try:
    from pygam import LogisticGAM, f, s
except ImportError:  # pragma: no cover - comparison needs pyGAM installed
    LogisticGAM = None

MODELS = (
    ("s(balance)", ("balance",)),
    ("s(balance) + s(income)", ("balance", "income")),
    ("student + s(balance) + s(income)", ("student", "balance", "income")),
)


def rare_event_data(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    balance = np.maximum(rng.normal(835.0, 480.0, n), 0.0)
    income = rng.normal(33500.0, 13300.0, n)
    student = rng.random(n) < 0.3
    p = 1.0 / (1.0 + np.exp(10.65 - 0.0055 * balance))
    y = (rng.random(n) < p).astype(float)
    return {
        "student": np.where(student, "s1", "s0"),
        "balance": balance,
        "income": income,
        "y": y,
    }


def time_gamfit(data: dict[str, np.ndarray], formula: str) -> tuple[float, float, str]:
    wall, cpu = time.perf_counter(), time.process_time()
    model = gamfit.fit(data, formula, family="binomial")
    summary = model.summary()
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    note = f"certified={summary.convergence.get('certified')} edf={summary.edf_total:.2f}"
    return wall, cpu, note


def time_pygam(data: dict[str, np.ndarray], columns: tuple[str, ...]) -> tuple[float, float]:
    cols, terms = [], None
    for j, name in enumerate(columns):
        if name == "student":
            cols.append((data[name] == "s1").astype(float))
            term = f(j)
        else:
            cols.append(data[name])
            term = s(j)
        terms = term if terms is None else terms + term
    x = np.column_stack(cols)
    wall, cpu = time.perf_counter(), time.process_time()
    LogisticGAM(terms).fit(x, data["y"])
    return time.perf_counter() - wall, time.process_time() - cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs="+", default=[2000, 10000])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    header = f"{'model':34s} {'n':>6s} {'positives':>9s} {'gamfit wall/cpu':>16s} {'pyGAM wall/cpu':>15s}  gamfit fit"
    print(header)
    print("-" * len(header))
    for n in args.n:
        data = rare_event_data(n, args.seed)
        positives = int(data["y"].sum())
        for label, columns in MODELS:
            formula = "y ~ " + " + ".join(
                name if name == "student" else f"s({name})" for name in columns
            )
            g_wall, g_cpu, note = time_gamfit(data, formula)
            if LogisticGAM is None:
                pg = "n/a"
            else:
                p_wall, p_cpu = time_pygam(data, columns)
                pg = f"{p_wall:.2f}/{p_cpu:.2f}"
            print(
                f"{label:34s} {n:6d} {positives:9d} {g_wall:7.2f}/{g_cpu:<8.2f} {pg:>15s}  {note}",
                flush=True,
            )


if __name__ == "__main__":
    main()
