"""Predictive-interval coverage: pyGAM against gamfit's posterior and conformal routes.

    python -m bench.pygam_compare.conformal_coverage [--reps R] [--ns 30,100,1000]
        [--dgps correct,...] [--workers W] [--out bench/pygam_audit/conformal_coverage.md]

For every data-generating process (DGP) and training size ``n`` the script
draws ``R`` seeded replicates. Each replicate draws ``n`` training rows and
``m`` fresh test rows from the DGP, fits every method on the training rows
and checks whether each test response lies in that method's 90% predictive
interval (``alpha = 0.1``). The methods are

| method                | interval |
|-----------------------|----------|
| `pygam`               | pyGAM `gridsearch` then `prediction_intervals(width=0.9)`; the GLM classes have no `prediction_intervals`, so for binomial and Poisson the family's plug-in 5%/95% quantiles at `predict_mu` |
| `gamfit_posterior`    | `predict(interval=0.9, observation_interval=True)`: `observation_lower/upper` |
| `gamfit_full_conf`    | `predict(interval="conformal", training_data=train)`: the exact full-conformal set |
| `gamfit_split_conf`   | fit on the first half, `predict(interval="conformal", calibration=second half)` |

Every method fits the same basis (``s(x, k=10)``; pyGAM ``s(0, n_splines=10)``).

DGPs (``x ~ U(0, 1)``, ``f(x) = sin(2 pi x)``):

| DGP            | response | model fitted |
|----------------|----------|--------------|
| `correct`      | `f(x) + N(0, 0.5^2)` | Gaussian `s(x)` |
| `misspecified` | `f(x) + 8 (x2 - 1/2)^2 + N(0, 0.3^2)`, `x2 ~ U(0, 1)` omitted | Gaussian `s(x)` |
| `heteroscedastic` | `f(x) + N(0, (0.1 + 0.9 x)^2)` | Gaussian `s(x)` |
| `heavy_tails`  | `f(x) + 0.3 t_2` | Gaussian `s(x)` |
| `binomial`     | `Bernoulli(logistic(1.5 f(x)))` | Bernoulli-logit `s(x)` |
| `poisson`      | `Poisson(exp(1 + f(x)))` | Poisson-log `s(x)` |

Coverage is the mean over replicates of the per-replicate fraction of covered
test rows; the Monte Carlo standard error (MCSE) is the standard deviation of
that fraction over replicates divided by ``sqrt(R)`` (replicates are
independent). Width is the median over all test rows of ``upper - lower`` for
the Gaussian DGPs and of the number of integer responses inside the band for
the binomial and Poisson DGPs (so an interval method and a conformal set are
measured on the same support).

A cell is **nominal** when ``1 - alpha - 2 MCSE <= coverage <= 1 - alpha +
1/(n_cal + 1) + 2 MCSE``, where ``n_cal`` is the number of points the method
ranks (``n`` for full conformal, the calibration half for split conformal):
conformal coverage is at least ``1 - alpha`` and exceeds it by at most the
``1/(n_cal + 1)`` rank granularity (the discrete full-conformal set breaks ties
by a seeded uniform, so it is exact rather than conservative). The same band
is applied to the model-based methods. **under** and **over** mark a miss.
A replicate whose fit or interval raises is recorded as an error of that
method and counted in the ``ok/errors`` column; the coverage and verdict are
over the completed replicates.

Bench-only dependency: ``pygam`` (bench/pygam_compare/requirements.txt).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np

ALPHA = 0.1
LEVEL = 1.0 - ALPHA
K = 10
M_TEST = 10
DGPS: tuple[str, ...] = (
    "correct",
    "misspecified",
    "heteroscedastic",
    "heavy_tails",
    "binomial",
    "poisson",
)
DISCRETE = {"binomial", "poisson"}
METHODS: tuple[str, ...] = ("pygam", "gamfit_posterior", "gamfit_full_conf", "gamfit_split_conf")
THREAD_VARS = (
    "RAYON_NUM_THREADS",
    "MATMUL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _draw(dgp: str, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    x = rng.uniform(0.0, 1.0, n)
    f = np.sin(2.0 * np.pi * x)
    if dgp == "correct":
        y = f + rng.normal(0.0, 0.5, n)
    elif dgp == "misspecified":
        x2 = rng.uniform(0.0, 1.0, n)
        y = f + 8.0 * (x2 - 0.5) ** 2 + rng.normal(0.0, 0.3, n)
    elif dgp == "heteroscedastic":
        y = f + rng.normal(0.0, 1.0, n) * (0.1 + 0.9 * x)
    elif dgp == "heavy_tails":
        y = f + 0.3 * rng.standard_t(2.0, n)
    elif dgp == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.5 * f))).astype(float)
    elif dgp == "poisson":
        y = rng.poisson(np.exp(1.0 + f)).astype(float)
    else:
        raise ValueError(f"unknown DGP {dgp!r}")
    return {"x": x, "y": y}


def _family(dgp: str) -> str:
    return {"binomial": "binomial", "poisson": "poisson"}.get(dgp, "gaussian")


def _seed(dgp: str, n: int, rep: int) -> int:
    return (DGPS.index(dgp) * 1_000_003 + n) * 100_003 + rep


def _width(dgp: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if dgp in DISCRETE:
        # Integer responses inside [lo, hi]; the binomial support is {0, 1}.
        top = 1.0 if dgp == "binomial" else np.inf
        lo_i = np.ceil(np.maximum(lo, 0.0))
        hi_i = np.floor(np.minimum(hi, top))
        return np.maximum(hi_i - lo_i + 1.0, 0.0)
    return hi - lo


def _pygam(dgp, train, test):
    import pygam

    cls = {"binomial": pygam.LogisticGAM, "poisson": pygam.PoissonGAM}.get(dgp, pygam.LinearGAM)
    gam = cls(pygam.s(0, n_splines=K))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam.gridsearch(train["x"][:, None], train["y"], progress=False)
        if dgp not in DISCRETE:
            band = np.asarray(gam.prediction_intervals(test["x"][:, None], width=LEVEL), dtype=float)
            return band[:, 0], band[:, 1], 0
        # pyGAM's GLM classes have no prediction_intervals; the band a pyGAM
        # user can form is the family's plug-in quantiles at the fitted mean.
        from scipy import stats

        mu = np.asarray(gam.predict_mu(test["x"][:, None]), dtype=float)
        dist = stats.bernoulli(mu) if dgp == "binomial" else stats.poisson(mu)
        return dist.ppf(ALPHA / 2.0), dist.ppf(1.0 - ALPHA / 2.0), 0


def _gamfit_fit(dgp, rows):
    import gamfit

    return gamfit.fit(rows, f"y ~ s(x, k={K})", family=_family(dgp))


def _gamfit_posterior(dgp, train, test):
    model = _gamfit_fit(dgp, train)
    pred = model.predict({"x": test["x"]}, interval=LEVEL, observation_interval=True)
    return np.asarray(pred["observation_lower"]), np.asarray(pred["observation_upper"]), 0


def _gamfit_full(dgp, train, test):
    model = _gamfit_fit(dgp, train)
    pred = model.predict(
        {"x": test["x"]}, interval="conformal", training_data=train, conformal_level=LEVEL
    )
    multi = int(np.sum(np.asarray(pred["conformal_set_components"]) > 1))
    return np.asarray(pred["posterior_mean_lower"]), np.asarray(pred["posterior_mean_upper"]), multi


def _gamfit_split(dgp, train, test):
    half = len(train["y"]) // 2
    fit_rows = {k: v[:half] for k, v in train.items()}
    cal_rows = {k: v[half:] for k, v in train.items()}
    model = _gamfit_fit(dgp, fit_rows)
    pred = model.predict(
        {"x": test["x"]}, interval="conformal", calibration=cal_rows, conformal_level=LEVEL
    )
    return np.asarray(pred["posterior_mean_lower"]), np.asarray(pred["posterior_mean_upper"]), 0


RUNNERS = {
    "pygam": _pygam,
    "gamfit_posterior": _gamfit_posterior,
    "gamfit_full_conf": _gamfit_full,
    "gamfit_split_conf": _gamfit_split,
}


def _rep(task: tuple[str, int, int, int]) -> dict:
    dgp, n, rep, m_test = task
    rng = np.random.default_rng(_seed(dgp, n, rep))
    train = _draw(dgp, rng, n)
    test = _draw(dgp, rng, m_test)
    out: dict = {"dgp": dgp, "n": n, "rep": rep}
    for method in METHODS:
        t0 = time.perf_counter()
        try:
            lo, hi, multi = RUNNERS[method](dgp, train, test)
            covered = (lo <= test["y"]) & (test["y"] <= hi)
            out[method] = {
                "status": "ok",
                "coverage": float(np.mean(covered)),
                "widths": _width(dgp, lo, hi).tolist(),
                "multi": multi,
                "seconds": time.perf_counter() - t0,
            }
        except Exception as exc:  # recorded, reported as a failure of that method
            out[method] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return out


@dataclass
class CellStats:
    ok: int
    errors: int
    coverage: float
    mcse: float
    width: float
    multi: int
    seconds: float
    first_error: str


def _summarise(records: list[dict], method: str) -> CellStats:
    ok = [r[method] for r in records if r[method]["status"] == "ok"]
    errs = [r[method] for r in records if r[method]["status"] != "ok"]
    if not ok:
        return CellStats(0, len(errs), math.nan, math.nan, math.nan, 0, math.nan, errs[0]["error"])
    cov = np.array([o["coverage"] for o in ok])
    widths = np.concatenate([np.asarray(o["widths"], dtype=float) for o in ok])
    mcse = float(np.std(cov, ddof=1) / math.sqrt(len(cov))) if len(cov) > 1 else math.nan
    return CellStats(
        ok=len(ok),
        errors=len(errs),
        coverage=float(np.mean(cov)),
        mcse=mcse,
        width=float(np.median(widths)),
        multi=int(sum(o["multi"] for o in ok)),
        seconds=float(np.median([o["seconds"] for o in ok])),
        first_error=errs[0]["error"] if errs else "",
    )


def _ranked_points(method: str, n: int) -> int:
    return n - n // 2 if method == "gamfit_split_conf" else n


def verdict(stats: CellStats, method: str, n: int) -> str:
    if stats.ok < 2:
        return "**error**"
    lower = LEVEL - 2.0 * stats.mcse
    upper = LEVEL + 1.0 / (_ranked_points(method, n) + 1) + 2.0 * stats.mcse
    if stats.coverage < lower:
        return "**under**"
    if stats.coverage > upper:
        return "**over**"
    return "nominal"


def render(records: list[dict], reps: int, m_test: int, ns: list[int], dgps: list[str]) -> str:
    lines = [
        "# Predictive-interval coverage: pyGAM vs gamfit (alpha = 0.1)",
        "",
        "Generated by `python -m bench.pygam_compare.conformal_coverage"
        f" --reps {reps} --ns {','.join(map(str, ns))}`; see that module's docstring"
        " for the DGPs, the methods and the verdict band. Each cell is"
        f" {reps} seeded replicates of n training rows and {m_test} test rows;"
        " coverage is of a fresh response by the method's 90% predictive interval.",
        "",
        "A verdict is **nominal** when `0.9 - 2 MCSE <= coverage <= 0.9 + 1/(n_cal + 1)"
        " + 2 MCSE`; `n_cal` is n for full conformal and the calibration half for split"
        " conformal. Width is the median band width (Gaussian DGPs) or the median number"
        " of integer responses in the band (binomial, Poisson). `multi` counts test rows"
        " whose full-conformal set has more than one component (its envelope is then"
        " a superset). Coverage and the verdict are over the replicates the method"
        " completed; `ok/errors` counts the rest, whose first error is in an HTML"
        " comment on the row.",
        "",
        "| DGP | n | method | coverage | MCSE | median width | verdict | ok/errors | median s |",
        "|-----|---|--------|----------|------|--------------|---------|-----------|----------|",
    ]
    wins = []
    for dgp in dgps:
        for n in ns:
            cell = [r for r in records if r["dgp"] == dgp and r["n"] == n]
            if not cell:
                continue
            verdicts = {}
            for method in METHODS:
                s = _summarise(cell, method)
                v = verdict(s, method, n)
                verdicts[method] = v
                extra = f" (multi {s.multi})" if method == "gamfit_full_conf" and s.multi else ""
                lines.append(
                    f"| {dgp} | {n} | `{method}` | {s.coverage:.4f} | {s.mcse:.4f} |"
                    f" {s.width:.3f} | {v}{extra} | {s.ok}/{s.errors} | {s.seconds:.3f} |"
                )
                if s.errors:
                    lines[-1] += f" <!-- {s.first_error} -->"
            wins.append((dgp, n, verdicts["pygam"], verdicts["gamfit_full_conf"]))
    lines += [
        "",
        "## Cells where pyGAM misses nominal coverage",
        "",
        "| DGP | n | pyGAM | gamfit full conformal |",
        "|-----|---|-------|------------------------|",
    ]
    missed = [w for w in wins if w[2] != "nominal"]
    for dgp, n, pv, fv in missed:
        lines.append(f"| {dgp} | {n} | {pv} | {fv} |")
    held = sum(1 for w in missed if w[3] == "nominal")
    lines += [
        "",
        f"gamfit full conformal is nominal in {held} of the {len(missed)} cells where pyGAM"
        " misses.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--ns", default="30,100,1000")
    ap.add_argument("--dgps", default=",".join(DGPS))
    ap.add_argument("--m-test", type=int, default=M_TEST)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="bench/pygam_audit/conformal_coverage.md")
    args = ap.parse_args(argv)
    ns = [int(v) for v in args.ns.split(",")]
    dgps = [d for d in args.dgps.split(",") if d]
    for d in dgps:
        if d not in DGPS:
            ap.error(f"unknown DGP {d!r}; choose from {', '.join(DGPS)}")
    # One thread per worker process: the replicates are the parallelism.
    for var in THREAD_VARS:
        os.environ[var] = "1"
    tasks = [(d, n, r, args.m_test) for n in ns for d in dgps for r in range(args.reps)]
    records: list[dict] = []
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=200) as pool:
        for i, rec in enumerate(pool.imap_unordered(_rep, tasks, chunksize=4), 1):
            records.append(rec)
            if i % 500 == 0 or i == len(tasks):
                print(
                    f"[{i}/{len(tasks)}] {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True
                )
    records.sort(key=lambda r: (dgps.index(r["dgp"]), r["n"], r["rep"]))
    report = render(records, args.reps, args.m_test, ns, dgps)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
