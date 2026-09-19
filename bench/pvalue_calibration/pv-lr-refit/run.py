"""Seeded Monte Carlo study of `Model.smooth_significance` (lane pv-lr-refit).

Measures, per cell, over the pyGAM audit's data-generating processes
(`bench/pygam_audit`, inference cells `gauss`, `binom`, `pois`):

* availability: the share of converged fits whose every smooth row carries
  exactly one of `p_value`, `p_value_upper_bound`, `unavailable_reason`, and
  how many of each;
* size of the null term `s(x2)` (no effect in the DGP) at alpha .10/.05/.01,
  with the Monte Carlo standard error sqrt(a(1-a)/R), and a one-sample KS test
  of its p-values against U(0, 1);
* power for `s(x1)` (strong) and `s(x3)` (weak) at the same alphas, counting
  a `p < p_value_upper_bound` row as rejecting at every alpha above the bound.

Usage:
    python run.py <cell> <first_rep> <last_rep_exclusive> <out.jsonl> [<stall_seconds>]
    python run.py --summarize <out.jsonl> [<out.jsonl> ...]

Each replicate is seeded `default_rng(1000 + rep)`, the audit's convention, so
a replicate here is the same dataset as the audit's replicate of that number.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

TWO_PI = 2 * np.pi
CELLS = {
    # name: (family, n, intercept, a1, a3, sigma) — the audit's cells.
    "gauss": ("gaussian", 200, 0.0, 1.0, 0.30, 1.0),
    "binom": ("binomial", 400, 0.0, 1.5, 0.60, None),
    "pois": ("poisson", 200, 0.5, 0.8, 0.25, None),
}
FORMULA = "y ~ s(x1) + s(x2) + s(x3)"
ALPHAS = (0.10, 0.05, 0.01)
KEYS = ("p_value", "p_value_upper_bound", "unavailable_reason")


def dataset(cell: str, rep: int) -> dict[str, np.ndarray]:
    fam, n, b0, a1, a3, sigma = CELLS[cell]
    rng = np.random.default_rng(1000 + rep)
    X = rng.uniform(0, 1, (n, 3))
    eta = b0 + a1 * np.sin(TWO_PI * X[:, 0]) + a3 * np.cos(TWO_PI * X[:, 2])
    if fam == "gaussian":
        y = eta + rng.normal(0, sigma, n)
    elif fam == "binomial":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)


def replicate(cell: str, rep: int) -> dict:
    import gamfit

    warnings.simplefilter("ignore")
    data = dataset(cell, rep)
    record: dict = {"cell": cell, "rep": rep}
    t0 = time.time()
    try:
        model = gamfit.fit(data, FORMULA, family=CELLS[cell][0])
    except Exception as exc:  # a fit that did not converge is not a fit
        record["fit_error"] = str(exc)[:300]
    else:
        try:
            rows = model.smooth_significance(data)
        except Exception as exc:
            record["raised"] = str(exc)[:300]
        else:
            record["rows"] = [
                {
                    "name": r["name"],
                    **{k: r.get(k) for k in KEYS},
                    "statistic_lr": r.get("statistic_lr"),
                    "p_value_bound": r.get("p_value_bound"),
                }
                for r in rows
            ]
    record["seconds"] = round(time.time() - t0, 2)
    return record


def _serve(conn) -> None:
    while (job := conn.recv()) is not None:
        conn.send(replicate(*job))


def run(cell: str, first: int, last: int, out_path: str, stall_seconds: float | None) -> None:
    """Run replicates `first..last`, appending one JSON line each.

    With `stall_seconds`, each replicate runs in a worker process and one that
    has not returned by then is recorded as `stalled` and the worker replaced.
    That is the harness finishing the study, not a result: the summary reports
    stalled replicates separately and never scores them.
    """
    with open(out_path, "a") as out:
        if stall_seconds is None:
            for rep in range(first, last):
                out.write(json.dumps(replicate(cell, rep)) + "\n")
                out.flush()
            return
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        worker = None
        for rep in range(first, last):
            if worker is None:
                parent, child = ctx.Pipe()
                worker = ctx.Process(target=_serve, args=(child,), daemon=True)
                worker.start()
            parent.send((cell, rep))
            if parent.poll(stall_seconds):
                record = parent.recv()
            else:
                worker.kill()
                worker.join()
                worker = None
                record = {"cell": cell, "rep": rep, "stalled": stall_seconds, "seconds": stall_seconds}
            out.write(json.dumps(record) + "\n")
            out.flush()
        if worker is not None:
            parent.send(None)
            worker.join()


def kind(row: dict) -> str | None:
    present = [k for k in KEYS if row.get(k) is not None]
    return present[0] if len(present) == 1 else None


def rejects(row: dict, alpha: float) -> bool:
    if row.get("p_value") is not None:
        return row["p_value"] < alpha
    bound = row.get("p_value_upper_bound")
    return bound is not None and bound < alpha


def ks_uniform(p: np.ndarray, alternative: str = "two-sided") -> tuple[float, float]:
    from scipy import stats

    res = stats.kstest(p, "uniform", alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def summarize(paths: list[str]) -> None:
    records: dict[str, list[dict]] = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                records.setdefault(rec["cell"], []).append(rec)
    for cell, recs in sorted(records.items()):
        # A replicate re-run after a stall supersedes the stall record.
        by_rep: dict[int, dict] = {}
        for r in recs:
            if "stalled" not in r or r["rep"] not in by_rep:
                by_rep[r["rep"]] = r
        recs = sorted(by_rep.values(), key=lambda r: r["rep"])
        stalled = [r for r in recs if "stalled" in r]
        fitted = [r for r in recs if "fit_error" not in r and "stalled" not in r]
        raised = [r for r in fitted if "raised" in r]
        rows = [row for r in fitted if "rows" in r for row in r["rows"]]
        kinds = [kind(row) for row in rows]
        print(f"## {cell}: {len(recs)} reps, {len(fitted)} converged fits, "
              f"{len(recs) - len(fitted) - len(stalled)} fit errors, {len(raised)} raised, "
              f"{len(stalled)} stalled (no return within the harness limit; not scored)")
        if stalled:
            print(f"stalled reps: {[r['rep'] for r in stalled]}")
        malformed = sum(k is None for k in kinds)
        print(f"rows: {len(rows)}; p_value {kinds.count('p_value')}, "
              f"p_value_upper_bound {kinds.count('p_value_upper_bound')}, "
              f"unavailable_reason {kinds.count('unavailable_reason')}, "
              f"malformed {malformed}")
        reasons: dict[str, int] = {}
        for row in rows:
            if row.get("unavailable_reason"):
                reasons[row["unavailable_reason"]] = reasons.get(row["unavailable_reason"], 0) + 1
        if reasons:
            print(f"reasons: {reasons}")
        for name, role in (("s(x2)", "size (null term)"), ("s(x1)", "power"), ("s(x3)", "power")):
            term = [row for row in rows if row["name"] == name and kind(row) != "unavailable_reason"]
            if not term:
                continue
            R = len(term)
            cells = []
            for a in ALPHAS:
                rate = sum(rejects(row, a) for row in term) / R
                mcse = np.sqrt(a * (1 - a) / R) if role.startswith("size") else np.sqrt(rate * (1 - rate) / R)
                cells.append(f"@{a:.2f} {rate:.4f} (MCSE {mcse:.4f})")
            line = f"{name} {role}, R={R}: " + ", ".join(cells)
            if role.startswith("size"):
                p = np.array([row["p_value"] for row in term if row.get("p_value") is not None])
                d, pk = ks_uniform(p)
                # `greater`: the empirical CDF above the uniform's somewhere,
                # i.e. the p-values anti-conservative at some level. A penalized
                # null term shrunk to W = 0 publishes p = 1, a conservative atom
                # the two-sided test also counts against the reference.
                dg, pg = ks_uniform(p, "greater")
                line += (f"; KS two-sided D={d:.4f} p={pk:.3f}, "
                         f"anti-conservative D+={dg:.4f} p={pg:.3f} (n={p.size}); "
                         f"P(p=1)={np.mean(p == 1.0):.3f}")
            print(line)
        secs = [r["seconds"] for r in recs]
        print(f"seconds per rep: median {np.median(secs):.1f}, max {max(secs):.1f}\n")


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        summarize(sys.argv[2:])
    else:
        stall = float(sys.argv[5]) if len(sys.argv) > 5 else None
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], stall)
