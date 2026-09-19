"""Tabulate the null-calibration and power runs written by null_calibration.py.

Usage: python analyze.py results/null200.jsonl [more.jsonl ...]

Per (family, n, effect) cell it prints the rejection rate at 0.10/0.05/0.01,
the Monte-Carlo standard error at 0.05, a KS test of p / 0.5 against U(0, 1)
over the replications with p < 0.5, the share of p > 0.99 (terms REML switched
off), and the median edf / ref_df. A null cell whose size at 0.05 or 0.01
exceeds alpha + 2 * MCSE is flagged ANTI.

--ablation additionally recomputes each p-value against a reference df of
max(1, round(edf)) (the rank the statistic sums, without the trace-ratio
floor), to show what that floor buys.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ESTIMATED_SCALE = {"gaussian", "gamma", "tweedie", "beta"}


def load(paths):
    cells, errors = defaultdict(list), defaultdict(list)
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            key = (r["family"], r["n"], r["effect"])
            if "error" in r:
                errors[key].append(r["error"])
            elif r.get("p") is None:
                errors[key].append("p_value absent")
            else:
                cells[key].append((r["p"], r["edf"], r["ref_df"], r["chi_sq"]))
    return cells, errors


def sizes(p):
    return [(p <= a).mean() for a in (0.10, 0.05, 0.01)]


def ks_below_half(p):
    low = p[p < 0.5] / 0.5
    return stats.kstest(low, "uniform").pvalue if len(low) > 5 else float("nan")


def main(argv):
    ablation = "--ablation" in argv
    cells, errors = load([a for a in argv if not a.startswith("--")])
    head = f"{'family':18s} {'n':>5s} {'eff':>4s} {'m':>4s} {'err':>3s} {'s.10':>6s} {'s.05':>6s} {'s.01':>6s} {'mcse05':>6s} {'KS<.5':>6s} {'p>.99':>6s} {'edf':>5s} {'refdf':>5s}"
    print(head)
    for key in sorted(set(cells) | set(errors)):
        a = np.array(cells.get(key, []), dtype=float).reshape(-1, 4)
        p, edf, _, stat = a.T
        m = len(p)
        if m == 0:
            print(key, "no p-values", errors[key][:1])
            continue
        s10, s05, s01 = sizes(p)
        mcse05 = np.sqrt(0.05 * 0.95 / m)
        flag = ""
        if key[2] == 0.0 and (s05 > 0.05 + 2 * mcse05 or s01 > 0.01 + 2 * np.sqrt(0.01 * 0.99 / m)):
            flag = " ANTI"
        print(f"{key[0]:18s} {key[1]:5d} {key[2]:4.1f} {m:4d} {len(errors.get(key, [])):3d} "
              f"{s10:6.3f} {s05:6.3f} {s01:6.3f} {mcse05:6.3f} {ks_below_half(p):6.3f} "
              f"{(p > 0.99).mean():6.3f} {np.median(a[:, 1]):5.2f} {np.median(a[:, 2]):5.2f}{flag}")
        if ablation:
            rank = np.maximum(1.0, np.round(edf))
            if key[0] in ESTIMATED_SCALE:
                # residual df is not stored per row; n - edf(x2) - edf(x1 + intercept) ~ n - edf - 10
                alt = stats.f.sf(stat / rank, rank, key[1] - edf - 10)
            else:
                alt = stats.chi2.sf(stat, rank)
            a10, a05, a01 = sizes(alt)
            print(f"{'  ref_df=rank':18s} {'':5s} {'':4s} {'':4s} {'':3s} {a10:6.3f} {a05:6.3f} {a01:6.3f}")
    for key, errs in sorted(errors.items()):
        if errs:
            print("fit errors", key, len(errs), "first:", errs[0][:160])


if __name__ == "__main__":
    main(sys.argv[1:])
