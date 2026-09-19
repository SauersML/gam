"""Tabulate the runs written by null_calibration.py.

Usage: python analyze.py results/*.jsonl

Per (cell, family, levels, balanced, effect) it prints the number of p-values
and of typed-unavailable / failed fits, the rejection rate at 0.10/0.05/0.01,
the half-width 2 * sqrt(alpha (1 - alpha) / m), the two-sided KS p-value of
the p-values against U(0, 1), the share of p > 0.99, and the median ref_df. A
null cell (effect 0) is flagged ANTI when its size exceeds alpha + half-width
at any alpha, CONS when it falls below alpha - half-width at any alpha (a
conservative p-value is miscalibrated too), and KS when the KS p-value is
below 0.01.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def load(paths):
    cells, missing = defaultdict(list), defaultdict(list)
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            key = (r["cell"], r["family"], r["levels"], r["balanced"], r["n"], r["effect"])
            if "error" in r:
                missing[key].append("error: " + r["error"])
            elif r.get("p") is None:
                missing[key].append("unavailable: " + str(r.get("unavailable")))
            else:
                cells[key].append((r["p"], r["ref_df"]))
    return cells, missing


ALPHAS = (0.10, 0.05, 0.01)


def half_width(alpha, m):
    return 2 * np.sqrt(alpha * (1 - alpha) / m)


def main(paths):
    cells, missing = load(paths)
    print(f"{'cell':4s} {'family':9s} {'L':>4s} {'bal':>3s} {'n':>5s} {'eff':>4s} {'m':>4s} "
          f"{'miss':>4s} {'s.10':>6s} {'s.05':>6s} {'s.01':>6s} {'hw.05':>6s} {'hw.01':>6s} "
          f"{'KS':>6s} {'p>.99':>6s} {'refdf':>6s}")
    for key in sorted(set(cells) | set(missing)):
        a = np.array(cells.get(key, []), dtype=float).reshape(-1, 2)
        p, ref_df = a.T
        m = len(p)
        if m == 0:
            print(key, "no p-values", missing[key][:1])
            continue
        sizes = [(p <= alpha).mean() for alpha in ALPHAS]
        s10, s05, s01 = sizes
        ks = stats.kstest(p, "uniform").pvalue
        flag = ""
        if key[5] == 0.0:
            if any(s > a + half_width(a, m) for s, a in zip(sizes, ALPHAS)):
                flag += " ANTI"
            if any(s < a - half_width(a, m) for s, a in zip(sizes, ALPHAS)):
                flag += " CONS"
            if ks < 0.01:
                flag += " KS"
        print(f"{key[0]:4s} {key[1]:9s} {key[2]:4d} {'B' if key[3] else 'U':>3s} {key[4]:5d} "
              f"{key[5]:4.2f} {m:4d} {len(missing.get(key, [])):4d} {s10:6.3f} {s05:6.3f} "
              f"{s01:6.3f} {half_width(0.05, m):6.3f} {half_width(0.01, m):6.3f} "
              f"{ks:6.3f} {(p > 0.99).mean():6.3f} "
              f"{np.median(ref_df):6.2f}{flag}")
    for key, reasons in sorted(missing.items()):
        if reasons:
            print("missing", key, len(reasons), "first:", reasons[0][:160])


if __name__ == "__main__":
    main(sys.argv[1:])
