"""Markdown tables and the two-sided gate for the pv-interactions README.

usage: python bench/pvalue_calibration/pv-interactions/tabulate.py OUT1.json OUT2.json ...

Every null term is checked seven times: KS against U(0, 1) and, at each level a in
.10/.05/.01, an exact two-sided binomial test of both P(p <= a) and P(p > 1 - a) against
a. The checks share a family-wise level of FAMILY_ALPHA (Bonferroni over all of them).
"""
import json
import sys

import numpy as np
from scipy import stats

ALPHAS = (0.10, 0.05, 0.01)
NULL = {
    "ti_gauss_200": ["ti(x1, x2)"], "ti_gauss_1000": ["ti(x1, x2)"],
    "ti_binom_200": ["ti(x1, x2)"], "ti_binom_1000": ["ti(x1, x2)"],
    "by_gauss_300": ["s(x, by=g):by=g[b]", "s(x, by=g):by=g[c]"],
    "by_binom_600": ["s(x, by=g):by=g[b]", "s(x, by=g):by=g[c]"],
    "te_gauss_300": ["te(x1, x2)"], "te_binom_600": ["te(x1, x2)"],
    "iso_gauss_300": ["s(x1, x2)"],
    "vc_gauss_300": ["s(x, by=z)"], "vc_binom_600": ["s(x, by=z)"],
}
POWER = {
    "ti_gauss_200_power": ("ti(x1, x2)", "0.6"), "ti_binom_1000_power": ("ti(x1, x2)", "0.8"),
    "te_gauss_300_power": ("te(x1, x2)", "0.5"), "iso_gauss_300_power": ("s(x1, x2)", "0.5"),
    "vc_gauss_300_power": ("s(x, by=z)", "0.4"),
    "by_gauss_300": ("s(x, by=g):by=g[a]", "1.5"), "by_binom_600": ("s(x, by=g):by=g[a]", "2.0"),
}
FAMILY_ALPHA = 0.01
CHECKS = sum(len(v) for v in NULL.values()) * (1 + 2 * len(ALPHAS))
CHECK_ALPHA = FAMILY_ALPHA / CHECKS

data = {}
for path in sys.argv[1:]:
    d = json.load(open(path))
    data[d["summary"]["cell"]] = d


def pvals(d, term):
    return np.array([r[term]["p_value"] for r in d["rows"] if term in r and r[term]["p_value"] is not None], float)


def edf(d, term):
    return float(np.median([r[term]["edf"] for r in d["rows"] if term in r]))


print(f"checks={CHECKS} check_alpha={CHECK_ALPHA:.3g}\n")
print("| cell | null term | m (failed) | median edf | size .10 | size .05 | size .01 | P(p>.90) | P(p>.95) | P(p>.99) | KS p | min check p |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
worst, allpass = 1.0, True
for cell, terms in NULL.items():
    d = data[cell]
    for term in terms:
        p = pvals(d, term)
        m = p.size
        checks = [stats.kstest(p, "uniform").pvalue]
        for a in ALPHAS:
            checks.append(stats.binomtest(int(np.sum(p <= a)), m, a).pvalue)
            checks.append(stats.binomtest(int(np.sum(p > 1 - a)), m, a).pvalue)
        lo = min(checks)
        worst = min(worst, lo)
        allpass &= lo > CHECK_ALPHA
        print(f"| {cell} | `{term}` | {m} ({d['summary']['failed']}) | {edf(d, term):.2f} | "
              + " | ".join(f"{np.mean(p <= a):.3f}" for a in ALPHAS) + " | "
              + " | ".join(f"{np.mean(p > 1 - a):.3f}" for a in ALPHAS)
              + f" | {checks[0]:.3f} | {lo:.3f} |")
print(f"\nworst check p = {worst:.4f}; all pass at {CHECK_ALPHA:.3g}: {allpass}\n")
print("| cell | term | amplitude | m (failed) | median edf | power .10 | power .05 | power .01 |")
print("|---|---|---|---|---|---|---|---|")
for cell, (term, amp) in POWER.items():
    d = data[cell]
    p = pvals(d, term)
    print(f"| {cell} | `{term}` | {amp} | {p.size} ({d['summary']['failed']}) | {edf(d, term):.2f} | "
          + " | ".join(f"{np.mean(p <= a):.3f}" for a in ALPHAS) + " |")
print()
for cell, d in data.items():
    for e in d["summary"]["errors"]:
        print(cell, e[:220])
