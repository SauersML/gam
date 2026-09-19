#!/usr/bin/env python3
"""Aggregate results/<case>.json (written by rho_marginal_bench.py) into a
markdown table.

For each case: mean held-out metric (dev, or truth_mse for synthetic cases)
for plugin / conditional (shipped) / marginal (first-order rho-marginal) over
the folds where all three exist, the paired per-fold relative change marginal
vs conditional, and its paired t-statistic across those folds. A change is
flagged when |t| exceeds the two-sided 5% Student-t quantile for the fold
count. CV folds share training data, so the fold t overstates significance;
the replicate studies (rho_marginal_replicates.py, full_laplace_replicates.py)
are the valid test.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
VARIANTS = ("plugin", "conditional", "marginal")


def main():
    lines = ["| case | family | n | metric | folds | plugin | conditional (shipped) | rho-marginal | change | paired t |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    better = worse = 0
    for p in sorted((HERE / "results").glob("*.json")):
        if p.name.startswith(("full_laplace", "gamfit_replicates")):
            continue
        d = json.loads(p.read_text())
        key = "truth_mse" if d["synth"] else "dev"
        ok = [r for r in d["rows"] if r["ok"]]
        full = [r for r in ok if all(v in r for v in VARIANTS)]
        folds = f"{len(full)}/{len(d['rows'])}"
        if not full:
            lines.append(f"| {d['name']} | {d['family']} | {d['n']} | {key} | {folds} | "
                         + ("no V_p on any fitted fold" if ok else "fit failed") + " | | | | |")
            continue
        col = {v: np.array([r[v][key] for r in full]) for v in VARIANTS}
        cells = [f"{np.mean(col[v]):.6g}" for v in VARIANTS]
        change = tstat = ""
        if d["family"] == "gaussian":
            change = "0 (identity link)"
        elif len(full) > 1:
            rel = col["marginal"] / col["conditional"] - 1
            sd = np.std(rel, ddof=1)
            t = np.mean(rel) / (sd / np.sqrt(len(rel))) if sd > 0 else 0.0
            crit = student_t.ppf(0.975, len(rel) - 1)
            change = f"{100 * np.mean(rel):+.3f}%"
            tstat = f"{t:+.2f}" + (" *" if abs(t) > crit else "")
            if abs(t) > crit:
                better += np.mean(rel) < 0
                worse += np.mean(rel) > 0
        lines.append(f"| {d['name']} | {d['family']} | {d['n']} | {key} | {folds} | " + " | ".join(cells)
                     + f" | {change} | {tstat} |")
    lines.append("")
    lines.append(f"Flagged (|t| above the 5% quantile) improvements: {better}; regressions: {worse}.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
