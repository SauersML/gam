"""Size, Monte-Carlo error, KS and power for calibrate.py results.

    python analyze.py RESULT.json [RESULT.json ...] > table.md

For each test in each result file: the rejection rate at a = 0.10, 0.05, 0.01
with its binomial Monte-Carlo standard error sqrt(a(1-a)/R) at the nominal
rate, the Kolmogorov-Smirnov p-value against Uniform(0,1), and a verdict for
null scenarios: PASS when every size is within 2 MCSE above a (a conservative
test passes; validity is P(p <= a) <= a).
"""

import json
import sys

import numpy as np
from scipy import stats

ALPHAS = (0.10, 0.05, 0.01)


def summarize(path):
    data = json.load(open(path))
    records = data["records"]
    errors = [r["error"] for r in records if "error" in r]
    by_test = {}
    for record in records:
        for name, p in record.get("tests", {}).items():
            by_test.setdefault(name, []).append(np.nan if p is None else p)
    rows = []
    for name, values in sorted(by_test.items()):
        values = np.asarray(values, dtype=float)
        usable = values[np.isfinite(values)]
        if len(usable) == 0:
            rows.append({"test": name, "usable": 0, "unavailable": int(len(values)), "size": [np.nan] * 3,
                         "mcse": [np.nan] * 3, "ks_p": np.nan, "within_2_mcse": False})
            continue
        size = [float((usable <= a).mean()) for a in ALPHAS]
        mcse = [float(np.sqrt(a * (1 - a) / len(usable))) for a in ALPHAS]
        ks = float(stats.kstest(usable, "uniform").pvalue)
        within = all(s <= a + 2 * m for s, a, m in zip(size, ALPHAS, mcse))
        rows.append(
            {
                "test": name,
                "usable": int(len(usable)),
                "unavailable": int(len(values) - len(usable)),
                "size": size,
                "mcse": mcse,
                "ks_p": ks,
                "within_2_mcse": within,
            }
        )
    return data, errors, rows


def main():
    print("| scenario | n | test | usable/R | p<=.10 | p<=.05 | p<=.01 | MCSE (.10/.05/.01) | KS p | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for path in sys.argv[1:]:
        data, errors, rows = summarize(path)
        null = data["scenario"].endswith("_null") or data["scenario"] == "ls_scale_only"
        for row in rows:
            verdict = ("PASS" if row["within_2_mcse"] else "FAIL") if null else "power"
            print(
                f"| {data['scenario']} | {data['n']} | {row['test']} | "
                f"{row['usable']}/{data['reps']} | "
                + " | ".join(f"{s:.3f}" for s in row["size"])
                + " | "
                + "/".join(f"{m:.3f}" for m in row["mcse"])
                + f" | {row['ks_p']:.3g} | {verdict} |"
            )
        if errors:
            print(f"| {data['scenario']} | {data['n']} | failed fits | {len(errors)} | {errors[0][:80]} |||||| |")


if __name__ == "__main__":
    main()
