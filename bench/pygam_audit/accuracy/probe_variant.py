#!/usr/bin/env python3
"""Rerun gamfit on benchmark cases with a formula variant, same folds, paired vs the default.

  python probe_variant.py CASE_REGEX VARIANT [VARIANT ...]
VARIANT: 'dp0' (double_penalty=false on every s()/te()), 'kNN' (k=NN on every s()),
         'default' (the unmodified formula).
Prints the per-case mean primary metric (truth_mse for synthetic cases, else held-out dev /
logloss) for each variant plus the stored pyGAM numbers from results/<case>.json when the
data are identical (non-g1d cases; g1d data seeds changed from hash() to crc32).
"""
import re
import sys
import json
import dataclasses
import numpy as np
import bench_accuracy as B
from sklearn.model_selection import KFold, StratifiedKFold


def transform(formula, variant):
    if variant == "default":
        return formula
    if variant == "dp0":
        opt = "double_penalty=false"
    elif variant.startswith("k"):
        opt = f"k={int(variant[1:])}"
    else:
        raise SystemExit(f"unknown variant {variant}")

    def add(m):
        head, body = m.group(1), m.group(2)
        if head == "te" and opt.startswith("k="):
            return m.group(0)
        if "k=" in body and opt.startswith("k="):
            return m.group(0)
        return f"{head}({body}, {opt})"
    return re.sub(r"\b(s|te)\(([^()]*)\)", add, formula)


def main():
    pat, variants = sys.argv[1], sys.argv[2:]
    for case in B.all_cases(big=False):
        if not re.search(pat, case.name):
            continue
        prim = "truth_mse" if case.kind == "synth" else {"binomial": "logloss"}.get(case.family, "dev")
        X = np.column_stack([case.cols[k] for k in case.cols]).astype(float)
        sp = (StratifiedKFold(5, shuffle=True, random_state=0).split(X, case.y) if case.family == "binomial"
              else KFold(5, shuffle=True, random_state=0).split(X))
        folds = list(sp)
        out = {}
        for v in variants:
            c2 = dataclasses.replace(case, formula=transform(case.formula, v))
            vals, edfs = [], []
            for tr, te in folds:
                mt = case.mu_true[te] if case.mu_true is not None else None
                try:
                    mu, info = B.fit_gamfit(c2, tr, te)
                    vals.append(B.metrics(case.family, case.y[te], mu, mt)[prim])
                    edfs.append(info.get("edf_total", np.nan))
                except Exception as e:
                    vals.append(np.nan)
                    edfs.append(np.nan)
                    print(f"   {case.name} {v} fail: {type(e).__name__}: {str(e)[:120]}")
            out[v] = (np.array(vals), np.array(edfs))
        line = f"{case.name:22s} {prim:9s}"
        for v, (vals, edfs) in out.items():
            line += f" | {v}: {np.nanmean(vals):.5g} (edf {np.nanmean(edfs):.1f})"
        p = B.RESULTS / f"{case.name}.json" if hasattr(B, "RESULTS") else None
        if p is not None and p.exists() and not case.name.startswith("g1d"):
            rows = json.loads(p.read_text())["rows"]
            for meth in ("pygam_default", "pygam_grid"):
                r = [x[prim] for x in rows if x["method"] == meth and x["ok"]]
                line += f" | {meth}: {np.mean(r):.5g}"
        if len(variants) >= 2 and "default" in out:
            base = out["default"][0]
            for v in variants:
                if v == "default":
                    continue
                d = out[v][0] - base
                d = d[np.isfinite(d)]
                if len(d) > 1:
                    line += f" | {v}-default d={d.mean():+.3g}+-{d.std(ddof=1)/np.sqrt(len(d)):.2g}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
