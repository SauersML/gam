"""#2951 manifold-native addition, phase B: the curve each problem variable traces in the residual stream,
fitted by gamfit (REML smoothing), at every layer and position of the phase-A harvest.

Analysis under SPEC 8's exception; every fit is gamfit's Rust engine (``gamfit.fit``), this file only arranges
data and records results.

For a (layer, position, variable) cell: the residuals are centred and projected on their top ``K`` principal
directions; each coordinate is fitted as ``y ~ s(z, periodic=true, period=10)`` for a digit variable or
``y ~ s(z)`` for an ordinal one (the smoothness chosen by REML), and against ``y ~ 1``. Recorded per coordinate:
deviance explained, EDF, and gam's evidence ratio against the intercept model. The fitted curve is evaluated
on a fine grid of ``z`` (non-integer values included) and saved with the principal directions, so phase C can
move a residual along the fitted manifold by any fraction of a step.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CELLS = [
    # position, variable, kind
    ("A_units", "A_units", "cyclic"),
    ("B_units", "B_units", "cyclic"),
    ("A_tens", "A_tens", "ordinal"),
    ("B_tens", "B_tens", "ordinal"),
    ("equals", "sum_units", "cyclic"),
    ("equals", "sum_tens", "ordinal"),
    ("equals", "A_units", "cyclic"),
    ("equals", "B_units", "cyclic"),
    ("E", "sum_units", "cyclic"),
    ("E", "sum_tens", "ordinal"),
]


def aic_corrected(model):
    summary = model.summary()
    value = getattr(summary, "aic_corrected", None)
    if value is None and isinstance(summary, dict):
        value = summary.get("aic_corrected")
    return float(value)


def fit_cell(task):
    import gamfit
    import pandas as pd

    layer, position, variable, kind, X, z, K = task
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:K]
    coords = Xc @ comps.T
    total = float((Xc ** 2).sum())
    if kind == "cyclic":
        formula, grid = "y ~ s(z, periodic=true, period=10)", np.linspace(0, 10, 201)
    else:
        formula, grid = "y ~ s(z)", np.linspace(z.min(), z.max(), 201)
    curve = np.zeros((len(grid), K))
    per = []
    for j in range(K):
        frame = pd.DataFrame({"y": coords[:, j], "z": z.astype(float)})
        try:
            smooth = gamfit.fit(frame, formula)
            null = gamfit.fit(frame, "y ~ 1")
            fitted = np.asarray(smooth.predict(frame)).reshape(-1)
            curve[:, j] = np.asarray(smooth.predict(pd.DataFrame({"z": grid}))).reshape(-1)
            resid = coords[:, j] - fitted
            r2 = 1.0 - float((resid ** 2).sum() / max(1e-300, ((coords[:, j] - coords[:, j].mean()) ** 2).sum()))
            # gamfit's evidence ratio is exp((AIC_c(null) - AIC_c(smooth)) / 2); it overflows past a gap of ~1420,
            # so the log is taken from the two smoothing-corrected AICs it is built from.
            evidence = (aic_corrected(null) - aic_corrected(smooth)) / 2.0
            per.append({"pc": j, "variance_share": float(S[j] ** 2 / total), "r2": r2, "edf": float(smooth.edf_total),
                        "log_evidence_ratio_vs_intercept": evidence})
        except Exception as error:
            per.append({"pc": j, "variance_share": float(S[j] ** 2 / total), "error": str(error)[:200]})
    explained = sum(p["variance_share"] * max(0.0, p.get("r2", 0.0)) for p in per)
    return {"layer": layer, "position": position, "variable": variable, "kind": kind, "K": K,
            "variance_explained_by_curve": explained, "per_pc": per}, comps, grid, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    meta = json.load(open(os.path.join(args.harvest, "meta.json")))
    R = np.load(os.path.join(args.harvest, "residuals.npy"), mmap_mode="r")
    V = dict(np.load(os.path.join(args.harvest, "variables.npz")))
    positions = meta["positions"]
    tasks = []
    for layer in range(R.shape[1]):
        for position, variable, kind in CELLS:
            X = np.asarray(R[:, layer, positions.index(position)], dtype=np.float64)
            tasks.append((layer, position, variable, kind, X, V[variable], args.K))
    results, arrays = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for info, comps, grid, curve in pool.map(fit_cell, tasks):
            results.append(info)
            key = f"{info['layer']}|{info['position']}|{info['variable']}"
            arrays[key + "|comps"] = comps.astype(np.float32)
            arrays[key + "|grid"] = grid
            arrays[key + "|curve"] = curve.astype(np.float32)
            print(f"[manifold] layer {info['layer']:2d} {info['position']:8s} {info['variable']:9s} "
                  f"curve explains {info['variance_explained_by_curve']:.3f} of the position's variance "
                  f"(top-3 PC r2 {[round(p.get('r2', float('nan')), 3) for p in info['per_pc'][:3]]})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "curves.npz"), **arrays)
    with open(os.path.join(args.out, "manifolds.json"), "w") as handle:
        json.dump({"meta": meta, "cells": results}, handle, indent=1)
    print(f"[manifold] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
