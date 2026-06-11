#!/usr/bin/env python
"""Targeted concept-manifold recovery with an identifiability certificate.

The classic mech-interp claim "concept X lives on a circle in activation
space" (hue, weekday, month) is usually supported by a PCA plot. This driver
makes it a statistical verdict on real activations: fit gam's manifold-SAE
with K=1 atom for each topology candidate, race them on REML evidence,
validate the winning chart against the known concept parameter (hue angle
from the prompt bank's RGB ground truth), and report the residual-gauge
certificate — for a circle the *expected* residual freedom is exactly the
U(1) rotation (+ reflection) of the chart, and nothing else.

Input: a bank directory holding ``activations.npy`` of shape
``(n_prompts, n_layers, d_model)`` (last-token, all layers) and
``prompts.jsonl`` with ``rgb`` ground truth per row (the Manifold-SAE
OLMo color-bank layout).

Example:
  python concept_circle_certificate.py \
      --bank runs/OLMO3_32B_TRAJ_SFT/5e-5-step10790/extra --layer 25 \
      --out color_circle_l25.json
"""
from __future__ import annotations

import argparse
import colorsys
import concurrent.futures
import json
import math
import os
import time
import traceback

import numpy as np


def load_bank(bank: str, layer: int):
    acts = np.load(f"{bank}/activations.npy", mmap_mode="r")
    X = np.asarray(acts[:, layer, :], dtype=np.float64)
    hue = []
    with open(f"{bank}/prompts.jsonl") as f:
        for line in f:
            row = json.loads(line)
            r, g, b = row["rgb"]
            h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hue.append(2.0 * math.pi * h)
    hue_arr = np.asarray(hue, dtype=np.float64)
    if hue_arr.shape[0] != X.shape[0]:
        raise SystemExit(f"bank mismatch: {X.shape[0]} activations vs {hue_arr.shape[0]} prompts")
    mu = X.mean(axis=0, keepdims=True)
    sd = np.maximum(X.std(axis=0, keepdims=True), 1e-6)
    return (X - mu) / sd, hue_arr


def circular_corr(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Jammalamadaka–SenGupta circular correlation coefficient."""
    a = alpha - math.atan2(np.sin(alpha).mean(), np.cos(alpha).mean())
    b = beta - math.atan2(np.sin(beta).mean(), np.cos(beta).mean())
    num = float((np.sin(a) * np.sin(b)).sum())
    den = math.sqrt(float((np.sin(a) ** 2).sum()) * float((np.sin(b) ** 2).sum()))
    return num / den if den > 0 else 0.0


CANDIDATES = [
    ("circle", 1),
    ("euclidean", 1),
    ("euclidean", 2),
    ("sphere", 2),
]


def worker_gamfit():
    os.environ["RAYON_NUM_THREADS"] = "4"
    import gamfit

    return gamfit


def fit_candidate(bank: str, layer: int, topo: str, d_atom: int,
                  n_iter: int, seed: int) -> dict:
    t0 = time.time()
    entry: dict = {"topology": topo, "d_atom": d_atom}
    try:
        gamfit = worker_gamfit()
        X, hue = load_bank(bank, layer)
        m = gamfit.sae_manifold_fit(
            X, K=1, d_atom=d_atom, atom_topology=topo,
            n_iter=n_iter, random_state=seed,
        )
    except Exception as e:
        entry["status"] = "error"
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc()[-1500:]
        entry["seconds"] = time.time() - t0
        return entry

    entry["status"] = "ok"
    entry["seconds"] = time.time() - t0
    entry["reml_score"] = float(m.reml_score) if m.reml_score is not None else None
    entry["reconstruction_r2"] = (
        float(m.reconstruction_r2) if m.reconstruction_r2 is not None else None
    )
    coords = np.asarray(m.coords[0], dtype=np.float64)
    entry["coords_dim"] = list(coords.shape)
    if topo == "circle":
        theta = coords[:, 0]
        entry["hue_circular_corr"] = circular_corr(theta, hue)
    else:
        cs = np.stack([np.cos(hue), np.sin(hue)], axis=1)
        cc = []
        for j in range(coords.shape[1]):
            c = coords[:, j] - coords[:, j].mean()
            denom = np.linalg.norm(c)
            if denom < 1e-12:
                cc.append(0.0)
                continue
            proj = np.linalg.lstsq(cs, c / denom, rcond=None)[1]
            resid = float(proj[0]) if proj.size else 0.0
            cc.append(1.0 - resid)
        entry["coord_hue_r2"] = cc
    for field in ("residual_gauge", "metric_provenance"):
        v = getattr(m, field, None)
        if v is not None:
            try:
                json.dumps(v)
                entry[field] = v
            except TypeError:
                entry[field] = repr(v)[:4000]
    atom = m.atoms[0]
    entry["active_dim"] = getattr(atom, "active_dim", None)
    entry["evidence"] = getattr(atom, "evidence", None)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, hue = load_bank(args.bank, args.layer)
    report: dict = {
        "bank": args.bank,
        "layer": args.layer,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "candidates": [],
    }
    print(f"[setup] X {X.shape} hue range ({hue.min():.2f},{hue.max():.2f})", flush=True)

    max_workers = min(len(CANDIDATES), 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for topo, d in CANDIDATES:
            future = executor.submit(
                fit_candidate,
                args.bank,
                args.layer,
                topo,
                d,
                args.n_iter,
                args.seed,
            )
            futures[future] = (topo, d)
            print(f"[fit] {topo} d={d}", flush=True)

        for future in concurrent.futures.as_completed(futures):
            topo, d = futures[future]
            try:
                entry = future.result()
            except Exception as e:
                entry = {
                    "topology": topo,
                    "d_atom": d,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-1500:],
                }
            report["candidates"].append(entry)
            if entry["status"] == "ok":
                print(
                    f"[fit] -> {topo} d={d} reml={entry['reml_score']} "
                    f"r2={entry['reconstruction_r2']} {entry['seconds']:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"[fit] -> {topo} d={d} status=error "
                    f"{entry.get('seconds', 0.0):.1f}s",
                    flush=True,
                )
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2, default=str)

    ok = [c for c in report["candidates"] if c["status"] == "ok" and c["reml_score"] is not None]
    if ok:
        winner = min(ok, key=lambda c: c["reml_score"])
        report["winner"] = {k: winner[k] for k in ("topology", "d_atom", "reml_score")}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
