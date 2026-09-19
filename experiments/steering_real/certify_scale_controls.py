#!/usr/bin/env python3
"""#2263 — what is the external certifier's inner KKT residual actually measuring?

The real-activation sweep found the residual ORDERS states correctly but barely:
an R^2 = -3.88 decoder sits 19% above the honest chart, and permuting the chart
coordinates (R^2 +0.29 -> -0.29) moves it 0.23%.  Every arm is ~5e7 above the
bound.  Two hypotheses:

  H_scale    the residual is dominated by the data's magnitude and row count,
             so it says "this cloud is big", not "this state is wrong";
  H_state    the residual is genuinely about the state, and real activations
             are simply very far from any stationary point.

They are separated by two controls that need no fit at all:

  scale ladder   the SAME real state on the SAME rows, with X (and the decoder
                 with it) multiplied by s.  Under H_scale the residual tracks a
                 power of s; under H_state a pure rescale of an exact state is
                 still exactly as stationary as it was.
  exact state    a NOISELESS planted circle whose least-squares decoder at the
                 true coordinates IS the unpenalised optimum.  If the certifier
                 refuses THAT with a comparable residual, the refusal is not
                 evidence about real activations.

`n` ladder is included because a sum of per-row gradients grows with the row
count whether or not any row is wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def periodic_basis(t, order=1):
    cols = [np.ones_like(t)]
    for h in range(1, order + 1):
        cols.append(np.cos(h * t))
        cols.append(np.sin(h * t))
    return np.stack(cols, axis=1)


def certify(gamfit, X, t, B, lam=0.0, assignment="softmax", order=1):
    plan = [{
        "kind": "periodic", "latent_dim": 1,
        "resolution": {"kind": "periodic_harmonics", "order": order},
        "reference_metric": {"kind": "unit_circle"},
    }]
    kw = dict(
        geometry_plans=plan,
        decoder_blocks=[np.ascontiguousarray(B)],
        t_init=[np.ascontiguousarray(t.reshape(-1, 1))],
        a_init=np.ones((X.shape[0], 1)),
        log_lambda_smooth=[float(lam)],
        log_ard=[[0.0]],
        tier0_mean=np.ascontiguousarray(X.mean(0)),
        tier0_scale=np.ascontiguousarray(X.std(0) + 1e-12),
        assignment=assignment, n_iter=1,
    )
    if assignment == "topk":
        kw["top_k"] = 1
    t0 = time.perf_counter()
    try:
        rep = gamfit.sae.sae_manifold_certify_external(np.ascontiguousarray(X), **kw)
        ik = rep.get("inner_kkt") or {}
        return {
            "status": rep.get("status"),
            "raw": ik.get("raw_gradient_norm"),
            "quotient": ik.get("quotient_gradient_norm"),
            "bound": ik.get("stationarity_bound"),
            "certifies": ik.get("certifies"),
            "wall_s": time.perf_counter() - t0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "raised", "error": f"{type(exc).__name__}: {exc}"[:300],
                "wall_s": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.expanduser("~/f2_site"))
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--split-module",
                    default=os.path.expanduser("~/i2502-baselines"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--pca-rows", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.site)
    sys.path.insert(0, args.split_module)
    import gamfit
    from issue_2502_doc_split import row_side, split_manifest
    from transformers import AutoTokenizer

    prov = {
        "node": platform.node(), "gamfit": gamfit.__version__,
        "code_pin": open(os.path.expanduser("~/f2_wheel/PIN.txt")).read().strip(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    log(f"provenance {json.dumps(prov)}")

    def emit(rec):
        rec["provenance"] = prov
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    # ---------------- control 1: exact, noiseless, stationary by construction --
    log("=== control: planted noiseless circle, decoder = the planting ===")
    for n in (200, 800, 3200):
        for p in (8, 32):
            for scale in (0.01, 1.0, 100.0):
                rng = np.random.default_rng(0)
                t = rng.uniform(-np.pi, np.pi, n)
                B_true = rng.standard_normal((3, p)) / np.sqrt(p)
                X = periodic_basis(t) @ B_true
                X *= scale
                B = B_true * scale
                r = certify(gamfit, X, t, B, lam=-20.0)
                r.update({"record": "exact_state", "n": n, "p": p,
                          "scale": scale, "noise": 0.0,
                          "ratio_raw_over_bound":
                              (r["raw"] / r["bound"])
                              if r.get("raw") and r.get("bound") else None})
                emit(r)
                log(f"exact n={n:5d} p={p:3d} scale={scale:>7g} -> "
                    f"{r.get('status')} raw={r.get('raw')} bound={r.get('bound')}")

    # ---------------- control 2: the same exact state plus noise -------------
    log("=== control: planted circle with noise (state no longer exact) ===")
    for noise in (0.0, 1e-6, 1e-3, 1e-1):
        rng = np.random.default_rng(1)
        n, p = 800, 32
        t = rng.uniform(-np.pi, np.pi, n)
        B_true = rng.standard_normal((3, p)) / np.sqrt(p)
        X = periodic_basis(t) @ B_true + noise * rng.standard_normal((n, p))
        phi = periodic_basis(t)
        B, *_ = np.linalg.lstsq(phi, X, rcond=None)
        r = certify(gamfit, X, t, B, lam=-20.0)
        r.update({"record": "noise_ladder", "n": n, "p": p, "scale": 1.0,
                  "noise": noise,
                  "ratio_raw_over_bound": (r["raw"] / r["bound"])
                  if r.get("raw") and r.get("bound") else None})
        emit(r)
        log(f"noise={noise:>8g} -> {r.get('status')} raw={r.get('raw')} "
            f"bound={r.get('bound')}")

    # ---------------- real activations: scale and n ladders ------------------
    log("=== real: month cloud, scale and row-count ladders ===")
    H = args.harvest
    doc_ids = np.load(os.path.join(H, "doc_ids.npy"))
    man = split_manifest(doc_ids)
    side = row_side(doc_ids)
    token_ids = np.load(os.path.join(H, "token_ids.npy"))
    acts = np.load(os.path.join(H, f"resid_L{args.layer}.npy"), mmap_mode="r")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base",
                                        trust_remote_code=True)
    names = [" January", " February", " March", " April", " May", " June",
             " July", " August", " September", " October", " November",
             " December"]
    cls = [tok.encode(nm, add_special_tokens=False)[0] for nm in names
           if len(tok.encode(nm, add_special_tokens=False)) == 1]
    rows = np.flatnonzero(
        np.isin(token_ids, np.asarray(cls, dtype=token_ids.dtype)) & ~side)
    X0 = np.asarray(acts[rows], dtype=np.float64)
    rng = np.random.default_rng(0)
    bg = np.sort(rng.choice(np.flatnonzero(~side), args.pca_rows, replace=False))
    bx = np.asarray(acts[bg], dtype=np.float64)
    bmu = bx.mean(0)
    bc = bx - bmu
    w, v = np.linalg.eigh((bc.T @ bc) / max(bc.shape[0] - 1, 1))
    P = v[:, np.argsort(w)[::-1][:args.pca_dim]]
    X0 = (X0 - bmu) @ P
    log(f"real month cloud {X0.shape}, split_hash={man['split_hash']}")

    xc = X0 - X0.mean(0)
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w2, v2 = np.linalg.eigh(cov)
    plane = v2[:, np.argsort(w2)[::-1][:2]]
    coords = xc @ plane
    t0_real = np.arctan2(coords[:, 1], coords[:, 0])

    for scale in (0.001, 0.01, 0.1, 1.0, 10.0):
        X = X0 * scale
        phi = periodic_basis(t0_real)
        B, *_ = np.linalg.lstsq(phi, X, rcond=None)
        r = certify(gamfit, X, t0_real, B, lam=-20.0)
        r.update({"record": "real_scale", "scale": scale,
                  "n": int(X.shape[0]), "p": int(X.shape[1]),
                  "rms": float(np.sqrt((X ** 2).mean())),
                  "ratio_raw_over_bound": (r["raw"] / r["bound"])
                  if r.get("raw") and r.get("bound") else None})
        emit(r)
        log(f"real scale={scale:>7g} rms={r['rms']:.4g} -> {r.get('status')} "
            f"raw={r.get('raw')} bound={r.get('bound')}")

    for n in (100, 400, 900, X0.shape[0]):
        X = X0[:n]
        tt = t0_real[:n]
        phi = periodic_basis(tt)
        B, *_ = np.linalg.lstsq(phi, X, rcond=None)
        r = certify(gamfit, X, tt, B, lam=-20.0)
        r.update({"record": "real_nrows", "scale": 1.0, "n": int(n),
                  "p": int(X.shape[1]),
                  "ratio_raw_over_bound": (r["raw"] / r["bound"])
                  if r.get("raw") and r.get("bound") else None})
        emit(r)
        log(f"real n={n:5d} -> {r.get('status')} raw={r.get('raw')} "
            f"bound={r.get('bound')}")

    # unit-RMS normalised real cloud: the scale hypothesis' own remedy
    Xn = X0 / np.sqrt((X0 ** 2).mean())
    phi = periodic_basis(t0_real)
    B, *_ = np.linalg.lstsq(phi, Xn, rcond=None)
    r = certify(gamfit, Xn, t0_real, B, lam=-20.0)
    r.update({"record": "real_unit_rms", "scale": None,
              "n": int(Xn.shape[0]), "p": int(Xn.shape[1]),
              "ratio_raw_over_bound": (r["raw"] / r["bound"])
              if r.get("raw") and r.get("bound") else None})
    emit(r)
    log(f"real unit-RMS -> {r.get('status')} raw={r.get('raw')} "
        f"bound={r.get('bound')}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
