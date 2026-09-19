#!/usr/bin/env python3
"""#2263 — does the external certifier refuse a state that is optimal BY CONSTRUCTION?

The scale-control run found `nonstationary` on a noiseless planted circle whose
decoder is the exact planting.  Before that can be reported as "the refusal is
not about real activations", one confound has to be cleared: that run passed
`log_ard = 0`, i.e. an ARD ridge of strength **1**, and `log_lambda_sparse`
defaulted to 0 as well.  Under those penalties the planted decoder is NOT the
optimum of the criterion being certified, so a refusal is correct and says
nothing.

So the penalties are laddered to zero.  As every penalty strength goes to zero
the certified criterion becomes the plain least-squares objective, whose exact
minimiser at the planted coordinates IS the planted decoder.  If the residual
collapses toward the bound, the certifier is right and my earlier payload was
wrong.  If it does not move, the refusal is unconditional in this regime.

The `n` ladder is repeated at the cleared penalties, because an extensive sum
of per-row gradient terms compared against a bar that does not grow with `n`
gets harder as data is added even when nothing about the state changed.
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


def certify(gamfit, X, t, B, *, lam_smooth, lam_ard, lam_sparse,
            assignment="softmax", order=1, tier0=True):
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
        log_lambda_smooth=[float(lam_smooth)],
        log_ard=[[float(lam_ard)]],
        log_lambda_sparse=float(lam_sparse),
        assignment=assignment, n_iter=1,
    )
    if tier0:
        kw["tier0_mean"] = np.ascontiguousarray(X.mean(0))
        kw["tier0_scale"] = np.ascontiguousarray(X.std(0) + 1e-12)
    if assignment == "topk":
        kw["top_k"] = 1
    t0 = time.perf_counter()
    try:
        rep = gamfit.sae.sae_manifold_certify_external(np.ascontiguousarray(X), **kw)
        ik = rep.get("inner_kkt") or {}
        raw, bound = ik.get("raw_gradient_norm"), ik.get("stationarity_bound")
        return {"status": rep.get("status"), "raw": raw,
                "quotient": ik.get("quotient_gradient_norm"), "bound": bound,
                "certifies": ik.get("certifies"),
                "ratio": (raw / bound) if (raw and bound) else None,
                "wall_s": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001
        return {"status": "raised",
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "wall_s": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.expanduser("~/f2_site"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.site)
    import gamfit

    prov = {"node": platform.node(), "gamfit": gamfit.__version__,
            "code_pin": open(os.path.expanduser("~/f2_wheel/PIN.txt")).read().strip(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",
                                                   "<unset>")}
    log(f"provenance {json.dumps(prov)}")

    def emit(rec):
        rec["provenance"] = prov
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    def planted(n, p, seed=0, noise=0.0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(-np.pi, np.pi, n)
        B = rng.standard_normal((3, p)) / np.sqrt(p)
        X = periodic_basis(t) @ B
        if noise:
            X = X + noise * rng.standard_normal((n, p))
        return X, t, B

    # ---- penalty ladder on the exactly-planted state ----------------------
    log("=== penalty ladder, exact planted state, n=800 p=32 ===")
    X, t, B = planted(800, 32)
    for lam in (0.0, -4.0, -10.0, -20.0, -40.0, -80.0):
        r = certify(gamfit, X, t, B, lam_smooth=lam, lam_ard=lam,
                    lam_sparse=lam)
        r.update({"record": "penalty_ladder", "n": 800, "p": 32,
                  "log_lambda_all": lam, "tier0": True})
        emit(r)
        log(f"log-lambda(all)={lam:+6.1f} -> {r.get('status')} "
            f"raw={r.get('raw')} bound={r.get('bound')} ratio={r.get('ratio')}")

    # ---- same, without the Tier-0 standardisation -------------------------
    log("=== same state, no tier0 frame ===")
    for lam in (0.0, -40.0):
        r = certify(gamfit, X, t, B, lam_smooth=lam, lam_ard=lam,
                    lam_sparse=lam, tier0=False)
        r.update({"record": "no_tier0", "n": 800, "p": 32,
                  "log_lambda_all": lam, "tier0": False})
        emit(r)
        log(f"no-tier0 log-lambda={lam:+6.1f} -> {r.get('status')} "
            f"raw={r.get('raw')} bound={r.get('bound')} ratio={r.get('ratio')}")

    # ---- n ladder at cleared penalties ------------------------------------
    log("=== n ladder at cleared penalties (state quality identical) ===")
    for n in (100, 200, 400, 800, 1600, 3200, 6400):
        Xn, tn, Bn = planted(n, 32)
        r = certify(gamfit, Xn, tn, Bn, lam_smooth=-40.0, lam_ard=-40.0,
                    lam_sparse=-40.0)
        r.update({"record": "n_ladder_cleared", "n": n, "p": 32,
                  "log_lambda_all": -40.0, "tier0": True})
        emit(r)
        log(f"n={n:5d} -> {r.get('status')} raw={r.get('raw')} "
            f"bound={r.get('bound')} ratio={r.get('ratio')}")

    # ---- p ladder at cleared penalties ------------------------------------
    log("=== p ladder at cleared penalties ===")
    for p in (4, 8, 16, 32, 64, 128):
        Xp, tp, Bp = planted(800, p)
        r = certify(gamfit, Xp, tp, Bp, lam_smooth=-40.0, lam_ard=-40.0,
                    lam_sparse=-40.0)
        r.update({"record": "p_ladder_cleared", "n": 800, "p": p,
                  "log_lambda_all": -40.0, "tier0": True})
        emit(r)
        log(f"p={p:5d} -> {r.get('status')} raw={r.get('raw')} "
            f"bound={r.get('bound')} ratio={r.get('ratio')}")

    # ---- does ANY perturbation of the exact state matter? -----------------
    log("=== decoder perturbation ladder at cleared penalties (n=800 p=32) ===")
    rng = np.random.default_rng(5)
    scaleB = float(np.std(B))
    for eps in (0.0, 1e-8, 1e-4, 1e-2, 1e-1, 1.0):
        Bp = B + eps * scaleB * rng.standard_normal(B.shape)
        r = certify(gamfit, X, t, Bp, lam_smooth=-40.0, lam_ard=-40.0,
                    lam_sparse=-40.0)
        recon = periodic_basis(t) @ Bp
        r2 = 1.0 - float(((X - recon) ** 2).sum()) / float(
            ((X - X.mean(0)) ** 2).sum())
        r.update({"record": "perturb_cleared", "n": 800, "p": 32,
                  "epsilon": eps, "chart_r2": r2, "log_lambda_all": -40.0})
        emit(r)
        log(f"eps={eps:>8g} R2={r2:+.6f} -> {r.get('status')} "
            f"raw={r.get('raw')} ratio={r.get('ratio')}")

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
