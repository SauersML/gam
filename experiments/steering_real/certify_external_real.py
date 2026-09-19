#!/usr/bin/env python3
"""#2263 — is the external certifier's real-activation verdict NON-VACUOUS?

`sae_manifold_certify_external` returns `nonstationary` on a real month-token
cloud with an honest least-squares circle chart.  A refusal is only evidence if
the quantity it refuses on can tell states apart.  So the same call is driven
across a ladder of states on the SAME real rows:

  lsq            the honest chart: top-2 plane angle, least-squares decoder
  null_random    same shapes, decoder replaced by noise at the same scale
  null_shuffled  honest decoder, chart coordinates permuted across rows
  order 1/2/3    richer periodic bases (the chart the plan declares gets wider)
  lambda ladder  the smoothing strength the state is certified under

If the inner KKT residual orders these the way state quality orders them, the
certifier is measuring the state.  If every arm returns the same number, the
refusal is a constant and says nothing about real activations.
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


CYCLIC = {
    "month": [" January", " February", " March", " April", " May", " June",
              " July", " August", " September", " October", " November",
              " December"],
    "weekday": [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday",
                " Saturday", " Sunday"],
}


def periodic_basis(t, order):
    cols = [np.ones_like(t)]
    for h in range(1, order + 1):
        cols.append(np.cos(h * t))
        cols.append(np.sin(h * t))
    return np.stack(cols, axis=1)          # (n, 2*order+1)


def chart_angles(cloud):
    mu = cloud.mean(0)
    xc = cloud - mu
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(w)[::-1][:2]
    coords = xc @ v[:, order]
    return np.arctan2(coords[:, 1], coords[:, 0]), w[order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=os.path.expanduser("~/i2502/harvest"))
    ap.add_argument("--split-module",
                    default=os.path.expanduser("~/i2502-baselines"))
    ap.add_argument("--site", default=os.path.expanduser("~/f2_site"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--cyclic-class", default="month", choices=sorted(CYCLIC))
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--pca-rows", type=int, default=40000)
    ap.add_argument("--assignment", default="softmax")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.site)
    sys.path.insert(0, args.split_module)
    import gamfit
    from issue_2502_doc_split import row_side, split_manifest
    from transformers import AutoTokenizer

    H = args.harvest
    doc_ids = np.load(os.path.join(H, "doc_ids.npy"))
    man = split_manifest(doc_ids)
    side = row_side(doc_ids)
    token_ids = np.load(os.path.join(H, "token_ids.npy"))
    acts = np.load(os.path.join(H, f"resid_L{args.layer}.npy"), mmap_mode="r")

    prov = {
        "node": platform.node(), "gamfit": gamfit.__version__,
        "code_pin": open(os.path.expanduser("~/f2_wheel/PIN.txt")).read().strip(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "split_hash": man["split_hash"],
        "harvest_doc_digest": man["harvest_doc_digest"],
        "layer": args.layer, "model": "Qwen/Qwen3.5-4B-Base",
    }
    log(f"provenance {json.dumps(prov)}")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base",
                                        trust_remote_code=True)
    cls = [tok.encode(n, add_special_tokens=False)[0]
           for n in CYCLIC[args.cyclic_class]
           if len(tok.encode(n, add_special_tokens=False)) == 1]
    rows = np.flatnonzero(np.isin(token_ids, np.asarray(cls, dtype=token_ids.dtype))
                          & ~side)
    X = np.asarray(acts[rows], dtype=np.float64)

    rng = np.random.default_rng(0)
    bg = np.sort(rng.choice(np.flatnonzero(~side), args.pca_rows, replace=False))
    bx = np.asarray(acts[bg], dtype=np.float64)
    bmu = bx.mean(0)
    bc = bx - bmu
    w, v = np.linalg.eigh((bc.T @ bc) / max(bc.shape[0] - 1, 1))
    P = v[:, np.argsort(w)[::-1][:args.pca_dim]]
    X = (X - bmu) @ P
    log(f"{args.cyclic_class} cloud in train-only PCA-{args.pca_dim}: {X.shape}")

    t, plane_eig = chart_angles(X)
    tier0_mean = np.ascontiguousarray(X.mean(0))
    tier0_scale = np.ascontiguousarray(X.std(0) + 1e-8)

    def run(arm, order, lam, decoder, tt):
        plan = [{
            "kind": "periodic", "latent_dim": 1,
            "resolution": {"kind": "periodic_harmonics", "order": order},
            "reference_metric": {"kind": "unit_circle"},
        }]
        phi = periodic_basis(tt, order)
        recon = phi @ decoder
        ss = float(((X - recon) ** 2).sum())
        tot = float(((X - X.mean(0)) ** 2).sum())
        r2 = 1.0 - ss / tot
        kw = dict(
            geometry_plans=plan,
            decoder_blocks=[np.ascontiguousarray(decoder)],
            t_init=[np.ascontiguousarray(tt.reshape(-1, 1))],
            a_init=np.ones((X.shape[0], 1)),
            log_lambda_smooth=[float(lam)],
            log_ard=[[0.0]],
            tier0_mean=tier0_mean, tier0_scale=tier0_scale,
            assignment=args.assignment, n_iter=1,
        )
        if args.assignment == "topk":
            kw["top_k"] = 1
        t0 = time.perf_counter()
        try:
            rep = gamfit.sae.sae_manifold_certify_external(
                np.ascontiguousarray(X), **kw)
            wall = time.perf_counter() - t0
            ik = rep.get("inner_kkt") or {}
            rec = {
                "record": "cert_sweep", "arm": arm, "order": order,
                "log_lambda_smooth": float(lam), "chart_r2": r2,
                "status": rep.get("status"),
                "inner_raw": ik.get("raw_gradient_norm"),
                "inner_quotient": ik.get("quotient_gradient_norm"),
                "inner_bound": ik.get("stationarity_bound"),
                "certifies": ik.get("certifies"),
                "reason": str(rep.get("reason"))[:300],
                "wall_s": wall,
            }
        except Exception as exc:  # noqa: BLE001
            rec = {"record": "cert_sweep", "arm": arm, "order": order,
                   "log_lambda_smooth": float(lam), "chart_r2": r2,
                   "status": "raised",
                   "error": f"{type(exc).__name__}: {exc}"[:400],
                   "wall_s": time.perf_counter() - t0}
        rec["provenance"] = prov
        rec["n_rows"] = int(X.shape[0])
        rec["p"] = int(X.shape[1])
        rec["plane_eigenvalues"] = [float(e) for e in plane_eig]
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        log(f"{arm:14s} order={order} lam={lam:+.1f} R2={r2:+.4f} "
            f"status={rec.get('status')} raw={rec.get('inner_raw')} "
            f"bound={rec.get('inner_bound')}")
        return rec

    nrng = np.random.default_rng(17)
    for order in (1, 2, 3):
        phi = periodic_basis(t, order)
        B, *_ = np.linalg.lstsq(phi, X, rcond=None)
        for lam in (-4.0, 0.0, 4.0):
            run("lsq", order, lam, B, t)
        # nulls at the same shapes, only at lambda = 0
        run("null_random", order, 0.0,
            nrng.standard_normal(B.shape) * float(np.std(B)), t)
        run("null_shuffled", order, 0.0, B, nrng.permutation(t))

    log("sweep done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
