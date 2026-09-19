#!/usr/bin/env python3
"""#2548 / #2517 — the K axis: is the certify-path residual extensive in n, or in ROWS-PER-ATOM?

A K=1 ladder cannot tell those apart, because rows-per-atom == n there. This
holds n FIXED and raises K with each row assigned to exactly one atom, so
rows-per-atom = n/K falls while n does not move.

sae-research's reading of the FIT path (`support_term.rs::raw_stationarity`) is
that the decoder block sums over `atom_rows[atom_idx]` — the rows assigned to
THAT atom — which predicts `raw` falls like 1/K here. Their proposed shared
normaliser ("normalise a block by what the block sums over") rests on that
being the right count. If `raw` does NOT fall like 1/K, the normaliser is wrong
about WHICH count and needs re-deriving before either site builds on it.

The state is exact by construction at every K: atom k gets its own planted
decoder, row i is assigned to atom k(i), and X_i = Phi(t_i) B_{k(i)}, so the
reconstruction residual is identically zero at every point on the ladder. That
is the property the fit path cannot hold constant and this fixture can.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def build_exact_k(n, p, K, rng, PHB):
    """Exact planted state with n rows split evenly across K circle atoms."""
    t = rng.uniform(-np.pi, np.pi, n)
    phi = np.asarray(PHB(t, 1))                       # (n, 3)
    blocks = [rng.standard_normal((3, p)) / np.sqrt(p) for _ in range(K)]
    owner = np.arange(n) % K                          # even split, rows/atom = n/K
    X = np.zeros((n, p))
    for k in range(K):
        m = owner == k
        X[m] = phi[m] @ blocks[k]
    # routing logits: row i names its own atom
    a = np.zeros((n, K))
    a[np.arange(n), owner] = 1.0
    # every atom carries a coordinate for every row; the assigned one is what fires
    t_init = [np.ascontiguousarray(t.reshape(-1, 1)) for _ in range(K)]
    return X, t, t_init, blocks, a, owner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.expanduser("~/f2_site"))
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--assignment", default="topk")
    ap.add_argument("--lam", type=float, default=-40.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.site)
    import gamfit
    from gamfit._rust import periodic_harmonic_basis as PHB

    rows_out = []

    def emit(rec):
        rows_out.append(rec)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    n, p = args.n, args.p
    print(f"K axis at FIXED n={n}, p={p}, assignment={args.assignment}, "
          f"penalties e^{args.lam:g}")
    print(f"{'K':>4} {'rows/atom':>10} {'recon R2':>10} {'raw':>14} "
          f"{'bound':>12} {'raw*K':>12} {'raw/bound':>12}")

    base_raw = None
    for K in [int(k) for k in args.ks.split(",")]:
        rng = np.random.default_rng(0)
        X, t, t_init, blocks, a, owner = build_exact_k(n, p, K, rng, PHB)

        # verify the planted state really is exact at this K
        phi = np.asarray(PHB(t, 1))
        recon = np.zeros_like(X)
        for k in range(K):
            m = owner == k
            recon[m] = phi[m] @ blocks[k]
        r2 = 1.0 - float(((X - recon) ** 2).sum()) / float(
            ((X - X.mean(0)) ** 2).sum())

        kw = dict(
            geometry_plans=[{
                "kind": "periodic", "latent_dim": 1,
                "resolution": {"kind": "periodic_harmonics", "order": 1},
                "reference_metric": {"kind": "unit_circle"},
            } for _ in range(K)],
            decoder_blocks=[np.ascontiguousarray(b) for b in blocks],
            t_init=t_init,
            a_init=np.ascontiguousarray(a),
            log_lambda_smooth=[args.lam] * K,
            log_ard=[[args.lam]] * K,
            log_lambda_sparse=args.lam,
            tier0_mean=np.zeros(p), tier0_scale=np.ones(p),
            assignment=args.assignment, n_iter=1,
        )
        if args.assignment == "topk":
            kw["top_k"] = 1
        try:
            rep = gamfit.sae.sae_manifold_certify_external(
                np.ascontiguousarray(X), **kw)
            ik = rep.get("inner_kkt") or {}
            raw, bound = ik.get("raw_gradient_norm"), ik.get("stationarity_bound")
        except Exception as exc:  # noqa: BLE001
            print(f"{K:>4} {n/K:>10.1f} {r2:>10.6f}   ERROR "
                  f"{type(exc).__name__}: {str(exc)[:70]}")
            emit({"record": "k_axis", "K": K, "n": n, "error":
                  f"{type(exc).__name__}: {exc}"[:300]})
            continue
        if base_raw is None:
            base_raw = raw
        print(f"{K:>4} {n/K:>10.1f} {r2:>10.6f} {raw:>14.6g} {bound:>12.5g} "
              f"{raw*K:>12.6g} {raw/bound:>12.5g}")
        emit({"record": "k_axis", "K": K, "n": n, "p": p,
              "rows_per_atom": n / K, "recon_r2": r2,
              "raw": raw, "bound": bound, "raw_times_K": raw * K,
              "ratio": raw / bound, "assignment": args.assignment})

    ok = [r for r in rows_out if r.get("raw")]
    if len(ok) >= 2:
        K = np.array([r["K"] for r in ok], dtype=float)
        raw = np.array([r["raw"] for r in ok], dtype=float)
        slope = float(np.polyfit(np.log(K), np.log(raw), 1)[0])
        print(f"\nlog-log slope of raw vs K: {slope:+.4f}")
        print("  -1 => extensive in ROWS-PER-ATOM (sae-research's reading)")
        print("   0 => extensive in n only; the per-atom count is NOT the axis")
        emit({"record": "k_axis_slope", "slope_log_raw_vs_log_K": slope,
              "n": n, "p": p, "assignment": args.assignment})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
