#!/usr/bin/env python3
"""#2132 close-bar — held-out EV vs K on PLANTED CIRCLE MIXTURES.

The #2132 defining measurement: on a synthetic mixture of planted 1-D circles the
curved-manifold SAE's held-out reconstruction EV must (a) be non-decreasing with
dictionary size K and (b) sit above the original fixed-rank linear-PCA floor. A single
1-D circle atom captures a whole planted circle from one intrinsic coordinate,
whereas an affine reconstruction needs a 2-D plane per circle — so at a MATCHED
rank K the curved fit strictly wins, and adding atoms captures more circles.

Ground truth (no reference tool required, this is an objective truth-recovery bar):
  M planted circles, each a unit circle living in its OWN random 2-plane, the planes
  mutually orthogonal (a QR frame), all concentric at the origin, plus isotropic
  Gaussian noise. Ambient P >= 2*M so the planes are genuinely orthogonal. Each
  token is drawn on one circle at a uniform angle. Total centered energy per token
  is ~radius^2; the 2*M nonzero PCA eigenvalues are each ~radius^2/2, so the
  EV-optimal affine rank-M reconstruction explains about half the signal variance.
  A curved fit whose atoms learn the M circles should clear that fixed floor and
  must not lose EV as spare atoms are added through K=2M. This is the exact
  comparison #2132 reported; the driver measures it end to end.

Uses gamfit's PUBLIC API only: gamfit.sae_manifold_fit(..., assignment='topk',
d_atom=1, atom_topology='circle', top_k=...) for the curved fit and
model.reconstruct(X_test) for the exact out-of-sample reconstruction. The affine
bar is plain PCA (numpy eigh) at fixed rank M on the identical train/test split.
Rank 2M would span every planted circle plane and is not the 0.73/0.55 floor in
the issue; demanding a positive margin over that full-signal oracle at K=2M would
be mathematically impossible.

Prints one machine-parseable RESULT line and a final PASS/FAIL VERDICT line so the
job log alone closes the issue.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
def held_out_ev(x_test: np.ndarray, recon: np.ndarray) -> float:
    """EV = 1 - ||X_test - recon||_F^2 / ||X_test - mean_test||_F^2.

    Both the curved fit and the PCA bar are scored with this identical formula on
    the identical held-out split, always in float64.
    """
    xt = x_test.astype(np.float64)
    rc = recon.astype(np.float64)
    mt = xt.mean(axis=0)
    ssr = float(np.sum((xt - rc) ** 2))
    sst = float(np.sum((xt - mt[None, :]) ** 2))
    return 1.0 - ssr / max(sst, 1e-300)


def planted_circle_mixture(n, p, m_circles, radius, noise, seed):
    """N tokens on a mixture of M unit circles in mutually-orthogonal 2-planes.

    Returns (X float32 [n,p], assign [n]). Requires p >= 2*m_circles so the QR
    frame yields M genuinely orthogonal planes (linear PCA then needs 2 dims per
    circle to represent it, the whole point of the bar)."""
    if p < 2 * m_circles:
        raise SystemExit(f"need p >= 2*m_circles for orthogonal planes: p={p} m={m_circles}")
    rng = np.random.default_rng(seed)
    # Orthonormal frame: 2*M orthonormal columns in R^p (u_0,v_0,u_1,v_1,...).
    frame, _ = np.linalg.qr(rng.standard_normal((p, 2 * m_circles)))
    u = frame[:, 0::2]  # p x M : first axis of each circle's plane
    v = frame[:, 1::2]  # p x M : second axis
    assign = rng.integers(0, m_circles, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = radius * (np.cos(theta)[:, None] * u[:, assign].T
                  + np.sin(theta)[:, None] * v[:, assign].T)
    x += noise * rng.standard_normal((n, p))
    return np.ascontiguousarray(x, dtype=np.float32), assign


def pca_rank_ev(x_tr, x_te, mean_tr, rank):
    """Affine PCA held-out EV at the fixed comparison rank."""
    xc = (x_tr.astype(np.float64) - mean_tr[None, :])
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    w, vecs = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    vr = vecs[:, order[:rank]]
    tc = x_te.astype(np.float64) - mean_tr[None, :]
    recon = (tc @ vr) @ vr.T + mean_tr[None, :]
    return held_out_ev(x_te, recon)


def curved_ev(x_tr, x_te, mean_tr, *, K, top_k, d_atom, topology, assignment,
              n_iter, seed):
    import gamfit

    model = gamfit.sae_manifold_fit(
        x_tr, K=K, d_atom=d_atom, atom_topology=topology,
        assignment=assignment, top_k=top_k, n_iter=n_iter, random_state=seed)
    recon = np.asarray(model.reconstruct(x_te), dtype=np.float64)
    return held_out_ev(x_te, recon)


def preflight(assignment, top_k, d_atom, topology):
    """Fail-fast deploy-skew guard: prove the installed wheel exposes the exact
    curved public API this driver needs (K>P-capable topk curved lane + OOS
    reconstruct) BEFORE spending the job on the full K sweep. A wheel that predates
    assignment='topk' / d_atom / model.reconstruct raises an actionable message
    here, not an AttributeError minutes in."""
    import os

    import gamfit

    ver = getattr(gamfit, "__version__", "?")
    where = os.path.dirname(getattr(gamfit, "__file__", "?"))
    if not hasattr(gamfit, "sae_manifold_fit"):
        raise SystemExit(f"[preflight] gamfit {ver} at {where} has no sae_manifold_fit")
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal((240, 16)), dtype=np.float32)
    try:
        m = gamfit.sae_manifold_fit(
            x, K=4, d_atom=d_atom, atom_topology=topology,
            assignment=assignment, top_k=top_k, n_iter=3, random_state=0)
        r = np.asarray(m.reconstruct(x[:8]))
    except TypeError as exc:
        raise SystemExit(
            f"[preflight] gamfit {ver} at {where} rejects the curved topk API "
            f"(assignment={assignment!r}, d_atom, top_k): {exc}; upgrade the wheel.")
    if r.shape != (8, 16):
        raise SystemExit(
            f"[preflight] model.reconstruct returned {r.shape}, expected (8, 16)")
    print(f"[#2132] preflight OK: gamfit {ver} at {where}", flush=True)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=384, help="ambient dim (>= 2*m-circles)")
    ap.add_argument("--m-circles", type=int, default=64, help="planted circle count M")
    ap.add_argument("--n", type=int, default=24000, help="tokens (train+test)")
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--k-grid", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--baseline-rank", type=int, default=None,
                    help="fixed PCA floor rank (default: planted circle count M)")
    ap.add_argument("--top-k", type=int, default=1,
                    help="active atoms per token (1 = one circle per token, the "
                         "generative model)")
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--atom-topology", default="circle")
    ap.add_argument("--assignment", default="topk")
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_2132.jsonl")
    args = ap.parse_args()

    if any(k > args.p for k in args.k_grid):
        raise SystemExit(
            f"#2132 is the K<=P curved-framed regime (K>P is the #2134 lane); "
            f"k-grid {args.k_grid} exceeds p={args.p}")

    preflight(args.assignment, args.top_k, args.d_atom, args.atom_topology)

    X, _assign = planted_circle_mixture(
        args.n, args.p, args.m_circles, args.radius, args.noise, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(args.n)
    n_test = max(1, int(round(args.test_frac * args.n)))
    te_idx, tr_idx = perm[:n_test], perm[n_test:]
    x_tr = np.ascontiguousarray(X[tr_idx])
    x_te = np.ascontiguousarray(X[te_idx])
    mean_tr = x_tr.mean(0)
    baseline_rank = args.m_circles if args.baseline_rank is None else args.baseline_rank
    if baseline_rank < 1 or baseline_rank > args.p:
        raise SystemExit(f"baseline rank must be in [1, p={args.p}], got {baseline_rank}")
    ev_pca_floor = pca_rank_ev(x_tr, x_te, mean_tr, baseline_rank)
    print(f"[#2132] planted circles M={args.m_circles} p={args.p} N={args.n} "
          f"train={x_tr.shape[0]} test={x_te.shape[0]} noise={args.noise} "
          f"k_grid={args.k_grid} assignment={args.assignment} top_k={args.top_k} "
          f"pca_floor_rank={baseline_rank} pca_floor_ev={ev_pca_floor:.4f}",
          flush=True)

    rows = []
    for K in args.k_grid:
        print(f"[#2132] START K={K:5d}", flush=True)
        t0 = time.time()
        ev_c = curved_ev(x_tr, x_te, mean_tr, K=K, top_k=args.top_k,
                         d_atom=args.d_atom, topology=args.atom_topology,
                         assignment=args.assignment, n_iter=args.n_iter,
                         seed=args.seed)
        dt = time.time() - t0
        rows.append({"K": K, "ev_curved": ev_c, "ev_pca_floor": ev_pca_floor,
                     "gap_over_pca_floor": ev_c - ev_pca_floor, "fit_seconds": dt})
        print(f"[#2132] K={K:5d} ev_curved={ev_c:.4f} "
              f"ev_pca(rank={baseline_rank})={ev_pca_floor:.4f} "
              f"gap={ev_c - ev_pca_floor:+.4f} ({dt:.0f}s)", flush=True)

    ev_curved = [r["ev_curved"] for r in rows]
    above_pca = all(c >= ev_pca_floor for c in ev_curved)
    # Permit only float32 arithmetic noise, derived from machine precision rather
    # than a fit-quality knob. A substantive EV dip is a failure.
    monotonicity_tolerance = (
        64.0 * np.finfo(np.float32).eps * max(1.0, *(abs(v) for v in ev_curved))
    )
    monotone = all(ev_curved[i + 1] >= ev_curved[i] - monotonicity_tolerance
                   for i in range(len(ev_curved) - 1))
    verdict = "PASS" if (above_pca and monotone) else "FAIL"

    result = {
        "issue": 2132, "p": args.p, "m_circles": args.m_circles, "N": args.n,
        "noise": args.noise, "assignment": args.assignment, "top_k": args.top_k,
        "d_atom": args.d_atom, "k_grid": args.k_grid, "rows": rows,
        "pca_floor_rank": baseline_rank, "pca_floor_ev": ev_pca_floor,
        "above_pca_floor": above_pca, "monotone": monotone,
        "monotonicity_tolerance": monotonicity_tolerance, "verdict": verdict,
    }
    print("[#2132] RESULT " + json.dumps(result), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[#2132] VERDICT={verdict} above_pca_floor={above_pca} "
          f"monotone={monotone} ev_curved={[round(c, 4) for c in ev_curved]} "
          f"ev_pca_floor={ev_pca_floor:.4f}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
