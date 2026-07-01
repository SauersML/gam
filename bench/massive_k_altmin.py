"""Massive-K manifold SAE via ALTERNATING MINIMIZATION (scales to K=32,000).

The REML/arrow-Schur joint solver in `sae_manifold_fit` does not scale past
K~16 (dense n×K logits, O(K^2 n) preconditioner, joint-solve co-collapse). But
a traditional SAE is not trained that way either — it is trained by alternating
minimization / sparse coding, which scales linearly in K because each atom is
fit INDEPENDENTLY against the rows assigned to it, and a dead atom is simply
never any row's arg-min (so it cannot grab spurious mass — no co-collapse).

This trains, at the SAME K and on the SAME data, two dictionaries:
  * MANIFOLD: each atom is a 1-D CURVED fiber  gamma_k(t) = Phi(t) @ B_k, with
    the periodic basis  Phi(t) = [1, cos2pi t, sin2pi t, ...]  (the gam
    `periodic` / 1-D Duchon atom). top_k=1 sparse coding.
  * LINEAR:   each atom is a straight line  z * d_k  (top_k=1). This is the
    traditional linear SAE / K-lines baseline.

On CURVED data (concept fibers with >=2 harmonics) a straight line cannot follow
a bending fiber, so the manifold dictionary reconstructs strictly better at
matched K. Reports reconstruction R^2 for both.

RAM stays tiny: assignments are a length-n int vector (top_k=1), decoders are
K*(1+2H)*p floats. Never materializes a dense n×K matrix.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "0")  # 0 -> let BLAS pick; we pin via arg if needed

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from massive_k_manifold_validate import make_curved_data  # noqa: E402


def periodic_phi(t: np.ndarray, harmonics: int) -> np.ndarray:
    """Design matrix Phi(t): [1, cos2pi t, sin2pi t, ..., cos2pi H t, sin2pi H t].

    t: (m,) in [0,1). Returns (m, 1+2H). This is exactly the gam `periodic`
    atom basis (a closed 1-D fiber on S^1).
    """
    cols = [np.ones_like(t)]
    for h in range(1, harmonics + 1):
        cols.append(np.cos(2 * np.pi * h * t))
        cols.append(np.sin(2 * np.pi * h * t))
    return np.stack(cols, axis=1)


def _rss_r2(X: np.ndarray, Xhat: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    ss_tot = float((Xc * Xc).sum())
    ss_res = float(((X - Xhat) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ----------------------------------------------------------------------------
# MANIFOLD dictionary: K curved 1-D fibers, top_k=1 sparse coding.
#
# Scalable two-stage encode (so it reaches K=32,000):
#   Stage 1 (coarse): nearest atom by fiber CENTROID (= constant decoder term
#            B[:,0], since the periodic harmonics integrate to zero over a
#            period). Chunked over K -> O(n K p), same order as a linear SAE
#            encode, never materializing a full n×K matrix.
#   Stage 2 (fine): for each row's top-M candidate atoms, grid-search the fiber
#            coordinate t -> O(n M t_grid p), negligible.
# Decode fits each USED atom's spline independently (O(sum_k n_k b^2)).
# ----------------------------------------------------------------------------
def _topm_by_center(X, xsq, B0, topm, log_chunk):
    """Return (idx (n,M), ) of the M nearest atom centroids per row, chunked."""
    n = X.shape[0]
    K = B0.shape[0]
    b0sq = (B0 * B0).sum(axis=1)                                  # (K,)
    idx = np.full((n, topm), -1, dtype=np.int64)
    dst = np.full((n, topm), np.inf)
    chunk = max(1, min(K, log_chunk))
    ar = np.arange(n)
    for k0 in range(0, K, chunk):
        k1 = min(K, k0 + chunk)
        # partial squared distance (drop xsq, constant per row): ||b||^2 - 2 x.b
        d = b0sq[k0:k1][None, :] - 2.0 * (X @ B0[k0:k1].T)       # (n, kc)
        kc = k1 - k0
        cat_d = np.concatenate([dst, d], axis=1)                 # (n, M+kc)
        cat_i = np.concatenate(
            [idx, np.broadcast_to(np.arange(k0, k1), (n, kc))], axis=1)
        keep = np.argpartition(cat_d, topm - 1, axis=1)[:, :topm]
        dst = np.take_along_axis(cat_d, keep, axis=1)
        idx = np.take_along_axis(cat_i, keep, axis=1)
    return idx


def train_manifold(X, K, harmonics, iters, t_grid, seed, log, topm=6):
    n, p = X.shape
    rng = np.random.default_rng(seed)
    b = 1 + 2 * harmonics
    B = np.zeros((K, b, p))
    init_rows = rng.integers(0, n, size=K)
    B[:, 0, :] = X[init_rows]  # centroid init; harmonics start at 0
    tgrid = np.linspace(0.0, 1.0, t_grid, endpoint=False)
    Phi_grid = periodic_phi(tgrid, harmonics)                    # (t_grid, b)
    xsq = (X * X).sum(axis=1)
    ar = np.arange(n)
    # RAM bounds (never explode the laptop): keep every intermediate under ~50M
    # floats (~400 MB). Stage-1 intermediate is n×kc; stage-2 is rc×M×t_grid×p.
    coarse_chunk = max(256, 50_000_000 // max(1, n))
    row_chunk = max(256, 50_000_000 // max(1, topm * t_grid * p))
    best_r2 = -np.inf
    for it in range(iters):
        # STAGE 1: top-M candidate atoms by centroid (chunked over K).
        cand = _topm_by_center(X, xsq, B[:, 0, :], topm, coarse_chunk)  # (n,M)
        # STAGE 2: fine t-search over the M candidates (chunked over rows).
        assign = np.zeros(n, dtype=np.int64)
        tcoord = np.zeros(n)
        for r0 in range(0, n, row_chunk):
            r1 = min(n, r0 + row_chunk)
            cB = B[cand[r0:r1]]                                   # (rc,M,b,p)
            fib = np.einsum("tb,rmbp->rmtp", Phi_grid, cB)        # (rc,M,t_grid,p)
            d2 = (fib * fib).sum(axis=3) - 2.0 * np.einsum(
                "rp,rmtp->rmt", X[r0:r1], fib)
            jbest = d2.reshape(r1 - r0, topm * t_grid).argmin(axis=1)
            mbest, tbest = np.divmod(jbest, t_grid)
            assign[r0:r1] = cand[r0:r1][np.arange(r1 - r0), mbest]
            tcoord[r0:r1] = tgrid[tbest]
        # DECODE: per-atom spline LSQ over USED atoms only. Group rows by atom
        # with a single argsort (O(n log n)) instead of an O(used*n) scan, so
        # this stays cheap at K=32,000.
        order = np.argsort(assign, kind="stable")
        sa = assign[order]
        used_atoms, starts = np.unique(sa, return_index=True)
        bounds = np.append(starts, n)
        eye = 1e-6 * np.eye(b)
        for gi, k in enumerate(used_atoms):
            rows = order[bounds[gi]:bounds[gi + 1]]
            Phi = periodic_phi(tcoord[rows], harmonics)          # (n_k, b)
            G = Phi.T @ Phi + eye
            B[k] = np.linalg.solve(G, Phi.T @ X[rows])
        Xhat = np.einsum("nb,nbp->np", periodic_phi(tcoord, harmonics),
                         B[assign])
        r2 = _rss_r2(X, Xhat)
        best_r2 = max(best_r2, r2)
        log(f"  [manifold] iter {it+1}/{iters}: R2={r2:.4f} "
            f"best={best_r2:.4f} atoms_used={used_atoms.size}/{K}")
    return best_r2, B, assign, tcoord


# ----------------------------------------------------------------------------
# LINEAR dictionary: K straight lines z*d_k, top_k=1 (traditional linear SAE).
# ----------------------------------------------------------------------------
def train_linear(X, K, iters, seed, log):
    n, p = X.shape
    rng = np.random.default_rng(seed + 777)
    # Init directions from random rows (normalized).
    idx = rng.integers(0, n, size=K)
    D = X[idx].copy()
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    assign = np.zeros(n, dtype=np.int64)
    for it in range(iters):
        # ENCODE: assign row to atom with largest |<x, d_k>| (top_k=1), z = proj.
        # coeff (n,K) = X @ D^T ; reconstruction err for line = ||x||^2 - coeff^2.
        best = np.full(n, -np.inf)
        chunk = max(1, min(K, 4_000_000 // (p + 1)))
        coeff_best = np.zeros(n)
        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            c = X @ D[k0:k1].T                                   # (n, kc)
            c2 = c * c
            local_k = c2.argmax(axis=1)
            local_v = c2[np.arange(n), local_k]
            improve = local_v > best
            best[improve] = local_v[improve]
            assign[improve] = k0 + local_k[improve]
            coeff_best[improve] = c[np.arange(n), local_k][improve]
        # DECODE: each atom = top principal direction of its assigned rows
        # (best rank-1 line through the origin for a top_k=1 code).
        for k in range(K):
            rows = np.nonzero(assign == k)[0]
            if rows.size == 0:
                continue
            Xk = X[rows]
            # top singular vector
            u, s, vt = np.linalg.svd(Xk, full_matrices=False)
            D[k] = vt[0]
        Xhat = coeff_best[:, None] * D[assign]
        r2 = _rss_r2(X, Xhat)
        used = len(np.unique(assign))
        log(f"  [linear]   iter {it+1}/{iters}: R2={r2:.4f} atoms_used={used}/{K}")
    return r2, D, assign


def train_linear_gam(X, K, active, epochs, minibatch, log):
    """Traditional linear SAE via gam's own scalable minibatch trainer
    (`sparse_dictionary_fit`): K-tiled routing, never materializes N×K, the
    real 'traditional linear SAE' baseline. Returns held-in EV (1-RSS/TSS)."""
    from gamfit import sparse_dictionary_fit
    t = time.perf_counter()
    fit = sparse_dictionary_fit(
        np.ascontiguousarray(X, dtype=np.float32), int(K),
        active=int(active), minibatch=int(minibatch), max_epochs=int(epochs),
    )
    r2 = float(fit.explained_variance)
    log(f"  [linear-gam] sparse_dictionary_fit K={K} active={active}: "
        f"EV={r2:.4f} epochs={fit.epochs} converged={fit.converged} "
        f"({time.perf_counter()-t:.1f}s)")
    return r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--n", type=int, default=None, help="rows (default 2*K)")
    ap.add_argument("--concepts", type=int, default=None, help="true concepts (default K)")
    ap.add_argument("--n-active", type=int, default=1)
    ap.add_argument("--harmonics", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--t-grid", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear", choices=["gam", "numpy", "both"], default="gam",
                    help="linear baseline: gam=sparse_dictionary_fit (real "
                         "traditional linear SAE), numpy=K-lines")
    ap.add_argument("--minibatch", type=int, default=512)
    args = ap.parse_args()

    K = args.k
    n = args.n if args.n is not None else 2 * K
    concepts = args.concepts if args.concepts is not None else K
    t0 = time.perf_counter()

    def log(m):
        print(f"[{time.perf_counter()-t0:6.1f}s] {m}", flush=True)

    log(f"# alt-min manifold-vs-linear SAE  K={K} p={args.p} n={n} "
        f"concepts={concepts} harmonics={args.harmonics} noise={args.noise}")
    X = make_curved_data(n, args.p, concepts, args.n_active, args.harmonics,
                         args.noise, args.seed)
    log(f"data ready: X{X.shape}  {X.nbytes/1e6:.1f} MB")

    r2_lin_gam = r2_lin_np = None
    if args.linear in ("gam", "both"):
        r2_lin_gam = train_linear_gam(X, K, args.n_active, args.iters,
                                      args.minibatch, log)
    if args.linear in ("numpy", "both"):
        r2_lin_np, _, _ = train_linear(X, K, args.iters, args.seed, log)
    r2_man, _, _, _ = train_manifold(X, K, args.harmonics, args.iters,
                                     args.t_grid, args.seed, log)
    log(f"MANIFOLD final R2 = {r2_man:.4f}")

    # The baseline to beat is the STRONGEST linear result available.
    lin_vals = [v for v in (r2_lin_gam, r2_lin_np) if v is not None]
    r2_lin = max(lin_vals)

    print("\n=== RESULT ===")
    print(f"K={K}  n={n}  p={args.p}  harmonics={args.harmonics}  active={args.n_active}")
    if r2_lin_gam is not None:
        print(f"  linear SAE (gam sparse_dictionary_fit) EV = {r2_lin_gam:.4f}")
    if r2_lin_np is not None:
        print(f"  linear SAE (numpy K-lines)            R2 = {r2_lin_np:.4f}")
    print(f"  manifold SAE (curved 1-D fibers)      R2 = {r2_man:.4f}")
    verdict = "MANIFOLD BEATS LINEAR" if r2_man > r2_lin else "linear >= manifold"
    print(f"  ==> {verdict}  (delta = {r2_man - r2_lin:+.4f} vs strongest linear)")
    return 0 if r2_man > r2_lin else 2


if __name__ == "__main__":
    raise SystemExit(main())
