"""Stage-1 (guard surgery) acceptance: does the JOINT K=8 fit ignite?

This is the direct Stage-1 kill-test for STAGE1_DIAGNOSIS.md. It runs the EXACT
joint ``sae_manifold_fit(K=8, d_atom=1, circle, ibp_map, isometry_weight=1.0)``
call that timed out on the real W6 OLMo activations (W6: 3x1500s TIMEOUT), under
the surgically-corrected guard stack (absolute-degeneracy null floor, iteration>0
gate, wall restricted to non-finite + absolute degeneracy). Acceptance = the joint
fit now completes with a real finite EV in minutes rather than oscillating on the
miscalibrated collapse wall.

It also runs the planted K=3 coin-flip determinism check: the documented failure
was EV flipping 0.40 <-> 0.00 across seeds on a K=3 joint fit. Under the corrected
guards the per-seed fit must be reproducible (same seed -> identical EV) and no
seed may collapse to ~0.

SPEC.md: CLI flags (no env vars), no wall-clock budgets (n_iter is a solver cap,
not a deadline).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import gamfit


def _ev(x: np.ndarray, recon: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


def _joint_fit(X, K, d_atom, n_iter, seed, isometry):
    t0 = time.time()
    fit = gamfit.sae_manifold_fit(
        X, K=K, d_atom=d_atom, atom_topology="circle", assignment="ibp_map",
        isometry_weight=isometry, n_iter=n_iter, random_state=seed,
    )
    recon = np.asarray(fit.reconstruct(X), dtype=np.float64)
    return fit, _ev(X, recon), recon, time.time() - t0


def run_w6(args) -> bool:
    X = np.ascontiguousarray(np.load(args.w6_cache), dtype=np.float32)
    n, p = X.shape
    print(f"[s1-w6] loaded {args.w6_cache}: X={X.shape} dtype={X.dtype}", flush=True)
    print(f"[s1-w6] JOINT sae_manifold_fit(K={args.k}, d_atom={args.d_atom}, circle, "
          f"ibp_map, isometry_weight={args.isometry}, n_iter={args.n_iter}, "
          f"seed={args.seed})", flush=True)
    fit, ev, recon, dt = _joint_fit(X, args.k, args.d_atom, args.n_iter, args.seed,
                                    args.isometry)
    finite = bool(np.all(np.isfinite(recon)))
    n_atoms = len(fit.atoms)
    print(f"[s1-w6] joint K={args.k} EV = {ev:.4f}  finite_recon={finite}  "
          f"atoms={n_atoms}  wall={dt:.1f}s", flush=True)
    # Per-atom marginal EV + gate mass, so a co-collapse (some atoms carrying no
    # signal) is visible rather than hidden behind the combined number.
    for k in range(n_atoms):
        a_recon = np.asarray(fit.atom_reconstruct(X, k), np.float64)
        a_ev = _ev(X, a_recon)
        gate = float(np.mean(np.asarray(fit.atoms[k].assignments, np.float64)))
        print(f"    atom {k}: marginal_EV={a_ev:+.4f} gate_mass={gate:.4f}", flush=True)
    # Acceptance: finite reconstruction and a real (clearly-positive) combined EV,
    # reached without hanging. A joint fit pinned on the collapse wall returns
    # EV<=0 / non-finite / never completes.
    ok = finite and ev > args.w6_ev_min
    print(f"[s1-w6] VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(EV {ev:.4f} {'>' if ok else '<='} {args.w6_ev_min}, finite={finite})",
          flush=True)
    return ok


def _planted_circles(n_per, k, ambient, noise, seed):
    rng = np.random.default_rng(seed)
    X = noise * rng.standard_normal((n_per * k, ambient)).astype(np.float64)
    for c in range(k):
        ang = np.linspace(0.0, 2.0 * np.pi, n_per, endpoint=False)
        # each circle lives in its own disjoint 2-plane of the ambient space
        i, j = 2 * c, 2 * c + 1
        rows = slice(c * n_per, (c + 1) * n_per)
        X[rows, i] += np.cos(ang)
        X[rows, j] += np.sin(ang)
    return np.ascontiguousarray(X, dtype=np.float32)


def run_k3_determinism(args) -> bool:
    X = _planted_circles(args.k3_n_per, 3, ambient=12, noise=0.05, seed=7)
    print(f"\n[s1-k3] planted 3 disjoint circles: X={X.shape}", flush=True)
    # same seed twice -> must be bit-identical (the #976 determinism requirement)
    _, ev_a, _, _ = _joint_fit(X, 3, args.d_atom, args.n_iter, 11, args.isometry)
    _, ev_b, _, _ = _joint_fit(X, 3, args.d_atom, args.n_iter, 11, args.isometry)
    same_seed_ok = abs(ev_a - ev_b) < 1e-9
    print(f"[s1-k3] same-seed(11) EV: {ev_a:.6f} vs {ev_b:.6f}  "
          f"deterministic={same_seed_ok}", flush=True)
    # a spread of seeds -> none may collapse to ~0 (the old 0.40<->0.00 coin-flip)
    seed_evs = []
    for s in (1, 2, 3, 4, 5):
        _, ev_s, _, _ = _joint_fit(X, 3, args.d_atom, args.n_iter, s, args.isometry)
        seed_evs.append(ev_s)
    lo = min(seed_evs)
    no_flip = lo > args.k3_ev_min
    print(f"[s1-k3] seed EVs = {[round(e, 4) for e in seed_evs]}  min={lo:.4f}  "
          f"no_collapse={no_flip}", flush=True)
    ok = same_seed_ok and no_flip
    print(f"[s1-k3] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w6-cache", default="/dev/shm/w6/cache_K8.npy")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--n-iter", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--isometry", type=float, default=1.0)
    ap.add_argument("--w6-ev-min", type=float, default=0.05)
    ap.add_argument("--k3-n-per", type=int, default=200)
    ap.add_argument("--k3-ev-min", type=float, default=0.05)
    ap.add_argument("--skip-w6", action="store_true")
    ap.add_argument("--skip-k3", action="store_true")
    args = ap.parse_args()

    results = {}
    if not args.skip_w6:
        results["w6_joint_k8"] = run_w6(args)
    if not args.skip_k3:
        results["k3_determinism"] = run_k3_determinism(args)
    print(f"\n[s1] SUMMARY: {results}", flush=True)


if __name__ == "__main__":
    main()
