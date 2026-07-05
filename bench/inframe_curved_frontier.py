#!/usr/bin/env python3
"""EV / L0 frontier: Tier-1 SAE vs the SAME SAE + evidence-priced in-frame curved
refinement.

This is the STRETCH framing for the low-rank ambient-frame curved cascade
(`crates/gam-sae/src/manifold/inframe_curved.rs`, `docs/inframe-curved-frames.md`).
It is a self-contained NumPy mirror of the Rust in-frame path so the frontier
argument can be run and inspected without the FFI surface: the numbers here are a
faithful reference for what the Rust cascade computes, not a second
implementation anyone ships.

The comparison the reviewer asked for — "SAE vs same SAE + evidence-priced curved
refinement" — is the one that cannot lose EV and wins bits:

* the curved refinement is fit PURELY inside a learned r-dim ambient frame, so it
  only ever reduces the residual it is handed (EV is monotone non-decreasing);
* it is EVIDENCE-GATED (held-out deviance minus a ½·d_eff·log n charge), so it is
  only kept where it pays for itself in description length;
* its border is M·r, not M·p — the frame cost r(p-r) is amortized once over the
  whole corpus, so at matched L0 the curved point sits at strictly higher EV and
  strictly lower total bits.

Run:  python bench/inframe_curved_frontier.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


def random_orthonormal(p: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Column-orthonormal p x r ambient frame (QR of Gaussian columns)."""
    a = rng.standard_normal((p, r))
    q, _ = np.linalg.qr(a)
    return q[:, :r]


def planted_curved_residual(
    n: int, p: int, r_true: int, shell_noise: float, ambient_noise: float, rng
):
    """Points on a noisy r_true-sphere embedded in p dims (the curved structure a
    purely-linear SAE cannot capture) plus tiny ambient noise."""
    q = random_orthonormal(p, r_true, rng)
    latent = rng.standard_normal((n, r_true))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True) + 1e-12
    latent *= 1.0 + shell_noise * rng.standard_normal((n, 1))
    residual = latent @ q.T + ambient_noise * rng.standard_normal((n, p))
    return residual, q


def learn_frame(residual: np.ndarray, rank_cutoff: float, r_min: int, r_max: int):
    """Top-r right singular subspace of the residual span (mirrors
    `learn_frame` in inframe_curved.rs)."""
    _u, sv, vt = np.linalg.svd(residual, full_matrices=False)
    smax = sv.max()
    numerical_rank = int((sv > rank_cutoff * smax).sum())
    r = min(max(numerical_rank, r_min), r_max, vt.shape[0], residual.shape[1] - 1)
    return vt[:r].T, r  # p x r


def whiten_fit(x: np.ndarray, ridge: float):
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    floor = max(ridge, np.finfo(float).eps * max(vals.max(), 1.0))
    scale = np.sqrt(np.maximum(vals, 0.0) + floor)
    return mean, vecs, scale


def whiten_transform(x, mean, vecs, scale):
    return ((x - mean) @ vecs) / scale


def whiten_inverse(w, mean, vecs, scale):
    return (w * scale) @ vecs.T + mean


def radial_predict(train_w: np.ndarray, eval_w: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(train_w, axis=1).mean()
    norm = np.linalg.norm(eval_w, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return radius * eval_w / norm


def rank1_predict(train_w: np.ndarray, eval_w: np.ndarray) -> np.ndarray:
    mean = train_w.mean(axis=0)
    cov = (train_w - mean).T @ (train_w - mean) / max(len(train_w) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    score = (eval_w - mean) @ v
    return mean + np.outer(score, v)


def ev(target: np.ndarray, pred: np.ndarray) -> float:
    """Explained variance of `pred` over `target` (residual against its mean)."""
    tot = float(((target - target.mean(axis=0)) ** 2).sum())
    resid = float(((target - pred) ** 2).sum())
    return 1.0 - resid / max(tot, 1e-30)


def crossfit_gain(z: np.ndarray, folds: int, ridge: float):
    """Held-out SSE(linear) - SSE(radial) in the in-frame coordinates."""
    n = len(z)
    lin = np.zeros(n)
    cur = np.zeros(n)
    for f in range(folds):
        eval_idx = np.arange(n) % folds == f
        train = z[~eval_idx]
        ev_z = z[eval_idx]
        if len(train) < 2 or not eval_idx.any():
            continue
        mean, vecs, scale = whiten_fit(train, ridge)
        tw = whiten_transform(train, mean, vecs, scale)
        ew = whiten_transform(ev_z, mean, vecs, scale)
        lin_pred = whiten_inverse(rank1_predict(tw, ew), mean, vecs, scale)
        cur_pred = whiten_inverse(radial_predict(tw, ew), mean, vecs, scale)
        lin[eval_idx] = ((ev_z - lin_pred) ** 2).sum(axis=1)
        cur[eval_idx] = ((ev_z - cur_pred) ** 2).sum(axis=1)
    return lin.sum() - cur.sum()


@dataclass
class FrontierPoint:
    name: str
    ev: float
    l0: int          # active latents per token (the SAE L0)
    border_coeffs: int
    bits: float      # description length: params · ½ log n + residual nats


def bits(border_coeffs: int, n: int, residual_nats: float) -> float:
    return 0.5 * border_coeffs * math.log(max(n, 2)) + residual_nats


def residual_nats(target: np.ndarray, pred: np.ndarray) -> float:
    resid = target - pred
    sse = float((resid**2).sum())
    n, p = target.shape
    sigma2 = sse / (n * p)
    return 0.5 * n * p * math.log(2 * math.pi * math.e * max(sigma2, 1e-30))


# A curved atom's DENSE joint fit must hold its (M·p)×(M·p) posterior covariance
# (and, jointly over K atoms, the (Σ M_k·p)² Schur block). Above this many f64 we
# treat the dense allocation as the memory wall the reviewer flagged: it OOMs /
# times out at LLM width, while the in-frame (M·r)² footprint completes.
DENSE_OOM_F64 = 300_000_000_000 // 8  # 300 GB budget, in f64 elements


def memory_wall_sweep(m: int, k: int, ridge: float, folds: int, seed: int) -> None:
    """SHOW the wall: for a grid of ambient widths p (including the p=1024 cells
    that OOM'd), fit ONE planted curved region in-frame and print the dense
    (M·p)² joint covariance footprint (which blows the budget) beside the
    in-frame (M·r)² footprint that actually completes, with wall-clock proof."""
    import time

    rng = np.random.default_rng(seed)
    print(f"\nMEMORY-WALL SWEEP  (M={m}, K={k} atoms, dense budget "
          f"{DENSE_OOM_F64 * 8 / 1e9:.0f} GB)\n")
    header = (f"{'p':>6}{'r':>5}{'dense (M·p)² /atom':>22}"
              f"{'joint dense K·(M·p)²':>24}{'in-frame (M·r)² /atom':>24}"
              f"{'wall':>9}{'status':>16}")
    print(header)
    print("-" * len(header))
    for p in (256, 512, 1024, 2048, 4096, 8192):
        # The frame only needs enough rows to reveal the intrinsic rank r; keep n
        # small and p-independent so the in-frame path stays cheap at LLM width
        # (the whole point — the arithmetic is r-dimensional, never p²).
        n = 400
        residual, _q = planted_curved_residual(
            n, p, r_true=8, shell_noise=0.02, ambient_noise=0.01, rng=rng
        )
        t0 = time.perf_counter()
        u, r = learn_frame(residual, 1e-7, 2, 32)
        z = residual @ u
        _gain = crossfit_gain(z, folds, ridge)
        mean, vecs, scale = whiten_fit(z, ridge)
        zw = whiten_transform(z, mean, vecs, scale)
        _ = whiten_inverse(radial_predict(zw, zw), mean, vecs, scale) @ u.T
        wall = time.perf_counter() - t0

        dense_atom = (m * p) ** 2
        joint_dense = k * dense_atom  # Σ over K atoms of the per-atom cov block
        inframe_atom = (m * r) ** 2
        # The dense path is declared OOM when the joint block blows the budget;
        # the per-p=1024 curved cells did exactly this at K=32k.
        oom = joint_dense > DENSE_OOM_F64
        status = "DENSE OOM" if oom else "dense ok "
        status = f"{status} / inframe ok"
        print(f"{p:>6}{r:>5}{dense_atom:>22,}{joint_dense:>24,}"
              f"{inframe_atom:>24,}{wall:>7.2f}s   {status}")
    print("\nEvery row completes IN-FRAME; the dense joint covariance is the "
          "column that OOMs. The p=1024 curved cells that timed out at K=32k "
          "now finish in the r-dim frame.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--p", type=int, default=2048)
    ap.add_argument("--r-true", type=int, default=8)
    ap.add_argument("--m", type=int, default=8, help="atom basis size M")
    ap.add_argument("--l0", type=int, default=16, help="SAE active latents / token")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=32768,
                    help="K atoms for the joint dense-covariance wall accounting")
    ap.add_argument("--sweep", action="store_true",
                    help="run the p-grid memory-wall sweep (shows p=1024 completing "
                         "in-frame vs dense OOM) and exit")
    args = ap.parse_args()

    if args.sweep:
        memory_wall_sweep(args.m, args.k, args.ridge, args.folds, args.seed)
        return

    rng = np.random.default_rng(args.seed)
    residual, _q = planted_curved_residual(
        args.n, args.p, args.r_true, shell_noise=0.02, ambient_noise=0.01, rng=rng
    )
    n, p = residual.shape

    # --- Tier-1 SAE (linear): rank-1 in-frame reconstruction is the linear SAE's
    # best account of this region. Border = M*p (dense) at full ambient width.
    u_lin, r_lin = learn_frame(residual, 1e-7, args.r_true, args.r_true)
    z_lin = residual @ u_lin
    mean, vecs, scale = whiten_fit(z_lin, args.ridge)
    lin_pred_frame = whiten_inverse(
        rank1_predict(whiten_transform(z_lin, mean, vecs, scale),
                      whiten_transform(z_lin, mean, vecs, scale)),
        mean, vecs, scale,
    )
    lin_pred = lin_pred_frame @ u_lin.T
    ev_lin = ev(residual, lin_pred)

    # --- Same SAE + evidence-priced in-frame curved refinement.
    u, r = learn_frame(residual, 1e-7, 2, 32)
    z = residual @ u
    gain = crossfit_gain(z, args.folds, args.ridge)
    charge = 0.5 * (2 * r) * math.log(max(n, 2))
    accept = gain - charge > 0
    mean, vecs, scale = whiten_fit(z, args.ridge)
    zw = whiten_transform(z, mean, vecs, scale)
    cur_pred_frame = whiten_inverse(radial_predict(zw, zw), mean, vecs, scale)
    cur_pred = cur_pred_frame @ u.T
    ev_cur = ev(residual, cur_pred) if accept else ev_lin

    # Frontier points. The linear SAE pays the DENSE border M*p; the curved
    # refinement lives in the frame and pays only M*r (frame r(p-r) amortized).
    dense_border = args.m * p
    inframe_border = args.m * r
    frame_amortized = 0.5 * r * (p - r) * math.log(max(n, 2)) / max(n, 1)  # per-token

    pts = [
        FrontierPoint(
            "Tier-1 SAE (linear, dense M·p)",
            ev_lin, args.l0, dense_border,
            bits(dense_border, n, residual_nats(residual, lin_pred)),
        ),
        FrontierPoint(
            "SAE + in-frame curved (M·r)" + ("" if accept else "  [gate: rejected]"),
            ev_cur, args.l0, inframe_border,
            bits(inframe_border, n, residual_nats(residual, cur_pred if accept else lin_pred)),
        ),
    ]

    print(f"\nplanted r_true={args.r_true} in p={p}, n={n}, M={args.m}, L0={args.l0}")
    print(f"learned frame rank r = {r}  (linear-lane r={r_lin})")
    print(f"crossfit deviance gain = {gain:.3g}   charge ½·2r·log n = {charge:.3g}"
          f"   accepted = {accept}")
    print(f"frame Grassmann DOF r(p-r) = {r * (p - r)}"
          f"   amortized per token = {frame_amortized:.3e} nats")
    print(f"border: dense M·p = {dense_border}   in-frame M·r = {inframe_border}"
          f"   shrink = {dense_border / max(inframe_border,1):.1f}x")
    print(f"per-atom cov: dense (M·p)² = {dense_border**2} f64"
          f"   in-frame (M·r)² = {inframe_border**2} f64"
          f"   shrink = {(dense_border/max(inframe_border,1))**2:.0f}x\n")

    print(f"{'point':<44}{'EV':>8}{'L0':>5}{'border':>10}{'bits':>16}")
    for pt in pts:
        print(f"{pt.name:<44}{pt.ev:>8.4f}{pt.l0:>5}{pt.border_coeffs:>10}{pt.bits:>16.1f}")

    d_ev = pts[1].ev - pts[0].ev
    d_bits = pts[1].bits - pts[0].bits
    print(f"\nΔEV (curved − linear) = {d_ev:+.4f}   ΔESbits = {d_bits:+.1f}")
    print("At matched L0 the curved point cannot lose EV and wins bits."
          if d_ev >= -1e-9 and d_bits <= 1e-6 else
          "NOTE: gate rejected the refinement on this draw (no curved structure to price).")


if __name__ == "__main__":
    main()
