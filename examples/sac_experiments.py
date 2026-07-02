"""Discriminating experiments for the SAC thesis (SAC_PLAN Part 2 / Part 5).

Exp 1 (this file, laptop): planted two-circles. Joint ``sae_manifold_fit(K=2)``
vs the SAC prototype. SAC must match the planted truth the joint fit passes:
per-circle EV and cyclic ordering (recovered phase monotone in planted angle).

Exp 2 (W6 OLMo K=8) runs on node2 via ``sac_w6_runner.py``.
"""

from __future__ import annotations

import argparse

import numpy as np

import gamfit
from sac_prototype import _ev, sac_fit


def planted_two_circles(
    n: int = 1500, p: int = 32, noise: float = 0.03, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Two circles in orthogonal 2D subspaces; each row lies on exactly one.

    Returns ``(X, assign, theta, scales)`` where ``assign[i] in {0,1}`` is the
    planted circle for row i, ``theta[i]`` its planted angle, and ``scales`` the
    per-circle amplitude (so the two circles carry a known EV split).
    """
    rng = np.random.default_rng(seed)
    scales = [1.0, 0.85]
    # Orthonormal frames for the two circles in disjoint coordinate blocks.
    frames = []
    q, _ = np.linalg.qr(rng.standard_normal((p, 4)))
    frames.append((q[:, 0], q[:, 1]))   # circle 0 spans dims 0,1 of the frame
    frames.append((q[:, 2], q[:, 3]))   # circle 1 spans dims 2,3

    assign = rng.integers(0, 2, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = np.zeros((n, p), dtype=np.float64)
    for c in (0, 1):
        m = assign == c
        u, v = frames[c]
        x[m] = scales[c] * (
            np.sin(theta[m])[:, None] * u[None, :]
            + np.cos(theta[m])[:, None] * v[None, :]
        )
    x += noise * rng.standard_normal((n, p))
    return np.ascontiguousarray(x.astype(np.float32)), assign, theta, scales


def circular_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Resultant-length circular association of two angle arrays, best sign.

    Returns max over reflection of ``|mean(exp(i*(a -/+ b)))|`` in [0, 1]: 1.0 =
    a rigid rotation/reflection maps one to the other (cyclic order preserved).
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    plus = np.abs(np.mean(np.exp(1j * (a - b))))
    minus = np.abs(np.mean(np.exp(1j * (a + b))))
    return float(max(plus, minus))


def _coord_angle(coords: np.ndarray) -> np.ndarray:
    """Map an atom's recovered on-chart coordinates to an angle.

    Circle atoms report a 1D normalized phase; if two columns are present treat
    them as (x, y) and take atan2.
    """
    c = np.asarray(coords, np.float64)
    if c.ndim == 2 and c.shape[1] >= 2:
        return np.arctan2(c[:, 1], c[:, 0])
    # 1D phase: scale to radians if it looks normalized to [0,1) or already rad.
    v = c.ravel()
    span = float(np.nanmax(v) - np.nanmin(v)) if v.size else 0.0
    return (v * 2.0 * np.pi) if span <= 1.5 else v


def evaluate(result_recon_per_atom, assign, theta, X, label: str) -> dict:
    """Per-atom: which circle it captured, its captured EV, cyclic-order corr."""
    out = {"label": label, "atoms": []}
    for k, (recon, coords, gate) in enumerate(result_recon_per_atom):
        # Which planted circle does this atom explain? The one whose rows it
        # reconstructs best (largest per-circle EV of this atom's recon).
        best_c, best_ev = -1, -np.inf
        for c in (0, 1):
            m = assign == c
            ev_c = _ev(X[m], recon[m])
            if ev_c > best_ev:
                best_c, best_ev = c, ev_c
        m = assign == best_c
        corr = circular_corr(theta[m], _coord_angle(coords[m]))
        out["atoms"].append(
            {"atom": k, "captured_circle": best_c, "circle_ev": float(best_ev),
             "cyclic_corr": corr, "gate_mass": float(np.mean(gate))}
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--d-atom", type=int, default=1,
                    help="circle is a 1-manifold -> d_atom=1 (matches W6 baseline)")
    ap.add_argument("--srp", type=int, default=0,
                    help="structured_residual_passes (Sigma whitening); 0 matches "
                         "the joint baseline for a fair comparison")
    ap.add_argument("--backfit", type=int, default=1)
    ap.add_argument("--isometry", type=float, default=1.0,
                    help="isometry_weight; 0 drops the whitening gauge (faster "
                         "outer convergence on clean planted data)")
    args = ap.parse_args()

    X, assign, theta, scales = planted_two_circles(args.n, args.p, args.noise, args.seed)
    print(f"[exp1] planted two circles: X={X.shape} scales={scales} "
          f"noise={args.noise}; circle counts={np.bincount(assign).tolist()}")
    tss = float(np.sum((X - X.mean(0)) ** 2))
    for c in (0, 1):
        m = assign == c
        ev_full = _ev(X, np.where(m[:, None], X, X.mean(0)))
        print(f"[exp1] planted circle {c}: rows={int(m.sum())} "
              f"variance share≈{float(np.sum((X[m]-X.mean(0))**2))/tss:.3f}")

    # ---- Joint K=2 baseline (the call SAC replaces) --------------------- #
    print("\n[exp1] === JOINT sae_manifold_fit(K=2) ===")
    joint = gamfit.sae_manifold_fit(
        X, K=2, d_atom=args.d_atom, atom_topology="circle", assignment="ibp_map",
        isometry_weight=args.isometry, n_iter=args.n_iter, random_state=args.seed,
    )
    joint_recon = np.asarray(joint.reconstruct(X), dtype=np.float64)
    joint_ev = _ev(X, joint_recon)
    print(f"[exp1] joint combined EV = {joint_ev:.4f}")
    joint_atoms = [
        (np.asarray(joint.atom_reconstruct(X, k), np.float64),
         joint.coords[k],
         np.asarray(joint.atoms[k].assignments, np.float64))
        for k in range(len(joint.atoms))
    ]
    joint_eval = evaluate(joint_atoms, assign, theta, X, "joint_K2")
    for a in joint_eval["atoms"]:
        print(f"    atom {a['atom']}: circle={a['captured_circle']} "
              f"circle_EV={a['circle_ev']:.3f} cyclic_corr={a['cyclic_corr']:.3f} "
              f"gate={a['gate_mass']:.3f}")

    # ---- SAC prototype -------------------------------------------------- #
    print("\n[exp1] === SAC prototype ===")
    sac = sac_fit(
        X, max_atoms=6, d_atom=args.d_atom, atom_topology="circle",
        assignment="ibp_map", ev_floor=5e-3, structured_residual_passes=args.srp,
        n_iter=args.n_iter, backfit_sweeps=args.backfit, isometry_weight=args.isometry,
        random_state=args.seed, verbose=True,
    )
    print(f"[exp1] SAC accepted K={sac.k} atoms; combined EV = {sac.combined_ev:.4f}")
    print(f"[exp1] SAC EV trace = {[round(e, 4) for e in sac.ev_trace]}")
    sac_atoms = [(a.recon, a.coords, a.assignments) for a in sac.atoms]
    sac_eval = evaluate(sac_atoms, assign, theta, X, "sac")
    for a in sac_eval["atoms"]:
        print(f"    atom {a['atom']}: circle={a['captured_circle']} "
              f"circle_EV={a['circle_ev']:.3f} cyclic_corr={a['cyclic_corr']:.3f} "
              f"gate={a['gate_mass']:.3f}")

    # ---- Verdict -------------------------------------------------------- #
    circles_found = {a["captured_circle"] for a in sac_eval["atoms"]
                     if a["circle_ev"] > 0.3}
    corr_ok = all(a["cyclic_corr"] > 0.8 for a in sac_eval["atoms"]
                  if a["circle_ev"] > 0.3)
    print("\n[exp1] VERDICT:")
    print(f"    joint combined EV      = {joint_ev:.4f}")
    print(f"    SAC   combined EV      = {sac.combined_ev:.4f}")
    print(f"    SAC found both circles = {circles_found == {0, 1}}")
    print(f"    SAC cyclic order OK    = {corr_ok}")
    print(f"    SAC EV monotone        = "
          f"{all(np.diff(sac.ev_trace[:sac.k]) >= -1e-6)}")


if __name__ == "__main__":
    main()
