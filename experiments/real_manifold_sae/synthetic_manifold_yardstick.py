"""Planted-manifold yardstick: TRUE curve-recovery quality + speed, no real data.

Plants K unit circles in random 2-planes of R^D with top_k-sparse superposed
rows and noise, fits the curved REML engine, and scores what point-cloud EV
cannot see:
  * coordinate fidelity — circular correlation between each atom's fitted
    latent coordinate and the planted angle (Hungarian-matched over atoms);
  * clean-curve EV — reconstruction quality on NOISELESS held-out samples of
    the planted curves (a fit that memorized the noise cloud scores low);
  * wall seconds.
One JSON row per scale; the before/after instrument for the speed levers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def plant(D: int, K: int, n: int, top_k: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    # Random orthonormal 2-frames per atom.
    frames = []
    for _ in range(K):
        a = rng.normal(size=(D, 2))
        q, _ = np.linalg.qr(a)
        frames.append(q)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, K))
    gates = np.zeros((n, K))
    for i in range(n):
        active = rng.choice(K, size=min(top_k, K), replace=False)
        gates[i, active] = rng.uniform(0.5, 1.5, size=len(active))
    x = np.zeros((n, D))
    for k in range(K):
        circ = np.stack([np.cos(angles[:, k]), np.sin(angles[:, k])], axis=1)
        x += gates[:, k:k + 1] * (circ @ frames[k].T)
    x += noise * rng.normal(size=x.shape)
    return x, angles, gates, frames


def circular_corr(theta_fit: np.ndarray, theta_true: np.ndarray) -> float:
    """|corr| between unit vectors of the two angle series, phase/direction free:
    max over sign of the mean resultant of (theta_fit ∓ theta_true)."""
    d1 = np.abs(np.mean(np.exp(1j * (theta_fit - theta_true))))
    d2 = np.abs(np.mean(np.exp(1j * (theta_fit + theta_true))))
    return float(max(d1, d2))


def run_scale(D: int, K: int, n: int, top_k: int, noise: float, seed: int, n_iter: int) -> dict:
    import gamfit

    x, angles, gates, frames = plant(D, K, n, top_k, noise, seed)
    rec: dict = {"D": D, "K": K, "n": n, "top_k": top_k, "noise": noise}
    t0 = time.time()
    try:
        m = gamfit.sae_manifold_fit(
            x, K=K, d_atom=1, atom_topology="circle", top_k=top_k,
            n_iter=n_iter, random_state=0,
        )
        rec["status"] = "survived"
        rec["ev_train"] = float(m.reconstruction_r2)
        # Coordinate fidelity: per fitted atom, best planted-atom circular corr
        # on rows where that planted atom is strongly active; then a Hungarian-
        # style greedy match (K small).
        coords = [np.asarray(c).reshape(len(x), -1)[:, 0] for c in m.coords]
        fid = np.zeros((K, K))
        for kf in range(K):
            for kt in range(K):
                rows = gates[:, kt] > 0.75
                if rows.sum() >= 32:
                    fid[kf, kt] = circular_corr(coords[kf][rows], angles[rows, kt])
        matched = []
        used: set[int] = set()
        for _ in range(K):
            kf, kt = divmod(int(np.argmax(np.where(
                np.isin(np.arange(K), list(used))[None, :], -1.0, fid))), K)
            used.add(kt)
            matched.append(fid[kf, kt])
            fid[kf, :] = -1.0
        rec["coord_fidelity_mean"] = float(np.mean(matched))
        rec["coord_fidelity_min"] = float(np.min(matched))
        # Clean-curve EV: noiseless single-atom probe rows through the SAME model.
        n_probe = 2000
        rng = np.random.default_rng(seed + 1)
        probe_angles = rng.uniform(0, 2 * np.pi, size=n_probe)
        probe_atom = rng.integers(0, K, size=n_probe)
        xc = np.stack([
            np.stack([np.cos(a), np.sin(a)]) @ frames[k].T
            for a, k in zip(probe_angles, probe_atom)
        ])
        recon = np.asarray(m.reconstruct(xc), dtype=np.float64)
        rec["clean_curve_ev"] = float(
            1.0 - ((xc - recon) ** 2).sum() / ((xc - xc.mean(0)) ** 2).sum()
        )
    except Exception as exc:  # noqa: BLE001
        rec["status"] = type(exc).__name__
        rec["error"] = str(exc)[:600]
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/dev/shm/gnome/results/yardstick.jsonl"))
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--n-iter", type=int, default=30)
    args = ap.parse_args()

    scales = [
        dict(D=8, K=2, n=4000, top_k=1, noise=0.05, seed=0),
        dict(D=16, K=4, n=8000, top_k=2, noise=0.05, seed=0),
        dict(D=32, K=8, n=16000, top_k=3, noise=0.05, seed=0),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for sc in scales:
        rec = run_scale(n_iter=args.n_iter, **sc)
        rec["label"] = args.label
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[yardstick] {rec}", flush=True)
    print("[yardstick] DONE", flush=True)


if __name__ == "__main__":
    main()
