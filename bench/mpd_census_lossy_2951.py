"""Lossy shared-subspace census (#2951): at a matched second-order KL budget, how many reals does a
basis SHARED by a group of matrices save over the best per-matrix low-rank code?

Distortion of a site: KL_s ~ 1/2 ||B_s^1/2 (W_s - W_hat_s) A_s^1/2||_F^2 (mpd_pertoken_frontier stats).
  per-matrix  Fisher-metric SVD of each matrix; ranks allocated across the group greedily by KL saved
              per real (exact second-order KL of every truncation from the whitened spectra)
  shared      reads share an input basis Q (d x k), writes an output basis; each matrix's coefficients
              are its exact metric-weighted least squares given Q, C = W A Q (Q^T A Q)^-1 (reads), so the
              shared code's KL is exact for any Q. Q is the top-k left/right singular space of the stacked
              whitened matrices (exact when the group shares one input metric: q, k, v, c_fc of a layer)
Reals: rank r of a d_out x d_in matrix costs r (d_out + d_in - r); a shared basis k x d costs
k d - k^2 (its GL(k) gauge is free) plus k d_out per member. Reported at group budgets equal to
the per-matrix code's KL at fractions of its full-rank reals.

usage: mpd_census_lossy_2951.py {vpd4l|pythia70m} OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sl

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402,F401

MODEL, OUT = sys.argv[1:3]
if MODEL == "vpd4l":
    from safetensors.torch import load_file
    from vpd_model import TARGET_DIR, site_names
    sd = load_file(str(TARGET_DIR / "model_step_99999.safetensors"))
    names = site_names()
    W = {n: sd[f"{n}.weight"].double().numpy() for n in names}
    L = 4
else:
    from mpd_pythia_model_2951 import KINDS, load_pythia, site_name
    m = load_pythia("cpu")
    names = [site_name(i, k) for i in range(m.n_layer) for k in KINDS]
    W = {n: m.site(n).W.double().numpy() for n in names}
    L = m.n_layer
z = np.load(Path.home() / f"mpd-data/frontier/stats_{MODEL}.npz")
A = {n: z["A:" + n].astype(np.float64) for n in names}
B = {n: z["B:" + n].astype(np.float64) for n in names}
kind = lambda n: n.split(".")[-1]
layer = lambda n: int(n.split(".")[1])
READS = [n for n in names if kind(n) in ("q_proj", "k_proj", "v_proj", "c_fc")]
WRITES = [n for n in names if kind(n) in ("o_proj", "down_proj")]


def psd_sqrt(M):
    w, E = sl.eigh(M)
    w = np.maximum(w, w.max() * M.shape[0] * np.finfo(np.float32).eps)
    return (E * np.sqrt(w)) @ E.T


Ah = {n: psd_sqrt(A[n]) for n in names}
Bh = {n: psd_sqrt(B[n]) for n in names}
Wt = {n: Bh[n] @ W[n] @ Ah[n] for n in names}
SPEC = {n: sl.svd(Wt[n], compute_uv=False, lapack_driver="gesvd") ** 2 for n in names}


def per_matrix_curve(group):
    """(reals, KL) along the greedy rank allocation: each step adds the rank-1 piece with the largest KL
    drop per real."""
    r = {n: 0 for n in group}
    kl = sum(0.5 * SPEC[n].sum() for n in group)
    reals, curve = 0, [(0, kl)]
    while True:
        best = None
        for n in group:
            if r[n] >= len(SPEC[n]):
                continue
            do, di = W[n].shape
            cost = do + di - 2 * r[n] - 1  # r(do+di-r) increments by do+di-2r-1
            gain = 0.5 * SPEC[n][r[n]] / cost
            if best is None or gain > best[0]:
                best = (gain, n, cost)
        if best is None:
            return curve
        _, n, cost = best
        kl -= 0.5 * SPEC[n][r[n]]
        r[n] += 1
        reals += cost
        curve.append((reals, kl))


def shared_curve(group, side, ks):
    if side == "in":
        M = np.vstack([Wt[n] for n in group])
        _, _, Vt = sl.svd(M, full_matrices=False, lapack_driver="gesvd")
        basis_w = Vt.T  # in whitened coordinates of the FIRST member's A (exact when A is shared)
        Q0 = np.linalg.solve(Ah[group[0]], basis_w)  # back to raw input coordinates: Ah^-1 basis
    else:
        M = np.hstack([Wt[n] for n in group])
        U, _, _ = sl.svd(M, full_matrices=False, lapack_driver="gesvd")
        Q0 = np.linalg.solve(Bh[group[0]], U)
    out = []
    d = Q0.shape[0]
    for k in ks:
        Q = Q0[:, :k]
        kl, reals = 0.0, k * d - k * k
        for n in group:
            if side == "in":
                C = W[n] @ A[n] @ Q @ np.linalg.pinv(Q.T @ A[n] @ Q)
                E = W[n] - C @ Q.T
                reals += k * W[n].shape[0]
            else:
                C = np.linalg.pinv(Q.T @ B[n] @ Q) @ Q.T @ B[n] @ W[n]
                E = W[n] - Q @ C
                reals += k * W[n].shape[1]
            kl += 0.5 * float(np.sum((Bh[n] @ E @ Ah[n]) ** 2))
        out.append((reals, kl, k))
    return out


def reals_at(curve, kl_budget):
    for reals, kl in curve:
        if kl <= kl_budget:
            return reals
    return None


groups = [(f"reads, layer {i}", [n for n in READS if layer(n) == i], "in") for i in range(L)]
groups += [(f"writes, layer {i}", [n for n in WRITES if layer(n) == i], "out") for i in range(L)]
groups += [("q,k across layers", [n for n in READS if kind(n) in ("q_proj", "k_proj")], "in"),
           ("v across layers", [n for n in READS if kind(n) == "v_proj"], "in"),
           ("o across layers", [n for n in WRITES if kind(n) == "o_proj"], "out")]
res = {}
for name, g, side in groups:
    pc = per_matrix_curve(g)
    full = sum(W[n].size for n in g)
    d = W[g[0]].shape[1] if side == "in" else W[g[0]].shape[0]
    ks = sorted({int(x) for x in np.unique(np.round(np.geomspace(4, d, 18)))})
    sc = shared_curve(g, side, ks)
    rows = []
    for frac in (0.05, 0.1, 0.2, 0.35, 0.5):
        budget = reals_at_frac = None
        # the per-matrix code's KL when it spends frac of the full reals
        for reals, kl in pc:
            if reals >= frac * full:
                budget = kl
                reals_at_frac = reals
                break
        sh = min((r for r, kl, _ in sc if kl <= budget), default=None)
        rows.append({"frac": frac, "kl_budget": budget, "per_matrix_reals": reals_at_frac, "shared_reals": sh,
                     "saved_reals": None if sh is None else reals_at_frac - sh})
    res[name] = {"full_reals": full, "rows": rows}
    print(name, json.dumps(rows), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
