"""Whole-model bits-vs-KL frontier of structure-blind compression (#2951), on the S1 spec: the site
matrices on gam's LatticeCode (vpd_bits), KL(target || program) on the frontier's rows (32 val rows at
offset 1024).

Per matrix, candidate codes, each with its exact code length and its second-order KL
    KL_s ~ 1/2 tr(B_s E A_s E^T),  E = W - W_hat
(A = E[x x^T] at the site input, B = sampled-label Fisher at the output; mpd_pertoken_frontier stats):
  svd    rank-r SVD in the Fisher metric (B^1/2 W A^1/2), factors on their own b-bit lattice
  prune  keep the k entries of largest W_ij^2 A_jj B_ii, on the b-bit lattice; the support is sent as an
         enumerative subset code (gam codec subset_code_len_bits) or as lattice zeros, whichever is shorter
  combo  svd at rank r plus a pruned residual
  dense  rounding at b (prune with every entry kept)
The rank, sparsity and precision of every matrix are chosen jointly by the two-part criterion
    min sum_s bits_s + lam * sum_s KL_s
(for each lam, each matrix takes its own argmin: the marginal bits per unit KL are equal across
matrices), lam swept; each chosen program is then measured for its true KL.

usage: mpd_wholemodel_frontier_2951.py {vpd4l|pythia70m} OUT.json
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpd_bits import delta_len, omega_len, signed_len  # noqa: E402
from vpd_eval import kl_per_pos  # noqa: E402
from vpd_model import val_tokens  # noqa: E402

MODEL, OUT = sys.argv[1:3]
DEV, MB = "mps", 8
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
if MODEL == "vpd4l":
    from vpd_model import load_target, site_names
    model = load_target(DEV)
    names = site_names()
else:
    from mpd_pythia_model_2951 import KINDS, load_pythia, site_name
    model = load_pythia(DEV)
    names = [site_name(i, k) for i in range(model.n_layer) for k in KINDS]
z = np.load(Path.home() / f"mpd-data/frontier/stats_{MODEL}.npz")
A = {n: z["A:" + n].astype(np.float64) for n in names}
B = {n: z["B:" + n].astype(np.float64) for n in names}
W0 = {n: model.site(n).W.cpu().double().numpy() for n in names}

TAB_N = 1 << 20
DELTA_TAB = delta_len(np.arange(1, TAB_N + 1)).astype(np.int64)
omega = lambda n: int(omega_len(np.array([n]))[0])


def prec(x, b):
    r = math.sqrt(float(np.mean(np.square(x))))
    return b - round(math.log2(r)) if r > 0 else b


def qz(x, p):
    return np.round(x * 2.0**p) * 2.0**-p


def idx_bits(k):
    zz = np.where(k >= 0, 2 * k, -2 * k - 1)
    small = zz < TAB_N
    out = int(DELTA_TAB[zz[small]].sum())
    if not small.all():
        out += int(delta_len(zz[~small].astype(np.uint64) + np.uint64(1)).sum())
    return out


def lb(x, p):
    k = np.round(np.asarray(x) * 2.0**p).astype(np.int64).ravel()
    return idx_bits(k) + omega(k.size + 1) + int(signed_len(np.array([p]))[0])


def subset_bits(n, k):
    return omega(k + 1) + math.ceil((math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2) - 1e-9)


def psd_sqrt(M):
    w, E = sl.eigh(M)
    w = np.maximum(w, w.max() * M.shape[0] * np.finfo(np.float32).eps)
    return (E * np.sqrt(w)) @ E.T, (E / np.sqrt(w)) @ E.T


BGRID = [2, 3, 4, 5, 6, 8, 10, 12]  # VPD 4L ran with [2..8]; Pythia h.5 k_proj needs > 8 bits
RANKS = [8, 16, 32, 64, 128, 192, 256, 384, 512, 640]
DENS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.02]
COMBO_R = [32, 64, 128, 256]
COMBO_D = [0.25, 0.1, 0.05, 0.02]


def candidates(n):
    """[(method, bits, kl_pred, W_hat-builder args)] for one site."""
    W = W0[n]
    Ah, Aih = psd_sqrt(A[n])
    Bh, Bih = psd_sqrt(B[n])
    kl = lambda E: 0.5 * float(np.sum((Bh @ E @ Ah) ** 2))
    sal = W**2 * np.outer(np.diag(B[n]), np.diag(A[n]))
    order = np.argsort(-sal, axis=None)
    out = []

    def pruned(R, dens, b, salR):
        """Keep the top dens fraction of R by saliency on the b-bit lattice of the kept entries."""
        n_el = R.size
        k = max(1, int(round(dens * n_el)))
        keep = np.zeros(n_el, bool)
        keep[(order if salR is None else np.argsort(-salR, axis=None))[:k]] = True
        keep = keep.reshape(R.shape)
        p = prec(R[keep], b)
        Rq = np.where(keep, qz(R, p), 0.0)
        kidx = np.round(Rq[keep] * 2.0**p).astype(np.int64)
        lattice_all = lb(Rq, p)
        sparse = subset_bits(n_el, k) + idx_bits(kidx) + omega(k + 1) + int(signed_len(np.array([p]))[0])
        return Rq, 1 + min(lattice_all, sparse), p

    for dens in DENS:
        for b in BGRID:
            Wq, bits, p = pruned(W, dens, b, None)
            out.append(("dense" if dens == 1.0 else "prune", bits, kl(W - Wq), {"dens": dens, "b": b}))
    Pm, s, Qt = sl.svd(Bh @ W @ Ah, full_matrices=False, lapack_driver="gesvd")
    for r in RANKS:
        if r >= len(s):
            continue
        Lf, Rf = (Bih @ Pm[:, :r]) * np.sqrt(s[:r]), np.sqrt(s[:r])[:, None] * (Qt[:r] @ Aih)
        for b in BGRID:
            pl, pr = prec(Lf, b), prec(Rf, b)
            Wl = qz(Lf, pl) @ qz(Rf, pr)
            bits = omega(r + 1) + lb(Lf, pl) + lb(Rf, pr)
            out.append(("svd", bits, kl(W - Wl), {"r": r, "b": b}))
            if r in COMBO_R and b in (3, 4, 5):
                Res = W - Wl
                for dens in COMBO_D:
                    for bs in (3, 4, 5):
                        Sq, sb, _ = pruned(Res, dens, bs, Res**2 * np.outer(np.diag(B[n]), np.diag(A[n])))
                        out.append(("combo", bits + sb, kl(Res - Sq), {"r": r, "b": b, "dens": dens, "bs": bs}))
    return out


_dec = {}


def decomp(n):
    """Per-site whitening and Fisher-metric SVD, computed once."""
    if n not in _dec:
        Ah, Aih = psd_sqrt(A[n])
        Bh, Bih = psd_sqrt(B[n])
        _dec[n] = (Bih, Aih, sl.svd(Bh @ W0[n] @ Ah, full_matrices=False, lapack_driver="gesvd"))
    return _dec[n]


def build(n, c):
    """Reconstruct W_hat for a chosen candidate (same arithmetic as candidates())."""
    W = W0[n]
    method, _, _, a = c

    def pruned(R, dens, b, salR):
        k = max(1, int(round(dens * R.size)))
        keep = np.zeros(R.size, bool)
        keep[np.argsort(-salR, axis=None)[:k]] = True
        keep = keep.reshape(R.shape)
        return np.where(keep, qz(R, prec(R[keep], b)), 0.0)

    sal = lambda R: R**2 * np.outer(np.diag(B[n]), np.diag(A[n]))
    if method in ("dense", "prune"):
        return pruned(W, a["dens"], a["b"], sal(W))
    Bih, Aih, (Pm, s, Qt) = decomp(n)
    r = a["r"]
    Lf, Rf = (Bih @ Pm[:, :r]) * np.sqrt(s[:r]), np.sqrt(s[:r])[:, None] * (Qt[:r] @ Aih)
    Wl = qz(Lf, prec(Lf, a["b"])) @ qz(Rf, prec(Rf, a["b"]))
    if method == "svd":
        return Wl
    return Wl + pruned(W - Wl, a["dens"], a["bs"], sal(W - Wl))


cache = Path.home() / f"mpd-data/frontier/wm_candidates_{MODEL}.json"
if cache.exists():
    CAND = json.load(open(cache))
else:
    CAND = {}
    for n in names:
        CAND[n] = candidates(n)
        log(f"{n}: {len(CAND[n])} candidates")
    json.dump(CAND, open(cache, "w"))

ids = val_tokens(32, offset=1024)


@torch.no_grad()
def true_kl(What):
    orig = {n: model.site(n).W for n in names}
    tot = 0.0
    for i in range(0, 32, MB):
        b = ids[i:i + MB].to(DEV)
        tgt = model(b)
        for n in names:
            model.site(n).W = torch.tensor(What[n], dtype=torch.float32, device=DEV)
        try:
            tot += kl_per_pos(model(b), tgt).mean().item() / (32 // MB)
        finally:
            for n in names:
                model.site(n).W = orig[n]
    return tot


results = json.load(open(OUT)) if os.path.exists(OUT) else {}
METHODS = {"dense": ("dense",), "svd": ("svd",), "prune": ("dense", "prune"), "combo": ("dense", "prune", "svd", "combo")}
for meth, allowed in METHODS.items():
    if meth in results:
        continue
    pts, seen = [], set()
    for lam in np.logspace(5, 12, 22):
        pick = {n: min((c for c in CAND[n] if c[0] in allowed), key=lambda c: c[1] + lam * c[2]) for n in names}
        key = tuple(json.dumps(pick[n][3], sort_keys=True) + pick[n][0] for n in names)
        if key in seen:
            continue
        seen.add(key)
        bits = sum(pick[n][1] for n in names)
        klp = sum(pick[n][2] for n in names)
        What = {n: build(n, pick[n]) for n in names}
        r = {"lam": float(lam), "bits": bits, "kl_pred": klp, "kl": true_kl(What),
             "mix": {m: sum(pick[n][0] == m for n in names) for m in allowed}}
        pts.append(r)
        log(f"{meth} lam {lam:.3g}: bits {bits:.4g} KL pred {klp:.4g} true {r['kl']:.4g} {r['mix']}")
    results[meth] = pts
    json.dump(results, open(OUT, "w"), indent=1)
log("done")
