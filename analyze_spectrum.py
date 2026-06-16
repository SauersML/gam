#!/usr/bin/env python3
"""Refined spectrum analysis for #1026 EV-vs-K: the raw residual stream is dominated
by a single rogue/outlier direction (top-1 EV ~0.99). The structured-minority vs
unstructured-bulk question lives in the RESIDUAL spectrum after removing that rogue
direction. Compute, per saved layer bank:
  - raw spectrum stats (top1, eff_rank, k90/95/99)
  - rogue-removed spectrum stats (project out top-r components, re-spectrum)
  - the EV-vs-K cumulative curve (raw and rogue-removed) to see where it flattens
Reuses the saved .npy banks; CPU-only, no GPU.
"""
import os, sys, json, glob, time
import numpy as np

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def spectrum(X, m=12000, seed=0):
    # eigendecompose the d x d covariance (d=3584) instead of SVD of n x d:
    # far faster and gives the full variance spectrum.
    Xc = X - X.mean(0, keepdims=True)
    if len(Xc) > m:
        idx = np.random.RandomState(seed).choice(len(Xc), m, replace=False)
        Xc = Xc[idx]
    n = len(Xc)
    C = (Xc.T @ Xc) / max(1, n-1)          # d x d
    w = np.linalg.eigvalsh(C)               # ascending
    var = np.clip(w[::-1], 0, None)         # descending, nonneg
    return var

def stats_from_var(var):
    ev = var / var.sum()
    cum = np.cumsum(ev)
    p = var / var.sum()
    eff = float(np.exp(-(p*np.log(p+1e-12)).sum()))
    pr = float((var.sum()**2)/(var**2).sum())
    def kf(f): return int(np.searchsorted(cum, f)+1)
    return {"ev_top1": float(ev[0]), "ev_top5": float(cum[4]), "ev_top10": float(cum[9]),
            "k50": kf(.50), "k90": kf(.90), "k95": kf(.95), "k99": kf(.99),
            "eff_rank": eff, "participation_ratio": pr,
            "cum_curve": [float(x) for x in cum[:128]]}

def main():
    d = sys.argv[1]
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, "resid_L*.npy"))):
        L = os.path.basename(fn).replace("resid_L","").replace(".npy","")
        log("loading", fn)
        X = np.load(fn, mmap_mode="r")
        X = np.asarray(X[:], dtype=np.float64)
        var = spectrum(X)
        raw = stats_from_var(var)
        # rogue-removed: project out top-R principal directions via covariance eigvecs
        Xc = X - X.mean(0, keepdims=True)
        m = min(len(Xc), 12000)
        idx = np.random.RandomState(1).choice(len(Xc), m, replace=False) if len(Xc) > m else np.arange(len(Xc))
        Xs = Xc[idx]
        C = (Xs.T @ Xs) / max(1, m-1)
        w, V = np.linalg.eigh(C)              # ascending; V columns = eigvecs
        for R in (1, 3):
            Vr = V[:, ::-1][:, :R]            # top-R eigvecs (d x R)
            Xres = Xs - (Xs @ Vr) @ Vr.T      # remove top-R subspace
            Cr = (Xres.T @ Xres) / max(1, m-1)
            wr = np.clip(np.linalg.eigvalsh(Cr)[::-1], 0, None)
            raw[f"rogue_removed_top{R}"] = stats_from_var(wr)
        out[f"L{L}"] = {"shape": list(X.shape), "raw": raw}
        r = raw
        log(f"L{L} RAW top1={r['ev_top1']:.4f} eff_rank={r['eff_rank']:.1f} k90={r['k90']} "
            f"| -top1: eff_rank={r['rogue_removed_top1']['eff_rank']:.1f} k90={r['rogue_removed_top1']['k90']} top1={r['rogue_removed_top1']['ev_top1']:.4f} "
            f"| -top3: eff_rank={r['rogue_removed_top3']['eff_rank']:.1f} k90={r['rogue_removed_top3']['k90']}")
    with open(os.path.join(d, "spectrum_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    log("wrote", os.path.join(d, "spectrum_analysis.json"))

if __name__ == "__main__":
    main()
