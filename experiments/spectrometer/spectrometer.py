#!/usr/bin/env python3
"""Dimension spectrometer on REAL LLM residual-stream activations.

Concept: for data on a d-dimensional feature manifold, a width-K linear sparse
dictionary at active budget s=1 has excess reconstruction loss
    L(K) - sigma^2  ~  c * K^(-2/d).
Sweep K on a doubling ladder, regress log(excess) on log K; slope m gives the
intrinsic dimension  d_hat = -2/m.

Dictionary fit: top-1 MOD ("k-lines"). Each atom is a unit vector (1D subspace
through the origin). Each token is coded by its single best atom (max |proj|);
atoms are refreshed by the closed-form MOD decoder update (sum coeff*x over the
atom's members, divided by sum coeff^2) then renormalized. Dead atoms are
revived onto the worst-reconstructed tokens. This is the trainer already
validated to reproduce theory (-1.90/-0.99 vs -2/-1 for synthetic d=1/2).

Data is read directly from a raw fp32 residual-stream matrix [n_tokens, d]
(env SPEC_DATA), subsampled to SPEC_N rows, centered by the EXACT full-matrix
per-dim mean and scaled so mean||x||^2 == d (slope is scale-invariant; this
just makes losses comparable across models/layers). Configure via env:
  SPEC_DATA  path to the raw .npy residual matrix
  SPEC_N     rows to subsample (default 40000)
  SPEC_SEED  subsample seed (default 0)
  SPEC_OUT   output dir for results.json / PNGs (default alongside SPEC_DATA)
  SPEC_TAG   label for this run (model/layer)
"""
import json, time, os
import numpy as np
from numpy.random import default_rng

EPS = 1e-12
OUT = os.environ.get("SPEC_OUT", os.path.dirname(os.path.abspath(
    os.environ.get("SPEC_DATA", __file__))))


def load_data():
    path = os.environ["SPEC_DATA"]
    N = int(os.environ.get("SPEC_N", 40000))
    seed = int(os.environ.get("SPEC_SEED", 0))
    tag = os.environ.get("SPEC_TAG", os.path.basename(path))
    mm = np.load(path, mmap_mode="r")
    n, d = mm.shape
    # exact full-matrix per-dim mean/var in a streaming pass
    s1 = np.zeros(d, np.float64); s2 = np.zeros(d, np.float64)
    CH = 50000
    for i in range(0, n, CH):
        blk = np.asarray(mm[i:i+CH], dtype=np.float64)
        s1 += blk.sum(0); s2 += (blk * blk).sum(0)
    mean = (s1 / n)
    var = np.maximum(s2 / n - mean * mean, 0.0)
    rng = default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(N, n), replace=False))
    X = np.asarray(mm[idx], dtype=np.float32) - mean.astype(np.float32)[None, :]
    # scale so mean per-token energy == d (comparable, slope-invariant)
    e = float((X.astype(np.float64) ** 2).sum(1).mean())
    X *= np.float32(np.sqrt(d / max(e, EPS)))
    meta = dict(tag=tag, path=path, n_total=int(n), d=int(d), n_sub=int(X.shape[0]),
                seed=seed, mean_norm=float(np.linalg.norm(mean)),
                raw_mean_energy=e, top_var_frac=float(var.max() / var.sum()))
    return X, meta


def _assign(X, D, chunk=4000):
    """Row-chunked top-1 assignment. Returns (best[N], coeff[N]) float32."""
    N = X.shape[0]
    best = np.empty(N, dtype=np.int32)
    coeff = np.empty(N, dtype=np.float32)
    DT = np.ascontiguousarray(D.T)
    for i in range(0, N, chunk):
        proj = X[i:i+chunk] @ DT                        # chunk x K
        b = np.argmax(proj * proj, axis=1)
        best[i:i+chunk] = b
        coeff[i:i+chunk] = proj[np.arange(proj.shape[0]), b]
    return best, coeff


def fit_dict_top1(X, K, n_epochs=25, seed=0, chunk=4000, verbose=False):
    """Top-1 MOD dictionary ("k-lines"). Low-memory, row-chunked.
    Returns (D, mean_residual_energy L(K))."""
    N, p = X.shape
    rng = default_rng(seed)
    D = X[rng.choice(N, K, replace=False)].copy()
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
    xnorm2 = (X * X).sum(1)                              # per-token energy
    prev = np.inf
    for ep in range(n_epochs):
        best, coeff = _assign(X, D, chunk)
        # closed-form MOD refresh: D_k = (sum_i coeff_i x_i)/(sum_i coeff_i^2),
        # accumulated in row-chunks to bound memory.
        numer = np.zeros((K, p), dtype=np.float64)
        denom = np.zeros(K, dtype=np.float64)
        for i in range(0, N, chunk):
            b = best[i:i+chunk]; c = coeff[i:i+chunk]
            np.add.at(numer, b, (c[:, None] * X[i:i+chunk]).astype(np.float64))
            np.add.at(denom, b, (c * c).astype(np.float64))
        alive = denom > EPS
        Dn = numer.copy()
        Dn[alive] = numer[alive] / denom[alive, None]
        nn = np.linalg.norm(Dn, axis=1)
        good = alive & (nn > EPS)
        D[good] = (Dn[good] / nn[good, None]).astype(np.float32)
        # revive dead atoms onto worst-reconstructed tokens
        dead = np.where(~good)[0]
        if dead.size:
            worst = np.argsort(xnorm2 - coeff * coeff)[::-1][:dead.size]
            v = X[worst]
            D[dead] = (v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)).astype(np.float32)
        L = max(float((xnorm2 - coeff * coeff).mean()), 0.0)
        if verbose:
            print(f"    K={K} ep{ep} L={L:.5f} dead={dead.size}", flush=True)
        if abs(prev - L) < 1e-6 * max(prev, 1.0) and ep >= 8:
            break
        prev = L
    best, coeff = _assign(X, D, chunk)
    L = max(float((xnorm2 - coeff * coeff).mean()), 0.0)  # CS => residual>=0
    return D, L


def sweep(X, Ks, n_epochs=25, restarts=2, tag=""):
    """Best-of-restarts L(K) for each K in ladder."""
    out = {}
    for K in Ks:
        t0 = time.time()
        best = np.inf
        for r in range(restarts):
            _, L = fit_dict_top1(X, K, n_epochs=n_epochs, seed=r)
            best = min(best, L)
        out[K] = best
        print(f"  [{tag}] K={K:5d}  L={best:.5f}  ({time.time()-t0:.1f}s)", flush=True)
    return out


# ---- floor / dimension estimation ----------------------------------------
def _safe_step(J, r):
    """Levenberg-style ridged normal-equation solve; robust to singular H."""
    H = J.T @ J
    lam = 1e-6 * (np.trace(H) / H.shape[0] + 1e-12)
    for _ in range(6):
        try:
            return np.linalg.solve(H + lam * np.eye(H.shape[0]), J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 100
    return np.linalg.lstsq(H + lam * np.eye(H.shape[0]), J.T @ r, rcond=None)[0]


def fit_single_power(Ks, Ls, restarts=40, seed=0):
    """L(K) = c*K^(-2/d) + s2  by nonlinear LS with seeded restarts.
    Returns dict with d, c, s2, rmse."""
    Ks = np.asarray(Ks, float); Ls = np.asarray(Ls, float)
    logK = np.log(Ks)
    rng = default_rng(seed)
    best = None
    for _ in range(restarts):
        d0 = rng.uniform(0.5, 40.0)
        s0 = rng.uniform(0.0, max(Ls.min(), 1e-9))
        c0 = max(Ls.max() - s0, 1e-6)
        theta = np.array([np.log(c0), 1.0 / d0, max(s0, 1e-9)])  # [log c, 1/d..., s2]
        for _ in range(400):
            theta[0] = np.clip(theta[0], -30, 30)      # log c
            theta[1] = np.clip(theta[1], 1e-4, 5.0)     # 1/d
            logc, invd, s2 = theta
            s2 = max(s2, 0.0)
            pred = np.exp(logc) * np.exp(-2 * invd * logK) + s2
            r = pred - Ls
            # jacobian
            base = np.exp(logc) * np.exp(-2 * invd * logK)
            J = np.stack([base, base * (-2 * logK), np.ones_like(Ks)], 1)
            step = _safe_step(J, r)
            theta = theta - 0.5 * step
            theta[2] = max(theta[2], 0.0)
        logc, invd, s2 = theta
        pred = np.exp(logc) * np.exp(-2 * invd * logK) + s2
        rmse = float(np.sqrt(np.mean((pred - Ls) ** 2)))
        d = 1.0 / invd if invd > 1e-9 else float('inf')
        if np.isfinite(rmse) and (best is None or rmse < best["rmse"]):
            best = dict(d=d, c=float(np.exp(logc)), s2=float(s2), rmse=rmse)
    if best is None:   # fallback: no-floor OLS on log-log
        m, dd, _ = ols_slope(Ks, Ls, 0.0)
        best = dict(d=dd, c=float(Ls.max()), s2=0.0, rmse=float("nan"))
    return best


def fit_mixture(Ks, Ls, n_comp=2, restarts=60, seed=0):
    """L(K) = sum_j c_j K^(-2/d_j) + s2 by multi-start Gauss-Newton."""
    Ks = np.asarray(Ks, float); Ls = np.asarray(Ls, float)
    logK = np.log(Ks)
    rng = default_rng(seed)
    best = None
    for _ in range(restarts):
        invd = rng.uniform(1.0 / 30, 1.0 / 1.0, n_comp)      # 1/d per comp
        c = rng.uniform(0.1, 1.0, n_comp) * (Ls.max() / n_comp)
        s2 = rng.uniform(0.0, max(Ls.min(), 1e-9))
        theta = np.concatenate([np.log(c), invd, [max(s2, 1e-9)]])
        for _ in range(600):
            theta[:n_comp] = np.clip(theta[:n_comp], -30, 30)
            theta[n_comp:2 * n_comp] = np.clip(theta[n_comp:2 * n_comp], 1e-4, 5.0)
            lc = theta[:n_comp]; iv = theta[n_comp:2 * n_comp]; s2 = max(theta[-1], 0.0)
            comps = np.exp(lc)[None, :] * np.exp(-2 * iv[None, :] * logK[:, None])  # N x J
            pred = comps.sum(1) + s2
            r = pred - Ls
            Jc = comps                                        # d/d(log c_j)
            Jiv = comps * (-2 * logK[:, None])                # d/d(invd_j)
            Js = np.ones((len(Ks), 1))
            J = np.concatenate([Jc, Jiv, Js], 1)
            step = _safe_step(J, r)
            theta = theta - 0.4 * step
            theta[-1] = max(theta[-1], 0.0)
        lc = theta[:n_comp]; iv = theta[n_comp:2 * n_comp]; s2 = max(theta[-1], 0.0)
        comps = np.exp(lc)[None, :] * np.exp(-2 * iv[None, :] * logK[:, None])
        pred = comps.sum(1) + s2
        rmse = float(np.sqrt(np.mean((pred - Ls) ** 2)))
        ds = [float(1.0 / v) if v > 1e-9 else float('inf') for v in iv]
        weights = [float(x) for x in np.exp(lc)]
        if np.isfinite(rmse) and (best is None or rmse < best["rmse"]):
            best = dict(dims=ds, weights=weights, s2=float(s2), rmse=rmse)
    if best is None:
        best = dict(dims=[float("nan")] * n_comp, weights=[float("nan")] * n_comp,
                    s2=0.0, rmse=float("nan"))
    # sort by dimension
    order = np.argsort(best["dims"])
    best["dims"] = [best["dims"][i] for i in order]
    best["weights"] = [best["weights"][i] for i in order]
    return best


def local_slopes(Ks, Ls, s2):
    """log-log slope of excess loss between consecutive rungs (diagnostic)."""
    Ks = np.asarray(Ks, float); Ls = np.asarray(Ls, float)
    ex = Ls - s2
    ok = ex > 0
    logK = np.log(Ks[ok]); logE = np.log(ex[ok])
    sl = np.diff(logE) / np.diff(logK)
    return list(map(float, sl)), list(map(int, Ks[ok]))


def ols_slope(Ks, Ls, s2):
    Ks = np.asarray(Ks, float); Ls = np.asarray(Ls, float)
    ex = Ls - s2
    ok = ex > 0
    x = np.log(Ks[ok]); y = np.log(ex[ok])
    A = np.stack([x, np.ones_like(x)], 1)
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    d = -2.0 / m
    return float(m), float(d), int(ok.sum())


def pca_remove(X, R, seed=0):
    """Project out the top-R principal directions (through the centered origin)."""
    if R == 0:
        return X
    # cov eigendecomposition (p x p); X already centered by full-data mean
    C = (X.T @ X) / X.shape[0]
    w, V = np.linalg.eigh(C.astype(np.float64))
    Vr = V[:, ::-1][:, :R].astype(np.float32)         # top-R eigenvectors
    return (X - (X @ Vr) @ Vr.T).astype(np.float32)


def main():
    t0 = time.time()
    X, meta = load_data()
    N, p = X.shape
    var_total = float((X * X).sum(1).mean())
    print(f"[{meta['tag']}] X {X.shape}  mean||x||^2={var_total:.3f}  "
          f"top_var_frac={meta['top_var_frac']:.4f}  ({meta['n_total']} total toks)", flush=True)

    Ks = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = {"meta": meta, "N": N, "p": p, "var_total": var_total, "Ks": Ks}

    # (1) GLOBAL spectrum
    print("== GLOBAL ==", flush=True)
    Ls = sweep(X, Ks, n_epochs=25, restarts=2, tag="global")
    Lvals = [Ls[K] for K in Ks]
    results["global_L"] = Lvals
    single = fit_single_power(Ks, Lvals)
    results["global_single_power"] = single
    m, d_ols, nrung = ols_slope(Ks, Lvals, single["s2"])
    results["global_ols"] = dict(slope=m, d_hat=d_ols, n_rungs=nrung)
    # sensitivity: drop last rung
    m2, d2, _ = ols_slope(Ks[:-1], Lvals[:-1], single["s2"])
    results["global_ols_drop_last"] = dict(slope=m2, d_hat=d2)
    ls_slopes, ls_Ks = local_slopes(Ks, Lvals, single["s2"])
    results["global_local_slopes"] = dict(Ks=ls_Ks, slopes=ls_slopes)
    for nc in (2, 3):
        results[f"global_mixture_{nc}"] = fit_mixture(Ks, Lvals, n_comp=nc)
    print("GLOBAL single:", single, flush=True)
    print("GLOBAL ols d_hat:", d_ols, "drop-last:", d2, flush=True)
    print("GLOBAL mix2:", results["global_mixture_2"], flush=True)

    # (3) PCA-STRATIFIED: remove top-R subspace, re-run ladder. Residual streams
    # are extremely anisotropic (one massive direction), so include R=1,2 to peel
    # the dominant direction(s), plus deeper heads R=16,64,256.
    results["pca_strata"] = {}
    for R in (0, 1, 2, 16, 64, 256):
        if R == 0:
            continue  # R=0 is the global run above
        print(f"== PCA remove top-{R} ==", flush=True)
        Xr = pca_remove(X, R)
        vr = float((Xr * Xr).sum(1).mean())
        Lr = sweep(Xr, Ks, n_epochs=25, restarts=1, tag=f"R{R}")
        Lrv = [Lr[K] for K in Ks]
        sp = fit_single_power(Ks, Lrv)
        mr, dr, _ = ols_slope(Ks, Lrv, sp["s2"])
        mrd, drd, _ = ols_slope(Ks[:-1], Lrv[:-1], sp["s2"])
        results["pca_strata"][str(R)] = dict(
            var_total=vr, L=Lrv, single_power=sp,
            ols=dict(slope=mr, d_hat=dr), ols_drop_last=dict(slope=mrd, d_hat=drd),
            mixture_2=fit_mixture(Ks, Lrv, n_comp=2))
        print(f"  R={R}: d_hat={dr:.2f} (drop-last {drd:.2f}) s2={sp['s2']:.4f}", flush=True)

    results["global_R0"] = dict(var_total=var_total, L=Lvals, single_power=single,
                                ols=results["global_ols"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
    print(f"WROTE {OUT}/results.json  ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
