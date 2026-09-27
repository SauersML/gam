"""#2951 probe: does an exact-GELU MLP split into (approximately) independent parallel modules?

Analysis under SPEC 8's exception (torch execution of a measurement), float64 on CPU throughout.

Exact identity (checked numerically below): gelu(t) = t Phi(t) = t/2 + psi(t), psi(t) = t (Phi(t) - 1/2) even,
psi' odd, psi'' = (2 - t^2) phi. A biased MLP F(x) = W_out gelu(W_in x + b_in) + b_out therefore has the normal form

    F(x) = A x + b + sum_j u_j psi(a_j^T x + beta_j),   A = 1/2 W_out W_in,   b = 1/2 W_out b_in + b_out,

with a_j = row j of W_in, u_j = column j of W_out, beta_j = b_in[j]. Duplicate units (a_k = +-a_j, beta_k = +-beta_j)
would merge with u_j + u_k since psi is even; they are counted and reported (none occur in Pythia).

Replacement independence of (P, Q): F(P x + (I-P) y) = Q F(x) + (I-Q) F(y) for all x, y. Given linear independence
of {1, psi'(a_j^T x + beta_j)} (checked numerically below on random inputs), it holds iff Q C_j = C_j P for
C_0 = A and C_j = u_j a_j^T. For rank-one C_j and projectors this forces a_j in ran P <=> u_j in ran Q, else
a_j in ker P and u_j in ker Q, so exact splits are unions of connected components of the unit graph with edges
<a_j,a_k> != 0 or <u_j,u_k> != 0 (A = 1/2 sum_j C_j then commutes automatically when no sign-merges occur).

Certified bound (exact algebra): with v = x - y, D_j = Q u_j a_j^T - u_j a_j^T P,
    || F(Px+(I-P)y) - Q F(x) - (I-Q) F(y) || <= eta ||v||,   eta = ||QA - AP||_2 + kappa sum_j ||D_j||_2,
kappa = max|psi'| = Phi(sqrt2) - 1/2 + sqrt2 phi(sqrt2). ||D_j||_2 = max(||Q u|| ||(I-P) a||, ||(I-Q) u|| ||P a||)
because D_j = Qu ((I-P)a)^T - (I-Q)u (Pa)^T has orthogonal column and row pieces.

Measured per layer (Pythia 70m/160m): exact-zero unit graph, |cos| overlap distributions vs a Gaussian null,
thresholded components, spectral clusterings into K = 2..8 groups, and for each group G vs its complement the
certified eta and the empirical replacement discrepancy on real post-LayerNorm MLP inputs, against a random-split
null of the same sizes (same P, Q construction) and Haar-random P, Q of the same ranks. Then a Lloyd-style local
search (refine) directly on the proxy at fixed ranks k in {d/4, d/2, spectral}, started from random and spectral
groups, compared with the same search (random starts) on a random twin: unit norms and biases kept, read and write
directions redrawn isotropically.
P, Q construction: positive eigenspace of M_a = sum_{j in G} |u_j|^2 a_j a_j^T - sum_{j notin G} |u_j|^2 a_j a_j^T
(the rank-k minimiser of the squared proxy sum |u|^2 |(I-P)a|^2 over G + |u|^2 |Pa|^2 over the rest), Q likewise
from M_u with weights |a_j|^2, with k = round(d |G| / n) for every split and null alike (P = Q = 0 or I would be
trivially exact, so the rank must be pinned).

--mode pi also runs on a SwiGLU decoder (read through bench/mpd_opfirst_decoder_2951.py, weights only, one
layer at a time) with --read up|gate as the read matrix W. That is NOT covered by the theorem above: a SwiGLU
unit reads two directions (gate and up) and is bilinear-gated, so there is no single-read normal form and Pi's
components are not the exact module splits. The receipt labels it; what remains exact is only the geometry of
the chosen read rows: E* is the whitened distance to the nearest read matrix whose rows split into two
mutually orthogonal subspaces, compared with random subsets and with the same proposer on a twin whose unit
directions are redrawn.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

KAPPA = 0.5 * (1 + math.erf(1.0)) - 0.5 + math.sqrt(2) * math.exp(-1.0) / math.sqrt(2 * math.pi)

FALLBACK_TEXT = (
    "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools "
    "and condenses into clouds, and falls again to the surface as precipitation. Plants take up water through their "
    "roots and release it through their leaves in a process called transpiration. In 1854 John Snow traced a cholera "
    "outbreak in London to a single public water pump on Broad Street, an early triumph of epidemiology. Students "
    "learning algebra often begin with linear equations such as 3x + 5 = 20, then move to quadratic equations whose "
    "solutions are given by the quadratic formula. "
)


def texts_tokens(tok, n_seq, seq_len, seed):
    try:
        from mpd_llm_chart_restriction_2951 import token_batches
        ids = next(token_batches(tok, seq_len, n_seq, seed, 0))
        return ids, "HuggingFaceFW/fineweb-edu sample-10BT (streamed, shuffle seed %d)" % seed
    except Exception as exc:  # offline fallback, recorded in the receipt
        buf = tok(FALLBACK_TEXT * 40).input_ids
        return torch.tensor(buf[: n_seq * seq_len]).view(n_seq, seq_len), "fallback fixed text (%s)" % type(exc).__name__


def Phi(t):
    return 0.5 * (1 + torch.erf(t / math.sqrt(2)))


def psi(t):
    return t * (Phi(t) - 0.5)


def dpsi(t):
    return Phi(t) - 0.5 + t * torch.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)


def q(x):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    # numpy's quantile (same linear interpolation): torch.quantile refuses inputs above 2^24 elements
    qs = np.quantile(x.numpy(), [0.5, 0.9, 0.99]) if x.numel() else [float("nan")] * 3
    return {"median": float(qs[0]), "p90": float(qs[1]), "p99": float(qs[2]), "max": float(x.max()),
            "mean": float(x.mean())}


class Mlp:
    def __init__(self, W_in, b_in, W_out, b_out):
        self.W_in, self.b_in, self.W_out, self.b_out = W_in, b_in, W_out, b_out
        self.A = 0.5 * W_out @ W_in
        self.b = 0.5 * W_out @ b_in + b_out

    def direct(self, X):
        return torch.nn.functional.gelu(X @ self.W_in.T + self.b_in) @ self.W_out.T + self.b_out

    def normal(self, X):
        return X @ self.A.T + self.b + psi(X @ self.W_in.T + self.b_in) @ self.W_out.T


def projector(M, k=None):
    ev, V = torch.linalg.eigh(0.5 * (M + M.T))
    if k is None:
        k = int((ev > 0).sum())
    Vk = V[:, ev.numel() - k:]
    return Vk @ Vk.T, k


def haar_projector(d, k, gen):
    Qm, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    return Qm[:, :k] @ Qm[:, :k].T


def split_projectors(mlp, mask, k=None):
    a, u = mlp.W_in, mlp.W_out.T
    if k is None:  # rank proportional to the group's share of units, so no side is trivially empty
        k = min(max(round(a.shape[1] * int(mask.sum()) / a.shape[0]), 1), a.shape[1] - 1)
    wa = (u * u).sum(1)
    wu = (a * a).sum(1)
    s = torch.where(mask, 1.0, -1.0).to(torch.float64)
    P, kp = projector(a.T @ ((s * wa)[:, None] * a), k)
    Qp, kq = projector(u.T @ ((s * wu)[:, None] * u), k)
    return P, Qp, kp, kq


def eta_bound(mlp, P, Qp):
    a, u = mlp.W_in, mlp.W_out.T
    d = P.shape[0]
    I = torch.eye(d, dtype=torch.float64)
    Qu, Iu = u @ Qp, u @ (I - Qp)
    Pa, Ia = a @ P, a @ (I - P)
    Dj = torch.maximum(Qu.norm(dim=1) * Ia.norm(dim=1), Iu.norm(dim=1) * Pa.norm(dim=1))
    lin = torch.linalg.matrix_norm(Qp @ mlp.A - mlp.A @ P, ord=2)
    return {"eta": float(lin + KAPPA * Dj.sum()), "lin": float(lin), "nonlin": float(KAPPA * Dj.sum())}


def replacement(mlp, P, Qp, X, Y):
    d = P.shape[0]
    I = torch.eye(d, dtype=torch.float64)
    Z = X @ P.T + Y @ (I - P).T
    Fx, Fy = mlp.direct(X), mlp.direct(Y)
    R = mlp.direct(Z) - Fx @ Qp.T - Fy @ (I - Qp).T
    nv = (X - Y).norm(dim=1)
    return R.norm(dim=1) / nv, R.norm(dim=1) / (Fx - Fy).norm(dim=1)


def components(adj):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n, lab = connected_components(csr_matrix(adj.numpy().astype(np.int8)), directed=False)
    sizes = np.sort(np.bincount(lab))[::-1]
    return int(n), sizes[:5].tolist()


def cos_matrix(V):
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return Vn @ Vn.T


def evaluate_split(mlp, mask, X, Y, scale):
    P, Qp, kp, kq = split_projectors(mlp, mask)
    eb = eta_bound(mlp, P, Qp)
    emp, emp_rel = replacement(mlp, P, Qp, X, Y)
    return {"size": int(mask.sum()), "rank_P": kp, "rank_Q": kq, **eb, "eta_over_scale": eb["eta"] / scale,
            "emp_over_dx": q(emp), "emp_over_dF": q(emp_rel), "bound_holds": bool((emp <= eb["eta"] * (1 + 1e-9)).all())}


def planted_check(gen):
    d, dims, units = 12, (5, 7), (10, 14)
    R, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    S, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    rows_a, rows_u, truth = [], [], []
    off = 0
    for m, (dm, nm) in enumerate(zip(dims, units)):
        a = torch.zeros(nm, d, dtype=torch.float64)
        u = torch.zeros(nm, d, dtype=torch.float64)
        a[:, off:off + dm] = torch.randn(nm, dm, generator=gen, dtype=torch.float64)
        u[:, off:off + dm] = torch.randn(nm, dm, generator=gen, dtype=torch.float64)
        rows_a.append(a @ R.T)
        rows_u.append(u @ S.T)
        truth += [m] * nm
        off += dm
    perm = torch.randperm(sum(units), generator=gen)
    W_in = torch.cat(rows_a)[perm]
    W_out = torch.cat(rows_u)[perm].T.contiguous()
    truth = torch.tensor(truth)[perm]
    mlp = Mlp(W_in, torch.randn(sum(units), generator=gen, dtype=torch.float64), W_out,
              torch.randn(d, generator=gen, dtype=torch.float64))
    tol = 1e-10
    adj = (cos_matrix(W_in).abs() > tol) | (cos_matrix(W_out.T).abs() > tol)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    ncomp, lab = connected_components(csr_matrix(adj.numpy().astype(np.int8)), directed=False)
    lab = torch.tensor(lab)
    mask = lab == lab[0]
    recovered = bool(((mask == (truth == truth[0])).all()) and ncomp == 2)
    Ua = torch.linalg.svd(W_in[mask].T, full_matrices=False)[0]
    Uu = torch.linalg.svd(W_out[:, mask], full_matrices=False)[0]
    ra = int((torch.linalg.svdvals(W_in[mask]) > 1e-10).sum())
    ru = int((torch.linalg.svdvals(W_out[:, mask]) > 1e-10).sum())
    P, Qp = Ua[:, :ra] @ Ua[:, :ra].T, Uu[:, :ru] @ Uu[:, :ru].T
    X = 3 * torch.randn(2000, d, generator=gen, dtype=torch.float64)
    Y = 3 * torch.randn(2000, d, generator=gen, dtype=torch.float64)
    emp, _ = replacement(mlp, P, Qp, X, Y)
    rmask = torch.zeros(sum(units), dtype=torch.bool)
    rmask[torch.randperm(sum(units), generator=gen)[: int(mask.sum())]] = True
    Pr, Qr, _, _ = split_projectors(mlp, rmask)
    emp_r, _ = replacement(mlp, Pr, Qr, X, Y)
    Pw, Qw, _, _ = split_projectors(mlp, mask)
    return {"d": d, "module_dims": dims, "module_units": units, "components_found": int(ncomp),
            "recovered_exact_partition": recovered, "rank_P": ra, "rank_Q": ru,
            "exact_split_eta": eta_bound(mlp, P, Qp)["eta"], "exact_split_emp_max": float(emp.max()),
            "weighted_construction_eta": eta_bound(mlp, Pw, Qw)["eta"],
            "random_split_emp_median": float(emp_r.median()), "random_split_eta": eta_bound(mlp, Pr, Qr)["eta"]}


def independence_check(mlp, sigma, mu, n_pts, gen):
    d = mlp.A.shape[0]
    X = mu + sigma * torch.randn(n_pts, d, generator=gen, dtype=torch.float64)
    G = dpsi(X @ mlp.W_in.T + mlp.b_in)
    G = torch.cat([torch.ones(n_pts, 1, dtype=torch.float64), G], 1)
    G = G / G.norm(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(G)
    return {"n_points": n_pts, "n_funcs": G.shape[1], "sv_min": float(sv[-1]), "sv_max": float(sv[0]),
            "cond": float(sv[0] / sv[-1])}


def refine(mlp, mask, k, iters):
    """Lloyd-style local search on the squared proxy: reassign each unit to the side it fits, rebuild P, Q (rank k)."""
    a, u = mlp.W_in, mlp.W_out.T
    na2, nu2 = (a * a).sum(1), (u * u).sum(1)
    for _ in range(iters):
        P, Qp, _, _ = split_projectors(mlp, mask, k)
        Pa2, Qu2 = ((a @ P) ** 2).sum(1), ((u @ Qp) ** 2).sum(1)
        cost_in = nu2 * (na2 - Pa2) + na2 * (nu2 - Qu2)
        cost_out = nu2 * Pa2 + na2 * Qu2
        new = cost_in < cost_out
        if new.sum() == 0 or new.all() or (new == mask).all():
            break
        mask = new
    P, Qp, _, _ = split_projectors(mlp, mask, k)
    return mask, P, Qp


def random_twin(mlp, gen):
    """Same unit norms, biases and output bias; read and write directions redrawn isotropically."""
    def redraw(V):
        G = torch.randn(V.shape, generator=gen, dtype=torch.float64)
        return G / G.norm(dim=1, keepdim=True) * V.norm(dim=1, keepdim=True)
    return Mlp(redraw(mlp.W_in), mlp.b_in, redraw(mlp.W_out.T).T.contiguous(), mlp.b_out)


def refined_search(mlp, X, Y, scale, starts, k, iters):
    best = None
    for mask in starts:
        m2, P, Qp = refine(mlp, mask, k, iters)
        eb = eta_bound(mlp, P, Qp)
        if best is None or eb["eta"] < best[0]["eta"]:
            best = (eb, m2, P, Qp)
    eb, m2, P, Qp = best
    emp, emp_rel = replacement(mlp, P, Qp, X, Y)
    return {"k": k, "size": int(m2.sum()), "eta_over_scale": eb["eta"] / scale, "lin_over_scale": eb["lin"] / scale,
            "emp_over_dx_median": float(emp.median()), "emp_over_dF_median": float(emp_rel.median()),
            "bound_holds": bool((emp <= eb["eta"] * (1 + 1e-9)).all())}


def spectral_labels(W, K, seed):
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed, assign_labels="cluster_qr")
    return torch.tensor(sc.fit_predict(W.numpy()))


def write_receipt(report, out, t0, t_fwd):
    report["runtime_s"] = {"forward_and_load": t_fwd, "total": time.time() - t0}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # one top-level key per line: build.rs caps tracked files at 10k lines
    Path(out).write_text("{\n" + ",\n".join("%s:%s" % (json.dumps(k), json.dumps(v, separators=(",", ":")))
                                             for k, v in report.items()) + "\n}\n")


def pi_estar(U, mask):
    """E* and chi of pi_split without the projector, from the smaller of the |S| x |S| and d x d Grams
    (the same nonzero eigenvalues; zero eigenvalues contribute nothing)."""
    Us = U[mask]
    G = Us @ Us.T if Us.shape[0] < Us.shape[1] else Us.T @ Us
    lam = torch.linalg.eigvalsh(0.5 * (G + G.T)).clamp(0, 1)
    return {"E_star": float(torch.minimum(lam, 1 - lam).sum()), "chi": float((lam * (1 - lam)).sum()),
            "rank_P": int((lam > 0.5).sum())}


def pi_split(U, T, mask):
    """Optimal exactly-separated reads for unit subset S (whitened): E* = sum min(lam, 1-lam), lam = eig(U^T D_S U)."""
    G = U[mask].T @ U[mask]
    lam, V = torch.linalg.eigh(0.5 * (G + G.T))
    lam = lam.clamp(0, 1)
    Vp = V[:, lam > 0.5]
    P = Vp @ Vp.T
    Uhat = torch.where(mask[:, None], U @ P, U - U @ P)
    return {"E_star": float(torch.minimum(lam, 1 - lam).sum()), "chi": float((lam * (1 - lam)).sum()),
            "rank_P": int(Vp.shape[1])}, Uhat @ T


def pi_fiedler(U, n_vec):
    """Smallest eigenpairs of the Pi^2 graph Laplacian L = diag(rowsum Pi^2) - Pi^2: Lanczos on the dense L
    (n^2 per matvec; matches a dense eigh to 1e-16 on OLMo's n = 8192 in seconds instead of an O(n^3) solve)."""
    from scipy.sparse.linalg import eigsh
    L = (U @ U.T).pow_(2)
    L.neg_()
    L.diagonal().sub_(L.sum(1))
    Ln = L.numpy()
    ev, vec = eigsh(Ln, k=n_vec + 1, which="SA", tol=1e-12, ncv=max(40, 4 * (n_vec + 1)), maxiter=100000,
                    v0=np.ones(Ln.shape[0]))
    order = np.argsort(ev)
    return ev[order], vec[:, order]


def pi_proposals(U, n_vec, fracs):
    ev, vec = pi_fiedler(U, n_vec)
    n = U.shape[0]
    d = U.shape[1]
    out = []
    for v in range(1, n_vec + 1):
        order = torch.tensor(np.argsort(vec[:, v]))
        for f in fracs:
            k = max(1, round(f * n))
            for side, idx in (("low", order[:k]), ("high", order[-k:])):
                mask = torch.zeros(n, dtype=torch.bool)
                mask[idx] = True
                r = pi_estar(U, mask)
                out.append({"vec": v, "k": k, "side": side, "E_over_U2": r["E_star"] / d, "chi_over_U2": r["chi"] / d,
                            "rank_P": r["rank_P"], "mask": mask})
    return ev, out


def pi_reads(W, n_rand, gen, fracs):
    """Weight-only Pi statistics of a read matrix W (n units x d): thresholded components, Fiedler proposals
    vs random subsets vs the same proposer on a twin with redrawn unit directions (norms kept)."""
    n, d = W.shape
    U, T = torch.linalg.qr(W)
    Pi = U @ U.T
    off = ~torch.eye(n, dtype=torch.bool)
    band = {}
    for tau in (0.0, 1e-13, 1e-10, 1e-6, 1e-3, 1e-2, 3e-2):
        nc, sizes = components((Pi.abs() > tau) & off)
        band[str(tau)] = {"components": nc, "largest": sizes}
    offdiag = q(Pi.abs()[off])
    del Pi, off
    ev, props = pi_proposals(U, 3, fracs)
    rand = {}
    for f in fracs:
        k = max(1, round(f * n))
        vals = []
        for _ in range(n_rand):
            mask = torch.zeros(n, dtype=torch.bool)
            mask[torch.randperm(n, generator=gen)[:k]] = True
            vals.append(pi_estar(U, mask)["E_star"] / d)
        rand[k] = (float(np.mean(vals)), float(np.std(vals)))
    G = torch.randn(W.shape, generator=gen, dtype=torch.float64)
    Wt = G / G.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True)
    Ut, _ = torch.linalg.qr(Wt)
    _, props_t = pi_proposals(Ut, 3, fracs)
    best_t = {}
    for pr in props_t:
        if pr["k"] not in best_t or pr["E_over_U2"] < best_t[pr["k"]]:
            best_t[pr["k"]] = pr["E_over_U2"]
    by_k = []
    for k in sorted(rand):
        cand = [pr for pr in props if pr["k"] == k]
        bp = min(cand, key=lambda pr: pr["E_over_U2"])
        mu, sd = rand[k]
        by_k.append({"k": k, "best_fiedler_E_over_U2": bp["E_over_U2"], "best_fiedler_chi_over_U2": bp["chi_over_U2"],
                     "vec": bp["vec"], "side": bp["side"], "random_mean": mu, "random_sd": sd,
                     "z_vs_random": (bp["E_over_U2"] - mu) / sd if sd > 0 else None,
                     "ratio_vs_random": bp["E_over_U2"] / mu, "twin_best_fiedler": best_t[k],
                     "ratio_vs_twin": bp["E_over_U2"] / best_t[k]})
    res = {"n": n, "d": d, "U_F2": float(U.pow(2).sum()), "pi_components_band": band, "pi_offdiag_abs": offdiag,
           "laplacian_eigs": ev.tolist(), "by_k": by_k}
    for r in by_k:
        print("   k=%d fiedler %.4f random %.4f+-%.4f twin %.4f" % (r["k"], r["best_fiedler_E_over_U2"], r["random_mean"],
                                                                  r["random_sd"], r["twin_best_fiedler"]), flush=True)
    return res, U, T, props, rand


def pi_layer(layer, li, Xin, n_rand, gen, fracs):
    tl = time.time()
    m = layer.mlp
    W, b_in, Vout = m.dense_h_to_4h.weight, m.dense_h_to_4h.bias, m.dense_4h_to_h.weight
    d = W.shape[1]
    res, U, T, props, rand = pi_reads(W, n_rand, gen, fracs)
    best = min(props, key=lambda pr: pr["E_over_U2"] / rand[pr["k"]][0])
    _, What = pi_split(U, T, best["mask"])
    ln = layer.post_attention_layernorm
    R_ln = float(ln.bias.norm() + math.sqrt(d) * ln.weight.abs().max())
    R_emp = float(Xin.norm(dim=1).max())
    specV = float(torch.linalg.matrix_norm(Vout, ord=2))
    specdW = float(torch.linalg.matrix_norm(W - What, ord=2))
    L_sigma = 0.5 + KAPPA
    F = torch.nn.functional.gelu(Xin @ W.T + b_in) @ Vout.T
    Fh = torch.nn.functional.gelu(Xin @ What.T + b_in) @ Vout.T
    err = (F - Fh).norm(dim=1)
    res = {"layer": li, **res,
           "best": {"k": best["k"], "vec": best["vec"], "side": best["side"], "E_over_U2": best["E_over_U2"],
                    "random_mean": rand[best["k"]][0], "ratio_vs_random": best["E_over_U2"] / rand[best["k"]][0],
                    "W_minus_What_2": specdW, "W_2": float(torch.linalg.matrix_norm(W, ord=2)), "V_2": specV,
                    "R_layernorm": R_ln, "R_empirical_max": R_emp,
                    "uniform_bound_R_ln": L_sigma * specV * specdW * R_ln,
                    "uniform_bound_R_emp": L_sigma * specV * specdW * R_emp,
                    "measured_err": q(err), "measured_err_over_F": q(err / F.norm(dim=1)),
                    "bound_holds": bool((err <= L_sigma * specV * specdW * R_ln * (1 + 1e-9)).all())},
           "seconds": time.time() - tl}
    b = res["best"]
    print("L%d comps(1e-13)=%d lam2=%.2e best k=%d E/U2=%.4f rand=%.4f ratio=%.3f | bound %.2f meas max %.2f "
          "(rel med %.2f) (%.0fs)" % (li, res["pi_components_band"]["1e-13"]["components"], res["laplacian_eigs"][1],
                                      b["k"], b["E_over_U2"], b["random_mean"], b["ratio_vs_random"],
                                      b["uniform_bound_R_ln"], b["measured_err"]["max"],
                                      b["measured_err_over_F"]["median"], res["seconds"]), flush=True)
    return res


def pi_swiglu(args, gen, fracs, t0):
    """--mode pi on a SwiGLU decoder: weight-only, labelled non-exact (see the module docstring)."""
    from mpd_opfirst_decoder_2951 import Decoder

    D = Decoder(args.model, args.revision)
    pick = [int(v) for v in args.layers.split(",")] if args.layers else range(D.L)
    report = {"model": args.model, "revision": args.revision, "architecture": D.describe(),
              "hidden_act": D.config["hidden_act"], "read": args.read, "d_model": D.d, "d_hidden": D.n_ff,
              "exact_module_theorem_applies": False,
              "label": "SwiGLU: E* is the read-row geometry of W_%s only; a unit's other read and its bilinear "
                       "gate are ignored, so this is not an exact module split (the exact-GELU theorem does not "
                       "cover it)" % args.read,
              "dtype": "float64", "device": "cpu", "args": vars(args), "layers": []}
    for li in pick:
        tl = time.time()
        W = torch.from_numpy(D(f"model.layers.{li}.mlp.{args.read}_proj.weight"))
        res = pi_reads(W, args.random_subsets, gen, fracs)[0]
        report["layers"].append({"layer": li, **res, "seconds": time.time() - tl})
        print("L%d comps(3e-2)=%d lam2=%.2e (%.0fs)" % (li, res["pi_components_band"]["0.03"]["components"],
                                                       res["laplacian_eigs"][1], time.time() - tl), flush=True)
    write_receipt(report, args.out, t0, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--read", choices=("up", "gate"), default="up", help="SwiGLU --mode pi: read matrix")
    parser.add_argument("--fracs", default="1/64,1/32,1/16,1/8,1/4,3/8,1/2", help="--mode pi subset sizes / n")
    parser.add_argument("--n-seq", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--null-draws", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--indep-points", type=int, default=0, help="0 = 3 x d_hidden")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("replacement", "pi"), default="replacement")
    parser.add_argument("--layers", default="", help="comma list for --mode pi; empty = all")
    parser.add_argument("--random-subsets", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    t0 = time.time()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    fracs = [float(a) / float(b) for a, b in (f.split("/") for f in args.fracs.split(","))]
    cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if cfg.hidden_act != "gelu":
        if args.mode != "pi":
            raise SystemExit("hidden_act=%r is not exact erf GELU; only the labelled --mode pi runs" % cfg.hidden_act)
        return pi_swiglu(args, gen, fracs, t0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float64).eval()
    ids, source = texts_tokens(tok, args.n_seq, args.seq_len, args.seed)
    layers = model.gpt_neox.layers
    caps = {}

    def hook(i):
        def f(mod, inp, out):
            caps[i] = (inp[0].detach().reshape(-1, inp[0].shape[-1]), out.detach().reshape(-1, out.shape[-1]))
        return f

    hs = [layers[i].mlp.register_forward_hook(hook(i)) for i in range(len(layers))]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    t_fwd = time.time() - t0
    if args.mode == "pi":
        pick = [int(v) for v in args.layers.split(",")] if args.layers else range(len(layers))
        report = {"model": args.model, "hidden_act": cfg.hidden_act, "d_model": cfg.hidden_size,
                  "d_hidden": cfg.intermediate_size, "tokens": int(ids.numel()), "text_source": source,
                  "L_sigma": 0.5 + KAPPA, "dtype": "float64", "device": "cpu", "args": vars(args), "layers": []}
        with torch.no_grad():
            for li in pick:
                report["layers"].append(pi_layer(layers[li], li, caps[li][0], args.random_subsets, gen, fracs))
        return write_receipt(report, args.out, t0, t_fwd)

    report = {"model": args.model, "hidden_act": cfg.hidden_act, "d_model": cfg.hidden_size,
              "d_hidden": cfg.intermediate_size, "tokens": int(ids.numel()), "text_source": source,
              "kappa": KAPPA, "dtype": "float64", "device": "cpu", "args": vars(args),
              "planted": planted_check(gen), "layers": []}
    d, n = cfg.hidden_size, cfg.intermediate_size
    gauss_cos_med = 0.6745 / math.sqrt(d)  # median |N(0, 1/d)|
    report["gaussian_null_abs_cos_median"] = gauss_cos_med
    with torch.no_grad():
        for li, layer in enumerate(layers):
            tl = time.time()
            m = layer.mlp
            act_ok = float((m.act(torch.linspace(-6, 6, 1001, dtype=torch.float64))
                            - torch.linspace(-6, 6, 1001, dtype=torch.float64)
                            * Phi(torch.linspace(-6, 6, 1001, dtype=torch.float64))).abs().max())
            mlp = Mlp(m.dense_h_to_4h.weight, m.dense_h_to_4h.bias, m.dense_4h_to_h.weight, m.dense_4h_to_h.bias)
            Xin, Fout = caps[li]
            mu, sigma = Xin.mean(0), Xin.std()
            Xr = mu + sigma * torch.randn(512, d, generator=gen, dtype=torch.float64)
            nf_real = float((mlp.normal(Xin) - Fout).norm() / Fout.norm())
            Fr = m(Xr)
            nf_rand = float((mlp.normal(Xr) - Fr).norm() / Fr.norm())
            a, u = mlp.W_in, mlp.W_out.T
            na, nu = a.norm(dim=1), u.norm(dim=1)
            Ca, Cu = cos_matrix(a), cos_matrix(u)
            off = ~torch.eye(n, dtype=torch.bool)
            Ga, Gu = a @ a.T, u @ u.T
            dup = int(((Ca.abs() > 1 - 1e-12) & off).sum() // 2)
            exact_edges = ((Ga != 0) | (Gu != 0)) & off
            comp_exact = components(exact_edges)
            thr = {}
            for tau in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
                adj = ((Ca.abs() > tau) | (Cu.abs() > tau)) & off
                nc, sizes = components(adj)
                thr[str(tau)] = {"components": nc, "largest": sizes,
                                 "edge_density": float(adj.sum() / (n * (n - 1)))}
            scale = float(torch.linalg.matrix_norm(mlp.A, ord=2) + KAPPA * (na * nu).sum())
            npairs = min(args.pairs, Xin.shape[0] // 2)
            perm = torch.randperm(Xin.shape[0], generator=gen)
            X, Y = Xin[perm[:npairs]], Xin[perm[npairs:2 * npairs]]
            distinct = (X - Y).norm(dim=1) > 1e-9 * X.norm(dim=1)  # layer 0 inputs repeat with the token id
            X, Y = X[distinct], Y[distinct]
            lip = ((mlp.direct(X) - mlp.direct(Y)).norm(dim=1) / (X - Y).norm(dim=1))
            W = (Ca * Ca + Cu * Cu) * off
            splits, spec_masks = [], []
            for K in range(2, args.kmax + 1):
                lab = spectral_labels(W, K, args.seed)
                groups = []
                for g in range(K):
                    mask = lab == g
                    if mask.sum() == 0 or mask.all():
                        continue
                    ev = evaluate_split(mlp, mask, X, Y, scale)
                    nulls = []
                    for _ in range(args.null_draws):
                        rm = torch.zeros(n, dtype=torch.bool)
                        rm[torch.randperm(n, generator=gen)[: int(mask.sum())]] = True
                        nulls.append(evaluate_split(mlp, rm, X, Y, scale))
                    Ph, Qh = haar_projector(d, ev["rank_P"], gen), haar_projector(d, ev["rank_Q"], gen)
                    eh = eta_bound(mlp, Ph, Qh)
                    emp_h, _ = replacement(mlp, Ph, Qh, X, Y)
                    ev["null_random_split"] = {
                        "eta_over_scale_mean": sum(z["eta_over_scale"] for z in nulls) / len(nulls),
                        "emp_over_dx_median_mean": sum(z["emp_over_dx"]["median"] for z in nulls) / len(nulls),
                        "emp_over_dF_median_mean": sum(z["emp_over_dF"]["median"] for z in nulls) / len(nulls)}
                    ev["null_haar_PQ"] = {"eta_over_scale": eh["eta"] / scale, "emp_over_dx_median": float(emp_h.median())}
                    ev["eta_ratio_vs_random_split"] = ev["eta_over_scale"] / ev["null_random_split"]["eta_over_scale_mean"]
                    ev["emp_ratio_vs_random_split"] = (ev["emp_over_dx"]["median"]
                                                       / ev["null_random_split"]["emp_over_dx_median_mean"])
                    groups.append(ev)
                    spec_masks.append(mask)
                splits.append({"K": K, "sizes": sorted(torch.bincount(lab, minlength=K).tolist(), reverse=True),
                               "groups": groups})
            allg = [g for s in splits for g in s["groups"]]
            balanced = [g for g in allg if min(g["size"], n - g["size"]) >= n // 20] or allg
            best = min(balanced, key=lambda g: g["eta_ratio_vs_random_split"])
            twin = random_twin(mlp, gen)
            scale_twin = float(torch.linalg.matrix_norm(twin.A, ord=2) + KAPPA * (na * nu).sum())
            refined = {}
            for kname, k in (("d/4", d // 4), ("d/2", d // 2), ("best_spectral", best["rank_P"])):
                size = round(n * k / d)
                starts = []
                for _ in range(args.restarts):
                    rm = torch.zeros(n, dtype=torch.bool)
                    rm[torch.randperm(n, generator=gen)[:size]] = True
                    starts.append(rm)
                spec = [mk for mk in spec_masks if 0.5 * size <= int(mk.sum()) <= 2 * size]
                trained = refined_search(mlp, X, Y, scale, starts + spec, k, args.iters)
                rnd = refined_search(twin, X, Y, scale_twin, starts, k, args.iters)
                refined[kname] = {"trained": trained, "random_twin": rnd,
                                  "eta_ratio_trained_vs_twin": trained["eta_over_scale"] / rnd["eta_over_scale"]}
            L = {
                "layer": li, "act_vs_erf_gelu_maxabs": act_ok,
                "normal_form_rel_real": nf_real, "normal_form_rel_random": nf_rand,
                "duplicate_units": dup,
                "exact_zero_graph": {"edges": int(exact_edges.sum() // 2), "possible": n * (n - 1) // 2,
                                     "components": comp_exact[0],
                                     "zero_read_inner": int(((Ga == 0) & off).sum() // 2),
                                     "zero_write_inner": int(((Gu == 0) & off).sum() // 2)},
                "abs_cos_read": q(Ca.abs()[off]), "abs_cos_write": q(Cu.abs()[off]),
                "threshold_components": thr, "scale_A_plus_kappa_sumC": scale,
                "A_norm": float(torch.linalg.matrix_norm(mlp.A, ord=2)), "kappa_sumC": float(KAPPA * (na * nu).sum()),
                "empirical_lipschitz_dF_over_dx": q(lip), "pairs_used": int(X.shape[0]),
                "independence": independence_check(mlp, sigma, mu, args.indep_points or 3 * n, gen),
                "splits": splits,
                "best_balanced_group": {k: best[k] for k in ("size", "rank_P", "rank_Q", "eta_over_scale",
                                                             "eta_ratio_vs_random_split", "emp_ratio_vs_random_split")}
                | {"emp_over_dF_median": best["emp_over_dF"]["median"]},
                "all_bounds_hold": all(g["bound_holds"] for g in allg),
                "refined": refined,
                "seconds": time.time() - tl,
            }
            report["layers"].append(L)
            print("L%d nf=%.1e/%.1e comp0=%d thr0.1=%d best: size=%d eta/scale=%.3f vsRand=%.3f empVsRand=%.3f "
                  "empdF=%.3f cond=%.1e (%.0fs)" % (
                      li, nf_real, nf_rand, comp_exact[0], thr["0.1"]["components"], best["size"],
                      best["eta_over_scale"], best["eta_ratio_vs_random_split"], best["emp_ratio_vs_random_split"],
                      best["emp_over_dF"]["median"], L["independence"]["cond"], L["seconds"]), flush=True)
            for kname, r in refined.items():
                print("   refined k=%s: trained eta/scale=%.3f empdF=%.3f | twin eta/scale=%.3f empdF=%.3f" % (
                    kname, r["trained"]["eta_over_scale"], r["trained"]["emp_over_dF_median"],
                    r["random_twin"]["eta_over_scale"], r["random_twin"]["emp_over_dF_median"]), flush=True)
    write_receipt(report, args.out, t0, t_fwd)
    print(json.dumps(report["planted"], indent=1))
    print("runtime %.0fs" % report["runtime_s"]["total"])


if __name__ == "__main__":
    main()
