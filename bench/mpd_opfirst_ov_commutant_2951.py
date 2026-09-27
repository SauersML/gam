"""#2951 operator-first: is one value law reused across attention heads? The OV compatible-edit algebra.

Analysis under SPEC 8's exception (benchmark evaluation, not an MPD input). No Rust owner computes a joint
intertwiner algebra, so the kernels below are a labelled numpy stand-in. CPU float64 on the exact weights,
one layer at a time, matrix-free where the unknowns are d^2.

Theory. A block linear in a payload, M(c) = sum_nu phi_nu(c) C_nu with linearly independent controls phi_nu,
admits the edit pair (A on the payload, B on the output) with B M(c) = M(c) A for all c iff B C_nu = C_nu A
for every nu. Those pairs form an algebra (the endomorphism algebra of the representation {C_nu} of the
Kronecker quiver). If it is sum_j M_{m_j}(R), the block is sum_j H_j(c) (x) I_{m_j}: one law H_j applied to an
m_j-dimensional payload. For attention the payload is the values, the controls are the per-head attention
patterns, and C_h = W_O[:, h] W_V[g(h)] (input RMSNorm gain folded, the per-token RMS excluded).

Exact reduction (derived, then checked). Write C_h = L_h R_g with R_g of full row rank hd and L_h of full
column rank hd. Then B C_h = C_h A iff R_g A = a_g R_g and B L_h = L_h a_g for block matrices a_g. Stack
W_V = [R_g]_g (KV hd x d) and W_O = [L_h]_h (d x H hd). If W_V has full row rank every a is reachable on the
input side; on the output side a is reachable iff the sibling-replicated diag(a) preserves ker W_O. So
  * H hd <= d and W_O injective (SmolLM2-135M: 9 x 64 = 576 = d): the algebra is exactly sum_g M_hd(R),
    dimension KV hd^2 on the core, FORCED by the architecture (each group's value payload is reused by its
    siblings through their own O; nothing learned is needed).
  * H hd > d (Qwen3-0.6B: 16 x 128 = 2d): ker W_O has dimension >= H hd - d and generically only the
    scalars survive. Whether the trained weights keep a larger algebra, exactly or approximately, is the
    measurement.
Outside the core (rowspace of W_V on the input, column space of W_O on the output) edits are free and
carry no law; they are counted, not analysed.

Measurement. With M = sum_h C_h^T C_h and G = sum_h C_h C_h^T on the core, whiten A-hat = M^(1/2) A,
B-hat = B G^(1/2). Then ||K(A,B)||^2 = ||A-hat||^2 + ||B-hat||^2 - 2 <A-hat, X B-hat>, K(A,B)_h = B C_h - C_h A,
  X B-hat = M^(-1/2) sum_h C_h^T B-hat G^(-1/2) C_h,   X^T A-hat = sum_h C_h M^(-1/2) A-hat C_h^T G^(-1/2),
||X|| <= 1, and the best B for a given A leaves the relative defect
  eps(A) = min_B ||K(A,B)|| / (sum_h ||C_h A||^2)^(1/2) = (1 - sigma^2)^(1/2).
The top of X X^T (Lanczos, matrix-free, low-rank factors) gives the A-directions with the smallest defect.
The exactly derived forced subspace (sum_g M_hd, or the scalars) is deflated by its M-orthogonal projector,
so what is reported beyond it is the approximate compatible-edit spectrum. Each Ritz direction's eps is
recomputed directly from B C_h - C_h A (no cancellation). The whitening is the conditioning choice; the exact
algebra does not depend on it (any invertible reference gives the same kernel), so only dimensions and gaps
are read, never coordinates.

Bands. Exact: eps <= tau = n u sqrt(kappa(M) kappa(G)) (first-order forward error of the direct residual with
the whitening's amplification; n = max core dim, u = f64 unit roundoff); the forced elements' measured
residual is reported next to it. Approximate: a k-dimensional block (eps_1..eps_k <= delta) is gap-certified
when 2 delta / gamma < 1 with gamma = eps_{k+1} - eps_k (the sin-theta subspace bound: the k-space is then a
stable perturbation of an exact k-dimensional kernel of a family within delta), and is "beyond the null" when
eps_k is below the smallest eps of both nulls: a twin (every head's write side and every group's read side
rotated by independent random orthogonal maps: each C_h's singular values and the GQA tying kept, cross-head
relations destroyed), a group twin (one write rotation per GQA group: sibling geometry kept, only
cross-group relations destroyed) and a Gaussian null (iid factors at matched Frobenius norms, tying kept).

Anchor. Row-stochastic routing fixes source-constant payloads, so for group g the payload space W_g = R^hd is
a common scalar-action anchor: a candidate block P is a genuine M_m(R) law iff restriction of the corner
P E P to End(W) is bijective (dim P E P = m^2, dim W = m). For the forced blocks the restriction
A -> R_g A R_g^+ is reported with its smallest singular value (bijective iff > 0). For approximate candidates the
closure of their span under products and the rank of the restriction to each W_g are reported.

Per GQA group (siblings share V, differ in O): the group family {C_h : h in g} is exactly M_hd on its core
iff the siblings' write images are independent; certified by the smallest singular value of the stacked
per-head orthonormal image bases (principal angles), against the twin. With group-shared controls (siblings
attending identically) the family is {sum_{h in g} C_h}_g, forced sum_g M_hd iff those images are independent.

Routing-invariant payload (exact, weight-only): the common null space of every head's query-side and key-side
read rows (the QK operators' row and column spaces), and of each rotary plane's pieces; compared with the
architectural minimum max(0, d - rows), which is what it is forced to be.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import Decoder, compact_json  # noqa: E402

U_RND = np.finfo(np.float64).eps / 2


# Needs NumPy / SciPy on OpenBLAS: the macOS wheels' Apple Accelerate LAPACK corrupts the heap on these
# rank-deficient, wide-spectrum matrices (segfaults here). SVDs use the gesvd driver.
def svd(Z, compute_uv=True):
    return sla.svd(Z, full_matrices=False, compute_uv=compute_uv, lapack_driver="gesvd", check_finite=False)


def qr(Z):
    return sla.qr(Z, mode="economic", check_finite=False)[0]


def pinv(Z):
    u, s, vt = svd(Z)
    keep = s > max(Z.shape) * np.finfo(np.float64).eps * s[0]
    return (vt[keep].T / s[keep]) @ u[:, keep].T


def orth_basis(Z, side):
    """Orthonormal basis of the column (side='col') or row (side='row') space of Z with the eps rank band."""
    u, s, vt = svd(Z)
    tau = max(Z.shape) * np.finfo(np.float64).eps * s[0]
    r = int((s > tau).sum())
    basis = u[:, :r] if side == "col" else vt[:r].T
    return basis, {"rank": r, "max_rank": min(Z.shape), "tau": float(tau),
                   "sigma_min_over_max": float(s[r - 1] / s[0]), "certified_full": bool(r == min(Z.shape))}


def psd_pow(S, p):
    w, v = sla.eigh(S)
    return (v * w**p) @ v.T, float(w[-1] / w[0])


class Family:
    """Core-coordinate OV family: C_h = l_h r_g(h), l_h (r_out x hd), r_g (hd x r_in)."""

    def __init__(self, Ls, Rs, grp):
        self.H, self.KV, self.grp = len(Ls), len(Rs), grp
        self.hd = Rs[0].shape[0]
        Lcat, Rcat = np.hstack(Ls), np.vstack(Rs)
        self.Uout, self.out_rank = orth_basis(Lcat, "col")
        self.Uin, self.in_rank = orth_basis(Rcat, "row")
        self.l = [self.Uout.T @ L for L in Ls]
        self.r = [R @ self.Uin for R in Rs]
        self.Lc, self.Rc = np.hstack(self.l), np.vstack(self.r)
        self.ri, self.ro = self.Uin.shape[1], self.Uout.shape[1]
        M = sum(self.r[h // grp].T @ (self.l[h].T @ self.l[h]) @ self.r[h // grp] for h in range(self.H))
        G = sum(self.l[h] @ (self.r[h // grp] @ self.r[h // grp].T) @ self.l[h].T for h in range(self.H))
        self.Mh, self.kM = psd_pow(M, 0.5)
        self.Mi, _ = psd_pow(M, -0.5)
        self.Gh, self.kG = psd_pow(G, 0.5)
        self.Gi, _ = psd_pow(G, -0.5)
        self.band = max(self.ri, self.ro) * U_RND * np.sqrt(self.kM * self.kG)

    def C(self, h):
        return self.l[h] @ self.r[h // self.grp]

    def XT(self, Ah):  # A-space (ri x ri) -> B-space (ro x ro)
        Ac = self.Mi @ Ah
        RA = self.Rc @ Ac
        hd, S = self.hd, np.zeros((self.ro, self.ro))
        T = [RA[g * hd:(g + 1) * hd] @ self.r[g].T for g in range(self.KV)]
        LT = np.hstack([self.l[h] @ T[h // self.grp] for h in range(self.H)])
        S = LT @ self.Lc.T
        return S @ self.Gi

    def X(self, Bh):  # B-space -> A-space
        Bt = Bh @ self.Gi
        LB = self.Lc.T @ Bt
        hd = self.hd
        V = [np.zeros((hd, hd)) for _ in range(self.KV)]
        for h in range(self.H):
            V[h // self.grp] += LB[h * hd:(h + 1) * hd] @ self.l[h]
        RV = np.vstack([V[g] @ self.r[g] for g in range(self.KV)])
        return self.Mi @ (self.Rc.T @ RV)

    def defect(self, A):
        """Direct eps(A): best B by the whitened normal equation, residual recomputed from B C_h - C_h A."""
        Ah = self.Mh @ A
        B = self.XT(Ah) @ self.Gi
        num = sum(np.linalg.norm(B @ self.C(h) - self.C(h) @ A) ** 2 for h in range(self.H))
        den = sum(np.linalg.norm(self.C(h) @ A) ** 2 for h in range(self.H))
        return float(np.sqrt(num / den))

    def forced(self):
        """The exactly derived forced subspace in whitened A-space, as (map y -> A-hat, adjoint, dimension,
        description). The adjoint is only an orthogonal projector's factor for the scalars (unit vector)."""
        g_forced = (self.out_rank["rank"] == self.H * self.hd and self.in_rank["rank"] == self.KV * self.hd)
        if not g_forced:
            v = self.Mh.copy()
            nv = np.linalg.norm(v)
            return (lambda y: y[0] * v / nv, lambda Z: np.array([np.sum(v * Z) / nv]), 1,
                    "scalars (H hd > rank W_O or W_V rank-deficient: no forced blocks)")
        W = self.Rc  # (KV hd) x ri, square invertible
        Winv = sla.inv(W)
        hd, KV = self.hd, self.KV
        MW = self.Mh @ Winv

        def fwd(y):
            Y = np.zeros((KV * hd, KV * hd))
            for g in range(KV):
                Y[g * hd:(g + 1) * hd, g * hd:(g + 1) * hd] = y[g * hd * hd:(g + 1) * hd * hd].reshape(hd, hd)
            return MW @ Y @ W

        def adj(Z):
            T = MW.T @ Z @ W.T
            return np.concatenate([T[g * hd:(g + 1) * hd, g * hd:(g + 1) * hd].ravel() for g in range(KV)])

        return fwd, adj, KV * hd * hd, "sum_g M_hd(R) (W_O injective, W_V full row rank)"


def spectrum(F, k, rng, tol):
    fwd, adj, nf, desc = F.forced()
    n = F.ri * F.ri

    def proj(Z):
        return Z - fwd(adj(Z))

    def mv(x):
        Z = proj(x.reshape(F.ri, F.ri))
        return proj(F.X(F.XT(Z))).ravel()

    # forced-element residuals (exactness check) and the identity sanity check
    y = rng.standard_normal(nf)
    Af = F.Mi @ fwd(y)
    forced_eps = F.defect(Af)
    ident_eps = F.defect(np.eye(F.ri))
    # adjointness of X
    Ar, Br = rng.standard_normal((F.ri, F.ri)), rng.standard_normal((F.ro, F.ro))
    adj_err = abs(np.sum(Ar * F.X(Br)) - np.sum(F.XT(Ar) * Br)) / (np.linalg.norm(Ar) * np.linalg.norm(Br))
    base = {"forced": desc, "forced_dim": nf, "forced_eps": forced_eps, "identity_eps": ident_eps,
            "X_adjoint_err": float(adj_err), "band_tau": F.band, "kappa_M": F.kM, "kappa_G": F.kG,
            "rank_X_max": F.KV * F.hd * F.hd}
    if nf == F.KV * F.hd * F.hd:
        # range(X) = {M^(-1/2) W_V^T blockdiag(v_g) W_V} has dimension <= KV hd^2 = forced dim, and the forced
        # directions have sigma = 1: every other singular value of X is exactly 0 (eps = 1). Probe it.
        # certificate: X X^T idempotent (spectrum in {0, 1}); its rank is <= KV hd^2 structurally and >= the
        # forced dimension because every forced direction has sigma = 1 (forced_eps).
        probe = 0.0
        for _ in range(8):
            Z = rng.standard_normal((F.ri, F.ri))
            P1 = F.X(F.XT(Z))
            probe = max(probe, float(np.linalg.norm(F.X(F.XT(P1)) - P1) / np.linalg.norm(Z)))
        return base | {"eps_beyond_forced_exact": 1.0, "XXT_idempotence_err_max": probe,
                       "beyond_forced": "none possible: rank X = forced dim, all other sigma = 0 exactly"}, []
    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    t = time.time()
    v0 = proj(rng.standard_normal((F.ri, F.ri))).ravel()
    w, V = eigsh(op, k=k, which="LA", tol=tol, v0=v0, ncv=max(3 * k, 40), maxiter=20000)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    eps_lanczos = np.sqrt(np.clip(1 - w, 0, None))
    As = [F.Mi @ V[:, i].reshape(F.ri, F.ri) for i in range(k)]
    eps_direct = [F.defect(A) for A in As]
    return base | {"eps_beyond_forced_lanczos": eps_lanczos.tolist(), "eps_beyond_forced_direct": eps_direct,
                   "lanczos_seconds": time.time() - t, "lanczos_tol": tol}, As


def gap_cert(eps, null_min):
    """Best k (1..len-1): 2 eps_k / (eps_{k+1} - eps_k), and whether eps_k is below both nulls' minimum."""
    best = None
    for k in range(1, len(eps)):
        gam = eps[k] - eps[k - 1]
        ratio = 2 * eps[k - 1] / gam if gam > 0 else np.inf
        cand = {"k": k, "delta": eps[k - 1], "gamma": gam, "ratio_2delta_over_gamma": ratio,
                "certified": bool(ratio < 1), "beyond_null": bool(eps[k - 1] < null_min)}
        if best is None or ratio < best["ratio_2delta_over_gamma"]:
            best = cand
    return best


def characterise(F, As, keep):
    """Where the candidate edits live: per-group restriction to the payload anchor W_g (a_g = r_g A r_g^+),
    within-group vs cross-group energy of W A W^+ (canonical input coords), and closure of span{I, A_i}."""
    A = As[:keep]
    Rp = [pinv(r) for r in F.r]
    out = []
    for Ai in A:
        blocks = F.Rc @ Ai @ pinv(F.Rc)
        hd = F.hd
        diag_e = sum(np.linalg.norm(blocks[g * hd:(g + 1) * hd, g * hd:(g + 1) * hd]) ** 2 for g in range(F.KV))
        a = [F.r[g] @ Ai @ Rp[g] for g in range(F.KV)]
        per_g = [float(np.linalg.norm(ag)) for ag in a]
        tot = sum(p**2 for p in per_g)
        out.append({"within_group_energy": float(diag_e / np.linalg.norm(blocks) ** 2),
                    "anchor_share_per_group": [p**2 / tot for p in per_g],
                    "group_scalar_share": float(sum(np.trace(ag) ** 2 / hd for ag in a) / tot)})
    # closure: products A_i A_j projected on span{I, A_1..A_keep} in the M-whitened metric
    basis = [F.Mh @ np.eye(F.ri)] + [F.Mh @ Ai for Ai in A]
    Qb = qr(np.stack([b.ravel() for b in basis], 1))
    res = []
    for i in range(len(A)):
        for j in range(len(A)):
            p = (F.Mh @ A[i] @ A[j]).ravel()
            res.append(float(np.linalg.norm(p - Qb @ (Qb.T @ p)) / np.linalg.norm(p)))
    # anchor restriction rank of the span, per group (m^2 structure would need rank m^2 on one W_g)
    ranks = []
    for g in range(F.KV):
        Z = np.stack([(F.r[g] @ Ai @ Rp[g]).ravel() for Ai in A], 1)
        s = svd(Z, compute_uv=False)
        ranks.append(int((s > 1e-3 * s[0]).sum()) if s[0] > 0 else 0)
    return {"directions": out, "closure_residual_median": float(np.median(res)),
            "closure_residual_max": float(np.max(res)), "anchor_restriction_rank_per_group": ranks}


def group_scalars(F):
    """The per-group scalar edits a_g = lambda_g I (payload of group g scaled, its siblings' writes scaled
    alike): span{A_g = W^-1 E_g W}, which contains the identity. Exact small generalized eigenproblem of
    ||X^T A-hat||^2 against ||A-hat||^2 on this span; returns eps sorted (the first is the identity, 0)."""
    if F.ri != F.KV * F.hd:
        return None
    Winv, hd = sla.inv(F.Rc), F.hd
    Ah = [F.Mh @ Winv[:, g * hd:(g + 1) * hd] @ F.Rc[g * hd:(g + 1) * hd] for g in range(F.KV)]
    Y = [F.XT(A) for A in Ah]
    P = np.array([[np.sum(a * b) for b in Ah] for a in Ah])
    Q = np.array([[np.sum(a * b) for b in Y] for a in Y])
    w, v = sla.eigh(Q, P)
    order = np.argsort(-w)
    lam = v[:, order[1]]
    return {"eps": np.sqrt(np.clip(1 - w[order], 0, None)).tolist(),
            "second_lambda_normalised": (lam / np.abs(lam).max()).tolist()}


def rot(d, rng):
    q, r = sla.qr(rng.standard_normal((d, d)), check_finite=False)
    return q * np.sign(np.diag(r))


def nulls(Ls, Rs, rng):
    d = Ls[0].shape[0]
    Zs = [rot(d, rng) for _ in Rs]
    twin = ([rot(d, rng) @ L for L in Ls], [R @ Z.T for R, Z in zip(Rs, Zs)])
    Yg = [rot(d, rng) for _ in Rs]
    grp = len(Ls) // len(Rs)
    twin_group = ([Yg[h // grp] @ L for h, L in enumerate(Ls)], [R @ Z.T for R, Z in zip(Rs, Zs)])
    gauss = ([rng.standard_normal(L.shape) * np.linalg.norm(L) / np.sqrt(L.size) for L in Ls],
             [rng.standard_normal(R.shape) * np.linalg.norm(R) / np.sqrt(R.size) for R in Rs])
    return {"twin": twin, "twin_group": twin_group, "gaussian": gauss}


def sibling_images(Ls, grp):
    """Per group: principal-angle cosines between sibling write images, and independence certificate."""
    out = []
    for g in range(len(Ls) // grp):
        Q = [qr(Ls[h]) for h in range(g * grp, (g + 1) * grp)]
        s = svd(np.hstack(Q), compute_uv=False)
        cos = [float(svd(Q[a].T @ Q[b], compute_uv=False)[0])
               for a in range(grp) for b in range(a + 1, grp)]
        out.append({"max_cos": max(cos), "stacked_sigma_min": float(s[-1])})
    return out


def grouped_controls(Ls, Rs, grp):
    """Siblings attending identically: family {sum_{h in g} L_h R_g}_g; forced sum_g M_hd iff the group images
    are independent (stacked sum_{h in g} L_h of full column rank KV hd <= d)."""
    Sg = [sum(Ls[h] for h in range(g * grp, (g + 1) * grp)) for g in range(len(Rs))]
    Q = [qr(S) for S in Sg]
    s = svd(np.hstack(Q), compute_uv=False)
    _, info = orth_basis(np.hstack(Sg), "col")
    return {"stacked_rank": info, "stacked_orthonormal_sigma_min": float(s[-1]),
            "forced_sum_g_M_hd": bool(info["certified_full"] and len(Rs) * Rs[0].shape[0] <= Ls[0].shape[0])}


def routing_invariant(Q, K):
    """Common null space of all query-side and key-side read rows, total and per rotary plane."""
    H, hd, d = Q.shape
    KV, P = K.shape[0], hd // 2

    def null_dim(rows):
        s = svd(rows, compute_uv=False)
        tau = max(rows.shape) * np.finfo(np.float64).eps * s[0]
        r = int((s > tau).sum())
        frac = s / s[0]
        return {"null": d - r, "arch_min_null": max(0, d - rows.shape[0]), "rank": r,
                "n_sigma_below_1e-3": int((frac < 1e-3).sum()) + max(0, d - len(s)),
                "n_sigma_below_1e-2": int((frac < 1e-2).sum()) + max(0, d - len(s))}

    rec = {"query_only": null_dim(Q.reshape(-1, d)), "key_only": null_dim(K.reshape(-1, d)),
           "joint": null_dim(np.vstack([Q.reshape(-1, d), K.reshape(-1, d)]))}
    planes = []
    for j in range(P):
        rows = np.vstack([Q[:, [j, j + P]].reshape(-1, d), K[:, [j, j + P]].reshape(-1, d)])
        planes.append(null_dim(rows))
    rec["per_plane_null"] = [p["null"] for p in planes]
    rec["per_plane_arch_min_null"] = planes[0]["arch_min_null"]
    rec["per_plane_null_equals_arch_min"] = all(p["null"] == p["arch_min_null"] for p in planes)
    return rec


def analyse(Ls, Rs, grp, k, rng, with_char, tol):
    F = Family(Ls, Rs, grp)
    spec, As = spectrum(F, k, rng, tol)
    rec = {"group_scalars": group_scalars(F), "core": {"in": F.in_rank, "out": F.out_rank,
                    "free_A_dims": int(Ls[0].shape[0] * (Ls[0].shape[0] - F.ri)),
                    "free_B_dims": int(Ls[0].shape[0] * (Ls[0].shape[0] - F.ro))}, **spec}
    if with_char and As:
        rec["candidates"] = characterise(F, As, min(6, k))
    # distance to losing the forced blocks: smallest singular value of the per-head orthonormal image bases
    Qo = np.hstack([qr(L) for L in Ls])
    Qi = np.hstack([qr(R.T) for R in Rs])
    rec["write_images_orthonormal_sigma_min"] = float(svd(Qo, compute_uv=False)[-1]) \
        if Qo.shape[1] <= Qo.shape[0] else None
    rec["read_spaces_orthonormal_sigma_min"] = float(svd(Qi, compute_uv=False)[-1])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-6, help="Lanczos Ritz tolerance (eps is recomputed directly)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reuse", help="receipt (or .partial) whose layers keep their Lanczos records; only the "
                                    "cheap per-group-scalar test is added to them")
    args = ap.parse_args()
    from threadpoolctl import ThreadpoolController
    for lib in ThreadpoolController().lib_controllers:  # numpy's BLAS gets the cap; scipy's (ARPACK) one
        lib.set_num_threads(1 if "scipy/.dylibs" in lib.filepath else max(1, args.threads - 1))
    reuse = {}
    if args.reuse:
        import json
        reuse = {r["layer"]: r for r in json.load(open(args.reuse))["layers"]}
    t0 = time.time()
    D = Decoder(args.model)
    rng = np.random.default_rng(2951)
    layers = range(D.L) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    recs = []
    for layer in layers:
        tl = time.time()
        pairs = D.ov(layer)
        Ls = [p[0] for p in pairs]
        Rs = [pairs[g * D.group][1] for g in range(D.KV)]
        assert all(np.array_equal(pairs[h][1], Rs[h // D.group]) for h in range(D.H))
        if layer in reuse:
            rec = reuse[layer]
            nl = nulls(Ls, Rs, rng)
            rec["full"]["group_scalars"] = group_scalars(Family(Ls, Rs, D.group))
            for name, (nL, nR) in nl.items():
                rec["null"][name]["group_scalars"] = group_scalars(Family(nL, nR, D.group))
                rec["null"][name]["group_scalars_draw"] = "fresh null draw (Lanczos record kept from the earlier run)"
            rec["full"]["lanczos_tol"] = rec["full"].get("lanczos_tol", 1e-10)
            recs.append(rec)
            print(f"L{layer:2d} reused; group scalars eps {rec['full']['group_scalars']['eps'][:3]}", flush=True)
            continue
        rec = {"layer": layer}
        rec["full"] = analyse(Ls, Rs, D.group, args.k, rng, True, args.tol)
        nl = nulls(Ls, Rs, rng)
        rec["null"] = {name: analyse(nL, nR, D.group, args.k, rng, False, args.tol)
                       for name, (nL, nR) in nl.items()}
        if "eps_beyond_forced_direct" in rec["full"]:
            e = rec["full"]["eps_beyond_forced_direct"]
            null_min = min(min(rec["null"][n]["eps_beyond_forced_direct"]) for n in ("twin", "gaussian"))
            rec["twin_group_min_eps"] = min(rec["null"]["twin_group"]["eps_beyond_forced_direct"])
            rec["full"]["gap"] = gap_cert(e, null_min)
            rec["full"]["n_below_band"] = int(sum(x <= rec["full"]["band_tau"] for x in e))
            rec["full"]["n_below_null_min"] = int(sum(x < null_min for x in e))
            rec["null_min_eps"] = null_min
        rec["per_group"] = {"siblings": sibling_images(Ls, D.group),
                            "twin_siblings": sibling_images(nl["twin"][0], D.group),
                            "exact_dim_each": D.hd**2, "forced": "M_hd iff sibling images independent"}
        rec["grouped_controls"] = grouped_controls(Ls, Rs, D.group)
        Q, K = D.qk(layer)
        rec["routing_invariant"] = routing_invariant(Q, K)
        rec["seconds"] = time.time() - tl
        recs.append(rec)
        f = rec["full"]
        if "gap" in f:
            beyond = (f"eps1..4 {' '.join(f'{x:.3f}' for x in f['eps_beyond_forced_direct'][:4])} "
                      f"null_min {rec['null_min_eps']:.3f} below_null {f['n_below_null_min']} "
                      f"gap k={f['gap']['k']} 2d/g={f['gap']['ratio_2delta_over_gamma']:.2f}")
        else:
            beyond = f"beyond forced: none (XX^T idempotence err {f['XXT_idempotence_err_max']:.1e})"
        print(f"L{layer:2d} core {f['core']['in']['rank']}/{f['core']['out']['rank']} forced {f['forced_dim']} "
              f"(eps {f['forced_eps']:.1e}, band {f['band_tau']:.1e}) {beyond} "
              f"wimg_smin {f['write_images_orthonormal_sigma_min']} "
              f"sib_cos {max(s['max_cos'] for s in rec['per_group']['siblings']):.3f} "
              f"route_null {rec['routing_invariant']['joint']['null']} {rec['seconds']:.0f}s", flush=True)
        with open(args.out + ".partial", "w") as fh:
            fh.write(compact_json({"layers": recs}))
    out = {"model": args.model, "architecture": D.describe(),
           "quantity": "eps = min_B ||B C_h - C_h A||_F / ||C_h A||_F over heads (M,G-whitened), smallest "
                       "beyond the exactly derived forced subspace",
           "stand_in": "numpy (no Rust owner for joint intertwiner algebras)",
           "nulls": "twin: C_h -> Y_h C_h Z_g^T (random orthogonal, GQA tying kept); twin_group: Y_g(h) instead of Y_h; gaussian: iid factors, "
                    "matched Frobenius norms, tying kept",
           "layers": recs, "seconds_total": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(compact_json(out))
    print(f"total {out['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
