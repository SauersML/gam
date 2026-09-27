"""#2951 known-answer suite: hand-built tiny networks (float64, d <= 32) scored against the operation-first tools.

Each toy's correct explanation is written down in TRUTH below (derived by hand, before any tool runs), and every
tool's output is compared with it. Categories: correct / correct_ambiguity (the truth is an equivalence class and the
tool reports the class) / wrong / na. A "confident wrong" is a wrong verdict stated without any ambiguity flag.

Tools (reused from the probes, imported, not rewritten, unless noted):
  T1  exact-GELU normal form + rank-one unit-graph module split + certified eta
      (bench/mpd_opfirst_gelu_modules_2951.py: Mlp, eta_bound, cos_matrix, refine, random_twin, KAPPA).
      Two pipelines: "raw" = the probe as written (duplicate units only counted), "merged" = sign/duplicate units
      merged first (psi is even, so a_k = s a_j, beta_k = s beta_j merges to u_j + u_k), zero writes dropped, and a
      unit-free block answered by the commutant of its linear part (continuous split family iff dim > d).
  T2  weighted observability (bench/mpd_opfirst_observability_2951.py: close, resolved, effective;
      bench/mpd_opfirst_task_observability_2951.py: principal_cosines, orth_rows). Through an MLP block the
      pull-back Gramian of a readout C is (CA)^T(CA) + sum_j |C u_j|^2 a_j a_j^T (write-weighted); the probe's own
      with_mlp_reads convention (C plus every read row, unweighted) is reported alongside. The closure is the
      probe's Python mirror of state.rs LinearStateQuotient::close (not yet on the CLI surface). Attention: one backward
      step [c; c C_h] per head (probe convention) vs per routing law. Data: the data Gramian X^T X / N.
  T3  activation splits for exact GELU: gelu(t) = t/2 + psi(t) (odd/even) and gelu = relu + e, e(t) = -|t| Phi(-|t|)
      even (GELU analogues of the SiLU splits in bench/mpd_opfirst_mlp_oddeven_2951.py; psi is the probe's).
  T4  spectral plane-rotation recovery: Rust parameter_decomposition::spectral::recover_plane_rotations called
      through `gam parameter-decomposition` (--gam-bin, default target/release/gam). A numpy stand-in of the same
      logic (t4_numpy_standin: polar distance, beta, clusters split only at gaps > 2 beta, RepeatedCosine for 2p
      dims with p >= 2) runs alongside as a labelled cross-check and is the fallback only when no binary exists;
      the receipt's T4_implementation says which one produced the scored numbers. A naive strawman
      (np.linalg.eig, one plane per conjugate pair, no grouping) is scored too, to show what the grouping prevents.
  T5  gauge counts by the declared families of parameter_decomposition::gauge (NUMPY STAND-IN: re-statement of the orbit
      formulas: pass-through GL(r) r^2 - (r - rank A)(r - rank B); GELU hidden units permutation only; rotary QK
      2 m^2 per frequency with m planes). Scored against the function-level fibre dimension, measured as the
      nullity of the Jacobian of theta -> F_theta(X) on generic inputs (torch autograd, float64).
  T6  QK sharing between heads (bench/mpd_opfirst_rope_span_2951.py: atoms, gram_from_factors, captured,
      top_cosine), plus an operator-equality test (same routing law iff equal score operators, no biases).

Run: uv run --no-project --with numpy --with scipy --with torch --with safetensors python bench/mpd_opfirst_toys_planted_2951.py
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
from mpd_opfirst_gelu_modules_2951 import KAPPA, Mlp, cos_matrix, eta_bound, psi, refine, random_twin  # noqa: E402
from mpd_opfirst_observability_2951 import close, effective, resolved  # noqa: E402
from mpd_opfirst_rope_span_2951 import atoms, captured, gram_from_factors, top_cosine  # noqa: E402
from mpd_opfirst_task_observability_2951 import orth_rows, principal_cosines  # noqa: E402

EPS = np.finfo(np.float64).eps
torch.set_default_dtype(torch.float64)

# Known answers, derived by hand before running anything below.
TRUTH = {
    "toy1_paired_copy": {
        "function": "F(h) = gelu(h) - gelu(-h) = h exactly: linear part 1/2 B A = I, psi part cancels pairwise",
        "T1": "no unit survives merging; exact splits = every P = Q (commutant of I, dim d^2 = 64): O(d) ambiguity",
        "T2": "readout c: observable subspace span(c), dim 1",
        "T3": "odd part = h, even/psi part = 0 exactly, e part = 0 exactly",
        "T4": "linear part I: no rotation plane, one fixed cluster of dim d",
        "T5": "fibre = {W_in = [M; -M], W_out = [N, -N], N M = I, b_in = [beta; -beta], b_out = -N beta}: dim d^2 + d = 72;"
              " declared GELU family (permutation only) gives 0 continuous",
    },
    "toy2_two_modules": {
        "T1": "exactly two components (6 and 10 units), rank P = rank Q = 3 and 5 on the planted subspaces, eta = 0",
        "T2": "readout = module-1 output basis: observable subspace = module-1 read subspace, dim 3",
        "T4": "linear part is not a rotation: no plane claimed",
        "T5": "fibre dim 0 (generic GELU units, discrete permutations only)",
    },
    "toy3_cross_edge": {
        "T1": "no exact nontrivial split for any eps > 0; at the planted split eta = eps ||r1 s2^T|| = eps exactly",
        "T2": "exact observable rank 4 (module-1 reads + s2), 4th Gramian eigenvalue O(eps^2)",
        "T5": "fibre dim 0",
    },
    "toy4_rotation": {
        "T4": "angle 1.1: one plane, identified (principal cosines 1); angle 0.3: a 4-dim invariant subspace only"
              " (repeated cosine, planes not identified)",
        "T1": "unit-free linear block; commutant of R has dim 2*2^2 + 2*1^2 = 10 > 6: continuous family of splits",
        "T2": "generic single readout c, transition R - I: closure = Krylov space, dim 4 (one per distinct eigenvalue"
              " e^{+-1.1i}-1, e^{+-0.3i}-1), containing the 1.1 plane and only a 2-dim slice of the 0.3 space",
    },
    "toy5_attention": {
        "routing": "heads 1,2 identical QK => one routing law, transport C1 + C2 of rank 4 (> 2 each); head 3 has"
                   " QK = 2x head 1: same atom span, different pattern, a second routing law",
        "T6": "heads 1-2 same law; head 3 NOT the same law although its atom span is identical",
        "T2": "one step, readout c: per-routing-law observable dim 3 = rank[c; c(C1+C2); c C3]; per-head closure"
              " over-counts to 4",
        "T5": "fibre: QK 2 per plane x 2 planes x 3 heads = 12; OV GL(4) for the merged pair + GL(2) = 20; total 32."
              " Declared per-head families: 12 + 3 x 4 = 24",
        "gauge": "per-head V -> S V, O -> O S^-1 and the cross-head GL(4) on heads 1,2 change no function; tool"
                 " outputs must not change",
    },
    "toy6_subspace_ambiguity": {
        "T2": "data Gramian has null space S^perp (dim 4) containing v: component reads identified only modulo S^perp;"
              " D1(t), D2(t) indistinguishable on data for every t, distinguishable off data",
    },
    "toy7_random_mlp": {
        "T1": "one component, no approximate module beyond the random-draw null",
        "T4": "no plane claimed",
        "T5": "fibre dim 0",
    },
}


def np64(t):
    return t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float64)


def tt(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def haar(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def gelu_np(t):
    return np64(torch.nn.functional.gelu(tt(t)))


def Phi_np(t):
    from scipy.special import ndtr
    return ndtr(t)


# --------------------------------------------------------------------------------------------- T1
def commutant_dim(A):
    d = A.shape[0]
    K = np.kron(np.eye(d), A) - np.kron(A.T, np.eye(d))
    s = np.linalg.svd(K, compute_uv=False)
    if s[0] == 0:
        return d * d
    return int((s <= d * d * EPS * max(s[0], np.linalg.norm(A, 2)) * 16).sum())


def merge_units(W_in, b_in, W_out):
    a, beta, u = W_in.copy(), b_in.copy(), W_out.T.copy()
    n = a.shape[0]
    keep = np.ones(n, bool)
    merges = 0
    for j in range(n):
        if not keep[j]:
            continue
        for k in range(j + 1, n):
            if not keep[k]:
                continue
            for s in (1.0, -1.0):
                if (np.abs(a[k] - s * a[j]).max() <= 1e-12 * np.abs(a[j]).max()
                        and abs(beta[k] - s * beta[j]) <= 1e-12 * (1 + abs(beta[j]))):
                    u[j] += u[k]  # psi even: u_j psi(t) + u_k psi(s t) = (u_j + u_k) psi(t)
                    keep[k] = False
                    merges += 1
                    break
    alive = keep & (np.linalg.norm(u, axis=1) > 1e-12 * max(1.0, np.abs(W_out).max()))
    return a[alive], beta[alive], u[alive], merges, int(keep.sum() - alive.sum())


def t1_mlp(W_in, b_in, W_out, b_out, L=None, merge=True, rng=None):
    d = W_in.shape[1]
    Lm = np.zeros((d, d)) if L is None else L
    A_full = 0.5 * W_out @ W_in + Lm
    b_full = 0.5 * W_out @ b_in + b_out
    if merge:
        a, beta, u, merges, dropped = merge_units(W_in, b_in, W_out)
    else:
        a, beta, u, merges, dropped = W_in, b_in, W_out.T, 0, 0

    def direct(X):
        return gelu_np(X @ W_in.T + b_in) @ W_out.T + b_out + X @ Lm.T

    X = 3 * rng.standard_normal((400, d))
    Fn = X @ A_full.T + b_full + (np64(psi(tt(X @ a.T + beta))) @ u if len(a) else 0)
    out = {"pipeline": "merged" if merge else "raw", "units_in": int(W_in.shape[0]), "sign_merges": merges,
           "zero_write_dropped": dropped, "units_after": int(len(a)),
           "normal_form_rel": float(np.linalg.norm(Fn - direct(X)) / np.linalg.norm(direct(X)))}
    if len(a) == 0:
        c = commutant_dim(A_full)
        out.update({"graph_components": 0, "linear_only": True, "commutant_dim": c, "d": d,
                    "verdict": "linear block; exact splits = invariant subspace pairs of A; "
                               + ("continuous family (commutant > d): internal axes not identified" if c > d
                                  else "finitely many")})
        out["ambiguity_reported"] = c > d
        return out
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    Ca, Cu = np64(cos_matrix(tt(a))), np64(cos_matrix(tt(u)))
    adj = (np.abs(Ca) > 1e-10) | (np.abs(Cu) > 1e-10)
    ncomp, lab = connected_components(csr_matrix(adj.astype(np.int8)), directed=False)
    m = Mlp(tt(a), tt(beta), tt(u.T.copy()), tt(b_out))
    m.A, m.b = tt(A_full), tt(b_full)
    scale = float(np.linalg.norm(A_full, 2) + KAPPA * (np.linalg.norm(a, axis=1) * np.linalg.norm(u, axis=1)).sum())
    band = 1e3 * EPS * scale
    comps = []
    for g in range(ncomp):
        mask = lab == g
        P = orth_rows(a[mask]).T
        Q = orth_rows(u[mask]).T
        Pp, Qp = P @ P.T, Q @ Q.T
        eb = eta_bound(m, tt(Pp), tt(Qp))
        # empirical replacement on generic pairs, against the certified bound
        Xa, Ya = 3 * rng.standard_normal((500, d)), 3 * rng.standard_normal((500, d))
        Z = Xa @ Pp.T + Ya @ (np.eye(d) - Pp).T
        R = direct(Z) - direct(Xa) @ Qp.T - direct(Ya) @ (np.eye(d) - Qp).T
        emp = np.linalg.norm(R, axis=1) / np.linalg.norm(Xa - Ya, axis=1)
        comps.append({"units": int(mask.sum()), "rank_P": int(P.shape[1]), "rank_Q": int(Q.shape[1]),
                      "eta": eb["eta"], "eta_lin": eb["lin"], "eta_nonlin": eb["nonlin"],
                      "emp_max_over_dx": float(emp.max()), "bound_holds": bool((emp <= eb["eta"] * (1 + 1e-9) + 1e-14).all()),
                      "P_basis": P})
    exact = ncomp >= 2 and max(c["eta"] for c in comps) <= band
    out.update({"graph_components": int(ncomp), "labels": lab.tolist(), "scale": scale, "exact_band": band,
                "graph_only_verdict": "split into %d modules" % ncomp if ncomp >= 2 else "no split",
                "eta_verdict": ("exact split" if exact else
                                ("approximate split, eta_max = %.3e" % max(c["eta"] for c in comps)) if ncomp >= 2
                                else "no split"),
                "components": [{k: v for k, v in c.items() if k != "P_basis"} for c in comps],
                "ambiguity_reported": False})
    out["_P"] = [c["P_basis"] for c in comps]
    return out


def t1_null_search(W_in, b_in, W_out, b_out, rng, gen, restarts=4, iters=20):
    """Probe's Lloyd refine at k = d/2 from random starts; eta / scale (lower = more module-like)."""
    m = Mlp(tt(W_in), tt(b_in), tt(W_out), tt(b_out))
    n, d = W_in.shape
    scale = float(np.linalg.norm(np64(m.A), 2) + KAPPA * (np.linalg.norm(W_in, axis=1)
                                                         * np.linalg.norm(W_out, axis=0)).sum())
    best = math.inf
    for _ in range(restarts):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[torch.randperm(n, generator=gen)[: n // 2]] = True
        _, P, Qp = refine(m, mask, d // 2, iters)
        best = min(best, eta_bound(m, P, Qp)["eta"] / scale)
    return best


# --------------------------------------------------------------------------------------------- T2
def gram_report(factor, planted=None):
    _, sigma, _, rank = resolved(factor, 0.0)
    _, s, vt = np.linalg.svd(factor, full_matrices=False)
    out = {"exact_rank": rank, "sigma_over_max": (s / s[0]).tolist(),
           "participation_ratio": effective(s)["participation_ratio"]}
    if planted is not None:
        top = vt[: planted.shape[0]]
        out["top_k_principal_cos_to_planted"] = principal_cosines(top, planted).tolist()
        out["exact_space_principal_cos_to_planted"] = principal_cosines(vt[:rank], planted).tolist()
    return out


def mlp_pullback(C, W_in, W_out, A):
    cu = np.linalg.norm(C @ W_out, axis=0)  # |C u_j|
    return np.vstack([C @ A, cu[:, None] * W_in])


# --------------------------------------------------------------------------------------------- T3
def t3_gelu(W_in, b_in, W_out, b_out, X):
    pre = X @ W_in.T + b_in
    F = gelu_np(pre) @ W_out.T + b_out
    ps = np64(psi(tt(pre)))
    e = -np.abs(pre) * Phi_np(-np.abs(pre))
    psi_part, e_part = ps @ W_out.T, e @ W_out.T
    nF = np.linalg.norm(F)
    per_unit = sum(np.linalg.norm(np.outer(ps[:, j], W_out[:, j])) for j in range(W_in.shape[0]))
    Fm = gelu_np(-X @ W_in.T + b_in) @ W_out.T + b_out
    return {"identity_rel": float(np.linalg.norm(F - (0.5 * pre @ W_out.T + b_out + psi_part)) / nF),
            "psi_part_over_F": float(np.linalg.norm(psi_part) / nF),
            "e_part_over_F": float(np.linalg.norm(e_part) / nF),
            "sum_unit_psi_norms_over_F": float(per_unit / nF),
            "even_part_over_F": float(np.linalg.norm((F + Fm) / 2 - b_out) / nF)}


# --------------------------------------------------------------------------------------------- T4
def t4_numpy_standin(W):
    """NUMPY STAND-IN for spectral.rs recover_plane_rotations (bands simplified to d eps scale); used only as a
    cross-check of the Rust result, or as the fallback when no `gam` binary is available (recorded in the receipt)."""
    d = W.shape[0]
    U, s, Vt = np.linalg.svd(W)
    band = 4 * d * EPS * s[0]
    if s[-1] <= band:
        return {"refused": "singular within band"}
    rho_bar = float(np.abs(s - 1).max() + band)
    S = 0.5 * (W + W.T)
    cosv, V = np.linalg.eigh(S)
    beta = rho_bar + 4 * d * EPS * np.linalg.norm(W) + 4 * d * EPS * np.abs(cosv).max()
    cuts = [0] + [i + 1 for i in range(d - 1) if cosv[i + 1] - cosv[i] > 2 * beta] + [d]
    clusters = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        c = cosv[lo:hi]
        dim = hi - lo
        if c.max() + beta >= 1:
            kind, planes = "fixed", 0
        elif c.min() - beta <= -1:
            kind, planes = "half_turn", 0
        else:
            kind, planes = ("rotation", dim // 2) if dim % 2 == 0 else ("unresolved_odd", 0)
        clusters.append({"dim": dim, "kind": kind, "planes": planes,
                         "angle": float(np.arccos(np.clip(c.mean(), -1, 1))),
                         "unresolved_beta_ge_1": bool(beta >= 1),
                         "planes_identified": kind == "rotation" and planes == 1,
                         "repeated_cosine_ambiguity": kind == "rotation" and planes >= 2, "_basis": V[:, lo:hi]})
    return {"rho_bar": rho_bar, "beta": float(beta), "clusters": clusters,
            "planes_claimed": int(sum(c["planes"] for c in clusters if c["planes_identified"])),
            "subspaces_claimed": int(sum(1 for c in clusters if c["kind"] == "rotation"))}


GAM_BIN = None


def t4_rust(W):
    """parameter_decomposition::spectral::recover_plane_rotations through `gam parameter-decomposition`."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "w.npy", np.ascontiguousarray(W, dtype=np.float64))
        (tmp / "req.json").write_text(json.dumps({"schema": "gam.mpd-request", "schema_version": 1,
                                                  "operation": {"kind": "recover_plane_rotations", "tensor": "w"}}))
        subprocess.run([GAM_BIN, "parameter-decomposition", "--request", str(tmp / "req.json"), "--tensor",
                        "w=%s" % (tmp / "w.npy"), "--out", str(tmp / "out")], check=True, capture_output=True)
        res = json.loads((tmp / "out" / "report.json").read_text())["result"]
        clusters = []
        for cl in res["clusters"]:
            st = cl["structure"]
            planes = st.get("planes", 0)
            clusters.append({"dim": cl["dimension"], "kind": st["kind"], "planes": planes,
                             "angle": st.get("angle", st.get("max_hidden_angle", st.get("min_hidden_angle"))),
                             "cosine_interval": cl["cosine_interval"], "projector_bar": cl["projector_bar"],
                             "planes_identified": st["kind"] == "rotation" and planes == 1,
                             "repeated_cosine_ambiguity": st["kind"] == "rotation" and planes >= 2,
                             "_basis": np.load(tmp / "out" / (cl["basis"] + ".npy"))})
    return {"source": "rust", "rho_bar": res["orthogonality_defect"], "beta": res["perturbation_bound"],
            "ambiguities": res["ambiguities"], "clusters": clusters,
            "planes_claimed": int(sum(c["planes"] for c in clusters if c["planes_identified"])),
            "subspaces_claimed": int(sum(1 for c in clusters if c["kind"] == "rotation"))}


def t4_planes(W):
    """Rust plane recovery when a `gam` binary is available, with the numpy stand-in recorded as a cross-check."""
    mirror = t4_numpy_standin(W)
    mirror_summary = [(c["dim"], c["kind"], c["planes"]) for c in mirror["clusters"]]
    if GAM_BIN is None:
        return mirror | {"source": "numpy_standin"}
    out = t4_rust(W)
    rust_summary = [(c["dim"], c["kind"], c["planes"]) for c in out["clusters"]]
    out["numpy_standin_clusters"] = mirror_summary
    out["numpy_standin_agrees"] = mirror_summary == rust_summary
    return out


def t4_naive(W):
    lam, vec = np.linalg.eig(W)
    planes = []
    for i in range(len(lam)):
        if lam[i].imag > 1e-12:
            planes.append((float(abs(np.angle(lam[i]))), orth_rows(np.vstack([vec[:, i].real, vec[:, i].imag]))))
    return planes


# --------------------------------------------------------------------------------------------- T5 oracle
def fibre_dim(f, theta):
    J = torch.autograd.functional.jacobian(f, theta)
    s = torch.linalg.svdvals(J).numpy()
    rel = s / s[0]
    null = int((rel < 1e-9).sum()) + max(0, theta.numel() - len(s))
    k = len(s) - int((rel < 1e-9).sum())
    return {"params": int(theta.numel()), "outputs": int(J.shape[0]), "nullity": null,
            "sigma_last_kept_rel": float(rel[k - 1]), "sigma_first_null_rel": float(rel[k]) if k < len(s) else 0.0}


def mlp_oracle(W_in, b_in, W_out, b_out, X, L=None):
    shapes = [W_in.shape, b_in.shape, W_out.shape, b_out.shape] + ([L.shape] if L is not None else [])
    flat = torch.cat([tt(p).flatten() for p in [W_in, b_in, W_out, b_out] + ([L] if L is not None else [])])
    Xt = tt(X)

    def f(th):
        parts, o = [], 0
        for sh in shapes:
            k = int(np.prod(sh))
            parts.append(th[o:o + k].reshape(sh))
            o += k
        y = torch.nn.functional.gelu(Xt @ parts[0].T + parts[1]) @ parts[2].T + parts[3]
        if L is not None:
            y = y + Xt @ parts[4].T
        return y.flatten()
    return fibre_dim(f, flat)


# --------------------------------------------------------------------------------------------- toys
def score(ok, ambiguity=False):
    return ("correct_ambiguity" if ambiguity else "correct") if ok else "wrong"


def toy1(rng):
    d = 8
    W_in = np.vstack([np.eye(d), -np.eye(d)])
    W_out = np.hstack([np.eye(d), -np.eye(d)])
    b_in, b_out = np.zeros(2 * d), np.zeros(d)
    X = 3 * rng.standard_normal((300, d))
    F = gelu_np(X @ W_in.T + b_in) @ W_out.T
    rep = {"d": d, "copy_rel_err": float(np.linalg.norm(F - X) / np.linalg.norm(X))}
    raw, mer = t1_mlp(W_in, b_in, W_out, b_out, merge=False, rng=rng), t1_mlp(W_in, b_in, W_out, b_out, rng=rng)
    raw.pop("_P", None)
    mer.pop("_P", None)
    rep["T1_raw"], rep["T1_merged"] = raw, mer
    c = rng.standard_normal((1, d))
    A = 0.5 * W_out @ W_in
    rep["T2_weighted_raw_units"] = gram_report(mlp_pullback(c, W_in, W_out, A), orth_rows(c))
    rep["T2_probe_with_mlp_reads"] = gram_report(np.vstack([c, W_in]), orth_rows(c))
    rep["T2_after_T1_merge"] = gram_report(c @ A, orth_rows(c))  # no unit survives the merge
    rep["T3"] = t3_gelu(W_in, b_in, W_out, b_out, X)
    t4 = t4_planes(A)
    rep["T4"] = {k: v for k, v in t4.items() if k != "clusters"} | {
        "clusters": [{k: v for k, v in cl.items() if k != "_basis"} for cl in t4["clusters"]]}
    orc = mlp_oracle(W_in, b_in, W_out, b_out, 3 * rng.standard_normal((120, d)))
    rep["T5"] = {"declared_continuous": 0, "declared_note": "GELU hidden units: permutation only",
                 "fibre_oracle": orc, "truth": d * d + d}
    sc = {
        "T1": {"raw": score(raw["graph_components"] == 0),  # raw claims d axis-aligned modules with eta = 0
               "merged": score(mer["units_after"] == 0 and mer["commutant_dim"] == d * d, ambiguity=True)},
        "T2": {"weighted_raw_units": score(rep["T2_weighted_raw_units"]["exact_rank"] == 1),
               "probe_with_mlp_reads": score(rep["T2_probe_with_mlp_reads"]["exact_rank"] == 1),
               "after_merge": score(rep["T2_after_T1_merge"]["exact_rank"] == 1)},
        "T3": score(rep["T3"]["psi_part_over_F"] < 1e-13 and rep["T3"]["e_part_over_F"] < 1e-13),
        "T4": score(t4["planes_claimed"] == 0 and t4["subspaces_claimed"] == 0 and len(t4["clusters"]) == 1),
        "T5": score(orc["nullity"] == 0),  # declared families give 0 continuous; oracle vs hand truth in rep["T5"]
    }
    return rep, sc


def planted_modules(rng, d=8, dims=(3, 5), units=(6, 10)):
    R = haar(d, rng)
    rows_a, rows_u, lab = [], [], []
    off = 0
    for mi, (dm, nm) in enumerate(zip(dims, units)):
        Rm = R[:, off:off + dm]
        rows_a.append(rng.standard_normal((nm, dm)) @ Rm.T)
        rows_u.append(rng.standard_normal((nm, dm)) @ Rm.T)
        lab += [mi] * nm
        off += dm
    perm = rng.permutation(sum(units))
    W_in = np.vstack(rows_a)[perm]
    W_out = np.vstack(rows_u)[perm].T.copy()
    return W_in, 0.5 * rng.standard_normal(sum(units)), W_out, 0.1 * rng.standard_normal(d), R, np.array(lab)[perm]


def t1_module_check(t1, truth_lab, R, dims):
    if t1["graph_components"] != 2:
        return False, None
    lab = np.array(t1["labels"])
    same = (lab == lab[0]) == (truth_lab == truth_lab[0])
    cos = []
    for P in t1["_P"]:
        k = P.shape[1]
        planted = R[:, :dims[0]].T if k == dims[0] else R[:, dims[0]:].T
        cos.append(float(principal_cosines(P.T, planted).min()) if planted.shape[0] == k else 0.0)
    return bool(same.all()), cos


def toy2(rng):
    d, dims = 8, (3, 5)
    W_in, b_in, W_out, b_out, R, lab = planted_modules(rng)
    t1 = t1_mlp(W_in, b_in, W_out, b_out, rng=rng)
    part, cos = t1_module_check(t1, lab, R, dims)
    t1["partition_matches_truth"], t1["min_principal_cos_to_planted"] = part, cos
    t1.pop("_P")
    C = R[:, :3].T
    A = 0.5 * W_out @ W_in
    rep = {"d": d, "dims": dims, "T1": t1,
           "T2_weighted": gram_report(mlp_pullback(C, W_in, W_out, A), R[:, :3].T),
           "T2_probe_with_mlp_reads": gram_report(np.vstack([C, W_in]), R[:, :3].T)}
    t4 = t4_planes(A)
    rep["T4"] = {k: v for k, v in t4.items() if k != "clusters"} | {
        "clusters": [{k: v for k, v in cl.items() if k != "_basis"} for cl in t4["clusters"]]}
    orc = mlp_oracle(W_in, b_in, W_out, b_out, 3 * rng.standard_normal((150, d)))
    rep["T5"] = {"declared_continuous": 0, "fibre_oracle": orc, "truth": 0}
    g = rep["T2_weighted"]
    sc = {"T1": score(part and "exact" == t1["eta_verdict"].split()[0] and min(cos) > 1 - 1e-10),
          "T2": {"weighted": score(g["exact_rank"] == 3 and min(g["exact_space_principal_cos_to_planted"]) > 1 - 1e-10),
                 "probe_with_mlp_reads": score(rep["T2_probe_with_mlp_reads"]["exact_rank"] == 3)},
          "T3": "na", "T4": score(t4["planes_claimed"] == 0),
          "T5": score(orc["nullity"] == 0)}
    return rep, sc


def toy3(rng):
    d, dims = 8, (3, 5)
    W_in, b_in, W_out, b_out, R, lab = planted_modules(rng)
    r1, s2 = R[:, 0], R[:, 3]
    reps, scs = {}, {}
    for eps in (1e-6, 1e-3, 1e-1):
        L = eps * np.outer(r1, s2)
        t1 = t1_mlp(W_in, b_in, W_out, b_out, L=L, rng=rng)
        part, cos = t1_module_check(t1, lab, R, dims)
        t1["partition_matches_truth"] = part
        t1.pop("_P")
        etas = [c["eta"] for c in t1["components"]]
        t1["eta_over_eps"] = [e / eps for e in etas]
        A = 0.5 * W_out @ W_in + L
        g = gram_report(mlp_pullback(R[:, :3].T, W_in, W_out, A), R[:, :3].T)
        sig = np.array(g["sigma_over_max"])
        g["sigma4_over_max_sq"] = float(sig[3] ** 2) if len(sig) > 3 else 0.0
        g["sigma4_sq_over_eps_sq"] = g["sigma4_over_max_sq"] / eps ** 2
        orc = mlp_oracle(W_in, b_in, W_out, b_out, 3 * rng.standard_normal((150, d)), L=L)
        reps[f"eps={eps:g}"] = {"T1": t1, "T2_weighted": g, "T5": {"declared_continuous": 0, "fibre_oracle": orc}}
        exact_claimed = t1["eta_verdict"] == "exact split"
        scs[f"eps={eps:g}"] = {
            "T1_graph_only": score(t1["graph_components"] < 2),  # the unit graph alone says "exact split"
            "T1_eta": score(not exact_claimed and all(abs(r - 1) < 1e-6 for r in t1["eta_over_eps"])),
            # exact space = planted reads + s2; the 4th direction's weight scales as eps^2 (top-3 tilt O(eps) is fine)
            "T2": score(g["exact_rank"] == 4 and min(g["exact_space_principal_cos_to_planted"]) > 1 - 1e-10),
            "T5": score(orc["nullity"] == 0)}
    return reps, scs


def toy4(rng):
    d = 6
    U = haar(d, rng)
    angles = (0.3, 0.3, 1.1)

    def rot(t):
        return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    B = np.zeros((d, d))
    for i, t in enumerate(angles):
        B[2 * i:2 * i + 2, 2 * i:2 * i + 2] = rot(t)
    Rm = U @ B @ U.T
    planes = [U[:, 2 * i:2 * i + 2].T for i in range(3)]
    t4 = t4_planes(Rm)
    rep = {"d": d, "angles": angles, "T4": {k: t4.get(k) for k in ("source", "rho_bar", "beta", "ambiguities",
                                                                    "numpy_standin_clusters", "numpy_standin_agrees")}}
    rep["T4"]["clusters"] = []
    ok_11, ok_03 = False, False
    for cl in t4["clusters"]:
        basis = cl["_basis"].T
        info = {k: v for k, v in cl.items() if k != "_basis"}
        if abs(cl["angle"] - 1.1) < 1e-6:
            info["principal_cos_to_planted_1.1"] = principal_cosines(basis, planes[2]).tolist()
            ok_11 = cl["planes_identified"] and min(info["principal_cos_to_planted_1.1"]) > 1 - 1e-10
        if abs(cl["angle"] - 0.3) < 1e-6:
            info["principal_cos_to_planted_0.3_span"] = principal_cosines(basis, np.vstack(planes[:2])).tolist()
            ok_03 = cl["repeated_cosine_ambiguity"] and cl["dim"] == 4 and \
                min(info["principal_cos_to_planted_0.3_span"]) > 1 - 1e-10
        rep["T4"]["clusters"].append(info)
    naive = t4_naive(Rm)
    rep["T4_naive_strawman"] = [{"angle": a, "best_match_min_principal_cos": max(
        float(principal_cosines(p, q).min()) for q in planes)} for a, p in naive]
    naive_wrong = any(r["best_match_min_principal_cos"] < 1 - 1e-6 for r in rep["T4_naive_strawman"])
    c_dim = commutant_dim(Rm)
    rep["T1"] = {"units": 0, "commutant_dim": c_dim, "d": d, "continuous_split_family": c_dim > d}
    c = rng.standard_normal((1, d))
    chart, steps = close(orth_rows(c), [Rm - np.eye(d)], None)
    rep["T2_closure"] = {"rank": int(chart.shape[0]), "steps": steps,
                         "principal_cos_to_1.1_plane": principal_cosines(chart, planes[2]).tolist(),
                         "principal_cos_to_0.3_space": principal_cosines(chart, np.vstack(planes[:2])).tolist()}
    cos03 = np.array(rep["T2_closure"]["principal_cos_to_0.3_space"])
    sc = {"T4": score(ok_11 and ok_03, ambiguity=True),
          "T4_naive_strawman": score(not naive_wrong),
          "T1": score(c_dim == 10, ambiguity=True),
          "T2": score(chart.shape[0] == 4 and min(rep["T2_closure"]["principal_cos_to_1.1_plane"]) > 1 - 1e-8
                      and int((cos03 > 1 - 1e-8).sum()) == 2),
          "T3": "na", "T5": "na"}
    return rep, sc


def rope(v, omega):
    Tn, P = v.shape[-2], v.shape[-1] // 2
    pos = torch.arange(Tn, dtype=torch.float64)[:, None] * omega[None, :]
    c, s = torch.cos(pos), torch.sin(pos)
    return torch.cat([v[..., :P] * c - v[..., P:] * s, v[..., P:] * c + v[..., :P] * s], -1)


def attn_forward(Q, K, V, O, X, omega):
    """Causal multi-head rope attention; Q, K: (H, hd, d); V: (H, r, d); O: (H, d, r); X: (B, T, d)."""
    Tn, hd = X.shape[1], Q.shape[1]
    mask = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1)
    out = 0
    for h in range(Q.shape[0]):
        q, k = rope(X @ Q[h].T, omega), rope(X @ K[h].T, omega)
        sc = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        a = torch.softmax(sc.masked_fill(mask, -math.inf), -1)
        out = out + a @ (X @ V[h].T) @ O[h].T
    return out


def toy5(rng):
    d, hd, r, H = 16, 4, 2, 3
    omega = torch.tensor([1.0, 0.1])
    Q1, K1 = rng.standard_normal((hd, d)) / 4, rng.standard_normal((hd, d)) / 4
    Q = np.stack([Q1, Q1, 2 * Q1])
    K = np.stack([K1, K1, K1])
    V = rng.standard_normal((H, r, d)) / 4
    O = rng.standard_normal((H, d, r)) / math.sqrt(r)
    X = torch.as_tensor(rng.standard_normal((24, 6, d)))
    c = rng.standard_normal((1, d))

    def tools(Vg, Og):
        Cs = [Og[h] @ Vg[h] for h in range(H)]
        U_, V_ = atoms(Q, K)
        G = gram_from_factors(U_, V_)
        idx = np.arange(G.shape[0]).reshape(H, hd // 2, 2)
        pairs = {}
        for h, h2 in ((0, 1), (0, 2), (1, 2)):
            a, b = idx[h].ravel(), idx[h2].ravel()
            diff2 = sum(G[i, i] + G[j, j] - 2 * G[i, j] for i, j in zip(a, b))
            pairs[f"{h + 1}-{h2 + 1}"] = {"captured": 0.5 * (captured(G, a, b) + captured(G, b, a)),
                                          "top_cos": top_cosine(G, a, b),
                                          "operator_rel_diff": float(math.sqrt(max(diff2, 0) / np.trace(G[np.ix_(a, a)])))}
        laws = [[0]]
        for h in range(1, H):
            for law in laws:
                key = f"{law[0] + 1}-{h + 1}"
                if pairs[key]["operator_rel_diff"] < 1e-12:
                    law.append(h)
                    break
            else:
                laws.append([h])
        transports = [sum(Cs[h] for h in law) for law in laws]
        per_head = np.vstack([c] + [c @ Ch for Ch in Cs])
        per_law = np.vstack([c] + [c @ Ct for Ct in transports])
        return {"pairs": pairs, "routing_laws": [[h + 1 for h in law] for law in laws],
                "law_transport_ranks": [int(resolved(Ct, 0.0)[3]) for Ct in transports],
                "per_head_ov_ranks": [int(resolved(Ch, 0.0)[3]) for Ch in Cs],
                "per_head_ov_fro": [float(np.linalg.norm(Ch)) for Ch in Cs],
                "T2_per_head_rank": int(resolved(per_head, 0.0)[3]), "T2_per_law_rank": int(resolved(per_law, 0.0)[3]),
                "_per_head_space": orth_rows(per_head), "_per_law_space": orth_rows(per_law), "_transports": transports}

    base = tools(V, O)
    # per-head OV gauge V -> S V, O -> O S^-1
    S = [np.eye(r) + 0.5 * rng.standard_normal((r, r)) for _ in range(H)]
    Vg = np.stack([S[h] @ V[h] for h in range(H)])
    Og = np.stack([O[h] @ np.linalg.inv(S[h]) for h in range(H)])
    per_head_g = tools(Vg, Og)
    # cross-head GL(4) on heads 1, 2 (identical patterns)
    M = np.eye(2 * r) + 0.5 * rng.standard_normal((2 * r, 2 * r))
    Vs = M @ np.vstack([V[0], V[1]])
    Os = np.hstack([O[0], O[1]]) @ np.linalg.inv(M)
    Vx, Ox = V.copy(), O.copy()
    Vx[0], Vx[1], Ox[0], Ox[1] = Vs[:r], Vs[r:], Os[:, :r], Os[:, r:]
    cross_g = tools(Vx, Ox)

    def fn_diff(Va, Oa):
        f0 = attn_forward(tt(Q), tt(K), tt(V), tt(O), X, omega)
        f1 = attn_forward(tt(Q), tt(K), tt(Va), tt(Oa), X, omega)
        return float((f1 - f0).norm() / f0.norm())

    def invariance(g):
        return {"function_rel_change": None,
                "routing_laws_same": g["routing_laws"] == base["routing_laws"],
                "law_transport_max_diff": max(float(np.abs(a - b).max()) for a, b in zip(g["_transports"], base["_transports"])),
                "T2_per_law_space_min_cos": float(principal_cosines(g["_per_law_space"], base["_per_law_space"]).min()),
                "T2_per_head_space_min_cos": float(principal_cosines(g["_per_head_space"], base["_per_head_space"]).min()),
                "per_head_ov_fro": g["per_head_ov_fro"]}

    inv_head, inv_cross = invariance(per_head_g), invariance(cross_g)
    inv_head["function_rel_change"] = fn_diff(Vg, Og)
    inv_cross["function_rel_change"] = fn_diff(Vx, Ox)
    # declared gauge counts (gauge.rs families) and the fibre oracle
    declared = {"rotary_qk": H * (hd // 2) * 2, "ov_per_head": H * r * r}
    shapes = [(H, hd, d), (H, hd, d), (H, r, d), (H, d, r)]
    flat = torch.cat([tt(p).flatten() for p in (Q, K, V, O)])

    def f(th):
        parts, o = [], 0
        for sh in shapes:
            k = int(np.prod(sh))
            parts.append(th[o:o + k].reshape(sh))
            o += k
        return attn_forward(*parts, X, omega).flatten()
    orc = fibre_dim(f, flat)
    rep = {"d": d, "head_dim": hd, "ov_rank": r, "heads": H,
           "tools": {k: v for k, v in base.items() if not k.startswith("_")},
           "gauge_per_head": inv_head, "gauge_cross_head_GL4": inv_cross,
           "T5": {"declared": declared, "declared_total": sum(declared.values()), "fibre_oracle": orc, "truth": 32}}
    p = base["pairs"]
    sc = {"T6_captured_as_sharing": {"1-2": score(p["1-2"]["captured"] > 1 - 1e-9),
                                     "1-3": score(p["1-3"]["captured"] < 1 - 1e-6)},
          "T6_operator_equality": score(base["routing_laws"] == [[1, 2], [3]]),
          "combined_transport": score(base["law_transport_ranks"] == [4, 2]),
          "T2": {"per_head": score(base["T2_per_head_rank"] == 3), "per_law": score(base["T2_per_law_rank"] == 3)},
          "gauge_invariance": {
              "per_head_S": score(inv_head["function_rel_change"] < 1e-12 and inv_head["law_transport_max_diff"] < 1e-12
                                  and inv_head["T2_per_head_space_min_cos"] > 1 - 1e-10),
              "cross_head_GL4_per_law_tools": score(inv_cross["law_transport_max_diff"] < 1e-12
                                                    and inv_cross["T2_per_law_space_min_cos"] > 1 - 1e-10),
              "cross_head_GL4_per_head_tools": score(inv_cross["T2_per_head_space_min_cos"] > 1 - 1e-10)},
          "T5": score(sum(declared.values()) == orc["nullity"]),
          "T1": "na", "T3": "na", "T4": "na"}
    return rep, sc


def toy6(rng):
    d, ds = 8, 4
    Qb = haar(d, rng)
    Sb, v = Qb[:, :ds], Qb[:, ds]
    u, a, b = rng.standard_normal(d), rng.standard_normal(d), rng.standard_normal(d)
    X = rng.standard_normal((256, ds)) @ Sb.T
    Xoff = rng.standard_normal((256, d))
    D1 = lambda t: np.outer(u, a + t * v)  # noqa: E731
    D2 = lambda t: np.outer(u, b - t * v)  # noqa: E731
    ts = (0.0, 0.5, 2.0)
    on = max(float(np.abs(X @ (D1(t) - D1(0)).T).max()) for t in ts)
    off = max(float(np.abs(Xoff @ (D1(t) - D1(0)).T).max()) for t in ts)
    sums = max(float(np.abs(D1(t) + D2(t) - np.outer(u, a + b)).max()) for t in ts)
    g = gram_report(X / math.sqrt(len(X)))
    _, s, vt = np.linalg.svd(X / math.sqrt(len(X)))
    null = vt[g["exact_rank"]:]
    rep = {"d": d, "data_dim": ds, "t_values": ts, "sum_invariant_maxdiff": sums,
           "component_on_data_maxdiff": on, "component_off_data_maxdiff": off,
           "T2_data_gramian": {"exact_rank": g["exact_rank"], "null_dim": int(null.shape[0]),
                               "v_in_null_norm": float(np.linalg.norm(null @ v)),
                               "v_energy_in_gramian": float(v @ (X.T @ X / len(X)) @ v),
                               "report": "component reads identified modulo null(X^T X) (dim %d)" % null.shape[0]}}
    t2 = rep["T2_data_gramian"]
    sc = {"T2": score(t2["null_dim"] == d - ds and abs(t2["v_in_null_norm"] - 1) < 1e-10, ambiguity=True),
          "T1": "na", "T3": "na", "T4": "na", "T5": "na"}
    return rep, sc


def rand_mlp(rng, d=16, n=64):
    return (rng.standard_normal((n, d)) / math.sqrt(d), 0.5 * rng.standard_normal(n),
            rng.standard_normal((d, n)) / math.sqrt(n), 0.1 * rng.standard_normal(d))


def toy7(rng, gen, n_null=6):
    d, n = 16, 64
    W_in, b_in, W_out, b_out = rand_mlp(rng, d, n)
    t1 = t1_mlp(W_in, b_in, W_out, b_out, rng=rng)
    t1.pop("_P")
    t1.pop("labels")
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    Ca, Cu = np64(cos_matrix(tt(W_in))), np64(cos_matrix(tt(W_out.T)))
    twin = random_twin(Mlp(tt(W_in), tt(b_in), tt(W_out), tt(b_out)), gen)
    Ta, Tu = np64(cos_matrix(twin.W_in)), np64(cos_matrix(twin.W_out.T))
    thr = {}
    for tau in (0.1, 0.2, 0.3, 0.5):
        row = {}
        for label, (ca, cu) in (("toy", (Ca, Cu)), ("twin", (Ta, Tu))):
            adj = ((np.abs(ca) > tau) | (np.abs(cu) > tau)) & ~np.eye(n, dtype=bool)
            nc, lab = connected_components(csr_matrix(adj.astype(np.int8)), directed=False)
            row[label] = {"components": int(nc), "largest": np.sort(np.bincount(lab))[::-1][:3].tolist()}
        thr[str(tau)] = row
    toy_eta = t1_null_search(W_in, b_in, W_out, b_out, rng, gen)
    null_eta = [t1_null_search(*rand_mlp(rng, d, n), rng, gen) for _ in range(n_null)]
    twin_eta = t1_null_search(np64(twin.W_in), b_in, np64(twin.W_out), b_out, rng, gen)
    ratio = toy_eta / float(np.mean(null_eta))
    t1.update({"threshold_components": thr, "refined_eta_over_scale": toy_eta, "null_draws_eta_over_scale": null_eta,
               "twin_eta_over_scale": twin_eta, "ratio_vs_null_mean": ratio,
               "module_claimed": bool(ratio < 0.5 or t1["graph_components"] >= 2)})
    A = 0.5 * W_out @ W_in
    t4 = t4_planes(A)
    t4c = {k: v for k, v in t4.items() if k != "clusters"} | {
        "clusters": [{k: v for k, v in cl.items() if k != "_basis"} for cl in t4["clusters"]]}
    # T4 on I + A as well (a residual reading of the block's linear part)
    t4r = t4_planes(np.eye(d) + A)
    C = rng.standard_normal((2, d))
    g = gram_report(mlp_pullback(C, W_in, W_out, A))
    null_pr = [gram_report(mlp_pullback(C, m[0], m[2], 0.5 * m[2] @ m[0]))["participation_ratio"]
               for m in (rand_mlp(rng, d, n) for _ in range(n_null))]
    orc = mlp_oracle(W_in, b_in, W_out, b_out, 2 * rng.standard_normal((300, d)))
    rep = {"d": d, "hidden": n, "T1": t1, "T4": t4c,
           "T4_residual_I_plus_A": {"rho_bar": t4r["rho_bar"], "planes_claimed": t4r["planes_claimed"],
                                    "subspaces_claimed": t4r["subspaces_claimed"], "source": t4r["source"],
                                    "numpy_standin_agrees": t4r.get("numpy_standin_agrees")},
           "T2_weighted": {"exact_rank": g["exact_rank"], "participation_ratio": g["participation_ratio"],
                           "null_participation_ratios": null_pr},
           "T5": {"declared_continuous": 0, "fibre_oracle": orc, "truth": 0}}
    sc = {"T1": score(not t1["module_claimed"]),
          "T4": score(t4["planes_claimed"] == 0 and t4["subspaces_claimed"] == 0),
          "T4_on_I_plus_A": score(t4r["planes_claimed"] == 0 and t4r["subspaces_claimed"] == 0),
          "T2": score(min(null_pr) * 0.8 <= g["participation_ratio"] <= max(null_pr) * 1.25),
          "T3": "na", "T5": score(orc["nullity"] == 0)}
    return rep, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2951)
    ap.add_argument("--out", default="experiments/issue-2951/receipts/opfirst_toys_planted.json")
    ap.add_argument("--gam-bin", default=str(Path(__file__).resolve().parents[1] / "target/release/gam"),
                    help="`gam` CLI for Rust plane recovery; the numpy stand-in is used (and labelled) if absent")
    args = ap.parse_args()
    global GAM_BIN
    GAM_BIN = args.gam_bin if Path(args.gam_bin).exists() else None
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    report = {"script": "bench/mpd_opfirst_toys_planted_2951.py", "seed": args.seed, "dtype": "float64",
              "T4_implementation": ("rust: gam parameter-decomposition recover_plane_rotations (numpy stand-in"
                                    " recorded per call as numpy_standin_clusters / numpy_standin_agrees)"
                                    if GAM_BIN else "numpy_standin (no gam binary found; Rust not called)"),
              "truth": TRUTH}
    scorecard = {}
    for name, fn in (("toy1_paired_copy", toy1), ("toy2_two_modules", toy2), ("toy3_cross_edge", toy3),
                     ("toy4_rotation", toy4), ("toy5_attention", toy5), ("toy6_subspace_ambiguity", toy6),
                     ("toy7_random_mlp", lambda r: toy7(r, gen))):
        ts = time.time()
        rep, sc = fn(rng)
        report[name] = rep
        scorecard[name] = sc
        print(name, "%.1fs" % (time.time() - ts), json.dumps(sc), flush=True)
    report["scorecard"] = scorecard
    report["runtime_s"] = time.time() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # one top-level key per line: build.rs caps tracked files at 10k lines
    out.write_text("{\n" + ",\n".join("%s:%s" % (json.dumps(k), json.dumps(v, separators=(",", ":")))
                                      for k, v in report.items()) + "\n}\n")
    print("wrote", out, "runtime %.1fs" % report["runtime_s"])


if __name__ == "__main__":
    main()
