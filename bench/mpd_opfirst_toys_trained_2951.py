"""#2951 probe: do the operation-first tools find structure in TRAINED networks whose ground-truth computation is known?

Pythia GELU MLPs showed no parallel modules (OPFIRST.md result 5). This asks whether that is a property of trained
networks or of our tools, using toys where the task's structure is known and a structured solution exists in the
student's own function class. Training runs in float32 on CPU (the toys are tiny); all analysis is float64 on CPU and
reuses the unit-graph machinery of bench/mpd_opfirst_gelu_modules_2951.py (normal form, certified eta, empirical
replacement, spectral split, Lloyd refine, random-split null) unchanged.

ReLU units use the same algebra: relu(t) = t/2 + |t|/2, so F(x) = A x + b + sum_j u_j psi(a_j^T x + beta_j) with
psi(t) = |t|/2 (even, Lipschitz 1/2), A = 1/2 W_out W_in. The certified bound is eta = ||QA - AP|| + 1/2 sum_j ||D_j||
(kappa = 1/2 in place of the GELU kappa), derived exactly as in the GELU module.

A. Modular task: x ~ N(0, I_8); y = S1 f1(R1^T x) + S2 f2(R2^T x) with [R1 R2], [S1 S2] Haar rotations (4 + 4 dims)
   and f_k fixed random GELU nets 4 -> 8 -> 4 (the teacher is itself a modular 16-unit GELU MLP, so an exact split
   with eta = 0 exists in every student width >= 16). Students: 8 -> h -> 8 exact-GELU MLP, h in {16, 64, 256}, with
   and without weight decay, from default init and from a modular init (units assigned to a module, reads and writes
   projected into its subspaces). Measured: eta and replacement at the TRUE projectors, per-unit mixing
   D_j / (|u_j| |a_j|), the function change when every unit is snapped into its dominant module, and what the tools
   find without the truth (thresholded unit graph, spectral + refine at rank 4) against a random-split null.
B. Toy Model of Superposition (Elhage et al. 2022): x in [0,1]^n sparse, x_hat = relu(W^T W x + b), importance
   0.9^i (n = 20, m = 5) or uniform (n = 5, m = 2). Measured: represented features, feature dimensionality, the unit
   graph of the readout relus (read a_i = row i of W^T W, write e_i) and its component split (certified eta and
   replacement), the task-weighted observability Gramian W diag(imp E[x^2]) W^T against the feature directions, and
   the relu active set against the true support.
C. Compressed computation (Braun et al. 2025 APD residual-MLP toy, MLP path only): x in [-1,1]^100, each feature
   active w.p. 0.01; r = x E with fixed random unit rows E (d = 1000); y_hat = (W_out relu(W_in r + b) + c) E^T,
   target relu(x). Every partition of features is an exact split of the TASK. h in {50, 100, 200} neurons. Analysis
   in feature coordinates (A = W_in E^T, U = E W_out): neurons per feature, features per neuron, the unit graph, a
   truth split (feature halves) certified and executed on sparse and dense pairs, and the tool's own spectral split.
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
from mpd_opfirst_gelu_modules_2951 import (  # noqa: E402
    KAPPA, Mlp, components, cos_matrix, eta_bound, haar_projector, pi_split, q, refine, replacement,
    spectral_labels, split_projectors)

F64 = torch.float64
TAUS = (0.05, 0.1, 0.2, 0.3, 0.5)
NUMERIC_OWNERS = (
    "python/torch analysis, no Rust owner: parameter_decomposition has no unit-graph module split or replacement "
    "certificate (checked state, spectral, gauge, secant, operators); the unit-graph split, certified eta, "
    "replacement, refine and nulls are imported unchanged from bench/mpd_opfirst_gelu_modules_2951.py; this file "
    "adds only the relu kappa = 1/2 rescaling, per-unit mixing, TMS/compressed-computation measurements and torch "
    "training (SPEC PyTorch exception). Pi-graph additive blocks: pi_split (E* = sum min(lam, 1-lam)) imported "
    "unchanged, Fiedler proposals of the Pi^2 Laplacian by dense eigh in this file; no Rust owner exists for either "
    "(surface.rs exposes recover_plane_rotations and mlp_block_receipt only)")


# LAPACK backend for the Pi / E_p eigen-computations. torch on macOS arm64 links Accelerate for LAPACK even in the
# OpenBLAS venv, and Accelerate is reported to return wrong results on rank-deficient SVD/QR/eigh; --linalg scipy routes
# every eigh/SVD/QR of the proposer and nulls through scipy.linalg (OpenBLAS in ~/mpd-data/venv).
LINALG = "torch"


def eigh(M):
    if LINALG == "scipy":
        from scipy.linalg import eigh as sp_eigh
        ev, V = sp_eigh(M.detach().numpy(), check_finite=True)
        return torch.from_numpy(ev), torch.from_numpy(V)
    return torch.linalg.eigh(M)


def svd(M):
    if LINALG == "scipy":
        from scipy.linalg import svd as sp_svd
        Us, sv, Vt = sp_svd(M.detach().numpy(), full_matrices=False, check_finite=True)
        return torch.from_numpy(Us), torch.from_numpy(sv), torch.from_numpy(Vt)
    return torch.linalg.svd(M, full_matrices=False)


def qr(M):
    if LINALG == "scipy":
        from scipy.linalg import qr as sp_qr
        Qm, Rm = sp_qr(M.detach().numpy(), mode="economic", check_finite=True)
        return torch.from_numpy(Qm), torch.from_numpy(Rm)
    return torch.linalg.qr(M)


def env_record():
    import platform
    rec = {"python": platform.python_version(), "executable": sys.executable, "torch": torch.__version__,
           "numpy": np.__version__, "linalg_backend": LINALG}
    try:
        cfg = np.show_config(mode="dicts")["Build Dependencies"]
        rec["numpy_blas"], rec["numpy_lapack"] = cfg["blas"]["name"], cfg["lapack"]["name"]
    except Exception as exc:  # older numpy
        rec["numpy_blas"] = "unknown (%s)" % type(exc).__name__
    show = torch.__config__.show()
    rec["torch_lapack"] = "accelerate" if "LAPACK_INFO=accelerate" in show else ("mkl" if "USE_MKL=ON" in show else "other")
    return rec


class ReluMlp(Mlp):
    kappa = 0.5

    def direct(self, X):
        return torch.relu(X @ self.W_in.T + self.b_in) @ self.W_out.T + self.b_out


def kappa_of(mlp):
    return getattr(mlp, "kappa", KAPPA)


def eta(mlp, P, Qp):
    eb = eta_bound(mlp, P, Qp)  # nonlin carries the GELU kappa; rescale for relu
    nonlin = eb["nonlin"] * kappa_of(mlp) / KAPPA
    return {"eta": eb["lin"] + nonlin, "lin": eb["lin"], "nonlin": nonlin}


def scale_of(mlp):
    a, u = mlp.W_in, mlp.W_out.T
    return float(torch.linalg.matrix_norm(mlp.A, ord=2) + kappa_of(mlp) * (a.norm(dim=1) * u.norm(dim=1)).sum())


def unit_mixing(mlp, P, Qp):
    """D_j / (|u_j| |a_j|) in [0, 1]: 0 iff unit j reads and writes inside one side of (P, Q)."""
    a, u = mlp.W_in, mlp.W_out.T
    I = torch.eye(P.shape[0], dtype=F64)
    D = torch.maximum((u @ Qp).norm(dim=1) * (a @ (I - P)).norm(dim=1), (u @ (I - Qp)).norm(dim=1) * (a @ P).norm(dim=1))
    w = a.norm(dim=1) * u.norm(dim=1)
    r = D / w.clamp_min(1e-300)
    return {"median": float(r.median()), "weighted_mean": float((D.sum() / w.sum())),
            "frac_below_0.1": float((r < 0.1).double().mean()), "frac_below_0.01": float((r < 0.01).double().mean())}


def graph_components(mlp, taus=TAUS):
    a, u = mlp.W_in, mlp.W_out.T
    n = a.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    keep = (a.norm(dim=1) * u.norm(dim=1)) > 1e-8 * (a.norm(dim=1) * u.norm(dim=1)).max()
    Ca, Cu = cos_matrix(a).abs(), cos_matrix(u).abs()
    out = {}
    for tau in taus:
        adj = ((Ca > tau) | (Cu > tau)) & off & keep[:, None] & keep[None, :]
        nc, sizes = components(adj)
        out[str(tau)] = {"components": nc - int((~keep).sum()), "largest": sizes[:4]}
    return out


def split_report(mlp, P, Qp, pairs, scale):
    eb = eta(mlp, P, Qp)
    rep = {"eta_over_scale": eb["eta"] / scale, "lin_over_scale": eb["lin"] / scale}
    for name, (X, Y) in pairs.items():
        emp, emp_rel = replacement(mlp, P, Qp, X, Y)
        emp_rel = emp_rel[torch.isfinite(emp_rel)]
        rep[name] = {"emp_over_dF_median": float(emp_rel.median()), "emp_over_dF_p90": float(emp_rel.quantile(0.9)),
                     "bound_holds": bool((emp <= eb["eta"] * (1 + 1e-9) + 1e-12).all())}
    return rep


def principal_cos(P1, P2, k):
    V1 = torch.linalg.eigh(P1)[1][:, -k:]
    V2 = torch.linalg.eigh(P2)[1][:, -k:]
    return torch.linalg.svdvals(V1.T @ V2).tolist()


def random_masks(n, size, count, gen):
    out = []
    for _ in range(count):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.randperm(n, generator=gen)[:size]] = True
        out.append(m)
    return out


def dump(path, report):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # one top-level key per line: build.rs caps tracked files at 10k lines
    Path(path).write_text("{\n" + ",\n".join("%s:%s" % (json.dumps(k), json.dumps(v, separators=(",", ":")))
                                            for k, v in report.items()) + "\n}\n")


# ------------------------------------------------------------------------------------------ Pi-graph additive blocks

PI_FRACS = (1 / 16, 1 / 8, 1 / 4, 3 / 8, 1 / 2)


def read_basis(W):
    """W = U T with U orthonormal on ran W (rank-revealing: SVD in place of the Pythia probe's QR, same Pi and E*)."""
    Us, sv, Vt = svd(W)
    r = int((sv > 1e-10 * sv[0]).sum())
    U = Us[:, :r]
    # canonical column signs (largest-magnitude entry positive): LAPACKs differ in SVD signs, and the proposer's
    # random-subspace starts are drawn in these coordinates, so without this the search depends on the backend
    sg = torch.sign(U.gather(0, U.abs().argmax(0, keepdim=True))).flatten()
    sg[sg == 0] = 1.0
    return U * sg, (sg[:, None] * sv[:r, None]) * Vt[:r]


def dense_proposals(U, T, n_vec, fracs):
    """The Pythia probe's Fiedler proposals (orderings by the smallest nontrivial eigenvectors of the Pi^2 Laplacian,
    low and high ends at each fraction), with a dense eigh (the probe's Lanczos solve did not always converge; the
    analyse receipts record where it fell back)."""
    vec = laplacian_vectors(U, n_vec)
    n, r = U.shape
    out = []
    for v in range(vec.shape[1]):
        order = vec[:, v].argsort()
        for f in fracs:
            k = max(1, round(f * n))
            for side, idx in (("low", order[:k]), ("high", order[-k:])):
                mask = torch.zeros(n, dtype=torch.bool)
                mask[idx] = True
                res, _ = pi_split(U, T, mask)
                out.append({"vec": v + 1, "k": k, "side": side, "E_over_U2": res["E_star"] / r, "mask": mask})
    return out


def pi_best(U, T, n_vec=3):
    """Best Fiedler proposal (lowest E*/rank) at each proposal size."""
    best = {}
    for pr in dense_proposals(U, T, min(n_vec, U.shape[0] - 2), PI_FRACS):
        if pr["k"] not in best or pr["E_over_U2"] < best[pr["k"]]["E_over_U2"]:
            best[pr["k"]] = pr
    return best


def pi_replace(mlp, U, T, mask, X):
    """Execute the exactly separated reads: rel. squared change of F on X (vs the variance of F). The reads are
    pi_split's (P = eigenvectors of U_S^T U_S above 1/2), computed through the dispatched eigh."""
    lam, V = eigh(U[mask].T @ U[mask])
    Vp = V[:, lam > 0.5]
    Pp = Vp @ Vp.T
    What = torch.where(mask[:, None], U @ Pp, U - U @ Pp) @ T
    Mh = type(mlp)(What, mlp.b_in, mlp.W_out, mlp.b_out)
    F, Fh = mlp.direct(X), Mh.direct(X)
    return float(((Fh - F) ** 2).sum() / ((F - F.mean(0)) ** 2).sum()), Mh


def pi_report(mlp, X, gen, truth=None, n_rand=20):
    """Pi = U U^T components, E*/rank for truth partitions and Fiedler proposals vs random subsets and a twin whose
    read directions are redrawn isotropically (norms kept), and executed replacement for the truth and best proposal.
    truth: {name: unit mask}. E*/rank = E_over_U2 of the Pythia probe."""
    W = mlp.W_in
    n = W.shape[0]
    U, T = read_basis(W)
    r = U.shape[1]
    out = {"units": n, "read_rank": r}
    if r >= n:  # generic reads with n <= rank: Pi = I, every unit is its own additive block in the coordinates T x
        out["trivial"] = "units <= read rank: Pi = I, every unit is an exact additive block"
        return out
    Pi = U @ U.T
    off = ~torch.eye(n, dtype=torch.bool)
    out["components"] = {str(t): dict(zip(("n", "largest"), components((Pi.abs() > t) & off)))
                         for t in (1e-10, 1e-6, 1e-3, 1e-2, 3e-2, 0.1)}
    best = pi_best(U, T)
    G = torch.randn(W.shape, generator=gen, dtype=F64)
    Ut, Tt = read_basis(G / G.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True))
    best_t = pi_best(Ut, Tt)

    def rand_E(k):
        vals = [pi_split(U, T, m)[0]["E_star"] / r for m in random_masks(n, k, n_rand, gen)]
        return float(np.mean(vals)), float(np.std(vals))

    by_k = []
    for k in sorted(best):
        mu, sd = rand_E(k)
        by_k.append({"k": k, "fiedler_E": best[k]["E_over_U2"], "random_mean": mu, "random_sd": sd,
                     "twin_fiedler_E": best_t[k]["E_over_U2"] if k in best_t else None,
                     "ratio_vs_random": best[k]["E_over_U2"] / mu if mu > 0 else None,
                     "ratio_vs_twin": best[k]["E_over_U2"] / best_t[k]["E_over_U2"]
                     if k in best_t and best_t[k]["E_over_U2"] > 0 else None})
    out["fiedler_by_k"] = by_k
    kh = max(best)  # the half split
    rep_f, Mf = pi_replace(mlp, U, T, best[kh]["mask"], X)
    out["fiedler_half"] = {"k": kh, "E": best[kh]["E_over_U2"], "replacement_rel_mse": rep_f,
                           "mask": best[kh]["mask"]}
    out["truth"] = {}
    for name, m in (truth or {}).items():
        if m.sum() == 0 or m.all():
            out["truth"][name] = {"size": int(m.sum()), "degenerate": True}
            continue
        res, _ = pi_split(U, T, m)
        mu, sd = rand_E(int(m.sum()))
        rep, _ = pi_replace(mlp, U, T, m, X)
        tw = [pi_split(Ut, Tt, mm)[0]["E_star"] / r for mm in random_masks(n, int(m.sum()), n_rand, gen)]
        out["truth"][name] = {"size": int(m.sum()), "E": res["E_star"] / r, "rank_P": res["rank_P"],
                              "random_mean": mu, "random_sd": sd, "ratio_vs_random": res["E_star"] / r / mu,
                              "twin_random_mean": float(np.mean(tw)), "replacement_rel_mse": rep,
                              "fiedler_half_agreement": float(max((best[kh]["mask"] == m).double().mean(),
                                                                  (best[kh]["mask"] != m).double().mean()))}
    return out


def pi_input_cos(U, T, mask, R1):
    """Principal cosines between the input subspace read by the P block of the optimal split and span(R1)."""
    G = U[mask].T @ U[mask]
    lam, V = torch.linalg.eigh(0.5 * (G + G.T))
    B = torch.linalg.qr(T.T @ V[:, lam > 0.5])[0]
    B2 = torch.linalg.qr(T.T @ V[:, lam <= 0.5])[0]
    c1 = torch.linalg.svdvals(B.T @ R1).tolist() if B.shape[1] else []
    c2 = torch.linalg.svdvals(B2.T @ R1).tolist() if B2.shape[1] else []
    return max(c1, c2, key=lambda c: (len(c) > 0) and (sum(c) / max(len(c), 1)))


def strip_masks(rep):
    if isinstance(rep, dict):
        return {k: strip_masks(v) for k, v in rep.items() if k != "mask"}
    if isinstance(rep, list):
        return [strip_masks(v) for v in rep]
    return rep


# ----------------------------------------------------------------------------------------------------------- toy A

def haar(d, gen):
    Qm, Rm = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=F64))
    return Qm * torch.sign(torch.diagonal(Rm))


def make_task_a(seed):
    gen = torch.Generator().manual_seed(seed)
    d, k, hid = 8, 4, 8
    R, S = haar(d, gen), haar(d, gen)
    mods = []
    for _ in range(2):
        U = 1.5 * torch.randn(hid, k, generator=gen, dtype=F64) / math.sqrt(k)
        c = 0.5 * torch.randn(hid, generator=gen, dtype=F64)
        V = torch.randn(k, hid, generator=gen, dtype=F64) / math.sqrt(hid)
        mods.append((U, c, V))
    # teacher as one modular 16-unit GELU MLP: W_in rows U_k R_k^T, W_out columns S_k V_k
    W_in = torch.cat([mods[0][0] @ R[:, :k].T, mods[1][0] @ R[:, k:].T])
    b_in = torch.cat([mods[0][1], mods[1][1]])
    W_out = torch.cat([S[:, :k] @ mods[0][2], S[:, k:] @ mods[1][2]], 1)
    teacher = Mlp(W_in, b_in, W_out, torch.zeros(d, dtype=F64))
    X = torch.randn(20000, d, generator=gen, dtype=F64)
    Yt = teacher.direct(X)
    sd = Yt.std(0).mean()
    teacher = Mlp(W_in, b_in, W_out / sd, -Yt.mean(0) / sd)
    P = R[:, :k] @ R[:, :k].T
    Qp = S[:, :k] @ S[:, :k].T
    return teacher, P, Qp


def student_a(d, width, init, seed, Pf, Qf):
    torch.manual_seed(1000 * width + seed)
    net = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Linear(width, d))
    if init == "modular":
        with torch.no_grad():
            side = torch.arange(width) < width // 2
            I = torch.eye(d)
            net[0].weight.copy_(torch.where(side[:, None], net[0].weight @ Pf, net[0].weight @ (I - Pf)))
            wo = net[2].weight.T
            net[2].weight.copy_(torch.where(side[:, None], wo @ Qf, wo @ (I - Qf)).T)
    return net


def a_configs(args):
    return [("default", "adamw", 0.0), ("default", "adamw", args.wd_a), ("default", "sgd", 0.0),
            ("modular", "adamw", 0.0), ("modular", "sgd", 0.0)]


def train_mlp(model, sample, steps, lr, wd, test, loss_fn=None, optimizer="adamw", probe=None):
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:  # rotation-equivariant: SGD commutes with an orthogonal change of input/output basis, Adam does not
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    loss_fn = loss_fn or (lambda p, y: ((p - y) ** 2).mean())
    curve = []
    for s in range(steps):
        X, Y = sample()
        loss = loss_fn(model(X), Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % max(steps // 10, 1) == 0 or s == steps - 1:
            with torch.no_grad():
                Xt, Yt = test
                point = {"step": s, "test_loss": float(loss_fn(model(Xt), Yt))}
                curve.append(point | (probe(model) if probe else {}))
    return curve


def student_to_mlp(net):
    l1, l2 = net[0], net[2]
    return Mlp(l1.weight.detach().double(), l1.bias.detach().double(), l2.weight.detach().double(),
               l2.bias.detach().double())


def analyse_a(mlp, teacher, P, Qp, X, Y, Xt, gen, restarts):
    n, d = mlp.W_in.shape
    k = 4
    scale = scale_of(mlp)
    pairs = {"gauss_pairs": (X, Y)}
    Yt = teacher.direct(Xt)
    Ft = mlp.direct(Xt)
    rel_mse = float(((Ft - Yt) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum())
    out = {"width": n, "task_rel_mse": rel_mse, "scale": scale}
    out["truth_split"] = split_report(mlp, P, Qp, pairs, scale) | {"unit_mixing": unit_mixing(mlp, P, Qp)}
    # snap every unit into its dominant module (read P or I-P, write Q or I-Q) and re-execute
    a, u = mlp.W_in, mlp.W_out.T
    I = torch.eye(d, dtype=F64)
    fit1 = (a @ P).norm(dim=1) * (u @ Qp).norm(dim=1)
    fit2 = (a @ (I - P)).norm(dim=1) * (u @ (I - Qp)).norm(dim=1)
    side = fit1 >= fit2
    a_s = torch.where(side[:, None], a @ P, a @ (I - P))
    u_s = torch.where(side[:, None], u @ Qp, u @ (I - Qp))
    snapped = Mlp(a_s, mlp.b_in, u_s.T.contiguous(), mlp.b_out)
    Fs = snapped.direct(Xt)
    out["snapped"] = {"rel_mse_vs_task": float(((Fs - Yt) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()),
                      "rel_change_vs_student": float(((Fs - Ft) ** 2).sum() / ((Ft - Ft.mean(0)) ** 2).sum()),
                      "units_side1": int(side.sum())}
    # relu split of gelu: gelu = relu + e, e even, |e| <= 0.17
    pre = Xt @ mlp.W_in.T + mlp.b_in
    Fr = torch.relu(pre) @ mlp.W_out.T + mlp.b_out
    out["relu_part_fve"] = float(1 - ((Ft - Fr) ** 2).sum() / ((Ft - Ft.mean(0)) ** 2).sum())
    # Pi-graph additive blocks (reads only, any invertible input change allowed): truth candidates and Fiedler
    ra, rb = (a @ P).pow(2).sum(1), (a @ (I - P)).pow(2).sum(1)
    w = a.norm(dim=1) * u.norm(dim=1)
    pi = pi_report(mlp, Xt[:4096], gen, {"read_dominant": ra >= rb, "read_write_dominant": side})
    R1 = torch.linalg.eigh(P)[1][:, -k:]
    if "trivial" not in pi:
        U, T = read_basis(a)
        pi["fiedler_half"]["input_cos_vs_truth"] = pi_input_cos(U, T, pi["fiedler_half"]["mask"], R1)
        pi["truth"]["read_dominant"]["input_cos_vs_truth"] = pi_input_cos(U, T, ra >= rb, R1)
    pi["read_minority_energy_weighted"] = float((w * torch.minimum(ra, rb) / (ra + rb)).sum() / w.sum())
    out["pi"] = strip_masks(pi)
    # the tools without the truth
    out["graph"] = graph_components(mlp)
    Ca, Cu = cos_matrix(a), cos_matrix(u)
    off = ~torch.eye(n, dtype=torch.bool)
    W = (Ca * Ca + Cu * Cu) * off
    spec = spectral_labels(W, 2, 0) == 0
    starts = [spec, ~spec] + random_masks(n, n // 2, restarts, gen)
    best = None
    for m0 in starts:
        if m0.sum() == 0 or m0.all():
            continue
        m2, Pf, Qf = refine(mlp, m0, k, 15)
        e = eta(mlp, Pf, Qf)["eta"]
        if best is None or e < best[0]:
            best = (e, m2, Pf, Qf)
    _, m2, Pf, Qf = best
    found = split_report(mlp, Pf, Qf, pairs, scale)
    cp = principal_cos(Pf, P, k)
    cp2 = principal_cos(Pf, I - P, k)
    cq = principal_cos(Qf, Qp, k) if cp[-1] >= cp2[-1] else principal_cos(Qf, I - Qp, k)
    found |= {"size": int(m2.sum()), "P_vs_truth_cos": max(cp, cp2, key=lambda c: c[-1]), "Q_vs_truth_cos": cq}
    out["tool_split_rank4"] = found
    # without the true rank: best refined eta at each rank k, against a random split refined at the same rank
    scan = {}
    for kk in range(1, d):
        size = round(n * kk / d)
        bestk = min(eta(mlp, *refine(mlp, m0, kk, 15)[1:])["eta"] for m0 in random_masks(n, size, restarts, gen))
        rnd = np.mean([eta(mlp, *split_projectors(mlp, m0, kk)[:2])["eta"] for m0 in random_masks(n, size, 3, gen)])
        scan[str(kk)] = {"eta_over_scale": bestk / scale, "ratio_vs_random_split": bestk / rnd}
    out["rank_scan"] = scan
    nulls = [split_report(mlp, *split_projectors(mlp, m, k)[:2], pairs, scale) for m in random_masks(n, n // 2, 3, gen)]
    out["random_split_null"] = {
        "eta_over_scale": float(np.mean([z["eta_over_scale"] for z in nulls])),
        "emp_over_dF_median": float(np.mean([z["gauss_pairs"]["emp_over_dF_median"] for z in nulls]))}
    Ph, Qh = haar_projector(d, k, gen), haar_projector(d, k, gen)
    out["haar_PQ_null"] = split_report(mlp, Ph, Qh, pairs, scale)
    return out


def run_a(args, gen):
    teacher, P, Qp = make_task_a(args.seed + 101)
    d = 8
    Xt = torch.randn(8192, d, generator=gen, dtype=F64)
    X = torch.randn(512, d, generator=gen, dtype=F64)
    Y = torch.randn(512, d, generator=gen, dtype=F64)
    Yt = teacher.direct(Xt)
    lin = torch.linalg.lstsq(torch.cat([Xt, torch.ones(len(Xt), 1, dtype=F64)], 1), Yt).solution
    lin_res = float(((torch.cat([Xt, torch.ones(len(Xt), 1, dtype=F64)], 1) @ lin - Yt) ** 2).sum()
                    / ((Yt - Yt.mean(0)) ** 2).sum())
    report = {"toy": "A_modular_task", "args": vars(args), "kappa_gelu": KAPPA,
              "task": {"d": d, "module_dims": [4, 4], "teacher_units": 16, "linear_fit_rel_mse": lin_res},
              "teacher_control": analyse_a(teacher, teacher, P, Qp, X, Y, Xt, gen, args.restarts)}
    tp = report["teacher_control"]["pi"]["truth"]["read_dominant"]
    print("A teacher: eta_true/scale=%.2e graph0.05=%s linear_rel_mse=%.3f | PI truth E=%.2e rand=%.3f fiedler half "
          "E=%.2e agree=%.2f" % (
              report["teacher_control"]["truth_split"]["eta_over_scale"], report["teacher_control"]["graph"]["0.05"],
              lin_res, tp["E"], tp["random_mean"], report["teacher_control"]["pi"]["fiedler_half"]["E"],
              tp["fiedler_half_agreement"]), flush=True)
    Pf, Qf = P.float(), Qp.float()
    tgen = torch.Generator().manual_seed(args.seed + 7)
    Xtest = torch.randn(4096, d, generator=tgen).float()
    test = (Xtest, teacher.direct(Xtest.double()).float())

    def sample():
        Xb = torch.randn(args.batch, d)
        return Xb, teacher.direct(Xb.double()).float()

    runs = []
    configs = a_configs(args)
    for width in args.widths_a:
        for init, optimizer, wd in configs:
            if True:
                for seed in range(args.seeds):
                    t0 = time.time()
                    net = student_a(d, width, init, seed, Pf, Qf)
                    at_init = analyse_a(student_to_mlp(net), teacher, P, Qp, X, Y, Xt, gen, args.restarts)
                    lr = args.lr_a if optimizer == "adamw" else args.lr_sgd_a
                    def probe(net):
                        m = student_to_mlp(net)
                        return {"truth_eta_over_scale": eta(m, P, Qp)["eta"] / scale_of(m)}

                    curve = train_mlp(net, sample, args.steps_a, lr, wd, test, optimizer=optimizer, probe=probe)
                    res = analyse_a(student_to_mlp(net), teacher, P, Qp, X, Y, Xt, gen, args.restarts)
                    res |= {"init": init, "optimizer": optimizer, "weight_decay": wd, "seed": seed, "loss_curve": curve,
                            "converged": res["task_rel_mse"] < args.conv_a,
                            "at_init": {k2: at_init[k2] for k2 in ("task_rel_mse", "truth_split", "graph", "pi",
                                                                   "tool_split_rank4", "random_split_null")},
                            "seconds": time.time() - t0}
                    runs.append(res)
                    ts, tl = res["truth_split"], res["tool_split_rank4"]
                    pt, pti = res["pi"], at_init["pi"]
                    if "trivial" not in pt:
                        rd, fh = pt["truth"]["read_dominant"], pt["fiedler_half"]
                        print("  PI truth E=%.4f (init %.4f) rand=%.4f twin=%.4f repl=%.2e cos=%s | fiedler half E=%.4f "
                              "agree=%.2f repl=%.2e cos=%s | minority=%.3f (init %.3f) comps1e-3=%s" % (
                                  rd["E"], pti["truth"]["read_dominant"]["E"], rd["random_mean"], rd["twin_random_mean"],
                                  rd["replacement_rel_mse"], [round(c, 3) for c in rd["input_cos_vs_truth"]], fh["E"],
                                  rd["fiedler_half_agreement"], fh["replacement_rel_mse"],
                                  [round(c, 3) for c in fh["input_cos_vs_truth"]], pt["read_minority_energy_weighted"],
                                  pti["read_minority_energy_weighted"], pt["components"]["0.001"]), flush=True)
                    print("A h=%d %s %s wd=%g s%d relmse=%.1e conv=%s | truth eta/sc=%.3f (init %.3f) mix=%.3f "
                          "empdF=%.3f | snap relmse=%.2e | tool eta/sc=%.3f empdF=%.3f Pcos_min=%.3f | rand eta/sc=%.3f "
                          "empdF=%.3f | g0.1=%s (%.0fs)" % (
                              width, init, optimizer, wd, seed, res["task_rel_mse"], res["converged"], ts["eta_over_scale"],
                              at_init["truth_split"]["eta_over_scale"], ts["unit_mixing"]["weighted_mean"],
                              ts["gauss_pairs"]["emp_over_dF_median"], res["snapped"]["rel_mse_vs_task"],
                              tl["eta_over_scale"], tl["gauss_pairs"]["emp_over_dF_median"], tl["P_vs_truth_cos"][-1],
                              res["random_split_null"]["eta_over_scale"], res["random_split_null"]["emp_over_dF_median"],
                              res["graph"]["0.1"], res["seconds"]), flush=True)
    report["runs"] = runs
    return report


# ----------------------------------------------------------------------------------------------------------- toy B

def tms_sample(n_inst, batch, n, sparsity, gen=None):
    x = torch.rand(n_inst, batch, n, generator=gen)
    keep = torch.rand(n_inst, batch, n, generator=gen) >= sparsity[:, None, None]
    return x * keep


def train_tms(n, m, imp_base, args):
    """All sparsities x seeds as one batch of independent instances (Adam is per-parameter, so instances do not mix)."""
    S = torch.tensor(args.sparsities_b).repeat_interleave(args.seeds)
    n_inst = len(S)
    imp = imp_base ** torch.arange(n, dtype=torch.float32)
    torch.manual_seed(args.seed + n)
    W = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(n_inst, m, n)))
    b = torch.nn.Parameter(torch.zeros(n_inst, n))
    opt = torch.optim.Adam([W, b], lr=1e-3)
    hist = []
    for step in range(args.steps_b):
        x = tms_sample(n_inst, 1024, n, S)
        xh = torch.relu(torch.einsum("ibn,imn,imk->ibk", x, W, W) + b[:, None])
        per = (imp * (x - xh) ** 2).mean((1, 2))
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
        if step >= args.steps_b - 1000:
            hist.append(per.detach())
    return W, b, torch.stack(hist)


def run_b(args, gen):
    report = {"toy": "B_superposition", "args": vars(args), "configs": []}
    for n, m, imp_base in ((5, 2, 1.0), (20, 5, 0.9)):
        n_inst = len(args.sparsities_b) * args.seeds
        imp = imp_base ** torch.arange(n, dtype=torch.float32)
        t0 = time.time()
        W, b, late = train_tms(n, m, imp_base, args)
        g2 = torch.Generator().manual_seed(args.seed + 3)
        cfg = {"n": n, "m": m, "importance_base": imp_base, "train_seconds": time.time() - t0, "instances": []}
        for i in range(n_inst):
            Wi, bi = W[i].detach().double(), b[i].detach().double()
            sp = args.sparsities_b[i // args.seeds]
            mlp = ReluMlp(Wi.T @ Wi, bi, torch.eye(n, dtype=F64), torch.zeros(n, dtype=F64))
            xs = tms_sample(1, 20000, n, torch.tensor([sp]), g2)[0].double()
            xh = mlp.direct(xs)
            loss = float((imp.double() * (xs - xh) ** 2).mean())
            norms = Wi.norm(dim=0)
            rep = norms > 0.5
            Wn = Wi / norms.clamp_min(1e-12)
            dim_i = norms ** 2 / ((Wn.T @ Wi) ** 2).sum(1).clamp_min(1e-300)
            scale = scale_of(mlp)
            # unit graph on the readout relus (reads = rows of W^T W, writes e_i) and the hidden-space feature graph
            graph = graph_components(mlp, (1e-6, 0.01, 0.1, 0.3))
            Ch = cos_matrix(Wi.T).abs()
            off = ~torch.eye(n, dtype=torch.bool)
            hgraph = {}
            for tau in (0.01, 0.1, 0.3):
                adj = (Ch > tau) & off & rep[:, None] & rep[None, :]
                nc, sizes = components(adj)
                hgraph[str(tau)] = {"components_of_represented": nc - int((~rep).sum()), "largest": sizes[:4]}
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            adj = ((Ch > 0.1) & off & rep[:, None] & rep[None, :]).numpy().astype(np.int8)
            _, lab = connected_components(csr_matrix(adj), directed=False)
            lab = torch.tensor(lab)
            # component split: the largest represented component vs everything else (coordinate projectors)
            xs2 = tms_sample(1, 1024, n, torch.tensor([sp]), g2)[0].double()
            pairs = {"sparse_pairs": (xs[:1024], xs2), "dense_pairs": (torch.rand(1024, n, generator=g2).double(),
                                                                      torch.rand(1024, n, generator=g2).double())}
            comp_split = None
            reps = lab[rep]
            if rep.sum() >= 2 and len(reps.unique()) >= 2:
                big = torch.bincount(reps).argmax()
                mask = (lab == big) & rep
                Pm = torch.diag(mask.double())
                comp_split = split_report(mlp, Pm, Pm, pairs, scale) | {"size": int(mask.sum())}
                rm = random_masks(n, int(mask.sum()), 1, g2)[0]
                Pr = torch.diag(rm.double())
                comp_split["random_same_size"] = split_report(mlp, Pr, Pr, pairs, scale)
            # Pi-graph additive blocks of the readout reads (rows of W^T W, rank m); truth = hidden-geometry component
            truth = {"hidden_component": (lab == torch.bincount(lab[rep]).argmax()) & rep} if rep.sum() >= 2 else {}
            pi = strip_masks(pi_report(mlp, xs[:4096], g2, truth))
            # task-weighted observability of the hidden space
            G = Wi @ torch.diag(imp.double() * (1 - sp) / 3) @ Wi.T
            ev, V = torch.linalg.eigh(G)
            fcos = (Wn.T @ V).abs().max(1).values
            # relu active set vs true support
            act = xh > 1e-9
            sup = xs > 0
            tp = (act & sup).sum()
            leak = float((xh * ~sup).pow(2).sum() / xh.pow(2).sum().clamp_min(1e-300))
            inst = {"sparsity": sp, "seed": i % args.seeds, "loss": loss,
                    "late_train_loss_mean": float(late[:, i].mean()),
                    "late_train_loss_rel_drift": float((late[:500, i].mean() - late[500:, i].mean()) / late[:, i].mean()),
                    "represented": int(rep.sum()), "feature_dim": [round(float(v), 4) for v in dim_i],
                    "norms": [round(float(v), 4) for v in norms],
                    "unit_graph": graph, "hidden_feature_graph": hgraph, "component_split": comp_split, "pi": pi,
                    "observability": {"gram_eigs": ev.flip(0).tolist(),
                                      "feature_max_cos_to_eigvec_median_represented":
                                          float(fcos[rep].median()) if rep.any() else None,
                                      "feature_max_cos_to_eigvec_min_represented":
                                          float(fcos[rep].min()) if rep.any() else None},
                    "relu_split": {"precision": float(tp / act.sum().clamp_min(1)), "recall": float(tp / sup.sum().clamp_min(1)),
                                   "exact_support_frac": float((act == sup).all(1).double().mean()),
                                   "offsupport_output_energy": leak}}
            cfg["instances"].append(inst)
        # convergence: a seed converged if its loss is within 5% of the best seed at the same sparsity
        for sp in args.sparsities_b:
            grp = [z for z in cfg["instances"] if z["sparsity"] == sp]
            best = min(z["loss"] for z in grp)
            for z in grp:
                z["converged_to_best_basin"] = z["loss"] <= 1.05 * best + 1e-9
            z0 = min(grp, key=lambda z: z["loss"])
            cs = z0["component_split"]
            print("B n=%d m=%d S=%.3f loss=%.2e rep=%d conv=%d/%d dims=%s hgraph0.1=%s obs_cos_med=%s relu P/R=%.2f/%.2f "
                  "leak=%.2e split=%s" % (
                      n, m, sp, z0["loss"], z0["represented"], sum(z["converged_to_best_basin"] for z in grp), len(grp),
                      sorted({round(v, 2) for v, r in zip(z0["feature_dim"], z0["norms"]) if r > 0.5}),
                      z0["hidden_feature_graph"]["0.1"], z0["observability"]["feature_max_cos_to_eigvec_median_represented"],
                      z0["relu_split"]["precision"], z0["relu_split"]["recall"], z0["relu_split"]["offsupport_output_energy"],
                      None if cs is None else "eta/sc=%.2e sparse=%.2e dense=%.2e | rand eta/sc=%.2e sparse=%.2e" % (
                          cs["eta_over_scale"], cs["sparse_pairs"]["emp_over_dF_median"],
                          cs["dense_pairs"]["emp_over_dF_median"], cs["random_same_size"]["eta_over_scale"],
                          cs["random_same_size"]["sparse_pairs"]["emp_over_dF_median"])), flush=True)
            pz = z0["pi"]
            if "trivial" not in pz:
                ht = pz["truth"].get("hidden_component", {})
                print("  PI comps1e-3=%s comps1e-2=%s | hidden-comp truth size=%s E=%s rand=%s | fiedler half E=%.4f "
                      "repl=%.2e" % (pz["components"]["0.001"], pz["components"]["0.01"], ht.get("size"),
                                     None if "E" not in ht else round(ht["E"], 5),
                                     None if "E" not in ht else round(ht["random_mean"], 4), pz["fiedler_half"]["E"],
                                     pz["fiedler_half"]["replacement_rel_mse"]), flush=True)
        report["configs"].append(cfg)
    return report


# ----------------------------------------------------------------------------------------------------------- toy C

C_FEATURES, C_RESID, C_P = 100, 1000, 0.01


def c_embedding(args):
    egen = torch.Generator().manual_seed(args.seed + 55)
    E = torch.randn(C_FEATURES, C_RESID, generator=egen)
    return E / E.norm(dim=1, keepdim=True)


class CNet(torch.nn.Module):
    def __init__(self, E, h):
        super().__init__()
        self.E = E
        self.l1 = torch.nn.Linear(E.shape[1], h)
        self.l2 = torch.nn.Linear(h, E.shape[1])

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x @ self.E))) @ self.E.T


def run_c(args, gen):
    n, d, p = C_FEATURES, C_RESID, C_P
    E = c_embedding(args)
    Ed = E.double()

    def sample_x(batch, g=None, dense=False):
        x = 2 * torch.rand(batch, n, generator=g) - 1
        if dense:
            return x
        return x * (torch.rand(batch, n, generator=g) < p)

    tg = torch.Generator().manual_seed(args.seed + 9)
    xt = sample_x(8192, tg)
    test = (xt, torch.relu(xt))
    report = {"toy": "C_compressed_computation", "args": vars(args), "n_features": n, "d_resid": d, "p_active": p,
              "zero_predictor_loss": float((torch.relu(xt) ** 2).mean()), "runs": []}
    for h in args.widths_c:
        for seed in range(args.seeds):
            t0 = time.time()
            torch.manual_seed(7000 + 31 * h + seed)
            net = CNet(E, h)

            def sample():
                x = sample_x(args.batch)
                return x, torch.relu(x)

            def analyse(net):
                A = net.l1.weight.detach().double() @ Ed.T
                U = Ed @ net.l2.weight.detach().double()
                mlp = ReluMlp(A, net.l1.bias.detach().double(), U, Ed @ net.l2.bias.detach().double())
                scale = scale_of(mlp)
                xs = sample_x(4096, gen).double()
                loss = float(((mlp.direct(xs) - torch.relu(xs)) ** 2).mean())
                # single-feature responses: neuron j's contribution to output i on x = t e_i, t in [-1, 1]
                t = torch.linspace(-1, 1, 41, dtype=F64)
                pre = t[:, None, None] * A.T[None] + mlp.b_in[None, None]  # (t, i, j)
                contrib = torch.relu(pre) * U[None]  # U[i, j]
                en = contrib.pow(2).sum(0)  # (i, j)
                share = en / en.sum(1, keepdim=True).clamp_min(1e-300)
                srt = share.sort(1, descending=True).values
                k90 = ((srt.cumsum(1) < 0.9).sum(1) + 1).double()
                in90 = torch.zeros_like(share, dtype=torch.bool)
                idx = share.argsort(1, descending=True)
                for i in range(n):
                    in90[i, idx[i, : int(k90[i])]] = True
                feats_per_neuron = in90.sum(0).double()
                bip = {}
                from scipy.sparse import bmat, csr_matrix
                from scipy.sparse.csgraph import connected_components
                for tau in (0.01, 0.05, 0.1, 0.2):
                    B = csr_matrix((share > tau).numpy().astype(np.int8))
                    nc, lab = connected_components(bmat([[None, B], [B.T, None]]), directed=False)
                    labf = lab[:n]
                    bip[str(tau)] = {"components_containing_features": int(len(np.unique(labf))),
                                     "largest_feature_count": int(np.bincount(labf).max())}
                pairs = {"sparse_pairs": (sample_x(1024, gen).double(), sample_x(1024, gen).double()),
                         "dense_pairs": (sample_x(1024, gen, True).double(), sample_x(1024, gen, True).double())}
                half = torch.zeros(n, dtype=torch.bool)
                half[torch.randperm(n, generator=gen)[: n // 2]] = True
                Pt = torch.diag(half.double())
                truth = split_report(mlp, Pt, Pt, pairs, scale) | {"unit_mixing": unit_mixing(mlp, Pt, Pt)}
                a, u = mlp.W_in, mlp.W_out.T
                Ca, Cu = cos_matrix(a), cos_matrix(u)
                W = (Ca * Ca + Cu * Cu) * ~torch.eye(h, dtype=torch.bool)
                spec = spectral_labels(W, 2, 0) == 0
                if spec.all() or not spec.any():
                    spec = random_masks(h, h // 2, 1, gen)[0]
                # rank pinned to the group's share of units (as in the Pythia probe) so neither side is trivially empty
                kk = min(max(round(n * int(spec.sum()) / h), 1), n - 1)
                m2, Pf, Qf = refine(mlp, spec, kk, 15)
                Pd = torch.diagonal(Pf)
                tool = split_report(mlp, Pf, Qf, pairs, scale) | {
                    "size": int(m2.sum()), "rank": kk,
                    "P_offdiag_frac": float((Pf - torch.diag(Pd)).norm() / Pf.norm())}
                nulls = [split_report(mlp, *split_projectors(mlp, mk, kk)[:2], pairs, scale)
                         for mk in random_masks(h, int(m2.sum()), 2, gen)]
                # units that never fire on sparse or dense data are invisible to every executed test but not to eta
                probe_x = torch.cat([xs, 2 * torch.rand(4096, n, generator=gen, dtype=F64) - 1])
                live = (torch.relu(probe_x @ A.T + mlp.b_in) > 0).any(0)
                live_mlp = ReluMlp(A[live], mlp.b_in[live], U[:, live], mlp.b_out)
                truth["live_units"] = int(live.sum())
                truth["eta_over_scale_live_units_only"] = eta(live_mlp, Pt, Pt)["eta"] / scale_of(live_mlp)
                # Pi-graph additive blocks in feature coordinates (trivial when h <= 100 = read rank)
                pi = strip_masks(pi_report(mlp, xs, gen, {"feature_half_read_dominant":
                                                          A[:, half].pow(2).sum(1) >= A[:, ~half].pow(2).sum(1)}))
                return {"loss": loss, "scale": scale, "pi": pi,
                        "neurons_per_feature_90": q(k90), "features_per_neuron_90": q(feats_per_neuron),
                        "neurons_serving_ge2_features": int((feats_per_neuron >= 2).sum()),
                        "dead_neurons": int((en.sum(0) == 0).sum()),
                        "active_neurons_per_sparse_input": q((torch.relu(xs @ A.T + mlp.b_in) > 0).double().sum(1)),
                        "unit_graph": graph_components(mlp), "neuron_feature_bipartite": bip,
                        "truth_split_feature_halves": truth, "tool_split": tool,
                        "random_unit_split_null": {"eta_over_scale": float(np.mean([z["eta_over_scale"] for z in nulls])),
                                                   "sparse_emp_over_dF_median": float(np.mean(
                                                       [z["sparse_pairs"]["emp_over_dF_median"] for z in nulls]))}}

            at_init = analyse(net)
            curve = train_mlp(net, sample, args.steps_c, args.lr_c, 0.0, test)
            res = analyse(net)
            res |= {"width": h, "seed": seed, "loss_curve": curve, "at_init": at_init, "seconds": time.time() - t0,
                    "converged_plateau": abs(curve[-3]["test_loss"] - curve[-1]["test_loss"])
                    <= 0.02 * curve[-1]["test_loss"] + 1e-12}
            report["runs"].append(res)
            tr = res["truth_split_feature_halves"]
            print("C h=%d s%d loss=%.2e (zero %.2e) k90/feat med=%.0f feats/neuron med=%.0f max=%.0f | truth eta/sc=%.3f "
                  "(init %.3f) live=%d eta_live=%.3f mix=%.3f sparse=%.3f dense=%.3f | tool size=%d eta/sc=%.3f "
                  "sparse=%.3f offdiag=%.2f rand eta/sc=%.3f | "
                  "bip0.05=%s (%.0fs)" % (
                      h, seed, res["loss"], report["zero_predictor_loss"], res["neurons_per_feature_90"]["median"],
                      res["features_per_neuron_90"]["median"], res["features_per_neuron_90"]["max"],
                      tr["eta_over_scale"], at_init["truth_split_feature_halves"]["eta_over_scale"],
                      tr["live_units"], tr["eta_over_scale_live_units_only"],
                      tr["unit_mixing"]["weighted_mean"], tr["sparse_pairs"]["emp_over_dF_median"],
                      tr["dense_pairs"]["emp_over_dF_median"], res["tool_split"]["size"],
                      res["tool_split"]["eta_over_scale"],
                      res["tool_split"]["sparse_pairs"]["emp_over_dF_median"], res["tool_split"]["P_offdiag_frac"],
                      res["random_unit_split_null"]["eta_over_scale"],
                      res["neuron_feature_bipartite"]["0.05"], res["seconds"]), flush=True)
            pz = res["pi"]
            if "trivial" in pz:
                print("  PI trivial: %s" % pz["trivial"], flush=True)
            else:
                ht = pz["truth"]["feature_half_read_dominant"]
                print("  PI comps1e-2=%s | feature-half truth E=%.4f (init %s) rand=%.4f twin=%.4f repl=%.2e | fiedler half "
                      "E=%.4f repl=%.2e" % (pz["components"]["0.01"], ht["E"],
                                            res["at_init"]["pi"].get("truth", {}).get("feature_half_read_dominant", {}).get("E"),
                                            ht["random_mean"], ht["twin_random_mean"], ht["replacement_rel_mse"],
                                            pz["fiedler_half"]["E"], pz["fiedler_half"]["replacement_rel_mse"]), flush=True)
    return report


# ------------------------------------------------------------------ E* local-search proposer and matched nulls
#
# For unit subset S with M_S = U_S^T U_S (whitened reads, W = U T) and eigenvalues lam, the closest exactly separated
# reads with a rank-p read projector have whitened error E_p(S) = sum_{top p} (1 - lam) + sum_{rest} lam; E_p = E* =
# sum min(lam, 1 - lam) (pi_split) whenever exactly p eigenvalues exceed 1/2, and E* = min_p E_p. The proposer holds
# p fixed and descends E_p directly: flipping unit i changes M by +-u_i u_i^T, the first-order change of E_p is
# +-sum_k s_k (v_k^T u_i)^2 with s_k = -1 on the top p eigenvectors and +1 elsewhere; see local_search for why every
# accepted step is a strict exact decrease.

PROPOSER_KICKS = 0
PROPOSER_RANDOM_STARTS = 16
PROPOSER_SUBSPACE_STARTS = 64
PROPOSER_POLISH = 0


def e_star_eig(M):
    lam, V = eigh(0.5 * (M + M.T))
    lc = lam.clamp(0, 1)
    return float(torch.minimum(lc, 1 - lc).sum()), int((lam > 0.5).sum()), lam, V


def e_p(lam, p):
    r = lam.numel()  # eigh is ascending: the top p are the last p
    return float(lam[: r - p].sum() + (1 - lam[r - p:]).sum())


def start_at_rank(U, order, p):
    """Prefix of `order` in the middle of the prefix range whose count #(lam > 1/2) is exactly p (monotone in the
    prefix, moving by at most one per unit by interlacing), so starts are balanced for the rank p."""
    n = len(order)

    def count(k):
        return e_star_eig(U[order[:k]].T @ U[order[:k]])[1]
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) >= p:
            hi = mid
        else:
            lo = mid + 1
    k_lo = lo
    lo, hi = k_lo, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= p:
            lo = mid
        else:
            hi = mid - 1
    mask = torch.zeros(n, dtype=torch.bool)
    mask[order[:(k_lo + lo) // 2]] = True
    return mask


def local_search(U, mask, p, max_single=PROPOSER_POLISH, max_iter=100000):
    """E_p = tr M - 2 (sum of the top p eigenvalues) + p is concave in M (Ky Fan), so the exact change of any flip set
    is at most its first-order change: flipping every unit with a negative first-order score strictly decreases E_p.
    Repeat; when no score is negative, evaluate the max_single lowest-scoring single flips exactly (a positive
    first-order score can still hide an exact decrease) and stop when none decreases E_p."""
    mask = mask.clone()
    r = U.shape[1]
    sgn = torch.ones(r, dtype=U.dtype)
    sgn[r - p:] = -1.0
    lam, V = eigh(U[mask].T @ U[mask])
    E = e_p(lam, p)
    moves = 0
    for _ in range(max_iter):
        g = ((U @ V) ** 2) @ sgn  # first-order change of E_p when unit i is added
        score = torch.where(mask, -g, g)
        neg = score < 0
        if neg.any():
            trials = [neg.nonzero().flatten()]
        else:
            trials = [j.view(1) for j in torch.argsort(score)[:max_single]]
        ok = None
        for ix in trials:
            m2 = mask.clone()
            m2[ix] = ~m2[ix]
            lam2, V2 = eigh(U[m2].T @ U[m2])
            E2 = e_p(lam2, p)
            if E2 < E - 1e-13 * max(E, 1.0):
                ok = (m2, E2, lam2, V2, len(ix))
                break
        if ok is None:
            break
        mask, E, lam, V, k = ok
        moves += k
    return mask, E, int((lam > 0.5).sum()), moves


def laplacian_vectors(U, n_vec):
    """Fiedler vectors of the Pi^2 Laplacian by dense eigh (the probe's Lanczos pi_fiedler does not always converge)."""
    Pi2 = (U @ U.T).pow(2)
    return eigh(torch.diag(Pi2.sum(1)) - Pi2)[1][:, 1:n_vec + 1]


def propose(U, p_list, gen, n_vec=3, n_rand=PROPOSER_RANDOM_STARTS, n_subspace=PROPOSER_SUBSPACE_STARTS,
            kicks=PROPOSER_KICKS):
    """Best E_p/r at each split rank p over starts: Fiedler orderings (both directions), n_rand random orderings and
    n_subspace random rank-p subspaces, then `kicks` rounds of iterated local search (flip a random 10% of units of the
    best split, descend again, keep only a strict improvement). Calibration on toy A width 256: recovery of the
    truth's local optimum is limited by starts (subspace starts about twice as efficient as random orderings), not by
    kicks or single-flip polish, which changed nothing, so both default to 0."""
    n, r = U.shape
    vecs = laplacian_vectors(U, min(n_vec, n - 2))
    orders = []
    for v in range(vecs.shape[1]):
        o = vecs[:, v].argsort()
        orders += [("fiedler%d_up" % (v + 1), o), ("fiedler%d_down" % (v + 1), o.flip(0))]
    orders += [("random%d" % j, torch.randperm(n, generator=gen)) for j in range(n_rand)]
    best = {}
    for p in p_list:
        starts = [(name, start_at_rank(U, o, p)) for name, o in orders]
        for j in range(n_subspace):  # K-subspaces style: units by their read share in a random rank-p subspace
            Qr = qr(torch.randn(r, p, generator=gen, dtype=U.dtype))[0]
            share = (U @ Qr).pow(2).sum(1) / U.pow(2).sum(1).clamp_min(1e-300)
            starts.append(("subspace%d" % j, share > p / r))
        for name, m0 in starts:
            E0 = e_p(eigh(U[m0].T @ U[m0])[0], p)
            m, E, p_half, moves = local_search(U, m0, p)
            if p not in best or E < best[p]["E"]:
                best[p] = {"E": E, "E_over_r": E / r, "start": name, "start_E_over_r": E0 / r, "moves": moves,
                           "size": int(m.sum()), "count_lam_gt_half": p_half,
                           "E_star_over_r": e_star_eig(U[m].T @ U[m])[0] / r, "mask": m}
        if p not in best:
            continue
        best[p]["kicks_accepted"] = 0
        for _ in range(kicks):
            m0 = best[p]["mask"].clone()
            flip = torch.randperm(n, generator=gen)[: max(1, n // 10)]
            m0[flip] = ~m0[flip]
            m, E, p_half, moves = local_search(U, m0, p)
            if E < best[p]["E"] - 1e-13 * max(best[p]["E"], 1.0):
                best[p] |= {"E": E, "E_over_r": E / r, "size": int(m.sum()), "mask": m, "count_lam_gt_half": p_half,
                            "E_star_over_r": e_star_eig(U[m].T @ U[m])[0] / r,
                            "kicks_accepted": best[p]["kicks_accepted"] + 1}
    return best


def leverage_null(U, gen, iters=300, tol=1e-8):
    """Orthonormal-column U0 with the same row leverages |u_i|^2 as U and isotropic directions (alternating row
    rescaling and the polar factor G (G^T G)^{-1/2}, via an r x r eigh). Every unit subset keeps its leverage profile,
    so concentration of the whitened read norms, which alone sets how easy a split is, is matched exactly while
    directional structure is not."""
    h = U.pow(2).sum(1)
    G = torch.randn(U.shape, generator=gen, dtype=F64)
    for _ in range(iters):
        G = G * (h.sqrt() / G.norm(dim=1).clamp_min(1e-300))[:, None]
        ev, V = eigh(G.T @ G)
        G = G @ (V * ev.clamp_min(1e-300).rsqrt()) @ V.T
        if float((G.pow(2).sum(1) - h).abs().max() / h.max()) < tol:
            break
    return G, float((G.pow(2).sum(1) - h).abs().max() / h.max())


def isotropic_twin_U(W, gen):
    """Read norms kept, directions isotropic within the read row space (so a rank-deficient W keeps its rank)."""
    _, sv, Vt = svd(W)
    Vr = Vt[: int((sv > 1e-10 * sv[0]).sum())]
    G = torch.randn(W.shape[0], Vr.shape[0], generator=gen, dtype=F64) @ Vr
    return read_basis(G / G.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True))[0]


def agreement(m, t, w=None):
    w = torch.ones(len(m), dtype=F64) if w is None else w
    a = float((w * (m == t)).sum() / w.sum())
    return max(a, 1 - a)


def proposer_report(mlp, gen, truth=None, X=None, null_draws=2, p_extra=(), minority=None):
    """Proposer on the network and on null_draws of each null at the same split ranks. minority: per-unit share of
    the read energy in the unit's non-dominant true module (a unit near 1/2 has no well-defined true module)."""
    W = mlp.W_in
    n = W.shape[0]
    U, T = read_basis(W)
    r = U.shape[1]
    out = {"units": n, "read_rank": r}
    if r >= n:
        out["trivial"] = "units <= read rank: Pi = I, every unit is an exact additive block"
        return out
    lev = U.pow(2).sum(1)
    out["leverage_participation"] = float(lev.sum() ** 2 / lev.pow(2).sum())
    p_list = sorted({max(1, round(r / 2))} | set(p_extra))
    tp = None
    if truth is not None and 0 < truth.sum() < n:
        tE, tp, _, _ = e_star_eig(U[truth].T @ U[truth])  # at its own rank tp, E_tp = E*
        p_list = sorted(set(p_list) | {tp}) if 0 < tp < r else p_list
        mt, Et, _, _ = local_search(U, truth, tp)  # the local optimum the truth descends to: the recoverable target
        out["truth"] = {"E_over_r": tE / r, "rank": tp, "size": int(truth.sum()),
                        "descent_E_over_r": Et / r, "descent_agreement": agreement(mt, truth)}
        if X is not None:
            out["truth"]["replacement_rel_mse"] = pi_replace(mlp, U, T, truth, X)[0]
    t0 = time.time()
    best = propose(U, p_list, gen)
    out["proposer_seconds"] = time.time() - t0
    nulls = {"isotropic_twin": [], "leverage_matched": []}
    for _ in range(null_draws):
        nulls["isotropic_twin"].append(propose(isotropic_twin_U(W, gen), p_list, gen))
        U0, dev = leverage_null(U, gen)
        out["leverage_null_max_rel_dev"] = dev
        nulls["leverage_matched"].append(propose(U0, p_list, gen))
    out["by_rank"] = []
    for p in p_list:
        if p not in best:
            continue
        b = best[p]
        row = dict(b) | {"p": p}  # mask stripped by the caller
        for name, draws in nulls.items():
            vals = [d[p]["E_over_r"] for d in draws if p in d]
            row[name + "_best_E_over_r"] = vals
            row["ratio_vs_" + name] = b["E_over_r"] / min(vals) if vals and min(vals) > 0 else None
        if X is not None:
            row["replacement_rel_mse"] = pi_replace(mlp, U, T, b["mask"], X)[0]
        if truth is not None and p == tp:
            row["truth_agreement"] = agreement(b["mask"], truth)
            row["truth_agreement_leverage_weighted"] = agreement(b["mask"], truth, lev)
            row["E_over_truth_E"] = b["E_over_r"] / out["truth"]["E_over_r"] if out["truth"]["E_over_r"] > 0 else None
            row["agreement_with_truth_descent"] = agreement(b["mask"], mt)
            dE = out["truth"]["descent_E_over_r"]
            row["E_over_truth_descent_E"] = b["E_over_r"] / dE if dE else None
            if minority is not None:
                same = (b["mask"] == truth) if (b["mask"] == truth).double().mean() >= 0.5 else (b["mask"] != truth)
                pure = minority < 0.2
                row["pure_units_minority_lt_0.2"] = int(pure.sum())
                row["truth_agreement_pure_units"] = float(same[pure].double().mean()) if pure.any() else None
                row["disagreeing_minority_median"] = float(minority[~same].median()) if (~same).any() else None
        out["by_rank"].append(row)
    return out


def fmt_prop(tag, rep):
    if "trivial" in rep:
        print("%s trivial" % tag, flush=True)
        return
    for row in rep["by_rank"]:
        tr = rep.get("truth", {})
        print("%s p=%d E/r=%.5f (start %s %.4f, %d moves) twin=%s lev=%s ratios %.2f/%.2f | truth E/r=%s agree=%s "
              "lev-agree=%s pure-agree=%s repl=%s" % (
                  tag, row["p"], row["E_over_r"], row["start"], row["start_E_over_r"], row["moves"],
                  [round(v, 5) for v in row["isotropic_twin_best_E_over_r"]],
                  [round(v, 5) for v in row["leverage_matched_best_E_over_r"]],
                  row["ratio_vs_isotropic_twin"] or float("nan"), row["ratio_vs_leverage_matched"] or float("nan"),
                  None if "E_over_r" not in tr or tr["rank"] != row["p"] else round(tr["E_over_r"], 5),
                  None if "truth_agreement" not in row else round(row["truth_agreement"], 3),
                  None if "truth_agreement" not in row else round(row["truth_agreement_leverage_weighted"], 3),
                  None if row.get("truth_agreement_pure_units") is None else round(row["truth_agreement_pure_units"], 3),
                  None if "replacement_rel_mse" not in row else "%.2e" % row["replacement_rel_mse"]), flush=True)


def cached(path, build):
    path = Path(path).expanduser()
    if path.exists():
        return torch.load(path)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return obj


def proposer_toys(args):
    wdir = Path(args.weights_dir).expanduser()
    gen = torch.Generator().manual_seed(args.seed + 1234)
    report = {"mode": "proposer_toys", "args": vars(args), "A": [], "B": [], "C": []}
    if "A" in args.toys:
        teacher, P, Qp = make_task_a(args.seed + 101)
        d, Pf, Qf = 8, P.float(), Qp.float()
        tgen = torch.Generator().manual_seed(args.seed + 7)
        Xtest = torch.randn(4096, d, generator=tgen).float()
        test = (Xtest, teacher.direct(Xtest.double()).float())
        Xt = torch.randn(4096, d, generator=tgen, dtype=F64)

        def sample():
            Xb = torch.randn(args.batch, d)
            return Xb, teacher.direct(Xb.double()).float()

        for width in args.widths_a:
            for init, optimizer, wd in a_configs(args):
                for seed in range(args.seeds):
                    def build():
                        net = student_a(d, width, init, seed, Pf, Qf)
                        lr = args.lr_a if optimizer == "adamw" else args.lr_sgd_a
                        train_mlp(net, sample, args.steps_a, lr, wd, test, optimizer=optimizer)
                        return net.state_dict()
                    net = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Linear(width, d))
                    net.load_state_dict(cached(wdir / ("A_h%d_%s_%s_wd%g_s%d.pt" % (width, init, optimizer, wd, seed)),
                                               build))
                    mlp = student_to_mlp(net)
                    Yt = teacher.direct(Xt)
                    rel = float(((mlp.direct(Xt) - Yt) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum())
                    a = mlp.W_in
                    ra, rb = (a @ P).pow(2).sum(1), (a @ (torch.eye(d, dtype=F64) - P)).pow(2).sum(1)
                    truth = ra >= rb
                    rep = proposer_report(mlp, gen, truth, Xt, args.null_draws, minority=torch.minimum(ra, rb) / (ra + rb))
                    rep |= {"width": width, "init": init, "optimizer": optimizer, "weight_decay": wd, "seed": seed,
                            "task_rel_mse": rel}
                    report["A"].append(strip_masks(rep))
                    fmt_prop("A h=%d %s %s wd=%g s%d" % (width, init, optimizer, wd, seed), rep)
    if "B" in args.toys:
        for n, m, imp_base in ((5, 2, 1.0), (20, 5, 0.9)):
            W, b = cached(wdir / ("B_n%d_m%d.pt" % (n, m)), lambda: train_tms(n, m, imp_base, args)[:2])
            g2 = torch.Generator().manual_seed(args.seed + 3)
            for i in range(W.shape[0]):
                Wi, bi = W[i].detach().double(), b[i].detach().double()
                sp = args.sparsities_b[i // args.seeds]
                mlp = ReluMlp(Wi.T @ Wi, bi, torch.eye(n, dtype=F64), torch.zeros(n, dtype=F64))
                xs = tms_sample(1, 4096, n, torch.tensor([sp]), g2)[0].double()
                rep_ = Wi.norm(dim=0) > 0.5
                Ch = cos_matrix(Wi.T).abs()
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
                adj = ((Ch > 0.1) & ~torch.eye(n, dtype=torch.bool) & rep_[:, None] & rep_[None, :]).numpy()
                _, lab = connected_components(csr_matrix(adj.astype(np.int8)), directed=False)
                lab = torch.tensor(lab)
                truth = (lab == torch.bincount(lab[rep_]).argmax()) & rep_ if rep_.sum() >= 2 else None
                rep = proposer_report(mlp, gen, truth, xs, args.null_draws)
                rep |= {"n": n, "m": m, "sparsity": sp, "seed": i % args.seeds,
                        "hidden_components_tau0.1": int(len(lab[rep_].unique())) if rep_.any() else 0}
                report["B"].append(strip_masks(rep))
                fmt_prop("B n=%d m=%d S=%.2f s%d" % (n, m, sp, i % args.seeds), rep)
    if "C" in args.toys:
        E = c_embedding(args)
        Ed = E.double()
        for h in args.widths_c:
            for seed in range(args.seeds):
                def build():
                    torch.manual_seed(7000 + 31 * h + seed)
                    net = CNet(E, h)

                    def sample():
                        x = 2 * torch.rand(args.batch, C_FEATURES) - 1
                        x = x * (torch.rand(args.batch, C_FEATURES) < C_P)
                        return x, torch.relu(x)
                    tg = torch.Generator().manual_seed(args.seed + 9)
                    xt = (2 * torch.rand(8192, C_FEATURES, generator=tg) - 1) * (
                        torch.rand(8192, C_FEATURES, generator=tg) < C_P)
                    train_mlp(net, sample, args.steps_c, args.lr_c, 0.0, (xt, torch.relu(xt)))
                    return {k: v for k, v in net.state_dict().items()}
                net = CNet(E, h)
                net.load_state_dict(cached(wdir / ("C_h%d_s%d.pt" % (h, seed)), build))
                A = net.l1.weight.detach().double() @ Ed.T
                U_ = Ed @ net.l2.weight.detach().double()
                mlp = ReluMlp(A, net.l1.bias.detach().double(), U_, Ed @ net.l2.bias.detach().double())
                half = torch.zeros(C_FEATURES, dtype=torch.bool)
                half[torch.randperm(C_FEATURES, generator=gen)[: C_FEATURES // 2]] = True
                ra, rb = A[:, half].pow(2).sum(1), A[:, ~half].pow(2).sum(1)
                truth = ra >= rb
                xs = (2 * torch.rand(4096, C_FEATURES, generator=gen, dtype=F64) - 1) * (
                    torch.rand(4096, C_FEATURES, generator=gen, dtype=F64) < C_P)
                rep = proposer_report(mlp, gen, truth, xs, args.null_draws, minority=torch.minimum(ra, rb) / (ra + rb))
                rep |= {"width": h, "seed": seed, "task_mse": float(((mlp.direct(xs) - torch.relu(xs)) ** 2).mean())}
                if "trivial" not in rep:  # every union of per-feature blocks is exact: score block consistency
                    dom = A.abs().argmax(1)
                    for row in rep["by_rank"]:
                        m = row["mask"]
                        cons = sum(max(int(m[dom == f].sum()), int((~m[dom == f]).sum())) for f in dom.unique().tolist())
                        row["feature_block_consistency"] = cons / len(m)
                        print("   C h=%d s%d p=%d feature-block consistency %.3f" % (h, seed, row["p"], cons / len(m)),
                              flush=True)
                report["C"].append(strip_masks(rep))
                fmt_prop("C h=%d s%d" % (h, seed), rep)
    return report


def proposer_pythia(args):
    from mpd_opfirst_gelu_modules_2951 import texts_tokens
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    gen = torch.Generator().manual_seed(args.seed + 4321)
    cfg = AutoConfig.from_pretrained(args.model)
    if cfg.hidden_act != "gelu":
        raise SystemExit("hidden_act=%r is not exact erf GELU; refusing" % cfg.hidden_act)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float64).eval()
    ids, source = texts_tokens(tok, 2, 256, args.seed)
    layers = model.gpt_neox.layers
    pick = [int(v) for v in args.layers.split(",")] if args.layers else list(range(len(layers)))
    caps = {}

    def hook(i):
        def f(mod, inp, out):
            caps[i] = inp[0].detach().reshape(-1, inp[0].shape[-1])
        return f

    hs = [layers[i].mlp.register_forward_hook(hook(i)) for i in pick]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    report = {"mode": "proposer_pythia", "model": args.model, "tokens": int(ids.numel()), "text_source": source,
              "args": vars(args), "env": env_record(), "layers": []}
    with torch.no_grad():
        for li in pick:
            m = layers[li].mlp
            mlp = Mlp(m.dense_h_to_4h.weight, m.dense_h_to_4h.bias, m.dense_4h_to_h.weight, m.dense_4h_to_h.bias)
            d = mlp.W_in.shape[1]
            t0 = time.time()
            rep = proposer_report(mlp, gen, None, caps[li], args.null_draws,
                                  p_extra=tuple(round(d * f) for f in args.p_fracs))
            rep |= {"layer": li, "seconds": time.time() - t0}
            report["layers"].append(strip_masks(rep))
            fmt_prop("%s L%d (%.0fs)" % (args.model.split("/")[-1], li, rep["seconds"]), rep)
            # a partial receipt after every layer: a killed run keeps what it finished
            dump("%s_proposer_%s.json" % (args.out_prefix, args.model.split("/")[-1]), report | {"complete": False})
    report["complete"] = True
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--widths-a", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--wd-a", type=float, default=0.1)
    parser.add_argument("--lr-sgd-a", type=float, default=0.05)
    parser.add_argument("--steps-a", type=int, default=20000)
    parser.add_argument("--lr-a", type=float, default=3e-3)
    parser.add_argument("--conv-a", type=float, default=1e-2, help="converged: test rel MSE below this")
    parser.add_argument("--sparsities-b", type=float, nargs="+", default=[0.0, 0.7, 0.9, 0.97, 0.99])
    parser.add_argument("--steps-b", type=int, default=20000)
    parser.add_argument("--widths-c", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--steps-c", type=int, default=10000)
    parser.add_argument("--lr-c", type=float, default=3e-3)
    parser.add_argument("--out-prefix", default="experiments/issue-2951/receipts/opfirst_toys_trained")
    parser.add_argument("--mode", choices=("analyse", "proposer_toys", "proposer_pythia"), default="analyse")
    parser.add_argument("--toys", default="ABC")
    parser.add_argument("--weights-dir", default="~/mpd-data/toys_trained/weights")
    parser.add_argument("--null-draws", type=int, default=2)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--layers", default="", help="proposer_pythia: comma list; empty = all")
    parser.add_argument("--p-fracs", type=float, nargs="*", default=[0.25], help="extra split ranks as fractions of d")
    parser.add_argument("--linalg", choices=("torch", "scipy"), default="scipy",
                        help="LAPACK for the Pi / E_p eigen-computations (scipy = OpenBLAS in ~/mpd-data/venv)")
    args = parser.parse_args()
    global LINALG
    LINALG = args.linalg
    torch.set_num_threads(6)
    if args.mode != "analyse":
        t0 = time.time()
        report = proposer_toys(args) if args.mode == "proposer_toys" else proposer_pythia(args)
        report["runtime_s"] = time.time() - t0
        report["numeric_owners"] = NUMERIC_OWNERS + "; E* local-search proposer and leverage-matched null: this file"
        report["env"] = env_record()
        suffix = "proposer" if args.mode == "proposer_toys" else "proposer_" + args.model.split("/")[-1]
        dump("%s_%s.json" % (args.out_prefix, suffix), report)
        print("%s done in %.0fs" % (args.mode, report["runtime_s"]), flush=True)
        return
    for toy, fn in (("A", run_a), ("B", run_b), ("C", run_c)):
        if args.toy not in (toy, "all"):
            continue
        t0 = time.time()
        gen = torch.Generator().manual_seed(args.seed + ord(toy))
        report = fn(args, gen)
        report["runtime_s"] = time.time() - t0
        report["numeric_owners"] = NUMERIC_OWNERS
        dump("%s_%s.json" % (args.out_prefix, toy), report)
        print("toy %s done in %.0fs" % (toy, report["runtime_s"]), flush=True)


if __name__ == "__main__":
    main()
