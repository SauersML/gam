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
    KAPPA, Mlp, components, cos_matrix, eta_bound, haar_projector, pi_proposals, pi_split, q, refine, replacement,
    spectral_labels, split_projectors)

F64 = torch.float64
TAUS = (0.05, 0.1, 0.2, 0.3, 0.5)
NUMERIC_OWNERS = (
    "python/torch analysis, no Rust owner: parameter_decomposition has no unit-graph module split or replacement "
    "certificate (checked state, spectral, gauge, secant, operators); the unit-graph split, certified eta, "
    "replacement, refine and nulls are imported unchanged from bench/mpd_opfirst_gelu_modules_2951.py; this file "
    "adds only the relu kappa = 1/2 rescaling, per-unit mixing, TMS/compressed-computation measurements and torch "
    "training (SPEC PyTorch exception). Pi-graph additive blocks: pi_split (E* = sum min(lam, 1-lam)) and "
    "pi_proposals (Fiedler vectors of the Pi^2 Laplacian) imported unchanged; no Rust owner exists for either "
    "(surface.rs exposes recover_plane_rotations and mlp_block_receipt only)")


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
    Us, sv, Vt = torch.linalg.svd(W, full_matrices=False)
    r = int((sv > 1e-10 * sv[0]).sum())
    return Us[:, :r], sv[:r, None] * Vt[:r]


def dense_proposals(U, T, n_vec, fracs):
    """pi_proposals with a dense eigh of the same Pi^2 Laplacian, for when its Lanczos solve does not converge."""
    Pi2 = (U @ U.T).pow(2)
    _, vec = torch.linalg.eigh(torch.diag(Pi2.sum(1)) - Pi2)
    n, r = U.shape
    out = []
    for v in range(1, n_vec + 1):
        order = vec[:, v].argsort()
        for f in fracs:
            k = max(1, round(f * n))
            for side, idx in (("low", order[:k]), ("high", order[-k:])):
                mask = torch.zeros(n, dtype=torch.bool)
                mask[idx] = True
                res, _ = pi_split(U, T, mask)
                out.append({"vec": v, "k": k, "side": side, "E_over_U2": res["E_star"] / r, "mask": mask})
    return out


LANCZOS_FALLBACKS = []


def pi_best(U, T, n_vec=3):
    """Best Fiedler proposal (lowest E*/rank) at each proposal size."""
    from scipy.sparse.linalg import ArpackNoConvergence
    n = U.shape[0]
    try:
        _, props = pi_proposals(U, T, min(n_vec, n - 2), PI_FRACS)
    except ArpackNoConvergence:
        LANCZOS_FALLBACKS.append(n)
        props = dense_proposals(U, T, min(n_vec, n - 2), PI_FRACS)
    best = {}
    for pr in props:
        if pr["k"] not in best or pr["E_over_U2"] < best[pr["k"]]["E_over_U2"]:
            best[pr["k"]] = pr
    return best


def pi_replace(mlp, U, T, mask, X):
    """Execute the exactly separated reads: rel. squared change of F on X (vs the variance of F)."""
    _, What = pi_split(U, T, mask)
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
    configs = [("default", "adamw", 0.0), ("default", "adamw", args.wd_a), ("default", "sgd", 0.0),
               ("modular", "adamw", 0.0), ("modular", "sgd", 0.0)]
    for width in args.widths_a:
        for init, optimizer, wd in configs:
            if True:
                for seed in range(args.seeds):
                    t0 = time.time()
                    torch.manual_seed(1000 * width + seed)
                    net = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Linear(width, d))
                    if init == "modular":
                        with torch.no_grad():
                            side = torch.arange(width) < width // 2
                            I = torch.eye(d)
                            net[0].weight.copy_(torch.where(side[:, None], net[0].weight @ Pf, net[0].weight @ (I - Pf)))
                            wo = net[2].weight.T
                            net[2].weight.copy_(torch.where(side[:, None], wo @ Qf, wo @ (I - Qf)).T)
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


def run_b(args, gen):
    report = {"toy": "B_superposition", "args": vars(args), "configs": []}
    for n, m, imp_base in ((5, 2, 1.0), (20, 5, 0.9)):
        sps = torch.tensor(args.sparsities_b)
        n_inst = len(sps) * args.seeds
        S = sps.repeat_interleave(args.seeds)
        imp = imp_base ** torch.arange(n, dtype=torch.float32)
        torch.manual_seed(args.seed + n)
        W = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(n_inst, m, n)))
        b = torch.nn.Parameter(torch.zeros(n_inst, n))
        opt = torch.optim.Adam([W, b], lr=1e-3)
        t0 = time.time()
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
        late = torch.stack(hist)
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

def run_c(args, gen):
    n, d, p = 100, 1000, 0.01
    egen = torch.Generator().manual_seed(args.seed + 55)
    E = torch.randn(n, d, generator=egen)
    E = E / E.norm(dim=1, keepdim=True)
    Ed = E.double()

    def sample_x(batch, g=None, dense=False):
        x = 2 * torch.rand(batch, n, generator=g) - 1
        if dense:
            return x
        return x * (torch.rand(batch, n, generator=g) < p)

    class Net(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.l1 = torch.nn.Linear(d, h)
            self.l2 = torch.nn.Linear(h, d)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x @ E))) @ E.T

    tg = torch.Generator().manual_seed(args.seed + 9)
    xt = sample_x(8192, tg)
    test = (xt, torch.relu(xt))
    report = {"toy": "C_compressed_computation", "args": vars(args), "n_features": n, "d_resid": d, "p_active": p,
              "zero_predictor_loss": float((torch.relu(xt) ** 2).mean()), "runs": []}
    for h in args.widths_c:
        for seed in range(args.seeds):
            t0 = time.time()
            torch.manual_seed(7000 + 31 * h + seed)
            net = Net(h)

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
    args = parser.parse_args()
    torch.set_num_threads(6)
    for toy, fn in (("A", run_a), ("B", run_b), ("C", run_c)):
        if args.toy not in (toy, "all"):
            continue
        t0 = time.time()
        gen = torch.Generator().manual_seed(args.seed + ord(toy))
        report = fn(args, gen)
        report["runtime_s"] = time.time() - t0
        report["numeric_owners"] = NUMERIC_OWNERS
        report["pi_fiedler_dense_fallbacks"] = {"count": len(LANCZOS_FALLBACKS), "unit_counts": sorted(set(LANCZOS_FALLBACKS)),
                                                "note": "pi_fiedler Lanczos (which=SA) did not converge; dense eigh of "
                                                        "the same Pi^2 Laplacian used instead"}
        LANCZOS_FALLBACKS.clear()
        dump("%s_%s.json" % (args.out_prefix, toy), report)
        print("toy %s done in %.0fs" % (toy, report["runtime_s"]), flush=True)


if __name__ == "__main__":
    main()
