"""#2951 probe: native-anchored conditional operators on a SwiGLU block.

Native block F(h) = D [s(G h) * (U h)], s = SiLU, with native units y_n(h) = d_n s(g_n) u_n (d_n = column n
of D). For a unit group c, y_c = sum_{n in c} y_n is an exact decomposition of F, and D_c <- rho_c D_c is a real
parameter edit. The explanation of a group is a conditional operator

    yhat_c = g_c(z_c) t_c(h),   t_c(h) = sum_{n in c} d_n u_n(h) = D_c U_c h   (native template, no fitted vectors)

with a scalar gate g_c fitted on a low-dimensional conditioning variable z_c. Exact per-token split, with
w = |t|^2, q = t'y / w, r = y - q t (r is orthogonal to t, so the first equality is an identity per token):

    |y - g t|^2 = |r|^2 + w (q - g)^2
    E|y - g(z) t|^2 = E|r|^2                  [operator inadequacy: no scalar on t reproduces y]
                    + E[w (q - g*(z))^2]      [missing conditioning: z does not determine q]
                    + E[w (g*(z) - g(z))^2]   [gate fit],     g*(z) = E[t'y | z] / E[w | z].

g* is not observable; it is estimated by the same REML smooth fitted in-sample on the held-out tokens, so the
conditioning term is that fit's weighted residual and the gate-fit term is the held-out error of the
train-fitted gate minus it (the remainder, which also absorbs the finite-sample cross term).

Control identity: for native gains rho (one per group) the edited block is F_rho = sum rho_c y_c, the edited
explanation is sum rho_c yhat_c, and E|F_rho - Fhat_rho|^2 = rho' K rho with K_cd = E[e_c' e_d],
e_c = y_c - yhat_c. With a_n = u_n (s(g_n) - g_c(n)) and M = D'D, K = P' (M o E[a a']) P (P the membership
matrix), so K is exact for any grouping, singletons included. F_rho is recomputed from the edited weights, not
from the decomposition.

Gate fits use gamfit's batched Gaussian REML (reml.gaussian_reml_fit_batched): a cubic B-spline on the
train-standardized z with interior knots at pooled quantiles (basis size = gamfit's formula default for s(x)), its exact second-derivative
roughness penalty, weights w, one smoothing parameter per group chosen by REML. The 2-D gate is the
sum f1(z1) + f2(z2) of two such smooths under their block penalty with one lambda. The unit-gate oracle regresses q on
[1, s(g_n) for n in c] with a REML ridge (unpenalized intercept) and is an upper bound on what conditioning on
the unit gates can reach; the q oracle (z = h, g = q exactly) removes the conditioning and gate terms and leaves
only operator inadequacy.

Groupings (clustered on the train tokens only, evaluated on held-out tokens):
  singleton            one unit per group (native-neuron baseline; q = s(g_n) exactly, so r = 0)
  gate_corr            Ward hierarchy on the z-scored unit gates s(g_n) over tokens (correlation geometry)
  gate_raw             Ward hierarchy on the raw unit gates (the r = 0 condition is equal gates within a group)
  random_<x>_s<seed>   random unit groups with the same size multiset as <x>, the null
The hierarchy is cut at every group count in --ks; the whole curve is reported, nothing is selected.

Multi-template mode (--mode multi, receipt opfirst_cond_operators_multi_*): templates are the group's own native
vectors d_n u_n(h), so yhat_c = sum_{n in c} shat_n d_n u_n has no operator term and every error is in the gates.
Per raw-gate Ward group (and its size-matched random null) it reports the singular spectrum of the centered
token x unit gate matrix s(g_c) against a per-column token-permutation null, and fits rank-r gate models from the
group's top-r principal gate coordinates z_c: linear (PCA reconstruction), per-unit additive REML splines, and a
sign-gated form g_n * bhat_n(z_c) whose r = group-size limit is the relu law.

Two stages, because the model runtime (transformers) and the gamfit build live in different environments:
  uv run --no-project --with torch --with numpy --with scipy --with transformers --with safetensors \
      --with huggingface_hub --with datasets python bench/mpd_opfirst_cond_operators_2951.py --stage capture
  <venv with gamfit>/bin/python bench/mpd_opfirst_cond_operators_2951.py --stage analyze
(the receipt was made with one 2-thread analyze process per layer, `--layers L --out part_L.json`, then
`--merge part_2.json part_8.json part_14.json --out <receipt>`). The capture stage runs the model once in float32 and stores each chosen layer's MLP input, MLP output and
weights; the analysis casts everything to float64.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = ("allenai/OLMo-2-0425-1B", "Qwen/Qwen3-0.6B-Base")


def compact_json(obj):
    """One top-level key per line (tracked-file line limit)."""
    return "{\n" + ",\n".join(json.dumps(k) + ": " + json.dumps(v, separators=(",", ":"))
                               for k, v in obj.items()) + "\n}\n"


def capture(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mpd_llm_chart_restriction_2951 import token_batches

    model = tok = None
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
            break
        except Exception as exc:  # the fallback model is recorded in the capture
            print(f"{name}: load failed ({type(exc).__name__}: {exc}); trying the next model", flush=True)
    if model is None:
        raise SystemExit("no model loaded")
    cfg = model.config
    L = cfg.num_hidden_layers
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [L // 8, L // 2, L - 2]
    ids = next(token_batches(tok, args.seq_len + 1, args.n_seq, args.seed, 0))
    blocks = model.model.layers
    got = {}

    def hook(i):
        def f(_m, inputs, output):
            got[i] = (inputs[0].detach().clone(), output.detach().clone())
        return f

    handles = [blocks[i].mlp.register_forward_hook(hook(i)) for i in layers]
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()
    out = {"model": np.array(name), "model_type": np.array(cfg.model_type), "layers": np.array(layers),
           "ids": ids.numpy(), "n_layers": np.array(L)}
    for i in layers:
        mlp = blocks[i].mlp
        out[f"x{i}"] = got[i][0].numpy()
        out[f"f{i}"] = got[i][1].numpy()
        out[f"wg{i}"] = mlp.gate_proj.weight.detach().numpy()
        out[f"wu{i}"] = mlp.up_proj.weight.detach().numpy()
        out[f"wd{i}"] = mlp.down_proj.weight.detach().numpy()
        post = getattr(blocks[i], "post_feedforward_layernorm", None)
        if post is not None:
            out[f"gff{i}"] = post.weight.detach().numpy()
            out[f"eps{i}"] = np.array(post.variance_epsilon)
    np.savez(args.cache, **out)
    print(f"captured {name} layers {layers} tokens {tuple(ids.shape)} -> {args.cache}", flush=True)


# ---------------------------------------------------------------- analysis


class Spline:
    """Cubic B-spline on a standardized scalar with interior knots at pooled quantiles (gamfit's default
    placement for s(x)); basis and exact roughness penalty from gamfit. Held-out values outside the knot range
    are clamped to it."""

    def __init__(self, n_default):
        from gamfit import basis

        self.basis, self.n_default = basis, n_default

    def fit_knots(self, pooled):
        probs = np.linspace(0.0, 1.0, self.n_default - 2)
        qs = np.quantile(pooled, probs)
        self.lo, self.hi = qs[0], qs[-1]
        self.knots = np.r_[[qs[0]] * 3, qs, [qs[-1]] * 3]
        self.S = self.basis.smoothness_penalty(self.knots)[0]
        self.p = self.S.shape[0]
        return self

    def design(self, v):
        return self.basis.bspline_basis(np.clip(v, self.lo, self.hi), self.knots)


def batched_reml(designs, ys, ws, S):
    """gamfit batched REML over a chunk of groups (same penalty, own lambda each): coefficients and edf."""
    from gamfit import reml

    off = np.cumsum([0] + [D.shape[0] for D in designs]).astype(np.uintp)
    try:
        out = reml.gaussian_reml_fit_batched(np.concatenate(designs), np.concatenate(ys)[:, None], off, S,
                                             weights=np.concatenate(ws))
        return [out["coefficients"][j, :, 0] for j in range(len(designs))], out["edf"]
    except ValueError:
        if len(designs) > 1:
            parts = [batched_reml([D], [y], [w], S) for D, y, w in zip(designs, ys, ws)]
            return [c[0] for c, _ in parts], np.concatenate([e for _, e in parts])
    # the design interpolates the response (REML dispersion undefined): weighted least squares is exact
    sw = np.sqrt(ws[0])
    coef = np.linalg.lstsq(designs[0] * sw[:, None], ys[0] * sw, rcond=None)[0]
    INTERPOLATED.append(1)
    return [coef], np.array([float(designs[0].shape[1])])


INTERPOLATED = []


def ridge_reml_numpy(X, y, w, Xh):
    """REML ridge with an unpenalized intercept (numpy stand-in, see the receipt's `unit_gate_oracle_fit`).

    Weighted y = 1 a + Z b + e, e ~ N(0, sigma^2 / w), b ~ N(0, sigma^2 / lambda I). Profile the intercept out
    in the whitened space, SVD the projected Z, and maximise the restricted likelihood over log lambda with a
    bounded scalar optimiser (no grid)."""
    from scipy.optimize import minimize_scalar

    sw = np.sqrt(w)
    one, Zw, yw = sw, X * sw[:, None], y * sw
    c = one / np.linalg.norm(one)
    Zp, yp = Zw - np.outer(c, c @ Zw), yw - c * (c @ yw)
    Uz, sv, Vt = np.linalg.svd(Zp, full_matrices=False)
    keep = sv > sv.max() * max(Zp.shape) * np.finfo(float).eps if sv.size else sv > 0
    Uz, sv, Vt = Uz[:, keep], sv[keep], Vt[keep]
    n_eff = X.shape[0] - 1
    uy = Uz.T @ yp
    yy = yp @ yp
    d2 = sv ** 2

    def neg_reml(rho):
        lam = math.exp(rho)
        shrink = lam / (d2 + lam)
        rss = yy - uy @ uy + (shrink * uy ** 2).sum()  # y'(I - H) y restricted to the intercept complement
        logdet = np.log(d2 / lam + 1.0).sum()
        return 0.5 * (n_eff * math.log(rss / n_eff) + logdet)

    scale = math.log(d2.mean()) if d2.size else 0.0
    rho = minimize_scalar(neg_reml, bounds=(scale - 30, scale + 30), method="bounded").x if d2.size else 0.0
    lam = math.exp(rho)
    b = Vt.T @ (sv / (d2 + lam) * uy) if d2.size else np.zeros(X.shape[1])
    a = (c @ (yw - Zw @ b)) / np.linalg.norm(one)
    edf = 1 + (d2 / (d2 + lam)).sum()
    return a + Xh @ b, float(edf)


def ward_labels(features, ks):
    from scipy.cluster.hierarchy import fcluster, linkage

    Z = linkage(features, method="ward")
    return {k: fcluster(Z, k, criterion="maxclust") - 1 for k in ks}


def size_matched_random(labels, gen):
    perm = gen.permutation(labels.size)
    out = np.empty_like(labels)
    out[perm] = labels
    return out


def fidelity(F, Fh):
    err = ((F - Fh) ** 2).sum()
    return {"rel_err": float(err / (F ** 2).sum()), "fve_centered": float(1 - err / ((F - F.mean(0)) ** 2).sum())}


def post_norm(F, gamma, eps):
    return gamma * F / torch.sqrt((F ** 2).mean(-1, keepdim=True) + eps)


class Layer:
    def __init__(self, X, Wg, Wu, Wd, split):
        self.Wd = Wd
        self.M = Wd.T @ Wd
        self.sp = {}
        for name, idx in split.items():
            x = X[idx]
            G, U = x @ Wg.T, x @ Wu.T
            S = torch.nn.functional.silu(G)
            self.sp[name] = {"G": G, "U": U, "S": S, "A": S * U, "F": (S * U) @ Wd.T,
                             "Frelu": (torch.relu(G) * U) @ Wd.T}

    def group_stats(self, name, labels, k):
        """Per group: w, t'y, |y|^2, z1 = mean pre-activation, z2 = mean gate, as float64 numpy arrays.

        All groups at once: with Mb = M masked to same-group pairs, w_c = sum_{n in c} u_n (U Mb)_n, and so on."""
        d = self.sp[name]
        P = torch.from_numpy(labels)
        Mb = self.M * (P[:, None] == P[None, :])
        UM, AM = d["U"] @ Mb, d["A"] @ Mb
        seg = lambda V: torch.zeros(V.shape[0], k, dtype=V.dtype).index_add_(1, P, V).numpy()  # noqa: E731
        size = np.bincount(labels, minlength=k)
        mats = {"w": seg(UM * d["U"]), "ty": seg(UM * d["A"]), "yy": seg(AM * d["A"]),
                "z1": seg(d["G"]) / size, "z2": seg(d["S"]) / size}
        return [{key: np.ascontiguousarray(m[:, c]) for key, m in mats.items()} for c in range(k)]


def gate_fits(n_default, tr, ho, two_d, max_entries=3e7):
    """Train-fitted gate evaluated on held-out, and the in-sample held-out fit in the same basis (the g*
    estimate). Each group's z is standardized by its train mean and sd; knots sit at quantiles of the pooled
    standardized train values of the grouping.

    2-D gate: additive f1(z1) + f2(z2) (second basis without its first column, which the partition of unity
    makes collinear with the first basis) under the block penalty diag(S1, S2[1:,1:]) with one lambda."""
    keys = ("z1", "z2") if two_d else ("z1",)
    std = [{k: (st[k].mean(), st[k].std()) for k in keys} for st in tr]
    zeta = lambda st, sd, k: (st[k] - sd[k][0]) / sd[k][1]  # noqa: E731
    spl = {k: Spline(n_default).fit_knots(np.concatenate([zeta(st, sd, k) for st, sd in zip(tr, std)]))
           for k in keys}
    if two_d:
        p1, p2 = spl["z1"].p, spl["z2"].p
        S = np.zeros((p1 + p2 - 1,) * 2)
        S[:p1, :p1], S[p1:, p1:] = spl["z1"].S, spl["z2"].S[1:, 1:]
    else:
        S = spl["z1"].S

    def designs(stats, sl):
        n = stats[sl[0]]["w"].size
        parts = [spl[k].design(np.concatenate([zeta(stats[i], std[i], k) for i in sl])) for k in keys]
        B = np.hstack([parts[0], parts[1][:, 1:]]) if two_d else parts[0]
        return [B[j * n:(j + 1) * n] for j in range(len(sl))]

    q = lambda st: st["ty"] / st["w"]  # noqa: E731
    per = max(1, int(max_entries // (tr[0]["w"].size * S.shape[0])))
    g, gstar, edf = [], [], np.zeros(len(tr))
    for s in range(0, len(tr), per):
        sl = list(range(s, min(s + per, len(tr))))
        c_tr, edf[s:s + len(sl)] = batched_reml(designs(tr, sl), [q(tr[i]) for i in sl],
                                                [tr[i]["w"] for i in sl], S)
        Dho = designs(ho, sl)
        c_ho, _ = batched_reml(Dho, [q(ho[i]) for i in sl], [ho[i]["w"] for i in sl], S)
        g += [D @ c for D, c in zip(Dho, c_tr)]
        gstar += [D @ c for D, c in zip(Dho, c_ho)]
    return g, gstar, edf, S.shape[0]


def evaluate(layer, labels, n_default, gen_rho, n_rho):
    k = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    groups = [torch.from_numpy(g) for g in np.split(order, np.cumsum(np.bincount(labels, minlength=k))[:-1])]
    tick = [time.time()]
    tr, ho = layer.group_stats("train", labels, k), layer.group_stats("held", labels, k)
    H = layer.sp["held"]
    F, N = H["F"], H["F"].shape[0]
    E_y = sum(st["yy"].sum() for st in ho)
    op = sum((st["yy"] - st["ty"] ** 2 / st["w"]).sum() for st in ho)
    res = {"k": k, "size_max": int(max(len(g) for g in groups)),
           "size_entropy_eff": float(np.exp(-sum((len(g) / labels.size) * np.log(len(g) / labels.size)
                                                  for g in groups))),
           "sum_group_energy_over_F": float(E_y / (F ** 2).sum().item())}

    def assemble(gates):
        return torch.from_numpy(np.stack([np.asarray(gv, dtype=np.float64) for gv in gates], 1))[:, labels]

    qs = [st["ty"] / st["w"] for st in ho]
    res["oracle_q"] = {"operator": float(op / E_y), **fidelity(F, (H["U"] * assemble(qs)) @ layer.Wd.T)}
    for tag, two_d in (("gate1d", False), ("gate2d", True)):
        tick.append(time.time())
        # a one-unit group has z2 = s(g_n) = q exactly: its 2-D gate is the native gate, with no fit
        fit = [i for i, idx in enumerate(groups) if not (two_d and idx.numel() == 1)]
        g, gstar, edf = [st["z2"] for st in ho], [st["z2"] for st in ho], np.zeros(k)
        n_par = 0
        if fit:
            gf, gsf, edf_f, p = gate_fits(n_default, [tr[i] for i in fit], [ho[i] for i in fit], two_d)
            for j, i in enumerate(fit):
                g[i], gstar[i], edf[i] = gf[j], gsf[j], edf_f[j]
            n_par = len(fit) * p
        tot = sum((st["w"] * (st["ty"] / st["w"] - gv) ** 2).sum() for st, gv in zip(ho, g))
        cond = sum((st["w"] * (st["ty"] / st["w"] - gv) ** 2).sum() for st, gv in zip(ho, gstar))
        Gt = assemble(g)
        Fh = (H["U"] * Gt) @ layer.Wd.T
        a = H["U"] * (H["S"] - Gt)
        block = {"operator": float(op / E_y), "conditioning": float(cond / E_y), "gate_fit": float((tot - cond) / E_y),
                 "sum_group_err": float((op + tot) / E_y), "gate_params": n_par, "gate_edf": float(edf.sum()),
                 **fidelity(F, Fh),
                 "cross_term_share": float(1 - ((F - Fh) ** 2).sum().item() / (op + tot))}
        if "gff" in layer.__dict__:
            block["post_norm_write"] = fidelity(post_norm(F, layer.gff, layer.eps), post_norm(Fh, layer.gff, layer.eps))
        if tag == "gate1d":
            block["edits"] = edits(layer, labels, k, a, Fh, Gt, gen_rho, n_rho)
        res[tag] = block
    # unit-gate oracle: q regressed on the group's own unit gates (REML ridge), bound on unit-level conditioning
    tick.append(time.time())
    Str, Sho = layer.sp["train"]["S"], H["S"]
    g_lin, edf_lin = [], 0.0
    for idx, st_tr, st_ho in zip(groups, tr, ho):
        if idx.numel() == 1:  # q = s(g_n) exactly: the unit gate is the oracle
            g_lin.append(Sho[:, idx[0]].numpy())
            continue
        gh, e = ridge_reml_numpy(Str[:, idx].numpy(), st_tr["ty"] / st_tr["w"], st_tr["w"], Sho[:, idx].numpy())
        g_lin.append(gh)
        edf_lin += e
    tot_lin = sum((st["w"] * (st["ty"] / st["w"] - gv) ** 2).sum() for st, gv in zip(ho, g_lin))
    res["oracle_unit_gates"] = {"operator": float(op / E_y), "cond_plus_fit": float(tot_lin / E_y), "edf": edf_lin,
                                **fidelity(F, (H["U"] * assemble(g_lin)) @ layer.Wd.T)}
    tick.append(time.time())
    res["seconds"] = dict(zip(("stats", "gate1d", "gate2d", "unit_oracle"), np.diff(tick).round(1).tolist()))
    return res


def edits(layer, labels, k, a, Fh, Gt, gen, n_rho):
    """Native gain edits: actual edited block from edited weights vs sum rho_c yhat_c; rho'K rho vs MSE."""
    H = layer.sp["held"]
    N = a.shape[0]
    EF2 = (H["F"] ** 2).sum(1).mean().item()
    C = (a.T @ a) / N
    P = torch.from_numpy(labels)
    KC = layer.M * C
    K = torch.zeros(k, KC.shape[1], dtype=KC.dtype).index_add_(0, P, KC)
    K = torch.zeros(k, k, dtype=KC.dtype).index_add_(1, P, K)
    rhos = {"uniform_0.5_1.5": 0.5 + torch.from_numpy(gen.random(k)),
            "half_ablate": torch.from_numpy((gen.random(k) < 0.5).astype(np.float64)),
            "half_double": 1.0 + torch.from_numpy((gen.random(k) < 0.5).astype(np.float64))}
    out = {}
    for name, rho in list(rhos.items())[:n_rho]:
        rn = rho[P]
        F_rho = H["A"] @ (layer.Wd * rn[None, :]).T  # edited weights D_c <- rho_c D_c
        Fh_rho = (H["U"] * Gt * rn[None, :]) @ layer.Wd.T
        mse = ((F_rho - Fh_rho) ** 2).sum(1).mean().item()
        pred = (rho @ K @ rho).item()
        dF, dFh = F_rho - H["F"], Fh_rho - Fh
        out[name] = {"mse": mse, "rhoKrho": pred, "identity_diff_over_EF2": abs(pred - mse) / EF2,
                     "rel_err": mse / max((F_rho ** 2).sum(1).mean().item(), EF2 * np.finfo(float).eps),
                     "change_rel_err": (((dF - dFh) ** 2).sum() / (dF ** 2).sum()).item(),
                     "change_over_F": ((dF ** 2).sum() / (H["F"] ** 2).sum()).item()}
    return out


def analyze(args):
    from gamfit import basis

    t0 = time.time()
    torch.set_num_threads(args.threads)
    cap = np.load(args.cache)
    ids = cap["ids"]
    n_seq, T = ids.shape
    ntr = n_seq // 2
    pos = np.arange(1, T)  # position 0 excluded
    split = {"train": torch.from_numpy((np.arange(ntr)[:, None] * T + pos).ravel()),
             "held": torch.from_numpy((np.arange(ntr, n_seq)[:, None] * T + pos).ravel())}
    n_default = basis.bspline_basis(np.linspace(0.0, 1.0, 64), None).shape[1]
    ks = [int(x) for x in args.ks.split(",")]
    receipt = {"model": str(cap["model"]), "model_type": str(cap["model_type"]), "n_layers": int(cap["n_layers"]),
               "tokens": {"sequences": int(n_seq), "seq_len": int(T), "train": int(split["train"].numel()),
                          "held_out": int(split["held"].numel()), "position0_excluded": True,
                          "split": "first half of sequences train, second half held-out"},
               "gate_basis": {"kind": "cubic B-spline on train-standardized z, interior knots at pooled train quantiles",
                              "cols_1d": n_default, "cols_2d": 2 * n_default - 1,
                              "penalty": "exact integrated squared 2nd derivative (gamfit.basis.smoothness_penalty)",
                              "gate2d": "additive f1(z1) + f2(z2), block penalty, one REML lambda",
                              "fit": "gamfit.reml.gaussian_reml_fit_batched (REML, weights w = |t|^2)",
                              "z1": "mean gate pre-activation over the group", "z2": "mean gate s(g) over the group"},
               "unit_gate_oracle_fit": "numpy stand-in: REML ridge (unpenalized intercept) of q on the group's unit gates",
               "ks": ks, "layers": {}}
    todo = [int(x) for x in args.layers.split(",")] if args.layers else [int(x) for x in cap["layers"]]
    for li in todo:
        tl = time.time()
        f64 = lambda key: torch.from_numpy(cap[key].astype(np.float64))  # noqa: E731
        X = f64(f"x{li}").reshape(-1, cap[f"x{li}"].shape[-1])
        Fcap = f64(f"f{li}").reshape(-1, cap[f"f{li}"].shape[-1])
        layer = Layer(X, f64(f"wg{li}"), f64(f"wu{li}"), f64(f"wd{li}"), split)
        if f"gff{li}" in cap:
            layer.gff, layer.eps = f64(f"gff{li}"), float(cap[f"eps{li}"])
        H = layer.sp["held"]
        n_ff = H["U"].shape[1]
        rec = {"n_ff": n_ff,
               "capture_check_rel": (((H["F"] - Fcap[split["held"]]) ** 2).sum() / (H["F"] ** 2).sum()).sqrt().item(),
               "relu_law": fidelity(H["F"], H["Frelu"])}
        if hasattr(layer, "gff"):
            rec["relu_law"]["post_norm_write"] = fidelity(post_norm(H["F"], layer.gff, layer.eps),
                                                          post_norm(H["Frelu"], layer.gff, layer.eps))
        Str = layer.sp["train"]["S"].T.numpy()
        feats = {"gate_corr": (Str - Str.mean(1, keepdims=True)) / Str.std(1, keepdims=True), "gate_raw": Str}
        gen = np.random.default_rng(args.seed + li)
        groupings = {"singleton": {n_ff: np.arange(n_ff)}}
        for fname, feat in feats.items():
            tc = time.time()
            groupings[fname] = ward_labels(feat, ks)
            print(f"  L{li} ward {fname} {time.time() - tc:.0f}s", flush=True)
            for s in range(args.random_seeds):
                groupings[f"random_{fname}_s{s}"] = {k: size_matched_random(lab, gen)
                                                      for k, lab in groupings[fname].items()}
        curves = {}
        for gname, bykey in groupings.items():
            curves[gname] = []
            for k, labels in bykey.items():
                tg = time.time()
                r = evaluate(layer, labels, n_default, gen, args.n_rho)
                curves[gname].append(r)
                g1 = r["gate1d"]
                print(f"  L{li} {gname:22s} k={r['k']:5d} op={g1['operator']:.3f} cond={g1['conditioning']:.3f} "
                      f"fit={g1['gate_fit']:.3f} fve1d={g1['fve_centered']:.3f} fve2d={r['gate2d']['fve_centered']:.3f} "
                      f"fveQ={r['oracle_q']['fve_centered']:.3f} fveLin={r['oracle_unit_gates']['fve_centered']:.3f} "
                      f"edit={g1['edits']['half_ablate']['identity_diff_over_EF2']:.1e} {r['seconds']}", flush=True)
        rec["curves"] = curves
        rec["runtime_s"] = time.time() - tl
        rec["reml_interpolated_groups"] = len(INTERPOLATED)
        INTERPOLATED.clear()
        receipt["layers"][str(li)] = rec
        with open(args.out, "w") as f:
            f.write(compact_json(receipt))
        print(f"L{li} relu fve={rec['relu_law']['fve_centered']:.3f} ({rec['runtime_s']:.0f}s)", flush=True)
        del layer
    receipt["runtime_s"] = time.time() - t0
    with open(args.out, "w") as f:
        f.write(compact_json(receipt))
    print("wrote", args.out)

def spectrum(sv, rs):
    """Energy spectrum of a centered token x unit matrix from its singular values."""
    ev = np.sort(sv ** 2)[::-1]
    tot = ev.sum()
    if tot <= 0:
        return None
    cum = np.cumsum(ev) / tot
    return {"energy": float(tot), "m": int(ev.size), "r90": int(np.searchsorted(cum, 0.90) + 1),
            "r99": int(np.searchsorted(cum, 0.99) + 1), "pr": float(tot ** 2 / (ev ** 2).sum()),
            "top": {r: float(ev[:r].sum()) for r in rs}}


def spectrum_summary(specs, rs):
    """Pooled over groups: top-r energy share, and energy-weighted means of r90/m, r99/m, participation/m."""
    specs = [x for x in specs if x is not None]
    E = np.array([x["energy"] for x in specs])
    m = np.array([x["m"] for x in specs], dtype=float)
    wmean = lambda v: float((E * v).sum() / E.sum())  # noqa: E731
    return {"top_r_share": {str(r): float(sum(x["top"][r] for x in specs) / E.sum()) for r in rs},
            "r90_over_m": wmean(np.array([x["r90"] for x in specs]) / m),
            "r99_over_m": wmean(np.array([x["r99"] for x in specs]) / m),
            "pr_over_m": wmean(np.array([x["pr"] for x in specs]) / m),
            "r90_median": float(np.median([x["r90"] for x in specs])),
            "pr_median": float(np.median([x["pr"] for x in specs]))}


def unit_splines(n_default, ztr, zho, Ytr, Yho, Wtr, Who, max_entries=4e7):
    """Every unit gate of a group as an additive smooth of the group's r gate coordinates: one shared design,
    one gamfit batched-REML fit per unit (own lambda, weights w_n = |d_n|^2 u_n^2). Returns the train-fitted
    prediction on held-out, the in-sample held-out fit (g* estimate), total edf and basis columns."""
    sd = ztr.std(0)
    spl = [Spline(n_default).fit_knots(ztr[:, j] / sd[j]) for j in range(ztr.shape[1])]

    def design(z):
        return np.hstack([sp.design(z[:, j] / sd[j])[:, (1 if j else 0):] for j, sp in enumerate(spl)])

    p = sum(sp.p for sp in spl) - (len(spl) - 1)
    S = np.zeros((p, p))
    o = 0
    for j, sp in enumerate(spl):
        blk = sp.S[1:, 1:] if j else sp.S
        S[o:o + blk.shape[0], o:o + blk.shape[0]] = blk
        o += blk.shape[0]
    Xtr, Xho = design(ztr), design(zho)
    per = max(1, int(max_entries // (Xtr.shape[0] * p)))
    pred, star, edf = np.empty_like(Yho), np.empty_like(Yho), 0.0
    for s0 in range(0, Ytr.shape[1], per):
        cols = range(s0, min(s0 + per, Ytr.shape[1]))
        c_tr, e = batched_reml([Xtr] * len(cols), [Ytr[:, n] for n in cols], [Wtr[:, n] for n in cols], S)
        c_ho, _ = batched_reml([Xho] * len(cols), [Yho[:, n] for n in cols], [Who[:, n] for n in cols], S)
        pred[:, s0:s0 + len(cols)] = Xho @ np.stack(c_tr, 1)
        star[:, s0:s0 + len(cols)] = Xho @ np.stack(c_ho, 1)
        edf += float(np.sum(e))
    return pred, star, edf, p


def multi_evaluate(layer, labels, rs, n_default, gen, n_rho):
    """Multi-template explanation: yhat_c = sum_{n in c} shat_n d_n u_n, so the operator term is zero and all
    error is in the gates. Gates of a group are modelled from its top-r principal gate coordinates z_c."""
    k = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(labels, minlength=k))[:-1])
    T, H = layer.sp["train"], layer.sp["held"]
    Str, Sho, Gtr, Gho = T["S"].numpy(), H["S"].numpy(), T["G"].numpy(), H["G"].numpy()
    dn2 = (layer.Wd ** 2).sum(0).numpy()
    Wtr, Who = T["U"].numpy() ** 2 * dn2, H["U"].numpy() ** 2 * dn2
    F = H["F"]
    specs = {"gates": [], "gates_perm": [], "active": [], "active_perm": []}
    models = {f"{name}_r{r}": {"S": Sho.copy(), "params": 0, "edf": 0.0, "star": Sho.copy() if name == "spline" else None}
              for r in rs for name in ("linear", "spline", "signgated")}
    act_frac = []
    for idx in groups:
        m = idx.size
        C = Str[:, idx] - Str[:, idx].mean(0)
        B = (Gtr[:, idx] > 0).astype(float)
        CB = B - B.mean(0)
        act_frac.append(B.mean() * m)
        _, sv, Vt = np.linalg.svd(C, full_matrices=False)
        perm = lambda A: np.stack([A[gen.permutation(A.shape[0]), j] for j in range(A.shape[1])], 1)  # noqa: E731
        specs["gates"].append(spectrum(sv, rs))
        specs["gates_perm"].append(spectrum(np.linalg.svd(perm(C), compute_uv=False), rs))
        specs["active"].append(spectrum(np.linalg.svd(CB, compute_uv=False), rs))
        specs["active_perm"].append(spectrum(np.linalg.svd(perm(CB), compute_uv=False), rs))
        rank = int((sv > sv[0] * max(C.shape) * np.finfo(float).eps).sum()) if sv.size and sv[0] > 0 else 0
        for r in rs:
            if r >= rank:  # r coordinates span every unit gate: the native gates, exactly, with no parameters
                continue
            Vr = Vt[:r].T
            ztr, zho = C @ Vr, (Sho[:, idx] - Str[:, idx].mean(0)) @ Vr
            lin = models[f"linear_r{r}"]
            lin["S"][:, idx] = Str[:, idx].mean(0) + zho @ Vr.T
            lin["params"] += m * (r + 1)
            sg = models[f"signgated_r{r}"]
            coef = np.linalg.lstsq(ztr, CB, rcond=None)[0]
            sg["S"][:, idx] = Gho[:, idx] * (B.mean(0) + zho @ coef)
            sg["params"] += m * (2 * r + 1)
            sp = models[f"spline_r{r}"]
            pred, star, edf, p = unit_splines(n_default, ztr, zho, Str[:, idx], Sho[:, idx], Wtr[:, idx], Who[:, idx])
            sp["S"][:, idx], sp["star"][:, idx] = pred, star
            sp["params"] += m * (r + p)
            sp["edf"] += edf
    E_unit = float((Who * Sho ** 2).sum())
    out = {"k": k, "size_max": int(max(g.size for g in groups)),
           "active_units_per_token_mean": float(np.sum(act_frac)) / labels.size,
           "spectra": {name: spectrum_summary(v, rs) for name, v in specs.items()}}
    for name, md in models.items():
        Shat = torch.from_numpy(md["S"])
        Fh = (H["U"] * Shat) @ layer.Wd.T
        err = float((Who * (Sho - md["S"]) ** 2).sum()) / E_unit
        res = {"params": md["params"], **fidelity(F, Fh), "unit_gate_err": err}
        if hasattr(layer, "gff"):
            res["post_norm_write"] = fidelity(post_norm(F, layer.gff, layer.eps), post_norm(Fh, layer.gff, layer.eps))
        if md["star"] is not None:
            cond = float((Who * (Sho - md["star"]) ** 2).sum()) / E_unit
            res.update(conditioning=cond, gate_fit=err - cond, edf=md["edf"])
        if not name.startswith("signgated"):
            res["edits"] = edits(layer, labels, k, H["U"] * (H["S"] - Shat), Fh, Shat, gen, n_rho)
        out[name] = res
    return out


def load_layer(cap, li, split):
    f64 = lambda key: torch.from_numpy(cap[key].astype(np.float64))  # noqa: E731
    X = f64(f"x{li}").reshape(-1, cap[f"x{li}"].shape[-1])
    layer = Layer(X, f64(f"wg{li}"), f64(f"wu{li}"), f64(f"wd{li}"), split)
    if f"gff{li}" in cap:
        layer.gff, layer.eps = f64(f"gff{li}"), float(cap[f"eps{li}"])
    return layer, f64(f"f{li}").reshape(-1, cap[f"f{li}"].shape[-1])


def analyze_multi(args):
    """Multi-template mode: templates are the group's own native vectors d_n u_n(h) (operator term zero by
    construction); the question is whether a group's unit gates are a low-dimensional function of shared
    coordinates. Groups: raw-gate Ward cuts and size-matched random groups (same seeds as the single mode)."""
    from gamfit import basis

    t0 = time.time()
    torch.set_num_threads(args.threads)
    cap = np.load(args.cache)
    n_seq, T = cap["ids"].shape
    ntr = n_seq // 2
    pos = np.arange(1, T)
    split = {"train": torch.from_numpy((np.arange(ntr)[:, None] * T + pos).ravel()),
             "held": torch.from_numpy((np.arange(ntr, n_seq)[:, None] * T + pos).ravel())}
    n_default = basis.bspline_basis(np.linspace(0.0, 1.0, 64), None).shape[1]
    ks = [int(x) for x in args.ks.split(",")]
    rs = [int(x) for x in args.ranks.split(",")]
    receipt = {"model": str(cap["model"]), "mode": "multi-template", "n_layers": int(cap["n_layers"]),
               "tokens": {"train": int(split["train"].numel()), "held_out": int(split["held"].numel()),
                          "position0_excluded": True, "split": "first half of sequences train, second half held-out"},
               "templates": "per-unit native vectors d_n u_n(h); operator inadequacy is zero by construction",
               "z": "top-r principal coordinates of the group's centered train gate matrix s(g_c) (held-out: same "
                    "projection of held-out gates)",
               "models": {"linear_r": "rank-r PCA reconstruction of the group's gates; params m(r+1)",
                          "spline_r": "each unit gate = additive cubic B-spline in the r coordinates (knots at the "
                                      "group's train quantiles, block 2nd-derivative penalty, one REML lambda per "
                                      "unit, gamfit.reml.gaussian_reml_fit_batched, weights |d_n|^2 u_n^2); params "
                                      "m(r + basis cols); conditioning = in-sample held-out refit",
                          "signgated_r": "s_n ~ g_n * bhat_n with bhat = least-squares map from z to the active "
                                         "indicator 1[g_n>0]; r = group size is the relu law; params m(2r+1)",
                          "exact_when_r_ge_rank": "groups whose gate rank <= r keep native gates (0 params)"},
               "spectra": "gates: centered s(g_c); gates_perm: each unit column permuted over tokens (null for "
                          "shared structure); active: centered 1[g>0]; top_r_share pooled over groups",
               "unit_gate_err": "sum_n E[|d_n|^2 u_n^2 (s_n - shat_n)^2] / sum_n E[|y_n|^2] (diagonal part of the "
                                "output error)",
               "ks": ks, "ranks": rs, "basis_cols_1d": n_default, "layers": {}}
    todo = [int(x) for x in args.layers.split(",")] if args.layers else [int(x) for x in cap["layers"]]
    for li in todo:
        tl = time.time()
        layer, _ = load_layer(cap, li, split)
        H = layer.sp["held"]
        n_ff = H["U"].shape[1]
        rec = {"n_ff": n_ff, "relu_law": fidelity(H["F"], H["Frelu"]),
               "relu_active_units_per_token": float((H["G"] > 0).double().mean().item())}
        Str = layer.sp["train"]["S"].T.numpy()
        gen = np.random.default_rng(args.seed + li)
        # same generator sequence as the single mode, so the random groups are the same groups
        corr = ward_labels((Str - Str.mean(1, keepdims=True)) / Str.std(1, keepdims=True), ks)
        _ = {k: size_matched_random(lab, gen) for k, lab in corr.items()}
        raw = ward_labels(Str, ks)
        groupings = {"gate_raw": raw, "random_gate_raw_s0": {k: size_matched_random(lab, gen) for k, lab in raw.items()}}
        curves = {}
        for gname, bykey in groupings.items():
            curves[gname] = []
            for k, labels in bykey.items():
                tg = time.time()
                r = multi_evaluate(layer, labels, rs, n_default, gen, args.n_rho)
                r["seconds"] = round(time.time() - tg, 1)
                curves[gname].append(r)
                sp = r["spectra"]["gates"]
                print(f"  L{li} {gname:20s} k={r['k']:5d} top1/2/4={[round(v, 3) for v in sp['top_r_share'].values()]} "
                      f"perm={[round(v, 3) for v in r['spectra']['gates_perm']['top_r_share'].values()]} "
                      + " ".join(f"r{q}: lin={r[f'linear_r{q}']['fve_centered']:.3f} spl={r[f'spline_r{q}']['fve_centered']:.3f} "
                                 f"sg={r[f'signgated_r{q}']['fve_centered']:.3f}" for q in rs)
                      + f" ({r['seconds']:.0f}s)", flush=True)
        rec["curves"] = curves
        rec["runtime_s"] = time.time() - tl
        rec["reml_interpolated_fits"] = len(INTERPOLATED)
        INTERPOLATED.clear()
        receipt["layers"][str(li)] = rec
        with open(args.out, "w") as f:
            f.write(compact_json(receipt))
        print(f"L{li} relu fve={rec['relu_law']['fve_centered']:.3f} ({rec['runtime_s']:.0f}s)", flush=True)
        del layer
    receipt["runtime_s"] = time.time() - t0
    with open(args.out, "w") as f:
        f.write(compact_json(receipt))
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["capture", "analyze"], required=True)
    ap.add_argument("--cache", default=os.path.expanduser("~/mpd-data/cond_operators_capture.npz"))
    ap.add_argument("--layers", default="", help="capture: layers to record; analyze: subset of captured layers")
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="4,16,64,256,1024,4096")
    ap.add_argument("--random-seeds", type=int, default=1)
    ap.add_argument("--n-rho", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--mode", choices=["single", "multi"], default="single",
                    help="analyze: single-template conditional operators, or multi-template low-rank gates")
    ap.add_argument("--ranks", default="1,2,4", help="multi mode: gate-coordinate ranks r")
    ap.add_argument("--merge", nargs="*", default=[], help="analyze: merge per-layer receipts into --out")
    ap.add_argument("--out", default="experiments/issue-2951/receipts/opfirst_cond_operators.json")
    args = ap.parse_args()
    if args.merge:
        parts = [json.load(open(p)) for p in args.merge]
        merged = parts[0]
        for part in parts[1:]:
            merged["layers"].update(part["layers"])
            merged["runtime_s"] = max(merged["runtime_s"], part["runtime_s"])
        merged["layers"] = dict(sorted(merged["layers"].items(), key=lambda kv: int(kv[0])))
        with open(args.out, "w") as f:
            f.write(compact_json(merged))
        return
    if args.stage == "capture":
        capture(args)
    else:
        analyze_multi(args) if args.mode == "multi" else analyze(args)


if __name__ == "__main__":
    main()
