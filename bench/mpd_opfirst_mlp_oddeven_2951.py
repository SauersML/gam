"""#2951 probe: how much of a SwiGLU MLP is its bilinear (even-in-x) part vs the odd remainder?

Analysis under SPEC 8's exception (torch execution of a measurement). The model runs on CPU at --dtype (float32
by default: a 1B model is never materialised in float64); every per-layer analysis casts that layer's weights and
captured MLP inputs to float64. The architecture is read from config (bench/mpd_opfirst_decoder_2951.py): in a
pre-norm model (Qwen3) the MLP reads the RMS-normalised residual; in a post-norm model (OLMo 2) it reads the raw
residual and its output F is rescaled by N_ff(F) = gamma_ff * F / rms(F) before the residual add, so relusplit
also reports how well N_ff(P) reproduces the actual residual write N_ff(F).

Exact identity (checked numerically below, not assumed): silu(s) = s/2 + psi(s) with psi(s) = s (sigmoid(s) - 1/2) even,
because sigmoid(-s) = 1 - sigmoid(s). With g = W_gate x, u = W_up x, the MLP F(x) = W_down (silu(g) * u) splits as

    F(x) = Q(x) + R(x),   Q(x) = 1/2 W_down (g * u),   R(x) = W_down (psi(g) * u).

Q is a homogeneous quadratic form (even in x) and R is odd in x, so Q = (F(x) + F(-x)) / 2 and R = (F(x) - F(-x)) / 2
exactly. Near x = 0, psi(s) = s^2/4 + O(s^4), so R = O(|x|^3) while Q = O(|x|^2).

Measured (empirical) on post-RMSNorm MLP inputs of fineweb-edu tokens:
  * per layer: ||Q||/||F||, ||R||/||F||, cos(Q, F), variance of F explained by Q alone, identity residuals;
  * the same under input scaling x -> t x (t in {0.1, 0.3, 1}) with fitted log-log exponents of ||Q||, ||R||.

Parameter-only (exact, no d^3 tensor formed): T_ijk = 1/2 sum_n Wd[i,n] Wg[n,j] Wu[n,k]. Each mode-m unfolding's Gram is a
d x d matrix built from Hadamard products of I x I Gram matrices, e.g. T_(1) T_(1)^T = 1/4 Wd [(Wg Wg^T) o (Wu Wu^T)] Wd^T.
The quadratic form only sees the (j,k)-symmetrised tensor; its mode-1 Gram adds the cross term (Wg Wu^T) o (Wu Wg^T).
Per-output quadratic forms M_r = 1/2 Wg^T diag(Wd^T r) Wu (symmetrised) are also spectrum-profiled for random unit r.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import mlp_input_norm, mlp_output_norm, placement  # noqa: E402

FALLBACK_TEXT = (
    "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools "
    "and condenses into clouds, and falls again to the surface as precipitation. Plants take up water through their "
    "roots and release it through their leaves in a process called transpiration. In 1854 John Snow traced a cholera "
    "outbreak in London to a single public water pump on Broad Street, an early triumph of epidemiology. Students "
    "learning algebra often begin with linear equations such as 3x + 5 = 20, then move to quadratic equations whose "
    "solutions are given by the quadratic formula. "
)


def texts_tokens(tok, n_seq, seq_len, seed):
    """The first n_seq x seq_len fineweb-edu tokens of the seeded shuffle, cached under ~/mpd-data/tokens per
    tokenizer (the stream is deterministic, so every checkpoint sharing a tokenizer reads identical tokens)."""
    source = "HuggingFaceFW/fineweb-edu sample-10BT (streamed, shuffle seed %d)" % seed
    cache = os.path.expanduser("~/mpd-data/tokens/%s_%dx%d_s%d.pt" % (
        tok.name_or_path.replace("/", "--"), n_seq, seq_len, seed))
    if os.path.exists(cache):
        return torch.load(cache), source
    try:
        from mpd_llm_chart_restriction_2951 import token_batches
        ids = next(token_batches(tok, seq_len, n_seq, seed, 0))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(ids, cache)
        return ids, source
    except Exception as exc:  # offline fallback, recorded in the receipt
        buf = tok(FALLBACK_TEXT * 40).input_ids
        return torch.tensor(buf[: n_seq * seq_len]).view(n_seq, seq_len), "fallback fixed text (%s)" % type(exc).__name__


def spectrum_stats(ev):
    ev = ev.clamp_min(0).sort(descending=True).values
    tot = ev.sum()
    cum = ev.cumsum(0) / tot
    return {
        "dim": int(ev.numel()),
        "rank90": int((cum < 0.90).sum()) + 1,
        "rank99": int((cum < 0.99).sum()) + 1,
        "participation": float(tot ** 2 / (ev ** 2).sum()),
        "top1_share": float(ev[0] / tot),
    }


def psi(s):
    return s * (torch.sigmoid(s) - 0.5)


def layer_metrics(X, Wg, Wu, Wd, ref=None):
    g, u = X @ Wg.T, X @ Wu.T
    F = (torch.nn.functional.silu(g) * u) @ Wd.T
    Q = 0.5 * (g * u) @ Wd.T
    R = (psi(g) * u) @ Wd.T
    Fm = (torch.nn.functional.silu(-g) * -u) @ Wd.T
    nF = F.norm()
    out = {
        "Q_over_F": float(Q.norm() / nF),
        "R_over_F": float(R.norm() / nF),
        "cos_QF_agg": float((Q * F).sum() / (Q.norm() * nF)),
        "cos_RF_agg": float((R * F).sum() / (R.norm() * nF)),
        "cos_QF_tok_median": float(torch.nn.functional.cosine_similarity(Q, F, dim=-1).median()),
        "cos_QR_agg": float((Q * R).sum() / (Q.norm() * R.norm())),
        "fve_Q_centered": float(1 - (F - Q).pow(2).sum() / (F - F.mean(0)).pow(2).sum()),
        "fve_Q_uncentered": float(1 - (F - Q).pow(2).sum() / F.pow(2).sum()),
        "fve_R_centered": float(1 - (F - R).pow(2).sum() / (F - F.mean(0)).pow(2).sum()),
        "Q_over_F_tok_median": float((Q.norm(dim=-1) / F.norm(dim=-1)).median()),
        "exact_identity_rel": float((F - Q - R).norm() / nF),
        "exact_even_rel": float(((F + Fm) / 2 - Q).norm() / nF),
        "exact_odd_rel": float(((F - Fm) / 2 - R).norm() / nF),
        "normQ": float(Q.norm()),
        "normR": float(R.norm()),
    }
    if ref is not None:
        out["vs_module_rel"] = float((F - ref).norm() / nF)
    return out


def relu_split(X, Wg, Wu, Wd, Wd_op, flat_ids, positions, tok, top, out_norm=None):
    """Exact split silu(g) = relu(g) + e(g), e(g) = -|g| sigmoid(-|g|) even, |e| <= 0.2785, exponentially small in |g|.

    F = P + C with P = W_down (relu(g) * u) (sign-gated bilinear) and C = W_down (e(g) * u) (bounded even correction).
    Certified per token: ||C|| <= ||W_down||_2 ||e(g) * u|| (exact given g, u).
    out_norm = (gain, eps) of a post-MLP RMSNorm: the residual write is N(F), and relu replacement writes N(P).
    """
    g, u = X @ Wg.T, X @ Wu.T
    F = (torch.nn.functional.silu(g) * u) @ Wd.T
    a = torch.relu(g) * u
    e = -g.abs() * torch.sigmoid(-g.abs())
    P, C = a @ Wd.T, (e * u) @ Wd.T
    nF_tok = F.norm(dim=-1)
    bound = Wd_op * (e * u).norm(dim=-1)
    ratio_tok = C.norm(dim=-1) / nF_tok
    e2 = a.pow(2).sort(-1, descending=True).values
    k90 = ((e2.cumsum(-1) / e2.sum(-1, keepdim=True)) < 0.90).sum(-1) + 1
    active = (g > 0).sum(-1)
    q = lambda v: {"median": float(v.median()), "min": float(v.min()), "max": float(v.max())}
    share = nF_tok.pow(2) / nF_tok.pow(2).sum()
    order = share.argsort(descending=True)[:top]
    heavy = [{"row": int(j), "position": int(positions[j]), "token": tok.decode([int(flat_ids[j])]),
              "share_of_sum_F2": float(share[j]), "F_norm": float(nF_tok[j]),
              "F_norm_over_median": float(nF_tok[j] / nF_tok.median())} for j in order]
    keep = torch.ones(F.shape[0], dtype=torch.bool)
    keep[order] = False
    Q = 0.5 * (g * u) @ Wd.T

    def pooled(mask):
        Fm, Qm, Pm = F[mask], Q[mask], P[mask]
        ss = (Fm - Fm.mean(0)).pow(2).sum()
        return {"cos_QF": float((Qm * Fm).sum() / (Qm.norm() * Fm.norm())),
                "fve_Q_centered": float(1 - (Fm - Qm).pow(2).sum() / ss),
                "fve_P_centered": float(1 - (Fm - Pm).pow(2).sum() / ss)}

    pos0 = positions != 0
    post = {}
    if out_norm is not None:
        gain, eps = out_norm
        norm = lambda v: gain * v / (v.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        NF, NP = norm(F), norm(P)
        post = {"post_norm_write": {
            "fve_NP_centered": float(1 - (NF - NP).pow(2).sum() / (NF - NF.mean(0)).pow(2).sum()),
            "rel_err_agg": float((NF - NP).norm() / NF.norm()),
            "rel_err_tok": q((NF - NP).norm(dim=-1) / NF.norm(dim=-1)),
            "rms_ratio_P_over_F_tok": q(P.pow(2).mean(-1).sqrt() / F.pow(2).mean(-1).sqrt())}}
    return post | {
        "exact_identity_rel": float((F - P - C).norm() / F.norm()),
        "e_abs_max": float(e.abs().max()),
        "C_over_F_agg": float(C.norm() / F.norm()),
        "C_over_F_tok": q(ratio_tok),
        "fve_P_centered": float(1 - C.pow(2).sum() / (F - F.mean(0)).pow(2).sum()),
        "cos_PF_agg": float((P * F).sum() / (P.norm() * F.norm())),
        "bound_over_F_tok": q(bound / nF_tok),
        "bound_holds_all": bool((C.norm(dim=-1) <= bound * (1 + 1e-12)).all()),
        "bound_tightness_tok_median": float((C.norm(dim=-1) / bound).median()),
        "active_frac_tok": q(active.double() / g.shape[1]),
        "units90_tok": q(k90.double()),
        "units90_over_active_tok": q(k90.double() / active.clamp_min(1)),
        "heavy_tokens": heavy,
        "pooled_all": pooled(torch.ones_like(keep)),
        "pooled_drop_heavy": pooled(keep),
        "pooled_drop_position0": pooled(pos0),
    }


E_PEAK = 1.2784645427610738  # argmax_t t sigmoid(-t); |e| <= E_PEAK sigmoid(-E_PEAK) = 0.2785 (exact)


def e_fn(g):
    return -g.abs() * torch.sigmoid(-g.abs())


def correction_bounds(X, Wg, Wu, Wd, gamma, k, chunk=64):
    """Certified bounds on ||C|| = ||W_d (e(g) * u)|| (C is what relu replacement drops), tightest-first comparison.

    per-token (exact given g, u): op = ||W_d||_2 ||e u||; tri = sum_n |e_n u_n| ||w_n||;
      split_k = ||W_d[:, A]||_2 ||(e u)_A|| + sum_{n not in A} |e_n u_n| ||w_n||, A = top-k units by |e_n u_n| ||w_n||.
    a priori: param = 0.2785 ||W_d||_2 ||W_u||_2 max|gamma| sqrt(d) (parameter-only; ||RMSNorm output|| <= max|gamma| sqrt(d);
      None when the MLP reads the raw residual, gamma None: no parameter-only bound on its input exists);
      box = min(||W_d||_2 ||emax * umax||, sum_n emax_n umax_n ||w_n||) over the observed per-unit box
      g_n in [min, max], |u_n| <= max (empirical sup: valid for any input inside the observed ranges).
    """
    g, u = X @ Wg.T, X @ Wu.T
    eu = e_fn(g) * u
    C = eu @ Wd.T
    nC = C.norm(dim=-1)
    wn = Wd.norm(dim=0)
    op = torch.linalg.matrix_norm(Wd, ord=2)
    b_op = op * eu.norm(dim=-1)
    contrib = eu.abs() * wn
    b_tri = contrib.sum(-1)
    top = contrib.topk(k, dim=-1).indices
    b_split = []
    for s in range(0, X.shape[0], chunk):
        idx = top[s:s + chunk]
        sub_op = torch.linalg.matrix_norm(Wd.T[idx], ord=2)
        in_a = eu[s:s + chunk].gather(-1, idx)
        b_split.append(sub_op * in_a.norm(dim=-1) + b_tri[s:s + chunk] - contrib[s:s + chunk].gather(-1, idx).sum(-1))
    b_split = torch.cat(b_split)
    gmin, gmax = g.min(0).values, g.max(0).values
    peak = torch.full_like(gmin, E_PEAK)
    cand = torch.stack([gmin, gmax, torch.clamp(peak, gmin, gmax), torch.clamp(-peak, gmin, gmax)])
    emax = e_fn(cand).abs().max(0).values
    umax = u.abs().max(0).values
    b_box = min(float(op * (emax * umax).norm()), float((emax * umax * wn).sum()))
    d = X.shape[1]
    b_param = None if gamma is None else float(
        0.27846454276107 * op * torch.linalg.matrix_norm(Wu, ord=2) * gamma.abs().max() * math.sqrt(d))
    nF = ((torch.nn.functional.silu(g) * u) @ Wd.T).norm(dim=-1)
    best = torch.minimum(b_op, b_split)
    med = lambda v: float(v.median())
    return {
        "C_norm_tok": {"median": med(nC), "max": float(nC.max())},
        "F_norm_tok_median": med(nF),
        "bound_over_C_tok_median": {"op": med(b_op / nC), "tri": med(b_tri / nC), f"split{k}": med(b_split / nC),
                                    "best": med(best / nC)},
        "bound_over_F_tok_median": {"op": med(b_op / nF), "tri": med(b_tri / nF), f"split{k}": med(b_split / nF),
                                    "best": med(best / nF)},
        "all_bounds_hold": bool(((nC <= b_op * (1 + 1e-12)) & (nC <= b_tri * (1 + 1e-12))
                                 & (nC <= b_split * (1 + 1e-12))).all()),
        "apriori_box": b_box, "apriori_box_over_maxC": b_box / float(nC.max()),
        "apriori_box_over_medianF": b_box / med(nF),
        "apriori_param": b_param, "apriori_param_over_maxC": None if b_param is None else b_param / float(nC.max()),
        "apriori_param_over_medianF": None if b_param is None else b_param / med(nF),
    }


class MaskedRelu(torch.nn.Module):
    """relu on the (batch, seq) tokens in mask, silu elsewhere."""

    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, g):
        return torch.where(self.mask[..., None], torch.relu(g), torch.nn.functional.silu(g))


def replace_runs(model, ids, clean_logits, configs, positions):
    """Swap silu -> relu (drop C exactly) in a layer set, rerun the full model, compare next-token distributions.

    configs: (kind, L0, layer_set, swap_mask) with swap_mask a (batch, seq) bool of tokens to swap, None for all tokens.
    """
    clean = torch.log_softmax(clean_logits, -1).reshape(-1, clean_logits.shape[-1])
    tgt = torch.cat([ids[:, 1:], torch.full((ids.shape[0], 1), -1)], 1).reshape(-1)
    has_tgt = tgt >= 0
    nll_clean = -clean[has_tgt].gather(-1, tgt[has_tgt, None]).squeeze(-1)
    keep = positions != 0
    out = []
    for kind, l0, layer_set, swap_mask in configs:
        saved = {i: model.model.layers[i].mlp.act_fn for i in layer_set}
        for i in layer_set:
            if swap_mask is None:
                model.model.layers[i].mlp.act_fn = torch.nn.ReLU()
            else:
                model.model.layers[i].mlp.act_fn = MaskedRelu(swap_mask)
        with torch.inference_mode():
            lp = torch.log_softmax(model(input_ids=ids).logits, -1).reshape(-1, clean.shape[-1])
        for i, fn in saved.items():
            model.model.layers[i].mlp.act_fn = fn
        kl = (clean.exp() * (clean - lp)).sum(-1)
        agree = clean.argmax(-1) == lp.argmax(-1)
        nll = -lp[has_tgt].gather(-1, tgt[has_tgt, None]).squeeze(-1)
        kt = keep[has_tgt]
        row = {"kind": kind, "L0": l0, "layers": f"{layer_set[0]}-{layer_set[-1]}" if layer_set else "none",
               "kl_mean": float(kl[keep].mean()), "kl_median": float(kl[keep].median()), "kl_max": float(kl[keep].max()),
               "top1_agree": float(agree[keep].double().mean()),
               "loss_clean": float(nll_clean[kt].mean()), "loss_delta": float((nll - nll_clean)[kt].mean()),
               "pos0_kl_mean": float(kl[~keep].mean()), "pos0_kl_max": float(kl[~keep].max()),
               "pos0_top1_agree": float(agree[~keep].double().mean())}
        out.append(row)
        print(f"{kind:6s} L0={l0:2d} [{row['layers']}] KL mean={row['kl_mean']:.2e} max={row['kl_max']:.2e} "
              f"top1={row['top1_agree']:.3f} dloss={row['loss_delta']:+.2e} pos0KL={row['pos0_kl_mean']:.2e}", flush=True)
    return out


def tensor_rank(Wg, Wu, Wd, n_probe, gen):
    Gg, Gu, C = Wg @ Wg.T, Wu @ Wu.T, Wg @ Wu.T
    res = {
        "mode1_out": spectrum_stats(torch.linalg.eigvalsh(0.25 * Wd @ (Gg * Gu) @ Wd.T)),
        "mode1_out_sym": spectrum_stats(torch.linalg.eigvalsh(0.125 * Wd @ (Gg * Gu + C * C.T) @ Wd.T)),
        "mode2_gate_in": spectrum_stats(torch.linalg.eigvalsh(0.25 * Wg.T @ ((Wd.T @ Wd) * Gu) @ Wg)),
        "mode3_up_in": spectrum_stats(torch.linalg.eigvalsh(0.25 * Wu.T @ ((Wd.T @ Wd) * Gg) @ Wu)),
        "Wd_alone": spectrum_stats(torch.linalg.svdvals(Wd) ** 2),
    }
    probes = []
    for _ in range(n_probe):
        r = torch.randn(Wd.shape[0], generator=gen, dtype=Wd.dtype)
        r /= r.norm()
        M = 0.5 * Wg.T @ ((Wd.T @ r)[:, None] * Wu)
        ev = torch.linalg.eigvalsh(0.5 * (M + M.T))
        st = spectrum_stats(ev ** 2)
        st["pos_mass"] = float(ev.clamp_min(0).pow(2).sum() / ev.pow(2).sum())
        probes.append(st)
    res["per_output_qform"] = {k: float(sum(p[k] for p in probes) / n_probe) for k in probes[0]}
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="model execution dtype; per-layer analyses are float64 either way")
    parser.add_argument("--n-seq", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--scales", default="0.1,0.3,1")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["oddeven", "relusplit", "replace", "single"], default="oddeven")
    parser.add_argument("--massive-factor", type=float, default=100.0)
    parser.add_argument("--worst", type=int, default=3)
    parser.add_argument("--include-pos0", action="store_true", help="position 0 (attention sink) is excluded by default")
    parser.add_argument("--suffix-l0", default="", help="empty: L-1, L-1-s, ..., L-1-4s, 0 with s = round(3L/28)")
    parser.add_argument("--prefix-l0", default="", help="empty: the suffix starts without 0")
    parser.add_argument("--bound-k", type=int, default=64)
    parser.add_argument("--heavy", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 dtype=getattr(torch, args.dtype)).eval()
    where = placement(model.config)
    ids, source = texts_tokens(tok, args.n_seq, args.seq_len, args.seed)
    captured = {}

    def make_hook(i):
        def hook(_m, inputs, output):
            captured[i] = (inputs[0].reshape(-1, inputs[0].shape[-1]).clone(), output.reshape(-1, output.shape[-1]).clone())
        return hook

    layers = model.model.layers
    handles = [layers[i].mlp.register_forward_hook(make_hook(i)) for i in range(len(layers))]
    resid = {}

    def make_resid_hook(i):
        def hook(_m, inputs):
            resid[i] = inputs[0].norm(dim=-1).clone()
        return hook

    # the residual entering the MLP block: input of the pre-MLP norm, or of the MLP itself in a post-norm model
    handles += [(mlp_input_norm(layers[i], where) or layers[i].mlp).register_forward_pre_hook(make_resid_hook(i))
                for i in range(len(layers))]
    with torch.inference_mode():
        clean_logits = model(input_ids=ids).logits
    for h in handles:
        h.remove()
    t_fwd = time.time() - t0
    positions = torch.arange(ids.shape[1]).repeat(ids.shape[0])
    rows = torch.ones_like(positions, dtype=torch.bool) if args.include_pos0 else positions != 0
    flat_ids = ids.reshape(-1)[rows]
    scales = [float(s) for s in args.scales.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    report = {"model": args.model, "revision": args.revision, "mode": args.mode, "text_source": source,
              "tokens": int(ids.numel()), "dtype": {"model": args.dtype, "analysis": "float64"},
              "norm_placement": where, "act": model.config.hidden_act,
              "stats_exclude_position0": not args.include_pos0, "layers": {}}
    n_layers = len(layers)
    step = max(1, round(3 * n_layers / 28))
    suffix = [int(v) for v in args.suffix_l0.split(",")] if args.suffix_l0 else [n_layers - 1 - k * step for k in range(5)] + [0]
    prefix = [int(v) for v in args.prefix_l0.split(",")] if args.prefix_l0 else [v for v in suffix if v]
    if args.mode == "replace":
        configs = [("suffix", l0, list(range(l0, n_layers)), None) for l0 in suffix]
        configs += [("prefix", l0, list(range(0, l0)), None) for l0 in prefix]
        report["replacement"] = replace_runs(model, ids, clean_logits, configs, positions)
    if args.mode == "single":
        # massive = pre-norm residual norm (input of post_attention_layernorm) > factor x that layer's median token
        rows_single = replace_runs(model, ids, clean_logits, [("single", i, [i], None) for i in range(n_layers)], positions)
        worst = sorted(range(n_layers), key=lambda i: -rows_single[i]["kl_mean"])[: args.worst]
        audit = []
        for i in worst:
            r = resid[i]
            massive = r > args.massive_factor * r.median()
            mlp_in = captured[i][0].norm(dim=-1).view_as(r)
            info = {"layer": i, "n_massive": int(massive.sum()),
                    "massive_positions": sorted({int(p) for p in massive.nonzero()[:, 1]}),
                    "massive_resid_over_median_min": float((r[massive] / r.median()).min()) if massive.any() else None,
                    "resid_max_over_median": float(r.max() / r.median()),
                    "mlp_input_max_over_median": float(mlp_in.max() / mlp_in.median())}
            runs = replace_runs(model, ids, clean_logits, [("ordinary_only", i, [i], ~massive),
                                                           ("massive_only", i, [i], massive)], positions)
            info["all_tokens"] = rows_single[i]
            info["ordinary_only"], info["massive_only"] = runs
            audit.append(info)
            print(f"L{i:02d} massive n={info['n_massive']} pos={info['massive_positions']} "
                  f"resid max/med={info['resid_max_over_median']:.0f} mlp_in max/med={info['mlp_input_max_over_median']:.1f}",
                  flush=True)
        report["single_layer"] = rows_single
        report["worst_layer_massive_audit"] = audit
        report["layers"] = {}
        report["runtime_s"] = {"forward_and_load": t_fwd, "total": time.time() - t0}
        with open(args.out, "w") as fh:
            fh.write("{\n" + ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}"
                                         for k, v in report.items()) + "\n}\n")
        print("runtime", report["runtime_s"])
        return
    with torch.inference_mode():
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            Wg, Wu, Wd = mlp.gate_proj.weight.double(), mlp.up_proj.weight.double(), mlp.down_proj.weight.double()
            X, ref = captured[i]
            X, ref = X[rows].double(), ref[rows].double()
            in_norm, out_norm = mlp_input_norm(layer, where), mlp_output_norm(layer, where)
            if args.mode == "replace":
                entry = correction_bounds(X, Wg, Wu, Wd, None if in_norm is None else in_norm.weight.double(),
                                          args.bound_k)
                report["layers"][i] = entry
                b = entry["bound_over_C_tok_median"]
                print(f"L{i:02d} |C|med={entry['C_norm_tok']['median']:.3f} |F|med={entry['F_norm_tok_median']:.3f} "
                      f"bound/|C| op={b['op']:.2f} tri={b['tri']:.2f} split={b[f'split{args.bound_k}']:.2f} "
                      f"best/|F|={entry['bound_over_F_tok_median']['best']:.3f} holds={entry['all_bounds_hold']} "
                      f"box/maxC={entry['apriori_box_over_maxC']:.1f} param/maxC={entry['apriori_param_over_maxC']}",
                      flush=True)
                continue
            if args.mode == "relusplit":
                Wd_op = float(torch.linalg.matrix_norm(Wd, ord=2))
                entry = relu_split(X, Wg, Wu, Wd, Wd_op, flat_ids, positions[rows], tok, args.heavy,
                                   None if out_norm is None else (out_norm.weight.double(), out_norm.variance_epsilon))
                entry["W_down_op_norm"] = Wd_op
                report["layers"][i] = entry
                h = entry["heavy_tokens"][0]
                print(f"L{i:02d} |C|/|F|={entry['C_over_F_agg']:.3f} tokmed={entry['C_over_F_tok']['median']:.3f} "
                      f"fveP={entry['fve_P_centered']:.4f} act={entry['active_frac_tok']['median']:.3f} "
                      f"k90={entry['units90_tok']['median']:.0f} bnd={entry['bound_over_F_tok']['median']:.3f} "
                      f"id={entry['exact_identity_rel']:.1e} heavy=pos{h['position']}:{h['token']!r}:{h['share_of_sum_F2']:.2f} "
                      f"postnorm fve={entry.get('post_norm_write', {}).get('fve_NP_centered', float('nan')):.4f}",
                      flush=True)
                continue
            entry = {"input_rms_median": float(X.pow(2).mean(-1).sqrt().median())}
            entry["real"] = layer_metrics(X, Wg, Wu, Wd, ref)
            entry["scaled"] = {str(t): layer_metrics(t * X, Wg, Wu, Wd) for t in scales}
            lt = [math.log(t) for t in scales]
            for part in ("normQ", "normR"):
                ly = [math.log(entry["scaled"][str(t)][part]) for t in scales]
                entry[part + "_exponent_lo"] = (ly[1] - ly[0]) / (lt[1] - lt[0])
            entry["tensor"] = tensor_rank(Wg, Wu, Wd, args.probes, gen)
            report["layers"][i] = entry
            r = entry["real"]
            print(f"L{i:02d} |Q|/|F|={r['Q_over_F']:.3f} |R|/|F|={r['R_over_F']:.3f} cos={r['cos_QF_agg']:.3f} "
                  f"fveQ={r['fve_Q_centered']:.3f} id={r['exact_identity_rel']:.1e} mod={r['vs_module_rel']:.1e} "
                  f"expQ={entry['normQ_exponent_lo']:.2f} expR={entry['normR_exponent_lo']:.2f} "
                  f"r99={entry['tensor']['mode1_out_sym']['rank99']}", flush=True)
    report["runtime_s"] = {"forward_and_load": t_fwd, "total": time.time() - t0}
    with open(args.out, "w") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}"
                                     for k, v in report.items()) + "\n}\n")
    print("runtime", report["runtime_s"])


if __name__ == "__main__":
    main()
