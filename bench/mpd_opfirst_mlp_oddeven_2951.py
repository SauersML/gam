"""#2951 probe: how much of a Qwen3 SwiGLU MLP is its bilinear (even-in-x) part vs the odd remainder?

Analysis under SPEC 8's exception (torch execution of a measurement), float64 on CPU throughout.

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
import time

import torch

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
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-seq", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--scales", default="0.1,0.3,1")
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float64).eval()
    ids, source = texts_tokens(tok, args.n_seq, args.seq_len, args.seed)
    captured = {}

    def make_hook(i):
        def hook(_m, inputs, output):
            captured[i] = (inputs[0].reshape(-1, inputs[0].shape[-1]).clone(), output.reshape(-1, output.shape[-1]).clone())
        return hook

    layers = model.model.layers
    handles = [layers[i].mlp.register_forward_hook(make_hook(i)) for i in range(len(layers))]
    with torch.inference_mode():
        model(input_ids=ids)
    for h in handles:
        h.remove()
    t_fwd = time.time() - t0
    scales = [float(s) for s in args.scales.split(",")]
    gen = torch.Generator().manual_seed(args.seed)
    report = {"model": args.model, "text_source": source, "tokens": int(ids.numel()), "dtype": "float64",
              "act": model.config.hidden_act, "layers": {}}
    with torch.inference_mode():
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            Wg, Wu, Wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
            X, ref = captured[i]
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
        json.dump(report, fh, indent=1)
    print("runtime", report["runtime_s"])


if __name__ == "__main__":
    main()
