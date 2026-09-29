"""#2951 probe: the sign-gated split of a SwiGLU MLP and its SiLU -> ReLU replacement.

Analysis under SPEC 8's exception (torch execution of a measurement). The model runs on --device (mps by default) at
float32; each captured residual moves to the CPU once as float64, and every layer's block is handed to the Rust owner `parameter_decomposition::sign_gated` through the MPD
surface op `sign_gated_swiglu` (gamfit.sae.run_parameter_decomposition), which executes it in float64 with derived
forward-error bands. This file does no split math of its own: it captures each MLP block's residual input, calls
the op, and shapes the receipt (quantiles, heavy tokens). The architecture is read from config
(bench/mpd_opfirst_decoder_2951.py): in a pre-norm model (Qwen3) the block reads N_in(h); in a post-norm model
(OLMo 2) it reads the raw residual h and writes N_ff(F), which the op executes as its output norm.

The split (derived in sign_gated.rs): silu(g) = relu(g) + e(g), e(g) = -|g| sigmoid(-|g|), even, |e| <= W(1/e) =
0.27846..., so F = P + R with the sign-gated bilinear law P = W_d (relu(g) * u) and the correction R = W_d (e(g) * u),
which a relu replacement drops exactly. The odd/even split F(x) = (F(x) + F(-x))/2 + (F(x) - F(-x))/2 was a settled
negative (its halves cancel) and is gone.

Modes:
  relusplit  per layer: the law's explained variance, |R|/|F|, cos(P, F), gate signs, units covering 90% of the law's
             hidden energy, the certified per-token enclosure of |R| against the operator bound sigma_1(W_d)|e u|,
             the post-norm write change, and the same statistics with the heaviest tokens dropped (a second op call).
  replace    the model-level KL of relu replacement on layer suffixes and prefixes (torch forward: Rust cannot own the
             rest of the network), beside each layer's op bounds; for the last layer the op also executes the final
             norm and unembedding and certifies the KL itself (verify_family), which is compared with torch's.
  single     one layer replaced at a time (torch KL), the op's certified block-level change per layer, and a
             massive-activation audit of the worst layers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpd_opfirst_linalg_2951 as la  # noqa: E402
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


def f64(t):
    return la.f64(t)


def sign_gated(layer, where, residual, change_tolerance=None, readout=None, return_rows=False):
    """The Rust op `sign_gated_swiglu` on one layer's MLP block at its residual input rows (float64 numpy).
    readout = (final_norm_module, unembedding, kl_tolerance) for a last layer."""
    from gamfit.sae import run_parameter_decomposition

    mlp = layer.mlp
    tensors = {"gate": f64(mlp.gate_proj.weight), "up": f64(mlp.up_proj.weight), "down": f64(mlp.down_proj.weight),
               "residual": residual}
    op = {"kind": "sign_gated_swiglu", "gate": "gate", "up": "up", "down": "down", "residual": "residual",
          "input_norm": None, "output_norm": None, "change_tolerance": change_tolerance, "return_rows": return_rows,
          "readout": None}
    for key, module in (("input_norm", mlp_input_norm(layer, where)), ("output_norm", mlp_output_norm(layer, where))):
        if module is not None:
            tensors[key] = f64(module.weight)
            op[key] = {"epsilon": float(module.variance_epsilon), "gain": key}
    if readout is not None:
        norm, unembedding, kl_tol = readout
        tensors["final_norm"], tensors["unembedding"] = f64(norm.weight), f64(unembedding)
        op["readout"] = {"norm": {"epsilon": float(norm.variance_epsilon), "gain": "final_norm"},
                         "unembedding": "unembedding", "tolerance": {"kl": kl_tol, "centred_logit_gap": 1e3},
                         "chunk_rows": 32}
    out = run_parameter_decomposition({"schema": "gam.mpd-request", "schema_version": 1, "operation": op}, tensors)
    return out.report["result"], out.arrays


def mid(enclosure):
    return 0.5 * (enclosure[..., 0] + enclosure[..., 1])


def q(v):
    v = np.asarray(v, dtype=np.float64)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def pair_mid(pair):
    return None if pair is None else 0.5 * (pair[0] + pair[1])


def split_entry(report, arrays, units):
    """Receipt fields of one op report (intervals kept, midpoints for the headline numbers)."""
    rows = report["per_row"]
    F, R = mid(arrays[rows["native_norm"]]), mid(arrays[rows["correction_norm"]])
    P = mid(arrays[rows["law_norm"]])
    signs = arrays[rows["gate_signs"]]
    active = signs[:, 0]
    k90 = arrays[rows["law_units90"]]
    op = arrays[rows["operator_bound"]]
    corr = arrays[rows["correction_norm"]]
    hidden = arrays[rows["correction_hidden_norm"]]
    return {
        "fve_P_centered": pair_mid(report["law_explained_variance"]),
        "fve_P_centered_interval": report["law_explained_variance"],
        "C_over_F_agg": pair_mid(report["correction_over_native"]),
        "C_over_F_agg_interval": report["correction_over_native"],
        "cos_PF_agg": pair_mid(report["law_native_cosine"]),
        "cos_PF_agg_interval": report["law_native_cosine"],
        "C_over_F_tok": q(R / F),
        "rms_ratio_P_over_F_tok": q(P / F),
        "identity_excess": report["identity_excess"],
        "enclosure_rel_width_tok_max": float(((corr[:, 1] - corr[:, 0]) / corr[:, 1]).max()),
        "operator_bound_over_F_tok": q(op / F),
        "operator_bound_over_C_tok": q(op / corr[:, 0]),
        "operator_bound_holds_all": bool((op >= corr[:, 1]).all()),
        "realized_gain_tok": q(R / hidden),
        "down_operator_norm": report["down_operator_norm"],
        "down_rms_gain": report["down_root_mean_square_gain"],
        "region_bound": report["region_bound"],
        "region_bound_over_maxC": None if report["region_bound"] is None else report["region_bound"] / float(corr[:, 1].max()),
        "region_bound_over_medianF": None if report["region_bound"] is None else report["region_bound"] / float(np.median(F)),
        "active_frac_tok": q(active / units),
        "undecided_gates_max": int(signs[:, 2].max()),
        "units90_tok": q(k90),
        "units90_over_active_tok": q(k90 / np.maximum(active, 1)),
        "change_supremum": report["change_supremum"],
    }


def write_entry(report, arrays):
    rows = report["per_row"]
    W, D = mid(arrays[rows["write_norm"]]), mid(arrays[rows["change_norm"]])
    return {"fve_NP_centered": pair_mid(report["write_explained_variance"]),
            "fve_NP_centered_interval": report["write_explained_variance"],
            "rel_err_agg": pair_mid(report["change_over_write"]),
            "rel_err_tok": q(D / W)}


class MaskedRelu(torch.nn.Module):
    """relu on the (batch, seq) tokens in mask, silu elsewhere."""

    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, g):
        return torch.where(self.mask[..., None], torch.relu(g), torch.nn.functional.silu(g))


def replace_runs(model, ids, clean_logits, configs, positions):
    """Swap silu -> relu (drop R exactly) in a layer set, rerun the full model, compare next-token distributions.

    configs: (kind, L0, layer_set, swap_mask) with swap_mask a (batch, seq) bool of tokens to swap, None for all tokens.
    """
    clean = torch.log_softmax(clean_logits, -1).reshape(-1, clean_logits.shape[-1])
    tgt = torch.cat([ids[:, 1:], torch.full((ids.shape[0], 1), -1, device=ids.device)], 1).reshape(-1)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="mps", help="torch device of the float32 forward passes")
    parser.add_argument("--n-seq", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["relusplit", "replace", "single"], default="relusplit")
    parser.add_argument("--massive-factor", type=float, default=100.0)
    parser.add_argument("--worst", type=int, default=3)
    parser.add_argument("--include-pos0", action="store_true", help="position 0 (attention sink) is excluded by default")
    parser.add_argument("--suffix-l0", default="", help="empty: L-1, L-1-s, ..., L-1-4s, 0 with s = round(3L/28)")
    parser.add_argument("--prefix-l0", default="", help="empty: the suffix starts without 0")
    parser.add_argument("--heavy", type=int, default=4)
    parser.add_argument("--readout-kl-tolerance", type=float, default=1.0,
                        help="declared per-row KL tolerance for the op's last-layer readout verification")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 dtype=torch.float32).eval().to(args.device)
    where = placement(model.config)
    ids, source = texts_tokens(tok, args.n_seq, args.seq_len, args.seed)
    ids = ids.to(args.device)
    layers = model.model.layers
    n_layers = len(layers)
    resid = {}

    def make_resid_hook(i):
        def hook(_m, inputs):
            resid[i] = f64(inputs[0].reshape(-1, inputs[0].shape[-1]))
        return hook

    # the MLP block's residual input: input of the pre-MLP norm, or of the MLP itself in a post-norm model
    handles = [(mlp_input_norm(layers[i], where) or layers[i].mlp).register_forward_pre_hook(make_resid_hook(i))
               for i in range(n_layers)]
    with torch.inference_mode():
        clean_logits = model(input_ids=ids).logits
    for h in handles:
        h.remove()
    t_fwd = time.time() - t0
    positions = torch.arange(ids.shape[1], device=args.device).repeat(ids.shape[0])
    rows = torch.ones_like(positions, dtype=torch.bool) if args.include_pos0 else positions != 0
    rows_np = rows.cpu().numpy()
    flat_ids = ids.reshape(-1)[rows].cpu()
    pos_rows = positions[rows].cpu()
    report = {"model": args.model, "revision": args.revision, "mode": args.mode, "text_source": source,
              "tokens": int(ids.numel()), "dtype": {"model": "float32 on " + args.device, "analysis": "float64 (Rust op sign_gated_swiglu)"},
              "norm_placement": where, "act": model.config.hidden_act,
              "stats_exclude_position0": not args.include_pos0, "args": vars(args), "env": la.env_record(),
              "layers": {}}
    step = max(1, round(3 * n_layers / 28))
    suffix = [int(v) for v in args.suffix_l0.split(",")] if args.suffix_l0 else [n_layers - 1 - k * step for k in range(5)] + [0]
    prefix = [int(v) for v in args.prefix_l0.split(",")] if args.prefix_l0 else [v for v in suffix if v]

    def residual(i):
        return resid[i][rows_np]

    if args.mode == "replace":
        configs = [("suffix", l0, list(range(l0, n_layers)), None) for l0 in suffix]
        configs += [("prefix", l0, list(range(0, l0)), None) for l0 in prefix]
        report["replacement"] = replace_runs(model, ids, clean_logits, configs, positions)
    if args.mode == "single":
        rows_single = replace_runs(model, ids, clean_logits, [("single", i, [i], None) for i in range(n_layers)], positions)
        worst = sorted(range(n_layers), key=lambda i: -rows_single[i]["kl_mean"])[: args.worst]
        audit = []
        for i in worst:
            r = torch.from_numpy(resid[i]).norm(dim=-1).view(ids.shape).to(args.device)
            massive = r > args.massive_factor * r.median()
            info = {"layer": i, "n_massive": int(massive.sum()),
                    "massive_positions": sorted({int(p) for p in massive.nonzero()[:, 1]}),
                    "massive_resid_over_median_min": float((r[massive] / r.median()).min()) if massive.any() else None,
                    "resid_max_over_median": float(r.max() / r.median())}
            runs = replace_runs(model, ids, clean_logits, [("ordinary_only", i, [i], ~massive),
                                                           ("massive_only", i, [i], massive)], positions)
            info["all_tokens"] = rows_single[i]
            info["ordinary_only"], info["massive_only"] = runs
            audit.append(info)
            print(f"L{i:02d} massive n={info['n_massive']} pos={info['massive_positions']} "
                  f"resid max/med={info['resid_max_over_median']:.0f}", flush=True)
        report["single_layer"] = rows_single
        report["worst_layer_massive_audit"] = audit
    for i, layer in enumerate(layers):
        units = layer.mlp.gate_proj.weight.shape[0]
        last = args.mode == "replace" and i == n_layers - 1
        readout = (model.model.norm, model.lm_head.weight, args.readout_kl_tolerance) if last else None
        res, arrays = sign_gated(layer, where, residual(i), readout=readout)
        entry = split_entry(res, arrays, units)
        if mlp_output_norm(layer, where) is not None:
            entry["post_norm_write"] = write_entry(res, arrays)
        if args.mode == "relusplit":
            F = mid(arrays[res["per_row"]["native_norm"]])
            share = F ** 2 / (F ** 2).sum()
            order = np.argsort(-share)[: args.heavy]
            entry["heavy_tokens"] = [{"row": int(j), "position": int(pos_rows[j]), "token": tok.decode([int(flat_ids[j])]),
                                      "share_of_sum_F2": float(share[j]), "F_norm": float(F[j]),
                                      "F_norm_over_median": float(F[j] / np.median(F))} for j in order]
            keep = np.ones(F.shape[0], dtype=bool)
            keep[order] = False
            light, _ = sign_gated(layer, where, residual(i)[keep])
            entry["pooled_drop_heavy"] = {"fve_P_centered": pair_mid(light["law_explained_variance"]),
                                          "cos_PF": pair_mid(light["law_native_cosine"])}
        if readout is not None:
            ro = res["readout"]
            fwd = arrays[ro["forward_kl_rows"]]
            torch_last = next(r for r in report["replacement"] if r["kind"] == "suffix" and r["L0"] == n_layers - 1)
            entry["readout"] = {"certified": ro["certified"], "forward_kl_supremum": ro["forward_kl"],
                                "kl_mean_interval": [float(fwd[:, 0].mean()), float(fwd[:, 1].mean())],
                                "kl_median": float(np.median(mid(fwd))),
                                "kl_rows_resolved": int(arrays[ro["kl_resolved"]].sum()),
                                "top1_agree": 1.0 - ro["argmax_disagreeing"] / ro["argmax_rows"],
                                "torch_kl_mean": torch_last["kl_mean"], "torch_top1_agree": torch_last["top1_agree"]}
        report["layers"][i] = entry
        print(f"L{i:02d} fveP={entry['fve_P_centered']:.4f} |R|/|F|={entry['C_over_F_agg']:.3f} "
              f"tokmed={entry['C_over_F_tok']['median']:.3f} op/|R|={entry['operator_bound_over_C_tok']['median']:.2f} "
              f"gain/rms={entry['realized_gain_tok']['median'] / entry['down_rms_gain']:.2f} "
              f"enc_w={entry['enclosure_rel_width_tok_max']:.1e} act={entry['active_frac_tok']['median']:.3f} "
              f"k90={entry['units90_tok']['median']:.0f} id={entry['identity_excess']:.1e}"
              + (f" readoutKL={entry['readout']['kl_mean_interval']} torchKL={entry['readout']['torch_kl_mean']:.4e}"
                 if readout is not None else ""), flush=True)
    report["runtime_s"] = {"forward_and_load": t_fwd, "total": time.time() - t0}
    with open(args.out, "w") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}"
                                     for k, v in report.items()) + "\n}\n")
    print("runtime", report["runtime_s"])


if __name__ == "__main__":
    main()
