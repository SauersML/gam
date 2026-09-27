"""Residual-stream observability of Qwen3-0.6B-Base as an exact linear system (#2951).

Readout C = W_U diag(g_final) (tied unembedding times the final RMSNorm gain; the
RMSNorm normalizer 1/rms(x) is a scalar gain that is excluded, i.e. treated as a
fixed positive scale). Transitions with attention patterns held fixed: for every
layer l and query head h, T = I + A_{l,h} with A_{l,h} = W_O[:, h] W_V[g(h)] diag(g_in,l)
(GQA: head h reads kv group h // 2; the input RMSNorm gain folded in as a diagonal).
MLPs are nonlinear; their read directions W_gate diag(g_post), W_up diag(g_post) are
added as extra readouts in the "+mlp" variant.

Three measurements:
1. rowspace(C) at the state.rs rank rule (the owner's resolved row space through the MPD
   surface, ``linear_state_quotient``), plus relative-threshold effective dimensions.
2. state.rs LinearStateQuotient::close itself (op ``linear_state_quotient``): the orthonormal
   chart of the time-invariant family of all 448 OV maps, with its measured quotient bounds.
3. Causal backward closure O_L = O_{L+1} + sum_h O_{L+1} A_{L,h} (+ MLP reads of layer L),
   kept both as an orthonormal chart at a relative band tau and as an unnormalized square-root
   Gramian factor F_L (F_L^T F_L = F_{L+1}^T F_{L+1} + sum_h A^T F_{L+1}^T F_{L+1} A + reads),
   whose singular values give the decay of how strongly each residual direction is read.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file



def compact_json(obj):
    """One top-level key per line, compact values: keeps receipts under the
    repository's tracked-file line limit (build.rs MAX_TRACKED_FILE_LINES)."""
    if isinstance(obj, dict):
        body = ",\n".join(
            json.dumps(k) + ": " + json.dumps(v, separators=(",", ":")) for k, v in obj.items()
        )
        return "{\n" + body + "\n}\n"
    return json.dumps(obj, separators=(",", ":")) + "\n"


def linear_quotient(readouts, transitions=(), chart=None):
    """state.rs ``LinearStateQuotient`` through the MPD surface (op ``linear_state_quotient``): ``close`` (the
    readouts' resolved row span closed under the transitions) or, given ``chart``, ``measure`` of that chart.
    Returns the chart Q (orthonormal rows) and the report, whose SpectralNormBounds certify the quotient."""
    from gamfit.sae import run_parameter_decomposition

    tensors = {f"readout/{i}": r for i, r in enumerate(readouts)}
    tensors.update({f"transition/{i}": t for i, t in enumerate(transitions)})
    declared = {"kind": "close"}
    if chart is not None:
        tensors["chart"] = chart
        declared = {"kind": "declared", "tensor": "chart"}
    operation = {"kind": "linear_state_quotient", "readouts": [f"readout/{i}" for i in range(len(readouts))],
                 "transitions": [f"transition/{i}" for i in range(len(transitions))], "chart": declared}
    out = run_parameter_decomposition({"schema": "gam.mpd-request", "schema_version": 1, "operation": operation},
                                      tensors)
    return out.arrays["chart"], out.report["result"]


def resolved_rows(matrix):
    """state.rs's resolved row space of one matrix (sigma > max(m, n) eps sigma_max) as orthonormal rows: the
    owner's quotient of ``matrix`` under no transitions. ``.shape[0]`` is the rank at the eps band."""
    return linear_quotient([matrix])[0]


def relative_rows(matrix, tau):
    """Right singular vectors above tau * sigma_max: a numerical conditioning statement, never a rank."""
    if matrix.shape[0] > 2 * matrix.shape[1]:
        matrix = np.linalg.qr(matrix, mode="r")
    _, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt[: int((sigma > tau * sigma[0]).sum())]


def singular_values(matrix):
    if matrix.shape[0] > 2 * matrix.shape[1]:
        matrix = np.linalg.qr(matrix, mode="r")
    return np.linalg.svd(matrix, compute_uv=False)


def effective(sigma, taus=(1e-2, 1e-3, 1e-6)):
    out = {f"{t:g}": int((sigma > t * sigma[0]).sum()) for t in taus}
    p = sigma**2 / (sigma**2).sum()
    out["entropy_rank"] = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
    out["participation_ratio"] = float((sigma**2).sum() ** 2 / (sigma**4).sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--out", default="experiments/issue-2951/receipts/opfirst_observability.json")
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(os.cpu_count())
    snap = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*/"))[0]
    cfg = json.load(open(os.path.join(snap, "config.json")))
    w = {k: v.double().numpy() for k, v in load_file(os.path.join(snap, "model.safetensors")).items()}
    d, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    n_heads, n_kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    group = n_heads // n_kv

    g_final = w["model.norm.weight"]
    readout = w["model.embed_tokens.weight"] * g_final[None, :]  # tied
    ov, mlp_reads = [], []
    for layer in range(n_layers):
        p = f"model.layers.{layer}."
        g_in = w[p + "input_layernorm.weight"]
        g_post = w[p + "post_attention_layernorm.weight"]
        wv, wo = w[p + "self_attn.v_proj.weight"], w[p + "self_attn.o_proj.weight"]
        ov.append([wo[:, h * hd:(h + 1) * hd] @ wv[(h // group) * hd:(h // group + 1) * hd] * g_in[None, :]
                   for h in range(n_heads)])
        mlp_reads.append(np.vstack([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]]) * g_post[None, :])
    load_s = time.time() - started

    # 1. rowspace(C)
    rank_c = int(resolved_rows(readout).shape[0])
    s_c = singular_values(readout)
    weak = np.argsort(np.abs(g_final))[:5]
    result = {
        "model": args.model,
        "conventions": {
            "readout": "C = W_U diag(g_final), tied embeddings, RMSNorm normalizer excluded (fixed positive scalar)",
            "transition": "A_{l,h} = W_O[:, h] W_V[h//2] diag(g_in,l); T = I + A; closing under T equals closing under A",
            "mlp_reads": "rows of W_gate diag(g_post), W_up diag(g_post) added as readouts at the post-attention residual",
            "rank_rule": "state.rs resolved_row_space and LinearStateQuotient::close, called through the MPD surface "
                         "(linear_state_quotient); 'tau' charts keep the singular directions above tau sigma_max",
            "labels": "rank at the eps band is exact-arithmetic rank to within roundoff (numerical-exact); "
                      "tau and effective dimensions are numerical/conditioning statements",
        },
        "readout": {
            "shape": list(readout.shape), "rank_at_band": rank_c,
            "sigma_max": float(s_c[0]), "sigma_min": float(s_c[-1]), "sigma_min_over_max": float(s_c[-1] / s_c[0]),
            "sigma_top5": s_c[:5].tolist(), "sigma_bottom8": s_c[-8:].tolist(),
            "effective": effective(s_c),
            "smallest_final_gain_coords": {int(i): float(g_final[i]) for i in weak},
        },
    }

    # 2. state.rs's closure under all OV maps (time-invariant family)
    all_ov = [a for layer in ov for a in layer]

    def closure(readouts):
        chart, report = linear_quotient(readouts, all_ov)
        return {"rank_closed": int(chart.shape[0]),
                "max_quotient_bound": max(b["upper"] for b in report["quotient_bounds"]),
                "max_readout_bound": max(b["upper"] for b in report["readout_bounds"]),
                "section_bound": report["section_bounds"]["upper"]}
    result["time_invariant_closure"] = {
        "attention_only": {"rank_C": rank_c, **closure([readout])},
        "with_mlp_reads": {"rank_C_plus_mlp_reads": int(resolved_rows(np.vstack([readout] + mlp_reads)).shape[0]),
                           **closure([readout] + mlp_reads)}}

    # 3. causal backward closure per layer
    per_layer = {"attention_only": [], "with_mlp_reads": []}
    for variant in per_layer:
        factor = np.linalg.qr(readout, mode="r")
        charts = {tau: relative_rows(readout, tau) for tau in (1e-2, 3e-2)}
        for layer in reversed(range(n_layers)):
            if variant == "with_mlp_reads":
                factor = np.linalg.qr(np.vstack([factor, mlp_reads[layer]]), mode="r")
                for tau in charts:
                    charts[tau] = relative_rows(np.vstack([charts[tau], mlp_reads[layer] /
                                                           np.linalg.norm(mlp_reads[layer], 2)]), tau)
            factor = np.linalg.qr(np.vstack([factor] + [factor @ a for a in ov[layer]]), mode="r")
            for tau in charts:
                charts[tau] = relative_rows(np.vstack([charts[tau]] + [charts[tau] @ a for a in ov[layer]]), tau)
            rank = int(resolved_rows(factor).shape[0])
            _, s, vt = np.linalg.svd(factor)
            per_layer[variant].append({
                "layer": layer, "rank_at_band": rank, "sigma_min_over_max": float(s[-1] / s[0]),
                "effective": effective(s),
                "chart_rank_tau_1e-2": int(charts[1e-2].shape[0]), "chart_rank_tau_3e-2": int(charts[3e-2].shape[0]),
                "weakest_dir_top_coords": {int(i): float(vt[-1, i]) for i in np.argsort(-np.abs(vt[-1]))[:3]},
                "sigma_bottom4_over_max": (s[-4:] / s[0]).tolist(),
            })
        per_layer[variant].reverse()
    result["causal_backward"] = per_layer
    result["ov_head_norm_range"] = [float(min(np.linalg.norm(a, 2) for a in all_ov)),
                                    float(max(np.linalg.norm(a, 2) for a in all_ov))]
    result["runtime_s"] = {"load": load_s, "total": time.time() - started}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(compact_json(result))
    print(json.dumps({k: result[k] for k in ("readout", "time_invariant_closure", "runtime_s")}, indent=1))
    for variant, rows in per_layer.items():
        print(variant)
        for r in rows:
            print(r["layer"], r["rank_at_band"], f"{r['sigma_min_over_max']:.2e}", r["effective"]["0.01"],
                  r["effective"]["0.001"], f"{r['effective']['entropy_rank']:.0f}", f"{r['effective']['participation_ratio']:.1f}",
                  r["chart_rank_tau_1e-2"], r["chart_rank_tau_3e-2"], r["weakest_dir_top_coords"])


if __name__ == "__main__":
    main()
