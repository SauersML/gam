"""Task-restricted residual-stream observability (#2951).

bench/mpd_opfirst_observability_2951.py closed the residual stream's observable subspace under all
attention OV maps with the full unembedding as readout; that is trivial for Qwen3-0.6B because the
151,936-row W_U already has rank d_model. Here the readout is a TASK's answer rows only:

* ``modadd``: the one-layer p=113 transformer of bench/mpd_modadd_2951.py (a ``train`` run file).
  C = the 113 answer rows of W_U, and separately the 112 logit differences against the correct token
  (span{W_U[j] - W_U[c]} is the row space of the centered W_U, the same for every c: it kills the
  constant direction). Transitions: the 4 head OV maps A_h = W_O[:, h] W_V[h]; MLP read rows W_in
  as extra readouts at the post-attention residual in a separate variant. The observable subspace is
  compared with the key-frequency Fourier planes of W_U and W_E (principal angles).
* ``weekday``: Qwen3-0.6B-Base, C = the 7 " Monday".." Sunday" rows of the tied unembedding times the
  final RMSNorm gain (and the 6-dimensional difference space), backward causal closure layer by layer
  under all 16 heads' OV maps (input RMSNorm gain folded), optionally with the MLP reads.

The rank rule, closure and Gramian recursion are those of the parent bench (mirroring state.rs
resolved_row_space and LinearStateQuotient::close). "exact" = rank at the eps band (exact-arithmetic
rank to within roundoff); tau / effective dimensions are numerical conditioning statements. The
Gramian factor F (F^T F = sum of pulled-back readout Gramians) weighs directions by how strongly
they are read, so its relative singular-value counts are the meaningful effective dimensions.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_observability_2951 import close, effective, resolved  # noqa: E402

TAUS = (1e-2, 1e-3, 1e-6)


def dims(factor):
    _, sigma, _, rank = resolved(factor, 0.0)
    return {"exact_rank_eps_band": rank, **effective(sigma, TAUS)}, sigma


def top_space(factor, tau):
    _, sigma, vt = np.linalg.svd(factor, full_matrices=False)
    return vt[: int((sigma > tau * sigma[0]).sum())]


def principal_cosines(a, b):
    """Cosines of the principal angles between the row spaces of orthonormal-row a and b."""
    return np.linalg.svd(a @ b.T, compute_uv=False)


def orth_rows(m):
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    return vt[: int((s > max(m.shape) * np.finfo(float).eps * s[0]).sum())]


def fourier_planes(table, freqs):
    p = table.shape[0]
    a = np.arange(p)
    rows = []
    for k in freqs:
        rows += [np.cos(2 * math.pi * k * a / p) @ table, np.sin(2 * math.pi * k * a / p) @ table]
    return orth_rows(np.array(rows))


def fourier_power(table):
    p = table.shape[0]
    a = np.arange(p)
    ks = np.arange(1, (p - 1) // 2 + 1)
    power = np.array([np.sum((np.cos(2 * math.pi * k * a / p) @ table) ** 2 + (np.sin(2 * math.pi * k * a / p) @ table) ** 2)
                      for k in ks])
    return ks, power / power.sum()


def compare(factor, planes, label):
    """How the Gramian-weighted observable space sits against a Fourier-plane subspace."""
    gram = factor.T @ factor
    out = {"plane_dim": int(planes.shape[0]),
           "gramian_energy_in_planes": float(np.trace(planes @ gram @ planes.T) / np.trace(gram))}
    for tau in (1e-2, 1e-3):
        top = top_space(factor, tau)
        cos = principal_cosines(top, planes)
        out[f"tau_{tau:g}"] = {"observable_dim": int(top.shape[0]), "principal_cosines": cos.round(6).tolist(),
                               "planes_contained_frac": float((cos ** 2).sum() / planes.shape[0])}
    top = top_space(factor, 0.0)[: planes.shape[0]]
    out["top_equal_dim_cosines"] = principal_cosines(top, planes).round(6).tolist()
    print(label, json.dumps({k: v for k, v in out.items() if k != "top_equal_dim_cosines"}), flush=True)
    return out


def modadd(args):
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    cfg = run["config"]
    step = max(run["checkpoints"])
    s = {k: v.double().numpy() for k, v in run["checkpoints"][step].items()}
    p, n_heads = cfg["p"], cfg["n_heads"]
    w_u, w_e, w_in = s["W_U"], s["W_E"][:p], s["W_in"]
    ov = [s["W_O"][:, h * cfg["d_head"]:(h + 1) * cfg["d_head"]] @ s["W_V"][h] for h in range(n_heads)]
    ks, pu = fourier_power(w_u)
    _, pe = fourier_power(w_e)
    key = [int(k) for k in ks[np.argsort(-pu)[: args.kmax]]]
    readouts = {"answers": w_u, "differences": w_u - w_u.mean(0, keepdims=True)}
    planes = {"W_U_key": fourier_planes(w_u, key), "W_E_key": fourier_planes(w_e, key)}
    res = {"step": step, "config": cfg, "final_test_acc": run["curves"]["test_acc"][-1],
           "key_frequencies_W_U": key, "W_U_power_top": [[int(k), float(x)] for k, x in
                                                          zip(ks[np.argsort(-pu)[:8]], np.sort(pu)[::-1][:8])],
           "W_E_power_top": [[int(k), float(x)] for k, x in zip(ks[np.argsort(-pe)[:8]], np.sort(pe)[::-1][:8])],
           "ov_norms": [float(np.linalg.norm(a, 2)) for a in ov], "variants": {}}
    for rname, c in readouts.items():
        for mlp in (False, True):
            name = rname + ("+mlp_reads" if mlp else "")
            post = np.vstack([c, w_in]) if mlp else c  # read at the post-attention residual
            v = {"readout": dims(c)[0], "readout_at_post_attn": dims(post)[0]}
            # one layer: the pre-attention residual is read through I + A_h (attention pattern fixed)
            factor = np.linalg.qr(np.vstack([post] + [post @ a for a in ov]), mode="r")
            v["pre_attention_gramian"], sigma = dims(factor)
            v["pre_attention_sigma_over_max_first24"] = (sigma[:24] / sigma[0]).tolist()
            for tau_label, tau in (("eps_band", None), ("tau_1e-3", 1e-3), ("tau_1e-6", 1e-6)):
                chart0, _, _, r0 = resolved(post, 0.0, tau)
                chart, steps = close(chart0, ov, tau)
                v[f"time_invariant_closure_{tau_label}"] = {"rank_C": r0, "rank_closed": int(chart.shape[0]),
                                                            "steps": steps}
            v["fourier"] = {pn: compare(factor, pl, f"[modadd] {name} vs {pn}") for pn, pl in planes.items()}
            v["fourier"]["readout_only"] = {pn: compare(np.linalg.qr(post, mode="r"), pl,
                                                        f"[modadd] {name} readout-only vs {pn}")
                                            for pn, pl in planes.items()}
            print(f"[modadd] {name} readout {v['readout']} pre-attn {v['pre_attention_gramian']}", flush=True)
            res["variants"][name] = v
    return res


def weekday(args):
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    snap = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*/"))[0]
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ids = []
    for day in days:
        t = tok.encode(" " + day, add_special_tokens=False).ids
        assert len(t) == 1, day
        ids.append(t[0])
    w = {k: v.double().numpy() for k, v in load_file(os.path.join(snap, "model.safetensors")).items()}
    d, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    n_heads, n_kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    group = n_heads // n_kv
    rows = w["model.embed_tokens.weight"][ids] * w["model.norm.weight"][None, :]
    readouts = {"answers": rows, "differences": rows - rows.mean(0, keepdims=True)}
    ov, mlp_reads = [], []
    for layer in range(n_layers):
        pre = f"model.layers.{layer}."
        g_in = w[pre + "input_layernorm.weight"]
        wv, wo = w[pre + "self_attn.v_proj.weight"], w[pre + "self_attn.o_proj.weight"]
        ov.append([wo[:, h * hd:(h + 1) * hd] @ wv[(h // group) * hd:(h // group + 1) * hd] * g_in[None, :]
                   for h in range(n_heads)])
        mlp_reads.append(np.vstack([w[pre + "mlp.gate_proj.weight"], w[pre + "mlp.up_proj.weight"]])
                         * w[pre + "post_attention_layernorm.weight"][None, :])
    del w
    res = {"model": args.model, "day_token_ids": ids, "d_model": d, "variants": {}}
    for rname, c in readouts.items():
        for mlp in (False, True):
            name = rname + ("+mlp_reads" if mlp else "")
            factor = c
            charts = {tau: resolved(c, 0.0, tau)[0] for tau in (None, 1e-3, 1e-6)}
            per_layer = [{"layer": n_layers, **dims(factor)[0],
                          **{f"chart_rank_{t if t else 'eps'}": int(ch.shape[0]) for t, ch in charts.items()}}]
            for layer in reversed(range(n_layers)):
                if mlp:
                    factor = np.vstack([factor, mlp_reads[layer]])
                    for t in charts:
                        charts[t] = resolved(np.vstack([charts[t], mlp_reads[layer] /
                                                        np.linalg.norm(mlp_reads[layer], 2)]), 0.0, t)[0]
                factor = np.vstack([factor] + [factor @ a for a in ov[layer]])
                if factor.shape[0] > d:
                    factor = np.linalg.qr(factor, mode="r")
                for t in charts:
                    if charts[t].shape[0] < d:
                        charts[t] = resolved(np.vstack([charts[t]] + [charts[t] @ a for a in ov[layer]]), 0.0, t)[0]
                entry, sigma = dims(factor)
                entry.update({"layer": layer, "sigma_over_max_first8": (sigma[:8] / sigma[0]).tolist(),
                              **{f"chart_rank_{t if t else 'eps'}": int(ch.shape[0]) for t, ch in charts.items()}})
                per_layer.append(entry)
                print(f"[weekday] {name} L{layer} exact={entry['exact_rank_eps_band']} "
                      f"eff1e-2={entry['0.01']} eff1e-3={entry['0.001']} eff1e-6={entry['1e-06']} "
                      f"PR={entry['participation_ratio']:.1f} charts={[e.shape[0] for e in charts.values()]}",
                      flush=True)
            res["variants"][name] = per_layer
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="bench/mpd_modadd_2951.py train output (.pt)")
    parser.add_argument("--kmax", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--tasks", default="modadd,weekday")
    parser.add_argument("--out-prefix", default="experiments/issue-2951/receipts/opfirst_task_observability")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    for name, fn in (("modadd", modadd), ("weekday", weekday)):
        if name not in args.tasks.split(",") or (name == "modadd" and not args.run):
            continue
        started = time.time()
        res = fn(args)
        res["runtime_s"] = time.time() - started
        res["labels"] = ("exact_rank_eps_band / chart_rank_eps / time_invariant eps_band: state.rs eps band "
                         "(numerical-exact); tau, 0.01/0.001/1e-06 relative counts, entropy_rank, "
                         "participation_ratio: numerical (Gramian-weighted)")
        path = f"{args.out_prefix}_{name}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:  # one top-level key per line (tracked files must stay < 10k lines)
            fh.write("{\n" + ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}"
                                        for k, v in res.items()) + "\n}\n")
        print(f"[{name}] wrote {path} in {res['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
