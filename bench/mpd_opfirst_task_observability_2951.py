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
* ``weekday``: a decoder read through bench/mpd_opfirst_decoder_2951.py (default OLMo 2 1B; the
  architecture comes from config), C = the 7 " Monday".." Sunday" rows of the unembedding times the final
  RMSNorm gain (and the 6-dimensional difference space), backward causal closure layer by layer under all
  heads' OV maps, kept factored (the input RMSNorm gain folded in a pre-norm model; the attention-output
  RMSNorm gain in a post-norm model, whose excluded normaliser is one scalar per token and layer shared by
  all heads), optionally with the MLP reads.

The rank rule and the closure are state.rs's resolved row space and LinearStateQuotient::close, and the
Fourier planes are cyclic_action::cyclic_planes, all called through the MPD surface (the parent bench's
``linear_quotient`` / ``resolved_rows``; ``cyclic_planes`` here). "exact" = rank at the eps band (exact-arithmetic
rank to within roundoff); tau / effective dimensions are numerical conditioning statements. The
Gramian factor F (F^T F = sum of pulled-back readout Gramians) weighs directions by how strongly
they are read, so its relative singular-value counts are the meaningful effective dimensions.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import Decoder  # noqa: E402
from mpd_opfirst_observability_2951 import (  # noqa: E402
    apply, effective, linear_quotient, relative_rows, resolved_rows, singular_values)

TAUS = (1e-2, 1e-3, 1e-6)


def dims(factor):
    sigma = singular_values(factor)
    return {"exact_rank_eps_band": int(resolved_rows(factor).shape[0]), **effective(sigma, TAUS)}, sigma


def top_space(factor, tau):
    _, sigma, vt = np.linalg.svd(factor, full_matrices=False)
    return vt[: int((sigma > tau * sigma[0]).sum())]


def principal_cosines(a, b):
    """Cosines of the principal angles between the row spaces of orthonormal-row a and b."""
    return np.linalg.svd(a @ b.T, compute_uv=False)


def cyclic_planes(table):
    """cyclic_action::cyclic_planes of ``table``'s rows under the token cycle x -> x + 1 (mod p), through the MPD
    surface: the planes [u_1c, u_1s, ...] (d x 2m) and each plane's power."""
    from gamfit.sae import run_parameter_decomposition

    p = table.shape[0]
    out = run_parameter_decomposition(
        {"schema": "gam.mpd-request", "schema_version": 1,
         "operation": {"kind": "cyclic_planes", "table": "table", "successor": [(x + 1) % p for x in range(p)]}},
        {"table": table})
    return out.arrays["planes"], np.array(out.report["result"]["power"])


def fourier_planes(table, freqs):
    planes, _ = cyclic_planes(table)
    return resolved_rows(np.vstack([planes[:, [2 * (k - 1), 2 * (k - 1) + 1]].T for k in freqs]))


def fourier_power(table):
    _, power = cyclic_planes(table)
    return np.arange(1, len(power) + 1), power / power.sum()


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
    with open(args.run, "rb") as fh:
        sha256 = hashlib.sha256(fh.read()).hexdigest()
    res = {"checkpoint": {"path": os.path.basename(args.run), "sha256": sha256},
           "step": step, "config": cfg, "final_test_acc": run["curves"]["test_acc"][-1],
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
            chart, report = linear_quotient([post], ov)
            v["time_invariant_closure_eps_band"] = {
                "rank_C": int(resolved_rows(post).shape[0]), "rank_closed": int(chart.shape[0]),
                "max_quotient_bound": max(b["upper"] for b in report["quotient_bounds"])}
            v["fourier"] = {pn: compare(factor, pl, f"[modadd] {name} vs {pn}") for pn, pl in planes.items()}
            v["fourier"]["readout_only"] = {pn: compare(np.linalg.qr(post, mode="r"), pl,
                                                        f"[modadd] {name} readout-only vs {pn}")
                                            for pn, pl in planes.items()}
            print(f"[modadd] {name} readout {v['readout']} pre-attn {v['pre_attention_gramian']}", flush=True)
            res["variants"][name] = v
    return res


def weekday(args):
    from tokenizers import Tokenizer

    D = Decoder(args.model, args.revision)
    tok = Tokenizer.from_file(os.path.join(D.snap, "tokenizer.json"))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ids = []
    for day in days:
        t = tok.encode(" " + day, add_special_tokens=False).ids
        assert len(t) == 1, day
        ids.append(t[0])
    d, n_layers = D.d, D.L
    rows = D.readout(ids)
    readouts = {"answers": rows, "differences": rows - rows.mean(0, keepdims=True)}
    ov = [D.ov(layer) for layer in range(n_layers)]
    res = {"model": args.model, "architecture": D.describe(), "day_token_ids": ids, "d_model": d, "variants": {}}
    for rname, c in readouts.items():
        for mlp in (False, True):
            name = rname + ("+mlp_reads" if mlp else "")
            factor = c
            charts = {tau: resolved_rows(c) if tau is None else relative_rows(c, tau) for tau in (None, 1e-3, 1e-6)}
            per_layer = [{"layer": n_layers, **dims(factor)[0],
                          **{f"chart_rank_{t if t else 'eps'}": int(ch.shape[0]) for t, ch in charts.items()}}]
            for layer in reversed(range(n_layers)):
                if mlp:
                    reads = D.mlp_reads(layer)
                    factor = np.vstack([factor, reads])
                    for t in charts:
                        stack = np.vstack([charts[t], reads / np.linalg.norm(reads, 2)])
                        charts[t] = resolved_rows(stack) if t is None else relative_rows(stack, t)
                factor = np.vstack([factor] + [apply(factor, a) for a in ov[layer]])
                if factor.shape[0] > d:
                    factor = np.linalg.qr(factor, mode="r")
                for t in charts:
                    if charts[t].shape[0] < d:
                        stack = np.vstack([charts[t]] + [apply(charts[t], a) for a in ov[layer]])
                        charts[t] = resolved_rows(stack) if t is None else relative_rows(stack, t)
                entry, sigma = dims(factor)
                entry.update({"layer": layer, "sigma_over_max_first8": (sigma[:8] / sigma[0]).tolist(),
                              **{f"chart_rank_{t if t else 'eps'}": int(ch.shape[0]) for t, ch in charts.items()}})
                per_layer.append(entry)
                with open(args.out_prefix + "_weekday.partial.json", "w") as fh:  # progress survives a stop
                    json.dump(res | {"partial": {name: per_layer}}, fh, separators=(",", ":"))
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
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", default="main")
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
        path = f"{args.out_prefix}_{name}" + (f"_{args.model.split('/')[-1]}" if name == "weekday" else "") + ".json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:  # one top-level key per line (tracked files must stay < 10k lines)
            fh.write("{\n" + ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}"
                                        for k, v in res.items()) + "\n}\n")
        print(f"[{name}] wrote {path} in {res['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
