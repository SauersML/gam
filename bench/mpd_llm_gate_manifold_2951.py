"""#2951 probe: is the manifold of weights a SwiGLU MLP actually uses per token low-dimensional?

Analysis under SPEC 8's exception (torch execution of a measurement).

A Qwen3 MLP is exactly ``y = W_eff(x) x`` with ``W_eff(x) = W_down diag(s(x)) W_up`` and ``s(x) = silu(W_gate x)``.
The token's used weights are the point ``W_eff(x)`` of the manifold ``{W_down diag(s) W_up : s in S}``, and the gate
state ``s(x)`` is its coordinate (``s -> W_eff`` is linear and injective on the neurons ``W_down``, ``W_up`` keep).
Here ``s(x)`` alone is replaced by a reconstruction ``s_hat`` while the read ``W_up x`` and write ``W_down`` stay exact,
``y_hat = W_down (s_hat * W_up x)``, and the next-token KL to the clean model is measured on held-out tokens:

* ``atlas C x d``: ``s`` is assigned to its nearest of ``C`` k-means centres of the bank's gate states and replaced by
  ``mu_c + P_{c,d}(s - mu_c)``, the chart's own top-``d`` local principal subspace;
* ``random C x d``: the same with a random partition (equal parameter count, no geometry);
* ``global r``: the bank's top-``r`` principal subspace of ``s``;
* ``neurons k``: the ``k`` gates with the largest ``|s_n - mean_n| * ||W_down[:, n] (W_up x)_n||``... kept, the rest at
  their bank mean (the sparse neuron-aligned basis, per token);
* ``mean``: ``s_hat`` = the bank mean (every token uses one fixed matrix).
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from mpd_llm_chart_restriction_2951 import chart_project, kmeans, local_frames, nearest, token_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--bank-tokens", type=int, required=True)
    parser.add_argument("--eval-batches", type=int, required=True)
    parser.add_argument("--charts", required=True)
    parser.add_argument("--dims", required=True)
    parser.add_argument("--global-ranks", required=True)
    parser.add_argument("--neurons", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    stream = token_batches(tok, args.seq_len, args.eval_batch, args.seed, 0)
    held = [next(stream) for _ in range(args.eval_batches)]
    charts = [int(c) for c in args.charts.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    ranks = [int(r) for r in args.global_ranks.split(",")]
    ks = [int(k) for k in args.neurons.split(",")]
    state = {"mode": None}
    report = {"args": vars(args), "layers": {}}
    with torch.inference_mode():
        clean = [torch.log_softmax(model(input_ids=h.to(dev)).logits.float(), -1).cpu() for h in held]

    def kl_under(fn):
        state["mode"] = fn
        tot, n = 0.0, 0
        with torch.inference_mode():
            for h, ref in zip(held, clean):
                lp = torch.log_softmax(model(input_ids=h.to(dev)).logits.float(), -1)
                r = ref.to(dev)
                tot += (r.exp() * (r - lp)).sum().item()
                n += r.shape[0] * r.shape[1]
        state["mode"] = None
        return tot / n

    for layer in [int(x) for x in args.layers.split(",")]:
        mlp = model.model.layers[layer].mlp
        Wd = mlp.down_proj.weight.float()
        bank = []

        def hook(_m, inputs, output):
            x = inputs[0]
            flat = x.reshape(-1, x.shape[-1])
            s = F.silu(mlp.gate_proj(flat).float())
            if state["mode"] == "grab":
                bank.append(s)
                return None
            if callable(state["mode"]):
                u = mlp.up_proj(flat).float()
                s_hat = state["mode"](s, u)
                return ((s_hat * u) @ Wd.T).reshape(output.shape).to(output.dtype)
            return None

        handle = mlp.register_forward_hook(hook)
        state["mode"] = "grab"
        feed = token_batches(tok, args.seq_len, args.batch, args.seed + 1, 0)
        with torch.inference_mode():
            while sum(b.shape[0] for b in bank) < args.bank_tokens:
                model(input_ids=next(feed).to(dev))
        state["mode"] = None
        S = torch.cat(bank)[: args.bank_tokens]
        bank.clear()
        mu = S.mean(0)
        entry = {"width": S.shape[1], "global": {}, "atlas": {}, "random": {}, "neurons": {}}
        entry["exact_rewrite"] = kl_under(lambda s, u: s)
        entry["mean"] = kl_under(lambda s, u: mu.expand_as(s))
        print(f"[gate] layer {layer} exact rewrite KL {entry['exact_rewrite']:.2e}  mean gate KL {entry['mean']:.4f}",
              flush=True)
        dnorm = Wd.norm(dim=0)
        for k in ks:
            def top(s, u, k=k):
                score = ((s - mu) * u).abs() * dnorm
                keep = torch.zeros_like(s).scatter_(1, score.topk(k, dim=1).indices, 1.0)
                return mu + (s - mu) * keep
            entry["neurons"][k] = kl_under(top)
            print(f"[gate] layer {layer} top-{k} neurons: KL {entry['neurons'][k]:.4f}", flush=True)
        _, _, Vt = torch.linalg.svd(S - mu, full_matrices=False)
        for r in ranks:
            P = Vt[:r]
            entry["global"][r] = kl_under(lambda s, u, P=P: mu + ((s - mu) @ P.T) @ P)
            print(f"[gate] layer {layer} global r={r}: KL {entry['global'][r]:.4f}", flush=True)
        del Vt
        gen = torch.Generator().manual_seed(args.seed)
        dmax = max(dims)
        for C in charts:
            centres, assign = kmeans(S, C, 25, gen)
            mus, frames = local_frames(S, assign, C, dmax)
            rand_assign = torch.randint(0, C, (S.shape[0],), generator=gen).to(dev)
            rmus, rframes = local_frames(S, rand_assign, C, dmax)
            for d in dims:
                entry["atlas"][f"{C}x{d}"] = kl_under(
                    lambda s, u, d=d: chart_project(s, nearest(s, centres), mus, frames[:, :d]))
                entry["random"][f"{C}x{d}"] = kl_under(
                    lambda s, u, d=d: chart_project(s, torch.randint(0, C, (s.shape[0],), device=s.device), rmus,
                                                    rframes[:, :d]))
                print(f"[gate] layer {layer} C={C} d={d}: atlas KL {entry['atlas'][f'{C}x{d}']:.4f}  "
                      f"random KL {entry['random'][f'{C}x{d}']:.4f}", flush=True)
            del frames, rframes
        report["layers"][layer] = entry
        handle.remove()
        del S
        torch.cuda.empty_cache()
        with open(args.out, "w") as handle_out:
            json.dump(report, handle_out, indent=1)
    print(f"[gate] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
