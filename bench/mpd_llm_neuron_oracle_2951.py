"""#2951 probe: is a Qwen3 MLP sparse per token in its OWN overcomplete dictionary, its SwiGLU neurons?

Analysis under SPEC 8's exception; companion of ``mpd_llm_token_oracle_2951.py`` (no orthonormal basis of the MLP
input gives a sparse per-token read). The MLP is ``y = sum_n h_n(x) w_down_n`` with ``h_n = silu(g_n . x)(u_n . x)``;
neuron ``n`` is one rank-1 piece in each of gate, up and down. Per token keep ``d`` neurons and give every other
neuron its bank-mean activation (mean ablation, the same convention as the other probes):

* ``magnitude``: the ``d`` largest ``|h_n - hbar_n| ||w_down_n||``;
* ``attribution``: the ``d`` largest ``E_s[((h_n - hbar_n) g_s . w_down_n)^2]``, ``g_s`` the gradient of a
  sampled-label log-probability of the whole sequence at the MLP output (second-order Fisher diagonal, as before).

Reported: next-token KL to the clean model on held-out tokens, and the constant-mean-output baseline.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from mpd_llm_chart_restriction_2951 import token_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--bank-tokens", type=int, required=True)
    parser.add_argument("--eval-batches", type=int, required=True)
    parser.add_argument("--dims", required=True)
    parser.add_argument("--samples", type=int, required=True)
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
    for prm in model.parameters():
        prm.requires_grad_(False)
    dims = [int(d) for d in args.dims.split(",")]
    stream = token_batches(tok, args.seq_len, args.eval_batch, args.seed, 0)
    held = [next(stream) for _ in range(args.eval_batches)]
    report = {"args": vars(args), "layers": {}}
    state = {"mode": None}

    for layer in [int(x) for x in args.layers.split(",")]:
        mlp = model.model.layers[layer].mlp
        Wd = mlp.down_proj.weight.float()  # width x neurons
        dnorm = Wd.norm(dim=0)
        acc = {"sum": None, "n": 0}

        def hidden(x):
            return F.silu(mlp.gate_proj(x).float()) * mlp.up_proj(x).float()

        def hook(_m, inputs, output):
            x = inputs[0]
            mode = state["mode"]
            if mode == "bank":
                h = hidden(x).reshape(-1, Wd.shape[1])
                acc["sum"] = h.sum(0) if acc["sum"] is None else acc["sum"] + h.sum(0)
                acc["n"] += h.shape[0]
                return None
            if mode == "leaf":
                state["h"] = hidden(x).reshape(-1, Wd.shape[1])
                leaf = output.detach().requires_grad_(True)
                state["leaf"] = leaf
                return leaf
            if isinstance(mode, torch.Tensor):
                return mode.reshape(output.shape).to(output.dtype)
            return None

        handle = mlp.register_forward_hook(hook)
        state["mode"] = "bank"
        bank = token_batches(tok, args.seq_len, args.batch, args.seed + 1, 0)
        with torch.inference_mode():
            while acc["n"] < args.bank_tokens:
                model(input_ids=next(bank).to(dev))
        hbar = acc["sum"] / acc["n"]
        ybar = hbar @ Wd.T
        totals = {m: {d: 0.0 for d in dims} for m in ("magnitude", "attribution")}
        mean_total, count = 0.0, 0
        for ids in held:
            ids = ids.to(dev)
            state["mode"] = "leaf"
            with torch.enable_grad():
                logits = model(input_ids=ids).logits.float()
                logp = torch.log_softmax(logits, -1)
                leaf = state["leaf"]
                h = state["h"] - hbar
                attr = torch.zeros_like(h)
                for s in range(args.samples):
                    y = torch.distributions.Categorical(logits=logits.detach()).sample()
                    (g,) = torch.autograd.grad(logp.gather(-1, y[..., None]).sum(), leaf,
                                               retain_graph=s < args.samples - 1)
                    attr += (h * (g.float().reshape(-1, Wd.shape[0]) @ Wd)).pow(2)
                attr /= args.samples
            state["mode"] = None
            clean = logp.detach()
            del logits, logp, leaf
            state.pop("leaf", None)
            state.pop("h", None)

            def kl(yhat):
                state["mode"] = yhat
                with torch.inference_mode():
                    lp = torch.log_softmax(model(input_ids=ids).logits.float(), -1)
                state["mode"] = None
                return (clean.exp() * (clean - lp)).sum().item()

            mean_total += kl(ybar.expand(h.shape[0], -1).clone())
            count += h.shape[0]
            for d in dims:
                for mode, score in (("magnitude", h.abs() * dnorm), ("attribution", attr)):
                    keep = torch.zeros_like(h).scatter_(1, score.topk(d, dim=1).indices, 1.0)
                    totals[mode][d] += kl(ybar + (h * keep) @ Wd.T)
            del attr, clean, h
        handle.remove()
        entry = {"mean_output": mean_total / count, "neurons": Wd.shape[1],
                 **{m: {d: v / count for d, v in totals[m].items()} for m in totals}}
        report["layers"][layer] = entry
        for d in dims:
            print(f"[neuron] layer {layer} d={d}/{Wd.shape[1]}: magnitude {entry['magnitude'][d]:.4f}  "
                  f"attribution {entry['attribution'][d]:.4f}  (mean output {entry['mean_output']:.4f})", flush=True)
        torch.cuda.empty_cache()
        with open(args.out, "w") as handle_out:
            json.dump(report, handle_out, indent=1)
    print(f"[neuron] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
