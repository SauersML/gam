"""#2951 probe: is a token's MLP read sparse once the selection may see that token's OWN output sensitivity?

Analysis under SPEC 8's exception (torch execution of a measurement; companion of
``mpd_llm_chart_restriction_2951.py``, which showed that no input-chosen rank-128 restriction carries the site).

In a fixed orthonormal basis ``B`` of the site (the bank's principal axes), token ``t``'s input is
``x_t = mu + sum_i c_ti b_i``. Keep ``d`` coefficients per token and replace the rest by the mean:
``x_hat = mu + sum_{i in S_t} c_ti b_i``. To second order the KL a dropped coefficient costs is
``(1/2) c_ti^2 b_i^T F_t b_i`` plus cross terms, where ``F_t`` is the output Fisher of the WHOLE downstream
prediction sequence pulled back to position ``t`` of the site. ``b_i^T F_t b_i`` is estimated without bias from
``k`` sampled-label gradients, ``mean_s (g_ts . b_i)^2``, with ``g_ts = d/dx_t log p(y_s)`` and every label of
the sequence drawn from the model itself. Three selections at equal ``d``:

* ``global``: the fixed top-``d`` axes (reproduces the global arm of the chart probe);
* ``input``: the ``d`` largest ``|c_ti|`` (token-adaptive, sees only the input);
* ``fisher``: the ``d`` largest ``c_ti^2 b_i^T F_t b_i`` (token-adaptive, sees the token's output sensitivity:
  an oracle for a decomposition's selector, and the diagonal of the local response metric).

Held-out tokens only; the basis is fitted on a disjoint bank.
"""
from __future__ import annotations

import argparse
import json

import torch

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
    held = []
    stream = token_batches(tok, args.seq_len, args.eval_batch, args.seed, 0)
    for _ in range(args.eval_batches):
        held.append(next(stream))
    report = {"args": vars(args), "layers": {}}
    state = {"mode": None}

    for layer in [int(x) for x in args.layers.split(",")]:
        mlp = model.model.layers[layer].mlp
        cap = []

        def hook(_m, inputs):
            x = inputs[0]
            if state["mode"] == "grab":
                cap.append(x.detach().float().reshape(-1, x.shape[-1]))
                return None
            if state["mode"] == "leaf":
                leaf = x.detach().requires_grad_(True)
                state["leaf"] = leaf
                return (leaf,)
            if isinstance(state["mode"], torch.Tensor):
                return (state["mode"].reshape(x.shape).to(x.dtype),)
            return None

        handle = mlp.register_forward_pre_hook(hook)
        state["mode"] = "grab"
        bank = token_batches(tok, args.seq_len, args.batch, args.seed + 1, 0)
        with torch.inference_mode():
            while sum(c.shape[0] for c in cap) < args.bank_tokens:
                model(input_ids=next(bank).to(dev))
        X = torch.cat(cap)[: args.bank_tokens]
        cap.clear()
        mu = X.mean(0)
        _, _, B = torch.linalg.svd(X - mu, full_matrices=False)  # rows: principal axes
        del X
        totals = {m: {d: 0.0 for d in dims} for m in ("global", "input", "fisher")}
        mean_total, count = 0.0, 0
        for ids in held:
            ids = ids.to(dev)
            state["mode"] = "leaf"
            with torch.enable_grad():
                logits = model(input_ids=ids).logits.float()
                logp = torch.log_softmax(logits, -1)
                leaf = state["leaf"]
                sens = torch.zeros(leaf.shape[0] * leaf.shape[1], B.shape[0], device=dev)
                for s in range(args.samples):
                    y = torch.distributions.Categorical(logits=logits.detach()).sample()
                    lp = logp.gather(-1, y[..., None]).sum()
                    (g,) = torch.autograd.grad(lp, leaf, retain_graph=s < args.samples - 1)
                    sens += (g.float().reshape(-1, g.shape[-1]) @ B.T).pow(2)
                sens /= args.samples
            state["mode"] = None
            clean = logp.detach()
            x = leaf.detach().float().reshape(-1, leaf.shape[-1])
            del logits, logp, leaf
            state.pop("leaf", None)
            c = (x - mu) @ B.T  # tokens x axes

            def kl(xhat):
                state["mode"] = xhat
                with torch.inference_mode():
                    lp = torch.log_softmax(model(input_ids=ids).logits.float(), -1)
                state["mode"] = None
                return (clean.exp() * (clean - lp)).sum().item()

            mean_total += kl(mu.expand_as(x).clone())
            count += x.shape[0]
            for d in dims:
                keep = torch.zeros_like(c)
                keep[:, :d] = 1
                totals["global"][d] += kl(mu + (c * keep) @ B)
                for mode, score in (("input", c.pow(2)), ("fisher", c.pow(2) * sens)):
                    keep = torch.zeros_like(c).scatter_(1, score.topk(d, dim=1).indices, 1.0)
                    totals[mode][d] += kl(mu + (c * keep) @ B)
            del sens, clean
        handle.remove()
        entry = {"mean_input": mean_total / count,
                 **{m: {d: v / count for d, v in totals[m].items()} for m in totals}}
        report["layers"][layer] = entry
        for d in dims:
            print(f"[oracle] layer {layer} d={d}: global {entry['global'][d]:.4f}  input-top {entry['input'][d]:.4f}  "
                  f"fisher-top {entry['fisher'][d]:.4f}  (mean input {entry['mean_input']:.4f})", flush=True)
        torch.cuda.empty_cache()
        with open(args.out, "w") as handle_out:
            json.dump(report, handle_out, indent=1)
    print(f"[oracle] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
