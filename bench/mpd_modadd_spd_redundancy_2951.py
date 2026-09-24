"""#2951: what a stochastic parameter decomposition's causal importances do with a redundant mechanism.

Analysis under SPEC 8's exception: a compact re-implementation of the SPD/VPD training recipe (Bushnaq,
Braun & Sharkey 2025; Bushnaq et al. 2026) on ONE matrix, the unembedding of the trained modular-addition
transformer, to test a prediction of #2951 comment 5805476411 (item 5). It is a baseline for comparison, not
an MPD input.

The grokked network's output is a covering code: any of several 3-4 frequency subsets of its five key
unembedding planes preserves the output, and the minimal sufficient sets are predicted by the chord matrix.
A decomposition whose causal importance is a per-component number must then pick ONE covering set per input.
Prediction: the importances keep the cheapest cover and call the other key planes unimportant, so which key
frequencies look "causally important" is a choice among interchangeable covers, not a property of the
mechanism.

Recipe (one matrix, the paper's losses):
* ``W_U ~ sum_c U_c V_c^T + Delta`` with ``C`` rank-one subcomponents;
* causal importance ``g_c = hard_sigmoid(MLP_c(h . V_c))`` per subcomponent (SPD's gate on the inner
  activation), with lower-leaky straight-through gradients;
* stochastic masks ``m_c = g_c + (1 - g_c) r_c``, ``r ~ U[0,1]`` (the Delta component masked with g = 0);
* loss = faithfulness MSE + stochastic-reconstruction KL + importance minimality ``sum g^p`` with ``p``
  annealed 2 -> 0.5.
After training, every subcomponent is attributed to the unembedding plane its ``U_c`` (a p-vector over output
tokens) lies in, and the per-input important set is compared with the exact minimal sufficient sets.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import torch

from mpd_modadd_2951 import all_pairs, build_model
from mpd_modadd_adversary_2951 import final_residual, kl_rows, unembed_planes


class LowerLeakyHardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        inside = ((x > 0) & (x < 1)).to(grad.dtype)
        below = (x <= 0).to(grad.dtype) * (grad < 0).to(grad.dtype) * 0.01
        return grad * (inside + below)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--components", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--importance-coeff", type=float, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dev = torch.device(args.device)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    model = model.float().eval().to(dev)
    pairs_all = all_pairs(p)
    train_pairs = pairs_all[run["train_idx"]].to(dev)
    test_pairs = pairs_all[run["test_idx"]].to(dev)
    with torch.no_grad():
        h_train = final_residual(model, train_pairs)
        h_test = final_residual(model, test_pairs)
        W = model.W_U.detach().clone()  # p x d
        logp_train = torch.log_softmax(h_train @ W.T, -1)
        logp_test = torch.log_softmax(h_test @ W.T, -1)
        _, planes, cos, sin = unembed_planes(W.double(), p)
        power = planes.pow(2).sum((1, 2))
        order = torch.argsort(power, descending=True)
        keys = order[:5]
        # the orthonormal characters of each key plane, for attributing U_c (a p-vector) to a plane
        char = torch.stack([torch.stack([cos[k], sin[k]], 0) for k in keys.tolist()], 0).float()  # 5 x 2 x p
        char = char / char.norm(dim=-1, keepdim=True)
    report = {"keys": (keys + 1).tolist(), "seeds": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(seed)
        C = args.components
        d = W.shape[1]
        U = torch.nn.Parameter(torch.randn(p, C, device=dev) * 0.1)
        V = torch.nn.Parameter(torch.randn(C, d, device=dev) * 0.1)
        gate_w1 = torch.nn.Parameter(torch.randn(C, 16, device=dev) * 0.5)
        gate_b1 = torch.nn.Parameter(torch.zeros(C, 16, device=dev))
        gate_w2 = torch.nn.Parameter(torch.randn(C, 16, device=dev) * 0.5)
        gate_b2 = torch.nn.Parameter(torch.ones(C, device=dev))
        params = [U, V, gate_w1, gate_b1, gate_w2, gate_b2]
        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)

        def importances(h):
            inner = h @ V.T  # n x C
            hidden = torch.nn.functional.gelu(inner[:, :, None] * gate_w1[None] + gate_b1[None])
            pre = (hidden * gate_w2[None]).sum(-1) + gate_b2[None]
            return LowerLeakyHardSigmoid.apply(pre)

        def masked_logits(h, mask):
            delta = W - U @ V
            comp = torch.einsum("nc,nc,pc->np", h @ V.T, mask, U)
            return comp + h @ delta.T * torch.rand(h.shape[0], 1, device=dev)  # Delta masked with g = 0

        for step in range(args.steps):
            frac = step / max(1, args.steps - 1)
            pnorm = 2.0 - 1.5 * frac
            idx = torch.randint(0, h_train.shape[0], (1024,), device=dev)
            h, lp = h_train[idx], logp_train[idx]
            g = importances(h)
            r = torch.rand_like(g)
            mask = g + (1 - g) * r
            logits = masked_logits(h, mask)
            recon = kl_rows(lp, logits.log_softmax(-1)).mean()
            faith = (W - U @ V).pow(2).mean()
            imp = (g.clamp_min(0) + 1e-8).pow(pnorm).sum(-1).mean()
            loss = 1e3 * faith + recon + args.importance_coeff * imp
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                print(f"[spd seed={seed}] step {step} faith={faith.item():.2e} recon={recon.item():.3e} "
                      f"L0={(g > 0).float().sum(-1).mean().item():.2f} p={pnorm:.2f}", flush=True)
        with torch.no_grad():
            g = importances(h_test)
            active = g > 0.1
            # attribute each subcomponent to the key plane holding most of its output direction U_c
            Un = U / U.norm(dim=0, keepdim=True)
            share = torch.einsum("kjp,pc->kjc", char, Un).pow(2).sum(1)  # 5 x C
            owner = share.argmax(0)
            owned = share.max(0).values > 0.5
            alive = g.mean(0) > 1e-3
            rounded = torch.where(active, torch.ones_like(g), torch.zeros_like(g))
            kl_rounded = kl_rows(logp_test, masked_logits_fixed(h_test, rounded, U, V, W).log_softmax(-1))
            per_input = []
            for n in range(g.shape[0]):
                sets = set()
                for c in torch.nonzero(active[n] & owned).flatten().tolist():
                    sets.add(int(keys[owner[c]]) + 1)
                per_input.append(tuple(sorted(sets)))
            counts = {}
            for s in per_input:
                counts[s] = counts.get(s, 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
            freq_active = {int(keys[k]) + 1: sum(1 for s in per_input if int(keys[k]) + 1 in s) / len(per_input)
                           for k in range(5)}
            entry = {
                "alive_subcomponents": int(alive.sum()),
                "subcomponents_owned_by_key_planes": {int(keys[k]) + 1: int(((owner == k) & owned & alive).sum()) for k in range(5)},
                "mean_L0": float(active.float().sum(-1).mean()),
                "rounded_mask_kl_median": float(kl_rounded.median()),
                "rounded_mask_argmax_kept": float((masked_logits_fixed(h_test, rounded, U, V, W).argmax(-1) == logp_test.argmax(-1)).float().mean()),
                "important_key_sets_top": [[list(k), v] for k, v in top],
                "fraction_of_inputs_where_each_key_frequency_is_important": freq_active,
            }
            report["seeds"][str(seed)] = entry
            print(f"[spd seed={seed}] alive={entry['alive_subcomponents']} owned={entry['subcomponents_owned_by_key_planes']} "
                  f"L0={entry['mean_L0']:.2f} rounded KL med={entry['rounded_mask_kl_median']:.2e} "
                  f"argmax kept={entry['rounded_mask_argmax_kept']:.4f}", flush=True)
            print(f"   important key sets: {entry['important_key_sets_top'][:6]}", flush=True)
            print(f"   P(frequency important): {entry['fraction_of_inputs_where_each_key_frequency_is_important']}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


def masked_logits_fixed(h, mask, U, V, W):
    delta = W - U @ V
    return torch.einsum("nc,nc,pc->np", h @ V.T, mask, U) + h @ delta.T * 0.0


if __name__ == "__main__":
    main()
