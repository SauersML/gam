"""#2951: per-example minimal supports of the grokked modular-addition model, found against the model (no gate).

Thin torch executor for ``gamfit.sae.minimal_support`` (SPEC 8: the search and its rules are the Rust owner's).
The model is one attention layer and one ReLU MLP read at the ``=`` position, so an example is one position.
Pieces (they sum exactly to the parameters): each MLP neuron (its W_in row, bias and W_out column; ReLU singles the
neuron basis out), and per head the singular directions of the exact bilinear maps ``W_Q,h^T W_K,h`` (query-key) and
``W_O,h W_V,h`` (output-value), masked at the destination position. Reported against the known mechanism: the
fraction of each kept neuron's input weight on the key Fourier planes, and how much the supports vary across
examples (a union every example uses shows up as a support shared by all examples).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from mpd_modadd_2951 import all_pairs, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True, help="declared per-example KL fidelity (nats)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import minimal_support
    torch.set_default_dtype(torch.float64)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    model = build_model(run["config"]).double()
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    p, H, dh = run["config"]["p"], run["config"]["n_heads"], run["config"]["d_head"]
    tokens = all_pairs(p)[torch.randperm(p * p, generator=torch.Generator().manual_seed(0))[:args.examples]]
    qk, ov = [], []
    for h in range(H):
        u, s, vt = torch.linalg.svd(model.W_Q[h].T @ model.W_K[h], full_matrices=False)  # d x d, rank dh
        qk.append((u[:, :dh], s[:dh], vt[:dh]))
        u, s, vt = torch.linalg.svd(model.W_O[:, h * dh:(h + 1) * dh] @ model.W_V[h], full_matrices=False)
        ov.append((u[:, :dh], s[:dh], vt[:dh]))
    n_mlp = model.W_in.shape[0]
    C = 2 * H * dh + n_mlp

    def forward(m):
        x = torch.stack([model.W_E[tokens[:, u]] for u in range(3)], 1) + model.W_pos  # N x 3 x d
        mq, mo, mn = m[:, :H * dh].view(-1, H, dh), m[:, H * dh:2 * H * dh].view(-1, H, dh), m[:, 2 * H * dh:]
        dest = x[:, -1]
        out = torch.zeros_like(dest)
        for h in range(H):
            u, s, vt = qk[h]
            qa = (dest @ u) * s * mq[:, h]                       # N x dh: masked query-key terms at '='
            scores = torch.einsum("nr,npr->np", qa, x @ vt.T) / math.sqrt(dh)
            pattern = torch.softmax(scores, -1)                  # '=' attends to all three positions
            u, s, vt = ov[h]
            out = out + ((torch.einsum("np,npd->nd", pattern, x) @ vt.T) * s * mo[:, h]) @ u.T
        dest = dest + out
        a = torch.relu(dest @ model.W_in.T + model.b_in) * mn
        dest = dest + a @ model.W_out.T + model.b_out
        return dest @ model.W_U.T

    with torch.no_grad():
        clean = model(tokens).log_softmax(-1)
        full = forward(torch.ones(len(tokens), C)).log_softmax(-1)
    print(f"[mams] p={p} examples={len(tokens)} pieces={C} (QK {H * dh}, OV {H * dh}, neurons {n_mlp}); "
          f"full-support max KL {(clean.exp() * (clean - full)).sum(-1).max():.2e}", flush=True)

    def evaluate(keep):
        m = torch.from_numpy(keep).double().requires_grad_(True)
        logits = forward(m)
        lp = logits.log_softmax(-1)
        kl = (clean.exp() * (clean - lp)).sum(-1)
        (g,) = torch.autograd.grad(kl.sum(), m, retain_graph=True)
        y = torch.distributions.Categorical(logits=logits.detach()).sample()
        (hh,) = torch.autograd.grad(lp.gather(-1, y[:, None]).sum(), m)
        return kl.detach().numpy(), (-g + 0.5 * hh.pow(2)).numpy(), (-g - 0.5 * hh.pow(2)).numpy()

    torch.manual_seed(0)
    result = minimal_support(evaluate, len(tokens), C, args.eps, "per_position", 1)
    keep = result["keep"]
    E = model.W_E.detach()[:p]
    D = E - E.mean(0)
    a = torch.arange(p, dtype=torch.float64)
    power = {k: ((torch.stack([torch.cos(2 * math.pi * k * a / p), torch.sin(2 * math.pi * k * a / p)], 1).T @ D)
                 .pow(2).sum().item()) for k in range(1, (p - 1) // 2 + 1)}
    key = sorted(power, key=lambda k: -power[k])[:5]
    kept_neurons = keep[:, 2 * H * dh:]
    usage = kept_neurons.mean(0)
    report = {
        "args": vars(args), "pieces": C, "key_frequencies": key, "rounds": result["rounds"],
        "kl_max": float(result["divergence"].max()), "kl_mean": float(result["divergence"].mean()),
        "kept_mean": {"qk": float(keep[:, :H * dh].sum(1).mean()), "ov": float(keep[:, H * dh:2 * H * dh].sum(1).mean()),
                      "neurons": float(kept_neurons.sum(1).mean())},
        "neurons_used_by_every_example": int((usage == 1).sum()),
        "neurons_used_by_no_example": int((usage == 0).sum()),
        "neurons_used_by_some": int(((usage > 0) & (usage < 1)).sum()),
    }
    print(f"[mams] FINAL KL max {report['kl_max']:.4f} mean {report['kl_mean']:.4f} (eps {args.eps}); kept per example "
          f"{report['kept_mean']}; neurons used by every/some/no example: {report['neurons_used_by_every_example']}/"
          f"{report['neurons_used_by_some']}/{report['neurons_used_by_no_example']}; rounds {len(result['rounds'])}",
          flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
