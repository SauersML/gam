"""#2951: do minimum-code robust supports recover modular addition's key frequencies, unprompted?

Thin torch executor (SPEC 8); the search is ``gamfit.sae.robust_support`` (``supports::minimum_code_support`` over
``adversary::ZonotopeSeparationOracle``, each kept piece charged its body, design section 11.1).

Pieces, exact and derived from the task's group rather than fitted: the embedding and unembedding tables are
functions on the cyclic group Z_p, so each expands exactly in its characters,

    W[c] = mean + sum_k ( cos(w_k c) A_k + sin(w_k c) B_k ),   w_k = 2 pi k / p,   k = 1 .. (p - 1) / 2,

and frequency ``k``'s plane (the rank-two piece ``cos(w_k c) A_k + sin(w_k c) B_k`` over all ``c``) is one piece:
a circle of token directions. Every MLP neuron (its read row, bias and write column) is one piece too. Attention and
the ``=`` token's embedding stay native.

Ground truth is read separately from the unembedding power ``|A_k|^2 + |B_k|^2`` per frequency (the known key
frequencies of a grokked model are few and carry most of it). The search is never told them. Reported per input:
which embedding and unembedding frequencies the certified support keeps, and how many neurons.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from mpd_modadd_2951 import all_pairs, build_model
from mpd_modadd_adversary_2951 import unembed_planes

UNIT_ROUNDOFF = 2.0 ** -53


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--eps", type=float, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import robust_support

    torch.set_default_dtype(torch.float64)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    model = build_model(run["config"]).double()
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    for prm in model.parameters():
        prm.requires_grad_(False)
    p = model.W_U.shape[0]
    n_mlp, d = model.W_in.shape
    m = (p - 1) // 2
    e_mean, e_planes, cos, sin = unembed_planes(model.W_E[:p], p)
    u_mean, u_planes, _, _ = unembed_planes(model.W_U, p)
    power = (u_planes ** 2).sum((1, 2))
    order = torch.argsort(power, descending=True)
    share = torch.cumsum(power[order], 0) / power.sum()
    # Controls: embedding mean, m embedding planes, unembedding mean, m unembedding planes, n_mlp neurons.
    E0, E1 = 0, 1 + m
    U0, U1 = 1 + m, 2 + 2 * m
    N0 = U1
    C = N0 + n_mlp

    def table(mean, planes, mask):
        return (mask[0] * mean)[None] + (cos.T * mask[1:]) @ planes[:, :, 0] + (sin.T * mask[1:]) @ planes[:, :, 1]

    def logits(tokens, mask):
        W_E = torch.cat([table(e_mean, e_planes, mask[E0:E1]), model.W_E[p:]], 0)
        W_U = table(u_mean, u_planes, mask[U0:U1])
        x = W_E[tokens] + model.W_pos
        q = torch.einsum("hkd,npd->nhpk", model.W_Q, x)
        k = torch.einsum("hkd,npd->nhpk", model.W_K, x)
        v = torch.einsum("hkd,npd->nhpk", model.W_V, x)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(model.d_head)
        pattern = torch.softmax(scores.masked_fill(~model.causal, float("-inf")), dim=-1)
        x = x + (pattern @ v).transpose(1, 2).reshape(x.shape[0], 3, -1) @ model.W_O.T
        x = x + (torch.relu(x @ model.W_in.T + model.b_in) * mask[N0:]) @ model.W_out.T + model.b_out
        return x[:, -1] @ W_U.T

    shuffled = torch.randperm(p * p, generator=torch.Generator().manual_seed(0))
    tokens_all = all_pairs(p)[shuffled[:args.examples]]
    with torch.no_grad():
        assert torch.allclose(logits(tokens_all, torch.ones(C)), model(tokens_all), atol=1e-9), "the piece lift is not exact"
    depth = 4 * d + n_mlp + 2 * p
    # A plane is two d-vectors, a neuron a read row, a bias and a write column: pad to the neuron, float64.
    piece_bits = 64 * (2 * d + 1)

    rows = []
    for eps in args.eps:
        for i in range(len(tokens_all)):
            tokens = tokens_all[i:i + 1]
            with torch.no_grad():
                clean_logits = model(tokens)
                clean = clean_logits.log_softmax(-1)
            calls = [0]

            def objective(mask):
                calls[0] += 1
                mk = torch.from_numpy(np.asarray(mask)).clone().requires_grad_(True)
                z = logits(tokens, mk)
                kl = (clean.exp() * (clean - z.log_softmax(-1))).sum()
                (g,) = torch.autograd.grad(kl, mk)
                scale = float(z.detach().abs().max() + clean_logits.abs().max())
                gradient = -g.numpy()
                return (float(kl.detach()), depth * UNIT_ROUNDOFF * scale, gradient,
                        depth * UNIT_ROUNDOFF * (float(np.linalg.norm(gradient)) + scale))

            start = time.time()
            found = robust_support(objective, None, np.zeros(C), np.ones(C), piece_bits, eps)
            kept = [int(c) for c in found["support"]]
            row = {
                "eps": eps, "example": i, "tokens": tokens[0].tolist(), "support_size": len(kept),
                "embedding_mean": E0 in kept, "unembedding_mean": U0 in kept,
                "embedding_freqs": [c - E0 for c in kept if E0 < c < E1],
                "unembedding_freqs": [c - U0 for c in kept if U0 < c < U1],
                "neurons": sum(c >= N0 for c in kept),
                "status": found["status"], "risk_lower": found["risk_lower"], "risk_upper": found["risk_upper"],
                "separations": found["separations"], "native_evaluations": calls[0], "seconds": time.time() - start,
            }
            rows.append(row)
            print(f"eps {eps:g} ex {i} {row['tokens'][:2]}: kept {row['support_size']} = E freqs {row['embedding_freqs']}"
                  f" U freqs {row['unembedding_freqs']} neurons {row['neurons']} means E{int(row['embedding_mean'])}"
                  f"U{int(row['unembedding_mean'])} | {row['status']} risk [{row['risk_lower']:.3g}, {row['risk_upper']:.3g}]"
                  f" {row['separations']} checks {row['seconds']:.1f}s", flush=True)
    truth = {"frequencies_by_unembedding_power": [int(k) + 1 for k in order[:10]],
             "cumulative_power_share": [round(float(s), 4) for s in share[:10]]}
    print(json.dumps(truth), flush=True)
    with open(args.out, "w") as f:
        json.dump({"p": p, "frequencies": m, "neurons": n_mlp, "pieces": C, "truth": truth, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
