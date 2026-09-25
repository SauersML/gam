"""#2951: minimum-code robust supports on the grokked modular-addition model, by the Rust core.

Thin torch executor (SPEC 8). The support search is ``supports::minimum_code_support`` and its separation oracle
is ``adversary::ZonotopeSeparationOracle`` (Frank-Wolfe ascent over the mask-moment zonotope, refutations
shortened to the controls that carry them); this file only runs the teacher at a mask and returns the divergence
and its derivative in the moment.

Pieces, each an exact parameter component of the teacher: every head's query-key and output-value SVD rank-one
pieces, and every MLP neuron (its read row and write column together). They are literal pieces, so the moment of
control ``c`` is its deletion amount and the generators are the identity. Each input is one example read at
``=``; a piece's control applies to every use of that piece in the forward pass.

The contract per input: keep the support at 1, let every other piece range over its declared interval (here the
full ablation ``[0, 1]``), and the KL to the clean output must stay at most ``eps``. The report per input is the
support, whether the oracle certified it or only failed to refute it (``unresolved``: the ascent found no
violation and no smoothness constant was stated), and the evidence bounds.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from mpd_modadd_2951 import all_pairs, build_model

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
    H, dh = run["config"]["n_heads"], run["config"]["d_head"]
    n_mlp, d = model.W_in.shape
    p = model.W_U.shape[0]
    qk, ov = [], []
    for h in range(H):
        u, s, vt = torch.linalg.svd(model.W_Q[h].T @ model.W_K[h], full_matrices=False)
        qk.append((u[:, :dh], s[:dh], vt[:dh]))
        u, s, vt = torch.linalg.svd(model.W_O[:, h * dh:(h + 1) * dh] @ model.W_V[h], full_matrices=False)
        ov.append((u[:, :dh], s[:dh], vt[:dh]))
    C = 2 * H * dh + n_mlp
    # The longest reduction any logit passes through: attention over d, the MLP read over d and write over
    # n_mlp, the unembedding over d, and the log-sum-exp over p.
    depth = 4 * d + n_mlp + p

    def logits(tokens, m):
        x = torch.stack([model.W_E[tokens[:, u]] for u in range(3)], 1) + model.W_pos
        mq, mo = m[:H * dh].view(H, dh), m[H * dh:2 * H * dh].view(H, dh)
        mn = m[2 * H * dh:]
        dest = x[:, -1]
        out = torch.zeros_like(dest)
        for h in range(H):
            u, s, vt = qk[h]
            scores = torch.einsum("nr,npr->np", (dest @ u) * s * mq[h], x @ vt.T) / math.sqrt(dh)
            pattern = torch.softmax(scores, -1)
            u, s, vt = ov[h]
            out = out + ((torch.einsum("np,npd->nd", pattern, x) @ vt.T) * s * mo[h]) @ u.T
        dest = dest + out
        a = torch.relu(dest @ model.W_in.T + model.b_in) * mn
        dest = dest + a @ model.W_out.T + model.b_out
        return dest @ model.W_U.T

    order = torch.randperm(p * p, generator=torch.Generator().manual_seed(0))
    tokens_all = all_pairs(p)[order[:args.examples]]
    with torch.no_grad():
        ones = torch.ones(C)
        assert torch.allclose(logits(tokens_all, ones), model(tokens_all), atol=1e-9), "the piece lift is not exact"

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
                m = torch.from_numpy(np.asarray(mask)).clone().requires_grad_(True)
                z = logits(tokens, m)
                kl = (clean.exp() * (clean - z.log_softmax(-1))).sum()
                (g,) = torch.autograd.grad(kl, m)
                # First-order rounding model: every logit carries at most `depth` rounded operations on
                # terms bounded by the largest logit magnitude, and the KL pairs two such vectors.
                scale = float(z.detach().abs().max() + clean_logits.abs().max())
                value_roundoff = depth * UNIT_ROUNDOFF * scale
                gradient = -g.numpy()  # q = sum_c (1 - m_c) e_c, so dF/dq = -dF/dm
                gradient_roundoff = depth * UNIT_ROUNDOFF * (float(np.linalg.norm(gradient)) + scale)
                return float(kl), value_roundoff, gradient, gradient_roundoff

            start = time.time()
            found = robust_support(objective, None, np.zeros(C), np.ones(C), eps)
            support = [int(c) for c in found["support"]]
            rows.append({
                "eps": eps,
                "example": i,
                "tokens": tokens[0].tolist(),
                "support_size": len(support),
                "support_qk": sum(c < H * dh for c in support),
                "support_ov": sum(H * dh <= c < 2 * H * dh for c in support),
                "support_neurons": sum(c >= 2 * H * dh for c in support),
                "status": found["status"],
                "risk_lower": found["risk_lower"],
                "risk_upper": found["risk_upper"],
                "separations": found["separations"],
                "edges": found["edges"],
                "native_evaluations": calls[0],
                "seconds": time.time() - start,
            })
            r = rows[-1]
            print(f"eps {eps:g} ex {i}: support {r['support_size']} (qk {r['support_qk']}, ov {r['support_ov']}, "
                  f"neurons {r['support_neurons']}) {r['status']} risk [{r['risk_lower']:.4g}, {r['risk_upper']:.4g}] "
                  f"{r['separations']} separations, {r['native_evaluations']} evals, {r['seconds']:.1f}s", flush=True)
    summary = {}
    for eps in args.eps:
        sizes = [r["support_size"] for r in rows if r["eps"] == eps]
        summary[str(eps)] = {"mean_support": float(np.mean(sizes)), "max_support": int(np.max(sizes)),
                             "of_pieces": C}
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"pieces": C, "heads": H, "d_head": dh, "neurons": n_mlp, "rows": rows, "summary": summary}, f,
                  indent=1)


if __name__ == "__main__":
    main()
