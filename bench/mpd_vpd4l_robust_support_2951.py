"""#2951: minimum-code robust supports on VPD's 4-layer target, by the Rust core, per position.

Thin torch executor (SPEC 8). The support search is ``supports::minimum_code_support`` over
``adversary::ZonotopeSeparationOracle`` (Frank-Wolfe ascent over the mask-moment zonotope, refutations shortened
to the controls that carry them); this file only runs the target at a mask.

Pieces: the rank-one SVD factors of each of the 24 decomposed matrices (q, k, v, o, c_fc, down_proj per layer),
each an exact parameter component. A control is one piece at one position, VPD's own mask scope, so the controls
of a sequence of length ``T`` are ``T x pieces`` and a support's size divided by ``T`` is directly comparable with
VPD's L0 per position (180 at KL 0.291). The contract per sequence: keep the support at 1, let every other
control range over ``[0, 1]`` jointly, and the mean KL over positions must stay at most ``eps``.

The SVD basis is the exact null decomposition, the starting point of the fit, not a claimed good one.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from mpd_vpd4l_2951 import Target

UNIT_ROUNDOFF = 2.0 ** -53
MATRICES = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test", required=True, help="pre-tokenized parquet with an input_ids column")
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--eps", type=float, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from gamfit.sae import robust_support

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    model = Target(args.weights, dev)
    model.w = {k: v.double() for k, v in model.w.items()}
    factors = {}
    for l in range(model.n_layer):
        for name in MATRICES:
            u, s, vt = torch.linalg.svd(model.w[f"h.{l}.{name}.weight"], full_matrices=False)
            factors[l, name] = (u, s, vt)
    ranks = [factors[key][1].numel() for key in sorted(factors)]
    offsets = dict(zip(sorted(factors), np.cumsum([0] + ranks[:-1]).tolist()))
    pieces = int(sum(ranks))
    d = model.d
    depth = 8 * d + 4 * d  # attention and MLP reductions per layer bound every logit's rounding chain
    depth *= model.n_layer

    def linear(x, l, name, m):
        u, s, vt = factors[l, name]
        o = offsets[l, name]
        return ((x @ vt.T) * s * m[:, o:o + s.numel()]) @ u.T

    def forward(ids, m):
        """Logits at every position with masks ``m`` of shape (T, pieces)."""
        w = model.w
        h = w["wte.weight"][ids]
        B, T, _ = h.shape
        for l in range(model.n_layer):
            p = f"h.{l}."
            x = model.rms(h, w[p + "rms_1.weight"])
            q, k, v = (linear(x, l, f"attn.{n}_proj", m).view(B, T, model.n_head, model.hd).transpose(1, 2)
                       for n in ("q", "k", "v"))
            o = F.scaled_dot_product_attention(model.rope(q), model.rope(k), v, is_causal=True)
            h = h + linear(o.transpose(1, 2).reshape(B, T, d), l, "attn.o_proj", m)
            x = model.rms(h, w[p + "rms_2.weight"])
            a = linear(x, l, "mlp.c_fc", m)
            h = h + linear(F.gelu(a, approximate="tanh"), l, "mlp.down_proj", m)
        return model.rms(h, w["ln_f.weight"]) @ w["wte.weight"].T

    import pyarrow.parquet as pq
    table = pq.read_table(args.test).slice(0, args.sequences)
    sequences = [torch.tensor(ids[:args.seq_len], dtype=torch.long, device=dev)[None]
                 for ids in table.column("input_ids").to_pylist()]
    T = args.seq_len
    C = T * pieces
    with torch.no_grad():
        ones = torch.ones(T, pieces, device=dev)
        assert torch.allclose(forward(sequences[0], ones), model(sequences[0]), atol=1e-8), "the piece lift is not exact"

    # A piece is u, s, v of one rank-one factor, float64; padded to the widest matrix.
    piece_bits = 64 * max(u.shape[0] + 1 + vt.shape[1] for u, _, vt in factors.values())
    rows = []
    for eps in args.eps:
        for i, ids in enumerate(sequences):
            with torch.no_grad():
                clean_logits = model(ids)
                clean = clean_logits.log_softmax(-1)
            calls = [0]

            def objective(mask):
                calls[0] += 1
                m = torch.from_numpy(np.asarray(mask)).to(dev).view(T, pieces).requires_grad_(True)
                z = forward(ids, m)
                kl = (clean.exp() * (clean - z.log_softmax(-1))).sum(-1).mean()
                (g,) = torch.autograd.grad(kl, m)
                scale = float(z.detach().abs().max() + clean_logits.abs().max())
                gradient = -g.reshape(-1).cpu().numpy()  # q = sum_c (1 - m_c) e_c
                return (float(kl.detach()), depth * UNIT_ROUNDOFF * scale, gradient,
                        depth * UNIT_ROUNDOFF * (float(np.linalg.norm(gradient)) + scale))

            start = time.time()
            found = robust_support(objective, None, np.zeros(C), np.ones(C), piece_bits, eps)
            support = np.asarray(found["support"], dtype=np.int64)
            per_position = np.bincount(support // pieces, minlength=T)
            rows.append({
                "eps": eps, "sequence": i, "support_size": int(support.size),
                "l0_per_position": float(support.size / T), "per_position": per_position.tolist(),
                "status": found["status"], "risk_lower": found["risk_lower"], "risk_upper": found["risk_upper"],
                "separations": found["separations"], "edges": found["edges"], "native_evaluations": calls[0],
                "seconds": time.time() - start,
            })
            r = rows[-1]
            print(f"eps {eps:g} seq {i}: L0/position {r['l0_per_position']:.1f} of {pieces} pieces, {r['status']} "
                  f"risk [{r['risk_lower']:.4g}, {r['risk_upper']:.4g}], {r['separations']} separations, "
                  f"{r['native_evaluations']} evals, {r['seconds']:.0f}s", flush=True)
    summary = {str(eps): float(np.mean([r["l0_per_position"] for r in rows if r["eps"] == eps])) for eps in args.eps}
    print(json.dumps({"l0_per_position": summary, "vpd": {"l0": 180, "kl": 0.291}}), flush=True)
    with open(args.out, "w") as f:
        json.dump({"pieces": pieces, "seq_len": T, "rows": rows, "l0_per_position": summary}, f, indent=1)


if __name__ == "__main__":
    main()
