"""#2951: does an unsupervised plane-atom frame recover the grokked model's rotation planes?

Analysis under SPEC 8's exception. The embedding rows of the grokked modular-addition model live on its key Fourier
planes: frequency k's plane is spanned by the two columns sum_a cos(2 pi k a / p) e_a and sum_a sin(...) of
W_E^T (centred). A complete orthogonal frame of the embedding space with m = 2 atoms (planes), each row coded by its
top-L planes, is fitted by orthogonal dictionary learning (alternating top-L coding and orthogonal Procrustes; the
same monotone alternation as ``parseval_frame::orthogonal_dictionary_fit``), with no knowledge of frequencies. Each
fitted plane is compared with every frequency's plane by principal angles: cos theta_1 * cos theta_2 = 1 means the
atom IS that frequency's plane.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from mpd_modadd_2951 import build_model


def odl(D, X, m, L, iters):
    trace = []
    for _ in range(iters):
        c = (D @ X.T).view(D.shape[0], -1, m)
        energy = c.pow(2).sum(-1)
        keep = torch.zeros_like(energy).scatter_(1, energy.topk(L, dim=1).indices, 1.0)
        S = (c * keep[..., None]).reshape(D.shape[0], -1)
        trace.append(((D - S @ X).pow(2).sum() / D.pow(2).sum()).item())
        U, _, Vh = torch.linalg.svd(S.T @ D)
        X = U @ Vh
    return X, trace


def plane_overlap(A, B):
    """Product of the cosines of the two principal angles between the column spans of A and B (d x 2 each)."""
    qa = torch.linalg.qr(A)[0]
    qb = torch.linalg.qr(B)[0]
    s = torch.linalg.svdvals(qa.T @ qb)
    return (s[0] * s[1]).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--active", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    model = build_model(run["config"])
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    p = run["config"]["p"]
    E = model.W_E.detach().double()[:p]
    D = E - E.mean(0)
    d = D.shape[1]
    a = torch.arange(p, dtype=torch.float64)
    planes, power = {}, {}
    for k in range(1, (p - 1) // 2 + 1):
        cos = torch.cos(2 * math.pi * k * a / p) @ D
        sin = torch.sin(2 * math.pi * k * a / p) @ D
        planes[k] = torch.stack([cos, sin], 1)
        power[k] = (cos.pow(2).sum() + sin.pow(2).sum()).item()
    total = sum(power.values())
    key = sorted(power, key=lambda k: -power[k])[:8]
    print(f"[planes] key frequencies by embedding power: {[(k, round(power[k] / total, 3)) for k in key]}", flush=True)
    report = {"args": vars(args), "key_frequencies": {k: power[k] / total for k in key}, "starts": {}}
    for start in ("identity", "random"):
        X0 = torch.eye(d, dtype=torch.float64) if start == "identity" else torch.linalg.qr(
            torch.randn(d, d, dtype=torch.float64))[0]
        X, trace = odl(D, X0, 2, args.active, args.iters)
        atoms = [X[2 * j:2 * j + 2].T for j in range(d // 2)]
        usage = torch.zeros(d // 2)
        c = (D @ X.T).view(p, -1, 2).pow(2).sum(-1)
        usage += torch.zeros_like(c).scatter_(1, c.topk(args.active, dim=1).indices, 1.0).sum(0)
        matches = {}
        for k in key:
            best = max(range(len(atoms)), key=lambda j: plane_overlap(atoms[j], planes[k]))
            matches[k] = {"atom": best, "overlap": plane_overlap(atoms[best], planes[k]),
                          "atom_used_by_tokens": int(usage[best].item())}
        # the gauge prediction: the most-used atoms span the union of the key planes even if no atom is one
        top = usage.topk(5).indices.tolist()
        used = torch.cat([atoms[j] for j in top], 1)  # d x 10
        union = torch.cat([planes[k] for k in key[:5]], 1)  # d x 10
        qa = torch.linalg.qr(used)[0]
        qb = torch.linalg.qr(union)[0]
        cosines = torch.linalg.svdvals(qa.T @ qb)
        matches["union"] = {"principal_cosines": [round(v, 4) for v in cosines.tolist()]}
        print(f"[planes] start={start}: principal cosines between the 5 most-used atoms' span and the 5 key planes' "
              f"span: {[round(v, 3) for v in cosines.tolist()]}", flush=True)
        report["starts"][start] = {"residual_trace": trace, "matches": matches}
        print(f"[planes] start={start}: residual share {trace[0]:.3f} -> {trace[-1]:.4f}; "
              f"best atom overlap per key frequency: "
              f"{ {k: (v['atom'], round(v['overlap'], 4), v['atom_used_by_tokens']) for k, v in matches.items()} }",
              flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[planes] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
