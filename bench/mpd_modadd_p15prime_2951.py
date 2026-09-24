"""#2951 P15' check on the trained modular-addition transformer: the margin-aware KL bound against the
exact divergence, at every vertex of the non-key unembedding-plane deletions.

Analysis under SPEC 8's exception. For logits ``z``, a gap ``delta`` and ``p = softmax z``,

    KL(softmax z || softmax(z + delta)) = log E_p e^{delta - E_p delta}
      <= osc^2 / 8                                   (P15, Hoeffding's lemma)
      <= (1 - p_max) (e^osc - 1 - osc)               (P15', Bennett's inequality)

since ``Var_p(delta) <= E_p (delta - delta_top)^2 <= (1 - p_max) osc^2`` and ``delta - E_p delta <= osc``.
Both bounds are evaluated at each vertex's own gap and compared with the exact divergence there.
"""
from __future__ import annotations

import argparse
import json

import torch

from mpd_modadd_2951 import all_pairs, build_model
from mpd_modadd_adversary_2951 import final_residual, kl_rows, unembed_planes
from mpd_modadd_structure_2951 import subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--keys", type=int, required=True)
    parser.add_argument("--free", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    config = run["config"]
    p = config["p"]
    model = build_model(config)
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    model = model.double().eval()
    pairs = all_pairs(p)[run["test_idx"]]
    with torch.inference_mode():
        h = final_residual(model, pairs)
        z0 = h @ model.W_U.T
        log_p0 = torch.log_softmax(z0, -1)
        rest_mass = 1.0 - log_p0.max(-1).values.exp()
        _, planes, cos, sin = unembed_planes(model.W_U.detach(), p)
        order = torch.argsort(planes.pow(2).sum((1, 2)), descending=True)
        free = order[args.keys: args.keys + args.free]
        coeff = torch.einsum("kdj,nd->nkj", planes[free], h)
        W = coeff[..., 0, None] * cos[free][None] + coeff[..., 1, None] * sin[free][None]
        verts = subsets(args.free, "cpu")
        rows = {"exact": [], "hoeffding": [], "bennett": []}
        for s in range(0, W.shape[0], 256):
            delta = -torch.einsum("vf,nfp->nvp", verts, W[s:s + 256])
            kl = kl_rows(log_p0[s:s + 256, None, :], z0[s:s + 256, None, :] + delta)
            osc = delta.max(-1).values - delta.min(-1).values
            hoeff = osc.pow(2) / 8
            benn = rest_mass[s:s + 256, None] * (torch.expm1(osc) - osc)
            assert bool((kl <= hoeff * (1 + 1e-9) + 1e-300).all()), "P15 violated"
            assert bool((kl <= benn * (1 + 1e-9) + 1e-300).all()), "P15' violated"
            rows["exact"].append(kl.max(1).values)
            rows["hoeffding"].append(hoeff.max(1).values)
            rows["bennett"].append(torch.minimum(hoeff, benn).max(1).values)
        ex, ho, be = (torch.cat(rows[k]) for k in ("exact", "hoeffding", "bennett"))
    q = lambda x: {f"q{int(100 * a)}": x.quantile(a).item() for a in (0.1, 0.5, 0.9, 0.99)} | {"max": x.max().item()}
    report = {"keys": args.keys, "free": args.free, "pairs": int(pairs.shape[0]),
              "exact_sup": q(ex), "p15_bound": q(ho), "p15prime_bound": q(be),
              "p15_over_exact": q(ho / ex), "p15prime_over_exact": q(be / ex),
              "tightening_p15_over_p15prime": q(ho / be)}
    for key, value in report.items():
        print(f"[p15'] {key}: {value}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
