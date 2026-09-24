"""#2951: planes against lines on a mechanism known to be planar — the grokked modular-addition transformer.

Analysis under SPEC 8's exception (torch execution; fitting is ``mpd_vpd4l_subspace_dict_2951``'s closed form).

Every weight of the one-layer model is restricted per token to a sparse union of ``m``-dimensional subspace atoms of
its output space (``W_E`` excluded; Q/K/V share their input and are stacked; ``W_O``, ``W_in``, ``W_out``, ``W_U``
each read their own stream). Atoms are fitted on the TRAIN pairs' streams; the next-token KL to the unmodified model
is measured on EVERY pair (train and held-out). At equal active directions per token ``L m``, planes (``m = 2``)
against lines (``m = 1``, VPD's rank-1 subcomponents): the grokked mechanism is rotations of planes, which a plane
atom carries whole and a line atom can only cover by several.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from mpd_modadd_2951 import all_pairs, build_model
from mpd_vpd4l_subspace_dict_2951 import fit_dictionary, pursue

SITES = ("attn_in", "o_in", "mlp_in", "down_in", "unembed_in")


def forward(model, tokens, edit=None, grab=None):
    def site(name, z):
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        if grab is not None:
            grab[name].append(flat)
        if edit is not None:
            flat = edit(name, flat)
        return flat.reshape(shape)

    x = model.W_E[tokens] + model.W_pos
    xa = site("attn_in", x)
    q = torch.einsum("hkd,npd->nhpk", model.W_Q, xa)
    k = torch.einsum("hkd,npd->nhpk", model.W_K, xa)
    v = torch.einsum("hkd,npd->nhpk", model.W_V, xa)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(model.d_head)
    pattern = torch.softmax(scores.masked_fill(~model.causal, float("-inf")), dim=-1)
    o = site("o_in", (pattern @ v).transpose(1, 2).reshape(x.shape[0], 3, -1))
    x = x + o @ model.W_O.T
    hid = site("down_in", torch.relu(site("mlp_in", x) @ model.W_in.T + model.b_in))
    x = x + hid @ model.W_out.T + model.b_out
    return site("unembed_in", x[:, -1]) @ model.W_U.T


def weights(model):
    return {"attn_in": torch.cat([model.W_Q.reshape(-1, model.W_Q.shape[-1]), model.W_K.reshape(-1, model.W_K.shape[-1]),
                                  model.W_V.reshape(-1, model.W_V.shape[-1])]),
            "o_in": model.W_O, "mlp_in": model.W_in, "down_in": model.W_out, "unembed_in": model.W_U}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--budgets", required=True, help="comma list of active directions per site L*m")
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    model = build_model(run["config"])
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    model = model.double().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    p = run["config"]["p"]
    tokens = all_pairs(p)
    train = tokens[run["train_idx"]]
    with torch.inference_mode():
        clean = torch.log_softmax(forward(model, tokens), -1)
        grab = {s: [] for s in SITES}
        forward(model, train, grab=grab)
    Z = {s: torch.cat(grab[s]) for s in SITES}
    Ws = weights(model)
    mus = {s: Z[s].mean(0) for s in SITES}
    pinvs = {s: torch.linalg.pinv(Ws[s]) for s in SITES}
    gen = torch.Generator().manual_seed(args.seed)
    report = {"args": vars(args), "arms": {}}

    def kl(edit):
        with torch.inference_mode():
            lp = torch.log_softmax(forward(model, tokens, edit=edit), -1)
        per = (clean.exp() * (clean - lp)).sum(-1)
        acc = (lp.argmax(-1) == (tokens[:, 0] + tokens[:, 1]) % p).double().mean().item()
        return {"kl_mean": per.mean().item(), "kl_max": per.max().item(), "accuracy": acc}

    report["mean_everywhere"] = kl(lambda s, z: mus[s].expand_as(z).clone())
    print(f"[atoms-modadd] every site at its mean: {report['mean_everywhere']}", flush=True)
    for budget in [int(b) for b in args.budgets.split(",")]:
        for m in (1, 2):
            if budget % m:
                continue
            L = budget // m
            dicts = {}
            for s in SITES:
                Y = (Z[s] - mus[s]) @ Ws[s].T
                dicts[s], _ = fit_dictionary(Y.float(), args.atoms, m, L, args.sweeps, gen)
                dicts[s] = dicts[s].double()

            def edit(s, z):
                y = (z - mus[s]) @ Ws[s].T
                _, yhat = pursue(y, dicts[s], L)
                return mus[s] + yhat @ pinvs[s].T
            res = kl(edit)
            report["arms"][f"budget{budget}_m{m}"] = res
            print(f"[atoms-modadd] {budget} directions per token per site: m={m} (L={L}): {res}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"[atoms-modadd] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
