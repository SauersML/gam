"""#2951 showcase figures from the modadd receipts (analysis only, SPEC 8 exception).

glide.png      one input's output distribution along the angle path and the mask path
charges.png    the charge table (planes x output harmonics) at a, b and ab, and charge formation over training
lattice.png    the covering-code lattice of the five key frequencies: predicted sufficiency vs measured minimal sets
snap.png       output phase vs shift fraction: the grokked transformer glides, Qwen3-4B's weekday snaps
p15.png        P15 and P15' against the exact worst deletion, per test pair
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpd_modadd_2951 import all_pairs, build_model
from mpd_modadd_adversary_2951 import final_residual, kl_rows, unembed_planes
from mpd_modadd_paths_2951 import angle_table, mask_table, planes_of

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load(path):
    with open(path) as handle:
        return json.load(handle)


def glide(model, p, out):
    table = model.W_E.detach().cpu()
    mean, planes = planes_of(table, p)
    a, b = 17, 40
    tokens = torch.tensor([[a, b, p]])
    ts = np.linspace(0, 1, 121)
    rows = {"angle": [], "mask": []}
    with torch.inference_mode():
        for t in ts:
            for name, edited in (("angle", angle_table(table, mean, planes, p, t)), ("mask", mask_table(table, p, t))):
                probs = torch.softmax(model(tokens, embed_at={0: edited}).double(), -1)[0].numpy()
                rows[name].append(probs)
    centre = (a + b) % p
    window = np.arange(centre - 6, centre + 8) % p
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
    for ax, name, title in zip(axes, ("angle", "mask"),
                               (f"angle path: every embedding plane of a={a} turned by w_k t",
                                f"mask path: the straight segment (1-t) e_a + t e_(a+1)")):
        img = np.array(rows[name])[:, window].T
        ax.imshow(img, aspect="auto", origin="lower", cmap="magma", extent=[0, 1, -0.5, len(window) - 0.5], vmin=0, vmax=1)
        ax.set_yticks(range(len(window)), [f"{c}" + ("  <- a+b" if c == centre else "  <- a+b+1" if c == (centre + 1) % p else "") for c in window])
        ax.set_title(title, loc="left")
        ax.set_ylabel("output token c")
    axes[1].set_xlabel("t (fraction of a one-step shift of the operand a)")
    fig.suptitle(f"Moving the input along its circle: the answer glides (angle) or dims and splits (mask). Input {a}+{b} (mod {p})")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


def charges(results, out):
    last = load(os.path.join(results, "charges_addition.json"))
    fig = plt.figure(figsize=(15, 5.2))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    planes = sorted({k.split("/")[0] for k in last["turned"]}, key=lambda s: -1)
    order = [k.split("/")[0] for k in last["turned"] if k.endswith("/a")]
    harmonics = [20, 22, 21, 5, 7, 10, 14, 3, 40, 53]
    cmap = plt.get_cmap("coolwarm")
    for col, site in enumerate(("a", "b", "ab")):
        ax = fig.add_subplot(grid[0, col])
        mat = np.zeros((len(order), len(harmonics)))
        weight = np.zeros_like(mat)
        for i, plane in enumerate(order):
            entry = last["turned"][f"{plane}/{site}"]
            for jx, h in enumerate(harmonics):
                ch = entry.get(str(h), {}).get("charges", [])
                if ch:
                    mat[i, jx] = ch[0][0]
                    weight[i, jx] = ch[0][1]
        shown = np.where(weight > 0.05, mat, np.nan)
        ax.imshow(shown, cmap=cmap, vmin=-2, vmax=2, aspect="auto")
        for i in range(len(order)):
            for jx in range(len(harmonics)):
                if not np.isnan(shown[i, jx]) and mat[i, jx] != 0:
                    ax.text(jx, i, f"{int(mat[i, jx]):+d}\n{weight[i, jx]:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(harmonics)), harmonics)
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel("output harmonic")
        if col == 0:
            ax.set_ylabel("embedding plane turned")
        ax.set_title(f"turned at operand {site}")
    ax = fig.add_subplot(grid[0, 3])
    steps = [2000, 5000, 10000, 20000]
    accs = {2000: 0.064, 5000: 0.106, 10000: 0.9994, 20000: 1.0}
    for key in ("20", "22", "21", "5", "7"):
        ys = []
        for s in steps:
            path = os.path.join(results, "charges_addition.json" if s == 20000 else f"charges_addition_{s}.json")
            rep = load(path)
            entry = rep["turned"].get(f"{key}/ab", {}).get(key, {"charges": []})["charges"]
            ys.append(dict(entry).get(2, 0.0))
        ax.plot(steps, ys, marker="o", label=f"frequency {key}")
    ax2 = ax.twinx()
    ax2.plot(steps, [accs[s] for s in steps], "k--", lw=1.2, label="test accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("test accuracy (dashed)")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("weight of charge +2 under a joint turn")
    ax.set_title("The algorithm's charges form before test accuracy moves")
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Charge spectroscopy: turning each parameter plane reads off the monomials the network computes "
                 "(diagonal +1/+1/+2 = cos(w_k(a+b-c)); blank rows = dead planes; off-diagonal = intermodulation)")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


def lattice(results, out):
    rep = load(os.path.join(results, "structure_addition.json"))["checkpoints"]["20000"]["redundancy_unembed"]
    measured = {tuple(sorted(k)): v for k, v in rep["most_common_minimal_sets"]}
    A = {20: 19.495, 5: 7.885, 22: 7.305, 7: 5.694, 21: 5.408}
    p = 113
    keys = [20, 5, 22, 7, 21]

    def kl(S):
        return math.log1p(sum(math.exp(-sum(A[k] * (1 - math.cos(2 * math.pi * k * d / p)) for k in S)) for d in range(1, p)))

    fig, ax = plt.subplots(figsize=(13, 6.2))
    levels = {r: list(itertools.combinations(keys, r)) for r in range(0, 6)}
    posn = {}
    for r, sets in levels.items():
        for i, S in enumerate(sets):
            posn[S] = ((i - (len(sets) - 1) / 2) * 1.35, r)
    for r in range(0, 5):
        for S in levels[r]:
            for T in levels[r + 1]:
                if set(S) < set(T):
                    ax.plot([posn[S][0], posn[T][0]], [posn[S][1], posn[T][1]], color="0.88", lw=0.6, zorder=1)
    top = max(measured.values())
    for S, (x, y) in posn.items():
        value = kl(S) if S else float("inf")
        suff = value <= 0.01
        minimal = suff and not any(kl(tuple(sorted(set(S) - {k}))) <= 0.01 for k in S) if S else False
        count = measured.get(tuple(sorted(S)), 0)
        face = plt.cm.viridis(count / top) if count else "white"
        edge = "crimson" if minimal else ("black" if suff else "0.7")
        ax.scatter([x], [y], s=900, c=[face], edgecolors=edge, linewidths=3 if minimal else 1, zorder=3)
        ax.text(x, y + 0.02, "{" + ",".join(map(str, S)) + "}", ha="center", va="center", fontsize=6.3,
                color="white" if count > top * 0.5 else "black", zorder=4)
        ax.text(x, y - 0.3, "KL=inf" if not S else f"{value:.1e}", ha="center", fontsize=5.5, color="0.35")
    ax.set_yticks(range(6), [f"{r} kept" for r in range(6)])
    ax.set_xticks([])
    ax.set_title("The grokked network is a covering code. Nodes: kept sets of key frequencies. Red ring: minimal sufficient set "
                 "PREDICTED from five amplitudes and the chord matrix (no fit).\nFill: how often the set is the MEASURED minimal "
                 "sufficient set across 8938 test pairs. The four red rings are exactly the four measured sets.", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, top))
    fig.colorbar(sm, ax=ax, label="test pairs with this measured minimal set", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=170)


def snap(results, out):
    paths = load(os.path.join(results, "paths_addition.json"))["paths"]["pos0/angle"]
    mask = load(os.path.join(results, "paths_addition.json"))["paths"]["pos0/mask"]
    week = load(os.path.join(results, "weekday_fs_Qwen3-4B-Base.json"))["paths"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    t = [r["t"] for r in paths]
    axes[0].plot(t, [-r["phase_shift_median"][0] for r in paths], "C0-", lw=2.2, label="angle path (measured)")
    axes[0].plot(t, [-r["phase_shift_median"][0] for r in mask], "C3-", lw=1.4, label="mask path (measured)")
    axes[0].plot(t, [-r["expected_phase"][0] for r in paths], "k:", label="prediction w_k t (no fit)")
    axes[0].set_title("Grokked transformer: the answer glides (k = 20)")
    d = [r["delta"] for r in week["angle"]]
    axes[1].plot(d, [r["phase_median"] for r in week["angle"]], "C0-o", lw=2.2, label="angle path (measured)")
    axes[1].plot(d, [r["phase_median"] for r in week["straight"]], "C3-o", lw=1.4, label="straight path (measured)")
    axes[1].plot(d, [r["predicted_phase"] for r in week["angle"]], "k:", label="circle computer would follow this")
    axes[1].set_title("Qwen3-4B, weekday token embedding: the answer snaps")
    for ax in axes:
        ax.set_xlabel("fraction of a one-step move along the input's circle")
        ax.set_ylabel("output phase shift (rad)")
        ax.legend(fontsize=7)
    fig.suptitle("Is the input's circle the variable the network computes with? A fit-free test from the parameters")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


def p15(model, run, p, out):
    pairs = all_pairs(p)[run["test_idx"]]
    with torch.inference_mode():
        h = final_residual(model, pairs)
        z0 = h @ model.W_U.T
        log_p0 = torch.log_softmax(z0, -1)
        rest = 1.0 - log_p0.max(-1).values.exp()
        _, planes, cos, sin = unembed_planes(model.W_U.detach(), p)
        order = torch.argsort(planes.pow(2).sum((1, 2)), descending=True)
        free = order[5:17]
        coeff = torch.einsum("kdj,nd->nkj", planes[free], h)
        W = coeff[..., 0, None] * cos[free][None] + coeff[..., 1, None] * sin[free][None]
        verts = torch.tensor(list(itertools.product((0.0, 1.0), repeat=12)), dtype=torch.float64)
        ex, ho, be = [], [], []
        for s in range(0, W.shape[0], 256):
            delta = -torch.einsum("vf,nfp->nvp", verts, W[s:s + 256])
            kl = kl_rows(log_p0[s:s + 256, None, :], z0[s:s + 256, None, :] + delta)
            osc = delta.max(-1).values - delta.min(-1).values
            i = kl.argmax(1)
            r = torch.arange(kl.shape[0])
            ex.append(kl[r, i])
            o = osc[r, i]
            ho.append(o.pow(2) / 8)
            be.append(torch.minimum(o.pow(2) / 8, rest[s:s + 256] * (torch.expm1(o) - o)))
        ex, ho, be = (torch.cat(x).numpy() for x in (ex, ho, be))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.loglog(ex, ho, ".", ms=2, alpha=0.4, color="C3", label="P15 = osc^2/8 (Hoeffding)")
    ax.loglog(ex, be, ".", ms=2, alpha=0.4, color="C0", label="P15' = (1-p_max)(e^osc-1-osc) (Bennett)")
    lo, hi = ex.min(), max(ho.max(), be.max())
    ax.loglog([lo, hi], [lo, hi], "k-", lw=0.8, label="bound = exact")
    ax.set_xlabel("exact worst deletion of 12 non-key W_U planes (nats)")
    ax.set_ylabel("certified upper bound (nats)")
    ax.set_title("A margin-aware KL bound: 10^5x tighter on a confident network\n(every one of 8938 test pairs, each at its exact worst mask)")
    ax.legend(fontsize=7, markerscale=5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    p = run["config"]["p"]
    model = build_model(run["config"])
    model.load_state_dict(run["checkpoints"][max(run["checkpoints"])])
    model = model.double().eval()
    for name, fn in (("glide", lambda o: glide(model, p, o)), ("charges", lambda o: charges(args.results, o)),
                     ("lattice", lambda o: lattice(args.results, o)), ("snap", lambda o: snap(args.results, o)),
                     ("p15", lambda o: p15(model, run, p, o))):
        fn(os.path.join(args.out_dir, f"{name}.png"))
        print(f"[showcase] {name}", flush=True)


if __name__ == "__main__":
    main()
