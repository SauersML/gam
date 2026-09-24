"""#2951 figure: KL to the target against rank units kept per token, VPD's training path and ours.

Reads VPD p-8383f5e5's logged evaluations (rounded-mask KL and L0 every 1k steps, saved from its public W&B history)
and our result JSONs (``mpd_vpd4l_minimal_support_2951.py``, ``mpd_vpd4l_support_fit_2951.py``). Every KL is under
joint masking, VPD's semantics; ours additionally satisfy their declared fidelity at every position (per-position form)
or on the batch mean (mean form). Log-log axes; VPD's steps are coloured by training tokens.
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vpd-history", required=True)
    parser.add_argument("--ours", nargs="*", default=[], help="label=path.json pairs")
    parser.add_argument("--path", nargs="*", default=[],
                        help="label=log pairs: a support_fit log's barrier-view lines (rank units and KL at the supports "
                             "the fit holds), drawn as that fit's sparsity-fidelity path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    hist = json.load(open(args.vpd_history))["data"]["project"]["run"]["sampledHistory"][0]
    steps = np.array([h["_step"] for h in hist])
    kl = np.array([h["eval/ce_kl/kl_rounded_masked"] for h in hist])
    l0 = np.array([h["eval/l0/0.0_total"] for h in hist])
    tokens = steps * 64 * 512
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(l0, kl, c=np.log10(tokens), cmap="viridis", s=8, label="VPD p-8383f5e5 (every 1k steps)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("VPD training tokens (log10)")
    ax.scatter([l0[-1]], [kl[-1]], marker="*", s=220, color="black", zorder=5, label=f"VPD final: {l0[-1]:.0f} at KL {kl[-1]:.3f}")
    markers = iter(["o", "s", "D", "^", "v", "P"])
    for spec in args.ours:
        label, path = spec.split("=", 1)
        r = json.load(open(path))
        ax.scatter([r["rank_units_mean"]], [r["kl_mean"]], marker=next(markers), s=90, edgecolor="black", zorder=6,
                   label=f"{label}: {r['rank_units_mean']:.0f} at KL {r['kl_mean']:.3f}")
        if "heldout" in r:
            h = r["heldout"]
            ax.scatter([h["rank_units_mean"]], [h["kl_mean"]], marker="X", s=110, edgecolor="black", zorder=6,
                       label=f"{label}, held out: {h['rank_units_mean']:.0f} at KL {h['kl_mean']:.3f}")
    view = re.compile(r"barrier view \d+: rank units/position ([0-9.]+) \(fixed supports\), KL mean ([0-9.]+)")
    for spec in args.path:
        label, path = spec.split("=", 1)
        pts = [(float(a), float(b)) for a, b in view.findall(open(path).read()) if float(a) > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", lw=1, alpha=0.7, label=f"{label} (fit path, {len(pts)} views)")
            ax.scatter([xs[-1]], [ys[-1]], s=40, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank units kept per token (VPD: L0 of rank-1 components)")
    ax.set_ylabel("KL to the target, joint masking (nats/token)")
    ax.set_title("VPD 4L target: sparsity vs fidelity")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
