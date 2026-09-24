"""#2951 figure: KL to VPD's 4-layer target against training tokens, VPD's public run against the Parseval-frame
decomposition. Analysis only (SPEC 8 exception).

VPD (goodfire/param-decomp p-8383f5e5, public W&B history): ``eval/ce_kl/kl_rounded_masked`` at every logged step,
batch 64 x 512 tokens per step, coloured by its total eval L0 (active rank-1 subcomponents per token), which it
anneals from ~12,700 to 180. Frame runs: the TEST lines of each log (``step N: TEST {'kl': ...}``), batch rows x 512
tokens per step, at a FIXED budget of rank-2 pieces per token from step 0 (208 matrix slices at the VPD-matched
budget, 192 uniform).
"""
from __future__ import annotations

import ast
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    history, out = sys.argv[1], sys.argv[-1]
    runs = sys.argv[2:-1]  # label=path=tokens_per_step
    d = json.load(open(history))["data"]["project"]["run"]["sampledHistory"][0]
    d = [r for r in d if "eval/ce_kl/kl_rounded_masked" in r]
    steps = np.array([r["_step"] for r in d], dtype=float)
    kl = np.array([r["eval/ce_kl/kl_rounded_masked"] for r in d])
    l0 = np.array([r["eval/l0/0.0_total"] for r in d])
    fig, (ax, fx) = plt.subplots(1, 2, figsize=(16, 5.6))
    sc = ax.scatter(steps * 64 * 512, kl, c=np.log10(l0), cmap="viridis", s=10, label="VPD (L0 anneals 12,700 → 180)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("VPD total L0 (log10 active rank-1 pieces per token)")
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, spec in enumerate(runs):
        label, path, per_step = spec.split("=")
        xs, ys = [], []
        for line in open(path):
            m = re.search(r"step (\d+): TEST (\{.*\})", line)
            if m:
                xs.append(int(m.group(1)) * int(per_step))
                ys.append(ast.literal_eval(m.group(2))["kl"])
        if xs:
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)], label=label)
    ax.axhline(0.2913, color="k", ls=":", lw=0.8)
    ax.text(ax.get_xlim()[0] if False else 1e6, 0.2913, " VPD final: KL 0.291 at L0 180 (26B tokens)", va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training tokens seen")
    ax.set_ylabel("KL to the target, every weight restricted (rounded masks)")
    ax.legend(fontsize=8, frameon=False)
    # frontier: KL against active pieces per token
    sc2 = fx.scatter(l0, kl, c=np.log10(steps * 64 * 512), cmap="magma", s=10)
    cb2 = fig.colorbar(sc2, ax=fx)
    cb2.set_label("VPD: log10 training tokens seen")
    for i, spec in enumerate(runs):
        label, path, per_step = spec.split("=")
        pieces, final = None, None
        for line in open(path):
            m = re.search(r"\((\d+) matrix slices", line)
            if m:
                pieces = int(m.group(1))
            m = re.search(r"step (\d+): TEST (\{.*\})", line)
            if m:
                final = (int(m.group(1)) * int(per_step), ast.literal_eval(m.group(2))["kl"])
        if pieces and final:
            fx.scatter([pieces], [final[1]], marker="*", s=220, color=colors[i % len(colors)],
                       label=f"{label}: {final[0] / 1e6:.1f}M tokens", zorder=5)
    fx.set_xscale("log")
    fx.set_yscale("log")
    fx.set_xlabel("active pieces per token (VPD: rank-1 L0; frame: rank-1 slices = rank-2 pieces x 2)")
    fx.set_ylabel("KL to the target (rounded masks)")
    fx.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
