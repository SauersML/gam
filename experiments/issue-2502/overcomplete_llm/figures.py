"""Figures for issue #2502 from the Rust fit's own dumps.

Plotting only: every number drawn here was produced by
`crates/gam-sae/examples/issue_2502_overcomplete_llm.rs` or by the two thin
PyTorch wrappers in this directory.

Produces, into --out-dir:
  manifold_charts.png       the b=2 blocks' held-out code clouds (the charts)
  dictionary_tiling.png     which block owns which region of the activation cloud
  overcompleteness.png      usage census + the value of the extra atoms
  benchmark.png             held-out FVU at matched rate, four arms
  splice.png                causal damage under the model's own loss
  steering.png              dose-response for the steered atoms
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True, help="the overcomplete arm's out dir")
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--arms-json", default="", help="json: {arm: numbers.json path}")
    ap.add_argument("--interp-json", default="")
    ap.add_argument("--splice-json", default="")
    ap.add_argument("--steer-json", default="")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--atoms", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def load_dump(fit_dir, k, b):
    blocks = np.fromfile(os.path.join(fit_dir, "eval_blocks.u32"), dtype=np.uint32)
    codes = np.fromfile(os.path.join(fit_dir, "eval_codes.f32"), dtype=np.float32)
    n = blocks.size // k
    return blocks.reshape(n, k), codes.reshape(n, k, b), n


def fig_charts(fit_dir, k, b, interp, out):
    blocks, codes, n = load_dump(fit_dir, k, b)
    counts = np.bincount(blocks.reshape(-1), minlength=blocks.max() + 1)
    order = np.argsort(-counts)[:12]
    labels = {}
    if interp:
        for entry in interp["blocks"]:
            top = entry["top_tokens"][0]
            labels[entry["block"]] = f"{top['token']!r} x{top['lift_over_corpus']:.0f}"
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, gid in zip(axes.ravel(), order):
        gid = int(gid)
        r, j = np.nonzero(blocks == gid)
        z = codes[r, j, :]
        if z.shape[0] > 20000:
            z = z[np.linspace(0, z.shape[0] - 1, 20000).astype(int)]
        ax.hexbin(z[:, 0], z[:, 1], gridsize=60, bins="log", cmap="magma", mincnt=1)
        ax.set_title(f"block {gid} · {counts[gid]} firings\n{labels.get(gid, '')}", fontsize=9)
        ax.set_xlabel("$z_0$", fontsize=8)
        ax.set_ylabel("$z_1$", fontsize=8)
        ax.axhline(0, color="w", lw=0.4, alpha=0.4)
        ax.axvline(0, color="w", lw=0.4, alpha=0.4)
    fig.suptitle(
        "Held-out code clouds of the 12 most-used blocks — each panel is one "
        "2-D chart of the Qwen3.5-4B layer-16 residual manifold",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_tiling(fit_dir, acts_dir, k, b, out):
    blocks, _codes, n = load_dump(fit_dir, k, b)
    acts = np.load(os.path.join(acts_dir, "eval.npy"), mmap_mode="r")
    take = min(n, 20000)
    idx = np.linspace(0, n - 1, take).astype(int)
    x = np.asarray(acts[idx]).astype(np.float64)
    x -= x.mean(0, keepdims=True)
    # Two leading directions of the held-out cloud, for a picture only. Via the
    # p x p second moment rather than an n x p SVD: same subspace, far cheaper.
    cov = x.T @ x
    w, v = np.linalg.eigh(cov)
    proj = x @ v[:, ::-1][:, :2]
    top_block = blocks[idx, 0]
    counts = np.bincount(blocks.reshape(-1), minlength=blocks.max() + 1)
    show = np.argsort(-counts)[:8]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    axes[0].hexbin(proj[:, 0], proj[:, 1], gridsize=80, bins="log", cmap="Greys", mincnt=1)
    axes[0].set_title("held-out layer-16 activations, 2 leading directions")
    cmap = plt.get_cmap("tab10")
    axes[1].scatter(proj[:, 0], proj[:, 1], s=1, c="0.85", linewidths=0)
    for i, gid in enumerate(show):
        sel = top_block == gid
        axes[1].scatter(
            proj[sel, 0], proj[sel, 1], s=2, color=cmap(i % 10), label=f"block {gid}", linewidths=0
        )
    axes[1].legend(markerscale=6, fontsize=8, loc="best")
    axes[1].set_title("rows whose FIRST-choice block is one of the 8 most-used")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle("The overcomplete dictionary tiles the activation manifold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_overcompleteness(fit_dir, arms, atoms, k, b, out):
    blocks, _codes, n = load_dump(fit_dir, k, b)
    counts = np.bincount(blocks.reshape(-1), minlength=atoms // b).astype(np.float64)
    numbers = json.load(open(os.path.join(fit_dir, "numbers.json")))
    p = numbers["inputs"]["p"]
    used = numbers["overcompleteness"]["atoms_used_on_heldout"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    srt = np.sort(counts)[::-1]
    axes[0].loglog(np.arange(1, srt.size + 1), np.maximum(srt, 0.5), lw=1.5)
    axes[0].set_xlabel("block rank")
    axes[0].set_ylabel("held-out firings")
    axes[0].set_title("block usage is heavy-tailed, not collapsed")

    axes[1].bar(["ambient p", "atoms used", "atoms K"], [p, used, atoms],
                color=["0.5", "tab:green", "tab:blue"])
    axes[1].axhline(p, color="k", ls="--", lw=1)
    axes[1].set_ylabel("count")
    axes[1].set_title(
        f"atoms actually used on held-out data: {used}\n"
        f"{used / p:.2f}x the ambient dimension {p}"
    )

    if arms:
        names = list(arms)
        fvu = [arms[a]["heldout"]["fvu"] for a in names]
        axes[2].bar(names, fvu, color=["tab:blue", "tab:orange", "tab:green", "0.5"][: len(names)])
        axes[2].set_ylabel("held-out FVU (lower is better)")
        axes[2].set_title("value of the extra atoms, at matched active scalars")
        axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("Overcompleteness, demonstrated", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_benchmark(arms, out):
    names = list(arms)
    fvu = [arms[a]["heldout"]["fvu"] for a in names]
    scalars = [arms[a]["rate"]["active_scalars_per_token"] for a in names]
    sel = [arms[a]["rate"]["selection_bits_per_token"] for a in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(names, fvu, color="tab:blue")
    for i, v in enumerate(fvu):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("held-out FVU")
    axes[0].set_title("matched active scalars per token")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].scatter(scalars, fvu, s=60)
    for nm, sc, fv in zip(names, scalars, fvu):
        axes[1].annotate(nm, (sc, fv), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("active scalars / token")
    axes[1].set_ylabel("held-out FVU")
    axes[1].set_title("rate-distortion placement (selection bits annotated)")
    for nm, sc, fv, sb in zip(names, scalars, fvu, sel):
        axes[1].annotate(f"+{sb:.0f} sel bits", (sc, fv), fontsize=7,
                         xytext=(4, -10), textcoords="offset points", color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_cofiring(fit_dir, k, b, out):
    """Joint code clouds of the most frequently CO-firing block pairs."""
    blocks, codes, n = load_dump(fit_dir, k, b)
    gate = np.linalg.norm(codes, axis=2)
    pair_count = {}
    for i in range(0, n, max(1, n // 40000)):
        row = blocks[i]
        for a in range(k):
            for c in range(a + 1, k):
                key = (int(min(row[a], row[c])), int(max(row[a], row[c])))
                pair_count[key] = pair_count.get(key, 0) + 1
    top = sorted(pair_count, key=pair_count.get, reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ga, gb) in zip(axes.ravel(), top):
        sel_a = blocks == ga
        sel_b = blocks == gb
        rows = np.nonzero(sel_a.any(1) & sel_b.any(1))[0]
        if rows.size == 0:
            continue
        va = gate[rows][sel_a[rows]]
        vb = gate[rows][sel_b[rows]]
        m = min(va.size, vb.size)
        ax.hexbin(va[:m], vb[:m], gridsize=50, bins="log", cmap="viridis", mincnt=1)
        ax.set_xlabel(f"gate of block {ga}")
        ax.set_ylabel(f"gate of block {gb}")
        ax.set_title(f"{pair_count[(ga, gb)]} co-firings (subsampled)", fontsize=9)
    fig.suptitle(
        "Joint amplitude law of the most co-firing block pairs — the shape a "
        "linear code cannot describe",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_chart_semantics(steer, out):
    sweeps = steer.get("angle_sweep", [])
    if not sweeps:
        return False
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.2 * len(sweeps), 5.6),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes)
    for ax, sw in zip(axes, sweeps):
        thetas = [bb["theta"] for bb in sw["bins"]]
        mags = [bb["top_dlogp"][0] for bb in sw["bins"]]
        ax.plot(thetas + [thetas[0]], mags + [mags[0]], "-o", ms=3)
        for th, mg, bb in zip(thetas, mags, sw["bins"]):
            ax.annotate(bb["top_tokens"][0], (th, mg), fontsize=7)
        ax.set_title(
            f"block {sw['block']}: {sw['distinct_top1_tokens_over_angles']} distinct "
            f"top-1 tokens around the chart",
            fontsize=9,
        )
    fig.suptitle(
        "Walking the chart causally: the token the model boosts most, as a "
        "function of the block's own angular coordinate",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def fig_splice(splice, out):
    arms = splice["arms"]
    names = [a for a in arms]
    d = [arms[a]["delta_ce"] for a in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["0.4" if n == "identity" else "tab:red" for n in names]
    ax.bar(names, d, color=colors)
    for i, v in enumerate(d):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$\Delta$ cross-entropy (nats), lower = less damage")
    ax.set_title(
        f"Splicing each reconstruction into layer {splice['layer']} of Qwen3.5-4B-Base\n"
        f"{splice['scored_tokens']} held-out tokens, clean CE = {splice['clean_ce']:.4f}"
    )
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_steering(steer, out):
    entries = steer["blocks"]
    show = entries[: min(6, len(entries))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, e in zip(axes.ravel(), show):
        doses = [c["dose"] for c in e["dose_curve"]]
        tgt = [c["target_mean_dlogp"] for c in e["dose_curve"]]
        ctl = [c["control_mean_dlogp"] for c in e["dose_curve"]]
        ax.plot(doses, tgt, "o-", label="target tokens")
        ax.plot(doses, ctl, "s--", color="0.5", label="frequency-matched controls")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        names = ", ".join(repr(t["token"]) for t in e["targets"][:4])
        ax.set_title(f"block {e['block']} · {names}", fontsize=9)
        ax.set_xlabel(r"dose $\alpha$ (block's own gate units)")
        ax.set_ylabel(r"mean $\Delta \log p$")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Causal steering with Rust-fitted atoms: targets chosen on half A, "
        "dose response measured on half B",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    interp = json.load(open(args.interp_json)) if args.interp_json else None
    arms = {}
    if args.arms_json:
        for name, path in json.load(open(args.arms_json)).items():
            arms[name] = json.load(open(path))

    fig_charts(args.fit_dir, args.topk, args.block_size, interp,
               os.path.join(args.out_dir, "manifold_charts.png"))
    print("wrote manifold_charts.png", flush=True)
    fig_tiling(args.fit_dir, args.acts_dir, args.topk, args.block_size,
               os.path.join(args.out_dir, "dictionary_tiling.png"))
    print("wrote dictionary_tiling.png", flush=True)
    fig_overcompleteness(args.fit_dir, arms, args.atoms, args.topk, args.block_size,
                         os.path.join(args.out_dir, "overcompleteness.png"))
    print("wrote overcompleteness.png", flush=True)
    if arms:
        fig_benchmark(arms, os.path.join(args.out_dir, "benchmark.png"))
        print("wrote benchmark.png", flush=True)
    fig_cofiring(args.fit_dir, args.topk, args.block_size,
                 os.path.join(args.out_dir, "cofiring_clouds.png"))
    print("wrote cofiring_clouds.png", flush=True)
    if args.splice_json:
        fig_splice(json.load(open(args.splice_json)),
                   os.path.join(args.out_dir, "splice.png"))
        print("wrote splice.png", flush=True)
    if args.steer_json:
        steer = json.load(open(args.steer_json))
        fig_steering(steer, os.path.join(args.out_dir, "steering.png"))
        print("wrote steering.png", flush=True)
        if fig_chart_semantics(steer, os.path.join(args.out_dir, "chart_semantics.png")):
            print("wrote chart_semantics.png", flush=True)


if __name__ == "__main__":
    main()
