"""Render the frozen Oct 4 measured results. Python plots; it does not fit models.

python oct4_results.py --out DIRECTORY [--open]
The adjacent JSON contains source hashes, full scope, limitations and captions.
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/oct4_results.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--open", action="store_true", help="Open the PDF in macOS Preview using subprocess")
    args = parser.parse_args()
    d = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 17, "axes.titlesize": 22, "axes.labelsize": 18,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "figure.facecolor": "white", "axes.facecolor": "white",
                        "axes.grid": False, "pdf.fonttype": 42, "savefig.dpi": 170})
    blue, orange, gray = "#1768A6", "#D56822", "#777777"
    pdf_path = args.out / "results.pdf"
    files = []
    with PdfPages(pdf_path) as pdf:
        def save(fig, name):
            fig.savefig(args.out / f"{name}.png", bbox_inches="tight")
            fig.savefig(args.out / f"{name}.pdf", bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            files.append(name)
            plt.close(fig)

        j = d["joint"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
        for family, color, name in [("NativeSvd", blue, "SVD"), ("CopyResidual", orange, "Copy + residual")]:
            for fit, marker, linestyle, suffix in [("weight_Frobenius", "s", "--", "weight fit"), ("training_native_inputs", "o", "-", "input fit")]:
                records = sorted([r for r in j["records"] if r["family"] == family and r["fit"] == fit], key=lambda r: r["savings_percent"])
                for ax, key in zip(axes, ["local", "run"]):
                    measured = [r for r in records if r[key] is not None]
                    ax.plot([r["savings_percent"] for r in measured], [r[key]["upper"] for r in measured],
                            color=color, marker=marker, linestyle=linestyle, markersize=8,
                            linewidth=2, label=f"{name}, {suffix}")
        for ax in axes:
            ax.scatter([0], [0], c="black", s=65, zorder=4)
            ax.annotate("Native", (0, 0), xytext=(9, 10), textcoords="offset points", fontsize=14)
            ax.set_xlabel("Whole-program description saved (%)")
            ax.set_xlim(-0.12, 3.5)
            ax.set_ylim(bottom=-0.055)
        axes[0].set_title("Local fidelity of joint replacements", pad=17)
        axes[0].set_ylabel("Maximum normalized state error")
        axes[0].axhline(0.2, color=gray, linestyle=":", linewidth=1.5)
        axes[0].text(3.4, .22, "0.2 tolerance", ha="right", color=gray, fontsize=13)
        axes[1].set_title("Counterfactual prediction error", pad=17)
        axes[1].set_ylabel("Worst-group KL (nats per token)")
        axes[1].axhline(0.1, color=gray, linestyle=":", linewidth=1.5)
        axes[1].text(3.4, .15, "0.1 tolerance", ha="right", color=gray, fontsize=13)
        axes[1].text(.98, .97, "6 local failures:\nrun error not measured", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=13, color=gray)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=14)
        fig.subplots_adjust(bottom=.25, wspace=.32, top=.89)
        save(fig, "01_joint_quality")

        local = sorted({p["constraint"]["local"] for p in j["points"]})
        run = sorted({p["constraint"]["run"] for p in j["points"]})
        matrix = np.zeros((len(local), len(run)))
        for p in j["points"]:
            matrix[local.index(p["constraint"]["local"]), run.index(p["constraint"]["run"])] = 100 * (j["native_bits"] - p["upper_cost"]) / j["native_bits"]
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1), aspect="auto")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.2f}%" if value else "0%", ha="center", va="center", color="white" if value > .85 else "#172B40", fontsize=21)
        ax.set_xticks(range(len(run)), [f"{v:g}" for v in run])
        ax.set_yticks(range(len(local)), [f"{v:g}" for v in local])
        ax.set_xlabel("Allowed worst-group KL (nats per token)", labelpad=14)
        ax.set_ylabel("Allowed normalized local error", labelpad=14)
        ax.set_title("Description savings across the fidelity frontier", pad=20)
        fig.colorbar(im, ax=ax, pad=.04, label="Whole-program description saved (%)")
        fig.tight_layout()
        save(fig, "02_joint_frontier")

        fig, ax = plt.subplots(figsize=(10, 6.8))
        values = [g["conditional_worst_group_KL_lower_nats_per_token"] for g in d["collision"]["groups"]]
        ax.bar(range(4), values, color=blue, width=.60)
        for i, value in enumerate(values):
            ax.text(i, value+.012, f"{value:.3f}", ha="center", fontsize=22)
        ax.set_xticks(range(4), [f"Layer {i}" for i in range(4)])
        ax.set_ylim(0, .55)
        ax.set_ylabel("Lower bound on worst-group KL\n(nats per token)")
        ax.set_title("Ignoring an omitted control imposes an error floor", pad=20)
        ax.text(.98, .93, "Same prediction for clean\nand full MLP removal", transform=ax.transAxes,
                ha="right", va="top", fontsize=15, color=gray)
        fig.tight_layout()
        save(fig, "03_control_floor")

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        b = d["decode"]
        x = np.arange(2)
        for offset, key, color, label in [(-.18, "before", gray, "Before"), (.18, "after", blue, "After")]:
            values = [b[key][mode]["median_seconds"] for mode in ["ordinary", "cached"]]
            bars = axes[0].bar(x+offset, values, width=.32, color=color, label=label)
            for bar, value, mode in zip(bars, values, ["ordinary", "cached"]):
                axes[0].text(bar.get_x()+bar.get_width()/2, max(b[key][mode]["seconds"])+.09, f"{value:.2f}", ha="center", fontsize=15)
            for at, mode in zip(x+offset, ["ordinary", "cached"]):
                times = b[key][mode]["seconds"]
                axes[0].scatter(at+np.linspace(-.055,.055,len(times)),times,s=15,c="black",zorder=4)
        axes[0].set_xticks(x, ["Standalone decode", "Cached decode"])
        axes[0].set_ylim(0, 5.5)
        axes[0].set_ylabel("Seconds per saved artifact")
        axes[0].set_title("Cached decoding is 1.98× faster", pad=20)
        axes[0].legend(frameon=False, fontsize=14)
        t = j["runtime_seconds"]
        keys = ["cpu_metric_reductions", "gpu_head_upload_norm_gemm_download", "cpu_head_log_normalization", "explained_execution"]
        labels = ["CPU KL / metrics", "GPU readout + transfers", "CPU normalization", "GPU candidate execution"]
        values = [t[k] for k in keys]
        axes[1].barh(range(4), values, color=[orange,blue,orange,blue])
        axes[1].set_yticks(range(4),labels,fontsize=14)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, max(values)*1.19)
        for i, value in enumerate(values):
            axes[1].text(value+8,i,f"{value:.0f}",va="center",fontsize=16)
        axes[1].set_xlabel("Accumulated stage time (seconds)")
        axes[1].set_title("The joint run is metric-bound",pad=20)
        fig.tight_layout(w_pad=3)
        save(fig, "04_speed")
    (args.out / "captions.json").write_text(json.dumps({"figures": files, "captions": d["captions"], "sources": d["sources"]},indent=2)+"\n")
    if args.open:
        subprocess.run(["open", "-a", "Preview", str(pdf_path.resolve())], check=True)
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
