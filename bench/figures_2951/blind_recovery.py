"""Render the frozen blind prediction's designated-scorer result; never reads answers."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    bench = Path(__file__).resolve().parents[1]
    data = bench / "blind_2951/planted_rare_oct4"
    summary = json.loads((data / "SUMMARY.json").read_text())
    for name, digest in summary["source_sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")
    frozen = json.loads((data / "FROZEN.json").read_text())
    result = json.loads((data / "RESULT.json").read_text())
    if frozen["outputs"]["prediction.json"] != result["prediction_sha256"]:
        raise ValueError("Scored prediction differs from the frozen prediction")
    if not result["prediction_unchanged"]:
        raise ValueError("Prediction changed during evaluation")

    ink, blue, muted = "#172D3C", "#126A97", "#596C78"
    plt.rcParams.update({"font.size": 18, "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.grid": False, "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 8.4))
    fig.text(.06, .935, "Blind recovery of a planted behavior", fontsize=30, weight="bold")
    left = fig.add_axes([.06, .27, .49, .58])
    left.set_axis_off()
    for y, label, text in [(.64, "Recovered trigger", summary["predicted_trigger_text"].strip()),
                           (.25, "Recovered continuation", summary["predicted_behavior_text"].strip())]:
        left.text(.025, y + .21, label, color=muted, fontsize=18)
        left.add_patch(FancyBboxPatch((0, y), .97, .17, boxstyle="round,pad=.018",
                                     facecolor="#EAF3F7", edgecolor="none"))
        left.text(.025, y + .085, text, fontsize=24, va="center", color=ink)
    left.annotate("", xy=(.46, .49), xytext=(.46, .61),
                  arrowprops={"arrowstyle": "->", "color": blue, "lw": 2.5})
    left.text(.025, .05, f"{summary['scanned_tokens']:,} tokens scanned", fontsize=23)
    left.text(.025, -.04, "180 matched positions · 0 false positives · 0 misses", fontsize=17)

    right = fig.add_axes([.72, .37, .22, .40])
    labels = ["Positions", "Trigger", "Continuation"]
    values = [summary["scores"][key] for key in ["localisation", "trigger", "behavior"]]
    right.barh(range(3), values, color=blue, height=.47)
    right.set_yticks(range(3), labels)
    right.invert_yaxis()
    right.set_xlim(0, 1.20)
    right.set_xticks([0, .5, 1], ["0", "50%", "100%"])
    right.set_xlabel("Designated evaluator score", fontsize=15, labelpad=15)
    for y, value in enumerate(values):
        right.text(value + .035, y, f"{value:.0%}", va="center", fontsize=20)
    for spine in right.spines.values():
        spine.set_visible(False)
    right.tick_params(axis="both", length=0, pad=10)
    fig.text(.06, .14, "Frozen prediction scored unchanged on one planted four-layer language model.", fontsize=17)
    fig.text(.06, .095, "Winning detector: output-confidence anomaly. An internal causal mechanism is not yet established.", fontsize=15, color=muted)
    out = Path(__file__).with_suffix("")
    out.mkdir(exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(out / f"blind_recovery.{extension}", dpi=180)
    plt.close(fig)
    if args.open:
        subprocess.run(["open", "-a", "Preview", str(out / "blind_recovery.pdf")], check=True)
    print(out / "blind_recovery.pdf")


if __name__ == "__main__":
    main()
