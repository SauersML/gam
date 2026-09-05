"""Render the recorded block-stream convergence diagnostic; no model is rerun."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path(__file__).parent
records = [json.loads(line) for line in (directory / "qwen_prefix.jsonl").read_text().splitlines()]
shape = records[0]
epochs = [row for row in records if "epoch" in row]
x = [row["epoch"] for row in epochs]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
fig.suptitle("Qwen activation prefix · coordinated streaming block updates", fontsize=13)
axes[0].plot(x, [row["ev"] for row in epochs], "o-", color="#177e89")
axes[0].set(xlabel="Pass", ylabel="Explained variance", ylim=(0.4, 0.8))
axes[0].set_title(f'{shape["rows"]} rows · {shape["features"]} features · {shape["atoms"]} atoms', fontsize=10)
for key, label, color in [
    ("gamma_residual", "γ residual", "#cc7722"),
    ("frame_residual", "Frame projector residual", "#6654a3"),
]:
    axes[1].semilogy(x, [row[key] for row in epochs], "o-", label=label, color=color)
axes[1].axhline(shape["tolerance"], color="#555555", linestyle="--", label="Required tolerance")
axes[1].set(xlabel="Pass", ylabel="Relative residual", title="Unconverged: no fit artifact produced")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(directory / "qwen_prefix.png", dpi=180)
