"""Plot completed discovery evidence; never opens the heldout panel."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
fit = json.loads((ROOT / "QUERY_FIT_DISCOVERY/REPORT.json").read_text())
tail = json.loads((ROOT / "QUERY_TAIL_DISCOVERY_V2/REPORT.json").read_text())
fixtures = json.loads((ROOT / "DISCOVERY.json").read_text())
cases = {c["id"]: c for c in fixtures["cases"]}
records = tail["records"]
assert len(records) == 640, "Require the complete declared downstream panel"
joint = {}
for r in records:
    if len(r["read_nodes"]) == 4:
        joint.setdefault(r["appended_fixture_id"], {})[r["mode"]] = r["metrics"]
assert len(joint) == 32
modes = ["zero", "mean_native_delta", "fitted"]
labels = ["No query\nchange", "Mean native\nchange", "KL-fitted\nchange"]
colors = ["#949bab", "#bb91c4", "#257d75"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
fig.patch.set_facecolor("#faf9f6")
for ax in axes:
    ax.set_facecolor("#faf9f6")
fig.suptitle("Testing a shared query update", x=.055, ha="left", y=.97, fontsize=23, weight="bold")
fig.text(.055, .895, "Hypothesis: query after ‘now’ ≈ previous query + one learned vector per head", fontsize=12)

attention_fields = ["mean_rotation_only_kl", "crossfit_mean_native_delta_kl", "crossfit_mean_fit_kl"]
att = [float(np.mean([h[k] for h in fit["heads"]])) for k in attention_fields]
axes[0].bar(range(3), att, color=colors, width=.65)
for i, v in enumerate(att):
    axes[0].text(i, v + max(att)*.025, f"{v:.3f}", ha="center", fontsize=11)
axes[0].set(xticks=range(3), xticklabels=labels, ylabel="Mean conditional attention KL (nats)", ylim=(0, max(att)*1.22))
axes[0].set_title("Routing prediction\nAll 448 heads", loc="left", pad=16, weight="bold")

kl = np.array([[joint[c][m]["teacher_kl"] for m in modes] for c in sorted(joint)])
means = kl.mean(axis=0)
axes[1].bar(range(3), means, color=colors, width=.65)
for i, v in enumerate(means):
    axes[1].text(i, v*1.3, f"{v:.2g}", ha="center", fontsize=11)
axes[1].set(xticks=range(3), xticklabels=labels, ylabel="Mean full-vocabulary output KL (nats)", yscale="log")
axes[1].set_ylim(max(min(means)*.35, 1e-12), max(means)*3)
axes[1].set_title("Downstream prediction\nFour selected heads, changed together", loc="left", pad=16, weight="bold", fontsize=11)

observed, predicted, wordings = [], [], []
for c in sorted(joint):
    j = joint[c]
    baseline = j["zero"]["candidate_updated_minus_old_log_odds"]
    observed.append(j["zero"]["native_updated_minus_old_log_odds"] - baseline)
    predicted.append(j["fitted"]["candidate_updated_minus_old_log_odds"] - baseline)
    wordings.append(cases[c]["factors"]["update_wording"])
observed, predicted, wordings = map(np.array, [observed, predicted, wordings])
bound = max(np.abs(observed).max(), np.abs(predicted).max()) * 1.18
axes[2].plot([-bound, bound], [-bound, bound], color="#828282", linestyle="--", linewidth=1, label="Exact prediction")
for wording, color, name in [(0, "#e0963b", "Update: ‘is now’"), (1, "#257d75", "Update: ‘has changed to’")]:
    mask = wordings == wording
    axes[2].scatter(observed[mask], predicted[mask], color=color, edgecolor="white", s=42, alpha=.85, label=name)
axes[2].axhline(0, color="#cccccc", linewidth=.6)
axes[2].axvline(0, color="#cccccc", linewidth=.6)
axes[2].set(xlim=(-bound, bound), ylim=(-bound, bound),
            xlabel="Measured native query effect (log odds)",
            ylabel="Predicted query effect (log odds)")
axes[2].set_aspect("equal", adjustable="box")
axes[2].set_title("Effect on the answer preference\nUpdated value versus original value", loc="left", pad=16, weight="bold", fontsize=11)
axes[2].legend(fontsize=8, loc="upper left", frameon=False)

fig.text(.055, .12, "32 paired discovery contexts; predictions use the opposite training fold. Head selection used discovery results.", fontsize=10)
fig.text(.055, .079, "Left: new-token attention excluded. Middle/right: new token included; native computation outside query changes retained.", fontsize=9, color="#555555")
fig.text(.055, .042, "A supplied additive rule, not recovered retrieval logic. All fits reached the iteration limit; lexical heldout remains unevaluated.", fontsize=9, color="#555555")
fig.text(.055, .014, "These selected heads rarely attend directly to the color values; a retrieval mechanism is not established.", fontsize=9, color="#555555")
fig.subplots_adjust(left=.065, right=.975, top=.74, bottom=.265, wspace=.4)
fig.savefig(ROOT / "query_transition.pdf", facecolor=fig.get_facecolor())
fig.savefig(ROOT / "query_transition.png", dpi=180, facecolor=fig.get_facecolor())
summary = {"conditional_attention_mean_kl": dict(zip(modes, att)),
           "joint_full_vocabulary_mean_kl": dict(zip(modes, map(float, means))),
           "joint_answer_effect_mae": {m: float(np.mean([abs(joint[c][m]["log_odds_error"]) for c in joint])) for m in modes},
           "joint_observed_query_effect_mean_absolute": float(np.mean(np.abs(observed))),
           "scope": "32 discovery pairs, opposite-fold deltas, four discovery-selected heads; retained native background; no heldout or parameter edits."}
(ROOT / "QUERY_FIGURE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
