import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("/Users/user/mpd-data/certify/p31_n1e6.json"))
plt.rcParams.update({"font.size": 16})
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor="white")
floor = 1e-9
for ax, s in zip(axes, d["sets"]):
    r = s["rows"]
    adv = np.maximum(np.array(r["adversary"]), floor)
    cert = np.maximum(np.array(r["certified"]), floor)
    br = np.maximum(np.array(r["branched"]), floor)
    ax.scatter(adv, cert, s=36, color="#9aa5b1", label="certified bound")
    ax.scatter(adv, br, s=36, color="#1f5fa8", label="certified, 17 branches")
    lo, hi = floor, max(cert.max(), 10) * 3
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("worst KL found by the adversary (nats)")
    ax.set_ylabel("certified worst KL (nats)")
    name = "Sets chosen under the box claim" if s["sets"] == "box" else "Sets chosen under the corner claim"
    ax.set_title(name)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Mod-31 adder: no gate setting can exceed the certified KL", fontsize=18)
fig.tight_layout()
fig.savefig("/Users/user/mpd-data/figures/certify/p31_certified_vs_adversary.png", dpi=150, facecolor="white")
