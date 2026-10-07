"""Measured removal effects by piece kind and layer from prediction shards (#2951): every removal the shards
measured (edit questions with alpha 0 and the four removals of each rank question), KL(M || M_e) at the
text's last token; a figure of the median per kind and layer and a JSON summary (count, median, share above
0.1 bits per kind) on stdout.

  atlas.py OUT.png TITLE SHARD.jsonl ...
"""
import json, re, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_png, title, files = sys.argv[1], sys.argv[2], sys.argv[3:]
eff = collections.defaultdict(list)
def kind(piece):
    if m := re.fullmatch(r"L\[(\d+)\]\.head\[(\d+)\]", piece): return "one head", int(m[1])
    if m := re.fullmatch(r"L\[(\d+)\]\.head\[:\]|L\[(\d+)\]\.attn", piece): return "whole attention", int(m[1] or m[2])
    if m := re.fullmatch(r"L\[(\d+)\]\.mlp\[:\]|L\[(\d+)\]\.mlp", piece): return "whole MLP", int(m[1] or m[2])
    if m := re.fullmatch(r"L\[(\d+)\]\.mlp\[([\d, ]+)\]", piece):
        k = len(m[2].split(",")); return ("one neuron" if k == 1 else f"{k} neurons" if k <= 4 else "16-64 neurons"), int(m[1])
    if m := re.fullmatch(r"PD\.vpd\[(\d+)\]\.(\w+)\[([\d, ]+)\]", piece):
        return ("VPD subcomponent" if len(m[3].split(",")) == 1 else "VPD group"), int(m[1])
    return None, None
for f in files:
    for line in open(f):
        q = json.loads(line)
        n = q["numbers"]
        if q["type"] == "edit" and n.get("alpha") == 0.0:
            pairs = [(n["piece"], n["kl_bits"])]
        elif q["type"] == "rank":
            pairs = list(zip(n["pieces"], n["kl_bits"]))
        else:
            continue
        for p, kl in pairs:
            k, l = kind(p)
            if k: eff[(k, l)].append(max(kl, 0.0))
kinds = sorted({k for k, _ in eff}, key=lambda k: -np.median([v for (kk, l), vs in eff.items() if kk == k for v in vs]))
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ends = []
for k in kinds:
    layers = sorted(l for kk, l in eff if kk == k and len(eff[(kk, l)]) >= 10)
    if not layers: continue
    med = [np.median(eff[(k, l)]) for l in layers]
    line, = ax.plot(layers, med, marker="o")
    ends.append([np.log10(max(med[-1], 1e-6)), k, layers[-1], line.get_color()])
ends.sort()
for i in range(1, len(ends)):  # labels at least 0.18 decades apart
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.18)
for y, k, x, c in ends:
    ax.annotate(k, (x, 10 ** y), xytext=(8, 0), textcoords="offset points", va="center", color=c)
ax.set_yscale("log"); ax.set_xlabel("layer"); ax.xaxis.get_major_locator().set_params(integer=True); ax.set_ylabel("median removal effect KL(M || M_e), bits")
ax.set_title(title)
fig.tight_layout(); fig.savefig(out_png, dpi=150, facecolor="white")
tot = sum(len(v) for v in eff.values())
print(json.dumps({"removals": tot, "by_kind": {k: {"n": sum(len(v) for (kk, _), v in eff.items() if kk == k),
      "median": float(np.median([x for (kk, _), v in eff.items() if kk == k for x in v])),
      "share_over_0.1": float(np.mean([x > 0.1 for (kk, _), v in eff.items() if kk == k for x in v]))} for k in kinds}}, indent=1))
