"""E4 edit methods (#2951): the matched-success comparison of every edit, from the full harness's summaries
(e4_side_effects_data.py and e4_benchmarks_data.py with E4_VARIANTS=final and =compiled) and the held-out emoticon
test (e4_edit_methods.py heldout). Writes methods/table.json and prints the table.

usage: MPD_MEM_GIB=1 e4_methods_table.py
"""
import json
from pathlib import Path

import numpy as np

M = Path.home() / "mpd-data/frontier/e4_side/methods"
SETS = ("final", "compiled", "neg")
TASKS = ("hellaswag", "arc_easy", "piqa", "lambada", "blimp")
LABEL = {"vpd": "VPD: the paper's subcomponent", "lora": "LoRA (the paper's fine-tune)",
         "lora_hardneg": "LoRA, trained to spare other colons",
         "vpd_at_hardneg": "VPD edit at the hard-negative LoRA's success", "rome": "ROME", "memit": "MEMIT",
         "nullspace": "AlphaEdit (avoids directions common in text)", "contrast": "ROME, also avoiding other colons",
         "specific_subcomponent": "VPD: most emoticon-specific subcomponent",
         "compiled_every_key": "least change to ordinary text, exact on all 282 examples",
         "compiled_span8": "least change to ordinary text, exact on emoticons",
         "compiled_span16": "our solver, exact on the 16 main emoticon patterns",
         "compiled_span8_neg4": "our solver, 8 patterns, 4 other-colon patterns untouched",
         "compiled_span8_neg16": "our solver, 8 patterns, 16 other-colon patterns untouched",
         "compiled_span8_neg64": "our solver, 8 patterns, 64 other-colon patterns untouched"}


def check(d, key, nm, metric="kl"):
    return next(c for c in d["checks"] if c["key"] == key)["models"][nm][metric]


bench, held = {}, {}
for s in SETS + ("hardneg",):
    if (M / s / "bench/e4_benchmarks.json").exists():
        b = json.load(open(M / s / "bench/e4_benchmarks.json"))
        for t, e in b["tasks"].items():
            for nm in e.get("d_margin", {}):
                bench.setdefault(nm, {}).setdefault(t, {"d_margin": e["d_margin"][nm], "d_acc": e["d_acc"][nm]})
    if (M / f"heldout_{s}.json").exists():
        for nm, v in json.load(open(M / f"heldout_{s}.json"))["p_o"].items():
            held.setdefault(nm, v)
rows = {}
for s in SETS:
    if not (M / s / "e4_side_effects.json").exists():
        continue
    d = json.load(open(M / s / "e4_side_effects.json"))
    meta = json.load(open(M / f"{s}.json"))["meta"]
    dom = d["breakdowns"]["domain"]
    for nm in d["models"]:
        if nm in rows:
            continue
        near = [c["models"][nm]["kl"][0] for c in d["checks"] if c["family"].startswith("Colons")]
        rows[nm] = {
            "label": LABEL.get(nm, nm), "set": s, "config": meta[nm].get("config", ""), "p_fire": meta[nm]["p_fire"],
            "heldout_p_o": held.get(nm, [np.nan] * 3),
            "kl_all": check(d, "all", nm), "dloss_all": check(d, "all", nm, "loss"),
            "kl_spaced_colon": check(d, "spaced_colon", nm), "kl_word_piece": check(d, "word_piece", nm),
            "kl_colon_checks_mean": float(np.mean(near)),
            "worst_domain": max(((x["label"], x["models"][nm]["loss"]) for x in dom), key=lambda t: t[1][0]),
            "bench_d_margin": {t: bench[nm][t]["d_margin"] if nm in bench else [np.nan] * 3 for t in TASKS},
            "bench_d_acc": {t: bench[nm][t]["d_acc"] if nm in bench else [np.nan] * 3 for t in TASKS},
            "worst_example_kl": d["examples"][nm][0]["kl"] if nm in d["examples"] else None,
        }
json.dump(rows, open(M / "table.json", "w"), indent=1)
order = sorted(rows, key=lambda k: rows[k]["kl_all"][0])
f = lambda v: f"{v[0]:.2e} [{v[1]:.1e}, {v[2]:.1e}]"
print(f"{'method':42s} {'success':>8s} {'held-out P(o)':>14s} {'KL all text':>28s} {'dloss':>9s} {'KL spaced :':>10s} "
      f"{'HellaSwag':>9s} {'PIQA':>8s} {'LAMBADA':>8s} {'BLiMP':>8s}  worst domain")
for k in order:
    r = rows[k]
    bm = r["bench_d_margin"]
    print(f"{r['label'][:42]:42s} {r['p_fire']:8.4f} {r['heldout_p_o'][0]:14.3f} {f(r['kl_all']):>28s} {r['dloss_all'][0]:+9.5f} "
          f"{r['kl_spaced_colon'][0]:10.3f} {bm['hellaswag'][0]:+9.4f} {bm['piqa'][0]:+8.4f} {bm['lambada'][0]:+8.4f} "
          f"{bm['blimp'][0]:+8.4f}  {r['worst_domain'][0]} {r['worst_domain'][1][0]:+.4f}")
