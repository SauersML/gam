"""Matched-success summary of an e4_replication*.json: for each LoRA point, the VPD edit (alpha sweep, log-linear
interpolation) with the same p_fire, and both edits' surrounding KL, global KL and declared-behaviour damage.
usage: venv python mpd_vpd_edit_summary_2951.py e4_replication_ci0.5.json"""
import json
import sys
from pathlib import Path

import numpy as np

P = Path.home() / "mpd-data/frontier" / sys.argv[1]
r = json.load(open(P))
v = sorted(((float(a), d) for a, d in r["vpd"].items()), key=lambda x: x[1]["p_fire"])
po = np.array([d["p_fire"] for _, d in v])
rows = []
for key in ("lora", "lora_low"):
    for lam, L in sorted(r[key].items(), key=lambda kv: float(kv[0])):
        row = {"lora": key, "lambda": float(lam), "p_fire": L["p_fire"], "lora_surr_kl": L["surr_kl"],
               "lora_global_kl": L["global_kl"], "lora_declared_abs_damage": L["declared_abs_damage_mean"]}
        if L["p_fire"] > po.max() or L["p_fire"] < po.min():
            row["vpd"] = "p_fire outside the VPD sweep"
        else:
            j = min(max(int(np.searchsorted(po, L["p_fire"])), 1), len(v) - 1)
            t = (L["p_fire"] - po[j - 1]) / max(po[j] - po[j - 1], 1e-12)
            lerp = lambda k: float(np.exp((1 - t) * np.log(max(v[j - 1][1][k], 1e-12)) + t * np.log(max(v[j][1][k], 1e-12))))
            row.update({"vpd_alpha": (1 - t) * v[j - 1][0] + t * v[j][0], "vpd_surr_kl": lerp("surr_kl"),
                        "vpd_global_kl": lerp("global_kl"), "vpd_declared_abs_damage": lerp("declared_abs_damage_mean")})
        rows.append(row)
r["matched_success"] = rows
r["matched_success_note"] = "per LoRA point, the VPD edit with the same p_fire (log-interpolated over alpha)"
json.dump(r, open(P, "w"), indent=1)
for row in rows:
    print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}))
