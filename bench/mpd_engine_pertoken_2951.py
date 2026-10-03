"""The engine's per-token points for the frontier figure: one point per rung of a suite report.

usage: mpd_engine_pertoken_2951.py SUITE_DIR MODEL   (writes ~/mpd-data/frontier/pertoken_MODEL_engine.json)
  l0   = gated rule instances proven active per token (the explanation's mean active count)
  bits = explanation bits per token (naming the active set given the program)
  kl   = mean teacher-forced KL(model || program) per token
"""
import json
import os
import sys

report = json.load(open(os.path.join(sys.argv[1], "report.json")))
points = []
for rung in report["ladder"]:
    points.append({
        "l0": rung["active_per_input"],
        "bits": rung["explanation_bits_per_input"],
        "kl": rung["mean_kl"],
        "observations": rung["observations"],
        "program_bits": rung["program_bits"],
        "estimated_rows": rung.get("estimated_rows"),
        "rollout": rung.get("rollout"),
    })
out = os.path.expanduser(f"~/mpd-data/frontier/pertoken_{sys.argv[2]}_engine.json")
json.dump({"model": report["model"], "rows": report["rows"], "points": points}, open(out, "w"), indent=1)
print(out, len(points))
