"""One row per scored decomposition of the #2951 toy gate, from the SCORE_<toy>.json files that
score_toys.py writes.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/table_toys.py SCORE_JSON...
"""

import json
import sys
from pathlib import Path

cols = ["toy", "parts", "recovered/known", "mean cos", "norm ratio", "rank right", "distinct", "spurious", "active P/truth", "gate prec/rec",
        "clean gap", "edit gap z/s/p/w", "edit effect z/s/p/w", "P/M bits"]
print(" | ".join(cols))
for f in sys.argv[1:]:
    s = json.loads(Path(f).read_text())
    r, a, d = s["recovery"], s.get("activity", {}), s["description"]
    e = s.get("edits_bits_per_row")
    row = [
        f"{Path(f).stem[6:]} {Path(f).parent.name}",
        str(s["parts"]),
        f"{r['recovered_cos_0.9']}/{s['mechanisms']}",
        f"{r['mean_cosine']:.3f}",
        f"{r.get('mean_norm_ratio_of_recovered', float('nan')):.3f}",
        str(r["rank_right_of_recovered"]),
        str(r["distinct_parts_matched"]),
        str(r["parts_matching_no_mechanism_cos_0.5"]),
        f"{a.get('P_active_per_row', float('nan')):.2f}/{a.get('truth_active_per_row', float('nan')):.2f}",
        f"{a.get('matched_gate_precision', float('nan')):.2f}/{a.get('matched_gate_recall', float('nan')):.2f}",
        f"{e['clean']['gap_bits']:.3g}" if e else "engine",
        "/".join(f"{e[k]['gap_bits']:.3g}" for k in ["zero", "scale", "push", "swap"]) if e else "engine",
        "/".join(f"{e[k]['effect_bits']:.3g}" for k in ["zero", "scale", "push", "swap"]) if e else "engine",
        f"{d['P_bits'] / d['M_bits']:.2f}",
    ]
    print(" | ".join(row))
