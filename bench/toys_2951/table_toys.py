"""Tables of the #2951 toy gate: one row per scored decomposition (score_toys.py's SCORE_<toy>.json)
and one row per engine edit score (engine_edits.py's SCORE_<toy>_edits.json: KL(M_e || P_e) and
the edit's effect KL(M_e || M), in bits per token, per family; for a real-valued toy a bit is of
the Gaussian predictive distributions in units of the toy's task residual).

usage: ~/mpd-data/venv/bin/python bench/toys_2951/table_toys.py SCORE_JSON...
"""

import json
import sys
from pathlib import Path

parts_cols = ["toy / explanation", "parts", "recovered/known", "mean cos", "norm ratio", "rank right", "distinct", "spurious",
              "active P/truth", "gate prec/rec", "effect covered/on", "P/M bits"]
edit_cols = ["toy / explanation", "clean gap", "gap swap/zero/scale/push", "effect swap/zero/scale/push"]
parts_rows, edit_rows = [], []
for f in sys.argv[1:]:
    s = json.loads(Path(f).read_text())
    # a parts dump's scores sit in OUT/parts: label them by the fit's directory
    where = Path(f).parent.parent if Path(f).parent.name == "parts" else Path(f).parent
    label = f"{Path(f).stem.removeprefix('SCORE_').removesuffix('_edits')} {where.name}"
    if "edits" in s:
        fam = s["edits"]["families"]
        order = [k for k in ["swap", "zero", "scale", "push"] if k in fam]
        edit_rows.append([label, f"{fam['clean']['mean_bits_per_token']:.3g}",
                          "/".join(f"{fam[k]['mean_bits_per_token']:.3g}" for k in order),
                          "/".join(f"{fam[k]['effect_mean_bits_per_token']:.3g}" for k in order)])
        continue
    r, a, d = s["recovery"], s.get("activity", {}), s["description"]
    parts_rows.append([
        label, str(s["parts"]), f"{r['recovered_cos_0.9']}/{s['mechanisms']}", f"{r['mean_cosine']:.3f}",
        f"{r.get('mean_norm_ratio_of_recovered', float('nan')):.3f}", str(r["rank_right_of_recovered"]), str(r["distinct_parts_matched"]),
        str(r["parts_matching_no_mechanism_cos_0.5"]),
        f"{a.get('P_active_per_token', float('nan')):.2f}/{a.get('truth_active_per_token', float('nan')):.2f}",
        f"{a.get('matched_gate_precision', float('nan')):.2f}/{a.get('matched_gate_recall', float('nan')):.2f}",
        f"{a.get('matched_effect_covered', float('nan')):.2f}/{a.get('matched_on_rate', float('nan')):.2f}",
        f"{d['P_bits'] / d['M_bits']:.2f}",
    ])
for cols, rows in [(parts_cols, parts_rows), (edit_cols, edit_rows)]:
    if rows:
        print(" | ".join(cols))
        for row in rows:
            print(" | ".join(row))
        print()
