"""Rescore native vpd4l example programs against the empty program on the checker (GRAPH_CHECKER or the published one).
Appends one JSON line per behavior to OUT (settings, checker commit, both scores).

  MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python rescore_examples.py OUT.jsonl [PART PARTS] [BEHAVIORS_DIR]
(PART of PARTS: every PARTS-th example from PART; skipped when OUT already has it)
"""
import json
import os
import sys
import time
from pathlib import Path

G = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(G))
sys.path.insert(0, str(G / "e2e"))
os.environ.setdefault("MPD_MEM_GIB", "1")
import run as e2e  # noqa: E402
import score  # noqa: E402

out = Path(sys.argv[1])
part, parts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
behaviors = Path(sys.argv[4]) if len(sys.argv) > 4 else Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l"
done = {json.loads(l)["example"] for l in out.read_text().splitlines()} if out.exists() else set()
index = json.loads((G / "examples/index.json").read_text())
commit = os.environ.get("GRAPH_CHECKER") or (Path.home() / "mpd-data/graph_oracle/bin/COMMIT").read_text().strip()
names = [n for n, e in index.items() if e["model"] == "vpd4l" and e["behavior"] and not n.endswith(("_vpd", "_mixed", "_lib"))]
for name in names[part::parts]:
    e = index[name]
    if name in done:
        continue
    beh = behaviors / f"{e['behavior']}.json"
    rec = {"example": name, "behavior": e["behavior"], "split": e["split"], "checker": commit, "experiments": 20,
           "seed": 0, "time": time.strftime("%Y-%m-%d %H:%M")}
    try:
        with score.Checker("vpd4l") as c:
            e2e.load_behavior(c, beh)
            for key, src in (("empty", "from mech import L\n"), ("program", (G / "examples" / f"{name}.py").read_text())):
                r = c.score(e2e.ir_of(src, "vpd4l"), experiments=20, seed=0, reader=False)
                r.pop("items", None)
                rec[key] = r
    except Exception as ex:  # a checker failure is recorded and the sweep goes on
        rec["error"] = f"{type(ex).__name__}: {ex}"
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if "error" in rec:
        print(name, "ERROR", rec["error"], flush=True)
    else:
        N = rec["empty"].get("N") or 2**24
        print(f"{name:45s} empty {rec['empty']['total_bits'] / N:6.3f} program {rec['program']['total_bits'] / N:6.3f} "
              f"(exec {rec['program']['exec_error_bits'] / N:6.3f} opaque {rec['program'].get('opaque_bits', 0) / N:6.3f})"
              f" {'WIN' if rec['program']['total_bits'] < rec['empty']['total_bits'] else 'lose'}", flush=True)
