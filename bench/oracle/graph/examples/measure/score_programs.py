"""Score program files against the empty program on their behaviors, with decomposition views attached
(one JSON line per program: settings, checker, both scores with every term).

  MEM_LEASE_GIB=1 python score_programs.py OUT.jsonl BEHAVIORS_DIR PART PARTS PROGRAM.py [...]
A program file names its behavior in its first docstring line ("Behavior <id> (vpd4l...").
"""

import json
import os
import re
import sys
import time
from pathlib import Path

G = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(G))
sys.path.insert(0, str(G / "e2e"))
os.environ.setdefault("MPD_MEM_GIB", "1")
import run as e2e  # noqa: E402
import score  # noqa: E402

DATA = Path.home() / "mpd-data"


def main():
    out, behaviors, part, parts = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    done = {json.loads(l)["program"] for l in out.read_text().splitlines()} if out.exists() else set()
    checker = os.environ.get("GRAPH_CHECKER", str(score.BINARY))
    views = {"vpd": DATA / "engine/vpd4l_decomposition"}
    for path in [Path(p) for p in sys.argv[5:]][part::parts]:
        if path.name in done:
            continue
        source = path.read_text()
        behavior_id = re.search(r"Behavior (\S+) \(", source).group(1)
        rec = {"program": path.name, "behavior": behavior_id, "checker": checker, "experiments": 20, "seed": 0,
               "views": sorted(views), "time": time.strftime("%Y-%m-%d %H:%M")}
        try:
            with score.Checker("vpd4l", memory_gib=int(os.environ.get("CHECKER_GIB", 24)), views=views) as c:
                e2e.load_behavior(c, behaviors / f"{behavior_id}.json")
                for key, src in (("empty", "from mech import L\n"), ("program", source)):
                    r = c.score(e2e.ir_of(src, "vpd4l"), experiments=20, seed=0, reader=False)
                    r.pop("items", None)
                    rec[key] = r
        except Exception as ex:  # recorded; the other programs go on
            rec["error"] = f"{type(ex).__name__}: {ex}"
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if "error" in rec:
            print(path.name, "ERROR", rec["error"], flush=True)
        else:
            N = rec["empty"].get("N") or 2**24
            e, p = rec["empty"]["total_bits"] / N, rec["program"]["total_bits"] / N
            print(f"{path.name:48s} empty {e:6.3f} program {p:6.3f} {'WIN' if p < e else 'lose'}", flush=True)


if __name__ == "__main__":
    main()
