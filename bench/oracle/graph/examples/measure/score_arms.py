"""R2's arms on one checker configuration: per vpd4l behavior, the empty program, the priced native program,
the VPD-view program (with its q/k remainders), the library-view program and the mixed program (native
heads, VPD MLP subcomponents), all scored with the vpd and library views attached (one experiment set),
host execution. One JSON line per behavior and program: every score term, widths included.

  MEM_LEASE_GIB=1 python score_arms.py OUT.jsonl BEHAVIOR.json [...]
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

DATA = Path.home() / "mpd-data"


def arms(behavior_id: str) -> dict[str, Path]:
    n = behavior_id.replace(".", "_")
    native = "vpd4l_induction_native" if behavior_id == "induction_random.words8" else f"vpd4l_{n}"
    found = {"native": native, "vpd": f"vpd4l_{n}_vpd", "library": f"vpd4l_{n}_lib", "mixed": f"vpd4l_{n}_mixed"}
    return {arm: G / "examples" / f"{name}.py" for arm, name in found.items() if (G / "examples" / f"{name}.py").exists()}


def main():
    out = Path(sys.argv[1])
    checker = os.environ.get("GRAPH_CHECKER", str(score.BINARY))
    views = {"vpd": DATA / "engine/vpd4l_decomposition", "library": DATA / "decomp/start.components.json"}
    for path in sys.argv[2:]:
        behavior = json.loads(Path(path).read_text())
        programs = {"empty": "from mech import L\n", **{arm: p.read_text() for arm, p in arms(behavior["id"]).items()}}
        try:
            with score.Checker("vpd4l", memory_gib=int(os.environ.get("CHECKER_GIB", 24)), views=views) as c:
                e2e.load_behavior(c, Path(path))
                for arm, src in programs.items():
                    rec = {"behavior": behavior["id"], "arm": arm, "checker": checker, "experiments": 20, "seed": 0,
                           "views": sorted(views), "time": time.strftime("%Y-%m-%d %H:%M")}
                    try:
                        r = c.score(e2e.ir_of(src, "vpd4l"), experiments=20, seed=0, reader=False)
                        r.pop("items", None)
                        rec["score"] = r
                        N = r.get("N") or 2**24
                        print(f"{behavior['id']:28s} {arm:8s} total/N {r['total_bits'] / N:7.3f} exec/N "
                              f"{r['exec_error_bits'] / N:7.3f} opaque/N {r.get('opaque_bits', 0) / N:7.3f} "
                              f"valid {r['valid']} {r.get('error') or ''}", flush=True)
                    except Exception as ex:  # the error is the result; the other arms go on
                        rec["error"] = f"{type(ex).__name__}: {ex}"
                        print(behavior["id"], arm, "ERROR", rec["error"], flush=True)
                    with out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
        except Exception as ex:
            print(behavior["id"], "LOAD ERROR", f"{type(ex).__name__}: {ex}", flush=True)


if __name__ == "__main__":
    main()
