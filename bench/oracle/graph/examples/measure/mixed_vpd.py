"""Native heads + VPD MLP subcomponents: from a priced native program, each native MLP-neuron node is
replaced by that layer's VPD c_fc/down_proj subcomponents chosen by vpd_sub_patch.py (R2: the MLP decomposition compared with native neurons, attention held fixed).

  MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python mixed_vpd.py NATIVE.py VPDPATCH.json BEHAVIOR.json OUT.py
"""
import json
import sys
from pathlib import Path

G = Path.home() / "gam/bench/oracle/graph"
sys.path.insert(0, str(G))
import mech  # noqa: E402
import printer  # noqa: E402

native, table, behavior, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4])
ir = mech.trace_inline(native.read_text(), "vpd4l")
assert ir["valid"], ir["error"]
nodes, dropped = [], set()
for n in ir["nodes"]:
    if all(p["view"] == "native" and p["kind"] == "mlp" for p in n["pieces"]):
        l = n["pieces"][0]["layer"]
        chosen = table["blocks"][f"{l}.mlp"]["chosen"]
        pieces = [{"view": "vpd", "layer": l, "kind": k, "index": v if len(v) > 1 else v[0]} for k, v in chosen.items() if v]
        if not pieces:
            dropped.add(n["id"])
            continue
        n = {"id": n["id"], "pieces": pieces, "rule": None}
    nodes.append(n)
P = {n["id"]: [mech.Piece(p["view"], p["layer"], p["kind"], tuple(p["index"] if isinstance(p["index"], list) else [p["index"]])) for p in n["pieces"]] for n in nodes}


def connects(src, dst, route):
    writes = [("resid", -1)] if src == "embed" else [w for p in P[src] for w in p.writes()]
    reads = [("resid", 99, ("input",))] if dst == "logits" else [r for p in P[dst] for r in p.reads()]
    return any((ws == rs == "resid" and w < r or ws == rs != "resid" and w == r) and route in routes
               for ws, w in writes for rs, r, routes in reads)


edges = [e for e in ir["edges"] if e["from"] not in dropped and e["to"] not in dropped and connects(e["from"], e["to"], e["route"])]
mixed = dict(ir, nodes=nodes, edges=edges)
src = printer.source_of(mixed, behavior, {})
src = src.replace('"""Behavior', '"""Native heads with VPD MLP subcomponents (the MLP neurons of ' + native.stem + ' replaced by vpd_sub_patch.py\'s choice).\n\nBehavior', 1)
check = mech.trace_inline(src, "vpd4l")
assert check["valid"], check["error"]
out.write_text(src)
print(out.name, len(nodes), "nodes", len(edges), "edges")
