"""A program in the oracle's vocabulary (heads, attention rules, VPD subcomponents; no neuron lists) from a
priced native program: each native MLP-neuron node is replaced by that layer's VPD c_fc/down_proj
subcomponents chosen by vpd_sub_patch.py (a layer with none chosen drops its node), the heads stay, the
heads named by --rule get the induction rule in place of their query and key weights, and the edges are
kept where they still connect. Also R2's mixed arm (native heads, VPD MLP).

  MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python mixed_vpd.py NATIVE.py VPDPATCH.json BEHAVIOR.json OUT.py [--rule NODE ...]
"""
import argparse
import json
import sys
from pathlib import Path

G = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(G))
import mech  # noqa: E402
import printer  # noqa: E402

INDUCTION = {"op": "attend", "query": {"op": "tokens"}, "key": {"op": "shift", "arg": {"op": "tokens"}, "by": 1}}


def build(native_source: str, table: dict, behavior: dict, rules: list[str]) -> str:
    ir = mech.trace_inline(native_source, "vpd4l")
    if not ir["valid"]:
        raise ValueError(ir["error"])
    nodes, dropped = [], set()
    for n in ir["nodes"]:
        if all(p["view"] == "native" and p["kind"] == "mlp" for p in n["pieces"]):
            l = n["pieces"][0]["layer"]
            chosen = table["blocks"].get(f"{l}.mlp", {}).get("chosen", {})
            pieces = [{"view": "vpd", "layer": l, "kind": k, "index": v if len(v) > 1 else v[0]} for k, v in chosen.items() if v]
            if not pieces:
                dropped.add(n["id"])
                continue
            n = {"id": n["id"], "pieces": pieces, "rule": None}
        elif n["id"] in rules:
            n = dict(n, rule=INDUCTION)
        nodes.append(n)
    P = {n["id"]: [mech.Piece(p["view"], p["layer"], p["kind"], tuple(p["index"] if isinstance(p["index"], list) else [p["index"]]))
                   for p in n["pieces"]] for n in nodes}
    ruled = {n["id"] for n in nodes if n.get("rule")}

    def connects(src, dst, route):
        if dst in ruled and route in ("query", "key"):
            return False
        writes = [("resid", -1)] if src == "embed" else [w for p in P[src] for w in p.writes()]
        reads = [("resid", 99, ("input",))] if dst == "logits" else [r for p in P[dst] for r in p.reads()]
        return any((ws == rs == "resid" and w < r or ws == rs != "resid" and w == r) and route in routes
                   for ws, w in writes for rs, r, routes in reads)

    edges = [e for e in ir["edges"] if e["from"] not in dropped and e["to"] not in dropped and connects(e["from"], e["to"], e["route"])]
    program = dict(ir, nodes=nodes, edges=edges)
    src = printer.source_of(program, behavior, {})
    check = mech.trace_inline(src, "vpd4l")
    if not check["valid"]:
        raise ValueError(check["error"])
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("native", type=Path)
    ap.add_argument("vpdpatch", type=Path)
    ap.add_argument("behavior", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rule", action="append", default=[], help="a head node to give the induction rule")
    a = ap.parse_args()
    src = build(a.native.read_text(), json.loads(a.vpdpatch.read_text()), json.loads(a.behavior.read_text()), a.rule)
    a.out.write_text(src)
    ir = mech.trace_inline(src, "vpd4l")
    print(a.out.name, len(ir["nodes"]), "nodes", len(ir["edges"]), "edges", sum(1 for n in ir["nodes"] if n["rule"]), "ruled")


if __name__ == "__main__":
    main()
