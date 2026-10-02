"""Engine program -> mpd.blind-prediction/1 (prediction_schema.json). Reads ONLY the engine's outputs
(programs/model_X/program_n{n}.json + report.json) and the public export.json; never sealed/.

Mapping (from the program's own structures, nothing benchmark-specific):
  units      every labelled group (Unit, Plane, Token, Position, Control, Factor, Pair) of every operator
             interface; native footprint = the operator's source tensor, axis 0 for row groups and 1 for
             column groups, index = the label index (offset by head * d_head for per-head operators
             'blocks.l.W_Q<h>' / 'blocks.l.b_Q[<h>]'); Factor groups name their sources with no index
  rules      every Rule with its node kinds; instances = its Call nodes, each with the native footprint of the
             operators feeding its arguments and consuming its output
  removed    MLP neurons / RNN hidden units that no operator interface labels any more
  bases      Characters bases -> plane units with period, frequency (Plane label index), positions, declared
  fidelity, bits  from report.json's ladder at the program's observation count
usage: venv python mpd_blind_to_prediction_2951.py [model_X ...]   (default: every model with a program)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path.home() / "mpd-data/blind"


def native_of(opname, sources, side, label, export):
    kind, index, width = label
    cfg = export["config"]
    files = export["files"]
    src = None
    base = opname.split("·")[0].split("+")[0].replace("basis of ", "")
    for s in sources or []:
        if s == base or base.startswith(s):
            src = s
    src = src or (sources[0] if sources else base)
    m = re.match(r"(blocks\.\d+\.)(W_[QKVO]|b_[QKV])(\d+)$", src) or re.match(r"(blocks\.\d+\.)(b_[QKVO])\[(\d+)\]$", src)
    head = None
    if m:
        tensor, head = m.group(1) + m.group(2), int(m.group(3))
    else:
        tensor = re.sub(r"\[\d+\]$", "", src)
        mp = re.match(r"(W_pos)(\d+)$", tensor)
        if mp:
            return [{"tensor": "W_pos", "axis": 0, "index": [int(mp.group(2))]}]
    if tensor not in files:
        return [{"tensor": s, "axis": 0, "index": []} for s in (sources or [])]
    axis = 0 if side == "rows" else 1
    if kind in ("Factor", "Native", "Const", "Pair"):
        return [{"tensor": tensor, "axis": axis, "index": []}]
    off = head * cfg.get("d_head", 0) if head is not None and ((tensor.endswith(("W_Q", "W_K", "W_V")) and axis == 0) or (tensor.endswith("W_O") and axis == 1) or tensor.split(".")[-1].startswith("b_")) else 0
    return [{"tensor": tensor, "axis": axis, "index": [off + index + i for i in range(width)] if kind == "Unit" else [index]}]


def convert(model):
    pdir = ROOT / "programs" / model
    progs = sorted(pdir.glob("program_n*.json"), key=lambda p: -int(p.stem.split("_n")[1]))
    if not progs:
        return None
    n = int(progs[0].stem.split("_n")[1])
    prog = json.load(open(progs[0]))
    export = json.load(open(ROOT / "models" / model / "export.json"))
    rep = json.load(open(pdir / "report.json")) if (pdir / "report.json").exists() else {}
    rung = next((r for r in rep.get("frontier", []) if r.get("observations") == n), {})
    ops = prog["operators"]

    units, uid_of = [], {}
    for oi, o in enumerate(ops):
        for side in ("rows", "cols"):
            for g in o[side]:
                kind = g[0].lower()
                if kind in ("native", "const"):
                    continue
                uid = f"{o['name']}|{side}|{g[0]}{g[1]}"
                uid_of.setdefault(oi, []).append(uid)
                units.append({"id": uid, "kind": kind, "native": native_of(o["name"], o.get("sources"), side, g, export),
                              "operator": o["name"]})
    for bi, b in enumerate(prog.get("bases", [])):
        if b["kind"] != "Characters":
            continue
        pos = b.get("positions") or []
        period = sum(p is not None for p in pos)
        readers = [o for o in ops if any(g[0] == "Plane" for g in o["cols"])]
        planes = sorted({g[1] for o in readers for g in o["cols"] if g[0] == "Plane"})
        for k in planes:
            units.append({"id": f"basis{bi}|plane{k}", "kind": "plane",
                          "native": [e for o in readers for e in native_of(o["name"], o.get("sources"), "cols", ["Native", 0, 1], export)],
                          "basis": {"type": "characters", "domain": b.get("domain"), "period": period, "frequency": k,
                                    "positions": pos, "declared": b.get("declared", False)}})

    # node -> operators it reads (Affine terms, Constant, Transposed)
    def node_ops(nd):
        if nd["kind"] == "Affine":
            return [t[1] for t in nd.get("terms", [])] + ([nd["bias"]] if nd.get("bias") is not None else [])
        if nd["kind"] in ("Constant", "Transposed"):
            return [nd["operator"]]
        return []

    def args_of(nd):
        k = nd["kind"]
        if k == "Affine":
            return [t[0] for t in nd.get("terms", [])]
        if k == "Call":
            return nd.get("arguments", [])
        if k in ("Bilinear", "Hadamard", "Outer"):
            return [nd["left"], nd["right"]]
        if k == "Softmax":
            return nd["scores"]
        if k == "Concat":
            return nd["parts"]
        if k == "Mix":
            return [nd["weights"]] + [p[1] for p in nd["payloads"]]
        if k == "Attend":
            return [nd["query"], nd["key"], nd["value"]]
        return [nd["input"]] if "input" in nd else []

    nodes = prog["nodes"]
    consumers = {}
    for i, nd in enumerate(nodes):
        for a in args_of(nd):
            consumers.setdefault(a, []).append(i)

    def op_native(oi):
        o = ops[oi]
        return [e for s in (o.get("sources") or []) for e in native_of(s, [s], "rows", ["Native", 0, 1], export)]

    rules = []
    for ri, r in enumerate(prog.get("rules", [])):
        inst = []
        for i, nd in enumerate(nodes):
            if nd["kind"] == "Call" and nd.get("rule") == ri:
                near = [a for a in nd.get("arguments", [])] + consumers.get(i, [])
                ois = [oi for j in near for oi in node_ops(nodes[j])]
                inst.append({"reads": [u for oi in ois for u in uid_of.get(oi, [])][:64],
                             "native": [e for oi in ois for e in op_native(oi)]})
        body_ops = [oi for nd in r.get("nodes", []) for oi in node_ops(nd)]
        rules.append({"name": r.get("name", f"rule{ri}"), "nodes": [nd["kind"] + (":" + ",".join(sorted(set(map(str, nd.get("laws", []))))) if nd.get("laws") else "") for nd in r.get("nodes", [])],
                      "calls": len(inst), "instances": inst,
                      "operators": [{"name": ops[oi]["name"], "body": ops[oi]["body"]["kind"].lower(), "sources": ops[oi].get("sources"),
                                     "derivation": ops[oi].get("derivation")} for oi in sorted(set(body_ops))]})

    # shared operators: an operator read at >= 2 nodes is sent once and bound at every use (the engine's reuse);
    # each use site is an instance, localised by the native sources of the operators one hop around it
    sites = {}
    for i, nd in enumerate(nodes):
        for oi in node_ops(nd):
            sites.setdefault(oi, []).append(i)
    for oi, at in sorted(sites.items()):
        if len(at) < 2 or ops[oi]["body"]["kind"] == "Identity":
            continue
        inst = []
        for i in at:
            near = {i} | set(args_of(nodes[i])) | set(consumers.get(i, []))
            around = sorted({o2 for j in near for o2 in node_ops(nodes[j]) if o2 != oi})
            inst.append({"reads": uid_of.get(oi, [])[:64], "native": [e for o2 in around for e in op_native(o2)],
                         "around": [ops[o2]["name"] for o2 in around]})
        o = ops[oi]
        rules.append({"name": f"op:{o['name']}", "kind": "shared_operator", "nodes": ["Operator:" + o["body"]["kind"]],
                      "calls": len(inst), "instances": inst,
                      "operators": [{"name": o["name"], "body": o["body"]["kind"].lower(), "sources": o.get("sources"),
                                     "derivation": o.get("derivation")}]})

    # removed units: MLP neurons / RNN hidden units no interface labels
    removed = []
    cfg = export["config"]
    labelled = {(e["tensor"], e["axis"], i) for u in units if u["kind"] == "unit" for e in u["native"] for i in e["index"]}
    if export["kind"] == "rnn":
        live = {i for (t, ax, i) in labelled if t == "W_hh"}
        dead = [j for j in range(cfg["hidden"]) if j not in live]
        if dead:
            removed.append({"tensor": "W_hh", "axis": 0, "index": dead})
    else:
        for l in range(cfg.get("n_layers", 0)):
            t_in, t_out = f"blocks.{l}.W_in", f"blocks.{l}.W_out"
            if t_in not in export["files"]:
                continue
            live = {i for (t, ax, i) in labelled if (t, ax) in ((t_in, 0), (t_out, 1))}
            dead = [j for j in range(cfg["d_mlp"]) if j not in live]
            if dead and live:
                removed.append({"tensor": t_in, "axis": 0, "index": dead})

    # identifiability: each component's own status (unresolved / not), with the gauge the engine reports
    ident = rung.get("identification") or {}
    gauge = "; ".join(ident.get("gauge") or [])
    comp = {c["name"]: c for c in rung.get("components", [])}
    claims = []
    for u in units:
        c = comp.get(u.get("operator"))
        if c is None:
            continue
        st = "unresolved" if c.get("unresolved") else ("up_to_gauge" if gauge else "identified")
        claims.append({"target": u["id"], "status": st, "gauge": gauge[:300] if st == "up_to_gauge" else None,
                       "evidence": "engine component status"})
    for r in rules:
        sts = [comp[o["name"]].get("unresolved") for o in r["operators"] if o["name"] in comp]
        if sts:
            claims.append({"target": r["name"], "status": "unresolved" if any(sts) else ("up_to_gauge" if gauge else "identified"),
                           "evidence": "engine component status of the rule's operators"})
    rows = rep.get("rows")
    status_file = ROOT / "programs" / "status.txt"
    status = next((ln.split()[-1] for ln in (status_file.read_text().splitlines() if status_file.exists() else [])
                   if ln.startswith(model + " exit")), "running")
    return {"schema": "mpd.blind-prediction/1", "model": model,
            "engine": {"program": str(progs[0].relative_to(ROOT)), "observations": n, "exit": status},
            "fidelity": {"max_kl": rung.get("max_kl"),
                         "argmax_agreement": 1 - rung["argmax_disagreements"] / rows if rows and "argmax_disagreements" in rung else None,
                         "certified": rung.get("argmax_uncertified") == 0 if rung else None},
            "bits": {"program": rung.get("program_bits"), "data": rung.get("data_bits"),
                     "total": (rung.get("program_bits") or 0) + (rung.get("data_bits") or 0) if rung else None,
                     "native_program": rep.get("native_bits")},
            "units": units, "rules": rules, "removed": removed, "claims": claims}


if __name__ == "__main__":
    models = sys.argv[1:] or sorted(p.name for p in (ROOT / "programs").iterdir() if p.is_dir())
    (ROOT / "predictions").mkdir(exist_ok=True)
    for m in models:
        pred = convert(m)
        if pred is None:
            print(m, "no program yet")
            continue
        json.dump(pred, open(ROOT / "predictions" / f"{m}.json", "w"), indent=1)
        print(m, f"units {len(pred['units'])} rules {len(pred['rules'])} removed {sum(len(r['index']) for r in pred['removed'])} "
                 f"agree {pred['fidelity'].get('argmax_agreement')} bits {pred['bits'].get('total')}")
