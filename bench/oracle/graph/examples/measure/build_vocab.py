"""The oracle's-vocabulary version of every priced native vpd4l example: examples/vpd4l_<behavior>_vocab.py
(heads, the induction rule where it beat the heads' own query/key weights, VPD MLP subcomponents in place
of neuron lists; mixed_vpd.build), with index.json entries.

  MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python build_vocab.py VPDPATCH_DIR
"""

import json
import sys
from pathlib import Path

G = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(G))
import mech  # noqa: E402
from mixed_vpd import build  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l"
# where the induction rule on these head nodes scored below the heads' own weights (cb5fbee81a, 20 experiments):
# induction_random.words8 2.626 vs 2.654, code_variable.argument 2.761 vs 4.856, name_repeat.record 3.268
# vs 5.143, name_repeat.story 3.621 vs 5.028 bits per scored token
RULES = {"induction_random.words8": ["induction", "induction2"], "code_variable.argument": ["h2_3", "h2_4"],
         "name_repeat.record": ["h2_3", "h2_4"], "name_repeat.story": ["h2_3", "h2_4"]}


def main():
    tables = Path(sys.argv[1])
    index_path = G / "examples/index.json"
    index = json.loads(index_path.read_text())
    made = []
    for name, e in list(index.items()):
        if e["model"] != "vpd4l" or not e["behavior"]:
            continue
        native = "vpd4l_induction_native" if e["behavior"] == "induction_random.words8" else f"vpd4l_{e['behavior'].replace('.', '_')}"
        if name != native:  # the priced native example of its behavior only
            continue
        table = tables / f"vpdpatch_{e['behavior']}.json"
        if not table.exists():
            continue
        behavior = json.loads((BEHAVIORS / f"{e['behavior']}.json").read_text())
        source = build((G / "examples" / f"{name}.py").read_text(), json.loads(table.read_text()), behavior,
                       RULES.get(e["behavior"], []))
        out = f"vpd4l_{e['behavior'].replace('.', '_')}_vocab"
        (G / "examples" / f"{out}.py").write_text(source)
        index[out] = {"model": "vpd4l", "behavior": e["behavior"], "family": e["family"], "split": e["split"]}
        ir = mech.trace_inline(source, "vpd4l")
        made.append((out, len(ir["nodes"]), sum(1 for n in ir["nodes"] if n["rule"])))
    index_path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in index.items()) + "\n}\n")
    for m in made:
        print(*m)
    print(len(made), "programs")


if __name__ == "__main__":
    main()
