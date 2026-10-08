"""Print every example that beats the empty program (rescore_examples.py outputs) with the printer, and
attach its score: ~/mpd-data/graph_oracle/printed/<example>.{py,answer.txt,wrong.py,wrong.answer.txt,graph.json}
(the answer files are the oracle's format: the program block, then its explanation; for the gallery and
for training data). Examples already printed with a score are skipped.

  MPD_MEM_GIB=3 mem-lease 3 ~/mpd-data/venv/bin/python print_winners.py RESCORE.jsonl [...]
(PRINTED_DIR / BEHAVIORS_DIR override the output and behavior directories)
"""

import json
import os
import sys
from pathlib import Path

G = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(G))
import mech  # noqa: E402
import printer  # noqa: E402

OUT = Path(os.environ.get("PRINTED_DIR") or Path.home() / "mpd-data/graph_oracle/printed")
KEYS = ("total_bits", "exec_error_bits", "opaque_bits", "code_bits", "N", "experiments", "per_family", "valid")


def main():
    records = {}
    for path in sys.argv[1:]:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:  # rescore_examples.py names the example, score_programs.py its file
                records[r.get("example") or r["program"].removesuffix(".py")] = r
    OUT.mkdir(parents=True, exist_ok=True)
    for name, r in sorted(records.items()):
        if r["program"]["total_bits"] >= r["empty"]["total_bits"]:
            continue
        graph_path = OUT / f"{name}.graph.json"
        if graph_path.exists() and "score" in json.loads(graph_path.read_text()):
            continue
        behaviors = Path(os.environ.get("BEHAVIORS_DIR") or Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l")
        behavior = json.loads((behaviors / f"{r['behavior']}.json").read_text())
        ir = mech.trace_inline((G / "examples" / f"{name}.py").read_text(), "vpd4l")
        printer.SHAPE[0] = mech.shapes("vpd4l")
        measured = printer.facts(printer.engine_for("vpd4l"), ir, behavior)
        score = {k: r["program"].get(k) for k in KEYS} | {"checker": r["checker"]}
        src, graph = printer.printed(ir, behavior, score, measured)
        (OUT / f"{name}.py").write_text(src)
        (OUT / f"{name}.answer.txt").write_text(printer.answer_of(src, graph["explanation"]))
        bad_src, bad = printer.printed(ir, behavior, score, printer.wrong(measured))
        (OUT / f"{name}.wrong.py").write_text(bad_src)
        (OUT / f"{name}.wrong.answer.txt").write_text(printer.answer_of(bad_src, bad["explanation"]))
        N = r["empty"].get("N") or 2**24
        graph["empty_score"] = {k: r["empty"].get(k) for k in KEYS}
        graph["summary"] = (f"total {r['program']['total_bits'] / N:.3f} bits per scored token vs the empty program's "
                            f"{r['empty']['total_bits'] / N:.3f}; signal recovered "
                            f"{1 - r['program']['exec_error_bits'] / r['empty']['exec_error_bits']:.0%}")
        graph_path.write_text(json.dumps(graph, indent=1))
        print(name, graph["summary"], flush=True)


if __name__ == "__main__":
    main()
