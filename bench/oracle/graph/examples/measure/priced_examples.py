"""Regenerates the generated vpd4l examples (index entries without "priority") with priced nodes: heads
above 0.28 bits per token, whole MLPs above 3.375, MLPs with a neuron table (vpd_neurons.py) as their
paying neurons; tables are copied to ~/mpd-data/graph_oracle/experiments/examples/neurons_vpd4l/.

  MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python priced_examples.py NEURONS_DIR
"""
import json
import subprocess
import sys
from pathlib import Path

G = Path.home() / "gam/bench/oracle/graph"
sys.path.insert(0, str(G))
import mech  # noqa: E402

E = Path.home() / "mpd-data/graph_oracle/experiments/examples"
B = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l"
neurons_dir = Path(sys.argv[1])
index = json.loads((G / "examples/index.json").read_text())
made = []
for name, entry in index.items():
    if entry["model"] != "vpd4l" or entry.get("priority") is not None or not entry["behavior"]:
        continue
    bid = entry["behavior"]
    patch = E / "patch_vpd4l" / f"patch_{bid}.json"
    tables = sorted(neurons_dir.glob(f"neurons_{bid}_l*.json"))
    if not patch.exists() or not tables:
        continue
    for t in tables:  # keep a copy beside the patch tables
        (E / "neurons_vpd4l").mkdir(exist_ok=True)
        (E / "neurons_vpd4l" / t.name).write_text(t.read_text())
    args = [sys.executable, str(G / "examples/measure/program_from_patch.py"), str(patch), str(B / f"{bid}.json"),
            "--head-price", "0.28", "--mlp-price", "3.375"]
    for t in tables:
        args += ["--neurons", str(t)]
    src = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    ir = mech.trace_inline(src, "vpd4l")
    if ir["valid"] and ir["nodes"]:
        (G / "examples" / f"{name}.py").write_text(src)
        made.append((name, len(ir["nodes"]), len(ir["edges"]), ir["python_tokens"]))
print(len(made), made)
