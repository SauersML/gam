#!/bin/zsh
# The #2951 toy gate, end to end: a regression check every change to the fitting method must pass,
# on toys whose mechanisms are known (never results).
#
# usage: bench/toys_2951/gate.sh ROOT BINARY [EPOCHS]
#   ROOT    where the toys live (trained here when missing), e.g. ~/mpd-data/toys_gate
#   BINARY  mpd_library_mdl_2951 built at the commit under test (fastcheck build --release)
#
# 1. trains each toy that ROOT lacks (train_toys.py);
# 2. scores the references, native start and truth, on every toy (score_toys.py --references);
# 3. for the language-model toys (induction, modadd_113): writes the per-head / per-neuron start
#    (toy_start.py), runs the engine's edits on the native start (gaps must be 0) and fits the
#    start with library_vpd (mpd_library_mdl_2951 with `vpd`), then the edits on its checkpoint;
# 4. prints the table (table_toys.py).
set -e
root=$1; binary=$2; epochs=${3:-30}
here=${0:A:h}
py=(env MPD_MEM_GIB=1 ~/mpd-data/venv/bin/python)
for toy in tms_40_10 tms_40_10_id resid_mlp_1l resid_mlp_2l resid_mlp_3l modadd_113 induction; do
  [[ -f $root/$toy/truth.json ]] || mem-lease 3 ~/mpd-data/venv/bin/python $here/train_toys.py $toy $root
done
for toy in tms_40_10 tms_40_10_id resid_mlp_1l resid_mlp_2l; do
  $py $here/score_toys.py $root/$toy --references $root/refs/$toy > /dev/null
done
env MPD_MEM_GIB=3 ~/mpd-data/venv/bin/python $here/score_toys.py $root/resid_mlp_3l --references $root/refs/resid_mlp_3l > /dev/null
for toy in induction modadd_113; do
  $py $here/score_toys.py $root/$toy --references $root/refs/$toy > /dev/null
  [[ -f $root/start/$toy/start.json ]] || $py $here/toy_start.py $root/$toy $root/start/$toy
  $py $here/score_toys.py $root/$toy $root/start/$toy/parts > /dev/null
  $py $here/engine_edits.py $binary $root/$toy $root/engine/${toy}_native > /dev/null
  # the fit: held-out sequences are the truth's, the training sequences the next ones
  settings=$root/fit/${toy}_vpd.json
  mkdir -p $root/fit
  $py - $root/$toy $root/start/$toy $settings $epochs <<'EOF'
import hashlib, json, sys
from pathlib import Path
toy, start, out, epochs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
record = json.loads((toy / "export.json").read_text())
rows, context = record["files"]["tokens"]["shape"]
held = json.loads((toy / "truth.json").read_text())["active"]["shape"][0] // context
out.write_text(json.dumps({
    "export_sha256": hashlib.sha256((toy / "export.json").read_bytes()).hexdigest(),
    "training_sequences": rows - held, "context": context, "held_out": [0, held],
    "vpd": {"decomposition": str(start / "decomposition"), "start": str(start / "start.json"), "arm": "heads_neurons"},
    "fit": {"batch_sequences": max(1, 32768 // context // 8), "seed": 0, "numeric_bytes": 1 << 30, "head_tile_rows": 4096, "epochs": epochs,
            "families": ["swap", "zero", "scale", "push"]}}))
EOF
  $binary $root/$toy $settings $root/fit/out_${toy} host > $root/fit/fit_${toy}.log 2>&1
  $py $here/engine_edits.py $binary $root/$toy $root/fit/out_${toy} $settings checkpoint.bin > /dev/null
done
$py $here/table_toys.py $root/refs/*/*/SCORE_*.json $root/start/*/parts/SCORE_*.json
for f in $root/engine/*/SCORE_*_edits.json $root/fit/out_*/SCORE_*_edits.json; do
  $py -c "import json,sys; r=json.load(open(sys.argv[1])); f=r['edits']['families']; print(r['toy'], r['explanation'], ' '.join(f'{k}={v[\"mean_bits_per_token\"]:.4g}/{v[\"effect_mean_bits_per_token\"]:.4g}' for k, v in f.items()))" $f
done
