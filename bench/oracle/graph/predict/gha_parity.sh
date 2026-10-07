#!/usr/bin/env bash
# Parity of generate.py against oracle.rs on a GitHub Actions runner (#2951; public inputs only):
#   GHA_EXAMPLES=gam-mpd:mpd_oracle_2951 bench/runpod/gha-run predict-parity -- 'bash bench/oracle/graph/predict/gha_parity.sh'
# Texts: the repository's own README and LICENSE files tokenized by Qwen3's tokenizer into rows of 128
# tokens (a windows file like the FineWeb release's). A small shard is generated on the CPU (float32),
# then every sampled question is measured again by mpd_oracle_2951 (float64 host execution);
# the report is $OUT/parity.json.
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# The runner's system pip (22.0) fails to resolve torch's index ("assert len(weights) == expected_node_count").
python3 -m venv "$RUNNER_TEMP/predict-venv"
export PATH="$RUNNER_TEMP/predict-venv/bin:$PATH"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -q transformers safetensors huggingface_hub numpy
SNAP=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))")
python3 - "$SNAP" "$OUT/windows_T128.u32" <<'PY'
import glob, sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
text = "\n".join(open(p, errors="ignore").read() for p in sorted(glob.glob("**/README*", recursive=True) + glob.glob("LICENSE*"))[:50])
ids = np.array(tok(text)["input_ids"], dtype="<u4")
rows = len(ids) // 128
ids[: rows * 128].tofile(sys.argv[2])
print("windows", rows)
PY
python3 "$here/generate.py" --model "$SNAP" --windows "$OUT/windows_T128.u32" --out "$OUT/parity_shard.jsonl" --texts 32 --batch 16 --lengths 24,32 --split heldout --seed 7
"$MPD_BIN/mpd_oracle_2951" 127.0.0.1:47123 6 "qwen=$SNAP" > "$OUT/server.log" 2>&1 &
server=$!
python3 -c "
import socket, sys, time
for _ in range(900):
    try:
        socket.create_connection(('127.0.0.1', 47123)).close(); sys.exit(0)
    except OSError:
        time.sleep(2)
sys.exit('server did not start')"
python3 "$here/parity.py" --shard "$OUT/parity_shard.jsonl" --windows "$OUT/windows_T128.u32" --server 127.0.0.1:47123 --model qwen --per-type 6 | tee "$OUT/parity.json"
kill $server
