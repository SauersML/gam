#!/usr/bin/env bash
# The reader term of e2e/run.py's programs on MATS (#2951): one reader_score.py server (Qwen3-8B on vLLM,
# g-rl's ~/rl-venv, built here as rl/mats_rl.sh builds it when missing) scores every DIR/<stem>.program.jsonl written by
# `run.py --items DIR` on its own DIR/<stem>.items.jsonl and writes OUT/<stem>.reader.json (the reader
# term with the empty-program and code-alone baselines).
#   MATS_GPUS=1 MATS_QOS=debug mats-run g-int-reader 8 48 1 -- bash /Users/user/gam/bench/oracle/graph/e2e/mats_reader.sh \
#       /Users/user/mpd-data/graph_oracle/experiments/e2e /Users/user/mpd-data/cluster/g-int-reader [vpd4l]
set -Eeuo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=$1 OUT=$2 TARGET=${3:-vpd4l}
py=$HOME/rl-venv/bin/python
(   # the venv recipe and lock of rl/mats_rl.sh
    flock 9
    if ! "$py" -c "import vllm, peft" 2> /dev/null; then
        export UV_CACHE_DIR=$HOME/.cache/uv
        rm -rf "$HOME/rl-venv"
        ~/.local/bin/uv venv -q --python 3.12 "$HOME/rl-venv"
        ~/.local/bin/uv pip install -q --python "$py" "vllm==0.10.2" "transformers>=4.56,<5" "peft==0.21.2" "accelerate==1.15.0" numpy
    fi
) 9> "$HOME/.rl-venv.lock"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
PORT=$((40000 + ${SLURM_JOB_ID:-1} % 20000))
"$py" "$here/../reader_score.py" serve --backend vllm --model Qwen/Qwen3-8B --target "$TARGET" --listen "127.0.0.1:$PORT" > "$OUT/server.log" 2>&1 &
server=$!
trap 'kill $server 2> /dev/null' EXIT
"$py" - "$DIR" "$OUT" "$PORT" <<'EOF'
import json, socket, sys, time
from pathlib import Path
d, out, port = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
for _ in range(900):
    try:
        socket.create_connection(("127.0.0.1", port)).close()
        break
    except OSError:
        time.sleep(2)
else:
    sys.exit("the reader server did not start (server.log)")
for program in sorted(d.glob("*.program.jsonl")):
    stem = program.name[: -len(".program.jsonl")]
    programs = [json.loads(l) for l in program.read_text().splitlines() if l.strip()]
    items = [json.loads(l) for l in (d / f"{stem}.items.jsonl").read_text().splitlines() if l.strip()]
    with socket.create_connection(("127.0.0.1", port)) as s:
        s.sendall((json.dumps({"op": "score", "programs": programs, "items": items, "N": 2**24}) + "\n").encode())
        reply = json.loads(s.makefile().readline())
    (out / f"{stem}.reader.json").write_text(json.dumps(reply, indent=1))
    r = reply.get("ok", {}).get("results", [{}])[0]
    print(stem, reply.get("error"), {k: r.get(k) for k in ("reader_error_bits", "mean_bits_per_item", "empty_mean_bits_per_item", "code_only_mean_bits_per_item")}, flush=True)
EOF
