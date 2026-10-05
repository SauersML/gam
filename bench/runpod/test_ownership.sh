#!/usr/bin/env bash
# rp-run never deletes a pod it did not create (#2951): the team account is shared, and a pod is
# the runner's only when it is named mpd2951-... and its id is in the runner's registry. Runs rp-run
# against a mock RunPod API (a local Python server holding three running pods: one of ours, one of
# another user, one named like ours but never created by the runner) in a scratch data directory,
# through every path that deletes: a run's stale record reaped at the next start, rp-run stop on a
# record naming each pod, and status. Passes when the only pod the mock saw deleted is ours.
#
#   bash bench/runpod/test_ownership.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(mktemp -d)
trap 'kill "${server:-0}" 2> /dev/null; rm -rf "$root"' EXIT
mkdir -p "$root/runpod/active"

cat > "$root/mock.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
pods = {
    "ours1": {"id": "ours1", "name": "mpd2951-mine", "desiredStatus": "RUNNING", "costPerHr": 0.3, "createdAt": "2026-10-05 12:00:00.000 +0000 UTC"},
    "theirs1": {"id": "theirs1", "name": "someone-else", "desiredStatus": "RUNNING", "costPerHr": 0.5, "createdAt": "2026-10-05 12:00:00.000 +0000 UTC"},
    "theirs2": {"id": "theirs2", "name": "mpd2951-lookalike", "desiredStatus": "RUNNING", "costPerHr": 0.5, "createdAt": "2026-10-05 12:00:00.000 +0000 UTC"},
}
log = open(sys.argv[2], "a")
class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/pods":
            return self.reply(200, list(pods.values()))
        pid = path.rsplit("/", 1)[-1]
        return self.reply(200, pods[pid]) if pid in pods else self.reply(404, {"error": "pod not found"})
    def do_DELETE(self):
        pid = self.path.rsplit("/", 1)[-1]
        log.write(pid + "\n"); log.flush()
        pods.pop(pid, None)
        self.send_response(204); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.reply(200, {"data": {"myself": {"clientBalance": 740, "currentSpendPerHr": 1.3},
                                  "gpuTypes": [{"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24, "securePrice": 0.74, "communityPrice": 0.34}]}})
HTTPServer(("127.0.0.1", int(sys.argv[1])), Mock).serve_forever()
PY
port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
python3 "$root/mock.py" "$port" "$root/deletes.log" &
server=$!
until curl -s "http://127.0.0.1:$port/v1/pods" > /dev/null; do sleep 0.1; done
export RP_TEST_ROOT=$root RP_TEST_API=http://127.0.0.1:$port
rp() { "$here/rp-run" "$@" > "$root/out.log" 2>&1 || true; }

# Registry: only ours1. Records for all three pods, each held by a process that is gone.
printf 'ours1\tmine\t2026-10-05 12:00:00\n' > "$root/runpod/created.tsv"
printf '# test\nprobe\n' > "$root/runpod/APPROVED.txt"
records() {
    local p
    for p in "ours1 mine" "theirs1 x1" "theirs2 x2"; do
        set -- $p
        echo "$2 NVIDIA_GeForce_RTX_4090 COMMUNITY 0.3 1791200000 999999 1.0 run" > "$root/runpod/active/$1"
    done
}
records
# 1. A run's start reaps stale records: only ours1 may be deleted.
RP_DRY=1 RP_PARALLEL=1 rp probe "RTX 4090" 0.1 -- true
# 2. rp-run stop on records naming the other pods.
records
rp stop x1
rp stop x2
# 3. status lists every pod and acts on none.
rp status
grep -q "theirs1 .*not ours" "$root/out.log" && grep -q "theirs2 .*not ours" "$root/out.log" ||
    { echo "FAIL: status does not mark the other pods as not ours"; cat "$root/out.log"; exit 1; }

deleted=$(sort -u "$root/deletes.log" 2> /dev/null | tr '\n' ' ')
if [ "$deleted" = "ours1 " ]; then
    echo "PASS: only ours1 was deleted; theirs1 (another user's) and theirs2 (named mpd2951- but not created by rp-run) were never touched"
else
    echo "FAIL: deleted: ${deleted:-nothing}"
    exit 1
fi
