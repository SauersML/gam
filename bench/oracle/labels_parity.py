"""Parity of the label table (labels.py) against oracle.rs (#2951): a handful of (neuron, context) cases of
a label run measured again by the investigator's server (crates/gam-mpd/src/oracle.rs, the host float64
execution of the native program) with the same edit, W_down[:, i] <- alpha W_down[:, i] (oracle's
"neuron" component, P = the down column, W(alpha) = W + (alpha - 1) P), at the same positions.

Compared per case and position: KL(clean || edited) in nats, and the actual next token's log-probability
change; reported as the largest absolute and relative differences. One JSON object goes to stdout.

  labels_parity.py --labels DIR --server ADDRESS --model NAME [--cases 8] [--seed 0]
The server must hold the labels' model under NAME (mpd_oracle_2951 ADDRESS WORK_GIB NAME=CHECKPOINT).
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def request(address: str, payload: dict) -> dict:
    host, sep, port = address.rpartition(":")
    if sep and port.isdigit() and "/" not in address:
        s = socket.create_connection((host, int(port)))
    else:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(address)
    with s:
        s.sendall((json.dumps(payload) + "\n").encode())
        chunks = []
        while not chunks or not chunks[-1].endswith(b"\n"):
            b = s.recv(1 << 20)
            if not b:
                break
            chunks.append(b)
    reply = json.loads(b"".join(chunks))
    if "error" in reply:
        raise SystemExit(f"server: {reply['error']}")
    return reply["ok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.labels)
    tokens = load_file(str(root / "tokens.safetensors"))["tokens"]
    shards = sorted(root.glob("shard_*.json"))
    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.cases):
        meta_path = shards[rng.integers(len(shards))]
        meta = json.loads(meta_path.read_text())
        data = load_file(str(meta_path.with_suffix(".safetensors")))
        n = int(rng.integers(len(meta["neurons"])))
        c = int(rng.integers(len(meta["contexts"])))
        layer, index = meta["neurons"][n]
        positions = [int(p) for p in data["positions"][n, c]]
        sequence = [int(t) for t in tokens[c]]
        for edit in meta["edits"]:
            reply = request(args.server, {
                "op": "run", "model": args.model, "sequences": [sequence], "positions": positions, "top": 1, "clean": True, "next": True,
                "intervention": {"edits": [{"component": {"kind": "neuron", "layer": layer, "index": index}, "alpha": edit["alpha"]}]},
            })
            for k, entry in enumerate(reply["runs"][0]["positions"]):
                table_kl = float(data[f"kl_{edit['name']}"][n, c, k])
                table_next = float(data[f"next_{edit['name']}"][n, c, k])
                row = {"layer": layer, "index": index, "context": c, "position": positions[k], "edit": edit["name"],
                       "kl_table": table_kl, "kl_reference": entry["kl_from_clean_nats"]}
                if "next" in entry and len(entry["next"]) == 3:
                    row["next_table"] = table_next
                    row["next_reference"] = entry["next"][1] - entry["next"][2]
                rows.append(row)
    kl_abs = [abs(r["kl_table"] - r["kl_reference"]) for r in rows]
    kl_rel = [abs(r["kl_table"] - r["kl_reference"]) / r["kl_reference"] for r in rows if r["kl_reference"] > 0]
    nx = [abs(r["next_table"] - r["next_reference"]) for r in rows if "next_reference" in r]
    print(json.dumps({"cases": args.cases, "positions": len(rows),
                      "kl_max_abs_difference_nats": max(kl_abs), "kl_max_relative_difference": max(kl_rel) if kl_rel else None,
                      "kl_reference_range_nats": [min(r["kl_reference"] for r in rows), max(r["kl_reference"] for r in rows)],
                      "next_max_abs_difference_nats": max(nx) if nx else None, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
