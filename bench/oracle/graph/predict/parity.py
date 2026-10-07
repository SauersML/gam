"""Parity of the prediction data (generate.py) against oracle.rs (#2951): a sample of a shard's questions
measured again by the measured-intervention server (crates/gam-mpd/src/oracle.rs, the host float64
execution of the native program, examples/mpd_oracle_2951.rs) with the same intervention:
  plain, prompt   a run of the (edited) text: the top-10 probabilities;
  edit, rank      native edits W(a) = W + (a - 1) P: a head's output map (Component::Head), neurons' down
                  columns (Component::Neuron), a whole MLP (the down_proj operator), a whole attention
                  (every head); the top-10 probabilities and KL(M || M_e);
  swap            patches at the last position from the source text (Site::Head, Site::Neurons with
                  the neurons' coordinates);
  continue        Request::Generate with the edits: the greedy tokens;
  where           neuron probes: the activations at the reported positions (Site::Neurons, full values).
Cut questions have no oracle.rs counterpart (path patching to one reader) and are not checked here.
Reported per type: the largest absolute differences (probabilities, KL in bits, activations) and the
fraction of identical greedy tokens. One JSON object goes to stdout.

  parity.py --shard SHARD.jsonl --windows WINDOWS_T128.u32 --server HOST:PORT --model NAME [--per-type 8] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
from pathlib import Path

import numpy as np

HEADS = {"qwen3-0.6b": 16, "qwen3-8b": 32, "vpd4l": 6}
H = [16]  # the shard's model's heads per layer (set in main)


def request(address: str, payload: dict) -> dict:
    host, _, port = address.rpartition(":")
    with socket.create_connection((host, int(port))) as s:
        s.sendall((json.dumps(payload) + "\n").encode())
        chunks = []
        while not chunks or not chunks[-1].endswith(b"\n"):
            b = s.recv(1 << 20)
            if not b:
                break
            chunks.append(b)
    reply = json.loads(b"".join(chunks))
    if "error" in reply:
        raise SystemExit(f"server: {reply['error']} for {json.dumps(payload)[:400]}")
    return reply["ok"]


def parse_piece(text: str):
    m = re.fullmatch(r"L\[(\d+)\]\.(?:mlp|attn)", text)  # the first shards' spelling of whole blocks
    if m:
        return ("mlp" if text.endswith("mlp") else "attn", int(m[1]))
    m = re.fullmatch(r"L\[(\d+)\]\.head\[(\d+)\]", text)
    if m:
        return ("head", int(m[1]), int(m[2]))
    m = re.fullmatch(r"L\[(\d+)\]\.mlp\[([\d, ]+)\]", text)
    if m:
        return ("neurons", int(m[1]), [int(x) for x in m[2].split(",")])
    m = re.fullmatch(r"PD\.vpd\[(\d+)\]\.(\w+)\[([\d, ]+)\]", text)
    if m:
        return ("vpd", int(m[1]), m[2], [int(x) for x in m[3].split(",")])
    m = re.fullmatch(r"PD\.tc\[(\d+)\]\[([\d, ]+)\]", text)
    if m:
        return ("tc", int(m[1]), [int(x) for x in m[2].split(",")])
    m = re.fullmatch(r"L\[(\d+)\]\.(mlp|head)\[:\]", text)
    if m:
        return ("mlp" if m[2] == "mlp" else "attn", int(m[1]))
    raise ValueError(text)


REGISTERED = []  # non-empty once VPD's subcomponents are registered with the server


class Unchecked(Exception):
    """A piece with no oracle.rs counterpart for this intervention (transcoder features; VPD activity swaps)."""


def edits(piece, alpha):
    kind = piece[0]
    if kind == "vpd" and not REGISTERED:
        raise Unchecked(kind)
    if kind == "vpd":  # registered by register_vpd as "h.{l}.{attn|mlp}.{site}:{c}"
        block = "mlp" if piece[2] in ("c_fc", "down_proj") else "attn"
        return [{"component": {"kind": "registered", "id": f"h.{piece[1]}.{block}.{piece[2]}:{c}"}, "alpha": alpha} for c in piece[3]]
    if kind == "tc":
        raise Unchecked(kind)
    if kind == "head":
        return [{"component": {"kind": "head", "layer": piece[1], "head": piece[2]}, "alpha": alpha}]
    if kind == "neurons":
        return [{"component": {"kind": "neuron", "layer": piece[1], "index": i}, "alpha": alpha} for i in piece[2]]
    if kind == "mlp":
        return [{"component": {"kind": "operator", "name": f"blocks.{piece[1]}.down_proj"}, "alpha": alpha}]
    return [{"component": {"kind": "head", "layer": piece[1], "head": h}, "alpha": alpha} for h in range(H[0])]


def swap_patches(piece, source):
    value = {"kind": "source", "tokens": source, "positions": [-1]}
    kind = piece[0]
    if kind in ("vpd", "tc"):
        raise Unchecked(kind)
    if kind == "head":
        return [{"site": {"kind": "head", "layer": piece[1], "head": piece[2]}, "positions": [-1], "value": value}]
    if kind == "neurons":
        return [{"site": {"kind": "neurons", "layer": piece[1]}, "positions": [-1], "coordinates": piece[2], "value": value}]
    if kind == "mlp":
        return [{"site": {"kind": "neurons", "layer": piece[1]}, "positions": [-1], "value": value}]
    return [{"site": {"kind": "head", "layer": piece[1], "head": h}, "positions": [-1], "value": value} for h in range(H[0])]


def tokens_of(windows, text_id: str):
    """The text of a corpus id "fineweb:ROW:T" / "pile:ROW:T": row ROW of the token file, first T tokens."""
    _, row, T = text_id.split(":")
    return [int(t) for t in windows[int(row), : int(T)]]


def prob_diff(mine: dict, top: list) -> float:
    ref = {int(t): math.exp(v) for t, v in top}
    return max(abs(p - ref[t]) for t, p in zip(mine["ids"], mine["p"]) if t in ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--register-vpd", default="", help="vpd4l: VPD's decomposition export, registered so PD.vpd edits are checked")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    windows = np.load(args.windows, mmap_mode="r") if args.windows.endswith(".npy") else np.memmap(args.windows, dtype="<u4", mode="r").reshape(-1, 128)
    if args.register_vpd:
        print(json.dumps({"registered": request(args.server, {"op": "register_vpd", "model": args.model, "decomposition": args.register_vpd})}), flush=True)
        REGISTERED.append(True)
    by_type = {}
    for line in open(args.shard):
        q = json.loads(line)
        H[0] = HEADS[q["model"]]
        if q["source"] in ("fineweb", "pile"):
            by_type.setdefault(q["type"], []).append(q)
    rng = np.random.default_rng(args.seed)
    report = {}
    for kind, qs in sorted(by_type.items()):
        if kind == "cut":
            continue
        picks = [qs[i] for i in rng.choice(len(qs), size=min(args.per_type, len(qs)), replace=False)]
        dp, dkl, dact, same, total, kls = [], [], [], 0, 0, []
        unchecked = 0
        for q in picks:
          try:
            toks = tokens_of(windows, q["text_id"])
            n = q["numbers"]
            run = lambda seq, iv: request(args.server, {"op": "run", "model": args.model, "sequences": [seq], "top": 10, "clean": True, "intervention": iv})["runs"][0]["positions"][0]  # noqa: E731
            if kind == "plain":
                dp.append(prob_diff(n["edited"], run(toks, {})["top"]))
            elif kind == "prompt":
                toks2 = list(toks)
                toks2[n["position"]] = n["new"]
                dp.append(prob_diff(n["edited"], run(toks2, {})["top"]))
            elif kind == "edit":
                e = run(toks, {"edits": edits(parse_piece(n["piece"]), n["alpha"])})
                dp.append(prob_diff(n["edited"], e["top"]))
                dkl.append(abs(n["kl_bits"] - e["kl_from_clean_nats"] / math.log(2)))
                kls.append(e["kl_from_clean_nats"] / math.log(2))
            elif kind == "swap":
                src = tokens_of(windows, n["source_text_id"])
                e = run(toks, {"patches": swap_patches(parse_piece(n["piece"]), src)})
                dp.append(prob_diff(n["edited"], e["top"]))
                dkl.append(abs(n["kl_bits"] - e["kl_from_clean_nats"] / math.log(2)))
                kls.append(e["kl_from_clean_nats"] / math.log(2))
            elif kind == "rank":
                for piece, kl in zip(n["pieces"], n["kl_bits"]):
                    e = run(toks, {"edits": edits(parse_piece(piece), 0.0)})
                    dkl.append(abs(kl - e["kl_from_clean_nats"] / math.log(2)))
                    kls.append(e["kl_from_clean_nats"] / math.log(2))
            elif kind == "continue":
                g = request(args.server, {"op": "generate", "model": args.model, "tokens": toks, "steps": len(n["edited_ids"]),
                                          "edits": edits(parse_piece(n["piece"]), n["alpha"])})["generated"]
                ref = [int(x["token"]) for x in g]
                same += sum(1 for a, b in zip(ref, n["edited_ids"]) if a == b)
                total += len(ref)
            elif kind == "where":
                piece = parse_piece(n["piece"])
                if piece[0] != "neurons":
                    continue
                l, i = piece[1], piece[2][0]
                reply = request(args.server, {"op": "run", "model": args.model, "sequences": [toks], "positions": n["positions"], "top": 1,
                                              "record": [{"kind": "neurons", "layer": l}], "full": True})["runs"][0]["positions"]
                for entry, level in zip(reply, n["levels"]):
                    dact.append(abs(entry["record"][0]["value"][i] - level))
          except Unchecked:
            unchecked += 1
        row = {"checked": len(picks) - unchecked, "unchecked": unchecked}
        if dp:
            row["max_abs_probability_difference"] = max(dp)
        if dkl:
            row["max_abs_kl_difference_bits"] = max(dkl)
            row["kl_reference_range_bits"] = [min(kls), max(kls)]
        if dact:
            row["max_abs_activation_difference"] = max(dact)
        if total:
            row["greedy_tokens_identical"] = same / total
        report[kind] = row
        print(json.dumps({kind: row}), flush=True)
    print(json.dumps({"parity": report}, indent=1))


if __name__ == "__main__":
    main()
