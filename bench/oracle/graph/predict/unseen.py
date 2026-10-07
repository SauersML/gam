"""Questions about pieces a trained oracle never saw (#2951): the heads and neurons named anywhere in its
training questions form the seen set; a held-out shard is filtered to the questions whose every named head
and neuron is outside it (questions naming whole blocks or no piece are dropped).

  unseen.py build --train TRAIN.jsonl ... --out SEEN.json
  unseen.py filter --seen SEEN.json --shard SHARD.jsonl --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

HEAD = re.compile(r"L\[(\d+)\]\.head\[(\d+)\]")
NEURONS = re.compile(r"L\[(\d+)\]\.mlp\[([\d, ]+)\]")
BLOCK = re.compile(r"L\[\d+\]\.(?:mlp|head)\[:\]|L\[\d+\]\.(?:mlp|attn)(?![\w\[])")


def pieces(text: str):
    heads = {(int(l), int(h)) for l, h in HEAD.findall(text)}
    neurons = {(int(l), int(i)) for l, ids in NEURONS.findall(text) for i in ids.split(",")}
    return heads, neurons, bool(BLOCK.search(text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train", nargs="+", required=True)
    b.add_argument("--out", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--seen", required=True)
    f.add_argument("--shard", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        heads, neurons = set(), set()
        for path in args.train:
            for line in open(path):
                h, n, _ = pieces(json.loads(line)["input"])
                heads |= h
                neurons |= n
        json.dump({"heads": sorted(heads), "neurons": sorted(neurons)}, open(args.out, "w"))
        print(json.dumps({"heads": len(heads), "neurons": len(neurons)}))
        return
    seen = json.load(open(args.seen))
    sh, sn = {tuple(x) for x in seen["heads"]}, {tuple(x) for x in seen["neurons"]}
    kept = Counter()
    with open(args.out, "w") as out:
        for line in open(args.shard):
            q = json.loads(line)
            h, n, block = pieces(q["input"])
            if (h or n) and not block and not (h & sh) and not (n & sn):
                out.write(line)
                kept[q["type"]] += 1
    print(json.dumps({"kept": dict(kept)}))


if __name__ == "__main__":
    main()
