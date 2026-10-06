"""Public top-activating examples of circuit-tracer transcoder features (#2951), the oracle's optional
text input for a library warm-started from those transcoders (mwhanna/qwen3-0.6b-transcoders-lowl0's
features/ directory). They come with the public files, measured on the transcoder authors' data, so they
carry no label of ours.

Files: index.json.gz maps each layer to its file layer_{l}.bin and the byte offsets of its features'
records; a record is a 4-byte little-endian length, then gzip'd JSON with examples_quantiles (each
example: 32 tokens, the feature's activation on each, train_token_ind the example's peak), act_max,
top_logits and bottom_logits (the transcoder's decoder row read through the unembedding).

Feature order. A library made by library_transcoder.rs keeps a layer's features in its kept file's order
(`features`, the transcoder indices, increasing); `kept(OUT)` reads them from a run's
transcoder_l{l}.safetensors, so function c of layer l is transcoder feature kept[l][c] while the library
has not removed any.

A subset (`extract`): the records of the features a label table asks about, as one gzip'd JSON
{layer: {feature: record}}, which Features also reads (pods need not download the 36 GB directory).

  transcoder_examples.py show --features DIR --layer L --feature F [--n 3]
  transcoder_examples.py extract --features DIR --transcoders OUT --labels TABLE... --out SUBSET.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

LEVELS = 10


class Features:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.subset = json.loads(gzip.decompress(self.root.read_bytes())) if self.root.is_file() else None
        self.index = None if self.subset is not None else json.loads(gzip.decompress((self.root / "index.json.gz").read_bytes()))
        self.files: dict[int, object] = {}

    def record(self, layer: int, feature: int) -> dict:
        if self.subset is not None:
            return self.subset[str(layer)][str(feature)]
        entry = self.index[str(layer)]
        if layer not in self.files:
            self.files[layer] = open(self.root / entry["filename"], "rb")
        f = self.files[layer]
        start, end = entry["offsets"][feature], entry["offsets"][feature + 1]
        f.seek(start)
        raw = f.read(end - start)
        (length,) = struct.unpack("<I", raw[:4])
        assert length == len(raw) - 4, (layer, feature, length, len(raw))
        return json.loads(gzip.decompress(raw[4:]))

    def text(self, layer: int, feature: int, n: int = 3, width: int = 12) -> str:
        """Its n strongest distinct top examples, each the `width` tokens up to its peak with the peak
        token marked and its level (0-9 of act_max), then its top and bottom logits."""
        r = self.record(layer, feature)
        top = next((q["examples"] for q in r["examples_quantiles"] if q["quantile_name"] == "Top"), [])
        peak = float(r.get("act_max") or 0.0) or max((max(e["tokens_acts_list"]) for e in top), default=0.0)
        lines, seen = [], set()
        for e in top:
            p = int(e["train_token_ind"])
            toks = [t.replace("⏎", "\n") for t in e["tokens"]]
            text = "".join(toks[max(0, p - width) : p]) + "⟦" + toks[p] + "⟧"
            if text in seen:
                continue
            seen.add(text)
            level = 0 if peak <= 0 else min(LEVELS - 1, int(LEVELS * float(e["tokens_acts_list"][p]) / peak))
            lines.append(f"- {text!r} (level {level})")
            if len(lines) == n:
                break
        logits = lambda key: ", ".join(json.dumps(t) for t in r.get(key, [])[:8])  # noqa: E731
        return ("Public examples where it is most active (the transcoder's data):\n" + "\n".join(lines)
                + f"\nThe transcoder's write raises {logits('top_logits')} and lowers {logits('bottom_logits')}.")


def kept(out: Path) -> dict[int, list[int]]:
    """Per layer, the transcoder indices a run's kept files hold (library_transcoder.rs's `features`)."""
    from safetensors.torch import load_file

    result = {}
    for path in sorted(Path(out).glob("transcoder_l*.safetensors")):
        layer = int(path.stem.removeprefix("transcoder_l"))
        result[layer] = [int(x) for x in load_file(str(path))["features"].reshape(-1).tolist()]
    return result


def extract(args):
    """The records of every part the label tables hold (site_{l}_function.json's subcomponents: functions
    0..C-1 of layer l, transcoder features kept[l][0..C-1])."""
    features, k = Features(Path(args.features)), kept(Path(args.transcoders))
    out: dict[str, dict] = {}
    for table in args.labels:
        for meta_path in sorted(Path(table).glob("site_*_function.json")):
            meta = json.loads(meta_path.read_text())
            layer = meta["layer"]
            for c in range(meta["subcomponents"]):
                f = k[layer][c]
                out.setdefault(str(layer), {})[str(f)] = features.record(layer, f)
    Path(args.out).write_bytes(gzip.compress(json.dumps(out).encode()))
    print(json.dumps({"layers": len(out), "features": sum(len(v) for v in out.values()), "bytes": Path(args.out).stat().st_size}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("show")
    s.add_argument("--features", required=True)
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--feature", type=int, required=True)
    s.add_argument("--n", type=int, default=3)
    e = sub.add_parser("extract")
    e.add_argument("--features", required=True)
    e.add_argument("--transcoders", required=True)
    e.add_argument("--labels", nargs="+", required=True)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        print(Features(Path(args.features)).text(args.layer, args.feature, args.n))


if __name__ == "__main__":
    main()
