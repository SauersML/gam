"""The Hugging Face forward of a causal language model on token windows, as a parity reference (#2951).

Runs ``AutoModelForCausalLM`` (eager attention, CPU, forward only) in ``--dtype`` (float32 or
float64; the stored weights widen exactly) on the first ``--sequences`` rows of a window file (rows of
``--context`` little-endian u32 tokens, ``mpd_qwen3_fineweb_2951.py``), each row its own sequence
from position 0, and writes the logits as raw little-endian float64, ``(sequences * context) x
vocab`` in row order, to ``--out``, with ``--out``.json recording the model, the dtype and the
files' sha256.

usage: mpd_qwen3_reference_2951.py MODEL WINDOWS --context T --sequences S --dtype D --out PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("windows")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
    rows = np.fromfile(args.windows, dtype="<u4", count=args.sequences * args.context).reshape(args.sequences, args.context)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager")
    model.eval()
    with open(args.out, "wb") as fh, torch.no_grad():
        for row in rows:
            ids = torch.from_numpy(row.astype(np.int64))[None]
            logits = model(ids).logits[0].double().numpy()
            np.ascontiguousarray(logits, dtype="<f8").tofile(fh)
    record = {
        "model": args.model,
        "dtype": args.dtype,
        "attention": "eager",
        "windows": args.windows,
        "windows_sha256_prefix": hashlib.sha256(rows.astype("<u4").tobytes()).hexdigest(),
        "sequences": args.sequences,
        "context": args.context,
        "vocab": int(model.config.vocab_size),
        "logits_sha256": sha256(args.out),
        "torch": torch.__version__,
    }
    with open(args.out + ".json", "w") as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
