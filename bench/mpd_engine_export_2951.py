"""Export a modular-addition run file to raw float64 tensors for the Rust decomposition (#2951).

The run file is what ``bench/mpd_modadd_2951.py train`` writes (``torch.save`` of config, curves and
checkpoints). ``gam_mpd::import`` reads each tensor as raw little-endian float64 in C order
(``<name>.f64``) with its shape from ``export.json``. Each per-head
tensor (``W_Q``, ``W_K``, ``W_V``: heads x d_head x d_model) is written head-major as (heads * d_head) x
d_model, the layout of torch's head concatenation; a vector is one row. Every tensor is widened from its
trained float32 to float64, which is exact. ``export.json`` records the config, the checkpoint step, and
the sha256 of the run file and of every exported array.

    python mpd_engine_export_2951.py RUN.pt OUT_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("out")
    args = parser.parse_args()
    run = torch.load(args.run, map_location="cpu", weights_only=True)
    step = max(run["checkpoints"])
    state = run["checkpoints"][step]
    os.makedirs(args.out, exist_ok=True)
    files = {}
    for name, tensor in state.items():
        if tensor.dtype == torch.bool:
            continue
        array = tensor.double().numpy()
        if array.ndim == 3:
            array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
        if array.ndim == 1:
            array = array.reshape(1, -1)
        path = os.path.join(args.out, f"{name}.f64")
        np.ascontiguousarray(array, dtype="<f8").tofile(path)
        files[name] = {"shape": list(array.shape), "sha256": sha256(path)}
    config = dict(run["config"])
    record = {"run": os.path.abspath(args.run), "run_sha256": sha256(args.run), "step": int(step), "config": config,
              "files": files}
    with open(os.path.join(args.out, "export.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps({"out": args.out, "step": int(step), "p": config["p"], "files": len(files)}))


if __name__ == "__main__":
    main()
