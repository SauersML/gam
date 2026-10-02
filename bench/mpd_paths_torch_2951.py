"""#2951 paths: torch's float64 logits of a modular-addition run, for crates/gam-mpd/examples/mpd_paths_2951.rs.

Analysis under SPEC 8's exception: torch runs the reference forward of the trained one-layer transformer
(bench/mpd_modadd_2951.py) in float64 on every (a, b) and writes ``logits.f64`` (raw little-endian, row
``a * p + b``) and ``torch.json``. The path decomposition itself is gam_mpd::paths in Rust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from mpd_modadd_2951 import all_pairs, build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="~/mpd-data/tiny_modadd/addition_p31_s0.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = torch.load(Path(args.run).expanduser(), map_location="cpu", weights_only=False)
    config = run["config"]
    step = max(run["checkpoints"])
    model = build_model(config).double()
    model.load_state_dict({k: v.double() for k, v in run["checkpoints"][step].items()})
    model.eval()
    tokens = all_pairs(config["p"])
    with torch.no_grad():
        logits = model(tokens).numpy().astype("<f8")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(logits).tofile(out / "logits.f64")
    (out / "torch.json").write_text(json.dumps({"run": args.run, "step": int(step), "p": config["p"],
                                                "shape": list(logits.shape), "rows": "a * p + b"}, indent=1))
    print("torch logits %s to %s" % (logits.shape, out))


if __name__ == "__main__":
    main()
