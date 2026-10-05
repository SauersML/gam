"""Export VPD's decomposition of the 4-layer Pile model (goodfire/spd/runs/s-55ea3f9b,
model_400000.pth) once into the engine's export format, for gam_mpd::explanation_battery (#2951).

Per site h.{l}.{attn,mlp}.{q,k,v,o,c_fc,down}_proj: `{site}.U` (C x d_out) and `{site}.V`
(d_in x C), so the site computes ((x V) * mask) U + delta_mask * x Delta^T with Delta = W - (V U)^T
from M's own export. The causal-importance network: its input projection (`ci.input.W`, inputs x
2048, and `ci.input.b`), per block `ci.blocks.{i}.{q,k,v,o}` (out x in), `ci.blocks.{i}.fc1.{W,b}`
and `ci.blocks.{i}.fc2.{W,b}` (W as in x out), and its head (`ci.head.W`, `ci.head.b`). Every file
is little-endian float64, row-major, its shape and sha256 in export.json.

usage: vpd_export.py OUT_DIR
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vpd_model import VPD_PTH, site_names

TARGET_EXPORT = Path.home() / "mpd-data/engine/vpd4l_clean4096"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    files = {}

    def write(name: str, tensor: torch.Tensor):
        values = tensor.detach().double().numpy()
        values = values.reshape(1, -1) if values.ndim == 1 else values
        path = out / f"{name}.f64"
        values.astype("<f8").tofile(path)
        files[name] = {"shape": list(values.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    sites = site_names()
    subcomponents = {}
    largest = 0.0
    target = json.loads((TARGET_EXPORT / "export.json").read_text())
    for site in sites:
        key = "_components." + site.replace(".", "-")
        u, v = raw[key + ".U"], raw[key + ".V"]
        subcomponents[site] = int(u.shape[0])
        write(f"{site}.U", u)
        write(f"{site}.V", v)
        # the checkpoint's target weights against M's export (the same model)
        name = "blocks." + site.split(".", 1)[1]
        w = np.fromfile(TARGET_EXPORT / f"{name}.f64", dtype="<f8")[-int(np.prod(target["files"][name]["shape"])):]
        w = w.reshape(target["files"][name]["shape"])
        largest = max(largest, float(np.abs(raw[f"target_model.{site}.weight"].double().numpy() - w).max()))
    ci = {k.split("_global_ci_fn.", 1)[1]: t for k, t in raw.items() if "_global_ci_fn." in k}
    write("ci.input.W", ci["_input_projector.W"])
    write("ci.input.b", ci["_input_projector.b"])
    write("ci.head.W", ci["_output_head.W"])
    write("ci.head.b", ci["_output_head.b"])
    blocks = 1 + max(int(k.split(".")[1]) for k in ci if k.startswith("_blocks."))
    for i in range(blocks):
        p = f"_blocks.{i}."
        for name, key in (("q", "attn.q_proj.weight"), ("k", "attn.k_proj.weight"), ("v", "attn.v_proj.weight"), ("o", "attn.out_proj.weight")):
            write(f"ci.blocks.{i}.{name}", ci[p + key])
        write(f"ci.blocks.{i}.fc1.W", ci[p + "mlp.0.W"])
        write(f"ci.blocks.{i}.fc1.b", ci[p + "mlp.0.b"])
        write(f"ci.blocks.{i}.fc2.W", ci[p + "mlp.2.W"])
        write(f"ci.blocks.{i}.fc2.b", ci[p + "mlp.2.b"])
    d_model = int(ci["_input_projector.W"].shape[1])
    record = {
        "source": {"checkpoint": str(VPD_PTH), "run": "goodfire/spd/runs/s-55ea3f9b", "target_export": str(TARGET_EXPORT),
                   "target_weight_max_difference": largest},
        "config": {
            "sites": sites,
            "subcomponents": subcomponents,
            "ci": {"order": sorted(sites), "d_model": d_model, "blocks": blocks, "heads": 16, "head_dim": d_model // 16,
                   "mlp_hidden": int(ci["_blocks.0.mlp.0.W"].shape[1]), "rope_base": 10000,
                   "epsilon": float(torch.finfo(torch.float32).eps), "activation": "gelu", "output": "clamp01"},
        },
        "files": files,
    }
    (out / "export.json").write_text(json.dumps(record, indent=1))
    print(f"{out}: {len(files)} files, target weights differ by at most {largest:.3g}")


if __name__ == "__main__":
    main()
