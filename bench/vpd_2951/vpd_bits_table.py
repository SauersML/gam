"""Code length of VPD's reals (U,V; CI network) and of the 24 target matrices, per resolution b."""
import json, sys, torch
from vpd_bits import lattice_bits, precision_for
raw = torch.load("s-55ea3f9b/model_400000.pth", map_location="cpu", weights_only=True, mmap=True)
parts = {"uv": [], "ci": [], "target24": []}
for k, x in raw.items():
    if k.startswith("_components."): parts["uv"].append(x)
    elif "_global_ci_fn." in k and not k.endswith("inv_freq"): parts["ci"].append(x)
    elif k.startswith("target_model.h.") and ("proj" in k or "c_fc" in k): parts["target24"].append(x)
out = {}
for b in [int(v) for v in sys.argv[1].split(",")]:
    row = {}
    for name, xs in parts.items():
        n = sum(x.numel() for x in xs)
        bits = sum(lattice_bits(x, precision_for(x, b)) for x in xs)
        row[name] = {"n_reals": n, "bits": bits, "bits_per_real": bits / n}
    out[b] = row
    print(b, json.dumps(row), flush=True)
json.dump(out, open("bits_table.json", "w"), indent=1)
