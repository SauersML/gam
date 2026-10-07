"""Which native head's rows (q/k/v: output rows of U_i) or columns (o: input rows of V_i) hold each
VPD attention subcomponent whose removal costs >= 0.05 bits in REMOVAL.json (vpd_induction_removal.py):
fraction of the squared norm per head (vpd4l, 6 heads x 128).

  ~/mpd-data/venv/bin/python vpd_heads_of.py REMOVAL.json
"""
import json, sys
import numpy as np
from safetensors.numpy import load_file
uv = load_file(str(__import__("pathlib").Path.home() / "mpd-data/oracle/vpd/uv.safetensors"))
d = json.load(open(sys.argv[1]))
for l in range(4):
    for site in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"h.{l}.attn.{site}"
        kl = d["sites"][name]["kl_bits"]
        chosen = [i for i in d["sites"][name]["top"] if kl[i] >= 0.05]
        out = []
        for i in chosen:
            w = uv[f"{name}.U"][i] if site != "o_proj" else uv[f"{name}.V"][:, i]
            f = (w.reshape(6, 128) ** 2).sum(1); f = f / f.sum()
            out.append((i, round(kl[i], 3), round(d["sites"][name]["gold_drop_bits"][i], 3), int(f.argmax()), round(float(f.max()), 2)))
        print(name, out)
