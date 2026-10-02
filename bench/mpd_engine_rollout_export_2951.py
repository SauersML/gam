"""A language-model export whose eval rows are the model's own continuations (#2951 suite).

usage: mpd_engine_rollout_export_2951.py HF_MODEL_DIR EXPORT_DIR OUT_DIR TRAIN EVAL PREFIX HORIZON

OUT_DIR links every tensor of EXPORT_DIR and replaces tokens.f64: rows [0, TRAIN) are the
export's own; rows [TRAIN, TRAIN + EVAL) are the export's held-out rows cut to PREFIX tokens and
continued HORIZON tokens by sampling the model (seeded), padded with the next sampled token to the
export's width. rollout.json records PREFIX, HORIZON and the rows, so the per-step KL of a program
on positions PREFIX-1 .. PREFIX+HORIZON-2 of each eval row sums to its sequence KL.
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

hf, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
train, ev, prefix, horizon = (int(x) for x in sys.argv[4:8])
os.makedirs(out, exist_ok=True)
rec = json.load(open(os.path.join(src, "export.json")))
rows, cols = rec["files"]["tokens"]["shape"]
tokens = np.fromfile(os.path.join(src, "tokens.f64"), dtype="<f8").reshape(rows, cols)
assert train + ev <= rows and prefix + horizon <= cols
for name in os.listdir(src):
    if name.endswith(".f64") and name != "tokens.f64":
        link = os.path.join(out, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, name), link)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
gen = torch.Generator().manual_seed(0x110A75)
table = tokens[: train + ev].copy()
with torch.no_grad():
    seq = torch.tensor(tokens[train:train + ev, :prefix], dtype=torch.long)
    while seq.shape[1] < cols:
        probs = torch.softmax(model(seq).logits[:, -1].double(), -1)
        seq = torch.cat([seq, torch.multinomial(probs, 1, generator=gen)], 1)
table[train:train + ev] = seq.numpy().astype(np.float64)
np.ascontiguousarray(table, dtype="<f8").tofile(os.path.join(out, "tokens.f64"))
rec["files"]["tokens"] = {"shape": [train + ev, cols]}
rec["rollout"] = {"source": src, "train": train, "eval": ev, "prefix": prefix, "horizon": horizon, "seed": 0x110A75}
rec["files"].pop("logits_row0", None)
json.dump(rec, open(os.path.join(out, "export.json"), "w"), indent=1)
json.dump(rec["rollout"], open(os.path.join(out, "rollout.json"), "w"), indent=1)
print(out, table.shape)
