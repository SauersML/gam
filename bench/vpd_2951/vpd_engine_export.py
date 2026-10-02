"""Export the 4L Pile target (t-9d2b8f02) in the decomposition engine's input format (#2951):
raw little-endian float64, C order, one <name>.f64 per tensor, vectors as [1, n], plus export.json
{config, files: {name: {shape, sha256}}} (bench/mpd_engine_export_2951.py is the reference writer).

Data, matching the paper's eval batch (128 rows x 512 context): tokens.f64 [N, T+1], val-00000 rows
ROW0..ROW0+N (the model reads tokens[:, :T]; tokens[:, 1:] are the next-token labels),
row_ids.f64 [1, N] (val-00000 row index; the dataset is shuffled 513-token chunks without doc ids).
Target outputs on tokens[:, :T] (fp32 forward, widened): logits_topk_values.f64 / logits_topk_indices.f64
[N * T, K] (descending) and logits_logsumexp.f64 [1, N * T] (full-vocab logsumexp), plus the full
logits of row 0, logits_row0.f64 [T, vocab], as an exact check.

usage: vpd_engine_export.py OUT_DIR [N T ROW0 K]
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from vpd_model import TARGET_DIR, load_target, val_tokens

out = sys.argv[1]
N, T, ROW0, K = (int(x) for x in sys.argv[2:6]) if len(sys.argv) > 2 else (128, 512, 0, 64)
os.makedirs(out, exist_ok=True)
files = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put(name: str, a: np.ndarray):
    if a.ndim == 1:
        a = a.reshape(1, -1)
    path = os.path.join(out, f"{name}.f64")
    np.ascontiguousarray(a, dtype="<f8").tofile(path)
    files[name] = {"shape": list(a.shape), "sha256": sha256(path)}


cfg = yaml.safe_load((TARGET_DIR / "model_config.yaml").read_text())
sd = load_file(str(TARGET_DIR / "model_step_99999.safetensors"))
f64 = lambda k: sd[k].double().numpy()
put("wte", f64("wte.weight"))
put("final_norm.gain", f64("ln_f.weight"))
for l in range(cfg["n_layer"]):
    for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
        put(f"blocks.{l}.attn.{k}", f64(f"h.{l}.attn.{k}.weight"))
    for k in ("c_fc", "down_proj"):
        put(f"blocks.{l}.mlp.{k}", f64(f"h.{l}.mlp.{k}.weight"))
    put(f"blocks.{l}.rms1.gain", f64(f"h.{l}.rms_1.weight"))
    put(f"blocks.{l}.rms2.gain", f64(f"h.{l}.rms_2.weight"))

ids = val_tokens(N, seq=T + 1, offset=ROW0)
put("tokens", ids.double().numpy())
put("row_ids", np.arange(ROW0, ROW0 + N, dtype=np.float64))
target = load_target("mps")
vals, idxs, lses = [], [], []
with torch.no_grad():
    for i in range(0, N, 4):
        lg = target(ids[i:i + 4, :T].to("mps")).flatten(0, 1)
        v, ix = lg.topk(K, dim=-1)
        vals.append(v.cpu().double())
        idxs.append(ix.cpu().double())
        lses.append(lg.logsumexp(-1).cpu().double())
        if i == 0:
            put("logits_row0", lg[:T].cpu().double().numpy())
        del lg
put("logits_topk_values", torch.cat(vals).numpy())
put("logits_topk_indices", torch.cat(idxs).numpy())
put("logits_logsumexp", torch.cat(lses).numpy())

config = {
    "d_model": cfg["n_embd"], "n_layers": cfg["n_layer"], "n_heads": cfg["n_head"],
    "n_kv_heads": cfg["n_key_value_heads"], "head_dim": cfg["n_embd"] // cfg["n_head"],
    "d_mlp": cfg["n_intermediate"], "vocab": cfg["vocab_size"], "rope_theta": float(cfg["rotary_base"]),
    "rope_pairing": "rotate_half", "norm_eps": float(cfg["rms_norm_eps"]), "mlp_act": "gelu_tanh",
    "tied_embeddings": True, "n_ctx": cfg["n_ctx"],
}
record = {
    "source": {"target_run": "goodfire/spd/runs/t-9d2b8f02",
               "checkpoint": str(TARGET_DIR / "model_step_99999.safetensors"),
               "checkpoint_sha256": sha256(TARGET_DIR / "model_step_99999.safetensors"),
               "data": "danbraunai/pile-uncopyrighted-tok-shuffled val-00000-of-00012.parquet",
               "token_rows": [ROW0, ROW0 + N], "context": T, "topk": K,
               "logits": "fp32 forward on MPS (vpd_model.Target), widened to f64"},
    "config": config, "files": files,
    "reference_forward": "~/mpd-data/vpd/vpd_model.py Target (pre-RMSNorm w * x * rsqrt(mean x^2 + eps); "
                         "rotate-half RoPE on q, k; causal softmax(q k^T / sqrt(hd)); GELU tanh; "
                         "final RMSNorm; logits = x @ wte^T)",
}
json.dump(record, open(os.path.join(out, "export.json"), "w"), indent=1)
print(json.dumps({"out": out, "files": len(files), "tokens": [N, T + 1]}))
