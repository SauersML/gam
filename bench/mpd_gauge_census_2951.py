"""#2951 gauge census: export Qwen3 weights for crates/gam-mpd/examples/mpd_gauge_census_2951.rs.

numpy (+ huggingface_hub to locate the snapshot). Reads the safetensors snapshot (bf16 -> float64 is exact)
with the lazy reader of bench/mpd_opfirst_decoder_2951.py, and writes one float64 .npy per tensor the gauge
detectors read:

    <out>/manifest.json                      config fields, tensor shapes, parameter counts
    <out>/final_norm.npy                     model.norm.weight
    <out>/embed_rows.npy                     every `--embed-stride`-th embedding row (a rank lower bound)
    <out>/layer_XX/{q,k,v,o,gate,up,down}.npy
    <out>/layer_XX/{input_norm,post_norm,q_norm,k_norm}.npy

Usage: python bench/mpd_gauge_census_2951.py --out /path/to/export [--model Qwen/Qwen3-0.6B-Base]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import Decoder  # noqa: E402

LAYER_TENSORS = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
    "input_norm": "input_layernorm.weight",
    "post_norm": "post_attention_layernorm.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-stride", type=int, default=16)
    args = ap.parse_args()

    w = Decoder(args.model)
    snap, cfg = w.snap, w.config
    os.makedirs(args.out, exist_ok=True)

    # Every stored tensor's parameter count, so the census total is the checkpoint's.
    stored = {name: int(np.prod(meta["shape"])) for name, (_, _, meta) in w.index.items()}
    unexpected = [n for n in stored if n.startswith("model.layers.") and not any(n.endswith(s) for s in LAYER_TENSORS.values())]
    if unexpected:
        raise SystemExit(f"tensors the census does not model: {unexpected[:5]}")
    for name in stored:
        if name.endswith(".bias"):
            raise SystemExit(f"bias {name}: the census models bias-free Qwen3 blocks only")

    np.save(os.path.join(args.out, "final_norm.npy"), w("model.norm.weight"))
    embed = w("model.embed_tokens.weight")
    np.save(os.path.join(args.out, "embed_rows.npy"), np.ascontiguousarray(embed[:: args.embed_stride]))
    tied = bool(cfg.get("tie_word_embeddings", False)) and "lm_head.weight" not in stored
    layers = int(cfg["num_hidden_layers"])
    for layer in range(layers):
        d = os.path.join(args.out, f"layer_{layer:02d}")
        os.makedirs(d, exist_ok=True)
        for short, suffix in LAYER_TENSORS.items():
            np.save(os.path.join(d, f"{short}.npy"), w(f"model.layers.{layer}.{suffix}"))
        print(f"layer {layer} exported", flush=True)

    manifest = {
        "model": args.model,
        "snapshot": os.path.basename(snap),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_hidden_layers": layers,
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(cfg["num_key_value_heads"]),
        "head_dim": int(cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])),
        "rope_theta": float(cfg["rope_theta"]),
        "rms_norm_eps": float(cfg["rms_norm_eps"]),
        "vocab_size": int(embed.shape[0]),
        "tie_word_embeddings": tied,
        "embed_stride": args.embed_stride,
        "stored_parameters": stored,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
