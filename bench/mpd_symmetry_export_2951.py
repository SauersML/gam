"""Export a network's tensors to float64 for the Rust symmetry drivers (#2951).

``modadd`` reads a ``bench/mpd_modadd_2951.py`` run (``.pt``) and writes its last checkpoint's
tensors (``W_Q``/``W_K``/``W_V`` flattened to ``(heads·d_head) × d_model``, biases as one row) with
the run's config, for ``examples/mpd_symmetry_modadd_2951.rs``.

Reads a safetensors checkpoint (Qwen3 or the VPD LlamaSimpleMLP) and writes, per layer ``L``,
``L{L}.W_Q``, ``W_K``, ``W_V``, ``W_O``, the pre-attention RMSNorm gain ``norm`` and, for Qwen3, the
per-head query and key RMSNorm gains ``q_norm``, ``k_norm`` (length ``d_head``), and for Qwen3 the
token embedding ``embed`` (tied to the unembedding), as raw little-endian float64 in C order
(``<name>.f64``), with ``export.json`` recording the heads, the key/value heads, the head width,
the layer count, ``rope_theta``, ``rms_norm_eps`` and every array's shape. ``W_Q`` is
``(heads·d_head) × d_model``, ``W_K``/``W_V`` are ``(kv_heads·d_head) × d_model``, ``W_O`` is
``d_model × (heads·d_head)``: torch's ``Linear`` layouts. Widening float32/bfloat16 to float64 is
exact. The embedding is streamed in row chunks. The Rust driver
``examples/mpd_symmetry_qwen_block_2951.rs`` reads it.

    python mpd_symmetry_export_2951.py modadd RUN.pt OUT_DIR
    python mpd_symmetry_export_2951.py qwen3 MODEL.safetensors OUT_DIR --config CONFIG.json
    python mpd_symmetry_export_2951.py vpd MODEL.safetensors OUT_DIR --config MODEL_CONFIG.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml
from safetensors import safe_open


def export_modadd(run, out):
    import hashlib
    import torch

    checkpoint = torch.load(run, map_location="cpu", weights_only=False)
    step = max(checkpoint["checkpoints"])
    state = checkpoint["checkpoints"][step]
    os.makedirs(out, exist_ok=True)
    files = {}
    for name in ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out", "W_U"]:
        array = state[name].double().numpy()
        array = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
        np.ascontiguousarray(array, dtype="<f8").tofile(os.path.join(out, f"{name}.f64"))
        files[name] = {"shape": list(array.shape)}
    with open(run, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    record = {"run": os.path.abspath(run), "run_sha256": digest, "step": int(step), "config": checkpoint["config"],
              "files": files}
    with open(os.path.join(out, "export.json"), "w") as fh:
        json.dump(record, fh, indent=1, default=str)
    print(json.dumps({"out": out, "step": int(step)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["modadd", "qwen3", "vpd"])
    parser.add_argument("weights")
    parser.add_argument("out")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.kind == "modadd":
        export_modadd(args.weights, args.out)
        return
    if args.config is None:
        parser.error(f"{args.kind} needs --config")
    if args.kind == "qwen3":
        with open(args.config) as fh:
            cfg = json.load(fh)
        layers, heads, kv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        d_head = cfg.get("head_dim", cfg["hidden_size"] // heads)
        extra = {"rope_theta": cfg["rope_theta"], "rms_norm_eps": cfg["rms_norm_eps"],
                 "tied": cfg.get("tie_word_embeddings", False)}
        names = {"W_Q": "model.layers.{}.self_attn.q_proj.weight", "W_K": "model.layers.{}.self_attn.k_proj.weight",
                 "W_V": "model.layers.{}.self_attn.v_proj.weight", "W_O": "model.layers.{}.self_attn.o_proj.weight",
                 "norm": "model.layers.{}.input_layernorm.weight", "q_norm": "model.layers.{}.self_attn.q_norm.weight",
                 "k_norm": "model.layers.{}.self_attn.k_norm.weight"}
    else:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        layers, heads, kv = cfg["n_layer"], cfg["n_head"], cfg["n_key_value_heads"]
        d_head = cfg["n_embd"] // heads
        extra = {}
        names = {"W_Q": "h.{}.attn.q_proj.weight", "W_K": "h.{}.attn.k_proj.weight", "W_V": "h.{}.attn.v_proj.weight",
                 "W_O": "h.{}.attn.o_proj.weight", "norm": "h.{}.rms_1.weight"}
    os.makedirs(args.out, exist_ok=True)
    files = {}
    with safe_open(args.weights, "pt") as f:
        for layer in range(layers):
            for short, pattern in names.items():
                array = f.get_tensor(pattern.format(layer)).double().numpy()
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                name = f"L{layer}.{short}"
                np.ascontiguousarray(array, dtype="<f8").tofile(os.path.join(args.out, f"{name}.f64"))
                files[name] = {"shape": list(array.shape)}
        if args.kind == "qwen3":
            embed = f.get_slice("model.embed_tokens.weight")
            rows, width = embed.get_shape()
            with open(os.path.join(args.out, "embed.f64"), "wb") as fh:
                for start in range(0, rows, 8192):
                    chunk = embed[start:min(rows, start + 8192)].double().numpy()
                    fh.write(np.ascontiguousarray(chunk, dtype="<f8").tobytes())
            files["embed"] = {"shape": [rows, width]}
    record = {"weights": os.path.abspath(args.weights), "kind": args.kind, "layers": layers, "heads": heads,
              "kv_heads": kv, "d_head": d_head, **extra, "files": files}
    with open(os.path.join(args.out, "export.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps({"out": args.out, "layers": layers, "heads": heads, "kv_heads": kv, "d_head": d_head}))


if __name__ == "__main__":
    main()
