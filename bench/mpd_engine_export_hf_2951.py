"""Export a Hugging Face causal language model (GPT-NeoX/Pythia, Qwen3, Llama) in the decomposition
engine's language-model format (#2951), the format ``gam_mpd::import::import_language_model`` reads:
raw little-endian float64, C order, one ``<name>.f64`` per tensor (a vector as ``[1, n]``), plus
``export.json`` {config, files: {name: {shape, sha256}}, source}.

Every stored weight (F16, BF16, F32) widens to float64 exactly. GPT-NeoX's fused ``query_key_value``
(per head ``[q; k; v]`` rows) is split into ``q_proj``, ``k_proj``, ``v_proj``; its layer norms are
``norm: layer`` with ``.bias`` tensors, its blocks ``parallel_residual``, its rotary partial
(``rotary_dims = rotary_pct * head_dim``) and its unembedding untied (``lm_head``). Qwen3 adds
``qk_norm`` gains and a gated (SwiGLU) MLP. ``--layers L`` keeps the first L blocks (then the final
norm and the unembedding: the model's own readout of its stream after block L).

Tokens: ``--tokens`` is a ``.pt`` tensor of token rows, an export directory holding ``tokens.f64``
(e.g. the VPD 4L export, whose Pile rows are GPT-NeoX tokens), or one or more ``WINDOWS.u32:ROWS``,
the first ROWS rows of ``--context`` little-endian u32 tokens of each window file
(``mpd_qwen3_fineweb_2951.py``), concatenated in order (training rows, then held-out rows). The check: the full fp32 forward of
the (truncated) model on token row 0, positions ``0..T``, as ``logits_row0.f64`` (``T × vocab``).

usage: mpd_engine_export_hf_2951.py MODEL_DIR OUT_DIR --tokens PATH [PATH ...] [--layers L] [--context T]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--tokens", nargs="+", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--context", type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    weights = os.path.join(args.model, "model.safetensors")
    sd = load_file(weights)
    files = {}

    def put(name, a):
        a = a.detach().double().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        path = os.path.join(args.out, f"{name}.f64")
        np.ascontiguousarray(a, dtype="<f8").tofile(path)
        files[name] = {"shape": list(a.shape), "sha256": sha256(path)}

    kind = cfg["model_type"]
    layers = min(args.layers or cfg["num_hidden_layers"], cfg["num_hidden_layers"])
    d, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or d // heads
    if kind == "gpt_neox":
        kv_heads = heads
        put("wte", sd["gpt_neox.embed_in.weight"])
        put("lm_head", sd["embed_out.weight"])
        put("final_norm.gain", sd["gpt_neox.final_layer_norm.weight"])
        put("final_norm.bias", sd["gpt_neox.final_layer_norm.bias"])
        for l in range(layers):
            p, q = f"gpt_neox.layers.{l}.", f"blocks.{l}."
            w = sd[p + "attention.query_key_value.weight"].reshape(heads, 3, hd, d)
            b = sd[p + "attention.query_key_value.bias"].reshape(heads, 3, hd)
            for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
                put(q + f"attn.{name}", w[:, i].reshape(heads * hd, d))
                put(q + f"attn.{name}.bias", b[:, i].reshape(heads * hd))
            put(q + "attn.o_proj", sd[p + "attention.dense.weight"])
            put(q + "attn.o_proj.bias", sd[p + "attention.dense.bias"])
            put(q + "mlp.c_fc", sd[p + "mlp.dense_h_to_4h.weight"])
            put(q + "mlp.c_fc.bias", sd[p + "mlp.dense_h_to_4h.bias"])
            put(q + "mlp.down_proj", sd[p + "mlp.dense_4h_to_h.weight"])
            put(q + "mlp.down_proj.bias", sd[p + "mlp.dense_4h_to_h.bias"])
            for n, src in (("rms1", "input_layernorm"), ("rms2", "post_attention_layernorm")):
                put(q + f"{n}.gain", sd[p + f"{src}.weight"])
                put(q + f"{n}.bias", sd[p + f"{src}.bias"])
        config = {
            "norm": "layer", "parallel_residual": cfg.get("use_parallel_residual", True),
            "rotary_dims": int(cfg["rotary_pct"] * hd), "rope_theta": float(cfg.get("rotary_emb_base", 10000)),
            "norm_eps": cfg["layer_norm_eps"], "mlp_act": {"gelu": "gelu", "gelu_new": "gelu_tanh"}[cfg["hidden_act"]],
            "tied_embeddings": False, "mlp_gated": False, "qk_norm": False,
        }
    elif kind in ("qwen3", "llama", "qwen2"):
        kv_heads = cfg["num_key_value_heads"]
        tied = cfg.get("tie_word_embeddings", False)
        put("wte", sd["model.embed_tokens.weight"])
        if not tied:
            put("lm_head", sd["lm_head.weight"])
        put("final_norm.gain", sd["model.norm.weight"])
        for l in range(layers):
            p, q = f"model.layers.{l}.", f"blocks.{l}."
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                put(q + f"attn.{name}", sd[p + f"self_attn.{name}.weight"])
                if p + f"self_attn.{name}.bias" in sd:
                    put(q + f"attn.{name}.bias", sd[p + f"self_attn.{name}.bias"])
            if kind == "qwen3":
                put(q + "attn.q_norm.gain", sd[p + "self_attn.q_norm.weight"])
                put(q + "attn.k_norm.gain", sd[p + "self_attn.k_norm.weight"])
            put(q + "mlp.c_fc", sd[p + "mlp.up_proj.weight"])
            put(q + "mlp.gate_proj", sd[p + "mlp.gate_proj.weight"])
            put(q + "mlp.down_proj", sd[p + "mlp.down_proj.weight"])
            put(q + "rms1.gain", sd[p + "input_layernorm.weight"])
            put(q + "rms2.gain", sd[p + "post_attention_layernorm.weight"])
        config = {
            "norm": "rms", "parallel_residual": False, "rotary_dims": hd, "rope_theta": float(cfg["rope_theta"]),
            "norm_eps": cfg["rms_norm_eps"], "mlp_act": {"silu": "silu"}[cfg["hidden_act"]],
            "tied_embeddings": tied, "mlp_gated": True, "qk_norm": kind == "qwen3",
        }
    else:
        raise SystemExit(f"unsupported model_type {kind}")
    vocab = files["wte"]["shape"][0]
    config.update({
        "d_model": d, "n_layers": layers, "n_heads": heads, "n_kv_heads": kv_heads, "head_dim": hd,
        "d_mlp": cfg["intermediate_size"], "vocab": vocab, "rope_pairing": "rotate_half",
        "n_ctx": cfg.get("max_position_embeddings"),
    })
    del sd

    if all(".u32:" in t for t in args.tokens):
        parts = []
        for item in args.tokens:
            path, rows = item.rsplit(":", 1)
            rows = int(rows)
            part = np.fromfile(path, dtype="<u4", count=rows * args.context)
            if part.size != rows * args.context:
                raise SystemExit(f"{path}: fewer than {rows} rows of {args.context} tokens")
            parts.append(part.reshape(rows, args.context))
        ids = torch.from_numpy(np.concatenate(parts).astype(np.int64))
    elif len(args.tokens) != 1:
        raise SystemExit("several --tokens are window files PATH.u32:ROWS")
    elif args.tokens[0].endswith(".pt"):
        ids = torch.load(args.tokens[0], map_location="cpu", weights_only=True)
        ids = ids["tokens"] if isinstance(ids, dict) else ids
    else:
        rec = json.load(open(os.path.join(args.tokens[0], "export.json")))
        shape = rec["files"]["tokens"]["shape"]
        ids = torch.from_numpy(np.fromfile(os.path.join(args.tokens[0], "tokens.f64"), dtype="<f8").reshape(shape)).long()
    ids = ids.long()
    if int(ids.max()) >= vocab:
        raise SystemExit(f"a token {int(ids.max())} beyond the vocabulary {vocab}")
    put("tokens", ids.double())
    T = min(args.context, ids.shape[1])

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    stack = model.gpt_neox.layers if kind == "gpt_neox" else model.model.layers
    del stack[layers:]
    model.config.num_hidden_layers = layers
    model.eval()
    with torch.no_grad():
        logits = model(ids[:1, :T]).logits[0]
    put("logits_row0", logits.double())

    record = {
        "source": {"model": args.model, "weights": weights, "weights_sha256": sha256(weights), "tokens": args.tokens,
                   "layers_kept": layers, "logits": "fp32 Hugging Face forward on CPU of the kept layers, widened to f64"},
        "config": config, "files": files,
    }
    json.dump(record, open(os.path.join(args.out, "export.json"), "w"), indent=1)
    print(json.dumps({"out": args.out, "files": len(files), "config": config}))


if __name__ == "__main__":
    main()
