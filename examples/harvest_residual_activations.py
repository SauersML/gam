#!/usr/bin/env python
"""Harvest token-level residual-stream activations from a causal LM.

Thin numeric adapter (the #977 boundary: activations are just a response
matrix). Streams documents from a Hugging Face dataset, tokenizes into
fixed-length sequences, hooks one decoder layer's output, and saves all
token positions (BOS dropped) until the requested token budget is reached.

Output format is a generic Manifold-SAE activation cache: a dict with
``X`` of shape ``(n_tokens, d_model)`` float32 and a ``sig`` metadata dict.

Example:
  python harvest_residual_activations.py --model Qwen/Qwen2.5-0.5B \
      --dataset wikitext --config wikitext-103-raw-v1 --layer 12 \
      --n-tokens 120000 --out qwen05_wikitext_l12.pt
"""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-tokens", type=int, default=120_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-seqs", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.to(device).eval()

    layers = model.model.layers
    if not (0 <= args.layer < len(layers)):
        raise SystemExit(f"--layer {args.layer} out of range (model has {len(layers)})")

    grabbed: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grabbed.append(h.detach().float().cpu())

    handle = layers[args.layer].register_forward_hook(hook)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    buf: list[int] = []
    chunks: list[torch.Tensor] = []
    collected = 0
    batch: list[list[int]] = []

    def flush_batch() -> None:
        nonlocal collected
        if not batch:
            return
        ids = torch.tensor(batch, device=device)
        with torch.no_grad():
            model(ids)
        h = grabbed.pop()  # (B, seq_len, D)
        h = h[:, 1:, :].reshape(-1, h.shape[-1])  # drop position 0 per sequence
        chunks.append(h)
        collected += h.shape[0]
        batch.clear()
        print(f"[harvest] {collected}/{args.n_tokens} tokens", flush=True)

    for doc in ds:
        text = doc.get(args.text_field) or ""
        if not text.strip():
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        while len(buf) >= args.seq_len:
            batch.append(buf[: args.seq_len])
            buf = buf[args.seq_len :]
            if len(batch) >= args.batch_seqs:
                flush_batch()
        if collected >= args.n_tokens:
            break
    flush_batch()
    handle.remove()

    X = torch.cat(chunks, dim=0)[: args.n_tokens]
    sig = {
        "model_name": args.model,
        "layer": str(args.layer),
        "text_dataset": f"{args.dataset}/{args.config}",
        "text_subset": args.split,
        "seq_len": str(args.seq_len),
        "n_tokens": int(X.shape[0]),
    }
    torch.save({"X": X, "sig": sig}, args.out)
    print(f"[harvest] saved {args.out} X={tuple(X.shape)}", flush=True)


if __name__ == "__main__":
    main()
