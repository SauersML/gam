#!/usr/bin/env python3
"""Standalone GPU harvest of Gemma residual-stream activations for the wall-closure rerun.

Byte-for-byte reproduces gemma_wall_closure.harvest(): same tokenization
(add_special_tokens=False), same within-document position bookkeeping
(positions restart at 0 per document), same layer indexing (hidden_states[layer+1]),
and the same output contract (acts_L{layer}.npy float32 + positions.npy int32).

This exists so the heavy GPU forward pass runs once as an sbatch and the CPU-bound
gemma_wall_closure.py driver consumes the cached arrays (it skips its own harvest
when acts_L{layer}.npy and positions.npy already exist in --out-dir).

A manifest.json is also written for provenance, and resid_L{layer}.npy /
positions_L{layer}.npy aliases are created to mirror the Qwen harvest naming.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_texts(args: argparse.Namespace) -> list[str]:
    # Offline path: a pre-materialised corpus (streamed + filtered on the login
    # node) so the GPU compute node never needs internet. Order and filter are
    # identical to the streaming path below, so the two produce the same corpus.
    if getattr(args, "texts_file", None):
        with open(args.texts_file) as handle:
            texts = json.load(handle)
        texts = [t for t in texts if len(t) >= args.min_chars][: args.max_docs]
        if not texts:
            raise SystemExit(f"texts file {args.texts_file} produced no usable texts")
        return texts

    from datasets import load_dataset

    texts: list[str] = []
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    for row in ds:
        text = (row.get(args.text_field) or row.get("content") or "").strip()
        if len(text) < args.min_chars:
            continue
        texts.append(text)
        if len(texts) >= args.max_docs:
            break
    if not texts:
        raise SystemExit("dataset stream produced no usable texts")
    return texts


def hidden_size_from_model(model) -> int:
    config = model.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    return int(config.text_config.hidden_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layers", default="12,25")
    ap.add_argument("--n-tokens", type=int, default=30_000)
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--batch-docs", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--texts-file", default=None,
                    help="optional pre-materialised JSON list of doc strings (offline corpus)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layers:
        raise SystemExit("at least one layer is required")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = {layer: out_dir / f"acts_L{layer}.npy" for layer in layers}
    pos_path = out_dir / "positions.npy"
    if all(path.exists() for path in paths.values()) and pos_path.exists():
        log("using cached activation arrays; nothing to do")
        return

    log(f"loading tokenizer/model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    d_model = hidden_size_from_model(model)
    n_layers = int(getattr(model.config, "num_hidden_layers", getattr(model.config, "text_config", model.config).num_hidden_layers))
    log(f"model loaded: n_layers {n_layers} hidden {d_model} device {device} "
        f"cuda_devices {torch.cuda.device_count()}")
    for layer in layers:
        if not 0 <= layer <= n_layers:
            raise SystemExit(f"layer {layer} out of range for hidden_states[layer+1] (n_layers={n_layers})")

    arrays = {
        layer: np.lib.format.open_memmap(
            paths[layer], mode="w+", dtype=np.float32, shape=(args.n_tokens, d_model)
        )
        for layer in layers
    }
    positions = np.lib.format.open_memmap(
        pos_path, mode="w+", dtype=np.int32, shape=(args.n_tokens,)
    )

    log(f"loading corpus {args.dataset}/{args.dataset_config}")
    texts = load_texts(args)
    log(f"collected {len(texts)} docs (min_chars {args.min_chars})")

    # NOTE: gemma-2's tokenizer pads on the LEFT (padding_side="left"), so a
    # positional slice hidden[row, :take] would grab PAD rows, not real tokens.
    # We select real tokens by the attention mask (like the Qwen harvest_acts.py
    # h[mask] path), keeping doc order, so position 0 == first real token of a
    # doc. This is padding-side-agnostic and correct for the pos0 nuisance peel.
    cursor = 0
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_docs):
            batch = texts[start : start + args.batch_docs]
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            mask = encoded["attention_mask"].bool()
            lengths = mask.sum(dim=1).cpu().tolist()
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            take_remaining = args.n_tokens - cursor
            for row, length in enumerate(lengths):
                if take_remaining <= 0:
                    break
                take = min(int(length), take_remaining)
                if take <= 0:
                    continue
                row_mask = mask[row]  # (T,) real-token selector, left-pad safe
                positions[cursor : cursor + take] = np.arange(take, dtype=np.int32)
                for layer in layers:
                    hidden = output.hidden_states[layer + 1][row][row_mask][:take].detach().float().cpu()
                    arrays[layer][cursor : cursor + take] = hidden.numpy()
                cursor += take
                take_remaining = args.n_tokens - cursor
            if (start // args.batch_docs) % 25 == 0:
                log(f"harvested {cursor}/{args.n_tokens} residual rows")
            del output, encoded
            if cursor >= args.n_tokens:
                break

    if cursor < args.n_tokens:
        raise SystemExit(f"only harvested {cursor} tokens; increase --max-docs")
    for arr in arrays.values():
        arr.flush()
    positions.flush()
    log(f"harvest complete: {cursor} rows per layer")

    # Qwen-style aliases for the peel/positions pipeline naming.
    for layer in layers:
        resid_alias = out_dir / f"resid_L{layer}.npy"
        pos_alias = out_dir / f"positions_L{layer}.npy"
        for alias, target in ((resid_alias, paths[layer].name), (pos_alias, pos_path.name)):
            try:
                if alias.exists() or alias.is_symlink():
                    alias.unlink()
                os.symlink(target, alias)
            except OSError as exc:
                log(f"alias {alias.name} skipped: {exc}")

    pos_arr = np.load(pos_path, mmap_mode="r")
    manifest = {
        "model": args.model,
        "canonical_model": "google/gemma-2-2b",
        "model_source_note": (
            "harvested from an ungated bit-identical mirror when args.model != google/gemma-2-2b "
            "(no HF token available on MSI to access the gated google/gemma-2-2b); "
            "config verified architecturally identical to google/gemma-2-2b"
        ),
        "n_layers": n_layers,
        "hidden": d_model,
        "seq_len": args.max_length,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "split": args.split,
        "min_chars": args.min_chars,
        "n_docs": len(texts),
        "add_special_tokens": False,
        "layer_index": "hidden_states[layer+1]",
        "n_tokens": int(cursor),
        "position0_rows": int(np.count_nonzero(np.asarray(pos_arr) == 0)),
        "layers": {
            str(layer): {
                "acts_file": paths[layer].name,
                "resid_alias": f"resid_L{layer}.npy",
                "positions_file": pos_path.name,
                "shape": [int(cursor), int(d_model)],
            }
            for layer in layers
        },
        "positions_file": pos_path.name,
        "ts": time.time(),
    }
    with open(out_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"manifest written {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
