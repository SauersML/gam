"""Joint activation + TRUE next-token behavior harvest for #2015.

The msae_l17 cache stores mid-stack residual activations with no token ids, and
a logit-lens probe at layer 23/40 measures near-uniform distributions (top-63
global tokens carry only ~0.3% of the mass) — honest but uninformative. The
behavior-anchored fit needs the model's REAL next-token distribution, so this
script runs a full causal LM forward over real corpus text and captures, for
the SAME token positions,

  - the residual-stream hidden state at the requested layers, and
  - softmax(logits) — the true next-token behavioral response,

coarse-grained to a global top-(V-1) token set plus one renormalized tail
bucket (exact coarse-grained distributions; the KL downstream is exact for this
readout and the coarse-graining is recorded).

Runs offline on a GPU node (pre-download the model + corpus on a login node):

    python scripts/qwen_joint_behavior_harvest.py \
        --model-dir <hf snapshot dir> --parquet <wikitext parquet> \
        --rows 4000 --layers 18,24 --top-v 64 --seq-len 512 \
        --out qwen3_8b_joint_behavior.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def iter_texts(parquet_path: str):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["text"])
    for value in table.column("text").to_pylist():
        text = (value or "").strip()
        if len(text) > 200:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument(
        "--layers", default="auto", help="hidden_states indices to save, or 'auto' (n/3, 2n/3)"
    )
    parser.add_argument("--top-v", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map="cuda"
        )
    except ValueError:
        # Qwen3.5/3.6 checkpoints are multimodal ConditionalGeneration models;
        # a text-only forward through them is still the plain causal LM.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map="cuda"
        )
    model.eval()
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    vocab = config.vocab_size
    n_layers = config.num_hidden_layers
    if args.layers == "auto":
        layers = sorted({n_layers // 3, (2 * n_layers) // 3})
    else:
        layers = [int(x) for x in args.layers.split(",")]
    print(f"model has {n_layers} layers; capturing hidden_states at {layers}", flush=True)

    acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    prob_chunks: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    mean_probs = np.zeros(vocab, dtype=np.float64)
    harvested = 0
    texts = iter_texts(args.parquet)
    with torch.no_grad():
        while harvested < args.rows:
            try:
                text = next(texts)
            except StopIteration:
                raise SystemExit(
                    f"corpus exhausted at {harvested}/{args.rows} rows; supply more text"
                )
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
            ids = {k: v.to("cuda") for k, v in ids.items()}
            n_tok = ids["input_ids"].shape[1]
            if n_tok < 32:
                continue
            out = model(**ids, output_hidden_states=True)
            # Positions 0..n-2 predict positions 1..n-1: keep rows where the
            # next-token distribution is a genuine prediction.
            take = min(n_tok - 1, args.rows - harvested)
            for layer in layers:
                hidden = out.hidden_states[layer][0, :take]
                acts[layer].append(hidden.float().cpu().numpy())
            logits = out.logits[0, :take].float()
            probs = torch.softmax(logits.double(), dim=-1).cpu().numpy()
            mean_probs += probs.sum(axis=0)
            prob_chunks.append(probs.astype(np.float32))
            token_rows.append(ids["input_ids"][0, :take].cpu().numpy())
            harvested += take
            if harvested % 512 < take:
                print(f"harvested {harvested}/{args.rows}", flush=True)

    mean_probs /= harvested
    keep = np.sort(np.argsort(mean_probs)[::-1][: args.top_v - 1])
    n = harvested
    out_probs = np.zeros((n, args.top_v), dtype=np.float64)
    row = 0
    for chunk in prob_chunks:
        c = chunk.astype(np.float64)
        out_probs[row : row + c.shape[0], : args.top_v - 1] = c[:, keep]
        out_probs[row : row + c.shape[0], args.top_v - 1] = np.clip(
            1.0 - c[:, keep].sum(axis=1), 0.0, 1.0
        )
        row += c.shape[0]
    out_probs /= out_probs.sum(axis=1, keepdims=True)

    payload = {
        "probs": out_probs,
        "token_ids_kept": keep.astype(np.int64),
        "input_token_ids": np.concatenate(token_rows),
        "provenance": json.dumps(
            {
                "issue": 2015,
                "model_dir": args.model_dir,
                "readout": "true next-token softmax from full forward",
                "layers": layers,
                "rows": n,
                "top_v": args.top_v,
                "seq_len": args.seq_len,
                "tail_bucket": "last column = renormalized residual vocab mass",
            }
        ),
    }
    for layer in layers:
        payload[f"acts_L{layer}"] = np.concatenate(acts[layer], axis=0).astype(np.float32)
    np.savez_compressed(args.out, **payload)
    tail = out_probs[:, -1]
    print(
        f"wrote {args.out}: probs {out_probs.shape}, "
        + ", ".join(f"acts_L{l} {np.concatenate(acts[l]).shape}" for l in layers)
        + f", tail mass mean={tail.mean():.4f} max={tail.max():.4f}"
    )


if __name__ == "__main__":
    main()
