"""Harvest the #2015 behavior response for the msae_l17 activation cache.

The activation shards (``acts_L11/acts_L17/acts_L23``) carry no token ids, so
the true next-token distribution cannot be replayed. What CAN be computed
exactly from the cache is the probe-head readout the issue itself admits as the
behavioral response: the model's own output head applied to the latest cached
residual-stream layer,

    p_i = softmax( lm_head( rms_norm(acts_L23[i]) ) ).

This is a deterministic function of the real activations through the real model
head - no fitting, no sampling. Because the downstream sphere-tangent behavior
block keeps all V-1 tangent coordinates, the full-vocabulary distribution is
coarse-grained to a fixed global top-(V-1) token set plus one renormalized tail
bucket. KL statements downstream are exact for this coarse-grained readout and
the coarse-graining is recorded in the output for provenance.

Only the head tensors (final norm + lm_head / tied embedding) are loaded - a
few hundred MB - so this runs on a login node without a GPU.

Usage (MSI):
    python scripts/qwen_behavior_probs_harvest.py \
        --shards "$GAM_MSI_DATA/msae_l17/data/shards" \
        --model-dir "$HF_HOME/hub/<qwen snapshot dir>" \
        --rows 4000 --layer acts_L23 --top-v 64 \
        --out behavior_probs_l23_top64.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from safetensors import safe_open


def load_head_tensors(model_dir: str):
    """Load (norm_weight, head_weight, rms_eps) without touching other shards."""
    with open(os.path.join(model_dir, "config.json")) as fh:
        config = json.load(fh)
    eps = float(config.get("rms_norm_eps", 1e-6))

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
    else:
        single = os.path.join(model_dir, "model.safetensors")
        with safe_open(single, framework="numpy") as fh:
            weight_map = {name: "model.safetensors" for name in fh.keys()}

    def fetch(name: str) -> np.ndarray:
        shard = os.path.join(model_dir, weight_map[name])
        with safe_open(shard, framework="numpy") as fh:
            return np.asarray(fh.get_tensor(name), dtype=np.float64)

    norm_w = fetch("model.norm.weight")
    head_name = (
        "lm_head.weight" if "lm_head.weight" in weight_map else "model.embed_tokens.weight"
    )
    head_w = fetch(head_name)
    return norm_w, head_w, eps, head_name


def load_activation_rows(shards_dir: str, layer: str, rows: int) -> np.ndarray:
    paths = sorted(glob.glob(os.path.join(shards_dir, "*.safetensors")))
    if not paths:
        raise SystemExit(f"no shards under {shards_dir}")
    chunks: list[np.ndarray] = []
    total = 0
    for path in paths:
        with safe_open(path, framework="numpy") as fh:
            acts = np.asarray(fh.get_tensor(layer), dtype=np.float64)
        chunks.append(acts)
        total += acts.shape[0]
        if total >= rows:
            break
    acts = np.concatenate(chunks, axis=0)[:rows]
    if acts.shape[0] < rows:
        raise SystemExit(f"cache holds only {acts.shape[0]} rows; asked for {rows}")
    return acts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--model-dir", required=True, help="HF snapshot dir with config.json")
    parser.add_argument("--layer", default="acts_L23")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument(
        "--top-v",
        type=int,
        default=64,
        help="output distribution width V: top V-1 global tokens + 1 tail bucket",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    acts = load_activation_rows(args.shards, args.layer, args.rows)
    norm_w, head_w, eps, head_name = load_head_tensors(args.model_dir)
    if acts.shape[1] != head_w.shape[1]:
        raise SystemExit(
            f"hidden mismatch: activations {acts.shape[1]}, head {head_w.shape[1]}"
        )

    rms = np.sqrt(np.mean(acts * acts, axis=1, keepdims=True) + eps)
    hidden = (acts / rms) * norm_w[None, :]

    n = hidden.shape[0]
    vocab = head_w.shape[0]
    mean_probs = np.zeros(vocab)
    # Two passes so the full n x vocab matrix never materializes.
    block = 256
    row_probs = []
    for start in range(0, n, block):
        logits = hidden[start : start + block] @ head_w.T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        mean_probs += probs.sum(axis=0)
        row_probs.append(probs.astype(np.float32))
    mean_probs /= n

    keep = np.argsort(mean_probs)[::-1][: args.top_v - 1]
    keep = np.sort(keep)
    out = np.zeros((n, args.top_v), dtype=np.float64)
    for idx, probs in enumerate(row_probs):
        start = idx * block
        chunk = probs.astype(np.float64)
        out[start : start + chunk.shape[0], : args.top_v - 1] = chunk[:, keep]
        out[start : start + chunk.shape[0], args.top_v - 1] = np.clip(
            1.0 - chunk[:, keep].sum(axis=1), 0.0, 1.0
        )
    # Exact renormalization (guards fp round-off; deviation is O(1e-15)).
    out /= out.sum(axis=1, keepdims=True)

    np.savez_compressed(
        args.out,
        probs=out,
        token_ids=keep.astype(np.int64),
        provenance=json.dumps(
            {
                "issue": 2015,
                "readout": "probe-head: softmax(lm_head(rms_norm(acts)))",
                "layer": args.layer,
                "rows": n,
                "top_v": args.top_v,
                "head_tensor": head_name,
                "rms_eps": eps,
                "tail_bucket": "last column = renormalized residual vocab mass",
            }
        ),
    )
    tail = out[:, -1]
    print(
        f"wrote {args.out}: probs {out.shape}, tail mass mean={tail.mean():.4f} "
        f"max={tail.max():.4f} (raise --top-v if the tail dominates)"
    )


if __name__ == "__main__":
    main()
