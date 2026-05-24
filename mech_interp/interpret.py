"""Find what each SAE feature represents.

For each (live) feature:
  - find the top-N activations across the harvested set
  - for each top activation, decode the surrounding context (last K tokens)
  - print a one-line summary: feature_id, max_activation, top frame names by
    weighted contribution, sample contexts.

Output: features.txt (human-readable) + features.json (machine-readable).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import LMConfig, SAEConfig, StackLM, TopKSAE


def main(out_dir: Path, n_features: int = 64, top_k: int = 8, ctx_window: int = 24):
    with open(out_dir / "vocab.json") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    lm_blob = torch.load(out_dir / "lm.pt", map_location="cpu", weights_only=False)
    lm_cfg = LMConfig(**lm_blob["cfg"])
    lm = StackLM(lm_cfg)
    lm.load_state_dict(lm_blob["state_dict"])
    lm.eval()

    sae_blob = torch.load(out_dir / "sae.pt", map_location="cpu", weights_only=False)
    sae_cfg = SAEConfig(**sae_blob["cfg"])
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(sae_blob["state_dict"], strict=False)
    sae.eval()

    blob = np.load(out_dir / "activations.npz")
    acts = torch.from_numpy(blob["acts"])
    ctx_toks = blob["ctx_toks"]
    win_idx = blob["win_idx"]
    pos_idx = blob["pos_idx"]
    seq_len = int(blob["seq_len"])

    stream = np.load(out_dir / "stream.npz")
    tokens_full = stream["tokens"]

    # Recompute LM windows (must match training stride=seq_len, non-overlapping).
    n_full = tokens_full.size
    win_starts = np.arange(0, n_full - seq_len - 1, seq_len, dtype=np.int64)

    def decode_context(win_id: int, pos_id: int) -> list[str]:
        # Position pos_id is in the input (length seq_len) of window win_id.
        # We want the actual stream tokens around this position.
        start = int(win_starts[win_id])
        center = start + int(pos_id)
        lo = max(0, center - ctx_window + 1)
        hi = center + 1
        return [inv_vocab.get(int(t), f"<id:{int(t)}>") for t in tokens_full[lo:hi]]

    print(f"computing feature activations on {acts.shape[0]} positions...")
    with torch.no_grad():
        # Encode all activations through SAE encoder, KEEPING all features
        # (we want top-activating examples per feature, not topk-per-position).
        z = (acts - sae.b_dec) @ sae.W_enc.T + sae.b_enc
        z = torch.relu(z).numpy()  # (N, dict_size)

    # Per-feature: max activation, fraction-of-positions firing.
    max_acts = z.max(axis=0)               # (dict_size,)
    frac_fire = (z > 0).mean(axis=0)        # (dict_size,)
    n_alive = int((max_acts > 1e-6).sum())
    print(f"alive features: {n_alive}/{sae_cfg.dict_size}")

    # Pick the most informative features: not dead, not too dense.
    # Score = max_act * log(1 / frac_fire) — favors strong + selective features.
    safe_frac = np.maximum(frac_fire, 1.0 / z.shape[0])
    score = max_acts * np.log(1.0 / safe_frac)
    score[max_acts < 1e-6] = -np.inf
    top_features = np.argsort(-score)[:n_features]

    out_lines: list[str] = []
    out_json: list[dict] = []
    for fid in top_features:
        col = z[:, fid]
        # Top-K positions for this feature
        top_idx = np.argsort(-col)[:top_k]
        examples = []
        all_frame_counts: dict[str, float] = {}
        for ti in top_idx:
            act = float(col[ti])
            ctx = decode_context(int(win_idx[ti]), int(pos_idx[ti]))
            examples.append({"activation": act, "context": ctx})
            for frame in ctx:
                if frame.startswith("<"):
                    continue
                all_frame_counts[frame] = all_frame_counts.get(frame, 0.0) + act
        top_frames = sorted(all_frame_counts.items(), key=lambda kv: -kv[1])[:6]
        out_json.append({
            "feature_id": int(fid),
            "max_activation": float(max_acts[fid]),
            "fraction_firing": float(frac_fire[fid]),
            "top_frames": top_frames,
            "examples": examples,
        })
        ctx_token_at_max = inv_vocab.get(int(ctx_toks[top_idx[0]]), "?")
        frame_blurb = ", ".join(f"{n}({w:.1f})" for n, w in top_frames[:3])
        out_lines.append(
            f"feat {int(fid):4d}  max {max_acts[fid]:6.2f}  fire {frac_fire[fid]*100:5.2f}%  "
            f"@{ctx_token_at_max:<40}  → {frame_blurb}"
        )

    summary = "\n".join(out_lines)
    print()
    print(summary)
    (out_dir / "features.txt").write_text(summary + "\n")
    (out_dir / "features.json").write_text(json.dumps(out_json, indent=1))
    print(f"\n[ok] wrote {n_features} features → {out_dir / 'features.txt'}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tokenized")
    main(out)
