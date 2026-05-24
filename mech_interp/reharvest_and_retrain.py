"""Re-harvest activations spanning the full profile, then retrain SAE.

The original harvest only sampled the first 80k tokens (= ~1% of profile
time). We need uniform coverage to study temporal structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from model import LMConfig, SAEConfig, StackLM
from train import harvest_activations, pick_device, train_sae


def main(out_dir: Path):
    device = pick_device()
    print(f"device: {device}")

    stream = np.load(out_dir / "stream.npz")
    tokens = stream["tokens"]

    lm_blob = torch.load(out_dir / "lm.pt", map_location="cpu", weights_only=False)
    lm_cfg = LMConfig(**lm_blob["cfg"])
    lm = StackLM(lm_cfg).to(device)
    lm.load_state_dict(lm_blob["state_dict"])

    seq_len = 256
    capture_layer = 1

    print("=== re-harvesting activations across full profile ===")
    acts, ctx_toks, win_idx, pos_idx = harvest_activations(
        lm, tokens, device, layer=capture_layer, seq_len=seq_len, max_acts=100_000
    )
    np.savez(
        out_dir / "activations.npz",
        acts=acts.astype(np.float32),
        ctx_toks=ctx_toks,
        win_idx=win_idx,
        pos_idx=pos_idx,
        capture_layer=np.int32(capture_layer),
        seq_len=np.int32(seq_len),
    )
    print(f"[ok] re-harvested: {acts.shape}, win_idx range [{win_idx.min()}, {win_idx.max()}]")

    print("=== retraining SAE on new activations ===")
    sae_cfg = SAEConfig(
        d_model=lm_cfg.d_model,
        dict_size=lm_cfg.d_model * 16,
        k=16,
        aux_k=64,
        aux_alpha=1.0 / 16,
    )
    sae = train_sae(acts, sae_cfg, device, steps=4000)
    torch.save({"state_dict": sae.state_dict(), "cfg": sae_cfg.__dict__}, out_dir / "sae.pt")
    print(f"[ok] saved SAE to {out_dir / 'sae.pt'}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
