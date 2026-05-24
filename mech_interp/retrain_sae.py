"""Retrain the SAE with AuxK enabled (LM checkpoint reused).

Reads lm.pt + activations.npz; writes sae.pt (overwriting the existing one).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import LMConfig, SAEConfig, StackLM, TopKSAE
from train import pick_device, train_sae


def main(out_dir: Path):
    device = pick_device()
    print(f"device: {device}")

    blob = np.load(out_dir / "activations.npz")
    acts = blob["acts"]
    print(f"acts: {acts.shape}")

    lm_blob = torch.load(out_dir / "lm.pt", map_location="cpu", weights_only=False)
    lm_cfg = LMConfig(**lm_blob["cfg"])

    sae_cfg = SAEConfig(
        d_model=lm_cfg.d_model,
        dict_size=lm_cfg.d_model * 16,
        k=16,
        aux_k=64,
        aux_alpha=1.0 / 16,  # Anthropic recipe ≈ 1/16 of main
    )
    sae = train_sae(acts, sae_cfg, device, steps=4000)

    torch.save({"state_dict": sae.state_dict(), "cfg": sae_cfg.__dict__}, out_dir / "sae.pt")
    print(f"[ok] saved SAE with AuxK to {out_dir / 'sae.pt'}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/tok")
    main(out)
