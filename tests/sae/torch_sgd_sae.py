#!/usr/bin/env python3
"""A REAL overcomplete SGD-trained sparse autoencoder (PyTorch).

This is the genuine Stage-0 of the honest #977 division of labor: a standard
wide-dictionary SAE that gam does NOT provide. gam's manifold-SAE recovers a
small number of *curved* atoms with a forced topology; this module instead
trains a large overcomplete linear dictionary by SGD with an L1 (or JumpReLU)
sparsity penalty — exactly the mechanistic-interpretability SAE that produces a
wide bank of monosemantic-ish features. Stage-1 (gam) then ADJUDICATES the
geometry of feature groups WITHOUT forcing any topology.

Design choices (standard, defensible):
  - Tied-ish but independent encoder/decoder (W_enc, W_dec) with a decoder-norm
    constraint (unit-norm decoder columns) so the L1 penalty is a true sparsity
    pressure and not absorbable by shrinking decoder norms.
  - Pre-encoder bias subtraction (b_dec) so the SAE reconstructs x - b_dec, the
    standard Anthropic/Cunningham convention.
  - ReLU activation (L1 penalty) by default; optional JumpReLU (straight-through
    estimator) for a harder L0 target.
  - Dead-feature resampling (ghost-grad-free; simple reinit of long-dead units).

Reported metrics are the honest reconstruction story: explained variance (EV),
mean L0 (active features/token), and dead-feature fraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SAEConfig:
    d_in: int
    dict_size: int
    l1_coeff: float = 5e-3
    activation: str = "relu"          # "relu" | "jumprelu"
    jump_theta: float = 0.03          # JumpReLU threshold (only if activation=="jumprelu")
    lr: float = 4e-4
    batch_size: int = 4096
    epochs: int = 8
    dead_steps: int = 2000            # a feature dead this many steps -> resample
    resample_period: int = 4000
    warmup_l1_steps: int = 1000       # ramp L1 from 0 to full over this many steps
    seed: int = 0
    device: str = "cuda"


class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        # Kaiming-ish encoder; decoder initialized as encoder transpose then
        # unit-normalized (standard SAE init).
        W_dec = torch.randn(cfg.dict_size, cfg.d_in, generator=g)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)                       # [F, d]
        self.W_enc = nn.Parameter(W_dec.clone().t().contiguous())  # [d, F]
        self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.cfg.activation == "jumprelu":
            # JumpReLU with a straight-through estimator: forward gates at theta,
            # backward uses the ReLU subgradient (standard STE).
            theta = self.cfg.jump_theta
            gated = pre * (pre > theta).float()
            relu = torch.relu(pre)
            return relu + (gated - relu).detach()
        return torch.relu(pre)

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        a = self.encode(x)
        xhat = self.decode(a)
        return xhat, a

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


def _explained_variance(x: torch.Tensor, xhat: torch.Tensor) -> float:
    resid = (x - xhat).pow(2).sum().item()
    total = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().item()
    return 1.0 - resid / max(total, 1e-12)


def train_sae(acts: np.ndarray, cfg: SAEConfig, log=print):
    """Train the SAE on `acts` ([N, d] float). Returns (model, stats_dict).

    `acts` is held in CPU RAM (mmap-friendly); minibatches are moved to GPU.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    N, d = acts.shape
    assert d == cfg.d_in, f"d_in {cfg.d_in} != activation dim {d}"

    # Dataset normalization (Anthropic convention): scale so the mean squared
    # token norm equals d_in. This calibrates the L1 coefficient to the standard
    # 1e-3..1e-2 regime regardless of the raw activation scale (without it the L1
    # penalty is negligible against the d-summed recon term). The scale is a pure
    # scalar, so the recovered geometry is unchanged; EV is scale-invariant.
    centered = acts - acts.mean(axis=0, keepdims=True)
    mean_sq_norm = float((centered ** 2).sum(axis=1).mean())
    norm_scale = math.sqrt(d / max(mean_sq_norm, 1e-12))
    log(f"  dataset norm: mean||x||^2={mean_sq_norm:.3g}, scaling by {norm_scale:.4g} "
        f"-> target mean||x||^2={d}")

    model = SparseAutoencoder(cfg).to(device)
    # Initialize b_dec at the (scaled) data mean (standard).
    with torch.no_grad():
        mean = torch.from_numpy(acts.mean(axis=0) * norm_scale).float().to(device)
        model.b_dec.copy_(mean)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    acts_t = torch.from_numpy(acts)  # CPU, float32; pinned for fast H2D if possible
    if acts_t.dtype != torch.float32:
        acts_t = acts_t.float()
    acts_t = acts_t * norm_scale
    try:
        acts_t = acts_t.pin_memory()
    except Exception:
        pass

    steps_per_epoch = math.ceil(N / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    # Cap the L1 warmup at a quarter of training so short runs still reach full
    # sparsity pressure (a fixed 1000-step warmup would never complete on a
    # small/short run, leaving the L1 penalty effectively off).
    warmup = min(cfg.warmup_l1_steps, max(1, total_steps // 4))
    rng = np.random.default_rng(cfg.seed)

    last_active = torch.zeros(cfg.dict_size, dtype=torch.long, device=device)
    step = 0
    for epoch in range(cfg.epochs):
        perm = rng.permutation(N)
        for bi in range(steps_per_epoch):
            idx = perm[bi * cfg.batch_size : (bi + 1) * cfg.batch_size]
            x = acts_t[idx].to(device, non_blocking=True)
            xhat, a = model(x)
            recon = (xhat - x).pow(2).sum(dim=1).mean()
            l1_w = cfg.l1_coeff * min(1.0, (step + 1) / warmup)
            # L1 on activations weighted by decoder column norm (=1 after norm).
            sparsity = a.abs().sum(dim=1).mean()
            loss = recon + l1_w * sparsity
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.normalize_decoder()

            with torch.no_grad():
                active = (a > 0).any(dim=0)
                last_active[active] = step
            step += 1

            if step % cfg.resample_period == 0:
                dead = (step - last_active) > cfg.dead_steps
                ndead = int(dead.sum().item())
                if ndead > 0:
                    _resample_dead(model, dead, acts_t, device, rng)
                    last_active[dead] = step

            if step % 200 == 0 or step == total_steps - 1:
                with torch.no_grad():
                    ev = 1.0 - (xhat - x).pow(2).sum().item() / max(
                        (x - x.mean(0, keepdim=True)).pow(2).sum().item(), 1e-12)
                    l0 = (a > 0).float().sum(dim=1).mean().item()
                log(f"  step {step:6d}/{total_steps} ep{epoch} "
                    f"recon={recon.item():.4f} l0={l0:.1f} ev~{ev:.4f} l1w={l1_w:.1e}")

    # Final full-data eval (chunked).
    model.eval()
    ev, l0, dead_frac, feat_act_count = _full_eval(model, acts_t, device, cfg.batch_size, cfg.dict_size)
    stats = {
        "explained_variance": ev,
        "mean_l0": l0,
        "dead_feature_fraction": dead_frac,
        "n_tokens": int(N),
        "dict_size": int(cfg.dict_size),
        "d_in": int(d),
        "norm_scale": norm_scale,
    }
    log(f"FINAL: EV={ev:.4f} mean_L0={l0:.1f} dead_frac={dead_frac:.3f}")
    return model, stats, feat_act_count


@torch.no_grad()
def _resample_dead(model, dead_mask, acts_t, device, rng):
    """Reinitialize dead features toward poorly-reconstructed inputs (standard
    Anthropic resampling, simplified: sample random inputs, point the dead
    decoder/encoder rows at them)."""
    dead_idx = torch.nonzero(dead_mask, as_tuple=False).flatten()
    n = dead_idx.numel()
    if n == 0:
        return
    samp = rng.choice(acts_t.shape[0], size=int(n), replace=False)
    x = acts_t[samp].to(device).float()
    x = x - model.b_dec
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    model.W_dec[dead_idx] = x
    model.W_enc[:, dead_idx] = (x * 0.2).t()
    model.b_enc[dead_idx] = 0.0


@torch.no_grad()
def _full_eval(model, acts_t, device, bs, F):
    N = acts_t.shape[0]
    resid = 0.0
    total = 0.0
    l0_sum = 0.0
    act_count = torch.zeros(F, dtype=torch.long, device=device)
    mean = acts_t.mean(dim=0).to(device)
    for s in range(0, N, bs):
        x = acts_t[s : s + bs].to(device).float()
        xhat, a = model(x)
        resid += (xhat - x).pow(2).sum().item()
        total += (x - mean).pow(2).sum().item()
        l0_sum += (a > 0).float().sum().item()
        act_count += (a > 0).sum(dim=0)
    ev = 1.0 - resid / max(total, 1e-12)
    l0 = l0_sum / N
    dead_frac = float((act_count == 0).float().mean().item())
    return ev, l0, dead_frac, act_count.cpu().numpy()
