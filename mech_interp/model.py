"""Tiny decoder-only transformer LM over interleaved frame+dwell tokens,
plus a TopK sparse autoencoder.

LM is intentionally small: ~5-10M params. Trains in minutes on MPS / CPU.
SAE follows the Anthropic 2024 TopK recipe (no L1, no auxiliary loss
beyond a tiny dead-feature revival term).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- transformer LM ----------

@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # B,H,T,Dh
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # SDPA picks an efficient backend (incl. flash on supported GPUs).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (post-block residual, residual stream after attn+mlp added).
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StackLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Tied output head: weight shared with tok_emb.
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # Sensible init.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,        # (B, T) int64
        capture_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = tokens.shape
        assert T <= self.cfg.max_seq_len
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        captured: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if capture_layer is not None and i == capture_layer:
                captured = x  # post-block residual stream
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, captured

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------- TopK sparse autoencoder ----------

@dataclass
class SAEConfig:
    d_model: int
    dict_size: int
    k: int = 16
    dead_feature_threshold_steps: int = 200  # steps without firing => "dead"
    aux_k: int = 64       # # of "next-best dead" features to revive each step
    aux_alpha: float = 0.0625  # auxiliary loss weight


class TopKSAE(nn.Module):
    """y_hat = b_dec + W_dec @ topk_k(W_enc @ (x - b_dec) + b_enc)

    Main training loss: ||x - y_hat||^2 (sparsity enforced by TopK).
    AuxK auxiliary loss (Anthropic 2024 §A.2): once a feature has been "dead"
    for N steps, it can win the auxiliary topk-among-deads competition and
    contribute to reconstructing the *residual* (x - y_hat). This drives the
    decoder direction of dead features toward useful directions and revives
    them. Without it ~80% of features die on small / repetitive corpora.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.empty(cfg.dict_size, cfg.d_model))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_model, cfg.dict_size))
        self.b_enc = nn.Parameter(torch.zeros(cfg.dict_size))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

        # Anthropic-style init: W_dec = W_enc.T, both column-normalized.
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-6)

        # Steps-since-last-fire tracker (for dead-feature detection / revival).
        self.register_buffer(
            "last_fired",
            torch.zeros(cfg.dict_size, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)

    def encode_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (..., d_model)
        z = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        z = F.relu(z)
        # TopK along last dim.
        top_vals, top_idx = z.topk(self.cfg.k, dim=-1)
        # Scatter back to a sparse-ish dense tensor.
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, top_idx, top_vals)
        return sparse, top_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse, top_idx = self.encode_topk(x)
        recon = sparse @ self.W_dec.T + self.b_dec
        # Main recon loss (MSE summed over features, mean over batch).
        main_loss = ((recon - x) ** 2).sum(dim=-1).mean()

        # AuxK: among DEAD features, pick the topk_aux that would reconstruct
        # the current residual best, scale them, decode, MSE-loss the result
        # against the residual. This pulls dead features toward directions
        # that the main code can't already explain.
        if self.cfg.aux_alpha > 0 and self.cfg.aux_k > 0:
            dead = self.dead_features()
            if dead.numel() >= self.cfg.aux_k:
                with torch.no_grad():
                    dead_mask = torch.zeros(self.cfg.dict_size, dtype=torch.bool, device=x.device)
                    dead_mask[dead] = True
                # pre-topk encoder activations
                z = (x - self.b_dec) @ self.W_enc.T + self.b_enc
                z = F.relu(z)
                z = z.masked_fill(~dead_mask[None, :], 0.0)
                k_aux = min(self.cfg.aux_k, int(dead_mask.sum().item()))
                top_aux_vals, top_aux_idx = z.topk(k_aux, dim=-1)
                aux_sparse = torch.zeros_like(z)
                aux_sparse.scatter_(-1, top_aux_idx, top_aux_vals)
                aux_recon = aux_sparse @ self.W_dec.T  # no b_dec — we model the residual
                residual = (x - recon).detach()
                aux_loss = ((aux_recon - residual) ** 2).sum(dim=-1).mean()
                loss = main_loss + self.cfg.aux_alpha * aux_loss
            else:
                loss = main_loss
        else:
            loss = main_loss
        return recon, sparse, loss

    def update_firing_stats(self, top_idx: torch.Tensor):
        # top_idx: (..., k); mark all of those as fired this step.
        self.step += 1
        flat = top_idx.reshape(-1)
        # last_fired = step for fired features; others unchanged.
        self.last_fired.index_fill_(0, flat.unique(), int(self.step.item()))

    @torch.no_grad()
    def normalize_decoder(self):
        """Project W_dec columns to unit norm (after each grad step)."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.W_dec.div_(norms)

    def dead_features(self) -> torch.Tensor:
        thresh = int(self.step.item()) - self.cfg.dead_feature_threshold_steps
        return (self.last_fired < max(0, thresh)).nonzero(as_tuple=True)[0]
