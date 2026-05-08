"""Train the stack-trace LM, then a TopK SAE on its mid-layer residual stream.

Pipeline:
  load stream.npz + vocab.json
  -> cut into LM training chunks
  -> train LM (next-token CE)
  -> sweep dataset to harvest layer-K residual-stream activations
  -> train TopK SAE on those activations
  -> save lm.pt, sae.pt, activations.npz

Designed to run in ~5-15 min on CPU/MPS.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import LMConfig, SAEConfig, StackLM, TopKSAE


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------- LM training ----------

def make_lm_batches(
    tokens: np.ndarray,
    seq_len: int,
    stride: int,
) -> torch.Tensor:
    """Cut into overlapping windows of length seq_len+1 (input + shifted target)."""
    windows = []
    n = tokens.size
    for i in range(0, n - seq_len - 1, stride):
        windows.append(tokens[i : i + seq_len + 1])
    if not windows:
        raise RuntimeError(f"token stream too short: {n} < {seq_len + 1}")
    return torch.from_numpy(np.stack(windows)).long()


def train_lm(
    cfg: LMConfig,
    tokens: np.ndarray,
    device: torch.device,
    *,
    seq_len: int = 256,
    batch_size: int = 32,
    steps: int = 3000,
    lr: float = 3e-4,
    log_every: int = 100,
) -> StackLM:
    print(f"[lm] config: {cfg}")
    model = StackLM(cfg).to(device)
    print(f"[lm] params: {model.n_params() / 1e6:.2f}M")

    windows = make_lm_batches(tokens, seq_len=seq_len, stride=seq_len // 2)
    print(f"[lm] training windows: {windows.shape}")
    ds = TensorDataset(windows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    step = 0
    t_start = time.time()
    losses: list[float] = []
    while step < steps:
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            if step % log_every == 0:
                recent = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
                ppl = float(np.exp(recent))
                elapsed = time.time() - t_start
                print(f"[lm] step {step}/{steps}  loss {recent:.3f}  ppl {ppl:.1f}  {elapsed:.0f}s")
            step += 1
            if step >= steps:
                break

    return model


# ---------- harvest activations ----------

@torch.no_grad()
def harvest_activations(
    model: StackLM,
    tokens: np.ndarray,
    device: torch.device,
    *,
    layer: int,
    seq_len: int,
    batch_size: int = 32,
    max_acts: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run forward pass over non-overlapping windows, collect post-block-`layer`
    residual-stream activations and the corresponding token at each position.
    Returns (acts, ctx_toks, win_idx, pos_idx) all length N."""
    model.eval()
    windows = make_lm_batches(tokens, seq_len=seq_len, stride=seq_len)  # non-overlapping
    n_total = windows.shape[0] * seq_len
    # Sub-sample windows uniformly across the corpus so harvested positions
    # span the full profile time (otherwise we only see the prefix).
    n_windows_needed = max(1, max_acts // seq_len)
    if windows.shape[0] > n_windows_needed:
        idx = torch.linspace(0, windows.shape[0] - 1, n_windows_needed).long()
        windows = windows[idx]
        print(f"[harvest] subsampled windows: {windows.shape[0]} (uniformly spaced)")
    print(f"[harvest] total positions available: {n_total}, capping at {max_acts}")
    acts_buf: list[np.ndarray] = []
    tok_buf: list[np.ndarray] = []
    pos_buf: list[np.ndarray] = []
    win_buf: list[np.ndarray] = []
    collected = 0
    n_wins = windows.shape[0]
    # Original-window indices for each retained window (used for token-offset lookup downstream).
    n_full = (tokens.size - seq_len - 1) // seq_len + 1
    if n_wins < n_full:
        win_orig_idx = np.linspace(0, n_full - 1, n_wins, dtype=np.int64)
    else:
        win_orig_idx = np.arange(n_wins, dtype=np.int64)
    for i in range(0, n_wins, batch_size):
        batch = windows[i : i + batch_size].to(device)
        inp = batch[:, :-1]
        _, captured = model(inp, capture_layer=layer)
        if captured is None:
            raise RuntimeError("layer not captured")
        # captured: (B, T, D)
        a = captured.detach().to("cpu", torch.float32).numpy()
        t = inp.detach().to("cpu").numpy().astype(np.int32)
        B, T, D = a.shape
        a = a.reshape(B * T, D)
        t = t.reshape(B * T)
        win_idx = np.repeat(win_orig_idx[i : i + B], T).astype(np.int32)
        pos_idx = np.tile(np.arange(T), B).astype(np.int32)
        acts_buf.append(a)
        tok_buf.append(t)
        win_buf.append(win_idx)
        pos_buf.append(pos_idx)
        collected += a.shape[0]
        if collected >= max_acts:
            break
    acts = np.concatenate(acts_buf, axis=0)[:max_acts]
    toks = np.concatenate(tok_buf, axis=0)[:max_acts]
    wins = np.concatenate(win_buf, axis=0)[:max_acts]
    pos = np.concatenate(pos_buf, axis=0)[:max_acts]
    return acts, toks, wins, pos


# ---------- SAE training ----------

def train_sae(
    acts: np.ndarray,
    cfg: SAEConfig,
    device: torch.device,
    *,
    batch_size: int = 1024,
    steps: int = 3000,
    lr: float = 1e-3,
    log_every: int = 100,
) -> TopKSAE:
    print(f"[sae] config: {cfg}, acts: {acts.shape}")
    sae = TopKSAE(cfg).to(device)
    acts_t = torch.from_numpy(acts).to(device)
    n = acts_t.shape[0]
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    sae.train()
    t_start = time.time()
    losses: list[float] = []
    var_x = ((acts_t - acts_t.mean(0)) ** 2).sum(-1).mean().item()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = acts_t[idx]
        recon, sparse, loss = sae(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        sae.normalize_decoder()
        # update firing stats periodically (cheap)
        if step % 5 == 0:
            with torch.no_grad():
                _, top_idx = sae.encode_topk(x)
                sae.update_firing_stats(top_idx)
        losses.append(loss.item())
        if step % log_every == 0:
            recent = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            frac_explained = 1.0 - recent / max(1e-9, var_x)
            n_dead = sae.dead_features().numel()
            elapsed = time.time() - t_start
            print(f"[sae] step {step}/{steps}  recon {recent:.3f}  expl {frac_explained:.3f}  "
                  f"dead {n_dead}/{cfg.dict_size}  {elapsed:.0f}s")
    return sae


# ---------- main ----------

def main(out_dir: Path):
    device = pick_device()
    print(f"device: {device}")

    stream = np.load(out_dir / "stream.npz")
    tokens = stream["tokens"]
    with open(out_dir / "vocab.json") as f:
        vocab = json.load(f)
    print(f"tokens: {tokens.size}, vocab: {len(vocab)}")

    seq_len = 256
    capture_layer = 1  # mid-stack on a 4-layer model
    # Auto-scale step counts to dataset size (small: cap fewer steps).
    n_chunks = max(1, tokens.size // seq_len)
    lm_steps = 1000  # converges by ~500 on this corpus; extra steps overfit/waste compute
    sae_steps = 3000

    lm_cfg = LMConfig(
        vocab_size=len(vocab),
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=seq_len,
    )
    lm = train_lm(lm_cfg, tokens, device, seq_len=seq_len, steps=lm_steps)
    torch.save({"state_dict": lm.state_dict(), "cfg": lm_cfg.__dict__}, out_dir / "lm.pt")
    print(f"[ok] saved LM to {out_dir / 'lm.pt'}")

    acts, ctx_toks, win_idx, pos_idx = harvest_activations(
        lm, tokens, device, layer=capture_layer, seq_len=seq_len, max_acts=80_000
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

    sae_cfg = SAEConfig(d_model=lm_cfg.d_model, dict_size=lm_cfg.d_model * 16, k=16)
    sae = train_sae(acts, sae_cfg, device, steps=sae_steps)
    torch.save({"state_dict": sae.state_dict(), "cfg": sae_cfg.__dict__}, out_dir / "sae.pt")
    print(f"[ok] saved SAE to {out_dir / 'sae.pt'}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tokenized")
    main(out)
