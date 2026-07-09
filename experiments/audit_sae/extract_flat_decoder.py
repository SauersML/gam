#!/usr/bin/env python3
"""Turnkey a flat K x P decoder.npy for the #1942 frozen-dictionary audit arm.

`gamfit.audit_sae` audits a *linear* dictionary: it takes a `K x P` decoder
(rows = atoms, cols = residual dims), the `N x P` activations, and (optionally)
`N x K` external codes. This script produces that `decoder.npy` from whichever
input actually exists, so `saebench_full_pass.sbatch --decoder ... --activations`
has something real to point at:

  * ``--checkpoint FILE`` — pull ``W_dec`` out of an existing flat SAE checkpoint
    (.safetensors / .npz / .pt). Orientation is normalised to ``K x P`` from the
    tensor shape; nothing is assumed about which axis is K.

  * ``--fit-topk`` — no team checkpoint on hand? Train a standard Gao-et-al. TopK
    SAE (tied init, unit-norm decoder rows, pre-bias) directly on a chunk_*.npy
    activation directory, mirroring experiments/1026_close/driver_1026_arms.py's
    ``fit_external_topk`` so the dictionary is the SAME "traditional SAE" bar the
    #1026 close campaign uses. Writes both ``decoder.npy`` (K x P) AND a reusable
    ``<out>.safetensors`` checkpoint (W_enc/W_dec/b_dec/k) that the public
    SAEBench suite's flat arm can consume unchanged.

Held-out EV of the fitted TopK dictionary is printed (and stored in the
checkpoint's sidecar json) so a ``--fit-topk`` run doubles as the #1026
external_topk data point — one job, two numbers, the shared-infra leverage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Checkpoint -> decoder.npy
# --------------------------------------------------------------------------- #
def _load_tensors(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.numpy import load_file

        return dict(load_file(str(path)))
    if suffix == ".npz":
        with np.load(path) as archive:
            return {k: archive[k] for k in archive.files}
    if suffix in (".pt", ".pth"):
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        out: dict[str, np.ndarray] = {}
        for k, v in dict(state).items():
            try:
                out[str(k)] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            except (TypeError, ValueError):
                continue  # non-tensor entries (e.g. a numpy mean vector) are kept below
            if not isinstance(out[str(k)], np.ndarray):
                del out[str(k)]
        return out
    raise SystemExit(f"unsupported checkpoint {suffix!r}; expected .safetensors/.npz/.pt")


def decoder_from_checkpoint(path: Path, *, p_hint: int | None, renormalize_rows: bool) -> np.ndarray:
    """Extract a ``K x P`` decoder from a flat SAE checkpoint.

    ``W_dec`` is stored either ``(K, P)`` (our #1026 convention) or ``(P, K)``.
    We orient to ``K x P`` using ``p_hint`` (the residual dim) when the matrix is
    square-ambiguous, else by matching the activation-dim axis to the other
    tensors. ``--renormalize-rows`` restores unit-norm atom rows (the trained
    invariant) if the checkpoint drifted."""
    tensors = _load_tensors(path)
    if "W_dec" not in tensors:
        raise SystemExit(
            f"{path} has no 'W_dec' (keys: {sorted(tensors)}); pass a flat SAE checkpoint"
        )
    w_dec = np.ascontiguousarray(np.asarray(tensors["W_dec"], dtype=np.float64))
    if w_dec.ndim != 2:
        raise SystemExit(f"W_dec must be 2-D, got shape {w_dec.shape}")
    rows, cols = w_dec.shape
    # Decide which axis is P (residual dim). Prefer an explicit hint; else infer
    # from b_dec (length P) or W_enc; else assume rows=K, cols=P (#1026 layout).
    p_dim = p_hint
    if p_dim is None and "b_dec" in tensors:
        p_dim = int(np.asarray(tensors["b_dec"]).reshape(-1).shape[0])
    if p_dim is None and "b_enc" in tensors:
        k_dim = int(np.asarray(tensors["b_enc"]).reshape(-1).shape[0])
        p_dim = cols if rows == k_dim else (rows if cols == k_dim else None)
    if p_dim is not None and p_dim == rows and p_dim != cols:
        w_dec = np.ascontiguousarray(w_dec.T)  # was (P, K) -> (K, P)
    elif p_dim is not None and p_dim not in (rows, cols):
        raise SystemExit(f"W_dec shape {(rows, cols)} matches neither K nor P={p_dim}")
    if renormalize_rows:
        norms = np.linalg.norm(w_dec, axis=1, keepdims=True)
        w_dec = w_dec / np.clip(norms, 1e-8, None)
    return np.ascontiguousarray(w_dec, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Fit a TopK SAE on a chunk dir (mirrors #1026 external_topk) -> decoder.npy + ckpt
# --------------------------------------------------------------------------- #
def _load_chunk_dir(chunk_dir: Path, max_rows: int, seed: int) -> np.ndarray:
    """mmap chunk_*.npy shards, deterministic max_rows subsample. Identical
    semantics to experiments/1026_close/driver_1026_arms.py:load_chunk_dir so the
    split (and hence the number) agrees with the #1026 harness."""
    files = sorted(chunk_dir.glob("chunk_*.npy"))
    if not files:
        raise SystemExit(f"no chunk_*.npy shards in {chunk_dir}")
    mms = [np.load(f, mmap_mode="r") for f in files]
    p = int(mms[0].shape[1])
    for m in mms:
        if int(m.shape[1]) != p:
            raise SystemExit(f"inconsistent p across shards: {p} vs {m.shape[1]}")
    counts = [int(m.shape[0]) for m in mms]
    offsets = np.cumsum([0] + counts)
    rng = np.random.default_rng(seed)
    take = min(max_rows, int(offsets[-1]))
    idx = np.sort(rng.choice(int(offsets[-1]), take, replace=False))
    parts = []
    for c, m in enumerate(mms):
        sel = idx[(idx >= offsets[c]) & (idx < offsets[c + 1])] - offsets[c]
        if sel.size:
            parts.append(np.asarray(m[sel], dtype=np.float32))
    out = np.concatenate(parts, axis=0)
    rng.shuffle(out)
    return np.ascontiguousarray(out, dtype=np.float32)


def _split(x: np.ndarray, test_frac: float, seed: int):
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(x.shape[0])
    n_test = max(1, int(round(test_frac * x.shape[0])))
    return np.ascontiguousarray(x[perm[n_test:]]), np.ascontiguousarray(x[perm[:n_test]])


def fit_topk_decoder(
    x_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    K: int,
    top_k: int,
    steps: int,
    lr: float,
    bs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Standard TopK SAE trainer, verbatim refinements from #1026 external_topk.

    Returns ``(W_enc (K,P), W_dec (K,P), b_dec (P,), held_out_ev)``. Decoder rows
    are unit-norm (the trained invariant) — exactly the K x P dictionary
    ``gamfit.audit_sae`` expects."""
    import torch

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = x_tr.shape[1]
    xtr = torch.from_numpy(x_tr).to(dev)
    xte = torch.from_numpy(x_te).to(dev)
    b_dec = torch.nn.Parameter(xtr.mean(0).clone())
    W_enc = torch.nn.Parameter(torch.randn(K, p, device=dev) / p ** 0.5)
    W_dec = torch.nn.Parameter(W_enc.detach().clone())
    with torch.no_grad():
        W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    opt = torch.optim.Adam([W_enc, W_dec, b_dec], lr=lr)

    def encode(x):
        pre = (x - b_dec) @ W_enc.t()
        topv, topi = pre.topk(top_k, dim=1)
        return torch.relu(topv), topi

    def decode(vals, idx):
        return torch.einsum("bk,bkp->bp", vals, W_dec[idx]) + b_dec

    n = xtr.shape[0]
    t0 = time.perf_counter()
    for step in range(steps):
        i = torch.randint(0, n, (min(bs, n),), device=dev)
        xb = xtr[i]
        vals, idx = encode(xb)
        loss = ((decode(vals, idx) - xb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if step % 500 == 0 or step == steps - 1:
            print(f"[fit-topk] step {step+1}/{steps} loss={float(loss):.5f} "
                  f"({(step+1)/max(time.perf_counter()-t0,1e-9):.1f} steps/s)", flush=True)
    with torch.no_grad():
        outs = []
        for s in range(0, xte.shape[0], 8192):
            vals, idx = encode(xte[s:s + 8192])
            outs.append(decode(vals, idx).float().cpu().numpy())
        recon = np.concatenate(outs, 0)
        mean_tr = x_tr.astype(np.float64).mean(0)
        ssr = float(np.sum((x_te.astype(np.float64) - recon.astype(np.float64)) ** 2))
        sst = float(np.sum((x_te.astype(np.float64) - mean_tr[None, :]) ** 2))
        ev = 1.0 - ssr / max(sst, 1e-300)
        return (
            W_enc.detach().cpu().numpy(),
            W_dec.detach().cpu().numpy(),
            b_dec.detach().cpu().numpy(),
            ev,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="output decoder.npy (K x P, f32)")
    # Mode 1: extract from an existing checkpoint.
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="flat SAE checkpoint (.safetensors/.npz/.pt) to pull W_dec from")
    ap.add_argument("--p-dim", type=int, default=None, help="residual dim P (disambiguates a square W_dec)")
    ap.add_argument("--renormalize-rows", action="store_true",
                    help="restore unit-norm atom rows (the trained invariant)")
    # Mode 2: fit a TopK dictionary on a chunk dir.
    ap.add_argument("--fit-topk", action="store_true", help="train a TopK SAE instead of loading a checkpoint")
    ap.add_argument("--chunk-dir", type=Path, default=None, help="dir of chunk_*.npy activation shards")
    ap.add_argument("--K", type=int, default=32768, help="dictionary size")
    ap.add_argument("--top-k", type=int, default=32, help="per-token active budget")
    ap.add_argument("--rows", type=int, default=120000, help="max activation rows to load")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.fit_topk:
        if args.chunk_dir is None:
            raise SystemExit("--fit-topk requires --chunk-dir")
        X = _load_chunk_dir(args.chunk_dir, args.rows, args.seed)
        x_tr, x_te = _split(X, args.test_frac, args.seed)
        print(f"[fit-topk] X={X.shape} train={x_tr.shape} test={x_te.shape} "
              f"K={args.K} top_k={args.top_k}", flush=True)
        w_enc, w_dec, b_dec, ev = fit_topk_decoder(
            x_tr, x_te, K=args.K, top_k=args.top_k, steps=args.steps,
            lr=args.lr, bs=args.bs, seed=args.seed,
        )
        np.save(args.out, np.ascontiguousarray(w_dec, dtype=np.float32))
        # Reusable flat checkpoint for the public suite's TopK arm.
        ckpt = args.out.with_suffix(".safetensors")
        try:
            from safetensors.numpy import save_file

            save_file(
                {
                    "W_enc": np.ascontiguousarray(w_enc, dtype=np.float32),
                    "W_dec": np.ascontiguousarray(w_dec, dtype=np.float32),
                    "b_dec": np.ascontiguousarray(b_dec, dtype=np.float32),
                    "k": np.asarray([args.top_k], dtype=np.int64),
                },
                str(ckpt),
            )
            ckpt_written = str(ckpt)
        except Exception as exc:  # noqa: BLE001
            ckpt_written = f"(not written: {type(exc).__name__}: {exc})"
        sidecar = args.out.with_suffix(".meta.json")
        sidecar.write_text(json.dumps({
            "source": "fit-topk", "chunk_dir": str(args.chunk_dir),
            "K": args.K, "top_k": args.top_k, "rows": int(X.shape[0]), "P": int(X.shape[1]),
            "held_out_ev": ev, "seed": args.seed, "checkpoint": ckpt_written,
        }, indent=2) + "\n")
        print(f"[fit-topk] held_out_ev={ev:.4f}  decoder={args.out}  ckpt={ckpt_written}", flush=True)
        return

    if args.checkpoint is None:
        raise SystemExit("supply either --checkpoint FILE or --fit-topk --chunk-dir DIR")
    w_dec = decoder_from_checkpoint(
        args.checkpoint, p_hint=args.p_dim, renormalize_rows=args.renormalize_rows
    )
    np.save(args.out, w_dec)
    print(f"[extract] {args.checkpoint} -> {args.out}  decoder K x P = {w_dec.shape}")


if __name__ == "__main__":
    main()
