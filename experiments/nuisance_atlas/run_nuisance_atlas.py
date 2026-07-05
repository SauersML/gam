#!/usr/bin/env python3
"""Nuisance-atlas pre-pass on real residual-stream activations.

Regress the KNOWN nuisance manifolds out of a residual-stream harvest before any
dictionary / coordinate charting, and report the fraction of activation variance
each nuisance block absorbs:

  * positional / rotary helix  -- RoPE paints a smooth helical manifold across
    the residual stream keyed on within-sequence token position. Position is
    recoverable from the harvest WITHOUT any extra metadata: rows are stored in
    token order, `seq_len - 1` rows per sequence (position 0 dropped at harvest),
    so global row `r` has within-sequence position `(r mod (seq_len-1)) + 1`.
    Design = Fourier features at RoPE-style geometric frequencies.
  * token-frequency direction -- unigram (log-)frequency is a nuisance axis.
    Needs per-row token ids (`--token-ids <npy>`, order-matched to the harvest);
    log-frequency is the empirical unigram log-probability of each row's token.

This is the experiment harness for the Rust core `gam_sae::nuisance_atlas`
(closed-form OLS regress-out); the math here mirrors it (single streaming pass:
accumulate G=ZᵀZ, C=ZᵀX, Σx, Σx², then SS_res = Σx² − 2·diag(BᵀC) + diag(BᵀGB)
in closed form — no second pass over the activations).

Usage:
  # positional-only on the OLMo-2-7B L18 harvest (no extra metadata needed):
  python run_nuisance_atlas.py --harvest /scratch/.../resid_olmo2_7b_l18_dir

  # add the token-frequency block with order-matched token ids:
  python run_nuisance_atlas.py --harvest <dir> --token-ids <ids.npy>

  # verify the math locally with no cluster data:
  python run_nuisance_atlas.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_ROPE_BASE = 10000.0


def positional_fourier_design(positions: np.ndarray, harmonics: int, base: float) -> np.ndarray:
    """(N, 2H) Fourier features cos/sin(theta_k * pos), theta_k = base**(-k/H)."""
    n = positions.shape[0]
    if harmonics == 0:
        return np.zeros((n, 0), dtype=np.float64)
    pos = positions.astype(np.float64)
    cols = []
    for k in range(harmonics):
        theta = base ** (-(k) / harmonics)
        ang = theta * pos
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    return np.stack(cols, axis=1)


def token_frequency_design(log_freq: np.ndarray, degree: int) -> np.ndarray:
    """(N, degree) standardized-log-frequency polynomial (powers 1..degree)."""
    n = log_freq.shape[0]
    if degree == 0:
        return np.zeros((n, 0), dtype=np.float64)
    lf = log_freq.astype(np.float64)
    mean = lf.mean()
    sd = lf.std()
    if sd <= 0.0:
        return np.zeros((n, degree), dtype=np.float64)
    z = (lf - mean) / sd
    return np.stack([z ** (d + 1) for d in range(degree)], axis=1)


def log_unigram_from_ids(token_ids: np.ndarray) -> np.ndarray:
    """Empirical per-row unigram log-probability from order-matched token ids."""
    ids = token_ids.astype(np.int64)
    counts = np.bincount(ids)
    total = ids.shape[0]
    logp = np.log(counts.astype(np.float64) / total + 1e-12)
    return logp[ids]


class _Accum:
    """Streaming ZᵀZ, ZᵀX, Σx, Σx², row count for a fixed design width M."""

    def __init__(self, m: int, d: int) -> None:
        self.g = np.zeros((m, m), dtype=np.float64)
        self.c = np.zeros((m, d), dtype=np.float64)
        self.sx = np.zeros(d, dtype=np.float64)
        self.sx2 = np.zeros(d, dtype=np.float64)
        self.n = 0

    def update(self, z: np.ndarray, x: np.ndarray) -> None:
        xf = x.astype(np.float64)
        self.g += z.T @ z
        self.c += z.T @ xf
        self.sx += xf.sum(0)
        self.sx2 += (xf * xf).sum(0)
        self.n += x.shape[0]


def _absorbed_from_accum(acc: _Accum, cols: list[int]) -> float:
    """Centred aggregate R² of the sub-design on `cols` (must include intercept)."""
    g = acc.g[np.ix_(cols, cols)]
    c = acc.c[cols, :]
    b = np.linalg.pinv(g) @ c  # (|cols|, d)
    ss_res = acc.sx2 - 2.0 * np.einsum("kj,kj->j", b, c) + np.einsum("kj,kj->j", b, g @ b)
    ss_tot = acc.sx2 - acc.sx * acc.sx / max(acc.n, 1)
    tot = float(ss_tot.sum())
    res = float(ss_res.sum())
    return 0.0 if tot <= 0.0 else 1.0 - res / tot


def run(
    activation_batches,
    positions: np.ndarray,
    d_model: int,
    harmonics: int,
    base: float,
    log_freq: np.ndarray | None,
    freq_degree: int,
    batch_rows: int,
) -> dict:
    """Single streaming pass; returns absorbed fractions for each nuisance block."""
    pos_design = positional_fourier_design(positions, harmonics, base)
    n = positions.shape[0]
    if log_freq is not None and freq_degree > 0:
        freq_design = token_frequency_design(log_freq, freq_degree)
    else:
        freq_design = np.zeros((n, 0), dtype=np.float64)
    # Full design: [intercept | positional | frequency].
    intercept = np.ones((n, 1), dtype=np.float64)
    design = np.concatenate([intercept, pos_design, freq_design], axis=1)
    m = design.shape[1]
    acc = _Accum(m, d_model)

    seen = 0
    for x in activation_batches:
        rows = x.shape[0]
        z = design[seen : seen + rows]
        acc.update(z, x)
        seen += rows
    if seen != n:
        raise SystemExit(f"row mismatch: consumed {seen} activation rows but {n} positions")

    n_pos = pos_design.shape[1]
    n_freq = freq_design.shape[1]
    pos_cols = [0] + list(range(1, 1 + n_pos))
    freq_cols = [0] + list(range(1 + n_pos, 1 + n_pos + n_freq))
    full_cols = list(range(m))
    out = {
        "n_rows": int(n),
        "d_model": int(d_model),
        "n_design": int(m),
        "positional_harmonics": int(harmonics),
        "rope_base": float(base),
        "freq_degree": int(n_freq),
        "positional_absorbed": _absorbed_from_accum(acc, pos_cols) if n_pos else 0.0,
        "frequency_absorbed": _absorbed_from_accum(acc, freq_cols) if n_freq else None,
        "combined_absorbed": _absorbed_from_accum(acc, full_cols),
    }
    return out


def _selftest() -> None:
    rng = np.random.default_rng(0)
    n, d = 8000, 32
    seq = 256
    base = DEFAULT_ROPE_BASE
    positions = (np.arange(n) % (seq - 1)) + 1
    # planted axes
    d_pos = rng.standard_normal(d)
    d_pos /= np.linalg.norm(d_pos)
    d_freq = rng.standard_normal(d)
    d_freq -= (d_freq @ d_pos) * d_pos
    d_freq /= np.linalg.norm(d_freq)
    d_s0 = rng.standard_normal(d)
    d_s0 -= (d_s0 @ d_pos) * d_pos + (d_s0 @ d_freq) * d_freq
    d_s0 /= np.linalg.norm(d_s0)
    # planted signals: positional cos at theta_0 = 1 (in the design span), freq linear
    log_freq = rng.uniform(0.0, 3.0, size=n)
    pos_scalar = 2.0 * np.cos(1.0 * positions)
    freq_scalar = 1.3 * (log_freq - log_freq.mean())
    sem = 1.5 * rng.standard_normal(n)
    x = (
        np.outer(pos_scalar, d_pos)
        + np.outer(freq_scalar, d_freq)
        + np.outer(sem, d_s0)
        + 0.02 * rng.standard_normal((n, d))
    ).astype(np.float32)

    pos_var = pos_scalar.var() * n
    freq_var = freq_scalar.var() * n
    sem_var = (sem * sem).sum()
    planted = (pos_var + freq_var) / (pos_var + freq_var + sem_var)

    def batches():
        for i in range(0, n, 1024):
            yield x[i : i + 1024]

    res = run(batches(), positions, d, 8, base, log_freq, 2, 1024)
    print(json.dumps(res, indent=2))
    print(f"planted nuisance fraction ~= {planted:.4f}")
    assert abs(res["combined_absorbed"] - planted) < 0.06, res
    assert res["positional_absorbed"] > 0.2, res
    assert res["frequency_absorbed"] > 0.1, res
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest", help="residual_shard_io harvest directory")
    ap.add_argument("--token-ids", help="order-matched per-row token ids .npy (frequency block)")
    ap.add_argument("--harmonics", type=int, default=8, help="positional Fourier harmonic pairs")
    ap.add_argument("--rope-base", type=float, default=DEFAULT_ROPE_BASE)
    ap.add_argument("--freq-degree", type=int, default=2, help="token log-freq polynomial degree")
    ap.add_argument("--batch-rows", type=int, default=8192)
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.harvest:
        raise SystemExit("provide --harvest <dir> (or --selftest)")

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples"))
    from residual_shard_io import ShardReader  # noqa: E402

    reader = ShardReader(args.harvest)
    seq_len = int(reader.manifest.get("seq_len", 0))
    if seq_len <= 1:
        raise SystemExit(f"harvest manifest lacks a usable seq_len ({seq_len}); cannot place positions")
    n = len(reader)
    d_model = reader.d_model
    # Rows are stored in token order with position 0 dropped: (seq_len-1) rows per
    # sequence, so within-sequence position is recoverable from the row index.
    per_seq = seq_len - 1
    positions = (np.arange(n) % per_seq) + 1

    log_freq = None
    if args.token_ids:
        ids = np.load(args.token_ids)
        if ids.shape[0] != n:
            raise SystemExit(f"--token-ids has {ids.shape[0]} rows but harvest has {n}")
        log_freq = log_unigram_from_ids(ids)

    res = run(
        reader.batches(args.batch_rows),
        positions,
        d_model,
        args.harmonics,
        args.rope_base,
        log_freq,
        args.freq_degree,
        args.batch_rows,
    )
    res["harvest"] = args.harvest
    res["model_name"] = reader.manifest.get("model_name")
    res["layer"] = reader.manifest.get("layer")
    res["seq_len"] = seq_len
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[nuisance-atlas] wrote {args.out}")


if __name__ == "__main__":
    main()
