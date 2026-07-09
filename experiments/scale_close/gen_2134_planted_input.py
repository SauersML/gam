#!/usr/bin/env python3
"""#2134 shape-matrix synthetic input — a planted low-rank + noise (N, P) matrix.

Writes a little-endian float32 C-order .npy that the compiled `scale_k` example
(crates/gam-sae/examples/scale_k.rs) accepts verbatim (its header parser requires
`<f4`, C-order, rank-2). The content is deliberately simple: `--dirs` shared
Gaussian directions in R^P with per-token random loadings plus isotropic noise, so
the block-sparse dictionary fit has genuine structure to explain (EV sanity leg)
while the run's purpose is to prove the previously-REFUSED overcomplete (N, K, P)
region — K > P — now streams through the front door with peak-RSS discipline (the
assertion lives inside scale_k, not here).

The input depends only on (N, P); K is the dictionary size swept by the sbatch, so
ONE file per (N, P) cell is reused across the K values. Rows are written in
`--chunk-rows` blocks so peak generator RSS stays at one chunk, never the full
(N, P) matrix — a 5e5 x 1024 f32 matrix is 2 GB and must not be built in one go.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def npy_v1_header(shape, dtype=np.dtype("<f4")):
    """Minimal .npy v1.0 header bytes for a C-order 2-D array (16-byte aligned)."""
    descr = dtype.str  # '<f4'
    dict_str = (
        "{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }"
        % (descr, shape[0], shape[1])
    )
    # magic(6) + version(2) + hlen(2) + dict + trailing \n, padded to 64-byte mult.
    base = 6 + 2 + 2 + len(dict_str) + 1
    pad = (64 - base % 64) % 64
    dict_str = dict_str + " " * pad + "\n"
    header = b"\x93NUMPY" + bytes([1, 0])
    header += struct.pack("<H", len(dict_str))
    header += dict_str.encode("latin1")
    return header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--dirs", type=int, default=32, help="shared planted directions")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-rows", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dirs = rng.standard_normal((args.dirs, args.p)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-8)

    with open(args.out, "wb") as f:
        f.write(npy_v1_header((args.n, args.p)))
        written = 0
        while written < args.n:
            rows = min(args.chunk_rows, args.n - written)
            loads = rng.standard_normal((rows, args.dirs)).astype(np.float32)
            block = loads @ dirs
            block += args.noise * rng.standard_normal((rows, args.p)).astype(np.float32)
            # np.tofile writes C-order little-endian f32 (the header's contract).
            np.ascontiguousarray(block, dtype="<f4").tofile(f)
            written += rows
    print(f"[#2134-gen] wrote {args.out} shape=({args.n},{args.p}) "
          f"dirs={args.dirs} noise={args.noise}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
