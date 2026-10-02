"""Code length of an explanation's reals under gam's #2951 codec
(crates/gam-sae/src/parameter_decomposition/{codec,precision}.rs):

  LatticeCode message = omega(count + 1) + signed_omega(p) + sum_k signed_delta(k),
  k = round(x * 2^p), signed_X(v) = X(zigzag(v) + 1); lattice indices use Elias delta,
  the count and precision fields Elias omega.

One LatticeCode per tensor (one operator = one declared precision), so p is chosen per tensor.
`precision_for(x, b)` puts b bits of resolution under the tensor's RMS: p = b - round(log2 rms).
"""

import math

import numpy as np
import torch
from torch import Tensor


def omega_len(n: np.ndarray) -> np.ndarray:
    """Elias omega codeword length for integers n >= 1 (codec.rs prefix_integer_len_bits)."""
    n = n.astype(np.uint64)
    bits = np.ones(n.shape, dtype=np.int64)
    g = n.copy()
    while True:
        live = g > 1
        if not live.any():
            return bits
        w = np.zeros_like(g)
        w[live] = np.floor(np.log2(g[live].astype(np.float64))).astype(np.uint64) + 1
        # exact bit width for large values (float log2 can round at powers of two)
        over = live & ((np.uint64(1) << np.minimum(w, 63).astype(np.uint64)) <= g) & (w < 64)
        w[over] += 1
        under = live & (w > 0) & ((np.uint64(1) << (w - 1).astype(np.uint64)) > g)
        w[under] -= 1
        bits[live] += w[live].astype(np.int64)
        g = np.where(live, w - 1, g)


def delta_len(n: np.ndarray) -> np.ndarray:
    """Elias delta codeword length for integers n >= 1: L + 2 floor(log2(L + 1)) + 1, L = floor(log2 n)."""
    n = n.astype(np.uint64)
    L = np.zeros(n.shape, dtype=np.int64)
    g = n.copy()
    while True:  # exact floor(log2) by shifting (no float rounding)
        live = g > 1
        if not live.any():
            break
        L[live] += 1
        g = np.where(live, g >> np.uint64(1), g)
    M = np.floor(np.log2((L + 1).astype(np.float64))).astype(np.int64)  # L + 1 <= 64: exact in float
    return L + 2 * M + 1


def zigzag1(k: np.ndarray) -> np.ndarray:
    k = k.astype(np.int64)
    return np.where(k >= 0, 2 * k, -2 * k - 1).astype(np.uint64) + np.uint64(1)


def signed_len(k: np.ndarray) -> np.ndarray:
    """Signed Elias omega (the precision field)."""
    return omega_len(zigzag1(k))


def signed_delta_len(k: np.ndarray) -> np.ndarray:
    """Signed Elias delta (lattice indices)."""
    return delta_len(zigzag1(k))


def precision_for(x: Tensor, b: int) -> int:
    rms = x.float().pow(2).mean().sqrt().item()
    return b - round(math.log2(rms)) if rms > 0 else b


def quantize(x: Tensor, p: int) -> Tensor:
    return torch.round(x * 2.0**p) * 2.0**-p


def lattice_bits(x: Tensor, p: int, chunk: int = 1 << 24) -> int:
    """Exact LatticeCode::write length for tensor x at precision p."""
    flat = x.detach().reshape(-1)
    total = int(omega_len(np.array([flat.numel() + 1]))[0]) + int(signed_len(np.array([p]))[0])
    for i in range(0, flat.numel(), chunk):
        k = torch.round(flat[i:i + chunk].double().cpu() * 2.0**p).numpy().astype(np.int64)
        total += int(signed_delta_len(k).sum())
    return total


def check():
    """omega lengths from the Elias omega table: 1->1, 2->3, 3->3, 4->6, 7->6, 8->7, 15->7, 16->11."""
    got = omega_len(np.array([1, 2, 3, 4, 7, 8, 15, 16, 100, 1 << 20]))
    want = [1, 3, 3, 6, 6, 7, 7, 11, 13, 32]
    assert list(got) == want, (list(got), want)
    # delta: 1->1, 2->4, 3->4, 4->5, 7->5, 8->8, 15->8, 16->9, 100->11
    got = delta_len(np.array([1, 2, 3, 4, 7, 8, 15, 16, 100]))
    want = [1, 4, 4, 5, 5, 8, 8, 9, 11]
    assert list(got) == want, (list(got), want)


if __name__ == "__main__":
    check()
    print("ok")
