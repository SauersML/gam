"""#2951 operation-first probes: the one owner of float64 dense decompositions.

torch 2.11 macOS wheels link Apple Accelerate for LAPACK (torch.__config__: LAPACK_INFO=accelerate), and
Accelerate's LAPACK returns wrong results / corrupts memory on rank-deficient, wide-spectrum matrices
(quicophy/mdopt#574). Every probe decomposition therefore goes through scipy.linalg on numpy float64 arrays,
which in ~/mpd-data/venv links scipy-openblas. Torch is for model forward passes and matmuls only.

Inputs may be numpy arrays or CPU torch tensors; every input is copied to float64 numpy and every output is
numpy float64. Vector signs are canonical and deterministic: each eigenvector, each left singular vector
(with its right partner), and each Q column (diag R >= 0) is flipped so the stated entry is positive. No
Rust surface op (gamfit.sae.run_parameter_decomposition) owns a generic dense decomposition, so none is
re-implemented or bypassed here.
"""
from __future__ import annotations

import platform
import re
import sys

import numpy as np
import scipy.linalg as sla


def f64(a):
    """float64 numpy copy of a numpy array or (CPU) torch tensor."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.array(a, dtype=np.float64)


def _canon_cols(V):
    """Flip each column so its largest-|entry| (first on ties) is positive; returns the signs."""
    if V.shape[0] == 0 or V.shape[1] == 0:
        return np.ones(V.shape[1])
    pick = V[np.abs(V).argmax(0), np.arange(V.shape[1])]
    s = np.where(pick < 0, -1.0, 1.0)
    V *= s
    return s


def eigh(A, subset_by_index=None):
    """Symmetric eigendecomposition (lower triangle read, ascending eigenvalues, canonical vector signs).
    subset_by_index=[lo, hi] keeps eigenpairs lo..hi inclusive (ascending order)."""
    w, V = sla.eigh(f64(A), subset_by_index=subset_by_index, overwrite_a=True, check_finite=True)
    _canon_cols(V)
    return w, V


def eigvalsh(A):
    """Ascending eigenvalues of a symmetric matrix (lower triangle read)."""
    return sla.eigh(f64(A), eigvals_only=True, overwrite_a=True, check_finite=True)


def svd(A, full_matrices=False):
    """A = U diag(s) Vt, s descending; each left singular vector's largest-|entry| is positive and its right
    partner flipped with it."""
    U, s, Vt = sla.svd(f64(A), full_matrices=full_matrices, overwrite_a=True, check_finite=True,
                       lapack_driver="gesdd")
    k = len(s)
    sign = _canon_cols(U[:, :k])
    Vt[:k] *= sign[:, None]
    return U, s, Vt


def svdvals(A):
    """Singular values, descending."""
    return sla.svd(f64(A), compute_uv=False, overwrite_a=True, check_finite=True, lapack_driver="gesdd")


def spectral_norm(A):
    """||A||_2; leading axes of an ndim > 2 input are a batch (one norm per trailing matrix)."""
    A = f64(A)
    if A.ndim == 2:
        return float(svdvals(A)[0]) if A.size else 0.0
    flat = A.reshape(-1, *A.shape[-2:])
    return np.array([svdvals(m)[0] if m.size else 0.0 for m in flat]).reshape(A.shape[:-2])


def qr(A, mode="economic"):
    """A = Q R with diag(R) >= 0 (unique for full column rank). mode 'economic' or 'full' returns (Q, R);
    mode 'r' returns R only (economic)."""
    if mode == "r":
        R = sla.qr(f64(A), mode="r", overwrite_a=True, check_finite=True)[0]
        k = min(R.shape)
        R[:k] *= np.where(np.diag(R)[:k] < 0, -1.0, 1.0)[:, None]
        return R
    Q, R = sla.qr(f64(A), mode=mode, overwrite_a=True, check_finite=True)
    k = min(R.shape)
    s = np.where(np.diag(R)[:k] < 0, -1.0, 1.0)
    Q[:, :k] *= s
    R[:k] *= s[:, None]
    return Q, R


def solve(A, B):
    """A X = B for square nonsingular A (LU with partial pivoting); raises on exact singularity."""
    return sla.solve(f64(A), f64(B), overwrite_a=True, overwrite_b=True, check_finite=True)


def inv(A):
    """A^{-1} via solve(A, I)."""
    A = f64(A)
    return solve(A, np.eye(A.shape[0]))


def lstsq(A, B, cond=None):
    """Minimum-norm least squares (gelsd). Returns (X, residues, rank, singular values)."""
    return sla.lstsq(f64(A), f64(B), cond=cond, overwrite_a=True, overwrite_b=True, check_finite=True,
                     lapack_driver="gelsd")


def eig(A):
    """General (non-symmetric) eigendecomposition: complex eigenvalues w and right eigenvectors V (columns,
    unit 2-norm, each scaled so its largest-modulus entry is real positive)."""
    w, V = sla.eig(f64(A), overwrite_a=True, check_finite=True)
    if V.size:
        pick = V[np.abs(V).argmax(0), np.arange(V.shape[1])]
        V = V * (np.abs(pick) / np.where(pick == 0, 1, pick))
    return w, V


def _np_like_lapack(mod):
    try:
        deps = mod.show_config(mode="dicts")["Build Dependencies"]
        return {k: "%s %s" % (deps[k].get("name"), deps[k].get("version")) for k in ("blas", "lapack")}
    except Exception as exc:  # recorded, never fatal
        return {"error": type(exc).__name__}


def env_record():
    """Python and linear-algebra backends, for receipts. Decompositions here use scipy's LAPACK."""
    import scipy
    rec = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "decompositions": "scipy.linalg float64 (bench/mpd_opfirst_linalg_2951.py)",
           "numpy": {"version": np.__version__, **_np_like_lapack(np)},
           "scipy": {"version": scipy.__version__, **_np_like_lapack(scipy)}}
    try:
        import torch
        cfg = torch.__config__.show()
        info = dict(re.findall(r"\b(BLAS_INFO|LAPACK_INFO)=([^,\s]+)", cfg))
        rec["torch"] = {"version": torch.__version__, "blas": info.get("BLAS_INFO"), "lapack": info.get("LAPACK_INFO"),
                        "used_for_decompositions": False}
    except ImportError:
        rec["torch"] = None
    return rec


if __name__ == "__main__":
    import json
    print(json.dumps(env_record(), indent=1))
