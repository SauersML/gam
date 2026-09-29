"""#2951 operation-first probes: the one owner of float64 dense decompositions.

Every probe decomposition is faer's, through the MPD surface's `dense` operation
(gamfit.sae.run_parameter_decomposition -> crates/gam-sae/src/parameter_decomposition/dense.rs), never
LAPACK's: torch's macOS wheels link Apple Accelerate, whose LAPACK returns wrong results on rank-deficient,
wide-spectrum matrices (quicophy/mdopt#574). Torch is for model forward passes and matmuls only.

Inputs may be numpy arrays or CPU torch tensors; each is sent as a float64 numpy array and every output is
numpy float64. Signs are the owner's canonical ones: each eigenvector and each left singular vector has its
largest-|entry| positive (the right singular vector flips with it), and QR has diag(R) >= 0. With band=True,
eigh/eigvalsh/svd/svdvals/spectral_norm also return the owner's rounding band (singular values:
max(m, n) eps sigma_1; eigenvalues: n (eps rho + eta)): a value within its band of zero is not resolved
from zero, so callers can threshold on it instead of an ad-hoc tolerance.
"""
from __future__ import annotations

import platform
import re
import sys

import numpy as np


def f64(a):
    """C-contiguous float64 numpy copy of a numpy array or (CPU) torch tensor."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.ascontiguousarray(np.array(a, dtype=np.float64))


def _dense(decomposition, **arrays):
    import gamfit
    out = gamfit.sae.run_parameter_decomposition(
        {"schema": "gam.mpd-request", "schema_version": 1,
         "operation": {"kind": "dense", "decomposition": decomposition}}, arrays)
    return out.report["result"]["decomposition"], out.arrays


def _assembly(psd_depth):
    """The request's symmetric assembly: mirrored (band zero) unless psd_depth names an accumulation depth."""
    return {"kind": "mirrored"} if psd_depth is None else {"kind": "psd_accumulation", "depth": int(psd_depth)}


def symmetrized(A):
    """(A + A^T)/2: one rounded value in both triangles, the mirrored assembly eigh/eigvalsh assume by default.
    For a product that is symmetric only in exact arithmetic (W D W^T, a GEMM Gram)."""
    A = f64(A)
    return 0.5 * (A + A.T)


def eigh(A, subset_by_index=None, band=False, psd_depth=None):
    """Symmetric eigendecomposition (ascending eigenvalues, canonical vector signs).
    subset_by_index=[lo, hi] keeps eigenpairs lo..hi inclusive (ascending order).
    The owner refuses A when its triangles disagree beyond the declared assembly's band: mirrored (exactly
    symmetric; see symmetrized) by default, or psd_depth=d for a sum of PSD pieces with at most d roundings
    per entry."""
    idx = None if subset_by_index is None else [int(subset_by_index[0]), int(subset_by_index[1]) + 1]
    rep, arr = _dense({"kind": "eigh", "matrix": "a", "assembly": _assembly(psd_depth), "indices": idx}, a=f64(A))
    out = (arr[rep["values"]], arr[rep["vectors"]])
    return out + (rep["band"],) if band else out


def eigvalsh(A, band=False, psd_depth=None):
    """Ascending eigenvalues of a symmetric matrix; assembly as for eigh."""
    rep, arr = _dense({"kind": "eigvalsh", "matrix": "a", "assembly": _assembly(psd_depth)}, a=f64(A))
    return (arr[rep["values"]], rep["band"]) if band else arr[rep["values"]]


def svd(A, full_matrices=False, band=False):
    """A = U diag(s) Vt, s descending, canonical signs."""
    rep, arr = _dense({"kind": "svd", "matrix": "a", "full": bool(full_matrices)}, a=f64(A))
    out = (arr[rep["u"]], arr[rep["s"]], arr[rep["vt"]])
    return out + (rep["band"],) if band else out


def svdvals(A, band=False):
    """Singular values, descending."""
    rep, arr = _dense({"kind": "svdvals", "matrix": "a"}, a=f64(A))
    return (arr[rep["s"]], rep["band"]) if band else arr[rep["s"]]


def spectral_norm(A, band=False):
    """||A||_2; leading axes of an ndim > 2 input are a batch (one norm per trailing matrix)."""
    A = f64(A)
    if A.ndim == 2:
        if not A.size:
            return (0.0, 0.0) if band else 0.0
        rep, _ = _dense({"kind": "spectral_norm", "matrix": "a"}, a=A)
        return (rep["norm"], rep["band"]) if band else rep["norm"]
    pairs = [spectral_norm(m, band=True) for m in A.reshape(-1, *A.shape[-2:])]
    norms = np.array([p[0] for p in pairs]).reshape(A.shape[:-2])
    return (norms, np.array([p[1] for p in pairs]).reshape(A.shape[:-2])) if band else norms


def qr(A, mode="economic"):
    """A = Q R with diag(R) >= 0. mode 'economic' or 'full' returns (Q, R); mode 'r' returns R only."""
    rep, arr = _dense({"kind": "qr", "matrix": "a", "mode": mode}, a=f64(A))
    return arr[rep["r"]] if mode == "r" else (arr[rep["q"]], arr[rep["r"]])


def solve(A, B):
    """A X = B for square A (partial-pivot LU); refused at a zero pivot. B a vector or a matrix."""
    rep, arr = _dense({"kind": "solve", "matrix": "a", "rhs": "b"}, a=f64(A), b=f64(B))
    return arr[rep["x"]]


def inv(A):
    """A^{-1} via solve(A, I)."""
    A = f64(A)
    return solve(A, np.eye(A.shape[0]))


def lstsq(A, B, cond=None):
    """Minimum-norm least squares through the thin SVD; singular values at or below the cutoff are dropped
    (cond None: the band max(m, n) eps sigma_1; else cond * sigma_1). Returns (X, per-column ||A x - b||^2,
    rank, kept singular values)."""
    cutoff = {"kind": "band"} if cond is None else {"kind": "relative", "rcond": float(cond)}
    rep, arr = _dense({"kind": "lstsq", "matrix": "a", "rhs": "b", "cutoff": cutoff}, a=f64(A), b=f64(B))
    return arr[rep["x"]], arr[rep["residuals"]], rep["rank"], arr[rep["s"]]


def env_record():
    """Python and linear-algebra backends, for receipts."""
    import gamfit
    rec = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "decompositions": "faer float64 via gamfit.sae.run_parameter_decomposition dense "
                             "(bench/mpd_opfirst_linalg_2951.py)",
           "gamfit": {"version": getattr(gamfit, "__version__", None), "path": gamfit.__file__},
           "numpy": np.__version__}
    try:
        import torch
        info = dict(re.findall(r"\b(BLAS_INFO|LAPACK_INFO)=([^,\s]+)", torch.__config__.show()))
        rec["torch"] = {"version": torch.__version__, "blas": info.get("BLAS_INFO"), "lapack": info.get("LAPACK_INFO"),
                        "used_for_decompositions": False}
    except ImportError:
        rec["torch"] = None
    return rec


if __name__ == "__main__":
    import json
    print(json.dumps(env_record(), indent=1))
