"""Python facade for the Rust fixed-K sparse, minibatched SAE trainer (#1026).

This is the "collapsed linear lane": an additive path for very large
dictionaries (``K`` up to tens of thousands) where the exact-REML / Arrow-Schur
dense joint manifold solver is the wrong engine. It routes each row against the
dictionary in ``K``-tiles, keeps only the top-``active`` atoms, and returns
fixed-width **sparse** routing (``indices[N, active]`` / ``codes[N, active]``)
so the ``N x K`` assignment matrix is never materialised. All heavy state is
FP32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._binding import rust_module


def _as_2d_f32(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1-D or 2-D numeric array; got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class SparseDictionaryFit:
    """Result of a collapsed-linear-lane fit.

    Attributes
    ----------
    decoder:
        ``K x P`` unit-norm decoder (one atom per row), FP32.
    indices:
        ``N x active`` active atom indices per row (``uint32``).
    codes:
        ``N x active`` sparse codes aligned with ``indices`` (FP32).
    fitted:
        ``N x P`` dense reconstruction of the training rows (FP32).
    explained_variance:
        Held-in EV (``1 - RSS/TSS``) of the fitted reconstruction.
    epochs, converged, active:
        Run metadata.
    """

    decoder: np.ndarray
    indices: np.ndarray
    codes: np.ndarray
    fitted: np.ndarray
    explained_variance: float
    epochs: int
    converged: bool
    active: int

    def reconstruct(self, indices: Any | None = None, codes: Any | None = None) -> np.ndarray:
        """Dense reconstruct from a sparse ``(indices, codes)`` routing.

        Defaults to the training routing. ``indices`` / ``codes`` must be
        ``N x active`` with matching shapes.
        """
        idx = self.indices if indices is None else np.asarray(indices, dtype=np.uint32)
        cod = self.codes if codes is None else np.asarray(codes, dtype=np.float32)
        if idx.shape != cod.shape:
            raise ValueError(f"indices {idx.shape} and codes {cod.shape} must match")
        n, s = idx.shape
        p = self.decoder.shape[1]
        out = np.zeros((n, p), dtype=np.float32)
        for j in range(s):
            out += cod[:, [j]] * self.decoder[idx[:, j]]
        return np.ascontiguousarray(out)

    def transform(self, X: Any, active: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Route held-out rows ``X`` (``M x P``) through the fitted decoder.

        Returns ``(indices, codes)`` of shape ``M x active`` (sparse routing),
        computed by the Rust core (``sparse_dictionary_transform``): the same
        tiled top-``active`` routing + active-set ridge solve the trainer uses,
        against the frozen decoder.
        """
        x = _as_2d_f32(X, "X")
        if x.shape[1] != self.decoder.shape[1]:
            raise ValueError(
                f"X must have P={self.decoder.shape[1]} columns; got {x.shape[1]}"
            )
        s = self.active if active is None else int(active)
        s = max(1, min(s, self.decoder.shape[0]))
        indices, codes = rust_module().sparse_dictionary_transform_ffi(
            np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(self.decoder, dtype=np.float32),
            int(s),
        )
        return np.ascontiguousarray(indices), np.ascontiguousarray(codes)


def sparse_dictionary_fit(
    X: Any,
    K: int,
    *,
    active: int = 1,
    minibatch: int = 512,
    max_epochs: int = 30,
    score_tile: int = 4096,
    code_ridge: float = 1.0e-6,
    decoder_ridge: float = 1.0e-6,
    tolerance: float = 1.0e-6,
) -> SparseDictionaryFit:
    """Fit a fixed-``K`` sparse, minibatched linear dictionary to ``X`` (``N x P``).

    Parameters
    ----------
    K:
        Dictionary width (number of atoms). May be very large.
    active:
        Routing sparsity ``s`` (atoms allowed to fire per row). Shared, not
        per-atom.
    minibatch, max_epochs, score_tile:
        Streaming / tiling controls.
    code_ridge, decoder_ridge, tolerance:
        Shared regularisation and stopping controls.
    """
    x = _as_2d_f32(X, "X")
    payload = rust_module().sparse_dictionary_fit(
        x,
        int(K),
        active=int(active),
        minibatch=int(minibatch),
        max_epochs=int(max_epochs),
        score_tile=int(score_tile),
        code_ridge=float(code_ridge),
        decoder_ridge=float(decoder_ridge),
        tolerance=float(tolerance),
    )
    data = dict(payload)
    return SparseDictionaryFit(
        decoder=np.ascontiguousarray(data["decoder"], dtype=np.float32),
        indices=np.ascontiguousarray(data["indices"], dtype=np.uint32),
        codes=np.ascontiguousarray(data["codes"], dtype=np.float32),
        fitted=np.ascontiguousarray(data["fitted"], dtype=np.float32),
        explained_variance=float(data["explained_variance"]),
        epochs=int(data["epochs"]),
        converged=bool(data["converged"]),
        active=int(data["active"]),
    )


__all__ = ["SparseDictionaryFit", "sparse_dictionary_fit"]
