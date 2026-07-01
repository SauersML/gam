"""Python facade for the Rust linear dictionary fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._binding import rust_module


def _as_2d_float(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
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
class LinearDictionaryFit:
    atoms: np.ndarray
    assignments: np.ndarray
    fitted: np.ndarray
    lambdas: np.ndarray
    reml_scores: np.ndarray
    explained_variance: float
    iterations: int
    converged: bool
    assignment: str
    top_k: int
    code_ridge: float
    training_data: np.ndarray
    # When the K=1 centered-PCA-ceiling lane is active the model is AFFINE
    # (mean + rank-1), so held-out transform/reconstruct must subtract/add the
    # training column mean. `centered` is only true for the K=1 centered lane;
    # for every other model it is False and `mean` is a zero vector, keeping the
    # linear (mean-free) behavior byte-identical.
    centered: bool = False
    mean: np.ndarray | None = None

    def reconstruct(self, assignments: Any | None = None) -> np.ndarray:
        codes = self.assignments if assignments is None else _as_2d_float(assignments, "assignments")
        if codes.shape[1] != self.atoms.shape[0]:
            raise ValueError(
                f"assignments must have K={self.atoms.shape[0]} columns; got {codes.shape[1]}"
            )
        recon = codes @ self.atoms
        if self.centered and self.mean is not None:
            recon = recon + self.mean
        return np.ascontiguousarray(recon)

    def transform(self, X: Any, top_k: int | None = None) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if x.shape[1] != self.atoms.shape[1]:
            raise ValueError(
                f"X must have p={self.atoms.shape[1]} columns; got {x.shape[1]}"
            )
        k_active = self.top_k if top_k is None else int(top_k)
        if k_active < 1 or k_active > self.atoms.shape[0]:
            raise ValueError(
                f"top_k must be in [1, K={self.atoms.shape[0]}]; got {k_active}"
            )
        # Affine centered lane: encode the mean-removed input so the codes match
        # the centered principal direction; `reconstruct` adds the mean back.
        x_eff = x - self.mean if (self.centered and self.mean is not None) else x
        scores = x_eff @ self.atoms.T
        gram = self.atoms @ self.atoms.T
        codes = np.zeros((x.shape[0], self.atoms.shape[0]), dtype=np.float64)
        for row in range(x.shape[0]):
            active = np.argpartition(np.abs(scores[row]), -k_active)[-k_active:]
            system = gram[np.ix_(active, active)] + self.code_ridge * np.eye(k_active)
            codes[row, active] = np.linalg.solve(system, scores[row, active])
        return np.ascontiguousarray(codes)


def linear_dictionary_fit(
    X: Any,
    K: int,
    *,
    max_iter: int = 30,
    top_k: int = 1,
    assignment: str = "top_k",
    temperature: float = 0.25,
    code_ridge: float = 1.0e-8,
    tolerance: float = 1.0e-7,
    center_rank_one: bool = False,
) -> LinearDictionaryFit:
    x = _as_2d_float(X, "X")
    payload = rust_module().linear_dictionary_fit(
        x,
        int(K),
        max_iter=int(max_iter),
        top_k=int(top_k),
        assignment=str(assignment),
        temperature=float(temperature),
        code_ridge=float(code_ridge),
        tolerance=float(tolerance),
        center_rank_one=bool(center_rank_one),
    )
    data = dict(payload)
    # The Rust centered lane only engages at K=1; mirror that so the affine
    # (mean-aware) transform/reconstruct is used exactly when the fit is centered.
    is_centered = bool(center_rank_one) and int(K) == 1
    mean = x.mean(axis=0) if is_centered else np.zeros(x.shape[1], dtype=np.float64)
    return LinearDictionaryFit(
        atoms=np.ascontiguousarray(data["atoms"], dtype=np.float64),
        assignments=np.ascontiguousarray(data["assignments"], dtype=np.float64),
        fitted=np.ascontiguousarray(data["fitted"], dtype=np.float64),
        lambdas=np.ascontiguousarray(data["lambdas"], dtype=np.float64),
        reml_scores=np.ascontiguousarray(data["reml_scores"], dtype=np.float64),
        explained_variance=float(data["explained_variance"]),
        iterations=int(data["iterations"]),
        converged=bool(data["converged"]),
        assignment=str(data["assignment"]),
        top_k=int(data["top_k"]),
        code_ridge=float(code_ridge),
        training_data=x,
        centered=is_centered,
        mean=np.ascontiguousarray(mean, dtype=np.float64),
    )


__all__ = ["LinearDictionaryFit", "linear_dictionary_fit"]
