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
    """A converged linear-dictionary model.

    This object exists only for a certified fit (SPEC 20): the Rust solver
    raises on non-convergence (with the sweeps/EV/tolerance evidence) instead
    of returning a best-effort iterate, so every instance is converged.
    """

    atoms: np.ndarray
    assignments: np.ndarray
    fitted: np.ndarray
    lambdas: np.ndarray
    reml_scores: np.ndarray
    explained_variance: float
    iterations: int
    convergence: dict[str, float | int]
    assignment: str
    top_k: int
    training_data: np.ndarray
    # Posterior-mean code shrinkage the fit applied, exactly as the Rust fit
    # returns it: ``1/(1 + λ̂)`` at the REML ridge for the K=1 rank-one lanes and
    # ``None`` for the multi-atom lane, whose codes are unshrunk. ``transform``
    # hands it back to Rust so held-out codes share the scale of
    # ``assignments``. It has no default: a fit selects it, a caller never does.
    posterior_shrinkage: float | None
    # Origin of the fitted model, exactly as the Rust fit returns it: the
    # training column means for the AFFINE centered K=1 lane (``mean +
    # code·atom``) and ``None`` for every LINEAR model. Python never re-derives
    # it; ``transform`` hands it back to Rust and ``reconstruct`` adds it.
    mean: np.ndarray | None = None
    # Softmax routing temperature the model was fitted with. ``transform`` must
    # encode with the fitted assignment rule (and, for ``"softmax"``, this
    # temperature) so held-out codes come from the same encoder as
    # ``assignments``.
    temperature: float = 0.25

    def reconstruct(self, assignments: Any | None = None) -> np.ndarray:
        codes = self.assignments if assignments is None else _as_2d_float(assignments, "assignments")
        if codes.shape[1] != self.atoms.shape[0]:
            raise ValueError(
                f"assignments must have K={self.atoms.shape[0]} columns; got {codes.shape[1]}"
            )
        recon = codes @ self.atoms
        if self.mean is not None:
            recon = recon + self.mean
        return np.ascontiguousarray(recon)

    @property
    def centered(self) -> bool:
        """Whether the fitted model is affine (the centered K=1 lane)."""
        return self.mean is not None

    def transform(self, X: Any, top_k: int | None = None) -> np.ndarray:
        """Encode held-out rows ``X`` (``M x P``) against the fitted dictionary.

        Routes the fitted model's assignment rule (the top-``top_k``
        minimum-norm least squares, or the top-``top_k`` softmax at the fitted
        ``temperature``)
        against the fitted origin ``mean``, scaled by the fitted
        ``posterior_shrinkage``, through the Rust core
        (``linear_dictionary_transform``), which also owns the input contract
        (finite ``X``, ``top_k`` in ``[1, K]``). Returns the ``M x K`` codes.
        """
        x = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        codes = rust_module().linear_dictionary_transform_ffi(
            x,
            np.ascontiguousarray(self.atoms, dtype=np.float64),
            int(self.top_k if top_k is None else top_k),
            self.posterior_shrinkage,
            mean=self.mean,
            assignment=str(self.assignment),
            temperature=float(self.temperature),
        )
        return np.ascontiguousarray(codes)


def linear_dictionary_fit(
    X: Any,
    K: int,
    *,
    max_iter: int = 30,
    top_k: int = 1,
    assignment: str = "top_k",
    temperature: float = 0.25,
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
        tolerance=float(tolerance),
        center_rank_one=bool(center_rank_one),
    )
    data = dict(payload)
    mean = data["mean"]
    posterior_shrinkage = data["posterior_shrinkage"]
    return LinearDictionaryFit(
        atoms=np.ascontiguousarray(data["atoms"], dtype=np.float64),
        assignments=np.ascontiguousarray(data["assignments"], dtype=np.float64),
        fitted=np.ascontiguousarray(data["fitted"], dtype=np.float64),
        lambdas=np.ascontiguousarray(data["lambdas"], dtype=np.float64),
        reml_scores=np.ascontiguousarray(data["reml_scores"], dtype=np.float64),
        explained_variance=float(data["explained_variance"]),
        iterations=int(data["iterations"]),
        convergence={
            "ev_residual": float(data["convergence"]["ev_residual"]),
            "routing_residual": float(data["convergence"]["routing_residual"]),
            "accepted_births": int(data["convergence"]["accepted_births"]),
            "tolerance": float(data["convergence"]["tolerance"]),
        },
        assignment=str(data["assignment"]),
        top_k=int(data["top_k"]),
        training_data=x,
        posterior_shrinkage=None if posterior_shrinkage is None else float(posterior_shrinkage),
        mean=None if mean is None else np.ascontiguousarray(mean, dtype=np.float64),
        temperature=float(temperature),
    )


__all__ = ["LinearDictionaryFit", "linear_dictionary_fit"]
