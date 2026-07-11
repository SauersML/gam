"""Torch tensor adapter for a converged Rust manifold-SAE fit.

There is deliberately no second, gradient-trained SAE in this module.  Model
construction, latent inference, assignment, smoothing selection, and
out-of-sample projection all belong to :func:`gamfit.sae_manifold_fit` and its
Rust implementation.  :class:`ManifoldSAE` only converts tensors at that fitted
model boundary and serializes the immutable fit inside an ``nn.Module`` state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .._binding import rust_module
from .._sae_manifold import ManifoldSAE as _FittedManifoldSAE
from ._coerce import from_numpy_like, to_numpy_f64


@dataclass(frozen=True, slots=True)
class ManifoldSAEOutput:
    """Tensor view of one converged native manifold-SAE inference.

    ``reconstruction``, ``codes``, and ``coordinates`` are all emitted by the
    same frozen Rust fit. ``penalized_loss_score`` is the inner fit-quality
    diagnostic; ``penalized_quasi_laplace_criterion`` is the terminal custom
    PSD/Gauss--Newton quasi-Laplace scalar with rank charges. It is not LAML,
    REML, or normalized model evidence. No encoder logits,
    surrogate gates, or eager-only smoothing parameters are exposed.
    """

    reconstruction: torch.Tensor
    codes: torch.Tensor
    coordinates: tuple[torch.Tensor, ...]
    penalized_loss_score: torch.Tensor | None
    penalized_quasi_laplace_criterion: torch.Tensor
    selected_smooth_lambdas: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class CircularReplicateCoverage:
    """Whether one replicate spans a two-dimensional circle embedding."""

    replicate: int
    minimum_embedding_eigenvalue: float
    maximum_embedding_eigenvalue: float
    isotropic_coverage: float
    rank_resolution: float
    well_posed: bool


@dataclass(frozen=True, slots=True)
class CircularPairConcordance:
    """One replicate pair aligned by rotation or reflection."""

    left: int
    right: int
    rotation_score: float | None
    reflection_score: float | None
    aligned_score: float | None
    reflected: bool | None
    phase_shift: float | None


@dataclass(frozen=True, slots=True)
class CircularConcordanceReport:
    """Cross-seed latent-angle stability modulo the circle's exact O(2) gauge."""

    n_replicates: int
    n_rows: int
    period: float
    coverage: tuple[CircularReplicateCoverage, ...]
    pairs: tuple[CircularPairConcordance, ...]
    minimum_aligned_score: float | None
    mean_aligned_score: float | None


def circular_concordance(
    coordinates: torch.Tensor | np.ndarray, *, period: float = 1.0
) -> CircularConcordanceReport:
    """Report cross-seed angle agreement after rotation/reflection alignment."""
    if isinstance(coordinates, torch.Tensor):
        values = to_numpy_f64(coordinates)
    else:
        values = np.ascontiguousarray(np.asarray(coordinates, dtype=np.float64))
    raw = rust_module().sae_circular_concordance(values, float(period))
    coverage = tuple(
        CircularReplicateCoverage(
            replicate=int(item["replicate"]),
            minimum_embedding_eigenvalue=float(
                item["minimum_embedding_eigenvalue"]
            ),
            maximum_embedding_eigenvalue=float(
                item["maximum_embedding_eigenvalue"]
            ),
            isotropic_coverage=float(item["isotropic_coverage"]),
            rank_resolution=float(item["rank_resolution"]),
            well_posed=bool(item["well_posed"]),
        )
        for item in raw["coverage"]
    )
    pairs = tuple(
        CircularPairConcordance(
            left=int(item["left"]),
            right=int(item["right"]),
            rotation_score=(
                None
                if item["rotation_score"] is None
                else float(item["rotation_score"])
            ),
            reflection_score=(
                None
                if item["reflection_score"] is None
                else float(item["reflection_score"])
            ),
            aligned_score=(
                None
                if item["aligned_score"] is None
                else float(item["aligned_score"])
            ),
            reflected=(
                None if item["reflected"] is None else bool(item["reflected"])
            ),
            phase_shift=(
                None
                if item["phase_shift"] is None
                else float(item["phase_shift"])
            ),
        )
        for item in raw["pairs"]
    )
    return CircularConcordanceReport(
        n_replicates=int(raw["n_replicates"]),
        n_rows=int(raw["n_rows"]),
        period=float(raw["period"]),
        coverage=coverage,
        pairs=pairs,
        minimum_aligned_score=(
            None
            if raw["minimum_aligned_score"] is None
            else float(raw["minimum_aligned_score"])
        ),
        mean_aligned_score=(
            None
            if raw["mean_aligned_score"] is None
            else float(raw["mean_aligned_score"])
        ),
    )


class ManifoldSAE(nn.Module):
    """Frozen ``nn.Module`` adapter around a converged native SAE fit.

    Construct the model with :func:`gamfit.sae_manifold_fit`, then wrap that
    returned native object.  ``forward`` runs the native converged-latent path
    for both training rows and unseen rows.  It is intentionally
    non-differentiable: fitting or differentiating a second torch objective
    would no longer describe the native model.
    """

    def __init__(self, fitted: _FittedManifoldSAE) -> None:
        super().__init__()
        if not isinstance(fitted, _FittedManifoldSAE):
            raise TypeError(
                "gamfit.torch.ManifoldSAE expects the converged object returned "
                "by gamfit.sae_manifold_fit"
            )
        self._fitted = fitted
        self.register_buffer(
            "_fit_blob", self._encode_fit(fitted), persistent=True
        )

    @staticmethod
    def _encode_fit(fitted: _FittedManifoldSAE) -> torch.Tensor:
        blob = json.dumps(fitted.to_dict()).encode("utf-8")
        return torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()

    @staticmethod
    def _decode_fit(blob: torch.Tensor) -> _FittedManifoldSAE:
        if blob.numel() == 0:
            raise ValueError("serialized manifold-SAE fit is empty")
        raw = blob.detach().cpu().numpy().tobytes()
        payload = json.loads(raw.decode("utf-8"))
        return _FittedManifoldSAE.from_dict(payload)

    @property
    def fitted(self) -> _FittedManifoldSAE:
        """The authoritative converged native fit."""
        return self._fitted

    @property
    def input_dim(self) -> int:
        """Ambient input dimension of the fitted dictionary."""
        return int(np.asarray(self._fitted.training_mean).size)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        if not isinstance(x, torch.Tensor):
            raise TypeError("ManifoldSAE forward expects a torch.Tensor")
        if x.layout != torch.strided or x.dim() != 2:
            raise ValueError(
                f"ManifoldSAE expected a dense (N, D) tensor; got {tuple(x.shape)}"
            )
        if int(x.shape[1]) != self.input_dim:
            raise ValueError(
                f"ManifoldSAE expected input_dim={self.input_dim}; got {x.shape[1]}"
            )
        if not (x.is_floating_point() and x.dtype in (torch.float32, torch.float64)):
            raise TypeError(
                "ManifoldSAE input must have dtype torch.float32 or torch.float64"
            )
        if x.requires_grad:
            raise ValueError(
                "gamfit.torch.ManifoldSAE is a frozen fitted-model adapter; "
                "input gradients are unavailable"
            )

        latents = self._fitted.converged_latents(to_numpy_f64(x))
        reconstruction = from_numpy_like(
            np.asarray(latents["fitted"], dtype=np.float64), x
        )
        codes = from_numpy_like(
            np.asarray(latents["assignments"], dtype=np.float64), x
        )
        coordinates = tuple(
            from_numpy_like(np.asarray(coords, dtype=np.float64), x)
            for coords in latents["coords"]
        )

        score_value = self._fitted.penalized_loss_score
        penalized_loss_score = (
            None
            if score_value is None
            else torch.as_tensor(score_value, dtype=x.dtype, device=x.device)
        )
        penalized_quasi_laplace_criterion = torch.as_tensor(
            self._fitted.penalized_quasi_laplace_criterion,
            dtype=x.dtype,
            device=x.device,
        )
        selected = self._fitted.selected_lambda_smooth
        selected_smooth_lambdas = (
            None
            if selected is None
            else from_numpy_like(np.asarray(selected, dtype=np.float64), x)
        )
        return ManifoldSAEOutput(
            reconstruction=reconstruction,
            codes=codes,
            coordinates=coordinates,
            penalized_loss_score=penalized_loss_score,
            penalized_quasi_laplace_criterion=penalized_quasi_laplace_criterion,
            selected_smooth_lambdas=selected_smooth_lambdas,
        )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        blob_key = prefix + "_fit_blob"
        if blob_key in state_dict:
            incoming = torch.as_tensor(
                state_dict[blob_key], dtype=torch.uint8
            ).reshape(-1)
            self._fit_blob = incoming.clone()
            state_dict = dict(state_dict)
            state_dict[blob_key] = self._fit_blob
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if blob_key in state_dict:
            self._fitted = self._decode_fit(self._fit_blob)

    def summary(self) -> dict[str, Any]:
        """Delegate to the native fit summary."""
        return self._fitted.summary()

    def description_length(
        self, *, l_param_bits: float | None = None
    ) -> dict[str, Any] | None:
        """Delegate to the native fit description-length report."""
        return self._fitted.description_length(l_param_bits=l_param_bits)


__all__ = [
    "CircularConcordanceReport",
    "CircularPairConcordance",
    "CircularReplicateCoverage",
    "ManifoldSAE",
    "ManifoldSAEOutput",
    "circular_concordance",
]
