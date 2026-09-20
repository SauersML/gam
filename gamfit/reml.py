"""Low-level Gaussian / GLM REML solves with forward and backward passes.

These are the differentiable building blocks the ``gamfit.torch`` and
``gamfit.kernels_jax`` layers wrap. Formula fitting lives in :func:`gamfit.fit`.
"""

from __future__ import annotations

from ._api import (
    gaussian_reml_fit,
    gaussian_reml_fit_backward,
    gaussian_reml_fit_batched,
    gaussian_reml_fit_batched_backward,
    gaussian_reml_fit_blocks_backward,
    gaussian_reml_fit_blocks_forward,
    gaussian_reml_fit_formula,
    gaussian_reml_fit_latent,
    gaussian_reml_fit_latent_backward,
    gaussian_reml_optimize_latent,
    glm_reml_fit_latent,
    glm_reml_fit_latent_backward,
    gaussian_reml_fit_positions,
    gaussian_reml_fit_positions_backward,
    gaussian_reml_fit_positions_batched,
    gaussian_reml_fit_positions_batched_backward,
    gaussian_reml_fit_with_constraints_backward,
    gaussian_reml_fit_with_constraints_forward,
    gaussian_weighted_ridge,
    gaussian_weighted_ridge_batch,
)

__all__ = [
    "gaussian_reml_fit",
    "gaussian_reml_fit_backward",
    "gaussian_reml_fit_batched",
    "gaussian_reml_fit_batched_backward",
    "gaussian_reml_fit_blocks_backward",
    "gaussian_reml_fit_blocks_forward",
    "gaussian_reml_fit_formula",
    "gaussian_reml_fit_latent",
    "gaussian_reml_fit_latent_backward",
    "gaussian_reml_fit_positions",
    "gaussian_reml_fit_positions_backward",
    "gaussian_reml_fit_positions_batched",
    "gaussian_reml_fit_positions_batched_backward",
    "gaussian_reml_fit_with_constraints_backward",
    "gaussian_reml_fit_with_constraints_forward",
    "gaussian_reml_optimize_latent",
    "gaussian_weighted_ridge",
    "gaussian_weighted_ridge_batch",
    "glm_reml_fit_latent",
    "glm_reml_fit_latent_backward",
]
