"""PyTorch bridge for gamfit's analytic primitives.

This subpackage exposes the gamfit Rust engine to PyTorch users so analytic
closed-form derivatives can participate in ``loss.backward()`` flows. Every
function here is a thin wrapper over the corresponding NumPy entry point in
:mod:`gamfit._api`; no derivative math is rewritten in Python or torch.

Layout:

- Closed-form Gaussian REML fits — :func:`gaussian_reml_fit`, its batched
  variant :func:`gaussian_reml_fit_batched`, and the multi-smooth additive
  wrapper :func:`gaussian_reml_fit_additive` (single-λ block-diagonal
  composition). Forward and backward both go through Rust.
- Basis evaluations — :func:`bspline_basis` and :func:`duchon_basis`. The
  B-spline backward routes through the Rust derivative primitive (chain
  rule applied at the boundary); the multi-dim Duchon basis is currently
  forward-only with respect to its points input.
- Penalty matrix construction and the closed-form ridge solver — forward-only.
- Response-geometry transforms — forward-only numpy passthrough; for
  differentiable variants compose the underlying torch ops directly.
- :func:`from_fitted` — wrap a fitted :class:`gamfit.Model` as a frozen
  ``nn.Module``.

The subpackage is an optional extra: ``pip install gamfit[torch]``. Importing
:mod:`gamfit` itself does not pull in torch.
"""

from __future__ import annotations

# Re-exporting the public surface below transitively imports ``torch`` via the
# submodules, so a missing optional ``torch`` dependency surfaces here as an
# :class:`ImportError`. Translate that into a user-actionable message before any
# downstream consumer sees a raw ``No module named 'torch'``.
try:
    from ._basis import (
        bspline_basis,
        bspline_basis_derivative,
        duchon_basis,
        gaussian_weighted_ridge,
        gaussian_weighted_ridge_batch,
        smoothness_penalty,
    )
    from ._reml import (
        AdditiveRemlOutput,
        GaussianRemlOutput,
        gaussian_reml_fit,
        gaussian_reml_fit_additive,
        gaussian_reml_fit_batched,
    )
    from .geometry import (
        alr,
        closure,
        clr,
        inverse_alr,
        simplex_exp_map,
        simplex_frechet_mean,
        simplex_log_map,
        sphere_exp_map,
        sphere_frechet_mean,
        sphere_log_map,
    )
    from .modules import from_fitted
except ImportError as _exc:  # pragma: no cover - import-time guard
    if _exc.name == "torch":
        raise ImportError(
            "gamfit.torch requires the optional `torch` dependency. "
            "Install via `pip install gamfit[torch]` or `pip install torch`."
        ) from _exc
    raise

__all__ = [
    "AdditiveRemlOutput",
    "GaussianRemlOutput",
    "alr",
    "bspline_basis",
    "bspline_basis_derivative",
    "closure",
    "clr",
    "duchon_basis",
    "from_fitted",
    "gaussian_reml_fit",
    "gaussian_reml_fit_additive",
    "gaussian_reml_fit_batched",
    "gaussian_weighted_ridge",
    "gaussian_weighted_ridge_batch",
    "inverse_alr",
    "simplex_exp_map",
    "simplex_frechet_mean",
    "simplex_log_map",
    "smoothness_penalty",
    "sphere_exp_map",
    "sphere_frechet_mean",
    "sphere_log_map",
]
