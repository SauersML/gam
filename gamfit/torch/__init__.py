"""PyTorch bridge for gamfit's analytic primitives.

This subpackage exposes the gamfit Rust engine to PyTorch users so analytic
closed-form derivatives can participate in ``loss.backward()`` flows. Every
function here is a thin wrapper over the corresponding NumPy entry point in
:mod:`gamfit._api`; no derivative math is rewritten in Python or torch.

Layout:

- Closed-form Gaussian REML fits — :func:`gaussian_reml_fit` and its batched
  and position-based variants. Forward and backward both go through Rust.
- Basis evaluations — :func:`bspline_basis`, :func:`duchon_basis_1d` and
  their derivative siblings. Backward routes through the matching Rust
  derivative primitive (chain rule applied at the boundary).
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
        duchon_basis_1d,
        duchon_basis_1d_derivative,
        gaussian_weighted_ridge,
        gaussian_weighted_ridge_batch,
        smoothness_penalty,
    )
    from ._cyclic_duchon import (
        CyclicDuchonFitOutput,
        CyclicDuchonTripleSmoother,
        cyclic_duchon_bernoulli_basis,
        cyclic_duchon_quadratic_fit,
        cyclic_duchon_triple_penalty,
    )
    from ._reml import (
        GaussianRemlOutput,
        gaussian_reml_fit,
        gaussian_reml_fit_batched,
        gaussian_reml_fit_positions,
        gaussian_reml_fit_positions_batched,
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
    "CyclicDuchonFitOutput",
    "CyclicDuchonTripleSmoother",
    "GaussianRemlOutput",
    "alr",
    "bspline_basis",
    "bspline_basis_derivative",
    "closure",
    "clr",
    "cyclic_duchon_bernoulli_basis",
    "cyclic_duchon_quadratic_fit",
    "cyclic_duchon_triple_penalty",
    "duchon_basis_1d",
    "duchon_basis_1d_derivative",
    "from_fitted",
    "gaussian_reml_fit",
    "gaussian_reml_fit_batched",
    "gaussian_reml_fit_positions",
    "gaussian_reml_fit_positions_batched",
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
