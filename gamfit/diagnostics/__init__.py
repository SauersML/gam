"""Identifiability diagnostics for fitted latent-factor / manifold-SAE models.

Each diagnostic checks one identifiability *theorem*'s preconditions
against a fitted model or its inputs and returns an
:class:`IdentifiabilityReport`. The composite helper
:func:`identifiability_report` auto-dispatches over the diagnostics that
apply to a given model type and returns a
:class:`CompositeIdentifiabilityReport`.

Available diagnostics
---------------------

* :func:`check_aux_richness` — iVAE auxiliary-richness (Khemakhem et al. 2020).
* :func:`check_jacobian_sparsity` — decoder-Jacobian sparsity
  (Hyvarinen-Morioka 2017; Lachapelle et al. 2024).
* :func:`check_anchor_consistency` — manifold-SAE atom-anchor coverage
  (project convention).
* :func:`identifiability_report` — composite, model-aware dispatcher.
"""

from __future__ import annotations

from typing import Any

from ._report import CompositeIdentifiabilityReport, IdentifiabilityReport
from .anchor_consistency import check_anchor_consistency
from .aux_richness import check_aux_richness
from .jacobian_sparsity import check_jacobian_sparsity

__all__ = [
    "IdentifiabilityReport",
    "CompositeIdentifiabilityReport",
    "check_aux_richness",
    "check_jacobian_sparsity",
    "check_anchor_consistency",
    "identifiability_report",
]


def _model_kind(model: Any) -> str:
    """Short string describing the family of a fitted-model object."""
    cls = type(model).__name__
    return cls


def identifiability_report(model: Any) -> CompositeIdentifiabilityReport:
    """Run every applicable identifiability diagnostic on ``model``.

    Dispatch is structural, not name-based:

    * If ``model`` exposes an ``aux`` plus latent attributes (e.g. the
      ``IdentifiableFactorFitResult`` returned by
      :func:`gamfit.identifiable_factor_fit`), :func:`check_aux_richness`
      is run.
    * If ``model`` exposes a ``decoder`` (linear) or ``decoder_blocks``
      attribute, :func:`check_jacobian_sparsity` is run.
    * If ``model`` exposes an ``assignments`` matrix (manifold-SAE-style),
      :func:`check_anchor_consistency` is run.

    Returns
    -------
    CompositeIdentifiabilityReport

    Examples
    --------
    >>> rep = gamfit.diagnostics.identifiability_report(model)
    >>> rep.summary()
    'Identified.'
    >>> print(rep.detail())
    """
    kind = _model_kind(model)
    reports: list[IdentifiabilityReport] = []

    # 1. iVAE-style aux-richness: need an `aux`-like attribute plus latents.
    aux_arr = None
    latents = None
    if hasattr(model, "_aux_diagnostic") and hasattr(model, "_latents_diagnostic"):
        aux_arr = getattr(model, "_aux_diagnostic")
        latents = getattr(model, "_latents_diagnostic")
    elif hasattr(model, "aux") and hasattr(model, "T_supervised"):
        aux_arr = model.aux
        latents = model.T_supervised

    if aux_arr is not None and latents is not None:
        reports.append(check_aux_richness(aux_arr, latents))

    # 2. Jacobian sparsity: need a decoder or decoder_blocks.
    if hasattr(model, "decoder") or hasattr(model, "decoder_blocks"):
        reports.append(check_jacobian_sparsity(model))

    # 3. Anchor consistency: need an assignment matrix.
    if hasattr(model, "assignments"):
        reports.append(check_anchor_consistency(model))

    return CompositeIdentifiabilityReport(model_kind=kind, reports=reports)
