"""Bayesian model-evidence comparison across fits.

Rank candidate gamfit fits by their REML / LAML marginal-likelihood scores —
the same scalar the inner outer-loop already maximizes to select smoothing
parameters. Because that score already includes the Occam factors
``log|H| − log|S|_+`` (the Laplace approximation to the log marginal likelihood
with the integration over the coefficient block folded in), differences are
log Bayes factors with the model-complexity penalty baked in: increasing the
basis dimension or relaxing a penalty no longer scores higher for free.

This module is the model-comparison face of REML. Use it to choose between
fits that differ in *structure* — basis topology (circle, torus, sphere,
Euclidean patch), penalty order, presence/absence of a term — rather than to
re-derive smoothing parameters within a single model class.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._model import Model


_REML_SCORE_KEYS = ("reml_score", "evidence", "laml", "score")
_EDF_KEYS = ("edf_total", "edf", "effective_dof")
_PENALTY_RANK_KEYS = ("penalty_rank", "rank_s", "rank_S", "cache_penalty_rank")
_NULLITY_KEYS = (
    "null_dim",
    "nullity",
    "penalty_nullity",
    "cache_nullity",
)
_DIM_KEYS = ("effective_dim", "dim_h", "dim_H", "hessian_dim")
_TK_LOG_2PI = math.log(2.0 * math.pi)


def _extract_reml_score(fit: Any) -> float:
    """Return the REML / LAML log marginal-likelihood score for ``fit``.

    Accepts a :class:`Model` (uses its ``evidence`` / summary), a ``dict``
    returned by ``gaussian_reml_fit`` and friends, or anything mapping with a
    ``reml_score`` / ``evidence`` key. When rank metadata is present, the
    score is converted to the Tierney-Kadane dimension-normalized convention.
    """
    return _with_tierney_kadane_normalizer(fit, _extract_reml_score_raw(fit))


def _extract_reml_score_raw(fit: Any) -> float:
    if isinstance(fit, Model):
        summary = fit.summary().payload
        for key in _REML_SCORE_KEYS:
            if key in summary and summary[key] is not None:
                return float(summary[key])
        raise ValueError("Model summary is missing a reml_score / evidence field")
    if isinstance(fit, Mapping):
        for key in _REML_SCORE_KEYS:
            if key in fit and fit[key] is not None:
                return float(fit[key])
        raise ValueError(
            "fit dict is missing a reml_score / evidence field; expected one of "
            + ", ".join(_REML_SCORE_KEYS)
        )
    score = getattr(fit, "evidence", None)
    if score is not None:
        return float(score)
    score = getattr(fit, "reml_score", None)
    if score is not None:
        return float(score)
    raise TypeError(
        f"compare_models: cannot extract reml_score from {type(fit).__name__}; "
        "pass a gamfit.Model, a dict with 'reml_score', or an object exposing .evidence"
    )


def _with_tierney_kadane_normalizer(fit: Any, score: float) -> float:
    """Apply ``-0.5 * (dim(H) - rank(S)) * log(2π)`` when metadata exists."""
    null_dim = _extract_null_dim(fit)
    if null_dim is None:
        return float(score)
    if not math.isfinite(null_dim) or null_dim < -1e-9:
        raise ValueError(
            "compare_models: invalid Tierney-Kadane null dimension "
            f"{null_dim!r}"
        )
    return float(score) + _tierney_kadane_normalizer_from_null_dim(null_dim)


def _tierney_kadane_normalizer_from_null_dim(null_dim: float) -> float:
    if not math.isfinite(null_dim) or null_dim < -1e-9:
        raise ValueError(
            "compare_models: invalid Tierney-Kadane null dimension "
            f"{null_dim!r}"
        )
    return -0.5 * max(0.0, null_dim) * _TK_LOG_2PI


def _extract_null_dim(fit: Any) -> float | None:
    nullity = _extract_float_metadata(fit, _NULLITY_KEYS)
    if nullity is not None:
        return nullity * _extract_output_dim(fit)
    dim_h = _extract_float_metadata(fit, _DIM_KEYS)
    penalty_rank = _extract_float_metadata(fit, _PENALTY_RANK_KEYS)
    if dim_h is None or penalty_rank is None:
        return None
    return dim_h - penalty_rank


def _extract_output_dim(fit: Any) -> float:
    coefficients = _extract_metadata_value(fit, ("coefficients",))
    shape = getattr(coefficients, "shape", None)
    if shape is not None and len(shape) >= 2:
        return float(shape[1])
    return 1.0


def _extract_float_metadata(fit: Any, keys: tuple[str, ...]) -> float | None:
    value = _extract_metadata_value(fit, keys)
    if value is None:
        return None
    return float(value)


def _extract_metadata_value(fit: Any, keys: tuple[str, ...]) -> Any | None:
    mappings: list[Mapping[str, Any]] = []
    if isinstance(fit, Model):
        mappings.append(fit.summary().payload)
    if isinstance(fit, Mapping):
        mappings.append(fit)
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
    for key in keys:
        value = getattr(fit, key, None)
        if value is not None:
            return value
    return None


def _extract_edf(fit: Any) -> float | None:
    if isinstance(fit, Model):
        payload = fit.summary().payload
        for key in _EDF_KEYS:
            value = payload.get(key)
            if value is not None:
                return float(value)
        return None
    if isinstance(fit, Mapping):
        for key in _EDF_KEYS:
            value = fit.get(key)
            if value is not None:
                try:
                    # ``edf`` may be a per-block vector in array-API fits.
                    return float(sum(value))  # type: ignore[arg-type]
                except TypeError:
                    return float(value)
        return None
    value = getattr(fit, "edf_total", None)
    if value is None:
        value = getattr(fit, "edf", None)
    return None if value is None else float(value)


def _format_bayes_factor(log_bf: float) -> str:
    if not math.isfinite(log_bf):
        return "inf"
    # Use scientific notation once the BF is large.
    if abs(log_bf) >= math.log(10) * 3:
        exponent = log_bf / math.log(10)
        return f"1e{exponent:+.1f}"
    return f"{math.exp(log_bf):.3g}"


def compare_models(
    fits: Sequence[Any],
    names: Sequence[str] | None = None,
    *,
    cv_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Rank candidate fits by Bayesian marginal-likelihood (REML / LAML).

    Each fit's ``reml_score`` is treated as the Tierney-Kadane normalized
    marginal likelihood: ``-0.5 * (dim(H) - rank(S)) * log(2π)`` is included
    whenever rank metadata is present. Differences are log Bayes factors. Use
    this to compare fits that differ in *model structure* (basis topology,
    penalty order, presence of a term). The Occam-factor interpretation is
    only as good as the prior normalization and model comparability: candidate
    families should share the same response likelihood, comparable priors,
    controlled latent gauges, and the relevant normalizers for REML-selected
    penalty strengths. It is **not** a substitute for held-out predictive
    scoring on data the inner REML loop also saw; pass ``cv_scores`` to
    surface a cross-validation cross-check alongside the evidence.

    Parameters
    ----------
    fits :
        Sequence of fitted :class:`Model` objects or dicts returned by
        ``gaussian_reml_fit`` / ``gaussian_reml_fit_blocks_forward`` etc.
        Each must expose a ``reml_score`` (or ``evidence``) scalar.
    names :
        Optional human-readable labels, one per fit. Defaults to
        ``["fit_0", "fit_1", ...]``.
    cv_scores :
        Optional per-fit held-out score (higher = better, e.g. test
        log-likelihood). Surfaced alongside the evidence ranking; never used
        to override it.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``ranking`` — list of tuples
          ``(name, reml_score, delta_reml, bayes_factor_vs_best, effective_dof)``
          sorted by ``reml_score`` descending.
        * ``winner`` — name of the best fit by ``reml_score``.
        * ``evidence_summary`` — one-line text describing the win margin.
        * ``cv_optional`` — when ``cv_scores`` is supplied, a list of
          ``(name, cv_score)`` aligned to the same ordering as ``ranking``;
          otherwise omitted from the dict.

    Examples
    --------
    >>> circle = gamfit.fit(df, "y ~ s(theta, type='cyclic')")
    >>> torus  = gamfit.fit(df, "y ~ te(theta, phi, type='cyclic')")
    >>> result = gamfit.compare_models([circle, torus], names=["circle", "torus"])
    >>> result["winner"]
    'circle'
    >>> print(result["evidence_summary"])
    circle wins by Bayes factor 1e+3.2 over torus
    """
    if not fits:
        raise ValueError("compare_models requires at least one fit")
    if names is None:
        labels = [f"fit_{i}" for i in range(len(fits))]
    else:
        labels = [str(n) for n in names]
        if len(labels) != len(fits):
            raise ValueError(
                f"len(names)={len(labels)} does not match len(fits)={len(fits)}"
            )
    if cv_scores is not None and len(cv_scores) != len(fits):
        raise ValueError(
            f"len(cv_scores)={len(cv_scores)} does not match len(fits)={len(fits)}"
        )

    scored: list[tuple[str, float, float | None]] = []
    for name, fit in zip(labels, fits):
        scored.append((name, _extract_reml_score(fit), _extract_edf(fit)))

    scored.sort(key=lambda row: row[1], reverse=True)
    best_score = scored[0][1]
    ranking = [
        (name, score, score - best_score, math.exp(score - best_score), edf)
        for name, score, edf in scored
    ]
    winner = ranking[0][0]

    if len(ranking) >= 2:
        runner_up = ranking[1]
        log_bf = best_score - runner_up[1]
        summary = (
            f"{winner} wins by Bayes factor {_format_bayes_factor(log_bf)} "
            f"over {runner_up[0]}"
        )
    else:
        summary = f"{winner} (single fit; no comparison)"

    result: dict[str, Any] = {
        "ranking": ranking,
        "winner": winner,
        "evidence_summary": summary,
    }
    if cv_scores is not None:
        cv_by_name = dict(zip(labels, (float(s) for s in cv_scores)))
        result["cv_optional"] = [(name, cv_by_name[name]) for name, *_ in ranking]
    return result
