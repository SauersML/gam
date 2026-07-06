"""Evidence-based topology selection over common smooth topologies.

Two public selectors are exposed:

* :func:`select_topology` builds candidate formulas around an
  ``s(..., type=AUTO)`` smooth and ranks fitted models by evidence-like scores.
* :class:`TopologyAutoSelector` is a descriptor-style helper for selecting the
  topology of one :class:`gamfit.LatentCoord` block while preserving the rest
  of the caller's fit configuration.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from . import topology
from ._api import fit
from ._binding import rust_module
from ._compare import (
    _extract_edf,
    _extract_reml_score_raw,
    _tierney_kadane_normalizer_from_null_dim,
    compare_models,
)
from ._tables import PreNormalizedTable, normalize_table, table_columns
from .smooth import (
    Duchon,
    LatentCoord,
    PeriodicSplineCurve,
    Smooth,
    Sphere,
    TensorBSpline,
)


# Sentinel placeholder for the `auto` tuple passed through `_formula_for_candidate`.
# The Rust assembler scans for `type=AUTO` itself, so the Python side only needs
# to know whether such a term exists; the tuple's interior is unused.
_AUTO_PRESENT: tuple[int, int, str] = (-1, -1, "")
_AUTO_RE = re.compile(r"\btype\s*=\s*(['\"]?)AUTO\1(?=\s*(?:,|\)))", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    topology: Smooth


class _TopologyRustModule(Protocol):
    def assemble_candidate_formula(
        self,
        formula: str,
        candidate_json: str,
        strict_dimension: bool,
    ) -> str | None: ...

    def rank_topology_candidates(self, evidence_json: str) -> str: ...

    def stacking_weights_from_log_density(
        self,
        names: list[str],
        log_density_rows: list[list[float]],
    ) -> str: ...

    def topology_bic_score(
        self, fit: Any, n_obs: int, basis_size: int
    ) -> float: ...


BasisSpec: TypeAlias = Smooth
ScoreKind: TypeAlias = Literal["reml", "laml", "bic", "tk"]
ScoreScale: TypeAlias = Literal["per_observation", "per_effective_dim", "raw"]
TopologyName: TypeAlias = Literal[
    "euclidean", "circle", "sphere", "torus", "cylinder"
]
TopologyScoreScale: TypeAlias = Literal["per_effective_dim", "per_observation"]
TopologyAutoSelectorRank: TypeAlias = tuple[str, float, float, float, int, Any]
_TOPOLOGY_SCREEN_OUTER_MAX_ITER = 4
_TOPOLOGY_SCREEN_SURVIVORS = 2

# Escalating cheap-cap cascade for candidate enumeration (#781). A single fixed
# screening cap silently drops every candidate when none converges inside it,
# forcing the caller into the empty-screen error even though a slightly larger
# budget would have admitted survivors. Mirror the seed-screening contract in
# `solver::outer_strategy` (`SEED_SCREENING_CASCADE_MULTIPLIERS = [1, 4, 16]`
# then uncapped): run every candidate at the cheapest cap, and escalate to the
# next cap ONLY when the whole stage rejected everything as non-finite. The
# first stage that admits at least one finite-scored candidate wins, so the
# common case still pays a single cheap pass over all candidates. `None`
# denotes the uncapped final stage (no `outer_max_iter` override).
_TOPOLOGY_SCREEN_CAP_MULTIPLIERS: tuple[int, ...] = (1, 4, 16)
_TOPOLOGY_SCREEN_CASCADE_CAPS: tuple[int | None, ...] = (
    *(_TOPOLOGY_SCREEN_OUTER_MAX_ITER * m for m in _TOPOLOGY_SCREEN_CAP_MULTIPLIERS),
    None,
)

_DEFAULT_TOPOLOGY_NAMES: tuple[TopologyName, ...] = (
    "euclidean",
    "circle",
    "sphere",
    "torus",
    "cylinder",
)

_NULL_HESSIAN_LOGDET_KEYS: tuple[str, ...] = (
    "null_space_logdet",
    "null_hessian_logdet",
    "h_null_logdet",
    "logdet_h_null",
)


def _fit_kwargs_with_outer_max_iter(fit_kwargs: Mapping[str, Any], cap: int) -> dict[str, Any]:
    if cap < 1:
        raise ValueError(f"outer_max_iter cap must be >= 1, got {cap}")
    out = dict(fit_kwargs)
    cfg = dict(cast(Mapping[str, Any], out.get("config") or {}))
    existing = cfg.get("outer_max_iter")
    cfg["outer_max_iter"] = min(int(existing), cap) if existing is not None else cap
    out["config"] = cfg
    return out


def _survivor_count(n_candidates: int) -> int:
    return min(n_candidates, _TOPOLOGY_SCREEN_SURVIVORS)


def _screen_kwargs_for_cap(
    fit_kwargs: Mapping[str, Any], cap: int | None
) -> dict[str, Any]:
    """Fit kwargs for one cascade stage: cap the outer iterations, or run
    uncapped when ``cap`` is ``None`` (the final escalation stage)."""
    if cap is None:
        return dict(fit_kwargs)
    return _fit_kwargs_with_outer_max_iter(fit_kwargs, cap)


def _screen_candidates_with_budget_cascade(
    candidates: Sequence[_Candidate],
    fit_kwargs: Mapping[str, Any],
    score_one: "Callable[[_Candidate, Mapping[str, Any]], float]",
) -> tuple[list[tuple[float, int, _Candidate]], dict[str, str]]:
    """Run candidate screening through the escalating cheap-cap cascade (#781).

    ``score_one`` fits ``candidate`` under the stage's capped ``fit_kwargs`` and
    returns a finite lower-is-better comparison score, or raises to reject the
    candidate at this cap. The cascade evaluates every candidate at the cheapest
    cap and escalates to the next cap only when the entire stage rejected every
    candidate as non-finite/errored — matching ``rank_indices_with_budget_cascade``
    in ``solver::priority_selection``, which breaks the moment a stage admits a
    finite survivor. Returns ``(screened, errors)`` where ``screened`` is the
    ``(score, original_index, candidate)`` list from the first non-empty stage
    and ``errors`` carries the most recent rejection reason per candidate name.
    """
    errors: dict[str, str] = {}
    for cap in _TOPOLOGY_SCREEN_CASCADE_CAPS:
        stage_kwargs = _screen_kwargs_for_cap(fit_kwargs, cap)
        screened: list[tuple[float, int, _Candidate]] = []
        stage_errors: dict[str, str] = {}
        for idx, candidate in enumerate(candidates):
            try:
                score = score_one(candidate, stage_kwargs)
            except Exception as exc:
                stage_errors[candidate.name] = str(exc)
                continue
            if not math.isfinite(score):
                stage_errors[candidate.name] = f"non-finite screening score {score!r}"
                continue
            screened.append((score, idx, candidate))
        if screened:
            return screened, stage_errors
        errors = stage_errors
    return [], errors


@dataclass(frozen=True, slots=True)
class SelectTopologyResult:
    """Result returned by :func:`select_topology`.

    Attributes
    ----------
    winner_name:
        Name of the selected candidate.
    winner_fit:
        Fitted model object for ``winner_name``.
    scores:
        Selected score for every fully refit survivor after applying
        ``score_scale``.
    rankings:
        Candidate names ordered best-first with their selected scores.
    score_kind, score_scale:
        Normalized scoring choices used for the run.
    basis_sizes:
        Fitted basis size per candidate.
    effective_dim:
        Effective degrees of freedom per candidate.
    n_obs:
        Observation count per candidate.
    warnings:
        Cross-score disagreement warnings; empty when rankings agree or cannot
        be compared.
    fits:
        Fully refit survivor models when ``return_fits=True``; otherwise
        ``None``.
    """

    winner_name: str
    winner_fit: Any
    scores: dict[str, float]
    rankings: list[tuple[str, float]]
    score_kind: str
    score_scale: str
    basis_sizes: dict[str, int]
    effective_dim: dict[str, float]
    n_obs: dict[str, int]
    warnings: list[str]
    fits: dict[str, Any] | None = None


def select_topology(
    data: Any,
    response: str,
    candidates: Sequence[tuple[str, BasisSpec] | Mapping[str, Any]] | None = None,
    *,
    score: ScoreKind = "reml",
    score_scale: ScoreScale = "per_observation",
    return_fits: bool = False,
    **fit_kwargs: Any,
) -> SelectTopologyResult:
    """Select a topology by fitting candidates and ranking model evidence.

    Parameters
    ----------
    data:
        Table-like input accepted by :func:`gamfit.fit`.
    response:
        Response column name. A full formula is rejected; this helper creates
        ``"<response> ~ s(<all other columns>, type=AUTO)"`` internally.
    candidates:
        Optional sequence of ``(name, Smooth)`` pairs or mappings with
        ``"name"`` / ``"topology"``. When omitted, candidates are chosen from
        Euclidean patch, circle, sphere, torus, and cylinder constructors whose
        required dimension matches the predictor count.
    score:
        ``"reml"``, ``"laml"``, ``"bic"``, or ``"tk"``. ``"tk"`` adds the
        Tierney-Kadane null-space normalizer to the raw REML/evidence score.
    score_scale:
        ``"per_observation"``, ``"per_effective_dim"``, or ``"raw"``.
    return_fits:
        Include all fitted candidate models on the result.
    **fit_kwargs:
        Forwarded unchanged to :func:`gamfit.fit` for every candidate.

    Returns
    -------
    SelectTopologyResult
        Winner, rankings, per-candidate diagnostics, and optionally all fits.

    Raises
    ------
    ValueError
        If the response is missing, fewer than two candidates are available, a
        candidate fit yields a non-finite score, or requested score metadata is
        unavailable.
    TypeError
        If explicit candidate entries are not mappings or ``(name, Smooth)``
        pairs.
    """
    # Gauge invariant: Tierney-Kadane comparisons require every candidate's
    # penalty null space to be represented with the same deterministic
    # orthonormal-basis convention. The Rust summary reports
    # log|N.T @ H_p @ N| from the engine's RRQR null-space basis; mixing that
    # with caller-supplied non-orthonormal gauges would change the normalizer.
    score_kind = _normalize_score_kind(score)
    score_scale_kind = _normalize_score_scale(score_scale)
    formula, feature_dim, n_obs = _formula_from_response(data, response)
    normalized = _normalize_candidates(candidates, feature_dim=feature_dim)
    _find_auto_smooth_call(formula)

    # Hoist topology-independent shared work above the AUTO candidate loop
    # (#869). Two computations are invariant to the candidate topology and to
    # the budget-cascade cap, yet were recomputed for every (candidate, stage)
    # pair plus each survivor refit:
    #
    #   1. Table ingestion. ``gamfit.fit`` re-runs ``normalize_table(data)`` —
    #      an O(n_rows * n_cols) cell-stringification — on every call. The same
    #      table is fit up to (#cascade_stages * #candidates + #survivors)
    #      times, so normalize it once here and pass a ``PreNormalizedTable``
    #      that ``fit`` returns verbatim instead of re-coercing.
    #   2. Candidate formula surgery. ``_formula_for_candidate`` depends only on
    #      ``(formula, candidate)``, not on the cap or the eventual winner, so
    #      build each candidate's formula once and reuse it across every
    #      screening stage and the survivor refit.
    headers, rows, table_kind = normalize_table(data)
    shared_table = PreNormalizedTable(headers, rows, table_kind)
    candidate_formulas: dict[int, str] = {}

    def _candidate_formula_cached(candidate: _Candidate) -> str:
        cached = candidate_formulas.get(id(candidate))
        if cached is not None:
            return cached
        candidate_formula = _formula_for_candidate(
            formula,
            candidate,
            strict_dimension=True,
        )
        if candidate_formula is None:  # defensive; strict_dimension=True raises.
            raise ValueError(f"candidate {candidate.name!r} is not constructible")
        candidate_formulas[id(candidate)] = candidate_formula
        return candidate_formula

    def _screen_score(candidate: _Candidate, stage_kwargs: Mapping[str, Any]) -> float:
        candidate_formula = _candidate_formula_cached(candidate)
        model = fit(shared_table, candidate_formula, **stage_kwargs)
        reml_score = _extract_reml_score_raw(model)
        if not math.isfinite(reml_score):
            raise ValueError(f"degenerate REML score {reml_score!r}")
        basis_size = _basis_size(model)
        null_dim = _fitted_null_dim(model)
        raw_score = _score_for_kind(model, score_kind, n_obs, basis_size, null_dim)
        screen_score = _scale_score(raw_score, score_scale_kind, n_obs, _effective_dim(model))
        return _comparison_score(screen_score, score_kind)

    screened, screen_errors = _screen_candidates_with_budget_cascade(
        normalized,
        fit_kwargs,
        _screen_score,
    )

    if not screened:
        detail = "; ".join(f"{name}: {err}" for name, err in screen_errors.items())
        raise ValueError(
            "select_topology: no candidate produced a finite screening score"
            + (f" ({detail})" if detail else "")
        )

    screened.sort(key=lambda row: (row[0], row[1]))
    survivors = [row[2] for row in screened[:_survivor_count(len(screened))]]

    fits: dict[str, Any] = {}
    names: list[str] = []
    fit_list: list[Any] = []
    for candidate in survivors:
        candidate_formula = _candidate_formula_cached(candidate)
        model = fit(shared_table, candidate_formula, **fit_kwargs)
        reml_score = _extract_reml_score_raw(model)
        if not math.isfinite(reml_score):
            raise ValueError(
                f"select_topology: survivor {candidate.name!r} produced "
                f"degenerate REML score {reml_score!r}"
            )
        fits[candidate.name] = model
        names.append(candidate.name)
        fit_list.append(model)

    basis_sizes = {name: _basis_size(fit_obj) for name, fit_obj in fits.items()}
    effective_dim = {
        name: _effective_dim(fit_obj)
        for name, fit_obj in fits.items()
    }
    n_obs_by_candidate = {name: n_obs for name in fits}
    null_dims = {
        candidate.name: _fitted_null_dim(fits[candidate.name])
        for candidate in survivors
        if candidate.name in fits
    }
    raw_scores = {
        name: _score_for_kind(
            fit_obj,
            score_kind,
            n_obs,
            basis_sizes[name],
            null_dims[name],
        )
        for name, fit_obj in fits.items()
    }
    selected_scores = {
        name: _scale_score(raw_scores[name], score_scale_kind, n_obs, effective_dim[name])
        for name in fits
    }
    comparison_scores = {
        name: _comparison_score(value, score_kind)
        for name, value in selected_scores.items()
    }
    compared = compare_models(
        [{"reml_score": comparison_scores[name], "edf": _extract_edf(fit_obj)}
         for name, fit_obj in zip(names, fit_list)],
        names=names,
    )
    rankings = [
        (name, selected_scores[name])
        for name, *_ in compared["ranking"]
    ]
    winner_name = compared["winner"]
    warnings_out = _score_disagreement_warnings(
        fits,
        n_obs,
        basis_sizes,
        effective_dim,
        null_dims,
        score_scale_kind,
    )

    return SelectTopologyResult(
        winner_name=winner_name,
        winner_fit=fits[winner_name],
        scores=selected_scores,
        rankings=rankings,
        score_kind=score_kind,
        score_scale=score_scale_kind,
        basis_sizes=basis_sizes,
        effective_dim=effective_dim,
        n_obs=n_obs_by_candidate,
        warnings=warnings_out,
        fits=fits if return_fits else None,
    )


# Coverage level whose Gaussian observation band is inverted to recover each
# candidate's per-point predictive standard deviation. 0.95 keeps the band wide
# enough that support clamping is rare while staying away from the extreme tails
# where the symmetric-Gaussian band approximation is weakest.
_STACK_INTERVAL_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class TopologyStack:
    """Stacked predictive mixture over retained topology candidate fits (#768).

    Built by :func:`stack_topologies` from the candidate fits that
    :func:`select_topology` retains and a held-out labeled fold. The mixture
    weights are the simplex maximiser of the held-out mean logarithmic score of
    the stacked predictive density (Yao, Vehtari, Simpson & Gelman 2018) — the
    principled alternative to winner-take-all selection. Calling :meth:`predict`
    returns the stacked response-scale predictive mean ``Σ_k w_k μ_k(x)`` at new
    rows.

    Attributes
    ----------
    weights:
        Stacking weight per candidate name; sums to one. Candidates the held-out
        fold could not score receive zero weight.
    mean_log_score:
        Achieved held-out mean log-score at ``weights`` (higher is better).
    names:
        Candidate names in a deterministic order.
    """

    weights: dict[str, float]
    mean_log_score: float
    names: tuple[str, ...]
    _fits: Mapping[str, Any]

    def predict(self, data: Any, **predict_kwargs: Any) -> "list[float]":
        """Stacked response-scale predictive mean at the rows of ``data``.

        Each retained candidate predicts the response-scale mean over ``data``;
        the per-candidate means are combined with the stacking weights. Extra
        keyword arguments are forwarded to each candidate's ``predict``.
        """
        cand_means: dict[str, list[float]] = {}
        n_rows: int | None = None
        for name in self.names:
            weight = self.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            means = _predict_response_mean(self._fits[name], data, **predict_kwargs)
            if n_rows is None:
                n_rows = len(means)
            elif len(means) != n_rows:
                raise ValueError(
                    "TopologyStack candidates disagree on prediction row count"
                )
            cand_means[name] = means
        if n_rows is None:
            raise ValueError("TopologyStack has no positively-weighted candidate")
        out = [0.0] * n_rows
        for name, means in cand_means.items():
            weight = self.weights[name]
            for i, value in enumerate(means):
                out[i] += weight * value
        return out


def stack_topologies(
    fits: Mapping[str, Any],
    holdout: Any,
    response: str,
    *,
    interval_level: float = _STACK_INTERVAL_LEVEL,
) -> TopologyStack:
    """Stack retained topology candidate fits into a predictive mixture (#768).

    Parameters
    ----------
    fits:
        Retained candidate fits keyed by name — e.g. the ``fits`` mapping from
        :func:`select_topology` called with ``return_fits=True``.
    holdout:
        A labeled held-out fold (independent of the fits' training data) in any
        format accepted by :func:`gamfit.fit`. Must carry the ``response``
        column alongside every predictor each candidate references. The
        per-candidate held-out log-predictive densities of this fold's
        responses define the stacking objective.
    response:
        Name of the response column in ``holdout``.
    interval_level:
        Coverage of the predictive band inverted to recover each candidate's
        per-point predictive standard deviation.

    Returns
    -------
    TopologyStack
        Stacking weights, achieved held-out mean log-score, and a stacked
        :meth:`~TopologyStack.predict`.

    Notes
    -----
    The held-out predictive density is Gaussian in the candidate's response-scale
    predictive moments: mean ``μ_k(x)`` and total predictive standard deviation
    ``σ_k(x)`` recovered from the family-correct observation interval the Rust
    predictor emits (``Var(μ̂) + Var(Y|μ)``). This keeps every family-specific
    variance in the Rust core; only the generic log-score weight solve and
    mixture run in Python (the solve itself is the Rust
    ``stacking_weights_from_log_density`` binding). Rows whose recovered σ is
    non-positive (e.g. fully clamped against the response support) carry no
    Gaussian density and are dropped from that candidate's column.
    """
    if not fits:
        raise ValueError("stack_topologies requires at least one candidate fit")
    if not (0.0 < interval_level < 1.0):
        raise ValueError("interval_level must lie in (0, 1)")
    names = tuple(fits.keys())
    columns, _kind = table_columns(holdout)
    if response not in columns:
        raise ValueError(f"response column {response!r} not found in holdout fold")
    try:
        y = [float(value) for value in columns[response]]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"holdout response column {response!r} is not numeric"
        ) from exc
    if not y:
        raise ValueError("stack_topologies holdout fold cannot be empty")

    z = NormalDist().inv_cdf(0.5 + 0.5 * interval_level)
    # log_density_rows[i][k] = log p_k(y_i | x_i) on the held-out fold.
    log_density_rows: list[list[float]] = [
        [float("-inf")] * len(names) for _ in range(len(y))
    ]
    for k, name in enumerate(names):
        means, sds = _holdout_predictive_moments(fits[name], holdout, interval_level, z)
        if len(means) != len(y) or len(sds) != len(y):
            raise ValueError(
                f"candidate {name!r} predicted {len(means)} rows for a "
                f"{len(y)}-row holdout fold"
            )
        for i, (mean, sd) in enumerate(zip(means, sds)):
            if sd > 0.0 and math.isfinite(mean) and math.isfinite(sd):
                log_density_rows[i][k] = _gaussian_logpdf(y[i], mean, sd)

    raw = _topology_rust().stacking_weights_from_log_density(
        list(names), log_density_rows
    )
    parsed = json.loads(raw)
    weights = {name: float(parsed["weights"].get(name, 0.0)) for name in names}
    return TopologyStack(
        weights=weights,
        mean_log_score=float(parsed["mean_log_score"]),
        names=names,
        _fits=dict(fits),
    )


def _gaussian_logpdf(y: float, mean: float, sd: float) -> float:
    # Generic scalar Gaussian algebra used only to populate the held-out
    # log-density matrix before the Rust stacking solve consumes it. There is no
    # family-specific scoring or penalty math here, and crossing the FFI once
    # per row would make the boundary noisier than the computation.
    z = (y - mean) / sd
    return -0.5 * math.log(2.0 * math.pi) - math.log(sd) - 0.5 * z * z


def _holdout_predictive_moments(
    model: Any,
    holdout: Any,
    interval_level: float,
    z: float,
) -> tuple[list[float], list[float]]:
    """Per-point response-scale predictive mean and total predictive std on the
    held-out fold, both sourced from the Rust predictor's family-correct
    observation interval ``μ ± z·σ`` with ``σ² = Var(μ̂) + Var(Y|μ)``."""
    prediction = model.predict(
        holdout,
        interval=interval_level,
        observation_interval=True,
        return_type="dict",
    )
    means = [float(value) for value in prediction["mean"]]
    lower = [float(value) for value in prediction["observation_lower"]]
    upper = [float(value) for value in prediction["observation_upper"]]
    sds = [(hi - lo) / (2.0 * z) for lo, hi in zip(lower, upper)]
    return means, sds


def _predict_response_mean(model: Any, data: Any, **predict_kwargs: Any) -> list[float]:
    prediction = model.predict(data, return_type="dict", **predict_kwargs)
    if isinstance(prediction, Mapping) and "mean" in prediction:
        return [float(value) for value in prediction["mean"]]
    # Families whose predict() returns a bare response vector (e.g. probabilities)
    # rather than the linear-predictor/mean table.
    return [float(value) for value in prediction]


def _normalize_candidates(
    candidates: Sequence[tuple[str, BasisSpec] | Mapping[str, Any]] | None,
    *,
    feature_dim: int,
) -> list[_Candidate]:
    if candidates is None:
        candidates_out = _default_candidates(feature_dim)
        if len(candidates_out) < 2:
            raise ValueError(
                "select_topology requires at least two default candidates "
                f"for {feature_dim}-D predictors"
            )
        return candidates_out
    if len(candidates) < 2:
        raise ValueError("select_topology requires at least two candidates")
    out: list[_Candidate] = []
    seen: set[str] = set()
    for i, spec in enumerate(candidates):
        if isinstance(spec, Mapping):
            topo = spec.get("topology")
            name_obj = spec.get("name")
        else:
            try:
                name_obj, topo = spec
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "candidate entries must be (name, topology) tuples or "
                    "mappings with 'name' and 'topology'"
                ) from exc
        if not isinstance(topo, Smooth):
            raise TypeError(f"candidate {i} has no gamfit Smooth topology object")
        name = str(name_obj or _infer_candidate_name(topo) or f"candidate_{i}")
        if name in seen:
            raise ValueError(f"duplicate topology candidate name {name!r}")
        seen.add(name)
        out.append(_Candidate(name, topo))
    if len(out) < 2:
        raise ValueError("select_topology requires at least two candidates")
    return out


def _default_candidates(feature_dim: int) -> list[_Candidate]:
    candidates = [
        _default_topology_candidate(name, feature_dim)
        for name in _DEFAULT_TOPOLOGY_NAMES
    ]
    return [
        candidate for candidate in candidates
        if _candidate_required_dim(candidate.topology) in {None, feature_dim}
    ]


def _default_topology_candidate(name: str, feature_dim: int) -> _Candidate:
    if name == "euclidean":
        return _Candidate("euclidean", topology.EuclideanPatch(d=feature_dim, name="x"))
    if name == "circle":
        return _Candidate("circle", topology.Circle(name="theta"))
    if name == "sphere":
        return _Candidate("sphere", topology.Sphere(dim=2, name="omega"))
    if name == "torus":
        return _Candidate("torus", topology.Torus(n_knots=(12, 12), name="theta_phi"))
    if name == "cylinder":
        return _Candidate("cylinder", topology.Cylinder(name="cyl"))
    raise AssertionError(name)


def _formula_from_response(data: Any, response: str) -> tuple[str, int, int]:
    text = str(response).strip()
    if "~" in text:
        raise ValueError("select_topology response must be a response column name")
    columns, _kind = table_columns(data)
    if text not in columns:
        raise ValueError(f"response column {text!r} not found in data")
    features = [name for name in columns if name != text]
    if not features:
        raise ValueError("plain-response select_topology needs at least one feature column")
    n_obs = len(columns[text])
    if n_obs == 0:
        raise ValueError("select_topology data cannot be empty")
    return f"{text} ~ s({', '.join(features)}, type=AUTO)", len(features), n_obs


def _find_auto_smooth_call(formula: str) -> tuple[int, int, str]:
    """Return a sentinel triple when `formula` contains a `type=AUTO` smooth.

    The Rust formula assembler does its own AUTO scan, paren matching, and
    argument splitting; this Python wrapper only needs to signal presence vs.
    absence. The returned tuple's interior is intentionally a sentinel — the
    sole consumer (`_formula_for_candidate`) ignores it and routes through
    the Rust pyfunction.
    """
    if _AUTO_RE.search(formula):
        return _AUTO_PRESENT
    raise ValueError("select_topology requires one s(..., type=AUTO) smooth term")


def _formula_for_candidate(
    formula: str,
    candidate: _Candidate,
    *,
    strict_dimension: bool,
) -> str | None:
    """Replace the `type=AUTO` term in `formula` with the candidate-specific term.

    The formula-string surgery (paren matching, comma splitting, option
    emission, dimension checks) lives in Rust. This wrapper translates the
    Python `Smooth` subclass instance into a typed JSON description and
    invokes the Rust assembler.
    """
    payload = _candidate_to_rust_payload(candidate)
    try:
        result = _topology_rust().assemble_candidate_formula(
            formula,
            json.dumps(payload),
            strict_dimension,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return result


def _topology_rust() -> _TopologyRustModule:
    return cast(_TopologyRustModule, rust_module())


def _candidate_to_rust_payload(candidate: _Candidate) -> dict[str, Any]:
    """Translate a Python `Smooth` topology into the typed JSON shape the
    Rust formula assembler consumes.
    """
    topo = candidate.topology
    if isinstance(topo, PeriodicSplineCurve):
        return {
            "kind": "periodic_spline_curve",
            "n_knots": int(topo.n_knots),
            "degree": int(topo.degree),
            "penalty_order": int(topo.penalty_order),
        }
    if isinstance(topo, Sphere):
        return {
            "kind": "sphere",
            "n_centers": int(topo.n_centers),
            "penalty_order": int(topo.penalty_order),
            "kernel": str(topo.kernel),
            "radians": bool(topo.radians),
        }
    if isinstance(topo, TensorBSpline):
        k_attr = getattr(topo, "_gamfit_tensor_k", None)
        periodic = [bool(marginal.periodic) for marginal in topo.marginals]
        periods_attr = getattr(topo, "_gamfit_tensor_periods", None)
        periods_payload: list[str | None] | None
        if periods_attr is None:
            periods_payload = None
        else:
            periods_payload = [
                None if value is None else str(value)
                for value in periods_attr
            ]
        return {
            "kind": "tensor",
            "k": [int(value) for value in (k_attr or ())],
            "periodic": periodic,
            "periods": periods_payload,
        }
    if isinstance(topo, Duchon):
        per_axis_periodic = bool(
            any(bool(v) for v in (topo.periodic_per_axis or ()))
        )
        centers_int = int(topo.centers) if isinstance(topo.centers, int) else None
        length_scale = (
            None if topo.length_scale is None else float(topo.length_scale)
        )
        return {
            "kind": "duchon",
            "m": int(topo.m),
            "centers_int": centers_int,
            "per_axis_periodic": per_axis_periodic,
            "length_scale": length_scale,
            "required_dim": _candidate_required_dim(topo),
        }
    raise TypeError(f"unsupported topology candidate {type(topo).__name__}")


def _candidate_required_dim(topo: Smooth) -> int | None:
    dim = getattr(topo, "_gamfit_topology_dim", None)
    if dim is not None:
        return int(dim)
    if isinstance(topo, PeriodicSplineCurve):
        return 1
    if isinstance(topo, Sphere):
        return 2
    if isinstance(topo, TensorBSpline):
        return len(topo.marginals)
    if isinstance(topo, Duchon):
        periodic = tuple(bool(v) for v in topo.periodic_per_axis or ())
        if periodic in {(True, False), (True, True)}:
            return 2
        return _centers_dim(topo.centers)
    return None


def _fitted_null_dim(fit_obj: Any) -> float:
    null_dim = _extract_null_dim(fit_obj)
    if null_dim is not None:
        return null_dim
    raise ValueError(
        "Topology selection requires Rust null-dimension metadata; "
        "fit summary is missing null_dim"
    )


def _centers_dim(centers: Any) -> int | None:
    shape = getattr(centers, "shape", None)
    if shape is not None:
        if len(shape) == 1:
            return 1
        if len(shape) >= 2:
            return int(shape[1])
    return None


def _infer_candidate_name(topo: Smooth) -> str | None:
    if isinstance(topo, PeriodicSplineCurve):
        return "Circle"
    if isinstance(topo, Sphere):
        return "Sphere"
    if isinstance(topo, TensorBSpline):
        periodic = tuple(bool(marginal.periodic) for marginal in topo.marginals)
        if periodic == (True, False):
            return "Cylinder"
        if periodic == (True, True):
            return "Torus"
    if isinstance(topo, Duchon):
        periodic = tuple(bool(v) for v in topo.periodic_per_axis or ())
        if periodic == (True, False):
            return "Cylinder"
        if periodic == (True, True):
            return "Torus"
        return "EuclideanPatch"
    return None


def _normalize_score_kind(score: str) -> ScoreKind:
    normalized = str(score).strip().lower()
    if normalized not in {"reml", "laml", "bic", "tk"}:
        raise ValueError("score must be one of: 'reml', 'laml', 'bic', 'tk'")
    return normalized  # type: ignore[return-value]


def _normalize_score_scale(score_scale: str) -> ScoreScale:
    normalized = str(score_scale).strip().lower()
    if normalized not in {"per_observation", "per_effective_dim", "raw"}:
        raise ValueError(
            "score_scale must be one of: 'per_observation', "
            "'per_effective_dim', 'raw'"
        )
    return normalized  # type: ignore[return-value]


def _score_for_kind(
    fit_obj: Any,
    score_kind: ScoreKind,
    n_obs: int,
    basis_size: int,
    null_dim: float = 0.0,
) -> float:
    if score_kind == "tk":
        return _extract_reml_score_raw(
            fit_obj
        ) + _tk_normalizer_for_fit(fit_obj, null_dim)
    if score_kind == "reml":
        # Bare raw REML/evidence with no Tierney-Kadane normalizer. The
        # `tk` kind is the only one that adds `_tk_normalizer_for_fit`; the
        # null-space gauge-invariance caveat in `select_topology` applies to
        # that normalizer alone, so `reml` deliberately returns the raw score.
        return _extract_reml_score_raw(fit_obj)
    if score_kind == "laml":
        return _extract_laml_score(
            fit_obj
        ) + _tk_normalizer_for_fit(fit_obj, null_dim)
    return _bic_value(fit_obj, n_obs, basis_size)


def _tk_normalizer_for_fit(fit_obj: Any, null_dim: float) -> float:
    return _tierney_kadane_normalizer_from_null_dim(
        null_dim,
        _extract_null_hessian_logdet(fit_obj),
    )


def _tk_score_from_parts(
    raw_reml: float,
    null_dim: float,
    null_hessian_logdet: float | None,
    effective_dim: float,
    n_obs: int,
    score_scale: TopologyScoreScale,
) -> float:
    raw_tk = raw_reml + _tierney_kadane_normalizer_from_null_dim(
        null_dim,
        null_hessian_logdet,
    )
    if score_scale == "per_observation":
        if n_obs <= 0:
            raise ValueError("TopologyAutoSelector requires n_obs > 0")
        return raw_tk / n_obs
    if score_scale == "per_effective_dim":
        if not math.isfinite(effective_dim) or effective_dim <= 0.0:
            raise ValueError(
                "TopologyAutoSelector requires finite positive effective_dim"
            )
        return raw_tk / effective_dim
    raise ValueError(
        "TopologyAutoSelector score_scale must be 'per_effective_dim' "
        "or 'per_observation'"
    )


def _extract_null_dim(fit_obj: Any) -> float | None:
    return _extract_float_field(fit_obj, ("null_dim",))


def _extract_null_hessian_logdet(fit_obj: Any) -> float | None:
    return _extract_float_field(fit_obj, _NULL_HESSIAN_LOGDET_KEYS)


def _comparison_score(score: float, score_kind: ScoreKind) -> float:
    # `compare_models` sorts its `reml_score` column ascending (lower is the
    # better model; see `gamfit._reml_common.compare_models` /
    # `solver::evidence`, issue #396). Every score kind handled here —
    # REML, LAML, TK, and BIC (deviance + log(n)*k) — is a minimised
    # lower-is-better cost, so all pass through with the SAME orientation.
    # Negating BIC inverted the comparison and selected the WORST topology.
    return float(score)


def _scale_score(
    score: float,
    score_scale: ScoreScale,
    n_obs: int,
    effective_dim: float,
) -> float:
    if score_scale == "raw":
        return float(score)
    if score_scale == "per_observation":
        if n_obs <= 0:
            raise ValueError("per_observation topology scoring requires n_obs > 0")
        return float(score) / float(n_obs)
    if not (math.isfinite(effective_dim) and effective_dim > 0.0):
        raise ValueError(
            "per_effective_dim topology scoring requires finite positive "
            f"effective_dim; got {effective_dim!r}"
        )
    return float(score) / effective_dim


def _extract_laml_score(fit_obj: Any) -> float:
    payload = _summary_payload(fit_obj)
    if payload is not None:
        value = payload.get("laml")
        if value is not None:
            return float(value)
    if isinstance(fit_obj, Mapping):
        value = fit_obj.get("laml")
        if value is not None:
            return float(value)
    raise ValueError(
        "score='laml' requires a real 'laml' field on the fitted result; "
        "REML/evidence must be requested with score='reml'"
    )


def _bic_value(fit_obj: Any, n_obs: int, basis_size: int) -> float:
    return float(
        _topology_rust().topology_bic_score(fit_obj, int(n_obs), int(basis_size))
    )


def _basis_size(fit_obj: Any) -> int:
    payload = _summary_payload(fit_obj)
    if payload is not None:
        coefficients = payload.get("coefficients")
        if isinstance(coefficients, Sequence) and not isinstance(
            coefficients,
            (str, bytes, bytearray),
        ):
            return len(coefficients)
        for key in ("n_coefficients", "n_coeffs", "basis_size"):
            value = payload.get(key)
            if value is not None:
                return int(value)
    if isinstance(fit_obj, Mapping):
        coefficients = fit_obj.get("coefficients")
        shape = getattr(coefficients, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[-1] if len(shape) > 1 else shape[0])
        if isinstance(coefficients, Sequence) and not isinstance(
            coefficients,
            (str, bytes, bytearray),
        ):
            return len(coefficients)
    raise ValueError("select_topology could not determine fitted basis size")


def _effective_dim(fit_obj: Any) -> float:
    payload = _summary_payload(fit_obj)
    if payload is not None:
        value = _first_mapping_value(
            payload,
            ("effective_dim", "effective_dimension", "edf_total", "edf", "effective_dof"),
        )
        if value is not None:
            return _effective_dim_value(value)
    if isinstance(fit_obj, Mapping):
        value = _first_mapping_value(
            fit_obj,
            ("effective_dim", "effective_dimension", "edf_total", "edf", "effective_dof"),
        )
        if value is not None:
            return _effective_dim_value(value)
    for key in ("effective_dim", "effective_dimension", "edf_total", "edf", "effective_dof"):
        value = getattr(fit_obj, key, None)
        if value is not None:
            return _effective_dim_value(value)
    raise ValueError("select_topology could not determine fitted effective_dim")


def _first_mapping_value(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _effective_dim_value(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(sum(value))
    if not math.isfinite(out):
        raise ValueError(f"select_topology effective_dim must be finite; got {out!r}")
    return out


def _extract_float_field(fit_obj: Any, keys: tuple[str, ...]) -> float | None:
    payload = _summary_payload(fit_obj)
    if payload is not None:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return float(value)
    if isinstance(fit_obj, Mapping):
        for key in keys:
            value = fit_obj.get(key)
            if value is not None:
                return float(value)
    for key in keys:
        value = getattr(fit_obj, key, None)
        if value is not None:
            return float(value)
    return None


def _summary_payload(fit_obj: Any) -> Mapping[str, Any] | None:
    summary = getattr(fit_obj, "summary", None)
    if not callable(summary):
        return None
    summary_obj = summary()
    payload = getattr(summary_obj, "payload", summary_obj)
    if isinstance(payload, Mapping):
        return payload
    # ``Model.summary()`` returns a ``gamfit._summary.Summary`` — a frozen
    # dataclass that duck-types the mapping protocol (``__getitem__`` /
    # ``__contains__`` / ``__iter__`` / ``.get`` / ``.to_dict``) but is not a
    # ``collections.abc.Mapping`` instance. Gating on ``isinstance(..., Mapping)``
    # would silently discard the entire summary and break downstream scoring
    # (``_basis_size`` raising "could not determine fitted basis size"). Flatten
    # it through its public ``to_dict`` so every key the engine emits is visible.
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        flattened = to_dict()
        if isinstance(flattened, Mapping):
            return flattened
    # Fall back to the mapping protocol directly for any future summary type
    # that iterates keys + supports subscripting but lacks ``to_dict``.
    if hasattr(payload, "get") and hasattr(payload, "__iter__"):
        try:
            return {str(key): payload[key] for key in payload}
        except (TypeError, KeyError):
            return None
    return None


def _score_disagreement_warnings(
    fits: Mapping[str, Any],
    n_obs: int,
    basis_sizes: Mapping[str, int],
    effective_dim: Mapping[str, float],
    null_dims: Mapping[str, float],
    score_scale: ScoreScale,
) -> list[str]:
    orders: dict[str, tuple[str, ...]] = {}
    for kind in ("reml", "laml", "bic"):
        try:
            scores = {
                name: _scale_score(
                    _score_for_kind(
                        fit_obj,
                        kind,
                        n_obs,
                        basis_sizes[name],
                        null_dims[name],
                    ),
                    score_scale,
                    n_obs,
                    effective_dim[name],
                )
                for name, fit_obj in fits.items()
            }
        except (NotImplementedError, ValueError):
            if kind in {"reml", "laml"}:
                continue
            raise
        comparison = compare_models(
            [{"reml_score": _comparison_score(scores[name], kind)}
             for name in fits],
            names=list(fits),
        )
        orders[kind] = tuple(name for name, *_ in comparison["ranking"])
    if len(orders) < 2:
        return []
    if len(set(orders.values())) == 1:
        return []
    detail = "; ".join(
        f"{kind}: {', '.join(order)}" for kind, order in orders.items()
    )
    if score_scale != "raw":
        return [
            "Scaled topology score rankings still differ across score kinds "
            f"under score_scale={score_scale!r} ({detail}). Treat BIC as a "
            "secondary diagnostic; the Tierney-Kadane Laplace normalizer "
            "handles the known cross-basis evidence scale issue."
        ]
    return [
        "Topology score rankings differ across score kinds "
        f"({detail}). BIC and REML can disagree when candidate basis sizes "
        "differ wildly."
    ]


@dataclass(frozen=True, slots=True)
class TopologyAutoSelectorResult:
    """Ranked latent-topology selector result.

    ``ranked`` is a best-first list of
    ``(name, tk_score, raw_reml, effective_dim, n_obs, model)`` tuples.
    ``winner`` is the selected tuple from that list. ``errors`` records
    candidate-specific fit failures for candidates skipped during ranking.
    """

    ranked: list[TopologyAutoSelectorRank]
    winner: TopologyAutoSelectorRank
    errors: dict[str, str]


class TopologyAutoSelector:
    """Builder for latent-coordinate topology selection.

    Parameters
    ----------
    candidates:
        ``None`` for default candidates, topology-name strings, ``Smooth``
        objects, or ``(name, Smooth)`` pairs.
    score_scale:
        ``"per_effective_dim"`` or ``"per_observation"`` for the Rust
        Tierney-Kadane ranking payload.
    latent:
        Optional latent block name. Required when the ``latents`` mapping
        passed to :meth:`fit` has more than one entry.
    """

    def __init__(
        self,
        candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None = None,
        *,
        score_scale: TopologyScoreScale = "per_effective_dim",
        latent: str | None = None,
    ) -> None:
        self.candidates = candidates
        self.score_scale = _normalize_selector_score_scale(score_scale)
        self.latent = latent

    def with_candidates(
        self,
        candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None,
    ) -> "TopologyAutoSelector":
        """Set candidate topologies and return ``self`` for fluent chaining."""
        self.candidates = candidates
        return self

    def with_score_scale(self, score_scale: TopologyScoreScale) -> "TopologyAutoSelector":
        """Set the TK score scale and return ``self``."""
        self.score_scale = _normalize_selector_score_scale(score_scale)
        return self

    def for_latent(self, latent: str) -> "TopologyAutoSelector":
        """Select which latent block name this selector should rank."""
        self.latent = str(latent)
        return self

    def to_rust_descriptor(self) -> dict[str, Any]:
        """Serialize selector configuration for composition-engine hosts."""
        payload: dict[str, Any] = {"score_scale": self.score_scale}
        if self.candidates is not None:
            payload["candidates"] = [
                _candidate_from_item(item, 1, idx)[0]
                for idx, item in enumerate(self.candidates)
            ]
        if self.latent is not None:
            payload["latent"] = self.latent
        return payload

    _to_rust_payload = to_rust_descriptor

    def fit(
        self,
        data: Any,
        formula: str,
        *,
        latents: Mapping[str, LatentCoord] | None = None,
        penalties: Sequence[Any] | None = None,
        **fit_kwargs: Any,
    ) -> TopologyAutoSelectorResult:
        """Fit all fittable latent-topology candidates and rank them.

        Parameters
        ----------
        data, formula:
            Fit inputs passed to :func:`gamfit.fit`.
        latents:
            Mapping containing the latent block to retopologize. If more than
            one latent is present, configure ``latent=...`` or call
            :meth:`for_latent`.
        penalties:
            Analytic penalties forwarded to each candidate fit.
        **fit_kwargs:
            Additional :func:`gamfit.fit` keyword arguments.

        Returns
        -------
        TopologyAutoSelectorResult
            Ranking tuples, winner tuple, and skipped-candidate errors.

        Raises
        ------
        ValueError
            If no latent is supplied, the requested latent is absent, no
            candidate can be fit, or required TK metadata is missing.
        """
        latent_name, latent = _single_latent(latents, self.latent)
        n_obs = _n_obs(data, latent_name, latent)
        auto = _maybe_auto_smooth(formula)
        normalized = _normalize_selector_candidates(self.candidates, latent.d)

        # Hoist topology-independent shared work above the candidate loop
        # (#869): the table ingestion (normalize_table) and the per-candidate
        # AUTO-formula surgery (_candidate_formula) and per-candidate latent
        # projection (_latent_for_topology) are invariant to the budget cap, so
        # the budget cascade and the survivor refit would otherwise recompute
        # them on every pass. Normalize the table once and cache each
        # candidate's (formula, latent) pair keyed by candidate identity.
        headers, rows, table_kind = normalize_table(data)
        shared_table = PreNormalizedTable(headers, rows, table_kind)
        candidate_inputs: dict[int, tuple[str, Any]] = {}

        def _candidate_inputs_cached(candidate: _Candidate) -> tuple[str, Any]:
            cached = candidate_inputs.get(id(candidate))
            if cached is not None:
                return cached
            built = (
                _candidate_formula(formula, auto, candidate),
                _latent_for_topology(latent, candidate.name),
            )
            candidate_inputs[id(candidate)] = built
            return built

        def _screen_score(
            candidate: _Candidate, stage_kwargs: Mapping[str, Any]
        ) -> float:
            candidate_formula, candidate_latent = _candidate_inputs_cached(candidate)
            model = fit(
                shared_table,
                candidate_formula,
                latents={latent_name: candidate_latent},
                penalties=penalties,
                **stage_kwargs,
            )
            raw_reml = _extract_reml_score_raw(model)
            effective_dim = _effective_dim(model)
            null_dim = _extract_null_dim(model)
            if null_dim is None:
                raise ValueError(
                    "TopologyAutoSelector requires TK null-dimension metadata; "
                    "fit summary is missing null_dim"
                )
            return _tk_score_from_parts(
                float(raw_reml),
                float(null_dim),
                _extract_null_hessian_logdet(model),
                float(effective_dim),
                int(n_obs),
                self.score_scale,
            )

        screened, screen_errors = _screen_candidates_with_budget_cascade(
            normalized,
            fit_kwargs,
            _screen_score,
        )

        if not screened:
            detail = "; ".join(f"{name}: {err}" for name, err in screen_errors.items())
            raise ValueError(
                "TopologyAutoSelector found no fittable topology candidates during screening"
                + (f" ({detail})" if detail else "")
            )
        screened.sort(key=lambda row: (row[0], row[1]))
        survivors = [row[2] for row in screened[:_survivor_count(len(screened))]]

        evidence_inputs: list[dict[str, Any]] = []
        models_by_name: dict[str, Any] = {}
        raw_reml_by_name: dict[str, float] = {}
        effective_dim_by_name: dict[str, float] = {}
        errors: dict[str, str] = {}

        for candidate in survivors:
            try:
                candidate_formula, candidate_latent = _candidate_inputs_cached(candidate)
                model = fit(
                    shared_table,
                    candidate_formula,
                    latents={latent_name: candidate_latent},
                    penalties=penalties,
                    **fit_kwargs,
                )
                raw_reml = _extract_reml_score_raw(model)
                effective_dim = _effective_dim(model)
                null_dim = _extract_null_dim(model)
                if null_dim is None:
                    raise ValueError(
                        "TopologyAutoSelector requires TK null-dimension metadata; "
                        "fit summary is missing null_dim"
                    )
            except Exception as exc:
                errors[candidate.name] = str(exc)
                continue
            models_by_name[candidate.name] = model
            raw_reml_by_name[candidate.name] = float(raw_reml)
            effective_dim_by_name[candidate.name] = float(effective_dim)
            evidence_inputs.append(
                {
                    "name": candidate.name,
                    "raw_reml": float(raw_reml),
                    "null_dim": float(null_dim),
                    "null_space_logdet": _extract_null_hessian_logdet(model),
                    "effective_dim": float(effective_dim),
                    "n_obs": int(n_obs),
                }
            )

        if not evidence_inputs:
            detail = "; ".join(f"{name}: {err}" for name, err in errors.items())
            raise ValueError(
                "TopologyAutoSelector found no fittable topology candidates"
                + (f" ({detail})" if detail else "")
            )

        ranking_json = _topology_rust().rank_topology_candidates(
            json.dumps(
                {
                    "score_scale": self.score_scale,
                    "candidates": evidence_inputs,
                }
            )
        )
        ranking = json.loads(ranking_json)
        ranked: list[TopologyAutoSelectorRank] = []
        for entry in ranking["ranked"]:
            name = entry["name"]
            ranked.append(
                (
                    name,
                    float(entry["tk_score"]),
                    raw_reml_by_name[name],
                    effective_dim_by_name[name],
                    int(entry["n_obs"]),
                    models_by_name[name],
                )
            )
        return TopologyAutoSelectorResult(
            ranked=ranked,
            winner=ranked[ranking["winner_index"]],
            errors=errors,
        )

    select = fit


def _single_latent(
    latents: Mapping[str, LatentCoord] | None,
    requested: str | None,
) -> tuple[str, LatentCoord]:
    if not latents:
        raise ValueError("TopologyAutoSelector requires a Smooth with latent coords")
    if requested is None:
        if len(latents) != 1:
            raise ValueError(
                "TopologyAutoSelector requires exactly one latent coord; "
                "pass latent=... to choose one"
            )
        name, latent = next(iter(latents.items()))
    else:
        if requested not in latents:
            raise ValueError(f"TopologyAutoSelector latent {requested!r} not found")
        name, latent = requested, latents[requested]
    if not isinstance(latent, LatentCoord):
        raise TypeError(
            "TopologyAutoSelector latents entries must be gamfit.LatentCoord"
        )
    return str(name), latent


def _n_obs(data: Any, latent_name: str, latent: LatentCoord) -> int:
    columns, _kind = table_columns(data)
    if not columns:
        raise ValueError("TopologyAutoSelector data cannot be empty")
    first = next(iter(columns.values()))
    n_obs = len(first)
    if n_obs != int(latent.n):
        raise ValueError(
            f"TopologyAutoSelector latent {latent_name!r} has n={latent.n}, "
            f"but data has {n_obs} rows"
        )
    return n_obs


def _maybe_auto_smooth(formula: str) -> tuple[int, int, str] | None:
    try:
        return _find_auto_smooth_call(formula)
    except ValueError:
        return None


def _candidate_formula(
    formula: str,
    auto: tuple[int, int, str] | None,
    candidate: _Candidate,
) -> str:
    if auto is None:
        return formula
    candidate_formula = _formula_for_candidate(
        formula,
        candidate,
        strict_dimension=False,
    )
    if candidate_formula is None:
        required = _candidate_required_dim(candidate.topology)
        raise ValueError(
            f"{candidate.name} is incompatible with this latent smooth"
            + (f" (requires {required}D)" if required is not None else "")
        )
    return candidate_formula


def _normalize_selector_candidates(
    candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None,
    latent_dim: int,
) -> list[_Candidate]:
    raw = list(_DEFAULT_TOPOLOGY_NAMES if candidates is None else candidates)
    if not raw:
        raise ValueError("TopologyAutoSelector requires at least one candidate")
    out: list[_Candidate] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        name, smooth = _candidate_from_item(item, latent_dim, idx)
        key = name.lower()
        if key in seen:
            raise ValueError(f"duplicate topology candidate {name!r}")
        seen.add(key)
        out.append(_Candidate(key, smooth))
    return out


def _candidate_from_item(
    item: str | Smooth | tuple[str, Smooth],
    latent_dim: int,
    idx: int,
) -> tuple[str, Smooth]:
    if isinstance(item, tuple):
        name, smooth = item
        if not isinstance(smooth, Smooth):
            raise TypeError(f"candidate {idx} topology must be a gamfit Smooth")
        return _normalize_topology_name(str(name)), smooth
    if isinstance(item, Smooth):
        name = _infer_candidate_name(item)
        if name is None:
            raise TypeError(f"candidate {idx} is not a supported topology Smooth")
        return _normalize_topology_name(name), item
    name = _normalize_topology_name(str(item))
    return name, _default_topology_candidate(name, latent_dim).topology


def _normalize_topology_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "flat": "euclidean",
        "euclideanpatch": "euclidean",
        "euclidean_patch": "euclidean",
        "periodic": "circle",
        "s1": "circle",
        "s2": "sphere",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _DEFAULT_TOPOLOGY_NAMES:
        raise ValueError(
            "topology candidate must be one of: "
            + ", ".join(_DEFAULT_TOPOLOGY_NAMES)
        )
    return normalized


def _latent_for_topology(latent: LatentCoord, name: str) -> LatentCoord:
    return LatentCoord(
        n=latent.n,
        d=latent.d,
        init=latent.init,
        aux_prior=latent.aux_prior,
        dim_selection=latent.dim_selection,
        manifold=name,
        retraction=getattr(latent, "retraction", "euclidean"),
        name=latent.name,
    )


def _normalize_selector_score_scale(score_scale: str) -> TopologyScoreScale:
    normalized = _normalize_score_scale(score_scale)
    if normalized == "raw":
        raise ValueError(
            "TopologyAutoSelector score_scale must be "
            "'per_effective_dim' or 'per_observation'"
        )
    return normalized


__all__ = [
    "BasisSpec",
    "ScoreKind",
    "ScoreScale",
    "SelectTopologyResult",
    "TopologyAutoSelector",
    "TopologyAutoSelectorRank",
    "TopologyAutoSelectorResult",
    "TopologyName",
    "TopologyScoreScale",
    "select_topology",
]
