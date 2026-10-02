"""Evidence-based topology selection over common smooth topologies.

Two public selectors are exposed:

* :func:`select_topology` builds candidate formulas around an
  ``s(..., type=AUTO)`` smooth and ranks fitted models by evidence-like scores.
* :class:`TopologyAutoSelector` is a multi-fit orchestrator for selecting the
  topology of one :class:`gamfit.smooth.LatentCoord` block while preserving the rest
  of the caller's fit configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from ._binding import rust_module
from ._exceptions import map_exception
from ._model import Model
from ._tables import normalize_table, table_columns
from .smooth import (
    Duchon,
    LatentCoord,
    PeriodicSplineCurve,
    Smooth,
    Sphere,
    TensorBSpline,
)



@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str
    topology: Smooth


class _TopologyRustModule(Protocol):
    def select_topology_table(
        self,
        headers: list[str],
        rows: Any,
        candidates_json: str,
        defaults: bool,
        score_kind: str,
        score_scale: str,
        config_json: str | None = None,
        response: str | None = None,
        formula: str | None = None,
        latent: str | None = None,
    ) -> tuple[str, list[tuple[str, bytes]], list[tuple[str, BaseException]]]: ...

    def stacking_weights_from_log_density(
        self,
        names: list[str],
        log_density_rows: list[list[float]],
    ) -> str: ...

    def stack_topologies_gaussian(
        self,
        names: list[str],
        y: list[float],
        means: list[list[float]],
        lowers: list[list[float]],
        uppers: list[list[float]],
        interval_level: float,
    ) -> str: ...

    def stacked_predictive_mean(
        self,
        weights: list[float],
        means: list[list[float]],
    ) -> list[float]: ...

BasisSpec: TypeAlias = Smooth
ScoreKind: TypeAlias = Literal["reml", "laml", "tk"]
ScoreScale: TypeAlias = Literal["per_observation", "raw"]
TopologyName: TypeAlias = Literal[
    "euclidean", "circle", "sphere", "torus", "cylinder"
]
TopologyScoreScale: TypeAlias = Literal["per_observation", "raw"]
TopologyAutoSelectorRank: TypeAlias = tuple[str, float, float, float, int, Any]

_SCORE_KINDS: tuple[ScoreKind, ...] = ("reml", "laml", "tk")

_DEFAULT_TOPOLOGY_NAMES: tuple[TopologyName, ...] = (
    "euclidean",
    "circle",
    "sphere",
    "torus",
    "cylinder",
)


FailureStage: TypeAlias = Literal["assembly", "fit", "evidence"]


@dataclass(frozen=True, slots=True)
class TopologyCandidateFailure:
    """One requested topology that could not enter the evidence ranking."""

    name: str
    stage: FailureStage
    error_type: str
    message: str
    evidence_at_failure: float | None = None
    checkpoint: object | None = None


class TopologySelectionError(ValueError):
    """No topology candidate produced a converged, selectable fit."""

    def __init__(self, failures: Sequence[TopologyCandidateFailure]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{failure.name} [{failure.stage}]: {failure.message}"
            for failure in self.failures
        )
        super().__init__(
            "no topology candidate produced a converged selectable fit"
            + (f" ({detail})" if detail else "")
        )


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
    failures: tuple[TopologyCandidateFailure, ...]
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
        ``"reml"``, ``"laml"``, or ``"tk"``. ``"tk"`` adds the
        Tierney-Kadane null-space normalizer to the raw REML/evidence score.
    score_scale:
        ``"per_observation"`` or ``"raw"``.
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
    selection = _run_selection(
        data,
        _candidate_payloads(
            _explicit_candidates(candidates)
            if candidates is not None
            else _default_candidates(_feature_dim(data, response))
        ),
        defaults=candidates is None,
        score_kind=score,
        score_scale=score_scale,
        fit_kwargs=fit_kwargs,
        response=response,
    )
    ranked = selection.ranking["ranked"]
    winner_name = str(ranked[int(selection.ranking["winner_index"])]["name"])
    return SelectTopologyResult(
        winner_name=winner_name,
        winner_fit=selection.fits[winner_name],
        scores={str(row["name"]): float(row["score"]) for row in ranked},
        rankings=[(str(row["name"]), float(row["score"])) for row in ranked],
        score_kind=score,
        score_scale=score_scale,
        basis_sizes={str(row["name"]): int(row["basis_size"]) for row in ranked},
        effective_dim={str(row["name"]): float(row["effective_dim"]) for row in ranked},
        n_obs={str(row["name"]): int(row["n_obs"]) for row in ranked},
        warnings=[str(warning) for warning in selection.ranking["warnings"]],
        failures=selection.failures,
        fits=selection.fits if return_fits else None,
    )


@dataclass(frozen=True, slots=True)
class _Selection:
    ranking: dict[str, Any]
    fits: dict[str, Model]
    failures: tuple[TopologyCandidateFailure, ...]


def _run_selection(
    data: Any,
    candidate_payloads: list[dict[str, Any]],
    *,
    defaults: bool,
    score_kind: str,
    score_scale: str,
    fit_kwargs: Mapping[str, Any],
    response: str | None = None,
    formula: str | None = None,
    latent: str | None = None,
) -> _Selection:
    """Marshal one selection to the Rust owner (``gam_predict::topology_selection``),
    which fits every candidate and ranks their evidence."""
    from ._api import _fit_request_document

    headers, rows, table_kind = normalize_table(data)
    document = _fit_request_document(fit_kwargs)
    document["training_table_kind"] = table_kind
    try:
        raw, fits, fit_errors = _topology_rust().select_topology_table(
            headers,
            rows,
            json.dumps(candidate_payloads),
            defaults,
            str(score_kind),
            str(score_scale),
            json.dumps(document),
            response,
            formula,
            latent,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    ranking = json.loads(raw)
    errors = dict(fit_errors)
    failures = tuple(
        TopologyCandidateFailure(
            name=str(entry["name"]),
            stage=cast(FailureStage, str(entry["stage"])),
            error_type=str(entry["error_type"]),
            message=str(entry["message"]),
            evidence_at_failure=(
                None
                if entry.get("evidence_at_failure") is None
                else float(entry["evidence_at_failure"])
            ),
            checkpoint=getattr(errors.get(str(entry["name"])), "checkpoint", None),
        )
        for entry in ranking["failed"]
    )
    if ranking["winner_index"] is None:
        raise TopologySelectionError(failures)
    return _Selection(
        ranking=ranking,
        fits={
            str(name): Model(_model_bytes=bytes(model_bytes), _training_table_kind=table_kind)
            for name, model_bytes in fits
        },
        failures=failures,
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

        Each positively-weighted candidate predicts the response-scale mean over
        ``data``; the Rust ``stacked_predictive_mean`` combines the columns with
        the stacking weights. Extra keyword arguments are forwarded to each
        candidate's ``predict``.
        """
        active = [name for name in self.names if self.weights.get(name, 0.0) != 0.0]
        if not active:
            raise ValueError("TopologyStack has no positively-weighted candidate")
        means = [
            _predict_response_mean(self._fits[name], data, **predict_kwargs)
            for name in active
        ]
        return list(
            _topology_rust().stacked_predictive_mean(
                [float(self.weights[name]) for name in active], means
            )
        )


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
    variance in the Rust core, and the σ-recovery quantile, Gaussian log-density
    table, and simplex weight solve all run behind the single Rust
    ``stack_topologies_gaussian`` binding (``gam::solver::topology_stack_gaussian``);
    Python only marshals the predictor's mean/interval columns across the FFI.
    Rows whose recovered σ is non-positive (e.g. fully clamped against the
    response support) carry no Gaussian density and are dropped from that
    candidate's column.
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

    # Marshal each retained candidate's held-out predictive mean and observation
    # interval; the Rust kernel recovers the per-point σ from the interval,
    # forms the Gaussian held-out log-density table, and solves for the stacking
    # weights (the quantile, σ-recovery, log-pdf, and simplex solve are all
    # Rust-side — see `gam::solver::topology_stack_gaussian`).
    means_by_cand: list[list[float]] = []
    lowers_by_cand: list[list[float]] = []
    uppers_by_cand: list[list[float]] = []
    for name in names:
        means, lowers, uppers = _holdout_predictive_interval(
            fits[name], holdout, interval_level
        )
        if len(means) != len(y):
            raise ValueError(
                f"candidate {name!r} predicted {len(means)} rows for a "
                f"{len(y)}-row holdout fold"
            )
        means_by_cand.append(means)
        lowers_by_cand.append(lowers)
        uppers_by_cand.append(uppers)

    raw = _topology_rust().stack_topologies_gaussian(
        list(names),
        y,
        means_by_cand,
        lowers_by_cand,
        uppers_by_cand,
        interval_level,
    )
    parsed = json.loads(raw)
    weights = {name: float(parsed["weights"].get(name, 0.0)) for name in names}
    return TopologyStack(
        weights=weights,
        mean_log_score=float(parsed["mean_log_score"]),
        names=names,
        _fits=dict(fits),
    )


def _holdout_predictive_interval(
    model: Any,
    holdout: Any,
    interval_level: float,
) -> tuple[list[float], list[float], list[float]]:
    """Per-point response-scale predictive mean and family-correct observation
    interval ``[lower, upper]`` (``μ ± z·σ`` with ``σ² = Var(μ̂) + Var(Y|μ)``) on
    the held-out fold, sourced verbatim from the Rust predictor. The interval is
    passed straight through to the Rust stacking kernel, which recovers ``σ`` and
    the Gaussian log-density; no scoring math runs here."""
    prediction = model.predict(
        holdout,
        interval=interval_level,
        observation_interval=True,
        return_type="dict",
    )
    means = [float(value) for value in prediction["posterior_mean"]]
    lower = [float(value) for value in prediction["observation_lower"]]
    upper = [float(value) for value in prediction["observation_upper"]]
    return means, lower, upper


def _predict_response_mean(model: Any, data: Any, **predict_kwargs: Any) -> list[float]:
    """The ``posterior_mean`` column, the same point the held-out stack scored."""
    prediction = model.predict(data, return_type="dict", **predict_kwargs)
    return [float(value) for value in prediction["posterior_mean"]]


def _feature_dim(data: Any, response: str) -> int:
    """How many predictor columns the AUTO smooth over every non-response column has,
    which the default portfolio's constructors take."""
    columns, _kind = table_columns(data)
    return sum(1 for name in columns if name != str(response).strip())


def _explicit_candidates(
    candidates: Sequence[tuple[str, BasisSpec] | Mapping[str, Any]],
) -> list[_Candidate]:
    out: list[_Candidate] = []
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
        out.append(_Candidate(name, topo))
    return out


def _candidate_payloads(candidates: Sequence[_Candidate]) -> list[dict[str, Any]]:
    return [
        {"name": candidate.name, "topology": _candidate_to_rust_payload(candidate)}
        for candidate in candidates
    ]


def _default_candidates(feature_dim: int) -> list[_Candidate]:
    return [
        _default_topology_candidate(name, feature_dim)
        for name in _DEFAULT_TOPOLOGY_NAMES
    ]


def _default_topology_candidate(name: str, feature_dim: int) -> _Candidate:
    # `gamfit.topology` re-exports this module, so it is bound at call time.
    from . import topology

    if name == "euclidean":
        return _Candidate("euclidean", topology.EuclideanPatch(d=feature_dim, name="x"))
    if name == "circle":
        return _Candidate("circle", topology.Circle(name="theta"))
    if name == "sphere":
        return _Candidate("sphere", topology.Sphere(name="omega"))
    if name == "torus":
        return _Candidate("torus", topology.Torus(name="theta_phi"))
    if name == "cylinder":
        return _Candidate("cylinder", topology.Cylinder(name="cyl"))
    raise AssertionError(name)


def _topology_rust() -> _TopologyRustModule:
    return cast(_TopologyRustModule, rust_module())


def _candidate_to_rust_payload(candidate: _Candidate) -> dict[str, Any]:
    """Translate a Python `Smooth` topology into the typed JSON shape the
    Rust formula assembler consumes.
    """
    topo = candidate.topology
    double_penalty = (
        None if topo.double_penalty is None else bool(topo.double_penalty)
    )
    if isinstance(topo, PeriodicSplineCurve):
        return {
            "kind": "periodic_spline_curve",
            "n_knots": int(topo.n_knots),
            "degree": int(topo.degree),
            "penalty_order": int(topo.penalty_order),
            "double_penalty": double_penalty,
        }
    if isinstance(topo, Sphere):
        return {
            "kind": "sphere",
            "n_centers": int(topo.n_centers),
            "penalty_order": int(topo.penalty_order),
            "kernel": str(topo.kernel),
            "radians": bool(topo.radians),
            "double_penalty": double_penalty,
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
            "double_penalty": double_penalty,
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
            "double_penalty": double_penalty,
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
        return "circle"
    if isinstance(topo, Sphere):
        return "sphere"
    if isinstance(topo, TensorBSpline):
        periodic = tuple(bool(marginal.periodic) for marginal in topo.marginals)
        if periodic == (True, False):
            return "cylinder"
        if periodic == (True, True):
            return "torus"
    if isinstance(topo, Duchon):
        periodic = tuple(bool(v) for v in topo.periodic_per_axis or ())
        if periodic == (True, False):
            return "cylinder"
        if periodic == (True, True):
            return "torus"
        return "euclidean"
    return None


@dataclass(frozen=True, slots=True)
class TopologyAutoSelectorResult:
    """Ranked latent-topology selector result.

    ``ranked`` is a best-first list of
    ``(name, tk_score, raw_reml, effective_dim, n_obs, model)`` tuples.
    ``winner`` is the selected tuple from that list. ``failures`` retains every
    requested candidate that could not enter the ranking.
    """

    ranked: list[TopologyAutoSelectorRank]
    winner: TopologyAutoSelectorRank
    failures: tuple[TopologyCandidateFailure, ...]


class TopologyAutoSelector:
    """Builder for latent-coordinate topology selection.

    Parameters
    ----------
    candidates:
        ``None`` for default candidates, topology-name strings, ``Smooth``
        objects, or ``(name, Smooth)`` pairs.
    score_scale:
        ``"per_observation"`` or ``"raw"`` for the Rust Tierney-Kadane ranking.
    latent:
        Optional latent block name. Required when the ``latents`` mapping
        passed to :meth:`fit` has more than one entry.
    """

    def __init__(
        self,
        candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None = None,
        *,
        score_scale: TopologyScoreScale = "per_observation",
        latent: str | None = None,
    ) -> None:
        self.candidates = candidates
        self.score_scale = score_scale
        self.latent = latent

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
            one latent is present, configure ``latent=...``.
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
        selection = _run_selection(
            data,
            _candidate_payloads(_normalize_selector_candidates(self.candidates, latent.d)),
            defaults=False,
            score_kind="tk",
            score_scale=self.score_scale,
            fit_kwargs={"latents": latents, "penalties": penalties, **fit_kwargs},
            formula=formula,
            latent=latent_name,
        )
        ranked: list[TopologyAutoSelectorRank] = [
            (
                str(entry["name"]),
                float(entry["score"]),
                float(entry["raw_reml"]),
                float(entry["effective_dim"]),
                int(entry["n_obs"]),
                selection.fits[str(entry["name"])],
            )
            for entry in selection.ranking["ranked"]
        ]
        return TopologyAutoSelectorResult(
            ranked=ranked,
            winner=ranked[int(selection.ranking["winner_index"])],
            failures=selection.failures,
        )


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
            "TopologyAutoSelector latents entries must be gamfit.smooth.LatentCoord"
        )
    return str(name), latent


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
        key = name
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
        inferred = _infer_candidate_name(item)
        if inferred is None:
            raise TypeError(f"candidate {idx} is not a supported topology Smooth")
        return _normalize_topology_name(inferred), item
    name = _normalize_topology_name(str(item))
    return name, _default_topology_candidate(name, latent_dim).topology


def _normalize_topology_name(name: str) -> str:
    if name not in _DEFAULT_TOPOLOGY_NAMES:
        raise ValueError(
            "topology candidate must be an exact canonical name: "
            + ", ".join(_DEFAULT_TOPOLOGY_NAMES)
        )
    return name


__all__ = [
    "BasisSpec",
    "ScoreKind",
    "ScoreScale",
    "SelectTopologyResult",
    "TopologyCandidateFailure",
    "TopologyAutoSelector",
    "TopologyAutoSelectorRank",
    "TopologyAutoSelectorResult",
    "TopologyName",
    "TopologySelectionError",
    "TopologyScoreScale",
    "select_topology",
]
