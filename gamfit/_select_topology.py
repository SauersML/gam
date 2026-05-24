"""Evidence-based topology selection over common smooth topologies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from . import topology
from ._api import fit
from ._compare import (
    _extract_edf,
    _extract_null_dim,
    _extract_null_hessian_logdet,
    _extract_reml_score_raw,
    _tierney_kadane_normalizer_from_null_dim,
    compare_models,
)
from ._tables import table_columns
from .smooth import BSpline, Duchon, PeriodicSplineCurve, Smooth, Sphere, TensorBSpline


_AUTO_RE = re.compile(r"\btype\s*=\s*(['\"]?)AUTO\1(?=\s*(?:,|\)))", re.IGNORECASE)
_SMOOTH_CALL_RE = re.compile(r"\b(s|smooth)\s*\(", re.IGNORECASE)
_SIZE_OPTION_KEYS = {"k", "basis_dim", "basis-dim", "basisdim", "centers", "knots"}


@dataclass(frozen=True)
class _Candidate:
    name: str
    topology: Smooth


@dataclass(frozen=True)
class _TopologyTerm:
    call: str
    options: tuple[str, ...]
    required_dim: int | None


BasisSpec: TypeAlias = Smooth
ScoreKind: TypeAlias = Literal["reml", "laml", "bic"]
ScoreScale: TypeAlias = Literal["per_observation", "per_effective_dim", "raw"]


@dataclass(frozen=True)
class SelectTopologyResult:
    """Result returned by :func:`select_topology`."""

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
    """Select a topology by fitting candidates and ranking model evidence."""
    score_kind = _normalize_score_kind(score)
    score_scale_kind = _normalize_score_scale(score_scale)
    formula, feature_dim, n_obs = _formula_from_response(data, response)
    normalized = _normalize_candidates(candidates, feature_dim=feature_dim)
    auto = _find_auto_smooth_call(formula)

    fits: dict[str, Any] = {}
    names: list[str] = []
    fit_list: list[Any] = []

    for candidate in normalized:
        candidate_formula = _formula_for_candidate(
            formula,
            auto,
            candidate,
            strict_dimension=True,
        )
        if candidate_formula is None:  # defensive; strict_dimension=True raises.
            raise ValueError(f"candidate {candidate.name!r} is not constructible")
        model = fit(data, candidate_formula, **fit_kwargs)
        reml_score = _extract_reml_score_raw(model)
        if not math.isfinite(reml_score):
            raise ValueError(
                f"select_topology: candidate {candidate.name!r} produced "
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
        candidate.name: _fitted_or_candidate_null_dim(
            fits[candidate.name],
            candidate,
            basis_sizes[candidate.name],
        )
        for candidate in normalized
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
        _Candidate("euclidean", topology.EuclideanPatch(d=feature_dim, name="x")),
        _Candidate("circle", topology.Circle(name="theta")),
        _Candidate("sphere", topology.Sphere(dim=2, name="omega")),
        _Candidate("torus", topology.Torus(n_knots=(12, 12), name="theta_phi")),
        _Candidate("cylinder", topology.Cylinder(name="cyl")),
    ]
    return [
        candidate for candidate in candidates
        if _candidate_required_dim(candidate.topology) in {None, feature_dim}
    ]


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
    match = _SMOOTH_CALL_RE.search(formula)
    while match is not None:
        open_paren = match.end() - 1
        close_paren = _matching_paren(formula, open_paren)
        term = formula[match.start() : close_paren + 1]
        if _AUTO_RE.search(term):
            return match.start(), close_paren + 1, term
        match = _SMOOTH_CALL_RE.search(formula, close_paren + 1)
    raise ValueError("select_topology requires one s(..., type=AUTO) smooth term")


def _matching_paren(text: str, open_paren: int) -> int:
    depth = 1
    quote: str | None = None
    i = open_paren + 1
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("select_topology: AUTO smooth has unbalanced parentheses")


def _split_top_level_args(arg_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(arg_text):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(arg_text[start:i].strip())
            start = i + 1
    tail = arg_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _auto_call_parts(term: str) -> tuple[list[str], list[str], set[str]]:
    open_paren = term.index("(")
    arg_text = term[open_paren + 1 : -1]
    args = _split_top_level_args(arg_text)
    vars: list[str] = []
    options: list[str] = []
    option_keys: set[str] = set()
    for arg in args:
        key = _named_arg_key(arg)
        if key is None:
            vars.append(arg)
            continue
        if key == "type":
            continue
        option_keys.add(key)
        if key in {"periodic", "cyclic", "bc", "period", "periods", "origin", "origins"}:
            continue
        options.append(arg)
    if not vars:
        raise ValueError("select_topology: AUTO smooth must have at least one covariate")
    return vars, options, option_keys


def _named_arg_key(arg: str) -> str | None:
    depth = 0
    quote: str | None = None
    for ch in arg:
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return arg.split("=", 1)[0].strip().lower()
    return None


def _formula_for_candidate(
    formula: str,
    auto: tuple[int, int, str],
    candidate: _Candidate,
    *,
    strict_dimension: bool,
) -> str | None:
    start, end, term = auto
    vars, user_options, option_keys = _auto_call_parts(term)
    topo_term = _topology_term(candidate, option_keys)
    if topo_term.required_dim is not None and len(vars) != topo_term.required_dim:
        message = (
            f"{candidate.name} needs {topo_term.required_dim}-D covariate; "
            f"AUTO smooth has {len(vars)} covariate"
            f"{'' if len(vars) == 1 else 's'} ({', '.join(vars)})"
        )
        if strict_dimension:
            raise ValueError(message)
        return None
    candidate_args = vars + user_options + list(topo_term.options)
    candidate_term = f"{topo_term.call}({', '.join(candidate_args)})"
    return formula[:start] + candidate_term + formula[end:]


def _topology_term(candidate: _Candidate, option_keys: set[str]) -> _TopologyTerm:
    topo = candidate.topology
    has_size = bool(_SIZE_OPTION_KEYS & option_keys)
    if isinstance(topo, PeriodicSplineCurve):
        options = ["type=cyclic"]
        if not has_size:
            options.append(f"k={topo.n_knots}")
        if topo.degree != 3:
            options.append(f"degree={topo.degree}")
        if topo.penalty_order != 2:
            options.append(f"penalty_order={topo.penalty_order}")
        return _TopologyTerm("s", tuple(options), 1)
    if isinstance(topo, Sphere):
        options = ["type=sphere"]
        if not has_size:
            options.append(f"centers={topo.n_centers}")
        if topo.penalty_order != 2:
            options.append(f"penalty_order={topo.penalty_order}")
        if topo.kernel != "sobolev":
            options.append(f"kernel={_quote(topo.kernel)}")
        if topo.radians:
            options.append("radians=true")
        return _TopologyTerm("s", tuple(options), 2)
    if isinstance(topo, TensorBSpline):
        options: list[str] = []
        k = getattr(topo, "_gamfit_tensor_k", None)
        if not has_size and k is not None:
            options.append(f"k={_format_int_list(k)}")
        periodic = tuple(bool(marginal.periodic) for marginal in topo.marginals)
        if any(periodic):
            options.append(f"periodic={_format_bool_list(periodic)}")
            periods = getattr(
                topo,
                "_gamfit_tensor_periods",
                tuple("2*pi" if value else None for value in periodic),
            )
            options.append(f"period={_format_period_list(periods)}")
        options.append("identifiability=sum_tozero")
        return _TopologyTerm("te", tuple(options), len(topo.marginals))
    if isinstance(topo, Duchon):
        periodic = tuple(bool(v) for v in topo.periodic_per_axis or ())
        if periodic:
            raise ValueError(
                "select_topology cannot fit per-axis periodic Duchon candidates "
                "through the formula AUTO path; use topology.Cylinder or "
                "topology.Torus tensor candidates"
            )
        options = ["type=duchon", f"order={_duchon_formula_order(topo)}"]
        if not has_size and isinstance(topo.centers, int):
            options.append(f"centers={topo.centers}")
        if topo.length_scale is not None:
            options.append(f"length_scale={float(topo.length_scale)!r}")
        required_dim = _candidate_required_dim(topo)
        return _TopologyTerm("s", tuple(options), required_dim)
    raise TypeError(f"unsupported topology candidate {type(topo).__name__}")


def _format_int_list(values: Sequence[Any]) -> str:
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def _format_bool_list(values: Sequence[bool]) -> str:
    return "[" + ", ".join("true" if value else "false" for value in values) + "]"


def _format_period_list(values: Sequence[Any]) -> str:
    parts = ["None" if value is None else str(value) for value in values]
    return "[" + ", ".join(parts) + "]"


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


def _fitted_or_candidate_null_dim(
    fit_obj: Any, candidate: _Candidate, basis_size: int
) -> float:
    null_dim = _extract_null_dim(fit_obj)
    if null_dim is not None:
        return null_dim
    return _candidate_null_dim(candidate, basis_size)


def _candidate_null_dim(candidate: _Candidate, basis_size: int) -> float:
    topo = candidate.topology
    if getattr(topo, "double_penalty", False):
        return 0.0
    if isinstance(topo, PeriodicSplineCurve):
        return 1.0
    if isinstance(topo, Sphere):
        return 1.0
    if isinstance(topo, TensorBSpline):
        null_dim = 1
        for marginal in topo.marginals:
            null_dim *= _bspline_marginal_nullity(marginal)
        identifiability = str(getattr(topo, "_gamfit_tensor_identifiability", "sum_tozero"))
        if identifiability.lower().replace("-", "_") != "none":
            null_dim = max(null_dim - 1, 0)
        return float(min(max(null_dim, 0), basis_size))
    if isinstance(topo, Duchon):
        periodic = tuple(bool(v) for v in topo.periodic_per_axis or ())
        if periodic and all(periodic):
            return 1.0
        dim = _candidate_required_dim(topo)
        if dim is None:
            dim = _centers_dim(topo.centers)
        if dim is None:
            dim = 1
        nonperiodic_dim = sum(1 for value in periodic if not value) if periodic else dim
        null_dim = math.comb(
            nonperiodic_dim + _duchon_formula_order(topo),
            nonperiodic_dim,
        )
        return float(min(max(null_dim, 0), basis_size))
    return 0.0


def _bspline_marginal_nullity(marginal: BSpline) -> int:
    return 1 if marginal.periodic else max(0, int(marginal.penalty_order))


def _duchon_formula_order(topo: Duchon) -> int:
    return max(0, int(topo.m) - 1)


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


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _normalize_score_kind(score: str) -> ScoreKind:
    normalized = str(score).strip().lower()
    if normalized not in {"reml", "laml", "bic"}:
        raise ValueError("score must be one of: 'reml', 'laml', 'bic'")
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
    if score_kind == "reml":
        return _extract_reml_score_raw(
            fit_obj
        ) + _tk_normalizer_for_fit(fit_obj, null_dim)
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


def _comparison_score(score: float, score_kind: ScoreKind) -> float:
    return -float(score) if score_kind == "bic" else float(score)


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
    if n_obs <= 1:
        raise ValueError("BIC scoring requires at least two observations")
    deviance = _extract_float_field(fit_obj, ("deviance",))
    if deviance is None:
        raise ValueError("BIC scoring requires fit.summary()['deviance']")
    return float(deviance) + math.log(float(n_obs)) * float(basis_size)


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
    if callable(summary):
        summary_obj = summary()
        payload = getattr(summary_obj, "payload", summary_obj)
        if isinstance(payload, Mapping):
            return payload
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
        f"({detail}). See memory "
        "`project_gumbel_anneal_population_sparsity_falsified`: BIC and REML "
        "can disagree when candidate basis sizes differ wildly."
    ]


__all__ = [
    "BasisSpec",
    "ScoreKind",
    "ScoreScale",
    "SelectTopologyResult",
    "select_topology",
]
