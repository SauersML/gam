"""Evidence-based topology selection over common smooth topologies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import topology
from ._api import fit
from ._compare import _extract_edf, _extract_reml_score, compare_models
from ._tables import table_columns
from .smooth import Duchon, PeriodicSplineCurve, Smooth, Sphere


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


BasisSpec = Smooth
ScoreKind = Literal["reml", "laml", "bic"]


@dataclass(frozen=True)
class SelectTopologyResult:
    """Result returned by :func:`select_topology`."""

    winner_name: str
    winner_fit: Any
    scores: dict[str, float]
    rankings: list[tuple[str, float]]
    score_kind: str
    basis_sizes: dict[str, int]
    warnings: list[str]
    fits: dict[str, Any] | None = None


def select_topology(
    data: Any,
    response: str,
    candidates: Sequence[tuple[str, BasisSpec] | Mapping[str, Any]] | None = None,
    *,
    score: ScoreKind = "reml",
    return_fits: bool = False,
    **fit_kwargs: Any,
) -> SelectTopologyResult:
    """Select a topology by fitting candidates and ranking model evidence."""
    score_kind = _normalize_score_kind(score)
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
        reml_score = _extract_reml_score(model)
        if not math.isfinite(reml_score):
            raise ValueError(
                f"select_topology: candidate {candidate.name!r} produced "
                f"degenerate REML score {reml_score!r}"
            )
        fits[candidate.name] = model
        names.append(candidate.name)
        fit_list.append(model)

    basis_sizes = {name: _basis_size(fit_obj) for name, fit_obj in fits.items()}
    selected_scores = {
        name: _score_for_kind(fit_obj, score_kind, n_obs, basis_sizes[name])
        for name, fit_obj in fits.items()
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
    warnings_out = _score_disagreement_warnings(fits, n_obs, basis_sizes)

    return SelectTopologyResult(
        winner_name=winner_name,
        winner_fit=fits[winner_name],
        scores=selected_scores,
        rankings=rankings,
        score_kind=score_kind,
        basis_sizes=basis_sizes,
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
        _Candidate("torus", topology.Torus(centers=12, name="theta_phi")),
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
    if isinstance(topo, Duchon):
        periodic = tuple(bool(v) for v in topo.periodic_per_axis or ())
        if periodic == (True, False):
            return _TopologyTerm(
                "te",
                ("bc=['periodic','natural']", "period=[2*pi, None]"),
                2,
            )
        if periodic == (True, True):
            return _TopologyTerm(
                "te",
                ("bc=['periodic','periodic']", "period=[2*pi, 2*pi]"),
                2,
            )
        options = ["type=duchon"]
        if not has_size and isinstance(topo.centers, int):
            options.append(f"centers={topo.centers}")
        if topo.length_scale is not None:
            options.append(f"length_scale={float(topo.length_scale)!r}")
        required_dim = _centers_dim(topo.centers)
        return _TopologyTerm("s", tuple(options), required_dim)
    raise TypeError(f"unsupported topology candidate {type(topo).__name__}")


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


__all__ = ["select_topology"]
