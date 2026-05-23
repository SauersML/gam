"""Evidence-based topology selection for AUTO smooth terms."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import topology
from ._api import fit
from ._compare import _extract_reml_score, compare_models
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


def select_topology(
    data: Any,
    response: str,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    *,
    return_fits: bool = True,
) -> dict[str, Any]:
    """Select a smooth topology by fitting candidates and ranking REML evidence.

    ``select_topology`` is the one-line "TDA proposes, REML disposes" helper:
    pass a formula containing ``type=AUTO`` in one ``s(...)`` term, or pass a
    plain response name and let the helper build ``response ~ s(features,
    type=AUTO)`` from the table columns. Each candidate topology is converted
    to the corresponding formula term, fitted with :func:`gamfit.fit`, and
    ranked by :func:`gamfit.compare_models`.

    The REML/LAML evidence uses the same Occam-factor machinery exposed by
    ``compare_models``. Audit-revised caveat: cross-fit evidence comparisons
    require identical response transforms, likelihoods, and comparable prior
    normalization; do not compare fits that changed the response scale or
    transformation-normal chart. The workflow follows the neuroscience
    precedent of TDA-suggested topology with REML-selected basis complexity,
    as in mGPLVM-style analyses.

    Parameters
    ----------
    data :
        Tabular data accepted by :func:`gamfit.fit`.
    response :
        Formula with one ``s(..., type=AUTO)`` smooth, or a plain response
        column name. Plain-response mode uses all other table columns as the
        candidate smooth covariates.
    candidates :
        Optional sequence of ``{"name": ..., "topology": ...}`` mappings.
        When omitted, candidates are Circle, Sphere, Torus, Cylinder, and
        EuclideanPatch(d=2).
    return_fits :
        Include the fitted model objects in ``result["fits"]``.

    Returns
    -------
    dict
        ``{"ranking", "winner", "evidence_summary"}``, plus ``"fits"`` when
        ``return_fits=True``. Ranking rows are
        ``(name, reml_score, delta_reml, bayes_factor_vs_best, cv_r2_optional)``.
    """
    normalized = _normalize_candidates(candidates)
    formula = _formula_from_response(data, response)
    auto = _find_auto_smooth_call(formula)
    explicit_candidates = candidates is not None

    fits: dict[str, Any] = {}
    valid_names: list[str] = []
    valid_fits: list[Any] = []
    excluded: list[tuple[str, str]] = []

    for candidate in normalized:
        candidate_formula = _formula_for_candidate(
            formula,
            auto,
            candidate,
            strict_dimension=explicit_candidates,
        )
        if candidate_formula is None:
            excluded.append((candidate.name, "covariate dimension mismatch"))
            continue
        model = fit(data, candidate_formula)
        score = _extract_reml_score(model)
        if not math.isfinite(score):
            excluded.append((candidate.name, f"degenerate REML score {score!r}"))
            continue
        fits[candidate.name] = model
        valid_names.append(candidate.name)
        valid_fits.append(model)

    for name, reason in excluded:
        warnings.warn(
            f"select_topology: excluded {name}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )

    if not valid_fits:
        raise ValueError("select_topology: no candidate topology produced a finite REML fit")

    if len(valid_fits) == 1:
        name = valid_names[0]
        score = _extract_reml_score(valid_fits[0])
        result: dict[str, Any] = {
            "ranking": [(name, score, 0.0, 1.0, None)],
            "winner": name,
            "evidence_summary": "single candidate",
        }
    else:
        compared = compare_models(valid_fits, names=valid_names)
        result = {
            "ranking": [
                (name, score, delta, bf, None)
                for name, score, delta, bf, _edf in compared["ranking"]
            ],
            "winner": compared["winner"],
            "evidence_summary": compared["evidence_summary"],
        }

    if return_fits:
        result["fits"] = fits
    return result


def _normalize_candidates(candidates: Sequence[Mapping[str, Any]] | None) -> list[_Candidate]:
    if candidates is None:
        return [
            _Candidate("Circle", topology.Circle()),
            _Candidate("Sphere", topology.Sphere()),
            _Candidate("Torus", topology.Torus()),
            _Candidate("Cylinder", topology.Cylinder()),
            _Candidate("EuclideanPatch", topology.EuclideanPatch(d=2)),
        ]
    if not candidates:
        raise ValueError("select_topology requires at least one candidate")
    out: list[_Candidate] = []
    seen: set[str] = set()
    for i, spec in enumerate(candidates):
        if not isinstance(spec, Mapping):
            raise TypeError("candidate entries must be mappings with 'name' and 'topology'")
        topo = spec.get("topology")
        if not isinstance(topo, Smooth):
            raise TypeError(f"candidate {i} has no gamfit Smooth topology object")
        name = str(spec.get("name") or _infer_candidate_name(topo) or f"candidate_{i}")
        if name in seen:
            raise ValueError(f"duplicate topology candidate name {name!r}")
        seen.add(name)
        out.append(_Candidate(name, topo))
    return out


def _formula_from_response(data: Any, response: str) -> str:
    text = str(response).strip()
    if "~" in text:
        return text
    columns, _kind = table_columns(data)
    if text not in columns:
        raise ValueError(f"response column {text!r} not found in data")
    features = [name for name in columns if name != text]
    if not features:
        raise ValueError("plain-response select_topology needs at least one feature column")
    return f"{text} ~ s({', '.join(features)}, type=AUTO)"


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
