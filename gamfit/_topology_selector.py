"""Latent-coordinate topology auto-selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from . import topology
from ._api import fit
from ._compare import (
    _extract_null_dim,
    _extract_null_hessian_logdet,
    _extract_reml_score_raw,
    _tierney_kadane_normalizer_from_null_dim,
    compare_models,
)
from ._select_topology import (
    _Candidate,
    _candidate_required_dim,
    _find_auto_smooth_call,
    _formula_for_candidate,
    _infer_candidate_name,
)
from ._tables import table_columns
from .smooth import LatentCoord, Smooth


TopologyName: TypeAlias = Literal[
    "euclidean", "circle", "sphere", "torus", "cylinder"
]
TopologyScoreScale: TypeAlias = Literal["per_effective_dim", "per_observation"]
TopologyAutoSelectorRank: TypeAlias = tuple[str, float, float, float, int, Any]

_DEFAULT_TOPOLOGIES: tuple[TopologyName, ...] = (
    "euclidean",
    "circle",
    "sphere",
    "torus",
    "cylinder",
)


@dataclass(frozen=True, slots=True)
class TopologyAutoSelectorResult:
    """Ranked latent-topology selector result."""

    ranked: list[TopologyAutoSelectorRank]
    winner: TopologyAutoSelectorRank
    errors: dict[str, str]


class TopologyAutoSelector:
    """Builder for latent-coordinate topology selection."""

    def __init__(
        self,
        candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None = None,
        *,
        score_scale: TopologyScoreScale = "per_effective_dim",
        latent: str | None = None,
    ) -> None:
        self.candidates = candidates
        self.score_scale = _normalize_score_scale(score_scale)
        self.latent = latent

    def with_candidates(
        self,
        candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None,
    ) -> "TopologyAutoSelector":
        self.candidates = candidates
        return self

    def with_score_scale(self, score_scale: TopologyScoreScale) -> "TopologyAutoSelector":
        self.score_scale = _normalize_score_scale(score_scale)
        return self

    def for_latent(self, latent: str) -> "TopologyAutoSelector":
        self.latent = str(latent)
        return self

    def to_rust_descriptor(self) -> dict[str, Any]:
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
        latent_name, latent = _single_latent(latents, self.latent)
        n_obs = _n_obs(data, latent_name, latent)
        auto = _maybe_auto_smooth(formula)
        normalized = _normalize_candidates(self.candidates, latent.d)

        ranks: list[TopologyAutoSelectorRank] = []
        errors: dict[str, str] = {}
        comparison_fits: list[dict[str, float]] = []
        comparison_names: list[str] = []

        for candidate in normalized:
            try:
                candidate_formula = _candidate_formula(formula, auto, candidate)
                candidate_latent = _latent_for_topology(latent, candidate.name)
                model = fit(
                    data,
                    candidate_formula,
                    latents={latent_name: candidate_latent},
                    penalties=penalties,
                    **fit_kwargs,
                )
                raw_reml = _extract_reml_score_raw(model)
                effective_dim = _effective_dim(model)
                tk_score = _tk_normalized_score(
                    model,
                    raw_reml,
                    effective_dim,
                    n_obs,
                    self.score_scale,
                )
                rank = (
                    candidate.name,
                    tk_score,
                    raw_reml,
                    effective_dim,
                    n_obs,
                    model,
                )
            except Exception as exc:
                errors[candidate.name] = str(exc)
                continue
            ranks.append(rank)
            comparison_names.append(candidate.name)
            comparison_fits.append({"reml_score": tk_score, "edf": effective_dim})

        if not ranks:
            detail = "; ".join(f"{name}: {err}" for name, err in errors.items())
            raise ValueError(
                "TopologyAutoSelector found no fittable topology candidates"
                + (f" ({detail})" if detail else "")
            )

        compared = compare_models(comparison_fits, names=comparison_names)
        order = [name for name, *_ in compared["ranking"]]
        by_name = {rank[0]: rank for rank in ranks}
        ranked = [by_name[name] for name in order]
        return TopologyAutoSelectorResult(
            ranked=ranked,
            winner=ranked[0],
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
        auto,
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


def _normalize_candidates(
    candidates: Sequence[str | Smooth | tuple[str, Smooth]] | None,
    latent_dim: int,
) -> list[_Candidate]:
    raw = list(_DEFAULT_TOPOLOGIES if candidates is None else candidates)
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
    return name, _smooth_for_name(name, latent_dim)


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
    if normalized not in _DEFAULT_TOPOLOGIES:
        raise ValueError(
            "topology candidate must be one of: "
            + ", ".join(_DEFAULT_TOPOLOGIES)
        )
    return normalized


def _smooth_for_name(name: str, latent_dim: int) -> Smooth:
    if name == "euclidean":
        return topology.EuclideanPatch(d=int(latent_dim), name="euclidean")
    if name == "circle":
        return topology.Circle(name="circle")
    if name == "sphere":
        return topology.Sphere(name="sphere")
    if name == "torus":
        return topology.Torus(name="torus")
    if name == "cylinder":
        return topology.Cylinder(name="cylinder")
    raise AssertionError(name)


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


def _tk_normalized_score(
    model: Any,
    raw_reml: float,
    effective_dim: float,
    n_obs: int,
    score_scale: TopologyScoreScale,
) -> float:
    null_dim = _extract_null_dim(model)
    if null_dim is None:
        raise ValueError(
            "TopologyAutoSelector requires TK null-dimension metadata; "
            "fit summary is missing null_dim"
        )
    score = raw_reml + _tierney_kadane_normalizer_from_null_dim(
        null_dim,
        _extract_null_hessian_logdet(model),
    )
    if score_scale == "per_observation":
        if n_obs <= 0:
            raise ValueError("TopologyAutoSelector requires n_obs > 0")
        return score / float(n_obs)
    if not (math.isfinite(effective_dim) and effective_dim > 0.0):
        raise ValueError(
            "TopologyAutoSelector requires finite positive effective_dim; "
            f"got {effective_dim!r}"
        )
    return score / effective_dim


def _effective_dim(model: Any) -> float:
    summary = model.summary().payload
    for key in ("effective_dim", "effective_dimension", "edf_total", "edf"):
        value = summary.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float(sum(value))
        if not math.isfinite(out):
            raise ValueError(f"TopologyAutoSelector effective_dim is not finite: {out!r}")
        return out
    raise ValueError("TopologyAutoSelector could not determine effective_dim")


def _normalize_score_scale(score_scale: str) -> TopologyScoreScale:
    normalized = str(score_scale).strip().lower()
    if normalized not in {"per_effective_dim", "per_observation"}:
        raise ValueError(
            "TopologyAutoSelector score_scale must be "
            "'per_effective_dim' or 'per_observation'"
        )
    return normalized  # type: ignore[return-value]


__all__ = [
    "TopologyAutoSelector",
    "TopologyAutoSelectorRank",
    "TopologyAutoSelectorResult",
    "TopologyName",
    "TopologyScoreScale",
]
