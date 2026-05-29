"""Response-geometry transforms (simplex + sphere) — thin FFI shims.

Numerical work (closure, CLR/ALR, Fréchet means, log/exp maps, fisher_rao_w
validation) lives in `gam-pyffi`. This module marshals NumPy arrays across the
boundary and hosts the small dataclasses that hold a fitted shared-tangent
model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ._binding import rust_module
from ._exceptions import map_exception
from ._tables import normalize_table, restore_output_table, table_columns


def _ffi(name: str, *args: Any) -> Any:
    try:
        return getattr(rust_module(), name)(*args)
    except Exception as exc:
        raise map_exception(exc) from exc


def _np():
    import numpy as np

    return np


def closure(values: Any) -> Any:
    """Normalize rows onto the probability simplex."""
    np = _np()
    return _ffi("response_geometry_closure", np.asarray(values, dtype=float))


def clr(values: Any) -> Any:
    """Centered log-ratio coordinates for positive compositions."""
    np = _np()
    return _ffi("response_geometry_clr", np.asarray(values, dtype=float))


def alr(values: Any, *, reference: int = -1) -> Any:
    """Additive log-ratio coordinates for positive compositions."""
    np = _np()
    return _ffi("response_geometry_alr", np.asarray(values, dtype=float), int(reference))


def inverse_alr(coords: Any, *, reference: int = -1) -> Any:
    """Map ALR coordinates back to the simplex."""
    np = _np()
    return _ffi(
        "response_geometry_inverse_alr", np.asarray(coords, dtype=float), int(reference)
    )


def simplex_frechet_mean(values: Any, weights: Any | None = None) -> Any:
    """Intrinsic Fréchet mean under Aitchison simplex geometry."""
    np = _np()
    w = None if weights is None else np.asarray(weights, dtype=float)
    return _ffi(
        "response_geometry_simplex_frechet_mean",
        np.asarray(values, dtype=float),
        w,
    )


def simplex_log_map(
    values: Any, base: Any, *, coordinates: str = "clr", reference: int = -1
) -> Any:
    """Log map at an intrinsic simplex base point in CLR or ALR coordinates."""
    np = _np()
    return _ffi(
        "response_geometry_simplex_log_map",
        np.asarray(values, dtype=float),
        np.asarray(base, dtype=float).reshape(-1),
        str(coordinates),
        int(reference),
    )


def simplex_exp_map(
    tangent: Any, base: Any, *, coordinates: str = "clr", reference: int = -1
) -> Any:
    """Exponential map from simplex tangent coordinates back to compositions."""
    np = _np()
    z = np.asarray(tangent, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    return _ffi(
        "response_geometry_simplex_exp_map",
        z,
        np.asarray(base, dtype=float).reshape(-1),
        str(coordinates),
        int(reference),
    )


def sphere_frechet_mean(
    values: Any,
    weights: Any | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> Any:
    """Intrinsic Fréchet/Karcher mean on the unit sphere."""
    np = _np()
    w = None if weights is None else np.asarray(weights, dtype=float)
    return _ffi(
        "sphere_frechet_mean",
        np.asarray(values, dtype=float),
        w,
        float(tol),
        int(max_iter),
    )


def sphere_log_map(values: Any, base: Any) -> Any:
    """Log map from the unit sphere to the tangent space at ``base``."""
    np = _np()
    return _ffi(
        "response_geometry_sphere_log_map",
        np.asarray(values, dtype=float),
        np.asarray(base, dtype=float).reshape(-1),
    )


def sphere_exp_map(tangent: Any, base: Any) -> Any:
    """Exponential map from the ambient tangent space at ``base`` to the sphere."""
    np = _np()
    z = np.asarray(tangent, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    return _ffi(
        "response_geometry_sphere_exp_map",
        z,
        np.asarray(base, dtype=float).reshape(-1),
    )


def response_matrix_from_table(data: Any, response_columns: Sequence[str]) -> Any:
    columns, _kind = table_columns(data)
    missing = [name for name in response_columns if name not in columns]
    if missing:
        raise ValueError(f"response geometry columns missing from data: {missing}")
    np = _np()
    return np.column_stack(
        [np.asarray(columns[name], dtype=float) for name in response_columns]
    )


_SIMPLEX_KINDS = {"simplex", "clr", "alr"}
_SPHERE_KINDS = {"spherical", "sphere"}


def _resolve_simplex_coord(kind: str, coordinates: str | None) -> str:
    if coordinates is not None:
        return coordinates.lower()
    return "alr" if kind == "alr" else "clr"


def geometry_log_map(
    values: Any,
    *,
    geometry: str,
    base: Any | None = None,
    coordinates: str | None = None,
    reference: int = -1,
) -> tuple[Any, Any, str]:
    np = _np()
    kind = geometry.lower()
    if kind in _SPHERE_KINDS:
        if base is None:
            base_point = sphere_frechet_mean(values)
        else:
            # Normalize a single base row via the sphere_log_map FFI by mapping
            # the base to itself: the impl normalizes its base argument internally.
            base_point = _ffi(
                "response_geometry_sphere_normalize_base",
                np.asarray(base, dtype=float).reshape(-1),
            )
        return sphere_log_map(values, base_point), base_point, "spherical"
    if kind in _SIMPLEX_KINDS:
        coord = _resolve_simplex_coord(kind, coordinates)
        if base is None:
            base_point = simplex_frechet_mean(values)
        else:
            base_point = closure(np.asarray(base, dtype=float).reshape(1, -1))[0]
        return (
            simplex_log_map(values, base_point, coordinates=coord, reference=reference),
            base_point,
            coord,
        )
    raise ValueError(
        "response_geometry must be one of 'spherical', 'simplex', 'clr', or 'alr'"
    )


def geometry_exp_map(
    tangent: Any,
    *,
    geometry: str,
    base: Any,
    coordinates: str | None = None,
    reference: int = -1,
) -> Any:
    kind = geometry.lower()
    if kind in _SPHERE_KINDS:
        return sphere_exp_map(tangent, base)
    if kind in _SIMPLEX_KINDS:
        coord = _resolve_simplex_coord(kind, coordinates)
        return simplex_exp_map(tangent, base, coordinates=coord, reference=reference)
    raise ValueError(
        "response_geometry must be one of 'spherical', 'simplex', 'clr', or 'alr'"
    )


@dataclass(slots=True)
class SharedGaussianRemlTangentFit:
    template_model: Any
    coefficients: Any
    fit: dict[str, Any]

    @property
    def tangent_dimension(self) -> int:
        return int(self.coefficients.shape[1])

    def predict_tangent(self, data: Any) -> Any:
        x = self.template_model.design_matrix(data)
        return x @ self.coefficients

    def summary(self) -> dict[str, Any]:
        return {
            "model_class": "joint-tangent-gaussian-reml",
            "reml_score": float(self.fit["reml_score"]),
            "lambdas": self.fit["lambdas"].tolist(),
            "edf": self.fit["edf"].tolist(),
            "sigma2": self.fit["sigma2"].tolist(),
            "template": self.template_model.summary(),
        }


@dataclass(slots=True)
class ResponseGeometryModel:
    """A fitted response-geometry GAM with shared smoothing across tangent coordinates."""

    models: Sequence[Any]
    response_geometry: str
    response_columns: tuple[str, ...]
    base_point: Any
    coordinates: str
    reference: int = -1
    training_table_kind: str | None = None
    shared_tangent_fit: SharedGaussianRemlTangentFit | None = None

    @property
    def tangent_dimension(self) -> int:
        if self.shared_tangent_fit is not None:
            return self.shared_tangent_fit.tangent_dimension
        return len(self.models)

    def predict(
        self,
        data: Any,
        *,
        return_type: str | None = None,
        include_tangent: bool = False,
        **kwargs: Any,
    ) -> Any:
        np = _np()
        _columns, input_kind = table_columns(data)
        if self.shared_tangent_fit is not None:
            tangent = np.asarray(self.shared_tangent_fit.predict_tangent(data), dtype=float)
        else:
            tangent_cols: list[Any] = []
            for model in self.models:
                pred = model.predict(data, return_type="dict", **kwargs)
                tangent_cols.append(np.asarray(pred["mean"], dtype=float))
            tangent = np.column_stack(tangent_cols)
        response = geometry_exp_map(
            tangent,
            geometry=self.response_geometry,
            base=self.base_point,
            coordinates=self.coordinates,
            reference=self.reference,
        )
        out: dict[str, list[Any]] = {
            name: response[:, idx].tolist() for idx, name in enumerate(self.response_columns)
        }
        if include_tangent:
            for idx in range(tangent.shape[1]):
                out[f"tangent_{idx}"] = tangent[:, idx].tolist()
        return restore_output_table(
            out,
            requested=return_type,
            input_kind=input_kind,
            training_kind=self.training_table_kind,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_class": "response-geometry",
            "response_geometry": self.response_geometry,
            "response_columns": list(self.response_columns),
            "base_point": list(map(float, self.base_point)),
            "coordinates": self.coordinates,
            "tangent_dimension": self.tangent_dimension,
            "shared_smoothing": self.shared_tangent_fit is not None,
            "coordinate_summaries": [m.summary() for m in self.models],
            "shared_fit": None
            if self.shared_tangent_fit is None
            else self.shared_tangent_fit.summary(),
        }


def fit_response_geometry(
    fit_func: Any,
    data: Any,
    formula: str,
    *,
    response_geometry: str,
    response_columns: Sequence[str],
    coordinates: str | None = None,
    reference: int = -1,
    weights: str | None = None,
    fisher_rao_w: Any | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> ResponseGeometryModel:
    columns, table_kind = table_columns(data)
    y = response_matrix_from_table(data, response_columns)
    tangent, base, resolved_coordinates = geometry_log_map(
        y,
        geometry=response_geometry,
        coordinates=coordinates,
        reference=reference,
    )
    rhs = _formula_rhs(formula)
    kwargs = dict(fit_kwargs or {})
    kwargs["family"] = "gaussian"
    kwargs["link"] = "identity"
    kwargs["transformation_normal"] = None
    kwargs["survival_likelihood"] = None
    kwargs["z_column"] = None
    kwargs["logslope_formula"] = None
    kwargs["frailty_kind"] = None
    kwargs["frailty_sd"] = None
    kwargs["hazard_loading"] = None
    fisher_w = None
    if fisher_rao_w is not None:
        np = _np()
        fisher_w = _ffi(
            "response_geometry_normalize_fisher_rao",
            np.asarray(fisher_rao_w, dtype=float),
            int(tangent.shape[0]),
            int(tangent.shape[1]),
        )
    kwargs.pop("fisher_rao_w", None)
    target = "__gamfit_response_geometry_shared"
    if target in columns:
        raise ValueError(f"response geometry reserved column already exists: {target}")
    augmented = {name: list(values) for name, values in columns.items()}
    augmented[target] = tangent[:, 0].tolist()
    template_formula = f"{target} ~ {rhs}"
    template_model = fit_func(augmented, template_formula, **kwargs)
    shared_fit = _fit_shared_tangent_reml(
        augmented, template_formula, tangent, kwargs, fisher_w
    )
    return ResponseGeometryModel(
        models=(),
        response_geometry=response_geometry.lower(),
        response_columns=tuple(response_columns),
        base_point=base,
        coordinates=resolved_coordinates,
        reference=reference,
        training_table_kind=table_kind,
        shared_tangent_fit=SharedGaussianRemlTangentFit(
            template_model=template_model,
            coefficients=shared_fit["coefficients"],
            fit=shared_fit,
        ),
    )


def _formula_rhs(formula: str) -> str:
    parts = formula.split("~", 1)
    if len(parts) != 2 or not parts[1].strip():
        raise ValueError(
            "response-geometry formula must have the form 'response ~ terms'"
        )
    return parts[1].strip()


def _fit_shared_tangent_reml(
    data: Any,
    formula: str,
    tangent: Any,
    fit_kwargs: dict[str, Any],
    fisher_rao_w: Any | None = None,
) -> dict[str, Any]:
    import json

    np = _np()
    if fit_kwargs.get("offset") is not None:
        raise ValueError("response geometry shared REML does not support offsets")
    headers, rows, _kind = normalize_table(data)
    config = {
        "family": "gaussian",
        "link": "identity",
        "weights": fit_kwargs.get("weights"),
    }
    y = np.ascontiguousarray(np.asarray(tangent, dtype=float))
    payload = _ffi(
        "gaussian_reml_fit_formula_table",
        headers,
        rows,
        formula,
        y,
        json.dumps(config),
        fisher_rao_w,
    )
    out = dict(payload)
    for key in ("coefficients", "fitted", "sigma2", "lambdas", "edf"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=float)
    if "reml_score" in out:
        out["reml_score"] = float(out["reml_score"])
    if str(payload.get("status", "ok")) != "ok":
        raise ValueError(
            f"joint tangent Gaussian REML failed with status={payload.get('status')!r}"
        )
    return out
