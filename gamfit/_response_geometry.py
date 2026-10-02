"""Response geometry — thin FFI shims.

The transforms (closure, CLR/ALR, Fréchet means, log/exp maps) and the whole
response-geometry GAM (curvature estimand, tangent chart, joint shared-smoothing
fit, prediction, summary and persistence) live in Rust
(`gam_geometry::response_geometry`, `gam_predict::response_geometry`). This
module marshals arrays and holds a fitted model's saved bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Sequence

from ._binding import rust_module
from ._exceptions import map_exception
from ._tables import normalize_table, restore_output_table


def _ffi(name: str, *args: Any) -> Any:
    try:
        return getattr(rust_module(), name)(*args)
    except Exception as exc:
        raise map_exception(exc) from exc


def _np() -> ModuleType:
    import numpy as np

    return np


def _composition_rows(np: ModuleType, values: Any) -> tuple[Any, bool]:
    """Marshal a composition argument to the ``(rows, parts)`` 2-D layout the
    Rust FFI requires, recording whether the caller passed a single composition.

    The compositional primitives (``closure`` / ``clr`` / ``alr``) are defined
    on a *single* composition, and the rest of the NumPy-facing surface accepts a
    1-D vector of points, so ``clr([0.2, 0.3, 0.5])`` is the natural call. But the
    ``#[pyfunction]`` signatures take a 2-D ``PyReadonlyArray2`` only, so a 1-D
    argument used to surface as the opaque ``TypeError: 'ndarray' object is not an
    instance of 'ndarray'``. We promote a 1-D composition to a single
    ``(1, parts)`` row here and let the caller squeeze the result back to 1-D, so
    the single-composition result matches the corresponding row of the 2-D batch
    call.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1), True
    return arr, False


def closure(values: Any) -> Any:
    """Normalize rows onto the probability simplex.

    Accepts either a 2-D ``(rows, parts)`` batch or a single 1-D composition; a
    1-D input yields 1-D coordinates matching the corresponding batch row.
    """
    np = _np()
    rows, was_1d = _composition_rows(np, values)
    out = _ffi("response_geometry_closure", rows)
    return np.asarray(out)[0] if was_1d else out


def clr(values: Any) -> Any:
    """Centered log-ratio coordinates for positive compositions.

    Accepts either a 2-D ``(rows, parts)`` batch or a single 1-D composition; a
    1-D input yields 1-D coordinates matching the corresponding batch row.
    """
    np = _np()
    rows, was_1d = _composition_rows(np, values)
    out = _ffi("response_geometry_clr", rows)
    return np.asarray(out)[0] if was_1d else out


def alr(values: Any, *, reference: int = -1) -> Any:
    """Additive log-ratio coordinates for positive compositions.

    Accepts either a 2-D ``(rows, parts)`` batch or a single 1-D composition; a
    1-D input yields 1-D coordinates matching the corresponding batch row.
    """
    np = _np()
    rows, was_1d = _composition_rows(np, values)
    out = _ffi("response_geometry_alr", rows, int(reference))
    return np.asarray(out)[0] if was_1d else out


def inverse_alr(coords: Any, *, reference: int = -1) -> Any:
    """Map ALR coordinates back to the simplex.

    Accepts either a 2-D ``(rows, parts - 1)`` ALR-coordinate batch or a single
    1-D ALR-coordinate vector; a 1-D input yields the reconstructed 1-D
    composition matching the corresponding batch row.
    """
    np = _np()
    rows, was_1d = _composition_rows(np, coords)
    out = _ffi("response_geometry_inverse_alr", rows, int(reference))
    return np.asarray(out)[0] if was_1d else out


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


def sphere_frechet_mean(values: Any, weights: Any | None = None) -> Any:
    """Intrinsic Fréchet/Karcher mean on the unit sphere."""
    np = _np()
    w = None if weights is None else np.asarray(weights, dtype=float)
    return _ffi("sphere_frechet_mean", np.asarray(values, dtype=float), w)


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


def geometry_log_map(
    values: Any,
    *,
    geometry: str,
    base: Any | None = None,
    coordinates: str | None = None,
    reference: int = -1,
    weights: Any | None = None,
) -> tuple[Any, Any, str]:
    """Map response observations to tangent coordinates at an intrinsic base.

    Geometry-kind routing, simplex-coordinate resolution, and base-point
    selection (intrinsic Fréchet mean when ``base`` is ``None``) all live in
    Rust (``gam_geometry::response_geometry::log_map``); this only marshals
    arrays.

    ``weights`` are the per-observation prior weights used ONLY to pick the
    intrinsic base point (ignored when an explicit ``base`` is supplied): a
    weighted fit linearizes at the weighted Fréchet mean, where the weighted
    mass lives (#2125).
    """
    np = _np()
    base_arr = None if base is None else np.asarray(base, dtype=float).reshape(-1)
    weights_arr = (
        None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    )
    tangent, base_point, coord = _ffi(
        "response_geometry_log_map",
        np.asarray(values, dtype=float),
        str(geometry),
        base_arr,
        None if coordinates is None else str(coordinates),
        int(reference),
        weights_arr,
    )
    return tangent, base_point, coord


def geometry_exp_map(
    tangent: Any,
    *,
    geometry: str,
    base: Any,
    coordinates: str | None = None,
    reference: int = -1,
) -> Any:
    """Map tangent coordinates back onto the response manifold (Rust-owned)."""
    np = _np()
    return _ffi(
        "response_geometry_exp_map",
        np.asarray(tangent, dtype=float),
        str(geometry),
        np.asarray(base, dtype=float).reshape(-1),
        None if coordinates is None else str(coordinates),
        int(reference),
    )


def fit_response_curvature(values: Any, *, geometry: str, level: float = 0.95) -> dict[str, Any]:
    """Estimate curvature κ̂ on a constant-curvature response geometry.

    κ is NOT supplied by the user: the geometry label is ``constant_curvature(
    dim=d)`` and κ̂ is fitted from the manifold-valued responses by the REML /
    evidence outer loop (the profiled Fréchet-dispersion criterion, owned in
    Rust). Returns the fit summary: the point estimate ``kappa_hat``, the
    profile-likelihood CI (``ci_lo``/``ci_hi`` with open-at-bound flags), the
    geometry ``verdict`` (spherical / hyperbolic / flat from the CI sign), and the
    interior-point Wilks flatness test of κ = 0 (``flatness_lr`` /
    ``flatness_pvalue``).

    Scale-awareness (#1104): κ has units 1/length², so ``kappa_hat`` is
    *scale-dependent*. ``railed_at_resolution_limit`` is ``True`` when the cloud
    is curved beyond what its spread can resolve (it fills the sphere) and κ̂
    railed to the chart conjugate cap — κ̂ / ``ci_hi`` are then a LOWER BOUND on
    |κ|, NOT a resolved point estimate. ``kappa_r2`` = κ̂·r² is the scale-FREE
    invariant the cloud actually determines (invariant under ``y → α·y``);
    ``characteristic_radius`` = r is the κ=0 spread it is dimensionless to.
    """
    np = _np()
    payload = _ffi(
        "response_geometry_fit_curvature",
        np.asarray(values, dtype=float),
        str(geometry),
        float(level),
    )
    return {
        "kappa_hat": float(payload["kappa_hat"]),
        "ci_level": float(level),
        "ci_lo": float(payload["ci_lo"]),
        "ci_hi": float(payload["ci_hi"]),
        "ci_lo_at_bound": bool(payload["ci_lo_at_bound"]),
        "ci_hi_at_bound": bool(payload["ci_hi_at_bound"]),
        "verdict": str(payload["verdict"]),
        "flatness_lr": float(payload["flatness_lr"]),
        "flatness_pvalue": float(payload["flatness_pvalue"]),
        "railed_at_resolution_limit": bool(payload["railed_at_resolution_limit"]),
        "railed_at_hyperbolic_resolution_limit": bool(
            payload["railed_at_hyperbolic_resolution_limit"]
        ),
        "kappa_r2": float(payload["kappa_r2"]),
        "characteristic_radius": float(payload["characteristic_radius"]),
        "base_point": list(map(float, payload["base_point"])),
    }


@dataclass(frozen=True, slots=True)
class ResponseGeometryModel:
    """A fitted response-geometry GAM.

    The tangent coordinates are fitted jointly as one vector-valued Gaussian GAM
    through the general multi-penalty REML solver, with **one smoothing
    parameter per smooth shared across every coordinate** (the penalty is
    ``Sᵇ ⊗ I_D`` with a single λ_b) and a single pooled isotropic residual
    variance, which makes the fit frame-equivariant. An optional Fisher-Rao
    precision metric couples the coordinate residuals on top of the shared
    smoothing. Fitting, prediction, the summary and persistence are the Rust
    ``gam_predict::response_geometry`` owner; this object holds its saved bytes.
    """

    _model_bytes: bytes

    def _fields(self) -> dict[str, Any]:
        return dict(_ffi("response_geometry_model_fields", self._model_bytes))

    @property
    def response_geometry(self) -> str:
        return str(self._fields()["response_geometry"])

    @property
    def response_columns(self) -> tuple[str, ...]:
        return tuple(self._fields()["response_columns"])

    @property
    def base_point(self) -> Any:
        return _np().asarray(self._fields()["base_point"], dtype=float)

    @property
    def coordinates(self) -> str:
        return str(self._fields()["coordinates"])

    @property
    def reference(self) -> int:
        return int(self._fields()["reference"])

    @property
    def training_table_kind(self) -> str:
        return str(self._fields()["training_table_kind"])

    @property
    def tangent_dimension(self) -> int:
        return int(self._fields()["tangent_dimension"])

    @property
    def curvature(self) -> dict[str, Any] | None:
        """κ̂ with its profile CI, verdict and flatness test; ``None`` off a
        constant-curvature geometry."""
        curvature = self._fields()["curvature"]
        return None if curvature is None else dict(curvature)

    def predict(
        self,
        data: Any,
        *,
        return_type: str | None = None,
        include_tangent: bool = False,
    ) -> Any:
        headers, rows, input_kind = normalize_table(data)
        response, tangent = _ffi(
            "response_geometry_predict_table", self._model_bytes, headers, rows
        )
        out: dict[str, list[Any]] = {
            name: response[:, idx].tolist()
            for idx, name in enumerate(self.response_columns)
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
        return dict(_ffi("response_geometry_summary", self._model_bytes))

    def to_dict(self) -> dict[str, Any]:
        """The saved container as a JSON-compatible mapping."""
        import json

        return dict(json.loads(self._model_bytes.decode("utf-8")))

    def dumps(self) -> bytes:
        """Return the serialized response-geometry model."""
        return bytes(self._model_bytes)

    def save(self, path: Any) -> None:
        """Serialise the fitted response-geometry model to ``path``.

        Mirrors :meth:`Model.save`, atomic and durable alike; the resulting
        file round-trips through :func:`gamfit.load`.
        """
        rust_module().write_saved_model_file(path, self._model_bytes)


def fit_response_geometry(
    data: Any,
    formula: str,
    fit_payload: dict[str, Any],
    *,
    response_geometry: str,
    response_columns: Sequence[str],
    coordinates: str | None = None,
    reference: int = -1,
    fisher_rao_w: Any | None = None,
) -> ResponseGeometryModel:
    """Fit a response-geometry GAM through the Rust owner.

    ``fit_payload`` is the scalar fit request :func:`gamfit.fit` builds; the Rust
    fit refuses every field of it the joint tangent fit cannot honour.
    """
    import json

    np = _np()
    headers, rows, table_kind = normalize_table(data)
    payload = dict(fit_payload)
    payload["training_table_kind"] = table_kind
    fisher = None if fisher_rao_w is None else np.asarray(fisher_rao_w, dtype=float)
    model_bytes = _ffi(
        "fit_response_geometry_table",
        headers,
        rows,
        formula,
        json.dumps(payload),
        str(response_geometry),
        [str(name) for name in response_columns],
        None if coordinates is None else str(coordinates),
        int(reference),
        fisher,
    )
    return ResponseGeometryModel(bytes(model_bytes))
