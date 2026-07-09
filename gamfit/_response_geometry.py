"""Response-geometry transforms (simplex + sphere) — thin FFI shims.

Numerical work (closure, CLR/ALR, Fréchet means, log/exp maps, fisher_rao_w
validation) lives in `gam-pyffi`. This module marshals NumPy arrays across the
boundary and hosts the small dataclasses that hold a fitted shared-tangent
model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


def _composition_rows(np: Any, values: Any) -> tuple[Any, bool]:
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
    Rust (``response_geometry_log_map``); this only marshals arrays.

    ``weights`` are the per-observation prior weights used ONLY to pick the
    intrinsic base point (ignored when an explicit ``base`` is supplied). A
    weighted response-geometry fit linearizes every response in the tangent
    space at the base point and runs a *weighted* tangent regression there, so
    the chart origin must be the **weighted** Fréchet mean — where the weighted
    mass lives — not the unweighted one (#2125).
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
    profile-likelihood 95% CI (``ci_lo``/``ci_hi`` with open-at-bound flags), the
    geometry ``verdict`` (spherical / hyperbolic / flat from the CI sign), and the
    interior-point Wilks flatness test of κ = 0 (``flatness_lr`` /
    ``flatness_pvalue``).

    Scale-awareness (#1104): κ has units 1/length², so ``kappa_hat`` is
    *scale-dependent*. ``railed_at_resolution_limit`` is ``True`` when the cloud
    is curved beyond what its spread can resolve (it fills the sphere) and κ̂
    railed to the chart conjugate cap — κ̂ / ``ci_hi`` are then a LOWER BOUND on
    |κ| ("curvature exceeds chart-resolvable range at this scale"), NOT a resolved
    point estimate. ``kappa_r2`` = κ̂·r² is the scale-FREE invariant the cloud
    actually determines (invariant under ``y → α·y``); ``characteristic_radius``
    = r is the κ=0 spread it is dimensionless to. For arbitrarily-rescaled data
    (e.g. unit-normalised activations) read ``kappa_r2`` / the rail flag, NOT the
    scale-dependent ``kappa_hat`` alone.
    """
    np = _np()
    (
        kappa_hat,
        ci_lo,
        ci_hi,
        lo_at_bound,
        hi_at_bound,
        verdict,
        lr_stat,
        p_value,
        railed,
        kappa_r2,
        characteristic_radius,
        base_point,
    ) = _ffi(
        "response_geometry_fit_curvature",
        np.asarray(values, dtype=float),
        str(geometry),
        float(level),
    )
    return {
        "kappa_hat": float(kappa_hat),
        "ci_level": float(level),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ci_lo_at_bound": bool(lo_at_bound),
        "ci_hi_at_bound": bool(hi_at_bound),
        "verdict": str(verdict),
        "flatness_lr": float(lr_stat),
        "flatness_pvalue": float(p_value),
        "railed_at_resolution_limit": bool(railed),
        "kappa_r2": float(kappa_r2),
        "characteristic_radius": float(characteristic_radius),
        "base_point": list(map(float, base_point)),
    }


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
        # ``lambdas`` / ``edf`` are shared per-smooth (length M, common to every
        # tangent coordinate); ``sigma2`` is the single pooled isotropic residual
        # variance. They are reported as lists so callers need not special-case
        # the array rank.
        return {
            "model_class": "joint-tangent-gaussian-reml",
            "reml_score": float(self.fit["reml_score"]),
            "lambdas": self.fit["lambdas"].tolist(),
            "edf": self.fit["edf"].tolist(),
            "sigma2": self.fit["sigma2"].tolist(),
            "shared_smoothing": True,
            "template": self.template_model.summary(),
        }


@dataclass(slots=True)
class ResponseGeometryModel:
    """A fitted response-geometry GAM.

    The tangent coordinates are fitted jointly as one vector-valued Gaussian GAM
    through the general multi-penalty REML solver, with **one smoothing
    parameter per smooth shared across every coordinate** (the penalty is
    ``Sᵇ ⊗ I_D`` with a single λ_b) and a single pooled isotropic residual
    variance. Sharing the smoothing is what makes the fit *frame-equivariant*:
    the tangent vector field is a single function of the predictor, so its
    smoothness is a property of the predictor, not of the arbitrary ambient
    coordinate axis. An optional Fisher-Rao precision metric couples the
    coordinate residuals on top of the shared smoothing.
    """

    models: Sequence[Any]
    response_geometry: str
    response_columns: tuple[str, ...]
    base_point: Any
    coordinates: str
    reference: int = -1
    training_table_kind: str | None = None
    shared_tangent_fit: SharedGaussianRemlTangentFit | None = None
    # Curvature-as-estimand summary (only populated for constant_curvature
    # geometries): κ̂, its profile-likelihood CI, geometry verdict, and the Wilks
    # flatness test of κ = 0. ``None`` for fixed-geometry response manifolds.
    curvature: dict[str, Any] | None = None

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
    ) -> Any:
        np = _np()
        _columns, input_kind = table_columns(data)
        if self.shared_tangent_fit is not None:
            tangent = np.asarray(self.shared_tangent_fit.predict_tangent(data), dtype=float)
        else:
            tangent_cols: list[Any] = []
            for model in self.models:
                pred = model.predict(data, return_type="dict")
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
        out: dict[str, Any] = {
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
        if self.curvature is not None:
            # κ̂ with profile CI, geometry verdict, and Wilks flatness test —
            # the curvature-as-estimand report for constant_curvature fits.
            out["curvature"] = dict(self.curvature)
        return out

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """Round-trippable JSON-compatible serialization of this fit.

        Mirrors the persistence added for :class:`MultinomialModel` (#2078): the
        response-geometry model is a Python composite (base point + coordinate
        chart + geometry label + the tangent-coordinate ``Model`` fits), so it is
        serialized as a small JSON container that embeds each constituent
        ``Model`` through its own binary archive (base64-encoded ``Model.dumps``)
        plus the base point, coordinate chart, and geometry metadata. Passed to
        :func:`gamfit.loads` (or written by :meth:`save` / :func:`gamfit.save`)
        it reconstructs a :class:`ResponseGeometryModel` that reproduces
        :meth:`predict`.
        """
        import base64

        np = _np()

        def _model_b64(model: Any) -> str:
            return base64.b64encode(bytes(model.dumps())).decode("ascii")

        def _jsonable(value: Any) -> Any:
            if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        shared: dict[str, Any] | None = None
        if self.shared_tangent_fit is not None:
            f = self.shared_tangent_fit
            shared = {
                "template_model_b64": _model_b64(f.template_model),
                "coefficients": np.asarray(f.coefficients, dtype=float).tolist(),
                "fit": {str(k): _jsonable(v) for k, v in f.fit.items()},
            }
        return {
            "schema": _RESPONSE_GEOMETRY_SCHEMA,
            "response_geometry": self.response_geometry,
            "response_columns": list(self.response_columns),
            "base_point": [
                float(v) for v in np.asarray(self.base_point, dtype=float).reshape(-1)
            ],
            "coordinates": self.coordinates,
            "reference": int(self.reference),
            "training_table_kind": self.training_table_kind,
            "curvature": None if self.curvature is None else dict(self.curvature),
            "coordinate_models_b64": [_model_b64(m) for m in self.models],
            "shared_tangent_fit": shared,
        }

    def dumps(self) -> bytes:
        """Return the serialized response-geometry model as raw JSON bytes."""
        import json

        return json.dumps(self.to_dict()).encode("utf-8")

    def save(self, path: Any) -> None:
        """Serialise the fitted response-geometry model to ``path``.

        Mirrors :meth:`Model.save`; the resulting file round-trips through
        :func:`gamfit.load`, which reconstructs a
        :class:`ResponseGeometryModel`.
        """
        from pathlib import Path

        Path(path).write_bytes(self.dumps())


_RESPONSE_GEOMETRY_SCHEMA = "gamfit.ResponseGeometryModel/v1"


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
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    precision_hyperpriors: Any | None = None,
    latents: Mapping[str, Any] | None = None,
    penalties: Sequence[Any] | None = None,
    smooths: Mapping[Any, Any] | None = None,
    constraints: Mapping[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ResponseGeometryModel:
    columns, table_kind = table_columns(data)
    y = response_matrix_from_table(data, response_columns)

    # Curvature-as-estimand (#944 / #1104): for a constant-curvature response
    # geometry κ is fitted from the responses (NOT user-supplied). Estimate κ̂
    # first, then build the tangent coordinates AT κ̂ so the whole fit is on the
    # evidence-optimal geometry. κ̂ + its profile CI + the Wilks flatness test are
    # carried into the model summary.
    curvature_summary: dict[str, Any] | None = None
    geometry_for_maps = response_geometry
    if response_geometry.strip().lower().split("(", 1)[0] == "constant_curvature":
        curvature_summary = fit_response_curvature(y, geometry=response_geometry)
        kappa_hat = curvature_summary["kappa_hat"]
        # Re-express the geometry at κ̂ for the log/exp maps (dim is inferred from
        # the response column count exactly as the estimand inferred it).
        geometry_for_maps = f"constant_curvature(dim={int(y.shape[1])},kappa={kappa_hat!r})"

    # Observation weights reach BOTH the tangent regression and the base point.
    # ``weights`` is the name of a prior-weight column; resolve it to the raw
    # per-row vector so the intrinsic base point is the *weighted* Fréchet mean
    # — the linearization point of a weighted fit must sit where the weighted
    # mass lives, not at the unweighted mean (#2125). The tangent GAM below is
    # handed the same column name, so both use identical weights.
    np = _np()
    base_weights = None
    if weights is not None:
        if weights not in columns:
            raise ValueError(
                f"response geometry weights column missing from data: {weights!r}"
            )
        base_weights = np.asarray(columns[weights], dtype=float)
    tangent, base, resolved_coordinates = geometry_log_map(
        y,
        geometry=geometry_for_maps,
        coordinates=coordinates,
        reference=reference,
        weights=base_weights,
    )
    rhs = _formula_rhs(formula)
    kwargs = {
        "family": "gaussian",
        "link": "identity",
        "weights": weights,
        "scale_dimensions": scale_dimensions,
        "adaptive_regularization": adaptive_regularization,
        "firth": firth,
        "precision_hyperpriors": precision_hyperpriors,
        "latents": latents,
        "penalties": penalties,
        "smooths": smooths,
        "constraints": constraints,
        "config": config,
    }
    np = _np()
    # A user-supplied fisher_rao_w is an explicit residual-metric override and is
    # normalised in Rust (`response_geometry_normalize_fisher_rao`) before it is
    # handed to the joint tangent REML. The Aitchison ALR whitener that used to be
    # synthesised here in NumPy (G^{1/2} via `np.linalg.eigh`) was SPEC-violating
    # Python math with no Rust FFI, so it has been removed; use CLR/ILR coordinates
    # (already Aitchison-isometric, G = I) for a reference-free simplex fit.
    fisher_source = fisher_rao_w
    fisher_w = None
    if fisher_source is not None:
        fisher_w = _ffi(
            "response_geometry_normalize_fisher_rao",
            np.asarray(fisher_source, dtype=float),
            int(tangent.shape[0]),
            int(tangent.shape[1]),
        )
    target = "__gamfit_response_geometry_shared"
    if target in columns:
        raise ValueError(f"response geometry reserved column already exists: {target}")
    augmented = {name: list(values) for name, values in columns.items()}
    augmented[target] = tangent[:, 0].tolist()
    template_formula = f"{target} ~ {rhs}"
    template_model = fit_func(augmented, template_formula, **kwargs)
    # The joint tangent REML runs on the native tangent coordinates. (The former
    # ALR whitening branch was NumPy-side linear algebra with no Rust FFI and has
    # been removed.)
    shared_fit = _fit_shared_tangent_reml(
        augmented, template_formula, tangent, kwargs, fisher_w
    )
    return ResponseGeometryModel(
        models=(),
        # The geometry used for the log/exp maps carries the fitted κ̂ for a
        # constant-curvature fit (so predict() round-trips on the SAME geometry
        # the tangent was built on); for fixed geometries it is just the label.
        response_geometry=geometry_for_maps.lower(),
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
        curvature=curvature_summary,
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
