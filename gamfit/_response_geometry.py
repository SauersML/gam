from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ._binding import rust_module
from ._exceptions import map_exception
from ._tables import normalize_table, restore_output_table, table_columns


def _as_array(values: Any, *, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2-D numeric array")
    if arr.shape[0] == 0 or arr.shape[1] < 2:
        raise ValueError(f"{label} must have at least one row and at least two columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def closure(values: Any) -> Any:
    """Normalize rows onto the probability simplex."""
    import numpy as np

    arr = _as_array(values, label="simplex values")
    if np.any(arr < 0.0):
        raise ValueError("simplex values must be non-negative")
    totals = arr.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("simplex rows must have positive total mass")
    return arr / totals


def clr(values: Any) -> Any:
    """Centered log-ratio coordinates for positive compositions."""
    import numpy as np

    comp = closure(values)
    if np.any(comp <= 0.0):
        raise ValueError("CLR coordinates require strictly positive simplex values")
    logs = np.log(comp)
    return logs - logs.mean(axis=1, keepdims=True)


def alr(values: Any, *, reference: int = -1) -> Any:
    """Additive log-ratio coordinates for positive compositions."""
    import numpy as np

    comp = closure(values)
    if np.any(comp <= 0.0):
        raise ValueError("ALR coordinates require strictly positive simplex values")
    d = comp.shape[1]
    ref = reference % d
    keep = [j for j in range(d) if j != ref]
    return np.log(comp[:, keep] / comp[:, [ref]])


def inverse_alr(coords: Any, *, reference: int = -1) -> Any:
    """Map ALR coordinates back to the simplex."""
    import numpy as np

    z = np.asarray(coords, dtype=float)
    if z.ndim != 2:
        raise ValueError("ALR coordinates must be a 2-D numeric array")
    d = z.shape[1] + 1
    ref = reference % d
    log_parts = np.zeros((z.shape[0], d), dtype=float)
    keep = [j for j in range(d) if j != ref]
    log_parts[:, keep] = z
    # Stable softmax/closure(exp(log_parts)).
    log_parts -= log_parts.max(axis=1, keepdims=True)
    parts = np.exp(log_parts)
    return parts / parts.sum(axis=1, keepdims=True)


def simplex_frechet_mean(values: Any, weights: Any | None = None) -> Any:
    """Intrinsic Fréchet mean under Aitchison simplex geometry."""
    import numpy as np

    comp = closure(values)
    if np.any(comp <= 0.0):
        raise ValueError("simplex Fréchet mean requires strictly positive values")
    w = _normalized_weights(comp.shape[0], weights)
    mean_log = (np.log(comp) * w[:, None]).sum(axis=0)
    mean_log -= mean_log.max()
    out = np.exp(mean_log)
    return out / out.sum()


def simplex_log_map(
    values: Any, base: Any, *, coordinates: str = "clr", reference: int = -1
) -> Any:
    """Log map at an intrinsic simplex base point in CLR or ALR coordinates."""
    import numpy as np

    comp = closure(values)
    base_arr = closure(np.asarray(base, dtype=float).reshape(1, -1))[0]
    if comp.shape[1] != base_arr.shape[0]:
        raise ValueError("simplex values and base point have different dimensions")
    if np.any(comp <= 0.0) or np.any(base_arr <= 0.0):
        raise ValueError("simplex log map requires strictly positive values and base point")
    coord = coordinates.lower()
    if coord in {"simplex", "clr"}:
        return clr(comp) - clr(base_arr.reshape(1, -1))
    if coord == "alr":
        return alr(comp, reference=reference) - alr(
            base_arr.reshape(1, -1), reference=reference
        )
    raise ValueError("simplex coordinates must be 'clr' or 'alr'")


def simplex_exp_map(
    tangent: Any, base: Any, *, coordinates: str = "clr", reference: int = -1
) -> Any:
    """Exponential map from simplex tangent coordinates back to compositions."""
    import numpy as np

    z = np.asarray(tangent, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    base_arr = closure(np.asarray(base, dtype=float).reshape(1, -1))[0]
    coord = coordinates.lower()
    if coord in {"simplex", "clr"}:
        if z.shape[1] != base_arr.shape[0]:
            raise ValueError("CLR tangent dimension must equal simplex dimension")
        log_parts = np.log(base_arr.reshape(1, -1)) + z
        log_parts -= log_parts.max(axis=1, keepdims=True)
        parts = np.exp(log_parts)
        return parts / parts.sum(axis=1, keepdims=True)
    if coord == "alr":
        if z.shape[1] != base_arr.shape[0] - 1:
            raise ValueError("ALR tangent dimension must be simplex dimension minus one")
        base_alr = alr(base_arr.reshape(1, -1), reference=reference)
        return inverse_alr(base_alr + z, reference=reference)
    raise ValueError("simplex coordinates must be 'clr' or 'alr'")


def sphere_frechet_mean(
    values: Any,
    weights: Any | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> Any:
    """Intrinsic Fréchet/Karcher mean on the unit sphere."""
    import numpy as np

    y = _normalize_sphere(values)
    w = _normalized_weights(y.shape[0], weights)
    mu = (y * w[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(mu))
    if norm <= 1e-14:
        mu = y[0].copy()
    else:
        mu = mu / norm
    for _ in range(max_iter):
        logs = sphere_log_map(y, mu)
        step = (logs * w[:, None]).sum(axis=0)
        step_norm = float(np.linalg.norm(step))
        if step_norm < tol:
            break
        mu = sphere_exp_map(step.reshape(1, -1), mu)[0]
    return mu


def sphere_log_map(values: Any, base: Any) -> Any:
    """Log map from the unit sphere to the ambient tangent space at ``base``.

    At the antipode (dot = −1) the Riemannian log is non-unique; any unit
    tangent vector scaled by π is a valid representative. To keep the
    Fréchet-mean iteration well posed we pick a deterministic non-zero
    tangent: project the standard basis vector least aligned with ``base``
    onto the tangent plane and rescale to length π.
    """
    import numpy as np

    y = _normalize_sphere(values)
    b = _normalize_sphere(np.asarray(base, dtype=float).reshape(1, -1))[0]
    dots = np.clip(y @ b, -1.0, 1.0)
    theta = np.arccos(dots)
    tangent = y - dots[:, None] * b.reshape(1, -1)
    sin_theta = np.sin(theta)
    scale = np.ones_like(theta)
    regular = sin_theta > 1e-12
    scale[regular] = theta[regular] / sin_theta[regular]
    out = tangent * scale[:, None]

    # Antipodal: tangent ≈ 0 but theta ≈ π. Substitute a canonical π-length
    # tangent perpendicular to b.
    antipodal = (~regular) & (dots < 0.0)
    if np.any(antipodal):
        k = int(np.argmin(np.abs(b)))
        e = np.zeros_like(b)
        e[k] = 1.0
        e = e - float(np.dot(e, b)) * b
        n = float(np.linalg.norm(e))
        if n <= 1e-12:
            # b nearly aligned with e_k; pick the next axis instead.
            k2 = (k + 1) % b.shape[0]
            e = np.zeros_like(b)
            e[k2] = 1.0
            e = e - float(np.dot(e, b)) * b
            n = float(np.linalg.norm(e))
        e = e / max(n, 1e-300)
        out[antipodal, :] = np.pi * e.reshape(1, -1)

    # Coincident points (theta ≈ 0): zero tangent.
    out[theta < 1e-12, :] = 0.0
    return out


def sphere_exp_map(tangent: Any, base: Any) -> Any:
    """Exponential map from the ambient tangent space at ``base`` to the sphere."""
    import numpy as np

    z = np.asarray(tangent, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    b = _normalize_sphere(np.asarray(base, dtype=float).reshape(1, -1))[0]
    # Project away tiny numerical radial components so fitted coordinates remain tangent.
    z = z - (z @ b)[:, None] * b.reshape(1, -1)
    r = np.linalg.norm(z, axis=1)
    out = np.empty_like(z)
    small = r < 1e-12
    out[small, :] = b.reshape(1, -1) + z[small, :]
    if np.any(~small):
        rr = r[~small]
        out[~small, :] = np.cos(rr)[:, None] * b.reshape(1, -1) + (
            np.sin(rr) / rr
        )[:, None] * z[~small, :]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / norms


def _normalize_sphere(values: Any) -> Any:
    import numpy as np

    arr = _as_array(values, label="spherical values")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("spherical rows must have non-zero norm")
    return arr / norms


def _normalized_weights(n: int, weights: Any | None) -> Any:
    import numpy as np

    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("weights length must match the number of rows")
    if np.any(~np.isfinite(w)) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive total")
    return w / float(w.sum())


def response_matrix_from_table(data: Any, response_columns: Sequence[str]) -> Any:
    columns, _kind = table_columns(data)
    missing = [name for name in response_columns if name not in columns]
    if missing:
        raise ValueError(f"response geometry columns missing from data: {missing}")
    import numpy as np

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
) -> tuple[Any, Any, str]:
    import numpy as np

    kind = geometry.lower()
    if kind in {"spherical", "sphere"}:
        base_point = (
            sphere_frechet_mean(values)
            if base is None
            else _normalize_sphere(np.asarray(base, dtype=float).reshape(1, -1))[0]
        )
        return sphere_log_map(values, base_point), base_point, "spherical"
    if kind in {"simplex", "clr", "alr"}:
        coord = (
            coordinates.lower()
            if coordinates is not None
            else ("alr" if kind == "alr" else "clr")
        )
        base_point = (
            simplex_frechet_mean(values)
            if base is None
            else closure(np.asarray(base, dtype=float).reshape(1, -1))[0]
        )
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
    if kind in {"spherical", "sphere"}:
        return sphere_exp_map(tangent, base)
    if kind in {"simplex", "clr", "alr"}:
        coord = (
            coordinates.lower()
            if coordinates is not None
            else ("alr" if kind == "alr" else "clr")
        )
        return simplex_exp_map(tangent, base, coordinates=coord, reference=reference)
    raise ValueError(
        "response_geometry must be one of 'spherical', 'simplex', 'clr', or 'alr'"
    )


@dataclass
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
            "model_class": "shared-gaussian-reml",
            "lambda": float(self.fit["lambda"]),
            "rho": float(self.fit["rho"]),
            "reml_score": float(self.fit["reml_score"]),
            "edf": float(self.fit["edf"]),
            "sigma2": self.fit["sigma2"].tolist(),
            "template": self.template_model.summary(),
        }


@dataclass
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
        import numpy as np

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
    target = "__gamfit_response_geometry_shared"
    if target in columns:
        raise ValueError(f"response geometry reserved column already exists: {target}")
    augmented = {name: list(values) for name, values in columns.items()}
    augmented[target] = tangent[:, 0].tolist()
    template_formula = f"{target} ~ {rhs}"
    template_model = fit_func(augmented, template_formula, **kwargs)
    shared_fit = _fit_shared_tangent_reml(augmented, template_formula, tangent, kwargs)
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
) -> dict[str, Any]:
    import json
    import numpy as np

    if fit_kwargs.get("offset") is not None:
        raise ValueError("response geometry shared REML does not support offsets")
    headers, rows, _kind = normalize_table(data)
    config = {
        "family": "gaussian",
        "link": "identity",
        "weights": fit_kwargs.get("weights"),
    }
    y = np.ascontiguousarray(np.asarray(tangent, dtype=float))
    try:
        payload = rust_module().gaussian_reml_fit_formula_table(
            headers,
            rows,
            formula,
            y,
            json.dumps(config),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    out = dict(payload)
    for key in (
        "coefficients",
        "fitted",
        "sigma2",
        "lambda",
        "rho",
        "reml_score",
        "edf",
    ):
        if key in out:
            out[key] = np.asarray(out[key], dtype=float)
    if str(payload.get("status", "ok")) != "ok":
        raise ValueError(f"shared Gaussian REML failed with status={payload.get('status')!r}")
    return out
