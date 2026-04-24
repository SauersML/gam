from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._schema import SchemaCheck
from ._summary import Summary
from ._tables import (
    coerce_numeric_vector,
    normalize_table,
    response_column_name,
    restore_output_table,
    table_columns,
)

DEFAULT_SURVIVAL_PEOPLE_CHUNK = 50_000
DEFAULT_SURVIVAL_TIME_GRID_CHUNK = 64
MAX_DENSE_SURVIVAL_CURVE_CELLS = 1_000_000

_SURVIVAL_MODEL_CLASSES = frozenset(
    {
        "survival marginal-slope",
        "survival location-scale",
        "latent survival",
    }
)
_MARGINAL_SLOPE_MODEL_CLASSES = frozenset(
    {
        "bernoulli marginal-slope",
        "survival marginal-slope",
    }
)
_TRANSFORMATION_NORMAL_MODEL_CLASSES = frozenset(
    {
        "transformation-normal",
    }
)


@dataclass
class SurvivalPrediction:
    """Per-row survival functions evaluated on demand.

    When the FFI produced a dense ``(n_samples, n_times)`` grid of
    hazard / survival / cumulative hazard values, the ``*_at`` helpers
    linearly interpolate against that grid. Otherwise they fall back to
    the legacy plug-in piecewise-constant hazard reconstructed from
    ``parameters`` so bare-dataclass construction keeps working.

    Attributes
    ----------
    model_class:
        The fitted model class string (e.g. ``"survival marginal-slope"``).
    parameters:
        Flat per-row parameters returned by the FFI. Shape
        ``(n_samples, n_params_per_row)``. The exact column semantics depend
        on ``model_class``; callers should treat this as opaque and prefer the
        ``*_at`` helpers.
    parameter_names:
        Column names corresponding to ``parameters``, in order.
    baseline_times:
        Optional per-sample baseline evaluation grid (if the FFI reports one).
    times:
        Shared 1-D time grid at which the hazard surfaces were evaluated.
    hazard:
        ``(n_samples, len(times))`` dense hazard surface from the FFI.
    survival:
        ``(n_samples, len(times))`` dense survival surface from the FFI.
    cumulative_hazard:
        ``(n_samples, len(times))`` dense cumulative-hazard surface from the FFI.
    linear_predictor:
        ``(n_samples,)`` per-row linear predictor at each row's own exit time.
    """

    model_class: str
    parameters: Any
    parameter_names: Sequence[str] = field(default_factory=tuple)
    baseline_times: Any | None = None
    times: Any | None = None
    hazard: Any | None = None
    survival: Any | None = None
    cumulative_hazard: Any | None = None
    linear_predictor: Any | None = None

    def _coerce_times(self, times: Any) -> Any:
        import numpy as np

        times_arr = np.asarray(times, dtype=float).reshape(-1)
        if times_arr.size == 0:
            raise ValueError("survival prediction requires at least one time")
        if not np.all(np.isfinite(times_arr)):
            raise ValueError("survival prediction times must be finite")
        return times_arr

    def _parameters_array(self) -> Any:
        import numpy as np

        params = np.asarray(self.parameters, dtype=float)
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        return params

    def _check_dense_size(self, n_rows: int, n_times: int) -> None:
        cells = int(n_rows) * int(n_times)
        if cells > MAX_DENSE_SURVIVAL_CURVE_CELLS:
            raise ValueError(
                "dense survival curves are limited to diagnostic subsets "
                f"({MAX_DENSE_SURVIVAL_CURVE_CELLS} cells); requested "
                f"{n_rows} rows x {n_times} times. Use survival_at_chunks(), "
                "hazard_at_chunks(), or write_survival_at_csv()."
            )

    def hazard_at(self, times: Any) -> Any:
        """Hazard rate ``h(t)`` evaluated at each time in ``times``.

        Returns an ``(n_samples, len(times))`` numpy array for diagnostic-sized
        requests. Large production grids must use ``hazard_at_chunks``.
        """
        import numpy as np

        times_arr = self._coerce_times(times)
        grid, hazard = self._ffi_surface("hazard")
        if grid is not None and hazard is not None:
            self._check_dense_size(hazard.shape[0], times_arr.size)
            return _interpolate_rows(grid, hazard, times_arr, clip=(0.0, None))

        self._check_dense_size(self._parameters_array().shape[0], times_arr.size)
        cumulative = self.cumulative_hazard_at(times_arr)
        if times_arr.size <= 1:
            return cumulative
        grid_full = np.concatenate([[0.0], times_arr])
        cumulative_full = np.concatenate(
            [np.zeros((cumulative.shape[0], 1)), cumulative], axis=1
        )
        diffs = np.diff(cumulative_full, axis=1)
        widths = np.diff(grid_full)
        widths = np.where(widths <= 0.0, 1.0, widths)
        return diffs / widths

    def cumulative_hazard_at(self, times: Any) -> Any:
        """Cumulative hazard ``H(t) = -log S(t)`` for diagnostic-sized grids."""
        import numpy as np

        times_arr = self._coerce_times(times)
        grid, cumulative = self._ffi_surface("cumulative_hazard")
        if grid is not None and cumulative is not None:
            self._check_dense_size(cumulative.shape[0], times_arr.size)
            return _interpolate_rows(grid, cumulative, times_arr, clip=(0.0, None))

        survival = self.survival_at(times)
        survival = np.clip(survival, 1e-12, 1.0)
        return -np.log(survival)

    def survival_at(self, times: Any) -> Any:
        """Survival probability ``S(t)`` at each requested time.

        When the FFI produced a dense hazard/survival surface we
        linearly interpolate against the returned grid. Otherwise we
        fall back to the plug-in identity ``S(t) = exp(-H(t))`` using a
        per-row piecewise-constant hazard derived from ``parameters``
        for backward compatibility with bare-dataclass construction.
        Dense output is intentionally bounded. Use ``survival_at_chunks`` or
        ``write_survival_at_csv`` for biobank-scale prediction.
        """
        import numpy as np

        times_arr = self._coerce_times(times)
        grid, surv = self._ffi_surface("survival")
        if grid is not None and surv is not None:
            self._check_dense_size(surv.shape[0], times_arr.size)
            interpolated = _interpolate_rows(grid, surv, times_arr, clip=(0.0, 1.0))
            return interpolated

        params = self._parameters_array()
        self._check_dense_size(params.shape[0], times_arr.size)
        return self._survival_block(params, times_arr)

    def _ffi_surface(self, kind: str) -> tuple[Any, Any]:
        """Return ``(grid, surface)`` for the FFI-provided surface or
        ``(None, None)`` when the caller constructed this object manually."""
        import numpy as np

        if self.times is None:
            return (None, None)
        grid = np.asarray(self.times, dtype=float).reshape(-1)
        if grid.size == 0:
            return (None, None)
        surface = getattr(self, kind, None)
        if surface is None:
            return (None, None)
        surface_arr = np.asarray(surface, dtype=float)
        if surface_arr.ndim != 2 or surface_arr.shape[1] != grid.size:
            return (None, None)
        return (grid, surface_arr)

    def survival_at_chunks(
        self,
        times: Any,
        *,
        people_chunk: int = DEFAULT_SURVIVAL_PEOPLE_CHUNK,
        time_grid_chunk: int = DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
    ) -> Any:
        """Yield ``(row_slice, time_slice, survival_block)`` chunks."""
        params = self._parameters_array()
        times_arr = self._coerce_times(times)
        people_chunk = _validate_survival_chunk_size(people_chunk, "people_chunk")
        time_grid_chunk = _validate_survival_chunk_size(time_grid_chunk, "time_grid_chunk")
        for row_start in range(0, params.shape[0], people_chunk):
            row_stop = min(row_start + people_chunk, params.shape[0])
            row_params = params[row_start:row_stop, :]
            for time_start in range(0, times_arr.size, time_grid_chunk):
                time_stop = min(time_start + time_grid_chunk, times_arr.size)
                yield (
                    slice(row_start, row_stop),
                    slice(time_start, time_stop),
                    self._survival_block(row_params, times_arr[time_start:time_stop]),
                )

    def cumulative_hazard_at_chunks(
        self,
        times: Any,
        *,
        people_chunk: int = DEFAULT_SURVIVAL_PEOPLE_CHUNK,
        time_grid_chunk: int = DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
    ) -> Any:
        """Yield ``(row_slice, time_slice, cumulative_hazard_block)`` chunks."""
        import numpy as np

        for row_slice, time_slice, survival in self.survival_at_chunks(
            times,
            people_chunk=people_chunk,
            time_grid_chunk=time_grid_chunk,
        ):
            yield row_slice, time_slice, -np.log(np.clip(survival, 1e-12, 1.0))

    def hazard_at_chunks(
        self,
        times: Any,
        *,
        people_chunk: int = DEFAULT_SURVIVAL_PEOPLE_CHUNK,
        time_grid_chunk: int = DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
    ) -> Any:
        """Yield ``(row_slice, time_slice, hazard_block)`` chunks."""
        import numpy as np

        times_arr = self._coerce_times(times)
        people_chunk = _validate_survival_chunk_size(people_chunk, "people_chunk")
        time_grid_chunk = _validate_survival_chunk_size(time_grid_chunk, "time_grid_chunk")
        params = self._parameters_array()
        for row_start in range(0, params.shape[0], people_chunk):
            row_stop = min(row_start + people_chunk, params.shape[0])
            row_params = params[row_start:row_stop, :]
            previous_cumulative = np.zeros((row_stop - row_start, 1), dtype=float)
            previous_time = 0.0
            for time_start in range(0, times_arr.size, time_grid_chunk):
                time_stop = min(time_start + time_grid_chunk, times_arr.size)
                time_block = times_arr[time_start:time_stop]
                survival = self._survival_block(row_params, time_block)
                cumulative = -np.log(np.clip(survival, 1e-12, 1.0))
                cumulative_full = np.concatenate([previous_cumulative, cumulative], axis=1)
                grid = np.concatenate([[previous_time], time_block])
                widths = np.diff(grid)
                widths = np.where(widths <= 0.0, 1.0, widths)
                yield (
                    slice(row_start, row_stop),
                    slice(time_start, time_stop),
                    np.diff(cumulative_full, axis=1) / widths.reshape(1, -1),
                )
                previous_cumulative = cumulative[:, -1:]
                previous_time = float(time_block[-1])

    def write_survival_at_csv(
        self,
        path: str | Path,
        times: Any,
        *,
        people_chunk: int = DEFAULT_SURVIVAL_PEOPLE_CHUNK,
        time_grid_chunk: int = DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
    ) -> str:
        """Write chunked survival predictions as row,time,survival CSV."""
        import csv

        times_arr = self._coerce_times(times)
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "time", "survival"])
            for row_slice, time_slice, block in self.survival_at_chunks(
                times_arr,
                people_chunk=people_chunk,
                time_grid_chunk=time_grid_chunk,
            ):
                time_block = times_arr[time_slice]
                for local_row, values in enumerate(block):
                    row_index = row_slice.start + local_row
                    for time, survival in zip(time_block, values, strict=True):
                        writer.writerow([row_index, float(time), float(survival)])
        return str(path)

    def _survival_block(self, params: Any, times_arr: Any) -> Any:
        import numpy as np

        anchor_log_hazard = params[:, 0:1]
        hazard = np.exp(anchor_log_hazard)
        cumulative = hazard * times_arr.reshape(1, -1)
        return np.exp(-cumulative)


class Model:
    def __init__(self, *, _model_bytes: bytes, _training_table_kind: str | None = None) -> None:
        self._model_bytes = _model_bytes
        self._training_table_kind = _training_table_kind
        self._summary_cache: Summary | None = None

    def predict(
        self,
        data: Any,
        *,
        interval: float | None = None,
        return_type: str | None = None,
    ) -> Any:
        """Predict from ``data``.

        The return shape depends on the fitted model class:

        * Gaussian / Binomial / Standard models: returns a table (dict, pandas
          DataFrame, pyarrow Table, ...) matching the training table kind with
          an ``eta`` and ``mean`` column (plus interval columns when
          ``interval`` is given).
        * Transformation-normal models: returns the per-row transformed
          z-score as a numpy array of shape ``(n_samples,)``.
        * Bernoulli marginal-slope: returns a calibrated probability array in
          ``(0, 1)`` of shape ``(n_samples,)``.
        * Survival models: returns a :class:`SurvivalPrediction` whose
          ``.hazard_at``, ``.survival_at``, and ``.cumulative_hazard_at``
          helpers evaluate the fitted hazard surface on a user-supplied time
          grid.
        """
        headers, rows, table_kind = normalize_table(data)
        payload: dict[str, Any] = {"interval": interval}
        # For survival models we auto-supply a default time grid built
        # from the exit/entry columns in the prediction frame so that
        # ``hazard_at`` / ``survival_at`` can interpolate at arbitrary
        # caller times.
        default_survival_grid = self._default_survival_time_grid(headers, rows)
        if default_survival_grid is not None:
            payload["time_grid"] = [float(t) for t in default_survival_grid]
        try:
            raw = rust_module().predict_table(
                self._model_bytes,
                headers,
                rows,
                json.dumps(payload),
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        parsed = json.loads(raw)

        # Structured survival payload: dense hazard/survival grid.
        if parsed.get("class") == "survival_prediction":
            return _survival_prediction_from_ffi_payload(parsed)

        columns = _ordered_prediction_columns(parsed["columns"])
        model_class = str(parsed.get("model_class") or self._model_class_from_summary())

        if model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES:
            import numpy as np

            return np.asarray(_transformation_normal_z(columns), dtype=float)

        if model_class == "bernoulli marginal-slope":
            import numpy as np

            return np.clip(
                np.asarray(columns.get("mean", []), dtype=float), 0.0, 1.0
            )

        if model_class in _SURVIVAL_MODEL_CLASSES:
            return _survival_prediction_from_columns(model_class, columns)

        return restore_output_table(
            columns,
            requested=return_type,
            input_kind=table_kind,
            training_kind=self._training_table_kind,
        )

    def summary(self) -> Summary:
        if self._summary_cache is None:
            try:
                payload = json.loads(rust_module().summary_json(self._model_bytes))
            except Exception as exc:
                raise map_exception(exc) from exc
            self._summary_cache = Summary.from_dict(payload)
        return self._summary_cache

    def check(self, data: Any) -> SchemaCheck:
        headers, rows, _table_kind = normalize_table(data)
        try:
            payload = json.loads(rust_module().check_json(self._model_bytes, headers, rows))
        except Exception as exc:
            raise map_exception(exc) from exc
        return SchemaCheck.from_dict(payload)

    def report(self, path: str | Path | None = None) -> str:
        try:
            html = rust_module().report_html(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        if path is None:
            return html
        Path(path).write_text(html, encoding="utf-8")
        return str(path)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self._model_bytes)

    def dumps(self) -> bytes:
        return self._model_bytes

    @property
    def formula(self) -> str:
        return str(self.summary()["formula"])

    @property
    def family_name(self) -> str:
        return str(self.summary()["family_name"])

    @property
    def model_class(self) -> str:
        """Fitted model class string (e.g. ``"standard"``, ``"survival marginal-slope"``)."""
        return self._model_class_from_summary()

    @property
    def is_survival(self) -> bool:
        """``True`` if this is a survival-family model."""
        return self.model_class in _SURVIVAL_MODEL_CLASSES

    @property
    def is_marginal_slope(self) -> bool:
        """``True`` if this model was fit with a marginal-slope likelihood."""
        return self.model_class in _MARGINAL_SLOPE_MODEL_CLASSES

    @property
    def is_transformation_normal(self) -> bool:
        """``True`` if this is a conditional transformation-normal model."""
        return self.model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES

    @property
    def response_name(self) -> str | None:
        return response_column_name(self.formula)

    @property
    def training_table_kind(self) -> str | None:
        return self._training_table_kind

    def _model_class_from_summary(self) -> str:
        value = self.summary().get("model_class")
        if value is None:
            metadata = self.summary().get("metadata")
            if isinstance(metadata, dict):
                value = metadata.get("model_class")
        return str(value) if value is not None else "standard"

    def diagnose(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
    ) -> Diagnostics:
        columns, _kind = table_columns(data)
        response_name = y or self.response_name
        if response_name is None:
            raise ValueError("could not infer the response column; pass y='column_name'")
        if response_name not in columns:
            raise ValueError(
                f"response column '{response_name}' is missing from the diagnostic data"
            )
        prediction_columns = {
            name: values for name, values in columns.items() if name != response_name
        }
        predicted = self.predict(
            prediction_columns,
            interval=interval,
            return_type="dict",
        )
        observed = coerce_numeric_vector(columns[response_name], label=response_name)
        return Diagnostics.from_predictions(
            formula=self.formula,
            response_name=response_name,
            observed=observed,
            predicted=predicted,
        )

    def plot(
        self,
        data: Any,
        *,
        x: str | None = None,
        y: str | None = None,
        interval: float | None = 0.95,
        kind: str = "prediction",
        ax: Any | None = None,
    ) -> Any:
        import matplotlib.pyplot as plt

        columns, _table_kind = table_columns(data)
        diagnostics = self.diagnose(data, y=y, interval=interval if kind == "prediction" else None)
        if ax is None:
            _, ax = plt.subplots()

        if kind == "prediction":
            response_name = diagnostics.response_name
            candidate_columns = [
                name for name in columns if name != response_name
            ]
            x_name = x or (candidate_columns[0] if len(candidate_columns) == 1 else None)
            if x_name is None:
                raise ValueError("prediction plots require x='column_name' when multiple feature columns are present")
            if x_name not in columns:
                raise ValueError(f"plot column '{x_name}' is missing from the supplied data")
            x_values = coerce_numeric_vector(columns[x_name], label=x_name)
            ordering = sorted(range(len(x_values)), key=x_values.__getitem__)
            x_sorted = [x_values[index] for index in ordering]
            mean_sorted = [diagnostics.predicted["mean"][index] for index in ordering]
            ax.plot(x_sorted, mean_sorted, color="#1d4ed8", linewidth=2, label="mean")
            if diagnostics.interval_lower is not None and diagnostics.interval_upper is not None:
                lower = [diagnostics.interval_lower[index] for index in ordering]
                upper = [diagnostics.interval_upper[index] for index in ordering]
                ax.fill_between(x_sorted, lower, upper, color="#93c5fd", alpha=0.35, label="interval")
            if diagnostics.observed:
                observed_sorted = [diagnostics.observed[index] for index in ordering]
                ax.scatter(x_sorted, observed_sorted, color="#0f172a", s=18, alpha=0.7, label="observed")
            ax.set_xlabel(x_name)
            ax.set_ylabel(diagnostics.response_name or "response")
        elif kind == "residuals":
            ax.scatter(diagnostics.predicted["mean"], diagnostics.residuals, color="#0f172a", s=18, alpha=0.75)
            ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
            ax.set_xlabel("predicted mean")
            ax.set_ylabel("residual")
        elif kind == "observed_vs_predicted":
            ax.scatter(diagnostics.predicted["mean"], diagnostics.observed, color="#0f172a", s=18, alpha=0.75)
            lo = min(min(diagnostics.predicted["mean"]), min(diagnostics.observed))
            hi = max(max(diagnostics.predicted["mean"]), max(diagnostics.observed))
            ax.plot([lo, hi], [lo, hi], color="#94a3b8", linestyle="--", linewidth=1)
            ax.set_xlabel("predicted mean")
            ax.set_ylabel("observed")
        else:
            raise ValueError("plot kind must be one of: prediction, residuals, observed_vs_predicted")

        ax.set_title(f"{self.family_name} ({kind.replace('_', ' ')})")
        if kind == "prediction":
            ax.legend()
        return ax

    def __repr__(self) -> str:
        parts = [f"formula={self.formula!r}", f"family_name={self.family_name!r}"]
        if self._training_table_kind is not None:
            parts.append(f"training_table_kind={self._training_table_kind!r}")
        return f"Model({', '.join(parts)})"

    def _repr_html_(self) -> str:
        return self.report()


def _ordered_prediction_columns(columns: dict[str, list[float]]) -> dict[str, list[float]]:
    preferred = ["eta", "mean", "effective_se", "mean_lower", "mean_upper"]
    ordered: dict[str, list[float]] = {}
    for key in preferred:
        if key in columns:
            ordered[key] = columns[key]
    for key, value in columns.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _transformation_normal_z(columns: dict[str, list[float]]) -> list[float]:
    for candidate in ("z", "z_score", "transformed", "eta"):
        if candidate in columns:
            return list(columns[candidate])
    if "mean" in columns:
        return list(columns["mean"])
    raise KeyError(
        "transformation-normal prediction payload is missing a z-score column"
    )


def _validate_survival_chunk_size(value: int, name: str) -> int:
    chunk = int(value)
    if chunk <= 0:
        raise ValueError(f"{name} must be positive")
    return chunk


def _survival_prediction_from_columns(
    model_class: str, columns: dict[str, list[float]]
) -> SurvivalPrediction:
    import numpy as np

    parameter_names = [
        name
        for name in columns
        if name not in {"mean_lower", "mean_upper", "effective_se"}
    ]
    if not parameter_names:
        raise KeyError(
            f"survival prediction payload for '{model_class}' was empty"
        )
    stacked = np.column_stack(
        [np.asarray(columns[name], dtype=float) for name in parameter_names]
    )
    baseline = columns.get("baseline_times")
    baseline_arr = np.asarray(baseline, dtype=float) if baseline is not None else None
    return SurvivalPrediction(
        model_class=model_class,
        parameters=stacked,
        parameter_names=tuple(parameter_names),
        baseline_times=baseline_arr,
    )


def _survival_prediction_from_ffi_payload(
    parsed: dict[str, Any],
) -> SurvivalPrediction:
    """Build a :class:`SurvivalPrediction` from the FFI's dense payload."""
    import numpy as np

    model_class = str(parsed.get("model_class") or "survival marginal-slope")
    times = np.asarray(parsed.get("times") or [], dtype=float).reshape(-1)
    hazard = _coerce_matrix(parsed.get("hazard"))
    survival = _coerce_matrix(parsed.get("survival"))
    cumulative = _coerce_matrix(parsed.get("cumulative_hazard"))
    linear_predictor = np.asarray(
        parsed.get("linear_predictor") or [], dtype=float
    ).reshape(-1)
    columns = parsed.get("columns") or {}
    parameter_names = tuple(columns.keys())
    if parameter_names:
        stacked = np.column_stack(
            [np.asarray(columns[name], dtype=float) for name in parameter_names]
        )
    else:
        stacked = linear_predictor.reshape(-1, 1) if linear_predictor.size else np.zeros((0, 0))
    baseline = columns.get("baseline_times")
    baseline_arr = np.asarray(baseline, dtype=float) if baseline is not None else None
    return SurvivalPrediction(
        model_class=model_class,
        parameters=stacked,
        parameter_names=parameter_names,
        baseline_times=baseline_arr,
        times=times if times.size else None,
        hazard=hazard,
        survival=survival,
        cumulative_hazard=cumulative,
        linear_predictor=linear_predictor if linear_predictor.size else None,
    )


def _coerce_matrix(value: Any) -> Any:
    import numpy as np

    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _interpolate_rows(grid: Any, surface: Any, query: Any, *, clip: tuple) -> Any:
    """Linear interpolation of each row of ``surface`` against ``grid``.

    ``grid`` — 1-D time grid (must be sorted for ``numpy.interp``; we
    sort a copy if the FFI returned an unsorted grid, e.g. per-row
    exit times on a shuffled dataset).
    ``surface`` — ``(n_rows, len(grid))`` dense surface.
    ``query`` — 1-D query times.
    ``clip`` — ``(lo, hi)`` bounds applied to every output cell (either
    bound may be ``None`` for no clipping on that side).
    """
    import numpy as np

    grid = np.asarray(grid, dtype=float).reshape(-1)
    query = np.asarray(query, dtype=float).reshape(-1)
    surface = np.asarray(surface, dtype=float)
    if grid.size == 0 or surface.shape[1] != grid.size:
        raise ValueError("survival interpolation requires a non-empty grid")

    order = np.argsort(grid, kind="stable")
    sorted_grid = grid[order]
    sorted_surface = surface[:, order]

    out = np.empty((sorted_surface.shape[0], query.size), dtype=float)
    for row_idx in range(sorted_surface.shape[0]):
        out[row_idx, :] = np.interp(query, sorted_grid, sorted_surface[row_idx, :])
    lo, hi = clip
    if lo is not None or hi is not None:
        out = np.clip(out, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
    return out


__all__ = ["Model", "SurvivalPrediction"]
