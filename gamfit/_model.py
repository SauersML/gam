from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._sampling import PosteriorSamples
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
DENSE_SURVIVAL_AUTO_CHUNK_CELLS = 1_000_000

_SURVIVAL_MODEL_CLASSES = frozenset(
    {
        "survival",
        "competing risks survival",
        "survival marginal-slope",
        "survival location-scale",
        "latent survival",
    }
)
_SURVIVAL_TIME_GRID_MODEL_CLASSES = frozenset(
    {
        "survival",
        "competing risks survival",
        "survival marginal-slope",
        "survival location-scale",
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


@dataclass(frozen=True, slots=True)
class TermBlock:
    """Per-term coefficient column range, exposed by :attr:`Model.term_blocks`.

    ``[start, end)`` indexes into the joint coefficient vector / design
    matrix in the same order as Rust's ``coefficient_state_json`` payload.
    ``kind`` is one of ``"intercept"``, ``"linear"``, ``"random_effect"``,
    ``"smooth_bspline1d"``, ``"tensor"``, ``"thin_plate"``, ``"sphere"``,
    ``"matern"``, ``"duchon"``, ``"factor_smooth"``, or ``"by_smooth"``.
    """

    name: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CompetingRisksPrediction:
    """Rust-computed joint cause-specific competing-risks prediction."""

    model_class: str
    likelihood_mode: str
    endpoint_names: tuple[str, ...]
    times: Any
    hazard: Any
    survival: Any
    cumulative_hazard: Any
    cif: Any
    overall_survival: Any
    linear_predictor: Any
    columns: dict[str, list[float]]


@dataclass(frozen=True, slots=True)
class CompetingRisksCIF:
    """Cause-specific cumulative incidence assembled by the Rust core."""

    times: Any
    cif: Any
    overall_survival: Any
    cumulative_hazard: Any
    endpoint_names: tuple[str, ...]


def competing_risks_cif(
    predictions: Mapping[str, "SurvivalPrediction"] | Sequence["SurvivalPrediction"],
    *,
    times: Any,
    endpoint_names: Sequence[str] | None = None,
) -> CompetingRisksCIF:
    """Assemble competing-risks CIFs from cause-specific survival predictions."""
    import numpy as np

    if isinstance(predictions, Mapping):
        names = tuple(str(name) for name in predictions)
        prediction_seq = tuple(predictions.values())
        if endpoint_names is not None and tuple(endpoint_names) != names:
            raise ValueError("endpoint_names must match the supplied prediction mapping keys")
    else:
        prediction_seq = tuple(predictions)
        names = (
            tuple(f"endpoint_{idx + 1}" for idx in range(len(prediction_seq)))
            if endpoint_names is None
            else tuple(str(name) for name in endpoint_names)
        )
    if not prediction_seq:
        raise ValueError("competing_risks_cif requires at least one endpoint prediction")
    if len(names) != len(prediction_seq):
        raise ValueError("endpoint_names must match the number of endpoint predictions")
    if len(set(names)) != len(names):
        raise ValueError("endpoint_names must be unique")
    for idx, prediction in enumerate(prediction_seq):
        if not isinstance(prediction, SurvivalPrediction):
            raise TypeError(
                "competing_risks_cif expects SurvivalPrediction objects; "
                f"endpoint {idx} has type {type(prediction).__name__}"
            )

    times_arr = np.asarray(times, dtype=float).reshape(-1)
    cumulative_hazards = tuple(
        np.asarray(prediction.cumulative_hazard_at(times_arr), dtype=float)
        for prediction in prediction_seq
    )
    expected_shape = cumulative_hazards[0].shape
    if len(expected_shape) != 2:
        raise ValueError("endpoint predictions must return 2D cumulative hazard arrays")
    for cumulative_hazard in cumulative_hazards[1:]:
        if cumulative_hazard.shape != expected_shape:
            raise ValueError(
                "all endpoint predictions must return the same (n_rows, n_times) shape"
            )
    cif, overall_survival = rust_module().competing_risks_cif(times_arr, cumulative_hazards)
    return CompetingRisksCIF(
        times=times_arr,
        cif=np.asarray(cif, dtype=float),
        overall_survival=np.asarray(overall_survival, dtype=float),
        cumulative_hazard=cumulative_hazards,
        endpoint_names=names,
    )


@dataclass(slots=True)
class SurvivalPrediction:
    """Per-row survival functions evaluated on demand.

    Returned by :meth:`Model.predict` for survival models. The ``*_at``
    helpers interpolate the FFI hazard/survival/cumulative-hazard surface
    (or fall back to a plug-in piecewise-constant hazard from
    ``parameters``). For very large queries the dense matrix is assembled
    from ``*_at_chunks``; :meth:`write_survival_at_csv` streams to disk.

    ``survival_se`` / ``eta_se`` are populated only when
    :meth:`Model.predict` was called with ``with_uncertainty=True``.
    """

    model_class: str
    parameters: Any
    parameter_names: Sequence[str] = field(default_factory=tuple)
    times: Any | None = None
    hazard: Any | None = None
    survival: Any | None = None
    cumulative_hazard: Any | None = None
    linear_predictor: Any | None = None
    id_column: str | None = None
    row_ids: Sequence[str] | None = None
    survival_se: Any | None = None
    eta_se: Any | None = None

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

    def _should_auto_chunk_dense(self, n_rows: int, n_times: int) -> bool:
        cells = int(n_rows) * int(n_times)
        return cells > DENSE_SURVIVAL_AUTO_CHUNK_CELLS

    def _collect_chunks(self, chunks: Any, *, n_rows: int, n_times: int) -> Any:
        import numpy as np

        dense = np.empty((n_rows, n_times), dtype=float)
        for row_slice, time_slice, block in chunks:
            dense[row_slice, time_slice] = block
        return dense

    def _prediction_row_count(self) -> int:
        for kind in ("hazard", "cumulative_hazard", "survival"):
            _grid, surface = self._ffi_surface(kind)
            if surface is not None:
                return int(surface.shape[0])
        return int(self._parameters_array().shape[0])

    def _hazard_from_cumulative(
        self,
        times_arr: Any,
        cumulative: Any,
        *,
        previous_cumulative: Any | None = None,
        previous_time: float = 0.0,
    ) -> Any:
        import numpy as np

        if previous_cumulative is None:
            previous_cumulative = np.zeros((cumulative.shape[0], 1), dtype=float)
        grid = np.concatenate([[float(previous_time)], times_arr])
        widths = np.diff(grid)
        widths = np.where(widths <= 0.0, 1.0, widths)
        cumulative_full = np.concatenate([previous_cumulative, cumulative], axis=1)
        return np.diff(cumulative_full, axis=1) / widths.reshape(1, -1)

    def hazard_at(self, times: Any) -> Any:
        """Evaluate the hazard rate ``h(t)`` at each requested time.

        Returns an ``(n_samples, len(times))`` numpy array. Large requests are
        evaluated in chunks internally before assembling the dense result.
        """
        times_arr = self._coerce_times(times)
        hazard = self._ffi_surface_at("hazard", times_arr, clip=(0.0, None))
        if hazard is not None:
            return hazard

        n_rows = self._prediction_row_count()
        if self._should_auto_chunk_dense(n_rows, times_arr.size):
            return self._collect_chunks(
                self.hazard_at_chunks(times_arr),
                n_rows=n_rows,
                n_times=times_arr.size,
            )
        cumulative = self.cumulative_hazard_at(times_arr)
        return self._hazard_from_cumulative(times_arr, cumulative)

    def cumulative_hazard_at(self, times: Any) -> Any:
        """Cumulative hazard ``H(t) = -log S(t)`` at each requested time."""
        import numpy as np

        times_arr = self._coerce_times(times)
        cumulative = self._ffi_surface_at("cumulative_hazard", times_arr, clip=(0.0, None))
        if cumulative is not None:
            return cumulative

        survival = self.survival_at(times)
        survival = np.clip(survival, 1e-12, 1.0)
        return -np.log(survival)

    def survival_at(self, times: Any) -> Any:
        """Evaluate the survival probability ``S(t)`` at each requested time.

        When the FFI produced a dense hazard/survival surface we
        linearly interpolate against the returned grid. Otherwise we
        fall back to the plug-in identity ``S(t) = exp(-H(t))`` using a
        per-row piecewise-constant hazard derived from ``parameters``
        for bare-dataclass construction. Large requests are evaluated in
        chunks internally before assembling the dense result.
        """
        times_arr = self._coerce_times(times)
        survival = self._ffi_surface_at("survival", times_arr, clip=(0.0, 1.0))
        if survival is not None:
            return survival

        params = self._parameters_array()
        if self._should_auto_chunk_dense(params.shape[0], times_arr.size):
            return self._collect_chunks(
                self.survival_at_chunks(times_arr),
                n_rows=params.shape[0],
                n_times=times_arr.size,
            )
        return self._survival_block(params, times_arr)

    def survival_se_at(self, times: Any) -> Any | None:
        """Delta-method standard error on ``S(t)`` at each requested time.

        Returns ``None`` when the prediction was not issued with
        ``with_uncertainty=True`` (or the model class does not yet
        support response-scale uncertainty).  When available, the
        returned array has shape ``(n_samples, len(times))`` and is
        clipped to be non-negative.
        """
        if self.survival_se is None:
            return None
        times_arr = self._coerce_times(times)
        return self._ffi_surface_at("survival_se", times_arr, clip=(0.0, None))

    def _ffi_surface_at(
        self,
        kind: str,
        times_arr: Any,
        *,
        clip: tuple[float | None, float | None],
    ) -> Any | None:
        grid, surface = self._ffi_surface(kind)
        if grid is None or surface is None:
            return None
        if self._should_auto_chunk_dense(surface.shape[0], times_arr.size):
            return self._collect_chunks(
                self._ffi_surface_at_chunks(
                    kind,
                    times_arr,
                    clip=clip,
                    people_chunk=DEFAULT_SURVIVAL_PEOPLE_CHUNK,
                    time_grid_chunk=DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
                ),
                n_rows=surface.shape[0],
                n_times=times_arr.size,
            )
        return _interpolate_rows(grid, surface, times_arr, clip=clip)

    def _ffi_surface(self, kind: str) -> tuple[Any | None, Any | None]:
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

    def _ffi_surface_at_chunks(
        self,
        kind: str,
        times_arr: Any,
        *,
        clip: tuple[float | None, float | None],
        people_chunk: int,
        time_grid_chunk: int,
    ) -> Any | None:
        grid, surface = self._ffi_surface(kind)
        if grid is None or surface is None:
            return None
        people_chunk = _validate_survival_chunk_size(people_chunk, "people_chunk")
        time_grid_chunk = _validate_survival_chunk_size(time_grid_chunk, "time_grid_chunk")

        def chunks() -> Any:
            for row_start in range(0, surface.shape[0], people_chunk):
                row_stop = min(row_start + people_chunk, surface.shape[0])
                row_surface = surface[row_start:row_stop, :]
                for time_start in range(0, times_arr.size, time_grid_chunk):
                    time_stop = min(time_start + time_grid_chunk, times_arr.size)
                    time_block = times_arr[time_start:time_stop]
                    yield (
                        slice(row_start, row_stop),
                        slice(time_start, time_stop),
                        _interpolate_rows(grid, row_surface, time_block, clip=clip),
                    )

        return chunks()

    def survival_at_chunks(
        self,
        times: Any,
        *,
        people_chunk: int = DEFAULT_SURVIVAL_PEOPLE_CHUNK,
        time_grid_chunk: int = DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
    ) -> Any:
        """Yield ``(row_slice, time_slice, survival_block)`` chunks."""
        times_arr = self._coerce_times(times)
        ffi_chunks = self._ffi_surface_at_chunks(
            "survival",
            times_arr,
            clip=(0.0, 1.0),
            people_chunk=people_chunk,
            time_grid_chunk=time_grid_chunk,
        )
        if ffi_chunks is not None:
            yield from ffi_chunks
            return

        params = self._parameters_array()
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
        """Yield ``(row_slice, time_slice, H_block)`` cumulative-hazard chunks."""
        import numpy as np

        times_arr = self._coerce_times(times)
        ffi_chunks = self._ffi_surface_at_chunks(
            "cumulative_hazard",
            times_arr,
            clip=(0.0, None),
            people_chunk=people_chunk,
            time_grid_chunk=time_grid_chunk,
        )
        if ffi_chunks is not None:
            yield from ffi_chunks
            return

        for row_slice, time_slice, survival in self.survival_at_chunks(
            times_arr,
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
        times_arr = self._coerce_times(times)
        ffi_chunks = self._ffi_surface_at_chunks(
            "hazard",
            times_arr,
            clip=(0.0, None),
            people_chunk=people_chunk,
            time_grid_chunk=time_grid_chunk,
        )
        if ffi_chunks is not None:
            yield from ffi_chunks
            return

        previous_row_key: tuple[int | None, int | None] | None = None
        previous_cumulative = None
        previous_time = 0.0
        for row_slice, time_slice, cumulative in self.cumulative_hazard_at_chunks(
            times_arr,
            people_chunk=people_chunk,
            time_grid_chunk=time_grid_chunk,
        ):
            row_key = (row_slice.start, row_slice.stop)
            if previous_row_key != row_key or time_slice.start == 0:
                previous_cumulative = None
                previous_time = 0.0
                previous_row_key = row_key
            time_block = times_arr[time_slice]
            yield (
                row_slice,
                time_slice,
                self._hazard_from_cumulative(
                    time_block,
                    cumulative,
                    previous_cumulative=previous_cumulative,
                    previous_time=previous_time,
                ),
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
        """Stream ``S(t)`` to a CSV file via :meth:`survival_at_chunks`.

        Writes ``row, time, survival`` (or ``row, <id_column>, time, survival``
        when ``id_column`` was set on :meth:`Model.predict`).
        """
        import csv

        times_arr = self._coerce_times(times)
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if self.id_column is not None and self.row_ids is not None:
                writer.writerow(["row", self.id_column, "time", "survival"])
            else:
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
                        if self.id_column is not None and self.row_ids is not None:
                            writer.writerow(
                                [
                                    row_index,
                                    self.row_ids[row_index],
                                    float(time),
                                    float(survival),
                                ]
                            )
                        else:
                            writer.writerow([row_index, float(time), float(survival)])
        return str(path)

    def _survival_block(self, params: Any, times_arr: Any) -> Any:
        import numpy as np

        anchor_log_hazard = params[:, 0:1]
        hazard = np.exp(anchor_log_hazard)
        cumulative = hazard * times_arr.reshape(1, -1)
        return np.exp(-cumulative)


class Model:
    __slots__ = ("_model_bytes", "_training_table_kind", "_summary_cache")

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
        id_column: str | None = None,
        with_uncertainty: bool = False,
    ) -> Any:
        """Predict from ``data``.

        Returns a table (standard/Gaussian/Binomial), z-score array
        (transformation-normal), probability array (bernoulli-marginal-slope),
        or :class:`SurvivalPrediction` (survival models). When
        ``with_uncertainty=True`` (survival only) the prediction carries
        delta-method ``survival_se`` and ``eta_se``.
        """
        headers, rows, table_kind = normalize_table(data)
        row_ids = _extract_row_ids(headers, rows, id_column)
        payload: dict[str, Any] = {"interval": interval}
        if with_uncertainty:
            payload["with_uncertainty"] = True
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
            return _survival_prediction_from_ffi_payload(
                parsed,
                id_column=id_column,
                row_ids=row_ids,
            )
        if parsed.get("class") == "competing_risks_prediction":
            return _competing_risks_prediction_from_ffi_payload(parsed)

        columns = _ordered_prediction_columns(parsed["columns"])
        model_class = str(parsed.get("model_class") or self._model_class_from_payload())

        if model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES:
            import numpy as np

            z = np.asarray(_transformation_normal_z(columns), dtype=float)
            # Default contract: bare per-row z-score array. An explicit
            # `return_type=` or an `id_column` opts into the table-style
            # return so the column survives a pandas/dict roundtrip.
            if id_column is None and return_type is None:
                return z
            out_columns = {"z": z.tolist()}
            if id_column is not None:
                out_columns = {id_column: row_ids or [], **out_columns}
            return restore_output_table(
                out_columns,
                requested=return_type,
                input_kind=table_kind,
                training_kind=self._training_table_kind,
            )

        if model_class == "bernoulli marginal-slope":
            import numpy as np

            probs = np.clip(
                np.asarray(columns.get("mean", []), dtype=float), 0.0, 1.0
            )
            # Default contract: bare per-row probability array. An explicit
            # `return_type=` or an `id_column` opts into the table-style
            # return so callers that asked for a table get one.
            if id_column is None and return_type is None:
                return probs
            out_columns = {"mean": probs.tolist()}
            if id_column is not None:
                out_columns = {id_column: row_ids or [], **out_columns}
            return restore_output_table(
                out_columns,
                requested=return_type,
                input_kind=table_kind,
                training_kind=self._training_table_kind,
            )

        if model_class in _SURVIVAL_MODEL_CLASSES:
            return _survival_prediction_from_columns(
                model_class,
                columns,
                id_column=id_column,
                row_ids=row_ids,
            )

        out_columns_any: dict[str, list[Any]] = dict(columns)
        if id_column is not None:
            out_columns_any = {id_column: list(row_ids or []), **out_columns_any}
        return restore_output_table(
            out_columns_any,
            requested=return_type,
            input_kind=table_kind,
            training_kind=self._training_table_kind,
        )

    def predict_array(
        self,
        X: Any,
        *,
        interval: float | None = None,
    ) -> Any:
        """Predict directly from a numeric NumPy-compatible feature matrix.

        Columns are named ``x0``, ``x1``, ... at the Rust formula boundary.
        The return value is a dense NumPy array with columns ordered as
        ``eta``, ``mean``, then any uncertainty columns.
        """
        import numpy as np

        X_arr = _numeric_matrix(X, "X")
        payload: dict[str, Any] = {"interval": interval}
        try:
            return np.asarray(
                rust_module().predict_array(
                    self._model_bytes,
                    X_arr,
                    json.dumps(payload),
                ),
                dtype=float,
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def summary(self) -> Summary:
        """Return the model :class:`Summary` (coefficients, family, deviance, REML score)."""
        if self._summary_cache is None:
            try:
                payload = json.loads(rust_module().summary_json(self._model_bytes))
            except Exception as exc:
                raise map_exception(exc) from exc
            self._summary_cache = Summary.from_dict(payload)
        return self._summary_cache

    def smoothing_parameters(self) -> dict[int, float]:
        """Return fitted smoothing/precision parameters by penalty index."""
        lambdas = self.summary().get("lambdas")
        if not isinstance(lambdas, list):
            return {}
        return {idx: float(value) for idx, value in enumerate(lambdas)}

    def check(self, data: Any) -> SchemaCheck:
        """Validate ``data`` against the training schema (no prediction)."""
        headers, rows, _table_kind = normalize_table(data)
        try:
            payload = json.loads(rust_module().check_json(self._model_bytes, headers, rows))
        except Exception as exc:
            raise map_exception(exc) from exc
        return SchemaCheck.from_dict(payload)

    def report(self, path: str | Path | None = None) -> str:
        """Render a standalone HTML report; writes to ``path`` if given else returns the HTML."""
        try:
            html = rust_module().report_html(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        if path is None:
            return str(html)
        Path(path).write_text(html, encoding="utf-8")
        return str(path)

    def sample(
        self,
        data: Any,
        *,
        samples: int | None = None,
        warmup: int | None = None,
        chains: int | None = None,
        target_accept: float | None = None,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Draw NUTS posterior samples; returns :class:`PosteriorSamples`.

        Defaults are dimension-aware (see ``gam::hmc::NutsConfig::for_dimension``).
        """
        headers, rows, _table_kind = normalize_table(data)
        payload: dict[str, Any] = {}
        if samples is not None:
            payload["samples"] = int(samples)
        if warmup is not None:
            payload["warmup"] = int(warmup)
        if chains is not None:
            payload["chains"] = int(chains)
        if target_accept is not None:
            payload["target_accept"] = float(target_accept)
        if seed is not None:
            payload["seed"] = int(seed)
        # Always pass a JSON document so the Rust side has a single decode
        # path; an empty object is fine because every field is optional.
        options_json = json.dumps(payload)
        try:
            raw = rust_module().sample_table(
                self._model_bytes,
                headers,
                rows,
                options_json,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        # Bundle the saved-model bytes into the posterior so downstream
        # methods like `posterior.predict(new_data)` can fetch a design
        # matrix without the caller passing the model back in.
        return PosteriorSamples.from_ffi_json(raw, model_bytes=self._model_bytes)

    def design_matrix(self, data: Any) -> Any:
        """Return the ``(n_rows, n_coeffs)`` design matrix for ``data`` (standard models only)."""
        import numpy as np

        headers, rows, _ = normalize_table(data)
        try:
            raw = rust_module().design_matrix_table(self._model_bytes, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        parsed = json.loads(raw)
        n_rows = int(parsed["n_rows"])
        n_cols = int(parsed["n_cols"])
        flat = np.asarray(parsed.get("x_flat", []), dtype=float)
        if flat.size != n_rows * n_cols:
            raise ValueError(
                "design matrix FFI payload shape mismatch: "
                f"got {flat.size} floats, expected {n_rows} * {n_cols}"
            )
        return flat.reshape(n_rows, n_cols)

    def design_matrix_array(self, X: Any) -> Any:
        """Materialised design matrix for a numeric feature matrix."""
        import numpy as np

        X_arr = _numeric_matrix(X, "X")
        try:
            return np.asarray(
                rust_module().design_matrix_array(self._model_bytes, X_arr),
                dtype=float,
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def difference_smooth(
        self,
        *,
        view: str,
        group: str | None = None,
        pairs: Sequence[tuple[Any, Any]] | None = None,
        n: int = 100,
        level: float = 0.95,
        simultaneous: bool = False,
        n_sim: int = 10_000,
        seed: int | None = 12345,
        marginalise_random: bool = True,
        group_means: bool = True,
        data: Any | None = None,
        return_type: str | None = None,
    ) -> Any:
        """Covariance-aware pairwise difference smooths.

        Builds two model matrices on a grid, subtracts them, and uses the
        fitted joint coefficient covariance for pointwise bands. With
        ``simultaneous=True`` the band critical value is estimated from
        posterior coefficient simulation using the max standardized deviation
        across the whole grid.
        """
        import json

        request: dict[str, Any] = {
            "view": view,
            "group": group,
            "pairs": pairs,
            "n": int(n),
            "level": float(level),
            "simultaneous": bool(simultaneous),
            "n_sim": int(n_sim),
            "seed": seed,
            "marginalise_random": bool(marginalise_random),
            "group_means": bool(group_means),
        }
        if data is not None:
            headers, rows, _ = normalize_table(data)
            if rows:
                request["template"] = {h: rows[0][i] for i, h in enumerate(headers)}
        try:
            raw = rust_module().difference_smooth_json(self._model_bytes, json.dumps(request))
        except Exception as exc:
            raise map_exception(exc) from exc
        rows_out = json.loads(raw)
        if return_type == "list":
            return rows_out
        try:
            import pandas as pd
            return pd.DataFrame(rows_out)
        except Exception:
            return rows_out

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to ``path`` (round-trips via :func:`gamfit.load`)."""
        Path(path).write_bytes(self._model_bytes)

    def extend_with_group(
        self,
        new_group_spec: dict[str, Any],
        metadata: Any | None = None,
        prior: Any | None = None,
    ) -> "Model":
        """Return a no-refit model extended with deployment-time group levels.

        ``new_group_spec`` currently targets an existing random-effect term:
        ``{"kind": "random-effect-level", "term": "group_term", "level": "new"}``
        or ``{"term": "group_term", "levels": ["a", "b"]}``.  The returned
        model reuses the fitted coefficients and inserts zero-initialized
        coefficients, or ``prior["mean"]`` / ``prior["mu"]`` when supplied.
        """
        if not isinstance(new_group_spec, dict):
            raise TypeError("new_group_spec must be a dict")
        payload = dict(new_group_spec)
        if metadata is not None:
            payload["metadata"] = metadata
        if prior is not None:
            payload["prior"] = prior
        try:
            model_bytes = bytes(
                rust_module().extend_model_with_group(
                    self._model_bytes,
                    json.dumps(payload),
                )
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return Model(
            _model_bytes=model_bytes,
            _training_table_kind=self._training_table_kind,
        )

    def dumps(self) -> bytes:
        """Return the serialised model as raw bytes (inverse of :func:`gamfit.loads`)."""
        return self._model_bytes

    def _saved_model_payload(self) -> dict[str, Any]:
        saved = json.loads(self._model_bytes)
        if not isinstance(saved, dict):
            raise ValueError("saved model payload must be a JSON object")
        payload = saved.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("saved model payload is missing its payload object")
        return payload

    def _saved_payload_string(self, key: str) -> str | None:
        value = self._saved_model_payload().get(key)
        return str(value) if value is not None else None

    @property
    def formula(self) -> str:
        """The fitted Wilkinson-style formula string."""
        value = self._saved_payload_string("formula")
        if value is None:
            raise ValueError("saved model payload is missing formula")
        return value

    @property
    def family_name(self) -> str:
        """Human-readable family + link name (e.g. ``"Gaussian Identity"``)."""
        return str(self.summary()["family_name"])

    @property
    def model_class(self) -> str:
        """Fitted model class string (e.g. ``"standard"``, ``"survival marginal-slope"``)."""
        return self._model_class_from_payload()

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
        """Name of the response column, inferred from the formula.

        Returns ``None`` for survival formulas (``Surv(...)``) and other
        cases where the left-hand side isn't a single identifier.
        """
        return response_column_name(self.formula)

    @property
    def training_table_kind(self) -> str | None:
        """The kind of table the model was fit on.

        One of ``"pandas"``, ``"polars"``, ``"pyarrow"``, ``"numpy"``,
        ``"mapping"`` (dict of columns), ``"records"`` (list of dicts),
        ``"rows"`` (2-D sequence), or ``None`` if the input kind wasn't
        retained.  Used as a default ``return_type`` for :meth:`predict`
        and :meth:`diagnose`.
        """
        return self._training_table_kind

    @property
    def group_metadata(self) -> dict[str, Any] | None:
        """Per-group metadata persisted with the fitted model, if present."""
        value = self.summary().get("group_metadata")
        return dict(value) if isinstance(value, dict) else None

    @property
    def deployment_extensions(self) -> tuple[dict[str, Any], ...]:
        """No-refit group extensions applied after fitting."""
        value = self.summary().get("deployment_extensions")
        if not isinstance(value, list):
            return ()
        return tuple(dict(item) for item in value if isinstance(item, dict))

    @property
    def term_blocks(self) -> tuple[TermBlock, ...]:
        """Per-term coefficient column ranges in fitted coefficient order."""
        try:
            state = json.loads(rust_module().coefficient_state_json(self._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc
        raw_blocks = state.get("term_blocks")
        if not isinstance(raw_blocks, list):
            raise ValueError("coefficient state payload is missing term_blocks")
        blocks: list[TermBlock] = []
        for idx, raw_block in enumerate(raw_blocks):
            if not isinstance(raw_block, dict):
                raise ValueError(f"term block {idx} must be an object")
            try:
                name = str(raw_block["name"])
                kind = str(raw_block["kind"])
                start = int(raw_block["start"])
                end = int(raw_block["end"])
            except KeyError as exc:
                raise ValueError(f"term block {idx} is missing {exc.args[0]!r}") from exc
            if start < 0 or end < start:
                raise ValueError(
                    f"term block {idx} has invalid range [{start}, {end})"
                )
            blocks.append(TermBlock(name=name, kind=kind, start=start, end=end))
        return tuple(blocks)

    @property
    def evidence(self) -> float:
        """REML/LAML log marginal-likelihood (alias for ``summary()["reml_score"]``)."""
        value = self.summary().get("reml_score")
        if value is None:
            raise ValueError("saved model payload is missing reml_score")
        return float(value)

    def bayes_factor_vs(self, other: "Model") -> float:
        """Bayes factor ``exp(self.evidence - other.evidence)`` against ``other``."""
        if not isinstance(other, Model):
            raise TypeError(
                f"bayes_factor_vs expects a gamfit.Model, got {type(other).__name__}"
            )
        return math.exp(self.evidence - other.evidence)

    def _model_class_from_payload(self) -> str:
        value = self._saved_payload_string("model_kind")
        if value is None:
            raise ValueError("saved model payload is missing model_kind")
        return value

    def _default_survival_time_grid(
        self, headers: list[str], rows: list[list[str]]
    ) -> list[float] | None:
        if self.model_class not in _SURVIVAL_TIME_GRID_MODEL_CLASSES:
            return None
        try:
            grid = rust_module().default_survival_time_grid(self.formula, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        return list(grid) if grid is not None else None

    def diagnose(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
    ) -> Diagnostics:
        """Score the fitted model on held-out ``data`` and return :class:`Diagnostics`."""
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
        """Diagnostic plot via matplotlib (prediction, residuals, observed_vs_predicted)."""
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
            ordering = sorted(range(len(x_values)), key=lambda index: x_values[index])
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


def _numeric_matrix(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D numeric array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


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


def _extract_row_ids(
    headers: list[str],
    rows: list[list[str]],
    id_column: str | None,
) -> list[str] | None:
    if id_column is None:
        return None
    if id_column not in headers:
        raise ValueError(f"id_column '{id_column}' is missing from prediction data")
    index = headers.index(id_column)
    return [row[index] for row in rows]


def _survival_prediction_from_columns(
    model_class: str,
    columns: dict[str, list[float]],
    *,
    id_column: str | None = None,
    row_ids: Sequence[str] | None = None,
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
    return SurvivalPrediction(
        model_class=model_class,
        parameters=stacked,
        parameter_names=tuple(parameter_names),
        id_column=id_column,
        row_ids=row_ids,
    )


def _survival_prediction_from_ffi_payload(
    parsed: dict[str, Any],
    *,
    id_column: str | None = None,
    row_ids: Sequence[str] | None = None,
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
    survival_se = _coerce_matrix(parsed.get("survival_se"))
    eta_se_raw = parsed.get("eta_se")
    eta_se = (
        np.asarray(eta_se_raw, dtype=float).reshape(-1)
        if eta_se_raw is not None
        else None
    )
    columns = parsed.get("columns") or {}
    parameter_names = tuple(columns.keys())
    if parameter_names:
        stacked = np.column_stack(
            [np.asarray(columns[name], dtype=float) for name in parameter_names]
        )
    else:
        stacked = linear_predictor.reshape(-1, 1) if linear_predictor.size else np.zeros((0, 0))
    return SurvivalPrediction(
        model_class=model_class,
        parameters=stacked,
        parameter_names=parameter_names,
        times=times if times.size else None,
        hazard=hazard,
        survival=survival,
        cumulative_hazard=cumulative,
        linear_predictor=linear_predictor if linear_predictor.size else None,
        id_column=id_column,
        row_ids=row_ids,
        survival_se=survival_se,
        eta_se=eta_se if eta_se is not None and eta_se.size else None,
    )


def _coerce_matrix(value: Any) -> Any | None:
    import numpy as np

    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _interpolate_rows(
    grid: Any,
    surface: Any,
    query: Any,
    *,
    clip: tuple[float | None, float | None],
) -> Any:
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

    grid_arr = np.asarray(grid, dtype=float).reshape(-1)
    query_arr = np.asarray(query, dtype=float).reshape(-1)
    surface_arr = np.ascontiguousarray(np.asarray(surface, dtype=float))
    if grid_arr.size == 0 or surface_arr.shape[1] != grid_arr.size:
        raise ValueError("survival interpolation requires a non-empty grid")
    lo, hi = clip
    return np.asarray(
        rust_module().interpolate_survival_surface(
            grid_arr, surface_arr, query_arr, lo, hi
        ),
        dtype=float,
    )


__all__ = [
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "SurvivalPrediction",
    "competing_risks_cif",
]
