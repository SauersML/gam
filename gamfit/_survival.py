"""Survival/competing-risks helpers and prediction containers.

Extracted from ``_model.py`` to keep that file's surface area focused on
the public :class:`Model` shell. The Rust core does the numerical work
behind ``Model.predict``; this module owns the lightweight per-row
hazard / survival interpolation and chunked CSV streaming that surround
those FFI payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._binding import rust_module

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
    """Per-term coefficient column range exposed by :attr:`Model.term_blocks`."""

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
        prediction_seq = tuple(predictions.values())
        default_names = tuple(str(name) for name in predictions)
    else:
        prediction_seq = tuple(predictions)
        default_names = tuple(f"endpoint_{idx + 1}" for idx in range(len(prediction_seq)))
    names = (
        default_names
        if endpoint_names is None
        else tuple(str(name) for name in endpoint_names)
    )

    times_arr = np.asarray(times, dtype=float).reshape(-1)
    cumulative_hazards = []
    for idx, prediction in enumerate(prediction_seq):
        if not isinstance(prediction, SurvivalPrediction):
            raise TypeError(
                "competing_risks_cif expects SurvivalPrediction objects; "
                f"endpoint {idx} has type {type(prediction).__name__}"
            )
        cumulative_hazards.append(
            np.asarray(prediction.cumulative_hazard_at(times_arr), dtype=float)
        )
    cif, overall_survival = rust_module().competing_risks_cif_from_predictions(
        times_arr,
        cumulative_hazards,
        names,
    )
    return CompetingRisksCIF(
        times=times_arr,
        cif=np.asarray(cif, dtype=float),
        overall_survival=np.asarray(overall_survival, dtype=float),
        cumulative_hazard=tuple(cumulative_hazards),
        endpoint_names=names,
    )


@dataclass(slots=True)
class SurvivalPrediction:
    """Per-row survival functions evaluated on demand."""

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
        return int(n_rows) * int(n_times) > DENSE_SURVIVAL_AUTO_CHUNK_CELLS

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
        return rust_module().hazard_from_cumulative(
            times_arr, cumulative, previous_cumulative, previous_time
        )

    def hazard_at(self, times: Any) -> Any:
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
        import numpy as np

        times_arr = self._coerce_times(times)
        cumulative = self._ffi_surface_at("cumulative_hazard", times_arr, clip=(0.0, None))
        if cumulative is not None:
            return cumulative
        survival = self.survival_at(times)
        survival = np.clip(survival, 1e-12, 1.0)
        return -np.log(survival)

    def survival_at(self, times: Any) -> Any:
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
        times_arr = self._coerce_times(times)
        grid, surface = self._ffi_surface("survival")
        if grid is not None and surface is not None:
            include_ids = self.id_column is not None and self.row_ids is not None
            return str(
                rust_module().write_survival_csv(
                    str(path),
                    grid,
                    surface,
                    times_arr,
                    self.id_column if include_ids else None,
                    list(self.row_ids) if include_ids else None,
                    people_chunk,
                    time_grid_chunk,
                )
            )

        import csv

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
        return rust_module().survival_block(params, times_arr)


def ordered_prediction_columns(columns: dict[str, list[float]]) -> dict[str, list[float]]:
    preferred = ["eta", "mean", "effective_se", "mean_lower", "mean_upper"]
    ordered: dict[str, list[float]] = {}
    for key in preferred:
        if key in columns:
            ordered[key] = columns[key]
    for key, value in columns.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def numeric_matrix(values: Any, label: str) -> Any:
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


def transformation_normal_z(columns: dict[str, list[float]]) -> list[float]:
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


def extract_row_ids(
    headers: list[str],
    rows: list[list[str]],
    id_column: str | None,
) -> list[str] | None:
    return rust_module().extract_row_ids(headers, rows, id_column)


def survival_prediction_from_columns(
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


def survival_prediction_from_ffi_payload(
    parsed: dict[str, Any],
    *,
    id_column: str | None = None,
    row_ids: Sequence[str] | None = None,
) -> SurvivalPrediction:
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


def competing_risks_prediction_from_ffi_payload(
    parsed: dict[str, Any],
) -> CompetingRisksPrediction:
    """Build a :class:`CompetingRisksPrediction` from the Rust FFI payload."""
    import numpy as np

    endpoint_names = tuple(str(name) for name in (parsed.get("endpoint_names") or ()))
    times = np.asarray(parsed.get("times") or [], dtype=float).reshape(-1)
    columns = parsed.get("columns") or {}
    return CompetingRisksPrediction(
        model_class=str(parsed.get("model_class") or "competing risks survival"),
        likelihood_mode=str(parsed.get("likelihood_mode") or ""),
        endpoint_names=endpoint_names,
        times=times if times.size else None,
        hazard=_coerce_matrix(parsed.get("hazard")),
        survival=_coerce_matrix(parsed.get("survival")),
        cumulative_hazard=_coerce_matrix(parsed.get("cumulative_hazard")),
        cif=_coerce_matrix(parsed.get("cif")),
        overall_survival=_coerce_matrix(parsed.get("overall_survival")),
        linear_predictor=np.asarray(
            parsed.get("linear_predictor") or [], dtype=float
        ).reshape(-1) if parsed.get("linear_predictor") is not None else None,
        columns={str(k): list(v) for k, v in columns.items()},
    )


def _coerce_matrix(value: Any) -> Any | None:
    import numpy as np

    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def shape_prediction_response(
    parsed: dict[str, Any],
    *,
    headers: list[str],
    rows: list[list[str]],
    table_kind: str | None,
    training_table_kind: str | None,
    fallback_model_class: str,
    interval: float | None,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    restore: Any,
) -> Any:
    """Convert a ``predict_table`` payload into the public return shape."""
    import numpy as np

    if parsed.get("class") == "survival_prediction":
        return survival_prediction_from_ffi_payload(
            parsed, id_column=id_column, row_ids=row_ids
        )
    if parsed.get("class") == "competing_risks_prediction":
        return competing_risks_prediction_from_ffi_payload(parsed)

    columns = ordered_prediction_columns(parsed["columns"])
    model_class = str(parsed.get("model_class") or fallback_model_class)

    if model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES:
        z = np.asarray(transformation_normal_z(columns), dtype=float)
        if id_column is None and return_type is None:
            return z
        out_columns: dict[str, list[Any]] = {"z": z.tolist()}
        if id_column is not None:
            out_columns = {id_column: list(row_ids or []), **out_columns}
        return restore(
            out_columns,
            requested=return_type,
            input_kind=table_kind,
            training_kind=training_table_kind,
        )

    if model_class == "bernoulli marginal-slope":
        probs = np.clip(np.asarray(columns.get("mean", []), dtype=float), 0.0, 1.0)
        if id_column is None and return_type is None:
            return probs
        out_columns = {"mean": probs.tolist()}
        if id_column is not None:
            out_columns = {id_column: list(row_ids or []), **out_columns}
        return restore(
            out_columns,
            requested=return_type,
            input_kind=table_kind,
            training_kind=training_table_kind,
        )

    if model_class in _SURVIVAL_MODEL_CLASSES:
        return survival_prediction_from_columns(
            model_class, columns, id_column=id_column, row_ids=row_ids
        )

    out_columns_any: dict[str, list[Any]] = dict(columns)
    if id_column is not None:
        out_columns_any = {id_column: list(row_ids or []), **out_columns_any}
    return restore(
        out_columns_any,
        requested=return_type,
        input_kind=table_kind,
        training_kind=training_table_kind,
    )


def default_survival_time_grid(
    model_class: str,
    formula: str,
    headers: list[str],
    rows: list[list[str]],
) -> list[float] | None:
    from ._binding import rust_module

    result = rust_module().default_survival_time_grid(
        model_class, formula, list(headers), [list(r) for r in rows]
    )
    return list(result) if result is not None else None


def term_blocks_from_state(state: dict[str, Any]) -> tuple[TermBlock, ...]:
    """Parse ``coefficient_state_json``'s ``term_blocks`` into :class:`TermBlock`."""
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
            raise ValueError(f"term block {idx} has invalid range [{start}, {end})")
        blocks.append(TermBlock(name=name, kind=kind, start=start, end=end))
    return tuple(blocks)


def _interpolate_rows(
    grid: Any,
    surface: Any,
    query: Any,
    *,
    clip: tuple[float | None, float | None],
) -> Any:
    import numpy as np
    from ._binding import rust_module

    clip_lo, clip_hi = clip
    return np.asarray(rust_module().interpolate_rows(
        np.asarray(grid, dtype=float).reshape(-1),
        np.asarray(surface, dtype=float),
        np.asarray(query, dtype=float).reshape(-1),
        clip_lo, clip_hi,
    ), dtype=float)


__all__ = [
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "DEFAULT_SURVIVAL_PEOPLE_CHUNK",
    "DEFAULT_SURVIVAL_TIME_GRID_CHUNK",
    "DENSE_SURVIVAL_AUTO_CHUNK_CELLS",
    "SurvivalPrediction",
    "TermBlock",
    "competing_risks_cif",
    "competing_risks_prediction_from_ffi_payload",
    "extract_row_ids",
    "numeric_matrix",
    "ordered_prediction_columns",
    "survival_prediction_from_columns",
    "survival_prediction_from_ffi_payload",
    "transformation_normal_z",
    "_MARGINAL_SLOPE_MODEL_CLASSES",
    "_SURVIVAL_MODEL_CLASSES",
    "_TRANSFORMATION_NORMAL_MODEL_CLASSES",
]
