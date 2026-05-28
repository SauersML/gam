"""Survival/competing-risks helpers and prediction containers.

Extracted from ``_model.py`` to keep that file's surface area focused on
the public :class:`Model` shell. The Rust core does the numerical work
behind ``Model.predict``; this module owns the lightweight per-row
hazard / survival interpolation and chunked CSV streaming that surround
those FFI payloads.
"""

from __future__ import annotations

import json
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
_BERNOULLI_FAMILY_PREFIXES = ("bernoulli", "binomial")


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

    times_arr = rust_module().survival_coerce_times(times)
    cumulative_hazards: list[Any] = []
    for idx, prediction in enumerate(prediction_seq):
        if not isinstance(prediction, SurvivalPrediction):
            raise TypeError(
                "competing_risks_cif expects SurvivalPrediction objects; "
                f"endpoint {idx} has type {type(prediction).__name__}"
            )
        cumulative_hazards.append(prediction.cumulative_hazard_at(times_arr))
    cif, overall_survival = rust_module().competing_risks_cif_from_predictions(
        times_arr,
        cumulative_hazards,
        names,
    )
    return CompetingRisksCIF(
        times=times_arr,
        cif=cif,
        overall_survival=overall_survival,
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
        return rust_module().survival_coerce_times(times)

    def _parameters_array(self) -> Any:
        return rust_module().survival_parameters_matrix(self.parameters)

    def _should_auto_chunk_dense(self, n_rows: int, n_times: int) -> bool:
        return int(n_rows) * int(n_times) > DENSE_SURVIVAL_AUTO_CHUNK_CELLS

    def _collect_chunks(self, chunks: Any, *, n_rows: int, n_times: int) -> Any:
        return rust_module().survival_collect_chunks(
            n_rows,
            n_times,
            [
                (
                    int(row_slice.start),
                    int(row_slice.stop),
                    int(time_slice.start),
                    int(time_slice.stop),
                    block,
                )
                for row_slice, time_slice, block in chunks
            ],
        )

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
        times_arr = self._coerce_times(times)
        cumulative = self._ffi_surface_at("cumulative_hazard", times_arr, clip=(0.0, None))
        if cumulative is not None:
            return cumulative
        survival = self.survival_at(times)
        return rust_module().survival_cumulative_from_survival(survival)

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
        left_value, right_value = _extrapolation_for(kind)
        if self._should_auto_chunk_dense(surface.shape[0], times_arr.size):
            clip_lo, clip_hi = clip
            return rust_module().survival_chunk_iter_collect(
                grid,
                surface,
                times_arr,
                kind,
                clip_lo,
                clip_hi,
                DEFAULT_SURVIVAL_PEOPLE_CHUNK,
                DEFAULT_SURVIVAL_TIME_GRID_CHUNK,
            )
        return _interpolate_rows(
            grid,
            surface,
            times_arr,
            clip=clip,
            left_value=left_value,
            right_value=right_value,
        )

    def _ffi_surface(self, kind: str) -> tuple[Any | None, Any | None]:
        if self.times is None:
            return (None, None)
        surface = getattr(self, kind, None)
        if surface is None:
            return (None, None)
        result = rust_module().survival_ffi_surface(self.times, surface)
        return (None, None) if result is None else result

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
        left_value, right_value = _extrapolation_for(kind)

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
                        _interpolate_rows(
                            grid,
                            row_surface,
                            time_block,
                            clip=clip,
                            left_value=left_value,
                            right_value=right_value,
                        ),
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
            yield (
                row_slice,
                time_slice,
                rust_module().survival_cumulative_from_survival(survival),
            )

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
    columns_json = json.dumps(columns, separators=(",", ":"))
    return json.loads(rust_module().ordered_prediction_columns(columns_json))


def numeric_matrix(values: Any, label: str) -> Any:
    return rust_module().numeric_matrix_validate(values, label)


def transformation_normal_z(columns: dict[str, list[float]]) -> list[float]:
    if not isinstance(columns, dict):
        raise TypeError("columns must be a dict")
    return list(
        rust_module().transformation_normal_z_from_columns(json.dumps(columns))
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
    # Strip the uncertainty / interval columns; what remains describes the
    # per-row survival parameter vector. ``std_error`` is the (renamed)
    # response-scale SE column (#310, formerly ``effective_se``); legacy
    # ``effective_se`` is filtered too so payloads produced by older
    # extensions during the migration window do not leak into the parameter
    # matrix.
    parameter_names = [
        name
        for name in columns
        if name
        not in {
            "mean_lower",
            "mean_upper",
            "std_error",
            "effective_se",
        }
    ]
    if not parameter_names:
        raise KeyError(
            f"survival prediction payload for '{model_class}' was empty"
        )
    stacked = rust_module().column_stack_f64(
        [[float(value) for value in columns[name]] for name in parameter_names]
    )
    return SurvivalPrediction(
        model_class=model_class,
        parameters=stacked,
        parameter_names=tuple(parameter_names),
        id_column=id_column,
        row_ids=row_ids,
    )


def survival_prediction_from_ffi_payload(
    raw: str,
    *,
    id_column: str | None = None,
    row_ids: Sequence[str] | None = None,
) -> SurvivalPrediction:
    payload = rust_module().survival_prediction_payload_from_json(raw)
    return SurvivalPrediction(
        **payload,
        id_column=id_column,
        row_ids=row_ids,
    )


def competing_risks_prediction_from_ffi_payload(
    raw: str,
) -> CompetingRisksPrediction:
    """Build a :class:`CompetingRisksPrediction` from the Rust FFI payload."""
    parsed = rust_module().competing_risks_prediction_payload_from_json(raw)
    return CompetingRisksPrediction(
        model_class=parsed["model_class"],
        likelihood_mode=parsed["likelihood_mode"],
        endpoint_names=tuple(parsed["endpoint_names"]),
        times=parsed["times"],
        hazard=parsed["hazard"],
        survival=parsed["survival"],
        cumulative_hazard=parsed["cumulative_hazard"],
        cif=parsed["cif"],
        overall_survival=parsed["overall_survival"],
        linear_predictor=parsed["linear_predictor"],
        columns=parsed["columns"],
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


def term_blocks_for_model(model_bytes: bytes) -> tuple[TermBlock, ...]:
    """Return per-term coefficient column ranges for a saved model."""
    return tuple(
        TermBlock(name=str(name), kind=str(kind), start=int(start), end=int(end))
        for name, kind, start, end in rust_module().term_blocks_for_model(model_bytes)
    )


def _interpolate_rows(
    grid: Any,
    surface: Any,
    query: Any,
    *,
    clip: tuple[float | None, float | None],
    left_value: float | None = None,
    right_value: float | None = None,
) -> Any:
    clip_lo, clip_hi = clip
    return rust_module().interpolate_rows(
        grid,
        surface,
        query,
        clip_lo, clip_hi,
        left_value,
        right_value,
    )


# Asymptotic boundary values for survival-related surfaces. Used when callers
# evaluate the surface strictly outside the modeled time grid: rather than
# clamping to the nearest endpoint (which is meaningful for an estimate but
# semantically wrong as an extrapolation), we return the value the surface
# *must* take in that regime by definition.
#   * survival(t) at t<=0 is 1 (no one has failed yet); at t->inf it is 0
#     (under the standard assumption that everyone eventually fails).
#   * cumulative hazard mirrors survival via H(t) = -log S(t).
#   * Hazard / standard-error surfaces have no canonical asymptote, so we
#     leave them on the default nearest-endpoint behavior.
_SURVIVAL_EXTRAPOLATION = {
    "survival": (1.0, 0.0),
    "cumulative_hazard": (0.0, None),
}


def _extrapolation_for(kind: str) -> tuple[float | None, float | None]:
    return _SURVIVAL_EXTRAPOLATION.get(kind, (None, None))


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
