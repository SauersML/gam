from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._sampling import PairedPosteriorSamples, PosteriorSamples
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

    Returned by :meth:`Model.predict` for survival-family models. The
    ``*_at`` helpers (:meth:`hazard_at`, :meth:`cumulative_hazard_at`,
    :meth:`survival_at`, :meth:`failure_at`) evaluate the fitted hazard
    surface at any user-supplied time grid.

    When the FFI produced a dense ``(n_samples, n_times)`` grid of
    hazard / survival / cumulative-hazard values, the ``*_at`` helpers
    linearly interpolate against that grid. Otherwise they fall back to
    the legacy plug-in piecewise-constant hazard reconstructed from
    ``parameters`` so bare-dataclass construction keeps working.

    For very large queries (``n_rows * n_times`` exceeds roughly one
    million cells), the ``*_at`` helpers internally evaluate the surface
    in blocks via the matching ``*_at_chunks`` generator and then
    assemble the dense result; callers that want to avoid the dense
    allocation can iterate the chunk generators directly or stream a CSV
    with :meth:`write_survival_at_csv`.

    Attributes
    ----------
    model_class : str
        The fitted model class string (e.g. ``"survival marginal-slope"``).
    parameters : ndarray
        Flat per-row parameters returned by the FFI. Shape
        ``(n_samples, n_params_per_row)``. The exact column semantics
        depend on ``model_class``; callers should treat this as opaque
        and prefer the ``*_at`` helpers.
    parameter_names : tuple of str
        Column names corresponding to ``parameters``, in order.
    times : ndarray or None
        Shared 1-D time grid at which the hazard surfaces were evaluated.
    hazard : ndarray or None
        ``(n_samples, len(times))`` dense hazard surface from the FFI.
    survival : ndarray or None
        ``(n_samples, len(times))`` dense survival surface from the FFI.
    cumulative_hazard:
        ``(n_samples, len(times))`` dense cumulative-hazard surface from the FFI.
    linear_predictor:
        ``(n_samples,)`` per-row linear predictor at each row's own exit time.
    survival_se:
        ``(n_samples, len(times))`` delta-method standard errors on the
        survival surface (response scale).  ``None`` unless the prediction
        was issued with ``with_uncertainty=True``; then populated for
        location-scale survival models.
    eta_se:
        ``(n_samples,)`` delta-method SE on the linear predictor at each
        row's own exit time, under the same conditions as ``survival_se``.
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
        """Yield ``H(t)`` evaluations in row/time blocks.

        Streaming counterpart to :meth:`cumulative_hazard_at`. When the
        FFI provided a dense cumulative-hazard surface this iterates
        that surface directly; otherwise it derives ``H(t)`` from each
        survival block returned by :meth:`survival_at_chunks`.

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times.
        people_chunk : int, optional
            Maximum number of rows per yielded block. Defaults to
            ``DEFAULT_SURVIVAL_PEOPLE_CHUNK``.
        time_grid_chunk : int, optional
            Maximum number of time points per yielded block. Defaults
            to ``DEFAULT_SURVIVAL_TIME_GRID_CHUNK``.

        Yields
        ------
        tuple of (slice, slice, ndarray)
            ``(row_slice, time_slice, block)`` of cumulative-hazard
            values with shape matching the slice extents.

        Examples
        --------
        >>> for r, t, H_block in pred.cumulative_hazard_at_chunks(times):
        ...     handle.write(H_block.tobytes())

        See Also
        --------
        SurvivalPrediction.cumulative_hazard_at
        SurvivalPrediction.survival_at_chunks
        """
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
        """Stream survival predictions to a CSV file.

        Iterates :meth:`survival_at_chunks` and writes one row per
        ``(prediction_row, time)`` pair, avoiding materialising the full
        ``(n_samples, len(times))`` matrix in memory. When the
        prediction was issued with an ``id_column`` (via
        :meth:`Model.predict`), that column is included.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination CSV file. Overwritten if it already exists.
        times : array_like
            1-D sequence of finite, non-negative times at which to
            evaluate ``S(t)``.
        people_chunk : int, optional
            Maximum number of rows per internal block. Defaults to
            ``DEFAULT_SURVIVAL_PEOPLE_CHUNK``.
        time_grid_chunk : int, optional
            Maximum number of time points per internal block. Defaults
            to ``DEFAULT_SURVIVAL_TIME_GRID_CHUNK``.

        Returns
        -------
        str
            The string form of ``path``.

        Notes
        -----
        Columns written are ``row, time, survival`` (or
        ``row, <id_column>, time, survival`` when an id column is
        present). The file is opened in text mode with UTF-8 encoding.

        Examples
        --------
        >>> import numpy as np
        >>> pred = model.predict(test_df, id_column="patient_id")
        >>> pred.write_survival_at_csv(
        ...     "survival.csv", np.linspace(0.0, 10.0, 64)
        ... )
        'survival.csv'

        See Also
        --------
        SurvivalPrediction.survival_at_chunks
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

        Default return (when ``id_column`` and ``return_type`` are both
        omitted) depends on the fitted model class:

        * Gaussian / Binomial / Standard models: a table (dict, pandas
          DataFrame, pyarrow Table, ...) matching the training table kind
          with an ``eta`` and ``mean`` column (plus interval columns when
          ``interval`` is given).
        * Transformation-normal models: returns the per-row transformed
          z-score as a numpy array of shape ``(n_samples,)``.
        * Bernoulli marginal-slope: returns a calibrated probability array in
          ``(0, 1)`` of shape ``(n_samples,)``.
        * Survival models: returns a :class:`SurvivalPrediction` whose
          ``.hazard_at``, ``.survival_at``, and ``.cumulative_hazard_at``
          helpers evaluate the fitted hazard surface on a user-supplied time
          grid.

        ``with_uncertainty`` (survival only): when ``True``, the returned
        :class:`SurvivalPrediction` also carries delta-method standard
        errors on the survival surface (``survival_se``) and the linear
        predictor (``eta_se``).  Currently honored for the location-scale
        survival likelihood mode.
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
        """Return the model summary (coefficients, family, deviance, REML score).

        Returns
        -------
        Summary
            A dict-like :class:`Summary` containing the fitted formula,
            family / link name, model class, deviance, REML or LAML score,
            iteration count, and the per-coefficient table (estimates,
            standard errors, credible-interval bounds).  The summary is
            cached on first call.

        Examples
        --------
        >>> model = gamfit.fit(train, "y ~ s(x)")
        >>> s = model.summary()
        >>> print(s["family_name"], s["deviance"])
        >>> s.coefficients_frame()      # pandas DataFrame, requires pandas
        """
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
        """Validate ``data`` against the model's training schema.

        Inexpensive: runs the schema validator only, no prediction.  Use
        this before :meth:`predict` to surface column-name or type issues
        as structured :class:`SchemaIssue` records rather than as a raised
        :class:`SchemaMismatchError`.

        Parameters
        ----------
        data:
            Any table-like input (pandas DataFrame, dict of columns, list
            of records, numpy array, etc.).

        Returns
        -------
        SchemaCheck
            ``check.ok`` is ``True`` when the data matches the training
            schema; otherwise ``check.issues`` enumerates the problems.
            ``check.raise_for_error()`` raises ``ValueError`` on failure.

        Examples
        --------
        >>> check = model.check(test_df)
        >>> if not check:
        ...     for issue in check.issues:
        ...         print(issue.kind, issue.column, issue.message)
        """
        headers, rows, _table_kind = normalize_table(data)
        try:
            payload = json.loads(rust_module().check_json(self._model_bytes, headers, rows))
        except Exception as exc:
            raise map_exception(exc) from exc
        return SchemaCheck.from_dict(payload)

    def report(self, path: str | Path | None = None) -> str:
        """Generate a standalone HTML report of the fitted model.

        The report contains the summary table, smooth-term visualisations,
        and convergence diagnostics.  It is self-contained (no external
        assets), so the file can be emailed or attached to a PR.

        Parameters
        ----------
        path:
            If given, write the HTML to this path and return the path.
            If ``None`` (default), return the HTML as a string.

        Returns
        -------
        str
            HTML string (when ``path is None``) or the written path.

        Examples
        --------
        >>> model.report("report.html")
        >>> html = model.report()             # for inline Jupyter display
        """
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
        """Draw from the model's posterior with NUTS.

        Returns a :class:`PosteriorSamples` object carrying the raw
        ``(n_draws, n_coeffs)`` numpy matrix, per-coefficient mean / std /
        credible intervals, and convergence diagnostics (``rhat``,
        ``ess``, ``converged``).

        Defaults are dimension-aware — leaving every keyword unset gives
        you a chain count, warmup length, and total sample budget tuned
        to the fitted coefficient size (see
        :func:`gam::hmc::NutsConfig::for_dimension` on the Rust side).
        That heuristic already covers most usage; the keywords are there
        for power users who want a longer run, a different acceptance
        target, or a fixed seed for reproducibility.

        Parameters
        ----------
        data:
            Table-like input matching the model's training schema; the
            same input formats accepted by :meth:`predict` are supported
            here.  For survival models, the entry/exit/event columns are
            consumed in addition to covariates.
        samples:
            Posterior draws per chain *after* warmup.  When omitted,
            chosen automatically from the coefficient count.
        warmup:
            Warmup iterations per chain (defaults to ``samples`` when both
            are left unset, otherwise to the adaptive default).
        chains:
            Number of independent chains.  Defaults adaptively to 2 or 4.
        target_accept:
            Target HMC acceptance rate in ``(0, 1)``.  Higher values give
            smaller leapfrog steps and slower-but-more-robust mixing.
        seed:
            RNG seed for deterministic chain initialisation.

        Notes
        -----
        Sampling currently supports standard GLM family models (Gaussian,
        Binomial logit/probit/cloglog, Poisson, Gamma — with or without a
        link-wiggle component) and survival likelihood modes other than
        the latent and location-scale variants.  Unsupported model
        classes raise :class:`gamfit.GamError` with a message mirroring
        the CLI's ``gam sample`` behaviour.
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

    def _design_matrix_payload(self, data: Any) -> dict[str, Any]:
        headers, rows, _ = normalize_table(data)
        try:
            raw = rust_module().design_matrix_table(self._model_bytes, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        return json.loads(raw)

    def difference_smooth(
        self,
        data: Any,
        *,
        group: str,
        view: str,
        pairs: Sequence[tuple[Any, Any]] | None = None,
        n: int = 100,
        level: float = 0.95,
        simultaneous: bool = False,
        n_sim: int = 10000,
        seed: int | None = 12345,
        group_means: bool = True,
        marginalise_random: bool = True,
    ) -> Any:
        """Estimate pairwise difference smooths with covariance-aware bands.

        The contrast is computed from the joint fitted-coefficient covariance as
        ``(X_b - X_a) V (X_b - X_a)'``.  Set ``simultaneous=True`` to replace
        the pointwise normal critical value with a posterior-simulation maximum
        statistic over the requested grid.
        """
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0, 1)")
        if n < 2:
            raise ValueError("n must be at least 2")
        if not group_means:
            raise NotImplementedError(
                "group_means=False requires term-level coefficient metadata that is not yet exposed; "
                "the default group_means=True returns the full trajectory contrast."
            )
        # The current design-matrix FFI represents the population-level fixed
        # predictor. Random-effect columns for supplied grouping levels are part
        # of X only when those random terms are present in the saved model; there
        # is not yet public column-role metadata to drop them selectively.
        _ = marginalise_random

        import math
        import statistics
        import numpy as np

        headers, rows, table_kind = normalize_table(data)
        if view not in headers:
            raise ValueError(f"view column {view!r} not present in data")
        if group not in headers:
            raise ValueError(f"group column {group!r} not present in data")
        view_idx = headers.index(view)
        group_idx = headers.index(group)
        view_values = np.asarray([float(row[view_idx]) for row in rows], dtype=float)
        grid = np.linspace(float(np.nanmin(view_values)), float(np.nanmax(view_values)), int(n))
        levels = []
        for row in rows:
            value = row[group_idx]
            if value not in levels:
                levels.append(value)
        if len(levels) < 2:
            raise ValueError("difference_smooth requires at least two group levels in data")
        if pairs is None:
            pairs = [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]

        template = list(rows[0])
        out_rows: list[dict[str, Any]] = []
        z = statistics.NormalDist().inv_cdf(0.5 + level / 2.0)
        for a, b in pairs:
            rows_a = []
            rows_b = []
            for x in grid:
                ra = list(template)
                rb = list(template)
                ra[view_idx] = str(float(x))
                rb[view_idx] = str(float(x))
                ra[group_idx] = str(a)
                rb[group_idx] = str(b)
                rows_a.append(ra)
                rows_b.append(rb)
            payload_a = self._design_matrix_payload({h: [r[j] for r in rows_a] for j, h in enumerate(headers)})
            payload_b = self._design_matrix_payload({h: [r[j] for r in rows_b] for j, h in enumerate(headers)})
            xa = np.asarray(payload_a["x_flat"], dtype=float).reshape(int(payload_a["n_rows"]), int(payload_a["n_cols"]))
            xb = np.asarray(payload_b["x_flat"], dtype=float).reshape(int(payload_b["n_rows"]), int(payload_b["n_cols"]))
            beta = np.asarray(payload_b.get("beta"), dtype=float)
            cov_n = int(payload_b.get("covariance_n") or 0)
            if beta.size == 0 or cov_n == 0:
                raise RuntimeError("fitted model did not expose coefficients/covariance for difference_smooth")
            cov = np.asarray(payload_b["covariance_flat"], dtype=float).reshape(cov_n, cov_n)
            xd = xb - xa
            diff = xd @ beta
            var = np.einsum("ij,jk,ik->i", xd, cov, xd)
            se = np.sqrt(np.maximum(var, 0.0))
            crit = z
            band_type = "pointwise"
            if simultaneous:
                rng = np.random.default_rng(seed)
                draws = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=int(n_sim), method="svd")
                centered = draws @ xd.T
                denom = np.where(se > 0, se, np.inf)
                maxima = np.max(np.abs(centered) / denom, axis=1)
                crit = float(np.quantile(maxima[np.isfinite(maxima)], level))
                band_type = "simultaneous"
            lower = diff - crit * se
            upper = diff + crit * se
            for x, d, s, lo, hi in zip(grid, diff, se, lower, upper, strict=True):
                out_rows.append({
                    view: float(x),
                    "level_1": a,
                    "level_2": b,
                    "diff": float(d),
                    "se": float(s),
                    "lower": float(lo),
                    "upper": float(hi),
                    "critical": float(crit),
                    "band": band_type,
                    "level": float(level),
                })
        try:
            import pandas as pd
            return pd.DataFrame(out_rows)
        except Exception:
            return out_rows

    def design_matrix(self, data: Any) -> Any:
        """Materialised design matrix for ``data`` against the saved model.

        Returns an ``(n_rows, n_coeffs)`` numpy array — exactly the
        matrix the engine uses internally for linear-predictor
        evaluation.  Useful for custom posterior reasoning (e.g.
        feeding draws into your own predictive routine) or for
        debugging term layouts.

        Currently restricted to standard non-link-wiggle GAM models;
        other classes raise a clear error pointing at
        :meth:`Model.predict` for the class-specific prediction path.
        """
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
        import math
        import statistics
        import numpy as np

        if not (0.0 < float(level) < 1.0):
            raise ValueError("difference_smooth level must be in (0, 1)")
        if int(n) < 2:
            raise ValueError("difference_smooth n must be at least 2")
        try:
            state = json.loads(rust_module().coefficient_state_json(self._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc
        schema_cols = list((state.get("schema") or {}).get("columns") or [])
        if not schema_cols:
            raise ValueError("difference_smooth requires a saved model schema")
        names = [str(c.get("name")) for c in schema_cols]
        if view not in names:
            raise ValueError(f"view column {view!r} not found in model schema: {names}")
        if group is None:
            cats = [c for c in schema_cols if c.get("kind") == "categorical" and c.get("name") != view]
            if not cats:
                raise ValueError("difference_smooth could not infer a categorical group column; pass group=")
            group = str(cats[0].get("name"))
        if group not in names:
            raise ValueError(f"group column {group!r} not found in model schema: {names}")
        group_col = next(c for c in schema_cols if c.get("name") == group)
        levels = [str(v) for v in group_col.get("levels") or []]
        if len(levels) < 2:
            raise ValueError(f"group column {group!r} must have at least two saved levels")
        if pairs is None:
            pairs = [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
        ranges = state.get("training_feature_ranges") or []
        view_idx = names.index(view)
        if view_idx < len(ranges):
            lo, hi = map(float, ranges[view_idx])
        else:
            lo, hi = 0.0, 1.0
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
            lo, hi = 0.0, 1.0
        grid = np.linspace(lo, hi, int(n))

        template: dict[str, Any] = {}
        if data is not None:
            headers, rows, _ = normalize_table(data)
            if rows:
                first = rows[0]
                template.update({h: first[i] for i, h in enumerate(headers)})
        for idx, col in enumerate(schema_cols):
            name = str(col.get("name"))
            if name in template:
                continue
            if col.get("kind") == "categorical":
                vals = col.get("levels") or ["0"]
                template[name] = str(vals[0])
            elif idx < len(ranges):
                a, b = map(float, ranges[idx])
                template[name] = str(0.5 * (a + b))
            else:
                template[name] = "0"

        beta = np.asarray(state.get("beta", []), dtype=float)
        cov_n = int(state.get("covariance_n", 0))
        cov_flat = np.asarray(state.get("covariance_flat", []), dtype=float)
        if cov_flat.size != cov_n * cov_n or beta.size != cov_n:
            raise ValueError("coefficient covariance payload has inconsistent dimensions")
        cov = cov_flat.reshape(cov_n, cov_n)
        z = statistics.NormalDist().inv_cdf(0.5 + float(level) / 2.0)
        rng = np.random.default_rng(seed)
        rows_out: list[dict[str, Any]] = []
        random_ranges = [(int(a), int(b)) for a, b in state.get("random_column_ranges", [])]

        for a, b in pairs:
            left_level = str(a)
            right_level = str(b)
            data_left = []
            data_right = []
            for x in grid:
                row_l = dict(template)
                row_r = dict(template)
                row_l[view] = str(float(x))
                row_r[view] = str(float(x))
                row_l[group] = left_level
                row_r[group] = right_level
                data_left.append(row_l)
                data_right.append(row_r)
            xl = self.design_matrix(data_left)
            xr = self.design_matrix(data_right)
            xd = xr - xl
            if marginalise_random:
                for start, stop in random_ranges:
                    xd[:, start:stop] = 0.0
            if not group_means:
                # Drop parametric main-effect columns from the contrast while
                # retaining smooth blocks. The global layout is intercept,
                # linear terms, random effects, smooths; categorical offsets in
                # this package are represented as random-effect blocks, so this
                # also removes those offsets under the population contrast.
                for start, stop in random_ranges:
                    xd[:, start:stop] = 0.0
            diff = xd @ beta
            var = np.einsum("ij,jk,ik->i", xd, cov, xd)
            se = np.sqrt(np.maximum(var, 0.0))
            crit = z
            if simultaneous:
                draws = rng.multivariate_normal(beta, cov, size=int(n_sim), check_valid="ignore")
                draw_diff = draws @ xd.T
                denom = np.where(se > 0.0, se, np.inf)
                max_dev = np.max(np.abs((draw_diff - diff.reshape(1, -1)) / denom.reshape(1, -1)), axis=1)
                crit = float(np.quantile(max_dev[np.isfinite(max_dev)], float(level)))
            lower = diff - crit * se
            upper = diff + crit * se
            for x, d, s_e, lo_b, hi_b in zip(grid, diff, se, lower, upper, strict=True):
                rows_out.append({
                    view: float(x),
                    "group": group,
                    "level_1": left_level,
                    "level_2": right_level,
                    "diff": float(d),
                    "se": float(s_e),
                    "lower": float(lo_b),
                    "upper": float(hi_b),
                    "level": float(level),
                    "simultaneous": bool(simultaneous),
                    "critical": float(crit),
                    "covariance_corrected": bool(state.get("covariance_corrected", False)),
                })
        if return_type == "list":
            return rows_out
        try:
            import pandas as pd
            return pd.DataFrame(rows_out)
        except Exception:
            return rows_out

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to ``path``.

        Writes a self-contained binary ``.gam`` file that
        :func:`gamfit.load` round-trips.

        Examples
        --------
        >>> model.save("model.gam")
        >>> loaded = gamfit.load("model.gam")
        """
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
        """Return the serialised model as raw bytes.

        Useful for in-memory transport.  :func:`gamfit.loads` is the
        inverse.

        Examples
        --------
        >>> blob = model.dumps()
        >>> loaded = gamfit.loads(blob)
        """
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
    def evidence(self) -> float:
        """REML / LAML log marginal-likelihood score for this fit.

        Alias for ``summary()["reml_score"]``. This is the same scalar the
        outer REML loop maximizes to select smoothing parameters; because the
        LAML approximation already folds in the ``log|H| − log|S|_+`` Occam
        factors, differences between two fits' ``evidence`` are log Bayes
        factors with model-complexity already penalised. Use
        :func:`gamfit.compare_models` to rank multiple fits, or
        :meth:`bayes_factor_vs` for a pairwise comparison.
        """
        value = self.summary().get("reml_score")
        if value is None:
            raise ValueError("saved model payload is missing reml_score")
        return float(value)

    def bayes_factor_vs(self, other: "Model") -> float:
        """Bayes factor of this fit against ``other``.

        Returns ``exp(self.evidence - other.evidence)``. A value greater than
        one means this fit is favoured; less than one means ``other`` is. For
        large differences the result may overflow — use ``self.evidence -
        other.evidence`` directly on the log scale in that case.

        Parameters
        ----------
        other : Model
            Competing fit, typically on the same response with a different
            basis topology or penalty structure.

        Examples
        --------
        >>> circle.bayes_factor_vs(torus)
        1234.5
        """
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
        """Build a default time grid spanning the data's survival window.

        Only triggered for survival-family models. Parses the response
        expression from the formula (``Surv(entry, exit, event)``) to
        locate the entry/exit columns, then returns a uniform grid of
        64 points between the observed min entry and max exit times.
        Returns ``None`` if the model is not a survival model or the
        formula cannot be parsed.
        """
        import re
        import numpy as np

        if self.model_class not in _SURVIVAL_TIME_GRID_MODEL_CLASSES:
            return None
        formula = self.formula
        match = re.match(
            r"\s*Surv\s*\(\s*([^\s,]+)\s*,\s*([^\s,]+)\s*,\s*[^\s,]+\s*\)",
            formula,
        )
        if not match:
            return None
        entry_name = match.group(1)
        exit_name = match.group(2)
        header_to_index = {name: i for i, name in enumerate(headers)}
        entry_idx = header_to_index.get(entry_name)
        exit_idx = header_to_index.get(exit_name)
        if entry_idx is None or exit_idx is None:
            missing = [
                name
                for name, idx in ((entry_name, entry_idx), (exit_name, exit_idx))
                if idx is None
            ]
            raise ValueError(
                "survival prediction data is missing required time column(s): "
                + ", ".join(missing)
            )
        entry_vals: list[float] = []
        exit_vals: list[float] = []
        for row_index, row in enumerate(rows):
            try:
                entry_vals.append(float(row[entry_idx]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"survival entry column {entry_name!r} has a non-numeric value "
                    f"at row {row_index + 1}: {row[entry_idx]!r}"
                ) from exc
            try:
                exit_vals.append(float(row[exit_idx]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"survival exit column {exit_name!r} has a non-numeric value "
                    f"at row {row_index + 1}: {row[exit_idx]!r}"
                ) from exc
        if not entry_vals or not exit_vals:
            return None
        lo = float(np.min(entry_vals))
        hi = float(np.max(exit_vals))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("survival time columns must contain only finite values")
        if hi <= lo:
            raise ValueError(
                f"survival exit times must extend beyond entry times; got min entry {lo} and max exit {hi}"
            )
        # Pad by a hair so interpolation queries at the exact max stay
        # inside the grid rather than clipping to the boundary.
        span = hi - lo
        hi_padded = hi + max(span * 1e-6, 1e-9)
        return list(np.linspace(lo, hi_padded, 64))

    def diagnose(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
    ) -> Diagnostics:
        """Score the fitted model on held-out ``data``.

        Calls :meth:`predict` on the feature columns of ``data`` and
        compares the result against the observed response, packaging
        the prediction, residuals, observed values, and (when
        requested) Wald bands into a :class:`Diagnostics` object.
        Useful for ad-hoc held-out checks and for feeding the
        :meth:`plot` method.

        Parameters
        ----------
        data : table-like
            Any table-like input accepted by :meth:`predict` that also
            carries the response column.
        y : str, optional
            Name of the response column. Defaults to
            :attr:`response_name`; required when that cannot be inferred
            (e.g. survival formulas).
        interval : float or None, optional
            Pointwise Wald-interval probability passed through to
            :meth:`predict`. Set to ``None`` to skip interval columns.
            Defaults to ``0.95``.

        Returns
        -------
        Diagnostics
            A :class:`Diagnostics` record containing the formula,
            response name, observed values, the predicted table, and
            residuals.

        Raises
        ------
        ValueError
            If the response column cannot be inferred or is missing from
            ``data``.

        Examples
        --------
        >>> diag = model.diagnose(test_df)
        >>> diag.rmse, diag.r_squared
        (0.42, 0.81)
        >>> diag.predicted["mean"][:3]
        [1.04, 1.21, 0.99]

        See Also
        --------
        Model.predict
        Model.plot
        """
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
        """Plot the model's behaviour on ``data`` with matplotlib.

        Runs :meth:`diagnose` against ``data`` and then renders one of
        three standard diagnostic plots onto a matplotlib ``Axes``.

        Parameters
        ----------
        data : table-like
            Held-out data with the response column present (same
            requirements as :meth:`diagnose`).
        x : str, optional
            Feature column to plot on the x-axis when
            ``kind="prediction"``. Inferred automatically when there is
            exactly one non-response feature column.
        y : str, optional
            Response column name. Defaults to :attr:`response_name`.
        interval : float or None, optional
            Pointwise Wald-interval probability for the shaded band on
            prediction plots. Ignored for ``residuals`` and
            ``observed_vs_predicted`` plots. Defaults to ``0.95``.
        kind : {"prediction", "residuals", "observed_vs_predicted"}, optional
            * ``"prediction"`` (default) — mean curve over ``x`` with a
              pointwise Wald band and observed scatter overlay.
            * ``"residuals"`` — residuals vs predicted mean.
            * ``"observed_vs_predicted"`` — observed vs predicted with
              a reference ``y = x`` line.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw onto. When omitted, a fresh
            ``Axes`` is created via ``plt.subplots()``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that were drawn on.

        Raises
        ------
        ValueError
            If ``kind`` is not one of the supported choices, or if
            ``x`` cannot be inferred for a multi-feature prediction
            plot, or if the named ``x`` column is missing from
            ``data``.

        Examples
        --------
        >>> model.plot(test_df)                       # prediction with band
        >>> model.plot(test_df, kind="residuals")
        >>> ax = model.plot(test_df, kind="observed_vs_predicted")
        >>> ax.set_title("Calibration on held-out fold")

        See Also
        --------
        Model.diagnose
        Model.predict
        """
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


def _coverage_provenance_groups(
    state: dict[str, Any],
    n_coeffs: int,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, list[int]]]:
    provenance = state.get("coefficient_provenance")
    if not isinstance(provenance, list) or len(provenance) != n_coeffs:
        provenance = [{"index": idx, "label": "__global__", "source": "global"} for idx in range(n_coeffs)]

    group_metadata = state.get("group_metadata")
    if not isinstance(group_metadata, dict):
        group_metadata = {}

    labels: list[str] = []
    label_indices: dict[str, list[int]] = {}
    raw_info: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = {}
    terms: dict[str, set[str]] = {}
    columns: dict[str, set[str]] = {}
    levels: dict[str, set[str]] = {}

    for fallback_index, item in enumerate(provenance):
        if not isinstance(item, dict):
            item = {}
        index = int(item.get("index", fallback_index))
        if index < 0 or index >= n_coeffs:
            continue
        label = str(item.get("label") or "__global__")
        if label not in label_indices:
            labels.append(label)
            label_indices[label] = []
            raw_info[label] = {"coefficient_indices": []}
            sources[label] = set()
            terms[label] = set()
            columns[label] = set()
            levels[label] = set()
        label_indices[label].append(index)
        raw_info[label]["coefficient_indices"].append(index)
        if item.get("metadata") is not None:
            raw_info[label]["metadata"] = item["metadata"]
        elif label in group_metadata:
            raw_info[label]["metadata"] = group_metadata[label]
        for key, target in (
            ("source", sources[label]),
            ("term", terms[label]),
            ("column", columns[label]),
            ("level", levels[label]),
        ):
            value = item.get(key)
            if value is not None:
                target.add(str(value))

    label_info: dict[str, dict[str, Any]] = {}
    for label in labels:
        info = dict(raw_info[label])
        if sources[label]:
            info["sources"] = sorted(sources[label])
        if terms[label]:
            info["terms"] = sorted(terms[label])
        if columns[label]:
            info["columns"] = sorted(columns[label])
        if levels[label]:
            info["levels"] = sorted(levels[label])
        label_info[label] = info

    return labels, label_info, label_indices


__all__ = [
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "SurvivalPrediction",
    "competing_risks_cif",
]
