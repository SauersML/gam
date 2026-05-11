from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
        "survival marginal-slope",
        "survival location-scale",
        "latent survival",
    }
)
_SURVIVAL_TIME_GRID_MODEL_CLASSES = frozenset(
    {
        "survival",
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
        """Hazard rate ``h(t)`` evaluated at each time in ``times``.

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
        """Survival probability ``S(t)`` at each requested time.

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

    def failure_at(self, times: Any) -> Any:
        """Failure/event probability ``1 - S(t)`` at each requested time."""
        import numpy as np

        survival = np.asarray(self.survival_at(times), dtype=float)
        return np.clip(1.0 - survival, 0.0, 1.0)

    def survival_se_at(self, times: Any) -> Any:
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
        """Yield ``(row_slice, time_slice, cumulative_hazard_block)`` chunks."""
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
        """Write chunked survival predictions as row,time,survival CSV."""
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
        * Transformation-normal models: a per-row transformed z-score as a
          1-D numpy array of shape ``(n_samples,)``.
        * Bernoulli marginal-slope: a calibrated probability vector in
          ``(0, 1)`` as a 1-D numpy array of shape ``(n_samples,)``.
        * Survival models: a :class:`SurvivalPrediction` whose
          ``.hazard_at``, ``.survival_at``, ``.failure_at``, and
          ``.cumulative_hazard_at`` helpers evaluate the fitted hazard
          surface on a user-supplied time grid.

        Passing ``id_column`` or ``return_type`` switches the
        array-returning model classes (transformation-normal and
        Bernoulli marginal-slope) to the **table form**: a 2-column table
        ``(id_column, "z" or "mean")`` rather than a bare 1-D array.
        Naively flattening that table with ``np.asarray(...)`` /
        ``.to_numpy()`` yields shape ``(n_samples, 2)``, which is a
        common cause of silent broadcasting bugs in downstream metric
        code that expects a 1-D probability vector.  When you need the
        probabilities as an array after asking for an id column, extract
        the column explicitly, e.g. ``out["mean"]`` /
        ``np.asarray(out["mean"], dtype=float)``.

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

        columns = _ordered_prediction_columns(parsed["columns"])
        model_class = str(parsed.get("model_class") or self._model_class_from_summary())

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
        return PosteriorSamples.from_ffi_json(raw)

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

    def _default_survival_time_grid(
        self, headers: list[str], rows: list[list[str]]
    ) -> list[float] | None:
        """Build a default time grid spanning the data's survival window.

        Only triggered for survival-family models. Parses the response
        expression from the formula (``Surv(entry, exit, event)``) to
        locate the entry/exit columns, then returns a uniform grid of
        64 points between the observed min entry and max exit times.
        Returns ``None`` if the model is not a survival model, the
        formula cannot be parsed, or the required columns aren't
        present in ``headers``.
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
            return None
        entry_vals: list[float] = []
        exit_vals: list[float] = []
        for row in rows:
            try:
                entry_vals.append(float(row[entry_idx]))
            except (TypeError, ValueError):
                continue
            try:
                exit_vals.append(float(row[exit_idx]))
            except (TypeError, ValueError):
                continue
        if not entry_vals or not exit_vals:
            return None
        lo = float(np.min(entry_vals))
        hi = float(np.max(exit_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
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


def _coerce_matrix(value: Any) -> Any:
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


__all__ = ["Model", "SurvivalPrediction"]
