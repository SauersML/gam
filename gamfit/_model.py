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
    cumulative_hazard : ndarray or None
        ``(n_samples, len(times))`` dense cumulative-hazard surface from
        the FFI.
    linear_predictor : ndarray or None
        ``(n_samples,)`` per-row linear predictor at each row's own exit
        time.
    id_column : str or None
        Optional name of the id column carried through from
        :meth:`Model.predict` for use by :meth:`write_survival_at_csv`.
    row_ids : sequence of str or None
        Per-row identifiers aligned with ``parameters`` rows, populated
        when ``id_column`` was supplied to :meth:`Model.predict`.
    survival_se : ndarray or None
        ``(n_samples, len(times))`` delta-method standard errors on the
        survival surface (response scale). ``None`` unless the
        prediction was issued with ``with_uncertainty=True``; then
        populated for location-scale survival models.
    eta_se : ndarray or None
        ``(n_samples,)`` delta-method SE on the linear predictor at each
        row's own exit time, under the same conditions as
        ``survival_se``.

    Examples
    --------
    >>> import numpy as np
    >>> pred = model.predict(test_df)        # survival model
    >>> times = np.linspace(0.0, 10.0, 50)
    >>> S = pred.survival_at(times)          # (n_rows, 50) ndarray
    >>> h = pred.hazard_at(times)
    >>> H = pred.cumulative_hazard_at(times)

    See Also
    --------
    Model.predict : Returns a :class:`SurvivalPrediction` for survival models.
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

        When the FFI produced a dense hazard surface this linearly
        interpolates against the returned grid; otherwise the hazard is
        reconstructed from the cumulative-hazard differences. Large
        requests are evaluated in chunks internally before assembling
        the dense result.

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times at which to
            evaluate the per-row hazard.

        Returns
        -------
        ndarray
            ``(n_samples, len(times))`` array of non-negative hazard
            values, one row per prediction sample.

        Examples
        --------
        >>> import numpy as np
        >>> pred = model.predict(test_df)
        >>> h = pred.hazard_at(np.linspace(0.0, 5.0, 11))
        >>> h.shape
        (len(test_df), 11)

        See Also
        --------
        SurvivalPrediction.hazard_at_chunks : streaming chunked variant.
        SurvivalPrediction.cumulative_hazard_at
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
        """Evaluate the cumulative hazard ``H(t) = -log S(t)``.

        When the FFI provided a dense cumulative-hazard surface this
        interpolates against it directly; otherwise ``H(t)`` is derived
        from :meth:`survival_at` via ``-log S(t)`` (clipped away from
        zero for numerical safety).

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times.

        Returns
        -------
        ndarray
            ``(n_samples, len(times))`` array of non-negative cumulative
            hazard values.

        Examples
        --------
        >>> import numpy as np
        >>> H = pred.cumulative_hazard_at(np.array([1.0, 2.0, 5.0]))
        >>> np.all(np.diff(H, axis=1) >= 0)   # monotone non-decreasing
        True

        See Also
        --------
        SurvivalPrediction.survival_at
        SurvivalPrediction.hazard_at
        """
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

        When the FFI produced a dense hazard/survival surface this
        linearly interpolates against the returned grid. Otherwise it
        falls back to the plug-in identity ``S(t) = exp(-H(t))`` using
        a per-row piecewise-constant hazard derived from
        ``parameters`` (supports bare-dataclass construction). Large
        requests are evaluated in chunks internally before assembling
        the dense result.

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times.

        Returns
        -------
        ndarray
            ``(n_samples, len(times))`` array of survival probabilities
            in ``[0, 1]``.

        Examples
        --------
        >>> import numpy as np
        >>> times = np.linspace(0.0, 5.0, 6)
        >>> S = pred.survival_at(times)
        >>> S[:, 0]                  # S(0) is 1 for every row
        array([1., 1., ..., 1.])

        See Also
        --------
        SurvivalPrediction.failure_at : returns ``1 - S(t)``.
        SurvivalPrediction.survival_se_at : delta-method standard error.
        SurvivalPrediction.survival_at_chunks : streaming chunked variant.
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
        """Evaluate the failure (event) probability ``F(t) = 1 - S(t)``.

        Convenience wrapper around :meth:`survival_at`; the output is
        clipped to ``[0, 1]`` to guard against tiny interpolation
        excursions.

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times.

        Returns
        -------
        ndarray
            ``(n_samples, len(times))`` array of failure probabilities
            in ``[0, 1]``.

        Examples
        --------
        >>> F = pred.failure_at([1.0, 5.0, 10.0])
        >>> F.shape[1]
        3

        See Also
        --------
        SurvivalPrediction.survival_at
        """
        import numpy as np

        survival = np.asarray(self.survival_at(times), dtype=float)
        return np.clip(1.0 - survival, 0.0, 1.0)

    def survival_se_at(self, times: Any) -> Any:
        """Delta-method standard error on ``S(t)`` at each requested time.

        Returns ``None`` when the prediction was not issued with
        ``with_uncertainty=True`` (or the model class does not yet
        support response-scale uncertainty). When available, the
        returned array has shape ``(n_samples, len(times))`` and is
        clipped to be non-negative.

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times.

        Returns
        -------
        ndarray or None
            ``(n_samples, len(times))`` array of standard errors on the
            survival surface, or ``None`` if no uncertainty was
            requested.

        Notes
        -----
        Pair with :meth:`survival_at` for response-scale Wald-style
        bands: ``S +/- z * SE`` with the standard caveats around the
        Gaussian approximation near the ``[0, 1]`` boundaries.

        Examples
        --------
        >>> pred = model.predict(test_df, with_uncertainty=True)
        >>> S = pred.survival_at([1.0, 2.0])
        >>> SE = pred.survival_se_at([1.0, 2.0])
        >>> lower = (S - 1.96 * SE).clip(0.0, 1.0)

        See Also
        --------
        SurvivalPrediction.survival_at
        Model.predict : pass ``with_uncertainty=True`` to populate this.
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
        """Yield ``S(t)`` evaluations in row/time blocks.

        Streaming counterpart to :meth:`survival_at` for queries large
        enough that the dense ``(n_samples, len(times))`` allocation is
        unwelcome. Each yielded block can be consumed (written to disk,
        reduced, fed into a metric) and discarded before the next one
        is produced.

        Parameters
        ----------
        times : array_like
            1-D sequence of finite, non-negative times.
        people_chunk : int, optional
            Maximum number of rows per yielded block. Defaults to
            ``DEFAULT_SURVIVAL_PEOPLE_CHUNK`` (50 000).
        time_grid_chunk : int, optional
            Maximum number of time points per yielded block. Defaults
            to ``DEFAULT_SURVIVAL_TIME_GRID_CHUNK`` (64).

        Yields
        ------
        tuple of (slice, slice, ndarray)
            ``(row_slice, time_slice, block)`` where ``block`` has
            shape ``(row_slice.stop - row_slice.start,
            time_slice.stop - time_slice.start)`` and the slices index
            into the full ``(n_samples, len(times))`` result.

        Examples
        --------
        >>> import numpy as np
        >>> times = np.linspace(0.0, 10.0, 200)
        >>> total = 0.0
        >>> for _r, _t, block in pred.survival_at_chunks(times):
        ...     total += float(block.sum())

        See Also
        --------
        SurvivalPrediction.survival_at
        SurvivalPrediction.write_survival_at_csv
        """
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
        """Yield ``h(t)`` evaluations in row/time blocks.

        Streaming counterpart to :meth:`hazard_at`. When the FFI
        provided a dense hazard surface this iterates that surface
        directly; otherwise the hazard is derived from successive
        cumulative-hazard blocks, carrying the previous block's tail
        forward so the finite-difference at each block boundary stays
        consistent with the non-chunked :meth:`hazard_at` result.

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
            ``(row_slice, time_slice, block)`` of non-negative hazard
            values with shape matching the slice extents.

        Examples
        --------
        >>> peak = 0.0
        >>> for _r, _t, h_block in pred.hazard_at_chunks(times):
        ...     peak = max(peak, float(h_block.max()))

        See Also
        --------
        SurvivalPrediction.hazard_at
        SurvivalPrediction.cumulative_hazard_at_chunks
        """
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

    def predict_with_coverage(
        self,
        rows: Any,
        *,
        coverage: float = 0.95,
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Predict with covariance-based confidence intervals and group attribution.

        Returns ``(point, lower, upper, per_group_variance_contributions)``.
        The first three entries are numpy arrays on the response-mean scale.
        The fourth entry is a covariance-block variance decomposition:
        per-group arrays contain ``x_g' Cov(beta_g, beta_g) x_g`` and
        cross-term arrays contain ``2 x_g' Cov(beta_g, beta_h) x_h``.
        """
        import numpy as np

        coverage = float(coverage)
        if not (0.0 < coverage < 1.0):
            raise ValueError("coverage must be in (0, 1)")
        if self.is_survival:
            raise ValueError("predict_with_coverage currently supports non-survival models")

        design = np.asarray(self.design_matrix(rows), dtype=float)
        try:
            state = json.loads(rust_module().coefficient_state_json(self._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc

        beta = np.asarray(state.get("beta", []), dtype=float)
        cov_n = int(state.get("covariance_n", 0))
        cov_flat = np.asarray(state.get("covariance_flat", []), dtype=float)
        if cov_flat.size != cov_n * cov_n or beta.size != cov_n:
            raise ValueError("coefficient covariance payload has inconsistent dimensions")
        if design.shape[1] != cov_n:
            raise ValueError(
                "design matrix and coefficient covariance dimensions differ: "
                f"design has {design.shape[1]} columns, covariance is {cov_n}x{cov_n}"
            )

        predicted = self.predict(rows, interval=coverage, return_type="dict")
        columns = dict(predicted)
        point = np.asarray(columns.get("mean", columns.get("eta", [])), dtype=float)
        lower = np.asarray(columns.get("mean_lower", []), dtype=float)
        upper = np.asarray(columns.get("mean_upper", []), dtype=float)
        if point.size != design.shape[0] or lower.size != point.size or upper.size != point.size:
            raise ValueError("prediction interval payload has inconsistent row count")

        raw_cov = cov_flat.reshape(cov_n, cov_n)
        covariance_symmetry_max_abs = float(np.max(np.abs(raw_cov - raw_cov.T))) if cov_n else 0.0
        cov = 0.5 * (raw_cov + raw_cov.T)
        eta_variance = np.einsum("ij,jk,ik->i", design, cov, design)
        labels, label_info, label_indices = _coverage_provenance_groups(state, cov_n)

        # Variance decomposition.
        #
        # Partition beta and the row design vector x by provenance label:
        #   beta = (beta_1, ..., beta_K),  x = (x_1, ..., x_K)
        # and conformably block V = Cov(beta) into V_g,h.  Then
        #
        #   Var[x beta] = x V x'
        #               = sum_g x_g V_g,g x_g'
        #                 + 2 sum_{g<h} x_g V_g,h x_h'.
        #
        # A covariance matrix is symmetric; if the serialized matrix carries
        # tiny numerical skew, the quadratic form depends only on
        # (V + V') / 2 because x (V - V') x' = 0 for any skew-symmetric part.
        #
        # The per-group contribution reported below is the recognized diagonal
        # block term sigma_g^2(x) = x_g V_g,g x_g'.  Cross-block covariance is
        # exposed separately, so group terms plus cross terms reconstruct the
        # total variance used by the interval calculation.
        label_arrays = {
            label: np.asarray(indices, dtype=int)
            for label, indices in label_indices.items()
        }
        flat_indices = sorted(int(idx) for indices in label_arrays.values() for idx in indices)
        if flat_indices != list(range(cov_n)):
            raise ValueError(
                "coefficient provenance must partition all covariance columns for "
                "coverage decomposition"
            )
        group_variance = {
            label: np.einsum(
                "ij,jk,ik->i",
                design[:, indices],
                cov[np.ix_(indices, indices)],
                design[:, indices],
            )
            for label, indices in label_arrays.items()
            if indices.size
        }

        widths = np.maximum(upper - lower, 0.0)
        group_payload: dict[str, dict[str, Any]] = {}
        for label in labels:
            variance = group_variance.get(label)
            if variance is None:
                continue
            entry: dict[str, Any] = {
                "variance": variance,
                "standard_error": np.sqrt(np.maximum(variance, 0.0)),
                "component": "within_group_covariance_block",
            }
            info = label_info.get(label)
            if info:
                entry.update(info)
            group_payload[label] = entry

        pairwise_cross_terms: list[dict[str, Any]] = []
        cross_variance = np.zeros(design.shape[0], dtype=float)
        for left_pos, left in enumerate(labels):
            left_indices = label_arrays.get(left)
            if left_indices is None or not left_indices.size:
                continue
            x_left = design[:, left_indices]
            for right in labels[left_pos + 1:]:
                right_indices = label_arrays.get(right)
                if right_indices is None or not right_indices.size:
                    continue
                x_right = design[:, right_indices]
                pair_variance = 2.0 * np.einsum(
                    "ij,jk,ik->i",
                    x_left,
                    cov[np.ix_(left_indices, right_indices)],
                    x_right,
                )
                cross_variance = cross_variance + pair_variance
                pairwise_cross_terms.append(
                    {
                        "left": left,
                        "right": right,
                        "variance": pair_variance,
                    }
                )

        group_variance_sum = sum(group_variance.values(), np.zeros(design.shape[0], dtype=float))
        reconstructed_variance = group_variance_sum + cross_variance
        per_group_variance_contributions: dict[str, Any] = {
            "groups": group_payload,
            "cross_terms": {
                "variance": cross_variance,
                "pairwise": pairwise_cross_terms,
                "component": "between_group_covariance_blocks",
            },
            "total": {
                "variance": eta_variance,
                "standard_error": np.sqrt(np.maximum(eta_variance, 0.0)),
                "ci_width": widths,
                "group_variance_sum": group_variance_sum,
                "cross_variance": cross_variance,
                "reconstructed_variance": reconstructed_variance,
            },
            "metadata": {
                "coverage": coverage,
                "covariance_corrected": bool(state.get("covariance_corrected", False)),
                "covariance_symmetry_max_abs": covariance_symmetry_max_abs,
            },
        }

        return point, lower, upper, per_group_variance_contributions

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

    @property
    def formula(self) -> str:
        """The fitted Wilkinson-style formula string."""
        return str(self.summary()["formula"])

    @property
    def family_name(self) -> str:
        """Human-readable family + link name (e.g. ``"Gaussian Identity"``)."""
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


__all__ = ["Model", "SurvivalPrediction"]
