"""Public :class:`Model` shell.

The numeric work all lives in the Rust core: this module marshals
arguments through the FFI, hands payloads off to ``_survival`` /
``_diagnose_plot`` helpers, and exposes Pythonic properties.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from ._binding import rust_module
from ._diagnostics import Diagnostics
from ._exceptions import map_exception
from ._sampling import PosteriorSamples
from ._schema import SchemaCheck
from ._summary import Summary
from ._survival import (
    CompetingRisksCIF,
    CompetingRisksPrediction,
    SurvivalPrediction,
    TermBlock,
    _MARGINAL_SLOPE_MODEL_CLASSES,
    _SURVIVAL_MODEL_CLASSES,
    _TRANSFORMATION_NORMAL_MODEL_CLASSES,
    competing_risks_cif,
    extract_row_ids,
    shape_prediction_response,
    term_blocks_for_model,
)
<<<<<<< Updated upstream
from ._tables import normalize_table, response_column_name, restore_output_table
=======

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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
>>>>>>> Stashed changes


class Model:
    __slots__ = ("_model_bytes", "_training_table_kind")

    def __init__(self, *, _model_bytes: bytes, _training_table_kind: str | None = None) -> None:
        self._model_bytes = _model_bytes
        self._training_table_kind = _training_table_kind

    def predict(
        self,
        data: Any,
        *,
        interval: float | None = None,
        return_type: str | None = None,
        id_column: str | None = None,
        with_uncertainty: bool = False,
    ) -> Any:
        """Predict from ``data``."""
        headers, rows, table_kind = normalize_table(data)
        row_ids = extract_row_ids(headers, rows, id_column)
        opts_json = rust_module().build_model_predict_payload_json(
            self._model_bytes, headers, rows, interval, with_uncertainty
        )
        try:
            raw = rust_module().predict_table(
                self._model_bytes, headers, rows, opts_json
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return shape_prediction_response(
            raw,
            headers=headers,
            rows=rows,
            table_kind=table_kind,
            training_table_kind=self._training_table_kind,
            fallback_model_class=self._model_class_from_payload(),
            interval=interval,
            return_type=return_type,
            id_column=id_column,
            row_ids=row_ids,
            restore=restore_output_table,
        )

    def predict_array(self, X: Any, *, interval: float | None = None) -> Any:
        """Predict directly from a numeric NumPy-compatible feature matrix."""
        try:
            rust = rust_module()
            return rust.predict_array(
                self._model_bytes,
                rust.numeric_matrix_f64(X, "X"),
                json.dumps({"interval": interval}),
            )
        except Exception as exc:
            raise map_exception(exc) from exc

    def summary(self) -> Summary:
        """Return the model summary (coefficients, family, deviance, REML score)."""
        try:
            payload = rust_module().summary_payload_from_model(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        return Summary.from_dict(payload)

    def smoothing_parameters(self) -> dict[int, float]:
        """Return fitted smoothing/precision parameters by penalty index."""
        return dict(rust_module().smoothing_parameters_from_model(self._model_bytes))

    def check(self, data: Any) -> SchemaCheck:
        """Validate ``data`` against the model's training schema."""
        headers, rows, _ = normalize_table(data)
        try:
            payload = rust_module().check_payload_from_model(self._model_bytes, headers, rows)
        except Exception as exc:
            raise map_exception(exc) from exc
        return SchemaCheck.from_dict(payload)

    def report(self, path: str | Path | None = None) -> str:
        """Generate a standalone HTML report of the fitted model."""
        try:
            html = rust_module().report_html(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc
        # allow-list (a): FFI response marshaling for optional file output.
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
        """Draw from the model's posterior with NUTS."""
        headers, rows, _ = normalize_table(data)
        try:
            ffi = rust_module()
            options_json = ffi.build_sample_payload_json(
                samples, warmup, chains, target_accept, seed
            )
            raw = ffi.sample_table(
                self._model_bytes,
                headers,
                rows,
                options_json,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return PosteriorSamples.from_ffi_json(raw, model_bytes=self._model_bytes)

    def design_matrix(self, data: Any) -> Any:
        """Materialised design matrix for ``data`` against the saved model."""
        headers, rows, _ = normalize_table(data)
        return rust_module().design_matrix_table_dense(self._model_bytes, headers, rows)

    def design_matrix_array(self, X: Any) -> Any:
        """Materialised design matrix for a numeric feature matrix."""
        try:
            rust = rust_module()
            return rust.design_matrix_array(
                self._model_bytes,
                rust.numeric_matrix_f64(X, "X"),
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
        """Covariance-aware pairwise difference smooths (Rust-backed)."""
        template: dict[str, str] = {}
        # allow-list (a): FFI input marshaling for an optional template row.
        if data is not None:
            headers, rows, _ = normalize_table(data)
            # allow-list (a): FFI input marshaling for empty prediction tables.
            if rows:
                first = rows[0]
                # allow-list (a): FFI payload marshaling.
                template = dict(zip(headers, map(str, first), strict=True))
        try:
            # allow-list (a): FFI optional argument marshaling.
            group_arg = str(group) if group is not None else None
            # allow-list (a): FFI payload sequence marshaling.
            pairs_arg = (
                list(map(lambda pair: (str(pair[0]), str(pair[1])), pairs))
                if pairs is not None
                else None
            )
            # allow-list (a): FFI optional argument marshaling.
            seed_arg = int(seed) if seed is not None else None
            # allow-list (a): FFI optional argument marshaling.
            template_arg = None if not template else template
            request_json = rust_module().build_difference_smooth_request_json(
                str(view),
                group_arg,
                pairs_arg,
                int(n),
                float(level),
                bool(simultaneous),
                int(n_sim),
                seed_arg,
                bool(marginalise_random),
                bool(group_means),
                template_arg,
            )
            rows_out = rust_module().difference_smooth_rows(
                self._model_bytes, request_json
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        # allow-list (a): FFI response marshaling for requested output type.
        if return_type == "list":
            return rows_out
        try:
            import pandas as pd

            return pd.DataFrame(rows_out)
        except Exception:
            return rows_out

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to ``path``."""
        Path(path).write_bytes(self._model_bytes)

    def extend_with_group(
        self,
        new_group_spec: dict[str, Any],
        metadata: Any | None = None,
        prior: Any | None = None,
    ) -> "Model":
        """Return a no-refit model extended with deployment-time group levels."""
        # allow-list (a): FFI input validation.
        if not isinstance(new_group_spec, dict):
            raise TypeError("new_group_spec must be a dict")
        try:
            rust = rust_module()
            # allow-list (a): FFI optional argument marshaling.
            metadata_json = json.dumps(metadata) if metadata is not None else None
            # allow-list (a): FFI optional argument marshaling.
            prior_json = json.dumps(prior) if prior is not None else None
            payload_json = rust.build_extend_group_payload_json(
                json.dumps(new_group_spec),
                metadata_json,
                prior_json,
            )
            model_bytes = bytes(
                rust.extend_model_with_group(self._model_bytes, payload_json)
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return Model(
            _model_bytes=model_bytes,
            _training_table_kind=self._training_table_kind,
        )

    def dumps(self) -> bytes:
        """Return the serialised model as raw bytes."""
        return self._model_bytes

    @property
    def formula(self) -> str:
        return rust_module().required_saved_model_payload_string(
            self._model_bytes, "formula"
        )

    @property
    def family_name(self) -> str:
        return str(self.summary()["family_name"])

    @property
    def model_class(self) -> str:
        return self._model_class_from_payload()

    @property
    def is_survival(self) -> bool:
        return self.model_class in _SURVIVAL_MODEL_CLASSES

    @property
    def is_marginal_slope(self) -> bool:
        return self.model_class in _MARGINAL_SLOPE_MODEL_CLASSES

    @property
    def is_transformation_normal(self) -> bool:
        return self.model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES

    @property
    def response_name(self) -> str | None:
        return response_column_name(self.formula)

    @property
    def training_table_kind(self) -> str | None:
        return self._training_table_kind

    @property
    def group_metadata(self) -> dict[str, Any] | None:
        return rust_module().model_group_metadata(self._model_bytes)

    @property
    def deployment_extensions(self) -> tuple[dict[str, Any], ...]:
        return tuple(rust_module().model_deployment_extensions(self._model_bytes))

    @property
    def term_blocks(self) -> tuple[TermBlock, ...]:
        """Per-term coefficient column ranges in fitted coefficient order."""
        try:
            return term_blocks_for_model(self._model_bytes)
        except Exception as exc:
            raise map_exception(exc) from exc

    @property
    def evidence(self) -> float:
        """REML / LAML log marginal-likelihood score for this fit."""
        return float(rust_module().model_evidence(self._model_bytes))

    def bayes_factor_vs(self, other: "Model") -> float:
        """Bayes factor of this fit against ``other``."""
        # allow-list (a): FFI input validation.
        if not isinstance(other, Model):
            raise TypeError(
                f"bayes_factor_vs expects a gamfit.Model, got {type(other).__name__}"
            )
        log_diff = rust_module().bayes_factor_log_diff(
            self._model_bytes, other._model_bytes
        )
        return math.exp(log_diff)

    def _model_class_from_payload(self) -> str:
        return rust_module().required_saved_model_payload_string(
            self._model_bytes, "model_kind"
        )

    def diagnose(
        self,
        data: Any,
        *,
        y: str | None = None,
        interval: float | None = 0.95,
    ) -> Diagnostics:
        """Score the fitted model on held-out ``data``."""
        from ._diagnose_plot import diagnose as _diagnose

        return _diagnose(self, data, y=y, interval=interval)

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
        """Plot the model's behaviour on ``data`` with matplotlib."""
        from ._diagnose_plot import plot as _plot

        return _plot(self, data, x=x, y=y, interval=interval, kind=kind, ax=ax)

    def __repr__(self) -> str:
        parts = [
            f"formula={self.formula!r}",
            f"family_name={self.family_name!r}",
            f"training_table_kind={self._training_table_kind!r}",
        ]
        return f"Model({', '.join(parts)})"

    def _repr_html_(self) -> str:
        return self.report()


__all__ = [
    "CompetingRisksCIF",
    "CompetingRisksPrediction",
    "Model",
    "SurvivalPrediction",
    "TermBlock",
    "competing_risks_cif",
]
