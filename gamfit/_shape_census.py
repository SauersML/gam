"""Whole-pipeline matched controls for representational-shape censuses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._rust import (
    label_shuffle_permutation,
    randomization_p_value,
    shape_matched_control,
    shape_matched_control_f32,
)

_ResultT = TypeVar("_ResultT")
_U64_MAX = (1 << 64) - 1
_FINITE_SCAN_SCALARS = 1 << 20
_ControlMatrix = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ShapeControlledCensus(Generic[_ResultT]):
    """Results from one observed and two matched full-pipeline runs.

    Reporting/fold-selected mixture orders, atom groups, PCA charts, and all
    other fitted objects belong inside ``_ResultT``. This wrapper deliberately
    knows nothing about those stages; its job is to guarantee that the exact
    same callback and pipeline seed see each of the three input matrices.
    """

    observed: _ResultT
    per_dimension_shuffle: _ResultT
    covariance_exact_hadamard: _ResultT
    pipeline_seed: int
    control_seed: int


@dataclass(frozen=True, slots=True)
class LabelShuffleMarginNull(Generic[_ResultT]):
    """Observed adjudication and its chart-rebuilt label-shuffle margin null."""

    observed: _ResultT
    observed_margin: float
    null_margins: NDArray[np.float64]
    exceedance_count: int
    p_value: float
    n_draws: int
    shuffle_seed: int
    pipeline_seed: int


def _u64(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer in [0, 2**64 - 1]")
    result = int(value)
    if not 0 <= result <= _U64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64 - 1]; got {result}")
    return result


def _require_finite(matrix: _ControlMatrix) -> None:
    """Validate finiteness with bounded temporary storage."""
    flattened = matrix.reshape(-1)
    for start in range(0, flattened.size, _FINITE_SCAN_SCALARS):
        if not np.isfinite(flattened[start : start + _FINITE_SCAN_SCALARS]).all():
            raise ValueError("data must contain only finite values")


def _native_shape_control(
    source: _ControlMatrix, kind: str, *, seed: int
) -> _ControlMatrix:
    """Run the native control matching ``source``'s (float32/float64) dtype.

    ``source`` is already normalized to exactly one of the two native dtypes,
    so ``astype(..., copy=False)`` returns ``source`` itself; it only restates
    the dtype the branch has established for the typed Rust entry point.
    """
    if source.dtype == np.float32:
        return np.asarray(
            shape_matched_control_f32(
                source.astype(np.float32, copy=False), kind, seed=seed
            )
        )
    return np.asarray(
        shape_matched_control(source.astype(np.float64, copy=False), kind, seed=seed)
    )


def _adjudication_margin(result: Any) -> float:
    if not isinstance(result, Mapping) or "circular_margin" not in result:
        raise TypeError(
            "label-shuffle pipeline must return a mapping with finite 'circular_margin'"
        )
    margin = float(result["circular_margin"])
    if not np.isfinite(margin):
        raise ValueError("label-shuffle pipeline returned a non-finite circular_margin")
    return margin


def run_label_shuffle_margin_null(
    data: ArrayLike,
    labels: ArrayLike,
    pipeline: Callable[[NDArray[Any], NDArray[Any], int], _ResultT],
    *,
    n_draws: int = 200,
    shuffle_seed: int = 11,
    pipeline_seed: int = 11,
) -> LabelShuffleMarginNull[_ResultT]:
    """Rebuild and adjudicate a label-aware chart under many label shuffles.

    ``pipeline`` receives ``(data, labels, pipeline_seed)`` and must rebuild the
    complete label-aware chart before returning an adjudication mapping with a
    finite ``circular_margin``. The observed labels and every independently
    shuffled draw traverse that same callback. Permuting an already-built chart
    is therefore outside this API's contract.

    Draw ``d`` permutes the labels through the native
    ``label_shuffle_permutation``, seeded only by ``(shuffle_seed, d)``. The
    one-sided randomization p-value is the native plus-one calibration
    ``(1 + count(null_margin >= observed_margin)) / (n_draws + 1)``. Ties are
    conservatively counted against the observed result. The source data are a
    shared read-only C-contiguous view so a 200-draw null does not copy a large
    activation corpus on every draw; each callback receives a private writable
    label vector.
    """

    if not callable(pipeline):
        raise TypeError("pipeline must be callable as pipeline(data, labels, seed)")
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(n_draws, Integral):
        raise TypeError("n_draws must be a positive integer")
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive; got {n_draws}")
    shuffle_seed = _u64(shuffle_seed, "shuffle_seed")
    pipeline_seed = _u64(pipeline_seed, "pipeline_seed")

    matrix = np.asarray(data)
    if matrix.dtype.kind == "c":
        raise TypeError(f"data must be real-valued; got complex dtype {matrix.dtype}")
    source_dtype = (
        np.float32
        if matrix.dtype.kind == "f" and matrix.dtype.itemsize == 4
        else np.float64
    )
    normalized = np.asarray(matrix, dtype=source_dtype, order="C")
    source = normalized.view()
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError(
            f"data must be a nonempty two-dimensional matrix; got shape {source.shape}"
        )
    _require_finite(source)
    source.setflags(write=False)

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.shape[0] != source.shape[0]:
        raise ValueError(
            "labels must be one-dimensional with one entry per data row; "
            f"got labels shape {label_array.shape} for {source.shape[0]} rows"
        )
    if label_array.dtype.kind in "fc" and not np.isfinite(label_array).all():
        raise ValueError("labels must not contain non-finite values")
    try:
        unique_label_count = np.unique(label_array).size
    except TypeError as error:
        raise TypeError(
            "labels must have one mutually comparable scalar dtype"
        ) from error
    if unique_label_count < 2:
        raise ValueError("a label-shuffle null requires at least two distinct labels")
    label_source = np.ascontiguousarray(label_array)

    observed = pipeline(source, label_source.copy(), pipeline_seed)
    observed_margin = _adjudication_margin(observed)
    null_margins = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        order = label_shuffle_permutation(label_source.shape[0], shuffle_seed, draw)
        null_result = pipeline(source, label_source[order], pipeline_seed)
        null_margins[draw] = _adjudication_margin(null_result)
    exceedance_count, p_value = randomization_p_value(observed_margin, null_margins)
    null_margins.setflags(write=False)
    return LabelShuffleMarginNull(
        observed=observed,
        observed_margin=observed_margin,
        null_margins=null_margins,
        exceedance_count=exceedance_count,
        p_value=p_value,
        n_draws=n_draws,
        shuffle_seed=shuffle_seed,
        pipeline_seed=pipeline_seed,
    )


def run_shape_controlled_census(
    data: ArrayLike,
    pipeline: Callable[[_ControlMatrix, int], _ResultT],
    *,
    control_seed: int = 11,
    pipeline_seed: int = 11,
) -> ShapeControlledCensus[_ResultT]:
    """Run one shape census and both controls through the identical pipeline.

    ``pipeline`` must be a deterministic callable of ``(matrix, seed)`` that
    fresh-fits the complete analysis under audit: SAE/dictionary training,
    co-activation grouping, intrinsic projection or subspace search, and shape
    adjudication. The callback receives a private writable matrix and the
    identical ``pipeline_seed`` on all three calls. The read-only internal
    source is never exposed, so callback mutation cannot contaminate a later
    control. Callers must not mutate the original input concurrently while this
    function is running.

    Native float32 and float64 inputs preserve their dtype in all three
    callbacks. An already C-contiguous native input is retained as a read-only
    view without a corpus-sized source copy. Other layouts and dtypes are
    normalized exactly once. The covariance-exact transform uses at most
    ``min(8, worker_threads)`` bounded row-block/column-band float64
    workspaces whose total size is at most that many ``B × p`` matrices, with
    ``B <= 1024``, and is hard-capped at 128 MiB. Wide inputs are split into
    at-most-32-MiB tiles; the transform never forms a ``p × p`` covariance or
    a corpus-sized float64 copy of float32 input.

    The controls are generated at pipeline entry, not from a fitted 2-D chart:
    a per-dimension permutation preserves every marginal, while a mean-fixing
    orthogonal randomized Hadamard transform preserves the full empirical mean
    and covariance in exact arithmetic. Both native controls take
    ``control_seed`` and separate their streams by control kind. Only one
    control matrix is resident at a time unless the callback retains it.
    """

    if not callable(pipeline):
        raise TypeError("pipeline must be callable as pipeline(matrix, seed)")
    control_seed = _u64(control_seed, "control_seed")
    pipeline_seed = _u64(pipeline_seed, "pipeline_seed")
    declared_dtype = getattr(data, "dtype", None)
    try:
        input_dtype = np.dtype(declared_dtype) if declared_dtype is not None else None
    except TypeError:
        input_dtype = None
    normalized_input = data
    # Array-like containers such as nested Python lists have no declared dtype.
    # Materialize them once without coercion so complex values are rejected
    # rather than silently losing their imaginary part in a float conversion.
    if input_dtype is None:
        normalized_input = np.asarray(data)
        input_dtype = normalized_input.dtype
    if input_dtype is not None and input_dtype.kind == "c":
        raise TypeError(
            f"data must be real-valued; complex dtype {input_dtype} is not supported"
        )
    if (
        input_dtype is not None
        and input_dtype.kind == "f"
        and input_dtype.itemsize == 4
    ):
        source_dtype: np.dtype[np.float32] | np.dtype[np.float64] = np.dtype(
            np.float32
        )
    else:
        source_dtype = np.dtype(np.float64)
    # np.asarray reuses an already native C-contiguous array and performs the
    # only normalization allocation otherwise. A view lets us enforce an
    # internal read-only contract without changing the caller's writeable flag.
    normalized = np.asarray(normalized_input, dtype=source_dtype, order="C")
    source = normalized.view()
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError(
            f"data must be a nonempty two-dimensional matrix; got shape {source.shape}"
        )
    _require_finite(source)
    source.setflags(write=False)

    observed = pipeline(source.copy(order="C"), pipeline_seed)

    shuffled_array = _native_shape_control(
        source,
        "per_dimension_shuffle",
        seed=control_seed,
    )
    if (
        shuffled_array.dtype != source_dtype
        or shuffled_array.shape != source.shape
        or not shuffled_array.flags.c_contiguous
        or not shuffled_array.flags.writeable
        or np.shares_memory(shuffled_array, source)
    ):
        raise RuntimeError(
            "native per-dimension shuffle violated its private writable C-array "
            f"contract: dtype={shuffled_array.dtype}, shape={shuffled_array.shape}, "
            f"C={shuffled_array.flags.c_contiguous}, "
            f"writeable={shuffled_array.flags.writeable}"
        )
    shuffled_result = pipeline(shuffled_array, pipeline_seed)
    del shuffled_array

    hadamard_array = _native_shape_control(
        source,
        "covariance_exact_hadamard",
        seed=control_seed,
    )
    if (
        hadamard_array.dtype != source_dtype
        or hadamard_array.shape != source.shape
        or not hadamard_array.flags.c_contiguous
        or not hadamard_array.flags.writeable
        or np.shares_memory(hadamard_array, source)
    ):
        raise RuntimeError(
            "native covariance-exact Hadamard control violated its private "
            f"writable C-array contract: dtype={hadamard_array.dtype}, "
            f"shape={hadamard_array.shape}, C={hadamard_array.flags.c_contiguous}, "
            f"writeable={hadamard_array.flags.writeable}"
        )
    hadamard_result = pipeline(hadamard_array, pipeline_seed)
    del hadamard_array

    return ShapeControlledCensus(
        observed=observed,
        per_dimension_shuffle=shuffled_result,
        covariance_exact_hadamard=hadamard_result,
        pipeline_seed=pipeline_seed,
        control_seed=control_seed,
    )
