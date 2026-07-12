"""Whole-pipeline matched controls for representational-shape censuses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._rust import shape_matched_control, shape_matched_control_f32


_ResultT = TypeVar("_ResultT")
_U64_MAX = (1 << 64) - 1
_SHUFFLE_SEED_DOMAIN = 0xD1AE_510F
_HADAMARD_SEED_DOMAIN = 0x4841_DA4D
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
    per_dimension_shuffle_seed: int
    covariance_exact_hadamard_seed: int


def _u64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in [0, 2**64 - 1]")
    if not 0 <= value <= _U64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64 - 1]; got {value}")
    return value


def _require_finite(matrix: _ControlMatrix) -> None:
    """Validate finiteness with bounded temporary storage."""
    flattened = matrix.reshape(-1)
    for start in range(0, flattened.size, _FINITE_SCAN_SCALARS):
        if not np.isfinite(flattened[start : start + _FINITE_SCAN_SCALARS]).all():
            raise ValueError("data must contain only finite values")


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
    normalized exactly once. The covariance-exact transform uses one bounded
    ``B × p`` float64 workspace with ``B <= 1024``; it never forms a ``p × p``
    covariance or a corpus-sized float64 copy of float32 input.

    The controls are generated at pipeline entry, not from a fitted 2-D chart:
    a per-dimension permutation preserves every marginal, while a mean-fixing
    orthogonal randomized Hadamard transform preserves the full empirical mean
    and covariance in exact arithmetic. Only one control matrix is resident at
    a time unless the callback retains it.
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
    if input_dtype is not None and input_dtype.kind == "c":
        raise TypeError(
            f"data must be real-valued; complex dtype {input_dtype} is not supported"
        )
    if input_dtype is not None and input_dtype.kind == "f" and input_dtype.itemsize == 4:
        source_dtype = np.dtype(np.float32)
        control_function = shape_matched_control_f32
    else:
        source_dtype = np.dtype(np.float64)
        control_function = shape_matched_control
    # np.asarray reuses an already native C-contiguous array and performs the
    # only normalization allocation otherwise. A view lets us enforce an
    # internal read-only contract without changing the caller's writeable flag.
    normalized = np.asarray(data, dtype=source_dtype, order="C")
    source = normalized.view()
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError(
            f"data must be a nonempty two-dimensional matrix; got shape {source.shape}"
        )
    _require_finite(source)
    source.setflags(write=False)

    shuffle_seed = control_seed ^ _SHUFFLE_SEED_DOMAIN
    hadamard_seed = control_seed ^ _HADAMARD_SEED_DOMAIN
    observed = pipeline(source.copy(order="C"), pipeline_seed)

    shuffled = control_function(
        source,
        "per_dimension_shuffle",
        seed=shuffle_seed,
    )
    shuffled_array = np.asarray(shuffled)
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
    del shuffled
    del shuffled_array

    hadamard = control_function(
        source,
        "covariance_exact_hadamard",
        seed=hadamard_seed,
    )
    hadamard_array = np.asarray(hadamard)
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
    del hadamard
    del hadamard_array

    return ShapeControlledCensus(
        observed=observed,
        per_dimension_shuffle=shuffled_result,
        covariance_exact_hadamard=hadamard_result,
        pipeline_seed=pipeline_seed,
        per_dimension_shuffle_seed=shuffle_seed,
        covariance_exact_hadamard_seed=hadamard_seed,
    )
