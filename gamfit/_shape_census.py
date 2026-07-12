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
_GAUSSIAN_SEED_DOMAIN = 0xC0A4_71A1
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
    covariance_matched_gaussian: _ResultT
    pipeline_seed: int
    per_dimension_shuffle_seed: int
    covariance_matched_gaussian_seed: int


def _u64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in [0, 2**64 - 1]")
    if not 0 <= value <= _U64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64 - 1]; got {value}")
    return value


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
    adjudication. The callback receives a private writable matrix copy and the
    identical ``pipeline_seed`` on all three calls. The immutable source matrix
    is never exposed, so callback mutation cannot contaminate a later control.

    Native float32 and float64 inputs preserve their dtype in all three
    callbacks. Float32 controls accumulate only their `O(p²)` moments and
    eigendecomposition in float64; no `n × p` float64 matrix is created. Every
    other input dtype is converted once to a native contiguous float64 source.

    The controls are generated at pipeline entry, not from a fitted 2-D chart:
    a per-dimension permutation preserves every marginal, while a Gaussian draw
    preserves the full empirical mean/covariance. Distinct deterministic seed
    domains match ``adjudicate_atom_shape``'s built-in coordinate-level controls.
    Only one control matrix is resident at a time unless the callback retains it.
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
    if input_dtype is not None and input_dtype.kind == "f" and input_dtype.itemsize == 4:
        source_dtype = np.dtype(np.float32)
        control_function = shape_matched_control_f32
    else:
        source_dtype = np.dtype(np.float64)
        control_function = shape_matched_control
    source = np.array(data, dtype=source_dtype, order="C", copy=True)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError(
            f"data must be a nonempty two-dimensional matrix; got shape {source.shape}"
        )
    if not np.isfinite(source).all():
        raise ValueError("data must contain only finite values")
    source.setflags(write=False)

    shuffle_seed = control_seed ^ _SHUFFLE_SEED_DOMAIN
    gaussian_seed = control_seed ^ _GAUSSIAN_SEED_DOMAIN
    observed = pipeline(source.copy(order="C"), pipeline_seed)

    shuffled = control_function(
        source,
        "per_dimension_shuffle",
        seed=shuffle_seed,
    )
    shuffled_array = np.asarray(shuffled)
    if shuffled_array.dtype != source_dtype or shuffled_array.shape != source.shape:
        raise RuntimeError(
            "native per-dimension shuffle returned the wrong dtype or shape: "
            f"{shuffled_array.dtype} {shuffled_array.shape}"
        )
    shuffled_result = pipeline(shuffled_array, pipeline_seed)
    del shuffled
    del shuffled_array

    gaussian = control_function(
        source,
        "covariance_matched_gaussian",
        seed=gaussian_seed,
    )
    gaussian_array = np.asarray(gaussian)
    if gaussian_array.dtype != source_dtype or gaussian_array.shape != source.shape:
        raise RuntimeError(
            "native covariance-matched Gaussian returned the wrong dtype or shape: "
            f"{gaussian_array.dtype} {gaussian_array.shape}"
        )
    gaussian_result = pipeline(gaussian_array, pipeline_seed)
    del gaussian
    del gaussian_array

    return ShapeControlledCensus(
        observed=observed,
        per_dimension_shuffle=shuffled_result,
        covariance_matched_gaussian=gaussian_result,
        pipeline_seed=pipeline_seed,
        per_dimension_shuffle_seed=shuffle_seed,
        covariance_matched_gaussian_seed=gaussian_seed,
    )
