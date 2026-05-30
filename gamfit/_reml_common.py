"""Shared cross-framework Gaussian REML VJP bridge contract.

Every Gaussian REML wrapper — the NumPy face in :mod:`gamfit._api`, the
PyTorch autograd faces in :mod:`gamfit.torch._reml`, and the Rust
``#[pyfunction]`` bindings — repeats the same marshalling contract:

* coerce framework tensors/arrays to f64 CPU,
* call the Rust forward,
* persist the converged forward-state schema,
* accept upstream gradients,
* call the Rust backward,
* return framework-shaped gradients in forward-positional order.

Duplicating that contract per framework lets dtype handling, optional
``weights``, ``by``-gating, the forward-state schema, gradient naming, and
batch-shape semantics drift apart. This module is the single source of truth
for the *point-design* (single + ragged-batched) REML fits, which share a
byte-for-byte identical contract modulo one positional argument
(``row_offsets``) and whether the ``λ`` / ``reml_score`` / ``edf`` gradients
are scalars or per-problem vectors.

The block-additive and constrained variants have genuinely different output
tuples, forward-state schemas, and gradient routing; they are *not* forced
through this factory (that would be a lossy abstraction). Instead they reuse
the shared coercion / forward-state helpers defined here so the contract
stays unified at the level where it is actually identical.

The factory here is framework-agnostic: it is parameterised by a small set
of callbacks (:class:`RemlBridgeOps`) that a concrete framework supplies for
its tensor type. :mod:`gamfit.torch._reml` plugs PyTorch into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class RemlForwardSchema(Protocol):
    """The forward outputs a point-design Gaussian REML fit always returns.

    All five fields are present for both the single and batched fits. For the
    single fit ``lam`` / ``reml_score`` / ``edf`` are scalars; for the batched
    fit they are length-``K`` vectors. ``coefficients`` and ``fitted`` follow
    the engine's packed layout in both cases.
    """

    coefficients: Any
    fitted: Any
    lam: Any
    reml_score: Any
    edf: Any


# Canonical forward-output key order. The Rust forward returns ``"lambda"``
# (a Python keyword-collision rename of ``lam``); every framework face reads
# the engine payload through this single mapping so the schema never drifts.
REML_FORWARD_KEYS: tuple[str, ...] = (
    "coefficients",
    "fitted",
    "lambda",
    "reml_score",
    "edf",
)

# Canonical backward-gradient key order, matching the engine payload. ``grad_by``
# is only populated when the forward was ``by``-gated; the others are always
# present for the point-design fits.
REML_BACKWARD_KEYS: tuple[str, ...] = (
    "grad_x",
    "grad_y",
    "grad_penalty",
    "grad_weights",
    "grad_by",
)


@dataclass(frozen=True, slots=True)
class RemlCallSpec:
    """Describes one point-design Gaussian REML variant for the bridge factory.

    ``forward`` / ``backward`` are the :mod:`gamfit._api` NumPy entrypoints for
    this variant (already framework-agnostic — they coerce to NumPy and call
    Rust). ``batched`` selects the ragged-batch contract: it adds the
    ``row_offsets`` positional argument and switches the ``λ`` / ``reml_score``
    / ``edf`` gradients from scalars to per-problem vectors.
    """

    name: str
    forward: Callable[..., dict[str, Any]]
    backward: Callable[..., dict[str, Any]]
    batched: bool


def check_forward_state(state: Any, *, name: str) -> None:
    """Assert a converged forward-state payload carries the required schema.

    The forward-state dict is the analytic-VJP hand-off: it carries the
    converged geometry caches the backward needs so it does not re-solve the
    inner problem. Every framework face routes its forward-state through this
    single check, so an incomplete payload is caught at one place instead of
    surfacing as a silent gradient discrepancy across frameworks.
    """
    if not isinstance(state, dict):
        raise TypeError(
            f"{name}: forward_state must be a dict, got {type(state).__name__}"
        )
    for required in ("coefficients", "fitted"):
        if required not in state:
            raise KeyError(
                f"{name}: forward_state is missing required key {required!r}; "
                "the analytic backward cannot reconstruct the converged fit"
            )


def validate_forward_state(state: dict[str, Any], *, name: str) -> dict[str, Any]:
    """Validate and defensively copy a converged forward-state payload.

    Wraps :func:`check_forward_state` with the defensive copy a framework face
    needs when it stashes the payload on an autograd context that outlives the
    forward call. NumPy arrays are shared by reference out of the engine
    payload; copying here makes the stashed state immune to caller-side
    in-place mutation between forward and backward — the same guarantee the
    torch version-counter gives for the differentiable inputs.
    """
    check_forward_state(state, name=name)
    import numpy as np

    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


class RemlBridgeOps(Protocol):
    """Framework callbacks the point-design REML factory needs.

    A concrete framework (PyTorch, JAX, …) supplies one implementation binding
    its tensor type into the shared marshalling. The factory never imports the
    framework directly, so the contract stays in one place no matter how many
    frameworks plug in.
    """

    def to_numpy_f64(self, value: Any) -> Any:
        """Detach ``value`` to a contiguous f64 CPU NumPy array."""

    def to_numpy_uintp(self, value: Any) -> Any:
        """Detach ``value`` to a contiguous ``uintp`` CPU NumPy array."""

    def from_numpy_like(self, array: Any, ref: Any) -> Any:
        """Wrap ``array`` as a framework tensor matching ``ref``'s device/dtype."""

    def scalar_grad(self, grad: Any | None) -> float:
        """Reduce an upstream scalar-output gradient to a Python float."""

    def vector_grad(self, grad: Any | None) -> Any | None:
        """Coerce an upstream vector-output gradient to a NumPy array (or None)."""


@dataclass(slots=True)
class RemlForwardRun:
    """Result of the shared forward marshalling.

    ``outputs`` are the five framework tensors in :data:`REML_FORWARD_KEYS`
    order; ``forward_state`` is the validated converged payload; ``arrays`` is
    the NumPy view of every coerced input keyed by argument name, reused by the
    backward so it never re-coerces.
    """

    outputs: tuple[Any, Any, Any, Any, Any]
    forward_state: dict[str, Any]
    arrays: dict[str, Any]


def run_point_design_forward(
    spec: RemlCallSpec,
    ops: RemlBridgeOps,
    *,
    x: Any,
    y: Any,
    penalty: Any,
    row_offsets: Any | None,
    weights: Any | None,
    by: Any | None,
    init_lambda: float | None,
    by_start_col: int,
    ref: Any,
) -> RemlForwardRun:
    """Coerce inputs, run the NumPy forward, wrap outputs back into ``ref``.

    This is the single forward marshalling for both the single and batched
    point-design fits. ``row_offsets`` is required iff ``spec.batched`` and is
    ignored otherwise.
    """
    import numpy as np

    x_np = ops.to_numpy_f64(x)
    y_np = ops.to_numpy_f64(y)
    penalty_np = ops.to_numpy_f64(penalty)
    weights_np = None if weights is None else ops.to_numpy_f64(weights)
    by_np = None if by is None else ops.to_numpy_f64(by)

    arrays: dict[str, Any] = {
        "x": x_np,
        "y": y_np,
        "penalty": penalty_np,
        "weights": weights_np,
        "by": by_np,
    }

    if spec.batched:
        if row_offsets is None:
            raise ValueError(f"{spec.name}: batched fit requires row_offsets")
        offsets_np = ops.to_numpy_uintp(row_offsets)
        arrays["row_offsets"] = offsets_np
        out = spec.forward(
            x_np,
            y_np,
            offsets_np,
            penalty_np,
            weights=weights_np,
            init_lambda=init_lambda,
            by=by_np,
            by_start_col=by_start_col,
        )
    else:
        out = spec.forward(
            x_np,
            y_np,
            penalty_np,
            weights=weights_np,
            init_lambda=init_lambda,
            by=by_np,
            by_start_col=by_start_col,
        )

    forward_state = validate_forward_state(out, name=spec.name)
    outputs = tuple(
        ops.from_numpy_like(np.asarray(out[key], dtype=np.float64), ref)
        for key in REML_FORWARD_KEYS
    )
    return RemlForwardRun(
        outputs=outputs,  # type: ignore[arg-type]
        forward_state=forward_state,
        arrays=arrays,
    )


def run_point_design_backward(
    spec: RemlCallSpec,
    ops: RemlBridgeOps,
    *,
    arrays: dict[str, Any],
    forward_state: dict[str, Any],
    grad_outputs: tuple[Any, Any, Any, Any, Any],
    init_lambda: float | None,
    by_start_col: int,
    has_weights: bool,
    has_by: bool,
    ref: Any,
    penalty_ref: Any,
) -> dict[str, Any | None]:
    """Marshal upstream gradients, run the NumPy backward, wrap grads to ``ref``.

    ``grad_outputs`` is the five upstream cotangents in
    :data:`REML_FORWARD_KEYS` order. For the batched fit the
    ``λ`` / ``reml_score`` / ``edf`` cotangents are per-problem vectors; for
    the single fit they are scalars. Returns a dict keyed by
    :data:`REML_BACKWARD_KEYS` with framework tensors (or ``None``).
    """
    (
        grad_coefficients,
        grad_fitted,
        grad_lam,
        grad_reml_score,
        grad_edf,
    ) = grad_outputs

    grad_coef_np = None if grad_coefficients is None else ops.to_numpy_f64(grad_coefficients)
    grad_fitted_np = None if grad_fitted is None else ops.to_numpy_f64(grad_fitted)

    if spec.batched:
        grad_lam_arg = ops.vector_grad(grad_lam)
        grad_reml_arg = ops.vector_grad(grad_reml_score)
        grad_edf_arg = ops.vector_grad(grad_edf)
        result = spec.backward(
            arrays["x"],
            arrays["y"],
            arrays["row_offsets"],
            arrays["penalty"],
            grad_lambda=grad_lam_arg,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_arg,
            grad_edf=grad_edf_arg,
            forward_state=forward_state,
            weights=arrays["weights"],
            init_lambda=init_lambda,
            by=arrays["by"],
            by_start_col=by_start_col,
        )
    else:
        result = spec.backward(
            arrays["x"],
            arrays["y"],
            arrays["penalty"],
            grad_lambda=ops.scalar_grad(grad_lam),
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=ops.scalar_grad(grad_reml_score),
            grad_edf=ops.scalar_grad(grad_edf),
            forward_state=forward_state,
            weights=arrays["weights"],
            init_lambda=init_lambda,
            by=arrays["by"],
            by_start_col=by_start_col,
        )

    def wrap(key: str, into: Any) -> Any | None:
        value = result.get(key)
        if value is None:
            return None
        import numpy as np

        return ops.from_numpy_like(np.asarray(value, dtype=np.float64), into)

    return {
        "grad_x": wrap("grad_x", ref),
        "grad_y": wrap("grad_y", ref),
        "grad_penalty": wrap("grad_penalty", penalty_ref),
        "grad_weights": wrap("grad_weights", ref) if has_weights else None,
        "grad_by": wrap("grad_by", ref) if has_by else None,
    }
