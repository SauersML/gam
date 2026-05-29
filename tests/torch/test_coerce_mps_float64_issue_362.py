"""Regression test for issue #362: torch penalties crash on MPS.

The gamfit torch bridge downloads accelerator tensors to a contiguous f64
NumPy array before handing them to the Rust engine. The buggy coercion fused
the host move and the dtype cast into a single ``Tensor.to(device="cpu",
dtype=float64)`` (or the symmetric ``.to(dtype=float64, device="cpu")`` /
``.to(dtype=float64).cpu()``). PyTorch resolves the float64 cast on the
*source* device, and MPS has no float64 — so on Apple Silicon the call raised::

    TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS
    framework doesn't support float64. Please use float32 instead.

CI has no MPS hardware, so we faithfully reproduce the semantics with a
``torch.Tensor`` subclass whose ``__torch_function__`` intercepts ``Tensor.to``
and raises exactly as MPS does whenever a float64 cast is requested while the
tensor has *not yet* been moved to CPU. A bridge that moves to CPU first and
casts second survives; a bridge that fuses the two ops (or casts before the
host move) trips the guard. This pins the host-first ordering for every bridge
helper that feeds the Rust engine.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")

from gamfit import _frame_torch  # type: ignore[attr-defined]
from gamfit.torch import _coerce  # type: ignore[attr-defined]
from gamfit.torch import interchange as _interchange  # type: ignore[attr-defined]


class PseudoMpsTensor(torch.Tensor):
    """A float32 tensor that refuses float64 until it has reached CPU.

    Mirrors the real MPS contract: the backend simply does not have a float64
    kernel, so casting to float64 while still "on device" raises rather than
    silently succeeding. Moving to CPU clears the marker, after which float64
    is allowed.
    """

    _on_pseudo_mps: bool = True

    @staticmethod
    def __new__(cls, data: Any) -> "PseudoMpsTensor":
        base = torch.as_tensor(np.asarray(data, dtype=np.float32))
        return base.as_subclass(cls)  # type: ignore[return-value]

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Any = None,
    ) -> Any:
        kwargs = kwargs or {}
        if func is torch.Tensor.to and args and isinstance(args[0], PseudoMpsTensor):
            self_t = args[0]
            on_device = bool(getattr(self_t, "_on_pseudo_mps", False))
            target_dtype = kwargs.get("dtype")
            target_device = kwargs.get("device")
            for extra in args[1:]:
                if isinstance(extra, torch.dtype):
                    target_dtype = extra
                elif isinstance(extra, (str, torch.device)):
                    target_device = extra
            moving_to_cpu = target_device is not None and str(target_device) == "cpu"
            if on_device and target_dtype == torch.float64 and not moving_to_cpu:
                raise TypeError(
                    "Cannot convert a MPS Tensor to float64 dtype as the MPS "
                    "framework doesn't support float64. Please use float32 instead."
                )
            result = super().__torch_function__(func, types, args, kwargs)
            if isinstance(result, PseudoMpsTensor):
                # Moving to CPU lands the tensor on the host: clear the marker
                # so the subsequent float64 cast is permitted.
                result._on_pseudo_mps = on_device and not moving_to_cpu
            return result
        return super().__torch_function__(func, types, args, kwargs)


def test_to_numpy_f64_moves_host_before_casting() -> None:
    """`gamfit.torch._coerce.to_numpy_f64` downloads to CPU *then* casts to f64."""
    t = PseudoMpsTensor([[1.5, -2.0, 3.25], [0.0, 4.5, -1.0]])
    # On the buggy code this raises the MPS TypeError; the fix returns the array.
    arr = _coerce.to_numpy_f64(t)
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous
    np.testing.assert_allclose(
        arr, np.array([[1.5, -2.0, 3.25], [0.0, 4.5, -1.0]], dtype=np.float64)
    )


def test_frame_to_numpy_f64_moves_host_before_casting() -> None:
    """`gamfit._frame_torch.to_numpy_f64` shares the host-first contract."""
    t = PseudoMpsTensor([7.0, -3.5, 0.25])
    arr = _frame_torch.to_numpy_f64(t)
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, np.array([7.0, -3.5, 0.25], dtype=np.float64))


def test_interchange_as_f64_moves_host_before_casting() -> None:
    """`gamfit.torch.interchange._as_f64_cpu` must not fuse the cpu+f64 cast."""
    t = PseudoMpsTensor([[2.0, -1.0], [0.5, 3.0]])
    arr = _interchange._as_f64_cpu(t)
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous
    np.testing.assert_allclose(
        arr, np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
    )


def test_pseudo_mps_guard_rejects_fused_cast() -> None:
    """Sanity check: the emulated guard truly reproduces the MPS failure.

    Without this, a no-op bridge could pass the tests above vacuously. The
    fused cast used by the original buggy code must still raise here.
    """
    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="MPS"):
        t.to(device="cpu", dtype=torch.float64)
    with pytest.raises(TypeError, match="MPS"):
        t.to(dtype=torch.float64)
    # The fixed ordering — host first, then cast — succeeds.
    moved = t.to(device="cpu").to(dtype=torch.float64)
    assert moved.dtype == torch.float64
