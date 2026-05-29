"""Regression test for issue #362: torch penalties crash on MPS.

The gamfit torch bridge downloads accelerator tensors to a contiguous f64
NumPy array before handing them to the Rust engine. The buggy coercion fused
the host move and the dtype cast into a single ``Tensor.to(device="cpu",
dtype=float64)`` (and the symmetric ``.to(dtype=float64, device="cpu")`` /
``.to(dtype=float64).cpu()``). As issue #362 documents, PyTorch resolves the
``float64`` cast on the *source* (MPS) device even when a CPU destination is
named in the same op, and MPS has no float64 — so on Apple Silicon the call
raised::

    TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS
    framework doesn't support float64. Please use float32 instead.

CI has no MPS hardware, so we faithfully reproduce the semantics with a
``torch.Tensor`` subclass that

  * reports its device as ``mps`` until it has actually been moved to CPU
    (the bridge helpers branch on ``tensor.device.type``), and
  * intercepts ``Tensor.to`` / ``Tensor.cpu`` and raises exactly as MPS does
    whenever a ``float64`` cast is requested **in the same op** as the tensor
    still being on device — *including* the fused ``.to(device="cpu",
    dtype=float64)`` form, which is the precise call issue #362 says crashes.

A bridge that moves to CPU first and casts second survives; a bridge that
fuses the two ops (or casts before the host move) trips the guard. This pins
the host-first ordering for every bridge helper that feeds the Rust engine.
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

_MPS_ERROR = (
    "Cannot convert a MPS Tensor to float64 dtype as the MPS "
    "framework doesn't support float64. Please use float32 instead."
)


class PseudoMpsTensor(torch.Tensor):
    """A float32 tensor that emulates the MPS no-float64 contract.

    It reports ``device.type == "mps"`` until a host move actually lands it on
    CPU, and refuses any float64 cast requested while still on device — even
    when that cast is fused with the CPU move (``.to(device="cpu",
    dtype=float64)``), exactly as the real MPS backend does in issue #362.
    """

    _on_pseudo_mps: bool = True

    @staticmethod
    def __new__(cls, data: Any) -> "PseudoMpsTensor":
        base = torch.as_tensor(np.asarray(data, dtype=np.float32))
        obj = base.as_subclass(cls)  # type: ignore[assignment]
        obj._on_pseudo_mps = True
        return obj  # type: ignore[return-value]

    @property  # type: ignore[override]
    def device(self) -> torch.device:
        # Real storage is CPU, but advertise MPS until a host move clears the
        # marker, so the bridge's ``tensor.device.type != "cpu"`` branch fires
        # the same way it would on Apple Silicon.
        if getattr(self, "_on_pseudo_mps", False):
            return torch.device("mps:0")
        return torch.device("cpu")

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Any = None,
    ) -> Any:
        kwargs = kwargs or {}
        is_to = func is torch.Tensor.to
        is_cpu = func is torch.Tensor.cpu
        if (is_to or is_cpu) and args and isinstance(args[0], PseudoMpsTensor):
            self_t = args[0]
            on_device = bool(getattr(self_t, "_on_pseudo_mps", False))

            target_dtype = kwargs.get("dtype")
            target_device = kwargs.get("device")
            if is_to:
                for extra in args[1:]:
                    if isinstance(extra, torch.dtype):
                        target_dtype = extra
                    elif isinstance(extra, (str, torch.device)):
                        target_device = extra
            else:  # Tensor.cpu(): an unconditional host move, no dtype change.
                target_device = "cpu"

            wants_f64 = target_dtype == torch.float64
            moving_to_cpu = target_device is not None and str(target_device).startswith(
                "cpu"
            )

            # MPS resolves the dtype cast on the *source* device, so a float64
            # cast requested while still on device raises regardless of whether
            # a CPU destination is named in the same op. Only a host move with
            # no float64 cast in the same call is allowed.
            if on_device and wants_f64:
                raise TypeError(_MPS_ERROR)

            result = super().__torch_function__(func, types, args, kwargs)
            if isinstance(result, PseudoMpsTensor):
                # The marker clears once (and only once) the tensor has been
                # moved to the host.
                result._on_pseudo_mps = on_device and not moving_to_cpu
            return result
        return super().__torch_function__(func, types, args, kwargs)


def test_to_numpy_f64_moves_host_before_casting() -> None:
    """`gamfit.torch._coerce.to_numpy_f64` downloads to CPU *then* casts to f64."""
    t = PseudoMpsTensor([[1.5, -2.0, 3.25], [0.0, 4.5, -1.0]])
    assert t.device.type == "mps"
    # On the buggy fused code this raises the MPS TypeError; the fix returns it.
    arr = _coerce.to_numpy_f64(t)
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous
    np.testing.assert_allclose(
        arr, np.array([[1.5, -2.0, 3.25], [0.0, 4.5, -1.0]], dtype=np.float64)
    )


def test_frame_to_numpy_f64_moves_host_before_casting() -> None:
    """`gamfit._frame_torch.to_numpy_f64` shares the host-first contract."""
    t = PseudoMpsTensor([7.0, -3.5, 0.25])
    assert t.device.type == "mps"
    arr = _frame_torch.to_numpy_f64(t)
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, np.array([7.0, -3.5, 0.25], dtype=np.float64))


def test_interchange_as_f64_moves_host_before_casting() -> None:
    """`gamfit.torch.interchange._as_f64_cpu` must not fuse the cpu+f64 cast."""
    t = PseudoMpsTensor([[2.0, -1.0], [0.5, 3.0]])
    assert t.device.type == "mps"
    arr = _interchange._as_f64_cpu(t)
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous
    np.testing.assert_allclose(
        arr, np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
    )


def test_pseudo_mps_guard_rejects_buggy_fused_casts() -> None:
    """Sanity check: the emulation reproduces every buggy form from issue #362.

    Without this, a no-op bridge could pass the tests above vacuously. All
    three fused/cast-first forms used by the original buggy code must raise the
    MPS TypeError here, while the fixed host-first ordering must succeed.
    """
    # 1. The fused form named in the issue and used by the old _coerce /
    #    _frame_torch helpers.
    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="MPS"):
        t.to(device="cpu", dtype=torch.float64)

    # 2. The symmetric kwarg order used by the old interchange helpers.
    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="MPS"):
        t.to(dtype=torch.float64, device="cpu")

    # 3. The cast-before-host-move form used by the old basis-eval shims.
    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="MPS"):
        t.to(dtype=torch.float64).cpu()

    # 4. A bare on-device float64 cast.
    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="MPS"):
        t.to(dtype=torch.float64)

    # The fixed ordering — host first, then cast — succeeds for both the
    # ``.to("cpu")`` and ``.cpu()`` host moves.
    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    moved = t.to(device="cpu").to(dtype=torch.float64)
    assert moved.dtype == torch.float64
    assert moved.device.type == "cpu"

    t = PseudoMpsTensor([1.0, 2.0, 3.0])
    moved = t.cpu().to(dtype=torch.float64)
    assert moved.dtype == torch.float64
    assert moved.device.type == "cpu"
