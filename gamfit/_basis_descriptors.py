"""Callable basis descriptors implementing :class:`BasisDescriptor`.

The basis math lives in Rust: :class:`PeriodicHarmonic` (and its alias
:class:`Fourier`) evaluates the trig columns via ``periodic_harmonic_basis``
and routes ``jacobian`` through ``periodic_harmonic_basis_derivative``. The
Python layer is glue — input coercion, autograd wrapping, and the
``BasisDescriptor`` protocol.

Torch interop: when ``t`` is a torch tensor we wrap the Rust call in a
:class:`torch.autograd.Function` so gradients flow back through ``t`` via
the analytic derivative basis the Rust side returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._binding import rust_module as _rust_module
from ._protocol import BasisDescriptor, _require_torch


def _rust_basis(theta: np.ndarray, harmonics: int) -> np.ndarray:
    out = _rust_module().periodic_harmonic_basis(np.ascontiguousarray(theta), int(harmonics))
    return np.asarray(out, dtype=np.float64)


def _rust_basis_derivative(theta: np.ndarray, harmonics: int) -> np.ndarray:
    out = _rust_module().periodic_harmonic_basis_derivative(
        np.ascontiguousarray(theta), int(harmonics)
    )
    return np.asarray(out, dtype=np.float64)


class _PeriodicHarmonicFn:
    """Lazy torch.autograd.Function bridging to the Rust periodic harmonic basis."""

    _impl = None

    @classmethod
    def get(cls) -> Any:
        if cls._impl is not None:
            return cls._impl
        torch = _require_torch()

        class _Impl(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, t: Any, harmonics: int) -> Any:
                t_np = t.detach().cpu().to(torch.float64).contiguous().numpy().reshape(-1)
                phi = _rust_basis(t_np, harmonics)
                ctx.save_for_backward(t)
                ctx.harmonics = harmonics
                return torch.as_tensor(phi, dtype=t.dtype, device=t.device)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
                (t,) = ctx.saved_tensors
                t_np = t.detach().cpu().to(torch.float64).contiguous().numpy().reshape(-1)
                d_phi = _rust_basis_derivative(t_np, ctx.harmonics)
                d_phi_t = torch.as_tensor(d_phi, dtype=t.dtype, device=t.device)
                grad_t = (grad_output.to(d_phi_t.dtype) * d_phi_t).sum(dim=-1).reshape_as(t)
                return grad_t, None

        cls._impl = _Impl
        return cls._impl


class PeriodicHarmonic(BasisDescriptor):
    """Periodic harmonic (Fourier) basis on the unit circle. Math via Rust.

    With ``harmonics=H`` the basis has width ``2H + 1`` and columns

        ``[1, cos(θ), sin(θ), cos(2θ), sin(2θ), …, cos(Hθ), sin(Hθ)]``.

    ``theta`` is interpreted in radians. Alias: :class:`Fourier`.
    """

    def __init__(self, harmonics: int = 3, *, num_basis: int | None = None) -> None:
        if num_basis is not None:
            if int(num_basis) < 1 or int(num_basis) % 2 == 0:
                raise ValueError(
                    "PeriodicHarmonic.num_basis must be a positive odd integer (2H+1)"
                )
            harmonics = (int(num_basis) - 1) // 2
        if int(harmonics) < 0:
            raise ValueError("PeriodicHarmonic.harmonics must be >= 0")
        self.harmonics = int(harmonics)

    @property
    def output_dim(self) -> int:
        return 2 * self.harmonics + 1

    @property
    def input_dim(self) -> int:
        return 1

    def evaluate(self, t: Any) -> Any:
        torch_mod = _maybe_import_torch()
        if torch_mod is not None and isinstance(t, torch_mod.Tensor):
            theta = t.squeeze(-1) if (t.dim() == 2 and t.shape[1] == 1) else t
            if theta.dim() != 1:
                raise ValueError(
                    f"PeriodicHarmonic.evaluate expects a 1-D tensor or (N, 1), "
                    f"got shape {tuple(t.shape)}"
                )
            fn = _PeriodicHarmonicFn.get()
            return fn.apply(theta, self.harmonics)
        arr = np.asarray(t, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(
                f"PeriodicHarmonic.evaluate expects a 1-D array or (N, 1), "
                f"got shape {arr.shape}"
            )
        return _rust_basis(arr, self.harmonics)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "periodic_harmonic", "harmonics": self.harmonics}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PeriodicHarmonic":
        return cls(harmonics=int(d.get("harmonics", 3)))

    def __repr__(self) -> str:
        return f"PeriodicHarmonic(harmonics={self.harmonics})"


def _maybe_import_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError:
        return None


# Alias requested by the protocol spec.
Fourier = PeriodicHarmonic


__all__ = ["PeriodicHarmonic", "Fourier"]
