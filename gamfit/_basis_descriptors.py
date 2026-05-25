"""Callable basis descriptors implementing :class:`BasisDescriptor`.

These wrap the torch basis primitives in :mod:`gamfit.torch._basis` (and add
a torch-native periodic harmonic basis) so every descriptor satisfies
``descriptor(t) -> torch.Tensor`` with grad through ``t``.

Lightweight: torch is imported only when ``evaluate`` is called.
"""

from __future__ import annotations

import math
from typing import Any

from ._protocol import BasisDescriptor, _require_torch


class PeriodicHarmonic(BasisDescriptor):
    """Trigonometric (Fourier) basis on the unit circle.

    With ``harmonics=H`` the basis is

        ``[1, cos(θ), sin(θ), cos(2θ), sin(2θ), …, cos(Hθ), sin(Hθ)]``

    of width ``2H + 1``. ``theta`` is interpreted in radians; this matches
    the periodic spline curve basis on ``[0, 2π]``.

    Available aliases at module-load time: :class:`Fourier`.
    """

    def __init__(self, harmonics: int = 3, *, num_basis: int | None = None) -> None:
        if num_basis is not None:
            # Allow construction by total basis width: num_basis must be odd
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
        torch = _require_torch()
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float64)
        if not torch.is_floating_point(t):
            t = t.to(torch.float64)
        # Accept (N,), (N, 1), or 1-D scalar shapes
        if t.dim() == 2 and t.shape[1] == 1:
            theta = t.squeeze(-1)
        elif t.dim() == 1:
            theta = t
        else:
            raise ValueError(
                f"PeriodicHarmonic.evaluate expects a 1-D tensor or (N, 1), "
                f"got shape {tuple(t.shape)}"
            )
        cols = [torch.ones_like(theta)]
        for h in range(1, self.harmonics + 1):
            cols.append(torch.cos(h * theta))
            cols.append(torch.sin(h * theta))
        return torch.stack(cols, dim=-1)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "periodic_harmonic", "harmonics": self.harmonics}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PeriodicHarmonic":
        return cls(harmonics=int(d.get("harmonics", 3)))

    def __repr__(self) -> str:
        return f"PeriodicHarmonic(harmonics={self.harmonics})"


# Common alias requested by the protocol spec.
Fourier = PeriodicHarmonic


__all__ = ["PeriodicHarmonic", "Fourier"]
