"""LieAtom / EquivariantPenalty / gauge_companion — thin FFI shims.

The numerical work (SO(2)/SO(3) representations, their JVPs, commutator
penalty, gauge-companion HSV/RGB/LCh loss) lives in `gam-pyffi`. This module
hosts the dataclass surface that REML's analytic-penalty machinery consumes,
plus a one-shot `equivariant_smooth` constructor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from ._binding import rust_module
from .smooth import Smooth
from ._penalties import _validate_weight, ScalarWeightSchedule


GroupName = Literal["SO2", "SO3", "R1", "Trivial"]
AuxName = Literal["HSV", "RGB", "LCh"]

GROUP_DIM = {"SO2": 1, "SO3": 3, "R1": 1, "Trivial": 0}
GROUP_REP_DIM = {"SO2": 2, "SO3": 3, "R1": 1, "Trivial": 1}


def _scalar_weight(weight: float | ScalarWeightSchedule, name: str) -> float:
    # allow-list (a): FFI input validation.
    if not isinstance(weight, (int, float)):
        raise TypeError(f"{name} must be a scalar for direct evaluation")
    value = float(weight)
    # allow-list (a): FFI input validation.
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def _nonnegative_scalar(value: float, name: str) -> float:
    out = float(value)
    # allow-list (a): FFI input validation.
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {out}")
    return out


# ---------------------------------------------------------------------------
# Group representations and JVPs (thin FFI shims).
# ---------------------------------------------------------------------------

def rho_so2(theta: np.ndarray) -> np.ndarray:
    """SO(2) rep. theta: (...,) -> (..., 2, 2)."""
    arr = np.asarray(theta, dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = rust_module().equivariant_rho_so2(flat)
    return out.reshape(arr.shape + (2, 2))


def rho_so2_jvp(theta: np.ndarray) -> np.ndarray:
    """d/dθ ρ_SO2(θ). Returns (..., 2, 2)."""
    arr = np.asarray(theta, dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = rust_module().equivariant_rho_so2_jvp(flat)
    return out.reshape(arr.shape + (2, 2))


def rho_so3(omega: np.ndarray) -> np.ndarray:
    """SO(3) rep via Rodrigues. omega: (..., 3) -> (..., 3, 3)."""
    arr = np.asarray(omega, dtype=np.float64)
    # allow-list (a): FFI input validation.
    if arr.shape[-1] != 3:
        raise ValueError("rho_so3 requires last axis of length 3")
    flat = np.ascontiguousarray(arr.reshape(-1, 3))
    out = rust_module().equivariant_rho_so3(flat)
    return out.reshape(arr.shape[:-1] + (3, 3))


def rho_so3_jvp(omega: np.ndarray, domega: np.ndarray) -> np.ndarray:
    """Directional derivative of ρ_SO3 at ω in direction dω."""
    arr = np.asarray(omega, dtype=np.float64)
    darr = np.asarray(domega, dtype=np.float64)
    # allow-list (a): FFI input validation.
    if arr.shape[-1] != 3 or darr.shape[-1] != 3:
        raise ValueError("rho_so3_jvp requires last axis of length 3 on both inputs")
    bd = np.broadcast_to(darr, arr.shape)
    flat_o = np.ascontiguousarray(arr.reshape(-1, 3))
    flat_d = np.ascontiguousarray(bd.reshape(-1, 3))
    out = rust_module().equivariant_rho_so3_jvp(flat_o, flat_d)
    return out.reshape(arr.shape[:-1] + (3, 3))


def rho(group: GroupName, g: np.ndarray) -> np.ndarray:
    arr = np.asarray(g, dtype=np.float64)
    return rust_module().equivariant_rho(group, np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# LieAtom Smooth (LatentBasisKind extension)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LieAtom(Smooth):
    """A Lie-group atom in the additive-decoder layer.

    Forward (per atom a, sample n):
        x̂_n  +=  ρ(g_a(z_n)) · W_a · z_a

    REML jointly selects (λ_eq, group-head log_bandwidth).
    """
    group: GroupName = "SO2"
    n_atoms: int = 64
    d_per_atom: int = 2
    bandwidth_init: float = 0.0

    # LieAtom is a config carrier for the additive Lie-decoder layer; it
    # does not carry its own basis-evaluator surface (the Rust decoder
    # consumes the dataclass directly). Empty set is the honest contract.
    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        expected = GROUP_REP_DIM[self.group]
        # allow-list (e): dataclass typed config normalization.
        if self.d_per_atom != expected:
            self.d_per_atom = expected


# ---------------------------------------------------------------------------
# EquivariantPenalty (AnalyticPenalty)
# ---------------------------------------------------------------------------

@dataclass
class EquivariantPenalty:
    """½ ‖[ρ(g), W] z‖² commutator residual + per-group bandwidth ARD."""
    target: str | int
    weight: float | ScalarWeightSchedule = 1.0
    ard_weight: float = 1e-3
    group: GroupName = "SO2"
    _weight_schedule: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_weight(self.weight, "EquivariantPenalty.weight")
        _nonnegative_scalar(self.ard_weight, "EquivariantPenalty.ard_weight")

    def __repr__(self) -> str:
        return (f"EquivariantPenalty(target={self.target!r}, weight={self.weight!r}, "
                f"ard_weight={self.ard_weight!r}, group={self.group!r})")

    def evaluate(
        self,
        W: np.ndarray,
        g: np.ndarray,
        z: np.ndarray,
        log_bandwidth: np.ndarray | None = None,
    ) -> float:
        return float(
            rust_module().equivariant_penalty_value(
                self.group,
                np.asarray(W, dtype=np.float64),
                np.asarray(g, dtype=np.float64),
                np.asarray(z, dtype=np.float64),
                _scalar_weight(self.weight, "EquivariantPenalty.weight"),
                _nonnegative_scalar(self.ard_weight, "EquivariantPenalty.ard_weight"),
                # allow-list (a): FFI input marshaling.
                None if log_bandwidth is None else np.asarray(log_bandwidth, dtype=np.float64),
            )
        )


# ---------------------------------------------------------------------------
# gauge_companion — auxiliary-supervised gauge-fix recipe as a one-shot helper
# ---------------------------------------------------------------------------

@dataclass
class GaugeCompanion:
    """Auxiliary-supervised gauge-fix recipe wrapped into one object."""
    aux: AuxName
    d_aux: int = 3
    target: str = "lie"
    weight: float = 1.0
    aux_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        # allow-list (e): dataclass typed config normalization.
        if self.aux_values is not None:
            self.aux_values = np.ascontiguousarray(np.asarray(self.aux_values, dtype=np.float64))

    def loss(self, theta: np.ndarray) -> float:
        """Hue circular MSE + sat/val cos-alignment."""
        return float(
            rust_module().equivariant_gauge_companion_loss(
                self.aux_values,
                np.ascontiguousarray(np.asarray(theta, dtype=np.float64)),
                int(self.d_aux),
                float(self.weight),
            )
        )


def gauge_companion(aux: AuxName = "HSV", d_aux: int = 3, weight: float = 1.0,
                    target: str = "lie", aux_values: np.ndarray | None = None) -> GaugeCompanion:
    """One-call gauge-fix companion. See `GaugeCompanion` docstring."""
    return GaugeCompanion(aux=aux, d_aux=d_aux, weight=weight, target=target,
                          aux_values=aux_values)


def equivariant_smooth(
    group: GroupName = "SO2",
    aux: AuxName | None = "HSV",
    n_atoms: int = 128,
    d_per_atom: int = 2,
    weight: float = 1.0,
    ard_weight: float = 1e-3,
    name: str = "lie",
) -> tuple[LieAtom, EquivariantPenalty, GaugeCompanion | None]:
    """Construct (LieAtom, EquivariantPenalty[, GaugeCompanion]) in one call."""
    atom = LieAtom(name=name, group=group, n_atoms=n_atoms, d_per_atom=d_per_atom)
    pen = EquivariantPenalty(target=name, weight=weight, ard_weight=ard_weight, group=group)
    gc = (
        lambda: None,
        lambda: gauge_companion(aux=aux),
    )[rust_module().equivariant_aux_enabled(aux)]()
    return atom, pen, gc


__all__ = [
    "GroupName", "GROUP_DIM", "GROUP_REP_DIM",
    "rho", "rho_so2", "rho_so2_jvp", "rho_so3", "rho_so3_jvp",
    "LieAtom", "EquivariantPenalty",
    "GaugeCompanion", "gauge_companion",
    "equivariant_smooth",
]
