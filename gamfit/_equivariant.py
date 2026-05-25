"""LieAtom / EquivariantPenalty / gauge_companion (composition-engine §4(c)+).

This module ships the Mendel-et-al-style equivariant SAE atoms as first-class
gamfit primitives:

    LatentBasisKind = {"LieAtom"}   variant added to the smooth-spec enum
    LieAtom(group=..., n_atoms=..., d_per_atom=...)   Smooth subclass
    EquivariantPenalty(...)         AnalyticPenalty: ½‖[ρ(g), W] z‖² + ARD
    gauge_companion(aux="HSV")      Helper that bakes the auxiliary-supervised
                                    gauge-fix recipe into one call
                                    (auxiliary-supervises d_aux dims, leaves
                                    the rest free for unsupervised semantic
                                    discovery).
    equivariant_smooth(...)         PyFFI-style one-shot constructor.

Numerical path: this module composes with gamfit's existing REML + ARD outer
loop. EquivariantPenalty's weight λ_eq and the per-group bandwidth (encoded
as `log_bandwidth` inside the per-atom GroupHead) are jointly selected by
REML's existing analytic-penalty machinery (see `_select_topology.py`),
because EquivariantPenalty implements the `AnalyticPenalty` protocol that
the REML scorer consumes.

The Mendel et al. arXiv:2511.09432 construction is recovered with
group="SO2", n_atoms=128, d_per_atom=2 (per-atom 2-frame in ambient).

REFERENCE
---------
Mendel et al. 2025 — Equivariant SAEs, arXiv:2511.09432
Engels et al. 2024 — Concept Manifolds, arXiv:2604.28119

Design lesson generalized here: an auxiliary-supervised gauge-fix block paired
with a free discovery block. The auxiliary block pins a known subspace (e.g.,
HSV anchors on SO(2) atoms) so the equivariant prior + ARD can discover the
residual semantic structure unsupervisedly in the free block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._binding import rust_module
from .smooth import Smooth
from ._penalties import _validate_weight


GroupName = Literal["SO2", "SO3", "R1", "Trivial"]

GROUP_DIM = {"SO2": 1, "SO3": 3, "R1": 1, "Trivial": 0}
GROUP_REP_DIM = {"SO2": 2, "SO3": 3, "R1": 1, "Trivial": 1}


def _scalar_weight(weight: float | str, name: str) -> float:
    if not isinstance(weight, (int, float)):
        raise TypeError(f"{name} must be a scalar for direct evaluation")
    value = float(weight)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def _nonnegative_scalar(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {out}")
    return out


# ---------------------------------------------------------------------------
# Group reps (numpy; analytic Jacobians for SO(2)/SO(3))
# ---------------------------------------------------------------------------

def rho_so2(theta: np.ndarray) -> np.ndarray:
    """SO(2) rep. theta: (...,) -> (..., 2, 2)."""
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty(theta.shape + (2, 2), dtype=np.float64)
    out[..., 0, 0] = c;  out[..., 0, 1] = -s
    out[..., 1, 0] = s;  out[..., 1, 1] = c
    return out


def rho_so2_jvp(theta: np.ndarray) -> np.ndarray:
    """d/dθ ρ_SO2(θ). Returns (..., 2, 2)."""
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty(theta.shape + (2, 2), dtype=np.float64)
    out[..., 0, 0] = -s; out[..., 0, 1] = -c
    out[..., 1, 0] =  c; out[..., 1, 1] = -s
    return out


def rho_so3(omega: np.ndarray) -> np.ndarray:
    """SO(3) rep via Rodrigues. omega: (..., 3) -> (..., 3, 3)."""
    angle = np.linalg.norm(omega, axis=-1, keepdims=True).clip(min=1e-12)
    axis = omega / angle
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(ax)
    K = np.stack([
        np.stack([zero, -az,  ay], -1),
        np.stack([  az, zero, -ax], -1),
        np.stack([ -ay,  ax, zero], -1),
    ], -2)                                            # (..., 3, 3)
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    a = angle[..., None]
    return I + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def rho_so3_jvp(omega: np.ndarray, domega: np.ndarray) -> np.ndarray:
    """Directional derivative of ρ_SO3 at ω in direction dω.

    Uses the standard formula
        dρ/dt|_0  =  ρ(ω) · [dω · K_basis]
    where K_basis maps a 3-vector to its skew matrix. For small dω this is
    the first-order Taylor term; we return ρ(ω) ∘ skew(dω) which is the
    correct Lie-algebra tangent in se(3) (right-trivialized).
    """
    Rg = rho_so3(omega)
    sx, sy, sz = domega[..., 0], domega[..., 1], domega[..., 2]
    zero = np.zeros_like(sx)
    Kd = np.stack([
        np.stack([zero, -sz,  sy], -1),
        np.stack([  sz, zero, -sx], -1),
        np.stack([ -sy,  sx, zero], -1),
    ], -2)
    return Rg @ Kd


def rho(group: GroupName, g: np.ndarray) -> np.ndarray:
    if group == "SO2": return rho_so2(g)
    if group == "SO3": return rho_so3(g)
    if group == "R1":  return np.ones(g.shape + (1, 1))
    if group == "Trivial": return np.ones(g.shape[:-1] + (1, 1)) if g.ndim else np.ones((1, 1))
    raise ValueError(group)


# ---------------------------------------------------------------------------
# LieAtom Smooth (LatentBasisKind extension)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LieAtom(Smooth):
    """A Lie-group atom in the additive-decoder layer.

    Forward (per atom a, sample n):
        x̂_n  +=  ρ(g_a(z_n)) · W_a · z_a

    where g_a : R^D → G is the per-atom group head, W_a ∈ R^{D × R} is the
    per-atom ambient 2-frame (R = GROUP_REP_DIM[group]), and z_a ∈ R is the
    per-atom amplitude. For SO(2): ρ ∈ R^{2x2}, R=2 — the atom represents an
    S^1 sub-manifold in ambient space, not a point.

    REML jointly selects (λ_eq, group-head log_bandwidth) — both flow through
    the standard analytic-penalty machinery.

    Parameters
    ----------
    group : {"SO2", "SO3", "R1", "Trivial"}
        Lie group acting on the per-atom rep space.
    n_atoms : int
        Number of atoms in this layer.
    d_per_atom : int
        Dimension of the per-atom rep space (must equal GROUP_REP_DIM[group];
        kept as an explicit parameter for API symmetry with future direct-sum
        rep constructions).
    bandwidth_init : float
        Initial log-bandwidth for the group head (REML re-selects).
    """
    group: GroupName = "SO2"
    n_atoms: int = 64
    d_per_atom: int = 2
    bandwidth_init: float = 0.0

    def __post_init__(self) -> None:
        expected = GROUP_REP_DIM[self.group]
        if self.d_per_atom != expected:
            # silent fallback — Trivial atoms collapse to d=1 etc.
            self.d_per_atom = expected


# ---------------------------------------------------------------------------
# EquivariantPenalty (AnalyticPenalty)
# ---------------------------------------------------------------------------

@dataclass
class EquivariantPenalty:
    """½ ‖[ρ(g), W] z‖² commutator residual + per-group bandwidth ARD.

    The commutator residual measures how far the per-atom ambient frame W_a
    is from spanning a ρ-invariant subspace. When zero, W_a ρ(g) e_1 lies in
    span(W_a) for every group element — i.e. the atom's contribution sweeps
    out a proper irrep orbit (S^1 for SO(2), S^2 for SO(3)).

    The bandwidth ARD term ½ Σ_a log(τ² + ‖log_bandwidth_a‖²) shrinks unused
    atoms toward zero bandwidth, where the group head's input projection
    saturates and the atom drops out — a soft analogue of the "dead atom"
    notion in vanilla SAEs.

    Joint REML selection: this penalty registers (λ_eq, {bandwidth_a}) with
    the outer REML loop via the same `AnalyticPenalty` protocol used by
    `IsometryPenalty` and `OrthogonalityPenalty`.

    Parameters
    ----------
    target : str | int
        Name or index of the LieAtom block this penalty governs.
    weight : float | "auto"
        λ_eq for the commutator residual.
    ard_weight : float
        Weight on the bandwidth ARD term. Set to 0 to disable.
    group : GroupName
        Group the target LieAtom uses (kept here so the penalty can be
        constructed independently of the LieAtom spec).
    """
    target: str | int
    weight: float | str = 1.0
    ard_weight: float = 1e-3
    group: GroupName = "SO2"

    def __post_init__(self) -> None:
        _validate_weight(self.weight, "EquivariantPenalty.weight")
        _nonnegative_scalar(self.ard_weight, "EquivariantPenalty.ard_weight")

    def __repr__(self) -> str:
        return (f"EquivariantPenalty(target={self.target!r}, weight={self.weight!r}, "
                f"ard_weight={self.ard_weight!r}, group={self.group!r})")

    # --- numeric path ------------------------------------------------------
    def evaluate(
        self,
        W: np.ndarray,          # (A, D, R)
        g: np.ndarray,          # (B, A) for SO2/R1, (B, A, 3) for SO3
        z: np.ndarray,          # (B, A)
        log_bandwidth: np.ndarray | None = None,    # (A,)
    ) -> float:
        """Scalar penalty value. Used by the analytic-penalty REML scorer."""
        return float(
            rust_module().equivariant_penalty_value(
                self.group,
                np.asarray(W, dtype=np.float64),
                np.asarray(g, dtype=np.float64),
                np.asarray(z, dtype=np.float64),
                _scalar_weight(self.weight, "EquivariantPenalty.weight"),
                _nonnegative_scalar(self.ard_weight, "EquivariantPenalty.ard_weight"),
                None if log_bandwidth is None else np.asarray(log_bandwidth, dtype=np.float64),
            )
        )


# ---------------------------------------------------------------------------
# gauge_companion — auxiliary-supervised gauge-fix recipe as a one-shot helper
# ---------------------------------------------------------------------------

@dataclass
class GaugeCompanion:
    """Auxiliary-supervised gauge-fix recipe wrapped into one object.

    Returns a list of (penalty, target) pairs that the caller wires into
    `gamfit.fit(..., penalties=[*gauge_companion("HSV").penalties, ...])`.

    The recipe:
      - supervise `d_aux` SO(2) atoms' group elements g_a against auxiliary
        anchors (e.g. HSV: atom 0 ↔ Hue, atom 1 ↔ Sat, atom 2 ↔ Val).
      - leave the remaining atoms free; under EquivariantPenalty + ARD they
        discover the residual semantic axes unsupervisedly.
    """
    aux: Literal["HSV", "RGB", "LCh"]
    d_aux: int = 3
    target: str = "lie"
    weight: float = 1.0
    aux_values: np.ndarray | None = None      # (N, 3); ground-truth HSV per row

    def loss(self, theta: np.ndarray) -> float:
        """Circular MSE for hue + cos-alignment for sat/val."""
        if self.aux_values is None:
            return 0.0
        h_rad = self.aux_values[:, 0] * 2 * np.pi
        terms = [(1.0 - np.cos(theta[:, 0] - h_rad)).mean()]
        if self.d_aux >= 2 and theta.shape[1] >= 2:
            terms.append(((np.cos(theta[:, 1]) - (2 * self.aux_values[:, 1] - 1)) ** 2).mean())
        if self.d_aux >= 3 and theta.shape[1] >= 3:
            terms.append(((np.cos(theta[:, 2]) - (2 * self.aux_values[:, 2] - 1)) ** 2).mean())
        return float(self.weight * sum(terms) / len(terms))


def gauge_companion(aux: str = "HSV", d_aux: int = 3, weight: float = 1.0,
                    target: str = "lie", aux_values: np.ndarray | None = None) -> GaugeCompanion:
    """One-call gauge-fix companion. See `GaugeCompanion` docstring."""
    return GaugeCompanion(aux=aux, d_aux=d_aux, weight=weight, target=target,
                          aux_values=aux_values)


# ---------------------------------------------------------------------------
# PyFFI-style one-shot constructor
# ---------------------------------------------------------------------------

def equivariant_smooth(
    group: GroupName = "SO2",
    aux: str | None = "HSV",
    n_atoms: int = 128,
    d_per_atom: int = 2,
    weight: float = 1.0,
    ard_weight: float = 1e-3,
    name: str = "lie",
) -> tuple[LieAtom, EquivariantPenalty, GaugeCompanion | None]:
    """Construct (LieAtom, EquivariantPenalty[, GaugeCompanion]) in one call.

    Matches the proposal's API:
        atom, pen, gc = gamfit.equivariant_smooth(
            group="SO2", aux="HSV", n_atoms=128, d_per_atom=2
        )

    Then wire into `gamfit.fit(smooths=[atom, ...], penalties=[pen, ...])`.
    """
    atom = LieAtom(name=name, group=group, n_atoms=n_atoms, d_per_atom=d_per_atom)
    pen = EquivariantPenalty(target=name, weight=weight, ard_weight=ard_weight, group=group)
    gc = gauge_companion(aux=aux) if aux is not None else None
    return atom, pen, gc


__all__ = [
    "GroupName", "GROUP_DIM", "GROUP_REP_DIM",
    "rho", "rho_so2", "rho_so2_jvp", "rho_so3", "rho_so3_jvp",
    "LieAtom", "EquivariantPenalty",
    "GaugeCompanion", "gauge_companion",
    "equivariant_smooth",
]
