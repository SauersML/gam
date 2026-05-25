"""Hyperbolic atom dictionary in the Poincaré ball.

:class:`PoincareAtoms` is a torch ``nn.Module`` that holds ``F`` learnable
points in the Poincaré ball ``B^d_c = {x in R^d : c|x|^2 < 1}`` and decodes
a gate vector ``z in R^F`` to a single ball point. With curvature ``c = -1``
(the default) the manifold is the open unit ball of constant sectional
curvature ``-1``.

Mixing convention
-----------------
Decoding uses **tangent-space aggregation at the origin followed by the
exponential map**:

    v       = sum_f z_f * log_0(a_f)                       (in T_0 B = R^d)
    x_hat   = exp_0(v)

with the Poincaré-ball logarithm and exponential at the origin

    log_0(y)   = artanh(sqrt(-c) |y|) / (sqrt(-c) |y|) * y
    exp_0(v)   = tanh(sqrt(-c) |v|)  / (sqrt(-c) |v|)  * v.

This convention is:

* **Closed-form and fully differentiable** — no inner optimisation, every
  step is a smooth elementary function of the atom positions and gates,
  so ``backward`` flows through both atoms and gates.
* **Euclidean-limit consistent** — as ``c -> 0`` the tangent maps collapse
  to identity and the decoder reduces to ordinary linear mixing
  ``x_hat = sum_f z_f * a_f``, matching the standard atom-dictionary
  decoder.
* **Origin-equivariant** — Möbius rotations fixing the origin commute with
  the construction; this is the property a "centred" hyperbolic decoder
  should have.

An alternative weighted Möbius / Karcher mean is *implicit* (defined as the
minimiser of weighted squared hyperbolic distance) and would require an
inner solver inside ``forward``; we intentionally avoid that. The tangent
construction is the standard choice in the hyperbolic neural-network
literature (cf. Ganea, Bécigneul, Hofmann 2018 and Hyperbolic-Mamba,
arXiv:2505.18973).

Lorentz model
-------------
When ``lorentz=True`` the same mixing convention is implemented on the
hyperboloid model ``H^d_c = {x in R^{d+1} : -x_0^2 + |x_{1:}|^2 = 1/c,
x_0 > 0}`` for ``c < 0``. The hyperboloid arithmetic avoids the
``1 - |x|^2`` denominator that diverges at the Poincaré boundary, so it is
the recommended path for atoms that drift near the boundary. The two paths
are isometric (one is a stereographic projection of the other from
``(-1/sqrt(-c), 0, ..., 0)``) and agree on small inputs to within
``1e-5`` — there is a test that pins this.

References
----------
* Nickel, M., Kiela, D. *Poincaré Embeddings for Learning Hierarchical
  Representations.* NeurIPS 2017. arXiv:1705.08039.
* Ganea, O., Bécigneul, G., Hofmann, T. *Hyperbolic Neural Networks.*
  NeurIPS 2018. arXiv:1805.09112.
* Hyperbolic-Mamba: state-space hyperbolic sequence model, arXiv:2505.18973
  (2025).
"""

from __future__ import annotations

import math

import torch
from torch import nn


# Numerical guards. These are deliberately small but non-trivial — they
# matter for points within ~1e-3 of the ball boundary, which is exactly
# where the Poincaré-ball formulas hurt.
_EPS = 1e-15
_BOUNDARY_EPS = 1e-5


def _safe_sqrt_neg_curvature(curvature: float) -> float:
    """Return ``sqrt(-c)`` after checking ``c < 0``."""

    if curvature >= 0.0:
        raise ValueError(
            "PoincareAtoms requires negative curvature (c < 0); "
            f"got curvature={curvature!r}"
        )
    return math.sqrt(-curvature)


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm along the last axis, guarded against zero."""

    return torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(_EPS)


def _project_into_ball(x: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Project ``x`` so that ``sqrt(-c) |x| <= 1 - boundary_eps``."""

    max_norm = (1.0 - _BOUNDARY_EPS) / sqrt_negc
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norm.clamp_min(_EPS), max=1.0)
    return x * scale


def _mobius_add(u: torch.Tensor, v: torch.Tensor, curvature: float) -> torch.Tensor:
    """Möbius addition ``u oplus_c v`` in the Poincaré ball for c < 0.

    Using the sign convention from Ganea et al. (2018) with ``c > 0`` denoting
    *negated* curvature, the formula they state is

        u oplus v = ((1 + 2 c <u,v> + c |v|^2) u + (1 - c |u|^2) v)
                    / (1 + 2 c <u,v> + c^2 |u|^2 |v|^2).

    Here we adopt the convention where ``curvature`` is the actual (negative)
    sectional curvature. Substituting ``c -> -curvature`` into Ganea's
    expression gives, with ``k := -curvature > 0``:

        u oplus v = ((1 + 2 k <u,v> + k |v|^2) u + (1 - k |u|^2) v)
                    / (1 + 2 k <u,v> + k^2 |u|^2 |v|^2).
    """

    k = -curvature
    uv = (u * v).sum(dim=-1, keepdim=True)
    uu = (u * u).sum(dim=-1, keepdim=True)
    vv = (v * v).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * k * uv + k * vv) * u + (1.0 - k * uu) * v
    den = (1.0 + 2.0 * k * uv + (k * k) * uu * vv).clamp_min(_EPS)
    return num / den


def _log0(y: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Poincaré logarithm at the origin: ``log_0(y)``."""

    norm = _l2_norm(y)
    # artanh is bounded by clamping the argument away from 1.
    arg = (sqrt_negc * norm).clamp(max=1.0 - _BOUNDARY_EPS)
    coeff = torch.atanh(arg) / (sqrt_negc * norm)
    return coeff * y


def _exp0(v: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Poincaré exponential at the origin: ``exp_0(v)``."""

    norm = _l2_norm(v)
    coeff = torch.tanh(sqrt_negc * norm) / (sqrt_negc * norm)
    return coeff * v


def _poincare_distance(
    a: torch.Tensor, b: torch.Tensor, curvature: float, sqrt_negc: float
) -> torch.Tensor:
    """Closed-form Poincaré-ball geodesic distance for c < 0."""

    diff_sq = ((a - b) * (a - b)).sum(dim=-1)
    a_sq = (a * a).sum(dim=-1)
    b_sq = (b * b).sum(dim=-1)
    # Denominators (1 + c |a|^2)(1 + c |b|^2) — strictly positive inside
    # the ball because c|x|^2 < 1 (and c < 0).
    denom = (1.0 + curvature * a_sq).clamp_min(_EPS) * (
        1.0 + curvature * b_sq
    ).clamp_min(_EPS)
    arg = 1.0 + 2.0 * (-curvature) * diff_sq / denom
    arg = arg.clamp_min(1.0 + _EPS)
    return torch.acosh(arg) / sqrt_negc


# ---------------------------------------------------------------------------
# Lorentz (hyperboloid) helpers
# ---------------------------------------------------------------------------


def _to_lorentz(x: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Stereographic projection Poincaré-ball -> hyperboloid.

    For a point ``y in B^d`` with ``|y| < 1`` the corresponding point on the
    hyperboloid ``{x : -x_0^2 + |x_s|^2 = -1/k, x_0 > 0}`` (with ``k = -c``)
    is ``(x_0, x_s) = (1 + k|y|^2, 2 sqrt(k) y) / (k (1 - k|y|^2)) * ...``.

    A cleaner equivalent form: starting from the unit-curvature mapping

        x_0 = (1 + |y|^2) / (1 - |y|^2),   x_s = 2 y / (1 - |y|^2),

    and then rescaling by ``1/sqrt(k)`` so that ``-x_0^2 + |x_s|^2 = -1/k``.
    """

    y_sq = (x * x).sum(dim=-1, keepdim=True)
    denom = (1.0 - y_sq).clamp_min(_EPS)
    x0 = (1.0 + y_sq) / denom
    xs = 2.0 * x / denom
    # Rescale to curvature c = -k.
    return torch.cat([x0, xs], dim=-1) / sqrt_negc


def _from_lorentz(x: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Inverse stereographic projection hyperboloid -> Poincaré ball."""

    # Undo the 1/sqrt(k) rescale.
    x = x * sqrt_negc
    x0 = x[..., :1]
    xs = x[..., 1:]
    return xs / (x0 + 1.0).clamp_min(_EPS)


def _lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product with signature ``(-,+,+,...,+)``."""

    prod = u * v
    inner = -prod[..., :1] + prod[..., 1:].sum(dim=-1, keepdim=True)
    return inner


def _lorentz_log0(y: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Lorentz logarithm at the origin ``o = (1/sqrt(k), 0, ..., 0)``.

    Returns a tangent vector ``v`` with ``v_0 = 0`` (in the canonical chart
    used by the rest of this module), then rescaled into Euclidean
    coordinates by dropping the leading zero.
    """

    k = sqrt_negc * sqrt_negc
    # <o, y>_L = -y_0 / sqrt(k) (because o = (1/sqrt(k), 0, ...)).
    inner = -y[..., :1] / sqrt_negc  # shape (..., 1)
    # acosh argument: -k <o, y>_L = sqrt(k) y_0. Clamp away from 1.
    arg = (-k * inner).clamp_min(1.0 + _EPS)
    dist = torch.acosh(arg) / sqrt_negc  # geodesic distance on H^d
    # Project y onto T_o by removing the o-component: y_proj = y - (k <o,y>) o.
    # In coordinates with o = (1/sqrt(k), 0, ...): y_proj_0 = 0, y_proj_s = y_s.
    y_s = y[..., 1:]
    norm_proj = torch.linalg.vector_norm(y_s, dim=-1, keepdim=True).clamp_min(_EPS)
    # Tangent vector lives in span perpendicular to o; we return its spatial
    # part (the time component is zero by construction).
    return (dist * y_s) / norm_proj


def _lorentz_exp0(v_spatial: torch.Tensor, sqrt_negc: float) -> torch.Tensor:
    """Lorentz exponential at the origin from a spatial tangent vector."""

    k = sqrt_negc * sqrt_negc
    norm = torch.linalg.vector_norm(v_spatial, dim=-1, keepdim=True).clamp_min(_EPS)
    # |v|_L = norm (because time component is zero) — geodesic param s = sqrt(k)*norm.
    s = sqrt_negc * norm
    x0 = torch.cosh(s) / sqrt_negc
    xs = (torch.sinh(s) / s) * v_spatial
    return torch.cat([x0, xs], dim=-1)


# ---------------------------------------------------------------------------
# Public module
# ---------------------------------------------------------------------------


class PoincareAtoms(nn.Module):
    """A learnable dictionary of ``F`` atoms in the Poincaré ball ``B^d_c``.

    Parameters
    ----------
    F:
        Number of atoms in the dictionary.
    ball_dim:
        Ambient (Euclidean-coordinate) dimension ``d`` of the ball.
    curvature:
        Sectional curvature ``c``. Must be strictly negative.
    lorentz:
        If ``True``, compute the decoder mixing on the hyperboloid model
        rather than the Poincaré-ball model. The two paths are isometric;
        the Lorentz path has no boundary singularity and is preferred when
        atoms may approach ``|a| -> 1/sqrt(-c)``.
    init_scale:
        Standard deviation of the Gaussian initialiser for atom positions
        (in Euclidean coordinates). Atoms are projected into the ball after
        sampling, so even large values are safe.
    device, dtype:
        Standard torch placement arguments.

    Examples
    --------
    >>> import torch
    >>> from gamfit.torch.hyperbolic import PoincareAtoms
    >>> atoms = PoincareAtoms(F=4, ball_dim=3)
    >>> z = torch.randn(2, 4)
    >>> x_hat = atoms(z)
    >>> x_hat.shape
    torch.Size([2, 3])
    """

    def __init__(
        self,
        F: int,
        ball_dim: int,
        curvature: float = -1.0,
        *,
        lorentz: bool = False,
        init_scale: float = 0.01,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(F, int) or F <= 0:
            raise ValueError(f"F must be a positive int, got {F!r}")
        if not isinstance(ball_dim, int) or ball_dim <= 0:
            raise ValueError(f"ball_dim must be a positive int, got {ball_dim!r}")
        if not isinstance(curvature, (int, float)):
            raise TypeError("curvature must be a real number")
        curvature = float(curvature)
        # Validate by computing sqrt(-c); this raises if c >= 0.
        self._sqrt_negc = _safe_sqrt_neg_curvature(curvature)
        if init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")
        if dtype is None:
            dtype = torch.get_default_dtype()

        self.F = F
        self.ball_dim = ball_dim
        self.curvature = curvature
        self.lorentz = bool(lorentz)
        self.init_scale = init_scale

        atoms = torch.randn(F, ball_dim, device=device, dtype=dtype) * init_scale
        atoms = _project_into_ball(atoms, self._sqrt_negc)
        self.atoms = nn.Parameter(atoms)

    # ------------------------------------------------------------------ utils

    def project_into_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` so it lies strictly inside the ball.

        Useful as a no-op-when-safe sanitiser after an external gradient
        step. ``x`` is returned unchanged when already strictly inside.
        """

        return _project_into_ball(x, self._sqrt_negc)

    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Möbius addition ``u oplus_c v`` for points in the ball.

        Both inputs must broadcast against each other along the last axis.
        """

        return _mobius_add(u, v, self.curvature)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance ``d_c(a, b)`` for points in the ball.

        Always evaluated through the Poincaré formula because the Lorentz
        formula's only practical advantage is for points within ~1e-15 of
        the boundary, well below the precision of float64 anyway. The
        denominator ``(1 + c |a|^2)(1 + c |b|^2)`` is clamped to avoid 0/0
        if a numerical drift puts a point exactly on the boundary.
        """

        if a.shape[-1] != self.ball_dim or b.shape[-1] != self.ball_dim:
            raise ValueError(
                f"distance expects last dim = {self.ball_dim}; "
                f"got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        return _poincare_distance(a, b, self.curvature, self._sqrt_negc)

    # ------------------------------------------------------------------ forward

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode gate vector ``z`` into a single ball point.

        ``z`` has shape ``(..., F)`` and the returned tensor has shape
        ``(..., ball_dim)`` — exactly the broadcasting contract of a linear
        decoder. The decoder collapses to ``z @ atoms`` in the Euclidean
        limit ``c -> 0``.
        """

        if z.shape[-1] != self.F:
            raise ValueError(
                f"PoincareAtoms.forward expects z.shape[-1] = {self.F}; "
                f"got {tuple(z.shape)}"
            )

        atoms = self.atoms.to(dtype=z.dtype, device=z.device)
        # Numerical safety: keep atoms strictly inside the ball every step.
        atoms = _project_into_ball(atoms, self._sqrt_negc)

        if self.lorentz:
            # Map atoms through stereographic projection, take log at the
            # canonical origin, mix linearly, and map back.
            atoms_h = _to_lorentz(atoms, self._sqrt_negc)  # (F, d+1)
            tangents = _lorentz_log0(atoms_h, self._sqrt_negc)  # (F, d) spatial part
            # Weighted sum: z (..., F) @ tangents (F, d) -> (..., d).
            v = z @ tangents
            x_h = _lorentz_exp0(v, self._sqrt_negc)  # (..., d+1)
            return _from_lorentz(x_h, self._sqrt_negc)

        # Poincaré path.
        tangents = _log0(atoms, self._sqrt_negc)  # (F, d) in T_0 = R^d
        v = z @ tangents
        return _exp0(v, self._sqrt_negc)


__all__ = ["PoincareAtoms"]
