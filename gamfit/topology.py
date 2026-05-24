"""Ergonomic one-line basis wrappers for common manifold topologies.

These are *thin* wrappers around the existing :mod:`gamfit.smooth` primitives
(``Duchon``, ``Sphere``, ``PeriodicSplineCurve``). The point is to let users
compare candidate topologies with one line per candidate instead of manually
constructing tensor products or remembering the right ``periodic_per_axis``
flag::

    import gamfit
    from gamfit import topology

    # candidates for an ambiguous latent coordinate
    candidates = [
        topology.Circle(name="circle"),
        topology.Cylinder(name="cylinder"),
        topology.Torus(name="torus"),
        topology.Sphere(name="sphere"),
        topology.EuclideanPatch(d=2, name="patch"),
    ]
    # then hand to gamfit.fit / a model-comparison helper

Each wrapper returns a :class:`~gamfit.smooth.Smooth` instance and so slots
directly into ``gamfit.fit(..., smooths=[...])``.

Design choices
--------------
* All wrappers accept the standard ``name``, ``by``, ``double_penalty``,
  ``shape_constraint`` kwargs for API consistency with other ``Smooth`` specs.
* Defaults mirror the underlying primitives (``n_knots=20`` for 1D curves;
  ``n_centers=64`` for Duchon patches; ``8`` per axis for 2D tensor / radial).
* ``Cylinder`` / ``Torus`` are explicit tensor B-spline candidates, matching
  the formula path used by topology selection.
* For d >= 2 the underlying Duchon basis requires explicit ``centers``; the
  ``centers`` kwarg is forwarded so users can pass them. When omitted the
  wrappers pass ``n_centers`` (an ``int``) through to the spec, which the
  formula path treats as a request to auto-derive — auto-derivation is
  presently only implemented for ``d=1`` in the Rust core. See the per-class
  "Admissibility" notes below.
"""

from __future__ import annotations

from typing import Any

from .smooth import (
    BSpline,
    Duchon,
    PeriodicSplineCurve,
    ShapeConstraintLiteral,
    Smooth,
    Sphere as _SphereSmooth,
    TensorBSpline,
)


__all__ = [
    "Circle",
    "Cylinder",
    "Torus",
    "Sphere",
    "EuclideanPatch",
]


def _common(
    name: str | None,
    by: Any | None,
    double_penalty: bool,
    shape_constraint: ShapeConstraintLiteral | None,
) -> dict[str, Any]:
    """Bundle the four Smooth-base kwargs into a dict for forwarding."""
    return {
        "name": name,
        "by": by,
        "double_penalty": double_penalty,
        "shape_constraint": shape_constraint,
    }


def _tensor_topology(
    name: str | None,
    n_knots: tuple[int, int],
    periodic: tuple[bool, bool],
    by: Any | None,
    double_penalty: bool,
    shape_constraint: ShapeConstraintLiteral | None,
) -> Smooth:
    spec = TensorBSpline(
        marginals=[BSpline(periodic=value) for value in periodic],
        **_common(name, by, double_penalty, shape_constraint),
    )
    setattr(spec, "_gamfit_topology_dim", 2)
    setattr(spec, "_gamfit_tensor_k", tuple(int(value) for value in n_knots))
    setattr(spec, "_gamfit_tensor_periods", tuple("2*pi" if value else None for value in periodic))
    setattr(spec, "_gamfit_tensor_identifiability", "sum_tozero")
    return spec


def Circle(
    name: str | None = None,
    n_knots: int = 20,
    *,
    degree: int = 3,
    penalty_order: int = 2,
    output_dim: int = 1,
    by: Any | None = None,
    double_penalty: bool = False,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Closed loop, S^1.

    Pick this when one coordinate is angularly periodic (e.g. hue, time of
    day, phase of an oscillation). The input is a scalar parameter ``t`` and
    the basis is the cyclic cubic spline on a periodic knot grid.

    Implemented as :class:`~gamfit.smooth.PeriodicSplineCurve` with
    ``output_dim=1``.

    Parameters
    ----------
    n_knots : number of basis knots along the periodic parameter.
    degree, penalty_order : forwarded to :class:`PeriodicSplineCurve`.
    output_dim : ambient dimension of the curve (default ``1``: a scalar
        smooth on the circle). Set ``>1`` to fit a closed loop in R^d.
    """
    return PeriodicSplineCurve(
        n_knots=int(n_knots),
        degree=int(degree),
        output_dim=int(output_dim),
        penalty_order=int(penalty_order),
        **_common(name, by, double_penalty, shape_constraint),
    )


def Cylinder(
    name: str | None = None,
    n_knots: tuple[int, int] = (20, 8),
    *,
    by: Any | None = None,
    double_penalty: bool = False,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Cylinder, S^1 x R.

    Pick this when one axis is periodic and one is not (e.g. hue x
    lightness, time-of-day x temperature). The input is ``(N, 2)`` with
    the first column the periodic axis.

    Implemented as a :class:`~gamfit.smooth.TensorBSpline` with a periodic
    first margin and a nonperiodic second margin.

    Parameters
    ----------
    n_knots : ``(n_periodic, n_nonperiodic)`` tensor basis sizes.
    """
    return _tensor_topology(name, n_knots, (True, False), by, double_penalty, shape_constraint)


def Torus(
    name: str | None = None,
    n_knots: tuple[int, int] = (20, 20),
    *,
    by: Any | None = None,
    double_penalty: bool = False,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Torus, S^1 x S^1.

    Pick this when both axes are angularly periodic (e.g. hue x phase,
    longitude x time-of-day for a daily-cycle equirectangular field). The
    input is ``(N, 2)`` with both columns periodic.

    Implemented as a :class:`~gamfit.smooth.TensorBSpline` with both margins
    periodic.

    Parameters
    ----------
    n_knots : ``(n_axis_0, n_axis_1)`` tensor basis sizes.
    """
    return _tensor_topology(name, n_knots, (True, True), by, double_penalty, shape_constraint)


def Sphere(
    name: str | None = None,
    n_knots: int = 20,
    *,
    dim: int = 2,
    penalty_order: int = 2,
    kernel: str = "sobolev",
    radians: bool = False,
    by: Any | None = None,
    double_penalty: bool = False,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Two-sphere, S^2.

    Pick this when the predictor is a direction in 3D — geographic
    (latitude, longitude), a unit-vector embedding, or any orientation
    coordinate that lives on the sphere. The input is ``(N, 2)``:
    (latitude, longitude), degrees by default.

    Implemented as :class:`~gamfit.smooth.Sphere` (spherical-harmonic /
    Sobolev kernel basis on S^2).

    Parameters
    ----------
    dim : coordinate dimension. Must be ``2`` for latitude/longitude.
    n_knots : number of basis centers / harmonic truncation degree
        (forwarded as ``n_centers`` to the underlying spec).
    penalty_order, kernel, radians : forwarded to
        :class:`~gamfit.smooth.Sphere`.
    """
    if int(dim) != 2:
        raise ValueError("topology.Sphere supports dim=2 latitude/longitude inputs")
    spec = _SphereSmooth(
        n_centers=int(n_knots),
        penalty_order=int(penalty_order),
        kernel=str(kernel),
        radians=bool(radians),
        **_common(name, by, double_penalty, shape_constraint),
    )
    setattr(spec, "_gamfit_topology_dim", 2)
    return spec


def EuclideanPatch(
    d: int = 2,
    name: str | None = None,
    n_centers: int = 64,
    *,
    centers: Any | None = None,
    m: int = 2,
    length_scale: float | None = None,
    by: Any | None = None,
    double_penalty: bool = False,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Open Euclidean patch in R^d (no periodicity).

    Pick this as the "no special topology" baseline against which closed
    candidates (Circle, Cylinder, Torus, Sphere) should be compared. The
    input is ``(N, d)``; the basis is the standard isotropic Duchon
    m-spline.

    Parameters
    ----------
    d : ambient dimension. ``d=2`` is the thin-plate spline. ``d>=3`` is
        the generalized Duchon spline.
    n_centers : default ``64``. Used as the auto-center count only when
        ``d == 1`` (Rust auto-derivation is d=1 only) — otherwise pass
        ``centers`` explicitly.
    centers : optional ``(K, d)`` array of control points.

    Admissibility
    -------------
    For ``d >= 2`` the user must pass explicit ``centers`` because the
    Rust core does not yet auto-derive multi-d centers. A uniform grid
    over the data's bounding box is the standard choice. This is a
    *maintainer-review item* — adding multi-d auto-centers in Rust would
    let the wrapper be a true one-liner for d>=2.
    """
    if centers is None and d == 1:
        centers_arg: Any = int(n_centers)
    else:
        centers_arg = centers  # may be None for d>=2; the spec will then
        # surface a clear "centers required for d>=2" error from the core.
    spec = Duchon(
        centers=centers_arg,
        m=int(m),
        length_scale=length_scale,
        periodic_per_axis=None,
        **_common(name, by, double_penalty, shape_constraint),
    )
    setattr(spec, "_gamfit_topology_dim", int(d))
    return spec
