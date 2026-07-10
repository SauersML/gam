"""Small public constructors for common manifold topology candidates."""

from __future__ import annotations

from typing import Any, Literal

from .smooth import (
    BSpline,
    Duchon,
    PeriodicSplineCurve,
    ShapeConstraintLiteral,
    Smooth,
    Sphere as _SphereSmooth,
    TensorBSpline,
)


__all__ = ["Circle", "Cylinder", "Torus", "Sphere", "EuclideanPatch"]


def _make_topology(
    kind: Literal["circle", "cylinder", "torus", "sphere", "euclidean_patch"],
    **opts: Any,
) -> Smooth:
    common = {
        "name": opts["name"],
        "by": opts["by"],
        "double_penalty": opts["double_penalty"],
        "shape_constraint": opts["shape_constraint"],
    }

    if kind == "circle":
        return PeriodicSplineCurve(
            n_knots=int(opts["n_knots"]),
            degree=int(opts["degree"]),
            output_dim=int(opts["output_dim"]),
            penalty_order=int(opts["penalty_order"]),
            **common,
        )

    if kind in {"cylinder", "torus"}:
        periodic = (True, kind == "torus")
        spec = TensorBSpline(
            marginals=[BSpline(periodic=value) for value in periodic],
            **common,
        )
        setattr(spec, "_gamfit_topology_dim", 2)
        setattr(spec, "_gamfit_tensor_k", tuple(int(value) for value in opts["n_knots"]))
        setattr(spec, "_gamfit_tensor_periods", tuple("2*pi" if value else None for value in periodic))
        setattr(spec, "_gamfit_tensor_identifiability", "sum_tozero")
        return spec

    if kind == "sphere":
        if int(opts["dim"]) != 2:
            raise ValueError("topology.Sphere supports dim=2 latitude/longitude inputs")
        spec = _SphereSmooth(
            n_centers=int(opts["n_knots"]),
            penalty_order=int(opts["penalty_order"]),
            kernel=str(opts["kernel"]),
            radians=bool(opts["radians"]),
            **common,
        )
        setattr(spec, "_gamfit_topology_dim", 2)
        return spec

    if kind == "euclidean_patch":
        d = int(opts["d"])
        centers = opts["centers"]
        spec = Duchon(
            centers=int(opts["n_centers"]) if centers is None and d == 1 else centers,
            m=int(opts["m"]),
            length_scale=opts["length_scale"],
            periodic_per_axis=None,
            **common,
        )
        setattr(spec, "_gamfit_topology_dim", d)
        return spec

    raise AssertionError(f"unknown topology kind: {kind}")


def Circle(
    name: str | None = None,
    n_knots: int = 20,
    *,
    degree: int = 3,
    penalty_order: int = 2,
    output_dim: int = 1,
    by: Any | None = None,
    double_penalty: bool | None = None,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Closed one-dimensional loop for angular scalar inputs.

    Use for phase, hue, time-of-day, or other periodic coordinates shaped
    like ``(N,)`` or one column. Set ``output_dim`` above one for a closed
    curve embedded in a higher-dimensional response.

    Returns
    -------
    Smooth
        ``PeriodicSplineCurve`` descriptor suitable for
        :func:`gamfit.select_topology` candidates or ``smooths=`` mappings.
    """
    return _make_topology("circle", name=name, n_knots=n_knots, degree=degree, penalty_order=penalty_order, output_dim=output_dim, by=by, double_penalty=double_penalty, shape_constraint=shape_constraint)


def Cylinder(
    name: str | None = None,
    n_knots: tuple[int, int] = (20, 8),
    *,
    by: Any | None = None,
    double_penalty: bool | None = None,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Tensor topology with one periodic axis and one open axis.

    Use when inputs are shaped ``(N, 2)`` and the first column wraps while
    the second remains Euclidean, such as angle by height or season by trend.

    Returns
    -------
    Smooth
        ``TensorBSpline`` descriptor with periodicity ``(True, False)``.
    """
    return _make_topology("cylinder", name=name, n_knots=n_knots, by=by, double_penalty=double_penalty, shape_constraint=shape_constraint)


def Torus(
    name: str | None = None,
    n_knots: tuple[int, int] = (20, 20),
    *,
    by: Any | None = None,
    double_penalty: bool | None = None,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Tensor topology with two periodic axes.

    Use when inputs are shaped ``(N, 2)`` and both columns wrap, such as
    phase by phase or longitude by time-of-day.

    Returns
    -------
    Smooth
        ``TensorBSpline`` descriptor with periodicity ``(True, True)``.
    """
    return _make_topology("torus", name=name, n_knots=n_knots, by=by, double_penalty=double_penalty, shape_constraint=shape_constraint)


def Sphere(
    name: str | None = None,
    n_knots: int = 20,
    *,
    dim: int = 2,
    penalty_order: int = 2,
    kernel: str = "sobolev",
    radians: bool = False,
    by: Any | None = None,
    double_penalty: bool | None = None,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Spherical topology for latitude/longitude style inputs.

    Use for directions on ``S^2`` represented as ``(N, 2)`` coordinates.
    ``dim`` remains part of the public signature but must be ``2``.

    Returns
    -------
    Smooth
        Spherical smooth descriptor.

    Raises
    ------
    ValueError
        If ``dim`` is not ``2``.
    """
    return _make_topology("sphere", name=name, n_knots=n_knots, dim=dim, penalty_order=penalty_order, kernel=kernel, radians=radians, by=by, double_penalty=double_penalty, shape_constraint=shape_constraint)


def EuclideanPatch(
    d: int = 2,
    name: str | None = None,
    n_centers: int = 64,
    *,
    centers: Any | None = None,
    m: int = 2,
    length_scale: float | None = None,
    by: Any | None = None,
    double_penalty: bool | None = None,
    shape_constraint: ShapeConstraintLiteral | None = None,
) -> Smooth:
    """Open Euclidean baseline with no periodic axes.

    Use for inputs shaped ``(N, d)`` when no closed topology is assumed.
    For ``d > 1``, pass explicit ``centers``; ``None`` is forwarded so the
    underlying spec reports the required-center error.

    Returns
    -------
    Smooth
        Duchon smooth descriptor marked with the requested Euclidean dimension.
    """
    return _make_topology("euclidean_patch", d=d, name=name, n_centers=n_centers, centers=centers, m=m, length_scale=length_scale, by=by, double_penalty=double_penalty, shape_constraint=shape_constraint)
