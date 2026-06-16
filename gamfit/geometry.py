"""Manifold descriptors for latent-coordinate (``LatentCoord``) smooths.

Each class is a thin Python alias for a corresponding Rust pyclass exposed by
the ``gam-pyffi`` extension (``gamfit._rust``). The Python source carries the
canonical docstring (mkdocstrings reads source statically and cannot introspect
the compiled Rust pyclasses), while the loop at the bottom of this module
rebinds each name at import time to the actual Rust implementation so runtime
behavior is unchanged.
"""

from __future__ import annotations

from ._binding import rust_module

__all__ = [
    "CircleManifold",
    "EuclideanManifold",
    "GrassmannManifold",
    "ProductManifold",
    "SpdManifold",
    "SphereManifold",
    "StiefelManifold",
    "TorusManifold",
]


# ---------------------------------------------------------------------------
# Static documentation stubs.
#
# Each class below is rebound at import time to the corresponding Rust
# pyclass on `gamfit._rust`. Static analyzers (griffe / mkdocstrings) see the
# Python source — that's where the docstring lives — while every caller hits
# the Rust implementation at runtime.
# ---------------------------------------------------------------------------


class EuclideanManifold:
    """Flat Euclidean manifold :math:`\\mathbb{R}^d`.

    The trivial choice for a ``LatentCoord`` smooth: no curvature, no
    boundary, ambient = intrinsic dimension. Use it as a baseline against
    curved manifolds such as :class:`SphereManifold` or :class:`SpdManifold`.
    """


class CircleManifold:
    """Unit circle :math:`S^1` parameterised by an angle :math:`\\theta`.

    One-dimensional periodic manifold. Smooths on a ``CircleManifold`` wrap
    cleanly across the :math:`2\\pi` boundary so cyclic predictors (hour of
    day, day of year, angular covariates) get a continuous fit instead of an
    artificial seam.
    """


class SphereManifold:
    """Unit sphere :math:`S^{n}` embedded in :math:`\\mathbb{R}^{n+1}`.

    Intrinsic dimension :math:`n`, ambient dimension :math:`n+1`. Use for
    directional data (geomagnetic field, gaze direction) or compositional /
    proportions data after a square-root map onto the simplex's spherical
    parameterisation.
    """


class TorusManifold:
    """Flat torus :math:`T^n = (S^1)^n`.

    Product of :math:`n` circles. Smooths factorise into one periodic basis
    per axis; this is the natural codomain for multi-angle predictors such
    as joint cyclic time-of-day × day-of-week interactions.
    """


class GrassmannManifold:
    """Grassmann manifold :math:`\\mathrm{Gr}(k, n)` of :math:`k`-planes in :math:`\\mathbb{R}^n`.

    Points are equivalence classes of orthonormal :math:`n \\times k` frames
    under right multiplication by :math:`\\mathrm{O}(k)`. Used for subspace
    estimation problems (principal subspace tracking, factor rotation) where
    the response space is a subspace rather than a specific basis for it.
    """


class StiefelManifold:
    """Stiefel manifold :math:`V_k(\\mathbb{R}^n)` of orthonormal :math:`k`-frames in :math:`\\mathbb{R}^n`.

    Points are :math:`n \\times k` matrices :math:`U` with
    :math:`U^\\top U = I_k`. Unlike Grassmann, the specific frame matters
    here, so use Stiefel when the orientation within the subspace is
    identifiable (e.g. orthogonal regression coefficients with a chosen
    basis).
    """


class SpdManifold:
    """Cone of symmetric positive-definite matrices.

    Points are :math:`n \\times n` matrices :math:`P` with
    :math:`P = P^\\top` and :math:`P \\succ 0`. The affine-invariant
    Riemannian metric makes geodesics close under congruence, which is
    the right structure for covariance-valued or correlation-valued
    responses (diffusion tensors, brain-connectivity matrices, financial
    covariance trajectories).
    """


class ProductManifold:
    """Cartesian product :math:`M_1 \\times M_2 \\times \\cdots \\times M_k`.

    Combine heterogeneous manifolds (e.g. ``Sphere × Euclidean × Circle``)
    into a single descriptor for a ``LatentCoord`` smooth that lives on the
    product geometry. Tangent vectors, geodesics, and inner products
    decompose blockwise; smooths factorise correspondingly.
    """


# ---------------------------------------------------------------------------
# Runtime rebind: replace each stub with the matching `gamfit._rust` pyclass.
# Static analysis sees the class definitions above; runtime sees the Rust
# implementation, so existing callers and `isinstance` checks against the
# top-level `gamfit.X` re-export resolve to the live Rust type.
# ---------------------------------------------------------------------------
_rust = rust_module()
for _name in __all__:
    _cls = getattr(_rust, _name, None)
    if _cls is None:
        def _missing(*args, _missing_name: str = _name, **kwargs):
            del args, kwargs
            raise AttributeError(
                f"gamfit._rust does not expose {_missing_name}; rebuild the local Rust extension"
            )

        _missing.__name__ = _name
        globals()[_name] = _missing
    else:
        globals()[_name] = _cls
del _name, _rust
