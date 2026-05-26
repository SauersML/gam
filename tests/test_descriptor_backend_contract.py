"""RED tests for issues #228, #232, #235.

#228: Every `Smooth` descriptor must expose `SUPPORTED_BACKENDS` as a
      `ClassVar[frozenset[str]]` reachable from both the class and instances.
#232: `Matern.SUPPORTED_BACKENDS` must be declared and consistent with which
      `_evaluate_*` methods exist; the cross-backend contract must hold for
      every descriptor (advertised backends work, unadvertised ones raise a
      clean error — not `AttributeError`).
#235: `basis_size` must be available after a single `.evaluate(x)` call on
      auto-resolving descriptors (e.g. `BSpline` with `knots=None`), so the
      JAX path can read it.
"""

from __future__ import annotations

import typing
from typing import ClassVar, get_type_hints

import numpy as np
import pytest

import gamfit
from gamfit.smooth import Smooth


def _descriptor_subclasses() -> list[type]:
    seen: dict[str, type] = {}

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("gamfit"):
                seen.setdefault(sub.__qualname__, sub)
            walk(sub)

    walk(Smooth)
    return list(seen.values())


# ---------------------------------------------------------------------------
# #228 — SUPPORTED_BACKENDS must exist on every descriptor, as a ClassVar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _descriptor_subclasses(), ids=lambda c: c.__qualname__)
def test_descriptor_declares_supported_backends_as_classvar(cls: type) -> None:
    """Class-level access returns a real frozenset, not a slot descriptor."""
    value = getattr(cls, "SUPPORTED_BACKENDS", None)
    assert value is not None, (
        f"{cls.__qualname__} is missing SUPPORTED_BACKENDS"
    )
    assert isinstance(value, frozenset), (
        f"{cls.__qualname__}.SUPPORTED_BACKENDS must be a frozenset "
        f"(got {type(value).__name__!r}); usually means it's a slot/member "
        "descriptor because it was declared as an instance field rather than "
        "ClassVar[frozenset[str]]"
    )
    # Non-empty iff the descriptor implements at least one _evaluate_<backend>
    # method. Carrier-only descriptors (e.g. Categorical) legitimately have
    # frozenset() because formula compilation consumes them directly.
    impl_methods = {
        name.removeprefix("_evaluate_")
        for name in dir(cls)
        if name.startswith("_evaluate_") and name != "_evaluate_impl"
    }
    if impl_methods & {"torch", "numpy", "jax"}:
        assert value, (
            f"{cls.__qualname__} has _evaluate_* methods but advertises no backends"
        )


@pytest.mark.parametrize("cls", _descriptor_subclasses(), ids=lambda c: c.__qualname__)
def test_descriptor_supported_backends_annotated_classvar(cls: type) -> None:
    """The annotation must say ClassVar so dataclass/slots don't grab it."""
    hints = get_type_hints(cls, include_extras=True)
    ann = hints.get("SUPPORTED_BACKENDS")
    assert ann is not None, (
        f"{cls.__qualname__} has no SUPPORTED_BACKENDS annotation"
    )
    origin = typing.get_origin(ann)
    assert origin is ClassVar or ann is ClassVar or (
        isinstance(ann, type) is False and "ClassVar" in repr(ann)
    ), (
        f"{cls.__qualname__}.SUPPORTED_BACKENDS must be annotated as "
        f"ClassVar[frozenset[str]] (got {ann!r}); otherwise dataclass(slots=True) "
        "stores it per-instance and AttributeError leaks at evaluate() time"
    )


def test_pca_instance_exposes_supported_backends() -> None:
    """Regression for #228: Pca uses dataclass(init=False, slots=True) with a
    manual __init__ that never assigns SUPPORTED_BACKENDS, so the attribute
    is invisible on instances even though the class-body line exists."""
    basis = np.random.default_rng(0).standard_normal((7, 4))
    spec = gamfit.Pca(basis=basis)
    assert hasattr(spec, "SUPPORTED_BACKENDS"), (
        "Pca() instance is missing SUPPORTED_BACKENDS — slots=True + manual "
        "__init__ ate it. Declare it as ClassVar[frozenset[str]]."
    )
    assert isinstance(spec.SUPPORTED_BACKENDS, frozenset)


# ---------------------------------------------------------------------------
# #232 — backend declaration must match implementation, and the contract
# (advertised backends evaluate; unadvertised backends raise cleanly) holds.
# ---------------------------------------------------------------------------


def test_matern_declares_supported_backends() -> None:
    """Regression for #232: Matern was missing the declaration entirely."""
    value = getattr(gamfit.Matern, "SUPPORTED_BACKENDS", None)
    assert isinstance(value, frozenset) and value, (
        "Matern.SUPPORTED_BACKENDS must be a non-empty frozenset"
    )


def test_matern_declaration_matches_implementation() -> None:
    """If `_evaluate_numpy` exists, 'numpy' must be advertised, and vice versa.
    Catches the drift described in #232 (both evaluators exist but tests
    expect torch-only)."""
    cls = gamfit.Matern
    has_numpy = "_evaluate_numpy" in cls.__dict__
    has_torch = "_evaluate_torch" in cls.__dict__
    advertised = set(cls.SUPPORTED_BACKENDS)
    if has_numpy:
        assert "numpy" in advertised, (
            "Matern._evaluate_numpy exists but 'numpy' not in SUPPORTED_BACKENDS"
        )
    else:
        assert "numpy" not in advertised, (
            "Matern advertises 'numpy' but has no _evaluate_numpy"
        )
    if has_torch:
        assert "torch" in advertised
    else:
        assert "torch" not in advertised


@pytest.mark.parametrize("cls", _descriptor_subclasses(), ids=lambda c: c.__qualname__)
def test_descriptor_declaration_matches_evaluators(cls: type) -> None:
    """For every descriptor: each advertised backend must have an evaluator,
    and every existing _evaluate_<backend> must be advertised. Prevents the
    drift class behind #232."""
    backends = set(getattr(cls, "SUPPORTED_BACKENDS", frozenset()))
    impl_methods = {
        name.removeprefix("_evaluate_")
        for name in dir(cls)
        if name.startswith("_evaluate_")
    }
    for b in backends:
        assert b in impl_methods, (
            f"{cls.__qualname__} advertises backend {b!r} but has no "
            f"_evaluate_{b}"
        )
    for m in impl_methods:
        if m in {"torch", "numpy", "jax"}:
            assert m in backends, (
                f"{cls.__qualname__} implements _evaluate_{m} but does not "
                "advertise it in SUPPORTED_BACKENDS"
            )


# ---------------------------------------------------------------------------
# #235 — basis_size must be available after one .evaluate(x).
# ---------------------------------------------------------------------------


def test_bspline_basis_size_after_one_evaluate() -> None:
    """Regression for #235: NumPy/Torch evaluate auto-resolve knots but never
    cache them on the spec, so JAX (which reads basis_size before dispatch)
    blows up."""
    spec = gamfit.BSpline(degree=3, periodic=False)
    x = np.linspace(0.0, 1.0, 9)
    spec.evaluate(x, backend="numpy")
    # The error message in bspline_basis_size promises this works.
    size = spec.basis_size
    assert isinstance(size, int) and size > 0, (
        "BSpline.basis_size must be defined after evaluate() resolves knots — "
        "the error message says 'call .evaluate(x) once' and that contract "
        "must hold for the JAX path to work."
    )


def test_bspline_jax_evaluate_after_numpy_evaluate() -> None:
    """End-to-end #235 repro: numpy evaluate then jax evaluate."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    spec = gamfit.BSpline(degree=3, periodic=False)
    x = np.linspace(0.0, 1.0, 9)
    spec.evaluate(x, backend="numpy")
    out = spec.evaluate(jnp.asarray(x), backend="jax")
    assert out.shape[0] == 9


def test_bspline_knots_cached_after_evaluate() -> None:
    """Stronger #235 invariant: knots are written back to the spec so any
    subsequent backend (including JAX) can read static output shape."""
    spec = gamfit.BSpline(degree=3, periodic=False)
    x = np.linspace(0.0, 1.0, 9)
    assert spec.knots is None or isinstance(spec.knots, int)
    spec.evaluate(x, backend="numpy")
    assert spec.knots is not None and not isinstance(spec.knots, int), (
        "BSpline.evaluate must cache resolved knots on the spec; otherwise "
        "the JAX path's static-shape contract is unreachable."
    )
