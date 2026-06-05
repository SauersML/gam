"""`_rust.*` exceptions must survive pickling so real engine errors propagate
across process boundaries (issue #773).

`pyo3::create_exception!` stamps every gamfit exception with
``__module__ == "_rust"``, but the compiled extension is importable only as
``gamfit._rust``. Pickle records a class by ``(__module__, __qualname__)`` and
reconstructs it with ``import _rust; getattr(...)``, which fails — so a
``_rust.*`` exception raised inside a ``ProcessPoolExecutor`` worker used to be
masked by an opaque ``PicklingError`` (hiding the real failure and taking down
the pool). The FFI module init repoints every ``GamError`` subclass's
``__module__`` at ``gamfit._rust``; these tests pin that round-trip.
"""
from __future__ import annotations

import importlib
import pickle

from gamfit._binding import rust_module

_rust = rust_module()


def _gam_error_subclasses() -> list[type]:
    """Every exception class the Rust extension exposes (GamError + subclasses).

    Discovered from the module dict rather than hardcoded, mirroring the
    GamError-subclass walk the FFI init uses to repoint ``__module__`` — so a
    newly added ``create_exception!`` is covered without editing this test.
    """
    base = _rust.GamError
    out = [
        value
        for value in vars(_rust).values()
        if isinstance(value, type) and issubclass(value, base)
    ]
    assert out, "expected at least GamError itself among _rust exception types"
    return out


def test_every_rust_exception_is_picklable() -> None:
    for exc_type in _gam_error_subclasses():
        instance = exc_type("boom")
        for protocol in (pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL):
            restored = pickle.loads(pickle.dumps(instance, protocol))
            assert type(restored) is exc_type, (
                f"{exc_type.__name__} did not round-trip through pickle "
                f"(protocol {protocol}): got {type(restored).__name__}"
            )
            assert str(restored) == "boom"


def test_rust_exception_module_is_importable() -> None:
    # Pickle resolves a class via its declared __module__; that module must be
    # importable and must expose the class under its __qualname__.
    for exc_type in _gam_error_subclasses():
        module = importlib.import_module(exc_type.__module__)
        assert getattr(module, exc_type.__qualname__) is exc_type, (
            f"{exc_type.__qualname__} is not resolvable from its declared "
            f"module {exc_type.__module__!r}"
        )
