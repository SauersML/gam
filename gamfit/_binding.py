from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

from ._cuda import preload_cuda_libraries


class RustExtensionUnavailableError(ImportError):
    """Raised when the compiled ``gamfit._rust`` extension cannot be imported.

    The Rust engine ships as a maturin-built extension module. When it is
    missing (typical in a fresh source checkout that has not been built yet),
    every Rust-backed API in :mod:`gamfit` raises this error eagerly so users
    see a single, actionable message instead of an opaque ``ImportError``.

    The fix is to build or install the package, e.g. ``maturin develop`` from
    the ``gamfit`` source tree, or ``pip install gamfit`` from PyPI.

    Examples
    --------
    >>> try:
    ...     gamfit.fit(df, "y ~ s(x)")
    ... except gamfit.RustExtensionUnavailableError as exc:
    ...     print("build the extension first:", exc)
    """


@lru_cache(maxsize=1)
def rust_module() -> ModuleType:
    preload_cuda_libraries()
    try:
        return importlib.import_module("gamfit._rust")
    except ImportError as exc:  # pragma: no cover - import environment specific
        raise RustExtensionUnavailableError(
            "gamfit._rust is not available. Build or install the package with maturin first."
        ) from exc


def extension_status() -> dict[str, object]:
    try:
        module = rust_module()
    except RustExtensionUnavailableError as exc:
        return {
            "available": False,
            "module": "gamfit._rust",
            "reason": str(exc),
        }
    build_info = module.build_info()
    return {
        "available": True,
        "module": "gamfit._rust",
        **dict(build_info),
    }
