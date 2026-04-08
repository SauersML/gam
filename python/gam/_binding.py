from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType


class RustExtensionUnavailableError(ImportError):
    pass


@lru_cache(maxsize=1)
def rust_module() -> ModuleType:
    try:
        return importlib.import_module("gam._rust")
    except ImportError as exc:  # pragma: no cover - import environment specific
        raise RustExtensionUnavailableError(
            "gam._rust is not available. Build or install the package with maturin first."
        ) from exc


def extension_status() -> dict[str, object]:
    try:
        module = rust_module()
    except RustExtensionUnavailableError as exc:
        return {
            "available": False,
            "module": "gam._rust",
            "reason": str(exc),
        }
    build_info = module.build_info()
    return {
        "available": True,
        "module": "gam._rust",
        **dict(build_info),
    }
