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

_rust = rust_module()
for _name in __all__:
    globals()[_name] = getattr(_rust, _name)
del _name, _rust
