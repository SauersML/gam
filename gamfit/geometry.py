from __future__ import annotations

from ._binding import rust_module

_rust = rust_module()

CircleManifold = _rust.CircleManifold
EuclideanManifold = _rust.EuclideanManifold
GrassmannManifold = _rust.GrassmannManifold
ProductManifold = _rust.ProductManifold
SpdManifold = _rust.SpdManifold
SphereManifold = _rust.SphereManifold
StiefelManifold = _rust.StiefelManifold
TorusManifold = _rust.TorusManifold

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
