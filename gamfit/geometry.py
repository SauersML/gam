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
