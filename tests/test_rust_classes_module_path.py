"""Every class exported from ``gamfit._rust`` names its real, importable home
(pyGAM audit PKG-02 / api F1, lane pyclass-module-path).

The research pyclasses declared ``module = "gam_pyffi._rust"`` (a crate name,
not an importable Python module), and ``_EncodedTable`` / ``_JointEventModel``
/ ``_EventHistoryModel`` declared no module at all, so they reported
``builtins``. Either way ``pickle`` could not resolve the class by reference,
``repr`` pointed at a path that does not exist, and ``importlib`` could not
find the class from its own ``__module__`` / ``__qualname__``.
"""

from __future__ import annotations

import importlib
import pickle

import pytest

import gamfit._rust as _rust

_EXTENSION_MODULE = "gamfit._rust"

_EXPORTED_CLASSES = sorted(
    name for name, obj in vars(_rust).items() if isinstance(obj, type)
)


def test_extension_exports_classes() -> None:
    # Guard against the parametrization below silently collapsing to nothing.
    assert "_FittedModel" in _EXPORTED_CLASSES
    assert "SphereManifold" in _EXPORTED_CLASSES
    assert "ARDPenalty" in _EXPORTED_CLASSES


@pytest.mark.parametrize("name", _EXPORTED_CLASSES)
def test_exported_class_resolves_through_its_module_path(name: str) -> None:
    cls = getattr(_rust, name)

    assert cls.__module__ == _EXTENSION_MODULE
    assert cls.__qualname__ == name
    home = importlib.import_module(cls.__module__)
    assert getattr(home, cls.__qualname__) is cls
    assert repr(cls) == f"<class '{_EXTENSION_MODULE}.{name}'>"
    # Classes pickle by reference, through exactly the lookup above.
    assert pickle.loads(pickle.dumps(cls)) is cls
