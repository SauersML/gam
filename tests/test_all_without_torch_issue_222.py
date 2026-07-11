"""RED tests for issue #222: torch-only names in __all__ break hasattr / star-import
without torch installed.

These tests simulate a torch-less environment by shimming ``sys.modules['torch']``
to ``None`` so any ``import torch`` raises ``ModuleNotFoundError``. In that state,
``hasattr(gamfit, name)`` must return True/False and never propagate the import
error, and ``from gamfit import *`` must succeed.

Currently FAILING: ``gamfit.__getattr__`` lazily imports torch for names like
``PoincareAtoms``, ``InterchangeSwapDecoder``, and ``Crosscoder``,
all of which appear in ``gamfit.__all__``.
"""

from __future__ import annotations

import sys

import pytest

import gamfit


TORCH_LAZY_NAMES = [
    "PoincareAtoms",
    "InterchangeSwapDecoder",
    "Crosscoder",
]


@pytest.fixture
def no_torch(monkeypatch):
    """Make ``import torch`` raise ModuleNotFoundError for the duration of the test."""
    # Drop any cached torch / torch submodules so a fresh import is attempted.
    for mod_name in [m for m in list(sys.modules) if m == "torch" or m.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
    # Also drop gamfit.torch submodules so their cached imports don't shortcut the lazy hook.
    for mod_name in [m for m in list(sys.modules) if m.startswith("gamfit.torch")]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
    # Setting to None makes ``import torch`` raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "torch", None)
    yield


@pytest.mark.parametrize("name", TORCH_LAZY_NAMES)
def test_hasattr_returns_bool_without_torch(no_torch, name):
    """hasattr must not propagate ModuleNotFoundError from the lazy torch import."""
    try:
        result = hasattr(gamfit, name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"hasattr(gamfit, {name!r}) raised ModuleNotFoundError instead of "
            f"returning a bool: {exc}. __getattr__ must convert the missing "
            f"optional torch dep into AttributeError."
        )
    assert isinstance(result, bool)


def test_star_import_without_torch_does_not_raise(no_torch):
    """`from gamfit import *` walks __all__; every name must resolve without torch.

    Either the torch-only names must be removed from __all__ when torch is
    unavailable, or __getattr__ must surface the missing optional dep as
    AttributeError so the name is treated as not exported.
    """
    ns: dict[str, object] = {}
    try:
        exec("from gamfit import *", ns)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"`from gamfit import *` raised ModuleNotFoundError due to torch-only "
            f"entries in __all__: {exc}"
        )


def test_every_name_in_all_is_resolvable_without_torch(no_torch):
    """Every entry in gamfit.__all__ must be retrievable (or cleanly absent)
    without torch installed.

    Concretely: ``getattr(gamfit, name)`` either returns a value, or raises
    AttributeError (clean optional-dep signal). It must NOT raise
    ModuleNotFoundError.
    """
    failures: list[str] = []
    for name in gamfit.__all__:
        try:
            getattr(gamfit, name)
        except AttributeError:
            # Acceptable: optional dep signalled as missing attribute.
            pass
        except ModuleNotFoundError as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, (
        "These names in gamfit.__all__ leak ModuleNotFoundError instead of "
        "AttributeError when torch is unavailable:\n  - "
        + "\n  - ".join(failures)
    )
