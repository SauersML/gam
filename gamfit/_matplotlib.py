"""Lazy access to the optional matplotlib dependency.

matplotlib is an optional extra (``pip install 'gamfit[plot]'``). Every
plotting entry point calls :func:`pyplot` at call time so ``import gamfit``
and ``import gamfit.plot`` never require it.
"""

from __future__ import annotations

from typing import Any


def pyplot() -> Any:
    """Return ``matplotlib.pyplot``, or raise a clear ImportError without it."""
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name is None or exc.name.split(".")[0] != "matplotlib":
            raise
        raise ImportError(
            "gamfit plotting requires matplotlib, which is not installed. "
            "Install it with: pip install 'gamfit[plot]'"
        ) from exc
    return plt
