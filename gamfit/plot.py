"""Plotting entry points.

matplotlib is optional (``pip install 'gamfit[plot]'``). This module imports
without it; each function imports matplotlib when called and raises an
``ImportError`` naming the missing extra when it is not installed.

- :func:`model` -- prediction, residual, or observed-vs-predicted plot of a
  fitted :class:`gamfit.Model` on a data table (same as ``Model.plot``).
- :func:`trace` -- trace and histogram panels for posterior draws (same as
  ``PosteriorSamples.plot_trace``).
- :func:`sae_atom` -- one SAE manifold atom, either inside a fitted SAE or a
  standalone atom.
- :func:`sae_fit` -- every atom of a fitted SAE in one figure.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._model import Model
    from ._sampling import PosteriorSamples


def model(
    model: "Model",
    data: Any,
    *,
    x: str | None = None,
    y: str | None = None,
    interval: float | None = 0.95,
    kind: str = "prediction",
    ax: Any | None = None,
) -> Any:
    """Plot a fitted model on ``data`` and return the matplotlib axes.

    ``kind`` is one of ``"prediction"``, ``"residuals"`` or
    ``"observed_vs_predicted"``.
    """
    from ._diagnose_plot import plot as _plot

    return _plot(model, data, x=x, y=y, interval=interval, kind=kind, ax=ax)


def trace(
    samples: "PosteriorSamples",
    *,
    coefficients: Any = None,
    max_panels: int = 8,
) -> Any:
    """Plot trace and histogram panels for posterior draws; return the figure."""
    return samples.plot_trace(coefficients=coefficients, max_panels=max_panels)


def sae_atom(
    target: Any,
    atom: int | None = None,
    *,
    ax: Any = None,
    color_by: str = "assignment",
) -> Any:
    """Plot one SAE manifold atom and return the matplotlib axes.

    ``sae_atom(fit, atom=k)`` renders atom ``k`` of a fitted SAE in its
    leading decoder SVD subspace. ``sae_atom(atom)`` renders a coordinate
    scatter for a standalone atom object or atom dictionary.
    """
    from ._sae_viz import plot as _plot

    return _plot(target, atom, ax=ax, color_by=color_by)


def sae_fit(fit: Any) -> Any:
    """Plot every atom of a fitted SAE in a grid and return the figure."""
    from ._sae_viz import plot_fit as _plot_fit

    return _plot_fit(fit)


__all__ = ["model", "sae_atom", "sae_fit", "trace"]
