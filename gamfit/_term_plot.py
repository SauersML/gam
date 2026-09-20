"""``Model.plot_terms``: draw each term's partial effect with matplotlib.

Drawing only. Every curve, standard error and band is a field of the
:class:`~gamfit.results.PartialEffect` the Rust core returns.
"""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

from ._partial_effect import PartialEffect

if TYPE_CHECKING:
    from ._model import Model

_LINE = "#1d4ed8"
_POINTWISE = "#93c5fd"
_SIMULTANEOUS = "#dbeafe"


def plot_terms(
    model: "Model",
    terms: str | Sequence[str] | None,
    *,
    level: float,
    n_points: int,
    axes: Any | None,
) -> list[Any]:
    import matplotlib.pyplot as plt

    if terms is None:
        names = [block.name for block in model.term_blocks if block.kind != "intercept"]
    elif isinstance(terms, str):
        names = [terms]
    else:
        names = [str(term) for term in terms]
    if not names:
        raise ValueError("plot_terms: the model has no non-intercept term to draw")
    effects = [model.partial_dependence(name, n_points=n_points, level=level) for name in names]
    if axes is None:
        _, grid = plt.subplots(1, len(effects), figsize=(4.5 * len(effects), 3.6), squeeze=False)
        ax_list = list(grid[0])
    else:
        ax_list = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
        if len(ax_list) != len(effects):
            raise ValueError(
                f"plot_terms: got {len(ax_list)} axes for {len(effects)} terms {names}"
            )
    for effect, ax in zip(effects, ax_list):
        draw_partial_effect(effect, ax)
    return ax_list


def draw_partial_effect(effect: PartialEffect, ax: Any) -> Any:
    """Draw one :class:`~gamfit.results.PartialEffect` on ``ax``."""
    factor_axes = [index for index, levels in enumerate(effect.axis_levels) if levels is not None]
    numeric_axes = [index for index, levels in enumerate(effect.axis_levels) if levels is None]
    if len(effect.axes) == 1 and factor_axes:
        _draw_levels(effect, ax)
    elif len(effect.axes) == 1:
        _draw_curve(effect, ax)
    elif len(effect.axes) == 2 and len(factor_axes) == 1:
        _draw_curve_per_level(effect, ax, numeric_axes[0], factor_axes[0])
    elif len(effect.axes) == 2 and effect.axis_values is not None:
        _draw_surface(effect, ax)
    else:
        raise ValueError(
            f"plot_terms: {effect.term} sweeps axes {list(effect.axes)}; a drawing needs one or "
            "two axes on the default grid. Use partial_dependence() for its numbers."
        )
    ax.set_title(effect.term)
    return ax


def _band_labels(effect: PartialEffect) -> tuple[str, str]:
    percent = f"{effect.level:.0%}"
    return f"{percent} pointwise", f"{percent} simultaneous"


def _draw_curve(effect: PartialEffect, ax: Any) -> None:
    pointwise, simultaneous = _band_labels(effect)
    x = effect.x
    ax.fill_between(
        x, effect.simultaneous_lower, effect.simultaneous_upper, color=_SIMULTANEOUS, label=simultaneous
    )
    ax.fill_between(x, effect.lower, effect.upper, color=_POINTWISE, label=pointwise)
    ax.plot(x, effect.fit, color=_LINE, linewidth=2, label="fit")
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_xlabel(effect.axes[0])
    ax.set_ylabel(effect.contribution)
    ax.legend(fontsize="small")


def _draw_levels(effect: PartialEffect, ax: Any) -> None:
    pointwise, simultaneous = _band_labels(effect)
    positions = np.arange(effect.fit.size)
    ax.errorbar(
        positions,
        effect.fit,
        yerr=[effect.fit - effect.simultaneous_lower, effect.simultaneous_upper - effect.fit],
        fmt="none",
        ecolor=_POINTWISE,
        elinewidth=1,
        capsize=6,
        label=simultaneous,
    )
    ax.errorbar(
        positions,
        effect.fit,
        yerr=[effect.fit - effect.lower, effect.upper - effect.fit],
        fmt="o",
        color=_LINE,
        elinewidth=2,
        capsize=3,
        label=pointwise,
    )
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(effect.labels(0))
    ax.set_xlabel(effect.axes[0])
    ax.set_ylabel(effect.contribution)
    ax.legend(fontsize="small")


def _draw_curve_per_level(effect: PartialEffect, ax: Any, numeric: int, factor: int) -> None:
    levels = effect.axis_levels[factor]
    assert levels is not None
    for code, label in zip(levels.values, levels.labels):
        rows = effect.grid[:, factor] == code
        x = effect.grid[rows, numeric]
        order = np.argsort(x)
        (line,) = ax.plot(x[order], effect.fit[rows][order], linewidth=2, label=label)
        ax.fill_between(
            x[order],
            effect.lower[rows][order],
            effect.upper[rows][order],
            color=line.get_color(),
            alpha=0.2,
        )
    ax.set_xlabel(effect.axes[numeric])
    ax.set_ylabel(effect.contribution)
    ax.legend(title=f"{effect.axes[factor]} ({_band_labels(effect)[0]})", fontsize="small")


def _draw_surface(effect: PartialEffect, ax: Any) -> None:
    assert effect.axis_values is not None
    first, second = effect.axis_values
    # surface()[i, j] is at (first[i], second[j]); contourf takes Z[row=y, col=x].
    fit = effect.surface("fit").T
    se = effect.surface("se").T
    filled = ax.contourf(first, second, fit, levels=12, cmap="RdBu_r")
    ax.figure.colorbar(filled, ax=ax, label=effect.contribution)
    se_lines = ax.contour(first, second, se, levels=5, colors="k", linewidths=0.8, linestyles="dashed")
    ax.clabel(se_lines, fontsize="x-small", fmt="se %.2g")
    ax.set_xlabel(effect.axes[0])
    ax.set_ylabel(effect.axes[1])


__all__ = ["draw_partial_effect", "plot_terms"]
