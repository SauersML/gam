"""The typed result of :meth:`gamfit.Model.partial_dependence`.

Every number here comes from the Rust core
(``gam_predict::partial_effect::partial_effect``); this module only holds and
reshapes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AxisLevels:
    """A factor axis's levels: the code a grid column holds, and its label."""

    values: np.ndarray
    labels: tuple[str, ...]

    def label_of(self, value: float) -> str:
        """The label of the level encoded as ``value``."""
        matches = np.flatnonzero(self.values == value)
        if matches.size == 0:
            raise ValueError(f"{value!r} is not a level code; codes: {self.values.tolist()}")
        return self.labels[int(matches[0])]


@dataclass(frozen=True)
class PartialEffect:
    """A term's partial effect ``f_t = X_t β_t`` on a grid, with its bands.

    The curve is on the linear-predictor scale and is the centred effect the fit
    estimated: the term's basis already carries its identifiability constraint.
    Its covariance is the term block of the one the fit publishes, named by
    :attr:`covariance_source` (``"smoothing-corrected"`` whenever the fit
    carries it, otherwise ``"conditional"``).

    Attributes
    ----------
    term, axes:
        The term, and the columns its grid sweeps, in grid-column order.
    grid:
        ``(n, len(axes))`` evaluation points. A factor axis holds level codes;
        :attr:`axis_levels` names them.
    axis_levels:
        Per axis, its :class:`AxisLevels` for a factor axis and ``None`` for a
        numeric one.
    axis_values:
        For the default grid, the values each axis sweeps. The grid is their
        Cartesian product with the last axis varying fastest, so
        :meth:`surface` can reshape it. ``None`` for a caller grid.
    fit, se:
        The curve and its standard error, one entry per grid row.
    lower, upper:
        Pointwise intervals at :attr:`level`: ``fit ∓ pointwise_critical · se``.
        The critical value is the two-sided quantile of the fit's pivot law:
        Student-t on ``n - edf_total`` when the fit estimates its dispersion
        (Gaussian, Gamma, ...), standard normal when the scale is known.
    simultaneous_lower, simultaneous_upper:
        A band that covers the whole curve over the grid with probability
        :attr:`level`: ``fit ∓ simultaneous_critical · se``. The critical value
        is the ``level`` quantile of ``max |standardized curve error|`` over
        the grid (a multivariate-t supremum when the dispersion is estimated),
        calibrated from :attr:`simulations` posterior draws (seed :attr:`seed`);
        the draw count holds the Monte-Carlo error of the attained coverage to
        a fixed fraction of its miss rate.
    scale, quantity, contribution, held:
        Always ``"linear_predictor"``; ``"term_contribution"`` or
        ``"coefficient_function"`` for a numeric ``by=`` smooth; how the curve
        enters the predictor, e.g. ``"z * f(x)"``; and the columns the term
        reads that every grid row holds fixed.
    """

    term: str
    axes: tuple[str, ...]
    grid: np.ndarray
    axis_levels: tuple[AxisLevels | None, ...]
    axis_values: tuple[np.ndarray, ...] | None
    fit: np.ndarray
    se: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    simultaneous_lower: np.ndarray
    simultaneous_upper: np.ndarray
    level: float
    pointwise_critical: float
    simultaneous_critical: float
    simulations: int
    seed: int
    covariance_source: str
    scale: str
    quantity: str
    contribution: str
    held: dict[str, Any]

    @classmethod
    def _from_rust(cls, raw: dict[str, Any]) -> "PartialEffect":
        axis_values = raw["axis_values"]
        return cls(
            term=str(raw["term"]),
            axes=tuple(str(axis) for axis in raw["axes"]),
            grid=np.asarray(raw["grid"], dtype=float),
            axis_levels=tuple(
                None
                if levels is None
                else AxisLevels(
                    values=np.asarray(levels["values"], dtype=float),
                    labels=tuple(str(label) for label in levels["labels"]),
                )
                for levels in raw["axis_levels"]
            ),
            axis_values=None
            if axis_values is None
            else tuple(np.asarray(values, dtype=float) for values in axis_values),
            fit=np.asarray(raw["fit"], dtype=float),
            se=np.asarray(raw["se"], dtype=float),
            lower=np.asarray(raw["lower"], dtype=float),
            upper=np.asarray(raw["upper"], dtype=float),
            simultaneous_lower=np.asarray(raw["simultaneous_lower"], dtype=float),
            simultaneous_upper=np.asarray(raw["simultaneous_upper"], dtype=float),
            level=float(raw["level"]),
            pointwise_critical=float(raw["pointwise_critical"]),
            simultaneous_critical=float(raw["simultaneous_critical"]),
            simulations=int(raw["simulations"]),
            seed=int(raw["seed"]),
            covariance_source=str(raw["covariance_source"]),
            scale=str(raw["scale"]),
            quantity=str(raw["quantity"]),
            contribution=str(raw["contribution"]),
            held=dict(raw["held"]),
        )

    @property
    def x(self) -> np.ndarray:
        """The grid as a vector, for a one-axis term."""
        if len(self.axes) != 1:
            raise ValueError(
                f"{self.term} has axes {list(self.axes)}; use .grid, or .surface() for a product grid"
            )
        return self.grid[:, 0]

    def labels(self, axis: int = 0) -> list[str]:
        """Each grid row's level label on the factor axis ``axis``."""
        levels = self.axis_levels[axis]
        if levels is None:
            raise ValueError(f"axis {self.axes[axis]!r} of {self.term} is numeric, not a factor")
        return [levels.label_of(value) for value in self.grid[:, axis]]

    def surface(self, name: str = "fit") -> np.ndarray:
        """A per-row series reshaped onto the product grid, one array axis per term axis.

        ``name`` is ``"fit"``, ``"se"``, ``"lower"``, ``"upper"``,
        ``"simultaneous_lower"`` or ``"simultaneous_upper"``. Entry
        ``[i, j, ...]`` is the value at ``axis_values[0][i], axis_values[1][j], ...``.
        """
        if self.axis_values is None:
            raise ValueError("surface() needs the default product grid; this effect was evaluated on a caller grid")
        if name not in _SERIES:
            raise ValueError(f"surface: unknown series {name!r}; choose one of {list(_SERIES)}")
        shape = tuple(len(values) for values in self.axis_values)
        return getattr(self, name).reshape(shape)


_SERIES = ("fit", "se", "lower", "upper", "simultaneous_lower", "simultaneous_upper")

__all__ = ["AxisLevels", "PartialEffect"]
