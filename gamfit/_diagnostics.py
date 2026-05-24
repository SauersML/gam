from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Held-out / in-sample diagnostics for a fitted GAM.

    Bundles observed responses, model-implied predictions, residuals, and
    aggregate fit metrics (MAE, RMSE, bias, optional :math:`R^2`) into a
    single immutable record. Returned by :meth:`Model.diagnose` and rendered
    inline in notebooks via :meth:`_repr_html_`.

    Key fields:

    - ``formula``: the model formula used to produce the predictions.
    - ``response_name``: name of the response column in the input table.
    - ``observed``: actual response values aligned with ``predicted["mean"]``.
    - ``residuals``: ``observed - predicted["mean"]`` per row.
    - ``predicted``: dictionary of prediction series (``mean`` plus optional
      ``mean_lower`` / ``mean_upper`` interval bounds).
    - ``metrics``: scalar fit metrics (``n_obs``, ``mae``, ``rmse``, ``bias``,
      and ``r_squared`` when the response varies).
    - ``interval_lower`` / ``interval_upper``: optional pointwise prediction
      bands when the underlying call requested an interval.

    Examples
    --------
    >>> diag = model.diagnose(test)
    >>> diag.metrics["rmse"]
    0.42
    """

    formula: str
    response_name: str
    observed: list[float]
    residuals: list[float]
    predicted: dict[str, list[float]]
    metrics: dict[str, float]
    interval_lower: list[float] | None = None
    interval_upper: list[float] | None = None

    @classmethod
    def from_predictions(
        cls,
        *,
        formula: str,
        response_name: str,
        observed: list[float],
        predicted: dict[str, list[float]],
    ) -> "Diagnostics":
        """Construct a :class:`Diagnostics` from raw observed and predicted series.

        Computes residuals and aggregate fit metrics (n, MAE, RMSE, bias, and
        :math:`R^2` when the response variance is positive) from the inputs.

        Parameters
        ----------
        formula : str
            Model formula associated with the predictions.
        response_name : str
            Name of the response column.
        observed : list of float
            Observed response values.
        predicted : dict of str to list of float
            Prediction series. Must contain key ``"mean"``; may contain
            ``"mean_lower"`` and ``"mean_upper"`` for interval bands.

        Returns
        -------
        Diagnostics
            Populated diagnostics record with computed residuals and metrics.

        Examples
        --------
        >>> Diagnostics.from_predictions(
        ...     formula="y ~ s(x)",
        ...     response_name="y",
        ...     observed=[1.0, 2.0, 3.0],
        ...     predicted={"mean": [1.1, 1.9, 3.2]},
        ... ).metrics["mae"]
        0.13333333333333336
        """
        mean = [float(value) for value in predicted["mean"]]
        residuals = [obs - pred for obs, pred in zip(observed, mean, strict=True)]
        n_obs = len(observed)
        mae = sum(abs(value) for value in residuals) / n_obs
        rmse = (sum(value * value for value in residuals) / n_obs) ** 0.5
        bias = sum(residuals) / n_obs
        observed_mean = sum(observed) / n_obs
        total_sum_squares = sum((value - observed_mean) ** 2 for value in observed)
        residual_sum_squares = sum(value * value for value in residuals)
        metrics = {
            "n_obs": float(n_obs),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
        }
        if total_sum_squares > 0.0:
            metrics["r_squared"] = 1.0 - residual_sum_squares / total_sum_squares
        return cls(
            formula=formula,
            response_name=response_name,
            observed=observed,
            residuals=residuals,
            predicted=predicted,
            metrics=metrics,
            interval_lower=predicted.get("mean_lower"),
            interval_upper=predicted.get("mean_upper"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` snapshot of the diagnostics record.

        Returns
        -------
        dict
            Mapping with copies of every field, suitable for JSON-style
            serialization or further inspection.

        Examples
        --------
        >>> diag.to_dict()["metrics"]["rmse"]
        0.42
        """
        return {
            "formula": self.formula,
            "response_name": self.response_name,
            "observed": list(self.observed),
            "residuals": list(self.residuals),
            "predicted": {key: list(value) for key, value in self.predicted.items()},
            "metrics": dict(self.metrics),
            "interval_lower": None if self.interval_lower is None else list(self.interval_lower),
            "interval_upper": None if self.interval_upper is None else list(self.interval_upper),
        }

    def __repr__(self) -> str:
        metric_text = ", ".join(
            f"{name}={value:.6g}" for name, value in self.metrics.items() if name != "n_obs"
        )
        return f"Diagnostics(n_obs={len(self.observed)}, {metric_text})"

    def _repr_html_(self) -> str:
        metric_rows = "".join(
            "<tr>"
            f"<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>{escape(name)}</th>"
            f"<td style='padding:0.25rem 0;'>{value:.6g}</td>"
            "</tr>"
            for name, value in self.metrics.items()
        )
        return (
            "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>"
            "<h3 style='margin:0 0 0.5rem 0;'>Diagnostics</h3>"
            f"<p style='margin:0 0 0.5rem 0;'>{escape(self.formula)}</p>"
            f"<table style='border-collapse:collapse;'>{metric_rows}</table>"
            "</div>"
        )
