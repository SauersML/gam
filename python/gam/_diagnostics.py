from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class Diagnostics:
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
