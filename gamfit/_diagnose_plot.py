"""``Model.diagnose`` and ``Model.plot`` implementations.

Extracted from ``_model.py`` so the public :class:`Model` shell keeps its
surface area small. These helpers take a :class:`Model` instance and
delegate the numeric work back through ``model.predict`` /
``model.formula`` / ``model.response_name``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ._diagnostics import Diagnostics
from ._tables import coerce_numeric_vector, table_columns

if TYPE_CHECKING:
    from ._model import Model


def diagnose(
    model: "Model",
    data: Any,
    *,
    y: str | None = None,
    interval: float | None = 0.95,
) -> Diagnostics:
    """Score the fitted ``model`` on held-out ``data``."""
    columns, _kind = table_columns(data)
    response_name = y or model.response_name
    if response_name is None:
        raise ValueError("could not infer the response column; pass y='column_name'")
    if response_name not in columns:
        raise ValueError(
            f"response column '{response_name}' is missing from the diagnostic data"
        )
    prediction_columns = {
        name: values for name, values in columns.items() if name != response_name
    }
    predicted = model.predict(
        prediction_columns,
        interval=interval,
        return_type="dict",
    )
    observed = coerce_numeric_vector(columns[response_name], label=response_name)
    return Diagnostics.from_predictions(
        formula=model.formula,
        response_name=response_name,
        observed=observed,
        predicted=predicted,
    )


def plot(
    model: "Model",
    data: Any,
    *,
    x: str | None = None,
    y: str | None = None,
    interval: float | None = 0.95,
    kind: str = "prediction",
    ax: Any | None = None,
) -> Any:
    """Plot the model's behaviour on ``data`` with matplotlib."""
    import matplotlib.pyplot as plt

    columns, _table_kind = table_columns(data)
    diagnostics = diagnose(
        model, data, y=y, interval=interval if kind == "prediction" else None
    )
    if ax is None:
        _, ax = plt.subplots()

    if kind == "prediction":
        response_name = diagnostics.response_name
        candidate_columns = [name for name in columns if name != response_name]
        x_name = x or (candidate_columns[0] if len(candidate_columns) == 1 else None)
        if x_name is None:
            raise ValueError(
                "prediction plots require x='column_name' when multiple feature columns are present"
            )
        if x_name not in columns:
            raise ValueError(f"plot column '{x_name}' is missing from the supplied data")
        x_values = coerce_numeric_vector(columns[x_name], label=x_name)
        ordering = sorted(range(len(x_values)), key=lambda index: x_values[index])
        x_sorted = [x_values[index] for index in ordering]
        mean_sorted = [diagnostics.predicted["mean"][index] for index in ordering]
        ax.plot(x_sorted, mean_sorted, color="#1d4ed8", linewidth=2, label="mean")
        if diagnostics.interval_lower is not None and diagnostics.interval_upper is not None:
            lower = [diagnostics.interval_lower[index] for index in ordering]
            upper = [diagnostics.interval_upper[index] for index in ordering]
            ax.fill_between(
                x_sorted, lower, upper, color="#93c5fd", alpha=0.35, label="interval"
            )
        if diagnostics.observed:
            observed_sorted = [diagnostics.observed[index] for index in ordering]
            ax.scatter(
                x_sorted,
                observed_sorted,
                color="#0f172a",
                s=18,
                alpha=0.7,
                label="observed",
            )
        ax.set_xlabel(x_name)
        ax.set_ylabel(diagnostics.response_name or "response")
    elif kind == "residuals":
        ax.scatter(
            diagnostics.predicted["mean"],
            diagnostics.residuals,
            color="#0f172a",
            s=18,
            alpha=0.75,
        )
        ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_xlabel("predicted mean")
        ax.set_ylabel("residual")
    elif kind == "observed_vs_predicted":
        ax.scatter(
            diagnostics.predicted["mean"],
            diagnostics.observed,
            color="#0f172a",
            s=18,
            alpha=0.75,
        )
        lo = min(min(diagnostics.predicted["mean"]), min(diagnostics.observed))
        hi = max(max(diagnostics.predicted["mean"]), max(diagnostics.observed))
        ax.plot([lo, hi], [lo, hi], color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_xlabel("predicted mean")
        ax.set_ylabel("observed")
    else:
        raise ValueError(
            "plot kind must be one of: prediction, residuals, observed_vs_predicted"
        )

    ax.set_title(f"{model.family_name} ({kind.replace('_', ' ')})")
    if kind == "prediction":
        ax.legend()
    return ax


__all__ = ["diagnose", "plot"]
