from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

_HTML_EXCLUDED_KEYS = frozenset({"coefficients", "covariance_flat"})
_HTML_COEFFICIENT_LIMIT = 50
_VALUE_PREVIEW_LIMIT = 6


@dataclass(frozen=True, slots=True)
class Summary:
    """Frozen view of a fitted-model summary payload.

    A ``Summary`` is the structured equivalent of ``print(model)`` for a fitted
    GAM. It wraps a plain ``dict`` returned by the Rust engine and exposes
    convenient accessors plus a notebook-friendly HTML representation. The
    typical entry point is :meth:`Model.summary`.

    The payload typically contains keys such as ``formula``, ``family_name``,
    ``model_class``, ``deviance``, ``reml_score``, and ``coefficients`` (a list
    of per-coefficient dictionaries). Use :meth:`coefficients_frame` to view the
    coefficient table as a pandas DataFrame.

    Examples
    --------
    >>> summary = model.summary()
    >>> summary["family_name"]
    'gaussian'
    >>> summary.coefficients_frame().head()
    """

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Summary":
        """Build a :class:`Summary` from a raw payload dictionary.

        Parameters
        ----------
        payload : dict
            Mapping of summary keys to values, as produced by the Rust engine.

        Returns
        -------
        Summary
            A new immutable summary view over a shallow copy of ``payload``.

        Examples
        --------
        >>> Summary.from_dict({"formula": "y ~ s(x)", "family_name": "gaussian"})
        Summary(formula='y ~ s(x)', family_name='gaussian')
        """
        return cls(payload=dict(payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``payload[key]`` if present, otherwise ``default``.

        Parameters
        ----------
        key : str
            Payload key to look up.
        default : Any, optional
            Value returned when ``key`` is not in the payload.

        Returns
        -------
        Any
            The looked-up value, or ``default`` when ``key`` is absent.

        Examples
        --------
        >>> summary.get("deviance", float("nan"))
        12.34
        """
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the underlying payload dictionary.

        Returns
        -------
        dict
            Plain ``dict`` mirror of the summary payload, safe to mutate.

        Examples
        --------
        >>> raw = summary.to_dict()
        >>> sorted(raw)[:3]
        ['coefficients', 'deviance', 'family_name']
        """
        return dict(self.payload)

    @property
    def coefficients(self) -> list[dict[str, Any]]:
        """List of per-coefficient records.

        Returns
        -------
        list of dict
            One record per fitted coefficient, each with keys such as
            ``index``, ``estimate``, and ``std_error`` depending on the model.

        Examples
        --------
        >>> summary.coefficients[0]["index"]
        0
        """
        return list(self.payload.get("coefficients", []))

    def coefficients_frame(self) -> Any:
        """Return :attr:`coefficients` as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            One row per coefficient, columns mirror the keys in
            :attr:`coefficients` records.

        Examples
        --------
        >>> frame = summary.coefficients_frame()
        >>> frame.columns.tolist()[:2]
        ['index', 'estimate']
        """
        import pandas as pd

        return pd.DataFrame(self.coefficients)

    def __repr__(self) -> str:
        fields = []
        for key in ("formula", "family_name", "model_class", "deviance", "reml_score"):
            if key in self.payload:
                fields.append(f"{key}={self.payload[key]!r}")
        return f"Summary({', '.join(fields)})"

    def _repr_html_(self) -> str:
        rows = "".join(
            "<tr>"
            f"<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>{escape(str(key))}</th>"
            f"<td style='padding:0.25rem 0;'>{escape(_render_value(value))}</td>"
            "</tr>"
            for key, value in self.payload.items()
            if key not in _HTML_EXCLUDED_KEYS
        )
        coefficient_table = _render_coefficients_html(self.coefficients)
        return (
            "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>"
            "<h3 style='margin:0 0 0.5rem 0;'>Model Summary</h3>"
            f"<table style='border-collapse:collapse;'>{rows}</table>"
            f"{coefficient_table}"
            "</div>"
        )


def _render_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        return _render_mapping_value(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _render_sequence_value(value)
    return str(value)


def _render_mapping_value(value: Mapping[Any, Any]) -> str:
    if not value:
        return "{}"
    items = list(value.items())
    preview = ", ".join(
        f"{key}: {_render_value(item_value)}"
        for key, item_value in items[:_VALUE_PREVIEW_LIMIT]
    )
    suffix = "" if len(items) <= _VALUE_PREVIEW_LIMIT else f", ... ({len(items)} total)"
    return "{" + preview + suffix + "}"


def _render_sequence_value(value: Sequence[Any]) -> str:
    if not value:
        return "[]"
    preview = ", ".join(_render_value(item) for item in value[:_VALUE_PREVIEW_LIMIT])
    suffix = "" if len(value) <= _VALUE_PREVIEW_LIMIT else f", ... ({len(value)} total)"
    return "[" + preview + suffix + "]"


def _render_coefficients_html(coefficients: list[dict[str, Any]]) -> str:
    if not coefficients:
        return ""
    columns = _coefficient_columns(coefficients)
    header_cells = "".join(
        "<th style='text-align:right;padding:0.25rem 0.75rem;"
        f"border-bottom:1px solid #ddd;'>{escape(column)}</th>"
        for column in columns
    )
    body_rows = "".join(
        "<tr>"
        + "".join(
            "<td style='text-align:right;padding:0.25rem 0.75rem;'>"
            f"{escape(_render_value(row.get(column, '')))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in coefficients[:_HTML_COEFFICIENT_LIMIT]
    )
    omitted = len(coefficients) - _HTML_COEFFICIENT_LIMIT
    note = (
        "<p style='margin:0.25rem 0 0 0;color:#666;'>"
        f"Showing first {_HTML_COEFFICIENT_LIMIT} of {len(coefficients)} "
        "coefficients.</p>"
        if omitted > 0
        else ""
    )
    return (
        "<h4 style='margin:1rem 0 0.35rem 0;'>Coefficients</h4>"
        "<table style='border-collapse:collapse;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        "</table>"
        f"{note}"
    )


def _coefficient_columns(coefficients: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in coefficients:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns
