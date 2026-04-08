from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class Summary:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Summary":
        return cls(payload=dict(payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def coefficients(self) -> list[dict[str, Any]]:
        return list(self.payload.get("coefficients", []))

    def coefficients_frame(self) -> Any:
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
            if key != "coefficients"
        )
        return (
            "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>"
            "<h3 style='margin:0 0 0.5rem 0;'>Model Summary</h3>"
            f"<table style='border-collapse:collapse;'>{rows}</table>"
            "</div>"
        )


def _render_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
