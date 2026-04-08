from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class FormulaValidation:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormulaValidation":
        return cls(payload=dict(payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def supported_by_python(self) -> bool:
        return bool(self.payload.get("supported_by_python", False))

    def __repr__(self) -> str:
        return (
            "FormulaValidation("
            f"formula={self.payload.get('formula')!r}, "
            f"model_class={self.payload.get('model_class')!r}, "
            f"family_name={self.payload.get('family_name')!r}, "
            f"supported_by_python={self.supported_by_python!r})"
        )

    def _repr_html_(self) -> str:
        rows = "".join(
            "<tr>"
            f"<th style='text-align:left;padding:0.25rem 0.75rem 0.25rem 0;'>{escape(str(key))}</th>"
            f"<td style='padding:0.25rem 0;'>{escape(str(value))}</td>"
            "</tr>"
            for key, value in self.payload.items()
        )
        return (
            "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>"
            "<h3 style='margin:0 0 0.5rem 0;'>Formula Validation</h3>"
            f"<table style='border-collapse:collapse;'>{rows}</table>"
            "</div>"
        )
