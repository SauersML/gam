from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class SchemaIssue:
    kind: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class SchemaCheck:
    ok: bool
    issues: tuple[SchemaIssue, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SchemaCheck":
        issues = tuple(
            SchemaIssue(
                kind=str(entry.get("kind", "issue")),
                message=str(entry.get("message", "")),
                column=None if entry.get("column") is None else str(entry.get("column")),
            )
            for entry in payload.get("issues", [])
        )
        return cls(ok=bool(payload.get("ok", not issues)), issues=issues)

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"SchemaCheck(ok={self.ok}, issues={len(self.issues)})"

    def raise_for_error(self) -> None:
        if self.ok:
            return
        messages = "; ".join(issue.message for issue in self.issues)
        raise ValueError(messages)

    def _repr_html_(self) -> str:
        if not self.issues:
            return (
                "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>"
                "<strong>SchemaCheck</strong><p style='margin:0.5rem 0 0 0;'>OK</p></div>"
            )
        rows = "".join(
            "<tr>"
            f"<td>{escape(issue.kind)}</td>"
            f"<td>{escape(issue.column or '')}</td>"
            f"<td>{escape(issue.message)}</td>"
            "</tr>"
            for issue in self.issues
        )
        return (
            "<div style='font-family: ui-sans-serif, system-ui, sans-serif;'>"
            "<strong>SchemaCheck</strong>"
            "<table style='margin-top:0.5rem;border-collapse:collapse;'>"
            "<thead><tr><th>kind</th><th>column</th><th>message</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
