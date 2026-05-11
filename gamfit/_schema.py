from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class SchemaIssue:
    """A single schema-validation problem detected against the training schema.

    Key fields:

    - ``kind``: short tag describing the issue category (e.g. ``"missing"``,
      ``"type_mismatch"``).
    - ``message``: human-readable explanation.
    - ``column``: name of the offending column, when applicable.

    Examples
    --------
    >>> SchemaIssue(kind="missing", message="column 'age' is missing", column="age")
    SchemaIssue(kind='missing', message="column 'age' is missing", column='age')
    """

    kind: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class SchemaCheck:
    """Result of comparing serving data against a fitted model's training schema.

    Returned by :meth:`Model.check`. Truthy when the check passes
    (``ok=True`` with no issues); rendered as an HTML table in notebooks.

    Key fields:

    - ``ok``: ``True`` when the data matches the training schema.
    - ``issues``: tuple of :class:`SchemaIssue` records describing each
      detected problem (empty when ``ok`` is ``True``).

    Examples
    --------
    >>> check = model.check(serving_df)
    >>> if not check:
    ...     check.raise_for_error()
    """

    ok: bool
    issues: tuple[SchemaIssue, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SchemaCheck":
        """Build a :class:`SchemaCheck` from a raw payload dictionary.

        Parameters
        ----------
        payload : dict
            Mapping with keys ``ok`` (bool) and ``issues`` (list of dicts with
            ``kind``, ``message``, and optional ``column``).

        Returns
        -------
        SchemaCheck
            Parsed schema-check result.

        Examples
        --------
        >>> SchemaCheck.from_dict({"ok": True, "issues": []}).ok
        True
        """
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
        """Raise :class:`ValueError` if the schema check failed.

        Concatenates every issue message into a single ``ValueError``. A no-op
        when :attr:`ok` is ``True``.

        Raises
        ------
        ValueError
            If at least one :class:`SchemaIssue` is recorded.

        Examples
        --------
        >>> check = model.check(serving_df)
        >>> check.raise_for_error()  # raises ValueError on mismatch
        """
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
