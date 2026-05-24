from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True, slots=True)
class FormulaValidation:
    """Outcome of :func:`gamfit.validate_formula` (no fit performed).

    Wraps the JSON payload returned by the Rust validator. Typical keys include
    ``formula``, ``model_class``, ``family_name``, and ``supported_by_python``.
    Use this to confirm a formula parses, infer the family that would be
    picked, and check whether the Python binding can fit the resulting model
    before committing to a full :func:`gamfit.fit` call.

    Examples
    --------
    >>> info = gamfit.validate_formula(df, "y ~ s(x)")
    >>> info["family_name"]
    'gaussian'
    >>> info.supported_by_python
    True
    """

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormulaValidation":
        """Build a :class:`FormulaValidation` from a raw payload dictionary.

        Parameters
        ----------
        payload : dict
            Mapping of validation keys to values, as produced by the Rust
            validator.

        Returns
        -------
        FormulaValidation
            Immutable view over a shallow copy of ``payload``.

        Examples
        --------
        >>> FormulaValidation.from_dict({"formula": "y ~ x", "supported_by_python": True})
        FormulaValidation(formula='y ~ x', model_class=None, family_name=None, supported_by_python=True)
        """
        return cls(payload=dict(payload))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the underlying payload dictionary.

        Returns
        -------
        dict
            Plain ``dict`` mirror of the validation payload.

        Examples
        --------
        >>> raw = info.to_dict()
        >>> raw["formula"]
        'y ~ s(x)'
        """
        return dict(self.payload)

    @property
    def supported_by_python(self) -> bool:
        """Whether the Python binding can fit the validated model.

        Returns
        -------
        bool
            ``True`` when :func:`gamfit.fit` can produce a fitted model for
            this formula/family combination, ``False`` if only the CLI / Rust
            engine can handle it.

        Examples
        --------
        >>> info.supported_by_python
        True
        """
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
