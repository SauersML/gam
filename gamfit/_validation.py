from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ._binding import rust_module


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

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormulaValidation":
        """Build a :class:`FormulaValidation` from a raw payload dictionary.

        Parameters
        ----------
        payload : Mapping
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
        return rust_module().formula_validation_supported_by_python_json(
            json.dumps(self.to_dict())
        )

    def __repr__(self) -> str:
        return rust_module().formula_validation_repr_json(
            json.dumps(self.to_dict())
        )

    def _repr_html_(self) -> str:
        return rust_module().formula_validation_html_json(
            json.dumps(self.to_dict())
        )
