from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._binding import rust_module


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

    Top-level payload keys are also reachable as attributes: ``summary.formula``,
    ``summary.family_name``, ``summary.deviance``, ``summary.reml_score``, etc.
    are equivalent to ``summary["..."]`` and avoid an opaque ``AttributeError``
    on the natural attribute-style access. Subscript access remains the
    canonical / fully-documented form.

    Examples
    --------
    >>> summary = model.summary()
    >>> summary["family_name"]
    'Gaussian Identity'
    >>> summary.family_name  # attribute-style mirror of the subscript form
    'Gaussian Identity'
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

    def __getattr__(self, name: str) -> Any:
        # Python only calls __getattr__ when normal attribute lookup fails, so
        # the dataclass field ``payload`` is already handled by the slot
        # descriptor and never routes through here. Top-level payload keys
        # (``formula``, ``family_name``, ``deviance``, ``reml_score``, ...)
        # are surfaced as attributes for ergonomics; the repr advertises them
        # like dataclass fields, and a bare ``AttributeError`` with no hint is
        # an unhelpful papercut. Subscript access (``summary["key"]``) remains
        # the documented form. (issue #311)
        try:
            payload = object.__getattribute__(self, "payload")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return payload[name]
        except KeyError:
            raise AttributeError(
                f"'Summary' object has no attribute {name!r} "
                f"(top-level payload keys: {sorted(payload)!r})"
            ) from None

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
        return rust_module().summary_repr(self.payload)

    def _repr_html_(self) -> str:
        return rust_module().summary_html(self.payload)
