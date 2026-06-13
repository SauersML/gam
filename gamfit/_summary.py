"""Typed view of a fitted-model summary.

The Rust engine emits a structured summary payload (formula, family, deviance,
REML score, ...). This module exposes that payload as a frozen dataclass with
named typed fields so IDEs / type-checkers see real attributes, not a dict
passthrough. Unknown keys (added by future engine versions) survive on
:attr:`Summary.extras` for forward-compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Mapping

from ._binding import rust_module


# Canonical schema mirrors ``SummaryPayload`` in
# ``crates/gam-pyffi/src/lib.rs::summary_json_impl``. Adding a new field on the
# Rust side that should be a typed Summary attribute means adding it here too;
# unknown keys land in :attr:`Summary.extras` rather than silently going
# missing.
_SUMMARY_FIELDS: tuple[str, ...] = (
    "formula",
    "family_name",
    "model_class",
    "deviance",
    "reml_score",
    "raw_reml_score",
    "null_space_logdet",
    "null_dim",
    "iterations",
    "edf_total",
    "lambdas",
    "coefficients",
    "smooth_terms",
    "curvature_estimands",
    "covariance_kind",
    "covariance_n",
    "covariance_flat",
    "group_metadata",
    "deployment_extensions",
)


@dataclass(frozen=True, slots=True)
class Summary:
    """Frozen, typed view of a fitted-model summary.

    Each attribute mirrors a field of the Rust ``SummaryPayload`` struct.
    Subscript access (``summary["formula"]``) is supported for callers that
    prefer dict-style iteration; the canonical access is attribute style
    (``summary.formula``) which is fully typed and works with IDE
    autocomplete.

    Attributes
    ----------
    formula : str
        The Wilkinson formula string the model was fitted with.
    family_name : str
        Human-readable family + link label, e.g. ``"Gaussian Identity"``.
    model_class : str
        Internal model class, e.g. ``"standard"`` / ``"marginal-slope"``.
    deviance : float or None
        Model deviance at the converged fit. ``None`` for models that do not
        report a deviance.
    reml_score : float or None
        Comparable REML / LAML cost at convergence, including the
        rank-aware Tierney-Kadane null-space normalizer when available.
    raw_reml_score : float or None
        Raw outer-loop REML / LAML cost before the null-space normalizer.
    null_space_logdet : float or None
        Log-determinant of the null-space penalty Gram block; used by the
        evidence calculation.
    null_dim : float or None
        Dimension of the penalty null space.
    iterations : int or None
        Outer-loop iteration count.
    edf_total : float or None
        Total effective degrees of freedom across all blocks.
    lambdas : list of float
        Fitted smoothing / precision parameters in penalty-block order.
    coefficients : list of dict
        One record per fitted coefficient with keys ``index``, ``estimate``,
        and ``std_error`` (when available).
    smooth_terms : list of dict
        The mgcv-style per-smooth significance table: one record per
        smooth / random-effect term with keys ``name``, ``edf``, ``ref_df``,
        and — for penalized smooths — ``chi_sq`` (Wood 2013 rank-truncated
        Wald statistic) and ``p_value``. Random-effect smooths report ``edf``
        only. Empty when the model has no smooth terms or when the design
        could not be reconstructed to recover per-term coefficient blocks.

        This ``p_value`` is the *first-order* Wald reference; computing it needs
        only the saved model. For the **second-order-accurate**, Bartlett-corrected
        likelihood-ratio p-value (the exact Lawley factor auto-applied whenever the
        family carries closed-form cumulant jets, #939/#1063) call
        :meth:`Model.smooth_significance(data) <gamfit.Model.smooth_significance>`,
        which runs the per-term constrained refits the saved-model summary cannot.
    covariance_kind : str or None
        ``"corrected"`` or ``"conditional"`` depending on which posterior
        covariance variant was returned.
    covariance_n : int or None
        Side length of the coefficient covariance matrix.
    covariance_flat : list of float or None
        Row-major flat coefficient covariance matrix.
    group_metadata : dict or None
        Saved group-level metadata for grouped fits.
    deployment_extensions : list of dict
        Post-fit deployment extensions attached via :meth:`Model.extend_with_group`.
    extras : dict
        Any keys returned by the Rust engine that are not in the typed
        schema. Kept so newer engine versions can add fields without
        breaking older Python wheels.

    Examples
    --------
    >>> summary = model.summary()
    >>> summary.family_name
    'Gaussian Identity'
    >>> summary["family_name"]  # subscript form, equivalent
    'Gaussian Identity'
    >>> summary.coefficients_frame().head()
    """

    formula: str = ""
    family_name: str = ""
    model_class: str = ""
    deviance: float | None = None
    reml_score: float | None = None
    raw_reml_score: float | None = None
    null_space_logdet: float | None = None
    null_dim: float | None = None
    iterations: int | None = None
    edf_total: float | None = None
    lambdas: list[float] = field(default_factory=list)
    coefficients: list[dict[str, Any]] = field(default_factory=list)
    smooth_terms: list[dict[str, Any]] = field(default_factory=list)
    #: Fitted curvature κ̂ point estimates for any ``curv(...)`` constant-curvature
    #: smooths (#944): one dict per term with ``name``, ``term_idx``,
    #: ``kappa_hat``, and a sign-of-κ̂ ``geometry`` tag. The profile CI and the
    #: κ = 0 flatness p-value (which need a refit) come from ``Model.curvature``.
    curvature_estimands: list[dict[str, Any]] = field(default_factory=list)
    covariance_kind: str | None = None
    covariance_n: int | None = None
    covariance_flat: list[float] | None = None
    group_metadata: dict[str, Any] | None = None
    deployment_extensions: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Summary":
        """Build a :class:`Summary` from a raw payload mapping.

        Known keys (see :data:`_SUMMARY_FIELDS`) become typed fields; any
        unrecognised keys land on :attr:`extras` so forward-compat is
        preserved without papering over breaking changes.

        Parameters
        ----------
        payload : Mapping[str, Any]
            Engine-produced summary mapping.

        Returns
        -------
        Summary
            Immutable typed summary.

        Examples
        --------
        >>> Summary.from_dict({"formula": "y ~ s(x)", "family_name": "Gaussian Identity"})
        Summary(formula='y ~ s(x)', family_name='Gaussian Identity')
        """
        known = {name: payload[name] for name in _SUMMARY_FIELDS if name in payload}
        extras = {key: value for key, value in payload.items() if key not in _SUMMARY_FIELDS}
        return cls(**known, extras=extras)

    # -- mapping-style accessors ------------------------------------------------
    # Subscript / membership / iteration are provided so callers using the older
    # dict-style API keep working through the schema migration. The canonical
    # form remains attribute access.

    def __getitem__(self, key: str) -> Any:
        if key in _SUMMARY_FIELDS:
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            if key in _SUMMARY_FIELDS:
                return True
            return key in self.extras
        return False

    def __iter__(self) -> Iterator[str]:
        yield from _SUMMARY_FIELDS
        yield from self.extras

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self[key]`` if present, otherwise ``default``."""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Flatten typed fields + extras into a single plain ``dict``."""
        out: dict[str, Any] = {name: getattr(self, name) for name in _SUMMARY_FIELDS}
        out.update(self.extras)
        return out

    def coefficients_frame(self) -> Any:
        """Return :attr:`coefficients` as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(self.coefficients)

    def smooth_terms_frame(self) -> Any:
        """Return :attr:`smooth_terms` as a :class:`pandas.DataFrame`.

        This is the canonical mgcv ``summary.gam`` per-smooth significance
        table: columns ``name``, ``edf``, ``ref_df``, ``chi_sq``, ``p_value``
        (``chi_sq`` / ``p_value`` are absent for random-effect smooths and any
        shape-constrained term, matching the engine, which only computes the
        Wood Wald test for ordinary penalized smooths).
        """
        import pandas as pd

        return pd.DataFrame(self.smooth_terms)

    # -- presentation -----------------------------------------------------------

    def __str__(self) -> str:
        """Multi-line human-readable summary (the ``print(summary)`` form).

        ``Model.__str__`` delegates here so the rendering lives in one place.
        """
        lines = ["GAM fitted model"]
        if self.formula:
            lines.append(f"  Formula: {self.formula}")
        if self.family_name:
            lines.append(f"  Family:  {self.family_name}")
        if self.model_class:
            lines.append(f"  Class:   {self.model_class}")
        if self.deviance is not None:
            lines.append(f"  Deviance: {self.deviance:g}")
        if self.reml_score is not None:
            lines.append(f"  REML score: {self.reml_score:g}")
        if self.edf_total is not None:
            lines.append(f"  Effective dof: {self.edf_total:g}")
        if self.iterations is not None:
            lines.append(f"  Outer iterations: {self.iterations}")
        n_coef = len(self.coefficients)
        if n_coef:
            lines.append(f"  Coefficients: {n_coef}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact developer one-liner. Stable across engine versions."""
        # Delegate to the Rust-side renderer for the canonical compact form
        # used in test snapshots / logs. The Rust renderer indexes by name so
        # it works against any dict; pass the typed view as a plain dict.
        return rust_module().summary_repr(self.to_dict())

    def _repr_html_(self) -> str:
        return rust_module().summary_html(self.to_dict())


def summary_field_names() -> tuple[str, ...]:
    """Public accessor for the canonical Summary field names.

    Used by ``Model.__str__`` and other call sites that want to walk the
    schema without depending on the private constant.
    """
    return _SUMMARY_FIELDS


__all__ = ["Summary", "summary_field_names"]


# Sanity-check at import: dataclass fields must mirror _SUMMARY_FIELDS (plus
# ``extras`` for the unknown-key bucket). If someone adds a typed field but
# forgets to register it in _SUMMARY_FIELDS the from_dict path would silently
# drop it; fail loudly instead.
_declared = tuple(f.name for f in fields(Summary) if f.name != "extras")
if _declared != _SUMMARY_FIELDS:
    raise RuntimeError(
        "Summary dataclass fields drifted from _SUMMARY_FIELDS: "
        f"dataclass={_declared!r} canonical={_SUMMARY_FIELDS!r}"
    )
