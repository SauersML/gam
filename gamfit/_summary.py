"""Typed view of a fitted-model summary.

The Rust engine emits a structured summary payload (formula, family, deviance,
REML score, ...). This module exposes that payload as a frozen dataclass with
named typed fields so IDEs / type-checkers see real attributes, not a dict
passthrough. Unknown keys (added by future engine versions) survive on
:attr:`Summary.extras` for forward-compat.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Mapping, overload

from ._binding import rust_module


#: Columns of :meth:`Summary.smooth_terms_frame`, in the documented order.
_SMOOTH_TERM_COLUMNS: tuple[str, ...] = ("name", "edf", "ref_df", "chi_sq", "p_value")

# Canonical schema mirrors ``SummaryPayload`` in
# ``crates/gam-models/src/inference/saved_summary.rs``. Adding a new field on the
# Rust side that should be a typed Summary attribute means adding it here too;
# unknown keys land in :attr:`Summary.extras` rather than silently going
# missing.
_SUMMARY_FIELDS: tuple[str, ...] = (
    "formula",
    "family_name",
    "model_class",
    "n_obs",
    "deviance",
    "log_likelihood",
    "reml_score",
    "raw_reml_score",
    "reml_score_unavailable",
    "null_space_logdet",
    "null_dim",
    "iterations",
    "edf_total",
    "edf_rank_bound",
    "lambdas",
    "coefficients",
    "smooth_terms",
    "smooth_terms_unavailable",
    "curvature_estimands",
    "basis_checks",
    "covariance_kind",
    "covariance_n",
    "covariance_flat",
    "coefficient_se_source",
    "group_metadata",
    "deployment_extensions",
    "convergence",
)


class _ColumnarCoefficientRecords(Sequence[Mapping[str, Any]]):
    """Lazy record view over typed posterior coefficient columns.

    Posterior means, standard errors, and credible intervals stay in their
    NumPy buffers until a caller asks for an individual record.  This keeps a
    posterior summary proportional to its numeric columns instead of eagerly
    allocating one Python ``dict`` (and five boxed floats) per coefficient.
    """

    __slots__ = ("_names", "_estimates", "_std_errors", "_intervals")

    def __init__(
        self,
        names: Sequence[str],
        estimates: Any,
        std_errors: Any,
        intervals: Any,
    ) -> None:
        count = len(names)
        if getattr(estimates, "ndim", None) != 1 or len(estimates) != count:
            raise ValueError(
                "posterior_mean must be a one-dimensional array with one "
                f"value per coefficient; got shape {getattr(estimates, 'shape', None)} "
                f"for {count} coefficients"
            )
        if getattr(std_errors, "ndim", None) != 1 or len(std_errors) != count:
            raise ValueError(
                "posterior_std must be a one-dimensional array with one "
                f"value per coefficient; got shape {getattr(std_errors, 'shape', None)} "
                f"for {count} coefficients"
            )
        if getattr(intervals, "ndim", None) != 2 or tuple(intervals.shape) != (count, 2):
            raise ValueError(
                "posterior credible intervals must have shape "
                f"({count}, 2); got {getattr(intervals, 'shape', None)}"
            )
        self._names = names
        self._estimates = estimates
        self._std_errors = std_errors
        self._intervals = intervals

    def __len__(self) -> int:
        return len(self._names)

    def _record(self, index: int) -> dict[str, Any]:
        position = operator.index(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("coefficient index out of range")
        return {
            "index": position,
            "name": self._names[position],
            "estimate": float(self._estimates[position]),
            "std_error": float(self._std_errors[position]),
            "ci_lower": float(self._intervals[position, 0]),
            "ci_upper": float(self._intervals[position, 1]),
        }

    @overload
    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Mapping[str, Any]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(index, slice):
            return [self._record(position) for position in range(*index.indices(len(self)))]
        return self._record(index)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for index in range(len(self)):
            yield self._record(index)

    def columns(self) -> Mapping[str, Any]:
        """Expose columns directly for dataframe construction."""
        return {
            "index": range(len(self)),
            "name": self._names,
            "estimate": self._estimates,
            "std_error": self._std_errors,
            "ci_lower": self._intervals[:, 0],
            "ci_upper": self._intervals[:, 1],
        }


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
    n_obs : int or None
        Number of original training rows. Fitted-model summaries always carry
        this persisted value; non-model summaries such as posterior-draw
        summaries leave it unset.
    deviance : float or None
        Model deviance at the converged fit. ``None`` for models that do not
        report a deviance.
    log_likelihood : float or None
        Ordinary reported log-likelihood at the converged fit. Every rankable
        model carries a finite value; ``None`` is reserved for the exact
        zero-dispersion boundary where no normalized Gaussian density exists.
    reml_score : float or None
        Comparable REML / LAML cost at convergence, including the
        rank-aware Tierney-Kadane null-space normalizer. ``None`` in two cases,
        each named by :attr:`reml_score_unavailable`, and never replaced by a
        stand-in value:

        - the fit has **no** criterion at all (:attr:`raw_reml_score` is
          ``None`` too), which is not the same as one that was not recorded: an
          exactly-interpolating Gaussian fit has :math:`\\hat\\varphi = 0`, so
          its restricted likelihood is unbounded and every score derived from
          it is undefined;
        - the fit has a raw criterion but no penalty null-space metadata, so
          the normalizer cannot be formed and no score comparable across fits
          exists (#2627).
    raw_reml_score : float or None
        Raw outer-loop REML / LAML cost before the null-space normalizer.
        ``None`` exactly when the fit has no criterion at all.
    reml_score_unavailable : str or None
        Why :attr:`reml_score` is ``None``. Present exactly when
        :attr:`reml_score` is ``None``, so an absence is never reported without
        its reason.
    null_space_logdet : float or None
        Log-determinant of the null-space penalty Gram block; used by the
        evidence calculation.
    null_dim : float or None
        Dimension of the penalty null space.
    iterations : int or None
        Outer-loop iteration count.
    edf_total : float or None
        Total effective degrees of freedom across all blocks.
    edf_rank_bound : list of mapping
        Each penalty block's rank-bound status (#2901): ``{"Certified": ...}`` when
        the block's trace is certified to lie in ``[0, rank]``;
        ``{"Uncertified": {"smallest_pivot": ..., "band": ...}}`` (rank bound not
        certified) when the factorization of the data curvature against the
        penalty has a negative pivot; or ``{"NotAssessed": {"reason": ...}}``
        (rank bound not assessed) when the memory governor refused the
        certificate's workspace. A block that is not certified publishes its
        trace, its EDF and ``edf_total`` unclamped. Empty when the fit recorded
        none.
    lambdas : list of float
        Fitted smoothing / precision parameters in penalty-block order.
    coefficients : sequence of mappings
        One record per fitted coefficient with keys ``index``, ``estimate``,
        and ``std_error`` (when available). Posterior summaries use a lazy
        columnar sequence so indexing and iteration do not require an eager
        list of per-coefficient dictionaries.
    smooth_terms : list of dict
        The mgcv-style per-smooth significance table: one record per
        smooth / random-effect term with keys ``name``, ``edf``, ``ref_df``,
        and — for penalized smooths — ``chi_sq`` (Wood 2013 rank-truncated
        Wald statistic) and ``p_value``. Random-effect smooths report ``edf``
        only. Empty when the model has no smooth or random-effect terms; every
        other absence is labeled by :attr:`smooth_terms_unavailable`.
    smooth_terms_unavailable : str or None
        Why :attr:`smooth_terms` could not be built (a model saved without its
        frozen term spec or training ranges, or a frozen-basis design replay
        that failed). Present exactly when the table was omitted for such a
        reason, so an absence is never reported without its reason — the same
        contract as :attr:`reml_score_unavailable`.

        This ``p_value`` is the *first-order* Wald reference; computing it needs
        only the saved model. For the **second-order-accurate**, Bartlett-corrected
        likelihood-ratio p-value (the exact Lawley factor auto-applied whenever the
        family carries closed-form cumulant jets, #939/#1063) call
        :meth:`Model.smooth_significance(data) <gamfit.Model.smooth_significance>`,
        which runs the per-term constrained refits the saved-model summary cannot.
    covariance_kind : str or None
        ``"smoothing-corrected"`` or ``"conditional"`` depending on which
        posterior covariance variant was returned. The kind, the ``std_error``
        column, and ``covariance_flat`` always come from the SAME covariance
        definition (#2296); see ``coefficient_se_source``.
    covariance_n : int or None
        Side length of the coefficient covariance matrix.
    covariance_flat : list of float or None
        Row-major flat coefficient covariance matrix.
    group_metadata : dict or None
        Saved group-level metadata for grouped fits.
    deployment_extensions : list of dict
        Post-fit deployment extensions attached via :meth:`Model.extend_with_group`.
    basis_checks : list of dict
        Per-smooth basis-adequacy evidence (#2774) measured at fit time: one
        record per smooth term with ``name``, ``term_idx``, ``basis_dim`` (the
        realized ``k'``), ``nullspace_dim``, ``edf``, ``enrichment_dim``,
        ``enrichment_rank``, ``statistic``, ``p_value`` and ``provenance``. A
        small ``p_value`` says the fit's residuals still carry structure in that
        smooth's covariates which its realized basis cannot represent. See
        :meth:`gamfit.Model.basis_check` for the construction and its limits.
        Empty when the fit recorded none.
    convergence : dict or None
        How the optimization that produced this fit terminated, read from the
        certificate the fit carries rather than recomputed.

        **``certified`` is a statement about the OPTIMIZER, not about the
        model.** It says the inner P-IRLS solve and the outer smoothing-parameter
        search reached a certified stationary point at the tolerances the mint
        gate applies. It makes no claim that the basis those iterations
        converged on is rich enough to represent the function it was asked to
        model, that the family is right, or that the fitted adjustment removes
        the confounding it was given (#2774). :attr:`basis_checks` answers the
        first of those; nothing here answers any of them.

        ``certified`` is the
        verdict the mint gate used; ``inner_status`` is the terminal P-IRLS
        status; ``outer_iterations`` is the iteration count the proof covers;
        ``outer`` is ``None`` when no smoothing coordinate was optimized (there
        is no outer stationarity equation, which is *not* the same as a zero
        gradient), else a mapping carrying ``kind``, ``gradient_norm``,
        ``projected_gradient_norm``, ``stationarity_bound``, ``hessian_psd`` and
        ``lambdas_railed``. The projected norm and the bound are in the same
        gauge, so a caller can impose a tolerance of their own without reading a
        log. ``None`` for routes that certify no optimizer (the O(n) spline
        scan).
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
    n_obs: int | None = None
    deviance: float | None = None
    log_likelihood: float | None = None
    reml_score: float | None = None
    raw_reml_score: float | None = None
    reml_score_unavailable: str | None = None
    null_space_logdet: float | None = None
    null_dim: float | None = None
    iterations: int | None = None
    edf_total: float | None = None
    #: Per-block rank-bound status beside the EDF fields (#2901).
    edf_rank_bound: list[Any] = field(default_factory=list)
    lambdas: list[float] = field(default_factory=list)
    coefficients: Sequence[Mapping[str, Any]] = field(default_factory=list)
    smooth_terms: list[dict[str, Any]] = field(default_factory=list)
    smooth_terms_unavailable: str | None = None
    #: Fitted curvature κ̂ point estimates for any ``curv(...)`` constant-curvature
    #: smooths (#944): one dict per term with ``name``, ``term_idx``,
    #: ``kappa_hat``, and a sign-of-κ̂ ``geometry`` tag. The profile CI and the
    #: κ = 0 flatness p-value (which need a refit) come from ``Model.curvature``.
    curvature_estimands: list[dict[str, Any]] = field(default_factory=list)
    #: Per-smooth basis-adequacy evidence (#2774), measured at fit time and
    #: persisted with the model: one dict per smooth term with ``name``,
    #: ``term_idx``, ``basis_dim`` (the realized ``k'``), ``nullspace_dim``,
    #: ``edf``, ``enrichment_dim`` / ``enrichment_rank``, ``statistic``,
    #: ``p_value`` and ``provenance``.
    #:
    #: A small ``p_value`` says the fit's residuals still carry structure in that
    #: smooth's covariates which its realized basis cannot represent — i.e. the
    #: basis is too small, not that the smoothing parameter is too large (the
    #: statistic is constructed to be blind to shrinkage; see
    #: :meth:`Model.basis_check`). ``p_value`` is present exactly when
    #: ``provenance == "radial_enrichment"``; every other provenance value names
    #: the evidence that was missing, so "adequate" and "not measured" are never
    #: confusable. Empty for models saved before the check existed.
    #:
    #: This is a DIFFERENT question from :attr:`convergence`: ``certified`` is a
    #: statement about the optimizer, and says nothing about whether the basis it
    #: converged on was rich enough.
    basis_checks: list[dict[str, Any]] = field(default_factory=list)
    covariance_kind: str | None = None
    covariance_n: int | None = None
    covariance_flat: list[float] | None = None
    #: Exact covariance definition behind the coefficient ``std_error`` column
    #: (#2296): ``"conditional"`` or ``"smoothing-corrected"``, recorded from
    #: the definition-consistent pair the engine summary actually consumed.
    coefficient_se_source: str | None = None
    group_metadata: dict[str, Any] | None = None
    deployment_extensions: list[dict[str, Any]] = field(default_factory=list)
    #: How the optimization that produced this fit terminated (#2411), read
    #: from the certificate the fit itself carries. Keys: ``certified``,
    #: ``inner_status``, ``outer_iterations``, and ``outer`` — the last being
    #: ``None`` when no smoothing coordinate was optimized, else a mapping with
    #: ``kind``, ``gradient_norm``, ``projected_gradient_norm``,
    #: ``stationarity_bound``, ``hessian_psd`` and ``lambdas_railed``.
    convergence: dict[str, Any] | None = None
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
        """Flatten typed fields + extras into a single plain ``dict``.

        This is the explicit materialization boundary for lazy posterior
        coefficient records.
        """
        out: dict[str, Any] = {name: getattr(self, name) for name in _SUMMARY_FIELDS}
        if isinstance(self.coefficients, _ColumnarCoefficientRecords):
            out["coefficients"] = list(self.coefficients)
        out.update(self.extras)
        return out

    def coefficients_frame(self) -> Any:
        """Return :attr:`coefficients` as a :class:`pandas.DataFrame`."""
        import pandas as pd

        if isinstance(self.coefficients, _ColumnarCoefficientRecords):
            return pd.DataFrame(self.coefficients.columns(), copy=False)
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

        if not self.smooth_terms:
            # Zero rows keep the documented schema: an empty table with no
            # column headers is what a swallowed replay failure looked like
            # (#2787), and it is not the same object as an empty table.
            return pd.DataFrame(columns=list(_SMOOTH_TERM_COLUMNS))
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
        if self.n_obs is not None:
            lines.append(f"  Training rows: {self.n_obs}")
        if self.deviance is not None:
            lines.append(f"  Deviance: {self.deviance:g}")
        if (
            self.reml_score is not None
            or self.raw_reml_score is not None
            or self.reml_score_unavailable is not None
        ):
            # gam-report owns the words for an absent criterion, so this line
            # names a fit without null-space metadata apart from an exact fit
            # the same way `gam fit` and the HTML report do (#2627).
            lines.append(
                f"  REML score: {rust_module().summary_criterion_row(self.to_dict())}"
            )
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
