"""Thin public facade for Rust-backed SAE manifold fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._binding import rust_module
from ._penalty_bridge import (
    GumbelTemperatureSchedule,
    validate_gumbel_schedule_fields as _validate_gumbel_schedule_fields,
)
from ._sparse_dictionary import sparse_dictionary_fit
from ._sae_trust import atom_trust_scores


class _SaeLazyFitArtifact:
    """Lazy backing store for dense SAE result arrays.

    The facade keeps this small handle instead of retaining duplicate dense
    result arrays in every public slot. Values may be arrays, callables, or
    sparse-code records shaped like ``{"indices", "values", "shape"}``.
    """

    __slots__ = ("_values", "_cache")

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = dict(values or {})
        self._cache: dict[str, Any] = {}

    def materialize(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        if name not in self._values:
            raise AttributeError(f"SAE lazy artifact has no field {name!r}")
        value = self._values[name]
        if callable(value):
            value = value()
        elif isinstance(value, Mapping) and {"indices", "values", "shape"} <= set(value):
            value = self._dense_from_sparse(value)
        self._cache[name] = value
        return value

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value
        self._cache.pop(name, None)

    @staticmethod
    def _dense_from_sparse(record: Mapping[str, Any]) -> np.ndarray:
        shape = tuple(int(v) for v in record["shape"])
        out = np.zeros(shape, dtype=float)
        indices = np.asarray(record["indices"])
        values = np.asarray(record["values"], dtype=float)
        if indices.shape != values.shape:
            raise ValueError(
                f"sparse SAE code indices {indices.shape} and values {values.shape} must match"
            )
        if len(shape) != 2 or indices.ndim != 2:
            raise ValueError(
                "sparse SAE dense reconstruction expects indices/values with shape (N, active)"
            )
        rows = np.arange(indices.shape[0])[:, None]
        out[rows, indices.astype(np.intp)] = values
        return out


class _SaeTrainingDataHandle:
    """Metadata-only public handle for training data that is not retained."""

    __slots__ = ("shape", "dtype")

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return 0

    @property
    def nbytes(self) -> int:
        return 0

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        raise ValueError(
            "ManifoldSAE.training_data is not retained. Pass activations explicitly "
            "or call reconstruct_training() to materialize the fitted reconstruction."
        )

    def copy(self) -> np.ndarray:
        return np.asarray(self).copy()

    def tolist(self) -> list[Any]:
        return np.asarray(self).tolist()

    def __repr__(self) -> str:
        return f"<training data not retained shape={self.shape} dtype={self.dtype}>"


def _training_data_handle(x: np.ndarray) -> _SaeTrainingDataHandle:
    return _SaeTrainingDataHandle(tuple(int(v) for v in x.shape), x.dtype)


def _sae_fit_admission(
    n_obs: int,
    output_dim: int,
    n_atoms: int,
    d_max: int = 1,
    topk_support: int | None = None,
) -> dict[str, Any]:
    return dict(
        rust_module().sae_fit_admission(
            int(n_obs),
            int(output_dim),
            int(n_atoms),
            int(d_max),
            None if topk_support is None else int(topk_support),
        )
    )


def _sae_manifold_reconstruct_native(
    atom_basis: list[str],
    atom_dims: list[int],
    decoder_blocks: list[np.ndarray],
    coords: list[np.ndarray],
    assignments: np.ndarray,
    p_out: int,
) -> np.ndarray:
    return np.ascontiguousarray(
        rust_module().sae_manifold_reconstruct_ffi(
            list(atom_basis),
            [int(dim) for dim in atom_dims],
            [np.ascontiguousarray(block, dtype=np.float64) for block in decoder_blocks],
            [np.ascontiguousarray(coord, dtype=np.float64) for coord in coords],
            np.ascontiguousarray(assignments, dtype=np.float64),
            int(p_out),
        ),
        dtype=np.float64,
    )


def _lazy_getattr(owner: Any, lazy_names: set[str], name: str) -> Any:
    if name in lazy_names:
        try:
            artifact = object.__getattribute__(owner, "_lazy_artifact")
        except AttributeError:
            artifact = None
        if artifact is not None:
            return artifact.materialize(name)
    return object.__getattribute__(owner, name)


def _lazy_setattr(owner: Any, lazy_names: set[str], name: str, value: Any) -> None:
    if name in lazy_names:
        try:
            artifact = object.__getattribute__(owner, "_lazy_artifact")
        except AttributeError:
            artifact = None
        if artifact is not None:
            artifact.set(name, value)
            object.__setattr__(owner, name, None)
            return
    object.__setattr__(owner, name, value)


_ATOM_LAZY_FIELDS = {"assignments", "coords"}
_LOW_LEVEL_LAZY_FIELDS = {"fitted", "assignments", "coords"}
_MODEL_LAZY_FIELDS = {"fitted", "assignments", "coords", "training_data", "low_level_logits"}


# #1777 — the hard-sigmoid gate family's primary token is ``"threshold_gate"``
# (the renamed Rust ``AssignmentMode::ThresholdGate``); the legacy ``"jumprelu"``
# spelling is retained as a deprecated alias that canonicalizes to it. The FFI
# accepts both tokens and emits ``"threshold_gate"``.
_ASSIGNMENT_KINDS: dict[str, str] = {
    "ibp_map": "ibp_map",
    "softmax": "softmax",
    "threshold_gate": "threshold_gate",
    "jumprelu": "threshold_gate",
    "topk": "topk",
}

_PUBLIC_ASSIGNMENT_KINDS: dict[str, str] = {
    "ibp_map": "ibp_map",
    "softmax": "softmax",
    "threshold_gate": "threshold_gate",
    "jumprelu": "threshold_gate",
    "topk": "topk",
}

# Public assignment alias table (#159). Both the ``assignment=`` and the
# ``assignment_prior=`` kwargs normalize through this single map so they can
# never validate differently. ``ibp``/``ibp-map``/``ibp_map`` all canonicalize
# to ``ibp_map``; ``threshold_gate`` (primary) and the deprecated aliases
# ``gated``/``jump_relu``/``jumprelu`` all canonicalize to ``threshold_gate`` (#1777).
_PUBLIC_ASSIGNMENT_ALIASES: dict[str, str] = {
    "ibp": "ibp_map",
    "ibp-map": "ibp_map",
    "ibp_map": "ibp_map",
    "softmax": "softmax",
    "threshold_gate": "threshold_gate",
    "gated": "threshold_gate",
    "jump_relu": "threshold_gate",
    "jumprelu": "threshold_gate",
    "topk": "topk",
    "top_k": "topk",
}


def _penalized_loss_score(payload: Mapping[str, Any]) -> float | None:
    """Read the SAE fit's penalized-loss score honestly (#1231).

    The Rust FFI surfaces the negative penalized loss under
    ``penalized_loss_score`` (in-sample) / ``oos_penalized_loss`` (fixed-decoder OOS).
    The closed-form Python shortcut payloads deliberately store ``None`` here (a
    reconstruction R² lives under ``reconstruction_r2`` instead, since R² is not
    the same quantity as the negative penalized loss), so ``None`` is a valid,
    tolerated value and is propagated as-is.
    """
    for key in ("penalized_loss_score", "oos_penalized_loss"):
        if key in payload:
            value = payload[key]
            return None if value is None else float(value)
    raise KeyError(
        "SAE fit payload is missing a penalized-loss score "
        "(penalized_loss_score / oos_penalized_loss)"
    )


def _active_threshold_for_assignment(assignment: str, k_atoms: int) -> float:
    """Per-assignment-kind 'active atom' threshold for the inclusive (>=) counter.

    The Rust ``sae_manifold_assignment_summary`` (and
    :meth:`ManifoldSAE.per_atom_active_set`) count an atom active when
    ``assignment >= threshold`` (INCLUSIVE). To realize the documented
    strictly-exclusive semantics we return the next representable value above
    each conceptual cutoff so the boundary case is NOT counted:

      * ``softmax``  -> active if its share strictly EXCEEDS the uniform mass
        ``1/K``; return ``nextafter(1/K, +inf)`` so an exactly-uniform row counts
        zero atoms (rather than all ``K``).
      * ``jumprelu`` -> active if the hard gate is NONZERO (> 0); return a tiny
        positive value so exact-zero gates are NOT counted.
      * ``ibp_map``  -> active if it carries responsibility mass above a small
        positive epsilon (``1e-8``), matching ``_closed_form_trust_diagnostics``
        (the normalized ``assignments_z`` responsibilities sum to ~1 per row, so
        a 0.5 bar would collapse ``avg_active_atoms`` once ``K>=2``; #1547).

    This is the single policy shared by :meth:`ManifoldSAE.summary` and
    :meth:`ManifoldSAE.per_atom_active_set` so they cannot drift.
    """
    canon = _canonical_assignment(assignment, "assignment")
    if canon == "softmax":
        return float(np.nextafter(1.0 / max(1, int(k_atoms)), np.inf))
    if canon == "threshold_gate":
        return float(np.finfo(float).tiny)
    # ibp_map
    return 1.0e-8


def _channel_cov_factors(
    decoder_covariance: np.ndarray | None, m_basis: int
) -> list | None:
    """Compact per-channel covariance factor a shape band consumes for save/load.

    The dense phi-scaled decoder covariance is ``(M_k·p, M_k·p)`` in row-major
    ``(basis, channel)`` flat layout (flat index ``b·p + c``). The posterior
    shape band reads ONLY the same-channel blocks ``Cov[(b1,c),(b2,c)]`` — its
    variance is ``Var_c(t) = Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]`` — so those
    ``p`` blocks of ``M_k × M_k`` are the complete, compact factor the band
    needs. This extracts them as a ``(p, M_k, M_k)`` array (pure reshaping /
    diagonal slicing — no numerical decomposition), replacing the dense
    ``(M_k·p)²`` joint covariance in the on-disk format. ``None`` when the fit
    carries no covariance (e.g. an LLM-scale ``p`` where even the fresh fit omits
    the dense lift); the band's stored ``shape_band_sd`` still round-trips.
    """
    if decoder_covariance is None:
        return None
    cov = np.ascontiguousarray(np.asarray(decoder_covariance, dtype=float))
    # Reshaping / diagonal-slicing math (#2091) lives in the Rust owner. `None`
    # (layout does not match an (M_k, p) decoder) round-trips as before — the
    # band's stored `shape_band_sd` still recovers.
    blocks = rust_module().decoder_channel_cov_factors(cov, int(m_basis))
    return None if blocks is None else blocks.tolist()


def _channel_cov_from_factors(factors: Any) -> np.ndarray | None:
    """Rebuild the full-shape ``(M_k·p, M_k·p)`` decoder covariance from the
    compact per-channel factor written by :func:`_channel_cov_factors`.

    The result is block-diagonal across channels: the same-channel ``M_k × M_k``
    blocks are restored exactly and the cross-channel entries (which no band
    reads) are zero. Reproduces the shape band ``Σ Φ Φ Cov[(·,c),(·,c)]`` to
    machine precision. ``None`` when no factor was stored.
    """
    if factors is None:
        return None
    blocks = np.ascontiguousarray(np.asarray(factors, dtype=float))
    if blocks.ndim != 3:
        raise ValueError(
            "decoder_covariance_channel_factors must be a (p, M_k, M_k) array; "
            f"got shape {blocks.shape}"
        )
    # Block-diagonal reassembly math (#2091) lives in the Rust owner; it raises
    # on a non-square trailing pair, matching the previous contract.
    return rust_module().decoder_cov_from_channel_factors(blocks)


def _canonical_n_harmonics(
    basis_kinds: list[str],
    raw_n_harmonics: list[int],
    decoder_widths: list[int],
) -> list[int]:
    """Repair stale/degenerate periodic ``n_harmonics`` at ingestion (#1132/N).

    A periodic atom's basis width is ``M = 2H + 1`` with ``H >= 1``. A plan
    value that collapsed to ``<= 0`` (a born/fissioned atom recovered with a
    degenerate constant-only width) is floored to the harmonic count implied by
    the trained decoder width. The repair formula lives in the Rust owner
    (#2091, SPEC thin-wrapper rule 8); canonicalizing ``self._n_harmonics`` at
    ingestion ensures OOS reconstruct and :meth:`ManifoldSAE.steer` use the
    recovered value rather than the raw (possibly 0/stale) plan value.
    Non-periodic atoms pass through unchanged.
    """
    return [
        int(h)
        for h in rust_module().sae_canonical_n_harmonics(
            [str(bk) for bk in basis_kinds],
            [int(h) for h in raw_n_harmonics],
            [int(w) for w in decoder_widths],
        )
    ]


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    return value


def _canonical_assignment(value: str, label: str) -> str:
    name = str(value).strip().lower()
    canon = _ASSIGNMENT_KINDS.get(name)
    if canon is None:
        raise ValueError(
            f"{label}={value!r} is not a recognized assignment kind; "
            f"expected one of {sorted(set(_ASSIGNMENT_KINDS))}"
        )
    return canon


def _coerce_atom_inference(raw: Any) -> list[dict[str, Any]] | None:
    """Normalize the Rust ``atom_inference`` payload list (#1097 / #1103).

    Each Rust entry is a per-atom dict carrying ``atom_index`` / ``atom_name``,
    an optional ``functionals`` block, and an optional ``smooth_significance``
    block whose ``log_e_nonconstant`` is the #1103 any-n-valid split-LRT e-value.
    We pass the report through as plain Python containers (a shallow copy so the
    accessor never aliases the raw FFI object), coercing the e-value to ``float``
    when present. ``None`` for payloads predating the report.
    """
    if raw is None:
        return None
    reports: list[dict[str, Any]] = []
    for entry in raw:
        report = dict(entry)
        sig = report.get("smooth_significance")
        if sig is not None:
            sig = dict(sig)
            log_e = sig.get("log_e_nonconstant")
            sig["log_e_nonconstant"] = None if log_e is None else float(log_e)
            report["smooth_significance"] = sig
        if report.get("functionals") is not None:
            report["functionals"] = dict(report["functionals"])
        reports.append(report)
    return reports


def _coerce_cotrain_report(raw: Any) -> dict[str, Any] | None:
    """Normalize the Rust co-trained amortized-encoder report (#1154)."""
    if raw is None:
        return None
    report = dict(raw)
    required = (
        "recon_consistency",
        "unconverged_fraction",
        "n_unconverged",
        "n_encodes",
    )
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"SAE cotrain report missing keys: {missing}")
    recon = float(report["recon_consistency"])
    unconverged = float(report["unconverged_fraction"])
    n_unconverged = int(report["n_unconverged"])
    n_encodes = int(report["n_encodes"])
    if not np.isfinite(recon) or recon < 0.0:
        raise ValueError(f"SAE cotrain recon_consistency must be finite >= 0, got {recon}")
    if not np.isfinite(unconverged) or unconverged < 0.0 or unconverged > 1.0:
        raise ValueError(
            "SAE cotrain unconverged_fraction must be finite in [0, 1], "
            f"got {unconverged}"
        )
    if n_unconverged < 0 or n_encodes < 0 or n_unconverged > n_encodes:
        raise ValueError(
            "SAE cotrain counts must satisfy 0 <= n_unconverged <= n_encodes; "
            f"got {n_unconverged} / {n_encodes}"
        )
    return {
        "recon_consistency": recon,
        "unconverged_fraction": unconverged,
        "n_unconverged": n_unconverged,
        "n_encodes": n_encodes,
    }


def _canonical_public_assignment(value: str) -> str:
    name = str(value).strip().lower()
    canon = _PUBLIC_ASSIGNMENT_ALIASES.get(name)
    if canon is None:
        raise ValueError(
            f"assignment={value!r} is not a recognized assignment kind; "
            f"expected one of {sorted(_PUBLIC_ASSIGNMENT_ALIASES)}"
        )
    return canon


# Sentinel so ``assignment_prior`` can tell "not supplied" apart from any
# explicit value.
_ASSIGNMENT_PRIOR_UNSET = object()

# Sentinel so ``alpha`` can tell "not supplied" apart from an explicit
# ``alpha=1.0``. When the caller does not set ``alpha`` and the assignment is
# an explicit ``ibp_map``, the concentration defaults to the K-aware value below
# rather than the historical fixed ``1.0`` (see #1784).
_ALPHA_UNSET: Any = object()


def _default_ibp_concentration_for_k_atoms(k_atoms: int) -> float:
    """K-aware default IBP concentration ``α`` (#1784).

    Thin wrapper over the Rust source of truth
    ``assignment::default_ibp_concentration_for_k_atoms`` (FFI
    ``sae_default_ibp_concentration_for_k_atoms``): the formula
    ``α = max(1, 1/(exp(1/K) − 1))`` is computed once in the core, never mirrored
    in Python. Choosing ``α`` so the last atom retains prior mass
    ``π_{K-1} = (α/(α+1))^K ≈ e^{-1}`` makes the ordered stick-breaking prior SPAN
    the whole dictionary (no atom structurally masked); floored at ``1.0`` so
    ``K = 1`` keeps the historical ``α = 1``.
    """
    return float(rust_module().sae_default_ibp_concentration_for_k_atoms(int(max(int(k_atoms), 1))))


def _default_top_k_for_large_dictionary(n_obs: int, k_atoms: int) -> int | None:
    """Default large-K active cap from the data-per-atom ratio.

    Thin wrapper over the Rust source of truth
    ``assignment::default_top_k_for_large_dictionary`` (FFI
    ``sae_default_top_k_for_large_dictionary``): ``None`` when the dense softmax
    path is admitted, else the per-row cap ``clamp(ceil(N/K), 1, K−1)``.
    """
    cap = rust_module().sae_default_top_k_for_large_dictionary(int(n_obs), int(k_atoms))
    return None if cap is None else int(cap)


def _resolve_public_assignment(assignment: Any, assignment_prior: Any) -> str:
    """Normalize the ``assignment`` / ``assignment_prior`` aliases (#159).

    Both kwargs route through the single :func:`_canonical_public_assignment`
    validator, so they can never accept different alias sets. If both are
    supplied and resolve to DIFFERENT canonical kinds, raise an eager
    ``ValueError`` naming both BEFORE any Rust call. Unknown values raise a
    ``ValueError`` listing the accepted alias set.
    """
    canon_assignment = _canonical_public_assignment(assignment)
    if assignment_prior is _ASSIGNMENT_PRIOR_UNSET:
        return canon_assignment
    canon_prior = _canonical_public_assignment(assignment_prior)
    if canon_prior != canon_assignment:
        raise ValueError(
            f"assignment={assignment!r} (resolves to {canon_assignment!r}) and "
            f"assignment_prior={assignment_prior!r} (resolves to {canon_prior!r}) "
            f"were both supplied with conflicting values; pass only one "
            f"(they are aliases)."
        )
    return canon_assignment


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _atom_functional_evidence(
    atom: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Pass through the Rust/Riesz per-atom functional-evidence payload.

    A native payload is returned as a shallow copy so the accessor never
    aliases the raw FFI object; absent, ``None``. There is no Python-side
    plugin any more: the decoder functional evidence (average value /
    derivative / peak contrast and their delta-method standard errors) is a
    numeric result computed in the Rust core, not reconstructed in Python.
    """
    del plan  # plan-driven Python basis re-evaluation removed; Rust owns it.
    native = atom.get("functional_evidence")
    return None if native is None else dict(native)


@dataclass(slots=True)
class SaeManifoldAtomFit:
    """Per-atom fit payload returned inside :class:`ManifoldSAE`.

    Attributes
    ----------
    basis
        Basis kind used by this atom, for example ``"periodic"``,
        ``"euclidean"``, ``"duchon"``, ``"sphere"``, or ``"torus"``.
    decoder_coefficients
        Gauge-free physical decoder coefficients ``exp(s_k) B_k`` with shape
        ``(M_k, p)`` where
        ``M_k`` is the atom basis size and ``p`` is the ambient/output
        dimension. The Rust fit's quotient-scale coordinate is materialized at
        the boundary, so save/load, reconstruction, prediction, and steering all
        consume this same block with no hidden scale state. Values are in the
        same units as ``X`` because the basis functions are dimensionless.
    assignments
        Per-observation assignment/gate values for this atom, shape ``(N,)``.
        For ``assignment="softmax"`` these are mixture masses; for
        ``"ibp_map"`` and ``"jumprelu"`` these are
        gate activations.
    coords
        Recovered on-atom latent coordinates ``t*`` for the training data,
        shape ``(N, d_k)``. Units are the atom's raw latent coordinate system:
        periodic/circle coordinates are normalized phase coordinates, while
        euclidean/duchon coordinates are raw chart coordinates.
    evidence
        The MODEL-level penalized-loss score copied from the full SAE result
        (the Rust ``penalized_loss_score``). This is NOT a REML / marginal-
        likelihood score and is NOT atom-specific -- it is the same value for
        every atom of a given fit. ``None`` for the closed-form shortcut
        payloads, which do not compute a penalized-loss objective.
    active_dim
        Estimated active intrinsic coordinate dimension for this atom.
    decoder_covariance
        Optional phi-scaled posterior covariance of the flattened decoder
        coefficients, shape ``(M_k * p, M_k * p)`` in row-major
        ``(basis, channel)`` layout. Entries have squared ``X`` units. Present
        on fresh fits when the Rust payload includes posterior uncertainty.
        Across :meth:`ManifoldSAE.save` / :meth:`ManifoldSAE.load` only the
        band-relevant same-channel Schur blocks are persisted (the compact
        per-atom factor), so a restored covariance is block-diagonal across
        channels: every quantity the shape band reads is reproduced exactly, and
        the cross-channel entries the band never touches are zero.
    shape_band_coords
        Optional coordinate grid for the posterior shape band, shape
        ``(G, d_k)``, in the same latent-coordinate units as ``coords``.
    shape_band_mean
        Optional fitted ambient manifold values on ``shape_band_coords``,
        shape ``(G, p)``, in the same units as ``X``.
    shape_band_sd
        Optional per-channel posterior standard deviation of
        ``shape_band_mean``, shape ``(G, p)``, in the same units as ``X``.
    functional_evidence
        Optional per-atom decoder functional evidence. Native Rust/Riesz
        payloads are passed through as-is; otherwise fresh fits may populate a
        conservative plugin block from decoder covariance with
        ``average_value``, ``average_derivative`` (the conditional-on-fit mean
        decoder derivative — NOT a population marginal slope; the latent
        coordinate is a generated regressor), and ``peak_contrast``.
    """

    basis: str
    decoder_coefficients: np.ndarray
    assignments: np.ndarray
    coords: np.ndarray
    evidence: float | None
    active_dim: int
    # Posterior shape uncertainty. These fields are ``None`` only when the
    # source payload did not include uncertainty arrays. ``decoder_covariance``
    # is the phi-scaled posterior covariance of this atom's decoder
    # coefficients, shape ``(M_k*p, M_k*p)`` in row-major ``(basis, channel)``
    # flat layout. The shape band is the closed-form push-forward to ambient
    # space along the on-atom coordinates: ``shape_band_mean`` is the fitted
    # point ``(G, p)``, ``shape_band_sd`` its per-channel posterior sd
    # ``(G, p)``, at ``shape_band_coords`` ``(G, d_k)``.
    decoder_covariance: np.ndarray | None = None
    shape_band_coords: np.ndarray | None = None
    shape_band_mean: np.ndarray | None = None
    shape_band_sd: np.ndarray | None = None
    functional_evidence: dict[str, Any] | None = None
    # #2081 — the honest arc-length coordinate ``u_arc = s(t_i)/L in [0, 1)`` for
    # every training row, the gauge-fixed complement to the raw ``coords`` (which
    # are chart-arbitrary: reconstruction EV cannot certify them). Present for a
    # ``d = 1`` circle/interval atom whose chart is non-degenerate; ``None`` for
    # higher-d atoms or a collapsed chart (see the coordinate-fidelity
    # ``verdict``). Downstream angle/dose/adjacency claims should read this, not
    # ``coords``, unless the certificate verdict is ``arclength_honest``. Use
    # :meth:`ManifoldSAE.atom_angle_coordinate` for the certificate-gated reader.
    coords_u_arc: np.ndarray | None = None
    _lazy_artifact: _SaeLazyFitArtifact | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._lazy_artifact is not None:
            return
        values = {
            "assignments": object.__getattribute__(self, "assignments"),
            "coords": object.__getattribute__(self, "coords"),
        }
        object.__setattr__(self, "_lazy_artifact", _SaeLazyFitArtifact(values))
        object.__setattr__(self, "assignments", None)
        object.__setattr__(self, "coords", None)

    def __getattribute__(self, name: str) -> Any:
        return _lazy_getattr(self, _ATOM_LAZY_FIELDS, name)

    def __setattr__(self, name: str, value: Any) -> None:
        _lazy_setattr(self, _ATOM_LAZY_FIELDS, name, value)


@dataclass(slots=True)
class SaeManifoldFitResult:
    atoms: list[SaeManifoldAtomFit]
    chosen_k: int
    evidence_by_candidate: dict[int, float]
    comparison: dict[str, Any]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    reml_score: float
    _lazy_artifact: _SaeLazyFitArtifact | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._lazy_artifact is not None:
            return
        values = {
            "fitted": object.__getattribute__(self, "fitted"),
            "assignments": object.__getattribute__(self, "assignments"),
            "coords": object.__getattribute__(self, "coords"),
        }
        object.__setattr__(self, "_lazy_artifact", _SaeLazyFitArtifact(values))
        object.__setattr__(self, "fitted", None)
        object.__setattr__(self, "assignments", None)
        object.__setattr__(self, "coords", None)

    def __getattribute__(self, name: str) -> Any:
        return _lazy_getattr(self, _LOW_LEVEL_LAZY_FIELDS, name)

    def __setattr__(self, name: str, value: Any) -> None:
        _lazy_setattr(self, _LOW_LEVEL_LAZY_FIELDS, name, value)


class _LowLevelView:
    """Read-only adapter reproducing the ``SaeManifoldFitResult`` surface consumers
    touch off ``model.low_level`` (in practice only ``.chosen_k``); the remaining
    fields are forwarded off the Rust-owned core for faithfulness. #2091."""

    __slots__ = ("_core",)

    def __init__(self, core: Any) -> None:
        self._core = core

    @property
    def chosen_k(self) -> int:
        return int(self._core.chosen_k)

    @property
    def atoms(self) -> list[Any]:
        return list(self._core.atoms)

    @property
    def fitted(self) -> np.ndarray:
        return self._core.fitted

    @property
    def assignments(self) -> np.ndarray:
        return self._core.assignments

    @property
    def coords(self) -> list[np.ndarray]:
        return list(self._core.coords)

    @property
    def reml_score(self) -> float | None:
        return self._core.reml_score


class ManifoldSAE:
    """Fitted SAE-manifold model returned by :func:`sae_manifold_fit` — a thin
    wrapper over a Rust-owned ``ManifoldSaeCore`` (#2091). ALL fit state (atoms,
    decoder blocks, coords, assignments, scalar config, report/certificate blocks,
    Fisher shard, selected ρ*) lives in the core and is read through
    :meth:`__getattr__`; the only Python-side field is the ``training_data``
    metadata overlay (the core never retains X). The public method surface
    (predict/reconstruct/encode/steer/project/…) is unchanged and reads
    ``self.<field>`` — now resolved off the core.
    """

    __slots__ = ("_core", "_training_data")

    # Private design-field names whose core getter drops the leading underscore.
    _CORE_ALIASES: dict[str, str] = {
        "_basis_kinds": "basis_kinds",
        "_atom_dims": "atom_dims",
        "_basis_sizes": "basis_sizes",
        "_n_harmonics": "n_harmonics",
        "_duchon_centers": "duchon_centers",
        "_oos_projection_top1": "oos_projection_top1",
    }

    # Fields with a Rust setter (attach_fisher / test mutation); all else read-only.
    _SETTABLE_CORE: frozenset[str] = frozenset(
        {
            "fisher_factors",
            "fisher_mass_residual",
            "fisher_provenance",
            "metric_provenance",
            "oos_projection_top1",
        }
    )

    def __init__(self, core: Any, *, training_data: Any = None) -> None:
        object.__setattr__(self, "_core", core)
        if training_data is None:
            # New fits do not retain X: expose a zero-byte handle whose shape/dtype
            # mirror the fitted array (the legacy lazy-artifact contract).
            fitted = core.fitted
            training_data = _SaeTrainingDataHandle(tuple(fitted.shape), fitted.dtype)
        object.__setattr__(self, "_training_data", training_data)

    # ── state delegation ──────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup (slots / methods / properties) misses,
        # i.e. for a STATE field → forward to the core getter.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        core = object.__getattribute__(self, "_core")
        alias = type(self)._CORE_ALIASES.get(name, name)
        try:
            return getattr(core, alias)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r} "
                f"(not a ManifoldSaeCore field)"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_core", "_training_data"):
            object.__setattr__(self, name, value)
            return
        if name == "training_data":
            object.__setattr__(self, "_training_data", value)
            return
        alias = type(self)._CORE_ALIASES.get(name, name)
        if alias in type(self)._SETTABLE_CORE:
            setattr(self._core, alias, value)
            return
        raise AttributeError(
            f"ManifoldSAE.{name} is read-only Rust-owned state (no core setter); "
            f"settable: {sorted(type(self)._SETTABLE_CORE)} + training_data"
        )

    # ── Python-side overlay / forwarded properties ────────────────────────────
    @property
    def training_data(self) -> Any:
        return self._training_data

    @property
    def low_level(self) -> "_LowLevelView":
        return _LowLevelView(self._core)

    @property
    def chosen_k(self) -> int:
        """The number of atoms the structure search settled on (#1205);
        forwards to the core (== ``len(self.atoms)`` for a resolved fit)."""
        return int(self._core.chosen_k)

    @property
    def reml_score(self) -> float | None:
        """DEPRECATED read alias for :attr:`penalized_loss_score` (#1231)."""
        return self._core.reml_score

    def __repr__(self) -> str:
        d_atom = int(self.coords[0].shape[1]) if self.coords else 0
        n, p = (self.fitted.shape if self.fitted.ndim == 2 else (self.fitted.shape[0], 1))
        # Lead with bits/token (the headline currency); keep EV as a demoted line.
        try:
            dl = self.description_length()
        except Exception:
            dl = None
        bpt = "n/a" if dl is None else f"{float(dl['bits_per_token']):.3f}"
        return (
            f"ManifoldSAE(bits/token={bpt}, K={len(self.atoms)}, d_atom={d_atom}, "
            f"atom_topology={self.atom_topology!r}, assignment={self.assignment!r}, "
            f"alpha={self.alpha!r}, learnable_alpha={self.learnable_alpha}, "
            f"n={n}, p={p}, r2={self.reconstruction_r2:.3f})"
        )

    @classmethod
    def from_payload(
        cls,
        x: np.ndarray,
        payload: Mapping[str, Any],
        topology: str,
        assignment: str,
        penalties: list[str],
        alpha: float = 1.0,
        learnable_alpha: bool = False,
        *,
        assignment_label: str | None = None,
        tau: float = 0.5,
        sparsity_strength: float = 1.0,
        smoothness: float = 1.0,
        learning_rate: float = 0.04,
        max_iter: int = 50,
        random_state: int = 0,
        top_k: int | None = None,
        jumprelu_threshold: float = 0.0,
        fisher_factors: np.ndarray | None = None,
        fisher_provenance: str | None = None,
        declared_bases: list[str] | None = None,
    ) -> "ManifoldSAE":
        """Build the Rust-owned core from the RAW ``sae_manifold_fit_minimal``
        payload and wrap it. ``sae_manifold_core_from_fit_payload`` reproduces the
        legacy ``from_payload ∘ to_dict`` bit-for-bit AND folds in the post-fit
        Fisher attach (``fisher_factors``/``fisher_provenance``) and the
        ``linear_block`` relabel (``declared_bases``), so those are builder args
        rather than post-construction mutations."""
        canonical_assignment = _canonical_assignment(assignment, "assignment")
        raw_json = json.dumps(_jsonable_value(dict(payload)))
        core = rust_module().sae_manifold_core_from_fit_payload(
            raw_json,
            np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
            str(topology),
            canonical_assignment,
            str(assignment if assignment_label is None else assignment_label),
            list(penalties),
            float(alpha),
            bool(learnable_alpha),
            float(tau),
            float(sparsity_strength),
            float(smoothness),
            float(learning_rate),
            int(max_iter),
            int(random_state),
            top_k,
            float(jumprelu_threshold),
            fisher_factors=(
                None
                if fisher_factors is None
                else np.ascontiguousarray(np.asarray(fisher_factors, dtype=np.float64))
            ),
            fisher_provenance=fisher_provenance,
            declared_bases=None if declared_bases is None else list(declared_bases),
        )
        # New fits do not retain X: overlay a zero-byte handle mirroring the input.
        return cls(core, training_data=_training_data_handle(np.asarray(x)))

    def structure_certificate(self, *, alpha: float | None = None) -> dict[str, Any]:
        """Anytime-valid structure-discovery certificate (#1058 / #984).

        Surfaces the e-BH certificate the structure search computed over the
        ledger of structural claims (atom-exists / binding-edge / geometry-kind)
        the fit proposed. Each claim carries an anytime-valid e-process, so the
        e-value and the gated/contested verdict are valid at this (or any)
        data-dependent stopping time — safe to peek.

        Parameters
        ----------
        alpha : float, optional
            FDR level to re-derive the gated set at. ``None`` (default) keeps the
            level the fit certified at (α = 0.05). A different α only re-runs the
            e-BH step over the stored per-claim e-values; it never refits.

        Returns
        -------
        dict
            ``{"alpha": float, "fdr_level": float, "n_confirmed": int,
            "claims": [{"claim_index": int, "claim": str, "kind": dict,
            "e_value": float, "log_e": float, "steps": int, "confirmed": bool,
            "evidence_remaining_nats": float}, ...]}``.
            ``evidence_remaining_nats`` is the anytime-valid budget ``max(0,
            ln(m / (alpha*k)) - log_e)`` measured against the SAME rank-aware
            e-BH threshold the confirmation rule uses (``e_(k) >= m/(alpha*k)``
            at the claim's descending-log_e rank ``k`` out of ``m`` claims) —
            the additional log-evidence a probe must accumulate before the claim
            crosses the confirmation threshold (0 once already confirmed).
        """
        if self.structure_certificate_json is None:
            raise ValueError(
                "this fitted model carries no structure certificate (payload "
                "predates #1058); refit to obtain one"
            )
        # The whole accessor computation — re-run the rank/multiplicity-aware e-BH
        # at `alpha`, and derive each claim's label, e-value, confirmed flag, and
        # anytime-valid `evidence_remaining_nats` budget — lives in the Rust owner
        # (#2091, SPEC thin-wrapper rule 8). We only json.loads its report.
        return json.loads(
            rust_module().sae_structure_certificate_report(
                self.structure_certificate_json,
                None if alpha is None else float(alpha),
            )
        )

    def atom_inference(self) -> list[dict[str, Any]]:
        """Per-atom smooth-functional inference reports (#1097 / #1103).

        One entry per fitted atom, in atom order, each
        ``{"atom_index": int, "atom_name": str, "functionals": {...} | None,
        "smooth_significance": {"log_e_nonconstant": float | None} | None}``.

        The #1103 ``smooth_significance.log_e_nonconstant`` is the any-n-valid
        split-likelihood-ratio e-value for "the atom's inner decoder smooth is
        non-constant" (null = constant), the same universal-inference instrument
        the atom-birth gate uses. With ``E_{H0}[E] <= 1`` it is finite-sample
        honest at the ``df ≈ n`` regime: a large positive ``log_e_nonconstant``
        is real evidence the atom carries smooth structure, ``<= 0`` does not
        favor non-constancy. An atom whose inner-decoder smooth was not harvested
        (no active rows / non-SPD inner Hessian / constant-only design) reports
        ``None`` fields rather than a fabricated value.

        Returns
        -------
        list of dict
            One report per atom. Empty list only for payloads predating the
            report (#1097 / #1103).
        """
        if self.atom_inference_reports is None:
            return []
        return [dict(report) for report in self.atom_inference_reports]

    def contested_claims(self, *, alpha: float | None = None) -> list[dict[str, Any]]:
        """The structure claims the held-out data did NOT confirm (#1058).

        Convenience filter over :meth:`structure_certificate`: returns only the
        contested claims (the inputs to a diagnostic probe-design loop), each
        with the anytime-valid ``evidence_remaining_nats`` budget that a probe
        would have to accumulate to confirm it. These are demoted, never
        rejected — they keep their evidence across future shards.
        """
        cert = self.structure_certificate(alpha=alpha)
        return [c for c in cert["claims"] if not c["confirmed"]]

    def _atom_index(self, atom: int) -> int:
        k = int(atom)
        if k < 0 or k >= len(self.atoms):
            raise IndexError(f"atom={atom} out of range for K={len(self.atoms)} atoms")
        return k

    def atom_trust(self, atom: int) -> float:
        """Scalar trust score for one atom, in ``[0, 1]``."""
        k = self._atom_index(atom)
        trust = np.asarray(self.diagnostics["atom_trust"], dtype=float)
        if trust.size == 0:
            raise ValueError(
                "this fit payload carries empty trust diagnostics; atom_trust is unavailable"
            )
        return float(trust[k])

    def atom_diagnostics(self, atom: int) -> dict[str, Any]:
        """All trust diagnostic components for one atom."""
        k = self._atom_index(atom)
        if not self.diagnostics["atoms"]:
            raise ValueError(
                "this fit payload carries empty trust diagnostics; atom_diagnostics is unavailable"
            )
        return dict(self.diagnostics["atoms"][k])

    def trust_scores(self, X: Any = None) -> dict[str, Any]:
        """Assignment-weighted row and per-atom trust scores.

        With ``X=None`` this uses the fitted assignments. Passing the training
        activation matrix reuses those assignments; any other matrix is encoded
        by this model's frozen decoder before the scores are assembled. The
        returned arrays therefore belong only to this explicit model handle and
        cannot be affected by another fit in the same process.
        """
        atom_trust = atom_trust_scores(self.diagnostics)
        n_atoms = int(atom_trust.shape[0])
        if X is None:
            assignments = np.asarray(self.assignments, dtype=float)
            n_rows = int(assignments.shape[0])
        else:
            x = _as_2d_float(X, "X")
            n_rows = int(x.shape[0])
            assignments = np.asarray(
                self.assignments if self._is_training_data(x) else self.encode(x),
                dtype=float,
            )
        if assignments.shape != (n_rows, n_atoms):
            raise ValueError(
                "trust score assignments shape mismatch: "
                f"expected {(n_rows, n_atoms)}, got {assignments.shape}"
            )
        row, per_atom = rust_module().sae_row_trust_scores(
            np.ascontiguousarray(assignments, dtype=np.float64),
            np.ascontiguousarray(atom_trust, dtype=np.float64),
        )
        return {
            "row": row,
            "per_atom": per_atom,
            "atom": atom_trust,
            "diagnostics": self.diagnostics,
        }

    def curvature(self) -> list[dict[str, Any]]:
        """Per-atom SAE curvature report (#1099, rescoped under #1115).

        Returns one record per atom: ``{"atom": int, "kappa_hat": float}``.
        ``kappa_hat`` is the fitted empirical second-fundamental-form sup-norm
        bound — a descriptive plug-in geometry summary. It is not an estimand
        with a confidence interval: a curvature bound has no profiled criterion,
        so no SE/CI/flatness fields are reported (the delta-method SE that #1099
        first shipped was conditioned on the generated latent coordinates as if
        known and under-covered, so #1115 removed it).
        """
        if self.curvature_report is None:
            raise ValueError(
                "this fitted model carries no SAE curvature report; refit to obtain one"
            )
        return [dict(atom) for atom in self.curvature_report.get("atoms", [])]

    def atom_curvature(self, atom: int) -> dict[str, Any]:
        """Curvature report record for one atom."""
        k = self._atom_index(atom)
        rows = self.curvature()
        if k >= len(rows):
            raise ValueError(
                f"curvature report has {len(rows)} atom rows but model has {len(self.atoms)} atoms"
            )
        return dict(rows[k])

    def coordinate_fidelity_report(self) -> list[dict[str, Any]]:
        """Per-atom chart coordinate-fidelity certificate (#2081).

        Returns one record per atom. An atom with a ``d = 1`` circle/interval
        chart carries ``{"atom": int, "topology": "circle" | "interval",
        "uniformity_statistic": float, "uniformity_p_value": float,
        "arclength_defect": float, "n_coords": int}``:

        * ``uniformity_statistic`` is Watson's ``U²`` of the fitted coordinate
          against the atom's uniform invariant measure (rotation/reflection
          invariant), and ``uniformity_p_value`` its closed-form asymptotic null
          p-value — small values flag a materially non-uniform coordinate.
        * ``arclength_defect`` is the speed coefficient-of-variation of the
          decoded curve on a uniform latent grid: ``0`` is an arc-length
          (unit-speed) chart, and a positive value means the parameterization
          squishes arc length — the failure reconstruction EV cannot see.
        * ``verdict`` is the certified reading rule — ``"arclength_honest"``
          (read the raw ``coords``), ``"recoverable_via_arclength"`` (read
          ``coords_u_arc`` instead), or ``"degenerate"`` (refuse; the chart
          collapses) — and ``certified`` is ``verdict != "degenerate"``. Use
          :meth:`atom_angle_coordinate` for the gated reader.
        * ``coords_u_arc`` is the honest arc-length coordinate ``s(t_i)/L`` in
          ``[0, 1)`` for every fitted row (a pure read, computed regardless of
          whether the decoder-mutating canonicalization committed), or ``None``
          for a degenerate chart. ``raw_arclength_defect_rms`` /
          ``raw_arclength_defect_max`` measure the raw-vs-``u_arc`` distance at
          the data rows after best gauge alignment, and ``min_speed_over_mean`` /
          ``max_speed_over_mean`` / ``log_speed_rms`` summarize the speed profile.

        Atoms without a ``d = 1`` chart carry a null ``topology``.
        """
        if self.coordinate_fidelity is None:
            raise ValueError(
                "this fitted model carries no SAE coordinate-fidelity report; refit to obtain one"
            )
        return [dict(atom) for atom in self.coordinate_fidelity.get("atoms", [])]

    def atom_coordinate_fidelity(self, atom: int) -> dict[str, Any]:
        """Coordinate-fidelity certificate record for one atom (#2081)."""
        k = self._atom_index(atom)
        rows = self.coordinate_fidelity_report()
        if k >= len(rows):
            raise ValueError(
                f"coordinate-fidelity report has {len(rows)} atom rows but model has "
                f"{len(self.atoms)} atoms"
            )
        return dict(rows[k])

    def topology_persistence_report(self) -> list[dict[str, Any]]:
        """Per-atom persistent-homology topology audit."""
        if self.topology_persistence is None:
            raise ValueError(
                "this fitted model carries no SAE topology-persistence report; refit to obtain one"
            )
        return [dict(atom) for atom in self.topology_persistence.get("atoms", [])]

    def atom_topology_persistence(self, atom: int) -> dict[str, Any]:
        """Topology-persistence audit record for one atom."""
        k = self._atom_index(atom)
        rows = self.topology_persistence_report()
        if k >= len(rows):
            raise ValueError(
                f"topology-persistence report has {len(rows)} atom rows but model has "
                f"{len(self.atoms)} atoms"
            )
        return dict(rows[k])

    def atom_angle_coordinate(self, atom: int) -> np.ndarray:
        """The certificate-gated honest coordinate for one ``d = 1`` atom (#2081).

        Returns the arc-length coordinate ``u_arc = s(t_i)/L in [0, 1)`` — the
        gauge-fixed reading of the atom's manifold — for every training row. This
        is the coordinate every angle / dose-in-nats / adjacency claim should
        consume, because reconstruction EV provably does NOT certify the raw
        latent coordinate (a chart can reconstruct its ring at high EV while
        reading a squished, non-uniform angle).

        The reader is gated on the coordinate-fidelity ``verdict``:

        * ``arclength_honest`` — the raw chart is already arc-length; ``coords``
          and ``coords_u_arc`` agree up to gauge. Returns ``coords_u_arc``.
        * ``recoverable_via_arclength`` — the raw chart squishes arc length but
          the honest coordinate is recoverable. Returns ``coords_u_arc``.
        * ``degenerate`` — the chart collapses (the decoded speed vanishes
          somewhere), so no faithful coordinate exists. **Raises** rather than
          hand back an arbitrary chart.

        Raises ``ValueError`` for an atom without a ``d = 1`` chart or without a
        coordinate-fidelity certificate (refit to obtain one).
        """
        record = self.atom_coordinate_fidelity(atom)
        if record.get("topology") is None:
            raise ValueError(
                f"atom {atom} has no d=1 circle/interval chart; no angle coordinate is defined"
            )
        verdict = record.get("verdict")
        if verdict == "degenerate" or record.get("coords_u_arc") is None:
            raise ValueError(
                f"atom {atom} chart is degenerate (verdict={verdict!r}); the decoded curve "
                "collapses, so no faithful angle coordinate exists — refusing rather than "
                "returning an arbitrary chart"
            )
        return np.asarray(record["coords_u_arc"], dtype=float)

    def shape_uncertainty(self, atom: int = 0, *, n_sd: float = 1.96) -> dict[str, np.ndarray]:
        """Posterior ambient shape uncertainty for one atom.

        Returns ``{"coords", "mean", "sd", "lower", "upper"}`` — the fitted
        ambient curve/surface and its closed-form posterior uncertainty on the
        atom's uncertainty grid. ``coords`` has shape ``(G, d_k)`` and uses the
        atom's raw latent-coordinate units. ``mean`` and ``sd`` have shape
        ``(G, p)`` and use the same ambient units as the training data ``X``;
        ``lower``/``upper`` are ``mean ± n_sd * sd``. ``n_sd=1.96`` gives the
        pointwise 95% posterior band. The per-atom decoder coefficient
        covariance that generated this band is available as
        ``self.atoms[atom].decoder_covariance``.
        """
        k = self._atom_index(atom)
        atom = self.atoms[k]
        if (
            atom.shape_band_coords is None
            or atom.shape_band_mean is None
            or atom.shape_band_sd is None
        ):
            raise ValueError(
                "shape_uncertainty is only available when the fit payload "
                "includes shape_band_coords, shape_band_mean, and shape_band_sd."
            )
        coords = np.asarray(atom.shape_band_coords, dtype=float)
        mean = np.asarray(atom.shape_band_mean, dtype=float)
        sd = np.asarray(atom.shape_band_sd, dtype=float)
        width = float(n_sd) * sd
        return {
            "coords": coords.copy(),
            "mean": mean.copy(),
            "sd": sd.copy(),
            "lower": mean - width,
            "upper": mean + width,
        }

    def _oos_payload(self, X: Any, *, t_init: Any = None, a_init: Any = None) -> dict[str, Any]:
        """Run the frozen-decoder OOS Newton solve on ``X`` and return the full
        payload dict (``assignments_z``, per-atom ``on_atom_coords_t``,
        ``logits``, ``fitted``).

        Optional ``t_init`` (K, N, D_max) and ``a_init`` (N, K) warm-start the
        refinement from an amortized encoder's per-token prediction (#357).
        """
        x = _as_2d_float(X, "X")
        kind = _canonical_assignment(self.assignment, "assignment")
        if t_init is None and a_init is None:
            # No warm start supplied; let the Rust fixed-decoder solve seed itself.
            coords_init, logits_init = None, None
        else:
            logits_init = None if a_init is None else np.ascontiguousarray(np.asarray(a_init, dtype=float))
            coords_init = None if t_init is None else np.ascontiguousarray(np.asarray(t_init, dtype=float))
        payload = rust_module().sae_manifold_predict_oos(
            np.ascontiguousarray(x), list(self._basis_kinds), list(self._atom_dims),
            [np.ascontiguousarray(b) for b in self.decoder_blocks],
            [None if c is None else np.ascontiguousarray(c) for c in self._duchon_centers],
            [
                (int(h) if k in {"periodic", "torus", "cylinder", "mobius"} else None)
                for k, h in zip(self._basis_kinds, self._n_harmonics)
            ],
            [int(s) for s in self._basis_sizes],
            alpha=float(self.alpha), tau=float(self.tau), assignment_kind=str(kind),
            max_iter=int(self.max_iter), learning_rate=float(self.learning_rate),
            initial_logits=logits_init, initial_coords=coords_init,
            top_k=self.top_k,
            jumprelu_threshold=float(self.jumprelu_threshold),
            # #1228 — thread the trained dictionary's hybrid-collapsed straight
            # sub-models so the held-out reconstruction decodes verdict-linear
            # d=1 slots by the SAME linear image the training reconstruction used.
            hybrid_linear_images=self._hybrid_linear_images_for_oos(),
            # #2132 — thread the trained TERMINAL ρ* so the frozen-decoder OOS
            # Newton solve descends the SAME penalized objective the training
            # state converged under. Without these the solve rebuilt ρ from the
            # INITIAL sparsity/smoothness scalars and zero ARD — a different
            # objective under which the trained optimum is not stationary, so
            # re-encoding the training rows collapsed (warm start decayed BELOW
            # the cold start). A complete terminal state is required.
            log_lambda_sparse=self.selected_log_lambda_sparse,
            log_lambda_smooth=(
                None
                if self.selected_log_lambda_smooth is None
                else [float(v) for v in np.asarray(self.selected_log_lambda_smooth, dtype=float)]
            ),
            log_ard=(
                None
                if self.selected_log_ard is None
                else [[float(v) for v in np.asarray(a, dtype=float).ravel()] for a in self.selected_log_ard]
            ),
            # #2132 — a learnable-IBP fit's OOS gates must resolve α through the
            # same learnable schedule (terminal log_lambda_sparse) as training.
            learnable_alpha=bool(self.learnable_alpha),
        )
        return dict(payload)

    def _hybrid_linear_images_for_oos(
        self,
    ) -> "list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray | None]] | None":
        """Extract the per-slot collapsed straight sub-models from the stored
        ``hybrid_split`` report as ``(atom_idx, t_bar, b0, b1, v)`` tuples for the
        OOS reconstruction (#1228/#1777). ``v`` is the collapse-rescue projection
        direction (length ``p``, unit norm) for a rescued slot, else ``None`` for
        an ordinary straight image. When ``v`` is present the held-out
        reconstruction recomputes each row's coordinate from its own
        leave-this-atom-out residual projected onto ``v`` (target-aware), so a
        collapse-rescued atom reconstructs IDENTICALLY train vs held-out. ``None``
        when no report is attached or no slot collapsed to linear (an all-curved
        OOS reconstruction)."""
        hs = self.hybrid_split
        if not hs:
            return None
        atoms = hs.get("atoms") if isinstance(hs, Mapping) else None
        if not atoms:
            return None
        images: "list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray | None]]" = []
        for entry in atoms:
            li = entry.get("linear_image") if isinstance(entry, Mapping) else None
            if not li:
                continue
            v_raw = li.get("v") if isinstance(li, Mapping) else None
            v_arr = (
                None
                if v_raw is None
                else np.ascontiguousarray(np.asarray(v_raw, dtype=float))
            )
            images.append((
                int(li["atom_idx"]),
                float(li["t_bar"]),
                np.ascontiguousarray(np.asarray(li["b0"], dtype=float)),
                np.ascontiguousarray(np.asarray(li["b1"], dtype=float)),
                v_arr,
            ))
        return images or None

    def _is_training_data(self, x: np.ndarray) -> bool:
        """True only for an input that is bit-exactly the training matrix.

        Reconstruction / encoding is a function of the EXACT input, so the
        cached-training shortcut must require exact equality — a near-duplicate
        input must take the OOS solve path. Identity is the fast path; otherwise
        compare shape, dtype, and every byte via :func:`np.array_equal`. Never
        use tolerance equality (``np.allclose``) for model semantics.
        """
        td = self.training_data
        if isinstance(td, _SaeTrainingDataHandle):
            return False
        if x is td:
            return True
        return (
            x.shape == td.shape
            and x.dtype == td.dtype
            and np.array_equal(x, td)
        )

    def build_encode_atlas(
        self,
        *,
        amplitude_bounds: Any | None = None,
        target_norm_bound: float | None = None,
        grid_resolution: int = 32,
        ridge: float = 1.0e-10,
        newton_steps: int = 8,
    ) -> Any:
        """Build the frozen-dictionary Kantorovich-certified encode atlas (#1010).

        Returns the Rust ``SaeEncodeAtlas``. Its
        ``certified_encode(X, amplitudes, atom_index)`` exposes the per-row
        ``h <= 1/2`` Newton-Kantorovich certificate directly — the honesty signal
        an amortized encoder reads INSTEAD of a cold exact multi-start probe per
        row (:meth:`converged_latents`). Rows that fail the certificate are
        flagged (``certified[i] is False``) and must be routed to that exact
        fallback; no approximation enters silently.

        This is a thin marshalling wrapper (SPEC: math lives in Rust): it hands
        the frozen decoder blocks + basis metadata to the FFI, which reconstructs
        the atoms with live analytic evaluators and lays down the certified charts.

        ``amplitude_bounds[k]`` bounds ``|z_k|`` and ``target_norm_bound`` bounds
        ``||x||`` over the encode data; both scale the offline Hessian-Lipschitz
        constant, so a larger bound only shrinks the certified radius (never a
        false certificate). When omitted they default to the trained assignment
        magnitudes and the training-row norms, respectively.
        """
        # Default bounds (per-atom max|assignment| and max row L2 of x) are
        # reduced inside the Rust builder: hand it the raw arrays and a `None`
        # sentinel rather than pre-reducing in NumPy. Explicit bounds pass
        # through verbatim and the arrays are omitted.
        assignments_arr = None
        if amplitude_bounds is None:
            assignments_arr = np.ascontiguousarray(
                np.asarray(self.assignments, dtype=float)
            )
        encode_rows_arr = None
        if target_norm_bound is None:
            if isinstance(self.training_data, _SaeTrainingDataHandle):
                raise ValueError(
                    "build_encode_atlas(target_norm_bound=None) requires retained "
                    "training data, but ManifoldSAE no longer stores the input matrix; "
                    "pass target_norm_bound explicitly."
                )
            encode_rows_arr = np.ascontiguousarray(
                np.asarray(self.training_data, dtype=float)
            )
        return rust_module().build_sae_encode_atlas(
            list(self._basis_kinds),
            list(self._atom_dims),
            [np.ascontiguousarray(b) for b in self.decoder_blocks],
            [None if c is None else np.ascontiguousarray(c) for c in self._duchon_centers],
            [int(s) for s in self._basis_sizes],
            None if amplitude_bounds is None else [float(a) for a in amplitude_bounds],
            None if target_norm_bound is None else float(target_norm_bound),
            assignments=assignments_arr,
            encode_rows=encode_rows_arr,
            grid_resolution=int(grid_resolution),
            ridge=float(ridge),
            newton_steps=int(newton_steps),
        )

    def reconstruct(self, X: Any, *, t_init: Any = None, a_init: Any = None) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if t_init is None and a_init is None and self._is_training_data(x):
            return self.fitted.copy()
        payload = self._oos_payload(x, t_init=t_init, a_init=a_init)
        return np.asarray(payload["fitted"], dtype=float)

    def reconstruct_training(self) -> np.ndarray:
        """Materialize the in-sample dense reconstruction from stored codes.

        The original input matrix is not retained on fitted results. This method
        rebuilds the fitted dense array from the per-atom coordinates, assignment
        codes, and decoder blocks through the Rust reconstruction kernel only
        when a caller explicitly requests it.
        """
        assignments = np.asarray(self.assignments, dtype=np.float64)
        if assignments.ndim != 2:
            raise ValueError(f"assignments must be 2D; got shape {assignments.shape}")
        n_rows, k_atoms = assignments.shape
        if k_atoms != len(self.decoder_blocks):
            raise ValueError(
                f"assignment columns {k_atoms} must equal decoder block count "
                f"{len(self.decoder_blocks)}"
            )
        p_out = int(self.fitted.shape[1]) if k_atoms == 0 else int(self.decoder_blocks[0].shape[1])
        return _sae_manifold_reconstruct_native(
            list(self._basis_kinds),
            [int(dim) for dim in self._atom_dims],
            [np.asarray(block, dtype=np.float64) for block in self.decoder_blocks],
            [np.asarray(coord, dtype=np.float64) for coord in self.coords],
            assignments,
            p_out,
        )

    def predict(self, X: Any) -> np.ndarray:
        return self.reconstruct(X)

    def distill_encoder(self, X: Any, **kwargs: Any) -> Any:
        """Train a post-hoc torch MLP encoder from exact OOS latent solves."""
        from .distill import distill_encoder

        return distill_encoder(self, X, **kwargs)

    def encode(
        self,
        X: Any,
        *,
        t_init: Any = None,
        a_init: Any = None,
        encoder: Any = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Out-of-sample per-token assignments ``a*`` of shape ``(N, K)``.

        Runs the frozen-decoder OOS solve on ``X`` and returns the converged
        assignment matrix. On training ``X`` (matched bit-exactly) the cached
        fit assignments are returned without re-solving. ``t_init`` / ``a_init``
        warm-start the refinement (#357).

        Passing ``encoder=fit.distill_encoder(...)`` runs the distilled encoder
        first, accepts only rows matching the exact warm-started solve within the
        encoder's calibrated gate, and reports rowwise fallback accounting when
        ``return_stats=True``. The exact solve remains the teacher and fallback;
        the encoder never defines the feature map.
        """
        x = _as_2d_float(X, "X")
        if encoder is not None:
            if t_init is not None or a_init is not None:
                raise ValueError("encode(..., encoder=...) cannot also take t_init or a_init")
            from .distill import encode_with_fallback

            encoded, stats = encode_with_fallback(self, x, encoder)
            if return_stats:
                return encoded, stats.to_dict()
            return encoded
        if t_init is None and a_init is None and self._is_training_data(x):
            encoded = self.assignments.copy()
            if return_stats:
                return encoded, {
                    "rows": int(encoded.shape[0]),
                    "accepted_rows": 0,
                    "fallback_rows": 0,
                    "fallback_rate": 0.0,
                    "exact_probe_rows": 0,
                }
            return encoded
        payload = self._oos_payload(x, t_init=t_init, a_init=a_init)
        encoded = np.asarray(payload["assignments_z"], dtype=float)
        if return_stats:
            return encoded, {
                "rows": int(encoded.shape[0]),
                "accepted_rows": 0,
                "fallback_rows": int(encoded.shape[0]),
                "fallback_rate": 1.0,
                "exact_probe_rows": int(encoded.shape[0]),
            }
        return encoded

    def converged_latents(self, X: Any | None = None, *, t_init: Any = None, a_init: Any = None) -> dict[str, Any]:
        """Converged supervision targets for an amortized encoder (#357).

        Returns ``{"coords": list[(N, d_k) ndarray], "assignments": (N, K)
        ndarray, "logits": (N, K) ndarray, "fitted": (N, p) ndarray}`` — the
        per-atom on-manifold coordinates ``t*`` and the assignments / gate
        ``a*`` the joint solver converged to. With ``X is None`` (or training
        ``X``) the stored training-fit latents are returned; otherwise the OOS
        solve is run on ``X``, optionally warm-started from ``t_init`` /
        ``a_init``."""
        x = None if X is None else _as_2d_float(X, "X")
        use_training = (
            t_init is None and a_init is None
            and (x is None or self._is_training_data(x))
        )
        if use_training:
            return {
                "coords": [c.copy() for c in self.coords],
                "assignments": self.assignments.copy(),
                "logits": self.low_level_logits.copy(),
                "fitted": self.fitted.copy(),
            }
        payload = self._oos_payload(x, t_init=t_init, a_init=a_init)
        return {
            "coords": [np.asarray(atom["on_atom_coords_t"], dtype=float) for atom in payload["atoms"]],
            "assignments": np.asarray(payload["assignments_z"], dtype=float),
            "logits": np.asarray(payload["logits"], dtype=float),
            "fitted": np.asarray(payload["fitted"], dtype=float),
        }

    def _amortized_encoder(self) -> Any:
        """Fit the distilled one-matmul encoder against this fit's exact per-row
        code (reviewer condition #2). The training corpus and the converged
        (logits, coords, amplitudes) are the supervision; the Rust
        ``SaeAmortizedEncoder`` learns the closed-form, evidence-selected map
        ``x -> (logits, coords, amplitudes)``. A thin marshalling wrapper — all
        math lives in Rust (SPEC)."""
        if isinstance(self.training_data, _SaeTrainingDataHandle):
            raise ValueError(
                "encode_amortized requires the training input matrix, but "
                "ManifoldSAE no longer retains it. Use distill_encoder(X, ...) "
                "with explicit training activations."
            )
        from .distill import _coordinate_periods

        rust = rust_module()
        coord_periods = _coordinate_periods(
            self, [int(dim) for dim in self._atom_dims]
        )
        return rust.SaeAmortizedEncoder(
            np.ascontiguousarray(np.asarray(self.training_data, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(self.low_level_logits, dtype=np.float64)),
            [np.ascontiguousarray(np.asarray(c, dtype=np.float64)) for c in self.coords],
            np.ascontiguousarray(np.asarray(self.assignments, dtype=np.float64)),
            [list(periods) for periods in coord_periods],
        )

    def encode_amortized(self, X: Any) -> dict[str, Any]:
        """DISTILLED one-matmul encode of ``X`` (reviewer condition #2).

        Our held-out reconstruction comes from a per-row test-time optimization
        (:meth:`converged_latents` / :meth:`project`, the exact frozen-decoder
        Newton solve); a sparse-autoencoder's comes from ONE matmul. This is that
        one-matmul path: a cheap encoder, distilled against this fit's exact
        per-row code by closed-form evidence maximization, predicts the code for
        fresh rows in a single matmul.

        Returns ``{"logits": (N, K), "coords": list[(N, d_k)], "amplitudes":
        (N, K), "used_quadratic_head": bool, "log_evidence": float,
        "feature_dim": int, "effective_dof": float}``. The amortization GAP — how
        far this deployed code sits from :meth:`converged_latents` (the exact
        oracle) — is the honest encode error; measure it by comparing the two
        codes / their reconstructions on held-out ``X``."""
        x = _as_2d_float(X, "X")
        encoder = self._amortized_encoder()
        code = encoder.encode_amortized(np.ascontiguousarray(x))
        return {
            "logits": np.asarray(code["logits"], dtype=float),
            "coords": [np.asarray(c, dtype=float) for c in code["coords"]],
            "amplitudes": np.asarray(code["amplitudes"], dtype=float),
            "used_quadratic_head": bool(encoder.used_quadratic_head),
            "log_evidence": float(encoder.log_evidence),
            "feature_dim": int(encoder.feature_dim),
            "effective_dof": float(encoder.effective_dof),
        }

    def project(self, X: Any, atom_k: int) -> np.ndarray:
        """Standalone per-atom projection ``project(x, atom_k) -> t`` (#357).

        Maps each ambient point in ``X`` to its on-manifold coordinate for atom
        ``atom_k`` under the trained decoder, via the same frozen-decoder OOS
        solve. Returns the ``(N, d_k)`` coordinate block for that atom — the
        minimal teacher signal for an encoder's coordinate head."""
        k = int(atom_k)
        if k < 0 or k >= len(self.atoms):
            raise IndexError(f"atom_k={atom_k} out of range for K={len(self.atoms)} atoms")
        x = _as_2d_float(X, "X")
        if self._is_training_data(x):
            return self.coords[k].copy()
        payload = self._oos_payload(x)
        return np.asarray(payload["atoms"][k]["on_atom_coords_t"], dtype=float)

    def atom_reconstruct(self, X: Any, atom_k: int) -> np.ndarray:
        """Single atom's reconstruction in data space ``(N, p)`` (#1777).

        Maps each ambient point in ``X`` to atom ``atom_k``'s decoded image
        ``Φ(t)·B`` — the atom's shape realized at that row's converged on-manifold
        coordinate — via the same frozen-decoder OOS solve as :meth:`project`
        (which returns the coordinate ``t``; this returns the decode of ``t`` in
        data space). This is the UNGATED per-atom decode (backed by the Rust
        ``fill_decoded_row``); the full reconstruction is the assignment-weighted
        sum ``Σ_k a_k · atom_reconstruct(X, k)``. Complements :meth:`project`
        (coords) with the data-space image."""
        k = int(atom_k)
        if k < 0 or k >= len(self.atoms):
            raise IndexError(f"atom_k={atom_k} out of range for K={len(self.atoms)} atoms")
        x = _as_2d_float(X, "X")
        payload = self._oos_payload(x)
        return np.asarray(payload["atoms"][k]["atom_reconstruction"], dtype=float)

    def attach_fisher(
        self, fisher_factors: Any, *, provenance: str | None = None
    ) -> "ManifoldSAE":
        """Install (or replace) the output-Fisher metric on a fitted model.

        Post-hoc companion to ``sae_manifold_fit(..., fisher_factors=...)``:
        lets a harvest→attach→steer flow add a WP-D output-Fisher shard to an
        already fitted model without refitting, so :meth:`steer` reports the
        path-integrated ``predicted_nats`` dose. Accepts the same inputs as the
        fit-time hook: a :class:`gamfit.torch.harvest.HarvestShard`, a mapping
        with ``"U"`` ``(n, p, r)`` (plus optional ``"mass_residual"`` /
        ``"provenance"``), or a raw ``(n, p, r)`` array. ``provenance``
        overrides the shard's own tag when given (must be ``"output_fisher"``
        or ``"output_fisher_downstream"``). Pass ``fisher_factors=None`` to
        detach and revert to the Euclidean metric. Returns ``self``.
        """
        if fisher_factors is None:
            self.fisher_factors = None
            self.fisher_mass_residual = None
            self.fisher_provenance = "output_fisher"
            self.metric_provenance = "Euclidean"
            return self
        if provenance is not None:
            # Route the override through the one normalizer so provenance
            # validation lives in exactly one place.
            shard_in = _normalize_fisher_factors(
                fisher_factors, int(self.fitted.shape[0]), int(self.fitted.shape[1])
            )
            fisher_factors = {
                "U": shard_in[0],
                "mass_residual": shard_in[1],
                "provenance": provenance,
            }
        u, mass_residual, shard_provenance = _normalize_fisher_factors(
            fisher_factors, int(self.fitted.shape[0]), int(self.fitted.shape[1])
        )
        self.fisher_factors = u
        self.fisher_mass_residual = mass_residual
        self.fisher_provenance = shard_provenance
        self.metric_provenance = "OutputFisher"
        return self

    def steer(self, atom_k: int, t_from: Any, t_to: Any) -> dict[str, Any]:
        """Steering plan with output dosimetry for one atom (#980).

        Drives atom ``atom_k``'s latent coordinate from ``t_from`` to ``t_to``
        and reports the *actionable* steering payload of the SAE-manifold machine
        (``gam::inference::steering::steer_delta``): the activation-space move and
        its predicted output effect, measured through the fitted model's installed
        per-row output-Fisher metric.

        Parameters
        ----------
        atom_k
            Atom index in ``[0, K)``.
        t_from, t_to
            Source / target latent coordinates, each length ``d_k`` (the atom's
            ``atom_dim``), in the atom's raw latent-coordinate units (the same
            units as ``self.coords[atom_k]``).

        Returns
        -------
        dict
            The :class:`gam::inference::steering::SteerPlan` fields:

            * ``atom`` / ``atom_name`` — the steered atom and its name;
            * ``t_from`` / ``t_to`` — the latent endpoints (lists);
            * ``amplitude`` — the atom's mean active assignment mass the move was
              scaled by;
            * ``measured_row`` — the most-active row whose per-row metric the dose
              was read through;
            * ``delta`` — ``(p,)`` activation-space move ``a·(g_k(t_to) −
              g_k(t_from))`` to add to a hidden state;
            * ``predicted_nats`` — path-integrated output-Fisher KL dose in nats,
              or ``None`` under a Euclidean (no behavioral axis) metric;
            * ``validity_radius`` — latent step length the linearization is
              trusted to, or ``None`` under a Euclidean metric;
            * ``off_manifold_norm`` — ``δ``'s component off the local decoder
              tangents (``≈ 0`` for an on-manifold move);
            * ``metric_provenance`` — ``"OutputFisher"`` when a Fisher shard was
              installed at fit time (and retained), else ``"Euclidean"``.

        The dose (``predicted_nats`` / ``validity_radius``) is available only when
        the fit installed an output-Fisher metric (``fisher_factors`` was supplied
        to :func:`sae_manifold_fit` and retained on this model); otherwise the
        geometry (``delta`` / ``off_manifold_norm``) is still returned but the dose
        degrades to ``None`` — not zero. The Fisher steering arrays now round-trip
        through :meth:`to_dict` / :meth:`save`, so a model recovered via
        :meth:`from_dict` / :meth:`load` reproduces the same dose it had at fit
        time. Only a legacy dict written before the round-trip was added (or a fit
        that never installed a Fisher metric) lacks the arrays; such a model
        steers geometry-only with ``predicted_nats`` / ``validity_radius`` ``None``.
        """
        # #2091 acceptance: the steering plan is computed by the Rust-owned core
        # off its OWN state — zero per-call re-marshalling of ``fisher_factors``
        # (or decoder blocks / coords / logits) across the FFI boundary. Only the
        # resolved atom index and the two endpoints cross. ``ManifoldSaeCore.steer``
        # is a bitwise mirror of the former Python path (guarded by
        # tests/test_manifold_sae_pyclass_steer_equiv.py); the per-kind
        # ``n_harmonics`` gate + provenance threading live in the core.
        k = self._atom_index(atom_k)
        t_from_arr = np.ascontiguousarray(np.asarray(t_from, dtype=np.float64).reshape(-1))
        t_to_arr = np.ascontiguousarray(np.asarray(t_to, dtype=np.float64).reshape(-1))
        return dict(self._core.steer(int(k), t_from_arr, t_to_arr))

    def per_atom_active_set(self, X: Any, threshold: float | None = None) -> np.ndarray:
        """Per-token active atom set ``(N, K)`` boolean mask for ``X``.

        On training ``X`` (matched bit-exactly) the cached fit assignments are
        thresholded without re-solving; otherwise the frozen-decoder OOS solve
        is run on ``X`` and its converged assignments are thresholded.

        With ``threshold=None`` the per-assignment-kind default policy from
        :func:`_active_threshold_for_assignment` is used -- the SAME policy
        :meth:`summary` applies (softmax: just above uniform mass ``1/K``;
        jumprelu: a tiny positive value so exact-zero gates are inactive;
        ibp_map: ``1e-8`` responsibility mass). The old flat ``0.5`` default was
        wrong for every kind (responsibilities sum to ~1 per row, so it
        collapsed the IBP/jumprelu mask once ``K>=2``)."""
        x = _as_2d_float(X, "X")
        cut = (
            _active_threshold_for_assignment(self.assignment, len(self.atoms))
            if threshold is None
            else float(threshold)
        )
        if self._is_training_data(x):
            return self.assignments >= cut
        payload = self._oos_payload(x)
        return np.asarray(payload["assignments_z"], dtype=float) >= cut

    def per_atom_latent_for(self, X: Any) -> list[np.ndarray]:
        """Per-atom on-manifold coordinates ``[(N, d_k), ...]`` for ``X``.

        On training ``X`` (matched bit-exactly) the cached fit coordinates are
        returned; otherwise the frozen-decoder OOS solve is run on ``X`` and its
        converged per-atom coordinates are returned."""
        x = _as_2d_float(X, "X")
        if self._is_training_data(x):
            return [c.copy() for c in self.coords]
        payload = self._oos_payload(x)
        return [np.asarray(atom["on_atom_coords_t"], dtype=float) for atom in payload["atoms"]]

    def featurize(self, X: Any) -> list[np.ndarray]:
        """Infer out-of-sample SAE coordinates for ``X``.

        This is the first-class research-loop spelling of the frozen-decoder
        OOS coordinate solve. It returns one ``(N, d_k)`` coordinate array per
        atom, in atom order, and reuses cached training coordinates when ``X``
        is the training activation matrix.
        """
        return self.per_atom_latent_for(X)

    def get_decoder(self) -> list[np.ndarray]:
        return [b.copy() for b in self.decoder_blocks]

    def get_anchors(self) -> list[np.ndarray]:
        return [c.copy() for c in self.coords]

    def description_length(self, *, l_param_bits: float | None = None) -> dict[str, Any] | None:
        """Fit-level bits/token description length — the honest headline currency.

        The manifold hypothesis is informational: a manifold in the data is a
        redundancy in the CODES, priced in **bits/token**, not in the
        matched-EV fraction of ambient residual variance that
        ``experiments/real_manifold_sae/results.md`` shows is insensitive to the
        thesis by construction. This returns the whole reconstruction's code
        length per token decomposed into code / selection / dictionary bits.

        The bits are computed by the Rust
        ``sae_manifold_description_length`` core from this fit's own empirical
        byproducts: the ``(N, K)`` assignment matrix (binarised into the support
        matrix that prices selection by the support-entropy universal code) and
        the per-atom ``(N, d_k)`` chart coordinates (whose covariance spectra set
        the reverse-water-filling latent code rate at the achieved distortion),
        plus the achieved EV and decoder scalar count; Python only marshals those
        arrays. ``l_param_bits`` overrides the per-decoder-scalar precision
        (default: the distortion-matched precision). Returns ``None`` for an
        empty fit (no atoms or no coded tokens).
        """
        k_atoms = len(self.atoms)
        n_tokens = int(self.fitted.shape[0]) if self.fitted.ndim >= 1 else 0
        if k_atoms == 0 or n_tokens == 0:
            return None
        module = rust_module()
        # The bits/token math lives entirely in the Rust core. When the loaded
        # engine predates these exports (older build or a test double), degrade
        # to ``None`` rather than crashing the report — the fit is unchanged.
        dl_fn = getattr(module, "sae_manifold_description_length", None)
        if dl_fn is None:
            return None
        assignments = np.ascontiguousarray(np.asarray(self.assignments, dtype=np.float64))
        coords: list[np.ndarray] = []
        for block in self.coords:
            arr = np.asarray(block, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            coords.append(np.ascontiguousarray(arr))
        n_params = int(sum(int(b.size) for b in self.decoder_blocks))
        return dict(
            dl_fn(
                assignments,
                coords,
                float(self.reconstruction_r2),
                int(n_params),
                l_param_bits=None if l_param_bits is None else float(l_param_bits),
            )
        )

    def summary(self) -> dict[str, Any]:
        # Active-atom detection is mode-specific. The Rust counter is INCLUSIVE
        # (active iff assignment >= threshold), so the threshold is the next
        # representable value above each conceptual cutoff (see
        # _active_threshold_for_assignment, the single shared policy):
        #   softmax   -> active if its share EXCEEDS the uniform mass 1/K
        #                (an exactly-uniform row must count zero, not all K);
        #   ibp_map   -> active if it carries responsibility mass (> 1e-8);
        #   jumprelu  -> active if the (hard) gate is NONZERO (> 0)
        #                (an exact-zero gate must NOT count as active).
        threshold = _active_threshold_for_assignment(self.assignment, len(self.atoms))
        avg_active, mean_mass = rust_module().sae_manifold_assignment_summary(self.assignments, threshold)
        # #1231: the primary key is the honest ``penalized_loss_score`` (a
        # negative penalized-loss objective, NOT REML / evidence). ``reml_score``
        # is kept as a deprecated alias key. ``None`` for closed-form shortcuts.
        score = (
            None if self.penalized_loss_score is None else float(self.penalized_loss_score)
        )
        # Lead with the honest headline currency: bits/token and the description-
        # length decomposition (code / selection / dictionary). Explained variance
        # (``reconstruction_r2``) is kept but DEMOTED below it — matched-EV is
        # insensitive to the manifold hypothesis by construction
        # (``experiments/real_manifold_sae/results.md``); bits/token is not.
        dl = self.description_length()
        bits_per_token = None if dl is None else float(dl["bits_per_token"])
        return {
            "bits_per_token": bits_per_token,
            "description_length": dl,
            "K": len(self.atoms),
            "d_atom": int(self.coords[0].shape[1]) if self.coords else 0,
            "atom_topology": self.atom_topology, "assignment": self.assignment,
            "alpha": float(self.alpha), "learnable_alpha": bool(self.learnable_alpha),
            "penalized_loss_score": score, "reml_score": score,
            # Secondary line: explained variance, demoted beneath bits/token.
            "reconstruction_r2": float(self.reconstruction_r2),
            "dispersion": float(self.dispersion),
            "atom_trust": np.asarray(self.diagnostics["atom_trust"], dtype=float).tolist(),
            "untyped_atoms": [
                i for i, diag in enumerate(self.diagnostics["atoms"]) if bool(diag["untyped"])
            ],
            "avg_active_atoms": float(avg_active), "mean_assignment_mass": float(mean_mass),
            "active_dims": [a.active_dim for a in self.atoms],
            "atom_functionals": [_json_ready(a.functional_evidence) for a in self.atoms],
            "cotrain": None if self.cotrain is None else dict(self.cotrain),
            "primitives": list(self.primitive_names),
        }

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable v1 JSON-compatible serialization — the core owns the
        schema (``reml_score`` write-alias, ``structured_residual_diagnostics``
        write-drop, channel-cov-factor compaction all live in Rust serde now)."""
        return dict(self._core.to_dict())

    def to_json(self) -> str:
        return self._core.to_json()

    def save(self, path: str | Path) -> None:
        """Write this fit to ``path`` as the canonical JSON payload the core emits."""
        Path(path).write_text(self._core.to_json())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifoldSAE":
        """Reconstruct from a :meth:`to_dict` payload. The core's ``#[new]``
        validates the ``gamfit.ManifoldSAE/v1`` schema tag and parses through
        ``ManifoldSaePayload::from_json`` (the same NaN→reject, ``reml_score``
        fallback, channel-cov reconstruction, and n_harmonics canonicalization the
        legacy reader had — now owned by the core)."""
        core = rust_module().ManifoldSaeCore(dict(payload))
        # training_data: the payload always writes null (new fits do not retain X);
        # mirror the legacy from_dict, which built a handle from the fitted shape.
        fitted = np.asarray(payload["fitted"], dtype=float)
        raw_td = payload.get("training_data")
        training_data = (
            _SaeTrainingDataHandle(tuple(fitted.shape), fitted.dtype)
            if raw_td is None
            else np.asarray(raw_td, dtype=float)
        )
        return cls(core, training_data=training_data)

    @classmethod
    def load(cls, path: str | Path) -> "ManifoldSAE":
        """Load a fit written by :meth:`save`.

        Fisher steering state is restored (see :meth:`from_dict`), so a loaded
        model reproduces :meth:`steer`'s output dosimetry. (Files written by
        versions predating Fisher round-tripping load geometry-only.)
        """
        return cls.from_dict(json.loads(Path(path).read_text()))


def gumbel_geometric_schedule(tau_start: float, tau_min: float, rate: float, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "geometric", rate=rate, iter_count=iter_count)


def gumbel_linear_schedule(tau_start: float, tau_min: float, steps: int, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "linear", steps=steps, iter_count=iter_count)


def gumbel_reciprocal_iter_schedule(tau_start: float, tau_min: float, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "reciprocal_iter", iter_count=iter_count)


_TOPOLOGY_UNSET: Any = object()
# #1777 — sentinel so `coord_sparsity` (primary) and its deprecated alias
# `gate_sparsity` can each be detected as explicitly-passed-or-not.
_COORD_SPARSITY_UNSET: Any = object()


def sae_manifold_fit(X: Any = None, K: int | None = None, d_atom: int = 2, atom_topology: Any = _TOPOLOGY_UNSET,
                     assignment: str = "softmax", schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None = None,
                     isometry_weight: float = 1.0, ard_per_atom: bool = True,
                     decoder_feature_sparsity_groups: list[list[int]] | None = None, n_iter: int = 50, *,
                     assignment_prior: Any = _ASSIGNMENT_PRIOR_UNSET, n_atoms: int | None = None,
                     sparsity_weight: float = 1.0,
                     coord_sparsity: Any = _COORD_SPARSITY_UNSET,
                     gate_sparsity: Any = _COORD_SPARSITY_UNSET, scad_mcp_gamma: float | None = None,
                     smoothness_weight: float = 1.0,
                     alpha: float | str | Any = _ALPHA_UNSET, learning_rate: float | None = None, random_state: int = 0,
                     block_orthogonality_weight: float = 0.0,
                     nuclear_norm_weight: float = 1.0, nuclear_norm_max_rank: int | None = None,
                     decoder_incoherence_weight: float = 1.0,
                     top_k: int | None = None, t_init: Any = None, a_init: Any = None,
                     tau: float | None = None, jumprelu_threshold: float = 0.0,
                     atom_basis: Any = None, fisher_factors: Any = None,
                     weights: Any = None,
                     separation_barrier_strength: float | None = None,
                     ibp_alpha: float | None = None,
                     structured_residual_passes: int = 2,
                     promote_from_residual: bool = False,
                     score_mode: str = "auto",
                     _run_structure_search: bool = True,
                     _run_outer_rho_search: bool = True) -> ManifoldSAE:
    """Fit an SAE-manifold model.

    Parameters
    ----------
    X
        Response data matrix reconstructed by the SAE. It may be a finite 1D
        or 2D numeric array; 1D input is reshaped to ``(N, 1)``.
    K
        Number of atoms. Must be positive, and the training set must satisfy
        ``N > K``. ``n_atoms`` is an alias for ``K`` (#160); supplying both with
        different values raises ``ValueError``.
    d_atom
        Intrinsic coordinate dimension per atom. Pass an int for a shared
        dimension or a length-``K`` iterable for heterogeneous atoms. ``None``
        and ``"auto"`` currently resolve to dimension 2 per atom.
    atom_topology
        Shared topology label used when ``atom_basis`` is not supplied. Common
        values are ``"circle"``, ``"periodic"``, ``"sphere"``, ``"torus"``,
        ``"linear"``, and ``"euclidean"``. If omitted, the default is
        ``"circle"``.

        NOTE (#1201): ``"euclidean"`` is a degree-2 QUADRATIC monomial patch
        (``{1, t, t²}`` at ``d_atom=1``), NOT a single straight decoder direction
        ``γ(t)=t·b``. Do not treat ``atom_topology="euclidean"`` as the "linear"
        SAE baseline — a curved-vs-``"euclidean"`` comparison is curved-vs-
        quadratic. Use ``atom_topology="linear"`` for the genuinely linear
        affine atom ``{1, t}``; the same candidate is used by the hybrid-split
        LINEAR verdicts (see :attr:`ManifoldSAE.hybrid_split`).
    assignment
        Assignment/gating family. ``"softmax"`` uses soft mixture masses and is
        the production default; at large ``K`` the fit derives a train-time
        ``top_k`` cap from rows per atom when the caller leaves ``top_k`` unset.
        ``"ibp_map"`` uses the IBP-MAP gate path as an explicit small-fit
        research mode, and ``"threshold_gate"`` uses the
        hard-sigmoid gate family (#1777, renamed from ``"jumprelu"``).
        ``"topk"`` is the hard per-row support gate (``AssignmentMode::TopK``):
        it requires an explicit ``top_k`` (the fixed active-set size), carries
        no live gate coordinates, and is therefore the one assignment admitted
        to the CURVED framed/streaming manifold lane in the overcomplete
        ``K > P`` regime (within the host memory budget; refused with an
        actionable error over it) instead of the penalty-gated sparse-code
        reroute. Public aliases are accepted (#159):
        ``"ibp"``/``"ibp-map"``/``"ibp_map"`` -> ``"ibp_map"``,
        ``"softmax"`` -> ``"softmax"``,
        ``"topk"``/``"top_k"`` -> ``"topk"``,
        ``"threshold_gate"`` (primary) and the deprecated
        ``"gated"``/``"jump_relu"``/``"jumprelu"`` -> ``"threshold_gate"``.
        ``assignment_prior`` is an alias for ``assignment`` and normalizes
        through the same validator; supplying both with conflicting resolved
        values raises ``ValueError``.
    schedule
        Optional :class:`GumbelTemperatureSchedule` or mapping forwarded to the
        IBP/Gumbel assignment path.
    isometry_weight
        Weight for ``IsometryPenalty`` on the latent coordinate block. Defaults
        to ``1.0`` (on). The Rust core compares ``g / gbar`` with the identity
        metric, where ``g = JᵀJ`` and ``gbar`` is the mean pullback trace per
        latent dimension, so the pin encourages a unit-average-speed chart
        without coupling to decoder scale (issue #795). The gauge is enabled by
        default now that both the value/gradient AND the Gauss-Newton curvature
        the joint solve majorizes with are decoder-scale-invariant (the
        curvature folds the frozen normalizer ``1 / gbar²`` so the ``‖B‖⁴``
        Gram block exactly cancels the ``‖B‖⁻⁴`` of the normalizer); the
        planted-circle default-on fit converges at every decoder scale instead
        of stalling at the proximal-ridge saturation. Set ``0.0`` to disable.
        Issue #673 (resolved): the decoder smoothness
        penalty is reparameterized by the pulled-back metric ``g = JᵀJ`` in the
        Rust core, so the roughness — and the ``penalized_loss_score`` topology
        comparison — is gauge-invariant under reparameterization of the latent
        coordinate ``t`` even with the isometry penalty off. ``IsometryPenalty``
        is purely a complementary regularizer when enabled (it drives ``g → I``
        for an interpretable, near-arc-length chart); it is not a precondition
        for comparing ``penalized_loss_score`` across topologies.
    ard_per_atom
        If true, adds per-atom ARD row-block regularization on the latent
        coordinate block to select active intrinsic coordinates.
    decoder_feature_sparsity_groups
        Optional disjoint partition of output feature indices. Emits
        ``MechanismSparsityPenalty`` on each atom's decoder block, encouraging
        basis-function rows to load on a single feature group.
    n_iter
        Maximum joint-solver iterations.
    sparsity_weight
        Non-negative assignment sparsity strength.
    coord_sparsity
        Coordinate-block sparsity penalty family (#1777, primary name for the
        former ``gate_sparsity``). The default ``"scad"`` enables adaptive
        non-convex sparsity for the recommended research objective. ``"l1"``
        keeps the historical assignment-prior sparsity path. ``"scad"`` and
        ``"mcp"`` emit the SAE row-block ``ScadMcpPenalty`` on the ``"t"``
        latent block with ``weight=sparsity_weight``.
    gate_sparsity
        Deprecated alias for ``coord_sparsity`` (#1777). Supplying both with
        different values raises ``ValueError``.
    separation_barrier_strength
        Optional per-fit value for this term's decoder-repulsion conditioner.
        ``None`` (default) uses the canonical evidence-derived strength; a finite
        value pins the strength for this fit. Threaded into the Rust
        ``SaeFitConfig``.
    ibp_alpha
        Optional per-fit IBP-α value, which controls the ordered geometric
        assignment prior. ``None`` uses the assignment mode's canonical fixed or
        learnable value; an explicit value pins α for this fit. Threaded into
        the Rust ``SaeFitConfig``.
    scad_mcp_gamma
        Optional SCAD/MCP concavity parameter. Defaults are SCAD ``3.7`` and
        MCP ``2.5``. SCAD requires ``gamma > 2``; MCP requires ``gamma > 1``.
    smoothness_weight
        Non-negative decoder smoothness weight.
        The penalty is ``0.5 * lambda * sum B.T @ S̃ @ B`` where ``S̃`` is the
        raw roughness Gram reparameterized by the decoder pullback metric
        (arc-length roughness), so it is gauge-invariant under reparameterizing
        the latent ``t`` (issue #673).
    alpha
        Assignment-prior concentration/scale. Pass a float for a fixed value or
        ``"auto"`` to mark alpha learnable in the Rust solve; returned metadata
        records ``alpha=1.0`` and ``learnable_alpha=True`` in that case. If left
        unset with an explicit ``ibp_map`` gate, the concentration defaults to
        the K-aware ``default_ibp_concentration_for_k_atoms(K) ≈ K − 1/2`` (#1784)
        so the ordered stick-breaking prior spans the whole dictionary instead of
        masking every atom past the first few (which underfit an equal-K linear
        dictionary and left the K=128 fit rank-deficient). A per-fit ``ibp_alpha``
        overrides it.
    structured_residual_passes
        Number of structured-residual whitening passes run after the primary
        joint fit. Defaults to ``2`` (#2021 — ON by default; "magic by default":
        the superposition-aware whitened metric is the best-known behavior, not
        an opt-in). Each pass fits the current reconstruction residual's
        structured covariance and installs the Σ-damped per-row metric under the
        annealed schedule ``γ_p = (p+1)/(N+1)``, so with the default ``N = 2``
        the final pass installs a majority-structured metric (``γ = 2/3``); ``2``
        is also the ``PROMOTION_NURSERY_MIN_PASSES`` dwell, so the default budget
        is coherent with the (opt-in) residual-atom promotion. Pass ``0`` to
        force the legacy iid fit (bit-identical to the pre-#2021 behavior). Must
        be a non-negative int; the native core clamps the effective count to
        ``STRUCTURED_RESIDUAL_PASSES_MAX`` (currently ``4``), so larger values
        behave like ``4``.
    promote_from_residual
        When ``True`` (default ``False``), atoms discovered in the structured
        residual passes are promoted into the primary atom tier rather than kept
        as a secondary residual dictionary. Only meaningful when
        ``structured_residual_passes > 0``. Coerced to ``bool``.
    score_mode
        Sparse front-door routing residency for the collapsed sparse-code lane.
        ``"auto"`` (default) uses CUDA only when admitted and otherwise runs the
        exact CPU router; ``"required"`` preserves fail-closed GPU residency;
        ``"off"`` is CPU-only.
    learning_rate
        Damped Newton/Gauss-Newton step size. If omitted, the Python facade uses
        ``1.0`` for IBP/softmax and ``0.05`` for JumpReLU.
    random_state
        Integer seed forwarded to the Rust initializer.
    block_orthogonality_weight
        Weight for ``BlockOrthogonalityPenalty`` on the latent coordinate block.
        Requires ``max(d_atom) >= 2`` and splits coordinate axes into singleton
        orthogonality groups.
    nuclear_norm_weight
        Weight for decoder embedding-rank selection (#672). It is on by
        default (``1.0``) for the recommended research objective. A positive value
        emits ``NuclearNormPenalty`` on each atom's ``(M_k, p)`` decoder matrix
        and shrinks its singular spectrum.
    nuclear_norm_max_rank
        Optional cap on the number of leading singular values penalized by the
        nuclear-norm decoder penalty. ``None`` leaves the rank cap disabled.
    decoder_incoherence_weight
        Cross-atom decoder column-space incoherence weight (#671). It is on by
        default (``1.0``) and applies when ``K >= 2``. The penalty uses the
        empirical co-activation ``mean_n gate_j * gate_k`` and penalizes
        ``||B_j @ B_k.T||_F^2`` for stored ``(M_k, p_out)`` decoder blocks on
        co-firing atom pairs.
    top_k
        Optional per-token active-set cap. ``None`` and ``0`` disable it;
        integers in ``[1, K]`` cap the number of atoms a token may activate. This
        is a TRAIN-TIME cap folded into the optimization (the engine builds the
        compact active×active solve over the capped support), not a cosmetic
        post-fit filter. The engine additionally applies an automatic
        memory-budget cap: when the dense ``K`` working set would exceed the
        host/device budget the compact active-set layout engages even without an
        explicit ``top_k``. ``fitted`` is computed from the (capped) support.
    t_init, a_init
        Warm starts for amortized encoder distillation (#357). ``a_init`` has
        shape ``(N, K)`` and seeds assignment logits. ``t_init`` has shape
        ``(K, N, D_max)`` with ``D_max >= max(d_atom)`` and seeds per-atom
        coordinates. ``converged_latents()``, ``encode()``, and ``project()``
        expose the refined supervision targets.
    tau
        Starting assignment temperature. If ``None`` (the default), it is
        inferred from ``schedule`` or defaults to ``0.5``.
    jumprelu_threshold
        JumpReLU hard-gate threshold. Must be finite. Defaults to ``0.0``.
    atom_basis
        Per-atom basis kind(s). If supplied with ``atom_topology``, both must
        resolve to the same topology.
    fisher_factors
        Optional WP-D output-Fisher shard (#980). Accepts a
        :class:`gamfit.torch.harvest.HarvestShard`, the dict returned by
        :func:`gamfit.torch.harvest.load_harvest_shard`, or a raw ``(n, p, r)``
        factor array. Its *presence* installs ``RowMetric::OutputFisher`` for the
        isometry gauge / lens — there is no flag (magic by default). The metric
        does not whiten the reconstruction likelihood, so the data-fit is
        identical to the Euclidean fit regardless of the isometry gauge (which
        defaults ON, ``isometry_weight=1.0``); the
        result's ``metric_provenance`` reports ``"OutputFisher"`` and the per-row
        ``fisher_mass_residual`` truncation diagnostic rides into the model.
        ``None`` (default) keeps the bit-identical Euclidean path.
    weights
        Optional per-row design-honesty reconstruction weights (#977): a
        length-``N`` array of strictly positive ``√w`` multipliers, one per
        observation. When supplied, each per-row reconstruction loss is scaled
        by its weight in the inner joint fit and the outer ρ (smoothness /
        sparsity / ARD) selection — the seam for honest fitting on a designed
        corpus subsample or an importance-weighted training set. The vector is
        self-normalized to mean 1 inside the core; a uniform or absent vector
        is the bit-identical unweighted path (magic by default — no flag).

    Returns
    -------
    ManifoldSAE
        Fitted result. Core attributes are ``atoms`` (list of
        :class:`SaeManifoldAtomFit`), ``fitted`` ``(N, p)``, ``assignments``
        ``(N, K)``, ``coords`` as per-atom ``(N, d_k)`` arrays,
        ``decoder_blocks`` as per-atom ``(M_k, p)`` decoder matrices,
        ``basis_specs``, ``atom_topology``/``atom_topologies``, ``assignment``
        and ``assignment_label``, ``penalized_loss_score`` (``reml_score`` is a
        deprecated read alias), ``reconstruction_r2``,
        ``dispersion``, ``training_mean``, metadata-only ``training_data``,
        lazy ``low_level_logits``, and fit-control metadata including ``alpha``,
        ``learnable_alpha``, ``tau``, ``sparsity_strength``, ``smoothness``,
        ``learning_rate``, ``max_iter``, ``random_state``, ``top_k``, and
        ``jumprelu_threshold``. Each atom exposes ``basis``,
        ``decoder_coefficients`` ``(M_k, p)``, per-atom ``assignments`` ``(N,)``,
        recovered ``coords`` ``(N, d_k)``, ``evidence``, ``active_dim``,
        ``decoder_covariance`` ``(M_k*p, M_k*p)``, ``shape_band_coords``
        ``(G, d_k)``, ``shape_band_mean`` ``(G, p)``, and ``shape_band_sd``
        ``(G, p)`` when the Rust payload includes posterior shape uncertainty.

        Useful public methods include ``predict``/``reconstruct``,
        ``reconstruct_training``, ``encode``, ``converged_latents``,
        ``project``, ``per_atom_active_set``, ``per_atom_latent_for``, and
        ``shape_uncertainty(atom=..., n_sd=...)``.
    """
    if X is None:
        raise TypeError("sae_manifold_fit requires X input array")
    x = _as_2d_float(X, "X")
    # `K` and `n_atoms` are aliases for the number of atoms (#160). If both are
    # supplied with DIFFERENT values, raise an eager ValueError naming both;
    # equal values pass through. Resolve before any Rust call.
    if K is not None and n_atoms is not None and int(K) != int(n_atoms):
        raise ValueError(
            f"K and n_atoms both supplied with different values "
            f"({int(K)} vs {int(n_atoms)}); pass only one (they are aliases)."
        )
    k_resolved = K if K is not None else n_atoms
    k_atoms = int(k_resolved if k_resolved is not None else 0)
    max_iter_total = int(n_iter)
    smoothness = float(smoothness_weight)
    sparsity = float(sparsity_weight)
    # #1777 — `coord_sparsity` is the primary name for the coordinate-block penalty
    # family; `gate_sparsity` is retained as a deprecated alias. Both normalize to a
    # single resolved value; supplying both with conflicting values raises.
    coord_given = coord_sparsity is not _COORD_SPARSITY_UNSET
    gate_given = gate_sparsity is not _COORD_SPARSITY_UNSET
    if coord_given and gate_given:
        if str(coord_sparsity).strip().lower() != str(gate_sparsity).strip().lower():
            raise ValueError(
                "coord_sparsity and gate_sparsity (a deprecated alias) were both "
                f"supplied with different values ({coord_sparsity!r} vs "
                f"{gate_sparsity!r}); pass only coord_sparsity."
            )
        coord_sparsity_resolved = coord_sparsity
    elif coord_given:
        coord_sparsity_resolved = coord_sparsity
    elif gate_given:
        coord_sparsity_resolved = gate_sparsity
    else:
        coord_sparsity_resolved = "scad"
    gate_sparsity = coord_sparsity_resolved
    gate_sparsity_kind = str(coord_sparsity_resolved).strip().lower()
    if gate_sparsity_kind not in {"l1", "scad", "mcp"}:
        raise ValueError(
            "coord_sparsity (alias gate_sparsity) must be one of 'l1', 'scad', or "
            f"'mcp'; got {coord_sparsity_resolved!r}"
        )
    # #1777 — per-fit overrides must be finite when supplied; ibp_alpha must be
    # strictly positive (it scales the ordered geometric assignment prior).
    if separation_barrier_strength is not None and not np.isfinite(
        float(separation_barrier_strength)
    ):
        raise ValueError(
            "separation_barrier_strength must be finite or None; "
            f"got {separation_barrier_strength}"
        )
    if ibp_alpha is not None and not (
        np.isfinite(float(ibp_alpha)) and float(ibp_alpha) > 0.0
    ):
        raise ValueError(
            f"ibp_alpha must be finite and > 0 or None; got {ibp_alpha}"
        )
    # Structured-residual sculpting is an explicit, typed opt-in. The count must
    # be a non-negative int (it is clamped natively to
    # `STRUCTURED_RESIDUAL_PASSES_MAX`); `promote_from_residual` is coerced to a
    # plain bool for the pyfunction kwarg.
    structured_residual_passes = int(structured_residual_passes)
    if structured_residual_passes < 0:
        raise ValueError(
            "structured_residual_passes must be a non-negative int (it is "
            "clamped natively to STRUCTURED_RESIDUAL_PASSES_MAX); "
            f"got {structured_residual_passes}"
        )
    promote_from_residual = bool(promote_from_residual)
    if scad_mcp_gamma is None:
        scad_mcp_gamma_value = 3.7 if gate_sparsity_kind == "scad" else 2.5
    else:
        scad_mcp_gamma_value = float(scad_mcp_gamma)
    tau = float(tau if tau is not None else _schedule_tau_start(schedule, 0.5))
    jumprelu_threshold = float(jumprelu_threshold)
    if k_atoms <= 0:
        raise ValueError(f"K must be positive, got {k_atoms}")
    if max_iter_total < 1:
        raise ValueError(f"n_iter must be >= 1, got {max_iter_total}")
    # Eager n-sample validation (issue #183). One sample yields a
    # degenerate decoder LSQ system and a near-zero total sum of squares
    # — the resulting R² can be astronomically negative. Require at least
    # two observations, and at least as many observations as atoms so the
    # joint decoder block is identifiable.
    n_obs = int(x.shape[0])
    if n_obs < 2:
        raise ValueError(
            f"sae_manifold_fit requires n >= 2 observations; got n={n_obs}"
        )
    if n_obs <= k_atoms:
        # Overcomplete regime (K >= n) — the normal sparse-autoencoder setting. The
        # joint decoder LSQ is underdetermined by raw counts, but the ARD coord
        # prior + smoothness penalty regularize it to identifiability, so this is
        # admissible: it relies on the priors rather than on n > K. Warn instead of
        # refusing so massive-K dictionaries (e.g. K=32,000) can be fit on a
        # RAM-tight box with modest n — the dense n×K assignment logits scale with
        # n, so a small n keeps peak memory bounded.
        import warnings as _warnings
        _warnings.warn(
            f"sae_manifold_fit: overcomplete K={k_atoms} >= n={n_obs}; decoder "
            f"identified by ARD/smoothness priors, not n > K.",
            stacklevel=2,
        )
    # WP-D output-Fisher shard (#980). Magic-by-default: a non-None
    # `fisher_factors` (HarvestShard / load_harvest_shard dict / raw (n, p, r)
    # array) activates `RowMetric::OutputFisher` in the Rust core. Validate +
    # coerce here against the (n, p) response; ship the (n, p, r) U and the
    # optional (n,) mass_residual through the FFI. Absent ⇒ Euclidean path.
    fisher_shard = _normalize_fisher_factors(fisher_factors, n_obs, int(x.shape[1]))
    # Per-row design-honesty reconstruction weights (#977). When supplied, the
    # length-`n_obs` √w vector reweights every per-row reconstruction loss in
    # the inner joint fit and the outer ρ selection (installed Rust-side via
    # `SaeManifoldTerm::set_row_loss_weights`). Validate against the response
    # row count here; a uniform / absent vector self-normalizes to the exact
    # unweighted path. No flag — its presence is the switch (magic by default).
    row_loss_weights_arr: np.ndarray | None
    if weights is None:
        row_loss_weights_arr = None
    else:
        row_loss_weights_arr = np.ascontiguousarray(
            np.asarray(weights, dtype=float).reshape(-1)
        )
        if row_loss_weights_arr.shape[0] != n_obs:
            raise ValueError(
                "sae_manifold_fit: weights must have one entry per observation; "
                f"got {row_loss_weights_arr.shape[0]} for n={n_obs}"
            )
        if not np.all(np.isfinite(row_loss_weights_arr)) or np.any(
            row_loss_weights_arr <= 0.0
        ):
            raise ValueError(
                "sae_manifold_fit: weights must be finite and strictly positive"
            )
    dims = _dims(k_atoms, d_atom)
    # Eager d_atom validation (issue #184). A zero-dimensional atom carries
    # no manifold coordinate, contributes nothing to reconstruction, and
    # leaves `active_dims = [0, ...]` — that is a silent no-op that should
    # be a hard error, matching how `K <= 0` and `n_iter <= 0` are
    # rejected.
    if any(d < 1 for d in dims):
        raise ValueError(
            f"d_atom must be >= 1 for every atom; got {dims}"
        )
    # #2098 (SPEC-8) / F6 — the heterogeneous-`d_atom` + row-block-penalty
    # compatibility rule is validated inside the Rust engine
    # (`SaeManifoldTerm::validate_heterogeneous_atom_compatibility`, called in
    # `sae_manifold_fit_inner`). The DIM-ADAPTIVE row-block penalties (native ARD,
    # SCAD-MCP coord sparsity, sparsity, isometry gauge) compose per atom over a
    # mixed "t" block and are ADMITTED; only the FIXED-`d` structural penalties
    # (block-orthogonality, TopK/JumpReLU, row-precision) require a uniform
    # atom_dim and are refused with a direct `ValueError` up front. The facade
    # stays thin and simply surfaces that engine decision rather than duplicating
    # the check here.
    # Eager sparsity_weight validation (issue #184). The signature
    # advertises `sparsity_weight: float = 1.0`; `0.0` is the canonical
    # "no sparsity" baseline and must be accepted. Reject only negative,
    # NaN, and infinite values here so the Rust kernel can apply its own
    # log-domain floor.
    if not np.isfinite(sparsity) or sparsity < 0.0:
        raise ValueError(
            f"sparsity_weight must be finite and non-negative; got {sparsity}"
        )
    if gate_sparsity_kind == "scad":
        if not (np.isfinite(scad_mcp_gamma_value) and scad_mcp_gamma_value > 2.0):
            raise ValueError(
                "scad_mcp_gamma must be finite and > 2 for coord_sparsity='scad'; "
                f"got {scad_mcp_gamma_value}"
            )
    elif gate_sparsity_kind == "mcp":
        if not (np.isfinite(scad_mcp_gamma_value) and scad_mcp_gamma_value > 1.0):
            raise ValueError(
                "scad_mcp_gamma must be finite and > 1 for coord_sparsity='mcp'; "
                f"got {scad_mcp_gamma_value}"
            )
    if not np.isfinite(jumprelu_threshold):
        raise ValueError(
            f"jumprelu_threshold must be finite; got {jumprelu_threshold}"
        )
    # Gauge-invariance of the topology evidence (issue #673, resolved). The
    # decoder smoothness penalty is reparameterized by the decoder pullback
    # metric g = J^T J in the Rust core (arc-length roughness; see
    # `SaeManifoldAtom::refresh_intrinsic_smooth_penalty`), so the roughness —
    # and therefore the Occam / joint-log-det terms that enter the
    # `penalized_loss_score` — is invariant under reparameterizing the latent
    # coordinate t. Topology comparison (e.g. circle vs euclidean) is thus well
    # posed regardless of `isometry_weight`. `IsometryPenalty` is purely a
    # complementary regularizer that drives g -> I for an interpretable
    # near-arc-length chart; turning it off does not make `penalized_loss_score`
    # gauge-dependent, so there is nothing to warn about.
    # NOTE(#795): isometry now defaults ON. The Rust penalty normalizes
    # g = J^T J by the mean trace per latent dimension (`gbar`) before comparing
    # to I, so the value and gradient no longer scale as decoder^4. The earlier
    # curvature-walk bifurcation that forced the stopgap default-off was the
    # SAE arrow-Schur Gauss-Newton curvature: it was assembled from the raw
    # weighted Jacobian (∝‖B‖⁴) while the gradient was scale-free, so a large
    # decoder collapsed the joint Newton step and the proximal ridge saturated
    # at 1e15. The assembled curvature now folds the frozen normalizer
    # `1 / gbar² (∝‖B‖⁻⁴)` into htt/htbeta/hbb, exactly cancelling the ‖B‖⁴
    # Gram block, so the planted-circle default-on fit converges at every decoder
    # scale (see `sae_isometry_joint_fit_converges_across_decoder_scales`).
    # Eager nuclear_norm_weight validation (issue #672). `0.0` is the canonical
    # "no rank penalty" baseline; reject negative / non-finite values so the
    # descriptor builder does not surface a cryptic Rust error.
    if not np.isfinite(nuclear_norm_weight) or nuclear_norm_weight < 0.0:
        raise ValueError(
            f"nuclear_norm_weight must be finite and non-negative; "
            f"got {nuclear_norm_weight}"
        )
    if nuclear_norm_max_rank is not None and int(nuclear_norm_max_rank) < 1:
        raise ValueError(
            f"nuclear_norm_max_rank must be >= 1 (or None to disable the cap); "
            f"got {nuclear_norm_max_rank}"
        )
    # Eager decoder_incoherence_weight validation (issue #671). On by default
    # (1.0); applies only for k_atoms >= 2 (it penalizes co-activating atom
    # pairs). Reject negative / non-finite values.
    if not np.isfinite(decoder_incoherence_weight) or decoder_incoherence_weight < 0.0:
        raise ValueError(
            f"decoder_incoherence_weight must be finite and non-negative; "
            f"got {decoder_incoherence_weight}"
        )
    topology_supplied = atom_topology is not _TOPOLOGY_UNSET
    # Magic default (#2238/#2239): when the caller names no topology, every
    # atom is seeded "auto" and the Rust fit entry races circle / torus /
    # sphere / flat-2-D per atom by REML evidence over its seed cluster —
    # the historical pinned circle hard-capped intrinsically 2-D factors at
    # R² ≈ 0.5. An explicit atom_topology still pins exactly as before.
    atom_topology_str = str(atom_topology) if topology_supplied else "auto"
    bases = _bases(k_atoms, atom_basis, atom_topology_str)
    resolved_topology = _topology_for_bases(bases)
    # O: compare CANONICAL forms on both sides. Comparing the resolved (already
    # canonical) topology against the RAW user string falsely flagged valid
    # documented alias pairs (e.g. atom_topology="periodic" + atom_basis=
    # ["periodic"], where the basis side resolves to "circle").
    if (
        topology_supplied
        and atom_basis is not None
        and resolved_topology != _canonical_topology(atom_topology_str)
    ):
        raise ValueError(
            f"sae_manifold_fit: atom_basis={atom_basis!r} resolves to topology "
            f"{resolved_topology!r} but atom_topology={atom_topology_str!r} "
            f"(canonical {_canonical_topology(atom_topology_str)!r}) was also "
            f"supplied; they must describe the same topology."
        )
    kind = _resolve_public_assignment(assignment, assignment_prior)
    # #1784 — K-aware default IBP concentration. When the caller does not set
    # `alpha` and explicitly chooses the ordered stick-breaking `ibp_map` gate,
    # default the concentration to `default_ibp_concentration_for_k_atoms(K)`
    # so the prior SPANS the whole dictionary instead of collapsing to a near-hard
    # mask past the first ~3 atoms (the fixed `alpha=1.0` failure that made the
    # manifold underfit an equal-K linear dictionary and left late atoms massless,
    # rank-deficient at K=128). A per-fit `ibp_alpha` still wins in Rust
    # (`resolved_ibp_alpha`), so this only moves the *base* default.
    # `alpha="auto"` (learnable) and every
    # non-`ibp_map` gate keep the historical `1.0` seed.
    alpha_is_auto = alpha == "auto"
    if alpha is _ALPHA_UNSET:
        if kind == "ibp_map" and ibp_alpha is None:
            alpha_value = _default_ibp_concentration_for_k_atoms(k_atoms)
        else:
            alpha_value = 1.0
        alpha_is_auto = False
    else:
        alpha_value = 1.0 if alpha_is_auto else float(alpha)
    # Magic-by-default learning rate: the SAE Newton kernel is a damped
    # Gauss-Newton step against a quadratic local model with Armijo
    # backtracking. For softmax / IBP-MAP assignments the natural full step
    # is `lr=1.0` (matches the Rust reference test
    # `sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2`, which reaches
    # R² ≥ 0.95 in 10 steps from a phase-shifted init). A small literal
    # `lr=0.05` starves the assignment posterior of gradient mass and lets
    # the IBP sigmoid drift into the saturated tail (the issue #165
    # collapse: assignment mass ~1e-146). The ThresholdGate (#1777, formerly
    # "jumprelu") keeps the historical smaller step because its hard-gate STE is
    # more sensitive to overshooting the threshold. Callers can still override.
    if learning_rate is None:
        effective_lr = 0.05 if kind == "threshold_gate" else 1.0
    else:
        effective_lr = float(learning_rate)
    penalties = [n for n, ok in (("IsometryPenalty", isometry_weight > 0.0), ("ARDPenalty", ard_per_atom),
        ("ScadMcpPenalty", gate_sparsity_kind in {"scad", "mcp"} and sparsity > 0.0),
        ("MechanismSparsityPenalty", decoder_feature_sparsity_groups is not None),
        ("BlockOrthogonalityPenalty", block_orthogonality_weight > 0.0),
        ("NuclearNormPenalty", nuclear_norm_weight > 0.0),
        ("DecoderIncoherencePenalty", decoder_incoherence_weight > 0.0 and k_atoms >= 2)) if ok]
    # Build the analytic-penalty registry payload that `sae_manifold_fit_minimal`
    # passes into `run_joint_fit_arrow_schur`. Row-block descriptors target the
    # SAE latent block "t" (shape (n_obs, d_max), where d_max = max(d_atom) —
    # matches the registry latent built in `sae_manifold_fit_inner`). Issue #240:
    # previously these knobs only populated `primitive_names` metadata.
    analytic_penalties_json = _build_analytic_penalties_payload(
        isometry_weight=isometry_weight,
        gate_sparsity=gate_sparsity_kind,
        sparsity_weight=sparsity,
        scad_mcp_gamma=scad_mcp_gamma_value,
        decoder_feature_sparsity_groups=decoder_feature_sparsity_groups,
        block_orthogonality_weight=block_orthogonality_weight,
        nuclear_norm_weight=nuclear_norm_weight,
        nuclear_norm_max_rank=nuclear_norm_max_rank,
        decoder_incoherence_weight=decoder_incoherence_weight,
        k_atoms=k_atoms,
        d_max=max(dims),
        p_out=int(x.shape[1]),
    )
    # `None` disables the active-set cap; anything in `[1, k_atoms]` is forwarded
    # to the Rust driver, which folds the cap into the OPTIMIZATION as a
    # train-time per-token active-set cap (it builds the compact active×active
    # solve over the capped support and computes `fitted` from it) — NOT a
    # cosmetic post-fit projection. The driver also auto-caps the active set when
    # the dense `K` working set would exceed the memory budget. The Rust kernel
    # owns the cap contract end to end — there is no Python-side mask. Any value
    # outside `[1, k_atoms]` is a caller error rather than a silent clamp/no-op.
    # I: the docstring advertises that both ``None`` and ``0`` disable the
    # active-set cap. Normalize ``0`` to ``None`` (disabled) BEFORE the
    # ``[1, K]`` range check so ``top_k=0`` is accepted rather than rejected.
    if top_k is None or int(top_k) == 0:
        top_k_arg = (
            _default_top_k_for_large_dictionary(n_obs, k_atoms)
            if kind == "softmax"
            else None
        )
    else:
        top_k_int = int(top_k)
        if top_k_int < 1 or top_k_int > k_atoms:
            raise ValueError(
                f"top_k must be in [1, K={k_atoms}] (or None to disable); "
                f"got {top_k_int}"
            )
        else:
            top_k_arg = top_k_int
    # The hard top-k support gate (`AssignmentMode::TopK`) has no default
    # support size: its per-row active set IS the model. Require it eagerly so
    # a K > P topk request can never fall through to the penalty-gated K-vs-P
    # rule below (which would reroute a MANIFOLD request to the linear trainer
    # — the exact silent substitution the front door exists to prevent).
    if kind == "topk" and top_k_arg is None:
        raise ValueError(
            f"sae_manifold_fit: assignment='topk' requires top_k (the fixed per-row "
            f"active-set size, in [1, K={k_atoms}])"
        )
    # Front-door lane admission, owned by the Rust front door so the Python
    # public entry and the FFI boundary share one rule:
    #   * penalty-gated assignments (softmax / ibp_map / threshold_gate) carry
    #     live N x K Newton logits, so the dense exact manifold engine is their
    #     small-K certification lane only: once K > P they route to the
    #     sparse-code trainer before constructing dense logits / coordinates;
    #   * assignment='topk' carries NO gate coordinates (read-only routing,
    #     per-row active sets of size top_k), so K > P is admitted to the
    #     CURVED framed/streaming lane ("curved_streaming") within the host
    #     memory budget, and refused with an actionable error over it — never
    #     substituted with the linear lane.
    admission = _sae_fit_admission(
        n_obs,
        int(x.shape[1]),
        k_atoms,
        d_max=max(dims),
        topk_support=(top_k_arg if kind == "topk" else None),
    )
    if admission["lane"] == "sparse_codes":
        if a_init is not None or t_init is not None:
            raise ValueError(
                "sae_manifold_fit sparse front-door lane does not accept dense "
                "a_init/t_init warm starts; provide sparse dictionary state instead."
            )
        sparse_active = int(top_k_arg if top_k_arg is not None else 1)
        return sparse_dictionary_fit(
            np.ascontiguousarray(x, dtype=np.float32),
            k_atoms,
            active=sparse_active,
            max_epochs=max_iter_total,
            score_mode=str(score_mode),
        )
    # Warm starts (issue #357): `a_init` (N, K) seeds the assignment logits and
    # `t_init` (K, N, D_max) seeds the per-atom on-manifold coordinates, so an
    # amortized encoder can predict `(a_init, t_init)` and have the joint solver
    # refine them for a bounded `n_iter` steps. Both are optional and validated
    # eagerly here against (N, K) / (K, N, D_max) where D_max = max(dims).
    d_max = max(dims)
    logits_init = None
    if a_init is not None:
        logits_init = np.ascontiguousarray(np.asarray(a_init, dtype=np.float64))
        if logits_init.shape != (n_obs, k_atoms):
            raise ValueError(
                f"sae_manifold_fit: a_init must have shape (N, K)=({n_obs}, {k_atoms}); "
                f"got {logits_init.shape}"
            )
    coords_init = None
    if t_init is not None:
        coords_init = np.ascontiguousarray(np.asarray(t_init, dtype=np.float64))
        if coords_init.ndim != 3 or coords_init.shape[0] != k_atoms or coords_init.shape[1] != n_obs:
            raise ValueError(
                f"sae_manifold_fit: t_init must have shape (K, N, D_max)=({k_atoms}, {n_obs}, >={d_max}); "
                f"got {coords_init.shape}"
            )
        if coords_init.shape[2] < d_max:
            raise ValueError(
                f"sae_manifold_fit: t_init D_max={coords_init.shape[2]} is too small for "
                f"max atom dim {d_max}"
            )
    # SPEC: the SAE fit is a Rust solver. All fits route through the
    # `sae_manifold_fit_minimal` FFI; the former numpy closed-form "fast path"
    # (disjoint-periodic top-1 / dense-periodic IBP-LSQ) was a Python
    # reimplementation of the Rust joint fit and has been removed.
    payload = rust_module().sae_manifold_fit_minimal(
        np.ascontiguousarray(x),
        [str(b) for b in bases],
        [int(d) for d in dims],
        float(alpha_value),
        float(tau),
        bool(alpha_is_auto),
        str(kind),
        sparsity_strength=float(sparsity),
        smoothness=float(smoothness),
        max_iter=int(max_iter_total),
        learning_rate=float(effective_lr),
        gumbel_schedule=_schedule_payload(schedule),
        analytic_penalties=analytic_penalties_json,
        random_state=int(random_state),
        top_k=top_k_arg,
        initial_logits=logits_init,
        initial_coords=coords_init,
        jumprelu_threshold=float(jumprelu_threshold),
        # #240: `ard_per_atom` is the user-facing ARD switch. The ONLY thing that
        # actually enables/disables ARD in the SAE objective is the native
        # `ArdAxisPrior`, gated by `native_ard_enabled` (it sizes each atom's
        # `log_ard` to `d` when on, length-0 when off, adding/removing those
        # per-atom precisions from the outer ρ search and the inner Arrow-Schur
        # prior). The registry `{"kind":"ard"}` descriptor is deliberately a
        # no-op on every SAE path (`AnalyticPenaltyKind::Ard(_)` is skipped in
        # both the gradient assembly and the value total — the native prior is
        # the single source of truth, avoiding a double-counted, period-
        # discontinuous ½λt² energy). So route the flag to the switch that works
        # instead of leaving it a dead toggle (bit-identical fits on/off).
        native_ard_enabled=bool(ard_per_atom),
        fisher_factors=None if fisher_shard is None else fisher_shard[0],
        fisher_mass_residual=None if fisher_shard is None else fisher_shard[1],
        fisher_provenance=None if fisher_shard is None else fisher_shard[2],
        row_loss_weights=row_loss_weights_arr,
        # Per-fit config. `None` selects the canonical data-derived or assignment
        # default; a value pins the strength/α for this fit.
        separation_barrier_strength_override=(
            None if separation_barrier_strength is None else float(separation_barrier_strength)
        ),
        ibp_alpha_override=None if ibp_alpha is None else float(ibp_alpha),
        structured_residual_passes=int(structured_residual_passes),
        promote_from_residual=bool(promote_from_residual),
        run_structure_search=bool(_run_structure_search),
        run_outer_rho_search=bool(_run_outer_rho_search),
    )
    payload_dict = dict(payload)
    # #2091 — ONE builder call returns the Rust-owned ManifoldSAE facade. The
    # post-fit Fisher attach (fisher_shard) and the linear_block relabel (bases)
    # are threaded into the Rust builder as `fisher_factors=`/`fisher_provenance=`
    # and `declared_bases=`, replacing the old post-construction mutations +
    # `_preserve_linear_block_labels`.
    model = ManifoldSAE.from_payload(
        x, payload_dict, resolved_topology, kind, penalties,
        assignment_label=str(assignment),
        alpha=float(alpha_value), learnable_alpha=bool(alpha_is_auto),
        tau=float(tau), sparsity_strength=float(sparsity), smoothness=float(smoothness),
        learning_rate=float(effective_lr), max_iter=int(max_iter_total),
        random_state=int(random_state), top_k=top_k_arg,
        jumprelu_threshold=float(jumprelu_threshold),
        fisher_factors=None if fisher_shard is None else np.ascontiguousarray(fisher_shard[0]),
        fisher_provenance=None if fisher_shard is None else fisher_shard[2],
        declared_bases=list(bases),
    )
    return model


# --------------------------------------------------------------------------- #
# Sequential Atom Composition (SAC) — the stagewise adapter (#2027 / SAC WS-A). #
# --------------------------------------------------------------------------- #
#
# The Rust ``fit_stagewise`` driver (crates/gam-sae/src/manifold/stagewise.rs)
# grows a curved dictionary ONE atom at a time from a single K=1 seed: forward
# births (each seeds from the running residual factor and races a new atom vs a
# chart extension under an evidence + minimum-effect gate), backfitting sweeps
# (keep-best, monotone at fixed ρ), then a terminal frozen joint-evidence pass.
# It exists because the cold-start joint fit of K>1 curved atoms co-collapses on
# real activations while every K=1 fit succeeds; SAC retires the simultaneous
# cold start and builds K from the proven K=1 path (guards disarmed — the K=1
# lane never trips them). The whole driver is Rust; this adapter is a THIN shell
# (SPEC): it assembles the K=1 SEED via the proven Rust ``sae_manifold_fit`` and
# rebuilds the atom's basis (Φ / dΦ / roughness Gram) via the Rust
# ``basis_with_jet`` kernel, then hands those verbatim to the compact stagewise
# FFI. No model math is computed here — only array packing.


@dataclass(slots=True)
class StagewiseAtom:
    """One atom of a SAC-composed dictionary.

    Attributes
    ----------
    decoder
        Decoder basis coefficients ``(M_k, p)`` (same contract as
        :attr:`SaeManifoldAtomFit.decoder_coefficients`).
    coords
        Recovered on-atom coordinates ``(N, d_k)``.
    assignments
        Per-observation gate for this atom ``(N,)``.
    topology
        Atom topology label (``"circle"`` / ``"sphere"`` / ...).
    latent_dim
        Intrinsic coordinate dimension ``d_k``.
    delta_ev
        The ΔEV this atom earned at its accepting birth (``None`` for the seed
        atom, which is atom 0). This is the per-atom salience the birth ledger
        recorded — the discriminator's "each atom earns its ΔEV" datum.
    theta
        Fitted turning ``Θ`` of the atom's chart. ``None`` here: the compact
        stagewise FFI does not emit the hybrid-split turning report, so ``Θ`` is
        left to the eval lane, which recomputes it from :attr:`decoder`. Kept in
        the schema so the ``(Θ, ΔEV)`` frontier has a home per atom.
    """

    decoder: np.ndarray
    coords: np.ndarray
    assignments: np.ndarray
    topology: str
    latent_dim: int
    delta_ev: float | None
    theta: float | None = None


@dataclass(slots=True)
class StagewiseSAE:
    """A SAC-composed manifold dictionary and its discriminator instrumentation.

    The headline discriminator lives in the traces and the birth ledger, not in
    a separate run (LANE_PLAN): ``ev_trace`` is non-decreasing in births *by
    construction* (every adopted candidate cleared ``ΔEV >= min_effect_ev >= 0``),
    ``backfit_ev_trace`` is non-decreasing under keep-best, ``birth_records`` logs
    every round (accepted new-atom / chart-extension / rejection) with its ΔEV
    and the frozen joint-REML before/after, and ``collapse_events`` is the
    live-decoder collapse log — empty by construction (atoms never compete inside
    one Hessian), which IS the answer to the old joint-vs-grown collapse question
    on the real target.
    """

    atoms: list[StagewiseAtom]
    logits: np.ndarray
    ev_trace: np.ndarray
    backfit_ev_trace: np.ndarray
    births_accepted: int
    births_rejected: int
    stopped_reason: str
    terminal_joint_reml: float
    terminal_data_fit: float
    birth_records: list[dict[str, Any]]
    collapse_events: list[dict[str, Any]]
    log_lambda_sparse: float
    log_lambda_smooth: np.ndarray
    log_ard: list[np.ndarray]
    assignment: str
    seed: ManifoldSAE
    training_data: np.ndarray
    #: #1939 — the resolved cone-atom RECOVERY opt-in the fit actually ran with
    #: (echoed from the FFI). Lets a harness verify the kwarg engaged rather than
    #: assuming; default false keeps older constructors valid.
    cone_atom_recovery_used: bool = False
    #: #5/(B) — echoed rank-charge opt-in (the value the fit ran with), so red-tree's
    #: A/B harness can verify the flag engaged; default false keeps older payloads valid.
    rank_charge_evidence_used: bool = False

    @property
    def k(self) -> int:
        """Number of atoms in the composed dictionary."""
        return len(self.atoms)

    def _in_sample_reconstruction(self) -> np.ndarray:
        """Composed reconstruction ``Σ_k a_k · (Φ_k B_k)`` of the training target.

        The atoms carry their converged coordinates/gates; Rust evaluates the
        bases, applies the decoders, and sums the gated atom contributions.
        Returns ``(N, p)``.
        """
        if not self.atoms:
            return np.zeros_like(self.training_data, dtype=np.float64)
        decoder_blocks = [
            np.asarray(atom.decoder, dtype=np.float64) for atom in self.atoms
        ]
        coords = [np.asarray(atom.coords, dtype=np.float64) for atom in self.atoms]
        assignments = np.ascontiguousarray(
            np.column_stack(
                [np.asarray(atom.assignments, dtype=np.float64).reshape(-1) for atom in self.atoms]
            )
        )
        atom_basis = [
            _TOPOLOGY_TO_BASIS.get(_canon_name(atom.topology), _canon_name(atom.topology))
            for atom in self.atoms
        ]
        atom_dims = [int(atom.latent_dim) for atom in self.atoms]
        return _sae_manifold_reconstruct_native(
            atom_basis,
            atom_dims,
            decoder_blocks,
            coords,
            assignments,
            int(decoder_blocks[0].shape[1]),
        )

    @property
    def fitted(self) -> np.ndarray:
        """In-sample composed reconstruction ``(N, p)`` (mirrors
        :attr:`ManifoldSAE.fitted`)."""
        return self._in_sample_reconstruction()

    def to_manifold_sae(self) -> "ManifoldSAE":
        """Lift the SAC-composed frozen dictionary into a :class:`ManifoldSAE`.

        The composed atoms (frozen decoders + their circle/sphere analytic bases)
        are packed into the SAME per-atom layout
        :meth:`ManifoldSAE._oos_payload` / the Rust ``sae_manifold_predict_oos``
        FFI already consume, so the returned object exposes the existing
        out-of-sample surface — ``reconstruct(X_new)`` / ``encode`` /
        ``project`` — with NO new numerical path: the frozen-decoder OOS chart
        routing/encode lives in Rust and is reused verbatim, this is pure array
        marshalling (SPEC thin-wrapper rule).

        Scalar fit controls (``alpha`` / ``tau`` / ``assignment`` / learning
        rate / ...) are inherited from the K=1 :attr:`seed`, so the held-out
        solve runs under the same gate family the dictionary was grown with. The
        lifted object's ``training_data``/``fitted`` are the SAC target and its
        composed in-sample reconstruction, so scoring the exact training matrix
        returns the SAC reconstruction bit-for-bit while any fresh ``X`` takes the
        Rust OOS solve.
        """
        seed = self.seed
        if not self.atoms:
            # Empty dictionary: nothing composed to route out of sample; the K=1
            # seed IS the model (it already exposes the OOS surface).
            return seed
        training = np.ascontiguousarray(np.asarray(self.training_data, dtype=np.float64))
        fitted = np.ascontiguousarray(self._in_sample_reconstruction())
        basis_kinds = [
            _TOPOLOGY_TO_BASIS.get(_canon_name(a.topology), _canon_name(a.topology))
            for a in self.atoms
        ]
        decoder_blocks = [
            np.ascontiguousarray(np.asarray(a.decoder, dtype=np.float64)) for a in self.atoms
        ]
        atom_dims = [int(a.latent_dim) for a in self.atoms]
        basis_sizes = [int(b.shape[0]) for b in decoder_blocks]
        # Periodic harmonics recovered DIRECTLY from the decoder width (M = 2H + 1):
        # ``(M - 1) // 2`` with no floor, so a DC-only born atom (M = 1 → H = 0)
        # keeps its constant basis instead of being inflated to a 3-column periodic
        # design the frozen decoder no longer matches. Sphere / non-periodic pass
        # through as 0 (their basis size is fixed, not harmonic-derived).
        n_harmonics = [
            ((size - 1) // 2 if kind in ("periodic", "periodic_spline") else 0)
            for kind, size in zip(basis_kinds, basis_sizes)
        ]
        coords = [np.ascontiguousarray(np.asarray(a.coords, dtype=np.float64)) for a in self.atoms]
        assignments = np.ascontiguousarray(
            np.column_stack(
                [np.asarray(a.assignments, dtype=np.float64).reshape(-1) for a in self.atoms]
            )
        )
        assignment = _canonical_assignment(self.assignment, "assignment")
        v1_dict = _stagewise_to_manifold_sae_dict(
            basis_kinds=basis_kinds,
            decoder_blocks=decoder_blocks,
            atom_dims=atom_dims,
            basis_sizes=basis_sizes,
            n_harmonics=list(n_harmonics),
            coords=[c.copy() for c in coords],
            assignments=assignments,
            fitted=fitted,
            logits=np.ascontiguousarray(np.asarray(self.logits, dtype=np.float64)),
            training=training,
            assignment=assignment,
            reconstruction_r2=self.reconstruction_ev(),
            seed=seed,
            atom_topology=_topology_for_bases(basis_kinds),
            atom_topologies=_topologies_for_bases(basis_kinds),
        )
        core = rust_module().ManifoldSaeCore(v1_dict)
        # SAC retains the training matrix so `reconstruct(X=training)` returns the
        # composed in-sample reconstruction bit-for-bit (the core's to_dict still
        # nulls training_data, matching the legacy dataclass serialization).
        return ManifoldSAE(core, training_data=training)

    def reconstruct(
        self, X: Any = None, *, t_init: Any = None, a_init: Any = None
    ) -> np.ndarray:
        """Reconstruct ``X`` through the composed dictionary, ``(N, p)``.

        ``X=None`` (or the exact training target) returns the in-sample composed
        reconstruction ``Σ_k a_k · (Φ_k B_k)``. Any OTHER ``X`` is scored
        OUT OF SAMPLE: the frozen decoders route each held-out row through the
        existing Rust fixed-decoder OOS solve (via :meth:`to_manifold_sae`), so
        passing fresh rows no longer silently returns the training reconstruction.
        ``t_init`` / ``a_init`` warm-start the OOS refinement (#357).
        """
        if X is None:
            return self._in_sample_reconstruction()
        return self.to_manifold_sae().reconstruct(X, t_init=t_init, a_init=a_init)

    def transform(
        self, X: Any, *, t_init: Any = None, a_init: Any = None
    ) -> np.ndarray:
        """Out-of-sample composed reconstruction of ``X`` (honors ``X``).

        Thin alias for :meth:`reconstruct` that always routes through the Rust
        OOS path, giving the composed dictionary the held-out ``transform``
        surface the joint :class:`ManifoldSAE` already exposes.
        """
        return self.to_manifold_sae().reconstruct(X, t_init=t_init, a_init=a_init)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for :meth:`transform` (out-of-sample reconstruction of ``X``)."""
        return self.to_manifold_sae().reconstruct(X)

    def encode(
        self, X: Any, **kwargs: Any
    ) -> "np.ndarray | tuple[np.ndarray, dict[str, Any]]":
        """Out-of-sample per-token assignments ``a*`` ``(N, K)`` for ``X``.

        Delegates to :meth:`ManifoldSAE.encode` on the lifted dictionary, so the
        frozen-decoder encode runs through the same Rust OOS solve the joint model
        uses. Keyword arguments (``t_init`` / ``a_init`` / ``encoder`` /
        ``return_stats``) are forwarded unchanged.
        """
        return self.to_manifold_sae().encode(X, **kwargs)

    def reconstruction_ev(self) -> float:
        """Centered explained variance of the in-sample reconstruction.

        The coefficient of determination ``R² = 1 − SSR/SST`` (column-mean-centered
        SST, residual SSR) is a numeric kernel owned by the Rust core —
        ``sae_manifold_reconstruction_r2``, the same FFI
        :class:`gamfit.crosscoder` reports — so the SSR/SST reduction, the
        ``SST == 0 → NaN`` convention, and the non-finite guards live there rather
        than being re-derived in Python (SPEC thin-wrapper rule). Marshals the
        target and composed reconstruction as contiguous ``f64`` and forwards.
        """
        x = np.ascontiguousarray(np.asarray(self.training_data, dtype=np.float64))
        recon = np.ascontiguousarray(self._in_sample_reconstruction())
        return float(rust_module().sae_manifold_reconstruction_r2(x, recon))


def _basis_with_jet_for_atom(
    topology: str,
    coords: np.ndarray,
    basis_size: int,
    latent_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild ``(Φ, dΦ/dt, roughness Gram)`` for a K=1 atom via the Rust kernel.

    The stagewise FFI is a *precomputed-basis* entry point (like the low-level
    ``sae_manifold_fit``): it carries no Duchon centers, so only the analytic,
    centers-free bases can refresh ``Φ(t)`` as the driver moves coordinates. This
    helper dispatches those kinds to the single Rust ``basis_with_jet`` kernel —
    the same one the torch bridge and the shape-band reconstruct use — so no
    basis math is reimplemented in Python.

    Returns ``phi`` ``(N, M)``, ``jet`` ``(N, M, d)``, ``penalty`` ``(M, M)``.
    Raises ``NotImplementedError`` for a centers-bearing basis (Duchon / linear /
    euclidean), which this precomputed FFI path cannot re-evaluate.
    """
    canon = _canonical_topology(str(topology))
    t = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if t.ndim != 2:
        raise ValueError(f"stagewise atom coords must be 2D (N, d); got shape {t.shape}")
    if canon == "circle":
        # A periodic atom's basis width is M = 2H + 1; recover H from the trained
        # decoder width so the rebuilt Φ has exactly the atom's columns. A born
        # atom can degenerate to the DC-only width M = 1 (H = 0) — the SAC driver
        # emits these — so H is ``(M - 1) // 2`` with NO ``max(1, …)`` floor: a
        # spurious floor would rebuild a 3-column Φ against a 1-row decoder and the
        # ``Φ @ B`` reconstruct would raise a shape mismatch. H = 0 is not a Python
        # special case: the Rust ``basis_with_jet`` periodic kernel honours
        # ``n_harmonics = 0`` natively (M = 1 constant column, zero jet, DC-only
        # penalty — the same DC treatment every wider periodic atom gets), so all
        # widths route through the one Rust kernel and no basis math lives here.
        n_harmonics = max((int(basis_size) - 1) // 2, 0)
        phi, jet, penalty = rust_module().basis_with_jet(
            "periodic", t[:, :1], {"n_harmonics": int(n_harmonics)}
        )
    elif canon == "sphere":
        phi, jet, penalty = rust_module().basis_with_jet("sphere", t[:, : max(1, latent_dim)], {})
    else:
        raise NotImplementedError(
            f"sae_manifold_fit_stagewise supports only centers-free analytic atom "
            f"bases (circle / sphere) — the precomputed stagewise FFI carries no "
            f"basis centers; got topology={topology!r} (canonical {canon!r}). Use "
            f"the joint sae_manifold_fit for a Duchon/linear/euclidean dictionary."
        )
    phi = np.ascontiguousarray(np.asarray(phi, dtype=np.float64))
    jet = np.ascontiguousarray(np.asarray(jet, dtype=np.float64))
    penalty = np.ascontiguousarray(np.asarray(penalty, dtype=np.float64))
    return phi, jet, penalty


def sae_manifold_fit_stagewise(
    X: Any = None,
    *,
    d_atom: int = 1,
    atom_topology: str = "circle",
    assignment: str = "ibp_map",
    structured_whitening: bool | None = None,
    fisher_factors: Any = None,
    cone_atom_recovery: bool = False,
    rank_charge_evidence: bool = False,
    min_effect_ev: float = 0.0,
    max_births: int = 24,
    max_backfit_sweeps: int = 4,
    max_factor_rank: int = 4,
    sample_weights: Any = None,
    n_iter: int = 64,
    seed_n_iter: int | None = None,
    learning_rate: float | None = None,
    sparsity_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    isometry_weight: float = 1.0,
    ridge_ext_coord: float = 1.0e-6,
    ridge_beta: float = 1.0e-6,
    alpha: float | str | None = None,
    tau: float | None = None,
    random_state: int = 0,
    progress_callback: Any = None,
) -> StagewiseSAE:
    """Grow a curved SAE dictionary by Sequential Atom Composition (SAC).

    Thin wrapper over the Rust ``fit_stagewise`` driver. It builds the single K=1
    SEED with the proven :func:`sae_manifold_fit` (Rust-seeded + fit — so the seed
    coordinates/decoder come from the certified K=1 path, not a Python
    reimplementation), rebuilds that atom's basis via the Rust ``basis_with_jet``
    kernel, and hands the arrays to the ``sae_manifold_fit_stagewise`` FFI, which
    runs forward births + backfitting + terminal joint evidence entirely in Rust.

    Parameters
    ----------
    X
        Target matrix the dictionary reconstructs, ``(N, p)`` (1D reshaped to
        ``(N, 1)``). At the composed-tier call site this is the T1 residual.
    d_atom
        Intrinsic coordinate dimension of the seed atom (``1`` for a circle).
    atom_topology
        Seed atom topology. Only centers-free analytic bases are supported by the
        precomputed stagewise FFI: ``"circle"`` (periodic) and ``"sphere"``.
    assignment
        Assignment/gate family (``"ibp_map"`` / ``"softmax"`` / ``"threshold_gate"``
        and their aliases), resolved through the shared public validator.
    structured_whitening
        Install the Σ-whitened per-row metric on each birth so the K=1 candidate
        fits run under the structured residual covariance from atom one (Σ is
        refit per birth internally). ``None`` (default) resolves to ``True`` for
        the ordinary path, but to ``False`` when ``fisher_factors`` carry a
        likelihood-whitening ``"behavioral_fisher"`` provenance — that fixed
        harvest metric and the per-birth Σ-refit are rival sources for the same
        per-row inner product, so the GLS lane fits under the fixed metric alone.
        Pass an explicit ``True``/``False`` to override the resolution.
    fisher_factors
        Optional harvest-emitted output-Fisher factor stack (a ``HarvestShard`` /
        ``load_harvest_shard`` dict / raw ``(n, p, r)`` array), installed on the
        seed term and carried across every birth / backfit clone. With
        ``provenance="behavioral_fisher"`` this is the **Rung 1** metric: the
        reconstruction residual is priced as ``½ eᵀ G_n e`` (nats, generalized
        least squares) at every stage of the composition, not just the seed.
        ``None`` (default) is the isotropic ``½‖e‖²`` path, bit-for-bit today's.
    min_effect_ev
        Explicit MINIMUM-EFFECT (salience) floor a birth's ΔEV must clear ON TOP
        of the evidence gate. ``0.0`` (default) recovers evidence-only, null-
        recovering acceptance; a positive value suppresses true-but-trivial
        wiggles at frontier ``n``. A config dial, never a magic constant.
    max_births
        Safety cap on forward births atop the seed (a BOUND, not the stop rule —
        two consecutive rejections / an empty residual factor stop the phase).
    max_backfit_sweeps
        Maximum keep-best backfitting sweeps (each monotone at fixed ρ).
    max_factor_rank
        Residual-factor ladder cap per birth (how many candidate factor
        directions the evidence ladder scores when mining the residual).
    sample_weights
        Optional length-``N`` per-row stratified importance weights (√w),
        installed on every inner fit via the reconstruction-weight seam. ``None``
        is the unweighted path.
    n_iter
        Inner Newton iterations per birth / per sweep.
    seed_n_iter
        Inner iterations for the K=1 SEED fit. ``None`` reuses ``n_iter``.
    learning_rate
        Inner step size. ``None`` uses ``0.05`` for ``threshold_gate`` and ``1.0``
        otherwise (matching :func:`sae_manifold_fit`).
    sparsity_weight, smoothness_weight, isometry_weight
        Forwarded to the seed fit; ``sparsity_weight`` / ``smoothness_weight`` are
        also handed to the stagewise driver's inner fits. (``isometry_weight`` and
        the other analytic penalties gauge only the SEED; the compact FFI's inner
        fits carry no analytic-penalty registry — a known scoping limit.)
    ridge_ext_coord, ridge_beta
        Inner coordinate / β ridges for the stagewise fits.
    alpha, tau
        Assignment concentration / temperature. ``None`` resolves to the seed
        fit's values (K-aware IBP α; τ = 0.5).
    random_state
        Seed forwarded to the K=1 seed fit's initializer.
    progress_callback
        Optional callable invoked from the Rust stagewise driver with progress
        dictionaries. Durable events carry ``checkpoint_available=True`` and a
        compact ``checkpoint`` payload containing the current atoms, logits, and
        ρ values so the caller can persist per-birth checkpoints.

    Returns
    -------
    StagewiseSAE
        The composed dictionary (per-atom decoder / coords / gate / topology /
        latent_dim / ΔEV) plus the by-construction-monotone ``ev_trace`` and
        ``backfit_ev_trace``, ``births_accepted``, the full ``birth_records``
        ledger, and the (empty-by-construction) ``collapse_events`` log.
    """
    if X is None:
        raise TypeError("sae_manifold_fit_stagewise requires X input array")
    x = _as_2d_float(X, "X")
    n_obs, p_out = int(x.shape[0]), int(x.shape[1])
    if n_obs < 2:
        raise ValueError(f"sae_manifold_fit_stagewise requires n >= 2; got n={n_obs}")
    d0 = int(d_atom)
    if d0 < 1:
        raise ValueError(f"d_atom must be >= 1; got {d0}")
    if int(max_births) < 0 or int(max_backfit_sweeps) < 0 or int(max_factor_rank) < 1:
        raise ValueError(
            "max_births / max_backfit_sweeps must be >= 0 and max_factor_rank >= 1"
        )
    kind = _canonical_public_assignment(assignment)
    seed_iter = int(n_iter if seed_n_iter is None else seed_n_iter)
    effective_lr = (0.05 if kind == "threshold_gate" else 1.0) if learning_rate is None else float(learning_rate)

    weights_arr: np.ndarray | None
    if sample_weights is None:
        weights_arr = None
    else:
        weights_arr = np.ascontiguousarray(np.asarray(sample_weights, dtype=np.float64).reshape(-1))
        if weights_arr.shape[0] != n_obs:
            raise ValueError(
                "sample_weights must have one entry per observation; "
                f"got {weights_arr.shape[0]} for n={n_obs}"
            )
        if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr <= 0.0):
            raise ValueError("sample_weights must be finite and strictly positive")

    # ── Rung 1 (B4): normalize the optional harvest Fisher shard once. It rides
    # into BOTH the K=1 seed fit and the stagewise FFI so the SAME GLS metric
    # prices the seed and every born atom. ``_normalize_fisher_factors`` accepts a
    # HarvestShard / dict / raw (n, p, r) and returns (U, mass_residual, provenance).
    fisher_shard = _normalize_fisher_factors(fisher_factors, n_obs, p_out)
    if fisher_shard is None:
        fisher_u = None
        fisher_prov = None
        fisher_whitens = False
    else:
        fisher_u = np.ascontiguousarray(np.asarray(fisher_shard[0], dtype=np.float64))
        fisher_prov = str(fisher_shard[2])
        # Only ``behavioral_fisher`` whitens the likelihood; the gauge-only
        # output-Fisher provenances do not (they would merely gauge the seed and
        # be clobbered by the per-birth Σ-refit, so they are not a GLS lane here).
        fisher_whitens = fisher_prov == "behavioral_fisher"
    # Resolve the structured-whitening default against the shard: a fixed
    # likelihood-whitening metric and the per-birth Σ-refit are mutually exclusive.
    if structured_whitening is None:
        structured_whitening_eff = not fisher_whitens
    else:
        structured_whitening_eff = bool(structured_whitening)
    if structured_whitening_eff and fisher_whitens:
        raise ValueError(
            "sae_manifold_fit_stagewise: a likelihood-whitening 'behavioral_fisher' "
            "fisher metric conflicts with structured_whitening=True (the per-birth Σ-refit "
            "would clobber it); pass structured_whitening=False for the GLS lane"
        )

    # ── Seed: the proven Rust K=1 fit (Rust-seeded coords/decoder, no Python
    # reimplementation of the topology-specific seeding). ─────────────────────
    seed_fit = sae_manifold_fit(
        x,
        K=1,
        d_atom=d0,
        atom_topology=atom_topology,
        assignment=assignment,
        isometry_weight=isometry_weight,
        sparsity_weight=sparsity_weight,
        smoothness_weight=smoothness_weight,
        n_iter=seed_iter,
        random_state=int(random_state),
        alpha=(_ALPHA_UNSET if alpha is None else alpha),
        tau=tau,
        weights=weights_arr,
        fisher_factors=(None if fisher_shard is None else fisher_factors),
        _run_structure_search=False,
        _run_outer_rho_search=False,
    )
    seed_topology = seed_fit.atom_topologies[0]
    seed_kind = str(seed_fit._basis_kinds[0])
    # Use the seed atom's ACTUAL intrinsic dimension, not the requested d_atom: a
    # circle is intrinsically 1-D whatever the caller asked, and the rebuilt jet /
    # initial_coords must agree with atom_dim the FFI installs on the term.
    d_seed = int(seed_fit._atom_dims[0])
    coords0 = np.ascontiguousarray(seed_fit.coords[0].astype(np.float64))
    if coords0.ndim != 2 or coords0.shape[1] != d_seed:
        coords0 = coords0.reshape(n_obs, d_seed)
    decoder0 = np.ascontiguousarray(seed_fit.decoder_blocks[0].astype(np.float64))
    m0 = int(decoder0.shape[0])
    phi0, jet0, penalty0 = _basis_with_jet_for_atom(seed_topology, coords0, m0, d_seed)
    if jet0.shape != (n_obs, m0, d_seed):
        raise ValueError(
            f"stagewise seed jet shape {jet0.shape} disagrees with (N, M, d)="
            f"({n_obs}, {m0}, {d_seed}); the rebuilt Jacobian must match atom_dim"
        )
    if phi0.shape != (n_obs, m0):
        raise ValueError(
            f"stagewise seed basis width {phi0.shape[1]} disagrees with the seed "
            f"decoder rows {m0}; the rebuilt Φ must match the fitted decoder basis"
        )
    logits_seed = np.asarray(seed_fit.low_level_logits, dtype=np.float64)
    if logits_seed.ndim == 1:
        if logits_seed.size != n_obs:
            raise ValueError(
                "stagewise seed logits must have one row per observation; "
                f"got flat length {logits_seed.size} for n={n_obs}"
            )
        logits0 = logits_seed.reshape(n_obs, 1)
    elif logits_seed.ndim == 2 and logits_seed.shape == (n_obs, 1):
        logits0 = logits_seed
    else:
        raise ValueError(
            "stagewise seed logits must be (N,) or (N, 1); "
            f"got shape {logits_seed.shape} for n={n_obs}"
        )
    logits0 = np.ascontiguousarray(logits0)

    basis_values = phi0[None, :, :]                    # (1, N, M)
    basis_jacobian = jet0[None, :, :, :]               # (1, N, M, d)
    decoder_coefficients = decoder0[None, :, :]        # (1, M, p)
    smooth_penalties = penalty0[None, :, :]            # (1, M, M)
    initial_coords = coords0[None, :, :]               # (1, N, d)

    payload = rust_module().sae_manifold_fit_stagewise(
        np.ascontiguousarray(x),
        [seed_kind],
        [d_seed],
        np.ascontiguousarray(basis_values),
        np.ascontiguousarray(basis_jacobian),
        [m0],
        np.ascontiguousarray(decoder_coefficients),
        np.ascontiguousarray(smooth_penalties),
        np.ascontiguousarray(logits0),
        np.ascontiguousarray(initial_coords),
        float(seed_fit.alpha),
        float(seed_fit.tau),
        bool(seed_fit.learnable_alpha),
        str(kind),
        sparsity_strength=float(sparsity_weight),
        smoothness=float(smoothness_weight),
        max_iter=int(n_iter),
        learning_rate=float(effective_lr),
        ridge_ext_coord=float(ridge_ext_coord),
        ridge_beta=float(ridge_beta),
        max_births=int(max_births),
        max_backfit_sweeps=int(max_backfit_sweeps),
        min_effect_ev=float(min_effect_ev),
        max_factor_rank=int(max_factor_rank),
        structured_whitening=bool(structured_whitening_eff),
        cone_atom_recovery=bool(cone_atom_recovery),
        rank_charge_evidence=bool(rank_charge_evidence),
        row_loss_weights=weights_arr,
        progress_callback=progress_callback,
        fisher_factors=(
            None
            if fisher_shard is None
            else np.ascontiguousarray(fisher_u.reshape(n_obs, p_out, -1))
        ),
        fisher_provenance=(None if fisher_shard is None else fisher_prov),
    )
    return _stagewise_from_payload(dict(payload), x, seed_fit)


def _stagewise_from_payload(
    payload: Mapping[str, Any],
    x: np.ndarray,
    seed_fit: ManifoldSAE,
) -> StagewiseSAE:
    """Assemble a :class:`StagewiseSAE` from the compact stagewise FFI payload."""
    logits = np.asarray(payload["logits"], dtype=np.float64)
    birth_records = [
        {
            "kind": str(rec["kind"]),
            "delta_ev": float(rec["delta_ev"]),
            "factor_energy": float(rec["factor_energy"]),
            "joint_reml_before": float(rec["joint_reml_before"]),
            "joint_reml_after": float(rec["joint_reml_after"]),
            "accepted": bool(rec["accepted"]),
        }
        for rec in payload["birth_records"]
    ]
    # Map the ΔEV each ACCEPTED NEW-ATOM birth earned onto its atom (atom 0 is the
    # seed; chart extensions refit the previous atom and do not create a new one).
    new_atom_deltas = [
        rec["delta_ev"] for rec in birth_records if rec["accepted"] and rec["kind"] == "new_atom"
    ]
    atoms: list[StagewiseAtom] = []
    for atom_idx, atom in enumerate(payload["atoms"]):
        topology = _basis_to_topology(str(atom["basis_kind"]))
        coords = np.asarray(atom["on_atom_coords_t"], dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        delta_ev: float | None
        if atom_idx == 0:
            delta_ev = None
        elif atom_idx - 1 < len(new_atom_deltas):
            delta_ev = float(new_atom_deltas[atom_idx - 1])
        else:
            delta_ev = None
        atoms.append(
            StagewiseAtom(
                decoder=np.asarray(atom["decoder_B"], dtype=np.float64),
                coords=coords,
                assignments=np.asarray(atom["assignments_z"], dtype=np.float64).reshape(-1),
                topology=topology,
                latent_dim=int(atom["latent_dim"]),
                delta_ev=delta_ev,
                theta=None,
            )
        )
    ev_trace = np.asarray(payload["ev_trace"], dtype=np.float64)
    backfit_ev_trace = np.asarray(payload["backfit_ev_trace"], dtype=np.float64)
    # Live-decoder collapse log: a monotonicity violation among adopted candidates.
    # Empty BY CONSTRUCTION (atoms never share a Hessian, every adoption cleared
    # ΔEV >= 0) — an empty log IS the discriminator's zero-collapse verdict.
    collapse_events = [
        dict(rec, reason="ev_regression")
        for rec in birth_records
        if rec["accepted"] and rec["delta_ev"] < -1.0e-9
    ]
    log_ard = [np.asarray(a, dtype=np.float64) for a in payload["log_ard"]]
    return StagewiseSAE(
        atoms=atoms,
        logits=logits,
        ev_trace=ev_trace,
        backfit_ev_trace=backfit_ev_trace,
        births_accepted=int(payload["births_accepted"]),
        births_rejected=int(payload["births_rejected"]),
        stopped_reason=str(payload["stopped_reason"]),
        terminal_joint_reml=float(payload["terminal_joint_reml"]),
        terminal_data_fit=float(payload["terminal_data_fit"]),
        birth_records=birth_records,
        collapse_events=collapse_events,
        log_lambda_sparse=float(payload["log_lambda_sparse"]),
        log_lambda_smooth=np.asarray(payload["log_lambda_smooth"], dtype=np.float64),
        log_ard=log_ard,
        assignment=str(seed_fit.assignment),
        seed=seed_fit,
        training_data=np.asarray(x, dtype=np.float64),
        # #1939 — surface the FFI's cone_atom_recovery echo on the result object so a
        # harness can verify the flag engaged; older payloads without the key default
        # to False.
        cone_atom_recovery_used=bool(payload.get("cone_atom_recovery_used", False)),
        rank_charge_evidence_used=bool(payload.get("rank_charge_evidence_used", False)),
    )


def _require_sae_row_block_penalty(kind: str, kwarg: str) -> None:
    """Refuse a SAE row-block penalty the running extension does not advertise.

    The compiled extension reports the row-block penalty kinds it supports via
    ``build_info()["sae_row_block_penalties"]`` (kept in lockstep with the Rust
    ``sae_penalty_is_row_block_supported`` matcher). A stale binary that predates
    a given penalty either omits the key entirely or lists a subset; forwarding
    the descriptor anyway would surface as a cryptic internal Schur-Cholesky
    error. Detect the mismatch here and raise a clear ``NotImplementedError``
    naming the user-facing kwarg (issue #338).
    """
    supported = rust_module().build_info().get("sae_row_block_penalties", [])
    if kind not in supported:
        raise NotImplementedError(
            f"sae_manifold_fit: {kwarg} requires SAE row-block penalty "
            f"'{kind}', which the installed gam-pyffi extension does not "
            "advertise (it predates row-block support for this penalty). "
            f"Upgrade gamfit to a build that supports '{kind}', or pass "
            f"{kwarg}=0.0 to disable it."
        )


def _build_analytic_penalties_payload(
    *,
    isometry_weight: float,
    decoder_feature_sparsity_groups: list[list[int]] | None,
    block_orthogonality_weight: float,
    d_max: int,
    p_out: int,
    gate_sparsity: str = "l1",
    sparsity_weight: float = 0.0,
    scad_mcp_gamma: float = 3.7,
    nuclear_norm_weight: float = 0.0,
    nuclear_norm_max_rank: int | None = None,
    decoder_incoherence_weight: float = 1.0,
    k_atoms: int = 1,
) -> str | None:
    """Translate the SAE regularizer knobs into the analytic-penalty JSON
    payload consumed by ``sae_manifold_fit_minimal``.

    The SAE regularizer knobs route through ``crates/gam-sae``.
    ``isometry_weight`` and ``block_orthogonality_weight`` target the row-block
    driver ("t" latent block). ``ard_per_atom`` is NOT a registry descriptor:
    it routes to the native ``ArdAxisPrior`` via the ``native_ard_enabled`` FFI
    flag (see ``sae_manifold_fit``), since the registry ``ard`` penalty is
    intentionally skipped on every SAE path.
    ``gate_sparsity="scad"`` or ``"mcp"`` emits the row-block
    ``scad_mcp`` descriptor on the same "t" block, using ``sparsity_weight`` as
    its non-convex sparsity strength. The default ``"l1"`` emits no analytic
    descriptor and preserves the existing assignment-prior sparsity path.
    ``decoder_feature_sparsity_groups`` targets the decoder coefficient
    block ("beta" latent block) and group-lassoes ``p_out`` features in rows
    of the per-basis-function decoder matrix. For ``k_atoms >= 2`` the Rust
    ``add_sae_beta_penalty`` dispatches the group-lasso per atom, rebuilding
    the penalty target to each atom's ``(M_k, p_out)`` decoder block, so the
    concatenated ``flatten_beta`` layout with distinct ``M_k`` is handled
    natively (#240).

    ``nuclear_norm_weight`` also targets the decoder ("beta") block (#672): it
    emits a ``nuclear_norm`` descriptor that the Rust ``add_sae_beta_penalty``
    dispatches per atom, treating each atom's ``(M_k, p_out)`` decoder block as
    a matrix and shrinking its singular spectrum (embedding rank). ``n_eff`` is
    deliberately *not* emitted — Rust sets it per atom to ``M_k``.
    ``nuclear_norm_max_rank`` optionally caps the number of leading singular
    values penalized.
    """
    items: list[dict[str, Any]] = []
    # #240: `ard_per_atom` does NOT emit a registry `ard` descriptor. The SAE
    # objective deliberately skips `AnalyticPenaltyKind::Ard(_)` on every path
    # (gradient assembly AND value total) because the native `ArdAxisPrior` is
    # the single source of truth for the per-atom coordinate precision — a
    # registry `½λt²` ridge would double-count it and is period-discontinuous on
    # the circular bases. The flag is instead routed to `native_ard_enabled` at
    # the FFI call (see `sae_manifold_fit`), which sizes / drops each atom's
    # `log_ard` precisions. Emitting a descriptor here would be a guaranteed
    # no-op (the exact issue-#240 silent-no-op anti-pattern).
    if gate_sparsity in {"scad", "mcp"} and float(sparsity_weight) > 0.0:
        _require_sae_row_block_penalty("scad_mcp", "gate_sparsity")
        items.append({
            "kind": "scad_mcp",
            "target": "t",
            "variant": str(gate_sparsity),
            "gamma": float(scad_mcp_gamma),
            "weight": float(sparsity_weight),
        })
    if isometry_weight is not None and float(isometry_weight) > 0.0:
        _require_sae_row_block_penalty("isometry", "isometry_weight")
        items.append({
            "kind": "isometry",
            "target": "t",
            "weight": float(isometry_weight),
        })
    if (
        block_orthogonality_weight is not None
        and float(block_orthogonality_weight) > 0.0
    ):
        _require_sae_row_block_penalty(
            "block_orthogonality", "block_orthogonality_weight"
        )
        # The latent block "t" is (n_obs, d_max). BlockOrth requires ≥2
        # groups that partition contiguous axes from 0 — split into
        # singletons so each axis is in its own group, which is the most
        # restrictive (and most informative) gauge available without
        # caller-supplied structure.
        if int(d_max) < 2:
            raise ValueError(
                "block_orthogonality_weight requires d_atom >= 2; "
                f"got d_max={d_max}"
            )
        groups = [[axis] for axis in range(int(d_max))]
        items.append({
            "kind": "block_orthogonality",
            "target": "t",
            "groups": groups,
            "weight": float(block_orthogonality_weight),
        })
    if decoder_feature_sparsity_groups is not None:
        # Validate group payload eagerly so the error surfaces in Python
        # with the user-facing kwarg name rather than as a Rust descriptor
        # error referring to "feature_groups".
        groups = [list(int(f) for f in g) for g in decoder_feature_sparsity_groups]
        if not groups or any(len(g) == 0 for g in groups):
            raise ValueError(
                "decoder_feature_sparsity_groups must be a non-empty list of "
                "non-empty index lists; got "
                f"{decoder_feature_sparsity_groups!r}"
            )
        flat = [int(f) for g in groups for f in g]
        if any(f < 0 or f >= int(p_out) for f in flat):
            raise ValueError(
                "decoder_feature_sparsity_groups indices must be in "
                f"[0, p_out={int(p_out)}); got {decoder_feature_sparsity_groups!r}"
            )
        if len(set(flat)) != len(flat):
            raise ValueError(
                "decoder_feature_sparsity_groups must form a disjoint "
                f"partition of feature indices; got {decoder_feature_sparsity_groups!r}"
            )
        if sorted(flat) != list(range(int(p_out))):
            raise ValueError(
                "decoder_feature_sparsity_groups must cover every feature "
                f"index in [0, p_out={int(p_out)}); got {decoder_feature_sparsity_groups!r}"
            )
        items.append({
            "kind": "mechanism_sparsity",
            "target": "beta",
            "feature_groups": groups,
        })
    if nuclear_norm_weight is not None and float(nuclear_norm_weight) > 0.0:
        # Targets the decoder ("beta") block. The Rust dispatch rebuilds the
        # penalty per atom (n_eff = M_k, latent_dim = p_out), so we deliberately
        # do NOT emit n_eff here — the registry-held base value is overridden.
        item: dict[str, Any] = {
            "kind": "nuclear_norm",
            "target": "beta",
            "weight": float(nuclear_norm_weight),
        }
        if nuclear_norm_max_rank is not None:
            item["max_rank"] = int(nuclear_norm_max_rank)
        items.append(item)
    # Cross-atom decoder column-space incoherence (issue #671), ON by default,
    # for k_atoms >= 2 (penalizes co-activating atom *pairs*). block_sizes/p_out
    # are placeholders: the Rust `add_sae_beta_penalty` injects the real per-atom
    # M_k, p_out, target, and the empirical co-activation (mean_n gate_j*gate_k)
    # from the live SAE at fit time. We only signal the descriptor + weight.
    if (
        decoder_incoherence_weight is not None
        and float(decoder_incoherence_weight) > 0.0
        and int(k_atoms) >= 2
    ):
        items.append({
            "kind": "decoder_incoherence",
            "target": "beta",
            "block_sizes": [1] * int(k_atoms),
            "p_out": int(p_out),
            "weight": float(decoder_incoherence_weight),
        })
    if not items:
        return None
    return json.dumps(items)


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D or 2D numeric array")
    return np.ascontiguousarray(arr)


def _normalize_fisher_factors(
    fisher_factors: Any, n_obs: int, p_out: int
) -> tuple[np.ndarray, np.ndarray | None, str] | None:
    """Coerce a WP-D output-Fisher shard into the ``(U, mass_residual, provenance)``
    the Rust ``sae_manifold_fit_minimal`` FFI consumes (#980).

    ``fisher_factors`` may be: ``None`` (Euclidean, no shard); a
    :class:`gamfit.torch.harvest.HarvestShard` (``.U`` ``(n, p, r)`` /
    ``.mass_residual`` ``(n,)``); the dict returned by
    :func:`gamfit.torch.harvest.load_harvest_shard` (keys ``"U"`` /
    ``"mass_residual"``); or a raw ``(n, p, r)`` array (no diagnostic). The
    *presence* of a non-``None`` value activates ``RowMetric::OutputFisher`` —
    there is no flag (magic by default). The U layout ``U[n, i, k]`` is shipped
    verbatim as a contiguous ``(n, p, r)`` f64 array; the Rust boundary flattens
    it row-major to ``u[n, i * r + k]`` for ``RowMetric::output_fisher``.
    """
    if fisher_factors is None:
        return None
    # HarvestShard dataclass or load_harvest_shard() dict — both carry U +
    # mass_residual; a bare array carries only U. The provenance tag (#980)
    # rides along so the FFI installs the matching output-Fisher `RowMetric`;
    # a bare array or a pre-#980 shard defaults to the same-position metric.
    provenance = "output_fisher"
    if hasattr(fisher_factors, "U") and hasattr(fisher_factors, "mass_residual"):
        u_src: Any = fisher_factors.U
        mr_src: Any = fisher_factors.mass_residual
        provenance = str(getattr(fisher_factors, "provenance", "output_fisher"))
    elif isinstance(fisher_factors, Mapping):
        if "U" not in fisher_factors:
            raise ValueError(
                "fisher_factors mapping must contain a 'U' (n, p, r) array"
            )
        u_src = fisher_factors["U"]
        mr_src = fisher_factors.get("mass_residual")
        provenance = str(fisher_factors.get("provenance", "output_fisher"))
    else:
        u_src = fisher_factors
        mr_src = None
    if provenance not in (
        "output_fisher",
        "output_fisher_downstream",
        "behavioral_fisher",
    ):
        raise ValueError(
            "fisher_factors provenance must be 'output_fisher', "
            "'output_fisher_downstream', or 'behavioral_fisher'; "
            f"got {provenance!r}"
        )
    u = np.asarray(u_src, dtype=np.float64)
    if u.ndim != 3:
        raise ValueError(
            f"fisher_factors U must be (n, p, r); got shape {u.shape}"
        )
    if u.shape[0] != n_obs or u.shape[1] != p_out:
        raise ValueError(
            f"fisher_factors U must be (n, p, r) = ({n_obs}, {p_out}, r); "
            f"got leading dims {u.shape[:2]}"
        )
    rank = int(u.shape[2])
    if rank < 1:
        raise ValueError("fisher_factors U rank (last axis) must be >= 1")
    if rank > p_out:
        raise ValueError(
            f"fisher_factors U rank {rank} exceeds output dim p={p_out}"
        )
    if not np.all(np.isfinite(u)):
        raise ValueError("fisher_factors U must be finite")
    u = np.ascontiguousarray(u)
    if mr_src is None:
        return u, None, provenance
    mr = np.asarray(mr_src, dtype=np.float64)
    if mr.shape != (n_obs,):
        raise ValueError(
            f"fisher_factors mass_residual must be (n,) = ({n_obs},); "
            f"got shape {mr.shape}"
        )
    if not np.all(np.isfinite(mr)):
        raise ValueError("fisher_factors mass_residual must be finite")
    return u, np.ascontiguousarray(mr), provenance


def _dims(k_atoms: int, d_atom: Any) -> list[int]:
    if d_atom is None or d_atom == "auto":
        return [2] * k_atoms
    if isinstance(d_atom, int):
        return [int(d_atom)] * k_atoms
    # J: a bare string would otherwise fall through to ``[int(d) for d in d_atom]``
    # and silently iterate per character (``"12"`` -> ``[1, 2]``). Only the
    # literal ``"auto"`` is meaningful; reject every other string explicitly.
    if isinstance(d_atom, str):
        raise ValueError(
            f"d_atom string must be 'auto'; got {d_atom!r}. Pass an int or a "
            "per-atom list of ints."
        )
    out = [int(d) for d in d_atom]
    if len(out) != k_atoms or min(out, default=0) < 0:
        raise ValueError("d_atom must provide one non-negative dimension per atom")
    return out


def _canon_name(name: Any) -> str:
    """Case-insensitive, ``-``/``_``-interchangeable name normalizer (O).

    String matching for topology / basis names is documented as
    case-insensitive and treating ``-`` and ``_`` interchangeably.
    """
    return str(name).strip().lower().replace("-", "_")


# Documented topology / basis ALIAS -> canonical basis kind. Mirrors the alias
# set in docs/manifold-sae.md (and the names the Rust ``SaeAtomBasisKind``
# accepts). Centralizing all aliases here keeps the resolution single-source.
_TOPOLOGY_TO_BASIS = {
    "circle": "periodic", "periodic": "periodic", "periodic_spline": "periodic",
    "sphere": "sphere", "torus": "torus",
    "linear": "linear", "linear_rank1": "linear", "affine": "linear",
    # BSF block AS a manifold-SAE atom (γ_g(t)=t·D_g, orthonormal frame + block
    # gating): kept as its OWN basis-kind string so the "linear_block" label
    # survives round-trip; the Rust `sae_atom_basis_kind_from_str` maps it to the
    # `Linear` kind for construction/evidence (see the FFI doc-comment). This is
    # the honest encoding of "BSF ⊂ ManifoldSAE" as config, not a new atom type.
    "linear_block": "linear_block", "flat_block": "linear_block",
    "euclidean": "euclidean", "euclidean_patch": "euclidean",
    "euclidean_quadratic_patch": "euclidean",
    "duchon": "duchon",
    "poincare": "poincare", "hyperbolic": "poincare", "poincare_patch": "poincare",
    "cylinder": "cylinder",
    # Per-atom evidence-raced topology discovery at fit entry (#2238): the
    # Rust driver rewrites each "auto" atom to the concrete race winner.
    "auto": "auto",
}
# Canonical / aliased basis kind -> canonical topology label.
_BASIS_TO_TOPOLOGY = {
    "periodic": "circle", "periodic_spline": "circle", "circle": "circle",
    "sphere": "sphere", "torus": "torus",
    "linear": "linear", "linear_rank1": "linear", "affine": "linear",
    # linear_block reports its own topology label (it is a flat block, not a plain
    # linear atom); the distinction is the orthonormal decoder frame + block gating.
    "linear_block": "linear_block", "flat_block": "linear_block",
    "duchon": "euclidean", "euclidean": "euclidean", "euclidean_patch": "euclidean",
    "euclidean_quadratic_patch": "euclidean",
    "poincare": "poincare", "hyperbolic": "poincare", "poincare_patch": "poincare",
    "cylinder": "cylinder",
    "auto": "auto",
}


def _basis_to_topology(basis: str) -> str:
    """Canonical topology label for a (possibly aliased) basis kind."""
    return _BASIS_TO_TOPOLOGY.get(_canon_name(basis), str(basis))


def _canonical_topology(name: str) -> str:
    """Canonical topology label for a (possibly aliased) topology/basis string.

    Resolves through the alias map to the basis kind, then to the canonical
    topology, so e.g. ``"periodic"`` and ``"circle"`` both canonicalize to
    ``"circle"``. Unknown (precomputed / caller-supplied) names pass through.
    """
    canon = _canon_name(name)
    basis = _TOPOLOGY_TO_BASIS.get(canon, canon)
    return _basis_to_topology(basis)


#: The two block-gating modes for ``atom_topology="linear_block"`` (BSF-as-atom).
#: ``norm_selection`` mirrors the BSF paper exactly — a block fires by its group ℓ2
#: coordinate norm (amplitude-driven selection), mapped to the ``ibp_map``
#: assignment. ``separate_gate`` is our reading — presence is a SEPARATE gate from
#: amplitude — mapped to the ``threshold_gate`` (hard-sigmoid) assignment. Both are
#: existing manifold-SAE assignment modes, so a linear_block atom races vs a curved
#: atom under one framework with no new solver path. No coordinate shrinkage is
#: applied beyond ARD (the flat block keeps its signed coordinates).
_FLAT_BLOCK_GATING_TO_ASSIGNMENT = {
    "norm_selection": "ibp_map",
    "norm": "ibp_map",
    "separate_gate": "threshold_gate",
    "separate": "threshold_gate",
}


def flat_block_assignment(gating: str) -> str:
    """Resolve a ``linear_block`` block-gating mode to a manifold-SAE assignment.

    Exposes the two gating modes of a BSF-block-as-manifold-atom
    (``atom_topology="linear_block"``): ``"norm_selection"`` (the paper's group-ℓ2
    block-TopK, → ``"ibp_map"``) and ``"separate_gate"`` (presence gate separate
    from amplitude, → ``"threshold_gate"``). Use it as
    ``sae_manifold_fit(..., atom_topology="linear_block",
    assignment=flat_block_assignment("norm_selection"))`` so the flat block and a
    curved atom race under ONE fit. Raises on an unknown mode.
    """
    key = _canon_name(gating)
    if key not in _FLAT_BLOCK_GATING_TO_ASSIGNMENT:
        raise ValueError(
            f"flat_block gating must be one of {sorted(set(_FLAT_BLOCK_GATING_TO_ASSIGNMENT))}; "
            f"got {gating!r}"
        )
    return _FLAT_BLOCK_GATING_TO_ASSIGNMENT[key]


def _stagewise_to_manifold_sae_dict(
    *,
    basis_kinds: list[str],
    decoder_blocks: list[np.ndarray],
    atom_dims: list[int],
    basis_sizes: list[int],
    n_harmonics: list[int],
    coords: list[np.ndarray],
    assignments: np.ndarray,
    fitted: np.ndarray,
    logits: np.ndarray,
    training: np.ndarray,
    assignment: str,
    reconstruction_r2: float,
    seed: Any,
    atom_topology: str,
    atom_topologies: list[str],
) -> dict[str, Any]:
    """Assemble a v1 ``to_dict``-schema payload for ``StagewiseSAE.to_manifold_sae``
    (#2091). The SAC lift no longer constructs ``ManifoldSAE(...)`` with dataclass
    kwargs; it builds this dict and loads it through the Rust core, so the core owns
    the state exactly like every other route. Per-atom entries carry the minimal SAC
    fields (no shape bands / covariance — SAC lifts a frozen dictionary). The
    centering-mean reduction is computed in Rust (the shared ``column_mean`` core,
    identical to the fit builder); this is pure marshaling of the result."""
    atoms_payload: list[dict[str, Any]] = []
    for k, block in enumerate(decoder_blocks):
        atoms_payload.append(
            {
                "basis": basis_kinds[k],
                "decoder_coefficients": np.asarray(block, dtype=float).tolist(),
                "assignments": np.asarray(assignments[:, k], dtype=float).tolist(),
                "coords": np.asarray(coords[k], dtype=float).tolist(),
                "coords_u_arc": None,
                "evidence": None,
                "active_dim": int(atom_dims[k]),
                "decoder_covariance_channel_factors": None,
                "shape_band_coords": None,
                "shape_band_mean": None,
                "shape_band_sd": None,
                "functional_evidence": None,
            }
        )
    return {
        "schema": "gamfit.ManifoldSAE/v1",
        "atom_topology": atom_topology,
        "atom_topologies": list(atom_topologies),
        "assignment": assignment,
        "assignment_label": str(seed.assignment) if hasattr(seed, "assignment") else assignment,
        "alpha": float(seed.alpha),
        "learnable_alpha": bool(seed.learnable_alpha),
        "tau": float(seed.tau),
        "sparsity_strength": float(seed.sparsity_strength),
        "smoothness": float(seed.smoothness),
        "learning_rate": float(seed.learning_rate),
        "max_iter": int(seed.max_iter),
        "random_state": int(seed.random_state),
        "top_k": None,
        "top_k_projection": None,
        "pre_topk": None,
        "jumprelu_threshold": float(seed.jumprelu_threshold),
        "oos_projection_top1": False,
        "dispersion": 1.0,
        "solver_plan": None,
        "primitive_names": ["sae_manifold_fit_stagewise"],
        "basis_specs": list(basis_kinds),
        "penalized_loss_score": None,
        "reml_score": None,
        "reconstruction_r2": float(reconstruction_r2),
        "training_mean": rust_module()
        .sae_manifold_training_mean(np.ascontiguousarray(np.asarray(training, dtype=float)))
        .tolist(),
        "training_data": None,
        "training_data_retained": False,
        "fitted": np.asarray(fitted, dtype=float).tolist(),
        "assignments": np.asarray(assignments, dtype=float).tolist(),
        "logits": np.asarray(logits, dtype=float).tolist(),
        "diagnostics": {"atom_trust": [], "atoms": []},
        "coords": [np.asarray(c, dtype=float).tolist() for c in coords],
        "decoder_blocks": [np.asarray(b, dtype=float).tolist() for b in decoder_blocks],
        "atoms": atoms_payload,
        "basis_kinds": list(basis_kinds),
        "atom_dims": [int(d) for d in atom_dims],
        "basis_sizes": [int(s) for s in basis_sizes],
        "n_harmonics": [int(h) for h in n_harmonics],
        "duchon_centers": [None] * len(decoder_blocks),
        "atom_two_lens": None,
        "residual_gauge": None,
        "incoherence_report": None,
        "curvature_report": None,
        "coordinate_fidelity": None,
        "topology_persistence": None,
        "atom_inference": None,
        "certificates": None,
        "structure_certificate": None,
        "cotrain": None,
        "hybrid_split": None,
        "fisher_factors": None,
        "fisher_provenance": None,
        "metric_provenance": "Euclidean",
        "fisher_mass_residual": None,
        "selected_log_lambda_sparse": None,
        "selected_log_lambda_smooth": None,
        "selected_log_ard": None,
    }


def _bases(k_atoms: int, atom_basis: Any, atom_topology: str) -> list[str]:
    if atom_basis is None:
        atom_basis = _TOPOLOGY_TO_BASIS.get(_canon_name(atom_topology), atom_topology)
    raw = [atom_basis] * k_atoms if isinstance(atom_basis, str) else list(atom_basis)
    if len(raw) != k_atoms:
        raise ValueError("atom_basis must provide one basis per atom")
    return [str(v) for v in raw]


def _topologies_for_bases(bases: list[str]) -> list[str]:
    """Per-atom topology labels for a resolved bases list (``basis_specs`` order)."""
    return [_basis_to_topology(b) for b in bases]


def _topology_for_bases(bases: list[str]) -> str:
    """Collapse a resolved bases list to a single topology label for metadata.

    When all atoms share one topology that common label is returned; when the
    atoms span more than one topology the honest scalar is ``"mixed"`` and the
    per-atom truth is exposed via ``atom_topologies`` (``basis_specs`` remains
    the per-atom source of truth)."""
    per_atom = _topologies_for_bases(bases)
    first = per_atom[0]
    return first if all(t == first for t in per_atom) else "mixed"


def _schedule_payload(schedule: Any) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, GumbelTemperatureSchedule):
        return schedule.to_rust_descriptor()
    descriptor = dict(schedule)
    decay = str(descriptor.get("decay", "geometric")).lower().replace("-", "_")
    if "tau_start" not in descriptor:
        raise ValueError("GumbelTemperatureSchedule (dict form): missing 'tau_start'")
    if "tau_min" not in descriptor:
        raise ValueError("GumbelTemperatureSchedule (dict form): missing 'tau_min'")
    tau_start = float(descriptor["tau_start"])
    tau_min = float(descriptor["tau_min"])
    rate = descriptor.get("rate")
    steps = descriptor.get("steps")
    iter_count = int(descriptor.get("iter_count", 0))
    _validate_gumbel_schedule_fields(
        tau_start=tau_start, tau_min=tau_min, decay=decay,
        rate=None if rate is None else float(rate),
        steps=None if steps is None else int(steps),
        iter_count=iter_count,
    )
    descriptor["decay"] = decay
    descriptor["tau_min"] = tau_min
    descriptor["tau_start"] = tau_start
    descriptor["iter_count"] = iter_count
    return descriptor


def _schedule_tau_start(schedule: Any, default: float) -> float:
    payload = _schedule_payload(schedule)
    return default if payload is None else float(payload["tau_start"])


def _default_research_k(n_obs: int) -> int:
    """Choose a conservative atom count for ``fit(activations)``."""
    return max(1, min(int(n_obs) - 1, 8, max(2, int(np.sqrt(max(1, int(n_obs)))))))


def fit(activations: Any, config: Mapping[str, Any] | None = None) -> ManifoldSAE:
    """Fit the recommended SAE-manifold research objective to activations.

    Parameters
    ----------
    activations
        Finite activation matrix ``(N, p)``. A vector is reshaped to ``(N, 1)``.
    config
        Optional keyword overrides forwarded to :func:`sae_manifold_fit`.

    Returns
    -------
    ManifoldSAE
        The fitted model handle. Its atoms, coordinates, assignments, summary,
        and trust diagnostics are available as attributes or methods. Infer
        coordinates for new activations with ``model.featurize(X)``; every
        operation is scoped to this returned model.
    """
    x = _as_2d_float(activations, "activations")
    cfg = {} if config is None else dict(config)
    if "K" not in cfg:
        cfg["K"] = _default_research_k(x.shape[0])
    return sae_manifold_fit(x, **cfg)


def plot(atom: Any, **kwargs: Any) -> Any:
    """Plot SAE atoms by delegating to ``gamfit._sae_viz``."""
    from . import _sae_viz

    return _sae_viz.plot(atom, **kwargs)


__all__ = ["GumbelTemperatureSchedule", "ManifoldSAE", "SaeManifoldAtomFit", "SaeManifoldFitResult",
           "gumbel_geometric_schedule", "gumbel_linear_schedule", "gumbel_reciprocal_iter_schedule",
           "fit", "plot", "sae_manifold_fit"]
