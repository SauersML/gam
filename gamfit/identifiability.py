"""High-level identifiable-factor recipe.

Composes an iVAE-style auxiliary-conditioned prior (Khemakhem 2107.10098) on
a *supervised* latent block with a mechanism-sparsity prior (Lachapelle
2401.04890) on a *free* latent block. Both factors live on a single shared
encoder ``E(X) -> (T_sup, T_free)`` and a single linear decoder.

Under the joint preconditions of the two papers — auxiliary covariate varies
across observations, the decoder is injective on the free block (its Jacobian
columns span an ``n_free``-rank subspace), the mechanism-sparsity penalty is
active, and the encoder is non-trivial — the free block ``T_free`` is
identified up to permutation and signed scaling of its components.

The runner is:

>>> result = gamfit.identifiable_factor_fit(
...     X, aux=labels, n_supervised=3, n_free=3,
...     mech_sparsity_weight="auto", aux_prior_weight="auto",
...     encoder="mlp[256, 256]",
... )
>>> result.T_supervised.shape
(N, 3)
>>> result.T_free.shape
(N, 3)
>>> result.evidence  # higher = better (Laplace-style log marginal-likelihood proxy)

If any precondition of the theorem fails, the corresponding warning is added
to ``result.warnings`` and emitted via :mod:`warnings.warn` as
``UserWarning``. The fit always completes — the warnings are informational
about which guarantee no longer formally holds.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._binding import rust_module

__all__ = [
    "IdentifiabilityReport",
    "IdentifiabilityTheoremResult",
    "IdentifiableFactorFitResult",
    "check",
    "identifiable_factor_fit",
]


# ---------------------------------------------------------------------------
# Per-theorem report dataclasses (Principle (f): identifiability theorems as
# runnable diagnostics — every guarantee a recipe claims is checked at
# fit-time and reported in the result).
# ---------------------------------------------------------------------------


# Numerical thresholds. Each one is tied to a paper citation; the rationale
# is documented inline so reviewers can re-derive the choice. None of these
# are exposed as CLI flags — they are constructor-overridable defaults only.
_IVAE_AUX_VAR_FLOOR = 1.0e-9
# Khemakhem 2107.10098 Thm. 1 requires aux to take at least ``2k + 1``
# distinct values for k-dim latent — we use std > floor as the operational
# proxy because the parametric prior family used here is Gaussian (T(z) = z),
# i.e. ``k = 1`` per aux column, so the rank-1 column-variation requirement
# reduces to "this column is not constant".
_IVAE_AUX_RANK_RTOL = 1.0e-8
# Khemakhem 2107.10098 §3 requires the auxiliary statistic to be of rank
# ``>= n_supervised`` (the "sufficient parametric variation" assumption).
# For Gaussian iVAE with the canonical sufficient statistic, "rank" of the
# aux design matrix collapses to the column rank of ``aux`` itself.
_IVAE_MIN_ENCODER_LAYERS = 2
# Khemakhem 2107.10098 §3: encoder must be "non-trivially nonlinear" — a
# bare Linear (1 affine layer) does not satisfy the universal-approximation
# argument used to push identifiability through the encoder.

_MECH_SPARSITY_FRACTION = 0.50
# Lachapelle 2401.04890 §2.4: mechanism sparsity identifies a latent up to
# permutation+sign **only after the L1 prox has reached equilibrium**. The
# paper's empirical analyses define "sparse enough" as >50% near-zero
# entries in the dependency matrix; we adopt that threshold.
_MECH_SPARSITY_ZERO_TOL = 1.0e-3
# A decoder column is "zero" when its magnitude is below this fraction of
# the column-max — relative thresholding mirrors the original paper.

_RANDPROJ_ACTIVATION_VAR_CEILING = 1.0e6
# Random projection identifiability (Hyvärinen & Pajunen 1999, restated for
# nonlinear ICA in Khemakhem §A.3): the encoder must not "explode" — its
# activation variance has to stay bounded so the change-of-variables term
# in the log-density is finite. We treat per-column variances above this
# ceiling as a hard fail and variances above 1e3 as a warn.
_RANDPROJ_ACTIVATION_VAR_WARN = 1.0e3


@dataclass(slots=True)
class IdentifiabilityTheoremResult:
    """Outcome of a single identifiability-theorem precondition check.

    Attributes
    ----------
    theorem_name : str
        Stable identifier — e.g. ``"iVAE"``, ``"MechanismSparsity"``,
        ``"RandomProjection"``. Use this to switch on the outcome from
        downstream code.
    status : {"pass", "warn", "fail"}
        ``pass`` = all preconditions met within numerical tolerance.
        ``warn`` = a precondition is degraded but the theorem may still
        hold in a weaker form (e.g. encoder shallower than the paper's
        canonical depth but still nonlinear). ``fail`` = a precondition
        is provably violated and the identifiability guarantee no longer
        applies.
    reason : str
        Human-readable rationale, including the paper citation and the
        observed numerical evidence.
    metric : dict[str, float]
        Numerical evidence used by the check. Keys are check-specific
        (e.g. ``"aux_min_std"``, ``"decoder_zero_fraction"``,
        ``"activation_var_max"``); values are floats. Callers can use
        these to build their own dashboards.
    """

    theorem_name: str
    status: str
    reason: str
    metric: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary view of this result."""

        return {
            "theorem_name": self.theorem_name,
            "status": self.status,
            "reason": self.reason,
            "metric": dict(self.metric),
        }


@dataclass(slots=True)
class IdentifiabilityReport:
    """Collection of per-theorem checks produced by :func:`check`.

    The report's overall status is the worst of its theorem statuses
    (``fail`` > ``warn`` > ``pass``). Iterate ``report.theorems`` for
    per-theorem detail or call :meth:`as_warnings` to format the warn/fail
    entries as plain strings (suitable for ``result.warnings``).
    """

    theorems: list[IdentifiabilityTheoremResult]

    @property
    def status(self) -> str:
        """Worst status among the contained theorem results."""

        order = {"pass": 0, "warn": 1, "fail": 2}
        worst = max(
            (order.get(t.status, 2) for t in self.theorems), default=0
        )
        for label, level in order.items():
            if level == worst:
                return label
        return "pass"

    def as_warnings(self) -> list[str]:
        """Return one ``"[theorem][status] reason"`` line per non-pass check."""

        return [
            f"[{t.theorem_name}][{t.status}] {t.reason}"
            for t in self.theorems
            if t.status != "pass"
        ]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary view of the entire report."""

        return {
            "status": self.status,
            "theorems": [t.as_dict() for t in self.theorems],
        }


def _check_ivae(
    *,
    aux: np.ndarray | None,
    n_supervised: int | None,
    encoder_depth: int | None,
    aux_var_floor: float = _IVAE_AUX_VAR_FLOOR,
    aux_rank_rtol: float = _IVAE_AUX_RANK_RTOL,
    min_encoder_layers: int = _IVAE_MIN_ENCODER_LAYERS,
) -> IdentifiabilityTheoremResult:
    """Khemakhem 2107.10098 Theorem 1 preconditions.

    Pass requires: aux varies across rows, aux column-rank equals the
    declared ``n_supervised`` (parametric richness), and the encoder has
    >= ``min_encoder_layers`` affine layers (non-trivial nonlinearity).

    Warn vs fail: a shallow-but-nonlinear encoder (1 < depth <
    min_encoder_layers) downgrades to ``warn`` because the theorem's
    universal-approximation argument is weakened but the encoder is still
    expressive enough to be useful empirically. A bare-linear encoder, a
    constant aux column, or a rank-deficient aux design matrix all fail
    outright — the theorem's hypothesis is provably violated.
    """

    metric: dict[str, float] = {}
    issues: list[str] = []
    status = "pass"

    if aux is None or n_supervised is None:
        return IdentifiabilityTheoremResult(
            theorem_name="iVAE",
            status="warn",
            reason=(
                "iVAE check skipped: this fit object does not carry the "
                "auxiliary covariate ``aux`` or its declared dimensionality, "
                "so Khemakhem 2107.10098 Thm. 1 preconditions cannot be "
                "verified."
            ),
            metric=metric,
        )

    aux_std = np.std(aux, axis=0)
    metric["aux_min_std"] = float(np.min(aux_std)) if aux_std.size else 0.0
    if aux_std.size == 0 or not np.all(aux_std > aux_var_floor):
        zero_axes = (
            np.where(aux_std <= aux_var_floor)[0].tolist()
            if aux_std.size
            else [0]
        )
        issues.append(
            f"aux axes {zero_axes} are constant across observations "
            f"(min std {metric['aux_min_std']:.3e} <= {aux_var_floor:.0e}), "
            f"so Khemakhem 2107.10098 Thm. 1 conditioning rank is zero."
        )
        status = "fail"

    rank = int(np.linalg.matrix_rank(aux, tol=aux_rank_rtol))
    metric["aux_column_rank"] = float(rank)
    metric["n_supervised"] = float(int(n_supervised))
    if rank < int(n_supervised):
        issues.append(
            f"aux column rank {rank} < n_supervised={int(n_supervised)}: "
            f"the parametric-richness assumption of Khemakhem 2107.10098 §3 "
            f"(rank of the sufficient-statistic design >= latent dim) fails."
        )
        status = "fail"

    if encoder_depth is None:
        issues.append(
            "encoder depth unknown — cannot verify the >=2-layer requirement "
            "of Khemakhem 2107.10098 §3."
        )
        if status == "pass":
            status = "warn"
    else:
        metric["encoder_depth"] = float(int(encoder_depth))
        if int(encoder_depth) < 1:
            issues.append(
                f"encoder depth {int(encoder_depth)} < 1; no encoder is "
                f"present. Khemakhem 2107.10098 §3 requires a smooth "
                f"non-trivial encoder."
            )
            status = "fail"
        elif int(encoder_depth) == 1:
            issues.append(
                f"encoder depth {int(encoder_depth)} == 1 (bare linear); "
                f"Khemakhem 2107.10098 §3 requires a non-linear encoder "
                f"for non-Gaussian sources. Identifiability is voided."
            )
            status = "fail"
        elif int(encoder_depth) < int(min_encoder_layers):
            issues.append(
                f"encoder depth {int(encoder_depth)} < canonical "
                f"min={int(min_encoder_layers)} of Khemakhem 2107.10098 §3; "
                f"the universal-approximation argument is weakened but the "
                f"encoder is still nonlinear."
            )
            if status == "pass":
                status = "warn"

    reason = (
        "all Khemakhem 2107.10098 Thm. 1 preconditions hold"
        if status == "pass"
        else " | ".join(issues)
    )
    return IdentifiabilityTheoremResult(
        theorem_name="iVAE", status=status, reason=reason, metric=metric,
    )


def _check_mechanism_sparsity(
    *,
    decoder: np.ndarray | None,
    n_supervised: int | None,
    n_free: int | None,
    state_dim: int | None,
    ground_truth_dim: int | None,
    mech_sparsity_weight: float | None,
    zero_tol: float = _MECH_SPARSITY_ZERO_TOL,
    min_zero_fraction: float = _MECH_SPARSITY_FRACTION,
) -> IdentifiabilityTheoremResult:
    """Lachapelle 2401.04890 Theorem preconditions.

    Pass requires: decoder Jacobian over the free-latent columns is sparse
    (>= ``min_zero_fraction`` near-zero entries) AFTER L1 reaches
    equilibrium, the sparsity weight was strictly positive, and the model
    state dimension is at least the ground-truth dim.

    Warn vs fail: a sparsity fraction below the paper's empirical threshold
    is a ``warn`` (the theorem still applies in a weaker form — fewer
    spurious paths than dense, but not the maximal identifiable subset).
    A zero sparsity weight, a rank-deficient free-Jacobian, or
    ``state_dim < ground_truth_dim`` all ``fail``.
    """

    metric: dict[str, float] = {}
    issues: list[str] = []
    status = "pass"

    if decoder is None or n_supervised is None or n_free is None:
        return IdentifiabilityTheoremResult(
            theorem_name="MechanismSparsity",
            status="warn",
            reason=(
                "MechanismSparsity check skipped: fit object does not carry "
                "decoder weights and latent split, so Lachapelle 2401.04890 "
                "preconditions cannot be verified."
            ),
            metric=metric,
        )

    free_cols = decoder[:, int(n_supervised) : int(n_supervised) + int(n_free)]
    metric["free_block_shape_rows"] = float(free_cols.shape[0])
    metric["free_block_shape_cols"] = float(free_cols.shape[1])

    col_max = (
        np.max(np.abs(free_cols), axis=0)
        if free_cols.size
        else np.zeros(int(n_free))
    )
    safe_max = np.where(col_max > 0.0, col_max, 1.0)
    relative = np.abs(free_cols) / safe_max[None, :]
    zero_mask = relative <= zero_tol
    zero_fraction = (
        float(zero_mask.mean()) if zero_mask.size else 0.0
    )
    metric["decoder_zero_fraction"] = zero_fraction

    rank = int(np.linalg.matrix_rank(free_cols, tol=1e-8))
    metric["decoder_free_rank"] = float(rank)
    if rank < int(n_free):
        issues.append(
            f"decoder Jacobian on the free block has rank {rank} < "
            f"n_free={int(n_free)}; Lachapelle 2401.04890 Thm. requires a "
            f"full-rank decoder Jacobian on the free latents."
        )
        status = "fail"

    if mech_sparsity_weight is None:
        issues.append(
            "mechanism-sparsity weight unknown — cannot confirm the L1 prox "
            "was active during fitting."
        )
        if status == "pass":
            status = "warn"
    else:
        metric["mech_sparsity_weight"] = float(mech_sparsity_weight)
        if not (float(mech_sparsity_weight) > 0.0):
            issues.append(
                f"mechanism-sparsity weight = {float(mech_sparsity_weight)} "
                f"is not strictly positive; the L1 prox contributed nothing "
                f"and Lachapelle 2401.04890 identification is void."
            )
            status = "fail"

    if zero_fraction < float(min_zero_fraction):
        issues.append(
            f"decoder zero-fraction {zero_fraction:.3f} < "
            f"{float(min_zero_fraction):.2f} threshold from "
            f"Lachapelle 2401.04890 §2.4: the L1 prox has not reached "
            f"equilibrium, identification is weakened."
        )
        if status == "pass":
            status = "warn"

    if ground_truth_dim is not None and state_dim is not None:
        metric["state_dim"] = float(int(state_dim))
        metric["ground_truth_dim"] = float(int(ground_truth_dim))
        if int(state_dim) < int(ground_truth_dim):
            issues.append(
                f"state_dim={int(state_dim)} < "
                f"ground_truth_dim={int(ground_truth_dim)}: Lachapelle "
                f"2401.04890 requires the model to have at least as many "
                f"latents as the data-generating process."
            )
            status = "fail"

    reason = (
        "all Lachapelle 2401.04890 preconditions hold"
        if status == "pass"
        else " | ".join(issues)
    )
    return IdentifiabilityTheoremResult(
        theorem_name="MechanismSparsity",
        status=status,
        reason=reason,
        metric=metric,
    )


def _check_random_projection(
    *,
    activations: np.ndarray | None,
    var_warn: float = _RANDPROJ_ACTIVATION_VAR_WARN,
    var_ceiling: float = _RANDPROJ_ACTIVATION_VAR_CEILING,
) -> IdentifiabilityTheoremResult:
    """Random-projection identifiability precondition.

    Pass requires: every latent column has finite variance below
    ``var_warn``. Khemakhem 2107.10098 Appendix A.3 (restating
    Hyvärinen & Pajunen 1999) requires bounded encoder activation
    variance — otherwise the change-of-variables term in the log-density
    diverges and the identifiability constructive argument breaks.

    Warn vs fail: ``var_warn < max_var <= var_ceiling`` is a ``warn``;
    above the ceiling, or non-finite, is a ``fail``.
    """

    metric: dict[str, float] = {}
    if activations is None:
        return IdentifiabilityTheoremResult(
            theorem_name="RandomProjection",
            status="warn",
            reason=(
                "RandomProjection check skipped: fit object does not carry "
                "encoder activations / latent samples."
            ),
            metric=metric,
        )
    if activations.size == 0:
        return IdentifiabilityTheoremResult(
            theorem_name="RandomProjection",
            status="fail",
            reason="encoder activations are empty.",
            metric={"activation_var_max": float("nan")},
        )
    variances = np.var(activations, axis=0)
    finite_mask = np.isfinite(variances)
    metric["activation_var_max"] = float(np.max(variances)) if variances.size else 0.0
    metric["activation_var_min"] = float(np.min(variances)) if variances.size else 0.0
    if not np.all(finite_mask):
        return IdentifiabilityTheoremResult(
            theorem_name="RandomProjection",
            status="fail",
            reason=(
                "encoder activations contain non-finite variance; "
                "Khemakhem 2107.10098 §A.3 requires bounded variance."
            ),
            metric=metric,
        )
    max_var = float(np.max(variances))
    if max_var > float(var_ceiling):
        return IdentifiabilityTheoremResult(
            theorem_name="RandomProjection",
            status="fail",
            reason=(
                f"max activation variance {max_var:.3e} > "
                f"ceiling {float(var_ceiling):.3e}; encoder is unbounded, "
                f"random-projection identifiability fails."
            ),
            metric=metric,
        )
    if max_var > float(var_warn):
        return IdentifiabilityTheoremResult(
            theorem_name="RandomProjection",
            status="warn",
            reason=(
                f"max activation variance {max_var:.3e} > "
                f"warn-floor {float(var_warn):.3e}; encoder is large but not "
                f"yet unbounded."
            ),
            metric=metric,
        )
    return IdentifiabilityTheoremResult(
        theorem_name="RandomProjection",
        status="pass",
        reason="encoder activation variance is bounded.",
        metric=metric,
    )


def check(
    fit: Any,
    *,
    aux: Any = None,
    ground_truth_dim: int | None = None,
    aux_var_floor: float = _IVAE_AUX_VAR_FLOOR,
    min_encoder_layers: int = _IVAE_MIN_ENCODER_LAYERS,
    mech_sparsity_zero_tol: float = _MECH_SPARSITY_ZERO_TOL,
    mech_sparsity_fraction: float = _MECH_SPARSITY_FRACTION,
    activation_var_warn: float = _RANDPROJ_ACTIVATION_VAR_WARN,
    activation_var_ceiling: float = _RANDPROJ_ACTIVATION_VAR_CEILING,
) -> IdentifiabilityReport:
    """Run every applicable identifiability theorem check on ``fit``.

    ``fit`` may be an :class:`IdentifiableFactorFitResult` (in which case
    all three theorems are checked), a
    :class:`gamfit.recipes.PartialSupervisionFit` (iVAE-aux + random
    projection only — no decoder is fit explicitly in that recipe), or any
    object that exposes the attributes ``T_supervised``/``T_free``/
    ``decoder``/``encoder_state``/``aux_prior_weight``/
    ``mech_sparsity_weight`` (duck-typing for forward compatibility).

    Parameters
    ----------
    fit : object
        Fit result to introspect.
    aux : array-like, optional
        Auxiliary covariates used at fit time, when the fit object does
        not already store them (e.g. ``PartialSupervisionFit`` stores
        ``aux`` internally on the recipe, not on the fit). Required for
        the iVAE check if no ``aux`` attribute is present on the fit.
    ground_truth_dim : int, optional
        Ground-truth latent dimension, when known (e.g. from a simulator).
        Enables the ``state_dim >= ground_truth_dim`` precondition of the
        MechanismSparsity check.
    aux_var_floor, min_encoder_layers, mech_sparsity_zero_tol,
    mech_sparsity_fraction, activation_var_warn, activation_var_ceiling : float / int
        Numerical thresholds overriding the paper-cited defaults. Override
        with care — the defaults are tied to the proofs in
        Khemakhem 2107.10098 and Lachapelle 2401.04890.

    Returns
    -------
    IdentifiabilityReport
        Per-theorem status / reason / metric breakdown. Overall status is
        ``report.status``.
    """

    aux_used = aux
    if aux_used is None:
        aux_used = getattr(fit, "aux", None)
    if aux_used is not None:
        aux_used = np.asarray(aux_used, dtype=float)
        if aux_used.ndim == 1:
            aux_used = aux_used.reshape(-1, 1)

    t_sup = getattr(fit, "T_supervised", None)
    t_free = getattr(fit, "T_free", None)
    if t_sup is not None:
        t_sup = np.asarray(t_sup)
    if t_free is not None:
        t_free = np.asarray(t_free)

    n_supervised: int | None = None
    if t_sup is not None and t_sup.ndim == 2:
        n_supervised = int(t_sup.shape[1])
    n_free: int | None = None
    if t_free is not None and t_free.ndim == 2:
        n_free = int(t_free.shape[1])

    decoder = getattr(fit, "decoder", None)
    if decoder is not None:
        decoder = np.asarray(decoder)

    encoder_state = getattr(fit, "encoder_state", None)
    encoder_depth: int | None = None
    if isinstance(encoder_state, dict):
        # ``state_dict`` keys for ``nn.Sequential`` are ``"<idx>.weight"`` /
        # ``"<idx>.bias"`` — counting unique ``.weight`` keys gives the
        # number of affine layers.
        weight_keys = {
            k.rsplit(".", 1)[0]
            for k in encoder_state
            if k.endswith(".weight")
        }
        encoder_depth = len(weight_keys) if weight_keys else None

    mech_w = getattr(fit, "mech_sparsity_weight", None)

    state_dim: int | None = None
    if n_supervised is not None and n_free is not None:
        state_dim = int(n_supervised) + int(n_free)

    activations: np.ndarray | None
    if t_sup is not None and t_free is not None:
        activations = np.concatenate(
            [np.asarray(t_sup), np.asarray(t_free)], axis=1
        )
    elif t_sup is not None:
        activations = np.asarray(t_sup)
    elif t_free is not None:
        activations = np.asarray(t_free)
    else:
        activations = None

    ivae = _check_ivae(
        aux=aux_used,
        n_supervised=n_supervised,
        encoder_depth=encoder_depth,
        aux_var_floor=float(aux_var_floor),
        min_encoder_layers=int(min_encoder_layers),
    )
    mech = _check_mechanism_sparsity(
        decoder=decoder,
        n_supervised=n_supervised,
        n_free=n_free,
        state_dim=state_dim,
        ground_truth_dim=(
            int(ground_truth_dim) if ground_truth_dim is not None else None
        ),
        mech_sparsity_weight=(
            float(mech_w) if mech_w is not None else None
        ),
        zero_tol=float(mech_sparsity_zero_tol),
        min_zero_fraction=float(mech_sparsity_fraction),
    )
    rand = _check_random_projection(
        activations=activations,
        var_warn=float(activation_var_warn),
        var_ceiling=float(activation_var_ceiling),
    )
    return IdentifiabilityReport(theorems=[ivae, mech, rand])


@dataclass(slots=True)
class IdentifiableFactorFitResult:
    """Output of :func:`identifiable_factor_fit`.

    Attributes
    ----------
    T_supervised : np.ndarray, shape ``(N, n_supervised)``
        Auxiliary-conditioned latent block. Identified by the iVAE theorem
        (Khemakhem 2107.10098 Thm. 1) up to a component-wise invertible
        transform when ``aux`` provides ``>= 2 n_supervised + 1`` distinct
        conditioning values.
    T_free : np.ndarray, shape ``(N, n_free)``
        Mechanism-sparsity-regularised latent block. Identified by the
        Lachapelle 2401.04890 theorem up to permutation + signed scaling
        when the decoder Jacobian on these columns is full rank and the
        sparsity penalty is active.
    evidence : float
        Laplace-style log marginal-likelihood proxy
        ``-0.5 * N * log(RSS/N) - 0.5 * total_penalty``. Higher is better.
        Sign convention matches "log evidence", not "negative log evidence".
    decoder : np.ndarray, shape ``(P, n_supervised + n_free)``
        Linear decoder ``X_hat = T @ decoder.T``.
    aux_prior_weight : float
        Final scalar weight used for the iVAE auxiliary prior.
    mech_sparsity_weight : float
        Final scalar weight used for the mechanism-sparsity prior.
    encoder_state : dict[str, np.ndarray]
        ``state_dict``-style snapshot of the encoder. Useful for
        out-of-sample prediction.
    warnings : list[str]
        Human-readable preconditions of the iVAE+mech-sparsity theorem that
        do *not* hold for this fit. An empty list means all preconditions
        are satisfied within numerical tolerance. Mirror of
        ``report.as_warnings()`` for backward compatibility.
    aux : np.ndarray
        Auxiliary covariates used at fit time. Stored on the result so
        downstream :func:`check` calls can re-verify the iVAE preconditions
        without the caller having to thread ``aux`` through manually.
    report : IdentifiabilityReport | None
        Structured per-theorem identifiability report. Populated when
        ``check_identifiability=True`` was passed to
        :func:`identifiable_factor_fit` (the default).
    """

    T_supervised: np.ndarray
    T_free: np.ndarray
    evidence: float
    decoder: np.ndarray
    aux_prior_weight: float
    mech_sparsity_weight: float
    encoder_state: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)
    aux: np.ndarray | None = None
    report: "IdentifiabilityReport | None" = None


_ENCODER_RE = re.compile(r"^\s*mlp\s*\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*$")


def _parse_encoder_spec(spec: str) -> list[int]:
    """Parse ``"mlp[256, 256]"`` / ``"linear"`` into a hidden-width list.

    Returns ``[]`` for ``"linear"`` (no hidden layer, the latent head is the
    only layer). Raises :class:`ValueError` for any other form.
    """

    if not isinstance(spec, str):
        raise ValueError(
            f"encoder must be a string like 'linear' or 'mlp[256, 256]'; "
            f"got {type(spec).__name__}"
        )
    text = spec.strip().lower()
    if text == "linear":
        return []
    m = _ENCODER_RE.match(text)
    if m is None:
        raise ValueError(
            f"encoder={spec!r} is not a recognized encoder spec; "
            f"expected 'linear' or 'mlp[w1, w2, ...]' with positive integer widths"
        )
    widths = [int(piece.strip()) for piece in m.group(1).split(",")]
    if any(w <= 0 for w in widths):
        raise ValueError(
            f"encoder={spec!r}: all hidden widths must be positive, got {widths}"
        )
    return widths


def _validate_inputs(
    X: Any, aux: Any, n_supervised: int, n_free: int
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.ascontiguousarray(np.asarray(X, dtype=float))
    if x_arr.ndim != 2:
        raise ValueError(
            f"X must be a 2D array of shape (N, P); got shape {x_arr.shape}"
        )
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("X must be finite")
    n, p = x_arr.shape
    if n < 4:
        raise ValueError(f"identifiable_factor_fit requires N >= 4; got N={n}")
    if p < 1:
        raise ValueError(f"X must have at least one feature; got P={p}")
    if int(n_supervised) < 1:
        raise ValueError(f"n_supervised must be >= 1; got {n_supervised}")
    if int(n_free) < 1:
        raise ValueError(f"n_free must be >= 1; got {n_free}")

    aux_arr = np.asarray(aux, dtype=float)
    if aux_arr.ndim == 1:
        if int(n_supervised) != 1:
            raise ValueError(
                f"aux is 1D but n_supervised={n_supervised}; pass aux with "
                f"shape (N, {n_supervised}) to disambiguate"
            )
        aux_arr = aux_arr.reshape(-1, 1)
    if aux_arr.ndim != 2:
        raise ValueError(
            f"aux must be 1D (when n_supervised == 1) or 2D of shape "
            f"(N, n_supervised); got shape {aux_arr.shape}"
        )
    if aux_arr.shape[0] != n:
        raise ValueError(
            f"aux first dim {aux_arr.shape[0]} must equal N={n}"
        )
    if aux_arr.shape[1] != int(n_supervised):
        raise ValueError(
            f"aux second dim {aux_arr.shape[1]} must equal n_supervised={n_supervised}"
        )
    if not np.all(np.isfinite(aux_arr)):
        raise ValueError("aux must be finite")
    return x_arr, np.ascontiguousarray(aux_arr)


def _check_preconditions(
    aux: np.ndarray,
    decoder: np.ndarray,
    n_supervised: int,
    n_free: int,
    encoder_depth: int,
    mech_sparsity_weight: float,
) -> list[str]:
    """Verify the iVAE + mechanism-sparsity theorem preconditions.

    Returns a list of human-readable strings, one per failed precondition.
    """

    issues: list[str] = []

    # 1. aux must vary across rows — a constant aux vector carries no
    # conditioning information and the iVAE identifiability theorem
    # collapses to standard (non-identified) ICA.
    aux_std = np.std(aux, axis=0)
    if not np.all(aux_std > 1e-9):
        zero_axes = np.where(aux_std <= 1e-9)[0].tolist()
        issues.append(
            f"iVAE identifiability requires auxiliary covariate variation; "
            f"aux axes {zero_axes} are constant across observations, so the "
            f"Khemakhem 2107.10098 Theorem 1 conditioning rank fails."
        )

    # 2. decoder Jacobian on T_free columns must be rank >= n_free.
    # Decoder is linear here, so the Jacobian is W itself; we check the
    # rank of its free-block columns directly.
    free_cols = decoder[:, n_supervised : n_supervised + n_free]
    rank = int(np.linalg.matrix_rank(free_cols, tol=1e-8))
    if rank < n_free:
        issues.append(
            f"mechanism-sparsity identifiability requires the decoder "
            f"Jacobian on T_free to be rank >= n_free={n_free}, but the "
            f"fitted decoder has rank {rank} on those columns. The free "
            f"block collapsed during fitting."
        )

    # 3. mechanism-sparsity penalty must be active. A zero weight means
    # the sparsity prior contributed nothing and Lachapelle 2401.04890
    # Theorem identification does not apply.
    if not (mech_sparsity_weight > 0.0):
        issues.append(
            "mechanism-sparsity identifiability requires a strictly positive "
            f"sparsity weight; got {mech_sparsity_weight}."
        )

    # 4. encoder must be non-trivial. Khemakhem 2107.10098 §3 requires
    # E to be a "sufficiently expressive" smooth map; a bare linear
    # encoder does not satisfy this for non-Gaussian sources.
    if encoder_depth < 2:
        issues.append(
            f"iVAE identifiability requires a non-trivial encoder "
            f"(>= 2 layers per Khemakhem 2107.10098 §3); got encoder depth "
            f"{encoder_depth}. Use encoder='mlp[w, w]' or deeper."
        )

    return issues


def _build_encoder(
    p_features: int, latent_dim: int, hidden_widths: list[int], torch_mod: Any
) -> Any:
    """Build a torch ``nn.Sequential`` encoder.

    ``hidden_widths == []`` produces a single linear layer. Otherwise stacks
    ``Linear -> GELU`` blocks ending with a ``Linear`` head onto ``latent_dim``.
    """

    nn = torch_mod.nn
    layers: list[Any] = []
    in_dim = p_features
    for w in hidden_widths:
        layers.append(nn.Linear(in_dim, w))
        layers.append(nn.GELU())
        in_dim = w
    layers.append(nn.Linear(in_dim, latent_dim))
    return nn.Sequential(*layers)


def _count_layers(encoder: Any) -> int:
    """Count ``nn.Linear`` modules — the canonical "layer count" for an MLP."""

    n = 0
    for module in encoder.modules():
        if module.__class__.__name__ == "Linear":
            n += 1
    return n


def _resolve_weight(
    weight: Any, name: str
) -> tuple[float, bool]:
    """Return (numeric_weight, auto_flag).

    ``"auto"`` resolves to ``(1.0, True)`` — REML wiring for an arbitrary
    custom torch encoder is not in scope here, so ``"auto"`` triggers a
    coarse golden-section search instead. Numeric values must be positive
    and finite.
    """

    if isinstance(weight, str):
        if weight.strip().lower() != "auto":
            raise ValueError(
                f"{name}: only 'auto' is accepted as a string; got {weight!r}"
            )
        return 1.0, True
    try:
        w = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be 'auto' or a positive finite float; got {weight!r}"
        ) from exc
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(
            f"{name} must be positive and finite; got {w}"
        )
    return w, False


def _one_fit(
    x_t: Any,
    aux_t: Any,
    n_supervised: int,
    n_free: int,
    hidden_widths: list[int],
    aux_w: float,
    mech_w: float,
    max_iter: int,
    learning_rate: float,
    seed: int,
    torch_mod: Any,
) -> tuple[Any, Any, float, float, float]:
    """Run one inner-loop fit at fixed scalar weights.

    Returns ``(encoder, decoder_W, rss, total_penalty, evidence)``. The
    encoder is the trained ``nn.Sequential``; ``decoder_W`` is a numpy array
    of shape ``(P, n_supervised + n_free)``.
    """

    torch = torch_mod
    nn = torch.nn

    n_obs, p_features = int(x_t.shape[0]), int(x_t.shape[1])
    latent_dim = int(n_supervised) + int(n_free)

    gen = torch.Generator(device=x_t.device).manual_seed(int(seed))
    # Manually seed parameters reproducibly via the Generator above.
    with torch.no_grad():
        encoder = _build_encoder(p_features, latent_dim, hidden_widths, torch_mod)
        encoder = encoder.to(dtype=x_t.dtype, device=x_t.device)
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                bound = 1.0 / math.sqrt(max(1, fan_in))
                module.weight.uniform_(-bound, bound, generator=gen)
                module.bias.zero_()

    decoder = nn.Linear(latent_dim, p_features, bias=False).to(
        dtype=x_t.dtype, device=x_t.device
    )
    with torch.no_grad():
        bound = 1.0 / math.sqrt(latent_dim)
        decoder.weight.uniform_(-bound, bound, generator=gen)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(params, lr=float(learning_rate))

    rust = rust_module()

    # Pre-build mechanism-sparsity descriptor parts: singleton feature groups
    # (one per output feature) give an element-wise smoothed-L1 over the
    # free-latent rows of the decoder, which is the Lachapelle 2401.04890
    # mechanism-sparsity functional.
    feature_groups = [[j] for j in range(p_features)]
    mech_pen = rust.MechanismSparsityPenalty(
        feature_groups, float(mech_w), float(max(1, n_obs))
    )

    aux_np = aux_t.detach().cpu().numpy()
    aux_scale = np.ones_like(aux_np)

    rss = 0.0
    total_pen = 0.0
    for _step in range(int(max_iter)):
        optim.zero_grad(set_to_none=True)
        t = encoder(x_t)
        t_sup = t[:, :n_supervised]
        t_free = t[:, n_supervised : n_supervised + n_free]
        x_hat = decoder(t)
        recon = ((x_hat - x_t) ** 2).sum()

        # Aux conditional prior (Gaussian iVAE prior on T_sup given aux).
        t_sup_np = np.ascontiguousarray(t_sup.detach().cpu().numpy())
        aux_val, aux_grad = rust.conditional_prior_ivae(
            float(aux_w), t_sup_np, aux_np, aux_scale
        )
        aux_grad_t = torch.as_tensor(
            np.asarray(aux_grad), dtype=t.dtype, device=t.device
        )
        # Inject the analytic Rust gradient into autograd via a surrogate loss
        # whose backward equals the precomputed gradient.
        aux_surrogate = (t_sup * aux_grad_t).sum()

        # Mechanism sparsity on the free-latent rows of the decoder
        # (decoder.weight is shape (P, latent_dim); transpose to
        # (latent_dim, P) and take the free rows).
        w_full = decoder.weight.t()
        w_free = w_full[n_supervised : n_supervised + n_free, :]
        w_free_np = np.ascontiguousarray(
            w_free.detach().cpu().numpy().astype(np.float64)
        )
        mech_val, mech_grad = mech_pen.value_grad(w_free_np)
        mech_grad_t = torch.as_tensor(
            np.asarray(mech_grad), dtype=w_free.dtype, device=w_free.device
        )
        mech_surrogate = (w_free * mech_grad_t).sum()

        # Total surrogate loss: recon has direct autograd; the two penalty
        # surrogates have value-matched gradients (Rust-analytic) but their
        # numeric value is replaced below for evidence reporting.
        loss = recon + aux_surrogate + mech_surrogate
        loss.backward()
        optim.step()

        rss = float(recon.detach().cpu().item())
        total_pen = float(aux_val) + float(mech_val)

    # Final-pass true (RSS, penalty) used by the Rust weight-selection /
    # evidence primitive. The Laplace-style log marginal-likelihood proxy
    # itself is computed entirely in Rust by
    # ``identifiable_factor_select_weights_array`` — no statistical math
    # lives in this Python file.
    with torch.no_grad():
        t = encoder(x_t)
        t_sup = t[:, :n_supervised]
        x_hat = decoder(t)
        rss = float(((x_hat - x_t) ** 2).sum().item())
        aux_val, _ = rust.conditional_prior_ivae(
            float(aux_w),
            np.ascontiguousarray(t_sup.detach().cpu().numpy()),
            aux_np,
            aux_scale,
        )
        w_full = decoder.weight.t()
        w_free = w_full[n_supervised : n_supervised + n_free, :]
        mech_val, _ = mech_pen.value_grad(
            np.ascontiguousarray(
                w_free.detach().cpu().numpy().astype(np.float64)
            )
        )
    total_pen = float(aux_val) + float(mech_val)
    return encoder, decoder, rss, total_pen


def _auto_weight_search(
    x_t: Any,
    aux_t: Any,
    n_supervised: int,
    n_free: int,
    hidden_widths: list[int],
    auto_aux: bool,
    auto_mech: bool,
    aux_w0: float,
    mech_w0: float,
    max_iter: int,
    learning_rate: float,
    seed: int,
    torch_mod: Any,
) -> tuple[float, float, Any, Any, float, float, float]:
    """Coarse log-grid search over the ``"auto"`` weights.

    For each weight set to ``"auto"`` we sweep a 5-point log-spaced grid
    centred on the seed value. Python's only responsibility is to evaluate
    ``(rss, penalty)`` at each grid cell by running ``_one_fit`` (the torch
    encoder loop is the legitimate Python escape). The argmax / evidence
    formula / tie-breaking is delegated to Rust via
    ``rust_module().identifiable_factor_select_weights_array``.
    """

    aux_grid = (
        [aux_w0 * 10 ** e for e in (-2.0, -1.0, 0.0, 1.0, 2.0)] if auto_aux else [aux_w0]
    )
    mech_grid = (
        [mech_w0 * 10 ** e for e in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        if auto_mech
        else [mech_w0]
    )

    n_obs = int(x_t.shape[0])
    g1 = len(aux_grid)
    g2 = len(mech_grid)
    rss_grid = np.zeros((g1, g2), dtype=np.float64)
    pen_grid = np.zeros((g1, g2), dtype=np.float64)
    cells: list[list[tuple[Any, Any] | None]] = [[None] * g2 for _ in range(g1)]
    for i, aw in enumerate(aux_grid):
        for j, mw in enumerate(mech_grid):
            encoder, decoder, rss, pen = _one_fit(
                x_t, aux_t, n_supervised, n_free, hidden_widths,
                aw, mw, max_iter, learning_rate, seed, torch_mod,
            )
            rss_grid[i, j] = rss
            pen_grid[i, j] = pen
            cells[i][j] = (encoder, decoder)

    selection = rust_module().identifiable_factor_select_weights_array(
        np.ascontiguousarray(rss_grid),
        np.ascontiguousarray(pen_grid),
        np.ascontiguousarray(np.asarray(aux_grid, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(mech_grid, dtype=np.float64)),
        int(n_obs),
    )
    bi = int(selection["best_i"])
    bj = int(selection["best_j"])
    best_cell = cells[bi][bj]
    assert best_cell is not None  # populated above
    encoder, decoder = best_cell
    return (
        float(selection["best_lam1"]),
        float(selection["best_lam2"]),
        encoder,
        decoder,
        float(rss_grid[bi, bj]),
        float(pen_grid[bi, bj]),
        float(selection["best_evidence"]),
    )


def identifiable_factor_fit(
    X: Any,
    aux: Any,
    n_supervised: int,
    n_free: int,
    *,
    mech_sparsity_weight: Any = "auto",
    aux_prior_weight: Any = "auto",
    encoder: str = "mlp[256, 256]",
    max_iter: int = 400,
    learning_rate: float = 1.0e-2,
    random_state: int = 0,
    check_identifiability: bool = True,
) -> IdentifiableFactorFitResult:
    """Fit an identifiable factor model combining iVAE + mechanism sparsity.

    The encoder ``E(X) -> (T_sup, T_free)`` produces a real-valued latent
    split. ``T_sup`` is supervised by ``aux`` via an iVAE-style Gaussian
    auxiliary-conditional prior; ``T_free`` is unsupervised and constrained
    by a mechanism-sparsity penalty on its decoder rows. Both penalty
    weights default to ``"auto"`` and are selected by a coarse log-grid
    search over a Laplace-style log marginal-likelihood proxy.

    Parameters
    ----------
    X : array-like, shape ``(N, P)``
        Observations.
    aux : array-like, shape ``(N, n_supervised)`` or ``(N,)`` if
        ``n_supervised == 1``
        Auxiliary covariates / labels. Each axis is treated as the mean of
        the Gaussian iVAE prior for the matching supervised latent axis.
    n_supervised, n_free : int
        Dimensions of the supervised / free latent blocks. Required —
        forcing the user to make the split explicit avoids silent
        identifiability surprises.
    mech_sparsity_weight, aux_prior_weight : ``"auto"`` or positive float
        Penalty weights. ``"auto"`` triggers a 5-point log-grid search per
        ``"auto"`` flag and selects the weight maximizing the Laplace
        evidence proxy.
    encoder : str
        ``"linear"`` for a single-Linear encoder, or ``"mlp[w1, w2, ...]"``
        for an MLP of widths ``w_i`` with GELU activations and a Linear
        head onto the latent dim.
    max_iter, learning_rate, random_state : optimiser controls.

    Returns
    -------
    :class:`IdentifiableFactorFitResult`
        Fitted latents, evidence, decoder, final weights, and a list of
        precondition warnings (empty if the identifiability theorems'
        preconditions all hold).

    Notes
    -----
    The ``evidence`` sign convention is "log evidence" — *higher is better*.
    The proxy is approximate: REML wiring for arbitrary custom torch
    encoders is not yet plumbed through the Rust engine, so ``"auto"``
    uses a coarse Laplace-style log marginal-likelihood proxy with a
    log-grid search rather than the exact REML score that
    :func:`gamfit.fit` returns for formula-based smooths.
    """

    x_np, aux_np = _validate_inputs(X, aux, int(n_supervised), int(n_free))
    hidden_widths = _parse_encoder_spec(encoder)

    try:
        import torch as torch_mod
    except ImportError as exc:  # pragma: no cover - torch is a required extra
        raise ImportError(
            "identifiable_factor_fit requires PyTorch; install gamfit[torch]"
        ) from exc

    aux_w0, auto_aux = _resolve_weight(aux_prior_weight, "aux_prior_weight")
    mech_w0, auto_mech = _resolve_weight(mech_sparsity_weight, "mech_sparsity_weight")

    torch_mod.manual_seed(int(random_state))
    x_t = torch_mod.as_tensor(x_np, dtype=torch_mod.float64)
    aux_t = torch_mod.as_tensor(aux_np, dtype=torch_mod.float64)

    if auto_aux or auto_mech:
        aux_w, mech_w, encoder_module, decoder_module, _rss, _pen, evidence = (
            _auto_weight_search(
                x_t, aux_t, int(n_supervised), int(n_free), hidden_widths,
                auto_aux, auto_mech, aux_w0, mech_w0,
                int(max_iter), float(learning_rate), int(random_state), torch_mod,
            )
        )
    else:
        encoder_module, decoder_module, rss_val, pen_val = _one_fit(
            x_t, aux_t, int(n_supervised), int(n_free), hidden_widths,
            aux_w0, mech_w0, int(max_iter), float(learning_rate),
            int(random_state), torch_mod,
        )
        aux_w, mech_w = aux_w0, mech_w0
        # Route the single-pair evidence through the same Rust primitive as
        # the grid search so the Laplace evidence formula has a single
        # source of truth in ``src/identifiability.rs``.
        selection = rust_module().identifiable_factor_select_weights_array(
            np.ascontiguousarray(np.array([[rss_val]], dtype=np.float64)),
            np.ascontiguousarray(np.array([[pen_val]], dtype=np.float64)),
            np.ascontiguousarray(np.array([aux_w], dtype=np.float64)),
            np.ascontiguousarray(np.array([mech_w], dtype=np.float64)),
            int(x_t.shape[0]),
        )
        evidence = float(selection["best_evidence"])

    with torch_mod.no_grad():
        t = encoder_module(x_t)
        t_sup_np = np.ascontiguousarray(
            t[:, : int(n_supervised)].detach().cpu().numpy()
        )
        t_free_np = np.ascontiguousarray(
            t[:, int(n_supervised) : int(n_supervised) + int(n_free)]
            .detach()
            .cpu()
            .numpy()
        )
        decoder_w = np.ascontiguousarray(
            decoder_module.weight.detach().cpu().numpy().astype(np.float64)
        )
        encoder_state = {
            k: v.detach().cpu().numpy().astype(np.float64).copy()
            for k, v in encoder_module.state_dict().items()
        }

    # `_count_layers` counts nn.Linear modules; "encoder depth" in
    # Khemakhem's sense is the number of affine layers. A bare 'linear'
    # encoder has depth 1 (single Linear). MLP[w, w] has depth 3
    # (Linear-GELU-Linear-GELU-Linear), which clears the >= 2 threshold.
    enc_depth = _count_layers(encoder_module)

    # Legacy precondition list — kept for stable backward-compat warnings.
    legacy_issues = _check_preconditions(
        aux=aux_np,
        decoder=decoder_w,
        n_supervised=int(n_supervised),
        n_free=int(n_free),
        encoder_depth=enc_depth,
        mech_sparsity_weight=float(mech_w),
    )

    result = IdentifiableFactorFitResult(
        T_supervised=t_sup_np,
        T_free=t_free_np,
        evidence=float(evidence),
        decoder=decoder_w,
        aux_prior_weight=float(aux_w),
        mech_sparsity_weight=float(mech_w),
        encoder_state=encoder_state,
        warnings=list(legacy_issues),
        aux=aux_np,
        report=None,
    )

    if bool(check_identifiability):
        report = check(result)
        result.report = report
        # Surface every non-pass theorem in the canonical warning channel
        # AND mirror them into ``result.warnings`` (in addition to the
        # legacy precondition strings) so callers that grep on warnings
        # see the structured-check output too.
        structured = report.as_warnings()
        for msg in structured:
            if msg not in result.warnings:
                result.warnings.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
    else:
        # Preserve legacy warning behaviour when the check is disabled.
        for msg in legacy_issues:
            warnings.warn(msg, UserWarning, stacklevel=2)

    return result
