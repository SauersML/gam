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

If any precondition of the theorem fails, the corresponding warning is emitted
via :mod:`warnings.warn` as ``UserWarning`` and recorded in
``result.report.as_warnings()``. The fit always completes — the warnings are
informational about which guarantee no longer formally holds.
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

_IVAE_AUX_SCALE_LOG_AMPLITUDE = 0.4
# Khemakhem 2107.10098 Thm. 1 identifies the latent up to a component-wise
# transform iff the conditional prior `p(t | u)` spans a 2k-dimensional set
# of natural parameters. For the diagonal Gaussian iVAE prior the natural
# parameters are `(η_1, η_2) = (μ(u) / σ(u)², −1 / (2 σ(u)²))`, which the
# constructor in ``src/identifiability/sae.rs`` checks via the column rank of the
# stacked signature ``[μ(u) ‖ log σ(u)]`` (an invertible reparameterisation).
# A *constant* scale collapses the ``log σ`` half of that signature to zero,
# leaving rank ≤ k < 2k — that is exactly the supervised-path failure mode of
# issue #576. The conditional scale must therefore be a genuine function of
# the auxiliary. A purely *linear* `log σ = a + b·u` would lie in the span of
# ``{1, μ}`` and still not lift the rank; the lift requires `log σ(u)` to be
# *nonlinear* in `u` (Khemakhem §3, and the SVD argument in
# ``ivae_precondition_pair``). We use a bounded, column-distinct nonlinear map
# of the standardised auxiliary — ``log σ_j(u) = A · tanh((j+1)·z_j)`` — whose
# amplitude ``A`` is small enough to keep `σ` well-conditioned (σ ∈ [e^−A, e^A])
# yet large enough that the ``tanh`` curvature gives `log σ` an SVD direction
# genuinely independent of the affine `μ` columns.
_AUTO_AUX_PRIOR_WEIGHT = 2.0
_AUTO_MECH_SPARSITY_WEIGHT = 1.0e-4
# The torch encoder used by this recipe is not wired into the Rust REML
# engine, so there is no exact marginal-likelihood optimizer for these two
# weights. The previous "auto" path ran a 5x5 Laplace-proxy grid centred at
# 1.0; on clean planted-factor problems it missed the useful
# mechanism-sparsity scale and paid 25 inner fits for a worse answer. "auto"
# now means these calibrated one-fit recipe weights.


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
    entries as plain strings.
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


def _gather_fit_summary(
    fit: Any,
    *,
    aux: Any,
    ground_truth_dim: int | None,
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    """Collect every numerical artefact the Rust checks need, with zero math.

    The only operation here is ``np.ndarray.tolist()`` reshaping — every
    statistic (std, rank, zero-fraction, variance) is computed in
    ``src/identifiability/precondition.rs``. This function is therefore a
    pure marshalling layer per Principle (f).
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
        t_sup = np.asarray(t_sup, dtype=float)
    if t_free is not None:
        t_free = np.asarray(t_free, dtype=float)

    n_supervised: int | None = None
    if t_sup is not None and t_sup.ndim == 2:
        n_supervised = int(t_sup.shape[1])
    n_free: int | None = None
    if t_free is not None and t_free.ndim == 2:
        n_free = int(t_free.shape[1])

    decoder = getattr(fit, "decoder", None)
    if decoder is not None:
        decoder = np.asarray(decoder, dtype=float)

    encoder_state = getattr(fit, "encoder_state", None)
    encoder_depth: int | None = None
    if isinstance(encoder_state, dict):
        # ``state_dict`` keys for ``nn.Sequential`` are ``"<idx>.weight"`` /
        # ``"<idx>.bias"`` — counting unique ``.weight`` keys is "encoder
        # depth" in Khemakhem 2107.10098 §3's sense.
        weight_keys = {
            k.rsplit(".", 1)[0]
            for k in encoder_state
            if k.endswith(".weight")
        }
        encoder_depth = len(weight_keys) if weight_keys else None

    mech_w = getattr(fit, "mech_sparsity_weight", None)

    activations: np.ndarray | None
    if t_sup is not None and t_free is not None:
        activations = np.concatenate([t_sup, t_free], axis=1)
    elif t_sup is not None:
        activations = t_sup
    elif t_free is not None:
        activations = t_free
    else:
        activations = None

    summary: dict[str, Any] = {
        "aux": aux_used.tolist() if aux_used is not None else None,
        "n_supervised": int(n_supervised) if n_supervised is not None else None,
        "n_free": int(n_free) if n_free is not None else None,
        "decoder": decoder.tolist() if decoder is not None else None,
        "encoder_depth": (
            int(encoder_depth) if encoder_depth is not None else None
        ),
        "mech_sparsity_weight": (
            float(mech_w) if mech_w is not None else None
        ),
        "activations": (
            activations.tolist() if activations is not None else None
        ),
        "ground_truth_dim": (
            int(ground_truth_dim) if ground_truth_dim is not None else None
        ),
        "thresholds": {
            "ivae_aux_var_floor": float(thresholds["ivae_aux_var_floor"]),
            "ivae_aux_rank_rtol": float(thresholds["ivae_aux_rank_rtol"]),
            "ivae_min_encoder_layers": int(
                thresholds["ivae_min_encoder_layers"]
            ),
            "mech_sparsity_fraction": float(
                thresholds["mech_sparsity_fraction"]
            ),
            "mech_sparsity_zero_tol": float(
                thresholds["mech_sparsity_zero_tol"]
            ),
            "randproj_var_warn": float(thresholds["randproj_var_warn"]),
            "randproj_var_ceiling": float(thresholds["randproj_var_ceiling"]),
        },
    }
    return summary


def check(
    fit: Any,
    *,
    aux: Any = None,
    ground_truth_dim: int | None = None,
    aux_var_floor: float = _IVAE_AUX_VAR_FLOOR,
    aux_rank_rtol: float = _IVAE_AUX_RANK_RTOL,
    min_encoder_layers: int = _IVAE_MIN_ENCODER_LAYERS,
    mech_sparsity_zero_tol: float = _MECH_SPARSITY_ZERO_TOL,
    mech_sparsity_fraction: float = _MECH_SPARSITY_FRACTION,
    activation_var_warn: float = _RANDPROJ_ACTIVATION_VAR_WARN,
    activation_var_ceiling: float = _RANDPROJ_ACTIVATION_VAR_CEILING,
) -> "IdentifiabilityReport":
    """Run every applicable identifiability theorem check on ``fit``.

    All numerical work — min-std, faer-SVD column-rank, decoder
    zero-fraction, latent variance bounds — happens in
    ``gam::identifiability::precondition``. This function is the Python
    marshalling layer: it gathers the relevant tensors off the fit, ships
    them as JSON to Rust, and rehydrates the resulting
    :class:`IdentifiabilityReport`.

    ``fit`` may be an :class:`IdentifiableFactorFitResult` (all three
    theorems are checked), a :class:`gamfit.examples.PartialSupervisionFit`
    (iVAE-aux + random projection only — no decoder is fit), or any object
    duck-typing the attributes ``T_supervised`` / ``T_free`` / ``decoder``
    / ``encoder_state`` / ``mech_sparsity_weight``.

    Parameters
    ----------
    fit : object
        Fit result to introspect.
    aux : array-like, optional
        Aux used at fit time, when not stored on the fit (e.g. for
        ``PartialSupervisionFit``).
    ground_truth_dim : int, optional
        Ground-truth latent dim from a simulator. Enables the
        ``state_dim >= ground_truth_dim`` precondition.
    aux_var_floor, aux_rank_rtol, min_encoder_layers,
    mech_sparsity_zero_tol, mech_sparsity_fraction, activation_var_warn,
    activation_var_ceiling : float / int
        Numerical thresholds overriding the paper-cited defaults.
    """

    thresholds: dict[str, float | int] = {
        "ivae_aux_var_floor": float(aux_var_floor),
        "ivae_aux_rank_rtol": float(aux_rank_rtol),
        "ivae_min_encoder_layers": int(min_encoder_layers),
        "mech_sparsity_fraction": float(mech_sparsity_fraction),
        "mech_sparsity_zero_tol": float(mech_sparsity_zero_tol),
        "randproj_var_warn": float(activation_var_warn),
        "randproj_var_ceiling": float(activation_var_ceiling),
    }
    summary = _gather_fit_summary(
        fit,
        aux=aux,
        ground_truth_dim=ground_truth_dim,
        thresholds=thresholds,
    )
    payload = json.dumps(summary)
    raw = rust_module().identifiability_check_json(payload)
    parsed = json.loads(raw)
    theorems = [
        IdentifiabilityTheoremResult(
            theorem_name=str(entry["theorem_name"]),
            status=str(entry["status"]),
            reason=str(entry["reason"]),
            metric={str(k): float(v) for k, v in entry["metric"].items()},
        )
        for entry in parsed
    ]
    return IdentifiabilityReport(theorems=theorems)




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


def _resolve_weight(weight: Any, name: str) -> float:
    """Return the numeric penalty weight.

    ``"auto"`` resolves to the calibrated recipe default for ``name``.
    Numeric values must be positive and finite.
    """

    if isinstance(weight, str):
        if weight.strip().lower() != "auto":
            raise ValueError(
                f"{name}: only 'auto' is accepted as a string; got {weight!r}"
            )
        if name == "aux_prior_weight":
            return _AUTO_AUX_PRIOR_WEIGHT
        if name == "mech_sparsity_weight":
            return _AUTO_MECH_SPARSITY_WEIGHT
        raise ValueError(f"unknown identifiable-factor weight name {name!r}")
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
    return w


def _derive_aux_scale(aux_np: np.ndarray) -> np.ndarray:
    """Derive the iVAE conditional scale ``σ(u)`` from the auxiliary.

    The Gaussian iVAE prior is ``p(t_i | u) = N(μ_i(u), σ_i(u)²)``. The mean
    is supplied directly by the auxiliary (``μ(u) = u``); this returns the
    matching conditional scale ``σ(u)`` so that the stacked natural-parameter
    signature ``[μ(u) ‖ log σ(u)]`` spans the full ``2k`` dimensions required
    by Khemakhem 2107.10098 Theorem 1 (the rank check in
    ``ConditionalPriorIvae::new``).

    For each auxiliary column we standardise to ``z_j`` (zero mean, unit
    spread over the rows) and set
    ``log σ_j(u) = A · tanh((j + 1) · z_j)`` with ``A =
    _IVAE_AUX_SCALE_LOG_AMPLITUDE``. The per-column frequency ``(j + 1)``
    pushes each ``log σ`` column into its own subspace (mirroring the
    distinct-frequency construction in ``ivae_precondition_pair``), and the
    ``tanh`` nonlinearity makes ``log σ_j`` linearly independent of the affine
    ``μ_j = u_j`` column — so for ``N ≥ 2k + 1`` rows of a genuinely varying
    auxiliary the signature reaches numerical rank ``2k``.

    A degenerate (constant) auxiliary column yields ``z_j ≡ 0`` and hence
    ``σ_j ≡ 1``; the resulting signature is then correctly rank-deficient and
    the Rust precondition reports the iVAE theorem as violated rather than
    silently fabricating identifiability. ``σ`` is always finite and strictly
    positive (``σ ∈ [e^−A, e^A]``).
    """

    aux2d = np.ascontiguousarray(np.asarray(aux_np, dtype=float))
    col_mean = aux2d.mean(axis=0, keepdims=True)
    col_std = aux2d.std(axis=0, keepdims=True)
    # A constant column has zero spread; standardising it to 0 keeps σ = 1
    # there so the (now provably) rank-deficient signature is surfaced by the
    # Rust theorem check instead of being masked.
    safe_std = np.where(col_std > 0.0, col_std, 1.0)
    z = (aux2d - col_mean) / safe_std
    freq = np.arange(1, aux2d.shape[1] + 1, dtype=float).reshape(1, -1)
    log_sigma = _IVAE_AUX_SCALE_LOG_AMPLITUDE * np.tanh(freq * z)
    return np.ascontiguousarray(np.exp(log_sigma))


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
) -> tuple[Any, Any, float, float]:
    """Run one inner-loop fit at fixed scalar weights.

    Returns ``(encoder, decoder_W, rss, total_penalty)``. The
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

    aux_np = np.ascontiguousarray(aux_t.detach().cpu().numpy())
    # Conditional scale σ(u) of the Gaussian iVAE prior, derived from the
    # auxiliary so that the natural-parameter signature [μ(u) ‖ log σ(u)]
    # spans the full 2k dimensions of Khemakhem 2107.10098 Thm. 1 (fixing
    # issue #576, where the hardcoded σ ≡ 1 made that signature rank-deficient
    # and every supervised fit raise). ``aux_prior_active`` records whether
    # the Khemakhem precondition is satisfiable for *this* auxiliary: when the
    # auxiliary is genuinely degenerate the conditional prior is mathematically
    # non-identifiable, so the Rust constructor raises — we then leave the
    # prior contribution at zero and let :func:`check` report iVAE = fail,
    # honouring the recipe contract that the fit always completes.
    aux_scale = _derive_aux_scale(aux_np)
    zero_sup = np.zeros((n_obs, n_supervised), dtype=np.float64)
    try:
        rust.conditional_prior_ivae(float(aux_w), zero_sup, aux_np, aux_scale)
        aux_prior_active = True
    except ValueError:
        aux_prior_active = False

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
        if aux_prior_active:
            t_sup_np = np.ascontiguousarray(t_sup.detach().cpu().numpy())
            aux_val, aux_grad = rust.conditional_prior_ivae(
                float(aux_w), t_sup_np, aux_np, aux_scale
            )
            aux_grad_t = torch.as_tensor(
                np.asarray(aux_grad), dtype=t.dtype, device=t.device
            )
            # Inject the analytic Rust gradient into autograd via a surrogate
            # loss whose backward equals the precomputed gradient.
            aux_surrogate = (t_sup * aux_grad_t).sum()
        else:
            aux_val = 0.0
            aux_surrogate = t_sup.sum() * 0.0

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
        if aux_prior_active:
            aux_val, _ = rust.conditional_prior_ivae(
                float(aux_w),
                np.ascontiguousarray(t_sup.detach().cpu().numpy()),
                aux_np,
                aux_scale,
            )
        else:
            aux_val = 0.0
        w_full = decoder.weight.t()
        w_free = w_full[n_supervised : n_supervised + n_free, :]
        mech_val, _ = mech_pen.value_grad(
            np.ascontiguousarray(
                w_free.detach().cpu().numpy().astype(np.float64)
            )
        )
    total_pen = float(aux_val) + float(mech_val)
    return encoder, decoder, rss, total_pen


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
    weights default to ``"auto"``, which resolves to calibrated recipe
    weights. The resulting single fit is scored with a Laplace-style log
    marginal-likelihood proxy.

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
        Penalty weights. ``"auto"`` resolves to the recipe defaults
        ``mech_sparsity_weight=1e-4`` and ``aux_prior_weight=2.0``.
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
    encoders is not yet plumbed through the Rust engine. ``"auto"`` therefore
    uses calibrated recipe weights rather than treating the Laplace proxy as
    an exact REML selector like :func:`gamfit.fit` returns for formula-based
    smooths.
    """

    x_np, aux_np = _validate_inputs(X, aux, int(n_supervised), int(n_free))
    hidden_widths = _parse_encoder_spec(encoder)

    try:
        import torch as torch_mod
    except ImportError as exc:  # pragma: no cover - torch is a required extra
        raise ImportError(
            "identifiable_factor_fit requires PyTorch; install gamfit[torch]"
        ) from exc

    aux_w = _resolve_weight(aux_prior_weight, "aux_prior_weight")
    mech_w = _resolve_weight(mech_sparsity_weight, "mech_sparsity_weight")

    torch_mod.manual_seed(int(random_state))
    x_t = torch_mod.as_tensor(x_np, dtype=torch_mod.float64)
    aux_t = torch_mod.as_tensor(aux_np, dtype=torch_mod.float64)

    encoder_module, decoder_module, rss_val, pen_val = _one_fit(
        x_t, aux_t, int(n_supervised), int(n_free), hidden_widths,
        aux_w, mech_w, int(max_iter), float(learning_rate),
        int(random_state), torch_mod,
    )
    # Route the single-pair evidence through the Rust primitive used by
    # Rust-side weight-selection tests so the Laplace evidence formula has a
    # single source of truth in ``src/identifiability/sae.rs``.
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

    # Encoder depth (number of nn.Linear layers) is recovered inside
    # ``check`` from ``encoder_state`` keys, so we do not duplicate the
    # _count_layers call here. Rust performs every numerical theorem
    # precondition test downstream.

    result = IdentifiableFactorFitResult(
        T_supervised=t_sup_np,
        T_free=t_free_np,
        evidence=float(evidence),
        decoder=decoder_w,
        aux_prior_weight=float(aux_w),
        mech_sparsity_weight=float(mech_w),
        encoder_state=encoder_state,
        aux=aux_np,
        report=None,
    )

    if bool(check_identifiability):
        report = check(result)
        result.report = report
        for msg in report.as_warnings():
            warnings.warn(msg, UserWarning, stacklevel=2)

    return result
