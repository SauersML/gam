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

>>> result = gamfit.identifiability.identifiable_factor_fit(
...     X, aux=labels, n_supervised=3, n_free=3,
...     encoder="mlp[256, 256]",
... )
>>> result.T_supervised.shape
(N, 3)
>>> result.T_free.shape
(N, 3)
>>> result.profile_log_likelihood  # higher = better at fixed weights; not evidence

A result is only ever returned from a certified stationary point of the
penalized objective; an optimization that does not certify raises
:class:`gamfit.errors.FitConvergenceError`. An auxiliary that cannot identify
the iVAE conditional prior (Khemakhem Thm. 1's 2k-rank condition) is a
precondition failure and raises before any optimization. The remaining
theorem preconditions (encoder depth, decoder sparsity, latent variance) are
properties of the certified fit: when one fails, the corresponding warning is
emitted via :mod:`warnings.warn` as ``UserWarning`` and recorded in
``result.report.as_warnings()``.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._api import conditional_prior_ivae, derive_ivae_aux_scale, mechanism_sparsity_jacobian
from ._binding import rust_module

if TYPE_CHECKING:
    from ._rust import FitConvergenceError
else:
    from ._exceptions import FitConvergenceError


class IdentifiableFactorFitConvergenceError(FitConvergenceError):
    """:class:`gamfit.errors.FitConvergenceError` from ``identifiable_factor_fit``,
    carrying the certificate that failed and a checkpoint to resume from.

    A subclass rather than attributes set on the base instance, so the fields
    are declared: ``grad_inf`` is the returned parameters' ``‖∇f‖∞``,
    ``grad_inf_init`` the starting one, ``grad_tol`` the relative tolerance the
    certificate compared them against, ``max_evals`` and ``n_iter`` the budget
    and the L-BFGS iterations spent, ``objective_value`` the loss there, and the
    two ``checkpoint_*`` fields the encoder state dict and decoder weights.
    Callers catching ``FitConvergenceError`` still catch it.
    """

    grad_inf: float
    grad_inf_init: float
    grad_tol: float
    max_evals: int
    n_iter: int
    objective_value: float
    checkpoint_encoder_state: dict[str, np.ndarray]
    checkpoint_decoder: np.ndarray

__all__ = [
    "IdentifiabilityReport",
    "IdentifiabilityTheoremResult",
    "IdentifiableFactorFitResult",
    "check",
    "conditional_prior_ivae",
    "derive_ivae_aux_scale",
    "identifiable_factor_fit",
    "mechanism_sparsity_jacobian",
]


# ---------------------------------------------------------------------------
# Per-theorem report dataclasses (Principle (f): identifiability theorems as
# runnable diagnostics — every guarantee a recipe claims is checked at
# fit-time and reported in the result).
# ---------------------------------------------------------------------------


# The theorem-check thresholds (encoder depth, mechanism-sparsity fraction and
# zero tolerance, random-projection variance bounds) live with their paper
# citations in ``gam_identifiability::precondition``. The recipe penalty
# weights and the iVAE aux-scale amplitude live in ``gam_sae::identifiability``.


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
    }
    return summary


def check(
    fit: Any,
    *,
    aux: Any = None,
    ground_truth_dim: int | None = None,
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

    The theorem thresholds are the paper-cited defaults owned by
    ``gam_identifiability::precondition::Thresholds``. Whether an aux column is
    constant, and the aux column rank, are decided by the resolution of the
    arithmetic that measures them and take no threshold.
    """

    summary = _gather_fit_summary(
        fit,
        aux=aux,
        ground_truth_dim=ground_truth_dim,
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
        sparsity penalty is active. Each column has unit second moment over
        the fitted rows: that fixes the scale the theorem leaves free, which
        the penalized objective could otherwise lower without bound by
        growing ``T_free`` and shrinking its decoder rows.
    free_scale : np.ndarray, shape ``(n_free,)``
        Root second moment of each raw free encoder output over the fitted
        rows, so ``T_free = encoder(X)[:, n_supervised:] / free_scale``.
        Apply the same division to encode new rows.
    profile_log_likelihood : float
        Penalized Gaussian profile log-likelihood at the fitted penalty weights,
        ``-0.5 * N * log(RSS/N) - 0.5 * total_penalty``. Higher is better. No
        log-determinant or Occam term enters, so it is not a marginal likelihood
        and does not price model complexity.
    decoder : np.ndarray, shape ``(P, n_supervised + n_free)``
        Linear decoder ``X_hat = T @ decoder.T``.
    aux_prior_weight : float
        Final scalar weight used for the iVAE auxiliary prior.
    mech_sparsity_weight : float
        Final scalar weight used for the mechanism-sparsity prior.
    encoder_state : dict[str, np.ndarray]
        ``state_dict``-style snapshot of the encoder. Useful for
        out-of-sample prediction together with ``free_scale``.
    stationarity : float
        Certified ``‖∇f(θ̂)‖∞ / ‖∇f(θ₀)‖∞`` of the penalized objective ``f``
        over all encoder and decoder parameters, at most the ``grad_tol``
        the fit was asked for.
    n_iter : int
        L-BFGS iterations taken to reach the certified point.
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
    free_scale: np.ndarray
    profile_log_likelihood: float
    decoder: np.ndarray
    aux_prior_weight: float
    mech_sparsity_weight: float
    encoder_state: dict[str, np.ndarray]
    stationarity: float
    n_iter: int
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
    ``log σ_j(u) = A · tanh((j + 1) · z_j)`` with the Rust-owned amplitude
    ``A = gam_sae::identifiability::IVAE_AUX_SCALE_LOG_AMPLITUDE``. The
    per-column frequency ``(j + 1)``
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
    return np.ascontiguousarray(rust_module().derive_ivae_aux_scale(aux2d))


def _flat_grad_inf(params: list[Any]) -> float:
    """``‖∇‖∞`` over every parameter's accumulated ``.grad``."""

    return max(
        float(p.grad.detach().abs().max().item()) if p.grad is not None else 0.0
        for p in params
    )


def _one_fit(
    x_t: Any,
    aux_t: Any,
    n_supervised: int,
    n_free: int,
    hidden_widths: list[int],
    aux_w: float,
    mech_w: float,
    max_evals: int,
    grad_tol: float,
    seed: int,
    torch_mod: Any,
) -> tuple[Any, Any, float, float, np.ndarray, float, int]:
    """Minimize the penalized objective at fixed weights to a certified point.

    The objective over encoder and decoder parameters ``θ`` is

    ``f(θ) = ‖X − T Wᵀ‖² + aux_w·iVAE(T_sup | aux) + mech_w·Ω(W_free)``

    with ``T = [T_sup ‖ T_free]`` and ``T_free`` the raw free encoder output
    divided by its per-column root second moment. Without that division ``f``
    has no minimizer: ``T_free → c·T_free`` with ``W_free → W_free / c``
    leaves the reconstruction unchanged and strictly lowers ``Ω`` for every
    ``c > 1``. With it, ``f`` is invariant to the raw free scale, and the
    mechanism-sparsity penalty prices decoder sparsity at a fixed latent
    scale, which is the setting of Lachapelle 2401.04890.

    ``f`` is minimized by L-BFGS with a strong-Wolfe line search. The only
    stopping rule is the stationarity certificate
    ``‖∇f(θ̂)‖∞ ≤ grad_tol · ‖∇f(θ₀)‖∞``, recomputed at the returned
    parameters. ``max_evals`` bounds objective-and-gradient evaluations and
    never ends a fit on its own: a run that exhausts it, or whose line search
    stalls, raises :class:`gamfit.errors.FitConvergenceError` carrying the
    reached parameters as a checkpoint.

    Returns ``(encoder, decoder, rss, total_penalty, free_scale,
    stationarity, n_iter)`` at the certified point.
    """

    torch = torch_mod
    nn = torch.nn

    n_obs, p_features = int(x_t.shape[0]), int(x_t.shape[1])
    n_sup = int(n_supervised)
    latent_dim = n_sup + int(n_free)

    gen = torch.Generator(device=x_t.device).manual_seed(int(seed))
    # Module constructors consume the process-global RNG even though every
    # parameter is replaced below from `gen`. Fork that state so this library
    # fit cannot perturb the caller's random stream.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
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
        bound = 1.0 / math.sqrt(latent_dim)
        decoder.weight.uniform_(-bound, bound, generator=gen)

    params = list(encoder.parameters()) + list(decoder.parameters())

    rust = rust_module()

    # Singleton feature groups (one per output feature) give an element-wise
    # smoothed-L1 over the free-latent rows of the decoder, which is the
    # Lachapelle 2401.04890 mechanism-sparsity functional.
    feature_groups = [[j] for j in range(p_features)]
    mech_pen = rust.MechanismSparsityPenalty(
        feature_groups, float(mech_w), float(max(1, n_obs))
    )

    aux_np = np.ascontiguousarray(aux_t.detach().cpu().numpy())
    # Conditional scale σ(u) of the Gaussian iVAE prior, derived from the
    # auxiliary so that the natural-parameter signature spans the 2k
    # dimensions of Khemakhem 2107.10098 Thm. 1 (#576). The prior is built
    # once here, before any optimization: an auxiliary that cannot identify
    # it (constant column, too few distinct states) raises the Rust
    # precondition error to the caller instead of fitting a model without
    # the prior the recipe promises.
    aux_scale = _derive_aux_scale(aux_np)
    rust.conditional_prior_ivae(
        float(aux_w), np.zeros((n_obs, n_sup), dtype=np.float64), aux_np, aux_scale
    )

    def latents() -> tuple[Any, Any, Any]:
        raw = encoder(x_t)
        raw_free = raw[:, n_sup:latent_dim]
        free_scale = torch.sqrt((raw_free * raw_free).mean(dim=0))
        return raw[:, :n_sup], raw_free / free_scale, free_scale

    def terms() -> tuple[Any, Any, Any, float, float]:
        t_sup, t_free, _ = latents()
        x_hat = decoder(torch.cat([t_sup, t_free], dim=1))
        recon = ((x_hat - x_t) ** 2).sum()
        aux_val, aux_grad = rust.conditional_prior_ivae(
            float(aux_w),
            np.ascontiguousarray(t_sup.detach().cpu().numpy()),
            aux_np,
            aux_scale,
        )
        # decoder.weight is (P, latent_dim); the free rows of its transpose
        # are the free latents' mechanisms.
        w_free = decoder.weight.t()[n_sup:latent_dim, :]
        mech_val, mech_grad = mech_pen.value_grad(
            np.ascontiguousarray(w_free.detach().cpu().numpy().astype(np.float64))
        )
        aux_s = (
            t_sup * torch.as_tensor(np.asarray(aux_grad), dtype=t_sup.dtype, device=t_sup.device)
        ).sum()
        mech_s = (
            w_free * torch.as_tensor(np.asarray(mech_grad), dtype=w_free.dtype, device=w_free.device)
        ).sum()
        return recon, aux_s, mech_s, float(aux_val), float(mech_val)

    def objective() -> Any:
        # The Rust penalties supply analytic values and gradients. `s − s.detach()`
        # is exactly zero in value and carries exactly the Rust gradient, so the
        # returned scalar is the true f(θ) the line search compares.
        recon, aux_s, mech_s, aux_val, mech_val = terms()
        return recon + (aux_s - aux_s.detach()) + (mech_s - mech_s.detach()) + (aux_val + mech_val)

    def closure() -> Any:
        optim.zero_grad(set_to_none=False)
        loss = objective()
        loss.backward()
        return loss

    # The stationarity reference ‖∇f(θ₀)‖∞ is the gradient scale of this
    # problem at its seeded start; the certificate is relative to it, as in
    # gaussian_reml_optimize_latent (#954), so it is invariant to the O(N·P)
    # scale of the objective and to additive constants.
    optim = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=int(max_evals),
        max_eval=int(max_evals),
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )
    closure()
    grad_inf_init = _flat_grad_inf(params)
    target = float(grad_tol) * grad_inf_init
    optim.param_groups[0]["tolerance_grad"] = target
    optim.step(closure)
    n_iter = int(optim.state[params[0]].get("n_iter", 0))

    # Certificate, recomputed at the returned parameters independently of the
    # optimizer's own bookkeeping.
    loss = closure()
    grad_inf = _flat_grad_inf(params)
    objective_value = float(loss.detach().cpu().item())
    if not (math.isfinite(grad_inf) and math.isfinite(objective_value) and grad_inf <= target):
        exc = IdentifiableFactorFitConvergenceError(
            "identifiable_factor_fit did not reach a stationary point: "
            f"‖∇f‖∞ = {grad_inf:.3e} > grad_tol·‖∇f(θ₀)‖∞ = "
            f"{float(grad_tol):.1e}·{grad_inf_init:.3e} after {n_iter} L-BFGS "
            f"iterations ({max_evals} objective evaluations allowed). Raise "
            "max_evals or resume from the checkpoint attributes."
        )
        exc.grad_inf = grad_inf
        exc.grad_inf_init = grad_inf_init
        exc.grad_tol = float(grad_tol)
        exc.max_evals = int(max_evals)
        exc.n_iter = n_iter
        exc.objective_value = objective_value
        exc.checkpoint_encoder_state = {
            k: v.detach().cpu().numpy().astype(np.float64).copy()
            for k, v in encoder.state_dict().items()
        }
        exc.checkpoint_decoder = np.ascontiguousarray(
            decoder.weight.detach().cpu().numpy().astype(np.float64)
        )
        raise exc

    with torch.no_grad():
        recon, _, _, aux_val, mech_val = terms()
        _, _, free_scale = latents()
    stationarity = grad_inf / grad_inf_init if grad_inf_init > 0.0 else 0.0
    return (
        encoder,
        decoder,
        float(recon.item()),
        aux_val + mech_val,
        np.ascontiguousarray(free_scale.detach().cpu().numpy().astype(np.float64)),
        float(stationarity),
        n_iter,
    )


def identifiable_factor_fit(
    X: Any,
    aux: Any,
    n_supervised: int,
    n_free: int,
    *,
    mech_sparsity_weight: float | None = None,
    aux_prior_weight: float | None = None,
    encoder: str = "mlp[256, 256]",
    max_evals: int = 5000,
    grad_tol: float = 1.0e-8,
    random_state: int = 0,
    check_identifiability: bool = True,
) -> IdentifiableFactorFitResult:
    """Fit an identifiable factor model combining iVAE + mechanism sparsity.

    The encoder ``E(X) -> (T_sup, T_free)`` produces a real-valued latent
    split. ``T_sup`` is supervised by ``aux`` via an iVAE-style Gaussian
    auxiliary-conditional prior; ``T_free`` is unsupervised, normalized to
    unit per-column second moment, and constrained by a mechanism-sparsity
    penalty on its decoder rows. Both penalty weights default to the
    calibrated recipe weights owned by Rust. The resulting single fit is
    scored by its Gaussian profile log-likelihood at the fitted weights,
    which is not a marginal likelihood.

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
    mech_sparsity_weight, aux_prior_weight : positive float or ``None``
        Penalty weights. ``None`` takes the Rust recipe weights
        ``mech_sparsity_weight=1e-4`` and ``aux_prior_weight=2.0``.
    encoder : str
        ``"linear"`` for a single-Linear encoder, or ``"mlp[w1, w2, ...]"``
        for an MLP of widths ``w_i`` with GELU activations and a Linear
        head onto the latent dim.
    max_evals : int
        Budget of objective-and-gradient evaluations for L-BFGS. It is a
        budget only: the fit stops when the stationarity certificate holds.
    grad_tol : float
        Relative stationarity the fit must certify,
        ``‖∇f(θ̂)‖∞ ≤ grad_tol · ‖∇f(θ₀)‖∞`` over every encoder and decoder
        parameter, with the same meaning as in
        :func:`gamfit.gaussian_reml_optimize_latent`.
    random_state : int
        Seed of the parameter initialization. The caller's global torch RNG
        is left untouched.

    Returns
    -------
    :class:`IdentifiableFactorFitResult`
        Fitted latents at the certified point, profile log-likelihood,
        decoder, final weights, the achieved ``stationarity`` and ``n_iter``,
        and the identifiability report.

    Raises
    ------
    gamfit.errors.FitConvergenceError
        The certificate did not hold within ``max_evals`` evaluations. The
        exception carries ``grad_inf``, ``grad_inf_init``, ``grad_tol``,
        ``max_evals``, ``n_iter``, ``objective_value``,
        ``checkpoint_encoder_state`` and ``checkpoint_decoder``.
    ValueError
        ``aux`` cannot identify the iVAE conditional prior (Khemakhem
        2107.10098 Thm. 1: a constant column, or fewer than ``2k + 1``
        distinct auxiliary states). Raised before any optimization.

    Notes
    -----
    ``profile_log_likelihood`` is, up to an additive constant, the Gaussian
    log-likelihood of this fixed-weight fit with the noise scale profiled
    out, minus half the penalty; *higher is better*. It has no
    log-determinant or Occam term, so it is not a marginal likelihood or
    evidence and does not price model complexity. REML wiring for arbitrary
    custom torch encoders is not yet plumbed through the Rust engine, so the
    default weights are calibrated recipe weights rather than the output of
    a REML selector like the one :func:`gamfit.fit` uses for formula-based
    smooths.
    """

    x_np, aux_np = _validate_inputs(X, aux, int(n_supervised), int(n_free))
    hidden_widths = _parse_encoder_spec(encoder)
    if int(max_evals) < 1:
        raise ValueError(f"max_evals must be >= 1; got {max_evals}")
    if not (math.isfinite(float(grad_tol)) and float(grad_tol) > 0.0):
        raise ValueError(f"grad_tol must be finite and > 0; got {grad_tol}")

    try:
        import torch as torch_mod
    except ImportError as exc:  # pragma: no cover - torch is a required extra
        raise ImportError(
            "identifiable_factor_fit requires PyTorch; install with `pip install torch`"
        ) from exc

    aux_w, mech_w = rust_module().identifiable_factor_weights(
        aux_prior_weight, mech_sparsity_weight
    )

    x_t = torch_mod.as_tensor(x_np, dtype=torch_mod.float64)
    aux_t = torch_mod.as_tensor(aux_np, dtype=torch_mod.float64)

    (
        encoder_module,
        decoder_module,
        rss_val,
        pen_val,
        free_scale,
        stationarity,
        n_iter,
    ) = _one_fit(
        x_t, aux_t, int(n_supervised), int(n_free), hidden_widths,
        aux_w, mech_w, int(max_evals), float(grad_tol),
        int(random_state), torch_mod,
    )
    # Score this one certified fixed-weight fit. The Rust boundary is scalar
    # on purpose: a 1x1 "grid" is not hyperparameter selection, and sampled
    # surfaces cannot certify a continuous two-log-weight optimum.
    profile_log_likelihood = float(
        rust_module().identifiable_factor_profile_log_likelihood(
            float(rss_val),
            float(pen_val),
            int(x_t.shape[0]),
        )
    )

    n_sup = int(n_supervised)
    with torch_mod.no_grad():
        t = encoder_module(x_t).detach().cpu().numpy().astype(np.float64)
        t_sup_np = np.ascontiguousarray(t[:, :n_sup])
        t_free_np = np.ascontiguousarray(t[:, n_sup : n_sup + int(n_free)] / free_scale)
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
        free_scale=free_scale,
        profile_log_likelihood=float(profile_log_likelihood),
        decoder=decoder_w,
        aux_prior_weight=float(aux_w),
        mech_sparsity_weight=float(mech_w),
        encoder_state=encoder_state,
        stationarity=float(stationarity),
        n_iter=int(n_iter),
        aux=aux_np,
        report=None,
    )

    if bool(check_identifiability):
        report = check(result)
        result.report = report
        for msg in report.as_warnings():
            warnings.warn(msg, UserWarning, stacklevel=2)

    return result
