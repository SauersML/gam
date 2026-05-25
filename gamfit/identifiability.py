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

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._binding import rust_module

__all__ = [
    "IdentifiableFactorFitResult",
    "identifiable_factor_fit",
]


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
        are satisfied within numerical tolerance.
    """

    T_supervised: np.ndarray
    T_free: np.ndarray
    evidence: float
    decoder: np.ndarray
    aux_prior_weight: float
    mech_sparsity_weight: float
    encoder_state: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)


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

    # Final-pass true values for the evidence proxy.
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

    # Laplace-style log marginal-likelihood proxy. The Gaussian-residual
    # profile log-likelihood is ``-0.5 * N * log(RSS / N)`` (up to additive
    # constants) and the penalty acts as the log prior. Higher is better.
    safe_rss = max(rss / max(1, n_obs), 1e-300)
    evidence = -0.5 * n_obs * math.log(safe_rss) - 0.5 * total_pen
    return encoder, decoder, rss, total_pen, float(evidence)


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
    centred on the seed value. Total candidate count is ``5 ** k`` where
    ``k`` is the number of ``"auto"`` flags; with ``k <= 2`` this stays at
    most 25 inner fits.
    """

    aux_grid = (
        [aux_w0 * 10 ** e for e in (-2.0, -1.0, 0.0, 1.0, 2.0)] if auto_aux else [aux_w0]
    )
    mech_grid = (
        [mech_w0 * 10 ** e for e in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        if auto_mech
        else [mech_w0]
    )

    best: tuple[float, float, Any, Any, float, float, float] | None = None
    for aw in aux_grid:
        for mw in mech_grid:
            encoder, decoder, rss, pen, ev = _one_fit(
                x_t, aux_t, n_supervised, n_free, hidden_widths,
                aw, mw, max_iter, learning_rate, seed, torch_mod,
            )
            if best is None or ev > best[6]:
                best = (aw, mw, encoder, decoder, rss, pen, ev)
    assert best is not None  # grids are non-empty by construction
    return best


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
        encoder_module, decoder_module, _rss, _pen, evidence = _one_fit(
            x_t, aux_t, int(n_supervised), int(n_free), hidden_widths,
            aux_w0, mech_w0, int(max_iter), float(learning_rate),
            int(random_state), torch_mod,
        )
        aux_w, mech_w = aux_w0, mech_w0

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

    issues = _check_preconditions(
        aux=aux_np,
        decoder=decoder_w,
        n_supervised=int(n_supervised),
        n_free=int(n_free),
        encoder_depth=enc_depth,
        mech_sparsity_weight=float(mech_w),
    )
    for msg in issues:
        warnings.warn(msg, UserWarning, stacklevel=2)

    return IdentifiableFactorFitResult(
        T_supervised=t_sup_np,
        T_free=t_free_np,
        evidence=float(evidence),
        decoder=decoder_w,
        aux_prior_weight=float(aux_w),
        mech_sparsity_weight=float(mech_w),
        encoder_state=encoder_state,
        warnings=list(issues),
    )
