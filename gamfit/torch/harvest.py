"""WP-D — the harvest contract: per-token output-Fisher factors from a model.

This module is the *torch end* of the output-Fisher pullback metric (#980). It
harvests, for each token, the exact low-rank input shard that gam's
``RowMetric::OutputFisher`` consumes:

* ``x_n`` — the activation at a chosen hook site, ``x_n ∈ ℝ^p``;
* ``U_n`` — the **top-r factors** of the pullback ``G_n = J_nᵀ F J_n``, where
  ``J_n = ∂logits/∂x_n`` is the model's output Jacobian *at that activation* and
  ``F = diag(p) − p pᵀ`` is the softmax (output) Fisher. ``U_n ∈ ℝ^{p × r}`` is
  scaled so that ``U_n U_nᵀ`` reconstructs the rank-r truncation of ``G_n``
  (column ``k`` is ``√λ_k · v_k``). This is exactly the factor convention
  ``RowMetric::output_fisher`` expects: it forms ``W_n = U_n U_nᵀ`` directly;
* ``mass_residual`` — an optional estimate of
  ``trace(G_n) − Σ_{k≤r} λ_k``. It is a scalar diagnostic, never an operator
  certificate and never evidence that a randomized factor is a PSD lower bound.

Everything is matrix-free. The Jacobian ``J_n`` (``C × p``) and the pullback
``G_n`` (``p × p``) are **never materialized**. The only primitive used is the
pullback matvec ``v ↦ G_n v = J_nᵀ (F (J_n v))``, assembled from one
JVP (``J_n v``), a Fisher-apply (``F u = p ⊙ u − p (pᵀu)``), and one VJP
(``J_nᵀ w``). The top-r eigenpairs come from subspace iteration + a small
``m × m`` Rayleigh–Ritz eigendecomposition (``m = r + oversample``); ``trace``
is a matrix-free Hutchinson estimate using the same matvec. No ``C × p`` or
``p × p`` object is ever allocated in the harvest loop.

**Rung 1 — the sketch that enters the reconstruction loss.**
:func:`harvest_behavioral_fisher_probes` emits the *same* pullback
``G_n = J_nᵀ F_n J_n`` in a cheaper, likelihood-weighting form: ``s`` random
probes ``vᵢ = J_nᵀ F_n^{1/2} uᵢ`` whose outer-product sum ``Σᵢ vᵢ vᵢᵀ`` is an
**unbiased sketch** of ``G_n`` — one VJP per probe, no JVP and no eigensolve, so
``s`` backward passes per token. Its shard carries ``provenance="behavioral_fisher"``
and routes to ``RowMetric::behavioral_fisher``, which prices the reconstruction
residual as ``½ eᵀ G_n e`` (nats, generalized least squares) — the only Fisher
provenance that whitens the likelihood. The eigenfactor harvests above stay
gauge-only (they never touch the loss).

Policy note: heavy gam math stays in the Rust core, but *harvesting from a torch
model* is the sanctioned torch-interop path — these are torch autograd ops on a
user-supplied model, not a reimplementation of any gam primitive. The shard is
written f32 (factors are inherently low-precision) and promoted to f64 only at
the gam boundary (``load_harvest_shard`` returns f64 arrays).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

__all__ = [
    "HarvestShard",
    "harvest_output_fisher_factors",
    "harvest_downstream_output_fisher_factors",
    "harvest_behavioral_fisher_probes",
    "save_harvest_shard",
    "load_harvest_shard",
]

# The provenance tags a harvest shard may carry:
#   * ``output_fisher`` / ``output_fisher_downstream`` — the gauge-only #980
#     metrics (top-r eigenfactor form; never whiten the likelihood); and
#   * ``behavioral_fisher`` — the **Rung 1** s-probe sketch of the same
#     output-Fisher ``G_n = J_nᵀ F_n J_n``, installed as the reconstruction
#     *likelihood weight* (GLS in nats). Its factor columns are the raw random
#     probes ``vᵢ = J_nᵀ F_n^{1/2} uᵢ`` (not eigenvectors); it maps onto the
#     gam-side ``RowMetric::behavioral_fisher``, the only Fisher provenance that
#     whitens the likelihood.
# Each maps one-to-one onto a gam-side ``RowMetric`` constructor.
_HARVEST_PROVENANCES = frozenset(
    {"output_fisher", "output_fisher_downstream", "behavioral_fisher"}
)
_HARVEST_SCHEMA = "gamfit.FisherHarvest/v1"
_FISHER_FACTOR_KINDS = frozenset(
    {"exact_full", "certified_psd_lower_bound", "uncertified_approximation"}
)

_ATTENTION_FORWARD_AD_MARKERS = (
    "scaled_dot_product",
    "flash_attention",
    "efficient_attention",
    "memory_efficient_attention",
    "sdpa",
)


def _jvp_with_attention_diagnostic(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
    tangent: torch.Tensor,
    *,
    harvest_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one JVP, translating only attention forward-AD failures.

    PyTorch attention backends may implement reverse AD while lacking the JVP
    needed by the matrix-free top-eigenfactor harvest. Mutating a loaded model's
    attention configuration is not a valid backend switch, and substituting the
    VJP-only behavioral sketch would change the statistical contract. Preserve
    every unrelated torch error verbatim; only a failing attention operator at
    this JVP boundary receives the actionable eager-attention diagnosis.
    """
    try:
        return torch.func.jvp(function, (primal,), (tangent,))
    except (RuntimeError, NotImplementedError) as error:
        message = str(error).lower()
        if any(marker in message for marker in _ATTENTION_FORWARD_AD_MARKERS):
            raise RuntimeError(
                f"{harvest_name} requires torch forward-mode AD through the "
                "downstream model, but the active SDPA/flash-attention operator "
                "has no JVP. Reload the model with "
                'attn_implementation="eager" (for Hugging Face: '
                'from_pretrained(..., attn_implementation="eager")) before '
                "harvesting. gamfit will not mutate the attention backend or "
                "replace this top-eigenfactor contract with a different Fisher "
                f"estimator. Original torch error: {error}"
            ) from error
        raise


# ---------------------------------------------------------------------------
# Shard container — factors plus explicit scientific status
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarvestShard:
    """The exact input shard ``RowMetric::OutputFisher`` consumes.

    Attributes
    ----------
    X
        ``(n, p)`` activations at the hook site, one row per harvested token.
    U
        ``(n, p, r)`` per-row output-Fisher factors. ``U[n]`` is ``U_n`` with
        ``U_n U_nᵀ`` equal to the rank-r truncation of ``G_n = J_nᵀ F J_n``;
        column ``k`` is ``√λ_k · v_k`` for the ``k``-th largest eigenpair.
        Flattened row-major this matches the gam layout
        ``u[n, i * r + k] = U[n, i, k]``.
    mass_residual
        Optional ``(n,)`` tail-trace diagnostic. It is not a certificate and is
        absent for exact factors. A certified PSD lower bound must carry it.
    rank
        ``r``, the number of factors per row (``U.shape[2]``).
    provenance
        Which output-Fisher pullback produced ``U`` (#980). ``"output_fisher"``
        (the default) is the same-position metric ``G_n = J_nᵀ F_n J_n`` that
        ``RowMetric::output_fisher`` consumes; ``"output_fisher_downstream"`` is
        the forward-looking metric ``G_n = Σ_{t≥n} J_{t←n}ᵀ F_t J_{t←n}``
        aggregated over the future positions ``n`` reaches through the KV path
        (mechanism 2 of the #980 revision), consumed by
        ``RowMetric::output_fisher_downstream``. The factor *layout* is identical
        across both; only the tag (and the science it certifies) differs, so the
        gauge/lens/enrichment machinery is provenance-generic and consumes either
        unchanged. Any other value is rejected.
    factor_kind
        Required operator status. It is never inferred from rank or from a
        scalar tail estimate: randomized/Ritz and probe factors are
        ``"uncertified_approximation"``; a complete factorization is
        ``"exact_full"``; ``"certified_psd_lower_bound"`` is reserved for a
        producer that proves the Loewner-order relation.
    """

    X: Any
    U: Any
    mass_residual: Any
    rank: int
    factor_kind: str
    provenance: str = "output_fisher"

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        U = np.asarray(self.U)
        mr = None if self.mass_residual is None else np.asarray(self.mass_residual)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, p); got shape {X.shape}")
        n, p = X.shape
        if isinstance(self.rank, (bool, np.bool_)) or not isinstance(
            self.rank, (int, np.integer)
        ):
            raise TypeError("rank must be an integer")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1; got {self.rank}")
        if U.shape != (n, p, self.rank):
            raise ValueError(
                f"U must be (n, p, r) = ({n}, {p}, {self.rank}); got shape {U.shape}"
            )
        if mr is not None and mr.shape != (n,):
            raise ValueError(f"mass_residual must be (n,) = ({n},); got shape {mr.shape}")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(U)):
            raise ValueError("X and U must contain only finite values")
        if mr is not None and (not np.all(np.isfinite(mr)) or np.any(mr < 0.0)):
            raise ValueError("mass_residual must contain finite non-negative values")
        if self.factor_kind not in _FISHER_FACTOR_KINDS:
            raise ValueError(
                f"factor_kind must be one of {sorted(_FISHER_FACTOR_KINDS)}; "
                f"got {self.factor_kind!r}"
            )
        if self.factor_kind == "exact_full" and (self.rank != p or mr is not None):
            raise ValueError(
                "exact_full requires rank == activation dimension and forbids "
                "mass_residual"
            )
        if self.factor_kind == "certified_psd_lower_bound" and mr is None:
            raise ValueError("certified_psd_lower_bound requires mass_residual")
        if self.provenance not in _HARVEST_PROVENANCES:
            raise ValueError(
                f"provenance must be one of {sorted(_HARVEST_PROVENANCES)}; "
                f"got {self.provenance!r}"
            )


# ---------------------------------------------------------------------------
# Matrix-free output-Fisher pullback matvec
# ---------------------------------------------------------------------------


def _softmax_fisher_apply(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Apply the softmax Fisher ``F = diag(p) − p pᵀ`` to ``u``, matrix-free.

    ``probs`` is ``(..., C)`` softmax probabilities; ``u`` is ``(..., C)``. The
    ``C × C`` matrix ``F`` is never formed: ``F u = p ⊙ u − p (pᵀ u)``.
    """
    weighted = probs * u
    inner = (probs * u).sum(dim=-1, keepdim=True)
    return weighted - probs * inner


def _sample_output_fisher_probes(
    probs: torch.Tensor,
    s: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Draw ``s`` iid samples ``u ~ N(0, F)`` of the softmax output Fisher.

    ``F = diag(p) − p pᵀ`` is the categorical Fisher. A sample is realized
    matrix-free from ``g ~ N(0, I_C)`` via ``u = √p ⊙ g − (√p · g) p``: with
    ``a = √p`` (unit vector, ``Σ p = 1``) and ``D = diag(√p)`` this is
    ``u = D(g − (aᵀg)a)``, whose covariance is
    ``D(I − aaᵀ)Dᵀ = diag(p) − p pᵀ = F`` exactly. So ``E[u uᵀ] = F`` and the
    probe carries the Fisher's directionality without any ``C × C`` matrix or a
    matrix square root. Returns a ``(C, s)`` block, one probe per column.

    ``g`` is sampled on the generator's CPU device then moved, so a fixed seed
    yields identical probes on CPU and GPU.
    """
    c = probs.shape[-1]
    g = torch.randn(c, s, generator=generator, dtype=dtype).to(device)  # (C, s)
    sqrt_p = probs.clamp_min(0.0).sqrt().to(dtype)  # (C,)
    dots = (sqrt_p.unsqueeze(1) * g).sum(dim=0)  # (s,) = √pᵀ g per probe
    return sqrt_p.unsqueeze(1) * g - probs.to(dtype).unsqueeze(1) * dots.unsqueeze(0)


def _pullback_matvec(
    jvp_fn: Callable[[torch.Tensor], torch.Tensor],
    vjp_fn: Callable[[torch.Tensor], torch.Tensor],
    probs: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """``V ↦ G V`` for ``G = Jᵀ F J`` applied to a stack of ``m`` directions.

    ``V`` is ``(p, m)``. Returns ``(p, m)``. Internally: ``J V`` is ``(C, m)``
    via a vectorized JVP, ``F (J V)`` is the matrix-free Fisher-apply, and
    ``Jᵀ (F J V)`` is a vectorized VJP. No ``C × p`` or ``p × p`` matrix exists.
    """
    # J V : (C, m) — one JVP per column, vectorized over the trailing axis.
    jv = jvp_fn(V)  # (C, m)
    # F (J V): Fisher contracts the C-axis independently per column. Put C last
    # ((m, C)), broadcast probs (C,) over the m rows, apply, then restore (C, m).
    fjv = _softmax_fisher_apply(probs, jv.transpose(0, 1)).transpose(0, 1)  # (C, m)
    # Jᵀ (F J V) : (p, m) — one VJP per column.
    return vjp_fn(fjv)  # (p, m)


def _orthonormalize(M: torch.Tensor) -> torch.Tensor:
    """Thin QR returning an orthonormal ``(p, m)`` basis for ``range(M)``."""
    q, _ = torch.linalg.qr(M, mode="reduced")
    return q


def _top_r_eigenpairs(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    r: int,
    *,
    oversample: int,
    n_iter: int,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-r eigenpairs of a PSD operator given only its matvec ``V ↦ G V``.

    Randomized subspace iteration with ``m = min(p, r + oversample)`` columns,
    ``n_iter`` power steps, then a Rayleigh–Ritz step: the only dense linear
    algebra is an ``m × m`` symmetric eig (``m`` small, never ``p × p``).

    Returns ``(eigvals, eigvecs)`` with ``eigvals`` ``(r,)`` descending and
    ``eigvecs`` ``(p, r)`` orthonormal (the leading-r Ritz vectors).
    """
    m = min(p, r + oversample)
    # Sample on the generator's device (CPU) then move to the operator device,
    # so the fixed seed gives identical bases on CPU and GPU.
    Q = torch.randn(p, m, generator=generator, dtype=dtype).to(device)
    Q = _orthonormalize(Q)
    for _ in range(n_iter):
        Q = _orthonormalize(matvec(Q))
    # Rayleigh–Ritz on the captured subspace: T = Qᵀ G Q is (m, m).
    GQ = matvec(Q)
    T = Q.transpose(0, 1) @ GQ
    T = 0.5 * (T + T.transpose(0, 1))  # symmetrize against round-off
    evals, evecs = torch.linalg.eigh(T)  # ascending
    # Descending, take leading r.
    order = torch.argsort(evals, descending=True)
    top = order[:r]
    ritz_vals = evals[top].clamp_min(0.0)  # PSD: clamp tiny negative round-off
    ritz_vecs = Q @ evecs[:, top]  # (p, r)
    return ritz_vals, ritz_vecs


def _trace_estimate(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    p: int,
    *,
    n_probes: int,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Matrix-free ``trace(G)`` from the matvec.

    Exact when ``n_probes >= p``: probe with the standard basis ``I_p`` so
    ``Σ_i e_iᵀ G e_i = trace(G)`` exactly (still only matvecs, no ``p × p``).
    Otherwise a Rademacher Hutchinson estimate ``E[zᵀ G z] = trace(G)``. Probes
    are batched as a ``(p, n_probes)`` block through a single matvec call.
    """
    if n_probes >= p:
        # Exact: identity-basis probes, no randomness, fully deterministic.
        Z = torch.eye(p, dtype=dtype, device=device)
        GZ = matvec(Z)  # (p, p)
        return (Z * GZ).sum()  # = trace(G) exactly
    n_probes = max(1, n_probes)
    # Sample Rademacher ±1 on the generator's device (CPU), then move.
    Z = (
        torch.randint(0, 2, (p, n_probes), generator=generator, dtype=torch.int64)
        .to(dtype)
        * 2
        - 1
    ).to(device)
    GZ = matvec(Z)  # (p, n_probes)
    quad = (Z * GZ).sum(dim=0)  # (n_probes,)
    return quad.mean()


# ---------------------------------------------------------------------------
# Hook-site activation capture
# ---------------------------------------------------------------------------


def _capture_activations(
    model: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: Any,
) -> tuple[torch.Tensor, Callable[[torch.Tensor, int], torch.Tensor]]:
    """Run ``model(inputs)`` capturing the output of ``hook_module``.

    Returns ``(acts, logits_from_acts)`` where ``acts`` is the detached hook-site
    activation tensor ``(n, p)`` (flattened over any leading token/batch axes)
    and ``logits_from_acts`` recomputes logits as a differentiable function of a
    *single* hook-site activation row ``ℝ^p → ℝ^C`` (the local map whose
    Jacobian is ``J_n``). The recompute splices a candidate activation back in at
    the hook site for one forward pass, so ``J_n`` is the true end-to-end
    ``∂logits/∂x_n`` through the rest of the network.
    """
    captured: dict[str, torch.Tensor] = {}

    def _grab(_mod: torch.nn.Module, _inp: Any, out: torch.Tensor) -> None:
        captured["act"] = out

    handle = hook_module.register_forward_hook(_grab)
    try:
        with torch.no_grad():
            model(inputs)
        act = captured["act"]
    finally:
        handle.remove()

    # Flatten leading axes to a token list: act (..., p) -> (n, p).
    act_flat = act.reshape(-1, act.shape[-1]).detach()
    feature_shape = act.shape

    def logits_from_act(single_row: torch.Tensor, row_index: int) -> torch.Tensor:
        """Logits as a function of one spliced-in activation row ``ℝ^p → ℝ^C``."""
        replacement = {"value": None, "index": row_index, "shape": feature_shape}

        def _splice(_mod: torch.nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor:
            flat = out.reshape(-1, out.shape[-1])
            rows = [flat[i] for i in range(flat.shape[0])]
            rows[replacement["index"]] = single_row
            new_flat = torch.stack(rows, dim=0)
            return new_flat.reshape(out.shape)

        h = hook_module.register_forward_hook(_splice)
        try:
            out_logits = model(inputs)
        finally:
            h.remove()
        return out_logits.reshape(-1, out_logits.shape[-1])[row_index]

    return act_flat, logits_from_act


def _capture_activations_downstream(
    model: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: Any,
) -> tuple[torch.Tensor, Callable[[torch.Tensor, int], torch.Tensor]]:
    """Like :func:`_capture_activations`, but the recompute returns the logits at
    **every** token position as a differentiable function of one spliced-in row.

    Returns ``(acts, logits_all_from_act)`` where ``acts`` is ``(n, p)`` and
    ``logits_all_from_act(x_n, n) -> (n_pos, C)`` is the full flattened logit
    block when activation row ``n`` is replaced by ``x_n``. The downstream
    harvest reads rows ``t ≥ n`` of this block: those are exactly the future
    positions the residual stream at ``n`` reaches through the KV path, so
    ``∂logits_t/∂x_n`` for ``t ≥ n`` is the forward-looking Jacobian the #980
    downstream metric aggregates. (For ``t < n`` causal attention makes the
    Jacobian zero; the harvest never probes them.)
    """
    captured: dict[str, torch.Tensor] = {}

    def _grab(_mod: torch.nn.Module, _inp: Any, out: torch.Tensor) -> None:
        captured["act"] = out

    handle = hook_module.register_forward_hook(_grab)
    try:
        with torch.no_grad():
            model(inputs)
        act = captured["act"]
    finally:
        handle.remove()

    act_flat = act.reshape(-1, act.shape[-1]).detach()
    feature_shape = act.shape

    def logits_all_from_act(single_row: torch.Tensor, row_index: int) -> torch.Tensor:
        """Full ``(n_pos, C)`` logit block as a function of one spliced row."""
        replacement = {"index": row_index, "shape": feature_shape}

        def _splice(_mod: torch.nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor:
            flat = out.reshape(-1, out.shape[-1])
            rows = [flat[i] for i in range(flat.shape[0])]
            # Cast into the hook site's dtype: probes work in f32/f64 while the
            # model may run bf16, and torch.stack rejects mixed dtypes. Autograd
            # differentiates through the cast, so the VJP still lands in the
            # probe's working dtype.
            rows[replacement["index"]] = single_row.to(dtype=flat.dtype)
            new_flat = torch.stack(rows, dim=0)
            return new_flat.reshape(out.shape)

        h = hook_module.register_forward_hook(_splice)
        try:
            out_logits = model(inputs)
        finally:
            h.remove()
        return out_logits.reshape(-1, out_logits.shape[-1])

    return act_flat, logits_all_from_act


# ---------------------------------------------------------------------------
# Public harvest entry point
# ---------------------------------------------------------------------------


def harvest_output_fisher_factors(
    model: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: Any,
    *,
    rank: int,
    oversample: int = 4,
    n_iter: int = 2,
    trace_probes: int = 8,
    seed: int = 0,
) -> HarvestShard:
    """Harvest per-token output-Fisher factors at ``hook_module``.

    Parameters
    ----------
    model
        The torch model. Called as ``model(inputs)`` and assumed to return
        logits of shape ``(..., C)`` (the leading axes are flattened to tokens).
        The downstream stack must support forward-mode AD. Hugging Face models
        using SDPA/flash attention must be loaded with
        ``attn_implementation="eager"`` for this JVP-based harvest.
    hook_module
        A submodule of ``model`` whose *output* is the hook-site activation
        ``x_n``. Its forward output's last axis is the activation dimension ``p``.
    inputs
        Whatever ``model.forward`` accepts (e.g. token ids). Harvesting is done
        for every token row produced at the hook site.
    rank
        ``r``, the number of output-Fisher factors retained per row.
    oversample, n_iter
        Subspace-iteration controls. ``m = min(p, r + oversample)`` subspace
        width and ``n_iter`` power steps before the Rayleigh–Ritz eig.
    trace_probes
        Number of Hutchinson probes for the ``trace(G_n)`` estimate that feeds
        ``mass_residual``.
    seed
        Fixed RNG seed for the randomized subspace + Hutchinson probes — fully
        deterministic, no clock entropy.

    Returns
    -------
    HarvestShard
        ``X (n, p)``, ``U (n, p, r)``, ``mass_residual (n,)`` — the exact shard
        ``RowMetric::output_fisher`` consumes. Factors are computed in the
        model's working dtype/device and returned as f32 numpy on CPU; f64
        promotion happens at the gam boundary in :func:`load_harvest_shard`.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1; got {rank}")

    act_flat, logits_from_act = _capture_activations(model, hook_module, inputs)
    n, p = int(act_flat.shape[0]), int(act_flat.shape[1])
    if rank > p:
        raise ValueError(f"rank {rank} exceeds activation dimension p = {p}")
    device = act_flat.device
    # Work in the activation dtype but force at least f32 for the eig.
    work_dtype = act_flat.dtype if act_flat.dtype in (torch.float32, torch.float64) else torch.float32

    U = np.empty((n, p, rank), dtype=np.float32)
    X = act_flat.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    mass_residual = np.empty((n,), dtype=np.float32)

    for row in range(n):
        x_row = act_flat[row].to(work_dtype).detach().requires_grad_(False)

        # Local map f_row : ℝ^p → ℝ^C whose Jacobian is J_n.
        def f_row(x: torch.Tensor, _row: int = row) -> torch.Tensor:
            return logits_from_act(x, _row).to(work_dtype)

        # Softmax probabilities at this token (for the Fisher F = diag(p) − p pᵀ).
        with torch.no_grad():
            probs = torch.softmax(f_row(x_row), dim=-1)  # (C,)

        # JVP: V (p, m) -> J_n V (C, m). One torch.func.jvp per column; J_n is
        # never formed. Columns are looped explicitly (m is small: r+oversample
        # or a handful of trace probes) — robust through the model's forward
        # hooks, which vmap would have to trace through.
        def jvp_fn(V: torch.Tensor, _f=f_row, _x=x_row) -> torch.Tensor:
            cols = []
            for j in range(V.shape[1]):
                _out, jv = _jvp_with_attention_diagnostic(
                    _f,
                    _x,
                    V[:, j].contiguous(),
                    harvest_name="harvest_output_fisher_factors",
                )
                cols.append(jv)
            return torch.stack(cols, dim=1)  # (C, m)

        # VJP: W (C, m) -> J_nᵀ W (p, m). Build the pullback closure once per row.
        _out0, vjp_raw = torch.func.vjp(f_row, x_row)

        def vjp_fn(W: torch.Tensor, _vjp=vjp_raw) -> torch.Tensor:
            cols = []
            for j in range(W.shape[1]):
                (gx,) = _vjp(W[:, j].contiguous())  # (p,)
                cols.append(gx)
            return torch.stack(cols, dim=1)  # (p, m)

        def matvec(V: torch.Tensor, _j=jvp_fn, _vj=vjp_fn, _p=probs) -> torch.Tensor:
            return _pullback_matvec(_j, _vj, _p, V)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + row)
        # Generators sample on CPU (randn/randint are CPU-portable), then the
        # basis is moved onto the activation device so the seed is honored
        # identically regardless of where the model lives.
        evals, evecs = _top_r_eigenpairs(
            matvec,
            p,
            rank,
            oversample=oversample,
            n_iter=n_iter,
            generator=gen,
            dtype=work_dtype,
            device=device,
        )
        # Factors: column k = sqrt(λ_k) · v_k, so U_n U_nᵀ = Σ_k λ_k v_k v_kᵀ.
        scaled = evecs * evals.clamp_min(0.0).sqrt().unsqueeze(0)  # (p, r)

        gen_tr = torch.Generator(device="cpu")
        gen_tr.manual_seed(seed + 10_000 + row)
        trace = _trace_estimate(
            matvec,
            p,
            n_probes=trace_probes,
            generator=gen_tr,
            dtype=work_dtype,
            device=device,
        )
        residual = float(trace.item() - float(evals.sum().item()))
        # PSD ⇒ residual ≥ 0; clamp tiny Hutchinson noise that dips below.
        mass_residual[row] = max(residual, 0.0)
        U[row] = scaled.detach().to(torch.float32).cpu().numpy()

    factor_kind = "exact_full" if rank == p else "uncertified_approximation"
    return HarvestShard(
        X=X,
        U=U,
        mass_residual=None if factor_kind == "exact_full" else mass_residual,
        rank=rank,
        factor_kind=factor_kind,
    )


def harvest_behavioral_fisher_probes(
    model: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: Any,
    *,
    probes: int,
    seed: int = 0,
) -> HarvestShard:
    """Harvest the **Rung 1** s-probe output-Fisher sketch at ``hook_module``.

    This is the cheap, likelihood-weighting cousin of
    :func:`harvest_output_fisher_factors`. Instead of the top-r eigenfactors of
    ``G_n = J_nᵀ F_n J_n`` (a JVP+VJP subspace iteration plus an eig per row), it
    emits ``s = probes`` **raw random probes** ``vᵢ = J_nᵀ F_n^{1/2} uᵢ`` whose
    outer-product sum is an unbiased sketch of ``G_n``:

        ``Σᵢ vᵢ vᵢᵀ = J_nᵀ F_n^{1/2} (Σᵢ uᵢ uᵢᵀ) F_n^{1/2} J_n ≈ J_nᵀ F_n J_n``

    because ``uᵢ ~ N(0, I)`` gives ``E[Σ uᵢ uᵢᵀ] = s·I`` — with the ``1/√s``
    column scaling below, ``E[Σᵢ vᵢ vᵢᵀ] = G_n``. The probe
    ``F_n^{1/2} uᵢ`` is realized directly as a draw from ``N(0, F_n)`` (see
    :func:`_sample_output_fisher_probes`), so each probe costs exactly **one VJP**
    (``J_nᵀ ·``) and **no JVP and no eigensolve**: ``s`` backward passes per token,
    the harvest-time cost the Rung 1 spec promises.

    The columns of ``U_n = [v₁/√s … v_s/√s] ∈ ℝ^{p × s}`` are consumed verbatim
    by ``RowMetric::behavioral_fisher``: it forms ``M_n = U_n U_nᵀ ≈ G_n`` and
    prices the reconstruction residual as ``½ eᵀ M_n e`` (nats), the generalized
    least-squares data-fit. This is a stochastic operator approximation, not a
    truncation with a certified tail, so it carries
    ``factor_kind="uncertified_approximation"`` and no ``mass_residual``.

    Parameters
    ----------
    model, hook_module, inputs
        As in :func:`harvest_output_fisher_factors`.
    probes
        ``s``, the number of random Fisher probes per row (the factor rank of the
        emitted metric). ``4…16`` suffices for the reconstruction weighting.
    seed
        Fixed RNG seed; per-row streams are ``seed + row`` — fully deterministic.
    """
    if probes < 1:
        raise ValueError(f"probes must be >= 1; got {probes}")

    act_flat, logits_from_act = _capture_activations(model, hook_module, inputs)
    n, p = int(act_flat.shape[0]), int(act_flat.shape[1])
    device = act_flat.device
    work_dtype = (
        act_flat.dtype
        if act_flat.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    inv_sqrt_s = 1.0 / math.sqrt(float(probes))

    U = np.empty((n, p, probes), dtype=np.float32)
    X = act_flat.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    for row in range(n):
        x_row = act_flat[row].to(work_dtype).detach().requires_grad_(False)

        def f_row(x: torch.Tensor, _row: int = row) -> torch.Tensor:
            return logits_from_act(x, _row).to(work_dtype)

        # One forward builds the VJP closure; its primal output IS the logits,
        # so the softmax probs come for free (no extra forward pass).
        logits_row, vjp_raw = torch.func.vjp(f_row, x_row)
        with torch.no_grad():
            probs_row = torch.softmax(logits_row, dim=-1)  # (C,)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + row)
        # (C, s): each column uᵢ ~ N(0, F_n) = F_n^{1/2} · (iid normal).
        u_block = _sample_output_fisher_probes(
            probs_row, probes, generator=gen, dtype=work_dtype, device=device
        )

        # vᵢ = J_nᵀ uᵢ — one VJP per probe (s backward passes), no JVP formed.
        cols = []
        for j in range(probes):
            (gx,) = vjp_raw(u_block[:, j].contiguous())  # (p,)
            cols.append(gx)
        v = torch.stack(cols, dim=1) * inv_sqrt_s  # (p, s), M_n = v vᵀ ≈ G_n

        U[row] = v.detach().to(torch.float32).cpu().numpy()

    return HarvestShard(
        X=X,
        U=U,
        mass_residual=None,
        rank=probes,
        factor_kind="uncertified_approximation",
        provenance="behavioral_fisher",
    )


def _downstream_pullback_matvec(
    jvp_fn: Callable[[torch.Tensor], torch.Tensor],
    vjp_fn: Callable[[torch.Tensor], torch.Tensor],
    probs_future: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """``V ↦ (Σ_t J_{t←n}ᵀ F_t J_{t←n}) V`` for the future positions ``t ≥ n``.

    ``jvp_fn`` maps ``V (p, m)`` to ``J V`` of shape ``(T, C, m)`` — the
    directional derivative of every future position's logits; ``probs_future`` is
    ``(T, C)`` the softmax at each future position (its Fisher
    ``F_t = diag(p_t) − p_t p_tᵀ``); ``vjp_fn`` maps a ``(T, C, m)`` cotangent
    back to ``(p, m)``. The per-position Fisher contracts the ``C`` axis
    independently, and the sum over ``t`` is folded into the single VJP, so the
    aggregated downstream operator is applied with no ``T·C × p`` matrix ever
    formed. With one future position this reduces bit-for-bit to
    :func:`_pullback_matvec`.
    """
    jv = jvp_fn(V)  # (T, C, m)
    # F_t (J V): apply each future position's softmax Fisher along C. Move C last
    # so `_softmax_fisher_apply` broadcasts probs_future (T, C) over the m axis.
    fjv = _softmax_fisher_apply(
        probs_future.unsqueeze(1),  # (T, 1, C)
        jv.transpose(1, 2),  # (T, m, C)
    ).transpose(1, 2)  # (T, C, m)
    return vjp_fn(fjv)  # (p, m), already summed over T inside the VJP


def harvest_downstream_output_fisher_factors(
    model: torch.nn.Module,
    hook_module: torch.nn.Module,
    inputs: Any,
    *,
    rank: int,
    oversample: int = 4,
    n_iter: int = 2,
    trace_probes: int = 8,
    seed: int = 0,
) -> HarvestShard:
    """Harvest per-token **downstream** output-Fisher factors at ``hook_module``.

    Identical interface to :func:`harvest_output_fisher_factors`, but the per-row
    metric is the forward-looking aggregate
    ``G_n = Σ_{t ≥ n} J_{t←n}ᵀ F_t J_{t←n}`` over the future positions ``n``
    reaches through the KV path, rather than the same-position
    ``G_n = J_nᵀ F_n J_n`` (#980, mechanism 2). The returned shard carries
    ``provenance="output_fisher_downstream"`` so the gam boundary routes it to
    ``RowMetric::output_fisher_downstream``; the factor *layout* (``U (n, p, r)``,
    flattened ``u[n, i*r + k] = U[n, i, k]``) and the ``mass_residual`` truncation
    diagnostic are exactly the same-position contract.

    Scientifically this is what makes dormant-feature detection forward-looking: a
    feature whose entire causal effect lands many tokens later has same-position
    Fisher ≈ 0 (the same-position lens reports it represented-but-not-driving),
    but registers nonzero downstream coupling here.

    This top-eigenfactor path requires forward-mode AD through every downstream
    attention block. Load Hugging Face models with
    ``attn_implementation="eager"``; unsupported SDPA/flash operators raise an
    actionable error at the JVP boundary and are never silently replaced.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1; got {rank}")

    act_flat, logits_all_from_act = _capture_activations_downstream(
        model, hook_module, inputs
    )
    n, p = int(act_flat.shape[0]), int(act_flat.shape[1])
    if rank > p:
        raise ValueError(f"rank {rank} exceeds activation dimension p = {p}")
    device = act_flat.device
    work_dtype = (
        act_flat.dtype
        if act_flat.dtype in (torch.float32, torch.float64)
        else torch.float32
    )

    U = np.empty((n, p, rank), dtype=np.float32)
    X = act_flat.to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    mass_residual = np.empty((n,), dtype=np.float32)

    for row in range(n):
        x_row = act_flat[row].to(work_dtype).detach().requires_grad_(False)

        # Future-position logit block as a function of x_n: f_future(x) -> (T, C),
        # T = n - row future positions (t = row .. n-1). Causal models give zero
        # Jacobian for t < row, so restricting to t >= row loses nothing and
        # avoids probing positions x_n cannot influence.
        def f_future(x: torch.Tensor, _row: int = row) -> torch.Tensor:
            allpos = logits_all_from_act(x, _row).to(work_dtype)  # (n_pos, C)
            return allpos[_row:]  # (T, C)

        with torch.no_grad():
            probs_future = torch.softmax(f_future(x_row), dim=-1)  # (T, C)

        def jvp_fn(V: torch.Tensor, _f=f_future, _x=x_row) -> torch.Tensor:
            cols = []
            for j in range(V.shape[1]):
                _out, jv = _jvp_with_attention_diagnostic(
                    _f,
                    _x,
                    V[:, j].contiguous(),
                    harvest_name="harvest_downstream_output_fisher_factors",
                )
                cols.append(jv)  # (T, C)
            return torch.stack(cols, dim=2)  # (T, C, m)

        _out0, vjp_raw = torch.func.vjp(f_future, x_row)

        def vjp_fn(W: torch.Tensor, _vjp=vjp_raw) -> torch.Tensor:
            cols = []
            for j in range(W.shape[2]):
                (gx,) = _vjp(W[:, :, j].contiguous())  # (p,)
                cols.append(gx)
            return torch.stack(cols, dim=1)  # (p, m)

        def matvec(V: torch.Tensor, _j=jvp_fn, _vj=vjp_fn, _pf=probs_future) -> torch.Tensor:
            return _downstream_pullback_matvec(_j, _vj, _pf, V)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + row)
        evals, evecs = _top_r_eigenpairs(
            matvec,
            p,
            rank,
            oversample=oversample,
            n_iter=n_iter,
            generator=gen,
            dtype=work_dtype,
            device=device,
        )
        scaled = evecs * evals.clamp_min(0.0).sqrt().unsqueeze(0)  # (p, r)

        gen_tr = torch.Generator(device="cpu")
        gen_tr.manual_seed(seed + 10_000 + row)
        trace = _trace_estimate(
            matvec,
            p,
            n_probes=trace_probes,
            generator=gen_tr,
            dtype=work_dtype,
            device=device,
        )
        residual = float(trace.item() - float(evals.sum().item()))
        mass_residual[row] = max(residual, 0.0)
        U[row] = scaled.detach().to(torch.float32).cpu().numpy()

    factor_kind = "exact_full" if rank == p else "uncertified_approximation"
    return HarvestShard(
        X=X,
        U=U,
        mass_residual=None if factor_kind == "exact_full" else mass_residual,
        rank=rank,
        factor_kind=factor_kind,
        provenance="output_fisher_downstream",
    )


# ---------------------------------------------------------------------------
# Shard I/O — the on-disk (X, U, mass_residual) contract
# ---------------------------------------------------------------------------


def save_harvest_shard(shard: HarvestShard, path: str | Path) -> str:
    """Write one strict, self-describing Fisher harvest artifact.

    Mirrors :mod:`gamfit._sampling`'s ``.npz`` suffix rule. Factors are stored
    f32 (their working precision); ``rank`` is recorded so the loader can assert
    the ``(n, p, r)`` layout. Returns the path that actually landed on disk.
    """
    out = Path(path)
    if out.suffix != ".npz":
        out = out.with_name(out.name + ".npz")
    fields = {
        "schema": np.str_(_HARVEST_SCHEMA),
        "X": np.ascontiguousarray(shard.X, dtype=np.float32),
        "U": np.ascontiguousarray(shard.U, dtype=np.float32),
        "rank": np.int64(shard.rank),
        "provenance": np.str_(shard.provenance),
        "factor_kind": np.str_(shard.factor_kind),
    }
    if shard.mass_residual is not None:
        fields["mass_residual"] = np.ascontiguousarray(
            shard.mass_residual, dtype=np.float32
        )
    np.savez(out, **fields)
    return str(out)


def load_harvest_shard(path: str | Path) -> dict[str, Any]:
    """Load a harvest shard, promoting to f64 at the gam boundary.

    Returns a dict carrying the required schema/provenance/factor-kind fields,
    f64 ``X (n, p)``, ``U (n, p, r)``, an optional f64 tail diagnostic, and the
    integer ``rank``. The f32 → f64 promotion happens here: the torch side
    keeps factors in their natural low precision; gam (which runs f64 CPU)
    consumes them promoted. The flattened ``U`` row-major layout
    ``u[n, i * r + k] = U[n, i, k]`` is exactly ``RowMetric::output_fisher``'s.
    """
    target = Path(path)
    if not target.exists() and target.suffix != ".npz":
        suffixed = target.with_name(target.name + ".npz")
        if suffixed.exists():
            target = suffixed
    npz = np.load(target)
    required = {"schema", "X", "U", "rank", "provenance", "factor_kind"}
    missing = required.difference(npz.files)
    if missing:
        raise ValueError(
            f"harvest shard at {target} is missing required fields {sorted(missing)}"
        )
    schema = str(npz["schema"].item())
    if schema != _HARVEST_SCHEMA:
        raise ValueError(
            f"harvest shard at {target} has unsupported schema {schema!r}; "
            f"expected {_HARVEST_SCHEMA!r}"
        )
    provenance = str(npz["provenance"].item())
    if provenance not in _HARVEST_PROVENANCES:
        raise ValueError(
            f"harvest shard at {target} has unknown provenance {provenance!r}; "
            f"expected one of {sorted(_HARVEST_PROVENANCES)}"
        )
    factor_kind = str(npz["factor_kind"].item())
    mass_residual = (
        np.asarray(npz["mass_residual"], dtype=np.float64)
        if "mass_residual" in npz.files
        else None
    )
    shard = HarvestShard(
        X=np.asarray(npz["X"], dtype=np.float64),
        U=np.asarray(npz["U"], dtype=np.float64),
        mass_residual=mass_residual,
        rank=int(npz["rank"].item()),
        provenance=provenance,
        factor_kind=factor_kind,
    )
    return {
        "schema": _HARVEST_SCHEMA,
        "X": np.asarray(npz["X"], dtype=np.float64),
        "U": np.asarray(npz["U"], dtype=np.float64),
        "mass_residual": mass_residual,
        "rank": shard.rank,
        "provenance": shard.provenance,
        "factor_kind": shard.factor_kind,
    }
