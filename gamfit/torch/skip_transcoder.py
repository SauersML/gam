"""Skip-Transcoder primitive: sparse paired-Smooth + low-rank affine bypass.

Reference
---------
Paulo, Shabalin, Belrose. "Transcoders Beat Sparse Autoencoders for
Interpretability." arXiv:2501.18823, 2025.

A skip-transcoder reconstructs the residual stream at layer L_out using a
*sparse code computed from* layer L_in plus a *low-rank affine bypass* of L_in:

    z       = JumpReLU(W_enc * x_in + b_enc)        # sparse code, F atoms
    y_hat   = W_dec * z       + A_skip * x_in + b_out
              (sparse circuit)   (rank-r bypass)

The skip path lets the dictionary specialize on residual structure that the
deep network truly added between L_in and L_out, instead of having to
re-encode the parts that are linearly preserved. This makes feature i at L_in
a CIRCUIT PRIMITIVE that causally explains feature j at L_out, enabling
attribution graphs (Anthropic-style).

This module is the *gamfit* end: a compositional Smooth that pairs an
identity-basis sparse Smooth (the dictionary) with a rank-constrained
linear Smooth (the bypass), sharing the design matrix on the input side
and targeting a DIFFERENT layer's residual. It re-uses gamfit's existing
JumpReLU penalty (already in the Rust core) and Pca-style low-rank smooth.

Outer-loop REML
---------------
``skip_transcoder`` expands an ``(lambda_sparse, jumprelu_threshold,
rank_skip)`` grid into per-config ``SkipAffineSmooth`` modules. The
analytic REML score and the argmin selector are computed in the gam Rust
core (``skip_transcoder_reml_metrics`` / ``skip_transcoder_select_reml``):
this Python module is a thin marshaling layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .._binding import rust_module
from ._coerce import to_numpy_f64
from .penalties import JumpReLUPenalty


# ---------------------------------------------------------------------------
# SkipAffineSmooth: paired Smooth bundling enc + dec + low-rank bypass
# ---------------------------------------------------------------------------


class SkipAffineSmooth(nn.Module):
    """Paired-Smooth: sparse code on x_in + rank-r affine bypass of x_in.

    The two halves share the same design matrix on the input side (``x_in``)
    but their codomain is a DIFFERENT layer's residual (``y_out``). Sparse
    encode/decode is a width-F atom dictionary gated by ``JumpReLUPenalty``;
    the bypass is ``A * x_in`` factored as ``U @ V^T`` with ``U`` in
    ``R^(d_out, r)`` and ``V`` in ``R^(d_in, r)``.

    Parameters
    ----------
    in_dim, out_dim:
        Layer L_in and L_out residual widths.
    n_atoms:
        Dictionary width F.
    rank_skip:
        Rank of the affine bypass A. 0 disables the skip (degenerates to a
        plain transcoder).
    jumprelu_threshold:
        Base threshold tau_k for JumpReLU gating (scalar broadcast to F).
    lambda_sparse:
        Weight on the JumpReLU sparsity penalty. This is the actual knob that
        trades fidelity against sparsity in the user's training loss
        (``loss += jumprelu.penalty(z_pre)``), so it must travel with the
        module — not just live in the outer-loop scoring metadata. Must be > 0.
    learnable_threshold:
        If True, REML can shift the threshold via ``log_threshold`` parameter.
    smoothing_eps:
        STE bandwidth (passed through to ``JumpReLUPenalty``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_atoms: int,
        rank_skip: int,
        jumprelu_threshold: float = 0.05,
        lambda_sparse: float = 1.0,
        *,
        learnable_threshold: bool = False,
        smoothing_eps: float = 1e-3,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if in_dim <= 0 or out_dim <= 0 or n_atoms <= 0:
            raise ValueError("SkipAffineSmooth dims must be > 0")
        if rank_skip < 0 or rank_skip > min(in_dim, out_dim):
            raise ValueError(f"rank_skip must be in [0, min(in, out)], got {rank_skip}")
        if jumprelu_threshold <= 0.0:
            raise ValueError("jumprelu_threshold must be > 0")
        if lambda_sparse <= 0.0:
            raise ValueError("lambda_sparse must be > 0")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_atoms = int(n_atoms)
        self.rank_skip = int(rank_skip)

        # Encoder + decoder (the atom dictionary).
        self.W_enc = nn.Parameter(torch.empty(in_dim, n_atoms, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(n_atoms, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(n_atoms, out_dim, device=device, dtype=dtype))
        self.b_out = nn.Parameter(torch.zeros(out_dim, device=device, dtype=dtype))

        # Low-rank affine bypass: A = U @ V^T,  U:(out, r), V:(in, r)
        if rank_skip > 0:
            self.skip_U = nn.Parameter(
                torch.empty(out_dim, rank_skip, device=device, dtype=dtype)
            )
            self.skip_V = nn.Parameter(
                torch.empty(in_dim, rank_skip, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("skip_U", None)
            self.register_parameter("skip_V", None)

        # JumpReLU prior shared with the gamfit composition engine.
        thresholds = torch.full((n_atoms,), float(jumprelu_threshold), dtype=torch.float64)
        self.jumprelu = JumpReLUPenalty(
            thresholds=thresholds,
            weight=float(lambda_sparse),
            smoothing_eps=float(smoothing_eps),
            learnable_threshold=learnable_threshold,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming-uniform on encoder; tied init on decoder when shapes permit
        # it (Paulo et al. report tied-init helps stability).
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            if self.in_dim == self.out_dim:
                self.W_dec.copy_(self.W_enc.t())
            else:
                nn.init.xavier_normal_(self.W_dec)
            # Decoder rows unit-normed (standard SAE convention).
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))
        if self.skip_U is not None and self.skip_V is not None:
            nn.init.normal_(self.skip_U, std=0.02)
            nn.init.normal_(self.skip_V, std=0.02)

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------

    def encode(self, x_in: torch.Tensor) -> torch.Tensor:
        """Pre-gate latents ``z_pre = x_in * W_enc + b_enc``."""
        return x_in @ self.W_enc + self.b_enc

    def code(self, x_in: torch.Tensor) -> torch.Tensor:
        """Sparse code after JumpReLU gating."""
        return self.jumprelu.gate(self.encode(x_in))

    def skip_projection(self, x_in: torch.Tensor) -> torch.Tensor | None:
        """Skip input projection ``XV = x_in @ skip_V``, shape ``(B, rank)``.

        This is the skip bypass's data-dependent activation: together with the
        output loading ``skip_U`` it is the *only* way the rank-``r`` map
        ``A = U V^T`` enters the prediction (``skip(x) = (x V) U^T``). Returns
        ``None`` when the skip is disabled (``rank_skip = 0``).
        """
        if self.skip_V is None:
            return None
        return x_in @ self.skip_V

    def skip_term(self, x_in: torch.Tensor) -> torch.Tensor:
        """Low-rank affine bypass A * x_in.  Returns 0 when rank_skip=0."""
        proj = self.skip_projection(x_in)
        if proj is None:
            return torch.zeros(
                x_in.shape[0], self.out_dim, device=x_in.device, dtype=x_in.dtype
            )
        return proj @ self.skip_U.t()

    def forward(
        self, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(y_hat, z)`` where ``z`` is the sparse code."""
        z = self.code(x_in)
        y_hat = z @ self.W_dec + self.skip_term(x_in) + self.b_out
        return y_hat, z

    def sparsity_penalty(self, x_in: torch.Tensor) -> torch.Tensor:
        """JumpReLU sparsity penalty on the pre-activation latents.

        Evaluates ``JumpReLUPenalty`` (weight ``= lambda_sparse``) on the
        pre-gate code ``z_pre = encode(x_in)``. This is the term that trades
        fidelity against sparsity in :meth:`loss`; its scale IS the module's
        ``lambda_sparse`` so different ``lambda_sparse`` give different values.
        """
        z_pre = self.encode(x_in)
        return self.jumprelu(z_pre)

    def loss(self, x_in: torch.Tensor, y_out: torch.Tensor) -> torch.Tensor:
        """Canonical training objective: reconstruction MSE + sparsity penalty.

        ``loss = mean((y_hat - y_out) ** 2) + jumprelu_penalty(z_pre)`` where the
        JumpReLU penalty already carries ``weight = lambda_sparse`` (set in
        ``__init__``). The sparse weight is therefore genuinely in the objective:
        two modules built with different ``lambda_sparse`` produce different
        ``loss`` on the same ``(x_in, y_out)``.
        """
        y_hat, _ = self.forward(x_in)
        recon = torch.mean((y_hat - y_out) ** 2)
        return recon + self.sparsity_penalty(x_in)

    # --------------------------------------------------------------
    # Convenience: attribution-graph edge weights at JumpReLU-active points
    # --------------------------------------------------------------

    @torch.no_grad()
    def attribution_edges(
        self,
        x_in: torch.Tensor,
        target_feature: int,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """For a target *output* feature ``j`` (an output-residual coordinate,
        ``0 <= j < out_dim``), return the top-k upstream atoms that drive it on
        ``x_in``.

        The model output is ``y_hat = z @ W_dec + skip(x_in) + b_out`` (see
        :meth:`forward`), so the direct, linearized contribution of atom ``i``
        to output coordinate ``j`` at the JumpReLU-active points is

            contrib_i = mean_b z_{b,i} * W_dec[i, j],

        i.e. the per-batch mean activation of atom ``i`` times the ``j``-th
        column of its decoder row. These are the circuit-edge weights that
        Anthropic-style attribution graphs consume. The skip bypass is a
        separate, atom-independent linear path and so contributes no atom edge.

        ``target_feature`` indexes the output residual coordinate, which lives
        in a different space than the ``n_atoms`` upstream atoms; it is
        validated against ``out_dim`` rather than ``n_atoms``.
        """
        if not (0 <= int(target_feature) < self.out_dim):
            raise IndexError(
                f"target_feature must be in [0, {self.out_dim}), got {target_feature}"
            )
        z = self.code(x_in)                                # (B, F)
        z_mean = z.mean(dim=0)                             # (F,)
        contrib = z_mean * self.W_dec[:, int(target_feature)]   # (F,)
        k = min(int(top_k), contrib.numel())
        vals, idx = torch.topk(contrib, k=k)
        return [(int(i.item()), float(v.item())) for i, v in zip(idx, vals)]


# ---------------------------------------------------------------------------
# skip_transcoder: PyFFI entry point with outer-loop REML model selection
# ---------------------------------------------------------------------------


@dataclass
class SkipTranscoderResult:
    """One configuration in the outer-loop sweep."""

    smooth: SkipAffineSmooth
    lambda_sparse: float
    jumprelu_threshold: float
    rank_skip: int
    reml_score: float          # negative log-marginal-likelihood (lower = better)
    mse: float
    sparsity: float            # mean nonzero fraction
    explained_variance: float


def _int_grid(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _float_grid(value: float | int | Sequence[float]) -> list[float]:
    if isinstance(value, (float, int)):
        return [float(value)]
    return [float(item) for item in value]


def skip_transcoder(
    in_dim: int,
    out_dim: int,
    n_atoms: int,
    rank_skip: int | Sequence[int] = 64,
    jumprelu_threshold: float | Sequence[float] = 0.05,
    lambda_sparse: float | Sequence[float] = 1e-3,
    *,
    learnable_threshold: bool = False,
    smoothing_eps: float = 1e-3,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> SkipAffineSmooth | list[SkipTranscoderResult]:
    """Build a skip-transcoder smooth, optionally swept over hyperparameters.

    When all of ``rank_skip``, ``jumprelu_threshold``, ``lambda_sparse`` are
    scalars, this returns a single ``SkipAffineSmooth`` ready for training.

    When any of them is a sequence, the caller opts into the outer-loop
    REML selector: train each returned candidate, then call
    :func:`score_and_select` to fill the REML/MSE/sparsity/EV fields and
    pick the best. This module deliberately does not own the training
    loop — gamfit's job is the composition primitive and the REML score;
    the user owns the optimizer.
    """
    rank_list = _int_grid(rank_skip)
    thr_list = _float_grid(jumprelu_threshold)
    lam_list = _float_grid(lambda_sparse)

    is_sweep = (len(rank_list) > 1) or (len(thr_list) > 1) or (len(lam_list) > 1)
    if not is_sweep:
        return SkipAffineSmooth(
            in_dim=in_dim,
            out_dim=out_dim,
            n_atoms=n_atoms,
            rank_skip=int(rank_list[0]),
            jumprelu_threshold=float(thr_list[0]),
            lambda_sparse=float(lam_list[0]),
            learnable_threshold=learnable_threshold,
            smoothing_eps=smoothing_eps,
            device=device,
            dtype=dtype,
        )

    out: list[SkipTranscoderResult] = []
    for r in rank_list:
        for tau in thr_list:
            for lam in lam_list:
                # Each candidate carries its own lambda_sparse so the module
                # the user trains uses the same sparsity weight it is scored
                # under (SkipTranscoderResult.lambda_sparse below).
                s = SkipAffineSmooth(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    n_atoms=n_atoms,
                    rank_skip=int(r),
                    jumprelu_threshold=float(tau),
                    lambda_sparse=float(lam),
                    learnable_threshold=learnable_threshold,
                    smoothing_eps=smoothing_eps,
                    device=device,
                    dtype=dtype,
                )
                out.append(
                    SkipTranscoderResult(
                        smooth=s,
                        lambda_sparse=float(lam),
                        jumprelu_threshold=float(tau),
                        rank_skip=int(r),
                        reml_score=float("nan"),
                        mse=float("nan"),
                        sparsity=float("nan"),
                        explained_variance=float("nan"),
                    )
                )
    return out


def _as_2d_f64_numpy(tensor: torch.Tensor):
    """Detach, cast to f64, ensure 2-D shape for the Rust metrics call."""
    t = tensor.detach()
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return to_numpy_f64(t)


def reml_score_skip_transcoder(
    smooth: SkipAffineSmooth,
    x_in: torch.Tensor,
    y_out: torch.Tensor,
    lambda_sparse: float,
) -> float:
    """Closed-form Gaussian REML score for one trained skip-transcoder.

    Thin marshaling over the Rust ``skip_transcoder_reml_metrics`` driver:
    the Rust core assembles the effective design — every feature column is the
    flattened outer product of a per-observation activation with an
    output-space loading: ``(z[:, a], W_dec[a, :])`` for sparse atoms and
    ``(XV[:, r], U[:, r])`` for the skip bypass, with ``XV = x_in @ skip_V``.
    It Cholesky-factors ``DᵀD + lambda_sparse * I`` and returns
    ``0.5 * (n * log sigma2 + logdet)`` — the gauge-invariant Laplace marginal
    likelihood used by gam's outer loop.
    """
    with torch.no_grad():
        y_hat, z = smooth(x_in)
        skip_proj = smooth.skip_projection(x_in)
        skip_u = None if smooth.skip_U is None else _as_2d_f64_numpy(smooth.skip_U)
        skip_proj = None if skip_proj is None else _as_2d_f64_numpy(skip_proj)
        metrics = rust_module().skip_transcoder_reml_metrics(
            _as_2d_f64_numpy(y_out),
            _as_2d_f64_numpy(y_hat),
            _as_2d_f64_numpy(z),
            _as_2d_f64_numpy(smooth.W_dec),
            float(lambda_sparse),
            skip_u,
            skip_proj,
        )
        return float(metrics["reml_score"])


def select_by_reml(results: list[SkipTranscoderResult]) -> SkipTranscoderResult:
    """Return the candidate with the lowest REML score (best Bayesian fit).

    Thin wrapper over the Rust ``skip_transcoder_select_reml`` argmin,
    which performs the NaN-skip + raise-on-empty contract.
    """
    idx = rust_module().skip_transcoder_select_reml([float(r.reml_score) for r in results])
    return results[int(idx)]


def score_and_select(
    results: list[SkipTranscoderResult],
    x_in: torch.Tensor,
    y_out: torch.Tensor,
) -> SkipTranscoderResult:
    """Score each (trained) candidate against ``(x_in, y_out)`` and return
    the best.

    For every entry in ``results`` this fills ``reml_score``, ``mse``,
    ``sparsity`` and ``explained_variance`` in place by calling the Rust
    ``skip_transcoder_reml_metrics`` driver, then picks the argmin REML
    via ``skip_transcoder_select_reml``.
    """
    rm = rust_module()
    with torch.no_grad():
        for r in results:
            y_hat, z = r.smooth(x_in)
            skip_proj = r.smooth.skip_projection(x_in)
            skip_u = None if r.smooth.skip_U is None else _as_2d_f64_numpy(r.smooth.skip_U)
            skip_proj = None if skip_proj is None else _as_2d_f64_numpy(skip_proj)
            m = rm.skip_transcoder_reml_metrics(
                _as_2d_f64_numpy(y_out),
                _as_2d_f64_numpy(y_hat),
                _as_2d_f64_numpy(z),
                _as_2d_f64_numpy(r.smooth.W_dec),
                float(r.lambda_sparse),
                skip_u,
                skip_proj,
            )
            r.reml_score = float(m["reml_score"])
            r.mse = float(m["mse"])
            r.sparsity = float(m["sparsity"])
            r.explained_variance = float(m["explained_variance"])
    return select_by_reml(results)


__all__ = [
    "SkipAffineSmooth",
    "SkipTranscoderResult",
    "skip_transcoder",
    "reml_score_skip_transcoder",
    "select_by_reml",
    "score_and_select",
]
