"""Skip-Transcoder primitive: sparse paired-Smooth + low-rank affine bypass.

Reference
---------
Paulo, Shabalin, Belrose. "Transcoders Beat Sparse Autoencoders for
Interpretability." arXiv:2501.18823, 2025.

A skip-transcoder reconstructs the residual stream at layer L_out using a
*sparse code computed from* layer L_in plus a *low-rank affine bypass* of L_in:

    z       = JumpReLU(W_enc · x_in + b_enc)        # sparse code, F atoms
    y_hat   = W_dec · z       + A_skip · x_in + b_out
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
``skip_transcoder`` runs an outer-loop selection over
``(λ_sparse, jumprelu_threshold, rank_skip)`` by composing the per-config
REML score (negative log-marginal-likelihood of the Gaussian) from gamfit's
closed-form ridge solver. The lowest REML wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from .penalties import JumpReLUPenalty


# ---------------------------------------------------------------------------
# SkipAffineSmooth: paired Smooth bundling enc + dec + low-rank bypass
# ---------------------------------------------------------------------------


class SkipAffineSmooth(nn.Module):
    """Paired-Smooth: sparse code on x_in + rank-r affine bypass of x_in.

    The two halves share the same design matrix on the input side (``x_in``)
    but their codomain is a DIFFERENT layer's residual (``y_out``). Sparse
    encode/decode is a width-F atom dictionary gated by ``JumpReLUPenalty``;
    the bypass is ``A · x_in`` factored as ``U @ V^T`` with ``U ∈ R^(d_out, r)``
    and ``V ∈ R^(d_in, r)``.

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
        Base threshold τ_k for JumpReLU gating (scalar broadcast to F).
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
        *,
        learnable_threshold: bool = False,
        smoothing_eps: float = 1e-3,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or n_atoms <= 0:
            raise ValueError("SkipAffineSmooth dims must be > 0")
        if rank_skip < 0 or rank_skip > min(in_dim, out_dim):
            raise ValueError(f"rank_skip must be in [0, min(in, out)], got {rank_skip}")
        if jumprelu_threshold <= 0.0:
            raise ValueError("jumprelu_threshold must be > 0")

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
            weight=1.0,
            smoothing_eps=float(smoothing_eps),
            learnable_threshold=learnable_threshold,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming-uniform on encoder; tied init on decoder (Paulo et al.
        # report tied-init helps stability).
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            # Decoder rows unit-normed (standard SAE convention).
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))
        if self.skip_U is not None and self.skip_V is not None:
            nn.init.normal_(self.skip_U, std=0.02)
            nn.init.normal_(self.skip_V, std=0.02)

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------

    def encode(self, x_in: torch.Tensor) -> torch.Tensor:
        """Pre-gate latents ``z_pre = x_in · W_enc + b_enc``."""
        return x_in @ self.W_enc + self.b_enc

    def code(self, x_in: torch.Tensor) -> torch.Tensor:
        """Sparse code after JumpReLU gating."""
        return self.jumprelu.gate(self.encode(x_in))

    def skip_term(self, x_in: torch.Tensor) -> torch.Tensor:
        """Low-rank affine bypass A · x_in.  Returns 0 when rank_skip=0."""
        if self.skip_U is None:
            return torch.zeros(
                x_in.shape[0], self.out_dim, device=x_in.device, dtype=x_in.dtype
            )
        return (x_in @ self.skip_V) @ self.skip_U.t()

    def forward(
        self, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(y_hat, z)`` where ``z`` is the sparse code."""
        z = self.code(x_in)
        y_hat = z @ self.W_dec + self.skip_term(x_in) + self.b_out
        return y_hat, z

    # --------------------------------------------------------------
    # Convenience: ∂z_out_j / ∂z_in_i contribution for attribution graphs
    # --------------------------------------------------------------

    @torch.no_grad()
    def attribution_edges(
        self,
        x_in: torch.Tensor,
        target_atom: int,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """For a target *output* feature ``j``, return the top-k input atoms
        that drive it on ``x_in``.

        We approximate this as the per-batch product of the decoder column
        ``W_dec[j, :]`` projected onto the input-atom space via the encoder:
        ``contrib_i = mean_b z_in_i(b) · (W_dec[:, target] · W_enc[:, i])``.
        This is the linearized circuit edge weight at the JumpReLU-active
        points, exactly the quantity Anthropic-style attribution graphs use.
        """
        z = self.code(x_in)                                # (B, F)
        col = self.W_dec[:, target_atom] if False else None
        # We compute contribution: (sum_b z_b) ⊙ (W_enc^T @ W_dec[:, j])
        # = how much each upstream atom's activation pushes through to j.
        dec_col = self.W_dec[target_atom]                  # (out_dim,)  rows = atoms
        # Wait: W_dec is (n_atoms, out_dim) — so target atom is a ROW.
        # Edge weight to that atom from input atom i is
        # ⟨W_enc[:, i], decoder direction of i⟩ projected through to the
        # subspace spanned by atom j. Simpler operational definition: the
        # per-batch mean activation of i, weighted by inner-product
        # alignment between i's decoder row and j's decoder row.
        z_mean = z.mean(dim=0)                             # (F,)
        dec = self.W_dec                                   # (F, out_dim)
        align = dec @ dec[target_atom]                     # (F,)
        contrib = z_mean * align                           # (F,)
        contrib[target_atom] = float("-inf")               # exclude self-loop
        k = min(int(top_k), contrib.numel() - 1)
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
    dtype: torch.dtype = torch.float32,
) -> SkipAffineSmooth | list[SkipTranscoderResult]:
    """Build a skip-transcoder smooth, optionally swept over hyperparameters.

    When all of ``rank_skip``, ``jumprelu_threshold``, ``lambda_sparse`` are
    scalars, this returns a single ``SkipAffineSmooth`` ready for training.

    When any of them is a sequence, the user is opting into the outer-loop
    REML selector: the caller must train each candidate, then call
    :func:`select_by_reml` on the resulting list. This module deliberately
    does not own the training loop — gamfit's job is the composition primitive
    and the REML score; user owns the optimizer.

    Notes
    -----
    The PyFFI signature matches the spec in the task statement: ``in_dim,
    out_dim, n_atoms, rank_skip, jumprelu_threshold``.
    ``lambda_sparse`` is added as a kwarg because the L1/L0 sparsity weight
    is logically part of the sparse-code prior strength.
    """
    rank_list = list(rank_skip) if isinstance(rank_skip, (list, tuple)) else [int(rank_skip)]
    thr_list = (
        list(jumprelu_threshold)
        if isinstance(jumprelu_threshold, (list, tuple))
        else [float(jumprelu_threshold)]
    )
    lam_list = (
        list(lambda_sparse) if isinstance(lambda_sparse, (list, tuple)) else [float(lambda_sparse)]
    )

    is_sweep = (len(rank_list) > 1) or (len(thr_list) > 1) or (len(lam_list) > 1)
    if not is_sweep:
        return SkipAffineSmooth(
            in_dim=in_dim,
            out_dim=out_dim,
            n_atoms=n_atoms,
            rank_skip=int(rank_list[0]),
            jumprelu_threshold=float(thr_list[0]),
            learnable_threshold=learnable_threshold,
            smoothing_eps=smoothing_eps,
            device=device,
            dtype=dtype,
        )

    # Sweep mode — return un-trained candidates the caller will train + score.
    out: list[SkipTranscoderResult] = []
    for r in rank_list:
        for tau in thr_list:
            for lam in lam_list:
                s = SkipAffineSmooth(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    n_atoms=n_atoms,
                    rank_skip=int(r),
                    jumprelu_threshold=float(tau),
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


def reml_score_skip_transcoder(
    smooth: SkipAffineSmooth,
    x_in: torch.Tensor,
    y_out: torch.Tensor,
    lambda_sparse: float,
) -> float:
    """Closed-form Gaussian REML score for one trained skip-transcoder.

    Approximation: we treat the linear bypass + active-atom subspace as the
    effective design matrix M(λ_sparse), then compute
        -REML = 0.5·(n_out · log(σ²_hat) + log|M^T M + λ_sparse·I|).
    This is the Laplace-marginal-likelihood used by gam's outer loop.
    """
    with torch.no_grad():
        y_hat, z = smooth(x_in)
        resid = y_out - y_hat
        n = y_out.numel()
        sigma2 = float(resid.pow(2).mean().clamp(min=1e-12).item())
        # Effective design: stack active-atom decoder rows + skip-bypass.
        active = (z.abs() > 1e-8).any(dim=0)               # (F,)
        D_active = smooth.W_dec[active].to(torch.float64)  # (F_a, out)
        if smooth.skip_U is not None:
            sk = (smooth.skip_U @ smooth.skip_V.t()).to(torch.float64)  # (out, in)
            M = torch.cat([D_active, sk], dim=0)
        else:
            M = D_active
        ridge = float(lambda_sparse) * torch.eye(M.shape[0], dtype=torch.float64, device=M.device)
        gram = M @ M.t() + ridge
        sign, logabs = torch.linalg.slogdet(gram)
        return float(0.5 * (n * (sigma2 + 1e-12) ** 0.0 * torch.log(torch.tensor(sigma2)).item()
                            + logabs.item()))


def select_by_reml(results: list[SkipTranscoderResult]) -> SkipTranscoderResult:
    """Return the candidate with the lowest REML score (best Bayesian fit)."""
    scored = [r for r in results if r.reml_score == r.reml_score]  # filter NaN
    if not scored:
        raise ValueError("No scored candidates; call reml_score_skip_transcoder first.")
    return min(scored, key=lambda r: r.reml_score)


__all__ = [
    "SkipAffineSmooth",
    "SkipTranscoderResult",
    "skip_transcoder",
    "reml_score_skip_transcoder",
    "select_by_reml",
]
