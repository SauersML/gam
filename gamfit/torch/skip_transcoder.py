"""Skip-Transcoder primitive: sparse paired-Smooth + low-rank affine bypass.

Reference
---------
Paulo, Shabalin, Belrose. "Transcoders Beat Sparse Autoencoders for
Interpretability." arXiv:2501.18823, 2025.

A skip-transcoder reconstructs the residual stream at layer L_out using a
*sparse code computed from* layer L_in plus a *low-rank affine bypass* of L_in:

    z       = SmoothThreshold(W_enc * x_in + b_enc) # sparse code, F atoms
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
smooth-threshold penalty (already in the Rust core) and Pca-style low-rank smooth.

Outer-loop REML
---------------
``skip_transcoder`` builds exactly one explicitly configured module.
``select_skip_transcoder`` profiles ``log(lambda_sparse)`` and the shared
``log(activation_threshold)`` continuously from an analytic evidence/gradient
oracle supplied by the caller's PyTorch training loop. Rank remains genuinely
discrete and is selected by evidence-priced sequential birth/death moves. The
result carries the two-neighbour stop certificate; failed or non-stationary
candidates are surfaced, never skipped or replaced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Literal

import torch
from torch import nn

from .penalties import SmoothThresholdPenalty


# ---------------------------------------------------------------------------
# SkipAffineSmooth: paired Smooth bundling enc + dec + low-rank bypass
# ---------------------------------------------------------------------------


class SkipAffineSmooth(nn.Module):
    """Paired-Smooth: sparse code on x_in + rank-r affine bypass of x_in.

    The two halves share the same design matrix on the input side (``x_in``)
    but their codomain is a DIFFERENT layer's residual (``y_out``). Sparse
    encode/decode is a width-F atom dictionary gated by ``SmoothThresholdPenalty``;
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
    activation_threshold:
        Base smooth activation threshold (scalar broadcast to F).
    lambda_sparse:
        Weight on the smooth-threshold sparsity penalty. This is the actual knob that
        trades fidelity against sparsity in the user's training loss
        (``loss += smooth_threshold(z_pre)``), so it must travel with the
        module — not just live in the outer-loop scoring metadata. Must be > 0.
    learnable_threshold:
        If True, REML can shift the threshold via ``log_threshold`` parameter.
    smoothing_eps:
        Sigmoid bandwidth (passed through to ``SmoothThresholdPenalty``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_atoms: int,
        rank_skip: int,
        activation_threshold: float = 0.05,
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
        if activation_threshold <= 0.0:
            raise ValueError("activation_threshold must be > 0")
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

        # Smooth-threshold prior shared with the gamfit composition engine.
        thresholds = torch.full((n_atoms,), float(activation_threshold), dtype=torch.float64)
        self.smooth_threshold = SmoothThresholdPenalty(
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
        """Sparse code after smooth threshold gating."""
        return self.smooth_threshold.gate(self.encode(x_in))

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
        """Smooth-threshold sparsity penalty on the pre-activation latents.

        Evaluates ``SmoothThresholdPenalty`` (weight ``= lambda_sparse``) on the
        pre-gate code ``z_pre = encode(x_in)``. This is the term that trades
        fidelity against sparsity in :meth:`loss`; its scale IS the module's
        ``lambda_sparse`` so different ``lambda_sparse`` give different values.
        """
        z_pre = self.encode(x_in)
        return self.smooth_threshold(z_pre)

    def loss(self, x_in: torch.Tensor, y_out: torch.Tensor) -> torch.Tensor:
        """Canonical training objective: reconstruction MSE + sparsity penalty.

        ``loss = mean((y_hat - y_out) ** 2) + smooth_threshold(z_pre)`` where the
        smooth-threshold penalty carries ``weight = lambda_sparse`` (set in
        ``__init__``). The sparse weight is therefore genuinely in the objective:
        two modules built with different ``lambda_sparse`` produce different
        ``loss`` on the same ``(x_in, y_out)``.
        """
        y_hat, _ = self.forward(x_in)
        recon = torch.mean((y_hat - y_out) ** 2)
        return recon + self.sparsity_penalty(x_in)

    # --------------------------------------------------------------
    # Convenience: attribution-graph edge weights at active points
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
        to output coordinate ``j`` at the smooth-threshold active points is

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
# Scalar builder and continuous/sequential evidence selector
# ---------------------------------------------------------------------------


def skip_transcoder(
    in_dim: int,
    out_dim: int,
    n_atoms: int,
    rank_skip: int,
    activation_threshold: float = 0.05,
    lambda_sparse: float = 1e-3,
    *,
    learnable_threshold: bool = False,
    smoothing_eps: float = 1e-3,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> SkipAffineSmooth:
    """Build one explicitly configured skip-transcoder.

    Hyperparameter sequences are intentionally rejected by the scalar type and
    by :class:`SkipAffineSmooth` validation. Model selection lives in
    :func:`select_skip_transcoder`; it never expands a Cartesian product.
    """
    if isinstance(rank_skip, bool) or not isinstance(rank_skip, Integral):
        raise TypeError("rank_skip must be one integer, not a candidate sequence")
    if isinstance(activation_threshold, bool) or not isinstance(
        activation_threshold, Real
    ):
        raise TypeError(
            "activation_threshold must be one real scalar, not a candidate sequence"
        )
    if isinstance(lambda_sparse, bool) or not isinstance(lambda_sparse, Real):
        raise TypeError(
            "lambda_sparse must be one real scalar, not a candidate sequence"
        )
    return SkipAffineSmooth(
        in_dim=in_dim,
        out_dim=out_dim,
        n_atoms=n_atoms,
        rank_skip=int(rank_skip),
        activation_threshold=float(activation_threshold),
        lambda_sparse=float(lambda_sparse),
        learnable_threshold=learnable_threshold,
        smoothing_eps=smoothing_eps,
        device=device,
        dtype=dtype,
    )


TransitionKind = Literal["seed", "birth", "death", "continuous"]


@dataclass(frozen=True)
class SkipTranscoderTrial:
    """One continuous evidence query at a genuinely discrete rank."""

    rank_skip: int
    log_lambda_sparse: float
    log_activation_threshold: float
    transition: TransitionKind
    warm_start: SkipAffineSmooth | None


@dataclass(frozen=True)
class SkipTranscoderProfile:
    """A converged fixed-rank fit and its analytic profile score jet.

    Constructing this type is the oracle's assertion that ``smooth`` converged
    for the same negative evidence whose two log-hyperparameter derivatives are
    returned. A partial/best-effort module must instead be represented by
    :class:`SkipTranscoderFailedCandidate`.
    """

    smooth: SkipAffineSmooth
    rank_skip: int
    log_lambda_sparse: float
    log_activation_threshold: float
    negative_log_evidence: float
    gradient_log_hyperparameters: tuple[float, float]
    gradient_scale: float

    def __post_init__(self) -> None:
        values = (
            self.log_lambda_sparse,
            self.log_activation_threshold,
            self.negative_log_evidence,
            *self.gradient_log_hyperparameters,
            self.gradient_scale,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("SkipTranscoderProfile score jet must be finite")
        if self.gradient_scale <= 0.0:
            raise ValueError("SkipTranscoderProfile.gradient_scale must be > 0")
        if self.rank_skip != self.smooth.rank_skip:
            raise ValueError(
                "SkipTranscoderProfile rank does not match its fitted smooth: "
                f"{self.rank_skip} != {self.smooth.rank_skip}"
            )

    @property
    def stationarity_tolerance(self) -> float:
        return math.sqrt(torch.finfo(torch.float64).eps) * self.gradient_scale

    @property
    def stationarity_defect(self) -> float:
        return math.hypot(*self.gradient_log_hyperparameters)


@dataclass(frozen=True)
class SkipTranscoderFailedCandidate:
    """Explicit failed trial record; failures are never made selectable."""

    trial: SkipTranscoderTrial
    error: Exception
    evidence_at_failure: float | None = None
    checkpoint: object | None = None


ProfileTrial = Callable[
    [SkipTranscoderTrial],
    SkipTranscoderProfile | SkipTranscoderFailedCandidate,
]


class SkipTranscoderSelectionError(RuntimeError):
    """Base class for typed skip-transcoder selection failures."""


class SkipTranscoderCandidateFailed(SkipTranscoderSelectionError):
    """A required continuous or neighbouring-rank candidate failed."""

    def __init__(self, failure: SkipTranscoderFailedCandidate) -> None:
        self.failure = failure
        super().__init__(
            f"skip-transcoder candidate rank={failure.trial.rank_skip} failed: "
            f"{failure.error}"
        )


class SkipTranscoderContinuousNonConvergence(SkipTranscoderSelectionError):
    """Continuous profiling stopped without a stationarity certificate."""

    def __init__(self, profile: SkipTranscoderProfile, reason: str) -> None:
        self.profile = profile
        self.reason = reason
        super().__init__(
            f"rank {profile.rank_skip} continuous evidence optimization did not "
            f"converge ({reason}); stationarity defect "
            f"{profile.stationarity_defect} exceeds "
            f"{profile.stationarity_tolerance}"
        )


@dataclass(frozen=True)
class RankTransition:
    from_rank: int
    to_rank: int
    negative_log_evidence_gap: float
    accepted: bool


@dataclass(frozen=True)
class RankNeighbourCertificate:
    direction: Literal["birth", "death"]
    rank_skip: int | None
    structurally_feasible: bool
    negative_log_evidence: float | None
    gap_from_selected: float | None
    stationarity_defect: float | None
    stationarity_tolerance: float | None


@dataclass(frozen=True)
class SkipTranscoderSelectionCertificate:
    selected_rank: int
    death: RankNeighbourCertificate
    birth: RankNeighbourCertificate
    transitions: tuple[RankTransition, ...]


@dataclass(frozen=True)
class SkipTranscoderSelectionResult:
    profile: SkipTranscoderProfile
    certificate: SkipTranscoderSelectionCertificate

    @property
    def smooth(self) -> SkipAffineSmooth:
        return self.profile.smooth

    @property
    def lambda_sparse(self) -> float:
        return math.exp(self.profile.log_lambda_sparse)

    @property
    def activation_threshold(self) -> float:
        return math.exp(self.profile.log_activation_threshold)

    @property
    def rank_skip(self) -> int:
        return self.profile.rank_skip

    @property
    def negative_log_evidence(self) -> float:
        return self.profile.negative_log_evidence


def _evaluate_profile_trial(
    profile_trial: ProfileTrial,
    trial: SkipTranscoderTrial,
) -> SkipTranscoderProfile:
    outcome = profile_trial(trial)
    if isinstance(outcome, SkipTranscoderFailedCandidate):
        raise SkipTranscoderCandidateFailed(outcome)
    if not isinstance(outcome, SkipTranscoderProfile):
        raise TypeError(
            "profile_trial must return SkipTranscoderProfile or "
            "SkipTranscoderFailedCandidate"
        )
    if outcome.rank_skip != trial.rank_skip:
        raise ValueError(
            f"profile_trial returned rank {outcome.rank_skip} for rank "
            f"{trial.rank_skip} trial"
        )
    coordinate_tolerance = math.sqrt(torch.finfo(torch.float64).eps)
    for name, actual, requested in (
        (
            "log_lambda_sparse",
            outcome.log_lambda_sparse,
            trial.log_lambda_sparse,
        ),
        (
            "log_activation_threshold",
            outcome.log_activation_threshold,
            trial.log_activation_threshold,
        ),
    ):
        if abs(actual - requested) > coordinate_tolerance * (1.0 + abs(requested)):
            raise ValueError(
                f"profile_trial returned {name}={actual} for requested {requested}"
            )
    return outcome


def _continuously_profile_rank(
    profile_trial: ProfileTrial,
    rank_skip: int,
    initial_logs: tuple[float, float],
    transition: TransitionKind,
    warm_start: SkipAffineSmooth | None,
) -> SkipTranscoderProfile:
    """Profile two log-hyperparameters with analytic-gradient Torch LBFGS.

    One strong-Wolfe LBFGS iteration is requested at a time. There is no
    arbitrary iteration cap: the loop terminates only at the oracle-calibrated
    stationarity tolerance or with a typed checkpoint when no representable
    evidence-decreasing step remains.
    """

    logs = torch.tensor(initial_logs, dtype=torch.float64, requires_grad=True)

    def evaluate(kind: TransitionKind, seed: SkipAffineSmooth | None) -> SkipTranscoderProfile:
        return _evaluate_profile_trial(
            profile_trial,
            SkipTranscoderTrial(
                rank_skip=rank_skip,
                log_lambda_sparse=float(logs[0].detach()),
                log_activation_threshold=float(logs[1].detach()),
                transition=kind,
                warm_start=seed,
            ),
        )

    current = evaluate(transition, warm_start)
    optimizer = torch.optim.LBFGS(
        [logs],
        max_iter=1,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )
    while current.stationarity_defect > current.stationarity_tolerance:
        previous_coordinates = tuple(float(value) for value in logs.detach())
        previous_score = current.negative_log_evidence
        closure_seed = current.smooth

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            profiled = evaluate("continuous", closure_seed)
            gradient = torch.tensor(
                profiled.gradient_log_hyperparameters,
                dtype=logs.dtype,
                device=logs.device,
            )
            # Value-matched linear surrogate: numeric value is the exact
            # profiled evidence; backward is the oracle's analytic gradient.
            loss = logs.new_tensor(profiled.negative_log_evidence) + (
                (logs - logs.detach()) * gradient
            ).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        current = evaluate("continuous", current.smooth)
        coordinates = tuple(float(value) for value in logs.detach())
        if coordinates == previous_coordinates:
            raise SkipTranscoderContinuousNonConvergence(
                current, "no representable parameter step remains"
            )
        score_roundoff = torch.finfo(torch.float64).eps * (
            1.0 + abs(previous_score) + abs(current.negative_log_evidence)
        )
        if current.negative_log_evidence >= previous_score - score_roundoff:
            raise SkipTranscoderContinuousNonConvergence(
                current, "strong-Wolfe step did not decrease evidence"
            )
    return current


def _rank_neighbour_certificate(
    direction: Literal["birth", "death"],
    selected: SkipTranscoderProfile,
    neighbour: SkipTranscoderProfile | None,
) -> RankNeighbourCertificate:
    if neighbour is None:
        return RankNeighbourCertificate(
            direction=direction,
            rank_skip=None,
            structurally_feasible=False,
            negative_log_evidence=None,
            gap_from_selected=None,
            stationarity_defect=None,
            stationarity_tolerance=None,
        )
    return RankNeighbourCertificate(
        direction=direction,
        rank_skip=neighbour.rank_skip,
        structurally_feasible=True,
        negative_log_evidence=neighbour.negative_log_evidence,
        gap_from_selected=(
            neighbour.negative_log_evidence - selected.negative_log_evidence
        ),
        stationarity_defect=neighbour.stationarity_defect,
        stationarity_tolerance=neighbour.stationarity_tolerance,
    )


def select_skip_transcoder(
    in_dim: int,
    out_dim: int,
    profile_trial: ProfileTrial,
    *,
    initial_lambda_sparse: float,
    initial_activation_threshold: float,
    initial_rank: int = 0,
) -> SkipTranscoderSelectionResult:
    """Continuously profile lambda/threshold and sequentially select rank.

    ``profile_trial`` is the PyTorch interaction boundary: for every requested
    rank and pair of log-hyperparameters it must train/profile the same evidence
    to convergence and return its analytic two-vector gradient. Rank selection
    starts at ``initial_rank``, evaluates every feasible birth/death neighbour,
    moves only on strict evidence improvement, and stops only after both
    neighbours have converged continuous optima that do not improve the fit.
    """

    if in_dim <= 0 or out_dim <= 0:
        raise ValueError("in_dim and out_dim must be > 0")
    structural_max_rank = min(in_dim, out_dim)
    if not 0 <= initial_rank <= structural_max_rank:
        raise ValueError(
            f"initial_rank must be in [0, {structural_max_rank}], got {initial_rank}"
        )
    if not math.isfinite(initial_lambda_sparse) or initial_lambda_sparse <= 0.0:
        raise ValueError("initial_lambda_sparse must be finite and > 0")
    if (
        not math.isfinite(initial_activation_threshold)
        or initial_activation_threshold <= 0.0
    ):
        raise ValueError("initial_activation_threshold must be finite and > 0")

    initial_logs = (
        math.log(initial_lambda_sparse),
        math.log(initial_activation_threshold),
    )
    profiles: dict[int, SkipTranscoderProfile] = {}
    transitions: list[RankTransition] = []
    current = _continuously_profile_rank(
        profile_trial,
        initial_rank,
        initial_logs,
        "seed",
        None,
    )
    profiles[initial_rank] = current

    while True:
        neighbours: list[SkipTranscoderProfile] = []
        for candidate_rank, direction in (
            (current.rank_skip - 1, "death"),
            (current.rank_skip + 1, "birth"),
        ):
            if not 0 <= candidate_rank <= structural_max_rank:
                continue
            candidate = profiles.get(candidate_rank)
            if candidate is None:
                candidate = _continuously_profile_rank(
                    profile_trial,
                    candidate_rank,
                    (
                        current.log_lambda_sparse,
                        current.log_activation_threshold,
                    ),
                    direction,
                    current.smooth,
                )
                profiles[candidate_rank] = candidate
            neighbours.append(candidate)

        best_neighbour = min(
            neighbours,
            key=lambda profile: profile.negative_log_evidence,
            default=None,
        )
        evidence_tolerance = torch.finfo(torch.float64).eps * (
            1.0
            + abs(current.negative_log_evidence)
            + (
                0.0
                if best_neighbour is None
                else abs(best_neighbour.negative_log_evidence)
            )
        )
        improves = (
            best_neighbour is not None
            and best_neighbour.negative_log_evidence
            < current.negative_log_evidence - evidence_tolerance
        )
        for neighbour in neighbours:
            transitions.append(
                RankTransition(
                    from_rank=current.rank_skip,
                    to_rank=neighbour.rank_skip,
                    negative_log_evidence_gap=(
                        neighbour.negative_log_evidence
                        - current.negative_log_evidence
                    ),
                    accepted=improves and neighbour is best_neighbour,
                )
            )
        if not improves:
            death = profiles.get(current.rank_skip - 1)
            birth = profiles.get(current.rank_skip + 1)
            certificate = SkipTranscoderSelectionCertificate(
                selected_rank=current.rank_skip,
                death=_rank_neighbour_certificate("death", current, death),
                birth=_rank_neighbour_certificate("birth", current, birth),
                transitions=tuple(transitions),
            )
            return SkipTranscoderSelectionResult(
                profile=current,
                certificate=certificate,
            )
        current = best_neighbour


__all__ = [
    "SkipAffineSmooth",
    "SkipTranscoderCandidateFailed",
    "SkipTranscoderContinuousNonConvergence",
    "SkipTranscoderFailedCandidate",
    "SkipTranscoderProfile",
    "SkipTranscoderSelectionCertificate",
    "SkipTranscoderSelectionError",
    "SkipTranscoderSelectionResult",
    "SkipTranscoderTrial",
    "select_skip_transcoder",
    "skip_transcoder",
]
