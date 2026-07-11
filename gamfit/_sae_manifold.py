"""Thin public facade for Rust-backed SAE manifold fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ._binding import rust_module
from ._penalty_bridge import (
    GumbelTemperatureSchedule,
    validate_gumbel_schedule_fields as _validate_gumbel_schedule_fields,
)


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


def _canonical_assignment(value: str, label: str) -> str:
    try:
        return str(rust_module().sae_canonical_assignment_kind(str(value)))
    except ValueError as exc:
        raise ValueError(
            f"{label}={value!r} is not a recognized assignment kind: {exc}"
        ) from None


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


ManifoldSAE = rust_module().ManifoldSAE

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
                     n_atoms: int | None = None,
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
                     promote_from_residual: bool = True,
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
        reroute. The accepted tokens are exactly ``"softmax"``, ``"ibp_map"``,
        ``"threshold_gate"``, and ``"topk"``.
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
    promote_from_residual
        When ``True`` (the default, #2239 magic-by-default), factor directions
        discovered in the structured residual passes that clear the full
        evidence certificate — evidence-ladder rank selection, energy above the
        idiosyncratic-noise floor, Beta-null persistence alignment, and the
        nursery dwell — are promoted (born) into the primary atom tier; the
        alternation self-extends its pass budget (hard-capped natively at
        ``STRUCTURED_RESIDUAL_PASSES_MAX``) only while certified lineages are
        live, so structureless data pays no extra passes. ``False`` keeps
        whitening while disabling evidence-certified atom birth. Coerced to
        ``bool``.
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
        Rust-owned atom objects), ``fitted`` ``(N, p)``, ``assignments``
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
    kind = _canonical_assignment(assignment, "assignment")
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
        raise ValueError(
            "sae_manifold_fit admits only the native manifold engine; this request "
            "belongs to sparse_dictionary_fit and is not silently substituted"
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
        promote_from_residual=bool(promote_from_residual),
        run_structure_search=bool(_run_structure_search),
        run_outer_rho_search=bool(_run_outer_rho_search),
    )
    payload_dict = dict(payload)
    # #2091 — the fit returns the PyO3 model itself.  Rust ingests the raw fit
    # payload, validates the complete schema, and builds the resident Fisher
    # RowMetric once; Python performs only array/config marshalling.
    return rust_module().sae_manifold_from_fit_payload(
        payload_dict,
        np.ascontiguousarray(x, dtype=np.float64),
        str(resolved_topology),
        str(kind),
        str(assignment),
        list(penalties),
        float(alpha_value),
        bool(alpha_is_auto),
        float(tau),
        float(sparsity),
        float(smoothness),
        float(effective_lr),
        int(max_iter_total),
        int(random_state),
        top_k_arg,
        float(jumprelu_threshold),
        fisher_factors=(
            None if fisher_shard is None else np.ascontiguousarray(fisher_shard[0])
        ),
        fisher_provenance=None if fisher_shard is None else fisher_shard[2],
        declared_bases=list(bases),
    )


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
        each atom object's ``decoder_coefficients`` attribute).
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
            _canonical_basis_kind(atom.topology)
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
        decoder_blocks = [
            np.ascontiguousarray(np.asarray(a.decoder, dtype=np.float64)) for a in self.atoms
        ]
        atom_dims = [int(a.latent_dim) for a in self.atoms]
        coords = [np.ascontiguousarray(np.asarray(a.coords, dtype=np.float64)) for a in self.atoms]
        assignments = np.ascontiguousarray(
            np.column_stack(
                [np.asarray(a.assignments, dtype=np.float64).reshape(-1) for a in self.atoms]
            )
        )
        assignment = _canonical_assignment(self.assignment, "assignment")
        return rust_module().sae_manifold_from_stagewise(
            [str(atom.topology) for atom in self.atoms],
            decoder_blocks,
            atom_dims,
            coords,
            assignments,
            fitted,
            np.ascontiguousarray(np.asarray(self.logits, dtype=np.float64)),
            training,
            assignment,
            str(seed.assignment),
            float(seed.alpha),
            bool(seed.learnable_alpha),
            float(seed.tau),
            float(seed.sparsity_strength),
            float(seed.smoothness),
            float(seed.learning_rate),
            int(seed.max_iter),
            int(seed.random_state),
            float(seed.jumprelu_threshold),
            float(self.reconstruction_ev()),
        )

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
    assignment: str = "softmax",
    structured_whitening: bool | None = None,
    fisher_factors: Any = None,
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
        Assignment/gate family, resolved through the shared public validator.
        ``"softmax"`` is the default and carries posterior responsibility mass
        through births and backfitting. ``"ibp_map"`` remains an explicit MAP
        opt-in; ``"threshold_gate"`` selects the hard threshold gate.
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
        fit's values (K-aware IBP α when ``assignment="ibp_map"``; τ = 0.5).
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
    kind = _canonical_assignment(assignment, "assignment")
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
        assignment=kind,
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


def _canonical_basis_kind(name: Any) -> str:
    """Canonical basis token from the Rust-owned SAE semantic schema."""
    return str(rust_module().sae_canonical_basis_kind(str(name)))


def _basis_to_topology(basis: Any) -> str:
    """Canonical topology label from the Rust-owned SAE semantic schema."""
    return str(rust_module().sae_topology_for_basis(str(basis)))


def _canonical_topology(name: str) -> str:
    """Canonical topology token from the Rust-owned SAE semantic schema."""
    return str(rust_module().sae_canonical_topology(str(name)))


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
    return str(rust_module().sae_flat_block_assignment(str(gating)))


def _bases(k_atoms: int, atom_basis: Any, atom_topology: str) -> list[str]:
    if atom_basis is None:
        atom_basis = rust_module().sae_basis_kind_for_topology(str(atom_topology))
    raw = [atom_basis] * k_atoms if isinstance(atom_basis, str) else list(atom_basis)
    if len(raw) != k_atoms:
        raise ValueError("atom_basis must provide one basis per atom")
    return [str(v) for v in raw]


def _topology_for_bases(bases: list[str]) -> str:
    """Collapse resolved bases through the Rust-owned semantic schema."""
    scalar, _per_atom = rust_module().sae_atom_topologies([str(b) for b in bases])
    if scalar is None:
        raise ValueError("a fitted SAE dictionary must contain at least one basis")
    return str(scalar)


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


__all__ = ["GumbelTemperatureSchedule", "ManifoldSAE",
           "gumbel_geometric_schedule", "gumbel_linear_schedule", "gumbel_reciprocal_iter_schedule",
           "fit", "plot", "sae_manifold_fit"]
