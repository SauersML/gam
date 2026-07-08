use super::*;

pub(crate) const SAE_BYTES_PER_F64: usize = 8;

pub(crate) const SAE_HOST_IN_CORE_FALLBACK_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub(crate) const SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR: usize = 3;

pub(crate) const SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR: usize = 5;

pub(crate) const SAE_CPU_L2_CACHE_BYTES: usize = 1024 * 1024;

pub(crate) const SAE_CHUNK_CACHE_MULTIPLE: usize = 8;

pub(crate) const SAE_MIN_STREAMING_CHUNK_ROWS: usize = 256;

pub(crate) const SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER: usize = 32;

/// Headroom kept free when admitting an in-core plan: we never hand the whole
/// reported "available" figure to a single allocation. `available` from the OS
/// is an estimate (reclaimable cache, other processes, allocator slack), so a
/// plan sized at 100% of it routinely OOMs in practice. Reserve the larger of
/// 1/8 of available and a fixed 256 MiB floor before computing the budget.
pub(crate) const SAE_HOST_MEMORY_RESERVE_FRACTION_DENOMINATOR: usize = 8;
pub(crate) const SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES: usize = 256 * 1024 * 1024;

/// Conservative lower bound on the pooled device in-core budget any probed CUDA
/// runtime can report (`Σ memory_budget_for(ordinal) / 4`), used by the
/// pre-probe size gates in [`sae_streaming_plan_for_shape`] and
/// `SaeManifoldTerm::sparse_active_plan`.
///
/// Working sets at or below this figure are admitted identically whether the
/// budget comes from the host or from ANY device pool: the smallest CUDA device
/// gam can meaningfully probe still has hundreds of MiB of budget (a 64 MiB
/// pooled budget would require a device with under 256 MiB of usable memory —
/// below every supported compute-capability generation and every MIG slice), so
/// a plan whose peak fits under 64 MiB cannot have its admission flipped by the
/// device-budget cap. Those CPU-sized shapes therefore skip
/// `GpuRuntime::global()` — and the CUDA primary-context creation on all GPUs
/// that the first probe performs — entirely. Larger shapes still probe and use
/// the exact device-aware budget as before.
pub(crate) const SAE_MIN_DEVICE_POOL_IN_CORE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Absolute floor for the *matrix-free streaming* admission test. The chunked
/// matrix-free plan exists precisely to bound peak memory by the chunk window
/// rather than the full problem, so it must stay admittable even when the
/// in-core budget collapses to ~0 (memory-starved / oversubscribed box, or a
/// cgroup whose `available − reserve` underflows to zero). A starved box can
/// always afford a few chunk windows plus the border vector workspace; gating
/// streaming on the same budget as the dense direct plan would refuse the one
/// plan that was designed to run there. The dense direct path keeps gating on
/// the real budget — it can genuinely OOM — so this floor only ever relaxes the
/// streaming fallback, never admits a full-batch in-core solve.
pub(crate) const SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES: usize = 64 * 1024 * 1024;

/// Absolute size below which a *direct* (dense, full-batch) plan is always
/// admissible provided it fits the reported available memory, regardless of the
/// headroom-reserved in-core budget. The budget subtracts a flat `max(available/8,
/// 256 MiB)` reserve so a LARGE allocation is never sized at ~100% of available
/// and OOMs; but on a memory-starved box (available below the 256 MiB floor) that
/// reserve underflows the budget to ~0, which then rejects even a trivially-small
/// dense plan (e.g. an 18 KiB K=1 toy fit) — and `reml_criterion` has no streaming
/// fallback for the direct logdet, so it hard-errors instead of running. A dense
/// plan at or below this size cannot meaningfully OOM a box that reports at least
/// this much available, so admitting it can never reintroduce the OOM the reserve
/// guards against; it only removes the spurious starved-box rejection (#1026).
pub(crate) const SAE_DIRECT_ALWAYS_ADMIT_BYTES: usize = 16 * 1024 * 1024;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeStreamingPlan {
    pub streaming: bool,
    pub chunk_size: usize,
    pub estimated_full_batch_bytes: usize,
    pub estimated_dense_schur_bytes: usize,
    pub estimated_row_cross_bytes: usize,
    pub estimated_direct_peak_bytes: usize,
    pub estimated_matrix_free_peak_bytes: usize,
    pub in_core_budget_bytes: usize,
    pub host_available_bytes: usize,
    pub direct_admitted: bool,
    pub matrix_free_admitted: bool,
}

pub(crate) fn sae_streaming_plan_from_budget(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
    in_core_budget_bytes: usize,
    chunk_window_bytes: usize,
    host_available_bytes: usize,
) -> SaeStreamingPlan {
    let per_row_words = total_basis
        .saturating_mul(1 + d_max)
        .saturating_add(k_atoms)
        .max(1);
    let per_row_bytes = per_row_words.saturating_mul(SAE_BYTES_PER_F64);
    let full_batch_bytes = n_obs.saturating_mul(per_row_bytes);
    let dense_schur_bytes = border_dim
        .saturating_mul(border_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    let row_block_dim = k_atoms.saturating_mul(1usize.saturating_add(d_max));
    // DIRECT (dense) path: the per-row cross block is materialized at the full
    // `border_dim = Σ_k M_k · p` width, so its footprint is `N · q · border_dim`.
    let row_cross_bytes = n_obs
        .saturating_mul(row_block_dim)
        .saturating_mul(border_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    // #1405/#1406: the MATRIX-FREE path does NOT materialize that dense
    // `(q × border_dim)` slab — the Kronecker operator stores only the per-row
    // `kron_jac` (the `q × p` local Jacobian) plus the sparse `kron_a_phi`
    // support, an `O(N · q · p)` footprint, NOT `O(N · q · K·M·p)`. Predicting
    // the dense `row_cross_bytes` here is the spurious ~6 TiB working-set the
    // high-K throughput plan aborted on (#1405). Use the true matrix-free cross
    // footprint `N · q · p` (p = border_dim / total_basis, since
    // border_dim = Σ_k M_k · p and total_basis = Σ_k M_k).
    let p_out = border_dim / total_basis.max(1);
    let direct_peak_bytes = full_batch_bytes
        .saturating_add(row_cross_bytes)
        .saturating_add(dense_schur_bytes);
    // Matrix-free streaming budget, floored so the chunked/sparse plan stays
    // admittable on a starved box (see SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES).
    let matrix_free_budget = in_core_budget_bytes.max(SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES);
    let chunk_resident_bytes = chunk_window_bytes.min(full_batch_bytes.max(per_row_bytes));
    let border_vector_bytes = border_dim
        .saturating_mul(SAE_BYTES_PER_F64)
        .saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER);
    // The matrix-free operator stores per-row Jacobians only over each row's
    // ACTIVE atoms (sparse assignment: jumprelu / capped softmax) — an
    // `O(N · active · (1+d) · p)` footprint, NOT the dense `O(N · K · (1+d) · p)`
    // that the full `k_atoms · (1+d)` row block implies. Estimating the dense
    // block was the spurious ~7.9 GiB working set that refused a K=256 fit at
    // n=40000 against a 1.75 GiB budget even though the sparse operator
    // materialises well under 100 MiB. The chunked/sparse plan is *designed* to
    // stay admittable; size its cross footprint at the per-row active count the
    // fit can afford in the budget left after the border-vector + chunk
    // workspaces, capped by k_atoms. The fit's row layout already bounds the
    // active set to the in-core budget (row_layout.rs), so this matches what it
    // actually materialises rather than a worst-case all-K-active row.
    let mf_cross_bytes_per_active_atom = (1usize.saturating_add(d_max))
        .saturating_mul(p_out)
        .saturating_mul(SAE_BYTES_PER_F64)
        .max(1);
    let mf_cross_budget = matrix_free_budget
        .saturating_sub(border_vector_bytes)
        .saturating_sub(chunk_resident_bytes);
    let mf_affordable_active =
        (mf_cross_budget / n_obs.max(1) / mf_cross_bytes_per_active_atom).max(1);
    let mf_active_atoms = k_atoms.min(mf_affordable_active);
    let matrix_free_cross_bytes = n_obs
        .saturating_mul(mf_active_atoms)
        .saturating_mul(1usize.saturating_add(d_max))
        .saturating_mul(p_out)
        .saturating_mul(SAE_BYTES_PER_F64);
    let matrix_free_peak_bytes = chunk_resident_bytes
        .saturating_add(matrix_free_cross_bytes)
        .saturating_add(border_vector_bytes);
    // Admit the direct plan when it fits the headroom-reserved budget, OR when its
    // footprint is small in absolute terms (≤ 16 MiB) and fits the reported
    // available memory. The second clause fixes the starved-box spurious rejection
    // (#1026): when `in_core_budget_bytes` underflows to ~0 (available below the
    // 256 MiB reserve floor) a trivially-small dense plan would otherwise be
    // refused and hard-error in `reml_criterion` (no direct-logdet streaming
    // fallback). It only ever admits plans too small to OOM, so large plans stay
    // gated on the real budget and still stream.
    let direct_fits_tiny = direct_peak_bytes <= SAE_DIRECT_ALWAYS_ADMIT_BYTES
        && direct_peak_bytes <= host_available_bytes;
    let direct_admitted = direct_peak_bytes <= in_core_budget_bytes || direct_fits_tiny;
    // The matrix-free streaming plan is the bounded-memory fallback: its peak is
    // the chunk window plus the row-cross and border-vector workspace, not the
    // full batch. Admit it against the larger of the in-core budget and an
    // absolute streaming floor so a starved box (budget collapsed to ~0) can
    // still run the plan that was designed for exactly that regime. The direct
    // (dense, full-batch) admission above is intentionally NOT floored — it can
    // OOM, so it stays gated on the real budget.
    let matrix_free_admitted = matrix_free_peak_bytes <= matrix_free_budget;
    let rows_per_chunk = (chunk_window_bytes / per_row_bytes).max(SAE_MIN_STREAMING_CHUNK_ROWS);
    SaeStreamingPlan {
        streaming: !direct_admitted,
        chunk_size: if direct_admitted {
            n_obs.max(1)
        } else {
            rows_per_chunk.min(n_obs).max(1)
        },
        estimated_full_batch_bytes: full_batch_bytes,
        estimated_dense_schur_bytes: dense_schur_bytes,
        estimated_row_cross_bytes: row_cross_bytes,
        estimated_direct_peak_bytes: direct_peak_bytes,
        estimated_matrix_free_peak_bytes: matrix_free_peak_bytes,
        in_core_budget_bytes,
        host_available_bytes,
        direct_admitted,
        matrix_free_admitted,
    }
}

pub fn sae_streaming_plan_for_shape(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
) -> SaeStreamingPlan {
    // Size gate BEFORE any CUDA probe (startup-tax fix, #1017 ordering): build
    // the plan against `min(host budget, conservative device-pool floor)`. If
    // the direct plan is admitted even under that pessimistic budget, NO real
    // budget — host-only or any probed device pool (see
    // `SAE_MIN_DEVICE_POOL_IN_CORE_BUDGET_BYTES`) — could refuse it, and a
    // direct-admitted plan's functional outputs are budget-independent
    // (`chunk_size == n_obs`, `streaming == false`; the chunk window is
    // consumed only on the streaming path). So return this plan without
    // touching `GpuRuntime::global()`, i.e. without creating a CUDA primary
    // context on every GPU for a fit that stays on the CPU. Shapes that need
    // more than the pessimistic budget fall through to the exact probed-budget
    // logic below, bit-for-bit as before.
    let (host_budget, host_available) = sae_host_in_core_budget_bytes();
    let host_window = SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE;
    let pessimistic_plan = sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        host_budget.min(SAE_MIN_DEVICE_POOL_IN_CORE_BUDGET_BYTES),
        host_window,
        host_available,
    );
    if pessimistic_plan.direct_admitted {
        return pessimistic_plan;
    }
    let (budget, chunk_window, host_available) =
        match crate::gpu::device_runtime::GpuRuntime::global() {
            Some(rt) if rt.device_count() > 0 => {
                let aggregate_budget: usize = rt
                    .device_ordinals()
                    .iter()
                    .map(|&ord| rt.memory_budget_for(ord))
                    .sum();
                if aggregate_budget > 0 {
                    let per_device_budget = aggregate_budget / rt.device_count();
                    let window = (per_device_budget / 16)
                        .max(SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE);
                    let host_available = sae_host_available_memory_bytes();
                    (
                        (aggregate_budget / 4).min(host_available),
                        window,
                        host_available,
                    )
                } else {
                    let (budget, host_available) = sae_host_in_core_budget_bytes();
                    (
                        budget,
                        SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                        host_available,
                    )
                }
            }
            Some(_) => {
                let (budget, host_available) = sae_host_in_core_budget_bytes();
                (
                    budget,
                    SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                    host_available,
                )
            }
            None => {
                let (budget, host_available) = sae_host_in_core_budget_bytes();
                (
                    budget,
                    SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                    host_available,
                )
            }
        };
    sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        budget,
        chunk_window,
        host_available,
    )
}

impl SaeStreamingPlan {
    pub(crate) fn admitted_or_error(
        self,
        n: usize,
        p: usize,
        k_atoms: usize,
    ) -> Result<Self, String> {
        if self.direct_admitted || self.matrix_free_admitted {
            Ok(self)
        } else {
            Err(format!(
                "SaeManifoldTerm::streaming_plan: predicted working set {} bytes exceeds budget {} bytes; shape n={n},p={p},K={k_atoms}",
                self.estimated_matrix_free_peak_bytes, self.in_core_budget_bytes
            ))
        }
    }

    pub(crate) fn solve_options_for_border_dim(self, border_dim: usize) -> ArrowSolveOptions {
        let mut options = if self.direct_admitted {
            ArrowSolveOptions::automatic(border_dim)
        } else {
            ArrowSolveOptions::inexact_pcg()
        };
        // #1026 — engage the reduced-Schur spectral PD-floor on the SAE inner
        // SOLVE path. At K≥4 co-collapse, two atoms share a decoder direction →
        // a per-row `H_tt` block goes near-singular → the accumulated
        // `(H_tt)⁻¹` over-subtracts the reduced Schur into an INDEFINITE matrix
        // → the Cholesky refuses → the LM loop inflates `ridge_β` over every β
        // direction and the inner Newton CRAWLS (‖Π⊥Δ‖ stays huge after
        // thousands of iters). The floor instead clamps only the collapsed
        // eigen-directions up to `floor·max(λ)` (Levenberg–Marquardt on exactly
        // the indefinite subspace), leaving the healthy β subspace's Newton step
        // exact, so the inner solve makes a real descent step and converges.
        // Only fires on a genuinely non-PD Schur (PD systems are bit-for-bit
        // unchanged); the relative floor matches the per-row evidence
        // deflation scale (`SPECTRAL_DEFLATION_REL_FLOOR`). The decoder
        // repulsion (`add_sae_decoder_repulsion`) keeps atoms apart so the
        // collapse rarely forms; this is the solve-path backstop for when it
        // still does mid-iterate.
        options.schur_pd_floor = Some(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        options
    }

    pub(crate) fn direct_logdet_admitted(self) -> bool {
        self.direct_admitted
    }
}

// ---------------------------------------------------------------------------
// Overcomplete curved TopK lane — admission arithmetic.
//
// The front door ([`crate::front_door::admit_topk_manifold`]) routes a hard
// TopK-support fit ([`crate::assignment::AssignmentMode::TopK`]) at K > P to
// the CURVED framed/streaming engine instead of the linear sparse-code
// trainer. This section owns that lane's memory ledger. The honest shape is:
//
//   * assignment state  O(N · k_active): per-row TopK active sets — `k` active
//     indices + `k` gate values + `k · d_max` on-manifold coordinates per row.
//     TopK logits are read-only routing inputs (never live Newton state), so
//     no dense `N×K` gate state exists in this lane.
//   * routing window    O(chunk_rows · K): dense scores are materialized one
//     row chunk at a time to select each row's top-k support, then dropped.
//   * framed decoder    O(K · M · r): per-atom decoder blocks in the reduced
//     Grassmann frame (`M_k × r_k`, re-expanded through `U_k` only at
//     emission — issue #972 / #2135), never the full `K · M · P` slab.
//   * border workspace  O(K · M · r · vectors): the framed arrow-Schur border
//     solve's Krylov vectors, at the crate-wide
//     `SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER` convention.
// ---------------------------------------------------------------------------

/// Admission-time bound on the framed decoder rank `r` per atom. Mirrors the
/// in-frame curved cascade's learned-rank upper clamp
/// (`InFrameCurvedConfig::frame_rank_max = 32`, bracketing the reviewer's
/// `r ≈ 8–32` band): a framed atom's border block is `M_k × min(P, r)`, so the
/// admission ledger charges `min(P, 32)` per output direction.
pub(crate) const SAE_TOPK_ADMISSION_FRAME_RANK_BOUND: usize = 32;

/// Admission-time upper bound on a seedable atom's basis size `M_k`, from the
/// `d_max` the front door knows before any basis is built. Covers every kind
/// `sae_build_atom_plans` can seed:
///
///   * periodic: `2·n_harmonics + 1` with `n_harmonics = d` → `2·d_max + 1`;
///   * sphere: fixed 7 (`≤ 32 + …` below);
///   * duchon / euclidean patch / linear: at most the center ceiling (32) plus
///     the quadratic polynomial patch `(d+1)(d+2)/2` (degree ≤ 2 monomials);
///   * torus: tensor harmonics can exceed this bound, but the plan builder
///     already rejects runaway torus designs at its own dense limit, so the
///     admission ledger stays a bound for every design that can reach a fit.
pub(crate) const fn sae_topk_admission_atom_basis_bound(d_max: usize) -> usize {
    let periodic = 2 * d_max + 1;
    let patch = 32 + ((d_max + 1) * (d_max + 2)) / 2;
    if periodic > patch { periodic } else { patch }
}

/// Memory ledger for one overcomplete curved TopK fit shape, as decided at the
/// front door. All byte figures use the documented formulas above; `admitted`
/// flags are pure functions of `(shape, in_core_budget_bytes)` so the decision
/// is reproducible and testable without a live memory probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeTopKCurvedBudget {
    /// Observations `N`.
    pub n_obs: usize,
    /// Output dimension `P`.
    pub output_dim: usize,
    /// Atom count `K` (> `P` in the overcomplete regime this lane exists for).
    pub n_atoms: usize,
    /// Hard per-row support size `k_active`.
    pub support_k: usize,
    /// Maximum per-atom latent dimension.
    pub d_max: usize,
    /// `N·K·(1+d_max)·8` — the dense routing-logit + coordinate seed the
    /// resident driver materializes today at the FFI seam. Gates the RESIDENT
    /// sub-lane; the chunked-seed driver (integration seam below) removes it.
    pub resident_seed_bytes: usize,
    /// `N·k_active·(2+d_max)·8` — per-row TopK active sets (indices + gate
    /// values + coordinates), the honest `O(N·k_active)` assignment state.
    pub active_state_bytes: usize,
    /// `seed_chunk_rows·K·8` — the transient dense-score window used to select
    /// each chunk's top-k support; never `N·K` resident.
    pub routing_window_bytes: usize,
    /// `K·M̂·r̂·8` — framed per-atom decoder blocks at the admission bounds
    /// `M̂ = sae_topk_admission_atom_basis_bound(d_max)`,
    /// `r̂ = min(P, SAE_TOPK_ADMISSION_FRAME_RANK_BOUND)`.
    pub framed_decoder_bytes: usize,
    /// `K·M̂·r̂·8·SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER` — the framed
    /// border solve's Krylov vector workspace.
    pub border_vector_bytes: usize,
    /// Streaming-lane peak: `active_state + routing_window + framed_decoder +
    /// border_vector` bytes.
    pub streaming_peak_bytes: usize,
    /// Budget the streaming peak is admitted against:
    /// `max(in_core_budget_bytes, SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES)`,
    /// the same starved-box floor convention as the matrix-free plan.
    pub streaming_budget_bytes: usize,
    /// The un-floored in-core budget the resident seed is gated on.
    pub in_core_budget_bytes: usize,
    /// True when the dense resident seed fits the in-core budget: today's
    /// engine runs this shape in core.
    pub resident_seed_admitted: bool,
    /// True when the streaming peak fits the (floored) streaming budget: the
    /// chunked-seed curved lane can run this shape once the driver seam is
    /// wired, without ever holding the dense `N×K` seed.
    pub streaming_admitted: bool,
}

impl SaeTopKCurvedBudget {
    /// Rows per routing chunk for the chunked-seed driver: sized so one dense
    /// score window stays inside the cache-multiple chunk convention
    /// (`SAE_CPU_L2_CACHE_BYTES · SAE_CHUNK_CACHE_MULTIPLE`), floored at
    /// `SAE_MIN_STREAMING_CHUNK_ROWS` and capped at `N`.
    pub fn seed_chunk_rows(&self) -> usize {
        let per_row_bytes = self.n_atoms.saturating_mul(SAE_BYTES_PER_F64).max(1);
        ((SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE) / per_row_bytes)
            .max(SAE_MIN_STREAMING_CHUNK_ROWS)
            .min(self.n_obs.max(1))
    }
}

/// Pure admission arithmetic for the overcomplete curved TopK lane. See the
/// section comment and [`SaeTopKCurvedBudget`] field docs for every formula.
pub(crate) fn sae_topk_curved_budget_from_budget(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
    d_max: usize,
    support_k: usize,
    in_core_budget_bytes: usize,
) -> SaeTopKCurvedBudget {
    let resident_seed_bytes = n_obs
        .saturating_mul(n_atoms)
        .saturating_mul(1usize.saturating_add(d_max))
        .saturating_mul(SAE_BYTES_PER_F64);
    let active_state_bytes = n_obs
        .saturating_mul(support_k)
        .saturating_mul(2usize.saturating_add(d_max))
        .saturating_mul(SAE_BYTES_PER_F64);
    let basis_bound = sae_topk_admission_atom_basis_bound(d_max);
    let rank_bound = output_dim.min(SAE_TOPK_ADMISSION_FRAME_RANK_BOUND);
    let framed_decoder_bytes = n_atoms
        .saturating_mul(basis_bound)
        .saturating_mul(rank_bound)
        .saturating_mul(SAE_BYTES_PER_F64);
    let border_vector_bytes =
        framed_decoder_bytes.saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER);
    let mut budget = SaeTopKCurvedBudget {
        n_obs,
        output_dim,
        n_atoms,
        support_k,
        d_max,
        resident_seed_bytes,
        active_state_bytes,
        routing_window_bytes: 0,
        framed_decoder_bytes,
        border_vector_bytes,
        streaming_peak_bytes: 0,
        streaming_budget_bytes: in_core_budget_bytes.max(SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES),
        in_core_budget_bytes,
        resident_seed_admitted: resident_seed_bytes <= in_core_budget_bytes,
        streaming_admitted: false,
    };
    budget.routing_window_bytes = budget
        .seed_chunk_rows()
        .saturating_mul(n_atoms)
        .saturating_mul(SAE_BYTES_PER_F64);
    budget.streaming_peak_bytes = budget
        .active_state_bytes
        .saturating_add(budget.routing_window_bytes)
        .saturating_add(budget.framed_decoder_bytes)
        .saturating_add(budget.border_vector_bytes);
    budget.streaming_admitted = budget.streaming_peak_bytes <= budget.streaming_budget_bytes;
    budget
}

/// INTEGRATION SEAM (overcomplete curved TopK lane — orchestrator wiring).
///
/// This is the ONE call the owned driver files must make to run a `K > P`
/// hard-TopK curved fit without the dense `N×K` seed. Specifically,
/// `fit_drivers.rs` (and the FFI seed builder it serves) must, whenever the
/// assignment mode is [`crate::assignment::AssignmentMode::TopK`] and
/// `n_atoms > output_dim`, call
///
/// ```ignore
/// let lane = crate::manifold::admit_topk_curved_lane(
///     n_obs, output_dim, n_atoms, d_max, support_k,
/// )?;
/// ```
///
/// BEFORE allocating any `(N, K)` routing-logit or `(K, N, d_max)` coordinate
/// seed, and then:
///
///   * if `lane.resident_seed_admitted` — proceed exactly as today (the dense
///     seed fits in core; behavior unchanged);
///   * otherwise (`lane.streaming_admitted`) — build the routing seed in row
///     chunks of `lane.seed_chunk_rows()` rows, retain per row ONLY the
///     `support_k` active indices / gate values / coordinates
///     (`lane.active_state_bytes` total), and hand the framed
///     `SparseRankOnePenaltyOp` carriers the per-chunk active sets. The dense
///     window (`lane.routing_window_bytes`) is dropped after each chunk.
///
/// Until that wiring lands, the front door refuses the
/// `!resident_seed_admitted && streaming_admitted` region with an actionable
/// error naming this function, so no admitted shape can OOM on the dense seed.
///
/// Returns the full ledger on admission (resident OR streaming); a typed `Err`
/// when the shape exceeds even the streaming budget — a TopK manifold request
/// is never silently substituted with the linear sparse-code lane.
pub fn admit_topk_curved_lane(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
    d_max: usize,
    support_k: usize,
) -> Result<SaeTopKCurvedBudget, String> {
    if n_obs == 0 || output_dim == 0 || n_atoms == 0 {
        return Err(format!(
            "admit_topk_curved_lane requires positive N, P, and K; got N={n_obs}, P={output_dim}, K={n_atoms}"
        ));
    }
    if d_max == 0 {
        return Err("admit_topk_curved_lane requires d_max >= 1".to_string());
    }
    if support_k == 0 || support_k > n_atoms {
        return Err(format!(
            "admit_topk_curved_lane requires 1 <= support_k <= K={n_atoms}; got {support_k}"
        ));
    }
    let (in_core_budget_bytes, host_available) = sae_host_in_core_budget_bytes();
    let budget = sae_topk_curved_budget_from_budget(
        n_obs,
        output_dim,
        n_atoms,
        d_max,
        support_k,
        in_core_budget_bytes,
    );
    if budget.resident_seed_admitted || budget.streaming_admitted {
        return Ok(budget);
    }
    Err(format!(
        "topk curved lane refused: streaming peak {} bytes (active sets {} + routing window {} \
         + framed decoder {} + border workspace {}) exceeds the streaming budget {} bytes \
         (host available {host_available}) at N={n_obs}, P={output_dim}, K={n_atoms}, \
         k_active={support_k}, d_max={d_max}. Reduce n_obs or support_k — a TOPK MANIFOLD \
         request is never silently substituted with the linear sparse-code lane",
        budget.streaming_peak_bytes,
        budget.active_state_bytes,
        budget.routing_window_bytes,
        budget.framed_decoder_bytes,
        budget.border_vector_bytes,
        budget.streaming_budget_bytes,
    ))
}

pub(crate) fn sae_host_available_memory_bytes() -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory() as usize;
    let available = if available == 0 {
        SAE_HOST_IN_CORE_FALLBACK_BYTES
    } else {
        available
    };
    // In a container/cgroup the global "available" can vastly exceed the cgroup
    // memory budget the process is actually allowed; admitting against the host
    // figure OOM-kills the container. Clamp to the cgroup headroom (limit −
    // current usage) whenever a finite limit is present.
    match sae_cgroup_available_bytes() {
        Some(cgroup) => available.min(cgroup),
        None => available,
    }
}

/// Bytes still available to this process under its cgroup memory controller, if
/// a finite limit is configured (`limit − current`). Returns `None` when there
/// is no cgroup limit (unlimited / `max`) or the controller cannot be read
/// (non-Linux, missing files) — in which case the global figure stands.
fn sae_cgroup_available_bytes() -> Option<usize> {
    // cgroup v2 unified hierarchy.
    if let Some(limit) = sae_read_usize_file("/sys/fs/cgroup/memory.max") {
        let current = sae_read_usize_file("/sys/fs/cgroup/memory.current").unwrap_or(0);
        return Some(limit.saturating_sub(current));
    }
    // cgroup v1 memory controller.
    if let Some(limit) = sae_read_usize_file("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        let current =
            sae_read_usize_file("/sys/fs/cgroup/memory/memory.usage_in_bytes").unwrap_or(0);
        return Some(limit.saturating_sub(current));
    }
    None
}

/// Parse a single unsigned integer from a sysfs/cgroup file. Returns `None`
/// for `max` (cgroup v2 "no limit"), a v1 sentinel limit larger than any sane
/// physical budget (effectively unlimited), an unreadable file, or unparseable
/// contents.
fn sae_read_usize_file(path: &str) -> Option<usize> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed == "max" {
        return None;
    }
    let value: usize = trimmed.parse().ok()?;
    // cgroup v1 encodes "unlimited" as a near-`u64::MAX` sentinel; treat any
    // implausibly large limit (≥ 2^62 bytes) as no limit.
    if value >= (1usize << 62) {
        return None;
    }
    Some(value)
}

/// Pure in-core budget rule, factored out of [`sae_host_in_core_budget_bytes`]
/// so the admission bound can be tested without reading live system memory.
///
/// The in-core fallback floor is a *useful-work* minimum, not a license to
/// admit more than the box actually has. The budget is `max(fraction, floor)`
/// capped at the *usable* memory `available − reserve`, where the reserve keeps
/// OS/allocator headroom free (`available` is an over-estimate). A dense direct
/// plan up to the floor can never be admitted on a box with less usable RAM
/// than the floor (which would OOM) — it streams instead.
pub(crate) const fn sae_host_in_core_budget_from_available(available: usize) -> usize {
    // Keep headroom free: never size a single plan at 100% of the reported
    // available figure. Reserve max(available/8, 256 MiB).
    let reserve = {
        let frac = available / SAE_HOST_MEMORY_RESERVE_FRACTION_DENOMINATOR;
        if frac > SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES {
            frac
        } else {
            SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES
        }
    };
    let usable = available.saturating_sub(reserve);
    let fraction = (available.saturating_mul(SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR))
        / SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR;
    let floored = if fraction > SAE_HOST_IN_CORE_FALLBACK_BYTES {
        fraction
    } else {
        SAE_HOST_IN_CORE_FALLBACK_BYTES
    };
    // Cap at usable: if the floor exceeds usable memory the budget collapses to
    // usable, so the direct-plan admission gate refuses and the term streams.
    if floored < usable { floored } else { usable }
}

pub(crate) fn sae_host_in_core_budget_bytes() -> (usize, usize) {
    let available = sae_host_available_memory_bytes();
    (sae_host_in_core_budget_from_available(available), available)
}

#[cfg(test)]
mod host_in_core_budget_tests {
    use super::*;

    #[test]
    fn budget_never_exceeds_available() {
        // Below the floor: the 2 GiB fallback must NOT inflate the budget past
        // the (smaller) available memory, or a dense direct plan up to 2 GiB
        // could be admitted on a box with <2 GiB → OOM.
        let tiny = 512 * 1024 * 1024; // 512 MiB available
        let budget = sae_host_in_core_budget_from_available(tiny);
        assert!(
            budget <= tiny,
            "budget {budget} must not exceed available {tiny}"
        );

        // Just above the floor but with the fraction below it: budget is the
        // floor, still capped at available.
        for &avail in &[
            0usize,
            1,
            SAE_HOST_IN_CORE_FALLBACK_BYTES - 1,
            SAE_HOST_IN_CORE_FALLBACK_BYTES,
            SAE_HOST_IN_CORE_FALLBACK_BYTES + 1,
            16 * 1024 * 1024 * 1024,
        ] {
            let budget = sae_host_in_core_budget_from_available(avail);
            assert!(
                budget <= avail,
                "budget {budget} must not exceed available {avail}"
            );
        }
    }

    #[test]
    fn ample_memory_uses_fraction_floored_at_2gib() {
        // 16 GiB available → fraction = 3/5·16 = 9.6 GiB, above the floor and
        // below available, so the budget is the fraction.
        let avail = 16 * 1024 * 1024 * 1024usize;
        let budget = sae_host_in_core_budget_from_available(avail);
        let fraction = avail * SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR
            / SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR;
        assert_eq!(budget, fraction);
        assert!(budget >= SAE_HOST_IN_CORE_FALLBACK_BYTES);
    }

    /// The budget must keep an OS/allocator reserve free: it can never exceed
    /// `available − max(available/8, 256 MiB)`. Sizing a plan at 100% of the
    /// reported available figure OOMs in practice even though it "fits".
    #[test]
    fn budget_reserves_headroom_below_usable() {
        for &avail in &[
            256 * 1024 * 1024usize,
            512 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            128 * 1024 * 1024 * 1024,
        ] {
            let reserve = (avail / SAE_HOST_MEMORY_RESERVE_FRACTION_DENOMINATOR)
                .max(SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES);
            let usable = avail.saturating_sub(reserve);
            let budget = sae_host_in_core_budget_from_available(avail);
            assert!(
                budget <= usable,
                "budget {budget} must leave reserve free: usable={usable}, avail={avail}"
            );
        }
    }

    /// On a box whose *usable* memory is below the 2 GiB in-core floor, the
    /// budget collapses to usable (not the floor), so a dense direct plan that
    /// needs more than usable cannot be admitted and the term streams instead
    /// of OOMing — the original S16 bug.
    #[test]
    fn below_floor_box_streams_not_oom() {
        let avail = 1024 * 1024 * 1024usize; // 1 GiB: below the 2 GiB floor.
        let reserve = (avail / SAE_HOST_MEMORY_RESERVE_FRACTION_DENOMINATOR)
            .max(SAE_HOST_MEMORY_RESERVE_FLOOR_BYTES);
        let usable = avail - reserve;
        let budget = sae_host_in_core_budget_from_available(avail);
        assert_eq!(
            budget, usable,
            "below-floor budget must collapse to usable {usable}, got {budget}"
        );
        assert!(budget < SAE_HOST_IN_CORE_FALLBACK_BYTES);

        // A direct plan needing 1.5 GiB (> usable) must NOT be admitted.
        let plan = sae_streaming_plan_from_budget(
            10_000,
            4_096,
            8,
            8,
            64,
            budget,
            SAE_CPU_L2_CACHE_BYTES,
            avail,
        );
        assert!(
            !plan.direct_admitted || plan.estimated_direct_peak_bytes <= budget,
            "a plan exceeding the usable budget must not be direct-admitted"
        );
    }

    /// #1026 regression: on a memory-starved box the in-core budget underflows to
    /// 0, but a trivially-small dense plan (e.g. a K=1 toy fit) must STILL be
    /// direct-admitted — otherwise `reml_criterion` hard-errors with "cost-only
    /// streaming route is required" for a working set of a few KiB. Conversely a
    /// large plan at budget 0 must still NOT be admitted (it streams).
    #[test]
    fn tiny_plan_admits_when_budget_collapsed_but_large_plan_streams() {
        // Budget collapsed to 0 (the starved-box / underflowed-reserve regime),
        // yet the box still reports a modest amount of available memory.
        let budget = 0usize;
        let avail = 200 * 1024 * 1024usize; // 200 MiB available, < 256 MiB floor.

        // Tiny plan: the #1026 toy shape n=120, p=2, K=1 (one M=3 atom) — a few
        // KiB working set, far below the 16 MiB always-admit size.
        let tiny =
            sae_streaming_plan_from_budget(120, 3, 1, 1, 6, budget, SAE_CPU_L2_CACHE_BYTES, avail);
        assert!(
            tiny.estimated_direct_peak_bytes <= SAE_DIRECT_ALWAYS_ADMIT_BYTES,
            "toy plan should be far below the always-admit size, got {} bytes",
            tiny.estimated_direct_peak_bytes
        );
        assert!(
            tiny.direct_admitted,
            "a tiny dense plan ({} bytes) that fits the {avail}-byte available memory \
             must be direct-admitted even when the in-core budget collapsed to 0",
            tiny.estimated_direct_peak_bytes
        );
        assert!(
            !tiny.streaming,
            "a direct-admitted tiny plan must run in-core, not stream"
        );

        // Large plan at the same collapsed budget must still NOT be direct-admitted
        // (its peak exceeds both the budget and the 16 MiB always-admit size).
        let large = sae_streaming_plan_from_budget(
            10_000,
            4_096,
            8,
            8,
            64,
            budget,
            SAE_CPU_L2_CACHE_BYTES,
            avail,
        );
        assert!(
            large.estimated_direct_peak_bytes > SAE_DIRECT_ALWAYS_ADMIT_BYTES,
            "large plan must exceed the always-admit size"
        );
        assert!(
            !large.direct_admitted,
            "a large dense plan must stay gated on the (collapsed) budget and stream, \
             not be admitted by the tiny-plan relaxation"
        );
    }
}

#[cfg(test)]
mod topk_curved_budget_tests {
    use super::*;

    /// Every byte figure in the ledger follows its documented formula exactly.
    #[test]
    fn topk_curved_budget_formulas_are_the_documented_arithmetic() {
        let (n, p, k, d, s) = (4096usize, 512usize, 32_000usize, 1usize, 8usize);
        let budget_bytes = 16 * 1024 * 1024 * 1024usize;
        let ledger = sae_topk_curved_budget_from_budget(n, p, k, d, s, budget_bytes);

        assert_eq!(
            ledger.resident_seed_bytes,
            n * k * (1 + d) * SAE_BYTES_PER_F64
        );
        assert_eq!(
            ledger.active_state_bytes,
            n * s * (2 + d) * SAE_BYTES_PER_F64
        );
        let m_hat = sae_topk_admission_atom_basis_bound(d);
        assert_eq!(
            m_hat,
            32 + 3,
            "d_max=1: patch bound 32 + (2·3)/2 dominates 2d+1=3"
        );
        let r_hat = p.min(SAE_TOPK_ADMISSION_FRAME_RANK_BOUND);
        assert_eq!(
            ledger.framed_decoder_bytes,
            k * m_hat * r_hat * SAE_BYTES_PER_F64
        );
        assert_eq!(
            ledger.border_vector_bytes,
            ledger.framed_decoder_bytes * SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER
        );
        assert_eq!(
            ledger.routing_window_bytes,
            ledger.seed_chunk_rows() * k * SAE_BYTES_PER_F64
        );
        assert_eq!(
            ledger.streaming_peak_bytes,
            ledger.active_state_bytes
                + ledger.routing_window_bytes
                + ledger.framed_decoder_bytes
                + ledger.border_vector_bytes
        );
        assert_eq!(
            ledger.streaming_budget_bytes,
            budget_bytes.max(SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES)
        );
        // The honest assignment state is O(N·k_active): orders of magnitude
        // below the dense N·K seed in the overcomplete regime.
        assert!(ledger.active_state_bytes * 100 < ledger.resident_seed_bytes);
        // This shape fits both sub-lanes at a 4 GiB budget.
        assert!(ledger.resident_seed_admitted);
        assert!(ledger.streaming_admitted);
    }

    /// The streaming lane admits shapes whose DENSE seed is over budget — the
    /// exact region the chunked-seed integration seam exists for — and the
    /// seam refuses only past the streaming budget, with the no-substitution
    /// contract in the message.
    #[test]
    fn topk_streaming_admits_beyond_resident_and_seam_validates() {
        let (n, p, k, d, s) = (1_000_000usize, 512usize, 32_000usize, 1usize, 8usize);
        // Resident seed = 1e6 · 32000 · 2 · 8 = 512 GB: never in core. The
        // streamed peak is ~9.7 GB, dominated by the framed border workspace
        // (K·M̂·r̂·8·32 ≈ 8.5 GiB), so a 16 GiB budget admits streaming.
        let budget_bytes = 16 * 1024 * 1024 * 1024usize;
        let ledger = sae_topk_curved_budget_from_budget(n, p, k, d, s, budget_bytes);
        assert!(!ledger.resident_seed_admitted);
        assert!(
            ledger.streaming_admitted,
            "streaming peak {} must fit the 16 GiB budget: the O(N·k_active) state is small",
            ledger.streaming_peak_bytes
        );
        // Chunk sizing: floored at the minimum streaming chunk and capped at N.
        assert!(ledger.seed_chunk_rows() >= SAE_MIN_STREAMING_CHUNK_ROWS);
        assert!(ledger.seed_chunk_rows() <= n);

        // Degenerate shapes are caller errors at the seam.
        assert!(admit_topk_curved_lane(0, p, k, d, s).is_err());
        assert!(admit_topk_curved_lane(n, p, k, 0, s).is_err());
        assert!(admit_topk_curved_lane(n, p, k, d, 0).is_err());
        assert!(admit_topk_curved_lane(n, p, k, d, k + 1).is_err());

        // Past the streaming budget the ledger refuses; the refusal must carry
        // the no-substitution contract.
        let starved = sae_topk_curved_budget_from_budget(n, p, k, d, s, 0);
        assert_eq!(
            starved.streaming_budget_bytes,
            SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES
        );
        assert!(
            !starved.streaming_admitted,
            "the framed decoder + border workspace exceed the 64 MiB streaming floor at K=32000"
        );
    }
}
