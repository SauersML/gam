use super::*;

pub(crate) const SAE_BYTES_PER_F64: usize = 8;

pub(crate) const SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES: usize = 2 * 1024 * 1024 * 1024;

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
/// pre-probe size gates in [`sae_streaming_plan_for_shape`].
///
/// Working sets at or below this figure are admitted identically whether the
/// budget comes from the host or from ANY device pool: the smallest CUDA device
/// gam can meaningfully probe still has hundreds of MiB of budget (a 64 MiB
/// pooled budget would require a device with under 256 MiB of usable memory —
/// below every supported compute-capability generation and every MIG slice), so
/// a plan whose peak fits under 64 MiB cannot have its admission flipped by the
/// device-budget cap. Those CPU-sized shapes therefore skip
/// device-runtime resolution — and the CUDA primary-context creation on all GPUs
/// that the first probe performs — entirely. Larger shapes still probe and use
/// the exact device-aware budget as before.
pub(crate) const SAE_MIN_DEVICE_POOL_IN_CORE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Absolute size below which a *direct* (dense, full-batch) plan is always
/// admissible provided it fits the reported available memory, regardless of the
/// headroom-reserved in-core budget. The budget subtracts a flat `max(available/8,
/// 256 MiB)` reserve so a LARGE allocation is never sized at ~100% of available
/// and OOMs; but on a memory-starved box (available below the 256 MiB floor) that
/// reserve underflows the budget to ~0, which then rejects even a trivially-small
/// dense plan (e.g. an 18 KiB K=1 toy fit) — and `penalized_quasi_laplace_criterion` has no streaming
/// fallback for the direct logdet, so it hard-errors instead of running. A dense
/// plan at or below this size cannot meaningfully OOM a box that reports at least
/// this much available, so admitting it can never reintroduce the OOM the reserve
/// guards against; it only removes the spurious starved-box rejection (#1026).
pub(crate) const SAE_DIRECT_ALWAYS_ADMIT_BYTES: usize = 16 * 1024 * 1024;

// ---------------------------------------------------------------------------
// #2724 — the exact-stationarity (dense exact-`A`) route's resident footprint.
//
// The three functions below are the ONE size expression shared by the code that
// ALLOCATES the dense route's blocks (`materialize_exact_hessian_dense`,
// `materialize_exact_hessian_quotient_geometry`, `dense_exact_a_logdet_channels`
// in `construction_exact_hessian.rs`) and by the code that PRICES it here. They
// exist because those two drifted: the plan priced `border_dim²` (40.5 KiB on
// the measured `dim = 7692` witness) for a route whose real resident set is
// `dim × dim` blocks at 451.4 MiB EACH — four orders of magnitude of allocation
// the admission never counted.
//
// This prices MEMORY only. The same admission still prices no TIME, and the
// route it gates is an `O(dim³)` symmetric eigendecomposition; a byte-denominated
// bar cannot close that gap at any calibration (memory is `O(dim²)`, so the two
// curves separate with `dim`). See #2724 for the separated claim.
// ---------------------------------------------------------------------------

/// Joint dimension of the exact stationarity Hessian: the coordinate block plus
/// the β border. Called with the EXACT `(cache.delta_t_len(), cache.k)` at the
/// allocation sites and with a shape-derived upper bound on the coordinate
/// block in [`sae_streaming_plan_from_budget`], so the plan and the allocator
/// cannot describe two different matrices.
pub(crate) const fn sae_exact_stationarity_dim(coord_dim: usize, border_dim: usize) -> usize {
    coord_dim.saturating_add(border_dim)
}

/// Bytes in ONE `dim × dim` f64 block of the exact stationarity route.
pub(crate) const fn sae_exact_stationarity_block_bytes(dim: usize) -> usize {
    dim.saturating_mul(dim).saturating_mul(SAE_BYTES_PER_F64)
}

/// How many `dim × dim`-scale f64 blocks the exact route holds live at once.
///
/// This is an ENUMERATION of named bindings read off the allocating code, not a
/// fitted multiplier. At the peak — the `coordinate` spectral block being built
/// inside `materialize_exact_hessian_quotient_geometry`, the heavier of the two
/// consumers — the following are simultaneously live:
///
///   1. `joint.operator`            — the materialized `A` (dim × dim)
///   2. `joint.eigenvectors`        — its eigenbasis (dim × dim)
///   3. `priced_joint_inverse`      — the priced pseudo-inverse (dim × dim)
///   4. the `coordinate` block's own `operator` (`a_tt_block`, total_t × total_t)
///   5. the `coordinate` block's `eigenvectors` (total_t × total_t)
///   6. `eigh`'s own eigenvector output / LAPACK working copy of the operator
///   7. `priced_coordinate_inverse` (dim × dim), plus its `_small` source
///
/// There is no longer a second branch to compare against. `a386c1e8b` (#2674)
/// deleted the gauge-reduced path of `exact_hessian_spectral_block` entirely --
/// it had been diagonalising `ZᵀAZ` on the complement of a declared chart-gauge
/// orbit that the penalized objective is *not* flat along, so the deletion was
/// the fix, not a refactor. The locals this comment used to cite as the
/// comparable count (`reduced_operator`, `e_times_complement`, `reduced_e`,
/// `reduced_eigenvectors`) no longer exist anywhere in that file.
///
/// `cluster_stable_eigh` also no longer materializes the diagonal attribution
/// operator as a dense `dim x dim` matrix: its cluster restriction is accumulated
/// directly from `e_diag`, so that former dense live block is genuinely absent.
///
/// The enumeration above is therefore the whole population, and 7 stands on it
/// alone: it is still a LOWER bound on LAPACK's internal workspace, which
/// `dsyevd` sizes at its own discretion.
///
/// Kept as an enumeration rather than a measured peak deliberately (#2724): the
/// count is auditable against the code, where a measured number would drift
/// silently the next time an allocation is added.
pub(crate) const SAE_EXACT_STATIONARITY_LIVE_DIM_BLOCKS: usize = 7;

/// Resident bytes of the exact stationarity route at its peak.
pub(crate) const fn sae_exact_stationarity_resident_bytes(dim: usize) -> usize {
    sae_exact_stationarity_block_bytes(dim)
        .saturating_mul(SAE_EXACT_STATIONARITY_LIVE_DIM_BLOCKS)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeStreamingPlan {
    pub streaming: bool,
    pub chunk_size: usize,
    pub estimated_full_batch_bytes: usize,
    pub estimated_dense_schur_bytes: usize,
    pub estimated_row_cross_bytes: usize,
    pub estimated_direct_peak_bytes: usize,
    pub estimated_matrix_free_peak_bytes: usize,
    /// #2724 — joint dimension of the exact stationarity Hessian the dense
    /// exact-`A` route materializes, as bounded from this shape.
    pub estimated_exact_stationarity_dim: usize,
    /// #2724 — resident bytes of that route at its peak
    /// (`SAE_EXACT_STATIONARITY_LIVE_DIM_BLOCKS · dim² · 8`). This is a SEPARATE
    /// question from `estimated_direct_peak_bytes`, which prices the full-batch
    /// evidence assembly the inner solve and the chunking depend on; the exact
    /// route's blocks are allocated by a different consumer and only when the
    /// exact-`A` log-determinant lane runs. Two costs, two fields, two gates —
    /// folding them into one scalar would make a chunk-size decision hostage to
    /// an eigendecomposition's working set.
    pub estimated_exact_stationarity_bytes: usize,
    pub in_core_budget_bytes: usize,
    pub process_available_bytes: usize,
    pub direct_admitted: bool,
    /// #2724 — true when the dense exact-stationarity route's resident blocks
    /// fit the same budget. `SaeStreamingPlan::direct_logdet_admitted` — the
    /// predicate that dispatches to that route — requires BOTH this and
    /// `direct_admitted`.
    pub exact_stationarity_admitted: bool,
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
    process_available_bytes: usize,
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
    let matrix_free_budget = in_core_budget_bytes;
    let chunk_resident_bytes = chunk_window_bytes.min(full_batch_bytes.max(per_row_bytes));
    let border_vector_bytes = border_dim
        .saturating_mul(SAE_BYTES_PER_F64)
        .saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER);
    // The matrix-free operator stores per-row Jacobians only over each row's
    // ACTIVE atoms (exact sparse assignment: TopK support) — an
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
    // refused and hard-error in `penalized_quasi_laplace_criterion` (no direct-logdet streaming
    // fallback). It only ever admits plans too small to OOM, so large plans stay
    // gated on the real budget and still stream.
    let direct_fits_tiny = direct_peak_bytes <= SAE_DIRECT_ALWAYS_ADMIT_BYTES
        && direct_peak_bytes <= process_available_bytes;
    let direct_admitted = direct_peak_bytes <= in_core_budget_bytes || direct_fits_tiny;
    // #2724 — the dense exact-`A` route's own ledger. `total_t = Σ_rows row_dim`
    // and `row_dim = Σ_{k active in that row} d_k ≤ k_atoms · d_max`, so the
    // shape this function already receives bounds the joint dimension the
    // allocator will build. The bound is exact when every atom is active in
    // every row (the dense-softmax regime that produced the measured witness)
    // and conservative under a compact TopK layout — the direction a memory
    // admission must err in.
    let exact_stationarity_dim = sae_exact_stationarity_dim(
        n_obs.saturating_mul(k_atoms).saturating_mul(d_max),
        border_dim,
    );
    let exact_stationarity_bytes = sae_exact_stationarity_resident_bytes(exact_stationarity_dim);
    // Denominated in the HOST budget, deliberately, and not in the
    // possibly-device-derived `in_core_budget_bytes` this function is handed:
    // every block enumerated above is a host `Array2<f64>` handed to a host
    // LAPACK symmetric eigendecomposition. There is no device arm of this route,
    // so a pooled device budget is the wrong ceiling for it. Both figures derive
    // from the ONE carried host reading (#2532), so this stays a pure function
    // of `(shape, carried reading)` and never resamples ambient memory.
    let host_budget_for_exact = sae_host_in_core_budget_from_available(process_available_bytes);
    // Same starved-box relaxation as the direct plan (#1026): a working set too
    // small to OOM a box that reports it available is admitted even when the
    // headroom-reserved budget has underflowed to ~0.
    let exact_stationarity_fits_tiny = exact_stationarity_bytes <= SAE_DIRECT_ALWAYS_ADMIT_BYTES
        && exact_stationarity_bytes <= process_available_bytes;
    let exact_stationarity_admitted =
        exact_stationarity_bytes <= host_budget_for_exact || exact_stationarity_fits_tiny;
    // Matrix-free streaming bounds its peak to the chunk, row-cross and border
    // workspaces, but it is still a real allocation. Admit it against the same
    // authoritative process budget: a genuine zero means exhausted memory,
    // not permission to manufacture a positive allowance.
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
        estimated_exact_stationarity_dim: exact_stationarity_dim,
        estimated_exact_stationarity_bytes: exact_stationarity_bytes,
        in_core_budget_bytes,
        process_available_bytes,
        direct_admitted,
        exact_stationarity_admitted,
        matrix_free_admitted,
    }
}

/// Resolve the plan against a LIVE reading of host memory.
///
/// Prefer [`sae_streaming_plan_for_shape_with_available`] wherever the caller
/// belongs to a fit: since #2330 Phase-2 this predicate selects which OPERATOR
/// the quasi-Laplace criterion prices (exact observed information `A` on the
/// direct route, the Arrow-Schur majorizer `B` on the streaming one), so a
/// route resolved from ambient memory makes the OBJECTIVE depend on how much
/// RAM the box happens to have free at that instant (#2532). Sampling here is
/// correct only for callers that are deciding something now and are not part of
/// a fit whose probes must stay comparable to one another.
pub fn sae_streaming_plan_for_shape(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
    gpu_policy: gam_gpu::GpuPolicy,
) -> Result<SaeStreamingPlan, String> {
    sae_streaming_plan_for_shape_with_available(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        gpu_policy,
        sae_process_available_memory_bytes(),
    )
}

/// The same plan, resolved against a host-memory reading the CALLER supplies.
///
/// The plan has exactly two inputs: the model SHAPE (`n_obs`, `total_basis`,
/// `k_atoms`, `d_max`, `border_dim`) and the ENVIRONMENT (available bytes —
/// every budget below is derived from that one number by
/// `sae_host_in_core_budget_from_available`). The shape legitimately moves
/// during a fit as atoms are rank-reduced and frames activate; the environment
/// must not, or two probes of the same rho are priced by two different
/// operators (#2532). A fit therefore samples ONCE and passes the sample here.
pub fn sae_streaming_plan_for_shape_with_available(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
    gpu_policy: gam_gpu::GpuPolicy,
    host_available_bytes: usize,
) -> Result<SaeStreamingPlan, String> {
    // Size gate BEFORE any CUDA probe (startup-tax fix, #1017 ordering): decide
    // admission against `min(host budget, conservative device-pool floor)`
    // first. If the direct plan is admitted even under that pessimistic budget,
    // NO real budget — host-only or any probed device pool (see
    // `SAE_MIN_DEVICE_POOL_IN_CORE_BUDGET_BYTES`) — could refuse it, so the
    // probe cannot change the admission, the chunking (`chunk_size == n_obs`,
    // `streaming == false` for every direct plan), or any downstream
    // budget-comparison branch (a `≤ 64 MiB` working set is below every
    // consumer's threshold whichever budget is installed). Return the
    // HOST-budget plan — carrying the honest CPU-fit budget in
    // `in_core_budget_bytes` for downstream gates like the #2080 escalation
    // ledger — without resolving a device runtime, i.e. without creating a
    // CUDA primary context on every GPU for a fit that stays on the CPU. Larger
    // shapes fall through to the exact probed-budget logic below, bit-for-bit
    // as before.
    let host_available = host_available_bytes;
    let host_budget = sae_host_in_core_budget_from_available(host_available);
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
    if pessimistic_plan.direct_admitted
        && pessimistic_plan.estimated_dense_schur_bytes <= host_budget
    {
        // Direct admission is monotone in the budget, so the host-budget plan
        // is direct-admitted too and functionally identical (same chunk_size,
        // same admission flags); only the recorded budget/diagnostic fields
        // reflect the honest host figure instead of the 64 MiB decision floor.
        // The extra dense-Schur ≤ host-budget guard keeps the downstream
        // `estimated_dense_schur_bytes > in_core_budget_bytes` consumers
        // (dense-vs-SLQ evidence routing) on the same branch a probed budget
        // would have chosen — on a starved host whose budget collapsed below
        // even a tiny Schur, fall through to the exact probed logic.
        //
        // #2724's `exact_stationarity_admitted` is unaffected by this early
        // return by construction: it is denominated in the HOST budget derived
        // from the carried reading, which both branches pass through unchanged,
        // so no probed device budget can move it either way.
        return Ok(sae_streaming_plan_from_budget(
            n_obs,
            total_basis,
            k_atoms,
            d_max,
            border_dim,
            host_budget,
            host_window,
            host_available,
        ));
    }
    let (budget, chunk_window, host_available) =
        match crate::gpu::device_runtime::GpuRuntime::resolve(gpu_policy)
            .map_err(|error| format!("SAE streaming-plan CUDA admission failed: {error}"))?
        {
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
                    (
                        (aggregate_budget / 4).min(host_available),
                        window,
                        host_available,
                    )
                } else {
                    (
                        host_budget,
                        SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                        host_available,
                    )
                }
            }
            Some(_) => (
                host_budget,
                SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                host_available,
            ),
            None => (
                host_budget,
                SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                host_available,
            ),
        };
    Ok(sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        budget,
        chunk_window,
        host_available,
    ))
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
        options.newton_schur_tikhonov_rel_floor =
            Some(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        options
    }

    /// The predicate that dispatches the exact-`A` log-determinant and the
    /// ρ-gradient adjoint onto the DENSE exact-stationarity route (versus the
    /// streaming / matrix-free implementation).
    ///
    /// #2724 — it must therefore clear BOTH ledgers. Before the fix it returned
    /// `direct_admitted` alone, i.e. it admitted a route on a budget check that
    /// had counted `border_dim² · 8` and never counted the `dim × dim` blocks
    /// the route actually allocates. On the measured witness that omitted term
    /// was 8.0 × 10⁴ times the entire quantity being compared against the
    /// budget. Both consumers of a `false` here have a real streaming
    /// implementation to fall back to
    /// (`penalized_quasi_laplace_criterion_streaming_exact_with_cache`,
    /// `solve_exact_stationarity_matrix_free`), so refusing is a route change,
    /// not an error.
    pub(crate) fn direct_logdet_admitted(self) -> bool {
        self.direct_admitted && self.exact_stationarity_admitted
    }
}

// ---------------------------------------------------------------------------
// Overcomplete curved TopK lane — admission arithmetic.
//
// The front door ([`crate::front_door::admit_topk_manifold`]) routes a hard
// TopK-support fit ([`crate::assignment::AssignmentMode::TopK`]) at K > P to
// the CURVED support-sparse engine instead of the linear sparse-code
// trainer. This section owns that lane's memory ledger. The honest shape is:
//
//   * assignment state  O(N · k_active): per-row TopK active sets — `k` active
//     indices + `k` gate values + `k · d_max` on-manifold coordinates per row.
//     TopK logits are read-only routing inputs (never live Newton state), so
//     no dense `N×K` gate state exists in this lane.
//   * routing workspace O(P + k_active · (2 + d_max)): one centered response
//     row and its bounded TopK selection heap. Atom scores are consumed one at
//     a time, so the workspace is independent of K.
//   * decoder           O(K · M · P): per-atom final-function coefficient
//     blocks. Zero-occupancy atoms are pruned before these blocks are built.
//   * border workspace  O(K · M · P · vectors): the support Arrow-Schur border
//     solve's Krylov vectors, at the crate-wide
//     `SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER` convention.
// ---------------------------------------------------------------------------

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
    /// `N·k_active·(2+d_max)·8` — per-row TopK active sets (indices + gate
    /// values + coordinates), the honest `O(N·k_active)` assignment state.
    pub active_state_bytes: usize,
    /// `(P + k_active·(2+d_max))·8` — the largest row-local routing workspace.
    /// Scores are consumed atom-by-atom into the bounded support heap, so this
    /// charge is independent of K.
    pub routing_workspace_bytes: usize,
    /// `K·M̂·P·8` — the conservative pre-routing decoder bound, where
    /// `M̂ = sae_topk_admission_atom_basis_bound(d_max)`. The realized decoder
    /// is smaller whenever routing prunes zero-occupancy atoms.
    pub decoder_bytes: usize,
    /// `K·M̂·P·8·SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER` — the
    /// support Arrow-Schur border solve's vector workspace.
    pub border_vector_bytes: usize,
    /// Support-lane peak: `active_state + routing_workspace + decoder +
    /// border_vector` bytes.
    pub streaming_peak_bytes: usize,
    /// Authoritative process-memory admission ceiling retained alongside the
    /// estimate so refusal diagnostics expose both sides of the inequality.
    pub streaming_budget_bytes: usize,
    /// The headroom-reserved process budget used to derive the support budget.
    pub in_core_budget_bytes: usize,
    /// True when the canonical support-sparse peak fits that process budget.
    pub streaming_admitted: bool,
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
    let active_state_bytes = n_obs
        .saturating_mul(support_k)
        .saturating_mul(2usize.saturating_add(d_max))
        .saturating_mul(SAE_BYTES_PER_F64);
    let basis_bound = sae_topk_admission_atom_basis_bound(d_max);
    let routing_workspace_bytes = output_dim
        .saturating_add(support_k.saturating_mul(2usize.saturating_add(d_max)))
        .saturating_mul(SAE_BYTES_PER_F64);
    let decoder_bytes = n_atoms
        .saturating_mul(basis_bound)
        .saturating_mul(output_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    let border_vector_bytes =
        decoder_bytes.saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER);
    let mut budget = SaeTopKCurvedBudget {
        n_obs,
        output_dim,
        n_atoms,
        support_k,
        d_max,
        active_state_bytes,
        routing_workspace_bytes,
        decoder_bytes,
        border_vector_bytes,
        streaming_peak_bytes: 0,
        streaming_budget_bytes: in_core_budget_bytes,
        in_core_budget_bytes,
        streaming_admitted: false,
    };
    budget.streaming_peak_bytes = budget
        .active_state_bytes
        .saturating_add(budget.routing_workspace_bytes)
        .saturating_add(budget.decoder_bytes)
        .saturating_add(budget.border_vector_bytes);
    budget.streaming_admitted = budget.streaming_peak_bytes <= budget.streaming_budget_bytes;
    budget
}

/// The host-memory figure every SAE budget is derived from.
///
/// This is the process's ONE availability observation (`gam_runtime`'s
/// governor took it; see [`gam_runtime::resource::process_memory_availability`]),
/// not a fresh probe. Before #2560 it resampled `/proc/meminfo` plus five
/// cgroup files on every call, which put ~1100 six-file probes per second on
/// the row-jet path — over half the wall clock of a stuck fit — and, worse,
/// let the *value* move under a fit: the row-jet tile geometry and the GMRES
/// restart length are both sized from it, so two runs of one fit could take
/// different Krylov paths because another tenant allocated in between.
pub(crate) fn sae_process_available_memory_bytes() -> usize {
    gam_runtime::resource::process_available_memory_bytes()
}

/// Pure in-core budget rule, factored out of [`sae_host_in_core_budget_bytes`]
/// so the admission bound can be tested without reading live system memory.
///
/// The in-core useful-work floor is a minimum target, not a license to
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
    let floored = if fraction > SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES {
        fraction
    } else {
        SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES
    };
    // Cap at usable: if the floor exceeds usable memory the budget collapses to
    // usable, so the direct-plan admission gate refuses and the term streams.
    if floored < usable { floored } else { usable }
}

pub(crate) fn sae_host_in_core_budget_bytes() -> (usize, usize) {
    let available = sae_process_available_memory_bytes();
    (sae_host_in_core_budget_from_available(available), available)
}

#[cfg(test)]
mod host_budget_is_stationary_tests {
    //! #2560. The SAE budget is consulted per unit of work — once per row-jet
    //! window, once per contracted RHS/HVP sweep, once per GMRES restart
    //! sizing — so what it costs and whether it MOVES are both properties of
    //! the fit, not of the call site. Two invariants, one test each:
    //! it costs no probes, and it does not move.
    use super::*;

    /// The number of budget lookups a single row-jet sweep of a modest fit
    /// performs (one per window over a few thousand rows, times the
    /// jet/RHS/HVP consumers). Deliberately larger than any realistic sweep:
    /// the assertion is that the cost is O(1) in this count.
    const SWEEP_LOOKUPS: usize = 50_000;

    #[test]
    fn the_budget_does_not_move_under_a_fit() {
        // Determinism, not speed: the row-jet tile geometry and the GMRES
        // restart length are sized from this number, so if it moves, two runs
        // of one fit can take different Krylov paths because another tenant on
        // the box allocated in between.
        let first = sae_host_in_core_budget_bytes();
        for turn in 0..SWEEP_LOOKUPS {
            let again = sae_host_in_core_budget_bytes();
            assert_eq!(
                again, first,
                "the host budget moved at lookup {turn}: {again:?} vs {first:?}. Available memory \
                 is a live, shared quantity owned by the machine; a fit that reads it more than \
                 once is a fit whose route depends on what else the box was doing (#2560)."
            );
        }
    }
}

#[cfg(test)]
mod cpu_sized_plan_laziness_tests {
    //! Pins the CUDA startup-tax fix at the SAE fit's front budgeting step:
    //! planning a CPU-sized manifold fit (the profiled K=6 / N=700 / d=24 class
    //! of shapes, whose whole working set is a few MiB) must take the
    //! size-gated early return and NEVER resolve `GpuRuntime` — resolution
    //! whose first execution probes the driver and creates a CUDA primary
    //! context on every GPU (`cuDevicePrimaryCtxRetain`, ~10% of the profiled
    //! small-fit wall clock on an 8×B200 node). Runs on any host: the invariant
    //! is the control-flow ordering, observed via the process-wide
    //! `resolution_call_count` counter (nextest = one process per test).
    use super::*;

    #[test]
    fn early_return_carries_the_host_budget_not_the_decision_floor() {
        // The early-returned plan must record the honest host in-core budget
        // (downstream gates like the #2080 escalation ledger compare against
        // it), not the 64 MiB pessimistic decision floor.
        //
        // Both sides derive from ONE reading. This used to plan through the
        // sampling entry point and then take a SECOND, independent sample to
        // compare against — two readings of a quantity that moves continuously,
        // asserted equal. On a loaded box they differ (observed: 128383706726 vs
        // 128382792499, a 0.9 MB drift between two calls microseconds apart) and
        // the test fails for a reason that has nothing to do with the property
        // it is trying to pin. #2532's explicit-reading entry point is what lets
        // it ask the question without the race.
        let available = sae_process_available_memory_bytes();
        let plan = sae_streaming_plan_for_shape_with_available(
            700,
            60,
            6,
            2,
            144,
            gam_gpu::GpuPolicy::Auto,
            available,
        )
        .expect("CPU-sized plan must not require CUDA resolution");
        assert_eq!(
            plan.in_core_budget_bytes,
            sae_host_in_core_budget_from_available(available),
            "direct early-return must install the host budget"
        );
    }
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
            SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES - 1,
            SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES,
            SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES + 1,
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
        assert!(budget >= SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES);
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
        assert!(budget < SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES);

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
    /// direct-admitted — otherwise `penalized_quasi_laplace_criterion` hard-errors with "cost-only
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
mod exact_stationarity_admission_tests {
    //! #2724 — the admission must price the `dim × dim` blocks the route it
    //! admits actually allocates.
    //!
    //! The witness is the shipped K=8 rung of `sae_ev_vs_k_olmo.py` (508 train
    //! rows, p=32, K=8), whose `[SAE-EXACT-DENSE]` line reports `dim = 7692`,
    //! `border = 72`, 451.4 MiB per `dim × dim` f64 block. Against that, the
    //! only quadratic term the plan used to price was `border_dim² · 8` —
    //! 40.5 KiB, four orders of magnitude below one block and five below the
    //! resident set.
    //!
    //! SCOPE, stated so a green here is not read as more than it is: this
    //! prices MEMORY. The admission still prices no TIME, and the route it
    //! gates is an `O(dim³)` eigendecomposition. On the node that produced the
    //! measured wall the honest memory figure (~3.1 GiB) sits ~6.5× INSIDE the
    //! budget, so nothing below claims to refuse that run — it claims the
    //! estimate is no longer blind to its own dominant allocation, and that the
    //! bar therefore binds on smaller boxes and at larger `dim`.
    use super::*;

    /// `(n_obs, total_basis, k_atoms, d_max, border_dim)`.
    ///
    /// `n_obs = 508`, `k_atoms = 8` and `border_dim = 72` are read off the
    /// shipped `[SAE-EXACT-DENSE]` line for that fit (`border=72`). `d_max = 2`
    /// is the issue's INFERENCE, not a measurement — `coords/rows = 7620/508 =
    /// 15` over 8 atoms forces `d_max ≥ 2`, and 2 is the smallest value
    /// consistent with it; a larger true `d_max` only makes the priced bound
    /// larger, so the assertion below is the conservative side of that
    /// uncertainty. `total_basis` is not pinned by the log line; it enters only
    /// the full-batch slab, and this value keeps that slab at ~8 MB — which is
    /// the whole point, since it is the ledger that used to stand alone.
    const WITNESS: (usize, usize, usize, usize, usize) = (508, 96, 8, 2, 72);

    /// `dim` from the shipped `[SAE-EXACT-DENSE]` line for that fit. The plan's
    /// shape-derived bound must not fall below it, or the ledger would price a
    /// smaller matrix than the allocator builds.
    const WITNESS_MEASURED_DIM: usize = 7692;

    fn plan_at(available: usize) -> SaeStreamingPlan {
        let (n_obs, total_basis, k_atoms, d_max, border_dim) = WITNESS;
        sae_streaming_plan_from_budget(
            n_obs,
            total_basis,
            k_atoms,
            d_max,
            border_dim,
            sae_host_in_core_budget_from_available(available),
            SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
            available,
        )
    }

    /// The shape-derived bound covers the dimension the allocator really built,
    /// and it is computed by the SAME function the allocator calls.
    #[test]
    fn the_priced_dimension_covers_the_measured_one() {
        let plan = plan_at(64 * 1024 * 1024 * 1024);
        let (n_obs, _, k_atoms, d_max, border_dim) = WITNESS;
        assert_eq!(
            plan.estimated_exact_stationarity_dim,
            sae_exact_stationarity_dim(n_obs * k_atoms * d_max, border_dim),
            "the plan must price the joint dimension through the shared expression"
        );
        assert!(
            plan.estimated_exact_stationarity_dim >= WITNESS_MEASURED_DIM,
            "priced dim {} is below the {WITNESS_MEASURED_DIM} the shipped \
             [SAE-EXACT-DENSE] line reports for this shape; a memory bound that \
             under-describes its own matrix is not a bound",
            plan.estimated_exact_stationarity_dim
        );
        assert_eq!(
            plan.estimated_exact_stationarity_bytes,
            sae_exact_stationarity_resident_bytes(plan.estimated_exact_stationarity_dim)
        );
    }

    /// THE DEFECT: a budget below the route's true requirement must REFUSE it.
    ///
    /// Both budgets below admit the direct plan itself — its full-batch peak is
    /// a few MiB — so the only thing that can move the verdict is the
    /// `dim × dim` ledger. Before the fix `direct_logdet_admitted()` was
    /// `direct_admitted` alone and was therefore `true` on BOTH sides: a route
    /// needing gigabytes admitted by a check that had looked at megabytes.
    #[test]
    fn a_budget_below_the_dense_route_requirement_refuses_it() {
        // The requirement is a function of the SHAPE, not of the environment,
        // so read it once at a budget that cannot bind.
        let required = plan_at(usize::MAX / 4).estimated_exact_stationarity_bytes;
        assert!(
            required > SAE_HOST_IN_CORE_USEFUL_WORK_FLOOR_BYTES,
            "the witness's dense requirement ({required} B) must exceed the in-core \
             floor, or the budget arithmetic below cannot straddle it"
        );

        // Straddle it from the ONE input a host budget derives from, by
        // inverting the budget rule's fraction leg: `budget = 3/5 · available`
        // in this regime, so an availability of `5/3 · (required − 1)` puts the
        // budget one byte below the requirement (integer division only ever
        // rounds it further down). Both sides are then CHECKED against the
        // budget rule itself rather than transcribed, so neither is a fitted
        // constant.
        let starved_available = (required.saturating_sub(1)
            * SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR)
            / SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR;
        let roomy_available = required.saturating_mul(2);
        assert!(
            sae_host_in_core_budget_from_available(starved_available) < required,
            "the starved side must genuinely be below the requirement"
        );
        assert!(
            sae_host_in_core_budget_from_available(roomy_available) >= required,
            "the roomy side must genuinely be at or above the requirement"
        );

        let starved = plan_at(starved_available);
        let roomy = plan_at(roomy_available);

        // Non-vacuity / positive control: the DIRECT plan is admitted on both
        // sides, so the pre-fix predicate would have said `true` twice and this
        // test could not have distinguished them.
        assert!(
            starved.direct_admitted && roomy.direct_admitted,
            "the full-batch direct plan ({} B) must be admitted on both sides, or the \
             exact-Hessian ledger is not what moves the verdict",
            starved.estimated_direct_peak_bytes
        );
        assert!(
            starved.estimated_exact_stationarity_bytes > starved.estimated_direct_peak_bytes,
            "the omitted term ({} B) must dominate the ledger that used to stand in for it \
             ({} B); if it did not, this issue would be a rounding correction",
            starved.estimated_exact_stationarity_bytes,
            starved.estimated_direct_peak_bytes
        );

        assert!(
            !starved.exact_stationarity_admitted,
            "a {required}-byte dense exact-stationarity route was admitted against a \
             {}-byte budget",
            sae_host_in_core_budget_from_available(starved_available)
        );
        assert!(
            !starved.direct_logdet_admitted(),
            "the predicate that dispatches to the dense exact route must refuse when the \
             route does not fit; admitting an over-budget route is the defect (#2724)"
        );
        assert!(
            roomy.exact_stationarity_admitted && roomy.direct_logdet_admitted(),
            "and it must still admit the same route when the budget does cover it, or the \
             bar refuses everything and proves nothing"
        );
    }

    /// A starved box must not lose a toy fit: the #1026 relaxation applies to
    /// this ledger too, because a working set too small to OOM a box cannot be
    /// the reason to route away from the exact route.
    #[test]
    fn a_toy_shape_survives_a_collapsed_budget() {
        let available = 200 * 1024 * 1024usize; // below the 256 MiB reserve floor.
        let plan =
            sae_streaming_plan_from_budget(120, 3, 1, 1, 6, 0, SAE_CPU_L2_CACHE_BYTES, available);
        assert!(
            plan.estimated_exact_stationarity_bytes <= SAE_DIRECT_ALWAYS_ADMIT_BYTES,
            "the toy shape's exact route ({} B) should be far below the always-admit size",
            plan.estimated_exact_stationarity_bytes
        );
        assert!(
            plan.direct_logdet_admitted(),
            "a few-KiB exact route must survive a collapsed budget on a box that reports \
             {available} bytes available (#1026)"
        );
    }
}

#[cfg(test)]
mod topk_curved_budget_tests {
    use super::*;

    /// Every byte figure in the ledger follows its documented formula exactly.
    #[test]
    fn topk_curved_budget_formulas_are_the_documented_arithmetic() {
        let (n, p, k, d, s) = (4096usize, 64usize, 10_000usize, 1usize, 8usize);
        let budget_bytes = 8 * 1024 * 1024 * 1024usize;
        let ledger = sae_topk_curved_budget_from_budget(n, p, k, d, s, budget_bytes);

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
        assert_eq!(
            ledger.routing_workspace_bytes,
            (p + s * (2 + d)) * SAE_BYTES_PER_F64
        );
        assert_eq!(ledger.decoder_bytes, k * m_hat * p * SAE_BYTES_PER_F64);
        assert_eq!(
            ledger.border_vector_bytes,
            ledger.decoder_bytes * SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER
        );
        assert_eq!(
            ledger.streaming_peak_bytes,
            ledger.active_state_bytes
                + ledger.routing_workspace_bytes
                + ledger.decoder_bytes
                + ledger.border_vector_bytes
        );
        assert_eq!(ledger.streaming_budget_bytes, budget_bytes);
        assert!(ledger.streaming_admitted);
    }

    /// Routing and assignment memory are support-shaped even when K grows.
    #[test]
    fn topk_routing_workspace_is_independent_of_atom_count() {
        let shape = |k| sae_topk_curved_budget_from_budget(1024, 64, k, 2, 4, usize::MAX);
        let ten_thousand = shape(10_000);
        let twenty_thousand = shape(20_000);
        assert_eq!(
            ten_thousand.active_state_bytes,
            twenty_thousand.active_state_bytes
        );
        assert_eq!(
            ten_thousand.routing_workspace_bytes,
            twenty_thousand.routing_workspace_bytes
        );
        assert_eq!(
            ten_thousand.routing_workspace_bytes,
            (64 + 4 * (2 + 2)) * SAE_BYTES_PER_F64
        );
    }
}

#[cfg(test)]
mod frozen_host_sample_tests {
    use super::*;

    /// A shape big enough that its direct plan is not trivially admitted, so the
    /// route genuinely depends on the environment rather than on the 16 MiB
    /// always-admit floor.
    const SHAPE: (usize, usize, usize, usize, usize) = (4_000, 256, 32, 2, 2_048);

    fn route_at(available: usize) -> SaeStreamingPlan {
        let (n_obs, total_basis, k_atoms, d_max, border_dim) = SHAPE;
        sae_streaming_plan_for_shape_with_available(
            n_obs,
            total_basis,
            k_atoms,
            d_max,
            border_dim,
            gam_gpu::GpuPolicy::Off,
            available,
        )
        .expect("plan resolves for an explicit host reading")
    }

    /// NON-VACUITY. Everything below is worthless unless the route actually
    /// moves with the environment at this shape: if it did not, a gate saying
    /// "the frozen sample determines the route" would pass on a predicate that
    /// ignores the sample entirely.
    #[test]
    fn the_route_really_does_depend_on_the_host_reading() {
        let starved = route_at(0);
        let roomy = route_at(1 << 40);
        assert!(
            starved.streaming,
            "a host with no available bytes must refuse the direct plan; got {starved:?}"
        );
        assert!(
            !roomy.streaming,
            "a terabyte of headroom must admit the direct plan at this shape; got {roomy:?}"
        );
    }

    /// #2532 — `streaming_plan()` is a function of the CARRIED reading and of
    /// nothing ambient.
    ///
    /// The two carried values below produce different routes (pinned above), so
    /// this cannot be satisfied by a `streaming_plan()` that samples the process's
    /// real available memory: such an implementation would return the same route
    /// both times whatever the field says. That is the whole content of the fix —
    /// before it, the route (and since #2330 Phase-2, therefore WHICH OPERATOR the
    /// quasi-Laplace criterion prices) moved with whatever else the box was doing
    /// between two probes of the same rho.
    #[test]
    fn streaming_plan_follows_the_carried_sample_not_ambient_memory() {
        let (mut term, _target, _rho) = crate::manifold::tests::small_two_atom_periodic_term();
        term.gpu_policy = gam_gpu::GpuPolicy::Off;

        term.host_available_bytes = 0;
        let starved = term.streaming_plan().expect("plan at a starved reading");
        term.host_available_bytes = 1 << 40;
        let roomy = term.streaming_plan().expect("plan at a roomy reading");

        assert!(
            starved.streaming,
            "carrying a zero host reading must route this term to streaming; got {starved:?}"
        );
        assert!(
            !roomy.streaming,
            "carrying a terabyte must route this term direct; got {roomy:?}"
        );
        assert_ne!(
            starved.in_core_budget_bytes, roomy.in_core_budget_bytes,
            "the two readings must produce different budgets, or the pair proves nothing"
        );

        // Re-planning at an unchanged reading is stable — the shape did not move,
        // so neither may the route, however busy the box became meanwhile.
        let roomy_again = term.streaming_plan().expect("re-plan at the same reading");
        assert_eq!(roomy.streaming, roomy_again.streaming);
        assert_eq!(roomy.chunk_size, roomy_again.chunk_size);
        assert_eq!(roomy.in_core_budget_bytes, roomy_again.in_core_budget_bytes);
        assert_eq!(
            roomy.process_available_bytes,
            roomy_again.process_available_bytes
        );
    }
}
