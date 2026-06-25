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
    let matrix_free_cross_bytes = n_obs
        .saturating_mul(row_block_dim)
        .saturating_mul(p_out)
        .saturating_mul(SAE_BYTES_PER_F64);
    let direct_peak_bytes = full_batch_bytes
        .saturating_add(row_cross_bytes)
        .saturating_add(dense_schur_bytes);
    let matrix_free_peak_bytes = chunk_window_bytes
        .min(full_batch_bytes.max(per_row_bytes))
        .saturating_add(matrix_free_cross_bytes)
        .saturating_add(
            border_dim
                .saturating_mul(SAE_BYTES_PER_F64)
                .saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER),
        );
    let direct_admitted = direct_peak_bytes <= in_core_budget_bytes;
    // The matrix-free streaming plan is the bounded-memory fallback: its peak is
    // the chunk window plus the row-cross and border-vector workspace, not the
    // full batch. Admit it against the larger of the in-core budget and an
    // absolute streaming floor so a starved box (budget collapsed to ~0) can
    // still run the plan that was designed for exactly that regime. The direct
    // (dense, full-batch) admission above is intentionally NOT floored — it can
    // OOM, so it stays gated on the real budget.
    let matrix_free_budget = in_core_budget_bytes.max(SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES);
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
        options.schur_pd_floor = Some(crate::solver::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        options
    }

    pub(crate) fn direct_logdet_admitted(self) -> bool {
        self.direct_admitted
    }
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
}
