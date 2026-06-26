//! Generic row-kernel operator framework.
//!
//! Every nonlinear family decomposes its coefficient-space Hessian as
//!
//! ```text
//! H_β = Σ_i  Jᵢᵀ Hᵢ Jᵢ
//! ```
//!
//! where Hᵢ is a small (K×K) row Hessian in "primary space" and Jᵢ is the
//! (sparse, linear) row Jacobian mapping coefficients → primary scalars.
//!
//! The [`RowKernel<K>`] trait captures the family-specific parts:
//!   - the analytic kernel (NLL, gradient, Hessian per row)
//!   - the Jacobian wiring (forward, adjoint, quadratic pullback, diagonal)
//!   - higher-order contracted derivatives (3rd, 4th) for REML outer
//!
//! All assembly — gradient, matvec, diagonal, dense Hessian, directional
//! derivatives — is then generic over any `RowKernel<K>`.

use crate::custom_family::{
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    JointHessianSourcePreference, MaterializationIntent, use_joint_matrix_free_path,
};
use gam_linalg::faer_ndarray::fast_ab;
use gam_linalg::matrix::DesignMatrix;
use crate::util::loop_progress::LoopProgress;
use gam_problem::{HyperOperator, ProjectedFactorCache, ProjectedFactorKey};
use ndarray::{Array1, Array2, ArrayView2, s};
use rayon::prelude::*;
use std::sync::Arc;

/// Minimum row count that justifies periodic loop-progress logging from
/// `build_row_kernel_cache`. Below this, the cache build finishes in
/// well under a second on large-scale hardware and the progress ticker
/// machinery is pure noise. Above this, a silent multi-minute build is
/// the documented failure mode this logging exists to expose.
const ROW_KERNEL_CACHE_PROGRESS_MIN_ROWS: usize = 100_000;
// `ARROW_ROW_CHUNK` / `arrow_row_chunk_count` + the `RowSet` reduction type
// moved DOWN to the `crate::outer_subsample` lower layer (#1135). Re-imported
// here so the in-file uses keep resolving; `RowSet` is `pub use`-d so the many
// in-family `crate::row_kernel::RowSet` call sites (allowed to depend
// on `families`) keep working, while `terms` and the type definition both
// reference the lower layer directly.
pub use crate::outer_subsample::RowSet;
use crate::outer_subsample::{ARROW_ROW_CHUNK, arrow_row_chunk_count};

/// Byte budget above which the full dense `J·F` projection (`n × K·rank` f64)
/// is no longer materialized-and-cached whole. Aligned with `ResourcePolicy`'s
/// default single-materialization budget (1 GiB). Below it, the cached fast
/// path builds the whole projection once and reuses it across the many trace
/// calls that share one factor within an outer iteration (the cross-call
/// amortization the cache exists for). Above it, the trace switches to the
/// block-tiled path — same BLAS-3 GEMM, but produced and consumed one row-tile
/// at a time so peak memory is one tile, not the whole `n × K·rank` (a 320K-row
/// fit at rank ~1000 would otherwise pin ~10 GiB and OOM the host).
const JF_MATERIALIZATION_BUDGET_BYTES: usize = 1024 * 1024 * 1024;

/// Working-set budget for a single `J·F` row-tile in the block-tiled trace.
/// 256 MiB matches `ResourcePolicy::analytic_operator_required`'s strict
/// single-materialization budget: large enough that each tile's GEMM amortizes
/// well and the reused factor stays hot, small enough to bound peak resident
/// memory regardless of `n`.
const JF_TILE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Whether the dense `n_rows × K·rank` f64 `J·F` projection would exceed
/// [`JF_MATERIALIZATION_BUDGET_BYTES`]. Saturating so the product can never
/// wrap on pathological dimensions (a wrap would falsely report "fits").
fn jf_projection_exceeds_budget<const K: usize>(n_rows: usize, rank: usize) -> bool {
    jf_projection_bytes::<K>(n_rows, rank) > JF_MATERIALIZATION_BUDGET_BYTES
}

/// Bytes of a dense `n_rows × K·rank` f64 projection (saturating).
#[inline]
fn jf_projection_bytes<const K: usize>(n_rows: usize, rank: usize) -> usize {
    n_rows
        .saturating_mul(K)
        .saturating_mul(rank)
        .saturating_mul(std::mem::size_of::<f64>())
}

/// Row-tile height for the block-tiled trace: the largest multiple of
/// [`ARROW_ROW_CHUNK`] whose `tile × K·rank` f64 projection fits
/// [`JF_TILE_BUDGET_BYTES`] (floored at one chunk so progress is always made).
/// A multiple of the chunk keeps the per-tile deterministic sub-chunk
/// summation associatively identical to the whole-projection path.
fn jf_tile_rows<const K: usize>(rank: usize) -> usize {
    let per_row = (K.saturating_mul(rank)).max(1) * std::mem::size_of::<f64>();
    let max_rows = (JF_TILE_BUDGET_BYTES / per_row).max(1);
    (max_rows / ARROW_ROW_CHUNK).max(1) * ARROW_ROW_CHUNK
}

/// Row-block size for the parallel per-row **cache build** (`build_row_kernel_cache`).
///
/// Unlike the trace/Gram folds, the cache build writes per-row scalars into
/// index-keyed slots and performs NO cross-row summation, so its chunk size is
/// free of the deterministic-associativity contract that pins [`ARROW_ROW_CHUNK`]
/// (= 256) for the reduction paths. At biobank scale the 256-row tiling fans the
/// build into `n / 256` tasks (≈760 for n ≈ 195k) of light per-row jet work; on a
/// wide `Par::rayon(0)` pool that many tiny tasks pays the crossbeam-epoch /
/// rayon-scheduling overhead documented as the dominant fanning cost for this
/// workload (issue #1045), not the per-row arithmetic itself. We instead size the
/// build to roughly `OVERSUBSCRIBE × workers` blocks — full pool occupancy with
/// load-balancing headroom, but two orders of magnitude fewer task entries — and
/// clamp to a multiple of [`ARROW_ROW_CHUNK`] so the published-value scatter
/// offsets stay chunk-aligned. Bit-identical output: each cache slot is written by
/// its own absolute row index regardless of how the row range is partitioned.
fn cache_build_chunk_rows(n_rows: usize) -> usize {
    const OVERSUBSCRIBE: usize = 4;
    if n_rows == 0 {
        return ARROW_ROW_CHUNK;
    }
    let workers = rayon::current_num_threads().max(1);
    let target_blocks = (workers * OVERSUBSCRIBE).max(1);
    let by_target = n_rows.div_ceil(target_blocks).max(1);
    // Round UP to a whole number of `ARROW_ROW_CHUNK` rows so each block spans an
    // integer count of arrow tiles and the scatter offset `block_idx * chunk_rows`
    // stays tile-aligned; floor at one tile so a tiny `n` still makes progress.
    by_target.div_ceil(ARROW_ROW_CHUNK).max(1) * ARROW_ROW_CHUNK
}

#[inline]
fn cache_build_block_count(n_rows: usize, chunk_rows: usize) -> usize {
    if n_rows == 0 {
        0
    } else {
        (n_rows - 1) / chunk_rows + 1
    }
}

// ── Row selector ─────────────────────────────────────────────────────
//
// `RowSet` is the contract every outer-only assembly path uses to declare
// *which* rows participate in this evaluation and *how* they should be
// weighted. Two shapes:
//
// * `All` — iterate `0..n_total`, every row contributes with weight 1.0.
//   The full-data behaviour every kernel had before subsampling existed.
// * `Subsample { rows, n_full }` — iterate the pre-built
//   `WeightedOuterRow` list, each row's `weight` is its Horvitz–Thompson
//   inverse-inclusion scale so partial sums remain unbiased estimators
//   of the full-data sum.
//
// The type lives here (alongside the `RowKernel` trait) so every
// `row_kernel_*` assembly function can pattern-match on it without
// importing the marginal-slope crate. Inner-PIRLS and final covariance
// passes always run on the full data; only outer score/gradient hot
// loops consume a non-`All` `RowSet`.
//
// Threading `RowSet` through every `row_kernel_*` function is Agent C's
// job — this module exposes only the type definition and basic
// constructors used by the κ-staging schedule in `smooth.rs`.
// `RowSet` (and its `par_reduce_fold`/`par_try_reduce_fold` reduction methods)
// moved DOWN to `crate::outer_subsample` (#1135) so `terms` and other consumers
// can name it without the `Subsample` field reaching up into `solver`. The
// family-specific `from_options` constructor below stays here because it reads
// `custom_family::BlockwiseFitOptions`.
impl RowSet {
    /// Build a `RowSet` directly from the outer-only subsample carried on
    /// `BlockwiseFitOptions`. When `outer_score_subsample` is `None` this
    /// returns `RowSet::All` with the caller-supplied `n_total`.
    ///
    /// `n_total` is the full data row count; it is recorded as `n_full`
    /// on the `Subsample` variant so downstream row-set consumers can validate
    /// Horvitz-Thompson weights against the population size.
    pub fn from_options(
        opts: &crate::custom_family::BlockwiseFitOptions,
        n_total: usize,
    ) -> Self {
        match opts.outer_score_subsample.as_ref() {
            None => Self::All,
            Some(s) => Self::Subsample {
                rows: Arc::clone(&s.rows),
                n_full: n_total,
            },
        }
    }
}

#[inline]
fn deterministic_chunked_sum<F>(n_items: usize, map_chunk: F) -> f64
where
    F: Fn(usize) -> f64 + Send + Sync,
{
    let partials: Vec<f64> = (0..arrow_row_chunk_count(n_items))
        .into_par_iter()
        .map(map_chunk)
        .collect();
    let mut total = 0.0_f64;
    for partial in partials {
        total += partial;
    }
    total
}

// ── Trait ────────────────────────────────────────────────────────────

/// A row-decomposable likelihood kernel with K primary scalars per row.
///
/// Implementors provide only:
///   1. The analytic kernel (NLL + gradient + Hessian in K-dim primary space)
///   2. The linear Jacobian wiring (coefficients ↔ primary scalars)
///   3. Assembly helpers (quadratic pullback, diagonal accumulation)
///   4. Higher-order contracted kernels for REML outer derivatives
///
/// All coefficient-space assembly (gradient, matvec, diagonal, dense Hessian,
/// directional derivatives) is derived generically.
pub trait RowKernel<const K: usize>: Send + Sync {
    /// Number of observations.
    fn n_rows(&self) -> usize;

    /// Total number of coefficients (flat β dimension).
    fn n_coefficients(&self) -> usize;

    /// Evaluate the row kernel at the current β.
    ///
    /// Returns `(nll_i, ∇_i[K], H_i[K×K])` — the negative log-likelihood,
    /// gradient, and Hessian in primary space for observation `row`.
    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; K], [[f64; K]; K]), String>;

    /// Forward Jacobian action: Jᵢ · d_beta → K-dim primary direction.
    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; K];

    /// Adjoint Jacobian action: out += Jᵢᵀ · v.
    ///
    /// Accumulates into `out` (length = n_coefficients).
    fn jacobian_transpose_action(&self, row: usize, v: &[f64; K], out: &mut [f64]);

    /// Quadratic pullback: target += Jᵢᵀ · h · Jᵢ.
    ///
    /// Accumulates the K×K matrix `h` pulled back through the row Jacobian
    /// into the dense p×p matrix `target`. Implementations should use
    /// sparse-aware primitives (syr_row_into, row_outer_into, etc.) for
    /// efficiency.
    fn add_pullback_hessian(&self, row: usize, h: &[[f64; K]; K], target: &mut Array2<f64>);

    /// Diagonal of quadratic form: diag += diag(Jᵢᵀ · h · Jᵢ).
    ///
    /// Accumulates into `diag` (length = n_coefficients). Implementations
    /// should use squared_axpy_row_into / crossdiag_axpy_row_into for
    /// efficiency with sparse designs.
    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; K]; K], diag: &mut [f64]);

    /// Third-order contracted derivative: `∂³ℓ_i / (∂p_a ∂p_b ∂[dir])`.
    ///
    /// Returns the K×K matrix of third derivatives contracted with one
    /// primary-space direction. Used for first directional derivatives of
    /// the Hessian (REML outer gradient).
    fn row_third_contracted(&self, row: usize, dir: &[f64; K]) -> Result<[[f64; K]; K], String>;

    /// Fourth-order contracted derivative: `∂⁴ℓ_i / (∂p_a ∂p_b ∂[dir_u] ∂[dir_v])`.
    ///
    /// Returns the K×K matrix of fourth derivatives contracted with two
    /// primary-space directions. Used for second directional derivatives of
    /// the Hessian (REML outer Hessian).
    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; K],
        dir_v: &[f64; K],
    ) -> Result<[[f64; K]; K], String>;

    /// Optional warm-up hook: triggers any per-row caches the kernel keeps for
    /// `row_third_contracted` / `row_fourth_contracted`. Called by
    /// [`RowKernelHessianWorkspace::new`] **before** the outer ext-coordinate
    /// `par_iter` enters, so the cache build runs at top-level rayon and
    /// fans out across all worker threads. If the build instead runs nested
    /// inside the outer `par_iter` (which holds 8 of 8 workers), the
    /// cache builder's own `par_iter` collapses to a single worker — the
    /// other seven threads are parked on the cache `OnceLock`. Default impl
    /// is a no-op for kernels with no per-row jet cache to prime.
    fn warm_up_directional_caches(&self) -> Result<(), String> {
        Ok(())
    }

    /// Optional batched all-rows `(nll, grad, hess)` fast path for the full-data
    /// (`RowSet::All`) row-kernel cache build. When a kernel can compute every
    /// row's primary-space `(v, g[K], H[K×K])` in one batched pass — e.g. an
    /// A100 NVRTC kernel that evaluates the transcendental per-row jet for all
    /// `n` rows in parallel — it returns the three parallel `n`-length vectors
    /// here; otherwise the default `None` keeps the per-row `row_kernel(row)`
    /// loop. The batched result MUST be bit-close (≤1e-9) to the per-row path
    /// (it runs the SAME unified jet on device), so the cache is identical; the
    /// fast path is a pure accelerator with a CPU fallback inside it.
    fn batched_value_grad_hess_all(
        &self,
    ) -> Option<Result<(Vec<f64>, Vec<[f64; K]>, Vec<[[f64; K]; K]>), String>> {
        None
    }

    /// Optional BLAS-3 Jacobian-action fast path: returns `Jᵢ · F` as an
    /// `(n_rows × stride)` dense matrix when the kernel can produce one
    /// (typically when the underlying design exposes a contiguous dense
    /// `Array2`). The default is the exact generic per-row implementation;
    /// implementations may override when a structured batched path is cheaper.
    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        Some(row_kernel_jacobian_action_matrix_generic(self, factor))
    }

    /// Row-range analogue of [`Self::jacobian_action_matrix`]: `Jᵢ · F` for the
    /// half-open row range `[start, end)`, returned as an `((end-start) ×
    /// stride)` dense block. Used by the block-tiled trace to bound peak memory
    /// to one tile while keeping the structured BLAS-3 path: a kernel that
    /// overrides `jacobian_action_matrix` with a GEMM should override this with
    /// the same GEMM restricted to a contiguous slice of its design rows. The
    /// default is the exact generic per-row build over the range.
    fn jacobian_action_matrix_rows(
        &self,
        factor: ArrayView2<'_, f64>,
        start: usize,
        end: usize,
    ) -> Array2<f64> {
        row_kernel_jacobian_action_matrix_generic_rows(self, factor, start, end)
    }

    /// Optional BLAS-3 fast path for the first directional derivative of the
    /// dense Hessian, `∂H/∂β[d_beta] = Σ_i w_i · Jᵢᵀ T³ᵢ[J·d_beta] Jᵢ`.
    ///
    /// The generic [`row_kernel_directional_derivative`] scatters, for every
    /// row, the `K×K` contracted third tensor through `add_pullback_hessian`
    /// — a per-row rank-`K` BLAS-1 update into the dense `p×p` accumulator.
    /// When the Jeffreys/Firth head term drives this once per Jeffreys-subspace
    /// column, that is `k·n·p²` BLAS-1 scatter (biobank rigid fit: the dominant
    /// inner joint-Newton `hessian_qp` cost). A kernel whose pullback is a pure
    /// design-row Gram (no structured cross terms) can instead accumulate the
    /// per-row contraction weights over a row chunk and close each chunk with a
    /// `Xᵀ diag(w) X` BLAS-3 product. The default returns the exact generic
    /// per-row path; overrides return `None` only for row sets they explicitly
    /// decline.
    ///
    /// `rows == RowSet::All` is the only case an override should claim; under a
    /// subsample / non-unit-weight `RowSet` the override must return `None` so
    /// the generic Horvitz-Thompson per-row path runs.
    fn directional_derivative_dense_override(
        &self,
        rows: &RowSet,
        d_beta: &[f64],
    ) -> Option<Result<Array2<f64>, String>> {
        // Default = the exact generic per-row path, which consumes `rows`/`d_beta`.
        // A kernel with a BLAS-3 fast path overrides this (see the rigid impl);
        // returning `Some` here keeps the dispatcher's fall-through reserved for
        // an override that explicitly declines (returns `None`) on a row-set it
        // cannot accelerate.
        Some(row_kernel_directional_derivative_generic(
            self, rows, d_beta,
        ))
    }

    /// Optional BLAS-3 fast path for the BATCHED all-axes FIRST directional
    /// derivative of the dense Hessian: with the direction sweeping every
    /// canonical axis `e_a`, return the `p` dense matrices `{Hdot[e_a]}_{a=0..p}`,
    ///
    /// ```text
    ///   Hdot[e_a] = Σ_i  Jᵢᵀ T³ᵢ[J·e_a] Jᵢ.
    /// ```
    ///
    /// This is the per-cycle hotspot of the inner-Newton Jeffreys/Firth term
    /// (`joint_jeffreys_term`'s `grad[k]`/`H_Φ` loop). The generic per-axis path
    /// asks for `Hdot[e_a]` `p` separate times; for a kernel the family
    /// reconstructs fresh per call (rigid Bernoulli marginal-slope) that rebuilds
    /// the `O(n)` per-row tensor cache `p` times every cycle the Jeffreys gate
    /// arms (gam#979). For a kernel whose pullback is a pure design-row Gram the
    /// per-row third tensor is INDEPENDENT of the swept axis, so it is built once
    /// and each axis closed with chunked `Xᵀ diag(w) X`-style BLAS-3 GEMMs. The
    /// default declines this batched optimization, so the dispatcher runs the
    /// exact generic per-axis path bit-for-bit. Overrides should claim only the
    /// full-data unit-weight
    /// `RowSet::All` case; under a subsample / non-unit-weight `RowSet` return
    /// `None` so the generic Horvitz-Thompson per-row path runs per axis.
    ///
    /// **Correctness contract.** Output `a` must equal, bit-for-bit, the generic
    /// per-axis `row_kernel_directional_derivative(self, rows, e_a)` reduced in
    /// deterministic in-row order (same contract as
    /// [`Self::hessian_dense_override`]).
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        // Default declines (the batched dispatcher then runs the generic
        // per-axis sweep). The dispatcher passes `p = n_coefficients()`; a
        // mismatch is a hard caller-contract violation regardless of which path
        // runs, so it is surfaced here where both `rows` and `p` are consumed —
        // keeping the default body free of unused bindings without masking a bad
        // call (same idiom as the second-directional default below).
        if p != self.n_coefficients() {
            let all = matches!(rows, RowSet::All);
            return Some(Err(format!(
                "directional_derivative_all_axes_dense_override: axis count {} \
                 disagrees with n_coefficients() {} (rows::All = {all})",
                p,
                self.n_coefficients(),
            )));
        }
        None
    }

    /// Optional BLAS-3 fast path for the dense joint Hessian assembly
    /// `H = Σ_i w_i · Jᵢᵀ Hᵢ Jᵢ` from the cached per-row `K×K` Hessians.
    ///
    /// The generic [`row_kernel_hessian_dense`] scatters every row's `K×K`
    /// block through `add_pullback_hessian` — a per-row rank-`K` BLAS-1 update
    /// into the dense `p×p` accumulator. For the rigid marginal-slope kernel
    /// that is `n·p²` scalar work that never reaches a BLAS-3 kernel; it is the
    /// base-Hessian leg of the post-gradient-reload Jeffreys/Firth residual
    /// term (`custom_family_joint_jeffreys_term` first materializes the
    /// observed joint Hessian, then its directional derivatives). A kernel whose
    /// pullback is a pure design-row Gram can instead gather the per-row
    /// contraction weights and close each row chunk with `Xᵀ diag(w) X`
    /// BLAS-3 products. The default returns the exact generic per-row path;
    /// overrides return `None` only for row sets they explicitly decline.
    /// Overrides should claim only the full-data unit-weight `RowSet::All` case;
    /// under a subsample / non-unit-weight `RowSet` return `None` so the generic
    /// HT path runs.
    fn hessian_dense_override(
        &self,
        rows: &RowSet,
        row_hessians: &[[[f64; K]; K]],
    ) -> Option<Array2<f64>> {
        // Default = the exact generic per-row pullback, which consumes
        // `rows`/`row_hessians`. A kernel with a BLAS-3 fast path overrides this;
        // returning `Some` keeps the dispatcher fall-through reserved for an
        // override that declines (`None`) on a row-set it cannot accelerate.
        Some(row_kernel_hessian_dense_generic(self, rows, row_hessians))
    }

    /// Optional BLAS-3 fast path for the BATCHED all-axes second directional
    /// derivative of the dense Hessian: with one direction `d_beta_u` held fixed
    /// and the second direction sweeping every canonical axis `e_a`, return the
    /// `p` dense matrices `{H²dot[d_beta_u, e_a]}_{a=0..p}`,
    ///
    /// ```text
    ///   H²dot[u, e_a] = Σ_i  Jᵢᵀ T⁴ᵢ[J·u, J·e_a] Jᵢ.
    /// ```
    ///
    /// This is the dominant cost of the outer-REML Jeffreys `H_Φ` drift
    /// (`coord_corrections`): the generic per-axis path
    /// ([`row_kernel_second_directional_derivative`]) runs `p` independent
    /// full-data sweeps, each scattering the `K×K` contracted fourth tensor
    /// through `add_pullback_hessian` — a per-row rank-`K` BLAS-1 update — for a
    /// total of `O(p · n · p²)` BLAS-1 scatter. For a kernel whose pullback is a
    /// pure design-row Gram, the per-row jet work (the `J·u` projection and the
    /// fourth-tensor partial contraction against `u`) is INDEPENDENT of the
    /// swept axis, so it can be hoisted out of the `p`-loop and each axis closed
    /// with chunked `Xᵀ diag(w) X`-style BLAS-3 GEMMs reading the shared cached
    /// fourth tensor. The default returns `None`, preserving the exact generic
    /// per-axis path for every other kernel bit-for-bit. Overrides should claim
    /// only the full-data unit-weight `RowSet::All` case; under a subsample /
    /// non-unit-weight `RowSet` return `None` so the generic Horvitz-Thompson
    /// per-row path runs per axis.
    ///
    /// **Correctness contract.** Output `a` must equal, bit-for-bit, the generic
    /// per-axis `row_kernel_second_directional_derivative(self, rows, d_beta_u,
    /// e_a)` reduced in deterministic in-row order (same contract as
    /// [`Self::hessian_dense_override`]).
    fn second_directional_derivative_all_axes_dense_override(
        &self,
        rows: &RowSet,
        d_beta_u: &[f64],
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        // Default declines (the batched dispatcher then runs the generic
        // per-axis sweep). A shape mismatch in the fixed direction is a hard
        // caller-contract violation regardless of which path runs, so it is
        // surfaced here where both `rows` and `d_beta_u` are consumed — keeping
        // the default body free of unused bindings without masking a bad call.
        if d_beta_u.len() != self.n_coefficients() {
            let all = matches!(rows, RowSet::All);
            return Some(Err(format!(
                "second_directional_derivative_all_axes_dense_override: fixed direction has \
                 {} entries, expected {} (rows::All = {all})",
                d_beta_u.len(),
                self.n_coefficients(),
            )));
        }
        None
    }
}

fn row_kernel_jacobian_action_matrix_generic<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    factor: ArrayView2<'_, f64>,
) -> Array2<f64> {
    assert_eq!(
        factor.nrows(),
        kern.n_coefficients(),
        "row-kernel JF factor row count must match coefficient dimension"
    );
    let n_rows = kern.n_rows();
    let rank = factor.ncols();
    let stride = K * rank;
    let mut jf = Array2::<f64>::zeros((n_rows, stride));
    if n_rows == 0 || rank == 0 {
        return jf;
    }
    let f_t: Array2<f64> = factor.t().as_standard_layout().into_owned();
    jf.as_slice_mut()
        .expect("row-major JF matrix must be contiguous")
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(row, jf_row)| {
            for k_col in 0..rank {
                let f_slice = f_t
                    .row(k_col)
                    .to_slice()
                    .expect("standard-layout row must be contiguous");
                let vec_k = kern.jacobian_action(row, f_slice);
                for k in 0..K {
                    jf_row[k * rank + k_col] = vec_k[k];
                }
            }
        });
    jf
}

/// Generic per-row `Jᵢ · F` over the half-open row range `[start, end)`.
/// Identical math to [`row_kernel_jacobian_action_matrix_generic`] restricted
/// to a row slice; `jf_row[local]` corresponds to global row `start + local`.
pub(crate) fn row_kernel_jacobian_action_matrix_generic_rows<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    factor: ArrayView2<'_, f64>,
    start: usize,
    end: usize,
) -> Array2<f64> {
    assert_eq!(
        factor.nrows(),
        kern.n_coefficients(),
        "row-kernel JF factor row count must match coefficient dimension"
    );
    let rank = factor.ncols();
    let stride = K * rank;
    let b = end.saturating_sub(start);
    let mut jf = Array2::<f64>::zeros((b, stride));
    if b == 0 || rank == 0 {
        return jf;
    }
    let f_t: Array2<f64> = factor.t().as_standard_layout().into_owned();
    jf.as_slice_mut()
        .expect("row-major JF matrix must be contiguous")
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(local, jf_row)| {
            let row = start + local;
            for k_col in 0..rank {
                let f_slice = f_t
                    .row(k_col)
                    .to_slice()
                    .expect("standard-layout row must be contiguous");
                let vec_k = kern.jacobian_action(row, f_slice);
                for k in 0..K {
                    jf_row[k * rank + k_col] = vec_k[k];
                }
            }
        });
    jf
}

/// One design's whole-row Jacobian-action block `design · factor_block`.
///
/// Dense designs use a BLAS-3 matrix multiply. Sparse/operator-backed designs
/// use one design matvec per factor column, matching the row-kernel generic
/// reference arithmetic while avoiding per-row dispatch.
pub(crate) fn row_kernel_design_jf(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    n_rows: usize,
) -> Array2<f64> {
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((n_rows, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => fast_ab(dense, &factor),
        None => row_kernel_design_jf_column_dot(design, &factor, n_rows),
    }
}

/// Row-range analogue of [`row_kernel_design_jf`]: one design's
/// `(end-start) × rank` Jacobian-action block over rows `[start, end)`.
pub(crate) fn row_kernel_design_jf_rows(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    start: usize,
    end: usize,
) -> Array2<f64> {
    let b = end.saturating_sub(start);
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((b, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => {
            let block = dense.slice(s![start..end, ..]);
            fast_ab(&block, &factor)
        }
        None => {
            let mut out = Array2::<f64>::zeros((b, rank));
            for (i, row) in (start..end).enumerate() {
                for c in 0..rank {
                    out[[i, c]] = design.dot_row_view(row, factor.column(c));
                }
            }
            out
        }
    }
}

/// Pack per-primary-axis `J_axis · F_axis` blocks into the row-kernel standard
/// row-major layout: `[axis0 rank cols | axis1 rank cols | ...]`.
pub(crate) fn row_kernel_pack_jf_axes<const K: usize>(
    n_rows: usize,
    rank: usize,
    axes: impl IntoIterator<Item = (usize, Array2<f64>)>,
) -> Array2<f64> {
    let mut jf = Array2::<f64>::zeros((n_rows, K * rank));
    if rank == 0 {
        return jf;
    }
    for (axis, block) in axes {
        assert!(
            axis < K,
            "row-kernel JF axis index {axis} out of range for K={K}"
        );
        assert_eq!(
            block.dim(),
            (n_rows, rank),
            "row-kernel JF axis {axis} block shape must be ({n_rows}, {rank})"
        );
        jf.slice_mut(s![.., axis * rank..(axis + 1) * rank])
            .assign(&block);
    }
    jf
}

/// Per-column matrix-vector dispatch for `design · factor_block` when no
/// contiguous dense backing is available.
pub(crate) fn row_kernel_design_jf_column_dot(
    design: &DesignMatrix,
    factor_block: &Array2<f64>,
    n_rows: usize,
) -> Array2<f64> {
    let rank = factor_block.ncols();
    let mut out = Array2::<f64>::zeros((n_rows, rank));
    for c in 0..rank {
        let result = design.dot(&factor_block.column(c).to_owned());
        out.column_mut(c).assign(&result);
    }
    out
}

/// Validate that shared row-kernel caches have one entry per observation.
pub(crate) fn validate_row_kernel_cache_lengths(
    context: &str,
    expected_len: usize,
    caches: &[(&str, usize)],
) -> Result<(), String> {
    let mismatches = caches
        .iter()
        .filter_map(|(name, actual)| {
            (*actual != expected_len).then_some(format!("{name}={actual}"))
        })
        .collect::<Vec<_>>();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{context} row-kernel cache length mismatch: {} expected={expected_len}",
            mismatches.join(" ")
        ))
    }
}

// ── Cache ────────────────────────────────────────────────────────────

/// Cached row-level kernel outputs (NLL + gradient + Hessian in primary space).
///
/// Built once per β update, reused for matvec / diagonal / dense assembly.
pub struct RowKernelCache<const K: usize> {
    pub n: usize,
    pub p: usize,
    pub nll: Vec<f64>,
    pub gradients: Vec<[f64; K]>,
    pub hessians: Vec<[[f64; K]; K]>,
}

/// Build the cache by evaluating all row kernels in parallel over the
/// supplied [`RowSet`].
///
/// * `RowSet::All` evaluates every row `0..kern.n_rows()`; the resulting
///   vectors satisfy `nll[i] = nll_i` for every i.
/// * `RowSet::Subsample` evaluates only the sampled indices; the
///   per-row slots not in the sample remain at their zero default. Aggregation
///   in the assembly functions iterates the same `RowSet`, so the unwritten
///   slots are never read.
///
/// Errors short-circuit via `Result` collection — the first failing row's
/// `Err` is returned and remaining work is dropped.
///
/// At large scale (n ≳ 3·10⁵) the per-row kernels for survival/GAMLSS
/// families dominate this build (multiple `exp`/`erf`/special calls per
/// row); serial evaluation was the last sequential step in the otherwise
/// fully-parallel row-kernel framework.
pub fn build_row_kernel_cache<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    rows: &RowSet,
) -> Result<RowKernelCache<K>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    let mut nll = vec![0.0_f64; n];
    let mut gradients = vec![[0.0_f64; K]; n];
    let mut hessians = vec![[[0.0_f64; K]; K]; n];
    let work_count = match rows {
        RowSet::All => n,
        RowSet::Subsample { rows: list, .. } => list.len(),
    };
    let progress_ticker =
        (work_count >= ROW_KERNEL_CACHE_PROGRESS_MIN_ROWS).then(LoopProgress::default_interval);
    match rows {
        RowSet::All => {
            // GPU fast path (#932-GPU): a kernel that can evaluate every row's
            // primary (v,g,H) in one batched device pass returns the three
            // parallel n-length vectors here. The result is the SAME unified
            // jet (≤1e-9), so the cache is bit-close; on `None` or any error
            // we fall through to the per-row CPU loop below.
            if let Some(Ok((bv, bg, bh))) = kern.batched_value_grad_hess_all() {
                if bv.len() == n && bg.len() == n && bh.len() == n {
                    return Ok(RowKernelCache {
                        n,
                        p,
                        nll: bv,
                        gradients: bg,
                        hessians: bh,
                    });
                }
            }
            // Pool-aware block size (issue #1045): a few-per-worker partition of
            // the row range instead of one task per 256-row arrow tile, so the
            // light per-row jet build does not pay `n/256` task entries of
            // crossbeam-epoch / rayon-scheduling overhead on a wide pool. Output
            // is bit-identical — every slot is written by its absolute row index.
            let block_rows = cache_build_chunk_rows(n);
            let evaluated_chunks: Vec<Vec<(f64, [f64; K], [[f64; K]; K])>> =
                (0..cache_build_block_count(n, block_rows))
                    .into_par_iter()
                    .map(|block_idx| {
                        let start = block_idx * block_rows;
                        let end = (start + block_rows).min(n);
                        let mut chunk = Vec::with_capacity(end - start);
                        for row in start..end {
                            let out = kern.row_kernel(row)?;
                            if let Some(ticker) = progress_ticker.as_ref() {
                                ticker.tick(1, |progress, elapsed| {
                                    log::info!(
                                        "[STAGE] row-kernel cache (all) progress={}/{} ({:.1}%) elapsed={:.1}s threads={}",
                                        progress.min(n),
                                        n,
                                        100.0 * progress.min(n) as f64 / n.max(1) as f64,
                                        elapsed,
                                        rayon::current_num_threads(),
                                    );
                                });
                            }
                            chunk.push(out);
                        }
                        Ok(chunk)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
            for (block_idx, chunk) in evaluated_chunks.into_iter().enumerate() {
                let start = block_idx * block_rows;
                for (local, (l, g, h)) in chunk.into_iter().enumerate() {
                    let i = start + local;
                    nll[i] = l;
                    gradients[i] = g;
                    hessians[i] = h;
                }
            }
        }
        RowSet::Subsample { rows: list, .. } => {
            // Evaluate only the sampled rows in parallel; scatter into
            // the n-sized cache slots keyed by their full-data index.
            let total = list.len();
            // Pool-aware block size over the SAMPLED rows (issue #1045): same
            // rationale as the `RowSet::All` arm — partition into a few blocks per
            // worker, not one task per 256-row tile. Output is bit-identical; each
            // slot is scattered by its full-data index `r.index`.
            let block_rows = cache_build_chunk_rows(total);
            let pair_chunks: Vec<Vec<(usize, (f64, [f64; K], [[f64; K]; K]))>> = list
                .par_chunks(block_rows)
                .map(|row_chunk| {
                    let mut chunk = Vec::with_capacity(row_chunk.len());
                    for r in row_chunk {
                        let out = kern.row_kernel(r.index).map(|out| (r.index, out))?;
                        if let Some(ticker) = progress_ticker.as_ref() {
                            ticker.tick(1, |progress, elapsed| {
                                log::info!(
                                    "[STAGE] row-kernel cache (subsample) progress={}/{} ({:.1}%) elapsed={:.1}s threads={}",
                                    progress.min(total),
                                    total,
                                    100.0 * progress.min(total) as f64 / total.max(1) as f64,
                                    elapsed,
                                    rayon::current_num_threads(),
                                );
                            });
                        }
                        chunk.push(out);
                    }
                    Ok(chunk)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for chunk in pair_chunks {
                for (idx, (l, g, h)) in chunk {
                    nll[idx] = l;
                    gradients[idx] = g;
                    hessians[idx] = h;
                }
            }
        }
    }
    Ok(RowKernelCache {
        n,
        p,
        nll,
        gradients,
        hessians,
    })
}

// ── Generic assembly functions ───────────────────────────────────────

/// Hessian–vector product: H · v = Σ_i w_i · Jᵢᵀ Hᵢ Jᵢ v over `rows`.
///
/// Uses cached row Hessians. No dense p×p matrix is formed. Each row
/// contributes its `RowSet` HT weight (`1.0` for `All`, `1/π_i` for
/// `Subsample`), so the sum is an unbiased estimator of the full-data
/// Hessian–vector product.
pub fn row_kernel_hessian_matvec<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
    rows: &RowSet,
    direction: &[f64],
) -> Array1<f64> {
    let p = cache.p;
    let out = rows.par_reduce_fold(
        cache.n,
        || vec![0.0_f64; p],
        |mut acc, row, w| {
            // Project to K-dim primary space
            let dir_k = kern.jacobian_action(row, direction);
            // Apply K×K row Hessian, scaled by HT weight
            let h = &cache.hessians[row];
            let mut action = [0.0_f64; K];
            for a in 0..K {
                let mut s = 0.0;
                for b in 0..K {
                    s += h[a][b] * dir_k[b];
                }
                action[a] = w * s;
            }
            // Pull back to coefficient space
            kern.jacobian_transpose_action(row, &action, &mut acc);
            acc
        },
        |mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        },
    );
    Array1::from_vec(out)
}

/// Diagonal of the Hessian: diag(H) = Σ_i w_i · diag(Jᵢᵀ Hᵢ Jᵢ) over `rows`.
///
/// Uses cached row Hessians and the family's sparse-aware diagonal
/// accumulation. No dense p×p matrix is formed.
pub fn row_kernel_hessian_diagonal<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
    rows: &RowSet,
) -> Array1<f64> {
    let p = cache.p;
    let out = rows.par_reduce_fold(
        cache.n,
        || vec![0.0_f64; p],
        |mut diag, row, w| {
            if w == 1.0 {
                kern.add_diagonal_quadratic(row, &cache.hessians[row], &mut diag);
            } else {
                // Multiply each row Hessian entry by HT weight before
                // contributing to the diagonal — equivalent to scaling
                // the resulting diag contributions by `w`.
                let h = &cache.hessians[row];
                let mut scaled = [[0.0_f64; K]; K];
                for a in 0..K {
                    for b in 0..K {
                        scaled[a][b] = w * h[a][b];
                    }
                }
                kern.add_diagonal_quadratic(row, &scaled, &mut diag);
            }
            diag
        },
        |mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        },
    );
    Array1::from_vec(out)
}

/// Gradient assembly: g = Σ_i w_i · Jᵢᵀ gᵢ over `rows`.
///
/// Uses cached row gradients and the family's sparse-aware adjoint.
/// The returned gradient is the negative log-likelihood gradient
/// (same sign convention as the cached `gradients`).
pub fn row_kernel_gradient<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
    rows: &RowSet,
) -> Array1<f64> {
    let p = cache.p;
    let out = rows.par_reduce_fold(
        cache.n,
        || vec![0.0_f64; p],
        |mut acc, row, w| {
            if w == 1.0 {
                kern.jacobian_transpose_action(row, &cache.gradients[row], &mut acc);
            } else {
                let g = &cache.gradients[row];
                let mut scaled = [0.0_f64; K];
                for a in 0..K {
                    scaled[a] = w * g[a];
                }
                kern.jacobian_transpose_action(row, &scaled, &mut acc);
            }
            acc
        },
        |mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        },
    );
    Array1::from_vec(out)
}

/// Log-likelihood from cached row kernels: ℓ = -Σ_i w_i · nll_i over `rows`.
pub fn row_kernel_log_likelihood<const K: usize>(cache: &RowKernelCache<K>, rows: &RowSet) -> f64 {
    let total = rows.par_reduce_fold(
        cache.n,
        || 0.0_f64,
        |acc, row, w| acc + w * cache.nll[row],
        |a, b| a + b,
    );
    -total
}

/// Dense Hessian assembly: H = Σ_i w_i · Jᵢᵀ Hᵢ Jᵢ over `rows`.
///
/// Uses cached row Hessians and the family's sparse-aware pullback.
/// Only needed for inference paths (ALO, posterior covariance) that
/// require a factored Hessian.
pub fn row_kernel_hessian_dense<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
    rows: &RowSet,
) -> Array2<f64> {
    if let Some(dense) = kern.hessian_dense_override(rows, &cache.hessians) {
        return dense;
    }
    row_kernel_hessian_dense_generic(kern, rows, &cache.hessians)
}

/// Generic per-row dense joint-Hessian pullback `H = Σ_i w_i · Jᵢᵀ Hᵢ Jᵢ`.
/// This is the default body of [`RowKernel::hessian_dense_override`] and the
/// dispatcher fall-through; a kernel with a BLAS-3 fast path overrides the hook
/// and may still call this for row-sets it does not accelerate.
pub fn row_kernel_hessian_dense_generic<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    rows: &RowSet,
    row_hessians: &[[[f64; K]; K]],
) -> Array2<f64> {
    let p = kern.n_coefficients();
    let n = row_hessians.len();
    rows.par_reduce_fold(
        n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, w| {
            if w == 1.0 {
                kern.add_pullback_hessian(row, &row_hessians[row], &mut acc);
            } else {
                let h = &row_hessians[row];
                let mut scaled = [[0.0_f64; K]; K];
                for a in 0..K {
                    for b in 0..K {
                        scaled[a][b] = w * h[a][b];
                    }
                }
                kern.add_pullback_hessian(row, &scaled, &mut acc);
            }
            acc
        },
        |a, b| a + b,
    )
}

/// First directional derivative of the Hessian: ∂H/∂β[d_beta] over `rows`.
///
/// For each row, computes the third-order contracted derivative in
/// primary space, then pulls back to coefficient space. Returns a
/// dense p×p matrix consumed by the REML outer gradient. Per-row
/// contributions are HT-weighted.
pub fn row_kernel_directional_derivative<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    rows: &RowSet,
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    if let Some(result) = kern.directional_derivative_dense_override(rows, d_beta) {
        return result;
    }
    row_kernel_directional_derivative_generic(kern, rows, d_beta)
}

/// Generic per-row first directional derivative of the Hessian ∂H/∂β[d_beta].
/// Default body of [`RowKernel::directional_derivative_dense_override`] and the
/// dispatcher fall-through; a kernel with a BLAS-3 fast path overrides the hook
/// and may still call this for row-sets it does not accelerate.
pub fn row_kernel_directional_derivative_generic<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    rows: &RowSet,
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    kern.warm_up_directional_caches()?;
    rows.par_try_reduce_fold(
        n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, w| -> Result<_, String> {
            let dir_k = kern.jacobian_action(row, d_beta);
            let third = kern.row_third_contracted(row, &dir_k)?;
            if w == 1.0 {
                kern.add_pullback_hessian(row, &third, &mut acc);
            } else {
                let mut scaled = [[0.0_f64; K]; K];
                for a in 0..K {
                    for b in 0..K {
                        scaled[a][b] = w * third[a][b];
                    }
                }
                kern.add_pullback_hessian(row, &scaled, &mut acc);
            }
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )
}

/// Batched all-axes FIRST directional derivative: with the direction sweeping
/// every canonical axis `e_a`, return the `p` dense matrices
/// `{Hdot[e_a]}_{a=0..p}`.
///
/// Dispatches to [`RowKernel::directional_derivative_all_axes_dense_override`]
/// when the kernel provides a BLAS-3 fast path on this row-set; otherwise falls
/// back to `p` independent [`row_kernel_directional_derivative`] sweeps, one per
/// unit axis `e_a` — bit-for-bit the generic per-axis path the inner-Newton
/// Jeffreys term consumed before the batched hook existed. The fall-back runs
/// the axis sweep on the Rayon pool (each axis is an independent full-data pure
/// evaluation) so it is no slower than the prior per-axis parallel loop; the
/// nested-BLAS guard pins each axis's GEMMs to `Par::Seq`.
pub fn row_kernel_directional_derivative_all_axes<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized + Sync),
    rows: &RowSet,
) -> Result<Vec<Array2<f64>>, String> {
    let p = kern.n_coefficients();
    if let Some(result) = kern.directional_derivative_all_axes_dense_override(rows, p) {
        return result;
    }
    // Generic fall-back: each axis `e_a` is one independent full-data
    // first-directional sweep. Fan the `p` axes across the pool with the
    // nested-BLAS guard so any inner GEMM stays `Par::Seq` (mirrors the prior
    // Jeffreys per-axis parallel sweep). Index-ordered collection keeps the
    // output bit-identical to a serial axis loop.
    (0..p)
        .into_par_iter()
        .map(|a| {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;
            gam_problem::with_nested_parallel(|| {
                row_kernel_directional_derivative(kern, rows, &axis)
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Second directional derivative of the Hessian: ∂²H/∂β²[d_u, d_v] over `rows`.
///
/// For each row, computes the fourth-order contracted derivative in
/// primary space, then pulls back to coefficient space. Returns a
/// dense p×p matrix consumed by the REML outer Hessian. Per-row
/// contributions are HT-weighted.
pub fn row_kernel_second_directional_derivative<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    rows: &RowSet,
    d_beta_u: &[f64],
    d_beta_v: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    kern.warm_up_directional_caches()?;
    rows.par_try_reduce_fold(
        n,
        || Array2::<f64>::zeros((p, p)),
        |mut acc, row, w| -> Result<_, String> {
            let dir_u = kern.jacobian_action(row, d_beta_u);
            let dir_v = kern.jacobian_action(row, d_beta_v);
            let fourth = kern.row_fourth_contracted(row, &dir_u, &dir_v)?;
            if w == 1.0 {
                kern.add_pullback_hessian(row, &fourth, &mut acc);
            } else {
                let mut scaled = [[0.0_f64; K]; K];
                for a in 0..K {
                    for b in 0..K {
                        scaled[a][b] = w * fourth[a][b];
                    }
                }
                kern.add_pullback_hessian(row, &scaled, &mut acc);
            }
            Ok(acc)
        },
        |a, b| Ok(a + b),
    )
}

/// Batched all-axes second directional derivative: with `d_beta_u` fixed and the
/// second direction sweeping every canonical axis `e_a`, return the `p` dense
/// matrices `{H²dot[d_beta_u, e_a]}_{a=0..p}`.
///
/// Dispatches to [`RowKernel::second_directional_derivative_all_axes_dense_override`]
/// when the kernel provides a BLAS-3 fast path on this row-set; otherwise falls
/// back to `p` independent [`row_kernel_second_directional_derivative`] sweeps,
/// one per unit axis `e_a` — bit-for-bit the generic per-axis path the Jeffreys
/// `H_Φ` drift consumed before the batched hook existed. The fall-back runs the
/// axis sweep on the Rayon pool (each axis is an independent full-data pure
/// evaluation) so it is no slower than the prior per-axis parallel loop; the
/// nested-BLAS guard pins each axis's GEMMs to `Par::Seq`.
pub fn row_kernel_second_directional_derivative_all_axes<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized + Sync),
    rows: &RowSet,
    d_beta_u: &[f64],
) -> Result<Vec<Array2<f64>>, String> {
    if let Some(result) = kern.second_directional_derivative_all_axes_dense_override(rows, d_beta_u)
    {
        return result;
    }
    let p = kern.n_coefficients();
    // Generic fall-back: each axis `e_a` is one independent full-data
    // second-directional sweep. Fan the `p` axes across the pool with the
    // nested-BLAS guard so any inner GEMM stays `Par::Seq` (mirrors the prior
    // Jeffreys per-axis parallel sweep). Index-ordered collection keeps the
    // output bit-identical to a serial axis loop.
    (0..p)
        .into_par_iter()
        .map(|a| {
            let mut axis = vec![0.0_f64; p];
            axis[a] = 1.0;
            gam_problem::with_nested_parallel(|| {
                row_kernel_second_directional_derivative(kern, rows, d_beta_u, &axis)
            })
        })
        .collect::<Result<Vec<_>, _>>()
}

struct RowKernelDirectionalDerivativeOperator<const K: usize, T: RowKernel<K>> {
    kern: Arc<T>,
    direction: Vec<f64>,
    p: usize,
    rows: RowSet,
}

impl<const K: usize, T: RowKernel<K>> HyperOperator
    for RowKernelDirectionalDerivativeOperator<K, T>
{
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let direction = v
            .as_slice()
            .expect("row-kernel directional derivative operator requires contiguous input");
        let out = self.rows.par_reduce_fold(
            self.kern.n_rows(),
            || vec![0.0_f64; self.p],
            |mut acc, row, w| {
                let dir_k = self.kern.jacobian_action(row, &self.direction);
                let vec_k = self.kern.jacobian_action(row, direction);
                let third = self
                    .kern
                    .row_third_contracted(row, &dir_k)
                    .expect("row-kernel third contraction should succeed for validated directions");
                let mut action = [0.0_f64; K];
                for a in 0..K {
                    let mut sum = 0.0;
                    for b in 0..K {
                        sum += third[a][b] * vec_k[b];
                    }
                    action[a] = w * sum;
                }
                self.kern.jacobian_transpose_action(row, &action, &mut acc);
                acc
            },
            |mut left, right| {
                for idx in 0..left.len() {
                    left[idx] += right[idx];
                }
                left
            },
        );
        Array1::from_vec(out)
    }

    /// Override: compute `tr(Fᵀ B F)` in a single row pass that amortises the
    /// (very expensive) `row_third_contracted` jet across all rank columns
    /// of `F`.
    ///
    /// **Why this matters:** the default trait route is `mul_mat` (per-column
    /// `mul_vec` over `rank` columns of `F`), and each `mul_vec` fires its own
    /// `into_par_iter` over the n rows that recomputes
    /// `T_r = row_third_contracted(row, J_r · self.direction)` per row. The
    /// `T_r` matrix only depends on `self.direction` — which is fixed for the
    /// operator — so the rank-many recomputations are pure waste. On the
    /// large-scale margslope-aniso-duchon16d shard the
    /// `BernoulliRigidRowKernel::row_third_contracted` evaluation (the closed-form
    /// IFT third-derivative tensor, which re-solves the per-row intercept and
    /// sweeps the grid moments) dominates the per-axis trace, with `rank≈p≈95`
    /// so the redundancy factor lands near the observed ~95×.
    ///
    /// Algebra: the operator action is
    /// ```text
    ///   B v = Σ_r Jᵣᵀ (Tᵣ · Jᵣ v),     Tᵣ = row_third_contracted(r, Jᵣ·direction)
    /// ```
    /// so for `F ∈ ℝ^{p × rank}`,
    /// ```text
    ///   tr(Fᵀ B F) = Σ_r Σ_k (Jᵣ F[:, k])ᵀ Tᵣ (Jᵣ F[:, k]).
    /// ```
    /// `Jᵣ · direction` and `Tᵣ` are computed once per row; the inner k-loop
    /// is `rank` cheap K×K bilinear forms (K=2 or 4 in practice — fully
    /// unrolled by the compiler).
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        if jf_projection_exceeds_budget::<K>(n_rows, rank) {
            return self.trace_projected_factor_tiled(factor);
        }
        let jf = self.compute_jf(factor);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    /// Cached variant — large-scale hot path. Within one outer iter
    /// `factor = g_factor` (or `w_factor`) is fixed and ~2000 trace calls
    /// against operators sharing the same kernel `Arc` recompute the same
    /// `n × rank` projection `J · F` redundantly. Caching keyed on
    /// `(Arc::as_ptr(kern), factor)` collapses all of those to a single
    /// row-streamed `J · F` build per outer iter; with `p_block = 24` at
    /// large-scale shape this is ~24× per trace, turning the ~30 min trace
    /// pile into ~1.5 min.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        if jf_projection_exceeds_budget::<K>(n_rows, rank) {
            return self.trace_projected_factor_tiled(factor);
        }
        let jf = self.cached_jf(factor, cache);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    /// BLAS-3 override of `F^T · B · F` for row-decomposable kernels.
    ///
    /// The default `HyperOperator::projected_matrix` routes through
    /// `mul_mat`, which does `rank` independent `mul_vec` calls each
    /// firing its own `par_reduce_fold` over n rows and recomputing
    /// `row_third_contracted(row, J_r·direction)` per row. At large-scale
    /// shape (n ≈ 1e5, rank ≈ 80) that's `n × rank` jet evaluations =
    /// ~8M per call — and `projected_matrix` is called multiple times
    /// per outer eval. Measured 3 s/call at N=100K.
    ///
    /// **Algebraic reformulation.** Using the row decomposition
    /// `B = Σ_r J_r^T T_r J_r`,
    ///
    /// ```text
    ///   (Fᵀ B F)[c, d] = Σ_r Σ_{a,b} jf[r, a, c] · T_r[a, b] · jf[r, b, d]
    ///                  = Σ_{a, b} (jf_a^T · diag(T_r[a, b]) · jf_b)[c, d]
    /// ```
    ///
    /// where `jf_a = jf[:, a·rank..(a+1)·rank]` is the (n × rank) slice
    /// for the a-th K-axis (already built by `compute_jf`'s BLAS-3 fast
    /// path) and `T_r[a, b]` is a per-row scalar. T_r is symmetric, so
    /// only `K·(K+1)/2` weighted matmuls are needed.
    ///
    /// **Cost.** Per (a, b) pair: one (n × rank) ← (n × rank) ⊙ weight
    /// scale and one (rank × rank) ← (n × rank)ᵀ · (n × rank) BLAS-3
    /// matmul. The per-row jet `row_third_contracted` is evaluated
    /// once per row (`n` total) and stored in a (`n × K × K`) tensor,
    /// not `n × rank` times like in the default mul_mat path.
    ///
    /// **Why this fires.** `compute_jf` is the same J·F build that the
    /// trace path already optimised, so caching is consistent. For K=2
    /// (bernoulli marginal-slope) this is 3 weighted matmuls + n
    /// jet calls. For K=4 (survival marginal-slope) it's 10 weighted
    /// matmuls; still dwarfed by the saved `rank × n` jet evaluations
    /// the default path would have done.
    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return Array2::<f64>::zeros((rank, rank));
        }

        // BLAS-3 threshold gate.
        //
        // The override reorganises `Fᵀ B F` into K(K+1)/2 weighted
        // matrix-matrix products + one per-row jet sweep, which is a
        // big win at large scale (n ≥ 1e4, where the default
        // mul_mat path's `rank × n` jet evaluations dominate). At
        // small n the BLAS-3 setup cost (per-row T tensor allocation,
        // axis-block copies for contiguous matmul layout, output
        // symmetrization) overshadows the rank-many extra jet calls
        // the per-column mul_mat path would do, and the override
        // regresses N=200 from 5 s → 15 s. The threshold lets the
        // override fire only where it materially wins.
        //
        // The gate is `n_rows * rank^2 > 2.5M` flop equivalence: at
        // n=300, rank=81 (n·rank²≈2M) the per-column mul_mat path is
        // ~1 ms cheaper than the BLAS-3 setup; at n=1000 the BLAS-3
        // path overtakes; at n=20K it's 4× faster. The threshold
        // matches that crossover. Below the threshold we fall through
        // to the trait default which is implemented inline here to
        // avoid a recursive call back into this method.
        const BLAS3_PROJECTED_MATRIX_FLOP_THRESHOLD: usize = 2_500_000;
        if n_rows.saturating_mul(rank).saturating_mul(rank) < BLAS3_PROJECTED_MATRIX_FLOP_THRESHOLD
        {
            let op_factor = self.mul_mat(factor);
            return factor.t().dot(&op_factor);
        }

        // Build J·F once (BLAS-3 fast path when the kernel exposes one).
        let jf = self.compute_jf(factor);
        assert_eq!(jf.dim(), (n_rows, K * rank));

        // Per-row T_r tensor: T[r, a, b] = row_third_contracted(r,
        // J_r·direction)[a][b]. Layout flat (n × K × K) row-major so
        // each weight vector `T[:, a, b]` is stride-K² contiguous in
        // memory after a transpose — but for the BLAS-3 step we only
        // need n-length slices, which we extract as owned vectors below.
        let direction = self.direction.as_slice();
        let t_flat = self.rows.par_reduce_fold(
            n_rows,
            || vec![0.0_f64; n_rows * K * K],
            |mut acc, row, w| {
                let dir_k = self.kern.jacobian_action(row, direction);
                let third = self
                    .kern
                    .row_third_contracted(row, &dir_k)
                    .expect("row-kernel third contraction should succeed for validated directions");
                let base = row * (K * K);
                for a in 0..K {
                    for b in 0..K {
                        acc[base + a * K + b] = w * third[a][b];
                    }
                }
                acc
            },
            |mut left, right| {
                // rayon's fold().reduce() partitions rows uniquely across
                // chunks, so every row's (a, b) slot is written by exactly
                // one accumulator. All other accumulators keep the
                // initial zero at that slot. Addition is therefore safe
                // (zero + value = value) and matches the merge semantic
                // used by the dense-output `mul_vec` reduce above.
                assert_eq!(left.len(), right.len());
                for (l, r) in left.iter_mut().zip(right.iter()) {
                    *l += *r;
                }
                left
            },
        );

        // 3 (K=2) or 10 (K=4) BLAS-3 weighted matmuls. Each:
        //   out += jf_a^T · diag(w_ab) · jf_b           (a == b)
        //   out += jf_a^T · diag(w_ab) · jf_b + transpose  (a < b)
        // Both use ndarray's `.dot(matrix)`; the elementwise scaling
        // happens by multiplying the (n × rank) view by a (n × 1)
        // broadcast column.
        let mut out = Array2::<f64>::zeros((rank, rank));
        // Owned (n × rank) blocks for cache-friendly BLAS-3 access.
        // The strided view jf.slice([:, a·rank..]) has row-stride K·rank
        // and would otherwise force BLAS into a slow per-row gemv.
        let mut jf_axis_blocks: Vec<Array2<f64>> = Vec::with_capacity(K);
        for a in 0..K {
            jf_axis_blocks.push(
                jf.slice(s![.., a * rank..(a + 1) * rank])
                    .as_standard_layout()
                    .into_owned(),
            );
        }
        let mut w_col = Array1::<f64>::zeros(n_rows);
        // Reusable (n × rank) working buffer for jf_a · diag(w_ab). Allocated
        // once here and overwritten via `assign` on every (a, b) iteration,
        // avoiding O(K²) Array2 allocations on the hot path.
        let mut jf_a_weighted: Array2<f64> = Array2::<f64>::zeros((n_rows, rank));
        for a in 0..K {
            for b in a..K {
                for r in 0..n_rows {
                    w_col[r] = t_flat[r * (K * K) + a * K + b];
                }
                // jf_a_weighted = jf_a · diag(w)
                jf_a_weighted.assign(&jf_axis_blocks[a]);
                for r in 0..n_rows {
                    let wr = w_col[r];
                    if wr == 0.0 {
                        for c in 0..rank {
                            jf_a_weighted[[r, c]] = 0.0;
                        }
                    } else {
                        for c in 0..rank {
                            jf_a_weighted[[r, c]] *= wr;
                        }
                    }
                }
                let contrib = jf_a_weighted.t().dot(&jf_axis_blocks[b]);
                if a == b {
                    out += &contrib;
                } else {
                    // T_r is symmetric: the (b, a) coefficient equals
                    // the (a, b) one, so the cross contribution lands
                    // once with its transpose to cover both off-diag
                    // blocks of the Σ_{a,b} T[a,b] outer-product sum.
                    out += &contrib;
                    out += &contrib.t();
                }
            }
        }
        // Force exact symmetry: `out = ½(out + outᵀ)`.
        //
        // Mathematically `out` is symmetric — every contribution is
        // either `M^T M` (a==b case) or `M + M^T` (a<b case). BLAS-3
        // GEMM rounds the (i, j) and (j, i) entries independently
        // through different summation orders, so the realised matrix
        // can carry sub-ulp asymmetry. The downstream ARC outer
        // optimizer reads this as a non-Hermitian Hessian and shrinks
        // its trust radius defensively — at the same ρ, v5 (which got
        // the symmetric `factor.T @ (B · F)` matrix back from the
        // default mul_mat path) accepted a unit-scale step where v6
        // takes a 1e-3 step. Symmetrizing here costs O(rank²) and
        // restores deterministic ARC steps.
        let out_t = out.t().to_owned();
        out += &out_t;
        out.mapv_inplace(|v| 0.5 * v);
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        row_kernel_directional_derivative(&*self.kern, &self.rows, &self.direction)
            .expect("row-kernel directional derivative dense materialization should succeed")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

impl<const K: usize, T: RowKernel<K>> RowKernelDirectionalDerivativeOperator<K, T> {
    /// Build the jacobian-projected factor `J · F`: an `(n, K * rank)`
    /// row-major matrix with `jf[r, k * rank + col] = (J_r · F[:, col])[k]`.
    ///
    /// The same per-row `jacobian_action(row, F[:, col])` calls the trace
    /// inner loop already performs — just stored in row-major (n × K·rank)
    /// layout so a single matrix covers every (row, col) pair.
    fn compute_jf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_rows = self.kern.n_rows();
        let rank = factor.ncols();
        let stride = K * rank;
        if n_rows == 0 || rank == 0 {
            return Array2::<f64>::zeros((n_rows, stride));
        }
        let jf = self
            .kern
            .jacobian_action_matrix(factor.view())
            .unwrap_or_else(|| {
                row_kernel_jacobian_action_matrix_generic(&*self.kern, factor.view())
            });
        assert_eq!(jf.dim(), (n_rows, stride));
        jf
    }

    /// Look up `J · F` from the cache (compute-on-miss). Identity is the
    /// kernel `Arc` pointer (every `directional_derivative_operator` built
    /// from the same workspace shares one `Arc<T>` and thus consults the
    /// same cache slot per `factor`) plus the factor's value fingerprint.
    fn cached_jf(&self, factor: &Array2<f64>, cache: &ProjectedFactorCache) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.kern) as *const () as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_jf(factor))
    }

    /// Evaluate `tr(F^T B F)` given a precomputed `J · F`. Identical inner
    /// math to `trace_projected_factor`, but the per-(row, col)
    /// `jacobian_action(row, F[:, col])` is replaced by a strided read out
    /// of `jf`.
    fn trace_projected_factor_with_jf(&self, factor: &Array2<f64>, jf: ArrayView2<'_, f64>) -> f64 {
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        assert_eq!(jf.dim(), (n_rows, K * rank));
        let direction = self.direction.as_slice();

        deterministic_chunked_sum(n_rows, |chunk_idx| -> f64 {
            let start = chunk_idx * ARROW_ROW_CHUNK;
            let end = (start + ARROW_ROW_CHUNK).min(n_rows);
            let mut chunk_total = 0.0_f64;
            for row in start..end {
                let dir_k = self.kern.jacobian_action(row, direction);
                let third = self
                    .kern
                    .row_third_contracted(row, &dir_k)
                    .expect("row-kernel third contraction should succeed for validated directions");
                let jf_row = jf.row(row);
                let jf_slice = jf_row
                    .to_slice()
                    .expect("J·F is built standard-layout (row-major)");
                let mut row_total = 0.0_f64;
                for k_col in 0..rank {
                    let mut vec_k = [0.0_f64; K];
                    for k in 0..K {
                        vec_k[k] = jf_slice[k * rank + k_col];
                    }
                    // (T_r vec_k)^T vec_k — K is a const-generic small int.
                    let mut quad = 0.0_f64;
                    for a in 0..K {
                        let mut t_dot = 0.0_f64;
                        for b in 0..K {
                            t_dot += third[a][b] * vec_k[b];
                        }
                        quad += vec_k[a] * t_dot;
                    }
                    row_total += quad;
                }
                chunk_total += row_total;
            }
            chunk_total
        })
    }

    /// Memory-bounded trace for large-scale shapes — the block-tiled form of
    /// [`Self::trace_projected_factor_with_jf`] computing the identical
    /// `tr(FᵀBF) = Σ_r Σ_k (Jᵣ·F[:,k])ᵀ Tᵣ (Jᵣ·F[:,k])`. Rather than build and
    /// cache the whole `n × K·rank` projection, it walks contiguous row-tiles:
    /// each tile's `J·F` slice is produced by the same structured BLAS-3 GEMM
    /// ([`RowKernel::jacobian_action_matrix_rows`]), consumed immediately by the
    /// per-row jet contraction, then dropped — so peak memory is one tile
    /// (≤ [`JF_TILE_BUDGET_BYTES`]) regardless of `n`, while the GEMM throughput
    /// and cache-blocking of the fast path are preserved. Tiles are a multiple
    /// of [`ARROW_ROW_CHUNK`] and summed in order, so the result is
    /// associatively identical to the whole-projection path.
    fn trace_projected_factor_tiled(&self, factor: &Array2<f64>) -> f64 {
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        let direction = self.direction.as_slice();
        let tile = jf_tile_rows::<K>(rank);

        let mut total = 0.0_f64;
        let mut tile_start = 0;
        while tile_start < n_rows {
            let tile_end = (tile_start + tile).min(n_rows);
            let jf = self
                .kern
                .jacobian_action_matrix_rows(factor.view(), tile_start, tile_end);
            let b = tile_end - tile_start;
            total += deterministic_chunked_sum(b, |chunk_idx| -> f64 {
                let start = chunk_idx * ARROW_ROW_CHUNK;
                let end = (start + ARROW_ROW_CHUNK).min(b);
                let mut chunk_total = 0.0_f64;
                for local in start..end {
                    let row = tile_start + local;
                    let dir_k = self.kern.jacobian_action(row, direction);
                    let third = self.kern.row_third_contracted(row, &dir_k).expect(
                        "row-kernel third contraction should succeed for validated directions",
                    );
                    let jf_slice = jf
                        .row(local)
                        .to_slice()
                        .expect("J·F tile is built standard-layout (row-major)");
                    let mut row_total = 0.0_f64;
                    for k_col in 0..rank {
                        let mut vec_k = [0.0_f64; K];
                        for k in 0..K {
                            vec_k[k] = jf_slice[k * rank + k_col];
                        }
                        let mut quad = 0.0_f64;
                        for a in 0..K {
                            let mut t_dot = 0.0_f64;
                            for b2 in 0..K {
                                t_dot += third[a][b2] * vec_k[b2];
                            }
                            quad += vec_k[a] * t_dot;
                        }
                        row_total += quad;
                    }
                    chunk_total += row_total;
                }
                chunk_total
            });
            tile_start = tile_end;
        }
        total
    }
}

struct RowKernelSecondDirectionalDerivativeOperator<const K: usize, T: RowKernel<K>> {
    kern: Arc<T>,
    direction_u: Vec<f64>,
    direction_v: Vec<f64>,
    p: usize,
    rows: RowSet,
}

impl<const K: usize, T: RowKernel<K>> HyperOperator
    for RowKernelSecondDirectionalDerivativeOperator<K, T>
{
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let direction = v
            .as_slice()
            .expect("row-kernel second directional derivative operator requires contiguous input");
        let out = self.rows.par_reduce_fold(
            self.kern.n_rows(),
            || vec![0.0_f64; self.p],
            |mut acc, row, w| {
                let dir_u = self.kern.jacobian_action(row, &self.direction_u);
                let dir_v = self.kern.jacobian_action(row, &self.direction_v);
                let vec_k = self.kern.jacobian_action(row, direction);
                let fourth = self.kern.row_fourth_contracted(row, &dir_u, &dir_v).expect(
                    "row-kernel fourth contraction should succeed for validated directions",
                );
                let mut action = [0.0_f64; K];
                for a in 0..K {
                    let mut sum = 0.0;
                    for b in 0..K {
                        sum += fourth[a][b] * vec_k[b];
                    }
                    action[a] = w * sum;
                }
                self.kern.jacobian_transpose_action(row, &action, &mut acc);
                acc
            },
            |mut left, right| {
                for idx in 0..left.len() {
                    left[idx] += right[idx];
                }
                left
            },
        );
        Array1::from_vec(out)
    }

    /// Override: same shape as the first-derivative operator's
    /// `trace_projected_factor` — amortise the `row_fourth_contracted` jet
    /// across all rank columns of `F`. See that override for the full rationale;
    /// the only change is the per-row matrix:
    /// ```text
    ///   Tᵣ = row_fourth_contracted(r, Jᵣ·direction_u, Jᵣ·direction_v)
    /// ```
    /// computed once per row instead of `rank` times.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        if jf_projection_exceeds_budget::<K>(n_rows, rank) {
            return self.trace_projected_factor_tiled(factor);
        }
        let jf = self.compute_jf(factor);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    /// Cached variant — see `RowKernelDirectionalDerivativeOperator::
    /// trace_projected_factor_cached` for the rationale. Same `J · F`
    /// projection, same per-(kernel, factor) cache key.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        if jf_projection_exceeds_budget::<K>(n_rows, rank) {
            return self.trace_projected_factor_tiled(factor);
        }
        let jf = self.cached_jf(factor, cache);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    fn to_dense(&self) -> Array2<f64> {
        row_kernel_second_directional_derivative(
            &*self.kern,
            &self.rows,
            &self.direction_u,
            &self.direction_v,
        )
        .expect("row-kernel second directional derivative dense materialization should succeed")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

impl<const K: usize, T: RowKernel<K>> RowKernelSecondDirectionalDerivativeOperator<K, T> {
    /// See `RowKernelDirectionalDerivativeOperator::compute_jf`.
    fn compute_jf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_rows = self.kern.n_rows();
        let rank = factor.ncols();
        let stride = K * rank;
        if n_rows == 0 || rank == 0 {
            return Array2::<f64>::zeros((n_rows, stride));
        }
        let jf = self
            .kern
            .jacobian_action_matrix(factor.view())
            .unwrap_or_else(|| {
                row_kernel_jacobian_action_matrix_generic(&*self.kern, factor.view())
            });
        assert_eq!(jf.dim(), (n_rows, stride));
        jf
    }

    fn cached_jf(&self, factor: &Array2<f64>, cache: &ProjectedFactorCache) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.kern) as *const () as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_jf(factor))
    }

    fn trace_projected_factor_with_jf(&self, factor: &Array2<f64>, jf: ArrayView2<'_, f64>) -> f64 {
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        assert_eq!(jf.dim(), (n_rows, K * rank));
        let direction_u = self.direction_u.as_slice();
        let direction_v = self.direction_v.as_slice();

        deterministic_chunked_sum(n_rows, |chunk_idx| -> f64 {
            let start = chunk_idx * ARROW_ROW_CHUNK;
            let end = (start + ARROW_ROW_CHUNK).min(n_rows);
            let mut chunk_total = 0.0_f64;
            for row in start..end {
                let dir_u = self.kern.jacobian_action(row, direction_u);
                let dir_v = self.kern.jacobian_action(row, direction_v);
                let fourth = self.kern.row_fourth_contracted(row, &dir_u, &dir_v).expect(
                    "row-kernel fourth contraction should succeed for validated directions",
                );
                let jf_row = jf.row(row);
                let jf_slice = jf_row
                    .to_slice()
                    .expect("J·F is built standard-layout (row-major)");
                let mut row_total = 0.0_f64;
                for k_col in 0..rank {
                    let mut vec_k = [0.0_f64; K];
                    for k in 0..K {
                        vec_k[k] = jf_slice[k * rank + k_col];
                    }
                    let mut quad = 0.0_f64;
                    for a in 0..K {
                        let mut t_dot = 0.0_f64;
                        for b in 0..K {
                            t_dot += fourth[a][b] * vec_k[b];
                        }
                        quad += vec_k[a] * t_dot;
                    }
                    row_total += quad;
                }
                chunk_total += row_total;
            }
            chunk_total
        })
    }

    /// Memory-bounded trace — second-derivative analogue of
    /// [`RowKernelDirectionalDerivativeOperator::trace_projected_factor_tiled`].
    /// Walks contiguous row-tiles, producing each tile's `J·F` slice by the
    /// structured BLAS-3 GEMM ([`RowKernel::jacobian_action_matrix_rows`]) and
    /// consuming it with the per-row `row_fourth_contracted` jet before
    /// dropping it. Peak memory one tile (≤ [`JF_TILE_BUDGET_BYTES`]); tiles are
    /// a multiple of [`ARROW_ROW_CHUNK`] and summed in order, so the result is
    /// associatively identical to the whole-projection path.
    fn trace_projected_factor_tiled(&self, factor: &Array2<f64>) -> f64 {
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        let direction_u = self.direction_u.as_slice();
        let direction_v = self.direction_v.as_slice();
        let tile = jf_tile_rows::<K>(rank);

        let mut total = 0.0_f64;
        let mut tile_start = 0;
        while tile_start < n_rows {
            let tile_end = (tile_start + tile).min(n_rows);
            let jf = self
                .kern
                .jacobian_action_matrix_rows(factor.view(), tile_start, tile_end);
            let b = tile_end - tile_start;
            total += deterministic_chunked_sum(b, |chunk_idx| -> f64 {
                let start = chunk_idx * ARROW_ROW_CHUNK;
                let end = (start + ARROW_ROW_CHUNK).min(b);
                let mut chunk_total = 0.0_f64;
                for local in start..end {
                    let row = tile_start + local;
                    let dir_u = self.kern.jacobian_action(row, direction_u);
                    let dir_v = self.kern.jacobian_action(row, direction_v);
                    let fourth = self.kern.row_fourth_contracted(row, &dir_u, &dir_v).expect(
                        "row-kernel fourth contraction should succeed for validated directions",
                    );
                    let jf_slice = jf
                        .row(local)
                        .to_slice()
                        .expect("J·F tile is built standard-layout (row-major)");
                    let mut row_total = 0.0_f64;
                    for k_col in 0..rank {
                        let mut vec_k = [0.0_f64; K];
                        for k in 0..K {
                            vec_k[k] = jf_slice[k * rank + k_col];
                        }
                        let mut quad = 0.0_f64;
                        for a in 0..K {
                            let mut t_dot = 0.0_f64;
                            for b2 in 0..K {
                                t_dot += fourth[a][b2] * vec_k[b2];
                            }
                            quad += vec_k[a] * t_dot;
                        }
                        row_total += quad;
                    }
                    chunk_total += row_total;
                }
                chunk_total
            });
            tile_start = tile_end;
        }
        total
    }
}

// ── Workspace adapter ────────────────────────────────────────────────

/// Generic adapter: any `RowKernel<K>` + its cache → `ExactNewtonJointHessianWorkspace`.
///
/// Plugs into the existing solver without any solver-side changes.
pub struct RowKernelHessianWorkspace<const K: usize, T: RowKernel<K>> {
    kern: Arc<T>,
    cache: RowKernelCache<K>,
    rows: RowSet,
}

impl<const K: usize, T: RowKernel<K>> RowKernelHessianWorkspace<K, T> {
    /// Full-data workspace: every row contributes with HT weight `1.0`.
    /// Equivalent to [`Self::with_rows`] with `RowSet::All`.
    pub fn new(kern: T) -> Result<Self, String> {
        Self::with_rows(kern, RowSet::All)
    }

    /// Build a workspace honouring the supplied [`RowSet`]. When the row
    /// set is a `Subsample`, the row-kernel cache only evaluates the
    /// sampled rows (the unsampled slots stay zero and are never read,
    /// because aggregation in the assembly functions iterates the same
    /// row set). All `joint_*_evaluation`, `hessian_*`, and
    /// `*_directional_derivative*` paths route through that row set with
    /// per-row Horvitz–Thompson weights, so the resulting trace and
    /// gradient are unbiased estimators of the full-data values.
    ///
    /// Higher-order jet caches (third/fourth contracted) are NOT primed
    /// here. PIRLS reuses this same workspace constructor for plain
    /// gradient/Hessian evaluations and never touches `row_third_contracted`,
    /// so priming at construction would burn ~3 s × n / scale on every
    /// PIRLS cycle for a cache the gradient path never reads. Outer-eval
    /// entry points instead call `warm_up_outer_caches` on the workspace
    /// trait once, at top-level rayon, before the ext-coord `par_iter`.
    pub fn with_rows(kern: T, rows: RowSet) -> Result<Self, String> {
        let kern = Arc::new(kern);
        let cache = build_row_kernel_cache(&*kern, &rows)?;
        // Higher-order jet caches (third/fourth contracted) are NOT primed
        // here. PIRLS reuses this same workspace constructor for plain
        // gradient/Hessian evaluations and never touches `row_third_contracted`,
        // so priming at construction would burn ~3 s × n / scale on every
        // PIRLS cycle for a cache the gradient path never reads. Outer-eval
        // entry points instead call `warm_up_outer_caches` on the workspace
        // trait once, at top-level rayon, before the ext-coord `par_iter`.
        Ok(Self { kern, cache, rows })
    }
}

impl<const K: usize, T: RowKernel<K> + 'static> ExactNewtonJointHessianWorkspace
    for RowKernelHessianWorkspace<K, T>
{
    fn warm_up_outer_caches(&self) -> Result<(), String> {
        // Forward to the kernel: any per-row third/fourth-contracted jet
        // cache it keeps gets primed here, at the top-level rayon site
        // where the outer-eval `compute_dh`/`compute_d2h` closures are
        // wired up. Called exactly once per outer iter, far outside the
        // ext-coord `par_iter`, so the cache build's `par_iter` enjoys
        // full 8-core parallelism instead of a single lock-holder worker.
        self.kern.warm_up_directional_caches()
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        Ok(Some(row_kernel_log_likelihood(&self.cache, &self.rows)))
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: row_kernel_log_likelihood(&self.cache, &self.rows),
            gradient: -row_kernel_gradient(&*self.kern, &self.cache, &self.rows),
        }))
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // The cached row-kernel state already encodes everything needed to
        // accumulate the dense joint Hessian in one row pass via
        // `row_kernel_hessian_dense`. Without this override the trace path
        // calls `MatrixFreeSpdOperator::materialize_dense_operator`, which
        // rebuilds the same dense matrix by applying the Hv operator to
        // every canonical basis vector: a `p * O(n*K^2)` redundant
        // re-stream of the row data. At large scale (n~320k, p~200) that
        // is hundreds of seconds of pure waste per outer-Hessian build.
        Ok(Some(row_kernel_hessian_dense(
            &*self.kern,
            &self.cache,
            &self.rows,
        )))
    }

    fn hessian_source_preference_for_intent(
        &self,
        intent: MaterializationIntent,
    ) -> JointHessianSourcePreference {
        match intent {
            // The inner Newton step only needs H·v and the diagonal
            // preconditioner. Keep large row-kernel families on the
            // matrix-free path instead of forcing the direct dense build.
            MaterializationIntent::InnerSolve
                if use_joint_matrix_free_path(self.cache.p, self.cache.n) =>
            {
                JointHessianSourcePreference::Operator
            }
            MaterializationIntent::InnerSolve => JointHessianSourcePreference::Dense,
            // Logdet and outer consumers either factorize/materialize H or
            // have row-kernel-specific projected trace paths. The one-pass
            // dense build is bounded and cheaper than reconstructing H from
            // p canonical HVPs.
            MaterializationIntent::LogdetFactorization
            | MaterializationIntent::OuterEvaluation
            | MaterializationIntent::OuterGradient => JointHessianSourcePreference::Dense,
        }
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let sl = v.as_slice().ok_or("hessian_matvec: non-contiguous input")?;
        Ok(Some(row_kernel_hessian_matvec(
            &*self.kern,
            &self.cache,
            &self.rows,
            sl,
        )))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        let result = self
            .hessian_matvec(v)?
            .ok_or_else(|| "row-kernel hessian_matvec unexpectedly unavailable".to_string())?;
        if result.len() != out.len() {
            return Err(format!(
                "row-kernel hessian_matvec_into: result length {} != out length {}",
                result.len(),
                out.len()
            ));
        }
        out.assign(&result);
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(row_kernel_hessian_diagonal(
            &*self.kern,
            &self.cache,
            &self.rows,
        )))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let sl = d_beta_flat
            .as_slice()
            .ok_or("directional_derivative: non-contiguous input")?;
        row_kernel_directional_derivative(&*self.kern, &self.rows, sl).map(Some)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let direction = d_beta_flat
            .as_slice()
            .ok_or("directional_derivative_operator: non-contiguous input")?
            .to_vec();
        Ok(Some(Arc::new(RowKernelDirectionalDerivativeOperator {
            kern: Arc::clone(&self.kern),
            direction,
            p: self.cache.p,
            rows: self.rows.clone(),
        })))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let su = d_beta_u
            .as_slice()
            .ok_or("second_directional_derivative: non-contiguous u")?;
        let sv = d_beta_v
            .as_slice()
            .ok_or("second_directional_derivative: non-contiguous v")?;
        row_kernel_second_directional_derivative(&*self.kern, &self.rows, su, sv).map(Some)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let direction_u = d_beta_u
            .as_slice()
            .ok_or("second_directional_derivative_operator: non-contiguous u")?
            .to_vec();
        let direction_v = d_beta_v
            .as_slice()
            .ok_or("second_directional_derivative_operator: non-contiguous v")?
            .to_vec();
        Ok(Some(Arc::new(
            RowKernelSecondDirectionalDerivativeOperator {
                kern: Arc::clone(&self.kern),
                direction_u,
                direction_v,
                p: self.cache.p,
                rows: self.rows.clone(),
            },
        )))
    }
}

#[cfg(test)]
mod gram_inner_contraction_tests {
    use super::*;
    use crate::custom_family::{
        JointHessianSource, exact_newton_joint_hessian_source_from_workspace,
    };
    use gam_problem::ProjectedFactorCache;
    use ndarray::Array2;

    #[test]
    fn pack_jf_axes_places_blocks_in_primary_axis_order() {
        let axis0 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let axis2 = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let packed = row_kernel_pack_jf_axes::<3>(2, 2, [(2, axis2), (0, axis0)]);

        assert_eq!(packed.dim(), (2, 6));
        assert_eq!(
            packed,
            Array2::from_shape_vec(
                (2, 6),
                vec![1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0,],
            )
            .unwrap()
        );
    }

    #[test]
    fn validate_row_kernel_cache_lengths_reports_all_mismatches() {
        validate_row_kernel_cache_lengths("ctx", 3, &[("third", 3), ("fourth", 3)])
            .expect("matching lengths pass");

        let err = validate_row_kernel_cache_lengths("ctx", 3, &[("third", 2), ("fourth", 4)])
            .expect_err("mismatches fail");

        assert_eq!(
            err,
            "ctx row-kernel cache length mismatch: third=2 fourth=4 expected=3"
        );
    }

    /// Synthetic K=4 row kernel: dense `(n × p)` design `X` per primary scalar
    /// (so each row's Jacobian is a sparse-style stack of K row vectors), with
    /// arbitrary K×K third / fourth contracted derivatives that depend on row
    /// and direction. Used only to exercise `trace_projected_factor_with_jf`
    /// against an independent reference contraction loop.
    struct SyntheticKernel {
        n: usize,
        p: usize,
        // Designs[k]: n × p, contributes the k-th primary scalar.
        designs: [Array2<f64>; 4],
    }

    impl SyntheticKernel {
        fn new(n: usize, p: usize, seed: u64) -> Self {
            let mut s = seed;
            let mut next = || -> f64 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (u32::MAX as f64)) - 0.5
            };
            let mut mk = || -> Array2<f64> { Array2::from_shape_fn((n, p), |_| next()) };
            let d0 = mk();
            let d1 = mk();
            let d2 = mk();
            let d3 = mk();
            Self {
                n,
                p,
                designs: [d0, d1, d2, d3],
            }
        }
    }

    impl RowKernel<4> for SyntheticKernel {
        fn n_rows(&self) -> usize {
            self.n
        }
        fn n_coefficients(&self) -> usize {
            self.p
        }
        fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
            if row >= self.n {
                return Err(format!("synthetic row {row} outside n={}", self.n));
            }
            let mut grad = [0.0_f64; 4];
            let mut hess = [[0.0_f64; 4]; 4];
            for k in 0..4 {
                grad[k] = self.designs[k].row(row).sum();
                hess[k][k] = 1.0 + (row as f64 + k as f64).abs() * 1.0e-6;
            }
            Ok((0.5 * grad.iter().map(|v| v * v).sum::<f64>(), grad, hess))
        }
        fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
            let mut out = [0.0_f64; 4];
            for k in 0..4 {
                let design_row = self.designs[k].row(row);
                let mut s = 0.0_f64;
                for j in 0..self.p {
                    s += design_row[j] * d_beta[j];
                }
                out[k] = s;
            }
            out
        }
        fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
            for k in 0..4 {
                let design_row = self.designs[k].row(row);
                for j in 0..self.p {
                    out[j] += design_row[j] * v[k];
                }
            }
        }
        fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
            for a in 0..4 {
                let row_a = self.designs[a].row(row);
                for b in 0..4 {
                    let scale = h[a][b];
                    if scale == 0.0 {
                        continue;
                    }
                    let row_b = self.designs[b].row(row);
                    for i in 0..self.p {
                        for j in 0..self.p {
                            target[[i, j]] += scale * row_a[i] * row_b[j];
                        }
                    }
                }
            }
        }
        fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
            for j in 0..self.p {
                let mut acc = 0.0;
                for a in 0..4 {
                    let x_a = self.designs[a][[row, j]];
                    for b in 0..4 {
                        acc += h[a][b] * x_a * self.designs[b][[row, j]];
                    }
                }
                diag[j] += acc;
            }
        }
        fn row_third_contracted(
            &self,
            row: usize,
            dir: &[f64; 4],
        ) -> Result<[[f64; 4]; 4], String> {
            // Arbitrary symmetric K×K matrix that depends on row and dir.
            let mut t = [[0.0_f64; 4]; 4];
            let row_f = (row as f64) * 0.013;
            for a in 0..4 {
                for b in a..4 {
                    let v = (row_f + a as f64 * 0.7 + b as f64 * 1.3).sin()
                        + dir[a] * 0.25
                        + dir[b] * 0.5
                        + dir[(a + b) % 4] * 0.125;
                    t[a][b] = v;
                    t[b][a] = v;
                }
            }
            Ok(t)
        }
        fn row_fourth_contracted(
            &self,
            row: usize,
            dir_u: &[f64; 4],
            dir_v: &[f64; 4],
        ) -> Result<[[f64; 4]; 4], String> {
            let mut t = [[0.0_f64; 4]; 4];
            let row_f = (row as f64) * 0.011 + 0.31;
            for a in 0..4 {
                for b in a..4 {
                    let v = (row_f + a as f64 * 0.9 + b as f64 * 1.7).cos()
                        + dir_u[a] * 0.13
                        + dir_v[b] * 0.27
                        + dir_u[(a + b) % 4] * dir_v[(a + 1) % 4] * 0.05;
                    t[a][b] = v;
                    t[b][a] = v;
                }
            }
            Ok(t)
        }
    }

    #[test]
    fn row_kernel_workspace_routes_inner_solve_to_operator() {
        let p = crate::custom_family::JOINT_MATRIX_FREE_MIN_DIM;
        let kernel = SyntheticKernel::new(8, p, 0x979);
        let workspace: Arc<dyn ExactNewtonJointHessianWorkspace> =
            Arc::new(RowKernelHessianWorkspace::new(kernel).expect("workspace"));

        let source = exact_newton_joint_hessian_source_from_workspace(
            &workspace,
            p,
            MaterializationIntent::InnerSolve,
            "row-kernel inner source",
        )
        .expect("source construction succeeds")
        .expect("source is present");

        let JointHessianSource::Operator {
            apply,
            apply_into,
            diagonal,
            ..
        } = source
        else {
            panic!("row-kernel inner solve must use operator source");
        };
        assert_eq!(diagonal.len(), p);

        let v = Array1::from_shape_fn(p, |i| (i as f64 % 7.0 - 3.0) * 0.125);
        let hv = apply(&v).expect("operator apply succeeds");
        let mut hv_into = Array1::<f64>::zeros(p);
        apply_into(&v, &mut hv_into).expect("operator apply_into succeeds");
        assert_eq!(hv, hv_into);
    }

    /// Independent reference: per-row, per-column bilinear-of-K-vector form,
    /// matching the pre-Gram code path. Locks the optimised
    /// `trace_projected_factor_with_jf` against the original contraction.
    fn reference_trace_first<const K: usize>(
        kern: &impl RowKernel<K>,
        direction: &[f64],
        factor: &Array2<f64>,
    ) -> f64 {
        let n_rows = kern.n_rows();
        let rank = factor.ncols();
        let p = factor.nrows();
        let mut total = 0.0_f64;
        for row in 0..n_rows {
            let dir_k_arr = kern.jacobian_action(row, direction);
            let third = kern.row_third_contracted(row, &dir_k_arr).expect("third");
            for k_col in 0..rank {
                // vec_k[k] = (J_r · F[:, k_col])[k]
                let mut col = vec![0.0_f64; p];
                for j in 0..p {
                    col[j] = factor[[j, k_col]];
                }
                let vec_k = kern.jacobian_action(row, &col);
                let mut quad = 0.0_f64;
                for a in 0..K {
                    let mut t_dot = 0.0_f64;
                    for b in 0..K {
                        t_dot += third[a][b] * vec_k[b];
                    }
                    quad += vec_k[a] * t_dot;
                }
                total += quad;
            }
        }
        total
    }

    fn reference_trace_second<const K: usize>(
        kern: &impl RowKernel<K>,
        direction_u: &[f64],
        direction_v: &[f64],
        factor: &Array2<f64>,
    ) -> f64 {
        let n_rows = kern.n_rows();
        let rank = factor.ncols();
        let p = factor.nrows();
        let mut total = 0.0_f64;
        for row in 0..n_rows {
            let dir_u = kern.jacobian_action(row, direction_u);
            let dir_v = kern.jacobian_action(row, direction_v);
            let fourth = kern
                .row_fourth_contracted(row, &dir_u, &dir_v)
                .expect("fourth");
            for k_col in 0..rank {
                let mut col = vec![0.0_f64; p];
                for j in 0..p {
                    col[j] = factor[[j, k_col]];
                }
                let vec_k = kern.jacobian_action(row, &col);
                let mut quad = 0.0_f64;
                for a in 0..K {
                    let mut t_dot = 0.0_f64;
                    for b in 0..K {
                        t_dot += fourth[a][b] * vec_k[b];
                    }
                    quad += vec_k[a] * t_dot;
                }
                total += quad;
            }
        }
        total
    }

    #[test]
    fn gram_inner_contraction_matches_reference() {
        let n = 32;
        let p = 11;
        let rank = 7;
        let kern = Arc::new(SyntheticKernel::new(n, p, 0xC0FFEE));

        // Build a random direction and factor.
        let mut s = 0xDEADBEEF_u64;
        let mut next = || -> f64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (u32::MAX as f64)) - 0.5
        };
        let direction: Vec<f64> = (0..p).map(|_| next()).collect();
        let direction_u: Vec<f64> = (0..p).map(|_| next()).collect();
        let direction_v: Vec<f64> = (0..p).map(|_| next()).collect();
        let factor = Array2::from_shape_fn((p, rank), |_| next());

        // First-derivative operator.
        let op1 = RowKernelDirectionalDerivativeOperator {
            kern: Arc::clone(&kern),
            direction: direction.clone(),
            p,
            rows: RowSet::All,
        };
        let cache = ProjectedFactorCache::default();
        let got1_uncached = HyperOperator::trace_projected_factor(&op1, &factor);
        let got1_cached = op1.trace_projected_factor_cached(&factor, &cache);
        let ref1 = reference_trace_first::<4>(&*kern, &direction, &factor);
        let rel1_uncached = (got1_uncached - ref1).abs() / ref1.abs().max(1e-12);
        let rel1_cached = (got1_cached - ref1).abs() / ref1.abs().max(1e-12);
        assert!(
            rel1_uncached < 1e-10,
            "first-derivative Gram path drifted: rel={rel1_uncached:.3e} got={got1_uncached} ref={ref1}",
        );
        assert!(
            rel1_cached < 1e-10,
            "first-derivative cached Gram path drifted: rel={rel1_cached:.3e} got={got1_cached} ref={ref1}",
        );

        // Second-derivative operator.
        let op2 = RowKernelSecondDirectionalDerivativeOperator {
            kern: Arc::clone(&kern),
            direction_u: direction_u.clone(),
            direction_v: direction_v.clone(),
            p,
            rows: RowSet::All,
        };
        let cache2 = ProjectedFactorCache::default();
        let got2_uncached = HyperOperator::trace_projected_factor(&op2, &factor);
        let got2_cached = op2.trace_projected_factor_cached(&factor, &cache2);
        let ref2 = reference_trace_second::<4>(&*kern, &direction_u, &direction_v, &factor);
        let rel2_uncached = (got2_uncached - ref2).abs() / ref2.abs().max(1e-12);
        let rel2_cached = (got2_cached - ref2).abs() / ref2.abs().max(1e-12);
        assert!(
            rel2_uncached < 1e-10,
            "second-derivative Gram path drifted: rel={rel2_uncached:.3e} got={got2_uncached} ref={ref2}",
        );
        assert!(
            rel2_cached < 1e-10,
            "second-derivative cached Gram path drifted: rel={rel2_cached:.3e} got={got2_cached} ref={ref2}",
        );
    }

    // ── #979: all-axes Jeffreys directional derivative is BUILD-ONCE + exact ──
    //
    // The rigid survival-marginal-slope Jeffreys hot path used to call
    // `exact_newton_joint_hessian_directional_derivative` once per coefficient
    // axis, and that rigid branch rebuilt a whole `SurvivalMarginalSlopeRowKernel`
    // (cloning the family + designs and its per-row cache) on EVERY axis — a
    // `p`-fold kernel rebuild paid every cycle the Jeffreys gate armed (#979).
    // The fix routes the all-axes sweep through
    // `row_kernel_directional_derivative_all_axes`, which takes the kernel by
    // reference (built ONCE) and either dispatches to the kernel's BLAS-3
    // override or runs the generic per-axis sweep over that SAME single kernel —
    // never rebuilding it.

    /// Wraps a `SyntheticKernel` and counts how many times the EXPENSIVE per-row
    /// cache is (re)built. A real kernel pays this build inside its constructor /
    /// first cache touch; the pre-#979 per-axis path rebuilt the kernel `p`
    /// times, so the build counter would read `p`. The all-axes dispatcher takes
    /// the kernel by reference, so a build that happens once at construction is
    /// observed exactly once regardless of `p`.
    struct BuildCountingKernel {
        inner: SyntheticKernel,
        builds: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl BuildCountingKernel {
        fn new(n: usize, p: usize, seed: u64) -> Self {
            let builds = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            // Constructing the kernel IS the expensive build the #979 hot path
            // paid per axis; count it here.
            builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Self {
                inner: SyntheticKernel::new(n, p, seed),
                builds,
            }
        }
    }

    // Forward every `RowKernel` method to the inner synthetic kernel. No method
    // touches the build counter — only construction does — so the counter
    // measures kernel REBUILDS, exactly the quantity #979 collapsed from `p` to
    // 1.
    impl RowKernel<4> for BuildCountingKernel {
        fn n_rows(&self) -> usize {
            self.inner.n_rows()
        }
        fn n_coefficients(&self) -> usize {
            self.inner.n_coefficients()
        }
        fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
            self.inner.row_kernel(row)
        }
        fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
            self.inner.jacobian_action(row, d_beta)
        }
        fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
            self.inner.jacobian_transpose_action(row, v, out)
        }
        fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
            self.inner.add_pullback_hessian(row, h, target)
        }
        fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
            self.inner.add_diagonal_quadratic(row, h, diag)
        }
        fn row_third_contracted(
            &self,
            row: usize,
            dir: &[f64; 4],
        ) -> Result<[[f64; 4]; 4], String> {
            self.inner.row_third_contracted(row, dir)
        }
        fn row_fourth_contracted(
            &self,
            row: usize,
            dir_u: &[f64; 4],
            dir_v: &[f64; 4],
        ) -> Result<[[f64; 4]; 4], String> {
            self.inner.row_fourth_contracted(row, dir_u, dir_v)
        }
    }

    #[test]
    fn all_axes_directional_derivative_is_build_once_and_matches_per_axis_loop_979() {
        let n = 24usize;
        let p = 7usize;
        let kern = BuildCountingKernel::new(n, p, 0x979);
        let builds = std::sync::Arc::clone(&kern.builds);

        // Drive the BATCHED all-axes dispatcher on the single kernel reference.
        let batched =
            row_kernel_directional_derivative_all_axes(&kern, &RowSet::All).expect("all-axes ok");

        // Build-once contract (#979): computing all `p` axes touched exactly ONE
        // kernel build. The pre-#979 per-axis path rebuilt the kernel per axis,
        // which on this fixture would read `p` = 7.
        let n_builds = builds.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            n_builds, 1,
            "all-axes Jeffreys sweep rebuilt the row kernel {n_builds} times for p={p}; \
             the #979 fix must build it ONCE and sweep every axis off that single kernel \
             (a revert to the per-axis `SurvivalMarginalSlopeRowKernel::new` loop reads p)",
        );

        // Correctness contract: each batched axis equals the generic per-axis
        // `row_kernel_directional_derivative(self, RowSet::All, e_a)` to ~1e-9.
        // (A BLAS-3 override, when present, must reproduce the per-axis sweep
        // bit-for-bit; the generic fallback IS that sweep.) This makes the
        // build-once route a true no-op on the math, not just faster.
        assert_eq!(batched.len(), p, "one Hdot matrix per coefficient axis");
        for (a, hdot_a) in batched.iter().enumerate() {
            let mut e_a = vec![0.0_f64; p];
            e_a[a] = 1.0;
            let per_axis = row_kernel_directional_derivative(&kern, &RowSet::All, &e_a)
                .expect("per-axis directional derivative ok");
            assert_eq!(hdot_a.dim(), per_axis.dim(), "axis {a} shape mismatch");
            let mut max_abs = 0.0_f64;
            for (g, r) in hdot_a.iter().zip(per_axis.iter()) {
                max_abs = max_abs.max((g - r).abs());
            }
            assert!(
                max_abs <= 1e-9,
                "batched all-axes Hdot[e_{a}] diverged from the per-axis sweep by \
                 {max_abs:.3e} (> 1e-9); the #979 build-once route must be numerically \
                 identical to the per-axis loop it replaced",
            );
        }
    }
}
