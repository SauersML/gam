use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GpuMixedPrecisionPolicy {
    /// Always use fp64 factorization; no refinement attempted.
    Off,
    /// Attempt fp32 Cholesky factorization followed by up to
    /// `REFINEMENT_MAX_STEPS` fp64-residual refinement steps. Policy admits
    /// the attempt only when `p ≥ REFINEMENT_MIN_P` (so that the fp64 GEMV
    /// overhead is amortized) and the measured residual drops monotonically.
    /// Falls back to fp64 factorization automatically when the residual does
    /// not decrease (κ(A)·u ≥ 1 regime) or when the fp32 POTRF itself fails.
    Refinement,
    /// Always use fp64 factorization; equivalent to `Off` but signals that
    /// an explicit policy decision was taken.
    Never,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuDispatchPolicy {
    pub xtwx_n_min: usize,
    pub xtwx_flops_min: usize,
    pub xtwx_use_fused_below_p: usize,
    pub gemm_min_flops: usize,
    pub potrf_min_p: usize,
    pub small_dense_batched_potrf_max_p: usize,
    pub small_dense_batched_potrf_min_batch: usize,
    pub syevd_min_p: usize,
    pub sparse_min_nnz: usize,
    pub fused_kernel_min_n: usize,
    pub keep_design_resident_min_bytes: usize,
    pub prefer_gpu_factorization_min_p: usize,
    pub row_kernel_min_n: usize,
    pub mixed_precision: GpuMixedPrecisionPolicy,
}

impl Default for GpuDispatchPolicy {
    /// Conservative seed thresholds used before device calibration and when
    /// calibration cannot run on the current host.
    ///
    /// The production runtime replaces these with
    /// [`crate::calibration::calibrated_policy_for_device`] after the CUDA
    /// probe selects a concrete device. Keep these values conservative: they
    /// are the typed baseline for CPU-only builds, failed calibration, and unit
    /// tests that exercise policy predicates without initializing CUDA.
    fn default() -> Self {
        Self {
            xtwx_n_min: 50_000,
            xtwx_flops_min: 100_000_000,
            xtwx_use_fused_below_p: 256,
            gemm_min_flops: 100_000_000,
            potrf_min_p: 512,
            small_dense_batched_potrf_max_p: 32,
            small_dense_batched_potrf_min_batch: 8,
            syevd_min_p: 256,
            sparse_min_nnz: 1_000_000,
            fused_kernel_min_n: 100_000,
            keep_design_resident_min_bytes: 32 * 1024 * 1024,
            prefer_gpu_factorization_min_p: 512,
            row_kernel_min_n: 50_000,
            mixed_precision: GpuMixedPrecisionPolicy::Refinement,
        }
    }
}

impl GpuDispatchPolicy {
    /// Minimum problem dimension for the fp32+refinement path.
    ///
    /// Below this threshold the fp64 GEMV needed for the residual check costs
    /// more than the savings from fp32 factorization. The threshold is set so
    /// that a single `p × p` DGEMV (2p² flops) is at least 10× cheaper than
    /// the `p³/3` POTRF (i.e. p ≥ 64) while still leaving margin for the
    /// POTRF/POTRS launches. In practice `p ≥ 64` matches the existing
    /// `potrf_min_p = 512` floor for GPU dispatch, so the refinement path only
    /// activates when the GPU factorization path is already chosen.
    pub const REFINEMENT_MIN_P: usize = 64;

    /// Maximum number of fp32-correction steps per solve.
    ///
    /// Two steps suffice for κ(A) ≤ 10⁵ at fp32 (u ≈ 6 × 10⁻⁸): after step
    /// 1 the error is O(κ u)² ≈ 10⁻⁶, after step 2 it is O(κ u)⁴ ≈ 10⁻¹²,
    /// which is well within the fp64 unit roundoff of 10⁻¹⁶ × κ. A cap of 3
    /// is used defensively.
    pub const REFINEMENT_MAX_STEPS: usize = 3;

    /// Relative residual tolerance for declaring convergence.
    ///
    /// `‖r‖ / ‖b‖ ≤ tol` is considered a converged solve. 10⁻¹² is two
    /// orders of magnitude above the fp64 machine epsilon times a moderate
    /// condition number, leaving the policy conservative.
    pub const REFINEMENT_TOL: f64 = 1e-12;

    /// Return `true` when the policy and problem size together suggest that
    /// attempting fp32 factorization + iterative refinement will be profitable.
    ///
    /// The predicate is conservative:
    ///   * `GpuMixedPrecisionPolicy::Off` or `Never` → always `false`.
    ///   * `Refinement` with `p < REFINEMENT_MIN_P` → `false` (GEMV overhead
    ///     not amortised by fp32 POTRF savings below this threshold).
    ///   * Otherwise `true`; the caller still falls back to fp64 factorization
    ///     when the runtime fp32 POTRF fails or when the measured residual is
    ///     non-monotone.
    #[inline]
    pub const fn iterative_refinement_should_attempt(&self, p: usize) -> bool {
        match self.mixed_precision {
            GpuMixedPrecisionPolicy::Off | GpuMixedPrecisionPolicy::Never => false,
            GpuMixedPrecisionPolicy::Refinement => p >= Self::REFINEMENT_MIN_P,
        }
    }

    pub const fn dense_gemv_target_is_gpu(&self, n: usize, p: usize, resident: bool) -> bool {
        resident || n.saturating_mul(p).saturating_mul(2) >= self.gemm_min_flops
    }

    pub const fn xtwx_target_is_gpu(&self, n: usize, p: usize, materialized: bool) -> bool {
        materialized && n > 0 && p > 0 && self.xtwx_flops(n, p) >= self.dense_reduction_flops_min()
    }

    pub const fn xtwy_target_is_gpu(
        &self,
        n: usize,
        px: usize,
        q: usize,
        materialized: bool,
    ) -> bool {
        materialized
            && n > 0
            && px > 0
            && q > 0
            && self.xtwy_flops(n, px, q) >= self.dense_reduction_flops_min()
    }

    pub const fn potrf_target_is_gpu(&self, p: usize, h_resident: bool) -> bool {
        h_resident && p >= self.potrf_min_p
    }

    pub const fn dense_hessian_work_target_is_gpu(&self, n: usize, p: usize) -> bool {
        n > 0
            && p >= Self::DEVICE_LOOP_MIN_P
            && self.xtwx_flops(n, p) >= self.dense_reduction_flops_min()
    }

    const fn dense_reduction_flops_min(&self) -> u128 {
        if self.xtwx_flops_min < self.gemm_min_flops {
            self.xtwx_flops_min as u128
        } else {
            self.gemm_min_flops as u128
        }
    }

    const fn xtwx_flops(&self, n: usize, p: usize) -> u128 {
        2u128 * (n as u128) * (p as u128) * (p as u128)
    }

    const fn xtwy_flops(&self, n: usize, px: usize, q: usize) -> u128 {
        2u128 * (n as u128) * (px as u128) * (q as u128)
    }

    /// Minimum total CG-amortised matvec flops below which the host↔device
    /// transfer of the row frames + CG vectors is not repaid by the device
    /// matvec, so the reduced-Schur PCG hot loop stays on the CPU.
    ///
    /// The dense-Direct path keys on `dense_reduction_flops_min` (a single big
    /// factorization). The matrix-free SAE matvec is different: no single apply
    /// trips that floor (each is a stack of `n` tiny `d×d` solves + sparse
    /// `m·k` gather/scatter), but the *whole CG solve* runs the apply
    /// `O(cg_iters)` times over the same resident frames. The device wins when
    /// the **summed** matvec work over the solve exceeds the one-time staging
    /// cost — so the gate keys on `cg_iters · per_apply_flops`, not one apply.
    ///
    /// Set one order of magnitude below the dense floor: the matvec frames stay
    /// resident across CG iterations (uploaded once), so the per-flop transfer
    /// amortization is `1/cg_iters` of a cold dense launch, and the breakeven
    /// drops accordingly.
    pub const MATVEC_OFFLOAD_FLOPS_MIN: u128 = 10_000_000;

    /// Thin-curve (`d_atom = 1`) SAE dictionaries are the common manifold-SAE
    /// production shape: each per-row frame is a scalar, so the staged device
    /// payload is much smaller than the general `d > 1` row-frame bundle, while
    /// the work is still a large batched gather/scatter over `K` atoms and `n`
    /// rows.  Use a lower admission floor for this scalar-frame regime so a
    /// realistic token block with a moderately wide curve dictionary is not kept
    /// on the CPU solely because the conservative general-frame lower-bound
    /// undercounts the transpose cross term.
    pub const THIN_CURVE_MATVEC_OFFLOAD_FLOPS_MIN: u128 = 1_000_000;

    /// Conservative seed for the reduced-Schur PCG iteration count when the
    /// caller cannot supply a measured budget. InexactPCG on an SAE β-block of
    /// width `k` converges in `O(√κ)` iterations; this floor keeps the work
    /// estimate honest (≥ this many applies) without over-claiming a tight
    /// solve. Used only to amortise the staging cost in the work estimate.
    pub const MATVEC_OFFLOAD_MIN_CG_ITERS: usize = 8;

    /// Per-apply flop estimate for one reduced-Schur matvec `S·x` of a
    /// matrix-free SAE Kronecker system, as a pure function of the system shape.
    ///
    /// Per row block `i` the apply does: a forward cross-block GEMV
    /// `v_i = H_tβ^(i)·x` (`≈ 2·d·k` multiply-adds, with the per-row latent
    /// depth `d` as the M-frame width and `k` the border), a `d×d` triangular
    /// solve through the cached Cholesky factor (`≈ d²`), and a transpose
    /// cross-block GEMV `H_βt^(i)·w_i` (`≈ 2·d·k`). The two `2·d·k` GEMVs would
    /// sum to `4·d·k`; this estimate deliberately undercounts to a single
    /// `2·d·k` cross term as a conservative (lower-bound) admission floor, so
    /// the apply is modelled as `≈ n·(2·d·k + d²)`. This is a deliberate
    /// lower bound on the true `≈ n·(4·d·k + d²)` arithmetic — admitting a
    /// shape under the smaller figure can only be more conservative, never
    /// over-eager. It is keyed on the *frame depth* `d` (M) and border width
    /// `k` (p), not row count alone, so LLM shapes (few rows, wide `k`, modest
    /// `d`) register arithmetic the row-count gate misses.
    ///
    /// USE FOR DISPATCH GATING ONLY. This is **not** a flop count: it omits the
    /// transpose cross-block GEMV (`2·d·k`), so it is a strict lower bound on the
    /// true per-apply work `n·(4·d·k + d²)`. The gate can therefore only
    /// under-admit, never over-admit. Do not reuse it for benchmark / speedup
    /// accounting.
    const fn admission_work_lower_bound(n: usize, k: usize, d: usize) -> u128 {
        let n = n as u128;
        let k = k as u128;
        let d = d as u128;
        // 2·d·k cross-block apply (forward only) + d² per-row solve — the
        // transpose GEMV is intentionally dropped so this stays a lower bound.
        n.saturating_mul(
            2u128
                .saturating_mul(d)
                .saturating_mul(k)
                .saturating_add(d * d),
        )
    }

    /// Work-based admission for offloading the **reduced-Schur PCG matvec**
    /// (the InexactPCG hot loop for matrix-free SAE β-blocks) to the device.
    ///
    /// This is the Phase-1 (#1017) re-keying: the dense gates key on row count
    /// (`xtwx_n_min`, `row_kernel_min_n` at 50k) or a single big-factorization
    /// flop floor, neither of which the SAE LLM shape trips — `(n≈2000) ×
    /// (k≈2048) × (d≈8)` is *thousands of small dense ops*, no single op large,
    /// so the row-count gate keeps the whole fit on one CPU core. Here the gate
    /// is the **total batched work over the CG solve**:
    ///
    /// ```text
    /// estimated_device_flops = cg_iters · per_apply_flops(n, k, d)
    /// should_offload = estimated_device_flops ≥ T_breakeven
    /// ```
    ///
    /// where `T_breakeven = MATVEC_OFFLOAD_FLOPS_MIN` accounts for the
    /// host↔device staging of the row frames + CG vectors amortised over the
    /// `cg_iters` applies that reuse the resident frames (so the per-flop
    /// transfer cost is `1/cg_iters` of a cold launch, an order of magnitude
    /// below the dense-Direct floor).
    ///
    /// Pure function of the shape: no device needed to evaluate, so it is unit-
    /// testable. The caller still falls back to the bit-identical CPU matvec
    /// whenever the backend build declines, so admitting a shape never changes
    /// the numerics — only where the `Σ_i Y_iᵀ(Y_i x)` flops execute.
    ///
    /// * `n`        — number of row blocks (SAE observations / latent rows).
    /// * `k`        — border β width (the SAE decoder atom count `K`).
    /// * `d`        — per-row latent / active-frame depth (the M dimension).
    /// * `cg_iters` — expected PCG iteration budget; the per-apply work is
    ///   multiplied by this because the frames stay resident across iterations.
    ///   Pass [`Self::MATVEC_OFFLOAD_MIN_CG_ITERS`] when no measured budget is
    ///   available; a tighter (smaller) value only makes the gate stricter.
    ///
    /// ## Live arrow-Schur call site
    ///
    /// `crate::solver::arrow_schur::maybe_inject_gpu_schur_matvec` gates the
    /// InexactPCG reduced-Schur matvec injection on this predicate:
    /// `reduced_schur_matvec_should_offload(sys.rows.len(), sys.k, sys.d,
    /// options.pcg.max_iterations.min(options.trust_region.max_iterations))`,
    /// where `sys.d` is the system's max per-row latent depth and the iteration
    /// budget is the same `max_iterations` the PCG loop launches with.
    /// `try_device_arrow_direct` (the **dense** Direct point solve) correctly
    /// keeps `dense_hessian_work_target_is_gpu`: that path is a single large
    /// factorization, not the amortised matvec.
    pub const fn reduced_schur_matvec_should_offload(
        &self,
        n: usize,
        k: usize,
        d: usize,
        cg_iters: usize,
    ) -> bool {
        if n == 0 || k == 0 || d == 0 || cg_iters == 0 {
            return false;
        }
        // The border width must clear the device-loop floor: below it the per-
        // apply launch latency (one kernel sequence per matvec) dominates any
        // arithmetic regardless of how many CG iterations run.
        if k < Self::DEVICE_LOOP_MIN_P {
            return false;
        }
        let per_apply = Self::admission_work_lower_bound(n, k, d);
        let total = per_apply.saturating_mul(cg_iters as u128);
        let floor = if d == 1 {
            Self::THIN_CURVE_MATVEC_OFFLOAD_FLOPS_MIN
        } else {
            Self::MATVEC_OFFLOAD_FLOPS_MIN
        };
        total >= floor
    }
}

/// Factorization strategy for the arrow-Schur border (shared `β`) solve, chosen
/// from the *shape* of the joint system rather than a single fixed border-width
/// cut (`ArrowSolverMode::automatic`'s `DIRECT_SOLVE_MAX_K = 2000`).
///
/// The border width alone is a blunt selector: it cannot see that the data-fit
/// contribution to the `k × k` border is only rank `Σ_i d_i ≈ n·d`. For the
/// #1017 color arm (`n = 180`, per-row depth `d = 2`, border `k = 15360`) the
/// data information is rank `360` yet a dense Direct solve pays a full `k³/3 ≈
/// 1.2e12`-flop Cholesky — the measured 26-min-class fit. This maps cleanly onto
/// the two `ArrowSolverMode` variants the solver already implements.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArrowBorderStrategy {
    /// Eliminate the per-row blocks, form the dense `k × k` reduced Schur, and
    /// Cholesky-factor it (`ArrowSolverMode::Direct`). Appropriate for modest,
    /// near-square borders where the `k³/3` factorization is cheap and the
    /// data-fit rank is comparable to `k`.
    DenseDirect,
    /// Solve the reduced Schur iteratively by matrix-free PCG
    /// (`ArrowSolverMode::InexactPCG`), never materialising the `k × k` factor.
    /// Appropriate when the dense `k³` factorization dominates and/or the
    /// data-fit contribution to the border is rank-deficient (`n·d < k`).
    ReducedIterative,
}

/// Cost model + recommendation for the arrow-Schur border solve, a pure function
/// of the joint-system shape (unit-testable, no device required).
///
/// This operationalises the measured #1017 finding that the full arrow-Schur
/// Newton solve is dominated by the dense `k × k` border Cholesky (the on-device
/// dense Direct solve was measured at ~0.94× — a slowdown — because the `k³/3`
/// factorization, not the GPU-favourable batched per-row work, is the bottleneck
/// at LLM/SAE border widths). The lever the issue calls for is to *shrink or
/// factor the dense border* so the batched `n`-row work dominates; the plan
/// makes that decision inspectable and honest.
///
/// ## Flop model (deliberate, documented approximations)
///
/// * **Dense Direct** ≈ `2·n·d·k²` (assemble the reduced Schur: per row a
///   rank-`d` symmetric update `H_βt (H_tt)⁻¹ H_tβ` to the `k × k` border,
///   `≈ 2·d·k²` flops) `+ k³/3` (Cholesky of the dense `k × k` Schur).
/// * **Reduced iterative** ≈ `cg_iters · n·(4·d·k + d²)` (matrix-free PCG:
///   per matvec a forward + transpose cross-block GEMV `4·d·k` plus the per-row
///   `d × d` solve `d²`, summed over `n` row blocks, over `cg_iters` applies).
///
/// Both are dispatch-grade estimates, not exact operation counts; they omit
/// preconditioner setup and lower-order terms symmetrically, so their ratio (the
/// only thing the recommendation consumes) is meaningful while neither figure
/// should be reused for speedup accounting.
///
/// ## Status
///
/// Advisory / diagnostic. It is **not** wired into the live
/// `ArrowSolverMode::automatic` selector: replacing the fixed `DIRECT_SOLVE_MAX_K`
/// cut with this shape-driven crossover changes which production fits take the
/// Direct vs PCG path and must be validated on GPU hardware (#1017 Phase 2–4)
/// before it can change numerics. Today it is consumed by the honest
/// `examples/full_color_fit_1017.rs` measurement harness (modeled-vs-measured)
/// and by the unit tests below.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArrowBorderSolvePlan {
    /// Number of per-row blocks (SAE observations / latent rows).
    pub n: usize,
    /// Border `β` width (the SAE decoder atom count `K` × basis width).
    pub k: usize,
    /// Per-row latent / active-frame depth (the `M` dimension).
    pub d: usize,
    /// CG iteration budget assumed for the iterative estimate.
    pub cg_iters: usize,
    /// Effective rank of the data-fit contribution to the `k × k` border,
    /// bounded by `Σ_i d_i ≈ n·d` and never more than `k`.
    pub data_fit_rank: usize,
    /// True when `n·d < k`: the dense `k × k` Cholesky spends `O(k³)` factorising
    /// a border whose data information is only rank `n·d` — the pathological
    /// wide-sparse-border regime (color arm: `n·d = 360 ≪ k = 15360`).
    pub dense_border_rank_deficient: bool,
    /// `≈ 2·n·d·k² + k³/3` — reduced-Schur assembly plus dense border Cholesky.
    pub dense_direct_flops: u128,
    /// `≈ cg_iters · n·(4·d·k + d²)` — matrix-free PCG matvecs.
    pub reduced_iterative_flops: u128,
    /// The recommended strategy: `ReducedIterative` iff the dense factorization
    /// path costs strictly more arithmetic than the iterative path at
    /// `cg_iters`.
    pub recommended: ArrowBorderStrategy,
    /// Whether running the *recommended* strategy on the device is expected to
    /// pay off. For `ReducedIterative` this is `reduced_schur_matvec_should_offload`;
    /// for `DenseDirect` the device wins only when the batched per-row assembly
    /// work (`2·n·d·k²`, GPU-favourable batched GEMM/POTRF) at least matches the
    /// border Cholesky (`k³/3`) *and* clears the dense flop floor — the honest
    /// encoding of the measured 0.94× dense-Direct-on-device slowdown.
    pub device_favorable: bool,
}

impl GpuDispatchPolicy {
    /// Assembly flops for the dense reduced Schur: per row a rank-`d` update to
    /// the `k × k` border (`≈ 2·d·k²`), summed over `n` rows.
    const fn dense_schur_assembly_flops(n: usize, k: usize, d: usize) -> u128 {
        2u128
            .saturating_mul(n as u128)
            .saturating_mul(d as u128)
            .saturating_mul((k as u128).saturating_mul(k as u128))
    }

    /// Cholesky flops for the dense `k × k` reduced Schur: `≈ k³/3`.
    const fn dense_border_cholesky_flops(k: usize) -> u128 {
        let k = k as u128;
        k.saturating_mul(k).saturating_mul(k) / 3
    }

    /// Total matrix-free PCG flops: `cg_iters · n·(4·d·k + d²)`.
    const fn reduced_iterative_flops(n: usize, k: usize, d: usize, cg_iters: usize) -> u128 {
        let n = n as u128;
        let k = k as u128;
        let d = d as u128;
        let per_apply = n.saturating_mul(
            4u128
                .saturating_mul(d)
                .saturating_mul(k)
                .saturating_add(d.saturating_mul(d)),
        );
        per_apply.saturating_mul(cg_iters as u128)
    }

    /// Build the shape-driven [`ArrowBorderSolvePlan`] for a joint arrow-Schur
    /// system with `n` row blocks, border width `k`, per-row depth `d`, and an
    /// assumed CG budget `cg_iters` (pass
    /// [`Self::MATVEC_OFFLOAD_MIN_CG_ITERS`] when none is measured; a smaller
    /// value only biases the recommendation toward `DenseDirect`, never the
    /// reverse).
    ///
    /// Degenerate shapes (`n`, `k`, or `d` zero) return an all-zero plan
    /// recommending `DenseDirect` (the trivial/empty solve stays on the simple
    /// path) with `device_favorable = false`.
    pub fn arrow_border_solve_plan(
        &self,
        n: usize,
        k: usize,
        d: usize,
        cg_iters: usize,
    ) -> ArrowBorderSolvePlan {
        if n == 0 || k == 0 || d == 0 {
            return ArrowBorderSolvePlan {
                n,
                k,
                d,
                cg_iters,
                data_fit_rank: 0,
                dense_border_rank_deficient: false,
                dense_direct_flops: 0,
                reduced_iterative_flops: 0,
                recommended: ArrowBorderStrategy::DenseDirect,
                device_favorable: false,
            };
        }

        let assembly = Self::dense_schur_assembly_flops(n, k, d);
        let border_chol = Self::dense_border_cholesky_flops(k);
        let dense_direct_flops = assembly.saturating_add(border_chol);
        let iters = if cg_iters == 0 { 1 } else { cg_iters };
        let reduced_iterative_flops = Self::reduced_iterative_flops(n, k, d, iters);

        let data_fit_rank = (n.saturating_mul(d)).min(k);
        let dense_border_rank_deficient = n.saturating_mul(d) < k;

        let recommended = if dense_direct_flops > reduced_iterative_flops {
            ArrowBorderStrategy::ReducedIterative
        } else {
            ArrowBorderStrategy::DenseDirect
        };

        let device_favorable = match recommended {
            ArrowBorderStrategy::ReducedIterative => {
                self.reduced_schur_matvec_should_offload(n, k, d, iters)
            }
            ArrowBorderStrategy::DenseDirect => {
                // Dense Direct wins on device only when the batched per-row
                // assembly work dominates the (poorly GPU-scaling, and here
                // rank-deficient) border Cholesky, and the total clears the
                // dense reduction floor. This is the honest encoding of the
                // measured 0.94× on-device dense-Direct slowdown: when the k³
                // Cholesky dominates, stay on the CPU.
                assembly >= border_chol
                    && dense_direct_flops >= self.dense_reduction_flops_min()
            }
        };

        ArrowBorderSolvePlan {
            n,
            k,
            d,
            cg_iters: iters,
            data_fit_rank,
            dense_border_rank_deficient,
            dense_direct_flops,
            reduced_iterative_flops,
            recommended,
            device_favorable,
        }
    }
}

/// The aspirational single-GPU design-row throughput the #1412 decision gate is
/// supposed to establish for the LLM-shape batched-Cholesky + tile-GEMM fit
/// pipeline: 100 000 design rows processed per wall-clock second per device.
///
/// The original gate *claimed* this number without ever measuring it. The
/// honest contract is the other way around: a benchmark
/// (`examples/throughput_1412.rs`) measures the true rows/sec on a real device,
/// and [`GpuThroughputVerdict::from_measurement`] reports whether the measured
/// value meets the target — the verdict is a *function of the measurement*, not
/// a hardcoded assertion. See `tests/owed_1412.rs`.
pub const GPU_THROUGHPUT_TARGET_ROWS_PER_SEC: f64 = 100_000.0;

/// Outcome of comparing a *measured* GPU throughput against the target. The
/// only way to construct one is [`Self::from_measurement`], so a verdict can
/// never assert a target that was not actually established by a measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuThroughputVerdict {
    /// The measured design-rows-per-second on the device under test.
    pub measured_rows_per_sec: f64,
    /// The target the measurement is compared against.
    pub target_rows_per_sec: f64,
    /// `measured / target`. ≥ 1.0 means the target was established.
    pub fraction_of_target: f64,
    /// True iff `measured_rows_per_sec >= target_rows_per_sec`.
    pub meets_target: bool,
}

impl GpuThroughputVerdict {
    /// Build a verdict from a measured throughput against
    /// [`GPU_THROUGHPUT_TARGET_ROWS_PER_SEC`]. A non-finite or non-positive
    /// measurement can never meet the target (it is not a usable measurement).
    #[inline]
    pub fn from_measurement(measured_rows_per_sec: f64) -> Self {
        Self::from_measurement_against(measured_rows_per_sec, GPU_THROUGHPUT_TARGET_ROWS_PER_SEC)
    }

    /// Build a verdict against an explicit target (used by tests that probe the
    /// comparison logic without depending on the global target constant).
    #[inline]
    pub fn from_measurement_against(measured_rows_per_sec: f64, target_rows_per_sec: f64) -> Self {
        let usable = measured_rows_per_sec.is_finite() && measured_rows_per_sec > 0.0;
        let fraction_of_target = if usable && target_rows_per_sec > 0.0 {
            measured_rows_per_sec / target_rows_per_sec
        } else {
            0.0
        };
        Self {
            measured_rows_per_sec,
            target_rows_per_sec,
            fraction_of_target,
            meets_target: usable && measured_rows_per_sec >= target_rows_per_sec,
        }
    }
}

/// Why a Stage-3 encode deployment decision could not be made from a real device
/// measurement (#988, #1412). Each variant is a state in which the
/// `100_000` rows/sec/GPU target was neither established NOR refuted on a
/// device — the decision is blocked on hardware, not green-washed from a CPU
/// proxy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeDecisionBlocked {
    /// No CUDA device on this host: the exact encode could not be measured on a
    /// device at all (a CPU rate cannot substitute — that was the #1412 defect).
    NoDevice,
    /// A device is present but there is no device-resident *exact-encode* kernel,
    /// so the FULL per-row encode cannot be measured on the device. (The resident
    /// normal-equations solve in [`crate::encode_throughput`] is only ONE
    /// component of the encode, not the encode; a component measurement cannot
    /// decide the encode surrogate question — #988.)
    NoDeviceEncodeKernel,
    /// A device is present and a measurement was attempted, but the device path
    /// did not engage (false routing) — refused rather than reported as a pass.
    DeviceNotEngaged,
}

/// Tri-state Stage-3 encode deployment / amortized-surrogate decision
/// (#988, #1412).
///
/// The decision the throughput gate exists to make is empirical: does the EXACT
/// per-row encode clear the `100_000` rows/sec/GPU deployment target on a real
/// device? Only a real device measurement can answer it:
///   * [`Self::Met`] — a device measurement CLEARED the target: ship the exact
///     encode; the certified amortized surrogate is NOT needed.
///   * [`Self::Unmet`] — a device measurement MISSED the target: the certified
///     amortized surrogate becomes justified.
///   * [`Self::Undetermined`] — no device measurement is available. The decision
///     is BLOCKED on hardware; it is neither "surrogate unneeded" nor "surrogate
///     justified".
///
/// The critical anti-green-wash property (#1412): there is NO constructor that
/// takes a CPU rate. A CPU measurement, however fast, can never move the decision
/// out of [`Self::Undetermined`]. Projecting a CPU rate through an assumed
/// CPU→GPU factor to declare the target met was the exact #1412 defect and is
/// structurally impossible here — [`Self::Met`] / [`Self::Unmet`] come only from
/// [`Self::from_device_measurement`] with `engaged == true`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EncodeDeploymentDecision {
    /// A device measurement established the deployment target.
    Met {
        /// The measured device rows/sec that cleared the target.
        measured_rows_per_sec: f64,
        /// The target it was compared against.
        target_rows_per_sec: f64,
    },
    /// A device measurement fell short of the deployment target.
    Unmet {
        /// The measured device rows/sec that missed the target.
        measured_rows_per_sec: f64,
        /// The target it was compared against.
        target_rows_per_sec: f64,
    },
    /// No device measurement is available; the decision is blocked on hardware.
    Undetermined {
        /// Why no device measurement could be made.
        reason: EncodeDecisionBlocked,
    },
}

impl EncodeDeploymentDecision {
    /// The ONLY path to a `Met`/`Unmet` decision: a device measurement that
    /// actually engaged the device and produced a usable rate. `engaged == false`
    /// (false routing / CPU decline) or a non-finite / non-positive rate yields
    /// [`Self::Undetermined`] — never a fabricated pass or fail.
    #[must_use]
    pub fn from_device_measurement(engaged: bool, measured_rows_per_sec: f64) -> Self {
        Self::from_device_measurement_against(
            engaged,
            measured_rows_per_sec,
            GPU_THROUGHPUT_TARGET_ROWS_PER_SEC,
        )
    }

    /// [`Self::from_device_measurement`] against an explicit target (for tests
    /// that probe the decision logic without the global target constant).
    #[must_use]
    pub fn from_device_measurement_against(
        engaged: bool,
        measured_rows_per_sec: f64,
        target_rows_per_sec: f64,
    ) -> Self {
        let usable = measured_rows_per_sec.is_finite() && measured_rows_per_sec > 0.0;
        if !engaged || !usable {
            return Self::Undetermined {
                reason: EncodeDecisionBlocked::DeviceNotEngaged,
            };
        }
        if measured_rows_per_sec >= target_rows_per_sec {
            Self::Met {
                measured_rows_per_sec,
                target_rows_per_sec,
            }
        } else {
            Self::Unmet {
                measured_rows_per_sec,
                target_rows_per_sec,
            }
        }
    }

    /// Construct the blocked decision for a host that cannot measure the exact
    /// encode on a device. This is the honest CPU-only / no-device-kernel outcome
    /// — the deployment target is left undetermined rather than projected.
    #[must_use]
    pub fn blocked(reason: EncodeDecisionBlocked) -> Self {
        Self::Undetermined { reason }
    }

    /// True ONLY when a device measurement cleared the target: the exact encode
    /// ships and no surrogate is built. Never true from a CPU proxy.
    #[must_use]
    pub fn surrogate_unneeded(&self) -> bool {
        matches!(self, Self::Met { .. })
    }

    /// True ONLY when a device measurement missed the target: the certified
    /// amortized surrogate becomes justified. Never true without a measurement.
    #[must_use]
    pub fn surrogate_justified(&self) -> bool {
        matches!(self, Self::Unmet { .. })
    }

    /// True when no device measurement is available and the decision is blocked
    /// on hardware (neither [`Self::surrogate_unneeded`] nor
    /// [`Self::surrogate_justified`]).
    #[must_use]
    pub fn is_undetermined(&self) -> bool {
        matches!(self, Self::Undetermined { .. })
    }
}

/// Which `(response, link)` family the Stage 3.3 device-resident PIRLS loop
/// can evaluate without going through the Level-B raw-body NVRTC path.
///
/// Mirrors `PirlsRowFamily::ALL` at the policy layer so the predicate stays
/// linkable from the CPU PIRLS entry without dragging a Linux-only enum into
/// every host compilation unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PirlsLoopFamilyKind {
    BernoulliLogit,
    BernoulliProbit,
    BernoulliCLogLog,
    PoissonLog,
    GaussianIdentity,
    GammaLog,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PirlsLoopCurvatureKind {
    Fisher,
    Observed,
}

/// Inputs to [`should_run_reml_outer_on_device`]. The admission predicate
/// for routing the *outer* REML BFGS-over-ρ loop onto a fully device-resident
/// driver (rather than the host orchestrator that hops out per step).
///
/// Fields are intentionally lifted from data the CPU REML entry has on hand
/// before it touches the seed generator or the inner P-IRLS loop, so the
/// admission check is allocation-free and can short-circuit before any
/// device call.
#[derive(Clone, Copy, Debug)]
pub struct RemlOuterAdmission {
    /// Active design rows (post-transform).
    pub n: usize,
    /// Active design columns / penalised-Hessian dimension.
    pub p: usize,
    /// Number of smoothing parameters ρ the outer BFGS optimises over.
    pub num_rho: usize,
    /// Inner family / link pair the device-resident PIRLS loop can evaluate.
    /// `None` means the family does not map onto the six JIT-cached row
    /// kernels — the outer loop must stay on the host orchestrator because
    /// the inner step would already hop out anyway.
    pub family: Option<PirlsLoopFamilyKind>,
    /// Curvature surface the inner loop will use; tied to `family` via
    /// `pirls_loop_curvature_for`.
    pub curvature: PirlsLoopCurvatureKind,
    /// True when the CUDA runtime is initialised on this host.
    pub gpu_available: bool,
}

/// Inputs to [`should_use_gpu_pirls_loop`]. Each field comes from data the
/// CPU PIRLS entry has on hand before it touches the eigendecomposition
/// engine, so the admission check itself is allocation-free and can short-
/// circuit before any heavy work happens.
#[derive(Clone, Copy, Debug)]
pub struct PirlsLoopAdmission {
    /// Number of rows in the active (post-transform) design matrix.
    pub n: usize,
    /// Number of columns in the active design (i.e. `p` of `Xᵀ X`).
    pub p: usize,
    /// `Some(_)` when the inner family maps onto one of the six JIT-cached
    /// `PirlsRowFamily` variants; `None` for custom families that still
    /// require Stage 6 Level B and have not yet been admitted here.
    pub family: Option<PirlsLoopFamilyKind>,
    /// Curvature surface the inner loop will use; the GPU loop has Fisher +
    /// Observed kernels, anything else (e.g. expected-projection surrogates)
    /// is not admitted.
    pub curvature: PirlsLoopCurvatureKind,
    /// True when the CUDA runtime is initialised on this host (i.e.
    /// `GpuRuntime::global().is_some()`).
    pub gpu_available: bool,
}

impl GpuDispatchPolicy {
    /// Minimum design column count for the device-resident inner/outer loops.
    ///
    /// Below this width the per-iteration `XᵀWX + Cholesky` is dominated by
    /// launch latency and PCIe staging rather than arithmetic, so the host LM
    /// loop (which populates the full `PirlsResult` surface as a free
    /// side-effect) is strictly cheaper. Shared by both the inner PIRLS and
    /// outer REML admission predicates so they cannot drift apart.
    pub const DEVICE_LOOP_MIN_P: usize = 32;

    /// Conservative admission predicate for routing
    /// `fit_model_for_fixed_rho_with_adaptive_kkt` through the Stage 3.3
    /// device-resident PIRLS loop instead of the CPU LM loop.
    ///
    /// The threshold is the dense `XᵀWX` work estimate, not row count alone:
    /// LLM/SAE fits can have only a few thousand rows but thousands of columns,
    /// so `2*n*p^2` already dwarfs launch/staging overhead. Smaller fits stay on
    /// the CPU LM loop where the full `PirlsResult` surface (firth, EDF,
    /// per-row weights, …) is already populated as a free side-effect of the
    /// iteration.
    pub const fn should_use_gpu_pirls_loop(&self, adm: PirlsLoopAdmission) -> bool {
        if !adm.gpu_available {
            return false;
        }
        if !self.dense_hessian_work_target_is_gpu(adm.n, adm.p) {
            return false;
        }
        match adm.family {
            Some(_) => true,
            None => false,
        }
    }

    /// Admission predicate for routing the outer REML BFGS-over-ρ loop onto
    /// a device-resident driver that keeps the BFGS state (ρ, gradient,
    /// Hessian approx) on-device and only downloads the per-step scalar
    /// metrics (objective value, gradient norm, convergence flag).
    ///
    /// The dense-work threshold piggybacks on the existing inner-PIRLS admission
    /// predicate because the device-resident outer loop calls
    /// `pirls_loop_on_stream` per step and must not pay the host hop for small
    /// fits the inner loop would have rejected anyway. The
    /// `num_rho ≥ 2` floor rules out the trivial single-smoother case where
    /// host orchestration is already negligible and the device BFGS state
    /// (one length-`num_rho` gradient + a `num_rho × num_rho` Hessian
    /// approx) collapses to a couple of scalars not worth keeping on device.
    pub const fn should_run_reml_outer_on_device(&self, adm: RemlOuterAdmission) -> bool {
        if !adm.gpu_available {
            return false;
        }
        if !self.dense_hessian_work_target_is_gpu(adm.n, adm.p) {
            return false;
        }
        if adm.num_rho < 2 {
            return false;
        }
        match adm.family {
            Some(_) => true,
            None => false,
        }
    }
}

#[cfg(test)]
mod refinement_policy_tests {
    use super::*;

    #[test]
    fn refinement_policy_admits_large_p() {
        let pol = GpuDispatchPolicy::default();
        // Default policy is Refinement; large p should be admitted.
        assert!(pol.iterative_refinement_should_attempt(512));
        assert!(pol.iterative_refinement_should_attempt(GpuDispatchPolicy::REFINEMENT_MIN_P));
    }

    #[test]
    fn refinement_policy_rejects_small_p() {
        let pol = GpuDispatchPolicy::default();
        assert!(!pol.iterative_refinement_should_attempt(GpuDispatchPolicy::REFINEMENT_MIN_P - 1));
        assert!(!pol.iterative_refinement_should_attempt(0));
    }

    #[test]
    fn off_policy_never_attempts_refinement() {
        let pol = GpuDispatchPolicy {
            mixed_precision: GpuMixedPrecisionPolicy::Off,
            ..Default::default()
        };
        assert!(!pol.iterative_refinement_should_attempt(1024));
    }

    #[test]
    fn never_policy_never_attempts_refinement() {
        let pol = GpuDispatchPolicy {
            mixed_precision: GpuMixedPrecisionPolicy::Never,
            ..Default::default()
        };
        assert!(!pol.iterative_refinement_should_attempt(1024));
    }
}

#[cfg(test)]
mod reduced_schur_matvec_offload_tests {
    use super::*;

    /// The LLM/SAE shape the whole #1017 Phase-1 re-keying targets: a few
    /// thousand row blocks, a *wide* border (decoder atom count in the
    /// thousands), a modest per-row frame depth, and a realistic CG budget.
    /// The row-count gate (50k) and the dense-Direct flop floor both miss this
    /// "thousands of tiny dense ops" shape; the work-amortised matvec gate must
    /// fire on it.
    #[test]
    fn admits_llm_sae_matvec_shape() {
        let pol = GpuDispatchPolicy::default();
        // n≈2000 rows, k≈2048 atoms, M≈8 frame depth — n is far below the 50k
        // row gate, yet the summed CG matvec work is large.
        assert!(pol.reduced_schur_matvec_should_offload(
            2_000,
            2_048,
            8,
            GpuDispatchPolicy::MATVEC_OFFLOAD_MIN_CG_ITERS,
        ));
        // The same shape would be rejected by the row-count-style dense gate,
        // confirming the re-keying is what admits it.
        assert!(!pol.dense_hessian_work_target_is_gpu(2_000, 8));
    }

    /// Even with only a single conservative CG iteration the wide LLM border
    /// clears the breakeven (the per-apply work alone is `2_000·(2·8·2_048 +
    /// 8²) ≈ 6.6e7` flops > 1e7 by the conservative `n·(2·d·k + d²)` model;
    /// the true `n·(4·d·k + d²)` arithmetic is ≈1.3e8),
    /// so the gate is not relying on an inflated iteration count.
    #[test]
    fn admits_llm_shape_with_one_cg_iter() {
        let pol = GpuDispatchPolicy::default();
        assert!(pol.reduced_schur_matvec_should_offload(2_000, 2_048, 8, 1));
    }

    /// #1783: the primary manifold-SAE regime is a `d_atom = 1` curve
    /// dictionary.  Its scalar row frames have much lower staging cost than the
    /// general framed matvec, so realistic token blocks must not be stranded on
    /// the CPU merely because the conservative admission lower bound is thin in
    /// `d`.
    #[test]
    fn admits_thin_curve_atoms_at_realistic_scale() {
        let pol = GpuDispatchPolicy::default();
        assert!(pol.reduced_schur_matvec_should_offload(24_576, 64, 1, 1));
        assert!(pol.reduced_schur_matvec_should_offload(40_456, 256, 1, 1));
        assert!(!pol.reduced_schur_matvec_should_offload(300, 6, 1, 8));
    }

    /// Tiny shapes where the host↔device transfer dominates must stay on the
    /// CPU: a handful of rows, a narrow border, shallow frames. The summed
    /// matvec work is orders of magnitude below the staging breakeven.
    #[test]
    fn rejects_tiny_shape_where_transfer_dominates() {
        let pol = GpuDispatchPolicy::default();
        assert!(!pol.reduced_schur_matvec_should_offload(
            30,
            8,
            2,
            GpuDispatchPolicy::MATVEC_OFFLOAD_MIN_CG_ITERS,
        ));
        // The 300×8 shape the production seam tests use as the "stay CPU"
        // canary is rejected here too.
        assert!(!pol.reduced_schur_matvec_should_offload(300, 8, 4, 16));
    }

    /// A narrow border (k below the device-loop floor) is rejected regardless
    /// of how much row/iteration work is piled on: per-apply launch latency
    /// dominates a sub-`DEVICE_LOOP_MIN_P` border.
    #[test]
    fn rejects_narrow_border_even_with_huge_row_count() {
        let pol = GpuDispatchPolicy::default();
        let narrow = GpuDispatchPolicy::DEVICE_LOOP_MIN_P - 1;
        assert!(!pol.reduced_schur_matvec_should_offload(1_000_000, narrow, 64, 64));
    }

    /// Degenerate dimensions are never offloaded (no work, or no solve).
    #[test]
    fn rejects_degenerate_dimensions() {
        let pol = GpuDispatchPolicy::default();
        assert!(!pol.reduced_schur_matvec_should_offload(0, 2_048, 8, 8));
        assert!(!pol.reduced_schur_matvec_should_offload(2_000, 0, 8, 8));
        assert!(!pol.reduced_schur_matvec_should_offload(2_000, 2_048, 0, 8));
        assert!(!pol.reduced_schur_matvec_should_offload(2_000, 2_048, 8, 0));
    }

    /// The gate is monotone in the CG budget: once a shape is admitted at a
    /// given iteration count it stays admitted for any larger count (more
    /// applies over the same resident frames only improves amortization), and
    /// a borderline shape crosses the breakeven as iterations grow.
    #[test]
    fn monotone_in_cg_iters() {
        let pol = GpuDispatchPolicy::default();
        // A border at the floor with shallow frames and few rows: per-apply
        // work ~ n·(2·d·k + d²). Choose a shape that is below breakeven at 1
        // iter but above it once enough iterations accumulate.
        let (n, k, d) = (200usize, GpuDispatchPolicy::DEVICE_LOOP_MIN_P, 4usize);
        // per_apply ≈ 200·(2·4·32 + 16) = 200·272 = 54_400 flops.
        assert!(!pol.reduced_schur_matvec_should_offload(n, k, d, 1));
        // Once the summed work clears 1e7 the gate fires; ~184 iters here.
        assert!(pol.reduced_schur_matvec_should_offload(n, k, d, 1_000));
        // Monotonicity: admitted at 1_000 ⇒ admitted at every larger budget.
        assert!(pol.reduced_schur_matvec_should_offload(n, k, d, 5_000));
    }

    /// The admission lower bound must stay strictly below the true per-apply
    /// work `n·(4·d·k + d²)` for any non-degenerate cross-block shape (it drops
    /// the transpose GEMV). Treating the lower bound as a flop count would
    /// over-report device speedups, so this asserts the gap is real.
    #[test]
    fn admission_lower_bound_undercounts_actual_work() {
        for &(n, k, d) in &[
            (2_000usize, 2_048usize, 8usize),
            (200, GpuDispatchPolicy::DEVICE_LOOP_MIN_P, 4),
            (1, 1, 1),
        ] {
            let lower = GpuDispatchPolicy::admission_work_lower_bound(n, k, d);
            // True per-apply work models the full forward+transpose GEMV pair
            // plus the d×d solve: n·(4·d·k + d²).
            let actual = (n as u128) * (4 * (d as u128) * (k as u128) + (d as u128) * (d as u128));
            assert!(
                lower < actual,
                "admission lower bound {lower} must undercount actual work {actual} for ({n},{k},{d})"
            );
        }
    }
}

#[cfg(test)]
mod arrow_border_solve_plan_tests {
    use super::*;

    /// The #1017 color arm — few rows, shallow per-row depth, a very wide border
    /// (`k = 15360 = 3 × 5120`). The dense `k³/3` Cholesky (`≈ 1.2e12` flops)
    /// dwarfs a matrix-free PCG solve at any realistic CG budget, and the border
    /// is grossly rank-deficient (`n·d = 360 ≪ k`). The plan must recommend
    /// `ReducedIterative` and flag the rank deficiency.
    #[test]
    fn color_arm_recommends_reduced_iterative_and_flags_rank_deficiency() {
        let pol = GpuDispatchPolicy::default();
        let plan = pol.arrow_border_solve_plan(180, 15_360, 2, 30);
        assert_eq!(plan.recommended, ArrowBorderStrategy::ReducedIterative);
        assert!(plan.dense_border_rank_deficient);
        assert_eq!(plan.data_fit_rank, 360);
        // The dense path is orders of magnitude more expensive here.
        assert!(plan.dense_direct_flops > plan.reduced_iterative_flops * 100);
        // The recommended (iterative) path is device-favorable at this shape:
        // the wide border × summed CG work clears the matvec offload floor.
        assert!(plan.device_favorable);
    }

    /// A modest, near-square border where the data-fit rank is comparable to `k`
    /// and the `k³/3` Cholesky is cheap: dense Direct is the right call.
    #[test]
    fn small_square_border_recommends_dense_direct() {
        let pol = GpuDispatchPolicy::default();
        // n·d = 400 > k = 64: not rank-deficient; a 64³/3 Cholesky is trivial.
        let plan = pol.arrow_border_solve_plan(200, 64, 2, 8);
        assert_eq!(plan.recommended, ArrowBorderStrategy::DenseDirect);
        assert!(!plan.dense_border_rank_deficient);
        assert_eq!(plan.data_fit_rank, 64);
    }

    /// The rank-deficiency flag is exactly `n·d < k`, and `data_fit_rank` is
    /// clamped at `k` (the border can carry no more than `k` data directions).
    #[test]
    fn rank_flag_and_clamp_track_n_d_versus_k() {
        let pol = GpuDispatchPolicy::default();
        // n·d == k exactly: full-rank border, not deficient.
        let exact = pol.arrow_border_solve_plan(50, 100, 2, 8);
        assert!(!exact.dense_border_rank_deficient);
        assert_eq!(exact.data_fit_rank, 100);
        // n·d one below k: deficient.
        let deficient = pol.arrow_border_solve_plan(49, 100, 2, 8);
        assert!(deficient.dense_border_rank_deficient);
        assert_eq!(deficient.data_fit_rank, 98);
    }

    /// The recommendation is monotone toward `ReducedIterative` as the border
    /// widens at fixed row work: once the dense `k³` term overtakes the linear-
    /// in-`k` iterative cost, growing `k` keeps it recommending iterative.
    #[test]
    fn wider_border_only_moves_toward_iterative() {
        let pol = GpuDispatchPolicy::default();
        let narrow = pol.arrow_border_solve_plan(200, 128, 4, 16);
        let wide = pol.arrow_border_solve_plan(200, 8_192, 4, 16);
        // The wide border must recommend iterative.
        assert_eq!(wide.recommended, ArrowBorderStrategy::ReducedIterative);
        // If the narrow one already recommends iterative, the wide one still
        // does (monotone); if not, the wide one is a strict switch. Either way
        // the wide border's dense/iterative flop ratio exceeds the narrow one's.
        let narrow_ratio = narrow.dense_direct_flops as f64 / narrow.reduced_iterative_flops as f64;
        let wide_ratio = wide.dense_direct_flops as f64 / wide.reduced_iterative_flops as f64;
        assert!(wide_ratio > narrow_ratio);
    }

    /// A larger CG budget makes the iterative path more expensive, so the
    /// crossover can only move toward `DenseDirect`, never away from it. If a
    /// shape is `DenseDirect` at a small budget it stays `DenseDirect` at a
    /// larger one.
    #[test]
    fn larger_cg_budget_never_switches_away_from_dense() {
        let pol = GpuDispatchPolicy::default();
        let shape = (200usize, 96usize, 3usize);
        let small = pol.arrow_border_solve_plan(shape.0, shape.1, shape.2, 4);
        let large = pol.arrow_border_solve_plan(shape.0, shape.1, shape.2, 400);
        if small.recommended == ArrowBorderStrategy::DenseDirect {
            assert_eq!(large.recommended, ArrowBorderStrategy::DenseDirect);
        }
        assert!(large.reduced_iterative_flops >= small.reduced_iterative_flops);
    }

    /// Degenerate shapes yield an all-zero plan on the trivial `DenseDirect`
    /// path and are never device-favorable.
    #[test]
    fn degenerate_shapes_are_trivial_dense_and_not_device_favorable() {
        let pol = GpuDispatchPolicy::default();
        for shape in [(0usize, 100usize, 2usize), (100, 0, 2), (100, 100, 0)] {
            let plan = pol.arrow_border_solve_plan(shape.0, shape.1, shape.2, 8);
            assert_eq!(plan.recommended, ArrowBorderStrategy::DenseDirect);
            assert!(!plan.device_favorable);
            assert_eq!(plan.dense_direct_flops, 0);
            assert_eq!(plan.reduced_iterative_flops, 0);
        }
    }

    /// A zero CG budget is treated as one apply (a plan must still be
    /// comparable), never a divide-by-zero or an all-free iterative path.
    #[test]
    fn zero_cg_budget_is_treated_as_one_apply() {
        let pol = GpuDispatchPolicy::default();
        let plan = pol.arrow_border_solve_plan(180, 15_360, 2, 0);
        assert_eq!(plan.cg_iters, 1);
        assert!(plan.reduced_iterative_flops > 0);
    }
}

#[cfg(test)]
mod encode_deployment_decision_tests {
    use super::*;

    /// #1412 anti-green-wash core: a CPU rate can NEVER produce a `Met`/`Unmet`
    /// decision. The only Met/Unmet constructor requires `engaged == true`; a
    /// CPU-only host has no device measurement, so it can only ever be
    /// `Undetermined`, no matter how fast the CPU is.
    #[test]
    fn cpu_rate_can_never_meet_or_refute_the_target() {
        // Even a CPU rate a thousand times the target cannot certify the gate:
        // there is simply no `from_cpu_measurement` — the type has no such door.
        // The blocked constructor is the only CPU-side option.
        let cpu_only = EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::NoDevice);
        assert!(cpu_only.is_undetermined());
        assert!(!cpu_only.surrogate_unneeded());
        assert!(!cpu_only.surrogate_justified());

        // A "device" measurement that did not engage (false routing) is refused —
        // it becomes Undetermined even with a huge rate.
        let false_routed = EncodeDeploymentDecision::from_device_measurement(false, 1.0e9);
        assert!(false_routed.is_undetermined());
        assert!(!false_routed.surrogate_unneeded());
    }

    #[test]
    fn engaged_measurement_decides_by_the_number() {
        let target = GPU_THROUGHPUT_TARGET_ROWS_PER_SEC;
        // Clears the target => Met => surrogate unneeded.
        let met = EncodeDeploymentDecision::from_device_measurement(true, target * 2.0);
        assert!(matches!(met, EncodeDeploymentDecision::Met { .. }));
        assert!(met.surrogate_unneeded());
        assert!(!met.surrogate_justified());
        assert!(!met.is_undetermined());

        // Misses the target => Unmet => surrogate justified.
        let unmet = EncodeDeploymentDecision::from_device_measurement(true, target * 0.25);
        assert!(matches!(unmet, EncodeDeploymentDecision::Unmet { .. }));
        assert!(unmet.surrogate_justified());
        assert!(!unmet.surrogate_unneeded());

        // Exact boundary meets the target.
        let boundary = EncodeDeploymentDecision::from_device_measurement(true, target);
        assert!(boundary.surrogate_unneeded());
    }

    #[test]
    fn engaged_but_non_usable_rate_is_undetermined_not_a_pass() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let d = EncodeDeploymentDecision::from_device_measurement(true, bad);
            assert!(
                d.is_undetermined(),
                "an engaged-but-unusable rate {bad} must be Undetermined, not a decision"
            );
            assert!(!d.surrogate_unneeded());
            assert!(!d.surrogate_justified());
        }
    }

    #[test]
    fn blocked_reasons_are_all_undetermined() {
        for reason in [
            EncodeDecisionBlocked::NoDevice,
            EncodeDecisionBlocked::NoDeviceEncodeKernel,
            EncodeDecisionBlocked::DeviceNotEngaged,
        ] {
            let d = EncodeDeploymentDecision::blocked(reason);
            assert!(d.is_undetermined());
            assert!(!d.surrogate_unneeded());
            assert!(!d.surrogate_justified());
        }
    }
}
