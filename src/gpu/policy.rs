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
    /// [`crate::gpu::calibration::calibrated_policy_for_device`] after the CUDA
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
    /// the apply is modelled as `≈ n·(2·d·k + d²)`. NOTE: this is a deliberate
    /// lower bound on the true `≈ n·(4·d·k + d²)` arithmetic — admitting a
    /// shape under the smaller figure can only be more conservative, never
    /// over-eager. It is keyed on the *frame depth* `d` (M) and border width
    /// `k` (p), not row count alone, so LLM shapes (few rows, wide `k`, modest
    /// `d`) register arithmetic the row-count gate misses.
    const fn reduced_schur_matvec_flops(n: usize, k: usize, d: usize) -> u128 {
        let n = n as u128;
        let k = k as u128;
        let d = d as u128;
        // 2·d·k cross-block apply (forward + transpose) + d² per-row solve.
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
        let per_apply = Self::reduced_schur_matvec_flops(n, k, d);
        let total = per_apply.saturating_mul(cg_iters as u128);
        total >= Self::MATVEC_OFFLOAD_FLOPS_MIN
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
}
