use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MixedPrecisionPolicy {
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
    pub mixed_precision: MixedPrecisionPolicy,
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
            mixed_precision: MixedPrecisionPolicy::Refinement,
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
    ///   * `MixedPrecisionPolicy::Off` or `Never` → always `false`.
    ///   * `Refinement` with `p < REFINEMENT_MIN_P` → `false` (GEMV overhead
    ///     not amortised by fp32 POTRF savings below this threshold).
    ///   * Otherwise `true`; the caller still falls back to fp64 factorization
    ///     when the runtime fp32 POTRF fails or when the measured residual is
    ///     non-monotone.
    #[inline]
    pub const fn iterative_refinement_should_attempt(&self, p: usize) -> bool {
        match self.mixed_precision {
            MixedPrecisionPolicy::Off | MixedPrecisionPolicy::Never => false,
            MixedPrecisionPolicy::Refinement => p >= Self::REFINEMENT_MIN_P,
        }
    }

    pub const fn dense_gemv_target_is_gpu(&self, n: usize, p: usize, resident: bool) -> bool {
        resident || n.saturating_mul(p).saturating_mul(2) >= self.gemm_min_flops
    }

    pub const fn xtwx_target_is_gpu(&self, n: usize, p: usize, materialized: bool) -> bool {
        materialized
            && n >= self.xtwx_n_min
            && n.saturating_mul(p).saturating_mul(p).saturating_mul(2) >= self.xtwx_flops_min
    }

    pub const fn potrf_target_is_gpu(&self, p: usize, h_resident: bool) -> bool {
        h_resident && p >= self.potrf_min_p
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
    /// The thresholds (`n ≥ row_kernel_min_n`, `p ≥ DEVICE_LOOP_MIN_P`) are
    /// deliberately well above the matrix-size where a single PIRLS iter's
    /// `XᵀWX + Cholesky` would be PCIe-bandwidth-bound. Smaller fits stay on
    /// the CPU LM loop where the full `PirlsResult` surface (firth, EDF,
    /// per-row weights, …) is already populated as a free side-effect of the
    /// iteration.
    pub const fn should_use_gpu_pirls_loop(&self, adm: PirlsLoopAdmission) -> bool {
        if !adm.gpu_available {
            return false;
        }
        if adm.n < self.row_kernel_min_n {
            return false;
        }
        if adm.p < Self::DEVICE_LOOP_MIN_P {
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
    /// The thresholds piggyback on the existing inner-PIRLS admission floor
    /// (`n ≥ row_kernel_min_n`, `p ≥ DEVICE_LOOP_MIN_P`) because the device-resident outer
    /// loop calls `pirls_loop_on_stream` per step and must not pay the host
    /// hop for small fits the inner loop would have rejected anyway. The
    /// `num_rho ≥ 2` floor rules out the trivial single-smoother case where
    /// host orchestration is already negligible and the device BFGS state
    /// (one length-`num_rho` gradient + a `num_rho × num_rho` Hessian
    /// approx) collapses to a couple of scalars not worth keeping on device.
    pub const fn should_run_reml_outer_on_device(&self, adm: RemlOuterAdmission) -> bool {
        if !adm.gpu_available {
            return false;
        }
        if adm.n < self.row_kernel_min_n {
            return false;
        }
        if adm.p < Self::DEVICE_LOOP_MIN_P {
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
            mixed_precision: MixedPrecisionPolicy::Off,
            ..Default::default()
        };
        assert!(!pol.iterative_refinement_should_attempt(1024));
    }

    #[test]
    fn never_policy_never_attempts_refinement() {
        let pol = GpuDispatchPolicy {
            mixed_precision: MixedPrecisionPolicy::Never,
            ..Default::default()
        };
        assert!(!pol.iterative_refinement_should_attempt(1024));
    }
}
