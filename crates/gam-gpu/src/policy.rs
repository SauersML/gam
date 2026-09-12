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
    /// `crate::calibration::calibrated_policy_for_device` after the CUDA
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
    /// The smallest `gemm_min_flops` ANY production dispatch policy can carry.
    ///
    /// Production policies are exactly two: [`Self::default`] (seed,
    /// `gemm_min_flops = 1e8`) and the device-calibrated policy, whose
    /// `crossover_flops` can lower the floor at most to the flop count of the
    /// smallest calibration measurement — the 64×64×64 GEMM in
    /// `calibration::GEMM_DIMS`, i.e. `2·64³ = 524_288` (a compile-time assert
    /// in `calibration.rs` pins the correspondence). Work below this floor is
    /// therefore inadmissible for GPU dispatch under EVERY reachable policy, so
    /// a caller may refuse it BEFORE probing the device — this is the pre-probe
    /// size gate that lets CPU-sized problems skip CUDA context creation
    /// entirely (the startup-tax ordering fix). Work at or above it must fall
    /// through to the probed runtime's real (possibly calibrated) policy gate,
    /// so genuinely GPU-sized problems behave exactly as before.
    pub const MIN_CALIBRATABLE_GEMM_FLOPS: u128 = 524_288;

    /// The smallest `potrf_min_p` ANY production dispatch policy can carry:
    /// the smallest POTRF calibration dimension (`calibration::POTRF_DIMS[0]`,
    /// pinned by a compile-time assert there). A single (batch ≤ 1) POTRF with
    /// `p` below this is inadmissible under every reachable policy.
    pub const MIN_CALIBRATABLE_POTRF_P: usize = 64;

    /// The smallest `row_kernel_min_n` / `xtwx_n_min` ANY production dispatch
    /// policy can carry: the smallest XtWX calibration row count
    /// (`calibration::XTWX_DIMS[0].0`, pinned by a compile-time assert there).
    /// A row-kernel workload with fewer rows is inadmissible under every
    /// reachable policy, so per-fit GPU-eligibility deciders may refuse it
    /// BEFORE probing the device.
    pub const MIN_CALIBRATABLE_ROW_KERNEL_N: usize = 2_048;

    /// The smallest `fused_kernel_min_n` ANY production dispatch policy can
    /// carry.
    ///
    /// Device calibration derives the fused-kernel crossover as twice its
    /// measured row-kernel crossover. The calibration grid pins that row floor
    /// to [`Self::MIN_CALIBRATABLE_ROW_KERNEL_N`], so a smaller fused batch is
    /// inadmissible under every reachable policy and can remain on the CPU
    /// without probing CUDA merely to discover the device-specific threshold.
    pub const MIN_CALIBRATABLE_FUSED_KERNEL_N: usize =
        2 * Self::MIN_CALIBRATABLE_ROW_KERNEL_N;

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

    /// Whether a batched Pólya-Gamma draw of `n` rows is worth dispatching to
    /// the device.
    ///
    /// A PG batch is a fused elementwise kernel: one independent rejection
    /// sampler per row, no reduction and no cross-row reuse. So the row count
    /// *is* the work, and what the device has to overcome is launch latency
    /// plus the `n·(4 + 8)` bytes staged in and `n·8` staged back out — a
    /// transfer/launch amortisation question rather than an arithmetic-intensity
    /// one. That is exactly what `fused_kernel_min_n` carries: calibration sets
    /// it to twice the device's *measured* XtWX crossover row count, so the
    /// crossover is a per-device measurement rather than a tuned literal, and a
    /// faster host CPU moves it up on that host instead of failing the kernel.
    #[inline]
    pub const fn polya_gamma_batch_target_is_gpu(&self, n: usize) -> bool {
        n >= self.fused_kernel_min_n
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

    /// Batched arithmetic of one reduced-Schur PCG solve over a matrix-free SAE
    /// Kronecker system: `cg_iters` applies of `S·x`. Per row block `i` an apply
    /// runs the forward cross-block GEMV `v_i = H_tβ^(i)·x` (`2·d·k`
    /// multiply-adds, with the per-row latent depth `d` as the M-frame width and
    /// `k` the border), a `d×d` triangular solve through the cached Cholesky
    /// factor (`d²`) and the transpose GEMV `H_βt^(i)·w_i` (`2·d·k`), so one apply
    /// is `n·(4·d·k + d²)`. It is keyed on the frame depth and the border width,
    /// not row count alone, so LLM shapes (few rows, wide `k`, modest `d`)
    /// register the arithmetic a row-count gate misses.
    const fn reduced_schur_matvec_solve_flops(
        n: usize,
        k: usize,
        d: usize,
        cg_iters: usize,
    ) -> u128 {
        let n = n as u128;
        let k = k as u128;
        let d = d as u128;
        let apply = n.saturating_mul(
            4u128
                .saturating_mul(d)
                .saturating_mul(k)
                .saturating_add(d.saturating_mul(d)),
        );
        apply.saturating_mul(cg_iters as u128)
    }

    /// Admission of a reduced-Schur PCG solve against a dense launch floor. The
    /// border must also clear the device-loop floor: below it the per-apply
    /// launch latency (one kernel sequence per matvec) dominates any arithmetic,
    /// however many CG iterations run.
    const fn reduced_schur_matvec_admits(
        n: usize,
        k: usize,
        d: usize,
        cg_iters: usize,
        dense_launch_flops_min: u128,
    ) -> bool {
        n > 0
            && d > 0
            && cg_iters > 0
            && k >= Self::DEVICE_LOOP_MIN_P
            && Self::reduced_schur_matvec_solve_flops(n, k, d, cg_iters) >= dense_launch_flops_min
    }

    /// Work-based admission for offloading the **reduced-Schur PCG matvec** (the
    /// InexactPCG hot loop for matrix-free SAE β-blocks) to the device.
    ///
    /// The dense gates key on row count (`xtwx_n_min`, `row_kernel_min_n`) or on
    /// one big factorization's flops, and the SAE LLM shape `(n≈2000) × (k≈2048)
    /// × (d≈8)` trips neither: it is thousands of small dense ops. But a CG solve
    /// stages the row frames once and reuses them for `cg_iters` applies, so its
    /// cost profile is one staging plus `cg_iters·n·(4·d·k + d²)` batched
    /// arithmetic, the profile of one dense launch of that many flops. The
    /// admission floor is therefore this policy's dense launch crossover
    /// (`dense_reduction_flops_min`), which device calibration measures per device
    /// (`calibration::calibrate_device`). The host matvec (gather/scatter plus
    /// small triangular solves) is slower per flop than the host GEMM that
    /// crossover is measured against, so this floor can only under-admit relative
    /// to a matvec-specific measurement.
    ///
    /// Pure function of the shape and the policy: no device is needed to evaluate
    /// it. The caller falls back to the CPU matvec whenever the backend build
    /// declines, so admission changes only where the `Σ_i Y_iᵀ(Y_i x)` flops run.
    ///
    /// * `n`        — number of row blocks (SAE observations / latent rows).
    /// * `k`        — border β width (the SAE decoder atom count `K`).
    /// * `d`        — per-row latent / active-frame depth (the M dimension).
    /// * `cg_iters` — the PCG iteration budget the solve launches with; the frames
    ///   stay resident across iterations, so the per-apply work multiplies by it.
    ///
    /// The probed runtime's calibrated policy gates the device matvec backends in
    /// `gam_solve::gpu_kernels::arrow_schur` with this predicate. Callers deciding
    /// before the device probe use [`Self::reduced_schur_matvec_admissible_under_any_policy`].
    pub const fn reduced_schur_matvec_should_offload(
        &self,
        n: usize,
        k: usize,
        d: usize,
        cg_iters: usize,
    ) -> bool {
        Self::reduced_schur_matvec_admits(n, k, d, cg_iters, self.dense_reduction_flops_min())
    }

    /// True when SOME reachable dispatch policy could admit this reduced-Schur PCG
    /// solve: [`Self::reduced_schur_matvec_should_offload`] at the most permissive
    /// dense launch floor any production policy can carry,
    /// [`Self::MIN_CALIBRATABLE_GEMM_FLOPS`]. Calibration cannot lower `gemm_min_flops`
    /// below it, and the XtWX crossover cannot calibrate below its smallest
    /// measurement, which exceeds it. So a `false` is exact for every policy and
    /// the caller may stay on the CPU without resolving GPU availability (whose
    /// first call creates a CUDA primary context on every GPU). A `true` decides
    /// nothing: the probed runtime's policy still gates the offload.
    pub const fn reduced_schur_matvec_admissible_under_any_policy(
        n: usize,
        k: usize,
        d: usize,
        cg_iters: usize,
    ) -> bool {
        Self::reduced_schur_matvec_admits(n, k, d, cg_iters, Self::MIN_CALIBRATABLE_GEMM_FLOPS)
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

/// Admission descriptor for routing the *outer* REML BFGS-over-ρ loop onto a
/// fully device-resident driver (rather than the host orchestrator that hops
/// out per step).
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

/// Inputs to `should_use_gpu_pirls_loop`. Each field comes from data the
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
    /// lossless Auto resolution returned an available runtime).
    pub gpu_available: bool,
}

impl GpuDispatchPolicy {
    /// Minimum design column count for the device-resident inner/outer loops.
    ///
    /// Below this width the per-iteration `XᵀWX + Cholesky` is dominated by
    /// launch latency and PCIe staging rather than arithmetic, so the host LM
    /// loop (which populates the full `PirlsResult` surface as a free
    /// side-effect) is strictly cheaper. Shared by the inner PIRLS admission
    /// predicate and the reduced-Schur matvec gate so they cannot drift apart.
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
mod fused_batch_dispatch_tests {
    use super::*;

    /// The dominant large-scale PG draw shape — one variate per data row per
    /// Gibbs iteration — is admitted, and a batch small enough that launch and
    /// staging dominate is refused. The refusal is the load-bearing half: a
    /// predicate that admitted everything would let a dispatch-worthiness test
    /// pass without saying anything about the shape it ran.
    #[test]
    fn polya_gamma_admits_large_batch_and_refuses_small() {
        let pol = GpuDispatchPolicy::default();
        assert!(pol.polya_gamma_batch_target_is_gpu(200_000));
        assert!(pol.polya_gamma_batch_target_is_gpu(pol.fused_kernel_min_n));
        assert!(!pol.polya_gamma_batch_target_is_gpu(pol.fused_kernel_min_n - 1));
        assert!(!pol.polya_gamma_batch_target_is_gpu(16));
        assert!(!pol.polya_gamma_batch_target_is_gpu(0));
    }

}

#[cfg(test)]
mod reduced_schur_matvec_offload_tests {
    use super::*;

    /// A policy carrying the most permissive dense launch floor calibration can
    /// reach.
    fn most_permissive_policy() -> GpuDispatchPolicy {
        let floor = usize::try_from(GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS)
            .expect("calibration floor fits usize");
        GpuDispatchPolicy {
            gemm_min_flops: floor,
            xtwx_flops_min: floor,
            ..GpuDispatchPolicy::default()
        }
    }

    /// The LLM/SAE shape the #1017 Phase-1 re-keying targets: a few thousand row
    /// blocks, a wide border and a modest frame depth. The row-count gate (50k)
    /// misses it, but one apply is `2_000·(4·8·2_048 + 8²) ≈ 1.3e8` flops, which
    /// clears even the uncalibrated seed floor (`1e8`) at a single CG iteration.
    #[test]
    fn admits_llm_sae_matvec_shape() {
        let pol = GpuDispatchPolicy::default();
        assert!(pol.reduced_schur_matvec_should_offload(2_000, 2_048, 8, 1));
        assert!(pol.reduced_schur_matvec_should_offload(2_000, 2_048, 8, 8));
        assert!(GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(
            2_000, 2_048, 8, 1
        ));
        // The row-count-style dense gate rejects the same shape, confirming the
        // work re-keying is what admits it.
        assert!(!pol.dense_hessian_work_target_is_gpu(2_000, 8));
    }

    /// #1783: thin-curve (`d_atom = 1`) dictionaries at token scale. Their summed
    /// work lies below the uncalibrated seed floor but above the most permissive
    /// calibrated floor, so they stay eligible before the probe and the device's
    /// measured crossover decides after it.
    #[test]
    fn thin_curve_atoms_are_eligible_and_decided_by_the_calibrated_floor() {
        // 40_456·(4·256 + 1) ≈ 4.1e7 and 24_576·(4·64 + 1) ≈ 6.3e6 flops.
        for &(n, k) in &[(40_456usize, 256usize), (24_576, 64)] {
            assert!(GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(n, k, 1, 1));
            assert!(most_permissive_policy().reduced_schur_matvec_should_offload(n, k, 1, 1));
            assert!(!GpuDispatchPolicy::default().reduced_schur_matvec_should_offload(n, k, 1, 1));
        }
        assert!(!GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(300, 6, 1, 8));
    }

    /// The any-policy predicate is the per-policy predicate at the most permissive
    /// floor, and every reachable floor is at least that floor, so a shape the
    /// any-policy predicate rejects is rejected by the seed policy as well.
    #[test]
    fn any_policy_predicate_bounds_every_policy() {
        let permissive = most_permissive_policy();
        let seed = GpuDispatchPolicy::default();
        for &(n, k, d, cg_iters) in &[
            (2_000usize, 2_048usize, 8usize, 8usize),
            (40_456, 256, 1, 1),
            (200, GpuDispatchPolicy::DEVICE_LOOP_MIN_P, 4, 1),
            (200, GpuDispatchPolicy::DEVICE_LOOP_MIN_P, 4, 1_000),
            (30, 8, 2, 8),
            (64, 9, 1, 1),
        ] {
            let eligible =
                GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(n, k, d, cg_iters);
            assert_eq!(
                eligible,
                permissive.reduced_schur_matvec_should_offload(n, k, d, cg_iters)
            );
            if !eligible {
                assert!(!seed.reduced_schur_matvec_should_offload(n, k, d, cg_iters));
            }
        }
    }

    /// Tiny shapes where launch latency dominates stay on the CPU under every
    /// policy: a border below `DEVICE_LOOP_MIN_P`.
    #[test]
    fn rejects_tiny_shape_where_transfer_dominates() {
        assert!(!GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(30, 8, 2, 8));
        // The 300×8 "stay CPU" canary of the production seam tests.
        assert!(!GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(
            300, 8, 4, 16
        ));
    }

    /// A border below the device-loop floor is rejected however much row or
    /// iteration work piles up.
    #[test]
    fn rejects_narrow_border_even_with_huge_row_count() {
        let narrow = GpuDispatchPolicy::DEVICE_LOOP_MIN_P - 1;
        assert!(!GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(
            1_000_000, narrow, 64, 64
        ));
    }

    /// Degenerate dimensions are never offloaded (no work, or no solve).
    #[test]
    fn rejects_degenerate_dimensions() {
        let pol = most_permissive_policy();
        assert!(!pol.reduced_schur_matvec_should_offload(0, 2_048, 8, 8));
        assert!(!pol.reduced_schur_matvec_should_offload(2_000, 0, 8, 8));
        assert!(!pol.reduced_schur_matvec_should_offload(2_000, 2_048, 0, 8));
        assert!(!pol.reduced_schur_matvec_should_offload(2_000, 2_048, 8, 0));
    }

    /// Monotone in the CG budget: more applies over the same resident frames only
    /// add arithmetic against the same one-time staging.
    #[test]
    fn monotone_in_cg_iters() {
        let pol = GpuDispatchPolicy::default();
        // One apply is 200·(4·4·32 + 16) = 105_600 flops, so the seed floor 1e8
        // is crossed between 946 and 947 iterations.
        let (n, k, d) = (200usize, GpuDispatchPolicy::DEVICE_LOOP_MIN_P, 4usize);
        assert!(!pol.reduced_schur_matvec_should_offload(n, k, d, 946));
        assert!(pol.reduced_schur_matvec_should_offload(n, k, d, 947));
        assert!(pol.reduced_schur_matvec_should_offload(n, k, d, 5_000));
    }

    /// The solve arithmetic counts the forward and transpose cross-block GEMVs and
    /// the per-row solve: `cg_iters·n·(4·d·k + d²)`.
    #[test]
    fn solve_flops_count_both_cross_block_gemvs_and_the_row_solve() {
        assert_eq!(
            GpuDispatchPolicy::reduced_schur_matvec_solve_flops(2_000, 2_048, 8, 3),
            3 * 2_000 * (4 * 8 * 2_048 + 8 * 8)
        );
        assert_eq!(GpuDispatchPolicy::reduced_schur_matvec_solve_flops(1, 1, 1, 1), 5);
    }
}

