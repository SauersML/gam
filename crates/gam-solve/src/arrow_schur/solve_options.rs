//! Solver configuration and the batched block-solve abstraction: BA solver
//! modes, PCG/trust-region/mixed-precision/proximal options, diagnostics, and
//! the [`BatchedBlockSolver`] trait with its CPU implementation.

use super::*;

/// BA Schur solve variant for the reduced shared `β` system.
///
/// * [`ArrowSolverMode::Direct`] is BA's dense reduced-camera-system solve:
///   eliminate the per-point/per-row blocks, form the reduced system, and
///   Cholesky factor it. This is the Ceres/g2o default for modest camera
///   counts and is appropriate here for `K <= 2000`.
///   **GPU support: ✓** — requires dense H_ββ and dense per-row H_tβ slabs.
///
/// * [`ArrowSolverMode::SqrtBA`] ports Square-Root BA (Demmel/Gao/Gu et al.,
///   CVPR 2021): Schur terms are formed as `(L_i^-1 H_tβ_i)^T
///   (L_i^-1 H_tβ_i)` from the per-row square-root factor `L_i`, avoiding
///   explicit `H_tt^-1 H_tβ` products. It is the preferred direct path when
///   single-precision assembly is introduced or when row blocks are poorly
///   conditioned.
///   **GPU support: ✓** — requires dense H_ββ and dense per-row H_tβ slabs.
///
/// * [`ArrowSolverMode::InexactPCG`] ports "Bundle Adjustment in the Large"
///   (Agarwal et al.): the Schur system is solved inexactly by PCG with a
///   Jacobi Schur preconditioner, avoiding dense `K × K` factorization for
///   SAE-manifold scale shared systems.
///   **GPU support: CPU only** until the row-procedural H_tβ GPU PCG path
///   (issue #288 Part B) is wired. The topology selector must not request
///   `InexactPCG` via the GPU entry point; `solve_arrow_newton_step` returns
///   `GpuRequiresDenseSystem` for matrix-free systems, and the wrapper in
///   `solver/gpu/arrow_schur_gpu.rs` routes those to CPU InexactPCG
///   automatically. At K ≥ 5000 the GPU PCG path will supersede the CPU path
///   once the row-procedural H_tβ kernel and boxed GPU matvec backend in
///   `run_pcg_with_preconditioner` are wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowSolverMode {
    Direct,
    SqrtBA,
    InexactPCG,
}

impl ArrowSolverMode {
    /// BA-size heuristic: dense RCS for modest `K`, inexact Schur PCG for
    /// large shared systems. This follows Agarwal et al.'s direct-vs-iterative
    /// split for large BA, mapped from cameras to decoder coefficients.
    pub const fn automatic(k: usize) -> Self {
        if k <= DIRECT_SOLVE_MAX_K {
            Self::Direct
        } else {
            Self::InexactPCG
        }
    }

    /// Square-Root BA is the direct-solve stability mode for future f32
    /// callers. Large `K` still routes to inexact PCG because dense Schur
    /// storage dominates precision concerns at that scale.
    pub const fn automatic_for_single_precision(k: usize) -> Self {
        if k <= DIRECT_SOLVE_MAX_K {
            Self::SqrtBA
        } else {
            Self::InexactPCG
        }
    }
}

/// Reason the Steihaug-CG loop stopped.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PcgStopReason {
    /// Residual fell below the relative tolerance threshold.
    #[default]
    Converged,
    /// Loop exhausted max_iterations without converging.
    MaxIter,
    /// Step hit the trust-region boundary (Steihaug boundary projection).
    TrustRegion,
    /// Negative curvature detected in an unbounded solve.
    Indefinite,
    /// Non-positive or non-finite preconditioned residual after an update.
    Stagnation,
}

/// Per-solve instrumentation counters returned alongside the PCG solution.
///
/// All fields default to zero; callers that do not need diagnostics simply
/// ignore the value. The struct is Copy so passing it through return tuples
/// is zero-overhead.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArrowPcgDiagnostics {
    /// Number of CG iterations executed.
    pub iterations: usize,
    /// Total calls to the Schur matvec A·p.
    pub matvec_calls: usize,
    /// Total calls to the preconditioner M^{-1}·r.
    pub precond_apply_calls: usize,
    /// Number of times the LM ridge was escalated before a successful factor.
    pub ridge_escalations: usize,
    /// Relative residual at termination; 0.0 when the RHS was zero.
    pub final_relative_residual: f64,
    /// Why the loop stopped.
    pub stopping_reason: PcgStopReason,
    /// Mixed-precision certificate outcome for this solve.
    pub mixed_precision_status: MixedPrecisionStatus,
    /// True only when the reduced-Schur solve was **actually executed on the
    /// device**: either the fully device-resident batched Arrow-Schur Direct
    /// sequence (`try_device_arrow_direct` → `solve_arrow_newton_step`) or the
    /// device-resident matrix-free SAE PCG (`solve_sae_matrix_free_pcg`, which
    /// runs the matvec in CUDA kernels over device-resident frames). It is NOT
    /// set merely because a GPU runtime exists and a dispatch gate fired (#1209).
    pub used_device_arrow: bool,
    /// True when a reduced-Schur matvec backend was injected through
    /// `maybe_inject_gpu_schur_matvec` but the matvec itself runs as a
    /// **host** (CPU Rust/Rayon) procedural closure — both the matrix-free
    /// `build_row_procedural_matvec` branch and the `cuda::build_schur_matvec_backend`
    /// branch return host closures that evaluate `Σ_i Y_iᵀ(Y_i x)` on the CPU,
    /// even when a CUDA context was opened to build the per-row factors. This
    /// path must NOT report `used_device_arrow`: the arithmetic is host-side
    /// (#1209). Distinct field so perf accounting never mistakes a host
    /// procedural matvec for true device execution.
    pub injected_host_procedural_matvec: bool,
}

/// Outcome of an opt-in mixed-precision arrow solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MixedPrecisionStatus {
    /// The caller did not request mixed precision or this solve mode cannot use it.
    #[default]
    Off,
    /// The f32 factor solve was refined until the f64 backward-error certificate held.
    Certified { refinement_steps: usize },
    /// The kappa gate or solve shape rejected mixed precision and the f64 path ran.
    /// The declining reason is logged at `info` level when the fallback fires.
    F64Fallback,
}

/// PCG controls for BA's inexact reduced-camera-system solve.
///
/// The defaults mirror the loose inner tolerances used by inexact-step LM in
/// "Bundle Adjustment in the Large": solve the Schur system only accurately
/// enough for a useful trust-region step, then let the outer LM iteration
/// correct the remaining error.
#[derive(Debug, Clone)]
pub struct ArrowPcgOptions {
    pub max_iterations: usize,
    pub relative_tolerance: f64,
}

impl Default for ArrowPcgOptions {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_PCG_MAX_ITERATIONS,
            relative_tolerance: DEFAULT_PCG_RELATIVE_TOLERANCE,
        }
    }
}

/// Trust-region controls for Steihaug-CG on the reduced BA system.
///
/// This is the Ceres-style guard around LM: `ridge_t`/`ridge_beta` provide
/// Levenberg damping, while the trust radius bounds the reduced shared step
/// in Euclidean β coordinates using Steihaug's truncated-CG stopping rules for
/// boundary hits and negative curvature.
#[derive(Debug, Clone)]
pub struct ArrowTrustRegionOptions {
    pub radius: f64,
    pub steihaug_relative_tolerance: f64,
    pub max_iterations: usize,
}

impl Default for ArrowTrustRegionOptions {
    fn default() -> Self {
        Self {
            radius: DEFAULT_TRUST_REGION_RADIUS,
            steihaug_relative_tolerance: DEFAULT_PCG_RELATIVE_TOLERANCE,
            max_iterations: DEFAULT_PCG_MAX_ITERATIONS,
        }
    }
}

/// Opt-in Carson--Higham mixed-precision refinement for dense arrow solves.
///
/// Default is [`ArrowSolvePrecisionPolicy::F64Only`]: exact f64 solves remain the default.
/// [`ArrowSolvePrecisionPolicy::CertifiedMixed`] stores f32 copies of the per-row Cholesky
/// factors and dense Schur factor, solves corrections in f32, and recomputes the
/// residual in f64 against the original arrow blocks. The standard refinement
/// certificate is the normwise backward error
///
/// `||r||_inf / (||H||_inf ||x||_inf + ||b||_inf) <= residual_relative_tolerance`.
///
/// The kappa gate enforces `kappa_estimate * u_f32 < kappa_unit_roundoff_margin`;
/// when it fails, the solve reports [`MixedPrecisionStatus::F64Fallback`] and
/// logs the reason before using the f64 path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArrowSolvePrecisionPolicy {
    F64Only,
    CertifiedMixed {
        max_refinement_steps: usize,
        residual_relative_tolerance: f64,
        kappa_unit_roundoff_margin: f64,
    },
}

impl Default for ArrowSolvePrecisionPolicy {
    fn default() -> Self {
        Self::F64Only
    }
}

impl ArrowSolvePrecisionPolicy {
    pub fn certified_mixed() -> Self {
        Self::CertifiedMixed {
            max_refinement_steps: DEFAULT_MIXED_PRECISION_MAX_REFINEMENTS,
            residual_relative_tolerance: DEFAULT_MIXED_PRECISION_CERTIFICATE_TOLERANCE,
            kappa_unit_roundoff_margin: DEFAULT_MIXED_PRECISION_KAPPA_MARGIN,
        }
    }

    pub(crate) fn is_enabled(self) -> bool {
        matches!(self, ArrowSolvePrecisionPolicy::CertifiedMixed { .. })
    }
}

/// Complete BA Schur solve options.
///
/// Use [`ArrowSolveOptions::automatic`] for normal latent-coordinate fits;
/// use [`ArrowSolveOptions::sqrt_ba`] when the assembler has single-precision
/// row blocks or an ill-conditioned gauge; use [`ArrowSolveOptions::inexact_pcg`]
/// for SAE-manifold scale `K`.
#[derive(Clone)]
pub struct ArrowSolveOptions {
    pub mode: ArrowSolverMode,
    pub pcg: ArrowPcgOptions,
    pub trust_region: ArrowTrustRegionOptions,
    /// Row chunk size for streaming direct/Square-Root Schur assembly.
    pub streaming_chunk_size: Option<usize>,
    /// Use the Riemannian latent projection before the Schur reduction. The
    /// reduced Steihaug solve itself remains in Euclidean β coordinates.
    pub riemannian_trust_region: bool,
    /// Optional GPU-backed Schur matvec for CPU-driven `InexactPCG` at K ≥ 5000.
    ///
    /// When set, `run_pcg_with_preconditioner` delegates each `S·p` call to
    /// this closure instead of the CPU `schur_matvec`. Constructed by
    /// `crate::gpu_kernels::arrow_schur::gpu_schur_matvec_backend` when `cuda_selected()`
    /// and the system has dense per-row H_tβ slabs. `None` means CPU-only PCG.
    pub gpu_matvec: Option<GpuSchurMatvec>,
    /// Skip the ill-conditioning *rejection* (the κ-based
    /// [`ArrowSchurError::PerRowFactorIllConditioned`] per-row guard and the
    /// matching reduced-Schur κ guard) while still requiring genuine positive
    /// definiteness (a non-PD Cholesky pivot still errors).
    ///
    /// The κ guards exist to protect the accuracy of the Newton *step*: a
    /// barely-PD `H_tt^(i)` or an over-conditioned reduced Schur yields an
    /// inaccurate `Δβ`/`Δt`. Evidence-only callers
    /// (e.g. `SaeManifoldTerm::penalized_quasi_laplace_criterion_with_cache`) do not consume the
    /// step — they need only the factor cache for the log-determinant
    /// (`½log|H|`, exact from `diag(L)` regardless of κ) and the selected-inverse
    /// traces. For those callers the κ rejection is a false abort when ρ sweeps
    /// to extreme values, so this flag lifts it and hands the
    /// "is this step trustworthy" decision back to the caller.
    ///
    /// Default `false`: ordinary solves keep the full guard.
    pub tolerate_ill_conditioning: bool,
    /// Arrow solve precision policy. Default is f64-only.
    pub solve_precision: ArrowSolvePrecisionPolicy,
    /// Optional spectral positive-definiteness floor on the *reduced Schur
    /// complement* `S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`,
    /// as a relative fraction of `S`'s largest eigenvalue.
    ///
    /// `None` (default) keeps the strict contract: a non-PD `S` errors as
    /// [`ArrowSchurError::SchurFactorFailed`] so the LM outer loop lifts
    /// `ridge_beta` globally and re-forms `S`.
    ///
    /// `Some(floor)` engages the #1026 SAE co-collapse cure on the SOLVE path:
    /// when the reduced Schur Cholesky refuses (collapsed atoms drive a per-row
    /// `H_tt` near-singular, so the accumulated `(H_tt)⁻¹` over-subtracts `S`
    /// into an INDEFINITE matrix), instead of rejecting and over-damping every
    /// β direction with a global ridge, symmetric-eigendecompose `S` and clamp
    /// every eigenvalue UP to `floor·max(λ)`. This is Levenberg–Marquardt
    /// restricted to exactly the indefinite/collapsed subspace: the
    /// well-conditioned β directions (`λ ≫ floor·max λ`) are untouched and the
    /// step in those directions is the exact Newton step, while only the
    /// collapsed directions receive the minimal damping needed for a PD solve.
    /// The inner Newton then makes a real descent step rather than crawling
    /// behind an inflated global ridge. Mirrors the per-row spectral floor the
    /// evidence path uses for #1377/#1117/#1118
    /// ([`super::factorization::factor_spectral_deflated_evidence_row`]); the
    /// difference is the floored value — a small positive `floor·max λ`
    /// (Tikhonov) for the solve, vs unit stiffness `+1` (`log 1 = 0`) for the
    /// evidence log-det.
    ///
    /// Only consulted by the dense Direct / SqrtBA reduced solve (the only
    /// caller of [`super::reduced_solve::solve_dense_reduced_system`]); the
    /// InexactPCG path is unaffected.
    pub schur_pd_floor: Option<f64>,
    /// #1017 device-resident framed SAE frame for the LM ridge ladder.
    ///
    /// When set (by [`super::newton_step::solve_with_lm_escalation_inner`] on a
    /// device-admitted matrix-free SAE system), both the Direct SAE-PCG seam
    /// ([`super::newton_step::try_device_arrow_direct_sae_pcg`]) and the native
    /// large-border InexactPCG branch recompute only the ridge-dependent per-row
    /// `ainv` per ladder trial and reuse the resident ridge-independent operand
    /// buffers, instead of re-marshalling and re-uploading every operand through
    /// `flatten_device_sae_frame_data` on each trial. The solve is bit-identical;
    /// only the redundant per-trial upload is removed. `None` (default) keeps the
    /// per-trial re-flatten path. A trait object (like [`GpuSchurMatvec`]) keeps
    /// the CUDA-only device buffers out of these cfg-independent options.
    pub sae_resident_frame:
        Option<std::sync::Arc<dyn crate::gpu_kernels::arrow_schur::SaeResidentFrame + Send + Sync>>,
}

impl std::fmt::Debug for ArrowSolveOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowSolveOptions")
            .field("mode", &self.mode)
            .field("pcg", &self.pcg)
            .field("trust_region", &self.trust_region)
            .field("streaming_chunk_size", &self.streaming_chunk_size)
            .field("riemannian_trust_region", &self.riemannian_trust_region)
            .field("gpu_matvec", &self.gpu_matvec.is_some())
            .field("tolerate_ill_conditioning", &self.tolerate_ill_conditioning)
            .field("solve_precision", &self.solve_precision)
            .field("schur_pd_floor", &self.schur_pd_floor)
            .field("sae_resident_frame", &self.sae_resident_frame.is_some())
            .finish()
    }
}

/// Globalization guard for non-convex arrow-Schur inner steps.
///
/// The raw Schur solve is exactly Newton. For non-convex analytic penalties,
/// full Newton can cycle. This controller adds a proximal LM shift `mu I` to
/// both blocks and accepts only Armijo-decreasing trial points.
#[derive(Debug, Clone)]
pub struct ArrowProximalCorrectionOptions {
    pub initial_ridge: f64,
    pub ridge_growth: f64,
    pub max_attempts: usize,
    pub armijo_c1: f64,
    pub gradient_tolerance: f64,
    /// Relative objective resolution below which the proximal correction
    /// declares convergence instead of failing.
    ///
    /// Near a stationary point the largest decrease the damped Newton model can
    /// still achieve shrinks to the floating-point resolution of the objective
    /// itself: at proximal ridge `μ → μ_max` the accepted step length is
    /// `O(‖g‖ / μ)`, so the realised change in the objective falls below
    /// `rel_tol · (|f| + 1)`. At that scale the Armijo sufficient-decrease test
    /// compares two values that differ only by rounding noise, and no further
    /// productive decrease is achievable. Rather than raise
    /// `AdaptiveCorrectionFailed`, the loop then returns the incumbent state
    /// (a zero step) as converged. This does NOT mask genuine non-convergence:
    /// it triggers only when every attempted step either fails to decrease the
    /// objective by more than this resolution OR increases it by no more than
    /// this resolution (pure rounding). A step that genuinely reduces the
    /// objective is always taken first.
    pub convergence_objective_rel_tol: f64,
}

impl Default for ArrowProximalCorrectionOptions {
    fn default() -> Self {
        Self {
            initial_ridge: DEFAULT_PROXIMAL_INITIAL_RIDGE,
            ridge_growth: DEFAULT_PROXIMAL_RIDGE_GROWTH,
            max_attempts: DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            armijo_c1: DEFAULT_ARMIJO_C1,
            gradient_tolerance: DEFAULT_GRADIENT_TOLERANCE,
            convergence_objective_rel_tol: DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL,
        }
    }
}

/// Accepted proximal arrow-Schur step and the damping that made it descent.
#[derive(Debug, Clone)]
pub struct ArrowAcceptedProximalStep {
    pub delta_t: Array1<f64>,
    pub delta_beta: Array1<f64>,
    pub ridge_t: f64,
    pub ridge_beta: f64,
    pub proximal_ridge: f64,
    pub objective_value: f64,
    pub trial_objective_value: f64,
    pub gradient_dot_step: f64,
    pub attempts: usize,
}

impl ArrowSolveOptions {
    /// Select Direct for `K <= 2000` and InexactPCG above, following BA RCS
    /// practice for dense-vs-iterative reduced systems.
    pub fn automatic(k: usize) -> Self {
        Self {
            mode: ArrowSolverMode::automatic(k),
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
            tolerate_ill_conditioning: false,
            solve_precision: ArrowSolvePrecisionPolicy::F64Only,
            schur_pd_floor: None,
            sae_resident_frame: None,
        }
    }

    /// Force dense reduced-camera-system Cholesky, the classic BA direct
    /// solve for small `K`.
    pub fn direct() -> Self {
        Self {
            mode: ArrowSolverMode::Direct,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
            tolerate_ill_conditioning: false,
            solve_precision: ArrowSolvePrecisionPolicy::F64Only,
            schur_pd_floor: None,
            sae_resident_frame: None,
        }
    }

    /// Force Square-Root BA Schur assembly for the direct reduced solve.
    pub fn sqrt_ba() -> Self {
        Self {
            mode: ArrowSolverMode::SqrtBA,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
            tolerate_ill_conditioning: false,
            solve_precision: ArrowSolvePrecisionPolicy::F64Only,
            schur_pd_floor: None,
            sae_resident_frame: None,
        }
    }

    /// Force inexact BA Schur PCG with Jacobi preconditioning.
    pub fn inexact_pcg() -> Self {
        Self {
            mode: ArrowSolverMode::InexactPCG,
            pcg: ArrowPcgOptions::default(),
            trust_region: ArrowTrustRegionOptions::default(),
            streaming_chunk_size: None,
            riemannian_trust_region: false,
            gpu_matvec: None,
            tolerate_ill_conditioning: false,
            solve_precision: ArrowSolvePrecisionPolicy::F64Only,
            schur_pd_floor: None,
            sae_resident_frame: None,
        }
    }

    pub fn with_streaming_chunk_size(mut self, chunk_size: Option<usize>) -> Self {
        self.streaming_chunk_size = chunk_size.filter(|&chunk| chunk > 0);
        self
    }

    /// Lift the ill-conditioning *rejection* for evidence/log-det-only callers
    /// while still requiring genuine PD. See [`Self::tolerate_ill_conditioning`].
    ///
    /// Use this when the returned `(Δt, Δβ)` Newton step is discarded and only
    /// the factor cache is consumed (log-determinant + selected-inverse traces).
    /// The cache stays undamped at `ridge_t = 0`, so the log-determinant is
    /// exact regardless of κ.
    pub fn with_ill_conditioning_tolerated(mut self) -> Self {
        self.tolerate_ill_conditioning = true;
        self
    }

    /// Enable the spectral PD-floor on an indefinite reduced Schur (the SAE solve
    /// path): floor the collapsed / dead-atom directions up to `floor·max(λ)` and
    /// re-factor instead of hard-erroring. An overcomplete manifold-SAE fit parks
    /// surplus atoms dead, so the reduced Schur (and the undamped evidence factor
    /// at the optimum) can have near-zero / slightly-negative eigenvalues on the
    /// dead subspace; flooring those lets the live subspace's exact Newton /
    /// log-det proceed instead of aborting the whole fit on a non-PD pivot. `None`
    /// (default) keeps the strict refusal for BA / non-SAE callers.
    pub fn with_schur_pd_floor(mut self, floor: f64) -> Self {
        self.schur_pd_floor = Some(floor);
        self
    }

    pub fn with_solve_precision_policy(mut self, policy: ArrowSolvePrecisionPolicy) -> Self {
        self.solve_precision = policy;
        self
    }

    /// Turn certified mixed precision ON for the streaming/residency reduced
    /// solve unless the caller already pinned an explicit policy (#1014).
    ///
    /// Only `F64Only` (the inherited default) is upgraded to `CertifiedMixed`;
    /// a caller that deliberately set a policy keeps it. The reduced-Schur f64
    /// factor and every evidence log-determinant are unaffected — see
    /// [`mixed_precision_reduced_beta`].
    #[must_use]
    pub fn with_streaming_solve_precision_default(&self) -> Self {
        let mut out = self.clone();
        if matches!(out.solve_precision, ArrowSolvePrecisionPolicy::F64Only) {
            out.solve_precision = ArrowSolvePrecisionPolicy::certified_mixed();
        }
        out
    }
}

/// CPU/GPU seam for BA point-block work.
///
/// BA systems spend most time in independent point-block factorizations,
/// triangular solves, and Schur block products. MegBA maps exactly these
/// operations to GPU kernels. This trait keeps that boundary explicit so a
/// CUDA/Ceres backend can replace [`CpuBatchedBlockSolver`] without changing
/// `ArrowSchurSystem` algebra.
pub trait BatchedBlockSolver {
    /// Factor every per-row point block `H_tt^(i) + ridge_t I`, as in BA's
    /// point elimination stage.
    ///
    /// `tolerate_ill_conditioning` lifts the per-row κ rejection (still
    /// requiring genuine PD); see [`ArrowSolveOptions::tolerate_ill_conditioning`].
    fn factor_blocks(
        &self,
        rows: &[ArrowRowBlock],
        ridge_t: f64,
        d: usize,
        tolerate_ill_conditioning: bool,
    ) -> Result<ArrowFactorSlab, ArrowSchurError>;

    /// Solve one factored point block against a vector RHS.
    fn solve_block_vector(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView1<'_, f64>,
    ) -> Array1<f64>;

    /// Solve one factored point block against a dense matrix RHS.
    fn solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64>;

    /// Apply the Square-Root BA lower-triangular solve `L_i^-1 rhs`.
    fn sqrt_solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64>;

    /// Subtract a row-local Schur product from the dense reduced system.
    fn block_gemm_subtract(&self, schur: &mut Array2<f64>, left: &Array2<f64>, right: &Array2<f64>);
}

#[derive(Debug, Clone)]
pub struct ArrowRowGaugeDeflation {
    pub directions: Arc<[Vec<Array1<f64>>]>,
}

/// Orthonormal gauge basis on the reduced shared `beta` border.
///
/// Evidence factors use this as a Faddeev--Popov pin: for
/// `Q = [q_1, ..., q_r]` and `P = I - Q Q^T`, the represented reduced
/// operator is
///
/// `S_quot = P S P + Q Q^T`.
///
/// The quotient directions therefore contribute exactly `log(1) = 0` to the
/// Laplace log-determinant, while inverse/trace consumers apply
/// `P S_quot^-1 P`.  Ordinary Newton solves do not consult this carrier; it is
/// an evidence-coordinate contract only.
#[derive(Debug, Clone)]
pub struct ArrowBetaGaugeQuotient {
    pub directions: Arc<[Array1<f64>]>,
}

impl ArrowBetaGaugeQuotient {
    /// Orthonormalize a non-empty set of same-width border directions.
    ///
    /// A zero or linearly-dependent direction is a malformed gauge declaration,
    /// not a numerical condition to hide, so construction fails loudly.
    pub fn new(directions: Vec<Array1<f64>>) -> Result<Self, String> {
        if directions.is_empty() {
            return Err("ArrowBetaGaugeQuotient requires at least one direction".to_string());
        }
        let dim = directions[0].len();
        if dim == 0 {
            return Err("ArrowBetaGaugeQuotient directions must be non-empty".to_string());
        }
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(directions.len());
        for (direction_idx, mut direction) in directions.into_iter().enumerate() {
            if direction.len() != dim {
                return Err(format!(
                    "ArrowBetaGaugeQuotient direction {direction_idx} length {} != {dim}",
                    direction.len()
                ));
            }
            if direction.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "ArrowBetaGaugeQuotient direction {direction_idx} contains a non-finite value"
                ));
            }
            for existing in &basis {
                let coefficient = direction.dot(existing);
                direction.scaled_add(-coefficient, existing);
            }
            let norm_sq = direction.dot(&direction);
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
                return Err(format!(
                    "ArrowBetaGaugeQuotient direction {direction_idx} is zero or linearly dependent"
                ));
            }
            direction *= norm_sq.sqrt().recip();
            basis.push(direction);
        }
        Ok(Self {
            directions: Arc::from(basis.into_boxed_slice()),
        })
    }

    pub fn dimension(&self) -> usize {
        self.directions.len()
    }

    pub(crate) fn border_dim(&self) -> usize {
        self.directions[0].len()
    }

    /// `P x`, where `P = I - Q Q^T`.
    pub fn project_complement(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(x.len(), self.border_dim());
        let mut out = x.to_owned();
        for direction in self.directions.iter() {
            let coefficient = out.dot(direction);
            out.scaled_add(-coefficient, direction);
        }
        out
    }

    /// Dense Faddeev--Popov pin `P S P + Q Q^T`.
    pub fn pin_reduced_schur(&self, schur: ArrayView2<'_, f64>) -> Array2<f64> {
        let dim = self.border_dim();
        assert_eq!(schur.dim(), (dim, dim));

        // Apply P on the right, then on the left. Keeping the two projections
        // explicit makes the implementation the exact matrix analogue of the
        // matrix-free apply and avoids materializing a dense projector.
        let mut right = schur.to_owned();
        for direction in self.directions.iter() {
            let schur_q = right.dot(direction);
            for row in 0..dim {
                for col in 0..dim {
                    right[[row, col]] -= schur_q[row] * direction[col];
                }
            }
        }
        let mut pinned = right;
        for direction in self.directions.iter() {
            let q_t_s = direction.dot(&pinned);
            for row in 0..dim {
                for col in 0..dim {
                    pinned[[row, col]] -= direction[row] * q_t_s[col];
                }
            }
        }
        for direction in self.directions.iter() {
            for row in 0..dim {
                for col in 0..dim {
                    pinned[[row, col]] += direction[row] * direction[col];
                }
            }
        }
        // Both the input Schur and the mathematical pin are symmetric. Clear
        // the last-bit asymmetry from the two ordered dense projections before
        // Cholesky so direct and matrix-free paths expose one self-adjoint op.
        for row in 0..dim {
            for col in (row + 1)..dim {
                let value = 0.5 * (pinned[[row, col]] + pinned[[col, row]]);
                pinned[[row, col]] = value;
                pinned[[col, row]] = value;
            }
        }
        pinned
    }
}

impl ArrowRowGaugeDeflation {
    pub fn new(directions: Vec<Vec<Array1<f64>>>) -> Self {
        Self {
            directions: Arc::from(directions.into_boxed_slice()),
        }
    }

    pub(crate) fn row(&self, row: usize) -> &[Array1<f64>] {
        self.directions.get(row).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// Current CPU implementation of the BA batched block interface.
///
/// It is intentionally plain Rust loops because `d` is tiny. The trait shape,
/// not this implementation, is the load-bearing part for the future MegBA or
/// Ceres backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBatchedBlockSolver;

impl BatchedBlockSolver for CpuBatchedBlockSolver {
    fn factor_blocks(
        &self,
        rows: &[ArrowRowBlock],
        ridge_t: f64,
        d: usize,
        tolerate_ill_conditioning: bool,
    ) -> Result<ArrowFactorSlab, ArrowSchurError> {
        // Multi-GPU fast path: the per-row blocks `H_tt^(i) + ridge_t·I` are
        // independent same-size SPD systems — exactly the batch
        // `gam_gpu::try_cholesky_batched_lower_inplace` spreads across ALL
        // usable devices (the batched POTRF tiles over the pool). It is only
        // valid when every row is the uniform `d×d` shape; heterogeneous row
        // dimensions keep the per-row CPU loop because the current cuSOLVER
        // batched POTRF wrapper accepts one `(d, d)` shape per launch. It only
        // succeeds when EVERY block is PD at
        // the base ridge; a non-PD block returns `None`, so we fall back to the
        // exact per-row CPU path that performs minimal per-block ridge
        // escalation. After a successful batched factorization we re-apply the
        // identical κ-conditioning rejection `factor_one_row` enforces, so the
        // result is bit-for-bit equivalent (modulo IEEE reduction order) to the
        // CPU loop: a barely-PD but ill-conditioned block forces the whole batch
        // back onto the per-row path so its ridge can lift, never silently using
        // a contaminated factor.
        if let Some(batched) =
            try_factor_blocks_batched(rows, ridge_t, d, tolerate_ill_conditioning)
        {
            return Ok(batched);
        }
        // Per-row Cholesky factorizations are INDEPENDENT (each reads only its own
        // read-only `rows[i]` block), so factor rows in parallel then collect in
        // row order — the ordered `collect` reproduces the serial push order
        // bit-for-bit (no cross-row reduction; each block factored once). #1557 —
        // pin any nested faer GEMM inside each row worker to `Par::Seq`.
        let n = rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        let out = if parallel {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|row_idx| {
                    gam_problem::with_nested_parallel(|| {
                        factor_one_row(
                            &rows[row_idx],
                            ridge_t,
                            d,
                            row_idx,
                            tolerate_ill_conditioning,
                        )
                    })
                })
                .collect::<Result<Vec<_>, ArrowSchurError>>()?
        } else {
            let mut out = Vec::with_capacity(n);
            for (row_idx, row) in rows.iter().enumerate() {
                out.push(factor_one_row(
                    row,
                    ridge_t,
                    d,
                    row_idx,
                    tolerate_ill_conditioning,
                )?);
            }
            out
        };
        Ok(ArrowFactorSlab::from_blocks(out))
    }

    fn solve_block_vector(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        match (factor.nrows(), factor.ncols(), rhs.len()) {
            (1, 1, 1) => cholesky_solve_vector_fixed::<1>(factor, rhs),
            (2, 2, 2) => cholesky_solve_vector_fixed::<2>(factor, rhs),
            (3, 3, 3) => cholesky_solve_vector_fixed::<3>(factor, rhs),
            (4, 4, 4) => cholesky_solve_vector_fixed::<4>(factor, rhs),
            _ => cholesky_solve_vector(factor, rhs),
        }
    }

    fn solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        cholesky_solve_matrix(factor, rhs)
    }

    fn sqrt_solve_block_matrix(
        &self,
        factor: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        forward_substitution_lower_matrix(factor, rhs)
    }

    fn block_gemm_subtract(
        &self,
        schur: &mut Array2<f64>,
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) {
        // Performance: ndarray Array2 is row-major, so `right[[c, b]]` is
        // unit-strided in `b`. The canonical (a, b, c) order produced
        // strided reads of `left[[c, a]]` for every (a, b); reorder to
        // (c, a, b) so the inner `b`-loop is contiguous in `right` and
        // `left[[c, a]]` is hoisted out of the inner loop.
        //
        // Sparse SAE rows install a matrix-free `H_tβ` operator but the direct
        // reduced-Schur path still asks `row_htbeta` for a dense `(d_i × K)`
        // scratch matrix. Those rows have support on only the active atoms
        // (`top_k · basis · p` columns), so blindly iterating the full `K × K`
        // product made the Schur assembly compute-bound even though almost every
        // entry multiplied by zero (#1995). Discover the non-zero column support
        // of the two row factors once (`O(d_i·K)`) and multiply only the touched
        // columns (`O(d_i·nnz_left·nnz_right)`). Dense callers still take the same
        // arithmetic path with all columns active, while compact top-k rows scale
        // with the active support rather than the global border width.
        let k = schur.nrows();
        let d = left.nrows();
        assert_eq!(left.ncols(), k);
        assert_eq!(right.nrows(), d);
        assert_eq!(right.ncols(), k);
        assert_eq!(schur.ncols(), k);

        let mut left_active = Vec::with_capacity(k);
        let mut right_active = Vec::with_capacity(k);
        // HOT PATH (measured 2026-07-11, A10 real-workload profile: 50% of ALL
        // fit cycles in non-inlined bounds-checked ndarray 2-D `IndexMut`, plus
        // 13% self-time here): every scalar of the scan and of the rank-1
        // update went through `[[i, j]]` element access — a function call, a
        // bounds check, and 2-D stride arithmetic per double. Go through the
        // contiguous row slices and flat 1-D indexing instead. The operation
        // SEQUENCE is identical (same c → a → b order, same subtractions), so
        // the result is bit-exact — no reduction-order change.
        let schur_cols = schur.ncols();
        let schur_flat = schur
            .as_slice_mut()
            .expect("block_gemm_subtract: reduced Schur must be standard-layout");
        for c in 0..d {
            left_active.clear();
            right_active.clear();
            let left_row = left.row(c);
            let right_row = right.row(c);
            let left_row = left_row
                .as_slice()
                .expect("block_gemm_subtract: left row must be contiguous");
            let right_row = right_row
                .as_slice()
                .expect("block_gemm_subtract: right row must be contiguous");
            for col in 0..k {
                let l = left_row[col];
                let r = right_row[col];
                if l != 0.0 {
                    left_active.push((col, l));
                }
                if r != 0.0 {
                    right_active.push((col, r));
                }
            }
            if left_active.is_empty() || right_active.is_empty() {
                continue;
            }
            for &(a, lca) in &left_active {
                let row_off = a * schur_cols;
                if right_active.len() == k {
                    // Dense right row: at nnz == k the active list is exactly
                    // `col = 0..k` in order, so this contiguous fused loop is
                    // BIT-IDENTICAL to the sparse one below (same b order,
                    // same subtractions) while being vectorizable — the old
                    // codegen was 771 instructions with 18 bounds/compare
                    // patterns around exactly 2 scalar FP ops and zero vector
                    // ops (A10 asm read, 2026-07-11).
                    let schur_row = &mut schur_flat[row_off..row_off + k];
                    for (s, &r) in schur_row.iter_mut().zip(right_row) {
                        *s -= lca * r;
                    }
                } else {
                    for &(b, rcb) in &right_active {
                        schur_flat[row_off + b] -= lca * rcb;
                    }
                }
            }
        }
    }
}
