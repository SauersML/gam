//! Stage 3.3 GPU PIRLS-loop dispatch wire-in.
//!
//! `try_gpu_pirls_loop_dispatch` is the single entry the CPU PIRLS driver
//! ([`crate::pirls::fit_model_for_fixed_rho_with_adaptive_kkt`])
//! calls before falling through to the host LM loop. Returns
//! `Some((PirlsResult, WorkingModelPirlsResult))` when the device-resident
//! loop fully completed and assembled the CPU-oracle-equivalent surface;
//! returns `None` when admission denied dispatch, the workload is shaped
//! in a way the GPU loop does not cover yet (sparse-native, Kronecker,
//! diagonal-penalty, constraints, Firth). Once admitted, device errors retain
//! their typed identity and are never retried on a different implementation.
//!
//! Linux-only — `pirls_loop_on_stream` is gated behind `target_os =
//! "linux"`, and so is the entire wire surface. Non-Linux builds expose a
//! no-op stub.

use gam_gpu::policy::{PirlsLoopAdmission, PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};

/// Result of mapping the engine-level `(ResponseFamily, InverseLink)` pair
/// to the six built-in JIT-cached families the Stage 3.3 PIRLS loop can
/// evaluate without going through a Level-B raw-body NVRTC compile.
///
/// `None` means the fit must stay on the CPU LM loop: either the response /
/// link combination is one of the engine's custom variants (Sas, Mixture,
/// LatentCLogLog, BetaLogistic, Tweedie, NegativeBinomial, Beta,
/// RoystonParmar) for which Stage 3.3 has no built-in row kernel, or the
/// response is supported but the link does not match a built-in pairing
/// (e.g. Poisson with Identity link).
pub fn pirls_loop_family_for(spec: &LikelihoodSpec) -> Option<PirlsLoopFamilyKind> {
    let link = match &spec.link {
        InverseLink::Standard(lf) => *lf,
        // Custom / blended inverse links have no Stage 3.3 row kernel; they
        // require Stage 6 Level B JIT, which the CPU LM loop calls through
        // different machinery.
        _ => return None,
    };
    match (&spec.response, link) {
        (ResponseFamily::Binomial, StandardLink::Logit) => {
            Some(PirlsLoopFamilyKind::BernoulliLogit)
        }
        (ResponseFamily::Binomial, StandardLink::Probit) => {
            Some(PirlsLoopFamilyKind::BernoulliProbit)
        }
        (ResponseFamily::Binomial, StandardLink::CLogLog) => {
            Some(PirlsLoopFamilyKind::BernoulliCLogLog)
        }
        (ResponseFamily::Poisson, StandardLink::Log) => Some(PirlsLoopFamilyKind::PoissonLog),
        (ResponseFamily::Gaussian, StandardLink::Identity) => {
            Some(PirlsLoopFamilyKind::GaussianIdentity)
        }
        (ResponseFamily::Gamma, StandardLink::Log) => Some(PirlsLoopFamilyKind::GammaLog),
        // Every other pairing is either not in the JIT-cache set or is a
        // canonical-pair the row kernels do not currently support.
        _ => None,
    }
}

/// Curvature surface the GPU loop should use given the family mapping and the
/// CPU PIRLS loop's preferred curvature.
pub fn pirls_loop_curvature_for(family: PirlsLoopFamilyKind) -> PirlsLoopCurvatureKind {
    match family {
        PirlsLoopFamilyKind::BernoulliProbit | PirlsLoopFamilyKind::BernoulliCLogLog => {
            PirlsLoopCurvatureKind::Observed
        }
        PirlsLoopFamilyKind::BernoulliLogit
        | PirlsLoopFamilyKind::PoissonLog
        | PirlsLoopFamilyKind::GaussianIdentity
        | PirlsLoopFamilyKind::GammaLog => PirlsLoopCurvatureKind::Fisher,
    }
}

/// Detect whether the CUDA runtime is initialised on this host. The probe
/// underneath returns `None` on every non-Linux target, so the function works
/// unconditionally.
pub fn gpu_runtime_available() -> bool {
    gam_gpu::device_runtime::GpuRuntime::is_available()
}

/// Strict admission shape for the Stage 3.3 PIRLS loop, computed from the
/// `(response, link)` spec and the active design shape `(n, p)`. Returns
/// `None` when the family / link is not in the JIT-cached set so the caller
/// skips both the GPU dispatch and the runtime probe.
pub fn admission_for(spec: &LikelihoodSpec, n: usize, p: usize) -> Option<PirlsLoopAdmission> {
    let family = pirls_loop_family_for(spec)?;
    let curvature = pirls_loop_curvature_for(family);
    Some(PirlsLoopAdmission {
        n,
        p,
        family: Some(family),
        curvature,
        gpu_available: gpu_runtime_available(),
    })
}

#[cfg(target_os = "linux")]
mod linux_impl {
    use ndarray::{Array1, ArrayView1, ArrayView2};

    use crate::active_set::compute_constraint_kkt_diagnostics;
    use crate::gpu::pirls_dispatch_wire::admission_for;
    use crate::gpu::pirls_gpu::{self, cuda};
    use crate::gpu_kernels::pirls_row::{CurvatureMode, PirlsRowFamily};
    use crate::pirls::{
        ExportedLaplaceCurvature, FirthDiagnostics, HessianCurvatureKind, PirlsCoordinateFrame,
        PirlsResult, PirlsStatus, WorkingModelPirlsResult, WorkingState,
        compute_observed_hessian_curvature_arrays, computeworkingweight_derivatives_from_eta,
        pirls_data_log_kernel_from_eta,
    };
    use gam_gpu::cuda_selected;
    use gam_gpu::device_runtime::GpuRuntime;
    use gam_gpu::policy::{PirlsLoopAdmission, PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
    use gam_linalg::matrix::DesignMatrix;
    use gam_linalg::matrix::SymmetricMatrix;
    use gam_problem::LinearInequalityConstraints;
    use gam_problem::{
        Coefficients, EstimationError, GlmLikelihoodSpec, InverseLink, LinearPredictor,
    };
    use gam_terms::construction::ReparamResult;

    /// All inputs needed for the GPU PIRLS loop end-to-end. Built by the
    /// CPU PIRLS driver right before it would invoke `runworking_model_pirls`,
    /// so every field is already in transformed coordinates.
    pub struct GpuPirlsDispatchInput<'a> {
        /// `LikelihoodSpec`-shaped view used by `admission_for`.
        pub likelihood: &'a GlmLikelihoodSpec,
        /// Inverse link the row kernel was driven by.
        pub inverse_link: &'a InverseLink,
        /// Original dense design `X_original` (before reparameterization), shape `n × p`, row-major.
        /// Uploaded once to device-resident shared model cache; Qs is uploaded separately per ρ/σ point.
        pub x_original: ArrayView2<'a, f64>,
        /// Transformed dense penalty `S_λ` in transformed coordinates,
        /// shape `p × p`.
        pub s_transformed: ArrayView2<'a, f64>,
        /// Linear shift `b` of the penalty `βᵀSβ − 2βᵀb + c`, length `p`.
        /// Mirrors `PirlsPenalty::linear_shift()` in the CPU oracle.
        pub linear_shift: ArrayView1<'a, f64>,
        /// Constant shift `c` of the penalty `βᵀSβ − 2βᵀb + c`.
        pub constant_shift: f64,
        /// Response vector `y`, length `n`.
        pub y: ArrayView1<'a, f64>,
        /// Prior weights, length `n`.
        pub priorweights: ArrayView1<'a, f64>,
        /// Observation offset, length `n`. Must equal `n`-sized vector
        /// (zeros when absent).
        pub offset: ArrayView1<'a, f64>,
        /// Initial β guess in *transformed* coordinates.
        pub initial_beta: ArrayView1<'a, f64>,
        /// LM ridge to seed the loop with. Mirrors
        /// `WorkingModelPirlsOptions::initial_lm_lambda` (defaulted to
        /// `1e-6` when `None`).
        pub initial_lm_lambda: Option<f64>,
        /// Outer iteration cap.
        pub max_iterations: usize,
        /// Convergence tolerance (deviance-relative; the loop's stop test
        /// is `|Δdev| < tol · max(1, |dev|)`).
        pub convergence_tolerance: f64,
        /// Linear inequality constraints in transformed coordinates.
        pub linear_constraints: Option<LinearInequalityConstraints>,
        /// Reparameterisation `qs` (transformed → original). Passed to
        /// the loop's postpass to populate `beta_transformed`.
        pub qs: Option<ArrayView2<'a, f64>>,
        /// Full `ReparamResult` for `PirlsResult.reparam_result`.
        pub reparam_result: ReparamResult,
        /// Cached `x_transformed` as `DesignMatrix` (for
        /// `PirlsResult.x_transformed`).
        pub x_transformed_design: DesignMatrix,
        /// Coordinate frame label propagated onto the result.
        pub coordinate_frame: PirlsCoordinateFrame,
        /// EDF computed host-side from the penalty root + Hessian.
        /// `None` makes the assembler use NaN (will be patched by REML
        /// downstream).
        pub edf: Option<f64>,
        /// Whether the outer caller wants the observed-information
        /// curvature exported. Drives the postpass and the
        /// `exported_laplace_curvature` label.
        pub exported_curvature: HessianCurvatureKind,
    }

    fn family_to_row(family: PirlsLoopFamilyKind) -> PirlsRowFamily {
        match family {
            PirlsLoopFamilyKind::BernoulliLogit => PirlsRowFamily::BernoulliLogit,
            PirlsLoopFamilyKind::BernoulliProbit => PirlsRowFamily::BernoulliProbit,
            PirlsLoopFamilyKind::BernoulliCLogLog => PirlsRowFamily::BernoulliCLogLog,
            PirlsLoopFamilyKind::PoissonLog => PirlsRowFamily::PoissonLog,
            PirlsLoopFamilyKind::GaussianIdentity => PirlsRowFamily::GaussianIdentity,
            PirlsLoopFamilyKind::GammaLog => PirlsRowFamily::GammaLog,
        }
    }

    fn curvature_to_row(curvature: PirlsLoopCurvatureKind) -> CurvatureMode {
        match curvature {
            PirlsLoopCurvatureKind::Fisher => CurvatureMode::Fisher,
            PirlsLoopCurvatureKind::Observed => CurvatureMode::Observed,
        }
    }

    fn exported_to_loop(kind: HessianCurvatureKind) -> PirlsLoopCurvatureKind {
        match kind {
            HessianCurvatureKind::Fisher => PirlsLoopCurvatureKind::Fisher,
            HessianCurvatureKind::Observed => PirlsLoopCurvatureKind::Observed,
        }
    }

    /// Cheap pre-materialization admission gate. Returns `true` only when
    /// all of the following hold without touching any O(N·p) work:
    ///
    /// - The global GPU policy selects CUDA (`cuda_selected()`).
    /// - A live `GpuRuntime` is present.
    /// - The (family, curvature) pair is in the JIT-cached set
    ///   (`admission_for` succeeds).
    /// - The runtime policy accepts the (n, p) shape
    ///   (`should_use_gpu_pirls_loop`).
    ///
    /// The caller should test this **before** materializing `X·Qs` or any
    /// other transformed design so that CPU-default / no-runtime /
    /// policy-rejected paths pay zero `fast_ab` cost.
    pub fn try_gpu_pirls_loop_admit(
        likelihood: &gam_problem::GlmLikelihoodSpec,
        n: usize,
        p: usize,
    ) -> bool {
        if !cuda_selected() {
            return false;
        }
        let Some(admission) = admission_for(&likelihood.spec, n, p) else {
            return false;
        };
        let Some(runtime) = GpuRuntime::global() else {
            return false;
        };
        runtime.policy().should_use_gpu_pirls_loop(admission)
    }

    /// Attempt to run the Stage 3.3 device-resident PIRLS loop for the
    /// dispatch input. Returns `Some` only when the loop ran end-to-end
    /// and the full CPU-oracle surface was assembled.
    pub fn try_gpu_pirls_loop_dispatch(
        input: GpuPirlsDispatchInput<'_>,
    ) -> Option<Result<(PirlsResult, WorkingModelPirlsResult), EstimationError>> {
        // Honor the documented GPU policy: never route to the GPU loop when
        // the caller has explicitly selected CPU execution.
        if !cuda_selected() {
            return None;
        }
        // Gaussian-identity fits have an exact GPU PLS path (issue #272) and
        // must NOT be routed through the row-kernel PIRLS loop on device.
        // The exact path (try_gpu_gaussian_pls_dispatch) fires before this
        // dispatch site in fit_model_for_fixed_rho_with_adaptive_kkt.
        // This gate ensures no future code path accidentally re-routes them
        // here.  Tests that explicitly exercise the row kernel may bypass
        // this gate by calling pirls_loop_on_stream directly.
        if input.likelihood.spec.is_gaussian_identity() {
            return None;
        }
        let n = input.x_original.nrows();
        let p = input.x_original.ncols();
        // Engine-level admission: shape + family + curvature + runtime probe.
        let admission = admission_for(&input.likelihood.spec, n, p)?;
        let runtime = GpuRuntime::global()?;
        if !runtime.policy().should_use_gpu_pirls_loop(admission) {
            return None;
        }
        let family = family_to_row(admission.family?);
        let curvature = curvature_to_row(admission.curvature);

        Some(run_gpu_pirls_loop(
            input, admission, family, curvature, n, p,
        ))
    }

    fn run_gpu_pirls_loop(
        input: GpuPirlsDispatchInput<'_>,
        admission: PirlsLoopAdmission,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
        n: usize,
        p: usize,
    ) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
        assert_eq!(admission.n, n);
        assert_eq!(admission.p, p);
        // --- Device upload + workspace allocation -----------------------
        // Upload X, y, prior_w, and offset (#258) once; shared by all iters.
        let shared = pirls_gpu::upload_shared_pirls_gpu(
            input.x_original,
            input.y,
            input.priorweights,
            input.offset,
        )
        .map_err(EstimationError::InvalidInput)?;
        // Upload Qs for this ρ/σ point. Identity when no reparameterization.
        let mut ws = pirls_gpu::allocate_sigma_pirls_workspace(&shared)
            .map_err(EstimationError::InvalidInput)?;
        if let Some(qs) = input.qs {
            pirls_gpu::upload_qs_pirls(&mut ws, qs).map_err(EstimationError::InvalidInput)?;
        } else {
            pirls_gpu::upload_qs_identity_pirls(&mut ws).map_err(EstimationError::InvalidInput)?;
        }
        let mut loop_ws = pirls_gpu::allocate_pirls_loop_workspace(&shared, &ws)
            .map_err(EstimationError::InvalidInput)?;

        let lm_ridge = input.initial_lm_lambda.unwrap_or(1e-6);
        let likelihood_scale = match family {
            PirlsRowFamily::GammaLog => pirls_gpu::PirlsLoopLikelihoodScale::gamma_shape(
                input
                    .likelihood
                    .resolved_gamma_shape()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
            )
            .map_err(EstimationError::InvalidInput)?,
            _ => {
                // Validate family and metadata even though these kernels have
                // no scalar likelihood parameter in their ABI contract.
                input
                    .likelihood
                    .resolved_scale()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                pirls_gpu::PirlsLoopLikelihoodScale::non_gamma()
            }
        };
        let firth_default = FirthDiagnostics::Inactive;
        // Sanity-check that the host-side enum maps round-trip; if a future
        // change to PirlsLoopCurvatureKind / HessianCurvatureKind drops a
        // case this assertion will catch the gap at the dispatch boundary.
        assert!(matches!(
            exported_to_loop(input.exported_curvature),
            PirlsLoopCurvatureKind::Fisher | PirlsLoopCurvatureKind::Observed
        ));

        let extra = cuda::PirlsLoopExtra {
            likelihood: input.likelihood,
            inverse_link: input.inverse_link,
            y: input.y,
            priorweights: input.priorweights,
            offset: input.offset,
            linear_constraints: input.linear_constraints.as_ref(),
            exported_curvature: input.exported_curvature,
            ridge_passport: None,
            firth: Some(firth_default.clone()),
            edf: input.edf,
        };
        // step_lm_lambda = lm_ridge (temporary Newton stabilization only).
        // objective_ridge = 0.0: the model's ridge is already baked into
        // s_transformed by the outer REML loop; no separate identity ridge
        // enters the exported Hessian / EDF / RidgePassport here.
        let outcome = pirls_gpu::pirls_loop_on_stream(
            &shared,
            &mut ws,
            &mut loop_ws,
            family,
            curvature,
            likelihood_scale,
            input.initial_beta,
            input.s_transformed,
            input.linear_shift,
            input.constant_shift,
            lm_ridge,
            0.0,
            input.max_iterations,
            input.convergence_tolerance,
            Some(&extra),
        )
        .map_err(|error| match error {
            cuda::PirlsGpuLoopError::Geometry(error) => error,
            cuda::PirlsGpuLoopError::Runtime(message) => {
                EstimationError::RemlOptimizationFailed(format!("GPU PIRLS runtime: {message}"))
            }
        })?;

        // --- Assemble PirlsResult + WorkingModelPirlsResult ------------
        let cuda::PirlsLoopOutcome {
            beta,
            penalized_hessian,
            logdet,
            deviance,
            iterations,
            converged,
            final_eta,
            final_mu,
            final_grad_eta,
            final_w_hessian,
            final_w_solver,
            final_offset,
            beta_transformed,
            finalweights,
            solveweights,
            solve_dmu_deta,
            solve_d2mu_deta2,
            solve_d3mu_deta3,
            solve_c_array,
            solve_d_array,
            derivatives_unsupported,
            status,
            ridge_passport,
            firth,
            constraint_kkt,
            edf,
            last_deviance_change,
            last_step_halving,
            last_step_size,
            final_lm_lambda,
            min_deviance,
            max_abs_eta,
        } = outcome;

        // `logdet` corresponds to log|H_penalized| at the converged β; it is
        // not on the CPU oracle's `PirlsResult` surface (REML recomputes it
        // from the assembled Hessian), but cross-checking finiteness here
        // catches a non-PD final factorisation before downstream code
        // touches the Hessian.
        if !logdet.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "GPU PIRLS loop returned non-finite log|H| = {logdet}"
            )));
        }
        // `converged` already feeds `status` (Converged / Unstable /
        // MaxIterationsReached); make the relationship explicit so a future
        // refactor that breaks the invariant trips here rather than silently
        // mis-stamping `PirlsResult.status`.
        assert_eq!(
            converged,
            matches!(
                status,
                PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
            ),
            "GPU outcome converged flag inconsistent with status",
        );

        // working response z_i = eta_i + (y - mu) / dmu/deta (0 on zero deriv).
        let finalz = {
            let mut z = final_eta.clone();
            for i in 0..n {
                let d = solve_dmu_deta.get(i).copied().unwrap_or(0.0);
                let resid = input.y[i] - final_mu[i];
                if d.is_finite() && d.abs() > 0.0 {
                    z[i] += resid / d;
                }
            }
            z
        };

        // If the outcome lacks derivative arrays (extra was None on a
        // mis-wired call), recompute host-side so PirlsResult is whole.
        let (final_dmu_deta, final_d2mu_deta2, final_d3mu_deta3, final_c, final_d) =
            if derivatives_unsupported
                || solve_dmu_deta.is_empty()
                || solve_d2mu_deta2.is_empty()
                || solve_d3mu_deta3.is_empty()
            {
                let (sc, sd, sdmu, sd2, sd3) = computeworkingweight_derivatives_from_eta(
                    input.likelihood,
                    input.inverse_link,
                    &final_eta,
                    input.priorweights,
                )?;
                (sdmu, sd2, sd3, sc, sd)
            } else {
                (
                    solve_dmu_deta.clone(),
                    solve_d2mu_deta2.clone(),
                    solve_d3mu_deta3.clone(),
                    solve_c_array.clone(),
                    solve_d_array.clone(),
                )
            };

        // Observed-curvature finalisation if the outer caller requested
        // it and the GPU loop did not already promote (i.e. ran Fisher).
        let (finalweights_arr, final_c_arr, final_d_arr) =
            if matches!(input.exported_curvature, HessianCurvatureKind::Observed)
                && curvature == CurvatureMode::Fisher
            {
                compute_observed_hessian_curvature_arrays(
                    input.likelihood,
                    input.inverse_link,
                    &final_eta,
                    input.y,
                    &final_w_solver,
                    input.priorweights,
                )?
            } else {
                (finalweights.clone(), final_c.clone(), final_d.clone())
            };
        // Echo through whichever finalweights array we ended with for use below.
        let finalweights_for_state = if finalweights_arr.is_empty() {
            final_w_hessian.clone()
        } else {
            finalweights_arr.clone()
        };

        // Stabilised Hessian = penalized_hessian + δI per ridge_passport.
        let delta = ridge_passport.delta();
        let mut stab = penalized_hessian.clone();
        if delta > 0.0 {
            for i in 0..p {
                stab[[i, i]] += delta;
            }
        }
        let penalized_hessian_sym = SymmetricMatrix::Dense(penalized_hessian.clone());
        let stabilizedhessian_sym = SymmetricMatrix::Dense(stab);

        // max_abs_eta — recompute from the actual eta if outcome's was zero
        // (older GPU outcomes pre-dating the field surface stamp 0.0).
        let max_abs_eta_used = if max_abs_eta > 0.0 {
            max_abs_eta
        } else {
            final_eta.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
        };

        // Gradient in transformed coordinates: Qsᵀ (X_originalᵀ · score_eta).
        // X_originalᵀ · score_eta is p-vector; then project through Qsᵀ.
        let xt_grad_eta = {
            let xo = input.x_original;
            let mut xo_score = Array1::<f64>::zeros(p);
            for j in 0..p {
                let mut acc = 0.0_f64;
                for i in 0..n {
                    acc += xo[[i, j]] * final_grad_eta[i];
                }
                xo_score[j] = acc;
            }
            // Project through Qsᵀ: xt_grad_eta = Qsᵀ · xo_score.
            if let Some(qs) = input.qs {
                qs.t().dot(&xo_score)
            } else {
                xo_score
            }
        };
        let s_beta = {
            let mut acc = Array1::<f64>::zeros(p);
            for i in 0..p {
                let mut s = 0.0_f64;
                for j in 0..p {
                    s += input.s_transformed[[i, j]] * beta[j];
                }
                acc[i] = s;
            }
            acc
        };
        // gradient = S·β − linear_shift − Xᵀ·score_eta
        let mut gradient_total = s_beta.clone();
        gradient_total -= &input.linear_shift;
        gradient_total -= &xt_grad_eta;
        let lastgradient_norm = gradient_total.dot(&gradient_total).sqrt();
        let score_norm = xt_grad_eta.dot(&xt_grad_eta).sqrt();
        let s_beta_norm = s_beta.dot(&s_beta).sqrt();
        let ridge_grad_norm = if delta > 0.0 {
            delta * beta.dot(&beta).sqrt()
        } else {
            0.0
        };
        let gradient_natural_scale = score_norm + s_beta_norm + ridge_grad_norm;

        // Penalty term = βᵀSβ + δ‖β‖².
        let penalty_term = beta.dot(&s_beta) + delta * beta.dot(&beta);
        let min_penalized_deviance = {
            let cand = min_deviance + penalty_term;
            if cand.is_finite() {
                cand
            } else {
                f64::INFINITY
            }
        };

        let coefficients = Coefficients::new(beta.clone());
        let beta_transformed_coef = Coefficients::new(beta_transformed.clone());

        let constraint_kkt_final = if constraint_kkt.is_some() {
            constraint_kkt.clone()
        } else if let Some(lin) = input.linear_constraints.as_ref() {
            Some(compute_constraint_kkt_diagnostics(
                &beta,
                &gradient_total,
                lin,
            ))
        } else {
            None
        };

        let exported_label = match (input.exported_curvature, derivatives_unsupported) {
            (HessianCurvatureKind::Observed, false) => ExportedLaplaceCurvature::ObservedExact,
            _ => ExportedLaplaceCurvature::ExpectedInformationSurrogate,
        };

        let working_state = WorkingState {
            eta: LinearPredictor::new(final_eta.clone()),
            gradient: gradient_total.clone(),
            hessian: penalized_hessian_sym.clone(),
            log_likelihood: pirls_data_log_kernel_from_eta(
                input.y,
                &final_eta,
                input.likelihood,
                input.inverse_link,
                input.priorweights,
                deviance,
            )?,
            deviance,
            penalty_term,
            firth: firth.clone(),
            ridge_used: delta,
            hessian_curvature: match curvature {
                CurvatureMode::Fisher => HessianCurvatureKind::Fisher,
                CurvatureMode::Observed => HessianCurvatureKind::Observed,
            },
            gradient_natural_scale,
        };

        let working_summary = WorkingModelPirlsResult {
            beta: coefficients.clone(),
            state: working_state,
            status,
            iterations,
            lastgradient_norm,
            last_deviance_change,
            last_step_size,
            last_step_halving,
            max_abs_eta: max_abs_eta_used,
            constraint_kkt: constraint_kkt_final.clone(),
            final_lm_lambda,
            final_accept_rho: None,
            min_penalized_deviance,
            exported_laplace_curvature: exported_label.clone(),
        };

        let edf_final = if edf.is_finite() { edf } else { f64::NAN };

        // final_offset is `n` zeros when the loop did not echo offset
        // through. Use the caller-supplied offset in that case.
        let final_offset_arr = if final_offset.len() == n {
            final_offset
        } else {
            input.offset.to_owned()
        };

        let pirls_result = PirlsResult {
            likelihood: input.likelihood.clone(),
            beta_transformed: beta_transformed_coef,
            penalized_hessian_transformed: penalized_hessian_sym,
            stabilizedhessian_transformed: stabilizedhessian_sym,
            ridge_passport,
            deviance,
            edf: edf_final,
            stable_penalty_term: penalty_term,
            firth,
            // #1868: length-n row fields are shared `ArcArray1`; `.into_shared()`
            // moves the owned GPU-returned buffers into an `Arc` (O(1)).
            finalweights: finalweights_for_state.into_shared(),
            final_offset: final_offset_arr.into_shared(),
            final_eta: final_eta.clone().into_shared(),
            finalmu: final_mu.clone().into_shared(),
            solveweights: if solveweights.is_empty() {
                final_w_solver.clone()
            } else {
                solveweights.clone()
            }
            .into_shared(),
            solveworking_response: finalz.into_shared(),
            solvemu: final_mu.clone().into_shared(),
            solve_dmu_deta: final_dmu_deta.into_shared(),
            solve_d2mu_deta2: final_d2mu_deta2.into_shared(),
            solve_d3mu_deta3: final_d3mu_deta3.into_shared(),
            solve_c_array: final_c_arr.into_shared(),
            solve_d_array: final_d_arr.into_shared(),
            derivatives_unsupported,
            status,
            iteration: iterations,
            max_abs_eta: max_abs_eta_used,
            lastgradient_norm,
            gradient_natural_scale,
            penalized_gradient_transformed: working_summary.state.gradient.clone(),
            last_deviance_change,
            last_step_halving,
            hessian_curvature: match curvature {
                CurvatureMode::Fisher => HessianCurvatureKind::Fisher,
                CurvatureMode::Observed => HessianCurvatureKind::Observed,
            },
            exported_laplace_curvature: exported_label,
            final_lm_lambda,
            final_accept_rho: None,
            constraint_kkt: constraint_kkt_final,
            linear_constraints_transformed: input.linear_constraints,
            reparam_result: input.reparam_result,
            x_transformed: input.x_transformed_design,
            coordinate_frame: input.coordinate_frame,
            used_device: true,
            cache_compacted: false,
            min_penalized_deviance,
        };

        // Hessian-side weights are kept on the working_summary surface for
        // outer LM consumers; if the loop did not stamp a separate
        // `finalweights`, fall back to `final_w_hessian` so REML's
        // `H = XᵀW_HX + S_λ` reconstruction has the curvature it expects.
        assert_eq!(final_w_hessian.len(), n);
        assert_eq!(final_grad_eta.len(), n);

        Ok((pirls_result, working_summary))
    }

    /// All inputs needed for the GPU Gaussian-identity exact PLS dispatch.
    /// Built by the CPU PIRLS driver immediately before the CPU
    /// `solve_penalized_least_squares_implicit` fast-path, so the GPU path
    /// fires first when available.
    pub struct GpuGaussianPlsInput<'a> {
        /// Precomputed `XᵀWX` in original (pre-Qs) coordinates, p×p.
        pub xtwx_orig: ArrayView2<'a, f64>,
        /// Precomputed `XᵀW(y − offset)` in original coordinates, length p.
        pub xtwy_orig: ArrayView1<'a, f64>,
        /// Penalty `Σλₖ Sₖ` in transformed (post-Qs) coordinates, p×p.
        pub s_transformed: ArrayView2<'a, f64>,
        /// Additive RHS correction in transformed coordinates, length p.
        pub linear_shift: ArrayView1<'a, f64>,
        /// Prior-mean vector for Tikhonov RHS, length p.
        pub prior_mean_target: ArrayView1<'a, f64>,
        /// Constant term of the shifted penalty quadratic (for penalty_term).
        pub constant_shift: f64,
        /// Reparameterisation matrix Qs (p×p).  `None` = identity transform.
        pub qs: Option<ArrayView2<'a, f64>>,
        /// Stabilisation ridge δ.
        pub ridge: f64,
        /// GLM likelihood spec.
        pub likelihood: &'a gam_problem::GlmLikelihoodSpec,
        /// Inverse link.
        pub inverse_link: &'a gam_problem::InverseLink,
        /// Original design X for computing η = offset + X·Qs·β.
        pub x_original: &'a DesignMatrix,
        /// Response y, length n.
        pub y: ArrayView1<'a, f64>,
        /// Prior weights, length n.
        pub priorweights: ArrayView1<'a, f64>,
        /// Observation offset, length n.
        pub offset: ArrayView1<'a, f64>,
        /// Full reparameterisation result for `PirlsResult::reparam_result`.
        pub reparam_result: ReparamResult,
        /// Design matrix for `PirlsResult::x_transformed`.
        pub x_transformed_design: DesignMatrix,
        /// Coordinate frame for `PirlsResult::coordinate_frame`.
        pub coordinate_frame: PirlsCoordinateFrame,
        /// Linear constraints (None when cache_eligible was true).
        pub linear_constraints: Option<LinearInequalityConstraints>,
    }

    /// Cheap admission gate for the GPU Gaussian-identity exact PLS path.
    /// Returns `true` iff cuda_selected(), runtime available, and the likelihood
    /// is Gaussian-identity.
    pub fn try_gpu_gaussian_pls_admit(likelihood: &gam_problem::GlmLikelihoodSpec) -> bool {
        if !gam_gpu::cuda_selected() {
            return false;
        }
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            return false;
        }
        likelihood.spec.is_gaussian_identity()
    }

    /// Attempt to run the exact GPU PLS for Gaussian-identity.
    ///
    /// Returns `Some(Ok(...))` when the device solve completed and the full
    /// CPU-oracle surface was assembled; returns `None` when admission was
    /// denied; returns `Some(Err(...))` on admitted-device failure, which is
    /// propagated without a CPU retry.
    pub fn try_gpu_gaussian_pls_dispatch(
        input: GpuGaussianPlsInput<'_>,
    ) -> Option<Result<(PirlsResult, WorkingModelPirlsResult), String>> {
        if !try_gpu_gaussian_pls_admit(input.likelihood) {
            return None;
        }
        Some(run_gpu_gaussian_pls(input))
    }

    fn run_gpu_gaussian_pls(
        input: GpuGaussianPlsInput<'_>,
    ) -> Result<(PirlsResult, WorkingModelPirlsResult), String> {
        use crate::pirls::{
            array1_l2_norm, calculate_deviance_from_eta,
            calculate_loglikelihood_omitting_constants_from_eta,
            computeworkingweight_derivatives_from_eta,
        };
        use gam_linalg::matrix::LinearOperator;
        use gam_linalg::utils::inf_norm;
        use gam_problem::{RidgePassport, RidgePolicy};
        use ndarray::Array1;

        let pls = pirls_gpu::solve_gaussian_pls_gpu(
            input.xtwx_orig,
            input.xtwy_orig,
            input.s_transformed,
            input.linear_shift,
            input.prior_mean_target,
            input.ridge,
            input.qs,
        )?;

        if !pls.logdet.is_finite() {
            return Err(format!(
                "GPU Gaussian PLS returned non-finite log|H| = {}",
                pls.logdet
            ));
        }

        let beta = pls.beta.clone();
        let penalized_hessian = pls.penalized_hessian;
        let p = beta.len();
        // eta = offset + X · (Qs · beta), or offset + X · beta if no Qs.
        let qbeta: Array1<f64> = if let Some(qs_v) = input.qs {
            qs_v.dot(&beta)
        } else {
            beta.clone()
        };
        let mut eta = input.offset.to_owned();
        eta += &input.x_original.apply(&qbeta);
        let finalmu = eta.clone();
        let finalz = input.y.to_owned();

        // gradient_data = QsᵀXᵀ W (mu - y) in transformed coordinates.
        let mut weighted_residual = finalmu.clone();
        weighted_residual -= &finalz;
        weighted_residual *= &input.priorweights;
        // Xᵀ W r (in original coords) via DesignMatrix::transpose_vector_multiply.
        let xt_wr_orig = input
            .x_original
            .transpose_vector_multiply(&weighted_residual);
        // Rotate to transformed coords: QsᵀXᵀWr.
        let gradient_data: Array1<f64> = if let Some(qs_v) = input.qs {
            qs_v.t().dot(&xt_wr_orig)
        } else {
            xt_wr_orig
        };
        let score_norm = array1_l2_norm(&gradient_data);

        // s_beta = S·β − linear_shift.
        let mut s_beta: Array1<f64> = Array1::zeros(p);
        for i in 0..p {
            let mut acc = 0.0_f64;
            for j in 0..p {
                acc += input.s_transformed[[i, j]] * beta[j];
            }
            s_beta[i] = acc - input.linear_shift[i];
        }
        let s_beta_norm = array1_l2_norm(&s_beta);

        let mut gradient = gradient_data.clone();
        gradient += &s_beta;

        // penalty_term = betaᵀ·S·β − 2·betaᵀ·linear_shift + constant_shift.
        let mut penalty_term: f64 = input.constant_shift;
        for i in 0..p {
            let mut s_row_b = 0.0_f64;
            for j in 0..p {
                s_row_b += input.s_transformed[[i, j]] * beta[j];
            }
            penalty_term += beta[i] * s_row_b;
            penalty_term -= 2.0 * beta[i] * input.linear_shift[i];
        }

        let ridge_used = input.ridge;
        let mut ridge_grad_norm = 0.0_f64;
        if ridge_used > 0.0 {
            let beta_sq: f64 = beta.dot(&beta);
            penalty_term += ridge_used * beta_sq;
            let ridge_contrib = beta.mapv(|v| ridge_used * v);
            gradient += &ridge_contrib;
            ridge_grad_norm = ridge_used * array1_l2_norm(&beta);
        }

        let gradient_norm = array1_l2_norm(&gradient);
        let max_abs_eta = inf_norm(finalmu.iter().copied());

        let deviance = calculate_deviance_from_eta(
            input.y,
            &eta,
            input.likelihood,
            input.inverse_link,
            input.priorweights,
        )
        .map_err(|error| format!("GPU Gaussian deviance evaluation failed: {error}"))?;
        let log_likelihood = pirls_data_log_kernel_from_eta(
            input.y,
            &eta,
            input.likelihood,
            input.inverse_link,
            input.priorweights,
            deviance,
        )
        .map_err(|error| format!("GPU Gaussian P-IRLS data-kernel evaluation failed: {error}"))?;

        // Stabilised Hessian = penalized_hessian + ridge_used·I.
        let mut stab = penalized_hessian.clone();
        if ridge_used > 0.0 {
            for i in 0..p {
                stab[[i, i]] += ridge_used;
            }
        }
        let penalized_hessian_sym = SymmetricMatrix::Dense(penalized_hessian.clone());
        let stabilizedhessian_sym = SymmetricMatrix::Dense(stab);

        let priorweights_owned = input.priorweights.to_owned();
        let beta_coef = Coefficients::new(beta.clone());

        let zero_iter_penalized = deviance + penalty_term;

        let working_state = WorkingState {
            eta: LinearPredictor::new(finalmu.clone()),
            gradient: gradient.clone(),
            hessian: penalized_hessian_sym.clone(),
            log_likelihood,
            deviance,
            penalty_term,
            firth: FirthDiagnostics::Inactive,
            ridge_used,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: score_norm + s_beta_norm + ridge_grad_norm,
        };

        let constraint_kkt_val = if let Some(lin) = input.linear_constraints.as_ref() {
            Some(compute_constraint_kkt_diagnostics(&beta, &gradient, lin))
        } else {
            None
        };

        let working_summary = WorkingModelPirlsResult {
            beta: beta_coef.clone(),
            state: working_state,
            status: PirlsStatus::Converged,
            iterations: 1,
            lastgradient_norm: gradient_norm,
            last_deviance_change: 0.0,
            last_step_size: 1.0,
            last_step_halving: 0,
            max_abs_eta,
            constraint_kkt: constraint_kkt_val.clone(),
            min_penalized_deviance: if zero_iter_penalized.is_finite() {
                zero_iter_penalized
            } else {
                f64::INFINITY
            },
            final_lm_lambda: 1e-6,
            final_accept_rho: None,
            exported_laplace_curvature: ExportedLaplaceCurvature::ExpectedInformationSurrogate,
        };

        let (solve_c_array, solve_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                input.likelihood,
                input.inverse_link,
                &eta,
                input.priorweights,
            )
            .map_err(|e| format!("derivative computation failed: {e:?}"))?;

        let pirls_result = PirlsResult {
            likelihood: input.likelihood.clone(),
            beta_transformed: beta_coef.clone(),
            penalized_hessian_transformed: penalized_hessian_sym,
            stabilizedhessian_transformed: stabilizedhessian_sym,
            ridge_passport: RidgePassport::scaled_identity(
                ridge_used,
                RidgePolicy::exact_full_objective(),
            )
            .map_err(|error| format!("invalid GPU PIRLS ridge metadata: {error}"))?,
            deviance,
            edf: f64::NAN, // recomputed by outer REML from penalized_hessian + e_transformed
            stable_penalty_term: penalty_term,
            firth: FirthDiagnostics::Inactive,
            // #1868: length-n row fields are shared `ArcArray1` (O(1) move).
            finalweights: priorweights_owned.clone().into_shared(),
            final_offset: input.offset.to_owned().into_shared(),
            final_eta: eta.clone().into_shared(),
            finalmu: finalmu.clone().into_shared(),
            solveweights: priorweights_owned.into_shared(),
            solveworking_response: finalz.clone().into_shared(),
            solvemu: finalmu.clone().into_shared(),
            solve_dmu_deta: solve_dmu_deta.into_shared(),
            solve_d2mu_deta2: solve_d2mu_deta2.into_shared(),
            solve_d3mu_deta3: solve_d3mu_deta3.into_shared(),
            solve_c_array: solve_c_array.into_shared(),
            solve_d_array: solve_d_array.into_shared(),
            derivatives_unsupported: false,
            status: PirlsStatus::Converged,
            iteration: 1,
            max_abs_eta,
            lastgradient_norm: gradient_norm,
            gradient_natural_scale: score_norm + s_beta_norm + ridge_grad_norm,
            penalized_gradient_transformed: working_summary.state.gradient.clone(),
            last_deviance_change: 0.0,
            last_step_halving: 0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            exported_laplace_curvature: ExportedLaplaceCurvature::ExpectedInformationSurrogate,
            final_lm_lambda: 1e-6,
            final_accept_rho: None,
            constraint_kkt: constraint_kkt_val,
            linear_constraints_transformed: input.linear_constraints,
            reparam_result: input.reparam_result,
            x_transformed: input.x_transformed_design,
            coordinate_frame: input.coordinate_frame,
            used_device: true,
            cache_compacted: false,
            min_penalized_deviance: working_summary.min_penalized_deviance,
        };

        Ok((pirls_result, working_summary))
    }
}

#[cfg(target_os = "linux")]
pub use linux_impl::{
    GpuGaussianPlsInput, GpuPirlsDispatchInput, try_gpu_gaussian_pls_dispatch,
    try_gpu_pirls_loop_admit, try_gpu_pirls_loop_dispatch,
};

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::{LikelihoodSpec, MixtureLinkState};
    use ndarray::Array1;

    fn dummy_mixture_state() -> MixtureLinkState {
        // K=2 components with the free logit at 0; softmax weights are uniform.
        MixtureLinkState {
            components: vec![
                gam_problem::LinkComponent::Logit,
                gam_problem::LinkComponent::Probit,
            ],
            rho: Array1::from(vec![0.0_f64]),
            pi: Array1::from(vec![0.5_f64, 0.5_f64]),
        }
    }

    #[test]
    fn maps_six_canonical_built_in_pairings() {
        for (spec, want) in [
            (
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Logit),
                ),
                PirlsLoopFamilyKind::BernoulliLogit,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Probit),
                ),
                PirlsLoopFamilyKind::BernoulliProbit,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::CLogLog),
                ),
                PirlsLoopFamilyKind::BernoulliCLogLog,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Poisson,
                    InverseLink::Standard(StandardLink::Log),
                ),
                PirlsLoopFamilyKind::PoissonLog,
            ),
            (
                LikelihoodSpec::gaussian_identity(),
                PirlsLoopFamilyKind::GaussianIdentity,
            ),
            (
                LikelihoodSpec::new(
                    ResponseFamily::Gamma,
                    InverseLink::Standard(StandardLink::Log),
                ),
                PirlsLoopFamilyKind::GammaLog,
            ),
        ] {
            assert_eq!(pirls_loop_family_for(&spec), Some(want), "for {:?}", spec);
        }
    }

    #[test]
    fn declines_unsupported_response_link_pairings() {
        let mixture_state = dummy_mixture_state();
        assert_eq!(
            pirls_loop_family_for(&LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Mixture(mixture_state),
            )),
            None
        );
        assert_eq!(
            pirls_loop_family_for(&LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Identity),
            )),
            None
        );
        assert_eq!(
            pirls_loop_family_for(&LikelihoodSpec::new(
                ResponseFamily::Tweedie { p: 1.5 },
                InverseLink::Standard(StandardLink::Log),
            )),
            None
        );
    }

    #[test]
    fn non_canonical_bernoulli_links_request_observed_curvature() {
        assert_eq!(
            pirls_loop_curvature_for(PirlsLoopFamilyKind::BernoulliProbit),
            PirlsLoopCurvatureKind::Observed
        );
        assert_eq!(
            pirls_loop_curvature_for(PirlsLoopFamilyKind::BernoulliCLogLog),
            PirlsLoopCurvatureKind::Observed
        );
        assert_eq!(
            pirls_loop_curvature_for(PirlsLoopFamilyKind::BernoulliLogit),
            PirlsLoopCurvatureKind::Fisher
        );
    }

    #[test]
    fn admission_is_none_for_unmapped_family() {
        let mixture_state = dummy_mixture_state();
        let spec = LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Mixture(mixture_state),
        );
        assert!(admission_for(&spec, 80_000, 44).is_none());
    }
}
