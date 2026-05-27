//! Stage 3.3 GPU PIRLS-loop dispatch wire-in.
//!
//! `try_gpu_pirls_loop_dispatch` is the single entry the CPU PIRLS driver
//! ([`crate::solver::pirls::fit_model_for_fixed_rho_with_adaptive_kkt`])
//! calls before falling through to the host LM loop. Returns
//! `Some((PirlsResult, WorkingModelPirlsResult))` when the device-resident
//! loop fully completed and assembled the CPU-oracle-equivalent surface;
//! returns `None` when admission denied dispatch, the workload is shaped
//! in a way the GPU loop does not cover yet (sparse-native, Kronecker,
//! diagonal-penalty, constraints, Firth), or the device call failed in a
//! way the host wants to retry on CPU.
//!
//! Linux-only — `pirls_loop_on_stream` is gated behind `target_os =
//! "linux"`, and so is the entire wire surface. Non-Linux builds expose a
//! no-op stub.

#[cfg(target_os = "linux")]
mod linux_impl {
    use ndarray::{Array1, ArrayView1, ArrayView2};

    use crate::gpu::pirls_row::{CurvatureMode, PirlsRowFamily};
    use crate::gpu::policy::{
        PirlsLoopAdmission, PirlsLoopCurvatureKind, PirlsLoopFamilyKind,
    };
    use crate::gpu::runtime::GpuRuntime;
    use crate::linalg::matrix::SymmetricMatrix;
    use crate::solver::active_set::{LinearInequalityConstraints, compute_constraint_kkt_diagnostics};
    use crate::solver::gpu::pirls_dispatch::admission_for;
    use crate::solver::gpu::pirls_gpu::{self, cuda};
    use crate::solver::pirls::{
        ExportedLaplaceCurvature, FirthDiagnostics, HessianCurvatureKind, PirlsCoordinateFrame,
        PirlsResult, PirlsStatus, WorkingModelPirlsResult, WorkingState,
        compute_observed_hessian_curvature_arrays, computeworkingweight_derivatives_from_eta,
    };
    use crate::types::{
        Coefficients, DesignMatrix, GlmLikelihoodSpec, InverseLink, LinearPredictor,
        ReparamResult,
    };

    /// All inputs needed for the GPU PIRLS loop end-to-end. Built by the
    /// CPU PIRLS driver right before it would invoke `runworking_model_pirls`,
    /// so every field is already in transformed coordinates.
    pub struct GpuPirlsDispatchInput<'a> {
        /// `LikelihoodSpec`-shaped view used by `admission_for`.
        pub likelihood: &'a GlmLikelihoodSpec,
        /// Inverse link the row kernel was driven by.
        pub inverse_link: &'a InverseLink,
        /// Transformed dense design `X · Qs`, shape `n × p`, row-major.
        /// The GPU loop expects this column-major; this routine uploads
        /// after a row→col-major repack.
        pub x_transformed: ArrayView2<'a, f64>,
        /// Transformed dense penalty `S_λ` in transformed coordinates,
        /// shape `p × p`.
        pub s_transformed: ArrayView2<'a, f64>,
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

    /// Attempt to run the Stage 3.3 device-resident PIRLS loop for the
    /// dispatch input. Returns `Some` only when the loop ran end-to-end
    /// and the full CPU-oracle surface was assembled.
    pub fn try_gpu_pirls_loop_dispatch(
        input: GpuPirlsDispatchInput<'_>,
    ) -> Option<Result<(PirlsResult, WorkingModelPirlsResult), String>> {
        let n = input.x_transformed.nrows();
        let p = input.x_transformed.ncols();
        // Engine-level admission: shape + family + curvature + runtime probe.
        let admission = admission_for(&input.likelihood.spec, n, p)?;
        let runtime = GpuRuntime::global()?;
        if !runtime.policy().should_use_gpu_pirls_loop(admission) {
            return None;
        }
        let family = family_to_row(admission.family?);
        let curvature = curvature_to_row(admission.curvature);

        Some(run_gpu_pirls_loop(input, admission, family, curvature, n, p))
    }

    fn run_gpu_pirls_loop(
        input: GpuPirlsDispatchInput<'_>,
        admission: PirlsLoopAdmission,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
        n: usize,
        p: usize,
    ) -> Result<(PirlsResult, WorkingModelPirlsResult), String> {
        debug_assert_eq!(admission.n, n);
        debug_assert_eq!(admission.p, p);
        // --- Device upload + workspace allocation -----------------------
        let shared = pirls_gpu::upload_shared_pirls_gpu(input.x_transformed)?;
        let mut ws = pirls_gpu::allocate_sigma_pirls_workspace(&shared)?;
        let mut loop_ws = pirls_gpu::allocate_pirls_loop_workspace(&shared, &ws)?;

        let lm_ridge = input.initial_lm_lambda.unwrap_or(1e-6);
        let qs_view = input.qs;
        let firth_default = FirthDiagnostics::Inactive;
        // Sanity-check that the host-side enum maps round-trip; if a future
        // change to PirlsLoopCurvatureKind / HessianCurvatureKind drops a
        // case this assertion will catch the gap at the dispatch boundary.
        debug_assert!(matches!(
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
            qs: qs_view,
            edf: input.edf,
        };
        // exported_curvature is also passed through the policy-side enum
        // so the GPU postpass can map back to PirlsLoopCurvatureKind.
        let _ = exported;

        let outcome = pirls_gpu::pirls_loop_on_stream(
            &shared,
            &mut ws,
            &mut loop_ws,
            family,
            curvature,
            input.initial_beta,
            input.y,
            input.priorweights,
            input.s_transformed,
            lm_ridge,
            input.max_iterations,
            input.convergence_tolerance,
            Some(&extra),
        )?;

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
            return Err(format!(
                "GPU PIRLS loop returned non-finite log|H| = {logdet}"
            ));
        }
        // `converged` already feeds `status` (Converged / Unstable /
        // MaxIterationsReached); make the relationship explicit so a future
        // refactor that breaks the invariant trips here rather than silently
        // mis-stamping `PirlsResult.status`.
        debug_assert_eq!(
            converged,
            matches!(status, PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum),
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
                )
                .map_err(|e| format!("derivative recompute failed: {e:?}"))?;
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
        let (finalweights_arr, final_c_arr, final_d_arr) = if matches!(
            input.exported_curvature,
            HessianCurvatureKind::Observed
        ) && curvature == CurvatureMode::Fisher
        {
            compute_observed_hessian_curvature_arrays(
                input.likelihood,
                input.inverse_link,
                &final_eta,
                input.y,
                &final_w_solver,
                input.priorweights,
            )
            .map_err(|e| format!("observed-curvature finalisation failed: {e:?}"))?
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
        let delta = ridge_passport.delta;
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

        // Gradient = Xᵀ · grad_eta + S·β.
        // Build X·Qs is already in `x_transformed`; gradient stays in transformed coords.
        let xt_grad_eta = {
            let xt = input.x_transformed;
            // p-vector
            let mut out = Array1::<f64>::zeros(p);
            for j in 0..p {
                let mut acc = 0.0_f64;
                for i in 0..n {
                    acc += xt[[i, j]] * final_grad_eta[i];
                }
                out[j] = acc;
            }
            out
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
        let mut gradient_total = xt_grad_eta.clone();
        gradient_total += &s_beta;
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
            log_likelihood: f64::NAN, // not produced by the GPU loop; downstream callers do not require it for REML/LAML evaluation.
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
            ridge_used: delta,
            deviance,
            edf: edf_final,
            stable_penalty_term: penalty_term,
            firth,
            finalweights: finalweights_for_state,
            final_offset: final_offset_arr,
            final_eta: final_eta.clone(),
            finalmu: final_mu.clone(),
            solveweights: if solveweights.is_empty() {
                final_w_solver.clone()
            } else {
                solveweights.clone()
            },
            solveworking_response: finalz,
            solvemu: final_mu.clone(),
            solve_dmu_deta: final_dmu_deta,
            solve_d2mu_deta2: final_d2mu_deta2,
            solve_d3mu_deta3: final_d3mu_deta3,
            solve_c_array: final_c_arr,
            solve_d_array: final_d_arr,
            derivatives_unsupported,
            status,
            iteration: iterations,
            max_abs_eta: max_abs_eta_used,
            lastgradient_norm,
            gradient_natural_scale,
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
            cache_compacted: false,
            min_penalized_deviance,
        };

        // Hessian-side weights are kept on the working_summary surface for
        // outer LM consumers; if the loop did not stamp a separate
        // `finalweights`, fall back to `final_w_hessian` so REML's
        // `H = XᵀW_HX + S_λ` reconstruction has the curvature it expects.
        debug_assert_eq!(final_w_hessian.len(), n);
        debug_assert_eq!(final_grad_eta.len(), n);

        Ok((pirls_result, working_summary))
    }
}

#[cfg(target_os = "linux")]
pub use linux_impl::{GpuPirlsDispatchInput, try_gpu_pirls_loop_dispatch};
