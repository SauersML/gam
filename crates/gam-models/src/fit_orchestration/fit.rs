use super::*;

pub(crate) fn survival_inverse_link_has_free_parameters(link: &InverseLink) -> bool {
    match link {
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => true,
        InverseLink::Mixture(state) => !state.rho.is_empty(),
        InverseLink::LatentCLogLog(_) | InverseLink::Standard(_) => false,
    }
}

#[derive(Debug)]
struct ProfiledOuterPayload<T> {
    theta: Array1<f64>,
    objective: f64,
    gradient: Array1<f64>,
    value: T,
}

/// Consume only the exact terminal payload reinstalled by the shared outer
/// runner. The sealed carrier owns the independently re-measured certificate;
/// bitwise identity across all analytic channels prevents a stale trial, an
/// independent refit, or caller-written convergence metadata from crossing
/// this fit-minting boundary.
fn consume_certified_profiled_outer_payload<T>(
    selected: Option<ProfiledOuterPayload<T>>,
    outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
    context: &str,
) -> Result<ProfiledOuterPayload<T>, String> {
    let selected = selected
        .ok_or_else(|| format!("{context} retained no optimizer-installed terminal profile"))?;
    if selected.theta.len() != outer.rho().len()
        || selected
            .theta
            .iter()
            .zip(outer.rho().iter())
            .any(|(selected, certified)| selected.to_bits() != certified.to_bits())
    {
        return Err(format!(
            "{context} terminal profile hyperparameters do not bitwise match the certified optimum"
        ));
    }
    if selected.objective.to_bits() != outer.final_value().to_bits() {
        return Err(format!(
            "{context} terminal profile objective does not bitwise match the certified optimum: selected={:.17e}, certified={:.17e}",
            selected.objective,
            outer.final_value(),
        ));
    }
    let certified_gradient = outer.final_gradient().ok_or_else(|| {
        format!("{context} certified result retained no analytic terminal gradient")
    })?;
    if selected.gradient.len() != certified_gradient.len()
        || selected
            .gradient
            .iter()
            .zip(certified_gradient.iter())
            .any(|(selected, certified)| selected.to_bits() != certified.to_bits())
    {
        return Err(format!(
            "{context} terminal profile gradient does not bitwise match the certified optimum"
        ));
    }
    Ok(selected)
}

#[cfg(test)]
mod profiled_outer_payload_tests {
    use super::*;
    use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
    use gam_solve::rho_optimizer::OuterProblem;

    fn certified_quadratic() -> gam_solve::rho_optimizer::CertifiedOuterResult {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_tolerance(1.0e-8)
            .with_max_iter(40)
            .with_initial_rho(Array1::from_vec(vec![0.5]))
            .with_seed_config(gam_problem::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        let mut objective = problem.build_objective(
            (),
            |_: &mut (), theta: &Array1<f64>| Ok(0.5 * (theta[0] - 0.25).powi(2)),
            |_: &mut (), theta: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * (theta[0] - 0.25).powi(2),
                    gradient: Array1::from_vec(vec![theta[0] - 0.25]),
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<
                fn(
                    &mut (),
                    &Array1<f64>,
                )
                    -> Result<gam_problem::EfsEval, gam_solve::estimate::EstimationError>,
            >,
        );
        problem
            .run_certified(&mut objective, "profiled-payload unit")
            .expect("quadratic outer problem must certify")
    }

    fn matching_payload(
        outer: &gam_solve::rho_optimizer::CertifiedOuterResult,
    ) -> ProfiledOuterPayload<&'static str> {
        ProfiledOuterPayload {
            theta: outer.rho().clone(),
            objective: outer.final_value(),
            gradient: outer
                .final_gradient()
                .expect("analytic fixture must retain its terminal gradient")
                .clone(),
            value: "terminal profile",
        }
    }

    #[test]
    fn selected_profile_requires_theta_objective_and_gradient_identity() {
        let outer = certified_quadratic();

        let mut wrong_theta = matching_payload(&outer);
        wrong_theta.theta[0] += 1.0;
        assert!(
            consume_certified_profiled_outer_payload(
                Some(wrong_theta),
                &outer,
                "theta substitution",
            )
            .expect_err("theta substitution must be rejected")
            .contains("hyperparameters")
        );

        let mut wrong_objective = matching_payload(&outer);
        wrong_objective.objective = f64::from_bits(wrong_objective.objective.to_bits() + 1);
        assert!(
            consume_certified_profiled_outer_payload(
                Some(wrong_objective),
                &outer,
                "objective substitution",
            )
            .expect_err("objective substitution must be rejected")
            .contains("objective")
        );

        let mut wrong_gradient = matching_payload(&outer);
        wrong_gradient.gradient[0] += 1.0;
        assert!(
            consume_certified_profiled_outer_payload(
                Some(wrong_gradient),
                &outer,
                "gradient substitution",
            )
            .expect_err("gradient substitution must be rejected")
            .contains("gradient")
        );

        let selected = consume_certified_profiled_outer_payload(
            Some(matching_payload(&outer)),
            &outer,
            "valid terminal profile",
        )
        .expect("the exact runner-installed terminal payload must be consumable");
        assert_eq!(selected.value, "terminal profile");
        assert!(outer.criterion_certificate().certifies());
    }
}

/// Lower floor applied before taking `ln(λ)` when mapping a smoothing parameter
/// into the log-λ optimization coordinate. `λ` is non-negative by construction;
/// flooring at the smallest positive normal `f64` keeps `ln` finite for an
/// exactly-zero (fully-relaxed) penalty without perturbing any λ above the
/// denormal range.

/// Inner-PIRLS controls shared by the survival-transformation baseline and
/// smoothing-coordinate eval closures. The baseline geometry is mildly
/// nonlinear, so the iteration budget is generous; the convergence/step floors
/// match the working-model PIRLS contract used throughout the survival path.
const SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS: usize = 400;

const SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL: f64 = 1e-6;

const SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING: usize = 40;

const SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE: f64 = 1e-12;

struct SurvivalLocationScaleProfile {
    fit: SurvivalLocationScaleTermFitResult,
    inverse_link: InverseLink,
    wiggle_knots: Option<Array1<f64>>,
    wiggle_degree: Option<usize>,
    inverse_link_outer: Option<gam_solve::rho_optimizer::CertifiedOuterResult>,
}

fn survival_inverse_link_profile_objective(
    profile: &SurvivalLocationScaleProfile,
    context: &str,
) -> Result<f64, String> {
    let objective = -profile.fit.fit.log_likelihood + 0.5 * profile.fit.fit.stable_penalty_term;
    if objective.is_finite() {
        Ok(objective)
    } else {
        Err(format!(
            "{context}: non-finite profile objective (log_likelihood={}, stable_penalty_term={})",
            profile.fit.fit.log_likelihood, profile.fit.fit.stable_penalty_term,
        ))
    }
}

fn survival_pirls_status_is_certified(status: gam_solve::pirls::PirlsStatus) -> bool {
    status.is_converged()
}

fn require_certified_survival_pirls(
    summary: &gam_solve::pirls::WorkingModelPirlsResult,
    context: &str,
    parameter_checkpoint: &[f64],
    durable_checkpoint_key: Option<&str>,
) -> Result<(), String> {
    if survival_pirls_status_is_certified(summary.status) {
        return Ok(());
    }
    Err(format!(
        "{context} did not produce a strict PIRLS convergence certificate \
         (status={:?}, iterations={}, projected_gradient_norm={:.6e}, \
         deviance={:.6e}, min_penalized_deviance={:.6e}, last_step_size={:.6e}, \
         last_step_halving={}, parameter_checkpoint={parameter_checkpoint:?}{}). The accepted \
         iterate is checkpoint evidence only; no fit was minted.",
        summary.status,
        summary.iterations,
        summary.lastgradient_norm,
        summary.state.deviance,
        summary.min_penalized_deviance,
        summary.last_step_size,
        summary.last_step_halving,
        durable_checkpoint_key
            .map(|key| format!(", durable_checkpoint_key={key}"))
            .unwrap_or_default(),
    ))
}

/// Encode a nonlinear baseline candidate in the exact coordinates consumed by
/// its outer optimizer, so non-convergence evidence can be passed back as a
/// directly resumable checkpoint rather than as raw distribution parameters.
fn survival_baseline_parameter_checkpoint(
    config: &crate::survival::construction::SurvivalBaselineConfig,
) -> Result<Vec<f64>, String> {
    let required = |name: &str, value: Option<f64>| {
        value
            .filter(|candidate| candidate.is_finite())
            .ok_or_else(|| format!("survival baseline checkpoint is missing finite {name}"))
    };
    let positive_log = |name: &str, value: Option<f64>| {
        let value = required(name, value)?;
        if value > 0.0 {
            Ok(value.ln())
        } else {
            Err(format!(
                "survival baseline checkpoint requires positive {name}, got {value}"
            ))
        }
    };

    use crate::survival::construction::SurvivalBaselineTarget;
    match config.target {
        SurvivalBaselineTarget::Linear => Ok(Vec::new()),
        SurvivalBaselineTarget::Weibull => Ok(vec![
            positive_log("Weibull scale", config.scale)?,
            positive_log("Weibull shape", config.shape)?,
        ]),
        SurvivalBaselineTarget::Gompertz => Ok(vec![
            positive_log("Gompertz rate", config.rate)?,
            required("Gompertz shape", config.shape)?,
        ]),
        SurvivalBaselineTarget::GompertzMakeham => Ok(vec![
            positive_log("Gompertz-Makeham rate", config.rate)?,
            required("Gompertz-Makeham shape", config.shape)?,
            positive_log("Gompertz-Makeham makeham", config.makeham)?,
        ]),
    }
}

impl SurvivalLocationScaleProfile {
    fn into_result(self) -> SurvivalLocationScaleFitResult {
        SurvivalLocationScaleFitResult {
            fit: self.fit,
            inverse_link: self.inverse_link,
            wiggle_knots: self.wiggle_knots,
            wiggle_degree: self.wiggle_degree,
            inverse_link_outer: self.inverse_link_outer,
        }
    }
}
fn resolved_wiggle_inverse_link(
    spec: &LikelihoodSpec,
    fit: &UnifiedFitResult,
    fallback: &InverseLink,
) -> Result<InverseLink, String> {
    let resolved = match fit.fitted_link_state(spec).map_err(|e| e.to_string())? {
        FittedLinkState::Standard(Some(link)) => InverseLink::Standard(link),
        FittedLinkState::Standard(None) => fallback.clone(),
        FittedLinkState::LatentCLogLog { state } => InverseLink::LatentCLogLog(state),
        FittedLinkState::Sas { state, .. } => InverseLink::Sas(state),
        FittedLinkState::BetaLogistic { state, .. } => InverseLink::BetaLogistic(state),
        FittedLinkState::Mixture { state, .. } => InverseLink::Mixture(state),
    };
    require_inverse_link_supports_joint_wiggle(&resolved, "standard link wiggle")?;
    Ok(resolved)
}

/// Run the base standard fit (the three-way latent / coefficient-group /
/// spatial dispatch) at an explicit [`FitOptions`], leaving the caller's
/// `request.options` untouched. Split out of [`fit_standard_model`] so the
/// #1762 near-separation Firth fallback can re-run the identical fit with the
/// Jeffreys penalty enabled without duplicating the dispatch.
type StandardBaseFit = crate::fit_orchestration::drivers::FittedTermCollectionWithSpec;

fn fit_standard_base(
    request: &StandardFitRequest<'_>,
    family: &LikelihoodSpec,
    options: &FitOptions,
) -> Result<StandardBaseFit, gam_solve::estimate::EstimationError> {
    if let Some(latent_coord) = request.latent_coord.as_ref() {
        if !request.coefficient_groups.is_empty() || !request.penalty_block_gamma_priors.is_empty()
        {
            return Err(gam_solve::estimate::EstimationError::InvalidInput(
                "latent-coordinate standard fits do not support coefficient_groups or \
                 penalty_block_gamma_priors in the same request"
                    .to_string(),
            ));
        }
        fit_term_collectionwith_latent_coord_optimization(
            request.data.view(),
            request.y.as_ref().clone(),
            request.weights.as_ref().clone(),
            request.offset.as_ref().clone(),
            &request.spec,
            latent_coord,
            family.clone(),
            options,
        )
    } else if !request.coefficient_groups.is_empty()
        || !request.penalty_block_gamma_priors.is_empty()
    {
        let fitted = fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors(
            request.data.view(),
            request.y.view(),
            request.weights.view(),
            request.offset.view(),
            &request.spec,
            &request.coefficient_groups,
            &request.penalty_block_gamma_priors,
            family.clone(),
            options,
        )?;
        let resolvedspec = crate::fit_orchestration::drivers::freeze_term_collection_from_design(
            &request.spec,
            &fitted.design,
        )?;
        Ok(
            crate::fit_orchestration::drivers::FittedTermCollectionWithSpec {
                fit: fitted.fit,
                design: fitted.design,
                resolvedspec,
                adaptive_diagnostics: fitted.adaptive_diagnostics,
                kappa_timing: None,
            },
        )
    } else {
        fit_term_collectionwith_spatial_length_scale_optimization(
            request.data.view(),
            request.y.as_ref().clone(),
            request.weights.as_ref().clone(),
            request.offset.as_ref().clone(),
            &request.spec,
            family.clone(),
            options,
            &request.kappa_options,
        )
    }
}

fn firth_can_rescue(error: &gam_solve::estimate::EstimationError) -> bool {
    use gam_solve::estimate::EstimationError;
    error.is_inner_solve_retreat()
        || matches!(
            error,
            EstimationError::PrefitPerfectSeparationDetected { .. }
                | EstimationError::PrefitLinearSeparationDetected { .. }
                | EstimationError::RemlDidNotConverge { .. }
        )
}

fn certified_retry_or_original<T, E>(original: E, retry: Result<T, E>) -> Result<T, E> {
    match retry {
        Ok(value) => Ok(value),
        Err(_) => Err(original),
    }
}

fn rescale_covariance_coordinates(covariance: &mut Array2<f64>, factors: &[f64]) {
    let dimension = factors.len();
    assert_eq!(
        covariance.dim(),
        (dimension, dimension),
        "covariance must align with the remapped coefficient vector"
    );
    for i in 0..dimension {
        for j in 0..dimension {
            covariance[[i, j]] *= factors[i] * factors[j];
        }
    }
}

fn rescale_precision_coordinates(
    precision: &mut gam_problem::dispersion_cov::UnscaledPrecision,
    factors: &[f64],
) {
    let dimension = factors.len();
    assert_eq!(
        precision.dim(),
        (dimension, dimension),
        "precision must align with the remapped coefficient vector"
    );
    for i in 0..dimension {
        for j in 0..dimension {
            precision[[i, j]] /= factors[i] * factors[j];
        }
    }
}

#[cfg(test)]
mod standard_convergence_gate_tests {
    use super::{
        certified_retry_or_original, firth_can_rescue, rescale_covariance_coordinates,
        rescale_precision_coordinates, survival_baseline_parameter_checkpoint,
        survival_pirls_status_is_certified,
    };
    use crate::survival::construction::{SurvivalBaselineConfig, SurvivalBaselineTarget};
    use gam_solve::estimate::EstimationError;
    use gam_solve::pirls::PirlsStatus;
    use ndarray::array;

    #[test]
    fn raw_coordinate_precision_is_the_inverse_congruence_of_covariance() {
        let mut covariance = array![[0.30, -0.10], [-0.10, 0.70]];
        let mut precision =
            gam_problem::dispersion_cov::UnscaledPrecision::wrap(array![[3.5, 0.5], [0.5, 1.5]]);
        let factors = [4.0, 1.0];

        rescale_covariance_coordinates(&mut covariance, &factors);
        rescale_precision_coordinates(&mut precision, &factors);

        assert_eq!(covariance, array![[4.8, -0.4], [-0.4, 0.7]]);
        assert_eq!(
            precision.as_array(),
            &array![[0.21875, 0.125], [0.125, 1.5]]
        );
        let identity = precision.as_array().dot(&covariance);
        for i in 0..2 {
            for j in 0..2 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((identity[[i, j]] - target).abs() <= 2e-15);
            }
        }
    }

    #[test]
    fn failed_retry_returns_original_evidence() {
        let result = certified_retry_or_original::<(), _>("base evidence", Err("retry evidence"));
        assert_eq!(result, Err("base evidence"));
        assert_eq!(
            certified_retry_or_original("base evidence", Ok::<_, &str>(7)),
            Ok(7)
        );
    }

    #[test]
    fn survival_gate_rejects_every_exhausted_or_stalled_status() {
        assert!(survival_pirls_status_is_certified(PirlsStatus::Converged));
        for status in [
            PirlsStatus::StalledAtValidMinimum,
            PirlsStatus::MaxIterationsReached,
            PirlsStatus::LmStepSearchExhausted,
            PirlsStatus::Unstable,
        ] {
            assert!(!survival_pirls_status_is_certified(status));
        }
    }

    #[test]
    fn survival_baseline_checkpoint_matches_outer_coordinates() {
        let checkpoint = survival_baseline_parameter_checkpoint(&SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(-0.25),
            rate: Some(2.0),
            makeham: Some(4.0),
        })
        .expect("valid baseline checkpoint");
        assert_eq!(checkpoint, vec![2.0_f64.ln(), -0.25, 4.0_f64.ln()]);
    }

    #[test]
    fn firth_retry_is_limited_to_separation_and_nonconvergence() {
        assert!(firth_can_rescue(&EstimationError::PirlsDidNotConverge {
            max_iterations: 20,
            last_change: 1.0,
        }));
        assert!(firth_can_rescue(
            &EstimationError::PrefitPerfectSeparationDetected {
                column_index: 0,
                threshold: 0.0,
                positive_above_threshold: true,
            }
        ));
        assert!(!firth_can_rescue(&EstimationError::InvalidInput(
            "structural mismatch".to_string()
        )));
    }
}

/// **Exact** Tweedie log-likelihood of a fit at a fixed variance power `p` — the
/// profile objective maximized to estimate `p` (#2026 / #2105).
///
/// Refits the mean/dispersion GLM at `Tweedie { p }` reusing the existing fit
/// machinery (no reimplementation), reconstructs the fitted mean
/// `μ = exp(Xβ̂ + offset)` on the log link, and evaluates the **exact** Jørgensen
/// compound-Poisson–gamma log-density (`gam_solve::pirls::tweedie_exact_loglik_total_from_eta`)
/// at the estimated dispersion `φ̂(p)`.
///
/// CRITICAL (#2105): the objective must be the *exact* EDM density, NOT the
/// separately named saddlepoint approximation. The saddlepoint is exact
/// only in the many-jumps (large Poisson-rate λ) limit; at the moderate λ of a
/// typical Tweedie fit its missing `O(1/λ)` normalizer correction, summed across
/// the sample, biases the profile maximizer **low** (e.g. `p̂ ≈ 1.33` on `p = 1.5`
/// data). Because the reported dispersion is the Pearson estimate
/// `φ̂ = Σw(y−μ)²/μ^p / Σw`, an under-estimated `p` inflates `φ̂` and every SE /
/// interval scaled by `√φ̂`. mgcv's `tw()` profiles `p` on the same exact series
/// (`ldTweedie`) for exactly this reason.
///
/// Returns `None` when the refit fails or yields a non-finite objective so the
/// profile search can skip that node rather than abort.
fn tweedie_profile_loglik(request: &StandardFitRequest<'_>, p: f64) -> Option<f64> {
    if !gam_spec::is_valid_tweedie_power(p) {
        return None;
    }
    let family = LikelihoodSpec::new(ResponseFamily::Tweedie { p }, request.family.link.clone());
    let fitted = fit_standard_base(request, &family, &request.options).ok()?;
    // μ = g⁻¹(Xβ̂ + offset); the Tweedie family is fixed to the log link by
    // `resolve_family`, so g⁻¹ = exp. `design.apply` reproduces the fitted
    // linear predictor exactly (same contract the expectile/predict paths use).
    // `design.apply` already folds the design's fixed affine channel (non-zero
    // B-spline endpoint anchor, #2297) into `Xβ̂`, so only the user offset is
    // added here; adding `affine_offset` again would double-count the pin.
    let mut eta = fitted.design.apply(fitted.fit.beta.view()).ok()?;
    if eta.len() != request.y.len() {
        return None;
    }
    eta += request.offset.as_ref();
    let mu = eta.mapv(f64::exp);
    // Profile the dispersion out at this `p` with the SAME prior-weighted Pearson
    // moment estimator the inner solver uses to report the Tweedie `φ̂`
    // (`estimate_tweedie_phi_from_eta`): `φ̂ = Σ wᵢ (yᵢ − μᵢ)² / μᵢ^p / Σ wᵢ`.
    // Computing it here from the reconstructed mean makes the profile objective
    // independent of whether the fit retained an inference/covariance block (the
    // seed `φ = 1` carried on `likelihood_scale` would otherwise bias every node
    // identically-but-wrongly) and gives each `p` its own maximized dispersion —
    // the whole point of a profile likelihood.
    const PHI_MIN: f64 = 1e-6;
    const PHI_MAX: f64 = 1e12;
    let mut weighted_pearson = 0.0_f64;
    let mut total_weight = 0.0_f64;
    for ((&yi, &mui), &wi) in request.y.iter().zip(mu.iter()).zip(request.weights.iter()) {
        let wi = wi.max(0.0);
        if wi == 0.0 {
            continue;
        }
        let resid = yi - mui;
        let var_unit = mui.powf(p).max(f64::MIN_POSITIVE);
        weighted_pearson += wi * resid * resid / var_unit;
        total_weight += wi;
    }
    if total_weight <= 0.0 || !weighted_pearson.is_finite() || weighted_pearson <= 0.0 {
        return None;
    }
    let phi = (weighted_pearson / total_weight).clamp(PHI_MIN, PHI_MAX);
    // Evaluate the EXACT compound-Poisson–gamma density at φ̂(p) — the profile
    // objective mgcv's `tw()` maximizes. Using a saddlepoint here is what biased
    // `p̂` low and inflated `φ̂` (#2105).
    let ll = gam_solve::pirls::tweedie_exact_loglik_total_from_eta(
        request.y.view(),
        eta.view(),
        request.weights.view(),
        p,
        phi,
    )
    .ok()?;
    Some(ll)
}

/// Estimate the Tweedie variance power `p ∈ (1, 2)` by profile likelihood, mgcv
/// `tw()`-style (#2026), via a **golden-section search** over the whole open
/// interval `(1, 2)` — NO fixed node grid (SPEC: grid search is never allowed;
/// #2064). The profile objective [`tweedie_profile_loglik`] (a full mean/
/// dispersion refit at each `p`) is smooth and unimodal in `p` for the
/// compound-Poisson-gamma law, so golden section brackets the maximizer directly
/// and contracts the bracket by the golden ratio each step. The mean fit is
/// robust to a misspecified `p`, but the observation-interval calibration is
/// not, so this is what makes bare `family="tweedie"` track data whose true
/// power ≠ 1.5.
fn estimate_tweedie_power(request: &StandardFitRequest<'_>) -> Result<f64, String> {
    // Endpoints excluded: the Tweedie unit-deviance / saddlepoint density is
    // singular at p = 1 (Poisson) and p = 2 (gamma).
    const EPS: f64 = 1e-3;
    // Converge the bracket to this width in `p`; finer than the physically
    // meaningful precision of the variance power.
    const TOL: f64 = 1e-3;
    let mut a = 1.0 + EPS;
    let mut b = 2.0 - EPS;
    let inv_phi_gr = (5.0_f64.sqrt() - 1.0) / 2.0; // 1/φ ≈ 0.618
    let eval = |p: f64| tweedie_profile_loglik(request, p).unwrap_or(f64::NEG_INFINITY);
    // Iteration count DERIVED from the tolerance (not a magic cap): each step
    // multiplies the bracket width by `inv_phi_gr`, so `n` steps with
    // `(b−a)·inv_phi_gr^n ≤ TOL` guarantees convergence to `TOL`.
    let n_iter = (((TOL / (b - a)).ln() / inv_phi_gr.ln()).ceil() as i64).max(1) as usize;
    let mut c = b - inv_phi_gr * (b - a);
    let mut d = a + inv_phi_gr * (b - a);
    let mut fc = eval(c);
    let mut fd = eval(d);
    for _ in 0..n_iter {
        if fc >= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi_gr * (b - a);
            fc = eval(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi_gr * (b - a);
            fd = eval(d);
        }
    }
    let p_hat = (0.5 * (a + b)).clamp(1.0 + EPS, 2.0 - EPS);
    // A finite profile likelihood at the maximizer certifies the search found a
    // usable optimum (degenerate data — every `p` non-finite — fails here rather
    // than silently returning the interval midpoint).
    if !tweedie_profile_loglik(request, p_hat).is_some_and(f64::is_finite) {
        return Err(
            "tweedie power profiling failed: the profile likelihood is non-finite across \
             (1, 2); set an explicit power via family=\"tweedie(p)\""
                .to_string(),
        );
    }
    log::info!(
        "[tweedie#2026] estimated variance power p={p_hat:.4} by golden-section profile \
         likelihood (bare family=\"tweedie\"); set family=\"tweedie(p)\" to pin it."
    );
    Ok(p_hat)
}

pub(crate) fn fit_standard_model(
    mut request: StandardFitRequest<'_>,
) -> Result<StandardFitResult, String> {
    // #2026: magic-by-default Tweedie power. A bare `family="tweedie"`/`"tw"`
    // arrives with a placeholder power and `estimate_tweedie_p = true`; profile
    // `p` over (1, 2) by likelihood and bake the estimate into `family` so the
    // reported fit — and every observation interval derived from it — uses the
    // data-driven power rather than the interior fallback. An explicit
    // `tweedie(1.6)` leaves the flag `false` and skips this entirely.
    if request.estimate_tweedie_p
        && matches!(request.family.response, ResponseFamily::Tweedie { .. })
    {
        let p_hat = estimate_tweedie_power(&request)?;
        request.family = LikelihoodSpec::new(
            ResponseFamily::Tweedie { p: p_hat },
            request.family.link.clone(),
        );
        // The power is now pinned; the final fit below is an ordinary fixed-p
        // Tweedie fit.
        request.estimate_tweedie_p = false;
    }

    // #1762: near-perfect linear separation drives the binomial REML/ARC
    // outer optimizer into a FLAT-VALLEY STALL. As the fit approaches
    // separation the coefficients want to run to infinity, the PIRLS working
    // weights w = μ̂(1−μ̂) collapse to ~0 over the saturated majority of rows,
    // and the inner solve can no longer certify a minimum at the small-λ REML
    // optimum — so the outer optimizer wanders the flat valley, burns its
    // cost-stall escapes, and reports NON-CONVERGED (the ~117s / |g|≫tol
    // pathology; separation can also surface as a hard PerfectSeparation /
    // PirlsDidNotConverge error). Firth's Jeffreys-prior penalty is the textbook
    // remedy: it bounds the coefficients and keeps the working weights from
    // collapsing, so the inner solve is well conditioned at every λ and the
    // outer optimizer certifies quickly.
    //
    // #2273: this pathology is NOT logit-specific. The coefficient runaway and
    // Fisher-weight collapse under separation happen on EVERY binomial link
    // (probit's Φ, cloglog, loglog, cauchit, and the stateful SAS/Beta-Logistic/
    // Mixture links) — measured directly on the issue's n=6 exact-separation
    // fixture, where the probit fit halts on a flat-valley stall (|g|≈1.9e2 ≫
    // bound) and mints only under Firth. Firth's Jeffreys prior is a link-general
    // remedy: it is defined for any binomial inverse link that exposes a
    // Fisher-weight jet (exactly `LikelihoodSpec::supports_firth`, the same gate
    // `--firth` validates against), so the reactive rescue must be armed for the
    // whole Firth-capable binomial family, not just the logit special case — else
    // the README's "Firth / Jeffreys bias reduction handles separation in
    // binomial fits" promise silently fails to hold off the default link.
    //
    // Retry ONCE with Firth when a plain (non-Firth) Firth-capable binomial fit
    // fails with typed separation/non-convergence evidence, and adopt it only if
    // the retry itself carries both inner and outer convergence certificates. If
    // the retry fails, return the ORIGINAL base error unchanged; a failed rescue
    // can never replace its evidence or mint the abandoned base iterate.
    // Structural errors are not Firth-retryable, and links without a Fisher-weight
    // jet fall straight through to the original error (arming Firth on them would
    // itself be rejected, then reduced back to the original error by
    // `certified_retry_or_original`).
    let is_firth_capable_binomial = request.family.supports_firth();
    let base = fit_standard_base(&request, &request.family, &request.options);
    let fitted = match base {
        Ok(fitted) => fitted,
        Err(original_error)
            if is_firth_capable_binomial
                && !request.options.firth_bias_reduction
                && firth_can_rescue(&original_error) =>
        {
            let original_report = original_error.to_string();
            let mut firth_options = request.options.clone();
            firth_options.firth_bias_reduction = true;
            let firth = fit_standard_base(&request, &request.family, &firth_options);
            let firth_failure = firth.as_ref().err().map(ToString::to_string);
            match certified_retry_or_original(original_error, firth) {
                Ok(firth_fitted) => {
                    log::info!(
                        "[#1762/#2273] Firth-capable binomial base fit ({}) failed with \
                         retryable separation/non-convergence evidence ({original_report}); Firth \
                         bias-reduction retry certified — adopting it (Firth edf {:.2}).",
                        request.family.pretty_name(),
                        firth_fitted.fit.edf_total().unwrap_or(f64::NAN),
                    );
                    firth_fitted
                }
                Err(original_error) => {
                    log::warn!(
                        "[#1762/#2273] Firth-capable binomial base fit ({}) failed \
                         ({original_report}); Firth retry also failed to certify ({}) — returning \
                         the original typed base evidence, not either abandoned iterate.",
                        request.family.pretty_name(),
                        firth_failure.unwrap_or_else(|| "unknown retry failure".to_string()),
                    );
                    return Err(original_error.to_string());
                }
            }
        }
        Err(error) => return Err(error.to_string()),
    };

    let adaptive_spatial_terms = adaptive_spatial_term_mask(&request.spec);
    let adaptive_spatial_center_counts = adaptive_spatial_center_counts(&request.spec);
    let result = StandardFitResult {
        saved_link_state: fitted.fit.fitted_link.clone(),
        fit: fitted.fit,
        design: fitted.design,
        resolvedspec: fitted.resolvedspec,
        adaptive_spatial_terms: adaptive_spatial_terms.clone(),
        adaptive_spatial_center_counts: adaptive_spatial_center_counts.clone(),
        adaptive_diagnostics: fitted.adaptive_diagnostics,
        kappa_timing: fitted.kappa_timing,
        wiggle_knots: None,
        wiggle_degree: None,
        wiggle_penalty_metadata: None,
        wiggle_saved_warp_beta: None,
        wiggle_saved_index_shift: None,
    };

    let Some(wiggle) = request.wiggle else {
        return Ok(result);
    };
    // `StandardBinomialWiggleConfig` now carries `refit_options` directly, so
    // the previous "pilot config present, blockwise options missing" failure
    // state (#320) is unrepresentable at the type level.
    let wiggle_options = wiggle.refit_options.clone();
    let wiggle_link_kind =
        resolved_wiggle_inverse_link(&request.family, &result.fit, &wiggle.link_kind)?;
    let selected_wiggle_basis = select_binomial_mean_link_wiggle_basis_from_pilot(
        &result.design,
        &result.fit,
        &WiggleBlockConfig {
            degree: wiggle.wiggle.degree,
            num_internal_knots: wiggle.wiggle.num_internal_knots,
            penalty_order: 2,
            double_penalty: wiggle.wiggle.double_penalty,
        },
        &wiggle.wiggle.penalty_orders,
    )?;
    let wiggle_penalty_metadata = selected_wiggle_basis.penalty_metadata.clone();

    // A penalized, monotone-constrained link-offset spline shrinks to zero at
    // large smoothing, so the no-wiggle pilot fit (`result`) is the *exact*
    // large-`λ` limit of the wiggle model — the wiggle model contains the
    // baseline as a limiting case. The wiggle refit is a coupled joint
    // Newton solve (`BinomialMeanWiggleFamily`) on top of that pilot; on the
    // hardest binomial regimes it can still fail to certify KKT convergence:
    // the I-spline warp `q = η + B(η)·β_w` can drive the linear predictor
    // toward link saturation, where the per-cycle data curvature collapses
    // and the joint trust region shrinks faster than the active-set QP can
    // pin the binding monotonicity rows (gam#872).
    //
    // #1596: when that solve does not converge we now surface the failure
    // LOUDLY (see the `Err` arm below) instead of silently returning the
    // no-wiggle baseline. The baseline IS the large-`λ` limit, so falling back
    // to it produces a finite, valid fit — but a `link(type=flexible(...))`
    // request answered with a model bit-identical to the fixed base link, with
    // no signal that the warp never engaged, is a silent contract violation:
    // the caller cannot tell a genuinely-flat learned link from a non-converged
    // one. The divergence failure mode (the unconditional Jeffreys/Firth
    // augmentation blowing the augmented objective up to ~1e9 on this path) is
    // fixed at the root by `BinomialMeanWiggleFamily::joint_jeffreys_term_required
    // = false`; the loud `Err` below catches the residual trust-region/active-set
    // non-convergence that the root fix cannot.
    let solved = match fit_binomial_mean_wiggle_terms_with_selected_basis(
        request.data.view(),
        &result.resolvedspec,
        &result.design,
        &result.fit,
        request.y.as_ref(),
        request.weights.as_ref(),
        wiggle_link_kind,
        selected_wiggle_basis,
        &wiggle_options,
        &request.kappa_options,
    ) {
        Ok(solved) => solved,
        Err(e) => {
            // The flexible/learnable link the formula asked for could not be
            // fitted: the coupled link-wiggle joint Newton solve failed to
            // certify convergence. Previously this arm silently `return
            // Ok(result)` with the *no-wiggle baseline* (the large-smoothing
            // limit), so a `link(type=flexible(...))` request returned a model
            // bit-identical to the fixed base link, with no signal to the
            // caller — the warp never engaged but the fit looked successful
            // (#1596). Returning the baseline as if the request were honored is
            // a silent contract violation. Surface the non-convergence LOUDLY
            // (a real `Err` the caller sees), matching how the SAS / mixture
            // adaptive-link paths now report startup-validation failures
            // (#1571/#1572). The fit is NOT silently downgraded.
            log::warn!("[linkwiggle] binomial mean link-wiggle joint solve did not converge ({e})");
            return Err(format!(
                "flexible/learnable link requested via link(type=flexible(...)) / \
                 linkwiggle(...), but the binomial mean link-wiggle joint solve did not \
                 converge ({e}). The fit was NOT silently downgraded to the fixed base \
                 link. Refit with a fixed link (e.g. logit/probit/cloglog) or adjust the \
                 wiggle spec (linkwiggle(internal_knots=...)). See gam#1596."
            ));
        }
    };

    Ok(StandardFitResult {
        saved_link_state: result.saved_link_state,
        fit: solved.fit,
        design: solved.design,
        resolvedspec: solved.resolvedspec,
        adaptive_spatial_terms,
        adaptive_spatial_center_counts,
        adaptive_diagnostics: result.adaptive_diagnostics,
        kappa_timing: result.kappa_timing,
        wiggle_knots: Some(solved.wiggle_knots),
        wiggle_degree: Some(solved.wiggle_degree),
        wiggle_penalty_metadata: Some(wiggle_penalty_metadata),
        wiggle_saved_warp_beta: solved.saved_warp_beta,
        wiggle_saved_index_shift: solved.saved_index_shift,
    })
}

/// Broken-out pieces of a location-scale fit request, family-agnostic.
///
/// Both the Gaussian and binomial location-scale requests are structurally the
/// same — a borrowed data matrix, a family-specific term spec, an optional link
/// wiggle config, and the two option bundles. The shared wiggle-pilot engine
/// ([`fit_location_scale_with_optional_wiggle`]) consumes these parts; each
/// family's request type lowers itself into them via
/// [`LocationScaleWorkflowAdapter::into_parts`].
struct LocationScaleWorkflowParts<'a, S> {
    data: ArrayView2<'a, f64>,
    spec: S,
    wiggle: Option<LinkWiggleConfig>,
    options: BlockwiseFitOptions,
    kappa_options: SpatialLengthScaleOptimizationOptions,
}

/// Family-specific glue for the shared location-scale wiggle-pilot workflow.
///
/// The workflow policy (pilot fit — which also enforces any family wiggle
/// compatibility guard — → select link wiggle basis from the pilot → refit with
/// the selected wiggle → extract `beta_link_wiggle` and assemble; otherwise the
/// plain non-wiggle fit) is identical across Gaussian and binomial
/// location-scale models — only the family fit/select/refit functions and result
/// type differ (#430). An adapter supplies exactly those family-specific
/// operations; the engine owns the policy.
trait LocationScaleWorkflowAdapter {
    /// The owned term spec for this family (`GaussianLocationScaleTermSpec` /
    /// `BinomialLocationScaleTermSpec`).
    type Spec;
    /// The borrowed request type the public model entry point receives.
    type Request<'a>;
    /// The family-specific fit result the engine assembles.
    type Result;

    /// Lower the borrowed request into the family-agnostic workflow parts.
    fn into_parts<'a>(request: Self::Request<'a>) -> LocationScaleWorkflowParts<'a, Self::Spec>;

    /// Pilot fit on the bare (non-wiggle) spec, used to seed the wiggle-basis
    /// selector. This is the first work the wiggle path performs, so any
    /// family-specific wiggle compatibility guard (e.g. the binomial inverse
    /// link must support a joint wiggle refit) is enforced here before fitting.
    /// The adapter clones whatever spec fields the pilot consumes so the caller
    /// retains ownership of `spec` for the subsequent refit.
    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String>;

    /// Select the link-wiggle basis from the pilot, then refit the full model
    /// with that selected wiggle block. Consumes `spec`.
    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, String>;

    /// Plain non-wiggle fit, used when no wiggle config is present. Consumes
    /// `spec`.
    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String>;

    /// Assemble the family result from a non-wiggle fit (knots/degree/wiggle
    /// coefficients all absent).
    fn assemble_plain(fit: BlockwiseTermFitResult) -> Self::Result;

    /// Assemble the family result from a wiggle refit, carrying the selected
    /// knots/degree and the extracted `beta_link_wiggle` block.
    fn assemble_with_wiggle(
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result;
}

/// Shared wiggle-pilot workflow for Gaussian and binomial location-scale models
/// (#430). The single source of truth for the policy; families differ only via
/// their [`LocationScaleWorkflowAdapter`].
fn fit_location_scale_with_optional_wiggle<A: LocationScaleWorkflowAdapter>(
    request: A::Request<'_>,
) -> Result<A::Result, String> {
    let LocationScaleWorkflowParts {
        data,
        spec,
        wiggle,
        options,
        kappa_options,
    } = A::into_parts(request);

    let Some(wiggle_cfg) = wiggle else {
        let fit = A::fit_plain(data, spec, &options, &kappa_options)?;
        return Ok(A::assemble_plain(fit));
    };

    let pilot = A::fit_pilot(data, &spec, &options, &kappa_options)?;
    let solved =
        A::refit_with_selected_wiggle(data, spec, &pilot, &wiggle_cfg, &options, &kappa_options)?;

    // The selected link-wiggle basis is appended as the third blockwise term
    // (after the mean/threshold and log-σ blocks), so its coefficients live in
    // block 2 of the refit.
    let fit = solved.fit.fit;
    let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
    let assembled_fit = BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
        fit,
        meanspec_resolved: solved.fit.meanspec_resolved,
        noisespec_resolved: solved.fit.noisespec_resolved,
        mean_design: solved.fit.mean_design,
        noise_design: solved.fit.noise_design,
    })?;
    Ok(A::assemble_with_wiggle(
        assembled_fit,
        solved.wiggle_knots,
        solved.wiggle_degree,
        beta_link_wiggle,
    ))
}

/// Gaussian location-scale adapter for the shared wiggle-pilot workflow.
struct GaussianLocationScaleWorkflow;

impl LocationScaleWorkflowAdapter for GaussianLocationScaleWorkflow {
    type Spec = GaussianLocationScaleTermSpec;
    type Request<'a> = GaussianLocationScaleFitRequest<'a>;
    type Result = GaussianLocationScaleFitResult;

    fn into_parts<'a>(request: Self::Request<'a>) -> LocationScaleWorkflowParts<'a, Self::Spec> {
        LocationScaleWorkflowParts {
            data: request.data,
            spec: request.spec,
            wiggle: request.wiggle,
            options: request.options,
            kappa_options: request.kappa_options,
        }
    }

    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        // Gaussian location-scale uses an identity mean link; the joint wiggle
        // refit is always admissible, so the pilot fits with no extra guard.
        fit_gaussian_location_scale_terms(
            data,
            GaussianLocationScaleTermSpec {
                y: spec.y.clone(),
                weights: spec.weights.clone(),
                meanspec: spec.meanspec.clone(),
                log_sigmaspec: spec.log_sigmaspec.clone(),
                mean_offset: spec.mean_offset.clone(),
                log_sigma_offset: spec.log_sigma_offset.clone(),
            },
            options,
            kappa_options,
        )
    }

    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, String> {
        let selected_wiggle_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        fit_gaussian_location_scale_terms_with_selected_wiggle(
            data,
            spec,
            selected_wiggle_basis,
            options,
            kappa_options,
        )
    }

    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        fit_gaussian_location_scale_terms(data, spec, options, kappa_options)
    }

    fn assemble_plain(fit: BlockwiseTermFitResult) -> Self::Result {
        GaussianLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
            // The wiggle-pilot workflow fits in standardized response units; the
            // Gaussian model wrapper (`fit_gaussian_location_scale_model`) maps
            // the coefficients back to raw units and overwrites this with the
            // applied factor. `1.0` here is the identity (no standardization).
            response_scale: 1.0,
        }
    }

    fn assemble_with_wiggle(
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result {
        GaussianLocationScaleFitResult {
            fit,
            wiggle_knots: Some(wiggle_knots),
            wiggle_degree: Some(wiggle_degree),
            beta_link_wiggle,
            // See `assemble_plain`: raw-unit remapping happens in the Gaussian
            // model wrapper, which overwrites this with the applied factor.
            response_scale: 1.0,
        }
    }
}

/// Binomial location-scale adapter for the shared wiggle-pilot workflow.
struct BinomialLocationScaleWorkflow;

impl LocationScaleWorkflowAdapter for BinomialLocationScaleWorkflow {
    type Spec = BinomialLocationScaleTermSpec;
    type Request<'a> = BinomialLocationScaleFitRequest<'a>;
    type Result = BinomialLocationScaleFitResult;

    fn into_parts<'a>(request: Self::Request<'a>) -> LocationScaleWorkflowParts<'a, Self::Spec> {
        LocationScaleWorkflowParts {
            data: request.data,
            spec: request.spec,
            wiggle: request.wiggle,
            options: request.options,
            kappa_options: request.kappa_options,
        }
    }

    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        // Binomial location-scale requires an inverse link that supports the
        // joint link-wiggle refit; gate it before any fitting work (the pilot
        // runs only on the wiggle path).
        require_inverse_link_supports_joint_wiggle(
            &spec.link_kind,
            "binomial location-scale link wiggle",
        )?;
        fit_binomial_location_scale_terms(
            data,
            BinomialLocationScaleTermSpec {
                y: spec.y.clone(),
                weights: spec.weights.clone(),
                link_kind: spec.link_kind.clone(),
                thresholdspec: spec.thresholdspec.clone(),
                log_sigmaspec: spec.log_sigmaspec.clone(),
                threshold_offset: spec.threshold_offset.clone(),
                log_sigma_offset: spec.log_sigma_offset.clone(),
            },
            options,
            kappa_options,
        )
    }

    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, String> {
        let selected_wiggle_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
            pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        fit_binomial_location_scale_terms_with_selected_wiggle(
            data,
            spec,
            selected_wiggle_basis,
            options,
            kappa_options,
        )
    }

    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, String> {
        fit_binomial_location_scale_terms(data, spec, options, kappa_options)
    }

    fn assemble_plain(fit: BlockwiseTermFitResult) -> Self::Result {
        BinomialLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        }
    }

    fn assemble_with_wiggle(
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result {
        BinomialLocationScaleFitResult {
            fit,
            wiggle_knots: Some(wiggle_knots),
            wiggle_degree: Some(wiggle_degree),
            beta_link_wiggle,
        }
    }
}

/// Population standard deviation of a response column (divide by `n`, not
/// `n-1`).
///
/// This is the single response-standardization factor for the Gaussian
/// location-scale path, so the standardized fit is identical whether the
/// request arrives from the library (`fit_from_formula` →
/// `materialize_location_scale`), the FFI marshaller, or the CLI.
pub(crate) fn gaussian_response_sample_std(v: ArrayView1<'_, f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = v.iter().copied().sum::<f64>() / n;
    let var = v
        .iter()
        .copied()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1.0);
    var.max(0.0).sqrt()
}

/// Map a Gaussian location-scale fit fitted in *standardized* response units
/// (`y / response_scale`) back to **raw** response units, in place.
///
/// The internal fit solves with `y_internal = y / s` where `s = response_scale`.
/// Reconstructing raw outputs requires
///
///   μ_raw  = s · μ_internal           ⇒ scale every Location/Mean coefficient by `s`,
///   σ_raw  = s · σ_internal           ⇒ since σ = b + exp(η_σ), shifting the
///                                         log-σ **intercept** by `+ln(s)` turns
///                                         `b + exp(η)` into `b + s·exp(η)`; the
///                                         multiplicative `exp(η)` part is now
///                                         correct, but the **floor must also be
///                                         scaled** to `s·b` so the reconstructed
///                                         σ = s·b + exp(η_raw) = s·σ_internal is
///                                         response-scale-equivariant (#884). The
///                                         floor cannot ride the intercept shift
///                                         (it sits outside the exp), so consumers
///                                         reconstruct with floor `s·LOGB_SIGMA_FLOOR`
///                                         (see `GaussianLocationScalePredictor`).
///
/// The link-wiggle lives on the mean (identity) channel, so its knots and
/// coefficients scale by `s` exactly like the Location block. Doing the remap
/// here — once, inside the single Gaussian model entry point — makes every
/// caller (library `fit_from_formula`, the FFI marshaller, the CLI save path)
/// observe raw-unit coefficients with **no** additional per-call rescaling,
/// which is what keeps the σ-floor scale-relative (κ ≈ 1) without leaving the
/// reconstruction half-applied in any one path.
pub(crate) fn rescale_gaussian_location_scale_to_raw(
    result: &mut GaussianLocationScaleFitResult,
    response_scale: f64,
) {
    use gam_problem::BlockRole;

    let s = response_scale;
    assert!(
        s.is_finite() && s > 0.0,
        "Gaussian location-scale response rescale must be finite and positive, got {s}"
    );
    let ln_s = s.ln();
    // Intercept columns of the log-σ (Scale) design, expressed as offsets into
    // the Scale block's coefficient vector (the block β is laid out in noise
    // design column order). These are the only constant directions in η_σ
    // (smooths are sum-to-zero), so shifting them adds `ln(s)` to η_σ uniformly.
    let scale_intercept_range = result.fit.noise_design.intercept_range.clone();

    // Per-block coefficient surgery. `blocks` is authoritative for
    // `block_by_role` (predict, the FFI payload, and the reference tests all
    // read it), and the joint `beta` / `block_states` mirror it.
    let mut joint_offset = 0usize;
    for (block_idx, block) in result.fit.fit.blocks.iter_mut().enumerate() {
        let block_len = block.beta.len();
        match block.role {
            BlockRole::Mean | BlockRole::Location | BlockRole::LinkWiggle => {
                block.beta.mapv_inplace(|v| v * s);
                if result.fit.fit.beta.len() >= joint_offset + block_len {
                    for i in 0..block_len {
                        result.fit.fit.beta[joint_offset + i] *= s;
                    }
                }
                if let Some(state) = result.fit.fit.block_states.get_mut(block_idx) {
                    state.beta.mapv_inplace(|v| v * s);
                    state.eta.mapv_inplace(|v| v * s);
                }
            }
            BlockRole::Scale => {
                for col in scale_intercept_range.clone() {
                    if col < block.beta.len() {
                        block.beta[col] += ln_s;
                    }
                    let joint_col = joint_offset + col;
                    if joint_col < result.fit.fit.beta.len() {
                        result.fit.fit.beta[joint_col] += ln_s;
                    }
                    if let Some(state) = result.fit.fit.block_states.get_mut(block_idx)
                        && col < state.beta.len()
                    {
                        state.beta[col] += ln_s;
                    }
                }
                if let Some(state) = result.fit.fit.block_states.get_mut(block_idx) {
                    state.eta.mapv_inplace(|v| v + ln_s);
                }
            }
            BlockRole::Time | BlockRole::Threshold => {
                // Survival-only roles are never produced by the Gaussian
                // location-scale path; leave them untouched if ever present.
            }
        }
        joint_offset += block_len;
    }

    // The link-wiggle knots/coefficients live on the mean (identity) channel.
    if let Some(knots) = result.wiggle_knots.as_mut() {
        knots.mapv_inplace(|v| v * s);
    }
    if let Some(beta_w) = result.beta_link_wiggle.as_mut() {
        for coef in beta_w.iter_mut() {
            *coef *= s;
        }
    }

    // Conditional/corrected covariances were computed in standardized units.
    // Var(s·β_loc) = s²·Var(β_loc); the Scale block only had a constant added to
    // its intercept, which does not change its (co)variance. Cross terms between
    // a Location and the Scale block pick up one factor of `s`. This is exactly
    // a per-coefficient diagonal scaling D·Σ·D with D = s on Location/Mean/Wiggle
    // rows and D = 1 on Scale rows.
    let mut row_factors: Vec<f64> = Vec::new();
    for block in &result.fit.fit.blocks {
        let f = match block.role {
            BlockRole::Mean | BlockRole::Location | BlockRole::LinkWiggle => s,
            BlockRole::Scale | BlockRole::Time | BlockRole::Threshold => 1.0,
        };
        row_factors.extend(std::iter::repeat_n(f, block.beta.len()));
    }
    if let Some(cov) = result.fit.fit.covariance_conditional.as_mut() {
        rescale_covariance_coordinates(cov, &row_factors);
    }
    if let Some(cov) = result.fit.fit.covariance_corrected.as_mut() {
        rescale_covariance_coordinates(cov, &row_factors);
    }

    // Precision transforms contravariantly to covariance. If
    // β_raw = D β_internal + shift, then
    //
    //   H_raw = D^{-T} H_internal D^{-1}.
    //
    // The old remap transformed β and Cov(β) but left the saved penalized
    // Hessian in the internal coordinate system. Any saved-model operation
    // that solved that H against raw-coordinate design rows (notably ALO case
    // deletion) therefore mixed parameter systems. Transform every persisted
    // precision copy at the producer boundary so the saved model has one
    // coordinate convention.
    if let Some(geometry) = result.fit.fit.geometry.as_mut() {
        rescale_precision_coordinates(&mut geometry.penalized_hessian, &row_factors);
    }
    if let Some(inference) = result.fit.fit.inference.as_mut() {
        rescale_precision_coordinates(&mut inference.penalized_hessian, &row_factors);
    }

    // The residual-scale summary `standard_deviation` is a response-units
    // quantity; the internal fit reports it in standardized units, so map it
    // back. `max_abs_eta` is the mean-channel η magnitude (raw μ = s·μ_internal).
    result.fit.fit.standard_deviation *= s;
    result.fit.fit.max_abs_eta *= s;

    // Change-of-variables correction for the likelihood-scale summaries. The
    // internal fit maximizes the density of y_internal = y/s; the raw-response
    // density is p_raw(y) = p_internal(y/s)/s, so per observation
    // log p_raw = log p_internal − ln(s). The deviance (−2·loglik) and the
    // REML/LAML objective (which carries the data log-likelihood) shift
    // accordingly. This keeps reported log-likelihood / deviance / REML in raw
    // response units, matching what an un-standardized fit would report.
    // The number of observations is the fitted eta length for any parameter
    // block. Use the first block state instead of optional geometry so the
    // public objective fields stay in one unit system even when covariance or
    // ALO geometry was not retained.
    if let Some(n_obs) = result
        .fit
        .fit
        .block_states
        .first()
        .map(|state| state.eta.len() as f64)
        .filter(|&n| n > 0.0)
    {
        let ln_s = s.ln();
        result.fit.fit.log_likelihood -= n_obs * ln_s;
        result.fit.fit.deviance += 2.0 * n_obs * ln_s;
        result.fit.fit.reml_score += n_obs * ln_s;
        result.fit.fit.penalized_objective += n_obs * ln_s;
    }

    result.response_scale = s;
}

pub(crate) fn fit_gaussian_location_scale_model(
    mut request: GaussianLocationScaleFitRequest<'_>,
) -> Result<GaussianLocationScaleFitResult, String> {
    // Standardize the response so the fixed log-σ soft floor
    // `LOGB_SIGMA_FLOOR = 0.01` is scale-relative (≈ 1 % of the response
    // spread) rather than absolute. Without this the link σ = 0.01 + exp(η)
    // gives κ = dlogσ/dη = exp(η)/(0.01+exp(η)) < 1 whenever the raw σ is small,
    // and the scale-block Fisher information 2κ²a is strictly below gamlss's
    // floorless 2a, systematically over-smoothing the log-σ envelope
    // (#686 #688 #684 #685 #687). Fitting on y/s restores κ ≈ 1.
    let response_scale = gaussian_response_sample_std(request.spec.y.view()).max(1e-6);
    if response_scale != 1.0 {
        request.spec.y.mapv_inplace(|v| v / response_scale);
        // The mean (identity-link) offset rides in the same units as y; the
        // log-σ offset is on the log-scale axis and is unaffected by the
        // multiplicative response rescale.
        request
            .spec
            .mean_offset
            .mapv_inplace(|v| v / response_scale);
    }

    // Gaussian location-scale prediction has two coupled linear predictors, so
    // uncertainty on either response channel must be assembled from the joint
    // `(β_μ, β_logσ)` Laplace posterior covariance. Request it
    // unconditionally; otherwise predict-time delta-method SEs for the second
    // predictor can only fall back to scalar/block-local approximations.
    request.options.compute_covariance = true;

    let mut result =
        fit_location_scale_with_optional_wiggle::<GaussianLocationScaleWorkflow>(request)?;

    rescale_gaussian_location_scale_to_raw(&mut result, response_scale);
    Ok(result)
}

pub(crate) fn fit_dispersion_location_scale_model(
    request: DispersionLocationScaleFitRequest<'_>,
) -> Result<DispersionLocationScaleFitResult, String> {
    let kind = request.spec.kind;
    // The joint (mean + log-precision) posterior covariance / EDF is requested
    // unconditionally inside `fit_dispersion_glm_location_scale_terms`, which is
    // the shared entry for all four genuine-dispersion mean families (gam#1119),
    // so no per-request override is needed here.
    let fit = fit_dispersion_glm_location_scale_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )?;
    Ok(DispersionLocationScaleFitResult { fit, kind })
}

pub(crate) fn fit_binomial_location_scale_model(
    request: BinomialLocationScaleFitRequest<'_>,
) -> Result<BinomialLocationScaleFitResult, String> {
    fit_location_scale_with_optional_wiggle::<BinomialLocationScaleWorkflow>(request)
}

/// Penalized effective degrees of freedom for a survival transformation fit.
///
/// Uses exactly the mgcv definition `edf_total = p − Σ_k λ_k·tr(H⁻¹ S_k)`, where
/// `H` is the converged penalized Hessian `X'W_HX + S(λ) + ridge·I` (held in
/// `state.hessian`) and `S_k` is the penalty matrix of block `k` (without its
/// `λ_k` factor, which is applied here). The per-block edf is
/// `edf_k = block_cols_k − λ_k·tr(H⁻¹ S_k)`, clamped to `[0, block_cols_k]`.
///
/// Returned alongside the dense penalized Hessian so the caller can populate the
/// inference block (`edf_total`, `edf_by_block`, `penalized_hessian`). This is the
/// same trace formula `estimate.rs` uses for the standard GAM path; the survival
/// Each per-penalty EDF starts from the structural `rank(S_k)`, not the width
/// of the coefficient block that contains it. Several penalties may share a
/// coefficient block, so block width is not a valid per-penalty rank proxy.
/// The path runs its own `runworking_model_pirls` optimizer and therefore never
/// reached that block, leaving edf uncomputed (issue #565).
fn survival_transformation_edf(
    state: &gam_solve::pirls::WorkingState,
    penalty_blocks: &[PenaltyBlock],
) -> Result<(f64, Vec<f64>, Vec<f64>, Array2<f64>), String> {
    let h_dense = state.hessian.to_dense();
    let p = h_dense.nrows();
    let h_sym = gam_linalg::matrix::SymmetricMatrix::Dense(h_dense.clone());
    // EDF is an exact trace of the fitted (unperturbed) penalized Hessian.
    // Factoring a different, ridged matrix silently changes the estimand, so a
    // singular/indefinite fitted Hessian is an inference failure rather than a
    // license to manufacture a nearby covariance.
    let factor = h_sym.factorize().map_err(|error| {
        format!("survival edf: exact penalized-Hessian factorization failed: {error}")
    })?;
    let mut edf_by_block = vec![0.0_f64; penalty_blocks.len()];
    // Raw per-block penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk) (issue #1219).
    let mut penalty_block_trace = vec![0.0_f64; penalty_blocks.len()];
    let mut total_trace = 0.0_f64;
    for (kk, block) in penalty_blocks.iter().enumerate() {
        let block_cols = block.range.end - block.range.start;
        let penalty_rank = block_cols.checked_sub(block.nullspace_dim).ok_or_else(|| {
            format!(
                "survival edf: penalty {kk} nullity {} exceeds block width {block_cols}",
                block.nullspace_dim
            )
        })?;
        if block.lambda <= 0.0 || block_cols == 0 {
            edf_by_block[kk] = penalty_rank as f64;
            penalty_block_trace[kk] = 0.0;
            continue;
        }
        // RHS = S_k embedded into the full p×block_cols layout: column j holds
        // column j of S_k placed in the block rows. Solving H Z = RHS gives the
        // block columns of H⁻¹ S_full, whose block-diagonal entries sum to
        // tr(H⁻¹ S_k).
        let mut rhs = Array2::<f64>::zeros((p, block_cols));
        for c in 0..block_cols {
            for r in 0..block_cols {
                rhs[[block.range.start + r, c]] = block.matrix[[r, c]];
            }
        }
        let sol = factor
            .solvemulti(&rhs)
            .map_err(|e| format!("survival edf trace solve failed: {e}"))?;
        let mut trace = 0.0_f64;
        for j in 0..block_cols {
            trace += sol[[block.range.start + j, j]];
        }
        // Per-block penalty trace `λ_kk·tr(H⁻¹ S_kk)` is the penalized EDF of the
        // block, bounded by `[0, rank(S_k)]`. A ceiling-`λ` redundant block
        // (gam#1379) can otherwise overflow `λ·trace` to `+∞` on a ridge-
        // stabilized Hessian; clamp to the valid interval so the stored trace and
        // EDF stay finite. In-range traces pass through unchanged.
        let penalty_rank = penalty_rank as f64;
        let lam_trace = (block.lambda * trace).clamp(0.0, penalty_rank);
        total_trace += lam_trace;
        penalty_block_trace[kk] = lam_trace;
        edf_by_block[kk] = (penalty_rank - lam_trace).clamp(0.0, penalty_rank);
    }
    let edf_total = (p as f64 - total_trace).clamp(0.0, p as f64);
    if !edf_total.is_finite()
        || edf_by_block.iter().any(|v| !v.is_finite())
        || penalty_block_trace.iter().any(|v| !v.is_finite())
    {
        return Err("survival edf: non-finite effective degrees of freedom".to_string());
    }
    Ok((edf_total, edf_by_block, penalty_block_trace, h_dense))
}

/// REML/LAML smoothing-parameter selection for the single-cause transformation
/// survival baseline (issue #563).
///
/// The transformation path solves a constrained PIRLS (`γ ≥ 0` I-spline box) at
/// a fixed time-penalty `λ`, which oversmooths: with `λ` pinned at its seed the
/// monotone baseline collapses toward an affine log-cumulative-hazard and cannot
/// recover real curvature (e.g. Gompertz convexity). This routine wraps that
/// inner solve in a proper outer LAML optimization over `ρ = log λ` for the
/// `num_smoothing` time-penalty blocks (the trailing stabilization ridge is held
/// fixed), exactly as the standard GAM path and mgcv/scam do. The inner solve
/// still honors the structural box at every candidate `λ`, so the constrained
/// optimum stays valid; only the outer `λ` becomes data-adaptive.
///
/// `model` is the working model at the seed `λ`; it is cloned per candidate so
/// the proposal never corrupts the warm model. The returned vector has one
/// `λ_k` per penalty block (smoothing blocks at REML-selected values, the ridge
/// at its fixed seed). Returns `None` when there are no smoothing blocks to
/// select (e.g. the Weibull linear-time path), so the caller keeps the seed.
///
/// # Left-truncation guard on the time baseline (issue #1790/#1791)
///
/// The transformation-survival LAML `−½·log|H|` term uses the **observed**
/// information `H = X_exitᵀW_exit X_exit − X_entryᵀW_entry X_entry + (event/deriv)
/// + S(λ) + ridge` (`WorkingState::hessian_curvature = Observed`), not the
/// positive-definite **Fisher** curvature the standard GAM/location-scale REML
/// path uses (`HessianCurvatureKind::Fisher`). Under right censoring
/// (`entry == 0`) the `−X_entryᵀW_entry X_entry` term vanishes and `H` is PD, so
/// the LAML surface is well-behaved. Under genuine **left truncation** the
/// delayed-entry rows contribute a rank-heavy NEGATIVE `−X_entryᵀW_entry X_entry`
/// block that can drive the time-block of `H` indefinite / near-singular. The
/// spectral-regularized `log|H|` then keeps DECREASING as the time smoothing `λ`
/// shrinks (the near-null time direction is rewarded), so the outer BFGS rails
/// the time-block `λ` down to the lower bound. That under-smoothed, unpenalized
/// I-spline linear-trend column then inflates the baseline log-cumulative-hazard
/// into a huge, covariate-flat constant offset (`H ≈ const`, `S(t) ≡ 0`,
/// covariate dependence erased) — exactly the degenerate fit #1790/#1791 report.
/// The median-exit anchor (#1790) improves conditioning but does not remove the
/// observed-information indefiniteness, so the selection still rails.
///
/// Fix: under left truncation, floor the outer lower bound of the **time**
/// smoothing blocks at their seed `ρ` (`λ ≥ seed`, the documented CLI-equivalent
/// known-good conditioning) so the selector can only ever HOLD or INCREASE the
/// time penalty — never shrink it into the observed-information-driven degenerate
/// under-smoothed region. Over-smoothing stays fully available, and the
/// covariate smoothing blocks keep their full ±window so the covariate effect is
/// unconstrained. Right-censored data (`entry == 0`) has a PD `H`, so its bounds
/// are untouched and the fit is bit-for-bit preserved. Time blocks are those
/// whose penalty range lies in the leading `[0, time_block_cols)` columns.
fn optimize_survival_transformation_smoothing(
    model: &crate::survival::WorkingModelSurvival,
    penalty_blocks: &[PenaltyBlock],
    num_smoothing: usize,
    beta0: &Array1<f64>,
    structural_lower_bounds: Option<&Array1<f64>>,
    time_block_cols: usize,
    left_truncated: bool,
) -> Result<Option<Vec<f64>>, String> {
    use gam_problem::{Derivative, HessianValue, OuterEval};
    use gam_solve::rho_optimizer::OuterProblem;
    if num_smoothing == 0 {
        return Ok(None);
    }
    if num_smoothing > penalty_blocks.len() {
        return Err(format!(
            "survival transformation smoothing count {num_smoothing} exceeds penalty count {}",
            penalty_blocks.len()
        ));
    }
    // Full λ vector (smoothing blocks + fixed ridge), used to rebuild each
    // candidate model. The ridge entries (indices >= num_smoothing) are frozen.
    let seed_lambdas: Vec<f64> = penalty_blocks.iter().map(|b| b.lambda).collect();
    let seed_log_lambdas = seed_lambdas
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, value)| {
            gam_problem::checked_log_strength(value).map_err(|error| {
                format!("survival transformation seed lambda {coordinate}: {error}")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let seed_rho = Array1::from_vec(seed_log_lambdas[..num_smoothing].to_vec());

    // Memoize the most recent (ρ, cost, gradient) triple. The outer BFGS bridge
    // queries this objective through TWO separate closures — a value-only probe
    // (line search) and a value+gradient probe (accepted step) — and routinely
    // re-asks for the SAME ρ across them (a successful line-search point becomes
    // the next gradient evaluation). Each `eval_at` call re-runs the full
    // constrained inner PIRLS over all n rows, so without memoization every
    // accepted step pays for the identical inner solve twice. Caching one ρ (the
    // last evaluated) collapses that duplicate to a hash-equality check; the
    // returned cost/gradient are bit-identical to recomputing, so the BFGS path
    // and every asserted recovery bar are unchanged — only redundant inner
    // solves are removed. This mirrors the gamlss outer evaluator's `last_eval`
    // cache (`families::gamlss::builders`).
    let eval_cache: std::cell::RefCell<Option<(Array1<f64>, f64, Array1<f64>)>> =
        std::cell::RefCell::new(None);
    // Evaluate the LAML objective and ρ-gradient at a smoothing-ρ proposal:
    // set the smoothing λ, re-run the constrained inner PIRLS, evaluate the
    // unified survival LAML, and project the gradient onto the smoothing
    // coordinates (the trailing ridge gradient component is discarded since the
    // ridge is fixed).
    let eval_at = |rho_smooth: &Array1<f64>| -> Result<
        (f64, Array1<f64>),
        gam_solve::estimate::EstimationError,
    > {
        let physical_smoothing =
            gam_problem::checked_exp_log_strengths(rho_smooth.iter().copied())?;
        if let Some((cached_rho, cached_cost, cached_grad)) = eval_cache.borrow().as_ref()
            && cached_rho == rho_smooth
        {
            return Ok((*cached_cost, cached_grad.clone()));
        }
        let mut candidate = model.clone();
        let mut lambdas = seed_lambdas.clone();
        for k in 0..num_smoothing {
            lambdas[k] = physical_smoothing[k];
        }
        candidate
            .set_penalty_lambdas(&lambdas)
            .map_err(|error| {
                gam_solve::estimate::EstimationError::InvalidInput(error.to_string())
            })?;
        let opts = gam_solve::pirls::WorkingModelPirlsOptions {
            max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
            convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
            adaptive_kkt_tolerance: None,
            max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
            min_step_size: SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE,
            firth_bias_reduction: false,
            coefficient_lower_bounds: structural_lower_bounds.cloned(),
            linear_constraints: None,
            initial_lm_lambda: None,
            arrow_schur: None,
        };
        let summary = gam_solve::pirls::runworking_model_pirls(
            &mut candidate,
            gam_problem::Coefficients::new(beta0.clone()),
            &opts,
            |_| {},
        )?;
        // The envelope gradient exists only at a certified beta optimum. A
        // finite exhausted state is a checkpoint, not a derivative-bearing
        // objective sample; return typed inner non-convergence so the generic
        // outer bridge can retreat from this rho without fabricating a cost or
        // zero gradient.
        if !survival_pirls_status_is_certified(summary.status) {
            return Err(gam_solve::estimate::EstimationError::PirlsDidNotConverge {
                max_iterations: opts.max_iterations,
                last_change: summary.lastgradient_norm,
            });
        }
        let beta = summary.beta.as_ref().to_owned();
        let state = candidate.update_state(&beta).map_err(|error| {
            gam_solve::estimate::EstimationError::InvalidInput(format!(
                "survival smoothing inner state evaluation failed: {error}"
            ))
        })?;
        // Active-penalty ρ over ALL active blocks (smoothing + fixed ridge), in
        // block order, as the unified survival LAML evaluator requires. The
        // candidate's λ are exactly `lambdas` (smoothing entries from the
        // proposal, ridge entries frozen), so build ρ from that vector directly.
        let full_rho = Array1::from_vec(
            lambdas
                .iter()
                .copied()
                .enumerate()
                .map(|(coordinate, value)| {
                    gam_problem::checked_log_strength(value).map_err(|error| {
                        gam_solve::estimate::EstimationError::InvalidInput(format!(
                            "survival smoothing candidate lambda {coordinate}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let (cost, grad_full) = candidate
            .unified_lamlobjective_and_rhogradient(&beta, &state, &full_rho)
            .map_err(|error| {
                gam_solve::estimate::EstimationError::InvalidInput(format!(
                    "survival smoothing LAML evaluation failed: {error}"
                ))
            })?;
        // Project onto the smoothing coordinates. The active-block enumeration
        // lists the smoothing blocks first (they are constructed first and the
        // ridge is appended last), so the leading `num_smoothing` gradient
        // entries are exactly ∂LAML/∂ρ_smooth with the ridge held fixed.
        if grad_full.len() < num_smoothing || !cost.is_finite() {
            return Err(gam_solve::estimate::EstimationError::InvalidInput(
                "survival smoothing LAML cost was non-finite or gradient was too short"
                    .to_string(),
            ));
        }
        let grad = grad_full.slice(s![..num_smoothing]).to_owned();
        if grad.iter().any(|g| !g.is_finite()) {
            return Err(gam_solve::estimate::EstimationError::InvalidInput(
                "survival smoothing LAML gradient was non-finite".to_string(),
            ));
        }
        *eval_cache.borrow_mut() = Some((rho_smooth.to_owned(), cost, grad.clone()));
        Ok((cost, grad))
    };

    let mut lower = seed_rho.mapv(|v| v - 12.0);
    let upper = seed_rho.mapv(|v| v + 12.0);
    // Under left truncation the observed-information LAML `log|H|` is unreliable
    // BELOW the seed for the baseline time block (see the doc header): the
    // delayed-entry `−X_entryᵀW_entry X_entry` term can drive `H` indefinite so
    // shrinking the time `λ` spuriously lowers the LAML cost and rails the
    // selection into a degenerate, covariate-flat under-smoothed baseline
    // (#1790/#1791). Floor the TIME smoothing blocks' lower bound at their seed
    // `ρ` so the selector may only hold or over-smooth the baseline — never
    // under-smooth it into that region. Covariate smoothing blocks (whose penalty
    // ranges start at or beyond `time_block_cols`) keep their full window so the
    // covariate effect stays unconstrained. Right-censored data has a PD `H`, so
    // its bounds are left exactly as before.
    if left_truncated {
        for k in 0..num_smoothing {
            let is_time_block = penalty_blocks
                .get(k)
                .is_some_and(|block| block.range.start < time_block_cols);
            if is_time_block {
                lower[k] = seed_rho[k];
            }
        }
    }
    let problem = OuterProblem::new(num_smoothing)
        .with_gradient(Derivative::Analytic)
        .with_hessian(gam_problem::DeclaredHessianForm::Unavailable)
        .with_tolerance(1e-4)
        .with_max_iter(120)
        .with_bounds(lower, upper)
        .with_initial_rho(seed_rho.clone())
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let context =
        format!("survival transformation smoothing-parameter selection (dim={num_smoothing})");
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| eval_at(rho).map(|(c, _)| c),
        |_: &mut (), rho: &Array1<f64>| {
            let (cost, gradient) = eval_at(rho)?;
            Ok(OuterEval {
                cost,
                gradient,
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<
            fn(
                &mut (),
                &Array1<f64>,
            ) -> Result<gam_problem::EfsEval, gam_solve::estimate::EstimationError>,
        >,
    );
    // `OuterProblem::run` returns `Ok` only after its analytic projected-KKT
    // certificate accepts the selected rho. Exhaustion is
    // `EstimationError::RemlDidNotConverge`, whose rho checkpoint is preserved
    // in the error; a seed or best-so-far smoothing value is never promoted to
    // an estimator merely because a fixed-lambda inner solve was finite.
    let result = problem
        .run(&mut obj, &context)
        .map_err(|error| error.to_string())?;
    let selected_rho = result.rho;
    if selected_rho.len() != num_smoothing {
        return Err(format!(
            "survival transformation smoothing selector returned {} coordinates for \
             {num_smoothing} smoothing parameters; selected-rho checkpoint={:?}",
            selected_rho.len(),
            selected_rho.to_vec(),
        ));
    }
    let selected_lambdas = gam_problem::checked_exp_log_strengths(selected_rho.iter().copied())
        .map_err(|error| format!("survival transformation selected rho: {error}"))?;
    let mut lambdas = seed_lambdas;
    for (slot, lambda) in lambdas.iter_mut().zip(selected_lambdas) {
        *slot = lambda;
    }
    Ok(Some(lambdas))
}

fn survival_unified_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    summary: &gam_solve::pirls::WorkingModelPirlsResult,
    state: &gam_solve::pirls::WorkingState,
    penalty_blocks: &[PenaltyBlock],
) -> Result<UnifiedFitResult, String> {
    let log_lambdas = Array1::from_vec(
        lambdas
            .iter()
            .copied()
            .enumerate()
            .map(|(coordinate, value)| {
                gam_problem::checked_log_strength(value).map_err(|error| {
                    format!("survival fit lambda coordinate {coordinate}: {error}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    require_certified_survival_pirls(
        summary,
        "survival transformation fit assembly",
        log_lambdas.as_slice().unwrap_or(&[]),
        None,
    )?;
    let reml_score = state.penalized_objective();
    gam_solve::estimate::validate_all_finite("survival fit beta", beta.iter().copied())?;
    gam_solve::estimate::validate_all_finite("survival fit lambdas", lambdas.iter().copied())?;
    gam_solve::estimate::ensure_finite_scalar("survival fit log_likelihood", state.log_likelihood)?;
    gam_solve::estimate::ensure_finite_scalar("survival fit deviance", state.deviance)?;
    gam_solve::estimate::ensure_finite_scalar("survival fit penalty", state.penalty_term)?;
    gam_solve::estimate::ensure_finite_scalar("survival fit reml_score", reml_score)?;
    gam_solve::estimate::ensure_finite_scalar(
        "survival fit gradient_norm",
        summary.lastgradient_norm,
    )?;
    gam_solve::estimate::ensure_finite_scalar("survival fit max_abs_eta", summary.max_abs_eta)?;

    // Penalized effective degrees of freedom from the converged penalized
    // Hessian and penalty roots (issue #565). `lambdas` is built one entry per
    // penalty block, so `edf_by_block` aligns 1:1 with `lambdas` as the
    // `try_from_parts` invariant requires.
    let (edf_total, edf_by_block, penalty_block_trace, penalized_hessian) =
        survival_transformation_edf(state, penalty_blocks)?;
    assert_eq!(edf_by_block.len(), lambdas.len());
    assert_eq!(penalty_block_trace.len(), lambdas.len());

    let penalized_hessian = gam_problem::dispersion_cov::UnscaledPrecision::wrap(penalized_hessian);
    let inference = gam_solve::estimate::FitInference {
        edf_by_block: edf_by_block.clone(),
        penalty_block_trace,
        edf_total,
        smoothing_correction: None,
        smoothing_correction_method: None,
        penalized_hessian: penalized_hessian.clone(),
        reparam_qs: None,
        dispersion: gam_solve::estimate::Dispersion::UNIT,
        beta_covariance: None,
        beta_standard_errors: None,
        beta_covariance_corrected: None,
        beta_standard_errors_corrected: None,
        beta_covariance_frequentist: None,
        coefficient_influence: None,
        weighted_gram: None,
        bias_correction_beta: None,
        bias_correction_jacobian: None,
    };

    UnifiedFitResult::try_from_parts(gam_solve::estimate::UnifiedFitResultParts {
        blocks: vec![gam_solve::estimate::FittedBlock {
            beta: beta.clone(),
            role: gam_problem::BlockRole::Mean,
            edf: edf_total,
            lambdas: lambdas.clone(),
        }],
        log_lambdas,
        lambdas,
        likelihood_family: Some(LikelihoodSpec::royston_parmar()),
        likelihood_scale: gam_problem::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: gam_problem::LogLikelihoodNormalization::UserProvided,
        log_likelihood: state.log_likelihood,
        deviance: state.deviance,
        reml_score,
        stable_penalty_term: state.penalty_term,
        penalized_objective: reml_score,
        used_device: false,
        outer_iterations: summary.iterations,
        outer_converged: true,
        outer_gradient_norm: Some(summary.lastgradient_norm),
        standard_deviation: 1.0,
        covariance_conditional: None,
        covariance_corrected: None,
        inference: Some(inference),
        fitted_link: FittedLinkState::Standard(None),
        geometry: Some(gam_solve::estimate::FitGeometry {
            coefficient_gauge: gam_problem::gauge::Gauge::identity(&[beta.len()]),
            penalized_hessian,
            working: None,
        }),
        block_states: Vec::new(),
        pirls_status: summary.status,
        max_abs_eta: summary.max_abs_eta,
        constraint_kkt: None,
        artifacts: gam_solve::estimate::FitArtifacts {
            pirls: None,
            ..Default::default()
        },
        inner_cycles: 0,
    })
    .map_err(|err| err.to_string())
}

/// Replicate the single pooled-baseline coefficient seed (length `p`) across
/// every competing-risks cause.
///
/// `build_working_model` fits one shared single-hazard Royston-Parmar baseline
/// and returns a length-`p` coefficient seed (the Weibull scale/shape seed for
/// the parametric path). The cause-specific assembly in
/// `fit_cause_specific_survival_transformation_custom` stacks one coefficient
/// block per cause and slices `cause * p..(cause + 1) * p` out of its
/// `beta0_flat`, so it requires exactly `p * cause_count` initial coefficients.
/// Passing the un-replicated length-`p` seed straight through (the original
/// #378 fix did) aborts every `cause_count > 1` fit with a length-mismatch
/// `SchemaMismatch`. Seeding every cause from the same pooled baseline is the
/// correct start: each cause-specific block treats the competing causes as
/// censored, so they share the pooled baseline hazard until PIRLS specializes.
/// For `cause_count == 1` this is the identity.
pub(crate) fn replicate_pooled_baseline_seed_per_cause(
    pooled_seed: ArrayView1<'_, f64>,
    cause_count: usize,
) -> Array1<f64> {
    let p = pooled_seed.len();
    let mut beta0_flat = Array1::<f64>::zeros(p * cause_count);
    for cause in 0..cause_count {
        beta0_flat
            .slice_mut(s![cause * p..(cause + 1) * p])
            .assign(&pooled_seed);
    }
    beta0_flat
}

fn fit_cause_specific_survival_transformation_custom(
    spec: &SurvivalTransformationTermSpec,
    resolvedspec: TermCollectionSpec,
    baseline_cfg: crate::survival::construction::SurvivalBaselineConfig,
    prepared: PreparedSurvivalTimeStack,
    dense_cov_design: &Array2<f64>,
    penalty_blocks: Vec<PenaltyBlock>,
    beta0_flat: Array1<f64>,
    derivative_floor: f64,
    penalty_block_gamma_priors: &[(String, f64, f64)],
) -> Result<SurvivalTransformationFitResult, String> {
    let cause_count = crate::survival::cause_count_from_event_codes(spec.event_target.view())
        .into_workflow_result()?;
    if cause_count == 0 {
        return Err(WorkflowError::MissingDependency {
            reason: "cause-specific custom survival fit requires at least one cause".to_string(),
        }
        .into());
    }
    let n = spec.event_target.len();
    let p_time_total = prepared.time_design_exit.ncols();
    let p_cov = dense_cov_design.ncols();
    let p = p_time_total + p_cov;
    if beta0_flat.len() != p * cause_count {
        return Err(WorkflowError::SchemaMismatch {
            reason: format!(
                "cause-specific survival initial beta length mismatch: got {}, expected {}",
                beta0_flat.len(),
                p * cause_count
            ),
        }
        .into());
    }

    let dense_time_entry = prepared.time_design_entry.to_dense();
    let dense_time_exit = prepared.time_design_exit.to_dense();
    let dense_time_derivative = prepared.time_design_derivative_exit.to_dense();
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    if p_time_total > 0 {
        x_entry
            .slice_mut(s![.., ..p_time_total])
            .assign(&dense_time_entry);
        x_exit
            .slice_mut(s![.., ..p_time_total])
            .assign(&dense_time_exit);
        x_derivative
            .slice_mut(s![.., ..p_time_total])
            .assign(&dense_time_derivative);
    }
    if p_cov > 0 {
        x_entry
            .slice_mut(s![.., p_time_total..])
            .assign(dense_cov_design);
        x_exit
            .slice_mut(s![.., p_time_total..])
            .assign(dense_cov_design);
    }

    let mut family_blocks = Vec::with_capacity(cause_count);
    let mut block_specs = Vec::with_capacity(cause_count);
    for cause in 0..cause_count {
        let cause_code = (cause + 1) as u8;
        let event_target = spec
            .event_target
            .mapv(|observed| u8::from(observed == cause_code));
        family_blocks.push(crate::survival::CauseSpecificRoystonParmarBlock {
            age_entry: spec.age_entry.clone(),
            age_exit: spec.age_exit.clone(),
            event_target,
            sampleweight: spec.weights.clone(),
            x_entry: x_entry.clone(),
            x_exit: x_exit.clone(),
            x_derivative: x_derivative.clone(),
            offset_eta_entry: prepared.eta_offset_entry.clone() + &spec.covariate_offset,
            offset_eta_exit: prepared.eta_offset_exit.clone() + &spec.covariate_offset,
            offset_derivative_exit: prepared.derivative_offset_exit.clone(),
            derivative_floor,
        });

        let mut penalties = Vec::with_capacity(penalty_blocks.len());
        let mut nullspace_dims = Vec::with_capacity(penalty_blocks.len());
        let mut initial_log_lambdas = Array1::<f64>::zeros(penalty_blocks.len());
        for (penalty_idx, block) in penalty_blocks.iter().enumerate() {
            if block.range.end > p || block.range.start > block.range.end {
                return Err(WorkflowError::SchemaMismatch {
                    reason: "cause-specific survival penalty range is out of bounds".to_string(),
                }
                .into());
            }
            let block_dim = block.range.end - block.range.start;
            if block.matrix.nrows() != block_dim || block.matrix.ncols() != block_dim {
                return Err(WorkflowError::SchemaMismatch {
                    reason: format!(
                        "cause-specific survival penalty {penalty_idx} has shape {}x{} but range has width {block_dim}",
                        block.matrix.nrows(),
                        block.matrix.ncols()
                    ),
                }
                .into());
            }
            penalties.push(
                PenaltyMatrix::Blockwise {
                    local: block.matrix.clone(),
                    col_range: block.range.clone(),
                    total_dim: p,
                }
                .with_precision_label(format!(
                    "cause_specific_survival_cause_{}_penalty_{penalty_idx}",
                    cause + 1
                )),
            );
            nullspace_dims.push(block.nullspace_dim);
            initial_log_lambdas[penalty_idx] = gam_problem::checked_log_strength(block.lambda)
                .map_err(|error| {
                    format!("cause-specific survival penalty {penalty_idx} strength: {error}")
                })?;
        }
        let beta_start = beta0_flat.slice(s![cause * p..(cause + 1) * p]).to_owned();
        // Cause-specific blocks share the same time-basis design `x_exit`
        // (the same I-spline evaluated at the same observed event times), so
        // the joint design carries K block-pairs of (near-)identical
        // columns. The model is identifiable because the cause-specific
        // likelihood routes each cause to disjoint risk sets and
        // event-indicator masks
        // (`CauseSpecificRoystonParmarFamily::likelihood_blocks_uncoupled =
        // true`), but the identifiability audit operates on the unweighted
        // joint design. With every cause carrying the same `gauge_priority`
        // and no Jacobian callback to declare channel ownership, the audit's
        // `hard_alias_pair` gate fires on the strongest cross-block pair and
        // refuses the full-rank fit even when `joint_rank == p_total`.
        //
        // Mirror the multinomial-class block convention: assign descending
        // priorities (cause 0 highest, cause K-1 lowest) so the audit's
        // `pa != pb` filter on cross-block alias pairs always succeeds, and
        // attach an `AdditiveBlockJacobian` with `own_output = cause` so the
        // channel-aware audit treats each cause's contribution as occupying
        // its own output-channel rows. The Jacobian callback also takes the
        // canonical-gauge orthogonalisation pass out of play (the
        // family-owned-geometry guard defers when any block exposes a
        // callback), so the shared near-aliased column is not residualised
        // into a degenerate near-zero column behind the family's back; the
        // penalty + line search at solve time still resolves any residual
        // near-collinearity.
        let cause_priority =
            100u8.saturating_add(u8::try_from(cause_count - cause).unwrap_or(u8::MAX));
        let cause_jacobian = std::sync::Arc::new(AdditiveBlockJacobian {
            design: x_exit.clone(),
            own_output: cause,
            n_family_outputs: cause_count,
        });
        block_specs.push(ParameterBlockSpec {
            name: format!("time_cause_{}", cause + 1),
            design: gam_linalg::matrix::DesignMatrix::from(x_exit.clone()),
            offset: prepared.eta_offset_exit.clone() + &spec.covariate_offset,
            penalties,
            nullspace_dims,
            initial_log_lambdas,
            initial_beta: Some(beta_start),
            gauge_priority: cause_priority,
            jacobian_callback: Some(cause_jacobian),
            stacked_design: None,
            stacked_offset: None,
        });
    }

    let family = crate::survival::CauseSpecificRoystonParmarFamily::new(family_blocks)?;
    let fit_options = BlockwiseFitOptions {
        // Joint posterior prediction and CIF uncertainty consume the complete
        // cross-cause conditional covariance. Computing it here is part of the
        // fitted competing-risks model contract; reconstructing independent
        // per-cause approximations at prediction time would discard the
        // cross-cause blocks and misstate CIF uncertainty (#2298).
        compute_covariance: true,
        ..Default::default()
    };
    let rho_prior = cause_specific_survival_rho_prior(
        cause_count,
        penalty_blocks.len(),
        penalty_block_gamma_priors,
    )?;
    let mut fit = fit_custom_family_with_rho_prior(&family, &block_specs, &fit_options, rho_prior)
        .map_err(|err| format!("cause-specific survival custom-family fit failed: {err}"))?;
    fit.likelihood_family = Some(LikelihoodSpec::royston_parmar());
    let time_basis = crate::survival::construction::SavedSurvivalTimeBasis::from_build(
        &spec.time_build,
        spec.time_anchor,
    );
    // Recover the FITTED Weibull baseline from the converged linear-time
    // coefficients, mirroring the single-cause path (issues #689/#690). The
    // seed `baseline_cfg` carried only the pre-fit pooled scale/shape
    // (`shape = 1`, `scale = time-seed`), so any caller reading
    // `fit.baseline_cfg.scale/shape` to reconstruct `H = (t/scale)^shape`
    // would build the CIF from the uninitialized baseline and collapse it to
    // null. For Weibull-without-timewiggle the time basis is the 2-column
    // `[1, log t]` linear basis whose per-cause coefficients carry the full
    // log-cumulative-hazard. Because the basis is anchor-centered the constant
    // column (and thus `beta[0]`) is unidentified, so the fitted scale is the
    // identified anchor (`scale = anchor`, `shape = beta[1]`), not `beta[0]`
    // (issue #899). The shared `SurvivalBaselineConfig` holds a
    // single (scale, shape), so we report the first cause's fitted baseline as
    // the representative shared value — the same pooled-baseline convention the
    // seed used, but post-fit rather than uninitialized.
    let fitted_baseline_cfg = if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull
        && spec.timewiggle.is_none()
    {
        let first_block = fit.blocks.first().ok_or_else(|| {
            "cause-specific survival fit produced no coefficient blocks".to_string()
        })?;
        let time_beta = first_block
            .beta
            .slice(s![..spec.time_build.x_exit_time.ncols()])
            .to_owned();
        fitted_weibull_baseline_from_linear_time_beta(&time_beta, spec.time_anchor).ok_or_else(|| {
            "failed to recover fitted Weibull scale/shape from the cause-specific linear time coefficients"
                .to_string()
        })?
    } else {
        baseline_cfg
    };
    Ok(SurvivalTransformationFitResult {
        fit,
        resolvedspec,
        baseline_cfg: fitted_baseline_cfg,
        likelihood_mode: spec.likelihood_mode,
        time_basis,
        time_base_ncols: spec.time_build.x_exit_time.ncols(),
        baseline_timewiggle: prepared.timewiggle_block,
    })
}

fn cause_specific_survival_rho_prior(
    cause_count: usize,
    penalty_count: usize,
    penalty_block_gamma_priors: &[(String, f64, f64)],
) -> Result<gam_problem::RhoPrior, String> {
    if penalty_block_gamma_priors.is_empty() {
        return Ok(gam_problem::RhoPrior::Flat);
    }
    let mut keyed = BTreeMap::<String, (f64, f64)>::new();
    for (label, shape, rate) in penalty_block_gamma_priors {
        if keyed.insert(label.clone(), (*shape, *rate)).is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "duplicate Gamma precision hyperprior for penalty block label '{label}'"
                ),
            }
            .into());
        }
        if !shape.is_finite() || *shape <= 0.0 {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "Gamma precision hyperprior for penalty block '{label}' requires shape > 0, got {shape}"
                ),
            }
            .into());
        }
        if !rate.is_finite() || *rate < 0.0 {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "Gamma precision hyperprior for penalty block '{label}' requires rate >= 0, got {rate}"
                ),
            }
            .into());
        }
    }
    let mut consumed = Vec::<String>::new();
    let mut priors = Vec::<gam_problem::RhoPrior>::with_capacity(cause_count * penalty_count);
    for cause in 0..cause_count {
        for penalty_idx in 0..penalty_count {
            let label = format!(
                "cause_specific_survival_cause_{}_penalty_{penalty_idx}",
                cause + 1
            );
            if let Some((shape, rate)) = keyed.get(&label) {
                consumed.push(label);
                priors.push(gam_problem::RhoPrior::GammaPrecision {
                    shape: *shape,
                    rate: *rate,
                });
            } else {
                priors.push(gam_problem::RhoPrior::Flat);
            }
        }
    }
    let unknown = keyed
        .keys()
        .filter(|label| !consumed.iter().any(|known| known == *label))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        let available = (0..cause_count)
            .flat_map(|cause| {
                (0..penalty_count).map(move |idx| {
                    format!("cause_specific_survival_cause_{}_penalty_{idx}", cause + 1)
                })
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "unknown Gamma precision hyperprior penalty block label(s): {}; available labels: {available}",
                unknown.join(", ")
            ),
        }
        .into());
    }
    Ok(gam_problem::RhoPrior::Independent(priors))
}

fn hash_workflow_array_view(
    hasher: &mut gam_runtime::warm_start::Fingerprinter,
    array: ArrayView1<'_, f64>,
) {
    hasher.write_usize(array.len());
    for &value in array {
        hasher.write_f64(value);
    }
}

fn hash_workflow_u8_array(
    hasher: &mut gam_runtime::warm_start::Fingerprinter,
    array: ArrayView1<'_, u8>,
) {
    hasher.write_usize(array.len());
    for &value in array {
        hasher.write_usize(usize::from(value));
    }
}

fn hash_workflow_array2(
    hasher: &mut gam_runtime::warm_start::Fingerprinter,
    array: ArrayView2<'_, f64>,
) {
    hasher.write_usize(array.nrows());
    hasher.write_usize(array.ncols());
    for row in array.rows() {
        for &value in row {
            hasher.write_f64(value);
        }
    }
}

fn hash_workflow_design_matrix(
    hasher: &mut gam_runtime::warm_start::Fingerprinter,
    matrix: &gam_linalg::matrix::DesignMatrix,
) {
    let dense = matrix.to_dense();
    hash_workflow_array2(hasher, dense.view());
}

fn survival_transformation_log_lambdas(
    penalty_blocks: &[crate::survival::PenaltyBlock],
) -> Result<Vec<f64>, String> {
    penalty_blocks
        .iter()
        .enumerate()
        .map(|(coordinate, block)| {
            gam_problem::checked_log_strength(block.lambda)
                .map_err(|error| format!("survival transformation penalty {coordinate}: {error}"))
        })
        .collect()
}

fn persistent_survival_transformation_key(
    spec: &SurvivalTransformationTermSpec,
    baseline_cfg: &crate::survival::construction::SurvivalBaselineConfig,
    dense_cov_design: ArrayView2<'_, f64>,
    prepared: &PreparedSurvivalTimeStack,
    penalty_blocks: &[crate::survival::PenaltyBlock],
    opts: &gam_solve::pirls::WorkingModelPirlsOptions,
    n_cols: usize,
) -> String {
    let mut hasher = gam_runtime::warm_start::Fingerprinter::new();
    hasher.write_str("gamfit-persistent-survival-transformation-working-pirls");
    // Use the cache schema tag (NOT CARGO_PKG_VERSION) so routine
    // library version bumps don't invalidate users' on-disk warm-start
    // caches.
    hasher.write_str(&gam_solve::persistent_warm_start::cache_schema_tag());
    hasher.write_str(&format!("{:?}", spec.likelihood_mode));
    hasher.write_f64(spec.time_anchor);
    hasher.write_f64(spec.ridge_lambda);
    hasher.write_str(&format!("{:?}", baseline_cfg.target));
    for value in [
        baseline_cfg.scale,
        baseline_cfg.shape,
        baseline_cfg.rate,
        baseline_cfg.makeham,
    ] {
        hasher.write_bool(value.is_some());
        if let Some(value) = value {
            hasher.write_f64(value);
        }
    }
    hasher.write_str(&spec.time_build.basisname);
    hasher.write_usize(spec.time_build.x_entry_time.nrows());
    hasher.write_usize(spec.time_build.x_entry_time.ncols());
    hasher.write_usize(spec.time_build.x_exit_time.nrows());
    hasher.write_usize(spec.time_build.x_exit_time.ncols());
    hasher.write_usize(spec.time_build.x_derivative_time.nrows());
    hasher.write_usize(spec.time_build.x_derivative_time.ncols());
    hasher.write_bool(spec.time_build.degree.is_some());
    if let Some(degree) = spec.time_build.degree {
        hasher.write_usize(degree);
    }
    match spec.time_build.knots.as_ref() {
        Some(knots) => {
            hasher.write_bool(true);
            hasher.write_usize(knots.len());
            for &knot in knots {
                hasher.write_f64(knot);
            }
        }
        None => hasher.write_bool(false),
    }
    match spec.time_build.keep_cols.as_ref() {
        Some(cols) => {
            hasher.write_bool(true);
            hasher.write_usize(cols.len());
            for &col in cols {
                hasher.write_usize(col);
            }
        }
        None => hasher.write_bool(false),
    }
    hasher.write_bool(spec.time_build.smooth_lambda.is_some());
    if let Some(lambda) = spec.time_build.smooth_lambda {
        hasher.write_f64(lambda);
    }
    hasher.write_usize(n_cols);
    hash_workflow_array_view(&mut hasher, spec.age_entry.view());
    hash_workflow_array_view(&mut hasher, spec.age_exit.view());
    hash_workflow_u8_array(&mut hasher, spec.event_target.view());
    hash_workflow_array_view(&mut hasher, spec.weights.view());
    hash_workflow_array_view(&mut hasher, spec.covariate_offset.view());
    hash_workflow_array2(&mut hasher, dense_cov_design);
    hash_workflow_array_view(&mut hasher, prepared.eta_offset_entry.view());
    hash_workflow_array_view(&mut hasher, prepared.eta_offset_exit.view());
    hash_workflow_array_view(&mut hasher, prepared.derivative_offset_exit.view());
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_entry);
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_exit);
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_derivative_exit);
    hasher.write_usize(penalty_blocks.len());
    for block in penalty_blocks {
        hasher.write_f64(block.lambda);
        hasher.write_usize(block.range.start);
        hasher.write_usize(block.range.end);
        hasher.write_usize(block.nullspace_dim);
        hash_workflow_array2(&mut hasher, block.matrix.view());
    }
    hasher.write_usize(opts.max_iterations);
    hasher.write_f64(opts.convergence_tolerance);
    hasher.write_usize(opts.max_step_halving);
    hasher.write_f64(opts.min_step_size);
    hasher.write_bool(opts.firth_bias_reduction);
    hasher.write_bool(opts.coefficient_lower_bounds.is_some());
    if let Some(bounds) = opts.coefficient_lower_bounds.as_ref() {
        hash_workflow_array_view(&mut hasher, bounds.view());
    }
    hasher.write_bool(opts.linear_constraints.is_some());
    format!("surv-transform-{}", hasher.finish_hex())
}

fn load_survival_transformation_persistent_warm_start(
    key: &str,
    spec: &SurvivalTransformationTermSpec,
    n_cols: usize,
    rho: &[f64],
) -> Option<(Array1<f64>, Option<f64>)> {
    let record = gam_solve::persistent_warm_start::load_record(key)?;
    if !record.is_compatible(key, spec.age_entry.len(), n_cols)
        || record.rho.len() != rho.len()
        || !record
            .rho
            .iter()
            .zip(rho.iter())
            .all(|(cached, expected)| (*cached - *expected).abs() <= 1e-10)
    {
        return None;
    }
    log::info!("[warm-start-cache] restored survival transformation warm start key={key}");
    let lm_lambda = record
        .last_pirls_lm_lambda
        .filter(|value| value.is_finite() && *value > 0.0);
    Some((Array1::from_vec(record.beta), lm_lambda))
}

fn store_survival_transformation_persistent_warm_start(
    key: &str,
    spec: &SurvivalTransformationTermSpec,
    n_cols: usize,
    rho: Vec<f64>,
    beta: &Array1<f64>,
    summary: &gam_solve::pirls::WorkingModelPirlsResult,
) -> bool {
    if beta.len() != n_cols
        || beta.iter().any(|value| !value.is_finite())
        || rho.iter().any(|value| !value.is_finite())
    {
        return false;
    }
    let mut record = gam_solve::persistent_warm_start::PersistentWarmStartRecord::new(
        key.to_string(),
        spec.age_entry.len(),
        n_cols,
    );
    record.rho = rho;
    record.beta = beta.to_vec();
    record.last_inner_iters = summary.iterations;
    record.last_inner_converged = summary.status.is_converged();
    record.last_pirls_lm_lambda = (summary.final_lm_lambda.is_finite()
        && summary.final_lm_lambda > 0.0)
        .then_some(summary.final_lm_lambda);
    record.last_pirls_accept_rho = summary
        .final_accept_rho
        .filter(|value| value.is_finite() && *value >= 0.0);
    match gam_solve::persistent_warm_start::store_record(&record) {
        Ok(()) => {
            gam_solve::persistent_warm_start::load_record(&record.key).is_some_and(|stored| {
                stored.rho == record.rho
                    && stored.beta == record.beta
                    && stored.last_inner_iters == record.last_inner_iters
                    && stored.last_inner_converged == record.last_inner_converged
            })
        }
        Err(err) => {
            log::warn!(
                "[warm-start-cache] failed to persist survival transformation warm start: {err}"
            );
            false
        }
    }
}

pub(crate) fn fit_survival_transformation_model(
    request: SurvivalTransformationFitRequest<'_>,
) -> Result<SurvivalTransformationFitResult, String> {
    use crate::survival::{PenaltyBlock, PenaltyBlocks, SurvivalMonotonicityPenalty, SurvivalSpec};

    let SurvivalTransformationFitRequest {
        data,
        spec,
        cache_session: _cache_session,
    } = request;
    let mut baseline_cfg = spec.baseline_cfg.clone();
    let covariate_design =
        build_term_collection_design(data, &spec.covariate_spec).map_err(|err| err.to_string())?;
    let resolvedspec = crate::fit_orchestration::drivers::freeze_term_collection_from_design(
        &spec.covariate_spec,
        &covariate_design,
    )
    .map_err(|err| err.to_string())?;
    let dense_cov_design = covariate_design.design.to_dense();
    let p_cov = dense_cov_design.ncols();
    let cause_count = crate::survival::cause_count_from_event_codes(spec.event_target.view())
        .into_workflow_result()?;
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(spec.likelihood_mode);

    let build_working_model =
        |candidate: &crate::survival::construction::SurvivalBaselineConfig| {
            let prepared = prepare_survival_time_stack(
                &spec.age_entry,
                &spec.age_exit,
                candidate,
                spec.likelihood_mode,
                None,
                spec.time_anchor,
                exact_derivative_guard,
                &spec.time_build,
                spec.timewiggle.as_ref(),
                None,
            )?;
            let mut eta_offset_entry = prepared.eta_offset_entry.clone();
            let mut eta_offset_exit = prepared.eta_offset_exit.clone();
            eta_offset_entry += &spec.covariate_offset;
            eta_offset_exit += &spec.covariate_offset;
            // Covariates enter both cumulative-hazard evaluations and are
            // constant with respect to survival time. Their fixed affine lift
            // therefore belongs in entry and exit, but not the time derivative.
            eta_offset_entry += &covariate_design.affine_offset;
            eta_offset_exit += &covariate_design.affine_offset;
            let p_time_total = prepared.time_design_exit.ncols();
            let p = p_time_total + p_cov;
            let mut penalty_blocks = Vec::<PenaltyBlock>::new();
            for (idx, penalty) in prepared.time_penalties.iter().enumerate() {
                if penalty.nrows() == p_time_total && penalty.ncols() == p_time_total {
                    penalty_blocks.push(PenaltyBlock {
                        matrix: penalty.clone(),
                        lambda: spec.time_build.smooth_lambda.unwrap_or(1e-2),
                        range: 0..p_time_total,
                        nullspace_dim: prepared.time_nullspace_dims.get(idx).copied().unwrap_or(0),
                    });
                }
            }
            // Covariate-smooth penalties (e.g. `s(x)`, `s(group, bs="re")`
            // frailty) live in the covariate term-collection design; the survival
            // transformation fit stacks the covariate columns at
            // `p_time_total..p`, so each covariate penalty's local block maps to
            // the joint range `p_time_total + col_range`. Penalizing them here
            // (rather than leaving them to the tiny stabilization ridge) is what
            // lets the frailty / covariate smooths shrink — and they are added
            // BEFORE the ridge so they too become REML-selected smoothing blocks
            // (issues #563/#565). Only zero-prior-mean blocks are admissible as a
            // plain quadratic `λ βᵀSβ`; a non-zero centering would need an offset
            // the survival PenaltyBlock does not model, so such blocks are left to
            // the ridge rather than mis-applied.
            for (penalty_idx, cov_penalty) in covariate_design.penalties.iter().enumerate() {
                let cr = &cov_penalty.col_range;
                let block_dim = cr.end - cr.start;
                let matches_dims = cov_penalty.local.nrows() == block_dim
                    && cov_penalty.local.ncols() == block_dim;
                let zero_prior = matches!(
                    cov_penalty.prior_mean,
                    gam_problem::CoefficientPriorMean::Zero
                );
                if block_dim > 0 && matches_dims && zero_prior && cr.end <= p_cov {
                    penalty_blocks.push(PenaltyBlock {
                        matrix: cov_penalty.local.clone(),
                        lambda: 1e-2,
                        range: (p_time_total + cr.start)..(p_time_total + cr.end),
                        nullspace_dim: covariate_design
                            .nullspace_dims
                            .get(penalty_idx)
                            .copied()
                            .unwrap_or(0),
                    });
                }
            }
            // The smoothing blocks are exactly those pushed above (time +
            // covariate penalties); any ridge appended below is a FIXED
            // stabilization, not a REML-selected smoothing parameter, so the
            // count of smoothing blocks is recorded before the ridge is added
            // (issue #563).
            let num_smoothing_blocks = penalty_blocks.len();
            // The Weibull linear time basis is `[1, log t]`. In the SINGLE-cause
            // dedicated-PIRLS path the constant column carries the baseline level
            // (β0 = −shape·ln(scale)), so the stabilization ridge excludes it
            // (`ridge_range_start = 1`) to avoid shrinking the scale. But that
            // same basis is ANCHOR-CENTERED, which makes the constant column
            // identically zero — a dead, gradient-free direction. The single-cause
            // PIRLS leaves β0 at its seed and tolerates the zero column; the
            // CAUSE-SPECIFIC competing-risks path instead routes through the
            // custom-family blockwise Newton solver, whose per-block Hessian then
            // has an exactly-zero row/column on β0. Excluding it from the ridge
            // leaves that Hessian singular, so the inner Newton step degenerates
            // and NO coefficient moves off its seed (#1590). For the cause-specific
            // path penalise the full width so the dead β0 is pinned (→ 0, which the
            // downstream baseline recovery already ignores: scale = anchor, shape =
            // β1), leaving the live time/covariate directions well-conditioned.
            let ridge_range_start = if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull
                && spec.time_build.basisname == "linear"
                && spec.timewiggle.is_none()
                && cause_count <= 1
            {
                1
            } else {
                0
            };
            if spec.ridge_lambda > 0.0 && p > ridge_range_start {
                let dim = p - ridge_range_start;
                let mut ridge = Array2::<f64>::zeros((dim, dim));
                for d in 0..dim {
                    ridge[[d, d]] = 1.0;
                }
                penalty_blocks.push(PenaltyBlock {
                    matrix: ridge,
                    lambda: spec.ridge_lambda,
                    range: ridge_range_start..p,
                    nullspace_dim: 0,
                });
            }
            let dense_time_entry = prepared.time_design_entry.to_dense();
            let dense_time_exit = prepared.time_design_exit.to_dense();
            let dense_time_derivative = prepared.time_design_derivative_exit.to_dense();
            let event_competing = Array1::<u8>::zeros(spec.event_target.len());
            // `spec.event_target` carries *cause labels* (0 = censored, k = cause k).
            // The shared baseline working model is a single-hazard Royston-Parmar
            // model whose binary `event_target` contract is {0, 1}. For the pooled
            // baseline that seeds scale/shape across all causes, every observed event
            // (any cause) informs the shared baseline hazard, so collapse cause labels
            // to a {0, 1} any-event indicator. The per-cause specialization (event for
            // cause k vs. competing-cause-as-censored) happens later when the
            // cause-specific blocks are built.
            let baseline_event_indicator = spec.event_target.mapv(|label| u8::from(label > 0));
            let mut model =
                crate::survival::royston_parmar::working_model_from_time_covariateshared(
                    PenaltyBlocks::new(penalty_blocks.clone()),
                    SurvivalMonotonicityPenalty { tolerance: 0.0 },
                    SurvivalSpec::Net,
                    crate::survival::royston_parmar::RoystonParmarSharedTimeCovariateInputs {
                        age_entry: spec.age_entry.view(),
                        age_exit: spec.age_exit.view(),
                        event_target: baseline_event_indicator.view(),
                        event_competing: event_competing.view(),
                        weights: spec.weights.view(),
                        time_entry: dense_time_entry.view(),
                        time_exit: dense_time_exit.view(),
                        time_derivative: dense_time_derivative.view(),
                        covariates: dense_cov_design.view(),
                        monotonicity_constraint_rows: None,
                        monotonicity_constraint_offsets: None,
                        eta_offset_entry: Some(eta_offset_entry.view()),
                        eta_offset_exit: Some(eta_offset_exit.view()),
                        derivative_offset_exit: Some(prepared.derivative_offset_exit.view()),
                    },
                )
                .map_err(|err| format!("failed to construct survival model: {err}"))?;
            if spec.likelihood_mode != SurvivalLikelihoodMode::Weibull {
                model
                    .set_structural_monotonicity(true, p_time_total)
                    .map_err(|err| format!("failed to enable structural monotonicity: {err}"))?;
            }
            let mut beta0 = Array1::<f64>::zeros(p);
            if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull && spec.timewiggle.is_none()
            {
                let (scale, shape) = spec
                    .weibull_seed
                    .ok_or_else(|| "weibull survival fit missing scale/shape seed".to_string())?;
                if p_time_total < 2 {
                    return Err(format!(
                        "weibull built-in time basis has {p_time_total} columns but needs 2 to seed scale/shape"
                    ));
                }
                beta0[0] = -shape * scale.ln();
                beta0[1] = shape;
            }
            let structural_lower_bounds =
                if spec.likelihood_mode != SurvivalLikelihoodMode::Weibull && p_time_total > 0 {
                    let mut lb = Array1::from_elem(p, f64::NEG_INFINITY);
                    for j in 0..p_time_total {
                        lb[j] = 0.0;
                        beta0[j] = 1e-4;
                    }
                    Some(lb)
                } else {
                    None
                };
            Ok::<_, String>((
                prepared,
                penalty_blocks,
                beta0,
                structural_lower_bounds,
                model,
                num_smoothing_blocks,
            ))
        };

    if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        // Analytic-gradient BFGS over the baseline shape params (weibull
        // scale/shape; gompertz rate/shape; gompertz-makeham rate/shape/makeham).
        //
        // The cost optimized here is the *profile penalized NLL*
        //   V(θ) = 0.5·deviance(β̂(θ); o(θ)) + 0.5·β̂ᵀSβ̂   (= survival_working_reml_score),
        // where the baseline θ enters the transformation working model only
        // through the three additive time-block offsets
        //   o_E(θ) = η_target(age_entry), o_X(θ) = η_target(age_exit),
        //   o_D(θ) = d/dt η_target |_{age_exit}.
        // β̂(θ) is the (constrained) PIRLS optimum, so ∂V/∂β = 0 there and by the
        // envelope theorem dV/dθ_k = ∂V/∂θ_k|_{β=β̂} — the explicit partial holding
        // β̂ fixed. The active-set inequality constraints {β_j ≥ 0} carry no
        // θ-dependence, so the constrained envelope identity is unchanged. That
        // explicit partial is exactly the residual×offset-partial contraction
        //   dV/dθ_k = Σ_i r^X_i ∂o_X_i/∂θ_k + r^E_i ∂o_E_i/∂θ_k + r^D_i ∂o_D_i/∂θ_k,
        // with r^* = WorkingModelSurvival::offset_channel_residuals(β̂) and the
        // η-channel offset partials supplied by baseline_offset_theta_partials
        // (contracted by baseline_chain_rule_gradient). See the derivation header
        // on baseline_chain_rule_gradient. BFGS over this exact gradient converges
        // in ≲10 outer evaluations on the 2–3 dim surface.
        baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "workflow survival transformation baseline",
            |candidate| {
                let (_, _, beta0, structural_lower_bounds, mut model, _) =
                    build_working_model(candidate)?;
                let opts = gam_solve::pirls::WorkingModelPirlsOptions {
                    max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
                    convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
                    adaptive_kkt_tolerance: None,
                    max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
                    min_step_size: SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE,
                    firth_bias_reduction: false,
                    coefficient_lower_bounds: structural_lower_bounds,
                    linear_constraints: None,
                    initial_lm_lambda: None,
                    arrow_schur: None,
                };
                let parameter_checkpoint = survival_baseline_parameter_checkpoint(candidate)?;
                let summary = gam_solve::pirls::runworking_model_pirls(
                    &mut model,
                    gam_problem::Coefficients::new(beta0),
                    &opts,
                    |_| {},
                )
                .map_err(|error| {
                    format!(
                        "survival baseline PIRLS failed at parameter_checkpoint=\
                         {parameter_checkpoint:?}: {error}; no fit was minted"
                    )
                })?;
                require_certified_survival_pirls(
                    &summary,
                    "survival transformation baseline profile",
                    &parameter_checkpoint,
                    None,
                )?;
                let beta = summary.beta.as_ref().to_owned();
                let state = model.update_state(&beta).map_err(|err| {
                    format!("failed to evaluate survival baseline candidate: {err}")
                })?;
                let cost = state.penalized_objective();
                let residuals = model.offset_channel_residuals(&beta).map_err(|err| {
                    format!("failed to form survival baseline offset residuals: {err}")
                })?;
                let gradient = baseline_chain_rule_gradient(
                    spec.age_entry.view(),
                    spec.age_exit.view(),
                    // RP transformation has no interval upper-bound channel;
                    // `residuals.right` is all-zero so `age_exit` is an unconsulted
                    // placeholder for `age_right`.
                    spec.age_exit.view(),
                    candidate,
                    &residuals,
                )?
                .ok_or_else(|| {
                    "workflow survival transformation baseline unexpectedly has no theta gradient"
                        .to_string()
                })?;
                Ok((cost, gradient))
            },
        )?;
    }

    let (
        prepared,
        mut penalty_blocks,
        beta0,
        structural_lower_bounds,
        mut model,
        num_smoothing_blocks,
    ) = build_working_model(&baseline_cfg)?;
    if cause_count > 1 || !spec.penalty_block_gamma_priors.is_empty() {
        let beta0_flat = replicate_pooled_baseline_seed_per_cause(beta0.view(), cause_count);
        return fit_cause_specific_survival_transformation_custom(
            &spec,
            resolvedspec,
            baseline_cfg,
            prepared,
            &dense_cov_design,
            penalty_blocks,
            beta0_flat,
            exact_derivative_guard,
            &spec.penalty_block_gamma_priors,
        );
    }
    // REML/LAML-select the time-smoothing λ (issue #563). With λ pinned at its
    // seed the monotone I-spline baseline oversmooths toward an affine
    // log-cumulative-hazard; selecting λ from the survival LAML lets it recover
    // real curvature. The inner solve keeps the structural γ ≥ 0 box at every
    // candidate, so the constrained optimum stays valid; the fixed stabilization
    // ridge is held at its seed. The selected λ is written back into both the
    // working model and `penalty_blocks` so the final fit, edf, and warm-start
    // cache all use the data-adaptive value.
    // Left-truncation status drives the observed-information LAML guard on the
    // baseline time smoothing block (#1790/#1791): genuine delayed entry
    // (`entry > ENTRY_AT_ORIGIN_THRESHOLD`) makes the observed-information `H`
    // indefinite below the seed λ, so the time block's outer lower bound is
    // floored at its seed to prevent the degenerate under-smoothed baseline.
    let is_left_truncated = spec
        .age_entry
        .iter()
        .any(|&t| t > crate::survival::ENTRY_AT_ORIGIN_THRESHOLD);
    let p_time_total = prepared.time_design_exit.ncols();
    if let Some(selected_lambdas) = optimize_survival_transformation_smoothing(
        &model,
        &penalty_blocks,
        num_smoothing_blocks,
        &beta0,
        structural_lower_bounds.as_ref(),
        p_time_total,
        is_left_truncated,
    )? {
        model
            .set_penalty_lambdas(&selected_lambdas)
            .map_err(|e| e.to_string())?;
        for (block, &lam) in penalty_blocks.iter_mut().zip(selected_lambdas.iter()) {
            block.lambda = lam;
        }
    }
    let opts = gam_solve::pirls::WorkingModelPirlsOptions {
        max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
        convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
        adaptive_kkt_tolerance: None,
        max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
        min_step_size: SURVIVAL_TRANSFORMATION_PIRLS_MIN_STEP_SIZE,
        firth_bias_reduction: false,
        coefficient_lower_bounds: structural_lower_bounds,
        linear_constraints: None,
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let rho_for_cache = survival_transformation_log_lambdas(&penalty_blocks)?;
    let expected_beta_len = beta0.len();
    let persistent_warm_start_key = persistent_survival_transformation_key(
        &spec,
        &baseline_cfg,
        dense_cov_design.view(),
        &prepared,
        &penalty_blocks,
        &opts,
        expected_beta_len,
    );
    let mut opts = opts;
    let beta_start = match load_survival_transformation_persistent_warm_start(
        &persistent_warm_start_key,
        &spec,
        expected_beta_len,
        &rho_for_cache,
    ) {
        Some((beta, lm_lambda)) => {
            opts.initial_lm_lambda = lm_lambda;
            beta
        }
        None => beta0,
    };
    let summary = gam_solve::pirls::runworking_model_pirls(
        &mut model,
        gam_problem::Coefficients::new(beta_start),
        &opts,
        |_| {},
    )
    .map_err(|error| {
        format!(
            "survival transformation final fixed-lambda PIRLS failed at \
             parameter_checkpoint={rho_for_cache:?} (warm_start_key=\
             {persistent_warm_start_key}): {error}; no fit was minted"
        )
    })?;
    let beta = summary.beta.as_ref().to_owned();
    // Persist every finite accepted iterate before enforcing the certificate:
    // an exhausted solve is resumable work, but it is never a fit. The record's
    // `last_inner_converged` bit distinguishes a final mode from a checkpoint.
    let checkpoint_persisted = store_survival_transformation_persistent_warm_start(
        &persistent_warm_start_key,
        &spec,
        expected_beta_len,
        rho_for_cache.clone(),
        &beta,
        &summary,
    );
    require_certified_survival_pirls(
        &summary,
        "survival transformation final fixed-lambda PIRLS",
        &rho_for_cache,
        checkpoint_persisted.then_some(persistent_warm_start_key.as_str()),
    )?;
    let state = model
        .update_state(&beta)
        .map_err(|err| format!("failed to evaluate survival optimum: {err}"))?;
    let lambdas = Array1::from_iter(penalty_blocks.iter().map(|block| block.lambda));
    let fitted_baseline_cfg =
        if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull && spec.timewiggle.is_none() {
            let time_beta = beta
                .slice(s![..spec.time_build.x_exit_time.ncols()])
                .to_owned();
            fitted_weibull_baseline_from_linear_time_beta(&time_beta, spec.time_anchor).ok_or_else(
                || {
                    "failed to recover fitted Weibull scale/shape from the linear time coefficients"
                        .to_string()
                },
            )?
        } else {
            baseline_cfg
        };
    let fit = survival_unified_fit_result(beta, lambdas, &summary, &state, &penalty_blocks)?;

    let time_base_ncols = spec.time_build.x_exit_time.ncols();
    let time_basis = crate::survival::construction::SavedSurvivalTimeBasis::from_build(
        &spec.time_build,
        spec.time_anchor,
    );
    Ok(SurvivalTransformationFitResult {
        fit,
        resolvedspec,
        baseline_cfg: fitted_baseline_cfg,
        likelihood_mode: spec.likelihood_mode,
        time_basis,
        time_base_ncols,
        baseline_timewiggle: prepared.timewiggle_block,
    })
}

pub(crate) fn fit_survival_location_scale_model(
    request: SurvivalLocationScaleFitRequest<'_>,
) -> Result<SurvivalLocationScaleFitResult, String> {
    // Profile one coherent survival subproblem at a fixed inverse-link state:
    // select/apply the link-wiggle basis for that state, then solve the full
    // penalized location-scale fit on the resulting model.
    fn profile_survival_location_scale(
        data: ArrayView2<'_, f64>,
        spec: SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let mut wiggle_knots = None;
        let mut wiggle_degree = None;
        let inverse_link = spec.inverse_link.clone();

        let fit = if let Some(wiggle) = wiggle {
            require_inverse_link_supports_joint_wiggle(&inverse_link, "survival link wiggle")?;
            let mut pilot_spec = spec.clone();
            pilot_spec.linkwiggle_block = None;
            let pilot = fit_survival_location_scale_terms(data, pilot_spec, kappa_options)?;
            let selected_wiggle_basis = select_survival_link_wiggle_basis_from_pilot(
                &pilot,
                &WiggleBlockConfig {
                    degree: wiggle.degree,
                    num_internal_knots: wiggle.num_internal_knots,
                    penalty_order: 2,
                    double_penalty: wiggle.double_penalty,
                },
                &wiggle.penalty_orders,
            )?;
            wiggle_knots = Some(selected_wiggle_basis.knots.clone());
            wiggle_degree = Some(selected_wiggle_basis.degree);
            fit_survival_location_scale_terms_with_selected_wiggle(
                data,
                spec,
                selected_wiggle_basis,
                kappa_options,
            )?
        } else {
            fit_survival_location_scale_terms(data, spec, kappa_options)?
        };

        Ok(SurvivalLocationScaleProfile {
            fit,
            inverse_link,
            wiggle_knots,
            wiggle_degree,
            inverse_link_outer: None,
        })
    }

    /// Profile the survival location-scale fit at a fixed inverse-link state:
    /// substitutes `inverse_link` into the spec and runs the full penalized fit.
    fn profile_survival_location_scale_with_inverse_link(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        inverse_link: InverseLink,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let mut spec_at_link = spec.clone();
        spec_at_link.inverse_link = inverse_link;
        profile_survival_location_scale(data, spec_at_link, wiggle, kappa_options)
    }

    fn optimize_survival_inverse_link_profile(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        // Analytic-gradient BFGS over the inverse-link parameters θ_link
        // (SAS ε/log_δ, BetaLogistic ε/log_δ, Mixture ρ). The link enters the
        // location-scale likelihood through the standardized residual it maps,
        // and the EXACT data-fit θ-gradient
        //   ∂(−ℓ)/∂θ_link = −Σ_i w_i·( event_mix(d, ∂logφ(u1), ∂logS(u1)) − ∂logS(u0) )
        // is formed analytically from the inverse-link param partials
        // (`SurvivalLocationScaleFamily::link_param_data_fit_gradient`, carried
        // on the fit result as `link_param_data_fit_gradient`). We optimize the
        // *profile penalized NLL* `−ℓ + ½βᵀSβ` — not the LAML `reml_score` whose
        // ½log|H+S_λ| term has its own θ_link dependence through H(β̂,θ) — so the
        // envelope-theorem gradient matches the cost surface. Each profile picks
        // ρ on the full REML surface; the runner-installed terminal profile is
        // retained directly once the link optimum certifies.
        fn optimize_link_parameters(
            data: ArrayView2<'_, f64>,
            spec: &SurvivalLocationScaleTermSpec,
            kappa_options: &SpatialLengthScaleOptimizationOptions,
            init: Array1<f64>,
            name: &str,
            wiggle_cfg: Option<LinkWiggleConfig>,
            make_link: impl Fn(&Array1<f64>) -> Result<InverseLink, String> + Clone,
        ) -> Result<SurvivalLocationScaleProfile, String> {
            use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
            use gam_solve::rho_optimizer::OuterProblem;
            let dim = init.len();
            // Box bounds keep line-search probes inside a physically admissible
            // region (|ε|, |log δ| ≤ 6 gives the SAS link a finite range on both
            // tails; mixture logits stay in a numerically sane band). With an
            // analytic gradient and no declared Hessian the planner routes this
            // to BFGS.
            let lower = init.mapv(|v| v - 6.0);
            let upper = init.mapv(|v| v + 6.0);
            let problem = OuterProblem::new(dim)
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Unavailable)
                .with_tolerance(1e-4)
                .with_max_iter(240)
                .with_bounds(lower, upper)
                .with_initial_rho(init.clone())
                .with_seed_config(gam_problem::SeedConfig {
                    max_seeds: 1,
                    seed_budget: 1,
                    num_auxiliary_trailing: dim,
                    ..Default::default()
                });
            let context = format!("survival inverse-link optimization ({name}, dim={dim})");
            // The objective returns the profile-NLL cost and the exact analytic
            // θ_link-gradient from the converged fit at this candidate link.
            let eval_link = move |theta: &Array1<f64>| -> Result<
                ProfiledOuterPayload<SurvivalLocationScaleProfile>,
                String,
            > {
                let link = make_link(theta)?;
                let profile = profile_survival_location_scale_with_inverse_link(
                    data,
                    spec,
                    link,
                    wiggle_cfg.clone(),
                    kappa_options,
                )?;
                let cost = survival_inverse_link_profile_objective(
                    &profile,
                    &format!("survival inverse-link ({name})"),
                )?;
                let gradient = profile
                    .fit
                    .link_param_data_fit_gradient
                    .clone()
                    .ok_or_else(|| {
                        format!(
                            "survival inverse-link ({name}): fit reported no link-parameter \
                             data-fit gradient"
                        )
                    })?;
                if gradient.len() != theta.len() {
                    return Err(format!(
                        "survival inverse-link ({name}): gradient dim {} != theta dim {}",
                        gradient.len(),
                        theta.len()
                    ));
                }
                Ok(ProfiledOuterPayload {
                    theta: theta.clone(),
                    objective: cost,
                    gradient,
                    value: profile,
                })
            };
            let cost_eval = eval_link.clone();
            let cost_fn =
                move |selected: &mut Option<ProfiledOuterPayload<SurvivalLocationScaleProfile>>,
                      theta: &Array1<f64>| {
                    let payload = cost_eval(theta)
                        .map_err(gam_solve::estimate::EstimationError::InvalidInput)?;
                    let cost = payload.objective;
                    *selected = Some(payload);
                    Ok(cost)
                };
            let eval_fn =
                move |selected: &mut Option<ProfiledOuterPayload<SurvivalLocationScaleProfile>>,
                      theta: &Array1<f64>| {
                    let payload = eval_link(theta)
                        .map_err(gam_solve::estimate::EstimationError::InvalidInput)?;
                    let evaluation = OuterEval {
                        cost: payload.objective,
                        gradient: payload.gradient.clone(),
                        hessian: HessianValue::Unavailable,
                        inner_beta_hint: None,
                    };
                    *selected = Some(payload);
                    Ok(evaluation)
                };
            let mut obj = problem.build_objective(
                None::<ProfiledOuterPayload<SurvivalLocationScaleProfile>>,
                cost_fn,
                eval_fn,
                None::<fn(&mut Option<ProfiledOuterPayload<SurvivalLocationScaleProfile>>)>,
                None::<
                    fn(
                        &mut Option<ProfiledOuterPayload<SurvivalLocationScaleProfile>>,
                        &Array1<f64>,
                    )
                        -> Result<gam_problem::EfsEval, gam_solve::estimate::EstimationError>,
                >,
            );
            let certified_outer = problem
                .run_certified(&mut obj, &context)
                .map_err(|err| format!("{context} failed: {err}"))?;
            let selected = consume_certified_profiled_outer_payload(
                obj.state.take(),
                &certified_outer,
                &context,
            )?;
            let replayed_objective =
                survival_inverse_link_profile_objective(&selected.value, &context)?;
            if replayed_objective.to_bits() != certified_outer.final_value().to_bits() {
                return Err(format!(
                    "{context} retained profile no longer reproduces its certified objective: replayed={replayed_objective:.17e}, certified={:.17e}",
                    certified_outer.final_value(),
                ));
            }
            let mut profile = selected.value;
            profile.inverse_link_outer = Some(certified_outer);
            Ok(profile)
        }

        match spec.inverse_link.clone() {
            InverseLink::Sas(state0) => optimize_link_parameters(
                data,
                spec,
                kappa_options,
                Array1::from_vec(vec![state0.epsilon, state0.log_delta]),
                "SAS",
                wiggle.clone(),
                |theta| {
                    state_from_sasspec(SasLinkSpec {
                        initial_epsilon: theta[0],
                        initial_log_delta: theta[1],
                    })
                    .map(InverseLink::Sas)
                },
            ),
            InverseLink::BetaLogistic(state0) => optimize_link_parameters(
                data,
                spec,
                kappa_options,
                Array1::from_vec(vec![state0.epsilon, state0.log_delta]),
                "BetaLogistic",
                wiggle.clone(),
                |theta| {
                    state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: theta[0],
                        initial_log_delta: theta[1],
                    })
                    .map(InverseLink::BetaLogistic)
                },
            ),
            InverseLink::Mixture(state0) if !state0.rho.is_empty() => {
                let components = state0.components.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    state0.rho.clone(),
                    "mixture",
                    wiggle.clone(),
                    move |rho| {
                        state_fromspec(&MixtureLinkSpec {
                            components: components.clone(),
                            initial_rho: rho.clone(),
                        })
                        .map(InverseLink::Mixture)
                    },
                )
            }
            _ => profile_survival_location_scale(data, spec.clone(), wiggle, kappa_options),
        }
    }

    let profile = if request.optimize_inverse_link {
        optimize_survival_inverse_link_profile(
            request.data,
            &request.spec,
            request.wiggle.clone(),
            &request.kappa_options,
        )?
    } else {
        profile_survival_location_scale(
            request.data,
            request.spec.clone(),
            request.wiggle.clone(),
            &request.kappa_options,
        )?
    };

    Ok(profile.into_result())
}

pub(crate) fn fit_bernoulli_marginal_slope_model(
    request: BernoulliMarginalSlopeFitRequest<'_>,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    fit_bernoulli_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
        &request.policy,
    )
}

pub(crate) fn fit_survival_marginal_slope_model(
    request: SurvivalMarginalSlopeFitRequest<'_>,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    fit_survival_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )
}

pub(crate) fn fit_latent_survival_model(
    request: LatentSurvivalFitRequest<'_>,
) -> Result<LatentSurvivalTermFitResult, String> {
    fit_latent_survival_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

pub(crate) fn fit_latent_binary_model(
    request: LatentBinaryFitRequest<'_>,
) -> Result<LatentBinaryTermFitResult, String> {
    fit_latent_binary_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

pub(crate) fn fit_transformation_normal_model(
    request: TransformationNormalFitRequest<'_>,
) -> Result<TransformationNormalFitResult, String> {
    fit_transformation_normal(
        &request.response,
        &request.weights,
        &request.offset,
        request.data,
        &request.covariate_spec,
        &request.config,
        &request.options,
        &request.kappa_options,
        request.warm_start.as_ref(),
    )
}

// ---------------------------------------------------------------------------
// Cross-fitted score calibration (Neyman-orthogonal marginal slope, #461)
//
// A marginal-slope Stage-2 model consumes the CTN Stage-1 latent score `z` as a
// generated regressor. `z` depends on θ̂₁, so the β estimating equation leaks
// Stage-1 sampling error (design `marginal_slope_orthogonal_design.md` §1-§4).
// The two DML ingredients are (i) the realized leakage directions
// `J = ∂z/∂θ₁` (an n × p₁ influence Jacobian, computed by the core
// `marginal_slope_orthogonal::score_influence_jacobian`) and (ii) cross-fitting:
// `z` and `J` are evaluated out-of-fold so own-row overfitting of θ̂₁ does not
// bias the absorbed projection. This module owns ingredient (ii) and the
// auto-enable detection: it partitions the rows into K folds, refits the CTN on
// each complement with a basis frozen on the full data, and concatenates the
// per-fold held-out `z` and `J` back into full-n order.

/// Number of cross-fit folds for a problem of `n` rows.
///
/// Cross-fitting refits the CTN once per fold, so the cost is `K` full Stage-1
/// fits. The standard DML default is `K = 5` for moderate `n`. At large scale
/// each CTN refit is expensive while the out-of-fold bias from a single split is
/// already negligible (θ̂₁ is precisely estimated on a complement of ≈ `n·(K−1)/K`
/// rows), so `K` is reduced toward 2 as `n` grows — keeping the refit budget
/// bounded without sacrificing OOF de-biasing. Small `n` keeps more folds (larger
/// per-fold training complements ⇒ less OOF estimation noise), dropping below 5
/// only when there are too few rows to populate 5 folds with a usable held-out
/// block.
///
/// The schedule (no flag, no env — derived purely from `n`):
///   - `n < 250`               : `K = min(n, 3)` (tiny data; keep ≥ 2 folds)
///   - `250 ≤ n < 200_000`      : `K = 5` (DML moderate-n default)
///   - `200_000 ≤ n < 2_000_000` : `K = 3` (large-scale: bound refit cost, ≈ ⅔ train)
///   - `n ≥ 2_000_000`          : `K = 2` (mega-large-scale: ½ train still ample)
fn crossfit_fold_count(n: usize) -> usize {
    if n < 250 {
        n.min(3).max(2)
    } else if n < 200_000 {
        5
    } else if n < 2_000_000 {
        3
    } else {
        2
    }
}

/// Partition `n` rows into `k` folds of balanced, contiguous blocks.
///
/// Entry `f` of the returned vector holds the ascending row indices held out in
/// fold `f`; the union over folds is exactly `0..n`. Sizes are `n / k` or
/// `n / k + 1`, so every fold's complement size differs by at most one row,
/// which keeps the response-basis sample cap — and therefore `p₁` — uniform
/// across folds (design §3). Contiguous blocks let the persistent warm-start
/// prefix cache seed each fold's CTN refit from a structurally identical prior
/// fold.
fn crossfit_partition(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = Vec::with_capacity(k);
    let base = n / k;
    let remainder = n % k;
    let mut start = 0usize;
    for f in 0..k {
        // The first `remainder` folds carry one extra row so the union is exactly n.
        let len = base + usize::from(f < remainder);
        let end = start + len;
        folds.push((start..end).collect());
        start = end;
    }
    folds
}

/// Gather `source[idx]` for each `idx` into a fresh contiguous `Array1`.
fn crossfit_select_rows_1d(source: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_iter(indices.iter().map(|&i| source[i]))
}

/// Cross-fitted out-of-fold `z` and score-influence Jacobian `J` for a CTN →
/// marginal-slope chain (design §4-§5).
///
/// Returns `None` when no CTN Stage-1 recipe is present (`recipe` is `None`):
/// the caller then leaves the Stage-2 spec's `score_influence_jacobian` field
/// `None` and Stage-2 uses the supplied raw `z` with the free-warp fallback.
///
/// When a recipe is present, the covariate basis is built from the recipe's
/// formula RHS and FROZEN once on the full data; every fold then refits the CTN
/// against that frozen spec, and the response-basis knot count is pre-resolved at
/// the *smallest* fold complement size, so every fold refits at an identical
/// `(p_resp, p_cov)` and therefore an identical `p₁ = p_resp · p_cov` column
/// layout — the per-fold `J` blocks concatenate into a coherent `n × p₁` matrix
/// (design §3). For each fold `f` the CTN is refit on the complement rows, then
/// `marginal_slope_orthogonal::score_influence_jacobian` evaluates the held-out
/// `z` and `J` on fold `f`'s rows; results scatter back into full-n order.
pub(crate) fn crossfit_score_calibration(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    recipe: Option<&CtnStage1Recipe>,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> Result<Option<CrossFitScoreCalibration>, String> {
    let Some(recipe) = recipe else {
        return Ok(None);
    };

    let n = data.values.nrows();
    if n == 0 {
        return Err("cross-fit score calibration requires a non-empty dataset".to_string());
    }

    // Stage-1 response / weights / offset, resolved against the full dataset.
    let y_col = resolve_role_col(col_map, &recipe.response_column, "response")
        .map_err(|e| e.to_string())?;
    let response_full = data.values.column(y_col).to_owned();
    let weights_full = resolve_weight_column(data, col_map, recipe.weight_column.as_deref())
        .map_err(|e| e.to_string())?;
    let offset_full = resolve_offset_column(data, col_map, recipe.offset_column.as_deref())
        .map_err(|e| e.to_string())?;

    // Build the CTN covariate basis from the recipe's formula RHS and FREEZE it
    // ONCE on full data, so every fold refit reuses identical spatial centers /
    // knots ⇒ identical p_cov across folds (design §3). The freeze is what makes
    // the per-fold covariate designs column-aligned.
    let parsed_cov = parse_formula(&format!(
        "{} ~ {}",
        recipe.response_column, recipe.covariate_formula_rhs
    ))
    .map_err(|e| e.to_string())?;
    let mut frozen_notes = Vec::new();
    let covariate_spec_raw = build_termspec_with_geometry_and_overrides(
        &parsed_cov.terms,
        data,
        col_map,
        &mut frozen_notes,
        false,
        policy,
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let full_cov_design = build_term_collection_design(data.values.view(), &covariate_spec_raw)
        .map_err(|e| e.to_string())?;
    let frozen_cov_spec = crate::fit_orchestration::drivers::freeze_term_collection_from_design(
        &covariate_spec_raw,
        &full_cov_design,
    )
    .map_err(|e| e.to_string())?;
    let p_cov = full_cov_design.design.ncols();

    let k = crossfit_fold_count(n);
    let folds = crossfit_partition(n, k);

    // Pre-resolve the response-basis internal-knot count at the *smallest* fold
    // complement size. The CTN sample cap on this count is monotone in the
    // complement size, so pinning the per-fold config to the value resolved at
    // the smallest complement makes every fold resolve to the same count —
    // hence a fold-invariant p_resp and an aligned p₁ across folds (design §3).
    let min_complement = folds.iter().map(|held| n - held.len()).min().unwrap_or(n);
    let mut fold_config = recipe.config.clone();
    fold_config.response_num_internal_knots =
        crate::transformation_normal::effective_response_num_internal_knots(
            &recipe.config,
            min_complement,
            p_cov,
            response_full.view(),
        );
    // Pin the resolved knot count: each fold's CTN refit must use exactly this
    // value, not re-derive it from its own response subsample. Without this the
    // data-driven complexity cap rounds to different counts per fold, so p_resp
    // (and p₁ = p_resp · p_cov) drifts across folds and the OOF Jacobian
    // assembly fails the fold-alignment check below ("cross-fit fold p₁ mismatch").
    fold_config.response_num_internal_knots_pinned = true;

    let mut z_oof = Array1::<f64>::zeros(n);
    let mut jac_oof: Option<Array2<f64>> = None;

    for held in &folds {
        if held.is_empty() {
            continue;
        }
        let held_set: std::collections::HashSet<usize> = held.iter().copied().collect();
        let complement: Vec<usize> = (0..n).filter(|i| !held_set.contains(i)).collect();
        if complement.is_empty() {
            return Err(
                "cross-fit fold left an empty training complement; too few rows for K folds"
                    .to_string(),
            );
        }

        // Refit the CTN on the complement (training) rows. The covariate design
        // it uses comes from the frozen spec, so its column geometry matches the
        // held-out Jacobian evaluation below and every other fold.
        let train_cov = data.values.select(Axis(0), &complement);
        let train_resp = crossfit_select_rows_1d(&response_full, &complement);
        let train_weights = crossfit_select_rows_1d(&weights_full, &complement);
        let train_offset = crossfit_select_rows_1d(&offset_full, &complement);

        let fold_fit = fit_transformation_normal(
            &train_resp,
            &train_weights,
            &train_offset,
            train_cov.view(),
            &frozen_cov_spec,
            &fold_config,
            &BlockwiseFitOptions::default(),
            &SpatialLengthScaleOptimizationOptions::default(),
            None,
        )?;

        // Evaluate the OOF score z and the OOF influence Jacobian J on fold f's
        // held-out rows from this fold's fitted CTN. The held-out offset rows
        // enter the PIT operating point identically to the Stage-1 row build, so
        // the emitted z matches the fitted model when the recipe carries an
        // offset column (zeros otherwise ⇒ no-op).
        let held_cov = data.values.select(Axis(0), held);
        let held_resp = crossfit_select_rows_1d(&response_full, held);
        let held_offset = crossfit_select_rows_1d(&offset_full, held);

        let jac = crate::marginal_slope_orthogonal::score_influence_jacobian(
            &fold_fit,
            &held_resp,
            held_cov.view(),
            &held_offset,
        )?;

        if jac.columns.nrows() != held.len() {
            return Err(format!(
                "cross-fit fold Jacobian row count {} != held-out fold size {}",
                jac.columns.nrows(),
                held.len()
            ));
        }
        if jac.z.len() != held.len() {
            return Err(format!(
                "cross-fit fold OOF z length {} != held-out fold size {}",
                jac.z.len(),
                held.len()
            ));
        }

        let p1 = jac.columns.ncols();
        let jac_full = jac_oof.get_or_insert_with(|| Array2::<f64>::zeros((n, p1)));
        if jac_full.ncols() != p1 {
            return Err(format!(
                "cross-fit fold p₁ mismatch: this fold has {p1} columns but a prior fold had {}; \
                 the frozen response/covariate basis failed to align across folds",
                jac_full.ncols()
            ));
        }

        for (local, &global) in held.iter().enumerate() {
            z_oof[global] = jac.z[local];
            for c in 0..p1 {
                jac_full[[global, c]] = jac.columns[[local, c]];
            }
        }
    }

    let jac_oof = jac_oof.ok_or_else(|| {
        "cross-fit produced no folds with held-out rows; cannot assemble OOF Jacobian".to_string()
    })?;

    Ok(Some(CrossFitScoreCalibration { z_oof, jac_oof }))
}
