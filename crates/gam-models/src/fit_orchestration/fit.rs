use super::*;
use gam_solve::estimate::reml::reml_outer_engine::penalty_matrix_root;

/// Inner-PIRLS controls shared by the survival-transformation baseline and
/// smoothing-coordinate eval closures. The baseline geometry is mildly
/// nonlinear, so the iteration budget is generous. The convergence target is
/// the same projected-KKT contract required by the survival LAML envelope; an
/// inner solve that only satisfies a looser tolerance is a checkpoint, not a
/// derivative-bearing objective sample.
const SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS: usize = 400;

const SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL: f64 =
    crate::survival::SURVIVAL_LAML_STATIONARITY_RELATIVE_TOL;

const SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING: usize = 40;

struct SurvivalLocationScaleProfile {
    fit: SurvivalLocationScaleTermFitResult,
    inverse_link: InverseLink,
    wiggle_knots: Option<Array1<f64>>,
    wiggle_degree: Option<usize>,
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
        }
    }
}
fn resolved_wiggle_inverse_link(
    spec: &LikelihoodSpec,
    fit: &UnifiedFitResult,
    fallback: &InverseLink,
) -> Result<InverseLink, FitFailure> {
    let resolved = match fit.fitted_link_state(spec)? {
        FittedLinkState::Standard(Some(link)) => InverseLink::Standard(link),
        FittedLinkState::Standard(None) => fallback.clone(),
        FittedLinkState::LatentCLogLog { state } => InverseLink::LatentCLogLog(state),
        FittedLinkState::Sas { state, .. } => InverseLink::Sas(state),
        FittedLinkState::BetaLogistic { state, .. } => InverseLink::BetaLogistic(state),
        FittedLinkState::Mixture { state, .. } => InverseLink::Mixture(state),
    };
    require_inverse_link_supports_joint_wiggle(&resolved, "standard link wiggle")
        .map_err(FitFailure::input)?;
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

/// The separation certificate that lets a binomial fit switch to the Jeffreys
/// prior, or `None`. Only a proof that the likelihood has no finite maximizer
/// changes the estimator: a solve that did not converge is reported as it is.
fn firth_rescue_evidence(
    error: &gam_solve::estimate::EstimationError,
) -> Option<gam_problem::jeffreys_arming::JeffreysArmingEvidence> {
    error.separation_arming_evidence()
}

/// Whether an automatic Firth retry can use the same outer-coordinate model as
/// the failed base fit.
///
/// Optimized SAS and mixture links append link-parameter coordinates to the
/// REML problem. The Firth outer derivative does not define those coordinates,
/// so the solver rejects that combination before evaluating its seed. Decline
/// the rescue here, where the configuration is already known, instead of
/// launching a retry that is statically incapable of producing a fit (#2654).
fn firth_rescue_has_compatible_outer_coordinates(
    options: &gam_solve::estimate::FitOptions,
) -> bool {
    !options.optimize_mixture && !options.optimize_sas
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
    precision: &mut Array2<f64>,
    factors: &[f64],
) -> Result<(), String> {
    let dimension = factors.len();
    if precision.dim() != (dimension, dimension) {
        return Err(format!(
            "precision must align with the remapped coefficient vector: the precision is {}x{} against {dimension} raw factors",
            precision.nrows(),
            precision.ncols()
        ));
    }
    for i in 0..dimension {
        for j in 0..dimension {
            precision[[i, j]] /= factors[i] * factors[j];
        }
    }
    Ok(())
}

/// Conjugate a coefficient-to-coefficient linear map (the influence matrix
/// `F = H⁻¹X'WX`, the bias-correction Jacobian `A = I + H⁻¹S(λ̂)`) into raw
/// coordinates: with `β_raw = D·β_internal`, `M_raw = D·M·D⁻¹`, which keeps
/// traces (the EDF) invariant.
fn rescale_influence_coordinates(matrix: &mut Array2<f64>, factors: &[f64]) {
    let dimension = factors.len();
    assert_eq!(
        matrix.dim(),
        (dimension, dimension),
        "influence map must align with the remapped coefficient vector"
    );
    for i in 0..dimension {
        for j in 0..dimension {
            matrix[[i, j]] *= factors[i] / factors[j];
        }
    }
}

/// Carry a change of raw coefficient units `β_new = D·β_saved + a` into a saved
/// coefficient gauge whose active frame is not the saved frame. The precision on
/// the active coordinates θ is left as solved: the returned section lifts θ to
/// `D·(T·θ + a_self) + a` (#1561).
fn compose_raw_unit_map_into_gauge(
    gauge: &gam_problem::gauge::Gauge,
    row_factors: &[f64],
    raw_shift: &Array1<f64>,
) -> Result<gam_problem::gauge::Gauge, String> {
    let widths = gauge.raw_widths();
    let total: usize = widths.iter().sum();
    if row_factors.len() != total || raw_shift.len() != total {
        return Err(format!(
            "raw unit map has {} factors and {} shift entries, but the saved coefficient gauge lifts to {total} raw coefficients",
            row_factors.len(),
            raw_shift.len()
        ));
    }
    let mut transforms = Vec::with_capacity(widths.len());
    let mut start = 0usize;
    for width in widths {
        transforms.push(Array2::from_diag(&Array1::from(
            row_factors[start..start + width].to_vec(),
        )));
        start += width;
    }
    let unit_map =
        gam_problem::gauge::Gauge::from_block_transforms_with_shift(&transforms, raw_shift.clone());
    gauge.left_compose(&unit_map).map_err(|reason| {
        format!("the raw unit map does not compose into the saved coefficient gauge: {reason}")
    })
}

#[cfg(test)]
mod standard_convergence_gate_tests {
    use super::{
        certified_retry_or_original, compose_raw_unit_map_into_gauge, firth_rescue_evidence,
        firth_rescue_has_compatible_outer_coordinates, rescale_covariance_coordinates,
        rescale_precision_coordinates, survival_baseline_parameter_checkpoint,
        survival_pirls_status_is_certified,
    };
    use crate::survival::construction::{SurvivalBaselineConfig, SurvivalBaselineTarget};
    use gam_solve::estimate::{EstimationError, FitOptions};
    use gam_solve::pirls::PirlsStatus;
    use ndarray::array;

    #[test]
    fn raw_unit_map_composes_into_a_reduced_coefficient_gauge() {
        // Two saved blocks of widths 3 and 2. The identifiability audit dropped raw
        // column 1 of block 0, so the saved precision lives on 4 active coordinates.
        let gauge = gam_problem::gauge::Gauge::from_block_transforms(&[
            array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            array![[1.0, 0.0], [0.0, 1.0]],
        ]);
        let s = 3.0_f64;
        let factors = [s, s, s, 1.0, 1.0];
        let shift = array![0.0, 0.0, 0.0, s.ln(), 0.0];
        let composed = compose_raw_unit_map_into_gauge(&gauge, &factors, &shift)
            .expect("the unit map composes into a reduced gauge");
        assert_eq!(composed.raw_widths(), vec![3, 2]);
        assert_eq!(composed.reduced_total(), gauge.reduced_total());
        assert!(!composed.is_identity());
        let theta = array![0.7, -1.3, 2.1, 0.4];
        let saved = gauge.t_full.dot(&theta) + &gauge.affine_shift;
        let lifted = composed.t_full.dot(&theta) + &composed.affine_shift;
        for i in 0..factors.len() {
            let expected = factors[i] * saved[i] + shift[i];
            assert!(
                (lifted[i] - expected).abs() <= 1e-15 * (1.0 + expected.abs()),
                "raw coefficient {i}: lifted {} against D·saved + a = {expected}",
                lifted[i]
            );
        }
    }

    #[test]
    fn raw_unit_map_refuses_a_width_that_is_not_the_gauges_raw_frame() {
        let gauge = gam_problem::gauge::Gauge::identity(&[2, 2]);
        let refusal =
            compose_raw_unit_map_into_gauge(&gauge, &[1.0, 1.0, 1.0], &array![0.0, 0.0, 0.0]);
        assert!(
            refusal.is_err(),
            "a unit map over 3 coefficients must not compose onto a 4-coefficient gauge"
        );
    }

    #[test]
    fn raw_units_congruence_refuses_an_active_frame_precision() {
        // The by-group shape: a precision on 4 active coordinates against 5 raw
        // factors. The congruence refuses it and leaves the precision untouched.
        let mut precision = ndarray::Array2::<f64>::eye(4);
        let refusal = rescale_precision_coordinates(&mut precision, &[3.0, 3.0, 3.0, 1.0, 1.0])
            .expect_err("a 4x4 active-frame precision must not align with 5 raw factors");
        assert!(
            refusal.contains("precision must align with the remapped coefficient vector")
                && refusal.contains("4x4 against 5 raw factors"),
            "unexpected refusal: {refusal}"
        );
        assert_eq!(precision, ndarray::Array2::<f64>::eye(4));
    }

    #[test]
    fn raw_coordinate_precision_is_the_inverse_congruence_of_covariance() {
        let mut covariance = array![[0.30, -0.10], [-0.10, 0.70]];
        let mut precision = array![[3.5, 0.5], [0.5, 1.5]];
        let factors = [4.0, 1.0];

        rescale_covariance_coordinates(&mut covariance, &factors);
        rescale_precision_coordinates(&mut precision, &factors)
            .expect("a 2x2 precision aligns with 2 raw factors");

        assert_eq!(covariance, array![[4.8, -0.4], [-0.4, 0.7]]);
        assert_eq!(precision, array![[0.21875, 0.125], [0.125, 1.5]]);
        let identity = precision.dot(&covariance);
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
    fn firth_retry_is_limited_to_proven_separation() {
        // A fit that did not converge is not evidence of separation, so it must
        // not switch the estimator.
        assert_eq!(
            firth_rescue_evidence(&EstimationError::PirlsDidNotConverge {
                iterations: 20,
                budget: 20,
                stop: "max iterations reached".to_string(),
                last_change: 1.0,
            }),
            None
        );
        assert_eq!(
            firth_rescue_evidence(&EstimationError::RemlOptimizationFailed(
                "railed smoothing strength".to_string()
            )),
            None
        );
        assert!(
            firth_rescue_evidence(&EstimationError::PrefitPerfectSeparationDetected {
                column_index: 0,
                threshold: 0.0,
                positive_above_threshold: true,
            })
            .is_some()
        );
        assert_eq!(
            firth_rescue_evidence(&EstimationError::InvalidInput(
                "structural mismatch".to_string()
            )),
            None
        );
    }

    #[test]
    fn firth_retry_declines_every_link_parameter_outer_problem() {
        let ordinary = FitOptions::default();
        assert!(firth_rescue_has_compatible_outer_coordinates(&ordinary));

        let sas = FitOptions {
            optimize_sas: true,
            ..ordinary.clone()
        };
        assert!(!firth_rescue_has_compatible_outer_coordinates(&sas));

        let mixture = FitOptions {
            optimize_mixture: true,
            ..ordinary
        };
        assert!(!firth_rescue_has_compatible_outer_coordinates(&mixture));
    }
}

pub(crate) fn fit_standard_model(
    mut request: StandardFitRequest<'_>,
) -> Result<StandardFitResult, FitFailure> {
    if request.estimate_tweedie_p {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Input,
            "automatic Tweedie power profiling is derivative-free hyperparameter search and is forbidden by SPEC.md; supply an explicit p strictly between 1 and 2",
        ));
    }
    // #2750: resolve every AUTO measure-jet representer range against the
    // response, once, before anything reads the spec.
    //
    // `length_scale == 0.0` is an unresolved request, and it had TWO resolvers:
    // a pure-geometry rule inside the basis builder (the median nearest-node
    // spacing) and the response screen. Which one a model got depended on which
    // branch of the dispatch below it happened to take — the screen ran inside
    // `fit_term_collectionwith_spatial_length_scale_optimization`, so a
    // collection carrying a latent coordinate or coefficient groups was resolved
    // by geometry alone. `ℓ` decides WHICH span the representers occupy and λ
    // cannot move a span, so that is not a tuning difference between branches;
    // it is a different model. One sentinel gets one resolver, and it runs where
    // every branch passes.
    //
    // Idempotent by construction: the screen only fires on the `0.0` sentinel, so
    // the call still inside the spatial driver (reached directly by other
    // drivers and by tests) is a no-op after this one, and the #1762 Firth retry
    // re-enters the dispatch with the range already resolved rather than
    // screening a second time. Failure to screen is never an error — every
    // refusal path leaves the term at the geometry heuristic, which is the
    // pre-#2750 behaviour.
    let seeded = crate::fit_orchestration::drivers::seed_measure_jet_auto_ranges(
        request.data.view(),
        request.y.view(),
        request.weights.view(),
        &mut request.spec,
    );
    if seeded > 0 {
        log::debug!(
            "[#2750] screened the representer range of {seeded} auto measure-jet term(s) against \
             the response before the standard-fit dispatch"
        );
    }

    // #1762/#2273: a separated binomial design has no finite maximum
    // likelihood, on every binomial link. The Jeffreys prior |I(β)|^½ bounds
    // the coefficients there, so a Firth-capable binomial fit whose base fit
    // refused with a pre-fit separation certificate is refit ONCE under it.
    //
    // The estimator changes only on that proof. A base fit that did not
    // converge, railed a smoothing strength or had a trial point refused is a
    // numerical failure of the penalized likelihood fit and is reported as it
    // is: refitting a different model to get past it would hand back an
    // estimator nobody asked for. The adopted fit records the certificate in
    // `FitArtifacts::jeffreys_arming_evidence`, and every summary surface
    // names the estimator and this reason.
    //
    // The retry is adopted only if it carries its own inner and outer
    // certificates. If it fails, the ORIGINAL base error is returned; a failed
    // rescue can never replace its evidence. Link-parameter outer problems are
    // declined before solving because the Firth outer derivative does not
    // define their appended link coordinates (#2654).
    let is_firth_capable_binomial = request.family.supports_firth();
    let base = fit_standard_base(&request, &request.family, &request.options);
    let fitted = match base {
        Ok(fitted) => fitted,
        Err(original_error) => {
            let rescue_is_defined = is_firth_capable_binomial
                && !request.options.firth_bias_reduction
                && firth_rescue_has_compatible_outer_coordinates(&request.options);
            let Some(evidence) = rescue_is_defined
                .then(|| firth_rescue_evidence(&original_error))
                .flatten()
            else {
                return Err(original_error.into());
            };
            let original_report = original_error.to_string();
            let mut firth_options = request.options.clone();
            firth_options.firth_bias_reduction = true;
            let firth = fit_standard_base(&request, &request.family, &firth_options);
            let firth_failure = firth.as_ref().err().map(ToString::to_string);
            match certified_retry_or_original(original_error, firth) {
                Ok(mut firth_fitted) => {
                    log::debug!(
                        "[#1762/#2273] Firth-capable binomial base fit ({}) refused with a \
                         separation certificate ({original_report}); the Jeffreys-prior refit \
                         certified — adopting it (edf {:.2}).",
                        request.family.pretty_name(),
                        firth_fitted.fit.edf_total().unwrap_or(f64::NAN),
                    );
                    firth_fitted.fit.artifacts.jeffreys_arming_evidence = Some(evidence);
                    firth_fitted
                }
                Err(original_error) => {
                    let retry_report = firth_failure
                        .unwrap_or_else(|| "unknown retry failure".to_string());
                    log::debug!(
                        "[#1762/#2273] Firth-capable binomial base fit ({}) failed \
                         ({original_report}); Firth retry also failed to certify \
                         ({retry_report}) — returning the original typed base evidence, not \
                         either abandoned iterate.",
                        request.family.pretty_name(),
                    );
                    // #2273 — the RETURNED error has to say that the rescue was
                    // attempted and why it failed, not only the log line.
                    //
                    // The base evidence is preserved unchanged, because a failed
                    // rescue may not replace it. But the base evidence for a
                    // separated design ends with "enable Firth/Jeffreys bias
                    // reduction or remove/reparameterize the separating column" --
                    // advice to do the thing that was just done automatically and
                    // failed. A caller following it gets the same refusal, and the
                    // reason the rescue failed lives only in a `log::debug!`, which
                    // is not present in a test panic message and is inert through
                    // the Python extension where this pathology is reported.
                    //
                    // So the outcome of the rescue is appended to the returned
                    // message. The typed original is still what was raised; what
                    // changes is that the refusal stops recommending a remedy it
                    // has already tried without saying so.
                    return Err(FitFailure::from(original_error).annotated(format!(
                        "the automatic Firth/Jeffreys rescue WAS attempted \
                         and also failed to certify, so enabling Firth explicitly will not \
                         change this outcome: {retry_report}"
                    )));
                }
            }
        }
    };

    let adaptive_bases = adaptive_bases(&request.spec);
    let result = StandardFitResult {
        saved_link_state: fitted.fit.fitted_link.clone(),
        fit: fitted.fit,
        design: fitted.design,
        resolvedspec: fitted.resolvedspec,
        basis_adequacy: Vec::new(),
        adaptive_bases: adaptive_bases.clone(),
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
    let mut wiggle_options = wiggle.refit_options.clone();
    // A link-wiggle makes the response map curved, so the fitted mode is not a
    // complete model: default prediction needs the joint [Mean, LinkWiggle]
    // posterior to integrate E[g⁻¹(η)]. This is a model invariant, not an
    // optional inference request. Force covariance assembly even for low-level
    // callers that supplied custom refit options with the generic default.
    wiggle_options.compute_covariance = true;
    let wiggle_link_kind =
        resolved_wiggle_inverse_link(&request.family, &result.fit, &wiggle.link_kind)?;
    let fitted_wiggle_family = LikelihoodSpec::try_new(
        request.family.response.clone(),
        wiggle_link_kind.clone(),
    )
    .map_err(|error| {
        // The link was resolved from the pilot's own fit (#2937).
        FitFailure::invariant(format!("invalid resolved link-wiggle likelihood: {error}"))
    })?;
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
    let mut solved = match fit_binomial_mean_wiggle_terms_with_selected_basis(
        request.data.view(),
        &result.resolvedspec,
        &result.design,
        &result.fit,
        request.y.as_ref(),
        request.weights.as_ref(),
        wiggle_link_kind,
        selected_wiggle_basis,
        &wiggle_options,
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
            log::debug!("[linkwiggle] binomial mean link-wiggle joint solve did not converge ({e})");
            return Err(FitFailure::raised(gam_problem::FailureCategory::Convergence, format!(
                "flexible/learnable link requested via link(type=flexible(...)) / \
                 linkwiggle(...), but the binomial mean link-wiggle joint solve did not \
                 converge ({e}). The fit was NOT silently downgraded to the fixed base \
                 link. Refit with a fixed link (e.g. logit/probit/cloglog) or adjust the \
                 wiggle spec (linkwiggle(internal_knots=...)). See gam#1596."
            )));
        }
    };
    if solved.fit.beta_covariance().is_none() {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            "link-wiggle fit reached assembly without its joint [Mean, LinkWiggle] posterior covariance; no model was minted",
        ));
    }
    // The joint link-wiggle solver is a custom block family and therefore does
    // not infer the observation-law metadata stored by the generic likelihood
    // solver. Preserve the resolved response and inverse link explicitly;
    // response-scale prediction after assembly or reload depends on it (#2748).
    solved.fit.likelihood_family = Some(fitted_wiggle_family);

    Ok(StandardFitResult {
        saved_link_state: result.saved_link_state,
        fit: solved.fit,
        design: solved.design,
        resolvedspec: solved.resolvedspec,
        basis_adequacy: Vec::new(),
        adaptive_bases,
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
struct LocationScaleWorkflowParts<'a, S, C> {
    data: ArrayView2<'a, f64>,
    spec: S,
    /// Spec-derived quantities the assembled result records alongside the fit.
    context: C,
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
    /// Spec-derived quantities the assembled result records alongside the fit
    /// (the Gaussian σ floor), computed once in [`Self::into_parts`] before any
    /// solve.
    type Context;

    /// Lower the borrowed request into the family-agnostic workflow parts,
    /// deriving the [`Self::Context`] from the spec the fits consume.
    fn into_parts<'a>(
        request: Self::Request<'a>,
    ) -> Result<LocationScaleWorkflowParts<'a, Self::Spec, Self::Context>, FitFailure>;

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
    ) -> Result<BlockwiseTermFitResult, FitFailure>;

    /// Select the link-wiggle basis from the pilot, then refit the full model
    /// with that selected wiggle block. Consumes `spec`.
    fn refit_with_selected_wiggle(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        pilot: &BlockwiseTermFitResult,
        wiggle_cfg: &LinkWiggleConfig,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermWiggleFitResult, FitFailure>;

    /// Plain non-wiggle fit, used when no wiggle config is present. Consumes
    /// `spec`.
    fn fit_plain(
        data: ArrayView2<'_, f64>,
        spec: Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, FitFailure>;

    /// Assemble the family result from a non-wiggle fit (knots/degree/wiggle
    /// coefficients all absent).
    fn assemble_plain(context: Self::Context, fit: BlockwiseTermFitResult) -> Self::Result;

    /// Assemble the family result from a wiggle refit, carrying the selected
    /// knots/degree and the extracted `beta_link_wiggle` block.
    fn assemble_with_wiggle(
        context: Self::Context,
        fit: BlockwiseTermFitResult,
        wiggle_knots: Array1<f64>,
        wiggle_degree: usize,
        beta_link_wiggle: Option<Vec<f64>>,
    ) -> Self::Result;
}

/// A location-scale solve without covariance is acceptable only when the
/// constrained-posterior layer retained an explicit moment decline. That is a
/// converged diagnostic fit, not a predictive model: saved-model assembly and
/// every fit-aware posterior-mean prediction path reject the same typed state.
fn require_location_scale_covariance_or_decline(
    fit: &UnifiedFitResult,
    context: &str,
) -> Result<(), FitFailure> {
    if fit.beta_covariance().is_some() {
        return Ok(());
    }
    if let Some(decline) = fit.posterior_moment_decline() {
        log::debug!(
            "[{context}] preserving converged constrained fit with unavailable posterior moments: {}",
            decline.summary(),
        );
        return Ok(());
    }
    // The fit was asked for its covariance, so returning neither it nor a typed
    // decline breaks the engine's own contract (#2937).
    Err(FitFailure::raised(
        gam_problem::FailureCategory::Invariant,
        format!(
            "{context} reached assembly without its joint posterior covariance or a typed constrained-posterior moment decline; no model was minted"
        ),
    ))
}

/// Shared wiggle-pilot workflow for Gaussian and binomial location-scale models
/// (#430). The single source of truth for the policy; families differ only via
/// their [`LocationScaleWorkflowAdapter`].
fn fit_location_scale_with_optional_wiggle<A: LocationScaleWorkflowAdapter>(
    request: A::Request<'_>,
) -> Result<A::Result, FitFailure> {
    let LocationScaleWorkflowParts {
        data,
        spec,
        context,
        wiggle,
        options,
        kappa_options,
    } = A::into_parts(request)?;

    let Some(wiggle_cfg) = wiggle else {
        // A location-scale model has two coupled predictors. For binomial
        // location-scale, default response prediction integrates their
        // nonlinear map over the joint Laplace posterior; for Gaussian
        // location-scale, second-channel/delta-method uncertainty needs that
        // same joint posterior. The fitted coefficient mode alone is therefore
        // not a complete model. Request covariance at the final plain fit
        // rather than making every low-level pilot pay for it.
        let mut fit_options = options.clone();
        fit_options.compute_covariance = true;
        let fit = A::fit_plain(data, spec, &fit_options, &kappa_options)?;
        require_location_scale_covariance_or_decline(
            &fit.fit,
            "plain location-scale fit",
        )?;
        return Ok(A::assemble_plain(context, fit));
    };

    let pilot = A::fit_pilot(data, &spec, &options, &kappa_options)?;
    let mut refit_options = options.clone();
    // Link-wiggle response geometry is curved even when the surrounding
    // location model uses an identity link. Its posterior mean therefore
    // requires the complete cross-block covariance at prediction time.
    refit_options.compute_covariance = true;
    let solved = A::refit_with_selected_wiggle(
        data,
        spec,
        &pilot,
        &wiggle_cfg,
        &refit_options,
        &kappa_options,
    )?;

    // The selected link-wiggle basis is appended as the third blockwise term
    // (after the mean/threshold and log-σ blocks), so its coefficients live in
    // block 2 of the refit.
    let fit = solved.fit.fit;
    require_location_scale_covariance_or_decline(
        &fit,
        "location-scale link-wiggle fit",
    )?;
    let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
    let assembled_fit = BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
        fit,
        meanspec_resolved: solved.fit.meanspec_resolved,
        noisespec_resolved: solved.fit.noisespec_resolved,
        mean_design: solved.fit.mean_design,
        noise_design: solved.fit.noise_design,
    })
    .map_err(crate::gamlss::assembly_failure)?;
    Ok(A::assemble_with_wiggle(
        context,
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
    type Context = f64;

    /// The context is the σ floor of the response the fits consume
    /// (`gaussian_resolution_sigma_floor`).
    fn into_parts<'a>(
        request: Self::Request<'a>,
    ) -> Result<LocationScaleWorkflowParts<'a, Self::Spec, f64>, FitFailure> {
        let sigma_floor = crate::sigma_link::gaussian_resolution_sigma_floor(
            request.spec.y.view(),
            request.spec.weights.view(),
        )
        .map_err(crate::gamlss::input_failure)?;
        Ok(LocationScaleWorkflowParts {
            data: request.data,
            spec: request.spec,
            context: sigma_floor,
            wiggle: request.wiggle,
            options: request.options,
            kappa_options: request.kappa_options,
        })
    }

    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, FitFailure> {
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
    ) -> Result<BlockwiseTermWiggleFitResult, FitFailure> {
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
    ) -> Result<BlockwiseTermFitResult, FitFailure> {
        fit_gaussian_location_scale_terms(data, spec, options, kappa_options)
    }

    fn assemble_plain(sigma_floor: f64, fit: BlockwiseTermFitResult) -> Self::Result {
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
            sigma_floor,
        }
    }

    fn assemble_with_wiggle(
        sigma_floor: f64,
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
            sigma_floor,
        }
    }
}

/// Binomial location-scale adapter for the shared wiggle-pilot workflow.
struct BinomialLocationScaleWorkflow;

impl LocationScaleWorkflowAdapter for BinomialLocationScaleWorkflow {
    type Spec = BinomialLocationScaleTermSpec;
    type Request<'a> = BinomialLocationScaleFitRequest<'a>;
    type Result = BinomialLocationScaleFitResult;
    type Context = ();

    fn into_parts<'a>(
        request: Self::Request<'a>,
    ) -> Result<LocationScaleWorkflowParts<'a, Self::Spec, ()>, FitFailure> {
        Ok(LocationScaleWorkflowParts {
            data: request.data,
            spec: request.spec,
            context: (),
            wiggle: request.wiggle,
            options: request.options,
            kappa_options: request.kappa_options,
        })
    }

    fn fit_pilot(
        data: ArrayView2<'_, f64>,
        spec: &Self::Spec,
        options: &BlockwiseFitOptions,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<BlockwiseTermFitResult, FitFailure> {
        // Binomial location-scale requires an inverse link that supports the
        // joint link-wiggle refit; gate it before any fitting work (the pilot
        // runs only on the wiggle path).
        require_inverse_link_supports_joint_wiggle(
            &spec.link_kind,
            "binomial location-scale link wiggle",
        )
        .map_err(|reason| FitFailure::raised(gam_problem::FailureCategory::Input, reason))?;
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
    ) -> Result<BlockwiseTermWiggleFitResult, FitFailure> {
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
    ) -> Result<BlockwiseTermFitResult, FitFailure> {
        fit_binomial_location_scale_terms(data, spec, options, kappa_options)
    }

    fn assemble_plain((): (), fit: BlockwiseTermFitResult) -> Self::Result {
        BinomialLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        }
    }

    fn assemble_with_wiggle(
        (): (),
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
///                                         reconstruct with floor `s·sigma_floor`
///                                         (see `GaussianLocationScalePredictor`).
///                                         `sigma_floor` itself is dimensionless
///                                         and is left unchanged here.
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
) -> Result<(), String> {
    let units = if result
        .fit
        .fit
        .geometry
        .as_ref()
        .map_or(true, |geometry| geometry.coefficient_gauge.is_identity())
    {
        ActiveFrameUnits::RescalePrecision
    } else {
        ActiveFrameUnits::ComposeIntoGauge
    };
    rescale_gaussian_location_scale_to_raw_with_units(result, response_scale, units)
}

/// How the raw remap carries the change of units on the precision side of a
/// saved fit.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ActiveFrameUnits {
    /// The precision lives on the saved coordinates (identity gauge): rescale it
    /// contravariantly.
    RescalePrecision,
    /// The precision lives on the gauge's active coordinates: compose the unit map
    /// into the gauge and leave every active-frame quantity as solved.
    ComposeIntoGauge,
}

/// [`rescale_gaussian_location_scale_to_raw`] with the precision-side
/// representation chosen by the caller. On an identity gauge both representations
/// describe one saved state (#1561).
pub(crate) fn rescale_gaussian_location_scale_to_raw_with_units(
    result: &mut GaussianLocationScaleFitResult,
    response_scale: f64,
    units: ActiveFrameUnits,
) -> Result<(), String> {
    use gam_problem::BlockRole;

    if matches!(units, ActiveFrameUnits::ComposeIntoGauge) && result.fit.fit.geometry.is_none() {
        return Err(
            "gaussian location-scale raw remap: composing the unit map into the coefficient \
             gauge needs saved geometry"
                .to_string(),
        );
    }

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
    //
    // The same change of units is one affine map on the saved coefficients,
    // `β_raw = D·β_internal + a`, with `a = ln(s)` on the Scale block's intercept
    // columns (the shift the block surgery above applies) and zero elsewhere.
    let mut row_factors: Vec<f64> = Vec::new();
    let mut raw_shift: Vec<f64> = Vec::new();
    for block in &result.fit.fit.blocks {
        let f = match block.role {
            BlockRole::Mean | BlockRole::Location | BlockRole::LinkWiggle => s,
            BlockRole::Scale | BlockRole::Time | BlockRole::Threshold => 1.0,
        };
        let block_start = raw_shift.len();
        row_factors.extend(std::iter::repeat_n(f, block.beta.len()));
        raw_shift.extend(std::iter::repeat_n(0.0, block.beta.len()));
        if matches!(block.role, BlockRole::Scale) {
            for col in scale_intercept_range.clone() {
                if col < block.beta.len() {
                    raw_shift[block_start + col] = ln_s;
                }
            }
        }
    }
    let raw_shift = Array1::from(raw_shift);
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
    //
    // That congruence holds only where the precision lives on the saved
    // coordinates. A saved precision lives on the active coordinates of the
    // geometry's coefficient gauge, `β_saved = T·θ + a`. When the custom-family
    // identifiability audit drops aliased columns, `T` is rectangular and `H_θ`
    // stays on the reduced frame. The change of units then moves the lift, not
    // θ: it composes into the gauge, and every active-frame quantity stays as
    // solved.
    let precision_on_saved_frame = matches!(units, ActiveFrameUnits::RescalePrecision);
    if let Some(geometry) = result.fit.fit.geometry.as_mut() {
        if precision_on_saved_frame {
            rescale_precision_coordinates(&mut geometry.penalized_hessian.0, &row_factors)?;
        } else {
            geometry.coefficient_gauge = compose_raw_unit_map_into_gauge(
                &geometry.coefficient_gauge,
                &row_factors,
                &raw_shift,
            )?;
        }
    }
    if let Some(inference) = result.fit.fit.inference.as_mut() {
        if precision_on_saved_frame {
            rescale_precision_coordinates(&mut inference.penalized_hessian.0, &row_factors)?;
        }

        // The conditional and corrected covariances have one store each, the
        // top-level matrices remapped above, and their standard errors derive
        // from it (#2955). The inference block's other coefficient-frame
        // covariance objects ride the same remap.
        if let Some(cov) = inference.beta_covariance_frequentist.as_mut() {
            rescale_covariance_coordinates(cov, &row_factors);
        }
        if let Some(correction) = inference.smoothing_correction.as_mut() {
            rescale_covariance_coordinates(correction, &row_factors);
        }
        if let Some(se) = inference.factorized_standard_errors.as_mut() {
            for (value, &factor) in se.iter_mut().zip(row_factors.iter()) {
                *value *= factor;
            }
        }
        // X'WX is a precision-side quadratic form exactly like H, and the influence
        // map acts on the same coordinates, so both change with the units only
        // where H does.
        if precision_on_saved_frame {
            if let Some(gram) = inference.weighted_gram.as_mut() {
                rescale_precision_coordinates(gram, &row_factors)?;
            }
            if let Some(influence) = inference.coefficient_influence.as_mut() {
                rescale_influence_coordinates(influence, &row_factors);
            }
        }
        // `β_saved = Qs·θ` puts the rows of the stabilizing reparameterization
        // in the saved coefficient frame: Qs_raw = D·Qs.
        if let Some(qs) = inference.reparam_qs.as_mut() {
            for (mut row, &factor) in qs.rows_mut().into_iter().zip(row_factors.iter()) {
                row.mapv_inplace(|v| v * factor);
            }
        }
    }

    // The residual-scale summary `standard_deviation` is a response-units
    // quantity; the internal fit reports it in standardized units, so map it
    // back. `max_abs_eta` is the mean-channel η magnitude (raw μ = s·μ_internal).
    result.fit.fit.standard_deviation *= s;
    result.fit.fit.max_abs_eta *= s;

    // Change-of-variables correction for the likelihood-scale summaries. The
    // internal fit maximizes the density of y_internal = y/s; the raw-response
    // density is p_raw(y) = p_internal(y/s)/s, so per observation
    // log p_raw = log p_internal − ln(s), and the REML/LAML objective (which
    // carries the data log-likelihood) shifts accordingly. The deviance is the
    // classical weighted RSS `Σ w (y − μ)²` (#2786), which scales by `s²`
    // rather than shifting. This keeps reported log-likelihood / deviance /
    // REML in raw response units, matching what an un-standardized fit would
    // report.
    result.fit.fit.deviance *= s * s;
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
        if let Some(mode_log_likelihood) = result
            .fit
            .fit
            .geometry
            .as_mut()
            .and_then(|geometry| geometry.constrained_posterior.as_mut())
            .and_then(|constrained| constrained.mode_log_likelihood.as_mut())
        {
            *mode_log_likelihood -= n_obs * ln_s;
        }
        result.fit.fit.shift_criterion(n_obs * ln_s);
    }

    result.response_scale = s;
    Ok(())
}

pub(crate) fn fit_gaussian_location_scale_model(
    mut request: GaussianLocationScaleFitRequest<'_>,
) -> Result<GaussianLocationScaleFitResult, FitFailure> {
    // Standardize the response so the scale block works on a unit-spread
    // response whatever the recording units. The σ floor is the recording-grid
    // bound of this standardized response (`gaussian_resolution_sigma_floor`),
    // so it scales with y and the fit is exactly response-scale-equivariant.
    let response_scale = gaussian_response_sample_std(request.spec.y.view());
    // A response with no spread has no scale to standardise by, and no
    // location-scale model either: refuse it rather than fit `y / 1e-6`
    // (#2469).
    if !(response_scale > 0.0) || !response_scale.is_finite() {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Input,
            format!(
                "gaussian location-scale fit: the response has no finite positive spread \
                 (sample std = {response_scale:.3e}); a location-scale model needs one"
            ),
        ));
    }
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

    let mut result =
        fit_location_scale_with_optional_wiggle::<GaussianLocationScaleWorkflow>(request)?;

    // The raw-unit remap rewrites a fitted result the engine assembled, so its
    // refusals are shape disagreements inside that result (#2937).
    rescale_gaussian_location_scale_to_raw(&mut result, response_scale)
        .map_err(crate::gamlss::assembly_failure)?;
    Ok(result)
}

pub(crate) fn fit_dispersion_location_scale_model(
    request: DispersionLocationScaleFitRequest<'_>,
) -> Result<DispersionLocationScaleFitResult, FitFailure> {
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
) -> Result<BinomialLocationScaleFitResult, FitFailure> {
    fit_location_scale_with_optional_wiggle::<BinomialLocationScaleWorkflow>(request)
}

/// Penalized effective degrees of freedom for a survival transformation fit.
///
/// Uses exactly the mgcv definition `edf_total = p − Σ_k λ_k·tr(H⁻¹ S_k)`, where
/// `H` is the converged penalized Hessian `X'W_HX + S(λ)` (held in
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
) -> Result<
    (f64, Vec<f64>, Vec<f64>, Vec<gam_solve::estimate::EdfRankBound>, Array2<f64>),
    String,
> {
    let h_dense = state.hessian.to_dense();
    let (edf_total, edf_by_block, penalty_block_trace, rank_bound) =
        survival_edf_from_dense_hessian(&h_dense, penalty_blocks)?;
    Ok((edf_total, edf_by_block, penalty_block_trace, rank_bound, h_dense))
}

/// Trace-form penalized EDF from a converged dense penalized Hessian.
///
/// Factored out of [`survival_transformation_edf`] so the exact-solve/naming
/// contract is unit-testable against synthetic Hessians without a full
/// `WorkingState`.
fn survival_edf_from_dense_hessian(
    h_dense: &Array2<f64>,
    penalty_blocks: &[PenaltyBlock],
) -> Result<(f64, Vec<f64>, Vec<f64>, Vec<gam_solve::estimate::EdfRankBound>), String> {
    let p = h_dense.nrows();
    let h_sym = gam_linalg::matrix::SymmetricMatrix::Dense(h_dense.clone());
    // EDF is an exact trace of the fitted (unperturbed) penalized Hessian.
    // Factoring a different, ridged matrix silently changes the estimand, so a
    // singular/indefinite fitted Hessian is an inference failure rather than a
    // license to manufacture a nearby covariance. The Weibull anchor gauge that
    // used to make this Hessian singular (#2301) is removed at design build — the
    // redundant Linear time-basis constant column is dropped in
    // `build_survival_time_basis` — so a singularity HERE is now a genuine defect
    // and refuses with the named flat direction (diag a0a9771ca).
    let factor = h_sym.factorize().map_err(|error| {
        format!("survival edf: exact penalized-Hessian factorization failed: {error}")
    })?;
    let solve = |values: &mut [f64]| -> Result<(), String> {
        let solved = factor.solve(&ndarray::Array1::from(values.to_vec()))?;
        for (slot, value) in values.iter_mut().zip(solved.iter()) {
            *slot = *value;
        }
        Ok(())
    };
    let inverse_one_norm = gam_linalg::condition::estimate_inverse_one_norm(p, solve, solve)
        .map_err(|error| format!("survival edf: inverse-norm estimate failed: {error}"))?;
    // Per-block penalty traces, the rounding band of the solve behind each, and
    // their ranks, handed to the shared accounting (#2470, #2901). The rank comes
    // from the realized penalty root, NOT from the declared `block.nullspace_dim`:
    // a declared nullity is a pre-transform statement that canonical pullback
    // intentionally clears, so consulting it here can price a block against a
    // rank the fitted penalty no longer has. `penalty_matrix_root` is the same
    // oracle the REML criterion uses when it charges `rank(S_k)·rho_k`.
    let mut raw_traces = vec![0.0_f64; penalty_blocks.len()];
    let mut trace_bands = vec![0.0_f64; penalty_blocks.len()];
    let mut rank_bounds = Vec::with_capacity(penalty_blocks.len());
    let mut block_ranks = vec![0_usize; penalty_blocks.len()];
    // `Σ_k S_k` in the joint layout. Summed UNSCALED on purpose: the penalty
    // null space is a structural property of the penalty geometry, so the floor
    // it induces must not move with `λ`.
    let mut joint_penalty = Array2::<f64>::zeros((p, p));
    for (kk, block) in penalty_blocks.iter().enumerate() {
        let block_cols = block.range.end - block.range.start;
        let root = if block_cols == 0 {
            Array2::<f64>::zeros((0, 0))
        } else {
            penalty_matrix_root(&block.matrix).map_err(|error| {
                format!("survival edf: penalty {kk} rank factorization failed: {error}")
            })?
        };
        let penalty_rank = root.nrows();
        block_ranks[kk] = penalty_rank;
        if block_cols > 0 {
            let r = block.range.start..block.range.end;
            let mut target = joint_penalty.slice_mut(ndarray::s![r.clone(), r]);
            target += &block.matrix;
        }
        // #2901: the survival Hessian is observed information, so `H ⪰ λ_k S_k` is
        // certified per block from the inertia of `H − λ_k S_k` shifted by its rounding
        // band.
        let scaled_penalty_block = if block_cols > 0 && block.lambda > 0.0 {
            &block.matrix * block.lambda
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        rank_bounds.push(
            gam_solve::estimate::numerical_rank_bound(
                h_dense.view(),
                scaled_penalty_block.view(),
                block.range.start,
                gam_runtime::resource::MemoryGovernor::global(),
            )
            .map_err(|error| {
                format!("survival edf: penalty block {kk} rank certificate failed: {error}")
            })?,
        );
        if block.lambda <= 0.0 || block_cols == 0 {
            raw_traces[kk] = 0.0;
            continue;
        }
        // RHS = the root's modes (`S_k = RᵀR`) placed in the block rows of the
        // p×rank layout, so λ_k tr(H⁻¹S_k) = λ_k Σ_c r_cᵀ H⁻¹ r_c.
        let mut rhs = Array2::<f64>::zeros((p, penalty_rank));
        rhs.slice_mut(ndarray::s![block.range.clone(), ..])
            .assign(&root.t());
        let sol = factor.solvemulti(&rhs).map_err(|e| {
            // A converged fit whose penalized Hessian cannot support a finite
            // trace solve is an identifiability failure; name the flat direction
            // (issue #2301, diag a0a9771ca) instead of the opaque solver string.
            let spectrum_note =
                match gam_linalg::faer_ndarray::FaerEigh::eigh(h_dense, faer::Side::Lower) {
                    Ok((eigenvalues, eigenvectors)) => {
                        let mut min_idx = 0usize;
                        for (idx, value) in eigenvalues.iter().enumerate() {
                            if value.abs() < eigenvalues[min_idx].abs() {
                                min_idx = idx;
                            }
                        }
                        let max_abs = eigenvalues
                            .iter()
                            .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
                        let flat_direction: Vec<f64> =
                            eigenvectors.column(min_idx).iter().copied().collect();
                        format!(
                            "penalized-Hessian spectrum: min_abs_eig={:.6e}, max_abs_eig={:.6e}, \
                             eigenvalues={:?}, flattest direction (coefficient loadings)={:?}",
                            eigenvalues[min_idx], max_abs, eigenvalues, flat_direction
                        )
                    }
                    Err(error) => {
                        format!("penalized-Hessian eigendecomposition also failed: {error:?}")
                    }
                };
            format!(
                "survival edf trace solve failed for penalty block {kk} \
                 (lambda={:.6e}, block_cols={block_cols}): {e}; {spectrum_note}",
                block.lambda
            )
        })?;
        let (trace, band) = gam_linalg::roundoff::solved_penalty_trace(
            block.lambda,
            rhs.view(),
            sol.view(),
            h_dense.view(),
            inverse_one_norm,
        )
        .map_err(|error| format!("survival edf: penalty block {kk} trace band failed: {error}"))?;
        raw_traces[kk] = trace;
        trace_bands[kk] = band;
    }
    let joint_penalty_rank = penalty_matrix_root(&joint_penalty)
        .map_err(|error| format!("survival edf: joint penalty rank failed: {error}"))?
        .nrows();
    let bundle = gam_solve::estimate::penalized_edf_bundle_within_bands(
        &raw_traces,
        &trace_bands,
        &rank_bounds,
        &block_ranks,
        p,
        (p - joint_penalty_rank.min(p)) as f64,
    )
    .map_err(|error| format!("survival edf: {error}"))?;
    let edf_by_block = bundle.edf_by_block;
    let penalty_block_trace = bundle.penalty_block_trace;
    let edf_total = bundle.edf_total;
    if !edf_total.is_finite()
        || edf_by_block.iter().any(|v| !v.is_finite())
        || penalty_block_trace.iter().any(|v| !v.is_finite())
    {
        return Err("survival edf: non-finite effective degrees of freedom".to_string());
    }
    Ok((edf_total, edf_by_block, penalty_block_trace, bundle.rank_bound))
}

/// REML/LAML smoothing-parameter selection for the single-cause transformation
/// survival baseline (issue #563).
///
/// The transformation path solves a constrained PIRLS (`γ ≥ 0` I-spline box) at
/// a fixed time-penalty `λ`, which oversmooths: with `λ` pinned at its seed the
/// monotone baseline collapses toward an affine log-cumulative-hazard and cannot
/// recover real curvature (e.g. Gompertz convexity). This routine wraps that
/// inner solve in a proper outer LAML optimization over `ρ = log λ` for every
/// penalty block, exactly as the standard GAM path and mgcv/scam do. The inner
/// solve still honors the structural box at every candidate `λ`, so the
/// constrained optimum stays valid; only the outer `λ` becomes data-adaptive.
/// Every block the working model carries is a REML-selected smoothing block:
/// there is no fixed-λ term in the survival objective (a coefficient ridge at a
/// hand-supplied λ was a penalty on the coefficients rather than on the
/// function, and its value was chosen by no criterion — #2670; conditioning of
/// the inner Newton path is the solver's Levenberg–Marquardt damping, see
/// `WorkingModelSurvival::update_state`).
///
/// `model` is the working model at the seed `λ`; it is cloned per candidate so
/// the proposal never corrupts the warm model. The returned vector has one
/// REML-selected `λ_k` per penalty block. Returns `None` when there are no
/// penalty blocks to select (e.g. the Weibull linear-time path), so the caller
/// keeps the seed.
///
/// # No left-truncation box on the time baseline (issues #1790/#1791, #2670)
///
/// The transformation-survival LAML `−½·log|H|` term uses the **observed**
/// information `H = X_exitᵀW_exit X_exit − X_entryᵀW_entry X_entry + (event/deriv)
/// + S(λ)`. Under genuine left truncation the delayed-entry rows contribute the
/// negative `−X_entryᵀW_entry X_entry` block, and below some time smoothing `λ`
/// the inner mode's `H` is indefinite. While the criterion turned such an `H`
/// into a spectrally regularised `log|H|`, shrinking the time `λ` was rewarded
/// (the near-null time direction lowered the cost), the outer search railed the
/// time block to its lower bound, and the under-smoothed baseline inflated into a
/// covariate-flat constant offset — the degenerate fit #1790/#1791 report. The
/// time blocks' outer lower bound was then floored at the seed `ρ` under left
/// truncation, a hand box that let the selector only hold or over-smooth.
///
/// That box guarded a criterion that no longer exists: an indefinite observed
/// `H` at a trial ρ is not a Laplace mode and is REFUSED by
/// `unified_lamlobjective_and_rhogradient` (a `TrialPointRefused` the outer
/// engine retreats from), never converted into a positive-subspace
/// pseudo-objective; and the inner loop no longer returns a saddle step on an
/// indefinite curvature (`newton_solve::descent_curvature`). So the ρ domain is
/// the outer engine's own for every block, left-truncated or not; the
/// heterogeneous-entry cohort of #2814 selects `ρ̂ = 10.5` for its time block
/// through this window with the same certificate as before.
/// Outcome of the survival smoothing-parameter selection: the selected λ plus
/// the OUTER convergence evidence (#2301 defect D). The analytic
/// stationarity certificate that `OuterProblem::run` mints is threaded through
/// to the fit's `FitArtifacts` so assembly can certify the outer optimum, rather
/// than being discarded (which left the assembly gate comparing the outer
/// residual `against None` and refusing a converged fit).
struct SurvivalSmoothingSelection {
    lambdas: Vec<f64>,
    outer_iterations: usize,
    criterion_certificate: Option<gam_solve::estimate::OuterCriterionCertificate>,
    /// The inner mode β̂ the selector certified at its last evaluation. The
    /// outer engine's finalize re-runs the inner solve at the selected ρ, so
    /// after a successful selection this is the strict-`Converged` coefficient
    /// vector at `lambdas` itself. The fixed-λ solve that mints the fit starts
    /// here and re-certifies it, instead of re-deriving it from the cold
    /// structural seed: on a heterogeneous delayed-entry cohort the cold path
    /// crawls from the corner of the `γ ≥ 0` box (396 iterations at the seed
    /// ρ, a 20-step objective plateau exit at the selected ρ) and the fit was
    /// refused although its certified mode already existed (#2670).
    certified_mode: Array1<f64>,
    /// The analytic LAML ρ-Hessian the terminal mint evaluated at the selected
    /// ρ, which the fit's smoothing correction inverts (#2912).
    outer_hessian: Option<Array2<f64>>,
    /// The outer gradient at the selected ρ: the certificate's gradient floor
    /// the correction's identified inverse is judged against (#2346).
    outer_gradient: Option<Array1<f64>>,
}

fn optimize_survival_transformation_smoothing(
    model: &crate::survival::WorkingModelSurvival,
    penalty_blocks: &[PenaltyBlock],
    beta0: &Array1<f64>,
    structural_lower_bounds: Option<&Array1<f64>>,
) -> Result<Option<SurvivalSmoothingSelection>, FitFailure> {
    use gam_problem::{Derivative, HessianValue, OuterEval};
    use gam_solve::rho_optimizer::{OuterEvalOrder, OuterProblem};
    // One outer coordinate per penalty block: every block is REML-selected.
    let num_smoothing = penalty_blocks.len();
    if num_smoothing == 0 {
        return Ok(None);
    }
    let seed_lambdas: Vec<f64> = penalty_blocks.iter().map(|b| b.lambda).collect();
    let seed_log_lambdas = seed_lambdas
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, value)| {
            gam_problem::checked_log_strength(value).map_err(|error| {
                FitFailure::raised(
                    gam_problem::FailureCategory::Numerical,
                    format!("survival transformation seed lambda {coordinate}: {error}"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let seed_rho = Array1::from_vec(seed_log_lambdas);

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
    //
    // The entry also carries the certified inner mode β̂ it was evaluated at. The
    // terminal mint asks for curvature at the ρ it has just valued, and the cache
    // holds no Hessian, so that request evaluates again (#2912). It evaluates at
    // THIS mode: continuing P-IRLS from it stops at a second "converged" β̂ whose
    // LAML differs from the valued one by more than the outer value-agreement
    // audit admits (#1082 q26: 1.390e-4 against a 2.274e-5 bound), and the mint
    // then refused a converged fit for pricing value and curvature at two modes.
    // A line-search value probe stores no gradient, so a gradient request at its ρ
    // evaluates at its mode instead of reading one.
    let eval_cache: std::cell::RefCell<
        Option<(Array1<f64>, f64, Option<Array1<f64>>, Array1<f64>)>,
    > = std::cell::RefCell::new(None);
    // Warm-start chaining for the inner PIRLS across outer probes (#2298). The
    // generic BFGS bridge evaluates this objective at a sequence of spatially
    // adjacent ρ — line-search probes walk along one direction and accepted steps
    // advance it — so the last CONVERGED β̂(ρ_prev) is a far better inner seed than
    // the fixed cold β̂(ρ_seed). As the selector over-smooths the baseline out
    // toward its box bound the probe ρ drift O(10) log-λ units from the seed,
    // where the stale cold seed leaves the coupled constrained inner PIRLS
    // non-convergent: `eval_at` then returns typed inner non-convergence, every
    // line-search step reads +∞ cost, and BFGS starves — the observed death at ~5
    // iterations with |Pg| ≫ tol and the baseline coordinate railed. The inner
    // penalized likelihood is strictly convex on the feasible (monotonicity) cone,
    // so β̂(ρ) is independent of the feasible seed: the LAML envelope value and
    // ρ-gradient are unchanged bit-for-bit, only the inner convergence at drifted
    // ρ is restored. A non-converged probe never advances the seed (see below), so
    // a bad probe cannot corrupt the warm start for the next attempt.
    let warm_beta: std::cell::RefCell<Array1<f64>> = std::cell::RefCell::new(beta0.clone());
    // Evaluate the LAML objective at a ρ proposal, its ρ-gradient when the order
    // asks for one, and its analytic ρ-Hessian when it asks for curvature: set the
    // λ, re-run the constrained inner PIRLS, and evaluate the unified survival
    // LAML. A line-search value probe discards any gradient, so it asks for none.
    let eval_at = |rho_smooth: &Array1<f64>, order: OuterEvalOrder| -> Result<
        (f64, Option<Array1<f64>>, HessianValue),
        gam_solve::estimate::EstimationError,
    > {
        let physical_smoothing =
            gam_problem::checked_exp_log_strengths(rho_smooth.iter().copied())?;
        // The cache holds no Hessian, so a curvature request always evaluates: at
        // the cached mode when the cache is at this ρ, otherwise from P-IRLS. A
        // gradient request at a value probe's ρ evaluates the same way.
        let wants_hessian = matches!(order, OuterEvalOrder::ValueGradientHessian);
        let wants_gradient = !matches!(order, OuterEvalOrder::Value);
        let cached_mode = match eval_cache.borrow().as_ref() {
            Some((cached_rho, cached_cost, cached_grad, cached_beta))
                if cached_rho == rho_smooth =>
            {
                match (wants_hessian, wants_gradient, cached_grad) {
                    (false, false, _) => {
                        return Ok((*cached_cost, None, HessianValue::Unavailable));
                    }
                    (false, true, Some(gradient)) => {
                        return Ok((
                            *cached_cost,
                            Some(gradient.clone()),
                            HessianValue::Unavailable,
                        ));
                    }
                    _ => Some(cached_beta.clone()),
                }
            }
            _ => None,
        };
        let mut candidate = model.clone();
        candidate
            .set_penalty_lambdas(&physical_smoothing)
            // A lambda THIS TRIAL RHO produced was refused by the model. The
            // outer search's only lever is rho, and moving it is the right
            // response, so the verdict must be per-trial-point, not per-problem
            // (#2531/#2590): minting `InvalidInput` here graded the refusal
            // Fatal (`is_trial_point_infeasible` is false for it) and killed the
            // whole outer search instead of retreating from one rho.
            //
            // `set_penalty_lambdas`'s length-mismatch arm is structurally
            // unreachable from this call site: the proposal has one coordinate
            // per penalty block by construction (`OuterProblem::new(num_smoothing)`
            // below). The only reachable arm is the lambda-VALUE arm, which is
            // rho-local.
            //
            // `wrap_preserving_trial_point` is deliberately NOT used here: the
            // source only ever produces `InvalidInput`, for which that helper is
            // a no-op, so it would be a silent non-fix.
            .map_err(|error| {
                gam_solve::estimate::EstimationError::TrialPointRefused {
                    reason: format!("survival smoothing trial lambda rejected: {error}"),
                }
            })?;
        let beta = match cached_mode {
            Some(beta) => beta,
            None => {
                let opts = gam_solve::pirls::WorkingModelPirlsOptions {
                    max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
                    convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
                    adaptive_kkt_tolerance: None,
                    max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
                    firth_bias_reduction: false,
                    coefficient_lower_bounds: structural_lower_bounds.cloned(),
                    linear_constraints: None,
                    initial_lm_lambda: None,
                };
                let summary = gam_solve::pirls::runworking_model_pirls(
                    &mut candidate,
                    gam_problem::Coefficients::new(warm_beta.borrow().clone()),
                    &opts,
                    Some(&mut |info: &gam_solve::pirls::WorkingModelIterationInfo| {
                        log::trace!(
                            "[SURV-TRANS inner] iter={} deviance={:.6e} |grad|={:.6e} step={:.3e} \
                             halvings={}",
                            info.iteration,
                            info.deviance,
                            info.gradient_norm,
                            info.step_size,
                            info.step_halving
                        );
                    }),
                )?;
                // The envelope gradient exists only at a certified beta optimum. A
                // finite exhausted state is a checkpoint, not a derivative-bearing
                // objective sample, so refuse this trial point and let the generic
                // outer bridge retreat from this rho without fabricating a cost or a
                // zero gradient.
                //
                // The refusal carries the solve's real terminal status. Mapping every
                // non-`Converged` status to `PirlsDidNotConverge { max_iterations }`
                // printed "did not converge within 400 iterations" for solves that had
                // stopped after 2 to 9 iterations on a numerical plateau
                // (`LmStepSearchExhausted`, exact decrement just above its threshold),
                // pointing diagnosis at a budget that was never spent (#2705, #1561).
                // Both variants grade as the same trial-point retreat
                // (`EstimationError::is_trial_point_infeasible`, one table since #2593).
                if !survival_pirls_status_is_certified(summary.status) {
                    // The exact decrement is the half of the certificate a plateau exit
                    // misses by, so the refusal reports it on the same monotonicity rows,
                    // curvature correction and deviance scale the LAML gate uses.
                    let decrement = gam_solve::pirls::exact_newton_decrement_evidence(
                        &summary.state,
                        summary.beta.as_ref(),
                        candidate.monotonicity_linear_constraints().as_ref(),
                        gam_solve::pirls::WorkingModel::objective_hessian_matrix_correction(
                            &candidate,
                        ),
                        gam_solve::pirls::WorkingModel::penalized_deviance_scale(&candidate)?,
                    );
                    let decrement_note = match decrement.decrement_sq {
                        Some(decrement_sq) => format!("{decrement_sq:.3e}"),
                        None => "unavailable (the face curvature did not factorize)".to_string(),
                    };
                    return Err(gam_solve::estimate::EstimationError::TrialPointRefused {
                        reason: format!(
                            "survival transformation inner P-IRLS at this trial rho ended with \
                             status {:?} after {} of {} iteration(s) (projected gradient norm \
                             {:.6e}, exact Newton decrement {decrement_note} against threshold \
                             {:.3e}) without a strict convergence certificate; no envelope \
                             gradient exists at this rho",
                            summary.status,
                            summary.iterations,
                            opts.max_iterations,
                            summary.lastgradient_norm,
                            decrement.threshold,
                        ),
                    });
                }
                summary.beta.as_ref().to_owned()
            }
        };
        // Advance the warm start: a CERTIFIED inner mode at this ρ (the
        // convergence gate above already rejected non-certified states) is the
        // best available seed for the next, adjacent probe. Reached only after
        // certification, so a refused probe leaves the previous good β̂ in place.
        *warm_beta.borrow_mut() = beta.clone();
        let state = candidate.update_state(&beta).map_err(|error| {
            // Same rule as the LAML wrapper below: keep the source's
            // classification, add only context (#2531).
            error.wrap_preserving_trial_point("survival smoothing inner state evaluation failed")
        })?;
        // The proposal IS the active-penalty ρ, in block order, as the unified
        // survival LAML evaluator requires: every block is an outer coordinate,
        // so the evaluator sees the optimizer's ρ itself rather than an
        // `exp`/`ln` round trip of it.
        let mode = match order {
            OuterEvalOrder::Value => gam_problem::EvalMode::ValueOnly,
            OuterEvalOrder::ValueAndGradient => gam_problem::EvalMode::ValueAndGradient,
            OuterEvalOrder::ValueGradientHessian => gam_problem::EvalMode::ValueGradientHessian,
        };
        let (cost, grad_full, hessian, _resolution) = candidate
            .unified_lamlobjective_and_rhogradient(&beta, &state, rho_smooth, mode)
            // Adding context must not change the verdict. Re-rendering the
            // source into `InvalidInput` overwrote the producer's "this trial
            // point, not this problem" with "this configuration is wrong", and
            // the outer boundary reads only the variant (#2531).
            .map_err(|error| {
                error.wrap_preserving_trial_point("survival smoothing LAML evaluation failed")
            })?;
        // The gradient is ∂LAML/∂ρ over the active blocks, which are exactly the
        // outer coordinates. A layout defect and a trial-point defect must not
        // share a verdict, so the layout check comes FIRST and each gets the
        // grading it earns.
        //
        // A gradient whose length is not the block count is a LAYOUT defect of
        // the evaluator, not a property of this rho: no choice of rho changes a
        // gradient's length. It stays Fatal `InvalidInput`.
        //
        // Recorded honestly, because it cuts the other way: the framework's own
        // `OuterThetaLayout::validate_gradient_len`
        // (`rho_optimizer/capability.rs`) grades a gradient-length mismatch
        // RECOVERABLE. So the "the layer below already says so" argument that
        // justifies the cost and gradient arms below points the OPPOSITE
        // direction here. This keeps the pre-existing Fatal grading for the
        // layout half and changes only the halves where the framework agrees;
        // resolving which of the two sites is miscategorized is separate work.
        if grad_full.len() != num_smoothing {
            return Err(gam_solve::estimate::EstimationError::InvalidInput(format!(
                "survival smoothing LAML gradient has {} entries for \
                 {num_smoothing} smoothing coordinates",
                grad_full.len()
            )));
        }
        // A non-finite cost IS rho-local, and the framework one layer down
        // grades exactly this condition recoverable:
        // `rho_optimizer/objective.rs` `finite_cost_or_error` returns
        // `ObjectiveEvalError::recoverable` for it. Grading it Fatal here
        // contradicted the layer this closure feeds (#2531/#2590).
        if !cost.is_finite() {
            return Err(gam_solve::estimate::EstimationError::TrialPointRefused {
                reason: "survival smoothing LAML cost was non-finite".to_string(),
            });
        }
        // A value probe's evaluator returns a placeholder gradient, which is not kept.
        let grad = wants_gradient.then_some(grad_full);
        // Also rho-local, and again the layer below agrees:
        // `rho_optimizer/objective.rs` `validate_outer_first_order` returns
        // `ObjectiveEvalError::recoverable` for a non-finite outer gradient.
        // `InvalidInput` was the one grading that short-circuits the seed
        // cascade (#2531/#2590).
        if grad
            .as_ref()
            .is_some_and(|gradient| gradient.iter().any(|g| !g.is_finite()))
        {
            return Err(gam_solve::estimate::EstimationError::TrialPointRefused {
                reason: "survival smoothing LAML gradient was non-finite".to_string(),
            });
        }
        *eval_cache.borrow_mut() = Some((rho_smooth.to_owned(), cost, grad.clone(), beta));
        Ok((cost, grad, hessian))
    };

    // The ρ domain is not a private `seed ± 12` box: that box was measured
    // binding on every survival fit (#2670) — the time block's λ sat at exactly
    // `seed·e¹²` with the search reporting convergence at the wall, because on
    // a log-linear baseline the second-difference penalty's honest optimum is
    // `λ → ∞` (the block collapsing to its null space). It is derived per block
    // from the exit design and the block's penalty (#2812): a search reaching
    // an edge has found a block that is unpenalized or collapsed to its null
    // space to working precision.
    let (lower, upper) = model.resolvability_rho_domain();
    if lower.len() != num_smoothing {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            format!(
                "survival smoothing domain has {} coordinates for {num_smoothing} smoothing \
                 coordinates",
                lower.len()
            ),
        ));
    }
    let context =
        format!("survival transformation smoothing-parameter selection (dim={num_smoothing})");
    // `OuterProblem::run` returns `Ok` only after its analytic projected-KKT
    // certificate accepts the selected rho. Exhaustion is
    // `EstimationError::RemlDidNotConverge`, whose rho checkpoint is preserved
    // in the error; a seed or best-so-far smoothing value is never promoted to
    // an estimator merely because a fixed-lambda inner solve was finite.
    //
    // The shared gradient-only outer route disables `opt`'s relative-stall
    // predicate. That predicate scales its stationarity band by
    // `(1 + ‖ρ‖∞)`, which is not the KKT contract for log smoothing
    // parameters and used to stop this fit before the certificate's own band.
    // Keep this caller on that single authoritative route: a refused result is
    // non-convergence, not an invitation to rebuild BFGS with an arbitrary
    // caller-owned retry budget.
    let problem = OuterProblem::new(num_smoothing)
        .with_problem_size(model.n_observations(), beta0.len())
        .with_gradient(Derivative::Analytic)
        // The analytic LAML ρ-Hessian is declared under #2359's
        // optimize-3/certify-4 lifecycle: the search stays on BFGS over the
        // analytic gradient, and the terminal mint requests
        // `ValueGradientHessian` once, so the certificate carries curvature
        // evidence and `final_hessian` holds the ρ-Hessian at the selected ρ
        // (#2912).
        .with_hessian(gam_problem::DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(true)
        .with_bounds(lower.clone(), upper.clone())
        .with_initial_rho(seed_rho.clone());
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), rho: &Array1<f64>| {
            eval_at(rho, OuterEvalOrder::Value).map(|(cost, _, _)| cost)
        },
        |_: &mut (), rho: &Array1<f64>| {
            let (cost, gradient, hessian) = eval_at(rho, OuterEvalOrder::ValueAndGradient)?;
            Ok(match gradient {
                Some(gradient) => OuterEval {
                    cost,
                    gradient,
                    hessian,
                    inner_beta_hint: None,
                },
                None => OuterEval::value_only(cost, rho.len(), None),
            })
        },
        |_: &mut (), rho: &Array1<f64>, order: OuterEvalOrder| {
            let (cost, gradient, hessian) = eval_at(rho, order)?;
            Ok(match gradient {
                Some(gradient) => OuterEval {
                    cost,
                    gradient,
                    hessian,
                    inner_beta_hint: None,
                },
                None => OuterEval::value_only(cost, rho.len(), None),
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
    let result = problem.run(&mut obj, &context).map_err(FitFailure::from)?;
    let outer_iterations = result.iterations;
    let criterion_certificate = result.criterion_certificate;
    let outer_hessian = result.final_hessian;
    let outer_gradient = result
        .final_measurement
        .map(gam_solve::rho_optimizer::OuterFirstOrderMeasurement::into_gradient);
    let selected_rho = result.rho;
    if selected_rho.len() != num_smoothing {
        return Err(FitFailure::raised(
            gam_problem::FailureCategory::Invariant,
            format!(
                "survival transformation smoothing selector returned {} coordinates for \
                 {num_smoothing} smoothing parameters; selected-rho checkpoint={:?}",
                selected_rho.len(),
                selected_rho.to_vec(),
            ),
        ));
    }
    let lambdas = gam_problem::checked_exp_log_strengths(selected_rho.iter().copied())
        .map_err(|error| {
            FitFailure::raised(
                gam_problem::FailureCategory::Numerical,
                format!("survival transformation selected rho: {error}"),
            )
        })?;
    Ok(Some(SurvivalSmoothingSelection {
        lambdas,
        outer_iterations,
        criterion_certificate,
        certified_mode: warm_beta.into_inner(),
        outer_hessian,
        outer_gradient,
    }))
}

/// Conditional Bayesian covariance `Vb = H⁻¹` for a converged single-cause
/// transformation/weibull survival fit (unit dispersion; #2373 defect C).
///
/// Returns `None` when the penalized Hessian is not SPD (or the inverse is
/// non-finite), matching the location-scale reduced-parametric path's
/// typed-absence semantics (`survival/location_scale/fit.rs`): predict then
/// errors honestly on a covariance-requiring mode instead of consuming a
/// fabricated matrix. The Hessian is already in the raw block coordinates β
/// lives in (identity coefficient gauge), so the inverse needs no gauge lift.
fn survival_conditional_covariance_from_penalized_hessian(
    penalized_hessian: &Array2<f64>,
) -> Option<Array2<f64>> {
    use gam_linalg::faer_ndarray::FaerCholesky;

    let p = penalized_hessian.nrows();
    let identity = Array2::<f64>::eye(p);
    let cov = match penalized_hessian.cholesky(faer::Side::Lower) {
        Ok(chol) => chol.solve_mat(&identity),
        Err(_) => return None,
    };
    if !cov.iter().all(|v| v.is_finite()) {
        return None;
    }
    // Symmetrize away round-off so the persisted conditional covariance is
    // exactly symmetric, as a covariance must be.
    let mut symm = cov.clone();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (cov[[i, j]] + cov[[j, i]]);
            symm[[i, j]] = avg;
            symm[[j, i]] = avg;
        }
    }
    Some(symm)
}

fn survival_unified_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    summary: &gam_solve::pirls::WorkingModelPirlsResult,
    state: &gam_solve::pirls::WorkingState,
    training_sample_size: usize,
    penalty_blocks: &[PenaltyBlock],
    // OUTER convergence evidence from the smoothing selection (#2301 defect D):
    // the real outer-iteration count (0 when no smoothing coordinate was
    // optimized) and the analytic stationarity certificate `OuterProblem::run`
    // minted. `summary` is the INNER PIRLS result, so its iteration count and
    // gradient norm are inner quantities and must NOT be used for the outer.
    outer_iterations: usize,
    criterion_certificate: Option<gam_solve::estimate::OuterCriterionCertificate>,
    // The analytic LAML ρ-Hessian at the selected ρ: the curvature the
    // smoothing correction inverts (#2912). `None` on the fixed-outer path.
    outer_hessian: Option<Array2<f64>>,
    // The outer gradient at the selected ρ, whose certificate floor decides
    // which ρ directions the correction resolves (#2346).
    outer_gradient: Option<Array1<f64>>,
) -> Result<UnifiedFitResult, String> {
    if state.eta.len() != training_sample_size {
        return Err(format!(
            "survival transformation state has {} rows but the training data has {training_sample_size}",
            state.eta.len()
        ));
    }
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
    // #2301 defect E: the `UnifiedFitResult` invariant requires
    // `exp(log_lambdas) == lambdas` BIT-exactly (the validator round-trips
    // `checked_exp_log_strength(log_λ)` against `λ`), but `ln` then `exp` is not
    // bit-stable, so deriving `log_lambdas = ln(λ)` from the raw penalty-block λ
    // fails the round-trip. log-λ (= ρ) is the canonical source — the outer
    // optimizer works in ρ-space — so re-derive `λ = exp(log_λ)` here to make the
    // two fields bit-consistent (a ≤1-ulp change to the stored λ). This was masked
    // until the certificate-wiring fix (defect D) let assembly reach the invariant.
    let lambdas = Array1::from_vec(
        log_lambdas
            .iter()
            .copied()
            .enumerate()
            .map(|(coordinate, log_value)| {
                gam_problem::checked_exp_log_strength(log_value).map_err(|error| {
                    format!("survival fit log-lambda coordinate {coordinate}: {error}")
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
    let (edf_total, edf_by_block, penalty_block_trace, edf_rank_bound, penalized_hessian) =
        survival_transformation_edf(state, penalty_blocks)?;
    assert_eq!(edf_by_block.len(), lambdas.len());
    assert_eq!(penalty_block_trace.len(), lambdas.len());

    // #2373 defect C: a converged single-cause transformation/weibull survival
    // fit carries the full observed-information penalized Hessian
    // `H = X'W_H X + S(λ)` — the very matrix the EDF trace-solves above already
    // factor — so the conditional Bayesian covariance `Vb = H⁻¹` (unit
    // dispersion) is available and MUST be persisted. Without it `predict()`
    // refuses every covariance-requiring mode with "fit result does not contain
    // conditional covariance" (survival/predict.rs). The coefficient gauge here
    // is the identity (`Gauge::identity(&[beta.len()])` below), so `H⁻¹` is
    // already in the raw block coordinates β lives in — no gauge lift is needed
    // (unlike the location-scale path's `lift_conditional_covariance`). Mirror
    // that path's PD-failure semantics (survival/location_scale/fit.rs): an SPD
    // Hessian yields the symmetrized inverse; a non-SPD one mints the fit with a
    // typed-absent covariance so predict refuses honestly rather than consuming
    // a fabricated nearby matrix.
    let covariance_conditional =
        survival_conditional_covariance_from_penalized_hessian(&penalized_hessian);
    // Standard errors derive from this matrix (#2955) under the one gate that
    // owns the negative-diagonal judgement (`gam_problem::se_from_covariance`),
    // not a local `max(0, ·)`. A clamp reports a materially negative variance as
    // `SE = 0` — an infinitely precise coefficient — where the shared gate
    // refuses anything outside its dimension-scaled backward-error bound.
    covariance_conditional
        .as_ref()
        .map(gam_problem::se_from_covariance)
        .transpose()
        .map_err(|reason| {
            format!("survival transformation conditional standard errors are invalid: {reason}")
        })?;
    let penalized_hessian = gam_problem::dispersion_cov::UnscaledPrecision::wrap(penalized_hessian);

    // #2627: on the FIXED-lambda survival path, lambda is a CONSTANT of the model
    // rather than an estimate, so `Vp = Vb` EXACTLY and the corrected covariance
    // must be persisted too.
    //
    // The caller sets `outer_iterations = 0` with no certificate on exactly one
    // branch, and says so there: "No smoothing coordinate was optimized (e.g.
    // the fixed-lambda parametric Weibull baseline path): the fit is
    // fixed-outer". With no rho estimated there is no rho-variance to integrate
    // over, so the smoothing correction `J*V_rho*J'` is the zero matrix and
    // `Vp = Vb + 0 = Vb`. This is an identity of the definition, not a fallback
    // to a weaker uncertainty object -- it is the same one `gam-predict`'s
    // `select_uncertainty_backend` applies when `lambdas.is_empty()`, stated on
    // the ESTIMATOR instead of on the coordinate count, which is what this path
    // needs: it carries penalty blocks (so `lambdas` is non-empty) whose lambdas
    // were never selected.
    //
    // What the absence cost: `gam predict` defaults to `--mode posterior-mean
    // --covariance-mode corrected`, so EVERY interval request on a fixed-lambda
    // survival model died with "saved model does not contain smoothing-corrected
    // covariance; refit before requesting --covariance-mode corrected" -- an
    // instruction no refit could satisfy, because there was no correction to
    // compute. The sibling conditional covariance right above was persisted for
    // the same reason under #2373; this is the other half of that fix.
    //
    // A fit whose lambda WAS selected publishes `Vp = Vb + C`, the first-order
    // rho-uncertainty correction `C = A·V_ρ·Aᵀ`, `A = Vb·U`, from the analytic
    // LAML rho-Hessian the certificate judged (#2912). Column k of `U` is
    // `∂(S_λ β̂)/∂ρ_k = λ_k S_k β̂` over block k's range, in the identity-gauge
    // coordinates `Vb` lives in, and railed coordinates carry no finite
    // rho-variance (#2337 Thm 2.3); the custom-family mint goes through the same
    // helper, which drops directions under the certificate's gradient floor. A
    // refused interior keeps the typed absence: handing back `Vb` under a
    // corrected request would silently under-report every interval.
    let lambda_is_fixed = outer_iterations == 0 && criterion_certificate.is_none();
    let (smoothing_corrected, smoothing_correction_absence) = if lambda_is_fixed {
        (None, None)
    } else {
        match (
            covariance_conditional.as_ref(),
            outer_hessian.as_ref(),
            criterion_certificate.as_ref(),
        ) {
            (Some(v_cond), Some(outer_hessian), Some(certificate)) => {
                let mut excluded: Vec<usize> = certificate.lambdas_railed.clone();
                for rail in certificate.stationarity.rails() {
                    if !excluded.contains(&rail.index) {
                        excluded.push(rail.index);
                    }
                }
                let mut u_mat = Array2::<f64>::zeros((beta.len(), lambdas.len()));
                for (coordinate, block) in penalty_blocks.iter().enumerate() {
                    let s_beta = block.matrix.dot(&beta.slice(s![block.range.clone()]));
                    u_mat
                        .slice_mut(s![block.range.clone(), coordinate])
                        .scaled_add(lambdas[coordinate], &s_beta);
                }
                let no_gradient = Array1::<f64>::zeros(0);
                match gam_custom_family::first_order_smoothing_correction(
                    v_cond,
                    &u_mat,
                    outer_hessian,
                    outer_gradient.as_ref().unwrap_or(&no_gradient),
                    &excluded,
                )
                .map_err(|reason| {
                    format!("survival transformation smoothing correction: {reason}")
                })? {
                    Ok((correction, active_rank)) => (
                        Some((
                            correction,
                            gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                                active_rank,
                                rho_dimension: lambdas.len(),
                            },
                        )),
                        None,
                    ),
                    Err(absence) => (None, Some(absence)),
                }
            }
            (Some(_), None, _) => {
                log::debug!(
                    "[smoothing-correction] branch=unavailable reason=outer-hessian-not-published \
                     rho_dimension={}",
                    lambdas.len(),
                );
                (
                    None,
                    Some(gam_solve::model_types::SmoothingCorrectionAbsence::OuterHessianUndeclared {
                        reason: gam_solve::model_types::OuterHessianAbsence::NotPublished,
                    }),
                )
            }
            _ => (None, None),
        }
    };
    let covariance_corrected = if lambda_is_fixed {
        covariance_conditional.clone()
    } else {
        smoothing_corrected
            .as_ref()
            .zip(covariance_conditional.as_ref())
            .map(|((correction, _), v_cond)| v_cond + correction)
    };
    covariance_corrected
        .as_ref()
        .map(gam_problem::se_from_covariance)
        .transpose()
        .map_err(|reason| {
            format!("survival transformation corrected standard errors are invalid: {reason}")
        })?;

    let inference = gam_solve::estimate::FitInference {
        edf_by_block: edf_by_block.clone(),
        penalty_block_trace,
        edf_rank_bound,
        edf_total,
        // This lane computes only the first-order correction, so its retained
        // first-order pair is its primary pair, as on the custom-family lane.
        smoothing_correction: smoothing_corrected
            .as_ref()
            .map(|(correction, _)| correction.clone()),
        smoothing_correction_method: smoothing_corrected.as_ref().map(|(_, method)| *method),
        smoothing_correction_first_order: smoothing_corrected
            .as_ref()
            .map(|(correction, _)| correction.clone()),
        smoothing_correction_method_first_order: smoothing_corrected
            .as_ref()
            .map(|(_, method)| *method),
        smoothing_correction_absence,
        penalized_hessian: penalized_hessian.clone(),
        reparam_qs: None,
        dispersion: gam_solve::estimate::Dispersion::UNIT,
        factorized_standard_errors: None,
        beta_covariance_frequentist: None,
        coefficient_influence: None,
        weighted_gram: None,
        identified_subspace: None,
    };

    UnifiedFitResult::try_from_parts(gam_solve::estimate::UnifiedFitResultParts {
        blocks: vec![gam_solve::estimate::FittedBlock {
            beta: beta.clone(),
            role: gam_problem::BlockRole::Mean,
            edf: edf_total,
            lambdas: lambdas.clone(),
        }],
        training_sample_size,
        log_lambdas,
        lambdas,
        likelihood_family: Some(LikelihoodSpec::royston_parmar()),
        likelihood_scale: gam_problem::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: gam_problem::LogLikelihoodNormalization::UserProvided,
        log_likelihood: state.log_likelihood,
        deviance: state.deviance,
        reml_score: Some(reml_score),
        stable_penalty_term: state.penalty_term,
        penalized_objective: Some(reml_score),
        used_device: false,
        // The OUTER counts come from the smoothing selection, NOT the inner PIRLS
        // `summary` (#2301 defect D). When a certificate is present its projected
        // stationarity residual is the outer gradient; a fixed-outer fit (no
        // smoothing coordinate) reports the inner residual as a diagnostic only.
        outer_iterations,
        outer_converged: true,
        outer_gradient_norm: criterion_certificate
            .as_ref()
            .map(|certificate| certificate.stationarity.projected_norm())
            .or(Some(summary.lastgradient_norm)),
        standard_deviation: 1.0,
        covariance_conditional,
        covariance_corrected,
        inference: Some(inference),
        fitted_link: FittedLinkState::Standard(None),
        geometry: Some(gam_solve::estimate::FitGeometry {
            coefficient_gauge: gam_problem::gauge::Gauge::identity(&[beta.len()]),
            penalized_hessian,
            constrained_posterior: None,
            working: None,
        }),
        block_states: Vec::new(),
        pirls_status: summary.status,
        max_abs_eta: summary.max_abs_eta,
        constraint_kkt: None,
        artifacts: gam_solve::estimate::FitArtifacts {
            pirls: None,
            // Thread the outer analytic stationarity certificate so assembly can
            // certify the outer optimum (#2301 defect D). `None` here with
            // `outer_iterations == 0` is a fixed-outer fit, which assembly accepts
            // as `Fixed` evidence.
            criterion_certificate,
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
    persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
) -> Result<SurvivalTransformationFitResult, FitFailure> {
    let cause_count = crate::survival::cause_count_from_event_codes(spec.event_target.view())
        .map_err(|err| FitFailure::raised(err.failure_category(), err.to_string()))?;
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

    // The joint designs are `[time | covariates]` at entry and exit, and `[time | 0]`
    // for the time derivative, where covariates do not enter. Each is written once,
    // straight from the time design's row chunks, and charged on the memory governor's
    // ledger for the life of the fit. Every cause's family block, block spec and channel
    // Jacobian shares them. C causes used to hold 5C + 6 row-scaled copies outside the
    // ledger (#2900).
    let joint_design_charge = gam_runtime::resource::MemoryGovernor::global()
        .try_reserve_dense_f64_copies(n, p, 3, "cause-specific survival joint designs")
        .map_err(|error| {
            FitFailure::input(format!(
                "cause-specific survival: refusing three {n}x{p} joint designs: {error}"
            ))
        })?;
    // The joint designs are assembled from the time and covariate designs above.
    let x_entry = std::sync::Arc::new(
        joint_time_covariate_design(&prepared.time_design_entry, Some(dense_cov_design), p)
            .map_err(FitFailure::invariant)?,
    );
    let x_exit = std::sync::Arc::new(
        joint_time_covariate_design(&prepared.time_design_exit, Some(dense_cov_design), p)
            .map_err(FitFailure::invariant)?,
    );
    let x_derivative = std::sync::Arc::new(
        joint_time_covariate_design(&prepared.time_design_derivative_exit, None, p)
            .map_err(FitFailure::invariant)?,
    );

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
            x_entry: std::sync::Arc::clone(&x_entry),
            x_exit: std::sync::Arc::clone(&x_exit),
            x_derivative: std::sync::Arc::clone(&x_derivative),
            offset_eta_entry: prepared.eta_offset_entry.clone() + &spec.covariate_offset,
            offset_eta_exit: prepared.eta_offset_exit.clone() + &spec.covariate_offset,
            offset_derivative_exit: prepared.derivative_offset_exit.clone(),
            derivative_floor,
            // Non-Weibull survival uses the structural monotone I-spline time
            // basis (`set_structural_monotonicity` above), so its leading
            // `p_time_total` columns get the domain-wide coefficient cone
            // `β_j ≥ 0`. The parametric Weibull `log t` baseline is not an
            // I-spline monotone block, so it carries no structural cone here.
            structural_time_columns: if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull {
                0
            } else {
                p_time_total
            },
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
                    FitFailure::numerical(format!(
                        "cause-specific survival penalty {penalty_idx} strength: {error}"
                    ))
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
            design: std::sync::Arc::clone(&x_exit),
            own_output: cause,
            n_family_outputs: cause_count,
        });
        block_specs.push(ParameterBlockSpec {
            name: format!("time_cause_{}", cause + 1),
            design: gam_linalg::matrix::DesignMatrix::from(std::sync::Arc::clone(&x_exit)),
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

    // Its endpoint, block and constraint-size refusals are different kinds.
    let family = crate::survival::CauseSpecificRoystonParmarFamily::new(family_blocks)
        .map_err(FitFailure::unclassified)?;
    let fit_options = BlockwiseFitOptions {
        // Joint posterior prediction and CIF uncertainty consume the complete
        // cross-cause conditional covariance. Computing it here is part of the
        // fitted competing-risks model contract; reconstructing independent
        // per-cause approximations at prediction time would discard the
        // cross-cause blocks and misstate CIF uncertainty (#2298).
        compute_covariance: true,
        persistent_warm_start_store,
        ..Default::default()
    };
    let rho_prior = cause_specific_survival_rho_prior(
        cause_count,
        penalty_blocks.len(),
        penalty_block_gamma_priors,
    )
    // The Gamma precision hyperpriors are the caller's configuration.
    .map_err(FitFailure::input)?;
    let mut fit = crate::custom_family::fit_custom_family_arming_on_evidence_with_rho_prior(
        &family,
        &block_specs,
        &fit_options,
        rho_prior,
    )
        .map_err(|err| {
            FitFailure::from(err).context("cause-specific survival custom-family fit failed")
        })?;
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
    // null. For Weibull-without-timewiggle the time basis is the single-column
    // `log t` linear basis whose per-cause coefficient carries the
    // log-cumulative-hazard slope. #2301 dropped the redundant `[1, ·]` constant
    // column (it was exactly confounded with the covariate intercept, which
    // absorbs the Weibull location, and left the penalized Hessian singular), so
    // the fitted scale is the identified anchor (`scale = anchor`) and the shape
    // is the sole slope coefficient `beta[0]` (issue #899). The shared
    // `SurvivalBaselineConfig` holds a
    // single (scale, shape), so we report the first cause's fitted baseline as
    // the representative shared value — the same pooled-baseline convention the
    // seed used, but post-fit rather than uninitialized.
    let fitted_baseline_cfg = if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull
        && spec.timewiggle.is_none()
    {
        let first_block = fit.blocks.first().ok_or_else(|| {
            FitFailure::invariant("cause-specific survival fit produced no coefficient blocks")
        })?;
        let time_beta = first_block
            .beta
            .slice(s![..spec.time_build.x_exit_time.ncols()])
            .to_owned();
        fitted_weibull_baseline_from_linear_time_beta(&time_beta, spec.time_anchor).ok_or_else(|| {
            FitFailure::numerical(
                "failed to recover fitted Weibull scale/shape from the cause-specific linear time coefficients",
            )
        })?
    } else {
        baseline_cfg
    };
    drop(joint_design_charge);
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

/// `[time | covariates]`, or `[time | 0]` when `covariates` is `None`, written straight
/// from the time design's row chunks, with no dense copy of the time design in between.
fn joint_time_covariate_design(
    time_design: &gam_linalg::matrix::DesignMatrix,
    covariates: Option<&Array2<f64>>,
    p: usize,
) -> Result<Array2<f64>, String> {
    let n = time_design.nrows();
    let p_time = time_design.ncols();
    let mut joint = Array2::<f64>::zeros((n, p));
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p_time, n);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        time_design
            .row_chunk_into(start..end, joint.slice_mut(s![start..end, ..p_time]))
            .map_err(|error| format!("survival time design rows {start}..{end}: {error}"))?;
    }
    if let Some(covariates) = covariates {
        joint.slice_mut(s![.., p_time..]).assign(covariates);
    }
    Ok(joint)
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
) -> Result<(), String> {
    // The byte stream `hash_workflow_array2` writes for the dense matrix, read one row
    // chunk at a time so the key never densifies the design (#2900).
    let n = matrix.nrows();
    let p = matrix.ncols();
    hasher.write_usize(n);
    hasher.write_usize(p);
    let chunk_rows = gam_runtime::resource::byte_balanced_row_chunk(p, n);
    let mut chunk = Array2::<f64>::zeros((chunk_rows.min(n), p));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let mut rows = chunk.slice_mut(s![..end - start, ..]);
        matrix
            .row_chunk_into(start..end, rows.view_mut())
            .map_err(|error| format!("survival warm-start key design rows {start}..{end}: {error}"))?;
        for row in rows.rows() {
            for &value in row {
                hasher.write_f64(value);
            }
        }
    }
    Ok(())
}

fn survival_transformation_log_lambdas(
    penalty_blocks: &[crate::survival::PenaltyBlock],
) -> Result<Vec<f64>, FitFailure> {
    penalty_blocks
        .iter()
        .enumerate()
        .map(|(coordinate, block)| {
            gam_problem::checked_log_strength(block.lambda).map_err(|error| {
                FitFailure::raised(
                    gam_problem::FailureCategory::Numerical,
                    format!("survival transformation penalty {coordinate}: {error}"),
                )
            })
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
) -> Result<String, String> {
    let mut hasher = gam_runtime::warm_start::Fingerprinter::new();
    hasher.write_str("gamfit-persistent-survival-transformation-working-pirls");
    // Use the cache schema tag (NOT CARGO_PKG_VERSION) so routine
    // library version bumps don't invalidate users' on-disk warm-start
    // caches.
    hasher.write_str(&gam_solve::persistent_warm_start::cache_schema_tag());
    hasher.write_str(&format!("{:?}", spec.likelihood_mode));
    hasher.write_f64(spec.time_anchor);
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
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_entry)?;
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_exit)?;
    hash_workflow_design_matrix(&mut hasher, &prepared.time_design_derivative_exit)?;
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
    hasher.write_bool(opts.firth_bias_reduction);
    hasher.write_bool(opts.coefficient_lower_bounds.is_some());
    if let Some(bounds) = opts.coefficient_lower_bounds.as_ref() {
        hash_workflow_array_view(&mut hasher, bounds.view());
    }
    hasher.write_bool(opts.linear_constraints.is_some());
    Ok(format!("surv-transform-{}", hasher.finish_hex()))
}

fn load_survival_transformation_persistent_warm_start(
    store: &gam_runtime::warm_start::ConfiguredWarmStartStore,
    key: &str,
    spec: &SurvivalTransformationTermSpec,
    n_cols: usize,
    rho: &[f64],
) -> Option<(Array1<f64>, Option<f64>)> {
    let record = gam_solve::persistent_warm_start::load_record(store, key)?;
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
    log::debug!("[warm-start-cache] restored survival transformation warm start key={key}");
    let lm_lambda = record
        .last_pirls_lm_lambda
        .filter(|value| value.is_finite() && *value > 0.0);
    Some((Array1::from_vec(record.beta), lm_lambda))
}

fn store_survival_transformation_persistent_warm_start(
    store: &gam_runtime::warm_start::ConfiguredWarmStartStore,
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
    gam_solve::persistent_warm_start::store_record(store, &record);
    gam_solve::persistent_warm_start::load_record(store, &record.key).is_some_and(|stored| {
        stored.rho == record.rho
            && stored.beta == record.beta
            && stored.last_inner_iters == record.last_inner_iters
            && stored.last_inner_converged == record.last_inner_converged
    })
}

pub(crate) fn fit_survival_transformation_model(
    request: SurvivalTransformationFitRequest<'_>,
) -> Result<SurvivalTransformationFitResult, FitFailure> {
    use crate::survival::{PenaltyBlock, PenaltyBlocks, SurvivalMonotonicityPenalty, SurvivalSpec};

    let SurvivalTransformationFitRequest {
        data,
        spec,
        persistent_warm_start_store,
    } = request;
    let mut baseline_cfg = spec.baseline_cfg.clone();
    let covariate_design = build_term_collection_design(data, &spec.covariate_spec)
        .map_err(|err| {
            let reason = err.to_string();
            FitFailure::raised(
                gam_solve::estimate::EstimationError::from(err).failure_category(),
                reason,
            )
        })?;
    let resolvedspec = crate::fit_orchestration::drivers::freeze_term_collection_from_design(
        &spec.covariate_spec,
        &covariate_design,
    )
    .map_err(FitFailure::from)?;
    // Densified once, on the governor's ledger, and shared by every working model
    // the baseline search builds rather than copied into each one (#2900).
    let dense_cov_design = std::sync::Arc::new(
        covariate_design
            .design
            // A design the process cannot hold is refused by its size (#2937).
            .try_to_dense_by_chunks("survival transformation covariate design")
            .map_err(FitFailure::input)?,
    );
    let p_cov = dense_cov_design.ncols();
    let cause_count = crate::survival::cause_count_from_event_codes(spec.event_target.view())
        .map_err(|err| FitFailure::raised(err.failure_category(), err.to_string()))?;
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
            )
            // Baseline offsets, the derivative guard and the time wiggle each
            // refuse for a different reason (#2937).
            .map_err(FitFailure::unclassified)?;
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
            // Each block's REML starting strength is its natural scale: the log
            // ratio of its design's mean Gram diagonal to the penalty's mean
            // diagonal. It is a starting point for the outer search, never an
            // estimate.
            if let Some(time_log_lambdas) = prepared.time_initial_log_lambdas.as_ref() {
                for (idx, penalty) in prepared.time_penalties.iter().enumerate() {
                    if penalty.nrows() == p_time_total && penalty.ncols() == p_time_total {
                        penalty_blocks.push(PenaltyBlock {
                            matrix: penalty.clone(),
                            lambda: time_log_lambdas[idx].exp(),
                            range: 0..p_time_total,
                            nullspace_dim: prepared
                                .time_nullspace_dims
                                .get(idx)
                                .copied()
                                .unwrap_or(0),
                        });
                    }
                }
            }
            // Covariate-smooth penalties (e.g. `s(x)`, `s(group, bs="re")`
            // frailty) live in the covariate term-collection design; the survival
            // transformation fit stacks the covariate columns at
            // `p_time_total..p`, so each covariate penalty's local block maps to
            // the joint range `p_time_total + col_range`. Penalizing them here is
            // what lets the frailty / covariate smooths shrink; like the time
            // blocks they are REML-selected smoothing blocks (issues #563/#565).
            // Only zero-prior-mean blocks are admissible as a plain quadratic
            // `λ βᵀSβ`; a non-zero centering would need an offset the survival
            // PenaltyBlock does not model, so such a block is not mis-applied
            // and its columns stay unpenalized.
            for block in crate::survival::covariate_penalty_blocks(
                &covariate_design.penalties,
                &covariate_design.nullspace_dims,
                p_cov,
                p_time_total,
            ) {
                // A seed refuses a block with no usable Gram scale, a degenerate
                // design the caller's data produced (#2937).
                let log_lambda = crate::survival::marginal_slope::block_log_lambda_seeds(
                    &covariate_design.design,
                    [block.matrix],
                )
                .map_err(FitFailure::input)?[0];
                penalty_blocks.push(PenaltyBlock {
                    matrix: block.matrix.clone(),
                    lambda: log_lambda.exp(),
                    range: block.range,
                    nullspace_dim: block.nullspace_dim,
                });
            }
            // The penalty set is exactly the time + covariate smoothing blocks
            // above, every one of them REML-selected. No fixed-λ identity ridge is
            // appended (#2670): such a ridge was a penalty on the coefficients, not
            // on the fitted function, at a strength no criterion chose, and it
            // entered the fit's objective, its LAML normalizer, its edf and its
            // posterior covariance. `WorkingModelSurvival::update_state` states the
            // invariant this relies on: indefinite or rank-deficient curvature
            // along the Newton path is the solver's Levenberg–Marquardt damping's
            // problem, so the converged estimator is a stationary point of the
            // exact penalized likelihood. A design whose likelihood does not
            // identify a coefficient direction is then refused (the LAML Hessian
            // is not positive definite) instead of being silently pinned by a
            // `1e-6` prior — a construction defect belongs to the identifiability
            // audit, not to the objective.
            // The time designs are densified on the governor's ledger and moved into
            // the working model, and the covariate design is shared. The model used
            // to copy each view again on every baseline-search evaluation (#2900).
            let dense_time_entry = prepared
                .time_design_entry
                .try_to_dense_by_chunks("survival transformation entry time design")
                .map_err(FitFailure::input)?;
            let dense_time_exit = prepared
                .time_design_exit
                .try_to_dense_by_chunks("survival transformation exit time design")
                .map_err(FitFailure::input)?;
            let dense_time_derivative = prepared
                .time_design_derivative_exit
                .try_to_dense_by_chunks("survival transformation derivative time design")
                .map_err(FitFailure::input)?;
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
                        time_entry: dense_time_entry,
                        time_exit: dense_time_exit,
                        time_derivative: dense_time_derivative,
                        covariates: std::sync::Arc::clone(&dense_cov_design),
                        monotonicity_constraint_rows: None,
                        monotonicity_constraint_offsets: None,
                        eta_offset_entry: Some(eta_offset_entry.view()),
                        eta_offset_exit: Some(eta_offset_exit.view()),
                        derivative_offset_exit: Some(prepared.derivative_offset_exit.view()),
                    },
                )
                .map_err(|err| {
                    FitFailure::raised(
                        err.failure_category(),
                        format!("failed to construct survival model: {err}"),
                    )
                })?;
            if spec.likelihood_mode != SurvivalLikelihoodMode::Weibull {
                model
                    .set_structural_monotonicity(true, p_time_total)
                    .map_err(|err| {
                        FitFailure::from(err).context("failed to enable structural monotonicity")
                    })?;
            }
            let mut beta0 = Array1::<f64>::zeros(p);
            if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull && spec.timewiggle.is_none()
            {
                let (scale, shape) = spec
                    .weibull_seed
                    .ok_or_else(|| {
                        FitFailure::raised(
                            gam_problem::FailureCategory::Invariant,
                            "weibull survival fit missing scale/shape seed",
                        )
                    })?;
                // #2301: the built-in Weibull time basis is now a single `log t`
                // column carrying the shape. The `−shape·log_scale` LOCATION that
                // the dropped constant column used to seed is folded into the mean
                // intercept instead. This REQUIRES the covariate block to carry an
                // intercept to absorb the location. A formula whose terms leave no
                // model constant (`~ x - 1` with nothing spanning 1, i.e.
                // `ModelLevel::NoIntercept`) gives the location no home, so the fit
                // refuses below rather than mis-seed a singular direction.
                if p_time_total < 1 {
                    return Err(FitFailure::raised(
                        gam_problem::FailureCategory::Invariant,
                        format!(
                            "weibull built-in time basis has {p_time_total} columns but needs 1 for the shape"
                        ),
                    ));
                }
                if covariate_design.intercept_range.is_empty() {
                    return Err(FitFailure::raised(
                        gam_problem::FailureCategory::Input,
                        "weibull survival fit requires a mean intercept to carry the baseline \
                         location, but the covariate design has none (the formula removes the \
                         intercept, e.g. `~ x - 1`, and no term spans the constant)",
                    ));
                }
                beta0[0] = shape;
                let intercept_col = p_time_total + covariate_design.intercept_range.start;
                beta0[intercept_col] = -shape * scale.ln();
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
            Ok::<_, FitFailure>((prepared, penalty_blocks, beta0, structural_lower_bounds, model))
        };

    if let Some(direct_sum) = crate::survival::construction::weibull_scaffold_direct_sum(
        &baseline_cfg,
        &spec.time_build,
        true,
        spec.timewiggle.is_some(),
        spec.covariate_spec.level,
    ) {
        // The Weibull scaffold lies in span{1, log t} = the location's constant
        // plus a flat, cone-interior direction of the I-spline time block, so
        // the fit's model space is exactly the Linear-target one and there is
        // no θ to search (Thm 5.1; see `weibull_scaffold_direct_sum`).
        baseline_cfg = direct_sum;
    } else if baseline_cfg.target != SurvivalBaselineTarget::Linear {
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
        // on baseline_chain_rule_gradient. Only scaffolds outside the Weibull
        // direct sum reach this search (Gompertz / Gompertz-Makeham shapes, a
        // time-wiggle, a non-I-spline time basis, or no model constant), where θ
        // is not absorbed by the time block.
        // The search takes text (survival construction), so a candidate's
        // failure is kept typed here (#2937). The outer engine never retries a
        // thrown objective error: it ends the search, so the kept failure is the
        // one that stopped it.
        let candidate_failure = std::cell::RefCell::new(None::<FitFailure>);
        let stop_on = |failure: FitFailure| {
            let reason = failure.to_string();
            *candidate_failure.borrow_mut() = Some(failure);
            reason
        };
        baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            spec.age_exit.view(),
            "workflow survival transformation baseline",
            |candidate| {
                let (_, _, beta0, structural_lower_bounds, mut model) =
                    build_working_model(candidate).map_err(stop_on)?;
                let opts = gam_solve::pirls::WorkingModelPirlsOptions {
                    max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
                    convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
                    adaptive_kkt_tolerance: None,
                    max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
                    firth_bias_reduction: false,
                    coefficient_lower_bounds: structural_lower_bounds,
                    linear_constraints: None,
                    initial_lm_lambda: None,
                };
                // The candidate is the search's own point on its domain.
                let parameter_checkpoint = survival_baseline_parameter_checkpoint(candidate)
                    .map_err(|reason| stop_on(FitFailure::invariant(reason)))?;
                let summary = gam_solve::pirls::runworking_model_pirls(
                    &mut model,
                    gam_problem::Coefficients::new(beta0),
                    &opts,
                    Some(&mut |info: &gam_solve::pirls::WorkingModelIterationInfo| {
                        log::trace!(
                            "[SURV-BASELINE pirls] parameter_checkpoint={:?} iter={} \
                                 deviance={:.6e} |grad|={:.6e} step={:.3e} halvings={}",
                            parameter_checkpoint,
                            info.iteration,
                            info.deviance,
                            info.gradient_norm,
                            info.step_size,
                            info.step_halving
                        );
                    }),
                )
                .map_err(|error| {
                    stop_on(
                        FitFailure::from(error)
                            .context(format!(
                                "survival baseline PIRLS failed at parameter_checkpoint=\
                                 {parameter_checkpoint:?}"
                            ))
                            .annotated("no fit was minted"),
                    )
                })?;
                require_certified_survival_pirls(
                    &summary,
                    "survival transformation baseline profile",
                    &parameter_checkpoint,
                    None,
                )
                .map_err(|reason| {
                    stop_on(FitFailure::raised(gam_problem::FailureCategory::Convergence, reason))
                })?;
                let beta = summary.beta.as_ref().to_owned();
                let state = model.update_state(&beta).map_err(|err| {
                    stop_on(FitFailure::from(err).context("failed to evaluate survival baseline candidate"))
                })?;
                let cost = state.penalized_objective();
                let residuals = model.offset_channel_residuals(&beta).map_err(|err| {
                    stop_on(
                        FitFailure::from(err)
                            .context("failed to form survival baseline offset residuals"),
                    )
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
                )
                .map_err(|reason| stop_on(FitFailure::unclassified(reason)))?
                .ok_or_else(|| {
                    stop_on(FitFailure::invariant(
                        "workflow survival transformation baseline unexpectedly has no theta gradient",
                    ))
                })?;
                Ok((cost, gradient))
            },
        )
        // A candidate that stopped the search raises its own failure. Otherwise
        // the search's typed verdict, or its configuration refusal, stands.
        .map_err(|search| candidate_failure.take().unwrap_or_else(|| FitFailure::from(search)))?;
    }

    let (prepared, mut penalty_blocks, beta0, structural_lower_bounds, mut model) =
        build_working_model(&baseline_cfg)?;
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
            persistent_warm_start_store.clone(),
        );
    }
    // REML/LAML-select the time-smoothing λ (issue #563). With λ pinned at its
    // seed the monotone I-spline baseline oversmooths toward an affine
    // log-cumulative-hazard; selecting λ from the survival LAML lets it recover
    // real curvature. The inner solve keeps the structural γ ≥ 0 box at every
    // candidate, so the constrained optimum stays valid. The selected λ is written
    // back into both the working model and `penalty_blocks` so the final fit,
    // edf, and warm-start cache all use the data-adaptive value.
    let (
        survival_outer_iterations,
        survival_outer_certificate,
        survival_outer_hessian,
        survival_outer_gradient,
        selected_mode,
    ) = if let Some(selection) = optimize_survival_transformation_smoothing(
        &model,
        &penalty_blocks,
        &beta0,
        structural_lower_bounds.as_ref(),
    )? {
        model
            .set_penalty_lambdas(&selection.lambdas)
            .map_err(FitFailure::from)?;
        for (block, &lam) in penalty_blocks.iter_mut().zip(selection.lambdas.iter()) {
            block.lambda = lam;
        }
        (
            selection.outer_iterations,
            selection.criterion_certificate,
            selection.outer_hessian,
            selection.outer_gradient,
            Some(selection.certified_mode),
        )
    } else {
        // No smoothing coordinate was optimized (e.g. the fixed-λ parametric
        // Weibull baseline path): the fit is fixed-outer, so it carries 0 outer
        // iterations and no analytic certificate — assembly reads that as
        // `Fixed` convergence evidence rather than demanding a certificate
        // (#2301 defect D).
        (0, None, None, None, None)
    };
    let opts = gam_solve::pirls::WorkingModelPirlsOptions {
        max_iterations: SURVIVAL_TRANSFORMATION_PIRLS_MAX_ITERATIONS,
        convergence_tolerance: SURVIVAL_TRANSFORMATION_PIRLS_CONVERGENCE_TOL,
        adaptive_kkt_tolerance: None,
        max_step_halving: SURVIVAL_TRANSFORMATION_PIRLS_MAX_STEP_HALVING,
        firth_bias_reduction: false,
        coefficient_lower_bounds: structural_lower_bounds,
        linear_constraints: None,
        initial_lm_lambda: None,
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
    )
    // The key hashes state this fit assembled.
    .map_err(FitFailure::invariant)?;
    let mut opts = opts;
    // The final fixed-λ solve is the inner problem at the selected ρ, which
    // the selector has just solved and certified: start from that mode so the
    // solve below is its re-certification (O(1) iterations), not a second cold
    // derivation from the structural seed. The persistent store is consulted
    // only when no selection ran (the fixed-outer paths), and the structural
    // seed only when neither exists.
    let beta_start = match selected_mode {
        Some(mode) => mode,
        None => match persistent_warm_start_store.as_ref().and_then(|store| {
            load_survival_transformation_persistent_warm_start(
                store,
                &persistent_warm_start_key,
                &spec,
                expected_beta_len,
                &rho_for_cache,
            )
        }) {
            Some((beta, lm_lambda)) => {
                opts.initial_lm_lambda = lm_lambda;
                beta
            }
            None => beta0,
        },
    };
    let summary = gam_solve::pirls::runworking_model_pirls(
        &mut model,
        gam_problem::Coefficients::new(beta_start),
        &opts,
        Some(&mut |info: &gam_solve::pirls::WorkingModelIterationInfo| {
            log::trace!(
                "[SURV-TRANS final] parameter_checkpoint={:?} iter={} deviance={:.6e} \
                     |grad|={:.6e} step={:.3e} halvings={}",
                rho_for_cache,
                info.iteration,
                info.deviance,
                info.gradient_norm,
                info.step_size,
                info.step_halving
            );
        }),
    )
    .map_err(|error| {
        FitFailure::from(error)
            .context(format!(
                "survival transformation final fixed-lambda PIRLS failed at \
                 parameter_checkpoint={rho_for_cache:?} (warm_start_key=\
                 {persistent_warm_start_key})"
            ))
            .annotated("no fit was minted")
    })?;
    let beta = summary.beta.as_ref().to_owned();
    // Persist every finite accepted iterate before enforcing the certificate:
    // an exhausted solve is resumable work, but it is never a fit. The record's
    // `last_inner_converged` bit distinguishes a final mode from a checkpoint.
    let checkpoint_persisted = persistent_warm_start_store.as_ref().is_some_and(|store| {
        store_survival_transformation_persistent_warm_start(
            store,
            &persistent_warm_start_key,
            &spec,
            expected_beta_len,
            rho_for_cache.clone(),
            &beta,
            &summary,
        )
    });
    require_certified_survival_pirls(
        &summary,
        "survival transformation final fixed-lambda PIRLS",
        &rho_for_cache,
        checkpoint_persisted.then_some(persistent_warm_start_key.as_str()),
    )
    .map_err(|reason| FitFailure::raised(gam_problem::FailureCategory::Convergence, reason))?;
    let state = model
        .update_state(&beta)
        .map_err(|err| FitFailure::from(err).context("failed to evaluate survival optimum"))?;
    let lambdas = Array1::from_iter(penalty_blocks.iter().map(|block| block.lambda));
    let fitted_baseline_cfg =
        if spec.likelihood_mode == SurvivalLikelihoodMode::Weibull && spec.timewiggle.is_none() {
            let time_beta = beta
                .slice(s![..spec.time_build.x_exit_time.ncols()])
                .to_owned();
            fitted_weibull_baseline_from_linear_time_beta(&time_beta, spec.time_anchor).ok_or_else(
                || {
                    FitFailure::raised(
                        gam_problem::FailureCategory::Numerical,
                        "failed to recover fitted Weibull scale/shape from the linear time coefficients",
                    )
                },
            )?
        } else {
            baseline_cfg
        };
    let fit = survival_unified_fit_result(
        beta,
        lambdas,
        &summary,
        &state,
        spec.age_exit.len(),
        &penalty_blocks,
        survival_outer_iterations,
        survival_outer_certificate,
        survival_outer_hessian,
        survival_outer_gradient,
    )
    // Result assembly from the fit's own state (#2937).
    .map_err(FitFailure::invariant)?;

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
) -> Result<SurvivalLocationScaleFitResult, FitFailure> {
    // Fit one coherent survival subproblem: select/apply the link-wiggle basis,
    // then solve the full penalized location-scale fit, whose outer selects the
    // inverse-link shape together with ρ (#2904).
    fn profile_survival_location_scale(
        data: ArrayView2<'_, f64>,
        spec: SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, FitFailure> {
        let mut wiggle_knots = None;
        let mut wiggle_degree = None;

        let fit = if let Some(wiggle) = wiggle {
            require_inverse_link_supports_joint_wiggle(&spec.inverse_link, "survival link wiggle")
                .map_err(|reason| {
                    FitFailure::raised(gam_problem::FailureCategory::Input, reason)
                })?;
            let mut pilot_spec = spec.clone();
            pilot_spec.linkwiggle_block = None;
            let pilot = fit_survival_location_scale_terms(data, pilot_spec, kappa_options)?;
            let selected_wiggle_basis = select_survival_link_wiggle_basis_from_pilot(
                &pilot,
                spec.age_exit.view(),
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
            inverse_link: fit.inverse_link.clone(),
            fit,
            wiggle_knots,
            wiggle_degree,
        })
    }

    let profile = profile_survival_location_scale(
        request.data,
        request.spec,
        request.wiggle,
        &request.kappa_options,
    )?;

    Ok(profile.into_result())
}

pub(crate) fn fit_bernoulli_marginal_slope_model(
    request: BernoulliMarginalSlopeFitRequest<'_>,
) -> Result<BernoulliMarginalSlopeFitResult, FitFailure> {
    fit_bernoulli_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
        &request.policy,
    )
    .map_err(FitFailure::from)
}

pub(crate) fn fit_survival_marginal_slope_model(
    request: SurvivalMarginalSlopeFitRequest<'_>,
) -> Result<SurvivalMarginalSlopeFitResult, FitFailure> {
    fit_survival_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )
}

pub(crate) fn fit_latent_survival_model(
    request: LatentSurvivalFitRequest<'_>,
) -> Result<LatentSurvivalTermFitResult, FitFailure> {
    fit_latent_survival_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

pub(crate) fn fit_latent_binary_model(
    request: LatentBinaryFitRequest<'_>,
) -> Result<LatentBinaryTermFitResult, FitFailure> {
    fit_latent_binary_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

pub(crate) fn fit_transformation_normal_model(
    request: TransformationNormalFitRequest<'_>,
) -> Result<TransformationNormalFitResult, FitFailure> {
    fit_transformation_normal(
        &request.response,
        &request.weights,
        &request.offset,
        request.data,
        &request.covariate_spec,
        &request.config,
        &request.options,
        &request.kappa_options,
    )
    .map_err(FitFailure::from)
}


#[cfg(test)]
mod survival_edf_tests {
    // The #2301 anchor gauge that used to make this Hessian singular is now
    // removed at design build (the Weibull Linear time-basis dropped its redundant
    // constant column), so `survival_edf_from_dense_hessian` is back to the exact
    // factorize + trace-solve path. The earlier rank-certified-pseudoinverse gauge
    // tests are therefore obsolete and dropped; the exact-solve contract retained
    // here is a correct exact trace on a well-conditioned Hessian. The singular-H
    // refusal (carrying the a0a9771ca flat-direction naming on the non-finite
    // trace-solve branch) is NOT unit-testable: `factorize_symmetricwith_fallback`
    // lifts any synthetic singular matrix by a √ε·‖H‖ ridge, so a small singular
    // fixture yields a finite (lifted) solve rather than the non-finite refusal the
    // real 1e12-conditioned anchor-gauge fit hit. That refusal path stays exercised
    // only by genuinely catastrophic real fits; the end-to-end healthy-fit gate is
    // the Weibull ALO regression (tests/bug_hunt_2301_diagnose_alo_multiclass_test.rs).
    use super::*;
    use crate::survival::PenaltyBlock;
    use ndarray::array;

    fn penalty_block(matrix: Array2<f64>, lambda: f64, start: usize) -> PenaltyBlock {
        let cols = matrix.ncols();
        PenaltyBlock {
            matrix,
            lambda,
            range: start..start + cols,
            nullspace_dim: 0,
        }
    }

    /// Exact penalized EDF on a well-conditioned Hessian, checked against an
    /// INDEPENDENT analytic trace (a hand-inverted 2×2 block), not the code's own
    /// solve route.
    #[test]
    fn survival_edf_exact_trace_on_well_conditioned_hessian() {
        // Block-diagonal PD H: a coupled 2×2 leading block plus an isolated
        // third direction. Penalty is the identity on the leading block.
        let h = array![[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]];
        let blocks = vec![penalty_block(array![[1.0, 0.0], [0.0, 1.0]], 1.0, 0)];

        let (edf_total, edf_by_block, penalty_block_trace, rank_bound) =
            survival_edf_from_dense_hessian(&h, &blocks).expect("PD Hessian must compute EDF");
        // #2901: `H − S` is `[[3, 1], [1, 2]] ⊕ [2]`, positive definite, so the block
        // is certified numerically.
        assert!(
            matches!(
                rank_bound[0],
                gam_solve::estimate::EdfRankBound::Certified(
                    gam_solve::estimate::EdfRankCertificate::Numerical { .. }
                )
            ),
            "{rank_bound:?}"
        );

        // Leading 2×2 block [[4,1],[1,3]] has det 11 and inverse (1/11)[[3,-1],[-1,4]];
        // tr(H⁻¹ S) over that block = (3 + 4)/11 = 7/11. p = 3, so
        // edf_total = 3 − 7/11 = 26/11 and edf_by_block[0] = 2 − 7/11 = 15/11.
        let expected_trace = 7.0 / 11.0;
        assert!(
            (penalty_block_trace[0] - expected_trace).abs() < 1e-9,
            "penalty trace {:.9} != analytic 7/11",
            penalty_block_trace[0]
        );
        assert!(
            (edf_by_block[0] - (2.0 - expected_trace)).abs() < 1e-9,
            "per-block EDF {:.9} != 15/11",
            edf_by_block[0]
        );
        assert!(
            (edf_total - (3.0 - expected_trace)).abs() < 1e-9,
            "total EDF {:.9} != 26/11",
            edf_total
        );
    }

    /// #2901: `H ⪰ λS` is what bounds a block's trace by its rank. `H = I` against
    /// `λS = 4I` on a rank-2 block has no certified rank bound, and its raw trace 8 publishes
    /// with `edf_by_block = 2 − 8` unclamped, where the old clamp published the rank.
    #[test]
    fn survival_edf_publishes_an_uncertified_trace_unclamped_2901() {
        let h = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let blocks = vec![penalty_block(array![[4.0, 0.0], [0.0, 4.0]], 1.0, 0)];
        let (edf_total, edf_by_block, penalty_block_trace, rank_bound) =
            survival_edf_from_dense_hessian(&h, &blocks)
                .expect("an indefinite data curvature publishes its raw trace");
        assert!(
            matches!(
                rank_bound[0],
                gam_solve::estimate::EdfRankBound::Uncertified { smallest_pivot, band }
                    if smallest_pivot < -band
            ),
            "{rank_bound:?}"
        );
        assert!((penalty_block_trace[0] - 8.0).abs() < 1e-12, "{penalty_block_trace:?}");
        assert!((edf_by_block[0] - (2.0 - 8.0)).abs() < 1e-12, "{edf_by_block:?}");
        assert!((edf_total - (3.0 - 8.0)).abs() < 1e-12, "{edf_total}");
    }

    /// #2901: `H ≻ 0` is certified only with the block's rank bound, and the
    /// transformation baseline's box-constrained mode need not have it. `H = −I`
    /// against `λS = 4I` is not certified, so its trace −8 below its band publishes raw
    /// with `edf_by_block = 2 + 8` and `edf_total = 3 + 8` unclamped.
    #[test]
    fn survival_edf_publishes_an_uncertified_negative_trace_2901() {
        let h = array![[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let blocks = vec![penalty_block(array![[4.0, 0.0], [0.0, 4.0]], 1.0, 0)];
        let (edf_total, edf_by_block, penalty_block_trace, rank_bound) =
            survival_edf_from_dense_hessian(&h, &blocks)
                .expect("an uncertified negative trace publishes raw");
        assert!(
            matches!(
                rank_bound[0],
                gam_solve::estimate::EdfRankBound::Uncertified { smallest_pivot, band }
                    if smallest_pivot < -band
            ),
            "{rank_bound:?}"
        );
        assert!((penalty_block_trace[0] + 8.0).abs() < 1e-12, "{penalty_block_trace:?}");
        assert!((edf_by_block[0] - (2.0 + 8.0)).abs() < 1e-12, "{edf_by_block:?}");
        assert!((edf_total - (3.0 + 8.0)).abs() < 1e-12, "{edf_total}");
    }
}

#[cfg(test)]
mod survival_design_sharing_2900_tests {
    use super::{hash_workflow_array2, hash_workflow_design_matrix, joint_time_covariate_design};
    use gam_linalg::matrix::DesignMatrix;
    use gam_runtime::warm_start::Fingerprinter;
    use ndarray::{Array2, s};

    fn time_design(n: usize, p_time: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p_time), |(i, j)| ((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.3)
    }

    /// #2900 — the cause-specific joint designs are written from the time design's row
    /// chunks instead of densifying the time design first. They equal the concatenation
    /// they replaced: `[time | covariates]` with covariates and `[time | 0]` without.
    #[test]
    fn joint_design_matches_the_densified_concatenation_2900() {
        let n = 1300;
        let p_time = 4;
        let p_cov = 3;
        let time = time_design(n, p_time);
        let covariates = Array2::from_shape_fn((n, p_cov), |(i, j)| (i as f64).sin() + j as f64);
        let design = DesignMatrix::from(time.clone());

        let joint = joint_time_covariate_design(&design, Some(&covariates), p_time + p_cov)
            .expect("joint design with covariates");
        assert_eq!(joint.dim(), (n, p_time + p_cov));
        assert_eq!(joint.slice(s![.., ..p_time]), time);
        assert_eq!(joint.slice(s![.., p_time..]), covariates);

        let derivative = joint_time_covariate_design(&design, None, p_time + p_cov)
            .expect("joint derivative design");
        assert_eq!(derivative.slice(s![.., ..p_time]), time);
        assert!(derivative.slice(s![.., p_time..]).iter().all(|&v| v == 0.0));
    }

    /// #2900 — the survival warm-start key reads the time design one row chunk at a
    /// time instead of densifying it. It writes the same byte stream the dense hash
    /// wrote, so every key already in a persistent store still matches.
    #[test]
    fn streamed_design_hash_matches_the_dense_hash_2900() {
        let time = time_design(1300, 5);
        let mut dense = Fingerprinter::new();
        hash_workflow_array2(&mut dense, time.view());
        let mut streamed = Fingerprinter::new();
        hash_workflow_design_matrix(&mut streamed, &DesignMatrix::from(time.clone()))
            .expect("streamed design hash");
        assert_eq!(streamed.finish_hex(), dense.finish_hex());
    }
}
