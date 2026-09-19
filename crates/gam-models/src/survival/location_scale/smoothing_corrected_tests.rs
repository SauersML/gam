#![cfg(test)]
//! #2677: a PENALIZED survival location-scale fit must carry its
//! smoothing-corrected covariance through finalization.
//!
//! The custom-family lane computes the first-order ρ-uncertainty correction
//! `C` and publishes `V_c = V_cond + C` (`beta_covariance_corrected`, #2346).
//! `finalize_survival_location_scale_fit` then rebuilds the fit in the RAW
//! coefficient frame — and used to hard-code `covariance_corrected: None`,
//! discarding a matrix the solver had already produced. The CLI save path
//! forwards `fit.covariance_corrected` verbatim, so every penalized survival
//! model saved by `gam fit --survival-likelihood location-scale` persisted
//! without it, and the DEFAULT `gam predict` invocation
//! (`--covariance-mode corrected`) refused it with
//!
//!   saved model does not contain smoothing-corrected covariance; refit before
//!   requesting --covariance-mode corrected
//!
//! — an instruction no refit could satisfy, because nothing on the refit path
//! was failing to compute the correction.

use super::*;
use ndarray::{Array1, Array2};

/// Deterministic lognormal-AFT sample with a smooth covariate effect on the
/// location: `log t = 1.0 + 0.6·sin(2x) + σ·z`, `x ∈ [0, 3]`, all uncensored.
fn penalized_location_sample(n: usize, sigma: f64, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut state = seed;
    let mut next_u01 = move || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = 3.0 * (i as f64) / ((n - 1) as f64);
        let u1 = next_u01().max(1e-12);
        let u2 = next_u01();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        x[i] = xi;
        age_exit[i] = (1.0 + 0.6 * (2.0 * xi).sin() + sigma * z).exp();
    }
    (x, age_exit)
}

/// Squared-second-difference penalty on the trailing `p_basis` columns of a
/// `1 + p_basis` design (the leading intercept column is unpenalized).
fn second_difference_penalty(p_basis: usize) -> Array2<f64> {
    let p = p_basis + 1;
    let mut penalty = Array2::<f64>::zeros((p, p));
    for j in 0..(p_basis - 2) {
        let idx = [1 + j, 2 + j, 3 + j];
        let row = [1.0_f64, -2.0, 1.0];
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                penalty[[ia, ib]] += row[a] * row[b];
            }
        }
    }
    penalty
}

/// A survival location-scale spec whose LOCATION block carries a real penalized
/// smooth (the bench's `s(x, type=ps)` shape), so the fit is NOT the reduced
/// unpenalized parametric-AFT regime and must route through the coupled
/// custom-family path that computes the smoothing correction.
fn penalized_location_spec(x: &Array1<f64>, age_exit: &Array1<f64>) -> SurvivalLocationScaleSpec {
    let n = age_exit.len();
    let p_time = 4usize;
    let p_basis = 6usize;

    let log_t: Vec<f64> = age_exit.iter().map(|&t| t.max(1e-12).ln()).collect();
    let lo = log_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = log_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-6);

    // Monotone I-spline-like time basis over log t (value + derivative rows).
    let mut design_exit = Array2::<f64>::zeros((n, p_time));
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    for i in 0..n {
        let lt = log_t[i];
        for j in 0..p_time {
            let center = lo + span * (j as f64 + 0.5) / (p_time as f64);
            let arg = 6.0 / span * (lt - center);
            let sigmoid = 1.0 / (1.0 + (-arg).exp());
            design_exit[[i, j]] = sigmoid;
            let dsig = sigmoid * (1.0 - sigmoid) * (6.0 / span);
            design_derivative_exit[[i, j]] = dsig / age_exit[i].max(1e-12);
        }
    }
    let mut time_penalty = Array2::<f64>::zeros((p_time, p_time));
    for j in 0..(p_time - 1) {
        time_penalty[[j, j]] += 1.0;
        time_penalty[[j, j + 1]] -= 1.0;
        time_penalty[[j + 1, j]] -= 1.0;
        time_penalty[[j + 1, j + 1]] += 1.0;
    }

    // Location design: intercept + a smooth Gaussian-RBF basis over x, with a
    // second-difference penalty on the basis columns. This is what makes
    // `k_threshold > 0`, i.e. what makes this a genuinely penalized fit with a
    // smoothing coordinate to be uncertain about.
    let x_lo = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x_span = (x_hi - x_lo).max(1e-6);
    let bandwidth = x_span / (p_basis as f64);
    let mut threshold_design = Array2::<f64>::zeros((n, p_basis + 1));
    for i in 0..n {
        threshold_design[[i, 0]] = 1.0;
        for j in 0..p_basis {
            let center = x_lo + x_span * (j as f64 + 0.5) / (p_basis as f64);
            let u = (x[i] - center) / bandwidth;
            threshold_design[[i, 1 + j]] = (-0.5 * u * u).exp();
        }
    }

    SurvivalLocationScaleSpec {
        age_entry: Array1::from_elem(n, 1e-9_f64),
        age_exit: age_exit.clone(),
        event_target: Array1::<f64>::ones(n),
        weights: Array1::<f64>::ones(n),
        inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::<f64>::zeros((n, p_time))),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::from_elem(
                n,
                DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            ),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: vec![time_penalty],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(Array1::from_elem(1, 0.0)),
            initial_beta: None,
        },
        threshold_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(threshold_design),
            offset: Array1::zeros(n),
            penalties: vec![gam_terms::penalty_spec::PenaltySpec::Dense(
                second_difference_penalty(p_basis),
            )],
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Some(Array1::from_elem(1, 0.0)),
            initial_beta: None,
        }),
        log_sigma_block: CovariateBlockKind::Static(ParameterBlockInput {
            design: DesignMatrix::from(Array2::ones((n, 1))),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }),
        timewiggle_block: None,
        linkwiggle_block: None,
        cache_session: None,
        persistent_warm_start_store: None,
        cache_mirror_sessions: Vec::new(),
    }
}

/// #2677: the finalized penalized survival location-scale fit must publish the
/// smoothing-corrected covariance, in the same raw coefficient frame as the
/// conditional one, with `V_c = V_cond + C` and `C` positive semi-definite.
///
/// Pre-fix this asserted `None`: the correction was computed by the
/// custom-family assembler and then discarded by the finalizer, which is why
/// `heart_failure_survival` in bench run 30658500823 fit successfully
/// (`status=Converged | iterations=16`) and then failed the very next
/// `gam predict` on "saved model does not contain smoothing-corrected
/// covariance".
#[test]
fn penalized_survival_location_scale_finalization_keeps_smoothing_corrected_covariance() {
    assert_selected_fit_keeps_smoothing_corrected_covariance(residual_distribution_inverse_link(
        ResidualDistribution::Gaussian,
    ));
}

/// #2903: since 0fa276e59 the loglog residual distribution declares the third
/// information derivative, so a penalized fit on it selects rho with an analytic
/// outer Hessian and must carry the correction through finalization as the
/// Gaussian fit does.
#[test]
fn penalized_loglog_survival_location_scale_fit_keeps_smoothing_corrected_covariance_2903() {
    assert_selected_fit_keeps_smoothing_corrected_covariance(InverseLink::Standard(
        StandardLink::LogLog,
    ));
}

/// #2903: the cauchit counterpart of the loglog witness.
#[test]
fn penalized_cauchit_survival_location_scale_fit_keeps_smoothing_corrected_covariance_2903() {
    assert_selected_fit_keeps_smoothing_corrected_covariance(InverseLink::Standard(
        StandardLink::Cauchit,
    ));
}

/// The penalized fixture fitted with `inverse_link` as its residual
/// distribution selects rho and publishes `V_c = V_cond + C` in the raw
/// coefficient frame, with `C` positive semi-definite and its typed provenance.
fn assert_selected_fit_keeps_smoothing_corrected_covariance(inverse_link: InverseLink) {
    let (x, age_exit) = penalized_location_sample(300, 0.35, 20260731);
    let mut spec = penalized_location_spec(&x, &age_exit);
    spec.inverse_link = inverse_link;

    // Guard the fixture: a reduced unpenalized AFT would never reach the
    // custom-family smoothing-correction code at all, so this test would be
    // mute rather than green.
    let prepared = prepare_survival_location_scale_model(&spec)
        .expect("penalized survival location-scale spec must prepare");
    assert!(
        !prepared.is_reduced_parametric_aft(),
        "fixture must exercise the PENALIZED coupled path, not the reduced parametric-AFT MLE"
    );

    let (fit, _geometry) = fit_survival_location_scale_with_geometry(spec)
        .expect("penalized survival location-scale fit must converge on benign lognormal data");

    // Separate this from the `lambda_is_fixed` fallback landed in 7bbf50639
    // (which publishes `Vp = Vb` when no rho was selected at all): this fit
    // SELECTED its smoothing coordinates, so the corrected covariance it
    // publishes must be a carried correction, not that identity.
    assert!(
        fit.outer_iterations > 0,
        "fixture must SELECT rho (the bench scenarios run outer iterations); with a fixed \
         lambda the corrected covariance is the conditional one by the zero-rho identity and \
         this test would measure the fallback instead of the carried correction"
    );
    assert!(
        !fit.lambdas.is_empty(),
        "a penalized fit must carry smoothing coordinates; without them the corrected \
         covariance is the conditional one by definition and this test proves nothing"
    );

    let conditional = fit
        .covariance_conditional
        .as_ref()
        .expect("penalized survival fit must publish a conditional covariance");
    let corrected = fit.covariance_corrected.as_ref().expect(
        "#2677: the finalized fit must carry the smoothing-corrected covariance the \
         custom-family assembler produced (it was dropped at finalization)",
    );

    assert_eq!(
        corrected.dim(),
        conditional.dim(),
        "corrected covariance must live in the same raw coefficient frame as the conditional one"
    );

    // `V_c = V_cond + C` with `C` PSD, so every marginal variance is at least
    // the conditional one. A strict-equality result would mean the correction
    // lifted to exactly zero, which is not what a live rho posterior produces.
    let p = conditional.nrows();
    let mut max_gain: f64 = 0.0;
    for j in 0..p {
        let gain = corrected[[j, j]] - conditional[[j, j]];
        let scale = conditional[[j, j]].abs().max(1e-12);
        assert!(
            gain >= -1e-9 * scale,
            "corrected variance {} is below the conditional {} at coefficient {j}: the \
             smoothing correction must be positive semi-definite",
            corrected[[j, j]],
            conditional[[j, j]]
        );
        max_gain = max_gain.max(gain / scale);
    }
    assert!(
        max_gain > 0.0,
        "the lifted smoothing correction is identically zero on the diagonal; a carried \
         correction must widen at least one marginal variance"
    );

    let inference = fit
        .inference
        .as_ref()
        .expect("penalized survival fit must publish an inference block");
    assert!(
        fit.covariance_corrected.is_some(),
        "#2677: the CLI saved-model path reads the corrected covariance; a fit that selected \
         lambda without publishing it refuses `--covariance-mode corrected`"
    );
    assert!(
        inference.smoothing_correction.is_some()
            && inference.smoothing_correction_method.is_some(),
        "the correction term and its typed provenance must survive finalization together"
    );
    assert!(
        fit.beta_standard_errors_corrected().is_some(),
        "corrected marginal SEs must be published alongside the corrected covariance"
    );
}
