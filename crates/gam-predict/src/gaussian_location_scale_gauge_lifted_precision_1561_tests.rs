#![cfg(test)]
//! #1561: the prediction precision route refused every non-identity coefficient gauge
//! (`usable_penalized_hessian`). So a custom-family fit whose penalized precision stays on
//! the reduced active coordinates of `β = T·θ + a` had no prediction standard errors
//! whenever the dense covariance was absent.
//!
//! The by-group Gaussian location-scale fit is such a fit. Each by-level carries its gated
//! indicator as a parametric column beside the global intercept, the identifiability
//! audit drops one column per block, and the precision is published on the reduced frame.
//!
//! This pin fits that shape and predicts in-sample η standard errors twice:
//! - from the published dense covariance;
//! - from the gauge-lifted precision `T·H_θ⁻¹·Tᵀ`, after clearing the dense conditional
//!   covariance.
//!
//! Both routes invert the same precision (`spd_covariance_from_precision` publishes the
//! dense one), so they agree to that precision's roundoff. The bound is `64·ε·κ(H_θ)`,
//! with the rank-reveal slack the identifiability compiler uses.

use crate::gaussian_location_scale::GaussianLocationScalePredictor;
use crate::{InferenceCovarianceMode, PredictInput, PredictUncertaintyOptions, PredictableModel};
use faer::Side;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_problem::BlockRole;
use ndarray::Array1;

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits).
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn gauge_lifted_precision_predicts_the_dense_covariance_bands_on_a_dropped_column_fit_1561() {
    let per_group = 100usize;
    let mut rng = Lcg(1561);
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(2 * per_group);
    for (label, shift, noise) in [("A", 0.0_f64, 0.15_f64), ("B", 0.5_f64, 0.25_f64)] {
        for i in 0..per_group {
            let x = (i as f64 + rng.unit()) / per_group as f64;
            let y = shift + (std::f64::consts::TAU * x).sin() + noise * rng.normal();
            records.push(csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{x:.17e}"),
                label.to_string(),
            ]));
        }
    }
    let data = encode_recordswith_inferred_schema(
        vec!["y".to_string(), "x".to_string(), "group".to_string()],
        records,
    )
    .expect("encode the two-group data");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp', by=group)", &data, &config)
        .expect("two-group Gaussian location-scale fit");
    let FitResult::GaussianLocationScale(fitted) = result else {
        panic!("expected a Gaussian location-scale fit result");
    };
    let unified = fitted.fit.fit.clone();
    let gauge = &unified
        .geometry
        .as_ref()
        .expect("the fit publishes its saved geometry")
        .coefficient_gauge;
    assert!(
        gauge.raw_total() > gauge.reduced_total(),
        "the by-group fixture must drop columns: the gauge lifts {} active coordinates to {}",
        gauge.reduced_total(),
        gauge.raw_total()
    );
    assert!(
        unified.beta_covariance().is_some(),
        "the production workflow publishes the dense conditional covariance"
    );

    let predictor = GaussianLocationScalePredictor {
        beta_mu: unified
            .block_by_role(BlockRole::Location)
            .expect("location block")
            .beta
            .clone(),
        beta_noise: unified
            .block_by_role(BlockRole::Scale)
            .expect("scale block")
            .beta
            .clone(),
        sigma_floor: gam_model_kernels::sigma_link::LOGB_SIGMA_FLOOR,
        response_scale: fitted.response_scale,
        covariance: None,
        link_wiggle: None,
    };
    let rows = fitted.fit.mean_design.design.nrows();
    let input = PredictInput {
        design: fitted.fit.mean_design.design.clone(),
        offset: Array1::<f64>::zeros(rows),
        design_noise: Some(fitted.fit.noise_design.design.clone()),
        offset_noise: Some(Array1::<f64>::zeros(rows)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };
    let options = PredictUncertaintyOptions {
        covariance_mode: InferenceCovarianceMode::Conditional,
        includeobservation_interval: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..Default::default()
    };

    let dense = predictor
        .predict_full_uncertainty(&input, &unified, &options)
        .expect("dense-covariance prediction");

    let mut precision_only = unified.clone();
    precision_only.covariance_conditional = None;
    assert!(
        precision_only.beta_covariance().is_none(),
        "the precision arm carries no dense conditional covariance"
    );
    let lifted = predictor
        .predict_full_uncertainty(&input, &precision_only, &options)
        .expect("the gauge-lifted precision predicts standard errors without a dense covariance");
    assert!(
        matches!(lifted.covariance_source, InferenceCovarianceMode::Conditional),
        "the precision arm answers the conditional covariance mode"
    );

    let hessian = precision_only
        .penalized_hessian()
        .expect("the fit publishes its penalized precision");
    assert_eq!(
        hessian.nrows(),
        gauge.reduced_total(),
        "the published precision lives on the gauge's active coordinates"
    );
    let (eigenvalues, _) = hessian.eigh(Side::Lower).expect("precision spectrum");
    let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        smallest > 0.0,
        "the active-frame precision is positive definite (smallest eigenvalue {smallest:e})"
    );
    let tolerance = 64.0 * f64::EPSILON * (largest / smallest);

    assert_eq!(dense.eta_standard_error.len(), rows);
    assert_eq!(lifted.eta_standard_error.len(), rows);
    let mut worst = 0.0_f64;
    for (row, (&from_dense, &from_precision)) in dense
        .eta_standard_error
        .iter()
        .zip(lifted.eta_standard_error.iter())
        .enumerate()
    {
        assert!(
            from_dense.is_finite() && from_dense > 0.0,
            "row {row}: the dense-covariance η standard error must be positive and finite, got {from_dense:e}"
        );
        worst = worst.max((from_dense - from_precision).abs() / from_dense);
    }
    assert!(
        worst <= tolerance,
        "gauge-lifted η standard errors depart from the dense-covariance bands by {worst:e} relative, \
         above 64·ε·κ(H_θ) = {tolerance:e}"
    );
}
