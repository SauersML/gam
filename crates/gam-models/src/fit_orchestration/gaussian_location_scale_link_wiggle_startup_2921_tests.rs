#![cfg(test)]
//! Regression for gam#2921: a Gaussian location-scale request with the default
//! link wiggle must fit the 48-row table that gnomon's calibration test trains
//! on (gnomon#2351), with and without the wiggle.
//!
//! Two requests are fitted. `Request::Gnomon` is the one gnomon builds in
//! `train_gaussian_location_scale`: identical mean and log-sigma term
//! collections, each a 4-center Duchon smooth on the score with a fixed kernel
//! scale plus a double-penalized linear `sex` term, zero offsets, and gnomon's
//! iteration caps. `Request::Issue` is the request as filed: a pure Duchon
//! smooth whose length scale is optimized, and the issue's caps. Both carry
//! `WigglePenaltyConfig::cubic_triple_operator_default()` as the link wiggle.

use super::entry::fit_model;
use super::request::{FitRequest, FitResult, GaussianLocationScaleFitRequest, LinkWiggleConfig};
use crate::custom_family::BlockwiseFitOptions;
use crate::gamlss::{GaussianLocationScaleFitResult, GaussianLocationScaleTermSpec};
use gam_solve::model_types::CurvatureAdmissibility;
use gam_spec::WigglePenaltyConfig;
use gam_terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
};
use gam_terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use ndarray::{Array1, Array2};

const ROWS: usize = 48;

#[derive(Clone, Copy)]
enum Request {
    Gnomon,
    Issue,
}

impl Request {
    fn label(self) -> &'static str {
        match self {
            Self::Gnomon => "gnomon request",
            Self::Issue => "issue request",
        }
    }
}

/// gnomon's `gaussian_public_train_save_load_predict_preserves_schema_and_single_rows`
/// table: columns `[score, sex]`, the response, and the mean it was generated
/// from. The `0.2 * sin(1.7 i)` term is deterministic, so the table has no
/// generating sigma to score against.
struct Table {
    data: Array2<f64>,
    y: Array1<f64>,
    true_mean: Array1<f64>,
}

fn gnomon_48_row_table() -> Table {
    let mut data = Array2::<f64>::zeros((ROWS, 2));
    let mut y = Array1::<f64>::zeros(ROWS);
    let mut true_mean = Array1::<f64>::zeros(ROWS);
    for i in 0..ROWS {
        let score = (i as f64 - 24.0) / 12.0;
        let sex = (i % 2) as f64;
        data[[i, 0]] = score;
        data[[i, 1]] = sex;
        true_mean[i] = 2.0 + 0.7 * score + 0.3 * sex;
        y[i] = true_mean[i] + 0.2 * (i as f64 * 1.7).sin();
    }
    Table { data, y, true_mean }
}

/// gnomon's `build_marginal_termspec` with no PCs, or the issue's pure Duchon
/// variant of it.
fn termspec(request: Request) -> TermCollectionSpec {
    let (length_scale, power, linear_feature_cols) = match request {
        Request::Gnomon => (Some(1.0), 1.0, Vec::new()),
        Request::Issue => (None, 0.0, vec![1]),
    };
    TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "sex".to_string(),
            feature_col: 1,
            feature_cols: linear_feature_cols,
            categorical_levels: Vec::new(),
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            name: "pgs".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    periodic: None,
                    length_scale,
                    power,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                    radial_reparam: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
            frozen_parametric_residualization: None,
        }],
        level: Default::default(),
    }
}

fn default_link_wiggle() -> LinkWiggleConfig {
    let cfg = WigglePenaltyConfig::cubic_triple_operator_default();
    LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    }
}

fn fit(
    request: Request,
    table: &Table,
    wiggle: Option<LinkWiggleConfig>,
) -> GaussianLocationScaleFitResult {
    let label = format!(
        "{}, {}",
        request.label(),
        if wiggle.is_some() { "default link wiggle" } else { "no wiggle" }
    );
    let (inner_max_cycles, outer_max_iter) = match request {
        Request::Gnomon => (200, 50),
        Request::Issue => (50, 100),
    };
    let spec = termspec(request);
    let fit_request = GaussianLocationScaleFitRequest {
        data: table.data.view(),
        spec: GaussianLocationScaleTermSpec {
            y: table.y.clone(),
            weights: Array1::ones(ROWS),
            meanspec: spec.clone(),
            log_sigmaspec: spec,
            mean_offset: Array1::zeros(ROWS),
            log_sigma_offset: Array1::zeros(ROWS),
        },
        wiggle,
        options: BlockwiseFitOptions {
            inner_max_cycles,
            inner_tol: 1e-7,
            outer_max_iter,
            outer_tol: 1e-3,
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        },
        kappa_options: SpatialLengthScaleOptimizationOptions {
            max_outer_iter: outer_max_iter,
            rel_tol: 1e-3,
            ..SpatialLengthScaleOptimizationOptions::default()
        },
    };
    match fit_model(FitRequest::GaussianLocationScale(fit_request)) {
        Ok(FitResult::GaussianLocationScale(fit)) => fit,
        Ok(_) => panic!("{label}: fit_model returned a non-Gaussian-location-scale result"),
        Err(error) => panic!("{label}: the 48-row table must fit: {error}"),
    }
}

/// The fit's inner mode is converged, its outer certificate certifies, and a
/// curvature question was actually answered at the certified point.
///
/// A wiggle fit certifies with `CriterionContradicted`, not `Admissible`
/// (gam#2612's adjudication), and the two requests reach it differently. On
/// gnomon's request the reported direction is flat: the criterion's second
/// difference along it is -4.1e-7 at every step from 1 down to 0.0625, and it
/// can lower the criterion by at most 5.7e-7. On the issue's request the
/// direction lies inside the box's feasible cone (the three railed coordinates
/// carry exact zeros, and no trial was projected), and the analytic curvature is
/// real: second differences -6.40e-3, -6.31e-3, -5.94e-3, -4.44e-3 at steps
/// 0.0625 .. 0.5 against the claimed -6.43e-3, turning to +2.04e-3 at step 1
/// because the criterion is not quadratic over a full e-fold. Both signs at step
/// 0.5 lower the criterion by 5.5e-4 to 5.6e-4, below its 1.417e-3 resolution,
/// so that point is a shallow saddle the certificate accepts by resolution, not
/// by feasibility (gam#2939).
fn assert_certified(fit: &GaussianLocationScaleFitResult, label: &str) -> CurvatureAdmissibility {
    let evidence = fit.fit.fit.convergence_evidence();
    assert!(
        evidence.inner_status().is_converged(),
        "{label}: the inner mode is not converged"
    );
    let certificate = evidence
        .outer_certificate()
        .unwrap_or_else(|| panic!("{label}: the outer smoothing search carries no certificate"));
    assert!(
        certificate.certifies(),
        "{label}: the outer certificate does not certify: {}",
        certificate.summary()
    );
    let verdict = certificate.curvature_verdict();
    assert!(
        matches!(
            verdict,
            CurvatureAdmissibility::Admissible | CurvatureAdmissibility::CriterionContradicted
        ),
        "{label}: no curvature verdict was measured and adjudicated at the certified point: {verdict}"
    );
    verdict
}

/// The Gaussian log-likelihood and the absolute mass of its row terms, which
/// bounds the round-off of the sum.
fn gaussian_log_likelihood(y: &Array1<f64>, mu: &Array1<f64>, sigma: &Array1<f64>) -> (f64, f64) {
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    (0..y.len())
        .map(|i| {
            let z = (y[i] - mu[i]) / sigma[i];
            -0.5 * ln2pi - sigma[i].ln() - 0.5 * z * z
        })
        .fold((0.0, 0.0), |(sum, mass), term| (sum + term, mass + term.abs()))
}

fn root_mean_square(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), v| (sum + v * v, count + 1));
    (sum / count as f64).sqrt()
}

/// The fitted raw-scale mean and sigma rebuilt from the returned model, scored.
struct Reading {
    log_likelihood_from_basis: f64,
    log_likelihood_from_states: f64,
    log_likelihood_mass: f64,
    wiggle_state_defect: f64,
    rmse_mean: f64,
}

fn read_fit(fit: &GaussianLocationScaleFitResult, table: &Table, label: &str) -> Reading {
    let states = &fit.fit.fit.block_states;
    let eta_mu = &states[0].eta;
    let sigma = states[1]
        .eta
        .mapv(|eta| fit.response_scale * fit.sigma_floor + eta.exp());
    let (mu_from_basis, mu_from_states, wiggle_state_defect) =
        match (&fit.wiggle_knots, fit.wiggle_degree, &fit.beta_link_wiggle) {
            (Some(knots), Some(degree), Some(beta)) => {
                let basis = crate::wiggle::monotone_wiggle_basis_from_knots(
                    eta_mu.view(),
                    knots,
                    degree,
                )
                .unwrap_or_else(|error| panic!("{label}: wiggle basis at the fitted mean: {error}"));
                assert_eq!(
                    basis.ncols(),
                    beta.len(),
                    "{label}: the wiggle basis and its coefficients disagree in width"
                );
                let from_basis = basis.dot(&Array1::from(beta.clone()));
                let from_states = &states[2].eta;
                let defect = from_basis
                    .iter()
                    .zip(from_states.iter())
                    .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
                (eta_mu + &from_basis, eta_mu + from_states, defect)
            }
            (None, None, None) => (eta_mu.clone(), eta_mu.clone(), 0.0),
            _ => panic!("{label}: the fit records only part of a link wiggle"),
        };
    let (log_likelihood_from_basis, _) = gaussian_log_likelihood(&table.y, &mu_from_basis, &sigma);
    let (log_likelihood_from_states, log_likelihood_mass) =
        gaussian_log_likelihood(&table.y, &mu_from_states, &sigma);
    Reading {
        log_likelihood_from_basis,
        log_likelihood_from_states,
        log_likelihood_mass,
        wiggle_state_defect,
        rmse_mean: root_mean_square(
            mu_from_basis
                .iter()
                .zip(table.true_mean.iter())
                .map(|(fitted, truth)| fitted - truth),
        ),
    }
}

/// The returned model reproduces the log-likelihood it reports, both from its
/// stored predictors and from the wiggle basis rebuilt at its fitted mean. The
/// bound is the round-off of the sum and of the raw-scale shift `n·ln s`.
fn assert_reproduces_log_likelihood(
    fit: &GaussianLocationScaleFitResult,
    reading: &Reading,
    label: &str,
) {
    let reported = fit.fit.fit.log_likelihood;
    let round_off = 256.0
        * f64::EPSILON
        * (reading.log_likelihood_mass + ROWS as f64 * fit.response_scale.ln().abs());
    for (source, recomputed) in [
        ("stored predictors", reading.log_likelihood_from_states),
        ("wiggle basis at the fitted mean", reading.log_likelihood_from_basis),
    ] {
        assert!(
            (reported - recomputed).abs() <= round_off,
            "{label}: the returned model does not reproduce its reported log-likelihood from its {source}: \
             reported {reported:.17e}, recomputed {recomputed:.17e}, round-off bound {round_off:.3e}"
        );
    }
}

fn check_request(request: Request) {
    gam_runtime::test_support::install_diagnostic_logger();
    let table = gnomon_48_row_table();
    let plain = fit(request, &table, None);
    let wiggle = fit(request, &table, Some(default_link_wiggle()));
    let plain_label = format!("{}, no wiggle", request.label());
    let wiggle_label = format!("{}, default link wiggle", request.label());
    let plain_verdict = assert_certified(&plain, &plain_label);
    let wiggle_verdict = assert_certified(&wiggle, &wiggle_label);
    assert!(wiggle.beta_link_wiggle.is_some(), "{wiggle_label}: no wiggle coefficients");
    let plain_reading = read_fit(&plain, &table, &plain_label);
    let wiggle_reading = read_fit(&wiggle, &table, &wiggle_label);
    for (label, fit, reading, verdict) in [
        (&plain_label, &plain, &plain_reading, plain_verdict),
        (&wiggle_label, &wiggle, &wiggle_reading, wiggle_verdict),
    ] {
        eprintln!(
            "[gam#2921] {label}: log-likelihood reported {:.12e}, from basis {:.12e}, from states {:.12e}; \
             wiggle state defect {:.3e}; mean RMSE against the generating mean {:.9e}; \
             curvature {verdict}; outer iterations {}",
            fit.fit.fit.log_likelihood,
            reading.log_likelihood_from_basis,
            reading.log_likelihood_from_states,
            reading.wiggle_state_defect,
            reading.rmse_mean,
            fit.fit.fit.outer_iterations,
        );
    }
    assert_reproduces_log_likelihood(&plain, &plain_reading, &plain_label);
    assert_reproduces_log_likelihood(&wiggle, &wiggle_reading, &wiggle_label);
    // The wiggle block is non-negative, so the fit publishes a truncated
    // posterior mean and keeps the mode's log-likelihood beside the mode.
    let mode_log_likelihood = wiggle
        .fit
        .fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
        .and_then(|constrained| constrained.mode_log_likelihood)
        .unwrap_or_else(|| panic!("{wiggle_label}: the published mean carries no mode log-likelihood"));
    assert_eq!(
        wiggle.fit.fit.log_likelihood_at_mode().to_bits(),
        mode_log_likelihood.to_bits()
    );
    eprintln!("[gam#2921] {wiggle_label}: log-likelihood at the mode {mode_log_likelihood:.12e}");
}

#[test]
fn gaussian_location_scale_link_wiggle_fits_the_gnomon_request_2921() {
    check_request(Request::Gnomon);
}

#[test]
fn gaussian_location_scale_link_wiggle_fits_the_issue_request_2921() {
    check_request(Request::Issue);
}
