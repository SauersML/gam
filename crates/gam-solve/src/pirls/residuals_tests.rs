//! Closed-form checks of [`glm_residuals`].
//!
//! Each family's residuals are compared with their textbook formulas written
//! out here from `(y, μ)`. None of them come from another call into the solver.
//! The deviance residuals are also checked against the solver's own deviance
//! (`Σ r_i² = D`), so the residual and the reported deviance stay one quantity.

use super::*;
use approx::assert_relative_eq;
use gam_problem::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::array;

/// Every residual here is a handful of flops on O(1) inputs, so a few ulps of
/// the largest term bounds the disagreement.
const TOL: f64 = 1e-12;

fn spec(family: ResponseFamily, link: StandardLink) -> (GlmLikelihoodSpec, InverseLink) {
    let inverse_link = InverseLink::Standard(link);
    (
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(family, inverse_link.clone())),
        inverse_link,
    )
}

fn all_kinds(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    w: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
) -> [Array1<f64>; 4] {
    ResidualKind::ALL.map(|kind| {
        glm_residuals(y.view(), eta.view(), w.view(), likelihood, inverse_link, kind)
            .expect("finite residuals")
    })
}

fn assert_close(label: &str, got: &Array1<f64>, want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= TOL * (1.0 + w.abs()),
            "{label}[{i}]: got {g}, want {w}"
        );
    }
}

/// `Σ r_i²` of the deviance residuals is the solver's reported deviance.
fn assert_deviance_identity(
    deviance_residuals: &Array1<f64>,
    y: &Array1<f64>,
    eta: &Array1<f64>,
    w: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
) {
    let deviance: f64 = (0..y.len())
        .map(|i| {
            2.0 * deviance_eta_row_with_log_measure_scale(
                i,
                y[i],
                eta[i],
                likelihood,
                inverse_link,
                w[i],
                0.0,
            )
            .expect("deviance row")
            .half_deviance
        })
        .sum();
    assert_relative_eq!(
        deviance_residuals.dot(deviance_residuals),
        deviance,
        max_relative = TOL
    );
}

#[test]
fn gaussian_identity_residuals_are_weighted_raw_residuals() {
    let (likelihood, link) = spec(ResponseFamily::Gaussian, StandardLink::Identity);
    let y = array![1.5, -0.25, 3.0, 0.0];
    let eta = array![1.0, 0.5, 2.0, 0.75];
    let w = array![1.0, 4.0, 0.25, 0.0];
    let [response, working, deviance, pearson] = all_kinds(&y, &eta, &w, &likelihood, &link);

    let raw: Vec<f64> = (0..4).map(|i| y[i] - eta[i]).collect();
    let scaled: Vec<f64> = (0..4).map(|i| w[i].sqrt() * raw[i]).collect();
    assert_close("response", &response, &raw);
    // dμ/dη = 1: the working residual is the raw residual.
    assert_close("working", &working, &raw);
    // V(μ) = 1 and the unit deviance is w(y−μ)², so deviance = Pearson = √w·(y−μ).
    assert_close("deviance", &deviance, &scaled);
    assert_close("pearson", &pearson, &scaled);
    assert_deviance_identity(&deviance, &y, &eta, &w, &likelihood, &link);
}

#[test]
fn poisson_log_residuals_match_closed_forms() {
    let (likelihood, link) = spec(ResponseFamily::Poisson, StandardLink::Log);
    let y = array![0.0, 1.0, 4.0, 2.0];
    let eta = array![0.3, -0.2, 1.1, 0.693];
    let w = array![1.0, 2.0, 1.0, 0.5];
    let [response, working, deviance, pearson] = all_kinds(&y, &eta, &w, &likelihood, &link);

    let mu: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
    let raw: Vec<f64> = (0..4).map(|i| y[i] - mu[i]).collect();
    // A y = 0 row has y·ln(y/μ) = 0.
    let unit_dev: Vec<f64> = (0..4)
        .map(|i| {
            let ylog = if y[i] == 0.0 { 0.0 } else { y[i] * (y[i] / mu[i]).ln() };
            2.0 * w[i] * (ylog - raw[i])
        })
        .collect();
    assert_close("response", &response, &raw);
    // dμ/dη = μ.
    assert_close("working", &working, &(0..4).map(|i| raw[i] / mu[i]).collect::<Vec<_>>());
    // V(μ) = μ.
    assert_close(
        "pearson",
        &pearson,
        &(0..4).map(|i| raw[i] * (w[i] / mu[i]).sqrt()).collect::<Vec<_>>(),
    );
    assert_close(
        "deviance",
        &deviance,
        &(0..4).map(|i| raw[i].signum() * unit_dev[i].sqrt()).collect::<Vec<_>>(),
    );
    assert_deviance_identity(&deviance, &y, &eta, &w, &likelihood, &link);
}

#[test]
fn gamma_log_residuals_match_closed_forms() {
    let (likelihood, link) = spec(ResponseFamily::Gamma, StandardLink::Log);
    let y = array![0.5, 2.0, 7.5, 1.0];
    let eta = array![0.1, 0.9, 1.6, -0.4];
    let w = array![1.0, 1.0, 3.0, 0.5];
    let [response, working, deviance, pearson] = all_kinds(&y, &eta, &w, &likelihood, &link);

    let mu: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
    let raw: Vec<f64> = (0..4).map(|i| y[i] - mu[i]).collect();
    let unit_dev: Vec<f64> = (0..4)
        .map(|i| 2.0 * w[i] * (-(y[i] / mu[i]).ln() + raw[i] / mu[i]))
        .collect();
    assert_close("response", &response, &raw);
    assert_close("working", &working, &(0..4).map(|i| raw[i] / mu[i]).collect::<Vec<_>>());
    // V(μ) = μ²: the Pearson residual is √w·(y−μ)/μ, without the dispersion.
    assert_close(
        "pearson",
        &pearson,
        &(0..4).map(|i| w[i].sqrt() * raw[i] / mu[i]).collect::<Vec<_>>(),
    );
    assert_close(
        "deviance",
        &deviance,
        &(0..4).map(|i| raw[i].signum() * unit_dev[i].sqrt()).collect::<Vec<_>>(),
    );
    assert_deviance_identity(&deviance, &y, &eta, &w, &likelihood, &link);
}

#[test]
fn binomial_logit_residuals_match_closed_forms() {
    let (likelihood, link) = spec(ResponseFamily::Binomial, StandardLink::Logit);
    let y = array![0.0, 1.0, 1.0, 0.0];
    let eta = array![-1.2, 0.4, 2.5, 0.8];
    let w = array![1.0, 1.0, 2.0, 1.0];
    let [response, working, deviance, pearson] = all_kinds(&y, &eta, &w, &likelihood, &link);

    let mu: Vec<f64> = eta.iter().map(|e| 1.0 / (1.0 + (-e).exp())).collect();
    let raw: Vec<f64> = (0..4).map(|i| y[i] - mu[i]).collect();
    let var: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
    // Binary y: the unit deviance is −2·ln P(y), so 2w·(−ln μ) or 2w·(−ln(1−μ)).
    let unit_dev: Vec<f64> = (0..4)
        .map(|i| {
            let p = if y[i] == 1.0 { mu[i] } else { 1.0 - mu[i] };
            -2.0 * w[i] * p.ln()
        })
        .collect();
    assert_close("response", &response, &raw);
    // Canonical logit: dμ/dη = μ(1−μ) = V(μ).
    assert_close("working", &working, &(0..4).map(|i| raw[i] / var[i]).collect::<Vec<_>>());
    assert_close(
        "pearson",
        &pearson,
        &(0..4).map(|i| raw[i] * (w[i] / var[i]).sqrt()).collect::<Vec<_>>(),
    );
    assert_close(
        "deviance",
        &deviance,
        &(0..4).map(|i| raw[i].signum() * unit_dev[i].sqrt()).collect::<Vec<_>>(),
    );
    assert_deviance_identity(&deviance, &y, &eta, &w, &likelihood, &link);
}

#[test]
fn residual_kind_names_round_trip_and_unknown_names_are_refused() {
    for kind in ResidualKind::ALL {
        assert_eq!(kind.name().parse::<ResidualKind>(), Ok(kind));
        assert_eq!(kind.to_string(), kind.name());
    }
    let err = "raw".parse::<ResidualKind>().unwrap_err();
    assert_eq!(
        err,
        "unknown residual type 'raw'; expected one of: response, working, deviance, pearson"
    );
}

#[test]
fn mismatched_lengths_are_refused() {
    let (likelihood, link) = spec(ResponseFamily::Gaussian, StandardLink::Identity);
    let y = array![1.0, 2.0];
    let eta = array![1.0];
    let w = array![1.0, 1.0];
    assert!(
        glm_residuals(
            y.view(),
            eta.view(),
            w.view(),
            &likelihood,
            &link,
            ResidualKind::Response
        )
        .is_err()
    );
}
