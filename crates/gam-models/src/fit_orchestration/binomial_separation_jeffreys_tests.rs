#![cfg(test)]
//! Separated and rare-event binomial smooths must fit, not fail.
//!
//! A flat prior on the unpenalized block of `y ~ smooth(x)` (the intercept and
//! the smooth's linear null-space direction) gives an improper posterior as
//! soon as that block admits a (quasi-)separating direction, and the
//! null-space shrinkage ridge does not cure it: REML drives the ridge's λ to 0
//! along the separating direction. The Jeffreys (Firth) prior makes the
//! posterior proper, so each fixture below must mint a certified fit with
//! finite coefficients, finite standard errors and fitted means in (0, 1).
//!
//! The #784 block quadrature correction integrates the flat-prior posterior,
//! so on a Firth fit of separated data it integrated an improper posterior:
//! the order search ground without end on the step, and its spliced gradient
//! stalled the outer search on the quasi-separated fixture. The Jeffreys prior
//! is armed before the first solve only on certified separation (#3129), and the quasi-separated fixture has
//! no strict separator to certify: its flat-prior fit returns an optimum with
//! the null-space ridge's λ railed at 0 and |η| near 50. So the pre-fit check
//! also certifies quasi-complete separation along a null-space direction.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use ndarray::Array1;

fn dataset(xs: &[f64], ys: &[f64]) -> gam_data::EncodedDataset {
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let rows = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| StringRecord::from(vec![x.to_string(), y.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// A step function on an even grid over [0, 1]: y = 1{x > 0.5}, complete
/// separation with no gap to speak of between the classes.
fn perfect_step() -> (Vec<f64>, Vec<f64>) {
    let n = 200;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let ys = xs.iter().map(|&x| f64::from(u8::from(x > 0.5))).collect();
    (xs, ys)
}

/// Quasi-complete separation: x on the tenths grid, y = 1{x > 0.5} except
/// that the rows tied at x = 0.5 alternate between the classes, so the
/// separating hyperplane passes through observed points of both classes.
fn quasi_separated() -> (Vec<f64>, Vec<f64>) {
    let n = 198;
    let xs: Vec<f64> = (0..n).map(|i| (i % 11) as f64 / 10.0).collect();
    let mut tie = 0usize;
    let ys = xs
        .iter()
        .map(|&x| {
            if (x - 0.5).abs() < 1e-12 {
                tie += 1;
                f64::from(u8::from(tie % 2 == 1))
            } else {
                f64::from(u8::from(x > 0.5))
            }
        })
        .collect();
    (xs, ys)
}

/// Rare events: 3 positives in 1000 rows.
fn rare_events() -> (Vec<f64>, Vec<f64>) {
    let n = 1000;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let ys = (0..n)
        .map(|i| f64::from(u8::from(matches!(i, 137 | 512 | 871))))
        .collect();
    (xs, ys)
}

/// The production embedding installs the #784 block quadrature corrector at
/// process start (`init_parallelism`); install it here so these fits run the
/// criterion the Python and CLI fits run.
fn install_production_correctors() {
    drop(gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
        gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
    )));
    drop(gam_problem::rho_posterior::set_rho_posterior_escalator(Box::new(
        gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator,
    )));
}

fn assert_certified_proper_fit(label: &str, xs: &[f64], ys: &[f64], firth: bool) {
    install_production_correctors();
    let ds = dataset(xs, ys);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        firth,
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x)", &ds, &cfg).unwrap_or_else(|err| {
        panic!("{label} (firth={firth}): binomial smooth must mint a certified fit: {err}")
    });
    let StandardFitResult { fit, design, .. } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("{label}: expected a standard fit"),
    };
    // A minted fit is the convergence certificate: under the sealed
    // `FitConvergenceEvidence` contract an uncertified optimum surfaces as a
    // typed error above.
    assert!(
        fit.beta.iter().all(|b| b.is_finite()),
        "{label}: coefficients must be finite, got {:?}",
        fit.beta
    );
    let se = fit
        .beta_standard_errors()
        .unwrap_or_else(|| panic!("{label}: fit must carry a posterior covariance"));
    assert!(
        se.iter().all(|s| s.is_finite()),
        "{label}: coefficient standard errors must be finite, got {se:?}"
    );
    let eta: Array1<f64> = design.design.matrixvectormultiply(&fit.beta);
    for (i, &e) in eta.iter().enumerate() {
        let mu = 1.0 / (1.0 + (-e).exp());
        assert!(
            e.is_finite() && mu > 0.0 && mu < 1.0,
            "{label}: fitted mean at row {i} must lie in (0, 1), got eta={e} mu={mu}"
        );
    }
    eprintln!(
        "{label} firth={firth}: edf={:.3} |g|={:.3e} eta∈[{:.3}, {:.3}]",
        fit.edf_total().unwrap_or(f64::NAN),
        fit.outer_gradient_norm.unwrap_or(f64::NAN),
        eta.iter().copied().fold(f64::INFINITY, f64::min),
        eta.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
}

#[test]
fn perfectly_separated_step_smooth_fits() {
    let (xs, ys) = perfect_step();
    assert_certified_proper_fit("perfect step", &xs, &ys, false);
}

#[test]
fn quasi_separated_smooth_fits() {
    let (xs, ys) = quasi_separated();
    assert_certified_proper_fit("quasi-separation", &xs, &ys, false);
}

#[test]
fn quasi_separated_smooth_fits_with_explicit_firth() {
    let (xs, ys) = quasi_separated();
    assert_certified_proper_fit("quasi-separation", &xs, &ys, true);
}

#[test]
fn rare_events_smooth_fits() {
    let (xs, ys) = rare_events();
    assert_certified_proper_fit("rare events", &xs, &ys, false);
}
