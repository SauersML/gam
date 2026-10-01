//! Two properties of the standard workflow's resolution loop.
//!
//! A Poisson prior weight is a frequency: `w` on a row is exactly `w`
//! replicated rows, so the weighted fit and the fit of the replicated rows must
//! reach the same basis and the same curve. The loop's pilot used to be sized
//! from the raw row count, so the two started (and ended) at different bases
//! and disagreed by 8e-4 relative.
//!
//! A formula-default tensor `te(x0, x1)` is grown on its own REML evidence like
//! every other default smooth, margin by margin. At its fixed provisioned size
//! it saturated on `x0·sin(x1)` (R² 0.982, recovery error 0.26 against a noise
//! sd of 0.1) while flagging its own lack of fit.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::matrix::LinearOperator;
use gam_models::fit_orchestration::{
    FitConfig, FitResult, fit_from_formula, fit_from_formula_with_notes,
};
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};

fn dataset(names: &[&str], rows: &[Vec<f64>]) -> EncodedDataset {
    let headers = names.iter().map(|s| s.to_string()).collect();
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|r| StringRecord::from(r.iter().map(|v| v.to_string()).collect::<Vec<_>>()))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

/// The fitted linear predictor of a standard fit at `rows` (one column per
/// dataset column, in the dataset's order).
fn eta_at(fit: &FitResult, rows: &Array2<f64>) -> Vec<f64> {
    let FitResult::Standard(fit) = fit else {
        panic!("expected a standard fit");
    };
    build_term_collection_design(rows.view(), &fit.resolvedspec)
        .expect("design at the evaluation rows")
        .design
        .apply(&fit.fit.beta)
        .to_vec()
}

#[test]
fn poisson_integer_weights_fit_the_replicated_rows() {
    let mut rng = StdRng::seed_from_u64(7);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let n = 300;
    let mut weighted = Vec::with_capacity(n);
    let mut replicated = Vec::new();
    for i in 0..n {
        let x: f64 = ux.sample(&mut rng);
        let rate = (0.5 + (6.0 * x).sin()).exp();
        let y: f64 = Poisson::new(rate).unwrap().sample(&mut rng);
        let w = (i % 3 + 1) as f64;
        weighted.push(vec![x, y, w]);
        for _ in 0..(i % 3 + 1) {
            replicated.push(vec![x, y]);
        }
    }
    let poisson = |weight_column: Option<&str>| FitConfig {
        family: Some("poisson".to_string()),
        weight_column: weight_column.map(str::to_string),
        ..FitConfig::default()
    };
    let by_weight = fit_from_formula(
        "y ~ s(x)",
        &dataset(&["x", "y", "w"], &weighted),
        &poisson(Some("w")),
    )
    .expect("weighted Poisson fit");
    let by_rows = fit_from_formula(
        "y ~ s(x)",
        &dataset(&["x", "y"], &replicated),
        &poisson(None),
    )
    .expect("replicated Poisson fit");
    let grid = 50;
    let at = |cols: usize| {
        Array2::from_shape_fn((grid, cols), |(i, j)| {
            if j == 0 {
                0.02 + 0.96 * i as f64 / (grid - 1) as f64
            } else {
                0.0
            }
        })
    };
    let mu_weight = eta_at(&by_weight, &at(3));
    let mu_rows = eta_at(&by_rows, &at(2));
    for (i, (a, b)) in mu_weight.iter().zip(&mu_rows).enumerate() {
        let relative = (a.exp() - b.exp()).abs() / b.exp();
        assert!(
            relative <= 1e-4,
            "grid point {i}: weighted mean {} vs replicated {} (relative {relative:.2e})",
            a.exp(),
            b.exp()
        );
    }
}

#[test]
fn default_tensor_grows_until_it_resolves_the_interaction() {
    let mut rng = StdRng::seed_from_u64(0);
    let ux = Uniform::new(-5.0, 5.0).unwrap();
    let noise_sd = 0.1;
    let noise = Normal::new(0.0, noise_sd).unwrap();
    let n = 3000;
    let mut rows = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        let x0: f64 = ux.sample(&mut rng);
        let x1: f64 = ux.sample(&mut rng);
        let f = x0 * x1.sin();
        truth.push(f);
        rows.push(vec![x0, x1, f + noise.sample(&mut rng)]);
    }
    let data = dataset(&["x0", "x1", "y"], &rows);
    let outcome = fit_from_formula_with_notes(
        "y ~ te(x0, x1)",
        &data,
        &FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        },
    )
    .expect("default tensor fit");
    let fit = outcome.result;
    let width = match &fit {
        FitResult::Standard(standard) => standard.design.smooth.terms[0].coeff_range.len(),
        _ => panic!("expected a standard fit"),
    };
    let fitted = eta_at(&fit, &data.values.to_owned());
    let rmse = (fitted
        .iter()
        .zip(&truth)
        .map(|(f, t)| (f - t).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    // A resolved surface estimates the mean far more precisely than one
    // observation's noise; a saturated basis leaves bias of the signal's order.
    assert!(
        rmse < 0.5 * noise_sd,
        "te(x0, x1) recovery RMSE {rmse:.4} (width {width}) must sit below half the noise \
         sd {noise_sd}; advisories: {:?}",
        outcome.inference_notes.advisories
    );
}
