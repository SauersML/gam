//! REML's marginal-likelihood objective is mathematically invariant under
//! positive scalar rescaling of the penalty matrix: if `S → c·S`, the
//! optimum simply shifts `λ → λ/c` and the fitted `β̂` is unchanged. Any
//! operation in the optimizer pipeline that breaks this invariance is a
//! bug.
//!
//! We test it indirectly via the two sphere constructions: the Sobolev
//! Wahba reproducing kernel and the spherical-harmonic basis carry the same
//! Laplace-Beltrami penalty order `m` but their penalty matrices sit on very
//! different numerical scales. If REML is scale-invariant, both reach
//! near-equivalent fit quality on the same data — neither collapses to the
//! response mean purely on account of its penalty scale.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Observation-noise standard deviation of the fixture.
const NOISE_SD: f64 = 0.05;
/// Number of fitted observations.
const N_OBS: usize = 400;

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, NOISE_SD).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.6 * lat.to_radians().sin()
            + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn truth(lat: f64, lon: f64) -> f64 {
    0.5 + 0.6 * lat.to_radians().sin() + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
}

/// Fit `formula` and return its predictions on the evaluation grid together
/// with the number of fitted coefficients `p`.
fn fit_predict(formula: &str) -> (Vec<f64>, usize) {
    let data = make_dataset(N_OBS);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..15 {
            let lon = -175.0 + 350.0 * (j as f64) / 14.0;
            pts.push((lat, lon));
        }
    }
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
    (
        design.design.apply(&fit.fit.beta).to_vec(),
        fit.fit.beta.len(),
    )
}

fn rmse(pred: &[f64]) -> f64 {
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..15 {
            let lon = -175.0 + 350.0 * (j as f64) / 14.0;
            pts.push((lat, lon));
        }
    }
    let sumsq: f64 = pred
        .iter()
        .zip(pts.iter())
        .map(|(p, (lat, lon))| (p - truth(*lat, *lon)).powi(2))
        .sum();
    (sumsq / pred.len() as f64).sqrt()
}

#[test]
fn reml_harmonic_and_sobolev_m4_both_recover_smooth_truth() {
    // m=4 is the high-order case where a scale-sensitive pipeline pushes one
    // construction's smooth to ~0. If REML weren't scale-invariant the two
    // fits would differ wildly. They should produce essentially the same
    // predictions (different λ in each construction's units, same effective
    // smoother).
    init_parallelism();
    let (pred_sob, p_sob) =
        fit_predict("y ~ sphere(lat, lon, k=30, penalty_order=4, method=sobolev)");
    let (pred_har, p_har) =
        fit_predict("y ~ sphere(lat, lon, k=30, penalty_order=4, method=harmonic)");
    let rmse_sob = rmse(&pred_sob);
    let rmse_har = rmse(&pred_har);
    // Both bars come from the fixture. A linear smoother with p
    // coefficients has design-averaged variance sigma^2 * edf / n
    // <= sigma^2 * p / n. The truth is a constant plus degree-1 spherical
    // harmonics, which the basis resolves, so the variance term bounds the
    // rmse. The equispaced grid follows the data's uniform lat/lon law.
    // Three times that ceiling leaves no room for noise-driven failure,
    // and still catches any over-smoothing that loses more than a few
    // sigma * sqrt(p / n).
    let p = p_sob.max(p_har) as f64;
    let rmse_bar = 3.0 * NOISE_SD * (p / N_OBS as f64).sqrt();
    eprintln!(
        "[reml-scale] m=4: rmse_sob={rmse_sob:.4} rmse_har={rmse_har:.4} bar={rmse_bar:.4} (p={p})"
    );
    assert!(
        rmse_sob < rmse_bar,
        "Sobolev m=4 collapsed: rmse={rmse_sob:.4} exceeds 3*sigma*sqrt(p/n)={rmse_bar:.4}",
    );
    assert!(
        rmse_har < rmse_bar,
        "harmonic m=4 collapsed: rmse={rmse_har:.4} exceeds 3*sigma*sqrt(p/n)={rmse_bar:.4}; \
         the REML pipeline needs to be scale-invariant for this case to work",
    );
    // Pointwise agreement. The largest of N Gaussian errors whose rms is
    // at most rmse_bar is at most sqrt(2 ln N) * rmse_bar. The triangle
    // inequality then bounds the disagreement of the two fits by twice
    // that.
    let n_grid = pred_sob.len() as f64;
    let agreement_bar = 2.0 * (2.0 * n_grid.ln()).sqrt() * rmse_bar;
    let max_abs_diff: f64 = pred_sob
        .iter()
        .zip(pred_har.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("[reml-scale] m=4: max |Δ pred| = {max_abs_diff:.4} (bar {agreement_bar:.4})");
    assert!(
        max_abs_diff < agreement_bar,
        "Sobolev m=4 and harmonic m=4 fits disagree by max {max_abs_diff:.4} \
         (bar 2*sqrt(2 ln N)*rmse_bar = {agreement_bar:.4}). REML picked different \
         effective smoothers for kernels that differ only in scale.",
    );
}
