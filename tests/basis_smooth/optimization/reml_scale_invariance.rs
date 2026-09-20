//! REML's marginal-likelihood objective is mathematically invariant under
//! positive scalar rescaling of the penalty matrix: if `S → c·S`, the
//! optimum simply shifts `λ → λ/c` and the fitted `β̂` is unchanged. Any
//! operation in the optimizer pipeline that breaks this invariance is a
//! bug.
//!
//! We test it indirectly via the Sobolev / pseudo-spline Wahba sphere
//! kernels: both define valid PSD reproducing kernels at every supported
//! `m`, but their Gram matrices differ in Frobenius scale by factors of
//! 8 – 60 (see `sphere_wahba_kernels_are_distinct.rs` for the numbers).
//! If REML is scale-invariant, both kernels should reach near-equivalent
//! fit quality on the same data — they should not produce one fit that
//! collapses and another that fits cleanly purely on account of the
//! kernel scale.
//!
//! At HEAD: pseudo-spline `m=4` historically collapsed (rmse 0.43,
//! predictions = response mean) while Sobolev `m=4` fits to rmse 0.0045.
//! After the agent's REML rho-adjoint fix, both kernels now reach
//! rmse ≈ 0.005 on the same data — confirming REML is at least *broadly*
//! scale-invariant on this test case.

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

/// Observations in the fixture.
const N_OBS: usize = 400;
/// Standard deviation of the Gaussian response noise.
const NOISE_SD: f64 = 0.05;
/// Sphere smooth basis dimension; with the intercept the model has at most
/// `K_BASIS + 1` coefficients.
const K_BASIS: usize = 30;

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

fn fit_predict(formula: &str) -> Vec<f64> {
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
    design.design.apply(&fit.fit.beta).to_vec()
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
fn reml_pseudo_and_sobolev_m4_both_recover_smooth_truth() {
    // The smoking-gun pair: pseudo-spline m=4 was the historical collapse
    // case (kernel values ~3e-4, REML pushed smooth to ~0). Sobolev m=4
    // has kernel values ~5× larger. If REML weren't scale-invariant the
    // two fits would differ wildly. They should produce essentially the
    // same predictions (different λ in the original kernel units, same
    // effective smoother).
    init_parallelism();
    let pred_sob = fit_predict(&format!(
        "y ~ sphere(lat, lon, k={K_BASIS}, penalty_order=4, method=sobolev)"
    ));
    let pred_pse = fit_predict(&format!(
        "y ~ sphere(lat, lon, k={K_BASIS}, penalty_order=4, method=pseudo)"
    ));
    let rmse_sob = rmse(&pred_sob);
    let rmse_pse = rmse(&pred_pse);
    // Bar derived from the fixture, not chosen. For a linear smoother the
    // design-averaged variance of the fitted mean is σ²·edf/n ≤ σ²·p/n, with
    // p = K_BASIS + 1 coefficients. The truth (a constant plus degree-1
    // spherical harmonics) is resolved by the basis, so the bias is negligible
    // (historical rmse 0.0045 < σ·√(p/n) ≈ 0.014). The evaluation grid follows
    // the data's uniform lat/lon law, so the design average applies to it.
    // Three times that ceiling leaves no noise-driven flakiness and still
    // refuses any REML over-smoothing that loses more than a few σ·√(p/n):
    // exactly what a scale-dependent smoothing choice would do.
    let p = (K_BASIS + 1) as f64;
    let rmse_bar = 3.0 * NOISE_SD * (p / N_OBS as f64).sqrt();
    eprintln!(
        "[reml-scale] m=4: rmse_sob={rmse_sob:.4} rmse_pse={rmse_pse:.4} bar={rmse_bar:.4}"
    );
    assert!(
        rmse_sob < rmse_bar,
        "Sobolev m=4 rmse={rmse_sob:.4} exceeds the derived bar {rmse_bar:.4} \
         (3σ√(p/n); historical 0.0045)",
    );
    assert!(
        rmse_pse < rmse_bar,
        "Pseudo m=4 rmse={rmse_pse:.4} exceeds the derived bar {rmse_bar:.4} \
         (3σ√(p/n)); REML chose a smoother the Sobolev kernel does not, so the \
         pipeline is not invariant to the penalty's scale (historical collapse: 0.43)",
    );
    // Pointwise agreement. They are different RKHS, so not byte-for-byte, but
    // each prediction's maximum error over the N grid points is at most the
    // Gaussian-max factor √(2 ln N) times its rms bound, and the triangle
    // inequality adds the two.
    let max_abs_diff: f64 = pred_sob
        .iter()
        .zip(pred_pse.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let n_grid = pred_sob.len() as f64;
    let diff_bar = 2.0 * (2.0 * n_grid.ln()).sqrt() * rmse_bar;
    eprintln!("[reml-scale] m=4: max |Δ pred| = {max_abs_diff:.4} bar={diff_bar:.4}");
    assert!(
        max_abs_diff < diff_bar,
        "Sobolev m=4 and Pseudo m=4 fits disagree by max {max_abs_diff:.4} \
         (derived budget {diff_bar:.4} = 2·√(2 ln N)·3σ√(p/n)); REML picked \
         different effective smoothers for the two kernel scales.",
    );
}
