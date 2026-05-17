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

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
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
    let data = make_dataset(400);
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
    let pred_sob = fit_predict("y ~ sphere(lat, lon, k=30, m=4, kernel=sobolev)");
    let pred_pse = fit_predict("y ~ sphere(lat, lon, k=30, m=4, kernel=pseudo)");
    let rmse_sob = rmse(&pred_sob);
    let rmse_pse = rmse(&pred_pse);
    eprintln!("[reml-scale] m=4: rmse_sob={rmse_sob:.4} rmse_pse={rmse_pse:.4}");
    assert!(
        rmse_sob < 0.10,
        "Sobolev m=4 collapsed: rmse={rmse_sob:.4} (historical 0.0045 expected)",
    );
    assert!(
        rmse_pse < 0.10,
        "Pseudo m=4 collapsed: rmse={rmse_pse:.4} — this is the historical mgcv-pseudo \
         m=4 collapse; the REML pipeline needs to be scale-invariant for this case to work",
    );
    // Pointwise the two fits should agree within a generous tolerance,
    // not byte-for-byte (they're different RKHS) but qualitatively
    // (REML chooses near-equivalent smoothers).
    let max_abs_diff: f64 = pred_sob
        .iter()
        .zip(pred_pse.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("[reml-scale] m=4: max |Δ pred| = {max_abs_diff:.4}");
    assert!(
        max_abs_diff < 0.50,
        "Sobolev m=4 and Pseudo m=4 fits disagree by max {max_abs_diff:.4} \
         (budget 0.50). Either the kernels are wildly different or REML \
         picked very different effective smoothers — likely a scale-invariance issue.",
    );
}
