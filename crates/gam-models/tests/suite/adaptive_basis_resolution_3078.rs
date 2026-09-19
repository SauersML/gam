//! #3078 regression: a smooth's default basis size is a data-derived pilot that
//! the fit grows on its own REML evidence, for every smooth family — not a
//! fixed per-family constant (factor-smooth `k = 10`, mgcv-like tensor margins,
//! heuristic cyclic knots, harmonic degree caps).
//!
//! Each recovery test draws a seeded signal whose resolution exceeds what the
//! old constant could represent, fits it through the public `fit_from_formula`
//! path with no `k` given, and requires the fitted surface to track the truth
//! to within the noise level. The null guard fits pure noise and requires the
//! basis to stay at its pilot and the fit to stay at zero, so growth happens
//! only when the data pay for it.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::matrix::LinearOperator;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::smooth::build_term_collection_design;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

fn dataset(names: &[&str], rows: &[Vec<String>]) -> EncodedDataset {
    let headers = names.iter().map(|s| s.to_string()).collect();
    let records: Vec<StringRecord> = rows.iter().map(|r| StringRecord::from(r.clone())).collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// Fit `formula` and return the width of its (single) smooth term together
/// with the fitted mean at every training row.
fn fit(formula: &str, data: &EncodedDataset) -> (usize, Vec<f64>) {
    let FitResult::Standard(fit) = fit_from_formula(formula, data, &gaussian_cfg())
        .unwrap_or_else(|e| panic!("{formula} must fit, got: {e}"))
    else {
        panic!("expected a standard fit for {formula}");
    };
    let design = build_term_collection_design(data.values.view(), &fit.resolvedspec)
        .expect("design at the training rows")
        .design;
    let width = fit.design.smooth.terms[0].coeff_range.len();
    (width, design.apply(&fit.fit.beta).to_vec())
}

/// Recovery error of the fitted mean against the true mean, after removing the
/// common offset (which the intercept and the smooth's centering share).
fn centered_rmse(fitted: &[f64], truth: &[f64]) -> f64 {
    let n = fitted.len() as f64;
    let offset = fitted.iter().zip(truth).map(|(f, t)| f - t).sum::<f64>() / n;
    let ss: f64 = fitted
        .iter()
        .zip(truth)
        .map(|(f, t)| (f - t - offset).powi(2))
        .sum();
    (ss / n).sqrt()
}

fn num(v: f64) -> String {
    v.to_string()
}

const NOISE_SD: f64 = 0.2;

/// Recovery bound: a correctly resolved smooth estimates the mean far more
/// precisely than one observation's noise, so its error must sit at a fraction
/// of the noise sd. A basis too coarse for the signal leaves bias of the order
/// of the signal amplitude, many times this.
const RECOVERY_BOUND: f64 = 0.5 * NOISE_SD;

fn cyclic_signal(x: f64) -> f64 {
    (2.0 * PI * 7.0 * x).sin() + 0.6 * (2.0 * PI * 4.0 * x).cos()
}

fn cyclic_data(n: usize, seed: u64, signal: fn(f64) -> f64) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let noise = Normal::new(0.0, NOISE_SD).unwrap();
    let mut rows = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        let x: f64 = ux.sample(&mut rng);
        truth.push(signal(x));
        rows.push(vec![num(x), num(signal(x) + noise.sample(&mut rng))]);
    }
    (dataset(&["x", "y"], &rows), truth)
}

#[test]
fn cyclic_smooth_resolves_a_high_frequency_periodic_signal() {
    let (data, truth) = cyclic_data(2000, 3078, cyclic_signal);
    let (width, fitted) = fit("y ~ s(x, bs=cc)", &data);
    let err = centered_rmse(&fitted, &truth);
    assert!(
        err < RECOVERY_BOUND,
        "cyclic recovery RMSE {err:.4} (width {width}) must sit below {RECOVERY_BOUND}"
    );
}

#[test]
fn cyclic_smooth_on_pure_noise_stays_at_its_pilot_and_at_zero() {
    let n = 2000;
    let (null_data, zero) = cyclic_data(n, 3078, |_| 0.0);
    let (signal_data, _) = cyclic_data(n, 3078, cyclic_signal);
    let (null_width, null_fitted) = fit("y ~ s(x, bs=cc)", &null_data);
    let (signal_width, _) = fit("y ~ s(x, bs=cc)", &signal_data);
    assert!(
        null_width < signal_width,
        "noise must not grow the basis: null width {null_width} vs signal width {signal_width}"
    );
    // One free coefficient estimated from n rows has sd NOISE_SD/√n; the null
    // fit must shrink to within a few of those rather than track the noise.
    let resolution = NOISE_SD / (n as f64).sqrt();
    let err = centered_rmse(&null_fitted, &zero);
    assert!(
        err < 5.0 * resolution,
        "null cyclic fit RMSE {err:.5} must shrink to zero (per-coefficient sd {resolution:.5})"
    );
}

fn group_signal(x: f64, g: usize) -> f64 {
    let freq = [5.0, 6.0, 7.0][g];
    (2.0 * PI * freq * x).sin() + 0.5 * g as f64
}

#[test]
fn factor_smooth_resolves_group_curves_beyond_a_fixed_basis_dimension() {
    let mut rng = StdRng::seed_from_u64(30781);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let noise = Normal::new(0.0, NOISE_SD).unwrap();
    let per_group = 1000;
    let mut rows = Vec::with_capacity(3 * per_group);
    let mut truth = Vec::with_capacity(3 * per_group);
    for g in 0..3 {
        for _ in 0..per_group {
            let x: f64 = ux.sample(&mut rng);
            truth.push(group_signal(x, g));
            rows.push(vec![
                num(x),
                format!("g{g}"),
                num(group_signal(x, g) + noise.sample(&mut rng)),
            ]);
        }
    }
    let data = dataset(&["x", "g", "y"], &rows);
    let (width, fitted) = fit("y ~ s(x, g, bs=fs)", &data);
    let err = centered_rmse(&fitted, &truth);
    assert!(
        err < RECOVERY_BOUND,
        "factor-smooth recovery RMSE {err:.4} (width {width}) must sit below {RECOVERY_BOUND}"
    );
}

fn tensor_signal(x: f64, z: f64) -> f64 {
    (2.0 * PI * 3.0 * x).sin() * (2.0 * PI * 3.0 * z).sin()
}

#[test]
fn tensor_smooth_margins_grow_to_resolve_an_oscillating_surface() {
    let mut rng = StdRng::seed_from_u64(30782);
    let ux = Uniform::new(0.0, 1.0).unwrap();
    let noise = Normal::new(0.0, NOISE_SD).unwrap();
    let n = 3000;
    let mut rows = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        let (x, z): (f64, f64) = (ux.sample(&mut rng), ux.sample(&mut rng));
        truth.push(tensor_signal(x, z));
        rows.push(vec![
            num(x),
            num(z),
            num(tensor_signal(x, z) + noise.sample(&mut rng)),
        ]);
    }
    let data = dataset(&["x", "z", "y"], &rows);
    let (width, fitted) = fit("y ~ te(x, z)", &data);
    let err = centered_rmse(&fitted, &truth);
    assert!(
        err < RECOVERY_BOUND,
        "tensor recovery RMSE {err:.4} (width {width}) must sit below {RECOVERY_BOUND}"
    );
}

/// Legendre polynomial `P_l(t)` by the three-term recurrence.
fn legendre(l: usize, t: f64) -> f64 {
    let (mut p0, mut p1) = (1.0, t);
    if l == 0 {
        return p0;
    }
    for k in 1..l {
        let kf = k as f64;
        (p0, p1) = (p1, ((2.0 * kf + 1.0) * t * p1 - kf * p0) / (kf + 1.0));
    }
    p1
}

/// Zonal degree-8 harmonic `P_8(sin lat)`.
fn sphere_signal(lat_deg: f64) -> f64 {
    legendre(8, lat_deg.to_radians().sin())
}

#[test]
fn harmonic_sphere_degree_grows_to_resolve_a_high_degree_harmonic() {
    let mut rng = StdRng::seed_from_u64(30783);
    let us = Uniform::new(-1.0, 1.0).unwrap();
    let ulon = Uniform::new(-180.0, 180.0).unwrap();
    let noise = Normal::new(0.0, NOISE_SD).unwrap();
    let n = 3000;
    let mut rows = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        // Uniform on the sphere: sin(lat) is uniform on [-1, 1].
        let s: f64 = us.sample(&mut rng);
        let lat = s.asin().to_degrees();
        let lon: f64 = ulon.sample(&mut rng);
        truth.push(sphere_signal(lat));
        rows.push(vec![
            num(lat),
            num(lon),
            num(sphere_signal(lat) + noise.sample(&mut rng)),
        ]);
    }
    let data = dataset(&["lat", "lon", "y"], &rows);
    let (width, fitted) = fit("y ~ sphere(lat, lon, method=harmonic)", &data);
    let err = centered_rmse(&fitted, &truth);
    assert!(
        err < RECOVERY_BOUND,
        "harmonic sphere recovery RMSE {err:.4} (width {width}) must sit below {RECOVERY_BOUND}"
    );
}
