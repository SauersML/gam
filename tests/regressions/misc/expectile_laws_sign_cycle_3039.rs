//! #3039: on seed 1001 at τ = 0.9 the LAWS sign map has no fixed point. One
//! boundary row's weight flip moves the REML-selected λ̂ enough to flip that
//! row's own residual back, so the map cycles with period 2 and the fit used
//! to end in a typed "sign-pattern cycle" error. The estimator is the
//! generalized (Clarke) fixed point of the subgradient weight map, where a row
//! at `r = 0` carries a fractional asymmetry; it exists by the intermediate
//! value theorem and the fit must reach it.
//!
//! The data are the heteroscedastic-truth draws of the expectile band coverage
//! quality test, `simulate(1001)`.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::TAU;

const N: usize = 800;

fn simulate(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform x");
    let z = Normal::new(0.0, 1.0).expect("normal");
    let x: Vec<f64> = (0..N).map(|_| unif.sample(&mut rng)).collect();
    let y = x
        .iter()
        .map(|&t| (TAU * t).sin() + (0.2 + 0.8 * t) * z.sample(&mut rng))
        .collect();
    (x, y)
}

fn encode(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y)
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode expectile data")
}

#[test]
fn expectile_laws_reaches_generalized_fixed_point_through_sign_cycle_3039() {
    let tau = 0.9;
    let (x, y) = simulate(1001);
    let cfg = FitConfig {
        family: Some(format!("expectile({tau})")),
        ..FitConfig::default()
    };
    let fit = match fit_from_formula("y ~ s(x)", &encode(&x, &y), &cfg)
        .expect("the τ = 0.9 expectile on seed 1001 must reach its certified fixed point")
    {
        FitResult::Standard(fit) => fit,
        _ => panic!("an expectile fit is a standard fit"),
    };
    let mu = fit
        .design
        .apply(fit.fit.beta.view())
        .expect("fitted expectile surface");
    assert!(mu.iter().all(|v| v.is_finite()));

    // Independent consequence of stationarity: the intercept column is
    // unpenalized and every smooth column is sum-to-zero centred, so the
    // intercept component of ∇J is the scalar balance Σ w_τ(r_i)·r_i. The
    // certificate bounds it by tol·sqrt(Σw · Σw r²) (Cauchy–Schwarz scale of
    // the all-ones column); the inner WLS stationarity adds its own
    // solver-precision residual, well below that scale.
    let (mut balance, mut weight_sum, mut energy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..N {
        let r = y[i] - mu[i];
        let w = if r > 0.0 { tau } else { 1.0 - tau };
        balance += w * r;
        weight_sum += w;
        energy += w * r * r;
    }
    let scale = (weight_sum * energy).sqrt();
    let relative_balance = balance.abs() / scale;
    eprintln!("#3039 seed 1001 tau 0.9: |Σ w r| / sqrt(Σw Σw r²) = {relative_balance:.3e}");
    assert!(
        relative_balance <= 1e-8,
        "expectile balance defect {relative_balance:.3e} is not at a stationary point"
    );

    // The fitted curve is the τ-expectile of y | x: the planted truth is
    // sin(2πx) + (0.2 + 0.8x)·e_τ with e_0.9 ≈ 0.8617, so the fraction of
    // responses above it is 1 − Φ(e_0.9) ≈ 0.194, with binomial SE ≈ 0.014 at
    // n = 800.
    let above = (0..N).filter(|&i| y[i] > mu[i]).count() as f64 / N as f64;
    eprintln!("#3039 seed 1001 tau 0.9: fraction above fitted expectile = {above:.4}");
    assert!(
        (above - 0.194).abs() <= 5.0 * 0.014,
        "fraction above the fitted 0.9-expectile {above:.4} is far from 1 − Φ(e_0.9) ≈ 0.194"
    );
}
