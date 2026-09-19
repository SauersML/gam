//! `bounded(x, min, max)` with no `prior=` shrinks toward the null.
//!
//! A box is a restriction on where a coefficient may sit, not evidence that it
//! is nonzero. SPEC.md: "In general, the default should allow a configuration
//! which recovers the null (empirical Bayes-like). Users must opt-in to
//! overfitting potential, not the other way around." Before this default
//! existed the bare `bounded()` term was a constrained MLE: a
//! covariate with no effect at all kept its full sampling-noise slope, while
//! the same covariate written as `linear(x)` was shrunk by its REML-weighted
//! null-recovery ridge.
//!
//! The fixture pins the noise's least-squares slope at half its standard
//! error, so the unpenalised fit (`prior=none`) reports a clearly nonzero
//! slope, while REML's single-coefficient variance estimate
//! `max(0, beta_hat^2 - se^2)` is zero and the default fit must return the
//! slope to `beta = 0`.

use gam::data::EncodedDataset;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use std::io::Write;

const N: usize = 200;
const INTERCEPT: f64 = 1.0;
const NOISE_SD: f64 = 0.5;
/// Least-squares slope of the noise, in units of its standard error.
const NOISE_SLOPE_IN_SE: f64 = 0.5;

/// Deterministic standard-normal draws (splitmix64 + Box-Muller).
fn standard_normals(count: usize, mut state: u64) -> Vec<f64> {
    let mut next_uniform = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let (u1, u2) = (next_uniform(), next_uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

/// `x` on an even grid over `[-1, 1]` and `y = INTERCEPT + e`, where `e` is
/// Gaussian noise whose component along centred `x` is replaced so that its
/// least-squares slope is exactly `NOISE_SLOPE_IN_SE` standard errors.
fn fixture() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..N)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((N - 1) as f64))
        .collect();
    let x_mean = x.iter().sum::<f64>() / N as f64;
    let xc: Vec<f64> = x.iter().map(|&xi| xi - x_mean).collect();
    let sxx: f64 = xc.iter().map(|v| v * v).sum();
    let mut e: Vec<f64> = standard_normals(N, 0x5EED_2026)
        .into_iter()
        .map(|z| NOISE_SD * z)
        .collect();
    let e_mean = e.iter().sum::<f64>() / N as f64;
    e.iter_mut().for_each(|v| *v -= e_mean);
    let slope = xc.iter().zip(&e).map(|(a, b)| a * b).sum::<f64>() / sxx;
    e.iter_mut().zip(&xc).for_each(|(v, a)| *v -= slope * a);
    let target = NOISE_SLOPE_IN_SE * NOISE_SD / sxx.sqrt();
    e.iter_mut().zip(&xc).for_each(|(v, a)| *v += target * a);
    let y = e.iter().map(|v| INTERCEPT + v).collect();
    (x, y)
}

fn dataset(x: &[f64], y: &[f64]) -> EncodedDataset {
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    static FIXTURE_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let serial = FIXTURE_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_bounded_shrinkage_{}_{serial}.csv",
        std::process::id()
    ));
    {
        let mut file = std::fs::File::create(&tmp).expect("create fixture csv");
        file.write_all(csv.as_bytes()).expect("write fixture csv");
    }
    let loaded = load_csvwith_inferred_schema(&tmp).expect("load fixture");
    std::fs::remove_file(&tmp).expect("remove the fixture csv this test created");
    loaded
}

fn bounded_slope(formula: &str, data: &EncodedDataset) -> f64 {
    match fit_from_formula(formula, data, &FitConfig::default()) {
        Ok(FitResult::Standard(fit)) => fit.fit.beta[1],
        Ok(_) => panic!("`{formula}` is a Standard GAM fit"),
        Err(error) => panic!("`{formula}` must fit: {error}"),
    }
}

#[test]
fn bounded_default_prior_shrinks_a_null_coefficient_to_zero() {
    init_parallelism();
    let (x, y) = fixture();
    let data = dataset(&x, &y);

    // Non-vacuity: without a prior the box is interior and the slope is the
    // noise's least-squares slope, half a standard error away from zero.
    let unpenalised = bounded_slope("y ~ bounded(x, min=-1, max=1, prior=none)", &data);
    assert!(
        unpenalised.abs() > 0.02,
        "the unpenalised bounded slope must carry the fixture's noise slope, got {unpenalised}"
    );

    let default = bounded_slope("y ~ bounded(x, min=-1, max=1)", &data);
    let explicit = bounded_slope("y ~ bounded(x, min=-1, max=1, prior=shrinkage)", &data);
    assert!(
        (default - explicit).abs() <= 1e-12,
        "the default bounded prior must be the shrinkage prior: {default} vs {explicit}"
    );
    assert!(
        default.abs() <= 0.1 * unpenalised.abs(),
        "a bounded coefficient with no true effect must shrink toward 0: default {default} \
         against the unpenalised {unpenalised}"
    );
}

/// When zero is outside the box the null is not an admissible value, and the
/// shrinkage prior centres at the box midpoint. A null covariate therefore
/// lands at the midpoint, not at a bound.
#[test]
fn bounded_shrinkage_prior_centres_at_the_midpoint_when_zero_is_outside_the_box() {
    init_parallelism();
    let (x, y) = fixture();
    let data = dataset(&x, &y);
    let slope = bounded_slope("y ~ bounded(x, min=1, max=3)", &data);
    let unpenalised = bounded_slope("y ~ bounded(x, min=1, max=3, prior=none)", &data);
    assert!(
        (1.0..=3.0).contains(&slope),
        "the shrunk slope must honour the box, got {slope}"
    );
    assert!(
        (slope - 2.0).abs() < (unpenalised - 2.0).abs(),
        "the shrinkage prior must pull a null slope toward the box midpoint 2: shrunk {slope}, \
         unpenalised {unpenalised}"
    );
}

/// The shrinkage strength is chosen against the residual variance, so the
/// default fit is equivariant in the units of `y`: rescaling the response and
/// the box together by `c` rescales the fitted slope by exactly `c`. The
/// bounded family's Gaussian rows carry unit dispersion, and a REML criterion
/// run at that dispersion weighs the prior against the data as if the residual
/// variance were 1 in whatever units `y` has, so the same data in milli-units
/// were shrunk differently.
#[test]
fn bounded_shrinkage_is_equivariant_in_the_units_of_the_response() {
    init_parallelism();
    const SLOPE: f64 = 0.1;
    const UNITS: f64 = 1000.0;
    let (x, y) = fixture();
    let y: Vec<f64> = x.iter().zip(&y).map(|(xi, yi)| yi + SLOPE * xi).collect();
    let y_scaled: Vec<f64> = y.iter().map(|v| UNITS * v).collect();
    let slope = bounded_slope("y ~ bounded(x, min=-1, max=1)", &dataset(&x, &y));
    let slope_scaled = bounded_slope(
        "y ~ bounded(x, min=-1000, max=1000)",
        &dataset(&x, &y_scaled),
    );
    assert!(
        (slope_scaled / UNITS - slope).abs() <= 1e-6 * slope.abs(),
        "the default bounded fit must not depend on the units of y: slope {slope} against \
         {slope_scaled} / {UNITS} = {}",
        slope_scaled / UNITS
    );
}
