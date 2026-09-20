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
//! error, so the unpenalised fit (`prior=uniform`) reports a clearly nonzero
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

fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let x_mean = x.iter().sum::<f64>() / x.len() as f64;
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - x_mean) * (b - y_mean)).sum();
    let sxx: f64 = x.iter().map(|a| (a - x_mean) * (a - x_mean)).sum();
    sxy / sxx
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
    let unpenalised = bounded_slope("y ~ bounded(x, min=-1, max=1, prior=uniform)", &data);
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
    // The unpenalised fit is the box-constrained least-squares slope. With the
    // intercept profiled out the Gaussian objective is a convex quadratic in
    // the slope with its minimum at the OLS slope, so the constrained optimum
    // is that slope clamped to the box: here the rail `min = 1`.
    let unpenalised = ols_slope(&x, &y).clamp(1.0, 3.0);
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

/// An unpenalised bounded fit whose constrained optimum sits on a bound has
/// its MODE on the rail (gam#3289, gam#3923). The least-squares slope here is
/// the noise's, about 0.035, so `bounded(x, min=1, max=3, prior=uniform)` has its
/// box-constrained optimum at exactly `beta = min = 1`: with the intercept
/// profiled out the Gaussian objective is a convex quadratic in the slope,
/// minimised at the OLS slope, and the constrained optimum is that slope
/// clamped to the box.
///
/// `prior=none` used to be a flat prior on the logit chart of the interval
/// transform. That prior is improper: as the latent coordinate runs to either
/// infinity the likelihood tends to its value at the rail, a positive
/// constant, so the flat-latent posterior has infinite mass and no mean. The
/// fit stalled at the chart's injective clamp, read that flat direction as an
/// improper posterior, armed the Jeffreys term (#979) and published 1.0076, a
/// summary of neither the constrained MLE nor any stated posterior. And
/// `prior=uniform` was flat on the box pulled back to the chart (the
/// log-Jacobian correction): proper, but the fit published the chart's mode,
/// 1.008280 here against the box posterior's mean 1.008165 (gam#3479).
///
/// Both were the same prior, flat on the box, so there is one spelling,
/// `prior=uniform` (`none` is refused with a pointer to it), and it is the
/// box-constrained linear term: flat on the box of the coefficient itself, a
/// proper posterior because the box is bounded. Its mode is the constrained
/// MLE, and what it publishes is that truncated posterior's mean (SPEC:
/// "Posterior mean must always be the default (never MAP)"), which sits
/// strictly inside the box. Inside a box that does not bind it is the
/// unconstrained unpenalised fit, the same numbers as `linear(x, min, max,
/// double_penalty=false)` there too.
#[test]
fn bounded_unpenalised_fit_is_the_box_constrained_linear_posterior() {
    init_parallelism();
    let (x, y) = fixture();
    let data = dataset(&x, &y);
    let ols = ols_slope(&x, &y);
    assert!(ols < 1.0, "non-vacuity: the least-squares slope {ols} must lie below the box");

    let standard = |formula: &str| match fit_from_formula(formula, &data, &FitConfig::default()) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("`{formula}` is a Standard GAM fit"),
        Err(error) => panic!("`{formula}` must fit: {error}"),
    };
    let bounded = standard("y ~ bounded(x, min=1, max=3, prior=uniform)");
    let linear = standard("y ~ linear(x, min=1, max=3, double_penalty=false)");

    let mode = bounded
        .fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
        .expect("prior=uniform carries the box as inequality constraints")
        .mode[1];
    // The active row is held as an equality in the solver's conditioned frame:
    // the bound is carried there as `min*s` for a column scale `s` and the mode
    // comes back as `beta/s`, two roundings of relative size at most `eps/2`
    // each, so the mode is within `2*eps*|min|` of `min`.
    assert!(
        (mode - 1.0).abs() <= 2.0 * f64::EPSILON,
        "the prior=uniform mode is the constrained maximum-likelihood fit, the rail min = 1; \
         got {mode:.17}"
    );

    let reported = bounded.fit.beta[1];
    assert!(
        reported > 1.0 && reported < 3.0,
        "the published coefficient is the truncated posterior's mean, strictly inside the \
         box; got {reported:.17}"
    );
    assert!(
        reported == linear.fit.beta[1],
        "prior=uniform must be the box-constrained linear fit: bounded {reported:.17} against \
         linear {:.17}",
        linear.fit.beta[1]
    );

    let interior = bounded_slope("y ~ bounded(x, min=-1, max=1, prior=uniform)", &data);
    let interior_linear =
        bounded_slope("y ~ linear(x, min=-1, max=1, double_penalty=false)", &data);
    assert!(
        interior == interior_linear,
        "inside a box that does not bind, prior=uniform must be the box-constrained linear \
         fit: bounded {interior:.17} against linear {interior_linear:.17}"
    );
}

/// A box that does not bind leaves the fit unconstrained, so on an unpenalised
/// coefficient the box-constrained `linear(x, min, max)` route must return the `linear()` route's slope
/// (#3339). Both solve the same two-column least-squares problem: the box is
/// interior (the noise slope 0.03 sits deep inside `[-1, 1]`), and neither term
/// carries a penalty. The constrained route used to refuse here: its
/// constraint-KKT gate measured stationarity against `‖score‖ + ‖S_λβ‖`, and at
/// an unpenalised interior optimum both vanish, so a fully converged fit read a
/// relative stationarity residual of about 1.
#[test]
fn interior_constrain_box_on_an_unpenalised_slope_is_the_unconstrained_fit() {
    init_parallelism();
    let (x, y) = fixture();
    let data = dataset(&x, &y);
    let free = bounded_slope("y ~ linear(x, double_penalty=false)", &data);
    let boxed = bounded_slope("y ~ linear(x, min=-1, max=1, double_penalty=false)", &data);
    let ols = ols_slope(&x, &y);
    // Every slope here solves the same two-column normal equations, whose
    // right-hand side accumulates `n = 200` products of magnitude about 1: a
    // rounding difference of order `n·ε ≈ 4e-14` there, divided by
    // `Σ(x − x̄)² ≈ 67`, moves the slope by about 1e-15. The 1e-12 band sits
    // three orders above that and ten orders below the slope itself.
    // Non-vacuity: the unpenalised slope is the fixture's noise slope, well
    // inside the box and well away from zero.
    assert!(
        (free - ols).abs() <= 1e-12 && free.abs() > 0.02 && free.abs() < 1.0,
        "the unpenalised linear slope must be the least-squares slope {ols}, got {free}"
    );
    assert!(
        (boxed - free).abs() <= 1e-12,
        "an interior box on an unpenalised slope must not move it: boxed {boxed} against \
         linear {free}"
    );
}
