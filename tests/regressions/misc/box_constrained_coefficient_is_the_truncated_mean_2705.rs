//! #2705 group C residual — what a binding `linear(x, min, max)` actually
//! reports, certified against its closed form.
//!
//! The thread reads the surviving group-C failure as *"the box does not bind at
//! its bound"*: `y ~ linear(x, min=0, max=1)` on the noise-free line `y = 2+5x`
//! reports a slope of `0.902139` where `bounded(x, min=0, max=1)` reports
//! `1.000000`, and three tests assert the reported coefficient must sit at the
//! bound. That reading attributes the gap to the active-set solver stopping
//! short. It does not.
//!
//! The box binds exactly. The fit's own `deviance` is `229.6`, which is
//! `(5−1)²·XᵀX` to the last digit — the residual sum of squares AT `β = 1`, so
//! the MODE is on the bound. What is reported is a different estimand: the MEAN
//! of the truncated Laplace posterior, which SPEC rule 3 makes the default
//! ("posterior mean must always be the default, never MAP") and which
//! `constrained_posterior`'s module documentation states outright — "a fit
//! carrying inequality constraints reports the mean of its truncated Laplace
//! posterior, never the boundary MAP", and "an INEQUALITY does not delete a
//! direction, it halves one: the posterior along the constraint normal is
//! supported on a half-line, its mode sits at the endpoint, and its variance
//! does not".
//!
//! For a single box on a single coefficient that mean has a closed form, and
//! this test pins the reported number to it:
//!
//! ```text
//!   β̂_unc = 5,   σ² = φ̂ / XᵀX,   a = (min − β̂_unc)/σ,  b = (max − β̂_unc)/σ
//!   E[β | β ∈ [min, max]]  =  β̂_unc + σ·(φ(a) − φ(b)) / (Φ(b) − Φ(a))
//! ```
//!
//! Measured agreement across four bounds is 8 significant figures, and the
//! reported VARIANCE matches the truncated-normal variance to the cubature's own
//! `1e-3` relative tolerance. So the number is not a solver shortfall with a
//! plausible size; it is a closed form evaluated correctly.
//!
//! What that leaves open is a genuine question about the ESTIMAND rather than
//! about the solver: `bounded()` reports `1.000000` on the same data because its
//! latent interval transform `β = min + width·σ(θ)` stretches the boundary to
//! `θ = ±∞`, so ITS posterior concentrates at the bound. The two documented ways
//! to box a coefficient impose different priors and therefore publish different
//! numbers. Deciding which one a user asking for a box should get is a scope
//! call for the issue, not something to settle by moving either number — so this
//! test certifies the identity that is provable and says so.

use gam::data::EncodedDataset;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use gam_math::probability::{normal_cdf, normal_logsf};
use std::io::Write;

const INTERCEPT: f64 = 2.0;
const SLOPE: f64 = 5.0;
const N: usize = 41;

/// Noise-free `y = 2 + 5x` on an even grid over `[-1, 1]` — the fixture
/// `misc::linear_box_constraint_violated_by_internal_scaling` uses.
fn fixture() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..N)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((N - 1) as f64))
        .collect();
    let y: Vec<f64> = x.iter().map(|&xi| INTERCEPT + SLOPE * xi).collect();
    (x, y)
}

fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// `E[X | X ∈ [lower, upper]]` for `X ~ N(mean, sd²)`.
///
/// The half-line case is evaluated in LOG SPACE, because it is the one that
/// cannot be done any other way. `Φ̄(6.246) = 2.1e-10`, so forming it as
/// `1 − Φ(6.246)` loses everything below the sixth significant figure — and the
/// first version of this test failed by exactly that much (`1.06e-5` relative)
/// with the ENGINE on the accurate side. `constrained_posterior` reaches for
/// `normal_logsf` / `signed_probit_logcdf_and_mills_ratio` for this reason; a
/// reference that does not is measuring its own cancellation.
fn truncated_normal_mean(mean: f64, sd: f64, lower: f64, upper: f64) -> f64 {
    let a = (lower - mean) / sd;
    if upper.is_infinite() && upper > 0.0 {
        // `φ(a)/Φ̄(a) = exp(ln φ(a) − ln Φ̄(a))`, with
        // `ln φ(a) = −a²/2 − ½·ln(2π)`.
        let log_pdf = -0.5 * a * a - 0.5 * (2.0 * std::f64::consts::PI).ln();
        return mean + sd * (log_pdf - normal_logsf(a)).exp();
    }
    let b = (upper - mean) / sd;
    let mass = normal_cdf(b) - normal_cdf(a);
    assert!(
        mass > 0.0,
        "the truncation interval must carry positive Gaussian mass: a={a}, b={b}"
    );
    mean + sd * (standard_normal_pdf(a) - standard_normal_pdf(b)) / mass
}

fn dataset(x: &[f64], y: &[f64]) -> EncodedDataset {
    let mut csv = String::from("x,y\n");
    for i in 0..x.len() {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    // Both tests in this file build a fixture in the same process, and each
    // encodes a different slope sign, so the path must be unique per CALL, not
    // per process: a shared name let one test read the other's csv (and the
    // second remover find nothing), which flipped the recovered slope's sign.
    static FIXTURE_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let serial = FIXTURE_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("gam_2705_boxmean_{}_{serial}.csv", std::process::id()));
    {
        let mut file = std::fs::File::create(&tmp).expect("create fixture csv");
        file.write_all(csv.as_bytes()).expect("write fixture csv");
    }
    let loaded = load_csvwith_inferred_schema(&tmp).expect("load fixture");
    std::fs::remove_file(&tmp).expect("remove the fixture csv this test created");
    loaded
}

#[test]
fn a_binding_coefficient_box_reports_the_truncated_posterior_mean_2705() {
    init_parallelism();
    let (x, y) = fixture();
    let data = dataset(&x, &y);
    let cross_product: f64 = x.iter().map(|xi| xi * xi).sum();

    // The unconstrained slope is the centre of the Gaussian that gets
    // truncated. Read it from a fit rather than from `SLOPE`, so the identity is
    // pinned against what the engine computed and not against the generator.
    let unconstrained = match fit_from_formula("y ~ x", &data, &FitConfig::default()) {
        Ok(FitResult::Standard(fit)) => fit,
        other => panic!("the unconstrained control must fit: {other:?}", other = other.is_ok()),
    };
    let centre = unconstrained.fit.beta[1];
    assert!(
        (centre - SLOPE).abs() < 1e-6,
        "the unconstrained control must recover the noise-free slope, got {centre}"
    );

    for upper in [1.0_f64, 2.0, 3.0, 4.0] {
        let formula = format!("y ~ linear(x, min=0, max={upper})");
        let fitted = match fit_from_formula(&formula, &data, &FitConfig::default()) {
            Ok(FitResult::Standard(fit)) => fit,
            Ok(_) => panic!("`{formula}` is a Standard GAM fit"),
            Err(error) => panic!("`{formula}` must fit: {error}"),
        };
        let reported = fitted.fit.beta[1];

        // 1. The box is honoured on the reported scale. This is the half of
        //    #791 that is genuinely a contract and that genuinely holds.
        assert!(
            (0.0..=upper).contains(&reported),
            "`{formula}` reported {reported:.9} outside the box it was given"
        );

        // 2. The MODE is on the bound: the fit's own deviance is the residual
        //    sum of squares at `β = upper`, which is what "the box binds" means
        //    and what the `did not bind` reading denies.
        let deviance_at_bound = (SLOPE - upper).powi(2) * cross_product;
        let deviance = fitted.fit.deviance;
        assert!(
            (deviance - deviance_at_bound).abs() <= 1e-9 * deviance_at_bound.max(1.0),
            "`{formula}` must be fitted AT the bound: deviance {deviance:.9} against the \
             residual sum of squares at β={upper}, {deviance_at_bound:.9}"
        );

        // 3. What is REPORTED is the mean of the truncated posterior, and it
        //    matches the closed form. `σ² = φ̂/XᵀX` with `φ̂` the profiled
        //    Gaussian scale the fit published.
        let phi = fitted.fit.standard_deviation.powi(2);
        let sd = (phi / cross_product).sqrt();
        let expected = truncated_normal_mean(centre, sd, 0.0, upper);
        assert!(
            (reported - expected).abs() <= 1e-6 * expected.abs().max(1e-3),
            "`{formula}` reported {reported:.12} but the truncated-posterior mean of \
             N({centre:.9}, {sd:.9}²) on [0, {upper}] is {expected:.12}; if these have \
             diverged, the reported coefficient is no longer the estimand SPEC rule 3 \
             mandates"
        );

        // 4. Non-vacuity: the mean must be strictly INSIDE the bound, or this
        //    test would pass for a fit that simply reported the mode.
        assert!(
            reported < upper - 1e-6,
            "`{formula}` reported {reported:.12}, which is the bound itself — the \
             truncated mean of a half-line-truncated Gaussian is strictly interior at \
             every finite multiplier, so this fixture has stopped exercising the \
             estimand it exists for"
        );
    }
}

/// The same identity on the ONE-SIDED half-line, which is the shape
/// `optimization::nonnegative_constraint_kkt_*` reports as "should bind at 0,
/// got 0.007857". A half-line is the case `constrained_posterior`'s doc
/// discusses explicitly — the mode sits at the endpoint and the mean does not —
/// so it is worth pinning separately from the interval.
#[test]
fn a_binding_half_line_reports_the_truncated_posterior_mean_2705() {
    init_parallelism();
    // Slope −5, so `β ≥ 0` binds and the unconstrained optimum is far outside.
    let x: Vec<f64> = (0..N)
        .map(|i| -1.0 + 2.0 * (i as f64) / ((N - 1) as f64))
        .collect();
    let y: Vec<f64> = x.iter().map(|&xi| INTERCEPT - SLOPE * xi).collect();
    let data = dataset(&x, &y);
    let cross_product: f64 = x.iter().map(|xi| xi * xi).sum();

    let unconstrained = match fit_from_formula("y ~ x", &data, &FitConfig::default()) {
        Ok(FitResult::Standard(fit)) => fit,
        _ => panic!("the unconstrained control must fit"),
    };
    let centre = unconstrained.fit.beta[1];
    assert!(
        (centre + SLOPE).abs() < 1e-6,
        "the control must recover the negative noise-free slope, got {centre}"
    );

    let fitted = match fit_from_formula("y ~ nonnegative(x)", &data, &FitConfig::default()) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("`y ~ nonnegative(x)` is a Standard GAM fit"),
        Err(error) => panic!("`y ~ nonnegative(x)` must fit: {error}"),
    };
    let reported = fitted.fit.beta[1];
    assert!(
        reported >= 0.0,
        "`nonnegative(x)` reported {reported:.12}, which is outside the half-line it \
         was given"
    );

    // Mode on the bound: the deviance is the residual sum of squares at β = 0.
    let deviance_at_bound = centre * centre * cross_product;
    let deviance = fitted.fit.deviance;
    assert!(
        (deviance - deviance_at_bound).abs() <= 1e-9 * deviance_at_bound.max(1.0),
        "`nonnegative(x)` must be fitted AT the bound: deviance {deviance:.9} against \
         the residual sum of squares at β=0, {deviance_at_bound:.9}"
    );

    // And what is reported is the half-line truncated mean.
    let phi = fitted.fit.standard_deviation.powi(2);
    let sd = (phi / cross_product).sqrt();
    let expected = truncated_normal_mean(centre, sd, 0.0, f64::INFINITY);
    assert!(
        (reported - expected).abs() <= 1e-6 * expected.abs().max(1e-6),
        "`nonnegative(x)` reported {reported:.12} but the half-line truncated mean of \
         N({centre:.9}, {sd:.9}²) on [0, ∞) is {expected:.12}"
    );
    assert!(
        reported > 0.0,
        "the half-line truncated mean is strictly interior at every finite multiplier, \
         so a reported exact zero means this fixture no longer measures the estimand"
    );
}
