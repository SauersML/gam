//! End-to-end regression for #1762 at the real smooth-spline pipeline layer.
//!
//! A binomial-logit GAM fit to a near-perfectly-separated 1-D surface
//! (η = 12·x on x ∈ U(-1, 1)) drove the ARC outer-optimizer into a
//! FLAT-VALLEY STALL: the inner P-IRLS "could not certify a valid minimum"
//! because the working weights w = μ̂(1−μ̂) collapse to ~0 on the saturated
//! majority of points, corrupting the outer REML curvature so every
//! cost-stall escape failed. The reported symptom was ~117 s at n=3200 and a
//! NON-CONVERGED status where mgcv finishes in ~0.08 s and converges.
//!
//! The top-level `gam` crate cannot build in this environment (a `build.rs`
//! author tripwire), so the issue's `fit_from_formula` path is exercised here
//! in `gam-models`, which builds standalone.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

/// Build a near-separated binomial dataset: η = slope·x, x ∈ U(-1, 1), and
/// y ~ Bernoulli(logistic(η)). With slope=12 roughly ~6% of points fall on the
/// "wrong" side of x=0, so the classes are almost linearly separable and the
/// PIRLS weights collapse over the saturated majority.
fn near_separated_binomial(n: usize, slope: f64, seed: u64) -> gam_data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-1.0_f64, 1.0).unwrap();
    let ubern = Uniform::new(0.0_f64, 1.0).unwrap();
    let headers: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = unif.sample(&mut rng);
        let p = 1.0 / (1.0 + (-slope * x).exp());
        let y = if ubern.sample(&mut rng) < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![x.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// One near-separation fit + all three #1762 convergence assertions
/// (converged, projected gradient below ceiling, sane wall-clock). Shared by
/// the moderate (slope=12) and stronger (slope=16) separation regimes so both
/// exercise the identical contract.
fn assert_near_separation_converges(n: usize, slope: f64, seed: u64) {
    let ds = near_separated_binomial(n, slope, seed);

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };

    let t0 = std::time::Instant::now();
    let result = fit_from_formula("y ~ smooth(x)", &ds, &cfg)
        .expect("near-separated binomial fit must not error");
    let elapsed = t0.elapsed();

    let StandardFitResult { fit, .. } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit"),
    };

    let edf = fit.edf_total().unwrap_or(f64::NAN);
    let gnorm = fit.outer_gradient_norm.unwrap_or(f64::NAN);
    eprintln!(
        "#1762 convergence: n={n} slope={slope} elapsed={:.2}s edf={edf:.2} \
         converged={} |g|={gnorm:.3e}",
        elapsed.as_secs_f64(),
        fit.outer_converged
    );

    assert!(
        fit.outer_converged,
        "near-separated binomial-logit fit (n={n}, slope={slope}) reported \
         NON-CONVERGED (FLAT-VALLEY STALL); the outer ARC optimizer must reach a \
         certified minimum under PIRLS weight collapse (#1762)"
    );
    // The issue's headline symptom is a projected outer gradient stranded at
    // |g|≈54, far above the ~5.0 stationarity ceiling. A genuinely converged
    // fit clears the score-relative bound; assert the authoritative shipped-θ̂
    // gradient is finite and modest so the exact reported stall (|g|=54 while
    // still flagged converged, or a huge residual gradient) cannot regress back
    // in undetected. 5.0 is the issue's own ceiling — NOT a weakened tolerance.
    assert!(
        gnorm.is_finite() && gnorm < 5.0,
        "near-separated binomial-logit fit (n={n}, slope={slope}) shipped with \
         projected outer gradient |g|={gnorm:.3e} ≥ 5.0 ceiling — the FLAT-VALLEY \
         STALL signature from #1762 (reported |g|≈54)"
    );
    // Generous wall-clock ceiling: the issue reports ~117s at n=3200; a healthy
    // fit at n=800 is well under a second. 20s is a loose regression guard that
    // still catches the stall-loop pathology without being timing-flaky in CI.
    assert!(
        elapsed.as_secs_f64() < 20.0,
        "near-separated binomial-logit fit (n={n}, slope={slope}) took {:.1}s \
         (should be ≪1s); the ARC cost-stall escape loop is spinning (#1762)",
        elapsed.as_secs_f64()
    );
}

/// The convergence contract #1762 pins: a near-separated binomial-logit GAM
/// must converge (not FLAT-VALLEY STALL) and in a sane wall-clock budget, with
/// a stationary (small projected-gradient) shipped fit.
#[test]
fn binomial_near_separation_converges_1762() {
    assert_near_separation_converges(800, 12.0, 7);
}

/// Stronger separation (slope=16 ⇒ ~4% wrong-side points, μ̂∈{~2e-7, ~1−2e-7})
/// drives the PIRLS Fisher weights w=μ̂(1−μ̂) an order of magnitude closer to
/// zero than slope=12, so this bites the ill-conditioned-curvature stall harder
/// while staying fast at n=800.
#[test]
fn binomial_stronger_separation_converges_1762() {
    assert_near_separation_converges(800, 16.0, 7);
}
