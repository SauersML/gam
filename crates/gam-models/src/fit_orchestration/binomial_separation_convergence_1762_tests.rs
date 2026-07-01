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

/// Minimal stderr logger so `RUST_LOG`-gated `log::` traces from the outer
/// ARC optimizer and inner PIRLS surface during diagnostic runs. Installed
/// once; ignores the double-install error on repeated test entry.
struct StderrLogger;
impl log::Log for StderrLogger {
    fn enabled(&self, _m: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        eprintln!("[{}] {}", record.level(), record.args());
    }
    fn flush(&self) {}
}
static DIAG_LOGGER: StderrLogger = StderrLogger;
fn install_diag_logger() {
    if std::env::var("DIAG_1762_LOG").is_ok() {
        let _ = log::set_logger(&DIAG_LOGGER);
        log::set_max_level(log::LevelFilter::Debug);
    }
}

/// Diagnostic: fit the #1762 repro at a moderate n and print convergence,
/// timing, and edf so the failure mode can be read off directly. Run with
/// `--nocapture`. No hard assertion — this is the measurement harness.
#[test]
fn binomial_near_separation_diagnostic_1762() {
    install_diag_logger();
    for &(n, slope) in &[(800usize, 12.0), (1600, 12.0), (3200, 12.0), (3200, 20.0)] {
        let ds = near_separated_binomial(n, slope, 7);

        let cfg = FitConfig {
            family: Some("binomial".to_string()),
            ..FitConfig::default()
        };

        let t0 = std::time::Instant::now();
        let result = fit_from_formula("y ~ smooth(x)", &ds, &cfg);
        let elapsed = t0.elapsed();

        match result {
            Ok(FitResult::Standard(StandardFitResult { fit, .. })) => {
                let edf = fit.edf_total().unwrap_or(f64::NAN);
                eprintln!(
                    "#1762 diag n={n} slope={slope}: OK elapsed={:.2}s edf={edf:.2} converged={}",
                    elapsed.as_secs_f64(),
                    fit.outer_converged
                );
            }
            Ok(_) => {
                eprintln!("#1762 diag n={n} slope={slope}: unexpected non-Standard result variant");
            }
            Err(e) => {
                eprintln!(
                    "#1762 diag n={n} slope={slope}: ERROR after {:.2}s: {e}",
                    elapsed.as_secs_f64()
                );
            }
        }
    }
}

/// Load a 2-column (x,y) CSV produced by the exact numpy repro so we can fit
/// the *identical* dataset the issue reports (isolating data-vs-config).
fn load_xy_csv(path: &str) -> Option<gam_data::EncodedDataset> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let header = lines.next()?;
    let headers: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        rows.push(StringRecord::from(fields));
    }
    encode_recordswith_inferred_schema(headers, rows).ok()
}

/// Data-vs-config isolation: fit the EXACT numpy repro data (dumped to
/// `/tmp/sep_n{800,3200}.csv`) through `fit_from_formula`. If the stall is
/// data-driven this reproduces it; if config-driven it converges (proving the
/// difference is the Python-layer FitConfig, not the sample). Skips silently
/// when the CSV is absent so CI without the fixture stays green.
#[test]
fn binomial_exact_numpy_data_diagnostic_1762() {
    install_diag_logger();
    for path in ["/tmp/sep_n800.csv", "/tmp/sep_n3200.csv"] {
        let Some(ds) = load_xy_csv(path) else {
            eprintln!("#1762 exact-data: {path} absent, skipping");
            continue;
        };
        let cfg = FitConfig {
            family: Some("binomial".to_string()),
            ..FitConfig::default()
        };
        let t0 = std::time::Instant::now();
        let result = fit_from_formula("y ~ smooth(x)", &ds, &cfg);
        let elapsed = t0.elapsed();
        match result {
            Ok(FitResult::Standard(StandardFitResult { fit, .. })) => {
                let edf = fit.edf_total().unwrap_or(f64::NAN);
                eprintln!(
                    "#1762 exact-data {path}: OK elapsed={:.2}s edf={edf:.2} converged={}",
                    elapsed.as_secs_f64(),
                    fit.outer_converged
                );
            }
            Ok(_) => eprintln!("#1762 exact-data {path}: unexpected non-Standard result"),
            Err(e) => eprintln!(
                "#1762 exact-data {path}: ERROR after {:.2}s: {e}",
                elapsed.as_secs_f64()
            ),
        }
    }
}

/// The convergence contract #1762 pins: a near-separated binomial-logit GAM
/// must converge (not FLAT-VALLEY STALL) and in a sane wall-clock budget.
#[test]
fn binomial_near_separation_converges_1762() {
    let n = 800usize;
    let ds = near_separated_binomial(n, 12.0, 7);

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
    eprintln!(
        "#1762 convergence: n={n} elapsed={:.2}s edf={edf:.2} converged={}",
        elapsed.as_secs_f64(),
        fit.outer_converged
    );

    assert!(
        fit.outer_converged,
        "near-separated binomial-logit fit reported NON-CONVERGED (FLAT-VALLEY STALL); \
         the outer ARC optimizer must reach a certified minimum under PIRLS weight \
         collapse (#1762)"
    );
    // Generous wall-clock ceiling: the issue reports ~117s at n=3200; a healthy
    // fit at n=800 is well under a second. 20s is a loose regression guard that
    // still catches the stall-loop pathology without being timing-flaky in CI.
    assert!(
        elapsed.as_secs_f64() < 20.0,
        "near-separated binomial-logit fit took {:.1}s at n={n} (should be ≪1s); \
         the ARC cost-stall escape loop is spinning (#1762)",
        elapsed.as_secs_f64()
    );
}
