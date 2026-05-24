//! Unit tests for `tests/common::{fit_power_law, report_power_law}`.
//!
//! The shared power-law analyzer is the gatekeeper for every scaling-law
//! probe in the suite (`standard_gam_scaling.rs`,
//! `margslope_inner_pirls_scaling.rs`, and any future probe). It enforces
//! the mission's "MEASURE FIRST" rule by refusing to extrapolate when the
//! fit is poor — which means a regression in this analyzer (e.g. swapping
//! `<` and `<=` on the R² gate, or letting NaN through the residual
//! computation) silently lets bad data drive biobank-budget verdicts.
//! Pin the policy down with these tests.

mod power_law_common;

use power_law_common::{PowerLawFit, fit_power_law, report_power_law_full};

/// Fit a clean `y = 2 · x^1.5` line and verify recovery to ~10 significant
/// figures (log-log OLS on noiseless data is exact up to roundoff).
#[test]
fn fit_recovers_clean_power_law() {
    let xs = [1.0_f64, 2.0, 4.0, 8.0, 16.0, 32.0];
    let alpha_true = 1.5_f64;
    let a_true = 2.0_f64;
    let points: Vec<(f64, f64)> = xs
        .iter()
        .map(|&x| (x, a_true * x.powf(alpha_true)))
        .collect();
    let fit = fit_power_law(&points).expect("clean data should fit");
    assert!(
        (fit.alpha - alpha_true).abs() < 1e-10,
        "alpha drift: got {} expected {}",
        fit.alpha,
        alpha_true
    );
    assert!(
        (fit.a - a_true).abs() < 1e-10,
        "a drift: got {} expected {}",
        fit.a,
        a_true
    );
    assert!(
        fit.r2 > 0.999_999,
        "R² should be ~1 on noiseless data, got {}",
        fit.r2
    );
    assert!(
        fit.max_abs_log_resid < 1e-10,
        "max-resid should be ~0 on noiseless data, got {}",
        fit.max_abs_log_resid
    );
    assert_eq!(fit.n_points, xs.len());
}

/// Fewer than 3 points → `None`. The honest-fit gate refuses to fit a
/// power law to ≤2 points because the residual cannot diagnose
/// goodness-of-fit (a 2-point fit is exact regardless of underlying law).
#[test]
fn fit_rejects_insufficient_data() {
    assert!(fit_power_law(&[]).is_none());
    assert!(fit_power_law(&[(1.0, 2.0)]).is_none());
    assert!(fit_power_law(&[(1.0, 2.0), (4.0, 8.0)]).is_none());
    // 3 points → fit. Boundary check.
    let three = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)];
    assert!(fit_power_law(&three).is_some());
}

/// Degenerate input (all-equal x) → fit must not produce NaN or Inf.
/// The `denom = n*Σx² − (Σx)²` is exactly zero when all x's coincide; the
/// helper guards against this and returns `None` rather than dividing.
#[test]
fn fit_rejects_degenerate_x_collapse() {
    let points = [(2.0, 1.0), (2.0, 4.0), (2.0, 16.0)];
    assert!(
        fit_power_law(&points).is_none(),
        "fit should refuse all-equal-x rather than divide-by-zero"
    );
}

/// Inject a single ~30% outlier into otherwise clean data — the fit
/// should produce a finite α/a but max_abs_log_resid should be large
/// enough that `report_power_law`'s refuse-to-extrapolate gate fires.
#[test]
fn fit_flags_outlier_in_max_log_resid() {
    let mut points: Vec<(f64, f64)> = (1..=10)
        .map(|i| (i as f64, 2.0 * (i as f64).powf(1.5)))
        .collect();
    // Force one point to be 2× the truth — log-distance ln(2) ≈ 0.69.
    points[3].1 *= 2.0;
    let fit = fit_power_law(&points).expect("noisy data still fits");
    assert!(
        fit.max_abs_log_resid > 0.5,
        "outlier should drive max-resid above the 0.5 honest-fit gate, got {}",
        fit.max_abs_log_resid
    );
}

/// Validate the fit's `n_points` field tracks the input length, not
/// some derived quantity.
#[test]
fn fit_reports_input_length() {
    for n in 3..=8 {
        let points: Vec<(f64, f64)> = (1..=n).map(|i| (i as f64, (i as f64).powi(2))).collect();
        let fit = fit_power_law(&points).expect("clean fit");
        assert_eq!(fit.n_points, n, "n_points mismatch for input of length {n}");
    }
}

/// Power-law parameters round-trip through `PowerLawFit` cleanly: a Clone
/// + Debug + Copy struct is the contract callers (downstream test code +
/// any future cross-command analyzer) expect.
#[test]
fn power_law_fit_struct_roundtrips() {
    let original = PowerLawFit {
        alpha: 1.234,
        a: 5.678,
        r2: 0.91,
        max_abs_log_resid: 0.12,
        n_points: 7,
    };
    let copy = original; // requires Copy
    let cloned = original.clone(); // requires Clone
    assert_eq!(copy.alpha, original.alpha);
    assert_eq!(cloned.alpha, original.alpha);
    let dbg = format!("{:?}", original); // requires Debug
    assert!(dbg.contains("PowerLawFit"));
}

/// Verdict reflects budget comparison correctly across the boundary.
/// Build clean `y = 0.001 · n^1.0` (one second per thousand). At
/// budget_y = 60 s, n = 30_000 should FIT (pred = 30 s) and
/// n = 90_000 should be OVER BUDGET (pred = 90 s).
#[test]
fn report_extrapolation_verdicts_track_budget_boundary() {
    let xs = [1_000.0_f64, 5_000.0, 10_000.0, 50_000.0, 100_000.0];
    let alpha_true = 1.0_f64;
    let a_true = 0.001_f64;
    let points: Vec<(f64, f64)> = xs
        .iter()
        .map(|&x| (x, a_true * x.powf(alpha_true)))
        .collect();
    let extrapolate = [("under", 30_000.0), ("over", 90_000.0)];
    let report = report_power_law_full("[BOUNDARY]", &points, &extrapolate, 60.0)
        .expect("clean fit should produce a report");
    // With the structured-extrapolations contract removed, validate the
    // fit recovered the linear law that drives the side-effect verdicts:
    // a · x^α with α=1, a=0.001 implies pred(30k)=30 < 60 (FITS) and
    // pred(90k)=90 > 60 (OVER BUDGET). The verdicts themselves are now
    // stderr-only.
    let fit = report.fit;
    assert!(
        (fit.alpha - alpha_true).abs() < 1e-9,
        "alpha drift: got {} expected {}",
        fit.alpha,
        alpha_true
    );
    assert!(
        (fit.a - a_true).abs() < 1e-9,
        "a drift: got {} expected {}",
        fit.a,
        a_true
    );
    let pred_under = fit.a * 30_000.0_f64.powf(fit.alpha);
    let pred_over = fit.a * 90_000.0_f64.powf(fit.alpha);
    assert!(
        (pred_under - 30.0).abs() < 1e-6,
        "pred at n=30k should be ~30s, got {pred_under}"
    );
    assert!(
        (pred_over - 90.0).abs() < 1e-6,
        "pred at n=90k should be ~90s, got {pred_over}"
    );
    assert!(pred_under <= 60.0, "n=30k should FIT (≤60s budget)");
    assert!(pred_over > 60.0, "n=90k should be OVER BUDGET (>60s)");
}

/// Refuse-to-extrapolate gate: when the fit's max-residual exceeds the
/// 0.5-in-log-space honesty threshold, `report_power_law_full` returns
/// `None` and emits "REFUSING EXTRAPOLATION", regardless of how strong
/// the budget would have looked under the noise-driven fit. Critical
/// invariant — the mission's MEASURE FIRST rule depends on this gate.
#[test]
fn report_refuses_extrapolation_when_fit_poor() {
    // Two-cluster pattern that fits a near-zero α (slope) but with
    // huge residuals: the same x range produces wildly different y.
    let points = [
        (1.0, 1.0),
        (2.0, 100.0),
        (3.0, 1.0),
        (4.0, 100.0),
        (5.0, 1.0),
    ];
    let extrapolate = [("biobank", 320_000.0)];
    let report = report_power_law_full("[NOISY]", &points, &extrapolate, 1e9);
    assert!(
        report.is_none(),
        "report must refuse extrapolation when fit's max log-resid > 0.5; got {:?}",
        report
    );
}

/// Random parameters should round-trip through the analyzer to within
/// a few orders of magnitude of floating-point noise. This is a
/// property check on the OLS fit's correctness — bugs in the closed-
/// form `(α, a)` formula would cause systematic drift on noiseless
/// data, which we can detect by sampling many random parameter
/// combinations and asserting recovery on each.
#[test]
fn fit_recovers_random_clean_power_laws() {
    // Deterministic LCG so the test is reproducible without pulling
    // in a full RNG dependency.
    let mut seed = 0xDEAD_BEEFu64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as u32) as f64 / (u32::MAX as f64)
    };
    let mut max_alpha_err = 0.0_f64;
    let mut max_a_err = 0.0_f64;
    for _ in 0..50 {
        // alpha in [-2, 2], a in [1e-3, 1e3].
        let alpha_true = -2.0 + 4.0 * next();
        let a_true = 1e-3 * (1e6_f64).powf(next());
        // 6 x values in [1, 1000].
        let xs: Vec<f64> = (0..6)
            .map(|i| 1.0 + (1000.0 - 1.0) * (i as f64 / 5.0))
            .collect();
        let points: Vec<(f64, f64)> = xs
            .iter()
            .map(|&x| (x, a_true * x.powf(alpha_true)))
            .collect();
        let fit = fit_power_law(&points).unwrap_or_else(|| {
            panic!("clean random fit must succeed: alpha={alpha_true}, a={a_true}")
        });
        max_alpha_err = max_alpha_err.max((fit.alpha - alpha_true).abs());
        max_a_err = max_a_err.max((fit.a - a_true).abs() / a_true.abs());
    }
    assert!(
        max_alpha_err < 1e-9,
        "alpha drift across 50 random clean fits: {max_alpha_err}"
    );
    assert!(
        max_a_err < 1e-9,
        "a relative drift across 50 random clean fits: {max_a_err}"
    );
}
