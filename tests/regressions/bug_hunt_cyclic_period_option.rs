//! Regression coverage for the `cyclic()` / `cc()` / `cp()` period-declaration
//! bugs #815 and #816.
//!
//! Both are "documented option silently ignored" failures in the cyclic
//! dispatch arm of `build_single_smooth_basis`:
//!
//!   * #816 — the arm read only `period_start`/`period_end`, so a `period=`
//!     declaration (even a plain numeric one) was silently dropped and the
//!     smooth wrapped at the observed data range instead of the declared
//!     period. It also never validated its option set, so typos / unsupported
//!     options were accepted and discarded.
//!   * #815 — the endpoint aliases were read with the lenient `option_f64`,
//!     which cannot parse the `2*pi` expression spelling the sibling `period=`
//!     option accepts; an unparseable endpoint was silently replaced by the
//!     data range rather than honoured or rejected.
//!
//! The discriminating setup: train `y ≈ sin(theta)` on `theta ∈ [0, 5]` — LESS
//! than one full period (2π ≈ 6.283) — so the declared period genuinely changes
//! the model. The θ grid starts exactly at 0 so that the `period=` form (whose
//! domain origin defaults to the data minimum) and the `period_start=0` form
//! describe the *same* domain `[0, 2π)` and must produce bitwise-identical fits.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const TAU: f64 = std::f64::consts::TAU; // 2π

/// `theta` on a deterministic grid over `[0, 5]` (minv == 0 exactly), with
/// `y = sin(theta)` plus a little seeded noise so the inner solve is
/// well-conditioned. Columns: `theta` (col 0), `y` (col 1).
fn partial_period_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(20250607);
    let noise = Normal::new(0.0, 0.02).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let theta = 5.0 * (i as f64) / ((n - 1) as f64);
            let y = theta.sin() + noise.sample(&mut rng);
            StringRecord::from(vec![theta.to_string(), y.to_string()])
        })
        .collect();
    let headers = ["theta", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `formula` on the partial-period dataset and evaluate the fitted smooth
/// at `probes` (values of `theta`).
fn fit_predict(formula: &str, probes: &[f64]) -> Vec<f64> {
    init_parallelism();
    let data = partial_period_dataset(120);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit for `{formula}`");
    };
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &x) in probes.iter().enumerate() {
        m[[i, 0]] = x;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("rebuild design for `{formula}`: {e}"));
    design.design.apply(&fit.fit.beta).to_vec()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// A dense grid over the full declared period plus the wrap point, used both
/// for whole-curve equivalence checks and for the seam-continuity assertion.
fn probe_grid() -> Vec<f64> {
    let mut g: Vec<f64> = (0..=60).map(|i| TAU * (i as f64) / 60.0).collect();
    // An explicit wrap pair strictly inside the period: f(0.3) must equal
    // f(0.3 + 2π) for a smooth whose true period is 2π.
    g.push(0.3);
    g.push(0.3 + TAU);
    g
}

/// #816: a plain numeric `period=` must set the cyclic period, yielding the same
/// model as the equivalent `period_start`/`period_end` declaration — and a model
/// distinctly different from the (buggy) data-range fallback.
#[test]
fn cyclic_numeric_period_option_matches_endpoint_form() {
    let grid = probe_grid();
    let period_form = fit_predict("y ~ cyclic(theta, k=10, period=6.283185307179586)", &grid);
    let endpoint_form = fit_predict(
        "y ~ cyclic(theta, k=10, period_start=0, period_end=6.283185307179586)",
        &grid,
    );
    // Same domain [0, 2π) declared two ways → bitwise-identical fits.
    let same = max_abs_diff(&period_form, &endpoint_form);
    assert!(
        same < 1e-9,
        "period= and period_start/period_end disagree (max abs diff {same:.3e}); \
         period= is being dropped"
    );

    // And NOT the old data-range fallback (period = maxv - minv = 5).
    let data_range_form = fit_predict(
        "y ~ cyclic(theta, k=10, period_start=0, period_end=5.0)",
        &grid,
    );
    let differs = max_abs_diff(&period_form, &data_range_form);
    assert!(
        differs > 0.1,
        "period=2π fit is indistinguishable from the data-range (period=5) fit \
         (max abs diff {differs:.3e}) — period= silently ignored"
    );
}

/// #816 + #815: the `period=2*pi` expression spelling must parse and produce the
/// same model as the numeric literal.
#[test]
fn cyclic_period_expression_matches_numeric() {
    let grid = probe_grid();
    let expr_form = fit_predict("y ~ cyclic(theta, k=10, period=2*pi)", &grid);
    let numeric_form = fit_predict("y ~ cyclic(theta, k=10, period=6.283185307179586)", &grid);
    let diff = max_abs_diff(&expr_form, &numeric_form);
    assert!(
        diff < 1e-12,
        "cyclic period=2*pi differs from numeric period (max abs diff {diff:.3e})"
    );
}

/// #815: the `period_end=2*pi` (and `period_start`) expression spelling must
/// parse via the same numeric-expression grammar `period=` uses, matching the
/// numeric-literal endpoint form rather than silently collapsing to the data
/// range.
#[test]
fn cyclic_period_end_expression_matches_numeric() {
    let grid = probe_grid();
    let expr_form = fit_predict(
        "y ~ cyclic(theta, k=10, period_start=0, period_end=2*pi)",
        &grid,
    );
    let numeric_form = fit_predict(
        "y ~ cyclic(theta, k=10, period_start=0, period_end=6.283185307179586)",
        &grid,
    );
    let diff = max_abs_diff(&expr_form, &numeric_form);
    assert!(
        diff < 1e-12,
        "cyclic period_end=2*pi differs from the numeric literal (max abs diff {diff:.3e}) — \
         the expression was dropped to the data range"
    );
}

/// The fitted cyclic smooth must genuinely wrap at the *declared* 2π period:
/// f(x) == f(x + 2π) at the seam, for every spelling. (Distinguishes a true
/// period-2π model from the data-range fallback, where the seam pair would
/// straddle two different phases.)
#[test]
fn cyclic_declared_period_wraps_at_seam() {
    for formula in [
        "y ~ cyclic(theta, k=10, period=2*pi)",
        "y ~ cyclic(theta, k=10, period=6.283185307179586)",
        "y ~ cyclic(theta, k=10, period_start=0, period_end=2*pi)",
        "y ~ cyclic(theta, k=10, period_start=0, period_end=6.283185307179586)",
    ] {
        let pred = fit_predict(formula, &[0.3, 0.3 + TAU]);
        let gap = (pred[0] - pred[1]).abs();
        assert!(
            gap < 1e-6,
            "`{formula}` does not wrap at 2π: f(0.3)={:.6}, f(0.3+2π)={:.6}, gap={gap:.3e}",
            pred[0],
            pred[1],
        );
    }
}

/// The `cc` / `cp` aliases share the dispatch arm and must honour `period=` too.
#[test]
fn cc_and_cp_aliases_honour_period_option() {
    let grid = probe_grid();
    let reference = fit_predict(
        "y ~ cyclic(theta, k=10, period_start=0, period_end=2*pi)",
        &grid,
    );
    for alias in ["cc", "cp"] {
        let form = fit_predict(&format!("y ~ {alias}(theta, k=10, period=2*pi)"), &grid);
        let diff = max_abs_diff(&form, &reference);
        assert!(
            diff < 1e-9,
            "`{alias}(theta, period=2*pi)` disagrees with the cyclic endpoint reference \
             (max abs diff {diff:.3e})"
        );
    }
}

/// #815: an unparseable endpoint must be a hard error, never a silent fallback
/// to the data range.
#[test]
fn cyclic_unparseable_endpoint_is_hard_error() {
    init_parallelism();
    let data = partial_period_dataset(60);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = match fit_from_formula(
        "y ~ cyclic(theta, period_start=0, period_end=banana)",
        &data,
        &cfg,
    ) {
        Ok(_) => panic!("an unparseable period_end must be rejected, not silently dropped"),
        Err(e) => e,
    };
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("period_end") || msg.contains("numeric"),
        "error should name the offending endpoint/value; got: {err}"
    );
}

/// #816: the cyclic arm must validate its option set (it previously did not), so
/// a typo'd / unsupported option is rejected instead of silently discarded.
#[test]
fn cyclic_rejects_unknown_option() {
    init_parallelism();
    let data = partial_period_dataset(60);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = match fit_from_formula("y ~ cyclic(theta, perdiod=2*pi)", &data, &cfg) {
        Ok(_) => panic!("a misspelled option must be rejected by the cyclic arm"),
        Err(e) => e,
    };
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("does not accept option") || msg.contains("perdiod"),
        "error should flag the unknown option; got: {err}"
    );
}
