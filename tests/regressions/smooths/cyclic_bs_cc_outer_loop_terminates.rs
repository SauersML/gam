//! Regression for #874: `fit()` must TERMINATE on a cyclic spline
//! (`s(x, bs='cc')`) whose period is left to default to the data range, even
//! with a small `outer_max_iter`.
//!
//! ## The bug
//!
//! `gamfit.fit(df, "y ~ s(hue, bs='cc')", config={"outer_max_iter": 12})` never
//! returned. The outer REML optimizer reached `|gradient| ~ 1.8e-11` (clearly
//! converged) but kept re-evaluating at the same point forever; `outer_max_iter`
//! was not honored as a hard backstop. A non-cyclic `s(hue)` / `bs='cr'` on the
//! same data returned fine — only the cyclic (periodic) basis spun.
//!
//! ## What is special about this repro vs the passing cyclic tests
//!
//! The existing cyclic coverage (`quality_vs_mgcv_cyclic_cubic.rs`,
//! `periodic_formula_integration.rs`) always fits cyclic smooths with an
//! EXPLICIT period (`period_start=…, period_end=…`) and the production default
//! outer-iteration budget. This repro exercises the previously-untested
//! combination that hangs:
//!
//!   1. `s(x, bs='cc')` with **no** period option, so the period defaults to the
//!      data range `[min(x), max(x)]` (mgcv `bs="cc"` semantics). The data point
//!      at `x = max` then wraps onto `x = min`.
//!   2. an explicit small `outer_max_iter`, which must be an unconditional
//!      termination backstop regardless of basis.
//!
//! ## What this test asserts
//!
//! - `fit_from_formula` RETURNS (the test process completing is the primary
//!   assertion — if the outer loop spins, CI times the test out and the
//!   backtrace localizes the wedge).
//! - The fitted curve is genuinely periodic: `fitted(min) == fitted(max)` to a
//!   tight tolerance — the defining guarantee of a cyclic basis, asserted on the
//!   converged fit so a "terminate but produce garbage" regression is caught
//!   too.
//! - A non-cyclic `s(x)` on the SAME data also returns (control arm: confirms
//!   the data itself is well-posed and isolates the defect to the cyclic path).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Build a Gaussian dataset whose response is a smooth periodic function of a
/// `hue`-like angular covariate spanning (close to) a full period, returned as
/// an encoded dataset plus the raw `hue` grid for the wrap check.
fn periodic_dataset(n: usize, seed: u64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let period = 2.0 * PI;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).expect("normal");
    // hue spans [0, 2π) on a uniform grid. The natural data range becomes the
    // default cyclic period, which is exactly the regime #874 hangs on.
    let hue: Vec<f64> = (0..n).map(|i| period * i as f64 / n as f64).collect();
    let y: Vec<f64> = hue
        .iter()
        .map(|&h| (h.sin() + 0.5 * (2.0 * h).cos()) + 0.1 * noise.sample(&mut rng))
        .collect();

    let headers = vec!["hue".to_string(), "y".to_string()];
    let rows = hue
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode periodic dataset");
    (ds, hue)
}

#[test]
fn cyclic_bs_cc_default_period_fit_terminates_and_is_periodic() {
    init_parallelism();

    let n = 120usize;
    let (ds, hue) = periodic_dataset(n, 874);
    let col = ds.column_map();
    let hue_idx = col["hue"];
    let (hue_min, hue_max) = hue
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });

    // Exactly the failing repro: cyclic basis via the mgcv `bs='cc'` idiom, NO
    // explicit period (defaults to the data range), and a small outer-iteration
    // budget that must guarantee termination.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        outer_max_iter: Some(12),
        ..FitConfig::default()
    };

    // If #874 is present this call never returns and the test times out; the
    // assertion is that it returns at all within the harness timeout.
    let result =
        fit_from_formula("y ~ s(hue, bs='cc')", &ds, &cfg).expect("cyclic bs='cc' fit must return");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian cyclic smooth");
    };

    // The fit must recognize the converged cyclic optimum and report a sane,
    // finite total EDF — not run away.
    let edf = fit.fit.edf_total().expect("gam reports total edf");
    assert!(
        edf.is_finite() && edf > 0.0 && edf < n as f64,
        "cyclic fit EDF must be finite and in (0, n); got {edf}"
    );

    // Periodic-wrap guarantee on the converged fit: fitted(min) == fitted(max).
    // This is the defining property of a cyclic basis and catches a
    // "terminate-but-wrong" regression in addition to the hang itself.
    let probe = [hue_min, hue_max];
    let mut design_pts = Array2::<f64>::zeros((probe.len(), ds.headers.len()));
    for (i, &x) in probe.iter().enumerate() {
        design_pts[[i, hue_idx]] = x;
    }
    let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
        .expect("rebuild design at wrap endpoints");
    let fitted = design.design.apply(&fit.fit.beta).to_vec();
    let wrap_gap = (fitted[0] - fitted[1]).abs();
    assert!(
        wrap_gap < 1e-6,
        "cyclic fit must wrap: |fitted(min) - fitted(max)| = {wrap_gap:.3e} (>= 1e-6)"
    );
}

#[test]
fn noncyclic_control_on_same_data_terminates() {
    // Control arm: the SAME data fit with an ordinary (non-cyclic) smooth must
    // also return. This isolates #874 to the cyclic path — the data is
    // well-posed and the small outer budget is honored by the non-cyclic route.
    init_parallelism();

    let n = 120usize;
    let (ds, _hue) = periodic_dataset(n, 874);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        outer_max_iter: Some(12),
        ..FitConfig::default()
    };

    let result =
        fit_from_formula("y ~ s(hue)", &ds, &cfg).expect("non-cyclic control fit must return");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the non-cyclic control");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");
    assert!(
        edf.is_finite() && edf > 0.0 && edf < n as f64,
        "non-cyclic control EDF must be finite and in (0, n); got {edf}"
    );
}
