//! Regression coverage for issue #978: a global smooth + a factor smooth on the
//! *same* covariate (`s(x) + s(g, x, bs=sz)` / `s(x) + fs(x, g)`) fits, but
//! cannot be predicted — not even on its own training rows.
//!
//! Root cause: at fit time the broader factor smooth shares column space with
//! `s(x)`, so the hierarchical-ownership pass residualizes it against the owner
//! (an owner-orthogonalization transform `Z` that *removes* columns). For an
//! ordinary B-spline that `Z` is folded into the term's frozen identifiability
//! transform and replayed at predict. But a block-replicated factor smooth
//! (`FactorSumToZero` / `FactorSmooth`) carries PER-MARGINAL metadata into which
//! the FULL-design `Z` cannot be folded, so `Z` was simply dropped. At predict
//! the term's inner basis is now frozen, which sets `skip_global_transform`, so
//! the global pass that would have recomputed `Z` is also skipped. Result: the
//! rebuilt prediction design keeps the columns the fit removed, and
//! `beta.len() < design.ncols()` aborts every prediction.
//!
//! The fix persists the realized `Z` for these factor smooths and replays it at
//! predict, so the rebuilt design matches the fitted coefficient count exactly.
//!
//! This is the Rust-core mirror of the committed Python repro
//! `tests/bug_hunt_overlapping_smooth_predict_design_mismatch_test.py`.

use gam::inference::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism};
use ndarray::Array2;

const N: usize = 600;
const LEVELS: [&str; 3] = ["A", "B", "C"];

/// Deterministic dataset: a global linear-ish trend in `x` plus a per-level
/// sinusoidal deviation, so the factor smooth carries real signal (a degenerate
/// "fix" that drops the factor smooth would fail the correlation check below).
fn dataset() -> (EncodedDataset, Array2<f64>) {
    let mut values = Array2::<f64>::zeros((N, 3)); // columns: y, x, g
    for i in 0..N {
        // Structured, rand-free jittered grid on [0, 1].
        let base = (i as f64 + 0.5) / N as f64;
        let jitter = ((i as f64 * 12.9898).sin() * 43758.5453).fract() * 1e-3;
        let x = (base + jitter).clamp(1e-6, 1.0 - 1e-6);
        let level = i % 3;
        let bump = match level {
            0 => 0.5 * (2.0 * std::f64::consts::PI * x).sin(),
            1 => -0.5 * (2.0 * std::f64::consts::PI * x).sin(),
            _ => 0.0,
        };
        // Small deterministic noise so the fit is not interpolating.
        let noise = ((i as f64 * 7.0).cos()) * 0.05;
        let y = 2.0 * x + bump + noise;
        values[[i, 0]] = y;
        values[[i, 1]] = x;
        values[[i, 2]] = level as f64;
    }

    let schema = DataSchema {
        columns: vec![
            SchemaColumn {
                name: "y".into(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
            SchemaColumn {
                name: "x".into(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
            SchemaColumn {
                name: "g".into(),
                kind: ColumnKindTag::Categorical,
                levels: LEVELS.iter().map(|s| (*s).to_string()).collect(),
            },
        ],
    };
    let ds = EncodedDataset {
        headers: vec!["y".into(), "x".into(), "g".into()],
        values: values.clone(),
        schema,
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    };
    (ds, values)
}

fn run_one(formula: &str) {
    init_parallelism();
    let (ds, values) = dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, &ds, &cfg).unwrap_or_else(|e| panic!("fit `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit for `{formula}`")
    };

    // ── Predict on the model's own training rows ───────────────────────
    // The frozen `resolvedspec` is exactly what the predict path rebuilds the
    // design from. Its column count MUST equal the fitted coefficient count;
    // before the fix it exceeded it (the residualized columns reappeared).
    let design_input = values.clone();
    let design = build_term_collection_design(design_input.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("predict design rebuild failed for `{formula}`: {e:?}"));
    assert_eq!(
        design.design.ncols(),
        fit.fit.beta.len(),
        "predict design column count ({}) must equal fitted beta length ({}) for `{formula}`",
        design.design.ncols(),
        fit.fit.beta.len(),
    );

    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(
        pred.len(),
        N,
        "prediction row count mismatch for `{formula}`"
    );
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "predictions must be finite for `{formula}`",
    );

    // ── The fit must be useful, not a degenerate constant ──────────────
    let y: Vec<f64> = (0..N).map(|i| values[[i, 0]]).collect();
    let corr = pearson(&pred, &y);
    assert!(
        corr > 0.5,
        "in-sample predictions barely track y (corr={corr:.3}) for `{formula}`",
    );

    // ── Predict on a fresh grid of brand-new rows across every level ───
    let m = 30usize;
    let mut grid = Array2::<f64>::zeros((m, 3));
    for i in 0..m {
        grid[[i, 0]] = 0.0; // y is ignored at predict
        grid[[i, 1]] = 0.02 + 0.96 * (i as f64) / ((m - 1) as f64);
        grid[[i, 2]] = (i % 3) as f64;
    }
    let grid_design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("predict grid rebuild failed for `{formula}`: {e:?}"));
    assert_eq!(
        grid_design.design.ncols(),
        fit.fit.beta.len(),
        "predict grid column count must equal fitted beta length for `{formula}`",
    );
    let grid_pred = grid_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(grid_pred.len(), m);
    assert!(grid_pred.iter().all(|v| v.is_finite()));
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-300)
}

#[test]
fn sum_to_zero_factor_smooth_over_global_smooth_predicts() {
    run_one("y ~ s(x) + s(g, x, bs=sz)");
}

#[test]
fn factor_smooth_random_effect_over_global_smooth_predicts() {
    run_one("y ~ s(x) + fs(x, g)");
}
