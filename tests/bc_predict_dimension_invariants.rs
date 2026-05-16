//! Regression coverage for the "frozen-resolvedspec stores pre-BC transform"
//! shape bug (ticket: see `bc_clamped_predict_shape_bug.rs` and commit
//! 9e3e40ed). The original symptom was
//!
//!     rebuild design failed: DimensionMismatch(
//!       "frozen identifiability transform mismatch: design has N columns but
//!        transform has N+2 rows")
//!
//! whenever a B-spline smooth declared any non-Free boundary condition and we
//! then asked for a prediction design from the frozen resolved spec. The fix
//! folds the boundary projection into the captured `identifiability_transform`
//! and clears `boundary_conditions` on the frozen spec, so replay rebuilds the
//! raw knot basis and applies the composed Zb · Zi exactly once.
//!
//! This file exercises every (bc_left, bc_right) combination drawn from
//! {free, clamped, anchored} on three prediction grids:
//!   - the original training rows verbatim (replay invariant),
//!   - 32 brand-new interior rows (the original failure mode), and
//!   - the boundary endpoints themselves (where clamped/anchored constraints
//!     are most numerically delicate).
//!
//! Each (bc_left, bc_right) pair gets its own `#[test]` so the failure
//! attribution stays sharp. Assertions: predict must not panic, must return
//! the exact number of rows asked for, and must contain only finite values.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N_TRAIN: usize = 200;

fn make_training_data() -> (Vec<f64>, gam::data::EncodedDataset) {
    // Deterministic, jittered uniform-on-[0,1] grid — avoids any rand crate
    // dependency drift between rustc versions.
    let mut x: Vec<f64> = (0..N_TRAIN)
        .map(|i| {
            let base = (i as f64 + 0.5) / N_TRAIN as f64;
            // Tiny structured jitter so the design isn't a perfect lattice.
            let jitter = ((i as f64 * 12.9898).sin() * 43758.5453).fract() * 0.5e-3;
            (base + jitter).clamp(1e-6, 1.0 - 1e-6)
        })
        .collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode training data");
    (x, data)
}

fn formula_for(bc_left: &str, bc_right: &str) -> String {
    // Anchored sides require an explicit anchor value (the parser defaults to
    // 0 if absent, but make the contract explicit so the test is unambiguous).
    let mut opts: Vec<String> = vec!["k=10".to_string()];
    opts.push(format!("bc_left={bc_left}"));
    opts.push(format!("bc_right={bc_right}"));
    if bc_left == "anchored" {
        opts.push("anchor_left=0".to_string());
    }
    if bc_right == "anchored" {
        opts.push("anchor_right=0".to_string());
    }
    format!("y ~ s(x, {})", opts.join(", "))
}

fn run_one(bc_left: &str, bc_right: &str) {
    init_parallelism();
    let (x_train, data) = make_training_data();
    let formula = formula_for(bc_left, bc_right);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(&formula, &data, &cfg).unwrap_or_else(|e| panic!("fit `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit for `{formula}`")
    };

    // ── Grid 1: predict at the exact training rows ─────────────────────
    {
        let n = x_train.len();
        let mut m = Array2::<f64>::zeros((n, 2));
        for (i, &x) in x_train.iter().enumerate() {
            m[[i, 0]] = x;
            m[[i, 1]] = 0.0;
        }
        let design =
            build_term_collection_design(m.view(), &fit.resolvedspec).unwrap_or_else(|e| {
                panic!("predict-at-training rows rebuild failed for `{formula}`: {e:?}")
            });
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        assert_eq!(
            pred.len(),
            n,
            "training-row predict length mismatch for `{formula}`",
        );
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "training-row predict produced non-finite values for `{formula}`",
        );
    }

    // ── Grid 2: predict at 32 brand-new interior rows ─────────────────
    {
        let n = 32usize;
        let mut m = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let xt = 0.005 + 0.99 * (i as f64) / ((n - 1) as f64);
            m[[i, 0]] = xt;
            m[[i, 1]] = 0.0;
        }
        let design =
            build_term_collection_design(m.view(), &fit.resolvedspec).unwrap_or_else(|e| {
                panic!("predict-at-new-points rebuild failed for `{formula}`: {e:?}")
            });
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        assert_eq!(
            pred.len(),
            n,
            "new-points predict length mismatch for `{formula}`",
        );
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "new-points predict produced non-finite values for `{formula}`",
        );
    }

    // ── Grid 3: predict at the boundary endpoints themselves ───────────
    //
    // For clamped/anchored sides this is where the basis is constructively
    // pinned, so any miscomposition of Zb · Zi will surface as either a
    // shape error inside `build_term_collection_design` or non-finite
    // predictions.
    {
        let probes = [
            0.0_f64,
            1e-9,
            1e-6,
            1e-3,
            0.5,
            1.0 - 1e-3,
            1.0 - 1e-6,
            1.0 - 1e-9,
            1.0,
        ];
        let n = probes.len();
        let mut m = Array2::<f64>::zeros((n, 2));
        for (i, &x) in probes.iter().enumerate() {
            m[[i, 0]] = x;
            m[[i, 1]] = 0.0;
        }
        let design =
            build_term_collection_design(m.view(), &fit.resolvedspec).unwrap_or_else(|e| {
                panic!("predict-at-boundary rebuild failed for `{formula}`: {e:?}")
            });
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        assert_eq!(
            pred.len(),
            n,
            "boundary predict length mismatch for `{formula}`",
        );
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "boundary predict produced non-finite values for `{formula}`: {pred:?}",
        );
    }
}

// 3 × 3 = 9 combinations of (bc_left, bc_right) ∈ {free, clamped, anchored}².
//
// We use one #[test] per combination so the failing variant is named in the
// test report rather than buried in an aggregate panic message.

#[test]
fn bc_predict_invariants_free_free() {
    run_one("free", "free");
}
#[test]
fn bc_predict_invariants_free_clamped() {
    run_one("free", "clamped");
}
#[test]
fn bc_predict_invariants_free_anchored() {
    run_one("free", "anchored");
}
#[test]
fn bc_predict_invariants_clamped_free() {
    run_one("clamped", "free");
}
#[test]
fn bc_predict_invariants_clamped_clamped() {
    run_one("clamped", "clamped");
}
#[test]
fn bc_predict_invariants_clamped_anchored() {
    run_one("clamped", "anchored");
}
#[test]
fn bc_predict_invariants_anchored_free() {
    run_one("anchored", "free");
}
#[test]
fn bc_predict_invariants_anchored_clamped() {
    run_one("anchored", "clamped");
}
#[test]
fn bc_predict_invariants_anchored_anchored() {
    run_one("anchored", "anchored");
}
