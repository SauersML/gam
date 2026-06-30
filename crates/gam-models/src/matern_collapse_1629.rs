//! Regression test for #1629: `matern(x1, x2)` recovered a fine 2-D surface
//! ~6× worse than `thinplate()`/`tensor()`, and `k=` had no effect.
//!
//! ROOT CAUSE (confirmed empirically through `build_term_collection_design`):
//! `default_matern_length_scale` seeded the κ-optimizer at the FULL DATA
//! DIAMETER (set by #1074 to mirror mgcv's `bs="gp"` default range). The cold
//! build evaluates the K Matérn kernel columns at that seed, realizes the design
//! over the n data rows, and applies the parametric-orthogonality identifiability
//! transform — whose spectral whitener is rank-revealing — BEFORE the κ-optimizer
//! runs, then freezes the surviving column count. At a range as wide as the
//! diameter every kernel column is near-constant over the cloud, so the realized
//! design Gram is numerically rank-deficient (≈ K/4) and the whitener legitimately
//! drops the collapsed directions (e.g. 199 → 50). The freeze pins that collapse,
//! so the optimizer can never recover the columns and `k=` is a near no-op.
//!
//! (Notably the per-center `matern_rank_reduce_centers` RRQR is NOT the culprit:
//! it reports full rank even at the diameter seed. The collapse is entirely in the
//! realized-design orthogonality whitener.)
//!
//! FIX: seed at the basis's natural operating scale — the inter-knot spacing
//! `D / K^(1/d)` — so adjacent kernel columns stay distinct, the realized design
//! Gram is full rank, the freeze keeps all K columns, and the κ-optimizer tunes
//! the range freely from a non-degenerate basis.
//!
//! This test asserts the COLD design (pre-REML) — the exact object the freeze
//! pins — resolves ~all of its requested basis columns instead of collapsing, and
//! that `k=` changes the resolved width. It is the fast (~15 s) faithful proxy for
//! the full-fit truth-RMSE recovery (a full 4-formula fit took >1700 s).
#![cfg(test)]

use crate::fit_orchestration::{FitConfig, FitRequest, materialize};
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::smooth::build_term_collection_design;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(a: f64, b: f64) -> f64 {
    (1.2 * a).sin() * (1.1 * b).cos()
        + 0.6 * (3.0 * a).sin() * (2.8 * b).cos()
        + 0.4 * (5.0 * a + 0.5 * b).sin()
}

fn make_dataset(n: usize) -> gam_data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(424242);
    let u = Uniform::new(-3.0, 3.0).expect("uniform");
    let noise = Normal::new(0.0, 0.15).expect("normal");
    let x1: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| truth(x1[i], x2[i]) + noise.sample(&mut rng))
        .collect();
    let headers = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Resolve a formula to its cold (pre-REML) `TermCollectionDesign` and return the
/// total realized design column count — the width that the freeze step pins
/// before the κ-optimizer runs.
fn cold_design_cols(formula: &str, ds: &gam_data::EncodedDataset, cfg: &FitConfig) -> usize {
    let mat = materialize(formula, ds, cfg).expect("materialize");
    let FitRequest::Standard(req) = &mat.request else {
        panic!("expected a standard fit request for '{formula}'");
    };
    let design = build_term_collection_design(req.data.view(), &req.spec).expect("cold design");
    design.design.ncols()
}

#[test]
fn matern_cold_design_does_not_collapse_and_k_has_effect_1629() {
    let ds = make_dataset(2000);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // Default matern: the auto center count for n=2000, d=2 is 200, so the cold
    // design (intercept + 199 centered kernel columns) is ~200 wide. Pre-fix
    // (diameter seed) it collapsed to ~51. Require the basis to stay essentially
    // full so the κ-optimizer starts from a rich, non-degenerate basis.
    let matern_default = cold_design_cols("y ~ matern(x1, x2)", &ds, &cfg);
    assert!(
        matern_default >= 180,
        "matern(x1, x2) cold design collapsed to {matern_default} columns \
         (#1629: the diameter seed's realized-design orthogonality whitener pruned \
         the basis pre-optimization). Expected ~200."
    );

    // thinplate is the never-collapsing reference (#1629 reported it at ~200).
    // matern must now resolve a comparable basis width, not a fraction of it.
    let thinplate = cold_design_cols("y ~ thinplate(x1, x2)", &ds, &cfg);
    assert!(
        matern_default + 40 >= thinplate,
        "matern(x1, x2) ({matern_default} cols) must resolve a basis comparable to \
         thinplate ({thinplate} cols), not a small fraction of it (#1629 6× gap)"
    );

    // `k=150` must actually shrink the resolved basis relative to the default
    // k=200 — pre-fix both rails collapsed to ~51/53 so `k=` was a no-op.
    let matern_k150 = cold_design_cols("y ~ matern(x1, x2, k=150)", &ds, &cfg);
    assert!(
        matern_k150 >= 130,
        "matern(x1, x2, k=150) cold design collapsed to {matern_k150} columns; \
         expected ~150 (#1629)"
    );
    assert!(
        matern_k150 < matern_default,
        "k=150 ({matern_k150} cols) must resolve a strictly smaller basis than the \
         default k=200 ({matern_default} cols); pre-fix `k=` had no effect because \
         both collapsed to the same realized rank (#1629)"
    );
}
