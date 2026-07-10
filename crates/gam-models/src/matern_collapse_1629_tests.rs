//! Regression test for #1629: `matern(x1, x2)` recovered a fine 2-D surface
//! ~6× worse than `thinplate()`/`tensor()`, and `k=` had no effect.
//!
//! ROOT CAUSE: the cold build evaluates the K Matérn kernel columns at the
//! default seed length scale, realizes the design over the n data rows, and
//! applies the parametric-orthogonality identifiability transform — whose
//! spectral whitener is rank-revealing — BEFORE the κ-optimizer runs, then
//! freezes the surviving column count. When the seed length scale is as wide as
//! the full data diameter (the pre-fix default), every kernel column is
//! near-constant over the cloud, so the realized design Gram is numerically
//! rank-deficient (≈ K/4) and the whitener legitimately drops the collapsed
//! directions (e.g. 199 → 50). The freeze pins that collapse, so the κ-optimizer
//! can never recover the columns and `k=` is a near no-op. The optimizer also
//! starts in the maximally over-smoothed basin and parks there.
//!
//! FIX (landed on `main`, NOT in this test module — see `term_builder.rs`
//! "gam#1629" comment and `term_specs::auto_initial_length_scale`): route Matérn
//! through the same `length_scale = 0.0` auto-init sentinel that thin-plate uses,
//! so the planner's `auto_init_length_scale_in_place` replaces it with the
//! data-derived wiggly-side init `max_range / √n`. That seeds the basis in the
//! resolving regime — the realized design Gram is full rank, the freeze keeps the
//! requested columns, and the κ-optimizer refines the range from a non-degenerate
//! basis it can actually escape from. (My earlier branch proposed an inter-knot
//! `D/K^(1/d)` seed instead; the owner superseded it with the sentinel approach,
//! which I rebased onto. These tests guard the BEHAVIOR — full-rank cold design,
//! `k=` effective, truth-RMSE parity with thin-plate — regardless of which
//! seeding mechanism delivers it.)
//!
//! The per-center `matern_rank_reduce_centers` RRQR is NOT the culprit: it
//! reports full rank even at the diameter seed. The collapse was entirely in the
//! realized-design orthogonality whitener.
//!
//! `matern_cold_design_does_not_collapse_and_k_has_effect_1629` asserts the COLD
//! design (pre-REML) — the exact object the freeze pins — resolves ~all of its
//! requested basis columns instead of collapsing, and that `k=` changes the
//! resolved width. It is the fast (~15 s) gate guard.
//!
//! The slow end-to-end truth-RMSE comparison against thin-plate (full REML fit,
//! >1000 s) was verified once by hand (n=900, train=700, matched k=100:
//! matern=0.1593 vs thinplate=0.1770 — matern matches/slightly beats thinplate,
//! vs the original ~6× deficit) and is not kept as a gate test: a >1000 s full
//! fit cannot run in the suite, and `#[ignore]` is banned workspace-wide. The
//! cold-design column-count guard below is the fast proxy for the same property.
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
            csv::StringRecord::from(vec![x1[i].to_string(), x2[i].to_string(), y[i].to_string()])
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

    // tensor() was the OTHER good reference in #1629 (matern 6× worse than BOTH
    // thinplate AND tensor). It uses a different basis construction entirely, so
    // checking matern against it too guards against a thinplate-specific fluke.
    let tensor = cold_design_cols("y ~ tensor(x1, x2)", &ds, &cfg);
    assert!(
        matern_default + 40 >= tensor,
        "matern(x1, x2) ({matern_default} cols) must resolve a basis comparable to \
         tensor ({tensor} cols), not a small fraction of it (#1629 6× gap)"
    );

    // `k=` must be a LIVE knob, monotone in the requested dimension — pre-fix the
    // realized rank (not k) capped the basis, so every k collapsed to ~51/53 and
    // `k=` was a no-op. Sweep a descending progression and require the resolved
    // width to track it: each smaller k resolves a strictly smaller basis, and
    // each stays near its requested size (no collapse to the old ~50 floor).
    let mut prev = matern_default;
    for &k in &[150usize, 100, 60] {
        let cols = cold_design_cols(&format!("y ~ matern(x1, x2, k={k})"), &ds, &cfg);
        assert!(
            cols + 25 >= k,
            "matern(x1, x2, k={k}) cold design collapsed to {cols} columns; \
             expected ~{k} (#1629: realized rank, not k, was capping the basis)"
        );
        assert!(
            cols < prev,
            "matern k={k} ({cols} cols) must resolve a strictly smaller basis than the \
             next-larger k ({prev} cols); pre-fix `k=` had no effect because every \
             rail collapsed to the same realized rank (#1629)"
        );
        prev = cols;
    }

    // nu= must not reintroduce the collapse: a rougher kernel (nu=3/2) still has
    // to resolve a full-rank basis, since the seed/whitener/freeze interaction
    // that drove #1629 is independent of the smoothness order.
    let matern_nu32 = cold_design_cols("y ~ matern(x1, x2, nu=3/2)", &ds, &cfg);
    assert!(
        matern_nu32 >= 180,
        "matern(x1, x2, nu=3/2) cold design collapsed to {matern_nu32} columns; \
         the #1629 basis-collapse must not depend on the Matérn smoothness order"
    );
}
