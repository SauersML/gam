//! Regression (#1541): a univariate `s(x, bs='cr')` / `bs='cs'` smooth must NOT
//! hard-fail the whole fit when the covariate has fewer distinct values than the
//! requested basis size `k`. Before commit 7f806ff, `select_cr_knots` demanded
//! "cubic regression spline with k=N requires at least N distinct values, got M"
//! and the entire model raised an `InvalidConfigurationError` — even though mgcv
//! (and gam's own already-fixed tensor margin) simply CAP `k` to the data
//! support and proceed.
//!
//! `capped_cr_marginal_knotspec()` (src/terms/term_builder.rs) now reduces the
//! cr/cs marginal `k` to the number of distinct covariate values, so a ternary
//! covariate `x ∈ {0,1,2}` fitted with `s(x, bs='cr', k=10)` builds a 3-knot cr
//! basis instead of erroring.
//!
//! This test fits `y ~ s(x, bs='cr', k=10) + s(z)` on a ternary covariate with
//! deliberately NON-MONOTONE per-level means ({0: 0.0, 1: +2.0, 2: -1.0}) and:
//!   (a) asserts the fit SUCCEEDS — the core regression guard; before 7f806ff it
//!       returned an error from the cr-knot selector;
//!   (b) predicts on the training grid and asserts the per-level x contrasts are
//!       recovered (x=1 ≈ +2, x=2 ≈ -1 relative to x=0), proving the *capped* cr
//!       basis still carries enough functions to represent all 3 levels.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_predict::predict_gam;
use ndarray::{Array1, Array2};

/// Deterministic SplitMix64 — no Python, no external RNG crate.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Non-monotone per-level mean for the ternary covariate x ∈ {0,1,2}.
fn group_mean(x: u8) -> f64 {
    match x {
        0 => 0.0,
        1 => 2.0,
        _ => -1.0,
    }
}

/// `n` rows: ternary covariate `x ∈ {0,1,2}` (deterministic, balanced), a smooth
/// covariate `z ~ U(0,1)`, and `y = group_mean(x) + sin(2π z) + 0.3 N(0,1)`.
fn build(seed: u64) -> (gam::data::EncodedDataset, Vec<(f64, f64)>) {
    let n = 501usize; // divisible by 3 → balanced ternary x
    let mut rng = SplitMix64::new(seed);
    let mut pts = Vec::with_capacity(n);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i % 3) as u8; // {0,1,2} in deterministic rotation
        let z = rng.unit();
        let f = group_mean(x) + (std::f64::consts::TAU * z).sin();
        let y = f + 0.3 * rng.normal();
        pts.push((x as f64, z));
        rows.push(StringRecord::from(vec![
            (x as f64).to_string(),
            z.to_string(),
            y.to_string(),
        ]));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode dataset"),
        pts,
    )
}

/// Fit `formula` (Gaussian identity) and predict the mean at every `(x, z)` in
/// `pts` through the public `build_term_collection_design` + `predict_gam` path.
fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> Vec<f64> {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) = fit_from_formula(formula, data, &cfg)
        .expect("low-cardinality cr fit must SUCCEED (#1541: capped to data support, not errored)")
    else {
        panic!("expected a standard Gaussian fit for {formula}");
    };

    let xi = data
        .headers
        .iter()
        .position(|h| h == "x")
        .expect("x column");
    let zi = data
        .headers
        .iter()
        .position(|h| h == "z")
        .expect("z column");
    let hlen = data.headers.len();
    let m = pts.len();
    let mut grid = Array2::<f64>::zeros((m, hlen));
    for (i, &(x, z)) in pts.iter().enumerate() {
        grid[[i, xi]] = x;
        grid[[i, zi]] = z;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at the prediction grid");
    let dense = design.design.to_dense();
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gam(dense, fit.fit.beta.view(), offset.view(), family)
        .expect("predict on the training grid");
    pred.mean.to_vec()
}

#[test]
fn univariate_cr_smooth_survives_low_cardinality_covariate() {
    // The z-marginal sin(2π z) averages out across each level's z's (all three
    // levels share the same U(0,1) z design), so the level-mean of the predicted
    // mean isolates the x main effect plus a common offset. Contrast vs x=0
    // recovers group_mean differences: +2 at x=1, -1 at x=2.
    //
    // ±0.4 is a real bound: 0.3·N(0,1) noise over ~167 rows/level gives a
    // level-mean SE on the order of 0.3/sqrt(167) ≈ 0.023, plus penalized-fit
    // shrinkage; ±0.4 is comfortably inside the +2 / -1 signal yet far tighter
    // than the contrast magnitudes themselves (a meaningless tolerance would be
    // ≳1.5, half the +2 separation).
    const CONTRAST_TOL: f64 = 0.4;

    let (data, pts) = build(20250624);

    // (a) Core regression guard: the fit must SUCCEED. fit_and_predict's
    // `.expect` on fit_from_formula is the assertion — before 7f806ff this is
    // where the InvalidConfigurationError surfaced.
    let pred = fit_and_predict("y ~ s(x, bs=\"cr\", k=10) + s(z)", &data, &pts);

    // (b) Recover the non-monotone per-level contrasts.
    let mut sum = [0.0_f64; 3];
    let mut cnt = [0usize; 3];
    for (&(x, _z), &mu) in pts.iter().zip(&pred) {
        let lvl = x.round() as usize;
        sum[lvl] += mu;
        cnt[lvl] += 1;
    }
    assert!(
        cnt.iter().all(|&c| c > 0),
        "fixture must populate all 3 x levels; got counts {cnt:?}"
    );
    let mean = [
        sum[0] / cnt[0] as f64,
        sum[1] / cnt[1] as f64,
        sum[2] / cnt[2] as f64,
    ];
    let c1 = mean[1] - mean[0]; // expected +2
    let c2 = mean[2] - mean[0]; // expected -1

    eprintln!(
        "[#1541] level means μ̂ = [{:.4}, {:.4}, {:.4}] | contrast x1−x0 = {c1:.4} (want +2) | \
         contrast x2−x0 = {c2:.4} (want −1)",
        mean[0], mean[1], mean[2]
    );

    assert!(
        (c1 - 2.0).abs() < CONTRAST_TOL,
        "capped cr basis failed to recover the x=1 contrast: got {c1:.4}, expected +2 within \
         ±{CONTRAST_TOL} — the data-support-capped cr basis does not carry enough functions for \
         3 levels (#1541)."
    );
    assert!(
        (c2 + 1.0).abs() < CONTRAST_TOL,
        "capped cr basis failed to recover the x=2 contrast: got {c2:.4}, expected −1 within \
         ±{CONTRAST_TOL} — the data-support-capped cr basis does not carry enough functions for \
         3 levels (#1541)."
    );
}
