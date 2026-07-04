//! Bug hunt (#2122): a 2-D `matern(x, z)` radial smooth SILENTLY COLLAPSES to a
//! near-constant (mean-only) fit on ordinary Gaussian data, while its siblings
//! `thinplate(x, z)` and `duchon(x, z)` recover the same signal almost perfectly
//! on the IDENTICAL frame. The fit reports success (no error) but returns
//! essentially the response mean.
//!
//! Root cause: Matérn carries an isotropic range/κ spatial-geometry
//! hyperparameter selected by a nested joint-REML optimizer. On this frame the
//! joint κ/range solve STALLS (hits its iteration cap with a huge residual
//! gradient) and returns `SpatialJointOutcome::NonConverged`. The driver then
//! shipped `fit_frozen_baseline_geometry(...)` — the SPEC-DEFAULT range, which is
//! grossly too long for this frame — so the penalized fit shrinks every kernel
//! coefficient to ~0 and the surface degenerates to the intercept (the mean).
//!
//! Observed before the fix (n=400, truth std ≈ 0.705):
//!   thinplate(x,z): R²_vs_truth = +0.976  pred_std = 0.683
//!   duchon(x,z):    R²_vs_truth = +0.983  pred_std = 0.683
//!   matern(x,z):    R²_vs_truth = +0.008  pred_std = 0.023   <-- mean-only collapse
//!
//! The test fits `thinplate(x, z)` as an ANCHOR (proves the frame carries
//! recoverable signal), then asserts `matern(x, z)` on the SAME data does NOT
//! collapse: fitted-value std > 0.3 · truth-std AND R² vs truth > 0.5. It is RED
//! before the fix (matern fitted std ≈ 0.023 vs truth std ≈ 0.705) and GREEN
//! after the driver retreats to a data-adaptive geometry on a stalled solve.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_predict::predict_gam;
use ndarray::{Array1, Array2};

/// Deterministic SplitMix64 → no Python, no external RNG crate.
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
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller (one of the pair).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The #2122 frame: n=400, x,z ~ U(0,1), u ~ U(-2,2),
/// truth = sin(2πx) + 0.5 z, y = truth + 0.3 u + N(0, 0.2).
/// Returns the encoded dataset, the (x,z) grid, and the noise-free `truth`.
fn build(seed: u64) -> (gam::data::EncodedDataset, Vec<(f64, f64)>, Vec<f64>) {
    let n = 400usize;
    let mut rng = SplitMix64::new(seed);
    let mut pts = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.unit();
        let z = rng.unit();
        let u = -2.0 + 4.0 * rng.unit();
        let t = (std::f64::consts::TAU * x).sin() + 0.5 * z;
        let y = t + 0.3 * u + 0.2 * rng.normal();
        pts.push((x, z));
        truth.push(t);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            z.to_string(),
            y.to_string(),
        ]));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode dataset"),
        pts,
        truth,
    )
}

/// Fit `formula` (Gaussian identity) and return the fitted mean at every point
/// in `pts`, produced through the public design + `predict_gam` path so it is
/// exactly the surface a user sees.
fn fit_and_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> Vec<f64> {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula(formula, data, &cfg).expect("standard radial GAM fit")
    else {
        panic!("expected a standard GAM fit for {formula}");
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
        .expect("rebuild radial design at the prediction grid");
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

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

/// R² of `pred` against `truth`: 1 − SS_res/SS_tot.
fn r2_vs_truth(pred: &[f64], truth: &[f64]) -> f64 {
    let tbar = mean(truth);
    let ss_tot: f64 = truth.iter().map(|t| (t - tbar).powi(2)).sum();
    let ss_res: f64 = pred
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

#[test]
fn matern_2d_geometry_stall_does_not_collapse_to_mean() {
    let (data, pts, truth) = build(20240704);
    let truth_std = std(&truth);
    assert!(
        truth_std > 0.5,
        "degenerate truth (std {truth_std:.4}); the invariant would be vacuous"
    );

    // ANCHOR: thinplate — no enrolled kernel-range hyperparameter, so no joint
    // solve to stall — recovers the surface. This proves the frame carries
    // recoverable signal, so a matern collapse is a defect, not missing signal.
    let tp = fit_and_predict("y ~ thinplate(x, z)", &data, &pts);
    let tp_r2 = r2_vs_truth(&tp, &truth);
    eprintln!(
        "[#2122] thinplate(x,z): R²_vs_truth={tp_r2:.3} pred_std={:.3} (truth_std={truth_std:.3})",
        std(&tp)
    );
    assert!(
        tp_r2 > 0.7,
        "anchor thinplate(x,z) failed to recover the surface (R²={tp_r2:.3}); the frame does \
         not carry recoverable signal, so the matern invariant would be vacuous"
    );

    // matern on the SAME data must NOT collapse to the response mean.
    let mat = fit_and_predict("y ~ matern(x, z)", &data, &pts);
    let mat_std = std(&mat);
    let mat_r2 = r2_vs_truth(&mat, &truth);
    eprintln!(
        "[#2122] matern(x,z):    R²_vs_truth={mat_r2:.3} pred_std={mat_std:.3} (truth_std={truth_std:.3})"
    );

    assert!(
        mat_std > 0.3 * truth_std,
        "matern(x,z) COLLAPSED to a near-constant surface: fitted std {mat_std:.4} is below \
         0.3·truth_std ({:.4}). The joint κ/range solve stalled (NonConverged) and the driver \
         shipped the spec-default geometry, over-smoothing every kernel coefficient to ~0 so the \
         surface degenerates to the response mean (#2122).",
        0.3 * truth_std
    );
    assert!(
        mat_r2 > 0.5,
        "matern(x,z) did not recover the surface: R²_vs_truth={mat_r2:.3} ≤ 0.5, while the \
         thinplate anchor on the same data reaches {tp_r2:.3}. A stalled joint geometry solve \
         must retreat to a data-adaptive range, not ship the mean-only spec-default (#2122)."
    );
}
