//! Bug hunt (#1845): `mjs(x)` (measure-jet) predictions in a training GAP —
//! a stretch of the input range with no training support — collapse toward the
//! TRAINING MEAN instead of continuing the flank-attested linear trend.
//!
//! Root cause: the measure-jet design is a pure Gaussian representer
//! `K(data, centers)·z`. Gaussian representers decay to ~0 away from their
//! centers, so inside a training gap every column vanishes and the fit reverts
//! to the intercept — the response mean. The jet energy already leaves affine
//! functions UNPENALIZED, but the representer *span* cannot carry a trend across
//! the gap that the *penalty* would happily allow, so the bridge goes flat.
//!
//! Fix (thin-plate/Duchon analogue): append an unpenalized ambient-affine
//! extrapolation head to the design so it extends linearly across the gap (the
//! jet energy annihilates it, so it stays the unpenalized null space).
//!
//! This test builds a 1-D `mjs(x)` fit of a linear truth with a held-out gap in
//! the middle, then asserts the in-gap predictions FOLLOW THE TREND: the fitted
//! gap slope has the correct sign and a healthy fraction of the truth slope, and
//! the in-gap predictions deviate from the training mean far more than a
//! mean-collapsed bridge would. It is RED before the fix (gap slope ≈ 0, the
//! representer bridge is flat / mean-reverting) and GREEN after the affine head
//! carries the trend across the gap. It does NOT gate coverage / standard errors
//! (that is the separate contract-5 under-dispersion issue, not #1845).

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

/// Truth slope of the linear signal across the input range.
const TRUE_SLOPE: f64 = 3.0;
/// Held-out gap in x: no training row lands here, so a representer-only bridge
/// has no support and must lean on the affine head to carry the trend.
const GAP_LO: f64 = 0.42;
const GAP_HI: f64 = 0.62;

fn truth(x: f64) -> f64 {
    TRUE_SLOPE * (x - 0.5)
}

/// Training frame: n rows with x ~ U(0,1) EXCLUDING the gap (0.42, 0.62),
/// y = TRUE_SLOPE·(x − 0.5) + N(0, 0.15). Returns the encoded dataset.
fn build(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(seed);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    let mut made = 0usize;
    while made < n {
        let x = rng.unit();
        if x > GAP_LO && x < GAP_HI {
            continue; // held out — no training support in the gap
        }
        let y = truth(x) + 0.15 * rng.normal();
        rows.push(StringRecord::from(vec![x.to_string(), y.to_string()]));
        made += 1;
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

/// Fit `formula` (Gaussian identity) and return the fitted mean at every `x` in
/// `xs`, through the public design + `predict_gam` path (the surface a user
/// sees).
fn fit_and_predict(formula: &str, data: &gam::data::EncodedDataset, xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula(formula, data, &cfg).expect("standard measure-jet GAM fit")
    else {
        panic!("expected a standard GAM fit for {formula}");
    };
    let xi = data
        .headers
        .iter()
        .position(|h| h == "x")
        .expect("x column");
    let hlen = data.headers.len();
    let m = xs.len();
    let mut grid = Array2::<f64>::zeros((m, hlen));
    for (i, &x) in xs.iter().enumerate() {
        grid[[i, xi]] = x;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild measure-jet design at the prediction grid");
    let dense = design.design.to_dense();
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gam(dense, fit.fit.beta.view(), offset.view(), family)
        .expect("predict on the gap grid");
    pred.mean.to_vec()
}

/// Least-squares slope of `y` on `x`.
fn ls_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let xbar = xs.iter().sum::<f64>() / n;
    let ybar = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov += (x - xbar) * (y - ybar);
        var += (x - xbar) * (x - xbar);
    }
    cov / var
}

#[test]
fn measure_jet_gap_prediction_follows_trend_not_mean() {
    let data = build(0x1845_0704, 600);

    // Training mean of y — the value a mean-collapsed bridge reverts to.
    let yi = data
        .headers
        .iter()
        .position(|h| h == "y")
        .expect("y column");
    let train_mean = data.values.column(yi).sum() / data.values.nrows() as f64;

    // Test grid INSIDE the held-out gap.
    let gap_xs: Vec<f64> = (0..40)
        .map(|k| GAP_LO + (GAP_HI - GAP_LO) * (k as f64 + 0.5) / 40.0)
        .collect();
    let gap_truth: Vec<f64> = gap_xs.iter().map(|&x| truth(x)).collect();

    let pred = fit_and_predict("y ~ mjs(x, centers=18)", &data, &gap_xs);

    let slope_hat = ls_slope(&gap_xs, &pred);
    let mad_pred = pred
        .iter()
        .map(|p| (p - train_mean).abs())
        .sum::<f64>()
        / pred.len() as f64;
    let mad_truth = gap_truth
        .iter()
        .map(|t| (t - train_mean).abs())
        .sum::<f64>()
        / gap_truth.len() as f64;

    eprintln!(
        "[#1845] mjs(x) gap: slope_hat={slope_hat:.3} (truth {TRUE_SLOPE:.3}) \
         mad_pred={mad_pred:.3} mad_truth={mad_truth:.3} train_mean={train_mean:.3} \
         gap_pred[0]={:.3} gap_pred[last]={:.3}",
        pred.first().copied().unwrap_or(f64::NAN),
        pred.last().copied().unwrap_or(f64::NAN),
    );

    // Direction: the gap bridge must lean the SAME way as the truth trend, not
    // flatten to the mean (mean collapse ⇒ slope_hat ≈ 0, fails this).
    assert!(
        slope_hat * TRUE_SLOPE > 0.0,
        "measure-jet gap bridge has the wrong trend direction: fitted slope {slope_hat:.3} \
         vs truth {TRUE_SLOPE:.3}. The Gaussian representers decay to zero in the gap and the \
         fit reverts to the training mean; the affine extrapolation head must carry the trend \
         across the gap (#1845)."
    );

    // Magnitude: the gap slope must be a healthy fraction of the truth slope
    // (not a token non-zero) and must not blow up. A representer-only bridge
    // collapses to ~0 here; the affine head reproduces the flank-attested slope.
    assert!(
        slope_hat >= 0.4 * TRUE_SLOPE && slope_hat <= 2.0 * TRUE_SLOPE,
        "measure-jet gap bridge does not carry the flank trend: fitted slope {slope_hat:.3} \
         outside [{:.3}, {:.3}] around truth {TRUE_SLOPE:.3} (#1845)",
        0.4 * TRUE_SLOPE,
        2.0 * TRUE_SLOPE
    );

    // No collapse to the mean: in-gap predictions must deviate from the training
    // mean at least half as strongly as the truth does. A mean-reverting bridge
    // scores near zero here.
    assert!(
        mad_pred >= 0.5 * mad_truth,
        "measure-jet gap predictions collapse toward the training mean: \
         MAD {mad_pred:.3} vs truth MAD {mad_truth:.3} (mean {train_mean:.3}) (#1845)"
    );

    // Sanity: the fitted gap surface is not a constant (a constant would have
    // zero slope and be caught above, but pin it explicitly against a
    // degenerate/flat bridge).
    let pred_range = pred
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - pred.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        pred_range > 0.2 * (TRUE_SLOPE * (GAP_HI - GAP_LO)),
        "measure-jet gap bridge is nearly flat (range {pred_range:.3}); it collapsed instead of \
         extrapolating the trend (#1845)"
    );
}
