//! #1346: the **dispersion location-scale** (GAMLSS) observation/prediction
//! interval was a symmetric `μ ± z·√(SE(μ̂)² + σ(x)²)` band — the equal-tailed
//! skew fix that landed for the standard single-block skewed families (#817
//! Gamma / #1193 NegativeBinomial / #1194 Beta) was never propagated to the
//! two-block dispersion path.
//!
//! The standard (single-block) Gamma path routes its observation interval
//! through the skew-aware `family_observation_band`, which builds the edges from
//! equal-tailed quantiles of a moment-matched Gamma predictive. On a right-skewed
//! Gamma the upper edge sits well above the mean and the lower edge well below
//! it, so `(hi−μ)/(μ−lo) ≫ 1` and each tail covers near nominal.
//!
//! The dispersion location-scale predictor overrode `observation_noise` to return
//! `Some(noise_sd)`, which the generic drivers consumed to build a *symmetric*
//! band — never calling the skew-aware construction. The fix routes the two-block
//! path through `DispersionLocationScalePredictor::observation_band`, the per-row
//! sibling of `family_observation_band`, folding the per-row `σ(x)` into the total
//! predictive variance. This test is the Rust gate for that contract: it mirrors
//! the committed Python acceptance test
//! `tests/test_bug_hunt_dispersion_location_scale_observation_interval_symmetric.py`.
//!
//! THE GATE: a Gamma dispersion-LS fit whose mean AND shape (hence σ(x)) both
//! vary across x must produce a *right-skewed* observation band
//! (`median (hi−μ)/(μ−lo) > 1.3`; the symmetric-band bug pins this at exactly
//! 1.000) with an upper tail that covers near nominal (`P(Y > hi) ≤ 0.04`; the
//! symmetric band undershoots the Gamma upper quantile, ~0.052 measured).

use gam::estimate::BlockRole;
use gam_predict::{
    DispersionLocationScalePredictor, InferenceCovarianceMode, PredictInput,
    PredictUncertaintyOptions, PredictableModel,
};
use gam::smooth::build_term_collection_design;
use gam::{
    DispersionLocationScaleFitResult, FitConfig, FitResult, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Gamma(shape k>0, scale theta) via Marsaglia–Tsang (k>=1) with the
    /// Ahrens–Dieter boost for k<1, off a single reproducible LCG.
    fn gamma(&mut self, k: f64, theta: f64) -> f64 {
        if k < 1.0 {
            let g = self.gamma(k + 1.0, theta);
            let u = self.unit().max(1e-300);
            return g * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * theta;
            }
        }
    }
}

/// Right-skewed Gamma whose mean AND shape (precision) both vary with x, so a
/// dispersion location-scale model has a genuine non-constant scale channel.
/// `Var(Y|μ) = μ²/ν` ⇒ skewness `2/√ν` (clearly right-skewed for these ν).
fn mu_true(x: f64) -> f64 {
    (0.6 + 1.0 * (2.0 * std::f64::consts::PI * x).sin()).exp()
}
fn shape_true(x: f64) -> f64 {
    (0.5 + 0.4 * (2.0 * std::f64::consts::PI * x).cos()).exp()
}

fn gen_gamma(n: usize, rng: &mut Lcg) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.unit();
        let mu = mu_true(xi);
        let nu = shape_true(xi);
        let yi = rng.gamma(nu, mu / nu).max(1e-9);
        xs.push(xi);
        ys.push(yi);
    }
    (xs, ys)
}

#[test]
fn dispersion_location_scale_observation_band_is_skewed_not_symmetric() {
    init_parallelism();

    let mut rng = Lcg(0x_13_46_u64);
    let (xtr, ytr) = gen_gamma(6000, &mut rng);
    let (xev, yev) = gen_gamma(30000, &mut rng);

    // ---- Gamma dispersion location-scale fit: smooth mean + smooth shape ----
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..xtr.len())
        .map(|i| {
            csv::StringRecord::from(vec![format!("{:.17e}", xtr[i]), format!("{:.17e}", ytr[i])])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode gamma data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        noise_formula: Some("s(x)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam gamma dispersion fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result");
    };

    let beta_mu = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-shape) block present")
        .beta
        .clone();
    let covariance = fit
        .fit
        .beta_covariance()
        .expect("joint covariance present")
        .clone();

    let predictor = DispersionLocationScalePredictor {
        beta_mu,
        beta_noise,
        likelihood: kind.likelihood_spec(),
        inverse_link: Some(kind.base_link()),
        covariance: Some(covariance),
    };

    // ---- predict the 95% observation interval on the 30k held-out sample ----
    let m = xev.len();
    let mut grid = Array2::<f64>::zeros((m, ncols));
    for (i, &xi) in xev.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let disp_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild dispersion design at grid");
    let input = PredictInput {
        design: mean_design.design,
        offset: Array1::<f64>::zeros(m),
        design_noise: Some(disp_design.design),
        offset_noise: Some(Array1::<f64>::zeros(m)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };

    let unc_options = PredictUncertaintyOptions {
        confidence_level: 0.95,
        covariance_mode: InferenceCovarianceMode::Conditional,
        includeobservation_interval: true,
        ..PredictUncertaintyOptions::default()
    };
    let pred = predictor
        .predict_full_uncertainty(&input, &fit.fit, &unc_options)
        .expect("full-uncertainty predict with observation interval");

    let lower = pred.observation_lower.expect("observation lower band");
    let upper = pred.observation_upper.expect("observation upper band");
    let mean = &pred.mean;

    // Bounds must be ordered, finite, and inside the positive Gamma support.
    for i in 0..m {
        assert!(
            lower[i].is_finite() && upper[i].is_finite() && lower[i] >= 0.0 && upper[i] > lower[i],
            "row {i}: degenerate observation bounds [{}, {}]",
            lower[i],
            upper[i]
        );
    }

    let n = m as f64;
    let two_sided = (0..m)
        .filter(|&i| yev[i] >= lower[i] && yev[i] <= upper[i])
        .count() as f64
        / n;
    let upper_tail = (0..m).filter(|&i| yev[i] > upper[i]).count() as f64 / n;

    // Median right-skew of the band: how much further the upper edge sits above
    // the mean than the lower edge sits below it. A symmetric μ ± z·σ band gives
    // exactly 1.0; an equal-tailed Gamma band gives ≫ 1.
    let mut ratios: Vec<f64> = (0..m)
        .filter_map(|i| {
            let below = mean[i] - lower[i];
            if below > 1e-12 {
                Some((upper[i] - mean[i]) / below)
            } else {
                None
            }
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let skew_ratio = if ratios.is_empty() {
        f64::NAN
    } else {
        ratios[ratios.len() / 2]
    };

    eprintln!(
        "[#1346 gamma dispersion-LS obs] two_sided={two_sided:.4} (nominal 0.95) \
         upper_tail={upper_tail:.4} (nominal 0.025) skew_ratio={skew_ratio:.4}"
    );

    // Sanity: it is a *shape* defect, not a width defect — total coverage near
    // nominal (passes both pre- and post-fix).
    assert!(
        two_sided > 0.90,
        "two-sided coverage collapsed: {two_sided:.4}"
    );

    // Core assertion: a right-skewed Gamma's predictive interval cannot be
    // symmetric about the mean. The buggy symmetric band gives skew_ratio == 1.0
    // exactly; an equal-tailed Gamma band gives ratio well above 1.
    assert!(
        skew_ratio > 1.3,
        "dispersion location-scale Gamma observation band is symmetric \
         (median (hi-μ)/(μ-lo)={skew_ratio:.4}); a right-skewed Gamma predictive \
         interval must be right-skewed (equal-tailed quantiles)."
    );

    // The symmetric band's upper edge sits below the true upper quantile, so the
    // upper tail under-covers (~0.052 measured) at a nominal 2.5%-per-tail band.
    assert!(
        upper_tail <= 0.04,
        "upper tail under-covers: P(Y > observation_upper)={upper_tail:.4} \
         (nominal 0.025); the symmetric band undershoots the Gamma upper quantile."
    );
}
