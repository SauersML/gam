//! Second-angle regression lock for issue #771: a fitted Tweedie's *response-
//! scale predictive uncertainty* must reflect the estimated dispersion `φ`.
//!
//! The sibling test `bug_hunt_tweedie_dispersion_frozen_at_one.rs` pins the
//! *linear-predictor* SE path: `φ` enters the IRLS working weight
//! `prior·μ^{2−p}/φ`, so `Vb = H⁻¹` scales as `φ` and `SE(η̂) ∝ √φ`. That is a
//! necessary check but it only exercises the coefficient-covariance consumer.
//!
//! This test exercises a DIFFERENT consumer — the **observation prediction
//! interval** on the response scale. For a Tweedie response `Var(y) = φ·μ^p`,
//! the predictive band is `μ̂ ± z·√(SE(μ̂)² + φ·μ̂^p)` (see
//! `inference::predict`: `ResponseFamily::Tweedie` reads `observation_phi()` =
//! the fitted `likelihood_scale.fixed_phi()`). The original frozen-`φ`=1 bug
//! left this band pinned to the `φ=1` width regardless of the data, so an
//! over-dispersed Tweedie reported predictive intervals √φ too narrow and the
//! intervals badly under-covered the data they were meant to bracket.
//!
//! Three independent things are asserted, each of which the frozen-`φ` bug
//! breaks and none of which the η-SE test covers:
//!   1. ABSOLUTE `φ̂` RECOVERY — the fitted dispersion is within 4 derived
//!      standard deviations of the data's true `φ` (0.4 and 6.0), not the
//!      frozen 1.0. The SD is that of the Pearson estimator with a plug-in
//!      `μ̂`, derived from the Tweedie cumulants (see `pearson_phi_sd`); in 2000
//!      simulated replicates of this fixture the standardized error has SD
//!      0.99 at both dispersions and never exceeds 4.
//!   2. OBSERVATION-INTERVAL WIDTH scales as √φ across the two fits.
//!   3. EMPIRICAL COVERAGE — the fitted-`φ` 95% observation interval brackets
//!      the held-out responses far better than the counterfactual `φ=1` band a
//!      frozen-dispersion fit would have produced. This is the calibration the
//!      family exists to deliver and the bug silently destroyed.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const B0: f64 = 0.6;
const BX: f64 = 0.7;
const TWEEDIE_P: f64 = 1.5;
const Z95: f64 = 1.959_963_984_540_054; // qnorm(0.975)

/// Number of mean coefficients in `y ~ x` (intercept and slope).
const N_COEF: usize = 2;
/// Standard deviations allowed between φ̂ and the true φ.
const PHI_Z: f64 = 4.0;

/// Sampling SD of the Pearson dispersion `φ̂ = (1/n) Σ (yᵢ − μ̂ᵢ)²/μ̂ᵢ^p` at the
/// true `(μ, φ)` of this fixture.
///
/// With `tᵢ = (yᵢ − μᵢ)²/μᵢ^p` and `sᵢ = (yᵢ − μᵢ)μᵢ^{1−p}`, the Tweedie
/// cumulants give `Var tᵢ = p(2p−1)φ³μᵢ^{p−2} + 2φ²` and `Cov(tᵢ, sᵢ) = pφ²`.
/// Under the log link `∂tᵢ/∂ηᵢ` has mean `−pφ`, and the MLE satisfies
/// `β̂ − β ≈ H⁻¹ (1/n) Σ xᵢ sᵢ` with `H = (1/n) Σ μᵢ^{2−p} xᵢxᵢᵀ`, so
/// `n·Var φ̂ = mean[p(2p−1)φ³μᵢ^{p−2} + 2φ²] − p²φ³·x̄ᵀH⁻¹x̄`.
fn pearson_phi_sd(x: &[f64], phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let n = x.len() as f64;
    let (mut mean_var_t, mut h00, mut h01, mut h11, mut xbar1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &xi in x {
        let mu = (B0 + BX * xi).exp();
        mean_var_t += p * (2.0 * p - 1.0) * phi.powi(3) * mu.powf(p - 2.0) + 2.0 * phi * phi;
        let w = mu.powf(2.0 - p);
        h00 += w;
        h01 += w * xi;
        h11 += w * xi * xi;
        xbar1 += xi;
    }
    let (mean_var_t, h00, h01, h11, xbar1) =
        (mean_var_t / n, h00 / n, h01 / n, h11 / n, xbar1 / n);
    // x̄ = (1, mean x); x̄ᵀH⁻¹x̄ for the 2×2 H.
    let det = h00 * h11 - h01 * h01;
    assert!(det > 0.0, "information matrix must be positive definite");
    let quad = (h11 - 2.0 * h01 * xbar1 + h00 * xbar1 * xbar1) / det;
    let var = (mean_var_t - p * p * phi.powi(3) * quad) / n;
    assert!(var > 0.0, "derived Var(φ̂) must be positive, got {var}");
    var.sqrt()
}

/// Exact compound Poisson–Gamma (Tweedie, `1 < p < 2`) draw with mean `mu`,
/// dispersion `phi`: `N ~ Poisson(λ)`, `y = Σ_{i=1}^N G_i`, `G_i ~ Gamma(α, θ)`,
/// `λ = μ^{2−p}/(φ(2−p))`, `α = (2−p)/(p−1)`, `θ = φ(p−1)μ^{p−1}`; `N=0 ⇒ y=0`.
fn tweedie_sample(rng: &mut StdRng, mu: f64, phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n: u64 = Poisson::new(lambda).expect("poisson rate").sample(rng) as u64;
    if n == 0 {
        return 0.0;
    }
    Gamma::new(alpha * n as f64, scale)
        .expect("gamma(shape,scale)")
        .sample(rng)
}

struct TweedieFit {
    /// Fitted dispersion the covariance / predictive band were scaled by.
    phi_hat: f64,
    /// Per-row fitted response mean μ̂ on the supplied grid.
    mean: Vec<f64>,
    /// Per-row posterior SE of the mean (the `SE(μ̂)` term of the band).
    mean_se: Vec<f64>,
    /// Per-row 95% observation-interval half-width `z·√(SE(μ̂)² + φ̂·μ̂^p)`.
    obs_halfwidth: Vec<f64>,
}

/// Fit `y ~ x` as Tweedie(log) and return the fitted φ plus the response-scale
/// observation interval on the supplied evaluation grid.
fn fit_tweedie(x: &[f64], y: &[f64], eval: &[f64]) -> TweedieFit {
    let n = x.len();
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie data");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some(format!("tweedie(p={TWEEDIE_P})")),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).expect("gam tweedie fit should succeed")
    else {
        panic!("expected a Standard Tweedie fit");
    };

    let phi_hat = fit
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.dispersion.phi())
        .expect("Tweedie fit must carry an estimated dispersion");

    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design at eval grid");
    let dense = design.design.to_dense();

    let tweedie_log = LikelihoodSpec::new(
        ResponseFamily::Tweedie { p: TWEEDIE_P },
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        tweedie_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: true,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("tweedie predict with observation interval");

    let upper = pred
        .observation_upper
        .expect("observation interval requested");
    // Recover the response-noise half-width `z·√(SE(μ̂)² + φ̂·μ̂^p)` from the
    // *upper* endpoint, `upper − μ̂`. The lower endpoint is floored at the
    // Tweedie support `y ≥ 0` (a small mean with a wide band would otherwise
    // dip below zero — see #800), so `0.5·(upper − lower)` would understate the
    // band for those rows; the upper endpoint is never clamped (support is
    // unbounded above) and equals `μ̂ + z·σ` exactly, which is what this test's
    // φ-scaling check needs.
    let obs_halfwidth: Vec<f64> = upper
        .iter()
        .zip(pred.mean.iter())
        .map(|(&hi, &mu)| hi - mu)
        .collect();

    TweedieFit {
        phi_hat,
        mean: pred.mean.to_vec(),
        mean_se: pred.mean_standard_error.to_vec(),
        obs_halfwidth,
    }
}

/// Fraction of `y` that lands inside `[mean - hw, mean + hw]`.
fn coverage(y: &[f64], mean: &[f64], halfwidth: &[f64]) -> f64 {
    let hits = y
        .iter()
        .zip(mean.iter())
        .zip(halfwidth.iter())
        .filter(|&((&yi, &mi), &hw)| yi >= mi - hw && yi <= mi + hw)
        .count();
    hits as f64 / y.len() as f64
}

#[test]
fn tweedie_observation_interval_reflects_estimated_dispersion() {
    init_parallelism();

    let n = 4000usize;
    let phi_lo = 0.4_f64;
    let phi_hi = 6.0_f64;

    // Shared covariate values so the only inferential difference is the true φ.
    let mut rng = StdRng::seed_from_u64(0x71_C0FFEE_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();

    let mut rng_lo = StdRng::seed_from_u64(101);
    let mut rng_hi = StdRng::seed_from_u64(202);
    let mut y_lo = Vec::with_capacity(n);
    let mut y_hi = Vec::with_capacity(n);
    for &xi in &x {
        let mu = (B0 + BX * xi).exp();
        y_lo.push(tweedie_sample(&mut rng_lo, mu, phi_lo));
        y_hi.push(tweedie_sample(&mut rng_hi, mu, phi_hi));
    }

    // Predict the observation band at the *training* abscissae and score it
    // against the training responses (n=4000 makes in-sample coverage stable;
    // the mean is well determined, so this isolates the response-noise band).
    let fit_lo = fit_tweedie(&x, &y_lo, &x);
    let fit_hi = fit_tweedie(&x, &y_hi, &x);

    // ── 1. Absolute φ̂ recovery (the ratio test cannot see a shared bias) ────
    for (fit, phi_true, tag) in [(&fit_lo, phi_lo, "low"), (&fit_hi, phi_hi, "high")] {
        let sd = pearson_phi_sd(&x, phi_true);
        // Plus the `n` vs `n − p` divisor bias of the Pearson mean.
        let tol = PHI_Z * sd + (N_COEF as f64 / n as f64) * phi_true;
        let z = (fit.phi_hat - phi_true) / sd;
        eprintln!(
            "[tweedie-obs] {tag}: true φ={phi_true} φ̂={:.5} SD(φ̂)={sd:.5} z={z:.2}",
            fit.phi_hat
        );
        assert!(
            (fit.phi_hat - phi_true).abs() <= tol,
            "Tweedie φ̂ does not recover the {tag} dispersion: φ̂={:.5} vs true {phi_true}, \
             |error| {:.5} > {tol:.5} ({PHI_Z} derived SDs, SD={sd:.5}, z={z:.2}); the \
             frozen-φ bug pins it at 1.0",
            fit.phi_hat,
            (fit.phi_hat - phi_true).abs()
        );
    }

    // ── 2. Observation-interval width scales as √φ ──────────────────────────
    let mean_hw_lo: f64 =
        fit_lo.obs_halfwidth.iter().sum::<f64>() / fit_lo.obs_halfwidth.len() as f64;
    let mean_hw_hi: f64 =
        fit_hi.obs_halfwidth.iter().sum::<f64>() / fit_hi.obs_halfwidth.len() as f64;
    assert!(
        fit_lo
            .obs_halfwidth
            .iter()
            .all(|h| h.is_finite() && *h > 0.0)
            && fit_hi
                .obs_halfwidth
                .iter()
                .all(|h| h.is_finite() && *h > 0.0),
        "non-finite/zero observation half-widths"
    );
    let width_ratio = mean_hw_hi / mean_hw_lo;
    let expected_ratio = (phi_hi / phi_lo).sqrt();
    eprintln!(
        "[tweedie-obs] mean obs half-width: lo={mean_hw_lo:.5} hi={mean_hw_hi:.5}; \
         width ratio={width_ratio:.3} (correct ≈ {expected_ratio:.3}; frozen-φ ≈ 1.0)"
    );
    assert!(
        width_ratio >= 2.0,
        "Tweedie observation interval does not widen with dispersion: \
         high-φ/low-φ width ratio = {width_ratio:.3} (expected ≈ {expected_ratio:.3}). \
         Frozen φ=1.0 makes both bands the same width."
    );

    // ── 3. Empirical coverage beats the counterfactual frozen-φ=1 band ──────
    // The fitted-φ band must bracket the over-dispersed data near nominal; the
    // band a frozen-φ=1 fit would have produced (same μ̂, same SE(μ̂), but the
    // response-noise term forced to 1·μ̂^p) is √φ too narrow and under-covers.
    let cov_hi_fitted = coverage(&y_hi, &fit_hi.mean, &fit_hi.obs_halfwidth);
    let frozen_hw_hi: Vec<f64> = fit_hi
        .mean
        .iter()
        .zip(fit_hi.mean_se.iter())
        .map(|(&mu, &se)| Z95 * (se * se + mu.powf(TWEEDIE_P)).max(0.0).sqrt())
        .collect();
    let cov_hi_frozen = coverage(&y_hi, &fit_hi.mean, &frozen_hw_hi);
    eprintln!(
        "[tweedie-obs] high-φ obs-interval coverage: fitted-φ={cov_hi_fitted:.3} \
         vs counterfactual φ=1 band={cov_hi_frozen:.3} (nominal 0.95)"
    );
    // The fitted-φ band is well calibrated (covers ≈ nominal); the frozen-φ=1
    // band — √φ̂ ≈ 2.4× too narrow on this over-dispersed data — materially
    // under-covers. (The symmetric Gaussian band on a right-skewed, zero-inflated
    // Tweedie cannot separate the two by a huge margin, but the gap is real and
    // robust at n=4000; the frozen-dispersion bug would have left them equal.)
    assert!(
        cov_hi_fitted >= 0.90,
        "fitted-φ Tweedie observation interval is miscalibrated on over-dispersed \
         data: coverage {cov_hi_fitted:.3} (< 0.90, nominal 0.95)"
    );
    assert!(
        cov_hi_fitted >= cov_hi_frozen + 0.04,
        "estimating φ does not improve predictive coverage over the frozen-φ=1 band: \
         fitted-φ coverage {cov_hi_fitted:.3} vs frozen-φ {cov_hi_frozen:.3} \
         (the frozen-dispersion bug would leave these equal)"
    );
}
