//! Fit-quality correctness sweep across geometric smooths.
//!
//! Each test:
//!   - constructs a small dataset with a deterministic LCG noise model,
//!   - fits a GAM via `fit_from_formula`,
//!   - rebuilds the prediction design via the frozen `resolvedspec`, and
//!   - compares predictions (on the linear-predictor scale, or after inverse
//!     link for binomial/poisson) against either ground truth or boundary
//!     conditions with mathematically-justified strict tolerances.
//!
//! Tolerances are derived analytically from the noise SD and sample size, not
//! tuned to a particular fit. They are STRICT (≤ 1.2 × noise variance for MSPE;
//! ≤ 3 noise/√N for BC anchor recovery) and are NOT to be relaxed.
//!
//! Conventions reused from the existing suite (sphere_pole_continuity.rs and
//! cylinder_tensor_seam_continuity.rs): rebuild the prediction design via
//! `build_term_collection_design` so the frozen `resolvedspec` indexes into the
//! correct columns of the prediction matrix.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::f64::consts::{PI, TAU};

// -----------------------------------------------------------------------------
// Deterministic noise (LCG + Box-Muller) — same construction the existing
// noisy_cylinder_data helper in biobank_perf_benchmark.rs uses.
// -----------------------------------------------------------------------------

struct LcgNormal {
    state: u64,
}

impl LcgNormal {
    fn new(seed: u64) -> Self {
        Self {
            // Avoid the all-zero state, which would freeze the LCG output.
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }

    /// Uniform on (0, 1].
    fn next_unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 1.0) / ((u32::MAX as f64) + 1.0)
    }

    /// Standard normal via Box-Muller (one draw per call; we discard the
    /// second.  This keeps draws independent of how callers consume them).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

// -----------------------------------------------------------------------------
// Common helpers
// -----------------------------------------------------------------------------

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

fn binomial_cfg() -> FitConfig {
    FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    }
}

fn poisson_cfg() -> FitConfig {
    FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    }
}

/// Build an EncodedDataset from named columns of equal length.
fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> gam::data::EncodedDataset {
    assert_eq!(
        headers.len(),
        columns.len(),
        "headers/columns count mismatch"
    );
    let n = columns[0].len();
    for c in columns.iter() {
        assert_eq!(c.len(), n, "all columns must have the same length");
    }
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode dataset")
}

/// Build a prediction matrix whose column layout matches `headers`. Caller
/// fills the predictor columns; the response column is left as zero (it is
/// never referenced by smooth construction).
fn predict_matrix(n_cols: usize, columns_in_order: &[&[f64]]) -> Array2<f64> {
    let n_rows = columns_in_order[0].len();
    let mut m = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, col) in columns_in_order.iter().enumerate() {
        for i in 0..n_rows {
            m[[i, j]] = col[i];
        }
    }
    m
}

/// Fit a standard Gaussian formula and return the prediction at `test_rows`.
/// `test_rows` must already be laid out to match the training column order.
fn fit_and_predict_eta(
    formula: &str,
    data: &gam::data::EncodedDataset,
    cfg: &FitConfig,
    test_rows: &Array2<f64>,
) -> Array1<f64> {
    let result = fit_from_formula(formula, data, cfg).expect("fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let test_design = build_term_collection_design(test_rows.view(), &fit.resolvedspec)
        .expect("rebuild prediction design");
    assert_eq!(
        test_design.design.ncols(),
        fit.fit.beta.len(),
        "predict design width != beta length"
    );
    test_design.design.apply(&fit.fit.beta)
}

fn mean_squared_error(predicted: &Array1<f64>, truth: &Array1<f64>) -> f64 {
    assert_eq!(predicted.len(), truth.len());
    let n = predicted.len() as f64;
    let mut acc = 0.0;
    for (p, t) in predicted.iter().zip(truth.iter()) {
        let d = p - t;
        acc += d * d;
    }
    acc / n
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// =============================================================================
// 1. Recovery on noisy data — MSPE ≤ 1.2 × noise variance against truth
// =============================================================================

/// y = cos(theta) + noise on [0, 2π); fit `cyclic(theta)`; check MSPE.
///
/// Justification for tolerance: with a well-specified smoother the prediction
/// bias is O(λ · ||f''||²) which the REML inner loop shrinks to negligible
/// for a single sinusoid sampled densely; the residual MSPE is bounded by
/// noise variance × (p / N).  For p ≈ 10 and N = 400 the bias-floor budget is
/// well below 0.2 × σ², so 1.2 × σ² is comfortably above the noise-only floor
/// while still strict (it rejects any pathology that doubles the noise floor).
#[test]
fn recovery_periodic_1d_cyclic_cos_noise_01() {
    init_parallelism();
    let n = 400usize;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0xC0FFEE_u64);
    let theta: Vec<f64> = (0..n).map(|i| TAU * (i as f64) / (n as f64)).collect();
    let truth: Vec<f64> = theta.iter().map(|t| t.cos()).collect();
    let y: Vec<f64> = truth
        .iter()
        .map(|t| t + noise_sd * rng.next_normal())
        .collect();
    let data = encode_columns(&["theta", "y"], &[&theta, &y]);

    // Evaluate on a dense grid distinct from training nodes.
    let n_grid = 200usize;
    let theta_g: Vec<f64> = (0..n_grid)
        .map(|i| TAU * (i as f64 + 0.37) / (n_grid as f64))
        .collect();
    let truth_g: Array1<f64> = theta_g.iter().map(|t| t.cos()).collect();
    let test = predict_matrix(2, &[&theta_g, &vec![0.0; n_grid]]);

    let pred = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_g);
    let tol = 1.2 * noise_var;
    eprintln!("[recovery-cyclic-1d] N={n} MSE={mse:.5e} noise_var={noise_var:.5e} tol={tol:.5e}");
    assert!(
        mse <= tol,
        "cyclic(theta) MSPE {mse:.5e} > 1.2·σ² = {tol:.5e}"
    );
}

/// Cylinder: y = cos(theta) + 0.3·h + noise; fit te(theta, h, periodic).
#[test]
fn recovery_cylinder_te_periodic_natural() {
    init_parallelism();
    let n_theta = 24usize;
    let n_h = 8usize;
    let n = n_theta * n_h;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0xBEEF1234_u64);

    let mut theta = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let f = t.cos() + 0.3 * hh;
            theta.push(t);
            h.push(hh);
            truth.push(f);
            y.push(f + noise_sd * rng.next_normal());
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);

    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let truth_arr: Array1<f64> = Array1::from(truth);
    let pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=5)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_arr);
    let tol = 1.2 * noise_var;
    eprintln!("[recovery-cylinder] N={n} MSE={mse:.5e} noise_var={noise_var:.5e} tol={tol:.5e}");
    assert!(
        mse <= tol,
        "te(theta,h periodic) MSPE {mse:.5e} > 1.2·σ² = {tol:.5e}"
    );
}

/// Sphere harmonic recovery: y = sin(lat_rad) + cos(2·lon_rad)·cos(lat_rad) + noise.
/// The truth is degree ≤ 2 so max_degree=4 contains it exactly.
#[test]
fn recovery_sphere_harmonic_low_degree_signal() {
    init_parallelism();
    let n_lat = 12usize;
    let n_lon = 24usize;
    let n = n_lat * n_lon;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0xA5A5_5A5A_u64);
    let mut lat = Vec::with_capacity(n);
    let mut lon = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n_lat {
        // Exclude poles (sphere construction prefers data off pole exactly).
        let la = -80.0 + 160.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lo = -180.0 + 360.0 * (j as f64) / (n_lon as f64);
            let lar = la.to_radians();
            let lor = lo.to_radians();
            let f = lar.sin() + (2.0 * lor).cos() * lar.cos();
            lat.push(la);
            lon.push(lo);
            truth.push(f);
            y.push(f + noise_sd * rng.next_normal());
        }
    }
    let data = encode_columns(&["lat", "lon", "y"], &[&lat, &lon, &y]);
    let test = predict_matrix(3, &[&lat, &lon, &vec![0.0; n]]);
    let truth_arr: Array1<f64> = Array1::from(truth);
    let pred = fit_and_predict_eta(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_arr);
    let tol = 1.2 * noise_var;
    eprintln!("[recovery-sphere-harm] N={n} MSE={mse:.5e} noise_var={noise_var:.5e} tol={tol:.5e}");
    assert!(
        mse <= tol,
        "sphere(harmonic) MSPE {mse:.5e} > 1.2·σ² = {tol:.5e}"
    );
}

/// Anchored BC at x=0: y = x(1-x) sin(πx) + noise, where f(0) = 0.
/// Tolerance for f̂(0) ≈ 0: 3·σ/√N ≈ 3·0.1/√(N) (predicted-mean SE bound).
#[test]
fn recovery_bc_anchored_left_endpoint_recovery() {
    init_parallelism();
    let n = 200usize;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0x1234_5678_u64);
    let x: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let truth: Vec<f64> = x
        .iter()
        .map(|xi| xi * (1.0 - xi) * (PI * xi).sin())
        .collect();
    let y: Vec<f64> = truth
        .iter()
        .map(|t| t + noise_sd * rng.next_normal())
        .collect();
    let data = encode_columns(&["x", "y"], &[&x, &y]);

    // Evaluate fit at original training x.
    let test = predict_matrix(2, &[&x, &vec![0.0; n]]);
    let truth_arr: Array1<f64> = Array1::from(truth.clone());
    let pred = fit_and_predict_eta(
        "y ~ s(x, bc_left=anchored, anchor_left=0, k=10)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_arr);
    let tol = 1.2 * noise_var;
    eprintln!("[recovery-bc-anchored] N={n} MSE={mse:.5e} noise_var={noise_var:.5e} tol={tol:.5e}");
    assert!(
        mse <= tol,
        "bc-anchored MSPE {mse:.5e} > 1.2·σ² = {tol:.5e}"
    );

    // Anchor recovery: f̂(0) should be exactly zero by construction of the
    // anchored basis (the row at x = 0 is identically zero in the
    // reparameterized design — see feature_correctness_sweep.rs).  Allow only
    // floating-point noise — the BC is structural, not statistical.
    let f0 = pred[0];
    eprintln!("[recovery-bc-anchored] f_hat(0) = {f0:.3e}");
    assert!(
        f0.abs() < 1.0e-9,
        "anchored BC at x=0 was violated: f_hat(0)={f0:.6e} (structural BC, not stochastic)"
    );
}

// =============================================================================
// 2. Boundary behavior — seam continuity for periodic, anchor for BC
// =============================================================================

#[test]
fn boundary_cyclic_seam_continuity_zero_vs_tau() {
    init_parallelism();
    let n = 200usize;
    let mut rng = LcgNormal::new(0xFACEFEED_u64);
    let theta: Vec<f64> = (0..n).map(|i| TAU * (i as f64) / (n as f64)).collect();
    let y: Vec<f64> = theta
        .iter()
        .map(|t| t.cos() + 0.05 * rng.next_normal())
        .collect();
    let data = encode_columns(&["theta", "y"], &[&theta, &y]);

    // Predict at seam: theta = 0 and theta = 2π must agree to numerical precision.
    let theta_zero = vec![0.0_f64; 4];
    let theta_tau = vec![TAU; 4];
    let test_zero = predict_matrix(2, &[&theta_zero, &[0.0; 4]]);
    let test_tau = predict_matrix(2, &[&theta_tau, &[0.0; 4]]);
    let pred_zero = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data,
        &gaussian_cfg(),
        &test_zero,
    );
    let pred_tau = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data,
        &gaussian_cfg(),
        &test_tau,
    );
    let mut max_gap = 0.0_f64;
    for (a, b) in pred_zero.iter().zip(pred_tau.iter()) {
        max_gap = max_gap.max((a - b).abs());
    }
    eprintln!("[boundary-cyclic-seam] max |f(0) - f(2π)| = {max_gap:.3e}");
    assert!(
        max_gap < 1.0e-9,
        "cyclic seam mismatch: max gap {max_gap:.3e} ≥ 1e-9"
    );
}

#[test]
fn boundary_bc_anchored_exact_zero_at_left_endpoint() {
    init_parallelism();
    let n = 80usize;
    let mut rng = LcgNormal::new(0x42_42_42_u64);
    let x: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|xi| (PI * xi).sin() * xi + 0.05 * rng.next_normal())
        .collect();
    let data = encode_columns(&["x", "y"], &[&x, &y]);

    // Predict at exactly x = 0 (and a few near-zero points to verify
    // continuity).
    let x_probe = vec![0.0_f64, 1.0e-9, 1.0e-6];
    let test = predict_matrix(2, &[&x_probe, &vec![0.0; x_probe.len()]]);
    let pred = fit_and_predict_eta(
        "y ~ s(x, bc_left=anchored, anchor_left=0, k=10)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    eprintln!(
        "[boundary-bc-anchored] f(0)={:.3e} f(1e-9)={:.3e} f(1e-6)={:.3e}",
        pred[0], pred[1], pred[2]
    );
    // f(0) is structurally zero (anchored BC reparameterization).
    assert!(
        pred[0].abs() < 1.0e-10,
        "anchored f(0) must be structurally zero; got {:.3e}",
        pred[0]
    );
    // Continuity at the anchor: f(1e-9) must be within O(slope · 1e-9) of zero.
    // Slope is bounded (smooth fit), so the perturbation should be at the
    // 1e-7 level at worst.  This is conservative.
    assert!(
        pred[1].abs() < 1.0e-6,
        "f(1e-9) should be near 0; got {:.3e}",
        pred[1]
    );
}

// =============================================================================
// 3. Multi-feature compound: cyclic + bc-anchored + sphere
// =============================================================================

/// Compound additive model exercising three distinct smooth families
/// simultaneously.  All three components are additive in the truth, so we
/// can predict and verify reasonable global fit quality.
#[test]
fn compound_cyclic_plus_anchored_plus_sphere_fits_jointly() {
    init_parallelism();
    let n = 800usize;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0xCABBA9E_u64);
    let mut theta = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut lat = Vec::with_capacity(n);
    let mut lon = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    // Use a low-discrepancy sequence over the four predictors for stable
    // coverage at modest N.
    let phi1 = 0.6180339887498949_f64;
    let phi2 = 0.7548776662466927_f64;
    let phi3 = 0.5698402909980532_f64;
    for i in 0..n {
        let frac = (i as f64 + 0.5) / (n as f64);
        let u1 = ((i as f64) * phi1).fract();
        let u2 = ((i as f64) * phi2).fract();
        let u3 = ((i as f64) * phi3).fract();
        let t = TAU * u1;
        let xi = frac;
        // Uniform on sphere (Lambert equal-area).
        let z = 1.0 - 2.0 * u2;
        let lo_rad = TAU * u3 - PI;
        let la_rad = z.asin();
        let la_deg = la_rad.to_degrees().clamp(-85.0, 85.0);
        let lo_deg = lo_rad.to_degrees();

        // Truth: additive cyclic + anchored + sphere components.  Anchored
        // component vanishes at x = 0 so the overall truth at x = 0 equals
        // cyclic(t) + sphere(la, lo).
        let f_cyclic = t.cos();
        let f_anchored = xi * (1.0 - xi) * (PI * xi).sin();
        let f_sphere = la_deg.to_radians().sin();
        let f = f_cyclic + f_anchored + f_sphere;
        theta.push(t);
        x.push(xi);
        lat.push(la_deg);
        lon.push(lo_deg);
        truth.push(f);
        y.push(f + noise_sd * rng.next_normal());
    }
    let data = encode_columns(
        &["theta", "x", "lat", "lon", "y"],
        &[&theta, &x, &lat, &lon, &y],
    );
    let test = predict_matrix(5, &[&theta, &x, &lat, &lon, &vec![0.0; n]]);
    let truth_arr: Array1<f64> = Array1::from(truth);
    let pred = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586) + s(x, bc_left=anchored, anchor_left=0) + sphere(lat, lon, method=harmonic, max_degree=3)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_arr);
    // Compound model has more parameters, so the bias-floor budget is
    // larger.  Allow 2 × noise variance for the compound aggregate — still
    // strict (it rejects pathologies that double the noise floor).
    let tol = 2.0 * noise_var;
    eprintln!("[compound] N={n} MSE={mse:.5e} noise_var={noise_var:.5e} tol={tol:.5e}");
    assert!(mse <= tol, "compound MSPE {mse:.5e} > 2·σ² = {tol:.5e}");
}

// =============================================================================
// 4. Family coverage: cylinder with binomial / poisson / gaussian
// =============================================================================

/// Binomial (logit) cylinder: data drawn from p = σ(η_true(theta,h)); recovery
/// is measured on the probability scale.  Bernoulli observations carry
/// variance p(1-p) ≤ 0.25, so even a perfect fit cannot drive sample-level
/// MSPE below the Bernoulli variance floor.  We instead compare against the
/// TRUE probability surface (denoising) — for which the bias-variance
/// tradeoff gives MSPE on order p / N.  Tolerance is 0.02 (much less than
/// the 0.25 Bernoulli variance floor; tight given N=600 and a smooth signal
/// with no aliasing).
#[test]
fn family_binomial_cylinder_recovers_probability_surface() {
    init_parallelism();
    let n_theta = 30usize;
    let n_h = 20usize;
    let n = n_theta * n_h;
    let mut rng = LcgNormal::new(0xB1B1_C0C0_u64);
    let mut theta = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut p_true = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let eta = 0.55 * t.cos() - 0.25 * (2.0 * t).sin() + 0.3 * hh;
            let p = logistic(eta);
            // Bernoulli draw via uniform RNG (re-using LCG via Box-Muller cosine
            // produces correlated u, so use the unit draw directly).
            let u = rng.next_unit();
            theta.push(t);
            h.push(hh);
            p_true.push(p);
            y.push(if u < p { 1.0 } else { 0.0 });
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let p_true_arr: Array1<f64> = Array1::from(p_true);
    let eta_pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &binomial_cfg(),
        &test,
    );
    let p_pred: Array1<f64> = eta_pred.mapv(logistic);
    let mse_p = mean_squared_error(&p_pred, &p_true_arr);
    let tol = 0.02;
    eprintln!("[family-binomial] N={n} MSE(p_pred, p_true)={mse_p:.4e} tol={tol:.4e}");
    assert!(
        mse_p <= tol,
        "binomial cylinder probability MSPE {mse_p:.4e} > {tol:.4e}"
    );
}

/// Poisson (log) cylinder: y = Poisson(exp(η_true)).  Recovery measured
/// against the TRUE rate exp(η_true).  Tolerance allows for Poisson
/// variability scaled by the mean.
#[test]
fn family_poisson_cylinder_recovers_rate_surface() {
    init_parallelism();
    let n_theta = 30usize;
    let n_h = 20usize;
    let n = n_theta * n_h;
    let mut rng = LcgNormal::new(0xDEADC0DE_u64);
    let mut theta = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut rate_true = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            // Keep rate in a moderate range so a small count is reasonable.
            let eta = 1.5 + 0.4 * t.cos() + 0.2 * hh;
            let lam = eta.exp();
            // Knuth Poisson via exponential interarrival (fine for small λ ~ 4-6).
            let mut k = 0u32;
            let mut s = 0.0_f64;
            loop {
                s += -rng.next_unit().ln();
                if s > lam {
                    break;
                }
                k += 1;
                if k > 100 {
                    break;
                }
            }
            theta.push(t);
            h.push(hh);
            rate_true.push(lam);
            y.push(k as f64);
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let rate_true_arr: Array1<f64> = Array1::from(rate_true.clone());
    let eta_pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=4)",
        &data,
        &poisson_cfg(),
        &test,
    );
    let rate_pred: Array1<f64> = eta_pred.mapv(f64::exp);
    // Relative MSPE bounded by (Var/mean) · (p/N) ≈ 1/mean · p/N.
    let mean_rate = rate_true.iter().sum::<f64>() / (n as f64);
    let mse_rate = mean_squared_error(&rate_pred, &rate_true_arr);
    // For mean rate ~ exp(1.5) ≈ 4.5, the Poisson variance per observation is
    // ~ 4.5, and the smoothed-rate variance is at most σ² · p / N ≈
    // 4.5 · 30/600 = 0.225.  A factor of 2 headroom gives 0.5.
    let tol = 0.5;
    eprintln!(
        "[family-poisson] N={n} mean_rate≈{mean_rate:.2} MSE(rate)={mse_rate:.4e} tol={tol:.4e}"
    );
    assert!(
        mse_rate <= tol,
        "poisson rate MSPE {mse_rate:.4e} > {tol:.4e}"
    );
}

/// Gaussian cylinder is already exercised by `recovery_cylinder_te_periodic_natural`,
/// but the family coverage matrix demands an explicit Gaussian fit too.
/// Here we use a smaller N to exercise the Gaussian path independently and
/// pin a tight MSPE.
#[test]
fn family_gaussian_cylinder_recovers_with_tight_tolerance() {
    init_parallelism();
    let n_theta = 20usize;
    let n_h = 10usize;
    let n = n_theta * n_h;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0xCAFE_BABE_u64);
    let mut theta = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hh = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let f = t.cos() + 0.3 * hh;
            theta.push(t);
            h.push(hh);
            truth.push(f);
            y.push(f + noise_sd * rng.next_normal());
        }
    }
    let data = encode_columns(&["theta", "h", "y"], &[&theta, &h, &y]);
    let test = predict_matrix(3, &[&theta, &h, &vec![0.0; n]]);
    let truth_arr: Array1<f64> = Array1::from(truth);
    let pred = fit_and_predict_eta(
        "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None], k=5)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_arr);
    let tol = 1.2 * noise_var;
    eprintln!("[family-gaussian] N={n} MSE={mse:.5e} tol={tol:.5e}");
    assert!(
        mse <= tol,
        "gaussian cylinder MSPE {mse:.5e} > 1.2·σ² = {tol:.5e}"
    );
}

// =============================================================================
// 5. Edge cases
// =============================================================================

/// All-y-identical (zero response variance): the fit must succeed, predictions
/// must equal the constant response, and the smoothing parameters must NOT
/// blow up to NaN/inf.  A pathological optimizer would either fail or
/// produce wild βs.
#[test]
fn edge_all_same_y_yields_constant_prediction() {
    init_parallelism();
    let n = 60usize;
    let theta: Vec<f64> = (0..n).map(|i| TAU * (i as f64) / (n as f64)).collect();
    let y_const = 1.7_f64;
    let y = vec![y_const; n];
    let data = encode_columns(&["theta", "y"], &[&theta, &y]);
    let test = predict_matrix(2, &[&theta, &vec![0.0; n]]);
    let pred = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mut max_dev = 0.0_f64;
    for v in pred.iter() {
        assert!(
            v.is_finite(),
            "non-finite prediction in zero-variance fit: {v}"
        );
        max_dev = max_dev.max((v - y_const).abs());
    }
    eprintln!("[edge-zero-variance] max |pred - {y_const}| = {max_dev:.3e}");
    // The mean-only solution is exact; allow only floating-point noise.
    // (The smooth nullspace contains the constant, so smoothing -> ∞ recovers
    // the mean exactly.)
    assert!(
        max_dev < 1.0e-6,
        "constant-y fit deviated by {max_dev:.3e} from the truth"
    );
}

/// N = 10·p: a comfortably over-determined fit with k = 8 (so p ≈ 8) and
/// N = 80.  Verifies that the fit converges and stays within the noise
/// envelope.
#[test]
fn edge_n_eq_ten_p_overdetermined_fit_within_envelope() {
    init_parallelism();
    let k = 8usize;
    let n = 10 * k;
    let noise_sd = 0.1_f64;
    let noise_var = noise_sd * noise_sd;
    let mut rng = LcgNormal::new(0x33_44_55_66_u64);
    let theta: Vec<f64> = (0..n).map(|i| TAU * (i as f64) / (n as f64)).collect();
    let truth: Vec<f64> = theta.iter().map(|t| t.cos()).collect();
    let y: Vec<f64> = truth
        .iter()
        .map(|t| t + noise_sd * rng.next_normal())
        .collect();
    let data = encode_columns(&["theta", "y"], &[&theta, &y]);
    let test = predict_matrix(2, &[&theta, &vec![0.0; n]]);
    let truth_arr: Array1<f64> = Array1::from(truth);
    let pred = fit_and_predict_eta(
        &format!("y ~ cyclic(theta, k={k}, period_start=0, period_end=6.283185307179586)"),
        &data,
        &gaussian_cfg(),
        &test,
    );
    let mse = mean_squared_error(&pred, &truth_arr);
    // At N = 10·p the bias-variance budget loosens by a factor of ~10/inf-ratio,
    // but the true noise is still σ² so we allow up to 1.5·σ².
    let tol = 1.5 * noise_var;
    eprintln!("[edge-N10p] N={n} k={k} MSE={mse:.5e} tol={tol:.5e}");
    assert!(mse <= tol, "N=10p fit MSPE {mse:.5e} > 1.5·σ² = {tol:.5e}");
}

/// Single category in a non-fit covariate: include a constant column that the
/// formula DOES NOT reference.  The fit must succeed regardless (the column
/// is invisible to construction).  This guards against silent dependence on
/// dataset shape beyond the named columns.
#[test]
fn edge_single_category_unused_covariate_does_not_affect_fit() {
    init_parallelism();
    let n = 120usize;
    let noise_sd = 0.05_f64;
    let mut rng = LcgNormal::new(0xC1C1_2D2D_u64);
    let theta: Vec<f64> = (0..n).map(|i| TAU * (i as f64) / (n as f64)).collect();
    let y: Vec<f64> = theta
        .iter()
        .map(|t| t.cos() + noise_sd * rng.next_normal())
        .collect();
    // Unused covariate is constant (single category as a numeric).
    let unused = vec![7.0_f64; n];
    let data_with = encode_columns(&["theta", "unused", "y"], &[&theta, &unused, &y]);
    let data_without = encode_columns(&["theta", "y"], &[&theta, &y]);

    let theta_grid: Vec<f64> = (0..50).map(|i| TAU * (i as f64 + 0.23) / 50.0).collect();
    let test_with = predict_matrix(3, &[&theta_grid, &vec![7.0; 50], &vec![0.0; 50]]);
    let test_without = predict_matrix(2, &[&theta_grid, &vec![0.0; 50]]);

    let pred_with = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data_with,
        &gaussian_cfg(),
        &test_with,
    );
    let pred_without = fit_and_predict_eta(
        "y ~ cyclic(theta, period_start=0, period_end=6.283185307179586)",
        &data_without,
        &gaussian_cfg(),
        &test_without,
    );
    let mut max_diff = 0.0_f64;
    for (a, b) in pred_with.iter().zip(pred_without.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    eprintln!("[edge-unused-covariate] max |pred_with - pred_without| = {max_diff:.3e}");
    // The unused covariate has no statistical role; fits must be identical
    // up to floating-point noise.
    assert!(
        max_diff < 1.0e-8,
        "unused constant covariate changed the fit by {max_diff:.3e}"
    );
}
