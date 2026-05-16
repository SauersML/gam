//! Adversarial fit-quality stress probes across the geometric smooth families.
//!
//! These tests do NOT enforce tight numerical tolerances: their job is to
//! generate honest, structured quality data by stressing each smooth with
//! known-hard signals (high frequency, sharp bumps, discontinuities,
//! heteroscedastic noise, outliers, etc.), fit it via `gam::fit_from_formula`,
//! and categorize the result as PASS / DEGRADED / COLLAPSED based on the
//! prediction RMSE relative to the noise scale and the recovered span
//! relative to the truth span.
//!
//! Each probe `eprintln!`s a single `[fit-quality]` line so the user can grep
//! the output. The only hard assertion is non-finiteness — a smooth that
//! produces NaN/Inf is an actual bug and fails the test.
//!
//! Deterministic LCG seeding is used throughout (no rand crate dependency on
//! the noise streams) so reruns and CI are bit-identical.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

// ---------- deterministic uniform / normal generators ----------

/// Tiny LCG (numerical recipes constants); state never exposed outside this
/// module, so reproducibility is bit-stable across rustc versions.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        )
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform_01(&mut self) -> f64 {
        // top 53 bits to f64 in [0, 1)
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    /// Marsaglia polar method (standard normal).
    fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.uniform_01() - 1.0;
            let v = 2.0 * self.uniform_01() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}

// ---------- categorization ----------

#[derive(Debug, Clone, Copy)]
enum Category {
    Pass,
    Degraded,
    Collapsed,
}

impl Category {
    fn label(&self) -> &'static str {
        match self {
            Category::Pass => "PASS",
            Category::Degraded => "DEGRADED",
            Category::Collapsed => "COLLAPSED",
        }
    }
}

/// Classify based on rmse vs noise sd and recovered-span vs truth-span ratio.
///
/// Categories (in order of precedence):
///   COLLAPSED if span_ratio < 0.30
///   PASS      if rmse < 2 σ AND span_ratio >= 0.70
///   DEGRADED  if rmse < 5 σ
///   COLLAPSED otherwise
fn categorize(rmse: f64, sigma: f64, span_ratio: f64) -> Category {
    let sigma_eff = sigma.max(1e-9);
    if span_ratio < 0.30 {
        Category::Collapsed
    } else if rmse < 2.0 * sigma_eff && span_ratio >= 0.70 {
        Category::Pass
    } else if rmse < 5.0 * sigma_eff {
        Category::Degraded
    } else {
        Category::Collapsed
    }
}

fn rmse(yhat: &[f64], truth: &[f64]) -> f64 {
    let n = yhat.len() as f64;
    let s: f64 = yhat
        .iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    (s / n).sqrt()
}

fn span(v: &[f64]) -> f64 {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

fn truth_residual(yhat: &[f64], truth: &[f64]) -> f64 {
    // L1 mean residual at truth — a complementary metric to rmse.
    let n = yhat.len() as f64;
    let s: f64 = yhat
        .iter()
        .zip(truth.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    s / n
}

fn check_finite(name: &str, formula: &str, yhat: &[f64], beta: &[f64]) {
    let bad_pred = yhat.iter().filter(|v| !v.is_finite()).count();
    let bad_beta = beta.iter().filter(|v| !v.is_finite()).count();
    assert!(
        bad_pred == 0 && bad_beta == 0,
        "[fit-quality] {name} produced non-finite output: {bad_pred} bad preds, \
         {bad_beta} bad betas — formula `{formula}`",
    );
}

/// Emit a structured error category when fit-or-predict fails. The task
/// asked us to hard-assert only on non-finite outputs — a fit that errors
/// before producing any prediction is reported (not panicked) so the
/// downstream probes still run and the suite produces an honest table.
fn report_fit_error(probe: &str, formula: &str, err: &str) {
    let cat = if err.contains("frozen identifiability transform mismatch") {
        // Specific known ticket: see tests/bc_clamped_predict_shape_bug.rs.
        "PREDICT_DESIGN_BUG"
    } else if err.contains("rebuild design failed") {
        "PREDICT_FAILED"
    } else {
        "FIT_FAILED"
    };
    eprintln!("[fit-quality] probe={probe} category={cat} formula=`{formula}` err=`{err}`",);
}

fn report(
    probe: &str,
    formula: &str,
    rmse_val: f64,
    sigma: f64,
    span_fit: f64,
    span_truth: f64,
    extra: &str,
) -> Category {
    let ratio = if span_truth.abs() < 1e-12 {
        1.0
    } else {
        span_fit / span_truth
    };
    let cat = categorize(rmse_val, sigma, ratio);
    eprintln!(
        "[fit-quality] probe={probe} category={cat} rmse={rmse_val:.4} \
         sigma_noise={sigma:.4} span_fit={span_fit:.3} span_truth={span_truth:.3} \
         span_ratio={ratio:.3}{ws}{extra} formula=`{formula}`",
        cat = cat.label(),
        ws = if extra.is_empty() { "" } else { " " },
    );
    cat
}

// ---------- dataset & predict helpers ----------

fn make_dataset_1d(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect::<Vec<_>>();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 1d")
}

fn make_dataset_named_1d(x_name: &str, x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = [x_name, "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode named 1d")
}

fn make_dataset_2d_named(
    a_name: &str,
    a: &[f64],
    b_name: &str,
    b: &[f64],
    y: &[f64],
) -> gam::data::EncodedDataset {
    let headers = [a_name, b_name, "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = a
        .iter()
        .zip(b.iter())
        .zip(y.iter())
        .map(|((a, b), c)| StringRecord::from(vec![a.to_string(), b.to_string(), c.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 2d")
}

fn fit_predict_1d(
    formula: &str,
    data: &gam::data::EncodedDataset,
    x_grid: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg)?;
    let FitResult::Standard(fit) = result else {
        return Err("expected standard fit".to_string());
    };
    let n = x_grid.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        m[[i, 0]] = x_grid[i];
        m[[i, 1]] = 0.0;
    }
    let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("rebuild design failed: {e:?}"))?;
    let pred = test_design.design.apply(&fit.fit.beta).to_vec();
    Ok((pred, fit.fit.beta.to_vec()))
}

fn fit_predict_2d(
    formula: &str,
    data: &gam::data::EncodedDataset,
    a_grid: &[f64],
    b_grid: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg)?;
    let FitResult::Standard(fit) = result else {
        return Err("expected standard fit".to_string());
    };
    let n = a_grid.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = a_grid[i];
        m[[i, 1]] = b_grid[i];
        m[[i, 2]] = 0.0;
    }
    let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("rebuild design failed: {e:?}"))?;
    let pred = test_design.design.apply(&fit.fit.beta).to_vec();
    Ok((pred, fit.fit.beta.to_vec()))
}

// =====================================================================
// Probe 1: high-frequency truths sin(2π k x), k in {4, 6, 8, 10}
// =====================================================================

fn hifreq_cyclic_probe(k: usize) -> Category {
    init_parallelism();
    let n: usize = 400;
    let sigma = 0.10;
    let mut rng = Lcg::new(0x51 * (k as u64) + 7);
    let theta: Vec<f64> = (0..n).map(|_| TAU * rng.uniform_01()).collect();
    let y_truth: Vec<f64> = theta.iter().map(|t| (k as f64 * t).sin()).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_named_1d("theta", &theta, &y_noisy);
    let kb = (2 * k + 6).max(12);
    let formula =
        format!("y ~ cyclic(theta, k={kb}, period_start=0, period_end=6.283185307179586)");

    let mgrid: usize = 400;
    let theta_grid: Vec<f64> = (0..mgrid)
        .map(|i| TAU * (i as f64 + 0.5) / mgrid as f64)
        .collect();
    let truth: Vec<f64> = theta_grid.iter().map(|t| (k as f64 * t).sin()).collect();

    let probe = format!("hifreq_cyclic_k{k}");
    let (yhat, beta) = match fit_predict_1d(&formula, &data, &theta_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error(&probe, &formula, &e);
            return Category::Collapsed;
        }
    };
    check_finite(&probe, &formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let extra = format!("k={k} l1_at_truth={l1:.4}");
    report(&probe, &formula, r, sigma, sf, st, &extra)
}

#[test]
fn hifreq_cyclic_k4() {
    let _ = hifreq_cyclic_probe(4);
}
#[test]
fn hifreq_cyclic_k6() {
    let _ = hifreq_cyclic_probe(6);
}
#[test]
fn hifreq_cyclic_k8() {
    let _ = hifreq_cyclic_probe(8);
}
#[test]
fn hifreq_cyclic_k10() {
    let _ = hifreq_cyclic_probe(10);
}

fn hifreq_bc_probe(k: usize) -> Category {
    init_parallelism();
    let n: usize = 400;
    let sigma = 0.10;
    let mut rng = Lcg::new(0x77 * (k as u64) + 19);
    let x: Vec<f64> = (0..n).map(|_| rng.uniform_01()).collect();
    let y_truth: Vec<f64> = x.iter().map(|t| (TAU * k as f64 * t).sin()).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_1d(&x, &y_noisy);
    let kb = (2 * k + 8).max(14);
    let formula = format!("y ~ s(x, bc=anchored, k={kb})");

    let mgrid: usize = 400;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.005 + 0.99 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = x_grid.iter().map(|t| (TAU * k as f64 * t).sin()).collect();

    let probe = format!("hifreq_bc_k{k}");
    let (yhat, beta) = match fit_predict_1d(&formula, &data, &x_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error(&probe, &formula, &e);
            return Category::Collapsed;
        }
    };
    check_finite(&probe, &formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let extra = format!("k={k} l1_at_truth={l1:.4}");
    report(&probe, &formula, r, sigma, sf, st, &extra)
}

#[test]
fn hifreq_bc_k4() {
    let _ = hifreq_bc_probe(4);
}
#[test]
fn hifreq_bc_k6() {
    let _ = hifreq_bc_probe(6);
}
#[test]
fn hifreq_bc_k8() {
    let _ = hifreq_bc_probe(8);
}
#[test]
fn hifreq_bc_k10() {
    let _ = hifreq_bc_probe(10);
}

/// Spherical-harmonic ground-truth signal of degree l (l in {4, 6, 8}).
/// We use the zonal harmonic P_l(sin(lat)) which has the cleanest closed
/// form and provides a high-frequency latitude oscillation; the
/// spherical smooth, if it works at this max_degree, should capture it.
fn legendre_p(l: usize, x: f64) -> f64 {
    if l == 0 {
        return 1.0;
    }
    if l == 1 {
        return x;
    }
    let mut p_prev = 1.0;
    let mut p_curr = x;
    for n in 2..=l {
        let nf = n as f64;
        let p_next = ((2.0 * nf - 1.0) * x * p_curr - (nf - 1.0) * p_prev) / nf;
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

fn hifreq_sphere_probe(l: usize) -> Category {
    init_parallelism();
    // Quasi-uniform sphere sampling via Fibonacci spiral so we cover all
    // latitudes including near-pole bands.
    let n: usize = 800;
    let sigma = 0.10;
    let mut rng = Lcg::new(0xA1 * (l as u64) + 3);
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut lat_deg = Vec::with_capacity(n);
    let mut lon_deg = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let z = 1.0 - 2.0 * t; // cos(colat)
        let lat = z.asin().to_degrees(); // = lat in degrees
        let lon = (((i as f64) / golden).fract() * 360.0) - 180.0;
        lat_deg.push(lat);
        lon_deg.push(lon);
    }
    let y_truth: Vec<f64> = lat_deg
        .iter()
        .map(|d| legendre_p(l, (d.to_radians()).sin()))
        .collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_2d_named("lat", &lat_deg, "lon", &lon_deg, &y_noisy);
    let max_deg = l + 2;
    let formula = format!("y ~ sphere(lat, lon, method=harmonic, max_degree={max_deg})");

    // Test grid: zonal stripes
    let lat_test: Vec<f64> = (0..180).map(|i| -89.0 + 178.0 * i as f64 / 179.0).collect();
    let lon_test: Vec<f64> = vec![0.0; lat_test.len()];
    let truth: Vec<f64> = lat_test
        .iter()
        .map(|d| legendre_p(l, (d.to_radians()).sin()))
        .collect();

    let probe = format!("hifreq_sphere_l{l}");
    let (yhat, beta) = match fit_predict_2d(&formula, &data, &lat_test, &lon_test) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error(&probe, &formula, &e);
            return Category::Collapsed;
        }
    };
    check_finite(&probe, &formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let extra = format!("l={l} l1_at_truth={l1:.4}");
    report(&probe, &formula, r, sigma, sf, st, &extra)
}

#[test]
fn hifreq_sphere_l4() {
    let _ = hifreq_sphere_probe(4);
}
#[test]
fn hifreq_sphere_l6() {
    let _ = hifreq_sphere_probe(6);
}
#[test]
fn hifreq_sphere_l8() {
    let _ = hifreq_sphere_probe(8);
}

fn hifreq_tensor_probe(k: usize) -> Category {
    init_parallelism();
    let n_theta = 24;
    let n_h = 24;
    let sigma = 0.10;
    let mut rng = Lcg::new(0xD7 * (k as u64) + 41);
    let mut theta = Vec::with_capacity(n_theta * n_h);
    let mut h = Vec::with_capacity(n_theta * n_h);
    let mut y_truth = Vec::with_capacity(n_theta * n_h);
    let mut y_noisy = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        let t = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let hv = -0.95 + 1.9 * (j as f64) / ((n_h - 1) as f64);
            let yt = (k as f64 * t).sin() * (PI * hv).cos();
            theta.push(t);
            h.push(hv);
            y_truth.push(yt);
            y_noisy.push(yt + sigma * rng.normal());
        }
    }
    let data = make_dataset_2d_named("theta", &theta, "h", &h, &y_noisy);
    let kb = (2 * k + 4).max(10);
    let formula =
        format!("y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k={kb})");

    // Test grid
    let g: Vec<f64> = (0..25).map(|i| 0.02 + 0.96 * i as f64 / 24.0).collect();
    let mut t_test = Vec::new();
    let mut h_test = Vec::new();
    let mut truth = Vec::new();
    for &gx in &g {
        let t = TAU * gx;
        for &gy in &g {
            let hv = -0.95 + 1.9 * gy;
            t_test.push(t);
            h_test.push(hv);
            truth.push((k as f64 * t).sin() * (PI * hv).cos());
        }
    }

    let probe = format!("hifreq_tensor_k{k}");
    let (yhat, beta) = match fit_predict_2d(&formula, &data, &t_test, &h_test) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error(&probe, &formula, &e);
            return Category::Collapsed;
        }
    };
    check_finite(&probe, &formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let extra = format!("k={k} l1_at_truth={l1:.4} n_train={}", theta.len());
    report(&probe, &formula, r, sigma, sf, st, &extra)
}

#[test]
fn hifreq_tensor_k4() {
    let _ = hifreq_tensor_probe(4);
}
#[test]
fn hifreq_tensor_k6() {
    let _ = hifreq_tensor_probe(6);
}
#[test]
fn hifreq_tensor_k8() {
    let _ = hifreq_tensor_probe(8);
}
#[test]
fn hifreq_tensor_k10() {
    let _ = hifreq_tensor_probe(10);
}

// =====================================================================
// Probe 2: Bimodal / sharp signal
// =====================================================================

fn bump_pair(x: f64) -> f64 {
    let a = ((x - 0.25) / 0.05).powi(2);
    let b = ((x - 0.75) / 0.05).powi(2);
    (-a).exp() - (-b).exp()
}

#[test]
fn bimodal_sharp_bumps_bc() {
    init_parallelism();
    let n = 500;
    let sigma = 0.05;
    let mut rng = Lcg::new(101);
    let x: Vec<f64> = (0..n).map(|_| rng.uniform_01()).collect();
    let y_truth: Vec<f64> = x.iter().map(|&t| bump_pair(t)).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=40)";

    let mgrid = 500;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.001 + 0.998 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = x_grid.iter().map(|&t| bump_pair(t)).collect();
    let (yhat, beta) = match fit_predict_1d(formula, &data, &x_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("bimodal_sharp_bumps", formula, &e);
            return;
        }
    };
    check_finite("bimodal_sharp_bumps", formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let _ = report(
        "bimodal_sharp_bumps",
        formula,
        r,
        sigma,
        sf,
        st,
        &format!("l1_at_truth={l1:.4}"),
    );
}

// =====================================================================
// Probe 3: Near-flat signal (vanishing variance)
// =====================================================================

#[test]
fn near_flat_signal() {
    init_parallelism();
    let n = 300;
    let sigma = 0.02;
    let mut rng = Lcg::new(202);
    let x: Vec<f64> = (0..n).map(|_| rng.uniform_01()).collect();
    let truth_const = 3.5_f64;
    let y_truth: Vec<f64> = vec![truth_const; n];
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=15)";

    let mgrid = 400;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.002 + 0.996 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = vec![truth_const; mgrid];
    let (yhat, beta) = match fit_predict_1d(formula, &data, &x_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("near_flat_signal", formula, &e);
            return;
        }
    };
    check_finite("near_flat_signal", formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    // For a constant truth, span_truth = 0; substitute the noise sigma as
    // the "trivial baseline span" so the ratio reflects over-fitting noise.
    let st = sigma; // a smoother should NOT exceed a few σ in fitted span
    let extra = format!("max_abs_dev_from_truth={:.4}", {
        yhat.iter()
            .map(|v| (v - truth_const).abs())
            .fold(0.0_f64, f64::max)
    });
    let _ = report("near_flat_signal", formula, r, sigma, sf, st, &extra);
}

// =====================================================================
// Probe 4: Heteroscedastic noise
// =====================================================================

#[test]
fn heteroscedastic_noise_mean_recovery() {
    init_parallelism();
    let n = 600;
    let mut rng = Lcg::new(303);
    let x: Vec<f64> = (0..n).map(|_| rng.uniform_01()).collect();
    let y_truth: Vec<f64> = x
        .iter()
        .map(|&t| (PI * t).sin() + 0.5 * (2.0 * PI * t).cos())
        .collect();
    // Variance grows linearly with x (5x at right edge vs left).
    let y_noisy: Vec<f64> = x
        .iter()
        .zip(y_truth.iter())
        .map(|(&xi, &yt)| {
            let sigma_x = 0.05 + 0.25 * xi; // [0.05, 0.30]
            yt + sigma_x * rng.normal()
        })
        .collect();
    let avg_sigma = 0.175_f64;
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=15)";

    let mgrid = 400;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.002 + 0.996 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = x_grid
        .iter()
        .map(|&t| (PI * t).sin() + 0.5 * (2.0 * PI * t).cos())
        .collect();
    let (yhat, beta) = match fit_predict_1d(formula, &data, &x_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("heteroscedastic", formula, &e);
            return;
        }
    };
    check_finite("heteroscedastic", formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let _ = report(
        "heteroscedastic",
        formula,
        r,
        avg_sigma,
        sf,
        st,
        &format!("l1_at_truth={l1:.4} sigma_range=[0.05,0.30]"),
    );
}

// =====================================================================
// Probe 5: Outliers (1% at y ± 10 σ)
// =====================================================================

#[test]
fn outlier_contamination() {
    init_parallelism();
    let n = 500;
    let sigma = 0.10;
    let mut rng = Lcg::new(404);
    let x: Vec<f64> = (0..n).map(|_| rng.uniform_01()).collect();
    let y_truth: Vec<f64> = x
        .iter()
        .map(|&t| (PI * t).sin() + 0.5 * (3.0 * PI * t).cos())
        .collect();
    let mut y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    // Plant 1% outliers at ±10 σ (alternating sign).
    let n_out = ((n as f64) * 0.01).ceil() as usize;
    for k in 0..n_out {
        let idx = (rng.next_u64() as usize) % n;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        y_noisy[idx] = y_truth[idx] + sign * 10.0 * sigma;
    }
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=15)";

    let mgrid = 400;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.002 + 0.996 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = x_grid
        .iter()
        .map(|&t| (PI * t).sin() + 0.5 * (3.0 * PI * t).cos())
        .collect();
    let (yhat, beta) = match fit_predict_1d(formula, &data, &x_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("outlier_contamination", formula, &e);
            return;
        }
    };
    check_finite("outlier_contamination", formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    let l1 = truth_residual(&yhat, &truth);
    let _ = report(
        "outlier_contamination",
        formula,
        r,
        sigma,
        sf,
        st,
        &format!("l1_at_truth={l1:.4} n_outliers={n_out}"),
    );
}

// =====================================================================
// Probe 6: Sparse-dense imbalance
// =====================================================================

#[test]
fn sparse_dense_imbalance() {
    init_parallelism();
    let sigma = 0.05;
    let n_sparse = 20;
    let n_dense = 2000;
    let mut rng = Lcg::new(505);
    let mut x = Vec::with_capacity(n_sparse + n_dense);
    for _ in 0..n_sparse {
        x.push(0.5 * rng.uniform_01());
    }
    for _ in 0..n_dense {
        x.push(0.5 + 0.5 * rng.uniform_01());
    }
    let f = |t: f64| (2.0 * PI * t).sin() + 0.3 * t;
    let y_truth: Vec<f64> = x.iter().map(|&t| f(t)).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=20)";

    // Test on sparse side only
    let xg_sparse: Vec<f64> = (0..100).map(|i| 0.005 + 0.49 * i as f64 / 99.0).collect();
    let truth_sparse: Vec<f64> = xg_sparse.iter().map(|&t| f(t)).collect();
    // Test on dense side only
    let xg_dense: Vec<f64> = (0..100).map(|i| 0.505 + 0.49 * i as f64 / 99.0).collect();
    let truth_dense: Vec<f64> = xg_dense.iter().map(|&t| f(t)).collect();

    let all_x: Vec<f64> = xg_sparse.iter().chain(xg_dense.iter()).copied().collect();
    let (yhat_all, beta) = match fit_predict_1d(formula, &data, &all_x) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("sparse_dense_imbalance", formula, &e);
            return;
        }
    };
    check_finite("sparse_dense_imbalance", formula, &yhat_all, &beta);
    let (yhat_sparse, yhat_dense) = yhat_all.split_at(100);
    let r_sparse = rmse(yhat_sparse, &truth_sparse);
    let r_dense = rmse(yhat_dense, &truth_dense);
    let r_worst = r_sparse.max(r_dense);
    let sf = span(&yhat_all);
    let truth_all: Vec<f64> = truth_sparse
        .iter()
        .chain(truth_dense.iter())
        .copied()
        .collect();
    let st = span(&truth_all);
    let extra = format!(
        "rmse_sparse={r_sparse:.4} rmse_dense={r_dense:.4} rmse_worst={r_worst:.4} n_sparse={n_sparse} n_dense={n_dense}"
    );
    let _ = report(
        "sparse_dense_imbalance",
        formula,
        r_worst,
        sigma,
        sf,
        st,
        &extra,
    );
}

// =====================================================================
// Probe 7: Boundary discontinuity (Gibbs)
// =====================================================================

#[test]
fn boundary_discontinuity_step() {
    init_parallelism();
    let n = 400;
    let sigma = 0.05;
    let mut rng = Lcg::new(606);
    let x: Vec<f64> = (0..n).map(|_| rng.uniform_01()).collect();
    let f = |t: f64| if t < 0.5 { 0.0 } else { 1.0 };
    let y_truth: Vec<f64> = x.iter().map(|&t| f(t)).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=25)";

    let mgrid = 400;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.002 + 0.996 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = x_grid.iter().map(|&t| f(t)).collect();
    let (yhat, beta) = match fit_predict_1d(formula, &data, &x_grid) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("step_discontinuity", formula, &e);
            return;
        }
    };
    check_finite("step_discontinuity", formula, &yhat, &beta);
    let r = rmse(&yhat, &truth);
    let sf = span(&yhat);
    let st = span(&truth);
    // Estimate Gibbs overshoot: max prediction above 1 or below 0.
    let over = yhat
        .iter()
        .map(|&v| (v - 1.0).max(0.0).max((-v).max(0.0)))
        .fold(0.0_f64, f64::max);
    let _ = report(
        "step_discontinuity",
        formula,
        r,
        sigma,
        sf,
        st,
        &format!("gibbs_overshoot={over:.3}"),
    );
}

// =====================================================================
// Probe 8: Multicollinear input in a tensor smooth
// =====================================================================

#[test]
fn tensor_multicollinear_inputs() {
    init_parallelism();
    let n = 500;
    let sigma = 0.10;
    let mut rng = Lcg::new(707);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.uniform_01();
        a.push(u);
        // b = 0.95 a + 0.05 noise — strongly correlated
        b.push(0.95 * u + 0.05 * rng.uniform_01());
    }
    let f = |aa: f64, bb: f64| (PI * aa).sin() + 0.5 * bb;
    let y_truth: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_2d_named("a", &a, "b", &b, &y_noisy);
    let formula = "y ~ te(a, b, k=6)";

    // Test on the support manifold (b ≈ 0.95 a) — extrapolating off it
    // is unfair.
    let g: Vec<f64> = (0..60).map(|i| 0.02 + 0.96 * i as f64 / 59.0).collect();
    let a_test: Vec<f64> = g.clone();
    let b_test: Vec<f64> = g.iter().map(|&u| 0.95 * u + 0.025).collect();
    let truth: Vec<f64> = a_test
        .iter()
        .zip(b_test.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();

    let res = fit_predict_2d(formula, &data, &a_test, &b_test);
    match res {
        Ok((yhat, beta)) => {
            check_finite("multicollinear_tensor", formula, &yhat, &beta);
            let r = rmse(&yhat, &truth);
            let sf = span(&yhat);
            let st = span(&truth);
            let l1 = truth_residual(&yhat, &truth);
            let _ = report(
                "multicollinear_tensor",
                formula,
                r,
                sigma,
                sf,
                st,
                &format!("l1_at_truth={l1:.4} corr_ab~0.95"),
            );
        }
        Err(e) => {
            // Some smooths refuse to build on near-singular tensor inputs.
            // That's an honest "fail-fast" outcome; record it.
            eprintln!(
                "[fit-quality] probe=multicollinear_tensor category=BUILD_REFUSED \
                 formula=`{formula}` err=`{e}`",
            );
        }
    }
}

// =====================================================================
// Probe 9: Wrong period for cyclic
// =====================================================================

#[test]
fn cyclic_wrong_period() {
    init_parallelism();
    let n = 300;
    let sigma = 0.05;
    let mut rng = Lcg::new(808);
    // True data is on [0, 2π]
    let theta: Vec<f64> = (0..n).map(|_| TAU * rng.uniform_01()).collect();
    let y_truth: Vec<f64> = theta.iter().map(|t| (2.0 * t).sin()).collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_named_1d("theta", &theta, &y_noisy);
    // Misconfiguration: declare period = π even though data spans [0, 2π].
    let formula = "y ~ cyclic(theta, k=10, period_start=0, period_end=3.141592653589793)";

    let mgrid = 400;
    let theta_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.005 + (TAU - 0.01) * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = theta_grid.iter().map(|t| (2.0 * t).sin()).collect();

    // The fit is allowed to either error out (preferred) or to succeed
    // with a degraded recovery (which we then record).
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let res = fit_from_formula(formula, &data, &cfg);
    match res {
        Err(e) => {
            eprintln!(
                "[fit-quality] probe=cyclic_wrong_period category=ERROR_RAISED \
                 formula=`{formula}` err=`{e}`",
            );
        }
        Ok(result) => {
            let FitResult::Standard(fit) = result else {
                panic!("expected standard fit for cyclic wrong-period probe");
            };
            let mut m = Array2::<f64>::zeros((theta_grid.len(), 2));
            for i in 0..theta_grid.len() {
                m[[i, 0]] = theta_grid[i];
                m[[i, 1]] = 0.0;
            }
            let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
                .expect("rebuild design (wrong period)");
            let yhat = test_design.design.apply(&fit.fit.beta).to_vec();
            let beta = fit.fit.beta.to_vec();
            check_finite("cyclic_wrong_period", formula, &yhat, &beta);
            let r = rmse(&yhat, &truth);
            let sf = span(&yhat);
            let st = span(&truth);
            // Headline category here is what the fit did despite the
            // wrong period — usually a forced wraparound that ruins
            // the recovery.
            let _ = report(
                "cyclic_wrong_period",
                formula,
                r,
                sigma,
                sf,
                st,
                "note=fit_accepted_wrong_period",
            );
        }
    }
}

// =====================================================================
// Probe 10: Antipodal sphere data (poles only)
// =====================================================================

#[test]
fn sphere_antipodal_only() {
    init_parallelism();
    let n_per_pole = 80;
    let sigma = 0.05;
    let mut rng = Lcg::new(909);
    let mut lat = Vec::with_capacity(2 * n_per_pole);
    let mut lon = Vec::with_capacity(2 * n_per_pole);
    let mut y_truth = Vec::with_capacity(2 * n_per_pole);
    // North polar cap: lat in [75°, 89°]
    for _ in 0..n_per_pole {
        lat.push(75.0 + 14.0 * rng.uniform_01());
        lon.push(-180.0 + 360.0 * rng.uniform_01());
        y_truth.push(1.0);
    }
    // South polar cap: lat in [-89°, -75°]
    for _ in 0..n_per_pole {
        lat.push(-89.0 + 14.0 * rng.uniform_01());
        lon.push(-180.0 + 360.0 * rng.uniform_01());
        y_truth.push(-1.0);
    }
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_2d_named("lat", &lat, "lon", &lon, &y_noisy);
    let formula = "y ~ sphere(lat, lon, method=harmonic, max_degree=4)";

    // Predict on the same caps to assess recovery; also predict at the
    // equator to check that the smoother gives a sensible interpolation
    // (no NaNs, no wild blow-up).
    let mut lat_test = Vec::new();
    let mut lon_test = Vec::new();
    let mut truth_test = Vec::new();
    for i in 0..60 {
        let lt = 78.0 + 10.0 * i as f64 / 59.0;
        for j in 0..6 {
            let ln = -180.0 + 60.0 * j as f64;
            lat_test.push(lt);
            lon_test.push(ln);
            truth_test.push(1.0);
        }
    }
    for i in 0..60 {
        let lt = -88.0 + 10.0 * i as f64 / 59.0;
        for j in 0..6 {
            let ln = -180.0 + 60.0 * j as f64;
            lat_test.push(lt);
            lon_test.push(ln);
            truth_test.push(-1.0);
        }
    }
    // Equator probes — truth unknown; we just verify finiteness and report
    // the magnitude as `extra`.
    let n_polar = lat_test.len();
    for j in 0..36 {
        lat_test.push(0.0);
        lon_test.push(-180.0 + 10.0 * j as f64);
        truth_test.push(0.0); // sentinel — unused for rmse calc
    }

    let (yhat, beta) = match fit_predict_2d(formula, &data, &lat_test, &lon_test) {
        Ok(v) => v,
        Err(e) => {
            report_fit_error("antipodal_sphere", formula, &e);
            return;
        }
    };
    check_finite("antipodal_sphere", formula, &yhat, &beta);
    let (yhat_polar, yhat_equator) = yhat.split_at(n_polar);
    let truth_polar = &truth_test[..n_polar];
    let r = rmse(yhat_polar, truth_polar);
    let sf = span(yhat_polar);
    let st = span(truth_polar);
    let eq_max = yhat_equator.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let _ = report(
        "antipodal_sphere",
        formula,
        r,
        sigma,
        sf,
        st,
        &format!("equator_max_abs={eq_max:.4} n_polar={n_polar}"),
    );
}
