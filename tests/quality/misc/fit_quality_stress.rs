//! Adversarial fit-quality stress probes across the geometric smooth families.
//!
//! Each probe stresses one smooth with a known-hard signal (high frequency,
//! sharp bumps, discontinuities, heteroscedastic noise, outliers, ...), fits it
//! via `gam::fit_from_formula`, and asserts, in this order:
//!
//! 1. the fit and the prediction-design rebuild succeed (an error panics);
//! 2. every prediction and coefficient is finite;
//! 3. the fit's own Bayesian band covers its error across the function
//!    (`assess`): with `f*` the in-span target (the unpenalized least-squares
//!    projection of the truth onto the fit's own training design) and `Vb` the
//!    fit's Bayesian coefficient covariance, the coverage statistic
//!    `Q = (1/P) Σ (f̂_i − f*_i)² / s_i²` must lie below its Satterthwaite
//!    `χ²` bound at the per-probe level `FAMILY_ALPHA / GATED_PROBES`.
//!
//! When the truth is in the basis, `f* = f` and gate 3 is exactly the
//! across-the-function coverage test. When it is not (a step, a truth that
//! breaks an anchored endpoint pin, a frequency the basis only approximates),
//! `f*` is the best function this basis can express on these rows, so the gate
//! charges the fit only for what it could have recovered. Each probe prints one
//! `[fit-quality]` line with `Q`, its bound and the RMSE split into the
//! approximation floor `‖f* − f‖` and the in-span error `‖f̂ − f*‖`.
//!
//! Deterministic LCG seeding is used throughout (no rand crate dependency on
//! the noise streams) so reruns and CI are bit-identical.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{across_function_coverage, least_squares_projection, rmse};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, s};
use std::ops::Range;

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

// ---------- the derived gate ----------

/// Family-wise false-alarm rate of this file's coverage gates: with every smooth
/// correct, the chance that any gated probe fails is at most this.
const FAMILY_ALPHA: f64 = 0.01;

/// Number of `#[test]` probes gated by [`assess`]: cyclic k4-k10 (4), anchored
/// k4-k10 (4), sphere l4-l8 (3), tensor k4-k10 (4), and the nine single probes.
/// Bonferroni splits [`FAMILY_ALPHA`] evenly across them.
const GATED_PROBES: f64 = 24.0;

/// A converged fit together with everything [`assess`] needs: its predictions
/// on the probe rows, its coefficients, the dense training and probe designs
/// (rebuilt from the frozen spec, so they are exactly the design the fit used)
/// with their fixed affine offsets, and the Bayesian covariance `Vb`.
struct ProbeFit {
    yhat: Vec<f64>,
    beta: Vec<f64>,
    x_train: Array2<f64>,
    offset_train: Vec<f64>,
    x_probe: Array2<f64>,
    offset_probe: Vec<f64>,
    cov: Array2<f64>,
}

fn span(v: &[f64]) -> f64 {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
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

/// A fit or its prediction-design rebuild failed. Every probe here is a
/// well-defined penalized regression, so a refusal is a failure of the probe.
fn fit_error(probe: &str, formula: &str, err: &str) -> ! {
    panic!("[fit-quality] probe={probe} verdict=FIT_FAILED formula=`{formula}` err=`{err}`");
}

/// Gate one probe on the rows `gated` of its probe design.
///
/// `truth` is the true function on those rows and `truth_train` the true
/// function on the training rows. The in-span target is `f* = X_p β*` with
/// `β* = X_train⁺ (f_train − o_train)` (plus the probe rows' fixed offset): the
/// unpenalized least-squares projection of the truth onto the fit's own design,
/// which is what the penalized fit estimates when the truth is not exactly in
/// the basis. The fit passes iff its Bayesian band covers `f̂ − f*` across the
/// function at level `FAMILY_ALPHA / GATED_PROBES`
/// (`gam::test_support::reference::across_function_coverage`).
fn assess(
    probe: &str,
    formula: &str,
    fit: &ProbeFit,
    gated: Range<usize>,
    truth: &[f64],
    truth_train: &[f64],
    extra: &str,
) {
    check_finite(probe, formula, &fit.yhat, &fit.beta);
    assert_eq!(truth.len(), gated.len(), "{probe}: truth rows vs gated rows");
    let target_train: Vec<f64> = truth_train
        .iter()
        .zip(&fit.offset_train)
        .map(|(f, o)| f - o)
        .collect();
    let beta_star = least_squares_projection(&fit.x_train, &target_train)
        .unwrap_or_else(|e| panic!("{probe}: in-span projection failed: {e}"));
    let x_gate = fit.x_probe.slice(s![gated.clone(), ..]).to_owned();
    let f_star: Vec<f64> = x_gate
        .dot(&beta_star)
        .iter()
        .zip(&fit.offset_probe[gated.clone()])
        .map(|(v, o)| v + o)
        .collect();
    let yhat = &fit.yhat[gated];
    let err: Vec<f64> = yhat.iter().zip(&f_star).map(|(a, b)| a - b).collect();
    let alpha = FAMILY_ALPHA / GATED_PROBES;
    let cov = across_function_coverage(&x_gate, &fit.cov, &err, alpha)
        .unwrap_or_else(|e| panic!("{probe}: coverage statistic failed: {e}"));
    let rmse_truth = rmse(yhat, truth);
    let floor = rmse(&f_star, truth);
    let in_span = rmse(yhat, &f_star);
    let span_truth = span(truth);
    let span_ratio = if span_truth > 0.0 {
        span(yhat) / span_truth
    } else {
        f64::NAN
    };
    let verdict = if cov.covers() { "PASS" } else { "FAIL" };
    eprintln!(
        "[fit-quality] probe={probe} verdict={verdict} Q={q:.4} bound={bound:.4} dof={dof:.2} \
         alpha={alpha:.2e} rmse={rmse_truth:.4} floor={floor:.4} in_span_rmse={in_span:.4} \
         span_ratio={span_ratio:.3}{ws}{extra} formula=`{formula}`",
        q = cov.statistic,
        bound = cov.bound,
        dof = cov.dof,
        ws = if extra.is_empty() { "" } else { " " },
    );
    assert!(
        cov.covers(),
        "[fit-quality] probe {probe}: the fit's Bayesian band does not cover its in-span error \
         (Q = {q:.4} > bound {bound:.4} at alpha {alpha:.2e}; in-span rmse {in_span:.4}, \
         approximation floor {floor:.4})",
        q = cov.statistic,
        bound = cov.bound,
    );
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

/// Covariate matrix for `build_term_collection_design`: the given covariate
/// columns in dataset order, then the response column (unused by any term).
fn covariate_matrix(cols: &[&[f64]]) -> Array2<f64> {
    let n = cols[0].len();
    let mut m = Array2::<f64>::zeros((n, cols.len() + 1));
    for (j, col) in cols.iter().enumerate() {
        assert_eq!(col.len(), n, "covariate column {j} length");
        for (i, &v) in col.iter().enumerate() {
            m[[i, j]] = v;
        }
    }
    m
}

fn fit_probe(
    formula: &str,
    data: &gam::data::EncodedDataset,
    train_cols: &[&[f64]],
    probe_cols: &[&[f64]],
) -> Result<ProbeFit, String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg)?;
    let FitResult::Standard(fit) = result else {
        return Err("expected standard fit".to_string());
    };
    let rebuild = |cols: &[&[f64]]| {
        build_term_collection_design(covariate_matrix(cols).view(), &fit.resolvedspec)
            .map_err(|e| format!("rebuild design failed: {e:?}"))
    };
    let train = rebuild(train_cols)?;
    let probe = rebuild(probe_cols)?;
    let x_train = train.design.to_dense();
    let x_probe = probe.design.to_dense();
    let yhat: Vec<f64> = x_probe
        .dot(&fit.fit.beta)
        .iter()
        .zip(probe.affine_offset.iter())
        .map(|(v, o)| v + o)
        .collect();
    let cov = fit
        .fit
        .beta_covariance()
        .ok_or_else(|| "fit reports no Bayesian coefficient covariance".to_string())?
        .clone();
    Ok(ProbeFit {
        yhat,
        beta: fit.fit.beta.to_vec(),
        x_train,
        offset_train: train.affine_offset.to_vec(),
        x_probe,
        offset_probe: probe.affine_offset.to_vec(),
        cov,
    })
}

fn fit_predict_1d(
    formula: &str,
    data: &gam::data::EncodedDataset,
    x_train: &[f64],
    x_grid: &[f64],
) -> Result<ProbeFit, String> {
    fit_probe(formula, data, &[x_train], &[x_grid])
}

fn fit_predict_2d(
    formula: &str,
    data: &gam::data::EncodedDataset,
    (a_train, b_train): (&[f64], &[f64]),
    (a_grid, b_grid): (&[f64], &[f64]),
) -> Result<ProbeFit, String> {
    fit_probe(formula, data, &[a_train, b_train], &[a_grid, b_grid])
}

// =====================================================================
// Probe 1: high-frequency truths sin(2π k x), k in {4, 6, 8, 10}
// =====================================================================

fn hifreq_cyclic_probe(k: usize) {
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
    let fit = fit_predict_1d(&formula, &data, &theta, &theta_grid)
        .unwrap_or_else(|e| fit_error(&probe, &formula, &e));
    let extra = format!("k={k} sigma_noise={sigma:.3}");
    assess(&probe, &formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
}

#[test]
fn hifreq_cyclic_k4() {
    hifreq_cyclic_probe(4)
}
#[test]
fn hifreq_cyclic_k6() {
    hifreq_cyclic_probe(6)
}
#[test]
fn hifreq_cyclic_k8() {
    hifreq_cyclic_probe(8)
}
#[test]
fn hifreq_cyclic_k10() {
    hifreq_cyclic_probe(10)
}

fn hifreq_bc_probe(k: usize) {
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

    // `bc=anchored` pins f = 0 and f' = 0 at both ends, while sin(2πkx) has
    // f'(0) = 2πk: the truth is not in the basis, and `assess` gates against
    // its in-span projection rather than against the unrepresentable slope.
    let probe = format!("hifreq_bc_k{k}");
    let fit = fit_predict_1d(&formula, &data, &x, &x_grid)
        .unwrap_or_else(|e| fit_error(&probe, &formula, &e));
    let extra = format!("k={k} sigma_noise={sigma:.3}");
    assess(&probe, &formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
}

#[test]
fn hifreq_bc_k4() {
    hifreq_bc_probe(4)
}
#[test]
fn hifreq_bc_k6() {
    hifreq_bc_probe(6)
}
#[test]
fn hifreq_bc_k8() {
    hifreq_bc_probe(8)
}
#[test]
fn hifreq_bc_k10() {
    hifreq_bc_probe(10)
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

fn hifreq_sphere_probe(l: usize) {
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
    let fit = fit_predict_2d(
        &formula,
        &data,
        (&lat_deg, &lon_deg),
        (&lat_test, &lon_test),
    )
    .unwrap_or_else(|e| fit_error(&probe, &formula, &e));
    let extra = format!("l={l} sigma_noise={sigma:.3}");
    let rows = 0..lat_test.len();
    assess(&probe, &formula, &fit, rows, &truth, &y_truth, &extra);
}

#[test]
fn hifreq_sphere_l4() {
    hifreq_sphere_probe(4)
}
#[test]
fn hifreq_sphere_l6() {
    hifreq_sphere_probe(6)
}
#[test]
fn hifreq_sphere_l8() {
    hifreq_sphere_probe(8)
}

/// Deterministic 2-D high-frequency tensor fixture shared by the k-arms and the
/// zz_measure λ-readout sibling: y = sin(k·θ)·cos(π·h) on a 24×24 grid, σ=0.10,
/// fit with te(theta[periodic], h[natural]), kb=(2k+4).max(10) per margin.
struct TensorFixture {
    data: gam::data::EncodedDataset,
    formula: String,
    theta: Vec<f64>,
    h: Vec<f64>,
    y_truth: Vec<f64>,
    sigma: f64,
}

/// Tensor-margin basis size for the `hifreq_tensor` family. Hoisted so the grid
/// can size itself against the basis (#2607) instead of the two drifting apart.
fn kb_for(k: usize) -> usize {
    (2 * k + 4).max(10)
}

fn hifreq_tensor_dataset(k: usize) -> TensorFixture {
    // #2607: the grid must stay ahead of the basis, or the design SATURATES and
    // the arm stops measuring what its siblings measure.
    //
    // `kb = (2k + 4).max(10)` and the tensor is `kb x kb`, so `p` grows with `k`
    // while the grid did not:
    //
    //   k =  4  kb = 12  p = 144  p/n = 0.25
    //   k =  6  kb = 16  p = 256  p/n = 0.44
    //   k =  8  kb = 20  p = 400  p/n = 0.69
    //   k = 10  kb = 24  p = 576  p/n = 1.00   <- p = n EXACTLY
    //
    // At `p = n` the model interpolates, REML has no residual degrees of freedom
    // to estimate a scale from, and maximum smoothing is a legitimate optimum OF
    // THE CRITERION -- measured, the rho ceiling scores 415 nats below the
    // neutral origin. The k10 arm was varying TWO things where its siblings vary
    // one (frequency AND saturation), which is also why its `edf` fell from the
    // 227.938 the issue records to 1.294: the ladder walked off the end of its own
    // grid rather than regressing.
    //
    // Grow the grid only when `kb` would push `p/n` past 0.75, so k4/k6/k8 are
    // byte-identical and only the k10 arm moves (to `p/n = 0.5625`, between the
    // k6 and k8 arms). The saturated case is worth testing -- but deliberately,
    // under its own name, not as the top rung of a frequency ladder.
    let side = if kb_for(k) * kb_for(k) * 100 > 24 * 24 * 75 {
        32
    } else {
        24
    };
    let n_theta = side;
    let n_h = side;
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
    let kb = kb_for(k);
    let formula =
        format!("y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k={kb})");
    TensorFixture {
        data,
        formula,
        theta,
        h,
        y_truth,
        sigma,
    }
}

fn hifreq_tensor_probe(k: usize) {
    init_parallelism();
    let fx = hifreq_tensor_dataset(k);
    let formula = &fx.formula;

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
    let fit = fit_predict_2d(formula, &fx.data, (&fx.theta, &fx.h), (&t_test, &h_test))
        .unwrap_or_else(|e| fit_error(&probe, formula, &e));
    let extra = format!(
        "k={k} sigma_noise={:.3} n_train={}",
        fx.sigma,
        fx.theta.len()
    );
    let rows = 0..t_test.len();
    assess(&probe, formula, &fit, rows, &truth, &fx.y_truth, &extra);
}

#[test]
fn hifreq_tensor_k4() {
    hifreq_tensor_probe(4)
}
#[test]
fn hifreq_tensor_k6() {
    hifreq_tensor_probe(6)
}
// gam#1082/#2585: hifreq_tensor_k8/k10 run LONGER than the default per-test
// slow-timeout (300s notice / 600s SIGKILL). They are NOT `#[ignore]`d — the
// high-frequency recovery they verify is real coverage we keep — they are given
// a dedicated, generous `slow-timeout` override in `.config/nextest.toml`
// (filter `test(/hifreq_tensor_k(8|10)/)`) so the nightly CI runs them to
// completion and asserts the recovery instead of bulk-killing them at 600s.
//
// THE COST IS A PERF BUG, AND THIS NOTE USED TO SAY OTHERWISE. It claimed the
// dominant cost is the dense O(p³) Cholesky of `XᵀWX + S_λ` and that this is
// "genuinely irreducible". The k8 sibling refutes it in two lines:
//
//   kb = 2k+4, p = kb²   ->   k8: p = 400,  k10: p = 576
//   a p³ law predicts    ->   k10 = 2.99 x k8 = 218 s
//   measured (CI)        ->   k10 = 6310 s = 86 x k8's 73 s
//
// 29x above the cube-law prediction; the implied exponent is ~12.
//
// WHERE IT ACTUALLY GOES — measured at k10 rather than extrapolated to it. An
// earlier revision of this note (mine) inferred from k = 4/6/8 that the outer
// EVALUATION COUNT must be exploding, and put a figure of ~4300 evaluations on
// it. That was wrong, and running k10 says so:
//
//   k    p     wall        outer evals    ms/eval
//   8    400     132.2 s          180        734
//   10   576    9452.3 s          191     49,489
//
// The evaluation count is FLAT (180 -> 191). It is the per-evaluation cost that
// jumps 67x, for p rising only 1.44x. So the extrapolation had the two factors
// exactly backwards, and the honest reading of k = 4/6/8 alone is that it does
// not determine k10.
//
// Attributing every second of that run to the last non-heartbeat log line
// preceding it puts **96.5% of the 9451 s after an `[HGB]` line**, and the
// instrumented `[STAGE]` brackets together account for about 185 s — 2%. The
// `[STAGE] outer eval start/end order` pair is 111 evaluations at 1.67 s each,
// so ~98% of this fit elapses OUTSIDE any bracketed outer evaluation. Neither
// the `O(n p²)` assembly nor the `O(p³)` factorization is the cost.
//
// `[HGB]` names the REGION, not the consumer, and the distinction matters. It
// was emitted by a BUDGET-SETTING routine (`reml/gradient_hessian.rs`, deleted
// with its hand-picked constants in #2469): it computed `k_target` and wrote `solve_rel_tol_override` /
// `monotone_probe_floor` into `trace_state`, then returns. It does no heavy
// work, and its own `k=` field — the requested stochastic-trace probe floor —
// reads **`k=0` in every run at every k** (4, 6, 8, 10). So what consumes the
// time is whatever runs after that line and before the next log line of any
// kind, and it is UNINSTRUMENTED. This attribution localises the cost to a
// region; it does not identify a component, and an earlier revision of this
// note (also mine) overstated it by naming one.
//
// The leading candidate is still the stochastic trace path, because that is
// what `[HGB]` configures, and #2576 measures the same estimator at 95% of
// wall clock in the overcomplete support lane. That is an inference from what
// the budget routine writes, not from where time was attributed — a hypothesis
// to test, not a finding. The next measurement is a `perf` profile of the k10
// fit, or a `[STAGE]` bracket around the un-logged region.
//
// EVERY k10 FIGURE ABOVE WAS MEASURED ON A DESIGN THAT NO LONGER EXISTS, and
// that is the first thing to know before reusing any of them. They were taken
// when the grid was a fixed 24 x 24, so `n = 576`, and `kb = 24` at k10 gave
// `p = kb² = 576` — **p = n exactly**, a saturated design with zero residual
// degrees of freedom, whose REML optimum is maximum smoothing (#2607). k8 was
// `p = 400` against the same `n = 576`, i.e. a perfectly ordinary problem, and
// cost 73 s; one step of `k` crossed from penalized regression into saturated
// interpolation, and the 86x is that crossing as much as it is any component's
// complexity. `hifreq_tensor_dataset` now sizes the grid against `kb` (the k10
// arm runs on 32 x 32, `p/n = 0.5625`), so the k10 cost has to be re-measured
// before it means anything. The k8 numbers stand: that arm is byte-identical.
//
// What does NOT depend on the saturation is the shape of the cost. The PENALTY
// side exploited the tensor's Kronecker structure through a Kronecker
// reparameterization, which a88c62eee removed with the Kronecker runtime; the
// DATA Gram `XᵀWX` never inherits that structure under a general PIRLS weight W,
// so its factorization is a true dense p×p one.
// And `kb` cannot be capped to shrink p: the ground truth is `sin(k·θ)`, whose
// periodic marginal needs ≥ k Fourier modes, so k8/k10 require kb ≳ 18. Growing
// the grid, not capping the basis, is what keeps `p/n` bounded — which is what
// the fixture now does.
#[test]
fn hifreq_tensor_k8() {
    hifreq_tensor_probe(8)
}
#[test]
fn hifreq_tensor_k10() {
    hifreq_tensor_probe(10)
}

// zz_measure DIAGNOSTIC (#1082/#2392): why does `hifreq_tensor_k8` collapse to
// flat (span 0.002)? It MINTS a fit (unlike k10, which refuses at an indefinite
// railed saddle), so REML genuinely converges to (near-)flat. Two candidate roots
// are indistinguishable from the [fit-quality] line (it carries no λ/edf):
//   (a) GENUINE single-λ high-frequency under-selection — the periodic marginal
//       has ONE λ against a modal roughness penalty ∝ m⁴, so the signal-carrying
//       8th Fourier mode costs ~8⁴≈4096× the fundamental; a single λ cannot admit
//       it without under-penalizing the (noise-only) low modes, so the REML
//       evidence shrinks the signal to the penalty null. Interior ρ + collapsed
//       edf is the signature.
//   (b) RAIL-TO-NULL — the double_penalty null-space λ railed at the ρ ceiling
//       (~30) like k10, but with a PSD Hessian that certified the flat optimum.
//       A ρ pinned at the box ceiling with collapsed edf is the signature (== the
//       #2392 family).
// This prints the converged fit's log_lambdas / edf_by_block / edf_total / reml so
// the reader can read the discriminator directly. zz_measure discipline: numbers
// eprintln'd; a refused or non-standard fit panics instead of returning green (SPEC 16).
// The name contains `hifreq_tensor_k8` so it inherits the dedicated slow-timeout
// override in `.config/nextest.toml` (the k8 fit is minutes-long, p=kb²=400).
#[test]
fn zz_measure_hifreq_tensor_k8_lambda_readout() {
    init_parallelism();
    let fx = hifreq_tensor_dataset(8);
    let (n_train, sigma) = (fx.theta.len(), fx.sigma);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(&fx.formula, &fx.data, &cfg) {
        Ok(r) => r,
        Err(e) => panic!("[zz:hifreq_k8] fit refused (no minted optimum): {e}"),
    };
    let FitResult::Standard(fit) = result else {
        panic!("[zz:hifreq_k8] unexpected non-standard fit result");
    };
    let log_lambdas: Vec<f64> = fit
        .fit
        .log_lambdas
        .iter()
        .map(|v| (v * 1000.0).round() / 1000.0)
        .collect();
    let edf_by_block: Vec<f64> = fit
        .fit
        .edf_by_block()
        .iter()
        .map(|v| (v * 1000.0).round() / 1000.0)
        .collect();
    eprintln!(
        "[zz:hifreq_k8] n_train={n_train} sigma_noise={sigma:.3} p_coef={} \
         edf_total={:.3} edf_by_block={edf_by_block:?} log_lambdas={log_lambdas:?} \
         reml={:.4} outer_iters={} | discriminator: all-interior ρ (box ceiling ~30) + \
         collapsed edf ⇒ (a) single-λ high-freq under-selection; a ρ pinned at ~30 with \
         collapsed edf ⇒ (b) rail-to-null / #2392 family",
        fit.fit.beta.len(),
        fit.fit.edf_total().unwrap_or(f64::NAN),
        fit.fit.reml_score().expect("the fit reports a REML/LAML criterion"),
        fit.fit.outer_iterations,
    );
}

// #2607: `hifreq_tensor_k10` converges in ONE outer iteration at edf ~ 1.294 of
// p = 576 -- the intercept -- because the seed prepass hands the optimizer the rho
// ceiling in every coordinate. The prepass is not a guess that can be "wrong": it
// keeps the STRICTLY-CHEAPEST of {base, initial_sp, summed_diagonal} under
// `compute_cost`, so the logged `[0,0,0] -> [30,30,30]` already proves
// `cost(wall) < cost(origin)` on this fixture.
//
// That leaves two readings, which demand completely different fixes:
//   (1) `compute_cost` degenerates at the boundary -- the pseudo-logdet / rank
//       handling stops being comparable to interior points as lambda -> e^30; or
//   (2) it is right, and the criterion genuinely cannot see the high-frequency
//       signal in this design. Truth here is y = sin(10*theta)*cos(pi*h), so (2)
//       would be a serious statement about the criterion, not about this fixture.
//
// `d71fb42f2` made the prepass report every candidate's cost next to the point it
// scored, but `log::debug!` needs a backend and this harness installs none, so that
// line is invisible here. This probe installs a capturing logger for exactly that
// one line. If `base` is finite and merely larger, that is reading (2); if it is
// non-finite or on an incomparable scale, that is reading (1). One run separates
// them, and until now nothing could.
//
// MEASURED, on the RESIZED design (`c0a076a71`, so `n = 1024`, `p = 576`,
// `p/n = 0.5625` -- no longer saturated). Reading (2): all three candidates are
// finite and on one scale, and `initial.sp` STILL picks the ceiling.
//
//   [STAGE] PIRLS row-chunk generation chunks=1 n=1024 p=576 nnz=589824
//   [STAGE] logdet S rho_dim=3 penalty_rank=575
//   [OUTER] standard REML initial.sp prepass candidate: [0,0,0] -> [30,30,30]
//           (scored: base=1.031664449e3 initial_sp=9.658386681e2
//            summed_diagonal=7.590703143e2; bounds -12.000..30.000)
//   [STAGE] standard REML: seed screening cascade start seeds=5 initial_cap=3
//   [STAGE] seed-screen stage=0 seed=3/5 cap=3 cost=2.070444e1   <- best of 5
//
// So un-saturating the design did NOT move this heuristic's preference, and the
// issue's premise needs one more correction on top of the saturation one: THIS
// LINE DID NOT NAME THE OPTIMIZER'S STARTING POINT. `initial.sp` contributed ONE
// candidate to a 5-seed screening cascade whose ranking picked the start (that
// cascade is gone: the outer search now enters from the one derived start,
// `rho_optimizer::run_plan::outer_start_point`). The two reported on different
// evaluators -- a `compute_cost` score against a cap-3 PIRLS cost -- so the
// numbers above are NOT comparable across that boundary and no ratio between them
// means anything. What is legitimate to conclude is narrower and still useful:
// the seed line alone never determined where the fit began, so "converged in one
// iteration because the seed is on the wall" cannot be read off it.
//
// `penalty_rank = 575` of `p = 576` also pins the joint penalty nullity at
// exactly 1 -- the intercept -- which is the `mp` the collapse check in
// `gam_solve::estimate::collapsed_to_penalty_null_space` measures `edf` against
// for this fixture.
//
// Still unmeasured: the k10 fit's converged `edf` on the resized grid. This probe
// exhausts the focused lane's 600s execution budget mid-fit, so that number needs
// the nightly slow-timeout lane rather than a dispatch.
//
// zz_measure discipline: numbers eprintln'd; a refused or non-standard fit panics
// instead of returning green (SPEC 16). The name contains
// `hifreq_tensor_k10` so it inherits the dedicated slow-timeout override in
// `.config/nextest.toml`.
struct SeedCostLogger;

static SEED_COST_LOGGER: SeedCostLogger = SeedCostLogger;

impl log::Log for SeedCostLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record<'_>) {
        let message = format!("{}", record.args());
        // Only the seed line. The rest of an `info` fit stream is six figures of
        // lines and would bury the one thing this exists to read.
        // The seed line answers WHICH point was chosen and why. The `[STAGE]`
        // lines answer where the wall-clock goes -- this fit does not finish
        // inside the focused lane's 600s run budget even after #2585's 920x on
        // the branch-and-bound, and nothing currently says what replaced it as
        // the dominant cost.
        if message.contains("initial.sp prepass candidate") || message.starts_with("[STAGE]") {
            eprintln!("[zz:2607] {message}");
        }
    }
    fn flush(&self) {}
}

#[test]
fn zz_measure_hifreq_tensor_k10_seed_costs() {
    init_parallelism();
    // Another test in this binary may have installed a logger first. Losing
    // that race is a reason to capture nothing, never to fail — but it is
    // said out loud, so an empty seed-cost log reads as "capture was off"
    // rather than "the fit produced no costs".
    if log::set_logger(&SEED_COST_LOGGER).is_err() {
        eprintln!(
            "[zz:2607] another test installed a logger first; \
             seed-cost capture is off for this run"
        );
    }
    log::set_max_level(log::LevelFilter::Debug);

    let fx = hifreq_tensor_dataset(10);
    let (n_train, sigma) = (fx.theta.len(), fx.sigma);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    eprintln!("[zz:2607] fitting hifreq_tensor k=10 (n_train={n_train} sigma={sigma:.3})");
    let result = match fit_from_formula(&fx.formula, &fx.data, &cfg) {
        Ok(r) => r,
        Err(e) => panic!("[zz:2607] fit refused (no minted optimum): {e}"),
    };
    let FitResult::Standard(fit) = result else {
        panic!("[zz:2607] unexpected non-standard fit result");
    };
    let log_lambdas: Vec<f64> = fit
        .fit
        .log_lambdas
        .iter()
        .map(|v| (v * 1000.0).round() / 1000.0)
        .collect();
    eprintln!(
        "[zz:2607] p_coef={} edf_total={:.4} log_lambdas={log_lambdas:?} outer_iters={} \
         reml={:.6e}",
        fit.fit.beta.len(),
        fit.fit.edf_total().unwrap_or(f64::NAN),
        fit.fit.outer_iterations,
        fit.fit.reml_score().unwrap_or(f64::NAN),
    );
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
    let probe = "bimodal_sharp_bumps";
    let fit = fit_predict_1d(formula, &data, &x, &x_grid)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let extra = format!("sigma_noise={sigma:.3}");
    assess(probe, formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
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
    // #1967/#2069: the probe formerly planted signal_amp == sigma (SNR ≈ 1), so a
    // fit that chased noise was indistinguishable from one that recovered the true
    // near-flat sinusoid — the over-fit/collapse verdict was muddy. Plant a signal
    // clearly above the noise floor (SNR = 0.06/0.02 ≈ 3) yet near-null in span
    // (peak-to-peak 0.12), so span_fit/span_truth cleanly separates genuine
    // recovery from a collapsed-to-flat or noise-chasing (over-fit) fit.
    //
    // The truth MUST vanish at the anchored endpoint. `bc=anchored` pins the fitted
    // function to f(0)=0 AND suppresses the global intercept — an anchored endpoint
    // is the model's level-setting gauge, so term-design construction drops the
    // intercept and applies no sum-to-zero chart (gam-terms term_builder.rs, the
    // `BSplineIdentifiability::None` branch for `has_anchor()`), and the anchored
    // I-spline additionally loses its constant direction (basis/derivative_penalty.rs).
    // A truth with a NONZERO value at x=0 is therefore structurally unrepresentable:
    // the fit is forced to ramp from the pinned 0 up to the baseline, a boundary
    // artifact unrelated to over/under-smoothing (it formerly made this probe report
    // a spurious 36× "overfit" — span_fit ≈ the forced 0→baseline ramp). So plant a
    // pure sinusoid 0.06·sin(2πx), which is 0 at x=0 and compatible with the anchor.
    let signal_amp = 0.06_f64;
    let y_truth: Vec<f64> = x
        .iter()
        .map(|&t| signal_amp * (2.0 * PI * t).sin())
        .collect();
    let y_noisy: Vec<f64> = y_truth.iter().map(|&v| v + sigma * rng.normal()).collect();
    let data = make_dataset_1d(&x, &y_noisy);
    let formula = "y ~ s(x, bc=anchored, k=15)";

    let mgrid = 400;
    let x_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.002 + 0.996 * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = x_grid
        .iter()
        .map(|&t| signal_amp * (2.0 * PI * t).sin())
        .collect();
    let probe = "near_flat_signal";
    let fit = fit_predict_1d(formula, &data, &x, &x_grid)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let max_abs_fit = fit.yhat.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let extra = format!("sigma_noise={sigma:.3} max_abs_fit={max_abs_fit:.4}");
    assess(probe, formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
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
    // The truth has f(0) = 0.5 against the anchored pin f(0) = 0, so it is not
    // in the basis; `assess` gates against its in-span projection.
    let probe = "heteroscedastic";
    let fit = fit_predict_1d(formula, &data, &x, &x_grid)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let extra = format!("sigma_noise_mean={avg_sigma:.3} sigma_range=[0.05,0.30]");
    assess(probe, formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
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
    // The truth has f(0) = 0.5 and f(1) = -0.5 against the anchored pins, so it
    // is not in the basis; `assess` gates against its in-span projection.
    let probe = "outlier_contamination";
    let fit = fit_predict_1d(formula, &data, &x, &x_grid)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let extra = format!("sigma_noise={sigma:.3} n_outliers={n_out}");
    assess(probe, formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
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
    let probe = "sparse_dense_imbalance";
    let fit = fit_predict_1d(formula, &data, &x, &all_x)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let (yhat_sparse, yhat_dense) = fit.yhat.split_at(xg_sparse.len());
    let r_sparse = rmse(yhat_sparse, &truth_sparse);
    let r_dense = rmse(yhat_dense, &truth_dense);
    let truth_all: Vec<f64> = truth_sparse
        .iter()
        .chain(truth_dense.iter())
        .copied()
        .collect();
    let extra = format!(
        "sigma_noise={sigma:.3} rmse_sparse={r_sparse:.4} rmse_dense={r_dense:.4} \
         n_sparse={n_sparse} n_dense={n_dense}"
    );
    // One gate over both halves: the band is wide where the data are sparse and
    // narrow where they are dense, and Q weighs each probe row by its own s_i.
    let rows = 0..all_x.len();
    assess(probe, formula, &fit, rows, &truth_all, &y_truth, &extra);
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
    // A step is not in any spline basis (and f(1) = 1 breaks the anchored pin):
    // the in-span projection carries the Gibbs ringing, and `assess` charges
    // the fit only for its error against that projection.
    let probe = "step_discontinuity";
    let fit = fit_predict_1d(formula, &data, &x, &x_grid)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    // Gibbs overshoot: max prediction above 1 or below 0.
    let over = fit
        .yhat
        .iter()
        .map(|&v| (v - 1.0).max(0.0).max((-v).max(0.0)))
        .fold(0.0_f64, f64::max);
    let extra = format!("sigma_noise={sigma:.3} gibbs_overshoot={over:.3}");
    assess(probe, formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
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

    // Near-collinear inputs leave directions of the tensor design weakly
    // identified by the data, but the penalty identifies them, so the penalized
    // fit is well defined and a refusal is a failure. The in-span projection
    // uses the pseudo-inverse, which carries no weight on directions the data
    // cannot resolve at rounding level.
    let probe = "multicollinear_tensor";
    let fit = fit_predict_2d(formula, &data, (&a, &b), (&a_test, &b_test))
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let extra = format!("sigma_noise={sigma:.3} corr_ab~0.95");
    let rows = 0..a_test.len();
    assess(probe, formula, &fit, rows, &truth, &y_truth, &extra);
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
    // Declared period π while the data span [0, 2π]. The cyclic basis wraps its
    // argument modulo the period, and π is a period of sin 2θ, so the wrapped
    // regression is well posed: the fit must recover the truth, and a refusal
    // is a failure.
    let formula = "y ~ cyclic(theta, k=10, period_start=0, period_end=3.141592653589793)";

    let mgrid = 400;
    let theta_grid: Vec<f64> = (0..mgrid)
        .map(|i| 0.005 + (TAU - 0.01) * i as f64 / (mgrid as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = theta_grid.iter().map(|t| (2.0 * t).sin()).collect();

    let probe = "cyclic_wrong_period";
    let fit = fit_predict_1d(formula, &data, &theta, &theta_grid)
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let extra = format!("sigma_noise={sigma:.3}");
    assess(probe, formula, &fit, 0..mgrid, &truth, &y_truth, &extra);
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
    }

    // Only the polar rows have a truth; the equator rows are checked for
    // finiteness (by `assess`) and their magnitude is reported.
    let probe = "antipodal_sphere";
    let fit = fit_predict_2d(formula, &data, (&lat, &lon), (&lat_test, &lon_test))
        .unwrap_or_else(|e| fit_error(probe, formula, &e));
    let eq_max = fit.yhat[n_polar..]
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    let extra =
        format!("sigma_noise={sigma:.3} equator_max_abs={eq_max:.4} n_polar={n_polar}");
    assess(
        probe,
        formula,
        &fit,
        0..n_polar,
        &truth_test[..n_polar],
        &y_truth,
        &extra,
    );
}
