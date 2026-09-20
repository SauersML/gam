//! Truth scoring for smooth suites with a known truth (#4377, #4411).
//!
//! A suite that passes on any error whose message lacks a substring cannot
//! fail for the behaviour it is named for, and one that checks a successful
//! fit only for finiteness, or against a multiple of the noise SD, passes a
//! fit that returns the mean. This helper scores a successful fit against
//! the known truth `f` with a derived bar instead.
//!
//! **What the fit estimates.** Let `f_B = X β_B` be the unpenalized
//! least-squares projection of the noise-free truth at the training rows
//! onto the fit's own training design `X` (minimum-norm through the SVD).
//! The remainder `r = f − f_B` is orthogonal to the columns of `X`, so for a
//! Gaussian identity-link fit at any fixed smoothing parameters
//! `β̂ = (XᵀX + S_λ)⁻¹ Xᵀ(X β_B + r + ε) = (XᵀX + S_λ)⁻¹ Xᵀ(X β_B + ε)`:
//! the fit is *exactly* the fit of the in-span truth `f_B`. The part of `f`
//! the basis cannot express (a step, an endpoint pin the truth violates, a
//! too-small basis) is the basis's own approximation floor `‖f_B − f‖`,
//! which no estimator in this basis can undo, and it is reported but not
//! charged to the fit. When the basis expresses `f`, `f_B = f`.
//!
//! **The gate.** The fit's Bayesian band must cover `f_B` across the
//! function (Nychka 1988; Marra & Wood 2012). With the smoothing-corrected
//! posterior covariance `V`, the probe design `X_p`, `C = X_p V X_pᵀ`,
//! `s_i = √C_ii`, `e_i = f̂(x_i) − f_B(x_i)` and the probe correlation
//! `R = D⁻¹ C D⁻¹`, the statistic is `Q = (1/P) Σ e_i² / s_i²`. Under the
//! model behind the band `e ~ N(0, C)`, so `E[Q] = 1` and
//! `Var[Q] = 2·tr(R²)/P²`; the bound is the `1 − α` quantile of the
//! moment-matched scaled `χ²` (Satterthwaite), `Q ≈ g·χ²_h` with
//! `g = tr(R²)/P²`, `h = 1/g`. Smoothing bias adds to `e` while
//! over-smoothing shrinks `s`, so a fit that loses signal drives `Q ≫ 1`.

use gam::data::EncodedDataset;
use gam::linalg::faer_ndarray::FaerSvd;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, StandardFitResult, fit_from_formula};
use gam_math::probability::chi_square_quantile;
use ndarray::{Array1, Array2};

/// Declared family-wise false-alarm rate of one truth-scoring test; a test
/// that scores several cases splits it over them (Bonferroni).
pub const FAMILY_WISE_ALPHA: f64 = 0.01;

/// Fit `formula` to `data` as a Gaussian GAM, returning the standard fit or
/// the fit's own error text.
pub fn fit_gaussian(formula: &str, data: &EncodedDataset) -> Result<StandardFitResult, String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(formula, data, &cfg).map_err(|e| e.to_string())? {
        FitResult::Standard(fit) => Ok(fit),
        _ => Err("a Gaussian formula fit must return the standard fit variant".to_string()),
    }
}

/// Probe rows laid out like the training table (the response column may
/// hold any value; the design rebuild does not read it).
pub fn probe_matrix(rows: &[Vec<f64>]) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((rows.len(), rows[0].len()));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m[[i, j]] = v;
        }
    }
    m
}

/// Densify a design operator by applying it to each unit coefficient vector.
fn dense<D: LinearOperator + ?Sized>(design: &D, p: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((design.nrows(), p));
    for j in 0..p {
        let mut e_j = Array1::<f64>::zeros(p);
        e_j[j] = 1.0;
        x.column_mut(j).assign(&design.apply(&e_j));
    }
    x
}

/// Coefficients of the unpenalized least-squares projection of `y` (one
/// value per training row) onto the fit's own training design: the
/// minimum-norm solution through the SVD, so a rank-deficient design is
/// projected, not regularized.
fn least_squares_projection(fit: &StandardFitResult, y: &[f64]) -> Result<Array1<f64>, String> {
    let p = fit.fit.beta.len();
    let x = dense(&fit.design.design, p);
    let (u, sv, vt) = x
        .svd(true, true)
        .map_err(|e| format!("training design SVD: {e:?}"))?;
    let (u, vt) = (u.ok_or("SVD returned no U")?, vt.ok_or("SVD returned no Vᵀ")?);
    // Singular values below the design's own rounding floor are the
    // numerical null space of X; the projection leaves them out.
    let floor = sv[0] * f64::EPSILON * x.nrows().max(p) as f64;
    let uty = u.t().dot(&Array1::from(y.to_vec()));
    let mut coef = Array1::<f64>::zeros(p);
    for (k, &s) in sv.iter().enumerate() {
        if s > floor {
            coef.scaled_add(uty[k] / s, &vt.row(k));
        }
    }
    Ok(coef)
}

fn rms(v: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = v.fold((0.0, 0usize), |(s, n), x| (s + x * x, n + 1));
    (sum / n as f64).sqrt()
}

/// The coverage verdict of one fit against its truth.
#[derive(Debug, Clone, Copy)]
pub struct Coverage {
    /// Across-the-function coverage statistic of `f̂ − f_B`.
    pub q: f64,
    /// Its `1 − α` bound.
    pub bound: f64,
    /// RMS of `f̂ − f` over the probes.
    pub rmse: f64,
    /// RMS of `f_B − f` over the probes (the basis's approximation floor).
    pub floor: f64,
}

impl Coverage {
    pub fn covers(&self) -> bool {
        self.q <= self.bound
    }
}

impl std::fmt::Display for Coverage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Q={:.3} (E=1, bound={:.3}) rmse(f̂−f)={:.4} floor rmse(f_B−f)={:.4}",
            self.q, self.bound, self.rmse, self.floor
        )
    }
}

/// Score `fit` against the known truth: `truth` at `probes` and
/// `train_truth` at the training rows (the noise-free signal, or `y` itself
/// for a noiseless fixture), at false-alarm rate `alpha`.
pub fn truth_coverage(
    fit: &StandardFitResult,
    probes: &Array2<f64>,
    truth: &[f64],
    train_truth: &[f64],
    alpha: f64,
) -> Result<Coverage, String> {
    let beta = &fit.fit.beta;
    let p = beta.len();
    let cov = fit
        .fit
        .beta_covariance_corrected()
        .ok_or("fit carries no smoothing-corrected posterior covariance")?;
    let design = build_term_collection_design(probes.view(), &fit.resolvedspec)
        .map_err(|e| format!("design rebuild at probes: {e}"))?;
    let x = dense(&design.design, p);
    let eta = x.dot(beta);
    let best = x.dot(&least_squares_projection(fit, train_truth)?);
    let c = x.dot(cov).dot(&x.t());
    let n_probe = truth.len();
    let s: Vec<f64> = (0..n_probe).map(|i| c[[i, i]].sqrt()).collect();
    if let Some(i) = (0..n_probe).find(|&i| !(eta[i].is_finite() && s[i].is_finite() && s[i] > 0.0))
    {
        return Err(format!(
            "probe {i}: posterior mean {} with standard error {}",
            eta[i], s[i]
        ));
    }
    let q = (0..n_probe)
        .map(|i| ((eta[i] - best[i]) / s[i]).powi(2))
        .sum::<f64>()
        / n_probe as f64;
    let mut tr_r2 = 0.0;
    for i in 0..n_probe {
        for j in 0..n_probe {
            tr_r2 += (c[[i, j]] / (s[i] * s[j])).powi(2);
        }
    }
    let g = tr_r2 / (n_probe * n_probe) as f64;
    Ok(Coverage {
        q,
        bound: g * chi_square_quantile(1.0 - alpha, 1.0 / g),
        rmse: rms((0..n_probe).map(|i| eta[i] - truth[i])),
        floor: rms((0..n_probe).map(|i| best[i] - truth[i])),
    })
}

/// Fit `formula` and score it at false-alarm rate `alpha`; `Err` names a
/// refusal or a band that does not cover the basis's best approximation.
pub fn fit_and_score(
    formula: &str,
    data: &EncodedDataset,
    probes: &Array2<f64>,
    truth: &[f64],
    train_truth: &[f64],
    alpha: f64,
) -> Result<Coverage, String> {
    let fit = fit_gaussian(formula, data).map_err(|e| format!("refused: {e}"))?;
    let coverage = truth_coverage(&fit, probes, truth, train_truth, alpha)?;
    eprintln!("[truth-coverage] `{formula}` {coverage}");
    if coverage.covers() {
        Ok(coverage)
    } else {
        Err(format!("band misses the basis's best approximation: {coverage}"))
    }
}
