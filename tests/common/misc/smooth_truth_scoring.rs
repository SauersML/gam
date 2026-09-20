//! Truth scoring for the small-n / corner-case smooth suites (#4377).
//!
//! A stability suite that passes on any error whose message lacks a
//! substring cannot fail for the behaviour it is named for, and one that
//! checks a successful fit only for finiteness passes a fit that returns
//! the mean. This helper scores a successful fit against the known truth
//! `f` with a derived bar instead of a range.
//!
//! The fit estimates the best approximation of `f` its basis can express,
//! `f_B`: the unpenalized least-squares projection of the noise-free truth
//! at the training rows onto the fit's own training design (for a noiseless
//! fixture, the projection of `y` itself). By the triangle inequality
//! `|f̂ − f| ≤ |f_B − f| + |f̂ − f_B|`. The first term is the basis's own
//! irreducible approximation error — zero up to rounding when the basis can
//! express `f` — and the second is the estimation error inside the span,
//! which the fit's posterior band bounds: `|f̂ − f_B| ≤ z·se`, with `z` the
//! two-sided normal quantile at a declared family-wise false-alarm rate
//! Bonferroni-split over the probes. The `se` is the
//! smoothing-parameter-corrected posterior standard error, the surface a
//! small-n band must use to be calibrated.

use gam::data::EncodedDataset;
use gam::linalg::faer_ndarray::FaerSvd;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, StandardFitResult, fit_from_formula};
use gam_test_support::calibration::standard_normal_quantile;
use ndarray::{Array1, Array2};

/// Declared family-wise false-alarm rate of one truth-scoring assertion.
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

/// Two-sided normal quantile at `FAMILY_WISE_ALPHA` Bonferroni-split over
/// `probes` pointwise comparisons.
fn bonferroni_z(probes: usize) -> f64 {
    standard_normal_quantile(1.0 - FAMILY_WISE_ALPHA / (2.0 * probes as f64))
}

/// Posterior mean and corrected posterior standard error of the linear
/// predictor at `probes` (rows laid out like the training table).
fn probe_posterior(
    fit: &StandardFitResult,
    probes: &Array2<f64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let beta = &fit.fit.beta;
    let p = beta.len();
    let cov = fit
        .fit
        .beta_covariance_corrected()
        .ok_or("fit carries no smoothing-corrected posterior covariance")?;
    let design = build_term_collection_design(probes.view(), &fit.resolvedspec)
        .map_err(|e| format!("design rebuild at probes: {e}"))?;
    let x = dense(&design.design, p);
    let eta = x.dot(beta).to_vec();
    let se = x
        .rows()
        .into_iter()
        .map(|row| row.dot(&cov.dot(&row)).max(0.0).sqrt())
        .collect::<Vec<_>>();
    if !eta.iter().chain(se.iter()).all(|v| v.is_finite()) {
        return Err(format!("non-finite probe posterior: eta={eta:?} se={se:?}"));
    }
    Ok((eta, se))
}

/// Predictions at `probes` of the unpenalized least-squares projection of
/// `y` (one value per training row) onto the fit's own training design
/// (minimum-norm solution through the SVD, so a rank-deficient design is
/// projected, not regularized).
fn least_squares_projection(
    fit: &StandardFitResult,
    y: &[f64],
    probes: &Array2<f64>,
) -> Vec<f64> {
    let p = fit.fit.beta.len();
    let x = dense(&fit.design.design, p);
    let (u, sv, vt) = x.svd(true, true).expect("training design SVD");
    let (u, vt) = (u.expect("U"), vt.expect("Vt"));
    // Singular values below the design's own rounding floor are the
    // numerical null space of X; the projection leaves them out.
    let floor = sv[0] * f64::EPSILON * x.nrows().max(p) as f64;
    let y = Array1::from(y.to_vec());
    let uty = u.t().dot(&y);
    let mut coef = Array1::<f64>::zeros(p);
    for (k, &s) in sv.iter().enumerate() {
        if s > floor {
            coef.scaled_add(uty[k] / s, &vt.row(k));
        }
    }
    let design = build_term_collection_design(probes.view(), &fit.resolvedspec)
        .expect("design rebuild at probes");
    design.design.apply(&coef).to_vec()
}

/// Score `fit` against the known truth: `truth` at `probes` and
/// `train_truth` at the training rows (the noise-free signal, or `y` itself
/// for a noiseless fixture). Returns one message per probe outside
/// `|f_B − f| + z·se`, or the reason the posterior could not be formed.
pub fn truth_misses(
    fit: &StandardFitResult,
    probes: &Array2<f64>,
    truth: &[f64],
    train_truth: &[f64],
) -> Result<Vec<String>, String> {
    let (eta, se) = probe_posterior(fit, probes)?;
    let best = least_squares_projection(fit, train_truth, probes);
    let z = bonferroni_z(truth.len());
    Ok((0..truth.len())
        .filter(|&i| (eta[i] - truth[i]).abs() > (best[i] - truth[i]).abs() + z * se[i])
        .map(|i| {
            format!(
                "probe {i}: f̂={:.4} f={:.4} |f̂−f|={:.4} > |f_B−f|={:.4} + z·se={:.4} (z={z:.3})",
                eta[i],
                truth[i],
                (eta[i] - truth[i]).abs(),
                (best[i] - truth[i]).abs(),
                z * se[i]
            )
        })
        .collect())
}

/// Fit `formula` and score it; `Err` names a refusal or the probes missed.
pub fn fit_and_score(
    formula: &str,
    data: &EncodedDataset,
    probes: &Array2<f64>,
    truth: &[f64],
    train_truth: &[f64],
) -> Result<(), String> {
    let fit = fit_gaussian(formula, data).map_err(|e| format!("refused: {e}"))?;
    let misses = truth_misses(&fit, probes, truth, train_truth)?;
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}
