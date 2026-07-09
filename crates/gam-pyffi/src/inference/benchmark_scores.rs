//! Benchmark scalar scoring kernels.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the pure
//! `&[f64]`-in scalar metric kernels used to score predictions against a mature
//! reference -- AUC (tie-aware Mann-Whitney), binary log-loss (with an
//! epsilon-clipped variant), the saturated-exponential helper, Nagelkerke R^2
//! (null-mean and explicit-null-mean forms), and Gaussian log-loss. They depend
//! on nothing in the rest of the module except the boundary error type
//! `PyValueError`; the `#[pyfunction]`s that aggregate them
//! (`benchmark_prediction_metrics`, etc.) stay in the parent module via a
//! focused re-import.

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

pub(crate) fn benchmark_auc_score(observed: &[f64], predicted_mean: &[f64]) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "auc length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    let mut pairs: Vec<(f64, bool)> = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| (p, y > 0.5))
        .collect();
    let n_pos = pairs.iter().filter(|(_, is_pos)| *is_pos).count();
    let n_neg = pairs.len().saturating_sub(n_pos);
    if n_pos == 0 || n_neg == 0 {
        return Ok(0.5);
    }
    pairs.sort_by(|(a, _), (b, _)| a.total_cmp(b));

    let mut concordant = 0.0;
    let mut negatives_below = 0usize;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let pos_in_group = pairs[i..j].iter().filter(|(_, is_pos)| *is_pos).count();
        let neg_in_group = (j - i).saturating_sub(pos_in_group);
        concordant += (pos_in_group * negatives_below) as f64;
        concordant += 0.5 * (pos_in_group * neg_in_group) as f64;
        negatives_below += neg_in_group;
        i = j;
    }
    Ok(concordant / ((n_pos * n_neg) as f64))
}

pub(crate) fn benchmark_binary_logloss(observed: &[f64], predicted_mean: &[f64]) -> PyResult<f64> {
    benchmark_binary_logloss_eps(observed, predicted_mean, 1.0e-12)
}

pub(crate) fn benchmark_binary_logloss_eps(
    observed: &[f64],
    predicted_mean: &[f64],
    eps: f64,
) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "logloss length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Err(PyValueError::new_err("logloss requires at least one row"));
    }
    let loss_sum = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(eps, 1.0 - eps);
            -(y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln())
        })
        .sum::<f64>();
    Ok(loss_sum / observed.len() as f64)
}

pub(crate) fn benchmark_exp_saturated(x: f64) -> f64 {
    if x >= 709.0 {
        f64::INFINITY
    } else if x <= -745.0 {
        0.0
    } else {
        x.exp()
    }
}

pub(crate) fn benchmark_nagelkerke_r2(
    observed: &[f64],
    predicted_mean: &[f64],
    train_observed: &[f64],
) -> PyResult<Option<f64>> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "nagelkerke length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() || train_observed.is_empty() {
        return Ok(None);
    }
    let null_mean = train_observed.iter().sum::<f64>() / train_observed.len() as f64;
    if !null_mean.is_finite() || null_mean <= 0.0 || null_mean >= 1.0 {
        return Ok(None);
    }
    let eps = 1.0e-12;
    let log_null = null_mean.ln();
    let log_not_null = (1.0 - null_mean).ln();
    let ll_null = observed
        .iter()
        .map(|&y| y * log_null + (1.0 - y) * log_not_null)
        .sum::<f64>();
    let ll_model = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(eps, 1.0 - eps);
            y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln()
        })
        .sum::<f64>();
    let n = observed.len() as f64;
    let r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * (ll_null - ll_model));
    let max_r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * ll_null);
    if !r2_cs.is_finite() || !max_r2_cs.is_finite() || max_r2_cs <= 0.0 {
        return Ok(None);
    }
    Ok(Some(r2_cs / max_r2_cs))
}

pub(crate) fn benchmark_nagelkerke_r2_with_null_mean(
    observed: &[f64],
    predicted_mean: &[f64],
    null_mean: f64,
    eps: f64,
) -> PyResult<Option<f64>> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "nagelkerke length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Ok(None);
    }
    if !null_mean.is_finite() || null_mean <= 0.0 || null_mean >= 1.0 {
        return Ok(None);
    }
    let log_null = null_mean.ln();
    let log_not_null = (1.0 - null_mean).ln();
    let ll_null = observed
        .iter()
        .map(|&y| y * log_null + (1.0 - y) * log_not_null)
        .sum::<f64>();
    let ll_model = observed
        .iter()
        .zip(predicted_mean.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(eps, 1.0 - eps);
            y * clipped.ln() + (1.0 - y) * (1.0 - clipped).ln()
        })
        .sum::<f64>();
    let n = observed.len() as f64;
    let r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * (ll_null - ll_model));
    let max_r2_cs = 1.0 - benchmark_exp_saturated((2.0 / n) * ll_null);
    if !r2_cs.is_finite() || !max_r2_cs.is_finite() || max_r2_cs <= 0.0 {
        return Ok(None);
    }
    Ok(Some(r2_cs / max_r2_cs))
}

pub(crate) fn benchmark_gaussian_logloss(
    observed: &[f64],
    predicted_mean: &[f64],
    sigma: &[f64],
) -> PyResult<f64> {
    if observed.len() != predicted_mean.len() {
        return Err(PyValueError::new_err(format!(
            "gaussian logloss length mismatch: observed={} predicted={}",
            observed.len(),
            predicted_mean.len()
        )));
    }
    if observed.is_empty() {
        return Err(PyValueError::new_err(
            "gaussian logloss requires at least one row",
        ));
    }
    if sigma.len() != 1 && sigma.len() != observed.len() {
        return Err(PyValueError::new_err(format!(
            "sigma length must be 1 or {}, got {}",
            observed.len(),
            sigma.len()
        )));
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut loss_sum = 0.0_f64;
    for (idx, (&y, &mu)) in observed.iter().zip(predicted_mean.iter()).enumerate() {
        let raw_sigma = if sigma.len() == 1 {
            sigma[0]
        } else {
            sigma[idx]
        };
        if !raw_sigma.is_finite() || raw_sigma <= 0.0 {
            // A standard deviation is strictly positive by definition; flooring
            // a nonpositive scale to eps would reward an invalid prediction
            // with a spuriously sharp density instead of rejecting it.
            return Err(PyValueError::new_err(format!(
                "gaussian logloss: sigma[{}] must be strictly positive and finite; \
                 got {raw_sigma}",
                if sigma.len() == 1 { 0 } else { idx }
            )));
        }
        let var = raw_sigma * raw_sigma;
        loss_sum += 0.5 * (two_pi * var).ln() + ((y - mu) * (y - mu)) / (2.0 * var);
    }
    Ok(loss_sum / observed.len() as f64)
}
