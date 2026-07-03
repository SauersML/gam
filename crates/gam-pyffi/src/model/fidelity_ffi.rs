//! FFI shims for the model-currency fidelity metric core (scorecard axis 1).
//!
//! Thin `#[pyfunction]` wrappers over `gam_math::fidelity_metrics` (the single
//! source of truth) — no math lives here. Matrices arrive row-major flat with
//! explicit shapes, exactly as `gamfit/_fidelity_metrics.py` sends them; that
//! Python module auto-detects `fidelity_loss_recovered` and switches its numpy
//! fallback to these kernels, and `tests/metrics/test_fidelity_metrics_parity.py`
//! then pins the two paths to double precision.

use gam_math::fidelity_metrics;

#[pyfunction]
fn fidelity_loss_recovered(l_clean: f64, l_recon: f64, l_ablate: f64) -> PyResult<f64> {
    Ok(fidelity_metrics::loss_recovered(l_clean, l_recon, l_ablate))
}

#[pyfunction]
fn fidelity_r2_score(
    clean: Vec<f64>,
    approx: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<f64> {
    if clean.len() != n_rows * n_cols || approx.len() != n_rows * n_cols {
        return Err(py_value_error(format!(
            "fidelity_r2_score shape mismatch: clean={}, approx={}, expected {n_rows}*{n_cols}",
            clean.len(),
            approx.len()
        )));
    }
    Ok(fidelity_metrics::r2_score(&clean, &approx, n_rows, n_cols))
}

#[pyfunction]
fn fidelity_kl_categorical_rows(
    clean_logprobs: Vec<f64>,
    other_logprobs: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<f64> {
    if clean_logprobs.len() != n_rows * n_cols || other_logprobs.len() != n_rows * n_cols {
        return Err(py_value_error(format!(
            "fidelity_kl_categorical_rows shape mismatch: clean={}, other={}, expected {n_rows}*{n_cols}",
            clean_logprobs.len(),
            other_logprobs.len()
        )));
    }
    Ok(fidelity_metrics::kl_categorical_rows(
        &clean_logprobs,
        &other_logprobs,
        n_rows,
        n_cols,
    ))
}

#[pyfunction]
fn fidelity_distortion_floor_r2(
    r2s: Vec<f64>,
    loss_recovereds: Vec<f64>,
    tol_frac: f64,
) -> PyResult<Option<f64>> {
    if r2s.len() != loss_recovereds.len() {
        return Err(py_value_error(format!(
            "fidelity_distortion_floor_r2 parallel-array mismatch: r2s={}, lr={}",
            r2s.len(),
            loss_recovereds.len()
        )));
    }
    Ok(fidelity_metrics::distortion_floor_r2(
        &r2s,
        &loss_recovereds,
        tol_frac,
    ))
}
