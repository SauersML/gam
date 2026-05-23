//! Prediction helper for vector-valued response models.
//!
//! Companion to `families/vector_response.rs`. The scalar prediction pipeline
//! (see `inference/predict.rs`) operates on `(N,)` outputs; this module wraps
//! M independent scalar predictions into the `(N, M)` reconstruction that the
//! latent-variable / SAE-manifold engine consumes.
//!
//! Uncertainty bands are scaffolded but not implemented end-to-end: by default
//! we fall through to the per-column scalar predictor's SE when a `predict_se`
//! is supplied. Joint posterior covariance across outputs is deferred.

use crate::estimate::EstimationError;
use ndarray::{Array1, Array2, ArrayView2};

/// Trait a vector-response model implements so this module can drive it.
///
/// `n_outputs` is M; `predict_column` produces η for output `m` at the rows of
/// the supplied design `X`. The model is responsible for any link inversion
/// before returning. The `X` and `t` shapes are intentionally loose at this
/// layer — concrete callers pass through their own design matrices and latent
/// coordinates.
pub trait VectorPredictableModel {
    type Design;
    type Latent;

    fn n_outputs(&self) -> usize;

    /// η (or mean, by link choice) for column `m`, shape (N,).
    fn predict_column(
        &self,
        x: &Self::Design,
        t: &Self::Latent,
        m: usize,
    ) -> Result<Array1<f64>, EstimationError>;

    /// Optional per-column SE on η. Default: None (scaffold the API).
    fn predict_column_se(
        &self,
        _x: &Self::Design,
        _t: &Self::Latent,
        _m: usize,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }
}

/// `(N, M)` reconstruction with optional `(N, M)` SE matrix.
#[derive(Clone, Debug)]
pub struct VectorPredictResult {
    /// (N, M) point prediction.
    pub mean: Array2<f64>,
    /// (N, M) per-cell standard error, if available.
    pub se: Option<Array2<f64>>,
}

/// Build the (N, M) prediction by stacking M scalar columns.
///
/// `predict_vector(model, X, t)` from the task spec — `X` and `t` are the
/// model's own associated types (intentionally loose so the latent-variable
/// engine can pass per-row latent coordinates without forcing a concrete
/// design-matrix encoding at this layer).
pub fn predict_vector<M: VectorPredictableModel>(
    model: &M,
    x: &M::Design,
    t: &M::Latent,
) -> Result<Array2<f64>, EstimationError> {
    let m = model.n_outputs();
    if m == 0 {
        return Err(EstimationError::InvalidInput(
            "predict_vector: model reports zero output dimensions".to_string(),
        ));
    }
    // Probe column 0 to discover N.
    let first = model.predict_column(x, t, 0)?;
    let n = first.len();
    let mut out = Array2::<f64>::zeros((n, m));
    out.column_mut(0).assign(&first);
    for j in 1..m {
        let col = model.predict_column(x, t, j)?;
        if col.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "predict_vector: column {j} length {} ≠ column 0 length {n}",
                col.len()
            )));
        }
        out.column_mut(j).assign(&col);
    }
    Ok(out)
}

/// Same as `predict_vector` but also accumulates per-column SE if the model
/// exposes one. SE entries default to NaN-free zeros only when *all* columns
/// return Some(SE); if any column returns None the joint SE matrix is omitted.
pub fn predict_vectorwith_uncertainty<M: VectorPredictableModel>(
    model: &M,
    x: &M::Design,
    t: &M::Latent,
) -> Result<VectorPredictResult, EstimationError> {
    let m_out = model.n_outputs();
    if m_out == 0 {
        return Err(EstimationError::InvalidInput(
            "predict_vectorwith_uncertainty: model reports zero output dimensions".to_string(),
        ));
    }
    let first_mean = model.predict_column(x, t, 0)?;
    let n = first_mean.len();
    let first_se = model.predict_column_se(x, t, 0)?;
    let mut mean = Array2::<f64>::zeros((n, m_out));
    let mut se_opt: Option<Array2<f64>> = first_se.as_ref().map(|_| Array2::zeros((n, m_out)));
    mean.column_mut(0).assign(&first_mean);
    if let (Some(se_mat), Some(se_col)) = (se_opt.as_mut(), first_se.as_ref()) {
        se_mat.column_mut(0).assign(se_col);
    }
    for j in 1..m_out {
        let col_mean = model.predict_column(x, t, j)?;
        if col_mean.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "predict_vectorwith_uncertainty: column {j} length {} ≠ N={n}",
                col_mean.len()
            )));
        }
        mean.column_mut(j).assign(&col_mean);
        let col_se = model.predict_column_se(x, t, j)?;
        match (se_opt.as_mut(), col_se) {
            (Some(se_mat), Some(s)) => {
                if s.len() != n {
                    return Err(EstimationError::InvalidInput(format!(
                        "predict_vectorwith_uncertainty: SE column {j} length {} ≠ N={n}",
                        s.len()
                    )));
                }
                se_mat.column_mut(j).assign(&s);
            }
            // Drop SE entirely if any column is missing it.
            _ => {
                se_opt = None;
            }
        }
    }
    Ok(VectorPredictResult { mean, se: se_opt })
}

/// Convenience: residuals on the response scale, `Y − Ŷ`, shape (N, M).
/// Useful for downstream diagnostics on vector targets.
pub fn vector_residuals(y: ArrayView2<f64>, y_hat: ArrayView2<f64>) -> Array2<f64> {
    debug_assert_eq!(y.dim(), y_hat.dim());
    &y.to_owned() - &y_hat.to_owned()
}
