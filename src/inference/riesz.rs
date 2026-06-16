use crate::faer_ndarray::FaerCholesky;
use crate::model_types::EstimationError;
use crate::solver::sensitivity::FitSensitivity;
use faer::Side;
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Closed-form Riesz representer for a linear functional of a fitted smooth.
///
/// `coefficients` is `H^-1 g`, where `H` is the penalized fitted Hessian and
/// `g = d theta / d beta`. `influence` contains the per-observation influence
/// values on the usual root-n scale. When leverages are supplied, the values are
/// analytically own-observation removed by dividing by `1 - h_ii`.
#[derive(Clone, Debug)]
pub struct RieszRepresenter {
    pub functional_gradient: Array1<f64>,
    pub coefficients: Array1<f64>,
    pub influence: Array1<f64>,
    pub centered_influence: Array1<f64>,
    pub leverage: Option<Array1<f64>>,
}

#[derive(Clone, Debug)]
pub struct RieszDebiasReport {
    pub theta_plugin: f64,
    pub theta_onestep: f64,
    pub se: f64,
    pub penalty_bias: f64,
    pub representer: RieszRepresenter,
}

/// Functional descriptor for the Layer-1 closed-form path. All variants are
/// linear in the fitted coefficient vector.
pub enum SmoothFunctional<'a> {
    /// `m(x0)`, represented by the prediction/design row at `x0`.
    PointEvaluation { design_row: ArrayView1<'a, f64> },
    /// `mean_i w_i * d m(x_i) / d x_j`, represented by derivative-basis rows.
    AverageDerivative {
        derivative_design: ArrayView2<'a, f64>,
        weights: Option<ArrayView1<'a, f64>>,
    },
    /// `m(x_a) - m(x_b)`, represented by the two prediction design rows.
    Contrast {
        design_row_a: ArrayView1<'a, f64>,
        design_row_b: ArrayView1<'a, f64>,
    },
    /// `mean_i w_i * m(x_i)`, represented by value-basis rows.
    AverageValue {
        value_design: ArrayView2<'a, f64>,
        weights: Option<ArrayView1<'a, f64>>,
    },
    /// Direct caller-supplied linear functional gradient.
    Linear { gradient: ArrayView1<'a, f64> },
}

impl<'a> SmoothFunctional<'a> {
    pub fn gradient(&self) -> Result<Array1<f64>, EstimationError> {
        match self {
            Self::PointEvaluation { design_row } => {
                if design_row.is_empty() || design_row.iter().any(|value| !value.is_finite()) {
                    crate::bail_invalid_estim!(
                        "Riesz point-evaluation functional requires a finite non-empty design row"
                    );
                }
                Ok(design_row.to_owned())
            }
            Self::AverageDerivative {
                derivative_design,
                weights,
            } => average_derivative_gradient(*derivative_design, *weights),
            Self::Contrast {
                design_row_a,
                design_row_b,
            } => contrast_gradient(*design_row_a, *design_row_b),
            Self::AverageValue {
                value_design,
                weights,
            } => weighted_row_mean(*value_design, *weights, "average-value"),
            Self::Linear { gradient } => {
                if gradient.is_empty() || gradient.iter().any(|value| !value.is_finite()) {
                    crate::bail_invalid_estim!(
                        "Riesz linear functional requires a finite non-empty gradient"
                    );
                }
                Ok(gradient.to_owned())
            }
        }
    }
}

pub struct RieszInput<'a> {
    /// Fitted coefficients in the same basis as the Hessian and gradient.
    pub beta: ArrayView1<'a, f64>,
    /// Functional gradient `g = d theta / d beta`.
    pub functional_gradient: ArrayView1<'a, f64>,
    /// Per-row objective score contributions `s_i = d nll_i / d beta`.
    pub row_scores: ArrayView2<'a, f64>,
    /// Penalty gradient `S beta` in the same coefficient basis.
    pub penalty_beta: ArrayView1<'a, f64>,
    /// Optional ALO leverage values for exact own-observation removal.
    pub leverage: Option<ArrayView1<'a, f64>>,
}

pub fn debias_with_dense_hessian(
    input: &RieszInput<'_>,
    penalized_hessian: ArrayView2<'_, f64>,
) -> Result<RieszDebiasReport, EstimationError> {
    let p = input.beta.len();
    validate_square_hessian(penalized_hessian, p)?;
    let h = penalized_hessian.to_owned();
    let factor = h.cholesky(Side::Lower).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "Riesz representer requires SPD penalized Hessian: {err}"
        ))
    })?;
    let sensitivity = FitSensitivity::from_faer_cholesky(&factor, p);
    debias_with_sensitivity(input, &sensitivity)
}

pub fn debias_with_sensitivity(
    input: &RieszInput<'_>,
    sensitivity: &FitSensitivity<'_>,
) -> Result<RieszDebiasReport, EstimationError> {
    validate_input(input)?;
    let p = input.beta.len();
    if sensitivity.dim() != p {
        crate::bail_invalid_estim!(
            "Riesz sensitivity dimension {} must equal beta length {p}",
            sensitivity.dim()
        );
    }

    let g = input.functional_gradient.to_owned();
    let coefficients = sensitivity.apply(&g);
    if coefficients.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("Riesz H^-1 gradient solve produced non-finite values");
    }

    let theta_plugin = g.dot(&input.beta);
    let penalty_correction = coefficients.dot(&input.penalty_beta);
    let penalty_bias = -penalty_correction;
    let theta_onestep = theta_plugin - penalty_bias;

    let influence = influence_values(input, &coefficients)?;
    let centered_influence = centered(&influence);
    let se = plugin_standard_error(&centered_influence)?;

    if !theta_plugin.is_finite()
        || !theta_onestep.is_finite()
        || !se.is_finite()
        || !penalty_bias.is_finite()
    {
        crate::bail_invalid_estim!("Riesz debiasing produced non-finite estimate");
    }

    Ok(RieszDebiasReport {
        theta_plugin,
        theta_onestep,
        se,
        penalty_bias,
        representer: RieszRepresenter {
            functional_gradient: g,
            coefficients,
            influence,
            centered_influence,
            leverage: input.leverage.map(|view| view.to_owned()),
        },
    })
}

pub fn average_derivative_gradient(
    derivative_design: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    weighted_row_mean(derivative_design, weights, "average-derivative")
}

pub fn contrast_gradient(
    design_row_a: ArrayView1<'_, f64>,
    design_row_b: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, EstimationError> {
    if design_row_a.is_empty() || design_row_a.len() != design_row_b.len() {
        crate::bail_invalid_estim!(
            "Riesz contrast functional requires two non-empty design rows of equal length, got {} and {}",
            design_row_a.len(),
            design_row_b.len()
        );
    }
    if design_row_a.iter().any(|value| !value.is_finite())
        || design_row_b.iter().any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!("Riesz contrast functional requires finite design rows");
    }
    Ok(&design_row_a.to_owned() - &design_row_b)
}

fn weighted_row_mean(
    rows: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    what: &str,
) -> Result<Array1<f64>, EstimationError> {
    let n = rows.nrows();
    let p = rows.ncols();
    if n == 0 || p == 0 {
        crate::bail_invalid_estim!(
            "Riesz {what} functional requires non-empty basis rows, got {n}x{p}"
        );
    }
    if rows.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("Riesz {what} functional requires finite basis rows");
    }

    let mut gradient = Array1::<f64>::zeros(p);
    match weights {
        None => {
            let scale = 1.0 / n as f64;
            for row in rows.rows() {
                for col in 0..p {
                    gradient[col] += scale * row[col];
                }
            }
        }
        Some(w) => {
            if w.len() != n || w.iter().any(|value| !value.is_finite()) {
                crate::bail_invalid_estim!(
                    "Riesz {what} weights must be finite with length {n}, got {}",
                    w.len()
                );
            }
            let weight_sum = w.sum();
            if !(weight_sum.is_finite() && weight_sum > 0.0) {
                crate::bail_invalid_estim!("Riesz {what} weights must have positive finite sum");
            }
            for row_idx in 0..n {
                let scale = w[row_idx] / weight_sum;
                for col in 0..p {
                    gradient[col] += scale * rows[[row_idx, col]];
                }
            }
        }
    }
    Ok(gradient)
}

pub fn point_evaluation_gradient(
    design_row: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, EstimationError> {
    SmoothFunctional::PointEvaluation { design_row }.gradient()
}

fn validate_input(input: &RieszInput<'_>) -> Result<(), EstimationError> {
    let p = input.beta.len();
    let n = input.row_scores.nrows();
    if p == 0 || n == 0 {
        crate::bail_invalid_estim!(
            "Riesz input requires non-empty beta and row scores, got beta length {p}, row count {n}"
        );
    }
    if input.functional_gradient.len() != p
        || input.row_scores.ncols() != p
        || input.penalty_beta.len() != p
    {
        crate::bail_invalid_estim!(
            "Riesz input dimension mismatch: beta={p}, gradient={}, row_scores={}x{}, penalty_beta={}",
            input.functional_gradient.len(),
            input.row_scores.nrows(),
            input.row_scores.ncols(),
            input.penalty_beta.len()
        );
    }
    if let Some(leverage) = input.leverage {
        if leverage.len() != n || leverage.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!(
                "Riesz leverage must be finite with length {n}, got {}",
                leverage.len()
            );
        }
    }
    if input.beta.iter().any(|value| !value.is_finite())
        || input
            .functional_gradient
            .iter()
            .any(|value| !value.is_finite())
        || input.row_scores.iter().any(|value| !value.is_finite())
        || input.penalty_beta.iter().any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!(
            "Riesz input requires finite beta, gradient, row scores, and penalty gradient"
        );
    }
    Ok(())
}

fn validate_square_hessian(
    penalized_hessian: ArrayView2<'_, f64>,
    p: usize,
) -> Result<(), EstimationError> {
    if penalized_hessian.nrows() != p || penalized_hessian.ncols() != p {
        crate::bail_invalid_estim!(
            "Riesz penalized Hessian must be {p}x{p}, got {}x{}",
            penalized_hessian.nrows(),
            penalized_hessian.ncols()
        );
    }
    if penalized_hessian.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("Riesz penalized Hessian must be finite");
    }
    Ok(())
}

fn influence_values(
    input: &RieszInput<'_>,
    coefficients: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let n = input.row_scores.nrows();
    let mut influence = Array1::<f64>::zeros(n);
    for row_idx in 0..n {
        let raw = -(n as f64) * input.row_scores.row(row_idx).dot(coefficients);
        influence[row_idx] = match input.leverage {
            None => raw,
            Some(leverage) => {
                let denom = 1.0 - leverage[row_idx];
                if !denom.is_finite() || denom.abs() <= f64::EPSILON {
                    crate::bail_invalid_estim!(
                        "Riesz own-observation removal is singular at row {row_idx}: leverage={}",
                        leverage[row_idx]
                    );
                }
                raw / denom
            }
        };
    }
    if influence.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("Riesz influence values must be finite");
    }
    Ok(influence)
}

fn centered(values: &Array1<f64>) -> Array1<f64> {
    let mean = values.sum() / values.len() as f64;
    values.mapv(|value| value - mean)
}

fn plugin_standard_error(centered_influence: &Array1<f64>) -> Result<f64, EstimationError> {
    let n = centered_influence.len();
    if n < 2 {
        crate::bail_invalid_estim!("Riesz plug-in SE requires at least two observations");
    }
    let variance = centered_influence.dot(centered_influence) / (n - 1) as f64;
    Ok(variance.sqrt() / (n as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn dense_solve(mut a: Array2<f64>, mut b: Array1<f64>) -> Array1<f64> {
        let n = b.len();
        for pivot in 0..n {
            let mut best = pivot;
            let mut best_abs = a[[pivot, pivot]].abs();
            for row in (pivot + 1)..n {
                let candidate = a[[row, pivot]].abs();
                if candidate > best_abs {
                    best = row;
                    best_abs = candidate;
                }
            }
            assert!(best_abs > 1e-14, "dense oracle pivot is singular");
            if best != pivot {
                for col in 0..n {
                    a.swap((pivot, col), (best, col));
                }
                b.swap(pivot, best);
            }
            let pivot_value = a[[pivot, pivot]];
            for col in pivot..n {
                a[[pivot, col]] /= pivot_value;
            }
            b[pivot] /= pivot_value;
            for row in 0..n {
                if row != pivot {
                    let factor = a[[row, pivot]];
                    for col in pivot..n {
                        a[[row, col]] -= factor * a[[pivot, col]];
                    }
                    b[row] -= factor * b[pivot];
                }
            }
        }
        b
    }

    #[test]
    fn representer_matches_dense_oracle_on_small_fixture() {
        let h = array![[6.0, 1.0, 0.5], [1.0, 4.5, -0.2], [0.5, -0.2, 3.5]];
        let beta = array![0.3, -0.7, 1.1];
        let gradient = array![1.0, 0.25, -0.5];
        let row_scores = array![
            [0.2, -0.1, 0.4],
            [-0.3, 0.5, 0.2],
            [0.1, 0.4, -0.6],
            [0.0, -0.2, 0.3]
        ];
        let penalty_beta = array![0.1, -0.4, 0.7];
        let input = RieszInput {
            beta: beta.view(),
            functional_gradient: gradient.view(),
            row_scores: row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        };

        let report = debias_with_dense_hessian(&input, h.view()).expect("Riesz report");
        let oracle = dense_solve(h, gradient.clone());
        for col in 0..oracle.len() {
            assert!(
                (report.representer.coefficients[col] - oracle[col]).abs() < 1e-12,
                "representer coefficient {col}: {} vs oracle {}",
                report.representer.coefficients[col],
                oracle[col]
            );
        }

        for row in 0..row_scores.nrows() {
            let expected = -(row_scores.nrows() as f64) * row_scores.row(row).dot(&oracle);
            assert!(
                (report.representer.influence[row] - expected).abs() < 1e-12,
                "influence row {row}: {} vs oracle {}",
                report.representer.influence[row],
                expected
            );
        }
        let expected_theta = gradient.dot(&beta) + oracle.dot(&penalty_beta);
        assert!((report.theta_onestep - expected_theta).abs() < 1e-12);
    }

    #[test]
    fn penalty_debiasing_reduces_average_derivative_bias_under_oversmoothing() {
        let n = 80usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut derivative_design = Array2::<f64>::zeros((n, p));
        let mut weights = Array1::<f64>::zeros(n);
        let beta_truth = array![0.2, -0.4, 2.5];
        for row in 0..n {
            let z = row as f64 / (n - 1) as f64;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = z;
            x[[row, 2]] = z * z;
            derivative_design[[row, 1]] = 1.0;
            derivative_design[[row, 2]] = 2.0 * z;
            weights[row] = 1.0 + 4.0 * z;
        }
        let y = x.dot(&beta_truth);
        let mut penalty = Array2::<f64>::zeros((p, p));
        penalty[[2, 2]] = 0.1;
        let h = &x.t().dot(&x) + &penalty;
        let rhs = x.t().dot(&y);
        let beta_hat = dense_solve(h.clone(), rhs);
        let mu = x.dot(&beta_hat);
        let mut row_scores = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let residual = mu[row] - y[row];
            for col in 0..p {
                row_scores[[row, col]] = x[[row, col]] * residual;
            }
        }
        let gradient = average_derivative_gradient(derivative_design.view(), Some(weights.view()))
            .expect("average derivative gradient");
        let penalty_beta = penalty.dot(&beta_hat);
        let input = RieszInput {
            beta: beta_hat.view(),
            functional_gradient: gradient.view(),
            row_scores: row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        };

        let report = debias_with_dense_hessian(&input, h.view()).expect("Riesz report");
        let truth = gradient.dot(&beta_truth);
        let plugin_bias = (report.theta_plugin - truth).abs();
        let debiased_bias = (report.theta_onestep - truth).abs();

        assert!(
            debiased_bias < 0.25 * plugin_bias,
            "debiased average derivative should remove most smoothing bias: plugin={plugin_bias:.6e}, debiased={debiased_bias:.6e}"
        );
        assert!(report.se.is_finite(), "plug-in SE must be finite");
    }
}
