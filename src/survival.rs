use crate::estimate::EstimationError;
use crate::pirls::{WorkingModel as PirlsWorkingModel, WorkingState};
use crate::types::{Coefficients, LinearPredictor};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("input dimensions are inconsistent")]
    DimensionMismatch,
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub minimum_age: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn transform(&self, age: f64) -> Result<f64, SurvivalError> {
        let shifted = age - self.minimum_age + self.delta;
        if !shifted.is_finite() || shifted <= 0.0 {
            return Err(SurvivalError::NonFiniteInput);
        }
        Ok(shifted.ln())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SurvivalSpec {
    #[default]
    Net,
    Crude,
}

#[derive(Debug, Clone)]
pub struct SurvivalEngineInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
    pub range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(beta.len());
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]).to_owned();
            let g = block.matrix.dot(&b);
            let mut dst = grad.slice_mut(ndarray::s![block.range.clone()]);
            dst += &(2.0 * block.lambda * g);
        }
        grad
    }

    pub fn hessian(&self, dim: usize) -> Array2<f64> {
        let mut h = Array2::zeros((dim, dim));
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let r = block.range.clone();
            for (i_local, i) in r.clone().enumerate() {
                for (j_local, j) in r.clone().enumerate() {
                    h[[i, j]] += 2.0 * block.lambda * block.matrix[[i_local, j_local]];
                }
            }
        }
        h
    }

    pub fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]).to_owned();
            value += block.lambda * b.dot(&block.matrix.dot(&b));
        }
        value
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MonotonicityPenalty {
    pub lambda: f64,
    pub tolerance: f64,
}

impl Default for MonotonicityPenalty {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            tolerance: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct WorkingModelSurvival {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
    x_entry: Array2<f64>,
    x_exit: Array2<f64>,
    x_derivative: Array2<f64>,
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
}

impl WorkingModelSurvival {
    pub fn from_engine_inputs(
        inputs: SurvivalEngineInputs<'_>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        let n = inputs.age_entry.len();
        if inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.event_competing.len() != n
            || inputs.sample_weight.len() != n
            || inputs.x_entry.nrows() != n
            || inputs.x_exit.nrows() != n
            || inputs.x_derivative.nrows() != n
            || inputs.x_entry.ncols() != inputs.x_exit.ncols()
            || inputs.x_entry.ncols() != inputs.x_derivative.ncols()
        {
            return Err(SurvivalError::DimensionMismatch);
        }

        if inputs.age_entry.iter().any(|v| !v.is_finite())
            || inputs.age_exit.iter().any(|v| !v.is_finite())
            || inputs
                .sample_weight
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || inputs.x_entry.iter().any(|v| !v.is_finite())
            || inputs.x_exit.iter().any(|v| !v.is_finite())
            || inputs.x_derivative.iter().any(|v| !v.is_finite())
        {
            return Err(SurvivalError::NonFiniteInput);
        }

        Ok(Self {
            age_entry: inputs.age_entry.to_owned(),
            age_exit: inputs.age_exit.to_owned(),
            event_target: inputs.event_target.to_owned(),
            event_competing: inputs.event_competing.to_owned(),
            sample_weight: inputs.sample_weight.to_owned(),
            x_entry: inputs.x_entry.to_owned(),
            x_exit: inputs.x_exit.to_owned(),
            x_derivative: inputs.x_derivative.to_owned(),
            penalties,
            monotonicity,
            spec,
        })
    }

    pub fn update_state(&self, beta: &Array1<f64>) -> Result<WorkingState, EstimationError> {
        if beta.len() != self.x_exit.ncols() {
            return Err(EstimationError::InvalidInput(
                "survival beta dimension mismatch".to_string(),
            ));
        }

        let n = self.x_exit.nrows();
        let p = self.x_exit.ncols();

        // Royston-Parmar contract used throughout engine and adapter:
        // eta(t) = log(H(t)), where H is cumulative hazard.
        let eta_entry = self.x_entry.dot(beta);
        let eta_exit = self.x_exit.dot(beta);
        let derivative_raw = self.x_derivative.dot(beta);

        let h_entry = eta_entry.mapv(f64::exp);
        let h_exit = eta_exit.mapv(f64::exp);
        if h_entry.iter().any(|v| !v.is_finite()) || h_exit.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::InvalidInput(
                "survival linear predictor produced non-finite cumulative hazard".to_string(),
            ));
        }

        let mut nll = 0.0;
        let mut grad = Array1::<f64>::zeros(p);
        let mut h = Array2::<f64>::zeros((p, p));

        // Guard derivative terms for log(hazard) = eta + log(d eta / d t)
        let derivative_guard = self.monotonicity.tolerance.max(1e-12);
        for i in 0..n {
            let w = self.sample_weight[i];
            if w <= 0.0 {
                continue;
            }
            let entry_age = self.age_entry[i];
            let exit_age = self.age_exit[i];
            if !entry_age.is_finite() || !exit_age.is_finite() || exit_age < entry_age {
                return Err(EstimationError::InvalidInput(
                    "survival ages must be finite with age_exit >= age_entry".to_string(),
                ));
            }
            let _e_competing = self.event_competing[i];
            let d = f64::from(self.event_target[i]);

            let h_s = h_entry[i];
            let h_e = h_exit[i];
            nll += w * (h_e - h_s);

            let x_s = self.x_entry.row(i);
            let x_e = self.x_exit.row(i);

            // Interval contribution: H(exit) - H(entry)
            for j in 0..p {
                grad[j] += w * (h_e * x_e[j] - h_s * x_s[j]);
            }

            // Hessian from interval contribution.
            for r in 0..p {
                let xe_r = x_e[r];
                let xs_r = x_s[r];
                for c in 0..p {
                    h[[r, c]] += w * (h_e * xe_r * x_e[c] - h_s * xs_r * x_s[c]);
                }
            }

            if d > 0.0 {
                let deriv = derivative_raw[i];
                let safe_deriv = if deriv > derivative_guard {
                    deriv
                } else {
                    derivative_guard
                };
                let inv_deriv = if deriv > derivative_guard {
                    1.0 / deriv
                } else {
                    0.0
                };
                let d_row = self.x_derivative.row(i);
                nll += -w * (eta_exit[i] + safe_deriv.ln());

                // Event contribution gradient: -x_exit - (1/deriv)*x_derivative
                for j in 0..p {
                    grad[j] += -w * (x_e[j] + inv_deriv * d_row[j]);
                }

                // Event contribution Hessian from -log(deriv): + (1/deriv^2) d d^T
                let inv_deriv_sq = inv_deriv * inv_deriv;
                if inv_deriv_sq > 0.0 {
                    for r in 0..p {
                        let dr = d_row[r];
                        for c in 0..p {
                            h[[r, c]] += w * inv_deriv_sq * dr * d_row[c];
                        }
                    }
                }
            }
        }

        let penalty_grad = self.penalties.gradient(beta);
        let penalty_hessian = self.penalties.hessian(p);
        let penalty_dev = self.penalties.deviance(beta);

        let slope = self.x_derivative.dot(beta);
        let mut mono_dev = 0.0;
        let mut mono_grad = Array1::<f64>::zeros(p);
        let mut mono_h = Array2::<f64>::zeros((p, p));
        if self.monotonicity.lambda > 0.0 {
            for i in 0..n {
                let s = slope[i] - self.monotonicity.tolerance;
                if s < 0.0 {
                    let v = -s;
                    mono_dev += self.monotonicity.lambda * v * v;
                    let xi = self.x_derivative.row(i).to_owned();
                    mono_grad += &(2.0 * self.monotonicity.lambda * s * &xi);
                    let outer = xi
                        .view()
                        .insert_axis(Axis(1))
                        .dot(&xi.view().insert_axis(Axis(0)));
                    mono_h += &(2.0 * self.monotonicity.lambda * outer);
                }
            }
        }

        let mut total_grad = grad;
        total_grad += &penalty_grad;
        total_grad += &mono_grad;

        h += &penalty_hessian;
        h += &mono_h;
        const SURVIVAL_STABILIZATION_RIDGE: f64 = 1e-8;
        let ridge_used = SURVIVAL_STABILIZATION_RIDGE;
        for d in 0..p {
            h[[d, d]] += ridge_used;
        }
        total_grad += &beta.mapv(|v| ridge_used * v);
        let ridge_penalty = ridge_used * beta.dot(beta);

        let mut deviance = 2.0 * (nll + mono_dev);
        if matches!(self.spec, SurvivalSpec::Crude) {
            deviance *= 1.0;
        }

        Ok(WorkingState {
            eta: LinearPredictor::new(eta_exit),
            gradient: total_grad,
            hessian: h,
            deviance,
            penalty_term: penalty_dev + ridge_penalty,
            firth_log_det: None,
            firth_hat_diag: None,
            ridge_used,
        })
    }
}

impl PirlsWorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_state(beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    fn toy_penalties() -> PenaltyBlocks {
        let s = array![[2.0, 0.5], [0.5, 3.0]];
        PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: s,
            lambda: 1.7,
            range: 1..3,
        }])
    }

    #[test]
    fn penalty_hessian_matches_gradient_jacobian() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];

        let grad = penalties.gradient(&beta);
        let h = penalties.hessian(beta.len());
        let b_block = beta.slice(s![1..3]).to_owned();
        let expected = 2.0 * 1.7 * array![[2.0, 0.5], [0.5, 3.0]].dot(&b_block);

        assert!((grad[1] - expected[0]).abs() < 1e-12);
        assert!((grad[2] - expected[1]).abs() < 1e-12);
        assert!((h[[1, 1]] - 2.0 * 1.7 * 2.0).abs() < 1e-12);
        assert!((h[[1, 2]] - 2.0 * 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 1]] - 2.0 * 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 2]] - 2.0 * 1.7 * 3.0).abs() < 1e-12);
    }

    #[test]
    fn penalty_gradient_matches_deviance_finite_difference() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];
        let grad = penalties.gradient(&beta);
        let eps = 1e-7;

        for idx in 0..beta.len() {
            let mut plus = beta.clone();
            let mut minus = beta.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let fd = (penalties.deviance(&plus) - penalties.deviance(&minus)) / (2.0 * eps);
            assert!(
                (grad[idx] - fd).abs() < 1e-6,
                "gradient/deviance mismatch at idx={idx}: grad={} fd={fd}",
                grad[idx]
            );
        }
    }
}
