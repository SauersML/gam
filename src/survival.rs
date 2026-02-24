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
            || inputs.sample_weight.iter().any(|v| !v.is_finite() || *v < 0.0)
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

        let eta_entry = self.x_entry.dot(beta);
        let eta_exit = self.x_exit.dot(beta);

        let hazard_entry = eta_entry.mapv(f64::exp);
        let hazard_exit = eta_exit.mapv(f64::exp);

        let dt = &self.age_exit - &self.age_entry;
        let cum = (&hazard_entry + &hazard_exit) * 0.5 * &dt;

        let mut nll = 0.0;
        let mut d_nll_d_eta_exit = Array1::<f64>::zeros(n);
        let mut d_nll_d_eta_entry = Array1::<f64>::zeros(n);

        for i in 0..n {
            let w = self.sample_weight[i];
            let e_target = self.event_target[i] as f64;
            let _e_competing = self.event_competing[i] as f64;

            nll += w * cum[i];
            if e_target > 0.0 {
                nll += -w * eta_exit[i];
            }

            d_nll_d_eta_exit[i] = w * (0.5 * dt[i] * hazard_exit[i] - e_target);
            d_nll_d_eta_entry[i] = w * (0.5 * dt[i] * hazard_entry[i]);
        }

        let mut grad = self.x_exit.t().dot(&d_nll_d_eta_exit);
        grad += &self.x_entry.t().dot(&d_nll_d_eta_entry);

        let mut h = Array2::<f64>::zeros((p, p));
        let w_exit = (&self.sample_weight * &dt * &hazard_exit) * 0.5;
        let w_entry = (&self.sample_weight * &dt * &hazard_entry) * 0.5;

        let mut wx_exit = self.x_exit.clone();
        for (mut row, &wi) in wx_exit.axis_iter_mut(Axis(0)).zip(w_exit.iter()) {
            row *= wi.sqrt();
        }
        let mut wx_entry = self.x_entry.clone();
        for (mut row, &wi) in wx_entry.axis_iter_mut(Axis(0)).zip(w_entry.iter()) {
            row *= wi.sqrt();
        }
        h += &wx_exit.t().dot(&wx_exit);
        h += &wx_entry.t().dot(&wx_entry);

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
        for d in 0..p {
            h[[d, d]] += 1e-8;
        }

        let mut deviance = 2.0 * (nll + penalty_dev + mono_dev);
        if matches!(self.spec, SurvivalSpec::Crude) {
            deviance *= 1.0;
        }

        Ok(WorkingState {
            eta: LinearPredictor::new(eta_exit),
            gradient: total_grad * 2.0,
            hessian: h * 2.0,
            deviance,
            penalty_term: penalty_dev,
            firth_log_det: None,
            firth_hat_diag: None,
            ridge_used: 1e-8,
        })
    }
}

impl PirlsWorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_state(beta)
    }
}
