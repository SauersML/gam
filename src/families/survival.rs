use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerCholesky, FaerEigh, fast_atv, fast_xt_diag_x, fast_xt_diag_y};
use crate::linalg::utils::{default_slq_parameters, stochastic_lanczos_logdet_spd};
use crate::pirls::{LinearInequalityConstraints, WorkingModel as PirlsWorkingModel, WorkingState};
use crate::types::{Coefficients, LinearPredictor};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::OnceLock;
use thiserror::Error;

const SURVIVAL_TRACE_SOLVE_TARGET_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("input dimensions are inconsistent")]
    DimensionMismatch,
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("survival spec '{0}' is not supported by the one-hazard survival engine")]
    UnsupportedSpec(&'static str),
    #[error("crude risk integration setup is invalid")]
    InvalidIntegrationSetup,
    #[error("cumulative hazard must be nondecreasing")]
    NonMonotoneCumulativeHazard,
    #[error("instantaneous hazard must stay strictly positive during integration")]
    NonPositiveHazard,
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
    pub sampleweight: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct SurvivalTimeCovarInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sampleweight: ArrayView1<'a, f64>,
    pub time_entry: ArrayView2<'a, f64>,
    pub time_exit: ArrayView2<'a, f64>,
    pub time_derivative: ArrayView2<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct SurvivalBaselineOffsets<'a> {
    /// Baseline target contribution to eta at entry time: eta_target(t_entry).
    pub eta_entry: ArrayView1<'a, f64>,
    /// Baseline target contribution to eta at exit time: eta_target(t_exit).
    pub eta_exit: ArrayView1<'a, f64>,
    /// Baseline target contribution to d eta / d t at exit: eta_target'(t_exit).
    ///
    /// This is used in event terms where log-hazard requires
    /// log(d eta / d t). By threading this as an explicit offset, we get
    /// "parametric default + spline deviation" behavior:
    /// - strong penalty => deviation ~ 0 => model collapses to baseline target,
    /// - weak penalty   => deviation can bend away where data supports it.
    pub derivative_exit: ArrayView1<'a, f64>,
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
            let b = beta.slice(ndarray::s![block.range.clone()]);
            let g = block.matrix.dot(&b);
            let mut dst = grad.slice_mut(ndarray::s![block.range.clone()]);
            dst += &(block.lambda * g);
        }
        grad
    }

    pub fn hessian(&self, dim: usize) -> Array2<f64> {
        let mut h = Array2::zeros((dim, dim));
        self.addhessian_inplace(&mut h);
        h
    }

    pub fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]);
            value += 0.5 * block.lambda * b.dot(&block.matrix.dot(&b));
        }
        value
    }

    pub fn addhessian_inplace(&self, h: &mut Array2<f64>) {
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let start = block.range.start;
            let end = block.range.end;
            h.slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(block.lambda, &block.matrix);
        }
    }
}

fn compress_positive_collinear_constraints(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> LinearInequalityConstraints {
    const SCALE_TOL: f64 = 1e-14;
    const KEY_TOL: f64 = 1e-8;

    let mut grouped: BTreeMap<Vec<i64>, (Vec<f64>, f64)> = BTreeMap::new();
    let mut fallbackrows: Vec<(Vec<f64>, f64)> = Vec::new();

    for i in 0..a.nrows() {
        let row = a.row(i);
        let scale = row.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !scale.is_finite() || scale <= SCALE_TOL {
            if b[i] > 0.0 {
                fallbackrows.push((row.to_vec(), b[i]));
            }
            continue;
        }

        let normalizedrow: Vec<f64> = row
            .iter()
            .map(|&v| {
                let scaled = v / scale;
                if scaled.abs() <= KEY_TOL { 0.0 } else { scaled }
            })
            .collect();
        let normalized_rhs = b[i] / scale;
        let key: Vec<i64> = normalizedrow
            .iter()
            .map(|&v| (v / KEY_TOL).round() as i64)
            .collect();

        match grouped.get_mut(&key) {
            Some((_, rhs_max)) => {
                if normalized_rhs > *rhs_max {
                    *rhs_max = normalized_rhs;
                }
            }
            None => {
                grouped.insert(key, (normalizedrow, normalized_rhs));
            }
        }
    }

    let nrows = grouped.len() + fallbackrows.len();
    let n_cols = a.ncols();
    let mut a_out = Array2::<f64>::zeros((nrows, n_cols));
    let mut b_out = Array1::<f64>::zeros(nrows);

    let mut outrow = 0usize;
    for (_, (row, rhs)) in grouped {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[outrow, j]] = value;
        }
        b_out[outrow] = rhs;
        outrow += 1;
    }
    for (row, rhs) in fallbackrows {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[outrow, j]] = value;
        }
        b_out[outrow] = rhs;
        outrow += 1;
    }

    LinearInequalityConstraints { a: a_out, b: b_out }
}

#[derive(Debug, Clone, Copy)]
pub struct MonotonicityPenalty {
    pub tolerance: f64,
}

impl Default for MonotonicityPenalty {
    fn default() -> Self {
        Self { tolerance: 0.0 }
    }
}

#[derive(Debug, Clone)]
enum SurvivalDesign {
    Flat {
        x_entry: Array2<f64>,
        x_exit: Array2<f64>,
        x_derivative: Array2<f64>,
    },
    TimeCovariateShared {
        time_entry: Array2<f64>,
        time_exit: Array2<f64>,
        time_derivative: Array2<f64>,
        covariates: Array2<f64>,
    },
}

impl SurvivalDesign {
    fn p_total(&self) -> usize {
        match self {
            Self::Flat { x_exit, .. } => x_exit.ncols(),
            Self::TimeCovariateShared {
                time_exit,
                covariates,
                ..
            } => time_exit.ncols() + covariates.ncols(),
        }
    }

    fn design_dot(&self, time_mat: &Array2<f64>, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Flat { .. } => time_mat.dot(beta),
            Self::TimeCovariateShared { covariates, .. } => {
                let p_time = time_mat.ncols();
                let mut out = time_mat.dot(&beta.slice(ndarray::s![..p_time]));
                if covariates.ncols() > 0 {
                    out += &covariates.dot(&beta.slice(ndarray::s![p_time..]));
                }
                out
            }
        }
    }

    fn fill_row(&self, time_mat: &Array2<f64>, i: usize, out: &mut [f64]) {
        match self {
            Self::Flat { .. } => {
                for (dst, &src) in out.iter_mut().zip(time_mat.row(i).iter()) {
                    *dst = src;
                }
            }
            Self::TimeCovariateShared { covariates, .. } => {
                let p_time = time_mat.ncols();
                for j in 0..p_time {
                    out[j] = time_mat[[i, j]];
                }
                for j in 0..covariates.ncols() {
                    out[p_time + j] = covariates[[i, j]];
                }
            }
        }
    }

    fn transpose_vector_multiply(
        &self,
        time_mat: &Array2<f64>,
        vector: &Array1<f64>,
        include_covariates: bool,
    ) -> Array1<f64> {
        match self {
            Self::Flat { .. } => fast_atv(time_mat, vector),
            Self::TimeCovariateShared { covariates, .. } => {
                let p_time = time_mat.ncols();
                let mut out = Array1::<f64>::zeros(p_time + covariates.ncols());
                out.slice_mut(ndarray::s![..p_time])
                    .assign(&fast_atv(time_mat, vector));
                if include_covariates && covariates.ncols() > 0 {
                    out.slice_mut(ndarray::s![p_time..])
                        .assign(&fast_atv(covariates, vector));
                }
                out
            }
        }
    }
}

/// Pre-allocated workspace buffers for `update_state` to avoid per-iteration allocations.
#[derive(Debug, Clone)]
struct SurvivalWorkspace {
    w_event: Array1<f64>,
    w_event_inv_deriv: Array1<f64>,
    w_event_outer: Array1<f64>,
    w_hess_exit: Array1<f64>,
    w_hess_entry: Array1<f64>,
}

impl SurvivalWorkspace {
    fn new(n: usize) -> Self {
        Self {
            w_event: Array1::zeros(n),
            w_event_inv_deriv: Array1::zeros(n),
            w_event_outer: Array1::zeros(n),
            w_hess_exit: Array1::zeros(n),
            w_hess_entry: Array1::zeros(n),
        }
    }

    fn reset(&mut self, n: usize) {
        if self.w_event.len() != n {
            *self = Self::new(n);
        } else {
            self.w_event.fill(0.0);
            self.w_event_inv_deriv.fill(0.0);
            self.w_event_outer.fill(0.0);
            self.w_hess_exit.fill(0.0);
            self.w_hess_entry.fill(0.0);
        }
    }
}

#[derive(Debug)]
pub struct WorkingModelSurvival {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    entry_at_origin: Array1<bool>,
    event_target: Array1<u8>,
    sampleweight: Array1<f64>,
    design: SurvivalDesign,
    offset_eta_entry: Array1<f64>,
    offset_eta_exit: Array1<f64>,
    offset_derivative_exit: Array1<f64>,
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    structurally_monotonic: bool,
    structural_time_columns: usize,
    workspace: std::sync::Mutex<SurvivalWorkspace>,
}

impl Clone for WorkingModelSurvival {
    fn clone(&self) -> Self {
        let workspace = self.workspace.lock().unwrap().clone();
        Self {
            age_entry: self.age_entry.clone(),
            age_exit: self.age_exit.clone(),
            entry_at_origin: self.entry_at_origin.clone(),
            event_target: self.event_target.clone(),
            sampleweight: self.sampleweight.clone(),
            design: self.design.clone(),
            offset_eta_entry: self.offset_eta_entry.clone(),
            offset_eta_exit: self.offset_eta_exit.clone(),
            offset_derivative_exit: self.offset_derivative_exit.clone(),
            penalties: self.penalties.clone(),
            monotonicity: self.monotonicity.clone(),
            structurally_monotonic: self.structurally_monotonic,
            structural_time_columns: self.structural_time_columns,
            workspace: std::sync::Mutex::new(workspace),
        }
    }
}

impl WorkingModelSurvival {
    const LOG_F64_MAX: f64 = 709.782712893384;

    #[inline]
    fn scaled_exp_component(log_scale: f64, base: f64) -> Result<f64, EstimationError> {
        if base == 0.0 {
            return Ok(0.0);
        }
        let log_abs = log_scale + base.abs().ln();
        if !log_abs.is_finite() {
            return Err(EstimationError::InvalidInput(
                "survival interval term produced non-finite log-magnitude".to_string(),
            ));
        }
        if log_abs > Self::LOG_F64_MAX {
            return Err(EstimationError::InvalidInput(format!(
                "survival interval term exceeds f64 range (log-magnitude={log_abs:.3e})"
            )));
        }
        Ok(base.signum() * log_abs.exp())
    }

    fn row_derivative_constraint_lower_bound(&self, _: usize) -> f64 {
        self.derivative_guard()
    }

    fn coefficient_dim(&self) -> usize {
        self.design.p_total()
    }

    fn nrows(&self) -> usize {
        self.sampleweight.len()
    }

    fn entry_dot(&self, beta: &Array1<f64>) -> Array1<f64> {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_entry, .. } => x_entry,
            SurvivalDesign::TimeCovariateShared { time_entry, .. } => time_entry,
        };
        self.design.design_dot(time_mat, beta)
    }

    fn exit_dot(&self, beta: &Array1<f64>) -> Array1<f64> {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_exit, .. } => x_exit,
            SurvivalDesign::TimeCovariateShared { time_exit, .. } => time_exit,
        };
        self.design.design_dot(time_mat, beta)
    }

    fn derivative_dot(&self, beta: &Array1<f64>) -> Array1<f64> {
        match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => x_derivative.dot(beta),
            SurvivalDesign::TimeCovariateShared {
                time_derivative, ..
            } => time_derivative.dot(&beta.slice(ndarray::s![..time_derivative.ncols()])),
        }
    }

    fn fill_entry_row(&self, i: usize, out: &mut [f64]) {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_entry, .. } => x_entry,
            SurvivalDesign::TimeCovariateShared { time_entry, .. } => time_entry,
        };
        self.design.fill_row(time_mat, i, out);
    }

    fn fill_exit_row(&self, i: usize, out: &mut [f64]) {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_exit, .. } => x_exit,
            SurvivalDesign::TimeCovariateShared { time_exit, .. } => time_exit,
        };
        self.design.fill_row(time_mat, i, out);
    }

    fn fill_derivative_row(&self, i: usize, out: &mut [f64]) {
        match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => {
                for (dst, &src) in out.iter_mut().zip(x_derivative.row(i).iter()) {
                    *dst = src;
                }
            }
            SurvivalDesign::TimeCovariateShared {
                time_derivative, ..
            } => {
                let p_time = time_derivative.ncols();
                for j in 0..p_time {
                    out[j] = time_derivative[[i, j]];
                }
                for dst in out.iter_mut().skip(p_time) {
                    *dst = 0.0;
                }
            }
        }
    }

    fn derivative_transpose_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => x_derivative,
            SurvivalDesign::TimeCovariateShared {
                time_derivative, ..
            } => time_derivative,
        };
        self.design
            .transpose_vector_multiply(time_mat, vector, false)
    }

    fn exit_transpose_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_exit, .. } => x_exit,
            SurvivalDesign::TimeCovariateShared { time_exit, .. } => time_exit,
        };
        self.design
            .transpose_vector_multiply(time_mat, vector, true)
    }

    fn derivative_xt_diag_x(&self, weights: &Array1<f64>) -> Array2<f64> {
        match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => fast_xt_diag_x(x_derivative, weights),
            SurvivalDesign::TimeCovariateShared {
                time_derivative,
                covariates,
                ..
            } => {
                let p_time = time_derivative.ncols();
                let p_cov = covariates.ncols();
                let mut out = Array2::<f64>::zeros((p_time + p_cov, p_time + p_cov));
                let time_block = fast_xt_diag_x(time_derivative, weights);
                out.slice_mut(ndarray::s![..p_time, ..p_time])
                    .assign(&time_block);
                out
            }
        }
    }

    /// Compute the full p×p Hessian contribution for the interval terms:
    ///   H = X_exit^T diag(w_exit) X_exit - X_entry^T diag(w_entry) X_entry
    /// using faer-accelerated BLAS on the stored design matrix blocks.
    fn interval_hessian_blas(&self, w_exit: &Array1<f64>, w_entry: &Array1<f64>) -> Array2<f64> {
        match &self.design {
            SurvivalDesign::Flat {
                x_entry, x_exit, ..
            } => {
                let mut h = fast_xt_diag_x(x_exit, w_exit);
                h -= &fast_xt_diag_x(x_entry, w_entry);
                h
            }
            SurvivalDesign::TimeCovariateShared {
                time_entry,
                time_exit,
                covariates,
                ..
            } => {
                let p_time = time_exit.ncols();
                let p_cov = covariates.ncols();
                let p = p_time + p_cov;
                let mut h = Array2::<f64>::zeros((p, p));
                // time-time block: T_exit^T W_exit T_exit - T_entry^T W_entry T_entry
                let tt = {
                    let mut block = fast_xt_diag_x(time_exit, w_exit);
                    block -= &fast_xt_diag_x(time_entry, w_entry);
                    block
                };
                h.slice_mut(ndarray::s![..p_time, ..p_time]).assign(&tt);
                if p_cov > 0 {
                    // time-cov block: T_exit^T W_exit C - T_entry^T W_entry C
                    let tc = {
                        let mut block = fast_xt_diag_y(time_exit, w_exit, covariates);
                        block -= &fast_xt_diag_y(time_entry, w_entry, covariates);
                        block
                    };
                    h.slice_mut(ndarray::s![..p_time, p_time..]).assign(&tc);
                    h.slice_mut(ndarray::s![p_time.., ..p_time]).assign(&tc.t());
                    // cov-cov block: C^T (W_exit - W_entry) C
                    let w_diff = w_exit - w_entry;
                    let cc = fast_xt_diag_x(covariates, &w_diff);
                    h.slice_mut(ndarray::s![p_time.., p_time..]).assign(&cc);
                }
                h
            }
        }
    }

    /// Compute the gradient contribution for the interval terms:
    ///   grad = X_exit^T w_exit_grad - X_entry^T w_entry_grad
    fn interval_gradient_blas(
        &self,
        w_exit_grad: &Array1<f64>,
        w_entry_grad: &Array1<f64>,
    ) -> Array1<f64> {
        match &self.design {
            SurvivalDesign::Flat {
                x_entry, x_exit, ..
            } => {
                let mut g = fast_atv(x_exit, w_exit_grad);
                g -= &fast_atv(x_entry, w_entry_grad);
                g
            }
            SurvivalDesign::TimeCovariateShared {
                time_entry,
                time_exit,
                covariates,
                ..
            } => {
                let p_time = time_exit.ncols();
                let p_cov = covariates.ncols();
                let mut g = Array1::<f64>::zeros(p_time + p_cov);
                {
                    let mut gt = fast_atv(time_exit, w_exit_grad);
                    gt -= &fast_atv(time_entry, w_entry_grad);
                    g.slice_mut(ndarray::s![..p_time]).assign(&gt);
                }
                if p_cov > 0 {
                    let w_diff = w_exit_grad - w_entry_grad;
                    g.slice_mut(ndarray::s![p_time..])
                        .assign(&fast_atv(covariates, &w_diff));
                }
                g
            }
        }
    }

    fn stabilized_structural_derivative(&self, deriv: f64) -> Option<f64> {
        const STRUCTURAL_MONO_ROUNDOFF_TOL: f64 = 1e-7;
        if !self.structurally_monotonic {
            return None;
        }
        if deriv >= 1e-12 {
            return Some(deriv);
        }
        if deriv >= -STRUCTURAL_MONO_ROUNDOFF_TOL {
            return Some(1e-12);
        }
        None
    }

    fn validate_penalties(
        penalties: &PenaltyBlocks,
        coefficient_dim: usize,
    ) -> Result<(), SurvivalError> {
        for block in &penalties.blocks {
            if !block.lambda.is_finite() || block.lambda < 0.0 {
                return Err(SurvivalError::NonFiniteInput);
            }
            if block.range.start > block.range.end || block.range.end > coefficient_dim {
                return Err(SurvivalError::DimensionMismatch);
            }
            let block_dim = block.range.end - block.range.start;
            if block.matrix.nrows() != block_dim || block.matrix.ncols() != block_dim {
                return Err(SurvivalError::DimensionMismatch);
            }
            if block.matrix.iter().any(|v| !v.is_finite()) {
                return Err(SurvivalError::NonFiniteInput);
            }
        }
        Ok(())
    }

    fn spd_logdet(hessian: &Array2<f64>) -> Result<f64, EstimationError> {
        if let Ok(chol) = hessian.cholesky(Side::Lower) {
            let l = chol.lower_triangular();
            return Ok(2.0 * (0..l.nrows()).map(|i| l[[i, i]].ln()).sum::<f64>());
        }

        let (eval, _) = match hessian.eigh(Side::Lower) {
            Ok(out) => out,
            Err(_) => {
                let (probes, steps) = default_slq_parameters(hessian.nrows());
                return stochastic_lanczos_logdet_spd(hessian, probes, steps, 97)
                    .map_err(EstimationError::InvalidInput);
            }
        };
        let max_eval = eval.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let tol = (max_eval * 1e-12).max(1e-14);
        Ok(eval.iter().filter(|&&v| v > tol).map(|&v| v.ln()).sum())
    }

    fn derivative_guard(&self) -> f64 {
        if self.structurally_monotonic {
            // I-spline basis is monotone by construction when coefficients ≥ 0.
            // A derivative of zero (flat hazard) is valid, so the guard only
            // rejects genuinely negative derivatives from floating-point noise.
            return 0.0;
        }
        self.monotonicity.tolerance.max(0.0)
    }

    fn derivative_guard_numerical(&self) -> f64 {
        let derivative_guard = self.derivative_guard();
        if derivative_guard <= 0.0 {
            // For structural monotonicity (guard = 0), allow tiny negative
            // values from floating-point arithmetic.
            -1e-10
        } else {
            (derivative_guard - (1e-10_f64).min(0.01 * derivative_guard)).max(1e-12)
        }
    }

    fn interval_increment_guard(&self, h_entry: f64, h_exit: f64) -> f64 {
        let scale = h_entry.abs().max(h_exit.abs()).max(1.0);
        1e-10 * scale
    }

    pub fn structural_time_initial_coefficient_floor(
        &self,
        derivative_target_floor: f64,
    ) -> Option<f64> {
        if !self.structurally_monotonic || self.structural_time_columns == 0 {
            return None;
        }
        let derivative_target_floor = derivative_target_floor.max(self.derivative_guard());
        let mut min_uniform_time_coef = 0.0_f64;
        let p = self.coefficient_dim();
        let mut row = vec![0.0_f64; p];

        for i in 0..self.nrows() {
            if self.sampleweight[i] <= 0.0 {
                continue;
            }
            self.fill_derivative_row(i, &mut row);
            let row_sum = row[0..self.structural_time_columns]
                .iter()
                .copied()
                .sum::<f64>();
            if row_sum <= 1e-12 {
                continue;
            }
            let rhs = (derivative_target_floor - self.offset_derivative_exit[i]).max(0.0);
            min_uniform_time_coef = min_uniform_time_coef.max(rhs / row_sum);
        }

        Some(min_uniform_time_coef)
    }

    pub fn monotonicity_linear_constraints(&self) -> Option<LinearInequalityConstraints> {
        let p = self.coefficient_dim();
        const DERIVATIVE_ROW_NORM_TOL: f64 = 1e-12;
        if p == 0 {
            return None;
        }
        let mut derivative_row = vec![0.0_f64; p];
        let activerows: Vec<usize> = (0..self.nrows())
            .filter(|&i| {
                self.fill_derivative_row(i, &mut derivative_row);
                self.sampleweight[i] > 0.0
                    && derivative_row
                        .iter()
                        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                        > DERIVATIVE_ROW_NORM_TOL
            })
            .collect();
        if activerows.is_empty() {
            return None;
        }
        let mut a = Array2::<f64>::zeros((activerows.len(), p));
        let mut b = Array1::<f64>::zeros(activerows.len());
        for (r, &i) in activerows.iter().enumerate() {
            self.fill_derivative_row(i, &mut derivative_row);
            for j in 0..p {
                a[[r, j]] = derivative_row[j];
            }
            b[r] = self.row_derivative_constraint_lower_bound(i) - self.offset_derivative_exit[i];
        }
        if self.structurally_monotonic {
            Some(LinearInequalityConstraints { a, b })
        } else {
            Some(compress_positive_collinear_constraints(&a, &b))
        }
    }

    pub fn from_engine_inputs(
        inputs: SurvivalEngineInputs<'_>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        Self::from_engine_inputswith_offsets(inputs, None, penalties, monotonicity, spec)
    }

    fn validate_offsets(
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        n: usize,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), SurvivalError> {
        if let Some(off) = offsets {
            if off.eta_entry.len() != n || off.eta_exit.len() != n || off.derivative_exit.len() != n
            {
                return Err(SurvivalError::DimensionMismatch);
            }
            if off.eta_entry.iter().any(|v| !v.is_finite())
                || off.eta_exit.iter().any(|v| !v.is_finite())
                || off.derivative_exit.iter().any(|v| !v.is_finite())
            {
                return Err(SurvivalError::NonFiniteInput);
            }
            Ok((
                off.eta_entry.to_owned(),
                off.eta_exit.to_owned(),
                off.derivative_exit.to_owned(),
            ))
        } else {
            Ok((Array1::zeros(n), Array1::zeros(n), Array1::zeros(n)))
        }
    }

    fn validate_common_inputs(
        age_entry: &ArrayView1<f64>,
        age_exit: &ArrayView1<f64>,
        event_target: &ArrayView1<u8>,
        event_competing: &ArrayView1<u8>,
        sampleweight: &ArrayView1<f64>,
    ) -> Result<(), SurvivalError> {
        if age_entry.iter().any(|v| !v.is_finite())
            || age_exit.iter().any(|v| !v.is_finite())
            || sampleweight.iter().any(|v| !v.is_finite() || *v < 0.0)
            || event_target.iter().any(|&v| v > 1)
            || event_competing.iter().any(|&v| v > 1)
            || event_target
                .iter()
                .zip(event_competing.iter())
                .any(|(&target, &competing)| target > 0 && competing > 0)
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        Ok(())
    }

    fn finish_construction(
        age_entry: ArrayView1<f64>,
        age_exit: ArrayView1<f64>,
        event_target: ArrayView1<u8>,
        sampleweight: ArrayView1<f64>,
        design: SurvivalDesign,
        offset_eta_entry: Array1<f64>,
        offset_eta_exit: Array1<f64>,
        offset_derivative_exit: Array1<f64>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
    ) -> Self {
        let n = age_entry.len();
        Self {
            age_entry: age_entry.to_owned(),
            age_exit: age_exit.to_owned(),
            entry_at_origin: age_entry.mapv(|t| t <= 1e-8),
            event_target: event_target.to_owned(),
            sampleweight: sampleweight.to_owned(),
            design,
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
            structurally_monotonic: false,
            structural_time_columns: 0,
            workspace: std::sync::Mutex::new(SurvivalWorkspace::new(n)),
        }
    }

    pub fn from_engine_inputswith_offsets(
        inputs: SurvivalEngineInputs<'_>,
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        if spec == SurvivalSpec::Crude {
            return Err(SurvivalError::UnsupportedSpec("crude"));
        }
        let n = inputs.age_entry.len();
        let p = inputs.x_entry.ncols();
        if inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.event_competing.len() != n
            || inputs.sampleweight.len() != n
            || inputs.x_entry.nrows() != n
            || inputs.x_exit.nrows() != n
            || inputs.x_derivative.nrows() != n
            || inputs.x_entry.ncols() != inputs.x_exit.ncols()
            || inputs.x_entry.ncols() != inputs.x_derivative.ncols()
        {
            return Err(SurvivalError::DimensionMismatch);
        }
        Self::validate_penalties(&penalties, p)?;
        Self::validate_common_inputs(
            &inputs.age_entry,
            &inputs.age_exit,
            &inputs.event_target,
            &inputs.event_competing,
            &inputs.sampleweight,
        )?;
        if inputs.x_entry.iter().any(|v| !v.is_finite())
            || inputs.x_exit.iter().any(|v| !v.is_finite())
            || inputs.x_derivative.iter().any(|v| !v.is_finite())
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        let (offset_eta_entry, offset_eta_exit, offset_derivative_exit) =
            Self::validate_offsets(offsets, n)?;

        Ok(Self::finish_construction(
            inputs.age_entry,
            inputs.age_exit,
            inputs.event_target,
            inputs.sampleweight,
            SurvivalDesign::Flat {
                x_entry: inputs.x_entry.to_owned(),
                x_exit: inputs.x_exit.to_owned(),
                x_derivative: inputs.x_derivative.to_owned(),
            },
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
        ))
    }

    pub fn from_time_covariate_inputswith_offsets(
        inputs: SurvivalTimeCovarInputs<'_>,
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        if spec == SurvivalSpec::Crude {
            return Err(SurvivalError::UnsupportedSpec("crude"));
        }
        let n = inputs.age_entry.len();
        let p_time = inputs.time_entry.ncols();
        let p_cov = inputs.covariates.ncols();
        let p = p_time + p_cov;
        if inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.event_competing.len() != n
            || inputs.sampleweight.len() != n
            || inputs.time_entry.nrows() != n
            || inputs.time_exit.nrows() != n
            || inputs.time_derivative.nrows() != n
            || inputs.covariates.nrows() != n
            || inputs.time_entry.ncols() != inputs.time_exit.ncols()
            || inputs.time_entry.ncols() != inputs.time_derivative.ncols()
        {
            return Err(SurvivalError::DimensionMismatch);
        }
        Self::validate_penalties(&penalties, p)?;
        Self::validate_common_inputs(
            &inputs.age_entry,
            &inputs.age_exit,
            &inputs.event_target,
            &inputs.event_competing,
            &inputs.sampleweight,
        )?;
        if inputs.time_entry.iter().any(|v| !v.is_finite())
            || inputs.time_exit.iter().any(|v| !v.is_finite())
            || inputs.time_derivative.iter().any(|v| !v.is_finite())
            || inputs.covariates.iter().any(|v| !v.is_finite())
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        let (offset_eta_entry, offset_eta_exit, offset_derivative_exit) =
            Self::validate_offsets(offsets, n)?;

        Ok(Self::finish_construction(
            inputs.age_entry,
            inputs.age_exit,
            inputs.event_target,
            inputs.sampleweight,
            SurvivalDesign::TimeCovariateShared {
                time_entry: inputs.time_entry.to_owned(),
                time_exit: inputs.time_exit.to_owned(),
                time_derivative: inputs.time_derivative.to_owned(),
                covariates: inputs.covariates.to_owned(),
            },
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
        ))
    }

    /// Enable/disable monotonic time-block enforcement metadata.
    ///
    /// Monotonicity is enforced through linear inequality constraints on the
    /// derivative design; enabling this records how many leading time columns
    /// belong to that constrained block.
    pub fn set_structural_monotonicity(
        &mut self,
        enabled: bool,
        time_columns: usize,
    ) -> Result<(), EstimationError> {
        let p = self.coefficient_dim();
        if time_columns > p {
            return Err(EstimationError::InvalidInput(format!(
                "structural time columns {} exceed coefficient dimension {}",
                time_columns, p
            )));
        }
        if enabled && time_columns == 0 {
            return Err(EstimationError::InvalidInput(
                "structural monotonicity requires at least one time column".to_string(),
            ));
        }
        if enabled {
            const STRUCTURAL_DERIV_TOL: f64 = 1e-12;
            let mut derivative_row = vec![0.0_f64; p];
            for i in 0..self.nrows() {
                self.fill_derivative_row(i, &mut derivative_row);
                for j in 0..time_columns {
                    let v = derivative_row[j];
                    if v < -STRUCTURAL_DERIV_TOL {
                        return Err(EstimationError::InvalidInput(format!(
                            "structural monotonicity requires nonnegative time-derivative basis entries; found x_derivative[{i},{j}]={v:.3e}"
                        )));
                    }
                }
                for j in time_columns..p {
                    let v = derivative_row[j];
                    if v.abs() > STRUCTURAL_DERIV_TOL {
                        return Err(EstimationError::InvalidInput(format!(
                            "structural monotonicity requires zero derivative contribution outside the time block; found x_derivative[{i},{j}]={v:.3e}"
                        )));
                    }
                }
            }
        }
        self.structurally_monotonic = enabled;
        self.structural_time_columns = if enabled { time_columns } else { 0 };
        Ok(())
    }

    pub fn update_state(&self, beta: &Array1<f64>) -> Result<WorkingState, EstimationError> {
        if beta.len() != self.coefficient_dim() {
            return Err(EstimationError::InvalidInput(
                "survival beta dimension mismatch".to_string(),
            ));
        }

        let n = self.nrows();
        let p = self.coefficient_dim();

        // Royston-Parmar contract used throughout the engine:
        //   eta(t) = log(H(t)), where H(t) is cumulative hazard.
        //
        // With row-vectors (per subject i):
        //   a1_i^T := x_exit_i^T,  a0_i^T := x_entry_i^T,  d_i^T := x_derivative_i^T
        // and scalars:
        //   eta1_i = a1_i^T beta,  eta0_i = a0_i^T beta,  s_i = d_i^T beta.
        //
        // The per-subject negative log-likelihood used below is
        //   NLL_i(beta) = exp(eta1_i) - exp(eta0_i) - delta_i * (eta1_i + log(s_i)),
        // with delta_i = event_target_i.
        //
        // This is exactly the form whose derivatives are:
        //   grad_i = exp(eta1_i) a1_i - exp(eta0_i) a0_i - delta_i * (a1_i + d_i / s_i)
        //   Hess_i = exp(eta1_i) a1_i a1_i^T - exp(eta0_i) a0_i a0_i^T
        //            + delta_i * (d_i d_i^T) / s_i^2.
        //
        // Monotonicity is enforced through linear inequality constraints on the
        // derivative design. This keeps the baseline smoothing penalty on the
        // actual spline coefficients and preserves zero-deviation as beta=0.
        //
        // The loop below computes exact beta-space derivatives and then adds penalties.
        // Total predictor = target offset + learned deviation.
        // This is the same architecture used for flexible binary links:
        // principled default, plus penalized wiggle/deviation.
        let eta_entry = self.entry_dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.exit_dot(beta) + &self.offset_eta_exit;
        let derivative_raw = self.derivative_dot(beta) + &self.offset_derivative_exit;

        let mut nll = 0.0;
        let derivative_guard = self.derivative_guard();
        let derivative_guard_numerical = self.derivative_guard_numerical();
        let mut workspace = self.workspace.lock().unwrap();
        workspace.reset(n);
        let SurvivalWorkspace {
            w_event,
            w_event_inv_deriv,
            w_event_outer,
            w_hess_exit,
            w_hess_entry,
        } = &mut *workspace;

        // Phase 1: Scalar loop — compute per-observation weights, NLL, validation.
        for i in 0..n {
            let w = self.sampleweight[i];
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
            let d = f64::from(self.event_target[i]);

            let has_entry_interval = !self.entry_at_origin[i];
            let interval_scale = if has_entry_interval {
                eta_exit[i].max(eta_entry[i])
            } else {
                eta_exit[i]
            };
            let h_e_scaled = (eta_exit[i] - interval_scale).exp();
            let h_s_scaled = if has_entry_interval {
                (eta_entry[i] - interval_scale).exp()
            } else {
                0.0
            };
            let interval_scaled = h_e_scaled - h_s_scaled;
            let interval = Self::scaled_exp_component(interval_scale, interval_scaled)?;
            let deriv = self
                .stabilized_structural_derivative(derivative_raw[i])
                .unwrap_or(derivative_raw[i]);
            if !deriv.is_finite() {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "survival monotonicity violated at row {}: d_eta/dt={:.3e} <= tolerance={:.3e}",
                    i, deriv, derivative_guard
                )));
            }
            if deriv < derivative_guard_numerical {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "survival monotonicity violated at row {}: d_eta/dt={:.3e} <= tolerance={:.3e}",
                    i, deriv, derivative_guard
                )));
            }
            if has_entry_interval {
                let increment_guard = self.interval_increment_guard(h_s_scaled, h_e_scaled);
                if interval_scaled + increment_guard < 0.0 {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "survival cumulative hazard decreased over row {}: H(exit)-H(entry)={:.6e}",
                        i, interval
                    )));
                }
            }
            nll += w * interval;

            // Per-observation weights for BLAS phase.
            // scaled_exp_component(interval_scale, h_e_scaled * x[r]) = exp(interval_scale) * h_e_scaled * x[r]
            // so the Hessian weight is w * exp(interval_scale) * h_e_scaled = w * exp(eta_exit).
            let w_exit_i = w * eta_exit[i].exp();
            let w_entry_i = if has_entry_interval {
                w * eta_entry[i].exp()
            } else {
                0.0
            };
            if !w_exit_i.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "survival interval term exceeds f64 range at row {i} (w*exp(eta_exit)={w_exit_i:.3e})"
                )));
            }
            w_hess_exit[i] = w_exit_i;
            w_hess_entry[i] = w_entry_i;

            if d > 0.0 {
                let inv_deriv = 1.0 / deriv;
                nll += -w * (eta_exit[i] + deriv.ln());
                w_event[i] = w;
                w_event_inv_deriv[i] = w * inv_deriv;
                w_event_outer[i] = w * inv_deriv * inv_deriv;
            }
        }

        // Phase 2: BLAS-accelerated Hessian and gradient via faer.
        //   H_interval = X_exit^T diag(w_exit) X_exit - X_entry^T diag(w_entry) X_entry
        //   grad_interval = X_exit^T w_exit - X_entry^T w_entry
        let mut h = self.interval_hessian_blas(&w_hess_exit, &w_hess_entry);
        let mut grad = self.interval_gradient_blas(&w_hess_exit, &w_hess_entry);

        grad -= &self.exit_transpose_vector_multiply(&w_event);
        grad -= &self.derivative_transpose_vector_multiply(&w_event_inv_deriv);

        h += &self.derivative_xt_diag_x(&w_event_outer);

        let penaltygrad = self.penalties.gradient(beta);
        let penalty_dev = self.penalties.deviance(beta);

        let mut totalgrad = grad;
        totalgrad += &penaltygrad;

        self.penalties.addhessian_inplace(&mut h);
        const SURVIVAL_STABILIZATION_RIDGE: f64 = 1e-8;
        let ridge_used = SURVIVAL_STABILIZATION_RIDGE;
        for d in 0..p {
            h[[d, d]] += ridge_used;
        }
        totalgrad += &beta.mapv(|v| ridge_used * v);
        // Keep scalar objective term consistent with:
        //   grad += ridge * beta,  Hess += ridge * I
        // which correspond to 0.5 * ridge * ||beta||^2.
        let ridge_penalty = 0.5 * ridge_used * beta.dot(beta);

        let deviance = 2.0 * nll;

        Ok(WorkingState {
            eta: LinearPredictor::new(eta_exit),
            gradient: totalgrad,
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(h),
            deviance,
            penalty_term: penalty_dev + ridge_penalty,
            firth: crate::pirls::FirthDiagnostics::Inactive,
            ridge_used,
        })
    }

    fn lamlobjective_from_state(
        &self,
        beta: &Array1<f64>,
        state: &WorkingState,
    ) -> Result<f64, EstimationError> {
        let p = beta.len();
        if state.hessian.nrows() != p || state.hessian.ncols() != p {
            return Err(EstimationError::LayoutError(
                "survival laml objective: Hessian/beta dimension mismatch".to_string(),
            ));
        }

        let h_dense = state.hessian.to_dense();
        let logdet_h = Self::spd_logdet(&h_dense)?;

        if self.penalties.blocks.is_empty() {
            // With no penalty blocks, S=0 and log|S|_+ = 0 by convention.
            return Ok(0.5 * state.deviance + state.penalty_term + 0.5 * logdet_h);
        }

        // Build S(rho) and compute pseudo-logdet over positive eigenspace.
        let mut s_total = Array2::<f64>::zeros((p, p));
        for block in &self.penalties.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let start = block.range.start;
            let end = block.range.end;
            s_total
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(block.lambda, &block.matrix);
        }
        let (s_eval, _) = s_total
            .eigh(Side::Lower)
            .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
        let max_s_eval = s_eval.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let s_tol = (max_s_eval * 1e-12).max(1e-14);
        let logdet_s = s_eval
            .iter()
            .filter(|&&ev| ev > s_tol)
            .map(|&ev| ev.ln())
            .sum::<f64>();

        Ok(0.5 * state.deviance + state.penalty_term + 0.5 * logdet_h - 0.5 * logdet_s)
    }

    /// Evaluate the survival outer objective and exact `rho`-gradient at a
    /// converged inner state.
    ///
    /// Objective and notation in this implementation:
    /// - `rho_k = log(lambda_k)`, `lambda_k = exp(rho_k)`.
    /// - `S(rho) = sum_k A_k`, where `A_k = dS/drho_k = lambda_k * S_k`.
    /// - `H = X^T W X + S(rho)` from the inner model state.
    /// - `V(rho) = 0.5*deviance(beta_hat) + penalty_term(beta_hat)
    ///           + 0.5*log|H| - 0.5*log|S|_+`.
    ///
    /// Exact gradient used here (no finite differences):
    ///   dV/drho_k
    ///     = 0.5 * beta_hat^T A_k beta_hat
    ///     + 0.5 * tr(H^{-1} dH/drho_k)
    ///     - 0.5 * tr(S^+ A_k).
    ///
    /// For this Royston-Parmar survival likelihood, the hard part is the
    /// `tr(H^{-1} dH/drho_k)` contraction. We evaluate it exactly as:
    ///   tr(H^{-1} dH/drho_k)
    ///     = tr(H^{-1} A_k) + tr(H^{-1} (dH_nll/d beta [u_k])),
    /// where:
    ///   u_k = d beta_hat / d rho_k = -H^{-1} A_k beta_hat
    /// and `dH_nll/d beta [u_k]` is the exact directional derivative of the
    /// (unpenalized) Hessian with respect to `beta` along `u_k`.
    ///
    /// This is exact for the constrained coefficient parameterization used by
    /// the survival engine.
    ///
    /// Exact trace-contraction implementation details:
    /// - factorize `H` once,
    /// - compute `H^{-1}` actions by solves (`solvevec` / `solve_mat`),
    /// - build directional Hessian derivative `dH_nll/d beta [u_k]` analytically,
    /// - evaluate trace as `tr(H^{-1} B)` using `solve_mat(B)`.
    ///
    /// Complexity per outer evaluation (dense):
    /// - shared precompute: `O(np^2 + p^3)`,
    /// - per penalty block `k`: `O(np + p^2 + nnz(S_k))`.
    ///
    /// Ridge note:
    /// - `state.penalty_term` includes the tiny stabilization ridge as
    ///   `0.5 * ridge * ||beta||^2`, so the objective value is consistent with
    ///   the solved mode and with the reported gradient/Hessian.
    /// - this ridge is constant in `rho`, so it does not enter `A_k` nor
    ///   `tr(S^+ A_k)`.
    pub fn lamlobjective_and_rhogradient(
        &self,
        beta: &Array1<f64>,
        state: &WorkingState,
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        let p = beta.len();
        let k_count = self.penalties.blocks.len();
        let objective = self.lamlobjective_from_state(beta, state)?;

        if k_count == 0 {
            return Ok((objective, Array1::zeros(0)));
        }

        // Reuse one symmetric factorization for all H^{-1} applications.
        // This is the core speed-up: exact contractions via solves, no dense
        // inverse assembly.
        let factor = state
            .hessian
            .factorize()
            .map_err(EstimationError::InvalidInput)?;

        let solve_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            factor
                .solvemulti(rhs)
                .expect("survival Hessian solve should succeed")
        };
        let solvevec = |rhs: &Array1<f64>| -> Array1<f64> {
            factor
                .solve(rhs)
                .expect("survival Hessian solve should succeed")
        };
        // Keep outer gradient contractions consistent with the fitted inner state:
        // the working predictor includes target baseline offsets plus learned deviation.
        let eta_entry = self.entry_dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.exit_dot(beta) + &self.offset_eta_exit;
        let deriv_raw = self.derivative_dot(beta) + &self.offset_derivative_exit;
        let exp_entry = eta_entry.mapv(f64::exp);
        let exp_exit = eta_exit.mapv(f64::exp);
        let guard = self.derivative_guard();
        let guard_numerical = self.derivative_guard_numerical();
        let n = self.nrows();
        let mut row_exit = vec![0.0_f64; p];
        let mut row_entry = vec![0.0_f64; p];
        let mut row_derivative = vec![0.0_f64; p];
        let mut b_dir = Array2::<f64>::zeros((p, p));
        let mut ge = vec![0.0_f64; p];
        let mut gs = vec![0.0_f64; p];
        let mut gsd = vec![0.0_f64; p];
        let mut he = vec![0.0_f64; p];
        let mut hs = vec![0.0_f64; p];
        let mut hsd = vec![0.0_f64; p];
        let mut te = vec![0.0_f64; p];
        let mut ts = vec![0.0_f64; p];
        let mut tsd = vec![0.0_f64; p];
        let mut rhs_block = Array2::<f64>::zeros((p, 1));

        let jac = Array1::<f64>::ones(p);
        let curvature = Array1::<f64>::zeros(p);
        let third = Array1::<f64>::zeros(p);

        // Assemble S(rho) in the same scaling used by update_state.
        let mut s_total = Array2::<f64>::zeros((p, p));
        for block in &self.penalties.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let start = block.range.start;
            let end = block.range.end;
            s_total
                .slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(block.lambda, &block.matrix);
        }

        // Pseudo-logdet and pseudoinverse for S:
        // d/drho_k [-0.5 log|S|_+] = -0.5 tr(S^+ A_k).
        let (s_eval, s_evec) = s_total
            .eigh(Side::Lower)
            .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
        let max_s_eval = s_eval.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let s_tol = (max_s_eval * 1e-12).max(1e-14);
        // Build S^+ = U_+ diag(1/ev_j) U_+^T via scaled columns of U.
        let mut u_scaled = s_evec.clone();
        for j in 0..p {
            let ev = s_eval[j];
            if ev > s_tol {
                let inv_ev_sqrt = (1.0 / ev).sqrt();
                u_scaled.column_mut(j).mapv_inplace(|v| v * inv_ev_sqrt);
            } else {
                u_scaled.column_mut(j).fill(0.0);
            }
        }
        let s_pinv = u_scaled.dot(&u_scaled.t());

        let mut grad = Array1::<f64>::zeros(k_count);
        for (k, block) in self.penalties.blocks.iter().enumerate() {
            let lambda = block.lambda;
            let start = block.range.start;
            let end = block.range.end;

            let b_block = beta.slice(ndarray::s![start..end]).to_owned();
            let a_k_beta_block = block.matrix.dot(&b_block).mapv(|v| lambda * v);
            let mut a_k_beta = Array1::<f64>::zeros(p);
            a_k_beta
                .slice_mut(ndarray::s![start..end])
                .assign(&a_k_beta_block);

            // Implicit inner derivative:
            // d beta_hat / d rho_k = -H^{-1} A_k beta_hat.
            let u_k = -solvevec(&a_k_beta);

            // Exact trace(H^{-1} A_k) without assembling the full dense inverse.
            // Since A_k is block-supported, solve only those block columns.
            let block_width = end - start;
            if rhs_block.ncols() != block_width {
                rhs_block = Array2::<f64>::zeros((p, block_width.max(1)));
            } else {
                rhs_block.fill(0.0);
            }
            rhs_block
                .slice_mut(ndarray::s![start..end, ..block_width])
                .assign(&block.matrix.mapv(|v| lambda * v));
            let solved_block = solve_mat(&rhs_block);
            let mut trace_hinv_ak = 0.0;
            for j in 0..block_width {
                trace_hinv_ak += solved_block[[start + j, j]];
            }

            // Exact directional derivative B = dH_nll/d beta [u_k].
            // dH/drho_k = A_k + B, so the non-penalty trace piece is tr(H^{-1} B).
            b_dir.fill(0.0);
            for i in 0..n {
                let w_i = self.sampleweight[i];
                if w_i <= 0.0 {
                    continue;
                }
                let has_entry = !self.entry_at_origin[i];
                let mut deta_e = 0.0_f64;
                let mut deta_s = 0.0_f64;
                let mut ds = 0.0_f64;
                self.fill_exit_row(i, &mut row_exit);
                self.fill_entry_row(i, &mut row_entry);
                self.fill_derivative_row(i, &mut row_derivative);
                for j in 0..p {
                    ge[j] = row_exit[j] * jac[j];
                    gs[j] = row_entry[j] * jac[j];
                    gsd[j] = row_derivative[j] * jac[j];
                    he[j] = row_exit[j] * curvature[j];
                    hs[j] = row_entry[j] * curvature[j];
                    hsd[j] = row_derivative[j] * curvature[j];
                    te[j] = row_exit[j] * third[j];
                    ts[j] = row_entry[j] * third[j];
                    tsd[j] = row_derivative[j] * third[j];
                    deta_e += ge[j] * u_k[j];
                    if has_entry {
                        deta_s += gs[j] * u_k[j];
                    }
                    ds += gsd[j] * u_k[j];
                }

                // Interval part:
                // d/dβ [ exp(eta) * (g g^T + diag(h)) ][u]
                for r in 0..p {
                    let dge_r = he[r] * u_k[r];
                    let dgs_r = hs[r] * u_k[r];
                    let dhe_r = te[r] * u_k[r];
                    let dhs_r = ts[r] * u_k[r];
                    for c in 0..p {
                        let dge_c = he[c] * u_k[c];
                        let dgs_c = hs[c] * u_k[c];
                        let mut d_h_rc =
                            exp_exit[i] * (deta_e * ge[r] * ge[c] + dge_r * ge[c] + ge[r] * dge_c);
                        if r == c {
                            d_h_rc += exp_exit[i] * (deta_e * he[r] + dhe_r);
                        }
                        if has_entry {
                            d_h_rc -= exp_entry[i]
                                * (deta_s * gs[r] * gs[c] + dgs_r * gs[c] + gs[r] * dgs_c);
                            if r == c {
                                d_h_rc -= exp_entry[i] * (deta_s * hs[r] + dhs_r);
                            }
                        }
                        b_dir[[r, c]] += w_i * d_h_rc;
                    }
                }

                // Event part:
                // d/dβ [ gsd gsd^T / s^2 - diag(he) - diag(hsd / s) ][u]
                let s_i = self
                    .stabilized_structural_derivative(deriv_raw[i])
                    .unwrap_or(deriv_raw[i]);
                if !s_i.is_finite() {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "survival monotonicity violated in LAML trace contraction at row {}: d_eta/dt={:.3e} <= tolerance={:.3e}",
                        i, s_i, guard
                    )));
                }
                if self.event_target[i] > 0 {
                    if s_i < guard_numerical {
                        return Err(EstimationError::ParameterConstraintViolation(format!(
                            "survival monotonicity violated in LAML trace contraction at row {}: d_eta/dt={:.3e} <= tolerance={:.3e}",
                            i, s_i, guard
                        )));
                    }
                    let inv_s = 1.0 / s_i;
                    let inv_s2 = inv_s * inv_s;
                    let inv_s3 = inv_s2 * inv_s;
                    for r in 0..p {
                        let dgd_r = hsd[r] * u_k[r];
                        let dte_r = te[r] * u_k[r];
                        let dtsd_r = tsd[r] * u_k[r];
                        for c in 0..p {
                            let dgd_c = hsd[c] * u_k[c];
                            let mut d_h_rc = (dgd_r * gsd[c] + gsd[r] * dgd_c) * inv_s2
                                - 2.0 * gsd[r] * gsd[c] * ds * inv_s3;
                            if r == c {
                                d_h_rc += -dte_r;
                                d_h_rc += -(dtsd_r * inv_s - hsd[r] * ds * inv_s2);
                            }
                            b_dir[[r, c]] += w_i * d_h_rc;
                        }
                    }
                }
            }

            let mut tracethird = 0.0;
            let trace_chunk_cols = (SURVIVAL_TRACE_SOLVE_TARGET_BYTES
                / (p.max(1) * std::mem::size_of::<f64>()))
            .max(1)
            .min(p);
            let mut rhs_trace = Array2::<f64>::zeros((p, trace_chunk_cols));
            for chunk_start in (0..p).step_by(trace_chunk_cols) {
                let width = (p - chunk_start).min(trace_chunk_cols);
                rhs_trace.fill(0.0);
                rhs_trace
                    .slice_mut(ndarray::s![.., ..width])
                    .assign(&b_dir.slice(ndarray::s![.., chunk_start..chunk_start + width]));
                let solved_chunk = solve_mat(&rhs_trace);
                for j in 0..width {
                    tracethird += solved_chunk[[chunk_start + j, j]];
                }
            }
            let t_k = trace_hinv_ak + tracethird;

            // p_k = tr(S^+ A_k) restricted to this block.
            let s_pinv_block = s_pinv.slice(ndarray::s![start..end, start..end]);
            let p_k = lambda * (&s_pinv_block * &block.matrix.t()).sum();

            // Final exact component:
            // 0.5 * beta^T A_k beta + 0.5 * tr(H^{-1} dH/drho_k) - 0.5 * tr(S^+ A_k),
            // with tr(H^{-1} dH/drho_k) = trace_hinv_ak + tracethird.
            grad[k] = 0.5 * beta.dot(&a_k_beta) + 0.5 * t_k - 0.5 * p_k;
        }

        Ok((objective, grad))
    }
}

#[derive(Debug, Clone)]
pub struct CrudeRiskResult {
    pub risk: f64,
    pub diseasegradient: Array1<f64>,
    pub mortalitygradient: Array1<f64>,
}

fn compute_gauss_legendre_nodes(n: usize) -> Vec<(f64, f64)> {
    let mut nodesweights = Vec::with_capacity(n);
    let m = n.div_ceil(2);

    for i in 0..m {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;

        for _ in 0..100 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z_prev = z;
            z = z_prev - p1 / pp;
            if (z - z_prev).abs() < 1e-14 {
                break;
            }
        }

        let x = z;
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        if !n.is_multiple_of(2) && i == m - 1 {
            nodesweights.push((0.0, w));
        } else {
            nodesweights.push((-x, w));
            nodesweights.push((x, w));
        }
    }

    nodesweights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    nodesweights
}

fn gauss_legendre_quadrature() -> &'static [(f64, f64)] {
    static CACHE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    CACHE.get_or_init(|| compute_gauss_legendre_nodes(40))
}

/// Engine-level crude risk quadrature with exact delta-method gradients.
///
/// This routine owns the numerical integration and gradient accumulation math:
/// - It integrates `h_d(u) * S_total(u | t0)` over `[t0, t1]` by high-order
///   Gauss-Legendre quadrature.
/// - It computes gradients w.r.t. disease and mortality coefficients:
///   d Risk / d beta_d and d Risk / d beta_m.
///
/// The adapter provides the domain-specific point evaluator callback `eval_at`,
/// which fills design rows and returns:
/// - instantaneous disease hazard at age `u`,
/// - cumulative disease hazard `H_d(u)`,
/// - cumulative mortality hazard `H_m(u)`.
///
/// This keeps biology/data wiring out of `gam` while centralizing the
/// integration engine in one place.
#[allow(clippy::too_many_arguments)]
pub fn calculate_crude_risk_quadrature<F>(
    t0: f64,
    t1: f64,
    breakpoints: &[f64],
    h_dis_t0: f64,
    h_mor_t0: f64,
    design_d_t0: ArrayView1<'_, f64>,
    design_m_t0: ArrayView1<'_, f64>,
    mut eval_at: F,
) -> Result<CrudeRiskResult, SurvivalError>
where
    F: FnMut(
        f64,
        &mut Array1<f64>,
        &mut Array1<f64>,
        &mut Array1<f64>,
    ) -> Result<(f64, f64, f64), SurvivalError>,
{
    let coeff_len_d = design_d_t0.len();
    let coeff_len_m = design_m_t0.len();
    if coeff_len_d == 0 || coeff_len_m == 0 {
        return Err(SurvivalError::InvalidIntegrationSetup);
    }
    if !t0.is_finite()
        || !t1.is_finite()
        || !h_dis_t0.is_finite()
        || !h_mor_t0.is_finite()
        || design_d_t0.iter().any(|v| !v.is_finite())
        || design_m_t0.iter().any(|v| !v.is_finite())
    {
        return Err(SurvivalError::NonFiniteInput);
    }
    if t1 <= t0 {
        return Ok(CrudeRiskResult {
            risk: 0.0,
            diseasegradient: Array1::zeros(coeff_len_d),
            mortalitygradient: Array1::zeros(coeff_len_m),
        });
    }

    let mut sorted_breaks: Vec<f64> = breakpoints
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x >= t0 && *x <= t1)
        .collect();
    sorted_breaks.push(t0);
    sorted_breaks.push(t1);
    sorted_breaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if sorted_breaks.len() < 2 {
        return Err(SurvivalError::InvalidIntegrationSetup);
    }

    let mut total_risk = 0.0;
    let mut diseasegradient = Array1::zeros(coeff_len_d);
    let mut mortalitygradient = Array1::zeros(coeff_len_m);
    let nodesweights = gauss_legendre_quadrature();

    let mut design_d = Array1::<f64>::zeros(coeff_len_d);
    let mut deriv_d = Array1::<f64>::zeros(coeff_len_d);
    let mut design_m = Array1::<f64>::zeros(coeff_len_m);

    for segment in sorted_breaks.windows(2) {
        let a = segment[0];
        let b = segment[1];
        let center = 0.5 * (b + a);
        let halfwidth = 0.5 * (b - a);
        if halfwidth <= 0.0 {
            continue;
        }

        for &(x, w) in nodesweights {
            let u = center + halfwidth * x;
            let (inst_hazard_d, hazard_d, hazard_m) =
                eval_at(u, &mut design_d, &mut deriv_d, &mut design_m)?;
            if !inst_hazard_d.is_finite() || !hazard_d.is_finite() || !hazard_m.is_finite() {
                return Err(SurvivalError::NonFiniteInput);
            }
            if inst_hazard_d <= 0.0 {
                return Err(SurvivalError::NonPositiveHazard);
            }

            if hazard_d < h_dis_t0 || hazard_m < h_mor_t0 {
                return Err(SurvivalError::NonMonotoneCumulativeHazard);
            }

            let h_dis_cond = hazard_d - h_dis_t0;
            let h_mor_cond = hazard_m - h_mor_t0;
            let s_total = (-(h_dis_cond + h_mor_cond)).exp();

            total_risk += w * inst_hazard_d * s_total * halfwidth;

            // d Risk / d beta_d:
            //   integral [ d h_d * S_total - h_d * S_total * d H_d ] du
            let weight = w * s_total * halfwidth;
            for j in 0..coeff_len_d {
                let d_inst_hazard =
                    inst_hazard_d * (1.0 - hazard_d) * design_d[j] + hazard_d * deriv_d[j];
                let d_hazard_cond = hazard_d * design_d[j] - h_dis_t0 * design_d_t0[j];
                let g = d_inst_hazard - inst_hazard_d * d_hazard_cond;
                diseasegradient[j] += weight * g;
            }

            // d Risk / d beta_m:
            //   -integral h_d * S_total * d H_m(u|t0) du
            let weight = w * inst_hazard_d * s_total * halfwidth;
            for j in 0..coeff_len_m {
                let g = -hazard_m * design_m[j] + h_mor_t0 * design_m_t0[j];
                mortalitygradient[j] += weight * g;
            }
        }
    }

    Ok(CrudeRiskResult {
        risk: total_risk,
        diseasegradient,
        mortalitygradient,
    })
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
    fn penaltyhessian_matchesgradient_jacobian() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];

        let grad = penalties.gradient(&beta);
        let h = penalties.hessian(beta.len());
        let b_block = beta.slice(s![1..3]).to_owned();
        let expected = 1.7 * array![[2.0, 0.5], [0.5, 3.0]].dot(&b_block);

        assert!((grad[1] - expected[0]).abs() < 1e-12);
        assert!((grad[2] - expected[1]).abs() < 1e-12);
        assert!((h[[1, 1]] - 1.7 * 2.0).abs() < 1e-12);
        assert!((h[[1, 2]] - 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 1]] - 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 2]] - 1.7 * 3.0).abs() < 1e-12);
    }

    #[test]
    fn penaltygradient_matches_deviance_finite_difference() {
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
            assert_eq!(
                grad[idx].signum(),
                fd.signum(),
                "gradient/deviance sign mismatch at idx={idx}: grad={} fd={fd}",
                grad[idx]
            );
            assert!(
                (grad[idx] - fd).abs() < 1e-6,
                "gradient/deviance mismatch at idx={idx}: grad={} fd={fd}",
                grad[idx]
            );
        }
    }

    #[test]
    fn zero_offsets_match_default_survival_state() {
        let age_entry = array![1.0_f64, 2.0_f64];
        let age_exit = array![2.0_f64, 3.5_f64];
        let event_target = array![1u8, 0u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[1.0, age_entry[0].ln()], [1.0, age_entry[1].ln()]];
        let x_exit = array![[1.0, age_exit[0].ln()], [1.0, age_exit[1].ln()]];
        let x_derivative = array![[0.0, 1.0 / age_exit[0]], [0.0, 1.0 / age_exit[1]]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let beta = array![-1.0, 0.8];

        let base = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties.clone(),
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct base survival model");

        let zero_offsets = WorkingModelSurvival::from_engine_inputswith_offsets(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            Some(SurvivalBaselineOffsets {
                eta_entry: array![0.0, 0.0].view(),
                eta_exit: array![0.0, 0.0].view(),
                derivative_exit: array![0.0, 0.0].view(),
            }),
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct offset survival model");

        let state_base = base.update_state(&beta).expect("base state");
        let statezero = zero_offsets.update_state(&beta).expect("zero-offset state");
        assert!((state_base.deviance - statezero.deviance).abs() < 1e-12);
        assert!(
            state_base
                .gradient
                .iter()
                .zip(statezero.gradient.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    #[test]
    fn crudespec_is_rejected_by_one_hazard_engine() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![0u8];
        let event_competing = array![1u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.1]];
        let x_exit = array![[0.4]];
        let x_derivative = array![[1.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };

        let err = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Crude,
        )
        .expect_err("crude fitting should be rejected by the one-hazard engine");
        assert!(matches!(err, SurvivalError::UnsupportedSpec("crude")));
    }

    #[test]
    fn monotonicity_constraints_cover_allweightedrows() {
        let age_entry = array![1.0_f64, 1.5_f64];
        let age_exit = array![2.0_f64, 2.5_f64];
        let event_target = array![0u8, 0u8];
        let event_competing = array![0u8, 1u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[0.2], [0.1]];
        let x_exit = array![[0.3], [0.2]];
        let x_derivative = array![[1.0], [1.0]];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct censored survival model");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("all weighted rows should contribute monotonicity constraints");
        assert_eq!(constraints.a.nrows(), 1);
        assert!(constraints.b[0].abs() < 1e-18);
        for i in 0..x_derivative.nrows() {
            let row = x_derivative.row(i);
            let original_rhs = model.derivative_guard();
            let scale = row.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            let normalized_rhs = original_rhs / scale;
            let lhs = constraints.a.row(0).dot(&array![1.0]);
            assert!(lhs >= normalized_rhs - 1e-18);
        }
    }

    #[test]
    fn decreasing_interval_is_rejectedwithout_target_events() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![0u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.5]];
        let x_exit = array![[0.0]];
        let x_derivative = array![[1.0]];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct censored survival model");

        let err = model
            .update_state(&array![1.0])
            .expect_err("decreasing cumulative hazard increment should be rejected");
        assert!(
            err.to_string().contains("cumulative hazard decreased"),
            "unexpected error: {err}"
        );
    }

    fn smooth_crude_risk(beta_d: f64, beta_m: f64) -> CrudeRiskResult {
        calculate_crude_risk_quadrature(
            0.0,
            1.0,
            &[0.0, 1.0],
            beta_d.exp(),
            beta_m.exp(),
            array![1.0].view(),
            array![1.0].view(),
            |u, design_d, deriv_d, design_m| {
                let cumulative_d = beta_d.exp() * (1.0 + 0.2 * u);
                let cumulative_m = beta_m.exp() * (1.0 + 0.1 * u);
                let inst_hazard_d = 0.2 * beta_d.exp();
                design_d[0] = 1.0;
                deriv_d[0] = inst_hazard_d;
                design_m[0] = 1.0;
                Ok((inst_hazard_d, cumulative_d, cumulative_m))
            },
        )
        .expect("smooth crude-risk quadrature should succeed")
    }

    #[test]
    fn crude_riskgradient_matches_monotoneobjective() {
        let beta_d = -0.2_f64;
        let beta_m = -0.5_f64;
        let result = smooth_crude_risk(beta_d, beta_m);
        let eps = 1e-6;

        let fd_d = (smooth_crude_risk(beta_d + eps, beta_m).risk
            - smooth_crude_risk(beta_d - eps, beta_m).risk)
            / (2.0 * eps);
        let fd_m = (smooth_crude_risk(beta_d, beta_m + eps).risk
            - smooth_crude_risk(beta_d, beta_m - eps).risk)
            / (2.0 * eps);

        assert!(
            (result.diseasegradient[0] - fd_d).abs() < 1e-5,
            "disease gradient mismatch for monotone crude risk: analytic={} fd={fd_d}",
            result.diseasegradient[0]
        );
        assert!(
            (result.mortalitygradient[0] - fd_m).abs() < 1e-5,
            "mortality gradient mismatch for monotone crude risk: analytic={} fd={fd_m}",
            result.mortalitygradient[0]
        );
    }

    #[test]
    fn survivalridge_penalty_scalar_matchesgradienthessian_scaling() {
        let age_entry = array![1.0_f64, 2.0_f64];
        let age_exit = array![2.0_f64, 3.5_f64];
        let event_target = array![1u8, 0u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[1.0, age_entry[0].ln()], [1.0, age_entry[1].ln()]];
        let x_exit = array![[1.0, age_exit[0].ln()], [1.0, age_exit[1].ln()]];
        let x_derivative = array![[0.0, 1.0 / age_exit[0]], [0.0, 1.0 / age_exit[1]]];
        let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: array![[2.0]],
            lambda: 1.7,
            range: 1..2,
        }]);
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let beta = array![-1.2, 0.4];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties.clone(),
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct survival model");

        let state = model.update_state(&beta).expect("survival state");
        let expected_penalty = penalties.deviance(&beta) + 0.5 * state.ridge_used * beta.dot(&beta);
        assert!(
            (state.penalty_term - expected_penalty).abs() < 1e-12,
            "penalty_term mismatch: state={} expected={}",
            state.penalty_term,
            expected_penalty
        );
    }

    #[test]
    fn negative_penalty_lambda_is_rejected() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.5]];
        let x_derivative = array![[0.0, 1.0]];
        let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: array![[1.0]],
            lambda: -0.1,
            range: 1..2,
        }]);

        let err = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect_err("negative lambda must be rejected");

        assert!(matches!(err, SurvivalError::NonFiniteInput));
    }

    #[test]
    fn penalty_block_range_and_shapemust_match_coefficients() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.5]];
        let x_derivative = array![[0.0, 1.0]];
        let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: array![[1.0]],
            lambda: 0.5,
            range: 0..2,
        }]);

        let err = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            MonotonicityPenalty { tolerance: 1e-8 },
            SurvivalSpec::Net,
        )
        .expect_err("penalty block geometry must match coefficient support");

        assert!(matches!(err, SurvivalError::DimensionMismatch));
    }

    #[test]
    fn survivalgradient_matchesobjectivefdwithridge_scaling() {
        let age_entry = array![1.0_f64, 2.0_f64, 3.0_f64];
        let age_exit = array![2.0_f64, 3.5_f64, 4.0_f64];
        let event_target = array![1u8, 0u8, 1u8];
        let event_competing = array![0u8, 0u8, 0u8];
        let sampleweight = array![1.0, 1.0, 1.0];
        let x_entry = array![
            [1.0, age_entry[0].ln()],
            [1.0, age_entry[1].ln()],
            [1.0, age_entry[2].ln()]
        ];
        let x_exit = array![
            [1.0, age_exit[0].ln()],
            [1.0, age_exit[1].ln()],
            [1.0, age_exit[2].ln()]
        ];
        let x_derivative = array![
            [0.0, 1.0 / age_exit[0]],
            [0.0, 1.0 / age_exit[1]],
            [0.0, 1.0 / age_exit[2]]
        ];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let beta = array![-1.0, 3.0];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct survival model");

        let state = model.update_state(&beta).expect("state at beta");
        let eps = 1e-7;
        for j in 0..beta.len() {
            let mut plus = beta.clone();
            let mut minus = beta.clone();
            plus[j] += eps;
            minus[j] -= eps;
            let state_plus = model.update_state(&plus).expect("state at beta + eps");
            let state_minus = model.update_state(&minus).expect("state at beta - eps");
            let obj_plus = 0.5 * state_plus.deviance + state_plus.penalty_term;
            let obj_minus = 0.5 * state_minus.deviance + state_minus.penalty_term;
            let fd = (obj_plus - obj_minus) / (2.0 * eps);
            assert_eq!(
                state.gradient[j].signum(),
                fd.signum(),
                "objective/gradient sign mismatch at j={j}: grad={} fd={fd}",
                state.gradient[j]
            );
            assert!(
                (state.gradient[j] - fd).abs() < 1e-5,
                "objective/gradient mismatch at j={j}: grad={} fd={fd}",
                state.gradient[j]
            );
        }
    }

    #[test]
    fn structural_monotonicgradient_matchesobjectivefd() {
        let age_entry = array![1.0_f64, 1.3_f64, 1.8_f64];
        let age_exit = array![1.6_f64, 2.1_f64, 2.7_f64];
        let event_target = array![1u8, 0u8, 1u8];
        let event_competing = array![0u8, 0u8, 0u8];
        let sampleweight = array![1.0, 1.0, 1.0];

        // Time block has 3 structural-monotone columns.
        // Final column is a covariate, left unconstrained.
        let x_entry = array![
            [1.0, 0.2, 0.05, -0.7],
            [1.0, 0.5, 0.20, 0.1],
            [1.0, 0.9, 0.60, 1.2]
        ];
        let x_exit = array![
            [1.0, 0.4, 0.16, -0.7],
            [1.0, 0.8, 0.64, 0.1],
            [1.0, 1.1, 1.21, 1.2]
        ];
        let x_derivative = array![
            [0.0, 0.8, 0.64, 0.0],
            [0.0, 0.7, 1.12, 0.0],
            [0.0, 0.6, 1.32, 0.0]
        ];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");
        model
            .set_structural_monotonicity(true, 3)
            .expect("enable structural monotonicity");
        let constraints = model
            .monotonicity_linear_constraints()
            .expect("structural derivative constraints");
        assert!(
            constraints
                .a
                .slice(ndarray::s![.., 0..3])
                .iter()
                .all(|v| v.is_finite()),
            "structural mode should keep explicit derivative-row constraints"
        );
        assert_eq!(constraints.a.ncols(), 4);

        let beta = array![0.2, 0.2, 0.1, 0.2];
        let state = model.update_state(&beta).expect("state at structural beta");
        let eps = 1e-7;
        for j in 0..beta.len() {
            let mut plus = beta.clone();
            let mut minus = beta.clone();
            plus[j] += eps;
            minus[j] -= eps;
            let state_plus = model.update_state(&plus).expect("state at beta + eps");
            let state_minus = model.update_state(&minus).expect("state at beta - eps");
            let obj_plus = 0.5 * state_plus.deviance + state_plus.penalty_term;
            let obj_minus = 0.5 * state_minus.deviance + state_minus.penalty_term;
            let fd = (obj_plus - obj_minus) / (2.0 * eps);
            assert_eq!(
                state.gradient[j].signum(),
                fd.signum(),
                "structural objective/gradient sign mismatch at j={j}: grad={} fd={fd}",
                state.gradient[j]
            );
            assert!(
                (state.gradient[j] - fd).abs() < 2e-5,
                "structural objective/gradient mismatch at j={j}: grad={} fd={fd}",
                state.gradient[j]
            );
        }
    }

    #[test]
    fn structural_monotonic_lamlgradient_returns_finitevalues() {
        let age_entry = array![1.0_f64, 1.2_f64];
        let age_exit = array![1.5_f64, 2.0_f64];
        let event_target = array![1u8, 0u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[1.0, 0.2, -0.5], [1.0, 0.4, 0.2]];
        let x_exit = array![[1.0, 0.5, -0.5], [1.0, 0.8, 0.2]];
        let x_derivative = array![[0.0, 0.9, 0.0], [0.0, 0.7, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");
        model
            .set_structural_monotonicity(true, 2)
            .expect("enable structural monotonicity");
        // One simple penalty block to exercise rho-gradient path.
        model.penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: array![[1.0]],
            lambda: 0.7,
            range: 1..2,
        }]);
        let beta = array![0.2, 0.2, 0.1];
        let state = model.update_state(&beta).expect("state at structural beta");
        let (obj, grad) = model
            .lamlobjective_and_rhogradient(&beta, &state)
            .expect("laml gradient should work in structural mode");
        assert!(obj.is_finite());
        assert_eq!(grad.len(), 1);
        assert!(grad[0].is_finite());
    }

    #[test]
    fn structural_monotonicity_switches_to_tiny_derivative_guard_constraints() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.0]];
        let x_exit = array![[0.2]];
        let x_derivative = array![[1.0]];

        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");

        let beta = array![-3.0];
        assert!(
            model.update_state(&beta).is_err(),
            "negative derivative coefficient should violate derivative guard"
        );

        model
            .set_structural_monotonicity(true, 1)
            .expect("enable structural monotonicity");
        let constraints = model
            .monotonicity_linear_constraints()
            .expect("structural derivative constraints");
        assert_eq!(constraints.a.nrows(), 1);
        assert_eq!(constraints.a.ncols(), 1);
        assert!((constraints.a[[0, 0]] - 1.0).abs() <= 1e-12);
        // Structural monotonicity uses derivative_guard() == 0.0
        assert!(constraints.b[0].abs() <= 1e-12);
        let state = model
            .update_state(&array![1e-6])
            .expect("small positive derivative coefficient should remain feasible");
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn derivative_offset_must_clear_nonstructural_monotonicity_threshold() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.0]];
        let x_derivative = array![[0.0, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = MonotonicityPenalty { tolerance: 3.0 };
        let eta_entry_offset = array![0.0];
        let eta_exit_offset = array![0.0];
        let derivative_offset_below_guard = array![2.0];
        let derivative_offset_above_guard = array![3.1];
        let offsets_below_guard = SurvivalBaselineOffsets {
            eta_entry: eta_entry_offset.view(),
            eta_exit: eta_exit_offset.view(),
            derivative_exit: derivative_offset_below_guard.view(),
        };
        let offsets_above_guard = SurvivalBaselineOffsets {
            eta_entry: eta_entry_offset.view(),
            eta_exit: eta_exit_offset.view(),
            derivative_exit: derivative_offset_above_guard.view(),
        };

        let model_below_guard = WorkingModelSurvival::from_engine_inputswith_offsets(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            Some(offsets_below_guard),
            penalties.clone(),
            monotonicity,
            SurvivalSpec::Net,
        )
        .expect("construct model with derivative offset below guard");
        let err = model_below_guard
            .update_state(&array![0.0, 0.0])
            .expect_err("derivative offset below guard should be rejected");
        let err_text = err.to_string();
        assert!(
            err_text.contains("d_eta/dt=2.000e0") && err_text.contains("tolerance=3.000e0"),
            "expected derivative guard rejection to report the offset-driven derivative: {err_text}"
        );

        let model_above_guard = WorkingModelSurvival::from_engine_inputswith_offsets(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            Some(offsets_above_guard),
            penalties,
            MonotonicityPenalty { tolerance: 3.0 },
            SurvivalSpec::Net,
        )
        .expect("construct model with derivative offset above guard");
        let state = model_above_guard
            .update_state(&array![0.0, 0.0])
            .expect("derivative offset above guard should remain feasible");
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn structural_monotonicity_emits_rowwise_constraints() {
        let age_entry = array![1.0_f64, 1.5_f64];
        let age_exit = array![2.0_f64, 3.0_f64];
        let event_target = array![1u8, 0u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        let x_exit = array![[0.2, 0.4, 1.0], [0.3, 0.5, 1.0]];
        let x_derivative = array![[0.3, 0.2, 0.0], [0.4, 0.1, 0.0]];

        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");
        model
            .set_structural_monotonicity(true, 2)
            .expect("enable structural monotonicity");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("structural derivative constraints");

        assert_eq!(constraints.a.nrows(), 2);
        assert_eq!(constraints.a.ncols(), 3);
        assert_eq!(constraints.a.row(0).to_vec(), vec![0.3, 0.2, 0.0]);
        assert_eq!(constraints.a.row(1).to_vec(), vec![0.4, 0.1, 0.0]);
        // Structural monotonicity uses derivative_guard() == 0.0
        assert!(constraints.b.iter().all(|&v| v.abs() <= 1e-12));
    }

    #[test]
    fn structural_monotonicity_preserves_inactive_time_columns_in_constraints() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.2]];
        let x_exit = array![[1.0, 0.6]];
        let x_derivative = array![[0.0, 1.0]];

        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");
        model
            .set_structural_monotonicity(true, 2)
            .expect("enable structural monotonicity");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("structural derivative constraints");

        assert!(
            constraints.a[[0, 0]].abs() <= 1e-12,
            "inactive time column should remain absent from the row-wise constraint"
        );
        assert!(
            (constraints.a[[0, 1]] - 1.0).abs() <= 1e-12,
            "active time column should remain in the row-wise constraint"
        );
    }

    #[test]
    fn structural_monotonicity_preserves_sparse_row_patterns() {
        let age_entry = array![1.0_f64, 1.5_f64];
        let age_exit = array![2.0_f64, 2.5_f64];
        let event_target = array![1u8, 1u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[0.0, 0.0], [0.0, 0.0]];
        let x_exit = array![[0.4, 0.2], [0.6, 0.3]];
        let x_derivative = array![[1.0, 0.0], [1.0, 0.5]];

        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");
        model
            .set_structural_monotonicity(true, 2)
            .expect("enable structural monotonicity");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("structural derivative constraints");

        assert!(
            (constraints.a[[0, 0]] - 1.0).abs() <= 1e-12,
            "first sparse row should constrain only column 0"
        );
        assert!(
            constraints.a[[0, 1]].abs() <= 1e-12,
            "first sparse row should leave column 1 unconstrained"
        );
    }

    fn modelwith_rho(base: &WorkingModelSurvival, rho: &Array1<f64>) -> WorkingModelSurvival {
        let mut model = base.clone();
        assert_eq!(model.penalties.blocks.len(), rho.len());
        for (k, block) in model.penalties.blocks.iter_mut().enumerate() {
            block.lambda = rho[k].exp();
        }
        model
    }

    fn solve_inner_mode(model: &WorkingModelSurvival, beta_init: &Array1<f64>) -> Array1<f64> {
        let mut model_local = model.clone();
        let opts = crate::pirls::WorkingModelPirlsOptions {
            max_iterations: 200,
            convergence_tolerance: 1e-10,
            max_step_halving: 40,
            min_step_size: 1e-12,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: model_local.monotonicity_linear_constraints(),
        };
        let out = crate::pirls::runworking_model_pirls(
            &mut model_local,
            crate::types::Coefficients::new(beta_init.clone()),
            &opts,
            |info| {
                let _ = info;
            },
        )
        .expect("survival constrained PIRLS inner mode");
        out.beta.0
    }

    fn lamlobjective_at_rho(
        base: &WorkingModelSurvival,
        rho: &Array1<f64>,
        beta_init: &Array1<f64>,
    ) -> (f64, Array1<f64>, Array1<f64>) {
        let model = modelwith_rho(base, rho);
        let beta_hat = solve_inner_mode(&model, beta_init);
        let state = model
            .update_state(&beta_hat)
            .expect("state at inner mode for outer objective");
        let (obj, grad) = model
            .lamlobjective_and_rhogradient(&beta_hat, &state)
            .expect("analytic laml objective/gradient");
        (obj, grad, beta_hat)
    }

    #[test]
    fn wip_outer_helper_signatures_stay_linked() {
        let _model_fn =
            modelwith_rho as fn(&WorkingModelSurvival, &Array1<f64>) -> WorkingModelSurvival;
        let _mode_fn = solve_inner_mode as fn(&WorkingModelSurvival, &Array1<f64>) -> Array1<f64>;
        let _laml_fn = lamlobjective_at_rho
            as fn(
                &WorkingModelSurvival,
                &Array1<f64>,
                &Array1<f64>,
            ) -> (f64, Array1<f64>, Array1<f64>);
    }

    #[test]
    fn update_state_rejects_negative_exit_derivative_for_censoredrows() {
        let age_entry = array![1.0_f64];
        let age_exit = array![1.1_f64];
        let event_target = array![0u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.0]];
        let x_exit = array![[0.0]];
        let x_derivative = array![[-1.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct censored survival model");

        let err = model
            .update_state(&array![1.0])
            .expect_err("censored row should still enforce monotonic derivative");
        assert!(
            matches!(err, EstimationError::ParameterConstraintViolation(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn crude_risk_quadrature_rejects_decreasing_cumulative_hazard() {
        let err = calculate_crude_risk_quadrature(
            1.0,
            2.0,
            &[],
            0.4,
            0.2,
            array![1.0].view(),
            array![1.0].view(),
            |u, design_d, deriv_d, design_m| {
                let _ = u;
                design_d[0] = 1.0;
                deriv_d[0] = 0.0;
                design_m[0] = 1.0;
                Ok((0.1, 0.3, 0.25))
            },
        )
        .expect_err("non-monotone cumulative hazards should fail");
        assert!(matches!(err, SurvivalError::NonMonotoneCumulativeHazard));
    }

    #[test]
    fn crude_risk_quadrature_rejects_nonpositive_instantaneous_hazard() {
        let err = calculate_crude_risk_quadrature(
            1.0,
            2.0,
            &[],
            0.4,
            0.2,
            array![1.0].view(),
            array![1.0].view(),
            |u, design_d, deriv_d, design_m| {
                let _ = u;
                design_d[0] = 1.0;
                deriv_d[0] = 0.0;
                design_m[0] = 1.0;
                Ok((0.0, 0.4, 0.25))
            },
        )
        .expect_err("nonpositive hazards should fail");
        assert!(matches!(err, SurvivalError::NonPositiveHazard));
    }

    #[test]
    fn laml_no_penalties_matches_documentedobjective() {
        let age_entry = array![40.0, 45.0, 50.0, 55.0];
        let age_exit = array![44.0, 49.0, 54.0, 59.0];
        let event_target = array![1u8, 0u8, 1u8, 0u8];
        let event_competing = Array1::<u8>::zeros(4);
        let sampleweight = Array1::ones(4);
        let x_entry = array![
            [1.0, -0.2, 0.04],
            [1.0, -0.1, 0.01],
            [1.0, 0.0, 0.0],
            [1.0, 0.1, 0.01]
        ];
        let x_exit = array![
            [1.0, -0.12, 0.0144],
            [1.0, -0.02, 0.0004],
            [1.0, 0.08, 0.0064],
            [1.0, 0.18, 0.0324]
        ];
        let x_derivative = array![
            [0.0, 0.02, 0.001],
            [0.0, 0.02, 0.001],
            [0.0, 0.02, 0.001],
            [0.0, 0.02, 0.001]
        ];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };
        let beta = array![-2.0, 0.7, 0.2];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct survival model");

        let state = model.update_state(&beta).expect("state at beta");
        let (obj, grad) = model
            .lamlobjective_and_rhogradient(&beta, &state)
            .expect("laml objective for no-penalty model");

        let h_dense = state.hessian.to_dense();
        let logdet_h = WorkingModelSurvival::spd_logdet(&h_dense).expect("SPD hessian logdet");
        let expected = 0.5 * state.deviance + state.penalty_term + 0.5 * logdet_h;

        assert_eq!(grad.len(), 0);
        assert!(
            (obj - expected).abs() < 1e-10,
            "no-penalty LAML objective mismatch: obj={} expected={}",
            obj,
            expected
        );
    }

    #[test]
    fn laml_objective_uses_positive_spectrum_logdet_fallback() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.0]];
        let x_exit = array![[0.2]];
        let x_derivative = array![[0.5]];
        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct survival model");
        let beta = array![0.3];
        let mut state = model.update_state(&beta).expect("state at beta");
        state.hessian = crate::linalg::matrix::SymmetricMatrix::Dense(array![[-1.0]]);

        let objective = model
            .lamlobjective_from_state(&beta, &state)
            .expect("indefinite hessian should use positive-spectrum logdet fallback");
        assert!(objective.is_finite());
    }

    #[test]
    fn monotonicity_constraints_collapse_positive_collinearrows() {
        let a = array![[0.0, 0.5, 0.0], [0.0, 0.25, 0.0], [0.0, 0.125, 0.0]];
        let b = array![1e-8, 1e-8, 1e-8];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 1);
        assert_eq!(compressed.a.ncols(), 3);
        assert!(compressed.a[[0, 0]].abs() <= 1e-12);
        assert!((compressed.a[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!(compressed.a[[0, 2]].abs() <= 1e-12);
        assert!((compressed.b[0] - 8e-8).abs() <= 1e-18);
    }

    #[test]
    fn monotonicity_constraints_preserve_distinct_directions() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]];
        let b = array![0.2, 0.3, 0.1];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 2);
        let mut saw_x = false;
        let mut saw_y = false;
        for i in 0..compressed.a.nrows() {
            if (compressed.a[[i, 0]] - 1.0).abs() <= 1e-12 && compressed.a[[i, 1]].abs() <= 1e-12 {
                saw_x = true;
                assert!((compressed.b[i] - 0.2).abs() <= 1e-12);
            }
            if compressed.a[[i, 0]].abs() <= 1e-12 && (compressed.a[[i, 1]] - 1.0).abs() <= 1e-12 {
                saw_y = true;
                assert!((compressed.b[i] - 0.3).abs() <= 1e-12);
            }
        }
        assert!(saw_x);
        assert!(saw_y);
    }

    #[test]
    fn monotonicity_constraints_cluster_near_collinearrows() {
        let a = array![
            [0.0, 0.5, 0.0],
            [0.0, 0.50000000003, 0.0],
            [0.0, 0.49999999997, 0.0]
        ];
        let b = array![1e-8, 1.00000000005e-8, 0.99999999995e-8];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 1);
        assert_eq!(compressed.a.ncols(), 3);
        assert!(compressed.a[[0, 0]].abs() <= 1e-12);
        assert!((compressed.a[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!(compressed.a[[0, 2]].abs() <= 1e-12);
        assert!((compressed.b[0] - 2.0e-8).abs() <= 1e-18);
    }

    #[test]
    fn monotonicity_constraints_cluster_spline_like_near_duplicates() {
        let a = array![
            [0.0, 0.401, 0.302, 0.197],
            [0.0, 0.40100000003, 0.30199999998, 0.19700000001],
            [0.0, 0.40099999997, 0.30200000002, 0.19699999999],
            [0.0, 0.125, 0.500, 0.375]
        ];
        let b = array![2.0e-8, 2.00000000004e-8, 1.99999999996e-8, 3.0e-8];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 2);
        let mut clustered_face = false;
        let mut distinct_face = false;
        for i in 0..compressed.a.nrows() {
            let row = compressed.a.row(i);
            if row[1] > 0.99 && row[2] > 0.7 && row[3] > 0.49 {
                clustered_face = true;
                assert!((compressed.b[i] - (2.0e-8 / 0.401)).abs() <= 1e-12);
            } else {
                distinct_face = true;
                assert!((row[1] - 0.25).abs() <= 1e-12);
                assert!((row[2] - 1.0).abs() <= 1e-12);
                assert!((row[3] - 0.75).abs() <= 1e-12);
                assert!((compressed.b[i] - 6.0e-8).abs() <= 1e-18);
            }
        }
        assert!(clustered_face);
        assert!(distinct_face);
    }

    #[test]
    fn linear_time_monotonicity_constraints_reduce_to_single_halfspace() {
        let age_entry = array![1.0_f64, 1.0, 1.0];
        let age_exit = array![2.0_f64, 4.0, 8.0];
        let event_target = array![0u8, 1u8, 0u8];
        let event_competing = array![0u8, 0u8, 0u8];
        let sampleweight = array![1.0, 1.0, 1.0];
        let x_entry = array![
            [1.0, age_entry[0].ln()],
            [1.0, age_entry[1].ln()],
            [1.0, age_entry[2].ln()]
        ];
        let x_exit = array![
            [1.0, age_exit[0].ln()],
            [1.0, age_exit[1].ln()],
            [1.0, age_exit[2].ln()]
        ];
        let x_derivative = array![[0.0, 0.5], [0.0, 0.25], [0.0, 0.125]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = MonotonicityPenalty { tolerance: 1e-8 };

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("construct linear survival model");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("monotonicity constraints");
        assert_eq!(constraints.a.nrows(), 1);
        assert!((constraints.a[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!((constraints.b[0] - 8e-8).abs() <= 1e-12);
    }

    #[test]
    fn monotonicity_constraints_skip_numericallyzerorows() {
        let age_entry = array![1.0_f64, 1.0, 1.0];
        let age_exit = array![2.0_f64, 3.0, 4.0];
        let event_target = array![0u8, 0u8, 0u8];
        let event_competing = array![0u8, 0u8, 0u8];
        let sampleweight = array![1.0, 1.0, 1.0];
        let x_entry = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let x_exit = x_entry.clone();
        let x_derivative = array![[0.0, 0.0], [0.0, 1e-16], [0.0, 0.25]];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct survival model");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("nonzero derivative row should remain");
        assert_eq!(constraints.a.nrows(), 1);
        assert!((constraints.a[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!(constraints.b[0].abs() <= 1e-18);
    }

    #[test]
    fn censoredrows_allowzero_boundary_derivative() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![0u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.0]];
        let x_exit = array![[0.0]];
        let x_derivative = array![[1.0]];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct censored survival model");

        let state = model
            .update_state(&array![0.0])
            .expect("censored boundary derivative should remain feasible with zero tolerance");
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn eventrows_keep_positive_derivative_constraint() {
        let age_entry = array![1.0_f64, 1.0];
        let age_exit = array![2.0_f64, 4.0];
        let event_target = array![0u8, 1u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[0.0], [0.0]];
        let x_exit = array![[0.0], [0.0]];
        let x_derivative = array![[0.5], [0.25]];

        let model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 1e-8 },
            SurvivalSpec::Net,
        )
        .expect("construct mixed survival model");

        let constraints = model
            .monotonicity_linear_constraints()
            .expect("event row should induce positive lower bound");
        assert_eq!(constraints.a.nrows(), 1);
        assert!((constraints.a[[0, 0]] - 1.0).abs() <= 1e-12);
        assert!((constraints.b[0] - 4e-8).abs() <= 1e-18);
    }

    #[test]
    fn structural_monotonicity_clamps_tiny_negative_roundoff() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.0]];
        let x_exit = array![[0.0]];
        let x_derivative = array![[1.0]];
        let mut model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 1e-8 },
            SurvivalSpec::Net,
        )
        .expect("construct survival model");
        model
            .set_structural_monotonicity(true, 1)
            .expect("enable structural monotonicity");

        let state = model
            .update_state(&array![-1e-8])
            .expect("tiny structural roundoff should be clamped");
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn compressed_monotonicity_constraints_preserve_uncompressed_feasible_region() {
        let uncompressed_constraints = LinearInequalityConstraints {
            a: array![
                [0.0, 0.5, 0.0],
                [0.0, 1.0 / 3.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.125, 0.0]
            ],
            b: Array1::from_elem(4, 1e-8),
        };
        let compressed_constraints = compress_positive_collinear_constraints(
            &uncompressed_constraints.a,
            &uncompressed_constraints.b,
        );

        let candidates = [
            array![0.0, 1e-9, 0.0],
            array![0.0, 4e-8, 0.0],
            array![0.0, 8e-8, 0.0],
            array![0.0, 2e-7, 1.5],
        ];
        for beta in candidates {
            let uncompressed_ok = (0..uncompressed_constraints.a.nrows()).all(|i| {
                uncompressed_constraints.a.row(i).dot(&beta) >= uncompressed_constraints.b[i]
            });
            let compressed_ok = (0..compressed_constraints.a.nrows())
                .all(|i| compressed_constraints.a.row(i).dot(&beta) >= compressed_constraints.b[i]);
            assert_eq!(compressed_ok, uncompressed_ok);
        }
    }

    #[test]
    fn exact_survival_derivatives_are_time_unit_invariant_up_to_constant_shift() {
        let age_entry = array![10.0_f64, 20.0, 25.0];
        let age_exit = array![15.0_f64, 30.0, 40.0];
        let event_target = array![1u8, 0u8, 1u8];
        let event_competing = array![0u8, 0u8, 0u8];
        let sampleweight = array![1.0, 2.0, 0.5];
        let x_entry = array![[0.1, 0.2, 1.0], [0.3, 0.4, 1.0], [0.2, 0.6, 1.0]];
        let x_exit = array![[0.2, 0.3, 1.0], [0.5, 0.7, 1.0], [0.4, 0.8, 1.0]];
        let x_derivative = array![[0.04, 0.02, 0.0], [0.03, 0.01, 0.0], [0.02, 0.03, 0.0]];
        let beta = array![0.8, 1.1, -0.2];

        let base_model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct base survival model");
        let base_state = base_model
            .update_state(&beta)
            .expect("evaluate base survival state");

        let time_scale = 365.25;
        let scaled_age_entry = age_entry.mapv(|v| v * time_scale);
        let scaled_age_exit = age_exit.mapv(|v| v * time_scale);
        let scaled_x_derivative = x_derivative.mapv(|v| v / time_scale);
        let scaled_model = WorkingModelSurvival::from_engine_inputs(
            SurvivalEngineInputs {
                age_entry: scaled_age_entry.view(),
                age_exit: scaled_age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: scaled_x_derivative.view(),
            },
            PenaltyBlocks::new(Vec::new()),
            MonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct scaled survival model");
        let scaled_state = scaled_model
            .update_state(&beta)
            .expect("evaluate scaled survival state");

        let weighted_events = sampleweight
            .iter()
            .zip(event_target.iter())
            .map(|(w, d)| *w * f64::from(*d))
            .sum::<f64>();
        let expected_deviance_shift = 2.0 * weighted_events * time_scale.ln();
        assert!(
            (scaled_state.deviance - base_state.deviance - expected_deviance_shift).abs() <= 1e-10,
            "deviance shift mismatch: scaled={} base={} expected_shift={expected_deviance_shift}",
            scaled_state.deviance,
            base_state.deviance
        );

        for j in 0..beta.len() {
            assert!(
                (scaled_state.gradient[j] - base_state.gradient[j]).abs() <= 1e-12,
                "gradient mismatch at j={j}: scaled={} base={}",
                scaled_state.gradient[j],
                base_state.gradient[j]
            );
        }

        let base_hessian = base_state.hessian.to_dense();
        let scaled_hessian = scaled_state.hessian.to_dense();
        for r in 0..beta.len() {
            for c in 0..beta.len() {
                assert!(
                    (scaled_hessian[[r, c]] - base_hessian[[r, c]]).abs() <= 1e-12,
                    "hessian mismatch at ({r},{c}): scaled={} base={}",
                    scaled_hessian[[r, c]],
                    base_hessian[[r, c]]
                );
            }
        }
    }
}
