//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure for Generalized Additive
//! Models (GAMs). It determines optimal smoothing parameters directly from the data,
//! moving beyond simple hyperparameter-driven models. This is achieved through a
//! nested optimization scheme, a standard approach for this class of models:
//!
//! 1.  Outer Loop (planner-selected optimizer): Optimizes the log-smoothing
//!     parameters (`rho`) by maximizing a marginal likelihood criterion. For
//!     non-Gaussian models (e.g., Logit), this is the Laplace Approximate
//!     Marginal Likelihood (LAML). The concrete solver is chosen centrally by
//!     `outer_strategy` from the derivative capability of the model path:
//!     ARC with analytic Hessian when available, Newton trust-region with a
//!     finite-difference Hessian for small analytic-gradient problems, BFGS
//!     for larger gradient-only problems, and EFS / hybrid EFS when the
//!     hyperparameter geometry admits those fixed-point updates.
//!
//! 2.  Inner Loop (P-IRLS): For each set of trial smoothing parameters from the
//!     outer loop, this routine finds the corresponding model coefficients (`beta`) by
//!     running a Penalized Iteratively Reweighted Least Squares (P-IRLS) algorithm
//!     to convergence.
//!
//! This two-tiered structure allows the model to learn the appropriate complexity for
//! each smooth term directly from the data.

use self::reml::{DirectionalHyperParam, RemlState};
// BlockwiseFitResult and SurvivalLocationScaleFitResult are now type aliases
// for UnifiedFitResult, defined in their respective modules.
use std::fmt;
use std::time::Instant;

// Crate-level imports
use crate::construction::ReparamInvariant;
use crate::diagnostics::should_emit_h_min_eig_diag;
use crate::inference::predict::se_from_covariance;
use crate::linalg::utils::{
    KahanSum, add_relative_diag_ridge, enforce_symmetry, matrix_inversewith_regularization,
    row_mismatch_message,
};
use crate::matrix::DesignMatrix;
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::pirls::{self, PirlsResult};
use crate::seeding::{SeedConfig, SeedRiskProfile};
use crate::terms::smooth::BlockwisePenalty;
use crate::types::{
    Coefficients, GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LatentCLogLogState,
    LikelihoodFamily, LikelihoodScaleMetadata, LinkFunction, LogLikelihoodNormalization,
    LogSmoothingParamsView, MixtureLinkState, RidgePassport, SasLinkState,
};
use crate::types::{MixtureLinkSpec, SasLinkSpec};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
// faer: high-performance dense solvers
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, fast_ab, fast_atb,
};
use faer::{MatRef, Side};
use rayon::prelude::*;

use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::linalg::utils::StableSolver;

// Note: deflateweights_by_se was removed. We now use integrated (GHQ)
// family-dispatched likelihood updates in PIRLS instead of weight deflation.
// The SE is passed through to PIRLS which integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

use std::ops::Range;

/// A penalty specification for the public estimate API.
///
/// `Block` stores only the active sub-block and its column range, avoiding
/// the O(p^2) cost of embedding into a full penalty matrix.
/// `Dense` stores a full `p x p` penalty matrix for callers that already
/// have one.
#[derive(Debug, Clone)]
pub enum PenaltySpec {
    /// Block-local penalty: `local` is `block_dim x block_dim`,
    /// applied to columns `col_range` of the coefficient vector.
    Block {
        local: Array2<f64>,
        col_range: Range<usize>,
        /// Optional structural hint for fast-path spectral decomposition.
        structure_hint: Option<crate::terms::smooth::PenaltyStructureHint>,
    },
    /// Full dense penalty matrix (`p x p`).
    Dense(Array2<f64>),
}

impl PenaltySpec {
    /// The column range this penalty covers.
    /// For `Dense`, this is `0..p` where `p = m.ncols()`.
    pub fn col_range(&self, p: usize) -> Range<usize> {
        match self {
            PenaltySpec::Block { col_range, .. } => col_range.clone(),
            PenaltySpec::Dense(m) => {
                debug_assert_eq!(m.ncols(), p);
                0..p
            }
        }
    }

    /// Convert from a `BlockwisePenalty`, preserving the structure hint.
    pub fn from_blockwise(bp: crate::terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local,
            col_range: bp.col_range,
            structure_hint: bp.structure_hint,
        }
    }

    pub fn from_blockwise_ref(bp: &crate::terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local.clone(),
            col_range: bp.col_range.clone(),
            structure_hint: bp.structure_hint.clone(),
        }
    }

    /// Materialize the full `p x p` dense penalty matrix.
    /// For `Dense`, this is a clone.  For `Block`, this embeds `local` into a
    /// zero matrix at the given `col_range`.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => m.clone(),
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let p = col_range.end.max(local.nrows());
                // Caller should supply p externally when the total dim is larger;
                // this is the best we can do without it.
                let mut out = Array2::zeros((p, p));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }

    /// Materialize the full `p_total x p_total` dense penalty matrix.
    /// For `Dense`, this is a clone (asserts that it matches `p_total`).
    /// For `Block`, this embeds `local` into a `p_total x p_total` zero matrix.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => {
                debug_assert_eq!(m.nrows(), p_total);
                m.clone()
            }
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let mut out = Array2::zeros((p_total, p_total));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }
}

const KAHAN_SWITCH_ELEMS: usize = 10_000;
const FD_REL_GAP_THRESHOLD: f64 = 0.2;
const FD_MIN_BASE_STEP: f64 = 1e-6;
const FD_MAX_REFINEMENTS: usize = 4;
const FD_RIDGE_REL_JITTER_THRESHOLD: f64 = 1e-3;
const FD_RIDGE_ABS_JITTER_THRESHOLD: f64 = 1e-12;

fn faer_frob_inner(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let (m, n) = (a.nrows(), a.ncols());
    let elem_count = m.saturating_mul(n);
    if elem_count < KAHAN_SWITCH_ELEMS {
        let mut sum = 0.0_f64;
        for j in 0..n {
            for i in 0..m {
                sum += a[(i, j)] * b[(i, j)];
            }
        }
        sum
    } else {
        let mut sum = KahanSum::default();
        for j in 0..n {
            for i in 0..m {
                sum.add(a[(i, j)] * b[(i, j)]);
            }
        }
        sum.sum()
    }
}

fn kahan_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut acc = KahanSum::default();
    for value in iter {
        acc.add(value);
    }
    acc.sum()
}

trait FdGradientState<E> {
    fn compute_cost(&self, rho: &Array1<f64>) -> Result<f64, E>;
    fn compute_gradient(&self, rho: &Array1<f64>) -> Result<Array1<f64>, E>;
    fn last_ridge_used(&self) -> Option<f64>;
}

struct FdEval {
    f_p: f64,
    f_m: f64,
    f_p2: f64,
    f_m2: f64,
    d_small: f64,
    d_big: f64,
    ridge_min: f64,
    ridge_max: f64,
    ridge_rel_span: f64,
    ridge_jitter: bool,
}

fn evaluate_fd_pair<S, E>(
    reml_state: &S,
    rho: &Array1<f64>,
    coord: usize,
    base_h: f64,
) -> Result<FdEval, E>
where
    S: FdGradientState<E>,
{
    let mut rho_p = rho.clone();
    rho_p[coord] += 0.5 * base_h;
    let mut rho_m = rho.clone();
    rho_m[coord] -= 0.5 * base_h;
    let f_p = reml_state.compute_cost(&rho_p)?;
    let ridge_p = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m = reml_state.compute_cost(&rho_m)?;
    let ridge_m = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_small = (f_p - f_m) / base_h;

    let h2 = 2.0 * base_h;
    let mut rho_p2 = rho.clone();
    rho_p2[coord] += 0.5 * h2;
    let mut rho_m2 = rho.clone();
    rho_m2[coord] -= 0.5 * h2;
    let f_p2 = reml_state.compute_cost(&rho_p2)?;
    let ridge_p2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m2 = reml_state.compute_cost(&rho_m2)?;
    let ridge_m2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_big = (f_p2 - f_m2) / h2;

    let finite_ridges: Vec<f64> = [ridge_p, ridge_m, ridge_p2, ridge_m2]
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    let (ridge_min, ridge_max, ridge_span, ridge_rel_span) = if finite_ridges.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for v in finite_ridges {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let span = max_v - min_v;
        let rel = span / max_v.abs().max(1e-12);
        (min_v, max_v, span, rel)
    };
    let ridge_jitter = ridge_span.is_finite()
        && ridge_rel_span.is_finite()
        && (ridge_span > FD_RIDGE_ABS_JITTER_THRESHOLD
            && ridge_rel_span > FD_RIDGE_REL_JITTER_THRESHOLD);

    Ok(FdEval {
        f_p,
        f_m,
        f_p2,
        f_m2,
        d_small,
        d_big,
        ridge_min,
        ridge_max,
        ridge_rel_span,
        ridge_jitter,
    })
}

fn fd_same_sign(d_small: f64, d_big: f64) -> bool {
    if !d_small.is_finite() || !d_big.is_finite() {
        false
    } else {
        (d_small >= 0.0 && d_big >= 0.0) || (d_small <= 0.0 && d_big <= 0.0)
    }
}

fn select_fd_derivative(d_small: f64, d_big: f64, same_sign: bool) -> f64 {
    match (d_small.is_finite(), d_big.is_finite()) {
        (true, true) => {
            if same_sign {
                d_small
            } else {
                d_big
            }
        }
        (true, false) => d_small,
        (false, true) => d_big,
        (false, false) => 0.0,
    }
}

fn compute_fd_gradient<S, E>(
    reml_state: &S,
    rho: &Array1<f64>,
    emit_logs: bool,
    allow_analytic_fallback: bool,
) -> Result<Array1<f64>, E>
where
    S: FdGradientState<E>,
{
    let mut fd_grad = Array1::zeros(rho.len());
    let mut analytic_fallback: Option<Array1<f64>> = None;

    let mut log_lines: Vec<String> = Vec::new();
    let (rho_min, rho_max) = rho
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let rho_summary = format!("len={} range=[{:.3e},{:.3e}]", rho.len(), rho_min, rho_max);
    match reml_state.last_ridge_used() {
        Some(ridge) => log_lines.push(format!(
            "[FD RIDGE] Baseline cached ridge: {ridge:.3e} for rho {rho_summary}",
        )),
        None => log_lines.push(format!(
            "[FD RIDGE] No cached baseline ridge available for rho {rho_summary}",
        )),
    }

    for i in 0..rho.len() {
        let h_rel = 1e-4_f64 * (1.0 + rho[i].abs());
        let h_abs = 1e-5_f64;
        let mut base_h = h_rel.max(h_abs);

        log_lines.push(format!("[FD RIDGE] coord {i} rho={:+.6e}", rho[i]));

        let mut d_small = 0.0;
        let mut d_big = 0.0;
        let mut derivative: Option<f64> = None;
        let mut best_rel_gap = f64::INFINITY;
        let mut best_derivative: Option<f64> = None;
        let mut last_rel_gap = f64::INFINITY;
        let mut refine_steps = 0usize;
        let mut rel_gap_first = None;
        let mut rel_gap_max = 0.0;
        let mut ridge_jitter_seen = false;
        let mut ridge_rel_span_max = 0.0;
        let h_start = base_h;

        for attempt in 0..=FD_MAX_REFINEMENTS {
            let eval = evaluate_fd_pair(reml_state, rho, i, base_h)?;
            d_small = eval.d_small;
            d_big = eval.d_big;
            ridge_jitter_seen |= eval.ridge_jitter;
            if eval.ridge_rel_span.is_finite() && eval.ridge_rel_span > ridge_rel_span_max {
                ridge_rel_span_max = eval.ridge_rel_span;
            }

            let denom = d_small.abs().max(d_big.abs()).max(1e-12);
            let rel_gap = (d_small - d_big).abs() / denom;
            let same_sign = fd_same_sign(d_small, d_big);

            if same_sign && !eval.ridge_jitter {
                if rel_gap <= best_rel_gap {
                    best_rel_gap = rel_gap;
                    best_derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
                }
                if rel_gap > last_rel_gap {
                    derivative = best_derivative;
                    break;
                }
                last_rel_gap = rel_gap;
            }

            let refine_for_rel_gap =
                same_sign && rel_gap > FD_REL_GAP_THRESHOLD && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refine_for_ridge = eval.ridge_jitter && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refining = refine_for_rel_gap || refine_for_ridge;
            if attempt == 0 {
                rel_gap_first = Some(rel_gap);
            }
            if rel_gap.is_finite() && rel_gap > rel_gap_max {
                rel_gap_max = rel_gap;
            }
            let last_attempt = attempt == FD_MAX_REFINEMENTS || !refining;
            if attempt == 0 || last_attempt {
                if attempt == 0 {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} f(+/-0.5h)={:+.9e}/{:+.9e} \
f(+/-1h)={:+.9e}/{:+.9e} d_small={:+.9e} d_big={:+.9e} ridge=[{:.3e},{:.3e}]",
                        attempt + 1,
                        base_h,
                        eval.f_p,
                        eval.f_m,
                        eval.f_p2,
                        eval.f_m2,
                        d_small,
                        d_big,
                        eval.ridge_min,
                        eval.ridge_max,
                    ));
                } else {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} d_small={:+.9e} d_big={:+.9e} \
rel_gap={:.3e} ridge=[{:.3e},{:.3e}] ridge_rel_span={:.3e}",
                        attempt + 1,
                        base_h,
                        d_small,
                        d_big,
                        rel_gap,
                        eval.ridge_min,
                        eval.ridge_max,
                        eval.ridge_rel_span
                    ));
                }
            }

            if refining {
                base_h *= 0.5;
                refine_steps += 1;
                continue;
            }

            if eval.ridge_jitter {
                derivative = None;
            } else {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
            break;
        }

        if derivative.is_none() {
            let same_sign = fd_same_sign(d_small, d_big);
            if same_sign && !ridge_jitter_seen {
                derivative = best_derivative
                    .or_else(|| Some(select_fd_derivative(d_small, d_big, same_sign)));
            } else if !ridge_jitter_seen {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
        }

        if derivative.is_none() && allow_analytic_fallback {
            if analytic_fallback.is_none() {
                analytic_fallback = Some(reml_state.compute_gradient(rho)?);
            }
            derivative = analytic_fallback.as_ref().map(|g| g[i]);
            log_lines.push(format!(
                "[FD RIDGE]   coord {} fallback to analytic gradient due to ridge jitter (max rel span {:.3e})",
                i, ridge_rel_span_max
            ));
        }

        fd_grad[i] = derivative.unwrap_or(f64::NAN);
        let rel_gap_first = rel_gap_first.unwrap_or(f64::NAN);
        log_lines.push(format!(
            "[FD RIDGE]   refine steps={} h_start={:.3e} h_final={:.3e} rel_gap_first={:.3e} rel_gap_max={:.3e} ridge_jitter_seen={} ridge_rel_span_max={:.3e}",
            refine_steps,
            h_start,
            base_h,
            rel_gap_first,
            rel_gap_max,
            ridge_jitter_seen,
            ridge_rel_span_max
        ));
        log_lines.push(format!(
            "[FD RIDGE]   chosen derivative = {:+.9e}",
            fd_grad[i]
        ));
    }

    if emit_logs && !log_lines.is_empty() {
        println!("{}", log_lines.join("\n"));
    }

    Ok(fd_grad)
}

impl FdGradientState<EstimationError> for RemlState<'_> {
    fn compute_cost(&self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        RemlState::compute_cost(self, rho)
    }

    fn compute_gradient(&self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        RemlState::compute_gradient(self, rho)
    }

    fn last_ridge_used(&self) -> Option<f64> {
        RemlState::last_ridge_used(self)
    }
}

#[derive(Clone, Debug)]
struct ParametricColumnConditioning {
    intercept_idx: Option<usize>,
    columns: Vec<(usize, f64, f64)>,
}

impl ParametricColumnConditioning {
    /// Build conditioning from explicit unpenalized column indices.
    ///
    /// Reads only the specified columns from `x` (via `extract_column`) to
    /// compute per-column mean/variance — no full-design densification.
    fn from_column_indices(x: &DesignMatrix, unpenalized_cols: &[usize]) -> Self {
        const SCALE_EPS: f64 = 1e-12;
        let n = x.nrows();
        if n == 0 {
            return Self {
                intercept_idx: None,
                columns: Vec::new(),
            };
        }
        let mut intercept_idx = None;
        let mut columns = Vec::new();
        for &j in unpenalized_cols {
            let col = x.extract_column(j);
            let first = col[0];
            let is_constant = col.iter().all(|&v| (v - first).abs() <= 1e-12);
            if is_constant {
                if (first - 1.0).abs() <= 1e-12 && intercept_idx.is_none() {
                    intercept_idx = Some(j);
                }
                continue;
            }
            let mean = col.iter().copied().sum::<f64>() / n as f64;
            let var = col
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            if !var.is_finite() || var <= SCALE_EPS * SCALE_EPS {
                continue;
            }
            columns.push((j, mean, var.sqrt()));
        }
        if intercept_idx.is_none() {
            for (_, mean, _) in &mut columns {
                *mean = 0.0;
            }
        }
        Self {
            intercept_idx,
            columns,
        }
    }

    /// Infer unpenalized columns from `PenaltySpec` slices.
    fn infer_from_penalty_specs(x: &DesignMatrix, specs: &[PenaltySpec]) -> Self {
        let p = x.ncols();
        let mut penalized = vec![false; p];
        for spec in specs {
            let range = spec.col_range(p);
            for j in range {
                penalized[j] = true;
            }
        }
        let unpenalized: Vec<usize> = (0..p).filter(|&j| !penalized[j]).collect();
        Self::from_column_indices(x, &unpenalized)
    }

    fn is_active(&self) -> bool {
        !self.columns.is_empty()
    }

    /// Return a lazily-conditioned design matrix (no materialization).
    ///
    /// Wraps `x` in a `ConditionedDesign` operator that applies per-column
    /// centering and scaling through matvec algebra, avoiding densification.
    fn apply_to_design(&self, x: &DesignMatrix) -> DesignMatrix {
        if !self.is_active() {
            return x.clone();
        }
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
            crate::matrix::ConditionedDesign::new(x.clone(), self.columns.clone()),
        )))
    }

    fn transform_constraint_matrix_to_internal(&self, a_original: &Array2<f64>) -> Array2<f64> {
        let mut out = a_original.clone();
        for &(j, mean, scale) in &self.columns {
            let intercept_col = self.intercept_idx.map(|idx| out.column(idx).to_owned());
            let mut target = out.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col
            {
                target += &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v * scale);
            }
        }
        out
    }

    fn transform_linear_constraints_to_internal(
        &self,
        constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Option<crate::pirls::LinearInequalityConstraints> {
        constraints.map(|constraints| crate::pirls::LinearInequalityConstraints {
            a: self.transform_constraint_matrix_to_internal(&constraints.a),
            b: constraints.b,
        })
    }

    fn backtransform_beta(&self, beta_internal: &Array1<f64>) -> Array1<f64> {
        let mut beta = beta_internal.clone();
        for &(j, mean, scale) in &self.columns {
            if let Some(intercept_idx) = self.intercept_idx {
                beta[intercept_idx] -= beta_internal[j] * mean / scale;
            }
            beta[j] = beta_internal[j] / scale;
        }
        beta
    }

    fn transform_matrix_columnswith_a(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        self.transform_matrix_columnswith_a_inplace(&mut out);
        out
    }

    fn transform_matrix_columnswith_a_inplace(&self, mat: &mut Array2<f64>) {
        if !self.is_active() {
            return;
        }
        let intercept_col = self.intercept_idx.map(|idx| mat.column(idx).to_owned());
        for &(j, mean, scale) in &self.columns {
            let mut target = mat.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col.as_ref()
            {
                target -= &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v / scale);
            }
        }
    }

    fn transform_matrixrowswith_a_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        for &(j, mean, scale) in &self.columns {
            let interceptrow = self.intercept_idx.map(|idx| out.row(idx).to_owned());
            let mut target = out.row_mut(j);
            if mean != 0.0
                && let Some(interceptrow) = interceptrow
            {
                target -= &(interceptrow * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v / scale);
            }
        }
        out
    }

    fn transform_matrix_columnswith_b(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        for &(j, mean, scale) in &self.columns {
            let intercept_col = self.intercept_idx.map(|idx| out.column(idx).to_owned());
            let mut target = out.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col
            {
                target += &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v * scale);
            }
        }
        out
    }

    fn transform_matrixrowswith_b_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        for &(j, mean, scale) in &self.columns {
            let interceptrow = self.intercept_idx.map(|idx| out.row(idx).to_owned());
            let mut target = out.row_mut(j);
            if mean != 0.0
                && let Some(interceptrow) = interceptrow
            {
                target += &(interceptrow * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v * scale);
            }
        }
        out
    }

    fn backtransform_covariance(&self, cov_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.transform_matrix_columnswith_a(cov_internal);
        self.transform_matrixrowswith_a_transpose(&right)
    }

    fn backtransform_penalized_hessian(&self, h_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.transform_matrix_columnswith_b(h_internal);
        self.transform_matrixrowswith_b_transpose(&right)
    }

    fn backtransform_external_result(
        &self,
        mut result: ExternalOptimResult,
    ) -> ExternalOptimResult {
        if !self.is_active() {
            return result;
        }
        result.beta = self.backtransform_beta(&result.beta);
        if let Some(inf) = result.inference.as_mut() {
            inf.penalized_hessian = self.backtransform_penalized_hessian(&inf.penalized_hessian);
            inf.beta_covariance = inf
                .beta_covariance
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.beta_standard_errors = inf.beta_covariance.as_ref().map(se_from_covariance);
            inf.beta_covariance_corrected = inf
                .beta_covariance_corrected
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.beta_standard_errors_corrected = inf
                .beta_covariance_corrected
                .as_ref()
                .map(se_from_covariance);
            inf.smoothing_correction = inf
                .smoothing_correction
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.reparam_qs = None;
        }
        result.constraint_kkt = None;
        result.artifacts = FitArtifacts {
            pirls: None,
            ..Default::default()
        };
        result
    }
}

fn map_hessian_to_original_basis(
    pirls: &crate::pirls::PirlsResult,
) -> Result<Array2<f64>, EstimationError> {
    let qs = &pirls.reparam_result.qs;
    let h_t = &pirls.penalized_hessian_transformed;
    // H_original = Qs * H_transformed * Qs'
    // left_dot_matrix avoids densification for sparse Hessians.
    let tmp = h_t.left_dot_matrix(qs);
    Ok(tmp.dot(&qs.t()))
}

#[derive(Clone)]
pub(crate) struct RemlConfig {
    likelihood: GlmLikelihoodSpec,
    link_kind: InverseLink,
    convergence_tolerance: f64,
    max_iterations: usize,
    reml_convergence_tolerance: f64,
    firth_bias_reduction: bool,
    objective_consistentfdgradient: bool,
}

impl RemlConfig {
    fn external(likelihood: GlmLikelihoodSpec, reml_tol: f64, firth_bias_reduction: bool) -> Self {
        Self {
            likelihood,
            link_kind: InverseLink::Standard(likelihood.link_function()),
            convergence_tolerance: reml_tol,
            max_iterations: 0,
            reml_convergence_tolerance: reml_tol,
            firth_bias_reduction,
            objective_consistentfdgradient: false,
        }
        .with_max_iterations(100)
    }

    pub(crate) fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }

    fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            likelihood: self.likelihood,
            link_kind: self.link_kind.clone(),
            max_iterations: self.max_iterations,
            convergence_tolerance: self.convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
        }
    }
}
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Small ridge added to the LAML Hessian before inversion, for numerical stability
/// when smoothing parameters are weakly identified.
const LAML_RIDGE: f64 = 1e-8;
/// Minimum penalized deviance floor.
pub(crate) const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region for the deviance floor.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;
const MAX_PIRLS_CACHE_ENTRIES: usize = 128;
// Unified rho bound corresponding to lambda in [exp(-RHO_BOUND), exp(RHO_BOUND)].
// Additional headroom reduces frequent contact with the hard box constraints.
pub(crate) const RHO_BOUND: f64 = 30.0;
// Soft interior prior on rho near the box boundaries.
const RHO_SOFT_PRIOR_WEIGHT: f64 = 1e-6;
const RHO_SOFT_PRIOR_SHARPNESS: f64 = 4.0;
// Adaptive cubature guardrails for bounded correction latency.
const AUTO_CUBATURE_MAX_RHO_DIM: usize = 12;
const AUTO_CUBATURE_MAX_EIGENVECTORS: usize = 4;
const AUTO_CUBATURE_TARGET_VAR_FRAC: f64 = 0.95;
const AUTO_CUBATURE_MAX_BETA_DIM: usize = 1600;
const AUTO_CUBATURE_BOUNDARY_MARGIN: f64 = 2.0;

/// Smooth approximation of `max(dp, DP_FLOOR)` that is differentiable.
///
/// Returns the smoothed value and its derivative with respect to `dp`.
pub(crate) fn smooth_floor_dp(dp: f64) -> (f64, f64) {
    let tau = DP_FLOOR_SMOOTH_WIDTH.max(f64::EPSILON);
    let scaled = (dp - DP_FLOOR) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = DP_FLOOR + tau * softplus;
    (dp_c, sigma)
}

/// Compute the smoothing parameter uncertainty correction matrix `V_corr = J * V_rho * J^T`.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for `beta` is: `V*_beta = V_beta + J * V_rho * J^T`.
/// where:
/// - `V_beta = H^{-1}` (conditional covariance treating `lambda` as fixed)
/// - `J = d(beta)/d(rho)` (Jacobian wrt log-smoothing parameters)
/// - `V_rho = (d^2 LAML / d rho^2)^{-1}` (outer covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
///
/// Full correction reference.
/// Let `rho ~ N(mu, Sigma)` with `mu = rho_hat`, `Sigma = V_rho`,
/// and define:
/// - `A(rho) = H_rho^{-1}`
/// - `b(rho) = beta_hat_rho`
///
/// The exact Gaussian-mixture identity is:
///   `Var(beta) = E[A(rho)] + Var(b(rho))`.
///
/// Around `mu`, this routine keeps the first-order terms:
///
///   `E[A(rho)]      ~= A(mu) = Hmu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= Hmu^{-1} + J V_rho J^T`.
///
/// Equivalent first-order propagation around the outer optimum `rho*`:
///
///   `Var(beta_hat) ~= Var(beta_hat | rho_hat) + (d beta_hat / d rho) Var(rho_hat) (d beta_hat / d rho)^T`
///                  `= V_beta + J V_rho J^T`.
///
/// Components:
///   `J[:,k] = d(beta_hat)/d(rho_k) = -H^{-1}(A_k beta_hat),  A_k = exp(rho_k) S_k`
///   `V_rho  = (d^2 V / d rho^2 at rho*)^{-1}`
///
/// Exact non-Gaussian V_ρ^{-1} requires the full Hessian with:
///   - tr(H^{-1}H_{kℓ})
///   - tr(H^{-1}H_k H^{-1}H_ℓ)
///   - pseudo-det second derivatives in S
///   - and H_{kℓ} terms containing fourth-likelihood derivatives.
///
/// This routine estimates V_ρ^{-1} by finite-differencing
/// the implemented analytic gradient and then regularizing before inversion.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
pub(crate) struct SmoothingCorrectionComputation {
    pub correction: Option<Array2<f64>>,
    pub hessian_rho: Option<Array2<f64>>,
}

fn invert_regularized_rho_hessian(hessian_rho: &Array2<f64>) -> Option<(Array2<f64>, bool)> {
    if let Ok(chol) = hessian_rho.cholesky(faer::Side::Lower) {
        let n = hessian_rho.nrows();
        let mut inverse = Array2::<f64>::eye(n);
        for col in 0..n {
            let colvec = inverse.column(col).to_owned();
            let solved = chol.solvevec(&colvec);
            inverse.column_mut(col).assign(&solved);
        }
        return Some((inverse, false));
    }

    let (eigenvalues, eigenvectors) = hessian_rho.eigh(faer::Side::Lower).ok()?;
    if eigenvalues.iter().any(|v| !v.is_finite()) || !eigenvectors.iter().all(|v| v.is_finite()) {
        return None;
    }

    let n = hessian_rho.nrows();
    let spectral_scale = eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let floor = (spectral_scale * 1e-10).max(LAML_RIDGE);
    let mut inverse = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let lambda = eigenvalues[i].max(floor);
        let inv_lambda = 1.0 / lambda;
        let v = eigenvectors.column(i);
        for row in 0..n {
            for col in 0..n {
                inverse[[row, col]] += inv_lambda * v[row] * v[col];
            }
        }
    }
    Some((inverse, true))
}

fn compute_smoothing_correction(
    reml_state: &RemlState<'_>,
    final_rho: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
) -> SmoothingCorrectionComputation {
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
        };
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    let n_coeffs_orig = final_fit.reparam_result.qs.nrows();
    let lambdas: Array1<f64> = final_rho.mapv(f64::exp);

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ β̂),   S_k^ρ = λ_k S_k, λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k β̂ = R_kᵀ(R_k β̂),
    // so each Jacobian column is one linear solve with H.

    // Use the same objective-consistent inner Hessian surface used by REML:
    // - non-Firth: H = X'W_HX + S (+ stabilization if present)
    // - Firth logit: H_total = H - d²Phi/dβ²
    // Fallback to PIRLS stabilized Hessian only if bundle recovery fails.
    //
    // Conclusion:
    //   J[:,k] = dβ̂/dρ_k must use the Jacobian of the actual stationarity
    //   system G*(β,ρ)=0, i.e. H_total for Firth-adjusted fits. Using only
    //   X'W_HX+S here would be inconsistent with the fitted objective and would
    //   misstate smoothing-parameter uncertainty propagation.
    let h_trans = reml_state
        .objective_innerhessian(final_rho)
        .unwrap_or_else(|_| final_fit.stabilizedhessian_transformed.to_dense());

    // Factor the Hessian for solving
    let h_chol = match h_trans.cholesky(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("Cholesky decomposition failed for smoothing correction; skipping.");
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
            };
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    let ct = &final_fit.reparam_result.canonical_transformed;

    // Build Jacobian matrix J where column k is dβ/dρ_k
    let mut jacobian_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    for k in 0..n_rho {
        if k >= ct.len() {
            continue;
        }
        let cp = &ct[k];
        if cp.rank() == 0 {
            continue;
        }
        // S_k β — block-local: R^T (R β[block]), embedded into p-vector.
        let r = &cp.col_range;
        let beta_block = beta_trans.slice(s![r.start..r.end]);
        let r_beta = cp.root.dot(&beta_block);
        let mut s_k_beta = Array1::<f64>::zeros(n_coeffs_trans);
        for a in 0..cp.block_dim() {
            s_k_beta[r.start + a] = (0..cp.rank())
                .map(|row| cp.root[[row, a]] * r_beta[row])
                .sum::<f64>();
        }

        // dβ/dρ_k = -H^{-1}(λ_k S_k β)
        let rhs = s_k_beta.mapv(|v| -lambdas[k] * v);
        let delta = h_chol.solvevec(&rhs);

        jacobian_trans.column_mut(k).assign(&delta);
    }

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // The authoritative inner-strategy path chooses the rho-space Hessian
    // evaluation policy here. Unified may still perform local numerical
    // salvage inside the exact branch, but the branch choice itself no longer
    // lives inline at the call site.
    let mut hessian_rho = match reml_state.compute_lamlhessian_consistent(final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::warn!(
                "LAML Hessian unavailable ({}); skipping smoothing correction.",
                err
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
            };
        }
    };

    // Symmetrize the Hessian
    enforce_symmetry(&mut hessian_rho);

    // Step 3: Invert Hessian to get V_rho.
    // Add a small ridge before factorization to regularize weakly identified ρ directions.
    add_relative_diag_ridge(&mut hessian_rho, LAML_RIDGE, LAML_RIDGE);

    let (v_rho, repaired_hessian) = match invert_regularized_rho_hessian(&hessian_rho) {
        Some(inverse) => inverse,
        None => {
            log::warn!(
                "Failed to invert LAML Hessian for smoothing correction after spectral repair; skipping."
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: Some(hessian_rho),
            };
        }
    };
    if repaired_hessian {
        log::debug!(
            "Projected indefinite LAML Hessian onto a positive spectrum before smoothing correction inversion."
        );
    }

    // Step 4: Compute V_corr = J * V_rho * J^T in transformed space.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let jv_rho = jacobian_trans.dot(&v_rho); // (n_coeffs_trans x n_rho)
    let v_corr_trans = jv_rho.dot(&jacobian_trans.t()); // (n_coeffs_trans x n_coeffs_trans)

    // Step 5: Transform back to original coefficient basis:
    // V_corr_orig = Qs * V_corr_trans * Qs^T
    let qs = &final_fit.reparam_result.qs;
    let qsv = qs.dot(&v_corr_trans);
    let v_corr_orig = qsv.dot(&qs.t());

    // Validate the result
    if !v_corr_orig.iter().all(|v| v.is_finite()) {
        log::warn!("Non-finite values in smoothing correction matrix; skipping.");
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
        };
    }

    // Ensure positive semi-definiteness by clamping negative eigenvalues
    // (can happen due to numerical noise)
    match v_corr_orig.eigh(faer::Side::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_eig < -1e-10 {
                log::debug!(
                    "Smoothing correction has negative eigenvalue {:.3e}; clamping to zero.",
                    min_eig
                );
                // Reconstruct with clamped eigenvalues
                let mut result = Array2::<f64>::zeros((n_coeffs_orig, n_coeffs_orig));
                for i in 0..n_coeffs_orig {
                    let eig = eigenvalues[i].max(0.0);
                    let v = eigenvectors.column(i);
                    for j in 0..n_coeffs_orig {
                        for k in 0..n_coeffs_orig {
                            result[[j, k]] += eig * v[j] * v[k];
                        }
                    }
                }
                return SmoothingCorrectionComputation {
                    correction: Some(result),
                    hessian_rho: Some(hessian_rho),
                };
            }
        }
        Err(_) => {
            log::warn!("Eigendecomposition failed for smoothing correction validation.");
        }
    }
    SmoothingCorrectionComputation {
        correction: Some(v_corr_orig),
        hessian_rho: Some(hessian_rho),
    }
}

/// A comprehensive error type for the model estimation process.
#[derive(Error)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] crate::basis::BasisError),

    #[error("A linear system solve failed. The penalized Hessian may be singular. Error: {0}")]
    LinearSystemSolveFailed(FaerLinalgError),

    #[error("Eigendecomposition failed: {0}")]
    EigendecompositionFailed(FaerLinalgError),

    #[error("Parameter constraint violation: {0}")]
    ParameterConstraintViolation(String),

    #[error(
        "The P-IRLS inner loop did not converge within {max_iterations} iterations. Last gradient norm was {last_change:.6e}."
    )]
    PirlsDidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during model fitting at iteration {iteration}. \
        The model cannot converge because a predictor perfectly separates the binary outcomes. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    PerfectSeparationDetected { iteration: usize, max_abs_eta: f64 },

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error("REML smoothing optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    #[error("An internal error occurred during model layout or coefficient mapping: {0}")]
    LayoutError(String),

    #[error(
        "Model is over-parameterized: {num_coeffs} coefficients for {num_samples} samples.\n\n\
        Coefficient Breakdown:\n\
          - Intercept:                     {intercept_coeffs}\n\
          - Binary Main Effects:           {binary_main_coeffs}\n\
          - Primary Smooth Effects:        {primary_smooth_coeffs}\n\
          - Binary×Primary Interactions:   {binary_primary_interaction_coeffs}\n\
          - Auxiliary Main Effects:        {aux_main_coeffs}\n\
          - Auxiliary Interactions:        {aux_interaction_coeffs}"
    )]
    ModelOverparameterized {
        num_coeffs: usize,
        num_samples: usize,
        intercept_coeffs: usize,
        binary_main_coeffs: usize,
        primary_smooth_coeffs: usize,
        aux_main_coeffs: usize,
        binary_primary_interaction_coeffs: usize,
        aux_interaction_coeffs: usize,
    },

    #[error(
        "Model is ill-conditioned with condition number {condition_number:.2e}. This typically occurs when the model is over-parameterized (too many knots relative to data points). Consider reducing the number of knots or increasing regularization."
    )]
    ModelIsIllConditioned { condition_number: f64 },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Calibrator training failed: {0}")]
    CalibratorTrainingFailed(String),

    #[error("Invalid specification: {0}")]
    InvalidSpecification(String),

    #[error("Prediction error")]
    PredictionError,
}

// Ensure Debug prints with actual line breaks by delegating to Display
impl core::fmt::Debug for EstimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

//
// This uses the joint model architecture where the base predictor and
// flexible link are fitted together in one optimization with REML.
//
// The model is: η = g(Xβ) where g is a learned flexible link function.
// Domain-specific training orchestration is handled by caller adapters.
// The gam engine exposes matrix/family-based external-design APIs for supported
// GLM-style families: fit_gam / optimize_external_design.

pub struct ExternalOptimResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: LikelihoodFamily,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    /// Residual scale on the response scale.
    ///
    /// Contract: Gaussian identity models store the residual standard
    /// deviation sigma here. Non-Gaussian families keep the response-scale
    /// summary used by their explicit likelihood-scale metadata.
    pub standard_deviation: f64,
    pub iterations: usize,
    pub finalgrad_norm: f64,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub deviance: f64,
    pub stable_penalty_term: f64,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    pub artifacts: FitArtifacts,
    pub inference: Option<FitInference>,
    pub reml_score: f64,
    pub fitted_link: FittedLinkState,
}

#[derive(Clone)]
pub struct ExternalOptimOptions {
    pub family: crate::types::LikelihoodFamily,
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Optional explicit Firth override for external fitting families that
    /// support Jeffreys/Firth bias reduction.
    /// - `Some(true)`: force Firth on
    /// - `Some(false)`: force Firth off
    /// - `None`: use family default behavior
    pub firth_bias_reduction: Option<bool>,
    /// Relative shrinkage floor for penalized block eigenvalues.
    /// See [`FitOptions::penalty_shrinkage_floor`] for details.
    pub penalty_shrinkage_floor: Option<f64>,
    /// Fixed prior on smoothing parameters used by joint HMC fallback.
    pub rho_prior: crate::types::RhoPrior,
    /// Kronecker-factored penalty system for tensor-product smooth terms.
    pub kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis for P-IRLS factored reparameterization.
    pub kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
}

fn resolve_external_family(
    family: crate::types::LikelihoodFamily,
    firth_override: Option<bool>,
) -> Result<(GlmLikelihoodSpec, bool), EstimationError> {
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "optimize_external_design does not support RoystonParmar; use survival training APIs"
                .to_string(),
        ));
    }

    if firth_override == Some(true) && !family.supports_firth() {
        return Err(EstimationError::InvalidInput(format!(
            "firth_bias_reduction is currently implemented only for {}; {} does not support it",
            crate::types::LikelihoodFamily::BinomialLogit.pretty_name(),
            family.pretty_name()
        )));
    }

    Ok((
        GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::try_from(family).map_err(|msg| {
            EstimationError::InvalidInput(format!(
                "optimize_external_design requires a GLM family; {msg}"
            ))
        })?),
        firth_override.unwrap_or_else(|| family.supports_firth()) && family.supports_firth(),
    ))
}

#[inline]
fn effective_sas_link_for_family(
    family: crate::types::LikelihoodFamily,
    sas_link: Option<SasLinkSpec>,
) -> Option<SasLinkSpec> {
    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic
    ) && sas_link.is_none()
    {
        Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        })
    } else {
        sas_link
    }
}

#[inline]
fn resolved_external_inverse_link(
    link: LinkFunction,
    latent_cloglog: Option<LatentCLogLogState>,
    mixture_link: Option<&MixtureLinkSpec>,
    sas_link: Option<SasLinkSpec>,
) -> Result<InverseLink, EstimationError> {
    if let Some(state) = latent_cloglog {
        return Ok(InverseLink::LatentCLogLog(state));
    }
    if let Some(spec) = mixture_link {
        return Ok(InverseLink::Mixture(state_fromspec(spec).map_err(|e| {
            EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
        })?));
    }
    if let Some(spec) = sas_link {
        return Ok(match link {
            LinkFunction::BetaLogistic => {
                InverseLink::BetaLogistic(state_from_beta_logisticspec(spec).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                })?)
            }
            _ => InverseLink::Sas(
                state_from_sasspec(spec)
                    .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?,
            ),
        });
    }
    Ok(InverseLink::Standard(link))
}

#[inline]
fn resolved_external_config(
    opts: &ExternalOptimOptions,
) -> Result<(RemlConfig, Option<SasLinkSpec>), EstimationError> {
    if opts.latent_cloglog.is_some() && (opts.mixture_link.is_some() || opts.sas_link.is_some()) {
        return Err(EstimationError::InvalidInput(
            "latent_cloglog cannot be combined with mixture_link or sas_link".to_string(),
        ));
    }
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        return Err(EstimationError::InvalidInput(
            "mixture_link and sas_link are mutually exclusive".to_string(),
        ));
    }
    if matches!(
        opts.family,
        crate::types::LikelihoodFamily::BinomialLatentCLogLog
    ) && opts.latent_cloglog.is_none()
    {
        return Err(EstimationError::InvalidInput(
            "BinomialLatentCLogLog requires latent_cloglog state".to_string(),
        ));
    }
    if opts.latent_cloglog.is_some()
        && !matches!(
            opts.family,
            crate::types::LikelihoodFamily::BinomialLatentCLogLog
        )
    {
        return Err(EstimationError::InvalidInput(
            "latent_cloglog is only supported with BinomialLatentCLogLog".to_string(),
        ));
    }
    let effective_sas_link = effective_sas_link_for_family(opts.family, opts.sas_link);
    let (likelihood, firth_active) =
        resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let mut cfg = RemlConfig::external(likelihood, opts.tol, firth_active);
    let link = likelihood.link_function();
    cfg.link_kind = resolved_external_inverse_link(
        link,
        opts.latent_cloglog,
        opts.mixture_link.as_ref(),
        effective_sas_link,
    )?;
    Ok((cfg, effective_sas_link))
}

#[inline]
fn ensure_exact_directional_hyper_supported(
    _: LinkFunction,
    _: bool,
    _: bool,
    _: &str,
) -> Result<(), EstimationError> {
    // Kept as a central compatibility hook for API-level validation.
    //
    // Current status:
    // - Dense exact path supports Firth-logit directional hyper-gradients for
    //   both penalty-only and design-moving directions.
    // - Sparse exact path supports the same Firth-logit directional
    //   hyper-gradients via sparse Cholesky solves plus the shared dense
    //   Fisher-information reduction.
    //
    // No additional API-level restrictions are needed here.
    Ok(())
}

fn validate_penalty_specs(
    specs: &[PenaltySpec],
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    for (idx, spec) in specs.iter().enumerate() {
        match spec {
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let bd = col_range.len();
                if local.nrows() != bd || local.ncols() != bd {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: block penalty {idx} local matrix must be {bd}x{bd}, got {}x{}",
                        local.nrows(),
                        local.ncols()
                    )));
                }
                if col_range.end > p {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: block penalty {idx} col_range {}..{} exceeds p={p}",
                        col_range.start, col_range.end
                    )));
                }
            }
            PenaltySpec::Dense(m) => {
                if m.nrows() != p || m.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                        m.nrows(),
                        m.ncols()
                    )));
                }
            }
        }
    }
    Ok(())
}

// canonicalize_active_penalties removed — replaced by
// crate::construction::canonicalize_penalty_specs.

/// Optimize smoothing parameters for an external design using the same REML/LAML machinery.
/// Contract: likelihood dispatch is determined by `opts.family`.
pub fn optimize_external_design<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    optimize_external_designwith_heuristic_lambdas(y, w, x, offset, s_list, None, opts)
}

/// Same as `optimize_external_design`, but allows heuristic λ warm-start seeds
/// for the outer smoothing search.
pub fn optimize_external_designwith_heuristic_lambdas<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    heuristic_lambdas: Option<&[f64]>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list
        .into_iter()
        .map(PenaltySpec::from_blockwise)
        .collect();
    optimize_external_designwith_heuristic_lambdas_andwarm_start(
        y,
        w,
        x,
        offset,
        specs,
        heuristic_lambdas,
        None,
        opts,
    )
}

fn optimize_external_designwith_heuristic_lambdas_andwarm_start<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<PenaltySpec>,
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    if matches!(opts.family, crate::types::LikelihoodFamily::BinomialMixture)
        && opts.mixture_link.is_none()
    {
        return Err(EstimationError::InvalidInput(
            "BinomialMixture requires mixture_link specification".to_string(),
        ));
    }
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&s_list, p, "optimize_external_design")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &s_list,
        &opts.nullspace_dims,
        p,
        "optimize_external_design",
    )?;
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &s_list);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
    let k = canonical.len();
    if active_nullspace_dims.len() != k {
        return Err(EstimationError::InvalidInput(format!(
            "nullspace_dims length mismatch: expected {k} entries for active penalties, got {}",
            active_nullspace_dims.len()
        )));
    }
    let (cfg, effective_sas_link) = resolved_external_config(opts)?;

    // Own the external arrays once; the conditioned design is shared through `reml_state`.
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x;
    let offset_o = offset.to_owned();
    let canonical_shared = Arc::new(canonical);
    let mut reml_state = RemlState::newwith_offset_shared(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        Arc::clone(&canonical_shared),
        p,
        &cfg,
        Some(active_nullspace_dims.clone()),
        None,
        fit_linear_constraints.clone(),
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    if let Some(kron) = opts.kronecker_penalty_system.clone() {
        reml_state.set_kronecker_penalty_system(kron);
    }
    if let Some(kf) = opts.kronecker_factored.clone() {
        reml_state.set_kronecker_factored(kf);
    }
    reml_state.setwarm_start_original_beta(warm_start_beta);

    let reml_seed_config = SeedConfig {
        bounds: (-12.0, 12.0),
        max_seeds: if k <= 4 {
            8
        } else if k <= 12 {
            10
        } else {
            12
        },
        screening_budget: if k <= 6 { 2 } else { 3 },
        screen_max_inner_iterations: if matches!(cfg.link_function(), LinkFunction::Identity) {
            3
        } else {
            5
        },
        risk_profile: if matches!(cfg.link_function(), LinkFunction::Identity) {
            SeedRiskProfile::Gaussian
        } else {
            SeedRiskProfile::GeneralizedLinear
        },
        num_auxiliary_trailing: 0,
    };
    let reml_tol = cfg.reml_convergence_tolerance;
    let reml_max_iter = opts.max_iter;
    let outer_eval_idx = AtomicUsize::new(0usize);
    let mixture_optspec = if opts.optimize_mixture {
        opts.mixture_link.clone()
    } else {
        None
    };
    let sas_optspec = if opts.optimize_sas {
        effective_sas_link
    } else {
        None
    };
    let mixture_dim = mixture_optspec
        .as_ref()
        .map(|s| s.initial_rho.len())
        .unwrap_or(0);
    let sas_dim = if sas_optspec.is_some() { 2 } else { 0 };
    let sasridgeweight = if sas_dim > 0 {
        sas_log_deltaridgeweight()
    } else {
        0.0
    };
    let (
        final_rho,
        final_mixture_state,
        final_sas_state,
        final_mixture_param_covariance,
        final_sas_param_covariance,
        outer_result,
    ) = if mixture_dim > 0 && sas_dim > 0 {
        return Err(EstimationError::InvalidInput(
            "simultaneous mixture and SAS optimization is not supported".to_string(),
        ));
    } else if mixture_dim == 0 && sas_dim == 0 {
        use crate::solver::outer_strategy::{Derivative, HessianResult, OuterEval, OuterProblem};

        let problem = OuterProblem::new(k)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Analytic)
            .with_barrier(self::reml::unified::BarrierConfig::from_constraints(
                fit_linear_constraints.as_ref(),
            ))
            .with_tolerance(reml_tol)
            .with_max_iter(reml_max_iter)
            .with_fd_step(1e-3)
            .with_seed_config(reml_seed_config.clone())
            .with_rho_bound(crate::estimate::RHO_BOUND)
            .with_screening_cap(reml_state.screening_max_inner_iterations.clone());
        let problem = if let Some(ref h) = heuristic_lambdas {
            problem.with_heuristic_lambdas(h.to_vec())
        } else {
            problem
        };

        let mut obj = problem.build_objective(
            &mut reml_state,
            |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| state.compute_cost(rho),
            |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| {
                outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                let cost = state.compute_cost(rho)?;
                let grad = state.compute_gradient(rho)?;
                let hessian = state.compute_lamlhessian_consistent(rho).ok();
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: match hessian {
                        Some(h) => HessianResult::Analytic(h),
                        None => HessianResult::Unavailable,
                    },
                })
            },
            None::<fn(&mut &mut self::reml::RemlState<'_>)>,
            Some(
                |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.compute_efs_steps(rho)
                },
            ),
        );

        let strategy_result = problem.run(&mut obj, "standard REML")?;
        (
            strategy_result.rho.clone(),
            cfg.link_kind.mixture_state().cloned(),
            cfg.link_kind.sas_state().copied(),
            None,
            None,
            strategy_result,
        )
    } else {
        let use_mixture = mixture_dim > 0;
        let use_sas = sas_dim > 0;
        let use_beta_logistic =
            use_sas && matches!(cfg.link_function(), LinkFunction::BetaLogistic);
        let theta_dim = k + mixture_dim + sas_dim;
        let sasspec = sas_optspec;
        let mixspec = mixture_optspec
            .clone()
            .or_else(|| {
                if use_mixture {
                    None
                } else {
                    Some(MixtureLinkSpec {
                        components: Vec::new(),
                        initial_rho: Array1::zeros(0),
                    })
                }
            })
            .ok_or_else(|| EstimationError::InvalidInput("missing mixture spec".to_string()))?;
        let mut heuristic_theta = Vec::new();
        if let Some(hvals) = heuristic_lambdas {
            if hvals.len() == k {
                heuristic_theta.extend_from_slice(hvals);
                if use_mixture {
                    heuristic_theta
                        .extend_from_slice(mixspec.initial_rho.as_slice().unwrap_or(&[]));
                }
                if let Some(spec) = sasspec {
                    heuristic_theta.push(spec.initial_epsilon);
                    heuristic_theta.push(spec.initial_log_delta);
                }
            }
        }
        let heuristic_theta_ref = if heuristic_theta.len() == theta_dim {
            Some(heuristic_theta.as_slice())
        } else {
            None
        };
        let aux_dim_outer = if use_mixture { mixture_dim } else { sas_dim };
        let mut reml_seed_config_mix = reml_seed_config.clone();
        reml_seed_config_mix.num_auxiliary_trailing = aux_dim_outer;
        use crate::solver::outer_strategy::{Derivative, HessianResult, OuterEval, OuterProblem};
        let initial_link_kind = cfg.link_kind.clone();
        let problem = OuterProblem::new(theta_dim)
            .with_gradient(Derivative::Analytic)
            .with_hessian(Derivative::Unavailable)
            .with_psi_dim(mixture_dim + sas_dim)
            .with_barrier(self::reml::unified::BarrierConfig::from_constraints(
                fit_linear_constraints.as_ref(),
            ))
            .with_tolerance(reml_tol)
            .with_max_iter(reml_max_iter)
            .with_fd_step(1e-3)
            .with_seed_config(reml_seed_config_mix.clone())
            .with_rho_bound(crate::estimate::RHO_BOUND)
            .with_screening_cap(reml_state.screening_max_inner_iterations.clone());
        let problem = if let Some(h) = heuristic_theta_ref {
            problem.with_heuristic_lambdas(h.to_vec())
        } else {
            problem
        };
        // Shared helper: parse theta into rho + link params, update link state.
        let apply_link_theta = |state: &mut &mut self::reml::RemlState<'_>,
                                theta: &Array1<f64>|
         -> Result<Array1<f64>, EstimationError> {
            let rho = theta.slice(s![..k]).to_owned();
            let mut cfg_eval = cfg.clone();
            if use_mixture {
                let mix_rho = theta.slice(s![k..(k + mixture_dim)]).to_owned();
                cfg_eval.link_kind = InverseLink::Mixture(
                    state_fromspec(&MixtureLinkSpec {
                        components: mixspec.components.clone(),
                        initial_rho: mix_rho,
                    })
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                    })?,
                );
            }
            if use_sas {
                let epsilon = if use_beta_logistic {
                    theta[k]
                } else {
                    let (v, _) = sas_effective_epsilon(theta[k]);
                    v
                };
                let delta_like = theta[k + 1];
                cfg_eval.link_kind = if use_beta_logistic {
                    InverseLink::BetaLogistic(
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: epsilon,
                            initial_log_delta: delta_like,
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!(
                                "invalid Beta-Logistic link: {e}"
                            ))
                        })?,
                    )
                } else {
                    InverseLink::Sas(
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: epsilon,
                            initial_log_delta: delta_like,
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                        })?,
                    )
                };
            }
            state.set_link_states(
                cfg_eval.link_kind.mixture_state().cloned(),
                cfg_eval.link_kind.sas_state().copied(),
            );
            Ok(rho)
        };

        // SAS ridge/barrier cost correction (shared between cost_fn, eval_fn, efs_fn).
        let sas_ridge_cost = |theta: &Array1<f64>| -> f64 {
            let sasridge = if use_sas && !use_beta_logistic {
                sasridgeweight
            } else {
                0.0
            };
            if use_sas && sasridge > 0.0 {
                let log_delta = theta[k + 1];
                let mut extra = 0.5 * sasridge * log_delta * log_delta;
                if !use_beta_logistic {
                    let (barriercost, _) = sas_log_delta_edge_barriercostgrad(log_delta);
                    extra += barriercost;
                }
                extra
            } else {
                0.0
            }
        };

        let mut obj = problem.build_objective(
            &mut reml_state,
            |state: &mut &mut self::reml::RemlState<'_>, theta: &Array1<f64>| {
                let rho = apply_link_theta(state, theta)?;
                let cost = state.compute_cost(&rho)? + sas_ridge_cost(theta);
                Ok(cost)
            },
            |state: &mut &mut self::reml::RemlState<'_>, theta: &Array1<f64>| {
                let eval_idx = outer_eval_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let rho = apply_link_theta(state, theta)?;
                let tcost = Instant::now();

                // Use the unified REML evaluator with link ext_coords.
                // This computes ρ gradient AND link parameter gradient jointly
                // through the same HyperCoord infrastructure used for aniso ψ.
                let eval_mode = self::reml::unified::EvalMode::ValueAndGradient;
                let result = state.evaluate_unified_with_link_ext(&rho, eval_mode)?;

                let cost = result.cost + sas_ridge_cost(theta);
                let mut grad = result.gradient.ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "unified evaluator returned no gradient in ValueAndGradient mode"
                            .to_string(),
                    )
                })?;

                debug_assert_eq!(
                    grad.len(),
                    theta_dim,
                    "unified evaluator gradient length {} != theta_dim {}",
                    grad.len(),
                    theta_dim
                );

                // SAS epsilon reparameterization chain rule.
                if use_sas && !use_beta_logistic {
                    let (_, d_eps_d_raw) = sas_effective_epsilon(theta[k]);
                    grad[k] *= d_eps_d_raw;
                }
                // SAS log_delta ridge + barrier gradient.
                if use_sas && !use_beta_logistic && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    grad[k + 1] += sasridgeweight * log_delta;
                    let (_, barriergrad) = sas_log_delta_edge_barriercostgrad(log_delta);
                    grad[k + 1] += barriergrad;
                }

                let cost_sec = tcost.elapsed().as_secs_f64();
                let aux_dim = if use_mixture { mixture_dim } else { sas_dim };
                log::debug!(
                    "[outer-eval {eval_idx}] theta_dim={} aux_dim={} unified_link_ext time_sec={:.3}",
                    theta_dim,
                    aux_dim,
                    cost_sec,
                );
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianResult::Unavailable,
                })
            },
            Some(|state: &mut &mut self::reml::RemlState<'_>| {
                state.set_link_states(
                    initial_link_kind.mixture_state().cloned(),
                    initial_link_kind.sas_state().copied(),
                );
            }),
            Some(
                |state: &mut &mut self::reml::RemlState<'_>, theta: &Array1<f64>| {
                    let rho = apply_link_theta(state, theta)?;
                    let mut efs_eval = state.compute_efs_steps_with_link_ext(&rho)?;

                    // SAS reparameterization chain rule on ψ steps.
                    if use_sas && !use_beta_logistic {
                        let (_, d_eps_d_raw) = sas_effective_epsilon(theta[k]);
                        if efs_eval.steps.len() > k {
                            efs_eval.steps[k] *= d_eps_d_raw;
                        }
                        if let Some(ref mut pg) = efs_eval.psi_gradient {
                            if !pg.is_empty() {
                                pg[0] *= d_eps_d_raw;
                            }
                        }
                    }

                    efs_eval.cost += sas_ridge_cost(theta);
                    Ok(efs_eval)
                },
            ),
        );
        let outer_result = problem.run(&mut obj, "mixture/SAS flexible link")?;
        let final_rho = outer_result.rho.slice(s![..k]).to_owned();
        let final_mix_state = if use_mixture {
            let final_mix_rho = outer_result.rho.slice(s![k..(k + mixture_dim)]).to_owned();
            Some(
                state_fromspec(&MixtureLinkSpec {
                    components: mixspec.components.clone(),
                    initial_rho: final_mix_rho,
                })
                .map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                })?,
            )
        } else {
            None
        };
        let final_sas_state = if use_sas {
            let epsilon_eff = if use_beta_logistic {
                outer_result.rho[k]
            } else {
                let (v, _) = sas_effective_epsilon(outer_result.rho[k]);
                v
            };
            Some(if use_beta_logistic {
                state_from_beta_logisticspec(SasLinkSpec {
                    initial_epsilon: epsilon_eff,
                    initial_log_delta: outer_result.rho[k + 1],
                })
                .map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                })?
            } else {
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: epsilon_eff,
                    initial_log_delta: outer_result.rho[k + 1],
                })
                .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?
            })
        } else {
            cfg.link_kind.sas_state().copied()
        };
        let aux_param_covariance = if aux_dim_outer > 0 {
            let mut theta_aux = Array1::<f64>::zeros(aux_dim_outer);
            if use_mixture {
                theta_aux.assign(&outer_result.rho.slice(s![k..(k + mixture_dim)]));
            } else if use_sas {
                theta_aux[0] = outer_result.rho[k];
                theta_aux[1] = outer_result.rho[k + 1];
            }
            let mut evaluate_auxcost = |aux: &Array1<f64>| -> Result<f64, EstimationError> {
                if use_mixture {
                    let spec_eval = MixtureLinkSpec {
                        components: mixspec.components.clone(),
                        initial_rho: aux.clone(),
                    };
                    let mix_state = state_fromspec(&spec_eval).map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                    })?;
                    reml_state.set_link_states(Some(mix_state), None);
                } else if use_sas {
                    let epsilon_eff = if use_beta_logistic {
                        aux[0]
                    } else {
                        let (v, _) = sas_effective_epsilon(aux[0]);
                        v
                    };
                    let sas_state = if use_beta_logistic {
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: epsilon_eff,
                            initial_log_delta: aux[1],
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!(
                                "invalid Beta-Logistic link: {e}"
                            ))
                        })?
                    } else {
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: epsilon_eff,
                            initial_log_delta: aux[1],
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                        })?
                    };
                    reml_state.set_link_states(None, Some(sas_state));
                }
                let mut cost = reml_state.compute_cost(&final_rho)?;
                if use_sas && !use_beta_logistic && sasridgeweight > 0.0 {
                    let log_delta = aux[1];
                    cost += 0.5 * sasridgeweight * log_delta * log_delta;
                    let (barriercost, _) = sas_log_delta_edge_barriercostgrad(log_delta);
                    cost += barriercost;
                }
                Ok(cost)
            };
            let f0 = evaluate_auxcost(&theta_aux)?;
            let d = aux_dim_outer;
            let mut h_aux = Array2::<f64>::zeros((d, d));
            for i in 0..d {
                let hi = 1e-3 * (1.0 + theta_aux[i].abs());
                let mut tp = theta_aux.clone();
                let mut tm = theta_aux.clone();
                tp[i] += hi;
                tm[i] -= hi;
                let fp = evaluate_auxcost(&tp)?;
                let fm = evaluate_auxcost(&tm)?;
                h_aux[[i, i]] = (fp - 2.0 * f0 + fm) / (hi * hi);
                for j in (i + 1)..d {
                    let hj = 1e-3 * (1.0 + theta_aux[j].abs());
                    let mut tpp = theta_aux.clone();
                    let mut tpm = theta_aux.clone();
                    let mut tmp = theta_aux.clone();
                    let mut tmm = theta_aux.clone();
                    tpp[i] += hi;
                    tpp[j] += hj;
                    tpm[i] += hi;
                    tpm[j] -= hj;
                    tmp[i] -= hi;
                    tmp[j] += hj;
                    tmm[i] -= hi;
                    tmm[j] -= hj;
                    let fpp = evaluate_auxcost(&tpp)?;
                    let fpm = evaluate_auxcost(&tpm)?;
                    let fmp = evaluate_auxcost(&tmp)?;
                    let fmm = evaluate_auxcost(&tmm)?;
                    let hij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                    h_aux[[i, j]] = hij;
                    h_aux[[j, i]] = hij;
                }
            }
            let max_diag = h_aux
                .diag()
                .iter()
                .copied()
                .map(f64::abs)
                .fold(1.0_f64, f64::max);
            for i in 0..d {
                if !h_aux[[i, i]].is_finite() || h_aux[[i, i]] <= 0.0 {
                    h_aux[[i, i]] = max_diag;
                }
            }
            matrix_inversewith_regularization(&h_aux, "link parameter covariance")
        } else {
            None
        };
        let (mix_cov, sas_cov) = if use_mixture {
            (aux_param_covariance, None)
        } else if use_sas {
            (None, aux_param_covariance)
        } else {
            (None, None)
        };
        (
            final_rho,
            final_mix_state,
            final_sas_state,
            mix_cov,
            sas_cov,
            outer_result,
        )
    };
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, outer_result.iterations);
    let (pirls_res, _) = pirls::fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(final_rho.view()),
        pirls::PirlsProblem {
            x: reml_state.x(),
            offset: offset_o.view(),
            y: y_o.view(),
            priorweights: w_o.view(),
            covariate_se: None,
        },
        pirls::PenaltyConfig {
            canonical_penalties: reml_state.canonical_penalties(),
            balanced_penalty_root: Some(reml_state.balanced_penalty_root()),
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: fit_linear_constraints.as_ref(),
            penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
            kronecker_factored: None,
        },
        &pirls::PirlsConfig {
            link_kind: if let Some(state) = final_mixture_state.clone() {
                InverseLink::Mixture(state)
            } else if let Some(state) = final_sas_state {
                if matches!(cfg.link_function(), LinkFunction::BetaLogistic) {
                    InverseLink::BetaLogistic(state)
                } else {
                    InverseLink::Sas(state)
                }
            } else {
                cfg.link_kind.clone()
            },
            ..cfg.as_pirls_config()
        },
        None,
    )?;

    // Map beta back to original basis
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());
    let beta_orig = conditioning.backtransform_beta(&beta_orig_internal);

    // Weighted residual sum of squares for Gaussian models
    let n = y_o.len() as f64;
    let weighted_rss = if matches!(cfg.link_function(), LinkFunction::Identity) {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrixvectormultiply(&beta_orig);
            eta
        };
        let resid = y_o.to_owned() - &fitted;
        w_o.iter()
            .zip(resid.iter())
            .map(|(&wi, &ri)| wi * ri * ri)
            .sum()
    } else {
        0.0
    };

    // ---- Skewness diagnostic & automatic joint (β, ρ) HMC fallback ----
    //
    // For non-Gaussian models, check if the Laplace approximation to the
    // marginal likelihood is reliable. If the posterior has high skewness,
    // LAML picks the wrong λ. We detect this and automatically fall back
    // to joint (β, ρ) NUTS sampling which bypasses Laplace entirely.
    let (final_rho, pirls_res) = if !matches!(cfg.link_function(), LinkFunction::Identity)
        && !pirls_res.solve_c_array.is_empty()
    {
        let h_eff = &pirls_res.stabilizedhessian_transformed;
        let p_eff = h_eff.ncols();
        let hessian_dense = h_eff.to_dense();

        let c_arr = {
            let mut c = pirls_res.solve_c_array.clone();
            for val in c.iter_mut() {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }
            c
        };
        let (max_skewness, skewness_vec) = crate::hmc::laplace_directional_cubic_diagnostic(
            &hessian_dense,
            &pirls_res.x_transformed,
            &c_arr,
        )
        .unwrap_or_else(|_| (0.0, Array1::zeros(p_eff)));
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum EstimationApproach {
            Reml,
            JointHmc,
        }

        // `run_outer` handles solver selection inside the REML approach.
        // This loop only handles cross-paradigm escalation from REML to joint HMC.
        let approaches = if max_skewness > crate::hmc::SKEWNESS_HMC_THRESHOLD {
            log::warn!(
                "[REML] High directional cubic non-Gaussianity detected (max |gamma_r| = {:.3}). \
                 Escalating from REML to joint (β, ρ) HMC refinement.",
                max_skewness,
            );
            let n_skewed = skewness_vec.iter().filter(|s| s.abs() > 0.3).count();
            log::info!(
                "[Joint HMC] {}/{} Hessian directions have |gamma_r| > 0.3",
                n_skewed,
                p_eff,
            );
            vec![EstimationApproach::Reml, EstimationApproach::JointHmc]
        } else {
            if max_skewness > 0.1 {
                log::info!(
                    "[REML] Directional cubic non-Gaussianity moderate (max |gamma_r| = {:.3}); \
                     TK correction sufficient.",
                    max_skewness,
                );
            }
            vec![EstimationApproach::Reml]
        };

        let mut selected_rho = final_rho;
        let mut selected_pirls_res = pirls_res;

        for approach in approaches {
            match approach {
                EstimationApproach::Reml => {}
                EstimationApproach::JointHmc => {
                    let firth_bias_reduction = selected_pirls_res.jeffreys_logdet().is_some();
                    if firth_bias_reduction && !opts.family.supports_firth() {
                        return Err(EstimationError::InvalidInput(format!(
                            "joint HMC Firth refinement requires a Firth-supported family; {} does not support it",
                            opts.family.pretty_name()
                        )));
                    }

                    let sampling_result = {
                        // Guard: joint HMC densifies design and Hessian — refuse at biobank scale.
                        let hmc_n = selected_pirls_res.x_transformed.nrows();
                        let hmc_p = selected_pirls_res.x_transformed.ncols();
                        const HMC_MAX_DENSE_WORK: usize = 50_000_000;
                        if hmc_n.saturating_mul(hmc_p) > HMC_MAX_DENSE_WORK {
                            return Err(EstimationError::InvalidInput(format!(
                                "joint HMC requires dense design materialization (n={hmc_n}, p={hmc_p}); \
                                 this is a small-model-only feature. Use --approach=laplace for large models."
                            )));
                        }
                        let x_dense = selected_pirls_res.x_transformed.to_dense_arc();
                        let hessian_dense =
                            selected_pirls_res.stabilizedhessian_transformed.to_dense();
                        let hmc_inverse_link = if let Some(state) = final_mixture_state.clone() {
                            InverseLink::Mixture(state)
                        } else if let Some(state) = final_sas_state.clone() {
                            if matches!(cfg.link_function(), LinkFunction::BetaLogistic) {
                                InverseLink::BetaLogistic(state)
                            } else {
                                InverseLink::Sas(state)
                            }
                        } else {
                            cfg.link_kind.clone()
                        };
                        let hmc_inputs = crate::hmc::JointBetaRhoInputs {
                            x: x_dense.view(),
                            y: y_o.view(),
                            weights: w_o.view(),
                            likelihood_family: opts.family,
                            inverse_link: hmc_inverse_link,
                            gamma_shape: selected_pirls_res.likelihood.gamma_shape(),
                            mode: selected_pirls_res.beta_transformed.view(),
                            hessian: hessian_dense.view(),
                            penalty_roots: selected_pirls_res
                                .reparam_result
                                .canonical_transformed
                                .clone(),
                            rho_mode: selected_rho.view(),
                            rho_prior: opts.rho_prior.clone(),
                            firth_bias_reduction,
                            trigger_skewness: max_skewness,
                        };

                        let total_dim = p_eff + selected_rho.len();
                        let hmc_config = crate::hmc::NutsConfig {
                            n_samples: (400 + 50 * total_dim).min(4000),
                            nwarmup: (400 + 50 * total_dim).min(4000),
                            n_chains: 2,
                            target_accept: 0.8,
                            seed: 31_415,
                        };
                        crate::hmc::run_joint_beta_rho_sampling(&hmc_inputs, &hmc_config)
                    };
                    match sampling_result {
                        Ok(result) if result.converged => {
                            log::info!(
                                "[Joint HMC] Converged (R-hat={:.3}, ESS={:.1}). \
                                 Updating smoothing parameters from posterior mean.",
                                result.rhat,
                                result.ess,
                            );
                            let new_rho = result.rho_mean;
                            match pirls::fit_model_for_fixed_rho(
                                LogSmoothingParamsView::new(new_rho.view()),
                                pirls::PirlsProblem {
                                    x: reml_state.x(),
                                    offset: offset_o.view(),
                                    y: y_o.view(),
                                    priorweights: w_o.view(),
                                    covariate_se: None,
                                },
                                pirls::PenaltyConfig {
                                    canonical_penalties: reml_state.canonical_penalties(),
                                    balanced_penalty_root: Some(reml_state.balanced_penalty_root()),
                                    reparam_invariant: None,
                                    p,
                                    coefficient_lower_bounds: None,
                                    linear_constraints_original: fit_linear_constraints.as_ref(),
                                    penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
                                    kronecker_factored: None,
                                },
                                &pirls::PirlsConfig {
                                    link_kind: if let Some(state) = final_mixture_state.clone() {
                                        InverseLink::Mixture(state)
                                    } else if let Some(state) = final_sas_state.clone() {
                                        if matches!(cfg.link_function(), LinkFunction::BetaLogistic)
                                        {
                                            InverseLink::BetaLogistic(state)
                                        } else {
                                            InverseLink::Sas(state)
                                        }
                                    } else {
                                        cfg.link_kind.clone()
                                    },
                                    ..cfg.as_pirls_config()
                                },
                                None,
                            ) {
                                Ok((new_pirls_res, _)) => {
                                    selected_rho = new_rho;
                                    selected_pirls_res = new_pirls_res;
                                }
                                Err(e) => {
                                    log::warn!(
                                        "[Joint HMC] Re-PIRLS at posterior ρ failed ({:?}); \
                                         keeping REML estimates.",
                                        e,
                                    );
                                }
                            }
                        }
                        Ok(result) => {
                            log::warn!(
                                "[Joint HMC] Did not converge (R-hat={:.3}, ESS={:.1}); \
                                 keeping REML estimates.",
                                result.rhat,
                                result.ess,
                            );
                        }
                        Err(e) => {
                            log::warn!(
                                "[Joint HMC] Sampling failed ({}); keeping REML estimates.",
                                e,
                            );
                        }
                    }
                }
            }
        }

        (selected_rho, selected_pirls_res)
    } else {
        (final_rho, pirls_res)
    };

    // Recompute beta in case pirls_res was updated by joint HMC
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());

    let lambdas = final_rho.mapv(f64::exp);
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank_total = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank_total as f64).max(0.0);
    let mut edf_by_block = vec![0.0; k];
    let mut edf_total = 0.0;
    let mut smoothing_correction = None;
    let mut penalized_hessian = Array2::<f64>::zeros((0, 0));
    let mut beta_covariance = None;
    let mut beta_standard_errors = None;
    let mut beta_covariance_corrected = None;
    let mut beta_standard_errors_corrected = None;

    if opts.compute_inference {
        // EDF by block using stabilized H and penalty roots in transformed basis.
        let h = &pirls_res.stabilizedhessian_transformed;
        let p_dim = h.nrows();
        // Sparse-aware factorization with ridge retry — no densification.
        // Uses SymmetricMatrix::factorize() -> sparse Cholesky for sparse,
        // dense Cholesky for dense.
        let factor = {
            let scale = h.max_abs_diag();
            let min_step = scale * 1e-10;
            let mut ridge = 0.0_f64;
            let mut attempts = 0_usize;
            loop {
                let candidate = if ridge > 0.0 {
                    match h.addridge(ridge) {
                        Ok(c) => c,
                        Err(_) => h.clone(),
                    }
                } else {
                    h.clone()
                };
                if let Ok(f) = candidate.factorize() {
                    if ridge > 0.0 {
                        log::warn!("Stabilized Hessian factorized with ridge {:.3e}", ridge,);
                    }
                    break f;
                }
                attempts += 1;
                if attempts >= MAX_FACTORIZATION_ATTEMPTS {
                    return Err(EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    });
                }
                ridge = if ridge <= 0.0 { min_step } else { ridge * 10.0 };
            }
        };
        let mut traces = vec![0.0f64; k];
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            // Build the p × rank RHS with nonzeros only in [start..end] rows.
            let r = &cp.col_range;
            let rank = cp.rank();
            let mut rhs = Array2::<f64>::zeros((p_dim, rank));
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    rhs[[r.start + row, col]] = cp.root[[col, row]];
                }
            }
            let sol =
                factor
                    .solvemulti(&rhs)
                    .map_err(|_| EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    })?;
            // Frobenius inner product: only the block rows of rhs are nonzero.
            let mut frob = 0.0f64;
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    frob += sol[[r.start + row, col]] * rhs[[r.start + row, col]];
                }
            }
            traces[kk] = lambdas[kk] * frob;
        }
        edf_total = (p_dim as f64 - kahan_sum(traces.iter().copied())).clamp(mp, p_dim as f64);
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            let p_k = cp.rank() as f64;
            let edf_k = (p_k - traces[kk]).clamp(0.0, p_k);
            edf_by_block[kk] = edf_k;
        }
    }

    // Persist residual-based scale for Gaussian identity models.
    // Contract: residual standard deviation sigma, not variance.
    let standard_deviation = match pirls_res.likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => {
            let denom = if opts.compute_inference {
                (n - mp).max(1.0)
            } else {
                n.max(1.0)
            };
            (weighted_rss / denom).sqrt()
        }
        GlmLikelihoodFamily::GammaLog => pirls_res.likelihood.gamma_shape().unwrap_or(1.0),
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture
        | GlmLikelihoodFamily::PoissonLog => 1.0,
    };

    // Compute gradient norm at final rho for reporting
    let finalgrad = reml_state
        .compute_gradient(&final_rho)
        .unwrap_or_else(|_| Array1::from_elem(final_rho.len(), f64::NAN));
    let finalgrad_norm_rho = finalgrad.dot(&finalgrad).sqrt();
    let finalgrad_norm = if finalgrad_norm_rho.is_finite() {
        finalgrad_norm_rho
    } else {
        outer_result.final_grad_norm
    };

    if opts.compute_inference {
        penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
        // Guard: full p×p covariance inversion is O(p³) and only viable for
        // small-to-medium models. For large models, fall back to diagonal-only
        // standard errors from the Hessian diagonal.
        const COV_MAX_P: usize = 5_000;
        let p_cov = penalized_hessian.nrows();
        let diag_fallback = || {
            let mut diag_inv = Array2::<f64>::zeros(penalized_hessian.dim());
            for i in 0..p_cov {
                let d = penalized_hessian[[i, i]];
                if d > 0.0 {
                    diag_inv[[i, i]] = 1.0 / d;
                }
            }
            diag_inv
        };
        beta_covariance = if p_cov > COV_MAX_P {
            log::warn!(
                "skipping full posterior covariance inversion (p={p_cov} > {COV_MAX_P}): \
                 using diagonal-only standard errors"
            );
            Some(diag_fallback())
        } else {
            match matrix_inversewith_regularization(&penalized_hessian, "posterior covariance") {
                Some(cov) => Some(cov),
                None => {
                    log::warn!(
                        "full posterior covariance inversion failed (p={p_cov}): \
                         falling back to diagonal-only standard errors"
                    );
                    Some(diag_fallback())
                }
            }
        };
        smoothing_correction = reml_state.compute_smoothing_correction_auto(
            &final_rho,
            &pirls_res,
            beta_covariance.as_ref(),
            finalgrad_norm,
        );
        beta_standard_errors = beta_covariance.as_ref().map(se_from_covariance);
        beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
            (Some(base_cov), Some(corr)) if base_cov.dim() == corr.dim() => {
                let mut corrected = base_cov.clone();
                corrected += corr;
                for i in 0..corrected.nrows() {
                    for j in (i + 1)..corrected.ncols() {
                        let avg = 0.5 * (corrected[[i, j]] + corrected[[j, i]]);
                        corrected[[i, j]] = avg;
                        corrected[[j, i]] = avg;
                    }
                }
                Some(corrected)
            }
            (Some(_), Some(corr)) => {
                log::warn!(
                    "Skipping corrected covariance: dimension mismatch (base {:?}, corr {:?})",
                    beta_covariance.as_ref().map(Array2::dim),
                    Some(corr.dim())
                );
                None
            }
            _ => None,
        };
        beta_standard_errors_corrected = beta_covariance_corrected.as_ref().map(se_from_covariance);
    }
    let inference = opts.compute_inference.then(|| FitInference {
        edf_by_block,
        edf_total,
        smoothing_correction,
        penalized_hessian,
        working_weights: pirls_res.solveweights.clone(),
        working_response: pirls_res.solveworking_response.clone(),
        reparam_qs: Some(pirls_res.reparam_result.qs.clone()),
        beta_covariance,
        beta_standard_errors,
        beta_covariance_corrected,
        beta_standard_errors_corrected,
    });

    let pirls_status = pirls_res.status;
    let likelihood_spec = pirls_res.likelihood;
    let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
        y_o.view(),
        &pirls_res.finalmu,
        likelihood_spec,
        w_o.view(),
    );

    let result = ExternalOptimResult {
        beta: beta_orig_internal,
        lambdas: lambdas.to_owned(),
        likelihood_family: likelihood_spec.response_family(),
        likelihood_scale: likelihood_spec.scale,
        log_likelihood_normalization: LogLikelihoodNormalization::OmittingResponseConstants,
        log_likelihood,
        standard_deviation,
        iterations: iters,
        finalgrad_norm,
        pirls_status,
        deviance: pirls_res.deviance,
        stable_penalty_term: pirls_res.stable_penalty_term,
        max_abs_eta: pirls_res.max_abs_eta,
        constraint_kkt: pirls_res.constraint_kkt.clone(),
        artifacts: FitArtifacts {
            pirls: Some(pirls_res),
            ..Default::default()
        },
        inference,
        reml_score: outer_result.final_value,
        fitted_link: if let Some(state) = final_mixture_state {
            FittedLinkState::Mixture {
                state,
                covariance: final_mixture_param_covariance,
            }
        } else if let Some(state) = opts.latent_cloglog {
            FittedLinkState::LatentCLogLog { state }
        } else if let Some(state) = final_sas_state {
            match opts.family {
                crate::types::LikelihoodFamily::BinomialSas => FittedLinkState::Sas {
                    state,
                    covariance: final_sas_param_covariance,
                },
                crate::types::LikelihoodFamily::BinomialBetaLogistic => {
                    FittedLinkState::BetaLogistic {
                        state,
                        covariance: final_sas_param_covariance,
                    }
                }
                _ => FittedLinkState::Standard(None),
            }
        } else {
            FittedLinkState::Standard(None)
        },
    };
    Ok(conditioning.backtransform_external_result(result))
}

fn validate_and_build_reml_state<X, T, F>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<PenaltySpec>,
    theta: &Array1<f64>,
    rho_dim: usize,
    mut hyper_dirs: Vec<DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    opts: &ExternalOptimOptions,
    context: &str,
    eval: F,
) -> Result<T, EstimationError>
where
    X: Into<DesignMatrix>,
    F: for<'a> FnOnce(&RemlState<'a>, &[DirectionalHyperParam]) -> Result<T, EstimationError>,
{
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }
    if rho_dim > theta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho_dim {} exceeds theta dimension {}",
            rho_dim,
            theta.len()
        )));
    }
    let p = x.ncols();
    validate_penalty_specs(&s_list, p, context)?;
    let (canonical, active_nullspace_dims) =
        crate::construction::canonicalize_penalty_specs(&s_list, &opts.nullspace_dims, p, context)?;
    if rho_dim != active_nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho_dim mismatch: rho_dim={}, active_penalties={}",
            rho_dim,
            active_nullspace_dims.len()
        )));
    }
    let psi_dim = theta.len() - rho_dim;
    if hyper_dirs.len() != psi_dim {
        return Err(EstimationError::InvalidInput(format!(
            "joint hyper-gradient derivative count mismatch: psi_dim={}, hyper_dirs={}",
            psi_dim,
            hyper_dirs.len()
        )));
    }
    for (idx, hyper_dir) in hyper_dirs.iter().enumerate() {
        for component in hyper_dir.penalty_first_components() {
            if component.penalty_index >= canonical.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "penalty_index for dir {idx} out of bounds: {} >= {}",
                    component.penalty_index,
                    canonical.len()
                )));
            }
        }
        if hyper_dir.x_tau_original.nrows() != x.nrows() || hyper_dir.x_tau_original.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "X_tau[{idx}] must be {}x{}, got {}x{}",
                x.nrows(),
                p,
                hyper_dir.x_tau_original.nrows(),
                hyper_dir.x_tau_original.ncols()
            )));
        }
        RemlState::validate_penalty_component_shapes(
            hyper_dir.penalty_first_components(),
            p,
            &format!("S_tau[{idx}]"),
        )?;
        if let Some(x2) = hyper_dir.x_tau_tau_original.as_ref() {
            if x2.len() != psi_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "X_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    x2.len()
                )));
            }
            for (j, x_ij) in x2.iter().enumerate() {
                let Some(x_ij) = x_ij.as_ref() else {
                    continue;
                };
                if x_ij.nrows() != x.nrows() || x_ij.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "X_tau_tau[{idx}][{j}] must be {}x{}, got {}x{}",
                        x.nrows(),
                        p,
                        x_ij.nrows(),
                        x_ij.ncols()
                    )));
                }
            }
        }
        if let Some(s2) = hyper_dir.penaltysecond_componentrows() {
            if s2.len() != psi_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "S_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    s2.len()
                )));
            }
            for (j, components) in s2.iter().enumerate() {
                let Some(components) = components.as_ref() else {
                    continue;
                };
                RemlState::validate_penalty_component_shapes(
                    components,
                    p,
                    &format!("S_tau_tau[{idx}][{j}]"),
                )?;
            }
        }
    }

    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &s_list);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
    for dir in &mut hyper_dirs {
        let mut x_tau = dir.x_tau_dense();
        conditioning.transform_matrix_columnswith_a_inplace(&mut x_tau);
        dir.x_tau_original = crate::estimate::reml::HyperDesignDerivative::from(x_tau);
        if let Some(rows) = dir.x_tau_tau_original.as_mut() {
            for mat in rows.iter_mut().flatten() {
                let mut dense = mat.materialize();
                conditioning.transform_matrix_columnswith_a_inplace(&mut dense);
                *mat = crate::estimate::reml::HyperDesignDerivative::from(dense);
            }
        }
    }
    let (cfg, _) = resolved_external_config(opts)?;
    let has_design_drift = hyper_dirs
        .iter()
        .any(|dir| dir.x_tau_original.any_nonzero());
    ensure_exact_directional_hyper_supported(
        cfg.link_function(),
        cfg.firth_bias_reduction,
        has_design_drift,
        context,
    )?;
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );
    reml_state.setwarm_start_original_beta(warm_start_beta);
    eval(&reml_state, &hyper_dirs)
}

pub(crate) fn compute_external_joint_hypercostgradienthessian<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: Vec<DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    opts: &ExternalOptimOptions,
) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list
        .into_iter()
        .map(PenaltySpec::from_blockwise)
        .collect();
    validate_and_build_reml_state(
        y,
        w,
        x,
        offset,
        specs,
        theta,
        rho_dim,
        hyper_dirs,
        warm_start_beta,
        opts,
        "compute_external_joint_hypercostgradienthessian",
        |reml_state, conditioned_hyper_dirs| {
            reml_state.compute_joint_hypercostgradienthessian(
                theta,
                rho_dim,
                conditioned_hyper_dirs,
            )
        },
    )
}

#[derive(Clone)]
pub struct FitOptions {
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    pub adaptive_regularization: Option<AdaptiveRegularizationOptions>,
    /// Relative shrinkage floor for penalized block eigenvalues.
    ///
    /// When `Some(epsilon)`, a rho-independent ridge of magnitude
    /// `epsilon * max_balanced_eigenvalue` is added to each eigenvalue of the
    /// combined penalty on the penalized block. This acts as a weak proper
    /// complexity prior that prevents barely-penalized directions from causing
    /// pathological non-Gaussianity in the posterior (e.g., extreme skewness
    /// under logit link with high-dimensional spatial smooths).
    ///
    /// The ridge is rho-independent, so LAML gradients remain correct without
    /// modification (d(epsilon*I)/d(rho_k) = 0).
    ///
    /// Typical value: `Some(1e-6)`. Set to `None` or `Some(0.0)` to disable.
    /// Default: `Some(1e-6)`.
    pub penalty_shrinkage_floor: Option<f64>,
    /// Fixed prior on smoothing parameters used by joint HMC fallback.
    ///
    /// This prior is part of the sampled target itself, unlike `rho_mode`,
    /// which is only used to initialize chains near the REML solution.
    pub rho_prior: crate::types::RhoPrior,
    /// Kronecker-factored penalty system for tensor-product smooth terms.
    /// When set, the REML evaluator uses O(∏q_j) logdet and KroneckerMarginal
    /// penalty coordinates instead of O(p³) eigendecomposition.
    pub kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis for P-IRLS factored reparameterization.
    pub kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRegularizationOptions {
    pub enabled: bool,
    pub max_mm_iter: usize,
    pub beta_rel_tol: f64,
    pub max_epsilon_outer_iter: usize,
    pub epsilon_log_step: f64,
    pub min_epsilon: f64,
    pub weight_floor: f64,
    pub weight_ceiling: f64,
}

impl Default for AdaptiveRegularizationOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            max_mm_iter: 10,
            beta_rel_tol: 1e-3,
            max_epsilon_outer_iter: 4,
            epsilon_log_step: std::f64::consts::LN_2,
            min_epsilon: 1e-8,
            weight_floor: 1e-8,
            weight_ceiling: 1e8,
        }
    }
}

/// Post-fit artifacts needed by downstream diagnostics/inference without
/// re-running PIRLS.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct FitArtifacts {
    #[serde(default, skip_serializing, skip_deserializing)]
    pub pirls: Option<crate::pirls::PirlsResult>,
    #[serde(default)]
    pub survival_link_wiggle_knots: Option<Array1<f64>>,
    #[serde(default)]
    pub survival_link_wiggle_degree: Option<usize>,
}

impl std::fmt::Debug for FitArtifacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitArtifacts")
            .field("pirls", &self.pirls.as_ref().map(|_| "..."))
            .field(
                "survival_link_wiggle_knots",
                &self
                    .survival_link_wiggle_knots
                    .as_ref()
                    .map(|knots| knots.len()),
            )
            .field(
                "survival_link_wiggle_degree",
                &self.survival_link_wiggle_degree,
            )
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitInference {
    pub edf_by_block: Vec<f64>,
    pub edf_total: f64,
    pub smoothing_correction: Option<Array2<f64>>,
    pub penalized_hessian: Array2<f64>,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Option<Array2<f64>>,
    /// Conditional posterior covariance under fixed smoothing parameters:
    /// Var(β | λ) ≈ (X'W_HX + S)^(-1), where `W_H` is the Hessian-side
    /// PIRLS curvature (Fisher for canonical links, observed/clamped for
    /// non-canonical links).
    pub beta_covariance: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected covariance.
    /// Usually this is first-order:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T.
    /// In high-risk regimes the engine may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected`.
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FittedLinkState {
    Standard(Option<LinkFunction>),
    LatentCLogLog {
        state: LatentCLogLogState,
    },
    Sas {
        state: SasLinkState,
        covariance: Option<Array2<f64>>,
    },
    BetaLogistic {
        state: SasLinkState,
        covariance: Option<Array2<f64>>,
    },
    Mixture {
        state: MixtureLinkState,
        covariance: Option<Array2<f64>>,
    },
}

fn validate_fitted_link_estimation(fitted_link: &FittedLinkState) -> Result<(), EstimationError> {
    match fitted_link {
        FittedLinkState::Standard(_) => Ok(()),
        FittedLinkState::LatentCLogLog { state } => {
            ensure_finite_scalar_estimation("fit_result.latent_cloglog.latent_sd", state.latent_sd)
        }
        FittedLinkState::Mixture { state, covariance } => {
            validate_all_finite_estimation(
                "fit_result.mixture_link_rho",
                state.rho.iter().copied(),
            )?;
            validate_all_finite_estimation(
                "fit_result.mixture_linkweights",
                state.pi.iter().copied(),
            )?;
            if let Some(v) = covariance.as_ref() {
                validate_all_finite_estimation(
                    "fit_result.mixture_link_param_covariance",
                    v.iter().copied(),
                )?;
            }
            Ok(())
        }
        FittedLinkState::Sas { state, covariance }
        | FittedLinkState::BetaLogistic { state, covariance } => {
            ensure_finite_scalar_estimation("fit_result.sas_epsilon", state.epsilon)?;
            ensure_finite_scalar_estimation("fit_result.sas_log_delta", state.log_delta)?;
            ensure_finite_scalar_estimation("fit_result.sas_delta", state.delta)?;
            if let Some(v) = covariance.as_ref() {
                validate_all_finite_estimation(
                    "fit_result.sas_param_covariance",
                    v.iter().copied(),
                )?;
            }
            Ok(())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unified fit result — single type for all model families
// ═══════════════════════════════════════════════════════════════════════════

/// Role of a coefficient block within a multi-parameter model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockRole {
    /// Single-parameter GAM (standard GLM/GAM mean model).
    Mean,
    /// Location parameter in GAMLSS / survival location-scale.
    Location,
    /// Scale (log-sigma) parameter in GAMLSS / survival location-scale.
    Scale,
    /// Time/baseline hazard block in survival models.
    Time,
    /// Threshold block in survival models.
    Threshold,
    /// Link-wiggle correction block.
    LinkWiggle,
}

/// Inference quantities for one coefficient block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedBlock {
    /// Coefficients at the converged mode.
    pub beta: Array1<f64>,
    /// Role of this block within the model.
    pub role: BlockRole,
    /// Effective degrees of freedom (sum of leverages).
    pub edf: f64,
    /// Smoothing parameters for this block.
    pub lambdas: Array1<f64>,
}

/// Working-set geometry at convergence needed by ALO and other post-fit
/// diagnostics. Only populated when the inner solver provides the data.
#[derive(Clone, Debug)]
pub struct FitGeometry {
    /// Joint penalized Hessian H = X'W_HX + S(λ) at convergence.
    pub penalized_hessian: Array2<f64>,
    /// Score-side Fisher IRLS weights paired with `working_response`.
    pub working_weights: Array1<f64>,
    /// IRLS working response at convergence.
    pub working_response: Array1<f64>,
}

pub struct UnifiedFitResultParts {
    pub blocks: Vec<FittedBlock>,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: Option<LikelihoodFamily>,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub reml_score: f64,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    pub outer_converged: bool,
    pub outer_gradient_norm: f64,
    pub standard_deviation: f64,
    pub covariance_conditional: Option<Array2<f64>>,
    pub covariance_corrected: Option<Array2<f64>>,
    pub inference: Option<FitInference>,
    pub fitted_link: FittedLinkState,
    pub geometry: Option<FitGeometry>,
    pub block_states: Vec<crate::families::custom_family::ParameterBlockState>,
    // Backward-compatible fields (all have sensible defaults).
    #[doc(hidden)]
    pub pirls_status: crate::pirls::PirlsStatus,
    #[doc(hidden)]
    pub max_abs_eta: f64,
    #[doc(hidden)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    #[doc(hidden)]
    pub artifacts: FitArtifacts,
    #[doc(hidden)]
    pub inner_cycles: usize,
}

/// Unified fit result for all model types (standard GAM, GAMLSS, survival).
///
/// Standard models have a single block; GAMLSS and survival models have
/// multiple blocks with different roles.
///
/// Backward-compatible field aliases are provided so that code written against
/// the old `FitResult`, `BlockwiseFitResult`, and `SurvivalLocationScaleFitResult`
/// types continues to compile after the unification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedFitResult {
    // ── canonical fields ──────────────────────────────────────────────────
    /// Coefficient blocks (1 for standard GAM, N for GAMLSS/survival).
    pub blocks: Vec<FittedBlock>,
    /// Log-smoothing parameters (all blocks concatenated in block order).
    pub log_lambdas: Array1<f64>,
    /// Smoothing parameters (exp of log_lambdas).
    pub lambdas: Array1<f64>,
    /// Explicit engine-level family, when the fit uses a built-in family.
    pub likelihood_family: Option<LikelihoodFamily>,
    /// Fixed-scale metadata for the fitted likelihood.
    pub likelihood_scale: LikelihoodScaleMetadata,
    /// Whether `log_likelihood` includes response-only normalization constants.
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    /// Log-likelihood at the converged mode.
    pub log_likelihood: f64,
    /// Explicit deviance reported by the fitting path.
    pub deviance: f64,
    /// REML/LAML score (penalized objective used for smoothing selection).
    pub reml_score: f64,
    /// Stable penalty term (sum of lambda * beta' S beta terms).
    pub stable_penalty_term: f64,
    /// Penalized objective value (−ℓ + penalty + REML terms).
    pub penalized_objective: f64,
    /// Number of outer (smoothing parameter) iterations.
    pub outer_iterations: usize,
    /// Whether the outer optimization converged.
    pub outer_converged: bool,
    /// Final gradient norm of the outer optimization.
    pub outer_gradient_norm: f64,
    /// Residual scale on the response scale.
    ///
    /// Contract: Gaussian identity models store residual standard deviation
    /// sigma here. Non-Gaussian families keep the response-scale summary used
    /// by their explicit likelihood-scale metadata.
    pub standard_deviation: f64,
    /// Conditional covariance Var(β | λ) for the joint coefficient vector.
    pub covariance_conditional: Option<Array2<f64>>,
    /// Smoothing-parameter-corrected covariance Var*(β).
    pub covariance_corrected: Option<Array2<f64>>,
    /// Inference quantities from the inner solver (EDF, Hessian, etc.).
    pub inference: Option<FitInference>,
    /// Fitted link parameters (SAS, BetaLogistic, Mixture).
    pub fitted_link: FittedLinkState,
    /// Working-set geometry at convergence (for ALO diagnostics).
    #[serde(skip)]
    pub geometry: Option<FitGeometry>,
    /// Internal block states from custom-family paths.
    #[serde(skip)]
    pub block_states: Vec<crate::families::custom_family::ParameterBlockState>,
    /// Joint coefficient vector (first block for standard GAMs, concatenated for multi-block).
    #[serde(default)]
    pub beta: Array1<f64>,
    /// Inner solver convergence status.
    #[serde(default = "default_pirls_status")]
    pub pirls_status: crate::pirls::PirlsStatus,
    /// Maximum absolute linear predictor value at convergence.
    #[serde(default)]
    pub max_abs_eta: f64,
    /// Constraint KKT diagnostics (monotone-constrained fits).
    #[serde(default)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    /// Solver artifacts (e.g. cached PIRLS result for ALO).
    #[serde(skip)]
    pub artifacts: FitArtifacts,
    /// Inner cycle count (blockwise path).
    #[serde(default)]
    pub inner_cycles: usize,
}

fn default_pirls_status() -> crate::pirls::PirlsStatus {
    crate::pirls::PirlsStatus::Converged
}

impl Default for FittedLinkState {
    fn default() -> Self {
        FittedLinkState::Standard(None)
    }
}

pub(crate) fn ensure_finite_scalar_estimation(
    name: &str,
    value: f64,
) -> Result<(), EstimationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

fn validate_likelihood_scale_estimation(
    scale: LikelihoodScaleMetadata,
) -> Result<(), EstimationError> {
    match scale {
        LikelihoodScaleMetadata::ProfiledGaussian | LikelihoodScaleMetadata::Unspecified => Ok(()),
        LikelihoodScaleMetadata::FixedDispersion { phi } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.phi", phi)?;
            if phi > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.phi must be > 0, got {phi}"
                )))
            }
        }
        LikelihoodScaleMetadata::FixedGammaShape { shape }
        | LikelihoodScaleMetadata::EstimatedGammaShape { shape } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.shape", shape)?;
            if shape > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.shape must be > 0, got {shape}"
                )))
            }
        }
    }
}

pub(crate) fn validate_all_finite_estimation<I>(
    label: &str,
    values: I,
) -> Result<(), EstimationError>
where
    I: IntoIterator<Item = f64>,
{
    for (idx, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "{label}[{idx}] must be finite, got {value}"
            )));
        }
    }
    Ok(())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn ensure_finite_scalar(name: &str, value: f64) -> Result<(), String> {
    ensure_finite_scalar_estimation(name, value).map_err(|e| e.to_string())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn validate_all_finite<I: IntoIterator<Item = f64>>(
    label: &str,
    values: I,
) -> Result<(), String> {
    validate_all_finite_estimation(label, values).map_err(|e| e.to_string())
}

impl FitGeometry {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        validate_all_finite_estimation(
            "fit_result.geometry.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working_weights",
            self.working_weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working_response",
            self.working_response.iter().copied(),
        )?;
        Ok(())
    }
}

impl FitInference {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        ensure_finite_scalar_estimation("fit_result.edf_total", self.edf_total)?;
        validate_all_finite_estimation(
            "fit_result.edf_by_block",
            self.edf_by_block.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.working_weights",
            self.working_weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.working_response",
            self.working_response.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        if let Some(v) = self.beta_covariance.as_ref() {
            validate_all_finite_estimation("fit_result.beta_covariance", v.iter().copied())?;
        }
        if let Some(v) = self.beta_covariance_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.beta_standard_errors.as_ref() {
            validate_all_finite_estimation("fit_result.beta_standard_errors", v.iter().copied())?;
        }
        if let Some(v) = self.beta_standard_errors_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_standard_errors_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.smoothing_correction.as_ref() {
            validate_all_finite_estimation("fit_result.smoothing_correction", v.iter().copied())?;
        }
        if let Some(v) = self.reparam_qs.as_ref() {
            validate_all_finite_estimation("fit_result.reparam_qs", v.iter().copied())?;
        }
        Ok(())
    }
}

impl UnifiedFitResult {
    pub fn try_from_parts(parts: UnifiedFitResultParts) -> Result<Self, EstimationError> {
        let UnifiedFitResultParts {
            blocks,
            log_lambdas,
            lambdas,
            likelihood_family,
            likelihood_scale,
            log_likelihood_normalization,
            log_likelihood,
            deviance,
            reml_score,
            stable_penalty_term,
            penalized_objective,
            outer_iterations,
            outer_converged,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            pirls_status,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        } = parts;

        if blocks.is_empty() {
            return Err(EstimationError::InvalidInput(
                "UnifiedFitResult requires at least one coefficient block".to_string(),
            ));
        }
        if log_lambdas.len() != lambdas.len() {
            return Err(EstimationError::InvalidInput(format!(
                "UnifiedFitResult lambda mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            )));
        }
        for (idx, block) in blocks.iter().enumerate() {
            validate_all_finite_estimation(
                &format!("fit_result.blocks[{idx}].beta"),
                block.beta.iter().copied(),
            )?;
            ensure_finite_scalar_estimation(&format!("fit_result.blocks[{idx}].edf"), block.edf)?;
            validate_all_finite_estimation(
                &format!("fit_result.blocks[{idx}].lambdas"),
                block.lambdas.iter().copied(),
            )?;
        }
        validate_all_finite_estimation("fit_result.log_lambdas", log_lambdas.iter().copied())?;
        validate_all_finite_estimation("fit_result.lambdas", lambdas.iter().copied())?;
        validate_likelihood_scale_estimation(likelihood_scale)?;
        ensure_finite_scalar_estimation("fit_result.log_likelihood", log_likelihood)?;
        ensure_finite_scalar_estimation("fit_result.deviance", deviance)?;
        ensure_finite_scalar_estimation("fit_result.reml_score", reml_score)?;
        ensure_finite_scalar_estimation("fit_result.stable_penalty_term", stable_penalty_term)?;
        ensure_finite_scalar_estimation("fit_result.penalized_objective", penalized_objective)?;
        ensure_finite_scalar_estimation("fit_result.outer_gradient_norm", outer_gradient_norm)?;
        ensure_finite_scalar_estimation("fit_result.standard_deviation", standard_deviation)?;
        if let Some(v) = covariance_conditional.as_ref() {
            validate_all_finite_estimation("fit_result.beta_covariance", v.iter().copied())?;
        }
        if let Some(v) = covariance_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(inf) = inference.as_ref() {
            inf.validate_numeric_finiteness()?;
        }
        if let Some(geom) = geometry.as_ref() {
            geom.validate_numeric_finiteness()?;
        }
        for (idx, state) in block_states.iter().enumerate() {
            validate_all_finite_estimation(
                &format!("fit_result.block_states[{idx}].beta"),
                state.beta.iter().copied(),
            )?;
            validate_all_finite_estimation(
                &format!("fit_result.block_states[{idx}].eta"),
                state.eta.iter().copied(),
            )?;
        }
        validate_fitted_link_estimation(&fitted_link)?;

        // Build the flat beta vector from all blocks.
        let beta = {
            let total: usize = blocks.iter().map(|b| b.beta.len()).sum();
            let mut flat = Array1::zeros(total);
            let mut off = 0;
            for b in &blocks {
                let p = b.beta.len();
                flat.slice_mut(ndarray::s![off..off + p]).assign(&b.beta);
                off += p;
            }
            flat
        };
        let p = beta.len();
        if let Some(cov) = covariance_conditional.as_ref() {
            if cov.nrows() != p || cov.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                )));
            }
        }
        if let Some(cov) = covariance_corrected.as_ref() {
            if cov.nrows() != p || cov.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                )));
            }
        }
        if let Some(inf) = inference.as_ref() {
            if inf.penalized_hessian.nrows() != p || inf.penalized_hessian.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    inf.penalized_hessian.nrows(),
                    inf.penalized_hessian.ncols(),
                    p,
                    p
                )));
            }
        }
        if !block_states.is_empty() && block_states.len() != blocks.len() {
            return Err(EstimationError::InvalidInput(format!(
                "UnifiedFitResult block state count mismatch: blocks={}, block_states={}",
                blocks.len(),
                block_states.len()
            )));
        }

        Ok(Self {
            blocks,
            log_lambdas,
            lambdas,
            likelihood_family,
            likelihood_scale,
            log_likelihood_normalization,
            log_likelihood,
            deviance,
            reml_score,
            stable_penalty_term,
            penalized_objective,
            outer_iterations,
            outer_converged,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            beta,
            pirls_status,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        })
    }

    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        Self::try_from_parts(UnifiedFitResultParts {
            blocks: self.blocks.clone(),
            log_lambdas: self.log_lambdas.clone(),
            lambdas: self.lambdas.clone(),
            likelihood_family: self.likelihood_family,
            likelihood_scale: self.likelihood_scale,
            log_likelihood_normalization: self.log_likelihood_normalization,
            log_likelihood: self.log_likelihood,
            deviance: self.deviance,
            reml_score: self.reml_score,
            stable_penalty_term: self.stable_penalty_term,
            penalized_objective: self.penalized_objective,
            outer_iterations: self.outer_iterations,
            outer_converged: self.outer_converged,
            outer_gradient_norm: self.outer_gradient_norm,
            standard_deviation: self.standard_deviation,
            covariance_conditional: self.covariance_conditional.clone(),
            covariance_corrected: self.covariance_corrected.clone(),
            inference: self.inference.clone(),
            fitted_link: self.fitted_link.clone(),
            geometry: self.geometry.clone(),
            block_states: self.block_states.clone(),
            pirls_status: self.pirls_status,
            max_abs_eta: self.max_abs_eta,
            constraint_kkt: self.constraint_kkt.clone(),
            artifacts: self.artifacts.clone(),
            inner_cycles: self.inner_cycles,
        })
        .map(|_| ())
    }
}

impl UnifiedFitResult {
    /// Get the beta covariance matrix (conditional) if available.
    pub fn beta_covariance(&self) -> Option<&Array2<f64>> {
        self.covariance_conditional.as_ref()
    }

    /// Get the smoothing-parameter-corrected beta covariance if available.
    pub fn beta_covariance_corrected(&self) -> Option<&Array2<f64>> {
        self.covariance_corrected.as_ref().or_else(|| {
            self.inference
                .as_ref()
                .and_then(|inf| inf.beta_covariance_corrected.as_ref())
        })
    }

    /// Get beta standard errors (conditional) if available.
    pub fn beta_standard_errors(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_standard_errors.as_ref())
    }

    /// Get smoothing-corrected beta standard errors if available.
    pub fn beta_standard_errors_corrected(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_standard_errors_corrected.as_ref())
    }

    /// Get the penalized Hessian if available.
    pub fn penalized_hessian(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .map(|inf| &inf.penalized_hessian)
            .or_else(|| self.geometry.as_ref().map(|geom| &geom.penalized_hessian))
    }

    /// Get working weights if available.
    pub fn working_weights(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_weights)
    }

    /// Get working response if available.
    pub fn working_response(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_response)
    }

    /// Total effective degrees of freedom.
    pub fn edf_total(&self) -> Option<f64> {
        self.inference.as_ref().map(|inf| inf.edf_total)
    }

    /// EDF by block.
    pub fn edf_by_block(&self) -> &[f64] {
        self.inference
            .as_ref()
            .map(|inf| inf.edf_by_block.as_slice())
            .unwrap_or(&[])
    }

    /// Find a block by role.
    pub fn block_by_role(&self, role: BlockRole) -> Option<&FittedBlock> {
        self.blocks.iter().find(|b| b.role == role)
    }

    /// Flat coefficient vector (all blocks concatenated).
    /// This is equivalent to `self.beta.clone()`.
    pub fn beta_flat(&self) -> Array1<f64> {
        self.beta.clone()
    }

    /// Time/baseline-hazard coefficients (survival location-scale).
    pub fn beta_time(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Time)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Threshold coefficients (survival location-scale).
    pub fn beta_threshold(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Threshold)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Log-sigma coefficients (survival location-scale).
    pub fn beta_log_sigma(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Scale)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Link-wiggle coefficients (survival location-scale, optional).
    pub fn beta_link_wiggle(&self) -> Option<Array1<f64>> {
        self.block_by_role(BlockRole::LinkWiggle)
            .map(|b| b.beta.clone())
    }

    /// Smoothing parameters for time block.
    pub fn lambdas_time(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Time)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for threshold block.
    pub fn lambdas_threshold(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Threshold)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for log-sigma block.
    pub fn lambdas_log_sigma(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Scale)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for link-wiggle block.
    pub fn lambdas_linkwiggle(&self) -> Option<Array1<f64>> {
        self.block_by_role(BlockRole::LinkWiggle)
            .map(|b| b.lambdas.clone())
    }

    /// Number of coefficient blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block roles.
    pub fn block_roles(&self) -> Vec<BlockRole> {
        self.blocks.iter().map(|b| b.role.clone()).collect()
    }

    /// Resolve the fitted link state for a given family.
    ///
    /// For standard (non-adaptive) link families, this enriches the stored
    /// `FittedLinkState::Standard` with the concrete `LinkFunction` derived
    /// from the family.  For adaptive links (SAS, BetaLogistic, Mixture) it
    /// validates that the stored state matches the family and clones it out.
    pub fn fitted_link_state(
        &self,
        family: crate::types::LikelihoodFamily,
    ) -> Result<FittedLinkState, EstimationError> {
        match family {
            crate::types::LikelihoodFamily::GaussianIdentity => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Identity)))
            }
            crate::types::LikelihoodFamily::BinomialLogit => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Logit)))
            }
            crate::types::LikelihoodFamily::BinomialProbit => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Probit)))
            }
            crate::types::LikelihoodFamily::BinomialCLogLog => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::CLogLog)))
            }
            crate::types::LikelihoodFamily::BinomialLatentCLogLog => match &self.fitted_link {
                FittedLinkState::LatentCLogLog { state } => {
                    Ok(FittedLinkState::LatentCLogLog { state: *state })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialLatentCLogLog requires fixed latent cloglog state".to_string(),
                )),
            },
            crate::types::LikelihoodFamily::BinomialSas => match &self.fitted_link {
                FittedLinkState::Sas { state, covariance } => Ok(FittedLinkState::Sas {
                    state: state.clone(),
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialSas requires fitted SAS link parameters".to_string(),
                )),
            },
            crate::types::LikelihoodFamily::BinomialBetaLogistic => match &self.fitted_link {
                FittedLinkState::BetaLogistic { state, covariance } => {
                    Ok(FittedLinkState::BetaLogistic {
                        state: state.clone(),
                        covariance: covariance.clone(),
                    })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialBetaLogistic requires fitted beta-logistic link parameters"
                        .to_string(),
                )),
            },
            crate::types::LikelihoodFamily::BinomialMixture => match &self.fitted_link {
                FittedLinkState::Mixture { state, covariance } => Ok(FittedLinkState::Mixture {
                    state: state.clone(),
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialMixture requires fitted mixture link parameters".to_string(),
                )),
            },
            crate::types::LikelihoodFamily::PoissonLog
            | crate::types::LikelihoodFamily::GammaLog => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Log)))
            }
            crate::types::LikelihoodFamily::RoystonParmar => Ok(FittedLinkState::Standard(None)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParametricTermSummary {
    pub name: String,
    pub estimate: f64,
    pub std_error: Option<f64>,
    pub zvalue: Option<f64>,
    pub pvalue: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SmoothTermSummary {
    pub name: String,
    pub edf: f64,
    pub ref_df: f64,
    pub chi_sq: Option<f64>,
    pub pvalue: Option<f64>,
    pub continuous_order: Option<ContinuousSmoothnessOrder>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContinuousSmoothnessOrderStatus {
    Ok,
    NonMaternRegime,
    FirstOrderLimit,
    IntrinsicLimit,
    UndefinedZeroLambda,
}

#[derive(Clone, Debug)]
pub struct ContinuousSmoothnessOrder {
    pub lambda0: f64,
    pub lambda1: f64,
    pub lambda2: f64,
    pub r_ratio: Option<f64>,
    pub nu: Option<f64>,
    pub kappa2: Option<f64>,
    pub status: ContinuousSmoothnessOrderStatus,
}

#[derive(Clone, Debug)]
pub struct ModelSummary {
    pub family: String,
    pub deviance_explained: Option<f64>,
    pub reml_score: Option<f64>,
    pub parametric_terms: Vec<ParametricTermSummary>,
    pub smooth_terms: Vec<SmoothTermSummary>,
}

/// Convert optimizer-scale lambdas into physical lambdas for raw operator penalties.
///
/// Derivation:
///   We optimize with normalized penalties
///     sum_k lambda_tilde_k * S_tilde_k
///   where
///     S_tilde_k = (1 / c_k) * S_k.
///
///   Define physical lambdas by requiring operator equality:
///     sum_k lambda_k * S_k  ==  sum_k lambda_tilde_k * S_tilde_k
///                           ==  sum_k lambda_tilde_k * (1/c_k) * S_k
///                           ==  sum_k (lambda_tilde_k / c_k) * S_k.
///
///   Therefore, coefficient matching gives:
///     lambda_k = lambda_tilde_k / c_k.
///
/// This helper performs exactly that mapping and validates positivity/finite values.
fn unscale_to_physical_lambdas(
    lambda_tilde: [f64; 3],
    normalization_scale: [f64; 3],
) -> Option<[f64; 3]> {
    let mut out = [f64::NAN; 3];
    for k in 0..3 {
        let c = normalization_scale[k];
        if !(c.is_finite() && c > 0.0) {
            return None;
        }
        out[k] = lambda_tilde[k] / c;
    }
    Some(out)
}

// Continuous smoothness/order diagnostic from three operator penalties.
//
// Full derivation and implementation contract
// We assume one smooth term has exactly three operator penalties in term-local order:
//   S0 = mass, S1 = tension (|grad f|^2), S2 = stiffness ((Delta f)^2).
//
// 1) Unscaling (physical lambda from optimizer lambda)
// If penalties were normalized before optimization:
//   S_tilde_k = S_k / c_k
// and the optimizer fits lambda_tilde_k, then
//   lambda_tilde_k * beta' S_tilde_k beta
// = lambda_tilde_k * beta' (S_k / c_k) beta
// = (lambda_tilde_k / c_k) * beta' S_k beta.
//
// Therefore physical lambdas are:
//   lambda_k = lambda_tilde_k / c_k,  k in {0,1,2}.
//
// 2) SPDE/binomial coefficient mapping
// If the fitted (lambda0,lambda1,lambda2) are interpreted as proportional to
//   a_m(kappa,nu) = C(nu,m) * kappa^(2*(nu-m)),  m=0,1,2,
// then
//   a0 = kappa^(2*nu)
//   a1 = nu * kappa^(2*nu-2)
//   a2 = nu*(nu-1)/2 * kappa^(2*nu-4)
//
// Ratios:
//   lambda0/lambda2 = a0/a2 = 2*kappa^4 / (nu*(nu-1))
//   lambda1/lambda2 = a1/a2 = 2*kappa^2 / (nu-1)
//
// Define:
//   R = lambda1^2 / (lambda0*lambda2).
// Then
//   R = a1^2/(a0*a2) = 2*nu/(nu-1).
// Solve for nu:
//   nu = R/(R-2), requiring R>2 for finite nu>1.
//
// And from lambda1/lambda2:
//   kappa^2 = ((nu-1)/2) * (lambda1/lambda2)
//           = lambda1 / ((R-2)*lambda2).
//
// 3) Boundary/discriminant interpretation
// Spectral polynomial in x=|omega|^2:
//   Q(x) = lambda0 + lambda1*x + lambda2*x^2.
//
// Perfect-square Matérn(2) form is:
//   Q(x) proportional to (kappa^2 + x)^2
// which implies:
//   lambda1^2 = 4*lambda0*lambda2  <=>  R = 4.
//
// Discriminant:
//   D = lambda1^2 - 4*lambda0*lambda2 = lambda0*lambda2*(R-4).
// Hence:
//   R < 4  => D < 0 => no real factorization into two real range terms
//            => flagged as NonMaternRegime.
//   R = 4  => exact boundary (perfect square) => treated as Matérn-compatible.
//
// 4) Degenerate limits and guards
// - If lambda0 or lambda2 is non-finite or <= eps, the 3-term inversion is unstable;
//   report UndefinedZeroLambda and do not divide by those terms.
// - Intrinsic limit (lambda0 -> 0+, with finite lambda1/lambda2):
//     R = lambda1^2/(lambda0*lambda2) -> +inf
//     nu = R/(R-2) -> 1+
//     kappa^2 = lambda1/((R-2)lambda2) -> 0+.
//   We expose this explicitly as IntrinsicLimit with nu≈1 and kappa^2≈0.
// - If R <= 2 (+eps), nu = R/(R-2) is undefined or numerically unstable; keep
//   nu/kappa2 unset.
//
// Status policy in this implementation:
// - Ok:                R >= 4 and valid finite nu/kappa2.
// - NonMaternRegime:   R < 4; if additionally R > 2, we still report effective
//                      nu/kappa2 as diagnostics, but mark non-Matérn status.
// - IntrinsicLimit:    lambda0 is negligible; report nu≈1, kappa^2≈0.
// - UndefinedZeroLambda: invalid scaling/lambda inputs or unstable inversion.
pub fn compute_continuous_smoothness_order(
    lambda_tilde: [f64; 3],
    normalization_scale: [f64; 3],
    eps: f64,
) -> ContinuousSmoothnessOrder {
    let Some(lambda) = unscale_to_physical_lambdas(lambda_tilde, normalization_scale) else {
        return ContinuousSmoothnessOrder {
            lambda0: f64::NAN,
            lambda1: f64::NAN,
            lambda2: f64::NAN,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    };
    let [lambda0, lambda1, lambda2] = lambda;
    if !lambda0.is_finite() || !lambda1.is_finite() || !lambda2.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }
    // Scale-aware degeneracy floor.
    // Using only an absolute epsilon can misclassify limits when lambdas are
    // globally tiny or globally huge, so we threshold relative to the largest
    // physical lambda magnitude in this term.
    let lambda_scale = lambda0.abs().max(lambda1.abs()).max(lambda2.abs()).max(1.0);
    let lambda_floor = eps * lambda_scale;

    // Intrinsic limit: mass term vanishes (kappa^2 -> 0).
    if lambda0 <= lambda_floor {
        if lambda1 > lambda_floor && lambda2 > lambda_floor {
            return ContinuousSmoothnessOrder {
                lambda0,
                lambda1,
                lambda2,
                r_ratio: None,
                nu: Some(1.0),
                kappa2: Some(0.0),
                status: ContinuousSmoothnessOrderStatus::IntrinsicLimit,
            };
        }
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }
    // First-order fallback when stiffness collapses:
    //   lambda2 ~ 0 => use lambda0/lambda1 = kappa^2 with nu ≈ 1.
    if lambda2 <= lambda_floor {
        if lambda1 > lambda_floor && lambda1.is_finite() {
            return ContinuousSmoothnessOrder {
                lambda0,
                lambda1,
                lambda2,
                r_ratio: None,
                nu: Some(1.0),
                kappa2: Some(lambda0 / lambda1),
                status: ContinuousSmoothnessOrderStatus::FirstOrderLimit,
            };
        }
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    let r_ratio = (lambda1 * lambda1) / (lambda0 * lambda2);
    if !r_ratio.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    // From a_m = binom(nu,m) * kappa^{2(nu-m)} with m=0,1,2:
    //   R = lambda1^2 / (lambda0*lambda2) = 2*nu/(nu-1)
    //   nu = R/(R-2), and kappa^2 = lambda1 / ((R-2)*lambda2).
    //
    // Discriminant of spectral quadratic P(t)=lambda0+lambda1*t+lambda2*t^2:
    //   Delta_P = lambda1^2 - 4*lambda0*lambda2 = lambda0*lambda2*(R-4).
    // Non-Matérn regime is flagged by Delta_P < 0 (equiv. R < 4),
    // but nu/kappa2 are still reported when R > 2 as effective diagnostics.
    let discriminant = lambda1 * lambda1 - 4.0 * lambda0 * lambda2;
    let disc_tol = eps * lambda_scale * lambda_scale;
    let status = if discriminant < -disc_tol {
        ContinuousSmoothnessOrderStatus::NonMaternRegime
    } else {
        // Includes exact boundary R=4 (perfect-square case) and numerically
        // indistinguishable near-boundary points.
        ContinuousSmoothnessOrderStatus::Ok
    };
    if r_ratio <= 2.0 + eps {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: Some(r_ratio),
            nu: None,
            kappa2: None,
            status,
        };
    }
    let nu = r_ratio / (r_ratio - 2.0);
    // Closed-form extraction required by the continuous-order benchmark:
    //
    //   R = lambda1^2 / (lambda0*lambda2) = 2*nu/(nu-1)
    //   => nu = R/(R-2).
    //
    //   lambda1/lambda2 = 2*kappa^2/(nu-1)
    //   => kappa^2 = ((nu-1)/2)*(lambda1/lambda2)
    //             = lambda1 / ((R-2)*lambda2).
    //
    // We use this exact closed form as the reported kappa^2.
    let kappa2 = lambda1 / ((r_ratio - 2.0) * lambda2);
    if !nu.is_finite() || !kappa2.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: Some(r_ratio),
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    ContinuousSmoothnessOrder {
        lambda0,
        lambda1,
        lambda2,
        r_ratio: Some(r_ratio),
        nu: Some(nu),
        kappa2: Some(kappa2),
        status,
    }
}

#[cfg(test)]
pub(crate) fn try_compute_continuous_smoothness_order(
    lambda_tilde: &[f64],
    normalization_scale: &[f64],
    eps: f64,
) -> Option<ContinuousSmoothnessOrder> {
    if lambda_tilde.len() != 3 || normalization_scale.len() != 3 {
        return None;
    }
    Some(compute_continuous_smoothness_order(
        [lambda_tilde[0], lambda_tilde[1], lambda_tilde[2]],
        [
            normalization_scale[0],
            normalization_scale[1],
            normalization_scale[2],
        ],
        eps,
    ))
}

fn significance_stars(p: Option<f64>) -> &'static str {
    match p {
        Some(v) if v.is_finite() && v < 0.001 => "***",
        Some(v) if v.is_finite() && v < 0.01 => "**",
        Some(v) if v.is_finite() && v < 0.05 => "*",
        Some(v) if v.is_finite() && v < 0.1 => ".",
        _ => "",
    }
}

fn format_pvalue(p: Option<f64>) -> String {
    let Some(v) = p else {
        return "NA".to_string();
    };
    if !v.is_finite() {
        return "NA".to_string();
    }
    if v < 2e-16 {
        "< 2e-16".to_string()
    } else if v < 1e-4 {
        format!("{v:.2e}")
    } else {
        format!("{v:.4}")
    }
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let paramnamew = self
            .parametric_terms
            .iter()
            .map(|t| t.name.len())
            .max()
            .unwrap_or(10)
            .max("Term".len());
        let smoothnamew = self
            .smooth_terms
            .iter()
            .map(|t| t.name.len())
            .max()
            .unwrap_or(10)
            .max("Term".len());

        writeln!(f, "Family: {}", self.family)?;
        let dev_txt = self
            .deviance_explained
            .map(|d| format!("{:.1}%", (100.0 * d).clamp(-9999.0, 9999.0)))
            .unwrap_or_else(|| "NA".to_string());
        let reml_txt = self
            .reml_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        writeln!(f, "Deviance Explained: {dev_txt} | REML Score: {reml_txt}")?;
        writeln!(f)?;

        writeln!(f, "Parametric Terms:")?;
        writeln!(f, "{:-<1$}", "", paramnamew + 59)?;
        writeln!(
            f,
            "{:<namew$} {:>10} {:>12} {:>10} {:>19}",
            "Term",
            "Estimate",
            "Standard Error",
            "Z Statistic",
            "Two-Sided P-Value",
            namew = paramnamew
        )?;
        writeln!(f, "{:-<1$}", "", paramnamew + 59)?;
        for term in &self.parametric_terms {
            let estimate = format!("{:.4}", term.estimate);
            let se = term
                .std_error
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "NA".to_string());
            let z = term
                .zvalue
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "NA".to_string());
            let p = format_pvalue(term.pvalue);
            let stars = significance_stars(term.pvalue);
            writeln!(
                f,
                "{:<namew$} {:>10} {:>12} {:>10} {:>19} {}",
                term.name,
                estimate,
                se,
                z,
                p,
                stars,
                namew = paramnamew
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Smooth Terms:")?;
        writeln!(f, "{:-<1$}", "", smoothnamew + 86)?;
        writeln!(
            f,
            "{:<namew$} {:>26} {:>30} {:>12} {:>10}",
            "Term",
            "Effective Degrees of Freedom",
            "Reference Degrees of Freedom",
            "Chi-Square",
            "P-Value",
            namew = smoothnamew
        )?;
        writeln!(f, "{:-<1$}", "", smoothnamew + 86)?;
        for term in &self.smooth_terms {
            let chisq = term
                .chi_sq
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "NA".to_string());
            let p = format_pvalue(term.pvalue);
            let stars = significance_stars(term.pvalue);
            writeln!(
                f,
                "{:<namew$} {:>26.2} {:>30.2} {:>12} {:>10} {}",
                term.name,
                term.edf,
                term.ref_df,
                chisq,
                p,
                stars,
                namew = smoothnamew
            )?;
        }
        writeln!(f)?;
        let order_terms = self
            .smooth_terms
            .iter()
            .filter_map(|t| t.continuous_order.as_ref().map(|o| (&t.name, o)))
            .collect::<Vec<_>>();
        if !order_terms.is_empty() {
            writeln!(f, "Continuous Smoothness Order:")?;
            writeln!(
                f,
                "{:<namew$} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>20}",
                "Term",
                "lambda0",
                "lambda1",
                "lambda2",
                "R",
                "nu",
                "kappa^2",
                "status",
                namew = smoothnamew
            )?;
            for (name, o) in order_terms {
                let r_txt = o
                    .r_ratio
                    .filter(|v| v.is_finite())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "NA".to_string());
                let nu_txt =
                    o.nu.filter(|v| v.is_finite())
                        .map(|v| format!("{v:.4}"))
                        .unwrap_or_else(|| "NA".to_string());
                let kappa_txt = o
                    .kappa2
                    .filter(|v| v.is_finite())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "NA".to_string());
                let status_txt = match o.status {
                    ContinuousSmoothnessOrderStatus::Ok => "Ok",
                    ContinuousSmoothnessOrderStatus::NonMaternRegime => "NonMaternRegime",
                    ContinuousSmoothnessOrderStatus::FirstOrderLimit => "FirstOrderLimit",
                    ContinuousSmoothnessOrderStatus::IntrinsicLimit => "IntrinsicLimit",
                    ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => "UndefinedZeroLambda",
                };
                writeln!(
                    f,
                    "{:<namew$} {:>10.3e} {:>10.3e} {:>10.3e} {:>10} {:>10} {:>10} {:>20}",
                    name,
                    o.lambda0,
                    o.lambda1,
                    o.lambda2,
                    r_txt,
                    nu_txt,
                    kappa_txt,
                    status_txt,
                    namew = smoothnamew
                )?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )?;
        Ok(())
    }
}

pub use crate::inference::predict::{
    CoefficientUncertaintyResult, InferenceCovarianceMode, MeanIntervalMethod, PredictInput,
    PredictPosteriorMeanResult, PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult,
    PredictableModel, coefficient_uncertainty, coefficient_uncertaintywith_mode,
    enrich_posterior_mean_bounds, predict_gam, predict_gam_posterior_mean,
    predict_gam_posterior_meanwith_fit, predict_gamwith_uncertainty,
};

/// Canonical engine entrypoint for external designs on supported GLM-style
/// families.
/// Likelihood dispatch is determined by `family` together with external-link
/// options in `opts`; survival families use survival-specific training APIs.
pub fn fit_gamwith_heuristic_lambdas<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    heuristic_lambdas: Option<&[f64]>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_heuristic_lambdas_andwarm_start(
        x,
        y,
        weights,
        offset,
        s_list,
        heuristic_lambdas,
        None,
        family,
        opts,
    )
}

pub(crate) fn fit_gamwith_heuristic_lambdas_andwarm_start<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if matches!(family, crate::types::LikelihoodFamily::BinomialMixture)
        && opts.mixture_link.is_none()
    {
        return Err(EstimationError::InvalidInput(
            "BinomialMixture requires mixture_link specification".to_string(),
        ));
    }
    let effective_sas_link = effective_sas_link_for_family(family, opts.sas_link);
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        return Err(EstimationError::InvalidInput(
            "mixture_link and sas_link cannot both be set".to_string(),
        ));
    }
    let resolved_family = if opts.mixture_link.is_some() {
        match family {
            crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialMixture => {
                crate::types::LikelihoodFamily::BinomialMixture
            }
            _ => {
                return Err(EstimationError::InvalidInput(
                    "mixture_link is only supported for binomial families".to_string(),
                ));
            }
        }
    } else if effective_sas_link.is_some() {
        match family {
            crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic => {
                if matches!(family, crate::types::LikelihoodFamily::BinomialBetaLogistic) {
                    crate::types::LikelihoodFamily::BinomialBetaLogistic
                } else {
                    crate::types::LikelihoodFamily::BinomialSas
                }
            }
            _ => {
                return Err(EstimationError::InvalidInput(
                    "sas_link is only supported for binomial families".to_string(),
                ));
            }
        }
    } else {
        family
    };
    if matches!(
        resolved_family,
        crate::types::LikelihoodFamily::RoystonParmar
    ) {
        return Err(EstimationError::InvalidInput(
            "fit_gam external design path does not support RoystonParmar; use survival training APIs".to_string(),
        ));
    }
    validate_penalty_specs(&specs, x.ncols(), "fit_gam")?;
    let mut ext_opts = ExternalOptimOptions {
        family: resolved_family,
        latent_cloglog: opts.latent_cloglog,
        mixture_link: opts.mixture_link.clone(),
        optimize_mixture: opts.optimize_mixture,
        sas_link: effective_sas_link,
        optimize_sas: opts.optimize_sas,
        compute_inference: opts.compute_inference,
        max_iter: opts.max_iter,
        tol: opts.tol,
        nullspace_dims: opts.nullspace_dims.clone(),
        linear_constraints: opts.linear_constraints.clone(),
        firth_bias_reduction: None,
        penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
        rho_prior: Default::default(),
        kronecker_penalty_system: opts.kronecker_penalty_system.clone(),
        kronecker_factored: opts.kronecker_factored.clone(),
    };

    let result = if matches!(
        resolved_family,
        crate::types::LikelihoodFamily::BinomialLogit
    ) {
        let weighted_events = y
            .iter()
            .zip(weights.iter())
            .map(|(&yy, &ww)| yy.clamp(0.0, 1.0) * ww.max(0.0))
            .sum::<f64>();
        let weighted_total = weights.iter().map(|w| w.max(0.0)).sum::<f64>();
        let weightednonevents = (weighted_total - weighted_events).max(0.0);
        let minority_support = weighted_events.min(weightednonevents);
        let startwith_firth = should_enable_firth_from_class_support(minority_support, x.ncols());

        // Start with Firth when class support is low relative to model complexity.
        ext_opts.firth_bias_reduction = Some(startwith_firth);
        let first_try = optimize_external_designwith_heuristic_lambdas_andwarm_start(
            y,
            weights,
            &x,
            offset,
            specs.clone(),
            heuristic_lambdas,
            warm_start_beta,
            &ext_opts,
        );

        match first_try {
            Ok(res) => {
                let unstable_status = matches!(
                    res.pirls_status,
                    crate::pirls::PirlsStatus::MaxIterationsReached
                        | crate::pirls::PirlsStatus::Unstable
                );
                let extreme_eta = res.max_abs_eta > 15.0;
                if !startwith_firth && (unstable_status || extreme_eta) {
                    ext_opts.firth_bias_reduction = Some(true);
                    optimize_external_designwith_heuristic_lambdas_andwarm_start(
                        y,
                        weights,
                        &x,
                        offset,
                        specs.clone(),
                        heuristic_lambdas,
                        warm_start_beta,
                        &ext_opts,
                    )?
                } else {
                    res
                }
            }
            Err(err) => {
                if startwith_firth {
                    return Err(err);
                }
                ext_opts.firth_bias_reduction = Some(true);
                optimize_external_designwith_heuristic_lambdas_andwarm_start(
                    y,
                    weights,
                    &x,
                    offset,
                    specs.clone(),
                    heuristic_lambdas,
                    warm_start_beta,
                    &ext_opts,
                )?
            }
        }
    } else {
        optimize_external_designwith_heuristic_lambdas_andwarm_start(
            y,
            weights,
            &x,
            offset,
            specs.clone(),
            heuristic_lambdas,
            warm_start_beta,
            &ext_opts,
        )?
    };
    let log_lambdas = result.lambdas.mapv(|v| v.max(1e-300).ln());
    let edf = result
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);
    let geometry = result.inference.as_ref().map(|inf| FitGeometry {
        penalized_hessian: inf.penalized_hessian.clone(),
        working_weights: inf.working_weights.clone(),
        working_response: inf.working_response.clone(),
    });
    let covariance_conditional = result
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance.clone());
    let covariance_corrected = result
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance_corrected.clone());
    let penalized_objective =
        -result.log_likelihood + result.stable_penalty_term + result.reml_score;
    UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: result.beta.clone(),
            role: BlockRole::Mean,
            edf,
            lambdas: result.lambdas.clone(),
        }],
        log_lambdas,
        lambdas: result.lambdas,
        likelihood_family: Some(result.likelihood_family),
        likelihood_scale: result.likelihood_scale,
        log_likelihood_normalization: result.log_likelihood_normalization,
        log_likelihood: result.log_likelihood,
        deviance: result.deviance,
        reml_score: result.reml_score,
        stable_penalty_term: result.stable_penalty_term,
        penalized_objective,
        outer_iterations: result.iterations,
        outer_converged: true,
        outer_gradient_norm: result.finalgrad_norm,
        standard_deviation: result.standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference: result.inference,
        fitted_link: result.fitted_link,
        geometry,
        block_states: Vec::new(),
        pirls_status: result.pirls_status,
        max_abs_eta: result.max_abs_eta,
        constraint_kkt: result.constraint_kkt,
        artifacts: result.artifacts,
        inner_cycles: 0,
    })
}

/// External-design GAM entrypoint for GLM-style families supported by
/// `optimize_external_design`.
/// Survival families such as `RoystonParmar` use survival-specific training APIs.
pub fn fit_gam<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_heuristic_lambdas(x, y, weights, offset, s_list, None, family, opts)
}

const FIRTH_BASE_MINORITY_SUPPORT: f64 = 20.0;
const FIRTH_MINORITY_PER_PARAMETER: f64 = 2.0;

#[inline]
fn sas_log_deltaridgeweight() -> f64 {
    // Weak fixed stabilization for the SAS tail parameter to avoid
    // boundary/flat-region pathologies in outer optimization.
    1e-4
}

#[inline]
fn sas_log_delta_edge_barrierweight() -> f64 {
    // Keep SAS raw log-delta away from tanh-saturation edges where
    // link sensitivities collapse and outer gradients become uninformative.
    1e-2
}

#[inline]
fn sas_log_delta_bound() -> f64 {
    // Must match mixture_link::SAS_LOG_DELTA_BOUND.
    12.0
}

#[inline]
fn sas_log_delta_edge_barriercostgrad(raw_log_delta: f64) -> (f64, f64) {
    let w = sas_log_delta_edge_barrierweight();
    if w <= 0.0 || !raw_log_delta.is_finite() {
        return (0.0, 0.0);
    }
    let b = sas_log_delta_bound().max(f64::EPSILON);
    let t = (raw_log_delta / b).tanh();
    let one_minus_t2 = (1.0 - t * t).max(1e-12);
    let cost = -w * one_minus_t2.ln();
    // d/draw[-w log(1-t^2)] = (2w/B) * t.
    let grad = (2.0 * w / b) * t;
    (cost, grad)
}

#[inline]
fn sas_epsilon_bound() -> f64 {
    // Fixed smooth bound on raw SAS epsilon during outer optimization.
    8.0
}

#[inline]
fn sas_effective_epsilon(raw_epsilon: f64) -> (f64, f64) {
    let bound = sas_epsilon_bound().max(f64::EPSILON);
    let t = (raw_epsilon / bound).tanh();
    let epsilon = bound * t;
    let d_epsilon_d_raw = 1.0 - t * t;
    (epsilon, d_epsilon_d_raw)
}

#[inline]
fn should_enable_firth_from_class_support(minority_support: f64, n_features: usize) -> bool {
    let complexity_threshold =
        (FIRTH_MINORITY_PER_PARAMETER * n_features as f64).max(FIRTH_BASE_MINORITY_SUPPORT);
    minority_support < complexity_threshold
}

fn computefdgradient(
    reml_state: &RemlState,
    rho: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    compute_fd_gradient(reml_state, rho, true, true)
}

/// Evaluate both analytic and finite-difference gradients for the external REML objective.
pub fn evaluate_externalgradients<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_externalgradients")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_externalgradients",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        )));
    }

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    let analytic_grad = reml_state.compute_gradient(rho)?;
    let fdgrad = computefdgradient(&reml_state, rho)?;

    Ok((analytic_grad, fdgrad))
}

/// Evaluate the external cost and report the stabilization ridge used.
/// This is a diagnostic helper for tests that need to detect ridge jitter.
pub fn evaluate_externalcost_andridge<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_externalcost_andridge")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_externalcost_andridge",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        )));
    }

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    let cost = reml_state.compute_cost(rho)?;
    let ridge = reml_state.last_ridge_used().unwrap_or(0.0);
    Ok((cost, ridge))
}

#[cfg(test)]
mod fd_policy_tests {
    use super::reml::hyper::{LINK_BINOMIAL_AUX_MU_EPS, link_binomial_aux};
    use super::*;
    use crate::linalg::utils::max_abs_diag;
    use crate::mixture_link::{sas_inverse_link_jet, sas_inverse_link_jetwith_param_partials};
    use crate::types::LikelihoodFamily;
    use ndarray::{Array1, Array2, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_firth_support_rule_respects_legacy_low_support_floor() {
        assert!(should_enable_firth_from_class_support(19.99, 1));
        assert!(!should_enable_firth_from_class_support(20.0, 1));
    }

    #[test]
    fn test_firth_support_rule_scaleswith_model_dimension() {
        // High-dimensional setting: minority support that would pass the old
        // <20 rule still triggers Firth because support per parameter is weak.
        assert!(should_enable_firth_from_class_support(425.0, 225));
        assert!(!should_enable_firth_from_class_support(460.0, 225));
    }

    #[test]
    fn resolve_external_family_rejects_unsupported_firth_request() {
        let err = resolve_external_family(LikelihoodFamily::PoissonLog, Some(true))
            .expect_err("Poisson fitting should reject unsupported Firth requests explicitly");
        assert!(
            err.to_string()
                .contains("firth_bias_reduction is currently implemented only for"),
            "unexpected error: {err}"
        );
    }

    fn build_tiny_design(n: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let x1 = -1.5 + 3.0 * t;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1;
            x[[i, 2]] = (2.1 * x1).sin();
        }
        x
    }

    fn one_penalty_non_intercept(p: usize) -> Vec<Array2<f64>> {
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }
        vec![s]
    }

    fn dense_penalty_test_inputs(
        s_list: &[Array2<f64>],
        p: usize,
        context: &str,
    ) -> (
        Vec<PenaltySpec>,
        Vec<crate::construction::CanonicalPenalty>,
        Vec<usize>,
    ) {
        let penalty_specs = s_list
            .iter()
            .cloned()
            .map(PenaltySpec::Dense)
            .collect::<Vec<_>>();
        let (canonical_penalties, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(
                &penalty_specs,
                &vec![1; penalty_specs.len()],
                p,
                context,
            )
            .expect("canonicalize dense penalties");
        (penalty_specs, canonical_penalties, active_nullspace_dims)
    }

    #[test]
    fn sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19() {
        let seed = 19_u64;
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta_true = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
        assert!(effective_sas_link.is_some());
        let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
            &s_list,
            x.ncols(),
            "sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19",
        );
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
        ));
        let mut reml_state = RemlState::newwith_offset(
            y.view(),
            x_fit,
            w.view(),
            offset.view(),
            canonical_penalties.clone(),
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims.clone()),
            None,
            None,
        )
        .expect("reml_state");
        let rho = theta.slice(s![..1]).to_owned();
        let (epsilon_eff, d_eps_d_raw) = sas_effective_epsilon(theta[1]);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon_eff,
            initial_log_delta: theta[2],
        })
        .expect("sas state");
        reml_state.set_link_states(None, Some(sas_state));

        let pirls_result = reml_state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("pirls_result");
        println!(
            "sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19 lastgradient_norm={:.6e} status={:?} iteration={}",
            pirls_result.lastgradient_norm, pirls_result.status, pirls_result.iteration
        );
        let eta = &pirls_result.final_eta;
        let x_t = &pirls_result.x_transformed;
        let mut du_by_eps = Array1::<f64>::zeros(eta.len());
        let mut clampedobs = 0usize;
        for i in 0..eta.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta[i],
                sas_state.epsilon,
                sas_state.log_delta,
            );
            let mu = jets.jet.mu;
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
            if (mu.is_finite()
                && (mu <= LINK_BINOMIAL_AUX_MU_EPS || mu >= 1.0 - LINK_BINOMIAL_AUX_MU_EPS))
                || !mu.is_finite()
            {
                clampedobs += 1;
            }
            let d1 = jets.jet.d1;
            let dmu = jets.djet_depsilon.mu;
            let dd1 = jets.djet_depsilon.d1;
            du_by_eps[i] = aux.a2 * dmu * d1 + aux.a1 * dd1;
        }
        let score_at = |raw_eps: f64| -> Array1<f64> {
            let (eps_eff, _) = sas_effective_epsilon(raw_eps);
            let sas_state = state_from_sasspec(SasLinkSpec {
                initial_epsilon: eps_eff,
                initial_log_delta: theta[2],
            })
            .expect("score sas state");
            let mut out = Array1::<f64>::zeros(eta.len());
            for i in 0..eta.len() {
                let jets = sas_inverse_link_jetwith_param_partials(
                    eta[i],
                    sas_state.epsilon,
                    sas_state.log_delta,
                );
                let mu = jets.jet.mu;
                let d1 = jets.jet.d1;
                let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
                out[i] = aux.a1 * d1;
            }
            out
        };
        let score_p = score_at(theta[1] + 1e-4 * (1.0 + theta[1].abs()));
        let score_m = score_at(theta[1] - 1e-4 * (1.0 + theta[1].abs()));
        let fd_du_raw = (&score_p - &score_m).mapv(|v| v / (2.0 * 1e-4 * (1.0 + theta[1].abs())));
        let du_raw = du_by_eps.mapv(|v| v * d_eps_d_raw);
        crate::testing::assert_matrix_derivativefd(
            &fd_du_raw.insert_axis(Axis(1)),
            &du_raw.insert_axis(Axis(1)),
            2e-3,
            "sas du / d raw epsilon at fixed eta",
        );
        let rhs = x_t.transpose_vector_multiply(&du_by_eps);
        println!("sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19 clampedobs={clampedobs}");
        let mut neg_du_deta = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            );
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let d2 = jets.jet.d2;
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
            neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
        }
        let score_beta_jacobian = {
            let x_dense = x_t.to_dense();
            let diag_v = Array2::from_diag(&neg_du_deta);
            let mut j = x_dense.t().dot(&diag_v).dot(&x_dense);
            for ((r, c), v) in pirls_result.reparam_result.s_transformed.indexed_iter() {
                j[[r, c]] += v;
            }
            if pirls_result.ridge_used > 0.0 {
                for d in 0..j.nrows() {
                    j[[d, d]] += pirls_result.ridge_used;
                }
            }
            j
        };
        let stable_solver = StableSolver::new("sas dbeta exact test");
        let mut dbeta_exact = stable_solver
            .solvevectorwithridge_retries(
                &score_beta_jacobian,
                &rhs,
                max_abs_diag(&score_beta_jacobian) * 1e-12,
            )
            .expect("observed-jacobian solve for dbeta");
        dbeta_exact *= d_eps_d_raw;

        let fd_h = 1e-4 * (1.0 + theta[1].abs());
        let beta_at = |raw_eps: f64| -> Array1<f64> {
            let mut state = RemlState::newwith_offset(
                y.view(),
                conditioning.apply_to_design(&DesignMatrix::Dense(
                    crate::matrix::DenseDesignMatrix::from(x.clone()),
                )),
                w.view(),
                offset.view(),
                canonical_penalties.clone(),
                x.ncols(),
                &cfg,
                Some(active_nullspace_dims.clone()),
                None,
                None,
            )
            .expect("fd state");
            let (eps_eff, _) = sas_effective_epsilon(raw_eps);
            let sas_state = state_from_sasspec(SasLinkSpec {
                initial_epsilon: eps_eff,
                initial_log_delta: theta[2],
            })
            .expect("fd sas state");
            state.set_link_states(None, Some(sas_state));
            let pirls = state
                .obtain_eval_bundle(&rho)
                .map(|b| b.pirls_result.clone())
                .expect("fd pirls");
            pirls.beta_transformed.as_ref().clone()
        };
        let beta_p = beta_at(theta[1] + fd_h);
        let beta_m = beta_at(theta[1] - fd_h);
        let fd_beta = (&beta_p - &beta_m).mapv(|v| v / (2.0 * fd_h));

        crate::testing::assert_matrix_derivativefd(
            &fd_beta.insert_axis(Axis(1)),
            &dbeta_exact.insert_axis(Axis(1)),
            2e-3,
            "sas observed-jacobian dbeta / d raw epsilon",
        );
    }

    #[test]
    fn sas_true_score_beta_jacobian_matchesfd_at_seed19() {
        let seed = 19_u64;
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta_true = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
        assert!(effective_sas_link.is_some());
        let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
            &s_list,
            x.ncols(),
            "sas_true_score_beta_jacobian_matchesfd_at_seed19",
        );
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
        ));
        let mut reml_state = RemlState::newwith_offset(
            y.view(),
            x_fit,
            w.view(),
            offset.view(),
            canonical_penalties,
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims),
            None,
            None,
        )
        .expect("reml_state");
        let rho = theta.slice(s![..1]).to_owned();
        let (epsilon_eff, _) = sas_effective_epsilon(theta[1]);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon_eff,
            initial_log_delta: theta[2],
        })
        .expect("sas state");
        reml_state.set_link_states(None, Some(sas_state));

        let pirls_result = reml_state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("pirls_result");
        let beta0 = pirls_result.beta_transformed.as_ref().clone();
        let s_transformed = pirls_result.reparam_result.s_transformed.clone();
        let ridge = pirls_result.ridge_used;
        let x_dense = match &pirls_result.x_transformed {
            DesignMatrix::Dense(x_dense) => x_dense.to_dense(),
            DesignMatrix::Sparse(_) => {
                panic!("expected dense transformed design in seed-19 SAS test")
            }
        };

        let gradient_at = |beta: &Array1<f64>| -> Array1<f64> {
            let mut eta = offset.clone();
            eta += &x_dense.dot(beta);
            let mut u = Array1::<f64>::zeros(eta.len());
            for i in 0..eta.len() {
                let jets = sas_inverse_link_jetwith_param_partials(
                    eta[i].clamp(-30.0, 30.0),
                    sas_state.epsilon,
                    sas_state.log_delta,
                );
                let mu = jets.jet.mu;
                let d1 = jets.jet.d1;
                let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
                u[i] = aux.a1 * d1;
            }
            let mut g = -x_dense.t().dot(&u);
            g += &s_transformed.dot(beta);
            if ridge > 0.0 {
                g += &beta.mapv(|v| ridge * v);
            }
            g
        };

        let mut analytic_j = Array2::<f64>::zeros((beta0.len(), beta0.len()));
        let mut eta0 = offset.clone();
        eta0 += &x_dense.dot(&beta0);
        let mut neg_du_deta = Array1::<f64>::zeros(eta0.len());
        for i in 0..eta0.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta0[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            );
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let d2 = jets.jet.d2;
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
            neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
        }
        let weighted_x = &x_dense * &neg_du_deta.insert_axis(Axis(1));
        analytic_j.assign(&x_dense.t().dot(&weighted_x));
        analytic_j += &s_transformed;
        if ridge > 0.0 {
            for j in 0..analytic_j.nrows() {
                analytic_j[[j, j]] += ridge;
            }
        }

        let mut fd_j = Array2::<f64>::zeros((beta0.len(), beta0.len()));
        for j in 0..beta0.len() {
            let h = 1e-5 * (1.0 + beta0[j].abs());
            let mut beta_p = beta0.clone();
            let mut beta_m = beta0.clone();
            beta_p[j] += h;
            beta_m[j] -= h;
            let g_p = gradient_at(&beta_p);
            let g_m = gradient_at(&beta_m);
            let fd_col = (&g_p - &g_m).mapv(|v| v / (2.0 * h));
            fd_j.column_mut(j).assign(&fd_col);
        }

        crate::testing::assert_matrix_derivativefd(
            &fd_j,
            &analytic_j,
            2e-3,
            "sas true beta-score jacobian at seed-19",
        );
    }

    #[test]
    fn sas_pirlshessian_matches_true_score_jacobian_at_seed19() {
        let seed = 19_u64;
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta_true = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
        assert!(effective_sas_link.is_some());
        let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
            &s_list,
            x.ncols(),
            "sas_pirlshessian_matches_true_score_jacobian_at_seed19",
        );
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
        ));
        let mut reml_state = RemlState::newwith_offset(
            y.view(),
            x_fit,
            w.view(),
            offset.view(),
            canonical_penalties,
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims),
            None,
            None,
        )
        .expect("reml_state");
        let rho = theta.slice(s![..1]).to_owned();
        let (epsilon_eff, _) = sas_effective_epsilon(theta[1]);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon_eff,
            initial_log_delta: theta[2],
        })
        .expect("sas state");
        reml_state.set_link_states(None, Some(sas_state));

        let pirls_result = reml_state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("pirls_result");
        let beta0 = pirls_result.beta_transformed.as_ref().clone();
        let s_transformed = pirls_result.reparam_result.s_transformed.clone();
        let ridge = pirls_result.ridge_used;
        let x_dense = match &pirls_result.x_transformed {
            DesignMatrix::Dense(x_dense) => x_dense.to_dense(),
            DesignMatrix::Sparse(_) => {
                panic!("expected dense transformed design in seed-19 SAS test")
            }
        };

        let mut eta0 = offset.clone();
        eta0 += &x_dense.dot(&beta0);
        let mut neg_du_deta = Array1::<f64>::zeros(eta0.len());
        for i in 0..eta0.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta0[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            );
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let d2 = jets.jet.d2;
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
            neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
        }
        let weighted_x = &x_dense * &neg_du_deta.insert_axis(Axis(1));
        let mut true_jacobian = x_dense.t().dot(&weighted_x);
        true_jacobian += &s_transformed;
        if ridge > 0.0 {
            for j in 0..true_jacobian.nrows() {
                true_jacobian[[j, j]] += ridge;
            }
        }

        let pht_dense = pirls_result.penalized_hessian_transformed.to_dense();
        let max_abs_diff = true_jacobian
            .iter()
            .zip(pht_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff <= 2e-3,
            "expected PIRLS Hessian to match the true SAS score Jacobian, got max_abs_diff={max_abs_diff:.3e}"
        );
    }

    #[test]
    fn link_binomial_aux_stay_finite_for_saturated_sas_probabilities() {
        let saturated_cases = [
            (
                0.0,
                sas_inverse_link_jetwith_param_partials(-30.0, 0.0, 12.0)
                    .jet
                    .mu,
            ),
            (
                1.0,
                sas_inverse_link_jetwith_param_partials(30.0, 0.0, 12.0)
                    .jet
                    .mu,
            ),
        ];
        for (yi, mu) in saturated_cases {
            let aux = link_binomial_aux(yi, 1.0, mu);
            assert!(aux.a1.is_finite(), "a1 must be finite for yi={yi} mu={mu}");
            assert!(aux.a2.is_finite(), "a2 must be finite for yi={yi} mu={mu}");
            assert!(
                aux.variance.is_finite() && aux.variance > 0.0,
                "variance must be finite and positive for yi={yi} mu={mu}"
            );
        }
    }
}

#[cfg(test)]
mod continuous_order_tests {
    use super::*;

    #[test]
    fn continuous_order_formula_matches_closed_form() {
        let out = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
        let r = out.r_ratio.expect("R");
        let nu = out.nu.expect("nu");
        let kappa2 = out.kappa2.expect("kappa2");
        assert!((r - (100.0 / 6.0)).abs() < 1e-12);
        assert!((nu - (r / (r - 2.0))).abs() < 1e-12);
        assert!((kappa2 - (10.0 / ((r - 2.0) * 3.0))).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_unscales_lambdas_exactly_by_ck() {
        let out = compute_continuous_smoothness_order([6.0, 15.0, 9.0], [3.0, 5.0, 9.0], 1e-12);
        // Physical lambdas must satisfy lambda_k = lambda_tilde_k / c_k.
        assert!((out.lambda0 - 2.0).abs() < 1e-12);
        assert!((out.lambda1 - 3.0).abs() < 1e-12);
        assert!((out.lambda2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_invalid_ck_is_guarded() {
        let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 0.0, 1.0], 1e-12);
        assert_eq!(
            out.status,
            ContinuousSmoothnessOrderStatus::UndefinedZeroLambda
        );
        assert!(out.r_ratio.is_none());
    }

    #[test]
    fn continuous_order_is_invariant_to_penalty_normalization_reversal() {
        let base = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        let scaled = compute_continuous_smoothness_order(
            [2.0 * 4.0, 10.0 * 0.5, 3.0 * 8.0],
            [4.0, 0.5, 8.0],
            1e-12,
        );
        assert_eq!(base.status, ContinuousSmoothnessOrderStatus::Ok);
        assert_eq!(scaled.status, ContinuousSmoothnessOrderStatus::Ok);
        assert!((base.r_ratio.unwrap() - scaled.r_ratio.unwrap()).abs() < 1e-12);
        assert!((base.nu.unwrap() - scaled.nu.unwrap()).abs() < 1e-12);
        assert!((base.kappa2.unwrap() - scaled.kappa2.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_flags_non_matern_regimewhen_r_le_4() {
        let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
        assert!(out.nu.is_none());
        assert!(out.kappa2.is_none());
    }

    #[test]
    fn continuous_order_reports_effective_nu_kappa_in_non_matern_bandwhen_r_gt_2() {
        let out = compute_continuous_smoothness_order([1.0, 3.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
        let r = out.r_ratio.expect("R");
        assert!(r > 2.0 && r < 4.0);
        assert!(out.nu.is_some());
        assert!(out.kappa2.is_some());
    }

    #[test]
    fn continuous_order_boundary_r_equals_four_is_matern_square_case() {
        let out = compute_continuous_smoothness_order([1.0, 2.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
        let nu = out.nu.expect("nu");
        assert!((nu - 2.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_guardszero_or_nearzero_lambda() {
        let out = compute_continuous_smoothness_order([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
        assert!(out.r_ratio.is_none());
    }

    #[test]
    fn continuous_order_first_order_limitwhen_lambda2_collapses() {
        let out = compute_continuous_smoothness_order([2.0, 4.0, 1e-20], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::FirstOrderLimit);
        assert_eq!(out.nu, Some(1.0));
        let k2 = out.kappa2.expect("kappa2");
        assert!((k2 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_intrinsic_limitwhen_lambda0_collapses() {
        let out = compute_continuous_smoothness_order([1e-20, 4.0, 2.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
        assert_eq!(out.nu, Some(1.0));
        assert_eq!(out.kappa2, Some(0.0));
    }

    #[test]
    fn continuous_order_is_only_defined_for_three_penalties_per_term() {
        let ok =
            try_compute_continuous_smoothness_order(&[2.0, 10.0, 3.0], &[1.0, 1.0, 1.0], 1e-12);
        let two = try_compute_continuous_smoothness_order(&[2.0, 10.0], &[1.0, 1.0], 1e-12);
        let four = try_compute_continuous_smoothness_order(
            &[2.0, 10.0, 3.0, 7.0],
            &[1.0, 1.0, 1.0, 1.0],
            1e-12,
        );
        assert!(ok.is_some());
        assert!(two.is_none());
        assert!(four.is_none());
    }
}

#[path = "reml/mod.rs"]
pub(crate) mod reml;
