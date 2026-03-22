use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyPsiDesignAction,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, build_block_spatial_psi_derivatives, cost_gated_outer_order,
    custom_family_outer_derivatives, evaluate_custom_family_joint_hyper, first_psi_linear_map,
    fit_custom_family, second_psi_linear_map,
};
use crate::estimate::UnifiedFitResult;
use crate::families::bernoulli_marginal_slope::{
    signed_probit_logcdf_and_mills_ratio, signed_probit_neglog_derivatives_up_to_fourth,
    unary_derivatives_log, unary_derivatives_log_normal_pdf, unary_derivatives_neglog_phi,
    unary_derivatives_sqrt,
};
use crate::families::gamlss::monotone_wiggle_basis_with_derivative_order;
use crate::families::row_kernel::{RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache};
use crate::families::survival_location_scale::{
    TimeBlockInput, TimeWiggleBlockInput, project_onto_linear_constraints,
    time_derivative_lower_bound_constraints,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::reml::unified::HyperOperator;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

// ── Spec and result types ─────────────────────────────────────────────

#[derive(Clone)]
pub struct SurvivalMarginalSlopeTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub marginalspec: TermCollectionSpec,
    /// Strict lower bound on q'(t) used by both the likelihood domain and
    /// the monotonicity constraints.
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub logslopespec: TermCollectionSpec,
}

pub const DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD: f64 = 1e-6;

pub struct SurvivalMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub logslopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub logslope_design: TermCollectionDesign,
    pub baseline_slope: f64,
    pub time_block_penalties_len: usize,
}

// ── Family struct ─────────────────────────────────────────────────────

/// The time block has one beta vector but THREE design matrices (entry, exit,
/// derivative-at-exit). The ParameterBlockSpec uses the exit design as its
/// "official" design, so block_states[0].eta = design_exit @ beta + offset_exit.
/// This eta is NOT used in the likelihood computation — row_neglog_directional
/// recomputes all 3 linear predictors from beta_time directly. The exit-design
/// eta exists only to satisfy the CustomFamily/PIRLS interface; ExactNewton
/// blocks do not use eta for working response/weights.
#[derive(Clone)]
struct SurvivalMarginalSlopeFamily {
    n: usize,
    event: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    z: Arc<Array1<f64>>,
    derivative_guard: f64,
    /// Time block: 3 designs sharing one beta vector.
    /// Stored as DesignMatrix to support sparse local-support bases at
    /// biobank scale (B-spline/I-spline rows have only degree+1 nonzeros).
    design_entry: DesignMatrix,
    design_exit: DesignMatrix,
    design_derivative_exit: DesignMatrix,
    offset_entry: Arc<Array1<f64>>,
    offset_exit: Arc<Array1<f64>>,
    derivative_offset_exit: Arc<Array1<f64>>,
    /// Baseline covariate block: contributes additively to q0 and q1, but not qd1.
    marginal_design: DesignMatrix,
    /// Log-slope block: standard single design.
    logslope_design: DesignMatrix,
    time_linear_constraints: Option<LinearInequalityConstraints>,
    time_wiggle_knots: Option<Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
}

// ── Block layout ──────────────────────────────────────────────────────

#[derive(Clone)]
struct BlockSlices {
    time: std::ops::Range<usize>,
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    total: usize,
}

fn block_slices(block_states: &[ParameterBlockState]) -> BlockSlices {
    let time = 0..block_states[0].beta.len();
    let marginal = time.end..time.end + block_states[1].beta.len();
    let logslope = marginal.end..marginal.end + block_states[2].beta.len();
    let total = logslope.end;
    BlockSlices {
        time,
        marginal,
        logslope,
        total,
    }
}

// ── Primary-space helpers ─────────────────────────────────────────────

// Primary scalar indices: 0=q0, 1=q1, 2=qd1, 3=g
const N_PRIMARY: usize = 4;

#[derive(Clone)]
struct SurvivalTimeWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
}

#[derive(Clone)]
struct SurvivalMarginalSlopeDynamicRow {
    q0: f64,
    q1: f64,
    qd1: f64,
    dq0_time: Array1<f64>,
    dq1_time: Array1<f64>,
    dqd1_time: Array1<f64>,
    dq0_marginal: Array1<f64>,
    dq1_marginal: Array1<f64>,
    dqd1_marginal: Array1<f64>,
    d2q0_time_time: Array2<f64>,
    d2q1_time_time: Array2<f64>,
    d2qd1_time_time: Array2<f64>,
    d2q0_time_marginal: Array2<f64>,
    d2q1_time_marginal: Array2<f64>,
    d2qd1_time_marginal: Array2<f64>,
    d2q0_marginal_marginal: Array2<f64>,
    d2q1_marginal_marginal: Array2<f64>,
    d2qd1_marginal_marginal: Array2<f64>,
}

fn unit_primary_direction(idx: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    out[idx] = 1.0;
    out
}

fn spatial_block_primary_loading(block_idx: usize) -> Result<Array1<f64>, String> {
    match block_idx {
        1 => Ok(Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0])),
        2 => Ok(Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0])),
        _ => Err(format!(
            "survival marginal-slope spatial psi loading requested for unsupported block {block_idx}"
        )),
    }
}

fn jet_subset_partitions(mask: usize) -> Vec<Vec<usize>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let first = mask & mask.wrapping_neg();
    let rest = mask ^ first;
    let mut out = Vec::new();
    let mut subset = rest;
    loop {
        let block = first | subset;
        for mut remainder in jet_subset_partitions(rest ^ subset) {
            remainder.push(block);
            out.push(remainder);
        }
        if subset == 0 {
            break;
        }
        subset = (subset - 1) & rest;
    }
    out
}

#[derive(Clone)]
struct MultiDirJet {
    coeffs: Vec<f64>,
}

impl MultiDirJet {
    fn zero(n_dirs: usize) -> Self {
        Self {
            coeffs: vec![0.0; 1usize << n_dirs],
        }
    }

    fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    fn linear(n_dirs: usize, base: f64, first: &[f64]) -> Self {
        let mut out = Self::constant(n_dirs, base);
        for (idx, &value) in first.iter().take(n_dirs).enumerate() {
            out.coeffs[1usize << idx] = value;
        }
        out
    }

    fn full_mask(&self) -> usize {
        self.coeffs.len() - 1
    }

    fn coeff(&self, mask: usize) -> f64 {
        self.coeffs[mask]
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(l, r)| l + r)
                .collect(),
        }
    }

    fn scale(&self, scalar: f64) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|v| scalar * v).collect(),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        for mask in 0..count {
            let mut total = 0.0;
            let mut submask = mask;
            loop {
                total += self.coeffs[submask] * other.coeffs[mask ^ submask];
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & mask;
            }
            out[mask] = total;
        }
        Self { coeffs: out }
    }

    fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        out[0] = derivs[0];
        for (mask, value) in out.iter_mut().enumerate().skip(1) {
            let mut total = 0.0;
            for partition in jet_subset_partitions(mask) {
                let order = partition.len();
                if order == 0 || order >= derivs.len() {
                    continue;
                }
                let mut prod = 1.0;
                for block in partition {
                    prod *= self.coeffs[block];
                }
                total += derivs[order] * prod;
            }
            *value = total;
        }
        Self { coeffs: out }
    }
}

/// Derive a primary-space direction from a precomputed psi design row and beta,
/// avoiding a redundant `psi_design_row_vector` call inside `row_primary_psi_direction`.
fn primary_direction_from_psi_row(
    block_idx: usize,
    psi_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    let value = psi_row.dot(beta_block);
    match block_idx {
        1 => {
            out[0] = value;
            out[1] = value;
        }
        2 => {
            out[3] = value;
        }
        _ => {}
    }
    out
}

/// Derive a primary-space psi action on a direction from a precomputed psi design row.
fn primary_psi_action_from_psi_row(
    block_idx: usize,
    psi_row: &Array1<f64>,
    d_beta_block: ndarray::ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    let value = psi_row.dot(&d_beta_block);
    match block_idx {
        1 => {
            out[0] = value;
            out[1] = value;
        }
        2 => {
            out[3] = value;
        }
        _ => {}
    }
    out
}

/// Derive a primary-space second-order direction from a precomputed second psi design row.
fn primary_second_direction_from_psi_row(
    block_idx: usize,
    psi_second_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    primary_direction_from_psi_row(block_idx, psi_second_row, beta_block)
}

// ── Block-local Hessian accumulator ────────────────────────────────────
//
// Avoids O(n p²) per-row allocation of full p×p matrices by accumulating
// the 6 independent block matrices (3 diagonal + 3 off-diagonal) directly.
// Assembly to a dense p×p matrix or an implicit operator is a single O(p²)
// pass at the end, after the n-loop.

struct BlockHessianAccumulator {
    h_tt: Array2<f64>,
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_tm: Array2<f64>,
    h_tg: Array2<f64>,
    h_mg: Array2<f64>,
}

impl BlockHessianAccumulator {
    fn new(p_t: usize, p_m: usize, p_g: usize) -> Self {
        Self {
            h_tt: Array2::zeros((p_t, p_t)),
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_tm: Array2::zeros((p_t, p_m)),
            h_tg: Array2::zeros((p_t, p_g)),
            h_mg: Array2::zeros((p_m, p_g)),
        }
    }

    /// Accumulate a primary-space Hessian into block-local matrices.
    /// Equivalent to `add_pullback_primary_hessian` but avoids the p×p target.
    fn add_pullback(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        primary_hessian: &Array2<f64>,
    ) {
        // Time×time block: 3×3 design cross-products
        let time_designs = [
            &family.design_entry,
            &family.design_exit,
            &family.design_derivative_exit,
        ];
        for a in 0..3 {
            for b in 0..3 {
                time_designs[a]
                    .row_outer_into(
                        row,
                        time_designs[b],
                        primary_hessian[[a, b]],
                        &mut self.h_tt,
                    )
                    .expect("time block row_outer_into dimension mismatch");
            }
        }

        // Marginal×marginal: single rank-1 with combined weight
        let mm_weight = primary_hessian[[0, 0]]
            + primary_hessian[[0, 1]]
            + primary_hessian[[1, 0]]
            + primary_hessian[[1, 1]];
        family
            .marginal_design
            .syr_row_into(row, mm_weight, &mut self.h_mm)
            .expect("marginal syr_row_into dimension mismatch");

        // Logslope×logslope: single rank-1
        family
            .logslope_design
            .syr_row_into(row, primary_hessian[[3, 3]], &mut self.h_gg)
            .expect("logslope syr_row_into dimension mismatch");

        // Marginal×logslope cross-block
        let mg_weight = primary_hessian[[0, 3]] + primary_hessian[[1, 3]];
        if mg_weight != 0.0 {
            let m_chunk = family.marginal_design.row_chunk(row..row + 1);
            let m_row = m_chunk.row(0);
            let g_chunk = family.logslope_design.row_chunk(row..row + 1);
            let g_row = g_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                mg_weight,
                &m_row.view().insert_axis(Axis(1)),
                &g_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_mg,
            );
        }

        // Time×logslope cross-block
        let tg_weights = [
            primary_hessian[[0, 3]],
            primary_hessian[[1, 3]],
            primary_hessian[[2, 3]],
        ];
        let g_chunk = family.logslope_design.row_chunk(row..row + 1);
        let g_row = g_chunk.row(0);
        for (des, alpha) in time_designs.iter().zip(tg_weights.iter()) {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des.row_chunk(row..row + 1);
            let t_row = t_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                *alpha,
                &t_row.view().insert_axis(Axis(1)),
                &g_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_tg,
            );
        }

        // Time×marginal cross-block
        let tm_weights = [
            primary_hessian[[0, 0]] + primary_hessian[[0, 1]],
            primary_hessian[[1, 0]] + primary_hessian[[1, 1]],
            primary_hessian[[2, 0]] + primary_hessian[[2, 1]],
        ];
        let m_chunk = family.marginal_design.row_chunk(row..row + 1);
        let m_row = m_chunk.row(0);
        for (des, alpha) in time_designs.iter().zip(tm_weights.iter()) {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des.row_chunk(row..row + 1);
            let t_row = t_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                *alpha,
                &t_row.view().insert_axis(Axis(1)),
                &m_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_tm,
            );
        }
    }

    /// Add a rank-1 update from psi_row (in the psi block) crossed with the
    /// pullback of a primary-space vector. Adds both left⊗right and right⊗left.
    fn add_rank1_psi_cross(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        psi_block_idx: usize,
        psi_row: &Array1<f64>,
        right_primary: &Array1<f64>,
    ) {
        // right_primary components mapped to blocks:
        // time:     entry*rp[0] + exit*rp[1] + deriv*rp[2]
        // marginal: marginal*(rp[0] + rp[1])
        // logslope: logslope*rp[3]
        let psi_col = psi_row.view().insert_axis(Axis(1));

        // Block (psi, time): psi_row ⊗ right_time
        // Block (time, psi): right_time ⊗ psi_row  (= transpose of above)
        let time_designs = [
            (&family.design_entry, right_primary[0]),
            (&family.design_exit, right_primary[1]),
            (&family.design_derivative_exit, right_primary[2]),
        ];
        for (des, alpha) in &time_designs {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des.row_chunk(row..row + 1);
            let t_row = t_chunk.row(0);
            let t_col = t_row.view().insert_axis(Axis(1));
            match psi_block_idx {
                1 => {
                    // psi=marginal: (time, marginal) block = h_tm
                    // right⊗left: right_time ⊗ psi_row → h_tm
                    ndarray::linalg::general_mat_mul(
                        *alpha,
                        &t_col,
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_tm,
                    );
                    // left⊗right: psi_row ⊗ right_time → h_tm^T (handled by symmetry)
                }
                2 => {
                    // psi=logslope: (time, logslope) block = h_tg
                    ndarray::linalg::general_mat_mul(
                        *alpha,
                        &t_col,
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_tg,
                    );
                }
                _ => {}
            }
        }

        // Block (psi, marginal) or (marginal, psi)
        let m_alpha = right_primary[0] + right_primary[1];
        if m_alpha != 0.0 {
            let m_chunk = family.marginal_design.row_chunk(row..row + 1);
            let m_row = m_chunk.row(0);
            match psi_block_idx {
                1 => {
                    // psi=marginal: (marginal, marginal) = h_mm, symmetric rank-2
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &psi_col,
                        &m_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mm,
                    );
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &m_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mm,
                    );
                }
                2 => {
                    // psi=logslope: (marginal, logslope) block = h_mg
                    // left⊗right: psi_row(logslope) ⊗ m_row → goes to h_mg^T
                    // right⊗left: m_row ⊗ psi_row(logslope) → goes to h_mg
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &m_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mg,
                    );
                }
                _ => {}
            }
        }

        // Block (psi, logslope) or (logslope, psi)
        if right_primary[3] != 0.0 {
            let g_chunk = family.logslope_design.row_chunk(row..row + 1);
            let g_row = g_chunk.row(0);
            match psi_block_idx {
                1 => {
                    // psi=marginal: (marginal, logslope) = h_mg
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &psi_col,
                        &g_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mg,
                    );
                }
                2 => {
                    // psi=logslope: (logslope, logslope) = h_gg, symmetric rank-2
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &psi_col,
                        &g_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_gg,
                    );
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &g_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_gg,
                    );
                }
                _ => {}
            }
        }
    }

    /// Add outer product of two psi block-local rows (possibly in different blocks).
    /// Adds both α·(a ⊗ b) and α·(b ⊗ a) to maintain symmetry.
    ///
    /// The full p×p symmetric Hessian has blocks:
    ///   (block_i, block_j) += α · psi_row_i ⊗ psi_row_j
    ///   (block_j, block_i) += α · psi_row_j ⊗ psi_row_i   (= transpose)
    /// Our off-diagonal storage convention (h_mg = marginal×logslope) handles
    /// the transpose automatically in to_dense/operator assembly.
    fn add_psi_psi_outer(
        &mut self,
        block_i: usize,
        psi_row_i: &Array1<f64>,
        block_j: usize,
        psi_row_j: &Array1<f64>,
        alpha: f64,
    ) {
        if alpha == 0.0 {
            return;
        }
        let col_i = psi_row_i.view().insert_axis(Axis(1));
        let row_j = psi_row_j.view().insert_axis(Axis(0));

        if block_i == block_j {
            // Same block: symmetric rank-2 update to the diagonal block
            let col_j = psi_row_j.view().insert_axis(Axis(1));
            let row_i = psi_row_i.view().insert_axis(Axis(0));
            let target = match block_i {
                1 => &mut self.h_mm,
                2 => &mut self.h_gg,
                _ => return,
            };
            ndarray::linalg::general_mat_mul(alpha, &col_i, &row_j, 1.0, target);
            ndarray::linalg::general_mat_mul(alpha, &col_j, &row_i, 1.0, target);
        } else {
            // Different blocks: one rank-1 update to h_mg.
            // Block (marginal, logslope) gets the psi_marginal ⊗ psi_logslope contribution;
            // the (logslope, marginal) transpose is assembled from h_mg^T automatically.
            let (marginal_row, logslope_row) = if block_i == 1 {
                (psi_row_i, psi_row_j)
            } else {
                (psi_row_j, psi_row_i)
            };
            let m_col = marginal_row.view().insert_axis(Axis(1));
            let g_row = logslope_row.view().insert_axis(Axis(0));
            ndarray::linalg::general_mat_mul(alpha, &m_col, &g_row, 1.0, &mut self.h_mg);
        }
    }

    /// Assemble into a dense p×p matrix.
    fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        out.slice_mut(s![slices.time.clone(), slices.time.clone()])
            .assign(&self.h_tt);
        out.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![slices.time.clone(), slices.marginal.clone()])
            .assign(&self.h_tm);
        out.slice_mut(s![slices.marginal.clone(), slices.time.clone()])
            .assign(&self.h_tm.t());
        out.slice_mut(s![slices.time.clone(), slices.logslope.clone()])
            .assign(&self.h_tg);
        out.slice_mut(s![slices.logslope.clone(), slices.time.clone()])
            .assign(&self.h_tg.t());
        out.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .assign(&self.h_mg.t());
        out
    }

    fn into_operator(self, slices: BlockSlices) -> BlockHessianOperator {
        BlockHessianOperator {
            h_tt: self.h_tt,
            h_mm: self.h_mm,
            h_gg: self.h_gg,
            h_tm: self.h_tm,
            h_tg: self.h_tg,
            h_mg: self.h_mg,
            slices,
        }
    }

    fn add(&mut self, other: &BlockHessianAccumulator) {
        self.h_tt += &other.h_tt;
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_tm += &other.h_tm;
        self.h_tg += &other.h_tg;
        self.h_mg += &other.h_mg;
    }
}

/// Block-structured HyperOperator for survival marginal-slope psi Hessians.
/// Stores 6 block matrices and performs matvec in O(sum of block²) instead of
/// O(p²), where p = p_time + p_marginal + p_logslope.
struct BlockHessianOperator {
    h_tt: Array2<f64>,
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_tm: Array2<f64>,
    h_tg: Array2<f64>,
    h_mg: Array2<f64>,
    slices: BlockSlices,
}

impl HyperOperator for BlockHessianOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let v_t = v.slice(s![self.slices.time.clone()]);
        let v_m = v.slice(s![self.slices.marginal.clone()]);
        let v_g = v.slice(s![self.slices.logslope.clone()]);
        let mut out = Array1::zeros(self.slices.total);
        {
            let mut o_t = out.slice_mut(s![self.slices.time.clone()]);
            o_t += &self.h_tt.dot(&v_t);
            o_t += &self.h_tm.dot(&v_m);
            o_t += &self.h_tg.dot(&v_g);
        }
        {
            let mut o_m = out.slice_mut(s![self.slices.marginal.clone()]);
            o_m += &self.h_tm.t().dot(&v_t);
            o_m += &self.h_mm.dot(&v_m);
            o_m += &self.h_mg.dot(&v_g);
        }
        {
            let mut o_g = out.slice_mut(s![self.slices.logslope.clone()]);
            o_g += &self.h_tg.t().dot(&v_t);
            o_g += &self.h_mg.t().dot(&v_m);
            o_g += &self.h_gg.dot(&v_g);
        }
        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_t = v.slice(s![self.slices.time.clone()]);
        let v_m = v.slice(s![self.slices.marginal.clone()]);
        let v_g = v.slice(s![self.slices.logslope.clone()]);
        let u_t = u.slice(s![self.slices.time.clone()]);
        let u_m = u.slice(s![self.slices.marginal.clone()]);
        let u_g = u.slice(s![self.slices.logslope.clone()]);
        // Diagonal blocks
        let mut total = v_t.dot(&self.h_tt.dot(&u_t));
        total += v_m.dot(&self.h_mm.dot(&u_m));
        total += v_g.dot(&self.h_gg.dot(&u_g));
        // Off-diagonal blocks (symmetric)
        total += v_t.dot(&self.h_tm.dot(&u_m));
        total += v_m.dot(&self.h_tm.t().dot(&u_t));
        total += v_t.dot(&self.h_tg.dot(&u_g));
        total += v_g.dot(&self.h_tg.t().dot(&u_t));
        total += v_m.dot(&self.h_mg.dot(&u_g));
        total += v_g.dot(&self.h_mg.t().dot(&u_m));
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.slices.total, self.slices.total));
        out.slice_mut(s![self.slices.time.clone(), self.slices.time.clone()])
            .assign(&self.h_tt);
        out.slice_mut(s![
            self.slices.marginal.clone(),
            self.slices.marginal.clone()
        ])
        .assign(&self.h_mm);
        out.slice_mut(s![
            self.slices.logslope.clone(),
            self.slices.logslope.clone()
        ])
        .assign(&self.h_gg);
        out.slice_mut(s![self.slices.time.clone(), self.slices.marginal.clone()])
            .assign(&self.h_tm);
        out.slice_mut(s![self.slices.marginal.clone(), self.slices.time.clone()])
            .assign(&self.h_tm.t());
        out.slice_mut(s![self.slices.time.clone(), self.slices.logslope.clone()])
            .assign(&self.h_tg);
        out.slice_mut(s![self.slices.logslope.clone(), self.slices.time.clone()])
            .assign(&self.h_tg.t());
        out.slice_mut(s![
            self.slices.marginal.clone(),
            self.slices.logslope.clone()
        ])
        .assign(&self.h_mg);
        out.slice_mut(s![
            self.slices.logslope.clone(),
            self.slices.marginal.clone()
        ])
        .assign(&self.h_mg.t());
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

// ── Closed-form row kernel ─────────────────────────────────────────────
//
// The survival marginal-slope NLL for row i is:
//
//   ℓ_i = w_i [ (1-d)·neglogΦ(-η₁) + logΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
//
// with η₀ = q₀c + gz, η₁ = q₁c + gz, a'₁ = qd₁·c, c = √(1+g²).
//
// All derivatives w.r.t. the 4 primary scalars (q₀, q₁, qd₁, g) are
// closed-form scalar formulas. No jets, no per-row heap allocation.

/// Derivatives of c(g) = √(1+g²) up to 4th order.
#[inline]
fn c_derivatives(g: f64) -> (f64, f64, f64, f64, f64) {
    let g2 = g * g;
    let c = (1.0 + g2).sqrt();
    let c2 = c * c;
    let c3 = c2 * c;
    let c5 = c3 * c2;
    let c7 = c5 * c2;
    let c1 = g / c;
    let c2d = 1.0 / c3;
    let c3d = -3.0 * g / c5;
    let c4d = (12.0 * g2 - 3.0) / c7;
    (c, c1, c2d, c3d, c4d)
}

/// Derivatives of neglog(x) = -log(x): [-1/x, 1/x², -2/x³, 6/x⁴].
#[inline]
fn neglog_derivatives(x: f64) -> (f64, f64, f64, f64) {
    let x1 = x.max(1e-300);
    let inv = 1.0 / x1;
    let inv2 = inv * inv;
    (-inv, inv2, -2.0 * inv2 * inv, 6.0 * inv2 * inv2)
}

/// Row-level primary gradient (4-vector) and Hessian (4×4 symmetric)
/// computed entirely from closed-form scalar formulas.
///
/// Returns (nll, gradient[4], hessian[4][4]) on the stack.
#[inline]
fn row_primary_closed_form(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    let (c, c1, c2, ..) = c_derivatives(g);

    // Linear predictors
    let eta0 = q0 * c + g * z;
    let eta1 = q1 * c + g * z;
    let ad1 = qd1 * c;

    if qd1 < derivative_guard {
        return Err(format!(
            "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
        ));
    }

    // ── NLL terms ──
    // Entry survival: -neglogΦ(-η₀) = logΦ(-η₀)
    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    // Exit survival: (1-d)·neglogΦ(-η₁)
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    // Event density: d·logφ(η₁)
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    // Time derivative: d·log(ad1)
    let log_ad1 = ad1.max(1e-300).ln();

    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * log_ad1);

    // ── First and second derivatives of each NLL component ──
    // signed_probit_neglog_derivatives gives derivatives with respect to m for
    // -weight * logΦ(m). Here m = -η, so odd derivatives flip sign when mapped
    // back to derivatives with respect to η.
    // For entry: m = -η₀, weight = -w because the NLL contains +w logΦ(-η₀)
    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w);
    // For exit: m = -η₁, weight = w(1-d)
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d));
    // Event density: -d·logφ(η₁) = d·(η₁²/2 + const).
    // d/dη₁ = d·w·η₁, d²/dη₁² = d·w.
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    // Time derivative: -d·log(ad1).
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    // ── Chain rule to primary space ──
    // η₀ depends on (q₀, g): ∂η₀/∂q₀ = c, ∂η₀/∂g = q₀c₁ + z
    // η₁ depends on (q₁, g): ∂η₁/∂q₁ = c, ∂η₁/∂g = q₁c₁ + z
    // ad1 depends on (qd1, g): ∂ad1/∂qd1 = c, ∂ad1/∂g = qd1·c₁
    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + z;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + z;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    // Combined first derivatives of total NLL:
    // u1 for η₀ terms = -e0_k1 (chain rule through m = -η₀)
    // u1 for η₁ terms = -e1_k1 + phi_u1 (chain rule through m = -η₁)
    // u1 for ad1 term = td_u1 (time derivative)
    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;

    let mut grad = [0.0_f64; N_PRIMARY];
    grad[0] = u1_eta0 * deta0_dq0; // ∂ℓ/∂q₀
    grad[1] = u1_eta1 * deta1_dq1; // ∂ℓ/∂q₁
    grad[2] = u1_ad1 * dad1_dqd1; // ∂ℓ/∂qd₁
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg; // ∂ℓ/∂g

    // Combined second derivatives:
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    // Second mixed derivatives of η w.r.t. primary scalars:
    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    // d²η₀/dg² = q₀·c₂ (z is linear in g, so its second derivative is 0)
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; N_PRIMARY]; N_PRIMARY];

    // (q0, q0)
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    // (q1, q1)
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    // (qd1, qd1)
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    // (q0, q1) = 0 (η₀ and η₁ share no primary scalars except g)
    hess[0][1] = 0.0;
    hess[1][0] = 0.0;
    // (q0, qd1) = 0
    hess[0][2] = 0.0;
    hess[2][0] = 0.0;
    // (q1, qd1) = 0
    hess[1][2] = 0.0;
    hess[2][1] = 0.0;
    // (q0, g) = u2_η₀ · (∂η₀/∂q₀)(∂η₀/∂g) + u1_η₀ · (∂²η₀/∂q₀∂g)
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    // (q1, g)
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    // (qd1, g)
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    // (g, g) = Σ_terms [u2·(dterm/dg)² + u1·(d²term/dg²)]
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg
        + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg
        + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg
        + u1_ad1 * d2ad1_dg2;

    Ok((nll, grad, hess))
}

// ── Eval cache ────────────────────────────────────────────────────────
//
// Third and fourth order contracted derivatives for the outer REML path
// continue to use the MultiDirJet engine via row_neglog_directional.
// That path is called O(n_rho²) times, not O(n × inner_iters) times,
// so the jet overhead is acceptable there.

#[derive(Clone)]
struct RowPrimaryBase {
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

struct EvalCache {
    row_bases: Vec<RowPrimaryBase>,
}

// ── Row-level NLL computation ─────────────────────────────────────────

impl SurvivalMarginalSlopeFamily {
    fn flex_timewiggle_active(&self) -> bool {
        self.time_wiggle_ncols > 0
    }

    fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.design_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        (p_total - p_w)..p_total
    }

    fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
            || basis_d3.ncols() != beta_w.len()
        {
            return Err(format!(
                "survival marginal-slope timewiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                basis.ncols(),
                basis_d1.ncols(),
                basis_d2.ncols(),
                basis_d3.ncols(),
                beta_w.len()
            ));
        }
        let dq_dq0 = basis_d1.dot(&beta_w) + 1.0;
        let d2q_dq02 = basis_d2.dot(&beta_w);
        let d3q_dq03 = basis_d3.dot(&beta_w);
        Ok(Some(SurvivalTimeWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
        }))
    }

    fn row_dynamic_q_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRow, String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        let mut out = SurvivalMarginalSlopeDynamicRow {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(p_time),
            dq1_time: Array1::zeros(p_time),
            dqd1_time: Array1::zeros(p_time),
            dq0_marginal: Array1::zeros(p_marginal),
            dq1_marginal: Array1::zeros(p_marginal),
            dqd1_marginal: Array1::zeros(p_marginal),
            d2q0_time_time: Array2::zeros((p_time, p_time)),
            d2q1_time_time: Array2::zeros((p_time, p_time)),
            d2qd1_time_time: Array2::zeros((p_time, p_time)),
            d2q0_time_marginal: Array2::zeros((p_time, p_marginal)),
            d2q1_time_marginal: Array2::zeros((p_time, p_marginal)),
            d2qd1_time_marginal: Array2::zeros((p_time, p_marginal)),
            d2q0_marginal_marginal: Array2::zeros((p_marginal, p_marginal)),
            d2q1_marginal_marginal: Array2::zeros((p_marginal, p_marginal)),
            d2qd1_marginal_marginal: Array2::zeros((p_marginal, p_marginal)),
        };

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + self.marginal_design.dot_row(row, beta_marginal);
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + self.marginal_design.dot_row(row, beta_marginal);
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self.design_entry.row_chunk(row..row + 1);
            let time_exit_chunk = self.design_exit.row_chunk(row..row + 1);
            let time_deriv_chunk = self.design_derivative_exit.row_chunk(row..row + 1);
            let time_row_entry = time_entry_chunk.row(0).to_owned();
            let time_row_exit = time_exit_chunk.row(0).to_owned();
            let time_row_deriv = time_deriv_chunk.row(0).to_owned();
            out.dq0_time.assign(&time_row_entry.view());
            out.dq1_time.assign(&time_row_exit.view());
            out.dqd1_time.assign(&time_row_deriv.view());
            if p_marginal > 0 {
                let marginal_chunk = self.marginal_design.row_chunk(row..row + 1);
                let marginal_row = marginal_chunk.row(0).to_owned();
                out.dq0_marginal.assign(&marginal_row.view());
                out.dq1_marginal.assign(&marginal_row.view());
            }
            return Ok(out);
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self.design_entry.row_chunk(row..row + 1);
        let exit_chunk = self.design_exit.row_chunk(row..row + 1);
        let deriv_chunk = self.design_derivative_exit.row_chunk(row..row + 1);
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_row = if p_marginal > 0 {
            let marginal_chunk = self.marginal_design.row_chunk(row..row + 1);
            Some(marginal_chunk.row(0).to_owned())
        } else {
            None
        };

        let base_marginal = marginal_row
            .as_ref()
            .map(|row_vec| row_vec.dot(beta_marginal))
            .unwrap_or(0.0);
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
            for k in 0..p_base {
                out.d2q0_time_time[[j, k]] =
                    entry_geom.d2q_dq02[0] * x_entry_base[j] * x_entry_base[k];
                out.d2q1_time_time[[j, k]] =
                    exit_geom.d2q_dq02[0] * x_exit_base[j] * x_exit_base[k];
                out.d2qd1_time_time[[j, k]] =
                    exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j] * x_exit_base[k]
                        + exit_geom.d2q_dq02[0]
                            * (x_exit_base[j] * x_deriv_base[k] + x_deriv_base[j] * x_exit_base[k]);
            }
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
            for j in 0..p_base {
                let q0_tw = entry_geom.basis_d1[[0, local_idx]] * x_entry_base[j];
                let q1_tw = exit_geom.basis_d1[[0, local_idx]] * x_exit_base[j];
                out.d2q0_time_time[[j, coeff_idx]] = q0_tw;
                out.d2q0_time_time[[coeff_idx, j]] = q0_tw;
                out.d2q1_time_time[[j, coeff_idx]] = q1_tw;
                out.d2q1_time_time[[coeff_idx, j]] = q1_tw;
                let qd1_cross = exit_geom.basis_d2[[0, local_idx]] * d_raw * x_exit_base[j]
                    + exit_geom.basis_d1[[0, local_idx]] * x_deriv_base[j];
                out.d2qd1_time_time[[j, coeff_idx]] = qd1_cross;
                out.d2qd1_time_time[[coeff_idx, j]] = qd1_cross;
            }
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
                for k in 0..p_marginal {
                    out.d2q0_marginal_marginal[[j, k]] =
                        entry_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2q1_marginal_marginal[[j, k]] =
                        exit_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2qd1_marginal_marginal[[j, k]] =
                        exit_geom.d3q_dq03[0] * d_raw * marginal_row[j] * marginal_row[k];
                }
                for k in 0..p_base {
                    out.d2q0_time_marginal[[k, j]] =
                        entry_geom.d2q_dq02[0] * x_entry_base[k] * marginal_row[j];
                    out.d2q1_time_marginal[[k, j]] =
                        exit_geom.d2q_dq02[0] * x_exit_base[k] * marginal_row[j];
                    out.d2qd1_time_marginal[[k, j]] =
                        exit_geom.d3q_dq03[0] * d_raw * x_exit_base[k] * marginal_row[j]
                            + exit_geom.d2q_dq02[0] * x_deriv_base[k] * marginal_row[j];
                }
                for local_idx in 0..time_tail.len() {
                    let coeff_idx = time_tail.start + local_idx;
                    out.d2q0_time_marginal[[coeff_idx, j]] =
                        entry_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2q1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2qd1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d2[[0, local_idx]] * d_raw * marginal_row[j];
                }
            }
        }

        Ok(out)
    }

    fn time_derivative_lower_bound(&self) -> f64 {
        assert!(self.derivative_guard.is_finite() && self.derivative_guard > 0.0);
        self.derivative_guard
    }

    fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.time_linear_constraints.as_ref() else {
            return Ok(None);
        };
        if beta.len() != constraints.a.ncols() || delta.len() != constraints.a.ncols() {
            return Err(format!(
                "survival marginal-slope time-step dimension mismatch: beta={}, delta={}, expected {}",
                beta.len(),
                delta.len(),
                constraints.a.ncols()
            ));
        }
        let mut alpha = 1.0f64;
        for row in 0..constraints.a.nrows() {
            let a_row = constraints.a.row(row);
            let slack = a_row.dot(beta) - constraints.b[row];
            if slack < -1e-10 {
                return Err(format!(
                    "survival marginal-slope current time block violates derivative guard at row {row}: slack={slack:.3e}"
                ));
            }
            let drift = a_row.dot(delta);
            if drift < 0.0 {
                alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
            }
        }
        if alpha >= 1.0 {
            Ok(Some(1.0))
        } else {
            Ok(Some((0.995 * alpha).clamp(0.0, 1.0)))
        }
    }

    /// Per-row NLL and its directional derivatives through 4 primary scalars.
    ///
    /// NLL_i = w_i * [ (1-d)·neglogΦ(-η₁) + logΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
    ///
    /// where η = a(t) + β·z, a(t) = q(t)·√(1+β²), β = g.
    ///
    /// block_states[0].eta is from the exit design and is NOT used here;
    /// all 3 time-block linear predictors are recomputed from beta_time
    /// because the time block has 3 design matrices sharing one coefficient vector.
    fn row_neglog_directional(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[Array1<f64>],
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "survival marginal-slope row directional expects 0..=4 directions, got {k}"
            ));
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let zi = self.z[row];

        // Primary scalar jets: q0, q1, qd1, g
        let q0_first: Vec<f64> = dirs.iter().map(|dir| dir[0]).collect();
        let q1_first: Vec<f64> = dirs.iter().map(|dir| dir[1]).collect();
        let qd1_first: Vec<f64> = dirs.iter().map(|dir| dir[2]).collect();
        let g_first: Vec<f64> = dirs.iter().map(|dir| dir[3]).collect();

        // Compute q0, q1, qd1 from the shared time block plus the baseline
        // covariate block. The time block has a single beta vector but 3
        // different design matrices (entry, exit, derivative).
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let q0_val = self.design_entry.dot_row(row, beta_time)
            + self.offset_entry[row]
            + self.marginal_design.dot_row(row, beta_marginal);
        let q1_val = self.design_exit.dot_row(row, beta_time)
            + self.offset_exit[row]
            + self.marginal_design.dot_row(row, beta_marginal);
        let qd1_val =
            self.design_derivative_exit.dot_row(row, beta_time) + self.derivative_offset_exit[row];
        let g_val = block_states[2].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, &q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, &q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, &qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, &g_first);

        // beta = g (signed slope allowed)
        let beta_jet = g_jet.clone();
        // c = sqrt(1 + beta^2)
        let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&beta_jet.mul(&beta_jet));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        // a0 = q0 * c, a1 = q1 * c, ad1 = qd1 * c
        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);

        // eta0 = a0 + beta * z, eta1 = a1 + beta * z
        let z_jet = MultiDirJet::constant(k, zi);
        let eta0_jet = a0_jet.add(&beta_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&beta_jet.mul(&z_jet));

        // NLL_i = w_i * {
        //   (1-d_i) * neglogphi(-eta1)  [exit survival for censored]
        //   + log Phi(-eta0)            [entry survival from left truncation]
        //   - d_i * log_phi(eta1)        [event log-density of normal]
        //   - d_i * log(ad1)             [event log time-derivative]
        // }

        // Entry survival term: +log Phi(-eta0) = log S(t0|z)
        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0); // note: -w * neglogphi(-eta0) = w * log Phi(-eta0)

        // Exit survival term: (1-d)*neglogphi(-eta1) = -(1-d)*log Phi(-eta1)
        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        // Event density: -d * log phi(eta1)
        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        // Time derivative: -d * log(ad1)
        // The model domain is enforced on qd1 itself, which is the same quantity
        // constrained by the time monotonicity inequalities. Since c = sqrt(1+beta^2)
        // is strictly positive, qd1 >= guard implies ad1 > 0 as required by log(ad1).
        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if qd1_val < qd1_lower {
            return Err(format!(
                "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
            ));
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        if k == 0 {
            Ok(total.coeff(0))
        } else {
            Ok(total.coeff(total.full_mask()))
        }
    }

    /// Compute per-row primary gradient and Hessian using the closed-form
    /// scalar kernel.  The hot inner computation uses stack arrays only;
    /// conversion to Array1/Array2 happens once at the boundary for API
    /// compatibility with outer-derivative paths.
    fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let g = block_states[2].eta[row];
        let (nll, grad_arr, hess_arr) = row_primary_closed_form(
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            g,
            self.z[row],
            self.weights[row],
            self.event[row],
            self.derivative_guard,
        )?;
        // Convert stack arrays to ndarray types at the boundary.
        let grad = Array1::from_vec(grad_arr.to_vec());
        let mut hess = Array2::zeros((N_PRIMARY, N_PRIMARY));
        for i in 0..N_PRIMARY {
            for j in 0..N_PRIMARY {
                hess[[i, j]] = hess_arr[i][j];
            }
        }
        Ok((nll, grad, hess))
    }

    fn build_eval_cache(&self, block_states: &[ParameterBlockState]) -> Result<EvalCache, String> {
        let row_bases = (0..self.n)
            .map(|row| {
                let (_, gradient, hessian) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                Ok(RowPrimaryBase { gradient, hessian })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(EvalCache { row_bases })
    }

    fn row_primary_gradient_hessian<'a>(
        &self,
        row: usize,
        cache: &'a EvalCache,
    ) -> (&'a Array1<f64>, &'a Array2<f64>) {
        let base = &cache.row_bases[row];
        (&base.gradient, &base.hessian)
    }

    fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value = self.row_neglog_directional(
                    row,
                    block_states,
                    &[da.clone(), db.clone(), dir.clone()],
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            let da = unit_primary_direction(a);
            for b in a..N_PRIMARY {
                let db = unit_primary_direction(b);
                let value = self.row_neglog_directional(
                    row,
                    block_states,
                    &[da.clone(), db.clone(), dir_u.clone(), dir_v.clone()],
                )?;
                out[[a, b]] = value;
                out[[b, a]] = value;
            }
        }
        Ok(out)
    }

    // ── Pullback through design matrices ──────────────────────────────

    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    ///
    /// Writes directly into `target` subslices via sparse-aware row primitives —
    /// no dense row buffers or temporary blocks are allocated.
    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];

        // time-time block: Σ_{a,b} H[a,b] * time_a_row ⊗ time_b_row
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(
                        row,
                        time_designs[b],
                        alpha,
                        target.slice_mut(s![slices.time.clone(), slices.time.clone()]),
                    )
                    .expect("time block row_outer_into dimension mismatch");
            }
        }

        // marginal-marginal block: (H[0,0]+H[0,1]+H[1,0]+H[1,1]) * m_row ⊗ m_row
        self.marginal_design
            .syr_row_into_view(
                row,
                h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]],
                target.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()]),
            )
            .expect("marginal syr_row_into dimension mismatch");

        // logslope-logslope block: H[3,3] * g_row ⊗ g_row
        self.logslope_design
            .syr_row_into_view(
                row,
                h[[3, 3]],
                target.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()]),
            )
            .expect("logslope syr_row_into dimension mismatch");

        // marginal-logslope block: (H[0,3]+H[1,3]) * m_row ⊗ g_row  (+ transpose)
        {
            let alpha_mg = h[[0, 3]] + h[[1, 3]];
            if alpha_mg != 0.0 {
                self.marginal_design
                    .row_outer_into_view(
                        row,
                        &self.logslope_design,
                        alpha_mg,
                        target.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()]),
                    )
                    .expect("marginal-logslope row_outer_into dimension mismatch");
                self.logslope_design
                    .row_outer_into_view(
                        row,
                        &self.marginal_design,
                        alpha_mg,
                        target.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()]),
                    )
                    .expect("logslope-marginal row_outer_into dimension mismatch");
            }
        }

        // time-logslope block: H[a,3] * time_a_row ⊗ g_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 3]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.logslope_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.logslope.clone()]),
                )
                .expect("time-logslope row_outer_into dimension mismatch");
            self.logslope_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.logslope.clone(), slices.time.clone()]),
                )
                .expect("logslope-time row_outer_into dimension mismatch");
        }

        // time-marginal block: (H[a,0]+H[a,1]) * time_a_row ⊗ m_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 0]] + h[[a, 1]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.marginal_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.marginal.clone()]),
                )
                .expect("time-marginal row_outer_into dimension mismatch");
            self.marginal_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.marginal.clone(), slices.time.clone()]),
                )
                .expect("marginal-time row_outer_into dimension mismatch");
        }
    }

    /// Map a coefficient-space direction to primary-space for a given row.
    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(N_PRIMARY);
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        out[0] = self.design_entry.dot_row_view(row, d_time);
        out[0] += self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[1] = self.design_exit.dot_row_view(row, d_time);
        out[1] += self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[2] = self.design_derivative_exit.dot_row_view(row, d_time);
        out[3] = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
        out
    }

    // ── Psi (spatial length-scale) derivatives ────────────────────────

    fn resolve_psi_location(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Option<(usize, usize)> {
        let mut cursor = 0usize;
        for (block_idx, block) in derivative_blocks.iter().enumerate() {
            if psi_index < cursor + block.len() {
                return Some((block_idx, psi_index - cursor));
            }
            cursor += block.len();
        }
        None
    }

    fn psi_design_row_vector(
        &self,
        row: usize,
        deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            total_rows,
            p,
            0..total_rows,
            label,
        )
        .ok();
        first_psi_linear_map(action.as_ref(), &deriv.x_psi, total_rows, p).row_vector(row)
    }

    fn psi_second_design_row_vector(
        &self,
        row: usize,
        deriv_i: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        deriv_j: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        local_j: usize,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            total_rows,
            p,
            0..total_rows,
            label,
        )?;
        let dense = deriv_i
            .x_psi_psi
            .as_ref()
            .and_then(|rows| rows.get(local_j));
        second_psi_linear_map(action.as_ref(), dense, total_rows, p).row_vector(row)
    }

    // ── Psi terms (first and second order) ────────────────────────────
    //
    // All three psi methods (first-order, second-order, directional derivative)
    // use block-local accumulation via BlockHessianAccumulator. Per-row work is
    // O(max(p_block²)) instead of O(p²), eliminating the dense p×p bottleneck
    // that breaks multi-axis Duchon / per-axis length scaling.

    /// Resolve psi block info: (block_idx, local_idx, p_block, label).
    fn psi_block_info(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, usize, usize, &'static str)>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        match block_idx {
            1 => Ok(Some((
                block_idx,
                local_idx,
                self.marginal_design.ncols(),
                "SurvivalMarginalSlope marginal",
            ))),
            2 => Ok(Some((
                block_idx,
                local_idx,
                self.logslope_design.ncols(),
                "SurvivalMarginalSlope logslope",
            ))),
            _ => Err(format!(
                "survival marginal-slope psi: only baseline/slope spatial blocks are supported, got block {block_idx}"
            )),
        }
    }

    /// Accumulate block-local score from a primary-space vector (replaces
    /// pullback_primary_vector + score += for score accumulation).
    fn accumulate_score_blockwise(
        &self,
        row: usize,
        primary: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
        score_g: &mut Array1<f64>,
    ) -> Result<(), String> {
        {
            let mut st = score_t.view_mut();
            self.design_entry
                .axpy_row_into(row, primary[0], &mut st)
                .expect("time entry axpy dim mismatch");
            self.design_exit
                .axpy_row_into(row, primary[1], &mut st)
                .expect("time exit axpy dim mismatch");
            self.design_derivative_exit
                .axpy_row_into(row, primary[2], &mut st)
                .expect("time deriv axpy dim mismatch");
        }
        self.marginal_design.axpy_row_into(
            row,
            primary[0] + primary[1],
            &mut score_m.view_mut(),
        )?;
        self.logslope_design
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    fn psi_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = spatial_block_primary_loading(block_idx)?;
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        // Parallel accumulation: each worker gets its own block-local accumulators.
        type Acc = (
            f64,                     // objective_psi
            Array1<f64>,             // score_t
            Array1<f64>,             // score_m
            Array1<f64>,             // score_g
            BlockHessianAccumulator, // Hessian blocks
        );
        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                BlockHessianAccumulator::new(p_t, p_m, p_g),
            )
        };

        let (objective_psi, score_t, score_m, score_g, acc) = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut a, row| -> Result<Acc, String> {
                // Compute psi design row once; derive direction from it.
                let psi_row = self.psi_design_row_vector(row, deriv, self.n, p_psi, psi_label)?;
                let dir = primary_direction_from_psi_row(block_idx, &psi_row, beta_psi);

                let (f_pi, f_pipi) = if let Some(c) = cache {
                    let (g, h) = self.row_primary_gradient_hessian(row, c);
                    (g.clone(), h.clone())
                } else {
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    (g, h)
                };
                let third = self.row_primary_third_contracted(row, block_states, &dir)?;

                a.0 += f_pi.dot(&dir);

                // Score: psi block gets scalar × psi_row
                let s1 = f_pi.dot(&loading);
                match block_idx {
                    1 => a.2.scaled_add(s1, &psi_row),
                    _ => a.3.scaled_add(s1, &psi_row),
                }
                // Score: pullback of f_pipi·dir into all 3 blocks
                let pb = f_pipi.dot(&dir);
                self.accumulate_score_blockwise(row, &pb, &mut a.1, &mut a.2, &mut a.3)?;

                // Hessian: rank-1 from psi_row × pullback(f_pipi·loading)
                let right_primary = f_pipi.dot(&loading);
                a.4.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary);
                // Hessian: pullback of third derivative
                a.4.add_pullback(self, row, &third);

                Ok(a)
            })
            .try_reduce(make_acc, |mut a, b| {
                a.0 += b.0;
                a.1 += &b.1;
                a.2 += &b.2;
                a.3 += &b.3;
                a.4.add(&b.4);
                Ok(a)
            })?;

        // Assemble score into flat vector
        let mut score_psi = Array1::zeros(slices.total);
        score_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Box::new(acc.into_operator(slices))),
        }))
    }

    fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner(block_states, derivative_blocks, psi_index, None)
    }

    fn psi_second_order_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx_i, local_idx_i, p_psi_i, label_i)) =
            self.psi_block_info(derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_idx_j, local_idx_j, p_psi_j, label_j)) =
            self.psi_block_info(derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let deriv_i = &derivative_blocks[block_idx_i][local_idx_i];
        let deriv_j = &derivative_blocks[block_idx_j][local_idx_j];
        let loading_i = spatial_block_primary_loading(block_idx_i)?;
        let loading_j = spatial_block_primary_loading(block_idx_j)?;
        let beta_i = match block_idx_i {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let beta_j = match block_idx_j {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let same_block = block_idx_i == block_idx_j;

        type Acc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            BlockHessianAccumulator,
        );
        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                BlockHessianAccumulator::new(p_t, p_m, p_g),
            )
        };

        let (objective_psi_psi, score_t, score_m, score_g, acc) = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut a, row| -> Result<Acc, String> {
                // Compute psi design rows once; derive directions from them.
                let psi_row_i =
                    self.psi_design_row_vector(row, deriv_i, self.n, p_psi_i, label_i)?;
                let psi_row_j =
                    self.psi_design_row_vector(row, deriv_j, self.n, p_psi_j, label_j)?;
                let dir_i = primary_direction_from_psi_row(block_idx_i, &psi_row_i, beta_i);
                let dir_j = primary_direction_from_psi_row(block_idx_j, &psi_row_j, beta_j);

                let (psi_row_ij, dir_ij) = if same_block {
                    let r = self.psi_second_design_row_vector(
                        row,
                        deriv_i,
                        deriv_j,
                        local_idx_j,
                        self.n,
                        p_psi_i,
                        label_i,
                    )?;
                    let d = primary_second_direction_from_psi_row(block_idx_i, &r, beta_i);
                    (Some(r), d)
                } else {
                    (None, Array1::<f64>::zeros(N_PRIMARY))
                };
                let has_ij = psi_row_ij
                    .as_ref()
                    .is_some_and(|r| r.iter().any(|v| v.abs() > 0.0));

                let (f_pi, f_pipi) = if let Some(c) = cache {
                    let (g, h) = self.row_primary_gradient_hessian(row, c);
                    (g.clone(), h.clone())
                } else {
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    (g, h)
                };
                let third_i = self.row_primary_third_contracted(row, block_states, &dir_i)?;
                let third_j = self.row_primary_third_contracted(row, block_states, &dir_j)?;
                let fourth =
                    self.row_primary_fourth_contracted(row, block_states, &dir_i, &dir_j)?;

                a.0 += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);

                // Score
                if has_ij {
                    let s_ij = f_pi.dot(&loading_i);
                    let psi_ij = psi_row_ij.as_ref().unwrap();
                    match block_idx_i {
                        1 => a.2.scaled_add(s_ij, psi_ij),
                        _ => a.3.scaled_add(s_ij, psi_ij),
                    }
                }
                let s_i = loading_i.dot(&f_pipi.dot(&dir_j));
                match block_idx_i {
                    1 => a.2.scaled_add(s_i, &psi_row_i),
                    _ => a.3.scaled_add(s_i, &psi_row_i),
                }
                let s_j = loading_j.dot(&f_pipi.dot(&dir_i));
                match block_idx_j {
                    1 => a.2.scaled_add(s_j, &psi_row_j),
                    _ => a.3.scaled_add(s_j, &psi_row_j),
                }
                let pb1 = f_pipi.dot(&dir_ij);
                self.accumulate_score_blockwise(row, &pb1, &mut a.1, &mut a.2, &mut a.3)?;
                let pb2 = third_i.dot(&dir_j);
                self.accumulate_score_blockwise(row, &pb2, &mut a.1, &mut a.2, &mut a.3)?;

                // Hessian
                if has_ij {
                    let rp_ij = f_pipi.dot(&loading_i);
                    a.4.add_rank1_psi_cross(
                        self,
                        row,
                        block_idx_i,
                        psi_row_ij.as_ref().unwrap(),
                        &rp_ij,
                    );
                }
                let scalar_ij = loading_i.dot(&f_pipi.dot(&loading_j));
                a.4.add_psi_psi_outer(block_idx_i, &psi_row_i, block_idx_j, &psi_row_j, scalar_ij);
                let rp_i = third_j.t().dot(&loading_i);
                a.4.add_rank1_psi_cross(self, row, block_idx_i, &psi_row_i, &rp_i);
                let rp_j = third_i.t().dot(&loading_j);
                a.4.add_rank1_psi_cross(self, row, block_idx_j, &psi_row_j, &rp_j);
                a.4.add_pullback(self, row, &fourth);
                let third_ij = self.row_primary_third_contracted(row, block_states, &dir_ij)?;
                a.4.add_pullback(self, row, &third_ij);

                Ok(a)
            })
            .try_reduce(make_acc, |mut a, b| {
                a.0 += b.0;
                a.1 += &b.1;
                a.2 += &b.2;
                a.3 += &b.3;
                a.4.add(&b.4);
                Ok(a)
            })?;

        let mut score_psi_psi = Array1::zeros(slices.total);
        score_psi_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(acc.into_operator(slices))),
        }))
    }

    fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner(block_states, derivative_blocks, psi_i, psi_j, None)
    }

    fn psi_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = spatial_block_primary_loading(block_idx)?;
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let d_beta_block = match block_idx {
            1 => d_beta_flat.slice(s![slices.marginal.clone()]),
            _ => d_beta_flat.slice(s![slices.logslope.clone()]),
        };

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let acc = (0..self.n)
            .into_par_iter()
            .try_fold(
                || BlockHessianAccumulator::new(p_t, p_m, p_g),
                |mut acc, row| -> Result<BlockHessianAccumulator, String> {
                    let psi_row =
                        self.psi_design_row_vector(row, deriv, self.n, p_psi, psi_label)?;
                    let psi_dir = primary_direction_from_psi_row(block_idx, &psi_row, beta_psi);
                    let psi_action =
                        primary_psi_action_from_psi_row(block_idx, &psi_row, d_beta_block);
                    let row_dir = self.row_primary_direction_from_flat(row, &slices, d_beta_flat);
                    let third_beta =
                        self.row_primary_third_contracted(row, block_states, &row_dir)?;
                    let fourth =
                        self.row_primary_fourth_contracted(row, block_states, &row_dir, &psi_dir)?;

                    let right_primary = third_beta.t().dot(&loading);
                    acc.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary);
                    acc.add_pullback(self, row, &fourth);
                    let third_action =
                        self.row_primary_third_contracted(row, block_states, &psi_action)?;
                    acc.add_pullback(self, row, &third_action);
                    Ok(acc)
                },
            )
            .try_reduce(
                || BlockHessianAccumulator::new(p_t, p_m, p_g),
                |mut a, b| {
                    a.add(&b);
                    Ok(a)
                },
            )?;

        Ok(Some(acc.to_dense(&slices)))
    }
}

// ── Workspace structs ─────────────────────────────────────────────────

struct SurvivalMarginalSlopePsiWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: EvalCache,
}

impl SurvivalMarginalSlopePsiWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let cache = family.build_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            cache,
        })
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalMarginalSlopePsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.psi_terms_inner(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            Some(&self.cache),
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.family.psi_second_order_terms_inner(
            &self.block_states,
            &self.derivative_blocks,
            psi_i,
            psi_j,
            Some(&self.cache),
        )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family.psi_hessian_directional_derivative(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }
}

// ── RowKernel<4> implementation ───────────────────────────────────────

struct SurvivalMarginalSlopeRowKernel {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
}

impl SurvivalMarginalSlopeRowKernel {
    fn new(family: SurvivalMarginalSlopeFamily, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = block_slices(&block_states);
        Self {
            family,
            block_states,
            slices,
        }
    }
}

impl RowKernel<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
        let beta_time = &self.block_states[0].beta;
        let beta_marginal = &self.block_states[1].beta;
        let q0 = self.family.design_entry.dot_row(row, beta_time)
            + self.family.offset_entry[row]
            + self.family.marginal_design.dot_row(row, beta_marginal);
        let q1 = self.family.design_exit.dot_row(row, beta_time)
            + self.family.offset_exit[row]
            + self.family.marginal_design.dot_row(row, beta_marginal);
        let qd1 = self.family.design_derivative_exit.dot_row(row, beta_time)
            + self.family.derivative_offset_exit[row];
        let g = self.block_states[2].eta[row];
        row_primary_closed_form(
            q0,
            q1,
            qd1,
            g,
            self.family.z[row],
            self.family.weights[row],
            self.family.event[row],
            self.family.derivative_guard,
        )
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_logslope = d_beta.slice(s![self.slices.logslope.clone()]);
        [
            self.family.design_entry.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_exit.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_derivative_exit.dot_row_view(row, d_time),
            self.family.logslope_design.dot_row_view(row, d_logslope),
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[2], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0] + v[1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut logslope = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[3], &mut logslope)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (0, &self.family.design_entry),
            (1, &self.family.design_exit),
            (2, &self.family.design_derivative_exit),
        ];
        for &(pi, des) in &designs {
            {
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.squared_axpy_row_into(row, h[pi][pi], &mut td)
                    .expect("time squared_axpy dim mismatch");
            }
            for &(pj, des_j) in &designs {
                if pj <= pi {
                    continue;
                }
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.crossdiag_axpy_row_into(row, des_j, 2.0 * h[pi][pj], &mut td)
                    .expect("time crossdiag dim mismatch");
            }
        }
        {
            let alpha = h[0][0] + 2.0 * h[0][1] + h[1][1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[3][3], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 4]) -> Result<[[f64; 4]; 4], String> {
        let dir_arr = Array1::from_vec(dir.to_vec());
        let out = self
            .family
            .row_primary_third_contracted(row, &self.block_states, &dir_arr)?;
        let mut r = [[0.0; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                r[a][b] = out[[a, b]];
            }
        }
        Ok(r)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        let u = Array1::from_vec(dir_u.to_vec());
        let v = Array1::from_vec(dir_v.to_vec());
        let out = self
            .family
            .row_primary_fourth_contracted(row, &self.block_states, &u, &v)?;
        let mut r = [[0.0; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                r[a][b] = out[[a, b]];
            }
        }
        Ok(r)
    }
}

impl SurvivalMarginalSlopeFamily {
    fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.flex_timewiggle_active() {
            return self.evaluate_blockwise_exact_newton_timewiggle_dense(block_states);
        }
        // Detect sparse designs → sparse Hessian path (O(nnz) memory per
        // worker instead of O(p²), sparse Cholesky downstream).
        let time_csrs = match (
            self.design_entry.as_sparse(),
            self.design_exit.as_sparse(),
            self.design_derivative_exit.as_sparse(),
        ) {
            (Some(e), Some(x), Some(d)) => Some((
                e.to_csr_arc().expect("entry CSR"),
                x.to_csr_arc().expect("exit CSR"),
                d.to_csr_arc().expect("deriv CSR"),
            )),
            _ => None,
        };
        let marginal_csr = self
            .marginal_design
            .as_sparse()
            .and_then(|s| s.to_csr_arc());
        let logslope_csr = self
            .logslope_design
            .as_sparse()
            .and_then(|s| s.to_csr_arc());

        let time_sparse = time_csrs.is_some();
        let marginal_sparse = marginal_csr.is_some();
        let logslope_sparse = logslope_csr.is_some();

        if time_sparse && marginal_sparse && logslope_sparse {
            self.evaluate_blockwise_exact_newton_sparse(
                block_states,
                &time_csrs.unwrap(),
                &marginal_csr.unwrap(),
                &logslope_csr.unwrap(),
            )
        } else if !time_sparse && !marginal_sparse && !logslope_sparse {
            self.evaluate_blockwise_exact_newton_dense(block_states)
        } else {
            self.evaluate_blockwise_exact_newton_mixed(
                block_states,
                time_csrs.as_ref(),
                marginal_csr.as_ref(),
                logslope_csr.as_ref(),
            )
        }
    }

    fn evaluate_blockwise_exact_newton_timewiggle_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let slices = block_slices(block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        type Acc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
        );

        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                Array2::zeros((p_t, p_t)),
                Array2::zeros((p_m, p_m)),
                Array2::zeros((p_g, p_g)),
            )
        };

        let (ll, grad_time, grad_marginal, grad_logslope, hess_time, hess_marginal, hess_logslope) =
            (0..self.n)
                .into_par_iter()
                .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    acc.1 -= &(f_pi[0] * &q_geom.dq0_time
                        + f_pi[1] * &q_geom.dq1_time
                        + f_pi[2] * &q_geom.dqd1_time);
                    if p_m > 0 {
                        acc.2 -= &(f_pi[0] * &q_geom.dq0_marginal
                            + f_pi[1] * &q_geom.dq1_marginal
                            + f_pi[2] * &q_geom.dqd1_marginal);
                    }
                    self.logslope_design
                        .axpy_row_into(row, -f_pi[3], &mut acc.3.view_mut())
                        .expect("survival logslope block axpy should match block dimensions");

                    for a in 0..p_t {
                        for b in 0..p_t {
                            acc.4[[a, b]] +=
                                f_pipi[[0, 0]] * q_geom.dq0_time[a] * q_geom.dq0_time[b]
                                    + f_pipi[[0, 1]] * q_geom.dq0_time[a] * q_geom.dq1_time[b]
                                    + f_pipi[[0, 2]] * q_geom.dq0_time[a] * q_geom.dqd1_time[b]
                                    + f_pipi[[1, 0]] * q_geom.dq1_time[a] * q_geom.dq0_time[b]
                                    + f_pipi[[1, 1]] * q_geom.dq1_time[a] * q_geom.dq1_time[b]
                                    + f_pipi[[1, 2]] * q_geom.dq1_time[a] * q_geom.dqd1_time[b]
                                    + f_pipi[[2, 0]] * q_geom.dqd1_time[a] * q_geom.dq0_time[b]
                                    + f_pipi[[2, 1]] * q_geom.dqd1_time[a] * q_geom.dq1_time[b]
                                    + f_pipi[[2, 2]] * q_geom.dqd1_time[a] * q_geom.dqd1_time[b]
                                    + f_pi[0] * q_geom.d2q0_time_time[[a, b]]
                                    + f_pi[1] * q_geom.d2q1_time_time[[a, b]]
                                    + f_pi[2] * q_geom.d2qd1_time_time[[a, b]];
                        }
                    }
                    for a in 0..p_m {
                        for b in 0..p_m {
                            acc.5[[a, b]] += f_pipi[[0, 0]]
                                * q_geom.dq0_marginal[a]
                                * q_geom.dq0_marginal[b]
                                + f_pipi[[0, 1]] * q_geom.dq0_marginal[a] * q_geom.dq1_marginal[b]
                                + f_pipi[[0, 2]] * q_geom.dq0_marginal[a] * q_geom.dqd1_marginal[b]
                                + f_pipi[[1, 0]] * q_geom.dq1_marginal[a] * q_geom.dq0_marginal[b]
                                + f_pipi[[1, 1]] * q_geom.dq1_marginal[a] * q_geom.dq1_marginal[b]
                                + f_pipi[[1, 2]] * q_geom.dq1_marginal[a] * q_geom.dqd1_marginal[b]
                                + f_pipi[[2, 0]] * q_geom.dqd1_marginal[a] * q_geom.dq0_marginal[b]
                                + f_pipi[[2, 1]] * q_geom.dqd1_marginal[a] * q_geom.dq1_marginal[b]
                                + f_pipi[[2, 2]]
                                    * q_geom.dqd1_marginal[a]
                                    * q_geom.dqd1_marginal[b]
                                + f_pi[0] * q_geom.d2q0_marginal_marginal[[a, b]]
                                + f_pi[1] * q_geom.d2q1_marginal_marginal[[a, b]]
                                + f_pi[2] * q_geom.d2qd1_marginal_marginal[[a, b]];
                        }
                    }
                    self.logslope_design
                        .syr_row_into(row, f_pipi[[3, 3]], &mut acc.6)
                        .expect("survival logslope block syr should match block dimensions");

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4 += &b.4;
                    a.5 += &b.5;
                    a.6 += &b.6;
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Dense(hess_time),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: SymmetricMatrix::Dense(hess_marginal),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: SymmetricMatrix::Dense(hess_logslope),
                },
            ],
        })
    }

    fn evaluate_blockwise_exact_newton_mixed(
        &self,
        block_states: &[ParameterBlockState],
        time_csrs: Option<&(
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
        )>,
        marginal_csr: Option<&Arc<faer::sparse::SparseRowMat<usize, f64>>>,
        logslope_csr: Option<&Arc<faer::sparse::SparseRowMat<usize, f64>>>,
    ) -> Result<FamilyEvaluation, String> {
        use crate::matrix::SparseHessianAccumulator;

        enum BlockwiseHessianAccumulator {
            Dense(Array2<f64>),
            Sparse(SparseHessianAccumulator),
        }

        impl BlockwiseHessianAccumulator {
            fn add_assign(&mut self, other: &Self) {
                match (self, other) {
                    (Self::Dense(lhs), Self::Dense(rhs)) => *lhs += rhs,
                    (Self::Sparse(lhs), Self::Sparse(rhs)) => lhs.add_values(&rhs.values),
                    _ => panic!("blockwise Hessian accumulator kind mismatch"),
                }
            }

            fn into_symmetric(self) -> SymmetricMatrix {
                match self {
                    Self::Dense(mat) => SymmetricMatrix::Dense(mat),
                    Self::Sparse(acc) => SymmetricMatrix::Sparse(acc.into_sparse_col_mat()),
                }
            }
        }

        let slices = block_slices(block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let time_pattern = time_csrs.map(|(entry, exit, deriv)| {
            SparseHessianAccumulator::from_multi_csr(
                &[entry.as_ref(), exit.as_ref(), deriv.as_ref()],
                p_t,
            )
        });
        let marginal_pattern =
            marginal_csr.map(|csr| SparseHessianAccumulator::from_single_csr(csr.as_ref(), p_m));
        let logslope_pattern =
            logslope_csr.map(|csr| SparseHessianAccumulator::from_single_csr(csr.as_ref(), p_g));

        let e_sparse = time_csrs.map(|(entry, _, _)| {
            let sym = entry.symbolic();
            (sym.row_ptr(), sym.col_idx(), entry.val())
        });
        let x_sparse = time_csrs.map(|(_, exit, _)| {
            let sym = exit.symbolic();
            (sym.row_ptr(), sym.col_idx(), exit.val())
        });
        let d_sparse = time_csrs.map(|(_, _, deriv)| {
            let sym = deriv.symbolic();
            (sym.row_ptr(), sym.col_idx(), deriv.val())
        });
        let m_sparse = marginal_csr.map(|csr| {
            let sym = csr.symbolic();
            (sym.row_ptr(), sym.col_idx(), csr.val())
        });
        let g_sparse = logslope_csr.map(|csr| {
            let sym = csr.symbolic();
            (sym.row_ptr(), sym.col_idx(), csr.val())
        });

        type MixedAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            BlockwiseHessianAccumulator,
            BlockwiseHessianAccumulator,
            BlockwiseHessianAccumulator,
        );

        let make_acc = || -> MixedAcc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                time_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_t, p_t))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
                marginal_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_m, p_m))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
                logslope_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_g, p_g))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
            )
        };

        let (ll, grad_time, grad_marginal, grad_logslope, hess_time, hess_marginal, hess_logslope) =
            (0..self.n)
                .into_par_iter()
                .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    match &e_sparse {
                        Some((e_rp, e_ci, e_v)) => {
                            let gt = &mut acc.1;
                            for p in e_rp[row]..e_rp[row + 1] {
                                gt[e_ci[p]] -= f_pi[0] * e_v[p];
                            }
                            let (x_rp, x_ci, x_v) = x_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for exit design");
                            for p in x_rp[row]..x_rp[row + 1] {
                                gt[x_ci[p]] -= f_pi[1] * x_v[p];
                            }
                            let (d_rp, d_ci, d_v) = d_sparse.as_ref().expect(
                                "time sparse metadata should be present for derivative design",
                            );
                            for p in d_rp[row]..d_rp[row + 1] {
                                gt[d_ci[p]] -= f_pi[2] * d_v[p];
                            }
                        }
                        None => {
                            let mut time = acc.1.view_mut();
                            self.design_entry
                                .axpy_row_into(row, -f_pi[0], &mut time)
                                .expect("time entry axpy dim mismatch");
                            self.design_exit
                                .axpy_row_into(row, -f_pi[1], &mut time)
                                .expect("time exit axpy dim mismatch");
                            self.design_derivative_exit
                                .axpy_row_into(row, -f_pi[2], &mut time)
                                .expect("time deriv axpy dim mismatch");
                        }
                    }

                    match &m_sparse {
                        Some((m_rp, m_ci, m_v)) => {
                            let gm = &mut acc.2;
                            let alpha_m = -(f_pi[0] + f_pi[1]);
                            for p in m_rp[row]..m_rp[row + 1] {
                                gm[m_ci[p]] += alpha_m * m_v[p];
                            }
                        }
                        None => {
                            self.marginal_design
                                .axpy_row_into(row, -(f_pi[0] + f_pi[1]), &mut acc.2.view_mut())
                                .expect(
                                    "survival marginal block axpy should match block dimensions",
                                );
                        }
                    }

                    match &g_sparse {
                        Some((g_rp, g_ci, g_v)) => {
                            let gg = &mut acc.3;
                            for p in g_rp[row]..g_rp[row + 1] {
                                gg[g_ci[p]] -= f_pi[3] * g_v[p];
                            }
                        }
                        None => {
                            self.logslope_design
                                .axpy_row_into(row, -f_pi[3], &mut acc.3.view_mut())
                                .expect(
                                    "survival logslope block axpy should match block dimensions",
                                );
                        }
                    }

                    match &mut acc.4 {
                        BlockwiseHessianAccumulator::Dense(hess_time) => {
                            let designs = [
                                &self.design_entry,
                                &self.design_exit,
                                &self.design_derivative_exit,
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    designs[a]
                                        .row_outer_into(row, designs[b], f_pipi[[a, b]], hess_time)
                                        .expect("time row_outer_into dim mismatch");
                                }
                            }
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_time) => {
                            let (e_rp, e_ci, e_v) = e_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for entry design");
                            let (x_rp, x_ci, x_v) = x_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for exit design");
                            let (d_rp, d_ci, d_v) = d_sparse.as_ref().expect(
                                "time sparse metadata should be present for derivative design",
                            );
                            let row_slices: [(std::ops::Range<usize>, &[usize], &[f64]); 3] = [
                                (e_rp[row]..e_rp[row + 1], e_ci, e_v),
                                (x_rp[row]..x_rp[row + 1], x_ci, x_v),
                                (d_rp[row]..d_rp[row + 1], d_ci, d_v),
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let alpha = f_pipi[[a, b]];
                                    if alpha == 0.0 {
                                        continue;
                                    }
                                    let (ref ra, cia, va) = row_slices[a];
                                    let (ref rb, cib, vb) = row_slices[b];
                                    for pi in ra.clone() {
                                        let ca = cia[pi];
                                        let xia = va[pi] * alpha;
                                        for pj in rb.clone() {
                                            hess_time.add_upper(ca, cib[pj], xia * vb[pj]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let alpha_m = f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    match &mut acc.5 {
                        BlockwiseHessianAccumulator::Dense(hess_marginal) => {
                            self.marginal_design
                                .syr_row_into(row, alpha_m, hess_marginal)
                                .expect(
                                    "survival marginal block syr should match block dimensions",
                                );
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_marginal) => {
                            if alpha_m != 0.0 {
                                let (m_rp, m_ci, m_v) = m_sparse.as_ref().expect(
                                    "marginal sparse metadata should be present for sparse block",
                                );
                                for pi in m_rp[row]..m_rp[row + 1] {
                                    let ca = m_ci[pi];
                                    let xia = m_v[pi] * alpha_m;
                                    for pj in m_rp[row]..m_rp[row + 1] {
                                        hess_marginal.add_upper(ca, m_ci[pj], xia * m_v[pj]);
                                    }
                                }
                            }
                        }
                    }

                    let alpha_g = f_pipi[[3, 3]];
                    match &mut acc.6 {
                        BlockwiseHessianAccumulator::Dense(hess_logslope) => {
                            self.logslope_design
                                .syr_row_into(row, alpha_g, hess_logslope)
                                .expect(
                                    "survival logslope block syr should match block dimensions",
                                );
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_logslope) => {
                            if alpha_g != 0.0 {
                                let (g_rp, g_ci, g_v) = g_sparse.as_ref().expect(
                                    "logslope sparse metadata should be present for sparse block",
                                );
                                for pi in g_rp[row]..g_rp[row + 1] {
                                    let ca = g_ci[pi];
                                    let xia = g_v[pi] * alpha_g;
                                    for pj in g_rp[row]..g_rp[row + 1] {
                                        hess_logslope.add_upper(ca, g_ci[pj], xia * g_v[pj]);
                                    }
                                }
                            }
                        }
                    }

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4.add_assign(&b.4);
                    a.5.add_assign(&b.5);
                    a.6.add_assign(&b.6);
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: hess_time.into_symmetric(),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: hess_marginal.into_symmetric(),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: hess_logslope.into_symmetric(),
                },
            ],
        })
    }

    // ── Dense path (original) ────────────────────────────────────────

    fn evaluate_blockwise_exact_newton_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let slices = block_slices(block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        type Acc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
        );

        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                Array2::zeros((p_t, p_t)),
                Array2::zeros((p_m, p_m)),
                Array2::zeros((p_g, p_g)),
            )
        };
        let (ll, grad_time, grad_marginal, grad_logslope, hess_time, hess_marginal, hess_logslope) =
            (0..self.n)
                .into_par_iter()
                .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    {
                        let mut time = acc.1.view_mut();
                        self.design_entry
                            .axpy_row_into(row, -f_pi[0], &mut time)
                            .expect("time entry axpy dim mismatch");
                        self.design_exit
                            .axpy_row_into(row, -f_pi[1], &mut time)
                            .expect("time exit axpy dim mismatch");
                        self.design_derivative_exit
                            .axpy_row_into(row, -f_pi[2], &mut time)
                            .expect("time deriv axpy dim mismatch");
                    }
                    self.marginal_design
                        .axpy_row_into(row, -(f_pi[0] + f_pi[1]), &mut acc.2.view_mut())
                        .expect("survival marginal block axpy should match block dimensions");
                    self.logslope_design
                        .axpy_row_into(row, -f_pi[3], &mut acc.3.view_mut())
                        .expect("survival logslope block axpy should match block dimensions");

                    let designs = [
                        &self.design_entry,
                        &self.design_exit,
                        &self.design_derivative_exit,
                    ];
                    for a in 0..3 {
                        for b in 0..3 {
                            designs[a]
                                .row_outer_into(row, designs[b], f_pipi[[a, b]], &mut acc.4)
                                .expect("time row_outer_into dim mismatch");
                        }
                    }

                    self.marginal_design
                        .syr_row_into(
                            row,
                            f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]],
                            &mut acc.5,
                        )
                        .expect("survival marginal block syr should match block dimensions");
                    self.logslope_design
                        .syr_row_into(row, f_pipi[[3, 3]], &mut acc.6)
                        .expect("survival logslope block syr should match block dimensions");
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4 += &b.4;
                    a.5 += &b.5;
                    a.6 += &b.6;
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Dense(hess_time),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: SymmetricMatrix::Dense(hess_marginal),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: SymmetricMatrix::Dense(hess_logslope),
                },
            ],
        })
    }

    // ── Sparse path ──────────────────────────────────────────────────

    fn evaluate_blockwise_exact_newton_sparse(
        &self,
        block_states: &[ParameterBlockState],
        time_csrs: &(
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
        ),
        marginal_csr: &Arc<faer::sparse::SparseRowMat<usize, f64>>,
        logslope_csr: &Arc<faer::sparse::SparseRowMat<usize, f64>>,
    ) -> Result<FamilyEvaluation, String> {
        use crate::matrix::SparseHessianAccumulator;

        let slices = block_slices(block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let (ref csr_entry, ref csr_exit, ref csr_deriv) = *time_csrs;

        // Build symbolic sparsity patterns once.
        let pattern_time = SparseHessianAccumulator::from_multi_csr(
            &[csr_entry.as_ref(), csr_exit.as_ref(), csr_deriv.as_ref()],
            p_t,
        );
        let pattern_marginal =
            SparseHessianAccumulator::from_single_csr(marginal_csr.as_ref(), p_m);
        let pattern_logslope =
            SparseHessianAccumulator::from_single_csr(logslope_csr.as_ref(), p_g);

        // Pre-extract CSR symbolic parts for zero-overhead inner loop access.
        let e_sym = csr_entry.symbolic();
        let e_rp = e_sym.row_ptr();
        let e_ci = e_sym.col_idx();
        let e_v = csr_entry.val();

        let x_sym = csr_exit.symbolic();
        let x_rp = x_sym.row_ptr();
        let x_ci = x_sym.col_idx();
        let x_v = csr_exit.val();

        let d_sym = csr_deriv.symbolic();
        let d_rp = d_sym.row_ptr();
        let d_ci = d_sym.col_idx();
        let d_v = csr_deriv.val();

        let m_sym = marginal_csr.symbolic();
        let m_rp = m_sym.row_ptr();
        let m_ci = m_sym.col_idx();
        let m_v = marginal_csr.val();

        let g_sym = logslope_csr.symbolic();
        let g_rp = g_sym.row_ptr();
        let g_ci = g_sym.col_idx();
        let g_v = logslope_csr.val();

        // Accumulator type: gradients dense, Hessians sparse value buffers.
        type SAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            SparseHessianAccumulator,
            SparseHessianAccumulator,
            SparseHessianAccumulator,
        );

        let make_acc = || -> SAcc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                pattern_time.empty_clone(),
                pattern_marginal.empty_clone(),
                pattern_logslope.empty_clone(),
            )
        };

        let (ll, grad_time, grad_marginal, grad_logslope, acc_time, acc_marginal, acc_logslope) =
            (0..self.n)
                .into_par_iter()
                .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    // ── gradients (dense axpy via CSR scatter) ────────────
                    {
                        let gt = &mut acc.1;
                        for p in e_rp[row]..e_rp[row + 1] {
                            gt[e_ci[p]] -= f_pi[0] * e_v[p];
                        }
                        for p in x_rp[row]..x_rp[row + 1] {
                            gt[x_ci[p]] -= f_pi[1] * x_v[p];
                        }
                        for p in d_rp[row]..d_rp[row + 1] {
                            gt[d_ci[p]] -= f_pi[2] * d_v[p];
                        }
                    }
                    {
                        let gm = &mut acc.2;
                        let alpha_m = -(f_pi[0] + f_pi[1]);
                        for p in m_rp[row]..m_rp[row + 1] {
                            gm[m_ci[p]] += alpha_m * m_v[p];
                        }
                    }
                    {
                        let gg = &mut acc.3;
                        for p in g_rp[row]..g_rp[row + 1] {
                            gg[g_ci[p]] -= f_pi[3] * g_v[p];
                        }
                    }

                    // ── time Hessian: 3×3 cross-product scatter ──────────
                    // Only emit upper-triangle entries (ca <= cb) to avoid
                    // double-counting: SymmetricMatrix::Sparse mirrors the
                    // upper triangle into the lower.
                    let row_slices: [(std::ops::Range<usize>, &[usize], &[f64]); 3] = [
                        (e_rp[row]..e_rp[row + 1], e_ci, e_v),
                        (x_rp[row]..x_rp[row + 1], x_ci, x_v),
                        (d_rp[row]..d_rp[row + 1], d_ci, d_v),
                    ];
                    let ht = &mut acc.4;
                    for a in 0..3 {
                        for b in 0..3 {
                            let alpha = f_pipi[[a, b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            let (ref ra, cia, va) = row_slices[a];
                            let (ref rb, cib, vb) = row_slices[b];
                            for pi in ra.clone() {
                                let ca = cia[pi];
                                let xia = va[pi] * alpha;
                                for pj in rb.clone() {
                                    let cb = cib[pj];
                                    if ca <= cb {
                                        ht.add_upper(ca, cb, xia * vb[pj]);
                                    }
                                }
                            }
                        }
                    }

                    // ── marginal Hessian: symmetric rank-1 scatter ───────
                    // CSR column indices are sorted, so pj >= pi ⟹ cb >= ca,
                    // giving us the upper triangle directly.
                    let alpha_m = f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    if alpha_m != 0.0 {
                        let hm = &mut acc.5;
                        let m_start = m_rp[row];
                        let m_end = m_rp[row + 1];
                        for pi in m_start..m_end {
                            let ca = m_ci[pi];
                            let xia = m_v[pi] * alpha_m;
                            for pj in pi..m_end {
                                hm.add_upper(ca, m_ci[pj], xia * m_v[pj]);
                            }
                        }
                    }

                    // ── logslope Hessian: symmetric rank-1 scatter ───────
                    let alpha_g = f_pipi[[3, 3]];
                    if alpha_g != 0.0 {
                        let hg = &mut acc.6;
                        let g_start = g_rp[row];
                        let g_end = g_rp[row + 1];
                        for pi in g_start..g_end {
                            let ca = g_ci[pi];
                            let xia = g_v[pi] * alpha_g;
                            for pj in pi..g_end {
                                hg.add_upper(ca, g_ci[pj], xia * g_v[pj]);
                            }
                        }
                    }

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4.add_values(&b.4.values);
                    a.5.add_values(&b.5.values);
                    a.6.add_values(&b.6.values);
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Sparse(acc_time.into_sparse_col_mat()),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_marginal,
                    hessian: SymmetricMatrix::Sparse(acc_marginal.into_sparse_col_mat()),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_logslope,
                    hessian: SymmetricMatrix::Sparse(acc_logslope.into_sparse_col_mat()),
                },
            ],
        })
    }
}

// ── CustomFamily impl ─────────────────────────────────────────────────

/// Maximum row-loop work proxy before downgrading to first-order:
///   n × K_pairs × primary²
const EXACT_OUTER_MAX_ROW_WORK: u64 = 2_000_000;

impl CustomFamily for SurvivalMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        if self.flex_timewiggle_active() {
            return ExactOuterDerivativeOrder::Zeroth;
        }
        // Shared memory gate: K(K+1)/2 × p² dense psi Hessians.
        if cost_gated_outer_order(specs) == ExactOuterDerivativeOrder::First {
            return ExactOuterDerivativeOrder::First;
        }
        // Family-specific row-loop gate.
        let n = self.n as u64;
        let k: u64 = specs.iter().map(|s| s.penalties.len() as u64).sum();
        let k_pairs = k.saturating_mul(k.saturating_add(1)) / 2;
        let primary = N_PRIMARY as u64;
        let row_work = n.saturating_mul(k_pairs).saturating_mul(primary * primary);
        if row_work > EXACT_OUTER_MAX_ROW_WORK {
            ExactOuterDerivativeOrder::First
        } else {
            ExactOuterDerivativeOrder::Second
        }
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        // True fast path: closed-form scalar NLL, no jets, no Vecs, no gradients.
        let guard = self.derivative_guard;
        let mut ll = 0.0;
        for i in 0..self.n {
            let q_geom = self.row_dynamic_q_geometry(i, block_states)?;
            let g = block_states[2].eta[i];
            let (nll, _, _) = row_primary_closed_form(
                q_geom.q0,
                q_geom.q1,
                q_geom.qd1,
                g,
                self.z[i],
                self.weights[i],
                self.event[i],
                guard,
            )?;
            ll -= nll;
        }
        Ok(ll)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX {
                return Err("unreachable survival hessian state".to_string());
            }
            return Ok(None);
        }
        let slices = block_slices(block_states);
        if slices.total >= 512 {
            return Ok(None);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let cache = build_row_kernel_cache(&kern)?;
        Ok(Some(crate::families::row_kernel::row_kernel_hessian_dense(
            &kern, &cache,
        )))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        if self.flex_timewiggle_active() {
            return false;
        }
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX {
                return Err("unreachable survival workspace state".to_string());
            }
            return Ok(None);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX || d_beta_flat.len() == usize::MAX {
                return Err("unreachable survival directional state".to_string());
            }
            return Ok(None);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
        crate::families::row_kernel::row_kernel_directional_derivative(&kern, sl).map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX
                || d_beta_u_flat.len() == usize::MAX
                || d_beta_v_flat.len() == usize::MAX
            {
                return Err("unreachable survival second directional state".to_string());
            }
            return Ok(None);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
        let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
        crate::families::row_kernel::row_kernel_second_directional_derivative(&kern, su, sv)
            .map(Some)
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX
                || derivative_blocks.len() == usize::MAX
                || psi_index == usize::MAX
            {
                return Err("unreachable survival psi state".to_string());
            }
            return Ok(None);
        }
        self.psi_terms(block_states, derivative_blocks, psi_index)
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX
                || derivative_blocks.len() == usize::MAX
                || psi_i == usize::MAX
                || psi_j == usize::MAX
            {
                return Err("unreachable survival psi second-order state".to_string());
            }
            return Ok(None);
        }
        self.psi_second_order_terms(block_states, derivative_blocks, psi_i, psi_j)
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX
                || derivative_blocks.len() == usize::MAX
                || psi_index == usize::MAX
                || d_beta_flat.len() == usize::MAX
            {
                return Err("unreachable survival psi directional state".to_string());
            }
            return Ok(None);
        }
        self.psi_hessian_directional_derivative(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if self.flex_timewiggle_active() {
            if block_states.len() == usize::MAX || derivative_blocks.len() == usize::MAX {
                return Err("unreachable survival psi workspace state".to_string());
            }
            return Ok(None);
        }
        Ok(Some(Arc::new(SurvivalMarginalSlopePsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        )?)))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == 0 {
            Ok(self.time_linear_constraints.clone())
        } else {
            Ok(None)
        }
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_idx == 0 {
            return self.max_feasible_time_step(&block_states[0].beta, delta);
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_idx >= block_states.len() {
            return Err(format!(
                "post-update block index {} out of range for {} blocks",
                block_idx,
                block_states.len()
            ));
        }
        Ok(beta)
    }
}

// ── Building block specs ──────────────────────────────────────────────

fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &DesignMatrix,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: design_exit.clone(),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_block.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_logslope_blockspec(
    design: &TermCollectionDesign,
    baseline: f64,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: design.design.clone(),
        offset: Array1::from_elem(design.design.nrows(), baseline),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_marginal_blockspec(
    design: &TermCollectionDesign,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: design.design.clone(),
        offset: Array1::zeros(design.design.nrows()),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn inner_fit(
    family: &SurvivalMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

fn joint_setup(
    time_penalties: usize,
    marginalspec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = time_penalties + marginal_penalties + logslope_penalties;
    let rho0vec = Array1::<f64>::zeros(rho_dim);
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    );
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    );
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(marginal_kappa.as_array().iter());
    values.extend(logslope_kappa.as_array().iter());
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(marginal_kappa.dims_per_term());
    dims.extend(logslope_kappa.dims_per_term());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values.clone()), dims.clone());
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&dims, kappa_options);
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&dims, kappa_options);
    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

fn validate_standardized_z(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &str,
) -> Result<(), String> {
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(format!("{context} requires positive finite total weight"));
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    if mean.abs() > 1e-6 || (sd - 1.0).abs() > 1e-6 {
        return Err(format!(
            "{context} requires z to already represent a latent N(0,1) score; weighted mean 0 and weighted sd 1 are necessary sanity checks. got mean={mean:.6e}, sd={sd:.6e}"
        ));
    }
    if sd > 1e-12 {
        let skew = z
            .iter()
            .zip(weights.iter())
            .map(|(&zi, &wi)| wi * ((zi - mean) / sd).powi(3))
            .sum::<f64>()
            / weight_sum;
        let kurt = z
            .iter()
            .zip(weights.iter())
            .map(|(&zi, &wi)| wi * ((zi - mean) / sd).powi(4))
            .sum::<f64>()
            / weight_sum
            - 3.0;
        if skew.abs() > 0.5 || kurt.abs() > 1.5 {
            log::warn!(
                "{context}: z has skewness={skew:.3} and excess kurtosis={kurt:.3}; the Gaussian marginalization identity is only exact for Z~N(0,1). Results may be biased if z is not approximately normal."
            );
        }
    }
    Ok(())
}

fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
    let n = spec.age_entry.len();
    if spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.z.len() != n
    {
        return Err(format!(
            "survival-marginal-slope row mismatch: entry={}, exit={}, event={}, weights={}, z={}",
            n,
            spec.age_exit.len(),
            spec.event_target.len(),
            spec.weights.len(),
            spec.z.len()
        ));
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err("survival-marginal-slope requires finite non-negative weights".to_string());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err("survival-marginal-slope requires finite z values".to_string());
    }
    validate_standardized_z(&spec.z, &spec.weights, "survival-marginal-slope")?;
    if spec.event_target.iter().any(|&d| d != 0.0 && d != 1.0) {
        return Err(
            "survival-marginal-slope requires binary event indicators (0.0 or 1.0)".to_string(),
        );
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(format!(
            "survival-marginal-slope requires derivative_guard > 0, got {}",
            spec.derivative_guard
        ));
    }
    for i in 0..n {
        if spec.age_exit[i] < spec.age_entry[i] {
            return Err(format!(
                "survival-marginal-slope row {i}: exit time ({}) < entry time ({})",
                spec.age_exit[i], spec.age_entry[i]
            ));
        }
    }
    let n_entry = spec.time_block.design_entry.nrows();
    let n_exit = spec.time_block.design_exit.nrows();
    let n_deriv = spec.time_block.design_derivative_exit.nrows();
    if n_entry != n || n_exit != n || n_deriv != n {
        return Err(format!(
            "survival-marginal-slope time block design row mismatch: \
             data={n}, design_entry={n_entry}, design_exit={n_exit}, design_derivative_exit={n_deriv}"
        ));
    }
    let p_entry = spec.time_block.design_entry.ncols();
    let p_exit = spec.time_block.design_exit.ncols();
    let p_deriv = spec.time_block.design_derivative_exit.ncols();
    if p_exit != p_entry || p_deriv != p_entry {
        return Err(format!(
            "survival-marginal-slope time block design column mismatch: entry={p_entry}, exit={p_exit}, deriv={p_deriv}"
        ));
    }
    if !spec.time_block.structural_monotonicity {
        return Err(
            "survival-marginal-slope requires structural time monotonicity by construction; non-structural time transforms are no longer supported"
                .to_string(),
        );
    }
    if let Some(beta0) = &spec.time_block.initial_beta {
        let derivative_constraints = time_derivative_lower_bound_constraints(
            &spec.time_block.design_derivative_exit.to_dense(),
            &spec.time_block.derivative_offset_exit,
            spec.derivative_guard,
        )?;
        if let Some(constraints) = derivative_constraints.as_ref() {
            if beta0.len() != constraints.a.ncols() {
                return Err(format!(
                    "survival-marginal-slope time_block initial_beta length mismatch: got {}, expected {}",
                    beta0.len(),
                    constraints.a.ncols()
                ));
            }
            for row in 0..constraints.a.nrows() {
                let slack = constraints.a.row(row).dot(beta0) - constraints.b[row];
                if slack < -1e-10 {
                    return Err(format!(
                        "survival-marginal-slope time_block initial_beta violates derivative guard at row {row}: slack={slack:.3e}"
                    ));
                }
            }
        }
    }
    if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
        if timewiggle.degree != 3 {
            return Err(format!(
                "survival-marginal-slope timewiggle requires cubic degree=3, got {}",
                timewiggle.degree
            ));
        }
        if timewiggle.ncols == 0 {
            return Err(
                "survival-marginal-slope timewiggle requires at least one wiggle coefficient"
                    .to_string(),
            );
        }
        if spec.time_block.design_exit.ncols() < timewiggle.ncols {
            return Err(format!(
                "survival-marginal-slope timewiggle requests {} tail columns but time block only has {} columns",
                timewiggle.ncols,
                spec.time_block.design_exit.ncols()
            ));
        }
    }
    Ok(())
}

/// Compute a baseline slope from the actual survival marginal-slope likelihood,
/// using the baseline offsets alone as a time-only pilot q(t).
///
/// This is a safeguarded 1D Newton solve on the true row objective. It does not
/// use a coarse fixed grid scan.
fn pooled_survival_baseline(
    event: &Array1<f64>,
    weights: &Array1<f64>,
    z: &Array1<f64>,
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
) -> f64 {
    let n = event.len();
    if n == 0 {
        return 0.0;
    }
    let objective_grad_hess = |slope: f64| -> Option<(f64, f64, f64)> {
        let mut obj = 0.0;
        let mut grad = 0.0;
        let mut hess = 0.0;
        for i in 0..n {
            let (row_obj, row_grad, row_hess) = row_primary_closed_form(
                q0[i], q1[i], qd1[i], slope, z[i], weights[i], event[i], 0.0,
            )
            .ok()?;
            obj += row_obj;
            grad += row_grad[3];
            hess += row_hess[3][3];
        }
        Some((obj, grad, hess))
    };

    let Some(state0) = objective_grad_hess(0.0) else {
        return 0.0;
    };
    if !state0.0.is_finite() {
        return 0.0;
    }
    if state0.1.abs() < 1e-8 {
        return 0.0;
    }

    let mut best_slope = 0.0;
    let mut best = state0;

    let mut bracket_lo = if state0.1 <= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut bracket_hi = if state0.1 >= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut step = 0.5f64;
    for _ in 0..48 {
        for &candidate in &[-step, step] {
            if let Some(state) = objective_grad_hess(candidate) {
                if state.0 < best.0 {
                    best_slope = candidate;
                    best = state;
                }
                if state.1 <= 0.0 {
                    bracket_lo = Some((candidate, state));
                }
                if state.1 >= 0.0 {
                    bracket_hi = Some((candidate, state));
                }
                if let (Some((lo, lo_state)), Some((hi, hi_state))) = (bracket_lo, bracket_hi)
                    && lo < hi
                    && lo_state.1 <= 0.0
                    && hi_state.1 >= 0.0
                {
                    let mut slope = best_slope.clamp(lo, hi);
                    let mut state = if (slope - lo).abs() < f64::EPSILON {
                        lo_state
                    } else if (slope - hi).abs() < f64::EPSILON {
                        hi_state
                    } else {
                        match objective_grad_hess(slope) {
                            Some(s) => s,
                            None => {
                                slope = 0.5 * (lo + hi);
                                objective_grad_hess(slope).unwrap_or(best)
                            }
                        }
                    };

                    let mut bracket_lo = (lo, lo_state);
                    let mut bracket_hi = (hi, hi_state);
                    for _ in 0..60 {
                        if state.1.abs() < 1e-8 || (bracket_hi.0 - bracket_lo.0).abs() < 1e-8 {
                            break;
                        }
                        let mut candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                        if state.2.is_finite() && state.2 > 0.0 {
                            let newton = slope - state.1 / state.2;
                            if newton > bracket_lo.0 && newton < bracket_hi.0 {
                                candidate = newton;
                            }
                        }
                        let Some(candidate_state) = objective_grad_hess(candidate) else {
                            candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                            let Some(mid_state) = objective_grad_hess(candidate) else {
                                break;
                            };
                            if mid_state.0 < best.0 {
                                best_slope = candidate;
                                best = mid_state;
                            }
                            if mid_state.1 <= 0.0 {
                                bracket_lo = (candidate, mid_state);
                            } else {
                                bracket_hi = (candidate, mid_state);
                            }
                            slope = candidate;
                            state = mid_state;
                            continue;
                        };
                        if candidate_state.0 < best.0 {
                            best_slope = candidate;
                            best = candidate_state;
                        }
                        if candidate_state.1 <= 0.0 {
                            bracket_lo = (candidate, candidate_state);
                        } else {
                            bracket_hi = (candidate, candidate_state);
                        }
                        slope = candidate;
                        state = candidate_state;
                    }
                    return if best.0.is_finite() { best_slope } else { 0.0 };
                }
            }
        }
        step *= 2.0;
    }
    if best.0.is_finite() { best_slope } else { 0.0 }
}

// ── Public fitting function ───────────────────────────────────────────

pub fn fit_survival_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: SurvivalMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    validate_spec(&spec)?;
    let n = spec.age_entry.len();
    let baseline_slope = pooled_survival_baseline(
        &spec.event_target,
        &spec.weights,
        &spec.z,
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
    );

    let marginal_design =
        build_term_collection_design(data, &spec.marginalspec).map_err(|e| e.to_string())?;
    let marginalspec_boot =
        freeze_term_collection_from_design(&spec.marginalspec, &marginal_design)
            .map_err(|e| e.to_string())?;
    let logslope_design =
        build_term_collection_design(data, &spec.logslopespec).map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_term_collection_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let time_penalties_len = spec.time_block.penalties.len();
    let setup = joint_setup(
        time_penalties_len,
        &marginalspec_boot,
        marginal_design.penalties.len(),
        &logslopespec_boot,
        logslope_design.penalties.len(),
        kappa_options,
    );

    let hints = RefCell::new((
        None::<Array1<f64>>,
        None::<Array1<f64>>,
        None::<Array1<f64>>,
    ));
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);

    let event = Arc::new(spec.event_target.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());
    let derivative_guard = spec.derivative_guard;
    // Time designs arrive as DesignMatrix (already sparse for local-support
    // bases like B-spline, or dense for I-spline).  No post-hoc scan needed.
    let design_entry = spec.time_block.design_entry.clone();
    let design_exit = spec.time_block.design_exit.clone();
    let design_derivative_exit = spec.time_block.design_derivative_exit.clone();
    let offset_entry = Arc::new(spec.time_block.offset_entry.clone());
    let offset_exit = Arc::new(spec.time_block.offset_exit.clone());
    let derivative_offset_exit = Arc::new(spec.time_block.derivative_offset_exit.clone());
    let time_block_ref = spec.time_block.clone();
    let time_linear_constraints = time_derivative_lower_bound_constraints(
        &design_derivative_exit.to_dense(),
        derivative_offset_exit.as_ref(),
        derivative_guard,
    )?;

    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign|
     -> SurvivalMarginalSlopeFamily {
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::clone(&event),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            derivative_guard,
            design_entry: design_entry.clone(),
            design_exit: design_exit.clone(),
            design_derivative_exit: design_derivative_exit.clone(),
            offset_entry: Arc::clone(&offset_entry),
            offset_exit: Arc::clone(&offset_exit),
            derivative_offset_exit: Arc::clone(&derivative_offset_exit),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            time_linear_constraints: time_linear_constraints.clone(),
            time_wiggle_knots: spec.timewiggle_block.as_ref().map(|w| w.knots.clone()),
            time_wiggle_degree: spec.timewiggle_block.as_ref().map(|w| w.degree),
            time_wiggle_ncols: spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols),
        }
    };

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_time = rho
            .slice(s![cursor..cursor + time_penalties_len])
            .to_owned();
        cursor += time_penalties_len;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        let time_beta_hint = if let Some(constraints) = time_linear_constraints.as_ref() {
            Some(project_onto_linear_constraints(
                design_exit.ncols(),
                constraints,
                hints.0.as_ref().or(time_block_ref.initial_beta.as_ref()),
            ))
        } else {
            hints
                .0
                .clone()
                .or_else(|| time_block_ref.initial_beta.clone())
        };
        Ok(vec![
            build_time_blockspec(&time_block_ref, &design_exit, rho_time, time_beta_hint),
            build_marginal_blockspec(marginal_design, rho_marginal, hints.1.clone()),
            build_logslope_blockspec(
                logslope_design,
                baseline_slope,
                rho_logslope,
                hints.2.clone(),
            ),
        ])
    };

    // ── Pilot fit: rigid (zero-penalty) to seed coefficients ────────────
    {
        let rigid_rho = Array1::<f64>::zeros(
            time_penalties_len + marginal_design.penalties.len() + logslope_design.penalties.len(),
        );
        let rigid_blocks = build_blocks(&rigid_rho, &marginal_design, &logslope_design)?;
        let rigid_family = make_family(&marginal_design, &logslope_design);
        if let Ok(rigid_fit) = inner_fit(&rigid_family, &rigid_blocks, options) {
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = rigid_fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = rigid_fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
            if let Some(block) = rigid_fit.block_states.get(2) {
                hints_mut.2 = Some(block.beta.clone());
            }
        }
    }

    // Check analytic derivatives
    let marginal_derivatives =
        build_block_spatial_psi_derivatives(data, &marginalspec_boot, &marginal_design)?;
    let logslope_derivatives =
        build_block_spatial_psi_derivatives(data, &logslopespec_boot, &logslope_design)?;
    let analytic_joint_derivatives_available = marginal_derivatives.is_some()
        || logslope_derivatives.is_some()
        || setup.log_kappa_dim() == 0;

    if setup.log_kappa_dim() > 0
        && !(marginal_derivatives.is_some() || logslope_derivatives.is_some())
    {
        return Err(
            "exact survival marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }

    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design);
    let (joint_gradient, joint_hessian) =
        custom_family_outer_derivatives(&initial_family, &initial_blocks, options);
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_joint_hessian_available = analytic_joint_derivatives_available
        && matches!(
            joint_hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );

    // Only the baseline and slope surface blocks can have spatial terms
    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms, logslope_terms],
        kappa_options,
        &setup,
        crate::seeding::SeedRiskProfile::Survival,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.get(0) {
                hints_mut.0 = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.1 = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(2) {
                hints_mut.2 = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            let derivative_blocks = vec![
                Vec::new(),
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?
                    .unwrap_or_default(),
                build_block_spatial_psi_derivatives(data, &specs[1], &designs[1])?
                    .unwrap_or_default(),
            ];
            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &blocks,
                options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian,
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            if need_hessian && eval.outer_hessian.is_none() {
                return Err(
                    "exact survival marginal-slope joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let designs = solved.designs;
    Ok(SurvivalMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs[0].clone(),
        logslope_design: designs[1].clone(),
        baseline_slope,
        time_block_penalties_len: time_penalties_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::CustomFamily;
    use crate::matrix::{DenseDesignMatrix, SymmetricMatrix};
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::array;

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn base_time_block() -> TimeBlockInput {
        TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Array1::zeros(1),
            offset_exit: Array1::zeros(1),
            derivative_offset_exit: Array1::zeros(1),
            structural_monotonicity: true,
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: Some(Array1::zeros(1)),
        }
    }

    fn sparse_design(dense: &Array2<f64>) -> DesignMatrix {
        let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let value = dense[[i, j]];
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
        }
        let sparse = SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("assemble sparse design");
        DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse))
    }

    #[test]
    fn validate_spec_rejects_nonstructural_time_block() {
        let spec = SurvivalMarginalSlopeTermSpec {
            age_entry: array![0.0, 0.0],
            age_exit: array![1.0, 1.0],
            event_target: array![0.0, 1.0],
            weights: array![1.0, 1.0],
            z: array![-1.0, 1.0],
            marginalspec: empty_termspec(),
            derivative_guard: 1e-4,
            time_block: TimeBlockInput {
                design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
                design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
                design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
                offset_entry: Array1::zeros(2),
                offset_exit: Array1::zeros(2),
                derivative_offset_exit: Array1::zeros(2),
                structural_monotonicity: false,
                ..base_time_block()
            },
            timewiggle_block: None,
            logslopespec: empty_termspec(),
        };

        let err = validate_spec(&spec).expect_err("non-structural time block should fail");
        assert!(
            err.contains("requires structural time monotonicity"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn censored_rows_still_reject_invalid_time_derivative() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0]),
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::ones((1, 1)),
            )),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::zeros(1)),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];

        let err = family
            .row_neglog_directional(0, &block_states, &[])
            .expect_err("censored rows must still enforce the time-derivative domain");
        assert!(
            err.contains("monotonicity violated at row 0"),
            "unexpected error: {err}"
        );
    }

    fn assert_close(lhs: f64, rhs: f64, tol: f64, label: &str) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "{label} mismatch: lhs={lhs:.12e}, rhs={rhs:.12e}, tol={tol:.3e}"
        );
    }

    #[test]
    fn closed_form_row_matches_exact_directional_derivatives() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.2]),
            z: Arc::new(array![0.3]),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.8]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.4]],
            )),
            offset_entry: Arc::new(array![0.1]),
            offset_exit: Arc::new(array![-0.2]),
            derivative_offset_exit: Arc::new(array![0.05]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.4],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![-0.1],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![0.7],
                eta: array![0.7],
            },
        ];

        let (nll_closed, grad_closed, hess_closed) = family
            .compute_row_primary_gradient_hessian_uncached(0, &block_states)
            .expect("closed-form row derivatives");
        let nll_exact = family
            .row_neglog_directional(0, &block_states, &[])
            .expect("exact row objective");
        assert_close(nll_closed, nll_exact, 1e-12, "nll");

        for a in 0..N_PRIMARY {
            let dir_a = unit_primary_direction(a);
            let grad_exact = family
                .row_neglog_directional(0, &block_states, &[dir_a.clone()])
                .expect("exact row gradient");
            assert_close(grad_closed[a], grad_exact, 1e-10, &format!("grad[{a}]"));
            for b in 0..N_PRIMARY {
                let dir_b = unit_primary_direction(b);
                let hess_exact = family
                    .row_neglog_directional(0, &block_states, &[dir_a.clone(), dir_b])
                    .expect("exact row hessian");
                assert_close(
                    hess_closed[[a, b]],
                    hess_exact,
                    1e-9,
                    &format!("hess[{a},{b}]"),
                );
            }
        }
    }

    #[test]
    fn exact_newton_evaluation_propagates_invalid_rows() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0]),
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![0.0]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];

        let err = family
            .evaluate(&block_states)
            .expect_err("invalid rows must abort exact-newton evaluation");
        assert!(
            err.contains("monotonicity violated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn structural_time_constraints_use_derivative_lower_bound_rows() {
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![0.0, 1.0]),
            weights: Arc::new(array![1.0, 1.0]),
            z: Arc::new(array![0.0, 0.0]),
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 2)),
            )),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 2)),
            )),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 2.0], [3.0, 4.0]],
            )),
            offset_entry: Arc::new(Array1::zeros(2)),
            offset_exit: Arc::new(Array1::zeros(2)),
            derivative_offset_exit: Arc::new(array![0.25, 0.5]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 0)),
            )),
            time_linear_constraints: time_derivative_lower_bound_constraints(
                &array![[1.0, 2.0], [3.0, 4.0]],
                &array![0.25, 0.5],
                1e-4,
            )
            .expect("time derivative constraints"),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let spec = ParameterBlockSpec {
            name: "time_surface".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [0.0, 0.0],
                [0.0, 0.0]
            ])),
            offset: Array1::zeros(2),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let constraints = family
            .block_linear_constraints(&[], 0, &spec)
            .expect("constraint lookup")
            .expect("time constraints");
        assert_eq!(constraints.a, array![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(constraints.b, array![-0.2499, -0.4999]);
    }

    #[test]
    fn time_block_post_update_leaves_beta_unchanged() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0]),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.0
            ]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 0.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![1e-6]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            time_linear_constraints: time_derivative_lower_bound_constraints(
                &array![[1.0, 0.0]],
                &array![1e-6],
                1e-6,
            )
            .expect("time derivative constraints"),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let spec = ParameterBlockSpec {
            name: "time_surface".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 0.0]])),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let beta = family
            .post_update_block_beta(
                &[ParameterBlockState {
                    beta: array![0.0, 0.0],
                    eta: array![0.0],
                }],
                0,
                &spec,
                array![-0.3, 0.2],
            )
            .expect("return time beta");
        assert_eq!(beta, array![-0.3, 0.2]);
    }

    #[test]
    fn time_block_feasible_step_stays_inside_derivative_guard() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0]),
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0, 0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0, 0.0
            ]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 0.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![0.2]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            time_linear_constraints: time_derivative_lower_bound_constraints(
                &array![[1.0, 0.0]],
                &array![0.2],
                1e-4,
            )
            .expect("time derivative constraints"),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];
        let alpha = family
            .max_feasible_step_size(&states, 0, &array![-1.0, 0.0])
            .expect("time step ceiling")
            .expect("time step should be bounded");
        assert!(alpha > 0.0 && alpha < 1.0);
        let feasible = &states[0].beta + &(array![-1.0, 0.0] * alpha);
        let slack = family
            .time_linear_constraints
            .as_ref()
            .expect("constraints")
            .a
            .row(0)
            .dot(&feasible)
            - family
                .time_linear_constraints
                .as_ref()
                .expect("constraints")
                .b[0];
        assert!(slack >= 0.0);
    }

    #[test]
    fn mixed_blockwise_exact_newton_preserves_sparse_block_hessians() {
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.1, -0.2]),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.6]])),
            design_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![[0.9], [0.5]])),
            design_derivative_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset_entry: Arc::new(array![0.0, 0.0]),
            offset_exit: Arc::new(array![0.0, 0.0]),
            derivative_offset_exit: Arc::new(array![0.05, 0.05]),
            marginal_design: sparse_design(&array![[1.0, 0.0], [0.0, 1.0]]),
            logslope_design: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.5]])),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.4],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.2, -0.1],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.3],
                eta: array![0.3, 0.3],
            },
        ];

        let eval = family
            .evaluate_blockwise_exact_newton(&block_states)
            .expect("mixed exact-newton evaluation");

        assert!(matches!(
            &eval.blockworking_sets[0],
            BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Dense(_),
                ..
            }
        ));
        assert!(matches!(
            &eval.blockworking_sets[1],
            BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Sparse(_),
                ..
            }
        ));
        assert!(matches!(
            &eval.blockworking_sets[2],
            BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Dense(_),
                ..
            }
        ));
    }
}
