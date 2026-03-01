use crate::faer_ndarray::FaerCholesky;
use crate::faer_ndarray::{FaerArrayView, FaerEigh, fast_ata, fast_atv};
use crate::matrix::DesignMatrix;
use crate::pirls::LinearInequalityConstraints;
use crate::types::{LinkFunction, RidgeDeterminantMode, RidgePolicy};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use ndarray::{Array1, Array2};
use wolfe_bfgs::{Bfgs, BfgsError};

/// Optional known link metadata when a family uses a learnable wiggle correction.
#[derive(Debug, Clone, Copy)]
pub struct KnownLinkWiggle {
    pub base_link: LinkFunction,
    pub wiggle_block: Option<usize>,
}

/// Static specification for one parameter block in a custom family.
#[derive(Clone)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    /// Block-local penalty matrices (all p_block x p_block).
    pub penalties: Vec<Array2<f64>>,
    /// Initial log-smoothing parameters for this block (same length as `penalties`).
    pub initial_log_lambdas: Array1<f64>,
    /// Optional initial coefficients (defaults to zeros if omitted).
    pub initial_beta: Option<Array1<f64>>,
}

/// Current state for a parameter block.
#[derive(Clone)]
pub struct ParameterBlockState {
    pub beta: Array1<f64>,
    pub eta: Array1<f64>,
}

/// Working quantities supplied by a custom family for one block.
#[derive(Clone)]
pub enum BlockWorkingSet {
    /// Standard IRLS/GLM-style diagonal working set for eta-space updates.
    Diagonal {
        /// IRLS pseudo-response for this block's linear predictor.
        working_response: Array1<f64>,
        /// IRLS working weights for this block (non-negative, length n).
        working_weights: Array1<f64>,
    },
    /// Exact Newton block update in coefficient space.
    ///
    /// `gradient` is ∇ log L wrt block coefficients.
    /// `hessian` is -∇² log L wrt block coefficients (positive semidefinite near optimum).
    ExactNewton {
        gradient: Array1<f64>,
        hessian: Array2<f64>,
    },
}

/// Family evaluation over all parameter blocks.
#[derive(Clone)]
pub struct FamilyEvaluation {
    pub log_likelihood: f64,
    pub block_working_sets: Vec<BlockWorkingSet>,
}

/// User-defined family contract for multi-block generalized models.
pub trait CustomFamily {
    /// Evaluate log-likelihood and per-block working quantities at current block predictors.
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String>;

    /// Optional metadata describing a known link with learnable wiggle.
    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        None
    }

    /// Optional dynamic geometry hook for blocks whose design/offset depend on
    /// current values of other blocks.
    fn block_geometry(
        &self,
        _block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        Ok((spec.design.clone(), spec.offset.clone()))
    }

    /// Optional per-block coefficient projection applied after each block update.
    fn post_update_beta(&self, beta: Array1<f64>) -> Result<Array1<f64>, String> {
        Ok(beta)
    }

    /// Optional linear inequality constraints for a block update:
    /// `A * beta_block >= b`.
    fn block_linear_constraints(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        Ok(None)
    }

    /// Optional exact directional derivative of a block's ExactNewton Hessian.
    ///
    /// Returns `Some(dH)` where:
    /// - `dH` is the directional derivative of the block Hessian with respect to
    ///   the provided coefficient-space direction `d_beta` at current state.
    /// - shape is `(p_block, p_block)`.
    ///
    /// Default `None` means the caller may fall back to numerical directional
    /// differentiation for the `H_rho` correction in LAML gradients.
    fn exact_newton_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        _d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact joint coefficient-space Hessian across all blocks.
    ///
    /// Returns the unpenalized matrix `H = -∇² log L` in the flattened block order.
    fn exact_newton_joint_hessian(
        &self,
        _block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Optional exact directional derivative of the joint coefficient-space Hessian.
    ///
    /// Returns `Some(dH)` where `dH` is the directional derivative of the
    /// unpenalized joint Hessian `H = -∇² log L` along the flattened
    /// coefficient-space direction `d_beta_flat`.
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }
}

#[derive(Clone)]
pub struct BlockwiseFitOptions {
    pub inner_max_cycles: usize,
    pub inner_tol: f64,
    pub outer_max_iter: usize,
    pub outer_tol: f64,
    pub min_weight: f64,
    pub ridge_floor: f64,
    /// Shared ridge semantics used by solve/quadratic/logdet terms.
    pub ridge_policy: RidgePolicy,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_reml_objective: bool,
}

impl Default for BlockwiseFitOptions {
    fn default() -> Self {
        Self {
            inner_max_cycles: 100,
            inner_tol: 1e-6,
            outer_max_iter: 60,
            outer_tol: 1e-5,
            min_weight: 1e-12,
            ridge_floor: 1e-12,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: true,
        }
    }
}

#[derive(Clone)]
pub struct BlockwiseInnerResult {
    pub block_states: Vec<ParameterBlockState>,
    pub active_sets: Vec<Option<Vec<usize>>>,
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    pub block_logdet_h: f64,
    pub block_logdet_s: f64,
}

#[derive(Clone)]
struct ConstrainedWarmStart {
    rho: Array1<f64>,
    block_beta: Vec<Array1<f64>>,
    active_sets: Vec<Option<Vec<usize>>>,
}

#[derive(Clone)]
pub struct BlockwiseFitResult {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub covariance_conditional: Option<Array2<f64>>,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    pub inner_cycles: usize,
    pub converged: bool,
}

fn validate_block_specs(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
    if specs.is_empty() {
        return Err("fit_custom_family requires at least one parameter block".to_string());
    }
    let mut penalty_counts = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let n = spec.design.nrows();
        if spec.offset.len() != n {
            return Err(format!(
                "block {b} offset length mismatch: got {}, expected {}",
                spec.offset.len(),
                n
            ));
        }
        let p = spec.design.ncols();
        if let Some(beta0) = &spec.initial_beta
            && beta0.len() != p
        {
            return Err(format!(
                "block {b} initial_beta length mismatch: got {}, expected {p}",
                beta0.len()
            ));
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(format!(
                "block {b} initial_log_lambdas length {} does not match penalties {}",
                spec.initial_log_lambdas.len(),
                spec.penalties.len()
            ));
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.dim();
            if r != p || c != p {
                return Err(format!(
                    "block {b} penalty {k} must be {p}x{p}, got {r}x{c}"
                ));
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok(penalty_counts)
}

fn flatten_log_lambdas(specs: &[ParameterBlockSpec]) -> Array1<f64> {
    let total = specs
        .iter()
        .map(|s| s.initial_log_lambdas.len())
        .sum::<usize>();
    let mut out = Array1::<f64>::zeros(total);
    let mut at = 0usize;
    for spec in specs {
        let len = spec.initial_log_lambdas.len();
        if len > 0 {
            out.slice_mut(ndarray::s![at..at + len])
                .assign(&spec.initial_log_lambdas);
        }
        at += len;
    }
    out
}

fn split_log_lambdas(
    flat: &Array1<f64>,
    penalty_counts: &[usize],
) -> Result<Vec<Array1<f64>>, String> {
    let expected: usize = penalty_counts.iter().sum();
    if flat.len() != expected {
        return Err(format!(
            "log-lambda length mismatch: got {}, expected {expected}",
            flat.len()
        ));
    }
    let mut out = Vec::with_capacity(penalty_counts.len());
    let mut at = 0usize;
    for &k in penalty_counts {
        out.push(flat.slice(ndarray::s![at..at + k]).to_owned());
        at += k;
    }
    Ok(out)
}

fn build_block_states<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
) -> Result<Vec<ParameterBlockState>, String> {
    let mut states = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let beta = spec
            .initial_beta
            .clone()
            .unwrap_or_else(|| Array1::<f64>::zeros(p));
        let (x_dyn, off_dyn) = family.block_geometry(&states, spec)?;
        if x_dyn.nrows() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic design row mismatch: got {}, expected {}",
                x_dyn.nrows(),
                spec.design.nrows()
            ));
        }
        if x_dyn.ncols() != p {
            return Err(format!(
                "block {b} dynamic design col mismatch: got {}, expected {p}",
                x_dyn.ncols()
            ));
        }
        if off_dyn.len() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic offset length mismatch: got {}, expected {}",
                off_dyn.len(),
                spec.design.nrows()
            ));
        }
        let mut eta = x_dyn.matrix_vector_multiply(&beta);
        eta += &off_dyn;
        states.push(ParameterBlockState { beta, eta });
    }
    Ok(states)
}

fn refresh_all_block_etas<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
) -> Result<(), String> {
    for b in 0..specs.len() {
        let spec = &specs[b];
        let p = states[b].beta.len();
        let (x_dyn, off_dyn) = family.block_geometry(states, spec)?;
        if x_dyn.nrows() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic design row mismatch: got {}, expected {}",
                x_dyn.nrows(),
                spec.design.nrows()
            ));
        }
        if x_dyn.ncols() != p {
            return Err(format!(
                "block {b} dynamic design col mismatch: got {}, expected {p}",
                x_dyn.ncols()
            ));
        }
        if off_dyn.len() != spec.design.nrows() {
            return Err(format!(
                "block {b} dynamic offset length mismatch: got {}, expected {}",
                off_dyn.len(),
                spec.design.nrows()
            ));
        }
        states[b].eta = x_dyn.matrix_vector_multiply(&states[b].beta) + &off_dyn;
    }
    Ok(())
}

fn weighted_normal_equations(
    x: &DesignMatrix,
    w: &Array1<f64>,
    y_star: Option<&Array1<f64>>,
) -> Result<(Array2<f64>, Option<Array1<f64>>), String> {
    let n = x.nrows();
    let p = x.ncols();
    if w.len() != n {
        return Err("weighted normal-equation dimension mismatch".to_string());
    }
    if let Some(y) = y_star
        && y.len() != n
    {
        return Err("weighted RHS dimension mismatch".to_string());
    }

    match x {
        DesignMatrix::Dense(xd) => {
            // Dense path: Xw = diag(sqrt(w)) X, XtWX = Xw'Xw, XtWy = Xw'(sqrt(w) y*)
            let mut xw = xd.clone();
            for i in 0..n {
                let sw = w[i].max(0.0).sqrt();
                if sw != 1.0 {
                    let mut row = xw.row_mut(i);
                    row *= sw;
                }
            }
            let xtwx = fast_ata(&xw);
            let xtwy = y_star.map(|y| {
                let mut y_w = y.clone();
                for i in 0..n {
                    y_w[i] *= w[i].max(0.0).sqrt();
                }
                fast_atv(&xw, &y_w)
            });
            Ok((xtwx, xtwy))
        }
        DesignMatrix::Sparse(xs) => {
            // Sparse path using CSR row iteration; cost is O(sum_i nnz_i^2).
            let csr = xs
                .as_ref()
                .to_row_major()
                .map_err(|_| "failed to obtain CSR view for sparse block design".to_string())?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();

            let mut xtwx = Array2::<f64>::zeros((p, p));
            let mut xtwy = y_star.map(|_| Array1::<f64>::zeros(p));

            for i in 0..n {
                let wi = w[i].max(0.0);
                if wi == 0.0 {
                    continue;
                }
                let start = row_ptr[i];
                let end = row_ptr[i + 1];

                for a_ptr in start..end {
                    let a = col_idx[a_ptr];
                    let xa = vals[a_ptr];

                    if let (Some(y), Some(ref mut rhs)) = (y_star, xtwy.as_mut()) {
                        rhs[a] += wi * xa * y[i];
                    }

                    for b_ptr in a_ptr..end {
                        let b = col_idx[b_ptr];
                        let xb = vals[b_ptr];
                        let v = wi * xa * xb;
                        xtwx[[a, b]] += v;
                        if a != b {
                            xtwx[[b, a]] += v;
                        }
                    }
                }
            }

            Ok((xtwx, xtwy))
        }
    }
}

fn solve_block_weighted_system(
    x: &DesignMatrix,
    y_star: &Array1<f64>,
    w: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y_star.len() != n || w.len() != n {
        return Err("weighted-system dimension mismatch".to_string());
    }

    let (xtwx_base, xtwy_opt) = weighted_normal_equations(x, w, Some(y_star))?;
    let xtwy = xtwy_opt.ok_or_else(|| "missing weighted RHS in block solve".to_string())?;
    let base_ridge = if ridge_policy.include_laplace_hessian {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };

    for retry in 0..8 {
        let ridge = if base_ridge > 0.0 {
            base_ridge * 10f64.powi(retry)
        } else {
            0.0
        };

        let mut xtwx = xtwx_base.clone();
        xtwx += s_lambda;
        for d in 0..p {
            xtwx[[d, d]] += ridge;
        }

        let h = crate::faer_ndarray::FaerArrayView::new(&xtwx);
        let mut rhs = xtwy.clone();
        let mut rhs_mat = FaerMat::zeros(p, 1);
        for i in 0..p {
            rhs_mat[(i, 0)] = rhs[i];
        }

        if let Ok(ch) = FaerLlt::new(h.as_ref(), Side::Lower) {
            ch.solve_in_place(rhs_mat.as_mut());
            for i in 0..p {
                rhs[i] = rhs_mat[(i, 0)];
            }
            if rhs.iter().all(|v| v.is_finite()) {
                return Ok(rhs);
            }
        }

        if !ridge_policy.include_laplace_hessian {
            break;
        }
    }
    Err("block solve failed after ridge retries".to_string())
}

fn solve_spd_system_with_policy(
    lhs: &Array2<f64>,
    rhs: &Array1<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let p = lhs.nrows();
    if lhs.ncols() != p || rhs.len() != p {
        return Err("exact-newton system dimension mismatch".to_string());
    }
    let base_ridge = if ridge_policy.include_laplace_hessian {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };
    for retry in 0..8 {
        let ridge = if base_ridge > 0.0 {
            base_ridge * 10f64.powi(retry)
        } else {
            0.0
        };
        let mut a = lhs.clone();
        for d in 0..p {
            a[[d, d]] += ridge;
        }
        let h = crate::faer_ndarray::FaerArrayView::new(&a);
        let mut rhs_mat = FaerMat::zeros(p, 1);
        for i in 0..p {
            rhs_mat[(i, 0)] = rhs[i];
        }
        if let Ok(ch) = FaerLlt::new(h.as_ref(), Side::Lower) {
            ch.solve_in_place(rhs_mat.as_mut());
        } else if let Ok(ld) = FaerLdlt::new(h.as_ref(), Side::Lower) {
            ld.solve_in_place(rhs_mat.as_mut());
        } else {
            let lb = FaerLblt::new(h.as_ref(), Side::Lower);
            lb.solve_in_place(rhs_mat.as_mut());
        }
        let mut out = Array1::<f64>::zeros(p);
        for i in 0..p {
            out[i] = rhs_mat[(i, 0)];
        }
        if out.iter().all(|v| v.is_finite()) {
            return Ok(out);
        }
        if !ridge_policy.include_laplace_hessian {
            break;
        }
    }
    Err("exact-newton block solve failed after ridge retries".to_string())
}

fn solve_kkt_step(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let p = hessian.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        return Err("constrained KKT step: dimension mismatch".to_string());
    }
    let m = active_a.nrows();
    if m == 0 {
        let p = gradient.len();
        let h_view = FaerArrayView::new(hessian);
        let mut rhs_mat = FaerMat::zeros(p, 1);
        for i in 0..p {
            rhs_mat[(i, 0)] = -gradient[i];
        }
        if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
            ch.solve_in_place(rhs_mat.as_mut());
        } else if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
            ld.solve_in_place(rhs_mat.as_mut());
        } else {
            let lb = FaerLblt::new(h_view.as_ref(), Side::Lower);
            lb.solve_in_place(rhs_mat.as_mut());
        }
        let mut direction = Array1::<f64>::zeros(p);
        for i in 0..p {
            direction[i] = rhs_mat[(i, 0)];
        }
        if !direction.iter().all(|v| v.is_finite()) {
            return Err(
                "constrained unconstrained-step solve produced non-finite values".to_string(),
            );
        }
        return Ok((direction, Array1::zeros(0)));
    }

    let mut kkt = Array2::<f64>::zeros((p + m, p + m));
    kkt.slice_mut(ndarray::s![0..p, 0..p]).assign(hessian);
    kkt.slice_mut(ndarray::s![0..p, p..(p + m)])
        .assign(&active_a.t());
    kkt.slice_mut(ndarray::s![p..(p + m), 0..p])
        .assign(active_a);

    let mut rhs = Array1::<f64>::zeros(p + m);
    for i in 0..p {
        rhs[i] = -gradient[i];
    }

    let kkt_view = FaerArrayView::new(&kkt);
    let lb = FaerLblt::new(kkt_view.as_ref(), Side::Lower);
    let mut rhs_mat = FaerMat::zeros(p + m, 1);
    for i in 0..(p + m) {
        rhs_mat[(i, 0)] = rhs[i];
    }
    lb.solve_in_place(rhs_mat.as_mut());

    let mut direction = Array1::<f64>::zeros(p);
    let mut lambda = Array1::<f64>::zeros(m);
    for i in 0..p {
        direction[i] = rhs_mat[(i, 0)];
    }
    for i in 0..m {
        lambda[i] = rhs_mat[(p + i, 0)];
    }
    if !direction.iter().all(|v| v.is_finite()) || !lambda.iter().all(|v| v.is_finite()) {
        return Err("constrained KKT step produced non-finite values".to_string());
    }
    Ok((direction, lambda))
}

fn check_linear_feasibility(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    tol: f64,
) -> Result<(), String> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return Err("linear constraints: shape mismatch".to_string());
    }
    let slack = constraints.a.dot(beta) - &constraints.b;
    let mut worst = 0.0_f64;
    let mut worst_idx = 0usize;
    for (i, &s) in slack.iter().enumerate() {
        let v = (-s).max(0.0);
        if v > worst {
            worst = v;
            worst_idx = i;
        }
    }
    if worst > tol {
        return Err(format!(
            "infeasible iterate: max(Aβ-b violation)={worst:.3e} at constraint row {worst_idx}"
        ));
    }
    Ok(())
}

fn solve_quadratic_with_linear_constraints(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    warm_active_set: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), String> {
    let p = hessian.nrows();
    let m = constraints.a.nrows();
    if hessian.ncols() != p || rhs.len() != p || beta_start.len() != p {
        return Err("constrained quadratic solve: system dimension mismatch".to_string());
    }
    if constraints.a.ncols() != p || constraints.b.len() != m {
        return Err("constrained quadratic solve: constraint dimension mismatch".to_string());
    }
    let tol_active = 1e-10;
    let tol_step = 1e-12;
    let tol_dual = 1e-10;
    let feas_tol = 1e-8;

    check_linear_feasibility(beta_start, constraints, feas_tol)?;

    // Fast path: unconstrained optimum is feasible.
    if let Ok(beta_unc) = solve_spd_system_with_policy(
        hessian,
        rhs,
        0.0,
        RidgePolicy::explicit_stabilization_pospart(),
    ) {
        let slack = constraints.a.dot(&beta_unc) - &constraints.b;
        if slack.iter().all(|&s| s >= -feas_tol) {
            return Ok((beta_unc, Vec::new()));
        }
    }

    let mut x = beta_start.to_owned();
    let mut slack = constraints.a.dot(&x) - &constraints.b;
    let mut active: Vec<usize> = Vec::new();
    let mut is_active = vec![false; m];
    if let Some(seed) = warm_active_set {
        for &idx in seed {
            if idx < m && !is_active[idx] {
                active.push(idx);
                is_active[idx] = true;
            }
        }
    }
    for i in 0..m {
        if slack[i] <= tol_active && !is_active[i] {
            active.push(i);
            is_active[i] = true;
        }
    }
    for &idx in &active {
        is_active[idx] = true;
    }

    for _ in 0..((p + m + 8) * 4) {
        let gradient = hessian.dot(&x) - rhs;
        let mut a_w = Array2::<f64>::zeros((active.len(), p));
        for (r, &idx) in active.iter().enumerate() {
            a_w.row_mut(r).assign(&constraints.a.row(idx));
        }
        let (direction, lambda_sys) = solve_kkt_step(hessian, &gradient, &a_w)?;
        let step_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= tol_step {
            if active.is_empty() {
                return Ok((x, active));
            }
            // solve_kkt_step returns multipliers from:
            //   H d + A_w^T lambda_sys = -gradient.
            // For constraints A*x >= b, true multipliers satisfy
            //   gradient + H d = A_w^T lambda_true, so lambda_true = -lambda_sys.
            // Release active rows with lambda_true < 0.
            let mut most_negative_true = -tol_dual;
            let mut remove_pos: Option<usize> = None;
            for (pos, &lam_sys) in lambda_sys.iter().enumerate() {
                let lam_true = -lam_sys;
                if lam_true < most_negative_true {
                    most_negative_true = lam_true;
                    remove_pos = Some(pos);
                }
            }
            if let Some(pos) = remove_pos {
                let idx = active.remove(pos);
                is_active[idx] = false;
                continue;
            }
            return Ok((x, active));
        }

        let mut alpha = 1.0_f64;
        let mut entering: Option<usize> = None;
        for i in 0..m {
            if is_active[i] {
                continue;
            }
            let ai = constraints.a.row(i);
            let ai_d = ai.dot(&direction);
            if ai_d < -1e-14 {
                let cand = (slack[i] / (-ai_d)).max(0.0);
                if cand < alpha {
                    alpha = cand;
                    entering = Some(i);
                }
            }
        }

        x += &direction.mapv(|v| alpha * v);
        slack = constraints.a.dot(&x) - &constraints.b;

        if let Some(idx) = entering
            && !is_active[idx]
        {
            active.push(idx);
            is_active[idx] = true;
        }
    }

    Err("constrained quadratic active-set solver failed to converge".to_string())
}

#[inline]
fn effective_solver_ridge(ridge_floor: f64) -> f64 {
    ridge_floor.max(1e-15)
}

fn block_quadratic_penalty(
    beta: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> f64 {
    let mut value = 0.5 * beta.dot(&s_lambda.dot(beta));
    if ridge_policy.include_quadratic_penalty {
        value += 0.5 * ridge * beta.dot(beta);
    }
    value
}

fn total_quadratic_penalty(
    states: &[ParameterBlockState],
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    ridge_policy: RidgePolicy,
) -> f64 {
    let mut total = 0.0;
    for (state, s_lambda) in states.iter().zip(s_lambdas.iter()) {
        total += block_quadratic_penalty(&state.beta, s_lambda, ridge, ridge_policy);
    }
    total
}

fn stable_logdet_with_ridge_policy(
    matrix: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<f64, String> {
    let mut a = matrix.clone();
    let p = a.nrows();
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
    let ridge = if ridge_policy.include_penalty_logdet {
        effective_solver_ridge(ridge_floor)
    } else {
        0.0
    };
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    match ridge_policy.determinant_mode {
        RidgeDeterminantMode::Full => {
            let chol = a.clone().cholesky(Side::Lower).map_err(|_| {
                "cholesky failed while computing full ridge-aware logdet".to_string()
            })?;
            Ok(2.0 * chol.diag().mapv(f64::ln).sum())
        }
        RidgeDeterminantMode::PositivePart => {
            let (evals, _) = crate::faer_ndarray::FaerEigh::eigh(&a, Side::Lower)
                .map_err(|e| format!("eigh failed while computing logdet: {e}"))?;
            let floor = ridge.max(1e-14);
            let mut logdet = 0.0;
            for &ev in &evals {
                if ev > floor {
                    logdet += ev.ln();
                }
            }
            Ok(logdet)
        }
    }
}

fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let (r, c) = a.dim();
    let mut t = 0.0;
    for i in 0..r {
        for j in 0..c {
            t += a[[i, j]] * b[[j, i]];
        }
    }
    t
}

fn matrix_inf_norm(a: &Array2<f64>) -> f64 {
    let (r, c) = a.dim();
    let mut best = 0.0f64;
    for i in 0..r {
        let mut row_sum = 0.0f64;
        for j in 0..c {
            row_sum += a[[i, j]].abs();
        }
        best = best.max(row_sum);
    }
    best
}

fn extract_exact_newton_hessian<F: CustomFamily>(
    family: &F,
    states: &[ParameterBlockState],
    block_idx: usize,
    p: usize,
) -> Result<Array2<f64>, String> {
    let eval = family.evaluate(states)?;
    let work = eval
        .block_working_sets
        .get(block_idx)
        .ok_or_else(|| format!("missing block working set at index {block_idx}"))?;
    match work {
        BlockWorkingSet::ExactNewton {
            gradient: _,
            hessian,
        } => {
            if hessian.nrows() != p || hessian.ncols() != p {
                return Err(format!(
                    "exact-newton Hessian shape mismatch while extracting block {block_idx}: got {}x{}, expected {}x{}",
                    hessian.nrows(),
                    hessian.ncols(),
                    p,
                    p
                ));
            }
            Ok(hessian.clone())
        }
        BlockWorkingSet::Diagonal { .. } => Err(format!(
            "requested exact-newton Hessian for diagonal block {block_idx}"
        )),
    }
}

fn inverse_spd_with_retry(
    matrix: &Array2<f64>,
    base_ridge: f64,
    max_retry: usize,
) -> Result<Array2<f64>, String> {
    let p = matrix.nrows();
    for retry in 0..max_retry {
        let ridge = if base_ridge > 0.0 {
            base_ridge * 10f64.powi(retry as i32)
        } else {
            0.0
        };
        let mut h = matrix.clone();
        for d in 0..p {
            h[[d, d]] += ridge;
        }
        let h_view = FaerArrayView::new(&h);
        if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
            let mut i_mat = FaerMat::zeros(p, p);
            for d in 0..p {
                i_mat[(d, d)] = 1.0;
            }
            ch.solve_in_place(i_mat.as_mut());
            let mut inv = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..p {
                    inv[[i, j]] = i_mat[(i, j)];
                }
            }
            if inv.iter().all(|v| v.is_finite()) {
                return Ok(inv);
            }
        }
    }
    Err("failed to invert SPD system after ridge retries".to_string())
}

fn pinv_positive_part(matrix: &Array2<f64>, ridge_floor: f64) -> Result<Array2<f64>, String> {
    let (evals, evecs) = FaerEigh::eigh(matrix, Side::Lower)
        .map_err(|e| format!("eigh failed in positive-part pseudoinverse: {e}"))?;
    let p = matrix.nrows();
    let max_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let tol = (max_eval * 1e-12).max(ridge_floor.max(1e-14));
    let mut pinv = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        let ev = evals[k];
        if ev > tol {
            let inv_ev = 1.0 / ev;
            for i in 0..p {
                let uik = evecs[(i, k)];
                for j in 0..p {
                    pinv[[i, j]] += inv_ev * uik * evecs[(j, k)];
                }
            }
        }
    }
    Ok(pinv)
}

fn blockwise_logdet_terms<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &mut [ParameterBlockState],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<(f64, f64), String> {
    refresh_all_block_etas(family, specs, states)?;
    if let Some(h_joint) = family.exact_newton_joint_hessian(states)? {
        let ranges = block_param_ranges(specs);
        let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
        if h_joint.nrows() != total || h_joint.ncols() != total {
            return Err(format!(
                "joint exact-newton Hessian shape mismatch in logdet terms: got {}x{}, expected {}x{}",
                h_joint.nrows(),
                h_joint.ncols(),
                total,
                total
            ));
        }
        let mut s_joint = Array2::<f64>::zeros((total, total));
        let mut logdet_s_total = 0.0;
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let p = end - start;
            let lambdas = block_log_lambdas[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }
            s_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .assign(&s_lambda);
            logdet_s_total += stable_logdet_with_ridge_policy(
                &s_lambda,
                options.ridge_floor,
                options.ridge_policy,
            )?;
        }
        let mut h = h_joint;
        h += &s_joint;
        let logdet_h_total =
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?;
        return Ok((logdet_h_total, logdet_s_total));
    }

    let eval = family.evaluate(states)?;
    if eval.block_working_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.block_working_sets.len(),
            specs.len()
        ));
    }

    let mut logdet_h_total = 0.0;
    let mut logdet_s_total = 0.0;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.block_working_sets[b];
        let p = spec.design.ncols();
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => {
                let (x_dyn, _) = family.block_geometry(states, spec)?;
                if x_dyn.ncols() != p {
                    return Err(format!(
                        "block {b} dynamic design col mismatch: got {}, expected {p}",
                        x_dyn.ncols()
                    ));
                }
                let w = working_weights.mapv(|wi| wi.max(options.min_weight));
                let (xtwx, _) = weighted_normal_equations(&x_dyn, &w, None)?;
                xtwx
            }
            BlockWorkingSet::ExactNewton {
                gradient: _,
                hessian,
            } => {
                if hessian.nrows() != p || hessian.ncols() != p {
                    return Err(format!(
                        "block {b} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        p,
                        p
                    ));
                }
                hessian.clone()
            }
        };

        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut h = xtwx;
        h += &s_lambda;
        logdet_h_total +=
            stable_logdet_with_ridge_policy(&h, options.ridge_floor, options.ridge_policy)?;
        logdet_s_total +=
            stable_logdet_with_ridge_policy(&s_lambda, options.ridge_floor, options.ridge_policy)?;
    }
    Ok((logdet_h_total, logdet_s_total))
}

fn inner_blockwise_fit<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, String> {
    let mut states = build_block_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let mut s_lambdas = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let p = spec.design.ncols();
        let lambdas = block_log_lambdas[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }
        s_lambdas.push(s_lambda);
    }
    let ridge = effective_solver_ridge(options.ridge_floor);
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        for (b, beta_seed) in seed.block_beta.iter().enumerate() {
            if beta_seed.len() == states[b].beta.len() {
                states[b].beta.assign(beta_seed);
            }
        }
        cached_active_sets = seed.active_sets.clone();
        refresh_all_block_etas(family, specs, &mut states)?;
    }
    let initial_eval = family.evaluate(&states)?;
    let mut current_penalty =
        total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
    let mut last_objective = -initial_eval.log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;

    for cycle in 0..options.inner_max_cycles {
        let mut max_beta_step = 0.0_f64;

        let mut objective_cycle_prev = last_objective;
        for b in 0..specs.len() {
            // Keep all blocks synchronized with any dynamic geometry.
            refresh_all_block_etas(family, specs, &mut states)?;
            let eval = family.evaluate(&states)?;
            if eval.block_working_sets.len() != specs.len() {
                return Err(format!(
                    "family returned {} block working sets, expected {}",
                    eval.block_working_sets.len(),
                    specs.len()
                ));
            }

            let spec = &specs[b];
            let work = &eval.block_working_sets[b];
            let p = spec.design.ncols();
            let linear_constraints = family.block_linear_constraints(&states, b, spec)?;
            let s_lambda = &s_lambdas[b];

            let beta_new_raw = match work {
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                } => {
                    if working_response.len() != spec.design.nrows()
                        || working_weights.len() != spec.design.nrows()
                    {
                        return Err(format!(
                            "family diagonal working-set size mismatch on block {b} ({})",
                            spec.name
                        ));
                    }

                    let (x_dyn, off_dyn) = family.block_geometry(&states, spec)?;
                    if x_dyn.nrows() != spec.design.nrows() {
                        return Err(format!(
                            "block {b} dynamic design row mismatch: got {}, expected {}",
                            x_dyn.nrows(),
                            spec.design.nrows()
                        ));
                    }
                    if x_dyn.ncols() != p {
                        return Err(format!(
                            "block {b} dynamic design col mismatch: got {}, expected {p}",
                            x_dyn.ncols()
                        ));
                    }
                    if off_dyn.len() != spec.design.nrows() {
                        return Err(format!(
                            "block {b} dynamic offset length mismatch: got {}, expected {}",
                            off_dyn.len(),
                            spec.design.nrows()
                        ));
                    }

                    let mut y_star = working_response.clone();
                    y_star -= &off_dyn;
                    let w_clamped = working_weights.mapv(|wi| wi.max(options.min_weight));
                    if let Some(constraints) = linear_constraints.as_ref() {
                        check_linear_feasibility(&states[b].beta, constraints, 1e-8).map_err(
                            |e| {
                                format!("block {b} ({}) constrained diagonal solve: {e}", spec.name)
                            },
                        )?;
                        let (mut lhs, rhs_opt) =
                            weighted_normal_equations(&x_dyn, &w_clamped, Some(&y_star))?;
                        let rhs = rhs_opt.ok_or_else(|| {
                            "missing weighted RHS in constrained diagonal solve".to_string()
                        })?;
                        lhs += s_lambda;
                        let (beta_constrained, active_set) =
                            solve_quadratic_with_linear_constraints(
                                &lhs,
                                &rhs,
                                &states[b].beta,
                                constraints,
                                cached_active_sets[b].as_deref(),
                            )
                            .map_err(|e| {
                                format!(
                                    "block {b} ({}) constrained diagonal solve failed: {e}",
                                    spec.name
                                )
                            })?;
                        cached_active_sets[b] = Some(active_set);
                        beta_constrained
                    } else {
                        solve_block_weighted_system(
                            &x_dyn,
                            &y_star,
                            &w_clamped,
                            s_lambda,
                            options.ridge_floor,
                            options.ridge_policy,
                        )?
                    }
                }
                BlockWorkingSet::ExactNewton { gradient, hessian } => {
                    if gradient.len() != p {
                        return Err(format!(
                            "block {b} exact-newton gradient length mismatch: got {}, expected {p}",
                            gradient.len()
                        ));
                    }
                    if hessian.nrows() != p || hessian.ncols() != p {
                        return Err(format!(
                            "block {b} exact-newton Hessian shape mismatch: got {}x{}, expected {}x{}",
                            hessian.nrows(),
                            hessian.ncols(),
                            p,
                            p
                        ));
                    }
                    let mut lhs = hessian.clone();
                    lhs += s_lambda;
                    // Newton system in coefficient space:
                    //   β_new = β_old - (H+S)^{-1}(-g + Sβ_old)
                    // Rearranged to a single linear solve:
                    //   (H+S) β_new = H β_old + g
                    // where H = -∇² log L and g = ∇ log L.
                    let mut rhs = hessian.dot(&states[b].beta);
                    rhs += gradient;
                    if let Some(constraints) = linear_constraints.as_ref() {
                        check_linear_feasibility(&states[b].beta, constraints, 1e-8).map_err(
                            |e| {
                                format!(
                                    "block {b} ({}) constrained exact-newton solve: {e}",
                                    spec.name
                                )
                            },
                        )?;
                        let (beta_constrained, active_set) =
                            solve_quadratic_with_linear_constraints(
                                &lhs,
                                &rhs,
                                &states[b].beta,
                                constraints,
                                cached_active_sets[b].as_deref(),
                            )
                            .map_err(|e| {
                                format!(
                                    "block {b} ({}) constrained exact-newton solve failed: {e}",
                                    spec.name
                                )
                            })?;
                        cached_active_sets[b] = Some(active_set);
                        beta_constrained
                    } else {
                        solve_spd_system_with_policy(
                            &lhs,
                            &rhs,
                            options.ridge_floor,
                            options.ridge_policy,
                        )?
                    }
                }
            };
            let beta_new = family.post_update_beta(beta_new_raw)?;
            let beta_old = states[b].beta.clone();
            let delta = &beta_new - &beta_old;
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_beta_step = max_beta_step.max(step);
            if step <= options.inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            let mut accepted = false;
            for bt in 0..8 {
                let alpha = 0.5f64.powi(bt);
                let trial_beta_raw = &beta_old + &delta.mapv(|v| alpha * v);
                let trial_beta = family.post_update_beta(trial_beta_raw)?;
                states[b].beta = trial_beta;
                refresh_all_block_etas(family, specs, &mut states)?;
                let trial_eval = family.evaluate(&states)?;
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let trial_objective = -trial_eval.log_likelihood + trial_penalty;
                if trial_objective.is_finite() && trial_objective <= objective_cycle_prev + 1e-10 {
                    objective_cycle_prev = trial_objective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                states[b].beta = beta_old;
                refresh_all_block_etas(family, specs, &mut states)?;
            }
        }

        refresh_all_block_etas(family, specs, &mut states)?;
        let eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);
        let objective = -eval.log_likelihood + current_penalty;
        let objective_change = (objective - last_objective).abs();
        last_objective = objective;
        cycles_done = cycle + 1;

        let objective_tol = options.inner_tol * (1.0 + objective.abs());
        if max_beta_step <= options.inner_tol && objective_change <= objective_tol {
            converged = true;
            break;
        }
    }

    let final_eval = family.evaluate(&states)?;
    let penalty_value = total_quadratic_penalty(&states, &s_lambdas, ridge, options.ridge_policy);

    let (block_logdet_h, block_logdet_s) =
        blockwise_logdet_terms(family, specs, &mut states, block_log_lambdas, options)?;

    Ok(BlockwiseInnerResult {
        block_states: states,
        active_sets: cached_active_sets,
        log_likelihood: final_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        block_logdet_h,
        block_logdet_s,
    })
}

/// Fit a custom multi-block family.
///
/// Inner loop: cyclic blockwise penalized weighted regressions.
/// Outer loop: joint gradient-only BFGS optimization of all log-smoothing parameters.
fn invalid_outer_bfgs_sample(rho: &Array1<f64>) -> (f64, Array1<f64>) {
    const COST_BARRIER: f64 = 1e50;
    const GRAD_SCALE: f64 = 1e6;

    let mut grad = rho.clone();
    for g in grad.iter_mut() {
        if !g.is_finite() || g.abs() < 1e-6 {
            *g = 1.0;
        }
    }
    grad *= GRAD_SCALE;
    (COST_BARRIER + 0.5 * rho.dot(rho), grad)
}

fn outer_objective_and_gradient<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(f64, Array1<f64>, ConstrainedWarmStart), String> {
    let per_block = split_log_lambdas(rho, penalty_counts)?;
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, warm_start)?;
    let reml_term = if options.use_reml_objective {
        0.5 * (inner.block_logdet_h - inner.block_logdet_s)
    } else {
        0.0
    };
    let objective = -inner.log_likelihood + inner.penalty_value + reml_term;
    let mut grad = Array1::<f64>::zeros(rho.len());

    refresh_all_block_etas(family, specs, &mut inner.block_states)?;
    let eval = family.evaluate(&inner.block_states)?;
    if let Some(h_joint_unpen) = family.exact_newton_joint_hessian(&inner.block_states)? {
        let ranges = block_param_ranges(specs);
        let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
        if h_joint_unpen.nrows() != total || h_joint_unpen.ncols() != total {
            return Err(format!(
                "joint exact-newton Hessian shape mismatch in outer gradient: got {}x{}, expected {}x{}",
                h_joint_unpen.nrows(),
                h_joint_unpen.ncols(),
                total,
                total
            ));
        }
        let beta_flat = flatten_state_betas(&inner.block_states, specs);
        let mut s_joint = Array2::<f64>::zeros((total, total));
        let mut s_pinv_joint = if options.use_reml_objective {
            Some(Array2::<f64>::zeros((total, total)))
        } else {
            None
        };
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let p = end - start;
            let lambdas = per_block[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((p, p));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }
            s_joint
                .slice_mut(ndarray::s![start..end, start..end])
                .assign(&s_lambda);
            if let Some(s_pinv) = s_pinv_joint.as_mut() {
                let s_part = pinv_positive_part(
                    &s_lambda,
                    if options.ridge_policy.include_penalty_logdet {
                        effective_solver_ridge(options.ridge_floor)
                    } else {
                        options.ridge_floor
                    },
                )?;
                s_pinv
                    .slice_mut(ndarray::s![start..end, start..end])
                    .assign(&s_part);
            }
        }
        let mut h_for_logdet = h_joint_unpen.clone();
        h_for_logdet += &s_joint;
        if options.ridge_policy.include_penalty_logdet {
            let ridge = effective_solver_ridge(options.ridge_floor);
            for d in 0..total {
                h_for_logdet[[d, d]] += ridge;
            }
        }
        let h_inv = inverse_spd_with_retry(
            &h_for_logdet,
            effective_solver_ridge(options.ridge_floor),
            8,
        )?;
        let mut at = 0usize;
        for b in 0..specs.len() {
            let spec = &specs[b];
            let (start, end) = ranges[b];
            let beta_block = beta_flat.slice(ndarray::s![start..end]).to_owned();
            let lambdas = per_block[b].mapv(f64::exp);
            for (k, s_k) in spec.penalties.iter().enumerate() {
                let mut a_k = Array2::<f64>::zeros((total, total));
                let local = s_k.mapv(|v| lambdas[k] * v);
                a_k.slice_mut(ndarray::s![start..end, start..end])
                    .assign(&local);
                let a_k_beta = a_k.dot(&beta_flat);
                let g_pen = 0.5 * beta_block.dot(&local.dot(&beta_block));
                let g = if options.use_reml_objective {
                    let g_logh = 0.5 * trace_product(&h_inv, &a_k);
                    let g_logs = 0.5
                        * trace_product(
                            s_pinv_joint
                                .as_ref()
                                .ok_or_else(|| "missing joint S^+ for REML gradient".to_string())?,
                            &a_k,
                        );
                    let u_k = -h_inv.dot(&a_k_beta);
                    let u_norm = u_k.dot(&u_k).sqrt();
                    let g_hbeta = if u_norm <= 1e-14 {
                        0.0
                    } else if let Some(h_rho) = family
                        .exact_newton_joint_hessian_directional_derivative(
                            &inner.block_states,
                            &u_k,
                        )?
                    {
                        if h_rho.nrows() != total || h_rho.ncols() != total {
                            return Err(format!(
                                "joint exact-newton dH shape mismatch: got {}x{}, expected {}x{}",
                                h_rho.nrows(),
                                h_rho.ncols(),
                                total,
                                total
                            ));
                        }
                        0.5 * trace_product(&h_inv, &h_rho)
                    } else {
                        0.0
                    };
                    g_pen + g_logh + g_hbeta - g_logs
                } else {
                    g_pen
                };
                grad[at + k] = g;
            }
            at += spec.penalties.len();
        }
        let warm = ConstrainedWarmStart {
            rho: rho.clone(),
            block_beta: inner
                .block_states
                .iter()
                .map(|st| st.beta.clone())
                .collect(),
            active_sets: inner.active_sets,
        };
        return Ok((objective, grad, warm));
    }
    let mut at = 0usize;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let work = &eval.block_working_sets[b];
        let p = spec.design.ncols();
        let xtwx = match work {
            BlockWorkingSet::Diagonal {
                working_response: _,
                working_weights,
            } => {
                let (x_dyn, _) = family.block_geometry(&inner.block_states, spec)?;
                let w = working_weights.mapv(|wi| wi.max(options.min_weight));
                let (xtwx, _) = weighted_normal_equations(&x_dyn, &w, None)?;
                xtwx
            }
            BlockWorkingSet::ExactNewton {
                gradient: _,
                hessian,
            } => {
                if hessian.nrows() != p || hessian.ncols() != p {
                    return Err(format!(
                        "block {b} exact-newton Hessian shape mismatch in outer gradient: got {}x{}, expected {}x{}",
                        hessian.nrows(),
                        hessian.ncols(),
                        p,
                        p
                    ));
                }
                hessian.clone()
            }
        };

        let lambdas = per_block[b].mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s_lambda.scaled_add(lambdas[k], s);
        }

        let mut h_for_logdet = xtwx;
        h_for_logdet += &s_lambda;
        if options.ridge_policy.include_penalty_logdet {
            let ridge = effective_solver_ridge(options.ridge_floor);
            for d in 0..p {
                h_for_logdet[[d, d]] += ridge;
            }
        }
        let h_inv = inverse_spd_with_retry(
            &h_for_logdet,
            effective_solver_ridge(options.ridge_floor),
            8,
        )?;

        let mut s_for_logdet = s_lambda.clone();
        if options.ridge_policy.include_penalty_logdet {
            let ridge = effective_solver_ridge(options.ridge_floor);
            for d in 0..p {
                s_for_logdet[[d, d]] += ridge;
            }
        }
        let s_pinv = if options.use_reml_objective {
            Some(pinv_positive_part(&s_for_logdet, options.ridge_floor)?)
        } else {
            None
        };

        let beta = &inner.block_states[b].beta;
        for (k, s_k) in spec.penalties.iter().enumerate() {
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            let a_k_beta = a_k.dot(beta);
            let g_pen = 0.5 * beta.dot(&a_k_beta);
            let g = if options.use_reml_objective {
                let g_logh = 0.5 * trace_product(&h_inv, &a_k);
                let g_logs = 0.5
                    * trace_product(
                        s_pinv
                            .as_ref()
                            .ok_or_else(|| "missing S^+ for REML gradient".to_string())?,
                        &a_k,
                    );
                // Exact-Newton hyper-gradient correction (H_rho term)
                //
                // For LAML:
                //   d/dρ [0.5 log|H(ρ)|] = 0.5 tr(H^{-1} dH/dρ),
                //   dH/dρ = A_k + H_ρ,  A_k = dS/dρ_k = λ_k S_k.
                //
                // The usual GAM term g_logh covers 0.5 tr(H^{-1} A_k).
                // When H depends on β (non-Gaussian / transformation models), we add:
                //   g_hbeta = 0.5 tr(H^{-1} H_ρ),
                // with implicit derivative
                //   dβ/dρ_k = -H^{-1} A_k β.
                //
                // We evaluate H_ρ via directional differentiation of the family's
                // exact block Hessian along u_k = dβ/dρ_k.
                let g_hbeta = if matches!(work, BlockWorkingSet::ExactNewton { .. }) {
                    let u_k = -h_inv.dot(&a_k_beta);
                    let u_norm = u_k.dot(&u_k).sqrt();
                    if u_norm <= 1e-14 {
                        0.0
                    } else {
                        let h_rho = if let Some(h_exact) = family
                            .exact_newton_hessian_directional_derivative(
                                &inner.block_states,
                                b,
                                &u_k,
                            )? {
                            if h_exact.nrows() != p || h_exact.ncols() != p {
                                return Err(format!(
                                    "block {b} exact-newton dH shape mismatch: got {}x{}, expected {}x{}",
                                    h_exact.nrows(),
                                    h_exact.ncols(),
                                    p,
                                    p
                                ));
                            }
                            h_exact
                        } else {
                            // Default path: no finite-difference correction.
                            // Fallback to FD only when the local system looks unstable.
                            let cond_proxy =
                                matrix_inf_norm(&h_for_logdet) * matrix_inf_norm(&h_inv);
                            let h_inv_max = h_inv.iter().copied().map(f64::abs).fold(0.0, f64::max);
                            let unstable = !cond_proxy.is_finite()
                                || cond_proxy > 1e10
                                || h_inv_max > 1e8
                                || !g_pen.is_finite()
                                || !g_logh.is_finite()
                                || !g_logs.is_finite();

                            if unstable {
                                let beta_norm = beta.dot(beta).sqrt().max(1.0);
                                let eps = (1e-5 / u_norm).min(1e-3 / beta_norm).max(1e-8);

                                let mut states_plus = inner.block_states.clone();
                                let mut states_minus = inner.block_states.clone();
                                states_plus[b].beta = &states_plus[b].beta + &u_k.mapv(|v| eps * v);
                                states_minus[b].beta =
                                    &states_minus[b].beta - &u_k.mapv(|v| eps * v);

                                refresh_all_block_etas(family, specs, &mut states_plus)?;
                                refresh_all_block_etas(family, specs, &mut states_minus)?;

                                let h_plus =
                                    extract_exact_newton_hessian(family, &states_plus, b, p)?;
                                let h_minus =
                                    extract_exact_newton_hessian(family, &states_minus, b, p)?;
                                (&h_plus - &h_minus).mapv(|v| v / (2.0 * eps))
                            } else {
                                Array2::<f64>::zeros((p, p))
                            }
                        };
                        0.5 * trace_product(&h_inv, &h_rho)
                    }
                } else {
                    0.0
                };

                g_pen + g_logh + g_hbeta - g_logs
            } else {
                g_pen
            };
            grad[at + k] = g;
        }
        at += spec.penalties.len();
    }
    let warm = ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|st| st.beta.clone())
            .collect(),
        active_sets: inner.active_sets,
    };
    Ok((objective, grad, warm))
}

fn block_param_ranges(specs: &[ParameterBlockSpec]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(specs.len());
    let mut at = 0usize;
    for spec in specs {
        let p = spec.design.ncols();
        out.push((at, at + p));
        at += p;
    }
    out
}

fn flatten_state_betas(
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
) -> Array1<f64> {
    let total = specs.iter().map(|s| s.design.ncols()).sum::<usize>();
    let mut beta = Array1::<f64>::zeros(total);
    let ranges = block_param_ranges(specs);
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        beta.slice_mut(ndarray::s![start..end])
            .assign(&states[b].beta);
    }
    beta
}

fn set_states_from_flat_beta(
    states: &mut [ParameterBlockState],
    specs: &[ParameterBlockSpec],
    beta_flat: &Array1<f64>,
) -> Result<(), String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    if beta_flat.len() != total {
        return Err(format!(
            "flat beta length mismatch: got {}, expected {}",
            beta_flat.len(),
            total
        ));
    }
    for (b, (start, end)) in ranges.into_iter().enumerate() {
        states[b]
            .beta
            .assign(&beta_flat.slice(ndarray::s![start..end]).to_owned());
    }
    Ok(())
}

fn penalized_objective_at_beta<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<f64, String> {
    let eval = family.evaluate(states)?;
    if eval.block_working_sets.len() != specs.len() {
        return Err(format!(
            "family returned {} block working sets, expected {}",
            eval.block_working_sets.len(),
            specs.len()
        ));
    }
    let mut penalty = 0.0_f64;
    for b in 0..specs.len() {
        let spec = &specs[b];
        let beta = &states[b].beta;
        let lambdas = per_block_log_lambdas
            .get(b)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(spec.penalties.len()))
            .mapv(f64::exp);
        for (k, s) in spec.penalties.iter().enumerate() {
            if k < lambdas.len() {
                let sb = s.dot(beta);
                penalty += 0.5 * lambdas[k] * beta.dot(&sb);
            }
        }
    }
    Ok(-eval.log_likelihood + penalty)
}

fn compute_joint_hessian_from_objective<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
) -> Result<Array2<f64>, String> {
    let ranges = block_param_ranges(specs);
    let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
    let beta_hat = flatten_state_betas(states, specs);
    let mut h = Array2::<f64>::zeros((total, total));

    let mut states_f0 = states.to_vec();
    set_states_from_flat_beta(&mut states_f0, specs, &beta_hat)?;
    refresh_all_block_etas(family, specs, &mut states_f0)?;
    let f0 = penalized_objective_at_beta(family, specs, &states_f0, per_block_log_lambdas)?;

    let steps = Array1::from_iter(beta_hat.iter().map(|&b| (1e-4 * (1.0 + b.abs())).max(1e-6)));

    for i in 0..total {
        let hi = steps[i];
        let mut bp = beta_hat.clone();
        bp[i] += hi;
        let mut sp = states.to_vec();
        set_states_from_flat_beta(&mut sp, specs, &bp)?;
        refresh_all_block_etas(family, specs, &mut sp)?;
        let fp = penalized_objective_at_beta(family, specs, &sp, per_block_log_lambdas)?;

        let mut bm = beta_hat.clone();
        bm[i] -= hi;
        let mut sm = states.to_vec();
        set_states_from_flat_beta(&mut sm, specs, &bm)?;
        refresh_all_block_etas(family, specs, &mut sm)?;
        let fm = penalized_objective_at_beta(family, specs, &sm, per_block_log_lambdas)?;

        h[[i, i]] = ((fp - 2.0 * f0 + fm) / (hi * hi)).max(0.0);

        for j in 0..i {
            let hj = steps[j];
            let mut bpp = beta_hat.clone();
            bpp[i] += hi;
            bpp[j] += hj;
            let mut spp = states.to_vec();
            set_states_from_flat_beta(&mut spp, specs, &bpp)?;
            refresh_all_block_etas(family, specs, &mut spp)?;
            let fpp = penalized_objective_at_beta(family, specs, &spp, per_block_log_lambdas)?;

            let mut bpm = beta_hat.clone();
            bpm[i] += hi;
            bpm[j] -= hj;
            let mut spm = states.to_vec();
            set_states_from_flat_beta(&mut spm, specs, &bpm)?;
            refresh_all_block_etas(family, specs, &mut spm)?;
            let fpm = penalized_objective_at_beta(family, specs, &spm, per_block_log_lambdas)?;

            let mut bmp = beta_hat.clone();
            bmp[i] -= hi;
            bmp[j] += hj;
            let mut smp = states.to_vec();
            set_states_from_flat_beta(&mut smp, specs, &bmp)?;
            refresh_all_block_etas(family, specs, &mut smp)?;
            let fmp = penalized_objective_at_beta(family, specs, &smp, per_block_log_lambdas)?;

            let mut bmm = beta_hat.clone();
            bmm[i] -= hi;
            bmm[j] -= hj;
            let mut smm = states.to_vec();
            set_states_from_flat_beta(&mut smm, specs, &bmm)?;
            refresh_all_block_etas(family, specs, &mut smm)?;
            let fmm = penalized_objective_at_beta(family, specs, &smm, per_block_log_lambdas)?;

            let hij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            h[[i, j]] = hij;
            h[[j, i]] = hij;
        }
    }
    for i in 0..total {
        h[[i, i]] = h[[i, i]].max(1e-12);
    }
    Ok(h)
}

fn compute_joint_covariance<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    per_block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
) -> Result<Array2<f64>, String> {
    let mut h = if let Some(h_exact) = family.exact_newton_joint_hessian(states)? {
        let ranges = block_param_ranges(specs);
        let total = ranges.last().map(|(_, e)| *e).unwrap_or(0);
        if h_exact.nrows() != total || h_exact.ncols() != total {
            return Err(format!(
                "joint exact-newton Hessian shape mismatch in covariance: got {}x{}, expected {}x{}",
                h_exact.nrows(),
                h_exact.ncols(),
                total,
                total
            ));
        }
        let mut h = h_exact;
        for (b, spec) in specs.iter().enumerate() {
            let (start, end) = ranges[b];
            let lambdas = per_block_log_lambdas[b].mapv(f64::exp);
            let mut s_lambda = Array2::<f64>::zeros((end - start, end - start));
            for (k, s) in spec.penalties.iter().enumerate() {
                s_lambda.scaled_add(lambdas[k], s);
            }
            h.slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(1.0, &s_lambda);
        }
        h
    } else {
        compute_joint_hessian_from_objective(family, specs, states, per_block_log_lambdas)?
    };
    let p_total = h.nrows();
    for i in 0..p_total {
        for j in 0..i {
            let v = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = v;
            h[[j, i]] = v;
        }
    }
    match inverse_spd_with_retry(&h, effective_solver_ridge(options.ridge_floor), 8) {
        Ok(cov) => Ok(cov),
        Err(_) => pinv_positive_part(&h, effective_solver_ridge(options.ridge_floor)),
    }
}

pub fn fit_custom_family<F: CustomFamily>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<BlockwiseFitResult, String> {
    let penalty_counts = validate_block_specs(specs)?;
    let rho0 = flatten_log_lambdas(specs);

    if rho0.is_empty() {
        let mut inner = inner_blockwise_fit(
            family,
            specs,
            &vec![Array1::zeros(0); specs.len()],
            options,
            None,
        )?;
        refresh_all_block_etas(family, specs, &mut inner.block_states)?;
        let covariance_conditional = compute_joint_covariance(
            family,
            specs,
            &inner.block_states,
            &vec![Array1::zeros(0); specs.len()],
            options,
        )
        .ok();
        let reml_term = if options.use_reml_objective {
            0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            0.0
        };
        return Ok(BlockwiseFitResult {
            block_states: inner.block_states,
            log_likelihood: inner.log_likelihood,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            covariance_conditional,
            penalized_objective: -inner.log_likelihood + inner.penalty_value + reml_term,
            outer_iterations: 0,
            inner_cycles: inner.cycles,
            converged: inner.converged,
        });
    }

    let warm_cache = std::sync::Mutex::new(None::<ConstrainedWarmStart>);
    let last_outer_error = std::sync::Mutex::new(None::<String>);
    let lower = Array1::<f64>::from_elem(rho0.len(), -30.0);
    let upper = Array1::<f64>::from_elem(rho0.len(), 30.0);
    let mut last_eval: Option<(Array1<f64>, f64, Array1<f64>)> = None;
    let mut solver = Bfgs::new(rho0.clone(), |x| {
        if let Some((rho_c, cost_c, grad_c)) = &last_eval
            && x.len() == rho_c.len()
            && x.iter()
                .zip(rho_c.iter())
                .all(|(&a, &b)| (a - b).abs() <= 1e-12)
        {
            return (*cost_c, grad_c.clone());
        }

        let cached = warm_cache.lock().ok().and_then(|g| g.clone());
        let sample = match outer_objective_and_gradient(
            family,
            specs,
            options,
            &penalty_counts,
            x,
            cached.as_ref(),
        ) {
            Ok((obj, grad, warm)) if obj.is_finite() && grad.iter().all(|v| v.is_finite()) => {
                if let Ok(mut guard) = warm_cache.lock() {
                    let seed_ok = cached
                        .as_ref()
                        .map(|c| {
                            c.rho.len() == x.len()
                                && c.rho
                                    .iter()
                                    .zip(x.iter())
                                    .all(|(&a, &b)| (a - b).abs() <= 1.5)
                        })
                        .unwrap_or(true);
                    if seed_ok {
                        *guard = Some(warm);
                    } else {
                        *guard = None;
                    }
                }
                if let Ok(mut guard) = last_outer_error.lock() {
                    *guard = None;
                }
                (obj, grad)
            }
            Ok((_obj, _grad, _warm)) => {
                if let Ok(mut guard) = last_outer_error.lock() {
                    *guard = Some(
                        "custom-family outer objective/gradient became non-finite".to_string(),
                    );
                }
                invalid_outer_bfgs_sample(x)
            }
            Err(e) => {
                if let Ok(mut guard) = last_outer_error.lock() {
                    *guard = Some(e);
                }
                invalid_outer_bfgs_sample(x)
            }
        };
        last_eval = Some((x.clone(), sample.0, sample.1.clone()));
        sample
    })
    .with_bounds(lower, upper, 1e-6)
    .with_tolerance(options.outer_tol)
    .with_no_improve_stop(1e-8, 8)
    .with_flat_stall_exit(true, 4)
    .with_curvature_slack_scale(2.0)
    .with_max_iterations(options.outer_max_iter);
    let last_eval_error = || {
        last_outer_error
            .lock()
            .ok()
            .and_then(|g| (*g).clone())
            .map(|e| format!(" last objective error: {e}"))
            .unwrap_or_default()
    };
    let sol = match solver.run() {
        Ok(sol) => sol,
        Err(BfgsError::MaxIterationsReached { last_solution })
        | Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
            if last_solution.final_value.is_finite()
                && last_solution.final_gradient_norm.is_finite()
            {
                log::warn!(
                    "Outer smoothing hit max iterations; using best-so-far solution (iter={}, f={:.6e}, ||g||={:.3e}).",
                    last_solution.iterations,
                    last_solution.final_value,
                    last_solution.final_gradient_norm
                );
                *last_solution
            } else {
                return Err(format!(
                    "outer smoothing optimization failed: MaxIterationsReached.{details}",
                    details = last_eval_error()
                ));
            }
        }
        Err(e) => {
            return Err(format!(
                "outer smoothing optimization failed: {e:?}.{details}",
                details = last_eval_error()
            ));
        }
    };

    let rho_star = sol.final_point;
    let per_block = split_log_lambdas(&rho_star, &penalty_counts)?;
    let final_seed = warm_cache.lock().ok().and_then(|g| g.clone());
    let mut inner = inner_blockwise_fit(family, specs, &per_block, options, final_seed.as_ref())
        .map_err(|e| {
            format!(
                "outer smoothing optimization failed during final inner refit: {e}.{details}",
                details = last_eval_error()
            )
        })?;
    refresh_all_block_etas(family, specs, &mut inner.block_states).map_err(|e| {
        format!(
            "outer smoothing optimization failed during final eta refresh: {e}.{details}",
            details = last_eval_error()
        )
    })?;
    let covariance_conditional =
        compute_joint_covariance(family, specs, &inner.block_states, &per_block, options).ok();

    Ok(BlockwiseFitResult {
        block_states: inner.block_states,
        log_likelihood: inner.log_likelihood,
        log_lambdas: rho_star.clone(),
        lambdas: rho_star.mapv(f64::exp),
        covariance_conditional,
        penalized_objective: if options.use_reml_objective {
            -inner.log_likelihood
                + inner.penalty_value
                + 0.5 * (inner.block_logdet_h - inner.block_logdet_s)
        } else {
            -inner.log_likelihood + inner.penalty_value
        },
        outer_iterations: sol.iterations,
        inner_cycles: inner.cycles,
        converged: inner.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::gamlss::BinomialLocationScaleProbitWiggleFamily;
    use crate::matrix::DesignMatrix;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, array};

    #[derive(Clone)]
    struct OneBlockIdentityFamily;

    impl CustomFamily for OneBlockIdentityFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::ones(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }
    }

    #[derive(Clone)]
    struct OneBlockGaussianFamily {
        y: Array1<f64>,
    }

    impl CustomFamily for OneBlockGaussianFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let eta = &block_states[0].eta;
            let resid = eta - &self.y;
            let ll = -0.5 * resid.dot(&resid);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                block_working_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: self.y.clone(),
                    working_weights: Array1::ones(self.y.len()),
                }],
            })
        }
    }

    #[derive(Clone)]
    struct OneBlockConstrainedExactFamily {
        target: f64,
        lower: f64,
    }

    impl CustomFamily for OneBlockConstrainedExactFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .beta
                .first()
                .copied()
                .ok_or_else(|| "missing coefficient".to_string())?;
            let g = self.target - beta;
            let ll = -0.5 * (beta - self.target) * (beta - self.target);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![g],
                    hessian: array![[1.0]],
                }],
            })
        }

        fn block_linear_constraints(
            &self,
            _block_states: &[ParameterBlockState],
            block_idx: usize,
            _spec: &ParameterBlockSpec,
        ) -> Result<Option<LinearInequalityConstraints>, String> {
            if block_idx != 0 {
                return Ok(None);
            }
            Ok(Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![self.lower],
            }))
        }
    }

    #[derive(Clone)]
    struct PreferJointExactFamily;

    impl CustomFamily for PreferJointExactFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                block_working_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: array![[2.0]],
                }],
            })
        }

        fn exact_newton_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _block_idx: usize,
            _d_beta: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Err(
                "blockwise exact-newton path should not be used when joint path is available"
                    .to_string(),
            )
        }

        fn exact_newton_joint_hessian(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[2.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _block_states: &[ParameterBlockState],
            _d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }
    }

    #[derive(Clone)]
    struct OneBlockAlwaysErrorFamily;

    impl CustomFamily for OneBlockAlwaysErrorFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Err("synthetic outer objective failure: block[0] evaluate()".to_string())
        }
    }

    #[test]
    fn effective_ridge_is_never_below_solver_floor() {
        assert!((effective_solver_ridge(0.0) - 1e-15).abs() < 1e-30);
        assert!((effective_solver_ridge(1e-8) - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn objective_includes_solver_ridge_quadratic_term() {
        // One-parameter block with X=1, y*=1, w=1, no explicit penalties.
        // Inner solve gives beta = 1 / (1 + ridge), so objective should include
        // 0.5 * ridge * beta^2 even when no smoothing penalties are present.
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 1,
            inner_tol: 0.0,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            min_weight: 1e-12,
            ridge_floor: 1e-4,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: false,
        };

        let result = fit_custom_family(&OneBlockIdentityFamily, &[spec], &options)
            .expect("custom family fit should succeed");
        let ridge = effective_solver_ridge(options.ridge_floor);
        let beta = result.block_states[0].beta[0];
        let expected_penalty = 0.5 * ridge * beta * beta;
        assert!(
            (result.penalized_objective - expected_penalty).abs() < 1e-12,
            "penalized objective should equal ridge quadratic term when ll=0 and S=0; got {}, expected {}",
            result.penalized_objective,
            expected_penalty
        );
    }

    #[test]
    fn inner_block_accepts_penalty_improving_step_even_if_loglik_drops() {
        let family = OneBlockGaussianFamily { y: array![1.0] };
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![array![[1.0]]],
            initial_log_lambdas: array![10.0_f64.ln()],
            initial_beta: Some(array![1.0]),
        };
        let options = BlockwiseFitOptions {
            inner_max_cycles: 20,
            inner_tol: 1e-10,
            outer_max_iter: 1,
            outer_tol: 1e-8,
            min_weight: 1e-12,
            ridge_floor: 0.0,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_reml_objective: false,
        };
        let per_block_log_lambdas = vec![array![10.0_f64.ln()]];
        let inner = inner_blockwise_fit(&family, &[spec], &per_block_log_lambdas, &options, None)
            .expect("inner blockwise fit should succeed");

        let beta = inner.block_states[0].beta[0];
        assert!(
            beta < 0.5,
            "beta should shrink toward penalized mode; got {}",
            beta
        );
        assert!(
            inner.log_likelihood < -1e-8,
            "raw log-likelihood should drop for this strongly penalized move; got {}",
            inner.log_likelihood
        );
    }

    #[test]
    fn outer_gradient_matches_finite_difference_for_one_block() {
        let n = 8usize;
        let y = Array1::from_vec(vec![0.4, -0.2, 0.8, 1.0, -0.5, 0.3, 0.1, -0.7]);
        let spec = ParameterBlockSpec {
            name: "b0".to_string(),
            design: DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0)),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.2],
            initial_beta: None,
        };
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize];
        let rho = array![0.1];
        let (f0, g0, _) = outer_objective_and_gradient(
            &OneBlockGaussianFamily { y: y.clone() },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho,
            None,
        )
        .expect("objective/gradient");

        let h = 1e-5;
        let rho_p = array![rho[0] + h];
        let rho_m = array![rho[0] - h];
        let (fp, _, _) = outer_objective_and_gradient(
            &OneBlockGaussianFamily { y: y.clone() },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho_p,
            None,
        )
        .expect("objective+");
        let (fm, _, _) = outer_objective_and_gradient(
            &OneBlockGaussianFamily { y },
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho_m,
            None,
        )
        .expect("objective-");
        let g_fd = (fp - fm) / (2.0 * h);
        let rel = (g0[0] - g_fd).abs() / g_fd.abs().max(1e-8);

        assert!(f0.is_finite());
        assert!(
            rel < 5e-3,
            "outer gradient mismatch: analytic={} fd={} rel={}",
            g0[0],
            g_fd,
            rel
        );
    }

    #[test]
    fn outer_gradient_prefers_joint_exact_path_when_available() {
        let spec = ParameterBlockSpec {
            name: "joint_exact".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            ..BlockwiseFitOptions::default()
        };
        let penalty_counts = vec![1usize];
        let rho = array![0.0];

        let result = outer_objective_and_gradient(
            &PreferJointExactFamily,
            std::slice::from_ref(&spec),
            &options,
            &penalty_counts,
            &rho,
            None,
        );
        assert!(
            result.is_ok(),
            "joint exact path should be preferred over blockwise fallback: {:?}",
            result.err()
        );
    }

    #[test]
    fn outer_laml_gradient_matches_finite_difference_when_joint_exact_path_is_active() {
        let n = 7usize;
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let weights = Array1::from_vec(vec![1.0; n]);
        let threshold_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let log_sigma_design = DesignMatrix::Dense(Array2::from_elem((n, 1), 1.0));
        let threshold_spec = ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.2]),
        };
        let log_sigma_spec = ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design.clone(),
            offset: Array1::zeros(n),
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![-0.2],
            initial_beta: Some(array![-0.1]),
        };
        let q_seed = Array1::linspace(-1.4, 1.4, n);
        let knots = crate::families::gamlss::initialize_wiggle_knots_from_seed(q_seed.view(), 3, 4)
            .expect("knots");
        let wiggle_block = crate::families::gamlss::build_wiggle_block_input_from_knots(
            q_seed.view(),
            &knots,
            3,
            2,
            false,
        )
        .expect("wiggle block");
        let wiggle_spec = ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design.clone(),
            offset: wiggle_block.offset.clone(),
            penalties: wiggle_block.penalties.clone(),
            initial_log_lambdas: array![0.1],
            initial_beta: Some(Array1::from_elem(wiggle_block.design.ncols(), 0.03)),
        };

        let family = BinomialLocationScaleProbitWiggleFamily {
            y,
            weights,
            sigma_min: 0.05,
            sigma_max: 4.0,
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots,
            wiggle_degree: 3,
        };

        let specs = vec![threshold_spec, log_sigma_spec, wiggle_spec];
        let penalty_counts = vec![1usize, 1usize, 1usize];
        let rho = array![0.05, -0.15, 0.1];
        let options = BlockwiseFitOptions {
            use_reml_objective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        };

        let (f0, g0, _) =
            outer_objective_and_gradient(&family, &specs, &options, &penalty_counts, &rho, None)
                .expect("objective/gradient");
        assert!(f0.is_finite());
        assert_eq!(g0.len(), rho.len());

        let h = 1e-5;
        for k in 0..rho.len() {
            let mut rho_p = rho.clone();
            let mut rho_m = rho.clone();
            rho_p[k] += h;
            rho_m[k] -= h;
            let (fp, _, _) = outer_objective_and_gradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_p,
                None,
            )
            .expect("objective+");
            let (fm, _, _) = outer_objective_and_gradient(
                &family,
                &specs,
                &options,
                &penalty_counts,
                &rho_m,
                None,
            )
            .expect("objective-");
            let g_fd = (fp - fm) / (2.0 * h);
            let rel = (g0[k] - g_fd).abs() / g_fd.abs().max(1e-8);
            assert!(
                rel < 2e-2,
                "outer LAML gradient mismatch at {}: analytic={} fd={} rel={}",
                k,
                g0[k],
                g_fd,
                rel
            );
        }
    }

    #[test]
    fn block_solve_sparse_matches_dense() {
        let x_dense = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
            [0.0, 6.0, 0.0]
        ];
        let y_star = array![1.0, -1.0, 0.5, 2.0];
        let w = array![1.0, 0.5, 2.0, 1.5];
        let s_lambda = Array2::<f64>::eye(3) * 0.1;

        let mut triplets = Vec::new();
        for i in 0..x_dense.nrows() {
            for j in 0..x_dense.ncols() {
                let v = x_dense[[i, j]];
                if v != 0.0 {
                    triplets.push(Triplet::new(i, j, v));
                }
            }
        }
        let x_sparse = SparseColMat::try_new_from_triplets(4, 3, &triplets)
            .expect("sparse matrix build should succeed");

        let beta_dense = solve_block_weighted_system(
            &DesignMatrix::Dense(x_dense.clone()),
            &y_star,
            &w,
            &s_lambda,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("dense solve should succeed");

        let beta_sparse = solve_block_weighted_system(
            &DesignMatrix::from(x_sparse),
            &y_star,
            &w,
            &s_lambda,
            1e-12,
            RidgePolicy::explicit_stabilization_pospart(),
        )
        .expect("sparse solve should succeed");

        for j in 0..beta_dense.len() {
            assert!(
                (beta_dense[j] - beta_sparse[j]).abs() < 1e-10,
                "dense/sparse mismatch at {}: {} vs {}",
                j,
                beta_dense[j],
                beta_sparse[j]
            );
        }
    }

    #[test]
    fn exact_newton_block_enforces_linear_constraints() {
        let spec = ParameterBlockSpec {
            name: "exact_block".to_string(),
            design: DesignMatrix::Dense(array![[1.0]]),
            offset: array![0.0],
            penalties: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![1.5]),
        };
        let family = OneBlockConstrainedExactFamily {
            target: 0.0,
            lower: 1.0,
        };
        let fit = fit_custom_family(&family, &[spec], &BlockwiseFitOptions::default())
            .expect("constrained exact-newton fit");
        let beta = fit.block_states[0].beta[0];
        assert!(
            (beta - 1.0).abs() < 1e-8,
            "expected constrained optimum at lower bound, got {beta}"
        );
    }

    #[test]
    fn quadratic_linear_constraints_release_positive_kkt_system_multiplier() {
        // max ll with exact Newton equivalent to minimizing
        // 0.5 * x^2 - rhs*x with rhs=1 under 0 <= x <= 0.1.
        // At x=0, active-set KKT solve gives lambda_sys=+1 for the lower bound,
        // which must be released (lambda_true = -lambda_sys).
        let hessian = array![[1.0]];
        let rhs = array![1.0];
        let beta_start = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [-1.0]],
            b: array![0.0, -0.1],
        };

        let (beta, active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            &constraints,
            None,
        )
        .expect("constrained quadratic solve should succeed");

        assert!(
            (beta[0] - 0.1).abs() <= 1e-10,
            "expected constrained optimum at upper bound 0.1, got {}",
            beta[0]
        );
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn outer_objective_failure_context_is_preserved() {
        // One penalty forces the outer rho optimizer to run, which should now preserve
        // the real evaluation error instead of returning an opaque line-search failure.
        let spec = ParameterBlockSpec {
            name: "err_block".to_string(),
            design: DesignMatrix::Dense(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            penalties: vec![Array2::eye(1)],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
        };
        let options = BlockwiseFitOptions {
            outer_max_iter: 3,
            ..BlockwiseFitOptions::default()
        };
        let err = match fit_custom_family(&OneBlockAlwaysErrorFamily, &[spec], &options) {
            Ok(_) => panic!("fit should fail when family evaluate always errors"),
            Err(e) => e,
        };
        assert!(
            err.contains(
                "last objective error: synthetic outer objective failure: block[0] evaluate()"
            ),
            "expected preserved root-cause context in error, got: {err}"
        );
    }
}
