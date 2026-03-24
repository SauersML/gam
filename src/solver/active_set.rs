use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerArrayView, FaerLinalgError, array1_to_col_matmut};
use crate::linalg::utils::{StableSolver, boundary_hit_step_fraction};
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use faer::{Side, Unbind};
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct LinearInequalityConstraints {
    pub a: Array2<f64>,
    pub b: Array1<f64>,
}

/// KKT diagnostics for inequality-constrained Newton subproblems.
///
/// Constraints are represented as `A * beta >= b` in the same coefficient
/// coordinate system as the returned `beta`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstraintKktDiagnostics {
    /// Number of inequality rows.
    pub n_constraints: usize,
    /// Number of rows considered active (`slack <= active_tolerance`).
    pub n_active: usize,
    /// Maximum primal feasibility violation: `max_i max(0, b_i - a_i^T beta)`.
    pub primal_feasibility: f64,
    /// Maximum dual feasibility violation: `max_i max(0, -lambda_i)`.
    pub dual_feasibility: f64,
    /// Maximum complementarity residual: `max_i |lambda_i * slack_i|`.
    pub complementarity: f64,
    /// Stationarity residual: `||grad - A^T lambda||_inf`.
    pub stationarity: f64,
    /// Tolerance used to classify active constraints from slacks.
    pub active_tolerance: f64,
}

#[inline]
fn array1_is_finite(values: &Array1<f64>) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn solve_newton_direction_dense(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    let factor = StableSolver::new("active-set newton direction")
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    direction_out.assign(gradient);
    let mut rhsview = array1_to_col_matmut(direction_out);
    factor.solve_in_place(rhsview.as_mut());
    direction_out.mapv_inplace(|v| -v);
    if array1_is_finite(direction_out) {
        return Ok(());
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed,
    ))
}

fn solve_symmetric_system(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    if out.len() != rhs.len() {
        unsafe {
            *out = Array1::uninit(rhs.len()).assume_init();
        }
    }
    out.assign(rhs);
    let factor = StableSolver::new("active-set symmetric system")
        .factorize(matrix)
        .map_err(|_| {
            EstimationError::InvalidInput("symmetric system factorization failed".to_string())
        })?;
    out.assign(rhs);
    let mut rhsview = array1_to_col_matmut(out);
    factor.solve_in_place(rhsview.as_mut());
    if array1_is_finite(out) {
        return Ok(());
    }
    Err(EstimationError::InvalidInput(
        "symmetric system solve produced non-finite values".to_string(),
    ))
}

pub(crate) fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    let m = constraints.a.nrows();
    let active_tolerance = 1e-8;

    let mut slack = Array1::<f64>::zeros(m);
    let mut primal_feasibility: f64 = 0.0;
    for i in 0..m {
        let s_i = constraints.a.row(i).dot(beta) - constraints.b[i];
        slack[i] = s_i;
        primal_feasibility = primal_feasibility.max((-s_i).max(0.0));
    }

    let active_idx: Vec<usize> = (0..m).filter(|&i| slack[i] <= active_tolerance).collect();
    let mut lambda = Array1::<f64>::zeros(m);
    if !active_idx.is_empty() {
        let n_active = active_idx.len();
        let p = constraints.a.ncols();
        let mut a_active = Array2::<f64>::zeros((n_active, p));
        for (r, &idx) in active_idx.iter().enumerate() {
            a_active.row_mut(r).assign(&constraints.a.row(idx));
        }
        let mut gram = a_active.dot(&a_active.t());
        let mut rhs = a_active.dot(gradient);
        let ridge_scale = gram.diag().iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let ridge = 1e-12 * ridge_scale.max(1.0);
        for i in 0..n_active {
            gram[[i, i]] += ridge;
        }
        let mut lambda_active = Array1::<f64>::zeros(n_active);
        if solve_symmetric_system(&gram, &rhs, &mut lambda_active).is_ok() {
            for (r, &idx) in active_idx.iter().enumerate() {
                lambda[idx] = lambda_active[r];
            }
        } else {
            rhs.fill(0.0);
        }
    }

    let mut dual_feasibility: f64 = 0.0;
    let mut complementarity: f64 = 0.0;
    for i in 0..m {
        dual_feasibility = dual_feasibility.max((-lambda[i]).max(0.0));
        complementarity = complementarity.max((lambda[i] * slack[i]).abs());
    }
    let stationarity = {
        let mut resid = gradient.to_owned();
        resid -= &constraints.a.t().dot(&lambda);
        resid.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    };

    ConstraintKktDiagnostics {
        n_constraints: m,
        n_active: active_idx.len(),
        primal_feasibility,
        dual_feasibility,
        complementarity,
        stationarity,
        active_tolerance,
    }
}

fn max_linear_constraint_violation(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut worst_row = 0usize;
    for i in 0..constraints.a.nrows() {
        let slack = constraints.a.row(i).dot(beta) - constraints.b[i];
        let viol = (-slack).max(0.0);
        if viol > worst {
            worst = viol;
            worst_row = i;
        }
    }
    (worst, worst_row)
}

pub(crate) fn solve_kkt_direction(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
    active_residual: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    let p = hessian.nrows();
    let m = active_a.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        return Err(EstimationError::InvalidInput(
            "KKT solve dimension mismatch".to_string(),
        ));
    }
    if let Some(residual) = active_residual
        && residual.len() != m
    {
        return Err(EstimationError::InvalidInput(format!(
            "KKT active residual length mismatch: got {}, expected {}",
            residual.len(),
            m
        )));
    }
    if m == 0 {
        let mut d = Array1::<f64>::zeros(p);
        solve_newton_direction_dense(hessian, gradient, &mut d)?;
        return Ok((d, Array1::zeros(0)));
    }
    let mut kkt = Array2::<f64>::zeros((p + m, p + m));
    kkt.slice_mut(s![0..p, 0..p]).assign(hessian);
    kkt.slice_mut(s![0..p, p..(p + m)]).assign(&active_a.t());
    kkt.slice_mut(s![p..(p + m), 0..p]).assign(active_a);

    let mut rhs = Array1::<f64>::zeros(p + m);
    for i in 0..p {
        rhs[i] = -gradient[i];
    }
    if let Some(residual) = active_residual {
        for i in 0..m {
            rhs[p + i] = residual[i];
        }
    }

    let kkt_view = FaerArrayView::new(&kkt);
    let factor = FaerLblt::new(kkt_view.as_ref(), Side::Lower);
    let mut rhs_col = array1_to_col_matmut(&mut rhs);
    factor.solve_in_place(rhs_col.as_mut());
    if !rhs.iter().all(|v| v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "KKT solve produced non-finite values".to_string(),
        ));
    }
    let d = rhs.slice(s![0..p]).to_owned();
    let lambda = rhs.slice(s![p..(p + m)]).to_owned();
    Ok((d, lambda))
}

#[derive(Clone, Debug)]
pub(crate) struct CompressedActiveWorkingSet {
    pub(crate) constraints: LinearInequalityConstraints,
    pub(crate) groups: Vec<Vec<usize>>,
}

pub(crate) fn compress_active_working_set(
    x: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    active: &[usize],
) -> Result<CompressedActiveWorkingSet, EstimationError> {
    const SCALE_TOL: f64 = 1e-14;
    const KEY_TOL: f64 = 1e-10;

    let p = constraints.a.ncols();
    if x.len() != p {
        return Err(EstimationError::InvalidInput(
            "active working-set compression dimension mismatch".to_string(),
        ));
    }

    let mut grouped: BTreeMap<Vec<i64>, (Vec<f64>, f64, Vec<usize>)> = BTreeMap::new();
    let mut fallback_rows: Vec<(Vec<f64>, f64, Vec<usize>)> = Vec::new();

    for (pos, &idx) in active.iter().enumerate() {
        if idx >= constraints.a.nrows() {
            return Err(EstimationError::InvalidInput(format!(
                "active working-set index {} out of bounds for {} constraints",
                idx,
                constraints.a.nrows()
            )));
        }
        let row = constraints.a.row(idx);
        let scale = row.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !scale.is_finite() || scale <= SCALE_TOL {
            let rhs = constraints.b[idx];
            fallback_rows.push((row.to_vec(), rhs, vec![pos]));
            continue;
        }

        let normalized_row: Vec<f64> = row
            .iter()
            .map(|&v| {
                let scaled = v / scale;
                if scaled.abs() <= KEY_TOL { 0.0 } else { scaled }
            })
            .collect();
        let normalized_rhs = constraints.b[idx] / scale;
        let key: Vec<i64> = normalized_row
            .iter()
            .map(|&v| (v / KEY_TOL).round() as i64)
            .collect();

        match grouped.get_mut(&key) {
            Some((row_rep, rhs_max, group_positions)) => {
                if normalized_rhs > *rhs_max {
                    *row_rep = normalized_row;
                    *rhs_max = normalized_rhs;
                }
                group_positions.push(pos);
            }
            None => {
                grouped.insert(key, (normalized_row, normalized_rhs, vec![pos]));
            }
        }
    }

    let nrows = grouped.len() + fallback_rows.len();
    let mut a_out = Array2::<f64>::zeros((nrows, p));
    let mut b_out = Array1::<f64>::zeros(nrows);
    let mut groups_out: Vec<Vec<usize>> = Vec::with_capacity(nrows);

    let mut out_row = 0usize;
    for (_, (row, rhs, positions)) in grouped {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[out_row, j]] = value;
        }
        b_out[out_row] = rhs;
        groups_out.push(positions);
        out_row += 1;
    }
    for (row, rhs, positions) in fallback_rows {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[out_row, j]] = value;
        }
        b_out[out_row] = rhs;
        groups_out.push(positions);
        out_row += 1;
    }

    let (a_out, b_out, groups_out) = rank_reduce_rows_pivoted_qr(a_out, b_out, groups_out);

    Ok(CompressedActiveWorkingSet {
        constraints: LinearInequalityConstraints { a: a_out, b: b_out },
        groups: groups_out,
    })
}

pub(crate) fn rank_reduce_rows_pivoted_qr(
    a: Array2<f64>,
    b: Array1<f64>,
    groups: Vec<Vec<usize>>,
) -> (Array2<f64>, Array1<f64>, Vec<Vec<usize>>) {
    let k = a.nrows();
    let p = a.ncols();
    if k <= 1 {
        return (a, b, groups);
    }

    let mut at_faer = faer::Mat::<f64>::zeros(p, k);
    for i in 0..k {
        for j in 0..p {
            at_faer[(j, i)] = a[[i, j]];
        }
    }

    let qr = at_faer.as_ref().col_piv_qr();
    let r_mat = qr.thin_R();
    let diag_len = r_mat.nrows().min(r_mat.ncols());
    let leading_diag = if diag_len > 0 {
        r_mat[(0, 0)].abs()
    } else {
        0.0
    };

    const RANK_ALPHA: f64 = 100.0;
    let tol = RANK_ALPHA * f64::EPSILON * (k.max(p).max(1) as f64) * leading_diag.max(1.0);

    let rank = (0..diag_len).filter(|&i| r_mat[(i, i)].abs() > tol).count();
    if rank >= k {
        return (a, b, groups);
    }

    let (perm_fwd, _) = qr.P().arrays();
    let kept_orig: Vec<usize> = (0..rank).map(|j| perm_fwd[j].unbound()).collect();
    let dropped_orig: Vec<usize> = (rank..k).map(|j| perm_fwd[j].unbound()).collect();

    let mut orig_to_out = std::collections::HashMap::with_capacity(rank);
    let mut a_out = Array2::<f64>::zeros((rank, p));
    let mut b_out = Array1::<f64>::zeros(rank);
    let mut groups_out: Vec<Vec<usize>> = Vec::with_capacity(rank);
    for (out_idx, &orig_idx) in kept_orig.iter().enumerate() {
        a_out.row_mut(out_idx).assign(&a.row(orig_idx));
        b_out[out_idx] = b[orig_idx];
        groups_out.push(groups[orig_idx].clone());
        orig_to_out.insert(orig_idx, out_idx);
    }

    for &dropped_idx in &dropped_orig {
        let mut best_align = -1.0_f64;
        let mut best_target = kept_orig[0];
        for &kept_idx in &kept_orig {
            let dot = a.row(kept_idx).dot(&a.row(dropped_idx)).abs();
            if dot > best_align {
                best_align = dot;
                best_target = kept_idx;
            }
        }
        let &out_idx = orig_to_out
            .get(&best_target)
            .expect("merge target must be a kept row");
        groups_out[out_idx].extend_from_slice(&groups[dropped_idx]);
    }

    if rank < k {
        log::debug!(
            "rank-reduced active constraints from {} to {} rows (rank deficiency {})",
            k,
            rank,
            k - rank
        );
    }

    (a_out, b_out, groups_out)
}

pub(crate) fn working_set_kkt_diagnostics_from_multipliers(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    working_constraints: &LinearInequalityConstraints,
    lambda_active_true: &Array1<f64>,
    n_total_constraints: usize,
) -> Result<ConstraintKktDiagnostics, EstimationError> {
    let p = working_constraints.a.ncols();
    if x.len() != p || gradient.len() != p {
        return Err(EstimationError::InvalidInput(
            "working-set KKT diagnostic dimension mismatch".to_string(),
        ));
    }
    if lambda_active_true.len() != working_constraints.a.nrows() {
        return Err(EstimationError::InvalidInput(format!(
            "working-set KKT multiplier length mismatch: got {}, expected {}",
            lambda_active_true.len(),
            working_constraints.a.nrows()
        )));
    }
    let m = working_constraints.a.nrows();
    let mut slack = Array1::<f64>::zeros(m);
    let mut primal_feasibility: f64 = 0.0;
    for i in 0..m {
        let s_i = working_constraints.a.row(i).dot(x) - working_constraints.b[i];
        slack[i] = s_i;
        primal_feasibility = primal_feasibility.max((-s_i).max(0.0));
    }

    let lambda = lambda_active_true.to_owned();

    let mut dual_feasibility: f64 = 0.0;
    let mut complementarity: f64 = 0.0;
    for i in 0..m {
        dual_feasibility = dual_feasibility.max((-lambda[i]).max(0.0));
        complementarity = complementarity.max((lambda[i] * slack[i]).abs());
    }
    let stationarity = {
        let mut resid = gradient.to_owned();
        resid -= &working_constraints.a.t().dot(&lambda);
        resid.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    };

    Ok(ConstraintKktDiagnostics {
        n_constraints: n_total_constraints,
        n_active: m,
        primal_feasibility,
        dual_feasibility,
        complementarity,
        stationarity,
        active_tolerance: 1e-8,
    })
}

pub(crate) fn solve_newton_direction_with_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    let p = gradient.len();
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    let m = constraints.a.nrows();
    if constraints.a.ncols() != p || constraints.b.len() != m || beta.len() != p {
        return Err(EstimationError::InvalidInput(format!(
            "linear constraint shape mismatch: A={}x{}, b={}, p={}",
            constraints.a.nrows(),
            constraints.a.ncols(),
            constraints.b.len(),
            p
        )));
    }

    let tol_active = 1e-10;
    let tol_step = 1e-12;
    let tol_dual = 1e-10;
    let mut x = beta.to_owned();
    let mut d_total = Array1::<f64>::zeros(p);
    let mut g_cur = gradient.to_owned();

    let has_active_hint = active_hint
        .as_ref()
        .map(|hint| !hint.is_empty())
        .unwrap_or(false);
    if !has_active_hint && solve_newton_direction_dense(hessian, gradient, direction_out).is_ok() {
        let candidate = beta + &*direction_out;
        let mut feasible = true;
        for i in 0..m {
            let slack = constraints.a.row(i).dot(&candidate) - constraints.b[i];
            if slack < -1e-10 {
                feasible = false;
                break;
            }
        }
        if feasible {
            return Ok(());
        }
    }

    let mut active: Vec<usize> = Vec::new();
    let mut is_active = vec![false; m];
    if let Some(hint) = active_hint.as_ref() {
        for &idx in hint.iter() {
            if idx < m && !is_active[idx] {
                active.push(idx);
                is_active[idx] = true;
            }
        }
    }
    for i in 0..m {
        let slack = constraints.a.row(i).dot(&x) - constraints.b[i];
        if slack <= tol_active && !is_active[i] {
            active.push(i);
            is_active[i] = true;
        }
    }
    let mut last_working_x = x.clone();
    let mut last_working_direction = d_total.clone();
    let mut last_working_gradient = g_cur.clone();
    let mut last_working_active = active.clone();
    let mut last_working_constraints = LinearInequalityConstraints {
        a: Array2::<f64>::zeros((0, p)),
        b: Array1::<f64>::zeros(0),
    };
    let mut last_working_lambda_true = Array1::<f64>::zeros(0);

    for _ in 0..((p + m + 8) * 4) {
        let compressed_working = compress_active_working_set(&x, constraints, &active)?;
        let mut residualw = Array1::<f64>::zeros(compressed_working.constraints.a.nrows());
        for r in 0..compressed_working.constraints.a.nrows() {
            residualw[r] = compressed_working.constraints.b[r]
                - compressed_working.constraints.a.row(r).dot(&x);
        }
        let (d, lambdaw) = solve_kkt_direction(
            hessian,
            &g_cur,
            &compressed_working.constraints.a,
            Some(&residualw),
        )?;
        last_working_x.assign(&x);
        last_working_direction.assign(&d_total);
        last_working_gradient.assign(&g_cur);
        last_working_active.clear();
        last_working_active.extend(active.iter().copied());
        last_working_constraints = LinearInequalityConstraints {
            a: compressed_working.constraints.a.clone(),
            b: compressed_working.constraints.b.clone(),
        };
        last_working_lambda_true = lambdaw.mapv(|lam_sys| -lam_sys);
        let step_norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= tol_step {
            if compressed_working.groups.is_empty() {
                direction_out.assign(&d_total);
                return Ok(());
            }
            let mut remove_pos: Option<usize> = None;
            let mut most_negative_true = -tol_dual;
            for (group_pos, &lam_sys) in lambdaw.iter().enumerate() {
                let lam_true = -lam_sys;
                if lam_true < most_negative_true {
                    most_negative_true = lam_true;
                    remove_pos = Some(group_pos);
                }
            }
            if let Some(group_pos) = remove_pos {
                for &active_pos in compressed_working.groups[group_pos].iter().rev() {
                    let idx = active.remove(active_pos);
                    is_active[idx] = false;
                }
                continue;
            }
            if let Some(hint) = active_hint {
                hint.clear();
                hint.extend(active.iter().copied());
            }
            direction_out.assign(&d_total);
            return Ok(());
        }

        let mut alpha = 1.0_f64;
        let mut entering: Option<usize> = None;
        for i in 0..m {
            if is_active[i] {
                continue;
            }
            let ai = constraints.a.row(i);
            let slack = ai.dot(&x) - constraints.b[i];
            let ai_d = ai.dot(&d);
            if let Some(cand) = boundary_hit_step_fraction(slack, ai_d, alpha) {
                alpha = cand;
                entering = Some(i);
            }
        }

        ndarray::Zip::from(&mut x)
            .and(&mut d_total)
            .and(&d)
            .for_each(|x_i, dt_i, &d_i| {
                let alpha_d = alpha * d_i;
                *x_i += alpha_d;
                *dt_i += alpha_d;
            });
        g_cur = gradient + &hessian.dot(&d_total);

        if active.is_empty() && entering.is_none() {
            if let Some(hint) = active_hint {
                hint.clear();
            }
            direction_out.assign(&d_total);
            return Ok(());
        }

        if let Some(idx) = entering
            && !is_active[idx]
        {
            active.push(idx);
            is_active[idx] = true;
        }
    }

    let (worst, row) = max_linear_constraint_violation(&last_working_x, constraints);
    let working_kkt = working_set_kkt_diagnostics_from_multipliers(
        &last_working_x,
        &last_working_gradient,
        &last_working_constraints,
        &last_working_lambda_true,
        m,
    )?;
    let kkt =
        compute_constraint_kkt_diagnostics(&last_working_x, &last_working_gradient, constraints);
    let grad_inf = last_working_gradient
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let stationarity_rel = working_kkt.stationarity / grad_inf.max(1.0);
    let step_inf = last_working_direction
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let hd_total = hessian.dot(&last_working_direction);
    let predicted_delta = gradient.dot(&last_working_direction)
        + 0.5
            * last_working_direction
                .iter()
                .zip(hd_total.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
    let kkt_strong_ok = (working_kkt.stationarity <= 2e-6 || stationarity_rel <= 2e-6)
        && working_kkt.complementarity <= 1e-6;
    let model_descent_ok = predicted_delta <= -1e-10 * (1.0 + grad_inf * step_inf);
    let near_null_step_ok = step_inf <= 1e-10;
    if worst <= 1e-8
        && working_kkt.dual_feasibility <= 1e-8
        && (kkt_strong_ok || model_descent_ok || near_null_step_ok)
    {
        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend(last_working_active.iter().copied());
        }
        direction_out.assign(&last_working_direction);
        return Ok(());
    }
    Err(EstimationError::ParameterConstraintViolation(format!(
        "linear-constrained Newton active-set failed to converge; max(Aβ-b violation)={worst:.3e} at row {row}; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}, active={}/{}]; diagnostic-reconstruction[dual={:.3e}, stat={:.3e}]",
        working_kkt.primal_feasibility,
        working_kkt.dual_feasibility,
        working_kkt.complementarity,
        working_kkt.stationarity,
        working_kkt.n_active,
        working_kkt.n_constraints,
        kkt.dual_feasibility,
        kkt.stationarity
    )))
}

pub(crate) fn solve_quadratic_with_linear_constraints(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta_start: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    warm_active_set: Option<&[usize]>,
) -> Result<(Array1<f64>, Vec<usize>), EstimationError> {
    if hessian.ncols() != hessian.nrows()
        || rhs.len() != hessian.nrows()
        || beta_start.len() != hessian.nrows()
        || constraints.a.ncols() != hessian.nrows()
    {
        return Err(EstimationError::InvalidInput(
            "constrained quadratic solve: system dimension mismatch".to_string(),
        ));
    }
    let gradient = hessian.dot(beta_start) - rhs;
    let mut delta = Array1::<f64>::zeros(beta_start.len());
    let mut active_hint = warm_active_set.map_or_else(Vec::new, |active| active.to_vec());
    solve_newton_direction_with_linear_constraints(
        hessian,
        &gradient,
        beta_start,
        constraints,
        &mut delta,
        Some(&mut active_hint),
    )?;
    Ok((beta_start + &delta, active_hint))
}
