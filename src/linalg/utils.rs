use crate::construction::calculate_condition_number;
use crate::faer_ndarray::FaerEigh;
use crate::faer_ndarray::{
    FaerArrayView, FaerLinalgError, array2_to_matmut, factorize_symmetricwith_fallback,
};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, Zip};

const HESSIAN_CONDITION_TARGET: f64 = 1e10;
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
const MAX_SOLVE_RETRIES: usize = 8;

#[derive(Default, Clone, Copy)]
pub(crate) struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    pub(crate) fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    pub(crate) fn sum(self) -> f64 {
        self.sum
    }
}

pub(crate) fn matrix_inversewith_regularization(
    matrix: &Array2<f64>,
    label: &str,
) -> Option<Array2<f64>> {
    StableSolver::new(label).inversewith_regularization(matrix)
}

pub(crate) struct StableSolver<'a> {
    label: &'a str,
}

impl<'a> StableSolver<'a> {
    pub(crate) fn new(label: &'a str) -> Self {
        Self { label }
    }

    pub(crate) fn factorize(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<crate::faer_ndarray::FaerSymmetricFactor, FaerLinalgError> {
        let view = FaerArrayView::new(matrix);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower)
    }

    pub(crate) fn inversewith_regularization(&self, matrix: &Array2<f64>) -> Option<Array2<f64>> {
        let p = matrix.nrows();
        if p == 0 || matrix.ncols() != p {
            return None;
        }

        let mut planner = RidgePlanner::new(matrix);
        let (factor, _, regularized) = self.factorize_with_ridge_plan(matrix, &mut planner)?;
        let mut inv = Array2::<f64>::eye(p);
        let mut invview = array2_to_matmut(&mut inv);
        factor.solve_in_place(invview.as_mut());

        if !inv.iter().all(|v| v.is_finite()) {
            log::warn!("Non-finite inverse produced for {}", self.label);
            return None;
        }

        // Numerical solves can leave tiny asymmetry; enforce symmetry explicitly.
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (inv[[i, j]] + inv[[j, i]]);
                inv[[i, j]] = avg;
                inv[[j, i]] = avg;
            }
        }
        debug_assert_eq!(regularized.nrows(), p);
        Some(inv)
    }

    pub(crate) fn solvevectorwithridge_retries(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        baseridge: f64,
    ) -> Option<Array1<f64>> {
        let p = matrix.nrows();
        if matrix.ncols() != p || rhs.len() != p {
            return None;
        }

        for retry in 0..MAX_SOLVE_RETRIES {
            let ridge = if baseridge > 0.0 {
                baseridge * 10f64.powi(retry as i32)
            } else {
                0.0
            };
            let h = addridge(matrix, ridge);
            let factor = match self.factorize(&h) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut out = rhs.clone();
            let mut out_mat = crate::faer_ndarray::array1_to_col_matmut(&mut out);
            factor.solve_in_place(out_mat.as_mut());
            if out.iter().all(|v| v.is_finite()) {
                return Some(out);
            }
        }
        None
    }

    fn factorize_with_ridge_plan(
        &self,
        matrix: &Array2<f64>,
        planner: &mut RidgePlanner,
    ) -> Option<(crate::faer_ndarray::FaerSymmetricFactor, f64, Array2<f64>)> {
        loop {
            let ridge = planner.ridge();
            let h_eff = addridge(matrix, ridge);
            if let Ok(factor) = self.factorize(&h_eff) {
                return Some((factor, ridge, h_eff));
            }
            if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                log::warn!(
                    "Failed to factorize {} after ridge {:.3e}",
                    self.label,
                    ridge
                );
                return None;
            }
            planner.bumpwith_matrix(matrix);
        }
    }
}

pub(crate) fn max_abs_diag(matrix: &Array2<f64>) -> f64 {
    matrix
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0)
}

pub(crate) fn row_mismatch_message(
    y_len: usize,
    w_len: usize,
    x_rows: usize,
    offset_len: usize,
) -> Option<String> {
    if y_len == w_len && y_len == x_rows && y_len == offset_len {
        None
    } else {
        Some(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y_len, w_len, x_rows, offset_len
        ))
    }
}

pub(crate) fn predict_gam_dimension_mismatch_message(
    x_rows: usize,
    x_cols: usize,
    beta_len: usize,
    offset_len: usize,
) -> Option<String> {
    if x_cols != beta_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} columns but beta has length {}",
            x_cols, beta_len
        ));
    }
    if x_rows != offset_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} rows but offset has length {}",
            x_rows, offset_len
        ));
    }
    None
}

pub(crate) fn add_relative_diag_ridge(matrix: &mut Array2<f64>, scale: f64, floor: f64) -> f64 {
    let ridge = scale
        * matrix
            .diag()
            .iter()
            .map(|&value| value.abs())
            .fold(0.0, f64::max)
            .max(floor);
    for idx in 0..matrix.nrows() {
        matrix[[idx, idx]] += ridge;
    }
    ridge
}

pub(crate) fn boundary_hit_indices(
    values: ArrayView1<'_, f64>,
    bound: f64,
    tolerance: f64,
) -> (Vec<usize>, Vec<usize>) {
    let at_lower = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value <= -bound + tolerance).then_some(idx))
        .collect();
    let at_upper = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value >= bound - tolerance).then_some(idx))
        .collect();
    (at_lower, at_upper)
}

pub(crate) fn symmetric_spectrum_condition_number(matrix: &Array2<f64>) -> f64 {
    matrix
        .eigh(Side::Lower)
        .ok()
        .map(|(evals, _)| {
            let min = evals
                .iter()
                .fold(f64::INFINITY, |acc, &value| acc.min(value));
            let max = evals
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
            max / min.max(1e-12)
        })
        .unwrap_or(f64::NAN)
}

/// Enforce exact symmetry on a square matrix by averaging off-diagonal pairs.
pub(crate) fn enforce_symmetry(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    debug_assert_eq!(n, matrix.ncols());
    for i in 0..n {
        for j in i + 1..n {
            let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = avg;
            matrix[[j, i]] = avg;
        }
    }
}

pub(crate) fn addridge(matrix: &Array2<f64>, ridge: f64) -> Array2<f64> {
    if ridge <= 0.0 {
        return matrix.clone();
    }
    let mut regularized = matrix.clone();
    let n = regularized.nrows();
    for i in 0..n {
        regularized[[i, i]] += ridge;
    }
    regularized
}

pub(crate) fn boundary_hit_step_fraction(
    slack: f64,
    directional_slack_change: f64,
    current_step_limit: f64,
) -> Option<f64> {
    if !slack.is_finite()
        || !directional_slack_change.is_finite()
        || !current_step_limit.is_finite()
        || current_step_limit <= 0.0
    {
        return None;
    }

    let scale = slack
        .abs()
        .max(directional_slack_change.abs())
        .max(current_step_limit.abs())
        .max(1.0);
    let directional_tol = (64.0 * f64::EPSILON * scale).max(1e-14);
    if directional_slack_change >= -directional_tol {
        return None;
    }

    let step = (slack / -directional_slack_change).max(0.0);
    if step.is_finite() && step < current_step_limit {
        return Some(step);
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgSolveInfo {
    pub iterations: usize,
    pub converged: bool,
    pub relative_residual_norm: f64,
}

pub fn solve_spd_pcg_with_info<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = rhs.len();
    if p == 0 || preconditioner_diag.len() != p || max_iter == 0 {
        return None;
    }
    let rhs_norm = rhs.dot(rhs).sqrt();
    if !rhs_norm.is_finite() {
        return None;
    }
    if rhs_norm == 0.0 {
        return Some((
            Array1::<f64>::zeros(p),
            PcgSolveInfo {
                iterations: 0,
                converged: true,
                relative_residual_norm: 0.0,
            },
        ));
    }

    let tol = rel_tol.max(1e-12) * rhs_norm.max(1.0);
    let mut x = Array1::<f64>::zeros(p);
    let mut r = rhs.clone();

    // Precompute reciprocal preconditioner once. Each PCG iteration applies
    // M^{-1} via a single elementwise multiply (z = inv_m * r), avoiding the
    // per-element `.abs().max(1e-12)` and division on every iteration. The
    // floor mirrors the previous guard against zero/negative diagonals.
    let mut inv_m = Array1::<f64>::zeros(p);
    Zip::from(&mut inv_m)
        .and(preconditioner_diag)
        .par_for_each(|inv, &m| {
            *inv = 1.0 / m.abs().max(1e-12);
        });

    let mut z = Array1::<f64>::zeros(p);
    Zip::from(&mut z)
        .and(&r)
        .and(&inv_m)
        .par_for_each(|zi, &ri, &im| {
            *zi = ri * im;
        });
    let mut p_dir = z.clone();
    let mut rz_old = r.dot(&z);
    if !rz_old.is_finite() || rz_old <= 0.0 {
        return None;
    }

    for iter in 0..max_iter {
        let ap = apply(&p_dir);
        if ap.len() != p {
            return None;
        }
        let denom = p_dir.dot(&ap);
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        let alpha = rz_old / denom;
        if !alpha.is_finite() {
            return None;
        }
        x.scaled_add(alpha, &p_dir);
        r.scaled_add(-alpha, &ap);
        if (iter + 1) % 32 == 0 {
            // Periodic residual refresh: r <- rhs - A x. Done in-place via
            // assign + scaled_add to avoid the prior fresh-allocation pattern
            // (`r = rhs - &ax;`) inside the hot loop.
            let ax = apply(&x);
            if ax.len() != p {
                return None;
            }
            r.assign(rhs);
            r.scaled_add(-1.0, &ax);
        }
        let r_norm = r.dot(&r).sqrt();
        if r_norm.is_finite() && r_norm <= tol {
            return x.iter().all(|v| v.is_finite()).then_some((
                x,
                PcgSolveInfo {
                    iterations: iter + 1,
                    converged: true,
                    relative_residual_norm: r_norm / rhs_norm.max(1.0),
                },
            ));
        }
        Zip::from(&mut z)
            .and(&r)
            .and(&inv_m)
            .par_for_each(|zi, &ri, &im| {
                *zi = ri * im;
            });
        let rz_new = r.dot(&z);
        if !rz_new.is_finite() || rz_new <= 0.0 {
            return None;
        }
        let beta = rz_new / rz_old;
        if !beta.is_finite() {
            return None;
        }
        // p <- z + beta * p (fused, SIMD-friendly via ndarray::Zip; parallel
        // over coefficient dimension at biobank-scale p).
        Zip::from(&mut p_dir).and(&z).par_for_each(|pi, &zi| {
            *pi = zi + beta * *pi;
        });
        rz_old = rz_new;
    }
    None
}

pub fn solve_spd_pcg<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    solve_spd_pcg_with_info(apply, rhs, preconditioner_diag, rel_tol, max_iter)
        .map(|(solution, _)| solution)
}

#[derive(Clone)]
pub(crate) struct RidgePlanner {
    cond_estimate: Option<f64>,
    ridge: f64,
    attempts: usize,
    scale: f64,
}

impl RidgePlanner {
    pub(crate) fn new(matrix: &Array2<f64>) -> Self {
        let scale = max_abs_diag(matrix);
        let min_step = scale * 1e-10;
        // Most Hessians factorize on the first attempt. Avoid an eager exact
        // condition-number decomposition here and only pay for spectral
        // diagnostics after an actual factorization failure.
        Self {
            cond_estimate: None,
            ridge: min_step,
            attempts: 0,
            scale,
        }
    }

    pub(crate) fn ridge(&self) -> f64 {
        self.ridge
    }

    #[inline]
    fn estimate_conditionwithridge(&self, matrix: &Array2<f64>, ridge: f64) -> Option<f64> {
        let regularized = if ridge > 0.0 {
            addridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        calculate_condition_number(&regularized)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0)
    }

    pub(crate) fn bumpwith_matrix(&mut self, matrix: &Array2<f64>) {
        self.attempts += 1;
        let min_step = self.scale * 1e-10;
        let base = self.ridge.max(min_step);

        // Estimate conditioning at the current ridge level.
        let cond_now = self.estimate_conditionwithridge(matrix, base);
        self.cond_estimate = cond_now;

        self.ridge = if let Some(cond) = cond_now {
            let ratio = cond / HESSIAN_CONDITION_TARGET;
            // Primary update from condition feedback.
            // sqrt-ratio avoids wild overshoot while still scaling with severity.
            let mut multiplier = if ratio > 1.0 {
                ratio.sqrt().clamp(1.5, 10.0)
            } else {
                // Factorization failed despite "acceptable" condition number.
                // This usually indicates indefiniteness/numerical fragility, so
                // use a stronger fallback than ×2, increasing with attempts.
                (2.0 + self.attempts as f64).clamp(3.0, 10.0)
            };

            let mut proposal = base * multiplier;
            // Verify whether the proposal actually improves condition enough.
            // If not, escalate once more before returning.
            if let Some(cond_next) = self.estimate_conditionwithridge(matrix, proposal)
                && cond_next > cond * 0.9
                && ratio > 1.0
            {
                multiplier = (multiplier * 1.8).clamp(2.0, 10.0);
                proposal = base * multiplier;
            }
            proposal.max(min_step)
        } else if self.ridge <= 0.0 {
            min_step
        } else {
            // Condition estimate unavailable: geometric fallback.
            (base * 10.0).max(min_step)
        };

        if !self.ridge.is_finite() || self.ridge <= 0.0 {
            self.ridge = self.scale;
        }
    }

    pub(crate) fn attempts(&self) -> usize {
        self.attempts
    }
}

#[cfg(test)]
mod tests {
    use super::{boundary_hit_step_fraction, solve_spd_pcg, solve_spd_pcg_with_info};
    use ndarray::{Array1, array};

    #[test]
    fn boundary_hit_step_fraction_ignores_near_tangential_direction() {
        let step = boundary_hit_step_fraction(1.0, -1e-16, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn boundary_hit_step_fraction_returns_first_finite_hit() {
        let step = boundary_hit_step_fraction(0.25, -0.5, 1.0);
        assert_eq!(step, Some(0.5));
    }

    #[test]
    fn boundary_hit_step_fraction_rejects_non_finite_candidate() {
        let step = boundary_hit_step_fraction(1.0, f64::NEG_INFINITY, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn solve_spd_pcg_matches_reference_solution() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        let x = solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 20).expect("pcg solve");
        assert!((x[0] - 0.0909090909).abs() < 1e-8);
        assert!((x[1] - 0.6363636363).abs() < 1e-8);
    }

    #[test]
    fn solve_spd_pcg_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(solve_spd_pcg_with_info(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
        assert!(solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
    }
}
