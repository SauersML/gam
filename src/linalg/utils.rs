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

/// Compute `matrix^{-1}` with a stabilization ridge added solely to make
/// the Cholesky factorization succeed.
///
/// **Stabilization semantics:** the ridge applied here is a
/// [`StabilizationKind::NumericalPerturbation`](crate::types::StabilizationKind)
/// — it does NOT change the model, the objective, the gradient, or
/// anything serialized. The returned matrix is treated by callers as
/// `(matrix)^{-1}`, not `(matrix + δ I)^{-1}`. Callers that genuinely
/// need a model-level prior must build that prior into `matrix` *before*
/// calling this function and pass through a `RidgePassport` /
/// `StabilizationLedger::explicit_prior` so the same δ is also accounted
/// for in objective, logdet, and saved state.
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

/// SPD-only spectrum condition number: λ_max / λ_min on the principal
/// (positive-eigenvalue) spectrum.
///
/// **Invariant:** caller must have already established the matrix is
/// positive definite. For indefinite matrices λ_min may be negative or
/// zero and the ratio max/min becomes meaningless (it can be negative or
/// infinite even when the matrix is well-scaled). Use
/// [`indefinite_safe_condition_number`] when the spectrum sign is unknown.
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

/// Indefinite-safe variant: returns `Err(min_eigenvalue)` when the matrix
/// is not numerically positive definite. The error payload exposes
/// `λ_min` so the caller can route into a stabilization-ledger
/// `bump_with_matrix` call with an inertia-target rule rather than
/// silently consuming a misleading "condition number" value.
pub(crate) fn indefinite_safe_condition_number(matrix: &Array2<f64>) -> Result<f64, f64> {
    let (evals, _) = matrix.eigh(Side::Lower).map_err(|_| f64::NAN)?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in evals.iter() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let pd_floor = max.abs().max(1.0) * 1e-14;
    if min <= pd_floor {
        Err(min)
    } else {
        Ok(max / min)
    }
}

/// Estimate min/max eigenvalues of a symmetric matrix via a short
/// `eigh` call. Used by the inertia-aware stabilization rule below.
/// Returns `None` if the eigensolver fails.
pub(crate) fn symmetric_extremes(matrix: &Array2<f64>) -> Option<(f64, f64)> {
    let (evals, _) = matrix.eigh(Side::Lower).ok()?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in evals.iter() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
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
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction: f64,
    pub condition_estimate: Option<f64>,
}

#[derive(Debug, Clone)]
struct PcgDiagnostics {
    residuals: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

impl PcgDiagnostics {
    fn new(initial_residual_norm: f64) -> Self {
        Self {
            residuals: vec![initial_residual_norm],
            alpha: Vec::new(),
            beta: Vec::new(),
        }
    }

    fn push_iteration(&mut self, alpha: f64, beta: Option<f64>, residual_norm: f64) {
        self.alpha.push(alpha);
        if let Some(beta) = beta {
            self.beta.push(beta);
        }
        self.residuals.push(residual_norm);
    }

    fn condition_estimate(&self) -> Option<f64> {
        // Build the CG Lanczos tridiagonal for the preconditioned operator.
        // For SPD CG, T has diagonal 1/a_i + b_{i-1}/a_{i-1} and off-diagonal
        // sqrt(b_i)/a_i. Its extremal eigenvalues are Ritz estimates.
        let k = self.alpha.len();
        if k == 0 || k > 256 {
            return None;
        }
        let mut t = ndarray::Array2::<f64>::zeros((k, k));
        for i in 0..k {
            let alpha_i = self.alpha[i];
            if !alpha_i.is_finite() || alpha_i <= 0.0 {
                return None;
            }
            let mut diag = 1.0 / alpha_i;
            if i > 0 {
                let beta_prev = self.beta.get(i - 1).copied()?;
                if !beta_prev.is_finite() || beta_prev < 0.0 {
                    return None;
                }
                diag += beta_prev / self.alpha[i - 1];
            }
            t[[i, i]] = diag;
            if i + 1 < k {
                let beta_i = self.beta.get(i).copied().unwrap_or(0.0);
                if !beta_i.is_finite() || beta_i < 0.0 {
                    return None;
                }
                let off = beta_i.sqrt() / alpha_i;
                t[[i, i + 1]] = off;
                t[[i + 1, i]] = off;
            }
        }
        let mut lower = f64::INFINITY;
        let mut upper = 0.0_f64;
        for i in 0..k {
            let radius = if i > 0 { t[[i, i - 1]].abs() } else { 0.0 }
                + if i + 1 < k { t[[i, i + 1]].abs() } else { 0.0 };
            lower = lower.min(t[[i, i]] - radius);
            upper = upper.max(t[[i, i]] + radius);
        }
        if lower.is_finite() && lower > 0.0 && upper.is_finite() && upper > 0.0 {
            Some(upper / lower)
        } else {
            None
        }
    }

    fn info(
        &self,
        iterations: usize,
        converged: bool,
        rhs_norm: f64,
        final_residual_norm: f64,
    ) -> PcgSolveInfo {
        let initial = self.residuals.first().copied().unwrap_or(rhs_norm);
        PcgSolveInfo {
            iterations,
            converged,
            relative_residual_norm: final_residual_norm / rhs_norm.max(1.0),
            initial_residual_norm: initial,
            final_residual_norm,
            residual_reduction: if initial > 0.0 {
                final_residual_norm / initial
            } else {
                0.0
            },
            condition_estimate: self.condition_estimate(),
        }
    }
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
                initial_residual_norm: 0.0,
                final_residual_norm: 0.0,
                residual_reduction: 0.0,
                condition_estimate: None,
            },
        ));
    }

    let tol = rel_tol.max(1e-12) * rhs_norm.max(1.0);
    let mut x = Array1::<f64>::zeros(p);
    let mut r = rhs.clone();
    let mut diagnostics = PcgDiagnostics::new(rhs_norm);

    // Precompute reciprocal preconditioner once. Each PCG iteration applies
    // M^{-1} via a single elementwise multiply (z = inv_m * r).
    //
    // SPD-PCG requires a strictly positive preconditioner (M ≻ 0). A
    // non-positive diagonal entry is a contract violation by the caller —
    // either the matrix is not actually SPD, or it has a structural zero.
    // Silently `abs()`-ing the value (the historical behavior) hides this
    // and produces a "solution" that does not minimize the SPD energy.
    // Instead, fall through to `None` so the caller routes to a
    // direct-factorization or indefinite Krylov path. We still tolerate
    // very small positive values via a 1e-12 floor for numerical noise.
    let mut inv_m = Array1::<f64>::zeros(p);
    let mut bad_diag = false;
    for (slot, &m) in inv_m.iter_mut().zip(preconditioner_diag.iter()) {
        if !m.is_finite() || m < 0.0 {
            // Negative or non-finite preconditioner diagonal violates the
            // SPD-PCG contract (M ≻ 0). Hard error rather than silent
            // `abs()`: caller should route to a direct factorization or
            // indefinite Krylov path. Exactly-zero entries are treated as
            // numerical noise and floored to 1e-12.
            bad_diag = true;
            break;
        }
        *slot = 1.0 / m.max(1e-12);
    }
    if bad_diag {
        log::warn!(
            "SPD PCG rejected: preconditioner diagonal contained a negative or \
             non-finite entry; caller should route to a direct factorization \
             or indefinite Krylov path."
        );
        return None;
    }

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
            diagnostics.push_iteration(alpha, None, r_norm);
            return x
                .iter()
                .all(|v| v.is_finite())
                .then_some((x, diagnostics.info(iter + 1, true, rhs_norm, r_norm)));
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
        diagnostics.push_iteration(alpha, Some(beta), r_norm);
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

/// Write-into variant of `solve_spd_pcg_with_info` that takes an apply closure
/// of the form `Fn(&Array1<f64>, &mut Array1<f64>)` so the matvec can write into
/// a caller-owned buffer. This eliminates the per-iteration `Array1::<f64>`
/// allocation for the matvec result that the legacy closure-returning variant
/// forces. See commit 83369abb for the analogous penalty-vector elimination.
pub fn solve_spd_pcg_with_info_into<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>, &mut Array1<f64>),
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
                initial_residual_norm: 0.0,
                final_residual_norm: 0.0,
                residual_reduction: 0.0,
                condition_estimate: None,
            },
        ));
    }

    let tol = rel_tol.max(1e-12) * rhs_norm.max(1.0);
    let mut x = Array1::<f64>::zeros(p);
    let mut r = rhs.clone();
    let mut diagnostics = PcgDiagnostics::new(rhs_norm);

    if preconditioner_diag
        .iter()
        .any(|&m| !m.is_finite() || m <= 0.0)
    {
        return None;
    }
    let mut inv_m = Array1::<f64>::zeros(p);
    Zip::from(&mut inv_m)
        .and(preconditioner_diag)
        .par_for_each(|inv, &m| {
            *inv = 1.0 / m.max(1e-12);
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

    // Reusable matvec scratch (filled by `apply`).
    let mut ap = Array1::<f64>::zeros(p);

    for iter in 0..max_iter {
        apply(&p_dir, &mut ap);
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
            // Periodic residual refresh: r <- rhs - A x. Reuse `ap` as scratch
            // for A x to avoid an extra allocation.
            apply(&x, &mut ap);
            if ap.len() != p {
                return None;
            }
            r.assign(rhs);
            r.scaled_add(-1.0, &ap);
        }
        let r_norm = r.dot(&r).sqrt();
        if r_norm.is_finite() && r_norm <= tol {
            diagnostics.push_iteration(alpha, None, r_norm);
            return x
                .iter()
                .all(|v| v.is_finite())
                .then_some((x, diagnostics.info(iter + 1, true, rhs_norm, r_norm)));
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
        diagnostics.push_iteration(alpha, Some(beta), r_norm);
        Zip::from(&mut p_dir).and(&z).par_for_each(|pi, &zi| {
            *pi = zi + beta * *pi;
        });
        rz_old = rz_new;
    }
    None
}

/// Write-into variant of `solve_spd_pcg`. Matches `solve_spd_pcg`'s return
/// shape but takes an `apply` closure that writes its result into a caller
/// buffer, enabling the inner-Newton PCG hot path to avoid per-iter
/// `Array1::<f64>` allocations for the matvec output (biobank-scale critical).
pub fn solve_spd_pcg_into<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<Array1<f64>>
where
    F: Fn(&Array1<f64>, &mut Array1<f64>),
{
    solve_spd_pcg_with_info_into(apply, rhs, preconditioner_diag, rel_tol, max_iter)
        .map(|(solution, _)| solution)
}

#[derive(Clone)]
pub(crate) struct RidgePlanner {
    cond_estimate: Option<f64>,
    ridge: f64,
    attempts: usize,
    scale: f64,
    ledger: crate::types::StabilizationLedger,
}

impl RidgePlanner {
    pub(crate) fn new(matrix: &Array2<f64>) -> Self {
        let scale = max_abs_diag(matrix);
        let min_step = scale * 1e-10;
        // Most Hessians factorize on the first attempt. Avoid an eager exact
        // condition-number decomposition here and only pay for spectral
        // diagnostics after an actual factorization failure.
        //
        // RidgePlanner is *strictly* a numerical-perturbation device: the
        // perturbation is applied so a Cholesky factorization succeeds for
        // an inverse / linear solve, and the matrix the caller hands back
        // to the rest of the system is the unperturbed one. The ledger
        // entry below is the canonical record of that semantic.
        let ledger = crate::types::StabilizationLedger::numerical_perturbation(
            min_step,
            crate::types::StabilizationRule::FixedConstant,
            None,
        );
        Self {
            cond_estimate: None,
            ridge: min_step,
            attempts: 0,
            scale,
            ledger,
        }
    }

    pub(crate) fn ridge(&self) -> f64 {
        self.ridge
    }

    /// Canonical accounting record for the ridge currently planned/applied.
    /// Always `NumericalPerturbation`-kind: the planner exists to make a
    /// linear solve succeed, never to change the model.
    #[allow(dead_code)] // exposed for downstream telemetry/tests
    pub(crate) fn ledger(&self) -> crate::types::StabilizationLedger {
        self.ledger
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

        // Primary rule: inertia-target. Estimate λ_min(H) on the unperturbed
        // matrix; pick δ so that λ_min(H + δ I) ≥ τ for an SPD floor τ tied
        // to the matrix scale. This is a defensible "make it positive
        // definite by exactly the amount needed" rule, in contrast with
        // condition-number sqrt heuristics that happen to land in the same
        // ballpark only by coincidence.
        let spd_floor = self.scale * 1e-8;
        let mut chose_via_inertia = false;
        let mut next_ridge = if let Some((lam_min, _lam_max)) = symmetric_extremes(matrix) {
            chose_via_inertia = true;
            // δ = max(min_step, τ - λ_min). Multiply by a small safety
            // factor (1.5×) on the deficit so a single eigensolver round-off
            // does not leave us a hair below τ on the first retry.
            let deficit = (spd_floor - lam_min).max(0.0);
            let proposal = (1.5 * deficit).max(base * 1.5).max(min_step);
            // Cap escalation per attempt so we don't shoot past what's
            // needed when λ_min is wildly negative; the surrounding loop
            // will re-bump up to MAX_FACTORIZATION_ATTEMPTS times.
            proposal.min(base * 10.0)
        } else {
            f64::NAN
        };

        // Fallback rule: condition-number heuristic. Used only when the
        // eigensolver itself failed (rare, usually means a non-finite
        // matrix or extreme scaling).
        if !next_ridge.is_finite() {
            let cond_now = self.estimate_conditionwithridge(matrix, base);
            self.cond_estimate = cond_now;
            next_ridge = if let Some(cond) = cond_now {
                let ratio = cond / HESSIAN_CONDITION_TARGET;
                let mut multiplier = if ratio > 1.0 {
                    ratio.sqrt().clamp(1.5, 10.0)
                } else {
                    (2.0 + self.attempts as f64).clamp(3.0, 10.0)
                };
                let mut proposal = base * multiplier;
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
                (base * 10.0).max(min_step)
            };
        }

        if !next_ridge.is_finite() || next_ridge <= 0.0 {
            next_ridge = self.scale;
        }

        self.ridge = next_ridge;
        // Reflect the new escalation state in the ledger so any downstream
        // consumer (telemetry, debug logging) sees a consistent record.
        self.ledger = crate::types::StabilizationLedger::numerical_perturbation(
            self.ridge,
            if chose_via_inertia {
                crate::types::StabilizationRule::InertiaTarget { spd_floor }
            } else {
                crate::types::StabilizationRule::BackoffEscalation {
                    attempts: self.attempts,
                }
            },
            None,
        );
    }

    pub(crate) fn attempts(&self) -> usize {
        self.attempts
    }
}

#[cfg(test)]
mod tests {
    use super::{
        boundary_hit_step_fraction, solve_spd_pcg, solve_spd_pcg_into, solve_spd_pcg_with_info,
        solve_spd_pcg_with_info_into,
    };
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

    #[test]
    fn solve_spd_pcg_into_matches_legacy_owned_variant() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]];
        let b = array![1.0, 2.0, -0.5];
        let m = Array1::from_vec(vec![4.0, 3.0, 2.0]);
        let owned = solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-12, 50).expect("legacy pcg");
        let writeinto = solve_spd_pcg_into(
            |v, out| {
                let prod = h.dot(v);
                out.assign(&prod);
            },
            &b,
            &m,
            1e-12,
            50,
        )
        .expect("write-into pcg");
        assert_eq!(owned.len(), writeinto.len());
        for (a, b) in owned.iter().zip(writeinto.iter()) {
            assert!((a - b).abs() < 1e-10, "owned={a} writeinto={b}");
        }
    }

    #[test]
    fn matrix_free_qp_beta_matches_dense_reference_with_diagnostics() {
        // Small synthetic stand-in for the FLEX marginal-slope joint system:
        // a coupled SPD Hessian plus a penalty/ridge Jacobi preconditioner. The
        // matrix-free solve must return the same beta as the dense reference,
        // while surfacing bounded iteration/residual diagnostics for cycle-0
        // triage.
        let h = array![
            [12.0, 2.0, 0.5, 0.0],
            [2.0, 9.0, 1.25, 0.25],
            [0.5, 1.25, 7.0, 1.5],
            [0.0, 0.25, 1.5, 5.0],
        ];
        let rhs = array![1.0, -0.5, 2.0, 0.75];
        let precond = h.diag().to_owned();
        let factor = super::StableSolver::new("synthetic dense reference")
            .factorize(&h)
            .expect("dense SPD reference");
        let mut dense = rhs.clone();
        let mut dense_view = crate::faer_ndarray::array1_to_col_matmut(&mut dense);
        factor.solve_in_place(dense_view.as_mut());
        let (pcg, info) = solve_spd_pcg_with_info_into(
            |v, out| {
                let prod = h.dot(v);
                out.assign(&prod);
            },
            &rhs,
            &precond,
            1e-12,
            4 * rhs.len(),
        )
        .expect("matrix-free pcg");

        assert!(info.converged);
        assert!(info.iterations <= 4 * rhs.len());
        assert!(info.final_residual_norm < info.initial_residual_norm);
        assert!(info.residual_reduction < 1e-10);
        assert!(info.condition_estimate.is_some());
        for (reference, actual) in dense.iter().zip(pcg.iter()) {
            assert!(
                (reference - actual).abs() < 1e-10,
                "dense={reference} pcg={actual}"
            );
        }
    }

    #[test]
    fn solve_spd_pcg_into_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(
            solve_spd_pcg_with_info_into(
                |v, out| {
                    let prod = h.dot(v);
                    out.assign(&prod);
                },
                &b,
                &m,
                1e-10,
                0,
            )
            .is_none()
        );
        assert!(
            solve_spd_pcg_into(
                |v, out| {
                    let prod = h.dot(v);
                    out.assign(&prod);
                },
                &b,
                &m,
                1e-10,
                0,
            )
            .is_none()
        );
    }
}
