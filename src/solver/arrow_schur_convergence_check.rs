use ndarray::{Array1, Array2, ArrayView1};

use crate::solver::arrow_schur::{ArrowSchurError, ArrowSchurSystem};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowConvergenceIssueSeverity {
    Warning,
    Failure,
}

#[derive(Debug, Clone)]
pub struct ArrowConvergenceIssue {
    pub severity: ArrowConvergenceIssueSeverity,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct LatentCompactBox {
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct ArrowSchurConvergenceCheckOptions {
    pub proximal_ridge_floor: f64,
    pub max_condition_number: f64,
    pub penalty_gradient_lipschitz_bound: Option<f64>,
    pub adaptive_proximal_correction_enabled: bool,
    pub require_compact_box: bool,
    pub fail_on_warnings: bool,
}

impl Default for ArrowSchurConvergenceCheckOptions {
    fn default() -> Self {
        Self {
            proximal_ridge_floor: 1e-8,
            max_condition_number: 1e12,
            penalty_gradient_lipschitz_bound: None,
            adaptive_proximal_correction_enabled: true,
            require_compact_box: true,
            fail_on_warnings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArrowSchurConvergenceReport {
    pub issues: Vec<ArrowConvergenceIssue>,
    pub min_row_eigenvalue: f64,
    pub schur_condition_number: f64,
}

impl ArrowSchurConvergenceReport {
    pub fn has_failures(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == ArrowConvergenceIssueSeverity::Failure)
    }

    pub fn warn_or_fail(&self) -> Result<(), ArrowSchurError> {
        for issue in self.issues.iter() {
            match issue.severity {
                ArrowConvergenceIssueSeverity::Warning => {
                    log::warn!("[arrow-Schur convergence] {}", issue.message);
                }
                ArrowConvergenceIssueSeverity::Failure => {
                    return Err(ArrowSchurError::AdaptiveCorrectionFailed {
                        reason: issue.message.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

pub fn check_arrow_schur_fit_start(
    sys: &ArrowSchurSystem,
    latent_flat: Option<ArrayView1<'_, f64>>,
    compact_box: Option<&LatentCompactBox>,
    options: &ArrowSchurConvergenceCheckOptions,
) -> ArrowSchurConvergenceReport {
    let mut issues = Vec::new();
    if !options.adaptive_proximal_correction_enabled {
        issues.push(issue(
            ArrowConvergenceIssueSeverity::Failure,
            "undamped full-step arrow-Schur Newton has a two-cycle counterexample; enable adaptive proximal correction",
        ));
    }
    check_latent_compactness(latent_flat, compact_box, options, &mut issues);

    if options.penalty_gradient_lipschitz_bound.is_none() {
        let severity = if options.fail_on_warnings {
            ArrowConvergenceIssueSeverity::Failure
        } else {
            ArrowConvergenceIssueSeverity::Warning
        };
        issues.push(issue(
            severity,
            "no penalty-gradient Lipschitz certificate was supplied; fit-start can only check finite local Hessian blocks",
        ));
    }

    let ridge = options.proximal_ridge_floor.max(0.0);
    let min_row_eigenvalue = min_shifted_row_eigenvalue(sys, ridge, &mut issues);
    let schur_condition_number = schur_condition_number(sys, ridge, options, &mut issues);

    ArrowSchurConvergenceReport {
        issues,
        min_row_eigenvalue,
        schur_condition_number,
    }
}

fn issue(severity: ArrowConvergenceIssueSeverity, message: &str) -> ArrowConvergenceIssue {
    ArrowConvergenceIssue {
        severity,
        message: message.to_string(),
    }
}

fn check_latent_compactness(
    latent_flat: Option<ArrayView1<'_, f64>>,
    compact_box: Option<&LatentCompactBox>,
    options: &ArrowSchurConvergenceCheckOptions,
    issues: &mut Vec<ArrowConvergenceIssue>,
) {
    let missing_severity = if options.require_compact_box || options.fail_on_warnings {
        ArrowConvergenceIssueSeverity::Failure
    } else {
        ArrowConvergenceIssueSeverity::Warning
    };
    let Some(latent) = latent_flat else {
        issues.push(issue(
            missing_severity,
            "latent compactness was not checked because no fit-start latent vector was supplied",
        ));
        return;
    };
    let Some(bounds) = compact_box else {
        issues.push(issue(
            missing_severity,
            "latent compactness was not checked because no compact box was supplied",
        ));
        return;
    };
    if bounds.lower.len() != latent.len() || bounds.upper.len() != latent.len() {
        issues.push(issue(
            ArrowConvergenceIssueSeverity::Failure,
            "latent compact box dimension does not match latent vector length",
        ));
        return;
    }
    for i in 0..latent.len() {
        let value = latent[i];
        if !(value.is_finite() && bounds.lower[i].is_finite() && bounds.upper[i].is_finite()) {
            issues.push(issue(
                ArrowConvergenceIssueSeverity::Failure,
                "latent compact box check found a non-finite value or bound",
            ));
            return;
        }
        if bounds.lower[i] > bounds.upper[i] {
            issues.push(issue(
                ArrowConvergenceIssueSeverity::Failure,
                "latent compact box has lower bound greater than upper bound",
            ));
            return;
        }
        if value < bounds.lower[i] || value > bounds.upper[i] {
            issues.push(issue(
                ArrowConvergenceIssueSeverity::Failure,
                "fit-start latent value lies outside the supplied compact box",
            ));
            return;
        }
    }
}

fn min_shifted_row_eigenvalue(
    sys: &ArrowSchurSystem,
    ridge: f64,
    issues: &mut Vec<ArrowConvergenceIssue>,
) -> f64 {
    let mut min_eval = f64::INFINITY;
    for (row_idx, row) in sys.rows.iter().enumerate() {
        if row.htt.dim() != (sys.d, sys.d)
            || row.htbeta.dim() != (sys.d, sys.k)
            || row.gt.len() != sys.d
        {
            issues.push(issue(
                ArrowConvergenceIssueSeverity::Failure,
                "arrow-Schur row dimensions do not match system dimensions",
            ));
            return f64::NAN;
        }
        if row.htt.iter().any(|v| !v.is_finite())
            || row.htbeta.iter().any(|v| !v.is_finite())
            || row.gt.iter().any(|v| !v.is_finite())
        {
            issues.push(issue(
                ArrowConvergenceIssueSeverity::Failure,
                "arrow-Schur row contains non-finite Hessian or gradient entries",
            ));
            return f64::NAN;
        }
        let mut shifted = row.htt.clone();
        for j in 0..sys.d {
            shifted[[j, j]] += ridge;
        }
        let (lo, _) = symmetric_eigenvalue_bounds(&shifted);
        min_eval = min_eval.min(lo);
        if !(lo.is_finite() && lo > 0.0) {
            issues.push(issue(
                ArrowConvergenceIssueSeverity::Failure,
                &format!(
                    "proximal row block {row_idx} is not positive definite at ridge {ridge}; min eigenvalue {lo}"
                ),
            ));
        }
    }
    min_eval
}

fn schur_condition_number(
    sys: &ArrowSchurSystem,
    ridge: f64,
    options: &ArrowSchurConvergenceCheckOptions,
    issues: &mut Vec<ArrowConvergenceIssue>,
) -> f64 {
    if sys.k == 0 {
        return 1.0;
    }
    if sys.hbb.dim() != (sys.k, sys.k) {
        let severity = if options.fail_on_warnings {
            ArrowConvergenceIssueSeverity::Failure
        } else {
            ArrowConvergenceIssueSeverity::Warning
        };
        issues.push(issue(
            severity,
            "matrix-free beta block: fit-start checker cannot materialize the Schur condition number",
        ));
        return f64::INFINITY;
    }
    let Some(mut schur) = build_shifted_schur(sys, ridge) else {
        issues.push(issue(
            ArrowConvergenceIssueSeverity::Failure,
            "could not build shifted Schur complement because a row block was not positive definite",
        ));
        return f64::INFINITY;
    };
    for j in 0..sys.k {
        schur[[j, j]] += ridge;
    }
    let (lo, hi) = symmetric_eigenvalue_bounds(&schur);
    if !(lo.is_finite() && lo > 0.0 && hi.is_finite()) {
        issues.push(issue(
            ArrowConvergenceIssueSeverity::Failure,
            &format!(
                "shifted Schur complement is not positive definite; eigen bounds [{lo}, {hi}]"
            ),
        ));
        return f64::INFINITY;
    }
    let condition = hi / lo;
    if condition > options.max_condition_number {
        issues.push(issue(
            ArrowConvergenceIssueSeverity::Failure,
            &format!(
                "shifted Schur condition number {condition:.3e} exceeds limit {:.3e}",
                options.max_condition_number
            ),
        ));
    }
    condition
}

fn build_shifted_schur(sys: &ArrowSchurSystem, ridge: f64) -> Option<Array2<f64>> {
    let mut schur = sys.hbb.clone();
    for row in sys.rows.iter() {
        let mut htt = row.htt.clone();
        for j in 0..sys.d {
            htt[[j, j]] += ridge;
        }
        let factor = cholesky_lower_local(&htt)?;
        let solved = chol_solve_matrix_local(&factor, &row.htbeta);
        for c in 0..sys.d {
            for a in 0..sys.k {
                let left = row.htbeta[[c, a]];
                if left == 0.0 {
                    continue;
                }
                for b in 0..sys.k {
                    schur[[a, b]] -= left * solved[[c, b]];
                }
            }
        }
    }
    symmetrize(&mut schur);
    Some(schur)
}

fn cholesky_lower_local(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n || a.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn chol_solve_matrix_local(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for col in 0..m {
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = b[[i, col]];
            for k in 0..i {
                sum -= l[[i, k]] * y[k];
            }
            y[i] = sum / l[[i, i]];
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * out[[k, col]];
            }
            out[[i, col]] = sum / l[[i, i]];
        }
    }
    out
}

fn symmetric_eigenvalue_bounds(a: &Array2<f64>) -> (f64, f64) {
    let n = a.nrows();
    if n == 0 {
        return (1.0, 1.0);
    }
    if a.ncols() != n || a.iter().any(|v| !v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let mut m = a.clone();
    for _ in 0..(80 * n.max(1)) {
        let mut p = 0;
        let mut q = 0;
        let mut max_off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = m[[i, j]].abs();
                if off > max_off {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off <= 1e-12 {
            break;
        }
        let app = m[[p, p]];
        let aqq = m[[q, q]];
        let apq = m[[p, q]];
        let tau = (aqq - app) / (2.0 * apq);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for r in 0..n {
            if r != p && r != q {
                let mrp = m[[r, p]];
                let mrq = m[[r, q]];
                m[[r, p]] = c * mrp - s * mrq;
                m[[p, r]] = m[[r, p]];
                m[[r, q]] = s * mrp + c * mrq;
                m[[q, r]] = m[[r, q]];
            }
        }
        m[[p, p]] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        m[[q, q]] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        m[[p, q]] = 0.0;
        m[[q, p]] = 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        lo = lo.min(m[[i, i]]);
        hi = hi.max(m[[i, i]]);
    }
    (lo, hi)
}

#[inline]
fn symmetrize(a: &mut Array2<f64>) {
    // Callers in this module always pass square matrices (Schur complements);
    // delegate to the canonical helper in `linalg::utils`.
    crate::linalg::utils::enforce_symmetry(a)
}
