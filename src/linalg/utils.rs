use crate::construction::calculate_condition_number;
use crate::faer_ndarray::{
    FaerArrayView, FaerLblt, FaerLdlt, FaerLlt, FaerSolve, array2_to_mat_mut,
};
use faer::Side;
use ndarray::Array2;

const HESSIAN_CONDITION_TARGET: f64 = 1e10;
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;

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

pub(crate) fn matrix_inverse_with_regularization(
    matrix: &Array2<f64>,
    label: &str,
) -> Option<Array2<f64>> {
    let p = matrix.nrows();
    if p == 0 || matrix.ncols() != p {
        return None;
    }

    enum Fact {
        Llt(FaerLlt<f64>),
        Ldlt(FaerLdlt<f64>),
        Lblt(FaerLblt<f64>),
    }
    impl Fact {
        fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
            match self {
                Fact::Llt(f) => f.solve_in_place(rhs),
                Fact::Ldlt(f) => f.solve_in_place(rhs),
                Fact::Lblt(f) => f.solve_in_place(rhs),
            }
        }
    }

    let mut planner = RidgePlanner::new(matrix);
    let factor = loop {
        let ridge = planner.ridge();
        let h_eff = if ridge > 0.0 {
            add_ridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        let h_view = FaerArrayView::new(&h_eff);
        if let Ok(chol) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
            break Fact::Llt(chol);
        }
        if let Ok(ldlt) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
            break Fact::Ldlt(ldlt);
        }
        if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
            log::warn!(
                "Falling back to LDLT pivoted inverse for {} after ridge {:.3e}",
                label,
                ridge
            );
            if let Ok(h_lblt) = std::panic::catch_unwind({
                let h_view = FaerArrayView::new(&h_eff);
                move || FaerLblt::new(h_view.as_ref(), Side::Lower)
            }) {
                break Fact::Lblt(h_lblt);
            }
            log::warn!("Failed to factorize {} for covariance", label);
            return None;
        }
        planner.bump_with_matrix(matrix);
    };

    let mut inv = Array2::<f64>::eye(p);
    let mut inv_view = array2_to_mat_mut(&mut inv);
    factor.solve_in_place(inv_view.as_mut());

    // Numerical solves can leave tiny asymmetry; enforce symmetry explicitly.
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (inv[[i, j]] + inv[[j, i]]);
            inv[[i, j]] = avg;
            inv[[j, i]] = avg;
        }
    }
    Some(inv)
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

pub(crate) fn add_ridge(matrix: &Array2<f64>, ridge: f64) -> Array2<f64> {
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
        let cond_estimate = calculate_condition_number(matrix)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0);
        let mut ridge = min_step;
        if let Some(cond) = cond_estimate {
            if !cond.is_finite() {
                ridge = scale * 1e-8;
            } else if cond > HESSIAN_CONDITION_TARGET {
                // If initial condition estimate is already above target, seed ridge
                // proportional to the excess so the first retry is meaningful.
                ridge = min_step * (cond / HESSIAN_CONDITION_TARGET);
            }
        } else {
            ridge = scale * 1e-8;
        }
        ridge = ridge.max(min_step);
        Self {
            cond_estimate,
            ridge,
            attempts: 0,
            scale,
        }
    }

    pub(crate) fn ridge(&self) -> f64 {
        self.ridge
    }

    pub(crate) fn cond_estimate(&self) -> Option<f64> {
        self.cond_estimate
    }

    #[inline]
    fn estimate_condition_with_ridge(&self, matrix: &Array2<f64>, ridge: f64) -> Option<f64> {
        let regularized = if ridge > 0.0 {
            add_ridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        calculate_condition_number(&regularized)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0)
    }

    pub(crate) fn bump_with_matrix(&mut self, matrix: &Array2<f64>) {
        self.attempts += 1;
        let min_step = self.scale * 1e-10;
        let base = self.ridge.max(min_step);

        // Estimate conditioning at the current ridge level.
        let cond_now = self.estimate_condition_with_ridge(matrix, base);
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
            if let Some(cond_next) = self.estimate_condition_with_ridge(matrix, proposal)
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
