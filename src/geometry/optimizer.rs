use ndarray::{Array1, ArrayView1};

use crate::geometry::manifold::{GeometryResult, RiemannianManifold, check_len, dot, norm};

pub trait RiemannianObjective {
    fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiemannianTrustRegion {
    pub radius: f64,
    pub max_iter: usize,
    pub grad_tol: f64,
}

impl Default for RiemannianTrustRegion {
    fn default() -> Self {
        Self {
            radius: 1.0,
            max_iter: 64,
            grad_tol: 1.0e-8,
        }
    }
}

impl RiemannianTrustRegion {
    pub fn minimize(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let mut x = initial.to_owned();
        let d = manifold.ambient_dim();
        check_len("trust-region initial point", x.len(), d)?;
        for _ in 0..self.max_iter {
            let (_, grad_e) = objective.value_gradient(x.view())?;
            let grad = manifold.project_tangent(x.view(), grad_e.view())?;
            let grad_norm = norm(grad.view());
            if grad_norm <= self.grad_tol {
                break;
            }
            let scale = self.radius.min(grad_norm) / grad_norm;
            let step = -grad * scale;
            x = manifold.retract(x.view(), step.view())?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiemannianLBFGS {
    pub history: usize,
    pub step_size: f64,
    pub max_iter: usize,
    pub grad_tol: f64,
}

impl Default for RiemannianLBFGS {
    fn default() -> Self {
        Self {
            history: 10,
            step_size: 1.0,
            max_iter: 100,
            grad_tol: 1.0e-8,
        }
    }
}

impl RiemannianLBFGS {
    /// Riemannian L-BFGS with a backtracking-and-expansion Armijo line search.
    ///
    /// The search starts at the user-supplied `step_size` (a hint, not a hard
    /// cap) and first *expands* by doubling while the Armijo sufficient-
    /// decrease condition continues to hold and the objective is still
    /// strictly improving. Once expansion stalls, it accepts the best step
    /// it has seen so far; if even the initial trial violates Armijo, it
    /// *contracts* by halving until Armijo holds or a safeguard floor is
    /// reached. This makes the optimizer robust to mis-scaled `step_size`
    /// inputs (including the Newton-natural α=1 that BFGS expects on
    /// well-conditioned quadratics) without forcing the caller to retune
    /// it, and preserves the secant pair (s, y) curvature condition so the
    /// L-BFGS inverse-Hessian approximation stays SPD.
    pub fn minimize(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let mut x = initial.to_owned();
        let mut s_hist: Vec<Array1<f64>> = Vec::new();
        let mut y_hist: Vec<Array1<f64>> = Vec::new();
        let (mut f_curr, grad_e0) = objective.value_gradient(x.view())?;
        let mut grad = manifold.project_tangent(x.view(), grad_e0.view())?;
        let armijo_c: f64 = 1.0e-4;
        let alpha_min: f64 = 1.0e-16;
        let alpha_max: f64 = 1.0e16;
        let initial_step = if self.step_size.is_finite() && self.step_size > 0.0 {
            self.step_size
        } else {
            1.0
        };
        for _ in 0..self.max_iter {
            if norm(grad.view()) <= self.grad_tol {
                break;
            }
            let direction = -two_loop(grad.view(), &s_hist, &y_hist);
            let slope = dot(grad.view(), direction.view());
            // Guard against ascent directions caused by stale curvature; if
            // the BFGS direction is not a descent direction, fall back to
            // the projected steepest-descent direction so progress is
            // guaranteed.
            let (direction, slope) = if slope < 0.0 {
                (direction, slope)
            } else {
                let sd = -grad.clone();
                let s_sd = dot(grad.view(), sd.view());
                (sd, s_sd)
            };
            let old_x = x.clone();
            let old_grad = grad.clone();
            // --- Armijo line search with bidirectional adaptation. ---
            let mut alpha = initial_step;
            let mut best_alpha = 0.0;
            let mut best_f = f_curr;
            let mut best_x = x.clone();
            let mut best_grad = grad.clone();
            // First try to expand: while Armijo holds and the objective keeps
            // improving, double the step.
            loop {
                let step = &direction * alpha;
                let trial_x = manifold.retract(x.view(), step.view())?;
                let (f_trial, g_trial_e) = objective.value_gradient(trial_x.view())?;
                let armijo_rhs = f_curr + armijo_c * alpha * slope;
                let accepted = f_trial.is_finite() && f_trial <= armijo_rhs;
                if accepted && f_trial < best_f {
                    best_alpha = alpha;
                    best_f = f_trial;
                    best_x = trial_x;
                    best_grad = manifold.project_tangent(best_x.view(), g_trial_e.view())?;
                    if alpha >= alpha_max {
                        break;
                    }
                    alpha *= 2.0;
                } else {
                    break;
                }
            }
            // If no expansion succeeded, contract from the initial trial.
            if best_alpha == 0.0 {
                alpha = initial_step;
                while alpha > alpha_min {
                    let step = &direction * alpha;
                    let trial_x = manifold.retract(x.view(), step.view())?;
                    let (f_trial, g_trial_e) = objective.value_gradient(trial_x.view())?;
                    let armijo_rhs = f_curr + armijo_c * alpha * slope;
                    if f_trial.is_finite() && f_trial <= armijo_rhs {
                        best_alpha = alpha;
                        best_f = f_trial;
                        best_x = trial_x;
                        best_grad = manifold.project_tangent(best_x.view(), g_trial_e.view())?;
                        break;
                    }
                    alpha *= 0.5;
                }
            }
            if best_alpha == 0.0 {
                // No admissible step found — terminate at the current point.
                break;
            }
            x = best_x;
            f_curr = best_f;
            grad = best_grad;
            let s = manifold.log_map(old_x.view(), x.view())?;
            let mut path = ndarray::Array2::<f64>::zeros((2, manifold.ambient_dim()));
            path.row_mut(0).assign(&old_x);
            path.row_mut(1).assign(&x);
            let transported_old_grad = manifold.parallel_transport(path.view(), old_grad.view())?;
            let y = &grad - &transported_old_grad;
            // Only commit the (s, y) pair when the curvature condition sᵀy > 0
            // holds (strict positivity, not just non-zero). This is required
            // for the implicit BFGS inverse-Hessian update to remain SPD.
            if dot(s.view(), y.view()) > 1.0e-14 {
                s_hist.push(s);
                y_hist.push(y);
                if s_hist.len() > self.history {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
            }
        }
        Ok(x)
    }
}

fn two_loop(
    grad: ArrayView1<'_, f64>,
    s_hist: &[Array1<f64>],
    y_hist: &[Array1<f64>],
) -> Array1<f64> {
    let mut q = grad.to_owned();
    let mut alpha = vec![0.0; s_hist.len()];
    for i in (0..s_hist.len()).rev() {
        let rho = 1.0 / dot(s_hist[i].view(), y_hist[i].view());
        alpha[i] = rho * dot(s_hist[i].view(), q.view());
        q -= &(y_hist[i].clone() * alpha[i]);
    }
    let mut r = q;
    if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
        let yy = dot(y.view(), y.view());
        if yy > 1.0e-14 {
            r *= dot(s.view(), y.view()) / yy;
        }
    }
    for i in 0..s_hist.len() {
        let rho = 1.0 / dot(s_hist[i].view(), y_hist[i].view());
        let beta = rho * dot(y_hist[i].view(), r.view());
        r += &(s_hist[i].clone() * (alpha[i] - beta));
    }
    r
}
