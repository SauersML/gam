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
    pub fn minimize(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let mut x = initial.to_owned();
        let mut s_hist: Vec<Array1<f64>> = Vec::new();
        let mut y_hist: Vec<Array1<f64>> = Vec::new();
        let (_, grad_e0) = objective.value_gradient(x.view())?;
        let mut grad = manifold.project_tangent(x.view(), grad_e0.view())?;
        for _ in 0..self.max_iter {
            if norm(grad.view()) <= self.grad_tol {
                break;
            }
            let direction = -two_loop(grad.view(), &s_hist, &y_hist);
            let old_x = x.clone();
            let old_grad = grad.clone();
            x = manifold.retract(x.view(), (direction * self.step_size).view())?;
            let (_, grad_e) = objective.value_gradient(x.view())?;
            grad = manifold.project_tangent(x.view(), grad_e.view())?;
            let s = manifold.log_map(old_x.view(), x.view())?;
            let mut path = ndarray::Array2::<f64>::zeros((2, manifold.ambient_dim()));
            path.row_mut(0).assign(&old_x);
            path.row_mut(1).assign(&x);
            let transported_old_grad = manifold.parallel_transport(path.view(), old_grad.view())?;
            let y = &grad - &transported_old_grad;
            if dot(s.view(), y.view()).abs() > 1.0e-14 {
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
