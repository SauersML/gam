use super::*;

// ---------------------------------------------------------------------------
// Total variation penalty
// ---------------------------------------------------------------------------

/// Shape of the first-difference operator used by [`TotalVariationPenalty`].
#[derive(Debug, Clone)]
pub enum DifferenceOpKind {
    /// Path graph with rows connected as `(0, 1), (1, 2), ...`.
    ForwardDiff1D,
    /// Explicit adjacency list; each edge row has `-1` at `from`, `+1` at `to`.
    GraphEdges(Vec<(usize, usize)>),
}


/// Coordinatewise/anisotropic smoothed-L¹ total variation on a row-major
/// `(n_eff, d)` latent block.
///
/// Uses the differentiable Huber-style kernel `φ(x)=sqrt(x²+ε²)-ε` separately
/// for each edge and latent axis. This is not vector-norm/isotropic edge TV:
/// the Hessian intentionally has no cross-axis terms. The difference operator
/// defines the prior shape: forward 1-D differences for ordered context
/// windows, or graph edges for adjacency-structured atoms. Pair TV with
/// Orthogonality when piecewise-constant atoms need a gauge-fixed basis.
#[derive(Debug, Clone)]
pub struct TotalVariationPenalty {
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major latent coefficient block.
    pub n_eff: usize,
    pub difference_op: DifferenceOpKind,
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl TotalVariationPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        weight: f64,
        n_eff: usize,
        difference_op: DifferenceOpKind,
        smoothing_eps: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "TotalVariationPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("TotalVariationPenalty::new requires n_eff > 0".to_string());
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "TotalVariationPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        if let DifferenceOpKind::GraphEdges(edges) = &difference_op {
            if edges.is_empty() {
                return Err(
                    "TotalVariationPenalty::new GraphEdges requires at least one edge".to_string(),
                );
            }
            for &(a, b) in edges {
                if a >= n_eff || b >= n_eff {
                    return Err(format!(
                        "TotalVariationPenalty::new graph edge ({a}, {b}) exceeds n_eff {n_eff}"
                    ));
                }
                if a == b {
                    return Err(format!(
                        "TotalVariationPenalty::new graph edge ({a}, {b}) is self-referential"
                    ));
                }
            }
        }
        Ok(Self {
            weight,
            n_eff,
            difference_op,
            smoothing_eps,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn edge_count(&self) -> usize {
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => self.n_eff.saturating_sub(1),
            DifferenceOpKind::GraphEdges(edges) => edges.len(),
        }
    }

    fn add_edge_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let r = (diff * diff + eps2).sqrt();
            let curvature = eps2 / (r * r * r);
            let dv = v[ib] - v[ia];
            let h = weight * curvature * dv;
            out[ia] -= h;
            out[ib] += h;
        }
    }

    fn add_edge_grad(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let smooth_sign = diff / (diff * diff + eps2).sqrt();
            let g = weight * smooth_sign;
            out[ia] -= g;
            out[ib] += g;
        }
    }

    fn add_edge_diag(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let r = (diff * diff + eps2).sqrt();
            let curvature = weight * eps2 / (r * r * r);
            out[ia] += curvature;
            out[ib] += curvature;
        }
    }

    fn add_edge_dense(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array2<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        for j in 0..d {
            let ia = a * d + j;
            let ib = b * d + j;
            let diff = target[ib] - target[ia];
            let r = (diff * diff + eps2).sqrt();
            let curvature = weight * eps2 / (r * r * r);
            out[[ia, ia]] += curvature;
            out[[ib, ib]] += curvature;
            out[[ia, ib]] -= curvature;
            out[[ib, ia]] -= curvature;
        }
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_diag(target, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_diag(target, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    /// Materialize `Dᵀ diag(φ''(D T)) D` for diagnostics and small graph cases.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(d) = self.latent_dim(n) else {
            return Array2::<f64>::zeros((n, n));
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array2::<f64>::zeros((n, n));
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_dense(target, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_dense(target, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    pub fn log_det_plus_lambda_i_forward_1d(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !matches!(&self.difference_op, DifferenceOpKind::ForwardDiff1D) {
            return Err(
                "TotalVariationPenalty::log_det_plus_lambda_i_forward_1d requires ForwardDiff1D"
                    .to_string(),
            );
        }
        let Some(d) = self.latent_dim(target.len()) else {
            return Err(format!(
                "TotalVariationPenalty target length {} is not divisible by n_eff {}",
                target.len(),
                self.n_eff
            ));
        };
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "TotalVariationPenalty::log_det_plus_lambda_i_forward_1d requires finite λ > 0; got {lambda}"
            ));
        }
        let n = self.n_eff;
        if n == 1 {
            return Ok((d as f64) * lambda.ln());
        }
        let weight = self.resolved_weight(rho);
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        let mut total = 0.0;
        for j in 0..d {
            let mut edge_w = vec![0.0; n - 1];
            for a in 0..n - 1 {
                let diff = target[(a + 1) * d + j] - target[a * d + j];
                let r = (diff * diff + eps2).sqrt();
                edge_w[a] = weight * eps2 / (r * r * r);
            }

            let mut prev_pivot = lambda + edge_w[0];
            if !prev_pivot.is_finite() || prev_pivot <= 0.0 {
                return Err(format!(
                    "TotalVariationPenalty log-det encountered non-positive pivot {prev_pivot:.3e}"
                ));
            }
            total += prev_pivot.ln();
            for row in 1..n {
                let left = edge_w[row - 1];
                let right = if row + 1 < n { edge_w[row] } else { 0.0 };
                let diag = lambda + left + right;
                let pivot = diag - left * left / prev_pivot;
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(format!(
                        "TotalVariationPenalty log-det encountered non-positive pivot {pivot:.3e}"
                    ));
                }
                total += pivot.ln();
                prev_pivot = pivot;
            }
        }
        Ok(total)
    }
}


impl AnalyticPenalty for TotalVariationPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(d) = self.latent_dim(target.len()) else {
            return 0.0;
        };
        if self.edge_count() == 0 {
            return 0.0;
        }
        let weight = self.resolved_weight(rho);
        let eps = self.smoothing_eps;
        let eps2 = eps * eps;
        let mut acc = 0.0;
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    let b = a + 1;
                    for j in 0..d {
                        let diff = target[b * d + j] - target[a * d + j];
                        acc += (diff * diff + eps2).sqrt() - eps;
                    }
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    for j in 0..d {
                        let diff = target[b * d + j] - target[a * d + j];
                        acc += (diff * diff + eps2).sqrt() - eps;
                    }
                }
            }
        }
        weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_grad(target, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_grad(target, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        match &self.difference_op {
            DifferenceOpKind::ForwardDiff1D => {
                for a in 0..self.n_eff.saturating_sub(1) {
                    self.add_edge_hvp(target, v, &mut out, d, a, a + 1, weight);
                }
            }
            DifferenceOpKind::GraphEdges(edges) => {
                for &(a, b) in edges {
                    self.add_edge_hvp(target, v, &mut out, d, a, b, weight);
                }
            }
        }
        out
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "total_variation"
    }

    impl_scalar_apply_schedule!(weight);
}


// ---------------------------------------------------------------------------
// Monotonicity penalty (1D shape constraint)
// ---------------------------------------------------------------------------

/// Soft monotonicity penalty over a row-major `(n_eff, d)` latent block.
///
/// For each adjacent pair `(a, a+1)` along the leading axis and each output
/// column `j`, the penalty contribution is
///
///     softplus(-direction * (target[a+1, j] - target[a, j]) / smoothing_eps)
///     * smoothing_eps
///
/// which is the smoothed hinge that hits zero when the slope agrees with
/// `direction` (+1 ⇒ non-decreasing, -1 ⇒ non-increasing) and grows
/// approximately linearly when it disagrees. The Hessian is positive
/// semidefinite (softplus is convex) so the penalty composes cleanly with
/// PIRLS/REML.
///
/// `n_eff` is the number of latent rows along the constrained axis; the
/// remaining `target.len() / n_eff` columns are penalized independently and
/// summed.
#[derive(Debug, Clone)]
pub struct MonotonicityPenalty {
    pub weight: f64,
    pub n_eff: usize,
    /// `+1.0` for non-decreasing, `-1.0` for non-increasing along the leading axis.
    pub direction: f64,
    pub smoothing_eps: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}


impl MonotonicityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        weight: f64,
        n_eff: usize,
        direction: f64,
        smoothing_eps: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "MonotonicityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("MonotonicityPenalty::new requires n_eff > 0".to_string());
        }
        if !(direction.is_finite() && direction.abs() > 0.0) {
            return Err(format!(
                "MonotonicityPenalty::new requires finite non-zero direction (+1 or -1), got {direction}"
            ));
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "MonotonicityPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        Ok(Self {
            weight,
            n_eff,
            direction: direction.signum(),
            smoothing_eps,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            return None;
        }
        Some(target_len / self.n_eff)
    }

    /// Smoothed-hinge contribution for a single edge `(a, b)` and column `j`.
    fn edge_value(&self, target: ArrayView1<'_, f64>, d: usize, a: usize, b: usize) -> f64 {
        let eps = self.smoothing_eps;
        let mut acc = 0.0;
        for j in 0..d {
            let slope = target[b * d + j] - target[a * d + j];
            let z = -self.direction * slope / eps;
            // softplus(z) * eps, computed in a numerically stable form.
            let sp = if z > 0.0 {
                z + (-z).exp().ln_1p()
            } else {
                z.exp().ln_1p()
            };
            acc += sp * eps;
        }
        acc
    }

    /// d softplus(-dir * slope / eps) * eps / d target = -dir * sigma(-dir*slope/eps).
    fn edge_grad(
        &self,
        target: ArrayView1<'_, f64>,
        out: &mut Array1<f64>,
        d: usize,
        a: usize,
        b: usize,
        weight: f64,
    ) {
        let eps = self.smoothing_eps;
        for j in 0..d {
            let slope = target[b * d + j] - target[a * d + j];
            let z = -self.direction * slope / eps;
            // Stable sigmoid(z).
            let sigma = if z > 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            };
            let g = weight * (-self.direction) * sigma;
            out[a * d + j] -= g;
            out[b * d + j] += g;
        }
    }
}


impl AnalyticPenalty for MonotonicityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(d) = self.latent_dim(target.len()) else {
            return 0.0;
        };
        if self.n_eff < 2 {
            return 0.0;
        }
        let weight = self.resolved_weight(rho);
        let mut acc = 0.0;
        for a in 0..self.n_eff.saturating_sub(1) {
            acc += self.edge_value(target, d, a, a + 1);
        }
        weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for a in 0..self.n_eff.saturating_sub(1) {
            self.edge_grad(target, &mut out, d, a, a + 1, weight);
        }
        out
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        let Some(d) = self.latent_dim(target.len()) else {
            return Array1::<f64>::zeros(target.len());
        };
        let weight = self.resolved_weight(rho);
        let eps = self.smoothing_eps;
        let mut out = Array1::<f64>::zeros(target.len());
        for a in 0..self.n_eff.saturating_sub(1) {
            let b = a + 1;
            for j in 0..d {
                let slope = target[b * d + j] - target[a * d + j];
                let z = -self.direction * slope / eps;
                let sigma = if z > 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                };
                // d²P/d(target_a)d(target_b) follows from the chain rule on
                // z = -dir * (target_b - target_a) / eps. The penalty value is
                // `softplus(z) * eps` (note the outer eps from `edge_value`).
                // softplus''(z) = sigma(z)(1 - sigma(z)) and the (dz/dtarget)²
                // factor is 1/eps², but the value's outer `* eps` cancels one of
                // those, leaving `sigma(1 - sigma) / eps` — exactly the eps power
                // that keeps `hvp` consistent with the finite difference of
                // `grad_target` (whose own eps already cancelled). Off-diagonal
                // entries carry an extra minus sign from the difference.
                let h = weight * sigma * (1.0 - sigma) / eps;
                let dv = v[b * d + j] - v[a * d + j];
                out[a * d + j] -= h * dv;
                out[b * d + j] += h * dv;
            }
        }
        out
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "monotonicity"
    }

    impl_scalar_apply_schedule!(weight);
}


