use super::*;
use statrs::function::gamma::{digamma, ln_gamma};

/// Ordered independent Beta--Bernoulli active-set prior over assignment logits.
///
/// Columns have independent `pi_k ~ Beta(a_k, 1)` rates whose means follow the
/// ordered schedule
///
/// `mu_k = (alpha / (alpha + 1))^(k + 1),  a_k = mu_k / (1 - mu_k)`.
///
/// The forward assignment remains the deterministic posterior-mean relaxation
/// `z_ik = sigmoid(ell_ik / tau)`.  The nuisance rate `pi_k` is integrated out
/// exactly in the penalty.  With weighted active mass `M_k = sum_i w_i z_ik`
/// and effective row count `N = sum_i w_i`, the per-column scalar is
///
/// ```text
/// L_k = -log(a_k) - log Gamma(M_k + a_k)
///       -log Gamma(N - M_k + 1) + log Gamma(N + a_k + 1).
/// ```
///
/// Consequently the logit gradient, Hessian, alpha update, and evidence
/// channels below are all derivatives of this one integrated scalar.  Ordered
/// shrinkage is scored here exactly once and is never multiplied into the
/// reconstructed function as a second prior factor.
#[derive(Debug, Clone)]
pub struct OrderedBetaBernoulliPenalty {
    pub k_max: usize,
    pub alpha: f64,
    pub tau: f64,
    pub temperature_schedule: Option<GumbelTemperatureSchedule>,
    pub learnable_alpha: bool,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
    /// Fixed/ungated columns are outside this prior and contribute no value or
    /// derivative channels.
    pub fixed_columns: Option<Vec<bool>>,
    /// Optional design weights.  They define both `M_k = sum_i w_i z_ik` and
    /// `N_eff = sum_i w_i`, so value and every derivative remain one operator.
    pub row_weights: Option<std::sync::Arc<[f64]>>,
}

#[derive(Debug, Clone, Copy)]
struct MarginalColumnDerivatives {
    mass: f64,
    a: f64,
    /// `dL/dM`.
    score: f64,
    /// `d²L/dM²`.
    score_derivative: f64,
}

impl OrderedBetaBernoulliPenalty {
    #[must_use]
    pub fn new(k_max: usize, alpha: f64, tau: f64, learnable_alpha: bool) -> Self {
        assert!(k_max > 0);
        assert!(alpha.is_finite() && alpha > 0.0);
        assert!(tau.is_finite() && tau > 0.0);
        Self {
            k_max,
            alpha,
            tau,
            temperature_schedule: None,
            learnable_alpha,
            weight: 1.0,
            weight_schedule: None,
            fixed_columns: None,
            row_weights: None,
        }
    }

    #[must_use]
    pub fn with_row_weights(mut self, weights: Option<&[f64]>) -> Self {
        if let Some(weights) = weights {
            assert!(
                weights.iter().all(|w| w.is_finite() && *w >= 0.0),
                "ordered Beta--Bernoulli row weights must be finite and nonnegative"
            );
            assert!(
                weights.iter().any(|w| *w > 0.0),
                "ordered Beta--Bernoulli row weights must contain positive mass"
            );
        }
        self.row_weights = weights.map(|w| std::sync::Arc::from(w.to_vec()));
        self
    }

    #[inline]
    fn row_weight(&self, row: usize) -> f64 {
        self.row_weights.as_ref().map_or(1.0, |w| w[row])
    }

    fn weighted_active_mass(&self, z: ArrayView1<'_, f64>) -> (Array1<f64>, f64) {
        assert_eq!(
            z.len() % self.k_max,
            0,
            "ordered Beta--Bernoulli target length must be divisible by k_max"
        );
        let n = z.len() / self.k_max;
        if let Some(weights) = self.row_weights.as_ref() {
            assert_eq!(
                weights.len(),
                n,
                "ordered Beta--Bernoulli row-weight length must equal the row count"
            );
        }
        let mut mass = Array1::<f64>::zeros(self.k_max);
        let mut n_eff = 0.0;
        for row in 0..n {
            let w = self.row_weight(row);
            n_eff += w;
            let start = row * self.k_max;
            for k in 0..self.k_max {
                mass[k] += w * z[start + k];
            }
        }
        (mass, n_eff)
    }

    #[inline]
    fn column_is_fixed(&self, k: usize) -> bool {
        self.fixed_columns
            .as_ref()
            .and_then(|m| m.get(k).copied())
            .unwrap_or(false)
    }

    /// Shapes of the independent `Beta(a_k, 1)` columns. The stable identity
    /// `a_k = 1 / expm1(-log(mu_k))` avoids subtracting an ordered mean rounded
    /// to one at large concentration.
    fn column_beta_shapes(&self, alpha: f64) -> Array1<f64> {
        let log_ratio = -(1.0 / alpha).ln_1p();
        let mut a_col = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let log_mu = ((k + 1) as f64) * log_ratio;
            a_col[k] = (1.0 / (-log_mu).exp_m1().max(f64::MIN_POSITIVE)).max(f64::MIN_POSITIVE);
        }
        a_col
    }

    /// `da_k/d rho` for `rho = log(alpha / alpha_base)`.
    fn column_beta_shape_rho_deriv(&self, alpha: f64, a_col: ArrayView1<'_, f64>) -> Array1<f64> {
        Array1::from_shape_fn(self.k_max, |k| {
            let a = a_col[k];
            ((k + 1) as f64) * (a / (alpha + 1.0)) * (a + 1.0)
        })
    }

    #[must_use]
    pub fn with_temperature_schedule(mut self, schedule: GumbelTemperatureSchedule) -> Self {
        self.tau = schedule.current_tau(schedule.iter_count);
        self.temperature_schedule = Some(schedule);
        self
    }

    impl_with_weight_schedule!(weight);

    fn resolved_alpha(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_alpha {
            resolve_learnable_weight(self.alpha, rho[0])
        } else {
            self.alpha
        }
    }

    fn concrete_logits(&self, target: ArrayView1<'_, f64>) -> Array1<f64> {
        let tau = self.tau;
        Array1::from_shape_fn(target.len(), |i| {
            let x = target[i] / tau;
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            }
        })
    }

    fn marginal_columns(
        &self,
        z: ArrayView1<'_, f64>,
        a_col: ArrayView1<'_, f64>,
    ) -> (Vec<MarginalColumnDerivatives>, f64) {
        let (active_mass, n_eff) = self.weighted_active_mass(z);
        let columns = (0..self.k_max)
            .map(|k| {
                let mass = active_mass[k].clamp(0.0, n_eff);
                let a = a_col[k];
                let active_arg = mass + a;
                let inactive_arg = n_eff - mass + 1.0;
                MarginalColumnDerivatives {
                    mass,
                    a,
                    score: -digamma(active_arg) + digamma(inactive_arg),
                    score_derivative: -trigamma(active_arg) - trigamma(inactive_arg),
                }
            })
            .collect();
        (columns, n_eff)
    }

    /// Total `rho = log alpha` derivatives of `dL/dM` and `d²L/dM²`.
    fn learnable_alpha_score_rho_derivs(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut d_score = Array1::<f64>::zeros(self.k_max);
        let mut d_score_derivative = Array1::<f64>::zeros(self.k_max);
        if !self.learnable_alpha {
            return (d_score, d_score_derivative);
        }
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let da_col = self.column_beta_shape_rho_deriv(alpha, a_col.view());
        let z = self.concrete_logits(target);
        let (columns, _) = self.marginal_columns(z.view(), a_col.view());
        for (k, column) in columns.iter().enumerate() {
            if self.column_is_fixed(k) {
                continue;
            }
            d_score[k] = -trigamma(column.mass + column.a) * da_col[k];
            d_score_derivative[k] = -tetragamma(column.mass + column.a) * da_col[k];
        }
        (d_score, d_score_derivative)
    }

    /// Exact derivatives of the PSD Loewner majorizer used by the Laplace
    /// curvature path.
    ///
    /// The integrated marginal has mass-Hessian coefficient
    /// `s'=-ψ₁(M+a)-ψ₁(N-M+1)<0`, so its cross-row rank-one Hessian
    /// block is negative semidefinite and contributes zero to the PSD
    /// majorizer. The only retained curvature is the positive part of the
    /// row-local term `s·d²z/dell²`. These channels differentiate that
    /// declared majorizer exactly.
    #[must_use]
    pub fn psd_majorizer_logit_third_channels(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> OrderedBetaBernoulliHessianDiagThirdChannels {
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, _) = self.marginal_columns(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / self.tau;
        let inv_tau2 = inv_tau * inv_tau;

        let mut z_jac = Array1::<f64>::zeros(target.len());
        let mut local_logit_third = Array1::<f64>::zeros(target.len());
        let mut m_channel = Array1::<f64>::zeros(target.len());
        let mut diagonal_term = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let column = columns[k];
                let zk = z[start + k];
                let jac = zk * (1.0 - zk) * inv_tau;
                let u = w_i * jac;
                let curvature = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                let dz_curvature = (1.0 - 6.0 * zk + 6.0 * zk * zk) * inv_tau2;
                let raw_diagonal_term = self.weight * column.score * w_i * curvature;
                let diagonal_gate = f64::from(raw_diagonal_term > 0.0);

                z_jac[start + k] = u;
                diagonal_term[start + k] = raw_diagonal_term;
                local_logit_third[start + k] =
                    self.weight * diagonal_gate * column.score * u * dz_curvature;
                m_channel[start + k] = self.weight
                    * diagonal_gate
                    * column.score_derivative
                    * w_i
                    * curvature;
            }
        }

        let mut mass_hessian_coefficient = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            if self.column_is_fixed(k) {
                continue;
            }
            mass_hessian_coefficient[k] = self.weight * columns[k].score_derivative;
        }

        let mut mass_hessian_log_alpha_derivative = Array1::<f64>::zeros(self.k_max);
        if self.learnable_alpha {
            let (_, d_score_derivative) = self.learnable_alpha_score_rho_derivs(target, rho);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                mass_hessian_log_alpha_derivative[k] = self.weight * d_score_derivative[k];
            }
        }

        OrderedBetaBernoulliHessianDiagThirdChannels {
            k_max: self.k_max,
            z_jac,
            local_logit_third,
            m_channel,
            mass_hessian_coefficient,
            mass_hessian_log_alpha_derivative,
            diagonal_term,
        }
    }

    /// `d²L / (d rho d ell_ik)` for the learnable concentration.
    #[must_use]
    pub fn log_alpha_target_mixed_derivative(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        if !self.learnable_alpha {
            return out;
        }
        let z = self.concrete_logits(target);
        let n = z.len() / self.k_max;
        let (d_score, _) = self.learnable_alpha_score_rho_derivs(target, rho);
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                out[start + k] = self.weight * d_score[k] * w_i * zk * (1.0 - zk) / self.tau;
            }
        }
        out
    }

    /// `d hessian_diag / d rho` for the learnable concentration.
    #[must_use]
    pub fn hessian_diag_log_alpha_derivative(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        if !self.learnable_alpha {
            return out;
        }
        let z = self.concrete_logits(target);
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / self.tau;
        let inv_tau2 = inv_tau * inv_tau;
        let (d_score, d_score_derivative) = self.learnable_alpha_score_rho_derivs(target, rho);
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let jac = zk * (1.0 - zk) * inv_tau;
                let u = w_i * jac;
                let curvature = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] =
                    self.weight * (d_score_derivative[k] * u * u + d_score[k] * w_i * curvature);
            }
        }
        out
    }
}

/// Third-derivative channels for the row-major `(N, K)` assignment-logit block.
#[derive(Debug, Clone)]
pub struct OrderedBetaBernoulliHessianDiagThirdChannels {
    pub k_max: usize,
    /// `u_ik = w_i dz_ik/dell_ik`, used both as the active-mass derivative and
    /// as the per-column rank-one carrier.
    pub z_jac: Array1<f64>,
    /// Row-local third derivative of the diagonal Hessian entry.
    pub local_logit_third: Array1<f64>,
    /// Active-mass derivative of each diagonal Hessian entry.
    pub m_channel: Array1<f64>,
    /// Raw per-column coefficient `weight·d²L/dM²` of the exact
    /// mass-coupled rank-one Hessian. It is strictly negative and is retained
    /// only to separate that term from a raw Hessian diagonal.
    pub mass_hessian_coefficient: Array1<f64>,
    /// Log-concentration derivative of [`Self::mass_hessian_coefficient`].
    pub mass_hessian_log_alpha_derivative: Array1<f64>,
    /// Raw row-local Hessian term
    /// `weight·(dL/dM)·w_i·d²z_i/dell_i²`. Its positive part is the
    /// ordered-prior PSD majorizer.
    pub diagonal_term: Array1<f64>,
}

impl AnalyticPenalty for OrderedBetaBernoulliPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, n_eff) = self.marginal_columns(z.view(), a_col.view());
        let mut value = 0.0;
        for (k, column) in columns.iter().enumerate() {
            if self.column_is_fixed(k) {
                continue;
            }
            value += -column.a.ln()
                - ln_gamma(column.mass + column.a)
                - ln_gamma(n_eff - column.mass + 1.0)
                + ln_gamma(n_eff + column.a + 1.0);
        }
        self.weight * value
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, _) = self.marginal_columns(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                out[start + k] = self.weight * columns[k].score * w_i * zk * (1.0 - zk) / self.tau;
            }
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, _) = self.marginal_columns(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / self.tau;
        let inv_tau2 = inv_tau * inv_tau;
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let jac = zk * (1.0 - zk) * inv_tau;
                let u = w_i * jac;
                let curvature = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] = self.weight
                    * (columns[k].score_derivative * u * u + columns[k].score * w_i * curvature);
            }
        }
        Some(out)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(
            v.len(),
            target.len(),
            "OrderedBetaBernoulliPenalty::hvp dimension mismatch"
        );
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, _) = self.marginal_columns(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / self.tau;
        let inv_tau2 = inv_tau * inv_tau;
        let mut contraction = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                contraction[k] += w_i * zk * (1.0 - zk) * inv_tau * v[start + k];
            }
        }
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let u = w_i * zk * (1.0 - zk) * inv_tau;
                let curvature = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] = self.weight
                    * (columns[k].score_derivative * u * contraction[k]
                        + columns[k].score * w_i * curvature * v[start + k]);
            }
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_alpha {
            return Array1::zeros(0);
        }
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let da_col = self.column_beta_shape_rho_deriv(alpha, a_col.view());
        let z = self.concrete_logits(target);
        let (columns, n_eff) = self.marginal_columns(z.view(), a_col.view());
        let mut gradient = 0.0;
        for (k, column) in columns.iter().enumerate() {
            if self.column_is_fixed(k) {
                continue;
            }
            let d_l_da =
                -1.0 / column.a - digamma(column.mass + column.a) + digamma(n_eff + column.a + 1.0);
            gradient += d_l_da * da_col[k];
        }
        Array1::from_vec(vec![self.weight * gradient])
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_alpha)
    }

    fn name(&self) -> &str {
        "ordered_beta_bernoulli"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.temperature_schedule.as_mut() {
            self.tau = schedule.current_tau(iter);
            schedule.iter_count = iter + 1;
        }
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}

/// Trigamma `psi_1(x)` for positive `x`.
fn trigamma(mut x: f64) -> f64 {
    assert!(x > 0.0);
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
        - inv2.powi(4) * inv / 30.0
        + 5.0 * inv2.powi(5) * inv / 66.0
}

/// Tetragamma `psi_2(x)`, the derivative of [`trigamma`], for positive `x`.
fn tetragamma(mut x: f64) -> f64 {
    assert!(x > 0.0);
    let mut acc = 0.0;
    while x < 8.0 {
        acc -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc - inv2 - inv2 * inv - 0.5 * inv2 * inv2 + inv2.powi(3) / 6.0 - inv2.powi(4) / 6.0
        + 3.0 * inv2.powi(5) / 10.0
        - 5.0 * inv2.powi(6) / 6.0
}
