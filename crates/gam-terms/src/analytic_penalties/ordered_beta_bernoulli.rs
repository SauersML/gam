use super::*;
use statrs::function::gamma::{digamma, ln_gamma};

/// Ordered independent Beta--Bernoulli prior over relaxed assignment logits.
///
/// Columns have independent `pi_k ~ Beta(a_k, 1)` rates whose means follow the
/// ordered schedule
///
/// `mu_k = (alpha / (alpha + 1))^(k + 1),  a_k = mu_k / (1 - mu_k)`.
///
/// The forward assignment is the deterministic relaxation
/// `z_ik = sigmoid(ell_ik / tau)`.  The nuisance rate `pi_k` is integrated out
/// exactly in the penalty.  With weighted active mass `M_k = sum_i w_i z_ik`
/// and effective row count `N = sum_i w_i`, the per-column scalar is
///
/// ```text
/// L_k = -log(a_k) - log Gamma(M_k + a_k)
///       -log Gamma(N - M_k + 1) + log Gamma(N + a_k + 1).
/// ```
///
/// Consequently the logit gradient, Hessian, concentration update, and criterion
/// channels below are all derivatives of this one integrated scalar.  Ordered
/// shrinkage is scored here exactly once and is never multiplied into the
/// reconstructed function as a second prior factor.
///
/// `exp(−L_k)` is a probability at binary gates, but it is not a density over the
/// relaxed gates `z_ik ∈ (0, 1)`: its mass there is `C(a_k, N) < 1` and depends on
/// `a_k` (#2933 F45). [`AnalyticPenalty::value`] is therefore the unnormalized
/// energy, and [`Self::log_partition`] is its normalizer, so that
/// `value + log_partition` is the normalized negative log prior of the untempered
/// penalty. A tempered energy `weight·L` has a different normalizer that is not
/// computed, and is refused there.
#[derive(Debug, Clone)]
pub struct OrderedBetaBernoulliPenalty {
    pub k_max: usize,
    pub alpha: f64,
    pub tau: f64,
    pub temperature_schedule: Option<GumbelTemperatureSchedule>,
    pub learnable_alpha: bool,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
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

    /// Per-row summands `w_i z_ik` of the weighted active mass `M_k`, row-major
    /// `N·K` like the target.
    pub fn weighted_active_mass_rows(&self, target: ArrayView1<'_, f64>) -> Array1<f64> {
        let z = self.concrete_logits(target);
        Array1::from_shape_fn(z.len(), |idx| self.row_weight(idx / self.k_max) * z[idx])
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
            validated_learnable_weight(self.alpha, rho[0])
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
                // These are derivatives of the *integrated* Beta--Bernoulli
                // scalar, not of a plug-in energy evaluated at E[pi | z]:
                //
                //   L(M,a) = -log(a) - log Gamma(M+a)
                //            -log Gamma(N-M+1) + log Gamma(N+a+1),
                //   dL/dM   = -psi(M+a) + psi(N-M+1),
                //   d2L/dM2 = -psi1(M+a) - psi1(N-M+1).
                //
                // Keeping both channels here ensures the value, logit
                // gradient, alpha update, and curvature all come from this
                // one marginal objective.
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
                m_channel[start + k] =
                    self.weight * diagonal_gate * column.score_derivative * w_i * curvature;
            }
        }

        let mut mass_hessian_coefficient = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            mass_hessian_coefficient[k] = self.weight * columns[k].score_derivative;
        }

        let mut mass_hessian_log_alpha_derivative = Array1::<f64>::zeros(self.k_max);
        if self.learnable_alpha {
            let (_, d_score_derivative) = self.learnable_alpha_score_rho_derivs(target, rho);
            for k in 0..self.k_max {
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

    /// #2330 Patch D — the structural data the exact-A θ-adjoint needs to
    /// contract the ordered-Beta--Bernoulli prior curvature `∂ΔC_obb/∂ℓ_w`
    /// against a joint pseudo-inverse in the SAE cache layout. `ΔC_obb` (per
    /// column `k`) is `weight·S'_k·uuᵀ + diag(min(D_i, 0))` with
    /// `u_i = w_i·z_i(1−z_i)/τ`, `D_i = weight·S_k·w_i·curv_i`,
    /// `curv_i = z_i(1−z_i)(1−2z_i)/τ²`. Its logit derivative needs the column
    /// integrated-marginal score and its first TWO M-derivatives
    /// `S_k, S'_k, S''_k` (the third is `−ψ₂(M+a)+ψ₂(N−M+1)`), plus the concrete
    /// gates `z` and the row weights — everything except the cache-layout
    /// contraction, which the caller owns.
    #[must_use]
    pub fn logit_theta_adjoint_data(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> OrderedBetaBernoulliLogitAdjointData {
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, n_eff) = self.marginal_columns(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let mut score = vec![0.0_f64; self.k_max];
        let mut score_derivative = vec![0.0_f64; self.k_max];
        let mut score_second = vec![0.0_f64; self.k_max];
        for k in 0..self.k_max {
            let column = columns[k];
            let active_arg = column.mass + column.a;
            let inactive_arg = n_eff - column.mass + 1.0;
            score[k] = column.score;
            score_derivative[k] = column.score_derivative;
            // d³L/dM³ = −ψ₂(M+a) + ψ₂(N−M+1).
            score_second[k] = -tetragamma(active_arg) + tetragamma(inactive_arg);
        }
        let row_weight = (0..n).map(|row| self.row_weight(row)).collect();
        OrderedBetaBernoulliLogitAdjointData {
            k_max: self.k_max,
            n,
            weight: self.weight,
            tau: self.tau,
            z: z.to_vec(),
            row_weight,
            score,
            score_derivative,
            score_second,
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

    /// `Σ_k log C(a_k, N)` over the columns ([`ordered_beta_bernoulli_log_partition`]),
    /// and its derivative in the learnable log concentration (empty when α is fixed).
    ///
    /// `N = Σ_i w_i` is the effective row count the integrated scalar already scores its
    /// weighted active mass against. The partition depends on neither the logits nor the
    /// gates, so the logit gradient, Hessian and third channels are unchanged by it. Only
    /// the untempered energy has this normalizer, so a penalty whose `weight` is not one, or
    /// that carries a weight schedule, is refused.
    pub fn log_partition(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        if self.weight != 1.0 || self.weight_schedule.is_some() {
            return Err(format!(
                "ordered Beta--Bernoulli log partition: the tempered energy weight·L (weight {}, \
                 schedule {}) has no computed partition function; only the untempered prior is \
                 normalized",
                self.weight,
                self.weight_schedule.is_some()
            ));
        }
        assert_eq!(
            target.len() % self.k_max,
            0,
            "ordered Beta--Bernoulli target length must be divisible by k_max"
        );
        let n = target.len() / self.k_max;
        if let Some(weights) = self.row_weights.as_ref() {
            assert_eq!(
                weights.len(),
                n,
                "ordered Beta--Bernoulli row-weight length must equal the row count"
            );
        }
        let n_eff: f64 = (0..n).map(|row| self.row_weight(row)).sum();
        let alpha = self.resolved_alpha(rho);
        let columns = self.column_log_partitions(alpha, n_eff)?;
        let rho_derivative = if self.learnable_alpha {
            Array1::from_vec(vec![columns.log_alpha_derivative])
        } else {
            Array1::zeros(0)
        };
        Ok((columns.value, rho_derivative))
    }

    /// `Σ_k log C(a_k, N)` and its log-concentration derivative at concentration `α` over
    /// `N` effective rows.
    ///
    /// They are a pure function of `(k_max, α, N)`, and an inner solve holds `ρ`, so `α`,
    /// fixed while every objective trial scores the prior's value with its partition. Each
    /// column is an adaptive quadrature, and in the anneal demo's 45 inner-thread stack
    /// samples 10 sat in it, 9 of them under the gauge-orbit line search (#2627, run
    /// 1212363). The last result on this thread is kept against the exact bits of its
    /// inputs, so a repeat returns the words the quadrature would compute.
    fn column_log_partitions(&self, alpha: f64, n_eff: f64) -> Result<ColumnLogPartitions, String> {
        let key = (self.k_max, alpha.to_bits(), n_eff.to_bits());
        if let Some(last) = LAST_COLUMN_LOG_PARTITIONS.with(std::cell::Cell::get)
            && last.key == key
        {
            return Ok(last);
        }
        let a_col = self.column_beta_shapes(alpha);
        let mut value = 0.0;
        let mut log_alpha_derivative = 0.0;
        for k in 0..self.k_max {
            let partition = ordered_beta_bernoulli_log_partition(a_col[k], n_eff)?;
            value += partition.value;
            // `da_k/dρ = (k + 1)·a_k(a_k + 1)/(α + 1)` and the partition carries `a_k·∂_a log C`.
            log_alpha_derivative += partition.log_shape_derivative * ((k + 1) as f64) * (a_col[k] + 1.0)
                / (alpha + 1.0);
        }
        let columns = ColumnLogPartitions {
            key,
            value,
            log_alpha_derivative,
        };
        LAST_COLUMN_LOG_PARTITIONS.with(|last| last.set(Some(columns)));
        Ok(columns)
    }
}

/// [`OrderedBetaBernoulliPenalty::column_log_partitions`] at `key = (k_max, α bits, N bits)`.
#[derive(Clone, Copy)]
struct ColumnLogPartitions {
    key: (usize, u64, u64),
    value: f64,
    log_alpha_derivative: f64,
}

thread_local! {
    static LAST_COLUMN_LOG_PARTITIONS: std::cell::Cell<Option<ColumnLogPartitions>> =
        const { std::cell::Cell::new(None) };
}

/// #2330 Patch D — ordered-Beta--Bernoulli prior curvature `∂ΔC_obb/∂ℓ`
/// structural data for the exact-A θ-adjoint (see
/// [`OrderedBetaBernoulliPenalty::logit_theta_adjoint_data`]). Per-column
/// integrated-marginal score and its first two M-derivatives, the concrete
/// gates, and the row weights; the caller forms the cache-layout contraction.
#[derive(Debug, Clone)]
pub struct OrderedBetaBernoulliLogitAdjointData {
    pub k_max: usize,
    pub n: usize,
    pub weight: f64,
    pub tau: f64,
    /// Concrete gates `z_i = σ(ℓ_i/τ)`, flat `n·k_max` (row-major).
    pub z: Vec<f64>,
    /// Row design weights (length `n`).
    pub row_weight: Vec<f64>,
    /// Column integrated-marginal `dL/dM`, `d²L/dM²`, `d³L/dM³`.
    pub score: Vec<f64>,
    pub score_derivative: Vec<f64>,
    pub score_second: Vec<f64>,
}

/// Third-derivative channels for the row-major `(N, K)` assignment-logit block.
#[derive(Debug, Clone)]
pub struct OrderedBetaBernoulliHessianDiagThirdChannels {
    pub k_max: usize,
    /// `u_ik = w_i dz_ik/dell_ik`, used both as the active-mass derivative and
    /// in the exact per-column rank-one Hessian term.
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

    fn validate_rho(&self, rho: ArrayView1<'_, f64>) -> Result<(), String> {
        if rho.len() != self.rho_count() {
            return Err(format!(
                "ordered Beta--Bernoulli rho length {} != declared {}",
                rho.len(),
                self.rho_count()
            ));
        }
        if self.learnable_alpha {
            resolve_learnable_weight(self.alpha, rho[0])?;
        }
        Ok(())
    }

    fn rho_coordinate_domains(&self) -> Result<Vec<(f64, f64)>, String> {
        if !self.learnable_alpha {
            return Ok(Vec::new());
        }
        Ok(vec![
            learnable_weight_coordinate_domain(self.alpha)?
                .ok_or_else(|| "ordered Beta--Bernoulli alpha must be positive".to_string())?,
        ])
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = self.resolved_alpha(rho);
        let a_col = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let (columns, n_eff) = self.marginal_columns(z.view(), a_col.view());
        let mut value = 0.0;
        for column in &columns {
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
                let zk = z[start + k];
                contraction[k] += w_i * zk * (1.0 - zk) * inv_tau * v[start + k];
            }
        }
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
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

}

// `ψ₁` and `ψ₂` come from the workspace's single polygamma implementation. The
// local copies they replace recursed only to `x ≥ 8` and stopped at `B₁₀`,
// leaving 6.3e−11 / 3.9e−11 relative error, and they were a THIRD independent
// transcription of the same series — `gam-sae` and `gam-solve` each had their
// own, agreeing with this one only to ten digits.
//
// The local copies asserted `x > 0`; `gam_math` returns `NaN` off-domain
// instead, which is the same contract the `gam-solve` copy already used.
use gam_math::special::{gauss_legendre, log_exprel, tetragamma, trigamma};
use std::f64::consts::LN_2;

/// The log partition function of the relaxed ordered Beta--Bernoulli prior on one column; see
/// [`ordered_beta_bernoulli_log_partition`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedBetaBernoulliLogPartition {
    /// `log C(a, N)`.
    pub value: f64,
    /// `a·∂_a log C(a, N)`.
    pub log_shape_derivative: f64,
}

/// `ln ∫₀¹ π^z (1 − π)^{1−z} dz` at the rate `π = e^{−t}`, `t = e^τ`.
///
/// With `u = logit π = −ln expm1(t)` the mass is `(2π − 1)/u = tanh(u/2)/u`, at most `1/2`
/// (at `t = ln 2`). `ln(tanh(x)/x)` has an absolute error of a few `ε` for every `x = |u|/2 > 0`.
/// Once `t` overflows, `u = −t` to within `e^{−t}` and the logarithm is `−τ`.
fn log_relaxed_gate_mass(tau: f64) -> f64 {
    let t = tau.exp();
    if t.is_infinite() {
        return -tau;
    }
    let u = -(tau + log_exprel(t));
    let x = 0.5 * u.abs();
    if x == 0.0 {
        -LN_2
    } else {
        (x.tanh() / x).ln() - LN_2
    }
}

/// `d/dτ` of [`log_relaxed_gate_mass`]: `(1/sinh u − 1/u)·du/dτ`, with
/// `du/dτ = −t/(1 − e^{−t}) = −1/exprel(−t)`, which is `−1` once `t` overflows.
fn log_relaxed_gate_mass_slope(tau: f64) -> f64 {
    let t = tau.exp();
    if t.is_infinite() {
        return -1.0;
    }
    let u = -(tau + log_exprel(t));
    let log_mass_slope = if u == 0.0 {
        0.0
    } else {
        u.sinh().recip() - u.recip()
    };
    -log_mass_slope * (-log_exprel(-t)).exp()
}

/// Gauss--Legendre order of each panel rule in [`ordered_beta_bernoulli_log_partition`].
const LOG_PARTITION_PANEL_ORDER: usize = 16;

/// Panel count at which [`ordered_beta_bernoulli_log_partition`] refuses instead of
/// returning an unconverged value.
const LOG_PARTITION_MAX_PANELS: usize = 1 << 12;

/// One panel `[left, right]` of the partition integral, priced by the panel rule on the
/// panel and on its two halves. The halves' sum is kept; the difference is its indicator.
struct LogPartitionPanel {
    left: f64,
    right: f64,
    mass: f64,
    moment: f64,
    mass_gap: f64,
    moment_gap: f64,
}

/// `log C(a, N)` and `a·∂_a log C`: the log partition function of the relaxed ordered
/// Beta--Bernoulli prior on one column with shape `a` and effective row count `N`.
///
/// #2933 F45. The integrated scalar is `exp(−L) = a·B(M + a, N − M + 1)
/// = E_{π∼Beta(a,1)}[π^M (1 − π)^{N−M}]` with `M = Σ_i z_i`. At binary gates it sums to one;
/// over relaxed gates `z ∈ (0, 1)^N` its mass factorizes over rows inside the rate integral,
/// `∫₀¹ π^z (1 − π)^{1−z} dz = (2π − 1)/logit π =: g(π)`, so
///
/// ```text
/// C(a, N) = ∫_{(0,1)^N} exp(−L) dz = E_{π∼Beta(a,1)}[g(π)^N] ≤ 2^{−N},
/// ```
///
/// `0.197662`, `0.426278` and `0.326664` at `N = 1`, `a = 0.1, 1, 10`. The normalized negative
/// log prior is `L + log C`, and the logit change of variables leaves the mass unchanged. A
/// design-weighted scalar scores `M = Σ w_i z_i` against `N = Σ w_i`, the population it stands
/// in for, and its partition is `C(a, N)` at that `N`.
///
/// Coordinates. With `π = e^{−t}`, `t ∼ Exp(a)`; with `w = ln(a·t)`,
///
/// ```text
/// C(a, N) = ∫_ℝ exp(w − e^w)·G(e^{w − ln a})^N dw,   ln G(e^τ) = log_relaxed_gate_mass(τ),
/// a·∂_a log C = a·(1/a + E[ln π]) = 1 − E[e^w],
/// ```
///
/// where the expectation is under the normalized integrand. For every `a` in `(0, f64::MAX]`
/// the whole integrand stays representable: the prior factor lives at `w = O(1)`, the peak of
/// `G^N` at `w = ln(a·ln 2)`, and between them the integrand varies on unit scales in `w`.
///
/// Quadrature. The interval `[w_lo, w_hi]` is cut at the mode of the log integrand (found by
/// bisection on its slope, which is `1` as `w → −∞` and `−∞` as `w → ∞`) and outward at widths
/// `2^j/√(1 + N)`, so the first rules already sample the peak, whose width in `w` is of order
/// `1/√N`. Each panel carries the 16-point rule on its two halves, with the difference from
/// the whole-panel rule as its indicator. For an integrand analytic around a panel, halving
/// the panel scales the rule's error by roughly `2^{−32}`, so indicators summing below `√ε` of
/// the mass and of the moment `∫ e^w·integrand` leave errors far below `ε` of both. The panel
/// with the largest relative indicator is bisected until then. With `G ≤ 1/2`, the tails are
/// bounded by `2^{−N}e^{w_lo}` and `2^{−N}e^{2w_lo}` on the left and by `2^{−N}e^{−e^{w_hi}}`
/// and `2^{−N}(1 + e^{w_hi})e^{−e^{w_hi}}` on the right; the interval is widened until each
/// bound is below `ε` of the computed mass or moment. Exhausting
/// [`LOG_PARTITION_MAX_PANELS`] is a refusal, not a value.
pub fn ordered_beta_bernoulli_log_partition(
    shape: f64,
    rows: f64,
) -> Result<OrderedBetaBernoulliLogPartition, String> {
    if !(shape.is_finite() && shape > 0.0 && rows.is_finite() && rows > 0.0) {
        return Err(format!(
            "ordered Beta--Bernoulli log partition needs a finite positive shape and row count; \
             got a={shape}, N={rows}"
        ));
    }
    let log_shape = shape.ln();
    let log_integrand = |w: f64| w - w.exp() + rows * log_relaxed_gate_mass(w - log_shape);
    let slope = |w: f64| 1.0 - w.exp() + rows * log_relaxed_gate_mass_slope(w - log_shape);
    let unbracketed = || {
        format!(
            "ordered Beta--Bernoulli log partition: no mode bracket for a={shape}, N={rows}"
        )
    };
    let mut lower = -1.0_f64;
    while !(slope(lower) > 0.0) {
        lower *= 2.0;
        if !lower.is_finite() {
            return Err(unbracketed());
        }
    }
    let mut upper = 1.0_f64;
    while !(slope(upper) < 0.0) {
        upper *= 2.0;
        if !upper.is_finite() {
            return Err(unbracketed());
        }
    }
    loop {
        let middle = 0.5 * (lower + upper);
        if middle <= lower || middle >= upper {
            break;
        }
        if slope(middle) > 0.0 {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let mode = 0.5 * (lower + upper);
    let peak = log_integrand(mode);
    let width = (1.0 + rows).sqrt().recip();
    let (nodes, weights) = gauss_legendre(LOG_PARTITION_PANEL_ORDER);
    // The integrand divided by its value at the mode, so no mass underflows as `N` grows.
    let rule = |left: f64, right: f64| {
        let centre = 0.5 * (left + right);
        let half = 0.5 * (right - left);
        let mut mass = 0.0;
        let mut moment = 0.0;
        for (node, weight) in nodes.iter().zip(&weights) {
            let w = centre + half * node;
            let f = (log_integrand(w) - peak).exp();
            mass += weight * f;
            moment += weight * w.exp() * f;
        }
        (half * mass, half * moment)
    };
    let priced = |left: f64, right: f64| -> Result<LogPartitionPanel, String> {
        let middle = 0.5 * (left + right);
        if !(middle > left && middle < right) {
            return Err(format!(
                "ordered Beta--Bernoulli log partition: panel [{left}, {right}] cannot be halved \
                 for a={shape}, N={rows}"
            ));
        }
        let (coarse_mass, coarse_moment) = rule(left, right);
        let (left_mass, left_moment) = rule(left, middle);
        let (right_mass, right_moment) = rule(middle, right);
        let mass = left_mass + right_mass;
        let moment = left_moment + right_moment;
        Ok(LogPartitionPanel {
            left,
            right,
            mass,
            moment,
            mass_gap: (coarse_mass - mass).abs(),
            moment_gap: (coarse_moment - moment).abs(),
        })
    };
    let integrate = |left: f64, right: f64| -> Result<(f64, f64), String> {
        let mut cuts = vec![left, mode, right];
        let mut step = width;
        while mode - step > left {
            cuts.push(mode - step);
            step *= 2.0;
        }
        step = width;
        while mode + step < right {
            cuts.push(mode + step);
            step *= 2.0;
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let mut panels = cuts
            .windows(2)
            .map(|pair| priced(pair[0], pair[1]))
            .collect::<Result<Vec<_>, _>>()?;
        loop {
            let mass: f64 = panels.iter().map(|panel| panel.mass).sum();
            let moment: f64 = panels.iter().map(|panel| panel.moment).sum();
            let mass_gap: f64 = panels.iter().map(|panel| panel.mass_gap).sum();
            let moment_gap: f64 = panels.iter().map(|panel| panel.moment_gap).sum();
            if !(mass > 0.0 && moment > 0.0 && mass.is_finite() && moment.is_finite()) {
                return Err(format!(
                    "ordered Beta--Bernoulli log partition: non-positive or non-finite mass \
                     {mass} / moment {moment} for a={shape}, N={rows}"
                ));
            }
            let tolerance = f64::EPSILON.sqrt();
            if mass_gap <= tolerance * mass && moment_gap <= tolerance * moment {
                return Ok((mass, moment));
            }
            if panels.len() >= LOG_PARTITION_MAX_PANELS {
                return Err(format!(
                    "ordered Beta--Bernoulli log partition did not converge within \
                     {LOG_PARTITION_MAX_PANELS} panels for a={shape}, N={rows}: relative \
                     indicators {:.3e} (mass), {:.3e} (moment)",
                    mass_gap / mass,
                    moment_gap / moment
                ));
            }
            let indicator =
                |panel: &LogPartitionPanel| panel.mass_gap / mass + panel.moment_gap / moment;
            let mut worst = 0;
            for (index, panel) in panels.iter().enumerate() {
                if indicator(panel) > indicator(&panels[worst]) {
                    worst = index;
                }
            }
            let panel = panels.swap_remove(worst);
            let middle = 0.5 * (panel.left + panel.right);
            panels.push(priced(panel.left, middle)?);
            panels.push(priced(middle, panel.right)?);
        }
    };
    let log_epsilon = f64::EPSILON.ln();
    let log_mass_ceiling = -rows * LN_2;
    let mut left = mode - width;
    let mut right = mode + width;
    loop {
        let (mass, moment) = integrate(left, right)?;
        let log_mass = mass.ln() + peak;
        let log_moment = moment.ln() + peak;
        let need_left = (log_epsilon + log_mass - log_mass_ceiling)
            .min(0.5 * (log_epsilon + log_moment - log_mass_ceiling));
        let reach = (-log_epsilon + log_mass_ceiling - log_mass.min(log_moment)).max(0.0);
        let need_right = (reach + reach.ln_1p() + 1.0).ln();
        if !(need_left.is_finite() && need_right.is_finite()) {
            return Err(format!(
                "ordered Beta--Bernoulli log partition: tail bound is not finite for a={shape}, \
                 N={rows}"
            ));
        }
        if left <= need_left && right >= need_right {
            return Ok(OrderedBetaBernoulliLogPartition {
                value: log_mass,
                log_shape_derivative: 1.0 - moment / mass,
            });
        }
        // Widening only adds mass, so the bounds recomputed on the wider interval are no
        // tighter; the extra `width` keeps rounding in the totals from asking again.
        left = left.min(need_left - width);
        right = right.max(need_right + width);
    }
}

#[cfg(test)]
mod log_partition_2933_tests {
    //! #2933 F45 — the relaxed ordered Beta--Bernoulli energy `L` has mass `C(a, N)` over the
    //! relaxed gates, and `L + log C` must integrate to one. These pin the production partition
    //! against integrals it does not share: tensor Gauss--Legendre over the gates of the
    //! production energy (`N ≤ 2`), the audit's independent values, a rate-space integral at
    //! larger `N`, and the closed-form leading behaviour at extreme shapes.
    use super::*;

    /// `∫_{(0,1)^dim} f` by tensor Gauss--Legendre of order `order`.
    fn legendre_cube(order: usize, dim: usize, mut f: impl FnMut(&[f64]) -> f64) -> f64 {
        let (nodes, weights) = gauss_legendre(order);
        let mut index = vec![0usize; dim];
        let mut point = vec![0.0; dim];
        let mut total = 0.0;
        loop {
            let mut weight = 1.0;
            for axis in 0..dim {
                point[axis] = 0.5 * (nodes[index[axis]] + 1.0);
                weight *= 0.5 * weights[index[axis]];
            }
            total += weight * f(&point);
            let mut axis = 0;
            loop {
                if axis == dim {
                    return total;
                }
                index[axis] += 1;
                if index[axis] < order {
                    break;
                }
                index[axis] = 0;
                axis += 1;
            }
        }
    }

    fn logit(z: f64) -> f64 {
        (z / (1.0 - z)).ln()
    }

    /// The production energy of one row-major `(N, K)` gate matrix at unit temperature.
    fn energy(penalty: &OrderedBetaBernoulliPenalty, gates: &[f64], rho: &[f64]) -> f64 {
        let target = Array1::from_iter(gates.iter().map(|&z| logit(z)));
        penalty.value(target.view(), ArrayView1::from(rho))
    }

    #[test]
    fn relaxed_mass_matches_the_audit_and_gate_space_integrals_2933() {
        // Audit §12 check 34 (Gauss--Legendre orders 128 and 256 over z, independently).
        for (shape, audit) in [(0.1, 0.197661809898), (1.0, 0.426278398818), (10.0, 0.326663966267)]
        {
            let partition = ordered_beta_bernoulli_log_partition(shape, 1.0).expect("a, N > 0");
            let penalty = OrderedBetaBernoulliPenalty::new(1, shape, 1.0, false);
            let gate_space = legendre_cube(256, 1, |z| (-energy(&penalty, z, &[])).exp());
            for (label, reference, bar) in
                [("audit", audit, 1.0e-11), ("gate-space", gate_space, 1.0e-10)]
            {
                assert!(
                    (partition.value.exp() / reference - 1.0).abs() <= bar,
                    "C({shape}, 1) = {:.12e}, {label} reference {reference:.12e}",
                    partition.value.exp()
                );
            }
        }
        for shape in [0.5, 1.7, 6.0] {
            let partition = ordered_beta_bernoulli_log_partition(shape, 2.0).expect("a, N > 0");
            let penalty = OrderedBetaBernoulliPenalty::new(1, shape, 1.0, false);
            let gate_space = legendre_cube(128, 2, |z| (-energy(&penalty, z, &[])).exp());
            assert!(
                (partition.value.exp() / gate_space - 1.0).abs() <= 1.0e-9,
                "C({shape}, 2) = {:.12e}, gate-space reference {gate_space:.12e}",
                partition.value.exp()
            );
        }
    }

    /// At larger `N` against `a·∫₀¹ π^{a−1} g(π)^N dπ` by composite Gauss--Legendre over the rate.
    #[test]
    fn relaxed_mass_matches_a_rate_space_integral_at_larger_row_counts_2933() {
        let (nodes, weights) = gauss_legendre(16);
        let panels = 512;
        for shape in [1.0, 3.0] {
            for rows in [50.0, 400.0] {
                let mut reference = 0.0;
                for panel in 0..panels {
                    let left = panel as f64 / panels as f64;
                    let half = 0.5 / panels as f64;
                    for (node, weight) in nodes.iter().zip(&weights) {
                        let rate = left + half * (node + 1.0);
                        let mass = (2.0 * rate - 1.0) / logit(rate);
                        reference +=
                            half * weight * shape * rate.powf(shape - 1.0) * mass.powf(rows);
                    }
                }
                let partition = ordered_beta_bernoulli_log_partition(shape, rows).expect("a, N > 0");
                assert!(
                    (partition.value - reference.ln()).abs() <= 1.0e-10,
                    "log C({shape}, {rows}) = {:.12e}, rate-space reference {:.12e}",
                    partition.value,
                    reference.ln()
                );
            }
        }
    }

    #[test]
    fn log_shape_derivative_is_a_central_difference_of_the_log_partition_2933() {
        let h = 1.0e-4;
        for shape in [1.0e-6_f64, 0.3, 7.0, 1.0e5] {
            for rows in [0.4, 1.0, 37.5, 1.0e4] {
                let at = |log_shape: f64| {
                    ordered_beta_bernoulli_log_partition(log_shape.exp(), rows)
                        .expect("a, N > 0")
                        .value
                };
                let difference = (at(shape.ln() + h) - at(shape.ln() - h)) / (2.0 * h);
                let derivative = ordered_beta_bernoulli_log_partition(shape, rows)
                    .expect("a, N > 0")
                    .log_shape_derivative;
                assert!(
                    (derivative - difference).abs() <= 1.0e-7 * (1.0 + derivative.abs()),
                    "a·∂_a log C at a={shape}, N={rows}: {derivative:.12e} vs central difference \
                     {difference:.12e}"
                );
            }
        }
    }

    /// Extreme shapes stay representable and meet `C ≤ 2^{−N}` and the leading behaviour
    /// `C ≈ a·ln(1/a)` as `a → 0` and `C ≈ 1/ln a` as `a → ∞` at `N = 1`.
    #[test]
    fn extreme_shapes_keep_the_partition_finite_and_bounded_2933() {
        for shape in [1.0e-300, 1.0e300] {
            for rows in [1.0e-3, 1.0, 1.0e5] {
                let partition = ordered_beta_bernoulli_log_partition(shape, rows).expect("a, N > 0");
                assert!(
                    partition.value.is_finite() && partition.log_shape_derivative.is_finite(),
                    "a={shape}, N={rows}: {partition:?}"
                );
                assert!(
                    partition.value <= -rows * LN_2 + 1.0e-12 * (1.0 + rows),
                    "a={shape}, N={rows}: log C = {} exceeds −N ln 2",
                    partition.value
                );
            }
        }
        let small = ordered_beta_bernoulli_log_partition(1.0e-300, 1.0).expect("a, N > 0");
        let log_leading_small = (1.0e-300_f64).ln() + (-(1.0e-300_f64).ln()).ln();
        assert!(
            (small.value - log_leading_small).abs() <= 1.0e-2,
            "log C(1e-300, 1) = {}, leading {log_leading_small}",
            small.value
        );
        let large = ordered_beta_bernoulli_log_partition(1.0e300, 1.0).expect("a, N > 0");
        let log_leading_large = -(1.0e300_f64).ln().ln();
        assert!(
            (large.value - log_leading_large).abs() <= 1.0e-2,
            "log C(1e300, 1) = {}, leading {log_leading_large}",
            large.value
        );
    }

    /// With a learnable concentration over two columns, the energy plus the penalty's
    /// partition integrates to one over the gates, and the complete log-concentration
    /// derivative (energy plus partition) has zero mean under it.
    #[test]
    fn learnable_penalty_normalizes_and_its_score_has_zero_mean_2933() {
        for alpha in [0.6_f64, 1.3, 4.0] {
            let penalty = OrderedBetaBernoulliPenalty::new(2, 1.0, 1.0, true);
            let rho = [alpha.ln()];
            let placeholder = Array1::<f64>::zeros(2);
            let rho_view = ArrayView1::from(&rho[..]);
            let (log_partition, partition_slope) = penalty
                .log_partition(placeholder.view(), rho_view)
                .expect("untempered penalty");
            let density = |z: &[f64]| {
                let target = Array1::from_iter(z.iter().map(|&gate| logit(gate)));
                let p = (-penalty.value(target.view(), rho_view) - log_partition).exp();
                let score = penalty.grad_rho(target.view(), rho_view)[0] + partition_slope[0];
                (p, score)
            };
            let mass = legendre_cube(128, 2, |z| density(z).0);
            let score = legendre_cube(128, 2, |z| {
                let (p, score) = density(z);
                p * score
            });
            assert!(
                (mass - 1.0).abs() <= 1.0e-9,
                "learnable ordered Beta--Bernoulli prior at α={alpha} has mass {mass:.12e}"
            );
            assert!(
                score.abs() <= 1.0e-9,
                "log-concentration score at α={alpha} has prior mean {score:.12e}"
            );
        }
        let mut tempered = OrderedBetaBernoulliPenalty::new(2, 1.0, 1.0, false);
        tempered.weight = 3.0;
        assert!(
            tempered
                .log_partition(Array1::<f64>::zeros(2).view(), Array1::<f64>::zeros(0).view())
                .is_err(),
            "a tempered energy has no computed partition and must be refused"
        );
    }

    /// #2627 — the partition kept per thread answers only its own `(k_max, α, N)`. Alternating
    /// concentrations and row weights each return the words of their own column quadratures,
    /// and the three inputs give three different values, so a kept entry answering the wrong
    /// input would fail the equality.
    #[test]
    fn a_kept_log_partition_answers_only_its_own_concentration_and_row_count_2627() {
        let unweighted = OrderedBetaBernoulliPenalty::new(3, 1.0, 1.0, true);
        let weighted = unweighted
            .clone()
            .with_row_weights(Some(&[0.5, 1.0, 2.0, 0.25]));
        let target = Array1::<f64>::zeros(3 * 4);
        let quadratures = |penalty: &OrderedBetaBernoulliPenalty, rho: &[f64]| {
            let alpha = penalty.resolved_alpha(ArrayView1::from(rho));
            let rows: f64 = (0..4).map(|row| penalty.row_weight(row)).sum();
            let a_col = penalty.column_beta_shapes(alpha);
            let mut value = 0.0;
            let mut slope = 0.0;
            for k in 0..penalty.k_max {
                let partition =
                    ordered_beta_bernoulli_log_partition(a_col[k], rows).expect("a, N > 0");
                value += partition.value;
                slope += partition.log_shape_derivative * ((k + 1) as f64) * (a_col[k] + 1.0)
                    / (alpha + 1.0);
            }
            (value, slope)
        };
        let low = [1.3_f64.ln()];
        let high = [4.0_f64.ln()];
        let mut values = Vec::new();
        for (penalty, rho) in [
            (&unweighted, &low),
            (&unweighted, &high),
            (&unweighted, &low),
            (&weighted, &low),
            (&unweighted, &low),
        ] {
            let (value, slope) = penalty
                .log_partition(target.view(), ArrayView1::from(&rho[..]))
                .expect("untempered penalty");
            let (expected_value, expected_slope) = quadratures(penalty, &rho[..]);
            assert_eq!(
                (value.to_bits(), slope[0].to_bits()),
                (expected_value.to_bits(), expected_slope.to_bits()),
                "log partition at ρ={rho:?}, weighted={}: kept {value:e}/{:e}, quadrature \
                 {expected_value:e}/{expected_slope:e}",
                penalty.row_weights.is_some(),
                slope[0],
            );
            values.push(value);
        }
        assert!(
            values[0] != values[1] && values[0] != values[3] && values[1] != values[3],
            "control failed: the three inputs share a value {values:?}, so a kept entry answering \
             the wrong input could not be told apart"
        );
    }
}
