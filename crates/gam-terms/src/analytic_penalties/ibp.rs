use super::*;

/// IBP-MAP active-set prior over SAE-manifold assignment logits.
///
/// Infinite GPFA / IBP-GPFA in neuroscience uses an Indian Buffet Process
/// prior over factor loadings to infer both a potentially unbounded factor
/// set and which factors contribute at each observation. The relevant
/// diagnosis carries over directly to SAE-manifold assignment: ordinary ARD
/// selects one global factor set for all observations, not a different set
/// for each observation. A per-row IBP active set is the established GPFA
/// remedy, adapted here to gamfit's REML/MAP engine with a finite truncation
/// and deterministic concrete relaxation.
///
/// The target is row-major `(N, K)` logits. For MAP we use a deterministic
/// binary-concrete score `z_ik = sigmoid(logit_ik / tau)`, with optional
/// Gumbel temperature annealing across outer iterations. Each column has
/// `pi_k ~ Beta(alpha / K, 1)` and `z_ik | pi_k ~ Bernoulli(pi_k)`. We plug in
/// the columnwise Beta-Bernoulli MAP `pi_k` from the relaxed active mass, so
/// the penalty is a gauge-fixing prior: it breaks the per-row
/// interchangeability of atom indices by making each row choose a sparse
/// binary-ish subset rather than assigning every atom a soft nonzero weight.
#[derive(Debug, Clone)]
pub struct IBPAssignmentPenalty {
    pub k_max: usize,
    pub alpha: f64,
    pub tau: f64,
    pub temperature_schedule: Option<GumbelTemperatureSchedule>,
    pub learnable_alpha: bool,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl IBPAssignmentPenalty {
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
        }
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

    fn concrete_temperature(&self) -> f64 {
        self.tau
    }

    fn concrete_logits(&self, target: ArrayView1<'_, f64>) -> Array1<f64> {
        let tau = self.concrete_temperature();
        let mut out = Array1::<f64>::zeros(target.len());
        for i in 0..target.len() {
            let x = target[i] / tau;
            out[i] = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            };
        }
        out
    }

    fn pi_map(&self, z: ArrayView1<'_, f64>, alpha: f64) -> Array1<f64> {
        let n = z.len() / self.k_max;
        let a = alpha / self.k_max as f64;
        // Use the Beta-Bernoulli posterior mean, `(M_k + a)/(N + a + 1)`, as
        // the smooth empirical-π plug-in.  The old MAP plug-in
        // `(M_k + a - 1)/(N + a - 1)` is pinned to the zero boundary whenever
        // `a < 1` and the fitted mass is sparse, which incorrectly makes
        // `∂π_k/∂M_k = 0` and drops the IBP cross-row Woodbury curvature for
        // precisely the active-sparsity regimes this prior is meant to model.
        let mut pi = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let mut active_mass = 0.0;
            for row in 0..n {
                active_mass += z[row * self.k_max + k];
            }
            let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let raw = (active_mass + a) / denom;
            pi[k] = raw.clamp(IBP_INTERIOR_TOL, 1.0 - IBP_INTERIOR_TOL);
        }
        pi
    }

    /// Exact third-derivative channels of [`Self::hessian_diag`] with respect to
    /// the logits, for the SAE outer-ρ log-det adjoint Γ (#1006).
    ///
    /// `hessian_diag` returns, per row `i` and column `k`, the on-diagonal
    /// curvature
    ///
    /// ```text
    ///   H_ik = w · [ sd_k · J_ik²  +  score_k · c_ik ],
    /// ```
    ///
    /// with `J_ik = z(1−z)/τ` the logit→concrete jacobian, `c_ik =
    /// z(1−z)(1−2z)/τ²` the second jacobian, and the column scalars
    /// `score_k`, `sd_k = ∂score_k/∂M_k` exactly as assembled there
    /// (`M_k = Σ_i z_ik` is the column active mass, `π_k(M_k)` the plug-in
    /// stick-breaking MAP). Because `π_k` couples every row in column `k`, the
    /// logit derivative splits into a row-local direct-`z` channel and a global
    /// empirical-`M_k` channel:
    ///
    /// ```text
    ///   ∂H_ik/∂ℓ_wk = δ_iw · (∂_z H_ik)·J_ik   +   (∂_M H_ik) · J_wk,
    ///   ∂_z H_ik = w·J_ik·[ sd_k·2J_ik·(1−2z)/τ + score_k·(1−6z+6z²)/τ² ],
    ///   ∂_M H_ik = w·[ sdd_k · J_ik²  +  sd_k · c_ik ],
    ///   sdd_k = ∂sd_k/∂M_k = ∂²score_k/∂M_k².
    /// ```
    ///
    /// `local_logit_third[i*K+k] = (∂_z H_ik)·J_ik` is the row-diagonal third
    /// derivative; `m_channel[i*K+k] = ∂_M H_ik` and `z_jac[i*K+k] = J_ik` let
    /// the caller form, per column, `C_k = Σ_i (H⁻¹)_ik,ik · ∂_M H_ik` and
    /// distribute `C_k · J_wk` to every row `w` (the cross-row coupling the
    /// row-local primitive cannot see). All boundary clamps (`pi_jac = 0` at the
    /// `π_k` clamp) ride the same convention as `hessian_diag`, so the channels
    /// are zero exactly where the assembled curvature is constant in `M_k`.
    #[must_use]
    pub fn hessian_diag_logit_third_channels(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> IbpHessianDiagThirdChannels {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);

        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }

        let mut score = Array1::<f64>::zeros(self.k_max);
        let mut score_derivative = Array1::<f64>::zeros(self.k_max);
        let mut score_second_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            let pi_jac = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                1.0 / denom
            } else {
                0.0
            };
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            let pi_score = bce_pi_score + beta_pi_score;
            let pi_score_derivative = -1.0 / pk + (mass + a - 1.0) * pi_jac / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n as f64 - mass) * pi_jac / ((1.0 - pk) * (1.0 - pk));
            let direct_z_score = ((1.0 - pk) / pk).ln();
            let implicit_pi_score = pi_score * pi_jac;
            score[k] = direct_z_score + implicit_pi_score;
            let direct_z_score_derivative = pi_jac * (-1.0 / pk - 1.0 / (1.0 - pk));
            score_derivative[k] = direct_z_score_derivative + pi_score_derivative * pi_jac;

            // sdd_k = ∂score_derivative_k/∂M_k, holding the explicit per-row z
            // fixed (the same partial `hessian_diag` takes for score/sd). With
            // π_k = (M_k+a−1)/D clamped, ∂π_k/∂M_k = pi_jac (0 at the clamp):
            //   ∂(direct_z_score_derivative)/∂M = pi_jac²·(1/π² − 1/(1−π)²),
            //   ∂(pi_score_derivative)/∂M = pi_jac·[ 2/π² − 2(M+a−1)·pi_jac/π³
            //                                        − 2/(1−π)² + 2(n−M)·pi_jac/(1−π)³ ].
            let one_minus = 1.0 - pk;
            let ddzd = pi_jac * pi_jac * (1.0 / (pk * pk) - 1.0 / (one_minus * one_minus));
            let dpisd = 2.0 / (pk * pk)
                - 2.0 * (mass + a - 1.0) * pi_jac / (pk * pk * pk)
                - 2.0 / (one_minus * one_minus)
                + 2.0 * (n as f64 - mass) * pi_jac / (one_minus * one_minus * one_minus);
            score_second_derivative[k] = ddzd + dpisd * pi_jac;
        }

        let len = target.len();
        let mut z_jac = Array1::<f64>::zeros(len);
        let mut local_logit_third = Array1::<f64>::zeros(len);
        let mut m_channel = Array1::<f64>::zeros(len);
        let mut logit_curvature = Array1::<f64>::zeros(len);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let jac = zk * (1.0 - zk) * inv_tau;
                let c_ik = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                // ∂_z J = (1−2z)/τ, ∂_z c = (1−6z+6z²)/τ².
                let dz_j = (1.0 - 2.0 * zk) * inv_tau;
                let dz_c = (1.0 - 6.0 * zk + 6.0 * zk * zk) * inv_tau2;
                let dz_h = score_derivative[k] * 2.0 * jac * dz_j + score[k] * dz_c;
                z_jac[start + k] = jac;
                local_logit_third[start + k] = self.weight * jac * dz_h;
                m_channel[start + k] = self.weight
                    * (score_second_derivative[k] * jac * jac + score_derivative[k] * c_ik);
                logit_curvature[start + k] = c_ik;
            }
        }

        // #1038 cross-row Woodbury: per column `k`, the EXACT IBP Hessian has the
        // rank-one cross-row block `H_(i,k),(j,k) += w·s'_k·z'_ik·z'_jk` (for all
        // `i,j`, including `i=j`). `cross_row_d[k] = w·s'_k = w·score_derivative_k`
        // is its scalar `D`-coefficient; `z_jac` already holds `u_k`'s entries
        // `z'_ik`. The consumer subtracts the `i=j` self term from `H₀` (the
        // assembled diagonal carries it) and adds the FULL rank-one via the
        // determinant lemma, so value/logdet/adjoint all differentiate one
        // operator. Built from the SAME `(score_derivative, z_jac)` source as the
        // diagonal `hessian_diag` and the `m_channel`/`local_logit_third` third
        // tensor — the issue's one-operator non-negotiable.
        let mut cross_row_d = Array1::<f64>::zeros(self.k_max);
        let mut cross_row_dd = Array1::<f64>::zeros(self.k_max);
        // logα-derivative of the rank-one coefficient, ONLY for learnable-α. It is
        // the SAME `∂s'_k/∂logα` the diagonal `hessian_diag_log_alpha_derivative`
        // uses, so the cross-row channel of the log-det α-gradient matches the
        // diagonal instead of injecting the undifferentiated value `s'_k`.
        let mut cross_row_d_logalpha = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            cross_row_d[k] = self.weight * score_derivative[k];
            // ∂d_k/∂M_k = w·∂s'_k/∂M_k = w·s''_k.
            cross_row_dd[k] = self.weight * score_second_derivative[k];
            if self.learnable_alpha {
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let mass = active_mass[k];
                let raw = (mass + a) / denom;
                // Same interior gate / zero-π-Jacobian convention as
                // `hessian_diag_log_alpha_derivative`; at the clamp the derivative is 0.
                if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                    let one_minus = 1.0 - pk;
                    let dpi_da = (n as f64 + 1.0 - mass) / (denom * denom);
                    let inv_p = 1.0 / pk;
                    let inv_q = 1.0 / one_minus;
                    let a_channel = inv_p + inv_q;
                    let d_a_channel_da = dpi_da * (-inv_p * inv_p + inv_q * inv_q);
                    let d_score_derivative_da =
                        a_channel / (denom * denom) - d_a_channel_da / denom;
                    cross_row_d_logalpha[k] = self.weight * a * d_score_derivative_da;
                }
            }
        }

        IbpHessianDiagThirdChannels {
            k_max: self.k_max,
            z_jac,
            local_logit_third,
            m_channel,
            cross_row_d,
            cross_row_dd,
            cross_row_d_logalpha,
            logit_curvature,
        }
    }

    /// Mixed derivative `∂/∂ℓ_ik [∂F/∂ρ_alpha]` for learnable-alpha IBP.
    ///
    /// This differentiates the implemented energy in [`Self::value`]. At the
    /// empirical-π interior, the BCE and `(a-1) log π` implicit-π terms cancel in
    /// `∂F/∂a`, leaving the normalized Beta(a,1) channel. At the probability
    /// clamp, the same zero-π-Jacobian convention as [`Self::grad_target`] and
    /// [`Self::hessian_diag`] applies.
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
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let raw = (active_mass[k] + a) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let z_jac = zk * (1.0 - zk) / tau;
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                out[start + k] = -self.weight * a * pi_jac[k] * z_jac / pk;
            }
        }
        out
    }

    /// `∂ hessian_diag / ∂ρ_alpha` for learnable-alpha IBP.
    ///
    /// The SAE log-det trace differentiates the diagonal returned by
    /// [`Self::hessian_diag`]. This channel differentiates that exact diagonal
    /// with respect to the learnable-alpha log-coordinate while holding logits
    /// fixed. IBP columns remain independent, so within-row off-diagonals are zero.
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
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut d_score = Array1::<f64>::zeros(self.k_max);
        let mut d_score_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw <= IBP_INTERIOR_TOL || raw >= 1.0 - IBP_INTERIOR_TOL {
                continue;
            }
            let one_minus = 1.0 - pk;
            let dpi_da = (n as f64 + 1.0 - mass) / (denom * denom);
            let dpi_drho = a * dpi_da;
            let d_score_dpi = -1.0 / pk - 1.0 / one_minus;
            d_score[k] = d_score_dpi * dpi_drho;

            let inv_p = 1.0 / pk;
            let inv_q = 1.0 / one_minus;
            let a_channel = inv_p + inv_q;
            let d_a_channel_da = dpi_da * (-inv_p * inv_p + inv_q * inv_q);
            let d_score_derivative_da = a_channel / (denom * denom) - d_a_channel_da / denom;
            d_score_derivative[k] = a * d_score_derivative_da;
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let z_jac = zk * (1.0 - zk) * inv_tau;
                let z_second = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] =
                    self.weight * (d_score_derivative[k] * z_jac * z_jac + d_score[k] * z_second);
            }
        }
        out
    }
}

/// Exact logit third-derivative channels of [`IBPAssignmentPenalty::hessian_diag`]
/// for the SAE outer-ρ log-det adjoint Γ (#1006). Row-major `(N, K)` layout.
#[derive(Debug, Clone)]
pub struct IbpHessianDiagThirdChannels {
    /// Number of columns `K` (atoms) in the row-major logit layout.
    pub k_max: usize,
    /// `J_ik = z(1−z)/τ`, the per-logit concrete jacobian (row-major `N·K`).
    pub z_jac: Array1<f64>,
    /// `(∂_z H_ik)·J_ik`: the row-local direct-`z` third derivative of the
    /// assembled diagonal curvature `H_ik` (row-major `N·K`).
    pub local_logit_third: Array1<f64>,
    /// `∂_M H_ik`: the empirical-`M_k` channel of `H_ik`. Contract against the
    /// selected-inverse diagonal per column, then distribute `C_k·J_wk` to every
    /// row `w` (row-major `N·K`).
    pub m_channel: Array1<f64>,
    /// `cross_row_d[k] = w·s'_k`: the scalar `D`-coefficient of the per-column
    /// cross-row rank-one Hessian block `H_(i,k),(j,k) = w·s'_k·z'_ik·z'_jk`
    /// (#1038). Paired with `u_k = z_jac[·,k]` this is the exact column-`k`
    /// Woodbury update `d_k·u_k·u_kᵀ` (full outer product, `i=j` included).
    /// Length `K`.
    pub cross_row_d: Array1<f64>,
    /// `cross_row_dd[k] = w·s''_k = ∂d_k/∂M_k`: the empirical-mass derivative of
    /// the column Woodbury coefficient (#1416). Since `d_k = w·s'_k(M_k)` and
    /// `∂M_k/∂ℓ_wk = J_wk`, the θ-derivative of the rank-one block carries
    /// `∂d_k/∂ℓ_wk = cross_row_dd[k]·J_wk`. Length `K`.
    pub cross_row_dd: Array1<f64>,
    /// `cross_row_d_logalpha[k] = w·∂s'_k/∂logα`: the **logα-derivative** of the
    /// column Woodbury coefficient `d_k`, for the learnable-α log-det ρ-gradient
    /// `½ tr(H⁻¹ ∂H_p/∂logα)`. The cross-row rank-one block is
    /// `W_k = d_k·u_k u_kᵀ` with `u_k = z_jac[·,k]` α-independent (the concrete
    /// Jacobian depends on logits, not α), so `∂W_k/∂logα = (∂d_k/∂logα)·u_k u_kᵀ`
    /// and the correct cross-row coefficient is `∂d_k/∂logα`, NOT the value `d_k`.
    /// The diagonal channel (`hessian_diag_log_alpha_derivative`) already uses this
    /// α-derivative; the off-diagonal must match it. Zero unless `learnable_alpha`
    /// (the fixed-α path scales linearly with `λ`, so `∂H_p/∂ρ = H_p` uses the
    /// value `cross_row_d` instead). Length `K`.
    pub cross_row_d_logalpha: Array1<f64>,
    /// `logit_curvature[i*K+k] = c_ik = ∂J_ik/∂ℓ_ik = z(1−z)(1−2z)/τ²`: the
    /// per-logit second derivative of the concrete map (#1416). The
    /// cross-row rank-one block's `J_ik` factors depend on `ℓ_ik`, so its
    /// θ-derivative carries `∂J_ik/∂ℓ_wk = δ_iw·c_ik`. Row-major `N·K`.
    pub logit_curvature: Array1<f64>,
}

impl AnalyticPenalty for IBPAssignmentPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut acc = 0.0;
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                acc -= zk * pk.ln() + (1.0 - zk) * (1.0 - pk).ln();
            }
        }
        for k in 0..self.k_max {
            // Normalized Beta(a,1) density is a*pi^(a-1), so its negative
            // log contribution is -ln(a) - (a - 1) ln(pi). The normalizer is
            // constant only for fixed alpha; keep it in both modes so the energy
            // has one mathematical definition across configurations.
            acc -= a.ln();
            acc -= (a - 1.0) * pi[k].ln();
        }
        self.weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut out = Array1::<f64>::zeros(target.len());
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut pi_score = Array1::<f64>::zeros(self.k_max);
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            pi_score[k] = bce_pi_score + beta_pi_score;
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let direct_z_score = ((1.0 - pk) / pk).ln();
                let implicit_pi_score = pi_score[k] * pi_jac[k];
                out[start + k] =
                    self.weight * (direct_z_score + implicit_pi_score) * zk * (1.0 - zk) / tau;
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
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau2 = 1.0 / (tau * tau);
        let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut pi_score = Array1::<f64>::zeros(self.k_max);
        let mut pi_score_derivative = Array1::<f64>::zeros(self.k_max);
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            pi_score[k] = bce_pi_score + beta_pi_score;
            pi_score_derivative[k] = -1.0 / pk + (mass + a - 1.0) * pi_jac[k] / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n as f64 - mass) * pi_jac[k] / ((1.0 - pk) * (1.0 - pk));
        }
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let direct_z_score = ((1.0 - pk) / pk).ln();
                let implicit_pi_score = pi_score[k] * pi_jac[k];
                let score = direct_z_score + implicit_pi_score;
                let direct_z_score_derivative = pi_jac[k] * (-1.0 / pk - 1.0 / (1.0 - pk));
                let score_derivative =
                    direct_z_score_derivative + pi_score_derivative[k] * pi_jac[k];
                let z_jac = zk * (1.0 - zk) / tau;
                out[start + k] = self.weight
                    * (score_derivative * z_jac * z_jac
                        + score * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2);
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
            "IBPAssignmentPenalty::hvp dimension mismatch"
        );
        let alpha = self.resolved_alpha(rho);
        let a = alpha / self.k_max as f64;
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let denom = (n as f64 + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);

        // Column aggregates (active_mass, pi_jac, pi_score, pi_score_derivative,
        // score, score_derivative). These are identical to hessian_diag and
        // share the same interior / boundary-clamp convention, so the on-row
        // diagonal returned by hvp(·, eⱼ) agrees with hessian_diag bit-for-bit.
        let mut active_mass = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                active_mass[k] += z[start + k];
            }
        }
        let mut score = Array1::<f64>::zeros(self.k_max);
        let mut score_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            let pi_jac = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                1.0 / denom
            } else {
                0.0
            };
            let bce_pi_score = -mass / pk + (n as f64 - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            let pi_score = bce_pi_score + beta_pi_score;
            let pi_score_derivative = -1.0 / pk + (mass + a - 1.0) * pi_jac / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n as f64 - mass) * pi_jac / ((1.0 - pk) * (1.0 - pk));
            let direct_z_score = ((1.0 - pk) / pk).ln();
            let implicit_pi_score = pi_score * pi_jac;
            score[k] = direct_z_score + implicit_pi_score;
            let direct_z_score_derivative = pi_jac * (-1.0 / pk - 1.0 / (1.0 - pk));
            score_derivative[k] = direct_z_score_derivative + pi_score_derivative * pi_jac;
        }

        // Within-column block structure: pi[k] and active_mass[k] depend on
        // EVERY row in column k, so the per-column Hessian block is a rank-1
        // perturbation of a diagonal,
        //
        //   H[(j,k), (j',k)] = w · score_derivative[k] · z_jac[j,k] · z_jac[j',k]
        //                    + δ_{jj'} · w · score[k] · (1-2z[j,k]) · z(1-z) / τ²,
        //
        // where z_jac[j,k] = z(1-z)/τ at row j in column k. Different
        // columns are decoupled (pi[k] depends only on column k), so the
        // full Hessian is block-diagonal by column.
        //
        // For an input vector v, the rank-1 contribution collapses to a
        // single per-column scalar sₖ = Σⱼ z_jac[j,k] · v[j,k]:
        //
        //   (Hv)[j,k] = w · score_derivative[k] · z_jac[j,k] · sₖ
        //             + w · score[k] · (1-2z[j,k]) · z(1-z)/τ² · v[j,k].
        //
        // The default diagonal-only hvp drops the off-diagonal rank-1 piece,
        // which empirically carries ≈85% of the operator's Frobenius norm.
        let mut s_per_col = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let zjac = zk * (1.0 - zk) * inv_tau;
                s_per_col[k] += zjac * v[start + k];
            }
        }
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            for k in 0..self.k_max {
                let zk = z[start + k];
                let zjac = zk * (1.0 - zk) * inv_tau;
                let rank1 = score_derivative[k] * zjac * s_per_col[k];
                let c_diag = score[k] * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] = self.weight * (rank1 + c_diag * v[start + k]);
            }
        }
        out
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_alpha {
            return Array1::<f64>::zeros(0);
        }
        let alpha = self.resolved_alpha(rho);
        let z = self.concrete_logits(target);
        let pi = self.pi_map(z.view(), alpha);
        let mut sum_log_pi = 0.0;
        for &pk in pi.iter() {
            sum_log_pi += pk
                .clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP)
                .ln();
        }
        Array1::from_vec(vec![
            -self.weight * (alpha * sum_log_pi / self.k_max as f64 + self.k_max as f64),
        ])
    }

    fn rho_count(&self) -> usize {
        usize::from(self.learnable_alpha)
    }

    fn name(&self) -> &str {
        "ibp_assignment_map"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.temperature_schedule.as_mut() {
            self.tau = schedule.current_tau(iter);
            schedule.iter_count = iter + 1;
        }
        advance_scalar_weight(&mut self.weight, &mut self.weight_schedule, iter);
    }
}
