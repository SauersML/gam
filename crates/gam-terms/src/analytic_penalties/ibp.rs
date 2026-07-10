use super::*;

/// IBP active-set prior over SAE-manifold assignment logits (posterior-mean
/// plug-in; NOT a MAP — see [`SPEC.md`] line 3).
///
/// Infinite GPFA / IBP-GPFA in neuroscience uses an Indian Buffet Process
/// prior over factor loadings to infer both a potentially unbounded factor
/// set and which factors contribute at each observation. The relevant
/// diagnosis carries over directly to SAE-manifold assignment: ordinary ARD
/// selects one global factor set for all observations, not a different set
/// for each observation. A per-row IBP active set is the established GPFA
/// remedy, adapted here to gamfit's REML/LAML engine with a finite truncation
/// and deterministic concrete relaxation.
///
/// # One posterior-mean model
///
/// This penalty is the negative-log-**posterior** of the same Bernoulli
/// indicators returned by the forward gate. The gate is the concrete posterior
/// mean `z_ik = sigmoid(ell_ik/tau)`. The ordered activation-rate prior is
/// `pi_k ~ Beta(a_k, 1)`, with
/// `E[pi_k] = a_k/(a_k+1) = mu_k = (alpha/(alpha+1))^(k+1)` and
/// `a_k = mu_k/(1-mu_k)`. Crucially, `mu_k` is scored here exactly once; it is
/// not multiplied into the final reconstructed function as a second prior.
///
/// The target is row-major `(N, K)` logits. We use a deterministic
/// binary-concrete score `z_ik = sigmoid(logit_ik / tau)`, with optional
/// Gumbel temperature annealing across outer iterations, and
/// `z_ik | pi_k ~ Bernoulli(pi_k)`. We plug in the columnwise Beta-Bernoulli
/// **posterior mean** `pi_k = (M_k + a_k)/(N + a_k + 1)` from the relaxed active
/// mass `M_k = Σ_i z_ik`, so the penalty is a gauge-fixing prior: it breaks the
/// per-row interchangeability of atom indices by making each row choose a sparse
/// binary-ish subset rather than assigning every atom a soft nonzero weight.
/// As `α → 0` every `μ_k → 0`, so the prior collapses the active set toward the
/// null (SPEC.md line 13); `α` is learnable for an empirical-Bayes fit.
#[derive(Debug, Clone)]
pub struct IBPAssignmentPenalty {
    pub k_max: usize,
    pub alpha: f64,
    pub tau: f64,
    pub temperature_schedule: Option<GumbelTemperatureSchedule>,
    pub learnable_alpha: bool,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
    /// Optional per-column (per-atom, length `k_max`) mask of FIXED columns that
    /// are INERT in this prior — their logits are held constant (an ungated /
    /// frozen SAE atom), so they contribute nothing to the value, gradient, or
    /// curvature and their per-column Beta normalizer is excluded. `None`
    /// (the default) scores every column, bit-for-bit the historical path. Set
    /// via a plain field assignment, like `weight`. (#Bug4)
    pub fixed_columns: Option<Vec<bool>>,
    /// #991 design-honesty per-row weights `w_i` (mean-1 inclusion weights of a
    /// design-weighted subsample; `None` ⇒ every row weighs 1, bit-for-bit the
    /// historical path). The IBP prior is NOT row-separable — the plug-in
    /// posterior mean couples rows through the column active mass — so the
    /// weights enter in exactly two places, identically in every channel:
    /// the column mass becomes `M_k = Σ_i w_i·z_ik` with effective row count
    /// `N_eff = Σ_i w_i` (all column scalars `π̂, score, s′, s″` and their
    /// α-derivatives read these), and each per-row emission carries `w_i`
    /// (energy `w_i·bce_i`, so the design-weighted concrete Jacobian is
    /// `u_ik = w_i·J_ik` and every gradient/curvature/third-channel slot is the
    /// exact derivative of the weighted energy — value/gradient/Hessian remain
    /// one operator).
    pub row_weights: Option<std::sync::Arc<[f64]>>,
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
            fixed_columns: None,
            row_weights: None,
        }
    }

    /// Install #991 design-honesty per-row weights (see [`Self::row_weights`]).
    #[must_use]
    pub fn with_row_weights(mut self, weights: Option<&[f64]>) -> Self {
        self.row_weights = weights.map(|w| std::sync::Arc::from(w.to_vec()));
        self
    }

    /// Per-row design weight `w_i` (`1.0` when no weights are installed).
    #[inline]
    fn row_weight(&self, row: usize) -> f64 {
        self.row_weights.as_ref().map_or(1.0, |w| w[row])
    }

    /// Design-weighted column active mass `M_k = Σ_i w_i·z_ik` and effective row
    /// count `N_eff = Σ_i w_i` — the ONE source every channel's column scalars
    /// read, so the weighted objective and all its derivatives share a single
    /// plug-in `π̂_k = (M_k + a_k)/(N_eff + a_k + 1)`.
    fn weighted_active_mass(&self, z: ArrayView1<'_, f64>) -> (Array1<f64>, f64) {
        let n = z.len() / self.k_max;
        let mut mass = Array1::<f64>::zeros(self.k_max);
        let mut n_eff = 0.0_f64;
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

    /// Whether column (atom) `k` is a FIXED / inert column excluded from this
    /// prior (see [`Self::fixed_columns`]). `false` for every column when no mask
    /// is set.
    #[inline]
    fn column_is_fixed(&self, k: usize) -> bool {
        self.fixed_columns
            .as_ref()
            .and_then(|m| m.get(k).copied())
            .unwrap_or(false)
    }

    /// Per-column Beta(`a_k`, 1) shapes and their prior means `μ_k`.
    ///
    /// `μ_k = (α/(α+1))^(k+1)` is the ordered stick-breaking prior mean and
    /// `a_k = μ_k/(1−μ_k)` is the corresponding Beta shape. Returns
    /// `(a_col, mu)`, each length `K`. `μ_k` is floored at the smallest positive
    /// normal so every
    /// atom keeps a live gradient path; the sparsity ordering is preserved.
    fn column_beta_shapes(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        let log_ratio = (alpha / (alpha + 1.0)).ln();
        let mut a_col = Array1::<f64>::zeros(self.k_max);
        let mut mu = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let m = (((k + 1) as f64) * log_ratio).exp().max(f64::MIN_POSITIVE);
            mu[k] = m;
            a_col[k] = m / (1.0 - m);
        }
        (a_col, mu)
    }

    /// `∂a_k/∂ρ` with `ρ = logα` and `α = α_base·e^ρ`, for the learnable-α
    /// channels. With `μ_k = (α/(α+1))^(k+1)` we have `∂μ_k/∂ρ = μ_k·(k+1)/(α+1)`
    /// and
    /// `a_k = μ_k/(1−μ_k)` gives `∂a_k/∂ρ = (∂μ_k/∂ρ)/(1−μ_k)²`.
    fn column_beta_shape_rho_deriv(&self, alpha: f64, mu: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut da = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let m = mu[k];
            let dmu = m * ((k + 1) as f64) / (alpha + 1.0);
            da[k] = dmu / ((1.0 - m) * (1.0 - m));
        }
        da
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

    /// Columnwise Beta-Bernoulli **posterior mean** `π̂_k = (M_k + a_k)/(N + a_k + 1)`
    /// of the per-atom activation rate `π_k ~ Beta(a_k, 1)`, with `M_k = Σ_i z_ik`
    /// the relaxed active mass and `a_col` the per-column Beta shapes from
    /// [`Self::column_beta_shapes`]. This is a genuine posterior MEAN (SPEC.md
    /// line 3), NOT a MAP: the mode `(M_k + a_k − 1)/(N + a_k − 1)` is pinned to the
    /// zero boundary whenever `a_k < 1` and the fitted mass is sparse, which would
    /// make `∂π̂_k/∂M_k = 0` and drop the IBP cross-row Woodbury curvature for
    /// precisely the active-sparsity regimes this prior is meant to model. The
    /// prior mean `E[π_k] = a_k/(a_k+1) = μ_k` supplies ordered shrinkage solely
    /// through this prior.
    fn pi_posterior_mean(&self, z: ArrayView1<'_, f64>, a_col: ArrayView1<'_, f64>) -> Array1<f64> {
        let (active_mass, n_eff) = self.weighted_active_mass(z);
        let mut pi = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let denom = (n_eff + a_col[k] + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let raw = (active_mass[k] + a_col[k]) / denom;
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
    /// When `majorize` is `true`, every channel is the exact derivative of the
    /// **PSD Loewner-majorized** column block `max(w·s',0)·J Jᵀ +
    /// diag(max(w·s·c,0))` (gam#2144), not the raw indefinite IBP Hessian. The
    /// majorizer clamps the rank-one coefficient `w·s'` and the per-slot diagonal
    /// `w·s·c` to their positive parts; since `max(x,0)` has derivative
    /// `𝟙[x>0]`, each smooth channel is gated by the sign of the piece it
    /// differentiates — the rank-one/`s'` pieces by the per-column `s'>0`, the
    /// diagonal/`s·c` pieces by the per-slot `s·c>0`. This is the metric-first
    /// consistency partner of the assembly's `ibp_psd_majorized_hdiag` + clamped
    /// Woodbury `d`; the majorization is UNCONDITIONAL (#2144/#1038), so the
    /// θ-adjoint/ρ-trace differentiate the SAME majorized operator the evidence
    /// log-det factors on every IBP path.
    #[must_use]
    pub fn hessian_diag_logit_third_channels(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        majorize: bool,
    ) -> IbpHessianDiagThirdChannels {
        let alpha = self.resolved_alpha(rho);
        // Per-column Beta(a_k, 1) shapes carrying the ordered prior mean μ_k.
        let (a_col, _mu) = self.column_beta_shapes(alpha);
        let tau = self.concrete_temperature();
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        let n = z.len() / self.k_max;

        // #991 design weights: column scalars read the weighted mass/N_eff, and
        // every per-slot emission below uses the weighted Jacobian `u = w·J`.
        let (active_mass, n_eff) = self.weighted_active_mass(z.view());

        let mut score = Array1::<f64>::zeros(self.k_max);
        let mut score_derivative = Array1::<f64>::zeros(self.k_max);
        let mut score_second_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let a = a_col[k];
            let denom = (n_eff + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            let pi_jac = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                1.0 / denom
            } else {
                0.0
            };
            let bce_pi_score = -mass / pk + (n_eff - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            let pi_score = bce_pi_score + beta_pi_score;
            let pi_score_derivative = -1.0 / pk + (mass + a - 1.0) * pi_jac / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n_eff - mass) * pi_jac / ((1.0 - pk) * (1.0 - pk));
            let direct_z_score = ((1.0 - pk) / pk).ln();
            let implicit_pi_score = pi_score * pi_jac;
            score[k] = direct_z_score + implicit_pi_score;
            let direct_z_score_derivative = pi_jac * (-1.0 / pk - 1.0 / (1.0 - pk));
            score_derivative[k] = direct_z_score_derivative + pi_score_derivative * pi_jac;

            // sdd_k = ∂score_derivative_k/∂M_k, holding the explicit per-row z
            // fixed (the same partial `hessian_diag` takes for score/sd). With
            // the posterior-mean plug-in π_k=(M_k+a)/D clamped, ∂π_k/∂M_k = pi_jac
            // (=1/D, independent of the +a/−1 numerator shift; 0 at the clamp).
            // score_derivative_k = direct_z_score_derivative + pi_score_derivative·pi_jac,
            // so with pi_jac constant in M:
            //   ∂(direct_z_score_derivative)/∂M = pi_jac²·(1/π² − 1/(1−π)²) = ddzd,
            //   ∂(pi_score_derivative)/∂M       = pi_jac·dpisd,
            //   sdd_k = ddzd + pi_jac·∂(pi_score_derivative)/∂M = ddzd + pi_jac²·dpisd.
            // (The earlier `ddzd + pi_jac·dpisd` dropped one pi_jac factor on the
            // implicit-π channel, so `cross_row_dd` disagreed with the FD of
            // `cross_row_d` w.r.t. the logits.)
            let one_minus = 1.0 - pk;
            let ddzd = pi_jac * pi_jac * (1.0 / (pk * pk) - 1.0 / (one_minus * one_minus));
            let dpisd = 2.0 / (pk * pk)
                - 2.0 * (mass + a - 1.0) * pi_jac / (pk * pk * pk)
                - 2.0 / (one_minus * one_minus)
                + 2.0 * (n_eff - mass) * pi_jac / (one_minus * one_minus * one_minus);
            score_second_derivative[k] = ddzd + dpisd * pi_jac * pi_jac;
        }

        let len = target.len();
        let mut z_jac = Array1::<f64>::zeros(len);
        let mut local_logit_third = Array1::<f64>::zeros(len);
        let mut m_channel = Array1::<f64>::zeros(len);
        let mut logit_curvature = Array1::<f64>::zeros(len);
        for row in 0..n {
            let start = row * self.k_max;
            // #991: the design-weighted energy is `Σ_i w_i·bce_i` with
            // `M_k = Σ_i w_i·z_ik`, so its concrete-Jacobian carrier is
            // `u_ik = w_i·J_ik` and the per-slot curvature is
            // `H_ik = w·[s'_k·u_ik² + s_k·(w_i·c_ik)]`. The slots below emit the
            // exact derivatives of THAT weighted operator: `z_jac` carries `u`,
            // `logit_curvature` carries `∂u_ik/∂ℓ_ik = w_i·c_ik`, and the local /
            // M-channels differentiate `H_ik` with the same folding — consumers
            // (Woodbury rank-one, θ-adjoint, ∂M/∂ℓ distribution) read the folded
            // slots and stay convention-free of the weights.
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                // #Bug4: fixed/inert column ⇒ every log-det adjoint channel slot
                // stays zero (matches `value`/`grad_target`/`hessian_diag`).
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let jac = zk * (1.0 - zk) * inv_tau;
                let u_ik = w_i * jac;
                let c_ik = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                // ∂_z J = (1−2z)/τ, ∂_z c = (1−6z+6z²)/τ².
                let dz_j = (1.0 - 2.0 * zk) * inv_tau;
                let dz_c = (1.0 - 6.0 * zk + 6.0 * zk * zk) * inv_tau2;
                // Split the channel into its rank-one-self (`s'`) and diagonal
                // (`s·c`) contributions so the majorizer can gate each by the
                // sign of the piece it clamps (gam#2144). `gate_col` = 𝟙[s'>0]
                // (the whole column's rank-one, incl. `s''` deriv), `gate_sc` =
                // 𝟙[s·c>0] (this slot's diagonal). Raw path keeps both = 1.
                let (gate_col, gate_sc) = if majorize {
                    let g_col = f64::from(score_derivative[k] > 0.0);
                    let g_sc = f64::from(score[k] * c_ik > 0.0);
                    (g_col, g_sc)
                } else {
                    (1.0, 1.0)
                };
                // ∂_z H_ik = w_i·w·[s'·2u·∂_zJ + s·∂_zc]; the trailing `jac` is
                // the ∂z/∂ℓ chain (unweighted J — the weight already rode in via
                // `w_i·`), so `local = (∂_z H)·J = w·u·[s'·2u·dz_j + s·dz_c]`.
                let dz_h =
                    gate_col * score_derivative[k] * 2.0 * u_ik * dz_j + gate_sc * score[k] * dz_c;
                z_jac[start + k] = u_ik;
                local_logit_third[start + k] = self.weight * u_ik * dz_h;
                // ∂_M H_ik = w·[s''·u_ik² + s'·(w_i·c_ik)]; the consumer's
                // `∂M_k/∂ℓ_wk = u_wk` factor comes from the folded `z_jac` slot.
                m_channel[start + k] = self.weight
                    * (gate_col * score_second_derivative[k] * u_ik * u_ik
                        + gate_sc * score_derivative[k] * w_i * c_ik);
                logit_curvature[start + k] = w_i * c_ik;
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
            // #Bug4: fixed/inert column ⇒ zero rank-one Woodbury coefficients.
            if self.column_is_fixed(k) {
                continue;
            }
            // Under majorization the rank-one coefficient is `max(w·s'_k,0)` and its
            // logit/mass derivative is gated by `𝟙[s'_k>0]` (the clamp's subgradient),
            // matching the per-column `gate_col` used for the diagonal channels above.
            if majorize && score_derivative[k] <= 0.0 {
                cross_row_d[k] = 0.0;
                cross_row_dd[k] = 0.0;
            } else {
                cross_row_d[k] = self.weight * score_derivative[k];
                // ∂d_k/∂M_k = w·∂s'_k/∂M_k = w·s''_k.
                cross_row_dd[k] = self.weight * score_second_derivative[k];
            }
        }
        if self.learnable_alpha {
            // `cross_row_d[k] = w·score_derivative_k`, so its ρ (=logα) derivative
            // is `w·∂score_derivative_k/∂ρ`. Share the SAME total-ρ-derivative
            // primitive as `hessian_diag_log_alpha_derivative` so the cross-row
            // off-diagonal channel of the log-det α-gradient matches the diagonal
            // exactly (one operator, one derivative).
            let (_d_score, d_score_derivative) = self.learnable_alpha_score_rho_derivs(target, rho);
            for k in 0..self.k_max {
                // #Bug4: fixed/inert column ⇒ zero α-derivative channel.
                if self.column_is_fixed(k) {
                    continue;
                }
                // Gated by the same `𝟙[s'_k>0]` clamp subgradient: where the
                // rank-one is clamped off, its α-derivative is 0.
                cross_row_d_logalpha[k] = if majorize && score_derivative[k] <= 0.0 {
                    0.0
                } else {
                    self.weight * d_score_derivative[k]
                };
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
        // Per-column Beta shapes a_k and their ρ-derivatives da_k/dρ (single-model
        // prior mean a_k/(a_k+1) = μ_k.
        let (a_col, mu) = self.column_beta_shapes(alpha);
        let da_col = self.column_beta_shape_rho_deriv(alpha, mu.view());
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        // #991: weighted mass/N_eff — one plug-in π̂ shared with every channel.
        let (active_mass, n_f) = self.weighted_active_mass(z.view());
        // mixed[i·K+k] = ∂(grad_rho)/∂ℓ_ik = ∂²F/∂ρ∂ℓ_ik. With grad_rho's per-column
        // summand G_k = pi_score_k·(a_k'(N+1−M_k)/D_k²) + a_k'·(−1/a_k − ln π_k),
        // where a_k' = ∂a_k/∂ρ, and ℓ_ik's only reach through M_k (∂M_k/∂ℓ_ik =
        // J_ik), this is (∂G_k/∂M_k)·J_ik:
        //   ∂G_k/∂M_k = (a_k'/D_k²)·[ PSD_k·(N+1−M_k) − pi_score_k ] − a_k'·pi_jac/π_k,
        // where PSD_k = ∂pi_score_k/∂M_k (D_k, a_k both ρ-quantities, constant in M).
        let mut d_g_dm = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            // #Bug4: fixed/inert column ⇒ zero mixed derivative (its column is
            // excluded from `grad_rho`, so the ℓ-derivative of G_k is zero too).
            if self.column_is_fixed(k) {
                continue;
            }
            let a = a_col[k];
            let da = da_col[k];
            let denom = (n_f + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw <= IBP_INTERIOR_TOL || raw >= 1.0 - IBP_INTERIOR_TOL {
                continue;
            }
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let one_minus = 1.0 - pk;
            let pj = 1.0 / denom;
            let pi_score = -mass / pk + (n_f - mass) / one_minus - (a - 1.0) / pk;
            let psd = -1.0 / pk + (mass + a - 1.0) * pj / (pk * pk) - 1.0 / one_minus
                + (n_f - mass) * pj / (one_minus * one_minus);
            d_g_dm[k] =
                (da / (denom * denom)) * (psd * (n_f + 1.0 - mass) - pi_score) - da * pj / pk;
        }
        for row in 0..n {
            let start = row * self.k_max;
            // #991: `∂M_k/∂ℓ_ik = w_i·J_ik` — the mixed channel differentiates
            // the SAME weighted mass every other channel reads.
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                let zk = z[start + k];
                let z_jac = zk * (1.0 - zk) / tau;
                out[start + k] = self.weight * d_g_dm[k] * w_i * z_jac;
            }
        }
        out
    }

    /// Total ρ (=logα) derivatives of the per-column score scalars `score_k` and
    /// `score_derivative_k` that [`Self::hessian_diag`] assembles, for
    /// learnable-α. Returns `(d_score, d_score_derivative)`, each length `K`,
    /// zero outside the π interior.
    ///
    /// The plug-in `π_k=(M_k+a)/(N+a+1)` is the posterior MEAN, not the energy's
    /// stationary point (the mode `(M_k+a−1)/(N+a−1)`), so π is a genuine
    /// function of α whose implicit channel does not vanish by any envelope
    /// argument. With `α(ρ)=α_base·e^ρ` we have `dα/dρ=α`, `a=α/K` so `da/dρ=a`,
    /// `D=N+a+1` so `dD/dρ=a`, and hence
    ///   `π'  = a·(N+1−M)/D²`   (=∂π/∂ρ),
    ///   `π_jac' = −a/D²`       (=∂(1/D)/∂ρ).
    /// Every term of `score`/`score_derivative` is differentiated through BOTH
    /// the explicit `a` and the implicit `π(ρ)`. This is the single primitive
    /// behind both [`Self::hessian_diag_log_alpha_derivative`] (diagonal) and the
    /// `cross_row_d_logalpha` off-diagonal channel, so the two agree by
    /// construction.
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
        // Per-column Beta shapes a_k and da_k/dρ. With a_k=μ_k/(1−μ_k),
        // μ_k=(α/(α+1))^(k+1), D_k=N+a_k+1 so dD_k/dρ=da_k/dρ, and
        //   π' = (da_k/dρ)·(N+1−M)/D_k²   (=∂π/∂ρ),
        //   pj' = −(da_k/dρ)/D_k²         (=∂(1/D_k)/∂ρ).
        // The leading da/dρ factor is the per-column da_col[k]; explicit value-a
        // is a_col[k]. (Previously a_k=α/K gave da/dρ=a.)
        let (a_col, mu) = self.column_beta_shapes(alpha);
        let da_col = self.column_beta_shape_rho_deriv(alpha, mu.view());
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        // #991: weighted mass/N_eff — the α-derivative scalars differentiate the
        // same design-weighted plug-in π̂ every other channel reads.
        let (active_mass, n_f) = self.weighted_active_mass(z.view());
        for k in 0..self.k_max {
            // #Bug4: fixed/inert column ⇒ zero ρ-derivative scalars (this masks
            // both `hessian_diag_log_alpha_derivative` and `cross_row_d_logalpha`).
            if self.column_is_fixed(k) {
                continue;
            }
            let a = a_col[k];
            let da = da_col[k];
            let denom = (n_f + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw <= IBP_INTERIOR_TOL || raw >= 1.0 - IBP_INTERIOR_TOL {
                continue;
            }
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let one_minus = 1.0 - pk;
            let pj = 1.0 / denom;
            let pi_p = da * (n_f + 1.0 - mass) / (denom * denom); // ∂π/∂ρ
            let pj_p = -da / (denom * denom); // ∂(1/D)/∂ρ
            // direct_z_score = ln((1−π)/π); pi_score = ∂F/∂π (BCE + Beta).
            let pi_score = -mass / pk + (n_f - mass) / one_minus - (a - 1.0) / pk;
            let pi_score_p = pi_p / (pk * pk) * (mass + a - 1.0)
                + (n_f - mass) * pi_p / (one_minus * one_minus)
                - da / pk;
            // score = direct_z_score + pi_score·pi_jac.
            d_score[k] = (-1.0 / pk - 1.0 / one_minus) * pi_p + pi_score_p * pj + pi_score * pj_p;
            // score_derivative = pi_jac·(D1 + PSD), D1 = −1/π − 1/(1−π).
            let d1 = -1.0 / pk - 1.0 / one_minus;
            let d1_p = pi_p / (pk * pk) - pi_p / (one_minus * one_minus);
            let psd = -1.0 / pk + (mass + a - 1.0) * pj / (pk * pk) - 1.0 / one_minus
                + (n_f - mass) * pj / (one_minus * one_minus);
            let psd_p =
                pi_p / (pk * pk) + da * pj / (pk * pk) + (mass + a - 1.0) * pj_p / (pk * pk)
                    - 2.0 * (mass + a - 1.0) * pj * pi_p / (pk * pk * pk)
                    - pi_p / (one_minus * one_minus)
                    + (n_f - mass) * pj_p / (one_minus * one_minus)
                    + 2.0 * (n_f - mass) * pj * pi_p / (one_minus * one_minus * one_minus);
            d_score_derivative[k] = pj_p * (d1 + psd) + pj * (d1_p + psd_p);
        }
        (d_score, d_score_derivative)
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
        let tau = self.concrete_temperature();
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;
        let z = self.concrete_logits(target);
        let n = z.len() / self.k_max;
        let (d_score, d_score_derivative) = self.learnable_alpha_score_rho_derivs(target, rho);
        for row in 0..n {
            let start = row * self.k_max;
            // #991: ∂ρ of the weighted diagonal H_ik = w·[s'·u² + s·(w_i·c)],
            // u = w_i·J — the α-derivative keeps the identical folding.
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                let zk = z[start + k];
                let z_jac = zk * (1.0 - zk) * inv_tau;
                let u_ik = w_i * z_jac;
                let z_second = zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
                out[start + k] = self.weight
                    * (d_score_derivative[k] * u_ik * u_ik + d_score[k] * w_i * z_second);
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
    /// `u_ik = w_i·J_ik`, the #991 design-weighted per-logit concrete jacobian
    /// (`J_ik = z(1−z)/τ`; `w_i = 1` without design weights). This is the
    /// rank-one carrier AND the `∂M_k/∂ℓ_ik` factor of the weighted operator,
    /// so consumers stay weight-convention-free. Row-major `N·K`.
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
    /// `logit_curvature[i*K+k] = w_i·c_ik`, `c_ik = ∂J_ik/∂ℓ_ik =
    /// z(1−z)(1−2z)/τ²`: the per-logit second derivative of the (#991
    /// design-weighted) concrete carrier, `∂u_ik/∂ℓ_ik = w_i·c_ik` (#1416). The
    /// cross-row rank-one block's `u_ik` factors depend on `ℓ_ik`, so its
    /// θ-derivative carries `∂u_ik/∂ℓ_wk = δ_iw·(w_i·c_ik)`. Row-major `N·K`.
    pub logit_curvature: Array1<f64>,
}

impl AnalyticPenalty for IBPAssignmentPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let alpha = self.resolved_alpha(rho);
        // Per-column Beta(a_k, 1) shapes; prior mean a_k/(a_k+1) = μ_k = the
        // ordered stick-breaking prior mean π_k.
        let (a_col, _mu) = self.column_beta_shapes(alpha);
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let mut acc = 0.0;
        for row in 0..n {
            let start = row * self.k_max;
            // #991: each retained row stands for `w_i` design rows, so its
            // Bernoulli score carries `w_i` — matching the weighted mass inside
            // `π̂` so value and every derivative describe ONE weighted energy.
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                // #Bug4: a fixed/inert column contributes nothing to the energy.
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                acc -= w_i * (zk * pk.ln() + (1.0 - zk) * (1.0 - pk).ln());
            }
        }
        for k in 0..self.k_max {
            // Normalized Beta(a_k, 1) density is a_k*pi^(a_k-1), so its negative
            // log contribution is -ln(a_k) - (a_k - 1) ln(pi). Kept in both modes
            // so the energy has one mathematical definition across configurations.
            // #Bug4: skip the per-column normalizer of fixed/inert columns too.
            if self.column_is_fixed(k) {
                continue;
            }
            let a = a_col[k];
            acc -= a.ln();
            acc -= (a - 1.0) * pi[k].ln();
        }
        self.weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let alpha = self.resolved_alpha(rho);
        let (a_col, _mu) = self.column_beta_shapes(alpha);
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        // #991: weighted mass/N_eff — the same plug-in π̂ the value scores.
        let (active_mass, n_eff) = self.weighted_active_mass(z.view());
        let mut pi_score = Array1::<f64>::zeros(self.k_max);
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let a = a_col[k];
            let denom = (n_eff + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
            let bce_pi_score = -mass / pk + (n_eff - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            pi_score[k] = bce_pi_score + beta_pi_score;
        }
        for row in 0..n {
            let start = row * self.k_max;
            // #991: ∂F/∂ℓ_ik = w_i·score_k·J_ik — BOTH the row's own `w_i·bce_i`
            // and the implicit-π̂ channel (∂M/∂ℓ_ik = w_i·J_ik) carry the same
            // w_i, so the slot is w_i × the column score.
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                // #Bug4: a fixed/inert column contributes nothing to the energy,
                // so its gradient block is exactly zero (matches `value`).
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let direct_z_score = ((1.0 - pk) / pk).ln();
                let implicit_pi_score = pi_score[k] * pi_jac[k];
                out[start + k] =
                    self.weight * w_i * (direct_z_score + implicit_pi_score) * zk * (1.0 - zk)
                        / tau;
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
        let (a_col, _mu) = self.column_beta_shapes(alpha);
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau2 = 1.0 / (tau * tau);
        // #991: weighted mass/N_eff — one plug-in π̂ shared with every channel.
        let (active_mass, n_eff) = self.weighted_active_mass(z.view());
        let mut pi_score = Array1::<f64>::zeros(self.k_max);
        let mut pi_score_derivative = Array1::<f64>::zeros(self.k_max);
        let mut pi_jac = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let a = a_col[k];
            let denom = (n_eff + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                pi_jac[k] = 1.0 / denom;
            }
            let bce_pi_score = -mass / pk + (n_eff - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            pi_score[k] = bce_pi_score + beta_pi_score;
            pi_score_derivative[k] = -1.0 / pk + (mass + a - 1.0) * pi_jac[k] / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n_eff - mass) * pi_jac[k] / ((1.0 - pk) * (1.0 - pk));
        }
        for row in 0..n {
            let start = row * self.k_max;
            // #991: H_ik = w·[s'_k·u_ik² + s_k·(w_i·c_ik)], u_ik = w_i·J_ik — the
            // exact second derivative of the weighted energy (the rank-one self
            // term carries w_i², the diagonal concrete-curvature term w_i).
            let w_i = self.row_weight(row);
            for k in 0..self.k_max {
                // #Bug4: a fixed/inert column contributes nothing to the energy,
                // so its curvature block is exactly zero (matches `value`).
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
                let direct_z_score = ((1.0 - pk) / pk).ln();
                let implicit_pi_score = pi_score[k] * pi_jac[k];
                let score = direct_z_score + implicit_pi_score;
                let direct_z_score_derivative = pi_jac[k] * (-1.0 / pk - 1.0 / (1.0 - pk));
                let score_derivative =
                    direct_z_score_derivative + pi_score_derivative[k] * pi_jac[k];
                let z_jac = zk * (1.0 - zk) / tau;
                let u_ik = w_i * z_jac;
                out[start + k] = self.weight
                    * (score_derivative * u_ik * u_ik
                        + score * w_i * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2);
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
        let (a_col, _mu) = self.column_beta_shapes(alpha);
        let tau = self.concrete_temperature();
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        let n = z.len() / self.k_max;
        let inv_tau = 1.0 / tau;
        let inv_tau2 = inv_tau * inv_tau;

        // Column aggregates (active_mass, pi_jac, pi_score, pi_score_derivative,
        // score, score_derivative). These are identical to hessian_diag and
        // share the same interior / boundary-clamp convention, so the on-row
        // diagonal returned by hvp(·, eⱼ) agrees with hessian_diag bit-for-bit.
        // #991: weighted mass/N_eff — one plug-in π̂ shared with every channel.
        let (active_mass, n_eff) = self.weighted_active_mass(z.view());
        let mut score = Array1::<f64>::zeros(self.k_max);
        let mut score_derivative = Array1::<f64>::zeros(self.k_max);
        for k in 0..self.k_max {
            let a = a_col[k];
            let denom = (n_eff + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            let pi_jac = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                1.0 / denom
            } else {
                0.0
            };
            let bce_pi_score = -mass / pk + (n_eff - mass) / (1.0 - pk);
            let beta_pi_score = -(a - 1.0) / pk;
            let pi_score = bce_pi_score + beta_pi_score;
            let pi_score_derivative = -1.0 / pk + (mass + a - 1.0) * pi_jac / (pk * pk)
                - 1.0 / (1.0 - pk)
                + (n_eff - mass) * pi_jac / ((1.0 - pk) * (1.0 - pk));
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
        // single per-column scalar sₖ = Σⱼ u[j,k] · v[j,k], with the #991
        // design-weighted carrier u[j,k] = w_j·z_jac[j,k]:
        //
        //   (Hv)[j,k] = w · score_derivative[k] · u[j,k] · sₖ
        //             + w · score[k] · w_j · (1-2z[j,k]) · z(1-z)/τ² · v[j,k],
        //
        // the exact Hessian-vector product of the weighted energy (rank-one
        // block u uᵀ, diagonal concrete curvature ×w_j).
        //
        // The default diagonal-only hvp drops the off-diagonal rank-1 piece,
        // which empirically carries ≈85% of the operator's Frobenius norm.
        let mut s_per_col = Array1::<f64>::zeros(self.k_max);
        for row in 0..n {
            let start = row * self.k_max;
            let w_row = self.row_weight(row);
            for k in 0..self.k_max {
                let zk = z[start + k];
                let zjac = zk * (1.0 - zk) * inv_tau;
                s_per_col[k] += w_row * zjac * v[start + k];
            }
        }
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_max;
            let w_row = self.row_weight(row);
            for k in 0..self.k_max {
                // #Bug4: fixed/inert column ⇒ zero Hessian block (columns are
                // decoupled, so skipping zeroes the whole block; matches `value`).
                if self.column_is_fixed(k) {
                    continue;
                }
                let zk = z[start + k];
                let u_jk = w_row * zk * (1.0 - zk) * inv_tau;
                let rank1 = score_derivative[k] * u_jk * s_per_col[k];
                let c_diag = score[k] * w_row * zk * (1.0 - zk) * (1.0 - 2.0 * zk) * inv_tau2;
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
        // Per-column Beta(a_k, 1) shapes a_k = μ_k/(1−μ_k) whose prior mean μ_k is
        // the ordered stick-breaking prior means, and their ρ
        // derivatives da_k/dρ — the single-model reconciliation (#4).
        let (a_col, mu) = self.column_beta_shapes(alpha);
        let da_col = self.column_beta_shape_rho_deriv(alpha, mu.view());
        let z = self.concrete_logits(target);
        let pi = self.pi_posterior_mean(z.view(), a_col.view());
        // #991: weighted mass/N_eff — this ρ-gradient differentiates the same
        // design-weighted energy `value` reports (column-level; the per-row w_i
        // already live inside `mass` and the weighted pi_score).
        let (active_mass, n_f) = self.weighted_active_mass(z.view());
        // ∂F/∂ρ, ρ = logα with α(ρ)=α_base·e^ρ. Each column has its OWN Beta shape
        // a_k=μ_k/(1−μ_k) (μ_k=(α/(α+1))^(k+1)), so
        // D_k=N+a_k+1 and da_k/dρ = column_beta_shape_rho_deriv. The plug-in π̂_k =
        // (M_k+a_k)/D_k is the posterior MEAN, NOT the mode, so the implicit-π
        // channel does not vanish and rides alongside the explicit Beta(a_k,1) one:
        //   ∂F/∂ρ = Σ_k [ (∂F/∂π̂_k)·(∂π̂_k/∂ρ)  +  (∂F/∂a_k)·(da_k/dρ) ],
        //   ∂F/∂π̂_k = pi_score_k,  ∂π̂_k/∂ρ = (da_k/dρ)·(N+1−M_k)/D_k² (0 at clamp),
        //   ∂F/∂a_k = −1/a_k − ln π̂_k.
        // (Previously a_k=α/K gave the scalar da/dρ=a; the per-column da_k/dρ makes
        // this the exact α-gradient of the reconciled single model.)
        let mut acc = 0.0;
        for k in 0..self.k_max {
            // #Bug4: a fixed/inert column contributes nothing to ∂F/∂ρ (its logits
            // are held constant), matching its exclusion from `value`.
            if self.column_is_fixed(k) {
                continue;
            }
            let a = a_col[k];
            let da = da_col[k];
            let denom = (n_f + a + 1.0).max(IBP_COUNT_DENOM_FLOOR);
            let mass = active_mass[k];
            let raw = (mass + a) / denom;
            let pk = pi[k].clamp(IBP_PROBABILITY_CLAMP, 1.0 - IBP_PROBABILITY_CLAMP);
            let dpi_drho = if raw > IBP_INTERIOR_TOL && raw < 1.0 - IBP_INTERIOR_TOL {
                da * (n_f + 1.0 - mass) / (denom * denom)
            } else {
                0.0
            };
            let pi_score = -mass / pk + (n_f - mass) / (1.0 - pk) - (a - 1.0) / pk;
            acc += pi_score * dpi_drho + (-1.0 / a - pk.ln()) * da;
        }
        Array1::from_vec(vec![self.weight * acc])
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
