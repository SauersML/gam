use super::*;

// ---------------------------------------------------------------------------
// Sparsity penalty
// ---------------------------------------------------------------------------

/// Sparsifier kernel.
///
/// * `SmoothedL1 { eps }` — `Σ_i sqrt(x_i² + ε²)`. The smoothing scale `ε`
///   may be REML-selected (`eps_rho_index = Some(_)`), in which case the
///   shrink rate `ε → 0` is governed by the marginal likelihood (Occam keeps
///   `ε` large when the data don't demand sharpness).
/// * `Hoyer` — `(√n · ‖x‖_1 − ‖x‖_2) / (√n − 1)`. Scale-invariant; encourages
///   absolute sparsity even when the global scale of `x` drifts.
/// * `Log { delta }` — `Σ_i log(1 + x_i² / δ²)`. Strongly concave; aggressive
///   sparsifier suitable for active-set / iterative-reweighted paths.
#[derive(Debug, Clone, Copy)]
pub enum SparsityKind {
    SmoothedL1 { eps: f64 },
    Hoyer,
    Log { delta: f64 },
}

/// Sparsity penalty on a slice of β (SAE codes) or ext-coords (soft atom assignments).
///
/// The smoothed-L¹ default `Σ_i sqrt(x_i² + ε²)` is the simplest analytic
/// option. Its gradient is `x_i / sqrt(x_i² + ε²)` (a smooth sign function),
/// and its Hessian is diagonal with entries `ε² / (x_i² + ε²)^{3/2}` — so
/// `hvp` is cheap and the inner Newton step inherits a benign block-diagonal
/// regularizer.
///
/// When to use: any time a parameter block carries a "this should be sparse"
/// prior — SAE atom codes (β slice), soft-routing weights on a latent
/// ext-coordinate slice. For SAE codes specifically, smoothed-L¹ with REML-selected `ε`
/// gives the principled relaxation of the L¹ objective without giving up
/// differentiability.
#[derive(Debug, Clone)]
pub struct SparsityPenalty {
    pub target_tier: PenaltyTier,
    pub kind: SparsityKind,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
    /// Index of `log strength` inside this penalty's local ρ view.
    pub strength_rho_index: usize,
    /// If `Some`, the index of `log ε` (or `log δ`) inside this penalty's
    /// local ρ view. If `None`, `ε` / `δ` is held fixed at the value baked
    /// into [`SparsityKind`].
    pub eps_rho_index: Option<usize>,
}

/// Entropy sparsity over row-wise softmax assignment logits.
///
/// This is the SAE-manifold soft-assignment penalty. The target is a flat
/// row-major `(N, K)` logit matrix. Assignments are
/// `a_i = softmax(logits_i / temperature)`, and the penalty is
///
/// ```text
///   lambda_sparse * sum_i H(a_i)
///   H(a_i) = -sum_k a_ik log a_ik
/// ```
///
/// Minimizing entropy drives each row toward a small active support while the
/// softmax keeps `a_ik >= 0` and `sum_k a_ik = 1`. The exact Hessian is dense
/// in each row and can be indefinite because entropy is concave in assignment
/// space, so callers must use the HVP rather than a diagonal Hessian shortcut.
#[derive(Debug, Clone)]
pub struct SoftmaxAssignmentSparsityPenalty {
    pub k_atoms: usize,
    pub temperature: f64,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl SoftmaxAssignmentSparsityPenalty {
    #[must_use]
    pub fn new(k_atoms: usize, temperature: f64) -> Self {
        assert!(k_atoms > 0);
        assert!(temperature > 0.0);
        Self {
            k_atoms,
            temperature,
            weight: 1.0,
            weight_schedule: None,
        }
    }

    impl_with_weight_schedule!(weight);

    fn softmax_row(&self, row: &[f64]) -> Vec<f64> {
        let inv_tau = 1.0 / self.temperature;
        let mut max_logit = f64::NEG_INFINITY;
        for (idx, &v) in row.iter().enumerate() {
            assert!(
                v.is_finite(),
                "SoftmaxAssignmentSparsityPenalty: non-finite logit at atom {idx}: {v}"
            );
            max_logit = max_logit.max(v);
        }
        let mut out = vec![0.0; self.k_atoms];
        let mut sum = 0.0;
        for i in 0..self.k_atoms {
            let v = ((row[i] - max_logit) * inv_tau).exp();
            out[i] = v;
            sum += v;
        }
        assert!(
            sum.is_finite() && sum > 0.0,
            "SoftmaxAssignmentSparsityPenalty: non-finite softmax normalizer"
        );
        for v in out.iter_mut() {
            *v /= sum;
        }
        out
    }

    /// Absolute row sums of the exact per-row dense entropy Hessian, used as a
    /// Gershgorin / diagonal-dominance PSD majorizer.
    ///
    /// The exact per-row Hessian wrt logits (symmetric, dense) is
    ///
    /// ```text
    ///   H_kj = (λ/τ²)·a_k·[ δ_kj·(m − L_k − 1) + a_j·(L_k + L_j + 1 − 2m) ],
    ///   L_k = ln a_k + 1,   m = Σ_j a_j L_j,
    /// ```
    ///
    /// whose diagonal coincides with [`AnalyticPenalty::hessian_diag`]. Entropy
    /// is concave in assignment space, so this block is indefinite (negative on
    /// near-uniform rows). Setting `D_kk = Σ_j |H_kj|` makes `D − H` symmetric
    /// with nonnegative diagonal and diagonally dominant
    /// (`D_kk − H_kk = |H_kk| − H_kk + Σ_{j≠k}|H_kj| ≥ Σ_{j≠k}|(D−H)_kj|`),
    /// hence PSD: `D ⪰ H` and `D ⪰ 0` both hold. `D` is a genuine PSD diagonal
    /// operator that dominates the dense Hessian's quadratic form — unlike the
    /// raw indefinite diagonal, which is neither PSD nor a faithful stand-in for
    /// the dense operator.
    fn psd_majorizer_abs_row_sums(&self, row: &[f64], scale: f64) -> Vec<f64> {
        let a = self.softmax_row(row);
        let k = self.k_atoms;
        let l: Vec<f64> = (0..k)
            .map(|i| a[i].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0)
            .collect();
        let m: f64 = (0..k).map(|i| a[i] * l[i]).sum();
        let mut d = vec![0.0_f64; k];
        for kk in 0..k {
            // Diagonal entry H_kk.
            let h_kk = scale * a[kk] * ((m - l[kk] - 1.0) + a[kk] * (2.0 * l[kk] + 1.0 - 2.0 * m));
            let mut acc = h_kk.abs();
            // Off-diagonal entries H_kj, j ≠ k.
            for jj in 0..k {
                if jj == kk {
                    continue;
                }
                let h_kj = scale * a[kk] * a[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m);
                acc += h_kj.abs();
            }
            d[kk] = acc;
        }
        d
    }

    /// Exact per-row dense softmax-entropy Hessian wrt the row's logits (#1038),
    /// scaled by `scale = λ/τ²`. Returns the symmetric `K×K` block
    ///
    /// ```text
    ///   H_kj = scale·a_k·[ δ_kj·(m − L_k − 1) + a_j·(L_k + L_j + 1 − 2m) ],
    ///   L_k = ln a_k + 1,   m = Σ_r a_r L_r,
    /// ```
    ///
    /// whose diagonal coincides with [`AnalyticPenalty::hessian_diag`] and whose
    /// quadratic form coincides with [`AnalyticPenalty::hvp`]. This is the dense
    /// block the Arrow-Schur row factor stores so the criterion's `log|H|` and
    /// the #1006 θ-adjoint differentiate the SAME operator (not just its
    /// diagonal). The entropy block alone is gauge-null (`H·𝟙 = 0`, softmax
    /// shift-invariance); callers must add it to the gauge-breaking data-fit
    /// row block before factoring — never factor it in isolation.
    #[must_use]
    pub fn row_dense_hessian(&self, row_logits: &[f64], scale: f64) -> Array2<f64> {
        let k = self.k_atoms;
        let a = self.softmax_row(row_logits);
        let l: Vec<f64> = (0..k)
            .map(|i| a[i].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0)
            .collect();
        let m: f64 = (0..k).map(|i| a[i] * l[i]).sum();
        let mut h = Array2::<f64>::zeros((k, k));
        for kk in 0..k {
            for jj in 0..k {
                let indicator = if kk == jj { 1.0 } else { 0.0 };
                h[[kk, jj]] = scale
                    * a[kk]
                    * (indicator * (m - l[kk] - 1.0) + a[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m));
            }
        }
        h
    }

    /// Derivative of the exact per-row dense entropy Hessian
    /// [`Self::row_dense_hessian`] with respect to a single row logit `z_w`,
    /// scaled by `scale = λ/τ²`. Returns the symmetric `K×K` block
    /// `∂H_kj/∂z_w`, the third-derivative tensor slice the #1006 θ-adjoint
    /// contracts against the row's selected inverse. Built from the SAME
    /// `(a, L, m)` as [`Self::row_dense_hessian`] (`∂a_r/∂z_w = a_r(δ_rw − a_w)/τ`),
    /// so value, logdet and adjoint stay on one branch.
    #[must_use]
    pub fn row_dense_hessian_logit_derivative(
        &self,
        row_logits: &[f64],
        scale: f64,
        w: usize,
    ) -> Array2<f64> {
        let k = self.k_atoms;
        let inv_tau = 1.0 / self.temperature;
        let a = self.softmax_row(row_logits);
        let l: Vec<f64> = (0..k)
            .map(|i| a[i].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0)
            .collect();
        let m: f64 = (0..k).map(|i| a[i] * l[i]).sum();
        // ∂a_r/∂z_w = a_r (δ_rw − a_w)/τ ; ∂L_r/∂z_w = (∂a_r/∂z_w)/a_r.
        let da: Vec<f64> = (0..k)
            .map(|r| a[r] * (if r == w { 1.0 } else { 0.0 } - a[w]) * inv_tau)
            .collect();
        let dl: Vec<f64> = (0..k)
            .map(|r| da[r] / a[r].max(ENTROPY_LOG_PROBABILITY_FLOOR))
            .collect();
        let dm: f64 = (0..k).map(|r| da[r] * l[r] + a[r] * dl[r]).sum();
        let mut dh = Array2::<f64>::zeros((k, k));
        for kk in 0..k {
            for jj in 0..k {
                let indicator = if kk == jj { 1.0 } else { 0.0 };
                // bracket = δ_kj(m − L_k − 1) + a_j(L_k + L_j + 1 − 2m).
                let bracket =
                    indicator * (m - l[kk] - 1.0) + a[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m);
                let dbracket = indicator * (dm - dl[kk])
                    + da[jj] * (l[kk] + l[jj] + 1.0 - 2.0 * m)
                    + a[jj] * (dl[kk] + dl[jj] - 2.0 * dm);
                dh[[kk, jj]] = scale * (da[kk] * bracket + a[kk] * dbracket);
            }
        }
        dh
    }

    /// Per-row softmax **Fisher-information metric** `G = scale·(diag(a) − a aᵀ)`
    /// over the row's logits, with `a = softmax(row_logits)` and
    /// `scale = λ/τ²` (#1190). Returns the symmetric `K×K` block
    ///
    /// ```text
    ///   G_kj = scale·a_k·(δ_kj − a_j).
    /// ```
    ///
    /// This is the PSD curvature operator the manifold-SAE evidence Hessian uses
    /// in place of the (indefinite) exact entropy Hessian
    /// [`Self::row_dense_hessian`]. `G` is a covariance/Gram matrix, hence
    /// exactly PSD and smooth in the logits, so the per-row evidence block stays
    /// PD by construction and the downstream Faddeev–Popov deflation never fires
    /// on the entropy block. Because the entropy penalty is a FIXED prior whose
    /// stationary point is set by its (unchanged) EXACT gradient, substituting
    /// its curvature with the Fisher metric only conditions the Newton step and
    /// the Laplace normalizer's curvature operator — it does NOT move the
    /// optimum (a fixed-prior curvature majorization, exactly like the ARD
    /// `prior.hess.max(0.0)` precedent). The criterion's `log|H|`, its θ-adjoint
    /// [`Self::row_fisher_metric_logit_derivative`], and the assembled Hessian
    /// all differentiate this SAME operator `G`, keeping value and adjoint on
    /// one exact branch.
    #[must_use]
    pub fn row_fisher_metric(&self, row_logits: &[f64], scale: f64) -> Array2<f64> {
        let k = self.k_atoms;
        let a = self.softmax_row(row_logits);
        let mut g = Array2::<f64>::zeros((k, k));
        for kk in 0..k {
            for jj in 0..k {
                let indicator = if kk == jj { 1.0 } else { 0.0 };
                g[[kk, jj]] = scale * a[kk] * (indicator - a[jj]);
            }
        }
        g
    }

    /// Derivative of the per-row softmax Fisher metric
    /// [`Self::row_fisher_metric`] with respect to a single row logit `z_w`,
    /// scaled by `scale = λ/τ²` (#1190). Returns the symmetric `K×K` block
    /// `∂G_kj/∂z_w`, the third-derivative tensor slice the θ-adjoint contracts
    /// against the row's selected inverse so the adjoint differentiates the SAME
    /// PSD `G = scale·(diag(a) − a aᵀ)` the assembly added (value/adjoint on one
    /// branch, no deflation needed). Built from the SAME softmax derivative
    /// convention as [`Self::row_dense_hessian_logit_derivative`]
    /// (`∂a_r/∂z_w = a_r(δ_rw − a_w)/τ`). For `G_kj = scale·a_k(δ_kj − a_j)`,
    /// the product rule gives
    /// `∂G_kj/∂z_w = scale·[ (∂a_k/∂z_w)(δ_kj − a_j) − a_k(∂a_j/∂z_w) ]`.
    #[must_use]
    pub fn row_fisher_metric_logit_derivative(
        &self,
        row_logits: &[f64],
        scale: f64,
        w: usize,
    ) -> Array2<f64> {
        let k = self.k_atoms;
        let inv_tau = 1.0 / self.temperature;
        let a = self.softmax_row(row_logits);
        // ∂a_r/∂z_w = a_r (δ_rw − a_w)/τ — identical convention to the entropy
        // Hessian derivative above.
        let da: Vec<f64> = (0..k)
            .map(|r| a[r] * (if r == w { 1.0 } else { 0.0 } - a[w]) * inv_tau)
            .collect();
        let mut dg = Array2::<f64>::zeros((k, k));
        for kk in 0..k {
            for jj in 0..k {
                let indicator = if kk == jj { 1.0 } else { 0.0 };
                dg[[kk, jj]] = scale * (da[kk] * (indicator - a[jj]) - a[kk] * da[jj]);
            }
        }
        dg
    }
}

impl AnalyticPenalty for SoftmaxAssignmentSparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut acc = 0.0;
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            for v in a {
                if v > 0.0 {
                    acc += -v * v.ln();
                }
            }
        }
        lambda * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau = 1.0 / self.temperature;
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            let mut d_h_da = vec![0.0; self.k_atoms];
            let mut mean = 0.0;
            for k in 0..self.k_atoms {
                let ak = a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR);
                d_h_da[k] = -lambda * (ak.ln() + 1.0);
                mean += a[k] * d_h_da[k];
            }
            for k in 0..self.k_atoms {
                out[start + k] = a[k] * (d_h_da[k] - mean) * inv_tau;
            }
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert_eq!(rho.len(), 1, "softmax entropy expects one rho parameter");
        assert!(
            rho.iter().all(|value| value.is_finite()),
            "softmax entropy rho must be finite"
        );
        assert_eq!(
            target.len() % self.k_atoms,
            0,
            "softmax entropy target length must be divisible by k_atoms"
        );
        // Closed-form diagonal of the softmax-entropy Hessian wrt logits.
        // Derived by probing the row-dense HVP with the unit vector e_k:
        // for a row with softmax weights a_k and L_k = ln a_k + 1,
        //   H_kk = (lambda / tau^2) * a_k *
        //          ((1 - 2 a_k) * (E_a[L] - L_k) + a_k - 1).
        // This matches `hvp(...) . e_k` analytically (see derivation in the
        // bug-fix comment on `hvp`) and gives Newton/Arrow-Schur callers a
        // principled diagonal surrogate without per-row dense factorization.
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let inv_tau = 1.0 / self.temperature;
        let scale = lambda * inv_tau * inv_tau;
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            let mut mean_log_plus_one = 0.0;
            for k in 0..self.k_atoms {
                mean_log_plus_one += a[k] * (a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0);
            }
            for k in 0..self.k_atoms {
                let log_plus_one = a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0;
                let term = (1.0 - 2.0 * a[k]) * (mean_log_plus_one - log_plus_one) + a[k] - 1.0;
                out[start + k] = scale * a[k] * term;
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
        /*
        Softmax entropy is not coordinate-separable in logits. The old
        `hessian_diag` returned λ p_k(1-p_k)/τ², which is only the softmax
        Jacobian diagonal and omits the entropy curvature and all cross-logit
        terms. For H(p(z)), p'=p*(v-E_p[v])/τ and
        (log p_k + 1)'=(v_k-E_p[v])/τ. Differentiating
        g_k=λ p_k(E_p[log p + 1]-(log p_k+1))/τ gives the row-dense product
        below. `hessian_diag` returns the analytic diagonal extracted from
        this HVP by setting v = e_k row-by-row.
        */
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        let inv_tau = 1.0 / self.temperature;
        let scale = lambda * inv_tau * inv_tau;
        for row in 0..n {
            let start = row * self.k_atoms;
            let a = self.softmax_row(&values[start..start + self.k_atoms]);
            let mut mean_log_plus_one = 0.0;
            let mut mean_v = 0.0;
            for k in 0..self.k_atoms {
                mean_log_plus_one += a[k] * (a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0);
                mean_v += a[k] * v[start + k];
            }
            let mut mean_centered_v_log_plus_one = 0.0;
            for k in 0..self.k_atoms {
                let centered_v = v[start + k] - mean_v;
                mean_centered_v_log_plus_one +=
                    a[k] * centered_v * (a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0);
            }
            for k in 0..self.k_atoms {
                let log_plus_one = a[k].max(ENTROPY_LOG_PROBABILITY_FLOOR).ln() + 1.0;
                let centered_v = v[start + k] - mean_v;
                out[start + k] = scale
                    * a[k]
                    * (centered_v * (mean_log_plus_one - log_plus_one - 1.0)
                        + mean_centered_v_log_plus_one);
            }
        }
        out
    }

    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert_eq!(rho.len(), 1, "softmax entropy expects one rho parameter");
        assert_eq!(
            target.len() % self.k_atoms,
            0,
            "softmax entropy target length must be divisible by k_atoms"
        );
        // Entropy minimization is nonconvex: the exact per-row Hessian is dense
        // and indefinite, so the convex-only trait default (which returns the
        // raw indefinite `hessian_diag`) violates the `B ⪰ 0` contract and is a
        // diagonal masquerading as a dense operator. Replace it with the
        // Gershgorin / diagonal-dominance majorizer of the dense per-row block
        // (see `psd_majorizer_abs_row_sums`): a genuine PSD diagonal with
        // `D ⪰ H` and `D ⪰ 0`. Coordinate-indexed, so the inherited
        // `psd_majorizer_hvp` applies `D` as a diagonal operator consistently.
        let lambda = resolve_learnable_weight(self.weight, rho[0]);
        let inv_tau = 1.0 / self.temperature;
        let scale = lambda * inv_tau * inv_tau;
        let n = target.len() / self.k_atoms;
        let values: Vec<f64> = target.iter().copied().collect();
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n {
            let start = row * self.k_atoms;
            let d = self.psd_majorizer_abs_row_sums(&values[start..start + self.k_atoms], scale);
            for k in 0..self.k_atoms {
                out[start + k] = d[k];
            }
        }
        Some(out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        Array1::from_vec(vec![self.value(target, rho)])
    }

    fn rho_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "softmax_assignment_sparsity"
    }

    impl_scalar_apply_schedule!(weight);
}

impl SparsityPenalty {
    #[must_use = "build error must be handled"]
    pub fn smoothed_l1(target_tier: PenaltyTier, eps: f64) -> Result<Self, String> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!(
                "SparsityPenalty::smoothed_l1 requires eps > 0 \
                 (Hessian / gradient have a `1/sqrt(x² + eps²)` factor that needs eps > 0 \
                 for differentiability at x = 0); got eps = {eps}"
            ));
        }
        Ok(Self {
            target_tier,
            kind: SparsityKind::SmoothedL1 { eps },
            weight: 1.0,
            weight_schedule: None,
            strength_rho_index: 0,
            eps_rho_index: None,
        })
    }

    #[must_use = "build error must be handled"]
    pub fn log(target_tier: PenaltyTier, delta: f64) -> Result<Self, String> {
        if !(delta.is_finite() && delta > 0.0) {
            return Err(format!(
                "SparsityPenalty::log requires delta > 0 \
                 (the log-sparsifier is log(1 + x²/δ²), undefined at δ = 0); \
                 got delta = {delta}"
            ));
        }
        Ok(Self {
            target_tier,
            kind: SparsityKind::Log { delta },
            weight: 1.0,
            weight_schedule: None,
            strength_rho_index: 0,
            eps_rho_index: None,
        })
    }

    /// Hoyer scale-invariant sparsifier. Requires a target of length > 1
    /// because the normalized form divides by `sqrt(n) - 1`.
    #[must_use]
    pub fn hoyer(target_tier: PenaltyTier) -> Self {
        Self {
            target_tier,
            kind: SparsityKind::Hoyer,
            weight: 1.0,
            weight_schedule: None,
            strength_rho_index: 0,
            eps_rho_index: None,
        }
    }

    impl_with_weight_schedule!(weight);

    #[must_use]
    pub fn with_eps_reml(mut self, eps_rho_index: usize) -> Self {
        self.eps_rho_index = Some(eps_rho_index);
        self
    }

    /// Resolve `(strength, eps_or_delta)` from the current ρ view.
    fn resolved(&self, rho: ArrayView1<'_, f64>) -> (f64, f64) {
        let strength = resolve_learnable_weight(self.weight, rho[self.strength_rho_index]);
        let smoothing = match (self.eps_rho_index, self.kind) {
            // A learnable smoothing `exp(rho)` underflows to exact `0.0` for
            // `rho ≲ -745`, which reintroduces a non-differentiable kink and a
            // `0/0` at `x = 0` in `sqrt(x² + ε²)` / the Log sparsifier. Floor it
            // at the smallest positive normal so the smoothing stays strictly
            // positive while still shrinking arbitrarily close to zero.
            (Some(idx), _) => rho[idx].exp().max(f64::MIN_POSITIVE),
            (None, SparsityKind::SmoothedL1 { eps }) => eps,
            (None, SparsityKind::Log { delta }) => delta,
            (None, SparsityKind::Hoyer) => 0.0,
        };
        (strength, smoothing)
    }
}

impl AnalyticPenalty for SparsityPenalty {
    fn tier(&self) -> PenaltyTier {
        self.target_tier
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let (lam, smooth) = self.resolved(rho);
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let mut acc = 0.0;
                for &x in target.iter() {
                    acc += (x * x + smooth * smooth).sqrt();
                }
                lam * acc
            }
            SparsityKind::Hoyer => {
                // Normalized anti-sparsity penalty
                //   P(x) = (||x||_1 / ||x||_2 - 1) / (sqrt(n) - 1)
                // maps [1, sqrt(n)] -> [0, 1]. A perfectly dense
                // equal-magnitude vector hits ||x||_1/||x||_2 = sqrt(n),
                // so P = 1; a 1-sparse vector has ratio 1, so P = 0
                // (sparse vectors minimize the penalty).
                let n = target.len() as f64;
                assert!(n > 1.0, "Hoyer requires n > 1");
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return 0.0;
                }
                let h = (l1 / l2 - 1.0) / (n.sqrt() - 1.0);
                lam * h
            }
            SparsityKind::Log { .. } => {
                let mut acc = 0.0;
                let d2 = smooth * smooth;
                for &x in target.iter() {
                    acc += (1.0 + x * x / d2).ln();
                }
                lam * acc
            }
        }
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let (lam, smooth) = self.resolved(rho);
        let mut g = Array1::<f64>::zeros(target.len());
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    g[i] = lam * x / (x * x + eps2).sqrt();
                }
            }
            SparsityKind::Hoyer => {
                // P(x) = A · (L1/L2 - 1), A = lam / (sqrt(n) - 1).
                // ∂P/∂x_i = A · (sign(x_i)/L2 - L1 · x_i / L2³).
                let n = target.len() as f64;
                assert!(n > 1.0, "Hoyer requires n > 1");
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 {
                    return g;
                }
                let denom = n.sqrt() - 1.0;
                let a = lam / denom;
                let inv_l2 = 1.0 / l2;
                let inv_l2_cubed = inv_l2 * inv_l2 * inv_l2;
                for (i, &x) in target.iter().enumerate() {
                    let sgn = if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    g[i] = a * (sgn * inv_l2 - l1 * x * inv_l2_cubed);
                }
            }
            SparsityKind::Log { .. } => {
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    g[i] = lam * 2.0 * x / (d2 + x * x);
                }
            }
        }
        g
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let (lam, smooth) = self.resolved(rho);
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let mut d = Array1::<f64>::zeros(target.len());
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let r = (x * x + eps2).sqrt();
                    d[i] = lam * eps2 / (r * r * r);
                }
                Some(d)
            }
            SparsityKind::Log { .. } => {
                let mut d = Array1::<f64>::zeros(target.len());
                // The EXACT second derivative of λ log(1 + x²/δ²):
                //   d/dx [ 2λx/(δ²+x²) ] = 2λ(δ² − x²)/(δ² + x²)²,
                // which is NEGATIVE for |x| > δ — Log is nonconvex. This is
                // the genuine Hessian diagonal and exactly differentiates
                // `grad_target`. PSD consumers (Newton block, preconditioner,
                // `log_det_plus_λI`, FrozenAnalyticPenaltyOp) must instead
                // route through `psd_majorizer_diag`/`psd_majorizer_hvp`,
                // which expose the IRLS/MM surrogate `2λ/(δ²+x²)`.
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    d[i] = lam * 2.0 * (d2 - x * x) / (denom * denom);
                }
                Some(d)
            }
            // Hoyer's Hessian is DENSE and NOT generally PSD (Hoyer is a
            // nonconvex sparsifier). We cannot return a meaningful diagonal
            // that would be safe to use as a preconditioner / Newton block
            // through the standard `hessian_diag` path, so we return `None`
            // and force callers through `hvp`. See `hvp` below for the exact
            // dense-Hessian-vector product.
            SparsityKind::Hoyer => None,
        }
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // For SmoothedL1/Log/Hoyer we route through the closed-form Hessian.
        // SmoothedL1 and Log have purely diagonal Hessians and would
        // ordinarily reach the diagonal branch of the default `hvp`; we
        // override here to also serve Hoyer (whose Hessian is dense
        // rank-1-plus-diagonal).
        let (lam, smooth) = self.resolved(rho);
        let n_target = target.len();
        assert_eq!(v.len(), n_target, "hvp dimension mismatch");
        match self.kind {
            SparsityKind::SmoothedL1 { .. } => {
                let mut out = Array1::<f64>::zeros(n_target);
                let eps2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let r = (x * x + eps2).sqrt();
                    out[i] = lam * eps2 / (r * r * r) * v[i];
                }
                out
            }
            SparsityKind::Log { .. } => {
                // EXACT Hessian-vector product: the Log Hessian is diagonal
                // with entries 2λ(δ²−x²)/(δ²+x²)², so (Hv)_i = h_i v_i. This
                // is the genuine second derivative (indefinite for |x|>δ).
                // PSD consumers use `psd_majorizer_hvp` for the IRLS/MM
                // surrogate 2λ/(δ²+x²) instead.
                let mut out = Array1::<f64>::zeros(n_target);
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    let denom = d2 + x * x;
                    out[i] = lam * 2.0 * (d2 - x * x) / (denom * denom) * v[i];
                }
                out
            }
            SparsityKind::Hoyer => {
                // P(x) = A · (L1/L2 - 1), A = lam / (sqrt(n) - 1).
                // H_ij = A · [ -s_i x_j/L2³ - x_i s_j/L2³
                //              - L1 δ_ij/L2³ + 3 L1 x_i x_j/L2⁵ ]
                // (Hv)_i = A · [ -s_i (xᵀv)/L2³ - x_i (sᵀv)/L2³
                //                - L1 v_i/L2³ + 3 L1 x_i (xᵀv)/L2⁵ ]
                let n = n_target as f64;
                assert!(n > 1.0, "Hoyer requires n > 1");
                let l1: f64 = target.iter().map(|x| x.abs()).sum();
                let l2: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
                let mut out = Array1::<f64>::zeros(n_target);
                if l2 == 0.0 {
                    return out;
                }
                let a = lam / (n.sqrt() - 1.0);
                let inv_l2_cubed = 1.0 / (l2 * l2 * l2);
                let inv_l2_5 = inv_l2_cubed / (l2 * l2);
                let mut x_dot_v = 0.0;
                let mut s_dot_v = 0.0;
                for i in 0..n_target {
                    let xi = target[i];
                    let si = if xi > 0.0 {
                        1.0
                    } else if xi < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    x_dot_v += xi * v[i];
                    s_dot_v += si * v[i];
                }
                for i in 0..n_target {
                    let xi = target[i];
                    let si = if xi > 0.0 {
                        1.0
                    } else if xi < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    out[i] = a
                        * (-si * x_dot_v * inv_l2_cubed
                            - xi * s_dot_v * inv_l2_cubed
                            - l1 * v[i] * inv_l2_cubed
                            + 3.0 * l1 * xi * x_dot_v * inv_l2_5);
                }
                out
            }
        }
    }

    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let (lam, smooth) = self.resolved(rho);
        match self.kind {
            // SmoothedL1 is convex: the majorizer equals the exact Hessian.
            SparsityKind::SmoothedL1 { .. } => self.hessian_diag(target, rho),
            // Log is nonconvex; expose the IRLS/MM re-weighted-ℓ₂ surrogate
            //   2λ/(δ²+x²) ⪰ 2λ(δ²−x²)/(δ²+x²)²,
            // strictly positive, agreeing with the exact Hessian at x = 0.
            SparsityKind::Log { .. } => {
                let mut d = Array1::<f64>::zeros(target.len());
                let d2 = smooth * smooth;
                for (i, &x) in target.iter().enumerate() {
                    d[i] = lam * 2.0 / (d2 + x * x);
                }
                Some(d)
            }
            // Hoyer's Hessian is dense; no diagonal majorizer. Callers fall
            // back to the exact dense `hvp` through `psd_majorizer_hvp`.
            SparsityKind::Hoyer => None,
        }
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Strength axis: ∂P/∂ρ_strength = P (chain rule through exp).
        // ε axis (if owned): ∂P/∂ρ_eps = ε · ∂P/∂ε.
        let n_rho = self.rho_count();
        let mut out = Array1::<f64>::zeros(n_rho);
        let p_val = self.value(target, rho);
        out[self.strength_rho_index] = p_val;
        if let Some(eps_idx) = self.eps_rho_index {
            let (lam, smooth) = self.resolved(rho);
            let mut dp_deps = 0.0;
            match self.kind {
                SparsityKind::SmoothedL1 { .. } => {
                    for &x in target.iter() {
                        dp_deps += smooth / (x * x + smooth * smooth).sqrt();
                    }
                    dp_deps *= lam;
                }
                SparsityKind::Log { .. } => {
                    // d/dδ log(1 + x²/δ²) = -2 x² / (δ (δ² + x²))
                    let d2 = smooth * smooth;
                    for &x in target.iter() {
                        dp_deps += -2.0 * x * x / (smooth * (d2 + x * x));
                    }
                    dp_deps *= lam;
                }
                SparsityKind::Hoyer => {}
            }
            // Chain through ρ_eps = log(ε)  ⇒  ∂ε/∂ρ_eps = ε.
            out[eps_idx] = smooth * dp_deps;
        }
        out
    }

    fn rho_count(&self) -> usize {
        1 + if self.eps_rho_index.is_some() { 1 } else { 0 }
    }

    fn name(&self) -> &str {
        "sparsity"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// TopK activation penalty
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TopKActivationPenalty {
    pub target: PsiSlice,
    pub k: usize,
    pub latent_dim: usize,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl TopKActivationPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(target: PsiSlice, k: usize, weight: f64) -> Result<Self, String> {
        let latent_dim = target
            .latent_dim
            .ok_or_else(|| "TopKActivationPenalty::new requires target.latent_dim".to_string())?;
        if latent_dim == 0 {
            return Err("TopKActivationPenalty::new requires latent_dim > 0".to_string());
        }
        if k == 0 || k > latent_dim {
            return Err(format!(
                "TopKActivationPenalty::new requires 0 < k <= latent_dim; got k={k}, latent_dim={latent_dim}"
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "TopKActivationPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        Ok(Self {
            target,
            k,
            latent_dim,
            weight,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn topk_mask_row(&self, target: ArrayView1<'_, f64>, row: usize, mask: &mut [bool]) {
        mask.fill(false);
        let d = self.latent_dim;
        let base = row * d;
        let mut order = (0..d).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            target[base + b]
                .abs()
                .total_cmp(&target[base + a].abs())
                .then_with(|| a.cmp(&b))
        });
        for &axis in order.iter().take(self.k) {
            mask[axis] = true;
        }
    }
}

impl AnalyticPenalty for TopKActivationPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut mask = vec![false; d];
        let mut acc = 0.0;
        for row in 0..n_obs {
            self.topk_mask_row(target, row, &mut mask);
            let base = row * d;
            for axis in 0..d {
                if mask[axis] {
                    let v = target[base + axis];
                    acc += 0.5 * self.weight * v * v;
                }
            }
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut mask = vec![false; d];
        let mut grad = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            self.topk_mask_row(target, row, &mut mask);
            let base = row * d;
            for axis in 0..d {
                if mask[axis] {
                    grad[base + axis] = self.weight * target[base + axis];
                }
            }
        }
        grad
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut mask = vec![false; d];
        let mut diag = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            self.topk_mask_row(target, row, &mut mask);
            let base = row * d;
            for axis in 0..d {
                if mask[axis] {
                    diag[base + axis] = self.weight;
                }
            }
        }
        Some(diag)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(rho.len(), 0, "TopKActivationPenalty has no rho parameters");
        assert_eq!(
            target.len() % self.latent_dim,
            0,
            "TopKActivationPenalty target length must be a multiple of latent_dim"
        );
        Array1::<f64>::zeros(0)
    }

    fn rho_count(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "topk_activation"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// JumpReLU penalty
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct JumpReLUPenalty {
    pub target: PsiSlice,
    pub latent_dim: usize,
    pub thresholds: Array1<f64>,
    pub weight: f64,
    pub smoothing_eps: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl JumpReLUPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        thresholds: Array1<f64>,
        weight: f64,
        smoothing_eps: f64,
    ) -> Result<Self, String> {
        let latent_dim = target
            .latent_dim
            .ok_or_else(|| "JumpReLUPenalty::new requires target.latent_dim".to_string())?;
        if latent_dim == 0 {
            return Err("JumpReLUPenalty::new requires latent_dim > 0".to_string());
        }
        if thresholds.len() != latent_dim {
            return Err(format!(
                "JumpReLUPenalty::new thresholds length {} does not match latent_dim {latent_dim}",
                thresholds.len()
            ));
        }
        for (idx, &tau) in thresholds.iter().enumerate() {
            if !(tau.is_finite() && tau > 0.0) {
                return Err(format!(
                    "JumpReLUPenalty::new thresholds[{idx}] must be finite and > 0, got {tau}"
                ));
            }
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "JumpReLUPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "JumpReLUPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        Ok(Self {
            target,
            latent_dim,
            thresholds,
            weight,
            smoothing_eps,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn threshold(&self, axis: usize, rho: ArrayView1<'_, f64>) -> f64 {
        // A learnable threshold `θ·exp(rho)` overflows to `inf` for large `rho`;
        // the downstream gate `σ((l−θ)/τ)` then evaluates `inf·gate = NaN`. Clamp
        // the log-magnitude so the threshold stays a finite normal.
        resolve_learnable_weight(self.thresholds[axis], rho[axis])
    }

    pub(crate) fn sigmoid_gate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }

    fn true_hessian_diag_entry(&self, tau: f64, gate: f64) -> f64 {
        self.weight * tau * gate * (1.0 - gate) * (1.0 - 2.0 * gate)
            / (self.smoothing_eps * self.smoothing_eps)
    }

    fn psd_hessian_diag_entry(&self, tau: f64, gate: f64) -> f64 {
        // Genuine PSD majorizer of the indefinite exact diagonal Hessian
        //   h(g) = λτ·g(1−g)(1−2g)/ε².
        // The bare re-weighted-ℓ₂ surrogate λτ·[g(1−g)]²/ε² is ≥ 0 but only
        // dominates h in the concave region g > ½. For g < (3−√5)/2 ≈ 0.382 the
        // exact curvature is positive and strictly larger, so the square alone
        // is NOT an upper bound — the `B ⪰ ∂²P` contract is violated for exactly
        // the comfortably-below-threshold (inactive) coordinates JumpReLU is
        // meant to suppress, costing the MM step its monotone-decrease guarantee.
        //
        // Take the elementwise max of that surrogate and the absolute exact
        // Hessian |h| = λτ·g(1−g)|1−2g|/ε². Since |h| ≥ h everywhere and ≥ 0, the
        // max is a true PSD upper bound; it equals |h| in the wings (tight where
        // the bare square failed) and keeps the surrogate's strictly-positive
        // floor near the inflection g ≈ ½ (where h ≈ 0) so the curvature block
        // never collapses to zero.
        let slope = gate * (1.0 - gate);
        let reweighted_l2 = slope * slope;
        let abs_exact = slope * (1.0 - 2.0 * gate).abs();
        self.weight * tau * reweighted_l2.max(abs_exact) / (self.smoothing_eps * self.smoothing_eps)
    }
}

/// JumpReLU activation gate `φ(z) = z · 1[z > τ]` together with the
/// straight-through-estimator derivatives of its smooth surrogate
/// `φ̃(z) = z · σ((z − τ)/ε)`. The forward value is the hard gate; the backward
/// uses the surrogate's gradients so the activation has a usable subgradient in
/// the smoothing band `|z − τ| ≲ ε`:
///
///   g       = σ((z − τ)/ε)
///   φ        = z · 1[z > τ]                 (returned value)
///   ∂φ̃/∂z   = g + z · g (1 − g) / ε          (`dphi_dz`)
///   ∂φ̃/∂τ   = − z · g (1 − g) / ε            (`dphi_dtau`)
///
/// This is the single Rust source of truth that `gamfit.torch`'s
/// `_JumpReLUSTEFn` consumes so the torch activation gate's backward matches the
/// smoothed gate exactly instead of re-deriving it in Python.
#[must_use]
pub fn jumprelu_gate_value_grad(z: f64, tau: f64, smoothing_eps: f64) -> (f64, f64, f64) {
    let g = crate::linalg::utils::stable_logistic((z - tau) / smoothing_eps);
    let value = if z > tau { z } else { 0.0 };
    let slope = z * g * (1.0 - g) / smoothing_eps;
    let dphi_dz = g + slope;
    let dphi_dtau = -slope;
    (value, dphi_dz, dphi_dtau)
}

impl AnalyticPenalty for JumpReLUPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut acc = 0.0;
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                acc += self.weight * tau * gate;
            }
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut grad = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                grad[base + axis] = self.weight * tau * gate * (1.0 - gate) / self.smoothing_eps;
            }
        }
        grad
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut diag = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                diag[base + axis] = self.true_hessian_diag_entry(tau, gate);
            }
        }
        Some(diag)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                out[base + axis] = self.true_hessian_diag_entry(tau, gate) * v[base + axis];
            }
        }
        out
    }

    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        // The smoothed JumpReLU surrogate's exact diagonal Hessian
        //   λτ·g(1−g)(1−2g)/ε²
        // is indefinite (negative once the gate passes the inflection
        // g = ½). The Newton / PIRLS pipeline needs a PSD curvature block, so
        // expose the PSD upper bound implemented by `psd_hessian_diag_entry`:
        // the elementwise max of the re-weighted surrogate and the absolute
        // exact curvature.
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut diag = Array1::<f64>::zeros(target.len());
        for row in 0..n_obs {
            let base = row * d;
            for axis in 0..d {
                let tau = self.threshold(axis, rho);
                let gate = self.sigmoid_gate((target[base + axis] - tau) / self.smoothing_eps);
                diag[base + axis] = self.psd_hessian_diag_entry(tau, gate);
            }
        }
        Some(diag)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(d);
        for axis in 0..d {
            let tau = self.threshold(axis, rho);
            let mut g_tau = 0.0;
            for row in 0..n_obs {
                let x = target[row * d + axis];
                let gate = self.sigmoid_gate((x - tau) / self.smoothing_eps);
                g_tau += gate - tau * gate * (1.0 - gate) / self.smoothing_eps;
            }
            out[axis] = self.weight * tau * g_tau;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.latent_dim
    }

    fn name(&self) -> &str {
        "jumprelu"
    }

    impl_scalar_apply_schedule!(weight);
}
