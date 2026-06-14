use super::*;

// ---------------------------------------------------------------------------
// NestedPrefixPenalty — Matryoshka SAE
// ---------------------------------------------------------------------------

/// Nested-prefix sparsity penalty used by the Matryoshka SAE
/// (Bussmann/Nabeshima/Karvonen/Nanda, ICML 2025, arXiv:2503.17547).
///
/// Given K nested prefix sizes `m_1 < m_2 < ... < m_K ≤ F` over the latent
/// dimension `F`, and per-shell weights `λ_k = w_k · exp(ρ_k)`, the penalty is
///
/// ```text
///   P(t; ρ) = Σ_k λ_k · Σ_{i=0}^{m_k - 1} sqrt(t_i² + ε²)
/// ```
///
/// summed over all rows of the latent target. Equivalently, coordinate `i`
/// contributes with effective weight `W_i = Σ_{k: m_k > i} λ_k`, so the
/// earliest atoms (small `i`) are penalized by every shell (= strongest L¹)
/// and the latest atoms only by the outermost shell. This is exactly the
/// mask-weighted sum-of-L¹ over K prefixes used to enforce shell-wise
/// reconstruction during Matryoshka training.
///
/// Closed forms (per row, summed across all rows):
///
/// ```text
///   ∂P/∂t_i      = W_i · t_i / sqrt(t_i² + ε²)
///   Hess_diag(i) = W_i · ε² / (t_i² + ε²)^{3/2}           (PSD)
///   ∂P/∂ρ_k      = λ_k · Σ_{i < m_k} sqrt(t_i² + ε²)
/// ```
///
/// `target` lays out `n_rows × latent_dim` in row-major order (`row * F + col`).
/// `latent_dim` is taken from `PsiSlice::latent_dim`; if absent we fall back to
/// the maximum prefix size, which is the standard Matryoshka convention.
#[derive(Debug, Clone)]
pub struct NestedPrefixPenalty {
    pub target: PsiSlice,
    pub target_tier: PenaltyTier,
    /// Sorted strictly-increasing prefix sizes `m_1 < m_2 < ... < m_K`.
    pub prefix_sizes: Vec<usize>,
    /// Per-shell base weights `w_k`. The effective strength is
    /// `λ_k = w_k · exp(ρ_k)`.
    pub shell_weights: Vec<f64>,
    /// Smoothing parameter ε > 0 for the smoothed-L¹ surrogate
    /// `sqrt(x² + ε²)`; the Hessian needs ε > 0 for differentiability at 0.
    pub eps: f64,
    /// Local ρ indices for the K per-shell log-strengths.
    pub rho_indices: Vec<usize>,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl NestedPrefixPenalty {
    /// Build a new nested-prefix penalty.
    ///
    /// Errors when:
    ///  * `prefix_sizes` is empty.
    ///  * `prefix_sizes` is not strictly increasing.
    ///  * any prefix exceeds the latent dimension (when known).
    ///  * `shell_weights.len() != prefix_sizes.len()`.
    ///  * `eps <= 0` (the smoothed-L¹ gradient `1/sqrt(x²+ε²)` and Hessian
    ///    `ε²/(x²+ε²)^{3/2}` both need ε > 0).
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        target_tier: PenaltyTier,
        prefix_sizes: Vec<usize>,
        shell_weights: Vec<f64>,
        eps: f64,
    ) -> Result<Self, String> {
        if prefix_sizes.is_empty() {
            return Err("NestedPrefixPenalty requires at least one prefix".into());
        }
        if shell_weights.len() != prefix_sizes.len() {
            return Err(format!(
                "NestedPrefixPenalty requires shell_weights.len() == prefix_sizes.len(); \
                 got {} weights for {} prefixes",
                shell_weights.len(),
                prefix_sizes.len()
            ));
        }
        for w in &shell_weights {
            if !w.is_finite() || *w < 0.0 {
                return Err(format!(
                    "NestedPrefixPenalty shell weights must be finite and ≥ 0; got {w}"
                ));
            }
        }
        for i in 0..prefix_sizes.len() {
            if prefix_sizes[i] == 0 {
                return Err("NestedPrefixPenalty prefixes must be > 0".into());
            }
            if i > 0 && prefix_sizes[i] <= prefix_sizes[i - 1] {
                return Err(format!(
                    "NestedPrefixPenalty prefixes must be strictly increasing; got {:?}",
                    prefix_sizes
                ));
            }
        }
        if let Some(d) = target.latent_dim {
            let max_prefix = *prefix_sizes.last().expect("non-empty");
            if max_prefix > d {
                return Err(format!(
                    "NestedPrefixPenalty largest prefix {max_prefix} exceeds latent_dim {d}"
                ));
            }
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!(
                "NestedPrefixPenalty requires eps > 0 (1/sqrt(x²+ε²) singularity at 0); got {eps}"
            ));
        }
        let rho_indices = (0..prefix_sizes.len()).collect();
        Ok(Self {
            target,
            target_tier,
            prefix_sizes,
            shell_weights,
            eps,
            rho_indices,
            weight_schedule: None,
        })
    }

    /// Attach a global annealing schedule shared by all shell weights. The
    /// REML loop still picks per-shell ρ_k on top of this baseline.
    #[must_use]
    pub fn with_weight_schedule(mut self, schedule: ScalarWeightSchedule) -> Self {
        self.weight_schedule = Some(schedule);
        self
    }

    /// Latent dimension used to slice rows. Falls back to the largest prefix.
    fn latent_dim(&self) -> usize {
        self.target
            .latent_dim
            .unwrap_or_else(|| *self.prefix_sizes.last().expect("non-empty"))
    }

    /// Resolve per-shell effective weights `λ_k = w_k · exp(ρ_k)`.
    fn lambdas(&self, rho: ArrayView1<'_, f64>) -> Vec<f64> {
        self.prefix_sizes
            .iter()
            .enumerate()
            .map(|(k, _)| resolve_learnable_weight(self.shell_weights[k], rho[self.rho_indices[k]]))
            .collect()
    }

    /// Per-axis cumulative weight `W_i = Σ_{k: m_k > i} λ_k`. Length = F.
    /// Computed in `O(F + K)` by scanning prefixes from outer to inner.
    fn per_axis_weights(&self, lambdas: &[f64]) -> Vec<f64> {
        let f = self.latent_dim();
        let mut w = vec![0.0_f64; f];
        // For each shell k, every axis i ∈ [0, m_k) gets +λ_k.
        // Equivalent reverse-cumulative form, but the direct O(K·F) loop is
        // K≤8 in practice, so this is O(F) for the use cases we ship.
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let lam = lambdas[k];
            if lam == 0.0 {
                continue;
            }
            let end = m_k.min(f);
            for entry in w.iter_mut().take(end) {
                *entry += lam;
            }
        }
        w
    }
}

impl AnalyticPenalty for NestedPrefixPenalty {
    fn tier(&self) -> PenaltyTier {
        self.target_tier
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let f = self.latent_dim();
        assert!(
            target.len().is_multiple_of(f),
            "target length must be n_rows · F"
        );
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let eps2 = self.eps * self.eps;
        // Per-axis L¹ totals s_i = Σ_n sqrt(t_{n,i}² + ε²).
        let mut s_axis = vec![0.0_f64; f];
        for n in 0..n_rows {
            let row = &target.as_slice().expect("contiguous")[n * f..(n + 1) * f];
            for (i, &x) in row.iter().enumerate() {
                s_axis[i] += (x * x + eps2).sqrt();
            }
        }
        // Now P = Σ_k λ_k · Σ_{i<m_k} s_i.
        let mut total = 0.0;
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let end = m_k.min(f);
            let mut acc = 0.0;
            for &v in s_axis.iter().take(end) {
                acc += v;
            }
            total += lambdas[k] * acc;
        }
        total
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let f = self.latent_dim();
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let w_per_axis = self.per_axis_weights(&lambdas);
        let eps2 = self.eps * self.eps;
        let src = target.as_slice().expect("contiguous");
        let mut g = Array1::<f64>::zeros(target.len());
        let g_slice = g.as_slice_mut().expect("contiguous");
        for n in 0..n_rows {
            for i in 0..f {
                let x = src[n * f + i];
                let w = w_per_axis[i];
                if w == 0.0 {
                    continue;
                }
                g_slice[n * f + i] = w * x / (x * x + eps2).sqrt();
            }
        }
        g
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let f = self.latent_dim();
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let w_per_axis = self.per_axis_weights(&lambdas);
        let eps2 = self.eps * self.eps;
        let src = target.as_slice().expect("contiguous");
        let mut d = Array1::<f64>::zeros(target.len());
        let d_slice = d.as_slice_mut().expect("contiguous");
        for n in 0..n_rows {
            for i in 0..f {
                let w = w_per_axis[i];
                if w == 0.0 {
                    continue;
                }
                let x = src[n * f + i];
                let r = (x * x + eps2).sqrt();
                d_slice[n * f + i] = w * eps2 / (r * r * r);
            }
        }
        Some(d)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let f = self.latent_dim();
        let n_rows = target.len() / f;
        let lambdas = self.lambdas(rho);
        let eps2 = self.eps * self.eps;
        // Same axis-wise reduction as `value`, but we need the per-shell
        // (not cumulative) sums for the ρ-gradient.
        let mut s_axis = vec![0.0_f64; f];
        let src = target.as_slice().expect("contiguous");
        for n in 0..n_rows {
            for i in 0..f {
                let x = src[n * f + i];
                s_axis[i] += (x * x + eps2).sqrt();
            }
        }
        let n_rho = self.rho_count();
        let mut out = Array1::<f64>::zeros(n_rho);
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let end = m_k.min(f);
            let mut shell_sum = 0.0;
            for &v in s_axis.iter().take(end) {
                shell_sum += v;
            }
            // ∂P/∂ρ_k = λ_k · shell_sum  because λ_k = w_k · exp(ρ_k).
            out[self.rho_indices[k]] = lambdas[k] * shell_sum;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.prefix_sizes.len()
    }

    fn name(&self) -> &str {
        "nested_prefix"
    }

    fn apply_schedule(&mut self, iter: usize) {
        if let Some(schedule) = self.weight_schedule.as_mut() {
            let prev = schedule.current_weight(schedule.iter_count);
            let next = schedule.current_weight(iter);
            if prev > 0.0 {
                let ratio = next / prev;
                for w in &mut self.shell_weights {
                    *w *= ratio;
                }
            }
            schedule.iter_count = iter + 1;
        }
    }
}
