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
///   ∂P/∂ρ_k      = λ_k · Σ_{i < m_k} sqrt(t_i² + ε²)      (energy only)
/// ```
///
/// `target` lays out `n_rows × latent_dim` in row-major order (`row * F + col`).
/// `latent_dim` is taken from `PsiSlice::latent_dim`; if absent we fall back to
/// the maximum prefix size, which is the standard Matryoshka convention.
///
/// # Shell strengths are fixed unless the caller asks (#4291)
///
/// The energy above is a negative log prior only up to the smoothed-Laplace mass
/// `Z(W_i, ε) = 2ε·K₁(W_i·ε)` of each covered axis. `∂P/∂ρ_k` is a sum of
/// `sqrt(t_i² + ε²) > 0` terms, so WITHOUT that mass its sign is `+` at every
/// target: an outer search minimizing the criterion in `ρ_k` can only walk to
/// the lower face, where `λ_k` is many orders of magnitude below the `w_k` the
/// caller wrote and the shell is effectively switched off, with no warning. The
/// shells therefore own no ρ-axis by default — `λ_k = w_k`, exactly what the
/// caller asked for — and [`NestedPrefixPenalty::with_learnable_shells`] turns
/// them into outer coordinates together with the exact normalizer
/// `n_rows · Σ_i ln Z(W_i, ε)` that gives them an interior optimum.
#[derive(Debug, Clone)]
pub struct NestedPrefixPenalty {
    pub target: PsiSlice,
    pub target_tier: PenaltyTier,
    /// Sorted strictly-increasing prefix sizes `m_1 < m_2 < ... < m_K`.
    pub prefix_sizes: Vec<usize>,
    /// Per-shell base weights `w_k > 0`. The effective strength is `λ_k = w_k`,
    /// or `λ_k = w_k · exp(ρ_k)` when [`Self::learnable_shells`] is set.
    pub shell_weights: Vec<f64>,
    /// Smoothing parameter ε > 0 for the smoothed-L¹ surrogate
    /// `sqrt(x² + ε²)`; the Hessian needs ε > 0 for differentiability at 0.
    pub eps: f64,
    /// Local ρ indices for the K per-shell log-strengths. Empty unless
    /// [`Self::learnable_shells`] is set.
    pub rho_indices: Vec<usize>,
    /// Whether the K shell strengths are outer coordinates (#4291). When set,
    /// [`AnalyticPenalty::value`] and [`AnalyticPenalty::grad_rho`] carry the
    /// smoothed-Laplace normalizer; when clear the penalty owns no ρ at all.
    pub learnable_shells: bool,
}

impl NestedPrefixPenalty {
    /// Build a new nested-prefix penalty.
    ///
    /// Errors when:
    ///  * `prefix_sizes` is empty.
    ///  * `prefix_sizes` is not strictly increasing.
    ///  * any prefix exceeds the latent dimension (when known).
    ///  * `shell_weights.len() != prefix_sizes.len()`.
    ///  * any shell weight is not strictly positive. A zero base weight used to
    ///    pass here and fail later, at fit setup, inside
    ///    `learnable_weight_coordinate_domain` — the refusal named the ρ
    ///    coordinate instead of the weight the caller wrote. A shell whose base
    ///    weight is zero is a shell no log-strength can ever switch on, and a
    ///    shell the caller wants off is a prefix the caller omits (#4291).
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
            if !w.is_finite() || *w <= 0.0 {
                return Err(format!(
                    "NestedPrefixPenalty shell weights must be finite and > 0; got {w}. \
                     A zero-weight shell is a shell no log-strength can switch on, and a \
                     shell that should not act is a prefix to omit (#4291)"
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
        Ok(Self {
            target,
            target_tier,
            prefix_sizes,
            shell_weights,
            eps,
            // No ρ until the caller asks for one (#4291); the layout is grown by
            // `with_learnable_shells` so an unasked-for coordinate is
            // unrepresentable rather than merely unused.
            rho_indices: Vec::new(),
            learnable_shells: false,
        })
    }

    /// Make the K shell strengths outer coordinates `λ_k = w_k · exp(ρ_k)`,
    /// with the smoothed-Laplace normalizer priced alongside them (#4291).
    ///
    /// This is admissible here, and refused on the sparsifiers whose energy is
    /// bounded, precisely because the smoothed-L¹ prior has a finite mass in
    /// closed form ([`smoothed_laplace_log_partition`]). Each covered axis `i`
    /// carries effective weight `W_i = Σ_{k: m_k > i} λ_k` and mass
    /// `Z(W_i, ε) = 2ε·K₁(W_i·ε)`, so the criterion the outer search sees is
    /// `P(t; ρ) + n_rows · Σ_{i: W_i > 0} ln Z(W_i, ε)`, whose `ρ_k`-derivative
    /// changes sign: it is `−∞` as `λ_k → 0` and non-negative as `λ_k → ∞`.
    #[must_use = "invalid learnable-shell requests must be handled"]
    pub fn with_learnable_shells(mut self) -> Result<Self, String> {
        let (lower, upper) = smoothed_laplace_strength_log_band(self.eps)
            .map_err(|error| format!("NestedPrefixPenalty: {error}"))?;
        for weight in &self.shell_weights {
            let log_weight = weight.ln();
            if !(lower..=upper).contains(&log_weight) {
                return Err(format!(
                    "NestedPrefixPenalty shell weight {weight} has ln w = {log_weight} outside \
                     the band [{lower}, {upper}] on which its smoothed-Laplace normalizer is \
                     computable at eps = {}; a learnable shell whose ρ = 0 is already outside \
                     its own domain is not a coordinate (#4291)",
                    self.eps
                ));
            }
        }
        self.rho_indices = (0..self.prefix_sizes.len()).collect();
        self.learnable_shells = true;
        Ok(self)
    }

    /// Whether the shell strengths are outer coordinates.
    #[must_use]
    pub fn learns_shells(&self) -> bool {
        self.learnable_shells
    }

    /// Latent dimension used to slice rows. Falls back to the largest prefix.
    fn latent_dim(&self) -> usize {
        self.target
            .latent_dim
            .unwrap_or_else(|| *self.prefix_sizes.last().expect("non-empty"))
    }

    /// Resolve per-shell effective weights: `λ_k = w_k` when the shells are
    /// fixed, `λ_k = w_k · exp(ρ_k)` when they are learnable (#4291).
    fn lambdas(&self, rho: ArrayView1<'_, f64>) -> Vec<f64> {
        if !self.learnable_shells {
            return self.shell_weights.clone();
        }
        self.prefix_sizes
            .iter()
            .enumerate()
            .map(|(k, _)| {
                validated_learnable_weight(self.shell_weights[k], rho[self.rho_indices[k]])
            })
            .collect()
    }

    /// `(Σ_{i: W_i > 0} ln Z(W_i, ε), [∂ ln Z(W_i, ε)/∂W_i]_i)` for one row.
    ///
    /// An axis with `W_i = 0` is an axis beyond the outermost prefix: the
    /// Matryoshka convention leaves it unpenalized, its improper flat measure
    /// does not depend on any `λ_k`, and it contributes to neither the mass nor
    /// any ρ-derivative. It is skipped rather than priced.
    fn per_axis_log_partition(&self, w_per_axis: &[f64]) -> Result<(f64, Vec<f64>), String> {
        let mut total = 0.0;
        let mut weight_derivative = vec![0.0_f64; w_per_axis.len()];
        for (i, &w) in w_per_axis.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let partition = smoothed_laplace_log_partition(w, self.eps).map_err(|error| {
                format!("nested-prefix axis {i} effective weight {w}: {error}")
            })?;
            total += partition.value;
            weight_derivative[i] = partition.log_strength_derivative / w;
        }
        Ok((total, weight_derivative))
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

    fn validate_rho(&self, rho: ArrayView1<'_, f64>) -> Result<(), String> {
        if rho.len() != self.rho_count() {
            return Err(format!(
                "nested-prefix rho length {} != declared {}",
                rho.len(),
                self.rho_count()
            ));
        }
        if !self.learnable_shells {
            return Ok(());
        }
        for shell in 0..self.prefix_sizes.len() {
            resolve_learnable_weight(self.shell_weights[shell], rho[self.rho_indices[shell]])?;
        }
        // The normalizer is priced on the per-axis CUMULATIVE weights, not on
        // one shell at a time, so the exact domain test is here and not in
        // `rho_coordinate_domains` (which can only bound one coordinate at a
        // time). Refusing here is what keeps `value` / `grad_rho` total.
        let lambdas = self.lambdas(rho);
        let w_per_axis = self.per_axis_weights(&lambdas);
        self.per_axis_log_partition(&w_per_axis)?;
        Ok(())
    }

    fn rho_coordinate_domains(&self) -> Result<Vec<(f64, f64)>, String> {
        if !self.learnable_shells {
            return Ok(Vec::new());
        }
        let (lower, upper) = smoothed_laplace_strength_log_band(self.eps)?;
        self.shell_weights
            .iter()
            .map(|&weight| {
                let log_weight = weight.ln();
                Ok((lower - log_weight, upper - log_weight))
            })
            .collect()
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
        if self.learnable_shells {
            // `−log p` carries the prior's own mass. It is a function of the
            // λ's alone, so it is priced exactly where a λ-derivative is taken
            // and omitted where the λ's are fixed — there it is an additive
            // constant that would move every pinned criterion value without
            // moving any fit (#4291).
            let w_per_axis = self.per_axis_weights(&lambdas);
            let (log_partition, _) = self
                .per_axis_log_partition(&w_per_axis)
                .expect("nested-prefix rho must be validated before value evaluation");
            total += (n_rows as f64) * log_partition;
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
        if n_rho == 0 {
            return out;
        }
        // `∂W_i/∂ρ_k = λ_k` for every axis `i < m_k`, so the normalizer's
        // contribution to shell `k` is `n_rows·λ_k·Σ_{i<m_k} ∂lnZ(W_i, ε)/∂W_i`
        // — the same shell-prefix reduction the energy uses, on a different
        // per-axis summand.
        let w_per_axis = self.per_axis_weights(&lambdas);
        let (_, partition_weight_derivative) = self
            .per_axis_log_partition(&w_per_axis)
            .expect("nested-prefix rho must be validated before grad_rho evaluation");
        let rows = n_rows as f64;
        for (k, &m_k) in self.prefix_sizes.iter().enumerate() {
            let end = m_k.min(f);
            let mut shell_sum = 0.0;
            for i in 0..end {
                shell_sum += s_axis[i] + rows * partition_weight_derivative[i];
            }
            // ∂/∂ρ_k = λ_k · shell_sum  because λ_k = w_k · exp(ρ_k).
            out[self.rho_indices[k]] = lambdas[k] * shell_sum;
        }
        out
    }

    fn rho_count(&self) -> usize {
        if self.learnable_shells {
            self.prefix_sizes.len()
        } else {
            0
        }
    }

    fn name(&self) -> &str {
        "nested_prefix"
    }

}
