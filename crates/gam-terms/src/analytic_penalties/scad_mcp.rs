use super::*;

// ---------------------------------------------------------------------------
// SCAD / MCP concave sparsity penalty
// ---------------------------------------------------------------------------

/// Concave alternative to smoothed-L¹ sparsity with less bias on large signals.
/// MCP (Zhang 2010) and SCAD (Fan-Li 2001) keep strong shrinkage near zero but
/// taper the derivative to zero for large coefficients; `gamma` controls
/// concavity, and `gamma -> infinity` recovers the L¹ limit. Fan-Li recommend
/// `gamma = 3.7` for SCAD.
///
/// `SparsityPenalty` uses Huber-smoothed L¹, `Σ_j sqrt(t_j² + ε²)`, whose
/// gradient magnitude stays constant outside the Huber region. That constant
/// pull over-shrinks moderate true signals. SCAD/MCP flatten the gradient for
/// large coefficients, which gives less-biased estimates while still shrinking
/// near-zero noise; paired with row-precision priors, the precision field
/// anchors which axes are active and this penalty fits their magnitudes without
/// L¹'s constant pull.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyConcavity {
    Mcp,
    Scad,
}

/// Element-wise SCAD/MCP family penalty on a row-major latent block.
///
/// Lives on the extension-coordinate tier. The target is a row-major
/// `(n_eff, d)` latent block, but the penalty is coordinate-separable:
/// `P(T) = Σ_i p(r_i; λ, γ)` with `r_i = sqrt(T_i² + ε²)` and
/// `λ = weight · exp(ρ)` when the weight is learnable.
///
/// MCP uses
///
/// ```text
///   p_MCP(r) = λr - (r² - ε²)/(2γ),       r ≤ γλ
///            = γλ²/2 + ε²/(2γ),          r > γλ,
/// ```
///
/// so the shrinkage derivative tapers linearly to zero at `γλ`. SCAD uses the
/// Fan-Li three-region derivative: L¹ shrinkage up to `λ`, a linear taper on
/// `(λ, γλ]`, and zero derivative beyond `γλ`.
///
/// In the SAE objective this is the nonconvex sparsity prior for latent
/// extension-coordinate amplitudes: it still suppresses near-zero activations
/// but avoids L¹'s bias on large coefficients that should remain active.
///
/// Gotchas:
///
/// * The exact Hessian diagonal can be negative in the taper region. This is a
///   row-block-diagonal penalty, but not a PSD curvature source.
/// * `γ` must be `> 1` for MCP and `> 2` for SCAD. Larger values approach
///   smoothed L¹; smaller valid values make the taper more aggressive.
/// * `ε > 0` smooths the absolute-value cusp and slightly shifts the effective
///   cutoffs because the piecewise rules use `r = sqrt(t² + ε²)`.
#[derive(Debug, Clone)]
pub struct ScadMcpPenalty {
    pub target: PsiSlice,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    /// Concavity parameter. Larger values approach smoothed L¹.
    pub gamma: f64,
    pub smoothing_eps: f64,
    pub variant: PenaltyConcavity,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl ScadMcpPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        weight: f64,
        n_eff: usize,
        gamma: f64,
        smoothing_eps: f64,
        variant: PenaltyConcavity,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("ScadMcpPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ScadMcpPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("ScadMcpPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "ScadMcpPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff
                .checked_mul(expected_dim)
                .ok_or_else(|| "ScadMcpPenalty::new target shape overflows usize".to_string())?;
            if expected != target.len() {
                return Err(format!(
                    "ScadMcpPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        match variant {
            PenaltyConcavity::Mcp if !(gamma.is_finite() && gamma > 1.0) => {
                return Err(format!(
                    "ScadMcpPenalty::new MCP requires finite gamma > 1, got {gamma}"
                ));
            }
            PenaltyConcavity::Scad if !(gamma.is_finite() && gamma > 2.0) => {
                return Err(format!(
                    "ScadMcpPenalty::new SCAD requires finite gamma > 2, got {gamma}"
                ));
            }
            PenaltyConcavity::Mcp | PenaltyConcavity::Scad => {}
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "ScadMcpPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        Ok(Self {
            target,
            weight,
            n_eff,
            gamma,
            smoothing_eps,
            variant,
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

    fn smooth_abs(&self, t: f64) -> f64 {
        (t * t + self.smoothing_eps * self.smoothing_eps).sqrt()
    }

    fn value_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        match self.variant {
            PenaltyConcavity::Mcp => {
                let cutoff = self.gamma * weight;
                if r <= cutoff {
                    weight * r
                        - (r * r - self.smoothing_eps * self.smoothing_eps) / (2.0 * self.gamma)
                } else {
                    0.5 * self.gamma * weight * weight
                        + self.smoothing_eps * self.smoothing_eps / (2.0 * self.gamma)
                }
            }
            PenaltyConcavity::Scad => {
                let cutoff1 = weight;
                let cutoff2 = self.gamma * weight;
                if r <= cutoff1 {
                    weight * r
                } else if r <= cutoff2 {
                    (-r * r + 2.0 * self.gamma * weight * r - weight * weight)
                        / (2.0 * (self.gamma - 1.0))
                } else {
                    0.5 * (self.gamma + 1.0) * weight * weight
                }
            }
        }
    }

    fn grad_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    weight * t / r - t / self.gamma
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                let denom = self.gamma - 1.0;
                if r <= weight {
                    weight * t / r
                } else if r <= self.gamma * weight {
                    self.gamma * weight * t / (denom * r) - t / denom
                } else {
                    0.0
                }
            }
        }
    }

    fn hess_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    weight * eps2 / (r * r * r) - 1.0 / self.gamma
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                let denom = self.gamma - 1.0;
                if r <= weight {
                    weight * eps2 / (r * r * r)
                } else if r <= self.gamma * weight {
                    self.gamma * weight * eps2 / (denom * r * r * r) - 1.0 / denom
                } else {
                    0.0
                }
            }
        }
    }

    /// Diagonal of the **PSD majorizer** for a single coordinate.
    ///
    /// SCAD/MCP are nonconvex: within their active region the penalty splits
    /// into a convex smoothed-ℓ¹ part (`λr`, resp. `γλr/(γ−1)`) and a concave
    /// quadratic taper (`−t²/(2γ)`, resp. `−t²/(2(γ−1))`). The exact Hessian
    /// [`Self::hess_one`] adds the concave constant (`−1/γ`, `−1/(γ−1)`) and is
    /// therefore negative across most of the active region.
    ///
    /// The MM/LLA majorizer keeps the convex part's reweighted-ℓ² curvature and
    /// majorizes the concave quadratic by its tangent line (zero curvature):
    /// this is exactly [`Self::hess_one`] with the concave constant dropped.
    /// Beyond the active cutoff the penalty is flat (Hessian `0`), so the
    /// majorizer is `0`. The result satisfies both legs of the trait contract:
    ///
    /// * `B ⪰ 0`: `λε²/r³ ≥ 0` and the constant `0` branch are nonnegative.
    /// * `B ⪰ ∂²P`: it exceeds the exact Hessian by exactly the dropped
    ///   concave constant (`1/γ` for MCP, `1/(γ−1)` for SCAD's middle region)
    ///   and equals it in the convex first SCAD region and the flat tail.
    fn psd_majorizer_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        let eps2 = self.smoothing_eps * self.smoothing_eps;
        match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    weight * eps2 / (r * r * r)
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad => {
                let denom = self.gamma - 1.0;
                if r <= weight {
                    weight * eps2 / (r * r * r)
                } else if r <= self.gamma * weight {
                    self.gamma * weight * eps2 / (denom * r * r * r)
                } else {
                    0.0
                }
            }
        }
    }

    fn grad_log_weight_one(&self, t: f64, weight: f64) -> f64 {
        let r = self.smooth_abs(t);
        let d_p_d_weight = match self.variant {
            PenaltyConcavity::Mcp => {
                if r <= self.gamma * weight {
                    r
                } else {
                    self.gamma * weight
                }
            }
            PenaltyConcavity::Scad => {
                if r <= weight {
                    r
                } else if r <= self.gamma * weight {
                    (self.gamma * r - weight) / (self.gamma - 1.0)
                } else {
                    (self.gamma + 1.0) * weight
                }
            }
        };
        weight * d_p_d_weight
    }

    pub fn diag_target(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for (i, &t) in target.iter().enumerate() {
            out[i] = self.hess_one(t, weight);
        }
        out
    }

    pub fn log_det_plus_lambda_i(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        lambda: f64,
    ) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "ScadMcpPenalty::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        let diag = self.diag_target(target, rho);
        let mut sum = 0.0;
        for &entry in diag.iter() {
            let shifted = lambda + entry;
            if !(shifted.is_finite() && shifted > 0.0) {
                return Err(format!(
                    "ScadMcpPenalty::log_det_plus_lambda_i non-positive shifted diagonal {shifted:.3e}"
                ));
            }
            sum += shifted.ln();
        }
        Ok(sum)
    }
}

impl AnalyticPenalty for ScadMcpPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let weight = self.resolved_weight(rho);
        let mut acc = 0.0;
        for &t in target.iter() {
            acc += self.value_one(t, weight);
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for (i, &t) in target.iter().enumerate() {
            out[i] = self.grad_one(t, weight);
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        Some(self.diag_target(target, rho))
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
        let diag = self.diag_target(target, rho);
        let mut out = Array1::<f64>::zeros(v.len());
        for i in 0..v.len() {
            out[i] = diag[i] * v[i];
        }
        out
    }

    /// PSD majorizer diagonal (see [`Self::psd_majorizer_one`]). SCAD/MCP are
    /// nonconvex, so this overrides the convex-only trait default — which would
    /// otherwise return the exact, negative [`Self::hessian_diag`] — with the
    /// reweighted-ℓ² MM surrogate. Coordinate-separable, so the inherited
    /// [`AnalyticPenalty::psd_majorizer_hvp`] correctly applies this as a
    /// diagonal operator.
    fn psd_majorizer_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let weight = self.resolved_weight(rho);
        let mut out = Array1::<f64>::zeros(target.len());
        for (i, &t) in target.iter().enumerate() {
            out[i] = self.psd_majorizer_one(t, weight);
        }
        Some(out)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        if !self.learnable_weight {
            return Array1::<f64>::zeros(0);
        }
        let weight = self.resolved_weight(rho);
        let mut grad = 0.0;
        for &t in target.iter() {
            grad += self.grad_log_weight_one(t, weight);
        }
        let mut out = Array1::<f64>::zeros(1);
        out[self.rho_index] = grad;
        out
    }

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "scad_mcp"
    }

    impl_scalar_apply_schedule!(weight);
}
