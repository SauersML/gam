use super::*;

// ---------------------------------------------------------------------------
// ARD penalty
// ---------------------------------------------------------------------------

/// ARD (Automatic Relevance Determination) over latent axes.
///
/// One independent quadratic ridge penalty per latent axis, with one
/// REML-selectable log-precision per axis. Penalty contribution for axis `j`:
///
/// ```text
///   P_j(t; ρ) = ½ α_j · ‖t[:, j]‖² - (n_eff / 2) · log α_j,
///   α_j = weight · exp(ρ_j)
/// ```
///
/// summed over `j ∈ [0, d)` for the extension-coordinate target block
/// `T ∈ ℝ^{n_eff × d}`. In the SAE objective this is the latent-axis
/// dimension-selection prior: under REML, axis `j` whose data evidence is too
/// weak gets `ρ_j → +∞` (precision → ∞, coefficients → 0), so the latent
/// dimension is effectively pruned.
///
/// Because the penalty is quadratic and block-diagonal in latent axes, it
/// reduces to a [`BlockwisePenalty`] per axis and slots into the existing
/// canonical-penalty pipeline with zero extra wiring beyond appending `d`
/// hyperparameter axes to `ρ`.
///
/// Gotchas:
///
/// * ARD is not a standalone identifiability fix. The intrinsic dimensionality
///   is meaningful only after a separate gauge-fixing prior (`AuxPrior`,
///   `IsometryPenalty`, or an equivalent basis constraint) has fixed rotations
///   and reparameterizations.
/// * `n_eff` controls the Gaussian normalizer / Occam term. Override it only
///   when rows have been aggregated or otherwise represent a different
///   effective observation count than `target.len() / latent_dim`.
/// * The row-major `LatentCoordValues` layout means each per-axis ridge is
///   strided in memory; [`Self::as_blockwise`] expands it into scalar
///   `BlockwisePenalty` entries rather than pretending each axis is contiguous.
///
/// When to use: any [`crate::terms::latent_coord::LatentCoordValues`] block
/// where the intrinsic dimension is unknown. Compose with `IsometryPenalty`
/// for full gauge fixing.
#[derive(Debug, Clone)]
pub struct ARDPenalty {
    pub target: PsiSlice,
    pub latent_dim: usize,
    pub weight: f64,
    pub weight_schedule: Option<ScalarWeightSchedule>,
    /// Local ρ indices for the `d` per-axis log-precisions.
    pub rho_indices: Vec<usize>,
    /// Effective number of observations contributing to each latent axis.
    /// Enters the per-axis log-determinant Occam term in `grad_rho`:
    /// at an unused axis (Σ_n t_{n,j}² = 0) the gradient becomes
    /// `-n_eff / 2`, which under minimization pushes ρ_j → +∞ and prunes
    /// the axis. Default is the number of latent-row observations
    /// (`target.len() / latent_dim`).
    pub n_eff: f64,
}


impl ARDPenalty {
    #[must_use]
    pub fn new(target: PsiSlice, latent_dim: usize) -> Self {
        assert!(latent_dim > 0, "ARDPenalty requires latent_dim > 0");
        let n_obs = if latent_dim == 0 {
            0
        } else {
            target.len() / latent_dim
        };
        let rho_indices = (0..latent_dim).collect();
        Self {
            target,
            latent_dim,
            weight: 1.0,
            weight_schedule: None,
            rho_indices,
            n_eff: n_obs as f64,
        }
    }

    impl_with_weight_schedule!(weight);

    /// Override the effective observation count used in the Occam log-det
    /// term (default: `target.len() / latent_dim`). Pass the number of
    /// latent rows that actually contribute to axis `j` (uniform across
    /// axes for the current implementation).
    #[must_use = "build error must be handled"]
    pub fn with_n_eff(mut self, n_eff: f64) -> Result<Self, String> {
        if !(n_eff.is_finite() && n_eff >= 0.0) {
            return Err(format!(
                "ARDPenalty::with_n_eff requires a finite non-negative value, got {n_eff}"
            ));
        }
        self.n_eff = n_eff;
        Ok(self)
    }

    /// Build scalar [`BlockwisePenalty`] entries for each latent-axis row.
    /// Fixes the audit finding that the row-major `LatentCoordValues` layout
    /// (`n * d + j`) cannot be represented as one contiguous per-axis range.
    pub fn as_blockwise(&self, global_offset: usize) -> Vec<BlockwisePenalty> {
        let n_obs = self.target.len() / self.latent_dim;
        let mut out = Vec::with_capacity(n_obs * self.latent_dim);
        for j in 0..self.latent_dim {
            for n in 0..n_obs {
                let idx = global_offset + self.target.range.start + n * self.latent_dim + j;
                out.push(BlockwisePenalty::ridge(idx..idx + 1, 1.0).with_op(None));
            }
        }
        out
    }
}


impl AnalyticPenalty for ARDPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut acc = 0.0;
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            let mut sq = 0.0;
            for n in 0..n_obs {
                let v = target[n * d + j];
                sq += v * v;
            }
            acc += 0.5 * lam_j * sq - 0.5 * self.n_eff * lam_j.ln();
        }
        acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut g = Array1::<f64>::zeros(target.len());
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            for n in 0..n_obs {
                g[n * d + j] = lam_j * target[n * d + j];
            }
        }
        g
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut diag = Array1::<f64>::zeros(target.len());
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            for n in 0..n_obs {
                diag[n * d + j] = lam_j;
            }
        }
        Some(diag)
    }

    fn grad_rho(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Uses the prior normalizer -0.5 * N_eff * log(weight * exp(rho_j)).
        let d = self.latent_dim;
        let n_obs = target.len() / d;
        let mut out = Array1::<f64>::zeros(self.rho_count());
        for j in 0..d {
            let lam_j = resolve_learnable_weight(self.weight, rho[self.rho_indices[j]]);
            let mut sq = 0.0;
            for n in 0..n_obs {
                let v = target[n * d + j];
                sq += v * v;
            }
            out[self.rho_indices[j]] = 0.5 * lam_j * sq - 0.5 * self.n_eff;
        }
        out
    }

    fn rho_count(&self) -> usize {
        self.latent_dim
    }

    fn name(&self) -> &str {
        "ard"
    }

    impl_scalar_apply_schedule!(weight);
}


