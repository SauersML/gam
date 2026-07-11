use super::*;

// ---------------------------------------------------------------------------
// Harmonic roughness penalty (graduated periodic-basis smoothness prior)
// ---------------------------------------------------------------------------

/// Graduated diagonal roughness prior on a row-major `(n_eff, d)` latent block
/// whose leading axis is tiled by a fixed-length **period** of per-row weights.
///
/// For every row `r` and output column `j` the contribution is
///
///     ½ · weight · row_weights[r mod period] · target[r, j]²
///
/// summed over all rows and columns. This is exactly the diagonal periodic
/// roughness Gram of a Fourier decoder: for the standard odd-`K` layout
/// `[DC, {sinθ,cosθ}, {sin2θ,cos2θ}, …]` the per-period weight vector is
/// `row_weights[k] = h⁴` on the `(sin, cos)` rows of harmonic `h ≥ 2` and `0`
/// on the DC (`h = 0`) and fundamental (`h = 1`) rows, so the fundamental
/// ellipse is left free while higher harmonics are graduated by `h⁴` (the same
/// `∫(f'')²` weighting the periodic basis builds for its own penalty). The
/// period is one atom's `K` basis coefficients; `n_eff = F·K` tiles it across
/// all `F` atoms of a stacked `(F, K, D)` decoder without materializing `F·K`
/// weights.
///
/// All weights are non-negative, so the Hessian is diagonal and PSD — the
/// penalty is convex and composes cleanly with PIRLS/REML. When
/// `learnable_weight` is set, the resolved strength is `weight · exp(ρ)` and the
/// penalty owns one outer-loop ρ-axis (the REML lane); otherwise it is fixed.
#[derive(Debug, Clone)]
pub struct HarmonicRoughnessPenalty {
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major latent coefficient block (`F·K`).
    pub n_eff: usize,
    /// Per-period diagonal weights (one atom's `K` basis-coefficient weights),
    /// tiled across the leading axis with period `row_weights.len()`.
    pub row_weights: Array1<f64>,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl HarmonicRoughnessPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        weight: f64,
        n_eff: usize,
        row_weights: Array1<f64>,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "HarmonicRoughnessPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("HarmonicRoughnessPenalty::new requires n_eff > 0".to_string());
        }
        let period = row_weights.len();
        if period == 0 {
            return Err("HarmonicRoughnessPenalty::new requires non-empty row_weights".to_string());
        }
        if !n_eff.is_multiple_of(period) {
            return Err(format!(
                "HarmonicRoughnessPenalty::new requires n_eff ({n_eff}) to be a multiple of the \
                 row_weights period ({period})"
            ));
        }
        for (k, &w) in row_weights.iter().enumerate() {
            if !(w.is_finite() && w >= 0.0) {
                return Err(format!(
                    "HarmonicRoughnessPenalty::new requires finite non-negative row_weights, \
                     got row_weights[{k}] = {w}"
                ));
            }
        }
        Ok(Self {
            weight,
            n_eff,
            row_weights,
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
}

impl AnalyticPenalty for HarmonicRoughnessPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Beta
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(d) = self.latent_dim(target.len()) else {
            return 0.0;
        };
        let weight = self.resolved_weight(rho);
        let period = self.row_weights.len();
        let mut acc = 0.0;
        for r in 0..self.n_eff {
            let w = self.row_weights[r % period];
            if w == 0.0 {
                continue;
            }
            let base = r * d;
            let mut row_sq = 0.0;
            for j in 0..d {
                let x = target[base + j];
                row_sq += x * x;
            }
            acc += w * row_sq;
        }
        0.5 * weight * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        let Some(d) = self.latent_dim(target.len()) else {
            return out;
        };
        let weight = self.resolved_weight(rho);
        let period = self.row_weights.len();
        for r in 0..self.n_eff {
            let w = self.row_weights[r % period];
            if w == 0.0 {
                continue;
            }
            let factor = weight * w;
            let base = r * d;
            for j in 0..d {
                out[base + j] = factor * target[base + j];
            }
        }
        out
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let mut out = Array1::<f64>::zeros(target.len());
        let Some(d) = self.latent_dim(target.len()) else {
            return Some(out);
        };
        let weight = self.resolved_weight(rho);
        let period = self.row_weights.len();
        for r in 0..self.n_eff {
            let value = weight * self.row_weights[r % period];
            let base = r * d;
            for j in 0..d {
                out[base + j] = value;
            }
        }
        Some(out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "harmonic_roughness"
    }

    impl_scalar_apply_schedule!(weight);
}
