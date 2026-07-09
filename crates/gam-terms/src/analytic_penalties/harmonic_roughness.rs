use super::*;

// ---------------------------------------------------------------------------
// Harmonic roughness penalty (graduated periodic-basis smoothness prior)
// ---------------------------------------------------------------------------

/// Default effective strength of the decoder-harmonic smoothness prior (#1282).
///
/// Provenance: introduced in `854319979` ("fix(#1282): suppress decoder
/// harmonics to force circle-atom specialization"). A circle atom's decoder maps
/// the periodic basis `[DC, cosθ, sinθ, cos2θ, sin2θ, …]` to `Rᴰ`; the harmonics
/// `h ≥ 2` let its image leave the fundamental 2-plane and *snake* through space
/// so one atom straddles two manifolds (routing collapses to chance while
/// reconstruction stays R² ≈ 0.99). Penalizing those harmonics with the periodic
/// basis's own `h⁴` roughness graduation confines each atom to a single 2-plane.
///
/// The torch training loss scales `regularization()` by a small coefficient
/// (~1e-5), so this internal weight of `5e3` lifts the harmonic term to an
/// effective magnitude (~5e-2) that actually suppresses snaking while leaving
/// the fundamental ellipse free. It is a hand-tuned constant, NOT data-derived:
/// the true manifolds are pure fundamental circles, so the prior is keyed on
/// nothing about the data (there is no data-scale-relative default to prefer
/// here the way `IsometryPenalty`'s `gbar` normalizer is). On the Rust REML lane
/// this strength should be handed to the outer loop (`learnable_weight = true`,
/// one owned ρ-axis) so the marginal likelihood selects it; the torch lane has
/// no REML, so it pins the fixed default below.
pub const DEFAULT_HARMONIC_ROUGHNESS_WEIGHT: f64 = 5.0e3;

/// Graduated diagonal roughness prior on a row-major `(n_eff, d)` latent block
/// whose leading axis is tiled by a fixed-length **period** of per-row weights.
///
/// For every row `r` and output column `j` the contribution is
///
///     weight · row_weights[r mod period] · target[r, j]²
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
        weight * acc
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
            let factor = 2.0 * weight * w;
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
            let value = 2.0 * weight * self.row_weights[r % period];
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

/// Floor on the weighted harmonic-coefficient energy `Σ Sᵢᵢ bᵢ²` used as the
/// denominator of the evidence-optimal precision, keeping `λ` finite as the
/// harmonics are driven toward zero.
const HARMONIC_EVIDENCE_ENERGY_FLOOR: f64 = 1.0e-12;

/// Evidence-optimal roughness precision `λ⋆` for a periodic decoder block under
/// the graduated Gaussian prior encoded by `row_weights` (the periodic penalty
/// Gram's diagonal, `Sᵢᵢ = h⁴` on harmonic rows, `0` on DC / fundamental).
///
/// This is the empirical-Bayes / REML variance-component optimum: treating each
/// penalized decoder coefficient `bᵢ` as a directly-observed draw of a zero-mean
/// Gaussian random effect with precision `λ·Sᵢᵢ`, the marginal-likelihood
/// stationary point is
///
/// ```text
///   ∂/∂λ Σᵢ [ ½ ln(λ Sᵢᵢ) − ½ λ Sᵢᵢ bᵢ² ] = 0
///     ⇒  λ⋆ = N_pen / Σᵢ Sᵢᵢ bᵢ²
/// ```
///
/// where the sum runs only over coefficients with `Sᵢᵢ > 0` and `N_pen` counts
/// them. This is exactly the criterion the Rust REML lane's outer ρ search
/// optimizes for a periodic atom's smoothness, specialized to the degenerate
/// (noise-free, no data-fit) observation model in which the coefficients are the
/// effects — so it needs only the current decoder blocks and the Gram the caller
/// already owns, and can be refreshed cheaply during torch training. It replaces
/// the former hand-tuned `5e3` constant with an evidence-sourced, decoder-scale-
/// relative precision: as the harmonics shrink the precision rises, and the
/// competing reconstruction loss in the joint objective keeps genuinely-needed
/// coefficients alive, so `λ⋆` self-calibrates instead of running away.
///
/// The target is the same row-major `(n_eff, d)` block the penalty consumes;
/// `row_weights` is tiled over the leading axis with period `row_weights.len()`.
/// Returns `0.0` when there is nothing to penalize (no positive Gram weight or a
/// malformed shape); the denominator is floored so the result is always finite.
#[must_use]
pub fn harmonic_roughness_evidence_weight(
    target: ArrayView1<'_, f64>,
    n_eff: usize,
    row_weights: ArrayView1<'_, f64>,
) -> f64 {
    let period = row_weights.len();
    if n_eff == 0 || period == 0 || target.is_empty() || !target.len().is_multiple_of(n_eff) {
        return 0.0;
    }
    let d = target.len() / n_eff;
    let mut weighted_energy = 0.0_f64;
    let mut n_penalized = 0.0_f64;
    for r in 0..n_eff {
        let w = row_weights[r % period];
        if !(w > 0.0) {
            continue;
        }
        let base = r * d;
        for j in 0..d {
            let x = target[base + j];
            weighted_energy += w * x * x;
            n_penalized += 1.0;
        }
    }
    if n_penalized == 0.0 {
        return 0.0;
    }
    n_penalized / weighted_energy.max(HARMONIC_EVIDENCE_ENERGY_FLOOR)
}
