//! The generic exponential-dispersion row kernel: one variance function
//! `V(μ)` composed with one inverse link `μ = h(η)`.
//!
//! An exponential-dispersion (EDM) row with prior weight `w` and dispersion
//! `φ` has the η-score
//!
//! ```text
//!   ℓ′(η) = s · (y − μ) · h′(η) / V(μ),     μ = h(η),   s = w/φ,
//! ```
//!
//! and every other row quantity a Laplace/LAML fit consumes is a derivative of
//! it: the observed information `W_obs = −ℓ″`, its η-derivatives
//! `c_obs = −ℓ‴`, `d_obs = −ℓ⁗`, `e_obs = −ℓ⁽⁵⁾`. The Fisher information
//! `W_F = s·h′²/V(μ)` and its η-derivatives drive the IRLS working state.
//!
//! [`EdmScoreProgram`] and [`EdmFisherProgram`] write those two expressions
//! ONCE, as [`RowProgram`]s over the generic [`JetScalar`]; evaluating either
//! on a seeded [`crate::jet_tower::Tower4`] yields the whole η-tower in one
//! pass by the exact Faà di Bruno / Leibniz algebra — no finite differences
//! and no hand-expanded derivative formulas. The only calculus supplied by
//! hand is the two derivative stacks the expression composes:
//!
//! * the variance-function jet `[V, V′, V″, V‴, V⁗]` ([`EdmVariance::jet`]),
//!   which is a polynomial in `μ` for every family here;
//! * the inverse-link jet `[h, h′, …, h⁽⁵⁾]` at `η`, supplied by the caller
//!   (the link vocabulary lives in `gam-solve`), of which the score tower
//!   through `d_obs` reads `h … h⁗` and `e_obs` reads `h⁽⁵⁾`.
//!
//! The per-family hand-written canonical kernels (Gaussian-identity,
//! Poisson-log, Gamma-log, the Bernoulli probability links) remain the fast
//! paths for their cells and are the oracles this kernel is pinned against.

use crate::jet_scalar::JetScalar;
use crate::jet_tower::RowProgram;
use crate::special::{bd0, log1p_minus_x};

/// The variance function of an exponential-dispersion response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdmVariance {
    /// `V(μ) = 1`.
    Gaussian,
    /// `V(μ) = μ`.
    Poisson,
    /// `V(μ) = μ²`.
    Gamma,
    /// `V(μ) = μ³`.
    InverseGaussian,
    /// `V(μ) = μ(1 − μ)`.
    Bernoulli,
}

impl EdmVariance {
    /// `[V, V′, V″, V‴, V⁗]` at `μ`. For the Bernoulli variance the value is
    /// formed from the caller's complement `1 − μ` (which a saturating link
    /// supplies without cancellation); the other families ignore it.
    #[inline]
    pub fn jet(self, mu: f64, one_minus_mu: f64) -> [f64; 5] {
        match self {
            Self::Gaussian => [1.0, 0.0, 0.0, 0.0, 0.0],
            Self::Poisson => [mu, 1.0, 0.0, 0.0, 0.0],
            Self::Gamma => [mu * mu, 2.0 * mu, 2.0, 0.0, 0.0],
            Self::InverseGaussian => [mu * mu * mu, 3.0 * mu * mu, 6.0 * mu, 6.0, 0.0],
            Self::Bernoulli => [mu * one_minus_mu, one_minus_mu - mu, -2.0, 0.0, 0.0],
        }
    }

    /// Whether `μ` lies in the open mean domain of the family: all of `ℝ` for
    /// the Gaussian, `μ > 0` for Poisson, Gamma and inverse Gaussian,
    /// `0 < μ < 1` for the Bernoulli.
    #[inline]
    pub fn mean_in_domain(self, mu: f64, one_minus_mu: f64) -> bool {
        match self {
            Self::Gaussian => mu.is_finite(),
            Self::Poisson | Self::Gamma | Self::InverseGaussian => mu.is_finite() && mu > 0.0,
            Self::Bernoulli => mu > 0.0 && one_minus_mu > 0.0,
        }
    }

    /// The unit deviance `d(y, μ)`, so a row's deviance is `w · d(y, μ)`:
    ///
    /// ```text
    /// Gaussian          (y − μ)²
    /// Poisson           2[y ln(y/μ) − (y − μ)]           (2μ at y = 0)
    /// Gamma             2[(y − μ)/μ − ln(y/μ)]
    /// inverse Gaussian  (y − μ)²/(y μ²)
    /// Bernoulli         2[y ln(y/μ) + (1 − y) ln((1 − y)/(1 − μ))]
    /// ```
    ///
    /// with `0 · ln 0 = 0`. `μ` must lie in [`Self::mean_in_domain`] and `y` in
    /// the family's response support; the caller certifies both.
    ///
    /// The Poisson, Gamma and Bernoulli forms above each subtract two `O(δ)`
    /// terms to leave an `O(δ²)` deviance, `δ = (y − μ)/μ`, so they are
    /// evaluated through the exact cancellation-free identities
    ///
    /// ```text
    /// Poisson    2·bd0(y, μ)
    /// Gamma      −2·log1p_minus_x((y − μ)/μ)
    /// Bernoulli  2[bd0(y, μ) + bd0(1 − y, 1 − μ)]
    /// ```
    ///
    /// (the Bernoulli pair's linear terms cancel exactly), the same primitives
    /// the canonical solver rows use.
    #[inline]
    pub fn unit_deviance(self, y: f64, mu: f64, one_minus_mu: f64) -> f64 {
        match self {
            Self::Gaussian => {
                let r = y - mu;
                r * r
            }
            Self::Poisson => 2.0 * bd0(y, mu),
            Self::Gamma => -2.0 * log1p_minus_x((y - mu) / mu),
            Self::InverseGaussian => {
                let r = y - mu;
                r * r / (y * mu * mu)
            }
            Self::Bernoulli => 2.0 * (bd0(y, mu) + bd0(1.0 - y, one_minus_mu)),
        }
    }
}

/// One exponential-dispersion row at its linear predictor `η`.
#[derive(Clone, Copy, Debug)]
pub struct EdmRow {
    /// Variance function of the response.
    pub variance: EdmVariance,
    /// Observed response.
    pub y: f64,
    /// Likelihood scale `s = w/φ` (prior weight over dispersion).
    pub scale: f64,
    /// Linear predictor.
    pub eta: f64,
    /// Inverse-link stack `[h, h′, h″, h‴, h⁗, h⁽⁵⁾]` evaluated at `eta`.
    pub link: [f64; 6],
    /// `1 − μ`, formed without cancellation by the link where it saturates
    /// (read by the Bernoulli variance and residual only).
    pub one_minus_mu: f64,
}

impl EdmRow {
    /// `μ = h(η)`.
    #[inline]
    pub fn mu(&self) -> f64 {
        self.link[0]
    }

    /// The residual `y − μ`. For the Bernoulli family it is formed as
    /// `(1 − μ) − (1 − y)` whenever `y > ½`, so a mean that rounds to one
    /// keeps its representable tail complement instead of cancelling.
    #[inline]
    pub fn residual(&self) -> f64 {
        match self.variance {
            EdmVariance::Bernoulli if self.y > 0.5 => self.one_minus_mu - (1.0 - self.y),
            _ => self.y - self.link[0],
        }
    }

    /// The row's unit deviance at its mean.
    #[inline]
    pub fn unit_deviance(&self) -> f64 {
        self.variance
            .unit_deviance(self.y, self.link[0], self.one_minus_mu)
    }

    #[inline]
    fn mean_and_variance<S: JetScalar<1>>(&self, eta: &S) -> (S, S, S) {
        let [h0, h1, h2, h3, h4, h5] = self.link;
        let mu = eta.compose_unary([h0, h1, h2, h3, h4]);
        let dmu = eta.compose_unary([h1, h2, h3, h4, h5]);
        let variance = mu.compose_unary(self.variance.jet(h0, self.one_minus_mu));
        (mu, dmu, variance)
    }
}

/// The η-score `ℓ′(η) = s (y − μ) h′(η)/V(μ)` of a block of rows as a
/// [`RowProgram`] in the single per-row primary `η`. A row's dense tower
/// carries `(ℓ′, ℓ″, ℓ‴, ℓ⁗, ℓ⁽⁵⁾) = (score, −W_obs, −c_obs, −d_obs, −e_obs)`.
#[derive(Clone, Copy, Debug)]
pub struct EdmScoreProgram<'a>(pub &'a [EdmRow]);

impl RowProgram<1> for EdmScoreProgram<'_> {
    fn n_rows(&self) -> usize {
        self.0.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 1], String> {
        Ok([self.0[row].eta])
    }

    fn eval<S: JetScalar<1>>(&self, row: usize, p: &[S; 1]) -> Result<S, String> {
        let row = &self.0[row];
        let (mu, dmu, variance) = row.mean_and_variance(&p[0]);
        // `y − μ` as a jet; its value channel is the cancellation-free residual.
        let residual = mu.neg().add_constant(row.y).with_value(row.residual());
        Ok(residual.mul(&dmu).mul(&variance.recip()).scale(row.scale))
    }
}

/// The Fisher weight `W_F = s h′(η)²/V(μ)` of a block of rows as a
/// [`RowProgram`] in the single per-row primary `η`. A row's dense tower
/// carries `(W_F, ∂W_F/∂η, ∂²W_F/∂η², …)`.
#[derive(Clone, Copy, Debug)]
pub struct EdmFisherProgram<'a>(pub &'a [EdmRow]);

impl RowProgram<1> for EdmFisherProgram<'_> {
    fn n_rows(&self) -> usize {
        self.0.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 1], String> {
        Ok([self.0[row].eta])
    }

    fn eval<S: JetScalar<1>>(&self, row: usize, p: &[S; 1]) -> Result<S, String> {
        let row = &self.0[row];
        let (_, dmu, variance) = row.mean_and_variance(&p[0]);
        Ok(dmu.mul(&dmu).mul(&variance.recip()).scale(row.scale))
    }
}

/// The η-derivative tower of one row's observed information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdmObservedTower {
    /// `ℓ′(η)`.
    pub score: f64,
    /// `W_obs = −ℓ″`.
    pub w: f64,
    /// `c_obs = ∂W_obs/∂η = −ℓ‴`.
    pub c: f64,
    /// `d_obs = ∂²W_obs/∂η² = −ℓ⁗`.
    pub d: f64,
    /// `e_obs = ∂³W_obs/∂η³ = −ℓ⁽⁵⁾`.
    pub e: f64,
}

/// The η-derivative tower of one row's Fisher information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdmFisherTower {
    /// `W_F`.
    pub w: f64,
    /// `∂W_F/∂η`.
    pub c: f64,
    /// `∂²W_F/∂η²`.
    pub d: f64,
}

impl EdmRow {
    /// The observed-information tower, from one dense evaluation of
    /// [`EdmScoreProgram`].
    pub fn observed_tower(&self) -> Result<EdmObservedTower, String> {
        let t = crate::jet_tower::program_full_tower(&EdmScoreProgram(std::slice::from_ref(self)), 0)?;
        Ok(EdmObservedTower {
            score: t.v,
            w: -t.g[0],
            c: -t.h[0][0],
            d: -t.t3[0][0][0],
            e: -t.t4[0][0][0][0],
        })
    }

    /// The Fisher-information tower, from one dense evaluation of
    /// [`EdmFisherProgram`].
    pub fn fisher_tower(&self) -> Result<EdmFisherTower, String> {
        let t = crate::jet_tower::program_full_tower(&EdmFisherProgram(std::slice::from_ref(self)), 0)?;
        Ok(EdmFisherTower {
            w: t.v,
            c: t.g[0],
            d: t.h[0][0],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::EdmVariance;

    fn assert_relative(label: &str, got: f64, expected: f64) {
        assert!(
            (got - expected).abs() <= 1e-13 * expected.abs(),
            "{label}: got {got:e}, expected {expected:e}"
        );
    }

    /// Near `y = μ` each deviance is `O(δ²)`, the difference of two `O(δ)`
    /// terms; the unit deviance must keep full relative precision there, where
    /// every converged fit evaluates it. `δ = 2⁻²⁰` is exact in binary, so the
    /// references are the Taylor series in `δ` truncated past rounding.
    #[test]
    fn unit_deviance_keeps_relative_precision_near_the_mean() {
        let delta = 2.0_f64.powi(-20);
        let (mu, y) = (1.0, 1.0 + delta);
        let d2 = delta * delta;
        assert_relative(
            "Poisson",
            EdmVariance::Poisson.unit_deviance(y, mu, 0.0),
            d2 * (1.0 - delta / 3.0 + d2 / 6.0),
        );
        assert_relative(
            "Gamma",
            EdmVariance::Gamma.unit_deviance(y, mu, 0.0),
            d2 * (1.0 - 2.0 * delta / 3.0 + d2 / 2.0),
        );
        // Bernoulli about one half: y = (1 + u)/2 gives
        // d = (1 + u)ln(1 + u) + (1 − u)ln(1 − u) = u² + u⁴/6 + u⁶/15 + ….
        let u = delta;
        assert_relative(
            "Bernoulli",
            EdmVariance::Bernoulli.unit_deviance(0.5 * (1.0 + u), 0.5, 0.5),
            u * u * (1.0 + u * u / 6.0),
        );
    }
}
