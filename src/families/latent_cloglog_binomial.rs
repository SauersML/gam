//! One-block diagonal IRLS family for the binary latent-cloglog model.
//!
//! Model:  U_i ~ N(η_i, σ²),  P(y=1 | U, m) = 1 − exp(−m · exp(U))
//!
//! The exposure mass enters through `log_mass_i` (an offset added to η).
//! `latent_sd` (σ) is fixed in this version and not learned.
//!
//! The per-row log-likelihood, score, and observed curvature are computed by
//! [`BinaryCloglogRowJet`], which uses exact kernel recurrences on the
//! lognormal moment generating function.

use crate::families::custom_family::{
    BlockWorkingSet, CustomFamily, FamilyEvaluation, ParameterBlockState,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::lognormal_kernel::BinaryCloglogRowJet;
use crate::quadrature::QuadratureContext;
use ndarray::Array1;
use std::sync::Arc;

const MIN_WEIGHT: f64 = 1e-12;

/// Floor a raw IRLS weight to a safe positive minimum, or zero it out if
/// non-finite / non-positive.
#[inline]
fn floor_positive_weight(raw: f64, min_w: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        0.0
    } else {
        raw.max(min_w)
    }
}

/// Extract the single block from a one-block family, returning an error if
/// the slice does not contain exactly one element.
fn expect_single_block<'a>(
    block_states: &'a [ParameterBlockState],
    name: &str,
) -> Result<&'a ParameterBlockState, String> {
    if block_states.len() != 1 {
        return Err(format!(
            "{name} requires exactly 1 block, got {}",
            block_states.len()
        ));
    }
    Ok(&block_states[0])
}

/// Binary latent-cloglog family with per-row exposure mass and fixed latent SD.
#[derive(Clone)]
pub struct LatentCloglogBinomialFamily {
    /// Binary response (0 or 1).
    pub y: Array1<f64>,
    /// Prior weights.
    pub weights: Array1<f64>,
    /// Log exposure mass per row (offset).
    pub log_mass: Array1<f64>,
    /// Fixed latent standard deviation.
    pub latent_sd: f64,
    /// Quadrature context for numerical integration.
    pub quadctx: Arc<QuadratureContext>,
}

impl LatentCloglogBinomialFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "latent_cloglog_binomial",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for LatentCloglogBinomialFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let state = expect_single_block(block_states, "LatentCloglogBinomialFamily")?;
        let eta = &state.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n || self.log_mass.len() != n {
            return Err("LatentCloglogBinomialFamily input size mismatch".to_string());
        }

        let sigma = self.latent_sd;
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);

        for i in 0..n {
            let yi = self.y[i];
            let wi = self.weights[i];
            let eta_i = eta[i];

            let jet = BinaryCloglogRowJet::evaluate(&self.quadctx, yi, eta_i, self.log_mass[i], sigma)
                .map_err(|e| e.to_string())?;

            ll += wi * jet.log_lik;

            if wi == 0.0 || jet.neg_hessian <= 0.0 {
                // Degenerate row: park at current eta with zero weight.
                w[i] = 0.0;
                z[i] = eta_i;
            } else {
                // IRLS pseudo-response (Newton step).
                z[i] = eta_i + jet.score / jet.neg_hessian;
                // Working weight: prior weight * observed curvature.
                w[i] = floor_positive_weight(wi * jet.neg_hessian, MIN_WEIGHT);
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let state = expect_single_block(block_states, "LatentCloglogBinomialFamily")?;
        let eta = &state.eta;
        let n = self.y.len();
        if eta.len() != n || self.weights.len() != n || self.log_mass.len() != n {
            return Err("LatentCloglogBinomialFamily input size mismatch".to_string());
        }

        let sigma = self.latent_sd;
        let mut ll = 0.0;

        for i in 0..n {
            let jet = BinaryCloglogRowJet::evaluate(&self.quadctx, self.y[i], eta[i], self.log_mass[i], sigma)
                .map_err(|e| e.to_string())?;
            ll += self.weights[i] * jet.log_lik;
        }

        Ok(ll)
    }
}
