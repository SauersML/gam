//! One-block diagonal IRLS family for latent-frailty survival with compiled
//! sufficient statistics.
//!
//! Model: Λ(a|U) = B(a)·exp(U),  U ~ N(μ, σ²),  σ fixed.

use crate::families::custom_family::{
    BlockWorkingSet, CustomFamily, FamilyEvaluation, ParameterBlockState,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::lognormal_kernel::{LatentSurvivalRow, LatentSurvivalRowJet};
use crate::quadrature::QuadratureContext;
use ndarray::Array1;
use std::sync::Arc;

const MIN_WEIGHT: f64 = 1e-12;

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

/// Latent-frailty survival family with compiled sufficient statistics.
///
/// Each row carries pre-computed masses and event type; the family integrates
/// over the latent variable U ~ N(μ, σ²) via Gauss–Hermite quadrature,
/// returning IRLS diagonal working quantities for the single linear predictor η = μ.
#[derive(Clone)]
pub struct LatentSurvivalFamily {
    pub rows: Vec<LatentSurvivalRow>,
    pub weights: Array1<f64>,
    pub latent_sd: f64,
    pub quadctx: Arc<QuadratureContext>,
}

impl LatentSurvivalFamily {
    pub const BLOCK_ETA: usize = 0;

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "latent_survival",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for LatentSurvivalFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let state = expect_single_block(block_states, "LatentSurvivalFamily")?;
        let eta = &state.eta;
        let n = self.rows.len();
        if eta.len() != n || self.weights.len() != n {
            return Err("LatentSurvivalFamily input size mismatch".to_string());
        }

        let sigma = self.latent_sd;
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);

        for i in 0..n {
            let jet = LatentSurvivalRowJet::evaluate(&self.quadctx, &self.rows[i], eta[i], sigma)
                .map_err(|e| format!("LatentSurvivalFamily row {i}: {e}"))?;

            ll += self.weights[i] * jet.log_lik;

            if self.weights[i] == 0.0 || jet.neg_hessian <= 0.0 {
                w[i] = 0.0;
                z[i] = eta[i];
            } else {
                let wi = self.weights[i] * jet.neg_hessian;
                w[i] = if wi < MIN_WEIGHT { MIN_WEIGHT } else { wi };
                z[i] = eta[i] + jet.score / jet.neg_hessian;
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

    fn log_likelihood_only(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let state = expect_single_block(block_states, "LatentSurvivalFamily")?;
        let eta = &state.eta;
        let n = self.rows.len();
        if eta.len() != n || self.weights.len() != n {
            return Err("LatentSurvivalFamily input size mismatch".to_string());
        }

        let sigma = self.latent_sd;
        let mut ll = 0.0;

        for i in 0..n {
            let jet = LatentSurvivalRowJet::evaluate(&self.quadctx, &self.rows[i], eta[i], sigma)
                .map_err(|e| format!("LatentSurvivalFamily row {i}: {e}"))?;
            ll += self.weights[i] * jet.log_lik;
        }

        Ok(ll)
    }
}
