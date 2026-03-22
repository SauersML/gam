//! Two-block diagonal IRLS family for the binary latent-cloglog model with
//! learnable latent standard deviation.
//!
//! Model:  U_i ~ N(mu_i, sigma^2),  P(y=1 | U, m) = 1 - exp(-m * exp(U))
//!
//! Block 0 ("eta"):       mu_i = X_i * beta   (genotype/covariate coefficients)
//! Block 1 ("log_sigma"): t = log(sigma)       (intercept-only, shared across rows)
//!
//! The per-row log-likelihood, scores, and observed curvatures for both blocks
//! are computed by [`BinaryCloglogRowJet2Block`], which uses heat-equation
//! identities to obtain sigma-derivatives from mu-derivatives.

use crate::families::custom_family::{
    BlockWorkingSet, CustomFamily, FamilyEvaluation, ParameterBlockState,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::lognormal_kernel::BinaryCloglogRowJet2Block;
use crate::quadrature::QuadratureContext;
use ndarray::Array1;
use std::sync::Arc;

const BLOCK_ETA: usize = 0;
const BLOCK_LOG_SIGMA: usize = 1;
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

/// Binary latent-cloglog family with per-row exposure mass and learnable sigma.
#[derive(Clone)]
pub struct LatentCloglogBinomialLearnableSigmaFamily {
    /// Binary response (0 or 1).
    pub y: Array1<f64>,
    /// Prior weights.
    pub weights: Array1<f64>,
    /// Log exposure mass per row (offset).
    pub log_mass: Array1<f64>,
    /// Quadrature context for numerical integration.
    pub quadctx: Arc<QuadratureContext>,
}

impl LatentCloglogBinomialLearnableSigmaFamily {
    pub fn parameter_names() -> &'static [&'static str] {
        &["eta", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "latent_cloglog_binomial_learnable_sigma",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for LatentCloglogBinomialLearnableSigmaFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "LatentCloglogBinomialLearnableSigmaFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_mu = &block_states[BLOCK_ETA].eta;
        let eta_t = &block_states[BLOCK_LOG_SIGMA].eta;
        if eta_mu.len() != n || eta_t.len() != n || self.weights.len() != n || self.log_mass.len() != n {
            return Err("LatentCloglogBinomialLearnableSigmaFamily input size mismatch".to_string());
        }

        let mut ll = 0.0;
        let mut z_mu = Array1::<f64>::zeros(n);
        let mut w_mu = Array1::<f64>::zeros(n);
        let mut z_t = Array1::<f64>::zeros(n);
        let mut w_t = Array1::<f64>::zeros(n);

        for i in 0..n {
            let yi = self.y[i];
            let wi = self.weights[i];
            let mu_i = eta_mu[i];
            let t_i = eta_t[i];
            let sigma_i = t_i.exp();

            let jet = BinaryCloglogRowJet2Block::evaluate(
                &self.quadctx,
                yi,
                mu_i,
                self.log_mass[i],
                sigma_i,
            )
            .map_err(|e| e.to_string())?;

            ll += wi * jet.log_lik;

            // Block 0 (mu)
            if wi == 0.0 || jet.neg_hessian_mu <= 0.0 {
                w_mu[i] = 0.0;
                z_mu[i] = mu_i;
            } else {
                z_mu[i] = mu_i + jet.score_mu / jet.neg_hessian_mu;
                w_mu[i] = floor_positive_weight(wi * jet.neg_hessian_mu, MIN_WEIGHT);
            }

            // Block 1 (log_sigma)
            if wi == 0.0 || jet.neg_hessian_t <= 0.0 {
                w_t[i] = 0.0;
                z_t[i] = t_i;
            } else {
                z_t[i] = t_i + jet.score_t / jet.neg_hessian_t;
                w_t[i] = floor_positive_weight(wi * jet.neg_hessian_t, MIN_WEIGHT);
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: z_mu,
                    working_weights: w_mu,
                },
                BlockWorkingSet::Diagonal {
                    working_response: z_t,
                    working_weights: w_t,
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "LatentCloglogBinomialLearnableSigmaFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_mu = &block_states[BLOCK_ETA].eta;
        let eta_t = &block_states[BLOCK_LOG_SIGMA].eta;
        if eta_mu.len() != n || eta_t.len() != n || self.weights.len() != n || self.log_mass.len() != n {
            return Err("LatentCloglogBinomialLearnableSigmaFamily input size mismatch".to_string());
        }

        let mut ll = 0.0;
        for i in 0..n {
            let wi = self.weights[i];
            if wi == 0.0 {
                continue;
            }
            let sigma_i = eta_t[i].exp();
            let jet = BinaryCloglogRowJet2Block::evaluate(
                &self.quadctx,
                self.y[i],
                eta_mu[i],
                self.log_mass[i],
                sigma_i,
            )
            .map_err(|e| e.to_string())?;
            ll += wi * jet.log_lik;
        }

        Ok(ll)
    }
}
