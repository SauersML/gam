//! Two-block diagonal IRLS family for latent models with learnable σ.
//!
//! Block 0 ("eta"):       μ_i = X_i · β   (covariate linear predictor)
//! Block 1 ("log_sigma"): t = log(σ)       (intercept-only, shared across rows)
//!
//! The σ-derivatives are computed from the heat-equation identities via
//! [`RowJet2Block`], which is shared across row kernels. The same 2-block
//! IRLS machinery works for binary, survival, and other latent families.

use crate::families::custom_family::{
    BlockWorkingSet, CustomFamily, FamilyEvaluation, ParameterBlockState,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::lognormal_kernel::{
    LatentSurvivalRow, RowJet2Block,
};
use crate::quadrature::QuadratureContext;
use ndarray::Array1;
use std::sync::Arc;

const MIN_WEIGHT: f64 = 1e-12;

#[inline]
fn floor_positive_weight(raw: f64, min_w: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        0.0
    } else {
        raw.max(min_w)
    }
}

fn expect_two_blocks<'a>(
    block_states: &'a [ParameterBlockState],
    name: &str,
) -> Result<(&'a ParameterBlockState, &'a ParameterBlockState), String> {
    if block_states.len() != 2 {
        return Err(format!("{name} expects 2 blocks, got {}", block_states.len()));
    }
    Ok((&block_states[0], &block_states[1]))
}

/// Per-row kernel callback: given (quadctx, mu, sigma), produce a [`RowJet2Block`].
///
/// This is the only thing that varies between binary and survival latent
/// families.  Everything else (IRLS assembly, σ heat-equation) is shared.
enum LatentRowKernel {
    /// Binary latent-cloglog: each row has y ∈ {0,1} and log_mass offset.
    BinaryCloglog {
        y: Array1<f64>,
        log_mass: Array1<f64>,
    },
    /// Latent survival: each row carries compiled sufficient statistics.
    Survival {
        rows: Vec<LatentSurvivalRow>,
    },
}

impl LatentRowKernel {
    fn n_rows(&self) -> usize {
        match self {
            Self::BinaryCloglog { y, .. } => y.len(),
            Self::Survival { rows } => rows.len(),
        }
    }

    fn eval_row(
        &self,
        quadctx: &QuadratureContext,
        i: usize,
        mu: f64,
        sigma: f64,
    ) -> Result<RowJet2Block, String> {
        match self {
            Self::BinaryCloglog { y, log_mass } => {
                RowJet2Block::binary_cloglog(quadctx, y[i], mu, log_mass[i], sigma)
                    .map_err(|e| e.to_string())
            }
            Self::Survival { rows } => {
                RowJet2Block::latent_survival(quadctx, &rows[i], mu, sigma)
                    .map_err(|e| e.to_string())
            }
        }
    }

    fn eval_row_ll_only(
        &self,
        quadctx: &QuadratureContext,
        i: usize,
        mu: f64,
        sigma: f64,
    ) -> Result<f64, String> {
        // For log-likelihood only, the 1-block jets are cheaper since we skip
        // the d4 computation needed for σ curvature.
        match self {
            Self::BinaryCloglog { y, log_mass } => {
                use crate::families::lognormal_kernel::BinaryCloglogRowJet;
                let jet = BinaryCloglogRowJet::evaluate(quadctx, y[i], mu, log_mass[i], sigma)
                    .map_err(|e| e.to_string())?;
                Ok(jet.log_lik)
            }
            Self::Survival { rows } => {
                use crate::families::lognormal_kernel::LatentSurvivalRowJet;
                let jet = LatentSurvivalRowJet::evaluate(quadctx, &rows[i], mu, sigma)
                    .map_err(|e| e.to_string())?;
                Ok(jet.log_lik)
            }
        }
    }
}

// ─── Shared 2-block IRLS assembly ────────────────────────────────────────────

/// Generic learnable-σ family.  Uses [`RowJet2Block`] for heat-equation
/// σ-derivatives, independent of which latent row kernel is in use.
#[derive(Clone)]
pub struct LearnableSigmaFamily {
    kernel: LatentRowKernel,
    pub weights: Array1<f64>,
    pub quadctx: Arc<QuadratureContext>,
}

impl Clone for LatentRowKernel {
    fn clone(&self) -> Self {
        match self {
            Self::BinaryCloglog { y, log_mass } => Self::BinaryCloglog {
                y: y.clone(),
                log_mass: log_mass.clone(),
            },
            Self::Survival { rows } => Self::Survival {
                rows: rows.clone(),
            },
        }
    }
}

impl LearnableSigmaFamily {
    /// Construct for binary latent-cloglog model.
    pub fn binary_cloglog(
        y: Array1<f64>,
        weights: Array1<f64>,
        log_mass: Array1<f64>,
        quadctx: Arc<QuadratureContext>,
    ) -> Self {
        Self {
            kernel: LatentRowKernel::BinaryCloglog { y, log_mass },
            weights,
            quadctx,
        }
    }

    /// Construct for latent survival model.
    pub fn latent_survival(
        rows: Vec<LatentSurvivalRow>,
        weights: Array1<f64>,
        quadctx: Arc<QuadratureContext>,
    ) -> Self {
        Self {
            kernel: LatentRowKernel::Survival { rows },
            weights,
            quadctx,
        }
    }

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "learnable_sigma",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for LearnableSigmaFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (state_mu, state_t) = expect_two_blocks(block_states, "LearnableSigmaFamily")?;
        let eta_mu = &state_mu.eta;
        let eta_t = &state_t.eta;
        let n = self.kernel.n_rows();
        if eta_mu.len() != n || eta_t.len() != n || self.weights.len() != n {
            return Err("LearnableSigmaFamily input size mismatch".to_string());
        }

        let mut ll = 0.0;
        let mut z_mu = Array1::<f64>::zeros(n);
        let mut w_mu = Array1::<f64>::zeros(n);
        let mut z_t = Array1::<f64>::zeros(n);
        let mut w_t = Array1::<f64>::zeros(n);

        for i in 0..n {
            let wi = self.weights[i];
            let mu_i = eta_mu[i];
            let t_i = eta_t[i];
            let sigma_i = t_i.exp();

            let jet = self.kernel.eval_row(&self.quadctx, i, mu_i, sigma_i)?;
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
        let (state_mu, state_t) = expect_two_blocks(block_states, "LearnableSigmaFamily")?;
        let eta_mu = &state_mu.eta;
        let eta_t = &state_t.eta;
        let n = self.kernel.n_rows();
        if eta_mu.len() != n || eta_t.len() != n || self.weights.len() != n {
            return Err("LearnableSigmaFamily input size mismatch".to_string());
        }

        let mut ll = 0.0;
        for i in 0..n {
            let wi = self.weights[i];
            if wi == 0.0 {
                continue;
            }
            let sigma_i = eta_t[i].exp();
            ll += wi * self.kernel.eval_row_ll_only(&self.quadctx, i, eta_mu[i], sigma_i)?;
        }
        Ok(ll)
    }
}
