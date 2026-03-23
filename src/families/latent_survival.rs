//! One-block diagonal IRLS family for latent-frailty survival with compiled
//! sufficient statistics.
//!
//! Model: Λ(a|U) = B(a)·exp(U),  U ~ N(μ, σ²),  σ fixed.

use crate::estimate::UnifiedFitResult;
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, fit_custom_family,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::lognormal_kernel::{
    FrailtySpec, HazardLoading, LatentSurvivalRow, LatentSurvivalRowJet, RowJet2Block,
};
use crate::quadrature::QuadratureContext;
use crate::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array1, ArrayView2};
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

pub struct LatentSurvivalTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
}

pub fn fit_latent_survival_terms(
    data: ArrayView2<'_, f64>,
    rows: Vec<LatentSurvivalRow>,
    weights: Array1<f64>,
    spec: TermCollectionSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentSurvivalTermFitResult, String> {
    let latent_sd = validate_latent_survival_inputs(data, &rows, &weights, &frailty)?;
    let design = build_term_collection_design(data, &spec).map_err(|e| e.to_string())?;
    let resolvedspec =
        freeze_term_collection_from_design(&spec, &design).map_err(|e| e.to_string())?;
    let family = LatentSurvivalFamily {
        rows,
        weights,
        latent_sd,
        quadctx: Arc::new(QuadratureContext::new()),
    };
    let block = build_eta_blockspec(&design, data.nrows());
    let fit = fit_custom_family(&family, &[block], options).map_err(|e| e.to_string())?;
    Ok(LatentSurvivalTermFitResult {
        fit,
        design,
        resolvedspec,
    })
}

fn validate_latent_survival_inputs(
    data: ArrayView2<'_, f64>,
    rows: &[LatentSurvivalRow],
    weights: &Array1<f64>,
    frailty: &FrailtySpec,
) -> Result<f64, String> {
    if data.nrows() == 0 {
        return Err("latent-survival requires a non-empty dataset".to_string());
    }
    if rows.len() != data.nrows() || weights.len() != data.nrows() {
        return Err(format!(
            "latent-survival size mismatch: data has {} rows, rows has {}, weights has {}",
            data.nrows(),
            rows.len(),
            weights.len()
        ));
    }
    match frailty {
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading: HazardLoading::Full,
        } if sigma.is_finite() && *sigma >= 0.0 => Ok(*sigma),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading: HazardLoading::Full,
        } => Err(format!(
            "latent-survival requires a finite fixed hazard-multiplier sigma >= 0, got {sigma}"
        )),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(_),
            loading,
        } => Err(format!(
            "latent-survival currently supports only HazardLoading::Full, got {loading:?}"
        )),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: None, ..
        } => Err("latent-survival currently requires a fixed hazard-multiplier sigma".to_string()),
        FrailtySpec::GaussianShift { .. } => {
            Err("latent-survival requires HazardMultiplier frailty, not GaussianShift".to_string())
        }
        FrailtySpec::None => Err(
            "latent-survival requires a fixed HazardMultiplier frailty specification".to_string(),
        ),
    }
}

fn build_eta_blockspec(design: &TermCollectionDesign, n_rows: usize) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "eta".to_string(),
        design: design.design.clone(),
        offset: Array1::zeros(n_rows),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: Array1::zeros(design.penalties.len()),
        initial_beta: None,
    }
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

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
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

// ─── Learnable-σ variant ─────────────────────────────────────────────────────

/// Two-block latent-frailty survival family with learnable σ.
///
/// Block 0 ("eta"):       covariate linear predictor μ_i = X_i·β
/// Block 1 ("log_sigma"): intercept-only t = log(σ), shared across rows
///
/// σ-derivatives come from the heat-equation identities via [`RowJet2Block`],
/// which uses exact 4th-order kernel recurrences.
#[derive(Clone)]
pub struct LatentSurvivalLearnableSigmaFamily {
    pub rows: Vec<LatentSurvivalRow>,
    pub weights: Array1<f64>,
    pub quadctx: Arc<QuadratureContext>,
}

impl LatentSurvivalLearnableSigmaFamily {
    pub const BLOCK_ETA: usize = 0;
    pub const BLOCK_LOG_SIGMA: usize = 1;

    pub fn parameter_names() -> &'static [&'static str] {
        &["eta", "log_sigma"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Log]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "latent_survival_learnable_sigma",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }
}

impl CustomFamily for LatentSurvivalLearnableSigmaFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 2 {
            return Err(format!(
                "LatentSurvivalLearnableSigmaFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.rows.len();
        let eta_mu = &block_states[Self::BLOCK_ETA].eta;
        let eta_t = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_mu.len() != n || eta_t.len() != n || self.weights.len() != n {
            return Err("LatentSurvivalLearnableSigmaFamily input size mismatch".to_string());
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

            let jet = RowJet2Block::latent_survival(&self.quadctx, &self.rows[i], mu_i, sigma_i)
                .map_err(|e| format!("LatentSurvivalLearnableSigmaFamily row {i}: {e}"))?;

            ll += wi * jet.log_lik;

            // Block 0 (mu)
            if wi == 0.0 || jet.neg_hessian_mu <= 0.0 {
                w_mu[i] = 0.0;
                z_mu[i] = mu_i;
            } else {
                z_mu[i] = mu_i + jet.score_mu / jet.neg_hessian_mu;
                let raw = wi * jet.neg_hessian_mu;
                w_mu[i] = if !raw.is_finite() || raw <= 0.0 {
                    0.0
                } else {
                    raw.max(MIN_WEIGHT)
                };
            }

            // Block 1 (log_sigma)
            if wi == 0.0 || jet.neg_hessian_t <= 0.0 {
                w_t[i] = 0.0;
                z_t[i] = t_i;
            } else {
                z_t[i] = t_i + jet.score_t / jet.neg_hessian_t;
                let raw = wi * jet.neg_hessian_t;
                w_t[i] = if !raw.is_finite() || raw <= 0.0 {
                    0.0
                } else {
                    raw.max(MIN_WEIGHT)
                };
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
                "LatentSurvivalLearnableSigmaFamily expects 2 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.rows.len();
        let eta_mu = &block_states[Self::BLOCK_ETA].eta;
        let eta_t = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_mu.len() != n || eta_t.len() != n || self.weights.len() != n {
            return Err("LatentSurvivalLearnableSigmaFamily input size mismatch".to_string());
        }

        let mut ll = 0.0;
        for i in 0..n {
            let wi = self.weights[i];
            if wi == 0.0 {
                continue;
            }
            let sigma_i = eta_t[i].exp();
            let jet =
                LatentSurvivalRowJet::evaluate(&self.quadctx, &self.rows[i], eta_mu[i], sigma_i)
                    .map_err(|e| format!("LatentSurvivalLearnableSigmaFamily row {i}: {e}"))?;
            ll += wi * jet.log_lik;
        }
        Ok(ll)
    }
}
