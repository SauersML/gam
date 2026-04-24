use crate::families::bernoulli_marginal_slope::{DeviationBlockConfig, LatentZPolicy};
use crate::families::survival_construction::SurvivalBaselineTarget;
use crate::types::{InverseLink, LinkFunction};

#[derive(Clone, Debug)]
pub enum LatentScoreSemantics {
    /// z is already on latent N(0,1) scale from a frozen, phenotype-free transform.
    FrozenConditionalNormal {
        transform_id: Option<String>,
        clip_eps: f64,
        require_approximately_standard: bool,
    },
    /// z will be centered/scaled inside the fit.
    FitWeightedNormalization,
}

impl LatentScoreSemantics {
    pub fn into_policy(self) -> LatentZPolicy {
        match self {
            Self::FrozenConditionalNormal { .. } => LatentZPolicy::frozen_transformation_normal(),
            Self::FitWeightedNormalization => LatentZPolicy::exploratory_fit_weighted(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarginalSlopeCalibrationProtocol {
    pub base_link: InverseLink,
    pub score_warp: DeviationBlockConfig,
    pub link_deviation: DeviationBlockConfig,
    pub latent_score: LatentScoreSemantics,
}

impl MarginalSlopeCalibrationProtocol {
    pub fn probit_with_score_and_link_wiggle() -> Self {
        let wiggle = DeviationBlockConfig::triple_penalty_default();
        Self {
            base_link: InverseLink::Standard(LinkFunction::Probit),
            score_warp: wiggle.clone(),
            link_deviation: wiggle,
            latent_score: LatentScoreSemantics::FrozenConditionalNormal {
                transform_id: None,
                clip_eps: 1e-6,
                require_approximately_standard: true,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeProtocol {
    pub marginal: MarginalSlopeCalibrationProtocol,
    pub baseline_target: SurvivalBaselineTarget,
    pub require_timewiggle: bool,
}

impl SurvivalMarginalSlopeProtocol {
    pub fn gompertz_makeham_probit_timewiggle() -> Self {
        Self {
            marginal: MarginalSlopeCalibrationProtocol::probit_with_score_and_link_wiggle(),
            baseline_target: SurvivalBaselineTarget::GompertzMakeham,
            require_timewiggle: true,
        }
    }
}
