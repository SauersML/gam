use crate::families::bernoulli_marginal_slope::{
    DEFAULT_EMPIRICAL_LATENT_GRID_SIZE, DeviationBlockConfig, LatentMeasureSpec, LatentZCheckMode,
    LatentZNormalizationMode, LatentZPolicy,
};
use crate::families::survival_construction::SurvivalBaselineTarget;
use crate::types::{InverseLink, LinkFunction};

/// Calibration semantics for the latent score `z` consumed by marginal-slope
/// families. Every variant is fully effective — there are no silently-ignored
/// metadata fields.
#[derive(Clone, Debug)]
pub enum LatentScoreSemantics {
    /// z is already on a frozen latent scale and the calibration law is
    /// assumed (approximately) standard normal. `check_mode` controls whether
    /// the fit aborts (`Strict`), only warns (`WarnOnly`), or skips the
    /// normality diagnostics entirely (`Off`).
    FrozenConditionalNormal { check_mode: LatentZCheckMode },
    /// z will be centered/scaled inside the fit.
    FitWeightedNormalization,
    /// z is carried by its observed empirical latent measure instead of
    /// pretending the downstream calibration law is standard normal.
    EmpiricalLatentMeasure { normalize_location_scale: bool },
}

impl LatentScoreSemantics {
    pub fn into_policy(self) -> LatentZPolicy {
        match self {
            Self::FrozenConditionalNormal { check_mode } => LatentZPolicy {
                check_mode,
                ..LatentZPolicy::frozen_transformation_normal()
            },
            Self::FitWeightedNormalization => LatentZPolicy::exploratory_fit_weighted(),
            Self::EmpiricalLatentMeasure {
                normalize_location_scale,
            } => LatentZPolicy {
                normalization: if normalize_location_scale {
                    LatentZNormalizationMode::FitWeighted
                } else {
                    LatentZNormalizationMode::None
                },
                latent_measure: LatentMeasureSpec::GlobalEmpirical {
                    grid_size: DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
                },
                ..LatentZPolicy::exploratory_fit_weighted()
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarginalSlopeCalibrationProtocol {
    pub base_link: InverseLink,
    /// Optional cubic score-warp block. `None` selects the rigid
    /// (algebraic closed-form) path for the score-warp axis.
    pub score_warp: Option<DeviationBlockConfig>,
    /// Optional cubic link-deviation block. `None` selects the rigid
    /// (algebraic closed-form) path for the link-deviation axis.
    pub link_deviation: Option<DeviationBlockConfig>,
    pub latent_score: LatentScoreSemantics,
}

impl MarginalSlopeCalibrationProtocol {
    fn default_latent_score() -> LatentScoreSemantics {
        // WarnOnly mirrors `LatentZPolicy::frozen_transformation_normal`'s
        // own default: at biobank dimensionality the upstream conditional
        // transformation-normal preprocessor can leave the global latent z
        // mildly heavy-tailed without violating per-strata calibration.
        LatentScoreSemantics::FrozenConditionalNormal {
            check_mode: LatentZCheckMode::WarnOnly,
        }
    }

    /// Construct a probit-link marginal-slope protocol with caller-supplied
    /// optional score-warp / link-deviation blocks and explicit latent-score
    /// semantics. Pass `None` for either block to select the rigid algebraic
    /// closed-form path on that axis.
    pub fn probit(
        score_warp: Option<DeviationBlockConfig>,
        link_deviation: Option<DeviationBlockConfig>,
        latent_score: LatentScoreSemantics,
    ) -> Self {
        Self {
            base_link: InverseLink::Standard(LinkFunction::Probit),
            score_warp,
            link_deviation,
            latent_score,
        }
    }

    /// Rigid probit marginal-slope: no score-warp, no link-deviation.
    pub fn probit_rigid() -> Self {
        Self::probit(None, None, Self::default_latent_score())
    }

    /// Probit marginal-slope with both cubic blocks at their triple-penalty
    /// defaults.
    pub fn probit_with_score_and_link_wiggle() -> Self {
        let wiggle = DeviationBlockConfig::triple_penalty_default();
        Self::probit(
            Some(wiggle.clone()),
            Some(wiggle),
            Self::default_latent_score(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeProtocol {
    pub marginal: MarginalSlopeCalibrationProtocol,
    pub baseline_target: SurvivalBaselineTarget,
}

impl SurvivalMarginalSlopeProtocol {
    /// Survival marginal-slope on a Gompertz-Makeham baseline with the
    /// supplied marginal-calibration protocol. Score-warp, link-deviation,
    /// and latent-score semantics all flow through from `marginal` —
    /// nothing is baked in.
    pub fn gompertz_makeham_probit(marginal: MarginalSlopeCalibrationProtocol) -> Self {
        Self {
            marginal,
            baseline_target: SurvivalBaselineTarget::GompertzMakeham,
        }
    }
}
