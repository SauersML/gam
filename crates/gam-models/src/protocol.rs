use crate::bms::{DeviationBlockConfig, LatentZCheckMode};
use crate::survival::construction::SurvivalBaselineTarget;
use gam_problem::{InverseLink, StandardLink};

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
            base_link: InverseLink::Standard(StandardLink::Probit),
            score_warp,
            link_deviation,
            latent_score,
        }
    }

}

#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeProtocol {
    pub marginal: MarginalSlopeCalibrationProtocol,
    pub baseline_target: SurvivalBaselineTarget,
}

