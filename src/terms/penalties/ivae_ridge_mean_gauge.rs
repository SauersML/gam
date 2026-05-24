use crate::terms::analytic_penalties::IvaeRidgeMeanGauge;

use super::PenaltyManifest;

impl PenaltyManifest for IvaeRidgeMeanGauge {
    const KIND_TAG: &'static str = "ivae_ridge_mean_gauge";
    const PYTHON_WRAPPER: &'static str = "IvaeRidgeMeanGauge";
    const ROW_BLOCK_DIAGONAL: bool = false;
}

