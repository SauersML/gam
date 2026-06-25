use crate::families::bms::{
    LatentMeasureKind, LatentZConditionalCalibration, LatentZRankIntCalibration,
};
use crate::inference::model::{SavedCompiledFlexBlock, SavedLatentZNormalization};
use crate::matrix::DesignMatrix;
use crate::types::InverseLink;
use ndarray::{Array1, Array2};

/// Input to prediction routines. Contains the design matrix and metadata
/// needed for point prediction plus uncertainty quantification.
pub struct PredictInput {
    /// Design matrix for the primary (mean/location) block.
    pub design: DesignMatrix,
    /// Offset vector for the primary block.
    pub offset: Array1<f64>,
    /// Optional design matrix for the noise/scale block (GAMLSS/survival).
    pub design_noise: Option<DesignMatrix>,
    /// Optional offset vector for the noise/scale block.
    pub offset_noise: Option<Array1<f64>>,
    /// Optional auxiliary scalar covariate used by specialized predictors.
    pub auxiliary_scalar: Option<Array1<f64>>,
    /// Optional auxiliary matrix used by specialized predictors.
    pub auxiliary_matrix: Option<Array2<f64>>,
}

pub struct BernoulliMarginalSlopePredictor {
    pub beta_marginal: Array1<f64>,
    pub beta_logslope: Array1<f64>,
    pub beta_score_warp: Option<Array1<f64>>,
    pub beta_link_dev: Option<Array1<f64>>,
    pub base_link: InverseLink,
    pub z_column: String,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub covariance: Option<Array2<f64>>,
    pub score_warp_runtime: Option<SavedCompiledFlexBlock>,
    pub link_deviation_runtime: Option<SavedCompiledFlexBlock>,
    pub gaussian_frailty_sd: Option<f64>,
    pub latent_z_calibration: Option<LatentZRankIntCalibration>,
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
}
