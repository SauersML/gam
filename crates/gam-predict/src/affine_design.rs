//! Exact affine representation of a fitted standard-model linear predictor.
//!
//! A raw mean-block design is not, by itself, a predictor contract: ordinary
//! GAMs also carry a row offset, while a fitted link-wiggle predictor is affine
//! in the complete saved joint coefficient vector only after its warp basis is
//! frozen at the fitted index.  This module owns that distinction so frontends
//! expose one shape-stable object instead of inferring coefficient coordinates
//! or covariance frames from matrix dimensions.

use std::ops::Range;

use gam_linalg::matrix::DesignMatrix;
use gam_models::inference::model::{FittedModel, PredictModelClass};
use gam_models::survival::predict::fit_result_from_saved_model_for_prediction;
use gam_problem::BlockRole;
use ndarray::{Array1, Array2};

use crate::PredictInput;

/// Named coordinate frame containing the coefficients multiplied by an
/// [`AffineDesign`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AffineCoefficientFrame {
    /// The complete coefficient vector of an ordinary standard GAM.
    Full,
    /// The complete saved `[Mean, LinkWiggle]` joint coefficient vector, with
    /// the link-wiggle basis frozen at its fitted index.
    LinkWiggleJoint,
}

impl AffineCoefficientFrame {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::LinkWiggleJoint => "link_wiggle_joint",
        }
    }
}

/// Exact coefficient covariances in the same named frame as an
/// [`AffineDesign`].
///
/// The definitions stay separate because they answer different inferential
/// questions.  Absence is represented as `None`; no covariance definition is
/// ever substituted for another.
#[derive(Clone, Debug, Default)]
pub struct AffineCovariances {
    /// Conditional Bayesian covariance `Vb = Var(beta | lambda_hat)`.
    pub conditional: Option<Array2<f64>>,
    /// Smoothing-parameter-corrected Bayesian covariance `Vp`.
    pub smoothing_corrected: Option<Array2<f64>>,
    /// Frequentist sandwich covariance `Ve`.
    pub frequentist: Option<Array2<f64>>,
}

/// A self-contained fitted affine predictor `offset + matrix * coefficients`.
#[derive(Clone, Debug)]
pub struct AffineDesign {
    pub offset: Array1<f64>,
    pub matrix: DesignMatrix,
    pub coefficients: Array1<f64>,
    pub coefficient_frame: AffineCoefficientFrame,
    /// Half-open slice in `coefficient_frame` represented by `coefficients`.
    pub coefficient_range: Range<usize>,
    /// Covariances in exactly `coefficient_frame`; definitions are never
    /// silently substituted when one is unavailable.
    pub covariances: AffineCovariances,
}

fn fitted_covariances(fit: &gam_solve::estimate::UnifiedFitResult) -> AffineCovariances {
    AffineCovariances {
        conditional: fit.beta_covariance().cloned(),
        smoothing_corrected: fit.beta_covariance_corrected().cloned(),
        frequentist: fit.beta_covariance_ve().cloned(),
    }
}

fn checked_affine_design(
    offset: Array1<f64>,
    matrix: DesignMatrix,
    coefficients: Array1<f64>,
    coefficient_frame: AffineCoefficientFrame,
    covariances: AffineCovariances,
) -> Result<AffineDesign, String> {
    if offset.len() != matrix.nrows() {
        return Err(format!(
            "affine design row mismatch: offset has {} rows but matrix has {}",
            offset.len(),
            matrix.nrows()
        ));
    }
    if coefficients.len() != matrix.ncols() {
        return Err(format!(
            "affine design coefficient mismatch in '{}' frame: matrix has {} columns but coefficient frame has {} entries",
            coefficient_frame.name(),
            matrix.ncols(),
            coefficients.len()
        ));
    }
    if let Some((row, value)) = offset
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "affine design offset is non-finite at row {row}: {value}"
        ));
    }
    if let Some((column, value)) = coefficients
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "affine design '{}' coefficient is non-finite at column {column}: {value}",
            coefficient_frame.name()
        ));
    }
    for (name, covariance) in [
        ("conditional", covariances.conditional.as_ref()),
        (
            "smoothing-corrected",
            covariances.smoothing_corrected.as_ref(),
        ),
        ("frequentist", covariances.frequentist.as_ref()),
    ] {
        let Some(covariance) = covariance else {
            continue;
        };
        if covariance.dim() != (coefficients.len(), coefficients.len()) {
            return Err(format!(
                "affine design {name} covariance in '{}' frame is {}x{}, expected {}x{}",
                coefficient_frame.name(),
                covariance.nrows(),
                covariance.ncols(),
                coefficients.len(),
                coefficients.len(),
            ));
        }
        if let Some(((row, column), value)) = covariance
            .indexed_iter()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "affine design {name} covariance in '{}' frame is non-finite at ({row}, {column}): {value}",
                coefficient_frame.name(),
            ));
        }
    }
    let width = coefficients.len();
    Ok(AffineDesign {
        offset,
        matrix,
        coefficients,
        coefficient_frame,
        coefficient_range: 0..width,
        covariances,
    })
}

/// Build the exact fitted affine predictor for a standard saved model.
///
/// Ordinary standard GAMs return their model offset, full design, and full
/// coefficient frame.  Link-wiggle fits return the model offset, the joint
/// `[X, B(warp_index)]` design, and the complete saved `[Mean, LinkWiggle]`
/// coefficient frame. `warp_index` includes the persisted #2141 frozen-index
/// shift and is fixed at the fitted state.  Keeping the mean block in the
/// matrix is essential: it lets the returned same-frame covariances represent
/// mean variance and mean--wiggle cross-covariance exactly.
pub fn fitted_standard_affine_design(
    model: &FittedModel,
    input: &PredictInput,
) -> Result<AffineDesign, String> {
    if model.predict_model_class() != PredictModelClass::Standard {
        return Err(format!(
            "affine design supports standard GAM models; got '{}'",
            model.predict_model_class().name()
        ));
    }
    let fit =
        fit_result_from_saved_model_for_prediction(model).map_err(|error| error.to_string())?;
    let link_wiggle = model
        .saved_link_wiggle()
        .map_err(|error| error.to_string())?;

    match link_wiggle {
        None => {
            if fit.blocks.len() != 1 || fit.blocks[0].role != BlockRole::Mean {
                return Err(format!(
                    "ordinary standard affine design requires exactly one Mean coefficient block; got roles {:?}",
                    fit.blocks
                        .iter()
                        .map(|block| block.role)
                        .collect::<Vec<_>>()
                ));
            }
            checked_affine_design(
                input.offset.clone(),
                input.design.clone(),
                fit.beta.clone(),
                AffineCoefficientFrame::Full,
                fitted_covariances(&fit),
            )
        }
        Some(runtime) => {
            if fit.blocks.len() != 2
                || fit.blocks[0].role != BlockRole::Mean
                || fit.blocks[1].role != BlockRole::LinkWiggle
            {
                return Err(format!(
                    "link-wiggle affine design requires fitted blocks in [Mean, LinkWiggle] order; got roles {:?}",
                    fit.blocks
                        .iter()
                        .map(|block| block.role)
                        .collect::<Vec<_>>()
                ));
            }
            if runtime.beta.is_empty() {
                return Err(
                    "link-wiggle affine design has an empty saved prediction coefficient frame"
                        .to_string(),
                );
            }
            let mean_beta = &fit.blocks[0].beta;
            let wiggle_beta = &fit.blocks[1].beta;
            if runtime.beta.len() != wiggle_beta.len()
                || runtime
                    .beta
                    .iter()
                    .zip(wiggle_beta.iter())
                    .any(|(saved, fitted)| saved.to_bits() != fitted.to_bits())
            {
                return Err(
                    "link-wiggle affine design requires saved prediction coefficients to equal the fitted LinkWiggle block"
                        .to_string(),
                );
            }
            if input.design.ncols() != mean_beta.len() {
                return Err(format!(
                    "link-wiggle affine base mismatch: mean design has {} columns but Mean block has {} coefficients",
                    input.design.ncols(),
                    mean_beta.len()
                ));
            }
            let base = input.design.dot(mean_beta) + &input.offset;
            let warp_index = runtime
                .warp_index(&base, &input.design)
                .map_err(|error| error.to_string())?;
            let wiggle_design = runtime
                .design(&warp_index)
                .map_err(|error| error.to_string())?;
            let joint_design = DesignMatrix::hstack(vec![
                input.design.clone(),
                DesignMatrix::from(wiggle_design),
            ])?;
            checked_affine_design(
                input.offset.clone(),
                joint_design,
                fit.beta.clone(),
                AffineCoefficientFrame::LinkWiggleJoint,
                fitted_covariances(&fit),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn checked_affine_design_reproduces_offset_matrix_coefficient_sum() {
        let result = checked_affine_design(
            array![0.5, -0.25],
            DesignMatrix::from(array![[1.0, 2.0], [-1.0, 3.0]]),
            array![0.4, -0.2],
            AffineCoefficientFrame::Full,
            AffineCovariances::default(),
        )
        .expect("valid affine design");
        let eta = result.matrix.dot(&result.coefficients) + &result.offset;
        assert_eq!(eta, array![0.5, -1.25]);
        assert_eq!(result.coefficient_frame, AffineCoefficientFrame::Full);
        assert_eq!(result.coefficient_range, 0..2);
    }

    #[test]
    fn checked_affine_design_rejects_frame_width_mismatch() {
        let error = checked_affine_design(
            array![0.0],
            DesignMatrix::from(array![[1.0, 2.0]]),
            array![0.5],
            AffineCoefficientFrame::LinkWiggleJoint,
            AffineCovariances::default(),
        )
        .expect_err("mismatched coefficient frame must fail");
        assert!(error.contains("matrix has 2 columns"));
        assert!(error.contains("link_wiggle_joint"));
    }

    #[test]
    fn checked_affine_design_rejects_covariance_from_another_frame() {
        let error = checked_affine_design(
            array![0.0],
            DesignMatrix::from(array![[1.0, 2.0]]),
            array![0.5, -0.25],
            AffineCoefficientFrame::Full,
            AffineCovariances {
                conditional: Some(Array2::eye(1)),
                smoothing_corrected: None,
                frequentist: None,
            },
        )
        .expect_err("cross-frame covariance must fail");
        assert!(error.contains("conditional covariance"));
        assert!(error.contains("1x1, expected 2x2"));
    }
}
