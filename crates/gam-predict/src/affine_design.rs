//! Exact affine representation of a fitted standard-model linear predictor.
//!
//! A raw mean-block design is not, by itself, a predictor contract: ordinary
//! GAMs also carry a row offset, while a fitted link-wiggle predictor is affine
//! in the complete saved joint coefficient vector only after its warp basis is
//! frozen at the fitted index.  This module owns that distinction so frontends
//! expose one shape-stable object instead of inferring coefficient coordinates
//! or covariance frames from matrix dimensions.
//!
//! Value and derivative are also distinct concepts here.  Reproducing `η̂` and
//! propagating a coefficient covariance are two different questions, and for a
//! curved fitted link they have two different operators; the object therefore
//! carries both rather than letting a caller pair the wrong one with the
//! covariances it ships.

use std::ops::Range;

use gam_linalg::matrix::DesignMatrix;
use gam_models::inference::model::{FittedModel, PredictModelClass};
use gam_models::survival::predict::fit_result_from_saved_model_for_prediction;
use gam_problem::BlockRole;
use ndarray::{Array1, Array2};

use crate::linalg::design_row_chunk;
use crate::{LinkWiggleGradientLayout, PredictInput, link_wiggle_eta_gradient_rows};

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

/// How the derivative `∂η/∂β` relates to the value operator of an
/// [`AffineDesign`].
///
/// These are the same matrix exactly when the fitted predictor is linear in its
/// coefficients.  A fitted link wiggle breaks that: its warp index moves with
/// the Mean coefficients, so the value operator and the derivative genuinely
/// differ and both are needed — the first reproduces `η̂`, the second is the
/// only one that may be combined with a coefficient covariance.
#[derive(Clone, Debug)]
pub enum AffineEtaGradient {
    /// The predictor is linear in `coefficients`, so the value operator IS the
    /// derivative and no second matrix exists.
    Design,
    /// A curved fitted link makes `∂η/∂β` differ from the value operator.
    Distinct(DesignMatrix),
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
    /// Relationship of `∂η/∂β` to `matrix`; read it through
    /// [`AffineDesign::eta_gradient_matrix`].
    pub eta_gradient: AffineEtaGradient,
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
    eta_gradient: AffineEtaGradient,
) -> Result<AffineDesign, String> {
    if let AffineEtaGradient::Distinct(gradient) = &eta_gradient
        && (gradient.nrows() != matrix.nrows() || gradient.ncols() != matrix.ncols())
    {
        return Err(format!(
            "affine design eta gradient in '{}' frame is {}x{}, expected {}x{}",
            coefficient_frame.name(),
            gradient.nrows(),
            gradient.ncols(),
            matrix.nrows(),
            matrix.ncols(),
        ));
    }
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
        eta_gradient,
    })
}

/// Why a saved model has no affine coefficient-frame design at all, or `None`
/// when [`fitted_standard_affine_design`] can serve it.
///
/// This is a property of the fitted model alone, not of any prediction rows, so
/// a frontend that must materialise those rows before it can call
/// [`fitted_standard_affine_design`] can decline first — with these words, not
/// its own. The decision and its wording live here so the Rust API, the CLI,
/// and the Python surface cannot drift apart on which models are refusable or
/// why (SPEC: one source of truth, behaviour parity).
pub fn affine_design_unavailable_reason(model: &FittedModel) -> Result<Option<String>, String> {
    // A scan-routed model intentionally owns no finite B-spline coefficient
    // frame: its exact O(n) state-space predictor has no coefficient vector to
    // multiply, so the refusal is structural rather than a missing feature
    // (#1046). Checked before the model class because a scan fit is otherwise
    // an ordinary standard model.
    if let Some((feature_column, _)) = model
        .saved_spline_scan()
        .map_err(|error| error.to_string())?
    {
        return Ok(Some(format!(
            "s({feature_column}) is fit by the exact O(n) state-space spline scan, which does not \
             have a finite coefficient-frame design; design_matrix() is unavailable for this \
             fitted model."
        )));
    }
    if model.predict_model_class() != PredictModelClass::Standard {
        return Ok(Some(format!(
            "design_matrix supports standard GAM models; got '{}'. For other classes use \
             Model.predict, whose saved predictor can contain multiple coupled parameter surfaces.",
            model.predict_model_class().name()
        )));
    }
    Ok(None)
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
///
/// The returned value operator satisfies `offset + matrix·β̂ == η̂` exactly, but
/// for a link-wiggle fit it is NOT `∂η/∂β`: the warp index itself moves with
/// the Mean coefficients. Variance and contrast math must use
/// [`AffineDesign::eta_gradient_matrix`], which is built by the same authority
/// the predict standard-error path uses.
pub fn fitted_standard_affine_design(
    model: &FittedModel,
    input: &PredictInput,
) -> Result<AffineDesign, String> {
    if let Some(reason) = affine_design_unavailable_reason(model)? {
        return Err(reason);
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
                // η = offset + X·β is linear in β, so the design IS ∂η/∂β.
                AffineEtaGradient::Design,
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
            // One evaluation of the frozen-index warp basis serves both
            // operators: the shared gradient authority builds
            // `[diag(dq/dq0)·X, B(index)]`, and the joint VALUE operator reuses
            // that exact `B(index)` block. Recomputing `B` separately would let
            // the two representations drift.
            let layout = LinkWiggleGradientLayout {
                p_main: mean_beta.len(),
                p_total: mean_beta.len() + wiggle_beta.len(),
                wiggle_col_start: mean_beta.len(),
            };
            let mean_rows = design_row_chunk(&input.design, 0..input.design.nrows())?;
            let gradient =
                link_wiggle_eta_gradient_rows(&mean_rows, &warp_index, &runtime, layout)?;
            let wiggle_design = gradient
                .slice(ndarray::s![.., layout.wiggle_col_start..layout.p_total])
                .to_owned();
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
                // The warp index `u = X·β_m + offset + X·s` moves with the Mean
                // coefficients, so `∂η/∂β_m = diag(1 + B'(u)·β_w)·X`, not `X`.
                AffineEtaGradient::Distinct(DesignMatrix::from(gradient)),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_models::inference::model::SavedLinkWiggleRuntime;
    use gam_models::wiggle::monotone_wiggle_basis_with_derivative_order;
    use ndarray::array;

    #[test]
    fn checked_affine_design_rejects_frame_width_mismatch() {
        let error = checked_affine_design(
            array![0.0],
            DesignMatrix::from(array![[1.0, 2.0]]),
            array![0.5],
            AffineCoefficientFrame::LinkWiggleJoint,
            AffineCovariances::default(),
            AffineEtaGradient::Design,
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
            AffineEtaGradient::Design,
        )
        .expect_err("cross-frame covariance must fail");
        assert!(error.contains("conditional covariance"));
        assert!(error.contains("1x1, expected 2x2"));
    }

    #[test]
    fn checked_affine_design_rejects_eta_gradient_of_another_shape() {
        let error = checked_affine_design(
            array![0.0, 0.0],
            DesignMatrix::from(array![[1.0, 2.0], [0.5, -1.0]]),
            array![0.5, -0.25],
            AffineCoefficientFrame::LinkWiggleJoint,
            AffineCovariances::default(),
            AffineEtaGradient::Distinct(DesignMatrix::from(array![[1.0, 2.0]])),
        )
        .expect_err("an eta gradient outside the value operator's shape must fail");
        assert!(error.contains("eta gradient"));
        assert!(error.contains("1x2, expected 2x2"));
    }

    /// The whole point of the second operator: for a fitted link wiggle the
    /// value operator is NOT `∂η/∂β`, so pairing it with the shipped covariance
    /// would silently disagree with the standard errors `predict` reports.
    ///
    /// The fitted predictor is reconstructed end to end from the saved runtime
    /// (`base` and the #2141 warp index are both recomputed at each perturbed
    /// `β_m`, exactly as a refit would), and its central difference is compared
    /// against the shared gradient authority. Finite differences are a test-only
    /// instrument here: production uses the closed form.
    #[test]
    fn link_wiggle_eta_gradient_is_the_derivative_the_value_operator_is_not() {
        let a = -1.6_f64;
        let b = 2.4_f64;
        let width = b - a;
        let mut knot_values = vec![a; 4];
        knot_values.extend([a + 0.18 * width, a + 0.41 * width, a + 0.77 * width]);
        knot_values.extend(vec![b; 4]);
        let knots = Array1::from_vec(knot_values);
        let probe = array![-0.5, 0.0, 0.5];
        let basis_width = monotone_wiggle_basis_with_derivative_order(probe.view(), &knots, 3, 0)
            .expect("monotone warp basis")
            .ncols();
        // A materially curved, monotone (β_w ≥ 0) warp.
        let wiggle_beta: Vec<f64> = (0..basis_width)
            .map(|index| 0.35 + 0.11 * index as f64)
            .collect();
        let runtime = SavedLinkWiggleRuntime {
            knots: knots.to_vec(),
            degree: 3,
            penalty_metadata: None,
            beta: wiggle_beta,
            // A nonzero #2141 frozen-index shift, so the index is not the base.
            index_shift: Some(vec![0.12, -0.07]),
        };
        let x = array![
            [1.0, -0.60],
            [1.0, -0.20],
            [1.0, 0.15],
            [1.0, 0.55],
            [1.0, 0.90],
        ];
        let design = DesignMatrix::from(x.clone());
        let offset = array![0.05, -0.10, 0.00, 0.20, -0.15];
        let mean_beta = array![0.30, 0.85];
        let layout = LinkWiggleGradientLayout {
            p_main: mean_beta.len(),
            p_total: mean_beta.len() + runtime.beta.len(),
            wiggle_col_start: mean_beta.len(),
        };

        // The fitted predictor as a function of the Mean coefficients, with the
        // warp index rebuilt from scratch at every evaluation.
        let eta_at = |beta_m: &Array1<f64>| -> Array1<f64> {
            let base = design.dot(beta_m) + &offset;
            let index = runtime
                .warp_index(&base, &design)
                .expect("warp index at the perturbed mean coefficients");
            runtime
                .apply_with_index(&base, &index)
                .expect("warped predictor at the perturbed mean coefficients")
        };

        let warp_index = runtime
            .warp_index(&(design.dot(&mean_beta) + &offset), &design)
            .expect("fitted warp index");
        let gradient = link_wiggle_eta_gradient_rows(&x, &warp_index, &runtime, layout)
            .expect("shared eta gradient authority");

        let h = 1e-6;
        let mut max_gradient_error = 0.0_f64;
        let mut max_value_operator_error = 0.0_f64;
        for column in 0..layout.p_main {
            let mut plus = mean_beta.clone();
            plus[column] += h;
            let mut minus = mean_beta.clone();
            minus[column] -= h;
            let central = (eta_at(&plus) - eta_at(&minus)) / (2.0 * h);
            for row in 0..x.nrows() {
                max_gradient_error =
                    max_gradient_error.max((gradient[[row, column]] - central[row]).abs());
                // `x` is the Mean block of the joint VALUE operator.
                max_value_operator_error =
                    max_value_operator_error.max((x[[row, column]] - central[row]).abs());
            }
        }
        assert!(
            max_gradient_error < 1e-6,
            "shared gradient authority must reproduce the fitted predictor's derivative; \
             max |analytic - central difference| = {max_gradient_error:.3e}"
        );
        assert!(
            max_value_operator_error > 1e-2,
            "the value operator's Mean block must be measurably NOT the derivative, otherwise \
             this fixture cannot distinguish the two operators; max |X - dη/dβ_m| = \
             {max_value_operator_error:.3e}"
        );

        // The wiggle block is shared verbatim between the two operators, so the
        // value identity and the derivative agree exactly on `β_w`.
        let value_wiggle = runtime
            .design(&warp_index)
            .expect("frozen-index warp basis");
        for row in 0..x.nrows() {
            for column in 0..runtime.beta.len() {
                assert_eq!(
                    gradient[[row, layout.wiggle_col_start + column]],
                    value_wiggle[[row, column]]
                );
            }
        }
    }
}
