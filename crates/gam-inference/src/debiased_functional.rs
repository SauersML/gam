//! The Riesz-representer debiased / Neyman-orthogonal estimate of a smooth
//! functional of a fitted standard GAM (#1055, #4550).
//!
//! The estimator chain used to live inside `gam-pyffi`'s
//! `model_debiased_functional_dataset_json_impl`: the training-row
//! materialization, the design rebuild, the recovery of the penalized Hessian
//! `H = XᵀWX + S(λ)` with its weighted-Gram fallback, the prior-weighted
//! Gaussian row scores, and the dispatch over target functionals. None of that
//! is a binding concern. Living in the FFI crate it could not be reached by the
//! CLI, could only be exercised through a Python build, and put a hand-derived
//! Gaussian score beside the engine's own weight convention — which is how
//! #3542 happened.
//!
//! What stays at the front end is what genuinely belongs there: parsing the
//! caller's request (JSON for Python, flags for the CLI) and encoding a query
//! frame into a design row under that front end's own frame conventions. The
//! entry point takes the result as data.

use gam_data::EncodedDataset;
use gam_models::fit_orchestration::{FitConfig, FitRequest, StandardFitRequest, materialize};
use gam_models::inference::model::{FittedModel, PredictModelClass};
use gam_models::inference::saved_summary::prediction_model_class_label;
use gam_models::survival::predict::fit_result_from_saved_model_for_prediction;
use gam_problem::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_sae::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use gam_terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_derivative_design,
    build_term_collection_design, smooth_term_feature_cols,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Which smooth functional of the fitted mean to estimate.
///
/// `Point`, `Linear` and `Contrast` are functionals of the mean at a QUERY row,
/// so they carry the design row that row evaluates to, together with the affine
/// offset the design's own gauge puts on it. Building that row means encoding a
/// frame against the model's saved schema, which is the front end's job and the
/// reason those two variants take a row rather than a dictionary.
pub enum DebiasedFunctionalTarget<'a> {
    /// `m(x0)` at one query row.
    Point {
        design_row: ArrayView1<'a, f64>,
        affine_offset: f64,
    },
    /// The same quantity, reported under the caller's `"linear"` name.
    Linear {
        design_row: ArrayView1<'a, f64>,
        affine_offset: f64,
    },
    /// `m(x0) − m(x1)`.
    Contrast {
        design_row_a: ArrayView1<'a, f64>,
        affine_offset_a: f64,
        design_row_b: ArrayView1<'a, f64>,
        affine_offset_b: f64,
    },
    /// `mean_i w_i · m(x_i)` over the training rows.
    AverageValue { weights: Option<Array1<f64>> },
    /// `mean_i w_i · m'(x_i)` over the training rows, differentiating the
    /// covariate at `derivative_column` of the training frame.
    ///
    /// `None` asks this module to resolve the column from the model: the single
    /// feature column every smooth term shares. A model with smooths over more
    /// than one covariate has no such column and is refused, because the
    /// estimand would otherwise depend on which smooth was written first.
    AverageDerivative {
        weights: Option<Array1<f64>>,
        derivative_column: Option<usize>,
    },
}

impl DebiasedFunctionalTarget<'_> {
    /// The name this target reports itself under.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Point { .. } => "point",
            Self::Linear { .. } => "linear",
            Self::Contrast { .. } => "contrast",
            Self::AverageValue { .. } => "average_value",
            Self::AverageDerivative { .. } => "average_derivative",
        }
    }
}

/// What a debiased estimate reports.
#[derive(Clone, Debug, PartialEq)]
pub struct DebiasedFunctionalReport {
    pub target: &'static str,
    pub theta_plugin: f64,
    pub theta_debiased: f64,
    pub se: f64,
    pub penalty_bias: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub ci_level: f64,
}

/// The two-sided normal quantile the reported interval uses, at `ci_level`.
const NORMAL_QUANTILE_975: f64 = 1.959_963_984_540_054;
const CI_LEVEL: f64 = 0.95;

/// Debias a smooth functional of `model` against its own training rows.
///
/// `materialization` must replay the fit's own weight and offset columns.
/// Checked rather than trusted: with a default config a `weights="w"` fit was
/// replayed with unit weights, so the scores and the Gram fallback described a
/// different model than the saved `H` and `β`, and the reported standard error
/// was off by the weight scale (#3542).
pub fn debiased_functional(
    model: &FittedModel,
    materialization: &FitConfig,
    dataset: &EncodedDataset,
    target: &DebiasedFunctionalTarget<'_>,
) -> Result<DebiasedFunctionalReport, String> {
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "debiased_functional: only standard GAM models are supported; got '{}'",
            prediction_model_class_label(model)
        ));
    }
    if materialization.weight_column != model.weight_column
        || materialization.offset_column != model.offset_column
    {
        return Err(
            "debiased_functional: the training rows must be materialized under the fit's own \
             weight and offset columns; a default config replays a weighted fit with unit \
             weights and reports a standard error off by the weight scale (#3542)"
                .to_string(),
        );
    }

    let formula = model.payload().formula.clone();
    let spec = model
        .payload()
        .resolved_termspec
        .as_ref()
        .ok_or_else(|| {
            "debiased_functional: model is missing resolved_termspec; refit to enable".to_string()
        })?
        .clone();

    let materialized =
        materialize(&formula, dataset, materialization).map_err(|e| format!("{e}"))?;
    let FitRequest::Standard(standard) = materialized.request else {
        return Err(
            "debiased_functional: formula materialized to a non-standard fit path".to_string(),
        );
    };

    // The design in the same coefficient-space columns the fit used.
    let design_built = build_term_collection_design(standard.data.view(), &spec)
        .map_err(|e| format!("debiased_functional: design rebuild failed: {e}"))?;
    let x = design_built
        .design
        .try_to_dense_arc("debiased_functional design")
        .map_err(|e| format!("debiased_functional: design densification failed: {e}"))?;

    let saved_fit = fit_result_from_saved_model_for_prediction(model)
        .map_err(|e| format!("debiased_functional: {e}"))?;
    // `H` and `X'WX` are read in the saved frame of `β` and the rebuilt design
    // (gam#3346).
    let h = saved_fit
        .saved_frame_penalized_hessian()
        .map_err(|reason| format!("debiased_functional: penalized Hessian: {reason}"))?
        .ok_or_else(|| {
            "debiased_functional: model does not carry a dense penalized Hessian; \
             refit with a smaller basis (dense fits only)"
                .to_string()
        })?;

    // Gaussian/identity is the only family whose row score this module derives,
    // and the weighted-Gram fallback below is only valid for the profiled
    // Gaussian weight convention, so the verdict is taken once, here.
    let family = model.likelihood();
    let is_gaussian_identity = matches!(family.response, ResponseFamily::Gaussian)
        && matches!(family.link, InverseLink::Standard(StandardLink::Identity));

    let saved_gram = saved_fit
        .saved_frame_weighted_gram()
        .map_err(|reason| format!("debiased_functional: weighted Gram: {reason}"))?;
    let xwx_owned: Array2<f64> = match saved_gram {
        Some(gram) => gram.into_owned(),
        None => {
            // #1622: the parametric-term fit path leaves `weighted_gram` at its
            // `None` default. For a profiled Gaussian/identity fit the stored
            // penalized Hessian is `H = XᵀWX + S(λ)` with `W = diag(prior
            // weights)` — the penalty is added UNSCALED, see the `cov_scale`
            // contract — so `XᵀWX = Xᵀ diag(w) X` exactly and `S(λ) = H − XᵀWX`
            // is recovered consistently.
            if !is_gaussian_identity {
                return Err(format!(
                    "debiased_functional: model does not carry the weighted Gram X'WX and \
                     it can only be reconstructed for Gaussian/identity models; this model \
                     uses family='{}'",
                    family.pretty_name()
                ));
            }
            let design = x.as_ref();
            let weights = standard.weights.view();
            if weights.len() != design.nrows() {
                return Err(format!(
                    "debiased_functional: prior-weight length {} does not match design rows {}",
                    weights.len(),
                    design.nrows()
                ));
            }
            let mut weighted = design.to_owned();
            for (mut row, &weight) in weighted.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|value| value * weight);
            }
            weighted.t().dot(design)
        }
    };

    let beta = saved_fit.beta_flat();
    if beta.len() != x.ncols() {
        return Err(format!(
            "debiased_functional: beta length {} does not match design width {}",
            beta.len(),
            x.ncols()
        ));
    }
    // The penalty gradient `S(λ)β = (H − XᵀWX)β`.
    let penalty_beta = (&*h - &xwx_owned).dot(&beta);

    let row_scores = gaussian_identity_row_scores(
        &design_built,
        x.as_ref(),
        &beta,
        &standard,
        is_gaussian_identity,
        &family,
    )?;

    let (gradient, functional_affine) = functional_gradient(
        target,
        &spec,
        &design_built,
        x.as_ref(),
        standard.data.view(),
    )?;

    let input = RieszInput {
        beta: beta.view(),
        functional_gradient: gradient.view(),
        row_scores: row_scores.view(),
        penalty_beta: penalty_beta.view(),
        leverage: None,
    };
    let report = debias_with_dense_hessian(&input, h.view())
        .map_err(|e| format!("debiased_functional: Riesz engine error: {e}"))?;

    let theta_plugin = report.theta_plugin + functional_affine;
    let theta_debiased = report.theta_onestep + functional_affine;
    let half_width = NORMAL_QUANTILE_975 * report.se;
    Ok(DebiasedFunctionalReport {
        target: target.label(),
        theta_plugin,
        theta_debiased,
        se: report.se,
        penalty_bias: report.penalty_bias,
        ci_lower: theta_debiased - half_width,
        ci_upper: theta_debiased + half_width,
        ci_level: CI_LEVEL,
    })
}

/// Per-row scores of the objective the saved `H` is the Hessian of.
///
/// For a Gaussian/identity model that objective is
/// `½ Σ_i w_i (y_i − η_i)² + ½ βᵀS(λ)β` with `w` the PRIOR weights — the same
/// `W` in `H = XᵀWX + S(λ)` — so `s_i = w_i · x_i · (η_i − y_i)`. Dropping `w_i`
/// pairs a weighted Hessian with unweighted scores: scaling every weight by `c`
/// is a pure dispersion change that leaves `β̂` where it was, yet it scaled `H`
/// by `c` and the influence values by `1/c`, so the reported standard error
/// moved by `1/c` (#3542).
///
/// Every other family needs its own derivative chain. None is derived here, and
/// the refusal is explicit rather than a silent Gaussian substitution.
fn gaussian_identity_row_scores(
    design_built: &TermCollectionDesign,
    x: &Array2<f64>,
    beta: &Array1<f64>,
    standard: &StandardFitRequest<'_>,
    is_gaussian_identity: bool,
    family: &LikelihoodSpec,
) -> Result<Array2<f64>, String> {
    if !is_gaussian_identity {
        return Err(format!(
            "debiased_functional: currently only supported for Gaussian/identity models; \
             this model uses family='{}'. Supply pre-computed row_scores via the low-level \
             gamfit._rust.debiased_functional() call for other families.",
            family.pretty_name()
        ));
    }
    let (n, p) = (x.nrows(), x.ncols());
    let prior_weights = standard.weights.view();
    if prior_weights.len() != n {
        return Err(format!(
            "debiased_functional: prior-weight length {} does not match design rows {n}",
            prior_weights.len()
        ));
    }
    let effective_offset = design_built
        .compose_offset(
            standard.offset.view(),
            "debiased functional training design",
        )
        .map_err(|error| error.to_string())?;
    let eta = x.dot(beta) + &effective_offset;
    let y = standard.y.view();
    let mut row_scores = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let weighted_residual = prior_weights[row] * (eta[row] - y[row]);
        let design_row = x.row(row);
        for column in 0..p {
            row_scores[[row, column]] = design_row[column] * weighted_residual;
        }
    }
    Ok(row_scores)
}

/// The functional's gradient `dθ/dβ` and the affine part of `θ` the design's
/// gauge carries outside the coefficient space.
fn functional_gradient(
    target: &DebiasedFunctionalTarget<'_>,
    spec: &TermCollectionSpec,
    design_built: &TermCollectionDesign,
    x: &Array2<f64>,
    data: ArrayView2<'_, f64>,
) -> Result<(Array1<f64>, f64), String> {
    match target {
        DebiasedFunctionalTarget::Point {
            design_row,
            affine_offset,
        }
        | DebiasedFunctionalTarget::Linear {
            design_row,
            affine_offset,
        } => Ok((design_row.to_owned(), *affine_offset)),
        DebiasedFunctionalTarget::Contrast {
            design_row_a,
            affine_offset_a,
            design_row_b,
            affine_offset_b,
        } => {
            let gradient = SmoothFunctional::Contrast {
                design_row_a: design_row_a.view(),
                design_row_b: design_row_b.view(),
            }
            .gradient()
            .map_err(|e| format!("debiased_functional: contrast gradient: {e}"))?;
            Ok((gradient, affine_offset_a - affine_offset_b))
        }
        DebiasedFunctionalTarget::AverageValue { weights } => {
            let gradient = SmoothFunctional::AverageValue {
                value_design: x.view(),
                weights: weights.as_ref().map(|w| w.view()),
            }
            .gradient()
            .map_err(|e| format!("debiased_functional: average_value gradient: {e}"))?;
            let affine = weighted_affine_mean(
                design_built.affine_offset.view(),
                weights.as_ref().map(|w| w.view()),
                "average_value",
            )?;
            Ok((gradient, affine))
        }
        DebiasedFunctionalTarget::AverageDerivative {
            weights,
            derivative_column,
        } => {
            // `average_derivative` needs rows of basis-function DERIVATIVES
            // `∂φ_j/∂x(x_i)`, NOT the value design `φ_j(x_i)`. Feeding the value
            // design returns the average VALUE instead (#1120). The derivative
            // design is built analytically and replayed through the same frozen
            // identifiability chart, so its columns align with `β`.
            let column = resolve_average_derivative_column(spec, *derivative_column)?;
            let derivative =
                build_term_collection_derivative_design(data, spec, column).map_err(|e| {
                    format!("debiased_functional: average_derivative design build failed: {e}")
                })?;
            if derivative.design.ncols() != x.ncols() {
                return Err(format!(
                    "debiased_functional: average_derivative design width {} does not \
                     match fitted coefficient width {}",
                    derivative.design.ncols(),
                    x.ncols()
                ));
            }
            let gradient = SmoothFunctional::AverageDerivative {
                derivative_design: derivative.design.view(),
                weights: weights.as_ref().map(|w| w.view()),
            }
            .gradient()
            .map_err(|e| format!("debiased_functional: average_derivative gradient: {e}"))?;
            let affine = weighted_affine_mean(
                derivative.affine_offset.view(),
                weights.as_ref().map(|w| w.view()),
                "average_derivative",
            )?;
            Ok((gradient, affine))
        }
    }
}

/// The covariate column an `average_derivative` differentiates.
///
/// An explicit column is the caller's; resolving a column NAME against a frame
/// is the front end's job and has already happened by here. Without one the
/// column is the single feature column every smooth term shares, which is a
/// property of the model rather than of the frame.
fn resolve_average_derivative_column(
    spec: &TermCollectionSpec,
    derivative_column: Option<usize>,
) -> Result<usize, String> {
    if let Some(column) = derivative_column {
        return Ok(column);
    }
    let mut columns: Vec<usize> = spec
        .smooth_terms
        .iter()
        .flat_map(smooth_term_feature_cols)
        .collect();
    columns.sort_unstable();
    columns.dedup();
    match columns.as_slice() {
        [single] => Ok(*single),
        [] => Err(
            "debiased_functional: average_derivative requires at least one smooth term \
             to differentiate; the model has no smooths"
                .to_string(),
        ),
        _ => Err(
            "debiased_functional: average_derivative is ambiguous because the model has \
             smooths over more than one covariate; specify the covariate via the \
             \"deriv_var\" key in the target spec"
                .to_string(),
        ),
    }
}

/// The mean of `values`, weighted by `weights` when they are given.
fn weighted_affine_mean(
    values: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    label: &str,
) -> Result<f64, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "debiased_functional: {label} affine rows must be finite and non-empty"
        ));
    }
    match weights {
        None => Ok(values.sum() / values.len() as f64),
        Some(weights) => {
            if weights.len() != values.len() || weights.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "debiased_functional: {label} weights must be finite with length {}, got {}",
                    values.len(),
                    weights.len()
                ));
            }
            let weight_sum = weights.sum();
            if !(weight_sum.is_finite() && weight_sum > 0.0) {
                return Err(format!(
                    "debiased_functional: {label} weights must have positive finite sum"
                ));
            }
            Ok(values.dot(&weights) / weight_sum)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_spec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
            level: Default::default(),
        }
    }

    /// An explicit column is taken as given; without one a model carrying no
    /// smooth has no covariate to differentiate and says so, rather than
    /// differentiating column zero.
    #[test]
    fn the_derivative_column_is_the_callers_or_the_models_single_smooth_column() {
        assert_eq!(
            resolve_average_derivative_column(&empty_spec(), Some(1)).unwrap(),
            1
        );
        let error = resolve_average_derivative_column(&empty_spec(), None).unwrap_err();
        assert!(error.contains("requires at least one smooth"), "{error}");
    }

    /// The weighted mean is exact: `(2·1 + 10·3)/(1 + 3) = 8`, and unweighted
    /// it is `(2 + 10)/2 = 6`. Both are sums and quotients of exactly
    /// representable values, so the comparison is an equality, not a band.
    #[test]
    fn the_affine_mean_is_the_weighted_mean_of_its_rows() {
        let values = ndarray::array![2.0_f64, 10.0];
        let weights = ndarray::array![1.0_f64, 3.0];
        assert_eq!(
            weighted_affine_mean(values.view(), Some(weights.view()), "test").unwrap(),
            8.0
        );
        assert_eq!(
            weighted_affine_mean(values.view(), None, "test").unwrap(),
            6.0
        );
    }

    /// Every refusal `weighted_affine_mean` owns, so a malformed weight vector
    /// cannot silently report a different estimand (#4277).
    #[test]
    fn a_malformed_affine_weighting_is_refused() {
        let values = ndarray::array![2.0_f64, 10.0];
        let empty = ndarray::Array1::<f64>::zeros(0);
        assert!(weighted_affine_mean(empty.view(), None, "test").is_err());
        assert!(weighted_affine_mean(ndarray::array![f64::NAN, 1.0].view(), None, "test").is_err());
        for weights in [
            ndarray::array![1.0_f64],
            ndarray::array![1.0_f64, f64::NAN],
            ndarray::array![0.0_f64, 0.0],
            ndarray::array![1.0_f64, -1.0],
        ] {
            assert!(
                weighted_affine_mean(values.view(), Some(weights.view()), "test").is_err(),
                "{weights:?}"
            );
        }
    }
}
