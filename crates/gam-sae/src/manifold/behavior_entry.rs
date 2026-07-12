//! Unified, binding-neutral behavior-anchored manifold-SAE entry (#2015).
//!
//! Behavior is an ordinary second output block of the shared-chart crosscoder:
//! the activation matrix is the anchor and the nats-unit sphere-tangent image of
//! the row-aligned probability distributions is the sole non-anchor block.  The
//! existing crosscoder outer objective therefore selects `log(lambda_y)` in the
//! same converged penalized quasi-Laplace run that selects every other variance component.
//! There is no binding-owned fit fork and no bounded inner-only fit object.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use ndarray::Array2;
use serde::Serialize;

use super::*;

/// Automatic behavior-fit request shared by Rust, Python, and CLI front doors.
pub struct SaeBehaviorAutoFitRequest {
    /// Activation response, `N x P_x`.
    pub activation: Array2<f64>,
    /// Row-aligned behavioral distributions, `N x V`.
    pub probabilities: Array2<f64>,
    /// The single Rust-owned fit configuration used by the crosscoder engine.
    pub config: SaeCrosscoderAutoFitConfig,
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Residual-variance certificate for the fitted behavior-block weight.
#[derive(Clone, Debug, Serialize)]
pub struct BehaviorWeightIdentifiability {
    pub identifiable: bool,
    pub activation_residual_variance: f64,
    pub behavior_residual_variance: f64,
    /// Conditional observed curvature of the profiled two-block penalized quasi-Laplace criterion
    /// with respect to `log(lambda_y)`. It is strictly positive exactly when
    /// both response blocks retain residual variance.
    pub log_lambda_curvature: f64,
}

/// Honest KL summary of the decoded fitted behavior distributions.
#[derive(Clone, Debug, Serialize)]
pub struct BehaviorKlSummary {
    pub finite_rows: usize,
    pub infinite_rows: usize,
    /// `None` when any row has infinite KL; an infinite loss is never averaged
    /// away into a finite-looking number.
    pub mean_kl_nats: Option<f64>,
    pub max_kl_nats: Option<f64>,
}

/// Completed behavior-anchored fit. The crosscoder owns the fitted term and the
/// honest activation/behavior tangent layers; the remaining fields are the
/// behavior-specific physical interpretation of that same state.
pub struct SaeBehaviorFitReport {
    pub crosscoder: SaeCrosscoderFitReport,
    pub behavior_block: BehaviorBlock,
    pub target_probabilities: Array2<f64>,
    pub fitted_probabilities: Array2<f64>,
    pub weight_identifiability: BehaviorWeightIdentifiability,
    pub kl: BehaviorKlSummary,
    pub isometry: Vec<Option<AtomBehaviorIsometry>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BehaviorPinnedChartWire {
    pub coords: Vec<f64>,
    pub anchor_row: usize,
    pub orientation: i8,
    pub behavior_length: f64,
    pub period: Option<f64>,
    pub nats_per_unit_coordinate: f64,
}

/// JSON-safe per-atom isometry certificate. Non-finite numerical readouts are
/// represented as `None`; engagement/collapse fields retain the reason.
#[derive(Clone, Debug, Serialize)]
pub struct AtomBehaviorIsometryWire {
    pub atom_idx: usize,
    pub n_rows: usize,
    pub support_mass: f64,
    pub behavior_engaged: bool,
    pub behavior_metric_collapse_rows: usize,
    pub activation_speed_rms: Option<f64>,
    pub behavior_speed_rms: Option<f64>,
    pub scale: Option<f64>,
    pub defect_cv: Option<f64>,
    pub min_ratio_over_scale: Option<f64>,
    pub max_ratio_over_scale: Option<f64>,
    pub nats_per_unit_t: Option<f64>,
    pub behavior_pinned_chart: Option<BehaviorPinnedChartWire>,
}

/// Stable report serialized unchanged by bindings.
#[derive(Clone, Debug, Serialize)]
pub struct SaeBehaviorWireReport {
    pub crosscoder: SaeCrosscoderWireReport,
    pub log_lambda_y: f64,
    pub lambda_y: f64,
    pub weight_identifiability: BehaviorWeightIdentifiability,
    pub target_probabilities: Vec<Vec<f64>>,
    pub fitted_probabilities: Vec<Vec<f64>>,
    pub kl: BehaviorKlSummary,
    pub isometry: Vec<Option<AtomBehaviorIsometryWire>>,
}

fn array2_to_nested(array: &Array2<f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn isometry_to_wire(report: &AtomBehaviorIsometry) -> AtomBehaviorIsometryWire {
    AtomBehaviorIsometryWire {
        atom_idx: report.atom_idx,
        n_rows: report.n_rows,
        support_mass: report.support_mass,
        behavior_engaged: report.behavior_engaged,
        behavior_metric_collapse_rows: report.behavior_metric_collapse_rows,
        activation_speed_rms: finite(report.activation_speed_rms),
        behavior_speed_rms: finite(report.behavior_speed_rms),
        scale: finite(report.scale),
        defect_cv: finite(report.defect_cv),
        min_ratio_over_scale: finite(report.min_ratio_over_scale),
        max_ratio_over_scale: finite(report.max_ratio_over_scale),
        nats_per_unit_t: finite(report.nats_per_unit_t),
        behavior_pinned_chart: report.behavior_pinned_chart.as_ref().map(|chart| {
            BehaviorPinnedChartWire {
                coords: chart.coords.to_vec(),
                anchor_row: chart.anchor_row,
                orientation: chart.orientation,
                behavior_length: chart.behavior_length,
                period: chart.period,
                nats_per_unit_coordinate: chart.nats_per_unit_coordinate,
            }
        }),
    }
}

fn residual_rss(target: &Array2<f64>, fitted: &Array2<f64>) -> Result<f64, String> {
    if target.dim() != fitted.dim() {
        return Err(format!(
            "behavior fit residual shape mismatch: target {:?}, fitted {:?}",
            target.dim(),
            fitted.dim()
        ));
    }
    Ok(target
        .iter()
        .zip(fitted.iter())
        .map(|(target, fitted)| {
            let residual = target - fitted;
            residual * residual
        })
        .sum())
}

fn weight_identifiability(
    activation_target: &Array2<f64>,
    activation_fitted: &Array2<f64>,
    behavior_target: &Array2<f64>,
    behavior_fitted: &Array2<f64>,
    log_lambda_y: f64,
) -> Result<BehaviorWeightIdentifiability, String> {
    let rx = residual_rss(activation_target, activation_fitted)?;
    let ry = residual_rss(behavior_target, behavior_fitted)?;
    let n = activation_target.nrows();
    let px = activation_target.ncols();
    let py = behavior_target.ncols();
    let lambda = log_lambda_y.exp();
    let denominator = rx + lambda * ry;
    let curvature = if rx > 0.0 && ry > 0.0 && denominator.is_finite() && denominator > 0.0 {
        0.5 * (n * (px + py)) as f64 * lambda * rx * ry / denominator.powi(2)
    } else {
        0.0
    };
    Ok(BehaviorWeightIdentifiability {
        identifiable: curvature.is_finite() && curvature > 0.0,
        activation_residual_variance: rx / (n * px) as f64,
        behavior_residual_variance: ry / (n * py) as f64,
        log_lambda_curvature: curvature,
    })
}

fn kl_summary(target: &Array2<f64>, fitted: &Array2<f64>) -> Result<BehaviorKlSummary, String> {
    if target.dim() != fitted.dim() {
        return Err(format!(
            "behavior KL shape mismatch: target {:?}, fitted {:?}",
            target.dim(),
            fitted.dim()
        ));
    }
    let mut finite_rows = 0usize;
    let mut infinite_rows = 0usize;
    let mut sum = 0.0_f64;
    let mut max = 0.0_f64;
    for row in 0..target.nrows() {
        let value = SphereTangentEmbedding::exact_kl(target.row(row), fitted.row(row))?;
        if value.is_finite() {
            finite_rows += 1;
            sum += value;
            max = max.max(value);
        } else {
            infinite_rows += 1;
        }
    }
    Ok(BehaviorKlSummary {
        finite_rows,
        infinite_rows,
        mean_kl_nats: (infinite_rows == 0 && finite_rows > 0).then_some(sum / finite_rows as f64),
        max_kl_nats: (infinite_rows == 0 && finite_rows > 0).then_some(max),
    })
}

/// Fit activations and behavioral distributions through one converged outer
/// penalized quasi-Laplace objective and one shared latent/routing state.
pub fn run_auto_sae_behavior_fit(
    request: SaeBehaviorAutoFitRequest,
) -> Result<SaeBehaviorFitReport, SaeFitError> {
    let (n, px) = request.activation.dim();
    if n == 0 || px == 0 {
        return Err(SaeFitError::Fit(format!(
            "run_auto_sae_behavior_fit: activation must be non-empty; got ({n}, {px})"
        )));
    }
    if request.probabilities.nrows() != n {
        return Err(SaeFitError::Fit(format!(
            "run_auto_sae_behavior_fit: probabilities have {} rows; activation has {n}",
            request.probabilities.nrows()
        )));
    }

    // A neutral lambda=1 is only a coordinate origin for the outer optimizer;
    // the converged REML answer is data-selected and returned from rho.
    let initial_behavior = BehaviorBlock::fit(request.probabilities.view(), px, 0.0)?;
    let behavior_target = initial_behavior.target.clone();
    // The fit report no longer retains layer targets (they doubled its resident
    // footprint); keep the two targets here for the identifiability diagnostic.
    let activation_target = request.activation;
    let mut crosscoder = run_auto_sae_crosscoder_fit(SaeCrosscoderAutoFitRequest {
        anchor_label: "activation".to_string(),
        anchor: activation_target.clone(),
        blocks: vec![NamedCrosscoderTarget {
            label: "behavior".to_string(),
            target: behavior_target.clone(),
        }],
        config: request.config,
        cancel: request.cancel,
    })?;
    if crosscoder.layers.len() != 2 || crosscoder.layout.num_blocks() != 1 {
        return Err(SaeFitError::Fit(format!(
            "run_auto_sae_behavior_fit: unified engine returned {} layers and {} non-anchor blocks; expected 2 and 1",
            crosscoder.layers.len(),
            crosscoder.layout.num_blocks()
        )));
    }

    let log_lambda_y = crosscoder.layout.block_log_lambda()[0];
    let behavior_block = initial_behavior.with_log_lambda_y(log_lambda_y)?;
    crosscoder.term.set_behavior_block(behavior_block.clone())?;

    let target_probabilities = behavior_block
        .embedding
        .decode_rows(behavior_block.target.view())?;
    let fitted_probabilities = behavior_block
        .embedding
        .decode_rows(crosscoder.layers[1].fitted.view())?;
    let weight_identifiability = weight_identifiability(
        &activation_target,
        &crosscoder.layers[0].fitted,
        &behavior_target,
        &crosscoder.layers[1].fitted,
        log_lambda_y,
    )?;
    let kl = kl_summary(&target_probabilities, &fitted_probabilities)?;
    let isometry = behavior_isometry_report(&crosscoder.term)?;

    Ok(SaeBehaviorFitReport {
        crosscoder,
        behavior_block,
        target_probabilities,
        fitted_probabilities,
        weight_identifiability,
        kl,
        isometry,
    })
}

impl SaeBehaviorFitReport {
    /// Materialize the stable behavior report shared unchanged by bindings.
    pub fn wire_report(&self) -> Result<SaeBehaviorWireReport, String> {
        let log_lambda_y = self.behavior_block.log_lambda_y();
        Ok(SaeBehaviorWireReport {
            crosscoder: self
                .crosscoder
                .wire_report(SaeCrosscoderEvaluationConfig::default())?,
            log_lambda_y,
            lambda_y: self.behavior_block.lambda_y(),
            weight_identifiability: self.weight_identifiability.clone(),
            target_probabilities: array2_to_nested(&self.target_probabilities),
            fitted_probabilities: array2_to_nested(&self.fitted_probabilities),
            kl: self.kl.clone(),
            isometry: self
                .isometry
                .iter()
                .map(|report| report.as_ref().map(isometry_to_wire))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_two_block_residuals_identify_log_lambda() {
        let activation = ndarray::array![[0.0], [2.0]];
        let activation_fit = ndarray::array![[0.5], [1.5]];
        let behavior = ndarray::array![[0.0], [1.0]];
        let behavior_fit = ndarray::array![[0.25], [0.75]];
        let report =
            weight_identifiability(&activation, &activation_fit, &behavior, &behavior_fit, 0.0)
                .unwrap();
        assert!(report.identifiable);
        assert!(report.log_lambda_curvature > 0.0);
    }

    #[test]
    fn zero_behavior_residual_is_not_identifiable() {
        let activation = ndarray::array![[0.0], [2.0]];
        let activation_fit = ndarray::array![[0.5], [1.5]];
        let behavior = ndarray::array![[0.0], [1.0]];
        let report =
            weight_identifiability(&activation, &activation_fit, &behavior, &behavior, 0.0)
                .unwrap();
        assert!(!report.identifiable);
        assert_eq!(report.log_lambda_curvature, 0.0);
    }
}
