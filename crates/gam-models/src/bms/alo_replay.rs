use std::sync::Arc;

use super::deviation_runtime::{AnchorComponentTag, InstalledFlexBlock};
use super::family::{BernoulliMarginalSlopeFamily, bernoulli_marginal_link_map};
use super::gradient_paths::rigid_standard_normal_row_kernel;
use super::hessian_paths::{
    block_slices, new_cell_moment_cache_stats, new_cell_moment_lru_cache, primary_slices,
};
use super::{DeviationRuntime, LatentMeasureKind};
use crate::inference::model::{SavedAnchorKind, SavedCompiledFlexBlock};
use gam_linalg::matrix::DesignMatrix;
use gam_problem::InverseLink;
use gam_problem::ParameterBlockState;
use ndarray::{Array1, Array2};

/// Complete saved-row state for exact rigid Bernoulli marginal-slope ALO replay.
pub struct BernoulliMarginalSlopeAloRowInput<'a> {
    pub base_link: &'a InverseLink,
    pub marginal_eta: f64,
    pub slope: f64,
    pub latent_z: f64,
    pub response: f64,
    pub prior_weight: f64,
    pub probit_frailty_scale: f64,
}

/// Negative-log-likelihood derivatives in the affine fitted coordinates
/// `[marginal eta, slope]` for one saved Bernoulli marginal-slope row.
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliMarginalSlopeAloRowGeometry {
    pub negative_log_likelihood: f64,
    pub nll_score: [f64; 2],
    pub observed_hessian: [[f64; 2]; 2],
}

/// Replay the exact rigid standard-normal row program used by fitting.
///
/// The latent score supplied here must already be in the fitted normalized and
/// calibrated coordinate system. Gaussian-shift frailty is represented by the
/// persisted probit scale, so no prediction-time approximation enters the
/// score or observed Hessian.
pub fn bernoulli_marginal_slope_alo_row_geometry(
    input: BernoulliMarginalSlopeAloRowInput<'_>,
) -> Result<BernoulliMarginalSlopeAloRowGeometry, String> {
    let marginal = bernoulli_marginal_link_map(input.base_link, input.marginal_eta)?;
    let (negative_log_likelihood, nll_score, observed_hessian) = rigid_standard_normal_row_kernel(
        marginal,
        input.slope,
        input.latent_z,
        input.response,
        input.prior_weight,
        input.probit_frailty_scale,
    )?;
    Ok(BernoulliMarginalSlopeAloRowGeometry {
        negative_log_likelihood,
        nll_score,
        observed_hessian,
    })
}

/// Exact saved-row geometry in the full local primary frame
/// `[marginal eta, slope, score-warp coefficients..., link-deviation
/// coefficients...]`.
#[derive(Clone, Debug)]
pub struct BernoulliMarginalSlopeSavedAloRowGeometry {
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
    pub coordinate_values: Array1<f64>,
}

/// Row-aligned replay of the fitted Bernoulli marginal-slope likelihood.
#[derive(Clone, Debug)]
pub struct BernoulliMarginalSlopeSavedAloReplay {
    pub rows: Vec<BernoulliMarginalSlopeSavedAloRowGeometry>,
    pub score_warp_dimension: usize,
    pub link_deviation_dimension: usize,
}

pub(crate) struct BernoulliMarginalSlopeSavedAloReplayInput<'a> {
    pub base_link: &'a InverseLink,
    pub marginal_design: &'a DesignMatrix,
    pub logslope_design: &'a DesignMatrix,
    pub marginal_beta: &'a Array1<f64>,
    pub logslope_beta: &'a Array1<f64>,
    pub score_warp_beta: Option<&'a Array1<f64>>,
    pub link_deviation_beta: Option<&'a Array1<f64>>,
    pub marginal_eta: &'a Array1<f64>,
    pub slope: &'a Array1<f64>,
    pub latent_z: &'a Array1<f64>,
    pub response: &'a Array1<f64>,
    pub prior_weights: &'a Array1<f64>,
    pub latent_measure: LatentMeasureKind,
    pub gaussian_frailty_sd: Option<f64>,
    pub score_warp_runtime: Option<&'a SavedCompiledFlexBlock>,
    pub link_deviation_runtime: Option<&'a SavedCompiledFlexBlock>,
    pub score_warp_anchor_rows: Option<&'a Array2<f64>>,
    pub link_deviation_anchor_rows: Option<&'a Array2<f64>>,
}

fn dense_saved_table(
    rows: &[Vec<f64>],
    n_spans: usize,
    basis_dim: usize,
    label: &str,
) -> Result<Array2<f64>, String> {
    if rows.len() != n_spans || rows.iter().any(|row| row.len() != basis_dim) {
        return Err(format!(
            "saved {label} table is ragged or mis-sized: rows={}, expected={n_spans}, basis_dim={basis_dim}",
            rows.len(),
        ));
    }
    let values = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((n_spans, basis_dim), values)
        .map_err(|error| format!("saved {label} table shape: {error}"))
}

fn dense_anchor_correction(rows: &[Vec<f64>], basis_dim: usize) -> Result<Array2<f64>, String> {
    let nrows = rows.len();
    if rows.iter().any(|row| row.len() != basis_dim) {
        return Err(format!(
            "saved anchor correction is ragged or has a row outside basis dimension {basis_dim}"
        ));
    }
    Array2::from_shape_vec(
        (nrows, basis_dim),
        rows.iter().flat_map(|row| row.iter().copied()).collect(),
    )
    .map_err(|error| format!("saved anchor correction shape: {error}"))
}

fn exact_runtime_from_saved(
    saved: &SavedCompiledFlexBlock,
    anchor_rows: Option<&Array2<f64>>,
    label: &str,
) -> Result<DeviationRuntime, String> {
    saved
        .validate_exact_replay_contract()
        .map_err(|error| format!("{label}: {error}"))?;
    let n_spans = saved.breakpoints.len() - 1;
    let c0 = dense_saved_table(
        &saved.span_c0,
        n_spans,
        saved.basis_dim,
        &format!("{label} c0"),
    )?;
    let c1 = dense_saved_table(
        &saved.span_c1,
        n_spans,
        saved.basis_dim,
        &format!("{label} c1"),
    )?;
    let c2 = dense_saved_table(
        &saved.span_c2,
        n_spans,
        saved.basis_dim,
        &format!("{label} c2"),
    )?;
    let c3 = dense_saved_table(
        &saved.span_c3,
        n_spans,
        saved.basis_dim,
        &format!("{label} c3"),
    )?;
    let installed = match saved.anchor_correction.as_ref() {
        Some(correction) => {
            let anchor_rows = anchor_rows.ok_or_else(|| {
                format!(
                    "saved {label} has a cross-block anchor map but no row-aligned anchor design"
                )
            })?;
            let anchor_components = saved
                .anchor_components
                .iter()
                .map(|component| match &component.kind {
                    SavedAnchorKind::Parametric { block, ncols } => {
                        AnchorComponentTag::Parametric {
                            block: *block,
                            ncols: *ncols,
                        }
                    }
                    SavedAnchorKind::FlexEvaluation { ncols } => {
                        AnchorComponentTag::FlexEvaluation { ncols: *ncols }
                    }
                })
                .collect::<Vec<_>>();
            let expected_anchor_columns = anchor_components
                .iter()
                .map(|component| match component {
                    AnchorComponentTag::Parametric { ncols, .. }
                    | AnchorComponentTag::FlexEvaluation { ncols } => *ncols,
                })
                .sum::<usize>();
            if expected_anchor_columns == 0 {
                return Err(format!("saved {label} anchor map has no anchor components"));
            }
            if anchor_rows.ncols() != expected_anchor_columns {
                return Err(format!(
                    "saved {label} anchor design has {} columns; component layout requires {expected_anchor_columns}",
                    anchor_rows.ncols(),
                ));
            }
            Some(InstalledFlexBlock {
                anchor_correction: dense_anchor_correction(correction, saved.basis_dim)?,
                anchor_components,
            })
        }
        None => {
            if anchor_rows.is_some_and(|rows| rows.ncols() != 0) {
                return Err(format!(
                    "saved {label} received anchor rows without a persisted anchor map"
                ));
            }
            None
        }
    };
    DeviationRuntime::from_exact_cubic_tables(
        Array1::from_vec(saved.breakpoints.clone()),
        c0,
        c1,
        c2,
        c3,
        installed,
        anchor_rows.cloned(),
    )
}

fn validate_optional_flex_block(
    runtime: Option<&SavedCompiledFlexBlock>,
    beta: Option<&Array1<f64>>,
    label: &str,
) -> Result<usize, String> {
    match (runtime, beta) {
        (None, None) => Ok(0),
        (Some(runtime), Some(beta)) if runtime.basis_dim == beta.len() => Ok(beta.len()),
        (Some(runtime), Some(beta)) => Err(format!(
            "saved {label} runtime has basis dimension {}; beta has {} entries",
            runtime.basis_dim,
            beta.len(),
        )),
        (Some(_), None) => Err(format!(
            "saved {label} runtime has no fitted coefficient block"
        )),
        (None, Some(_)) => Err(format!("saved {label} coefficients have no exact runtime")),
    }
}

/// Replay the exact fit-time BMS row program from frozen saved state.
///
/// No basis compilation, fitting, numerical differentiation, or alternate
/// likelihood is permitted here.  The saved cubic tables are rehydrated in
/// their fitted coefficient frame and passed to the same observed-Hessian
/// row authority used by the optimizer.
pub(crate) fn replay_saved_bernoulli_marginal_slope_alo(
    input: BernoulliMarginalSlopeSavedAloReplayInput<'_>,
) -> Result<BernoulliMarginalSlopeSavedAloReplay, String> {
    let n = input.response.len();
    if n == 0
        || input.prior_weights.len() != n
        || input.marginal_design.nrows() != n
        || input.logslope_design.nrows() != n
        || input.marginal_eta.len() != n
        || input.slope.len() != n
        || input.latent_z.len() != n
    {
        return Err(format!(
            "saved BMS ALO row mismatch: response={n}, weights={}, marginal_design={}, logslope_design={}, marginal_eta={}, slope={}, z={}",
            input.prior_weights.len(),
            input.marginal_design.nrows(),
            input.logslope_design.nrows(),
            input.marginal_eta.len(),
            input.slope.len(),
            input.latent_z.len(),
        ));
    }
    if input.marginal_design.ncols() != input.marginal_beta.len()
        || input.logslope_design.ncols() != input.logslope_beta.len()
    {
        return Err(format!(
            "saved BMS ALO affine frame mismatch: marginal design/beta={}/{}, logslope design/beta={}/{}",
            input.marginal_design.ncols(),
            input.marginal_beta.len(),
            input.logslope_design.ncols(),
            input.logslope_beta.len(),
        ));
    }
    if let Some((row, weight)) = input
        .prior_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite() || *weight < 0.0)
    {
        return Err(format!(
            "saved BMS ALO prior weight[{row}] must be finite and non-negative, got {weight}"
        ));
    }
    if let Some((row, response)) = input
        .response
        .iter()
        .copied()
        .enumerate()
        .find(|(_, response)| *response != 0.0 && *response != 1.0)
    {
        return Err(format!(
            "saved BMS ALO response[{row}] must be exactly 0 or 1, got {response}"
        ));
    }
    for (label, values) in [
        ("marginal eta", input.marginal_eta),
        ("slope", input.slope),
        ("latent z", input.latent_z),
        ("marginal beta", input.marginal_beta),
        ("logslope beta", input.logslope_beta),
    ] {
        if let Some((row, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "saved BMS ALO {label}[{row}] must be finite, got {value}"
            ));
        }
    }
    for (label, beta) in [
        ("score-warp", input.score_warp_beta),
        ("link-deviation", input.link_deviation_beta),
    ] {
        if let Some((coordinate, value)) = beta.and_then(|beta| {
            beta.iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
        }) {
            return Err(format!(
                "saved BMS ALO {label} beta[{coordinate}] must be finite, got {value}"
            ));
        }
    }
    for (label, rows) in [
        ("score-warp", input.score_warp_anchor_rows),
        ("link-deviation", input.link_deviation_anchor_rows),
    ] {
        if let Some(rows) = rows
            && rows.nrows() != n
        {
            return Err(format!(
                "saved BMS ALO {label} anchor design has {} rows; expected {n}",
                rows.nrows(),
            ));
        }
    }
    input
        .latent_measure
        .validate("saved BMS ALO latent measure")?;
    let score_warp_dimension = validate_optional_flex_block(
        input.score_warp_runtime,
        input.score_warp_beta,
        "score-warp",
    )?;
    let link_deviation_dimension = validate_optional_flex_block(
        input.link_deviation_runtime,
        input.link_deviation_beta,
        "link-deviation",
    )?;
    let score_warp = input
        .score_warp_runtime
        .map(|runtime| {
            exact_runtime_from_saved(runtime, input.score_warp_anchor_rows, "score-warp")
        })
        .transpose()?;
    let link_dev = input
        .link_deviation_runtime
        .map(|runtime| {
            exact_runtime_from_saved(runtime, input.link_deviation_anchor_rows, "link-deviation")
        })
        .transpose()?;

    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(input.response.clone()),
        weights: Arc::new(input.prior_weights.clone()),
        z: Arc::new(input.latent_z.clone()),
        latent_measure: input.latent_measure,
        gaussian_frailty_sd: input.gaussian_frailty_sd,
        base_link: input.base_link.clone(),
        marginal_design: input.marginal_design.clone(),
        logslope_design: input.logslope_design.clone(),
        score_warp,
        link_dev,
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(std::sync::Mutex::new(None)),
    };
    let slices = block_slices(&family);
    let primary = primary_slices(&slices);
    let mut block_states = vec![
        ParameterBlockState {
            beta: input.marginal_beta.clone(),
            eta: input.marginal_eta.clone(),
        },
        ParameterBlockState {
            beta: input.logslope_beta.clone(),
            eta: input.slope.clone(),
        },
    ];
    if let Some(beta) = input.score_warp_beta {
        block_states.push(ParameterBlockState {
            beta: beta.clone(),
            // The exact row program consumes the fitted flex coefficient
            // vector directly.  Its framework-level block eta is shape-only
            // state and never enters the likelihood geometry.
            eta: Array1::zeros(n),
        });
    }
    if let Some(beta) = input.link_deviation_beta {
        block_states.push(ParameterBlockState {
            beta: beta.clone(),
            eta: Array1::zeros(n),
        });
    }
    family.validate_exact_block_state_shapes(&block_states)?;

    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let row_context = family.build_row_exact_context_with_stats_and_cell_cache(
            row,
            &block_states,
            None,
            false,
        )?;
        let (negative_log_likelihood, nll_score, observed_hessian) = family
            .compute_row_primary_gradient_hessian(row, &block_states, &primary, &row_context)?;
        if nll_score.len() != primary.total
            || observed_hessian.dim() != (primary.total, primary.total)
            || !negative_log_likelihood.is_finite()
            || nll_score.iter().any(|value| !value.is_finite())
            || observed_hessian.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "saved BMS ALO row {row} returned invalid local geometry: nll={negative_log_likelihood}, score={}, hessian={}x{}, expected primary width {}",
                nll_score.len(),
                observed_hessian.nrows(),
                observed_hessian.ncols(),
                primary.total,
            ));
        }
        let mut coordinate_values = Array1::<f64>::zeros(primary.total);
        coordinate_values[primary.q] = input.marginal_eta[row];
        coordinate_values[primary.logslope] = input.slope[row];
        if let (Some(range), Some(beta)) = (primary.h.as_ref(), input.score_warp_beta) {
            coordinate_values
                .slice_mut(ndarray::s![range.clone()])
                .assign(beta);
        }
        if let (Some(range), Some(beta)) = (primary.w.as_ref(), input.link_deviation_beta) {
            coordinate_values
                .slice_mut(ndarray::s![range.clone()])
                .assign(beta);
        }
        rows.push(BernoulliMarginalSlopeSavedAloRowGeometry {
            nll_score,
            observed_hessian,
            coordinate_values,
        });
    }
    Ok(BernoulliMarginalSlopeSavedAloReplay {
        rows,
        score_warp_dimension,
        link_deviation_dimension,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::DenseDesignMatrix;
    use gam_math::probability::{normal_cdf, normal_pdf};
    use gam_problem::StandardLink;

    fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn rigid_saved_alo_geometry_matches_independent_probit_chain_rule() {
        let marginal_eta: f64 = 0.35;
        let slope: f64 = -0.6;
        let latent_z: f64 = 0.8;
        let response: f64 = 1.0;
        let weight: f64 = 1.7;
        let scale: f64 = 0.75;
        let geometry =
            bernoulli_marginal_slope_alo_row_geometry(BernoulliMarginalSlopeAloRowInput {
                base_link: &InverseLink::Standard(StandardLink::Probit),
                marginal_eta,
                slope,
                latent_z,
                response,
                prior_weight: weight,
                probit_frailty_scale: scale,
            })
            .expect("rigid saved marginal-slope row must replay");

        // Independent closed form for eta(q, g) = q sqrt(1 + (s g)^2) + s g z.
        // At an interior probit marginal map q == marginal_eta exactly.
        let sg = scale * slope;
        let c = (1.0 + sg * sg).sqrt();
        let eta = marginal_eta * c + sg * latent_z;
        let sign = 2.0 * response - 1.0;
        let margin = sign * eta;
        let cdf = normal_cdf(margin);
        let mills = normal_pdf(margin) / cdf;
        let nll_first_eta = -weight * sign * mills;
        let nll_second_eta = weight * mills * (margin + mills);

        let eta_q = c;
        let eta_g = marginal_eta * scale * scale * slope / c + scale * latent_z;
        let eta_qg = scale * scale * slope / c;
        let eta_gg = marginal_eta * scale * scale / c.powi(3);
        let expected_score = [nll_first_eta * eta_q, nll_first_eta * eta_g];
        let expected_hessian = [
            [
                nll_second_eta * eta_q * eta_q,
                nll_second_eta * eta_q * eta_g + nll_first_eta * eta_qg,
            ],
            [
                nll_second_eta * eta_q * eta_g + nll_first_eta * eta_qg,
                nll_second_eta * eta_g * eta_g + nll_first_eta * eta_gg,
            ],
        ];

        assert_close(
            "negative log likelihood",
            geometry.negative_log_likelihood,
            -weight * cdf.ln(),
            2e-13,
        );
        for axis in 0..2 {
            assert_close(
                &format!("score[{axis}]"),
                geometry.nll_score[axis],
                expected_score[axis],
                2e-12,
            );
            for other in 0..2 {
                assert_close(
                    &format!("hessian[{axis},{other}]"),
                    geometry.observed_hessian[axis][other],
                    expected_hessian[axis][other],
                    3e-12,
                );
            }
        }

        let score_meat = geometry.nll_score[0] * geometry.nll_score[0];
        assert!(
            (geometry.observed_hessian[0][0] - score_meat).abs() > 1e-3,
            "observed Hessian and empirical score meat must remain distinct"
        );
    }

    fn independent_empirical_score_warp_nll(point: [f64; 3]) -> f64 {
        let [marginal_eta, slope, score_beta] = point;
        let nodes = [-0.8_f64, 0.9_f64];
        let grid_weights = [0.35_f64, 0.65_f64];
        let target = normal_cdf(marginal_eta);
        let calibration = |intercept: f64| {
            nodes
                .iter()
                .zip(grid_weights.iter())
                .map(|(&z, &weight)| weight * normal_cdf(intercept + slope * (z + score_beta * z)))
                .sum::<f64>()
                - target
        };
        let mut lower = -40.0_f64;
        let mut upper = 40.0_f64;
        assert!(calibration(lower) < 0.0 && calibration(upper) > 0.0);
        for _iteration in 0..180 {
            let midpoint = 0.5 * (lower + upper);
            if calibration(midpoint) < 0.0 {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        let intercept = 0.5 * (lower + upper);
        let observed_z = 0.25_f64;
        let observed_eta = intercept + slope * (observed_z + score_beta * observed_z);
        -1.3 * normal_cdf(observed_eta).ln()
    }

    #[test]
    fn empirical_flex_saved_alo_matches_independent_resolved_likelihood_oracle() {
        let marginal_eta = 0.2_f64;
        let slope = -0.35_f64;
        let score_beta = 0.12_f64;
        let score_runtime = SavedCompiledFlexBlock {
            kernel: crate::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
            breakpoints: vec![-2.0, 2.0],
            basis_dim: 1,
            // One frozen local cubic representing h(z)=z on the entire
            // empirical/observed support.  The independent oracle above uses
            // the global expression directly and never calls this runtime.
            span_c0: vec![vec![-2.0]],
            span_c1: vec![vec![1.0]],
            span_c2: vec![vec![0.0]],
            span_c3: vec![vec![0.0]],
            anchor_correction: None,
            anchor_components: Vec::new(),
        };
        let marginal_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::ones((1, 1))));
        let logslope_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::ones((1, 1))));
        let marginal_beta = Array1::from_vec(vec![marginal_eta]);
        let logslope_beta = Array1::from_vec(vec![slope]);
        let score_warp_beta = Array1::from_vec(vec![score_beta]);
        let marginal_rows = Array1::from_vec(vec![marginal_eta]);
        let slope_rows = Array1::from_vec(vec![slope]);
        let latent_z = Array1::from_vec(vec![0.25]);
        let response = Array1::from_vec(vec![1.0]);
        let prior_weights = Array1::from_vec(vec![1.3]);
        let replay =
            replay_saved_bernoulli_marginal_slope_alo(BernoulliMarginalSlopeSavedAloReplayInput {
                base_link: &InverseLink::Standard(StandardLink::Probit),
                marginal_design: &marginal_design,
                logslope_design: &logslope_design,
                marginal_beta: &marginal_beta,
                logslope_beta: &logslope_beta,
                score_warp_beta: Some(&score_warp_beta),
                link_deviation_beta: None,
                marginal_eta: &marginal_rows,
                slope: &slope_rows,
                latent_z: &latent_z,
                response: &response,
                prior_weights: &prior_weights,
                latent_measure: LatentMeasureKind::GlobalEmpirical {
                    grid: super::super::EmpiricalZGrid::new(
                        vec![-0.8, 0.9],
                        vec![0.35, 0.65],
                        "saved ALO empirical-flex oracle",
                    )
                    .expect("valid empirical grid"),
                },
                gaussian_frailty_sd: None,
                score_warp_runtime: Some(&score_runtime),
                link_deviation_runtime: None,
                score_warp_anchor_rows: None,
                link_deviation_anchor_rows: None,
            })
            .expect("saved empirical-flex row must replay");
        assert_eq!(replay.score_warp_dimension, 1);
        assert_eq!(replay.link_deviation_dimension, 0);
        let row = &replay.rows[0];
        assert_eq!(
            row.coordinate_values.to_vec(),
            vec![marginal_eta, slope, score_beta]
        );

        let point = [marginal_eta, slope, score_beta];
        let gradient_step = 2.0e-5_f64;
        for axis in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[axis] += gradient_step;
            minus[axis] -= gradient_step;
            let expected = (independent_empirical_score_warp_nll(plus)
                - independent_empirical_score_warp_nll(minus))
                / (2.0 * gradient_step);
            assert_close(
                &format!("empirical-flex score[{axis}]"),
                row.nll_score[axis],
                expected,
                3.0e-7,
            );
        }

        let hessian_step = 3.0e-4_f64;
        let center = independent_empirical_score_warp_nll(point);
        for first in 0..3 {
            for second in first..3 {
                let expected = if first == second {
                    let mut plus = point;
                    let mut minus = point;
                    plus[first] += hessian_step;
                    minus[first] -= hessian_step;
                    (independent_empirical_score_warp_nll(plus) - 2.0 * center
                        + independent_empirical_score_warp_nll(minus))
                        / hessian_step.powi(2)
                } else {
                    let mut plus_plus = point;
                    let mut plus_minus = point;
                    let mut minus_plus = point;
                    let mut minus_minus = point;
                    plus_plus[first] += hessian_step;
                    plus_plus[second] += hessian_step;
                    plus_minus[first] += hessian_step;
                    plus_minus[second] -= hessian_step;
                    minus_plus[first] -= hessian_step;
                    minus_plus[second] += hessian_step;
                    minus_minus[first] -= hessian_step;
                    minus_minus[second] -= hessian_step;
                    (independent_empirical_score_warp_nll(plus_plus)
                        - independent_empirical_score_warp_nll(plus_minus)
                        - independent_empirical_score_warp_nll(minus_plus)
                        + independent_empirical_score_warp_nll(minus_minus))
                        / (4.0 * hessian_step.powi(2))
                };
                assert_close(
                    &format!("empirical-flex Hessian[{first},{second}]"),
                    row.observed_hessian[[first, second]],
                    expected,
                    4.0e-5,
                );
                assert_close(
                    &format!("empirical-flex symmetry[{second},{first}]"),
                    row.observed_hessian[[second, first]],
                    expected,
                    4.0e-5,
                );
            }
        }
        let score_meat = row.nll_score[0] * row.nll_score[0];
        assert!(
            (row.observed_hessian[[0, 0]] - score_meat).abs() > 1.0e-3,
            "observed Hessian W and empirical score meat C must remain separate"
        );
    }
}
