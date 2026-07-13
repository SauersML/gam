//! Exact saved-row replay for survival marginal-slope ALO.
//!
//! The nonlinear baseline time-wiggle means `(q0, q1, qd1, g, ...)` is not an
//! affine chart in the fitted coefficients.  Saved ALO therefore works in the
//! raw reported coefficient chart itself.  This module rehydrates the frozen
//! flex runtimes, invokes the same row program and nonlinear pullback as the
//! optimizer, and returns one exact coefficient score/Hessian per row.

use super::*;
use crate::bms::exact_runtime_from_saved;
use crate::inference::model::{SavedAnchorKind, SavedCompiledFlexBlock};

pub struct SurvivalMarginalSlopeSavedAloReplayInput<'a> {
    pub design_entry: &'a DesignMatrix,
    pub design_exit: &'a DesignMatrix,
    pub design_derivative_exit: &'a DesignMatrix,
    pub offset_entry: &'a Array1<f64>,
    pub offset_exit: &'a Array1<f64>,
    pub derivative_offset_exit: &'a Array1<f64>,
    pub marginal_design: &'a DesignMatrix,
    pub marginal_offset: &'a Array1<f64>,
    pub logslope_design: &'a DesignMatrix,
    pub logslope_offset: &'a Array1<f64>,
    pub latent_z: &'a Array1<f64>,
    pub event: &'a Array1<f64>,
    pub prior_weights: &'a Array1<f64>,
    pub score_variance: f64,
    pub derivative_guard: f64,
    pub time_wiggle_knots: Option<&'a Array1<f64>>,
    pub time_wiggle_degree: Option<usize>,
    pub time_wiggle_ncols: usize,
    pub time_beta: &'a Array1<f64>,
    pub marginal_beta: &'a Array1<f64>,
    pub logslope_beta: &'a Array1<f64>,
    pub score_warp_beta: Option<&'a Array1<f64>>,
    pub link_deviation_beta: Option<&'a Array1<f64>>,
    pub influence_beta: Option<&'a Array1<f64>>,
    pub score_warp_runtime: Option<&'a SavedCompiledFlexBlock>,
    pub link_deviation_runtime: Option<&'a SavedCompiledFlexBlock>,
    pub influence_design: Option<&'a Array2<f64>>,
    pub gaussian_frailty_sd: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeSavedAloRowGeometry {
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
    pub coordinate_values: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeSavedAloReplay {
    pub rows: Vec<SurvivalMarginalSlopeSavedAloRowGeometry>,
    pub block_dimensions: Vec<usize>,
}

fn dense_hstack(designs: &[DesignMatrix], context: &str) -> Result<Array2<f64>, String> {
    DesignMatrix::hstack(designs.to_vec())?
        .try_to_dense_arc(context)
        .map(|matrix| matrix.as_ref().clone())
}

fn flex_anchor_rows(
    saved: &SavedCompiledFlexBlock,
    parametric_rows: &Array2<f64>,
    score_warp: Option<&DeviationRuntime>,
    score_argument: &Array1<f64>,
    label: &str,
) -> Result<Option<Array2<f64>>, String> {
    if saved.anchor_correction.is_none() {
        if !saved.anchor_components.is_empty() {
            return Err(format!(
                "saved survival marginal-slope {label} has anchor components without an anchor correction"
            ));
        }
        return Ok(None);
    }
    let mut parametric_width = 0usize;
    let mut flex_width = None;
    let mut saw_flex = false;
    for (component, saved_component) in saved.anchor_components.iter().enumerate() {
        match &saved_component.kind {
            SavedAnchorKind::Parametric { ncols, .. } if !saw_flex => {
                parametric_width = parametric_width.checked_add(*ncols).ok_or_else(|| {
                    format!("saved survival marginal-slope {label} anchor width overflows usize")
                })?;
            }
            SavedAnchorKind::Parametric { .. } => {
                return Err(format!(
                    "saved survival marginal-slope {label} parametric anchor component {component} follows a flex component"
                ));
            }
            SavedAnchorKind::FlexEvaluation { ncols } if !saw_flex => {
                saw_flex = true;
                flex_width = Some(*ncols);
            }
            SavedAnchorKind::FlexEvaluation { .. } => {
                return Err(format!(
                    "saved survival marginal-slope {label} has more than one flex anchor component"
                ));
            }
        }
    }
    if parametric_width != parametric_rows.ncols() {
        return Err(format!(
            "saved survival marginal-slope {label} parametric anchor width is {parametric_width}; rebuilt fit-time rows have {} columns",
            parametric_rows.ncols(),
        ));
    }
    let Some(flex_width) = flex_width else {
        return Ok(Some(parametric_rows.clone()));
    };
    let score_warp = score_warp.ok_or_else(|| {
        format!(
            "saved survival marginal-slope {label} needs a score-warp flex anchor, but no score-warp runtime exists"
        )
    })?;
    let score_rows = score_warp.design_at_training_with_residual(score_argument)?;
    if score_rows.ncols() != flex_width {
        return Err(format!(
            "saved survival marginal-slope {label} flex anchor width is {flex_width}; score-warp runtime emits {}",
            score_rows.ncols(),
        ));
    }
    let mut rows = Array2::<f64>::zeros((
        parametric_rows.nrows(),
        parametric_rows.ncols() + flex_width,
    ));
    rows.slice_mut(s![.., ..parametric_rows.ncols()])
        .assign(parametric_rows);
    rows.slice_mut(s![.., parametric_rows.ncols()..])
        .assign(&score_rows);
    Ok(Some(rows))
}

fn validate_replay_input(
    input: &SurvivalMarginalSlopeSavedAloReplayInput<'_>,
) -> Result<(), String> {
    let n = input.event.len();
    if n == 0
        || input.prior_weights.len() != n
        || input.latent_z.len() != n
        || input.offset_entry.len() != n
        || input.offset_exit.len() != n
        || input.derivative_offset_exit.len() != n
        || input.marginal_offset.len() != n
        || input.logslope_offset.len() != n
        || input.design_entry.nrows() != n
        || input.design_exit.nrows() != n
        || input.design_derivative_exit.nrows() != n
        || input.marginal_design.nrows() != n
        || input.logslope_design.nrows() != n
    {
        return Err("saved survival marginal-slope ALO row channels are not aligned".to_string());
    }
    for (label, design, beta) in [
        ("entry time", input.design_entry, input.time_beta),
        ("exit time", input.design_exit, input.time_beta),
        (
            "derivative time",
            input.design_derivative_exit,
            input.time_beta,
        ),
        ("marginal", input.marginal_design, input.marginal_beta),
        ("logslope", input.logslope_design, input.logslope_beta),
    ] {
        if design.ncols() != beta.len() {
            return Err(format!(
                "saved survival marginal-slope ALO {label} design/beta mismatch: {}/{}",
                design.ncols(),
                beta.len(),
            ));
        }
    }
    if !input.derivative_guard.is_finite() || input.derivative_guard <= 0.0 {
        return Err(format!(
            "saved survival marginal-slope ALO derivative guard must be positive and finite, got {}",
            input.derivative_guard,
        ));
    }
    if !input.score_variance.is_finite() || input.score_variance < 0.0 {
        return Err(format!(
            "saved survival marginal-slope ALO score variance must be finite and non-negative, got {}",
            input.score_variance,
        ));
    }
    for (row, ((event, weight), z)) in input
        .event
        .iter()
        .zip(input.prior_weights)
        .zip(input.latent_z)
        .enumerate()
    {
        if (*event != 0.0 && *event != 1.0)
            || !weight.is_finite()
            || *weight < 0.0
            || !z.is_finite()
        {
            return Err(format!(
                "saved survival marginal-slope ALO invalid event/weight/z at row {row}: event={event}, weight={weight}, z={z}"
            ));
        }
    }
    let expected_time_wiggle = match (
        input.time_wiggle_knots,
        input.time_wiggle_degree,
        input.time_wiggle_ncols,
    ) {
        (None, None, 0) => 0,
        (Some(knots), Some(degree), ncols) if ncols > 0 => {
            let expected = time_wiggle_basis_ncols(knots, degree)?;
            if expected != ncols {
                return Err(format!(
                    "saved survival marginal-slope ALO timewiggle basis has {expected} columns; fitted tail has {ncols}"
                ));
            }
            ncols
        }
        _ => {
            return Err(
                "saved survival marginal-slope ALO has incomplete timewiggle authority".to_string(),
            );
        }
    };
    if expected_time_wiggle > input.time_beta.len() {
        return Err(
            "saved survival marginal-slope ALO timewiggle tail exceeds time beta".to_string(),
        );
    }
    match (input.influence_beta, input.influence_design) {
        (None, None) => {}
        (Some(beta), Some(design))
            if !beta.is_empty()
                && design.dim() == (n, beta.len())
                && design.iter().all(|value| value.is_finite()) => {}
        _ => {
            return Err(
                "saved survival marginal-slope ALO influence beta/design are inconsistent"
                    .to_string(),
            );
        }
    }
    for (label, runtime, beta) in [
        (
            "score-warp",
            input.score_warp_runtime,
            input.score_warp_beta,
        ),
        (
            "link-deviation",
            input.link_deviation_runtime,
            input.link_deviation_beta,
        ),
    ] {
        match (runtime, beta) {
            (None, None) => {}
            (Some(runtime), Some(beta)) if runtime.basis_dim == beta.len() => {}
            (Some(runtime), Some(beta)) => {
                return Err(format!(
                    "saved survival marginal-slope ALO {label} runtime width is {}; beta has {} entries",
                    runtime.basis_dim,
                    beta.len(),
                ));
            }
            _ => {
                return Err(format!(
                    "saved survival marginal-slope ALO {label} runtime and coefficient block must be present together"
                ));
            }
        }
    }
    Ok(())
}

/// Replay the exact fitted row program and its full raw-coefficient geometry.
pub fn replay_saved_survival_marginal_slope_alo(
    input: SurvivalMarginalSlopeSavedAloReplayInput<'_>,
) -> Result<SurvivalMarginalSlopeSavedAloReplay, String> {
    validate_replay_input(&input)?;
    let n = input.event.len();
    let location_anchor = dense_hstack(
        &[input.design_exit.clone(), input.marginal_design.clone()],
        "saved survival marginal-slope location anchor",
    )?;
    let logslope_dense = input
        .logslope_design
        .try_to_dense_arc("saved survival marginal-slope logslope anchor")?;
    let mut parametric_anchor =
        Array2::<f64>::zeros((n, location_anchor.ncols() + logslope_dense.ncols()));
    parametric_anchor
        .slice_mut(s![.., ..location_anchor.ncols()])
        .assign(&location_anchor);
    parametric_anchor
        .slice_mut(s![.., location_anchor.ncols()..])
        .assign(&logslope_dense.view());

    let score_anchor = input
        .score_warp_runtime
        .map(|saved| {
            flex_anchor_rows(
                saved,
                &parametric_anchor,
                None,
                input.latent_z,
                "score-warp",
            )
        })
        .transpose()?
        .flatten();
    let score_warp = input
        .score_warp_runtime
        .map(|saved| exact_runtime_from_saved(saved, score_anchor.as_ref(), "score-warp"))
        .transpose()?;
    let link_anchor = input
        .link_deviation_runtime
        .map(|saved| {
            flex_anchor_rows(
                saved,
                &parametric_anchor,
                score_warp.as_ref(),
                input.latent_z,
                "link-deviation",
            )
        })
        .transpose()?
        .flatten();
    let link_dev = input
        .link_deviation_runtime
        .map(|saved| exact_runtime_from_saved(saved, link_anchor.as_ref(), "link-deviation"))
        .transpose()?;

    let z_matrix = input.latent_z.clone().insert_axis(Axis(1));
    let score_covariance =
        MarginalSlopeCovariance::diagonal(Array1::from_vec(vec![input.score_variance]))?;
    let logslope_layout = LogslopeTopology::shared()
        .materialize_identity(input.logslope_design.clone(), input.logslope_offset)?;
    let family = SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(input.event.clone()),
        weights: Arc::new(input.prior_weights.clone()),
        z: Arc::new(z_matrix),
        score_covariance,
        gaussian_frailty_sd: input.gaussian_frailty_sd,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
        derivative_guard: input.derivative_guard,
        design_entry: input.design_entry.clone(),
        design_exit: input.design_exit.clone(),
        design_derivative_exit: input.design_derivative_exit.clone(),
        offset_entry: Arc::new(input.offset_entry.clone()),
        offset_exit: Arc::new(input.offset_exit.clone()),
        derivative_offset_exit: Arc::new(input.derivative_offset_exit.clone()),
        marginal_design: input.marginal_design.clone(),
        logslope_layout,
        score_warp,
        link_dev,
        influence_absorber: input.influence_design.cloned(),
        time_linear_constraints: None,
        time_wiggle_knots: input.time_wiggle_knots.cloned(),
        time_wiggle_degree: input.time_wiggle_degree,
        time_wiggle_ncols: input.time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let mut block_states = vec![
        ParameterBlockState {
            beta: input.time_beta.clone(),
            eta: input.design_exit.dot(input.time_beta) + input.offset_exit,
        },
        ParameterBlockState {
            beta: input.marginal_beta.clone(),
            eta: input.marginal_design.dot(input.marginal_beta) + input.marginal_offset,
        },
        ParameterBlockState {
            beta: input.logslope_beta.clone(),
            eta: input.logslope_design.dot(input.logslope_beta) + input.logslope_offset,
        },
    ];
    for beta in [input.score_warp_beta, input.link_deviation_beta]
        .into_iter()
        .flatten()
    {
        block_states.push(ParameterBlockState {
            beta: beta.clone(),
            eta: Array1::zeros(n),
        });
    }
    if let (Some(beta), Some(design)) = (input.influence_beta, input.influence_design) {
        block_states.push(ParameterBlockState {
            beta: beta.clone(),
            eta: design.dot(beta),
        });
    }

    let slices = block_slices(&family, &block_states);
    let primary = flex_primary_slices(&family);
    let identity_blocks = flex_identity_block_pairs(&primary, &slices);
    let flex_active = family.effective_flex_active(&block_states)?;
    let coordinate_values = Array1::from_iter(
        block_states
            .iter()
            .flat_map(|state| state.beta.iter().copied()),
    );
    if coordinate_values.len() != slices.total {
        return Err(format!(
            "saved survival marginal-slope ALO raw coefficient layout has {} entries; family row layout has {}",
            coordinate_values.len(),
            slices.total,
        ));
    }
    let mut rows = Vec::with_capacity(n);
    let mut q_geometry = SurvivalMarginalSlopeDynamicRow::empty_workspace();
    for row in 0..n {
        if input.prior_weights[row] == 0.0 {
            rows.push(SurvivalMarginalSlopeSavedAloRowGeometry {
                nll_score: Array1::zeros(slices.total),
                observed_hessian: Array2::zeros((slices.total, slices.total)),
                coordinate_values: coordinate_values.clone(),
            });
            continue;
        }
        family.row_dynamic_q_geometry_into(row, &block_states, &mut q_geometry)?;
        let (_, primary_score, primary_hessian) = if flex_active {
            family.compute_row_flex_primary_gradient_hessian_exact(
                row,
                &block_states,
                &q_geometry,
                &primary,
            )?
        } else {
            family.compute_row_primary_gradient_hessian_uncached(row, &block_states)?
        };
        let mut log_likelihood_score = Array1::<f64>::zeros(slices.total);
        let mut observed_hessian = Array2::<f64>::zeros((slices.total, slices.total));
        family.accumulate_dynamic_q_joint_row(
            row,
            &slices,
            &q_geometry,
            primary_score.view(),
            primary_hessian.view(),
            &identity_blocks,
            &mut log_likelihood_score,
            &mut observed_hessian,
        )?;
        rows.push(SurvivalMarginalSlopeSavedAloRowGeometry {
            nll_score: -log_likelihood_score,
            observed_hessian,
            coordinate_values: coordinate_values.clone(),
        });
    }
    Ok(SurvivalMarginalSlopeSavedAloReplay {
        rows,
        block_dimensions: block_states.iter().map(|state| state.beta.len()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::DenseDesignMatrix;
    use gam_math::probability::{normal_cdf, normal_pdf};

    fn dense(values: Array2<f64>) -> DesignMatrix {
        DesignMatrix::Dense(DenseDesignMatrix::from(values))
    }

    #[test]
    fn rigid_saved_replay_matches_independent_probit_survival_oracle() {
        let q0 = -0.35_f64;
        let q1 = 0.55_f64;
        let qd = 0.8_f64;
        let g = 0.0_f64;
        let weight = 1.4_f64;
        let event = 1.0_f64;
        let time_beta = Array1::from_vec(vec![q0, q1, qd]);
        let marginal_beta = Array1::zeros(1);
        let logslope_beta = Array1::from_vec(vec![g]);
        let n = 3;
        let time_entry = dense(Array2::from_shape_fn((n, 3), |(_, column)| {
            if column == 0 { 1.0 } else { 0.0 }
        }));
        let time_exit = dense(Array2::from_shape_fn((n, 3), |(_, column)| {
            if column == 1 { 1.0 } else { 0.0 }
        }));
        let time_derivative = dense(Array2::from_shape_fn((n, 3), |(_, column)| {
            if column == 2 { 1.0 } else { 0.0 }
        }));
        let marginal_design = dense(Array2::zeros((n, 1)));
        let logslope_design = dense(Array2::ones((n, 1)));
        let zero = Array1::zeros(n);
        let latent_z = Array1::from_vec(vec![-1.5_f64.sqrt(), 0.0, 1.5_f64.sqrt()]);
        let events = Array1::from_elem(n, event);
        let weights = Array1::from_elem(n, weight);
        let replay =
            replay_saved_survival_marginal_slope_alo(SurvivalMarginalSlopeSavedAloReplayInput {
                design_entry: &time_entry,
                design_exit: &time_exit,
                design_derivative_exit: &time_derivative,
                offset_entry: &zero,
                offset_exit: &zero,
                derivative_offset_exit: &zero,
                marginal_design: &marginal_design,
                marginal_offset: &zero,
                logslope_design: &logslope_design,
                logslope_offset: &zero,
                latent_z: &latent_z,
                event: &events,
                prior_weights: &weights,
                score_variance: 1.0,
                derivative_guard: 1.0e-6,
                time_wiggle_knots: None,
                time_wiggle_degree: None,
                time_wiggle_ncols: 0,
                time_beta: &time_beta,
                marginal_beta: &marginal_beta,
                logslope_beta: &logslope_beta,
                score_warp_beta: None,
                link_deviation_beta: None,
                influence_beta: None,
                score_warp_runtime: None,
                link_deviation_runtime: None,
                influence_design: None,
                gaussian_frailty_sd: None,
            })
            .expect("rigid saved survival row replays");
        let row = &replay.rows[1];
        let mills0 = normal_pdf(-q0) / normal_cdf(-q0);
        let mills1 = normal_pdf(-q1) / normal_cdf(-q1);
        let score0 = -weight * mills0;
        let score1 = weight * ((1.0 - event) * mills1 + event * q1);
        let score_d = -weight * event / qd;
        let h00 = weight * mills0 * (q0 - mills0);
        let h11 = weight * ((1.0 - event) * mills1 * (mills1 - q1) + event);
        let hdd = weight * event / (qd * qd);
        let expected_gg = score0 * q0 + score1 * q1 + score_d * qd;
        for (index, expected) in [score0, score1, score_d, 0.0, 0.0].into_iter().enumerate() {
            assert!((row.nll_score[index] - expected).abs() < 2.0e-12);
        }
        for (index, expected) in [(0, h00), (1, h11), (2, hdd), (4, expected_gg)] {
            assert!((row.observed_hessian[[index, index]] - expected).abs() < 3.0e-12);
        }
        assert!(
            (row.observed_hessian[[0, 0]] - row.nll_score[0].powi(2)).abs() > 1.0e-3,
            "observed W and score-outer-product C are distinct authorities"
        );
    }
}
