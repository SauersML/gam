//! Family-coordinate derivatives of the canonical survival marginal-slope row
//! programs.
//!
//! Family coordinates are an outer [`Dual2`] direction; coefficient primaries
//! remain the inner fixed-width jet.  One evaluation therefore owns the value,
//! coefficient score, coefficient Hessian, and their family derivative.  The
//! beta-directional variant nests [`OneSeed`] inside the same outer dual, so the
//! Jeffreys/LAML Hessian drift is a derivative of that identical row program.

use super::timepoint_exact::flex_jet::{FlexFamilyCoefficientTerms, FlexFamilyRowDirection};
use super::*;
use crate::row_kernel::RowKernel;
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2};
use gam_math::nested_dual::{Dual2, JetField};

/// One family-direction channel in the rigid four-primary coordinates.
///
/// `objective`, `gradient`, and `hessian` are respectively the family
/// derivative of the row objective, its primary score, and its primary
/// Hessian.  Callers pull these through the row's canonical coefficient map;
/// no likelihood formula is reconstructed outside the row program.
pub(crate) struct RigidFamilyPrimaryTerms {
    pub(crate) objective: f64,
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Array2<f64>,
}

fn combine_rigid_family_primary_terms(
    left: &RigidFamilyPrimaryTerms,
    left_scale: f64,
    middle: &RigidFamilyPrimaryTerms,
    middle_scale: f64,
    right: &RigidFamilyPrimaryTerms,
    right_scale: f64,
) -> RigidFamilyPrimaryTerms {
    RigidFamilyPrimaryTerms {
        objective: left_scale * left.objective
            + middle_scale * middle.objective
            + right_scale * right.objective,
        gradient: left_scale * &left.gradient
            + middle_scale * &middle.gradient
            + right_scale * &right.gradient,
        hessian: left_scale * &left.hessian
            + middle_scale * &middle.hessian
            + right_scale * &right.hessian,
    }
}

fn combine_flex_family_coefficient_terms(
    left: &FlexFamilyCoefficientTerms,
    left_scale: f64,
    middle: &FlexFamilyCoefficientTerms,
    middle_scale: f64,
    right: &FlexFamilyCoefficientTerms,
    right_scale: f64,
) -> FlexFamilyCoefficientTerms {
    FlexFamilyCoefficientTerms {
        objective: left_scale * left.objective
            + middle_scale * middle.objective
            + right_scale * right.objective,
        gradient: left_scale * &left.gradient
            + middle_scale * &middle.gradient
            + right_scale * &right.gradient,
        hessian: left_scale * &left.hessian
            + middle_scale * &middle.hessian
            + right_scale * &right.hessian,
    }
}

fn rigid_family_primary_terms<const P: usize>(channel: Order2<P>) -> RigidFamilyPrimaryTerms {
    let gradient = channel.g();
    let hessian = channel.h();
    RigidFamilyPrimaryTerms {
        objective: channel.value(),
        gradient: Array1::from_vec(gradient.to_vec()),
        hessian: Array2::from_shape_fn((P, P), |(row, column)| hessian[row][column]),
    }
}

impl SurvivalMarginalSlopeFamily {
    fn rigid_baseline_geometry(
        &self,
    ) -> Result<&crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry, String> {
        self.family_hyper.baseline_geometry.as_deref().ok_or_else(|| {
            "survival marginal-slope baseline family derivative requested without frozen baseline geometry"
                .to_string()
        })
    }

    /// The baseline chart moves the OFFSET channels of the location index only.
    /// It cannot move the slope, so every slope primary of the frame is exactly
    /// zero here whichever frame is in play.
    fn rigid_baseline_primary_first<const P: usize>(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
    ) -> Result<[f64; P], String> {
        if axis >= geometry.theta.len() {
            return Err(format!(
                "survival marginal-slope baseline axis {axis} is out of range for {} coordinates",
                geometry.theta.len(),
            ));
        }
        let mut direction = [0.0; P];
        direction[PRIMARY_Q0] = geometry.offset_entry_theta_first[[row, axis]];
        direction[PRIMARY_Q1] = geometry.offset_exit_theta_first[[row, axis]];
        direction[PRIMARY_QD1] = geometry.derivative_offset_exit_theta_first[[row, axis]];
        Ok(direction)
    }

    fn rigid_baseline_primary_second<const P: usize>(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
        other_axis: usize,
    ) -> Result<[f64; P], String> {
        if axis >= geometry.theta.len() || other_axis >= geometry.theta.len() {
            return Err(format!(
                "survival marginal-slope baseline pair ({axis}, {other_axis}) is out of range for {} coordinates",
                geometry.theta.len(),
            ));
        }
        let mut direction = [0.0; P];
        direction[PRIMARY_Q0] = geometry.offset_entry_theta_second[[row, axis, other_axis]];
        direction[PRIMARY_Q1] = geometry.offset_exit_theta_second[[row, axis, other_axis]];
        direction[PRIMARY_QD1] =
            geometry.derivative_offset_exit_theta_second[[row, axis, other_axis]];
        Ok(direction)
    }

    fn flex_baseline_first(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
    ) -> Result<FlexFamilyRowDirection, String> {
        let [entry, exit, derivative_exit, _] =
            Self::rigid_baseline_primary_first(geometry, row, axis)?;
        Ok(FlexFamilyRowDirection {
            entry,
            exit,
            derivative_exit,
            probit_scale: 0.0,
        })
    }

    fn flex_baseline_second(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
        other_axis: usize,
    ) -> Result<FlexFamilyRowDirection, String> {
        let [entry, exit, derivative_exit, _] =
            Self::rigid_baseline_primary_second(geometry, row, axis, other_axis)?;
        Ok(FlexFamilyRowDirection {
            entry,
            exit,
            derivative_exit,
            probit_scale: 0.0,
        })
    }

    fn reduce_rigid_family_primary_terms<F>(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
        row_terms: F,
    ) -> Result<(f64, Array1<f64>, Arc<dyn HyperOperator>), String>
    where
        F: Fn(usize) -> Result<RigidFamilyPrimaryTerms, String> + Sync,
    {
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.slope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let p_i = slices.influence.as_ref().map_or(0, |range| range.len());
        let rows = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let (objective, score_t, score_m, score_g, score_h, score_w, accumulator) =
            chunked_row_reduction(
                rows.as_slice(),
                || {
                    (
                        0.0,
                        Array1::zeros(p_t),
                        Array1::zeros(p_m),
                        Array1::zeros(p_g),
                        Array1::zeros(p_h),
                        Array1::zeros(p_w),
                        BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, p_i),
                    )
                },
                |row, accumulated| -> Result<(), String> {
                    let mut terms = row_terms(row)?;
                    let weight = row_weights[row];
                    if weight != 1.0 {
                        terms.objective *= weight;
                        terms.gradient.mapv_inplace(|value| value * weight);
                        terms.hessian.mapv_inplace(|value| value * weight);
                    }
                    accumulated.0 += terms.objective;
                    let q_geometry = self.row_dynamic_q_geometry(row, block_states)?;
                    self.accumulate_score_with_q_geometry(
                        row,
                        &q_geometry,
                        &terms.gradient,
                        &mut accumulated.1,
                        &mut accumulated.2,
                        &mut accumulated.3,
                    )?;
                    accumulated.6.add_pullback_with_q_geometry(
                        self,
                        row,
                        &q_geometry,
                        &terms.gradient,
                        &terms.hessian,
                    )?;
                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6.add(&chunk.6);
                },
            )?;

        let mut score = Array1::zeros(slices.total);
        score.slice_mut(s![slices.time.clone()]).assign(&score_t);
        score
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score
            .slice_mut(s![slices.slope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score.slice_mut(s![range.clone()]).assign(&score_w);
        }
        Ok((
            objective,
            score,
            Arc::new(accumulator.into_operator(slices)),
        ))
    }

    /// Sum row terms which already live in the canonical flattened coefficient
    /// coordinates.  Unlike the rigid reducer, this must not apply a second
    /// q/design pullback: the nested FLEX row program owns nonlinear
    /// time-wiggle composition and every coefficient-map derivative itself.
    fn reduce_flex_family_coefficient_terms<F>(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
        row_terms: F,
    ) -> Result<(f64, Array1<f64>, Arc<dyn HyperOperator>), String>
    where
        F: Fn(usize) -> Result<FlexFamilyCoefficientTerms, String> + Sync,
    {
        let dimension = block_slices(self, block_states).total;
        let rows = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let (objective, score, hessian) = chunked_row_reduction(
            rows.as_slice(),
            || {
                (
                    0.0,
                    Array1::zeros(dimension),
                    Array2::zeros((dimension, dimension)),
                )
            },
            |row, accumulated| -> Result<(), String> {
                let mut terms = row_terms(row)?;
                if terms.gradient.len() != dimension
                    || terms.hessian.dim() != (dimension, dimension)
                {
                    return Err(format!(
                        "FLEX family row {row} returned coefficient shape gradient={}, Hessian={:?}, expected {dimension} and ({dimension}, {dimension})",
                        terms.gradient.len(),
                        terms.hessian.dim(),
                    ));
                }
                if !terms.objective.is_finite()
                    || terms.gradient.iter().any(|value| !value.is_finite())
                    || terms.hessian.iter().any(|value| !value.is_finite())
                {
                    return Err(format!(
                        "FLEX family row {row} returned non-finite flattened coefficient terms"
                    ));
                }
                let weight = row_weights[row];
                if weight != 1.0 {
                    terms.objective *= weight;
                    terms.gradient.mapv_inplace(|value| value * weight);
                    terms.hessian.mapv_inplace(|value| value * weight);
                }
                accumulated.0 += terms.objective;
                accumulated.1 += &terms.gradient;
                accumulated.2 += &terms.hessian;
                Ok(())
            },
            |total, chunk| {
                total.0 += chunk.0;
                total.1 += &chunk.1;
                total.2 += &chunk.2;
            },
        )?;
        Ok((
            objective,
            score,
            Arc::new(gam_problem::DenseMatrixHyperOperator { matrix: hessian }),
        ))
    }

    /// Exact first and same-direction second family derivatives of one rigid
    /// row, including complete primary value/gradient/Hessian channels.
    ///
    /// `primary_first` and `primary_second` are the first and second motion of
    /// `(q0,q1,qd1,g)` along one declared family direction.  Baseline axes move
    /// the first three entries; learned frailty is represented inside the row
    /// program by its own scalar and therefore does not call this baseline
    /// helper.  This route is deliberately restricted to the scalar/shared,
    /// non-time-wiggle stratum whose coefficient map is affine.  FLEX,
    /// time-wiggle, and per-score rows use the runtime-width nested-dual row
    /// program rather than pretending this four-primary map still applies.
    pub(crate) fn rigid_family_direction_terms<const P: usize, G: SlopeRowGeometry<P>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_first: [f64; P],
        primary_second: [f64; P],
    ) -> Result<(RigidFamilyPrimaryTerms, RigidFamilyPrimaryTerms), String> {
        if self.flex_active() || self.flex_timewiggle_active() || self.per_z_slope_active() {
            return Err(
                "rigid family-direction calculus requires scalar/shared non-FLEX, non-time-wiggle geometry"
                    .to_string(),
            );
        }
        let primaries = rigid_row_kernel_primaries::<P, G>(self, block_states, row)?;
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid family-direction row program",
        )?;
        let variables: [Dual2<Order2<P>>; P] = std::array::from_fn(|axis| Dual2 {
            v: Order2::variable(primaries[axis], axis),
            g: Order2::constant(primary_first[axis]),
            h: Order2::constant(primary_second[axis]),
        });
        let output = rigid_row_nll::<P, G, _>(&variables, &inputs)?;
        Ok((
            rigid_family_primary_terms(output.g),
            rigid_family_primary_terms(output.h),
        ))
    }

    /// Directional beta drift of the rigid family-Hessian channel.
    ///
    /// The outer `Dual2::g` is the selected family derivative.  The inner
    /// `OneSeed::eps` is one arbitrary primary/beta direction, so
    /// `output.g.eps` carries the exact directional derivative of the family
    /// objective, score, and Hessian without materialising a fourth-order
    /// tensor or differencing neighbouring fits.
    pub(crate) fn rigid_family_direction_beta_drift<const P: usize, G: SlopeRowGeometry<P>>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_first: [f64; P],
        primary_beta_direction: [f64; P],
    ) -> Result<RigidFamilyPrimaryTerms, String> {
        if self.flex_active() || self.flex_timewiggle_active() || self.per_z_slope_active() {
            return Err(
                "rigid family-direction drift requires scalar/shared non-FLEX, non-time-wiggle geometry"
                    .to_string(),
            );
        }
        let primaries = rigid_row_kernel_primaries::<P, G>(self, block_states, row)?;
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid family-direction drift row program",
        )?;
        let variables: [Dual2<OneSeed<P>>; P] = std::array::from_fn(|axis| Dual2 {
            v: OneSeed::seed_direction(primaries[axis], axis, primary_beta_direction[axis]),
            g: OneSeed::constant(primary_first[axis]),
            h: OneSeed::constant(0.0),
        });
        let output = rigid_row_nll::<P, G, _>(&variables, &inputs)?;
        Ok(rigid_family_primary_terms(output.g.eps))
    }

    pub(crate) fn baseline_exact_joint_psi_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        if self.per_z_slope_active() {
            return Err(
                "survival marginal-slope baseline family derivatives do not support per-score slope geometry"
                    .to_string(),
            );
        }
        let use_flex = self.effective_flex_active(block_states)? || self.flex_timewiggle_active();
        let (objective_psi, score_psi, hessian_psi_operator) = if use_flex {
            self.reduce_flex_family_coefficient_terms(block_states, options, |row| {
                let first = Self::flex_baseline_first(geometry, row, axis)?;
                self.flex_family_direction_row_terms(
                    row,
                    block_states,
                    first,
                    FlexFamilyRowDirection::default(),
                    None,
                )
                .map(|terms| terms.first)
            })?
        } else {
            in_slope_frame!(self, P, Frame, {
                self.reduce_rigid_family_primary_terms(block_states, options, |row| {
                    let first = Self::rigid_baseline_primary_first::<P>(geometry, row, axis)?;
                    self.rigid_family_direction_terms::<P, Frame>(
                        row,
                        block_states,
                        first,
                        [0.0; P],
                    )
                    .map(|terms| terms.0)
                })
            })?
        };
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(hessian_psi_operator),
        }))
    }

    pub(crate) fn baseline_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        other_axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        if self.per_z_slope_active() {
            return Err(
                "survival marginal-slope baseline family pairs do not support per-score slope geometry"
                    .to_string(),
            );
        }
        let use_flex = self.effective_flex_active(block_states)? || self.flex_timewiggle_active();
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi_operator) = if use_flex {
            self.reduce_flex_family_coefficient_terms(block_states, options, |row| {
                let first = Self::flex_baseline_first(geometry, row, axis)?;
                let other_first = Self::flex_baseline_first(geometry, row, other_axis)?;
                let second = Self::flex_baseline_second(geometry, row, axis, other_axis)?;
                if axis == other_axis {
                    return self
                        .flex_family_direction_row_terms(row, block_states, first, second, None)
                        .map(|terms| terms.second);
                }

                let combined_first = FlexFamilyRowDirection {
                    entry: first.entry + other_first.entry,
                    exit: first.exit + other_first.exit,
                    derivative_exit: first.derivative_exit + other_first.derivative_exit,
                    probit_scale: first.probit_scale + other_first.probit_scale,
                };
                let twice_cross = FlexFamilyRowDirection {
                    entry: 2.0 * second.entry,
                    exit: 2.0 * second.exit,
                    derivative_exit: 2.0 * second.derivative_exit,
                    probit_scale: 2.0 * second.probit_scale,
                };
                let combined = self
                    .flex_family_direction_row_terms(
                        row,
                        block_states,
                        combined_first,
                        twice_cross,
                        None,
                    )?
                    .second;
                let axis_diagonal = self
                    .flex_family_direction_row_terms(
                        row,
                        block_states,
                        first,
                        FlexFamilyRowDirection::default(),
                        None,
                    )?
                    .second;
                let other_diagonal = self
                    .flex_family_direction_row_terms(
                        row,
                        block_states,
                        other_first,
                        FlexFamilyRowDirection::default(),
                        None,
                    )?
                    .second;
                Ok(combine_flex_family_coefficient_terms(
                    &combined,
                    0.5,
                    &axis_diagonal,
                    -0.5,
                    &other_diagonal,
                    -0.5,
                ))
            })?
        } else {
            in_slope_frame!(self, P, Frame, {
                self.reduce_rigid_family_primary_terms(block_states, options, |row| {
                    let first = Self::rigid_baseline_primary_first::<P>(geometry, row, axis)?;
                    let other_first =
                        Self::rigid_baseline_primary_first::<P>(geometry, row, other_axis)?;
                    let second =
                        Self::rigid_baseline_primary_second::<P>(geometry, row, axis, other_axis)?;
                    if axis == other_axis {
                        return self
                            .rigid_family_direction_terms::<P, Frame>(
                                row,
                                block_states,
                                first,
                                second,
                            )
                            .map(|terms| terms.1);
                    }

                    let combined_first: [f64; P] =
                        std::array::from_fn(|index| first[index] + other_first[index]);
                    let twice_cross: [f64; P] =
                        std::array::from_fn(|index| 2.0 * second[index]);
                    let combined = self
                        .rigid_family_direction_terms::<P, Frame>(
                            row,
                            block_states,
                            combined_first,
                            twice_cross,
                        )?
                        .1;
                    let axis_diagonal = self
                        .rigid_family_direction_terms::<P, Frame>(
                            row,
                            block_states,
                            first,
                            [0.0; P],
                        )?
                        .1;
                    let other_diagonal = self
                        .rigid_family_direction_terms::<P, Frame>(
                            row,
                            block_states,
                            other_first,
                            [0.0; P],
                        )?
                        .1;
                    Ok(combine_rigid_family_primary_terms(
                        &combined,
                        0.5,
                        &axis_diagonal,
                        -0.5,
                        &other_diagonal,
                        -0.5,
                    ))
                })
            })?
        };
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    /// Exact baseline-family × design-hyper pair at fixed coefficients.
    ///
    /// The FLEX Jet3 direction receives the actual `X_psi` row.  Its value
    /// seed is `X_psi beta` and its inner gradient seed is `X_psi`, so this
    /// one row program differentiates both the predictor and the
    /// coefficient-space pullback.  Treating `first.gradient` as this mixed
    /// pair would omit the latter and is therefore forbidden.
    pub(crate) fn baseline_design_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        baseline_axis: usize,
        design_psi_index: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        if self.per_z_slope_active() {
            return Err(
                "survival marginal-slope baseline-by-design calculus does not support per-score slope geometry"
                    .to_string(),
            );
        }
        let use_flex = self.effective_flex_active(block_states)? || self.flex_timewiggle_active();
        if !use_flex {
            return Err(
                "survival marginal-slope rigid baseline-by-design calculus is not installed; refusing to substitute FLEX or a zero mixed pair"
                    .to_string(),
            );
        }
        let Some((block, local_index, coefficient_width, label)) =
            self.psi_block_info(derivative_blocks, design_psi_index)?
        else {
            return Err(format!(
                "survival marginal-slope design hyper axis {design_psi_index} has no derivative block"
            ));
        };
        let derivative = &derivative_blocks[block][local_index];
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            derivative,
            self.n,
            coefficient_width,
            0..self.n,
            label,
            &policy,
        ).map_err(|error| error.to_string())?;
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi_operator) = self
            .reduce_flex_family_coefficient_terms(block_states, options, |row| {
                let derivative_row = psi_map
                    .row_vector(row)
                    .map_err(|error| format!("survival family-by-design psi row: {error}"))?;
                let first = Self::flex_baseline_first(geometry, row, baseline_axis)?;
                self.flex_family_design_direction_row_terms(
                    row,
                    block_states,
                    first,
                    FlexFamilyRowDirection::default(),
                    block,
                    &derivative_row,
                )?
                .directional
                .ok_or_else(|| {
                    "FLEX family design-direction row did not return its Jet3 channel".to_string()
                })
            })?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    pub(crate) fn baseline_exact_joint_psihessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        if self.per_z_slope_active() {
            return Err(
                "survival marginal-slope baseline family Hessian drift does not support per-score slope geometry"
                    .to_string(),
            );
        }
        let slices = block_slices(self, block_states);
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "survival marginal-slope baseline family beta direction length {} != flattened coefficient width {}",
                d_beta_flat.len(),
                slices.total,
            ));
        }
        let use_flex = self.effective_flex_active(block_states)? || self.flex_timewiggle_active();
        let (_, _, operator) = if use_flex {
            self.reduce_flex_family_coefficient_terms(block_states, options, |row| {
                let first = Self::flex_baseline_first(geometry, row, axis)?;
                self.flex_family_direction_row_terms(
                    row,
                    block_states,
                    first,
                    FlexFamilyRowDirection::default(),
                    Some(d_beta_flat),
                )?
                .directional
                .ok_or_else(|| {
                    "FLEX family beta-direction row did not return its Jet3 channel".to_string()
                })
            })?
        } else {
            in_slope_frame!(self, P, Frame, {
                self.reduce_rigid_family_primary_terms(block_states, options, |row| {
                    let first = Self::rigid_baseline_primary_first::<P>(geometry, row, axis)?;
                    let direction = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_beta_flat,
                    )?;
                    let primary_direction: [f64; P] =
                        std::array::from_fn(|index| direction[index]);
                    self.rigid_family_direction_beta_drift::<P, Frame>(
                        row,
                        block_states,
                        first,
                        primary_direction,
                    )
                })
            })?
        };
        Ok(Some(operator.to_dense()))
    }

    /// The outer row measure as one weight per row: a retained row carries its
    /// Horvitz–Thompson weight and a row the measure leaves out carries zero.
    fn rigid_third_row_weights(&self, options: &BlockwiseFitOptions) -> Vec<f64> {
        let mut weights = vec![0.0; self.n];
        for row in crate::marginal_slope_shared::outer_weighted_rows(options, self.n) {
            weights[row.index] = row.weight;
        }
        weights
    }

    /// Closed-form fifth likelihood derivatives exist for the rigid shared-slope
    /// row program only.
    fn require_rigid_third(
        &self,
        block_states: &[ParameterBlockState],
        context: &str,
    ) -> Result<(), String> {
        if self.per_z_slope_active()
            || self.effective_flex_active(block_states)?
            || self.flex_timewiggle_active()
        {
            return Err(format!(
                "survival marginal-slope {context} has closed-form fifth likelihood derivatives on the rigid shared-slope row program only; FLEX, time-wiggle and per-score slopes have none"
            ));
        }
        Ok(())
    }

    /// `{D_β_a D_β ∂_θ H[v]}` along every coefficient axis `a`: the mixed third
    /// information derivative of a baseline-chart coordinate `θ` that the explicit
    /// Jeffreys curvature reads along `(θ, v)` (gam#2765).
    ///
    /// The chart moves the offsets `o` of the location index and leaves the
    /// coefficient map `J` fixed, so a row contributes `Jᵀ T⁵[o, Jv, J e_a] J`.
    pub(crate) fn baseline_exact_joint_psihessian_second_directional_derivative_all_beta_axes_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Vec<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        self.require_rigid_third(
            block_states,
            "baseline-by-coefficient third information derivative",
        )?;
        let d_beta = self.finite_flat_direction(block_states, d_beta_flat)?;
        let row_weights = self.rigid_third_row_weights(options);
        in_slope_frame!(self, P, Frame, {
            let kernel = SurvivalMarginalSlopeRowKernel::<P, Frame>::new(
                self.clone(),
                block_states.to_vec(),
            );
            kernel.primary_third_information_all_axes(&row_weights, |row| {
                Ok((
                    Self::rigid_baseline_primary_first::<P>(geometry, row, axis)?,
                    kernel.jacobian_action(row, d_beta),
                    None,
                ))
            })
        })
    }

    /// `{D_β_a ∂²_θθ' H}` along every coefficient axis `a` for a pair of
    /// baseline-chart coordinates (gam#2765). A row contributes
    /// `Jᵀ(T⁵[o, o', J e_a] + T⁴[o_θθ', J e_a])J`, where `o_θθ'` is the chart's
    /// second motion of the offsets.
    pub(crate) fn baseline_exact_joint_psisecond_order_hessian_directional_derivative_all_beta_axes_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        other_axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Vec<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        self.require_rigid_third(block_states, "baseline-pair third information derivative")?;
        let row_weights = self.rigid_third_row_weights(options);
        in_slope_frame!(self, P, Frame, {
            let kernel = SurvivalMarginalSlopeRowKernel::<P, Frame>::new(
                self.clone(),
                block_states.to_vec(),
            );
            kernel.primary_third_information_all_axes(&row_weights, |row| {
                Ok((
                    Self::rigid_baseline_primary_first::<P>(geometry, row, axis)?,
                    Self::rigid_baseline_primary_first::<P>(geometry, row, other_axis)?,
                    Some(Self::rigid_baseline_primary_second::<P>(
                        geometry, row, axis, other_axis,
                    )?),
                ))
            })
        })
    }

    /// `{D_β_a D_β ∂_ψ H[v]}` along every coefficient axis `a` for a design
    /// hyperparameter ψ (gam#2765); see
    /// `SurvivalMarginalSlopeRowKernel::design_psi_third_information_all_axes_from`.
    pub(crate) fn design_psi_hessian_second_directional_derivative_all_beta_axes_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.require_rigid_third(
            block_states,
            "design-by-coefficient third information derivative",
        )?;
        let d_beta = self.finite_flat_direction(block_states, d_beta_flat)?;
        let row_weights = self.rigid_third_row_weights(options);
        in_slope_frame!(self, P, Frame, {
            SurvivalMarginalSlopeRowKernel::<P, Frame>::new(self.clone(), block_states.to_vec())
                .design_psi_third_information_all_axes(
                    derivative_blocks,
                    psi_index,
                    d_beta,
                    &row_weights,
                )
        })
    }

    /// A flat coefficient direction as a finite contiguous slice of the joint width.
    fn finite_flat_direction<'direction>(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &'direction Array1<f64>,
    ) -> Result<&'direction [f64], String> {
        let total = block_slices(self, block_states).total;
        d_beta_flat
            .as_slice()
            .filter(|direction| {
                direction.len() == total && direction.iter().all(|value| value.is_finite())
            })
            .ok_or_else(|| {
                format!(
                    "survival marginal-slope third information derivative needs a finite contiguous coefficient direction of length {total}"
                )
            })
    }
}
