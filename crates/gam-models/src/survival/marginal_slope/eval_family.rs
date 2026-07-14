//! Family-coordinate derivatives of the canonical survival marginal-slope row
//! programs.
//!
//! Family coordinates are an outer [`Dual2`] direction; coefficient primaries
//! remain the inner fixed-width jet.  One evaluation therefore owns the value,
//! coefficient score, coefficient Hessian, and their family derivative.  The
//! beta-directional variant nests [`OneSeed`] inside the same outer dual, so the
//! Jeffreys/LAML Hessian drift is a derivative of that identical row program.

use super::*;
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2};
use gam_math::nested_dual::Dual2;

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

fn rigid_family_primary_terms(channel: Order2<N_PRIMARY>) -> RigidFamilyPrimaryTerms {
    let gradient = channel.g();
    let hessian = channel.h();
    RigidFamilyPrimaryTerms {
        objective: channel.value(),
        gradient: Array1::from_vec(gradient.to_vec()),
        hessian: Array2::from_shape_fn((N_PRIMARY, N_PRIMARY), |(row, column)| {
            hessian[row][column]
        }),
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

    fn rigid_baseline_primary_first(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
    ) -> Result<[f64; N_PRIMARY], String> {
        if axis >= geometry.theta.len() {
            return Err(format!(
                "survival marginal-slope baseline axis {axis} is out of range for {} coordinates",
                geometry.theta.len(),
            ));
        }
        Ok([
            geometry.offset_entry_theta_first[[row, axis]],
            geometry.offset_exit_theta_first[[row, axis]],
            geometry.derivative_offset_exit_theta_first[[row, axis]],
            0.0,
        ])
    }

    fn rigid_baseline_primary_second(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
        other_axis: usize,
    ) -> Result<[f64; N_PRIMARY], String> {
        if axis >= geometry.theta.len() || other_axis >= geometry.theta.len() {
            return Err(format!(
                "survival marginal-slope baseline pair ({axis}, {other_axis}) is out of range for {} coordinates",
                geometry.theta.len(),
            ));
        }
        Ok([
            geometry.offset_entry_theta_second[[row, axis, other_axis]],
            geometry.offset_exit_theta_second[[row, axis, other_axis]],
            geometry.derivative_offset_exit_theta_second[[row, axis, other_axis]],
            0.0,
        ])
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
        let p_g = slices.logslope.len();
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
            .slice_mut(s![slices.logslope.clone()])
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
    pub(crate) fn rigid_family_direction_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_first: [f64; N_PRIMARY],
        primary_second: [f64; N_PRIMARY],
    ) -> Result<(RigidFamilyPrimaryTerms, RigidFamilyPrimaryTerms), String> {
        if self.flex_active() || self.flex_timewiggle_active() || self.per_z_logslope_active() {
            return Err(
                "rigid family-direction calculus requires scalar/shared non-FLEX, non-time-wiggle geometry"
                    .to_string(),
            );
        }
        let primaries = rigid_row_kernel_primaries(self, block_states, row)?;
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid family-direction row program",
        )?;
        let variables: [Dual2<Order2<N_PRIMARY>>; N_PRIMARY] =
            std::array::from_fn(|axis| Dual2 {
                v: Order2::variable(primaries[axis], axis),
                g: Order2::constant(primary_first[axis]),
                h: Order2::constant(primary_second[axis]),
            });
        let output = rigid_row_nll(&variables, &inputs)?;
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
    pub(crate) fn rigid_family_direction_beta_drift(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_first: [f64; N_PRIMARY],
        primary_beta_direction: [f64; N_PRIMARY],
    ) -> Result<RigidFamilyPrimaryTerms, String> {
        if self.flex_active() || self.flex_timewiggle_active() || self.per_z_logslope_active() {
            return Err(
                "rigid family-direction drift requires scalar/shared non-FLEX, non-time-wiggle geometry"
                    .to_string(),
            );
        }
        let primaries = rigid_row_kernel_primaries(self, block_states, row)?;
        let inputs = rigid_row_inputs(
            self,
            block_states,
            row,
            "survival marginal-slope rigid family-direction drift row program",
        )?;
        let variables: [Dual2<OneSeed<N_PRIMARY>>; N_PRIMARY] =
            std::array::from_fn(|axis| Dual2 {
                v: OneSeed::seed_direction(
                    primaries[axis],
                    axis,
                    primary_beta_direction[axis],
                ),
                g: OneSeed::constant(primary_first[axis]),
                h: OneSeed::constant(0.0),
            });
        let output = rigid_row_nll(&variables, &inputs)?;
        Ok(rigid_family_primary_terms(output.g.eps))
    }

    pub(crate) fn baseline_exact_joint_psi_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        if self.per_z_logslope_active() {
            return Err(
                "survival marginal-slope baseline family derivatives do not support per-score logslope geometry"
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
            self.reduce_rigid_family_primary_terms(block_states, options, |row| {
                let first = Self::rigid_baseline_primary_first(geometry, row, axis)?;
                self.rigid_family_direction_terms(
                    row,
                    block_states,
                    first,
                    [0.0; N_PRIMARY],
                )
                .map(|terms| terms.0)
            })?
        };
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(hessian_psi_operator),
        }))
    }

    pub(crate) fn rigid_baseline_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        other_axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi_operator) = self
            .reduce_rigid_family_primary_terms(block_states, options, |row| {
                let first = Self::rigid_baseline_primary_first(geometry, row, axis)?;
                let other_first =
                    Self::rigid_baseline_primary_first(geometry, row, other_axis)?;
                let second =
                    Self::rigid_baseline_primary_second(geometry, row, axis, other_axis)?;
                if axis == other_axis {
                    return self
                        .rigid_family_direction_terms(row, block_states, first, second)
                        .map(|terms| terms.1);
                }

                let combined_first =
                    std::array::from_fn(|index| first[index] + other_first[index]);
                let twice_cross = std::array::from_fn(|index| 2.0 * second[index]);
                let combined = self
                    .rigid_family_direction_terms(
                        row,
                        block_states,
                        combined_first,
                        twice_cross,
                    )?
                    .1;
                let axis_diagonal = self
                    .rigid_family_direction_terms(
                        row,
                        block_states,
                        first,
                        [0.0; N_PRIMARY],
                    )?
                    .1;
                let other_diagonal = self
                    .rigid_family_direction_terms(
                        row,
                        block_states,
                        other_first,
                        [0.0; N_PRIMARY],
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
            })?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    pub(crate) fn rigid_baseline_exact_joint_psihessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let slices = block_slices(self, block_states);
        let (_, _, operator) = self.reduce_rigid_family_primary_terms(
            block_states,
            options,
            |row| {
                let first = Self::rigid_baseline_primary_first(geometry, row, axis)?;
                let direction = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let primary_direction = std::array::from_fn(|index| direction[index]);
                self.rigid_family_direction_beta_drift(
                    row,
                    block_states,
                    first,
                    primary_direction,
                )
            },
        )?;
        let mut dense = Array2::<f64>::zeros((slices.total, slices.total));
        for column in 0..slices.total {
            let mut basis = Array1::<f64>::zeros(slices.total);
            basis[column] = 1.0;
            dense.column_mut(column).assign(&operator.mul_vec(&basis));
        }
        Ok(Some(dense))
    }
}
