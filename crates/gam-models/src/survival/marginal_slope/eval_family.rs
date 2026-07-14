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
}
