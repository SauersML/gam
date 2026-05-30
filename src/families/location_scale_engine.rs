//! Generic exact-joint location-scale machinery shared by the multi-block
//! families (GAMLSS two-block mean/noise, survival location-scale
//! threshold/log-sigma, …).
//!
//! Every location-scale family with several linear predictors needs the same
//! structural setup for the exact-joint spatial optimizer: concatenate the
//! per-block anisotropic `log κ` seeds, lower/upper data-aware bounds, project
//! the seed onto those bounds, and assemble the [`ExactJointHyperSetup`] over
//! `theta = [rho, psi]`. Only the row likelihood and the meaning of each block
//! differ across families — the κ-coordinate assembly does not. This module is
//! the single home for that assembly so improvements to it land once.

use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionSpec, spatial_length_scale_term_indices,
};
use ndarray::{Array1, ArrayView2};

/// Bound on every `rho` (log smoothing / log dispersion) coordinate in the
/// exact-joint theta vector. Shared by all location-scale families.
pub(crate) const EXACT_JOINT_RHO_BOUND: f64 = 12.0;

/// Assemble the exact-joint hyperparameter setup for a location-scale family
/// whose linear predictors are described, in theta order, by `blocks`.
///
/// `blocks` lists the per-predictor [`TermCollectionSpec`]s (e.g.
/// `[meanspec, noisespec]` for GAMLSS, `[thresholdspec, log_sigmaspec]` for
/// survival location-scale). The spatial `log κ` seed and its data-aware
/// lower/upper bounds are built per block — using `kappa_options` and the
/// term indices flagged for spatial length-scale optimization — and
/// concatenated in block order, matching the layout the exact-joint optimizer
/// expects.
///
/// `rho0` carries the caller-assembled smoothing/dispersion seed (already
/// ordered to match the penalty layout); its `[-EXACT_JOINT_RHO_BOUND,
/// EXACT_JOINT_RHO_BOUND]` box bounds are supplied here so the seed assembly
/// and bounding live in one place.
pub(crate) fn build_location_scale_exact_joint_setup(
    data: ArrayView2<'_, f64>,
    blocks: &[&TermCollectionSpec],
    rho0: Array1<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let rho_lower = Array1::<f64>::from_elem(rho0.len(), -EXACT_JOINT_RHO_BOUND);
    let rho_upper = Array1::<f64>::from_elem(rho0.len(), EXACT_JOINT_RHO_BOUND);

    // Concatenate per-block anisotropic log(kappa) seeds and their dims in
    // block order. The exact-joint setup stores the spatial tail in log(kappa),
    // not log(length_scale); each aniso term contributes d psi entries.
    let mut all_values = Vec::new();
    let mut all_dims = Vec::new();
    let mut lower_vals = Vec::new();
    let mut upper_vals = Vec::new();

    for spec in blocks {
        let term_indices = spatial_length_scale_term_indices(spec);

        // Re-seed psi from data geometry when the spec does not pin a
        // length_scale.
        let kappa =
            SpatialLogKappaCoords::from_length_scales_aniso(spec, &term_indices, kappa_options)
                .reseed_from_data(data, spec, &term_indices, kappa_options);
        let dims = kappa.dims_per_term().to_vec();

        let lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            data,
            spec,
            &term_indices,
            &dims,
            kappa_options,
        );
        let upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            data,
            spec,
            &term_indices,
            &dims,
            kappa_options,
        );

        all_values.extend(kappa.as_array().iter());
        lower_vals.extend(lower.as_array().iter());
        upper_vals.extend(upper.as_array().iter());
        all_dims.extend(dims);
    }

    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(all_values), all_dims.clone());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), all_dims.clone());
    let log_kappa_upper =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), all_dims);
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);

    ExactJointHyperSetup::new(
        rho0,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}
