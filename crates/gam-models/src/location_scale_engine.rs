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

use crate::fit_orchestration::drivers::{
    ExactJointHyperSetup, spatial_length_scale_term_indices,
};
use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionSpec,
};
use ndarray::{Array1, ArrayView2};

/// Shared operator-aware coefficient-Hessian cost for joint-coupled
/// location-scale families.
///
/// Every Gaussian/Binomial location-scale variant exposes the same inner
/// coefficient Hessian representation: the exact dense fallback is one
/// row-coupled Hessian over all parameter blocks, while the matrix-free path
/// applies the joint Hessian in `O(n · Σp_b)`. Keep that trait-method body in
/// one place so each family implementation only supplies its observation
/// count.
pub(crate) fn location_scale_coefficient_hessian_cost(
    n: u64,
    specs: &[crate::custom_family::ParameterBlockSpec],
) -> u64 {
    crate::coefficient_cost::joint_coupled_operator_aware_hessian_cost(n, specs)
}

/// The exact-joint hyperparameter setup of a location-scale fit: the ρ seed,
/// and the κ coordinates and their bounds from the data. The ρ domain is not
/// this builder's to supply: the driver derives it per coordinate from the
/// seed design of each block once it has built them (#2812). A location-scale
/// fit with NO spatial terms never reaches this — it routes through
/// `fit_custom_family`, whose ρ domain is the same per-term resolvability
/// interval.
pub(crate) fn build_location_scale_exact_joint_setup(
    data: ArrayView2<'_, f64>,
    blocks: &[&TermCollectionSpec],
    rho0: Array1<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<ExactJointHyperSetup, gam_terms::basis::BasisError> {
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
                .reseed_from_data(data, spec, &term_indices, kappa_options)?;
        let dims = kappa.dims_per_term().to_vec();

        let lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            data,
            spec,
            &term_indices,
            &dims,
            kappa_options,
        )?;
        let upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            data,
            spec,
            &term_indices,
            &dims,
            kappa_options,
        )?;

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

    Ok(ExactJointHyperSetup::new(
        rho0,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
    use gam_terms::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
    use ndarray::{Array2, array};

    /// Build a single-term spec carrying a 1-D Matérn smooth with an explicit
    /// `length_scale`. Such a term supports outer hyper-optimization and
    /// contributes exactly one log(kappa) coordinate, so a two-block setup over
    /// `[block_a, block_b]` has a fully predictable κ layout: one coordinate per
    /// block, block A before block B.
    fn one_term_block(name: &str, feature_col: usize, length_scale: f64) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: name.to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![feature_col],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: false,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        }
    }

    /// Reference per-block κ assembly: exactly the steps the engine performs for
    /// one block, kept independent of `build_location_scale_exact_joint_setup`
    /// so the parity test compares two genuinely separate code paths rather than
    /// a value against itself.
    fn reference_block_kappa(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        let term_indices = spatial_length_scale_term_indices(spec);
        let kappa =
            SpatialLogKappaCoords::from_length_scales_aniso(spec, &term_indices, kappa_options)
                .reseed_from_data(data, spec, &term_indices, kappa_options)
                .expect("reference isotropic-scale geometry");
        let dims = kappa.dims_per_term().to_vec();
        let lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            data,
            spec,
            &term_indices,
            &dims,
            kappa_options,
        )
        .expect("reference lower spatial bounds");
        let upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            data,
            spec,
            &term_indices,
            &dims,
            kappa_options,
        )
        .expect("reference upper spatial bounds");
        // Mirror the engine's seed projection so the reference seed is comparable
        // post-clamp coordinate-for-coordinate.
        let kappa = kappa.clamp_to_bounds(&lower, &upper);
        (
            kappa.as_array().to_vec(),
            lower.as_array().to_vec(),
            upper.as_array().to_vec(),
            dims,
        )
    }

    /// Parity test for the exact-joint Hessian matvec coordinate layout that
    /// GAMLSS and survival location-scale both consume.
    ///
    /// The matvec operates over `theta = [rho | log_kappa | auxiliary]`, indexed
    /// through [`ExactJointHyperSetup::theta0`] / `lower` / `upper`. Both
    /// families route their two predictors through
    /// [`build_location_scale_exact_joint_setup`] in block order — GAMLSS as
    /// `[mean, noise]`, survival as `[threshold, log_sigma]`. This test pins the
    /// invariant they rely on: the engine concatenates per-block κ seeds and
    /// bounds in block order, with the rho head boxed to `±RHO_BOUND`, exactly
    /// matching an independent per-block assembly. If the layout ever drifts,
    /// every family's exact Newton ψ direction would index the wrong
    /// coordinates; this catches it before the matvec runs.
    #[test]
    fn exact_joint_block_layout_parity_across_families() {
        // Two distinct 1-D feature columns so each block's geometry — and hence
        // its data-aware κ seed/bounds — differs, making block-order errors
        // observable rather than masked by identical coordinates.
        let mut data = Array2::<f64>::zeros((6, 2));
        let col0 = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let col1 = array![-2.0, -1.0, 0.0, 1.0, 3.0, 5.0];
        data.column_mut(0).assign(&col0);
        data.column_mut(1).assign(&col1);

        let kappa_options = SpatialLengthScaleOptimizationOptions::default();

        // Block A plays the location predictor (GAMLSS mean / survival
        // threshold), block B the scale predictor (GAMLSS noise / survival
        // log-sigma). Distinct length scales give distinct κ seeds.
        let block_a = one_term_block("loc", 0, 0.5);
        let block_b = one_term_block("scale", 1, 2.0);

        // A non-trivial rho seed in penalty order so the head slice is checked,
        // not just the κ tail.
        let rho0 = array![0.3, -0.7];

        let setup = build_location_scale_exact_joint_setup(
            data.view(),
            &[&block_a, &block_b],
            rho0.clone(),
            &kappa_options,
        )
        .expect("two-block isotropic-scale geometry");

        let rho_dim = rho0.len();
        assert_eq!(
            setup.rho_dim(),
            rho_dim,
            "rho head must carry exactly the caller-supplied seed"
        );

        // Reference per-block assembly, independent of the engine.
        let (seed_a, lo_a, hi_a, dims_a) =
            reference_block_kappa(data.view(), &block_a, &kappa_options);
        let (seed_b, lo_b, hi_b, dims_b) =
            reference_block_kappa(data.view(), &block_b, &kappa_options);

        // Each single-Matérn block contributes exactly one κ coordinate.
        assert_eq!(dims_a, vec![1], "block A must be a single isotropic κ");
        assert_eq!(dims_b, vec![1], "block B must be a single isotropic κ");

        let expected_kappa: Vec<f64> = seed_a.iter().chain(seed_b.iter()).copied().collect();
        let expected_lower: Vec<f64> = lo_a.iter().chain(lo_b.iter()).copied().collect();
        let expected_upper: Vec<f64> = hi_a.iter().chain(hi_b.iter()).copied().collect();

        assert_eq!(
            setup.log_kappa_dim(),
            expected_kappa.len(),
            "engine κ dim must equal block-A + block-B κ dims"
        );
        assert_eq!(
            setup.auxiliary_dim(),
            0,
            "no auxiliary axis is supplied for the two-block location-scale setup"
        );

        let theta0 = setup.theta0();
        let lower = setup.lower();
        let upper = setup.upper();

        // Rho head: seed (sanitized clamp leaves these untouched) and the shared
        // joint-search box: every seed here is strictly inside the prior, so the
        // box is the prior itself.
        for k in 0..rho_dim {
            assert!(
                (theta0[k] - rho0[k]).abs() <= 1e-12,
                "rho seed mismatch at {k}: {} vs {}",
                theta0[k],
                rho0[k]
            );
            // #2812: the domain is derived from each block's own spectrum, and
            // the seed is interior to it.
            assert!(
                lower[k] < rho0[k] && rho0[k] < upper[k],
                "seed {k} = {} must be interior to its derived domain [{}, {}]",
                rho0[k],
                lower[k],
                upper[k]
            );
        }

        // κ tail: must be block A then block B, coordinate-for-coordinate equal
        // to the independent per-block assembly. This is the matvec layout both
        // families index.
        for (j, &want) in expected_kappa.iter().enumerate() {
            let got = theta0[rho_dim + j];
            assert!(
                (got - want).abs() <= 1e-12,
                "κ seed mismatch at tail index {j}: {got} vs {want}"
            );
        }
        for (j, &want) in expected_lower.iter().enumerate() {
            let got = lower[rho_dim + j];
            assert!(
                (got - want).abs() <= 1e-12,
                "κ lower-bound mismatch at tail index {j}: {got} vs {want}"
            );
        }
        for (j, &want) in expected_upper.iter().enumerate() {
            let got = upper[rho_dim + j];
            assert!(
                (got - want).abs() <= 1e-12,
                "κ upper-bound mismatch at tail index {j}: {got} vs {want}"
            );
        }

        // Block order must be observable: block A's κ seed differs from block
        // B's. If the engine ever concatenated in the wrong order (or collapsed
        // the blocks), this distinguishes it even though both are 1-D κ.
        assert!(
            (theta0[rho_dim] - theta0[rho_dim + 1]).abs() > 1e-9,
            "block A and block B κ seeds must differ so block order is testable: \
             a={}, b={}",
            theta0[rho_dim],
            theta0[rho_dim + 1]
        );
        assert!(
            (theta0[rho_dim] - seed_a[0]).abs() <= 1e-12,
            "first κ coordinate must be block A's, not block B's"
        );
        assert!(
            (theta0[rho_dim + 1] - seed_b[0]).abs() <= 1e-12,
            "second κ coordinate must be block B's, not block A's"
        );

        // Bounds bracket the seed in every κ coordinate (projection invariant
        // the matvec relies on for a feasible start).
        for j in 0..expected_kappa.len() {
            let s = theta0[rho_dim + j];
            assert!(
                lower[rho_dim + j] <= s + 1e-12 && s <= upper[rho_dim + j] + 1e-12,
                "κ seed at {j} must lie within its bounds: {} not in [{}, {}]",
                s,
                lower[rho_dim + j],
                upper[rho_dim + j]
            );
        }
    }
}
