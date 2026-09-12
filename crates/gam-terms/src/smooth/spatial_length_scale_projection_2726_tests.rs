//! gam#2726: the joint `[ρ, ψ]` route's ψ seed and the scalar-ρ incumbent it is
//! graded against must be the SAME ψ.
//!
//! They were not: the seed constructors projected `length_scale` onto the
//! caller's `[min_length_scale, max_length_scale]` window while the spec — and
//! therefore the incumbent fit — kept the raw value, putting the two routes
//! exactly one projection step (`ln 10` on the measured arm) apart.
//!
//! Every assertion here is bit-exact, and the control arm is asserted first:
//! the unprojected spec really does drive the two apart, so a no-op change
//! cannot make the gate pass vacuously.

use super::*;
use crate::basis::{
    DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, OperatorPenaltySpec, OneDimensionalBoundary,
    SpatialIdentifiability,
};

const RAW_LENGTH_SCALE: f64 = 1.0e-3;
const MIN_LENGTH_SCALE: f64 = 1.0e-2;
const MAX_LENGTH_SCALE: f64 = 1.0e2;

fn options() -> SpatialLengthScaleOptimizationOptions {
    SpatialLengthScaleOptimizationOptions {
        min_length_scale: MIN_LENGTH_SCALE,
        max_length_scale: MAX_LENGTH_SCALE,
        ..SpatialLengthScaleOptimizationOptions::default()
    }
}

fn duchon_1d_spec(length_scale: f64) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon_1d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: Some(length_scale),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec {
                        mass: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                        tension: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                        stiffness: OperatorPenaltySpec::Active {
                            initial_log_lambda: 0.0,
                            prior: None,
                        },
                    },
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

fn grid_1d(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0)
}

#[test]
fn joint_psi_seed_equals_the_spec_incumbent_after_upstream_projection() {
    let opts = options();
    let terms = [0usize];

    // Control: the raw spec disagrees with the seed, and by EXACTLY the
    // projection step — the seed is −ln(min_length_scale) while the incumbent
    // is −ln(length_scale), with nothing else in between.
    let raw_spec = duchon_1d_spec(RAW_LENGTH_SCALE);
    let raw_incumbent_psi = -get_spatial_length_scale(&raw_spec, 0)
        .expect("duchon term exposes a length scale")
        .ln();
    let raw_seed = SpatialLogKappaCoords::from_length_scales(&raw_spec, &terms, &opts);
    assert_ne!(
        raw_seed.term_slice(0)[0],
        raw_incumbent_psi,
        "control: the unprojected spec must still drive the two routes apart"
    );
    assert_eq!(
        raw_seed.term_slice(0)[0] - raw_incumbent_psi,
        RAW_LENGTH_SCALE.ln() - MIN_LENGTH_SCALE.ln(),
        "control: the whole #2726 psi step must be the projection, seed={:.17e} \
         incumbent={raw_incumbent_psi:.17e}",
        raw_seed.term_slice(0)[0],
    );

    // The repair: project once, in the spec, upstream of every consumer.
    let mut spec = duchon_1d_spec(RAW_LENGTH_SCALE);
    let moved = project_spatial_length_scales_in_spec(&mut spec, &terms, &opts)
        .expect("duchon term accepts a projected length scale");
    assert_eq!(moved.len(), 1, "exactly the out-of-window term moves");
    assert_eq!(moved[0], (0, RAW_LENGTH_SCALE, MIN_LENGTH_SCALE));

    let incumbent = get_spatial_length_scale(&spec, 0).expect("length scale survives");
    assert_eq!(incumbent, MIN_LENGTH_SCALE);
    let incumbent_psi = -incumbent.ln();

    // BIT-exact, for both constructors: the two routes now derive ψ from one
    // expression applied to one spec value.
    for (name, seed) in [
        (
            "isotropic",
            SpatialLogKappaCoords::from_length_scales(&spec, &terms, &opts),
        ),
        (
            "aniso",
            SpatialLogKappaCoords::from_length_scales_aniso(&spec, &terms, &opts),
        ),
    ] {
        assert_eq!(
            seed.term_slice(0)[0],
            incumbent_psi,
            "{name} seed must be bit-identical to the spec incumbent's psi"
        );
    }

    // ...and the SECOND projection site — `clamp_to_bounds` against the
    // data-derived search box — is now inert, so a repair that touched only one
    // site cannot masquerade as a fix while the other re-projects.
    let data = grid_1d(64);
    let (psi_lo, psi_hi) =
        spatial_term_psi_search_box(data.view(), &spec, 0, &opts).expect("duchon psi search box");
    assert!(
        psi_lo <= incumbent_psi && incumbent_psi <= psi_hi,
        "the search box must contain the projected incumbent psi: \
         [{psi_lo:.17e}, {psi_hi:.17e}] vs {incumbent_psi:.17e}"
    );
}

/// An in-window `length_scale` is untouched: the projection is a projection,
/// not a rewrite, and it never invents motion where the caller's window is
/// already satisfied.
#[test]
fn in_window_length_scale_is_not_moved() {
    let opts = options();
    let mut spec = duchon_1d_spec(1.0);
    let moved =
        project_spatial_length_scales_in_spec(&mut spec, &[0], &opts).expect("projection succeeds");
    assert!(moved.is_empty(), "in-window scale must not move: {moved:?}");
    assert_eq!(get_spatial_length_scale(&spec, 0), Some(1.0));
}
