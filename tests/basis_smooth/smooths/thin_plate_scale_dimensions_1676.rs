//! `scale_dimensions=True` must genuinely engage per-axis anisotropy for a
//! thin-plate (`bs="tp"`) smooth, not be a silent no-op (gam#1676).
//!
//! A canonical thin-plate regression spline carries a *single* curvature
//! penalty (the exact `∫|Dᵐf|²` RKHS Gram) with no per-axis structure, so the
//! `scale_dimensions` flag had nothing to engage and was silently dropped for
//! `bs="tp"` while it worked for `duchon()`/`matern()`. The fix rewrites a
//! multi-axis thin-plate term to its mathematically-equivalent anisotropic s=0
//! Duchon spline (the thin-plate kernel `r^{2m−d}` IS the s=0 Duchon kernel), so
//! the per-axis tension-ARD machinery (`Σ‖∇f‖²` → `d` directional
//! `Σ(∂f/∂x_a)²`, each its own REML `λ_a`) engages exactly as it does for
//! Duchon.
//!
//! These tests pin the contract at the `enable_scale_dimensions` rewrite layer
//! (the Rust core behind gamfit's `scale_dimensions` kwarg):
//!   * a multi-axis `tp` term becomes an s=0 Duchon term carrying
//!     `aniso_log_scales`, and the realized basis then emits one
//!     `OperatorRelevance{axis}` penalty per input axis (anisotropy ENGAGED);
//!   * a 1-D `tp` term is untouched (anisotropy is meaningless on one axis);
//!   * WITHOUT `scale_dimensions` a `tp` term stays canonical thin-plate.

use gam::basis::{
    CenterStrategy, DuchonNullspaceOrder, PenaltySource, SpatialIdentifiability,
    ThinPlateBasisSpec, build_duchon_basis,
};
use gam::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

fn thin_plate_term(d: usize, num_centers: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "tp".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: (0..d).collect(),
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers },
                    periodic: None,
                    length_scale: 0.0,
                    double_penalty: true,
                    identifiability: SpatialIdentifiability::None,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

#[test]
fn thin_plate_scale_dimensions_promotes_to_anisotropic_s0_duchon() {
    let mut spec = thin_plate_term(2, 25);

    // The Rust core behind gamfit's `scale_dimensions=True`.
    gam::term_builder::enable_scale_dimensions(&mut spec);

    // The thin-plate term is rewritten to the mathematically-equivalent s=0
    // Duchon spline (the thin-plate kernel `r^{2m−d}` is the s=0 Duchon kernel),
    // which CAN carry per-axis relevance, with the geometry seed sentinel set.
    let SmoothBasisSpec::Duchon { spec: duchon, .. } = &spec.smooth_terms[0].basis else {
        panic!(
            "scale_dimensions must rewrite a multi-axis thin-plate term to its \
             anisotropic Duchon twin, got {:?}",
            spec.smooth_terms[0].basis
        );
    };
    assert_eq!(duchon.power, 0.0, "s=0 keeps the exact thin-plate kernel");
    assert_eq!(
        duchon.length_scale, None,
        "pure scale-free Duchon (no kernel length scale): thin-plate is scale-free"
    );
    assert_eq!(
        duchon.nullspace_order,
        DuchonNullspaceOrder::Linear,
        "2-D thin-plate penalty order m=2 maps to the Linear Duchon null space"
    );
    assert_eq!(
        duchon.aniso_log_scales.as_deref(),
        Some(&[0.0, 0.0][..]),
        "the per-axis geometry seed sentinel must be set so anisotropy engages"
    );

    // And the realized basis genuinely engages anisotropy: the single isotropic
    // gradient penalty is REPLACED by one per-axis relevance penalty per axis.
    let data = synthetic_data(400, 2, 7);
    let srcs: Vec<PenaltySource> = build_duchon_basis(data.view(), duchon)
        .expect("anisotropic s=0 Duchon basis builds")
        .penaltyinfo
        .iter()
        .map(|info| info.source.clone())
        .collect();
    let relevance_axes: Vec<usize> = srcs
        .iter()
        .filter_map(|s| match s {
            PenaltySource::OperatorRelevance { axis } => Some(*axis),
            _ => None,
        })
        .collect();
    assert_eq!(
        relevance_axes,
        vec![0, 1],
        "thin-plate scale_dimensions must emit one per-axis relevance penalty \
         per input axis (anisotropy engaged), got {srcs:?}"
    );
    assert!(
        !srcs
            .iter()
            .any(|s| matches!(s, PenaltySource::OperatorTension)),
        "the isotropic gradient penalty must be replaced, not duplicated, got {srcs:?}"
    );
    // The exact thin-plate curvature RKHS Gram stays single / isotropic.
    assert!(
        srcs.iter().any(|s| matches!(s, PenaltySource::Primary)),
        "the thin-plate curvature penalty (Primary) must survive, got {srcs:?}"
    );
}

#[test]
fn thin_plate_scale_dimensions_is_a_noop_for_one_dimensional_term() {
    let mut spec = thin_plate_term(1, 15);
    gam::term_builder::enable_scale_dimensions(&mut spec);
    // Anisotropy is meaningless on a single axis (its Σ η = 0 contrast vector is
    // empty), so a 1-D thin-plate term is left exactly as it was.
    assert!(
        matches!(
            &spec.smooth_terms[0].basis,
            SmoothBasisSpec::ThinPlate { .. }
        ),
        "a 1-D thin-plate term must not be rewritten by scale_dimensions, got {:?}",
        spec.smooth_terms[0].basis
    );
}

#[test]
fn thin_plate_without_scale_dimensions_stays_canonical_thin_plate() {
    // The default path (scale_dimensions off) must be untouched: a multi-axis
    // thin-plate term that never sees `enable_scale_dimensions` stays a
    // canonical TPS term.
    let spec = thin_plate_term(2, 25);
    assert!(
        matches!(
            &spec.smooth_terms[0].basis,
            SmoothBasisSpec::ThinPlate { .. }
        ),
        "without scale_dimensions the term must stay canonical thin-plate"
    );
}
