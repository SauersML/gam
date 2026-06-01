//! Per-axis RELEVANCE penalties for a multivariate Duchon smooth — the
//! penalty-based replacement for brittle kernel-η ("scale_dims") optimization.
//!
//! Contract: when `scale_dims` is on (the spec carries `aniso_log_scales`), the
//! single isotropic gradient penalty `Σ‖∇f‖²` (`PenaltySource::OperatorTension`)
//! is REPLACED by `dim` per-axis penalties `Σ(∂f/∂x_a)²`
//! (`PenaltySource::OperatorRelevance { axis }`), each carrying its own REML λ_a.
//! REML can then shrink an axis's nonlinear contribution toward flat only when
//! that axis does not earn its keep — automatic relevance determination via
//! plain quadratic penalties, no kernel-metric derivatives. The isotropic-order
//! rungs (curvature `Primary`, amplitude `OperatorMass`, the affine trend ridge)
//! stay single, so a smooth, linearly-useful axis keeps its slope.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis,
};
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

/// A 2-D cubic Duchon spec (`s = (d−1)/2 = 0.5`), with `scale_dims` toggled via
/// the presence of `aniso_log_scales`.
fn duchon_2d_spec(k: usize, scale_dims: bool) -> DuchonBasisSpec {
    DuchonBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: scale_dims.then(|| vec![0.0, 0.0]),
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

fn sources(spec: &DuchonBasisSpec, data: &Array2<f64>) -> Vec<PenaltySource> {
    build_duchon_basis(data.view(), spec)
        .expect("build_duchon_basis succeeded")
        .penaltyinfo
        .iter()
        .map(|info| info.source.clone())
        .collect()
}

#[test]
fn scale_dims_emits_one_relevance_penalty_per_axis() {
    let data = synthetic_data(400, 2, 7);
    let srcs = sources(&duchon_2d_spec(20, true), &data);

    // One relevance penalty per input axis, each its own REML λ.
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
        "expected one OperatorRelevance per axis (0, 1), got {srcs:?}"
    );

    // The isotropic gradient penalty is REPLACED, not duplicated.
    assert!(
        !srcs
            .iter()
            .any(|s| matches!(s, PenaltySource::OperatorTension)),
        "isotropic OperatorTension must be replaced by per-axis relevance, got {srcs:?}"
    );

    // Curvature, amplitude, and the affine trend stay single / isotropic.
    assert!(srcs.iter().any(|s| matches!(s, PenaltySource::Primary)));
    assert!(
        srcs.iter()
            .any(|s| matches!(s, PenaltySource::OperatorMass))
    );
    assert!(
        srcs.iter()
            .any(|s| matches!(s, PenaltySource::DoublePenaltyNullspace))
    );
}

#[test]
fn without_scale_dims_the_gradient_penalty_stays_isotropic() {
    let data = synthetic_data(400, 2, 7);
    let srcs = sources(&duchon_2d_spec(20, false), &data);

    assert!(
        srcs.iter()
            .any(|s| matches!(s, PenaltySource::OperatorTension)),
        "isotropic default must emit a single OperatorTension, got {srcs:?}"
    );
    assert!(
        !srcs
            .iter()
            .any(|s| matches!(s, PenaltySource::OperatorRelevance { .. })),
        "no per-axis relevance without scale_dims, got {srcs:?}"
    );
}
