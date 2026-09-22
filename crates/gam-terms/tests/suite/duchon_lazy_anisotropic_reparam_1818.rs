//! Regression for the lazy anisotropic gap in gam#1818.
//!
//! A cold dense Duchon build and the lazy/operator build represent the same
//! function and therefore must use the same data-metric radial chart.  The old
//! lazy implementation computed `V` only in its isotropic branch; anisotropic
//! builds stayed in the raw constrained-kernel frame, so their penalty spectrum
//! and REML geometry depended on which memory route happened to be selected.

use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_runtime::resource::ResourcePolicy;
use gam_terms::basis::{
    BasisMetadata, BasisWorkspace, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, SpatialIdentifiability, build_duchon_basiswithworkspace,
};
use ndarray::{Array1, Array2, Axis, s};

#[test]
fn lazy_anisotropic_duchon_uses_data_metric_radial_chart() {
    let n = 48usize;
    let data = Array2::from_shape_fn((n, 2), |(row, axis)| {
        let t = row as f64 / (n - 1) as f64;
        match axis {
            0 => 2.0 * t - 1.0,
            _ => (3.0 * t).sin() + 0.2 * t,
        }
    });
    let center_rows = [0usize, 5, 10, 15, 21, 27, 33, 39, 43, 47];
    let centers = data.select(Axis(0), &center_rows);
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers),
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        // `SpatialIdentifiability::None`, NOT the default, and the certificate
        // below is why. The default is `OrthogonalToParametric`, and for a
        // direct `build_duchon_basiswithworkspace` call that is not inert:
        // `spatial_identifiability_transform_from_design_matrix` returns
        // `Some(Z)` for it and the builder then wraps the design in `Z`. The
        // realized columns would be `K·Z_k·V·Z`, so the Gram measured below
        // would be `ZᵀVᵀG_cVZ`, which is not `I` and is not supposed to be —
        // the assertion would be reporting the centering, not the chart.
        // (The term-collection path never hits this: its Duchon arm downgrades
        // `OrthogonalToParametric` to `None` and leaves the centering to the
        // collection. A direct caller like this fixture does hit it.)
        // Identifiability is orthogonal to what this test covers — whether the
        // LAZY anisotropic branch computes and applies `V` at all — so removing
        // it removes a confound rather than any coverage.
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: Some(vec![0.7, -0.7]),
        operator_penalties: DuchonOperatorPenaltySpec::all_disabled(),
        boundary: Default::default(),
        radial_reparam: None,
    };

    // Force the operator route from the predicted materialization footprint,
    // without making the fixture itself large.  The owned-data/operator-cache
    // allowances remain at their detected defaults.
    let mut policy = ResourcePolicy::default_library();
    policy.max_single_materialization_bytes = 1;
    let mut workspace = BasisWorkspace::with_policy(policy);
    let built = build_duchon_basiswithworkspace(data.view(), &spec, &mut workspace)
        .expect("lazy anisotropic Duchon basis should build");

    let radial_reparam = match &built.metadata {
        BasisMetadata::Duchon {
            radial_reparam: Some(v),
            ..
        } => v,
        BasisMetadata::Duchon {
            radial_reparam: None,
            ..
        } => panic!("lazy anisotropic Duchon omitted its data-metric radial chart"),
        other => panic!("expected Duchon metadata, got {other:?}"),
    };
    assert!(
        radial_reparam.ncols() > 0,
        "data-supported anisotropic radial block must retain at least one mode"
    );

    // `V` is defined by Vᵀ G_c V = I.  The first `V.ncols()` columns of the
    // operator are exactly K·Z·V — which holds because the spec above declares
    // `SpatialIdentifiability::None`, so no further transform is wrapped around
    // the design — and their training-data Gram is therefore the coordinate-free
    // executable certificate that the lazy branch applied the generalized-eigen
    // chart rather than merely persisting metadata.
    //
    // THE BAR IS NOT MOVED. It stays at 1e-8. What changed is that the columns
    // being measured are now the ones the sentence above claims they are. If
    // this still fails, the residual is the lazy branch's and the number can be
    // trusted as such (gam#2959).
    //
    // Compute that Gram through operator products. Calling `to_dense()` here
    // would contradict the fixture's one-byte materialization ceiling and test
    // the refusal mechanism rather than the anisotropic chart.
    assert!(
        matches!(built.design, DesignMatrix::Dense(DenseDesignMatrix::Lazy(_))),
        "one-byte materialization policy must select the lazy design"
    );
    let radial_width = radial_reparam.ncols();
    let mut gram = Array2::<f64>::zeros((radial_width, radial_width));
    for col in 0..radial_width {
        let mut basis_vector = Array1::<f64>::zeros(built.design.ncols());
        basis_vector[col] = 1.0;
        let realized_column = built.design.dot(&basis_vector);
        let cross_products = built.design.transpose_vector_multiply(&realized_column);
        gram.column_mut(col)
            .assign(&cross_products.slice(s![..radial_width]));
    }
    assert!(
        matches!(built.design, DesignMatrix::Dense(DenseDesignMatrix::Lazy(_))),
        "matrix-free Gram evaluation must preserve the lazy design"
    );
    let mut max_identity_residual = 0.0_f64;
    for row in 0..gram.nrows() {
        for col in 0..gram.ncols() {
            let target = if row == col { 1.0 } else { 0.0 };
            max_identity_residual = max_identity_residual.max((gram[[row, col]] - target).abs());
        }
    }
    assert!(
        max_identity_residual <= 1e-8,
        "lazy anisotropic radial design is not G-orthonormal: max |V^T G_c V - I| = {max_identity_residual:.3e}"
    );
}
