//! The recovered universal profile must retain the hybrid kernel's algebraic tail.

use gam_terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability, build_duchon_basis,
};
use ndarray::Array2;

#[test]
fn six_dimensional_hybrid_design_retains_its_algebraic_tail_2827() {
    // For p=1, s=3, d=6, substitute t=rho*sqrt(w) in the defining integral:
    // G(rho) = 2/rho^4 * integral_0^rho t^4 K_1(t) dt.
    // The complete integral is 16, hence G(rho)=32/rho^4 minus an
    // exponentially small positive remainder. Every nonzero distance below
    // is at least 300: the omitted relative remainder is below 1e-120,
    // far below the profile's independently checked 1e-11 relative accuracy.
    let mut centers = Array2::<f64>::zeros((2, 6));
    centers[[1, 0]] = 300.0;
    let mut data = Array2::<f64>::zeros((3, 6));
    for row in 0..3 {
        data[[row, 0]] = -300.0 * (row + 1) as f64;
    }
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers),
        periodic: None,
        length_scale: Some(1.0),
        power: 3.0,
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::all_disabled(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: Some(Array2::eye(1)),
    };
    let built = build_duchon_basis(data.view(), &spec).expect("valid hybrid basis");
    let design = built
        .design
        .try_to_dense_arc("hybrid tail regression")
        .expect("tiny design");
    assert_eq!(design.dim(), (3, 2));
    assert!(design.iter().all(|value| value.is_finite()));
    // The constant side condition makes the radial column proportional to
    // phi(distance_to_center_0)-phi(distance_to_center_1). Ratios remove the
    // arbitrary null-vector sign and the basis's common kernel amplification.
    let reference = |row: usize| {
        let first = 300.0 * (row + 1) as f64;
        let second = first + 300.0;
        32.0 * (first.powi(-4) - second.powi(-4))
    };
    assert_ne!(
        design[[0, 0]],
        0.0,
        "a far-field kernel is not identically zero"
    );
    for row in 1..3 {
        let got = design[[row, 0]] / design[[0, 0]];
        let want = reference(row) / reference(0);
        assert!(
            (got - want).abs() <= 1e-11 * want.abs(),
            "row {row}: hybrid column ratio {got:.16e}, analytic tail {want:.16e}"
        );
    }
}
