//! Regression for the #1476-class double-penalty bug on the Wahba spherical
//! spline (a wrong-SUBSPACE variant of the wrong-chart family).
//!
//! The Marra & Wood double-penalty ridge must shrink the NULL SPACE of the
//! primary RKHS penalty — for the Wahba sphere that is the unpenalized low-degree
//! spherical-harmonic block. `build_wahba_decomposed_null_shrinkage` instead put
//! an identity on the KERNEL block (the curvature directions the primary ALREADY
//! penalizes) whenever the low-degree split was non-empty (`low_degree_cols > 0`,
//! i.e. > 3 centers at SPHERE_UNPENALIZED_LOW_DEGREE=1). It therefore shrank the
//! wrong subspace and left the genuinely-unpenalized harmonics free.
//!
//! Contract pinned on the realized `penalties`/`penaltyinfo`: the double-penalty
//! ridge must ANNIHILATE the primary penalty (complementary subspaces in the
//! same chart), `‖ridge·primary‖/(‖ridge‖‖primary‖) ≈ 0`. The as-built
//! kernel-block ridge left ≈ 0.41 of that product; the fixed null-space ridge
//! leaves ~0. A ≥4-center fixture puts the build in the `low_degree_cols > 0`
//! regime so the wrong-subspace path is exercised.

use faer::Side;
use gam::basis::{
    CenterStrategy, PenaltySource, SphereMethod, SphericalSplineBasisSpec,
    build_spherical_spline_basis,
};
use gam::faer_ndarray::FaerEigh;
use ndarray::{Array2, array};

fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn wahba_sphere_double_penalty_ridge_shrinks_primary_null_space() {
    // Six well-spread lat/lon centers (> 3 ⇒ low_degree_cols > 0, the regime that
    // exposes the wrong-subspace ridge).
    let data = array![
        [-60.0, -120.0],
        [-30.0, -40.0],
        [-5.0, 20.0],
        [25.0, 90.0],
        [50.0, 150.0],
        [75.0, -170.0],
    ];
    let spec = SphericalSplineBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        penalty_order: 2,
        double_penalty: true,
        radians: false,
        method: SphereMethod::Wahba,
        max_degree: None,
        wahba_kernel: Default::default(),
        identifiability: Default::default(),
    };

    let built = build_spherical_spline_basis(data.view(), &spec).expect("Wahba sphere basis");

    let primary = built
        .penalties
        .iter()
        .zip(built.penaltyinfo.iter())
        .find(|(_, i)| matches!(i.source, PenaltySource::Primary))
        .map(|(s, _)| s.clone())
        .expect("a Primary RKHS penalty");
    let ridge = built
        .penalties
        .iter()
        .zip(built.penaltyinfo.iter())
        .find(|(_, i)| matches!(i.source, PenaltySource::DoublePenaltyNullspace))
        .map(|(s, _)| s.clone())
        .expect("a DoublePenaltyNullspace ridge (double_penalty=true)");

    let pn = frob(&primary);
    let rn = frob(&ridge);
    assert!(pn > 0.0 && rn > 0.0, "degenerate primary/ridge");

    // Complementary-subspace contract: the ridge range is the primary's null
    // space (the low-degree harmonics), so ridge·primary ≈ 0. The old kernel-block
    // ridge leaves ≈ 0.41 here.
    let rel = frob(&ridge.dot(&primary)) / (rn * pn);
    assert!(
        rel < 1e-8,
        "Wahba sphere double-penalty ridge does not annihilate the primary RKHS \
         penalty (‖ridge·primary‖/(‖ridge‖‖primary‖) = {rel:.3e} ≥ 1e-8); it shrinks \
         the already-penalized kernel block instead of the unpenalized low-degree \
         null space (#1476 wrong-subspace)."
    );

    // The ridge's principal direction must earn ~zero primary energy (it is a
    // genuine null-space direction of the primary).
    let ridge_sym = (&ridge + &ridge.t()) * 0.5;
    let (evals, evecs) = ridge_sym
        .eigh(Side::Lower)
        .expect("ridge eigendecomposition");
    let top = evals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .expect("ridge eigenvalues");
    let v = evecs.column(top).to_owned();
    let primary_form = v.dot(&primary.dot(&v));
    assert!(
        primary_form.abs() / pn < 1e-6,
        "Wahba sphere ridge's principal shrinkage direction earns non-zero primary \
         energy (vᵀ S_primary v = {primary_form:.3e}); not a null-space direction (#1476)."
    );
}
