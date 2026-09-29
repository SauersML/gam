#![cfg(test)]
//! Plane-rotation recovery on the planted toys of #2951: the rotation block with a
//! repeated angle, the paired copy's merged linear part, and linear parts that are not
//! rotations, on which no plane may be claimed.

use crate::parameter_decomposition::module_split::MlpNormalForm;
use crate::parameter_decomposition::spectral::{
    PlaneRotationError, PlaneRotationRecovery, RotationAmbiguity, RotationClusterKind, recover_plane_rotations,
};
use crate::parameter_decomposition::test_support::planted_toys::{
    MlpToy, RANDOM_NULL_SEED, hadamard_modules, paired_copy, random_mlp, rotation_toy,
};
use crate::parameter_decomposition::test_support::{projector_distance, test_governor};
use gam_linalg::decision::projector_error_bar;
use gam_math::gaussian_activation::GaussianActivation;
use ndarray::{Array2, s};

/// Toy 4: angles `0.3, 0.3, 1.1` in a hidden basis of `ℝ⁶`. The 1.1 plane is one
/// identified plane; the repeated 0.3 pair is reported as its 4-dimensional invariant
/// subspace with the repeated-cosine ambiguity, never as two planes. Each cluster spans
/// its planted columns within the Davis–Kahan bar of its separation, widened by the
/// planted matrix's own defect.
#[test]
fn toy4_rotation_reports_one_plane_and_one_repeated_pair() {
    let toy = rotation_toy();
    let planted = &toy.planted;
    let recovery = recover_plane_rotations(test_governor(), planted.matrix.view()).expect("recovery");
    assert_eq!(recovery.clusters.len(), 2, "{recovery:?}");
    assert_eq!(
        recovery.ambiguities(),
        vec![RotationAmbiguity::RepeatedCosine { cluster: 1, planes: toy.repeated.2 }, RotationAmbiguity::Winding]
    );
    // Increasing cosine: the 1.1 plane, then the repeated 0.3 pair.
    for (cluster, (angle, planes, columns)) in recovery.clusters.iter().zip([
        (toy.identified_plane.0, 1, toy.identified_plane.1.clone()),
        (toy.repeated.0, toy.repeated.2, toy.repeated.1.clone()),
    ]) {
        match &cluster.kind {
            RotationClusterKind::Rotation { planes: found, .. } => assert_eq!(*found, planes),
            other => panic!("expected a rotation cluster, got {other:?}"),
        }
        let widen = recovery.orthogonality_defect + planted.matrix_defect + planted.form_defect;
        let cosine = angle.cos();
        assert!(cluster.cosine_interval.0 - widen <= cosine && cosine <= cluster.cosine_interval.1 + widen);
        let bar = projector_error_bar(
            cluster.separation,
            recovery.perturbation_bound + recovery.orthogonality_defect + planted.matrix_defect,
        );
        let column_defect = planted.basis_defect * (2.0 + planted.basis_defect);
        let (distance, band) = projector_distance(cluster.basis.view(), planted.basis.slice(s![.., columns]));
        assert!(distance <= bar + column_defect + band, "angle {angle}: {distance:e} beyond {:e}", bar + column_defect + band);
    }
}

fn claims_no_plane(result: Result<PlaneRotationRecovery, PlaneRotationError>, context: &str) {
    match result {
        Ok(recovery) => {
            for cluster in &recovery.clusters {
                assert!(
                    !matches!(cluster.kind, RotationClusterKind::Rotation { .. }),
                    "{context}: a plane was claimed: {cluster:?}"
                );
            }
        }
        Err(PlaneRotationError::NotInvertible { .. }) => {}
        Err(other) => panic!("{context}: unexpected refusal {other}"),
    }
}

/// `½ W_out W_in` (+ `L`), the linear part of the exact GELU `σ(t) = t/2 + ψ(t)`.
fn gelu_linear_part(block: &MlpToy) -> Array2<f64> {
    let mut linear = block.w_out.dot(&block.w_in) * 0.5;
    if let Some(skip) = &block.skip {
        linear += skip;
    }
    linear
}

/// Toy 1's merged linear part is the identity exactly: one fixed cluster and the identity
/// ambiguity, no plane. Toy 2's and toy 7's GELU linear parts, and toy 7's residual
/// reading `I + A`, are not rotations: no plane is claimed on any of them.
#[test]
fn toys_one_two_and_seven_claim_no_plane() {
    let copy = paired_copy(8);
    let form = MlpNormalForm::new(
        GaussianActivation::ExactGelu,
        copy.w_in.view(),
        copy.b_in.view(),
        copy.w_out.view(),
        copy.b_out.view(),
        None,
    )
    .expect("normal form");
    let recovery = recover_plane_rotations(test_governor(), form.linear.view()).expect("recovery");
    assert_eq!(recovery.clusters.len(), 1);
    assert!(matches!(recovery.clusters[0].kind, RotationClusterKind::Fixed { .. }));
    assert_eq!(recovery.clusters[0].basis.ncols(), 8);
    assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Identity]);

    let (modules, _) = hadamard_modules(2951);
    claims_no_plane(recover_plane_rotations(test_governor(), gelu_linear_part(&modules).view()), "planted modules");
    let random = random_mlp(RANDOM_NULL_SEED, 64, 16);
    let linear = gelu_linear_part(&random);
    claims_no_plane(recover_plane_rotations(test_governor(), linear.view()), "random block");
    let residual = &linear + &Array2::<f64>::eye(16);
    claims_no_plane(recover_plane_rotations(test_governor(), residual.view()), "random residual block");
}
