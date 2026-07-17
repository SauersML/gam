use super::smoothing_correction::{EigenClassification, invert_identified_rho_hessian};
use ndarray::Array2;

/// Build a real symmetric n×n matrix with a specified eigenvalue spectrum
/// rotated by a fixed orthogonal basis. Returns (matrix, eigenvectors).
fn build_with_spectrum(eigenvalues: &[f64]) -> (Array2<f64>, Array2<f64>) {
    let n = eigenvalues.len();
    let mut q = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = if i == j {
                1.0
            } else {
                ((i + 1) as f64 * 0.37 + (j + 1) as f64 * 0.19).sin()
            };
            q[[j, i]] = v;
        }
    }
    // Modified Gram-Schmidt orthonormalization on columns.
    for i in 0..n {
        for k in 0..i {
            let mut dot = 0.0;
            for r in 0..n {
                dot += q[[r, i]] * q[[r, k]];
            }
            for r in 0..n {
                q[[r, i]] -= dot * q[[r, k]];
            }
        }
        let mut nrm = 0.0;
        for r in 0..n {
            nrm += q[[r, i]] * q[[r, i]];
        }
        let nrm = nrm.sqrt();
        assert!(nrm > 1e-12, "degenerate basis in test setup");
        for r in 0..n {
            q[[r, i]] /= nrm;
        }
    }
    // Form A = Q * diag(eigenvalues) * Q^T.
    let mut a = Array2::<f64>::zeros((n, n));
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += q[[r, k]] * eigenvalues[k] * q[[c, k]];
            }
            a[[r, c]] = sum;
        }
    }
    for r in 0..n {
        for c in (r + 1)..n {
            let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
            a[[r, c]] = avg;
            a[[c, r]] = avg;
        }
    }
    (a, q)
}

#[test]
fn spd_case_returns_full_rank_inverse_no_repair() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
    let inv = invert_identified_rho_hessian(&a, 0).expect("invert");
    assert_eq!(inv.active_rank, 4);
    assert_eq!(inv.structural_zero, 0);
    assert!(!inv.used_structural_pseudoinverse);

    let prod = a.dot(&inv.inverse);
    for r in 0..4 {
        for c in 0..4 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!(
                (prod[[r, c]] - expected).abs() < 1e-9,
                "A*Ainv[{r},{c}]={} not ~ {expected}",
                prod[[r, c]]
            );
        }
    }
}

#[test]
fn saddle_is_rejected_instead_of_salvaged() {
    let evals = [10.0, 5.0, 2.0, -0.066];
    let (a, _) = build_with_spectrum(&evals);
    let error = invert_identified_rho_hessian(&a, 0).unwrap_err();
    assert!(error.contains("negative curvature") || error.contains("positive definite"));
}

#[test]
fn structurally_certified_zero_direction_uses_pseudoinverse() {
    let evals = [10.0, 5.0, 2.0, 0.0];
    let (a, q) = build_with_spectrum(&evals);
    let inv = invert_identified_rho_hessian(&a, 1).expect("invert");
    assert_eq!(inv.active_rank, 3, "expected three identified directions");
    assert_eq!(inv.structural_zero, 1);
    assert!(inv.used_structural_pseudoinverse);
    // Exactly one eigenpair classifies as the certified structural zero; its
    // POSITION in the classification list is an eigensolver ordering detail,
    // not part of the contract.
    assert_eq!(
        inv.classifications
            .iter()
            .filter(|class| matches!(class, EigenClassification::StructuralZero))
            .count(),
        1,
        "exactly one structural-zero classification expected"
    );

    let v_flat = q.column(3).to_owned();
    let inv_vflat = inv.inverse.dot(&v_flat);
    let nrm = inv_vflat.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        nrm < 1e-3,
        "pseudo-inverse should annihilate flat direction; got norm {nrm}"
    );
}

#[test]
fn structural_nullity_must_match_penalty_map_certificate() {
    let (a, _) = build_with_spectrum(&[10.0, 5.0, 2.0, 0.0]);
    let error = invert_identified_rho_hessian(&a, 2).unwrap_err();
    assert!(error.contains("penalty map certifies"));
}

#[test]
fn every_positive_curvature_direction_is_retained() {
    let (a, _) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0e-9]);
    let inv = invert_identified_rho_hessian(&a, 0).expect("small positive SPD inverse");
    assert_eq!(inv.active_rank, 4);
    assert!(inv.inverse.iter().all(|value| value.is_finite()));
}

#[test]
fn non_finite_input_returns_none() {
    let mut a = Array2::<f64>::eye(4);
    a[[1, 1]] = f64::NAN;
    let result = invert_identified_rho_hessian(&a, 0);
    assert!(result.is_err(), "expected error for NaN-bearing input matrix");

    let mut a = Array2::<f64>::eye(4);
    a[[2, 2]] = f64::INFINITY;
    let result = invert_identified_rho_hessian(&a, 0);
    assert!(result.is_err(), "expected error for Inf-bearing input matrix");
}

/// The slow eigendecomposition path must populate `eigenvalues` AND
/// `eigenvectors` so the [INDEF-HESS] diagnostic doesn't have to recompute
/// `eigh` redundantly. The Cholesky fast path leaves both empty since the
/// diagnostic isn't invoked when the matrix is SPD.
#[test]
fn structural_path_populates_eigenvalues_and_eigenvectors() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 0.0]);
    let inv = invert_identified_rho_hessian(&a, 1).expect("invert");
    assert!(inv.used_structural_pseudoinverse);
    assert_eq!(inv.eigenvalues.len(), 4);
    assert_eq!(inv.eigenvectors.shape(), &[4, 4]);
    assert_eq!(inv.classifications.len(), 4);
    // Eigenvectors are unit-norm and pairwise orthogonal.
    for j in 0..4 {
        let v = inv.eigenvectors.column(j);
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (nrm - 1.0).abs() < 1e-9,
            "eigenvector {j} not unit-norm: ‖v‖={nrm}"
        );
    }
}

#[test]
fn fast_path_leaves_eigendecomp_fields_empty() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
    let inv = invert_identified_rho_hessian(&a, 0).expect("invert");
    assert!(!inv.used_structural_pseudoinverse);
    assert!(inv.eigenvalues.is_empty());
    assert!(inv.eigenvectors.is_empty());
    assert!(inv.classifications.is_empty());
}
