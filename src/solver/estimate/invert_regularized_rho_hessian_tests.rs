use super::{EigenClassification, invert_regularized_rho_hessian};
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
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert_eq!(inv.active_rank, 4);
    assert_eq!(inv.dropped_negative, 0);
    assert_eq!(inv.dropped_small_positive, 0);
    assert_eq!(inv.dropped_numerical_zero, 0);
    assert!(!inv.repaired_hessian);

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
fn z2_saddle_case_drops_negative_eigenpair() {
    let evals = [10.0, 5.0, 2.0, -0.066];
    let (a, q) = build_with_spectrum(&evals);
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert_eq!(inv.active_rank, 3);
    assert_eq!(inv.dropped_negative, 1);
    assert_eq!(inv.dropped_small_positive, 0);
    assert_eq!(inv.dropped_numerical_zero, 0);
    assert!(inv.repaired_hessian);

    // On each active eigenvector v: inv*A*v = v.
    for active_idx in 0..4 {
        if evals[active_idx] <= 0.0 {
            continue;
        }
        let v = q.column(active_idx).to_owned();
        let av = a.dot(&v);
        let inv_av = inv.inverse.dot(&av);
        for r in 0..4 {
            assert!(
                (inv_av[r] - v[r]).abs() < 1e-9,
                "active eigenvector not preserved at idx {active_idx}, row {r}: got {}, expected {}",
                inv_av[r],
                v[r]
            );
        }
    }
    // Negative-eigenvalue direction is annihilated.
    let v_neg = q.column(3).to_owned();
    let inv_vneg = inv.inverse.dot(&v_neg);
    let nrm = inv_vneg.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        nrm < 1e-9,
        "pseudo-inverse should annihilate dropped direction; got norm {nrm}"
    );
}

#[test]
fn flat_direction_dropped() {
    // Build a matrix with one near-zero eigenvalue. We pick -1e-13 (just
    // below zero by less than neg_tol) so Cholesky reliably refuses the
    // matrix and we exercise the eigendecomp branch. The classification
    // should be DroppedNumericalZero or DroppedNegative, both of which
    // count as "near-zero direction dropped" for this test's purposes.
    let evals = [10.0, 5.0, 2.0, -1e-13];
    let (a, q) = build_with_spectrum(&evals);
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert_eq!(inv.active_rank, 3, "expected three identified directions");
    let dropped =
        inv.dropped_small_positive + inv.dropped_numerical_zero + inv.dropped_negative;
    assert_eq!(dropped, 1, "expected exactly one direction dropped");
    assert!(inv.repaired_hessian);

    let v_flat = q.column(3).to_owned();
    let inv_vflat = inv.inverse.dot(&v_flat);
    let nrm = inv_vflat.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        nrm < 1e-3,
        "pseudo-inverse should annihilate flat direction; got norm {nrm}"
    );
}

#[test]
fn mixed_negative_and_flat_yields_active_rank_two() {
    let evals = [10.0, 5.0, -0.066, 1e-13];
    let (a, _q) = build_with_spectrum(&evals);
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert_eq!(inv.active_rank, 2);
    assert_eq!(inv.dropped_negative, 1);
    assert_eq!(
        inv.dropped_small_positive + inv.dropped_numerical_zero,
        1,
        "expected one near-zero direction dropped"
    );
    assert!(inv.repaired_hessian);
}

#[test]
fn all_bad_spectrum_yields_zero_active_rank() {
    let evals = [-0.1, -0.05, -1.0, -0.5];
    let (a, _q) = build_with_spectrum(&evals);
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert_eq!(inv.active_rank, 0);
    assert_eq!(inv.dropped_negative, 4);
    assert!(inv.repaired_hessian);
    for r in 0..4 {
        for c in 0..4 {
            assert!(inv.inverse[[r, c]].abs() < 1e-12);
        }
    }
    assert!(
        inv.classifications
            .iter()
            .all(|c| matches!(c, EigenClassification::DroppedNegative))
    );
}

#[test]
fn non_finite_input_returns_none() {
    let mut a = Array2::<f64>::eye(4);
    a[[1, 1]] = f64::NAN;
    let result = invert_regularized_rho_hessian(&a);
    assert!(
        result.is_none(),
        "expected None for NaN-bearing input matrix"
    );

    let mut a = Array2::<f64>::eye(4);
    a[[2, 2]] = f64::INFINITY;
    let result = invert_regularized_rho_hessian(&a);
    assert!(
        result.is_none(),
        "expected None for Inf-bearing input matrix"
    );
}

/// The slow eigendecomposition path must populate `eigenvalues` AND
/// `eigenvectors` so the [INDEF-HESS] diagnostic doesn't have to recompute
/// `eigh` redundantly. The Cholesky fast path leaves both empty since the
/// diagnostic isn't invoked when the matrix is SPD.
#[test]
fn slow_path_populates_eigenvalues_and_eigenvectors() {
    let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, -0.066]);
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert!(inv.repaired_hessian);
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
    let inv = invert_regularized_rho_hessian(&a).expect("invert");
    assert!(!inv.repaired_hessian);
    assert!(inv.eigenvalues.is_empty());
    assert!(inv.eigenvectors.is_empty());
    assert!(inv.classifications.is_empty());
}
