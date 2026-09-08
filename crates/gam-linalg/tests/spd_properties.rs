use faer::Side;
use gam_linalg::faer_ndarray::{fast_ata, strict_symmetric_eigh, FaerCholesky};
use gam_linalg::matrix::{FactorizedSystem, SymmetricMatrix};
use gam_linalg::roundoff::accumulation_growth;
use gam_linalg::sparse_exact::{
    dense_to_sparse_symmetric_upper, factorize_sparse_spd_strict, logdet_from_factor,
    solve_sparse_spd,
};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

const MAX_DIMENSION: usize = 6;

fn spd_case() -> impl Strategy<Value = (Array2<f64>, Array1<f64>)> {
    (
        1usize..=MAX_DIMENSION,
        -8i32..=8,
        prop::collection::vec(-0.25f64..=0.25, MAX_DIMENSION * MAX_DIMENSION),
        prop::collection::vec(-1.0f64..=1.0, MAX_DIMENSION),
    )
        .prop_map(|(n, scale_exponent, coefficients, x)| {
            let mut lower = Array2::<f64>::zeros((n, n));
            for row in 0..n {
                for column in 0..row {
                    lower[[row, column]] = coefficients[row * MAX_DIMENSION + column];
                }
                // Keeping every pivot in [1, 3/2] makes the generated matrices
                // positive definite by construction without imposing a search box.
                lower[[row, row]] = 1.25 + coefficients[row * MAX_DIMENSION + row];
            }
            let scale = 10.0f64.powi(scale_exponent);
            let mut matrix = lower.dot(&lower.t());
            matrix *= scale;
            (matrix, Array1::from_vec(x[..n].to_vec()))
        })
}

fn condition_number(matrix: &Array2<f64>) -> f64 {
    let (eigenvalues, _) = strict_symmetric_eigh(matrix, Side::Lower).unwrap();
    let smallest = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = eigenvalues.iter().copied().fold(0.0, f64::max);
    largest / smallest
}

fn infinity_norm(values: &Array1<f64>) -> f64 {
    values.iter().copied().map(f64::abs).fold(0.0, f64::max)
}

proptest! {
    #[test]
    fn strict_spd_solve_recovers_x_with_condition_derived_error((matrix, expected) in spd_case()) {
        let n = matrix.nrows();
        let rhs = matrix.dot(&expected);
        let factor = SymmetricMatrix::Dense(matrix.clone()).factorize_spd().unwrap();
        let actual = factor.solve(&rhs).unwrap();
        let error = infinity_norm(&(&actual - &expected));

        // Standard floating-point analysis gives gamma_k = k*u/(1-k*u) for k
        // rounded operations.  Forming A*x costs at most 2n operations per row;
        // Cholesky costs at most 2n per entry, and its two triangular solves cost
        // at most 2n each.  Thus gamma_(8n) bounds the combined backward error.
        // Perturbation of an SPD solve amplifies that error by kappa(A), giving
        // ||x_hat-x||/||x|| <= kappa*gamma/(1-kappa*gamma).
        let relative_backward_error = accumulation_growth(8 * n);
        let amplified = condition_number(&matrix) * relative_backward_error;
        prop_assume!(amplified < 1.0);
        let bound = infinity_norm(&expected) * amplified / (1.0 - amplified);
        prop_assert!(error <= bound, "error {error:e} exceeded derived bound {bound:e}");
    }

    #[test]
    fn sparse_logdet_matches_dense_reference_across_scales((matrix, _) in spd_case()) {
        let n = matrix.nrows();
        let condition = condition_number(&matrix);
        let dense_logdet = matrix.cholesky(Side::Lower).unwrap().logdet();
        let sparse = dense_to_sparse_symmetric_upper(&matrix, 0.0).unwrap();
        let sparse_factor = factorize_sparse_spd_strict(&sparse).unwrap();
        let sparse_logdet = logdet_from_factor(&sparse_factor).unwrap();

        // Each factorization has at most gamma_(2n) elementwise backward error.
        // log(det(.)) has first-order relative-matrix condition at most n*kappa;
        // adding the two independent errors gives gamma_(4n).  The final n logs
        // and sum contribute six roundings per diagonal across the two paths:
        // square root, logarithm, and the final multiplication by two.
        let perturbation = condition * n as f64 * accumulation_growth(4 * n);
        prop_assume!(perturbation < 1.0);
        let factorization_bound = perturbation / (1.0 - perturbation);
        let log_accumulation_bound =
            accumulation_growth(6 * n) * (dense_logdet.abs() + n as f64);
        let bound = factorization_bound + log_accumulation_bound;
        prop_assert!((sparse_logdet - dense_logdet).abs() <= bound,
            "sparse {sparse_logdet:e}, dense {dense_logdet:e}, derived bound {bound:e}");
    }

    #[test]
    fn gram_of_random_input_is_bitwise_symmetric(
        rows in 1usize..=MAX_DIMENSION,
        columns in 1usize..=MAX_DIMENSION,
        values in prop::collection::vec(-1.0f64..=1.0, MAX_DIMENSION * MAX_DIMENSION),
    ) {
        let input = Array2::from_shape_fn((rows, columns), |(row, column)| {
            values[row * MAX_DIMENSION + column]
        });
        let gram = fast_ata(&input);
        for row in 0..columns {
            for column in 0..columns {
                prop_assert_eq!(gram[[row, column]].to_bits(), gram[[column, row]].to_bits());
            }
        }
    }

    #[test]
    fn rank_deficient_inputs_are_rejected(
        n in 2usize..=MAX_DIMENSION,
        values in prop::collection::vec(-1.0f64..=1.0, MAX_DIMENSION * MAX_DIMENSION),
    ) {
        let mut matrix = Array2::from_shape_fn((n, n), |(row, column)| {
            values[row * MAX_DIMENSION + column]
        });
        matrix = matrix.t().dot(&matrix);
        // Make the final row and column exact copies of the first. This gives
        // the stored floating-point matrix an exact null vector e_last-e_first;
        // it does not merely rely on a real-arithmetic rank statement that the
        // preceding Gram accumulation could perturb away.
        for index in 0..n {
            matrix[[n - 1, index]] = matrix[[0, index]];
            matrix[[index, n - 1]] = matrix[[index, 0]];
        }
        prop_assert!(SymmetricMatrix::Dense(matrix.clone()).factorize_spd().is_err());
        let sparse = dense_to_sparse_symmetric_upper(&matrix, 0.0).unwrap();
        prop_assert!(factorize_sparse_spd_strict(&sparse).is_err());
    }

    #[test]
    fn sparse_spd_solve_recovers_x_with_condition_derived_error((matrix, expected) in spd_case()) {
        let n = matrix.nrows();
        let rhs = matrix.dot(&expected);
        let sparse = dense_to_sparse_symmetric_upper(&matrix, 0.0).unwrap();
        let factor = factorize_sparse_spd_strict(&sparse).unwrap();
        let actual = solve_sparse_spd(&factor, &rhs).unwrap();
        let error = infinity_norm(&(&actual - &expected));
        let relative_backward_error = accumulation_growth(8 * n);
        let amplified = condition_number(&matrix) * relative_backward_error;
        prop_assume!(amplified < 1.0);
        let bound = infinity_norm(&expected) * amplified / (1.0 - amplified);
        prop_assert!(error <= bound, "error {error:e} exceeded derived bound {bound:e}");
    }
}
