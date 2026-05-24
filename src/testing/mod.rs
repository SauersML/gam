//! Generic testing utilities.

use crate::families::custom_family::{ParameterBlockSpec, PenaltyMatrix};
use crate::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use crate::resource::MatrixMaterializationError;
use ndarray::{Array1, Array2, Axis, array, s};
use std::ops::Range;
use std::sync::Arc;

pub struct BinomialLocationScaleBaseFixture {
    pub n: usize,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub threshold_design: DesignMatrix,
    pub log_sigma_design: DesignMatrix,
    pub threshold_spec: ParameterBlockSpec,
    pub log_sigma_spec: ParameterBlockSpec,
}

pub fn binomial_location_scale_base_fixture() -> BinomialLocationScaleBaseFixture {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design =
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((n, 1), 1.0)));
    let log_sigma_design =
        DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((n, 1), 1.0)));
    let threshold_spec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: threshold_design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.2]),
    };
    let log_sigma_spec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_design.clone(),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![-0.2],
        initial_beta: Some(array![-0.1]),
    };
    BinomialLocationScaleBaseFixture {
        n,
        y,
        weights,
        threshold_design,
        log_sigma_design,
        threshold_spec,
        log_sigma_spec,
    }
}

#[derive(Clone)]
struct NoDensifyOperator {
    dense: Array2<f64>,
}

impl LinearOperator for NoDensifyOperator {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    fn ncols(&self) -> usize {
        self.dense.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.dot(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.dense.t().dot(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "NoDensifyOperator weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows()
            ));
        }
        let weighted = &self.dense * &weights.view().insert_axis(Axis(1));
        Ok(self.dense.t().dot(&weighted))
    }
}

impl DenseDesignOperator for NoDensifyOperator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        out.assign(&self.dense.slice(s![rows, ..]));
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        // SAFETY: `NoDensifyOperator` is a test fixture whose entire purpose
        // is asserting that operator-aware code paths never reach the
        // `to_dense` fallback. Any call here means a code path under test
        // bypassed `row_chunk_into` and tried to materialize a dense
        // matrix that the fixture forbids — exactly the regression this
        // operator is designed to detect.
        panic!("NoDensifyOperator must stay lazy")
    }
}

pub fn no_densify_design(dense: Array2<f64>) -> DesignMatrix {
    DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyOperator {
        dense,
    })))
}

/// Assert that a central difference of an array-producing function matches the analytical derivative.
#[macro_export]
macro_rules! assert_central_difference_array {
    ($x:expr, $h:expr, |$var:ident| $eval:expr, $analytical:expr, $tol:expr) => {
        let f_plus = {
            let $var = $x + $h;
            $eval
        };
        let f_minus = {
            let $var = $x - $h;
            $eval
        };
        assert_eq!(f_plus.len(), $analytical.len());
        for j in 0..$analytical.len() {
            let fd = (f_plus[j] - f_minus[j]) / (2.0 * $h);
            approx::assert_abs_diff_eq!(fd, $analytical[j], epsilon = $tol);
        }
    };
}

/// Asserts that a finite difference dense matrix closely matches an analytically computed
/// directional derivative matrix, both in tolerance and in component-wise sign.
pub fn assert_matrix_derivativefd(fd: &Array2<f64>, analytic: &Array2<f64>, tol: f64, label: &str) {
    assert_eq!(analytic.dim(), fd.dim(), "{} dimensions must match", label);
    for i in 0..analytic.nrows() {
        for j in 0..analytic.ncols() {
            let analytic_ij = analytic[[i, j]];
            let fd_ij = fd[[i, j]];
            let diff = (analytic_ij - fd_ij).abs();

            if analytic_ij.abs() > tol && fd_ij.abs() > tol {
                assert_eq!(
                    analytic_ij.signum(),
                    fd_ij.signum(),
                    "{} sign mismatch at ({}, {}): analytic={}, fd={}",
                    label,
                    i,
                    j,
                    analytic_ij,
                    fd_ij
                );
            }
            assert!(
                diff <= tol,
                "{} value mismatch at ({}, {}): analytic={}, fd={}, abs_diff={}, tol={}",
                label,
                i,
                j,
                analytic_ij,
                fd_ij,
                diff,
                tol
            );
        }
    }
}
