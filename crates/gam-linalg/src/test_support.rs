//! Linear-algebra test fixtures shared across the workspace.
//!
//! These fixtures exercise `gam-linalg`'s operator-backed design machinery, so
//! they live here — the crate that owns [`LinearOperator`], [`DesignMatrix`],
//! and friends — rather than in a downstream crate. Following the workspace
//! convention for `test_support` modules (and matching the root `gam` crate),
//! this is a plain always-compiled `pub mod`: feature gates and `#[cfg(test)]`
//! module gates are banned here, and a `cfg(test)` module would be invisible to
//! downstream crates' test builds anyway. The contents are `pub`, so they are
//! reachable (no dead-code lint) yet only ever called from `#[cfg(test)]` code.

use crate::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use gam_runtime::resource::MatrixMaterializationError;
use ndarray::{Array1, Array2, Axis, s};
use std::ops::Range;
use std::sync::Arc;

/// A dense-backed [`LinearOperator`] that refuses to materialize itself.
///
/// It services every operator-aware code path (`apply`, `apply_transpose`,
/// `row_chunk_into`, `diag_xtw_x`) but panics from [`to_dense`](DenseDesignOperator::to_dense).
/// Wrapping a design in this fixture turns "a code path densified when it should
/// have stayed lazy" — the regression we guard against — into a hard test
/// failure instead of a silent, slow correctness-preserving fallback.
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
        // `NoDensifyOperator` is a test fixture asserting that
        // operator-aware code paths never densify.
        // SAFETY: a call here means a code path under test bypassed
        // `row_chunk_into` and tried to materialize — the regression
        // this fixture is designed to catch.
        panic!("NoDensifyOperator must stay lazy")
    }
}

/// Build an operator-backed [`DesignMatrix`] from a dense array that will panic
/// if any consumer tries to densify it. See [`NoDensifyOperator`].
pub fn no_densify_design(dense: Array2<f64>) -> DesignMatrix {
    DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyOperator { dense })))
}

#[cfg(test)]
mod tests {
    use super::no_densify_design;
    use ndarray::array;

    /// Regression guard for #1566: `no_densify_design` must live in `gam-linalg`
    /// (the crate that owns the operator traits) and yield an operator-backed
    /// design that services the lazy paths without ever materializing. If the
    /// fixture is dropped or moved back out of this crate, this test stops
    /// compiling in the very lib-test phase the issue was about.
    #[test]
    fn no_densify_design_is_operator_backed_and_stays_lazy() {
        let design = no_densify_design(array![[1.0, 2.0], [3.0, 4.0]]);
        assert!(design.as_dense_ref().is_none(), "must not be materialized");
        assert!(!design.is_materialized_dense());
        assert!(design.is_operator_backed());
        assert_eq!(design.nrows(), 2);
        assert_eq!(design.ncols(), 2);

        // Operator-aware paths still work: y = X·β and lazy row chunks.
        let beta = array![1.0, -1.0];
        let got = design.dot(&beta);
        assert!((got[0] - (-1.0)).abs() < 1e-12); // 1·1 + 2·(-1)
        assert!((got[1] - (-1.0)).abs() < 1e-12); // 3·1 + 4·(-1)
        let chunk = design
            .try_row_chunk(0..2)
            .expect("row chunk must stay lazy, not densify");
        assert_eq!(chunk, array![[1.0, 2.0], [3.0, 4.0]]);
    }

    /// The whole point of the fixture: any code path that tries to collapse it to
    /// a dense matrix trips a hard panic, turning a silent densification
    /// regression into a test failure.
    #[test]
    #[should_panic(expected = "operator-backed design")]
    fn no_densify_design_rejects_materialization() {
        let design = no_densify_design(array![[1.0, 2.0], [3.0, 4.0]]);
        design.as_dense_cow();
    }
}
