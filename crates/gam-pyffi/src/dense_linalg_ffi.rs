//! Dense decompositions the torch-interop harvest needs, evaluated by the engine.
//!
//! SPEC: Python is a thin wrapper. `gamfit/torch/harvest.py` drives a
//! user-supplied model's autograd — the JVPs and VJPs that build the pullback
//! matvec — which is torch's job and stays in torch. The two DECOMPOSITIONS its
//! subspace iteration runs are linear algebra, and linear algebra lives in the
//! Rust core. These are the entries it calls: an orthonormal basis for a dense
//! matrix's column space, and the symmetric eigendecomposition of the small
//! Rayleigh–Ritz matrix. Both route to `gam::linalg`, the same code every Rust
//! caller reads, so the harvest and the engine cannot disagree about a range or
//! a spectrum (#2899).

use crate::ffi::ffi_errors::detach_py_result;
use faer::Side;
use gam::linalg::faer_ndarray::{FaerSvd, strict_symmetric_eigh};
use gam::linalg::roundoff::SymmetricAssembly;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// An orthonormal basis for the column space of a `rows × cols` matrix
/// (`cols ≤ rows`), returned as a `rows × cols` array.
///
/// The basis is the left singular factor of the engine's SVD. A thin QR spans
/// the same space; the SVD is what every other range basis in the engine is
/// read from (`solve_design_least_squares`, `sae_residual_seed_logits`), and it
/// orders the directions by how much of the matrix each one carries, which is
/// what a subspace iteration wants from its orthonormalization.
///
/// Columns beyond the matrix's numerical rank complete the basis and carry no
/// range. A caller that needs the rank reads the singular values, not this.
#[pyfunction]
fn dense_orthonormal_range_basis<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let owned = matrix.as_array().to_owned();
    let basis = detach_py_result(py, "dense_orthonormal_range_basis", move || {
        let (rows, cols) = owned.dim();
        if rows == 0 || cols == 0 {
            return Err(format!(
                "dense_orthonormal_range_basis needs a non-empty matrix, got {rows}x{cols}"
            ));
        }
        if cols > rows {
            return Err(format!(
                "dense_orthonormal_range_basis needs at least as many rows as columns, got \
                 {rows}x{cols}"
            ));
        }
        let (left, _, _) = owned
            .svd(true, false)
            .map_err(|error| format!("dense_orthonormal_range_basis: {error}"))?;
        left.ok_or_else(|| {
            "dense_orthonormal_range_basis: the SVD omitted its left factor".to_string()
        })
    })?;
    Ok(basis.into_pyarray(py).unbind())
}

/// The eigenvalues and eigenvectors of a symmetric `n × n` matrix, in the order
/// the engine's self-adjoint eigensolver returns them.
///
/// The matrix is averaged with its transpose here, before the decomposition, so
/// each off-diagonal pair holds ONE rounded value and the engine's strict
/// routine is entered with `SymmetricAssembly::Mirrored` declared: IEEE addition
/// is commutative and correctly rounded, so `(M + Mᵀ)/2` is bitwise symmetric
/// and the two triangles may not disagree at all. A caller whose matrix is meant
/// to be symmetric already gets that average unchanged; one whose two triangles
/// hold different information gets their mean rather than a refusal, and should
/// check the symmetry it means itself.
#[pyfunction]
fn dense_symmetric_eigen<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let owned = matrix.as_array().to_owned();
    let (values, vectors) = detach_py_result(py, "dense_symmetric_eigen", move || {
        let (rows, cols) = owned.dim();
        if rows == 0 || rows != cols {
            return Err(format!(
                "dense_symmetric_eigen needs a non-empty square matrix, got {rows}x{cols}"
            ));
        }
        let symmetric = (&owned + &owned.t()) * 0.5;
        strict_symmetric_eigh(&symmetric, SymmetricAssembly::Mirrored, Side::Lower)
            .map_err(|error| format!("dense_symmetric_eigen: {error}"))
    })?;
    Ok((
        values.into_pyarray(py).unbind(),
        vectors.into_pyarray(py).unbind(),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dense_orthonormal_range_basis, module)?)?;
    module.add_function(wrap_pyfunction!(dense_symmetric_eigen, module)?)?;
    Ok(())
}
