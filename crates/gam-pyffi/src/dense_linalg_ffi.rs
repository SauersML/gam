//! The torch-interop harvest's eigen and trace algorithms, driven by the caller's
//! operator products.
//!
//! SPEC: Python is a thin wrapper. `gamfit/torch/harvest.py` drives a
//! user-supplied model's autograd — the JVPs and VJPs that build the pullback
//! product `V ↦ G V` and the quadratic forms `zᵀ G z` — which is torch's job and
//! stays in torch. The subspace iteration, the Rayleigh–Ritz step and the deflated
//! trace estimate are linear algebra, and they run in
//! `gam::linalg::randomized_eigen`; these entries only carry the caller's products
//! across the boundary (#2899 P33).

use gam::linalg::randomized_eigen::{deflated_trace, psd_top_eigenpairs_by_subspace_iteration};
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::py_value_error;

/// Call `callback` on a float64 block and read back a float64 result.
fn call_on_block<T>(
    py: Python<'_>,
    callback: &Bound<'_, PyAny>,
    block: ArrayView2<'_, f64>,
    read: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>,
) -> Result<T, String> {
    let argument = block.to_owned().into_pyarray(py);
    let result = callback
        .call1((argument,))
        .map_err(|error| format!("operator callback raised: {error}"))?;
    read(&result).map_err(|error| format!("operator callback returned an unreadable result: {error}"))
}

/// The top `rank` eigenpairs of the PSD operator `matvec` on `ℝ^dim`, eigenvalues
/// descending, by randomized subspace iteration with `oversample` extra columns and
/// `power_steps` power steps, seeded by `seed`
/// ([`psd_top_eigenpairs_by_subspace_iteration`]). `matvec` maps a float64
/// `(dim, m)` array to `G` applied to it.
#[pyfunction]
fn psd_top_eigenpairs<'py>(
    py: Python<'py>,
    matvec: Bound<'py, PyAny>,
    dim: usize,
    rank: usize,
    oversample: usize,
    power_steps: usize,
    seed: u64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let pairs = psd_top_eigenpairs_by_subspace_iteration(
        dim,
        rank,
        oversample,
        power_steps,
        seed,
        |block| {
            call_on_block(py, &matvec, block, |result| {
                Ok::<Array2<f64>, PyErr>(
                    result.extract::<PyReadonlyArray2<f64>>()?.as_array().to_owned(),
                )
            })
        },
    )
    .map_err(py_value_error)?;
    Ok((
        pairs.values.into_pyarray(py).unbind(),
        pairs.vectors.into_pyarray(py).unbind(),
    ))
}

/// `trace(P G P)` for `P = I − Q Qᵀ` with `Q = basis` orthonormal: exact from the
/// identity columns when `probes ≥ dim`, otherwise Hutchinson over `probes`
/// Rademacher vectors seeded by `seed` ([`deflated_trace`]). `quadratic_forms` maps
/// a float64 `(dim, s)` array to the `s` values `zᵢᵀ G zᵢ`.
#[pyfunction]
fn psd_deflated_trace<'py>(
    py: Python<'py>,
    quadratic_forms: Bound<'py, PyAny>,
    basis: PyReadonlyArray2<'py, f64>,
    probes: usize,
    seed: u64,
) -> PyResult<f64> {
    deflated_trace(basis.as_array(), probes, seed, |block| {
        call_on_block(py, &quadratic_forms, block, |result| {
            Ok::<Array1<f64>, PyErr>(
                result.extract::<PyReadonlyArray1<f64>>()?.as_array().to_owned(),
            )
        })
    })
    .map_err(py_value_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(psd_top_eigenpairs, module)?)?;
    module.add_function(wrap_pyfunction!(psd_deflated_trace, module)?)?;
    Ok(())
}
