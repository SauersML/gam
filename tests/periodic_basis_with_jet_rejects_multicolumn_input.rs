#[path = "../crates/gam-pyffi/src/lib.rs"]
mod pyffi;

#[test]
fn periodic_basis_with_jet_rejects_multicolumn_input() {
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        let t = ndarray::arr2(&[[0.1_f64, 0.9_f64], [0.3_f64, 0.7_f64]]);
        let py_t = numpy::PyArray2::from_owned_array(py, t).readonly();

        let result = pyffi::basis_with_jet(py, "periodic", py_t, { let params = pyo3::types::PyDict::new(py); params.set_item("n_harmonics", 2_usize).expect("set n_harmonics"); params });

        assert!(
            result.is_err(),
            "BUG: basis_with_jet(kind='periodic') silently ignores extra input columns instead of rejecting non-(N,1) input"
        );
    });
}
