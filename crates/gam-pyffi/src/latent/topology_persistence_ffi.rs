use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;

fn betti_signature_dict<'py>(
    py: Python<'py>,
    betti: gam::terms::sae::manifold::BettiSignature,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("b0", betti.b0)?;
    d.set_item("b1", betti.b1)?;
    match betti.b2 {
        Some(b2) => d.set_item("b2", b2)?,
        None => d.set_item("b2", py.None())?,
    }
    Ok(d)
}

pub(crate) fn sae_topology_persistence_dict<'py>(
    py: Python<'py>,
    persistence: &[Option<gam::terms::sae::manifold::AtomTopologyPersistence>],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let atoms = pyo3::types::PyList::empty(py);
    for (atom_idx, entry) in persistence.iter().enumerate() {
        let atom = PyDict::new(py);
        atom.set_item("atom", atom_idx)?;
        match entry {
            Some(report) => {
                let inferred = if report.measured_betti.b0 != 1 {
                    "disconnected"
                } else if report.measured_betti.b2 == Some(1) && report.measured_betti.b1 == 0 {
                    "sphere"
                } else if report.measured_betti.b1 == 2 {
                    "torus"
                } else if report.measured_betti.b1 == 1 {
                    "loop"
                } else {
                    "contractible"
                };
                let summary = PyDict::new(py);
                summary.set_item("h0_bars", report.h0.len())?;
                summary.set_item("h1_bars", report.h1.len())?;
                summary.set_item("h2_bars", report.h2.len())?;
                summary.set_item("dominant_h1_persistence", report.dominant_h1_persistence)?;
                summary.set_item("dominant_h2_persistence", report.dominant_h2_persistence)?;
                atom.set_item("raced_kind", crate::sae_atom_basis_kind_name(&report.raced_kind))?;
                atom.set_item("support_size", report.support_size)?;
                atom.set_item("landmark_count", report.landmark_count)?;
                atom.set_item("stability_band", report.stability_band.as_str())?;
                atom.set_item("support_mass", report.support_mass)?;
                atom.set_item("effective_n", report.effective_n)?;
                atom.set_item("support_ess", report.support_ess)?;
                atom.set_item("measured_betti", betti_signature_dict(py, report.measured_betti)?)?;
                atom.set_item("expected_betti", betti_signature_dict(py, report.expected_betti)?)?;
                atom.set_item("inferred_kind", inferred)?;
                atom.set_item("persistence_summary", summary)?;
                atom.set_item("contested", report.contested)?;
                atom.set_item("note", &report.note)?;
            }
            None => {
                atom.set_item("raced_kind", py.None())?;
                atom.set_item("support_size", py.None())?;
                atom.set_item("landmark_count", py.None())?;
                atom.set_item("stability_band", py.None())?;
                atom.set_item("measured_betti", py.None())?;
                atom.set_item("expected_betti", py.None())?;
                atom.set_item("inferred_kind", py.None())?;
                atom.set_item("contested", py.None())?;
            }
        }
        atoms.append(atom)?;
    }
    d.set_item("atoms", atoms)?;
    Ok(d)
}
