use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
                atom.set_item(
                    "raced_kind",
                    crate::sae_atom_basis_kind_name(&report.raced_kind),
                )?;
                atom.set_item("support_size", report.support_size)?;
                atom.set_item("landmark_count", report.landmark_count)?;
                atom.set_item("stability_band", report.stability_band.as_str())?;
                atom.set_item("support_mass", report.support_mass)?;
                atom.set_item("effective_n", report.effective_n)?;
                atom.set_item("support_ess", report.support_ess)?;
                atom.set_item("covering_side", report.covering_side.as_str())?;
                atom.set_item(
                    "measured_betti",
                    betti_signature_dict(py, report.measured_betti)?,
                )?;
                atom.set_item(
                    "expected_betti",
                    betti_signature_dict(py, report.expected_betti)?,
                )?;
                atom.set_item("kind", inferred)?;
                atom.set_item("persistence_summary", summary)?;
                atom.set_item("contested", report.contested)?;
                atom.set_item("note", &report.note)?;
            }
            None => {
                atom.set_item("raced_kind", py.None())?;
                atom.set_item("support_size", py.None())?;
                atom.set_item("landmark_count", py.None())?;
                atom.set_item("stability_band", py.None())?;
                atom.set_item("covering_side", py.None())?;
                atom.set_item("measured_betti", py.None())?;
                atom.set_item("expected_betti", py.None())?;
                atom.set_item("kind", py.None())?;
                atom.set_item("contested", py.None())?;
            }
        }
        atoms.append(atom)?;
    }
    d.set_item("atoms", atoms)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::terms::sae::manifold::{
        AtlasCoveringSide, AtomTopologyPersistence, BettiSignature, PersistenceBar,
        PersistenceStabilityBand, SaeAtomBasisKind,
    };
    use pyo3::types::PyList;

    #[test]
    fn topology_persistence_payload_surfaces_covering_side() {
        Python::attach(|py| {
            let report = AtomTopologyPersistence {
                raced_kind: SaeAtomBasisKind::Periodic,
                support_size: 48,
                landmark_count: 48,
                stability_band: PersistenceStabilityBand::BelowLandmarkCap,
                covering_side: AtlasCoveringSide::AtOrAboveCoveringNumber,
                support_mass: 48.0,
                effective_n: 48.0,
                support_ess: 48.0,
                measured_betti: BettiSignature {
                    b0: 1,
                    b1: 1,
                    b2: None,
                },
                expected_betti: BettiSignature {
                    b0: 1,
                    b1: 1,
                    b2: None,
                },
                dominant_h1_persistence: f64::INFINITY,
                dominant_h2_persistence: 0.0,
                h0: vec![PersistenceBar {
                    birth: 0.0,
                    death: f64::INFINITY,
                }],
                h1: Vec::new(),
                h2: Vec::new(),
                contested: false,
                note: "topology agrees".to_string(),
                null_calibration: None,
            };
            let payload = sae_topology_persistence_dict(py, &[Some(report)])
                .expect("topology persistence payload");
            let atoms_any = payload
                .get_item("atoms")
                .expect("read atoms key")
                .expect("atoms key present");
            let atoms = atoms_any.cast::<PyList>().expect("atoms list");
            let atom_any = atoms.get_item(0).expect("first atom row");
            let atom = atom_any.cast::<PyDict>().expect("atom row dict");
            let covering_side = atom
                .get_item("covering_side")
                .expect("read covering_side")
                .expect("covering_side present")
                .extract::<String>()
                .expect("covering_side string");
            assert_eq!(covering_side, "at_or_above_covering_number");
            let kind = atom
                .get_item("kind")
                .expect("read kind")
                .expect("kind present")
                .extract::<String>()
                .expect("kind string");
            assert_eq!(kind, "loop");
            assert!(
                atom.get_item("measured_betti")
                    .expect("read measured_betti")
                    .is_some()
            );
            assert!(
                atom.get_item("expected_betti")
                    .expect("read expected_betti")
                    .is_some()
            );
            assert!(
                atom.get_item("contested")
                    .expect("read contested")
                    .is_some()
            );
        });
    }
}
