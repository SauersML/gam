use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
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

fn null_distribution_dict<'py>(
    py: Python<'py>,
    calibration: &gam::terms::sae::null_battery::ClaimNullCalibration,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("claim", &calibration.claim)?;
    out.set_item("observed_statistic", calibration.observed_statistic)?;
    out.set_item("null_pvalue", calibration.null_pvalue)?;
    out.set_item("null_z", calibration.null_z)?;
    out.set_item("claimed_snr", calibration.claimed_snr)?;
    out.set_item(
        "claimed_false_positive_rate",
        calibration.claimed_false_positive_rate,
    )?;
    out.set_item("spikein_power", calibration.spikein_power)?;
    let nulls = PyList::empty(py);
    for summary in &calibration.null_distribution {
        let row = PyDict::new(py);
        row.set_item("kind", summary.kind.as_str())?;
        row.set_item("observed", summary.observed)?;
        row.set_item("n", summary.n)?;
        row.set_item("mean", summary.mean)?;
        row.set_item("sd", summary.sd)?;
        row.set_item("min", summary.min)?;
        row.set_item("q25", summary.q25)?;
        row.set_item("median", summary.median)?;
        row.set_item("q75", summary.q75)?;
        row.set_item("max", summary.max)?;
        row.set_item("z", summary.z)?;
        row.set_item("p_value", summary.p_value)?;
        row.set_item("samples", summary.samples.clone())?;
        nulls.append(row)?;
    }
    out.set_item("null_distribution", nulls)?;
    Ok(out)
}

fn attach_null_fields<'py>(
    py: Python<'py>,
    atom: &Bound<'py, PyDict>,
    calibration: Option<&gam::terms::sae::null_battery::ClaimNullCalibration>,
) -> PyResult<()> {
    match calibration {
        Some(calibration) => {
            atom.set_item("observed_statistic", calibration.observed_statistic)?;
            atom.set_item("null_pvalue", calibration.null_pvalue)?;
            atom.set_item("null_z", calibration.null_z)?;
            atom.set_item("spikein_power", calibration.spikein_power)?;
            atom.set_item("null_calibration", null_distribution_dict(py, calibration)?)?;
        }
        None => {
            atom.set_item("observed_statistic", py.None())?;
            atom.set_item("null_pvalue", py.None())?;
            atom.set_item("null_z", py.None())?;
            atom.set_item("spikein_power", py.None())?;
            atom.set_item("null_calibration", py.None())?;
        }
    }
    Ok(())
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
                atom.set_item("covering_side", report.covering_side.as_str())?;
                atom.set_item("measured_betti", betti_signature_dict(py, report.measured_betti)?)?;
                atom.set_item("expected_betti", betti_signature_dict(py, report.expected_betti)?)?;
                attach_null_fields(py, &atom, report.null_calibration.as_ref())?;
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
                attach_null_fields(py, &atom, None)?;
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
    use gam::terms::sae::null_battery::{
        ClaimNullCalibration, NullKind, NullSummary, SpikeInRocPoint, SpikeInRocThreshold,
    };
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
                null_calibration: Some(ClaimNullCalibration {
                    claim: "atom 0 topology".to_string(),
                    observed_statistic: 0.91,
                    null_pvalue: 0.02,
                    null_z: 3.4,
                    claimed_snr: 1.0,
                    claimed_false_positive_rate: 0.05,
                    spikein_power: 0.875,
                    null_distribution: vec![NullSummary {
                        kind: NullKind::PhaseRandomized,
                        observed: 0.91,
                        n: 4,
                        mean: 0.12,
                        sd: 0.04,
                        min: 0.08,
                        q25: 0.10,
                        median: 0.12,
                        q75: 0.14,
                        max: 0.16,
                        z: 19.75,
                        p_value: 0.02,
                        samples: vec![0.08, 0.11, 0.13, 0.16],
                    }],
                    spike_in_roc: vec![SpikeInRocPoint {
                        snr: 1.0,
                        trials: 8,
                        mean_stat: 0.8,
                        promoted_fraction: 1.0,
                        topology_accept_fraction: 0.875,
                        roc: vec![SpikeInRocThreshold {
                            false_positive_rate: 0.05,
                            threshold: 0.3,
                            true_positive_rate: 0.875,
                        }],
                    }],
                }),
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
            let null_pvalue = atom
                .get_item("null_pvalue")
                .expect("read null_pvalue")
                .expect("null_pvalue present")
                .extract::<f64>()
                .expect("null_pvalue f64");
            assert_eq!(null_pvalue, 0.02);
            let spikein_power = atom
                .get_item("spikein_power")
                .expect("read spikein_power")
                .expect("spikein_power present")
                .extract::<f64>()
                .expect("spikein_power f64");
            assert_eq!(spikein_power, 0.875);
            let kind = atom
                .get_item("kind")
                .expect("read kind")
                .expect("kind present")
                .extract::<String>()
                .expect("kind string");
            assert_eq!(kind, "loop");
            assert!(atom.get_item("measured_betti").expect("read measured_betti").is_some());
            assert!(atom.get_item("expected_betti").expect("read expected_betti").is_some());
            assert!(atom.get_item("contested").expect("read contested").is_some());
        });
    }
}
