//! `linear_state_quotient`: [`LinearStateQuotient`] on the wire.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{ArrayD, ArrayView2};
use serde::{Deserialize, Serialize};

use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, matrix, output, reserve};
use crate::state::{LinearStateQuotient, SpectralNormBounds};

/// The readouts `M_k` (`p_k × d`) and transitions `T_a` (`d × d`), each the id of an
/// input array, and the chart to report.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LinearStateQuotientRequest {
    pub readouts: Vec<String>,
    pub transitions: Vec<String>,
    pub chart: LinearChart,
}

/// Which chart the owner measures.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LinearChart {
    /// The readouts' row span closed under the transitions
    /// (`LinearStateQuotient::close`). A struct variant with no fields, so a stray
    /// field is refused (serde ignores fields beside a tagged unit variant).
    Close {},
    /// A declared chart `Q` (`r × d`), measured as given
    /// (`LinearStateQuotient::measure`).
    Declared { tensor: String },
}

/// [`LinearStateQuotient`] on the wire. Every bound holds at every state, scaled by
/// the state's norm (see the owner). The chart's row count is not a certified
/// dimension; the measured bounds are the certificate.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LinearStateQuotientReport {
    /// Id of the output array `Q` (`r × d`).
    pub chart: String,
    /// `r`, the chart's rows.
    pub rows: usize,
    /// Ids of `Ψ_k = M_k Qᵀ` (`p_k × r`), in readout order.
    pub readout_maps: Vec<String>,
    /// Ids of `G_a = Q T_a Qᵀ` (`r × r`), in transition order.
    pub descended: Vec<String>,
    /// `‖M_k − Ψ_k Q‖₂`.
    pub readout_bounds: Vec<SpectralNormBoundsReport>,
    /// `‖Q T_a − G_a Q‖₂`.
    pub quotient_bounds: Vec<SpectralNormBoundsReport>,
    /// `‖T_a Qᵀ − Qᵀ G_a‖₂`.
    pub realization_bounds: Vec<SpectralNormBoundsReport>,
    /// `‖Q Qᵀ − I_r‖₂`.
    pub section_bounds: SpectralNormBoundsReport,
}

/// [`SpectralNormBounds`] on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SpectralNormBoundsReport {
    /// A positive value certifies a nonzero exact matrix.
    pub lower: f64,
    /// The exact norm is at most this.
    pub upper: f64,
}

pub(super) fn bounds_report(
    field: &'static str,
    bounds: SpectralNormBounds,
) -> Result<SpectralNormBoundsReport, MpdSurfaceError> {
    Ok(SpectralNormBoundsReport {
        lower: finite(field, bounds.lower)?,
        upper: finite(field, bounds.upper)?,
    })
}

fn matrices<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    ids: &[String],
) -> Result<Vec<ArrayView2<'a, f64>>, MpdSurfaceError> {
    ids.iter().map(|id| matrix(tensors, id)).collect()
}

pub(super) fn run(
    request: LinearStateQuotientRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let readouts = matrices(tensors, &request.readouts)?;
    let transitions = matrices(tensors, &request.transitions)?;
    let quotient = match &request.chart {
        LinearChart::Close {} => LinearStateQuotient::close(governor, &readouts, &transitions),
        LinearChart::Declared { tensor } => {
            let declared = matrix(tensors, tensor)?;
            // The owner takes the chart by value, so the surface copies it.
            let copy = reserve(
                governor,
                declared.nrows(),
                declared.ncols(),
                1,
                "linear state quotient: declared chart",
            )?;
            let result =
                LinearStateQuotient::measure(governor, declared.to_owned(), &readouts, &transitions);
            drop(copy);
            result
        }
    }
    .map_err(MpdSurfaceError::State)?;
    project(quotient)
}

pub(super) fn project(quotient: LinearStateQuotient) -> Result<MpdOutput, MpdSurfaceError> {
    let bounds = |field, list: Vec<SpectralNormBounds>| {
        list.into_iter()
            .map(|bounds| bounds_report(field, bounds))
            .collect::<Result<Vec<_>, _>>()
    };
    let readout_bounds = bounds("readout_bounds", quotient.readout_bounds)?;
    let quotient_bounds = bounds("quotient_bounds", quotient.quotient_bounds)?;
    let realization_bounds = bounds("realization_bounds", quotient.realization_bounds)?;
    let section_bounds = bounds_report("section_bounds", quotient.section_bounds)?;
    let mut arrays = BTreeMap::new();
    let rows = quotient.chart.nrows();
    arrays.insert("chart".to_string(), quotient.chart.into_dyn());
    let mut named = |prefix: &str, list: Vec<ndarray::Array2<f64>>| -> Vec<String> {
        list.into_iter()
            .enumerate()
            .map(|(index, map)| {
                let id = format!("{prefix}/{index}");
                arrays.insert(id.clone(), map.into_dyn());
                id
            })
            .collect()
    };
    let readout_maps = named("readout_maps", quotient.readout_maps);
    let descended = named("descended", quotient.descended);
    Ok(output(
        MpdResult::LinearStateQuotient(LinearStateQuotientReport {
            chart: "chart".to_string(),
            rows,
            readout_maps,
            descended,
            readout_bounds,
            quotient_bounds,
            realization_bounds,
            section_bounds,
        }),
        arrays,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::MpdOperation;
    use super::super::MpdRequest;
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::test_support::test_governor;
    use ndarray::{Array2, array};

    /// A shear that moves the second coordinate into the first, observed on the first
    /// only: the observable chart is the whole plane. A third coordinate is never
    /// reached.
    fn fixture() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([
            ("m".to_string(), array![[1.0, 0.0, 0.0]].into_dyn()),
            (
                "t".to_string(),
                array![[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]].into_dyn(),
            ),
            (
                "q".to_string(),
                array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into_dyn(),
            ),
        ])
    }

    fn request(chart: &str) -> String {
        request_json(&format!(
            r#"{{"kind": "linear_state_quotient", "readouts": ["m"], "transitions": ["t"], "chart": {chart}}}"#
        ))
    }

    fn report(output: &MpdOutput) -> &LinearStateQuotientReport {
        let MpdResult::LinearStateQuotient(report) = &output.report.result else {
            panic!("expected a linear state quotient, got {:?}", output.report.result);
        };
        report
    }

    fn assert_projects(output: &MpdOutput, owner: &LinearStateQuotient) {
        let report = report(output);
        assert_eq!(output.arrays[&report.chart], owner.chart.clone().into_dyn());
        assert_eq!(report.rows, owner.chart.nrows());
        assert_eq!(report.readout_maps.len(), owner.readout_maps.len());
        for (id, map) in report.readout_maps.iter().zip(&owner.readout_maps) {
            assert_eq!(output.arrays[id], map.clone().into_dyn());
        }
        assert_eq!(report.descended.len(), owner.descended.len());
        for (id, map) in report.descended.iter().zip(&owner.descended) {
            assert_eq!(output.arrays[id], map.clone().into_dyn());
        }
        let wire = |bounds: &SpectralNormBounds| SpectralNormBoundsReport {
            lower: bounds.lower,
            upper: bounds.upper,
        };
        assert_eq!(
            report.readout_bounds,
            owner.readout_bounds.iter().map(wire).collect::<Vec<_>>()
        );
        assert_eq!(
            report.quotient_bounds,
            owner.quotient_bounds.iter().map(wire).collect::<Vec<_>>()
        );
        assert_eq!(
            report.realization_bounds,
            owner.realization_bounds.iter().map(wire).collect::<Vec<_>>()
        );
        assert_eq!(report.section_bounds, wire(&owner.section_bounds));
        let mut named: Vec<&String> = report
            .readout_maps
            .iter()
            .chain(&report.descended)
            .chain(std::iter::once(&report.chart))
            .collect();
        named.sort();
        assert_eq!(named, output.arrays.keys().collect::<Vec<_>>());
    }

    #[test]
    fn closed_quotient_report_is_the_owner_result_field_for_field() {
        let tensors = fixture();
        let readout = matrix(&tensors, "m").expect("m");
        let transition = matrix(&tensors, "t").expect("t");
        let direct = LinearStateQuotient::close(test_governor(), &[readout], &[transition])
            .expect("owner closure");
        assert_eq!(direct.chart.nrows(), 2, "the shear reaches exactly the plane");
        let output = run_parameter_decomposition(&request(r#"{"kind": "close"}"#), &tensors, test_governor())
            .expect("surface run");
        assert_projects(&output, &direct);
        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert_eq!(json["result"]["kind"], "linear_state_quotient");
        assert_eq!(json["result"]["descended"][0], "descended/0");
    }

    #[test]
    fn declared_chart_report_is_the_owner_measurement_field_for_field() {
        let tensors = fixture();
        let readout = matrix(&tensors, "m").expect("m");
        let transition = matrix(&tensors, "t").expect("t");
        let declared = matrix(&tensors, "q").expect("q").to_owned();
        let direct = LinearStateQuotient::measure(test_governor(), declared, &[readout], &[transition])
            .expect("owner measurement");
        let output = run_parameter_decomposition(
            &request(r#"{"kind": "declared", "tensor": "q"}"#),
            &tensors,
            test_governor(),
        )
        .expect("surface run");
        assert_projects(&output, &direct);
        // The first coordinate alone is not closed under the shear: its quotient bound
        // certifies a nonzero defect.
        let mut unclosed = fixture();
        unclosed.insert("q".to_string(), array![[1.0, 0.0, 0.0]].into_dyn());
        let output = run_parameter_decomposition(
            &request(r#"{"kind": "declared", "tensor": "q"}"#),
            &unclosed,
            test_governor(),
        )
        .expect("surface run");
        assert!(report(&output).quotient_bounds[0].lower > 0.0, "{:?}", report(&output));
    }

    #[test]
    fn a_quotient_request_the_owner_refuses_reaches_the_caller() {
        let tensors = fixture();
        assert!(run_parameter_decomposition(&request(r#"{"kind": "close"}"#), &tensors, test_governor()).is_ok());
        // No readouts: the owner refuses an empty family.
        let empty = request_json(
            r#"{"kind": "linear_state_quotient", "readouts": [], "transitions": ["t"], "chart": {"kind": "close"}}"#,
        );
        assert!(matches!(
            run_parameter_decomposition(&empty, &tensors, test_governor()),
            Err(MpdSurfaceError::State(_))
        ));
        // A transition of the wrong width.
        let mut misshapen = fixture();
        misshapen.insert("t".to_string(), Array2::<f64>::eye(2).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"kind": "close"}"#), &misshapen, test_governor()),
            Err(MpdSurfaceError::State(_))
        ));
        // A declared chart of the wrong width.
        let mut narrow = fixture();
        narrow.insert("q".to_string(), Array2::<f64>::eye(2).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"kind": "declared", "tensor": "q"}"#), &narrow, test_governor()),
            Err(MpdSurfaceError::State(_))
        ));
        // A chart kind the owner does not declare, and a stray field.
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"kind": "guess"}"#), &tensors, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            run_parameter_decomposition(
                &request(r#"{"kind": "close", "tolerance": 1e-3}"#),
                &tensors,
                test_governor()
            ),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        let stray = request_json(
            r#"{"kind": "linear_state_quotient", "readouts": ["m"], "transitions": ["t"], "chart": {"kind": "close"}, "rank": 2}"#,
        );
        assert!(matches!(
            MpdRequest::from_json(&stray),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            MpdRequest::from_json(&request(r#"{"kind": "close"}"#)).expect("parses").operation,
            MpdOperation::LinearStateQuotient(_)
        ));
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"kind": "declared", "tensor": "absent"}"#), &tensors, test_governor()),
            Err(MpdSurfaceError::MissingTensor { .. })
        ));
    }
}
