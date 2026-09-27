//! `cyclic_planes`, `frequency_edit` and `plane_program_code`: the
//! [`crate::parameter_decomposition::cyclic_action`] owners on the wire.
//!
//! The owners allocate without a governor, so the surface reserves what they form
//! before calling them.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array2, ArrayD};
use serde::{Deserialize, Serialize};

use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, matrix, output, reserve};
use crate::parameter_decomposition::cyclic_action::{
    CyclicFrequencyEdit, CyclicPlanes, PlaneProgramCode, RowCycle, cyclic_planes, frequency_edit,
    plane_program_code,
};
use crate::parameter_decomposition::precision::{DecodableArtifact, DeclaredPrecision};

/// A table (`rows × d`) and a declared successor map over its rows.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicPlanesRequest {
    /// Id of the input table `E`.
    pub table: String,
    /// `successor[r]` is the row that follows row `r`; a fixed row maps to itself. The
    /// moved rows must form one odd cycle of at least three rows.
    pub successor: Vec<usize>,
}

/// [`CyclicPlanes`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CyclicPlanesReport {
    /// The table row at each cycle position; position 0 is the smallest moved row.
    pub cycle_rows: Vec<usize>,
    /// `m = (p − 1)/2`.
    pub plane_count: usize,
    /// Id of `c_0`, the mean of the cycled rows (`d`).
    pub mean: String,
    /// Id of `[u_1c, u_1s, u_2c, u_2s, …]` (`d × 2m`).
    pub planes: String,
    /// `‖u_kc‖² + ‖u_ks‖²` for `k = 1..m`.
    pub power: Vec<f64>,
}

/// Where the plane basis `U_S` (`d × 2|S|`) of a frequency edit comes from.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicBasis {
    /// The closed-form planes of `table` under the declared cycle
    /// (`CyclicPlanes::basis`).
    ClosedForm { table: String },
    /// A declared basis, e.g. a decoded plane program.
    Declared { tensor: String },
}

/// The frequency edit of `frequencies` at `shift`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencyEditRequest {
    pub successor: Vec<usize>,
    pub basis: CyclicBasis,
    /// Non-empty, strictly ascending, inside `1..=m`.
    pub frequencies: Vec<usize>,
    pub shift: usize,
}

/// [`CyclicFrequencyEdit`] on the wire: the edited table is `E + left · rightᵀ`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FrequencyEditReport {
    pub frequencies: Vec<usize>,
    pub shift: usize,
    /// Id of `left` (`rows × 2|S|`).
    pub left: String,
    /// Id of `right = U_S` (`d × 2|S|`).
    pub right: String,
}

/// The plane program of `frequencies` over the closed-form planes, at the declared
/// precision `2^-fraction_bits`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlaneProgramCodeRequest {
    pub table: String,
    pub successor: Vec<usize>,
    pub frequencies: Vec<usize>,
    pub fraction_bits: i32,
}

/// [`PlaneProgramCode`] on the wire, with the artifact its decoder rebuilds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaneProgramCodeReport {
    /// `L_int(|S| + 1) + ⌈log₂ C(m, |S|)⌉`.
    pub subset_bits: u64,
    /// The basis reals written as one lattice message.
    pub basis_bits: u64,
    /// `subset_bits + basis_bits`.
    pub total_bits: u64,
    pub fraction_bits: i32,
    /// Id of the decoded basis `U_S` (`d × 2|S|`): what fidelity must be measured on.
    pub decoded_basis: String,
}

fn cycle(successor: &[usize]) -> Result<RowCycle, MpdSurfaceError> {
    RowCycle::from_successor(successor).map_err(MpdSurfaceError::CyclicAction)
}

fn planes_of(
    tensors: &BTreeMap<String, ArrayD<f64>>,
    table: &str,
    cycle: &RowCycle,
    governor: &MemoryGovernor,
) -> Result<CyclicPlanes, MpdSurfaceError> {
    let table = matrix(tensors, table)?;
    // `c_0` and the `d × 2m` planes.
    let formed = reserve(
        governor,
        table.ncols(),
        2 * cycle.plane_count() + 1,
        1,
        "cyclic planes",
    )?;
    let planes = cyclic_planes(table, cycle).map_err(MpdSurfaceError::CyclicAction);
    drop(formed);
    planes
}

pub(super) fn run_planes(
    request: CyclicPlanesRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let cycle = cycle(&request.successor)?;
    let planes = planes_of(tensors, &request.table, &cycle, governor)?;
    project_planes(planes)
}

pub(super) fn project_planes(planes: CyclicPlanes) -> Result<MpdOutput, MpdSurfaceError> {
    let power = planes
        .power
        .iter()
        .map(|&value| finite("power", value))
        .collect::<Result<Vec<_>, _>>()?;
    let report = CyclicPlanesReport {
        cycle_rows: planes.cycle().rows().to_vec(),
        plane_count: planes.cycle().plane_count(),
        mean: "mean".to_string(),
        planes: "planes".to_string(),
        power,
    };
    let arrays = BTreeMap::from([
        ("mean".to_string(), planes.mean.into_dyn()),
        ("planes".to_string(), planes.planes.into_dyn()),
    ]);
    Ok(output(MpdResult::CyclicPlanes(report), arrays))
}

pub(super) fn run_edit(
    request: FrequencyEditRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let cycle = cycle(&request.successor)?;
    let basis = match &request.basis {
        CyclicBasis::ClosedForm { table } => {
            let planes = planes_of(tensors, table, &cycle, governor)?;
            planes
                .basis(&request.frequencies)
                .map_err(MpdSurfaceError::CyclicAction)?
        }
        CyclicBasis::Declared { tensor } => {
            let declared = matrix(tensors, tensor)?;
            // The owner copies the basis into `right`.
            let copy = reserve(governor, declared.nrows(), declared.ncols(), 1, "frequency edit: basis")?;
            let edited = edit(&cycle, declared, &request, governor);
            drop(copy);
            return edited;
        }
    };
    edit(&cycle, basis.view(), &request, governor)
}

fn edit(
    cycle: &RowCycle,
    basis: ndarray::ArrayView2<'_, f64>,
    request: &FrequencyEditRequest,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let left = reserve(
        governor,
        cycle.table_rows(),
        2 * request.frequencies.len(),
        1,
        "frequency edit: left factor",
    )?;
    let edit = frequency_edit(cycle, basis, &request.frequencies, request.shift)
        .map_err(MpdSurfaceError::CyclicAction);
    drop(left);
    Ok(project_edit(edit?))
}

pub(super) fn project_edit(edit: CyclicFrequencyEdit) -> MpdOutput {
    let report = FrequencyEditReport {
        frequencies: edit.frequencies,
        shift: edit.shift,
        left: "left".to_string(),
        right: "right".to_string(),
    };
    let arrays = BTreeMap::from([
        ("left".to_string(), edit.left.into_dyn()),
        ("right".to_string(), edit.right.into_dyn()),
    ]);
    output(MpdResult::FrequencyEdit(report), arrays)
}

pub(super) fn run_code(
    request: PlaneProgramCodeRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let cycle = cycle(&request.successor)?;
    let planes = planes_of(tensors, &request.table, &cycle, governor)?;
    let precision =
        DeclaredPrecision::new(request.fraction_bits).map_err(MpdSurfaceError::InvalidRequest)?;
    // The basis, its column-major reals, the lattice indices and the decoded reals.
    let formed = reserve(
        governor,
        planes.planes.nrows(),
        2 * request.frequencies.len(),
        4,
        "plane program code",
    )?;
    let projected = plane_program_code(&planes, &request.frequencies, precision)
        .map_err(MpdSurfaceError::CyclicAction)
        .and_then(|code| project_code(code, planes.planes.nrows()));
    drop(formed);
    projected
}

/// The decoder's reals are `U_S` column-major (`cyclic_action::plane_program_code`),
/// so they are laid back out as `d × 2|S|`.
pub(super) fn project_code(
    code: PlaneProgramCode,
    width: usize,
) -> Result<MpdOutput, MpdSurfaceError> {
    let decoded = code.basis.decode().map_err(MpdSurfaceError::InvalidRequest)?;
    let columns = if width == 0 { 0 } else { decoded.len() / width };
    let decoded = Array2::from_shape_vec((columns, width), decoded)
        .map_err(|error| MpdSurfaceError::TensorShape {
            tensor: "decoded_basis".to_string(),
            reason: error.to_string(),
        })?
        .reversed_axes()
        .as_standard_layout()
        .into_owned();
    let report = PlaneProgramCodeReport {
        subset_bits: code.subset_bits,
        basis_bits: code.basis_bits,
        total_bits: code.total_bits(),
        fraction_bits: code.basis.precision().fraction_bits(),
        decoded_basis: "decoded_basis".to_string(),
    };
    Ok(output(
        MpdResult::PlaneProgramCode(report),
        BTreeMap::from([("decoded_basis".to_string(), decoded.into_dyn())]),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::parameter_decomposition::cyclic_action::CyclicActionError;
    use crate::parameter_decomposition::test_support::test_governor;

    /// A 7-cycle over rows `0, 5, 2, 7, 1, 6, 4` with row 3 fixed, on a table whose
    /// entries are distinct.
    const CYCLE: [usize; 7] = [0, 5, 2, 7, 1, 6, 4];

    fn successor() -> Vec<usize> {
        let mut map: Vec<usize> = (0..8).collect();
        for (position, &row) in CYCLE.iter().enumerate() {
            map[row] = CYCLE[(position + 1) % CYCLE.len()];
        }
        map
    }

    fn successor_json() -> String {
        serde_json::to_string(&successor()).expect("successor json")
    }

    fn table() -> Array2<f64> {
        Array2::from_shape_fn((8, 5), |(row, col)| {
            ((row * 7 + col * 3) % 11) as f64 - 0.25 * col as f64
        })
    }

    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([("e".to_string(), table().into_dyn())])
    }

    fn owner_planes() -> CyclicPlanes {
        cyclic_planes(
            table().view(),
            &RowCycle::from_successor(&successor()).expect("cycle"),
        )
        .expect("owner planes")
    }

    #[test]
    fn cyclic_planes_report_is_the_owner_result_field_for_field() {
        let direct = owner_planes();
        let output = run_parameter_decomposition(
            &request_json(&format!(
                r#"{{"kind": "cyclic_planes", "table": "e", "successor": {}}}"#,
                successor_json()
            )),
            &tensors(),
            test_governor(),
        )
        .expect("surface run");
        let MpdResult::CyclicPlanes(report) = &output.report.result else {
            panic!("expected cyclic planes, got {:?}", output.report.result);
        };
        assert_eq!(report.cycle_rows, direct.cycle().rows());
        assert_eq!(report.plane_count, direct.cycle().plane_count());
        assert_eq!(report.power, direct.power);
        assert_eq!(output.arrays[&report.mean], direct.mean.clone().into_dyn());
        assert_eq!(output.arrays[&report.planes], direct.planes.clone().into_dyn());
        assert_eq!(output.arrays.len(), 2);
    }

    #[test]
    fn frequency_edit_report_is_the_owner_result_field_for_field() {
        let planes = owner_planes();
        let frequencies = [1, 3];
        let basis = planes.basis(&frequencies).expect("basis");
        let direct = frequency_edit(planes.cycle(), basis.view(), &frequencies, 4).expect("owner edit");
        let closed = format!(
            r#"{{"kind": "frequency_edit", "successor": {}, "basis": {{"kind": "closed_form", "table": "e"}}, "frequencies": [1, 3], "shift": 4}}"#,
            successor_json()
        );
        let declared = closed.replace(
            r#"{"kind": "closed_form", "table": "e"}"#,
            r#"{"kind": "declared", "tensor": "u"}"#,
        );
        let mut inputs = tensors();
        inputs.insert("u".to_string(), basis.clone().into_dyn());
        for json in [closed, declared] {
            let output = run_parameter_decomposition(&request_json(&json), &inputs, test_governor())
                .expect("surface run");
            let MpdResult::FrequencyEdit(report) = &output.report.result else {
                panic!("expected a frequency edit, got {:?}", output.report.result);
            };
            assert_eq!(report.frequencies, direct.frequencies);
            assert_eq!(report.shift, direct.shift);
            assert_eq!(output.arrays[&report.left], direct.left.clone().into_dyn());
            assert_eq!(output.arrays[&report.right], direct.right.clone().into_dyn());
        }
    }

    #[test]
    fn plane_program_code_report_is_the_owner_result_field_for_field() {
        let planes = owner_planes();
        let precision = DeclaredPrecision::new(6).expect("precision");
        let direct = plane_program_code(&planes, &[2], precision).expect("owner code");
        let output = run_parameter_decomposition(
            &request_json(&format!(
                r#"{{"kind": "plane_program_code", "table": "e", "successor": {}, "frequencies": [2], "fraction_bits": 6}}"#,
                successor_json()
            )),
            &tensors(),
            test_governor(),
        )
        .expect("surface run");
        let MpdResult::PlaneProgramCode(report) = &output.report.result else {
            panic!("expected a plane program code, got {:?}", output.report.result);
        };
        assert_eq!(report.subset_bits, direct.subset_bits);
        assert_eq!(report.basis_bits, direct.basis_bits);
        assert_eq!(report.total_bits, direct.total_bits());
        assert_eq!(report.fraction_bits, 6);
        // The decoded basis is the owner's decoded reals, laid out as `U_S`, and lies
        // within half a lattice step of the closed-form basis.
        let decoded = &output.arrays[&report.decoded_basis];
        let basis = planes.basis(&[2]).expect("basis");
        assert_eq!(decoded.shape(), basis.shape());
        let reals = direct.basis.decode().expect("decode");
        let width = basis.nrows();
        for ((row, col), value) in basis.indexed_iter() {
            assert_eq!(decoded[[row, col]], reals[col * width + row]);
            assert!((decoded[[row, col]] - value).abs() <= precision.worst_case_error());
        }
    }

    #[test]
    fn cyclic_requests_the_owner_refuses_reach_the_caller() {
        let good = format!(
            r#"{{"kind": "cyclic_planes", "table": "e", "successor": {}}}"#,
            successor_json()
        );
        assert!(run_parameter_decomposition(&request_json(&good), &tensors(), test_governor()).is_ok());
        // Two 2-cycles and a fixed point: not a single odd cycle.
        let two_cycles = r#"{"kind": "cyclic_planes", "table": "e", "successor": [1, 0, 3, 2, 4, 5, 6, 7]}"#;
        assert!(matches!(
            run_parameter_decomposition(&request_json(two_cycles), &tensors(), test_governor()),
            Err(MpdSurfaceError::CyclicAction(CyclicActionError::NotSingleCycle { .. }))
        ));
        // A successor map over the wrong number of rows.
        let short = r#"{"kind": "cyclic_planes", "table": "e", "successor": [1, 2, 0]}"#;
        assert!(matches!(
            run_parameter_decomposition(&request_json(short), &tensors(), test_governor()),
            Err(MpdSurfaceError::CyclicAction(CyclicActionError::ShapeMismatch { .. }))
        ));
        // A frequency outside `1..=m`.
        let outside = format!(
            r#"{{"kind": "frequency_edit", "successor": {}, "basis": {{"kind": "closed_form", "table": "e"}}, "frequencies": [4], "shift": 1}}"#,
            successor_json()
        );
        assert!(matches!(
            run_parameter_decomposition(&request_json(&outside), &tensors(), test_governor()),
            Err(MpdSurfaceError::CyclicAction(CyclicActionError::InvalidFrequencies { .. }))
        ));
        // A declared basis of the wrong width.
        let mut inputs = tensors();
        inputs.insert("u".to_string(), Array2::<f64>::zeros((5, 3)).into_dyn());
        let wrong_basis = format!(
            r#"{{"kind": "frequency_edit", "successor": {}, "basis": {{"kind": "declared", "tensor": "u"}}, "frequencies": [1], "shift": 1}}"#,
            successor_json()
        );
        assert!(matches!(
            run_parameter_decomposition(&request_json(&wrong_basis), &inputs, test_governor()),
            Err(MpdSurfaceError::CyclicAction(CyclicActionError::ShapeMismatch { .. }))
        ));
        // A precision outside the normal exponents, and a stray field.
        let coarse = format!(
            r#"{{"kind": "plane_program_code", "table": "e", "successor": {}, "frequencies": [1], "fraction_bits": 5000}}"#,
            successor_json()
        );
        assert!(matches!(
            run_parameter_decomposition(&request_json(&coarse), &tensors(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        let stray = good.replacen("\"successor\"", "\"power_floor\": 0.1, \"successor\"", 1);
        assert!(matches!(
            run_parameter_decomposition(&request_json(&stray), &tensors(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
