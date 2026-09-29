//! `sign_gated_swiglu`: the executed sign-gated split of a residual SwiGLU block
//! ([`crate::sign_gated`]) on the wire, its SiLU → ReLU replacement
//! contract at the block, and, for a last layer, at the logits.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayD};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::layer::RmsNormRequest;
use super::verify::{FamilyDomainReport, ToleranceRequest};
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, input, matrix, output, reserve, vector};
use crate::canonical::LayerRmsNorm;
use crate::gauge::SwigluUnits;
use crate::sign_gated::{
    CORRECTION_PEAK_GATE, ChangeDomain, Interval, Readout, ResidualSwiglu, SignGatedError, SignGatedSplit,
    correction_peak, replacement_readout,
};
use crate::verify::{FamilyVerification, RowValue, Tolerance};

/// A residual SwiGLU block `h ↦ h + w(W_d[s(W_g x) ⊙ W_u x])`, `x = N_in(h)`, at declared
/// residual rows. Every string is the id of an input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignGatedSwigluRequest {
    /// `W_g`, `W_u` (`units × width`) and `W_d` (`out × units`).
    pub gate: String,
    pub up: String,
    pub down: String,
    /// The block's residual input rows `h` (`rows × width`), exact.
    pub residual: String,
    /// The RMSNorm the MLP reads through (pre-norm), or null (the MLP reads `h`).
    pub input_norm: Option<RmsNormRequest>,
    /// The RMSNorm on the MLP output before the residual add (post-norm), or null.
    pub output_norm: Option<RmsNormRequest>,
    /// A declared tolerance on each row's `‖Δ‖`, or null.
    pub change_tolerance: Option<f64>,
    /// Return `F`, `P`, `R` and `Δ` with their radii as arrays.
    pub return_rows: bool,
    /// The final norm and unembedding reading the block's output residual (a last layer),
    /// or null.
    pub readout: Option<ReadoutRequest>,
}

/// A last-layer readout and its declared tolerances.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadoutRequest {
    pub norm: RmsNormRequest,
    /// `vocabulary × width`.
    pub unembedding: String,
    pub tolerance: ToleranceRequest,
    /// Rows executed at once (the verification family's inputs).
    pub chunk_rows: usize,
}

/// [`SignGatedSplit`] on the wire. Intervals are `[lower, upper]` and hold the exact value.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SignGatedSwigluReport {
    /// `W(1/e)` rounded up, the peak of `|silu − relu|`, and the gate magnitude where it peaks.
    pub correction_peak: f64,
    pub correction_peak_gate: f64,
    /// `max(|F̂ − P̂ − R̂| − allowance)`, at most zero.
    pub identity_excess: f64,
    /// Certified bounds on `σ₁(W_d)`, and the computed `‖W_d‖_F / √units`.
    pub down_operator_norm: [f64; 2],
    pub down_root_mean_square_gain: f64,
    /// A bound on `‖R‖` at every residual (input norm only).
    pub region_bound: Option<f64>,
    pub law_explained_variance: Option<[f64; 2]>,
    pub correction_over_native: Option<[f64; 2]>,
    pub law_native_cosine: Option<[f64; 2]>,
    pub write_explained_variance: Option<[f64; 2]>,
    pub change_over_write: Option<[f64; 2]>,
    /// `sup_rows ‖Δ‖`, witnessed by a row index.
    pub change_supremum: EvidenceStatusWire<usize, usize>,
    /// Ids of per-row arrays.
    pub per_row: PerRowReport,
    pub rows: Option<RowsReport>,
    pub readout: Option<ReadoutReport>,
}

/// Ids of per-row arrays: `rows × 2` enclosures `[lower, upper]`, `rows` vectors otherwise.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PerRowReport {
    pub native_norm: String,
    pub law_norm: String,
    pub correction_norm: String,
    pub write_norm: String,
    pub change_norm: String,
    /// Upper bounds on `‖e(g) ⊙ u‖`.
    pub correction_hidden_norm: String,
    /// `σ₁(W_d) ‖e(g) ⊙ u‖ ≥ ‖R‖`.
    pub operator_bound: String,
    /// `rows × 3`: certified active, certified inactive, undecided gate counts.
    pub gate_signs: String,
    pub law_units90: String,
}

/// Ids of the executed rows and their radii.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RowsReport {
    pub native: [String; 2],
    pub law: [String; 2],
    pub correction: [String; 2],
    pub change: [String; 2],
}

/// [`FamilyVerification`] of a last-layer replacement, with row witnesses as residual row
/// indices and per-row arrays in row order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReadoutReport {
    pub certified: bool,
    pub forward_kl: EvidenceStatusWire<usize, FamilyDomainReport>,
    pub reverse_kl: EvidenceStatusWire<usize, FamilyDomainReport>,
    pub centred_logit_gap: EvidenceStatusWire<usize, FamilyDomainReport>,
    pub argmax_rows: u64,
    pub argmax_disagreeing: u64,
    pub argmax_certified_rows: u64,
    /// Ids of `rows × 2` enclosures of each row's forward and reverse KL (an unresolved row
    /// holds its lower end twice and is flagged in `kl_resolved`), and `rows` vectors.
    pub forward_kl_rows: String,
    pub reverse_kl_rows: String,
    pub kl_resolved: String,
    pub reference_argmax: String,
    pub candidate_argmax: String,
    pub centred_gap: String,
    pub centred_gap_band: String,
}

fn norm(tensors: &BTreeMap<String, ArrayD<f64>>, request: &RmsNormRequest) -> Result<LayerRmsNorm, MpdSurfaceError> {
    Ok(LayerRmsNorm::native(request.epsilon, vector(tensors, &request.gain)?.to_owned()))
}

fn refused(error: SignGatedError) -> MpdSurfaceError {
    MpdSurfaceError::SignGated(error.to_string())
}

fn pair(interval: Interval) -> Result<[f64; 2], MpdSurfaceError> {
    Ok([finite("interval lower", interval.lower)?, finite("interval upper", interval.upper)?])
}

fn optional_pair(interval: Option<Interval>) -> Result<Option<[f64; 2]>, MpdSurfaceError> {
    interval.map(pair).transpose()
}

fn enclosures(intervals: &[Interval]) -> Array2<f64> {
    Array2::from_shape_fn((intervals.len(), 2), |(row, end)| {
        if end == 0 { intervals[row].lower } else { intervals[row].upper }
    })
}

fn insert(arrays: &mut BTreeMap<String, ArrayD<f64>>, id: &str, array: ArrayD<f64>) -> String {
    arrays.insert(id.to_string(), array);
    id.to_string()
}

fn row_enclosure(value: RowValue) -> ([f64; 2], bool) {
    match value {
        RowValue::Resolved { value, numerical_error } => {
            ([(value - numerical_error).next_down().max(0.0), (value + numerical_error).next_up()], true)
        }
        RowValue::Unresolved { lower } => ([lower, lower], false),
    }
}

fn project_readout(
    verification: FamilyVerification,
    tolerance: &Tolerance,
    chunk_rows: usize,
    rows: usize,
    arrays: &mut BTreeMap<String, ArrayD<f64>>,
) -> Result<ReadoutReport, MpdSurfaceError> {
    let certified = verification.certified_within(tolerance);
    let row_of = |witness: crate::verify::FamilyWitness| witness.input * chunk_rows + witness.row;
    let domain = |domain: crate::verify::FamilyDomain| FamilyDomainReport {
        intervention: domain.intervention,
        inputs: domain.inputs,
        rows: domain.rows,
    };
    let mut forward = Array2::zeros((rows, 2));
    let mut reverse = Array2::zeros((rows, 2));
    let mut columns: [Array1<f64>; 5] = std::array::from_fn(|_| Array1::zeros(rows));
    for (witness, comparison) in &verification.rows {
        let row = row_of(*witness);
        let (forward_row, forward_resolved) = row_enclosure(RowValue::of(&comparison.forward_kl));
        let (reverse_row, reverse_resolved) = row_enclosure(RowValue::of(&comparison.reverse_kl));
        forward[[row, 0]] = finite("forward KL lower", forward_row[0])?;
        forward[[row, 1]] = finite("forward KL upper", forward_row[1])?;
        reverse[[row, 0]] = finite("reverse KL lower", reverse_row[0])?;
        reverse[[row, 1]] = finite("reverse KL upper", reverse_row[1])?;
        columns[0][row] = f64::from(u8::from(forward_resolved && reverse_resolved));
        columns[1][row] = comparison.reference_argmax as f64;
        columns[2][row] = comparison.candidate_argmax as f64;
        columns[3][row] = finite("centred_gap", comparison.centred_gap)?;
        columns[4][row] = finite("centred_gap_band", comparison.centred_gap_band)?;
    }
    let [resolved, reference_argmax, candidate_argmax, centred_gap, centred_gap_band] = columns;
    let status = |status| EvidenceStatusWire::from_status_with(status, row_of, domain);
    Ok(ReadoutReport {
        certified,
        forward_kl: status(verification.forward_kl)?,
        reverse_kl: status(verification.reverse_kl)?,
        centred_logit_gap: status(verification.centred_logit_gap)?,
        argmax_rows: verification.argmax.rows,
        argmax_disagreeing: verification.argmax.disagreeing,
        argmax_certified_rows: verification.argmax.certified_rows,
        forward_kl_rows: insert(arrays, "readout/forward_kl", forward.into_dyn()),
        reverse_kl_rows: insert(arrays, "readout/reverse_kl", reverse.into_dyn()),
        kl_resolved: insert(arrays, "readout/kl_resolved", resolved.into_dyn()),
        reference_argmax: insert(arrays, "readout/reference_argmax", reference_argmax.into_dyn()),
        candidate_argmax: insert(arrays, "readout/candidate_argmax", candidate_argmax.into_dyn()),
        centred_gap: insert(arrays, "readout/centred_gap", centred_gap.into_dyn()),
        centred_gap_band: insert(arrays, "readout/centred_gap_band", centred_gap_band.into_dyn()),
    })
}

fn project_rows(split: &SignGatedSplit, arrays: &mut BTreeMap<String, ArrayD<f64>>) -> RowsReport {
    let mut banded = |name: &str, rows: &crate::sign_gated::BandedRows| {
        [
            insert(arrays, &format!("rows/{name}"), rows.values.clone().into_dyn()),
            insert(arrays, &format!("rows/{name}_radius"), rows.radius.clone().into_dyn()),
        ]
    };
    RowsReport {
        native: banded("native", &split.native),
        law: banded("law", &split.law),
        correction: banded("correction", &split.correction),
        change: banded("change", &split.change),
    }
}

pub(super) fn run(
    request: SignGatedSwigluRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut entries = 0usize;
    for id in [&request.gate, &request.up, &request.down] {
        entries = entries.saturating_add(input(tensors, id)?.len());
    }
    // The owned block and the owner's scaled up copy and operator-norm working copies.
    let copies = reserve(governor, entries, 1, 3, "sign-gated split: owned block")?;
    let block = ResidualSwiglu {
        units: SwigluUnits::new(
            matrix(tensors, &request.gate)?.to_owned(),
            matrix(tensors, &request.up)?.to_owned(),
            matrix(tensors, &request.down)?.to_owned(),
        )
        .map_err(MpdSurfaceError::Gauge)?,
        input_norm: request.input_norm.as_ref().map(|request| norm(tensors, request)).transpose()?,
        output_norm: request.output_norm.as_ref().map(|request| norm(tensors, request)).transpose()?,
    };
    let residual = matrix(tensors, &request.residual)?;
    let split = block.split(governor, residual).map_err(refused)?;
    let change_supremum = EvidenceStatusWire::from_status_with(
        split.change_supremum(request.change_tolerance).map_err(refused)?,
        |row| row,
        |domain: ChangeDomain| domain.rows,
    )?;
    let mut arrays = BTreeMap::new();
    let gate_signs = Array2::from_shape_fn((split.gate_signs.len(), 3), |(row, kind)| {
        let signs = split.gate_signs[row];
        [signs.active, signs.inactive, signs.undecided][kind] as f64
    });
    let per_row = PerRowReport {
        native_norm: insert(&mut arrays, "per_row/native_norm", enclosures(&split.native_norms).into_dyn()),
        law_norm: insert(&mut arrays, "per_row/law_norm", enclosures(&split.law_norms).into_dyn()),
        correction_norm: insert(&mut arrays, "per_row/correction_norm", enclosures(&split.correction_norms).into_dyn()),
        write_norm: insert(&mut arrays, "per_row/write_norm", enclosures(&split.write_norms).into_dyn()),
        change_norm: insert(&mut arrays, "per_row/change_norm", enclosures(&split.change_norms).into_dyn()),
        correction_hidden_norm: insert(
            &mut arrays,
            "per_row/correction_hidden_norm",
            Array1::from(split.correction_hidden_norms.clone()).into_dyn(),
        ),
        operator_bound: insert(&mut arrays, "per_row/operator_bound", Array1::from(split.operator_bounds.clone()).into_dyn()),
        gate_signs: insert(&mut arrays, "per_row/gate_signs", gate_signs.into_dyn()),
        law_units90: insert(
            &mut arrays,
            "per_row/law_units90",
            split.law_units90.iter().map(|&count| count as f64).collect::<Array1<f64>>().into_dyn(),
        ),
    };
    let rows = request.return_rows.then(|| project_rows(&split, &mut arrays));
    let readout = match &request.readout {
        None => None,
        Some(readout_request) => {
            let unembedding = matrix(tensors, &readout_request.unembedding)?;
            let chunk = readout_request.chunk_rows.max(1);
            // The owned unembedding, and one chunk's four logit blocks.
            let owned = reserve(governor, unembedding.nrows(), unembedding.ncols(), 1, "sign-gated readout: unembedding")?;
            let working = reserve(governor, chunk, unembedding.nrows(), 8, "sign-gated readout: one chunk")?;
            let readout = Readout { norm: norm(tensors, &readout_request.norm)?, unembedding: unembedding.to_owned() };
            let tolerance = Tolerance {
                kl: readout_request.tolerance.kl,
                centred_logit_gap: readout_request.tolerance.centred_logit_gap,
            };
            let verification =
                replacement_readout(governor, &block, residual, &split, &readout, &tolerance, chunk).map_err(refused)?;
            drop(working);
            drop(owned);
            Some(project_readout(verification, &tolerance, chunk, residual.nrows(), &mut arrays)?)
        }
    };
    drop(copies);
    let shares = split.shares;
    let report = SignGatedSwigluReport {
        correction_peak: correction_peak(),
        correction_peak_gate: CORRECTION_PEAK_GATE,
        identity_excess: finite("identity_excess", split.identity_excess)?,
        down_operator_norm: [
            finite("down operator norm lower", split.down_gains.operator.lower)?,
            finite("down operator norm upper", split.down_gains.operator.upper)?,
        ],
        down_root_mean_square_gain: finite("down root-mean-square gain", split.down_gains.root_mean_square)?,
        region_bound: split.region_bound.map(|bound| finite("region_bound", bound)).transpose()?,
        law_explained_variance: optional_pair(shares.law_explained_variance)?,
        correction_over_native: optional_pair(shares.correction_over_native)?,
        law_native_cosine: optional_pair(shares.law_native_cosine)?,
        write_explained_variance: optional_pair(shares.write_explained_variance)?,
        change_over_write: optional_pair(shares.change_over_write)?,
        change_supremum,
        per_row,
        rows,
        readout,
    };
    Ok(output(MpdResult::SignGatedSwiglu(Box::new(report)), arrays))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::test_support::test_governor;

    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        let (width, units, vocabulary) = (4, 7, 5);
        let entry = |seed: usize, scale: f64| move |(i, j): (usize, usize)| scale * (((i * 7 + j * 3 + seed) % 11) as f64 - 5.0) / 5.0;
        BTreeMap::from([
            ("g".to_string(), Array2::from_shape_fn((units, width), entry(1, 1.5)).into_dyn()),
            ("u".to_string(), Array2::from_shape_fn((units, width), entry(2, 1.0)).into_dyn()),
            ("d".to_string(), Array2::from_shape_fn((width, units), entry(3, 0.5)).into_dyn()),
            ("h".to_string(), Array2::from_shape_fn((6, width), entry(4, 2.0)).into_dyn()),
            ("gain".to_string(), Array1::from_shape_fn(width, |j| 0.75 + 0.125 * j as f64).into_dyn()),
            ("w".to_string(), Array2::from_shape_fn((vocabulary, width), entry(5, 1.0)).into_dyn()),
        ])
    }

    fn request(readout: bool, rows: &str) -> String {
        let readout = if readout {
            r#"{"norm": {"epsilon": 1e-6, "gain": "gain"}, "unembedding": "w",
                "tolerance": {"kl": 1.0, "centred_logit_gap": 100.0}, "chunk_rows": 4}"#
        } else {
            "null"
        };
        request_json(&format!(
            r#"{{"kind": "sign_gated_swiglu", "gate": "g", "up": "u", "down": "d", "residual": "{rows}",
                "input_norm": {{"epsilon": 1e-6, "gain": "gain"}}, "output_norm": null, "change_tolerance": null,
                "return_rows": true, "readout": {readout}}}"#
        ))
    }

    #[test]
    fn the_report_is_the_owner_split_field_for_field() {
        let tensors = tensors();
        let output = run_parameter_decomposition(&request(true, "h"), &tensors, test_governor()).expect("surface run");
        let MpdResult::SignGatedSwiglu(report) = &output.report.result else {
            panic!("expected a sign-gated report, got {:?}", output.report.result);
        };
        let block = ResidualSwiglu {
            units: SwigluUnits::new(
                tensors["g"].clone().into_dimensionality().expect("gate"),
                tensors["u"].clone().into_dimensionality().expect("up"),
                tensors["d"].clone().into_dimensionality().expect("down"),
            )
            .expect("block"),
            input_norm: Some(LayerRmsNorm::native(1e-6, tensors["gain"].clone().into_dimensionality().expect("gain"))),
            output_norm: None,
        };
        let residual = tensors["h"].view().into_dimensionality().expect("rows");
        let split = block.split(test_governor(), residual).expect("owner split");
        assert_eq!(report.identity_excess, split.identity_excess);
        assert_eq!(report.region_bound, split.region_bound);
        assert_eq!(report.law_explained_variance, split.shares.law_explained_variance.map(|i| [i.lower, i.upper]));
        assert_eq!(output.arrays[&report.per_row.correction_norm], enclosures(&split.correction_norms).into_dyn());
        assert_eq!(output.arrays[&report.per_row.operator_bound], Array1::from(split.operator_bounds.clone()).into_dyn());
        let rows = report.rows.as_ref().expect("rows");
        assert_eq!(output.arrays[&rows.correction[0]], split.correction.values.clone().into_dyn());
        let readout = report.readout.as_ref().expect("readout");
        assert!(readout.certified);
        assert_eq!(readout.argmax_rows, 6);
        assert_eq!(output.arrays[&readout.forward_kl_rows].shape(), &[6, 2]);
    }

    #[test]
    fn a_split_the_owner_refuses_reaches_the_caller() {
        let mut tensors = tensors();
        tensors.insert("bad".to_string(), Array2::<f64>::zeros((3, 5)).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(false, "bad"), &tensors, test_governor()),
            Err(MpdSurfaceError::SignGated(_))
        ));
        assert!(matches!(
            run_parameter_decomposition(&request(false, "missing"), &tensors, test_governor()),
            Err(MpdSurfaceError::MissingTensor { .. })
        ));
        let stray = request(false, "h").replacen("\"return_rows\"", "\"extra\": 1, \"return_rows\"", 1);
        assert!(matches!(
            run_parameter_decomposition(&stray, &tensors, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
