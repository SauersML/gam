//! The single Rust entry behind MPD's Python and CLI surfaces (#2951).
//!
//! A request is a versioned JSON document plus named `f64` arrays, and a report is
//! a versioned JSON document plus named `f64` arrays. Arrays never travel inside
//! the JSON: an eigenbasis at model width is not text, so the document names each
//! array by id and the transport carries it (numpy in `gam-pyffi`, NPY files in
//! `gam-cli`). Both front ends call only [`run_parameter_decomposition`], so
//! parsing, validation, dispatch and projection happen once, and an operation
//! added here reaches Python and the CLI with no front-end change.
//!
//! Nothing here computes. Each operation calls its owner and projects the owner's
//! result into the wire report without strengthening any claim. A value the owner
//! reports as `+inf` to mean "no such quantity" (a cluster that is the whole
//! spectrum has no separation; a refused Davis–Kahan bound has no bar) becomes an
//! absent value with that stated meaning. A report-only quotient the owner leaves
//! non-finite (a stage ratio) is absent too, paired with a typed reason naming the
//! owner's case, and nothing is substituted for it. Any other non-finite value is
//! refused instead of being written as JSON `null`.

use std::collections::BTreeMap;
use std::fmt;

use gam_math::gaussian_activation::GaussianActivation;
use ndarray::{ArrayD, ArrayView1, ArrayView2, Ix1, Ix2};
use serde::{Deserialize, Serialize};

use super::receipts::{
    ExternalExecution, MeasuredDiscrepancy, MlpBlockReceipt, MlpBlockReceiptInputs, ReceiptRefusal,
    StageAgreement, mlp_block_receipt,
};
use super::spectral::{
    PlaneRotationError, PlaneRotationRecovery, RotationAmbiguity, RotationClusterKind,
    recover_plane_rotations,
};

/// Identity of the request document.
pub const MPD_REQUEST_SCHEMA: &str = "gam.mpd-request";

/// Identity of the report document.
pub const MPD_REPORT_SCHEMA: &str = "gam.mpd-report";

/// Version shared by the request and report documents.
pub const MPD_SCHEMA_VERSION: u32 = 1;

/// A complete, front-end-neutral MPD request. Arrays are not embedded; each
/// operation names the input arrays it reads.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MpdRequest {
    pub schema: String,
    pub schema_version: u32,
    pub operation: MpdOperation,
}

/// The closed set of operations. Each variant carries exactly its owner's declared
/// inputs, with no defaults.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MpdOperation {
    /// P3: the rotation planes of one square matrix (`spectral`).
    RecoverPlaneRotations {
        /// Id of the input array holding the matrix.
        tensor: String,
        /// The declared 2-norm distance from the matrix to the orthogonal group. It is an
        /// experiment declaration, required and with no default; the owner refuses a matrix
        /// provably further away.
        declared_error: f64,
    },
    /// A12: one residual MLP block's stages as an external executor ran them, compared
    /// with its native execution (`receipts::mlp_block_receipt`). Every array field is
    /// the id of an input array, and stage rows match `inputs`.
    MlpBlockReceipt {
        /// The executed model's Hugging Face `hidden_act` tag. The owner refuses an
        /// approximate GELU and every tag it does not name.
        hidden_act: String,
        /// What the executor reports it ran. The owner refuses anything but binary64
        /// with TF32 matrix multiplication off.
        external_execution: ExternalExecutionRequest,
        /// The read-in weight `W₁` (`hidden x width`).
        weight: String,
        /// The read-in bias `b₁` (`hidden`).
        bias: String,
        /// The edit's left factor `L` (`hidden x components`).
        left: String,
        /// The edit's coefficients `s` (`components`).
        coefficients: String,
        /// The edit's right factor `R` (`width x components`).
        right: String,
        /// The write-out weight `W₂` (`out x hidden`).
        weight_out: String,
        /// The write-out bias `b₂` (`out`).
        bias_out: String,
        /// The input rows `x` (`rows x width`).
        inputs: String,
        /// The executor's `(W₁ + L diag(s) Rᵀ) x + b₁` (`rows x hidden`).
        external_pre_activation: String,
        /// The executor's activations (`rows x hidden`).
        external_activation: String,
        /// The executor's block output `W₂ a + b₂`, without the residual (`rows x out`).
        external_output: String,
    },
}

/// [`ExternalExecution`] on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalExecutionRequest {
    /// The executed dtype's name, e.g. torch's `"float64"`.
    pub dtype: String,
    /// The executing device. It is recorded, not gated.
    pub device: String,
    /// Whether TF32 matrix multiplication was enabled.
    pub tf32_matmul: bool,
}

impl MpdRequest {
    /// Parses and validates a request document.
    pub fn from_json(raw: &str) -> Result<Self, MpdSurfaceError> {
        let request: Self = serde_json::from_str(raw)
            .map_err(|error| MpdSurfaceError::InvalidRequest(error.to_string()))?;
        if request.schema != MPD_REQUEST_SCHEMA {
            return Err(MpdSurfaceError::InvalidRequest(format!(
                "request schema must be {MPD_REQUEST_SCHEMA:?}, got {:?}",
                request.schema
            )));
        }
        if request.schema_version != MPD_SCHEMA_VERSION {
            return Err(MpdSurfaceError::InvalidRequest(format!(
                "unsupported request schema_version {}; expected {MPD_SCHEMA_VERSION}",
                request.schema_version
            )));
        }
        Ok(request)
    }
}

/// The report document.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MpdReport {
    pub schema: &'static str,
    pub schema_version: u32,
    pub result: MpdResult,
}

/// One operation's projected result.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MpdResult {
    RecoverPlaneRotations(PlaneRotationReport),
    MlpBlockReceipt(MlpBlockReceiptReport),
}

/// [`PlaneRotationRecovery`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaneRotationReport {
    /// The input array the matrix came from.
    pub tensor: String,
    pub orthogonality_defect: f64,
    pub perturbation_bound: f64,
    /// Clusters in increasing cosine order.
    pub clusters: Vec<RotationClusterReport>,
    pub ambiguities: Vec<RotationAmbiguityReport>,
}

/// One spectral cluster on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RotationClusterReport {
    /// Id of the output array holding the cluster's orthonormal basis (`d x m`).
    pub basis: String,
    /// `m`, the dimension of the cluster's eigenspace.
    pub dimension: usize,
    pub cosine_interval: [f64; 2],
    /// Distance to the nearest other computed eigenvalue; absent when the cluster
    /// is the whole spectrum.
    pub separation: Option<f64>,
    /// Davis–Kahan bar on the projector; absent when the bound refuses because the
    /// separation does not exceed the perturbation bound, so no subspace claim
    /// stands.
    pub projector_bar: Option<f64>,
    pub structure: RotationClusterStructure,
}

/// [`RotationClusterKind`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RotationClusterStructure {
    Fixed {
        max_hidden_angle: f64,
    },
    Rotation {
        planes: usize,
        angle: f64,
        /// Id of the output array holding `J` in the cluster basis; absent when the
        /// orientation is not certified.
        complex_structure: Option<String>,
    },
    HalfTurn {
        min_hidden_angle: f64,
    },
    Unresolved,
}

/// [`RotationAmbiguity`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RotationAmbiguityReport {
    Identity,
    RepeatedCosine { cluster: usize, planes: usize },
    HalfTurn { cluster: usize },
    Unresolved { cluster: usize },
    Winding,
}

impl From<RotationAmbiguity> for RotationAmbiguityReport {
    fn from(ambiguity: RotationAmbiguity) -> Self {
        match ambiguity {
            RotationAmbiguity::Identity => Self::Identity,
            RotationAmbiguity::RepeatedCosine { cluster, planes } => {
                Self::RepeatedCosine { cluster, planes }
            }
            RotationAmbiguity::HalfTurn { cluster } => Self::HalfTurn { cluster },
            RotationAmbiguity::Unresolved { cluster } => Self::Unresolved { cluster },
            RotationAmbiguity::Winding => Self::Winding,
        }
    }
}

/// [`MlpBlockReceipt`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MlpBlockReceiptReport {
    /// The executing device as the executor reported it; recorded, not gated.
    pub device: String,
    pub pre_activation: StageAgreementReport,
    pub activation_measured: MeasuredDiscrepancyReport,
    pub output: StageAgreementReport,
    pub end_to_end_measured: MeasuredDiscrepancyReport,
}

/// [`StageAgreement`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageAgreementReport {
    pub agrees: bool,
    pub refutes: bool,
    /// `[row, column]` of the entry with the largest discrepancy-to-band ratio.
    pub witness: [usize; 2],
    pub discrepancy: f64,
    pub band: f64,
    /// `discrepancy / band` at the witness, for reports only: `agrees` and `refutes`
    /// are decided entry by entry without it.
    pub ratio: StageRatio,
}

/// The owner's report-only stage ratio: its value, or the owner's case that leaves it
/// without a finite value.
///
/// The owner keeps the largest entry ratio, starting from `-inf`, where an entry's
/// ratio is `0` at zero discrepancy and `discrepancy / band` otherwise, with the
/// discrepancy rounded up (positive) and the band rounded down (at least `0`, at most
/// the largest `f64`). A stored ratio is therefore finite, `-inf` exactly when no
/// entry was compared, or `+inf` from a division by a zero band or an overflowing
/// quotient. These are the variants; the owner never stores a NaN, and one would be
/// refused rather than named.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageRatio {
    Finite { value: f64 },
    /// The stage has no entries, so no ratio was taken.
    EmptyStage,
    /// The band rounds down to zero at a nonzero discrepancy; the stage refutes.
    BandRoundsToZero,
    /// A nonzero discrepancy over a positive band exceeds the largest `f64`.
    QuotientOverflows,
}

/// [`MeasuredDiscrepancy`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MeasuredDiscrepancyReport {
    pub largest: f64,
    /// `[row, column]` of the largest entry.
    pub witness: [usize; 2],
    pub native_at_witness: f64,
}

/// A report and the arrays it names.
#[derive(Clone, Debug, PartialEq)]
pub struct MpdOutput {
    pub report: MpdReport,
    pub arrays: BTreeMap<String, ArrayD<f64>>,
}

impl MpdOutput {
    /// The report document, the same bytes for every front end.
    pub fn report_json(&self) -> Result<String, MpdSurfaceError> {
        serde_json::to_string_pretty(&self.report)
            .map_err(|error| MpdSurfaceError::Serialize(error.to_string()))
    }
}

/// Why a request produced no report.
#[derive(Debug)]
pub enum MpdSurfaceError {
    /// The document did not parse, or named another schema or version.
    InvalidRequest(String),
    /// The operation names an input array that was not supplied.
    MissingTensor { tensor: String },
    /// An input array does not have the shape the operation reads.
    TensorShape { tensor: String, reason: String },
    PlaneRotation(PlaneRotationError),
    /// The receipts owner refused the block, its execution or its activation tag.
    Receipt(ReceiptRefusal),
    /// An owner returned a non-finite value where the wire report has no meaning
    /// for one.
    NonFiniteReport { field: &'static str, value: f64 },
    Serialize(String),
}

impl fmt::Display for MpdSurfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(formatter, "invalid MPD request: {reason}"),
            Self::MissingTensor { tensor } => write!(
                formatter,
                "MPD request names input array {tensor:?}, which was not supplied"
            ),
            Self::TensorShape { tensor, reason } => {
                write!(formatter, "MPD input array {tensor:?}: {reason}")
            }
            Self::PlaneRotation(error) => write!(formatter, "{error}"),
            Self::Receipt(error) => write!(formatter, "{error}"),
            Self::NonFiniteReport { field, value } => write!(
                formatter,
                "MPD report field {field} is {value}, which the wire report cannot state"
            ),
            Self::Serialize(reason) => write!(formatter, "serialize MPD report: {reason}"),
        }
    }
}

impl std::error::Error for MpdSurfaceError {}

/// Runs one MPD request against its named input arrays.
pub fn run_parameter_decomposition(
    request_json: &str,
    tensors: &BTreeMap<String, ArrayD<f64>>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let request = MpdRequest::from_json(request_json)?;
    match request.operation {
        MpdOperation::RecoverPlaneRotations {
            tensor,
            declared_error,
        } => {
            let recovery = recover_plane_rotations(matrix(tensors, &tensor)?, declared_error)
                .map_err(MpdSurfaceError::PlaneRotation)?;
            project_plane_rotations(tensor, recovery)
        }
        MpdOperation::MlpBlockReceipt {
            hidden_act,
            external_execution,
            weight,
            bias,
            left,
            coefficients,
            right,
            weight_out,
            bias_out,
            inputs,
            external_pre_activation,
            external_activation,
            external_output,
        } => {
            let activation = GaussianActivation::from_hidden_act(&hidden_act)
                .map_err(|error| MpdSurfaceError::Receipt(ReceiptRefusal::Activation(error)))?;
            let receipt = mlp_block_receipt(MlpBlockReceiptInputs {
                external_execution: ExternalExecution {
                    dtype: &external_execution.dtype,
                    device: &external_execution.device,
                    tf32_matmul: external_execution.tf32_matmul,
                },
                activation,
                weight: matrix(tensors, &weight)?,
                bias: vector(tensors, &bias)?,
                left: matrix(tensors, &left)?,
                coefficients: vector(tensors, &coefficients)?,
                right: matrix(tensors, &right)?,
                weight_out: matrix(tensors, &weight_out)?,
                bias_out: vector(tensors, &bias_out)?,
                inputs: matrix(tensors, &inputs)?,
                external_pre_activation: matrix(tensors, &external_pre_activation)?,
                external_activation: matrix(tensors, &external_activation)?,
                external_output: matrix(tensors, &external_output)?,
            })
            .map_err(MpdSurfaceError::Receipt)?;
            project_mlp_block_receipt(external_execution.device, receipt)
        }
    }
}

/// The input array a request names.
fn input<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    id: &str,
) -> Result<&'a ArrayD<f64>, MpdSurfaceError> {
    tensors.get(id).ok_or_else(|| MpdSurfaceError::MissingTensor {
        tensor: id.to_string(),
    })
}

/// The input array a request names, read as a matrix.
fn matrix<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    id: &str,
) -> Result<ArrayView2<'a, f64>, MpdSurfaceError> {
    let array = input(tensors, id)?;
    array
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|error| MpdSurfaceError::TensorShape {
            tensor: id.to_string(),
            reason: format!("expected a matrix, got shape {:?}: {error}", array.shape()),
        })
}

/// The input array a request names, read as a vector.
fn vector<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    id: &str,
) -> Result<ArrayView1<'a, f64>, MpdSurfaceError> {
    let array = input(tensors, id)?;
    array
        .view()
        .into_dimensionality::<Ix1>()
        .map_err(|error| MpdSurfaceError::TensorShape {
            tensor: id.to_string(),
            reason: format!("expected a vector, got shape {:?}: {error}", array.shape()),
        })
}

fn project_plane_rotations(
    tensor: String,
    recovery: PlaneRotationRecovery,
) -> Result<MpdOutput, MpdSurfaceError> {
    let ambiguities = recovery
        .ambiguities()
        .into_iter()
        .map(RotationAmbiguityReport::from)
        .collect();
    let orthogonality_defect = finite("orthogonality_defect", recovery.orthogonality_defect)?;
    let perturbation_bound = finite("perturbation_bound", recovery.perturbation_bound)?;
    let mut arrays = BTreeMap::new();
    let mut clusters = Vec::with_capacity(recovery.clusters.len());
    for (index, cluster) in recovery.clusters.into_iter().enumerate() {
        let structure = match cluster.kind {
            RotationClusterKind::Fixed { max_hidden_angle } => RotationClusterStructure::Fixed {
                max_hidden_angle: finite("max_hidden_angle", max_hidden_angle)?,
            },
            RotationClusterKind::Rotation {
                planes,
                angle,
                complex_structure,
            } => RotationClusterStructure::Rotation {
                planes,
                angle: finite("angle", angle)?,
                complex_structure: complex_structure.map(|structure| {
                    let id = format!("clusters/{index}/complex_structure");
                    arrays.insert(id.clone(), structure.into_dyn());
                    id
                }),
            },
            RotationClusterKind::HalfTurn { min_hidden_angle } => {
                RotationClusterStructure::HalfTurn {
                    min_hidden_angle: finite("min_hidden_angle", min_hidden_angle)?,
                }
            }
            RotationClusterKind::Unresolved => RotationClusterStructure::Unresolved,
        };
        let basis = format!("clusters/{index}/basis");
        let dimension = cluster.basis.ncols();
        arrays.insert(basis.clone(), cluster.basis.into_dyn());
        clusters.push(RotationClusterReport {
            basis,
            dimension,
            cosine_interval: [
                finite("cosine_interval", cluster.cosine_interval.0)?,
                finite("cosine_interval", cluster.cosine_interval.1)?,
            ],
            separation: absent_when_infinite("separation", cluster.separation)?,
            projector_bar: absent_when_infinite("projector_bar", cluster.projector_bar)?,
            structure,
        });
    }
    Ok(MpdOutput {
        report: MpdReport {
            schema: MPD_REPORT_SCHEMA,
            schema_version: MPD_SCHEMA_VERSION,
            result: MpdResult::RecoverPlaneRotations(PlaneRotationReport {
                tensor,
                orthogonality_defect,
                perturbation_bound,
                clusters,
                ambiguities,
            }),
        },
        arrays,
    })
}

fn project_mlp_block_receipt(
    device: String,
    receipt: MlpBlockReceipt,
) -> Result<MpdOutput, MpdSurfaceError> {
    let report = MlpBlockReceiptReport {
        device,
        pre_activation: stage_agreement_report(
            [
                "pre_activation.discrepancy",
                "pre_activation.band",
                "pre_activation.ratio",
            ],
            receipt.pre_activation,
        )?,
        activation_measured: measured_discrepancy_report(
            [
                "activation_measured.largest",
                "activation_measured.native_at_witness",
            ],
            receipt.activation_measured,
        )?,
        output: stage_agreement_report(
            ["output.discrepancy", "output.band", "output.ratio"],
            receipt.output,
        )?,
        end_to_end_measured: measured_discrepancy_report(
            [
                "end_to_end_measured.largest",
                "end_to_end_measured.native_at_witness",
            ],
            receipt.end_to_end_measured,
        )?,
    };
    Ok(MpdOutput {
        report: MpdReport {
            schema: MPD_REPORT_SCHEMA,
            schema_version: MPD_SCHEMA_VERSION,
            result: MpdResult::MlpBlockReceipt(report),
        },
        arrays: BTreeMap::new(),
    })
}

/// `fields` names the discrepancy, the band and the ratio for a refusal.
fn stage_agreement_report(
    fields: [&'static str; 3],
    agreement: StageAgreement,
) -> Result<StageAgreementReport, MpdSurfaceError> {
    let ratio = if agreement.ratio.is_finite() {
        StageRatio::Finite {
            value: agreement.ratio,
        }
    } else if agreement.ratio == f64::NEG_INFINITY {
        StageRatio::EmptyStage
    } else if agreement.ratio == f64::INFINITY && agreement.band == 0.0 {
        StageRatio::BandRoundsToZero
    } else if agreement.ratio == f64::INFINITY {
        StageRatio::QuotientOverflows
    } else {
        return Err(MpdSurfaceError::NonFiniteReport {
            field: fields[2],
            value: agreement.ratio,
        });
    };
    Ok(StageAgreementReport {
        agrees: agreement.agrees,
        refutes: agreement.refutes,
        witness: [agreement.witness.0, agreement.witness.1],
        discrepancy: finite(fields[0], agreement.discrepancy)?,
        band: finite(fields[1], agreement.band)?,
        ratio,
    })
}

/// `fields` names the largest entry and the native value at its witness for a refusal.
fn measured_discrepancy_report(
    fields: [&'static str; 2],
    measured: MeasuredDiscrepancy,
) -> Result<MeasuredDiscrepancyReport, MpdSurfaceError> {
    Ok(MeasuredDiscrepancyReport {
        largest: finite(fields[0], measured.largest)?,
        witness: [measured.witness.0, measured.witness.1],
        native_at_witness: finite(fields[1], measured.native_at_witness)?,
    })
}

fn finite(field: &'static str, value: f64) -> Result<f64, MpdSurfaceError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MpdSurfaceError::NonFiniteReport { field, value })
    }
}

/// `+inf` is an owner's "no such quantity"; every other non-finite value is refused.
fn absent_when_infinite(field: &'static str, value: f64) -> Result<Option<f64>, MpdSurfaceError> {
    if value == f64::INFINITY {
        Ok(None)
    } else {
        finite(field, value).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::super::receipts::compare_stage;
    use super::*;
    use gam_linalg::roundoff::accumulation_growth;
    use gam_math::gaussian_activation::GaussianActivationError;
    use ndarray::{Array1, Array2, Axis, array};

    fn request_json(operation: &str) -> String {
        format!(
            r#"{{"schema": "{MPD_REQUEST_SCHEMA}", "schema_version": {MPD_SCHEMA_VERSION}, "operation": {operation}}}"#
        )
    }

    fn plane_request(declared_error: f64) -> String {
        request_json(&format!(
            r#"{{"kind": "recover_plane_rotations", "tensor": "w", "declared_error": {declared_error:?}}}"#
        ))
    }

    fn inputs(id: &str, matrix: Array2<f64>) -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([(id.to_string(), matrix.into_dyn())])
    }

    fn frobenius(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    /// A declared distance to the orthogonal group that covers the true one:
    /// `||W - W_o||_2 = max |sigma_i - 1| <= ||W^T W - I||_F`, plus the rounding of the
    /// computed Gram, `gamma_n` times the Frobenius norm of `|W|^T |W|`.
    fn orthogonality_declaration(matrix: &Array2<f64>) -> f64 {
        let columns = matrix.ncols();
        let gram = matrix.t().dot(matrix) - Array2::<f64>::eye(columns);
        let absolute = matrix.mapv(f64::abs);
        frobenius(&gram) + accumulation_growth(columns) * frobenius(&absolute.t().dot(&absolute))
    }

    /// Rotations by `alpha` and `beta` in the planes (e1, e2) and (e3, e4), with e5
    /// fixed.
    fn two_plane_rotation(alpha: f64, beta: f64) -> Array2<f64> {
        let (sa, ca) = alpha.sin_cos();
        let (sb, cb) = beta.sin_cos();
        array![
            [ca, -sa, 0.0, 0.0, 0.0],
            [sa, ca, 0.0, 0.0, 0.0],
            [0.0, 0.0, cb, -sb, 0.0],
            [0.0, 0.0, sb, cb, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    }

    #[test]
    fn request_document_round_trips_and_refuses_what_it_does_not_declare() {
        let valid = plane_request(0.0);
        let request = MpdRequest::from_json(&valid).expect("a declared request parses");
        assert_eq!(
            request.operation,
            MpdOperation::RecoverPlaneRotations {
                tensor: "w".to_string(),
                declared_error: 0.0,
            }
        );
        let reserialized = serde_json::to_string(&request).expect("serialize request");
        assert_eq!(
            MpdRequest::from_json(&reserialized).expect("round trip parses"),
            request
        );

        // Each refusal is a small change to the accepted document above, so a guard
        // that refused everything would already have failed.
        let unknown_top = valid.replacen("\"operation\"", "\"extra\": 1, \"operation\"", 1);
        assert!(MpdRequest::from_json(&unknown_top).is_err());
        let unknown_operation_field = request_json(
            r#"{"kind": "recover_plane_rotations", "tensor": "w", "declared_error": 0.0, "tolerance": 1e-8}"#,
        );
        assert!(MpdRequest::from_json(&unknown_operation_field).is_err());
        let missing_tensor = request_json(r#"{"kind": "recover_plane_rotations", "declared_error": 0.0}"#);
        assert!(MpdRequest::from_json(&missing_tensor).is_err());
        // The declared error is an experiment declaration with no default.
        let missing_declaration = request_json(r#"{"kind": "recover_plane_rotations", "tensor": "w"}"#);
        assert!(MpdRequest::from_json(&missing_declaration).is_err());
        let unknown_kind =
            request_json(r#"{"kind": "guess_the_planes", "tensor": "w", "declared_error": 0.0}"#);
        assert!(MpdRequest::from_json(&unknown_kind).is_err());
        let other_schema = valid.replacen(MPD_REQUEST_SCHEMA, "gam.fit-request", 1);
        assert!(MpdRequest::from_json(&other_schema).is_err());
        let other_version = valid.replacen(
            &format!("\"schema_version\": {MPD_SCHEMA_VERSION}"),
            "\"schema_version\": 0",
            1,
        );
        assert!(MpdRequest::from_json(&other_version).is_err());
    }

    #[test]
    fn plane_rotation_report_is_the_owner_result_field_for_field() {
        let matrix = two_plane_rotation(0.7, 1.9);
        let declared = orthogonality_declaration(&matrix);
        let direct = recover_plane_rotations(matrix.view(), declared).expect("owner recovery");
        let output = run_parameter_decomposition(&plane_request(declared), &inputs("w", matrix))
            .expect("surface run");
        let MpdResult::RecoverPlaneRotations(report) = &output.report.result else {
            panic!("expected a plane-rotation report, got {:?}", output.report.result);
        };

        assert_eq!(report.tensor, "w");
        assert_eq!(report.orthogonality_defect, direct.orthogonality_defect);
        assert_eq!(report.perturbation_bound, direct.perturbation_bound);
        assert_eq!(report.clusters.len(), direct.clusters.len());
        let rotations = report
            .clusters
            .iter()
            .filter(|cluster| matches!(cluster.structure, RotationClusterStructure::Rotation { .. }))
            .count();
        assert_eq!(rotations, 2, "two planted planes with distinct cosines");
        for (wire, owner) in report.clusters.iter().zip(&direct.clusters) {
            assert_eq!(output.arrays[&wire.basis], owner.basis.clone().into_dyn());
            assert_eq!(wire.dimension, owner.basis.ncols());
            assert_eq!(
                wire.cosine_interval,
                [owner.cosine_interval.0, owner.cosine_interval.1]
            );
            assert_eq!(
                wire.separation,
                (owner.separation != f64::INFINITY).then_some(owner.separation)
            );
            assert_eq!(
                wire.projector_bar,
                (owner.projector_bar != f64::INFINITY).then_some(owner.projector_bar)
            );
            match (&wire.structure, &owner.kind) {
                (
                    RotationClusterStructure::Fixed { max_hidden_angle },
                    RotationClusterKind::Fixed {
                        max_hidden_angle: owner_angle,
                    },
                ) => assert_eq!(max_hidden_angle, owner_angle),
                (
                    RotationClusterStructure::HalfTurn { min_hidden_angle },
                    RotationClusterKind::HalfTurn {
                        min_hidden_angle: owner_angle,
                    },
                ) => assert_eq!(min_hidden_angle, owner_angle),
                (RotationClusterStructure::Unresolved, RotationClusterKind::Unresolved) => assert!(
                    wire.cosine_interval[0] <= -1.0 && wire.cosine_interval[1] >= 1.0,
                    "an unresolved cluster admits both +1 and -1"
                ),
                (
                    RotationClusterStructure::Rotation {
                        planes,
                        angle,
                        complex_structure,
                    },
                    RotationClusterKind::Rotation {
                        planes: owner_planes,
                        angle: owner_angle,
                        complex_structure: owner_structure,
                    },
                ) => {
                    assert_eq!(planes, owner_planes);
                    assert_eq!(angle, owner_angle);
                    assert_eq!(
                        complex_structure.is_some(),
                        owner_structure.is_some(),
                        "orientation certificate changed on the wire"
                    );
                    if let (Some(id), Some(j)) = (complex_structure, owner_structure) {
                        assert_eq!(output.arrays[id], j.clone().into_dyn());
                    }
                }
                (wire_structure, owner_kind) => panic!(
                    "cluster structure changed on the wire: {wire_structure:?} vs {owner_kind:?}"
                ),
            }
        }
        let expected: Vec<RotationAmbiguityReport> = direct
            .ambiguities()
            .into_iter()
            .map(RotationAmbiguityReport::from)
            .collect();
        assert_eq!(report.ambiguities, expected);

        // Every array the report names is supplied, and nothing else is.
        let mut named: Vec<&String> = report.clusters.iter().map(|cluster| &cluster.basis).collect();
        for cluster in &report.clusters {
            if let RotationClusterStructure::Rotation {
                complex_structure: Some(id),
                ..
            } = &cluster.structure
            {
                named.push(id);
            }
        }
        named.sort();
        assert_eq!(named, output.arrays.keys().collect::<Vec<_>>());

        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert_eq!(json["schema"], MPD_REPORT_SCHEMA);
        assert_eq!(json["result"]["kind"], "recover_plane_rotations");
    }

    #[test]
    fn identity_reports_its_ambiguity_and_an_absent_separation() {
        let output = run_parameter_decomposition(&plane_request(0.0), &inputs("w", Array2::eye(3)))
            .expect("surface run on the identity");
        let MpdResult::RecoverPlaneRotations(report) = &output.report.result else {
            panic!("expected a plane-rotation report, got {:?}", output.report.result);
        };
        assert_eq!(report.ambiguities, vec![RotationAmbiguityReport::Identity]);
        assert_eq!(report.clusters.len(), 1);
        assert_eq!(report.clusters[0].separation, None);
        // The whole spectrum's projector is the identity, a bar of exactly 0, not an
        // absent one.
        assert_eq!(report.clusters[0].projector_bar, Some(0.0));
        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert!(json["result"]["clusters"][0]["separation"].is_null());
        assert_eq!(json["result"]["clusters"][0]["projector_bar"], 0.0);
    }

    #[test]
    fn the_owners_refusal_of_a_non_orthogonal_matrix_reaches_the_caller() {
        // 2 I is 1 away from the orthogonal group, far beyond a declared error of 0.
        assert!(matches!(
            run_parameter_decomposition(&plane_request(0.0), &inputs("w", Array2::eye(2) * 2.0)),
            Err(MpdSurfaceError::PlaneRotation(
                PlaneRotationError::NotOrthogonal { .. }
            ))
        ));
        // Positive control: the same request on an orthogonal matrix is accepted.
        assert!(run_parameter_decomposition(&plane_request(0.0), &inputs("w", Array2::eye(2))).is_ok());
    }

    #[test]
    fn missing_or_misshapen_inputs_are_refused() {
        let json = plane_request(0.0);
        assert!(run_parameter_decomposition(&json, &inputs("w", Array2::eye(2))).is_ok());
        assert!(matches!(
            run_parameter_decomposition(&json, &inputs("other", Array2::eye(2))),
            Err(MpdSurfaceError::MissingTensor { .. })
        ));
        let cube = BTreeMap::from([("w".to_string(), ArrayD::<f64>::zeros(vec![2, 2, 2]))]);
        assert!(matches!(
            run_parameter_decomposition(&json, &cube),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
    }

    #[test]
    fn only_an_owners_infinite_no_such_quantity_becomes_absent() {
        assert_eq!(absent_when_infinite("x", 0.5).expect("finite"), Some(0.5));
        assert_eq!(absent_when_infinite("x", f64::INFINITY).expect("+inf"), None);
        assert!(absent_when_infinite("x", f64::NEG_INFINITY).is_err());
        assert!(absent_when_infinite("x", f64::NAN).is_err());
        assert!(finite("x", f64::INFINITY).is_err());
        assert_eq!(finite("x", -2.0).expect("finite"), -2.0);
    }

    /// A ReLU block whose entries are dyadic rationals of few bits, so every product,
    /// sum and activation is exact in binary64 whatever the evaluation order: the
    /// executor stages computed here equal the native execution bitwise.
    struct DyadicBlock {
        weight: Array2<f64>,
        bias: Array1<f64>,
        left: Array2<f64>,
        coefficients: Array1<f64>,
        right: Array2<f64>,
        weight_out: Array2<f64>,
        bias_out: Array1<f64>,
        inputs: Array2<f64>,
        pre_activation: Array2<f64>,
        activation: Array2<f64>,
        output: Array2<f64>,
    }

    fn dyadic_block_from(
        weight_out: Array2<f64>,
        bias_out: Array1<f64>,
        inputs: Array2<f64>,
    ) -> DyadicBlock {
        let weight = array![[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]];
        let bias = array![0.125, -0.25, 0.375];
        let left = array![[1.0], [0.0], [-1.0]];
        let coefficients = array![0.5];
        let right = array![[0.25], [1.0]];
        let edited = &weight + &left.dot(&Array2::from_diag(&coefficients)).dot(&right.t());
        let pre_activation = inputs.dot(&edited.t()) + &bias;
        let activation = pre_activation.mapv(|value| value.max(0.0));
        let output = activation.dot(&weight_out.t()) + &bias_out;
        DyadicBlock {
            weight,
            bias,
            left,
            coefficients,
            right,
            weight_out,
            bias_out,
            inputs,
            pre_activation,
            activation,
            output,
        }
    }

    fn dyadic_block() -> DyadicBlock {
        dyadic_block_from(
            array![[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]],
            array![0.0625, -0.125],
            array![[1.0, 2.0], [-0.5, 0.75]],
        )
    }

    fn receipt_request(hidden_act: &str, dtype: &str, tf32_matmul: bool) -> String {
        request_json(&format!(
            r#"{{"kind": "mlp_block_receipt", "hidden_act": "{hidden_act}", "external_execution": {{"dtype": "{dtype}", "device": "cpu", "tf32_matmul": {tf32_matmul}}}, "weight": "w1", "bias": "b1", "left": "l", "coefficients": "s", "right": "r", "weight_out": "w2", "bias_out": "b2", "inputs": "x", "external_pre_activation": "pre", "external_activation": "act", "external_output": "out"}}"#
        ))
    }

    fn block_tensors(block: &DyadicBlock) -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([
            ("w1".to_string(), block.weight.clone().into_dyn()),
            ("b1".to_string(), block.bias.clone().into_dyn()),
            ("l".to_string(), block.left.clone().into_dyn()),
            ("s".to_string(), block.coefficients.clone().into_dyn()),
            ("r".to_string(), block.right.clone().into_dyn()),
            ("w2".to_string(), block.weight_out.clone().into_dyn()),
            ("b2".to_string(), block.bias_out.clone().into_dyn()),
            ("x".to_string(), block.inputs.clone().into_dyn()),
            ("pre".to_string(), block.pre_activation.clone().into_dyn()),
            ("act".to_string(), block.activation.clone().into_dyn()),
            ("out".to_string(), block.output.clone().into_dyn()),
        ])
    }

    fn receipt_report(output: &MpdOutput) -> &MlpBlockReceiptReport {
        let MpdResult::MlpBlockReceipt(report) = &output.report.result else {
            panic!("expected an MLP block receipt, got {:?}", output.report.result);
        };
        report
    }

    fn assert_stage_projects(wire: &StageAgreementReport, owner: &StageAgreement) {
        assert_eq!(wire.agrees, owner.agrees);
        assert_eq!(wire.refutes, owner.refutes);
        assert_eq!(wire.witness, [owner.witness.0, owner.witness.1]);
        assert_eq!(wire.discrepancy, owner.discrepancy);
        assert_eq!(wire.band, owner.band);
        match wire.ratio {
            StageRatio::Finite { value } => assert_eq!(value, owner.ratio),
            reason => panic!("the owner's finite ratio {} projected as {reason:?}", owner.ratio),
        }
    }

    fn assert_measured_projects(wire: &MeasuredDiscrepancyReport, owner: &MeasuredDiscrepancy) {
        assert_eq!(wire.largest, owner.largest);
        assert_eq!(wire.witness, [owner.witness.0, owner.witness.1]);
        assert_eq!(wire.native_at_witness, owner.native_at_witness);
    }

    const STAGE_FIELDS: [&str; 3] = ["stage.discrepancy", "stage.band", "stage.ratio"];

    const BINARY64_CPU: ExternalExecution<'static> = ExternalExecution {
        dtype: "float64",
        device: "cpu",
        tf32_matmul: false,
    };

    #[test]
    fn mlp_block_receipt_report_is_the_owner_result_field_for_field() {
        let block = dyadic_block();
        assert_eq!(
            block.pre_activation,
            array![[-0.25, 1.75, 2.5], [-0.5625, -0.8125, 1.9375]],
            "the fixture's stages are exact"
        );
        assert_eq!(block.output, array![[-0.1875, 0.875], [0.546875, -2.0625]]);
        let direct = mlp_block_receipt(MlpBlockReceiptInputs {
            external_execution: BINARY64_CPU,
            activation: GaussianActivation::Relu,
            weight: block.weight.view(),
            bias: block.bias.view(),
            left: block.left.view(),
            coefficients: block.coefficients.view(),
            right: block.right.view(),
            weight_out: block.weight_out.view(),
            bias_out: block.bias_out.view(),
            inputs: block.inputs.view(),
            external_pre_activation: block.pre_activation.view(),
            external_activation: block.activation.view(),
            external_output: block.output.view(),
        })
        .expect("owner receipt");
        let output = run_parameter_decomposition(
            &receipt_request("relu", "float64", false),
            &block_tensors(&block),
        )
        .expect("surface run");
        let report = receipt_report(&output);

        assert_eq!(report.device, "cpu");
        assert_stage_projects(&report.pre_activation, &direct.pre_activation);
        assert_stage_projects(&report.output, &direct.output);
        assert_measured_projects(&report.activation_measured, &direct.activation_measured);
        assert_measured_projects(&report.end_to_end_measured, &direct.end_to_end_measured);
        // Exact stages: both compared stages agree with no discrepancy, and ReLU's
        // measured stages are exact.
        assert!(report.pre_activation.agrees && !report.pre_activation.refutes, "{report:?}");
        assert!(report.output.agrees && !report.output.refutes, "{report:?}");
        assert_eq!(report.pre_activation.ratio, StageRatio::Finite { value: 0.0 });
        assert_eq!(report.activation_measured.largest, 0.0);
        assert_eq!(report.end_to_end_measured.largest, 0.0);
        assert!(output.arrays.is_empty());

        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert_eq!(json["result"]["kind"], "mlp_block_receipt");
        assert_eq!(json["result"]["output"]["agrees"], true);
        assert_eq!(json["result"]["output"]["ratio"]["kind"], "finite");
        assert_eq!(json["result"]["device"], "cpu");
    }

    #[test]
    fn an_executed_output_outside_its_band_refutes_through_the_surface() {
        let mut block = dyadic_block();
        let json = receipt_request("relu", "float64", false);
        // Positive control: the exact stages agree.
        let agreeing = run_parameter_decomposition(&json, &block_tensors(&block)).expect("surface run");
        assert!(receipt_report(&agreeing).output.agrees);

        block.output[[0, 0]] += 1.0;
        let displaced = run_parameter_decomposition(&json, &block_tensors(&block)).expect("surface run");
        let report = receipt_report(&displaced);
        assert!(report.output.refutes && !report.output.agrees, "{report:?}");
        assert_eq!(report.output.witness, [0, 0]);
        assert!(matches!(report.output.ratio, StageRatio::Finite { value } if value > 1.0));
        assert!(report.pre_activation.agrees, "{report:?}");
        assert_eq!(report.end_to_end_measured.largest, 1.0);
        assert_eq!(report.end_to_end_measured.witness, [0, 0]);
    }

    #[test]
    fn a_zero_band_ratio_is_absent_as_band_rounds_to_zero() {
        // Every band the block receipt derives is at least the smallest subnormal
        // (`evaluation_band` rounds up), so a zero band is driven through the owner's
        // `compare_stage` with zero bands supplied.
        let native = array![[0.0, 0.0]];
        let zero_band = Array2::<f64>::zeros((1, 2));
        // Positive control: equal stages keep a finite ratio of 0.
        let equal = compare_stage(BINARY64_CPU, native.view(), native.view(), zero_band.view(), zero_band.view())
            .expect("owner comparison");
        let wire = stage_agreement_report(STAGE_FIELDS, equal).expect("projection");
        assert_eq!(wire.ratio, StageRatio::Finite { value: 0.0 });

        let external = array![[1.0, 0.0]];
        let agreement = compare_stage(BINARY64_CPU, external.view(), native.view(), zero_band.view(), zero_band.view())
            .expect("owner comparison");
        assert_eq!((agreement.ratio, agreement.band), (f64::INFINITY, 0.0));
        let wire = stage_agreement_report(STAGE_FIELDS, agreement).expect("projection");
        assert_eq!(wire.ratio, StageRatio::BandRoundsToZero);
        assert!(wire.refutes && !wire.agrees);
        assert_eq!((wire.witness, wire.band), ([0, 0], 0.0));
        let json = serde_json::to_value(&wire).expect("serialize stage");
        assert_eq!(json["ratio"]["kind"], "band_rounds_to_zero");
        assert!(json["ratio"]["value"].is_null());
    }

    #[test]
    fn an_overflowing_ratio_is_absent_as_quotient_overflows_through_the_surface() {
        // A zero write-out gives the output stage the smallest positive band, so one
        // executed output entry 1 away from the native 0 divides past the largest f64.
        let zero_write = dyadic_block_from(
            Array2::zeros((2, 3)),
            Array1::zeros(2),
            array![[1.0, 2.0], [-0.5, 0.75]],
        );
        let json = receipt_request("relu", "float64", false);
        // Positive control: the undisplaced stages keep a finite ratio of 0.
        let agreeing = run_parameter_decomposition(&json, &block_tensors(&zero_write)).expect("surface run");
        assert_eq!(receipt_report(&agreeing).output.ratio, StageRatio::Finite { value: 0.0 });

        let mut displaced_block = zero_write;
        displaced_block.output[[0, 0]] = 1.0;
        let displaced =
            run_parameter_decomposition(&json, &block_tensors(&displaced_block)).expect("surface run");
        let report = receipt_report(&displaced);
        assert_eq!(report.output.ratio, StageRatio::QuotientOverflows, "{report:?}");
        assert!(report.output.band > 0.0 && report.output.refutes, "{report:?}");
        assert_eq!(report.output.witness, [0, 0]);
        assert_eq!(report.pre_activation.ratio, StageRatio::Finite { value: 0.0 });
    }

    #[test]
    fn an_empty_stage_ratio_is_absent_as_empty_stage() {
        // Through the owner's comparison: a stage with no entries takes no ratio.
        let empty = Array2::<f64>::zeros((0, 2));
        let agreement = compare_stage(BINARY64_CPU, empty.view(), empty.view(), empty.view(), empty.view())
            .expect("owner comparison");
        assert_eq!(agreement.ratio, f64::NEG_INFINITY);
        let wire = stage_agreement_report(STAGE_FIELDS, agreement).expect("projection");
        assert_eq!(wire.ratio, StageRatio::EmptyStage);

        // Through the surface: a block receipt over no input rows.
        let no_rows = dyadic_block_from(
            array![[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]],
            array![0.0625, -0.125],
            Array2::zeros((0, 2)),
        );
        let output = run_parameter_decomposition(
            &receipt_request("relu", "float64", false),
            &block_tensors(&no_rows),
        )
        .expect("surface run over no rows");
        let report = receipt_report(&output);
        assert_eq!(report.pre_activation.ratio, StageRatio::EmptyStage, "{report:?}");
        assert_eq!(report.output.ratio, StageRatio::EmptyStage, "{report:?}");
        // Positive control: the same block over its rows takes finite ratios.
        let rows = run_parameter_decomposition(
            &receipt_request("relu", "float64", false),
            &block_tensors(&dyadic_block()),
        )
        .expect("surface run");
        assert_eq!(receipt_report(&rows).output.ratio, StageRatio::Finite { value: 0.0 });
    }

    #[test]
    fn a_non_finite_ratio_no_owner_case_names_is_refused() {
        let unnamed = StageAgreement {
            agrees: false,
            refutes: false,
            witness: (0, 0),
            discrepancy: 0.0,
            band: 0.0,
            ratio: f64::NAN,
        };
        assert!(matches!(
            stage_agreement_report(STAGE_FIELDS, unnamed),
            Err(MpdSurfaceError::NonFiniteReport {
                field: "stage.ratio",
                ..
            })
        ));
        // Positive control: the same stage with a finite ratio projects.
        let finite_stage = StageAgreement {
            ratio: 0.5,
            ..unnamed
        };
        assert_eq!(
            stage_agreement_report(STAGE_FIELDS, finite_stage).expect("projection").ratio,
            StageRatio::Finite { value: 0.5 }
        );
    }

    #[test]
    fn a_receipt_request_the_owner_cannot_certify_is_refused() {
        let block = dyadic_block();
        let tensors = block_tensors(&block);
        // Positive control: the accepted request.
        assert!(run_parameter_decomposition(&receipt_request("relu", "float64", false), &tensors).is_ok());
        assert!(matches!(
            run_parameter_decomposition(&receipt_request("relu", "float32", false), &tensors),
            Err(MpdSurfaceError::Receipt(ReceiptRefusal::ExternalPrecision {
                float64: false,
                tf32_matmul: false,
            }))
        ));
        assert!(matches!(
            run_parameter_decomposition(&receipt_request("relu", "float64", true), &tensors),
            Err(MpdSurfaceError::Receipt(ReceiptRefusal::ExternalPrecision {
                float64: true,
                tf32_matmul: true,
            }))
        ));
        assert!(matches!(
            run_parameter_decomposition(&receipt_request("gelu_new", "float64", false), &tensors),
            Err(MpdSurfaceError::Receipt(ReceiptRefusal::Activation(
                GaussianActivationError::ApproximateGelu { .. }
            )))
        ));
        let mut misshapen = tensors.clone();
        misshapen.insert(
            "b1".to_string(),
            block.bias.clone().insert_axis(Axis(1)).into_dyn(),
        );
        assert!(matches!(
            run_parameter_decomposition(&receipt_request("relu", "float64", false), &misshapen),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
        let unknown_execution_field = receipt_request("relu", "float64", false).replacen(
            "\"tf32_matmul\"",
            "\"precision\": \"high\", \"tf32_matmul\"",
            1,
        );
        assert!(matches!(
            run_parameter_decomposition(&unknown_execution_field, &tensors),
            Err(MpdSurfaceError::InvalidRequest(..))
        ));
    }
}
