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
//! absent value with that stated meaning. Any other non-finite value is refused
//! instead of being written as JSON `null`.

use std::collections::BTreeMap;
use std::fmt;

use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{ArrayD, ArrayView1, ArrayView2, Ix1, Ix2};
use serde::{Deserialize, Serialize};

use super::secant::SecantError;
use super::attention::AttentionProgramError;
use super::bounds::BoundError;
use super::dense::DenseError;
use super::canonical::CanonicalRefusal;
use super::gauge::GaugeRefusal;
use super::gauge_census::CensusRefusal;
use super::joint_operators::JointRefusal;
use super::state::StateError;
use crate::finite_grid::FiniteGridError;
use gam_sae::response::interaction::InteractionError;
use super::supports::EvidenceStatusError;
use super::spectral::{
    PlaneRotationError, PlaneRotationRecovery, RotationAmbiguity, RotationClusterKind,
    recover_plane_rotations,
};

mod bounds;
mod canonical;
mod code;
mod compile;
mod dense;
mod finite_grid;
mod gauge_census;
mod joint;
mod layer;
mod module_split;
mod observability;
mod secant;
mod sign_gated;
mod state_quotient;
mod verify;

pub use bounds::{
    AttentionReadRegionReport, BoxStatusWire, HeadReadRequest, LogitBoundRequest, LogitBoundsReport,
    LogitBoundsRequest, LogitBoxRegion, LogitBoxes,
};
pub use canonical::{
    CanonicalLayerReport, CanonicalLayerRequest, DefectReport, ElementsReport, ExecuteRequest,
    ExecutionIds, ExecutionReport, NormReport, ProjectionReport, QueryKeyElementReport,
    QueryKeyNormReport,
};
pub use code::{
    CodeItem, CodeLengthReport, CodeLengthsReport, CodeLengthsRequest, DecideProposalReport,
    DecideProposalRequest, EvidenceStatusWire, ExactBasisWire, ExtremumWire, FidelityVerdictWire,
    LatticeReport, ProposalDecision, ProposalKindWire, StatedArtifact,
};
pub use compile::{CompileFindings, CompileProblem, CompileReport, CompileRequest, RealizationWire, WitnessWire};
pub use dense::{
    AssemblyRequest, CutoffRequest, DenseOperation, DenseReport, DenseRequest, DenseResult,
    QrModeRequest,
};
pub use finite_grid::{
    AdditiveAcrossPairReport, BandedEnergyReport, CrossBlockReport, FiniteGridReport,
    FiniteGridRequest, GridCells, GridReindex, OutputFactor, RectangleComplementReport,
    RectangleReport,
};
pub use gauge_census::{
    CensusBlock, CensusBlockReport, CensusChargeReport, GaugeCensusReport, GaugeCensusRequest,
    GaugeFamilyReport, NamedCharge, NamedFamily, TiedResidualReport,
};
pub use joint::{
    ComparisonReport, FactorsReport, GramReport, HeadEnergyReport, HeadOperatorsReport,
    JointOperatorsReport, JointOperatorsRequest, OperatorPairReport, OperatorRef, PairStatusWire,
};
pub use module_split::{
    AdditiveBlocksReport, MlpBlockRequest, ModuleSplitReport, ModuleSplitRequest,
    NormalFormReport, OptimalSplitReport, SplitDomainReport, SplitStatusWire, UnitSourceReport,
    UnresolvedJoinReport,
};
pub use observability::{
    CaptureReport, LetterRequest, SpectrumReport, StateDomainReport, StateStatusWire, StepRequest,
    WeightedObservabilityReport, WeightedObservabilityRequest,
};
pub use layer::{
    AttentionRequest, GeometryRequest, ProjectionRequest, QueryKeyNormRequest, RmsNormRequest,
    RotaryPairingRequest, RotaryRequest,
};
pub use secant::{SecantActivationWire, SecantOperator, SecantReport, SecantRequest};
pub use sign_gated::{
    PerRowReport, ReadoutReport, ReadoutRequest, RowsReport, SignGatedSwigluReport, SignGatedSwigluRequest,
};

pub use verify::{
    BandedLogits, FamilyDomainReport, FamilyStatusWire, FamilyVerificationReport,
    FamilyWitnessReport, ToleranceRequest, VerifyLogitsReport, VerifyLogitsRequest,
};
pub use state_quotient::{
    LinearChart, LinearClosedChartReport, LinearClosedChartRequest, LinearStateQuotientReport,
    LinearStateQuotientRequest, SpectralNormBoundsReport,
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
        /// Id of the input array holding the matrix. The owner measures its distance to
        /// the orthogonal group and reports it as `orthogonality_defect`.
        tensor: String,
    },
    /// The linear sufficient state of linear readouts under linear transitions
    /// (`state::LinearStateQuotient`): the closed observable chart, or a declared one
    /// measured.
    LinearStateQuotient(LinearStateQuotientRequest),
    /// The closed observable chart alone, unmeasured
    /// (`state::LinearStateQuotient::closed_chart`), for callers that read only the
    /// resolved row space and its rank.
    LinearClosedChart(LinearClosedChartRequest),
    /// Exact bit counts of declared message items (`codec`, `precision::LatticeCode`).
    CodeLengths(CodeLengthsRequest),
    /// One structural proposal decided by `fit::decide_proposal` from supplied code
    /// lengths and evidence statuses.
    DecideProposal(DecideProposalRequest),
    /// One exact two-endpoint change operator with its bands (`secant`).
    Secant(SecantRequest),
    /// The exhaustive FANOVA of a response on a declared finite product grid
    /// (`response::finite_grid`, `response::interaction`).
    FiniteGrid(FiniteGridRequest),
    /// The implementation-gauge families of declared blocks, with their census charges
    /// (`gauge`, `gauge_census`).
    GaugeCensus(GaugeCensusRequest),
    /// The canonical gauge form of a native decoder layer, with the group elements
    /// applied (`canonical::DecoderLayer::canonical`).
    CanonicalLayer(Box<CanonicalLayerRequest>),
    /// Exhaustive verification and the counterfactual contract over supplied banded
    /// logits (`verify::verify_counterfactual_contract`).
    VerifyLogits(VerifyLogitsRequest),
    /// One dense float64 decomposition on faer with canonical signs (`dense`):
    /// `eigh`, `eigvalsh`, `svd`, `svdvals`, `qr`, `solve`, `lstsq`, `spectral_norm`.
    Dense(DenseOperation),
    /// The factored gauge-invariant query/key and value/output operators of an
    /// attention block, their energies, Grams and comparisons (`joint_operators`).
    JointOperators(Box<JointOperatorsRequest>),
    /// The weighted observability Gramian of readouts pulled back through declared
    /// steps, and candidate captures (`state::WeightedObservability`).
    WeightedObservability(WeightedObservabilityRequest),
    /// An MLP block's merged normal form, finest additive blocks, optimal splits and
    /// pair-weight Laplacian products (`module_split`).
    ModuleSplit(ModuleSplitRequest),
    /// The native edit compiler: control settings to native parameter edits, or
    /// infeasibility witnesses (`compile`).
    Compile(Box<CompileRequest>),
    /// The executed sign-gated split `F = P + R` of a residual SwiGLU block, its correction
    /// bounds and SiLU → ReLU replacement contract (`sign_gated`).
    SignGatedSwiglu(Box<SignGatedSwigluRequest>),
    /// The logit-box KL and total-variation bounds, and the attention-read bound
    /// (`bounds`).
    LogitBounds(LogitBoundsRequest),
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
    LinearStateQuotient(LinearStateQuotientReport),
    LinearClosedChart(LinearClosedChartReport),
    CodeLengths(CodeLengthsReport),
    DecideProposal(DecideProposalReport),
    Secant(SecantReport),
    FiniteGrid(FiniteGridReport),
    GaugeCensus(GaugeCensusReport),
    CanonicalLayer(Box<CanonicalLayerReport>),
    VerifyLogits(VerifyLogitsReport),
    Dense(DenseResult),
    JointOperators(Box<JointOperatorsReport>),
    WeightedObservability(WeightedObservabilityReport),
    ModuleSplit(Box<ModuleSplitReport>),
    Compile(Box<CompileReport>),
    SignGatedSwiglu(Box<SignGatedSwigluReport>),
    LogitBounds(LogitBoundsReport),
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
    State(StateError),
    /// A code owner refused an item.
    Code(String),
    /// An evidence-status constructor refused a supplied status.
    Evidence(EvidenceStatusError),
    Secant(SecantError),
    FiniteGrid(FiniteGridError),
    Interaction(InteractionError),
    Gauge(GaugeRefusal),
    Census(CensusRefusal),
    Attention(AttentionProgramError),
    Canonical(Box<CanonicalRefusal>),
    Dense(DenseError),
    Joint(JointRefusal),
    Bound(BoundError),
    /// The module-split owner refused the block, a subset or a vector.
    ModuleSplit(String),
    /// The sign-gated split owner refused the block, the rows or the readout.
    SignGated(String),
    /// The verification owner refused the family, the tolerance or a row.
    Verify(String),
    /// A dense copy the surface forms does not fit the memory budget.
    Memory(MemoryReservationError),
    /// The native edit compiler refused the problem.
    Compile(Box<super::compile::CompileError>),
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
            Self::State(error) => write!(formatter, "{error}"),
            Self::Memory(error) => write!(formatter, "{error}"),
            Self::Code(reason) => write!(formatter, "{reason}"),
            Self::Evidence(error) => write!(formatter, "{error}"),
            Self::Secant(error) => write!(formatter, "{error}"),
            Self::FiniteGrid(error) => write!(formatter, "{error}"),
            Self::Interaction(error) => write!(formatter, "{error}"),
            Self::Gauge(refusal) => write!(formatter, "gauge refused: {refusal:?}"),
            Self::Census(refusal) => write!(formatter, "gauge census refused: {refusal:?}"),
            Self::Attention(error) => write!(formatter, "{error}"),
            Self::Canonical(refusal) => write!(formatter, "canonical form refused: {refusal:?}"),
            Self::Verify(reason) => write!(formatter, "verification refused: {reason}"),
            Self::Dense(error) => write!(formatter, "{error}"),
            Self::Joint(refusal) => write!(formatter, "joint operators refused: {refusal:?}"),
            Self::Bound(error) => write!(formatter, "{error}"),
            Self::ModuleSplit(reason) => write!(formatter, "{reason}"),
            Self::Compile(error) => write!(formatter, "{error}"),
            Self::SignGated(reason) => write!(formatter, "{reason}"),
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
///
/// Every dense allocation of the run reserves on `governor` first. The CLI and the
/// Python entry pass the process-wide governor; a caller with a budget of its own
/// passes that.
pub fn run_parameter_decomposition(
    request_json: &str,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let request = MpdRequest::from_json(request_json)?;
    match request.operation {
        MpdOperation::RecoverPlaneRotations { tensor } => {
            let recovery = recover_plane_rotations(governor, matrix(tensors, &tensor)?)
                .map_err(MpdSurfaceError::PlaneRotation)?;
            project_plane_rotations(tensor, recovery)
        }
        MpdOperation::LinearStateQuotient(request) => {
            state_quotient::run(request, tensors, governor)
        }
        MpdOperation::LinearClosedChart(request) => state_quotient::run_chart(request, tensors, governor),
        MpdOperation::CodeLengths(request) => code::run_lengths(request, tensors, governor),
        MpdOperation::DecideProposal(request) => code::run_decide(request),
        MpdOperation::Secant(request) => secant::run(request, tensors, governor),
        MpdOperation::FiniteGrid(request) => finite_grid::run(request, tensors, governor),
        MpdOperation::GaugeCensus(request) => gauge_census::run(request, tensors, governor),
        MpdOperation::CanonicalLayer(request) => canonical::run(*request, tensors, governor),
        MpdOperation::VerifyLogits(request) => verify::run(request, tensors, governor),
        MpdOperation::Dense(request) => dense::run(request, tensors, governor),
        MpdOperation::JointOperators(request) => joint::run(*request, tensors, governor),
        MpdOperation::WeightedObservability(request) => observability::run(request, tensors, governor),
        MpdOperation::ModuleSplit(request) => module_split::run(request, tensors, governor),
        MpdOperation::Compile(request) => compile::run(*request, tensors, governor),
        MpdOperation::SignGatedSwiglu(request) => sign_gated::run(*request, tensors, governor),
        MpdOperation::LogitBounds(request) => bounds::run(request, tensors),
    }
}

/// Wraps one operation's result in the versioned report.
fn output(result: MpdResult, arrays: BTreeMap<String, ArrayD<f64>>) -> MpdOutput {
    MpdOutput {
        report: MpdReport {
            schema: MPD_REPORT_SCHEMA,
            schema_version: MPD_SCHEMA_VERSION,
            result,
        },
        arrays,
    }
}

/// Reserves `copies` dense `rows × cols` matrices the surface forms or an ungoverned
/// owner allocates.
fn reserve(
    governor: &MemoryGovernor,
    rows: usize,
    cols: usize,
    copies: usize,
    context: &'static str,
) -> Result<MemoryReservation, MpdSurfaceError> {
    governor
        .try_reserve_dense_f64_copies(rows, cols, copies, context)
        .map_err(MpdSurfaceError::Memory)
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
    use super::*;
    use crate::test_support::test_governor;
    use ndarray::{Array2, array};

    pub(super) fn request_json(operation: &str) -> String {
        format!(
            r#"{{"schema": "{MPD_REQUEST_SCHEMA}", "schema_version": {MPD_SCHEMA_VERSION}, "operation": {operation}}}"#
        )
    }

    fn plane_request() -> String {
        request_json(r#"{"kind": "recover_plane_rotations", "tensor": "w"}"#)
    }

    fn inputs(id: &str, matrix: Array2<f64>) -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([(id.to_string(), matrix.into_dyn())])
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
        let valid = plane_request();
        let request = MpdRequest::from_json(&valid).expect("a declared request parses");
        assert_eq!(
            request.operation,
            MpdOperation::RecoverPlaneRotations {
                tensor: "w".to_string(),
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
            r#"{"kind": "recover_plane_rotations", "tensor": "w", "tolerance": 1e-8}"#,
        );
        assert!(MpdRequest::from_json(&unknown_operation_field).is_err());
        // The distance to the orthogonal group is measured by the owner, so a
        // hand-supplied one is an unknown field, not an input.
        let hand_supplied_error = request_json(
            r#"{"kind": "recover_plane_rotations", "tensor": "w", "declared_error": 0.0}"#,
        );
        assert!(MpdRequest::from_json(&hand_supplied_error).is_err());
        let missing_tensor = request_json(r#"{"kind": "recover_plane_rotations"}"#);
        assert!(MpdRequest::from_json(&missing_tensor).is_err());
        let unknown_kind = request_json(r#"{"kind": "guess_the_planes", "tensor": "w"}"#);
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
        let direct = recover_plane_rotations(test_governor(), matrix.view()).expect("owner recovery");
        let output = run_parameter_decomposition(&plane_request(), &inputs("w", matrix), test_governor())
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
        let output = run_parameter_decomposition(&plane_request(), &inputs("w", Array2::eye(3)), test_governor())
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
    fn the_owners_refusal_of_a_singular_matrix_reaches_the_caller() {
        // A rank-one matrix has no unique nearest orthogonal matrix.
        assert!(matches!(
            run_parameter_decomposition(&plane_request(), &inputs("w", Array2::ones((2, 2))), test_governor()),
            Err(MpdSurfaceError::PlaneRotation(
                PlaneRotationError::NotInvertible { .. }
            ))
        ));
        // Positive control: the same request on an orthogonal matrix is accepted.
        assert!(run_parameter_decomposition(&plane_request(), &inputs("w", Array2::eye(2)), test_governor()).is_ok());
    }

    #[test]
    fn missing_or_misshapen_inputs_are_refused() {
        let json = plane_request();
        assert!(run_parameter_decomposition(&json, &inputs("w", Array2::eye(2)), test_governor()).is_ok());
        assert!(matches!(
            run_parameter_decomposition(&json, &inputs("other", Array2::eye(2)), test_governor()),
            Err(MpdSurfaceError::MissingTensor { .. })
        ));
        let cube = BTreeMap::from([("w".to_string(), ArrayD::<f64>::zeros(vec![2, 2, 2]))]);
        assert!(matches!(
            run_parameter_decomposition(&json, &cube, test_governor()),
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
}
