//! `compile`: the native edit compiler ([`crate::compile`]) on the
//! wire. A request declares the teacher's tensor registry (storage ids naming input arrays,
//! aliases, use sites) and one compilation problem; the report carries the compiled
//! control's status and its plan, each global edit as the ids of its `left` and `right`
//! factor arrays (`ΔW = left · rightᵀ`, stored orientation), which
//! `gamfit.torch.parameter_interventions.compiled_parameter_edits` turns into the torch
//! runner's `GlobalParameterEdit`s.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

use super::code::{EvidenceStatusWire, ExactBasisWire, ExtremumWire};
use super::{MpdOutput, MpdResult, MpdSurfaceError, absent_when_infinite, finite, input, matrix, output, vector};
use crate::attention::RotaryEmbedding;
use crate::compile::bilinear::{
    HeadRows, QueryKeyDomain, QueryKeyEditProblem, QueryKeySolveProblem, ScoreClaim, ScoreFamily, ScoreRequirement,
    compile_query_key_edit, solve_query_key_setting,
};
use crate::compile::path::{
    PathActivation, PathFamily, PathLayer, PathProblem, PathSite, PathWitness, compile_path,
};
use crate::compile::chart::{
    ChartSetting, FactorBinding, FixedRankChart, compile_chart_edit,
};
use crate::compile::controls::{
    ControlFamily, CoupledControlProblem, CouplingClass, CouplingWitness, GainAxis, GainSite, ReaderHomogeneity,
    ReaderSite, SiteChoice, SupportNode, compile_coupled_controls,
};
use crate::compile::linear::{
    ConstraintSide, Coverage, EditMetric, LinearSiteDomain, LinearSiteProblem, LinearWitness, OffTargetDomain,
    OffTargetInputs, OffTargetWitness, Requirement, ResponseClass, compile_linear_site,
};
use crate::compile::null::{EditBall, EditDirection, NullEditOutcome, physically_null_supremum};
use crate::compile::ties::{BlockRef, TieConstraint};
use crate::compile::{CompileError, CompiledControl, ControlRealization, DescriptiveReason};
use crate::gauge::AllInputs;
use crate::lift::{TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId};
use crate::supports::{EvidenceStatus, ExactBasis, Extremum};

/// A compilation problem over a declared registry.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompileRequest {
    /// The compiled control's name, echoed in the report.
    pub control: String,
    pub registry: RegistryRequest,
    pub problem: CompileProblem,
}

/// The teacher's tensors as the executing framework names them.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryRequest {
    /// Storage tensor id → id of the input array holding its values.
    pub storage: BTreeMap<String, String>,
    /// Alias → the storage it names.
    pub aliases: BTreeMap<String, String>,
    pub use_sites: Vec<UseSiteRequest>,
}

/// One use site: the name it reads and what it does with it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UseSiteRequest {
    pub site: String,
    pub reads: String,
    pub map: UseMapWire,
}

/// [`UseMap`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UseMapWire {
    Identity,
    Transpose,
    Stored,
}

/// The owner problems. Every array field is the id of an input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CompileProblem {
    /// `compile::linear::compile_linear_site`.
    LinearSite {
        storage: String,
        /// Id of the stored values `θ` of the edited tensor.
        native: String,
        requirements: Vec<RequirementWire>,
        /// Off-target inputs of linear uses: `[site, inputs id]`.
        off_target: Vec<[String; 2]>,
        row_scale: Option<String>,
        col_scale: Option<String>,
        ties: Vec<TieWire>,
    },
    /// `compile::controls::compile_coupled_controls`.
    CoupledControls {
        writer: String,
        alpha: String,
        writer_site: Option<GainSiteWire>,
        reader_site: Option<ReaderSiteWire>,
    },
    /// `compile::chart::compile_chart_edit` on the chart of `write · read`; absent blocks
    /// are unchanged.
    FixedRankChart {
        write: String,
        read: String,
        a: Option<String>,
        b: Option<String>,
        c: Option<String>,
        write_binding: FactorBindingWire,
        read_binding: FactorBindingWire,
    },
    /// `compile::bilinear::compile_query_key_edit`.
    QueryKeyEdit(Box<QueryKeyRequest>),
    /// `compile::null::physically_null_supremum`.
    NullEdit { edit_gram: String, response_gram: String },
    /// `compile::path::compile_path`: a set-type requirement downstream of norms and MLPs,
    /// solved at the path's editable linear sites through the exact forward.
    Path(Box<PathRequest>),
    /// `compile::bilinear::solve_query_key_setting`: set-type scores solved for `(Q′, K′)`.
    QueryKeySolve(Box<QueryKeySolveRequest>),
}

/// A linear read of a path: storage, the id of its weight (`out × in`) and optional bias.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PathSiteWire {
    pub storage: String,
    pub weight: String,
    pub bias: Option<String>,
    pub editable: bool,
}

/// [`PathLayer`] on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PathLayerWire {
    Linear { site: usize },
    RmsNorm { gain: Option<String>, epsilon: f64 },
    Silu {},
    ExactGelu {},
    Relu {},
    Swiglu { gate: usize, up: usize },
    Residual { layers: Vec<PathLayerWire> },
}

/// A path requirement.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PathRequest {
    pub sites: Vec<PathSiteWire>,
    pub layers: Vec<PathLayerWire>,
    pub inputs: String,
    pub targets: String,
    pub target_radius: f64,
    pub off_target: Option<String>,
    pub max_iterations: usize,
}

/// A set-type score requirement solved on both sides of one head.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryKeySolveRequest {
    pub query: String,
    pub key: String,
    pub query_rows: HeadRowsWire,
    pub key_rows: HeadRowsWire,
    pub rotary: Option<RotaryEmbedding>,
    pub score_scale: f64,
    pub queries: String,
    pub query_positions: Vec<i64>,
    pub keys: String,
    pub key_positions: Vec<i64>,
    pub causal: bool,
    /// `[query row, key row, target score]` triples; the rows are integral.
    pub requirements: Vec<(usize, usize, f64)>,
    pub target_radius: f64,
    pub max_rounds: usize,
}

/// One requirement on the edited storage.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RequirementWire {
    /// Set the use's output on `inputs` to `targets`.
    Linear {
        site: String,
        inputs: String,
        targets: String,
        target_radius: f64,
        class: ResponseClassWire,
    },
    /// Set stored rows to `targets`.
    StoredRows {
        rows: Vec<usize>,
        targets: String,
        target_radius: f64,
    },
}

/// [`ResponseClass`] on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseClassWire {
    Sample,
    AllInputs,
    Span { basis: String },
}

/// [`TieConstraint`] on the wire: `second = scale · first` (transposed when declared).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TieWire {
    pub first: BlockWire,
    pub second: BlockWire,
    pub transposed: bool,
    pub scale: f64,
}

/// A block `[row_start, row_end) × [col_start, col_end)` of a storage tensor.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BlockWire {
    pub storage: String,
    pub rows: [usize; 2],
    pub cols: [usize; 2],
}

/// [`GainSite`] on the wire.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GainSiteWire {
    pub storage: String,
    /// Id of the stored weight's values.
    pub weight: String,
    pub axis: GainAxisWire,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GainAxisWire {
    Rows,
    Columns,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderSiteWire {
    pub site: GainSiteWire,
    pub homogeneity: HomogeneityWire,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HomogeneityWire {
    Linear,
    PositivelyHomogeneous,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FactorBindingWire {
    pub storage: String,
    pub stored_transposed: bool,
}

/// A query/key edit to certify.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryKeyRequest {
    pub query: String,
    pub key: String,
    /// The set values `Q′`, `K′`.
    pub query_setting: String,
    pub key_setting: String,
    pub query_rows: HeadRowsWire,
    pub key_rows: HeadRowsWire,
    pub rotary: Option<RotaryEmbedding>,
    pub score_scale: f64,
    pub queries: String,
    pub query_positions: Vec<i64>,
    pub keys: String,
    pub key_positions: Vec<i64>,
    pub causal: bool,
    /// Id of declared score changes (`queries × keys`), or absent for the first order.
    pub claimed_changes: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HeadRowsWire {
    pub storage: String,
    pub row_offset: usize,
}

/// The compiled control and its problem's own findings.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompileReport {
    pub control: String,
    /// The compiled control's status; absent for a null-edit supremum, which compiles no
    /// control.
    pub realization: Option<RealizationWire>,
    /// Absent when no native control was established; empty for `ρ(0) = θ`.
    pub plan: Option<Vec<PlanEditReport>>,
    pub findings: CompileFindings,
}

/// [`ControlRealization`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RealizationWire {
    ExactlyRealized { residual: EvidenceStatusWire<WitnessWire, String> },
    EmpiricallyValidated { residual: EvidenceStatusWire<WitnessWire, String> },
    Descriptive {
        reason: DescriptiveReasonWire,
        witness: Option<EvidenceStatusWire<WitnessWire, String>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptiveReasonWire {
    KernelViolation,
    IncompatibleSides,
    TieBroken,
    CoupledControls,
    ResidualResolved,
}

/// Every owner witness on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WitnessWire {
    Kernel {
        side: SideWire,
        /// Id of the unit direction over the side's constraint columns.
        direction: String,
        input_norm_upper: f64,
        /// Absent when `X v = 0` is certified, so no finite edit realizes the request.
        edit_norm_lower_bound: Option<f64>,
    },
    Compatibility { left: usize, right: usize },
    OffTarget { set: usize, observation: usize },
    /// An observation and output coordinate of a path, or a requirement index of a score
    /// setting (`coordinate` 0).
    Entry { observation: usize, coordinate: usize },
    Residual { side: SideWire, column: usize, entry: usize },
    Tie { tie: usize, row: usize, col: usize },
    Coupling { path: Vec<SupportNodeWire> },
    /// Id of an input direction (chart operator difference).
    Input { direction: String },
    /// Id of an edit direction and its two quadratic forms.
    EditDirection { direction: String, edit_size: f64, response_error: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SideWire {
    Right,
    Left,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SupportNodeWire {
    Control { index: usize },
    Unit { index: usize },
}

/// One global edit: ids of its factor arrays.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlanEditReport {
    pub storage: String,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub left: String,
    pub right: String,
}

/// Findings specific to each problem.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompileFindings {
    LinearSite {
        metric_norm: Option<[f64; 2]>,
        right_rank: usize,
        left_rank: usize,
        coverage: Vec<CoverageWire>,
        undeclared_uses: Vec<String>,
        allowance: [f64; 2],
        /// `sup ‖ΔW x‖₂` over the declared off-target inputs (approximate-transformation
        /// distance restricted to them).
        off_target_damage: Option<EvidenceStatusWire<WitnessWire, String>>,
    },
    CoupledControls {
        writer_classes: Vec<ClassWire>,
        sign_classes: Vec<ClassWire>,
        best_writer_gains: String,
        writer_residual: [f64; 2],
        choice: Option<SiteChoiceWire>,
        unit_gains: Option<String>,
        control_gains: Option<String>,
    },
    FixedRankChart {
        lifted_write: String,
        lifted_read: String,
        dependent: String,
        dependent_frobenius_band: f64,
    },
    QueryKeyEdit {
        exact_change: String,
        exact_change_bands: String,
        first_order: String,
        cross: String,
        claim_total_variation: Vec<f64>,
    },
    Path {
        iterations: usize,
        residual: String,
        residual_band: String,
        off_target_damage: Option<EvidenceStatusWire<WitnessWire, String>>,
    },
    QueryKeySolve {
        query_setting: String,
        key_setting: String,
        rounds: usize,
        /// `[residual, band]` per requirement.
        residuals: Vec<[f64; 2]>,
        claim_total_variation: Vec<f64>,
    },
    /// `sup uᵀKu` over the unit `G`-ball of `range(G)`, or the counterexample refusing it.
    NullEdit {
        refused: bool,
        supremum: EvidenceStatusWire<WitnessWire, String>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CoverageWire {
    SampleOnly,
    StoredRows,
    Covered,
    Uncovered { dimension: usize, direction: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ClassWire {
    pub controls: Vec<usize>,
    pub units: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteChoiceWire {
    Writer,
    Reader,
    WriterAndReader,
}

fn compile_error(error: CompileError) -> MpdSurfaceError {
    MpdSurfaceError::Compile(Box::new(error))
}

fn registry(request: &RegistryRequest, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<TensorRegistry, MpdSurfaceError> {
    let mut registry = TensorRegistry::default();
    let lift = |error| compile_error(CompileError::Lift(error));
    for (id, values) in &request.storage {
        registry
            .register_storage(TensorId(id.clone()), input(tensors, values)?.view())
            .map_err(lift)?;
    }
    for (alias, storage) in &request.aliases {
        registry
            .register_alias(TensorId(alias.clone()), TensorId(storage.clone()))
            .map_err(lift)?;
    }
    for site in &request.use_sites {
        let map = match site.map {
            UseMapWire::Identity => UseMap::Linear(TieOrientation::Identity),
            UseMapWire::Transpose => UseMap::Linear(TieOrientation::Transpose),
            UseMapWire::Stored => UseMap::Stored,
        };
        registry
            .register_use_site(UseSiteId(site.site.clone()), TensorId(site.reads.clone()), map)
            .map_err(lift)?;
    }
    Ok(registry)
}

/// Collects witness and output arrays under report-unique ids.
struct Arrays {
    arrays: BTreeMap<String, ArrayD<f64>>,
}

impl Arrays {
    fn put(&mut self, id: String, values: ArrayD<f64>) -> String {
        self.arrays.insert(id.clone(), values);
        id
    }

    fn vector(&mut self, id: &str, values: &[f64]) -> String {
        self.put(id.to_string(), Array1::from(values.to_vec()).into_dyn())
    }
}

fn status_wire<W, D>(
    status: EvidenceStatus<W, D>,
    arrays: &mut Arrays,
    witness_of: &impl Fn(W, &mut Arrays) -> Result<WitnessWire, MpdSurfaceError>,
    domain_of: &impl Fn(&D) -> String,
) -> Result<EvidenceStatusWire<WitnessWire, String>, MpdSurfaceError> {
    Ok(match status {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            basis,
            witness,
            domain,
            ..
        } => EvidenceStatusWire::Exact {
            value,
            numerical_error,
            basis: match basis {
                ExactBasis::Algebraic => ExactBasisWire::Algebraic {},
                ExactBasis::Exhaustive { cardinality } => ExactBasisWire::Exhaustive { cardinality },
            },
            witness: witness.map(|witness| witness_of(witness, arrays)).transpose()?,
            domain: domain_of(&domain),
        },
        EvidenceStatus::UniformBound {
            upper,
            numerical_error,
            region,
            ..
        } => EvidenceStatusWire::UniformBound {
            upper,
            numerical_error,
            region: domain_of(&region),
        },
        EvidenceStatus::StatisticalEstimate {
            estimate,
            standard_error,
            samples,
            law,
            ..
        } => EvidenceStatusWire::StatisticalEstimate {
            estimate,
            standard_error,
            samples,
            law: domain_of(&law),
        },
        EvidenceStatus::Counterexample {
            value,
            numerical_error,
            threshold,
            witness,
            ..
        } => EvidenceStatusWire::Counterexample {
            value,
            numerical_error,
            threshold,
            witness: witness_of(witness, arrays)?,
        },
        EvidenceStatus::Unresolved {
            lower,
            upper,
            extremum,
            witness,
            domain,
            ..
        } => EvidenceStatusWire::Unresolved {
            lower: lower.is_finite().then_some(lower),
            upper: upper.is_finite().then_some(upper),
            extremum: match extremum {
                Extremum::Supremum => ExtremumWire::Supremum,
                Extremum::Infimum => ExtremumWire::Infimum,
            },
            witness: witness.map(|witness| witness_of(witness, arrays)).transpose()?,
            domain: domain_of(&domain),
        },
    })
}

fn realization<W, D>(
    realization: ControlRealization<W, D>,
    arrays: &mut Arrays,
    witness_of: impl Fn(W, &mut Arrays) -> Result<WitnessWire, MpdSurfaceError>,
    domain_of: impl Fn(&D) -> String,
) -> Result<RealizationWire, MpdSurfaceError> {
    Ok(match realization {
        ControlRealization::ExactlyRealized { residual, .. } => RealizationWire::ExactlyRealized {
            residual: status_wire(residual, arrays, &witness_of, &domain_of)?,
        },
        ControlRealization::EmpiricallyValidated { residual, .. } => RealizationWire::EmpiricallyValidated {
            residual: status_wire(residual, arrays, &witness_of, &domain_of)?,
        },
        ControlRealization::Descriptive { reason, witness, .. } => RealizationWire::Descriptive {
            reason: match reason {
                DescriptiveReason::KernelViolation => DescriptiveReasonWire::KernelViolation,
                DescriptiveReason::IncompatibleSides => DescriptiveReasonWire::IncompatibleSides,
                DescriptiveReason::TieBroken => DescriptiveReasonWire::TieBroken,
                DescriptiveReason::CoupledControls => DescriptiveReasonWire::CoupledControls,
                DescriptiveReason::ResidualResolved => DescriptiveReasonWire::ResidualResolved,
            },
            witness: witness
                .map(|status| status_wire(status, arrays, &witness_of, &domain_of))
                .transpose()?,
        },
    })
}

fn side(side: ConstraintSide) -> SideWire {
    match side {
        ConstraintSide::Right => SideWire::Right,
        ConstraintSide::Left => SideWire::Left,
    }
}

fn linear_witness(witness: LinearWitness, arrays: &mut Arrays) -> Result<WitnessWire, MpdSurfaceError> {
    Ok(match witness {
        LinearWitness::Kernel {
            side: constraint_side,
            direction,
            input_norm_upper,
            edit_norm_lower_bound,
        } => WitnessWire::Kernel {
            side: side(constraint_side),
            direction: arrays.vector("witness/kernel_direction", &direction),
            input_norm_upper: finite("input_norm_upper", input_norm_upper)?,
            edit_norm_lower_bound: absent_when_infinite("edit_norm_lower_bound", edit_norm_lower_bound)?,
        },
        LinearWitness::Compatibility { left, right } => WitnessWire::Compatibility { left, right },
        LinearWitness::Residual {
            side: constraint_side,
            column,
            entry,
        } => WitnessWire::Residual {
            side: side(constraint_side),
            column,
            entry,
        },
        LinearWitness::Tie(violation) => WitnessWire::Tie {
            tie: violation.tie,
            row: violation.row,
            col: violation.col,
        },
    })
}

fn plan_report<W, D>(compiled: &CompiledControl<W, D>, arrays: &mut Arrays) -> Option<Vec<PlanEditReport>> {
    compiled.plan.as_ref().map(|plan| {
        plan.edits()
            .iter()
            .enumerate()
            .map(|(index, edit)| PlanEditReport {
                storage: edit.storage.0.clone(),
                rows: edit.delta.output_dim(),
                cols: edit.delta.input_dim(),
                rank: edit.delta.term_count(),
                left: arrays.put(format!("plan/{index}/left"), edit.delta.left().to_owned().into_dyn()),
                right: arrays.put(format!("plan/{index}/right"), edit.delta.right().to_owned().into_dyn()),
            })
            .collect()
    })
}

fn classes(classes: Vec<CouplingClass>) -> Vec<ClassWire> {
    classes
        .into_iter()
        .map(|class| ClassWire {
            controls: class.controls,
            units: class.units,
        })
        .collect()
}

fn gain_site<'a>(
    wire: &GainSiteWire,
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
) -> Result<GainSite<'a>, MpdSurfaceError> {
    Ok(GainSite {
        storage: TensorId(wire.storage.clone()),
        weight: matrix(tensors, &wire.weight)?,
        axis: match wire.axis {
            GainAxisWire::Rows => GainAxis::Rows,
            GainAxisWire::Columns => GainAxis::Columns,
        },
    })
}

fn path_layers(
    wires: &[PathLayerWire],
    tensors: &BTreeMap<String, ArrayD<f64>>,
) -> Result<Vec<PathLayer>, MpdSurfaceError> {
    let mut layers = Vec::with_capacity(wires.len());
    for wire in wires {
        layers.push(match wire {
            PathLayerWire::Linear { site } => PathLayer::Linear { site: *site },
            PathLayerWire::RmsNorm { gain, epsilon } => PathLayer::RmsNorm {
                gain: gain.as_deref().map(|id| slice_of(tensors, id)).transpose()?,
                epsilon: *epsilon,
            },
            PathLayerWire::Silu {} => PathLayer::Activation(PathActivation::Silu),
            PathLayerWire::ExactGelu {} => PathLayer::Activation(PathActivation::ExactGelu),
            PathLayerWire::Relu {} => PathLayer::Activation(PathActivation::Relu),
            PathLayerWire::Swiglu { gate, up } => PathLayer::Swiglu { gate: *gate, up: *up },
            PathLayerWire::Residual { layers } => PathLayer::Residual(path_layers(layers, tensors)?),
        });
    }
    Ok(layers)
}

fn slice_of(tensors: &BTreeMap<String, ArrayD<f64>>, id: &str) -> Result<Vec<f64>, MpdSurfaceError> {
    Ok(vector(tensors, id)?.to_vec())
}

pub(super) fn run(
    request: CompileRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let registry = registry(&request.registry, tensors)?;
    let control = request.control.as_str();
    let mut arrays = Arrays { arrays: BTreeMap::new() };
    let (realization_wire, plan, findings) = match request.problem {
        CompileProblem::LinearSite {
            storage,
            native,
            requirements,
            off_target,
            row_scale,
            col_scale,
            ties,
        } => {
            let mut declared = Vec::with_capacity(requirements.len());
            for requirement in &requirements {
                declared.push(match requirement {
                    RequirementWire::Linear {
                        site,
                        inputs,
                        targets,
                        target_radius,
                        class,
                    } => Requirement::Linear {
                        site: UseSiteId(site.clone()),
                        inputs: matrix(tensors, inputs)?,
                        targets: matrix(tensors, targets)?,
                        target_radius: *target_radius,
                        class: match class {
                            ResponseClassWire::Sample => ResponseClass::Sample,
                            ResponseClassWire::AllInputs => ResponseClass::AllInputs,
                            ResponseClassWire::Span { basis } => ResponseClass::Span(matrix(tensors, basis)?),
                        },
                    },
                    RequirementWire::StoredRows {
                        rows,
                        targets,
                        target_radius,
                    } => Requirement::StoredRows {
                        rows: rows.clone(),
                        targets: matrix(tensors, targets)?,
                        target_radius: *target_radius,
                    },
                });
            }
            let scale = |id: &Option<String>| id.as_deref().map(|id| slice_of(tensors, id)).transpose();
            let metric = EditMetric::weighted(scale(&row_scale)?, scale(&col_scale)?).map_err(compile_error)?;
            let block = |wire: &BlockWire| BlockRef {
                storage: TensorId(wire.storage.clone()),
                rows: wire.rows[0]..wire.rows[1],
                cols: wire.cols[0]..wire.cols[1],
            };
            let mut off_target_sets = Vec::with_capacity(off_target.len());
            for [site, inputs] in &off_target {
                off_target_sets.push(OffTargetInputs {
                    site: UseSiteId(site.clone()),
                    inputs: matrix(tensors, inputs)?,
                });
            }
            let problem = LinearSiteProblem {
                registry: &registry,
                storage: TensorId(storage),
                native: matrix(tensors, &native)?,
                off_target: off_target_sets,
                requirements: declared,
                metric,
                ties: ties
                    .iter()
                    .map(|tie| TieConstraint {
                        first: block(&tie.first),
                        second: block(&tie.second),
                        transposed: tie.transposed,
                        scale: tie.scale,
                    })
                    .collect(),
            };
            let report = compile_linear_site(&problem, control, governor).map_err(compile_error)?;
            let plan = plan_report(&report.compiled, &mut arrays);
            let mut coverage = Vec::with_capacity(report.coverage.len());
            for (index, entry) in report.coverage.iter().enumerate() {
                coverage.push(match entry {
                    Coverage::SampleOnly => CoverageWire::SampleOnly,
                    Coverage::StoredRows => CoverageWire::StoredRows,
                    Coverage::Covered => CoverageWire::Covered,
                    Coverage::Uncovered { dimension, direction } => CoverageWire::Uncovered {
                        dimension: *dimension,
                        direction: arrays.vector(&format!("coverage/{index}/direction"), direction),
                    },
                });
            }
            let findings = CompileFindings::LinearSite {
                metric_norm: report
                    .metric_norm
                    .map(|(value, band)| Ok::<_, MpdSurfaceError>([finite("metric_norm", value)?, finite("metric_band", band)?]))
                    .transpose()?,
                right_rank: report.right_rank,
                left_rank: report.left_rank,
                coverage,
                undeclared_uses: report.undeclared_uses.iter().map(|site| site.0.clone()).collect(),
                allowance: [finite("allowance", report.allowance[0])?, finite("allowance", report.allowance[1])?],
                off_target_damage: report
                    .off_target_damage
                    .map(|status| {
                        status_wire(
                            status,
                            &mut arrays,
                            &|witness: OffTargetWitness, _: &mut Arrays| {
                                Ok(WitnessWire::OffTarget {
                                    set: witness.set,
                                    observation: witness.observation,
                                })
                            },
                            &|domain: &OffTargetDomain| format!("{} off-target observations", domain.observations),
                        )
                    })
                    .transpose()?,
            };
            let wire = realization(report.compiled.realization, &mut arrays, linear_witness, |domain: &LinearSiteDomain| {
                format!(
                    "{} right and {} left constraint columns{}",
                    domain.right_columns,
                    domain.left_columns,
                    if domain.classes_covered { ", spanning every declared class" } else { "" }
                )
            })?;
            (Some(wire), plan, findings)
        }
        CompileProblem::CoupledControls {
            writer,
            alpha,
            writer_site,
            reader_site,
        } => {
            let alpha = slice_of(tensors, &alpha)?;
            let problem = CoupledControlProblem {
                registry: &registry,
                writer: matrix(tensors, &writer)?,
                alpha: &alpha,
                writer_site: writer_site.as_ref().map(|site| gain_site(site, tensors)).transpose()?,
                reader_site: reader_site
                    .as_ref()
                    .map(|reader| {
                        Ok::<_, MpdSurfaceError>(ReaderSite {
                            site: gain_site(&reader.site, tensors)?,
                            homogeneity: match reader.homogeneity {
                                HomogeneityWire::Linear => ReaderHomogeneity::Linear,
                                HomogeneityWire::PositivelyHomogeneous => ReaderHomogeneity::PositivelyHomogeneous,
                            },
                        })
                    })
                    .transpose()?,
            };
            let report = compile_coupled_controls(&problem, control).map_err(compile_error)?;
            let plan = plan_report(&report.compiled, &mut arrays);
            let findings = CompileFindings::CoupledControls {
                writer_classes: classes(report.writer_classes),
                sign_classes: classes(report.sign_classes),
                best_writer_gains: arrays.vector("best_writer_gains", &report.best_writer_gains),
                writer_residual: [
                    finite("writer_residual", report.writer_residual.0)?,
                    finite("writer_residual_band", report.writer_residual.1)?,
                ],
                choice: report.choice.map(|choice| match choice {
                    SiteChoice::Writer => SiteChoiceWire::Writer,
                    SiteChoice::Reader => SiteChoiceWire::Reader,
                    SiteChoice::WriterAndReader => SiteChoiceWire::WriterAndReader,
                }),
                unit_gains: report.unit_gains.as_deref().map(|gains| arrays.vector("unit_gains", gains)),
                control_gains: report.control_gains.as_deref().map(|gains| arrays.vector("control_gains", gains)),
            };
            let wire = realization(
                report.compiled.realization,
                &mut arrays,
                |witness: CouplingWitness, _| {
                    Ok(WitnessWire::Coupling {
                        path: witness
                            .path
                            .into_iter()
                            .map(|node| match node {
                                SupportNode::Control(index) => SupportNodeWire::Control { index },
                                SupportNode::Unit(index) => SupportNodeWire::Unit { index },
                            })
                            .collect(),
                    })
                },
                |family: &ControlFamily| format!("every input, {} controls", family.controls),
            )?;
            (Some(wire), plan, findings)
        }
        CompileProblem::FixedRankChart {
            write,
            read,
            a,
            b,
            c,
            write_binding,
            read_binding,
        } => {
            let chart = FixedRankChart::from_factors(matrix(tensors, &write)?, matrix(tensors, &read)?)
                .map_err(compile_error)?;
            let block = |id: &Option<String>| id.as_deref().map(|id| matrix(tensors, id)).transpose();
            let setting = ChartSetting {
                a: block(&a)?,
                b: block(&b)?,
                c: block(&c)?,
            };
            let binding = |wire: &FactorBindingWire| FactorBinding {
                storage: TensorId(wire.storage.clone()),
                stored_transposed: wire.stored_transposed,
            };
            let report = compile_chart_edit(
                &registry,
                &chart,
                &setting,
                &binding(&write_binding),
                &binding(&read_binding),
                control,
            )
            .map_err(compile_error)?;
            let plan = plan_report(&report.compiled, &mut arrays);
            let findings = CompileFindings::FixedRankChart {
                lifted_write: arrays.put("lifted_write".to_string(), report.lifted_write.into_dyn()),
                lifted_read: arrays.put("lifted_read".to_string(), report.lifted_read.into_dyn()),
                dependent: arrays.put("dependent".to_string(), report.dependent.values.into_dyn()),
                dependent_frobenius_band: finite("dependent_frobenius_band", report.dependent.frobenius_band)?,
            };
            let wire = realization(
                report.compiled.realization,
                &mut arrays,
                |direction: Vec<f64>, arrays: &mut Arrays| {
                    Ok(WitnessWire::Input {
                        direction: arrays.vector("witness/input", &direction),
                    })
                },
                |domain: &AllInputs| format!("every input of width {}", domain.width),
            )?;
            (Some(wire), plan, findings)
        }
        CompileProblem::QueryKeyEdit(request) => {
            let claimed = request.claimed_changes.as_deref().map(|id| matrix(tensors, id)).transpose()?;
            let head_rows = |wire: &HeadRowsWire| HeadRows {
                storage: TensorId(wire.storage.clone()),
                row_offset: wire.row_offset,
            };
            let problem = QueryKeyEditProblem {
                registry: &registry,
                query: matrix(tensors, &request.query)?,
                key: matrix(tensors, &request.key)?,
                query_setting: matrix(tensors, &request.query_setting)?,
                key_setting: matrix(tensors, &request.key_setting)?,
                query_rows: head_rows(&request.query_rows),
                key_rows: head_rows(&request.key_rows),
                rotary: request.rotary.as_ref(),
                score_scale: request.score_scale,
                queries: matrix(tensors, &request.queries)?,
                query_positions: &request.query_positions,
                keys: matrix(tensors, &request.keys)?,
                key_positions: &request.key_positions,
                causal: request.causal,
                claim: claimed.map_or(ScoreClaim::FirstOrder, ScoreClaim::Declared),
            };
            let report = compile_query_key_edit(&problem, control).map_err(compile_error)?;
            let plan = plan_report(&report.compiled, &mut arrays);
            let findings = CompileFindings::QueryKeyEdit {
                exact_change: arrays.put("exact_change".to_string(), report.exact_change.values.into_dyn()),
                exact_change_bands: arrays.put("exact_change_bands".to_string(), report.exact_change.bands.into_dyn()),
                first_order: arrays.put("first_order".to_string(), report.first_order.into_dyn()),
                cross: arrays.put("cross".to_string(), report.cross.into_dyn()),
                claim_total_variation: report.rows.iter().map(|row| row.claim_total_variation).collect(),
            };
            let wire = realization(
                report.compiled.realization,
                &mut arrays,
                |row: usize, _| Ok(WitnessWire::Residual { side: SideWire::Right, column: row, entry: 0 }),
                |domain: &QueryKeyDomain| format!("{} query rows over {} key rows", domain.query_rows, domain.key_rows),
            )?;
            (Some(wire), plan, findings)
        }
        CompileProblem::Path(request) => {
            let mut sites = Vec::with_capacity(request.sites.len());
            for site in &request.sites {
                sites.push(PathSite {
                    storage: TensorId(site.storage.clone()),
                    weight: matrix(tensors, &site.weight)?,
                    bias: site.bias.as_deref().map(|id| vector(tensors, id)).transpose()?,
                    editable: site.editable,
                });
            }
            let layers = path_layers(&request.layers, tensors)?;
            let off_target = request.off_target.as_deref().map(|id| matrix(tensors, id)).transpose()?;
            let problem = PathProblem {
                registry: &registry,
                sites,
                layers,
                inputs: matrix(tensors, &request.inputs)?,
                targets: matrix(tensors, &request.targets)?,
                target_radius: request.target_radius,
                off_target,
                max_iterations: request.max_iterations,
            };
            let report = compile_path(&problem, control).map_err(compile_error)?;
            let plan = plan_report(&report.compiled, &mut arrays);
            let entry = |witness: PathWitness, _: &mut Arrays| {
                Ok(WitnessWire::Entry {
                    observation: witness.observation,
                    coordinate: witness.coordinate,
                })
            };
            let domain = |family: &PathFamily| format!("{} declared inputs", family.inputs);
            let findings = CompileFindings::Path {
                iterations: report.iterations,
                residual: arrays.put("residual".to_string(), report.residual.into_dyn()),
                residual_band: arrays.put("residual_band".to_string(), report.residual_band.into_dyn()),
                off_target_damage: report
                    .off_target_damage
                    .map(|status| status_wire(status, &mut arrays, &entry, &domain))
                    .transpose()?,
            };
            let wire = realization(report.compiled.realization, &mut arrays, entry, domain)?;
            (Some(wire), plan, findings)
        }
        CompileProblem::QueryKeySolve(request) => {
            let requirements: Vec<ScoreRequirement> = request
                .requirements
                .iter()
                .map(|&(query, key, target)| ScoreRequirement { query, key, target })
                .collect();
            let head_rows = |wire: &HeadRowsWire| HeadRows {
                storage: TensorId(wire.storage.clone()),
                row_offset: wire.row_offset,
            };
            let problem = QueryKeySolveProblem {
                registry: &registry,
                query: matrix(tensors, &request.query)?,
                key: matrix(tensors, &request.key)?,
                query_rows: head_rows(&request.query_rows),
                key_rows: head_rows(&request.key_rows),
                rotary: request.rotary.as_ref(),
                score_scale: request.score_scale,
                queries: matrix(tensors, &request.queries)?,
                query_positions: &request.query_positions,
                keys: matrix(tensors, &request.keys)?,
                key_positions: &request.key_positions,
                causal: request.causal,
                requirements: &requirements,
                target_radius: request.target_radius,
                max_rounds: request.max_rounds,
            };
            let report = solve_query_key_setting(&problem, control).map_err(compile_error)?;
            let plan = plan_report(&report.compiled, &mut arrays);
            let mut residuals = Vec::with_capacity(report.residuals.len());
            for (value, band) in &report.residuals {
                residuals.push([finite("score residual", *value)?, finite("score band", *band)?]);
            }
            let findings = CompileFindings::QueryKeySolve {
                query_setting: arrays.put("query_setting".to_string(), report.query_setting.into_dyn()),
                key_setting: arrays.put("key_setting".to_string(), report.key_setting.into_dyn()),
                rounds: report.rounds,
                residuals,
                claim_total_variation: report.certification.rows.iter().map(|row| row.claim_total_variation).collect(),
            };
            let wire = realization(
                report.compiled.realization,
                &mut arrays,
                |index: usize, _| Ok(WitnessWire::Entry { observation: index, coordinate: 0 }),
                |family: &ScoreFamily| format!("{} declared score requirements", family.requirements),
            )?;
            (Some(wire), plan, findings)
        }
        CompileProblem::NullEdit {
            edit_gram,
            response_gram,
        } => {
            let outcome = physically_null_supremum(matrix(tensors, &edit_gram)?, matrix(tensors, &response_gram)?)
                .map_err(compile_error)?;
            let witness = |direction: EditDirection, arrays: &mut Arrays| {
                Ok(WitnessWire::EditDirection {
                    direction: arrays.vector("witness/edit_direction", &direction.direction),
                    edit_size: finite("edit_size", direction.edit_size)?,
                    response_error: finite("response_error", direction.response_error)?,
                })
            };
            let domain = |ball: &EditBall| format!("the unit G-ball of range(G), rank {} of {}", ball.range_rank, ball.dimension);
            let (refused, status) = match outcome {
                NullEditOutcome::Bounded(status) => (false, status),
                NullEditOutcome::Refused(status) => (true, status),
            };
            let supremum = status_wire(status, &mut arrays, &witness, &domain)?;
            (None, None, CompileFindings::NullEdit { refused, supremum })
        }
    };
    Ok(output(
        MpdResult::Compile(Box::new(CompileReport {
            control: request.control,
            realization: realization_wire,
            plan,
            findings,
        })),
        arrays.arrays,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ndarray::{ArrayD, Ix2, array};

    use super::super::{MpdResult, run_parameter_decomposition};
    use super::{CompileFindings, RealizationWire};
    use crate::test_support::test_governor;

    fn request(problem: &str) -> String {
        format!(
            r#"{{"schema": "gam.mpd-request", "schema_version": 1, "operation": {{"kind": "compile",
            "control": "edit",
            "registry": {{"storage": {{"w": "w"}}, "aliases": {{}},
                          "use_sites": [{{"site": "w#0", "reads": "w", "map": "identity"}}]}},
            "problem": {problem}}}}}"#
        )
    }

    #[test]
    fn a_linear_site_request_reports_its_plan_and_status() {
        let tensors: BTreeMap<String, ArrayD<f64>> = BTreeMap::from([
            ("w".to_string(), array![[1.0, 0.0], [0.5, -1.0]].into_dyn()),
            ("x".to_string(), array![[1.0, 0.0], [0.0, 1.0]].into_dyn()),
            ("y".to_string(), array![[2.0, 0.5], [0.0, -1.0]].into_dyn()),
        ]);
        let problem = r#"{"kind": "linear_site", "storage": "w", "native": "w",
            "requirements": [{"kind": "linear", "site": "w#0", "inputs": "x", "targets": "y",
                              "target_radius": 0.0, "class": {"kind": "all_inputs"}}],
            "off_target": [], "row_scale": null, "col_scale": null, "ties": []}"#;
        let output = run_parameter_decomposition(&request(problem), &tensors, test_governor()).expect("runs");
        let MpdResult::Compile(report) = &output.report.result else {
            panic!("compile result");
        };
        assert!(matches!(report.realization, Some(RealizationWire::ExactlyRealized { .. })));
        let plan = report.plan.as_ref().expect("plan");
        assert_eq!(plan.len(), 1);
        let left = output.arrays[&plan[0].left].view().into_dimensionality::<Ix2>().expect("matrix");
        let right = output.arrays[&plan[0].right].view().into_dimensionality::<Ix2>().expect("matrix");
        let edit = left.dot(&right.t());
        // The set map is [[2, 0], [0.5, -1]]; the edit is its difference from W.
        let expected = array![[1.0, 0.0], [0.0, 0.0]];
        assert!(edit.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-14));
        assert!(matches!(report.findings, CompileFindings::LinearSite { .. }));
        assert!(output.report_json().expect("json").contains("\"exactly_realized\""));
    }

    #[test]
    fn a_null_edit_request_refuses_a_response_on_the_edit_kernel() {
        let tensors: BTreeMap<String, ArrayD<f64>> = BTreeMap::from([
            ("w".to_string(), array![[1.0]].into_dyn()),
            ("g".to_string(), array![[1.0, 0.0], [0.0, 0.0]].into_dyn()),
            ("k".to_string(), array![[1.0, 0.0], [0.0, 2.0]].into_dyn()),
        ]);
        let problem = r#"{"kind": "null_edit", "edit_gram": "g", "response_gram": "k"}"#;
        let output = run_parameter_decomposition(&request(problem), &tensors, test_governor()).expect("runs");
        let MpdResult::Compile(report) = &output.report.result else {
            panic!("compile result");
        };
        assert!(report.realization.is_none() && report.plan.is_none());
        let CompileFindings::NullEdit { refused, .. } = report.findings else {
            panic!("null edit findings");
        };
        assert!(refused);
    }
}
