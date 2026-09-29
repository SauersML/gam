//! Exact finite-change accounting through a mechanism program (#2951).
//!
//! Two executions of one [`Program`] differ in their inputs, their dense parameters, their
//! mask assignments, or any mix. [`account_finite_change`] propagates the change node by node
//! with the exact two-endpoint operators of `secant`, so that
//!
//! ```text
//! y(end) − y(start) = Σ_sources contribution(source)
//! ```
//!
//! holds as an identity in real arithmetic: every stage's operator is built from its two
//! stored endpoints and maps the change of its arguments to the change of its output EXACTLY,
//! not to first order. A source is where a change enters: an entry input, an initial state
//! slot, one use of a formal parameter at one invocation, or one control at one invocation.
//! Each source's contribution is its change carried through every stage it reaches, and the
//! contributions of one source arriving along several edges add.
//!
//! # The split is a convention; the total is an identity
//!
//! A stage with two changing arguments splits its change between them by a rule. The bilinear
//! midpoint rule `L′R′ − LR = L̄ ΔR + ΔL R̄` (a weight times rows, a Hadamard product, a mask
//! times a value, attention weights times values, query times key) is symmetric, but it is one
//! choice among many: `L′ ΔR + ΔL R` is exact too and splits differently. Where the change of
//! an entry is zero while two sources' contributions to it cancel, a divided difference may take
//! any slope and still sum exactly. So the per-path split among interacting paths depends on the
//! midpoint and secant conventions stated here, while the TOTAL over sources is an identity
//! whatever the convention. Only the total is checked against the direct difference.
//!
//! # Stages
//!
//! * Linear stages (a Linear node, a bias, a sum, a mask, a rotation, a composition's straight
//!   path) carry changes exactly by the midpoint rule; a changed parameter or mask value adds
//!   its own source `Δθ x̄`.
//! * ReLU: the divided difference `(relu(b) − relu(a))/(b − a)`, `1` or `0` when both
//!   endpoints share a side, and the indicator of `a > 0` when `a = b`.
//! * RMSNorm: `secant::RmsNormSecant` per row, `g ⊙ N(y) − g ⊙ N(x) = ḡ ⊙ B(x, y)Δ + Δg ⊙ N̄`.
//! * Causal self-attention: rotations are linear; scores `σ (Rq)·(Rk)` by the midpoint rule;
//!   each causal softmax row by `secant::SoftmaxSecant`, the logarithmic-mean operator that stays
//!   exact where the softmax saturates and its Jacobian vanishes; the value read by the midpoint
//!   rule.
//!
//! # Bands
//!
//! Every contribution carries a per-entry band against the exact contribution of the stored
//! endpoints, built as `replay` builds forward bands. The stored values are not exact functions
//! of their stored arguments, so beside the contributions the walk carries a slack `s` with
//! `|Σ exact contributions − (ŷ′ − ŷ)| ≤ s`: a stage maps its arguments' slacks through the
//! absolute value of its operator and adds the local bands of its own two evaluations. The
//! identity is then checked as `|Σ ĉ − fl(ŷ′ − ŷ)| ≤ Σ bands + slack + the direct difference's
//! rounding`, entry by entry, and an accounting that fails it is refused, never returned.
//!
//! Primitives with no derived evaluation band (SiLU, the exact GELU, SwiGLU, LayerNorm, the
//! per-head norm) are refused, as in `replay`. The positions are shared by both executions.

use super::attention::{AttentionGeometry, AttentionProgramError, ProjectedRows, RotaryCausalAttention, RotaryEmbedding};
use super::block::{NormBandUnbounded, rms_norm_band};
use super::gated_rewrite::{GatedRewriteError, MaskedNorm};
use super::program::{
    BodyId, CallSite, ControlId, DenseParameters, MaskAssignment, NativePrimitive, Node, NodeId, ParameterSlot,
    Program, SlotId,
};
use super::receipts::evaluation_band;
use super::replay::{
    BandedExecution, ReplayError, abs_product, bound_vector, compose_step, execute_banded, linear_matrix, raised,
    underived_primitive,
};
use super::secant::{BandedMatrix, RmsNormSecant, SecantError, SoftmaxSecant};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, Array3, ArrayView2, Zip};
use std::collections::BTreeMap;
use std::fmt;

/// Where a change enters the program.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChangeSource {
    /// An entry input port.
    Input { port: u32 },
    /// A state slot's initial value.
    InitialSlot { slot: SlotId },
    /// One use of a formal parameter: the node that reads it at one invocation.
    Parameter { parameter: ParameterSlot, invocation: Vec<CallSite>, body: BodyId, node: NodeId },
    /// One control's value at one invocation.
    Control { control: ControlId, invocation: Vec<CallSite> },
}

/// One execution of the program: its inputs, initial slots, masks and dense parameters.
#[derive(Clone, Debug)]
pub struct Endpoint<'a> {
    pub masks: &'a MaskAssignment,
    pub parameters: &'a DenseParameters,
    pub inputs: Vec<Array2<f64>>,
    pub slots: Vec<Option<Array2<f64>>>,
}

/// The identity check at its worst entry: `discrepancy = |Σ ĉ − fl(ŷ′ − ŷ)|` against the
/// allowance its bands give.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentityCheck {
    pub worst: (usize, usize),
    pub discrepancy: f64,
    pub allowance: f64,
}

/// A finite change accounted for source by source.
#[derive(Clone, Debug)]
pub struct FiniteChangeAccounting {
    /// Each source's contribution to the output change, in source order.
    pub contributions: Vec<(ChangeSource, BandedMatrix)>,
    /// `Σ` of the contributions with the sum of their bands and its own rounding.
    pub total: BandedMatrix,
    /// `fl(ŷ′ − ŷ)` of the two executed outputs, with its rounding.
    pub direct: BandedMatrix,
    /// The bound on `|Σ exact contributions − (ŷ′ − ŷ)|` from the stored values' own rounding.
    pub slack: Array2<f64>,
    pub identity: IdentityCheck,
}

/// Why an edited gain's normalized rows have no band.
#[derive(Debug)]
pub enum NormRefusal {
    Rewrite(GatedRewriteError),
    Band(NormBandUnbounded),
}

/// A refused accounting.
#[derive(Debug)]
pub enum AccountingError {
    Replay(Box<ReplayError>),
    Secant { body: BodyId, node: NodeId, error: Box<SecantError> },
    Attention { body: BodyId, node: NodeId, error: Box<AttentionProgramError> },
    /// A primitive with no derived evaluation band.
    NoDerivedBand { body: BodyId, node: NodeId, primitive: &'static str },
    /// The two executions carried different bodies of a refinement downstream.
    RefinementDiffers { body: BodyId, node: NodeId },
    /// A composition stage ran at one endpoint only (its mask is `0` at the other), so its
    /// value at that endpoint does not exist.
    StageSkipped { body: BodyId, node: NodeId, stage: usize },
    /// The two executions' outputs, inputs or slots do not pair.
    Shape { what: &'static str, start: (usize, usize), end: (usize, usize) },
    MissingValue { body: BodyId, node: NodeId },
    /// An edited RMSNorm gain's normalized rows, or their band, were refused.
    Norm { body: BodyId, node: NodeId, error: Box<NormRefusal> },
    /// The contributions do not sum to the direct difference within their bands: an
    /// accounting defect, refused rather than returned.
    IdentityViolated(IdentityCheck),
}

impl fmt::Display for AccountingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Replay(error) => write!(formatter, "finite-change accounting: {error}"),
            Self::Secant { body, node, error } => write!(formatter, "node {} of body {}: {error}", node.0, body.0),
            Self::Attention { body, node, error } => {
                write!(formatter, "attention node {} of body {}: {error}", node.0, body.0)
            }
            Self::NoDerivedBand { body, node, primitive } => write!(
                formatter,
                "node {} of body {} is {primitive}, whose evaluation error no owner states",
                node.0, body.0
            ),
            Self::RefinementDiffers { body, node } => write!(
                formatter,
                "refinement node {} of body {} carried the native body at one endpoint and the mechanism at the other",
                node.0, body.0
            ),
            Self::StageSkipped { body, node, stage } => write!(
                formatter,
                "stage {stage} of composition node {} of body {} ran at one endpoint only",
                node.0, body.0
            ),
            Self::Shape { what, start, end } => {
                write!(formatter, "the two executions' {what} have shapes {start:?} and {end:?}")
            }
            Self::MissingValue { body, node } => {
                write!(formatter, "no traced value of node {} of body {}", node.0, body.0)
            }
            Self::Norm { body, node, error } => match error.as_ref() {
                NormRefusal::Rewrite(error) => write!(formatter, "norm node {} of body {}: {error}", node.0, body.0),
                NormRefusal::Band(error) => write!(formatter, "norm node {} of body {}: {error}", node.0, body.0),
            },
            Self::IdentityViolated(check) => write!(
                formatter,
                "the contributions miss the direct difference by {} at {:?}, beyond the allowance {}",
                check.discrepancy, check.worst, check.allowance
            ),
        }
    }
}

impl std::error::Error for AccountingError {}

impl From<ReplayError> for AccountingError {
    fn from(error: ReplayError) -> Self {
        Self::Replay(Box::new(error))
    }
}

/// Checks `|total − direct| ≤ total.bands + direct.bands + slack` entry by entry. A violation
/// is decided with the discrepancy rounded down and the allowance rounded up, so it is certain.
pub fn check_identity(
    total: &BandedMatrix,
    direct: &BandedMatrix,
    slack: &Array2<f64>,
) -> Result<IdentityCheck, AccountingError> {
    let shape = direct.values.dim();
    for (what, found) in [("total", total.values.dim()), ("total bands", total.bands.dim()), ("slack", slack.dim())] {
        if found != shape {
            return Err(AccountingError::Shape { what, start: found, end: shape });
        }
    }
    let mut check = IdentityCheck { worst: (0, 0), discrepancy: 0.0, allowance: 0.0 };
    let mut worst_ratio = -1.0_f64;
    for ((row, column), &direct_value) in direct.values.indexed_iter() {
        let at = [row, column];
        let discrepancy = (total.values[at] - direct_value).abs();
        let allowance = ((total.bands[at] + direct.bands[at]).next_up() + slack[at]).next_up();
        if discrepancy.next_down() > allowance {
            return Err(AccountingError::IdentityViolated(IdentityCheck { worst: (row, column), discrepancy, allowance }));
        }
        let ratio = if discrepancy == 0.0 { 0.0 } else { discrepancy / allowance };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            check = IdentityCheck { worst: (row, column), discrepancy, allowance };
        }
    }
    Ok(check)
}

/// The change of one value: each source's contribution and the slack.
#[derive(Clone, Debug)]
struct Change {
    contributions: BTreeMap<ChangeSource, BandedMatrix>,
    slack: Array2<f64>,
}

impl Change {
    fn zero(shape: (usize, usize)) -> Self {
        Self { contributions: BTreeMap::new(), slack: Array2::zeros(shape) }
    }

    /// Adds a source's contribution, summing with one already present: one rounding per entry.
    fn add(&mut self, source: ChangeSource, contribution: BandedMatrix) {
        if contribution.values.iter().all(|&v| v == 0.0) && contribution.bands.iter().all(|&b| b == 0.0) {
            return;
        }
        match self.contributions.remove(&source) {
            None => {
                self.contributions.insert(source, contribution);
            }
            Some(present) => {
                let values = &present.values + &contribution.values;
                let mut bands = &present.bands + &contribution.bands;
                Zip::from(&mut bands).and(&present.values).and(&contribution.values).for_each(|band, &a, &b| {
                    *band = (band.next_up() + evaluation_band(1, (a.abs() + b.abs()).next_up(), 0.0)).next_up();
                });
                self.contributions.insert(source, BandedMatrix { values, bands });
            }
        }
    }

    fn add_slack(&mut self, slack: &Array2<f64>) {
        Zip::from(&mut self.slack).and(slack).for_each(|total, &own| *total = (*total + own).next_up());
    }
}

/// `fl(b − a)` with its band `γ_1 |fl(b − a)|`.
fn difference(start: &Array2<f64>, end: &Array2<f64>) -> BandedMatrix {
    let values = end - start;
    let bands = values.mapv(|v| evaluation_band(1, v.abs(), 0.0));
    BandedMatrix { values, bands }
}

/// `fl(½a + ½b)` with its band: one rounding of the sum, and a subnormal spacing for the halvings.
fn midpoint(start: &Array2<f64>, end: &Array2<f64>) -> BandedMatrix {
    let values = start * 0.5 + end * 0.5;
    let bands = if start == end {
        Array2::zeros(start.raw_dim())
    } else {
        values.mapv(|v| evaluation_band(1, v.abs(), 2.0))
    };
    BandedMatrix { values, bands }
}

/// `s ⊙ c` for a factor `s` within `sb` of exact, entrywise:
/// `|s| b + sb |c| + sb b` propagated and one rounded product.
fn scale(contribution: &BandedMatrix, factor: &BandedMatrix) -> BandedMatrix {
    let values = &factor.values * &contribution.values;
    let mut bands = Array2::zeros(values.raw_dim());
    Zip::from(&mut bands)
        .and(&factor.values)
        .and(&factor.bands)
        .and(&contribution.values)
        .and(&contribution.bands)
        .for_each(|band, &s, &sb, &c, &cb| {
            let propagated = raised(s.abs() * cb + sb * c.abs() + sb * cb, 4, 3);
            *band = (propagated + evaluation_band(1, (s * c).abs().next_up(), 1.0)).next_up();
        });
    BandedMatrix { values, bands }
}

/// `(|s| + sb) ⊙ slack`, rounded up.
fn scale_slack(slack: &Array2<f64>, factor: &BandedMatrix) -> Array2<f64> {
    let mut out = Array2::zeros(slack.raw_dim());
    Zip::from(&mut out).and(slack).and(&factor.values).and(&factor.bands).for_each(|out, &s, &f, &fb| {
        *out = raised((f.abs() + fb) * s, 2, 1);
    });
    out
}

/// `c Aᵀ` for a matrix `A` whose entries carry `roundings` roundings each (`0` for a stored
/// matrix, `1` for a midpoint): the propagated `b |A|ᵀ` and the product's `γ_{d + roundings}`.
fn linear(contribution: &BandedMatrix, matrix: ArrayView2<'_, f64>, roundings: usize) -> BandedMatrix {
    let width = matrix.ncols();
    let values = contribution.values.dot(&matrix.t());
    let upper = matrix.mapv(|a| raised(a.abs(), roundings, roundings));
    let propagated = abs_product(contribution.bands.view(), upper.t());
    let absolute = abs_product(contribution.values.view(), matrix.t());
    let mut bands = propagated;
    Zip::from(&mut bands).and(&absolute).for_each(|band, &sum| {
        *band = (*band + evaluation_band(width + roundings, sum, (width * (1 + roundings)) as f64)).next_up();
    });
    BandedMatrix { values, bands }
}

fn broadcast_row(vector: &Array1<f64>, rows: usize) -> Array2<f64> {
    let mut out = Array2::zeros((rows, vector.len()));
    for mut row in out.rows_mut() {
        row.assign(vector);
    }
    out
}

/// A `heads × tokens × tokens` array with a band.
#[derive(Clone, Debug)]
struct Banded3 {
    values: Array3<f64>,
    bands: Array3<f64>,
}

/// `|a| rb + ra |b| + ra rb`.
fn box_product(a: f64, ra: f64, b: f64, rb: f64) -> f64 {
    a.abs() * rb + ra * b.abs() + ra * rb
}

/// The two executions and the walk's state.
struct Accountant<'a> {
    governor: &'a MemoryGovernor,
    program: &'a Program,
    positions: &'a [i64],
    sides: [Side<'a>; 2],
    slots: Vec<Option<Change>>,
}

struct Side<'a> {
    masks: &'a MaskAssignment,
    parameters: &'a DenseParameters,
    banded: BandedExecution,
}

const START: usize = 0;
const END: usize = 1;

/// Accounts for the change of the entry output from `start` to `end` (module documentation).
pub fn account_finite_change(
    governor: &MemoryGovernor,
    program: &Program,
    positions: &[i64],
    start: Endpoint<'_>,
    end: Endpoint<'_>,
) -> Result<FiniteChangeAccounting, AccountingError> {
    let pair = |what: &'static str, left: &Array2<f64>, right: &Array2<f64>| {
        if left.dim() == right.dim() {
            Ok(())
        } else {
            Err(AccountingError::Shape { what, start: left.dim(), end: right.dim() })
        }
    };
    if start.inputs.len() != end.inputs.len() || start.slots.len() != end.slots.len() {
        return Err(AccountingError::Shape {
            what: "input and slot counts",
            start: (start.inputs.len(), start.slots.len()),
            end: (end.inputs.len(), end.slots.len()),
        });
    }
    let mut input_changes = Vec::with_capacity(start.inputs.len());
    for (port, (a, b)) in start.inputs.iter().zip(&end.inputs).enumerate() {
        pair("inputs", a, b)?;
        let mut change = Change::zero(a.dim());
        change.add(ChangeSource::Input { port: port as u32 }, difference(a, b));
        input_changes.push(change);
    }
    let mut slots = Vec::with_capacity(start.slots.len());
    for (slot, (a, b)) in start.slots.iter().zip(&end.slots).enumerate() {
        slots.push(match (a, b) {
            (Some(a), Some(b)) => {
                pair("initial slots", a, b)?;
                let mut change = Change::zero(a.dim());
                change.add(ChangeSource::InitialSlot { slot: SlotId(slot as u32) }, difference(a, b));
                Some(change)
            }
            (None, None) => None,
            (Some(a), None) | (None, Some(a)) => {
                return Err(AccountingError::Shape { what: "initial slots", start: a.dim(), end: (0, 0) });
            }
        });
    }
    let run = |endpoint: &Endpoint<'_>| {
        execute_banded(
            governor,
            program,
            endpoint.parameters,
            endpoint.masks,
            endpoint.inputs.clone(),
            endpoint.slots.clone(),
            positions,
        )
    };
    let (start_run, end_run) = (run(&start)?, run(&end)?);
    let mut accountant = Accountant {
        governor,
        program,
        positions,
        sides: [
            Side { masks: start.masks, parameters: start.parameters, banded: start_run },
            Side { masks: end.masks, parameters: end.parameters, banded: end_run },
        ],
        slots,
    };
    let mut path = Vec::new();
    let output = accountant.body(program.parts().entry, input_changes, &mut path)?;
    let (start_out, end_out) = (&accountant.sides[START].banded.output.values, &accountant.sides[END].banded.output.values);
    pair("outputs", start_out, end_out)?;
    let direct = difference(start_out, end_out);
    let shape = direct.values.dim();
    let mut values = Array2::<f64>::zeros(shape);
    let mut bands = Array2::<f64>::zeros(shape);
    let mut absolute = Array2::<f64>::zeros(shape);
    for contribution in output.contributions.values() {
        values += &contribution.values;
        Zip::from(&mut bands).and(&contribution.bands).for_each(|total, &own| *total = (*total + own).next_up());
        Zip::from(&mut absolute).and(&contribution.values).for_each(|total, &own| *total = (*total + own.abs()).next_up());
    }
    let terms = output.contributions.len();
    Zip::from(&mut bands).and(&absolute).for_each(|band, &sum| {
        *band = (*band + evaluation_band(terms.saturating_sub(1), sum, 0.0)).next_up();
    });
    let total = BandedMatrix { values, bands };
    let identity = check_identity(&total, &direct, &output.slack)?;
    Ok(FiniteChangeAccounting {
        contributions: output.contributions.into_iter().collect(),
        total,
        direct,
        slack: output.slack,
        identity,
    })
}

impl Accountant<'_> {
    fn value(&self, side: usize, path: &[CallSite], body: BodyId, node: NodeId) -> Result<&Array2<f64>, AccountingError> {
        self.sides[side].banded.trace.value(path, body, node).ok_or(AccountingError::MissingValue { body, node })
    }

    /// The local band of a node at both endpoints, summed.
    fn local_bands(&self, path: &[CallSite], body: BodyId, node: NodeId) -> Result<Array2<f64>, AccountingError> {
        let band = |side: usize| {
            self.sides[side]
                .banded
                .bands(path, body, node)
                .map(|bands| (*bands.local).clone())
                .ok_or(AccountingError::MissingValue { body, node })
        };
        let mut total = band(START)?;
        Zip::from(&mut total).and(&band(END)?).for_each(|total, &other| *total = (*total + other).next_up());
        Ok(total)
    }

    fn control(&self, side: usize, control: ControlId, path: &[CallSite]) -> Result<f64, AccountingError> {
        self.program
            .control_value(self.sides[side].masks, control, path)
            .map_err(|error| AccountingError::Replay(Box::new(ReplayError::Program(Box::new(error)))))
    }

    fn body(&mut self, body_id: BodyId, inputs: Vec<Change>, path: &mut Vec<CallSite>) -> Result<Change, AccountingError> {
        let program = self.program;
        let body = &program.parts().bodies[body_id.index()];
        let mut changes: Vec<Option<Change>> = vec![None; body.nodes.len()];
        for (i, node) in body.nodes.iter().enumerate() {
            let node_id = NodeId(i as u32);
            let site = CallSite { body: body_id, node: node_id, stage: 0 };
            let shape = self.value(START, path, body_id, node_id)?.dim();
            let change_of = |argument: NodeId| -> Result<&Change, AccountingError> {
                changes[argument.index()].as_ref().ok_or(AccountingError::MissingValue { body: body_id, node: argument })
            };
            let change = match node {
                Node::Input { port } => inputs[*port as usize].clone(),
                Node::Read { slot } => self.slots[slot.index()].clone().unwrap_or_else(|| Change::zero(shape)),
                Node::Write { slot, value } => {
                    let change = change_of(*value)?.clone();
                    self.slots[slot.index()] = Some(change.clone());
                    change
                }
                Node::Native { primitive, arguments } => {
                    let argument_changes =
                        arguments.iter().map(|&argument| change_of(argument).cloned()).collect::<Result<Vec<_>, _>>()?;
                    self.native(site, path, primitive, arguments, argument_changes)?
                }
                Node::Sum { terms } => {
                    let mut change = Change::zero(shape);
                    for term in terms {
                        let (m_start, m_end) = match term.control {
                            Some(control) => (self.control(START, control, path)?, self.control(END, control, path)?),
                            None => (1.0, 1.0),
                        };
                        let factor = scalar_midpoint(m_start, m_end, shape);
                        let argument = change_of(term.value)?;
                        for (source, contribution) in &argument.contributions {
                            change.add(source.clone(), scale(contribution, &factor));
                        }
                        change.add_slack(&scale_slack(&argument.slack, &factor));
                        if let (Some(control), true) = (term.control, m_start != m_end) {
                            let x_bar = midpoint(
                                self.value(START, path, body_id, term.value)?,
                                self.value(END, path, body_id, term.value)?,
                            );
                            let step = scalar_step(m_start, m_end, shape);
                            change.add(ChangeSource::Control { control, invocation: path.clone() }, scale(&x_bar, &step));
                        }
                    }
                    change.add_slack(&self.local_bands(path, body_id, node_id)?);
                    change
                }
                Node::Compose { value, stages } => {
                    let mut current = change_of(*value)?.clone();
                    let mut values =
                        [self.value(START, path, body_id, *value)?.clone(), self.value(END, path, body_id, *value)?.clone()];
                    for (k, stage) in stages.iter().enumerate() {
                        let m = match stage.control {
                            Some(control) => [self.control(START, control, path)?, self.control(END, control, path)?],
                            None => [1.0, 1.0],
                        };
                        if m == [0.0, 0.0] {
                            continue;
                        }
                        if m[START] == 0.0 || m[END] == 0.0 {
                            return Err(AccountingError::StageSkipped { body: body_id, node: node_id, stage: k });
                        }
                        let stage_output = program.parts().bodies[stage.body.index()].output;
                        path.push(CallSite { stage: k as u32, ..site });
                        let staged = self.body(stage.body, vec![current.clone()], path);
                        let staged_values = [START, END]
                            .map(|side| self.value(side, path, stage.body, stage_output).cloned());
                        path.pop();
                        let staged = staged?;
                        let [f_start, f_end] = staged_values;
                        let staged_values = [f_start?, f_end?];
                        if m == [1.0, 1.0] {
                            current = staged;
                            values = staged_values;
                            continue;
                        }
                        // z = (1 − m) x + m f: Δz = (1 − m̄) Δx + m̄ Δf + Δm (f̄ − x̄).
                        let keep = scalar_midpoint(1.0 - m[START], 1.0 - m[END], shape);
                        let take = scalar_midpoint(m[START], m[END], shape);
                        let mut next = Change::zero(shape);
                        for (source, contribution) in &current.contributions {
                            next.add(source.clone(), scale(contribution, &keep));
                        }
                        for (source, contribution) in &staged.contributions {
                            next.add(source.clone(), scale(contribution, &take));
                        }
                        next.add_slack(&scale_slack(&current.slack, &keep));
                        next.add_slack(&scale_slack(&staged.slack, &take));
                        if let (Some(control), true) = (stage.control, m[START] != m[END]) {
                            let f_bar = midpoint(&staged_values[START], &staged_values[END]);
                            let x_bar = midpoint(&values[START], &values[END]);
                            let gap = difference(&x_bar.values, &f_bar.values);
                            let gap = BandedMatrix {
                                values: gap.values,
                                bands: (&gap.bands + &f_bar.bands + &x_bar.bands).mapv(|b| raised(b, 2, 0)),
                            };
                            next.add(
                                ChangeSource::Control { control, invocation: path.clone() },
                                scale(&gap, &scalar_step(m[START], m[END], shape)),
                            );
                        }
                        for side in [START, END] {
                            if m[side] != 1.0 {
                                let (combined, band) = compose_step(&values[side], &staged_values[side], m[side]);
                                next.add_slack(&band);
                                values[side] = combined;
                            } else {
                                values[side] = staged_values[side].clone();
                            }
                        }
                        current = next;
                    }
                    current
                }
                Node::Call { body: callee, arguments } => {
                    let passed =
                        arguments.iter().map(|&argument| change_of(argument).cloned()).collect::<Result<Vec<_>, _>>()?;
                    path.push(site);
                    let called = self.body(*callee, passed, path);
                    path.pop();
                    called?
                }
                Node::Refine { native, mechanism, arguments } => {
                    let passed =
                        arguments.iter().map(|&argument| change_of(argument).cloned()).collect::<Result<Vec<_>, _>>()?;
                    path.push(site);
                    let ran = [START, END].map(|side| self.sides[side].banded.trace.ran(path, *native));
                    if ran[START] != ran[END] {
                        path.pop();
                        return Err(AccountingError::RefinementDiffers { body: body_id, node: node_id });
                    }
                    let chosen = if ran[START] { *native } else { *mechanism };
                    let refined = self.body(chosen, passed, path);
                    path.pop();
                    refined?
                }
            };
            changes[i] = Some(change);
        }
        changes[body.output.index()].take().ok_or(AccountingError::MissingValue { body: body_id, node: body.output })
    }

    fn native(
        &self,
        site: CallSite,
        path: &[CallSite],
        primitive: &NativePrimitive,
        arguments: &[NodeId],
        changes: Vec<Change>,
    ) -> Result<Change, AccountingError> {
        let (body, node) = (site.body, site.node);
        if let Some(name) = underived_primitive(primitive) {
            return Err(AccountingError::NoDerivedBand { body, node, primitive: name });
        }
        let x = [self.value(START, path, body, arguments[0])?, self.value(END, path, body, arguments[0])?];
        let shape = self.value(START, path, body, node)?.dim();
        let mut change = Change::zero(shape);
        let parameter_source =
            |parameter: ParameterSlot| ChangeSource::Parameter { parameter, invocation: path.to_vec(), body, node };
        match primitive {
            NativePrimitive::Linear { weight, orientation } => {
                let a = [
                    linear_matrix(self.sides[START].parameters, *weight, *orientation)?,
                    linear_matrix(self.sides[END].parameters, *weight, *orientation)?,
                ];
                let edited = a[START] != a[END];
                let mean = if edited { a[START].to_owned() * 0.5 + a[END].to_owned() * 0.5 } else { a[START].to_owned() };
                let roundings = usize::from(edited);
                for (source, contribution) in &changes[0].contributions {
                    change.add(source.clone(), linear(contribution, mean.view(), roundings));
                }
                let upper = mean.mapv(|v| raised(v.abs(), roundings, roundings));
                change.add_slack(&abs_product(changes[0].slack.view(), upper.t()));
                if edited {
                    let step = &a[END] - &a[START];
                    let x_bar = midpoint(x[START], x[END]);
                    let width = step.ncols();
                    let values = x_bar.values.dot(&step.t());
                    let absolute = abs_product(x_bar.values.view(), step.t());
                    let bands = absolute.mapv(|sum| evaluation_band(width + 2, sum, (3 * width) as f64));
                    change.add(parameter_source(*weight), BandedMatrix { values, bands });
                }
            }
            NativePrimitive::AddBias { bias } => {
                for (source, contribution) in &changes[0].contributions {
                    change.add(source.clone(), contribution.clone());
                }
                change.add_slack(&changes[0].slack);
                let b = [bound_vector(self.sides[START].parameters, *bias)?, bound_vector(self.sides[END].parameters, *bias)?];
                if b[START] != b[END] {
                    let rows = shape.0;
                    change.add(
                        parameter_source(*bias),
                        difference(&broadcast_row(b[START], rows), &broadcast_row(b[END], rows)),
                    );
                }
            }
            NativePrimitive::Activation { .. } => {
                // ReLU, the only activation with a derived band (`underived_primitive`).
                let mut slope = BandedMatrix { values: Array2::zeros(shape), bands: Array2::zeros(shape) };
                for ((row, column), value) in slope.values.indexed_iter_mut() {
                    let (a, b) = (x[START][[row, column]], x[END][[row, column]]);
                    if a == b {
                        *value = if a > 0.0 { 1.0 } else { 0.0 };
                    } else if a > 0.0 && b > 0.0 {
                        *value = 1.0;
                    } else if a <= 0.0 && b <= 0.0 {
                        *value = 0.0;
                    } else {
                        *value = (b.max(0.0) - a.max(0.0)) / (b - a);
                        slope.bands[[row, column]] = evaluation_band(2, value.abs(), 1.0);
                    }
                }
                for (source, contribution) in &changes[0].contributions {
                    change.add(source.clone(), scale(contribution, &slope));
                }
                change.add_slack(&scale_slack(&changes[0].slack, &slope));
            }
            NativePrimitive::Hadamard => {
                let y = [self.value(START, path, body, arguments[1])?, self.value(END, path, body, arguments[1])?];
                let (x_bar, y_bar) = (midpoint(x[START], x[END]), midpoint(y[START], y[END]));
                for (source, contribution) in &changes[0].contributions {
                    change.add(source.clone(), scale(contribution, &y_bar));
                }
                for (source, contribution) in &changes[1].contributions {
                    change.add(source.clone(), scale(contribution, &x_bar));
                }
                change.add_slack(&scale_slack(&changes[0].slack, &y_bar));
                change.add_slack(&scale_slack(&changes[1].slack, &x_bar));
            }
            NativePrimitive::CoordinateMask { controls } => {
                let mut factor = BandedMatrix { values: Array2::zeros(shape), bands: Array2::zeros(shape) };
                let x_bar = midpoint(x[START], x[END]);
                for (c, &control) in controls.iter().enumerate() {
                    let (m_start, m_end) = (self.control(START, control, path)?, self.control(END, control, path)?);
                    let column = scalar_midpoint(m_start, m_end, (shape.0, 1));
                    factor.values.column_mut(c).assign(&column.values.column(0));
                    factor.bands.column_mut(c).assign(&column.bands.column(0));
                    if m_start != m_end {
                        let mut step = BandedMatrix { values: Array2::zeros(shape), bands: Array2::zeros(shape) };
                        let scalar = scalar_step(m_start, m_end, (shape.0, 1));
                        step.values.column_mut(c).assign(&scalar.values.column(0));
                        step.bands.column_mut(c).assign(&scalar.bands.column(0));
                        change.add(ChangeSource::Control { control, invocation: path.to_vec() }, scale(&x_bar, &step));
                    }
                }
                for (source, contribution) in &changes[0].contributions {
                    change.add(source.clone(), scale(contribution, &factor));
                }
                change.add_slack(&scale_slack(&changes[0].slack, &factor));
            }
            NativePrimitive::RmsNorm { epsilon, gain } => {
                self.rms_norm(site, path, (*epsilon, *gain), x, &changes[0], &mut change)?;
            }
            NativePrimitive::CausalSelfAttention { geometry, rotary, score_scale } => {
                self.attention(site, path, (*geometry, rotary, *score_scale), arguments, &changes, &mut change)?;
            }
            NativePrimitive::SwiGlu | NativePrimitive::LayerNorm { .. } | NativePrimitive::HeadRmsNorm { .. } => {
                return Err(AccountingError::NoDerivedBand { body, node, primitive: "an unsupported primitive" });
            }
        }
        change.add_slack(&self.local_bands(path, body, node)?);
        Ok(change)
    }

    /// `g′ ⊙ N(y) − g ⊙ N(x) = ḡ ⊙ B(x, y) Δ + Δg ⊙ N̄`, row by row.
    fn rms_norm(
        &self,
        site: CallSite,
        path: &[CallSite],
        (epsilon, gain): (f64, ParameterSlot),
        x: [&Array2<f64>; 2],
        argument: &Change,
        change: &mut Change,
    ) -> Result<(), AccountingError> {
        let (body, node) = (site.body, site.node);
        let secant_error = |error| AccountingError::Secant { body, node, error: Box::new(error) };
        let g = [bound_vector(self.sides[START].parameters, gain)?, bound_vector(self.sides[END].parameters, gain)?];
        let rows = x[START].nrows();
        let g_bar = midpoint(&broadcast_row(g[START], rows), &broadcast_row(g[END], rows));
        let operators = (0..rows)
            .map(|row| {
                RmsNormSecant::between(
                    &x[START].row(row).to_vec(),
                    &x[END].row(row).to_vec(),
                    epsilon,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(secant_error)?;
        let apply = |values: &Array2<f64>, bands: &Array2<f64>| -> Result<BandedMatrix, AccountingError> {
            let mut out = BandedMatrix { values: Array2::zeros(values.raw_dim()), bands: Array2::zeros(values.raw_dim()) };
            for (row, operator) in operators.iter().enumerate() {
                let applied = operator
                    .apply_within(&values.row(row).to_vec(), &bands.row(row).to_vec())
                    .map_err(secant_error)?;
                out.values.row_mut(row).assign(&Array1::from(applied.values));
                out.bands.row_mut(row).assign(&Array1::from(applied.bands));
            }
            Ok(out)
        };
        for (source, contribution) in &argument.contributions {
            let normalized = apply(&contribution.values, &contribution.bands)?;
            change.add(source.clone(), scale(&normalized, &g_bar));
        }
        let moved = apply(&Array2::zeros(argument.slack.raw_dim()), &argument.slack)?;
        change.add_slack(&scale_slack(&moved.bands, &g_bar));
        if g[START] != g[END] {
            let ones = Array1::from_elem(x[START].ncols(), 1.0);
            let unit = MaskedNorm::Rms { epsilon, gain: ones.view() };
            let normalize = |side: usize| -> Result<BandedMatrix, AccountingError> {
                let rows = unit
                    .apply(x[side].view())
                    .map_err(|error| AccountingError::Norm { body, node, error: Box::new(NormRefusal::Rewrite(error)) })?;
                let bands = rms_norm_band(epsilon, ones.view(), x[side].view(), rows.view())
                    .map_err(|error| AccountingError::Norm { body, node, error: Box::new(NormRefusal::Band(error)) })?;
                Ok(BandedMatrix { values: rows, bands })
            };
            let (n_start, n_end) = (normalize(START)?, normalize(END)?);
            let mut n_bar = midpoint(&n_start.values, &n_end.values);
            Zip::from(&mut n_bar.bands).and(&n_start.bands).and(&n_end.bands).for_each(|band, &a, &b| {
                *band = (*band + (0.5 * (a + b)).next_up()).next_up();
            });
            let step = difference(&broadcast_row(g[START], rows), &broadcast_row(g[END], rows));
            change.add(ChangeSource::Parameter { parameter: gain, invocation: path.to_vec(), body, node }, scale(&n_bar, &step));
        }
        Ok(())
    }

    /// The attention stages of the module documentation, source by source.
    fn attention(
        &self,
        site: CallSite,
        path: &[CallSite],
        (geometry, rotary, score_scale): (AttentionGeometry, &RotaryEmbedding, f64),
        arguments: &[NodeId],
        changes: &[Change],
        change: &mut Change,
    ) -> Result<(), AccountingError> {
        let (body, node) = (site.body, site.node);
        let attention_error = |error| AccountingError::Attention { body, node, error: Box::new(error) };
        let secant_error = |error| AccountingError::Secant { body, node, error: Box::new(error) };
        let attention = RotaryCausalAttention::new(geometry, rotary.clone(), score_scale).map_err(attention_error)?;
        let (heads, kv_heads, hd) = (geometry.n_heads, geometry.n_kv_heads, geometry.head_dim);
        let positions = self.positions;
        let tokens = positions.len();
        let rows = |side: usize, k: usize| self.value(side, path, body, arguments[k]);
        struct Endpoint3 {
            scores: Array3<f64>,
            score_band: Array3<f64>,
            weights: Array3<f64>,
            weight_band: Array3<f64>,
            mix_band: Array2<f64>,
            queries: Array2<f64>,
            query_band: Array2<f64>,
            keys: Array2<f64>,
            key_band: Array2<f64>,
        }
        let mut ends = Vec::with_capacity(2);
        for side in [START, END] {
            let (q, k, v) = (rows(side, 0)?, rows(side, 1)?, rows(side, 2)?);
            let full = attention
                .attend_projected(
                    self.governor,
                    ProjectedRows::exact(q.view()),
                    ProjectedRows::exact(k.view()),
                    ProjectedRows::exact(v.view()),
                    positions,
                )
                .map_err(attention_error)?;
            let zero_scores = Array3::zeros(full.scores.raw_dim());
            let at_scores = attention
                .weights_at_scores(self.governor, full.scores.view(), zero_scores.view())
                .map_err(attention_error)?;
            let (_, mix_band) = attention
                .mix_at_weights(full.weights.view(), zero_scores.view(), ProjectedRows::exact(v.view()))
                .map_err(attention_error)?;
            let (queries, query_band) =
                attention.rotate_heads(ProjectedRows::exact(q.view()), positions, heads).map_err(attention_error)?;
            let (keys, key_band) =
                attention.rotate_heads(ProjectedRows::exact(k.view()), positions, kv_heads).map_err(attention_error)?;
            ends.push(Endpoint3 {
                scores: full.scores.clone(),
                score_band: full.score_radius.clone(),
                weights: full.weights.clone(),
                weight_band: at_scores.weight_radius.clone(),
                mix_band,
                queries,
                query_band,
                keys,
                key_band,
            });
        }
        let mean_with = |a: &Array2<f64>, ab: &Array2<f64>, b: &Array2<f64>, bb: &Array2<f64>| {
            let mut mean = midpoint(a, b);
            Zip::from(&mut mean.bands).and(ab).and(bb).for_each(|band, &x, &y| *band = (*band + (0.5 * (x + y)).next_up()).next_up());
            mean
        };
        let q_bar = mean_with(&ends[START].queries, &ends[START].query_band, &ends[END].queries, &ends[END].query_band);
        let k_bar = mean_with(&ends[START].keys, &ends[START].key_band, &ends[END].keys, &ends[END].key_band);
        let v_bar = midpoint(rows(START, 2)?, rows(END, 2)?);
        let w_bar = {
            let values = &ends[START].weights * 0.5 + &ends[END].weights * 0.5;
            let mut bands = values.mapv(|v| evaluation_band(1, v.abs(), 2.0));
            Zip::from(&mut bands)
                .and(&ends[START].weight_band)
                .and(&ends[END].weight_band)
                .for_each(|band, &a, &b| *band = (*band + (0.5 * (a + b)).next_up()).next_up());
            Banded3 { values, bands }
        };
        let sigma = score_scale.abs();
        let growth = accumulation_growth(hd + 1).next_up();
        // σ Σ_c left[t, lo + c] right[s, ro + c] for s ≤ t, with its band.
        let bilinear = |left: &BandedMatrix, right: &BandedMatrix, out: &mut Banded3| {
            for head in 0..heads {
                let (lo, ro) = (head * hd, geometry.key_value_head(head) * hd);
                for t in 0..tokens {
                    for s in 0..=t {
                        let (mut value, mut absolute, mut propagated) = (0.0, 0.0, 0.0);
                        for c in 0..hd {
                            let (l, r) = ((t, lo + c), (s, ro + c));
                            let (a, b) = (left.values[l], right.values[r]);
                            value += a * b;
                            absolute += (a * b).abs();
                            propagated += box_product(a, left.bands[l], b, right.bands[r]);
                        }
                        // Added to what the entry already holds: one more rounding.
                        let at = [head, t, s];
                        let term = score_scale * value;
                        let joined = evaluation_band(1, (out.values[at].abs() + term.abs()).next_up(), 0.0);
                        out.values[at] += term;
                        let own = raised(sigma * (propagated + growth * absolute), 4, hd + 1);
                        out.bands[at] = ((out.bands[at] + own).next_up() + joined).next_up();
                    }
                }
            }
        };
        let rotate = |contribution: &BandedMatrix, count: usize| -> Result<BandedMatrix, AccountingError> {
            let (values, bands) = attention
                .rotate_heads(
                    ProjectedRows { values: contribution.values.view(), radius: contribution.bands.view() },
                    positions,
                    count,
                )
                .map_err(attention_error)?;
            Ok(BandedMatrix { values, bands })
        };
        let empty3 = || Banded3 { values: Array3::zeros((heads, tokens, tokens)), bands: Array3::zeros((heads, tokens, tokens)) };
        // Scores: σ [Δ(Rq) · (Rk)‾ + (Rq)‾ · Δ(Rk)].
        let mut scores: BTreeMap<ChangeSource, Banded3> = BTreeMap::new();
        for (source, contribution) in &changes[0].contributions {
            let rotated = rotate(contribution, heads)?;
            bilinear(&rotated, &k_bar, scores.entry(source.clone()).or_insert_with(empty3));
        }
        for (source, contribution) in &changes[1].contributions {
            let rotated = rotate(contribution, kv_heads)?;
            // The query side is the left factor; the key contribution is the right.
            bilinear(&q_bar, &rotated, scores.entry(source.clone()).or_insert_with(empty3));
        }
        // The score slack: |σ| (|K̄| |R| s_q + |Q̄| |R| s_k) and both evaluations' own bands.
        let mut score_slack = empty3();
        {
            let zero_q = BandedMatrix { values: Array2::zeros(changes[0].slack.raw_dim()), bands: changes[0].slack.clone() };
            let zero_k = BandedMatrix { values: Array2::zeros(changes[1].slack.raw_dim()), bands: changes[1].slack.clone() };
            let (moved_q, moved_k) = (rotate(&zero_q, heads)?, rotate(&zero_k, kv_heads)?);
            bilinear(&moved_q, &k_bar, &mut score_slack);
            let mut other = empty3();
            bilinear(&q_bar, &moved_k, &mut other);
            let mut slack = score_slack.bands;
            Zip::from(&mut slack)
                .and(&other.bands)
                .and(&ends[START].score_band)
                .and(&ends[END].score_band)
                .for_each(|total, &b, &e0, &e1| *total = (((*total + b).next_up() + e0).next_up() + e1).next_up());
            score_slack.bands = slack;
        }
        // Weights: each causal row by the softmax secant of the stored scores.
        let mut weights: BTreeMap<ChangeSource, Banded3> = BTreeMap::new();
        let mut weight_slack = Array3::<f64>::zeros((heads, tokens, tokens));
        for head in 0..heads {
            for t in 0..tokens {
                let row = |array: &Array3<f64>| (0..=t).map(|s| array[[head, t, s]]).collect::<Vec<f64>>();
                let secant = SoftmaxSecant::between(&row(&ends[START].scores), &row(&ends[END].scores))
                    .map_err(secant_error)?;
                for (source, contribution) in &scores {
                    let applied = secant
                        .apply_within(&row(&contribution.values), &row(&contribution.bands))
                        .map_err(secant_error)?;
                    let entry = weights.entry(source.clone()).or_insert_with(empty3);
                    for s in 0..=t {
                        entry.values[[head, t, s]] = applied.values[s];
                        entry.bands[[head, t, s]] = applied.bands[s];
                    }
                }
                let moved = secant
                    .apply_within(&vec![0.0; t + 1], &row(&score_slack.bands))
                    .map_err(secant_error)?;
                for s in 0..=t {
                    weight_slack[[head, t, s]] = ((moved.bands[s] + ends[START].weight_band[[head, t, s]]).next_up()
                        + ends[END].weight_band[[head, t, s]])
                        .next_up();
                }
            }
        }
        // The value read: Σ_s w̄ Δv + Δw v̄.
        let mix = |weights: &Banded3, values: &BandedMatrix| -> Result<BandedMatrix, AccountingError> {
            let (values, bands) = attention
                .mix_at_weights(
                    weights.values.view(),
                    weights.bands.view(),
                    ProjectedRows { values: values.values.view(), radius: values.bands.view() },
                )
                .map_err(attention_error)?;
            Ok(BandedMatrix { values, bands })
        };
        for (source, contribution) in &changes[2].contributions {
            change.add(source.clone(), mix(&w_bar, contribution)?);
        }
        for (source, contribution) in &weights {
            change.add(source.clone(), mix(contribution, &v_bar)?);
        }
        let zero_v = BandedMatrix { values: Array2::zeros(changes[2].slack.raw_dim()), bands: changes[2].slack.clone() };
        change.add_slack(&mix(&w_bar, &zero_v)?.bands);
        let zero_w = Banded3 { values: Array3::zeros(weight_slack.raw_dim()), bands: weight_slack };
        change.add_slack(&mix(&zero_w, &v_bar)?.bands);
        change.add_slack(&ends[START].mix_band);
        change.add_slack(&ends[END].mix_band);
        // The caller also adds the node's local band, the owner's radius at exact rows, which
        // the stage bands above already split into scores, weights and the read: it only
        // widens the slack.
        Ok(())
    }
}

/// `½(a + b)` of two scalars as a matrix, exact when they are equal.
fn scalar_midpoint(a: f64, b: f64, shape: (usize, usize)) -> BandedMatrix {
    let value = 0.5 * a + 0.5 * b;
    let band = if a == b { 0.0 } else { evaluation_band(1, value.abs(), 2.0) };
    BandedMatrix { values: Array2::from_elem(shape, value), bands: Array2::from_elem(shape, band) }
}

/// `fl(b − a)` of two scalars as a matrix, with its rounding.
fn scalar_step(a: f64, b: f64, shape: (usize, usize)) -> BandedMatrix {
    let value = b - a;
    BandedMatrix {
        values: Array2::from_elem(shape, value),
        bands: Array2::from_elem(shape, (UNIT_ROUNDOFF * value.abs()).next_up()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::RotaryPairing;
    use crate::lift::TieOrientation;
    use crate::program::{
        Body, ComposeStage, ControlDecl, DenseTensor, MaskGroup, MaskGroupId, NativeActivation, ParameterDecl,
        ProgramParts, SumTerm,
    };
    use crate::replay::ReplayError;
    use crate::test_support::test_governor;
    use ndarray::array;

    fn native(primitive: NativePrimitive, arguments: &[u32]) -> Node {
        Node::Native { primitive, arguments: arguments.iter().map(|&index| NodeId(index)).collect() }
    }

    fn linear(parameter: u32, argument: u32) -> Node {
        native(NativePrimitive::Linear { weight: ParameterSlot(parameter), orientation: TieOrientation::Identity }, &[argument])
    }

    fn sum(values: &[u32]) -> Node {
        Node::Sum { terms: values.iter().map(|&value| SumTerm { value: NodeId(value), control: None }).collect() }
    }

    fn body(name: &str, nodes: Vec<Node>) -> Body {
        let output = NodeId(nodes.len() as u32 - 1);
        Body { name: name.to_string(), inputs: 1, nodes, output }
    }

    fn parameters(count: usize) -> Vec<ParameterDecl> {
        (0..count).map(|index| ParameterDecl { name: format!("p{index}"), controls: Vec::new() }).collect()
    }

    fn singleton_controls(count: usize) -> (Vec<ControlDecl>, Vec<MaskGroup>) {
        (
            (0..count).map(|index| ControlDecl { name: format!("c{index}") }).collect(),
            (0..count).map(|index| MaskGroup { name: format!("g{index}"), controls: vec![ControlId(index as u32)] }).collect(),
        )
    }

    /// A deterministic fill with entries in `[-0.6, 0.65]`.
    fn filled(rows: usize, cols: usize, seed: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| ((i * 7 + j * 3 + seed * 5) % 11) as f64 / 8.0 - 0.6)
    }

    fn global(values: &[(u32, f64)]) -> MaskAssignment {
        let mut masks = MaskAssignment::all_on();
        for &(group, value) in values {
            masks.set(MaskGroupId(group), Vec::new(), value).expect("a finite mask");
        }
        masks
    }

    const TOKENS: usize = 3;
    const WIDTH: usize = 4;
    const HIDDEN: usize = 6;

    /// RMSNorm, one rotary attention head, a residual, a masked ReLU MLP and a residual.
    fn attention_mlp() -> Program {
        let (controls, mask_groups) = singleton_controls(HIDDEN);
        let geometry = AttentionGeometry { model_dim: WIDTH, n_heads: 1, n_kv_heads: 1, head_dim: 2 };
        let rotary = RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: vec![0.5], attention_scaling: 1.0 };
        Program::new(ProgramParts {
            parameters: parameters(8),
            slots: Vec::new(),
            controls,
            mask_groups,
            bodies: vec![body(
                "block",
                vec![
                    Node::Input { port: 0 },
                    native(NativePrimitive::RmsNorm { epsilon: 1e-6, gain: ParameterSlot(0) }, &[0]),
                    linear(1, 1),
                    linear(2, 1),
                    linear(3, 1),
                    native(NativePrimitive::CausalSelfAttention { geometry, rotary, score_scale: 0.7 }, &[2, 3, 4]),
                    linear(4, 5),
                    sum(&[0, 6]),
                    linear(5, 7),
                    native(NativePrimitive::AddBias { bias: ParameterSlot(6) }, &[8]),
                    native(NativePrimitive::Activation { activation: NativeActivation::Relu }, &[9]),
                    native(NativePrimitive::CoordinateMask { controls: (0..HIDDEN as u32).map(ControlId).collect() }, &[10]),
                    linear(7, 11),
                    sum(&[7, 12]),
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid block")
    }

    fn block_parameters(edit: f64) -> DenseParameters {
        let gain = Array1::from_shape_fn(WIDTH, |i| 1.0 + 0.125 * i as f64 + edit * 0.5);
        let bias = Array1::from_shape_fn(HIDDEN, |i| 0.1 * i as f64 - 0.2 - edit);
        let w1 = filled(HIDDEN, WIDTH, 6) + &(filled(HIDDEN, WIDTH, 9) * edit);
        DenseParameters::new(vec![
            DenseTensor::Vector(gain),
            DenseTensor::Matrix(filled(2, WIDTH, 1) * 2.0),
            DenseTensor::Matrix(filled(2, WIDTH, 2) * 2.0),
            DenseTensor::Matrix(filled(2, WIDTH, 3)),
            DenseTensor::Matrix(filled(WIDTH, 2, 4)),
            DenseTensor::Matrix(w1),
            DenseTensor::Vector(bias),
            DenseTensor::Matrix(filled(WIDTH, HIDDEN, 7)),
        ])
    }

    fn largest(matrix: &Array2<f64>) -> f64 {
        matrix.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()))
    }

    #[test]
    fn contributions_of_inputs_parameters_and_masks_sum_to_the_direct_change() {
        let program = attention_mlp();
        let (start_parameters, end_parameters) = (block_parameters(0.0), block_parameters(0.25));
        let (start_masks, end_masks) = (MaskAssignment::all_on(), global(&[(2, 0.25), (4, 0.0)]));
        let x = filled(TOKENS, WIDTH, 0) * 2.0;
        let x_end = &x + &(filled(TOKENS, WIDTH, 10) * 0.5);
        let positions: Vec<i64> = (0..TOKENS as i64).collect();
        let accounting = account_finite_change(
            test_governor(),
            &program,
            &positions,
            Endpoint { masks: &start_masks, parameters: &start_parameters, inputs: vec![x], slots: Vec::new() },
            Endpoint { masks: &end_masks, parameters: &end_parameters, inputs: vec![x_end], slots: Vec::new() },
        )
        .expect("an accounting whose contributions meet the direct difference");
        let sources: Vec<&ChangeSource> = accounting.contributions.iter().map(|(source, _)| source).collect();
        let has = |wanted: &dyn Fn(&ChangeSource) -> bool| sources.iter().any(|source| wanted(source));
        assert!(has(&|s| matches!(s, ChangeSource::Input { port: 0 })));
        for slot in [0, 5, 6] {
            assert!(
                has(&|s| matches!(s, ChangeSource::Parameter { parameter, .. } if parameter.0 == slot)),
                "parameter {slot}"
            );
        }
        for control in [2, 4] {
            assert!(has(&|s| matches!(s, ChangeSource::Control { control: c, .. } if c.0 == control)), "control {control}");
        }
        // No unchanged parameter or control is a source.
        assert!(!has(&|s| matches!(s, ChangeSource::Parameter { parameter, .. } if parameter.0 == 1)));
        assert!(!has(&|s| matches!(s, ChangeSource::Control { control, .. } if control.0 == 0)));
        // The change is real and the bands are tight, so the identity is not vacuous.
        assert!(largest(&accounting.direct.values) > 1e-2);
        assert!(largest(&accounting.total.bands) < 1e-10, "{}", largest(&accounting.total.bands));
        assert!(largest(&accounting.slack) < 1e-10, "{}", largest(&accounting.slack));
        // Positive control: dropping one source's contribution breaks the identity.
        let (dropped, contribution) = accounting
            .contributions
            .iter()
            .find(|(source, _)| matches!(source, ChangeSource::Control { control, .. } if control.0 == 4))
            .expect("the deleted unit's control");
        assert!(largest(&contribution.values) > 1e-3, "{dropped:?} moves the output");
        let short = BandedMatrix {
            values: &accounting.total.values - &contribution.values,
            bands: accounting.total.bands.clone(),
        };
        assert!(matches!(
            check_identity(&short, &accounting.direct, &accounting.slack),
            Err(AccountingError::IdentityViolated(..))
        ));
    }

    /// `q_t = (x_t0, 0)`, `k_t = v_t = (x_t1, 0)`: token 1 scores the keys `(x_10, −x_10)`.
    fn saturating_attention() -> Program {
        let geometry = AttentionGeometry { model_dim: 2, n_heads: 1, n_kv_heads: 1, head_dim: 2 };
        let rotary = RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: Vec::new(), attention_scaling: 1.0 };
        Program::new(ProgramParts {
            parameters: parameters(3),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "attention",
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    linear(1, 0),
                    linear(2, 0),
                    native(NativePrimitive::CausalSelfAttention { geometry, rotary, score_scale: 1.0 }, &[1, 2, 3]),
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid attention program")
    }

    #[test]
    fn the_secant_recovers_a_saturated_softmax_change_the_local_gradient_misses() {
        let program = saturating_attention();
        let parameters = DenseParameters::new(vec![
            DenseTensor::Matrix(array![[1.0, 0.0], [0.0, 0.0]]),
            DenseTensor::Matrix(array![[0.0, 1.0], [0.0, 0.0]]),
            DenseTensor::Matrix(array![[0.0, 1.0], [0.0, 0.0]]),
        ]);
        let masks = MaskAssignment::all_on();
        // Token 1's scores go from (12, −12) to (−12, 12); its values are (1, −1).
        let (x, x_end) = (array![[0.0, 1.0], [12.0, -1.0]], array![[0.0, 1.0], [-12.0, -1.0]]);
        let accounting = account_finite_change(
            test_governor(),
            &program,
            &[0, 1],
            Endpoint { masks: &masks, parameters: &parameters, inputs: vec![x], slots: Vec::new() },
            Endpoint { masks: &masks, parameters: &parameters, inputs: vec![x_end], slots: Vec::new() },
        )
        .expect("an accounting");
        let secant = accounting.total.values[[1, 0]];
        let direct = accounting.direct.values[[1, 0]];
        assert!((direct + 2.0).abs() < 1e-9, "the read swings from about 1 to about −1: {direct}");
        assert!((secant + 2.0).abs() < 1e-9, "{secant}");
        // The local-gradient attribution at the start: the softmax Jacobian diag p − p pᵀ at
        // p = softmax(12, −12) applied to the score change (−24, 24), read against (1, −1).
        let p1 = 1.0 / (1.0 + 24.0_f64.exp());
        let p0 = 1.0 - p1;
        let jacobian = [[p0 - p0 * p0, -p0 * p1], [-p1 * p0, p1 - p1 * p1]];
        let score_change = [-24.0, 24.0];
        let weight_change: Vec<f64> =
            (0..2).map(|i| jacobian[i][0] * score_change[0] + jacobian[i][1] * score_change[1]).collect();
        let local = weight_change[0] - weight_change[1];
        assert!(local.abs() < 1e-7, "the local gradient sees almost nothing: {local}");
        assert!(secant.abs() > 1e7 * local.abs());
        // Token 0 attends only to itself and does not move.
        assert_eq!(accounting.direct.values[[0, 0]], 0.0);
        assert_eq!(accounting.contributions.len(), 1);
    }

    /// A composition of a linear stage and a ReLU stage, each deletable, around a shared call.
    fn composed() -> Program {
        let (controls, mask_groups) = singleton_controls(2);
        Program::new(ProgramParts {
            parameters: parameters(2),
            slots: Vec::new(),
            controls,
            mask_groups,
            bodies: vec![
                body(
                    "entry",
                    vec![
                        Node::Input { port: 0 },
                        Node::Compose {
                            value: NodeId(0),
                            stages: vec![
                                ComposeStage { body: BodyId(1), control: Some(ControlId(0)) },
                                ComposeStage { body: BodyId(2), control: Some(ControlId(1)) },
                            ],
                        },
                        Node::Call { body: BodyId(1), arguments: vec![NodeId(1)] },
                    ],
                ),
                body("linear", vec![Node::Input { port: 0 }, linear(0, 0)]),
                body(
                    "relu",
                    vec![
                        Node::Input { port: 0 },
                        linear(1, 0),
                        native(NativePrimitive::Activation { activation: NativeActivation::Relu }, &[1]),
                    ],
                ),
            ],
            entry: BodyId(0),
        })
        .expect("a valid composition")
    }

    #[test]
    fn composition_stage_masks_account_exactly_and_a_skipped_stage_is_refused() {
        let program = composed();
        let parameters = DenseParameters::new(vec![
            DenseTensor::Matrix(filled(3, 3, 1)),
            DenseTensor::Matrix(filled(3, 3, 2) * 2.0),
        ]);
        let x = filled(2, 3, 3) * 2.0;
        let (start, end) = (global(&[(0, 1.0), (1, 0.5)]), global(&[(0, 0.25), (1, 0.75)]));
        let accounting = account_finite_change(
            test_governor(),
            &program,
            &[0, 1],
            Endpoint { masks: &start, parameters: &parameters, inputs: vec![x.clone()], slots: Vec::new() },
            Endpoint { masks: &end, parameters: &parameters, inputs: vec![x.clone()], slots: Vec::new() },
        )
        .expect("an accounting");
        let controls: Vec<u32> = accounting
            .contributions
            .iter()
            .filter_map(|(source, _)| match source {
                ChangeSource::Control { control, .. } => Some(control.0),
                _ => None,
            })
            .collect();
        assert_eq!(controls, vec![0, 1]);
        assert!(largest(&accounting.direct.values) > 1e-2);
        // A stage deleted at one endpoint has no value there.
        let deleted = global(&[(0, 0.0), (1, 0.5)]);
        assert!(matches!(
            account_finite_change(
                test_governor(),
                &program,
                &[0, 1],
                Endpoint { masks: &deleted, parameters: &parameters, inputs: vec![x.clone()], slots: Vec::new() },
                Endpoint { masks: &end, parameters: &parameters, inputs: vec![x], slots: Vec::new() },
            ),
            Err(AccountingError::StageSkipped { stage: 0, .. })
        ));
    }

    #[test]
    fn a_primitive_without_a_derived_band_is_refused() {
        let program = Program::new(ProgramParts {
            parameters: parameters(1),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "gelu",
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    native(NativePrimitive::Activation { activation: NativeActivation::ExactGelu }, &[1]),
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let parameters = DenseParameters::new(vec![DenseTensor::Matrix(Array2::eye(2))]);
        let masks = MaskAssignment::all_on();
        let x = array![[0.5, -0.25]];
        match account_finite_change(
            test_governor(),
            &program,
            &[0],
            Endpoint { masks: &masks, parameters: &parameters, inputs: vec![x.clone()], slots: Vec::new() },
            Endpoint { masks: &masks, parameters: &parameters, inputs: vec![x * 2.0], slots: Vec::new() },
        ) {
            Err(AccountingError::Replay(error)) => {
                assert!(matches!(*error, ReplayError::NoDerivedBand { primitive: "exact GELU", .. }), "{error}");
            }
            other => panic!("expected a refused GELU band, got {other:?}"),
        }
    }
}
