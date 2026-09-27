//! Forward-error bands of a traced mechanism-program execution under dense
//! parameters (#2951).
//!
//! [`Program::execute_traced`] returns every node value the executor formed. [`execute_banded`]
//! walks the same graph over that trace and gives each stored value `ŷ` two per-entry bands in
//! real arithmetic:
//!
//! * **local**: `|ŷ − f(x̂)|`, the rounding of the node's own evaluation at its STORED
//!   arguments `x̂`;
//! * **forward**: `|ŷ − y|`, against the exact program at the exact inputs. The arguments'
//!   forward bands `r` span a box around `x̂`; the forward band is a bound on how far `f`
//!   moves over that box plus the local band.
//!
//! The declared inputs and initial slot values are exact. Nothing here re-executes a value:
//! every value is the executor's, bit for bit, and a band that cannot be derived is refused
//! rather than guessed.
//!
//! # Per primitive
//!
//! Rounding follows `receipts`: [`evaluation_band`] is `γ_k` times an upward-rounded bound on
//! the absolute sum of the entry's terms plus the subnormal allowance. A sum of nonnegative
//! computed terms is raised to an upper bound on its exact value by [`inflated`] and one
//! subnormal spacing per product.
//!
//! * `Linear` `y = x Aᵀ`: local [`affine_stage_band`]; forward adds `r |A|ᵀ`.
//! * `AddBias`: local `γ_1 (|x| + |b|)`; forward adds `r`.
//! * ReLU: exact, and 1-Lipschitz, so forward is `r`. SiLU and the exact GELU are evaluated by
//!   gam-math owners that state no error bound, so they are refused
//!   ([`ReplayError::NoDerivedBand`]); so are SwiGLU, LayerNorm and the per-head norm.
//! * `Hadamard`: local `γ_1 |x y|` plus one subnormal spacing; forward adds
//!   `|x| r_y + r_x |y| + r_x r_y`.
//! * `CoordinateMask`, `Sum`: one product per entry whose mask is not `1`, then the sums;
//!   forward adds `Σ |m_i| r_i`.
//! * `Compose` stage `z = x + m (f(x) − x)`: at `m ∉ {0, 1}` three roundings,
//!   `γ_3 (|x| + |m| (|f| + |x|))`; forward `|1 − m| r_x + |m| r_f`, since `z = (1 − m) x + m f`
//!   exactly.
//! * RMSNorm `y = g ⊙ x/ρ(x)`, `ρ(x) = √(ε + ‖x‖²/d)`: local [`rms_norm_band`]. The Jacobian of
//!   `x/ρ(x)` is `(I − x xᵀ/(d ρ²))/ρ`, whose eigenvalues are `1/ρ` and
//!   `(1 − ‖x‖²/(d ρ²))/ρ ∈ (0, 1/ρ]`, so its spectral norm is at most `1/ρ`. Every point of the
//!   box has `|z_j| ≥ max(|x̂_j| − r_j, 0)`, so `ρ ≥ ρ_min = √(ε + Σ_j max(|x̂_j| − r_j, 0)²/d)` on
//!   it, and by the mean value theorem entry `i` moves by at most `|g_i| ‖r‖₂ / ρ_min`.
//! * Causal self-attention: the owner [`RotaryCausalAttention::attend_projected`] carries
//!   radii through rotation, scores, softmax and the value read; at zero input radius its
//!   radius is the local band, at the arguments' forward bands the forward band.

use super::attention::{AttentionProgramError, ProjectedRows, RotaryCausalAttention};
use super::block::{NormBandUnbounded, SUBNORMAL_SPACING, rms_norm_band};
use super::lift::TieOrientation;
use super::program::{
    BodyId, CallSite, DenseParameterError, DenseParameters, DenseTensor, ExecutionError, ExecutionTrace,
    MaskAssignment, NativeActivation, NativePrimitive, Node, NodeId, ParameterSlot, Program, ProgramError,
};
use super::receipts::{affine_stage_band, evaluation_band};
use super::rewrite::ShapeMismatch;
use super::secant::BandedMatrix;
use gam_math::roundoff::inflated;
use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayView2, Zip};
use std::collections::BTreeMap;
use std::fmt;

/// A refused banded execution.
#[derive(Debug)]
pub enum ReplayError {
    /// The executor refused the program, a parameter use or a reservation.
    Execution(Box<ExecutionError<DenseParameterError>>),
    Program(Box<ProgramError>),
    /// A primitive whose evaluation error no owner states, so no band is derived for it.
    NoDerivedBand { body: BodyId, node: NodeId, primitive: &'static str },
    /// The trace has no value for a node the walk reads: an executor invariant.
    MissingValue { body: BodyId, node: NodeId },
    /// A formal parameter bound to no tensor, or to a tensor of the wrong kind.
    Unbound { parameter: ParameterSlot },
    Shape(ShapeMismatch),
    Attention { body: BodyId, node: NodeId, error: Box<AttentionProgramError> },
    /// The attention owner's value differs from the executor's at the same arguments.
    AttentionMismatch { body: BodyId, node: NodeId },
    Norm { body: BodyId, node: NodeId, error: NormBandUnbounded },
    Memory(Box<MemoryReservationError>),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execution(error) => write!(formatter, "banded execution: {error}"),
            Self::Program(error) => write!(formatter, "banded execution: {error}"),
            Self::NoDerivedBand { body, node, primitive } => write!(
                formatter,
                "node {} of body {} is {primitive}, whose evaluation error no owner states",
                node.0, body.0
            ),
            Self::MissingValue { body, node } => {
                write!(formatter, "the trace holds no value of node {} of body {}", node.0, body.0)
            }
            Self::Unbound { parameter } => {
                write!(formatter, "formal parameter {} has no dense tensor of its kind", parameter.0)
            }
            Self::Shape(mismatch) => write!(formatter, "banded execution: {mismatch:?}"),
            Self::Attention { body, node, error } => {
                write!(formatter, "attention node {} of body {}: {error}", node.0, body.0)
            }
            Self::AttentionMismatch { body, node } => write!(
                formatter,
                "the attention owner's rows differ from the executor's at node {} of body {}",
                node.0, body.0
            ),
            Self::Norm { body, node, error } => {
                write!(formatter, "norm node {} of body {}: {error}", node.0, body.0)
            }
            Self::Memory(error) => write!(formatter, "banded execution: {error}"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<ProgramError> for ReplayError {
    fn from(error: ProgramError) -> Self {
        Self::Program(Box::new(error))
    }
}

impl From<ShapeMismatch> for ReplayError {
    fn from(error: ShapeMismatch) -> Self {
        Self::Shape(error)
    }
}

impl From<MemoryReservationError> for ReplayError {
    fn from(error: MemoryReservationError) -> Self {
        Self::Memory(Box::new(error))
    }
}

/// The two bands of one stored value.
#[derive(Debug)]
pub struct NodeBands {
    /// `|ŷ − f(x̂)|` at the stored arguments.
    pub local: Governed<Array2<f64>>,
    /// `|ŷ − y|` against the exact program at the exact inputs.
    pub forward: Governed<Array2<f64>>,
}

/// A traced execution with both bands of every value it formed.
#[derive(Debug)]
pub struct BandedExecution {
    /// The entry body's output and its forward band.
    pub output: BandedMatrix,
    pub trace: ExecutionTrace,
    bands: BTreeMap<(Vec<CallSite>, BodyId, NodeId), NodeBands>,
}

impl BandedExecution {
    /// The bands of node `node` of body `body` at the invocation `invocation`.
    pub fn bands(&self, invocation: &[CallSite], body: BodyId, node: NodeId) -> Option<&NodeBands> {
        self.bands.get(&(invocation.to_vec(), body, node))
    }
}

/// Runs `program` on exact `inputs` and exact initial `slots` under `masks`, with its
/// parameters bound to dense tensors, and derives both bands of every value.
pub fn execute_banded(
    governor: &MemoryGovernor,
    program: &Program,
    parameters: &DenseParameters,
    masks: &MaskAssignment,
    inputs: Vec<Array2<f64>>,
    slots: Vec<Option<Array2<f64>>>,
    positions: &[i64],
) -> Result<BandedExecution, ReplayError> {
    let input_radii: Vec<Array2<f64>> = inputs.iter().map(|rows| Array2::zeros(rows.raw_dim())).collect();
    let slot_radii = slots.iter().map(|slot| slot.as_ref().map(|rows| Array2::zeros(rows.raw_dim()))).collect();
    let (execution, trace) = program
        .execute_traced(governor, parameters, masks, inputs, slots, positions)
        .map_err(|error| ReplayError::Execution(Box::new(error)))?;
    let mut walker = Walker {
        governor,
        program,
        parameters,
        masks,
        trace: &trace,
        positions,
        slot_radii,
        bands: BTreeMap::new(),
    };
    let mut path = Vec::new();
    let radius = walker.body(program.parts().entry, input_radii, &mut path)?;
    let bands = walker.bands;
    Ok(BandedExecution {
        output: BandedMatrix { values: (*execution.output).clone(), bands: radius },
        trace,
        bands,
    })
}

/// `up(inflated(v, k) + terms · 2^-1074)`: an upper bound on the exact value of a nonnegative
/// sum of `terms` products whose computed value is `v` after `k` roundings on a path.
pub(super) fn raised(value: f64, roundings: usize, terms: usize) -> f64 {
    (inflated(value, roundings) + terms as f64 * SUBNORMAL_SPACING).next_up()
}

/// An upper bound on `|left| |right|` for nonnegative matrices, entrywise exact products summed.
pub(super) fn abs_product(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    let inner = left.ncols();
    left.mapv(f64::abs).dot(&right.mapv(f64::abs)).mapv(|value| raised(value, inner + 1, inner))
}

/// The matrix `A` a Linear node multiplies rows by, `y = x Aᵀ`.
pub(super) fn linear_matrix(
    parameters: &DenseParameters,
    weight: ParameterSlot,
    orientation: TieOrientation,
) -> Result<ArrayView2<'_, f64>, ReplayError> {
    match parameters.bound(weight) {
        Some(DenseTensor::Matrix(matrix)) => Ok(match orientation {
            TieOrientation::Identity => matrix.view(),
            TieOrientation::Transpose => matrix.t(),
        }),
        Some(DenseTensor::Vector(..)) | None => Err(ReplayError::Unbound { parameter: weight }),
    }
}

/// The vector bound to a formal parameter.
pub(super) fn bound_vector(parameters: &DenseParameters, parameter: ParameterSlot) -> Result<&Array1<f64>, ReplayError> {
    match parameters.bound(parameter) {
        Some(DenseTensor::Vector(vector)) => Ok(vector),
        Some(DenseTensor::Matrix(..)) | None => Err(ReplayError::Unbound { parameter }),
    }
}

/// The name of a primitive with no derived evaluation band, or `None`.
pub(super) fn underived_primitive(primitive: &NativePrimitive) -> Option<&'static str> {
    match primitive {
        NativePrimitive::Activation { activation: NativeActivation::Silu } => Some("SiLU"),
        NativePrimitive::Activation { activation: NativeActivation::ExactGelu } => Some("exact GELU"),
        NativePrimitive::SwiGlu => Some("SwiGLU"),
        NativePrimitive::LayerNorm { .. } => Some("LayerNorm"),
        NativePrimitive::HeadRmsNorm { .. } => Some("a per-head RMSNorm"),
        NativePrimitive::Linear { .. }
        | NativePrimitive::AddBias { .. }
        | NativePrimitive::Activation { activation: NativeActivation::Relu }
        | NativePrimitive::Hadamard
        | NativePrimitive::CoordinateMask { .. }
        | NativePrimitive::RmsNorm { .. }
        | NativePrimitive::CausalSelfAttention { .. } => None,
    }
}

/// The local band of a Sum-shaped combination `Σ_i m_i x_i` of stored values, the
/// executor's order: a product for every `m ∉ {0, 1}`, then the additions.
pub(super) fn weighted_sum_band(terms: &[(f64, &Array2<f64>)], shape: (usize, usize)) -> Array2<f64> {
    let active: Vec<&(f64, &Array2<f64>)> = terms.iter().filter(|(m, _)| *m != 0.0).collect();
    let products = active.iter().filter(|(m, _)| *m != 1.0).count();
    let roundings = active.len().saturating_sub(1) + usize::from(products > 0);
    let mut band = Array2::zeros(shape);
    if roundings == 0 {
        return band;
    }
    for ((row, column), slot) in band.indexed_iter_mut() {
        let absolute = active
            .iter()
            .fold(0.0_f64, |sum, (m, x)| (sum + (m * x[[row, column]]).abs().next_up()).next_up());
        *slot = evaluation_band(roundings, absolute, products as f64);
    }
    band
}

/// One composition stage `x + m (f − x)` at `m ∉ {0, 1}` in the executor's order of
/// operations, `fl(fl(m fl(f − x)) + x)`, and its local band
/// `γ_3 (|x| + |m| (|f| + |x|))` plus one subnormal spacing for the product.
pub(super) fn compose_step(current: &Array2<f64>, staged: &Array2<f64>, m: f64) -> (Array2<f64>, Array2<f64>) {
    let mut combined = staged - current;
    combined *= m;
    combined += current;
    let mut band = Array2::<f64>::zeros(current.raw_dim());
    Zip::from(&mut band).and(current).and(staged).for_each(|slot, &x, &f| {
        let absolute = (x.abs() + (m.abs() * (f.abs() + x.abs()).next_up()).next_up()).next_up();
        *slot = evaluation_band(3, absolute, 1.0);
    });
    (combined, band)
}

/// `ρ_min` of the module documentation for one row and its radius, rounded down.
pub(super) fn rms_floor(row: ndarray::ArrayView1<'_, f64>, radius: ndarray::ArrayView1<'_, f64>, epsilon: f64) -> f64 {
    let squares = row.iter().zip(radius.iter()).fold(0.0_f64, |sum, (&x, &r)| {
        let low = (x.abs() - r).next_down().max(0.0);
        (sum + (low * low).next_down().max(0.0)).next_down().max(0.0)
    });
    let mean = (squares / row.len() as f64).next_down().max(0.0);
    (epsilon + mean).next_down().max(0.0).sqrt().next_down().max(0.0)
}

struct Walker<'a> {
    governor: &'a MemoryGovernor,
    program: &'a Program,
    parameters: &'a DenseParameters,
    masks: &'a MaskAssignment,
    trace: &'a ExecutionTrace,
    positions: &'a [i64],
    slot_radii: Vec<Option<Array2<f64>>>,
    bands: BTreeMap<(Vec<CallSite>, BodyId, NodeId), NodeBands>,
}

impl Walker<'_> {
    fn value(&self, path: &[CallSite], body: BodyId, node: NodeId) -> Result<&Array2<f64>, ReplayError> {
        self.trace.value(path, body, node).ok_or(ReplayError::MissingValue { body, node })
    }

    fn body(
        &mut self,
        body_id: BodyId,
        input_radii: Vec<Array2<f64>>,
        path: &mut Vec<CallSite>,
    ) -> Result<Array2<f64>, ReplayError> {
        let program = self.program;
        let body = &program.parts().bodies[body_id.index()];
        let mut forward: Vec<Option<Array2<f64>>> = vec![None; body.nodes.len()];
        for (i, node) in body.nodes.iter().enumerate() {
            let node_id = NodeId(i as u32);
            let site = CallSite { body: body_id, node: node_id, stage: 0 };
            let value = self.value(path, body_id, node_id)?;
            let shape = value.dim();
            let radius_of = |argument: NodeId| -> Result<&Array2<f64>, ReplayError> {
                forward[argument.index()]
                    .as_ref()
                    .ok_or(ReplayError::MissingValue { body: body_id, node: argument })
            };
            let (local, radius) = match node {
                Node::Input { port } => (Array2::zeros(shape), input_radii[*port as usize].clone()),
                Node::Read { slot } => {
                    let radius = self.slot_radii[slot.index()].clone().unwrap_or_else(|| Array2::zeros(shape));
                    (Array2::zeros(shape), radius)
                }
                Node::Write { slot, value: argument } => {
                    let radius = radius_of(*argument)?.clone();
                    self.slot_radii[slot.index()] = Some(radius.clone());
                    (Array2::zeros(shape), radius)
                }
                Node::Native { primitive, arguments } => {
                    let radii = arguments
                        .iter()
                        .map(|&argument| radius_of(argument).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    self.native(site, path, primitive, arguments, &radii)?
                }
                Node::Sum { terms } => {
                    let mut weighted = Vec::with_capacity(terms.len());
                    let mut radius = Array2::<f64>::zeros(shape);
                    for term in terms {
                        let m = match term.control {
                            Some(control) => program.control_value(self.masks, control, path)?,
                            None => 1.0,
                        };
                        let argument = self.value(path, body_id, term.value)?;
                        weighted.push((m, argument));
                        let own = radius_of(term.value)?;
                        Zip::from(&mut radius).and(own).for_each(|slot, &r| *slot = (*slot + (m.abs() * r).next_up()).next_up());
                    }
                    let local = weighted_sum_band(&weighted, shape);
                    radius += &local;
                    radius.mapv_inplace(f64::next_up);
                    (local, radius)
                }
                Node::Compose { value: argument, stages } => {
                    let mut current_radius = radius_of(*argument)?.clone();
                    let mut current = self.value(path, body_id, *argument)?.clone();
                    let mut local = Array2::<f64>::zeros(shape);
                    for (k, stage) in stages.iter().enumerate() {
                        let m = match stage.control {
                            Some(control) => program.control_value(self.masks, control, path)?,
                            None => 1.0,
                        };
                        if m == 0.0 {
                            continue;
                        }
                        path.push(CallSite { stage: k as u32, ..site });
                        let staged_radius = self.body(stage.body, vec![current_radius.clone()], path);
                        let stage_output = program.parts().bodies[stage.body.index()].output;
                        let staged = self.value(path, stage.body, stage_output).cloned();
                        path.pop();
                        let (staged_radius, staged) = (staged_radius?, staged?);
                        if m == 1.0 {
                            current_radius = staged_radius;
                            current = staged;
                            continue;
                        }
                        let (combined, own) = compose_step(&current, &staged, m);
                        let mut moved = Array2::<f64>::zeros(shape);
                        for ((row, column), slot) in moved.indexed_iter_mut() {
                            let reach = ((1.0 - m).abs().next_up() * current_radius[[row, column]]).next_up()
                                + (m.abs() * staged_radius[[row, column]]).next_up();
                            *slot = (reach.next_up() + own[[row, column]]).next_up();
                        }
                        current = combined;
                        local = own;
                        current_radius = moved;
                    }
                    (local, current_radius)
                }
                Node::Call { body: callee, arguments } => {
                    let radii = arguments
                        .iter()
                        .map(|&argument| radius_of(argument).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    path.push(site);
                    let radius = self.body(*callee, radii, path);
                    path.pop();
                    (Array2::zeros(shape), radius?)
                }
                Node::Refine { native, mechanism, arguments } => {
                    let radii = arguments
                        .iter()
                        .map(|&argument| radius_of(argument).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    path.push(site);
                    let chosen = if self.trace.ran(path, *native) { *native } else { *mechanism };
                    let radius = self.body(chosen, radii, path);
                    path.pop();
                    (Array2::zeros(shape), radius?)
                }
            };
            let governed = |rows: Array2<f64>| -> Result<Governed<Array2<f64>>, ReplayError> {
                Ok(self.governor.try_reserve_dense_f64(rows.nrows(), rows.ncols(), "program value band")?.bind(rows))
            };
            let bands = NodeBands { local: governed(local)?, forward: governed(radius.clone())? };
            self.bands.insert((path.clone(), body_id, node_id), bands);
            forward[i] = Some(radius);
        }
        forward[body.output.index()]
            .take()
            .ok_or(ReplayError::MissingValue { body: body_id, node: body.output })
    }

    /// The local and forward bands of one native node.
    fn native(
        &self,
        site: CallSite,
        path: &[CallSite],
        primitive: &NativePrimitive,
        arguments: &[NodeId],
        radii: &[Array2<f64>],
    ) -> Result<(Array2<f64>, Array2<f64>), ReplayError> {
        let (body, node) = (site.body, site.node);
        if let Some(name) = underived_primitive(primitive) {
            return Err(ReplayError::NoDerivedBand { body, node, primitive: name });
        }
        let x = self.value(path, body, arguments[0])?;
        let y = self.value(path, body, node)?;
        let shape = y.dim();
        let mut local = Array2::<f64>::zeros(shape);
        let mut radius = Array2::<f64>::zeros(shape);
        match primitive {
            NativePrimitive::Linear { weight, orientation } => {
                let matrix = linear_matrix(self.parameters, *weight, *orientation)?;
                local = affine_stage_band(matrix, None, x.view())?;
                radius = abs_product(radii[0].view(), matrix.t());
            }
            NativePrimitive::AddBias { bias } => {
                let vector = bound_vector(self.parameters, *bias)?;
                for ((row, column), slot) in local.indexed_iter_mut() {
                    let absolute = (x[[row, column]].abs() + vector[column].abs()).next_up();
                    *slot = evaluation_band(1, absolute, 0.0);
                }
                radius.assign(&radii[0]);
            }
            NativePrimitive::Activation { .. } => radius.assign(&radii[0]),
            NativePrimitive::Hadamard => {
                let other = self.value(path, body, arguments[1])?;
                for ((row, column), slot) in local.indexed_iter_mut() {
                    let (a, b) = (x[[row, column]], other[[row, column]]);
                    let (ra, rb) = (radii[0][[row, column]], radii[1][[row, column]]);
                    *slot = evaluation_band(1, (a * b).abs().next_up(), 1.0);
                    radius[[row, column]] = raised(a.abs() * rb + ra * b.abs() + ra * rb, 4, 3);
                }
            }
            NativePrimitive::CoordinateMask { controls } => {
                for (c, &control) in controls.iter().enumerate() {
                    let m = self.program.control_value(self.masks, control, path)?;
                    for row in 0..shape.0 {
                        if m != 1.0 {
                            local[[row, c]] = evaluation_band(1, (m * x[[row, c]]).abs().next_up(), 1.0);
                        }
                        radius[[row, c]] = (m.abs() * radii[0][[row, c]]).next_up();
                    }
                }
            }
            NativePrimitive::RmsNorm { epsilon, gain } => {
                let gain = bound_vector(self.parameters, *gain)?;
                local = rms_norm_band(*epsilon, gain.view(), x.view(), y.view())
                    .map_err(|error| ReplayError::Norm { body, node, error })?;
                for row in 0..shape.0 {
                    let floor = rms_floor(x.row(row), radii[0].row(row), *epsilon);
                    let norm = radii[0].row(row).iter().fold(0.0_f64, |sum, r| (sum + (r * r).next_up()).next_up());
                    let reach = if norm == 0.0 { 0.0 } else { (norm.sqrt().next_up() / floor).next_up() };
                    for column in 0..shape.1 {
                        radius[[row, column]] = (gain[column].abs() * reach).next_up();
                    }
                }
            }
            NativePrimitive::CausalSelfAttention { geometry, rotary, score_scale } => {
                let attention_error = |error| ReplayError::Attention { body, node, error: Box::new(error) };
                let attention =
                    RotaryCausalAttention::new(*geometry, rotary.clone(), *score_scale).map_err(attention_error)?;
                let keys = self.value(path, body, arguments[1])?;
                let values = self.value(path, body, arguments[2])?;
                let exact = attention
                    .attend_projected(
                        self.governor,
                        ProjectedRows::exact(x.view()),
                        ProjectedRows::exact(keys.view()),
                        ProjectedRows::exact(values.view()),
                        self.positions,
                    )
                    .map_err(attention_error)?;
                if exact.mixed != *y {
                    return Err(ReplayError::AttentionMismatch { body, node });
                }
                let boxed = attention
                    .attend_projected(
                        self.governor,
                        ProjectedRows { values: x.view(), radius: radii[0].view() },
                        ProjectedRows { values: keys.view(), radius: radii[1].view() },
                        ProjectedRows { values: values.view(), radius: radii[2].view() },
                        self.positions,
                    )
                    .map_err(attention_error)?;
                local = exact.mixed_radius.clone();
                // The owner's radius at the arguments' boxes already carries its own rounding.
                return Ok((local, boxed.mixed_radius.clone()));
            }
            NativePrimitive::SwiGlu
            | NativePrimitive::LayerNorm { .. }
            | NativePrimitive::HeadRmsNorm { .. } => {}
        }
        Zip::from(&mut radius).and(&local).for_each(|slot, &own| *slot = (*slot + own).next_up());
        Ok((local, radius))
    }
}
