//! An operator program executed on a device, values resident (#2951).
//!
//! [`DeviceProgram`] lowers a program's nodes onto `gam_gpu::tensor`'s operations once, keeps its
//! operators on the device, and then runs forward passes ([`DeviceProgram::forward`]), the KL of
//! the logits against a target with its cotangent ([`DeviceProgram::kl`]), sampled-label
//! cotangents ([`DeviceProgram::sampled`]), reverse passes ([`DeviceProgram::vjp`]) and tangents
//! in the operators' reals ([`DeviceProgram::jvp`], [`DeviceProgram::quadratic`]) with every node
//! value staying on the device: only what a caller asks for comes back.
//!
//! The rules are the CPU's (`OperatorProgram::execute`, `derivatives::{vjp, jvp}`) term for term:
//! an affine node adds its terms in order and then its bias; attention rotates its queries and
//! keys by the CPU's own angles (`Rotary::turn`, tabulated on the host), scores `c q·k`, takes the
//! max-shifted softmax over the keys at or before each position and reads the values; a norm is
//! `x / √(mean(x²) + ε)`. Each sequence is one block of rows, so the attention of every sequence
//! runs as one strided-batched product: a family of many sequences fills the device in one launch.
//! On the host backend the same lowering runs on the CPU; that is how it is tested everywhere.
//!
//! The logits are never formed whole. The program's output must be a linear head on a hidden
//! node (`h A` for a tied unembedding read transposed, or `h Aᵀ`, then an indicator readout), and
//! the head runs in row tiles of at most [`TILE_BYTES`]: each tile's logits, its KL against the
//! target's tile and its cotangent, pulled straight back to the hidden node.
//!
//! What is lowered: features of an indicator basis read only through affine terms (a gather of the
//! operator's columns), raw slots, constants, affine nodes over identity, diagonal, dense and
//! low-rank operators, pointwise laws, Hadamard products, RMS norms, attention, and the head.
//! Any other node refuses the whole program with its reason; its caller then runs on the CPU.

use super::operator_program::{
    Basis, FamilyInputs, Law, Node, Operator, OperatorBody, OperatorProgram, Rotary, SlotValues,
};
use gam_gpu::tensor::{Arithmetic, Device, Indices, Op, PointwiseLaw, Tensor};
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The largest logits tile the head forms at once, in bytes.
pub const TILE_BYTES: usize = 1 << 30;

fn error(e: impl std::fmt::Display) -> String {
    format!("device: {e}")
}

/// The roles an operator is held on the device in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    /// Products `x Aᵀ` (an affine term), `g A` (its cotangent), or `h A` (a transposed read).
    Product,
    /// Its transpose as a table of rows, gathered by token (an affine term on a one-hot feature).
    Table,
    /// Its single column as a row (a bias or a constant).
    Column,
}

/// An operator on the device in one role.
enum Held {
    Identity,
    Diagonal(Tensor),
    Dense(Tensor),
    /// `left` (rows × r) and `right` (r × cols).
    LowRank(Tensor, Tensor),
    Table(Tensor),
    Column(Tensor),
}

struct HeldOperator {
    source: Arc<Operator>,
    held: Held,
}

fn hold(device: &Device, op: &Operator, role: Role) -> Result<Held, String> {
    Ok(match role {
        Role::Column => Held::Column(device.upload(op.matrix().t().view()).map_err(error)?),
        Role::Table => Held::Table(device.upload(op.matrix_cow().t()).map_err(error)?),
        Role::Product => match &op.body {
            OperatorBody::Identity => Held::Identity,
            OperatorBody::Diagonal { values, .. } => Held::Diagonal(device.upload_vec(1, values.len(), values.to_vec()).map_err(error)?),
            OperatorBody::LowRank { left, right, .. } => {
                Held::LowRank(device.upload(left.view()).map_err(error)?, device.upload(right.view()).map_err(error)?)
            }
            OperatorBody::Dense { values, .. } => match op.diagonal() {
                Some(d) => Held::Diagonal(device.upload_vec(1, d.len(), d.to_vec()).map_err(error)?),
                None => Held::Dense(device.upload(values.view()).map_err(error)?),
            },
        },
    })
}

/// How one node executes.
enum Step {
    /// A one-hot feature of an indicator basis: only its token ids (its readers gather).
    Feature { slot: usize },
    Raw { slot: usize },
    Constant { operator: usize },
    Affine { terms: Vec<(usize, usize)>, bias: Option<usize> },
    Pointwise { input: usize, codes: Indices },
    Hadamard { left: usize, right: usize },
    RmsNorm { input: usize, epsilon: f64 },
    Attend { query: usize, key: usize, value: usize, scale: f64, rotary: Option<Rotary>, causal: bool },
    Transposed { input: usize, operator: usize },
    /// The head's logits or their readout: never formed whole.
    Head,
}

/// The output's linear head: logits `h A` (`transposed`) or `h Aᵀ` from the hidden node `h`.
#[derive(Clone, Copy, Debug)]
struct Head {
    hidden: usize,
    operator: usize,
    transposed: bool,
    classes: usize,
}

/// The constant of the tanh GELU as the CPU forms it (`operator_program`'s `gelu_tanh`).
fn gelu_tanh_constant() -> f64 {
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2
}

fn law_of(law: Law) -> PointwiseLaw {
    match law {
        Law::Relu => PointwiseLaw::Relu,
        Law::Identity => PointwiseLaw::Identity,
        Law::Zero => PointwiseLaw::Zero,
        Law::Silu => PointwiseLaw::Silu,
        Law::Gelu => PointwiseLaw::Gelu,
        Law::GeluTanh => PointwiseLaw::GeluTanh,
    }
}

/// A program lowered onto a device (module note).
pub struct DeviceProgram {
    device: Device,
    steps: Vec<Step>,
    /// Each node's width.
    widths: Vec<usize>,
    head: Head,
    operators: BTreeMap<(usize, Role), HeldOperator>,
}

/// One forward pass's node values on the device (`None` for features and the head), with the
/// family's blocks and rotation tables.
pub struct DeviceTrace {
    pub values: Vec<Option<Tensor>>,
    pub rows: usize,
    ids: BTreeMap<usize, Indices>,
    /// Sequences (equal row blocks) and their length.
    blocks: usize,
    /// Per rotary configuration, `(cos, sin)` per row and plane.
    rotations: Vec<(Rotary, Tensor, Tensor)>,
}

impl DeviceTrace {
    /// Node `n`'s value.
    pub fn value(&self, n: usize) -> Result<&Tensor, String> {
        self.values.get(n).and_then(Option::as_ref).ok_or_else(|| format!("device: node {n} has no resident value"))
    }
}

/// `out`, zeros of `rows × width` first when it is empty.
fn ensure<'a>(d: &Device, out: &'a mut Option<Tensor>, rows: usize, width: usize) -> Result<&'a mut Tensor, String> {
    if out.is_none() {
        *out = Some(d.zeros(rows, width).map_err(error)?);
    }
    out.as_mut().ok_or_else(|| "device: tangent slot".to_string())
}

fn rotations_of(trace: &DeviceTrace, rotary: Rotary) -> Result<(&Tensor, &Tensor), String> {
    trace.rotations.iter().find(|(r, _, _)| *r == rotary).map(|(_, c, s)| (c, s)).ok_or_else(|| "device: no rotation table".to_string())
}

impl DeviceProgram {
    /// Lower `program` onto `device`, or the reason it cannot be.
    pub fn compile(device: &Device, program: &OperatorProgram) -> Result<Self, String> {
        let interfaces = program.interfaces().map_err(|e| e.to_string())?;
        let widths: Vec<usize> = interfaces.iter().map(|i| i.width()).collect();
        let head = Self::head_of(program)?;
        let head_nodes: Vec<usize> = match &program.nodes[program.output] {
            Node::Readout { input, .. } => vec![program.output, *input],
            _ => vec![program.output],
        };
        // Which nodes read each node: a feature may only be read by affine terms.
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); program.nodes.len()];
        for (index, node) in program.nodes.iter().enumerate() {
            for argument in node.arguments() {
                readers[argument].push(index);
            }
        }
        let mut steps = Vec::with_capacity(program.nodes.len());
        let mut wanted: Vec<(usize, Role)> = vec![(head.operator, Role::Product)];
        for (index, node) in program.nodes.iter().enumerate() {
            if head_nodes.contains(&index) {
                steps.push(Step::Head);
                continue;
            }
            let refuse = |what: &str| Err(format!("node {index}: {what} has no device rule"));
            let step = match node {
                Node::Feature { slot, basis } => {
                    if !matches!(program.bases[*basis], Basis::Indicator { .. }) {
                        return refuse("a feature of a character basis");
                    }
                    if readers[index].iter().any(|r| !matches!(program.nodes[*r], Node::Affine { .. })) {
                        return refuse("a feature read outside an affine term");
                    }
                    Step::Feature { slot: *slot }
                }
                Node::Raw { slot } => Step::Raw { slot: *slot },
                Node::Constant { operator } => {
                    wanted.push((*operator, Role::Column));
                    Step::Constant { operator: *operator }
                }
                Node::Affine { terms, bias } => {
                    for (argument, operator) in terms {
                        let feature = matches!(program.nodes[*argument], Node::Feature { .. });
                        wanted.push((*operator, if feature { Role::Table } else { Role::Product }));
                    }
                    if let Some(b) = bias {
                        wanted.push((*b, Role::Column));
                    }
                    Step::Affine { terms: terms.clone(), bias: *bias }
                }
                Node::Pointwise { input, laws } => {
                    let interface = &interfaces[*input];
                    let mut codes = vec![PointwiseLaw::Identity.code(); widths[*input]];
                    for (group, law) in laws.iter().enumerate() {
                        for c in interface.range(group) {
                            codes[c] = law_of(*law).code();
                        }
                    }
                    Step::Pointwise { input: *input, codes: device.upload_indices(&codes).map_err(error)? }
                }
                Node::Hadamard { left, right } => Step::Hadamard { left: *left, right: *right },
                Node::RmsNorm { input, epsilon } => Step::RmsNorm { input: *input, epsilon: *epsilon },
                Node::Attend { query, key, value, scale, rotary, causal } => {
                    if let Some(r) = rotary
                        && 2 * r.pairs().len() > widths[*query]
                    {
                        return refuse("a rotation wider than its head");
                    }
                    Step::Attend { query: *query, key: *key, value: *value, scale: scale.value(), rotary: *rotary, causal: *causal }
                }
                Node::Transposed { input, operator } => {
                    wanted.push((*operator, Role::Product));
                    Step::Transposed { input: *input, operator: *operator }
                }
                Node::Readout { .. } => return refuse("a readout before the output"),
                Node::Bilinear { .. } => return refuse("a bilinear node"),
                Node::Softmax { .. } => return refuse("a softmax node"),
                Node::Mix { .. } => return refuse("a mix node"),
                Node::Outer { .. } => return refuse("an outer product"),
                Node::Concat { .. } => return refuse("a concatenation"),
                Node::Param { .. } | Node::Call { .. } => return refuse("a rule"),
                Node::Gain { .. } => return refuse("a parameter gain"),
            };
            steps.push(step);
        }
        let mut operators = BTreeMap::new();
        for key in wanted {
            if operators.contains_key(&key) {
                continue;
            }
            let source = Arc::clone(&program.operators[key.0]);
            let held = hold(device, &source, key.1)?;
            operators.insert(key, HeldOperator { source, held });
        }
        Ok(Self { device: device.clone(), steps, widths, head, operators })
    }

    fn head_of(program: &OperatorProgram) -> Result<Head, String> {
        let logits = match &program.nodes[program.output] {
            Node::Readout { input, basis } => {
                if !matches!(program.bases[*basis], Basis::Indicator { .. }) {
                    return Err("the output reads a character basis".to_string());
                }
                *input
            }
            _ => program.output,
        };
        let (hidden, operator, transposed) = match &program.nodes[logits] {
            Node::Transposed { input, operator } => (*input, *operator, true),
            Node::Affine { terms, bias: None } if terms.len() == 1 => (terms[0].0, terms[0].1, false),
            _ => return Err("the output is not a linear head on a hidden node".to_string()),
        };
        if !matches!(program.operators[operator].body, OperatorBody::Dense { .. }) || program.operators[operator].diagonal().is_some() {
            return Err("the head's operator is not a dense matrix".to_string());
        }
        let a = program.operators[operator].matrix_cow();
        let classes = if transposed { a.ncols() } else { a.nrows() };
        Ok(Head { hidden, operator, transposed, classes })
    }

    /// The device it runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The node the head reads.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.head.hidden
    }

    /// The number of classes the head scores.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.head.classes
    }

    /// Re-upload every operator `program` now holds a different copy of (a stepped library).
    pub fn refresh(&mut self, program: &OperatorProgram) -> Result<(), String> {
        for ((op, role), held) in &mut self.operators {
            if !Arc::ptr_eq(&held.source, &program.operators[*op]) {
                held.source = Arc::clone(&program.operators[*op]);
                held.held = hold(&self.device, &held.source, *role)?;
            }
        }
        Ok(())
    }

    fn held(&self, op: usize, role: Role) -> Result<&Held, String> {
        self.operators.get(&(op, role)).map(|h| &h.held).ok_or_else(|| format!("device: operator {op} not held as {role:?}"))
    }

    /// The bytes a forward pass keeps per row (every resident node value), for sizing batches.
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        self.steps
            .iter()
            .zip(&self.widths)
            .filter(|(s, _)| !matches!(s, Step::Feature { .. } | Step::Head))
            .map(|(_, w)| w * 8)
            .sum()
    }

    /// The family's blocks and rotation tables: every sequence one contiguous block of equal
    /// length, positions strictly increasing within it.
    fn layout(&self, family: &FamilyInputs) -> Result<(usize, Vec<(Rotary, Tensor, Tensor)>), String> {
        let rotaries: Vec<Rotary> = {
            let mut list: Vec<Rotary> = Vec::new();
            for step in &self.steps {
                if let Step::Attend { rotary: Some(r), .. } = step
                    && !list.contains(r)
                {
                    list.push(*r);
                }
            }
            list
        };
        let attends = self.steps.iter().any(|s| matches!(s, Step::Attend { .. }));
        let Some(layout) = &family.layout else {
            if attends {
                return Err("device: attention needs a sequence layout".to_string());
            }
            return Ok((1, Vec::new()));
        };
        let rows = family.rows;
        let length = (1..=rows).find(|&l| l == rows || layout.sequence[l] != layout.sequence[0]).unwrap_or(rows);
        if attends {
            if length == 0 || rows % length != 0 {
                return Err(format!("device: {rows} rows are not equal sequence blocks of {length}"));
            }
            for b in 0..rows / length {
                let block = b * length..(b + 1) * length;
                let id = layout.sequence[block.start];
                if layout.sequence[block.clone()].iter().any(|s| *s != id)
                    || (b > 0 && layout.sequence[block.start - 1] == id)
                    || layout.position[block].windows(2).any(|w| w[1] <= w[0])
                {
                    return Err("device: a sequence is not one block of increasing positions".to_string());
                }
            }
        }
        let mut rotations = Vec::new();
        for rotary in rotaries {
            let planes = rotary.pairs().len();
            let (mut cos, mut sin) = (Vec::with_capacity(rows * planes), Vec::with_capacity(rows * planes));
            for &position in &layout.position {
                for plane in 0..planes {
                    let (c, s) = rotary.turn(plane, position);
                    cos.push(c);
                    sin.push(s);
                }
            }
            rotations.push((
                rotary,
                self.device.upload_vec(rows, planes, cos).map_err(error)?,
                self.device.upload_vec(rows, planes, sin).map_err(error)?,
            ));
        }
        Ok((if attends { rows / length } else { 1 }, rotations))
    }

    /// `out ← out + x op(A)` for an affine term (`x Aᵀ`) or a transposed read (`x A`).
    fn add_product(&self, out: &mut Tensor, x: &Tensor, op: usize, transposed: bool, arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        match self.held(op, Role::Product)? {
            Held::Identity => d.axpy(out, 1.0, x).map_err(error),
            Held::Diagonal(diag) => d.scale_columns(out, x, diag, true).map_err(error),
            Held::Dense(a) => d.gemm(out, 1.0, x, Op::N, a, if transposed { Op::N } else { Op::T }, 1.0, arithmetic).map_err(error),
            Held::LowRank(left, right) => {
                // x (L R)ᵀ = (x Rᵀ) Lᵀ;  x (L R) = (x L) R.
                let (first, first_op, second, second_op) =
                    if transposed { (left, Op::N, right, Op::N) } else { (right, Op::T, left, Op::T) };
                let inner = if transposed { left.cols() } else { right.rows() };
                let mut middle = d.zeros(x.rows(), inner).map_err(error)?;
                d.gemm(&mut middle, 1.0, x, Op::N, first, first_op, 0.0, arithmetic).map_err(error)?;
                d.gemm(out, 1.0, &middle, Op::N, second, second_op, 1.0, arithmetic).map_err(error)
            }
            Held::Table(_) | Held::Column(_) => Err("device: an operator held in the wrong role".to_string()),
        }
    }

    fn column(&self, op: usize) -> Result<&Tensor, String> {
        match self.held(op, Role::Column)? {
            Held::Column(c) => Ok(c),
            _ => Err("device: an operator held in the wrong role".to_string()),
        }
    }

    /// One forward pass on `family` (module note).
    pub fn forward(&self, family: &FamilyInputs) -> Result<DeviceTrace, String> {
        let d = &self.device;
        let rows = family.rows;
        let (blocks, rotations) = self.layout(family)?;
        let mut trace = DeviceTrace { values: Vec::with_capacity(self.steps.len()), rows, ids: BTreeMap::new(), blocks, rotations };
        for (index, step) in self.steps.iter().enumerate() {
            let width = self.widths[index];
            let value = match step {
                Step::Head => None,
                Step::Feature { slot } => {
                    let SlotValues::Tokens(tokens) = &family.slots[*slot] else {
                        return Err(format!("device: slot {slot} holds no tokens"));
                    };
                    if !trace.ids.contains_key(slot) {
                        trace.ids.insert(*slot, d.upload_indices(tokens).map_err(error)?);
                    }
                    None
                }
                Step::Raw { slot } => {
                    let SlotValues::Raw(values) = &family.slots[*slot] else {
                        return Err(format!("device: slot {slot} holds no raw rows"));
                    };
                    Some(d.upload(values.view()).map_err(error)?)
                }
                Step::Constant { operator } => Some(d.broadcast_rows(self.column(*operator)?, rows).map_err(error)?),
                Step::Affine { terms, bias } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    for (argument, operator) in terms {
                        if let Step::Feature { slot } = &self.steps[*argument] {
                            let Held::Table(table) = self.held(*operator, Role::Table)? else {
                                return Err("device: an operator held in the wrong role".to_string());
                            };
                            let ids = trace.ids.get(slot).ok_or("device: feature ids missing")?;
                            let gathered = d.gather_rows(table, ids).map_err(error)?;
                            d.axpy(&mut out, 1.0, &gathered).map_err(error)?;
                        } else {
                            self.add_product(&mut out, trace.value(*argument)?, *operator, false, Arithmetic::F64)?;
                        }
                    }
                    if let Some(b) = bias {
                        d.add_row(&mut out, 1.0, self.column(*b)?).map_err(error)?;
                    }
                    Some(out)
                }
                Step::Pointwise { input, codes } => Some(d.law_values(trace.value(*input)?, codes, gelu_tanh_constant()).map_err(error)?),
                Step::Hadamard { left, right } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    d.hadamard(&mut out, trace.value(*left)?, trace.value(*right)?, false).map_err(error)?;
                    Some(out)
                }
                Step::RmsNorm { input, epsilon } => Some(d.rms_norm(trace.value(*input)?, *epsilon).map_err(error)?),
                Step::Attend { query, key, value, scale, rotary, causal } => {
                    let (q, k) = self.rotated(&trace, *query, *key, *rotary)?;
                    let alpha = self.attention(&trace, &q, &k, *scale, *causal)?;
                    let v = trace.value(*value)?;
                    let mut out = d.zeros(rows, v.cols()).map_err(error)?;
                    d.gemm_batched(trace.blocks, &mut out, 1.0, &alpha, Op::N, v, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
                    Some(out)
                }
                Step::Transposed { input, operator } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    self.add_product(&mut out, trace.value(*input)?, *operator, true, Arithmetic::F64)?;
                    Some(out)
                }
            };
            trace.values.push(value);
        }
        Ok(trace)
    }

    /// The query and key of an attend node turned to their positions.
    fn rotated(&self, trace: &DeviceTrace, query: usize, key: usize, rotary: Option<Rotary>) -> Result<(Tensor, Tensor), String> {
        let d = &self.device;
        let (q, k) = (trace.value(query)?, trace.value(key)?);
        match rotary {
            None => Ok((d.copy(q).map_err(error)?, d.copy(k).map_err(error)?)),
            Some(r) => {
                let (cos, sin) = rotations_of(trace, r)?;
                Ok((
                    d.rotate(q, cos, sin, r.half_split, false).map_err(error)?,
                    d.rotate(k, cos, sin, r.half_split, false).map_err(error)?,
                ))
            }
        }
    }

    /// The attention weights `softmax(c q kᵀ)` of every block (`blocks · L × L`).
    fn attention(&self, trace: &DeviceTrace, q: &Tensor, k: &Tensor, scale: f64, causal: bool) -> Result<Tensor, String> {
        let d = &self.device;
        let length = trace.rows / trace.blocks;
        let mut scores = d.zeros(trace.rows, length).map_err(error)?;
        d.gemm_batched(trace.blocks, &mut scores, scale, q, Op::N, k, Op::T, 0.0, Arithmetic::F64).map_err(error)?;
        d.softmax_rows(&mut scores, causal).map_err(error)?;
        Ok(scores)
    }

    /// Rows `start..start + n` of the logits.
    fn logits_tile(&self, hidden: &Tensor, start: usize, n: usize, arithmetic: Arithmetic) -> Result<Tensor, String> {
        let d = &self.device;
        let h = d.rows_of(hidden, start, n).map_err(error)?;
        let mut logits = d.zeros(n, self.head.classes).map_err(error)?;
        self.add_product(&mut logits, &h, self.head.operator, self.head.transposed, arithmetic)?;
        Ok(logits)
    }

    /// `g_h ← g_h + g_logits · ∂logits/∂h` for one tile, written at rows `start..`.
    fn pull_tile(&self, g_hidden: &mut Tensor, start: usize, cotangent: &Tensor, arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        let Held::Dense(a) = self.held(self.head.operator, Role::Product)? else {
            return Err("device: the head is not dense".to_string());
        };
        let mut part = d.zeros(cotangent.rows(), self.widths[self.head.hidden]).map_err(error)?;
        let op = if self.head.transposed { Op::T } else { Op::N };
        d.gemm(&mut part, 1.0, cotangent, Op::N, a, op, 0.0, arithmetic).map_err(error)?;
        d.set_rows(g_hidden, start, &part).map_err(error)
    }

    fn tile_rows(&self) -> usize {
        (TILE_BYTES / (8 * self.head.classes.max(1))).max(1)
    }

    fn flags(&self, scored: Option<&[bool]>, start: usize, n: usize) -> Result<Option<Indices>, String> {
        scored
            .map(|s| self.device.upload_indices(&s[start..start + n].iter().map(|b| u32::from(*b)).collect::<Vec<_>>()).map_err(error))
            .transpose()
    }

    /// Per row, `KL(softmax(target) ‖ softmax(logits))` (zero on rows `scored` leaves out), and
    /// its cotangent pulled back to the hidden node.
    pub fn kl(&self, trace: &DeviceTrace, target: &Tensor, scored: Option<&[bool]>) -> Result<(Array1<f64>, Tensor), String> {
        let (values, gradient) = self.kl_impl(trace, target, scored, true)?;
        Ok((values, gradient.ok_or("device: missing KL gradient")?))
    }

    /// Per-row KL for candidate acceptance, without a vocabulary cotangent or head pullback.
    pub fn score_only(&self, trace: &DeviceTrace, target: &Tensor, scored: Option<&[bool]>) -> Result<Array1<f64>, String> {
        self.kl_impl(trace, target, scored, false).map(|(values, _)| values)
    }

    fn kl_impl(&self, trace: &DeviceTrace, target: &Tensor, scored: Option<&[bool]>, gradient: bool) -> Result<(Array1<f64>, Option<Tensor>), String> {
        let d = &self.device;
        let hidden = trace.value(self.head.hidden)?;
        if target.dim() != (trace.rows, self.head.classes) {
            return Err(format!("device: a {:?} target for {} rows of {} classes", target.dim(), trace.rows, self.head.classes));
        }
        if scored.is_some_and(|s| s.len() != trace.rows) {
            return Err("device: scored flags do not match trace rows".to_string());
        }
        let mut kl = Vec::with_capacity(trace.rows);
        let mut g = gradient.then(|| d.zeros(trace.rows, hidden.cols()).map_err(error)).transpose()?;
        let tile = self.tile_rows();
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let mut logits = self.logits_tile(hidden, start, n, Arithmetic::F64)?;
            let t = d.rows_of(target, start, n).map_err(error)?;
            let flags = self.flags(scored, start, n)?;
            if let Some(g) = &mut g {
                kl.extend(d.kl_rows(&t, &mut logits, flags.as_ref()).map_err(error)?);
                self.pull_tile(g, start, &logits, Arithmetic::F64)?;
            } else {
                kl.extend(d.kl_score_rows(&t, &mut logits, flags.as_ref()).map_err(error)?);
            }
        }
        Ok((Array1::from(kl), g))
    }

    /// The cotangent of `−log q_y` pulled back to the hidden node, `y` drawn per row from the
    /// logits' softmax by `uniforms[r]` (rows `scored` leaves out stay zero).
    pub fn sampled(&self, trace: &DeviceTrace, uniforms: &[f64], scored: Option<&[bool]>, arithmetic: Arithmetic) -> Result<Tensor, String> {
        let d = &self.device;
        let hidden = trace.value(self.head.hidden)?;
        let mut g = d.zeros(trace.rows, hidden.cols()).map_err(error)?;
        let tile = self.tile_rows();
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let mut logits = self.logits_tile(hidden, start, n, Arithmetic::F64)?;
            let u = d.upload_vec(n, 1, uniforms[start..start + n].to_vec()).map_err(error)?;
            d.sampled_cotangent(&mut logits, &u, self.flags(scored, start, n)?.as_ref()).map_err(error)?;
            self.pull_tile(&mut g, start, &logits, arithmetic)?;
        }
        Ok(g)
    }

    /// `Σ_rows Σ_c q_c (t_c − Σ_j q_j t_j)²`: the output Fisher's quadratic form on the logits'
    /// tangent `t = t_h · ∂logits/∂h` from the hidden node's tangent (rows `scored` leaves out
    /// add nothing).
    pub fn quadratic(&self, trace: &DeviceTrace, tangent: &Tensor, scored: Option<&[bool]>, arithmetic: Arithmetic) -> Result<f64, String> {
        let hidden = trace.value(self.head.hidden)?;
        let tile = self.tile_rows();
        let mut total = 0.0;
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let logits = self.logits_tile(hidden, start, n, Arithmetic::F64)?;
            let t = self.logits_tile(tangent, start, n, arithmetic)?;
            let per_row = self.device.softmax_quadratic(&logits, &t).map_err(error)?;
            total += per_row.iter().enumerate().filter(|(r, _)| scored.is_none_or(|s| s[start + r])).map(|(_, v)| v).sum::<f64>();
        }
        Ok(total)
    }

    /// The logits of rows `start..start + n`, on the host.
    pub fn logits(&self, trace: &DeviceTrace, start: usize, n: usize) -> Result<Array2<f64>, String> {
        let tile = self.logits_tile(trace.value(self.head.hidden)?, start, n, Arithmetic::F64)?;
        self.device.download(&tile).map_err(error)
    }

    /// Reverse pass (`derivatives::vjp`'s rules) from the hidden node's cotangent `seed`: the
    /// cotangents of the nodes in `keep` (those any cotangent reaches), each node's dropped once
    /// its own rule has run. Products run in `arithmetic`.
    pub fn vjp(&self, trace: &DeviceTrace, seed: Tensor, keep: &[usize], arithmetic: Arithmetic) -> Result<BTreeMap<usize, Tensor>, String> {
        let d = &self.device;
        let mut g: Vec<Option<Tensor>> = (0..self.steps.len()).map(|_| None).collect();
        g[self.head.hidden] = Some(seed);
        let mut kept = BTreeMap::new();
        // Adds `term` into node `n`'s cotangent.
        let add = |g: &mut Vec<Option<Tensor>>, n: usize, term: Tensor| -> Result<(), String> {
            match g[n].as_mut() {
                Some(existing) => d.axpy(existing, 1.0, &term).map_err(error),
                None => {
                    g[n] = Some(term);
                    Ok(())
                }
            }
        };
        let slot = |g: &mut Vec<Option<Tensor>>, n: usize, rows: usize| -> Result<(), String> {
            if g[n].is_none() {
                g[n] = Some(d.zeros(rows, self.widths[n]).map_err(error)?);
            }
            Ok(())
        };
        for index in (0..=self.head.hidden).rev() {
            let Some(cot) = g[index].take() else { continue };
            match &self.steps[index] {
                Step::Head | Step::Feature { .. } | Step::Raw { .. } | Step::Constant { .. } => {}
                Step::Affine { terms, .. } => {
                    for (argument, operator) in terms {
                        if matches!(self.steps[*argument], Step::Feature { .. }) {
                            continue;
                        }
                        slot(&mut g, *argument, trace.rows)?;
                        let target = g[*argument].as_mut().ok_or("device: cotangent slot")?;
                        match self.held(*operator, Role::Product)? {
                            Held::Identity => d.axpy(target, 1.0, &cot).map_err(error)?,
                            Held::Diagonal(diag) => d.scale_columns(target, &cot, diag, true).map_err(error)?,
                            Held::Dense(a) => d.gemm(target, 1.0, &cot, Op::N, a, Op::N, 1.0, arithmetic).map_err(error)?,
                            Held::LowRank(left, right) => {
                                // g (L R) = (g L) R.
                                let mut middle = d.zeros(cot.rows(), left.cols()).map_err(error)?;
                                d.gemm(&mut middle, 1.0, &cot, Op::N, left, Op::N, 0.0, arithmetic).map_err(error)?;
                                d.gemm(target, 1.0, &middle, Op::N, right, Op::N, 1.0, arithmetic).map_err(error)?;
                            }
                            Held::Table(_) | Held::Column(_) => return Err("device: an operator held in the wrong role".to_string()),
                        }
                    }
                }
                Step::Transposed { input, operator } => {
                    let Held::Dense(a) = self.held(*operator, Role::Product)? else {
                        return Err("device: a transposed read of a non-dense operator".to_string());
                    };
                    slot(&mut g, *input, trace.rows)?;
                    let target = g[*input].as_mut().ok_or("device: cotangent slot")?;
                    d.gemm(target, 1.0, &cot, Op::N, a, Op::T, 1.0, arithmetic).map_err(error)?;
                }
                Step::Pointwise { input, codes } => {
                    let term = d.law_slopes(&cot, trace.value(*input)?, codes, gelu_tanh_constant()).map_err(error)?;
                    add(&mut g, *input, term)?;
                }
                Step::Hadamard { left, right } => {
                    let mut gl = d.zeros(trace.rows, self.widths[*left]).map_err(error)?;
                    d.hadamard(&mut gl, &cot, trace.value(*right)?, false).map_err(error)?;
                    let mut gr = d.zeros(trace.rows, self.widths[*right]).map_err(error)?;
                    d.hadamard(&mut gr, &cot, trace.value(*left)?, false).map_err(error)?;
                    add(&mut g, *left, gl)?;
                    add(&mut g, *right, gr)?;
                }
                Step::RmsNorm { input, epsilon } => {
                    let term = d.rms_norm_backward(trace.value(*input)?, &cot, *epsilon).map_err(error)?;
                    add(&mut g, *input, term)?;
                }
                Step::Attend { query, key, value, scale, rotary, causal } => {
                    let (gq, gk, gv) = self.attend_cotangent(trace, (*query, *key, *value), &cot, *scale, *rotary, *causal, arithmetic)?;
                    add(&mut g, *query, gq)?;
                    add(&mut g, *key, gk)?;
                    add(&mut g, *value, gv)?;
                }
            }
            if keep.contains(&index) {
                kept.insert(index, cot);
            }
        }
        Ok(kept)
    }

    fn attend_cotangent(
        &self,
        trace: &DeviceTrace,
        (query, key, value): (usize, usize, usize),
        cot: &Tensor,
        scale: f64,
        rotary: Option<Rotary>,
        causal: bool,
        arithmetic: Arithmetic,
    ) -> Result<(Tensor, Tensor, Tensor), String> {
        let d = &self.device;
        let blocks = trace.blocks;
        let (q, k) = self.rotated(trace, query, key, rotary)?;
        let alpha = self.attention(trace, &q, &k, scale, causal)?;
        let v = trace.value(value)?;
        let length = trace.rows / blocks;
        let mut dalpha = d.zeros(trace.rows, length).map_err(error)?;
        d.gemm_batched(blocks, &mut dalpha, 1.0, cot, Op::N, v, Op::T, 0.0, arithmetic).map_err(error)?;
        let mut gv = d.zeros(trace.rows, v.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut gv, 1.0, &alpha, Op::T, cot, Op::N, 0.0, arithmetic).map_err(error)?;
        let ds = d.softmax_backward(&alpha, &dalpha).map_err(error)?;
        drop((alpha, dalpha));
        let mut gq = d.zeros(trace.rows, q.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut gq, scale, &ds, Op::N, &k, Op::N, 0.0, arithmetic).map_err(error)?;
        let mut gk = d.zeros(trace.rows, k.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut gk, scale, &ds, Op::T, &q, Op::N, 0.0, arithmetic).map_err(error)?;
        let (gq, gk) = match rotary {
            None => (gq, gk),
            Some(r) => {
                let (cos, sin) = rotations_of(trace, r)?;
                (d.rotate(&gq, cos, sin, r.half_split, true).map_err(error)?, d.rotate(&gk, cos, sin, r.half_split, true).map_err(error)?)
            }
        };
        Ok((gq, gk, gv))
    }

    /// Forward tangent (`derivatives::jvp`'s rules) of the hidden node when each operator in
    /// `tangents` moves along its entry (a host matrix of the operator's shape); `None` when no
    /// tangent reaches it. Products run in `arithmetic`.
    pub fn jvp(&self, trace: &DeviceTrace, tangents: &BTreeMap<usize, Array2<f64>>, arithmetic: Arithmetic) -> Result<Option<Tensor>, String> {
        let d = &self.device;
        let rows = trace.rows;
        let upload = |m: &Array2<f64>| d.upload(m.view()).map_err(error);
        let mut dv: Vec<Option<Tensor>> = (0..self.steps.len()).map(|_| None).collect();
        for index in 0..=self.head.hidden {
            let width = self.widths[index];
            let t = match &self.steps[index] {
                Step::Head | Step::Feature { .. } | Step::Raw { .. } => None,
                Step::Constant { operator } => match tangents.get(operator) {
                    Some(dc) => Some(d.broadcast_rows(&upload(&dc.t().to_owned())?, rows).map_err(error)?),
                    None => None,
                },
                Step::Affine { terms, bias } => {
                    let mut out: Option<Tensor> = None;
                    for (argument, operator) in terms {
                        if let Some(dx) = dv[*argument].as_ref() {
                            self.add_product(ensure(d, &mut out, rows, width)?, dx, *operator, false, arithmetic)?;
                        }
                        if let Some(da) = tangents.get(operator) {
                            let o = ensure(d, &mut out, rows, width)?;
                            let diagonal_row = self.operators.get(&(*operator, Role::Product)).is_some_and(|held| {
                                matches!(held.source.body, OperatorBody::Diagonal { .. }) && da.dim() == (1, width)
                            });
                            if let Step::Feature { slot } = &self.steps[*argument] {
                                let ids = trace.ids.get(slot).ok_or("device: feature ids missing")?;
                                let gathered = d.gather_rows(&upload(&da.t().to_owned())?, ids).map_err(error)?;
                                d.axpy(o, 1.0, &gathered).map_err(error)?;
                            } else if diagonal_row {
                                // A diagonal's tangent given as its diagonal: a column scale.
                                d.scale_columns(o, trace.value(*argument)?, &upload(da)?, true).map_err(error)?;
                            } else {
                                d.gemm(o, 1.0, trace.value(*argument)?, Op::N, &upload(da)?, Op::T, 1.0, arithmetic).map_err(error)?;
                            }
                        }
                    }
                    if let Some(db) = bias.and_then(|b| tangents.get(&b)) {
                        d.add_row(ensure(d, &mut out, rows, width)?, 1.0, &upload(&db.t().to_owned())?).map_err(error)?;
                    }
                    out
                }
                Step::Transposed { input, operator } => {
                    let mut out: Option<Tensor> = None;
                    if let Some(dx) = dv[*input].as_ref() {
                        let mut o = d.zeros(rows, width).map_err(error)?;
                        self.add_product(&mut o, dx, *operator, true, arithmetic)?;
                        out = Some(o);
                    }
                    if let Some(da) = tangents.get(operator) {
                        d.gemm(ensure(d, &mut out, rows, width)?, 1.0, trace.value(*input)?, Op::N, &upload(da)?, Op::N, 1.0, arithmetic)
                            .map_err(error)?;
                    }
                    out
                }
                Step::Pointwise { input, codes } => match dv[*input].as_ref() {
                    Some(dx) => Some(d.law_slopes(dx, trace.value(*input)?, codes, gelu_tanh_constant()).map_err(error)?),
                    None => None,
                },
                Step::Hadamard { left, right } => match (dv[*left].as_ref(), dv[*right].as_ref()) {
                    (None, None) => None,
                    (dl, dr) => {
                        let mut out = d.zeros(rows, width).map_err(error)?;
                        if let Some(dl) = dl {
                            d.hadamard(&mut out, dl, trace.value(*right)?, true).map_err(error)?;
                        }
                        if let Some(dr) = dr {
                            d.hadamard(&mut out, trace.value(*left)?, dr, true).map_err(error)?;
                        }
                        Some(out)
                    }
                },
                Step::RmsNorm { input, epsilon } => match dv[*input].as_ref() {
                    Some(dx) => Some(d.rms_norm_tangent(trace.value(*input)?, dx, *epsilon).map_err(error)?),
                    None => None,
                },
                Step::Attend { query, key, value, scale, rotary, causal } => {
                    if dv[*query].is_none() && dv[*key].is_none() && dv[*value].is_none() {
                        None
                    } else {
                        Some(self.attend_tangent(trace, (*query, *key, *value), &dv, *scale, *rotary, *causal, arithmetic)?)
                    }
                }
            };
            dv[index] = t;
        }
        Ok(dv[self.head.hidden].take())
    }

    fn attend_tangent(
        &self,
        trace: &DeviceTrace,
        (query, key, value): (usize, usize, usize),
        dv: &[Option<Tensor>],
        scale: f64,
        rotary: Option<Rotary>,
        causal: bool,
        arithmetic: Arithmetic,
    ) -> Result<Tensor, String> {
        let d = &self.device;
        let blocks = trace.blocks;
        let (q, k) = self.rotated(trace, query, key, rotary)?;
        let alpha = self.attention(trace, &q, &k, scale, causal)?;
        let turn = |t: &Tensor| -> Result<Tensor, String> {
            match rotary {
                None => d.copy(t).map_err(error),
                Some(r) => {
                    let (cos, sin) = rotations_of(trace, r)?;
                    d.rotate(t, cos, sin, r.half_split, false).map_err(error)
                }
            }
        };
        let length = trace.rows / blocks;
        let mut ds = d.zeros(trace.rows, length).map_err(error)?;
        if let Some(dq) = dv[query].as_ref() {
            d.gemm_batched(blocks, &mut ds, scale, &turn(dq)?, Op::N, &k, Op::T, 1.0, arithmetic).map_err(error)?;
        }
        if let Some(dk) = dv[key].as_ref() {
            d.gemm_batched(blocks, &mut ds, scale, &q, Op::N, &turn(dk)?, Op::T, 1.0, arithmetic).map_err(error)?;
        }
        let dalpha = d.softmax_backward(&alpha, &ds).map_err(error)?;
        let v = trace.value(value)?;
        let mut out = d.zeros(trace.rows, v.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut out, 1.0, &dalpha, Op::N, v, Op::N, 0.0, arithmetic).map_err(error)?;
        if let Some(dvv) = dv[value].as_ref() {
            d.gemm_batched(blocks, &mut out, 1.0, &alpha, Op::N, dvv, Op::N, 1.0, arithmetic).map_err(error)?;
        }
        Ok(out)
    }
}
