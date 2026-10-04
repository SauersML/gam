//! The explanation as one artifact (#2951): the object the search accepts is the object that is
//! serialized, decoded, executed, intervened on and reported.
//!
//! # The object
//!
//! An [`Artifact`] is an executable [`OperatorProgram`] `P` over the model's declarations, in which
//! the replaced native blocks are [`Node::Call`]s of rules (a rule body is stored once however often
//! it is called), together with what ties it to the native model `M`:
//!
//! * **Blocks** ([`Binding`]): each replaced block's native parent state (the native nodes it reads)
//!   and native output (the node it writes), and the nodes of `P` that hold them.
//! * **Places**: every native node `P` still holds, as `(native node, node of P)`. A native
//!   intervention acts on `P` at the place it names; a native node `P` does not hold (one inside a
//!   replaced block) has no place, and `P` runs clean there.
//! * **Exceptions** ([`Exception`]): input-specific values `P` cannot compute, each added at one
//!   coordinate of one node on the rows whose causal context it names.
//! * **Derived operators** ([`Derived`]): operators of `P` its decoder computes from operators
//!   decoded before them rather than reads, `λ · law(sources) + residual rows` (an attention head
//!   by a legacy template or an explicit matrix rule): the graph is the native one, so every
//!   native place inside it stays a place.
//!   The message holds a derived operator with no reals; its reals are recomputed, in an order in
//!   which every source precedes what reads it, whenever a source changes and when decoding.
//!
//! Legacy Copy/Match tags select formulas supplied by the decoder: their scores are
//! conditional on that fixed template language. New matrix-rule artifacts transmit
//! complete typed arithmetic bodies once, with explicit source bindings per call.
//! Neither representation alone constitutes automatic rule discovery.
//!
//! Learned operator and coefficient literals are projected to 32-bit floats by
//! [`Artifact::f32_literals`]; native RMSNorm epsilon values remain exact architecture
//! constants. The fixed-32 objective C32 charges every independently transmitted numeric literal
//! once at 32 bits, including architecture epsilon values, rotary bases, freely
//! specified inverse-square-root arguments and exception values. This price
//! is separate from exact wire length; changing a price never rounds execution. A
//! derived operator's computed reals are not literals; its scale and residual rows are.
//!
//! # The message
//!
//! [`Artifact::encode`] writes one self-delimiting message through `codec`'s integer codes:
//!
//! 1. the native node count `N_M + 1` in the prefix code;
//! 2. the program's message length `+ 1` in the prefix code, then the program's message
//!    ([`OperatorProgram::encode`]);
//! 3. the blocks: count `+ 1`, then per block its name (byte count `+ 1`, each byte in 8 bits), its
//!    read count `+ 1`, each native read as a fixed index into `N_M`, the native write likewise, each
//!    read of `P` and the write of `P` as fixed indices into `P`'s nodes;
//! 4. the places: count `+ 1`, each as two fixed indices (native node, node of `P`);
//! 5. the exceptions: count `+ 1`, each its context length `+ 1` and tokens (`+ 1`) in the prefix
//!    code, its node and column as fixed indices, its value as the 32 bits of its float;
//! 6. the derived operators: count `+ 1`, each its operator as a fixed index into `P`'s operators, its
//!    law as a fixed index into the laws, the law's source operators as fixed indices and its
//!    integers (`+ 1`) in the prefix code, the scale's 32 bits, and its residual rows: count `+ 1`,
//!    each row as a fixed index into the operator's rows and its values' 32 bits each.
//! 7. only in grammar version 3, explicit native control bindings: positive count,
//!    law tag, native source/write indices, explanatory write index, and source width.
//!    Empty bindings keep the previous byte format and intervention semantics.
//!
//! [`Artifact::decode`] reads it back given only the declarations; nothing of the native model, of a
//! fit or of a discovery is read. The names are labels: they are sent so a decoded artifact reports
//! itself, and no length charges them.
//!
//! Explicit-body artifacts use grammar version 2: an impossible legacy zero-native-node
//! marker and version precede the message, and a shared body pool precedes its derivations.
//! Their byte envelope starts with u64::MAX, version, then bit length. Legacy messages
//! and their original length-only byte envelopes remain unchanged.
//!
//! # A block on its native parent state
//!
//! [`Artifact::local_program`] grafts a block of `P` onto `M`: `M`'s program, followed by the nodes of
//! `P` that compute the block's write from its reads, reading `M`'s read nodes instead of `P`'s, and
//! a difference node `write(P) − write(M)`. Executed on any input, it is the block's own error on the
//! native parent state, in the native interface, and being itself an operator program it executes
//! with forward-error bands and differentiates (`derivatives::vjp`) like any other.

use super::codec::{
    BitReader, BitString, CodecError, decode_fixed_index, decode_prefix_integer, encode_fixed_index, encode_prefix_integer, fixed_index_len_bits,
    prefix_integer_len_bits,
};
use super::operator_program::{
    Coefficient, Declarations, FamilyInputs, Interface, Node, Operator, OperatorBody, OperatorProgram, ProgramError, Provenance, Rule, SlotValues, Trace,
    exact_precision, remap_node,
};
use super::precision::DecodableArtifact;
use super::matrix_rule::{MatrixRule, Type as MatrixType, Value as MatrixValue};
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A replaced native block: where its parent state and output sit in the native model and in `P`.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    pub name: String,
    /// The native nodes whose values the block reads.
    pub native_reads: Vec<usize>,
    /// The native node the block writes.
    pub native_write: usize,
    /// `P`'s nodes holding the same values, in order.
    pub reads: Vec<usize>,
    pub write: usize,
}

/// An input-specific value the program cannot compute: `value` added at `(node, column)` on every
/// row whose causal context (the token-slot values of the rows of its unit up to and including it,
/// slot-major per row) is `context`.
#[derive(Clone, Debug, PartialEq)]
pub struct Exception {
    pub context: Vec<u32>,
    pub node: usize,
    pub column: usize,
    pub value: f32,
}

/// A call argument of a replaced block: a native node `P` holds, or a constant the replacement
/// supplies (a per-instance literal of a shared rule).
#[derive(Clone, Debug)]
pub enum Argument {
    Native(usize),
    Constant(Operator),
}

/// The rule a replacement calls: a new one, or one `P` already stores (a shared body).
#[derive(Clone, Debug)]
pub enum Callee {
    New(Rule),
    Existing(usize),
}

/// What computes a derived operator from operators decoded before it (`rules`).
#[derive(Clone, Debug, PartialEq)]
pub enum OperatorLaw {
    /// An explicit shared matrix-valued body; no named composite is supplied by the decoder.
    Expression { body: Arc<MatrixRule>, sources: Vec<usize> },
    /// An output head `diag(g / g_f) V⁺` (`rules::copy_prediction`): `value` the head's value
    /// operator (`width × d`), `gain` its layer's norm gain and `final_gain` the final norm's (diagonal
    /// operators).
    Copy { value: usize, gain: usize, final_gain: usize },
    /// A query head's content rows `R = rules::content_rows(width, first)` as
    /// `rules::match_prediction(rules::match_reading(K, O_s, V_s, g, g_s, R), g, directions)`, its
    /// other rows zero: `key` the head's key operator (`width × d`), `source_output` (`d × width`) and
    /// `source_value` an earlier head's output and value operators, `gain` and `source_gain` the
    /// norm gains the two layers read through.
    Match { key: usize, source_output: usize, source_value: usize, gain: usize, source_gain: usize, first: usize, directions: Option<usize> },
}

const LAWS: usize = 2; // Legacy template language: never change this alphabet.
const MATRIX_LAWS: usize = 3;
const MATRIX_ARTIFACT_VERSION: u64 = 2;
const CONTROL_ARTIFACT_VERSION: u64 = 3;
const VERSIONED_ENVELOPE: u64 = u64::MAX;

impl OperatorLaw {
    fn index(&self) -> usize {
        match self {
            Self::Expression { .. } => 2,
            Self::Copy { .. } => 0,
            Self::Match { .. } => 1,
        }
    }

    /// The operators it reads.
    pub fn sources(&self) -> Vec<usize> {
        match self {
            Self::Expression { sources, .. } => sources.clone(),
            Self::Copy { value, gain, final_gain } => vec![*value, *gain, *final_gain],
            Self::Match { key, source_output, source_value, gain, source_gain, .. } => vec![*key, *source_output, *source_value, *gain, *source_gain],
        }
    }

    /// Its integers.
    fn integers(&self) -> Vec<u64> {
        match self {
            Self::Expression { .. } => Vec::new(),
            Self::Copy { .. } => Vec::new(),
            Self::Match { first, directions, .. } => vec![*first as u64, directions.map_or(0, |k| k as u64 + 1)],
        }
    }

    fn with_sources(&self, sources: &[usize]) -> Self {
        match self {
            Self::Expression { body, .. } => Self::Expression { body: Arc::clone(body), sources: sources.to_vec() },
            Self::Copy { .. } => Self::Copy { value: sources[0], gain: sources[1], final_gain: sources[2] },
            Self::Match { first, directions, .. } => Self::Match {
                key: sources[0],
                source_output: sources[1],
                source_value: sources[2],
                gain: sources[3],
                source_gain: sources[4],
                first: *first,
                directions: *directions,
            },
        }
    }

    fn of(law: usize, sources: &[usize], integers: &[u64]) -> Result<Self, String> {
        let template = match law {
            0 => Self::Copy { value: 0, gain: 0, final_gain: 0 },
            1 => Self::Match {
                key: 0,
                source_output: 0,
                source_value: 0,
                gain: 0,
                source_gain: 0,
                first: integers.first().copied().ok_or("a match law without its first plane")? as usize,
                directions: integers.get(1).copied().ok_or("a match law without its directions")?.checked_sub(1).map(|k| k as usize),
            },
            other => return Err(format!("no operator law {other}")),
        };
        Ok(template.with_sources(sources))
    }

    fn arity(law: usize) -> (usize, usize) {
        if law == 0 { (3, 0) } else { (5, 2) }
    }
}

/// An operator of `P` its decoder computes (module note): `scale · law(sources)`, plus each residual
/// row's literal values added to that row.
#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    pub operator: usize,
    pub law: OperatorLaw,
    pub scale: f32,
    pub residual: Vec<(usize, Vec<f32>)>,
}

impl Derived {
    /// Its literals: the scale and the residual rows' values.
    pub fn literals(&self) -> u64 {
        1 + self.residual.iter().map(|(_, row)| row.len() as u64).sum::<u64>()
    }
}

/// The diagonal of operator `op` of `program`.
fn diagonal_of(program: &OperatorProgram, op: usize) -> Result<Array1<f64>, String> {
    let operator = program.operators.get(op).ok_or_else(|| format!("no operator {op}"))?;
    operator.diagonal().ok_or_else(|| format!("operator {} is not diagonal", operator.name))
}

/// The reals `derived` computes in `program`.
fn derived_values(program: &OperatorProgram, derived: &Derived) -> Result<Array2<f64>, String> {
    let matrix = |op: usize| -> Result<Array2<f64>, String> { Ok(program.operators.get(op).ok_or_else(|| format!("no operator {op}"))?.matrix()) };
    let target = program.operators.get(derived.operator).ok_or_else(|| format!("no derived operator {}", derived.operator))?;
    let mut values = match &derived.law {
        OperatorLaw::Expression { body, sources } => {
            if sources.len() != body.inputs.len() { return Err("a matrix-rule call has the wrong source arity".into()); }
            let arguments = sources.iter().zip(&body.inputs).map(|(source, ty)| {
                match ty {
                    MatrixType::Matrix { .. } => matrix(*source).map(MatrixValue::Matrix),
                    MatrixType::Vector { .. } => diagonal_of(program, *source).map(MatrixValue::Vector),
                }
            }).collect::<Result<Vec<_>, String>>()?;
            match body.evaluate(&arguments)? {
                MatrixValue::Matrix(value) => value,
                MatrixValue::Vector(_) => return Err("a derived operator matrix rule returned a vector".into()),
            }
        },
        OperatorLaw::Copy { value, gain, final_gain } => {
            super::rules::copy_prediction(&matrix(*value)?, &diagonal_of(program, *gain)?, &diagonal_of(program, *final_gain)?)?
        }
        OperatorLaw::Match { key, source_output, source_value, gain, source_gain, first, directions } => {
            let key = matrix(*key)?;
            let rows = super::rules::content_rows(key.nrows(), *first);
            let g = diagonal_of(program, *gain)?;
            let reading =
                super::rules::match_reading(&key, &matrix(*source_output)?, &matrix(*source_value)?, &g, &diagonal_of(program, *source_gain)?, &rows);
            let prediction = super::rules::match_prediction(&reading, &g, *directions)?;
            let mut out = Array2::<f64>::zeros(key.dim());
            for (i, row) in rows.iter().enumerate() {
                out.row_mut(*row).assign(&prediction.row(i));
            }
            out
        }
    };
    values.mapv_inplace(|v| v * f64::from(derived.scale));
    if values.dim() != (target.rows.width(), target.cols.width()) {
        return Err(format!("{}: a law of {:?} for a {}×{} operator", target.name, values.dim(), target.rows.width(), target.cols.width()));
    }
    for (row, entries) in &derived.residual {
        if *row >= values.nrows() || entries.len() != values.ncols() {
            return Err(format!("{}: a residual row {row} of {} values", target.name, entries.len()));
        }
        for (v, e) in values.row_mut(*row).iter_mut().zip(entries) {
            *v += f64::from(*e);
        }
    }
    Ok(values)
}

/// `derived` with every source before what reads it, or refused for a cycle.
fn ordered(derived: Vec<Derived>) -> Result<Vec<Derived>, String> {
    let mut left = derived;
    let mut out: Vec<Derived> = Vec::with_capacity(left.len());
    while !left.is_empty() {
        let pending: std::collections::BTreeSet<usize> = left.iter().map(|d| d.operator).collect();
        let ready =
            left.iter().position(|d| d.law.sources().iter().all(|s| !pending.contains(s))).ok_or("derived operators that derive from each other")?;
        out.push(left.remove(ready));
    }
    Ok(out)
}

/// `program` with every derived operator's reals computed, in order.
fn compute_derived(program: &mut OperatorProgram, derived: &[Derived]) -> Result<(), String> {
    for d in derived {
        let values = derived_values(program, d)?;
        let old = &program.operators[d.operator];
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let op = Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone())
            .map_err(|e| e.to_string())?;
        program.operators[d.operator] = Arc::new(op);
    }
    Ok(())
}

/// The explanation (module note).
#[derive(Clone, Debug, PartialEq)]
pub struct Artifact {
    pub program: OperatorProgram,
    /// The native program's node count: the alphabet of native node references.
    pub native_nodes: usize,
    pub blocks: Vec<Binding>,
    /// `(native node, node of P)`, ascending in the native node.
    pub places: Vec<(usize, usize)>,
    pub exceptions: Vec<Exception>,
    /// In an order in which every source precedes what reads it.
    pub derived: Vec<Derived>,
    /// Explicitly priced native intervention-response laws; empty retains legacy semantics.
    pub controls: Vec<crate::native_control::UniformScaleBinding>,
}

/// Old to new indices of the kept entries (`usize::MAX` for a removed one).
fn compaction(keep: &[bool]) -> Vec<usize> {
    let mut map = vec![usize::MAX; keep.len()];
    let mut next = 0;
    for (index, kept) in keep.iter().enumerate() {
        if *kept {
            map[index] = next;
            next += 1;
        }
    }
    map
}

/// `program` compacted as [`OperatorProgram::prune`] compacts it (the nodes its output reads, the
/// rules those call, the operators and bases they use), also keeping the operators `keep`; the
/// old-to-new node and operator maps (`usize::MAX` for what left).
fn compact(program: &mut OperatorProgram, keep: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut live = vec![false; program.nodes.len()];
    live[program.output] = true;
    for index in (0..program.nodes.len()).rev() {
        if live[index] {
            for argument in program.nodes[index].arguments() {
                live[argument] = true;
            }
        }
    }
    let mut used_rules = vec![false; program.rules.len()];
    let mut pending: Vec<&Node> = program.nodes.iter().zip(&live).filter(|(_, l)| **l).map(|(n, _)| n).collect();
    while let Some(node) = pending.pop() {
        if let Node::Call { rule, .. } = node
            && !used_rules[*rule]
        {
            used_rules[*rule] = true;
            pending.extend(program.rules[*rule].nodes.iter());
        }
    }
    let mut used_ops = vec![false; program.operators.len()];
    let mut used_bases = vec![false; program.bases.len()];
    for &op in keep {
        used_ops[op] = true;
    }
    let live_nodes = program.nodes.iter().zip(&live).filter(|(_, l)| **l).map(|(n, _)| n);
    let rule_nodes = program.rules.iter().zip(&used_rules).filter(|(_, u)| **u).flat_map(|(rule, _)| rule.nodes.iter());
    for node in live_nodes.chain(rule_nodes) {
        for op in node.operators() {
            used_ops[op] = true;
        }
        if let Node::Feature { basis, .. } | Node::Readout { basis, .. } = node {
            used_bases[*basis] = true;
        }
    }
    let (node_map, op_map, basis_map, rule_map) = (compaction(&live), compaction(&used_ops), compaction(&used_bases), compaction(&used_rules));
    let mut rules: Vec<Rule> = program.rules.iter().zip(&used_rules).filter(|(_, u)| **u).map(|(rule, _)| rule.clone()).collect();
    for rule in &mut rules {
        let body = identity(rule.nodes.len());
        for node in &mut rule.nodes {
            remap_node(node, &body, &op_map, &basis_map, &rule_map);
        }
    }
    let mut nodes: Vec<Node> = program.nodes.iter().zip(&live).filter(|(_, l)| **l).map(|(n, _)| n.clone()).collect();
    for node in &mut nodes {
        remap_node(node, &node_map, &op_map, &basis_map, &rule_map);
    }
    program.operators = program.operators.iter().zip(&used_ops).filter(|(_, u)| **u).map(|(op, _)| op.clone()).collect();
    program.bases = program.bases.iter().zip(&used_bases).filter(|(_, u)| **u).map(|(b, _)| b.clone()).collect();
    program.output = node_map[program.output];
    program.nodes = nodes;
    program.rules = rules;
    (node_map, op_map)
}

fn identity(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// The f32 nearest each real, held on the finest dyadic lattice that holds them all.
fn f32_reals(values: &mut [f64]) {
    for value in values.iter_mut() {
        *value = f64::from(*value as f32);
    }
}

/// `op` with every real rounded to its nearest 32-bit float.
pub fn f32_operator(op: &Operator) -> Result<Operator, String> {
    let error = |e: ProgramError| format!("{}: {e}", op.name);
    Ok(match &op.body {
        OperatorBody::Identity => op.clone(),
        OperatorBody::Dense { values, present, .. } => {
            let mut values = values.clone();
            f32_reals(values.as_slice_mut().ok_or("a non-contiguous operator")?);
            let precision = exact_precision(values.iter().copied()).map_err(error)?;
            Operator::blocks(op.name.clone(), op.rows.clone(), op.cols.clone(), values, present.clone(), precision, op.provenance.clone())
                .map_err(error)?
        }
        OperatorBody::LowRank { left, right, .. } => {
            let (mut left, mut right) = (left.clone(), right.clone());
            f32_reals(left.as_slice_mut().ok_or("a non-contiguous factor")?);
            f32_reals(right.as_slice_mut().ok_or("a non-contiguous factor")?);
            let precision = exact_precision(left.iter().chain(right.iter()).copied()).map_err(error)?;
            Operator::low_rank(op.name.clone(), op.rows.clone(), op.cols.clone(), left, right, precision, op.provenance.clone()).map_err(error)?
        }
        OperatorBody::Diagonal { values, .. } => {
            let mut values = values.clone();
            f32_reals(values.as_slice_mut().ok_or("a non-contiguous diagonal")?);
            let precision = exact_precision(values.iter().copied()).map_err(error)?;
            Operator::diag(op.name.clone(), op.rows.clone(), values, precision, op.provenance.clone()).map_err(error)?
        }
    })
}

/// Whether every real of `op` is a 32-bit float.
pub fn has_f32_reals(op: &Operator) -> bool {
    let f32_exact = |v: &f64| f64::from(*v as f32) == *v;
    match &op.body {
        OperatorBody::Identity => true,
        OperatorBody::Dense { values, .. } => values.iter().all(f32_exact),
        OperatorBody::LowRank { left, right, .. } => left.iter().chain(right.iter()).all(f32_exact),
        OperatorBody::Diagonal { values, .. } => values.iter().all(f32_exact),
    }
}

/// `program` with every rule application replaced by a copy of the rule's body, its parameters
/// bound to the call's arguments: the same function, node for node, with no rules.
pub fn inlined(program: &OperatorProgram) -> Result<OperatorProgram, String> {
    let (operators, bases, rules) = (identity(program.operators.len()), identity(program.bases.len()), identity(program.rules.len()));
    let mut nodes: Vec<Node> = Vec::new();
    // Appends `body`'s nodes with `Param { i }` bound to `arguments[i]`; returns the output's index.
    fn expand(
        program: &OperatorProgram,
        body: &[Node],
        output: usize,
        arguments: &[usize],
        nodes: &mut Vec<Node>,
        maps: (&[usize], &[usize], &[usize]),
    ) -> Result<usize, String> {
        let mut map: Vec<usize> = Vec::with_capacity(body.len());
        for node in body {
            let index = match node {
                Node::Param { index } => {
                    *arguments.get(*index).ok_or_else(|| format!("a rule parameter {index} with {} arguments", arguments.len()))?
                }
                Node::Call { rule, arguments: call } => {
                    let rule = program.rules.get(*rule).ok_or_else(|| format!("no rule {rule}"))?;
                    let bound: Vec<usize> = call.iter().map(|a| map[*a]).collect();
                    expand(program, &rule.nodes, rule.output, &bound, nodes, maps)?
                }
                other => {
                    let mut copy = other.clone();
                    remap_node(&mut copy, &map, maps.0, maps.1, maps.2);
                    nodes.push(copy);
                    nodes.len() - 1
                }
            };
            map.push(index);
        }
        Ok(map[output])
    }
    let output = expand(program, &program.nodes, program.output, &[], &mut nodes, (&operators, &bases, &rules))?;
    Ok(OperatorProgram {
        declarations: program.declarations.clone(),
        bases: program.bases.clone(),
        operators: program.operators.clone(),
        rules: Vec::new(),
        nodes,
        output,
    })
}

/// Per row of `inputs`, its causal context (module note on [`Exception`]).
pub fn contexts(inputs: &FamilyInputs) -> Vec<Vec<u32>> {
    let token_slots: Vec<&Vec<u32>> = inputs
        .slots
        .iter()
        .filter_map(|slot| match slot {
            SlotValues::Tokens(tokens) => Some(tokens),
            SlotValues::Raw(_) => None,
        })
        .collect();
    let row_tokens = |row: usize| token_slots.iter().map(move |tokens| tokens[row]);
    match &inputs.layout {
        None => (0..inputs.rows).map(|row| row_tokens(row).collect()).collect(),
        Some(layout) => {
            // Rows of one sequence, by position.
            let mut by_sequence: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new();
            for row in 0..inputs.rows {
                by_sequence.entry(layout.sequence[row]).or_default().push((layout.position[row], row));
            }
            let mut out = vec![Vec::new(); inputs.rows];
            for rows in by_sequence.values_mut() {
                rows.sort_unstable();
                let mut context = Vec::new();
                for &(_, row) in rows.iter() {
                    context.extend(row_tokens(row));
                    out[row] = context.clone();
                }
            }
            out
        }
    }
}

fn codec(error: CodecError) -> String {
    format!("{error:?}")
}

fn coefficient_f32(coefficient: &Coefficient) -> bool {
    match coefficient {
        Coefficient::Parameter(_) => true,
        Coefficient::Number(value) => value.is_finite() && f64::from(*value as f32) == *value,
        Coefficient::Sum(terms) | Coefficient::Product(terms) => terms.iter().all(coefficient_f32),
    }
}

fn round_coefficient(coefficient: &mut Coefficient) {
    if let Coefficient::Number(value) = coefficient {
        *value = f64::from(*value as f32);
    } else if let Coefficient::Sum(terms) | Coefficient::Product(terms) = coefficient {
        terms.iter_mut().for_each(round_coefficient);
    }
}

impl Artifact {
    /// Add a paid response law for uniform scaling of a wholly omitted native
    /// interface through its exclusive homogeneous linear write. Validation
    /// checks the native graph; execution needs only the serialized binding.
    /// Partial-coordinate actions remain unsupported by this law.
    pub fn with_uniform_scale_control(&self, model: &OperatorProgram, native_source: usize, native_write: usize) -> Result<Self, String> {
        let write = self.place(native_write).ok_or("control boundary is not held")?;
        let width = model.node_interface(native_source).map_err(|e|e.to_string())?.width();
        let mut out = self.clone();
        out.controls.push(crate::native_control::UniformScaleBinding { native_source, native_write, write, width });
        out.controls.sort_by_key(|c| c.native_source);
        crate::native_control::validate(&out, model)?;
        Ok(out)
    }

    /// The native model as its own explanation: every node a place, no block replaced.
    pub fn native(model: &OperatorProgram) -> Result<Self, String> {
        model.interfaces().map_err(|e| e.to_string())?;
        Ok(Self {
            program: model.clone(),
            native_nodes: model.nodes.len(),
            blocks: Vec::new(),
            places: (0..model.nodes.len()).map(|n| (n, n)).collect(),
            exceptions: Vec::new(),
            derived: Vec::new(),
            controls: Vec::new(),
        })
    }

    /// The operators derived operators read or are, which compaction keeps.
    fn derived_operators(&self) -> Vec<usize> {
        self.derived.iter().flat_map(|d| std::iter::once(d.operator).chain(d.law.sources())).collect()
    }

    /// The derived operators with their operators renumbered by `op_map` (old to new).
    fn renumbered_derived(&self, op_map: &[usize]) -> Result<Vec<Derived>, String> {
        self.derived
            .iter()
            .map(|d| {
                let sources: Vec<usize> = d.law.sources().iter().map(|s| op_map[*s]).collect();
                if op_map[d.operator] == usize::MAX || sources.contains(&usize::MAX) {
                    return Err("a derived operator lost its operator or a source".to_string());
                }
                Ok(Derived { operator: op_map[d.operator], law: d.law.with_sources(&sources), ..d.clone() })
            })
            .collect()
    }

    /// This artifact with operator `operator` of `P` derived (module note): `scale · law(sources)`
    /// plus `residual` rows, replacing any derivation it had; every derived operator is then
    /// recomputed in order. The literals are 32-bit floats.
    pub fn derive(&self, operator: usize, law: OperatorLaw, scale: f32, residual: Vec<(usize, Vec<f32>)>) -> Result<Self, String> {
        let count = self.program.operators.len();
        if operator >= count || law.sources().iter().any(|s| *s >= count || *s == operator) {
            return Err(format!("a derivation of operator {operator} from {:?} among {count}", law.sources()));
        }
        let mut derived: Vec<Derived> = self.derived.iter().filter(|d| d.operator != operator).cloned().collect();
        derived.push(Derived { operator, law, scale, residual });
        let derived = ordered(derived)?;
        let mut out = self.clone();
        compute_derived(&mut out.program, &derived)?;
        out.derived = derived;
        Ok(out)
    }

    /// Expand legacy Copy templates into complete arithmetic bodies. This is
    /// desugaring a fixed template baseline, not automatic rule discovery.
    /// The SVD convention and each call's scale/residual are retained. The
    /// generic matrix product may change IEEE zero signs or accumulation order;
    /// fidelity must be measured on the independently decoded new artifact.
    pub fn expand_copy_templates(&self) -> Result<Self, String> {
        use super::matrix_rule::{Node as MatrixNode, PinvConvention};
        let mut out = self.clone();
        for d in &self.derived {
            if let OperatorLaw::Copy { value, gain, final_gain } = d.law {
                let v = self.program.operators.get(value).ok_or("a Copy template has no value operator")?;
                let width = v.cols.width();
                let body = Arc::new(MatrixRule {
                    inputs: vec![MatrixType::Matrix { rows: v.rows.width(), cols: width }, MatrixType::Vector { len: width }, MatrixType::Vector { len: width }],
                    nodes: vec![MatrixNode::Param { index: 0 }, MatrixNode::Param { index: 1 }, MatrixNode::Param { index: 2 }, MatrixNode::Divide { numerator: 1, denominator: 2 }, MatrixNode::Diag { input: 3 }, MatrixNode::Pinv { input: 0, convention: PinvConvention::SvdResolutionBand }, MatrixNode::MatMul { left: 4, right: 5 }],
                    output: 6,
                });
                out = out.derive(d.operator, OperatorLaw::Expression { body, sources: vec![value, gain, final_gain] }, d.scale, d.residual.clone())?;
            }
        }
        Ok(out)
    }

    /// `P` as its message holds it: every derived operator with no reals (its interfaces kept).
    pub fn message_program(&self) -> Result<OperatorProgram, String> {
        let mut program = self.program.clone();
        for d in &self.derived {
            let op = &program.operators[d.operator];
            let (rows, cols) = (op.rows.clone(), op.cols.clone());
            let empty = Array2::from_elem((rows.group_count(), cols.group_count()), false);
            let precision = exact_precision([0.0]).map_err(|e| e.to_string())?;
            let blank = Operator::blocks(
                op.name.clone(),
                rows.clone(),
                cols.clone(),
                Array2::zeros((rows.width(), cols.width())),
                empty,
                precision,
                op.provenance.clone(),
            )
            .map_err(|e| e.to_string())?;
            program.operators[d.operator] = Arc::new(blank);
        }
        Ok(program)
    }

    /// Exact encoded bodies identify sharing, including the sign of zero.
    /// New-format dependencies must already be in their executable topological order.
    fn matrix_rules(&self) -> Result<Vec<(BitString, Arc<MatrixRule>)>, String> {
        let mut bodies = Vec::new();
        for d in &self.derived {
            if let OperatorLaw::Expression { body, sources } = &d.law {
                if sources.len() != body.inputs.len() { return Err("a matrix-rule call has the wrong source arity".into()); }
                let message = body.encode()?;
                if !bodies.iter().any(|(previous, _)| *previous == message) { bodies.push((message, Arc::clone(body))); }
            }
        }
        if !bodies.is_empty() {
            let count = self.program.operators.len();
            let mut pending: std::collections::BTreeSet<usize> = self.derived.iter().map(|d| d.operator).collect();
            if pending.len() != self.derived.len() { return Err("duplicate derived operator targets".into()); }
            for d in &self.derived {
                if d.operator >= count || d.law.sources().iter().any(|source| *source >= count || *source == d.operator || pending.contains(source)) { return Err("matrix artifact has absent, cyclic or out-of-order derived sources".into()); }
                if !d.scale.is_finite() || d.residual.iter().flat_map(|(_, row)| row).any(|v| !v.is_finite()) { return Err("a matrix artifact has nonfinite derived literals".into()); }
                pending.remove(&d.operator);
            }
        }
        Ok(bodies)
    }

    /// Derived scales/residuals plus each shared explicit body's coefficients once.
    pub fn derived_literals(&self) -> Result<u64, String> {
        let mut literals = self.derived.iter().map(Derived::literals).sum();
        for (_, body) in self.matrix_rules()? { literals += body.cost()?.literals; }
        Ok(literals)
    }

    /// Verify that every live change is hidden behind a measured block boundary.
    /// Names, provenance and lattice metadata do not affect executable operators.
    pub fn validate_coverage(&self, model: &OperatorProgram) -> Result<(), String> {
        use std::collections::BTreeSet;
        crate::native_control::validate(self, model)?;
        if model.nodes.len() != self.native_nodes || model.declarations != self.program.declarations {
            return Err("the artifact is not of this model".to_string());
        }
        let p = &self.program;
        model.interfaces().map_err(|e| e.to_string())?;
        p.interfaces().map_err(|e| e.to_string())?;
        if self.places.windows(2).any(|w| w[0].0 >= w[1].0) || self.places.iter().any(|&(n, v)| n >= model.nodes.len() || v >= p.nodes.len()) {
            return Err("invalid native places".to_string());
        }
        if self.place(model.output) != Some(p.output) {
            return Err("the artifact output is not the native output's place".to_string());
        }
        let mut writes = BTreeSet::new();
        for b in &self.blocks {
            if b.native_reads.len() != b.reads.len()
                || b.native_write >= model.nodes.len()
                || b.write >= p.nodes.len()
                || !writes.insert(b.native_write)
                || self.place(b.native_write) != Some(b.write)
                || b.native_reads.iter().zip(&b.reads).any(|(&n, &v)| n >= b.native_write || v >= b.write || self.place(n) != Some(v))
            {
                return Err(format!("{}: invalid block correspondence", b.name));
            }
        }
        fn op_equal(a: &Operator, b: &Operator) -> bool {
            if a.rows != b.rows || a.cols != b.cols {
                return false;
            }
            match (&a.body, &b.body) {
                (OperatorBody::Identity, OperatorBody::Identity) => true,
                (OperatorBody::Dense { values: av, present: ap, .. }, OperatorBody::Dense { values: bv, present: bp, .. }) => av == bv && ap == bp,
                (OperatorBody::LowRank { left: al, right: ar, .. }, OperatorBody::LowRank { left: bl, right: br, .. }) => al == bl && ar == br,
                (OperatorBody::Diagonal { values: av, .. }, OperatorBody::Diagonal { values: bv, .. }) => av == bv,
                _ => false,
            }
        }
        fn node_equal(
            a: &Node,
            b: &Node,
            m: &OperatorProgram,
            p: &OperatorProgram,
            node_count: usize,
            rules_seen: &mut BTreeSet<(usize, usize)>,
        ) -> bool {
            let (aa, ba) = (a.arguments(), b.arguments());
            let (ao, bo) = (a.operators(), b.operators());
            if aa.len() != ba.len() || ao.len() != bo.len() {
                return false;
            }
            let mut nodes = vec![usize::MAX; node_count];
            for (&n, &v) in aa.iter().zip(&ba) {
                if nodes[n] != usize::MAX && nodes[n] != v {
                    return false;
                }
                nodes[n] = v;
            }
            let mut operators = vec![usize::MAX; m.operators.len()];
            for (&n, &v) in ao.iter().zip(&bo) {
                if !op_equal(&m.operators[n], &p.operators[v]) || (operators[n] != usize::MAX && operators[n] != v) {
                    return false;
                }
                operators[n] = v;
            }
            let mut bases = vec![usize::MAX; m.bases.len()];
            if let (Node::Feature { basis: n, .. }, Node::Feature { basis: v, .. })
            | (Node::Readout { basis: n, .. }, Node::Readout { basis: v, .. }) = (a, b) {
                if m.bases[*n] != p.bases[*v] {
                    return false;
                }
                bases[*n] = *v;
            }
            let mut rules = vec![usize::MAX; m.rules.len()];
            if let (Node::Call { rule: n, .. }, Node::Call { rule: v, .. }) = (a, b) {
                rules[*n] = *v;
                if rules_seen.insert((*n, *v)) {
                    let (nr, pr) = (&m.rules[*n], &p.rules[*v]);
                    if nr.inputs != pr.inputs || nr.nodes.len() != pr.nodes.len() || nr.output != pr.output {
                        return false;
                    }
                    for (an, bn) in nr.nodes.iter().zip(&pr.nodes) {
                        // Unchanged rule body retains its internal wiring; only global
                        // operators, bases and called rule indices may be renumbered.
                        if an.arguments() != bn.arguments() || !node_equal(an, bn, m, p, nr.nodes.len(), rules_seen) {
                            return false;
                        }
                    }
                }
            }
            // Refuse differing kinds before remap_node can index an absent resource map.
            if std::mem::discriminant(a) != std::mem::discriminant(b) {
                return false;
            }
            let mut copy = a.clone();
            remap_node(&mut copy, &nodes, &operators, &bases, &rules);
            copy == *b
        }
        let mut pending = vec![(model.output, p.output)];
        let mut visited = BTreeSet::new();
        while let Some((native, node)) = pending.pop() {
            if !visited.insert((native, node)) {
                continue;
            }
            if self.place(native) != Some(node) {
                return Err(format!("native node {native} has an undeclared correspondence to node {node}"));
            }
            if let Some(block) = self.blocks.iter().find(|b| b.native_write == native && b.write == node) {
                pending.extend(block.native_reads.iter().copied().zip(block.reads.iter().copied()));
                continue;
            }
            if self.exceptions.iter().any(|e| e.node == node) {
                return Err(format!("node {node}: executable exception outside a measured block"));
            }
            let (a, b) = (&model.nodes[native], &p.nodes[node]);
            if !node_equal(a, b, model, p, model.nodes.len(), &mut BTreeSet::new()) {
                return Err(format!("native node {native}, node {node}: executable change outside a measured block: {a:?} versus {b:?}"));
            }
            pending.extend(a.arguments().into_iter().zip(b.arguments()));
        }
        Ok(())
    }

    /// The node of `P` holding native node `native`, if `P` holds it.
    pub fn place(&self, native: usize) -> Option<usize> {
        self.places.binary_search_by_key(&native, |(n, _)| *n).ok().map(|at| self.places[at].1)
    }

    /// This artifact with the native block that writes native node `native_write` replaced by a
    /// call of `callee` on `arguments`. The block's native parent state is the native arguments. A
    /// new rule's body refers to operators by index into `P`'s operators followed by `operators`
    /// (so the first new one is `self.program.operators.len()`), and may call rules `P` stores. The
    /// call takes the write's place: every reader of the write reads the call, the native nodes only
    /// the old block read leave `P`, and so do their places; a block this one contains is replaced
    /// with it.
    pub fn replace_block(
        &self,
        name: &str,
        callee: Callee,
        arguments: Vec<Argument>,
        native_write: usize,
        operators: Vec<Operator>,
    ) -> Result<Self, String> {
        let write = self.place(native_write).ok_or_else(|| format!("{name}: P does not hold native node {native_write}"))?;
        let mut program = self.program.clone();
        program.operators.extend(operators.into_iter().map(Arc::new));
        let rule = match callee {
            Callee::New(rule) => {
                program.rules.push(rule);
                program.rules.len() - 1
            }
            Callee::Existing(rule) if rule < program.rules.len() => rule,
            Callee::Existing(rule) => return Err(format!("{name}: P stores no rule {rule}")),
        };
        // The constants, then the call, are inserted at the write's position.
        let mut native_reads = Vec::new();
        let mut reads = Vec::new();
        let mut inserted: Vec<Node> = Vec::new();
        let mut call_arguments = Vec::new();
        for argument in arguments {
            match argument {
                Argument::Native(native) => {
                    let node = self.place(native).ok_or_else(|| format!("{name}: P does not hold native read {native}"))?;
                    if node >= write {
                        return Err(format!("{name}: read {native} is not computed before the write {native_write}"));
                    }
                    native_reads.push(native);
                    reads.push(node);
                    call_arguments.push(node);
                }
                Argument::Constant(op) => {
                    program.operators.push(Arc::new(op));
                    call_arguments.push(write + inserted.len());
                    inserted.push(Node::Constant { operator: program.operators.len() - 1 });
                }
            }
        }
        let call = write + inserted.len();
        inserted.push(Node::Call { rule, arguments: call_arguments });
        let shift = inserted.len();
        // Old node `n` moves to `n + shift` from the write on; readers of the write read the call.
        let mut map: Vec<usize> = (0..program.nodes.len()).map(|n| if n < write { n } else { n + shift }).collect();
        let readers_map: Vec<usize> = (0..program.nodes.len()).map(|n| if n == write { call } else { map[n] }).collect();
        let (ops, bases, rules) = (identity(program.operators.len()), identity(program.bases.len()), identity(program.rules.len()));
        let mut nodes = Vec::with_capacity(program.nodes.len() + shift);
        for (index, node) in program.nodes.iter().enumerate() {
            if index == write {
                nodes.append(&mut inserted);
            }
            let mut node = node.clone();
            remap_node(&mut node, &readers_map, &ops, &bases, &rules);
            nodes.push(node);
        }
        program.nodes = nodes;
        program.output = readers_map[program.output];
        map[write] = call;
        // Compact: what only the old block read is gone.
        let (live, op_map) = compact(&mut program, &self.derived_operators());
        let derived = self.renumbered_derived(&op_map)?;
        let at = |old: usize| -> Option<usize> { Some(live[map[old]]).filter(|n| *n != usize::MAX) };
        let at_new = |new: usize| -> Option<usize> { Some(live[new]).filter(|n| *n != usize::MAX) };
        let places: Vec<(usize, usize)> = self.places.iter().filter_map(|&(native, node)| at(node).map(|n| (native, n))).collect();
        let mut blocks: Vec<Binding> = Vec::new();
        for block in &self.blocks {
            if block.native_write == native_write {
                continue;
            }
            let (Some(write), Some(reads)) = (at(block.write), block.reads.iter().map(|r| at(*r)).collect::<Option<Vec<_>>>()) else {
                continue;
            };
            blocks.push(Binding { reads, write, ..block.clone() });
        }
        blocks.push(Binding {
            name: name.to_string(),
            native_reads,
            native_write,
            reads: reads.iter().map(|r| at_new(*r)).collect::<Option<Vec<_>>>().ok_or_else(|| format!("{name}: a read left P"))?,
            write: at_new(call).ok_or_else(|| format!("{name}: the call is not read"))?,
        });
        let exceptions = self.exceptions.iter().filter_map(|e| at(e.node).map(|node| Exception { node, ..e.clone() })).collect();
        let controls = self.controls.iter().filter_map(|c| at(c.write).map(|write| crate::native_control::UniformScaleBinding { write, ..c.clone() })).filter(|c| blocks.iter().any(|b| b.native_write == c.native_write && b.write == c.write)).collect();
        let artifact = Self { program, native_nodes: self.native_nodes, blocks, places, exceptions, derived, controls };
        artifact.program.interfaces().map_err(|e| format!("{name}: {e}"))?;
        Ok(artifact)
    }

    /// This artifact with the native block from `native_reads` to `native_write` declared replaced,
    /// its computation in `P` being whatever `P` computes between the places of those nodes (an
    /// operator a rule derives, a changed law): its local disagreement is then measured like any
    /// replaced block's. A block already declared with this write is declared anew.
    pub fn bind(&self, name: &str, native_reads: &[usize], native_write: usize) -> Result<Self, String> {
        let place = |native: usize| self.place(native).ok_or_else(|| format!("{name}: P does not hold native node {native}"));
        let reads = native_reads.iter().map(|n| place(*n)).collect::<Result<Vec<_>, _>>()?;
        let write = place(native_write)?;
        if reads.iter().any(|r| *r >= write) {
            return Err(format!("{name}: a read is not computed before the write"));
        }
        let mut out = self.clone();
        out.blocks.retain(|b| b.native_write != native_write);
        out.blocks.push(Binding { name: name.to_string(), native_reads: native_reads.to_vec(), native_write, reads, write });
        Ok(out)
    }

    /// This artifact with native node `native` (a place of `P`) as its output: what only the nodes
    /// past it read leaves, with its places, blocks and exceptions.
    pub fn truncated(&self, native: usize) -> Result<Self, String> {
        let output = self.place(native).ok_or_else(|| format!("P does not hold native node {native}"))?;
        let mut program = self.program.clone();
        program.output = output;
        let (live, op_map) = compact(&mut program, &self.derived_operators());
        let derived = self.renumbered_derived(&op_map)?;
        let at = |node: usize| Some(live[node]).filter(|n| *n != usize::MAX);
        let places = self.places.iter().filter_map(|&(n, node)| at(node).map(|m| (n, m))).collect();
        let blocks: Vec<Binding> = self
            .blocks
            .iter()
            .filter_map(|b| {
                let reads = b.reads.iter().map(|r| at(*r)).collect::<Option<Vec<_>>>()?;
                Some(Binding { reads, write: at(b.write)?, ..b.clone() })
            })
            .collect();
        let exceptions = self.exceptions.iter().filter_map(|e| at(e.node).map(|node| Exception { node, ..e.clone() })).collect();
        let controls = self.controls.iter().filter_map(|c| at(c.write).map(|write| crate::native_control::UniformScaleBinding { write, ..c.clone() })).filter(|c| blocks.iter().any(|b| b.native_write == c.native_write && b.write == c.write)).collect();
        Ok(Self { program, native_nodes: self.native_nodes, blocks, places, exceptions, derived, controls })
    }

    /// Whether learned operator and coefficient literals are 32-bit floats.
    /// Exact architecture RMSNorm epsilon values are preserved; derived computed
    /// reals are not independently sent parameter literals.
    pub fn has_f32_literals(&self) -> bool {
        let derived: std::collections::BTreeSet<usize> = self.derived.iter().map(|d| d.operator).collect();
        self.program.operators.iter().enumerate().all(|(i, op)| derived.contains(&i) || has_f32_reals(op))
            && self.program.nodes.iter().chain(self.program.rules.iter().flat_map(|r| &r.nodes)).all(|node| match node {
                Node::Gain { coefficient, .. } => coefficient_f32(coefficient),
                _ => true,
            })
    }

    /// Round learned operator and coefficient literals to 32-bit floats, and
    /// recompute derived operators. Preserve exact architecture RMSNorm epsilon
    /// values and their execution; this is not a change to the C32 price.
    pub fn f32_literals(&self) -> Result<Self, String> {
        let derived: std::collections::BTreeSet<usize> = self.derived.iter().map(|d| d.operator).collect();
        let mut out = self.clone();
        let mut changed = false;
        for (i, op) in out.program.operators.iter_mut().enumerate() {
            if !derived.contains(&i) && !has_f32_reals(op) {
                *op = Arc::new(f32_operator(op)?);
                changed = true;
            }
        }
        for node in out.program.nodes.iter_mut().chain(out.program.rules.iter_mut().flat_map(|r| &mut r.nodes)) {
            if let Node::Gain { coefficient, .. } = node {
                round_coefficient(coefficient);
            }
        }
        if changed {
            compute_derived(&mut out.program, &self.derived)?;
        }
        Ok(out)
    }

    /// `P`'s unbanded trace of `inputs`, its exceptions added where their contexts occur, each node
    /// passed through `edit` after them (an intervention, a patched state).
    pub fn execute_edited<F>(&self, inputs: &FamilyInputs, mut edit: F) -> Result<Trace, String>
    where
        F: FnMut(usize, &mut Array2<f64>, &[Array2<f64>]) -> Result<(), String>,
    {
        let mut at: BTreeMap<usize, Vec<(usize, usize, f64)>> = BTreeMap::new();
        if !self.exceptions.is_empty() {
            let rows = contexts(inputs);
            for exception in &self.exceptions {
                for (row, context) in rows.iter().enumerate() {
                    if *context == exception.context {
                        at.entry(exception.node).or_default().push((row, exception.column, f64::from(exception.value)));
                    }
                }
            }
        }
        self.program
            .execute_edited(inputs, |node, value, earlier| {
                for &(row, column, add) in at.get(&node).map_or(&[][..], Vec::as_slice) {
                    value[[row, column]] += add;
                }
                edit(node, value, earlier)
            })
            .map_err(|e| e.to_string())
    }

    /// `P`'s unbanded trace of `inputs`.
    pub fn execute(&self, inputs: &FamilyInputs) -> Result<Trace, String> {
        self.execute_edited(inputs, |_, _, _| Ok(()))
    }

    /// Every block of `P` grafted onto `model` (module note): the output is, block after block, the
    /// block's write in `P` minus `model`'s, both on `model`'s parent state, and the second value is
    /// each block's columns of it. Every node of `P` a write depends on is copied unless it is one
    /// of its block's reads (it reads `model`'s instead). A block whose write depends on an input
    /// slot by a path that passes none of its reads is refused: it would compute part of its parent
    /// state from `P`'s own upstream, not read the native one.
    pub fn local_program(&self, model: &OperatorProgram) -> Result<(OperatorProgram, Vec<std::ops::Range<usize>>), String> {
        let (artifact, columns) = self.local_artifact(model)?;
        if !artifact.exceptions.is_empty() {
            return Err("local_program cannot discard executable exceptions; use local_artifact".to_string());
        }
        Ok((artifact.program, columns))
    }

    /// Local blocks with executable exceptions, excluding clamped native reads.
    pub fn local_artifact(&self, model: &OperatorProgram) -> Result<(Self, Vec<std::ops::Range<usize>>), String> {
        let (artifact, columns, _) = self.local_artifact_with_writes(model)?;
        Ok((artifact, columns))
    }

    /// One replacement evaluated on its declared native parent states. The second
    /// result is its direct write node in the grafted artifact, before subtraction
    /// of the native write; internal/write exceptions execute, clamped-read ones do not.
    pub fn local_block_artifact(&self, model: &OperatorProgram, block: usize) -> Result<(Self, usize), String> {
        let binding = self.blocks.get(block).ok_or("local block index outside artifact")?;
        let mut one = self.clone();
        one.blocks = vec![binding.clone()];
        let (artifact, _, writes) = one.local_artifact_with_writes(model)?;
        Ok((artifact, writes[0]))
    }

    fn local_artifact_with_writes(&self, model: &OperatorProgram) -> Result<(Self, Vec<std::ops::Range<usize>>, Vec<usize>), String> {
        if model.nodes.len() != self.native_nodes || model.declarations != self.program.declarations {
            return Err("the artifact is not of this model".to_string());
        }
        if self.blocks.is_empty() {
            return Err("an artifact with no replaced block".to_string());
        }
        let p = &self.program;
        let mut out = model.clone();
        let (op_base, basis_base, rule_base) = (out.operators.len(), out.bases.len(), out.rules.len());
        out.operators.extend(p.operators.iter().cloned());
        out.bases.extend(p.bases.iter().cloned());
        let op_map: Vec<usize> = (0..p.operators.len()).map(|i| op_base + i).collect();
        let basis_map: Vec<usize> = (0..p.bases.len()).map(|i| basis_base + i).collect();
        let rule_map: Vec<usize> = (0..p.rules.len()).map(|i| rule_base + i).collect();
        for rule in &p.rules {
            let mut rule = rule.clone();
            let body = identity(rule.nodes.len());
            for node in rule.nodes.iter_mut() {
                remap_node(node, &body, &op_map, &basis_map, &rule_map);
            }
            out.rules.push(rule);
        }
        let precision = exact_precision([-1.0]).map_err(|e| e.to_string())?;
        let mut differences = Vec::new();
        let mut writes = Vec::new();
        let mut exceptions = Vec::new();
        for binding in &self.blocks {
            // The cone of the write, bounded by the reads.
            let reads: BTreeMap<usize, usize> = binding.reads.iter().copied().zip(binding.native_reads.iter().copied()).collect();
            let mut cone = vec![false; p.nodes.len()];
            cone[binding.write] = true;
            for index in (0..=binding.write).rev() {
                if !cone[index] || reads.contains_key(&index) {
                    continue;
                }
                let node = &p.nodes[index];
                if matches!(node, Node::Feature { .. } | Node::Raw { .. }) {
                    return Err(format!("{}: the block reads input node {index} beyond its declared reads", binding.name));
                }
                for argument in node.arguments() {
                    cone[argument] = true;
                }
            }
            let mut node_map = vec![usize::MAX; p.nodes.len()];
            for (index, keep) in cone.iter().enumerate() {
                if !*keep {
                    continue;
                }
                if let Some(native) = reads.get(&index) {
                    node_map[index] = *native;
                    continue;
                }
                let mut copy = p.nodes[index].clone();
                remap_node(&mut copy, &node_map, &op_map, &basis_map, &rule_map);
                out.nodes.push(copy);
                node_map[index] = out.nodes.len() - 1;
            }
            for exception in &self.exceptions {
                if cone.get(exception.node).copied().unwrap_or(false) && !reads.contains_key(&exception.node) {
                    exceptions.push(Exception { node: node_map[exception.node], ..exception.clone() });
                }
            }
            let interfaces = out.interfaces().map_err(|e| e.to_string())?;
            let (mine, theirs) = (node_map[binding.write], binding.native_write);
            writes.push(mine);
            let interface: Interface = interfaces[theirs].clone();
            if interfaces[mine] != interface {
                return Err(format!(
                    "{}: the block writes an interface of width {}, the native block one of width {}",
                    binding.name,
                    interfaces[mine].width(),
                    interface.width()
                ));
            }
            out.operators.push(Arc::new(Operator::identity("block write", interface.clone())));
            let plus = out.operators.len() - 1;
            let minus_one = Array1::from_elem(interface.width(), -1.0);
            let negate =
                Operator::diag("native write, negated", interface, minus_one, precision, Provenance::default()).map_err(|e| e.to_string())?;
            out.operators.push(Arc::new(negate));
            let minus = out.operators.len() - 1;
            out.nodes.push(Node::Affine { terms: vec![(mine, plus), (theirs, minus)], bias: None });
            differences.push(out.nodes.len() - 1);
        }
        let interfaces = out.interfaces().map_err(|e| e.to_string())?;
        let mut columns = Vec::new();
        let mut at = 0;
        for &node in &differences {
            let width = interfaces[node].width();
            columns.push(at..at + width);
            at += width;
        }
        out.output = if differences.len() == 1 {
            differences[0]
        } else {
            out.nodes.push(Node::Concat { parts: differences });
            out.nodes.len() - 1
        };
        let (live, _) = compact(&mut out, &[]);
        for write in &mut writes {
            *write = live[*write];
        }
        for exception in &mut exceptions {
            exception.node = live[exception.node];
        }
        exceptions.retain(|e| e.node != usize::MAX);
        out.interfaces().map_err(|e| e.to_string())?;
        Ok((Self { program: out, native_nodes: self.native_nodes, blocks: Vec::new(), places: Vec::new(), exceptions, derived: Vec::new(), controls: Vec::new() }, columns, writes))
    }

    /// Numeric-free blocks, places, exceptions and derivation structure, excluding
    /// diagnostic names. Exception values, derived scales and residual values are
    /// independently priced at 32 bits each by C32, outside this binding remainder.
    pub fn binding_bits(&self) -> Result<u64, String> {
        crate::native_control::validate_shape(self)?;
        let bodies = self.matrix_rules()?;
        let versioned = !bodies.is_empty() || !self.controls.is_empty();
        let version = if self.controls.is_empty() { MATRIX_ARTIFACT_VERSION } else { CONTROL_ARTIFACT_VERSION };
        let nodes = self.program.nodes.len();
        let fixed = |alphabet: usize| -> Result<u64, String> { Ok(u64::from(fixed_index_len_bits(alphabet).map_err(codec)?)) };
        let prefix = |value: u64| prefix_integer_len_bits(value).map_err(codec);
        let mut bits = prefix(self.blocks.len() as u64 + 1)?;
        for block in &self.blocks {
            bits += prefix(block.reads.len() as u64 + 1)? + (block.reads.len() as u64 + 1) * (fixed(self.native_nodes)? + fixed(nodes)?);
        }
        bits += prefix(self.places.len() as u64 + 1)? + self.places.len() as u64 * (fixed(self.native_nodes)? + fixed(nodes)?);
        bits += prefix(self.exceptions.len() as u64 + 1)?;
        let interfaces = self.program.interfaces().map_err(|e| e.to_string())?;
        for exception in &self.exceptions {
            bits += prefix(exception.context.len() as u64 + 1)?;
            for token in &exception.context {
                bits += prefix(u64::from(*token) + 1)?;
            }
            bits += fixed(nodes)? + fixed(interfaces[exception.node].width())?;
        }
        // The derived operators less their literals' 32 bits each (charged as literals).
        let operators = self.program.operators.len();
        if versioned {
            // Internal dispatch marker, explicit format version and body-pool framing.
            bits += prefix(1)? + prefix(version)? + prefix(bodies.len() as u64 + 1)?;
            for (message, body) in &bodies { bits += prefix(message.len_bits() + 1)? + body.cost()?.structure_bits; }
        }
        bits += prefix(self.derived.len() as u64 + 1)?;
        for d in &self.derived {
            bits += fixed(operators)? + fixed(if versioned { MATRIX_LAWS } else { LAWS })? + d.law.sources().len() as u64 * fixed(operators)?;
            if matches!(d.law, OperatorLaw::Expression { .. }) { bits += fixed(bodies.len())?; }
            for integer in d.law.integers() {
                bits += prefix(integer + 1)?;
            }
            bits += prefix(d.residual.len() as u64 + 1)? + d.residual.len() as u64 * fixed(self.program.operators[d.operator].rows.width())?;
        }
        if !self.controls.is_empty() {
            crate::native_control::validate_shape(self)?;
            bits += prefix(self.controls.len() as u64 + 1)?;
            for c in &self.controls {
                bits += prefix(1)? + 2 * fixed(self.native_nodes)? + fixed(nodes)?
                    + prefix((c.width as u64).checked_add(1).ok_or("control width overflow")?)?;
            }
        }
        Ok(bits)
    }

    /// The artifact's message (module note).
    pub fn encode(&self) -> Result<BitString, String> {
        self.encode_using(None)
    }

    /// Encode the identical standalone message using a bounded native operator cache.
    pub fn encode_with_native_codec(&self, cache: &crate::operator_program::NativeOperatorCodec) -> Result<BitString, String> {
        self.encode_using(Some(cache))
    }

    fn encode_using(&self, cache: Option<&crate::operator_program::NativeOperatorCodec>) -> Result<BitString, String> {
        crate::native_control::validate_shape(self)?;
        let bodies = self.matrix_rules()?;
        let versioned = !bodies.is_empty() || !self.controls.is_empty();
        let version = if self.controls.is_empty() { MATRIX_ARTIFACT_VERSION } else { CONTROL_ARTIFACT_VERSION };
        let source = self.message_program()?;
        let program = match cache {
            Some(cache) => source.encode_with_native_codec(cache),
            None => source.encode(),
        }.map_err(|e| e.to_string())?;
        let nodes = self.program.nodes.len();
        let mut out = BitString::new();
        if versioned {
            encode_prefix_integer(&mut out, 1).map_err(codec)?; // Invalid zero native-node count in legacy grammar.
            encode_prefix_integer(&mut out, version).map_err(codec)?;
        }
        encode_prefix_integer(&mut out, self.native_nodes as u64 + 1).map_err(codec)?;
        encode_prefix_integer(&mut out, program.len_bits() + 1).map_err(codec)?;
        out.append(&program);
        encode_prefix_integer(&mut out, self.blocks.len() as u64 + 1).map_err(codec)?;
        for block in &self.blocks {
            encode_prefix_integer(&mut out, block.name.len() as u64 + 1).map_err(codec)?;
            for byte in block.name.bytes() {
                out.push_bits(u64::from(byte), 8).map_err(codec)?;
            }
            if block.reads.len() != block.native_reads.len() {
                return Err(format!("{}: {} reads for {} native reads", block.name, block.reads.len(), block.native_reads.len()));
            }
            encode_prefix_integer(&mut out, block.reads.len() as u64 + 1).map_err(codec)?;
            for native in block.native_reads.iter().chain(std::iter::once(&block.native_write)) {
                encode_fixed_index(&mut out, *native, self.native_nodes).map_err(codec)?;
            }
            for node in block.reads.iter().chain(std::iter::once(&block.write)) {
                encode_fixed_index(&mut out, *node, nodes).map_err(codec)?;
            }
        }
        encode_prefix_integer(&mut out, self.places.len() as u64 + 1).map_err(codec)?;
        for &(native, node) in &self.places {
            encode_fixed_index(&mut out, native, self.native_nodes).map_err(codec)?;
            encode_fixed_index(&mut out, node, nodes).map_err(codec)?;
        }
        let interfaces = self.program.interfaces().map_err(|e| e.to_string())?;
        encode_prefix_integer(&mut out, self.exceptions.len() as u64 + 1).map_err(codec)?;
        for exception in &self.exceptions {
            encode_prefix_integer(&mut out, exception.context.len() as u64 + 1).map_err(codec)?;
            for token in &exception.context {
                encode_prefix_integer(&mut out, u64::from(*token) + 1).map_err(codec)?;
            }
            encode_fixed_index(&mut out, exception.node, nodes).map_err(codec)?;
            encode_fixed_index(&mut out, exception.column, interfaces[exception.node].width()).map_err(codec)?;
            out.push_bits(u64::from(exception.value.to_bits()), 32).map_err(codec)?;
        }
        let operators = self.program.operators.len();
        if versioned {
            encode_prefix_integer(&mut out, bodies.len() as u64 + 1).map_err(codec)?;
            for (message, _) in &bodies {
                encode_prefix_integer(&mut out, message.len_bits() + 1).map_err(codec)?;
                out.append(message);
            }
        }
        encode_prefix_integer(&mut out, self.derived.len() as u64 + 1).map_err(codec)?;
        for d in &self.derived {
            encode_fixed_index(&mut out, d.operator, operators).map_err(codec)?;
            encode_fixed_index(&mut out, d.law.index(), if versioned { MATRIX_LAWS } else { LAWS }).map_err(codec)?;
            if let OperatorLaw::Expression { body, .. } = &d.law {
                let message = body.encode()?;
                let index = bodies.iter().position(|(previous, _)| *previous == message).ok_or("a missing shared matrix-rule body")?;
                encode_fixed_index(&mut out, index, bodies.len()).map_err(codec)?;
            }
            for source in d.law.sources() {
                encode_fixed_index(&mut out, source, operators).map_err(codec)?;
            }
            for integer in d.law.integers() {
                encode_prefix_integer(&mut out, integer + 1).map_err(codec)?;
            }
            out.push_bits(u64::from(d.scale.to_bits()), 32).map_err(codec)?;
            let rows = self.program.operators[d.operator].rows.width();
            encode_prefix_integer(&mut out, d.residual.len() as u64 + 1).map_err(codec)?;
            for (row, values) in &d.residual {
                encode_fixed_index(&mut out, *row, rows).map_err(codec)?;
                if values.len() != self.program.operators[d.operator].cols.width() {
                    return Err(format!("a residual row of {} values", values.len()));
                }
                for v in values {
                    out.push_bits(u64::from(v.to_bits()), 32).map_err(codec)?;
                }
            }
        }
        if !self.controls.is_empty() {
            encode_prefix_integer(&mut out, self.controls.len() as u64 + 1).map_err(codec)?;
            for c in &self.controls {
                encode_prefix_integer(&mut out, 1).map_err(codec)?; // UniformScaleThroughLinearWrite.
                encode_fixed_index(&mut out, c.native_source, self.native_nodes).map_err(codec)?;
                encode_fixed_index(&mut out, c.native_write, self.native_nodes).map_err(codec)?;
                encode_fixed_index(&mut out, c.write, nodes).map_err(codec)?;
                encode_prefix_integer(&mut out, (c.width as u64).checked_add(1).ok_or("control width overflow")?).map_err(codec)?;
            }
        }
        Ok(out)
    }

    /// The artifact a message holds, given the declarations alone.
    pub fn decode(message: &BitString, declarations: &Declarations) -> Result<Self, String> {
        Self::decode_using(message, declarations, None)
    }

    /// Decode the same message, reusing only exact witnessed native codewords.
    pub fn decode_with_native_codec(message: &BitString, declarations: &Declarations, cache: &crate::operator_program::NativeOperatorCodec) -> Result<Self, String> {
        Self::decode_using(message, declarations, Some(cache))
    }

    fn decode_using(message: &BitString, declarations: &Declarations, cache: Option<&crate::operator_program::NativeOperatorCodec>) -> Result<Self, String> {
        let mut reader = message.reader();
        let reader = &mut reader;
        // A count of items each taking at least one bit, so no larger than the bits left.
        let count = |reader: &mut BitReader<'_>| -> Result<usize, String> {
            let n = decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or("a zero count codeword")?;
            if n > reader.remaining_bits() {
                return Err(format!("a count of {n} beyond the message"));
            }
            Ok(n as usize)
        };
        let first = decode_prefix_integer(reader).map_err(codec)?;
        let versioned = first == 1;
        let version = if versioned { decode_prefix_integer(reader).map_err(codec)? } else { 0 };
        if versioned && version != MATRIX_ARTIFACT_VERSION && version != CONTROL_ARTIFACT_VERSION {
            return Err(format!("unsupported artifact grammar version {version}"));
        }
        let native_code = if versioned { decode_prefix_integer(reader).map_err(codec)? } else { first };
        let native_nodes = usize::try_from(native_code.checked_sub(1).ok_or("a zero count codeword")?).map_err(|e| e.to_string())?;
        if native_nodes == 0 {
            return Err("a native model of no nodes".to_string());
        }
        let program_bits = decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or("a zero length codeword")?;
        if program_bits > reader.remaining_bits() {
            return Err(format!("a program of {program_bits} bits in {} remaining", reader.remaining_bits()));
        }
        let mut program_reader = reader.bounded_subreader(program_bits).map_err(codec)?;
        let program = OperatorProgram::decode_reader(&mut program_reader, declarations, cache)
            .map_err(|e| e.to_string())?;
        let nodes = program.nodes.len();
        let interfaces = program.interfaces().map_err(|e| e.to_string())?;
        let mut blocks = Vec::new();
        for _ in 0..count(reader)? {
            let length = count(reader)?;
            let bytes = (0..length).map(|_| reader.read_bits(8).map(|b| b as u8)).collect::<Result<Vec<u8>, _>>().map_err(codec)?;
            let name = String::from_utf8(bytes).map_err(|e| e.to_string())?;
            // Reads are fixed indices, possibly of no bits, so their count is bounded by the nodes.
            let reads = decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or("a zero count codeword")? as usize;
            if reads > nodes {
                return Err(format!("a block of {reads} reads in a program of {nodes} nodes"));
            }
            let native: Vec<usize> = (0..=reads).map(|_| decode_fixed_index(reader, native_nodes)).collect::<Result<_, _>>().map_err(codec)?;
            let own: Vec<usize> = (0..=reads).map(|_| decode_fixed_index(reader, nodes)).collect::<Result<_, _>>().map_err(codec)?;
            blocks.push(Binding {
                name,
                native_reads: native[..reads].to_vec(),
                native_write: native[reads],
                reads: own[..reads].to_vec(),
                write: own[reads],
            });
        }
        let mut places = Vec::new();
        let place_count = decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or("a zero count codeword")? as usize;
        if place_count > native_nodes {
            return Err(format!("{place_count} places of {native_nodes} native nodes"));
        }
        for _ in 0..place_count {
            let native = decode_fixed_index(reader, native_nodes).map_err(codec)?;
            places.push((native, decode_fixed_index(reader, nodes).map_err(codec)?));
        }
        let mut exceptions = Vec::new();
        for _ in 0..count(reader)? {
            let length = count(reader)?;
            let context = (0..length)
                .map(|_| {
                    let token = decode_prefix_integer(reader).map_err(codec)?;
                    token.checked_sub(1).map(|t| t as u32).ok_or_else(|| "a zero token codeword".to_string())
                })
                .collect::<Result<Vec<u32>, String>>()?;
            let node = decode_fixed_index(reader, nodes).map_err(codec)?;
            let column = decode_fixed_index(reader, interfaces[node].width()).map_err(codec)?;
            let value = f32::from_bits(reader.read_bits(32).map_err(codec)? as u32);
            exceptions.push(Exception { context, node, column, value });
        }
        let mut program = program;
        let operators = program.operators.len();
        let mut bodies: Vec<(BitString, Arc<MatrixRule>)> = Vec::new();
        if versioned {
            for _ in 0..count(reader)? {
                let bits = decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or("zero matrix-rule length codeword")?;
                let message = reader.read_bit_string(bits).map_err(codec)?;
                let body = MatrixRule::decode(&message)?;
                if bodies.iter().any(|(previous, _)| *previous == message) { return Err("duplicate shared matrix-rule body".into()); }
                bodies.push((message, Arc::new(body)));
            }
            if bodies.is_empty() && version == MATRIX_ARTIFACT_VERSION { return Err("a versioned matrix artifact has no rule bodies".into()); }
        }
        let mut used_bodies = std::collections::BTreeSet::new();
        let mut derived = Vec::new();
        for _ in 0..count(reader)? {
            let operator = decode_fixed_index(reader, operators).map_err(codec)?;
            let law = decode_fixed_index(reader, if versioned { MATRIX_LAWS } else { LAWS }).map_err(codec)?;
            let body_index = if law == 2 { let index = decode_fixed_index(reader, bodies.len()).map_err(codec)?; used_bodies.insert(index); Some(index) } else { None };
            let (sources, integers) = match body_index { Some(index) => (bodies[index].1.inputs.len(), 0), None => OperatorLaw::arity(law) };
            let sources: Vec<usize> = (0..sources).map(|_| decode_fixed_index(reader, operators)).collect::<Result<_, _>>().map_err(codec)?;
            let integers: Vec<u64> = (0..integers)
                .map(|_| decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or_else(|| "a zero integer codeword".to_string()))
                .collect::<Result<_, String>>()?;
            let scale = f32::from_bits(reader.read_bits(32).map_err(codec)? as u32);
            let (rows, cols) = (program.operators[operator].rows.width(), program.operators[operator].cols.width());
            let mut residual = Vec::new();
            for _ in 0..count(reader)? {
                let row = decode_fixed_index(reader, rows).map_err(codec)?;
                let values =
                    (0..cols).map(|_| reader.read_bits(32).map(|b| f32::from_bits(b as u32))).collect::<Result<Vec<f32>, _>>().map_err(codec)?;
                residual.push((row, values));
            }
            let law = match body_index { Some(index) => OperatorLaw::Expression { body: Arc::clone(&bodies[index].1), sources }, None => OperatorLaw::of(law, &sources, &integers)? };
            derived.push(Derived { operator, law, scale, residual });
        }
        if versioned {
            if used_bodies.len() != bodies.len() { return Err("unused shared matrix-rule body".into()); }
            // Validation uses only this decoded program and explicit bodies.
            Self { program: program.clone(), native_nodes, blocks: Vec::new(), places: Vec::new(), exceptions: Vec::new(), derived: derived.clone(), controls: Vec::new() }.matrix_rules()?;
        }
        let mut controls = Vec::new();
        if version == CONTROL_ARTIFACT_VERSION {
            let n = count(reader)?;
            if n == 0 || n > native_nodes { return Err("invalid control binding count".into()); }
            for _ in 0..n {
                let tag = decode_prefix_integer(reader).map_err(codec)?;
                if tag != 1 { return Err(format!("unsupported native control law {tag}")); }
                let native_source = decode_fixed_index(reader, native_nodes).map_err(codec)?;
                let native_write = decode_fixed_index(reader, native_nodes).map_err(codec)?;
                let write = decode_fixed_index(reader, nodes).map_err(codec)?;
                let width = usize::try_from(decode_prefix_integer(reader).map_err(codec)?.checked_sub(1).ok_or("zero control width codeword")?).map_err(|e|e.to_string())?;
                controls.push(crate::native_control::UniformScaleBinding { native_source, native_write, write, width });
            }
        }
        compute_derived(&mut program, &derived)?;
        if reader.remaining_bits() != 0 {
            return Err(format!("{} bits left after the artifact", reader.remaining_bits()));
        }
        if places.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err("places out of order".to_string());
        }
        let artifact = Self { program, native_nodes, blocks, places, exceptions, derived, controls };
        crate::native_control::validate_shape(&artifact)?;
        Ok(artifact)
    }

    /// The message as bytes: its bit length (8 bytes, little-endian), then its bits, most
    /// significant first, the last byte padded with zeros.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let message = self.encode()?;
        let mut out = Vec::with_capacity(8 + message.packed_bytes().len());
        if !self.matrix_rules()?.is_empty() || !self.controls.is_empty() {
            let version = if self.controls.is_empty() { MATRIX_ARTIFACT_VERSION } else { CONTROL_ARTIFACT_VERSION };
            out.extend_from_slice(&VERSIONED_ENVELOPE.to_le_bytes());
            out.extend_from_slice(&version.to_le_bytes());
        }
        out.extend_from_slice(&message.len_bits().to_le_bytes());
        out.extend_from_slice(message.packed_bytes());
        Ok(out)
    }

    /// The artifact [`Artifact::to_bytes`] wrote, given the declarations alone.
    pub fn from_bytes(bytes: &[u8], declarations: &Declarations) -> Result<Self, String> {
        let header: [u8; 8] = bytes.get(..8).ok_or("no length header")?.try_into().map_err(|_| "no length header")?;
        let first = u64::from_le_bytes(header);
        let (length, offset, version) = if first == VERSIONED_ENVELOPE {
            let version = u64::from_le_bytes(bytes.get(8..16).ok_or("truncated artifact version")?.try_into().map_err(|_| "invalid artifact version")?);
            if version != MATRIX_ARTIFACT_VERSION && version != CONTROL_ARTIFACT_VERSION { return Err(format!("unsupported artifact envelope version {version}")); }
            let length = u64::from_le_bytes(bytes.get(16..24).ok_or("truncated artifact length")?.try_into().map_err(|_| "invalid artifact length")?);
            (length, 24, version)
        } else { (first, 8, 0) };
        if length.div_ceil(8) != (bytes.len() - offset) as u64 {
            return Err(format!("{} message bits in {} bytes", length, bytes.len() - offset));
        }
        let message = BitString::from_packed(&bytes[offset..], length).map_err(codec)?;
        let mut grammar = message.reader();
        let internal_version = if decode_prefix_integer(&mut grammar).map_err(codec)? == 1 {
            decode_prefix_integer(&mut grammar).map_err(codec)?
        } else { 0 };
        if internal_version != version { return Err("artifact envelope and internal grammar version disagree".into()); }
        Self::decode(&message, declarations)
    }
}

/// An artifact's message as a decodable artifact (`precision::decode_then_evaluate`).
#[derive(Clone, Debug)]
pub struct EncodedArtifact {
    pub message: BitString,
    pub declarations: Declarations,
}

impl EncodedArtifact {
    pub fn of(artifact: &Artifact) -> Result<Self, String> {
        Ok(Self { message: artifact.encode()?, declarations: artifact.program.declarations.clone() })
    }
}

/// A borrowing cache view; the stored message remains independently decodable.
pub struct NativeCodecArtifact<'a> {
    encoded: &'a EncodedArtifact,
    cache: &'a crate::operator_program::NativeOperatorCodec,
}

impl EncodedArtifact {
    pub fn of_with_native_codec(artifact: &Artifact, cache: &crate::operator_program::NativeOperatorCodec) -> Result<Self, String> {
        Ok(Self { message: artifact.encode_with_native_codec(cache)?, declarations: artifact.program.declarations.clone() })
    }

    pub fn using_native_codec<'a>(&'a self, cache: &'a crate::operator_program::NativeOperatorCodec) -> NativeCodecArtifact<'a> {
        NativeCodecArtifact { encoded: self, cache }
    }
}

impl DecodableArtifact for NativeCodecArtifact<'_> {
    type Decoded = Artifact;
    fn decode(&self) -> Result<Artifact, String> {
        Artifact::decode_with_native_codec(&self.encoded.message, &self.encoded.declarations, self.cache)
    }
}

impl DecodableArtifact for EncodedArtifact {
    type Decoded = Artifact;

    fn decode(&self) -> Result<Artifact, String> {
        Artifact::decode(&self.message, &self.declarations)
    }
}

#[cfg(test)]
#[path = "matrix_artifact_tests.rs"]
mod matrix_artifact_tests;

#[cfg(test)]
#[path = "control_artifact_tests.rs"]
mod control_artifact_tests;

#[cfg(test)]
mod borrowed_program_decode_tests {
    use super::*;
    use crate::operator_program::{Slot, NativeOperatorCodec};
    fn fixture() -> Artifact {
        let interface=Interface::native(1).expect("onecolumn interface");
        let program=OperatorProgram {
            declarations:Declarations {parameters:0,domains:vec![],slots:vec![Slot::Raw{width:1}]},
            bases:vec![],rules:vec![],operators:vec![Arc::new(Operator::identity("identity",interface))],
            nodes:vec![Node::Raw{slot:0},Node::Affine{terms:vec![(0,0)],bias:None}],output:1,
        };
        Artifact::native(&program).expect("native artifact")
    }
    #[test]
    fn borrowed_nested_program_matches_ordinary_and_cached_wire() {
        let artifact=fixture();let message=artifact.encode().expect("artifact encoding");
        let cache=NativeOperatorCodec::new(&artifact.program,1<<20).expect("bounded cache");
        let a=Artifact::decode(&message,&artifact.program.declarations).expect("ordinary decoder");
        let b=Artifact::decode_with_native_codec(&message,&artifact.program.declarations,&cache).expect("cached decoder");
        assert_eq!(a.encode().expect("canonical ordinary"),message);
        assert_eq!(b.encode().expect("canonical cached"),message);
        assert_eq!(a.program.operators[0].name,b.program.operators[0].name);
        let mut wrong=artifact.program.declarations.clone();wrong.parameters=1;
        assert!(Artifact::decode_with_native_codec(&message,&wrong,&cache).is_err());
    }
    #[test]
    fn borrowed_nested_program_rejects_wrong_lengths_without_artifact_tail_reads() {
        let artifact=fixture();let message=artifact.encode().expect("artifact encoding");
        let mut reader=message.reader();
        let native=decode_prefix_integer(&mut reader).expect("native count");
        let length=decode_prefix_integer(&mut reader).expect("program length")-1;
        let program=reader.read_bit_string(length).expect("test extraction");
        let tail=reader.read_bit_string(reader.remaining_bits()).expect("artifact tail");
        for bad in [0,length-1,length+1,u64::MAX-1] {
            let mut altered=BitString::new();
            encode_prefix_integer(&mut altered,native).expect("native prefix");
            encode_prefix_integer(&mut altered,bad+1).expect("malicious length code");
            altered.append(&program);altered.append(&tail);
            assert!(Artifact::decode(&altered,&artifact.program.declarations).is_err(),"length {bad}");
        }
    }
}
