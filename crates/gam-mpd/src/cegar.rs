//! Counterexample-guided refinement of a program decomposition (#2951).
//!
//! The two-part code scores a program on the behaviour it is shown. This module finds the
//! behaviour the program explains worst and shows it: a verifier searches the declared input
//! domain for inputs where `KL(model ‖ program)` is largest, and each input it certifies as worse
//! than every row already in the data becomes one more row of the data, observed like every other
//! row. The fit then runs again on the larger family. Nothing is labelled, weighted or trained
//! against: the verifier only adds rows, and the fit only reads the two-part code.
//!
//! [`verify`] is one round; `engine::decompose_refined` runs the loop:
//!
//! ```text
//! W₀ = the contract's family
//! loop:  P = decompose(model, W)
//!        X = { ascent endpoints x from every row of W and of the pool : KL_x(P) certified > max_{w∈W} KL_w(P) }
//!        X = ∅  →  certified: no ascent from the data finds an input worse than the data's worst
//!        W = W ∪ X
//! ```
//!
//! The loop is monotone in `W`, and a counterexample is never a row of `W` (its certified lower
//! end exceeds every row's upper end), so on a finite domain it ends after at most `|D|` rows.
//!
//! # The verifier
//!
//! Tokens are relaxed to the product of simplices over their allowed sets and raw slots to their
//! declared box. Every basis is linear in the one-hot, so the gradient of the input's KL with
//! respect to slot `j`'s one-hot is `Σ_nodes G_node Φ_nodeᵀ` over the feature nodes of the model
//! and of the program that read the slot (reverse mode through both programs, [`backward`]). At the
//! current input the Frank–Wolfe linear oracle is the best token of every slot and the box vertex
//! `sign(∇)` of every raw slot; the Frank–Wolfe gap is the linearised gain of that vertex.
//!
//! One ascent step evaluates, exactly and in one batch, the oracle vertex, every single-slot token
//! move with a positive linearised gain, and the box step toward the oracle at `γ = 2^{-k}` for
//! every `k` until the step no longer moves a coordinate's float. It takes the candidate whose
//! certified lower end is largest when that exceeds the current computed value. There is no step
//! size, iteration cap or restart count:
//!
//! * the starts are the data's rows, deterministic;
//! * a row stops when its gap is zero (no linearised gain) or no candidate certifies an increase;
//! * every accepted step raises the computed value strictly, so on a finite token domain an ascent
//!   is finite, and on a box the step ladder is bounded by the float resolution.
//!
//! A counterexample is always a real input of the declared domain (a vertex of the token simplices,
//! a point of the box) evaluated natively; the relaxation only proposes.
//!
//! The ascent is local, so its certificate is local: at termination every ascent from the data
//! ends at or below the data's worst row. On a domain small enough to enumerate,
//! [`exhaustive_worst`] states the supremum over the whole domain exactly.

use super::derivatives::vjp;
use super::operator_program::{FamilyInputs, Node, OperatorProgram, ProgramError, SlotValues, Trace, remap_node};
use super::verify::{RowValue, compare_logit_row};
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, s};
use std::collections::BTreeSet;
use std::fmt;

/// What one input slot may hold.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotDomain {
    /// Any of these tokens (one token is a fixed slot).
    Tokens(Vec<u32>),
    /// Any vector in the closed box `[lower, upper]`.
    Box { lower: Array1<f64>, upper: Array1<f64> },
    /// Any corner of the box: each coordinate at its lower or its upper end (a bit vector).
    Corners { lower: Array1<f64>, upper: Array1<f64> },
}

/// The declared input domain: one [`SlotDomain`] per declared slot.
#[derive(Clone, Debug, PartialEq)]
pub struct InputDomain {
    pub slots: Vec<SlotDomain>,
}

impl InputDomain {
    /// The number of inputs when every slot is a token set, `None` when some slot is a box or the
    /// count overflows.
    pub fn cardinality(&self) -> Option<u64> {
        self.slots.iter().try_fold(1u64, |count, slot| match slot {
            SlotDomain::Tokens(tokens) => count.checked_mul(tokens.len() as u64),
            SlotDomain::Corners { lower, .. } => 1u64.checked_shl(lower.len() as u32).and_then(|c| count.checked_mul(c)),
            SlotDomain::Box { .. } => None,
        })
    }

    fn contains(&self, input: &Input) -> bool {
        input.len() == self.slots.len()
            && input.iter().zip(&self.slots).all(|(value, slot)| match (value, slot) {
                (SlotValue::Token(t), SlotDomain::Tokens(tokens)) => tokens.contains(t),
                (SlotValue::Raw(x), SlotDomain::Box { lower, upper }) => {
                    x.len() == lower.len() && x.iter().zip(lower).zip(upper).all(|((v, l), u)| l <= v && v <= u)
                }
                (SlotValue::Raw(x), SlotDomain::Corners { lower, upper }) => {
                    x.len() == lower.len() && x.iter().zip(lower).zip(upper).all(|((v, l), u)| v == l || v == u)
                }
                _ => false,
            })
    }
}

/// One slot's value at one input.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotValue {
    Token(u32),
    Raw(Array1<f64>),
}

/// One input: a value per slot.
pub type Input = Vec<SlotValue>;

/// Row `row` of a family.
pub fn input_at(family: &FamilyInputs, row: usize) -> Input {
    family
        .slots
        .iter()
        .map(|slot| match slot {
            SlotValues::Tokens(tokens) => SlotValue::Token(tokens[row]),
            SlotValues::Raw(rows) => SlotValue::Raw(rows.row(row).to_owned()),
        })
        .collect()
}

/// The family of `inputs`, in order; `inputs` must share one slot layout.
pub fn family_of(inputs: &[Input]) -> Result<FamilyInputs, CegarError> {
    let first = inputs.first().ok_or_else(|| CegarError::Domain("an empty family".to_string()))?;
    let slots = (0..first.len())
        .map(|j| match &first[j] {
            SlotValue::Token(_) => inputs
                .iter()
                .map(|input| match &input[j] {
                    SlotValue::Token(t) => Ok(*t),
                    SlotValue::Raw(_) => Err(CegarError::Domain(format!("slot {j} mixes tokens and raw rows"))),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SlotValues::Tokens),
            SlotValue::Raw(x) => {
                let mut rows = Array2::<f64>::zeros((inputs.len(), x.len()));
                for (r, input) in inputs.iter().enumerate() {
                    match &input[j] {
                        SlotValue::Raw(v) if v.len() == x.len() => rows.row_mut(r).assign(v),
                        _ => return Err(CegarError::Domain(format!("slot {j} mixes widths or kinds"))),
                    }
                }
                Ok(SlotValues::Raw(rows))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FamilyInputs { rows: inputs.len(), slots, layout: None })
}

/// A total order key of an input (raw values by their bits), for deduplication.
fn key(input: &Input) -> Vec<u64> {
    let mut out = Vec::new();
    for value in input {
        match value {
            SlotValue::Token(t) => out.push(u64::from(*t)),
            SlotValue::Raw(x) => out.extend(x.iter().map(|v| v.to_bits())),
        }
    }
    out
}

/// A refused refinement.
#[derive(Debug)]
pub enum CegarError {
    Program(ProgramError),
    Domain(String),
    Bound(String),
    Fit(String),
}

impl fmt::Display for CegarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(f, "cegar: {error}"),
            Self::Domain(message) => write!(f, "cegar: domain: {message}"),
            Self::Bound(message) => write!(f, "cegar: bound: {message}"),
            Self::Fit(message) => write!(f, "cegar: fit: {message}"),
        }
    }
}

impl std::error::Error for CegarError {}

impl From<ProgramError> for CegarError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

// ------------------------------------------------------------------------------ the objective

/// One input's `Σ_readouts KL(model ‖ program)`, certified: the computed value, and a proven
/// lower and upper end (`+∞` when a row is unresolved).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputKl {
    pub value: f64,
    pub lower: f64,
    pub upper: f64,
}

/// The certified KL of every input of `family`: both programs executed with bands, each readout
/// row compared over its logit boxes.
pub fn certified_kl(
    model: &OperatorProgram,
    program: &OperatorProgram,
    family: &FamilyInputs,
    readouts: usize,
) -> Result<Vec<InputKl>, CegarError> {
    let reference = model.execute(family, true)?.banded(model.output);
    let candidate = program.execute(family, true)?.banded(program.output);
    let width = reference.values.ncols();
    if candidate.values.ncols() != width || width % readouts != 0 {
        return Err(CegarError::Domain(format!("outputs {width} and {} for {readouts} readouts", candidate.values.ncols())));
    }
    let classes = width / readouts;
    let growth = accumulation_growth(readouts + 1);
    (0..family.rows)
        .map(|row| {
            let (mut value, mut error, mut lower, mut resolved) = (0.0_f64, 0.0_f64, 0.0_f64, true);
            for k in 0..readouts {
                let range = k * classes..(k + 1) * classes;
                let comparison = compare_logit_row(
                    reference.values.slice(s![row, range.clone()]),
                    reference.bands.slice(s![row, range.clone()]),
                    candidate.values.slice(s![row, range.clone()]),
                    candidate.bands.slice(s![row, range]),
                )
                .map_err(|e| CegarError::Bound(format!("{e:?}")))?;
                match RowValue::of(&comparison.forward_kl) {
                    RowValue::Resolved { value: v, numerical_error } => {
                        value += v;
                        error += numerical_error;
                        lower += (v - numerical_error).max(0.0);
                    }
                    RowValue::Unresolved { lower: l } => {
                        resolved = false;
                        value += l;
                        lower += l;
                    }
                }
            }
            let band = (error + growth * value.abs()).next_up();
            Ok(InputKl {
                value,
                lower: (lower - growth * lower).max(0.0).next_down().max(0.0),
                upper: if resolved { (value + band).next_up() } else { f64::INFINITY },
            })
        })
        .collect()
}

/// The unbanded KL of each input and its gradient with respect to each program's output.
fn kl_and_output_gradients(
    model_logits: &Array2<f64>,
    program_logits: &Array2<f64>,
    readouts: usize,
) -> (Vec<f64>, Array2<f64>, Array2<f64>) {
    let (rows, width) = model_logits.dim();
    let classes = width / readouts;
    let mut values = vec![0.0; rows];
    let mut grad_model = Array2::<f64>::zeros((rows, width));
    let mut grad_program = Array2::<f64>::zeros((rows, width));
    let log_softmax = |z: &[f64]| {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
        z.iter().map(|v| v - lse).collect::<Vec<f64>>()
    };
    for row in 0..rows {
        for k in 0..readouts {
            let range = k * classes..(k + 1) * classes;
            let z: Vec<f64> = model_logits.slice(s![row, range.clone()]).to_vec();
            let w: Vec<f64> = program_logits.slice(s![row, range.clone()]).to_vec();
            let (lp, lq) = (log_softmax(&z), log_softmax(&w));
            let kl: f64 = lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum();
            values[row] += kl;
            for (c, col) in range.enumerate() {
                let (p, q) = (lp[c].exp(), lq[c].exp());
                // d KL / d z_c = p_c (log p_c − log q_c − KL); d KL / d w_c = q_c − p_c.
                grad_model[[row, col]] = p * (lp[c] - lq[c] - kl);
                grad_program[[row, col]] = q - p;
            }
        }
    }
    (values, grad_model, grad_program)
}

// ---------------------------------------------------------------------------- reverse mode

/// The gradient of `Σ_rows ⟨seed_row, output_row⟩` with respect to every slot's relaxed input:
/// for a token slot `(rows × domain size)`, the derivative along each token's one-hot; for a raw
/// slot `(rows × width)`, the derivative along each coordinate. Slots the program does not read
/// are `None`. Every node's cotangent is [`vjp`]'s on the program's unbanded `trace` of `family`;
/// `program` has no rules ([`inlined`]). A basis is linear in the one-hot, so a feature node's
/// cotangent `G` reaches its slot as `G Φᵀ`.
pub fn backward(
    program: &OperatorProgram,
    family: &FamilyInputs,
    trace: &Trace,
    seed: &Array2<f64>,
) -> Result<Vec<Option<Array2<f64>>>, CegarError> {
    let cotangents = vjp(program, family, trace, seed.clone())?;
    let mut slots: Vec<Option<Array2<f64>>> = vec![None; program.declarations.slots.len()];
    for (node, cotangent) in program.nodes.iter().zip(cotangents) {
        let Some(cotangent) = cotangent else { continue };
        let (slot, delta) = match node {
            Node::Feature { slot, basis } => {
                let size = program.declarations.domains[basis_domain(program, *basis)].size;
                let classes: Vec<u32> = (0..size as u32).collect();
                let phi = program.bases[*basis].evaluate(&program.declarations, &classes)?.values;
                (*slot, cotangent.dot(&phi.t()))
            }
            Node::Raw { slot } => (*slot, cotangent),
            _ => continue,
        };
        match &mut slots[slot] {
            Some(existing) => *existing += &delta,
            empty => *empty = Some(delta),
        }
    }
    Ok(slots)
}

/// `program` with every rule application replaced by a copy of the rule's body, its parameters
/// bound to the call's arguments: the same function, node for node, with no rules.
pub fn inlined(program: &OperatorProgram) -> Result<OperatorProgram, CegarError> {
    let identity = |n: usize| (0..n).collect::<Vec<usize>>();
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
    ) -> Result<usize, CegarError> {
        let mut map: Vec<usize> = Vec::with_capacity(body.len());
        for node in body {
            let index = match node {
                Node::Param { index } => *arguments
                    .get(*index)
                    .ok_or_else(|| CegarError::Domain(format!("a rule parameter {index} with {} arguments", arguments.len())))?,
                Node::Call { rule, arguments: call } => {
                    let rule = program.rules.get(*rule).ok_or_else(|| CegarError::Domain(format!("no rule {rule}")))?;
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

fn basis_domain(program: &OperatorProgram, basis: usize) -> usize {
    match &program.bases[basis] {
        super::operator_program::Basis::Indicator { domain } | super::operator_program::Basis::Characters { domain, .. } => {
            *domain
        }
    }
}

/// The KL of each input of `family` (unbanded) and its gradient with respect to each slot's
/// relaxed input, through both programs.
pub fn kl_gradients(
    model: &OperatorProgram,
    program: &OperatorProgram,
    family: &FamilyInputs,
    readouts: usize,
) -> Result<(Vec<f64>, Vec<Option<Array2<f64>>>), CegarError> {
    let (model, program) = (&inlined(model)?, &inlined(program)?);
    let model_trace = model.execute(family, false)?;
    let program_trace = program.execute(family, false)?;
    let (kl, grad_model, grad_program) =
        kl_and_output_gradients(&model_trace.values[model.output], &program_trace.values[program.output], readouts);
    let mut slots = backward(model, family, &model_trace, &grad_model)?;
    for (slot, delta) in slots.iter_mut().zip(backward(program, family, &program_trace, &grad_program)?) {
        if let Some(delta) = delta {
            match slot {
                Some(existing) => *existing += &delta,
                empty => *empty = Some(delta),
            }
        }
    }
    Ok((kl, slots))
}

// ------------------------------------------------------------------------------ the ascent

/// Why one ascent stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AscentStop {
    /// No move has a positive linearised gain: the Frank–Wolfe gap is zero.
    Stationary,
    /// No candidate certified an increase over the current value.
    NoCertifiedIncrease,
}

/// One ascent from one start.
#[derive(Clone, Debug, PartialEq)]
pub struct Ascent {
    pub start: usize,
    /// The start's certified KL.
    pub start_kl: InputKl,
    /// The endpoint: the last accepted input.
    pub input: Input,
    pub kl: InputKl,
    /// The computed KL after each accepted step, starting with the start's.
    pub path: Vec<f64>,
    /// Candidate inputs evaluated natively, including rejected ones.
    pub evaluations: usize,
    pub stop: AscentStop,
}

/// The candidates of one step at `input` with slot gradients `gradients` (the row's).
fn candidates(domain: &InputDomain, input: &Input, gradients: &[Option<Array1<f64>>]) -> Vec<Input> {
    let mut moves: Vec<(f64, usize, u32)> = Vec::new();
    // Single-coordinate flips of corner slots: (gain, slot, coordinate).
    let mut flips: Vec<(f64, usize, usize)> = Vec::new();
    let mut vertex = input.clone();
    let mut vertex_moves = 0usize;
    let mut box_direction: Vec<Option<(Array1<f64>, Array1<f64>)>> = vec![None; input.len()];
    for (j, (slot, value)) in domain.slots.iter().zip(input).enumerate() {
        let Some(g) = &gradients[j] else { continue };
        match (slot, value) {
            (SlotDomain::Tokens(tokens), SlotValue::Token(current)) => {
                let here = g[*current as usize];
                let mut best: Option<(f64, u32)> = None;
                for &t in tokens {
                    let gain = g[t as usize] - here;
                    if t != *current && gain > 0.0 {
                        moves.push((gain, j, t));
                        if best.is_none_or(|(b, _)| gain > b) {
                            best = Some((gain, t));
                        }
                    }
                }
                if let Some((_, t)) = best {
                    vertex[j] = SlotValue::Token(t);
                    vertex_moves += 1;
                }
            }
            (SlotDomain::Corners { lower, upper }, SlotValue::Raw(x)) => {
                let mut corner = x.clone();
                let mut flipped = false;
                for i in 0..x.len() {
                    let other = if x[i] == lower[i] { upper[i] } else { lower[i] };
                    let gain = g[i] * (other - x[i]);
                    if gain > 0.0 {
                        flips.push((gain, j, i));
                        corner[i] = other;
                        flipped = true;
                    }
                }
                if flipped {
                    vertex[j] = SlotValue::Raw(corner);
                }
            }
            (SlotDomain::Box { lower, upper }, SlotValue::Raw(x)) => {
                let target = Array1::from_shape_fn(x.len(), |i| if g[i] > 0.0 { upper[i] } else if g[i] < 0.0 { lower[i] } else { x[i] });
                let gain: f64 = g.iter().zip(target.iter().zip(x)).map(|(gi, (s, xi))| gi * (s - xi)).sum();
                if gain > 0.0 {
                    box_direction[j] = Some((x.clone(), target));
                }
            }
            _ => continue,
        }
    }
    moves.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    // The oracle vertex is a candidate of its own when it changes more than one coordinate.
    let mut out = Vec::new();
    if vertex_moves + flips.len() > 1 {
        out.push(vertex);
    }
    for (_, j, t) in moves {
        let mut next = input.clone();
        next[j] = SlotValue::Token(t);
        out.push(next);
    }
    flips.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    for (_, j, i) in flips {
        let mut next = input.clone();
        if let (SlotValue::Raw(x), SlotDomain::Corners { lower, upper }) = (&mut next[j], &domain.slots[j]) {
            x[i] = if x[i] == lower[i] { upper[i] } else { lower[i] };
        }
        out.push(next);
    }
    // The box step toward the oracle vertex at γ = 2^{-k}, until the step leaves every float in place.
    if box_direction.iter().any(Option::is_some) {
        let mut gamma = 1.0_f64;
        loop {
            let mut next = input.clone();
            let mut moved = false;
            for (j, direction) in box_direction.iter().enumerate() {
                if let Some((x, target)) = direction {
                    let y = Array1::from_shape_fn(x.len(), |i| x[i] + gamma * (target[i] - x[i]));
                    moved |= y.iter().zip(x).any(|(a, b)| a != b);
                    next[j] = SlotValue::Raw(y);
                }
            }
            if !moved {
                break;
            }
            out.push(next);
            gamma *= 0.5;
        }
    }
    out
}

/// Ascends `KL(model ‖ program)` from every start over `domain` (module note, *The verifier*).
pub fn ascend(
    model: &OperatorProgram,
    program: &OperatorProgram,
    domain: &InputDomain,
    starts: &[Input],
    readouts: usize,
) -> Result<Vec<Ascent>, CegarError> {
    for (index, start) in starts.iter().enumerate() {
        if !domain.contains(start) {
            return Err(CegarError::Domain(format!("start {index} is outside the declared domain")));
        }
    }
    if starts.is_empty() {
        return Ok(Vec::new());
    }
    let first = certified_kl(model, program, &family_of(starts)?, readouts)?;
    let mut ascents: Vec<Ascent> = starts
        .iter()
        .zip(first)
        .enumerate()
        .map(|(start, (input, kl))| Ascent {
            start,
            start_kl: kl,
            input: input.clone(),
            kl,
            path: vec![kl.value],
            evaluations: 1,
            stop: AscentStop::Stationary,
        })
        .collect();
    let mut active: Vec<usize> = (0..ascents.len()).collect();
    while !active.is_empty() {
        let current: Vec<Input> = active.iter().map(|&a| ascents[a].input.clone()).collect();
        let (_, gradients) = kl_gradients(model, program, &family_of(&current)?, readouts)?;
        let mut batch: Vec<Input> = Vec::new();
        let mut owners: Vec<usize> = Vec::new();
        let mut next_active = Vec::new();
        for (k, &a) in active.iter().enumerate() {
            let row: Vec<Option<Array1<f64>>> = gradients.iter().map(|g| g.as_ref().map(|g| g.row(k).to_owned())).collect();
            let proposals = candidates(domain, &ascents[a].input, &row);
            if proposals.is_empty() {
                ascents[a].stop = AscentStop::Stationary;
                continue;
            }
            owners.extend(std::iter::repeat_n(a, proposals.len()));
            batch.extend(proposals);
            next_active.push(a);
        }
        if batch.is_empty() {
            break;
        }
        let kls = certified_kl(model, program, &family_of(&batch)?, readouts)?;
        let mut best: Vec<Option<usize>> = vec![None; ascents.len()];
        for (c, &a) in owners.iter().enumerate() {
            ascents[a].evaluations += 1;
            if kls[c].lower > ascents[a].kl.value && best[a].is_none_or(|b| kls[c].lower > kls[b].lower) {
                best[a] = Some(c);
            }
        }
        active.clear();
        for a in next_active {
            match best[a] {
                Some(c) => {
                    ascents[a].input = batch[c].clone();
                    ascents[a].kl = kls[c];
                    ascents[a].path.push(kls[c].value);
                    active.push(a);
                }
                None => ascents[a].stop = AscentStop::NoCertifiedIncrease,
            }
        }
    }
    Ok(ascents)
}

/// The worst computed KL over all ascents after `steps` accepted steps each (an ascent that
/// stopped earlier contributes its endpoint): the adversary's ladder.
pub fn ladder(ascents: &[Ascent], steps: &[usize]) -> Vec<(usize, f64)> {
    steps
        .iter()
        .map(|&k| (k, ascents.iter().map(|a| a.path[k.min(a.path.len() - 1)]).fold(0.0, f64::max)))
        .collect()
}

/// The exact supremum of the KL over an enumerable domain: every input evaluated with bands, in
/// batches of `batch` inputs. Returns the worst input and its certified KL, and the largest upper
/// end over the domain.
pub fn exhaustive_worst(
    model: &OperatorProgram,
    program: &OperatorProgram,
    domain: &InputDomain,
    readouts: usize,
    batch: usize,
) -> Result<(Input, InputKl, f64), CegarError> {
    // Each slot as a list of its values; a corner slot lists its 2^width corners.
    let sets: Vec<Vec<SlotValue>> = domain
        .slots
        .iter()
        .map(|slot| match slot {
            SlotDomain::Tokens(tokens) if !tokens.is_empty() => Ok(tokens.iter().map(|t| SlotValue::Token(*t)).collect()),
            SlotDomain::Corners { lower, upper } if lower.len() < 32 => Ok((0..1u64 << lower.len())
                .map(|bits| {
                    SlotValue::Raw(Array1::from_shape_fn(lower.len(), |i| if (bits >> i) & 1 == 1 { upper[i] } else { lower[i] }))
                })
                .collect()),
            _ => Err(CegarError::Domain("an exhaustive search needs nonempty token or corner slots".to_string())),
        })
        .collect::<Result<_, _>>()?;
    let total = domain.cardinality().ok_or_else(|| CegarError::Domain("the domain does not enumerate".to_string()))?;
    let mut worst: Option<(Input, InputKl)> = None;
    let mut sup_upper = 0.0_f64;
    let mut index = 0u64;
    while index < total {
        let end = (index + batch.max(1) as u64).min(total);
        let inputs: Vec<Input> = (index..end)
            .map(|mut i| {
                let mut input = vec![SlotValue::Token(0); sets.len()];
                for (j, set) in sets.iter().enumerate().rev() {
                    input[j] = set[(i % set.len() as u64) as usize].clone();
                    i /= set.len() as u64;
                }
                input
            })
            .collect();
        for (input, kl) in inputs.iter().zip(certified_kl(model, program, &family_of(&inputs)?, readouts)?) {
            sup_upper = sup_upper.max(kl.upper);
            if worst.as_ref().is_none_or(|(_, w)| kl.value > w.value) {
                worst = Some((input.clone(), kl));
            }
        }
        index = end;
    }
    let (input, kl) = worst.ok_or_else(|| CegarError::Domain("an empty domain".to_string()))?;
    Ok((input, kl, sup_upper))
}

// ------------------------------------------------------------------------------ the loop

/// One verifier round.
#[derive(Clone, Debug)]
pub struct Round {
    pub rows: usize,
    /// The largest certified upper end of the KL over the data's rows.
    pub data_worst_upper: f64,
    /// The largest computed KL any ascent reached.
    pub ascent_worst: f64,
    pub counterexamples: usize,
    pub evaluations: usize,
    pub ladder: Vec<(usize, f64)>,
}

/// What one verifier round found.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub round: Round,
    /// Endpoints certified worse than every row of the data, none of them a row, each once.
    pub counterexamples: Vec<Input>,
    pub ascents: Vec<Ascent>,
}

/// The ladder rungs every round reports: the step counts VPD's adversaries report, and the ends.
pub const LADDER: [usize; 7] = [0, 1, 2, 5, 20, 40, 100];

/// One verifier round against `program`: ascents from every row of `data` and every input of
/// `pool`; the data's certified worst is the largest upper end at the data's rows, and every
/// endpoint whose certified lower end exceeds it is a counterexample.
pub fn verify(
    model: &OperatorProgram,
    program: &OperatorProgram,
    domain: &InputDomain,
    data: &FamilyInputs,
    pool: &[Input],
    readouts: usize,
) -> Result<Verdict, CegarError> {
    if data.layout.is_some() {
        return Err(CegarError::Domain(
            "a per-position family: the verifier's inputs are single rows, a sequence spans several".to_string(),
        ));
    }
    let rows: Vec<Input> = (0..data.rows).map(|row| input_at(data, row)).collect();
    let mut seen: BTreeSet<Vec<u64>> = rows.iter().map(key).collect();
    let mut starts = rows;
    starts.extend(pool.iter().filter(|input| !seen.contains(&key(input))).cloned());
    let ascents = ascend(model, program, domain, &starts, readouts)?;
    let data_worst_upper = ascents[..data.rows].iter().map(|a| a.start_kl.upper).fold(0.0, f64::max);
    let mut counterexamples = Vec::new();
    for ascent in &ascents {
        if ascent.kl.lower > data_worst_upper && seen.insert(key(&ascent.input)) {
            counterexamples.push(ascent.input.clone());
        }
    }
    let round = Round {
        rows: data.rows,
        data_worst_upper,
        ascent_worst: ascents.iter().map(|a| a.kl.value).fold(0.0, f64::max),
        counterexamples: counterexamples.len(),
        evaluations: ascents.iter().map(|a| a.evaluations).sum(),
        ladder: ladder(&ascents, &LADDER),
    };
    Ok(Verdict { round, counterexamples, ascents })
}
