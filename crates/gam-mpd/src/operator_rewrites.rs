//! Exact rewrites of an operator program (#2951): algebraic identities, each a [`Primitive`]
//! the engine accepts only when the decoded message gets shorter.
//!
//! * [`FoldConstants`]: a node whose value is the same on every input of the declared family is
//!   that constant on the family. The folded constant is the computed row, so the rewrite is exact
//!   for the executed program up to that node's band.
//! * [`BilinearConstantSide`]: `c Σ_i l_i r_i` with `l` a constant is the affine map `r ↦ (c l)ᵀ r`.
//! * [`ComposeAffine`]: an affine node read by an affine term, `A (Σ_t B_t z_t + b)`, is
//!   `Σ_t (A B_t) z_t + A b`. A composed block that is exactly zero is absent.
//! * [`PushThroughMix`]: an affine map of a routed mix whose payloads share one affine map,
//!   `A Σ_j α_j (B z_j + b_j) = (A B) Σ_j α_j z_j + Σ_j α_j (A b_j)`: the mix moves the inputs
//!   `z_j`, and the payload biases become a term read from the routing weights themselves.
//! * [`PlaneBasis`]: an indicator basis `e_t` on a domain is `Φ⁻¹` times the
//!   character basis of a single odd cycle on it, for ANY labelling of the cycle's tokens, so every
//!   operator reading the indicator features becomes `A Φ⁻¹` (a table's discrete Fourier transform
//!   over the labelling), and a readout over the classes becomes `Φᵀ` applied to `Φ⁻ᵀ A`. With
//!   positions `a(t)`, `p` the period and `ω_k = 2πk/p`, `Φ⁻¹` has rows `(1/p, (2/p) cos ω_k a(t),
//!   (2/p) sin ω_k a(t))` on the cycle and the identity off it (the orthogonality of the characters
//!   of `Z_p`, `p` odd). [`PlaneBasis`] reads the labelling from the weights with nothing named:
//!   the tokens' table (every operator reading the domain, stacked) gives the projector onto each
//!   leading singular subspace its spectrum resolves ([`TokenOperators::leading_subspaces`]),
//!   [`discover_permutations`] finds each projector's gap-certified symmetry group, and each
//!   group's canonical odd cycle ([`super::symmetry::PermutationGroup::cycle_positions`]) is a
//!   labelling. Every distinct labelling is proposed: the rewrite is exact for any labelling,
//!   the engine's code length picks the subspace (and says whether any labelling makes the
//!   table short), and the labelling is sent in full.
//!   [`change_basis`] is the rewrite itself, for a labelling from anywhere: a contract that declares
//!   a cycle (a prior, sent as one bit) is a caller's choice, not a library move.

use super::engine::{EngineError, Exactness, Edit, Primitive, Proposal, SearchContext};
use super::fit::ProposalKind;
use super::symmetry::{SymmetryError, TokenOperators, discover_permutations};
use super::precision::DeclaredPrecision;
use super::operator_program::{
    Basis, Interface, Node, Operator, OperatorBody, OperatorProgram, Provenance, band_precision, exact_precision,
    remap_node,
};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use ndarray::{Array2, Axis, s};
use std::collections::BTreeSet;
use std::f64::consts::TAU;
use std::sync::Arc;

fn exact(derivation: String) -> Exactness {
    Exactness::Exact { derivation }
}

fn structural(primitive: &'static str, kind: ProposalKind, derivation: String, description: String, program: OperatorProgram) -> Proposal {
    Proposal { primitive, kind, exactness: exact(derivation), description, edit: Edit::Program(Box::new(program)) }
}

/// Insert `node` at position `at`, shifting every later node reference; returns `at`.
pub fn insert_node(program: &mut OperatorProgram, at: usize, node: Node) -> usize {
    let map: Vec<usize> = (0..program.nodes.len()).map(|i| if i < at { i } else { i + 1 }).collect();
    let identity_ops: Vec<usize> = (0..program.operators.len()).collect();
    let identity_bases: Vec<usize> = (0..program.bases.len()).collect();
    let identity_rules: Vec<usize> = (0..program.rules.len()).collect();
    for existing in program.nodes.iter_mut() {
        remap_node(existing, &map, &identity_ops, &identity_bases, &identity_rules);
    }
    program.output = map[program.output];
    program.nodes.insert(at, node);
    at
}

/// A dense operator whose exactly-zero blocks are absent, on the lattice that resolves `band`.
pub fn product_operator(
    name: String,
    rows: Interface,
    cols: Interface,
    values: Array2<f64>,
    band: f64,
    provenance: Provenance,
) -> Result<Operator, EngineError> {
    let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let precision = band_precision(band, largest)?;
    let mut present = Array2::from_elem((rows.group_count(), cols.group_count()), false);
    for r in 0..rows.group_count() {
        for c in 0..cols.group_count() {
            present[[r, c]] = values.slice(s![rows.range(r), cols.range(c)]).iter().any(|v| *v != 0.0);
        }
    }
    Ok(Operator::blocks(name, rows, cols, values, present, precision, provenance)?)
}

/// Half the lattice step of an operator's reals: how far each real may sit from the value it
/// stands for. Resolving a product beyond what its factors resolve carries no information.
fn half_step(op: &Operator) -> f64 {
    match &op.body {
        OperatorBody::Dense { precision, .. } | OperatorBody::LowRank { precision, .. } => precision.worst_case_error(),
        OperatorBody::Identity => 0.0,
    }
}

/// The largest absolute row sum of `m`.
fn row_sum(m: &Array2<f64>) -> f64 {
    m.outer_iter().map(|row| row.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// `A B` with its band: the rounding `γ_k max(|A||B|)` plus what the factors' own lattices leave
/// unresolved, `δ_A max_j Σ_k |B_kj| + δ_B max_i Σ_k |A_ik|`.
fn product(a: &Operator, b: &Operator) -> (Array2<f64>, f64) {
    let (am, bm) = (a.matrix(), b.matrix());
    let value = am.dot(&bm);
    let magnitude = am.mapv(f64::abs).dot(&bm.mapv(f64::abs));
    let rounding = accumulation_growth(am.ncols()) * magnitude.iter().fold(0.0_f64, |m, v| m.max(*v));
    let resolution = half_step(a) * row_sum(&bm.t().to_owned()) + half_step(b) * row_sum(&am);
    (value, rounding + resolution)
}

/// Nodes whose value is identical on every input.
fn constant_nodes(program: &OperatorProgram, context: &SearchContext<'_>) -> Vec<bool> {
    context
        .trace
        .values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            !matches!(program.nodes[index], Node::Constant { .. })
                && value.nrows() > 1
                && value.outer_iter().all(|row| row.iter().zip(value.row(0).iter()).all(|(a, b)| a.to_bits() == b.to_bits()))
        })
        .collect()
}

/// Fold input-invariant nodes into constants.
pub struct FoldConstants;

impl Primitive for FoldConstants {
    fn name(&self) -> &'static str {
        "fold_constants"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let constant = constant_nodes(program, context);
        let mut frontier = Vec::new();
        for (index, &is_constant) in constant.iter().enumerate() {
            if !is_constant {
                continue;
            }
            let read_by_varying = program
                .nodes
                .iter()
                .enumerate()
                .any(|(reader, node)| node.arguments().contains(&index) && !constant[reader]);
            if read_by_varying || index == program.output {
                frontier.push(index);
            }
        }
        let fold = |targets: &[usize]| -> Result<OperatorProgram, EngineError> {
            let mut candidate = program.clone();
            for &index in targets {
                let row = context.trace.values[index].row(0).to_owned();
                let precision = exact_precision(row.iter().copied())?;
                let operator = Operator::dense(
                    format!("const{index}"),
                    context.interfaces[index].clone(),
                    Interface::constant(),
                    row.insert_axis(Axis(1)),
                    precision,
                    Provenance::derived(&[], format!("node {index} is constant on the family")),
                )?;
                candidate.operators.push(Arc::new(operator));
                candidate.nodes[index] = Node::Constant { operator: candidate.operators.len() - 1 };
            }
            candidate.prune();
            Ok(candidate)
        };
        let mut out = Vec::new();
        for &index in &frontier {
            out.push(structural(
                "fold_constants",
                ProposalKind::Reduce,
                format!("node {index} takes one value on every input of the family"),
                format!("fold constant node {index}"),
                fold(&[index])?,
            ));
        }
        if frontier.len() > 1 {
            out.push(structural(
                "fold_constants",
                ProposalKind::Reduce,
                "every frontier node takes one value on every input of the family".to_string(),
                format!("fold constant nodes {frontier:?}"),
                fold(&frontier)?,
            ));
        }
        Ok(out)
    }
}

/// A bilinear score with one constant side is affine in the other.
pub struct BilinearConstantSide;

impl Primitive for BilinearConstantSide {
    fn name(&self) -> &'static str {
        "bilinear_constant_side"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let Node::Bilinear { left, right, scale } = node else { continue };
            let side = match (&program.nodes[*left], &program.nodes[*right]) {
                (Node::Constant { operator }, _) => Some((*operator, *right)),
                (_, Node::Constant { operator }) => Some((*operator, *left)),
                _ => None,
            };
            let Some((constant, varying)) = side else { continue };
            let c = scale.value();
            let column = program.operators[constant].matrix();
            let row = column.t().mapv(|v| c * v);
            let band = UNIT_ROUNDOFF * row.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let mut candidate = program.clone();
            let operator = product_operator(
                format!("score{index}"),
                context.interfaces[index].clone(),
                context.interfaces[varying].clone(),
                row,
                band,
                Provenance::derived(&[&program.operators[constant].provenance], format!("scale {c} times the constant side")),
            )?;
            candidate.operators.push(Arc::new(operator));
            candidate.nodes[index] = Node::Affine { terms: vec![(varying, candidate.operators.len() - 1)], bias: None };
            candidate.prune();
            out.push(structural(
                "bilinear_constant_side",
                ProposalKind::Share,
                "c <l, r> = (c l)^T r for a constant l".to_string(),
                format!("bilinear node {index} with a constant side is affine"),
                candidate,
            ));
        }
        Ok(out)
    }
}

/// Compose an affine term with the affine node it reads.
pub struct ComposeAffine;

impl Primitive for ComposeAffine {
    fn name(&self) -> &'static str {
        "compose_affine"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let Node::Affine { terms, bias } = node else { continue };
            for (term_index, (argument, outer)) in terms.iter().enumerate() {
                let Node::Affine { terms: inner_terms, bias: inner_bias } = &program.nodes[*argument] else { continue };
                let mut candidate = program.clone();
                let a = &program.operators[*outer];
                let mut new_terms: Vec<(usize, usize)> =
                    terms.iter().enumerate().filter(|(i, _)| *i != term_index).map(|(_, t)| *t).collect();
                for (z, inner) in inner_terms {
                    let b = &program.operators[*inner];
                    let (values, band) = product(a, b);
                    let operator = product_operator(
                        format!("{}·{}", a.name, b.name),
                        a.rows.clone(),
                        b.cols.clone(),
                        values,
                        band,
                        Provenance::derived(&[&a.provenance, &b.provenance], format!("{} {}", a.name, b.name)),
                    )?;
                    candidate.operators.push(Arc::new(operator));
                    new_terms.push((*z, candidate.operators.len() - 1));
                }
                let mut new_bias = *bias;
                if let Some(inner_bias) = inner_bias {
                    let b = &program.operators[*inner_bias];
                    let (mut values, mut band) = product(a, b);
                    if let Some(outer_bias) = bias {
                        let existing = program.operators[*outer_bias].matrix();
                        band += UNIT_ROUNDOFF * (values.iter().fold(0.0_f64, |m, v| m.max(v.abs())) + existing.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
                        values += &existing;
                    }
                    let operator = product_operator(
                        format!("bias{index}"),
                        a.rows.clone(),
                        Interface::constant(),
                        values,
                        band,
                        Provenance::derived(&[&a.provenance, &b.provenance], format!("{} {} (+ bias)", a.name, b.name)),
                    )?;
                    candidate.operators.push(Arc::new(operator));
                    new_bias = Some(candidate.operators.len() - 1);
                }
                candidate.nodes[index] = Node::Affine { terms: new_terms, bias: new_bias };
                candidate.prune();
                out.push(structural(
                    "compose_affine",
                    ProposalKind::Share,
                    "A (sum_t B_t z_t + b) = sum_t (A B_t) z_t + A b".to_string(),
                    format!("compose term {term_index} of node {index} through node {argument}"),
                    candidate,
                ));
            }
        }
        Ok(out)
    }
}

/// Push an affine map through a routed mix whose payloads share one affine map.
pub struct PushThroughMix;

impl Primitive for PushThroughMix {
    fn name(&self) -> &'static str {
        "push_through_mix"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let Node::Affine { terms, bias } = node else { continue };
            for (term_index, (argument, outer)) in terms.iter().enumerate() {
                let Node::Mix { weights, payloads } = &program.nodes[*argument] else { continue };
                let mut shared: Option<usize> = None;
                let mut moved: Vec<(usize, usize)> = Vec::new();
                let mut biases: Vec<(usize, usize)> = Vec::new();
                let mut fits = true;
                for &(column, payload) in payloads {
                    match &program.nodes[payload] {
                        Node::Affine { terms: inner, bias: inner_bias } if inner.len() == 1 => {
                            let (z, b) = inner[0];
                            fits &= shared.is_none_or(|s| s == b);
                            shared = Some(b);
                            moved.push((column, z));
                            if let Some(inner_bias) = inner_bias {
                                biases.push((column, *inner_bias));
                            }
                        }
                        Node::Constant { operator } => biases.push((column, *operator)),
                        _ => fits = false,
                    }
                }
                let Some(shared) = shared else { continue };
                if !fits {
                    continue;
                }
                let a = &program.operators[*outer];
                let b = &program.operators[shared];
                let mut candidate = program.clone();
                let (values, band) = product(a, b);
                candidate.operators.push(Arc::new(product_operator(
                    format!("{}·{}", a.name, b.name),
                    a.rows.clone(),
                    b.cols.clone(),
                    values,
                    band,
                    Provenance::derived(&[&a.provenance, &b.provenance], format!("{} {} through the mix", a.name, b.name)),
                )?));
                let moved_op = candidate.operators.len() - 1;
                let weight_interface = context.interfaces[*weights].clone();
                let mut routed = Array2::<f64>::zeros((a.rows.width(), weight_interface.width()));
                let mut routed_band = 0.0_f64;
                let mut parts = vec![&a.provenance];
                for &(column, operator) in &biases {
                    let bias_op = &program.operators[operator];
                    let (values, band) = product(a, bias_op);
                    routed.column_mut(column).assign(&values.column(0));
                    routed_band = routed_band.max(band);
                    parts.push(&bias_op.provenance);
                }
                let new_mix = Node::Mix { weights: *weights, payloads: moved };
                let mix_index = insert_node(&mut candidate, index, new_mix);
                let y = index + 1;
                let Node::Affine { terms: shifted, bias: shifted_bias } = candidate.nodes[y].clone() else {
                    return Err(EngineError::Primitive("the affine node moved under insertion".to_string()));
                };
                let mut new_terms: Vec<(usize, usize)> =
                    shifted.iter().enumerate().filter(|(i, _)| *i != term_index).map(|(_, t)| *t).collect();
                new_terms.push((mix_index, moved_op));
                if routed.iter().any(|v| *v != 0.0) {
                    let weights_shifted = if *weights >= index { *weights + 1 } else { *weights };
                    candidate.operators.push(Arc::new(product_operator(
                        format!("{}·bias", a.name),
                        a.rows.clone(),
                        weight_interface,
                        routed,
                        routed_band,
                        Provenance::derived(&parts, format!("{} times each payload's bias, read by its routing weight", a.name)),
                    )?));
                    new_terms.push((weights_shifted, candidate.operators.len() - 1));
                }
                if shifted_bias != *bias {
                    return Err(EngineError::Primitive("an inserted node moved an operator reference".to_string()));
                }
                candidate.nodes[y] = Node::Affine { terms: new_terms, bias: shifted_bias };
                candidate.prune();
                out.push(structural(
                    "push_through_mix",
                    ProposalKind::Share,
                    "A sum_j a_j (B z_j + b_j) = (A B) sum_j a_j z_j + sum_j a_j (A b_j)".to_string(),
                    format!("push term {term_index} of node {index} through mix node {argument}"),
                    candidate,
                ));
            }
        }
        Ok(out)
    }
}

/// Stack the terms of an affine node into one operator over the concatenation of their inputs:
/// `Σ_t A_t x_t = [A_1 … A_T] [x_1; …; x_T]`. Exact; it pays one operator's structure instead of
/// `T`, and it lets a restriction or a factorization see the terms together (one unit's reads of a
/// plane in every head are one row of one operator).
pub struct StackTerms;

impl Primitive for StackTerms {
    fn name(&self) -> &'static str {
        "stack_terms"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut uses = vec![0usize; program.operators.len()];
        for node in &program.nodes {
            for op in node.operators() {
                uses[op] += 1;
            }
        }
        let mut out = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let Node::Affine { terms, bias } = node else { continue };
            let stackable: Vec<(usize, usize)> = terms
                .iter()
                .copied()
                .filter(|(_, op)| uses[*op] == 1 && matches!(program.operators[*op].body, OperatorBody::Dense { .. }))
                .collect();
            if stackable.len() < 2 {
                continue;
            }
            let parts: Vec<usize> = stackable.iter().map(|(argument, _)| *argument).collect();
            let matrices: Vec<Array2<f64>> = stackable.iter().map(|(_, op)| program.operators[*op].matrix()).collect();
            let views: Vec<_> = matrices.iter().map(|m| m.view()).collect();
            let values = ndarray::concatenate(Axis(1), &views).map_err(|error| EngineError::Primitive(error.to_string()))?;
            let finest = stackable
                .iter()
                .filter_map(|(_, op)| match &program.operators[*op].body {
                    OperatorBody::Dense { precision, .. } => Some(precision.fraction_bits()),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            let precision = DeclaredPrecision::new(finest).map_err(EngineError::Primitive)?;
            let cols = Interface::new(parts.iter().flat_map(|p| context.interfaces[*p].groups().iter().copied()).collect())?;
            let rows = program.operators[stackable[0].1].rows.clone();
            let mut present = Array2::from_elem((rows.group_count(), cols.group_count()), false);
            let mut offset = 0;
            for (_, op) in &stackable {
                if let OperatorBody::Dense { present: own, .. } = &program.operators[*op].body {
                    present.slice_mut(s![.., offset..offset + own.ncols()]).assign(own);
                    offset += own.ncols();
                }
            }
            let parts_provenance: Vec<&Provenance> = stackable.iter().map(|(_, op)| &program.operators[*op].provenance).collect();
            let mut candidate = program.clone();
            candidate.operators.push(Arc::new(Operator::blocks(
                format!("stack{index}"),
                rows,
                cols,
                values,
                present,
                precision,
                Provenance::derived(&parts_provenance, "the terms side by side".to_string()),
            )?));
            let stacked = candidate.operators.len() - 1;
            let concat = insert_node(&mut candidate, index, Node::Concat { parts });
            let mut new_terms: Vec<(usize, usize)> = terms
                .iter()
                .copied()
                .filter(|t| !stackable.contains(t))
                .map(|(argument, op)| (if argument >= index { argument + 1 } else { argument }, op))
                .collect();
            new_terms.push((concat, stacked));
            candidate.nodes[index + 1] = Node::Affine { terms: new_terms, bias: *bias };
            candidate.prune();
            out.push(structural(
                "stack_terms",
                ProposalKind::Share,
                "sum_t A_t x_t = [A_1 ... A_T] concat(x_1, ..., x_T)".to_string(),
                format!("stack {} terms of node {index}", stackable.len()),
                candidate,
            ));
        }
        Ok(out)
    }
}

/// `Φ⁻¹` for a character basis on `size` tokens (module note), with each entry's radius.
fn inverse_characters(positions: &[Option<u32>]) -> (Array2<f64>, Array2<f64>, Vec<usize>) {
    let period = positions.iter().filter(|p| p.is_some()).count();
    let planes = (period - 1) / 2;
    let off: Vec<usize> = positions.iter().enumerate().filter(|(_, p)| p.is_none()).map(|(t, _)| t).collect();
    let width = 1 + 2 * planes + off.len();
    let mut m = Array2::<f64>::zeros((positions.len(), width));
    let mut r = Array2::<f64>::zeros((positions.len(), width));
    let trig = accumulation_growth(3) * TAU + 2.0 * UNIT_ROUNDOFF;
    let scale = 2.0 / period as f64;
    for (t, position) in positions.iter().enumerate() {
        match position {
            Some(a) => {
                m[[t, 0]] = 1.0 / period as f64;
                for k in 1..=planes {
                    let angle = TAU * ((k * *a as usize) % period) as f64 / period as f64;
                    let (sine, cosine) = angle.sin_cos();
                    m[[t, 2 * k - 1]] = scale * cosine;
                    m[[t, 2 * k]] = scale * sine;
                    r[[t, 2 * k - 1]] = scale * (trig + 2.0 * UNIT_ROUNDOFF);
                    r[[t, 2 * k]] = scale * (trig + 2.0 * UNIT_ROUNDOFF);
                }
                r[[t, 0]] = UNIT_ROUNDOFF / period as f64;
            }
            None => {
                let column = 1 + 2 * planes + off.iter().position(|o| *o == t).unwrap_or(0);
                m[[t, column]] = 1.0;
            }
        }
    }
    (m, r, off)
}

/// `A M` with its largest band `γ_k max(|A||M|) + max(|A| r_M)`, plus what `A`'s lattice leaves
/// unresolved, `δ_A max_j Σ_k |M_kj|`.
fn transform(a: &Array2<f64>, m: &Array2<f64>, r: &Array2<f64>, source_half_step: f64) -> (Array2<f64>, f64) {
    let value = a.dot(m);
    let abs_a = a.mapv(f64::abs);
    let rounding = abs_a.dot(&m.mapv(f64::abs)).iter().fold(0.0_f64, |x, v| x.max(*v)) * accumulation_growth(a.ncols() + 1);
    let propagated = abs_a.dot(r).iter().fold(0.0_f64, |x, v| x.max(*v));
    (value, rounding + propagated + source_half_step * row_sum(&m.t().to_owned()))
}

/// Rewrite the domain of indicator basis `basis` into the characters of `positions`: every
/// feature reading it and every readout over it. `None` when an operator reading the features is
/// also read elsewhere, or a readout's input is not an affine node.
pub fn change_basis(
    program: &OperatorProgram,
    basis: usize,
    positions: Vec<Option<u32>>,
    declared: bool,
) -> Result<Option<OperatorProgram>, EngineError> {
    let Basis::Indicator { domain } = program.bases[basis] else { return Ok(None) };
    let (m, r, _) = inverse_characters(&positions);
    let mut candidate = program.clone();
    candidate.bases.push(Basis::Characters { domain, positions, declared });
    let new_basis = candidate.bases.len() - 1;
    let character_interface = candidate.bases[new_basis].interface(&candidate.declarations)?;
    let features: BTreeSet<usize> = program
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n, Node::Feature { basis: b, .. } if *b == basis))
        .map(|(i, _)| i)
        .collect();
    let mut readers: BTreeSet<usize> = BTreeSet::new();
    let mut others: BTreeSet<usize> = BTreeSet::new();
    for node in &program.nodes {
        if let Node::Affine { terms, bias } = node {
            for (argument, op) in terms {
                if features.contains(argument) {
                    readers.insert(*op);
                } else {
                    others.insert(*op);
                }
            }
            others.extend(bias.iter().copied());
        }
        if let Node::Constant { operator } = node {
            others.insert(*operator);
        }
    }
    if readers.iter().any(|op| others.contains(op)) {
        return Ok(None);
    }
    for &op in &readers {
        let old = &program.operators[op];
        let (values, band) = transform(&old.matrix(), &m, &r, half_step(old));
        candidate.operators[op] = Arc::new(product_operator(
            old.name.clone(),
            old.rows.clone(),
            character_interface.clone(),
            values,
            band,
            Provenance::derived(&[&old.provenance], format!("times Φ⁻¹ of the cycle on domain {domain}")),
        )?);
    }
    for index in &features {
        if let Node::Feature { basis: b, .. } = &mut candidate.nodes[*index] {
            *b = new_basis;
        }
    }
    let mt = m.t().to_owned();
    let rt = r.t().to_owned();
    for index in 0..program.nodes.len() {
        let Node::Readout { input, basis: b } = program.nodes[index] else { continue };
        if b != basis {
            continue;
        }
        let Node::Affine { terms, bias } = program.nodes[input].clone() else { return Ok(None) };
        let rotated = |op: usize, candidate: &mut OperatorProgram| -> Result<usize, EngineError> {
            let old = &program.operators[op];
            let (values, band) = transform_rows(&mt, &rt, &old.matrix(), half_step(old));
            candidate.operators.push(Arc::new(product_operator(
                format!("Φ⁻ᵀ·{}", old.name),
                character_interface.clone(),
                old.cols.clone(),
                values,
                band,
                Provenance::derived(&[&old.provenance], format!("Φ⁻ᵀ of the cycle on domain {domain} times")),
            )?));
            Ok(candidate.operators.len() - 1)
        };
        let mut new_terms = Vec::new();
        for (argument, op) in &terms {
            new_terms.push((*argument, rotated(*op, &mut candidate)?));
        }
        let new_bias = match bias {
            Some(op) => Some(rotated(op, &mut candidate)?),
            None => None,
        };
        candidate.nodes[input] = Node::Affine { terms: new_terms, bias: new_bias };
        candidate.nodes[index] = Node::Readout { input, basis: new_basis };
    }
    candidate.prune();
    candidate.interfaces()?;
    Ok(Some(candidate))
}

/// `Mᵀ A` with its band.
fn transform_rows(mt: &Array2<f64>, rt: &Array2<f64>, a: &Array2<f64>, source_half_step: f64) -> (Array2<f64>, f64) {
    let (value_t, band) = transform(&a.t().to_owned(), &mt.t().to_owned(), &rt.t().to_owned(), source_half_step);
    (value_t.t().to_owned(), band)
}

/// Discover a cycle labelling of a domain from the weights and rewrite into its characters.
pub struct PlaneBasis;

/// The tokens' table of an indicator basis: every operator reading its features (columns are
/// tokens) and every readout operator over it (rows are tokens), stacked as tokens × features,
/// with a bound on `‖table − table₀‖_F` for the table its operators' reals stand for, each real
/// known to its lattice's half-step ([`lattice_radius`]).
fn token_table(program: &OperatorProgram, basis: usize) -> Option<(Array2<f64>, f64)> {
    let mut blocks: Vec<Array2<f64>> = Vec::new();
    let mut radius_squared = 0.0;
    let features: BTreeSet<usize> = program
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n, Node::Feature { basis: b, .. } if *b == basis))
        .map(|(i, _)| i)
        .collect();
    let mut seen = BTreeSet::new();
    for node in &program.nodes {
        match node {
            Node::Affine { terms, .. } => {
                for (argument, op) in terms {
                    if features.contains(argument) && seen.insert(*op) {
                        blocks.push(program.operators[*op].matrix().t().to_owned());
                        radius_squared += lattice_radius(&program.operators[*op]).powi(2);
                    }
                }
            }
            Node::Readout { input, basis: b } if *b == basis => {
                if let Node::Affine { terms, .. } = &program.nodes[*input] {
                    for (_, op) in terms {
                        if seen.insert(*op) {
                            blocks.push(program.operators[*op].matrix());
                            radius_squared += lattice_radius(&program.operators[*op]).powi(2);
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    if blocks.is_empty() {
        return None;
    }
    let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
    Some((ndarray::concatenate(Axis(1), &views).ok()?, radius_squared.sqrt()))
}

/// `‖A − A₀‖_F` for the operator `A₀` whose reals `A`'s lattice indices stand for, each within
/// the half-step `h`: `h √(present reals)` for a dense operator, and for a low-rank `L R` with both
/// factors within `h`, `h (√|L| ‖R‖_F + ‖L‖_F √|R|) + h² √(|L| |R|)`.
fn lattice_radius(operator: &Operator) -> f64 {
    let frobenius = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
    match &operator.body {
        OperatorBody::Identity => 0.0,
        OperatorBody::Dense { present, precision, .. } => {
            let reals: usize = present
                .indexed_iter()
                .filter(|(_, keep)| **keep)
                .map(|((r, c), _)| operator.rows.range(r).len() * operator.cols.range(c).len())
                .sum();
            precision.worst_case_error() * (reals as f64).sqrt()
        }
        OperatorBody::LowRank { left, right, precision } => {
            let h = precision.worst_case_error();
            let (l, r) = ((left.len() as f64).sqrt(), (right.len() as f64).sqrt());
            (h * (l * frobenius(right) + frobenius(left) * r) + h * h * l * r).next_up()
        }
    }
}

/// The distinct labellings of the tokens of `table`, one per resolved leading subspace whose
/// symmetry group has a canonical odd cycle (module note), with the subspace's rank.
fn discovered_labellings(table: &Array2<f64>, radius: f64) -> Result<Vec<(usize, Vec<Option<u32>>)>, EngineError> {
    let refused = |error: SymmetryError| EngineError::Primitive(error.to_string());
    let mut out: Vec<(usize, Vec<Option<u32>>)> = Vec::new();
    for (rank, ops) in TokenOperators::leading_subspaces(table.view(), radius).map_err(refused)? {
        let discovery = discover_permutations(&ops).map_err(refused)?;
        if let Some(positions) = discovery.group.cycle_positions() {
            if out.iter().all(|(_, seen)| *seen != positions) {
                out.push((rank, positions));
            }
        }
    }
    Ok(out)
}

impl Primitive for PlaneBasis {
    fn name(&self) -> &'static str {
        "plane_basis"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, basis) in program.bases.iter().enumerate() {
            let Basis::Indicator { domain } = basis else { continue };
            let Some((table, radius)) = token_table(program, index) else { continue };
            for (rank, positions) in discovered_labellings(&table, radius)? {
                let n = positions.iter().flatten().count();
                if let Some(candidate) = change_basis(program, index, positions, false)? {
                    out.push(structural(
                        "plane_basis",
                        ProposalKind::Expose,
                        format!("indicators of domain {domain} are Φ⁻¹ times the characters of the {n}-cycle discovered on its rank-{rank} leading subspace"),
                        format!("discovered {n}-cycle characters on domain {domain} (rank {rank})"),
                        candidate,
                    ));
                }
            }
        }
        Ok(out)
    }
}

/// The token of each cycle position of a character basis, for reports.
pub fn cycle_order(positions: &[Option<u32>]) -> Vec<usize> {
    let period = positions.iter().filter(|p| p.is_some()).count();
    let mut order = vec![0; period];
    for (t, p) in positions.iter().enumerate() {
        if let Some(a) = p {
            order[*a as usize] = t;
        }
    }
    order
}

/// Whether an operator is dense with every block present (for reports).
pub fn fully_present(op: &Operator) -> bool {
    matches!(&op.body, OperatorBody::Dense { present, .. } if present.iter().all(|k| *k))
}
