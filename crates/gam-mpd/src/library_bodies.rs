//! Reusable rule bodies of a library explanation (#2951).
//!
//! # Bodies and calls
//!
//! The library (`library_mdl`) starts at `M`'s own functions, and its other moves remove, tie and
//! share them. A body is a function that no single native unit implements: a small program in
//! `M`'s primitives, stored once as a rule of the explanation (`Node::Call` applies a stored rule
//! however often it is called) and applied at several sites, each through bindings of its own.
//! An MLP body of `m` units reads `z` of width `k` and writes `y` of width `k′`,
//!
//! `β(z) = U φ(G z + c)`, or for a gated law `β(z) = U (φ(G z + c) ⊙ (B z + e))`,
//!
//! with `M`'s law `φ`, and the biases `c`, `e` only where `M`'s MLP has them. A call at layer `l`'s
//! MLP reads `z = R x` from the layer's normed stream `x` and adds `W β(z)` to the MLP's output:
//! the read binding `R` (`k × d`) and the write binding `W` (`d × k′`) are the call's own. The prior
//! groups of `library_mdl`'s code length: per unit its gate row with its bias, its up row with its
//! bias, and its output column, paid once however many calls apply the body; per call each row of
//! `R` and each column of `W`. The fit's removal step removes any of them whose information does
//! not pay for itself, so a body's units and widths and a call's widths are chosen by `F`. The
//! discrete structure is priced in `Explanation::fixed_nats`: per call which of its MLP's functions
//! it replaces (the enumerative subset code) and which body it applies (each call names one of the
//! bodies earlier calls apply, or a new one: `ln(b + 1)` nats with `b` bodies so far), and per body
//! its widths `k` and `k′` (Elias δ).
//!
//! # Region rewrite
//!
//! A region is a set of functions of one MLP. Its rewrite ([`rewrite`]) replaces them by a call of a
//! new body. With `A` the region's gate rows (and `D` its up rows), `S = [A; D]` and
//! `S = Σ_r s_r a_r v_rᵀ` its singular value decomposition over the singular values resolved from
//! zero (above the decomposition's rounding band), `R = [v_r]ᵀ`, `G = A Rᵀ` and `B = D Rᵀ`; with
//! `O` the region's output columns and `O = Σ_r t_r p_r q_rᵀ` likewise, `W = [p_r]` and
//! `U = Wᵀ O`. Then `G R = A − ΔA`, `B R = D − ΔD` and `W U = O − ΔO`, where `[ΔA; ΔD]` and `ΔO` are
//! the parts along the singular directions the decomposition does not resolve from zero; the
//! largest singular value dropped is the spectral norm of each, recorded as [`Call::discarded`]
//! (zero when nothing is dropped). On an input `x` the region's pre-activations therefore change by
//! at most `discarded[0] ‖x‖`, and its written output by at most `‖O‖` times the activations'
//! change (at most the law's Lipschitz constant times the pre-activations' change, ungated) plus
//! `discarded[1] ‖h(x)‖`, `h(x)` the region's activations. Unit `j` of the body is the region's
//! `j`-th function: each replaced native block's owner (`Artifact::owners`) becomes the body's
//! block of that unit at the call with the call's binding as its factor (`a_i = g_j R`,
//! `b_i = b_j R`, `u_i = W u_j`, up to the same discarded parts; a bias is the body's own entry and
//! takes no binding; [`Call::replaced`] lists the same correspondence). The native functions'
//! groups leave the explanation (`Explanation::removed`); their reads stay among `M`'s read
//! variables, so the experiments do not change.
//!
//! A rewrite or a merge maps the means: the rewritten explanation's values compute what the
//! region's values computed (up to the discarded parts). It does not map the posterior: the fit
//! that follows starts a new factorized posterior over the body's and the bindings' entries
//! (`Posterior::new`), a new approximation and not the image of the old one (a linear change of
//! coordinates turns a diagonal covariance into a full one, and a product of two uncertain factors
//! is not Gaussian).
//!
//! # Reuse
//!
//! A call's output is unchanged when `z ↦ A z` with `G ↦ G A⁻¹` and `B ↦ B A⁻¹`, when `y ↦ C y`
//! with `U ↦ C U` and `W ↦ W C⁻¹`, and when the body's units are permuted; for a gated law also
//! when a unit's up row is scaled by `α ≠ 0` and its output column by `1/α`, which leaves the
//! unit's rank-one map `M_j = u_j b_jᵀ` (and `N_j = e_j u_j`) unchanged. These are the
//! architecture's exact symmetries of a body. Two bodies compute one function up to their calls'
//! bindings when, for a permutation `π` of the units and linear `A`, `C`, every unit `i` of the
//! first has `g_i = g_{π(i)} A`, `c_i = c_{π(i)}` and `u_i = C u_{π(i)}` (gated: `M_i = C M_{π(i)} A`,
//! `N_i = C N_{π(i)}`), and both bodies have the same law `φ`. [`align`] searches for them by
//! alternating weighted least squares in `A` and `C` with an optimal assignment of the units
//! ([`hungarian`]), each step minimizing over its own block the misfit `Σ (θ₁ − T(θ₀))² / σ₁²` (the
//! first body's posterior means `θ₁` against the transformed second's, each value over its marginal
//! variance; an entry of variance zero is an exact constraint); the alternation finds a local
//! optimum, not necessarily the global one. It reports the same misfit in units of both
//! posteriors' marginal variances, unmatched units of either body included: a heuristic score of the
//! match, since the values are correlated, the gauge is fitted and the matching chosen on the same
//! values. [`merge`] makes every call of the first body a call of the second with the bindings
//! `A R` and `W C`: the element relating the call to the shared body is part of the call's
//! bindings, which are priced, and the matching of the units and the scalars the ownership map
//! needs are priced in `Explanation::fixed_nats`. A rewrite with its reuse is accepted only if `F`
//! falls after the fit re-converges on the fixed native experiments.
//!
//! # Regions
//!
//! The functions of a body are parallel: they read the same few directions and write the same few.
//! They need not interact with each other, so they are not a community of the flow graph (the
//! functions an MLP's units interact with are much the same for all of them); they are the
//! functions whose union a rewrite compresses. Among an MLP's functions that carry RelP flow
//! (`library_readout`), [`regions`] groups them greedily by a heuristic count of the parameters a
//! rewrite of the union saves at the posterior's resolution, and orders the groups by it; the
//! count decides no rewrite (the code length does).

use crate::{
    library_mdl::{Cells, Explanation, Group, Posterior},
    operator_program::{Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Rule, exact_precision, remap_node},
};
use gam_linalg::decompose::svd;
use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::LN_2,
    sync::Arc,
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn operator_named(program: &OperatorProgram, name: &str) -> Option<usize> {
    program.operators.iter().position(|op| op.name == name)
}

fn operator_index(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(format!("no unique operator {name}")),
    }
}

fn rule_index(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    program.rules.iter().position(|r| r.name == name).ok_or_else(|| format!("no rule {name}"))
}

/// A dense operator holding `values` exactly.
fn dense(name: String, rows: Interface, cols: Interface, values: Array2<f64>, provenance: Provenance) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, provenance).map_err(error)
}

fn units(count: usize) -> Result<Interface, String> {
    Interface::uniform(count, 1, LabelKind::Unit, 0).map_err(error)
}

/// A call of a body: its name (`library.l{layer}.call{c}`, its bindings `{name}.read` and
/// `{name}.write`), its layer, its body's rule name, per native function of the layer it replaced
/// the body unit computing it at this call (the native parameters' ownership; a function whose unit
/// a merge did not match keeps no unit), and the spectral norms of the read and the write parts its
/// rewrite discarded (module note).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Call {
    pub name: String,
    pub layer: usize,
    pub body: String,
    pub replaced: Vec<(usize, Option<usize>)>,
    #[serde(default)]
    pub discarded: [f64; 2],
}

/// A body's operators in the explanation's program.
#[derive(Clone, Copy, Debug)]
struct BodyOperators {
    gate: usize,
    gate_bias: Option<usize>,
    up: Option<usize>,
    up_bias: Option<usize>,
    out: usize,
}

impl BodyOperators {
    fn of(program: &OperatorProgram, body: &str) -> Result<Self, String> {
        Ok(Self {
            gate: operator_index(program, &format!("{body}.gate"))?,
            gate_bias: operator_named(program, &format!("{body}.gate_bias")),
            up: operator_named(program, &format!("{body}.up")),
            up_bias: operator_named(program, &format!("{body}.up_bias")),
            out: operator_index(program, &format!("{body}.out"))?,
        })
    }

    fn all(&self) -> Vec<usize> {
        [Some(self.gate), self.gate_bias, self.up, self.up_bias, Some(self.out)].into_iter().flatten().collect()
    }
}

/// The node of `rule` applying `gate` as its first term.
fn applying(rule: &Rule, gate: usize) -> Result<usize, String> {
    rule.nodes
        .iter()
        .position(|n| matches!(n, Node::Affine { terms, .. } if terms.first().is_some_and(|t| t.1 == gate)))
        .ok_or_else(|| format!("{}: no node applies its gate", rule.name))
}

/// The law of the MLP rule `rule`'s units: the pointwise node reading its gate's node.
fn mlp_law(rule: &Rule, gate: usize) -> Result<Law, String> {
    let at = applying(rule, gate)?;
    let laws = rule
        .nodes
        .iter()
        .find_map(|n| match n {
            Node::Pointwise { input, laws } if *input == at => Some(laws),
            _ => None,
        })
        .ok_or_else(|| format!("{}: no law reads its gate", rule.name))?;
    let law = *laws.first().ok_or("an empty law list")?;
    if laws.iter().any(|l| *l != law) {
        return Err(format!("{}: its units have different laws", rule.name));
    }
    Ok(law)
}

/// The next body number: one past the largest `i` of a rule `library.body{i}`.
fn next_body(program: &OperatorProgram) -> usize {
    program.rules.iter().filter_map(|r| r.name.strip_prefix("library.body")?.parse::<usize>().ok()).map(|i| i + 1).max().unwrap_or(0)
}

/// The next call number: one past the largest `c` of an operator `library.l{l}.call{c}.read`.
fn next_call(program: &OperatorProgram) -> usize {
    program
        .operators
        .iter()
        .filter_map(|op| op.name.strip_prefix("library.l")?.split_once(".call")?.1.strip_suffix(".read")?.parse::<usize>().ok())
        .map(|c| c + 1)
        .max()
        .unwrap_or(0)
}

/// The nats of which body each call applies, in the order of the calls' numbers: each call names
/// one of the bodies the calls before it apply or a new one, `ln(b + 1)` nats with `b` bodies so
/// far (a code of the partition of the calls by their bodies).
fn assignment_nats(program: &OperatorProgram) -> Result<f64, String> {
    let mut calls: Vec<(usize, usize)> = Vec::new();
    for rule in &program.rules {
        for (n, node) in rule.nodes.iter().enumerate() {
            let Node::Call { rule: body, arguments } = node else { continue };
            if !program.rules.get(*body).is_some_and(|b| b.name.starts_with("library.body")) {
                continue;
            }
            let read = match arguments.first().and_then(|z| rule.nodes.get(*z)) {
                Some(Node::Affine { terms, .. }) if terms.len() == 1 => terms[0].1,
                other => return Err(format!("{}: call {n}'s argument is {other:?}", rule.name)),
            };
            let number = program.operators[read].name.split_once(".call").and_then(|(_, r)| r.strip_suffix(".read")?.parse::<usize>().ok()).ok_or("a call's read binding's name")?;
            calls.push((number, *body));
        }
    }
    calls.sort_unstable();
    let mut seen: Vec<usize> = Vec::new();
    let mut nats = 0.0;
    for (_, body) in calls {
        nats += ((seen.len() + 1) as f64).ln();
        if !seen.contains(&body) {
            seen.push(body);
        }
    }
    Ok(nats)
}

/// `program` with `rule` inserted as rule 0 (a rule calls only the rules before it), every call's
/// rule index moved.
fn insert_first_rule(program: &mut OperatorProgram, rule: Rule) {
    let rules: Vec<usize> = (0..program.rules.len()).map(|r| r + 1).collect();
    renumber_rules(program, &rules);
    program.rules.insert(0, rule);
}

/// Every call in `program` (its nodes and its rules') made to read rule `rules[r]` for rule `r`.
fn renumber_rules(program: &mut OperatorProgram, rules: &[usize]) {
    let ops: Vec<usize> = (0..program.operators.len()).collect();
    let bases: Vec<usize> = (0..program.bases.len()).collect();
    let nodes: Vec<usize> = (0..program.nodes.len()).collect();
    for node in &mut program.nodes {
        remap_node(node, &nodes, &ops, &bases, rules);
    }
    for r in &mut program.rules {
        let nodes: Vec<usize> = (0..r.nodes.len()).collect();
        for node in &mut r.nodes {
            remap_node(node, &nodes, &ops, &bases, rules);
        }
    }
}

/// The singular triplets of `a`.
type Triplets = (Array2<f64>, Array1<f64>, Array2<f64>);

/// The singular vectors of `a` whose singular values are resolved from zero (above the
/// decomposition's rounding band): `(left, values, right)`, none when every one is within it; and
/// the largest singular value left out, the spectral norm of the part of `a` along the directions
/// not kept (zero when none is left out).
fn split(a: &Array2<f64>) -> Result<(Option<Triplets>, f64), String> {
    if a.is_empty() {
        return Ok((None, 0.0));
    }
    let decomposition = svd(a.view(), false).map_err(error)?;
    let rank = decomposition.singular_values.iter().filter(|s| **s > decomposition.band).count();
    let dropped = decomposition.singular_values.get(rank).copied().unwrap_or(0.0);
    if rank == 0 {
        return Ok((None, dropped));
    }
    Ok((
        Some((
            decomposition.u.slice(s![.., ..rank]).to_owned(),
            decomposition.singular_values.slice(s![..rank]).to_owned(),
            decomposition.vt.slice(s![..rank, ..]).to_owned(),
        )),
        dropped,
    ))
}

/// [`split`]'s resolved singular vectors of `a`.
fn resolved(a: &Array2<f64>) -> Result<Option<Triplets>, String> {
    Ok(split(a)?.0)
}

/// `explanation` with the functions `functions` of layer `layer`'s MLP replaced by a call of a new
/// body (module note: up to the parts along unresolved directions, whose norms the call records),
/// and the call.
pub fn rewrite(explanation: &Explanation, layer: usize, functions: &[usize]) -> Result<(Explanation, Call), String> {
    let mut sorted = functions.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if functions.is_empty() || sorted.len() != functions.len() {
        return Err("a region is a nonempty set of distinct functions".into());
    }
    let mlp = format!("library.l{layer}.mlp");
    let known = &explanation.layers.get(layer).ok_or_else(|| format!("no layer {layer}"))?.functions;
    let program = &explanation.artifact.program;
    let up = operator_named(program, &format!("{mlp}.up"));
    let parts: &[&str] = if up.is_some() { &["gate", "up", "out"] } else { &["gate", "out"] };
    // Only native functions in the explanation: neither removed nor tied.
    let mut retired = Vec::new();
    for &i in functions {
        let own = known.get(i).ok_or_else(|| format!("layer {layer} has no function {i}"))?;
        for part in parts {
            let name = format!("{mlp}.f{i}.{part}");
            let g = explanation.groups.iter().position(|g| g.name == name).ok_or_else(|| format!("no group {name}"))?;
            if !own.contains(&g) || explanation.removed.contains(&g) {
                return Err(format!("{name} is not a native group in the explanation"));
            }
            retired.push(g);
        }
    }
    let gate = operator_index(program, &format!("{mlp}.gate"))?;
    let (gate_bias, up_bias) = (operator_named(program, &format!("{mlp}.gate_bias")), operator_named(program, &format!("{mlp}.up_bias")));
    let out = operator_index(program, &format!("{mlp}.out"))?;
    let rows_of = |op: usize| program.operators[op].matrix().select(Axis(0), functions);
    let a = rows_of(gate);
    let d_up = up.map(rows_of);
    let stacked = match &d_up {
        Some(d) => ndarray::concatenate(Axis(0), &[a.view(), d.view()]).map_err(error)?,
        None => a.clone(),
    };
    let (kept, read_dropped) = split(&stacked)?;
    let (_, _, read) = kept.ok_or("a region whose reads are zero")?;
    let g = a.dot(&read.t());
    let b = d_up.map(|d| d.dot(&read.t()));
    let o = program.operators[out].matrix().select(Axis(1), functions);
    let (kept, write_dropped) = split(&o)?;
    let (write, _, _) = kept.ok_or("a region whose writes are zero")?;
    let u = write.t().dot(&o);
    let (c, e) = (gate_bias.map(rows_of), up_bias.map(rows_of));
    let (n, k, k_out) = (functions.len(), read.nrows(), write.ncols());
    let (d_in, d_out) = (program.operators[gate].cols.width(), program.operators[out].rows.width());

    let mut rewritten = explanation.clone();
    let program = &mut rewritten.artifact.program;
    // The replaced functions leave the MLP: their rows of its input maps and columns of its output
    // are zero, as their removed groups are in the posterior.
    for (op, rows) in [(Some(gate), true), (gate_bias, true), (up, true), (up_bias, true), (Some(out), false)] {
        let Some(op) = op else { continue };
        let source = Arc::clone(&program.operators[op]);
        let mut values = source.matrix();
        for &i in functions {
            if rows { values.row_mut(i).fill(0.0) } else { values.column_mut(i).fill(0.0) }
        }
        program.operators[op] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, source.provenance.clone())?);
    }
    let body = format!("library.body{}", next_body(program));
    let call = format!("library.l{layer}.call{}", next_call(program));
    let mlp_rule = rule_index(program, &mlp)?;
    let law = mlp_law(&program.rules[mlp_rule], gate)?;
    let input = match &program.rules[mlp_rule].nodes[applying(&program.rules[mlp_rule], gate)?] {
        Node::Affine { terms, .. } => terms[0].0,
        other => return Err(format!("{mlp}: its gate node is {other:?}")),
    };
    let provenance = Provenance::derived(&[&program.operators[gate].provenance, &program.operators[out].provenance], format!("rewrite of {mlp} functions {functions:?}"));
    let (z, h, y) = (units(k)?, units(n)?, units(k_out)?);
    let (stream_in, stream_out) = (program.operators[gate].cols.clone(), program.operators[out].rows.clone());
    // The new operators in order: the body's gate, gate bias, up map, up bias and output, then the
    // call's read and write bindings.
    let base = program.operators.len();
    let mut added: Vec<Operator> = Vec::new();
    let add = |added: &mut Vec<Operator>, op: Operator| {
        added.push(op);
        base + added.len() - 1
    };
    let body_gate = add(&mut added, dense(format!("{body}.gate"), h.clone(), z.clone(), g, provenance.clone())?);
    let body_gate_bias = match c {
        Some(c) => Some(add(&mut added, dense(format!("{body}.gate_bias"), h.clone(), Interface::constant(), c, provenance.clone())?)),
        None => None,
    };
    let body_up = match b {
        Some(b) => Some(add(&mut added, dense(format!("{body}.up"), h.clone(), z.clone(), b, provenance.clone())?)),
        None => None,
    };
    let body_up_bias = match e {
        Some(e) => Some(add(&mut added, dense(format!("{body}.up_bias"), h.clone(), Interface::constant(), e, provenance.clone())?)),
        None => None,
    };
    let body_out = add(&mut added, dense(format!("{body}.out"), y.clone(), h, u, provenance.clone())?);
    let read_op = add(&mut added, dense(format!("{call}.read"), z.clone(), stream_in, read, provenance.clone())?);
    let write_op = add(&mut added, dense(format!("{call}.write"), stream_out, y, write, provenance)?);
    program.operators.extend(added.into_iter().map(Arc::new));
    let mut nodes = vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, body_gate)], bias: body_gate_bias }, Node::Pointwise { input: 1, laws: vec![law; n] }];
    if let Some(up) = body_up {
        nodes.push(Node::Affine { terms: vec![(0, up)], bias: body_up_bias });
        nodes.push(Node::Hadamard { left: 2, right: 3 });
    }
    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, body_out)], bias: None });
    insert_first_rule(program, Rule { name: body.clone(), inputs: vec![z], output: nodes.len() - 1, nodes });
    // The call: `z = R x`, `y = β(z)`, and `W y` added to the MLP's output.
    let r = &mut program.rules[mlp_rule + 1];
    let terms = match r.nodes.last() {
        Some(Node::Affine { terms, bias: None }) if r.output + 1 == r.nodes.len() => terms.clone(),
        _ => return Err(format!("{mlp}: its output is not its last node, a bias-free affine sum")),
    };
    r.nodes.pop();
    r.nodes.push(Node::Affine { terms: vec![(input, read_op)], bias: None });
    r.nodes.push(Node::Call { rule: 0, arguments: vec![r.nodes.len() - 1] });
    let mut terms = terms;
    terms.push((r.nodes.len() - 1, write_op));
    r.nodes.push(Node::Affine { terms, bias: None });
    r.output = r.nodes.len() - 1;
    program.interfaces().map_err(error)?;
    // The body's groups once, the call's per row of `R` and column of `W`. A unit's group takes the
    // reference of the native group it replaces at the same total square (`g_j = a_i Rᵀ` and
    // `u_j = Wᵀ o_i` keep a direction's norm), and a binding's that of a unit direction.
    let (groups, reference) = (&mut rewritten.groups, &mut rewritten.reference);
    for j in 0..n {
        let mut gate_cells = vec![Cells { operator: body_gate, rows: vec![j], cols: 0..k }];
        gate_cells.extend(body_gate_bias.map(|op| Cells { operator: op, rows: vec![j], cols: 0..1 }));
        let mut unit = vec![Group { name: format!("{body}.u{j}.gate"), cells: gate_cells }];
        if let Some(up) = body_up {
            let mut up_cells = vec![Cells { operator: up, rows: vec![j], cols: 0..k }];
            up_cells.extend(body_up_bias.map(|op| Cells { operator: op, rows: vec![j], cols: 0..1 }));
            unit.push(Group { name: format!("{body}.u{j}.up"), cells: up_cells });
        }
        unit.push(Group { name: format!("{body}.u{j}.out"), cells: vec![Cells { operator: body_out, rows: (0..k_out).collect(), cols: j..j + 1 }] });
        for (p, group) in unit.into_iter().enumerate() {
            let native = retired[j * parts.len() + p];
            reference.push(explanation.reference[native] * size(&explanation.groups[native]) / size(&group));
            groups.push(group);
        }
    }
    let bindings = binding_groups(&call, read_op, write_op, k, k_out, d_in, d_out);
    reference.extend(bindings.iter().map(binding_reference));
    groups.extend(bindings);
    // Each replaced native block's owner is now the body's block of its unit at this call, read
    // through the call's bindings (a gate or up row through `R`, an output column through `W`); a
    // bias is the body's own entry and takes no binding, whatever the read's width.
    let row_block = |operator: &str, i: usize| -> Option<(usize, std::ops::Range<usize>, std::ops::Range<usize>, bool)> {
        let j = functions.iter().position(|f| *f == i)?;
        let parts = [("gate", Some(body_gate), k, true), ("up", body_up, k, true), ("gate_bias", body_gate_bias, 1, false), ("up_bias", body_up_bias, 1, false)];
        let (_, op, width, reads) = parts.into_iter().find(|(part, ..)| operator == format!("{mlp}.{part}"))?;
        Some((op?, j..j + 1, 0..width, reads))
    };
    let (read_name, write_name) = (format!("{call}.read"), format!("{call}.write"));
    for owner in &mut rewritten.artifact.owners {
        // `a_i = g_j R`, `b_i = b_j R` and `u_i = W u_j` (biases are the body's own entries).
        let target = if owner.operator == format!("{mlp}.out") && owner.cols.len() == 1 {
            functions.iter().position(|f| *f == owner.cols.start).map(|j| (body_out, 0..k_out, j..j + 1, vec![write_name.clone()], Vec::new()))
        } else if owner.rows.len() == 1 {
            row_block(&owner.operator, owner.rows.start).map(|(op, rows, cols, reads)| {
                let right = if reads { vec![read_name.clone()] } else { Vec::new() };
                (op, rows, cols, Vec::new(), right)
            })
        } else {
            None
        };
        if let Some((op, rows, cols, left, right)) = target {
            owner.repoint(&program.operators[op].name, rows, cols, &left, &right);
            owner.body = body.clone();
            owner.site = call.clone();
        }
    }
    rewritten.trainable.extend(base..program.operators.len());
    rewritten.trainable.sort_unstable();
    // The rewrite's choices: which of the MLP's functions the call replaces (the enumerative subset
    // code), the body's widths `k` and `k′` (Elias δ), and which body the call applies.
    let widths = crate::codec::elias_delta_len_bits(k as u64).map_err(error)? + crate::codec::elias_delta_len_bits(k_out as u64).map_err(error)?;
    let subset = crate::codec::subset_code_len_bits(known.len(), n).map_err(error)?;
    rewritten.fixed_nats += (subset + widths) as f64 * std::f64::consts::LN_2 + assignment_nats(program)? - assignment_nats(&explanation.artifact.program)?;
    rewritten.removed.extend(retired);
    rewritten.removed.sort_unstable();
    let replaced = functions.iter().enumerate().map(|(j, i)| (*i, Some(j))).collect();
    Ok((rewritten, Call { name: call, layer, body, replaced, discarded: [read_dropped, write_dropped] }))
}

/// A call's binding groups: each row of its read binding (`k × d_in`) and each column of its write
/// binding (`d_out × k′`).
/// A group's number of entries.
fn size(group: &Group) -> f64 {
    group.cells.iter().map(|c| c.rows.len() * c.cols.len()).sum::<usize>() as f64
}

/// A binding group's reference variance: the mean square of a unit direction over its entries.
fn binding_reference(group: &Group) -> f64 {
    1.0 / size(group)
}

fn binding_groups(call: &str, read: usize, write: usize, k: usize, k_out: usize, d_in: usize, d_out: usize) -> Vec<Group> {
    let reads = (0..k).map(|q| Group { name: format!("{call}.read{q}"), cells: vec![Cells { operator: read, rows: vec![q], cols: 0..d_in }] });
    let writes = (0..k_out).map(|q| Group { name: format!("{call}.write{q}"), cells: vec![Cells { operator: write, rows: (0..d_out).collect(), cols: q..q + 1 }] });
    reads.chain(writes).collect()
}

/// One call site of a body in the explanation's program: the MLP rule, the call node, and its read
/// and write binding operators.
#[derive(Clone, Copy, Debug)]
struct Site {
    rule: usize,
    node: usize,
    read: usize,
    write: usize,
}

/// Every call site of rule `body` among the MLP rules of `program`.
fn sites(program: &OperatorProgram, body: usize) -> Result<Vec<Site>, String> {
    let mut out = Vec::new();
    for (r, rule) in program.rules.iter().enumerate() {
        for (n, node) in rule.nodes.iter().enumerate() {
            let Node::Call { rule: called, arguments } = node else { continue };
            if *called != body {
                continue;
            }
            let read = match arguments.first().and_then(|z| rule.nodes.get(*z)) {
                Some(Node::Affine { terms, bias: None }) if terms.len() == 1 => terms[0].1,
                other => return Err(format!("{}: a call's argument is not a read binding: {other:?}", rule.name)),
            };
            let write = match rule.nodes.get(rule.output) {
                Some(Node::Affine { terms, .. }) => terms.iter().find(|t| t.0 == n).map(|t| t.1),
                _ => None,
            }
            .ok_or_else(|| format!("{}: a call whose value the output does not write", rule.name))?;
            out.push(Site { rule: r, node: n, read, write });
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------ alignment

/// Posterior means and deviations of one part of a body.
#[derive(Clone, Debug)]
pub struct Part {
    pub mean: Array2<f64>,
    pub sd: Array2<f64>,
}

/// One body's posterior: its law, per unit its gate row (`m × k`), gate bias, up row, up bias, and
/// its output column (`k′ × m`); which units compute a function that is not identically zero
/// (`live`), and which input and output coordinates some call of the body reads or writes.
#[derive(Clone, Debug)]
pub struct BodyValues {
    pub law: Law,
    pub gate: Part,
    pub gate_bias: Option<Part>,
    pub up: Option<Part>,
    pub up_bias: Option<Part>,
    pub out: Part,
    pub units: Vec<bool>,
    pub inputs: Vec<bool>,
    pub outputs: Vec<bool>,
}

/// Whether unit `j` of `values` computes a function that is not identically zero on the used
/// coordinates over the posterior's support, an entry of deviation zero being fixed at its mean
/// (with `exact`, every entry is fixed: values without a posterior). The unit computes
/// `φ(g·z + c) u` (gated: `φ(g·z + c) (b·z + e) u`), so it is identically zero exactly when its
/// output column is fixed at zero, or its activation is (the gate row fixed at zero and the bias
/// fixed at a value `c` with `φ(c) = 0`, `c = 0` without a bias), or, gated, its payload is (the up
/// row and its bias fixed at zero).
fn live(values: &BodyValues, j: usize, exact: bool) -> bool {
    let fixed = |sd: f64| exact || sd == 0.0;
    let zero_row = |p: &Part| (0..values.inputs.len()).filter(|q| values.inputs[*q]).all(|q| p.mean[[j, q]] == 0.0 && fixed(p.sd[[j, q]]));
    let out_zero = (0..values.outputs.len()).filter(|q| values.outputs[*q]).all(|q| values.out.mean[[q, j]] == 0.0 && fixed(values.out.sd[[q, j]]));
    // A bias's value where it is fixed (`0` without one), none where it varies.
    let constant = |bias: &Option<Part>| match bias {
        None => Some(0.0),
        Some(b) => fixed(b.sd[[j, 0]]).then_some(b.mean[[j, 0]]),
    };
    let activation_zero = zero_row(&values.gate) && constant(&values.gate_bias).is_some_and(|c| values.law.apply(c) == 0.0);
    let payload_zero = values.up.as_ref().is_some_and(|up| zero_row(up) && constant(&values.up_bias) == Some(0.0));
    !(out_zero || activation_zero || payload_zero)
}

/// `body`'s posterior (removed groups are zero with zero deviation).
pub fn body_values(explanation: &Explanation, posterior: &Posterior, body: &str) -> Result<BodyValues, String> {
    let program = &explanation.artifact.program;
    let ops = BodyOperators::of(program, body)?;
    let rule = rule_index(program, body)?;
    let law = mlp_law(&program.rules[rule], ops.gate)?;
    let means = posterior.means();
    let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("{body}: operator {op} is not trainable"));
    let sd = |i: usize| posterior.log_sd[i].mapv(f64::exp);
    let take = |op: usize| -> Result<Part, String> {
        let i = position(op)?;
        Ok(Part { mean: means[i].clone(), sd: sd(i) })
    };
    let (gate, out) = (take(ops.gate)?, take(ops.out)?);
    let (k, k_out) = (gate.mean.ncols(), out.mean.nrows());
    // A binding's row or column is used unless it is fixed at zero (a removed group).
    let used = |m: ndarray::ArrayView1<'_, f64>, s: ndarray::ArrayView1<'_, f64>| m.iter().zip(s).any(|(m, s)| *m != 0.0 || *s != 0.0);
    let (mut inputs, mut outputs) = (vec![false; k], vec![false; k_out]);
    for site in sites(program, rule)? {
        let (r, w) = (position(site.read)?, position(site.write)?);
        let (read_sd, write_sd) = (sd(r), sd(w));
        inputs.iter_mut().enumerate().for_each(|(q, u)| *u |= used(means[r].row(q), read_sd.row(q)));
        outputs.iter_mut().enumerate().for_each(|(q, u)| *u |= used(means[w].column(q), write_sd.column(q)));
    }
    let mut values = BodyValues {
        law,
        gate_bias: ops.gate_bias.map(take).transpose()?,
        up: ops.up.map(take).transpose()?,
        up_bias: ops.up_bias.map(take).transpose()?,
        gate,
        out,
        units: Vec::new(),
        inputs,
        outputs,
    };
    values.units = (0..values.gate.mean.nrows()).map(|j| live(&values, j, false)).collect();
    Ok(values)
}

/// `body`'s values in `program` (its operators' values, unit deviations): what an alignment of
/// fitted means needs where no posterior is at hand. Its values are exact, so a unit is live when
/// its function at them is not identically zero.
fn artifact_values(program: &OperatorProgram, body: &str) -> Result<BodyValues, String> {
    let ops = BodyOperators::of(program, body)?;
    let law = mlp_law(&program.rules[rule_index(program, body)?], ops.gate)?;
    let part = |op: usize| {
        let mean = program.operators[op].matrix();
        Part { sd: Array2::ones(mean.dim()), mean }
    };
    let (gate, out) = (part(ops.gate), part(ops.out));
    let (inputs, outputs) = (vec![true; gate.mean.ncols()], vec![true; out.mean.nrows()]);
    let mut values = BodyValues { law, gate_bias: ops.gate_bias.map(part), up: ops.up.map(part), up_bias: ops.up_bias.map(part), gate, out, units: Vec::new(), inputs, outputs };
    values.units = (0..values.gate.mean.nrows()).map(|j| live(&values, j, true)).collect();
    Ok(values)
}

/// The gauge relating body `from` to body `onto` (module note): unit `i` of `from` is unit
/// `units[i]` of `onto` (none when unmatched); `input` is `A` (`k_onto × k_from`) and `output` is
/// `C` (`k′_from × k′_onto`), zero outside the coordinates the calls use. `misfit` sums, over the
/// `entries` values compared (the matched units' and every unmatched unit's of either body), each
/// squared difference over its marginal variance under both posteriors: a heuristic score of the
/// match, with no established null distribution (the values are correlated, and the gauge and the
/// matching are fitted to them). `gauge` counts the entries of `A` and `C` fitted, and `live` the
/// units of `from` and of `onto` that compute a function (`live`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    pub units: Vec<Option<usize>>,
    pub input: Array2<f64>,
    pub output: Array2<f64>,
    pub misfit: f64,
    pub entries: usize,
    pub gauge: usize,
    #[serde(default)]
    pub live: [usize; 2],
}

impl Alignment {
    /// Whether the compared values outnumber the gauge's fitted entries; otherwise the gauge can
    /// fit every value, and the alignment does not constrain the two bodies to one function.
    #[must_use]
    pub fn constrains(&self) -> bool {
        self.entries > self.gauge
    }

    /// The nats of the matching of the units, uniform over the partial matchings of `from`'s `m`
    /// live units to `onto`'s `n`: `ln Σ_j C(m, j) C(n, j) j!` (at least `ln m!` for a full
    /// permutation); infinite when the matching has more pairs than either body has live units.
    #[must_use]
    pub fn matching_nats(&self) -> f64 {
        let [m, n] = self.live;
        if self.units.iter().filter(|u| u.is_some()).count() > m.min(n) {
            return f64::INFINITY;
        }
        let ln_factorial = |x: usize| statrs::function::gamma::ln_gamma(x as f64 + 1.0);
        let ln_choose = |a: usize, b: usize| ln_factorial(a) - ln_factorial(b) - ln_factorial(a - b);
        let terms: Vec<f64> = (0..=m.min(n)).map(|j| ln_choose(m, j) + ln_choose(n, j) + ln_factorial(j)).collect();
        gam_math::categorical::log_sum_exp(&terms).unwrap_or(f64::INFINITY)
    }
}

/// One unit of a body restricted to the used coordinates: its gate row and bias, and its write,
/// each value with its marginal variance.
#[derive(Clone, Debug)]
struct Unit {
    gate: (Array1<f64>, Array1<f64>),
    bias: Option<(f64, f64)>,
    write: Write,
}

/// A unit's write: its output column `u`, or for a gated law its factors, the output column `u`
/// and the up row `b` with its bias `e`, whose rank-one map `M = u bᵀ` and offset `N = e u` the
/// comparison reads ([`Write::map`]).
#[derive(Clone, Debug)]
enum Write {
    Plain((Array1<f64>, Array1<f64>)),
    Gated { out: (Array1<f64>, Array1<f64>), up: (Array1<f64>, Array1<f64>), bias: Option<(f64, f64)> },
}

/// The mean and the marginal variances of `x yᵀ` for independent random vectors `x` and `y` with
/// means and marginal variances `(x̄, v_x)` and `(ȳ, v_y)`:
/// `Var(x_r y_s) = x̄_r² v_y,s + v_x,r ȳ_s² + v_x,r v_y,s`. It reads only each factor's marginals,
/// however its own entries are correlated, so it holds for the transformed factors `C u` and `b A`,
/// where treating the entries of `C (u bᵀ) A` as independent does not (they share their factors).
fn product(x: &(Array1<f64>, Array1<f64>), y: &(Array1<f64>, Array1<f64>)) -> (Array2<f64>, Array2<f64>) {
    let squared = |a: &Array1<f64>| a.mapv(|v| v * v);
    (outer(&x.0, &y.0), outer(&squared(&x.0), &y.1) + outer(&x.1, &squared(&y.0)) + outer(&x.1, &y.1))
}

/// A gated write's map and offset.
type Map = ((Array2<f64>, Array2<f64>), Option<(Array1<f64>, Array1<f64>)>);

impl Write {
    /// A gated write's map `M = u bᵀ` and offset `N = e u` (none without an up bias), each with its
    /// marginal variances ([`product`]); none for an ungated write.
    fn map(&self) -> Option<Map> {
        let Self::Gated { out, up, bias } = self else { return None };
        let offset = bias.map(|(e, ve)| {
            let (n, v) = product(out, &(Array1::from_elem(1, e), Array1::from_elem(1, ve)));
            (n.column(0).to_owned(), v.column(0).to_owned())
        });
        Some((product(out, up), offset))
    }
}

/// The live units of `values` on the used coordinates, with their indices in the body.
fn live_units(values: &BodyValues) -> Vec<(usize, Unit)> {
    let ins: Vec<usize> = (0..values.inputs.len()).filter(|q| values.inputs[*q]).collect();
    let outs: Vec<usize> = (0..values.outputs.len()).filter(|q| values.outputs[*q]).collect();
    let squared = |a: &Array1<f64>| a.mapv(|v| v * v);
    (0..values.units.len())
        .filter(|j| values.units[*j])
        .map(|j| {
            let row = |p: &Part| (p.mean.row(j).select(Axis(0), &ins), squared(&p.sd.row(j).select(Axis(0), &ins)));
            let scalar = |p: &Part| (p.mean[[j, 0]], p.sd[[j, 0]].powi(2));
            let u = (values.out.mean.column(j).select(Axis(0), &outs), squared(&values.out.sd.column(j).select(Axis(0), &outs)));
            let write = match &values.up {
                Some(up) => Write::Gated { out: u, up: row(up), bias: values.up_bias.as_ref().map(scalar) },
                None => Write::Plain(u),
            };
            (j, Unit { gate: row(&values.gate), bias: values.gate_bias.as_ref().map(scalar), write })
        })
        .collect()
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.len(), b.len()), |(i, j)| a[i] * b[j])
}

/// `onto`'s unit in `from`'s coordinates under the gauge: its values carried through `A` and `C`
/// with their marginal variances (a row `g A` and a column `C u` of independent entries have the
/// variances `v_g A²` and `C² v_u`; a gated map's follow from its transformed factors,
/// [`product`]).
fn transformed(unit: &Unit, a: &Array2<f64>, c: &Array2<f64>) -> Unit {
    let (a2, c2) = (a.mapv(|v| v * v), c.mapv(|v| v * v));
    let column = |(u, v): &(Array1<f64>, Array1<f64>)| (c.dot(u), c2.dot(v));
    let row = |(g, v): &(Array1<f64>, Array1<f64>)| (g.dot(a), v.dot(&a2));
    let write = match &unit.write {
        Write::Plain(u) => Write::Plain(column(u)),
        Write::Gated { out, up, bias } => Write::Gated { out: column(out), up: row(up), bias: *bias },
    };
    Unit { gate: row(&unit.gate), bias: unit.bias, write }
}

/// `Σ (x − y)² / v` over paired values with variances `v`; a value of variance zero is exact, so
/// any difference there is infinite.
fn chi2(x: ndarray::ArrayView1<'_, f64>, y: ndarray::ArrayView1<'_, f64>, v: ndarray::ArrayView1<'_, f64>) -> f64 {
    x.iter().zip(y).zip(v).map(|((a, b), v)| if *v > 0.0 { (a - b).powi(2) / v } else if a == b { 0.0 } else { f64::INFINITY }).sum()
}

fn flat(m: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(m.iter().copied())
}

/// The misfit of `from`'s unit against `onto`'s transformed unit `t`: in units of `from`'s
/// variances alone (`both` false), or of both posteriors' (`both` true); and the values compared.
fn unit_misfit(from: &Unit, t: &Unit, both: bool) -> (f64, usize) {
    let v = |a: &Array1<f64>, b: &Array1<f64>| if both { a + b } else { a.clone() };
    let mut total = chi2(from.gate.0.view(), t.gate.0.view(), v(&from.gate.1, &t.gate.1).view());
    let mut entries = from.gate.0.len();
    if let (Some((a, va)), Some((b, vb))) = (from.bias, t.bias) {
        let var = if both { va + vb } else { va };
        total += chi2(ndarray::aview1(&[a]), ndarray::aview1(&[b]), ndarray::aview1(&[var]));
        entries += 1;
    }
    match (&from.write, &t.write, from.write.map(), t.write.map()) {
        (Write::Plain(x), Write::Plain(y), ..) => {
            total += chi2(x.0.view(), y.0.view(), v(&x.1, &y.1).view());
            entries += x.0.len();
        }
        (.., Some((x, xo)), Some((y, yo))) => {
            total += chi2(flat(&x.0).view(), flat(&y.0).view(), v(&flat(&x.1), &flat(&y.1)).view());
            entries += x.0.len();
            if let (Some(x), Some(y)) = (xo, yo) {
                total += chi2(x.0.view(), y.0.view(), v(&x.1, &y.1).view());
                entries += x.0.len();
            }
        }
        _ => return (f64::INFINITY, entries),
    }
    (total, entries)
}

/// The cost of leaving `unit` unmatched: its values in units of its own variances.
fn alone(unit: &Unit) -> f64 {
    let zero = |n: usize| Array1::zeros(n);
    let mut total = chi2(unit.gate.0.view(), zero(unit.gate.0.len()).view(), unit.gate.1.view());
    if let Some((a, va)) = unit.bias {
        total += chi2(ndarray::aview1(&[a]), ndarray::aview1(&[0.0]), ndarray::aview1(&[va]));
    }
    total += match (&unit.write, unit.write.map()) {
        (Write::Plain((u, v)), _) => chi2(u.view(), zero(u.len()).view(), v.view()),
        (_, Some(((m, v), offset))) => {
            chi2(flat(&m).view(), zero(m.len()).view(), flat(&v).view()) + offset.as_ref().map_or(0.0, |(n, v)| chi2(n.view(), zero(n.len()).view(), v.view()))
        }
        (Write::Gated { .. }, None) => 0.0,
    };
    total
}

/// The assignment cost of `from`'s units (rows) to `onto`'s (columns) under the gauge, padded to a
/// square with unmatched units (module note).
fn assignment_costs(from: &[(usize, Unit)], onto: &[(usize, Unit)], a: &Array2<f64>, c: &Array2<f64>) -> Array2<f64> {
    let n = from.len().max(onto.len());
    let t: Vec<Unit> = onto.iter().map(|(_, u)| transformed(u, a, c)).collect();
    Array2::from_shape_fn((n, n), |(i, j)| match (from.get(i), t.get(j)) {
        (Some((_, f)), Some(o)) => unit_misfit(f, o, false).0,
        (Some((_, f)), None) => alone(f),
        (None, Some(_)) => alone(&onto[j].1),
        (None, None) => 0.0,
    })
}

/// Weighted least squares per target column: `x` minimizing `Σ_i w_ic (y_ic − (D x)_ic)²` for each
/// column `c`, `D` the design (`rows × p`), `y` and `w` (`rows × q`); `x` is `p × q`. A row of
/// infinite weight (a value of variance zero) is an equality constraint `(D x)_ic = y_ic`: `x` is
/// the constraints' minimum-norm solution plus the weighted least-squares solution over the
/// directions they leave free, and constraints that conflict (a residual beyond the rounding of
/// `D x` and the decomposition's band) are an error. The directions the weighted design does not
/// resolve get zero.
fn weighted_columns(design: &Array2<f64>, y: &Array2<f64>, w: &Array2<f64>) -> Result<Array2<f64>, String> {
    let p = design.ncols();
    let mut x = Array2::zeros((p, y.ncols()));
    // `γ_{p+1}`: the forward rounding bound of a `p`-term product and its difference.
    let units = (p + 1) as f64 * f64::EPSILON;
    let gamma = units / (1.0 - units);
    for c in 0..y.ncols() {
        let (exact, noisy): (Vec<usize>, Vec<usize>) = (0..design.nrows()).partition(|r| w[[*r, c]].is_infinite());
        // The constraints' minimum-norm solution and the projector onto the directions they leave free.
        let (mut base, mut free) = (Array1::zeros(p), Array2::eye(p));
        if !exact.is_empty() {
            let (de, ye) = (design.select(Axis(0), &exact), y.column(c).select(Axis(0), &exact));
            let decomposition = svd(de.view(), false).map_err(error)?;
            let rank = decomposition.singular_values.iter().filter(|s| **s > decomposition.band).count();
            if rank > 0 {
                let (left, values, right) = (decomposition.u.slice(s![.., ..rank]), decomposition.singular_values.slice(s![..rank]), decomposition.vt.slice(s![..rank, ..]));
                base = right.t().dot(&(&left.t().dot(&ye) / &values));
                free = free - right.t().dot(&right);
            }
            let scale = base.dot(&base).sqrt();
            for (r, row) in de.outer_iter().enumerate() {
                let size: f64 = row.iter().zip(&base).map(|(d, b)| (d * b).abs()).sum::<f64>() + ye[r].abs();
                if (row.dot(&base) - ye[r]).abs() > gamma * size + decomposition.band * scale {
                    return Err("conflicting exact constraints".into());
                }
            }
        }
        let mut column = base.clone();
        if !noisy.is_empty() {
            let root: Array1<f64> = Array1::from_iter(noisy.iter().map(|r| w[[*r, c]].sqrt()));
            let dn = design.select(Axis(0), &noisy);
            let scaled = dn.dot(&free) * &root.view().insert_axis(Axis(1));
            let target = (&y.column(c).select(Axis(0), &noisy) - &dn.dot(&base)) * &root;
            if let Some((left, values, right)) = resolved(&scaled)? {
                column = column + free.dot(&right.t().dot(&(left.t().dot(&target) / &values)));
            }
        }
        x.column_mut(c).assign(&column);
    }
    Ok(x)
}

/// Precisions `1/v`, infinite where the variance is zero: a value known exactly is an equality
/// constraint ([`weighted_columns`]).
fn precisions(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(|v| if v > 0.0 { 1.0 / v } else { f64::INFINITY })
}

/// `A` minimizing the misfit over the matched pairs `(from unit, onto unit)` with `C` fixed: the
/// gate rows `g_i ≈ g_j A` and, gated, the rows of `M_i ≈ (C M_j) A`.
fn input_gauge(pairs: &[(&Unit, &Unit)], c: &Array2<f64>, k_onto: usize, k_from: usize) -> Result<Array2<f64>, String> {
    let (mut design, mut targets, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for (f, o) in pairs {
        design.push(o.gate.0.clone());
        targets.push(f.gate.0.clone());
        weights.push(precisions(&f.gate.1));
        if let (Some(((mf, vf), _)), Some(((mo, _), _))) = (f.write.map(), o.write.map()) {
            let cm = c.dot(&mo);
            for r in 0..mf.nrows() {
                design.push(cm.row(r).to_owned());
                targets.push(mf.row(r).to_owned());
                weights.push(precisions(&vf.row(r).to_owned()));
            }
        }
    }
    if design.is_empty() {
        return Ok(Array2::zeros((k_onto, k_from)));
    }
    let stack = |rows: Vec<Array1<f64>>, width: usize| Array2::from_shape_fn((rows.len(), width), |(i, j)| rows[i][j]);
    weighted_columns(&stack(design, k_onto), &stack(targets, k_from), &stack(weights, k_from))
}

/// `C` minimizing the misfit over the matched pairs with `A` fixed: the output columns
/// `u_i ≈ C u_j`, or gated the columns of `M_i ≈ C (M_j A)` and `N_i ≈ C N_j`.
fn output_gauge(pairs: &[(&Unit, &Unit)], a: &Array2<f64>, k_from: usize, k_onto: usize) -> Result<Array2<f64>, String> {
    // Rows of the design are `C`'s inputs (`k′_onto`), targets and weights `k′_from` wide; the
    // solution is `Cᵀ`.
    let (mut design, mut targets, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for (f, o) in pairs {
        match (&f.write, &o.write, f.write.map(), o.write.map()) {
            (Write::Plain((u, v)), Write::Plain((uo, _)), ..) => {
                design.push(uo.clone());
                targets.push(u.clone());
                weights.push(precisions(v));
            }
            (.., Some(((mf, vf), of)), Some(((mo, _), oo))) => {
                let ma = mo.dot(a);
                for col in 0..mf.ncols() {
                    design.push(ma.column(col).to_owned());
                    targets.push(mf.column(col).to_owned());
                    weights.push(precisions(&vf.column(col).to_owned()));
                }
                if let (Some((nf, vn)), Some((no, _))) = (of, oo) {
                    design.push(no);
                    targets.push(nf);
                    weights.push(precisions(&vn));
                }
            }
            _ => return Err("bodies of different laws".into()),
        }
    }
    if design.is_empty() {
        return Ok(Array2::zeros((k_from, k_onto)));
    }
    let stack = |rows: Vec<Array1<f64>>, width: usize| Array2::from_shape_fn((rows.len(), width), |(i, j)| rows[i][j]);
    Ok(weighted_columns(&stack(design, k_onto), &stack(targets, k_from), &stack(weights, k_from))?.reversed_axes())
}

/// The projection onto the column span of `m`, unchanged by every invertible map of its columns.
fn span_projection(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    Ok(match resolved(m)? {
        Some((left, _, _)) => left.dot(&left.t()),
        None => Array2::zeros((m.nrows(), m.nrows())),
    })
}

/// A unit's signature: the sorted magnitudes of its row of the projection onto the units' read
/// span (gate rows), and of its row of the projection onto their write span (output columns, or
/// gated the rank-one maps), kept apart.
type Signature = (Array1<f64>, Array1<f64>);

/// Per live unit a signature unchanged by the gauges and by permutations of the other units.
fn signatures(units: &[(usize, Unit)]) -> Result<Vec<Signature>, String> {
    let m = units.len();
    let reads = Array2::from_shape_fn((m, units.first().map_or(0, |u| u.1.gate.0.len())), |(i, j)| units[i].1.gate.0[j]);
    let written = |u: &Unit| -> Array1<f64> {
        match (&u.write, u.write.map()) {
            (Write::Plain((w, _)), _) => w.clone(),
            (_, Some(((w, _), _))) => flat(&w),
            (Write::Gated { .. }, None) => Array1::zeros(0),
        }
    };
    let writes: Vec<Array1<f64>> = units.iter().map(|(_, u)| written(u)).collect();
    let width = writes.first().map_or(0, Array1::len);
    let writes = Array2::from_shape_fn((m, width), |(i, j)| writes[i][j]);
    let (pr, pw) = (span_projection(&reads)?, span_projection(&writes)?);
    let sorted = |p: &Array2<f64>, i: usize| {
        let mut row: Vec<f64> = p.row(i).iter().map(|v| v.abs()).collect();
        row.sort_by(|a, b| b.total_cmp(a));
        Array1::from_vec(row)
    };
    Ok((0..m).map(|i| (sorted(&pr, i), sorted(&pw, i))).collect())
}

/// The squared distance of two signatures: per segment (reads with reads, writes with writes), over
/// the largest magnitudes both have.
fn signature_distance(a: &Signature, b: &Signature) -> f64 {
    let segment = |x: &Array1<f64>, y: &Array1<f64>| {
        let common = x.len().min(y.len());
        (x.slice(s![..common]).to_owned() - y.slice(s![..common])).mapv(|v| v * v).sum()
    };
    segment(&a.0, &b.0) + segment(&a.1, &b.1)
}

/// The alignment of body `from` to body `onto` (module note).
pub fn align(from: &BodyValues, onto: &BodyValues) -> Result<Alignment, String> {
    if from.law != onto.law || from.up.is_some() != onto.up.is_some() || from.gate_bias.is_some() != onto.gate_bias.is_some() || from.up_bias.is_some() != onto.up_bias.is_some() {
        return Err("bodies of different laws".into());
    }
    let (f_units, o_units) = (live_units(from), live_units(onto));
    let (k_from, k_onto) = (from.inputs.iter().filter(|u| **u).count(), onto.inputs.iter().filter(|u| **u).count());
    let (kout_from, kout_onto) = (from.outputs.iter().filter(|u| **u).count(), onto.outputs.iter().filter(|u| **u).count());
    let n = f_units.len().max(o_units.len());
    // The start: the assignment of the signatures, padded with unmatched units at zero cost.
    let (sf, so) = (signatures(&f_units)?, signatures(&o_units)?);
    let start = Array2::from_shape_fn((n, n), |(i, j)| match (sf.get(i), so.get(j)) {
        (Some(a), Some(b)) => signature_distance(a, b),
        _ => 0.0,
    });
    let mut assignment = hungarian(&start)?;
    let pairs_of = |assignment: &[usize]| -> Vec<(&Unit, &Unit)> {
        (0..f_units.len()).filter_map(|i| o_units.get(assignment[i]).map(|o| (&f_units[i].1, &o.1))).collect()
    };
    // The misfit of an assignment under a gauge, in units of `from`'s deviations.
    let total = |assignment: &[usize], a: &Array2<f64>, c: &Array2<f64>| -> f64 {
        (0..n)
            .map(|i| match (f_units.get(i), o_units.get(assignment[i])) {
                (Some((_, f)), Some((_, o))) => unit_misfit(f, &transformed(o, a, c), false).0,
                (Some((_, f)), None) => alone(f),
                (None, Some((_, o))) => alone(o),
                (None, None) => 0.0,
            })
            .sum()
    };
    // The misfit's rounding: `γ_n` over its compared values, each term three roundings (a
    // difference, its square and the quotient).
    let compared: usize = f_units.iter().chain(&o_units).map(|(_, u)| unit_misfit(u, u, false).1).sum();
    let units_off = (compared + 3) as f64 * f64::EPSILON;
    let gamma = units_off / (1.0 - units_off);
    let lowers = |next: f64, current: f64| next < current - gamma * current.abs();
    // The output gauge starts as the identity on the common coordinates, so the first input gauge
    // sees the gated maps at a start. Each step minimizes the misfit over its own block (the input
    // gauge, the output gauge, the assignment) and is taken only when it lowers the misfit by more
    // than the misfit's rounding; the alternation stops at the first step that does not: a local
    // optimum of the alternation, not necessarily the global one.
    let mut c = Array2::eye(kout_from.max(kout_onto)).slice(s![..kout_from, ..kout_onto]).to_owned();
    let mut a = input_gauge(&pairs_of(&assignment), &c, k_onto, k_from)?;
    let mut current = total(&assignment, &a, &c);
    loop {
        loop {
            let c_next = output_gauge(&pairs_of(&assignment), &a, kout_from, kout_onto)?;
            let a_next = input_gauge(&pairs_of(&assignment), &c_next, k_onto, k_from)?;
            let next = total(&assignment, &a_next, &c_next);
            if !lowers(next, current) {
                break;
            }
            (a, c, current) = (a_next, c_next, next);
        }
        let next = hungarian(&assignment_costs(&f_units, &o_units, &a, &c))?;
        let value = total(&next, &a, &c);
        if !lowers(value, current) {
            break;
        }
        (assignment, current) = (next, value);
    }
    // The misfit in units of both posteriors' variances: the matched pairs, then every unit left
    // unmatched. The assignment is a padded permutation, so a unit of `onto` assigned to a padding
    // row is unmatched.
    let (mut misfit, mut entries) = (0.0, 0);
    let mut units = vec![None; from.units.len()];
    let mut matched = vec![false; o_units.len()];
    for (i, (fi, f)) in f_units.iter().enumerate() {
        match o_units.get(assignment[i]) {
            Some((oj, o)) => {
                let (m, e) = unit_misfit(f, &transformed(o, &a, &c), true);
                misfit += m;
                entries += e;
                units[*fi] = Some(*oj);
                matched[assignment[i]] = true;
            }
            None => {
                misfit += alone(f);
                entries += unit_misfit(f, f, false).1;
            }
        }
    }
    for (_, o) in o_units.iter().zip(&matched).filter(|(_, m)| !**m).map(|(o, _)| o) {
        misfit += alone(o);
        entries += unit_misfit(o, o, false).1;
    }
    let embed = |small: &Array2<f64>, rows: &[bool], cols: &[bool]| {
        let (r, c): (Vec<usize>, Vec<usize>) = ((0..rows.len()).filter(|i| rows[*i]).collect(), (0..cols.len()).filter(|j| cols[*j]).collect());
        let mut full = Array2::zeros((rows.len(), cols.len()));
        for (a, &i) in r.iter().enumerate() {
            for (b, &j) in c.iter().enumerate() {
                full[[i, j]] = small[[a, b]];
            }
        }
        full
    };
    let gauge = k_onto * k_from + kout_from * kout_onto;
    Ok(Alignment {
        units,
        input: embed(&a, &onto.inputs, &from.inputs),
        output: embed(&c, &from.outputs, &onto.outputs),
        misfit,
        entries,
        gauge,
        live: [f_units.len(), o_units.len()],
    })
}

/// A minimum-cost perfect assignment of the rows of the square `cost` to its columns (Kuhn's
/// Hungarian method with potentials, `O(n³)`): `row → column`.
pub fn hungarian(cost: &Array2<f64>) -> Result<Vec<usize>, String> {
    let n = cost.nrows();
    if cost.ncols() != n || cost.iter().any(|c| !c.is_finite()) {
        return Err("an assignment needs a square finite cost".into());
    }
    // Potentials `u` (rows) and `v` (columns), one-based; `p[j]` the row assigned to column `j`
    // (0 for none), `way` the augmenting path.
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let current = cost[[i0 - 1, j - 1]] - u[i0] - v[j];
                    if current < minv[j] {
                        minv[j] = current;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            if j1 == 0 {
                return Err("the assignment found no augmenting column".into());
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        while j0 != 0 {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }
    let mut out = vec![0; n];
    for j in 1..=n {
        out[p[j] - 1] = j - 1;
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------- merging

/// Whether `name` is an up scale of call `site` that a merge made: `{site}.alpha{g}`, or the
/// earlier form `{site}.u{unit}.alpha`.
fn is_up_scale(site: &str, name: &str) -> bool {
    let Some(rest) = name.strip_prefix(site).and_then(|r| r.strip_prefix('.')) else { return false };
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    rest.strip_prefix("alpha").is_some_and(digits) || rest.strip_prefix('u').and_then(|r| r.strip_suffix(".alpha")).is_some_and(digits)
}

/// The nats of the operators the ownership map reads beyond the trainable ones (a gated unit's up
/// scales after a merge): no decoded value determines them, so each costs its message length in
/// the program's codec (`Operator::code_bits`).
fn ownership_nats(explanation: &Explanation) -> Result<f64, String> {
    let program = &explanation.artifact.program;
    let names: BTreeSet<&str> = explanation.artifact.owners.iter().flat_map(|o| o.left.iter().chain(&o.right)).map(String::as_str).collect();
    let mut bits = 0_u64;
    for name in names {
        let op = operator_index(program, name)?;
        if !explanation.trainable.contains(&op) {
            let (structure, reals) = program.operators[op].code_bits().map_err(error)?;
            bits += structure + reals;
        }
    }
    Ok(bits as f64 * LN_2)
}

/// `explanation` with every call of body `from` made a call of body `onto` with its bindings
/// `A R` and `W C` (`alignment` of `from` to `onto`), `from`'s rule and operators gone, and the
/// calls' records with it (module note). The matching of the units (`Alignment::matching_nats`)
/// and the up scales the ownership map gains (their literals) are charged in
/// `Explanation::fixed_nats`.
pub fn merge(explanation: &Explanation, calls: &[Call], from: &str, onto: &str, alignment: &Alignment) -> Result<(Explanation, Vec<Call>), String> {
    if from == onto {
        return Err("a body merges with another body".into());
    }
    let mut merged = explanation.clone();
    let program = &mut merged.artifact.program;
    let (from_rule, onto_rule) = (rule_index(program, from)?, rule_index(program, onto)?);
    let (from_ops, onto_ops) = (BodyOperators::of(program, from)?, BodyOperators::of(program, onto)?);
    let (z, y) = (program.operators[onto_ops.gate].cols.clone(), program.operators[onto_ops.out].rows.clone());
    let (a, c) = (&alignment.input, &alignment.output);
    if a.dim() != (z.width(), program.operators[from_ops.gate].cols.width()) || c.dim() != (program.operators[from_ops.out].rows.width(), y.width()) {
        return Err("an alignment of other bodies".into());
    }
    let moved = sites(program, from_rule)?;
    if moved.is_empty() {
        return Err(format!("{from} has no call"));
    }
    let mut bindings = Vec::new();
    for site in &moved {
        let (read, write) = (Arc::clone(&program.operators[site.read]), Arc::clone(&program.operators[site.write]));
        let provenance = Provenance::derived(&[&read.provenance, &program.operators[onto_ops.gate].provenance], format!("merge of {from} into {onto}"));
        program.operators[site.read] = Arc::new(dense(read.name.clone(), z.clone(), read.cols.clone(), a.dot(&read.matrix()), provenance.clone())?);
        program.operators[site.write] = Arc::new(dense(write.name.clone(), write.rows.clone(), y.clone(), write.matrix().dot(c), provenance)?);
        if let Node::Call { rule, .. } = &mut program.rules[site.rule].nodes[site.node] {
            *rule = onto_rule;
        }
        let call = read.name.strip_suffix(".read").ok_or("a read binding's name")?.to_string();
        bindings.push((call, site.read, site.write, read.cols.width(), write.rows.width()));
    }
    // `from`'s rule leaves; the rules after it move down by one.
    let rules: Vec<usize> = (0..program.rules.len()).map(|r| if r > from_rule { r - 1 } else { r }).collect();
    renumber_rules(program, &rules);
    program.rules.remove(from_rule);
    program.interfaces().map_err(error)?;
    // Groups: those of `from`'s operators and of the moved bindings leave; the moved bindings'
    // groups come back at their new widths, a zero row or column outside the explanation.
    let retired = from_ops.all();
    let rebound: Vec<usize> = bindings.iter().flat_map(|b| [b.1, b.2]).collect();
    let mut index = BTreeMap::new();
    let (mut groups, mut reference) = (Vec::new(), Vec::new());
    for (g, group) in explanation.groups.iter().enumerate() {
        if group.cells.iter().any(|cell| retired.contains(&cell.operator) || rebound.contains(&cell.operator)) {
            continue;
        }
        index.insert(g, groups.len());
        groups.push(group.clone());
        reference.push(explanation.reference[g]);
    }
    let mut removed: Vec<usize> = explanation.removed.iter().filter_map(|g| index.get(g).copied()).collect();
    for (call, read, write, d_in, d_out) in &bindings {
        let (r, w) = (program.operators[*read].matrix(), program.operators[*write].matrix());
        for group in binding_groups(call, *read, *write, z.width(), y.width(), *d_in, *d_out) {
            let cells = &group.cells[0];
            let zero = if cells.operator == *read { r.row(cells.rows[0]).iter().all(|v| *v == 0.0) } else { w.column(cells.cols.start).iter().all(|v| *v == 0.0) };
            if zero {
                removed.push(groups.len());
            }
            reference.push(binding_reference(&group));
            groups.push(group);
        }
    }
    removed.sort_unstable();
    for layer in &mut merged.layers {
        for (planes, values) in &mut layer.heads {
            *planes = planes.iter().filter_map(|g| index.get(g).copied()).collect();
            *values = values.iter().filter_map(|g| index.get(g).copied()).collect();
        }
        for function in &mut layer.functions {
            *function = function.iter().filter_map(|g| index.get(g).copied()).collect();
        }
    }
    merged.groups = groups;
    merged.reference = reference;
    merged.removed = removed;
    merged.trainable.retain(|op| !retired.contains(op));
    // The calls' choice of body changes; `from`'s widths are no longer sent.
    let program = &merged.artifact.program;
    let from_widths = crate::codec::elias_delta_len_bits(explanation.artifact.program.operators[from_ops.gate].cols.width() as u64).map_err(error)?
        + crate::codec::elias_delta_len_bits(explanation.artifact.program.operators[from_ops.out].rows.width() as u64).map_err(error)?;
    merged.fixed_nats += assignment_nats(program)? - assignment_nats(&explanation.artifact.program)? - from_widths as f64 * LN_2 + alignment.matching_nats();
    // Owners of `from`'s blocks now own `onto`'s block of the matched unit at the same call, whose
    // bindings took the gauge (`g_i = g_j A`, so `a_i = g_j (A R)`; `u_i = C u_j`, so `W u_i =
    // (W C) u_j`), and for a gated law the unit's up scale `α` (`b_i = α b_j A`, `u_i = C u_j / α`),
    // kept as 1 × 1 operators of the call that no node reads: one scalar per call and unit, with
    // its inverse; a unit an earlier merge gave one keeps it, multiplied by this merge's `α`, so its
    // ownership expression stays one uniquely named factor. A native function whose unit is
    // unmatched leaves `P`: it is owned by its zeroed block of the native MLP's operator, as a
    // removed function is.
    let alphas = up_scales(&artifact_values(&explanation.artifact.program, from)?, &artifact_values(&explanation.artifact.program, onto)?, alignment);
    let gated = from_ops.up.is_some();
    let pieces = [
        (Some(from_ops.gate), Some(onto_ops.gate), Piece::Gate, "gate"),
        (from_ops.gate_bias, onto_ops.gate_bias, Piece::GateBias, "gate_bias"),
        (from_ops.up, onto_ops.up, Piece::Up, "up"),
        (from_ops.up_bias, onto_ops.up_bias, Piece::UpBias, "up_bias"),
        (Some(from_ops.out), Some(onto_ops.out), Piece::Out, "out"),
    ];
    let name = |op: usize| explanation.artifact.program.operators[op].name.clone();
    let (k_onto, k_out_onto) = (z.width(), y.width());
    // Per call and unit of `from`: its up scale's name, and whether an earlier merge made it.
    let mut scalars: BTreeMap<(String, usize), (String, bool)> = BTreeMap::new();
    let mut owners = std::mem::take(&mut merged.artifact.owners);
    for owner in &mut owners {
        let Some((_, Some(onto_op), piece, part)) = pieces.iter().find(|(f, ..)| f.is_some_and(|f| name(f) == owner.operator)) else { continue };
        let unit = if *piece == Piece::Out { owner.cols.start } else { owner.rows.start };
        match alignment.units.get(unit).copied().flatten() {
            Some(j) => {
                let (rows, cols) = match piece {
                    Piece::Gate | Piece::Up => (j..j + 1, 0..k_onto),
                    Piece::GateBias | Piece::UpBias => (j..j + 1, 0..1),
                    Piece::Out => (0..k_out_onto, j..j + 1),
                };
                owner.operator = name(*onto_op);
                owner.rows = rows;
                owner.cols = cols;
                if gated && matches!(piece, Piece::Up | Piece::UpBias | Piece::Out) {
                    let key = (owner.site.clone(), unit);
                    if !scalars.contains_key(&key) {
                        let earlier = if *piece == Piece::Out {
                            owner.right.iter().find_map(|n| n.strip_suffix("_inverse").filter(|a| is_up_scale(&owner.site, a)).map(str::to_string))
                        } else {
                            owner.left.iter().find(|n| is_up_scale(&owner.site, n)).cloned()
                        };
                        let named = match earlier {
                            Some(name) => (name, true),
                            None => {
                                let taken = |name: &str| operator_named(&merged.artifact.program, name).is_some() || scalars.values().any(|(n, _)| n == name);
                                let fresh = (0..).map(|g| format!("{}.alpha{g}", owner.site)).find(|n| !taken(n)).ok_or("no free scale name")?;
                                (fresh, false)
                            }
                        };
                        scalars.insert(key.clone(), named);
                    }
                    let alpha = &scalars[&key].0;
                    let (list, name) = if *piece == Piece::Out { (&mut owner.right, format!("{alpha}_inverse")) } else { (&mut owner.left, alpha.clone()) };
                    if !list.contains(&name) {
                        list.push(name);
                    }
                }
            }
            None => {
                let layer = owner.site.strip_prefix("library.l").and_then(|r| r.split_once('.')).map(|(l, _)| l.to_string()).ok_or("a call's name")?;
                let mlp = format!("library.l{layer}.mlp");
                owner.operator = format!("{mlp}.{part}");
                owner.rows = owner.native_rows.clone();
                owner.cols = owner.native_cols.clone();
                owner.left.clear();
                owner.right.clear();
                owner.site = mlp.clone();
                owner.body = mlp;
                continue;
            }
        }
        owner.body = onto.to_string();
    }
    merged.artifact.owners = owners;
    let one = units(1)?;
    for ((_, unit), (alpha, earlier)) in &scalars {
        let mut value = alphas.get(*unit).copied().unwrap_or(1.0);
        let program = &mut merged.artifact.program;
        let provenance = Provenance::derived(&[], format!("merge of {from} into {onto}: a unit's up scale"));
        for (scalar, inverse) in [(alpha.clone(), false), (format!("{alpha}_inverse"), true)] {
            let place = if *earlier { Some(operator_index(program, &scalar)?) } else { None };
            if !inverse && let Some(at) = place {
                value *= program.operators[at].matrix()[[0, 0]];
            }
            let v = if inverse { 1.0 / value } else { value };
            let op = Arc::new(dense(scalar, one.clone(), one.clone(), Array2::from_elem((1, 1), v), provenance.clone())?);
            match place {
                Some(at) => program.operators[at] = op,
                None => program.operators.push(op),
            }
        }
    }
    merged.fixed_nats += ownership_nats(&merged)? - ownership_nats(explanation)?;
    let calls = calls
        .iter()
        .map(|call| {
            let mut call = call.clone();
            if call.body == from {
                call.body = onto.to_string();
                for (_, unit) in &mut call.replaced {
                    *unit = unit.and_then(|j| alignment.units.get(j).copied().flatten());
                }
            }
            call
        })
        .collect();
    Ok((merged, calls))
}

// -------------------------------------------------------------------------------------- regions

/// `a` over the posterior deviations `exp(log_sd)` in the separable form that keeps its rank:
/// `log σ_ij ≈ r_i + c_j` fitted by least squares (`r_i` the row means of `log σ`, `c_j` the
/// column means of what is left; exact where the deviations are a row scale times a column scale),
/// and `diag(e^{−r}) a diag(e^{−c})`; with the column scales `c`. Dividing each entry by its own
/// deviation would not keep the rank (`100 [[1,1],[1,1]]` over `[[1,1],[1,2]]` has two singular
/// values far from zero).
fn separable(a: &Array2<f64>, log_sd: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let rows = log_sd.mean_axis(Axis(1)).unwrap_or_else(|| Array1::zeros(a.nrows()));
    let columns = (log_sd - &rows.view().insert_axis(Axis(1))).mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(a.ncols()));
    let whitened = Array2::from_shape_fn(a.dim(), |(i, j)| a[[i, j]] * (-rows[i] - columns[j]).exp());
    (whitened, columns)
}

/// The noise-edge heuristic for a `rows × cols` matrix: `√rows + √cols`, the asymptotic largest
/// singular value of a matrix of independent unit-variance entries (Bai and Yin 1988). It is not a
/// finite-sample bound, and whitened posterior means are not drawn from that null: it orders
/// proposals and certifies no rank.
fn noise_edge(rows: usize, cols: usize) -> f64 {
    (rows as f64).sqrt() + (cols as f64).sqrt()
}

/// A heuristic count of the parameters a rewrite of the functions with posterior-whitened reads
/// `reads` (their gate rows, and up rows when gated, in the separable whitening of [`separable`];
/// `parts` rows per function) and whitened writes `writes` (their output columns, one row each)
/// saves: `|S| (parts d + d′)` native entries against the body's `k (d + parts |S|) + k′ (d′ + |S|)`,
/// with `k` and `k′` the numbers of whitened singular values above the noise edge ([`noise_edge`]).
fn saving(reads: &Array2<f64>, writes: &Array2<f64>, parts: usize) -> Result<f64, String> {
    let resolved_rank = |m: &Array2<f64>| -> Result<usize, String> {
        let edge = noise_edge(m.nrows(), m.ncols());
        Ok(svd(m.view(), false).map_err(error)?.singular_values.iter().filter(|s| **s > edge).count())
    };
    let n = writes.nrows() as f64;
    let (d, d_out) = (reads.ncols() as f64, writes.ncols() as f64);
    let (k, k_out) = (resolved_rank(reads)? as f64, resolved_rank(writes)? as f64);
    Ok(n * (parts as f64 * d + d_out) - (k * (d + parts as f64 * n) + k_out * (d_out + n)))
}

/// The candidate regions of layer `layer`'s MLP among the native functions `pool` (those carrying
/// flow) at `posterior` (module note), each with its heuristic saving (`saving`), largest first:
/// from each function a group is grown by adding, one at a time, the function whose union's rewrite
/// saves the most, and the best group along the way is that function's candidate; the candidates
/// are taken in order of their saving, each disjoint from those before it, and none is left out for
/// its saving. Parallel
/// functions of one body read and write the same few directions, so their union saves what each
/// alone cannot; a function reading a direction of its own adds a coordinate to each binding.
pub fn regions(explanation: &Explanation, posterior: &Posterior, layer: usize, pool: &[usize]) -> Result<Vec<(Vec<usize>, f64)>, String> {
    let program = &explanation.artifact.program;
    let mlp = format!("library.l{layer}.mlp");
    let maps: Vec<usize> = ["gate", "up"].iter().filter_map(|part| operator_named(program, &format!("{mlp}.{part}"))).collect();
    let out = operator_index(program, &format!("{mlp}.out"))?;
    let means = posterior.means();
    let at = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("{mlp}: operator {op} is not trainable"));
    let known = &explanation.layers.get(layer).ok_or_else(|| format!("no layer {layer}"))?.functions;
    // A function's row (or column) of an operator's means and of its log deviations.
    let entries = |i: usize, op: usize, row: bool| -> Result<(Array1<f64>, Array1<f64>), String> {
        let p = at(op)?;
        let (mean, log_sd) = (&means[p], &posterior.log_sd[p]);
        Ok(if row { (mean.row(i).to_owned(), log_sd.row(i).to_owned()) } else { (mean.column(i).to_owned(), log_sd.column(i).to_owned()) })
    };
    // Each function of the pool in the explanation: its read rows and its write column.
    let mut functions = Vec::new();
    for &i in pool {
        let groups = known.get(i).ok_or_else(|| format!("layer {layer} has no function {i}"))?;
        if groups.iter().all(|g| posterior.active[*g]) {
            let reads = maps.iter().map(|op| entries(i, *op, true)).collect::<Result<Vec<_>, _>>()?;
            functions.push((i, reads, entries(i, out, false)?));
        }
    }
    let parts = maps.len();
    // A group's reads and writes, means and log deviations, stacked.
    type Block = (Array2<f64>, Array2<f64>);
    let stack = |members: &[usize]| -> (Block, Block) {
        let d = functions[members[0]].1[0].0.len();
        let read = |k: usize| Array2::from_shape_fn((members.len() * parts, d), |(r, c)| if k == 0 { functions[members[r / parts]].1[r % parts].0[c] } else { functions[members[r / parts]].1[r % parts].1[c] });
        let d_out = functions[members[0]].2.0.len();
        let write = |k: usize| Array2::from_shape_fn((members.len(), d_out), |(r, c)| if k == 0 { functions[members[r]].2.0[c] } else { functions[members[r]].2.1[c] });
        ((read(0), read(1)), (write(0), write(1)))
    };
    let value = |members: &[usize]| -> Result<f64, String> {
        let ((reads, read_sd), (writes, write_sd)) = stack(members);
        saving(&separable(&reads, &read_sd).0, &separable(&writes, &write_sd).0, parts)
    };
    // A group's resolved read and write directions (right singular vectors above the noise edge).
    let resolved_basis = |m: &Array2<f64>| -> Result<Array2<f64>, String> {
        let edge = noise_edge(m.nrows(), m.ncols());
        let decomposition = svd(m.view(), false).map_err(error)?;
        let k = decomposition.singular_values.iter().filter(|s| **s > edge).count();
        Ok(decomposition.vt.slice(s![..k, ..]).to_owned())
    };
    // Whether `block`'s rows lie within `basis`'s span up to noise: the largest singular value of
    // the residual is within the noise edge of a same-shaped block in the complement.
    let within = |block: &Array2<f64>, basis: &Array2<f64>| -> Result<bool, String> {
        let residual = block - &block.dot(&basis.t()).dot(basis);
        let complement = block.ncols() - basis.nrows();
        let edge = noise_edge(block.nrows(), complement);
        Ok(svd(residual.view(), false).map_err(error)?.singular_values.first().is_none_or(|s| *s <= edge))
    };
    // A function's block whitened in a group's column scales `c`, with its own row scales.
    let whitened_in = |(a, log_sd): &Block, columns: &Array1<f64>| -> Array2<f64> {
        let rows = (log_sd - &columns.view().insert_axis(Axis(0))).mean_axis(Axis(1)).unwrap_or_else(|| Array1::zeros(a.nrows()));
        Array2::from_shape_fn(a.dim(), |(i, j)| a[[i, j]] * (-rows[i] - columns[j]).exp())
    };
    // The functions outside `members` whose reads and writes lie within the members' resolved
    // directions: those the group could take in without a new coordinate.
    let inliers = |members: &[usize]| -> Result<usize, String> {
        let ((reads, read_sd), (writes, write_sd)) = stack(members);
        let ((reads, read_columns), (writes, write_columns)) = (separable(&reads, &read_sd), separable(&writes, &write_sd));
        let (read_basis, write_basis) = (resolved_basis(&reads)?, resolved_basis(&writes)?);
        let mut count = 0;
        for f in (0..functions.len()).filter(|f| !members.contains(f)) {
            let (r, w) = stack(&[f]);
            if within(&whitened_in(&r, &read_columns), &read_basis)? && within(&whitened_in(&w, &write_columns), &write_basis)? {
                count += 1;
            }
        }
        Ok(count)
    };
    // From each function, the group grown by adding the function whose union saves most (ties
    // broken by the functions the union could take in without a new coordinate), over the whole
    // path; the best prefix of each path is a candidate. A body's functions save together what no
    // pair of them saves: two rows of a two-dimensional read already have its rank.
    let n = functions.len();
    let mut candidates: Vec<(f64, Vec<usize>)> = (0..n)
        .into_par_iter()
        .map(|seed| -> Result<(f64, Vec<usize>), String> {
            let mut members = vec![seed];
            let mut best = (value(&members)?, members.clone());
            let mut rest: Vec<usize> = (0..n).filter(|f| *f != seed).collect();
            while !rest.is_empty() {
                let mut pick: Option<((f64, usize), usize)> = None;
                for (at, &f) in rest.iter().enumerate() {
                    members.push(f);
                    let score = (value(&members)?, inliers(&members)?);
                    members.pop();
                    if pick.is_none_or(|(b, _)| score.0 > b.0 || (score.0 == b.0 && score.1 > b.1)) {
                        pick = Some((score, at));
                    }
                }
                let Some(((v, _), at)) = pick else { break };
                members.push(rest.remove(at));
                if v > best.0 {
                    best = (v, members.clone());
                }
            }
            Ok(best)
        })
        .collect::<Result<_, String>>()?;
    // The candidates, largest saving first, each disjoint from those taken.
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut taken = vec![false; n];
    let mut out = Vec::new();
    for (saved, members) in candidates {
        if members.iter().all(|f| !taken[*f]) {
            members.iter().for_each(|f| taken[*f] = true);
            let mut region: Vec<usize> = members.iter().map(|f| functions[*f].0).collect();
            region.sort_unstable();
            out.push((region, saved));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------- reuse by gradient

/// A part of a body unit's parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Piece {
    Gate,
    GateBias,
    Up,
    UpBias,
    Out,
}

/// One entry of a body's parameters as the mixture reads it: its piece, unit and coordinate, its
/// operator (an index into `Explanation::trainable`), row, column and prior group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    piece: Piece,
    unit: usize,
    coordinate: usize,
    operator: usize,
    row: usize,
    col: usize,
    group: usize,
}

/// A body's operators by trainable index, and its entries in groups of the explanation: per unit
/// its gate row, gate bias, up row, up bias and output column.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Layout {
    operators: Vec<(Piece, usize)>,
    entries: Vec<Entry>,
}

impl Layout {
    fn of(explanation: &Explanation, body: &str) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        let ops = BodyOperators::of(program, body)?;
        let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("{body}: operator {op} is not trainable"));
        let pieces = [(Piece::Gate, Some(ops.gate)), (Piece::GateBias, ops.gate_bias), (Piece::Up, ops.up), (Piece::UpBias, ops.up_bias), (Piece::Out, Some(ops.out))];
        let operators: Vec<(Piece, usize, usize)> = pieces.iter().filter_map(|(p, op)| op.map(|op| position(op).map(|i| (*p, op, i)))).collect::<Result<_, _>>()?;
        // Each entry's group, from the explanation's groups of these operators.
        let mut group_of: BTreeMap<(usize, usize, usize), usize> = BTreeMap::new();
        for (g, group) in explanation.groups.iter().enumerate() {
            for cell in group.cells.iter().filter(|c| operators.iter().any(|o| o.1 == c.operator)) {
                for &row in &cell.rows {
                    for col in cell.cols.clone() {
                        group_of.insert((cell.operator, row, col), g);
                    }
                }
            }
        }
        let mut entries = Vec::new();
        let units = program.operators[ops.gate].rows.width();
        for unit in 0..units {
            for &(piece, op, i) in &operators {
                let width = match piece {
                    Piece::Gate | Piece::Up => program.operators[op].cols.width(),
                    Piece::GateBias | Piece::UpBias => 1,
                    Piece::Out => program.operators[op].rows.width(),
                };
                for coordinate in 0..width {
                    let (row, col) = if piece == Piece::Out { (coordinate, unit) } else { (unit, coordinate) };
                    let group = *group_of.get(&(op, row, col)).ok_or_else(|| format!("{body}: an entry in no group"))?;
                    entries.push(Entry { piece, unit, coordinate, operator: i, row, col, group });
                }
            }
        }
        Ok(Self { operators: operators.into_iter().map(|(p, _, i)| (p, i)).collect(), entries })
    }

    fn operator(&self, piece: Piece) -> Option<usize> {
        self.operators.iter().find(|(p, _)| *p == piece).map(|(_, i)| *i)
    }

    /// The body's units, input width and output width.
    fn widths(&self) -> (usize, usize, usize) {
        let units = self.entries.iter().map(|e| e.unit + 1).max().unwrap_or(0);
        let count = |piece: Piece| self.entries.iter().filter(|e| e.piece == piece && e.unit == 0).count();
        (units, count(Piece::Gate), count(Piece::Out))
    }
}

/// One component of a body's mixture prior: an earlier body, the gauge relating them (fixed for an
/// epoch: the alignment of the target to it and each target unit's up scale `α`) and its logit.
/// Its centre is the earlier body under the gauge with no further scale: a common scale of a
/// body's parameters is not one of its symmetries, so a merge could not carry it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyComponent {
    pub body: String,
    pub alignment: Alignment,
    pub up_scales: Vec<f64>,
    pub logit: f64,
}

/// The mixture prior of one body's parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyTarget {
    pub body: String,
    /// The earlier bodies it may choose among.
    pub choices: usize,
    pub zero_logit: f64,
    pub components: Vec<BodyComponent>,
    /// `ln s²`.
    pub log_variance: f64,
}

impl BodyTarget {
    /// The mixture weights `π`, the zero component's first.
    pub fn weights(&self) -> Result<Vec<f64>, String> {
        let logits: Vec<f64> = std::iter::once(self.zero_logit).chain(self.components.iter().map(|c| c.logit)).collect();
        Ok(gam_math::categorical::log_softmax(&logits).map_err(error)?.into_iter().map(f64::exp).collect())
    }
}

/// Adam's moments of one parameter.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Moment {
    first: f64,
    second: f64,
}

/// The Adam moments [`BodyMixture`]'s step keeps per component: its logit.
const MOMENTS_PER_COMPONENT: usize = 1;

/// The mixture prior over bodies (module note, # Reuse by gradient).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyMixture {
    pub targets: Vec<BodyTarget>,
    steps: crate::library_mixture::Steps,
    taken: u64,
    /// Per target, Adam's moments of its zero logit, then per component its logit, then its `ln s²`.
    moments: Vec<Vec<Moment>>,
    layouts: BTreeMap<String, Layout>,
    /// Each target body's groups' cells (trainable index, rows, columns), for their variances.
    cells: BTreeMap<usize, Vec<(usize, Vec<usize>, std::ops::Range<usize>)>>,
}

impl BodyMixture {
    /// The mixture prior of every body of `explanation` (bodies in order of their numbers), stepped
    /// with `steps`; the components are chosen by the first epoch.
    pub fn new(explanation: &Explanation, steps: crate::library_mixture::Steps) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        let mut bodies: Vec<(usize, String)> = program.rules.iter().filter_map(|r| r.name.strip_prefix("library.body")?.parse::<usize>().ok().map(|i| (i, r.name.clone()))).collect();
        bodies.sort_unstable();
        let mut layouts = BTreeMap::new();
        for (_, body) in &bodies {
            layouts.insert(body.clone(), Layout::of(explanation, body)?);
        }
        let position = |op: usize| explanation.trainable.iter().position(|t| *t == op);
        let mut cells = BTreeMap::new();
        for layout in layouts.values() {
            for entry in &layout.entries {
                cells.entry(entry.group).or_insert_with(|| {
                    explanation.groups[entry.group].cells.iter().filter_map(|c| position(c.operator).map(|i| (i, c.rows.clone(), c.cols.clone()))).collect()
                });
            }
        }
        let targets: Vec<BodyTarget> = bodies.iter().enumerate().map(|(t, (_, body))| BodyTarget { body: body.clone(), choices: t, zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 }).collect();
        let moments = targets.iter().map(|_| vec![Moment::default(); 2]).collect();
        Ok(Self { targets, steps, taken: 0, moments, layouts, cells })
    }

    /// Each group's empirical-Bayes variance `v_G` at `posterior`.
    fn variance(&self, group: usize, posterior: &Posterior) -> f64 {
        let (mut count, mut second) = (0.0, 0.0);
        for (i, rows, cols) in self.cells.get(&group).map(Vec::as_slice).unwrap_or_default() {
            for &r in rows {
                for c in cols.clone() {
                    let (m, s) = (posterior.mean[*i][[r, c]], posterior.log_sd[*i][[r, c]]);
                    count += 1.0;
                    second += m * m + (2.0 * s).exp();
                }
            }
        }
        second / count
    }

    /// The entries of target `t` in the explanation at `posterior`.
    fn live(&self, t: usize, posterior: &Posterior) -> Result<Vec<Entry>, String> {
        let layout = self.layouts.get(&self.targets[t].body).ok_or("a target's layout")?;
        Ok(layout.entries.iter().filter(|e| posterior.active[e.group]).copied().collect())
    }

    /// Component `component`'s prediction of the target entries `entries` from its body's values
    /// `theta` (by trainable index).
    fn predicted(&self, component: &BodyComponent, entries: &[Entry], theta: &BTreeMap<usize, Array2<f64>>) -> Result<Array1<f64>, String> {
        let layout = self.layouts.get(&component.body).ok_or("a component's layout")?;
        let get = |piece: Piece| -> Result<Option<&Array2<f64>>, String> {
            layout.operator(piece).map(|i| theta.get(&i).ok_or_else(|| "a component's sample".to_string())).transpose()
        };
        let (gate, bias, up, up_bias, out) = (get(Piece::Gate)?, get(Piece::GateBias)?, get(Piece::Up)?, get(Piece::UpBias)?, get(Piece::Out)?);
        let (a, c) = (&component.alignment.input, &component.alignment.output);
        let mut out_vector = Array1::zeros(entries.len());
        for (k, e) in entries.iter().enumerate() {
            let Some(j) = component.alignment.units.get(e.unit).copied().flatten() else { continue };
            let alpha = component.up_scales.get(e.unit).copied().unwrap_or(1.0);
            out_vector[k] = match (e.piece, gate, bias, up, up_bias, out) {
                (Piece::Gate, Some(g), ..) => g.row(j).dot(&a.column(e.coordinate)),
                (Piece::GateBias, _, Some(b), ..) => b[[j, 0]],
                (Piece::Up, _, _, Some(u), ..) => alpha * u.row(j).dot(&a.column(e.coordinate)),
                (Piece::UpBias, _, _, _, Some(b), _) => alpha * b[[j, 0]],
                (Piece::Out, .., Some(o)) => c.row(e.coordinate).dot(&o.column(j)) / alpha,
                _ => return Err("a component of another law than its target".into()),
            };
        }
        Ok(out_vector)
    }

    /// The derivative in the component's body's operators of `w · predicted`.
    fn predicted_gradient(&self, component: &BodyComponent, entries: &[Entry], w: &Array1<f64>, gradient: &mut BTreeMap<usize, Array2<f64>>, theta: &BTreeMap<usize, Array2<f64>>) -> Result<(), String> {
        let layout = self.layouts.get(&component.body).ok_or("a component's layout")?;
        let (a, c) = (&component.alignment.input, &component.alignment.output);
        for (k, e) in entries.iter().enumerate() {
            let Some(j) = component.alignment.units.get(e.unit).copied().flatten() else { continue };
            let alpha = component.up_scales.get(e.unit).copied().unwrap_or(1.0);
            let piece = e.piece;
            let i = layout.operator(piece).ok_or("a component of another law than its target")?;
            let shape = theta.get(&i).ok_or("a component's sample")?.dim();
            let g = gradient.entry(i).or_insert_with(|| Array2::zeros(shape));
            match piece {
                Piece::Gate => g.row_mut(j).scaled_add(w[k], &a.column(e.coordinate)),
                Piece::Up => g.row_mut(j).scaled_add(w[k] * alpha, &a.column(e.coordinate)),
                Piece::GateBias => g[[j, 0]] += w[k],
                Piece::UpBias => g[[j, 0]] += w[k] * alpha,
                Piece::Out => g.column_mut(j).scaled_add(w[k] / alpha, &c.row(e.coordinate)),
            }
        }
        Ok(())
    }

    /// The targets whose one component holds more than half of the mixture weight, with it.
    pub fn dominant(&self, posterior: &Posterior) -> Result<Vec<(usize, usize)>, String> {
        let mut out = Vec::new();
        for (t, target) in self.targets.iter().enumerate() {
            if self.live(t, posterior)?.is_empty() {
                continue;
            }
            let weights = target.weights()?;
            if let Some(j) = (1..weights.len()).find(|&j| weights[j] > 0.5) {
                out.push((t, j - 1));
            }
        }
        Ok(out)
    }

    /// `explanation` and its `calls` with every dominant component made exact (module note): the
    /// target body merged into the component's body with its alignment, each body in one merge.
    pub fn harden(&self, explanation: &Explanation, calls: &[Call], posterior: &Posterior) -> Result<(Explanation, Vec<Call>, Vec<(String, String)>), String> {
        let (mut out, mut calls) = (explanation.clone(), calls.to_vec());
        let mut involved: Vec<String> = Vec::new();
        let mut merged = Vec::new();
        for (t, j) in self.dominant(posterior)? {
            let (from, component) = (&self.targets[t].body, &self.targets[t].components[j]);
            if involved.contains(from) || involved.contains(&component.body) {
                continue;
            }
            (out, calls) = merge(&out, &calls, from, &component.body, &component.alignment)?;
            involved.extend([from.clone(), component.body.clone()]);
            merged.push((from.clone(), component.body.clone()));
        }
        Ok((out, calls, merged))
    }

    /// One Adam step of the mixture's own parameters along `gradients` (per target, in the moments'
    /// order).
    fn learn(&mut self, gradients: &[Vec<f64>]) {
        self.taken += 1;
        let crate::library_mixture::Steps { rate, beta1, beta2, epsilon } = self.steps;
        let (c1, c2) = (1.0 - beta1.powf(self.taken as f64), 1.0 - beta2.powf(self.taken as f64));
        for (t, gradient) in gradients.iter().enumerate() {
            if gradient.is_empty() {
                continue;
            }
            let step = |moment: &mut Moment, value: &mut f64, g: f64| {
                moment.first = beta1 * moment.first + (1.0 - beta1) * g;
                moment.second = beta2 * moment.second + (1.0 - beta2) * g * g;
                *value -= rate * (moment.first / c1) / ((moment.second / c2).sqrt() + epsilon);
            };
            let (target, moments) = (&mut self.targets[t], &mut self.moments[t]);
            step(&mut moments[0], &mut target.zero_logit, gradient[0]);
            for (j, component) in target.components.iter_mut().enumerate() {
                step(&mut moments[1 + j], &mut component.logit, gradient[1 + j]);
            }
            let last = moments.len() - 1;
            step(&mut moments[last], &mut target.log_variance, gradient[gradient.len() - 1]);
        }
    }
}

/// Each target unit's up scale `α` relating it to the matched unit of the component under
/// `alignment` (gated laws; module note): the geometric mean of the up rows' ratio of norms and the
/// output columns' inverse ratio, signed by the up rows' inner product; 1 for an ungated law or a
/// unit with a zero row.
fn up_scales(target: &BodyValues, component: &BodyValues, alignment: &Alignment) -> Vec<f64> {
    let (Some(up_t), Some(up_c)) = (&target.up, &component.up) else { return vec![1.0; alignment.units.len()] };
    alignment
        .units
        .iter()
        .enumerate()
        .map(|(i, j)| {
            let Some(j) = j else { return 1.0 };
            let (b, b_c) = (up_t.mean.row(i).to_owned(), up_c.mean.row(*j).dot(&alignment.input));
            let (u, u_c) = (target.out.mean.column(i).to_owned(), alignment.output.dot(&component.out.mean.column(*j)));
            let norm = |v: &Array1<f64>| v.dot(v).sqrt();
            let (nb, nbc, nu, nuc) = (norm(&b), norm(&b_c), norm(&u), norm(&u_c));
            if nb == 0.0 || nbc == 0.0 || nu == 0.0 || nuc == 0.0 {
                return 1.0;
            }
            let sign = if b.dot(&b_c) < 0.0 { -1.0 } else { 1.0 };
            sign * (nb * nuc / (nbc * nu)).sqrt()
        })
        .collect()
}

impl crate::library_mdl::PriorTerm for BodyMixture {
    fn operators(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.layouts.values().flat_map(|l| l.operators.iter().map(|(_, i)| *i)).collect();
        out.extend(self.cells.values().flatten().map(|c| c.0));
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Every target's components re-chosen at `posterior`: each earlier body of the same law the
    /// target aligns to with evidence, with its alignment and up scales; a component kept keeps its
    /// logit and scale.
    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        let values: BTreeMap<String, BodyValues> =
            self.targets.iter().map(|t| Ok((t.body.clone(), body_values(explanation, posterior, &t.body)?))).collect::<Result<_, String>>()?;
        for t in 0..self.targets.len() {
            let old = std::mem::take(&mut self.targets[t].components);
            let zero = self.targets[t].zero_logit;
            let mut moments = vec![self.moments[t][0]];
            let mut components = Vec::new();
            let target = &values[&self.targets[t].body];
            for earlier in &self.targets[..t] {
                let candidate = &values[&earlier.body];
                let Ok(alignment) = align(target, candidate) else { continue };
                if !alignment.constrains() {
                    continue;
                }
                let up = up_scales(target, candidate, &alignment);
                let (component, moment) = match old.iter().position(|c| c.body == earlier.body) {
                    Some(at) => (BodyComponent { alignment, up_scales: up, ..old[at].clone() }, self.moments[t][1 + at]),
                    None => (BodyComponent { body: earlier.body.clone(), alignment, up_scales: up, logit: zero }, Moment::default()),
                };
                components.push(component);
                moments.push(moment);
            }
            if old.is_empty() && !components.is_empty() {
                // A new target's variance starts at its entries' mean posterior variance.
                let live = self.live(t, posterior)?;
                let mean: f64 = live.iter().map(|e| (2.0 * posterior.log_sd[e.operator][[e.row, e.col]]).exp()).sum::<f64>() / live.len().max(1) as f64;
                if mean > 0.0 {
                    self.targets[t].log_variance = mean.ln();
                }
            }
            moments.push(*self.moments[t].last().ok_or("moments")?);
            self.targets[t].components = components;
            self.moments[t] = moments;
        }
        Ok(())
    }

    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
        let mut value = 0.0;
        let mut gradient: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        let mut learned = vec![Vec::new(); self.targets.len()];
        for t in 0..self.targets.len() {
            let target = &self.targets[t];
            let entries = self.live(t, posterior)?;
            if target.components.is_empty() || entries.is_empty() {
                continue;
            }
            let g = Array1::from_iter(entries.iter().map(|e| theta.get(&e.operator).map_or(0.0, |m| m[[e.row, e.col]])));
            let v = Array1::from_iter(entries.iter().map(|e| self.variance(e.group, posterior)));
            let predictions = target.components.iter().map(|c| self.predicted(c, &entries, theta)).collect::<Result<Vec<_>, _>>()?;
            let writes: Vec<(ndarray::ArrayView1<'_, f64>, f64)> = predictions.iter().map(|p| (p.view(), 1.0)).collect();
            let logits: Vec<f64> = std::iter::once(target.zero_logit).chain(target.components.iter().map(|c| c.logit)).collect();
            let found = crate::library_mixture::term(g.view(), &writes, &logits, target.log_variance, v.view())?;
            value += found.value;
            for (k, e) in entries.iter().enumerate() {
                let shape = theta.get(&e.operator).ok_or("a target's sample")?.dim();
                gradient.entry(e.operator).or_insert_with(|| Array2::zeros(shape))[[e.row, e.col]] += found.target[k];
            }
            for (component, derivative) in target.components.iter().zip(&found.writes) {
                self.predicted_gradient(component, &entries, derivative, &mut gradient, theta)?;
            }
            let mut own = vec![found.logits[0]];
            own.extend(&found.logits[1..]);
            own.push(found.log_variance);
            learned[t] = own;
        }
        if learn {
            self.learn(&learned);
        }
        Ok((value, gradient))
    }

    /// Per target with components: its `K` components among its `n` earlier bodies (`ln C(n, K)`
    /// nats); each component's matching of the target's live units to the earlier body's
    /// (`Alignment::matching_nats`); and its `K` logits, its variance and every component's gauge (its alignment's
    /// fitted entries and, gated, its up scales), each at the precision of a value estimated from
    /// the target's live entries (`½ ln |G|` nats).
    fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
        use statrs::function::gamma::ln_gamma;
        let mut total = 0.0;
        for (t, target) in self.targets.iter().enumerate() {
            let size = self.live(t, posterior)?.len() as f64;
            if target.components.is_empty() || size == 0.0 {
                continue;
            }
            let (n, k) = (target.choices as f64, target.components.len() as f64);
            let mut gauges = 0.0;
            for component in &target.components {
                total += component.alignment.matching_nats();
                gauges += (component.alignment.gauge + if component.up_scales.iter().any(|a| *a != 1.0) { component.up_scales.len() } else { 0 }) as f64;
            }
            total += ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0) + (k + 1.0 + gauges) * 0.5 * size.ln();
        }
        Ok(total)
    }

    fn save(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(error)
    }

    /// The mixture of a checkpoint, refused unless it is a mixture of this explanation: the same
    /// bodies with the same layouts and groups, each target's choices and Adam's moments in its
    /// shape, and each component an earlier body (the order that makes the product of the targets'
    /// mixtures a joint density) whose alignment and up scales fit the two bodies' widths.
    fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let restored: BodyMixture = serde_json::from_value(value.clone()).map_err(error)?;
        let other = || "a checkpoint's body mixture of another explanation".to_string();
        if restored.layouts != self.layouts || restored.cells != self.cells || restored.targets.len() != self.targets.len() || restored.moments.len() != restored.targets.len() {
            return Err(other());
        }
        for (t, (found, own)) in restored.targets.iter().zip(&self.targets).enumerate() {
            if found.body != own.body || found.choices != own.choices || restored.moments[t].len() != 2 + MOMENTS_PER_COMPONENT * found.components.len() {
                return Err(other());
            }
            let (units, k, k_out) = self.layouts.get(&own.body).ok_or_else(other)?.widths();
            for component in &found.components {
                let earlier = self.targets[..t].iter().find(|e| e.body == component.body).ok_or_else(other)?;
                let (_, k_c, k_out_c) = self.layouts.get(&earlier.body).ok_or_else(other)?.widths();
                let a = &component.alignment;
                if a.units.len() != units || a.input.dim() != (k_c, k) || a.output.dim() != (k_out, k_out_c) || component.up_scales.len() != units || !a.matching_nats().is_finite() {
                    return Err(other());
                }
            }
        }
        *self = restored;
        Ok(())
    }
}

// ------------------------------------------------------------------------------------ reading

/// A token and its score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    pub token: u32,
    pub score: f64,
}

/// What one call of a body reads and writes in token terms ([`describe`]): per input coordinate
/// `q` the tokens whose embeddings it reads most and least, `r_q · (γ ⊙ e_t) / rms(e_t)` with the
/// layer's MLP norm gain `γ` (a direct read of the embedding), and per output coordinate the tokens
/// its write `w_q` promotes and suppresses, `W_U (γ_f ⊙ w_q)` centred over the vocabulary with the
/// final norm's gain `γ_f` (a direct path to the logits).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CallReading {
    pub call: String,
    pub layer: usize,
    pub body: String,
    pub reads: Vec<(Vec<Scored>, Vec<Scored>)>,
    pub writes: Vec<(Vec<Scored>, Vec<Scored>)>,
}

/// The gain and `ε` of the RMS norm whose gain node is `normed` in `native`.
fn norm_of(native: &OperatorProgram, normed: usize) -> Result<(Array1<f64>, f64), String> {
    let Node::Affine { terms, .. } = &native.nodes[normed] else { return Err(format!("node {normed} is not a normed stream")) };
    let [(rms, gain)] = terms[..] else { return Err(format!("node {normed} is not one gain of a norm")) };
    let Node::RmsNorm { epsilon, .. } = native.nodes[rms] else { return Err(format!("node {normed} does not read an RMS norm")) };
    Ok((native.operators[gain].matrix().diag().to_owned(), epsilon))
}

/// The `top` largest and smallest of `scores`, each with its token.
fn extremes(scores: &Array1<f64>, top: usize) -> (Vec<Scored>, Vec<Scored>) {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let pick = |i: &usize| Scored { token: *i as u32, score: scores[*i] };
    (order.iter().take(top).map(pick).collect(), order.iter().rev().take(top).map(pick).collect())
}

/// Each call of `explanation` in token terms ([`CallReading`]), `top` tokens per list; `native` is
/// the split native program and `layers` its sites (`run_check::layer_nodes`).
pub fn describe(native: &OperatorProgram, layers: &[crate::run_check::LayerNodes], explanation: &Explanation, calls: &[Call], top: usize) -> Result<Vec<CallReading>, String> {
    let head = crate::resident_causal_fit::fixed_head_target::Head::of(native)?;
    let unembedding = &head.embedding;
    let (final_gain, _) = norm_of(native, head.hidden)?;
    let feature = native.nodes.iter().position(|n| matches!(n, Node::Feature { .. })).ok_or("no token feature")?;
    let embedding = native
        .nodes
        .iter()
        .find_map(|n| match n {
            Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == feature => Some(native.operators[terms[0].1].matrix()),
            _ => None,
        })
        .ok_or("no token embedding")?;
    // Rows are tokens.
    let embedding = if embedding.ncols() == unembedding.ncols() { embedding } else { embedding.t().to_owned() };
    let program = &explanation.artifact.program;
    let mut out = Vec::with_capacity(calls.len());
    for call in calls {
        let site = layers.get(call.layer).ok_or_else(|| format!("{}: no layer {}", call.name, call.layer))?;
        let (gain, epsilon) = norm_of(native, site.normed)?;
        let normed = Array2::from_shape_fn(embedding.dim(), |(t, j)| {
            let row = embedding.row(t);
            gain[j] * row[j] / (row.dot(&row) / row.len() as f64 + epsilon).sqrt()
        });
        let read = program.operators[operator_index(program, &format!("{}.read", call.name))?].matrix();
        let write = program.operators[operator_index(program, &format!("{}.write", call.name))?].matrix();
        let reads = read.outer_iter().map(|r| extremes(&normed.dot(&r), top)).collect();
        let writes = write
            .columns()
            .into_iter()
            .map(|w| {
                let logits = unembedding.dot(&(&w * &final_gain));
                let mean = logits.mean().unwrap_or(0.0);
                extremes(&logits.mapv(|v| v - mean), top)
            })
            .collect();
        out.push(CallReading { call: call.name.clone(), layer: call.layer, body: call.body.clone(), reads, writes });
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------- editing

/// `explanation` with the native block `owner` stands for edited by `delta` (its shape), where the
/// block is a body's (`owner.body` a body, `owner.site` one of its calls), at that call alone: the
/// call applies a private copy of the body with one coordinate more, so every other call computes
/// as before. A gate row's edit `Δ` is read through a new input coordinate `Δ · x` that only the
/// edited unit's gate (or, for an up row, its up map) reads with weight 1; an output column's edit
/// is written through a new output coordinate that only the edited unit writes with weight 1; a
/// bias is edited in the copy; each over the block's scalar factors (a gated unit's up scale), which
/// act on the native block. The new coordinates' maps are constants of the edited explanation
/// (outside `Explanation::trainable`): an edit is a question asked of a fitted explanation.
pub fn edit_native(explanation: &Explanation, owner: &crate::artifact::Owner, delta: &Array2<f64>) -> Result<Explanation, String> {
    if delta.dim() != (owner.native_rows.len(), owner.native_cols.len()) {
        return Err(format!("an edit of {} is {:?}, not {:?}", owner.native, (owner.native_rows.len(), owner.native_cols.len()), delta.dim()));
    }
    let mut out = explanation.clone();
    let program = &mut out.artifact.program;
    // The block's scalar factors (a gated unit's up scale) act on the native block; the edit in the
    // body's terms is the native edit over them.
    let mut factor = 1.0;
    for name in owner.left.iter().chain(&owner.right) {
        let value = program.operators[operator_index(program, name)?].matrix();
        if value.dim() == (1, 1) {
            if value[[0, 0]] == 0.0 {
                return Err(format!("{}: the factor {name} is zero", owner.site));
            }
            factor *= value[[0, 0]];
        }
    }
    let delta = delta / factor;
    let ops = BodyOperators::of(program, &owner.body)?;
    let (read, write) = (operator_index(program, &format!("{}.read", owner.site))?, operator_index(program, &format!("{}.write", owner.site))?);
    let body_rule = rule_index(program, &owner.body)?;
    let site = sites(program, body_rule)?.into_iter().find(|s| s.read == read).ok_or_else(|| format!("{} does not call {}", owner.site, owner.body))?;
    let unit = match owner.role.as_str() {
        "out" => owner.cols.start,
        "gate" | "up" | "gate_bias" | "up_bias" => owner.rows.start,
        other => return Err(format!("{}: no body part {other}", owner.native)),
    };
    let (gate, out_map) = (program.operators[ops.gate].matrix(), program.operators[ops.out].matrix());
    let (m, k, k_out) = (gate.nrows(), gate.ncols(), out_map.nrows());
    let take = |op: Option<usize>| op.map(|o| program.operators[o].matrix());
    let (mut gate_bias, mut up, mut up_bias) = (take(ops.gate_bias), take(ops.up), take(ops.up_bias));
    // The copy's maps, one input coordinate (or output coordinate) wider where the edit needs it.
    let wider_in = matches!(owner.role.as_str(), "gate" | "up");
    let wider_out = owner.role == "out";
    let widen = |m: &Array2<f64>, hit: bool| -> Array2<f64> {
        let mut out = Array2::zeros((m.nrows(), m.ncols() + 1));
        out.slice_mut(s![.., ..m.ncols()]).assign(m);
        if hit {
            out[[unit, m.ncols()]] = 1.0;
        }
        out
    };
    let mut gate_values = gate.clone();
    let mut out_values = out_map.clone();
    let (mut read_values, mut write_values) = (program.operators[read].matrix(), program.operators[write].matrix());
    match owner.role.as_str() {
        "gate" | "up" => {
            gate_values = widen(&gate, owner.role == "gate");
            up = up.map(|u| widen(&u, owner.role == "up"));
            read_values = ndarray::concatenate(Axis(0), &[read_values.view(), delta.view()]).map_err(error)?;
        }
        "out" => {
            let mut wider = Array2::zeros((k_out + 1, m));
            wider.slice_mut(s![..k_out, ..]).assign(&out_map);
            wider[[k_out, unit]] = 1.0;
            out_values = wider;
            write_values = ndarray::concatenate(Axis(1), &[write_values.view(), delta.view()]).map_err(error)?;
        }
        "gate_bias" => gate_bias.as_mut().ok_or("no gate bias")?[[unit, 0]] += delta[[0, 0]],
        _ => up_bias.as_mut().ok_or("no up bias")?[[unit, 0]] += delta[[0, 0]],
    }
    let (z, h, y) = (units(k + usize::from(wider_in))?, units(m)?, units(k_out + usize::from(wider_out))?);
    let copy = format!("library.body{}", next_body(program));
    let provenance = Provenance::derived(&[], format!("edit of {} at {}", owner.native, owner.site));
    let base = program.operators.len();
    let mut added = vec![dense(format!("{copy}.gate"), h.clone(), z.clone(), gate_values, provenance.clone())?];
    let index = |added: &mut Vec<Operator>, op: Operator| {
        added.push(op);
        base + added.len() - 1
    };
    let gate_bias = gate_bias.map(|b| dense(format!("{copy}.gate_bias"), h.clone(), Interface::constant(), b, provenance.clone())).transpose()?.map(|op| index(&mut added, op));
    let up = up.map(|u| dense(format!("{copy}.up"), h.clone(), z.clone(), u, provenance.clone())).transpose()?.map(|op| index(&mut added, op));
    let up_bias = up_bias.map(|b| dense(format!("{copy}.up_bias"), h.clone(), Interface::constant(), b, provenance.clone())).transpose()?.map(|op| index(&mut added, op));
    let out_op = index(&mut added, dense(format!("{copy}.out"), y.clone(), h, out_values, provenance.clone())?);
    let read_op = index(&mut added, dense(format!("{}.read_edited", owner.site), z.clone(), program.operators[read].cols.clone(), read_values, provenance.clone())?);
    let write_op = index(&mut added, dense(format!("{}.write_edited", owner.site), program.operators[write].rows.clone(), y, write_values, provenance)?);
    program.operators.extend(added.into_iter().map(Arc::new));
    let law = match &program.rules[body_rule].nodes[2] {
        Node::Pointwise { laws, .. } => *laws.first().ok_or("an empty law list")?,
        other => return Err(format!("{}: node 2 is {other:?}", owner.body)),
    };
    let mut nodes = vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, base)], bias: gate_bias }, Node::Pointwise { input: 1, laws: vec![law; m] }];
    if let Some(up) = up {
        nodes.push(Node::Affine { terms: vec![(0, up)], bias: up_bias });
        nodes.push(Node::Hadamard { left: 2, right: 3 });
    }
    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, out_op)], bias: None });
    insert_first_rule(program, Rule { name: copy, inputs: vec![z], output: nodes.len() - 1, nodes });
    // The call reads, applies and writes through the copy; its rule moved down by one.
    let rule = &mut program.rules[site.rule + 1];
    let z_node = match &rule.nodes[site.node] {
        Node::Call { arguments, .. } => *arguments.first().ok_or("a call without an argument")?,
        other => return Err(format!("{}: node {} is {other:?}", rule.name, site.node)),
    };
    if let Node::Affine { terms, .. } = &mut rule.nodes[z_node] {
        terms[0].1 = read_op;
    }
    if let Node::Call { rule: called, .. } = &mut rule.nodes[site.node] {
        *called = 0;
    }
    let output = rule.output;
    if let Node::Affine { terms, .. } = &mut rule.nodes[output]
        && let Some(term) = terms.iter_mut().find(|t| t.0 == site.node)
    {
        term.1 = write_op;
    }
    program.interfaces().map_err(error)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        interchange::{self, Batch, Interchange},
        library_mdl::explanation,
        operator_program::{FamilyInputs, SlotValues},
        run_check::{LayerNodes, layer_nodes, split_sites},
    };
    use gam_gpu::tensor::Device;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// The tiny two-layer decoder (`d = 8`, sixteen MLP functions per layer) with MLP law `law`,
    /// gated like Qwen3 when `gated`: its split program, layers, inputs and sequences.
    fn tiny(tag: &str, law: &str, gated: bool) -> (OperatorProgram, Vec<LayerNodes>, FamilyInputs, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let path = dir.join("export.json");
        let mut record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        record["config"]["mlp_act"] = law.into();
        if gated {
            let mut rng = StdRng::seed_from_u64(17);
            for l in 0..2 {
                let name = format!("blocks.{l}.mlp.gate_proj");
                let bytes: Vec<u8> = (0..16 * 8).flat_map(|_| (rng.random::<f64>() - 0.5).to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}.f64")), bytes).unwrap();
                record["files"][name] = serde_json::json!({"shape": [16, 8]});
            }
            record["config"]["mlp_gated"] = true.into();
        }
        std::fs::write(&path, record.to_string()).unwrap();
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let family = imported.family;
        let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, family, sequences)
    }

    fn set(program: &mut OperatorProgram, name: &str, values: Array2<f64>) {
        let op = operator_index(program, name).unwrap();
        let source = Arc::clone(&program.operators[op]);
        program.operators[op] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, source.provenance.clone()).unwrap());
    }

    fn random(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 2.0 - 1.0)
    }

    fn outputs(explanation: &Explanation, family: &FamilyInputs) -> Array2<f64> {
        explanation.artifact.execute(family).unwrap().values[explanation.artifact.program.output].clone()
    }

    fn close(a: &Array2<f64>, b: &Array2<f64>, relative: f64) {
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let gap = a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(gap <= relative * scale, "outputs differ by {gap} at scale {scale}");
    }

    /// One subroutine of four units on a two-dimensional input and output, planted at layer 0 as
    /// functions `SITE0` (in the body's order) and at layer 1 as `SITE1`, in other bases; gated, with
    /// each unit's up row and output column at layer 1 scaled by a factor and its inverse.
    const SITE0: [usize; 4] = [1, 4, 6, 9];
    const SITE1: [usize; 4] = [14, 2, 11, 3];

    fn planted(explanation: &mut Explanation, gated: bool) {
        let mut rng = StdRng::seed_from_u64(23);
        let (g, b, u) = (random(&mut rng, 4, 2), random(&mut rng, 4, 2), random(&mut rng, 2, 4));
        let program = &mut explanation.artifact.program;
        // A gated law's units keep their function when the up row is scaled and the output column
        // divided by one factor; an ungated law's do not.
        let rescaled = if gated { [0.5, -2.0, 1.5, 3.0] } else { [1.0; 4] };
        for (l, site, scales) in [(0, SITE0, [1.0; 4]), (1, SITE1, rescaled)] {
            let (read, write) = (random(&mut rng, 2, 8), random(&mut rng, 8, 2));
            let name = format!("library.l{l}.mlp");
            let mut gate = program.operators[operator_index(program, &format!("{name}.gate")).unwrap()].matrix();
            let mut out = program.operators[operator_index(program, &format!("{name}.out")).unwrap()].matrix();
            let (gr, wu) = (g.dot(&read), write.dot(&u));
            for (j, &f) in site.iter().enumerate() {
                gate.row_mut(f).assign(&gr.row(j));
                out.column_mut(f).assign(&(&wu.column(j) / scales[j]));
            }
            set(program, &format!("{name}.gate"), gate);
            set(program, &format!("{name}.out"), out);
            if gated {
                let mut up = program.operators[operator_index(program, &format!("{name}.up")).unwrap()].matrix();
                let br = b.dot(&read);
                for (j, &f) in site.iter().enumerate() {
                    up.row_mut(f).assign(&(&br.row(j) * scales[j]));
                }
                set(program, &format!("{name}.up"), up);
            }
        }
    }

    #[test]
    fn a_rewrite_computes_what_the_region_computed_and_charges_the_body_once() {
        for (law, gated) in [("gelu_tanh", false), ("silu", true)] {
            let (native, layers, family, _) = tiny(&format!("bodies_rewrite_{gated}"), law, gated);
            let start = explanation(&native, &layers).unwrap();
            let (rewritten, call) = rewrite(&start, 0, &SITE0).unwrap();
            close(&outputs(&start, &family), &outputs(&rewritten, &family), 1e-12);
            assert_eq!(call.replaced, SITE0.iter().enumerate().map(|(j, i)| (*i, Some(j))).collect::<Vec<_>>());
            // Every entry in one group, the replaced functions' groups out, the body's groups in.
            let posterior = Posterior::new(&rewritten, 72).unwrap();
            let parts = if gated { 3 } else { 2 };
            assert_eq!(posterior.active.iter().filter(|a| !**a).count(), parts * SITE0.len());
            let body = rewritten.groups.iter().filter(|g| g.name.starts_with(&call.body)).count();
            assert_eq!(body, parts * SITE0.len());
            assert!(rewritten.groups.iter().any(|g| g.name == format!("{}.read0", call.name)));
            // A second region gets its own body and call.
            let (twice, second) = rewrite(&rewritten, 1, &[0, 5]).unwrap();
            assert_ne!(second.body, call.body);
            assert_ne!(second.name, call.name);
            close(&outputs(&start, &family), &outputs(&twice, &family), 1e-12);
            assert!(rewrite(&rewritten, 0, &[1, 2]).is_err(), "a replaced function is no longer native");
        }
    }

    #[test]
    fn a_subroutine_in_two_bases_is_found_as_one_body_and_merged_exactly() {
        for (law, gated) in [("gelu_tanh", false), ("silu", true)] {
            let (native, layers, family, _) = tiny(&format!("bodies_merge_{gated}"), law, gated);
            let mut start = explanation(&native, &layers).unwrap();
            planted(&mut start, gated);
            let (one, first) = rewrite(&start, 0, &SITE0).unwrap();
            let (two, second) = rewrite(&one, 1, &SITE1).unwrap();
            // The planted reads and writes have rank two: each body reads and writes two coordinates.
            let program = &two.artifact.program;
            assert_eq!(program.operators[operator_index(program, &format!("{}.read", second.name)).unwrap()].rows.width(), 2);
            let posterior = Posterior::new(&two, 72).unwrap();
            let (from, onto) = (body_values(&two, &posterior, &second.body).unwrap(), body_values(&two, &posterior, &first.body).unwrap());
            let alignment = align(&from, &onto).unwrap();
            // Unit j of the second body is the region's j-th function, planted as unit j.
            assert_eq!(alignment.units, (0..4).map(Some).collect::<Vec<_>>());
            assert!(alignment.misfit < 1e-12 * alignment.entries as f64, "misfit {}", alignment.misfit);
            let calls = vec![first.clone(), second.clone()];
            let (merged, calls) = merge(&two, &calls, &second.body, &first.body, &alignment).unwrap();
            close(&outputs(&start, &family), &outputs(&merged, &family), 1e-9);
            assert!(calls.iter().all(|c| c.body == first.body));
            // The merge no longer sends the second body's widths (two coordinates in and out: Elias δ
            // of 2 is 4 bits each); the calls' choice of body costs the same ln 2.
            let ln2 = std::f64::consts::LN_2;
            assert!((two.fixed_nats - one.fixed_nats - (crate::codec::subset_code_len_bits(16, 4).unwrap() + 8) as f64 * ln2 - 2_f64.ln()).abs() < 1e-12);
            // The merge charges the matching of the four units (`ln Σ_j C(4, j)² j! = ln 209`) and,
            // gated, the literals of the up scales the ownership map gains.
            assert!((alignment.matching_nats() - 209_f64.ln()).abs() < 1e-12);
            let scales: f64 = merged
                .artifact
                .program
                .operators
                .iter()
                .filter(|o| o.name.contains(".alpha"))
                .map(|o| o.code_bits().map(|(s, r)| (s + r) as f64 * ln2).unwrap())
                .sum();
            assert_eq!(scales > 0.0, gated);
            assert!((merged.fixed_nats - two.fixed_nats + 8.0 * ln2 - 209_f64.ln() - scales).abs() < 1e-9, "{} against {}", merged.fixed_nats, two.fixed_nats);
            let program = &merged.artifact.program;
            assert_eq!(program.rules.iter().filter(|r| r.name.starts_with("library.body")).count(), 1);
            assert_eq!(sites(program, rule_index(program, &first.body).unwrap()).unwrap().len(), 2, "one body, two calls");
            // The merged library pays for one body: the second's groups are gone.
            assert!(!merged.groups.iter().any(|g| g.name.starts_with(&second.body)));
            assert_eq!(merged.groups.len(), two.groups.len() - (if gated { 3 } else { 2 }) * 4);
            Posterior::new(&merged, 72).unwrap();
            // Every native block of both planted copies is stated exactly by the ownership map: the
            // shared body's block of its unit times its call's bindings (and up scales).
            let natives = |e: &Explanation| -> BTreeMap<(String, usize, usize), Array2<f64>> {
                e.artifact.owners.iter().map(|o| ((o.native.clone(), o.native_rows.start, o.native_cols.start), e.artifact.native_block(o).unwrap())).collect()
            };
            let (before, after) = (natives(&start), natives(&merged));
            assert_eq!(before.len(), after.len());
            for (key, value) in &before {
                let moved = &after[key];
                let scale = value.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
                assert!(value.iter().zip(moved).all(|(a, b)| (a - b).abs() <= 1e-9 * scale), "native block {key:?} is not stated exactly after the merge");
            }
            for (j, &f) in SITE1.iter().enumerate() {
                let gate = format!("{}.gate", first.body);
                assert!(merged.artifact.owners.iter().any(|o| o.site == second.name && o.operator == gate && o.native_rows == (f..f + 1) && o.rows == (j..j + 1)), "unit {j} of the second call owns native row {f}");
            }
            assert!(merged.artifact.owners.iter().all(|o| !o.operator.starts_with(&second.body) && o.body != second.body));
        }
    }

    #[test]
    fn a_body_called_twice_is_differentiated_through_both_calls() {
        let (native, layers, _, sequences) = tiny("bodies_gradient", "gelu_tanh", false);
        let mut start = explanation(&native, &layers).unwrap();
        planted(&mut start, false);
        let (one, first) = rewrite(&start, 0, &SITE0).unwrap();
        let (two, second) = rewrite(&one, 1, &SITE1).unwrap();
        let posterior = Posterior::new(&two, 72).unwrap();
        let alignment = align(&body_values(&two, &posterior, &second.body).unwrap(), &body_values(&two, &posterior, &first.body).unwrap()).unwrap();
        let (merged, _) = merge(&two, &[first.clone(), second], "library.body1", &first.body, &alignment).unwrap();
        let reads = interchange::library_reads(&merged.artifact.program, layers.len()).unwrap();
        let device = Device::host();
        let mut x = Interchange::new(&device, &native, &layers, &merged.artifact, &merged.trainable, reads.clone(), usize::MAX, 16).unwrap();
        let batch = Batch::new(sequences[..2].to_vec(), sequences[2..4].to_vec()).unwrap();
        let experiments = interchange::sample(&mut StdRng::seed_from_u64(5), 2, &reads, 4, 12).unwrap();
        let values: Vec<Array2<f64>> = merged.trainable.iter().map(|op| merged.artifact.program.operators[*op].matrix()).collect();
        let design = x.design_at(&reads, &experiments, &values).unwrap();
        x.load(&values).unwrap();
        let scored = x.evaluate(&batch, &experiments, &design, true).unwrap();
        let program = &merged.artifact.program;
        let at = |name: &str| merged.trainable.iter().position(|op| program.operators[*op].name == name).unwrap();
        let mut bits = |values: &[Array2<f64>]| -> f64 {
            x.load(values).unwrap();
            x.evaluate(&batch, &experiments, &design, false).unwrap().bits.iter().flatten().sum()
        };
        for (i, entry) in [(at(&format!("{}.gate", first.body)), (2, 1)), (at(&format!("{}.out", first.body)), (1, 3)), (at("library.l1.call1.read"), (0, 5))] {
            let h = 1e-5;
            let (mut up, mut down) = (values.clone(), values.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!(central.abs() > 0.0, "the entry moves the score");
            assert!((scored.gradient[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "gradient {} against {central}", scored.gradient[i][entry]);
        }
    }

    #[test]
    fn the_assignment_is_optimal() {
        let mut rng = StdRng::seed_from_u64(3);
        for n in 1..=6 {
            let cost = random(&mut rng, n, n);
            let found = hungarian(&cost).unwrap();
            let value = |p: &[usize]| (0..n).map(|i| cost[[i, p[i]]]).sum::<f64>();
            // Every permutation, by Heap's algorithm.
            let mut p: Vec<usize> = (0..n).collect();
            let mut best = value(&p);
            let mut c = vec![0; n];
            let mut i = 0;
            while i < n {
                if c[i] < i {
                    if i % 2 == 0 { p.swap(0, i) } else { p.swap(c[i], i) }
                    best = best.min(value(&p));
                    c[i] += 1;
                    i = 0;
                } else {
                    c[i] = 0;
                    i += 1;
                }
            }
            let mut sorted = found.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            assert!((value(&found) - best).abs() < 1e-12, "{} against the best {best}", value(&found));
        }
    }

    /// A body's values with every entry at deviation `sd` (fixed where `sd` is zero), its units'
    /// liveness from them.
    fn values(law: Law, gate: Array2<f64>, gate_bias: Option<Array2<f64>>, up: Option<Array2<f64>>, out: Array2<f64>, sd: f64) -> BodyValues {
        let part = |m: Array2<f64>| Part { sd: Array2::from_elem(m.dim(), sd), mean: m };
        let (m, k, k_out) = (gate.nrows(), gate.ncols(), out.nrows());
        let mut v = BodyValues { law, gate: part(gate), gate_bias: gate_bias.map(part), up: up.map(part), up_bias: None, out: part(out), units: Vec::new(), inputs: vec![true; k], outputs: vec![true; k_out] };
        v.units = (0..m).map(|j| live(&v, j, sd == 0.0)).collect();
        v
    }

    fn matrix(rows: &[&[f64]]) -> Array2<f64> {
        Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
    }

    #[test]
    fn a_unit_is_live_when_its_whole_function_is_not_identically_zero() {
        let out = matrix(&[&[1.0], &[0.0]]);
        // A zero reader with bias one under ReLU computes the constant one.
        let constant = values(Law::Relu, matrix(&[&[0.0, 0.0]]), Some(matrix(&[&[1.0]])), None, out.clone(), 0.0);
        assert!(live(&constant, 0, true));
        // With a bias where ReLU is zero it computes nothing.
        let dead = values(Law::Relu, matrix(&[&[0.0, 0.0]]), Some(matrix(&[&[-1.0]])), None, out.clone(), 0.0);
        assert!(!live(&dead, 0, true));
        // A negative mean pre-activation the posterior lets vary is active on part of its support.
        let uncertain = values(Law::Relu, matrix(&[&[0.0, 0.0]]), Some(matrix(&[&[-1.0]])), None, out.clone(), 1.0);
        assert!(live(&uncertain, 0, false));
        // A gated unit with a constant nonzero gate and a payload that reads is live; with a zero
        // payload it is not.
        let gated = |payload: f64| values(Law::Silu, matrix(&[&[0.0, 0.0]]), Some(matrix(&[&[2.0]])), Some(matrix(&[&[payload, 0.0]])), out.clone(), 0.0);
        assert!(live(&gated(1.0), 0, true));
        assert!(!live(&gated(0.0), 0, true));
    }

    #[test]
    fn bodies_of_different_laws_do_not_align() {
        let body = values(Law::GeluTanh, matrix(&[&[1.0, 0.5]]), None, None, matrix(&[&[1.0], &[-1.0]]), 1.0);
        let mut other = body.clone();
        other.law = Law::Silu;
        assert!(align(&body, &other).is_err(), "equal coefficients under another law are another function");
        assert!(align(&body, &body).is_ok());
    }

    #[test]
    fn a_gated_map_s_variances_come_from_its_factors() {
        // `u₁, u₂` independent with mean 1 and variance 0.01, `b` of mean 0 and variance 1:
        // `Var((u₁ − u₂) b) = 0.02`, where entries treated as independent give 2.02.
        let unit = Unit {
            gate: (Array1::zeros(1), Array1::ones(1)),
            bias: None,
            write: Write::Gated { out: (Array1::from_vec(vec![1.0, 1.0]), Array1::from_vec(vec![0.01, 0.01])), up: (Array1::zeros(1), Array1::ones(1)), bias: None },
        };
        let t = transformed(&unit, &Array2::eye(1), &matrix(&[&[1.0, -1.0]]));
        let ((mean, variance), _) = t.write.map().expect("a gated write has a map");
        assert_eq!(mean[[0, 0]], 0.0);
        assert!((variance[[0, 0]] - 0.02).abs() < 1e-15, "{}", variance[[0, 0]]);
    }

    #[test]
    fn a_value_of_variance_zero_is_an_equality_constraint() {
        let (design, y) = (matrix(&[&[1.0], &[1.0]]), matrix(&[&[0.0], &[2.0]]));
        assert!(precisions(&Array1::from_vec(vec![0.0, 2.0]))[0].is_infinite());
        let x = weighted_columns(&design, &y, &matrix(&[&[f64::INFINITY], &[1.0]])).unwrap();
        assert!(x[[0, 0]].abs() < 1e-15, "the exact value holds: {}", x[[0, 0]]);
        assert!(weighted_columns(&design, &y, &matrix(&[&[f64::INFINITY], &[f64::INFINITY]])).is_err(), "conflicting exact values");
        let free = weighted_columns(&design, &y, &matrix(&[&[1.0], &[1.0]])).unwrap();
        assert!((free[[0, 0]] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn an_unmatched_unit_of_either_body_is_charged() {
        let from = values(Law::GeluTanh, matrix(&[&[1.0, 0.0]]), None, None, matrix(&[&[1.0], &[0.0]]), 1.0);
        let onto = values(Law::GeluTanh, matrix(&[&[1.0, 0.0], &[0.0, 3.0]]), None, None, matrix(&[&[1.0, 0.0], &[0.0, 3.0]]), 1.0);
        let alignment = align(&from, &onto).unwrap();
        // The gauge fits either pair exactly; the matching leaves out the unit cheaper to leave out
        // (unit 0, its four values costing 1 + 1), and that unit's values are charged.
        assert_eq!(alignment.units, vec![Some(1)]);
        assert!((alignment.misfit - 2.0).abs() < 1e-9, "{}", alignment.misfit);
        assert_eq!(alignment.entries, 8);
        assert_eq!(alignment.live, [1, 2]);
        assert!(!alignment.constrains(), "a gauge of eight entries fits eight values");
        // `ln (1 + 1·2·1)`: no pair, or one of the two units of `onto`.
        assert!((alignment.matching_nats() - 3_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn signatures_compare_reads_with_reads_and_writes_with_writes() {
        let a: Signature = (Array1::from_vec(vec![1.0, 0.5, 0.2]), Array1::from_vec(vec![0.9, 0.1, 0.0]));
        let b: Signature = (Array1::from_vec(vec![1.0, 0.5]), Array1::from_vec(vec![0.9, 0.1]));
        assert_eq!(signature_distance(&a, &b), 0.0);
    }

    #[test]
    fn a_rewrite_records_the_parts_it_discards() {
        let (native, layers, _, _) = tiny("bodies_discarded", "gelu_tanh", false);
        let mut start = explanation(&native, &layers).unwrap();
        // The region's fourth gate row is the sum of its first two: its read has rank three.
        let program = &mut start.artifact.program;
        let mut gate = program.operators[operator_index(program, "library.l0.mlp.gate").unwrap()].matrix();
        let sum = &gate.row(SITE0[0]) + &gate.row(SITE0[1]);
        gate.row_mut(SITE0[3]).assign(&sum);
        set(program, "library.l0.mlp.gate", gate);
        let (rewritten, call) = rewrite(&start, 0, &SITE0).unwrap();
        let get = |p: &OperatorProgram, name: &str| p.operators[operator_index(p, name).unwrap()].matrix();
        let (before, after) = (&start.artifact.program, &rewritten.artifact.program);
        let a = get(before, "library.l0.mlp.gate").select(Axis(0), &SITE0);
        let (g, r) = (get(after, &format!("{}.gate", call.body)), get(after, &format!("{}.read", call.name)));
        assert_eq!(r.nrows(), 3, "the body reads the three resolved directions");
        let spectral = |m: &Array2<f64>| svd(m.view(), false).unwrap().singular_values.first().copied().unwrap_or(0.0);
        let dropped = spectral(&(&a - &g.dot(&r)));
        assert!((dropped - call.discarded[0]).abs() <= 1e-12 * spectral(&a), "{dropped} against {}", call.discarded[0]);
        assert!(call.discarded[0] <= 1e-12 * spectral(&a));
    }

    #[test]
    fn a_bias_takes_no_binding_at_a_rank_one_read() {
        let (native, layers, _, _) = tiny("bodies_bias", "gelu_tanh", false);
        let mut start = explanation(&native, &layers).unwrap();
        // Layer 0's MLP with a gate bias, each function's entry in its gate group and owned as the
        // native bias of its unit.
        let mut rng = StdRng::seed_from_u64(41);
        let program = &mut start.artifact.program;
        let gate = operator_index(program, "library.l0.mlp.gate").unwrap();
        let units_interface = program.operators[gate].rows.clone();
        let bias = program.operators.len();
        program.operators.push(Arc::new(dense("library.l0.mlp.gate_bias".into(), units_interface, Interface::constant(), random(&mut rng, 16, 1), Provenance::default()).unwrap()));
        let rule = rule_index(program, "library.l0.mlp").unwrap();
        for node in &mut program.rules[rule].nodes {
            if let Node::Affine { terms, bias: b } = node
                && terms[0].1 == gate
            {
                *b = Some(bias);
            }
        }
        // Two functions reading one direction: the body reads one coordinate, as wide as a bias.
        let mut values = program.operators[gate].matrix();
        let doubled = &values.row(1) * 2.0;
        values.row_mut(4).assign(&doubled);
        set(program, "library.l0.mlp.gate", values);
        program.interfaces().unwrap();
        for (i, function) in start.layers[0].functions.iter().enumerate() {
            start.groups[function[0]].cells.push(Cells { operator: bias, rows: vec![i], cols: 0..1 });
            start.artifact.owners.push(crate::artifact::Owner {
                operator: "library.l0.mlp.gate_bias".into(),
                rows: i..i + 1,
                cols: 0..1,
                body: "library.l0.mlp".into(),
                site: "library.l0.mlp".into(),
                native: "blocks.0.c_fc.bias".into(),
                native_rows: i..i + 1,
                native_cols: 0..1,
                role: "gate_bias".into(),
                ..Default::default()
            });
        }
        start.trainable.push(bias);
        start.trainable.sort_unstable();
        let (rewritten, call) = rewrite(&start, 0, &[1, 4]).unwrap();
        let program = &rewritten.artifact.program;
        assert_eq!(program.operators[operator_index(program, &format!("{}.read", call.name)).unwrap()].rows.width(), 1);
        let natives = |e: &Explanation| -> BTreeMap<(String, usize, usize), Array2<f64>> {
            e.artifact.owners.iter().map(|o| ((o.native.clone(), o.native_rows.start, o.native_cols.start), e.artifact.native_block(o).unwrap())).collect()
        };
        let (before, after) = (natives(&start), natives(&rewritten));
        for (key, value) in &before {
            let moved = &after[key];
            assert_eq!(value.dim(), moved.dim(), "native block {key:?}");
            let scale = value.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            assert!(value.iter().zip(moved).all(|(a, b)| (a - b).abs() <= 1e-9 * scale), "native block {key:?}");
        }
    }

    #[test]
    fn a_unit_merged_twice_keeps_one_composed_up_scale() {
        let (native, layers, family, _) = tiny("bodies_twice", "silu", true);
        let mut start = explanation(&native, &layers).unwrap();
        planted(&mut start, true);
        // A third copy of the subroutine at layer 0, in another basis and with other up scales.
        let third = [0, 3, 5, 7];
        {
            let mut rng = StdRng::seed_from_u64(23);
            let (g, b, u) = (random(&mut rng, 4, 2), random(&mut rng, 4, 2), random(&mut rng, 2, 4));
            let mut other = StdRng::seed_from_u64(29);
            let (read, write) = (random(&mut other, 2, 8), random(&mut other, 8, 2));
            let scales = [2.0, 0.5, -1.0, 1.5];
            let program = &mut start.artifact.program;
            let get = |p: &OperatorProgram, name: &str| p.operators[operator_index(p, name).unwrap()].matrix();
            let (mut gate, mut up, mut out) = (get(program, "library.l0.mlp.gate"), get(program, "library.l0.mlp.up"), get(program, "library.l0.mlp.out"));
            let (gr, br, wu) = (g.dot(&read), b.dot(&read), write.dot(&u));
            for (j, &f) in third.iter().enumerate() {
                gate.row_mut(f).assign(&gr.row(j));
                up.row_mut(f).assign(&(&br.row(j) * scales[j]));
                out.column_mut(f).assign(&(&wu.column(j) / scales[j]));
            }
            set(program, "library.l0.mlp.gate", gate);
            set(program, "library.l0.mlp.up", up);
            set(program, "library.l0.mlp.out", out);
        }
        let (one, first) = rewrite(&start, 0, &SITE0).unwrap();
        let (two, second) = rewrite(&one, 1, &SITE1).unwrap();
        let (three, last) = rewrite(&two, 0, &third).unwrap();
        let posterior = Posterior::new(&three, 72).unwrap();
        let onto_first = align(&body_values(&three, &posterior, &second.body).unwrap(), &body_values(&three, &posterior, &first.body).unwrap()).unwrap();
        let (merged, calls) = merge(&three, &[first.clone(), second.clone(), last.clone()], &second.body, &first.body, &onto_first).unwrap();
        // The second call's units now carry up scales; merging their body again composes them.
        let posterior = Posterior::new(&merged, 72).unwrap();
        let onto_last = align(&body_values(&merged, &posterior, &first.body).unwrap(), &body_values(&merged, &posterior, &last.body).unwrap()).unwrap();
        let (twice, _) = merge(&merged, &calls, &first.body, &last.body, &onto_last).unwrap();
        close(&outputs(&start, &family), &outputs(&twice, &family), 1e-9);
        let names: Vec<&str> = twice.artifact.program.operators.iter().map(|o| o.name.as_str()).collect();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "every operator name is unique");
        let natives = |e: &Explanation| -> BTreeMap<(String, usize, usize), Array2<f64>> {
            e.artifact.owners.iter().map(|o| ((o.native.clone(), o.native_rows.start, o.native_cols.start), e.artifact.native_block(o).unwrap())).collect()
        };
        let (before, after) = (natives(&start), natives(&twice));
        for (key, value) in &before {
            let moved = &after[key];
            let scale = value.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            assert!(value.iter().zip(moved).all(|(a, b)| (a - b).abs() <= 1e-9 * scale), "native block {key:?} after two merges");
        }
        // Each owner of the second call's units holds one up scale.
        for owner in twice.artifact.owners.iter().filter(|o| o.site == second.name) {
            assert!(owner.left.iter().chain(&owner.right).filter(|n| n.contains("alpha")).count() <= 1, "{owner:?}");
        }
    }

    #[test]
    fn a_restored_mixture_must_be_of_this_explanation() {
        use crate::library_mdl::PriorTerm;
        let (native, layers, _, _) = tiny("bodies_restore", "gelu_tanh", false);
        let mut start = explanation(&native, &layers).unwrap();
        planted(&mut start, false);
        let (one, first) = rewrite(&start, 0, &SITE0).unwrap();
        let (two, second) = rewrite(&one, 1, &SITE1).unwrap();
        let posterior = Posterior::new(&two, 72).unwrap();
        let steps = crate::library_mixture::Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
        let mut mixture = BodyMixture::new(&two, steps).unwrap();
        mixture.epoch(&two, &posterior).unwrap();
        assert_eq!(mixture.targets[1].components.len(), 1);
        let saved = mixture.save().unwrap();
        let mut fresh = BodyMixture::new(&two, steps).unwrap();
        fresh.load(&saved).unwrap();
        // A component on the target's own body breaks the order that makes the prior a density.
        let mut later = saved.clone();
        later["targets"][1]["components"][0]["body"] = serde_json::json!(second.body);
        assert!(fresh.load(&later).is_err());
        // A layout of other groups.
        let mut moved = saved.clone();
        let group = moved["layouts"][first.body.as_str()]["entries"][0]["group"].as_u64().unwrap();
        moved["layouts"][first.body.as_str()]["entries"][0]["group"] = serde_json::json!(group + 1);
        assert!(fresh.load(&moved).is_err());
    }

    #[test]
    fn the_parallel_functions_of_a_planted_subroutine_are_its_regions() {
        for (law, gated) in [("gelu_tanh", false), ("silu", true)] {
            let (native, layers, _, _) = tiny(&format!("bodies_regions_{gated}"), law, gated);
            let mut start = explanation(&native, &layers).unwrap();
            planted(&mut start, gated);
            // At a coarse and at a fine posterior resolution (at the fine one no two units of the
            // subroutine save anything on their own).
            for tokens in [72, 1_000_000] {
                let posterior = Posterior::new(&start, tokens).unwrap();
                let pool: Vec<usize> = (0..16).collect();
                for (l, site) in [(0, SITE0), (1, SITE1)] {
                    let mut expected = site.to_vec();
                    expected.sort_unstable();
                    let found = regions(&start, &posterior, l, &pool).unwrap();
                    // The planted copy first, the only candidate the heuristic expects to save (a
                    // candidate overlapping it is not listed).
                    assert_eq!(found[0].0, expected, "layer {l} at {tokens} tokens");
                    assert!(found[0].1 > 0.0 && found[1..].iter().all(|(_, saved)| *saved <= 0.0), "layer {l} at {tokens} tokens: {found:?}");
                }
            }
        }
    }

    #[test]
    fn the_body_mixture_is_differentiated_exactly_and_hardens_into_a_merge() {
        use crate::library_mdl::PriorTerm;
        for (law, gated) in [("gelu_tanh", false), ("silu", true)] {
            let (native, layers, family, _) = tiny(&format!("bodies_mixture_{gated}"), law, gated);
            let mut start = explanation(&native, &layers).unwrap();
            planted(&mut start, gated);
            let (one, first) = rewrite(&start, 0, &SITE0).unwrap();
            let (two, second) = rewrite(&one, 1, &SITE1).unwrap();
            let posterior = Posterior::new(&two, 72).unwrap();
            let steps = crate::library_mixture::Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
            let mut mixture = BodyMixture::new(&two, steps).unwrap();
            mixture.epoch(&two, &posterior).unwrap();
            assert_eq!(mixture.targets.len(), 2);
            assert!(mixture.targets[0].components.is_empty(), "the first body has no earlier body");
            assert_eq!(mixture.targets[1].components.len(), 1);
            assert_eq!(mixture.targets[1].components[0].body, first.body);
            // A weight sample: the means perturbed, so neither body equals the other's image.
            let mut rng = StdRng::seed_from_u64(9);
            let theta: BTreeMap<usize, Array2<f64>> =
                mixture.operators().into_iter().map(|i| (i, posterior.mean[i].mapv(|m| m + 0.01 * (rng.random::<f64>() - 0.5)))).collect();
            let (_, gradient) = mixture.sample(&posterior, &theta, false).unwrap();
            let program = &two.artifact.program;
            let at = |name: String| two.trainable.iter().position(|op| program.operators[*op].name == name).unwrap();
            for (i, entry) in [(at(format!("{}.gate", second.body)), (2, 1)), (at(format!("{}.out", second.body)), (1, 3)), (at(format!("{}.gate", first.body)), (0, 0)), (at(format!("{}.out", first.body)), (0, 2))] {
                let h = 1e-6;
                let (mut up, mut down) = (theta.clone(), theta.clone());
                up.get_mut(&i).unwrap()[entry] += h;
                down.get_mut(&i).unwrap()[entry] -= h;
                let central = (mixture.sample(&posterior, &up, false).unwrap().0 - mixture.sample(&posterior, &down, false).unwrap().0) / (2.0 * h);
                let analytic = gradient.get(&i).map_or(0.0, |g| g[entry]);
                assert!((analytic - central).abs() <= 1e-5 * (1.0 + central.abs()), "entry {entry:?} of operator {i}: {analytic} against {central}");
            }
            // At the means the component's centre is the target itself (the gauge and, gated, the
            // non-unit up scales of the planted copy relate them exactly): the relation the mixture
            // fits is the one the merge compiles.
            let means: BTreeMap<usize, Array2<f64>> = mixture.operators().into_iter().map(|i| (i, posterior.mean[i].clone())).collect();
            let entries = mixture.live(1, &posterior).unwrap();
            let centre = mixture.predicted(&mixture.targets[1].components[0], &entries, &means).unwrap();
            let target = Array1::from_iter(entries.iter().map(|e| means[&e.operator][[e.row, e.col]]));
            let scale = target.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            assert!(centre.iter().zip(&target).all(|(a, b)| (a - b).abs() <= 1e-9 * scale), "the component's centre is the target");
            // The planted copy dominates once its weight is learned; hardened, it is one body.
            mixture.targets[1].components[0].logit = 5.0;
            assert_eq!(mixture.dominant(&posterior).unwrap(), vec![(1, 0)]);
            let (merged, calls, pairs) = mixture.harden(&two, &[first.clone(), second.clone()], &posterior).unwrap();
            assert_eq!(pairs, vec![(second.body.clone(), first.body.clone())]);
            assert!(calls.iter().all(|c| c.body == first.body));
            close(&outputs(&start, &family), &outputs(&merged, &family), 1e-9);
            assert!(mixture.cost(&posterior).unwrap() > 0.0, "the mixture pays for its choice, weights and gauge");
        }
    }

    #[test]
    fn a_call_is_read_in_token_terms() {
        let (native, layers, _, _) = tiny("bodies_describe", "gelu_tanh", false);
        let start = explanation(&native, &layers).unwrap();
        let (one, call) = rewrite(&start, 0, &SITE0).unwrap();
        let readings = describe(&native, &layers, &one, std::slice::from_ref(&call), 3).unwrap();
        assert_eq!(readings.len(), 1);
        let program = &one.artifact.program;
        let k = program.operators[operator_index(program, &format!("{}.read", call.name)).unwrap()].rows.width();
        assert_eq!(readings[0].reads.len(), k);
        for (most, least) in readings[0].reads.iter().chain(&readings[0].writes) {
            assert_eq!((most.len(), least.len()), (3, 3));
            assert!(most.windows(2).all(|w| w[0].score >= w[1].score) && most[0].score >= least[0].score);
        }
    }

    #[test]
    fn a_native_edit_inside_a_shared_body_changes_its_own_call_alone() {
        for (law, gated) in [("gelu_tanh", false), ("silu", true)] {
            let (native, layers, family, _) = tiny(&format!("bodies_edit_{gated}"), law, gated);
            let mut start = explanation(&native, &layers).unwrap();
            planted(&mut start, gated);
            let (one, first) = rewrite(&start, 0, &SITE0).unwrap();
            let (two, second) = rewrite(&one, 1, &SITE1).unwrap();
            let posterior = Posterior::new(&two, 72).unwrap();
            let alignment = align(&body_values(&two, &posterior, &second.body).unwrap(), &body_values(&two, &posterior, &first.body).unwrap()).unwrap();
            let (merged, _) = merge(&two, &[first.clone(), second.clone()], &second.body, &first.body, &alignment).unwrap();
            let mut rng = StdRng::seed_from_u64(31);
            let mut roles = vec!["gate", "out"];
            if gated {
                roles.push("up");
            }
            for role in roles {
                // The native block of the planted unit at the second call (layer 1, function SITE1[2]).
                let f = SITE1[2];
                let owner = merged.artifact.owners.iter().find(|o| o.site == second.name && o.role == role && (if role == "out" { o.native_cols == (f..f + 1) } else { o.native_rows == (f..f + 1) })).unwrap().clone();
                let delta = random(&mut rng, owner.native_rows.len(), owner.native_cols.len());
                let edited = edit_native(&merged, &owner, &delta).unwrap();
                // The reference: the native library with the same block of layer 1's MLP edited.
                let mut reference = start.clone();
                let name = format!("library.l1.mlp.{role}");
                let mut values = reference.artifact.program.operators[operator_index(&reference.artifact.program, &name).unwrap()].matrix();
                values.slice_mut(s![owner.native_rows.clone(), owner.native_cols.clone()]).scaled_add(1.0, &delta);
                set(&mut reference.artifact.program, &name, values);
                close(&outputs(&reference, &family), &outputs(&edited, &family), 1e-9);
                assert!(outputs(&reference, &family).iter().zip(outputs(&start, &family).iter()).any(|(a, b)| (a - b).abs() > 1e-6), "the edit moves the outputs");
            }
        }
    }
}


