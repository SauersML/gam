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
//! new body, exactly. With `A` the region's gate rows (and `D` its up rows), `S = [A; D]` and
//! `S = Σ_r s_r a_r v_rᵀ` its singular value decomposition over the singular values resolved from
//! zero (above the decomposition's rounding band), `R = [v_r]ᵀ`, `G = A Rᵀ` and `B = D Rᵀ`; with
//! `O` the region's output columns and `O = Σ_r t_r p_r q_rᵀ` likewise, `W = [p_r]` and
//! `U = Wᵀ O`. Then `G R = A`, `B R = D` and `W U = O`, so the rewritten explanation computes what
//! the region computed. Unit `j` of the body is the region's `j`-th function: each replaced native
//! block's owner (`Artifact::owners`) becomes the body's block of that unit at the call with the
//! call's binding as its factor, exactly (`a_i = g_j R`, `b_i = b_j R`, `u_i = W u_j`;
//! [`Call::replaced`] lists the same correspondence). The native functions' groups leave the
//! explanation (`Explanation::removed`); their reads stay among `M`'s read variables, so the
//! experiments do not change.
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
//! `N_i = C N_{π(i)}`). [`align`] finds them by alternating weighted least squares in `A` and `C`
//! with an optimal assignment of the units ([`hungarian`]), each step minimizing over its own
//! block the misfit `Σ (θ₁ − T(θ₀))² / σ₁²` (the first body's posterior means `θ₁` against the
//! transformed second's, in units of the first's posterior deviations), so the misfit never
//! increases; it reports the misfit in units of both posteriors' deviations. [`merge`] makes every
//! call of the first body a call of the second with the bindings `A R` and `W C`: the element
//! relating the call to the shared body is part of the call's bindings, which are priced. A rewrite
//! or a merge is accepted only if `F` falls after the fit re-converges on the fixed native
//! experiments.
//!
//! # Regions
//!
//! The functions of a body are parallel: they read the same few directions and write the same few.
//! They need not interact with each other, so they are not a community of the flow graph (the
//! functions an MLP's units interact with are much the same for all of them); they are the
//! functions whose union a rewrite compresses. Among an MLP's functions that carry RelP flow
//! (`library_readout`), [`regions`] groups them greedily by the parameters a rewrite of the union
//! saves at the posterior's own resolution.

use crate::{
    library_mdl::{Cells, Explanation, Group, Posterior},
    operator_program::{Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Rule, exact_precision, remap_node},
};
use gam_linalg::decompose::svd;
use ndarray::{Array1, Array2, Axis, s};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

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
/// `{name}.write`), its layer, its body's rule name, and per native function of the layer it
/// replaced the body unit computing it at this call (the native parameters' ownership; a function
/// whose unit a merge did not match keeps no unit).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Call {
    pub name: String,
    pub layer: usize,
    pub body: String,
    pub replaced: Vec<(usize, Option<usize>)>,
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

/// The singular vectors of `a` whose singular values are resolved from zero (above the
/// decomposition's rounding band): `(left, values, right)`; none when every one is within it.
fn resolved(a: &Array2<f64>) -> Result<Option<(Array2<f64>, Array1<f64>, Array2<f64>)>, String> {
    if a.is_empty() {
        return Ok(None);
    }
    let decomposition = svd(a.view(), false).map_err(error)?;
    let rank = decomposition.singular_values.iter().filter(|s| **s > decomposition.band).count();
    if rank == 0 {
        return Ok(None);
    }
    Ok(Some((
        decomposition.u.slice(s![.., ..rank]).to_owned(),
        decomposition.singular_values.slice(s![..rank]).to_owned(),
        decomposition.vt.slice(s![..rank, ..]).to_owned(),
    )))
}

/// `explanation` with the functions `functions` of layer `layer`'s MLP replaced by a call of a new
/// body, exactly (module note), and the call.
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
    let (_, _, read) = resolved(&stacked)?.ok_or("a region whose reads are zero")?;
    let g = a.dot(&read.t());
    let b = d_up.map(|d| d.dot(&read.t()));
    let o = program.operators[out].matrix().select(Axis(1), functions);
    let (write, _, _) = resolved(&o)?.ok_or("a region whose writes are zero")?;
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
    // The body's groups once, the call's per row of `R` and column of `W`.
    let groups = &mut rewritten.groups;
    for j in 0..n {
        let mut gate_cells = vec![Cells { operator: body_gate, rows: vec![j], cols: 0..k }];
        gate_cells.extend(body_gate_bias.map(|op| Cells { operator: op, rows: vec![j], cols: 0..1 }));
        groups.push(Group { name: format!("{body}.u{j}.gate"), cells: gate_cells });
        if let Some(up) = body_up {
            let mut up_cells = vec![Cells { operator: up, rows: vec![j], cols: 0..k }];
            up_cells.extend(body_up_bias.map(|op| Cells { operator: op, rows: vec![j], cols: 0..1 }));
            groups.push(Group { name: format!("{body}.u{j}.up"), cells: up_cells });
        }
        groups.push(Group { name: format!("{body}.u{j}.out"), cells: vec![Cells { operator: body_out, rows: (0..k_out).collect(), cols: j..j + 1 }] });
    }
    groups.extend(binding_groups(&call, read_op, write_op, k, k_out, d_in, d_out));
    // Each replaced native block's owner is now the body's block of its unit at this call, read
    // through the call's bindings (a gate row through `R`, an output column through `W`).
    let row_block = |operator: &str, i: usize| -> Option<(usize, std::ops::Range<usize>, std::ops::Range<usize>)> {
        let j = functions.iter().position(|f| *f == i)?;
        let parts = [("gate", Some(body_gate), k), ("up", body_up, k), ("gate_bias", body_gate_bias, 1), ("up_bias", body_up_bias, 1)];
        let (_, op, width) = parts.into_iter().find(|(part, _, _)| operator == format!("{mlp}.{part}"))?;
        Some((op?, j..j + 1, 0..width))
    };
    let (read_name, write_name) = (format!("{call}.read"), format!("{call}.write"));
    for owner in &mut rewritten.artifact.owners {
        // `a_i = g_j R`, `b_i = b_j R` and `u_i = W u_j` (biases are the body's own entries).
        let target = if owner.operator == format!("{mlp}.out") && owner.cols.len() == 1 {
            functions.iter().position(|f| *f == owner.cols.start).map(|j| (body_out, 0..k_out, j..j + 1, vec![write_name.clone()], Vec::new()))
        } else if owner.rows.len() == 1 {
            row_block(&owner.operator, owner.rows.start).map(|(op, rows, cols)| {
                let right = if cols.len() == k { vec![read_name.clone()] } else { Vec::new() };
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
    Ok((rewritten, Call { name: call, layer, body, replaced }))
}

/// A call's binding groups: each row of its read binding (`k × d_in`) and each column of its write
/// binding (`d_out × k′`).
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

/// One body's posterior: per unit its gate row (`m × k`), gate bias, up row, up bias, and its
/// output column (`k′ × m`); which units are in the explanation, and which input and output
/// coordinates some call of the body reads or writes.
#[derive(Clone, Debug)]
pub struct BodyValues {
    pub gate: Part,
    pub gate_bias: Option<Part>,
    pub up: Option<Part>,
    pub up_bias: Option<Part>,
    pub out: Part,
    pub units: Vec<bool>,
    pub inputs: Vec<bool>,
    pub outputs: Vec<bool>,
}

/// `body`'s posterior (removed groups are zero with zero deviation).
pub fn body_values(explanation: &Explanation, posterior: &Posterior, body: &str) -> Result<BodyValues, String> {
    let program = &explanation.artifact.program;
    let ops = BodyOperators::of(program, body)?;
    let means = posterior.means();
    let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("{body}: operator {op} is not trainable"));
    let take = |op: usize| -> Result<Part, String> {
        let i = position(op)?;
        Ok(Part { mean: means[i].clone(), sd: posterior.log_sd[i].mapv(f64::exp) })
    };
    let (gate, out) = (take(ops.gate)?, take(ops.out)?);
    let (k, k_out) = (gate.mean.ncols(), out.mean.nrows());
    let alive = |row: ndarray::ArrayView1<'_, f64>| row.iter().any(|v| *v != 0.0);
    let units = (0..gate.mean.nrows()).map(|j| alive(gate.mean.row(j)) && alive(out.mean.column(j))).collect();
    let (mut inputs, mut outputs) = (vec![false; k], vec![false; k_out]);
    for site in sites(program, rule_index(program, body)?)? {
        let (read, write) = (&means[position(site.read)?], &means[position(site.write)?]);
        inputs.iter_mut().enumerate().for_each(|(q, used)| *used |= alive(read.row(q)));
        outputs.iter_mut().enumerate().for_each(|(q, used)| *used |= alive(write.column(q)));
    }
    Ok(BodyValues {
        gate,
        gate_bias: ops.gate_bias.map(take).transpose()?,
        up: ops.up.map(take).transpose()?,
        up_bias: ops.up_bias.map(take).transpose()?,
        out,
        units,
        inputs,
        outputs,
    })
}

/// `body`'s values in `program` (its operators' values, unit deviations): what an alignment of
/// fitted means needs where no posterior is at hand.
fn artifact_values(program: &OperatorProgram, body: &str) -> Result<BodyValues, String> {
    let ops = BodyOperators::of(program, body)?;
    let part = |op: usize| {
        let mean = program.operators[op].matrix();
        Part { sd: Array2::ones(mean.dim()), mean }
    };
    let (gate, out) = (part(ops.gate), part(ops.out));
    let alive = |row: ndarray::ArrayView1<'_, f64>| row.iter().any(|v| *v != 0.0);
    let units = (0..gate.mean.nrows()).map(|j| alive(gate.mean.row(j)) && alive(out.mean.column(j))).collect();
    let (inputs, outputs) = (vec![true; gate.mean.ncols()], vec![true; out.mean.nrows()]);
    Ok(BodyValues { gate_bias: ops.gate_bias.map(part), up: ops.up.map(part), up_bias: ops.up_bias.map(part), gate, out, units, inputs, outputs })
}

/// The gauge relating body `from` to body `onto` (module note): unit `i` of `from` is unit
/// `units[i]` of `onto` (none when unmatched); `input` is `A` (`k_onto × k_from`) and `output` is
/// `C` (`k′_from × k′_onto`), zero outside the coordinates the calls use; `misfit` is the squared
/// misfit in units of both posteriors' deviations over `entries` compared values, of which the
/// gauge's `gauge` entries were fitted: when the two bodies are one function the misfit's
/// expectation is `entries − gauge`, and with `entries ≤ gauge` the alignment holds no evidence
/// that they are ([`Alignment::evidence`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    pub units: Vec<Option<usize>>,
    pub input: Array2<f64>,
    pub output: Array2<f64>,
    pub misfit: f64,
    pub entries: usize,
    pub gauge: usize,
}

impl Alignment {
    /// The misfit per value the gauge did not fit (the reduced χ², near 1 when the two bodies are
    /// one function), or none when the gauge can fit every compared value.
    #[must_use]
    pub fn evidence(&self) -> Option<f64> {
        (self.entries > self.gauge).then(|| self.misfit / (self.entries - self.gauge) as f64)
    }
}

/// One unit of a body restricted to the used coordinates: its gate row and bias, and its write
/// (the output column, or for a gated law `M = u bᵀ` with `N = e u`), each with its variances.
#[derive(Clone, Debug)]
struct Unit {
    gate: (Array1<f64>, Array1<f64>),
    bias: Option<(f64, f64)>,
    write: Write,
}

#[derive(Clone, Debug)]
enum Write {
    Plain((Array1<f64>, Array1<f64>)),
    Gated { map: (Array2<f64>, Array2<f64>), offset: Option<(Array1<f64>, Array1<f64>)> },
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
            let u = (values.out.mean.column(j).select(Axis(0), &outs), squared(&values.out.sd.column(j).select(Axis(0), &outs)));
            let write = match &values.up {
                // `Var(u b) = u² σ_b² + b² σ_u² + σ_u² σ_b²` for independent factors.
                Some(up) => {
                    let (b, vb) = row(up);
                    let map = (outer(&u.0, &b), outer(&u.0.mapv(|v| v * v), &vb) + outer(&u.1, &b.mapv(|v| v * v)) + outer(&u.1, &vb));
                    let offset = values.up_bias.as_ref().map(|e| {
                        let (e, ve) = (e.mean[[j, 0]], e.sd[[j, 0]].powi(2));
                        (&u.0 * e, u.0.mapv(|v| v * v) * ve + &u.1 * (e * e) + &u.1 * ve)
                    });
                    Write::Gated { map, offset }
                }
                None => Write::Plain(u),
            };
            let bias = values.gate_bias.as_ref().map(|c| (c.mean[[j, 0]], c.sd[[j, 0]].powi(2)));
            (j, Unit { gate: row(&values.gate), bias, write })
        })
        .collect()
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.len(), b.len()), |(i, j)| a[i] * b[j])
}

/// `onto`'s unit in `from`'s coordinates under the gauge: its values and their variances carried
/// through `A` and `C` (independent entries).
fn transformed(unit: &Unit, a: &Array2<f64>, c: &Array2<f64>) -> Unit {
    let (a2, c2) = (a.mapv(|v| v * v), c.mapv(|v| v * v));
    let write = match &unit.write {
        Write::Plain((u, v)) => Write::Plain((c.dot(u), c2.dot(v))),
        Write::Gated { map: (m, v), offset } => Write::Gated {
            map: (c.dot(m).dot(a), c2.dot(v).dot(&a2)),
            offset: offset.as_ref().map(|(n, v)| (c.dot(n), c2.dot(v))),
        },
    };
    Unit { gate: (unit.gate.0.dot(a), unit.gate.1.dot(&a2)), bias: unit.bias, write }
}

/// `Σ (x − y)² / v` over paired values with variances `v`.
fn chi2(x: ndarray::ArrayView1<'_, f64>, y: ndarray::ArrayView1<'_, f64>, v: ndarray::ArrayView1<'_, f64>) -> f64 {
    x.iter().zip(y).zip(v).map(|((a, b), v)| if *v > 0.0 { (a - b).powi(2) / v } else if a == b { 0.0 } else { f64::INFINITY }).sum()
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
    match (&from.write, &t.write) {
        (Write::Plain(x), Write::Plain(y)) => {
            total += chi2(x.0.view(), y.0.view(), v(&x.1, &y.1).view());
            entries += x.0.len();
        }
        (Write::Gated { map: x, offset: xo }, Write::Gated { map: y, offset: yo }) => {
            let flat = |m: &Array2<f64>| Array1::from_iter(m.iter().copied());
            total += chi2(flat(&x.0).view(), flat(&y.0).view(), v(&flat(&x.1), &flat(&y.1)).view());
            entries += x.0.len();
            if let (Some(x), Some(y)) = (xo, yo) {
                total += chi2(x.0.view(), y.0.view(), v(&x.1, &y.1).view());
                entries += x.0.len();
            }
        }
        (Write::Plain(_), Write::Gated { .. }) | (Write::Gated { .. }, Write::Plain(_)) => return (f64::INFINITY, entries),
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
    total += match &unit.write {
        Write::Plain((u, v)) => chi2(u.view(), zero(u.len()).view(), v.view()),
        Write::Gated { map: (m, v), offset } => {
            let flat = |m: &Array2<f64>| Array1::from_iter(m.iter().copied());
            chi2(flat(m).view(), zero(m.len()).view(), flat(v).view()) + offset.as_ref().map_or(0.0, |(n, v)| chi2(n.view(), zero(n.len()).view(), v.view()))
        }
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

/// Weighted least squares per target column: `x` minimizing `Σ_i w_ic (y_ic − (D x)_ic)²` for
/// each column `c`, `D` the design (`rows × p`), `y` and `w` (`rows × q`); `x` is `p × q`. The
/// directions the weighted design does not resolve get zero.
fn weighted_columns(design: &Array2<f64>, y: &Array2<f64>, w: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut x = Array2::zeros((design.ncols(), y.ncols()));
    for c in 0..y.ncols() {
        let root: Array1<f64> = w.column(c).mapv(f64::sqrt);
        let scaled = design * &root.view().insert_axis(Axis(1));
        let target = &y.column(c) * &root;
        if let Some((left, values, right)) = resolved(&scaled)? {
            x.column_mut(c).assign(&right.t().dot(&(left.t().dot(&target) / &values)));
        }
    }
    Ok(x)
}

/// Precisions `1/v` (zero where the variance is not positive: a removed entry is not data).
fn precisions(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(|v| if v > 0.0 { 1.0 / v } else { 0.0 })
}

/// `A` minimizing the misfit over the matched pairs `(from unit, onto unit)` with `C` fixed: the
/// gate rows `g_i ≈ g_j A` and, gated, the rows of `M_i ≈ (C M_j) A`.
fn input_gauge(pairs: &[(&Unit, &Unit)], c: &Array2<f64>, k_onto: usize, k_from: usize) -> Result<Array2<f64>, String> {
    let (mut design, mut targets, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for (f, o) in pairs {
        design.push(o.gate.0.clone());
        targets.push(f.gate.0.clone());
        weights.push(precisions(&f.gate.1));
        if let (Write::Gated { map: (mf, vf), .. }, Write::Gated { map: (mo, _), .. }) = (&f.write, &o.write) {
            let cm = c.dot(mo);
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
        match (&f.write, &o.write) {
            (Write::Plain((u, v)), Write::Plain((uo, _))) => {
                design.push(uo.clone());
                targets.push(u.clone());
                weights.push(precisions(v));
            }
            (Write::Gated { map: (mf, vf), offset: of }, Write::Gated { map: (mo, _), offset: oo }) => {
                let ma = mo.dot(a);
                for col in 0..mf.ncols() {
                    design.push(ma.column(col).to_owned());
                    targets.push(mf.column(col).to_owned());
                    weights.push(precisions(&vf.column(col).to_owned()));
                }
                if let (Some((nf, vn)), Some((no, _))) = (of, oo) {
                    design.push(no.clone());
                    targets.push(nf.clone());
                    weights.push(precisions(vn));
                }
            }
            (Write::Plain(_), Write::Gated { .. }) | (Write::Gated { .. }, Write::Plain(_)) => return Err("bodies of different laws".into()),
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

/// Per live unit a signature unchanged by the gauges and by permutations of the other units: the
/// sorted magnitudes of its row of the projection onto the units' read span (gate rows) and onto
/// their write span (output columns, or gated the rank-one maps).
fn signatures(units: &[(usize, Unit)]) -> Result<Vec<Array1<f64>>, String> {
    let m = units.len();
    let reads = Array2::from_shape_fn((m, units.first().map_or(0, |u| u.1.gate.0.len())), |(i, j)| units[i].1.gate.0[j]);
    let flat = |u: &Unit| -> Array1<f64> {
        match &u.write {
            Write::Plain((w, _)) => w.clone(),
            Write::Gated { map: (w, _), .. } => Array1::from_iter(w.iter().copied()),
        }
    };
    let width = units.first().map_or(0, |u| flat(&u.1).len());
    let writes = Array2::from_shape_fn((m, width), |(i, j)| flat(&units[i].1)[j]);
    let (pr, pw) = (span_projection(&reads)?, span_projection(&writes)?);
    Ok((0..m)
        .map(|i| {
            let sorted = |p: &Array2<f64>| {
                let mut row: Vec<f64> = p.row(i).iter().map(|v| v.abs()).collect();
                row.sort_by(|a, b| b.total_cmp(a));
                row
            };
            Array1::from_iter(sorted(&pr).into_iter().chain(sorted(&pw)))
        })
        .collect())
}

/// The alignment of body `from` to body `onto` (module note).
pub fn align(from: &BodyValues, onto: &BodyValues) -> Result<Alignment, String> {
    if from.up.is_some() != onto.up.is_some() || from.gate_bias.is_some() != onto.gate_bias.is_some() || from.up_bias.is_some() != onto.up_bias.is_some() {
        return Err("bodies of different laws".into());
    }
    let (f_units, o_units) = (live_units(from), live_units(onto));
    let (k_from, k_onto) = (from.inputs.iter().filter(|u| **u).count(), onto.inputs.iter().filter(|u| **u).count());
    let (kout_from, kout_onto) = (from.outputs.iter().filter(|u| **u).count(), onto.outputs.iter().filter(|u| **u).count());
    let n = f_units.len().max(o_units.len());
    // The start: the assignment of the signatures, padded with unmatched units at zero cost.
    let (sf, so) = (signatures(&f_units)?, signatures(&o_units)?);
    let start = Array2::from_shape_fn((n, n), |(i, j)| match (sf.get(i), so.get(j)) {
        (Some(a), Some(b)) if a.len() == b.len() => (a - b).mapv(|v| v * v).sum(),
        (Some(a), Some(b)) => {
            let common = a.len().min(b.len());
            (a.slice(s![..common]).to_owned() - b.slice(s![..common])).mapv(|v| v * v).sum()
        }
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
    // The output gauge starts as the identity on the common coordinates, so the first input gauge
    // sees the gated maps at a start. Each step minimizes the misfit over its own block (the input
    // gauge, the output gauge, the assignment) and is taken only when it lowers the misfit, so the
    // misfit decreases strictly and the alternation ends.
    let mut c = Array2::eye(kout_from.max(kout_onto)).slice(s![..kout_from, ..kout_onto]).to_owned();
    let mut a = input_gauge(&pairs_of(&assignment), &c, k_onto, k_from)?;
    let mut current = total(&assignment, &a, &c);
    loop {
        loop {
            let c_next = output_gauge(&pairs_of(&assignment), &a, kout_from, kout_onto)?;
            let a_next = input_gauge(&pairs_of(&assignment), &c_next, k_onto, k_from)?;
            let next = total(&assignment, &a_next, &c_next);
            if !(next < current) {
                break;
            }
            (a, c, current) = (a_next, c_next, next);
        }
        let next = hungarian(&assignment_costs(&f_units, &o_units, &a, &c))?;
        let value = total(&next, &a, &c);
        if !(value < current) {
            break;
        }
        (assignment, current) = (next, value);
    }
    // The misfit in units of both posteriors' deviations, and the full-size gauges.
    let (mut misfit, mut entries) = (0.0, 0);
    let mut units = vec![None; from.units.len()];
    for (i, (fi, f)) in f_units.iter().enumerate() {
        match o_units.get(assignment[i]) {
            Some((oj, o)) => {
                let (m, e) = unit_misfit(f, &transformed(o, &a, &c), true);
                misfit += m;
                entries += e;
                units[*fi] = Some(*oj);
            }
            None => {
                misfit += alone(f);
                entries += unit_misfit(f, f, false).1;
            }
        }
    }
    for j in (0..n).filter(|j| *j < o_units.len() && !assignment.contains(j)).collect::<Vec<_>>() {
        misfit += alone(&o_units[j].1);
        entries += unit_misfit(&o_units[j].1, &o_units[j].1, false).1;
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
    Ok(Alignment { units, input: embed(&a, &onto.inputs, &from.inputs), output: embed(&c, &from.outputs, &onto.outputs), misfit, entries, gauge })
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

/// `explanation` with every call of body `from` made a call of body `onto` with its bindings
/// `A R` and `W C` (`alignment` of `from` to `onto`), `from`'s rule and operators gone, and the
/// calls' records with it (module note).
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
    let mut groups = Vec::new();
    for (g, group) in explanation.groups.iter().enumerate() {
        if group.cells.iter().any(|cell| retired.contains(&cell.operator) || rebound.contains(&cell.operator)) {
            continue;
        }
        index.insert(g, groups.len());
        groups.push(group.clone());
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
    merged.removed = removed;
    merged.trainable.retain(|op| !retired.contains(op));
    // The calls' choice of body changes; `from`'s widths are no longer sent.
    let program = &merged.artifact.program;
    let from_widths = crate::codec::elias_delta_len_bits(explanation.artifact.program.operators[from_ops.gate].cols.width() as u64).map_err(error)?
        + crate::codec::elias_delta_len_bits(explanation.artifact.program.operators[from_ops.out].rows.width() as u64).map_err(error)?;
    merged.fixed_nats += assignment_nats(program)? - assignment_nats(&explanation.artifact.program)? - from_widths as f64 * std::f64::consts::LN_2;
    // Owners of `from`'s blocks now own `onto`'s block of the matched unit at the same call, whose
    // bindings took the gauge (`g_i = g_j A`, so `a_i = g_j (A R)`; `u_i = C u_j`, so `W u_i =
    // (W C) u_j`), and for a gated law the unit's up scale `α` (`b_i = α b_j A`, `u_i = C u_j / α`),
    // kept as 1 × 1 operators of the call that no node reads. A native function whose unit is
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
    let mut scalars: BTreeMap<(String, usize), (String, String)> = BTreeMap::new();
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
                    let (alpha, inverse) = scalars.entry((owner.site.clone(), unit)).or_insert_with(|| (format!("{}.u{unit}.alpha", owner.site), format!("{}.u{unit}.alpha_inverse", owner.site))).clone();
                    if *piece == Piece::Out { owner.right.push(inverse) } else { owner.left.push(alpha) }
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
    for ((_, unit), (alpha, inverse)) in &scalars {
        let value = alphas.get(*unit).copied().unwrap_or(1.0);
        for (scalar, v) in [(alpha, value), (inverse, 1.0 / value)] {
            merged.artifact.program.operators.push(Arc::new(dense(scalar.clone(), one.clone(), one.clone(), Array2::from_elem((1, 1), v), Provenance::derived(&[], format!("merge of {from} into {onto}: a unit's up scale")))?));
        }
    }
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

/// The parameters a rewrite of the functions with posterior-whitened reads `reads` (their gate rows,
/// and up rows when gated, each entry over its posterior deviation; `parts` rows per function) and
/// whitened writes `writes` (their output columns, one row each) saves: `|S| (parts d + d′)` native
/// entries against the body's `k (d + parts |S|) + k′ (d′ + |S|)`, with `k` and `k′` the numbers of
/// whitened singular values above the largest singular value of a matrix of the same shape of
/// independent unit-variance noise, `√rows + √cols` (Bai and Yin 1988): the directions the
/// posterior resolves from its own noise.
fn saving(reads: &Array2<f64>, writes: &Array2<f64>, parts: usize) -> Result<f64, String> {
    let resolved_rank = |m: &Array2<f64>| -> Result<usize, String> {
        let edge = (m.nrows() as f64).sqrt() + (m.ncols() as f64).sqrt();
        Ok(svd(m.view(), false).map_err(error)?.singular_values.iter().filter(|s| **s > edge).count())
    };
    let n = writes.nrows() as f64;
    let (d, d_out) = (reads.ncols() as f64, writes.ncols() as f64);
    let (k, k_out) = (resolved_rank(reads)? as f64, resolved_rank(writes)? as f64);
    Ok(n * (parts as f64 * d + d_out) - (k * (d + parts as f64 * n) + k_out * (d_out + n)))
}

/// The candidate regions of layer `layer`'s MLP among the native functions `pool` (those carrying
/// flow) at `posterior` (module note): functions are grouped greedily, the pair of groups whose
/// union's rewrite saves the most parameters beyond the two apart ([`saving`]) first, while one
/// does; the groups whose rewrite saves parameters are the regions. Parallel functions of one body
/// read and write the same few directions, so their union saves what each alone cannot; a function
/// reading a direction of its own adds a coordinate to each binding and is left out.
pub fn regions(explanation: &Explanation, posterior: &Posterior, layer: usize, pool: &[usize]) -> Result<Vec<Vec<usize>>, String> {
    let program = &explanation.artifact.program;
    let mlp = format!("library.l{layer}.mlp");
    let maps: Vec<usize> = ["gate", "up"].iter().filter_map(|part| operator_named(program, &format!("{mlp}.{part}"))).collect();
    let out = operator_index(program, &format!("{mlp}.out"))?;
    let means = posterior.means();
    let at = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("{mlp}: operator {op} is not trainable"));
    let known = &explanation.layers.get(layer).ok_or_else(|| format!("no layer {layer}"))?.functions;
    let whitened = |i: usize, op: usize, row: bool| -> Result<Array1<f64>, String> {
        let p = at(op)?;
        let (mean, sd) = (&means[p], posterior.log_sd[p].mapv(f64::exp));
        Ok(if row { &mean.row(i) / &sd.row(i) } else { &mean.column(i) / &sd.column(i) })
    };
    // Each function of the pool in the explanation: its whitened read rows and write column.
    let mut functions = Vec::new();
    for &i in pool {
        let groups = known.get(i).ok_or_else(|| format!("layer {layer} has no function {i}"))?;
        if groups.iter().all(|g| posterior.active[*g]) {
            let reads = maps.iter().map(|op| whitened(i, *op, true)).collect::<Result<Vec<_>, _>>()?;
            functions.push((i, reads, whitened(i, out, false)?));
        }
    }
    let parts = maps.len();
    let stack = |members: &[usize]| -> (Array2<f64>, Array2<f64>) {
        let d = functions[members[0]].1[0].len();
        let reads = Array2::from_shape_fn((members.len() * parts, d), |(r, c)| functions[members[r / parts]].1[r % parts][c]);
        let writes = Array2::from_shape_fn((members.len(), functions[members[0]].2.len()), |(r, c)| functions[members[r]].2[c]);
        (reads, writes)
    };
    let value = |members: &[usize]| -> Result<f64, String> {
        let (reads, writes) = stack(members);
        saving(&reads, &writes, parts)
    };
    let mut groups: Vec<(Vec<usize>, f64)> = (0..functions.len()).map(|f| Ok((vec![f], value(&[f])?))).collect::<Result<_, String>>()?;
    // The gain of joining each pair, kept and recomputed only for pairs with a new group.
    let gain = |a: &(Vec<usize>, f64), b: &(Vec<usize>, f64)| -> Result<f64, String> {
        let union: Vec<usize> = a.0.iter().chain(&b.0).copied().collect();
        Ok(value(&union)? - a.1 - b.1)
    };
    let mut gains: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for a in 0..groups.len() {
        for b in a + 1..groups.len() {
            gains.insert((a, b), gain(&groups[a], &groups[b])?);
        }
    }
    let mut alive: Vec<bool> = vec![true; groups.len()];
    loop {
        let best = gains.iter().filter(|(_, g)| **g > 0.0).max_by(|x, y| x.1.total_cmp(y.1)).map(|(k, g)| (*k, *g));
        let Some(((a, b), value_gain)) = best else { break };
        let members: Vec<usize> = groups[a].0.iter().chain(&groups[b].0).copied().collect();
        let total = groups[a].1 + groups[b].1 + value_gain;
        alive[a] = false;
        alive[b] = false;
        gains.retain(|(x, y), _| ![a, b].contains(x) && ![a, b].contains(y));
        groups.push((members, total));
        alive.push(true);
        let new = groups.len() - 1;
        for other in (0..new).filter(|o| alive[*o]) {
            gains.insert((other, new), gain(&groups[other], &groups[new])?);
        }
    }
    Ok(groups
        .into_iter()
        .zip(alive)
        .filter(|((_, value), alive)| *alive && *value > 0.0)
        .map(|((members, _), _)| {
            let mut region: Vec<usize> = members.iter().map(|f| functions[*f].0).collect();
            region.sort_unstable();
            region
        })
        .collect())
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
}

/// One component of a body's mixture prior: an earlier body, the gauge relating them (fixed for an
/// epoch: the alignment of the target to it and each target unit's up scale `α`), its scale `c` and
/// its logit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyComponent {
    pub body: String,
    pub alignment: Alignment,
    pub up_scales: Vec<f64>,
    pub scale: f64,
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

/// The mixture prior over bodies (module note, # Reuse by gradient).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyMixture {
    pub targets: Vec<BodyTarget>,
    steps: crate::library_mixture::Steps,
    taken: u64,
    /// Per target, Adam's moments of its zero logit, then per component its logit and scale, then
    /// its `ln s²`.
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
                step(&mut moments[1 + 2 * j], &mut component.logit, gradient[1 + 2 * j]);
                step(&mut moments[2 + 2 * j], &mut component.scale, gradient[2 + 2 * j]);
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
                if alignment.evidence().is_none() {
                    continue;
                }
                let up = up_scales(target, candidate, &alignment);
                let (component, moment) = match old.iter().position(|c| c.body == earlier.body) {
                    Some(at) => (BodyComponent { alignment, up_scales: up, ..old[at].clone() }, [self.moments[t][1 + 2 * at], self.moments[t][2 + 2 * at]]),
                    None => (BodyComponent { body: earlier.body.clone(), alignment, up_scales: up, scale: 1.0, logit: zero }, [Moment::default(); 2]),
                };
                components.push(component);
                moments.extend(moment);
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
            let writes: Vec<(ndarray::ArrayView1<'_, f64>, f64)> = predictions.iter().zip(&target.components).map(|(p, c)| (p.view(), c.scale)).collect();
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
            for j in 0..target.components.len() {
                own.extend([found.logits[j + 1], found.scales[j]]);
            }
            own.push(found.log_variance);
            learned[t] = own;
        }
        if learn {
            self.learn(&learned);
        }
        Ok((value, gradient))
    }

    /// Per target with components: its `K` components among its `n` earlier bodies (`ln C(n, K)`
    /// nats), and its `K` logits, `K` scales, variance and every component's gauge (its alignment's
    /// fitted entries and, gated, its up scales), each at the precision of a value estimated from
    /// the target's live entries (`½ ln |G|` nats).
    fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
        let mut total = 0.0;
        for (t, target) in self.targets.iter().enumerate() {
            let size = self.live(t, posterior)?.len() as f64;
            if target.components.is_empty() || size == 0.0 {
                continue;
            }
            let (n, k) = (target.choices as f64, target.components.len() as f64);
            let gauges: f64 = target.components.iter().map(|c| (c.alignment.gauge + if c.up_scales.iter().any(|a| *a != 1.0) { c.up_scales.len() } else { 0 }) as f64).sum();
            total += statrs::function::gamma::ln_gamma(n + 1.0) - statrs::function::gamma::ln_gamma(k + 1.0) - statrs::function::gamma::ln_gamma(n - k + 1.0)
                + (2.0 * k + 1.0 + gauges) * 0.5 * size.ln();
        }
        Ok(total)
    }

    fn save(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(error)
    }

    fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let restored: BodyMixture = serde_json::from_value(value.clone()).map_err(error)?;
        if restored.targets.len() != self.targets.len() || restored.targets.iter().zip(&self.targets).any(|(a, b)| a.body != b.body) {
            return Err("a checkpoint's body mixture of another explanation".into());
        }
        *self = restored;
        Ok(())
    }
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
            assert!((two.fixed_nats - merged.fixed_nats - 8.0 * ln2).abs() < 1e-12, "{} against {}", merged.fixed_nats, two.fixed_nats);
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

    #[test]
    fn the_parallel_functions_of_a_planted_subroutine_are_its_regions() {
        for (law, gated) in [("gelu_tanh", false), ("silu", true)] {
            let (native, layers, _, _) = tiny(&format!("bodies_regions_{gated}"), law, gated);
            let mut start = explanation(&native, &layers).unwrap();
            planted(&mut start, gated);
            let posterior = Posterior::new(&start, 72).unwrap();
            let pool: Vec<usize> = (0..16).collect();
            for (l, site) in [(0, SITE0), (1, SITE1)] {
                let mut expected = site.to_vec();
                expected.sort_unstable();
                assert_eq!(regions(&start, &posterior, l, &pool).unwrap(), vec![expected], "layer {l}");
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
}
