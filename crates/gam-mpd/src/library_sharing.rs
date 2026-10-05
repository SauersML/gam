//! Shared functions of a library explanation (#2951).
//!
//! Several heads of a library ([`crate::library_mdl`]) may compute one attention pattern (VPD's
//! previous-token and induction behaviours appear in more than one head). A shared query–key
//! function replaces the members' query and key maps by one pair `(Q, K)` that every member reads:
//! the first member attends with `q = Q x`, `k = K x`, and every other member `m` with
//! `q = c_m Q x`, `k = K x`, where the scalar `c_m` (starting at 1) is its own part. Every member
//! keeps its value map. The shared maps carry the prior groups of one head (a group per rotary
//! plane, as in the library), paid once in `KL(q ‖ p)`; each `c_m` is one group. Members sit in
//! different layers, so each layer's read of `Q x` stays its own variable, and each reads a key of
//! its own: a key the query heads of one key-value group share stays that group's. The move is
//! accepted only if the code length `F` of the re-converged fit falls.
//!
//! # Candidates
//!
//! A head's scores are `qᵀ R k` with `R` the rotary turn between the two positions; a rotation and
//! a scaling of a plane, `(s R q, R k / s)`, leave them unchanged. The pairs to share are found by
//! the learned mixture prior over the heads' query–key maps (`library_mixture`).
//!
//! # The start
//!
//! The shared maps start at the members' mean after each member is brought to the first member's
//! gauge, plane by plane: the rotation `R` that minimizes `‖R Q_m − Q_1‖² + ‖R K_m − K_1‖²` over a
//! plane's two rows (the orthogonal Procrustes solution in two dimensions; a sign on a coordinate
//! no rotary plane holds), and the scale `s` with `(s R Q_m, R K_m / s)` in the first member's
//! ratio of query to key norm. Both leave the member's scores unchanged, so the start is close to
//! the current fit.
//!
//! # Read–write ties
//!
//! A residual-stream feature that an earlier MLP function `j` writes (its output `u_j`) and a
//! later MLP function `i` reads (its gate direction `g_i`) can be described once. A tie stores the
//! one vector `g_i` and makes the earlier output `u_j = c g_i`, with `c` one new scalar in a prior
//! group of its own; `u_j`'s own group leaves the explanation ([`Explanation::removed`]). In the
//! earlier layer's MLP each tied function's activation `a_j` is selected, scaled by its `c`
//! (a 1 × 1 operator) and scattered to the later layer's function row, and the sum is read through
//! that layer's gate operator in its transposed orientation, `Σ_t c_t a_{j_t} g_{i_t}`, which joins
//! the layer's output. The gate
//! operator is the one the later layer applies, so its gradient sums over both uses and its
//! `KL(q ‖ p)` is charged once; read variables stay gate rows.
//!
//! Ties are proposed by a learned mixture prior over the gate directions (`library_mixture`); a
//! tie is accepted only if the code length `F` falls after the fit re-converges on the same
//! experiments.

use crate::{
    library_mdl::{Cells, Explanation, Group},
    operator_program::{Interface, LabelKind, Node, Operator, OperatorProgram, Provenance, Rotary, exact_precision, remap_node},
};
use ndarray::Array2;
use std::{collections::BTreeMap, sync::Arc};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A head's block in the explanation's program: its rule's index, its query, key and value
/// operators, and its rotary planes.
pub(crate) struct Head {
    rule: usize,
    pub(crate) query: usize,
    pub(crate) key: usize,
    pub(crate) rotary: Option<Rotary>,
}

fn head(program: &OperatorProgram, layer: usize, h: usize) -> Result<Option<Head>, String> {
    let name = format!("library.l{layer}.h{h}");
    let Some(rule) = program.rules.iter().position(|r| r.name == name) else { return Ok(None) };
    let body = &program.rules[rule];
    let map = |node: usize| match body.nodes.get(node) {
        Some(Node::Affine { terms, bias: None }) if terms.len() == 1 && terms[0].0 == 0 => Ok(terms[0].1),
        other => Err(format!("{name}: node {node} is not a map of the head's input: {other:?}")),
    };
    let rotary = match body.nodes.get(body.output) {
        Some(Node::Attend { rotary, .. }) => *rotary,
        other => Err(format!("{name}: the output is not an attention node: {other:?}"))?,
    };
    Ok(Some(Head { rule, query: map(1)?, key: map(2)?, rotary }))
}

/// Every head of the explanation, by `(layer, head)`, whose key no other head reads: a key that
/// the query heads of one key-value group share (`library_mdl::explanation`) stays that group's.
pub(crate) fn heads(explanation: &Explanation) -> Result<BTreeMap<(usize, usize), Head>, String> {
    let program = &explanation.artifact.program;
    let mut out = BTreeMap::new();
    for l in 0..explanation.layers.len() {
        for h in 0.. {
            let Some(found) = head(program, l, h)? else { break };
            out.insert((l, h), found);
        }
    }
    let mut readers: BTreeMap<usize, usize> = BTreeMap::new();
    for found in out.values() {
        *readers.entry(found.key).or_default() += 1;
    }
    out.retain(|_, found| readers[&found.key] == 1);
    Ok(out)
}

/// The row sets the score map's gauge acts on: each rotary plane's two rows, then each coordinate
/// no plane holds.
pub(crate) fn planes(width: usize, rotary: Option<Rotary>) -> Vec<Vec<usize>> {
    let pairs = rotary.map(|r| r.pairs()).unwrap_or_default();
    let rotated: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    pairs.iter().map(|&(a, b)| vec![a, b]).chain((0..width).filter(|c| !rotated.contains(c)).map(|c| vec![c])).collect()
}

fn norm(x: &Array2<f64>) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Per plane of `planes`, the rotation (on one coordinate, the sign) and the scale that bring
/// `(q, k)` to the gauge of `(q1, k1)` (module note).
pub(crate) fn gauge(q1: &Array2<f64>, k1: &Array2<f64>, q: &Array2<f64>, k: &Array2<f64>, planes: &[Vec<usize>]) -> Vec<(Array2<f64>, f64)> {
    planes
        .iter()
        .map(|rows| {
            let pick = |m: &Array2<f64>| m.select(ndarray::Axis(0), rows);
            let (a_q, a_k, b_q, b_k) = (pick(q1), pick(k1), pick(q), pick(k));
            // The rotation (or, on one coordinate, the sign) nearest the first member's rows.
            let m = a_q.dot(&b_q.t()) + a_k.dot(&b_k.t());
            let rotation = if rows.len() == 2 {
                let angle = (m[[1, 0]] - m[[0, 1]]).atan2(m[[0, 0]] + m[[1, 1]]);
                let (sine, cosine) = angle.sin_cos();
                ndarray::array![[cosine, -sine], [sine, cosine]]
            } else {
                ndarray::array![[if m[[0, 0]] < 0.0 { -1.0 } else { 1.0 }]]
            };
            let (r_q, r_k) = (rotation.dot(&b_q), rotation.dot(&b_k));
            let (nq, nk, n1q, n1k) = (norm(&r_q), norm(&r_k), norm(&a_q), norm(&a_k));
            let scale = if nq > 0.0 && nk > 0.0 && n1q > 0.0 && n1k > 0.0 { (n1q * nk / (nq * n1k)).sqrt() } else { 1.0 };
            (rotation, scale)
        })
        .collect()
}

/// `(q, k)` moved by `gauge` plane by plane, `(s R q, R k / s)`; with `transpose`, the transpose
/// of that map (a cotangent of the result back to `(q, k)`).
pub(crate) fn apply_gauge(q: &Array2<f64>, k: &Array2<f64>, planes: &[Vec<usize>], gauge: &[(Array2<f64>, f64)], transpose: bool) -> (Array2<f64>, Array2<f64>) {
    let (mut q_out, mut k_out) = (q.clone(), k.clone());
    for (rows, (rotation, scale)) in planes.iter().zip(gauge) {
        let rotation = if transpose { rotation.t().to_owned() } else { rotation.clone() };
        let pick = |m: &Array2<f64>| m.select(ndarray::Axis(0), rows);
        let (r_q, r_k) = (rotation.dot(&pick(q)), rotation.dot(&pick(k)));
        for (i, &row) in rows.iter().enumerate() {
            q_out.row_mut(row).assign(&(&r_q.row(i) * *scale));
            k_out.row_mut(row).assign(&(&r_k.row(i) / *scale));
        }
    }
    (q_out, k_out)
}

/// `(q, k)` of a member brought to the gauge of `(q1, k1)` plane by plane (module note).
fn aligned(q1: &Array2<f64>, k1: &Array2<f64>, q: &Array2<f64>, k: &Array2<f64>, planes: &[Vec<usize>]) -> (Array2<f64>, Array2<f64>) {
    apply_gauge(q, k, planes, &gauge(q1, k1, q, k, planes), false)
}

/// A dense library operator holding `values`.
fn dense(name: String, rows: Interface, cols: Interface, values: Array2<f64>, provenance: Provenance) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, provenance).map_err(error)
}

/// `explanation` with the heads `members` (`(layer, head)`, at most one per layer, the first the
/// shared function's owner) reading one shared query–key function (module note), at the
/// explanation's operator values.
pub fn share_query_key(explanation: &Explanation, members: &[(usize, usize)]) -> Result<Explanation, String> {
    if members.len() < 2 {
        return Err("a shared function needs two members".into());
    }
    if (1..members.len()).any(|i| members[..i].iter().any(|m| m.0 == members[i].0)) {
        return Err("the members of a shared query-key function lie in different layers".into());
    }
    let all = heads(explanation)?;
    let found: Vec<&Head> = members.iter().map(|m| all.get(m).ok_or_else(|| format!("no head {m:?} with a key of its own"))).collect::<Result<_, _>>()?;
    let mut artifact = explanation.artifact.clone();
    // Each member keeps its native owners, now read through the shared maps.
    let names = |h: &Head| (artifact.program.operators[h.query].name.clone(), artifact.program.operators[h.key].name.clone());
    let shared = names(found[0]);
    let retiring: Vec<((String, String), String)> = found[1..].iter().zip(&members[1..]).map(|(h, m)| (names(h), format!("library.l{}.h{}.q_shared_scale", m.0, m.1))).collect();
    for owner in &mut artifact.owners {
        for ((q, k), scale) in &retiring {
            let (rows, cols) = (owner.rows.clone(), owner.cols.clone());
            if owner.operator == *q {
                owner.repoint(&shared.0, rows, cols, std::slice::from_ref(scale), &[]);
            } else if owner.operator == *k {
                owner.repoint(&shared.1, rows, cols, &[], &[]);
            }
        }
    }
    let program = &mut artifact.program;
    let owner = found[0];
    let (q1, k1) = (program.operators[owner.query].matrix(), program.operators[owner.key].matrix());
    let gauge = planes(q1.nrows(), owner.rotary);
    let (mut q_sum, mut k_sum) = (q1.clone(), k1.clone());
    for m in &found[1..] {
        let (q, k) = (program.operators[m.query].matrix(), program.operators[m.key].matrix());
        if q.dim() != q1.dim() || k.dim() != k1.dim() || m.rotary != owner.rotary {
            return Err("the members' query and key maps differ in shape or rotary planes".into());
        }
        let (q, k) = aligned(&q1, &k1, &q, &k, &gauge);
        q_sum += &q;
        k_sum += &k;
    }
    let n = members.len() as f64;
    let provenance = Provenance::derived(&[&program.operators[owner.query].provenance], "shared query-key function".into());
    for (op, values) in [(owner.query, q_sum / n), (owner.key, k_sum / n)] {
        let source = &program.operators[op];
        program.operators[op] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, provenance.clone())?);
    }
    // Every other member reads the shared maps, its query scaled by its own `c_m` (a 1 × 1
    // operator, broadcast over the query's coordinates by a fixed column of ones).
    let coordinates = program.operators[owner.query].rows.clone();
    let one = Interface::uniform(1, 1, LabelKind::Unit, 0).map_err(error)?;
    let mut retired = Vec::new();
    let mut parts = Vec::new();
    for (m, member) in members.iter().zip(&found).skip(1) {
        retired.extend([member.query, member.key]);
        let scale = program.operators.len();
        let name = format!("library.l{}.h{}.q_shared_scale", m.0, m.1);
        program.operators.push(Arc::new(dense(name.clone(), one.clone(), Interface::constant(), Array2::ones((1, 1)), provenance.clone())?));
        program.operators.push(Arc::new(dense(format!("library.l{}.h{}.q_shared_ones", m.0, m.1), coordinates.clone(), one.clone(), Array2::ones((coordinates.width(), 1)), provenance.clone())?));
        parts.push((name, scale));
        let rule = &mut program.rules[member.rule];
        rule.nodes[1] = Node::Affine { terms: vec![(0, owner.query)], bias: None };
        rule.nodes[2] = Node::Affine { terms: vec![(0, owner.key)], bias: None };
        let output = rule.output;
        let Node::Attend { query, .. } = rule.nodes[output] else {
            return Err(format!("{}: the output is not an attention node", rule.name));
        };
        rule.nodes.insert(output, Node::Constant { operator: scale });
        rule.nodes.insert(output + 1, Node::Affine { terms: vec![(output, scale + 1)], bias: None });
        rule.nodes.insert(output + 2, Node::Hadamard { left: query, right: output + 1 });
        rule.output = output + 3;
        if let Node::Attend { query, .. } = &mut rule.nodes[output + 3] {
            *query = output + 2;
        }
    }
    program.interfaces().map_err(error)?;
    // The retired maps leave the library with their groups; each part is one group.
    let mut index = BTreeMap::new();
    let mut groups = Vec::new();
    for (g, group) in explanation.groups.iter().enumerate() {
        if group.cells.iter().any(|c| retired.contains(&c.operator)) {
            continue;
        }
        index.insert(g, groups.len());
        groups.push(group.clone());
    }
    for (name, scale) in &parts {
        groups.push(Group { name: name.clone(), cells: vec![Cells { operator: *scale, rows: vec![0], cols: 0..1 }] });
    }
    let mut trainable: Vec<usize> = explanation.trainable.iter().copied().filter(|op| !retired.contains(op)).chain(parts.iter().map(|p| p.1)).collect();
    trainable.sort_unstable();
    let owner_planes: Vec<usize> = explanation.layers[members[0].0].heads[members[0].1].0.iter().filter_map(|g| index.get(g).copied()).collect();
    let mut layers = explanation.layers.clone();
    for (l, layer) in layers.iter_mut().enumerate() {
        for (h, (planes, values)) in layer.heads.iter_mut().enumerate() {
            *planes = if members[1..].contains(&(l, h)) { owner_planes.clone() } else { planes.iter().filter_map(|g| index.get(g).copied()).collect() };
            *values = values.iter().filter_map(|g| index.get(g).copied()).collect();
        }
        for function in &mut layer.functions {
            *function = function.iter().filter_map(|g| index.get(g).copied()).collect();
        }
    }
    let removed = explanation.removed.iter().filter_map(|g| index.get(g).copied()).collect();
    Ok(Explanation { artifact, trainable, groups, layers, removed, fixed_nats: explanation.fixed_nats })
}

/// `explanation` with its library operators set to `artifact`'s (a posterior-mean artifact of a fit
/// of this explanation), so a fit of it starts where that fit stood.
pub fn warm(explanation: &Explanation, artifact: &crate::artifact::Artifact) -> Result<Explanation, String> {
    let mut out = explanation.clone();
    for &op in &explanation.trainable {
        let (source, fitted) = (&explanation.artifact.program.operators[op], artifact.program.operators.get(op).ok_or("the fitted artifact lacks a library operator")?);
        if fitted.name != source.name || fitted.rows.width() != source.rows.width() || fitted.cols.width() != source.cols.width() {
            return Err(format!("{}: the fitted artifact's operator differs in name or shape", source.name));
        }
        let values = fitted.matrix();
        let provenance = Provenance::derived(&[&source.provenance], "fitted posterior mean".into());
        out.artifact.program.operators[op] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, provenance)?);
    }
    Ok(out)
}

/// A read–write tie (module note): function `source.1` of layer `source.0` writes `scale` times the
/// gate direction of function `target.1` of the later layer `target.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tie {
    pub source: (usize, usize),
    pub target: (usize, usize),
    pub scale: f64,
}

fn operator_index(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(format!("no unique operator {name}")),
    }
}

/// `explanation` with the read–write `ties` made (module note): each tied output leaves the
/// explanation and is written through the later gate it is tied to, scaled by its own `c`.
pub fn tie(explanation: &Explanation, ties: &[Tie]) -> Result<Explanation, String> {
    let mut out = explanation.clone();
    let mut sources: BTreeMap<(usize, usize), Vec<&Tie>> = BTreeMap::new();
    for tie in ties {
        if tie.target.0 <= tie.source.0 {
            return Err(format!("a tie of layer {} to layer {} reads a write before it is made", tie.source.0, tie.target.0));
        }
        if ties.iter().filter(|t| t.source == tie.source).count() > 1 {
            return Err(format!("function {:?} is tied twice", tie.source));
        }
        sources.entry((tie.source.0, tie.target.0)).or_default().push(tie);
    }
    let group = |out: &Explanation, name: &str| out.groups.iter().position(|g| g.name == name).ok_or_else(|| format!("no group {name}"));
    for (&(source, target), ties) in &sources {
        let k = ties.len();
        let program = &mut out.artifact.program;
        let rule = program.rules.iter().position(|r| r.name == format!("library.l{source}.mlp")).ok_or("no MLP rule")?;
        let output = program.operators[operator_index(program, &format!("library.l{source}.mlp.out"))?].clone();
        let gate_index = operator_index(program, &format!("library.l{target}.mlp.gate"))?;
        let gate = program.operators[gate_index].clone();
        let (out_node, act, out_index, others) = {
            let r = &program.rules[rule];
            match &r.nodes[r.output] {
                Node::Affine { terms, bias: None } if r.output + 1 == r.nodes.len() => {
                    let first = *terms.first().ok_or("an empty MLP output")?;
                    (r.output, first.0, first.1, terms[1..].to_vec())
                }
                other => return Err(format!("layer {source}: the MLP's output is {other:?}")),
            }
        };
        let one = Interface::uniform(1, 1, LabelKind::Unit, 0).map_err(error)?;
        let provenance = Provenance::derived(&[&gate.provenance, &output.provenance], "read-write tie".into());
        let name = format!("library.l{source}.mlp.tie{target}");
        // Per tie its selection of `a_j`, its scale `c` (the one parameter) and its scatter to row
        // `i`; then the later gate read transposed, and an identity into the output's interface.
        let base = program.operators.len();
        for tie in ties.iter() {
            let (j, i) = (tie.source.1, tie.target.1);
            let mut select = Array2::zeros((1, output.cols.width()));
            select[[0, j]] = 1.0;
            let mut scatter = Array2::zeros((gate.rows.width(), 1));
            scatter[[i, 0]] = 1.0;
            for (part, rows, cols, values) in [
                ("select", one.clone(), output.cols.clone(), select),
                ("scale", one.clone(), one.clone(), Array2::from_elem((1, 1), tie.scale)),
                ("scatter", gate.rows.clone(), one.clone(), scatter),
            ] {
                program.operators.push(Arc::new(dense(format!("{name}.f{j}.{part}"), rows, cols, values, provenance.clone())?));
            }
        }
        let identity = program.operators.len();
        program.operators.push(Arc::new(dense(format!("{name}.identity"), output.rows.clone(), gate.cols.clone(), Array2::eye(gate.cols.width()), provenance)?));
        // The tied outputs leave `OUT`: their columns are zero and their groups removed; each one's
        // native owner now reads the gate row it is tied to.
        for tie in ties.iter() {
            let scale = format!("{name}.f{}.scale", tie.source.1);
            for owner in &mut out.artifact.owners {
                if owner.operator == output.name && owner.cols == (tie.source.1..tie.source.1 + 1) {
                    owner.repoint(&gate.name, tie.target.1..tie.target.1 + 1, 0..gate.cols.width(), std::slice::from_ref(&scale), &[]);
                    owner.transposed = !owner.transposed;
                }
            }
        }
        let program = &mut out.artifact.program;
        let mut values = output.matrix();
        for tie in ties.iter() {
            values.column_mut(tie.source.1).fill(0.0);
        }
        program.operators[out_index] = Arc::new(dense(output.name.clone(), output.rows.clone(), output.cols.clone(), values, output.provenance.clone())?);
        let r = &mut program.rules[rule];
        r.nodes.truncate(out_node);
        let mut scattered = Vec::with_capacity(k);
        for t in 0..k {
            let at = r.nodes.len();
            r.nodes.push(Node::Affine { terms: vec![(act, base + 3 * t)], bias: None });
            r.nodes.push(Node::Affine { terms: vec![(at, base + 3 * t + 1)], bias: None });
            scattered.push((at + 1, base + 3 * t + 2));
        }
        r.nodes.push(Node::Affine { terms: scattered, bias: None });
        r.nodes.push(Node::Transposed { input: r.nodes.len() - 1, operator: gate_index });
        let mut terms = vec![(act, out_index)];
        terms.extend(others);
        terms.push((r.nodes.len() - 1, identity));
        r.nodes.push(Node::Affine { terms, bias: None });
        r.output = r.nodes.len() - 1;
        program.interfaces().map_err(error)?;
        // One group per scale, standing for the tied output's own.
        for (t, tie) in ties.iter().enumerate() {
            let own = group(&out, &format!("library.l{source}.mlp.f{}.out", tie.source.1))?;
            let scale = out.groups.len();
            out.groups.push(Group { name: format!("library.l{source}.mlp.f{}.tie", tie.source.1), cells: vec![Cells { operator: base + 3 * t + 1, rows: vec![0], cols: 0..1 }] });
            out.removed.push(own);
            out.trainable.push(base + 3 * t + 1);
            for function in &mut out.layers[source].functions {
                for g in function.iter_mut() {
                    if *g == own {
                        *g = scale;
                    }
                }
            }
        }
    }
    out.trainable.sort_unstable();
    Ok(out)
}

/// Where a tied read row comes from: a token's embedding row (`M`'s, sent with `M`), or row
/// `function` of another layer's `part` operator (`gate` or `up`) of the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSource {
    Token(usize),
    Row { layer: usize, part: &'static str, function: usize },
}

/// `explanation` with the gate direction of function `target.1` of layer `target.0` made `scale`
/// times the embedding row of `token` ([`tie_row`]).
pub fn tie_token(explanation: &Explanation, target: (usize, usize), token: usize, scale: f64) -> Result<Explanation, String> {
    tie_row(explanation, "gate", target, RowSource::Token(token), scale)
}

/// `explanation` with row `target.1` of layer `target.0`'s `part` operator (`gate` or `up`) made
/// `scale` times `source`: the row's own group leaves the explanation, and the function reads
/// `scale (s · x)` through the source (a fixed copy of an embedding row, or the other layer's
/// operator applied to this layer's input and its row selected), a 1 × 1 scale (one prior group)
/// and a scatter into the row. A source row is stored once: its gradient sums both uses.
pub fn tie_row(explanation: &Explanation, part: &str, target: (usize, usize), source: RowSource, scale: f64) -> Result<Explanation, String> {
    let mut out = explanation.clone();
    let (l, i) = target;
    let program = &mut out.artifact.program;
    let rule = program.rules.iter().position(|r| r.name == format!("library.l{l}.mlp")).ok_or("no MLP rule")?;
    let own_index = operator_index(program, &format!("library.l{l}.mlp.{part}"))?;
    let own = program.operators[own_index].clone();
    if i >= own.rows.width() {
        return Err(format!("no function {i} of layer {l}"));
    }
    let one = Interface::uniform(1, 1, LabelKind::Unit, 0).map_err(error)?;
    let (name, first, owner_to) = match source {
        RowSource::Token(token) => {
            let embedding = program.operators[operator_index(program, "wte")?].matrix();
            if token >= embedding.ncols() {
                return Err(format!("no token {token}"));
            }
            let name = format!("library.l{l}.mlp.f{i}.{part}.token{token}");
            let provenance = Provenance::derived(&[&own.provenance], format!("token {token} tie"));
            let copy = Arc::new(dense(format!("{name}.row"), one.clone(), own.cols.clone(), embedding.column(token).to_owned().insert_axis(ndarray::Axis(0)), provenance)?);
            (name.clone(), vec![copy], (format!("{name}.row"), 0..1))
        }
        RowSource::Row { layer, part: from, function } => {
            let index = operator_index(program, &format!("library.l{layer}.mlp.{from}"))?;
            let other = program.operators[index].clone();
            if other.cols != own.cols || function >= other.rows.width() {
                return Err(format!("row {function} of layer {layer}'s {from} cannot stand for row {i} of layer {l}'s {part}"));
            }
            let mut select = Array2::zeros((1, other.rows.width()));
            select[[0, function]] = 1.0;
            let name = format!("library.l{l}.mlp.f{i}.{part}.from_l{layer}_{from}{function}");
            let provenance = Provenance::derived(&[&own.provenance, &other.provenance], "row tie".into());
            (name.clone(), vec![Arc::new(dense(format!("{name}.select"), one.clone(), other.rows.clone(), select, provenance)?)], (other.name.clone(), function..function + 1))
        }
    };
    let provenance = Provenance::derived(&[&own.provenance], "row tie".into());
    let mut scatter = Array2::zeros((own.rows.width(), 1));
    scatter[[i, 0]] = 1.0;
    let base = program.operators.len();
    program.operators.extend(first);
    let scale_index = program.operators.len();
    program.operators.push(Arc::new(dense(format!("{name}.scale"), one.clone(), one.clone(), Array2::from_elem((1, 1), scale), provenance.clone())?));
    program.operators.push(Arc::new(dense(format!("{name}.scatter"), own.rows.clone(), one.clone(), scatter, provenance)?));
    let mut values = own.matrix();
    values.row_mut(i).fill(0.0);
    program.operators[own_index] = Arc::new(dense(own.name.clone(), own.rows.clone(), own.cols.clone(), values, own.provenance.clone())?);
    // The row's native owners now read the source, through the scale.
    let factor = [format!("{name}.scale")];
    for owner in &mut out.artifact.owners {
        if owner.operator == own.name && owner.rows == (i..i + 1) {
            let cols = owner.cols.clone();
            owner.repoint(&owner_to.0, owner_to.1.clone(), cols, &factor, &[]);
        }
    }
    let program = &mut out.artifact.program;
    // The source's value of this layer's input, its row, its scale: first in the rule after its
    // input; the tied operator's node reads them last.
    let leading: Vec<Node> = match source {
        RowSource::Token(_) => vec![Node::Affine { terms: vec![(0, base)], bias: None }, Node::Affine { terms: vec![(1, scale_index)], bias: None }],
        RowSource::Row { layer, part: from, .. } => {
            let other = operator_index(program, &format!("library.l{layer}.mlp.{from}"))?;
            vec![
                Node::Affine { terms: vec![(0, other)], bias: None },
                Node::Affine { terms: vec![(1, base)], bias: None },
                Node::Affine { terms: vec![(2, scale_index)], bias: None },
            ]
        }
    };
    let shift = leading.len();
    let (ops, bases, rules): (Vec<usize>, Vec<usize>, Vec<usize>) = ((0..program.operators.len()).collect(), (0..program.bases.len()).collect(), (0..program.rules.len()).collect());
    let r = &mut program.rules[rule];
    let map: Vec<usize> = (0..r.nodes.len()).map(|n| if n == 0 { 0 } else { n + shift }).collect();
    let mut nodes = vec![r.nodes[0].clone()];
    nodes.extend(leading);
    for node in &r.nodes[1..] {
        let mut node = node.clone();
        remap_node(&mut node, &map, &ops, &bases, &rules);
        if let Node::Affine { terms, .. } = &mut node
            && terms.first().is_some_and(|t| t.0 == 0 && t.1 == own_index)
        {
            terms.push((shift, scale_index + 1));
        }
        nodes.push(node);
    }
    r.output = map[r.output];
    r.nodes = nodes;
    program.interfaces().map_err(error)?;
    let own_group = out.groups.iter().position(|g| g.name == format!("library.l{l}.mlp.f{i}.{part}")).ok_or_else(|| format!("no {part} group"))?;
    let tied = out.groups.len();
    out.groups.push(Group { name: format!("library.l{l}.mlp.f{i}.{part}.tie"), cells: vec![Cells { operator: scale_index, rows: vec![0], cols: 0..1 }] });
    out.removed.push(own_group);
    out.trainable.push(scale_index);
    out.trainable.sort_unstable();
    for g in out.layers[l].functions[i].iter_mut() {
        if *g == own_group {
            *g = tied;
        }
    }
    Ok(out)
}

/// `explanation` with the output vector of function `target.1` of layer `target.0` made `scale`
/// times the output vector of function `source.1` of layer `source.0`: the column's own group
/// leaves the explanation, and the function's activation, selected, scaled (a 1 × 1 operator, one
/// prior group) and scattered to the source's function, is written through the source layer's
/// output operator. The source column is stored once: its gradient sums both uses.
pub fn tie_column(explanation: &Explanation, target: (usize, usize), source: (usize, usize), scale: f64) -> Result<Explanation, String> {
    let mut out = explanation.clone();
    let ((l, i), (sl, j)) = (target, source);
    let program = &mut out.artifact.program;
    let rule = program.rules.iter().position(|r| r.name == format!("library.l{l}.mlp")).ok_or("no MLP rule")?;
    let own_index = operator_index(program, &format!("library.l{l}.mlp.out"))?;
    let source_index = operator_index(program, &format!("library.l{sl}.mlp.out"))?;
    let (own, other) = (program.operators[own_index].clone(), program.operators[source_index].clone());
    if own.rows != other.rows || i >= own.cols.width() || j >= other.cols.width() {
        return Err(format!("output {j} of layer {sl} cannot stand for output {i} of layer {l}"));
    }
    let (out_node, act) = {
        let r = &program.rules[rule];
        match &r.nodes[r.output] {
            Node::Affine { terms, bias: None } if r.output + 1 == r.nodes.len() && terms.first().is_some_and(|t| t.1 == own_index) => (r.output, terms[0].0),
            other => return Err(format!("layer {l}: the MLP's output is {other:?}")),
        }
    };
    let one = Interface::uniform(1, 1, LabelKind::Unit, 0).map_err(error)?;
    let name = format!("library.l{l}.mlp.f{i}.out.from_l{sl}_{j}");
    let provenance = Provenance::derived(&[&own.provenance, &other.provenance], "column tie".into());
    let mut select = Array2::zeros((1, own.cols.width()));
    select[[0, i]] = 1.0;
    let mut scatter = Array2::zeros((other.cols.width(), 1));
    scatter[[j, 0]] = 1.0;
    let base = program.operators.len();
    for (part, rows, cols, values) in [
        ("select", one.clone(), own.cols.clone(), select),
        ("scale", one.clone(), one.clone(), Array2::from_elem((1, 1), scale)),
        ("scatter", other.cols.clone(), one.clone(), scatter),
    ] {
        program.operators.push(Arc::new(dense(format!("{name}.{part}"), rows, cols, values, provenance.clone())?));
    }
    let mut values = own.matrix();
    values.column_mut(i).fill(0.0);
    program.operators[own_index] = Arc::new(dense(own.name.clone(), own.rows.clone(), own.cols.clone(), values, own.provenance.clone())?);
    let factor = [format!("{name}.scale")];
    for owner in &mut out.artifact.owners {
        if owner.operator == own.name && owner.cols == (i..i + 1) {
            let rows = owner.rows.clone();
            owner.repoint(&other.name, rows, j..j + 1, &factor, &[]);
        }
    }
    let program = &mut out.artifact.program;
    let r = &mut program.rules[rule];
    let output = r.nodes[out_node].clone();
    r.nodes.truncate(out_node);
    let at = r.nodes.len();
    r.nodes.extend([
        Node::Affine { terms: vec![(act, base)], bias: None },
        Node::Affine { terms: vec![(at, base + 1)], bias: None },
        Node::Affine { terms: vec![(at + 1, base + 2)], bias: None },
    ]);
    let Node::Affine { mut terms, .. } = output else { return Err("the MLP's output".into()) };
    terms.push((at + 2, source_index));
    r.nodes.push(Node::Affine { terms, bias: None });
    r.output = r.nodes.len() - 1;
    program.interfaces().map_err(error)?;
    let own_group = out.groups.iter().position(|g| g.name == format!("library.l{l}.mlp.f{i}.out")).ok_or("no output group")?;
    let tied = out.groups.len();
    out.groups.push(Group { name: format!("library.l{l}.mlp.f{i}.out.tie"), cells: vec![Cells { operator: base + 1, rows: vec![0], cols: 0..1 }] });
    out.removed.push(own_group);
    out.trainable.push(base + 1);
    out.trainable.sort_unstable();
    for g in out.layers[l].functions[i].iter_mut() {
        if *g == own_group {
            *g = tied;
        }
    }
    Ok(out)
}

/// Every native block `a`'s owners stand for, `b`'s owners stand for at the same site with the
/// same values: an exact sharing moves the native parameters, it does not change them.
#[cfg(test)]
pub(crate) fn same_native_blocks(a: &Explanation, b: &Explanation) {
    assert_eq!(a.artifact.owners.len(), b.artifact.owners.len());
    for owner in &a.artifact.owners {
        let other = b
            .artifact
            .owners
            .iter()
            .find(|o| o.native == owner.native && o.native_rows == owner.native_rows && o.native_cols == owner.native_cols && o.site == owner.site)
            .expect("every native block keeps an owner at its site");
        let (x, y) = (a.artifact.native_block(owner).unwrap(), b.artifact.native_block(other).unwrap());
        assert_eq!(x.dim(), y.dim());
        let scale = x.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        assert!(x.iter().zip(y.iter()).all(|(u, v)| (u - v).abs() <= 1e-12 * scale), "{} at {} through {other:?}", owner.native, owner.site);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        library_mdl::{Posterior, explanation},
        run_check::{layer_nodes, split_sites},
    };

    #[test]
    fn a_shared_query_key_function_is_paid_once_and_keeps_identical_members_exact() {
        let dir = crate::test_support::tiny_export("library_sharing", 2);
        let imported = import_language_model(&dir, 6, 12).expect("import");
        let native = split_sites(&imported.program).expect("split");
        let layers = layer_nodes(&native, 2).expect("layers");
        let mut start = explanation(&native, &layers).expect("explanation");
        // Head 0 of layer 1 made a copy of head 0 of layer 0, so sharing must reproduce both.
        let program = &mut start.artifact.program;
        let (first, second) = (head(program, 0, 0).unwrap().unwrap(), head(program, 1, 0).unwrap().unwrap());
        for (from, to) in [(first.query, second.query), (first.key, second.key)] {
            program.operators[to] = Arc::new(dense(program.operators[to].name.clone(), program.operators[to].rows.clone(), program.operators[to].cols.clone(), program.operators[from].matrix(), Provenance::default()).unwrap());
        }
        let shared = share_query_key(&start, &[(0, 0), (1, 0)]).expect("shared");
        // Sharing keeps every native owner; the member's now read the shared maps at its own site.
        let natives = |e: &Explanation| { let mut n: Vec<(String, String)> = e.artifact.owners.iter().map(|o| (o.native.clone(), o.site.clone())).collect(); n.sort(); n };
        assert_eq!(natives(&start), natives(&shared));
        let (q, member_q) = (&start.artifact.program.operators[head(&start.artifact.program, 0, 0).unwrap().unwrap().query].name, &start.artifact.program.operators[head(&start.artifact.program, 1, 0).unwrap().unwrap().query].name);
        let native_member = start.artifact.owners.iter().find(|o| &o.operator == member_q).unwrap().native.clone();
        let now = shared.artifact.owners.iter().find(|o| o.native == native_member && o.site == "library.l1.h0").unwrap();
        assert_eq!(&now.operator, q);
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), shared.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[shared.artifact.program.output]);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs())), "identical members keep the explanation's output");
        same_native_blocks(&start, &shared);
        // The second member's planes leave with its maps; its part is one new group.
        let planes = start.layers[1].heads[0].0.len();
        assert_eq!(shared.groups.len(), start.groups.len() - planes + 1);
        assert_eq!(shared.trainable.len(), start.trainable.len() - 1);
        Posterior::new(&shared, 2 * 6 * 12).expect("every shared entry in one group");
        assert!(share_query_key(&start, &[(0, 0), (0, 1)]).is_err());
    }

    #[test]
    fn a_key_shared_by_a_key_value_group_is_never_a_member() {
        let dir = crate::test_support::tiny_export("library_sharing_grouped", 2);
        let path = dir.join("export.json");
        let mut record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        record["config"]["n_kv_heads"] = 1.into();
        for l in 0..2 {
            for name in ["attn.k_proj", "attn.v_proj"] {
                let name = format!("blocks.{l}.{name}");
                let values = std::fs::read(dir.join(format!("{name}.f64"))).unwrap();
                std::fs::write(dir.join(format!("{name}.f64")), &values[..4 * 8 * 8]).unwrap();
                record["files"][name] = serde_json::json!({"shape": [4, 8]});
            }
        }
        std::fs::write(&path, record.to_string()).unwrap();
        let imported = import_language_model(&dir, 6, 12).expect("import");
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).expect("split");
        let start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        assert!(heads(&start).unwrap().is_empty(), "every head's key is its group's");
        assert!(share_query_key(&start, &[(0, 0), (1, 0)]).is_err());
    }

    #[test]
    fn tying_an_exact_copy_keeps_the_outputs_and_charges_the_vector_once() {
        let dir = crate::test_support::tiny_export("library_tie_copy", 2);
        let imported = import_language_model(&dir, 6, 12).expect("import");
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).expect("split");
        let mut start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        // Function 3 of layer 0 writes 2.5 times the direction function 5 of layer 1 reads.
        let program = &mut start.artifact.program;
        let (out, gate) = (operator_index(program, "library.l0.mlp.out").unwrap(), operator_index(program, "library.l1.mlp.gate").unwrap());
        let mut values = program.operators[out].matrix();
        values.column_mut(3).assign(&(&program.operators[gate].matrix().row(5) * 2.5));
        program.operators[out] = Arc::new(dense(program.operators[out].name.clone(), program.operators[out].rows.clone(), program.operators[out].cols.clone(), values, Provenance::default()).unwrap());
        let tokens = 2 * 6 * 12;
        let untied = Posterior::new(&start, tokens).unwrap();
        let tied = tie(&start, &[Tie { source: (0, 3), target: (1, 5), scale: 2.5 }]).unwrap();
        // The tied output's native owner now reads the gate row at its own site.
        let owner = tied.artifact.owners.iter().find(|o| o.site == "library.l0.mlp" && o.native_cols == (3..4) && o.native.contains("down")).expect("the output's native owner");
        assert_eq!((owner.operator.as_str(), owner.rows.clone()), ("library.l1.mlp.gate", 5..6));
        assert_eq!(tied.artifact.owners.len(), start.artifact.owners.len());
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), tied.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[tied.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the tie of an exact copy keeps the outputs");
        same_native_blocks(&start, &tied);
        // The gate row is charged once, the copy's own output group nothing, and the scale once.
        let posterior = Posterior::new(&tied, tokens).unwrap();
        let own = tied.groups.iter().position(|g| g.name == "library.l0.mlp.f3.out").unwrap();
        let scale_group = tied.groups.iter().position(|g| g.name == "library.l0.mlp.f3.tie").unwrap();
        assert!(!posterior.active[own] && posterior.costs()[own] == 0.0);
        assert_eq!(tied.groups[scale_group].cells.iter().map(|c| c.rows.len() * c.cols.len()).sum::<usize>(), 1);
        let (costs, before_costs) = (posterior.costs(), untied.costs());
        for g in 0..start.groups.len() {
            if g != own {
                assert!((costs[g] - before_costs[g]).abs() <= 1e-9 * before_costs[g].abs().max(1.0), "{} costs the same", start.groups[g].name);
            }
        }
        assert_eq!(posterior.active.iter().filter(|a| **a).count(), untied.active.len());
        assert!(tie(&start, &[Tie { source: (1, 0), target: (0, 0), scale: 1.0 }]).is_err(), "a later write cannot feed an earlier read");
    }

    #[test]
    fn a_function_copied_across_layers_is_stored_once_and_keeps_the_outputs() {
        let dir = crate::test_support::tiny_export("library_function_copy", 2);
        let imported = import_language_model(&dir, 6, 12).expect("import");
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).expect("split");
        let mut start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        // Function 7 of layer 1 is 2 times function 4 of layer 0 in its read and 0.5 times in its write.
        let program = &mut start.artifact.program;
        for (part, column, scale) in [("gate", false, 2.0), ("out", true, 0.5)] {
            let (from, to) = (operator_index(program, &format!("library.l0.mlp.{part}")).unwrap(), operator_index(program, &format!("library.l1.mlp.{part}")).unwrap());
            let mut values = program.operators[to].matrix();
            if column {
                values.column_mut(7).assign(&(&program.operators[from].matrix().column(4) * scale));
            } else {
                values.row_mut(7).assign(&(&program.operators[from].matrix().row(4) * scale));
            }
            program.operators[to] = Arc::new(dense(program.operators[to].name.clone(), program.operators[to].rows.clone(), program.operators[to].cols.clone(), values, Provenance::default()).unwrap());
        }
        let row = tie_row(&start, "gate", (1, 7), RowSource::Row { layer: 0, part: "gate", function: 4 }, 2.0).unwrap();
        let tied = tie_column(&row, (1, 7), (0, 4), 0.5).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), tied.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[tied.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the tied function keeps the outputs");
        same_native_blocks(&start, &tied);
        // Its own read and write leave; two scale groups replace them; its native owners read layer 0's.
        let tokens = 2 * 6 * 12;
        let (untied, posterior) = (Posterior::new(&start, tokens).unwrap(), Posterior::new(&tied, tokens).unwrap());
        assert_eq!(posterior.active.iter().filter(|a| **a).count(), untied.active.len());
        for name in ["library.l1.mlp.f7.gate", "library.l1.mlp.f7.out"] {
            let g = tied.groups.iter().position(|g| g.name == name).unwrap();
            assert!(!posterior.active[g] && posterior.costs()[g] == 0.0, "{name} is not charged");
        }
        let owners: Vec<_> = tied.artifact.owners.iter().filter(|o| o.site == "library.l1.mlp" && (o.native_rows == (7..8) || o.native_cols == (7..8))).collect();
        assert!(owners.iter().any(|o| o.operator == "library.l0.mlp.gate" && o.rows == (4..5)));
        assert!(owners.iter().any(|o| o.operator == "library.l0.mlp.out" && o.cols == (4..5)));
    }

    #[test]
    fn alignment_undoes_a_plane_rotation_and_scale() {
        let q1 = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f64 + 1.0);
        let k1 = Array2::from_shape_fn((2, 3), |(i, j)| ((i + 2 * j) as f64).sin() + 2.0);
        let (sine, cosine) = 0.7_f64.sin_cos();
        let r = ndarray::array![[cosine, -sine], [sine, cosine]];
        let (q, k) = (r.dot(&q1) * 3.0, r.dot(&k1) / 3.0);
        let (qa, ka) = aligned(&q1, &k1, &q, &k, &[vec![0, 1]]);
        assert!((&qa - &q1).iter().chain((&ka - &k1).iter()).all(|d| d.abs() < 1e-12));
    }
}
