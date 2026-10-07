//! Shared functions of a library explanation (#2951).
//!
//! Several heads of a library ([`crate::library_mdl`]) may compute one attention pattern (VPD's
//! previous-token and induction behaviours appear in more than one head). The unit of sharing is
//! a native key-value group: the query heads that read one key map in `M` (grouped-query
//! attention), or one head where every head has its own key. A shared query–key function replaces
//! the members' query and key maps by one set `(Q_1, …, Q_m, K)` that every member reads: the
//! first member's head `j` attends with `q = Q_j x`, `k = K x`, and every other member's head `i`
//! with `q = c_i Q_{π(i)} x`, `k = K x`, where `π` assigns the member's query heads to the
//! owner's and the scalar `c_i` (starting at 1) is the head's own part. Every member keeps its
//! value map. The shared maps carry the prior groups of one key-value group (a group per rotary
//! plane, as in the library), paid once in `KL(q ‖ p)`; each `c_i` is one group. Members sit in
//! different layers, so each layer's read of `Q x` stays its own variable.
//! The move is accepted only if the code length `F` of the re-converged fit falls.
//!
//! # Candidates
//!
//! A head's scores are `qᵀ R k` with `R` the rotary turn between the two positions; a rotation and
//! a scaling of a plane, `(s R q, R k / s)`, leave them unchanged, for every query head of a key
//! at once. Where the heads norm their queries and keys (Qwen3), the scores are
//! `(γ_q ⊙ N(Q x))ᵀ R (γ_k ⊙ N(K y))` with `N` the RMS norm of the whole head and `γ` its gains: a
//! plane's scaling changes `N` (with `q = k = (1, 0, 1, 0)`, doubling the first plane's query and
//! halving its key keeps `q·k = 2` and moves the normed score from 4 to 3.2), and a rotation of a
//! plane keeps the scores only when it commutes with the gains there. The gauges of normed heads
//! are therefore rotations alone: any rotation of a plane whose two coordinates have equal gains
//! in every head involved, else the plane's identity or its half turn (`Symmetry`). The groups to
//! share, and the assignment of their query heads, are found by the learned mixture prior over the
//! groups' query–key maps (`library_mixture`).
//!
//! # The start
//!
//! The shared maps start at the members' mean after each member is brought to the first member's
//! gauge, plane by plane: the rotation `R` that minimizes `Σ_i ‖R Q_{m,i} − Q_{1,π(i)}‖² +
//! ‖R K_m − K_1‖²` over a plane's two rows (the orthogonal Procrustes solution in two dimensions;
//! a sign on a coordinate no rotary plane holds), and the scale `s` with `(s R Q_m, R K_m / s)` in
//! the first member's ratio of query to key norm. Both leave the member's scores unchanged, so the
//! start is close to the current fit.
//!
//! # Value maps
//!
//! `M` keeps each head's output projection `O_h`, so a group's value–output maps are `O_i V` over
//! its query heads `i`. Another group's are the same maps when `V = T V_s` with
//! `(π, T) = argmin Σ_i ‖O_{t,i} T − O_{s,π(i)}‖²` over the assignments `π` of query heads and the
//! transports `T` jointly (`transports`). Where the residual is zero this is the symmetry
//! `V → R V`, `O → O R⁻¹` with `O` fixed by `M`; otherwise the shared map is an approximation,
//! which the code length `F` of the re-converged fit accepts or rejects. A shared value map stores
//! `V_s` once and the target's heads read `c T V_s`, `T` fixed and `c` one scalar
//! ([`share_value`]).
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

/// The reference variance (`Explanation::reference`) of a scale a tie or share adds: the square of
/// the identity scale 1 it stands beside.
const SCALE_REFERENCE: f64 = 1.0;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A head's block in the explanation's program: its rule's index, its query, key and value
/// operators, and its rotary planes.
pub(crate) struct Head {
    rule: usize,
    pub(crate) query: usize,
    pub(crate) key: usize,
    pub(crate) value: usize,
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
    Ok(Some(Head { rule, query: map(1)?, key: map(2)?, value: map(3)?, rotary }))
}

/// A key-value group of one layer (`library_mdl::explanation`): the query heads that read one key
/// map and one value map, in head order, and those maps. Grouped-query attention makes a group of
/// several heads; otherwise every head is a group of its own. A group whose key (value) map is
/// not its own (`library.l{layer}.kv{group}.k`) reads another group's in a sharing.
pub(crate) struct KeyValue {
    pub(crate) heads: Vec<(usize, Head)>,
    pub(crate) key: usize,
    pub(crate) value: usize,
    pub(crate) rotary: Option<Rotary>,
    pub(crate) own_key: bool,
    pub(crate) own_value: bool,
}

impl KeyValue {
    /// Its query heads' query operators, in head order.
    pub(crate) fn queries(&self) -> Vec<usize> {
        self.heads.iter().map(|(_, h)| h.query).collect()
    }
}

/// Every key-value group of the explanation by `(layer, group)`, the groups of a layer numbered in
/// the order of their first heads as `library_mdl::explanation` numbers them.
pub(crate) fn key_values(explanation: &Explanation) -> Result<BTreeMap<(usize, usize), KeyValue>, String> {
    let program = &explanation.artifact.program;
    let mut out = BTreeMap::new();
    for l in 0..explanation.layers.len() {
        let mut groups: Vec<KeyValue> = Vec::new();
        for h in 0.. {
            let Some(found) = head(program, l, h)? else { break };
            match groups.iter_mut().find(|g| g.key == found.key) {
                Some(group) if group.value == found.value && group.rotary == found.rotary => group.heads.push((h, found)),
                Some(_) => return Err(format!("head {l}.{h}: the query heads of one key read different values or planes")),
                None => groups.push(KeyValue { key: found.key, value: found.value, rotary: found.rotary, own_key: false, own_value: false, heads: vec![(h, found)] }),
            }
        }
        for (g, mut group) in groups.into_iter().enumerate() {
            group.own_key = program.operators[group.key].name == format!("library.l{l}.kv{g}.k");
            group.own_value = program.operators[group.value].name == format!("library.l{l}.kv{g}.v");
            out.insert((l, g), group);
        }
    }
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

/// The gauges that leave the scores of the heads of some key-value groups unchanged (module note):
/// whether a plane may be scaled (heads without query and key norms), and per plane whether any of
/// its rotations may be taken (else the identity or the half turn, which commute with every gain).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Symmetry {
    pub(crate) scale: bool,
    pub(crate) turn: Vec<bool>,
}

impl Symmetry {
    /// The gauges both `self` and `other` allow.
    #[must_use]
    pub(crate) fn and(&self, other: &Self) -> Self {
        Self { scale: self.scale && other.scale, turn: self.turn.iter().zip(&other.turn).map(|(a, b)| *a && *b).collect() }
    }
}

/// The gain operator of a head's normed query or key at node `n` of `rule` (through the scale a
/// shared function's member multiplies its query by), or none where it is not normed.
fn head_gain(rule: &crate::operator_program::Rule, n: usize) -> Option<usize> {
    match rule.nodes.get(n)? {
        Node::Hadamard { left, .. } => head_gain(rule, *left),
        Node::Affine { terms, bias: None } if terms.len() == 1 && matches!(rule.nodes.get(terms[0].0), Some(Node::RmsNorm { .. })) => Some(terms[0].1),
        _ => None,
    }
}

/// The gauges of the heads of `groups` with planes `planes` ([`Symmetry`]).
pub(crate) fn symmetry(program: &OperatorProgram, groups: &[&KeyValue], planes: &[Vec<usize>]) -> Result<Symmetry, String> {
    let mut gains = Vec::new();
    let mut normed = false;
    for group in groups {
        for (_, head) in &group.heads {
            let rule = &program.rules[head.rule];
            let Some(Node::Attend { query, key, .. }) = rule.nodes.get(rule.output) else {
                return Err(format!("{}: the output is not an attention node", rule.name));
            };
            for gain in [head_gain(rule, *query), head_gain(rule, *key)] {
                if let Some(op) = gain {
                    normed = true;
                    gains.push(program.operators[op].diagonal().ok_or_else(|| format!("{}: a head norm's gain is not diagonal", program.operators[op].name))?);
                }
            }
        }
    }
    let turn = planes.iter().map(|rows| rows.len() != 2 || gains.iter().all(|g| g[rows[0]] == g[rows[1]])).collect();
    Ok(Symmetry { scale: !normed, turn })
}

/// Per plane of `planes`, the rotation (on one coordinate, the sign) and the scale that bring a
/// group's maps `(q, k)`, its query maps in turn and its key map, to the gauge of `(q1, k1)`
/// (module note), among the gauges `symmetry` allows: a plane whose rotations are not all allowed
/// takes the identity or the half turn, and the scale is 1 where scaling is not allowed.
pub(crate) fn gauge(q1: &[&Array2<f64>], k1: &Array2<f64>, q: &[&Array2<f64>], k: &Array2<f64>, planes: &[Vec<usize>], symmetry: &Symmetry) -> Vec<(Array2<f64>, f64)> {
    planes
        .iter()
        .zip(&symmetry.turn)
        .map(|(rows, &turning)| {
            let pick = |m: &Array2<f64>| m.select(ndarray::Axis(0), rows);
            let (a_q, b_q): (Vec<Array2<f64>>, Vec<Array2<f64>>) = (q1.iter().map(|m| pick(m)).collect(), q.iter().map(|m| pick(m)).collect());
            let (a_k, b_k) = (pick(k1), pick(k));
            // The rotation (or, on one coordinate, the sign) nearest the first member's rows.
            let mut m = a_k.dot(&b_k.t());
            for (a, b) in a_q.iter().zip(&b_q) {
                m += &a.dot(&b.t());
            }
            let rotation = if rows.len() == 2 && turning {
                let angle = (m[[1, 0]] - m[[0, 1]]).atan2(m[[0, 0]] + m[[1, 1]]);
                let (sine, cosine) = angle.sin_cos();
                ndarray::array![[cosine, -sine], [sine, cosine]]
            } else if rows.len() == 2 {
                // The identity or the half turn, whichever brings the maps closer.
                let sign = if m[[0, 0]] + m[[1, 1]] < 0.0 { -1.0 } else { 1.0 };
                ndarray::array![[sign, 0.0], [0.0, sign]]
            } else {
                ndarray::array![[if m[[0, 0]] < 0.0 { -1.0 } else { 1.0 }]]
            };
            // A rotation keeps norms: the scale puts the query maps' norm against the key's in the
            // first member's ratio.
            let joint = |ms: &[Array2<f64>]| ms.iter().map(|m| m.iter().map(|v| v * v).sum::<f64>()).sum::<f64>().sqrt();
            let (nq, nk, n1q, n1k) = (joint(&b_q), norm(&b_k), joint(&a_q), norm(&a_k));
            let scale = if symmetry.scale && nq > 0.0 && nk > 0.0 && n1q > 0.0 && n1k > 0.0 { (n1q * nk / (nq * n1k)).sqrt() } else { 1.0 };
            (rotation, scale)
        })
        .collect()
}

/// `m` moved by `gauge` plane by plane, `s R m` for a query map and `R m / s` for a key map; with
/// `transpose`, the transpose of that map (a cotangent of the result back to `m`).
pub(crate) fn turn(m: &Array2<f64>, planes: &[Vec<usize>], gauge: &[(Array2<f64>, f64)], query: bool, transpose: bool) -> Array2<f64> {
    let mut out = m.clone();
    for (rows, (rotation, scale)) in planes.iter().zip(gauge) {
        let rotation = if transpose { rotation.t().to_owned() } else { rotation.clone() };
        let turned = rotation.dot(&m.select(ndarray::Axis(0), rows));
        let factor = if query { *scale } else { 1.0 / *scale };
        for (i, &row) in rows.iter().enumerate() {
            out.row_mut(row).assign(&(&turned.row(i) * factor));
        }
    }
    out
}

/// A dense library operator holding `values`.
fn dense(name: String, rows: Interface, cols: Interface, values: Array2<f64>, provenance: Provenance) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, provenance).map_err(error)
}

/// A member of a shared query–key function: key-value group `group` of layer `layer`, whose query
/// head `i` (in head order) reads the shared query map of the first member's head `queries[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub layer: usize,
    pub group: usize,
    pub queries: Vec<usize>,
}

/// `explanation` with the key-value groups `members` (at most one per layer, the first the shared
/// function's owner, its `queries` in order) reading one shared query–key function (module note),
/// at the explanation's operator values.
pub fn share_query_key(explanation: &Explanation, members: &[Member]) -> Result<Explanation, String> {
    if members.len() < 2 {
        return Err("a shared function needs two members".into());
    }
    if (1..members.len()).any(|i| members[..i].iter().any(|m| m.layer == members[i].layer)) {
        return Err("the members of a shared query-key function lie in different layers".into());
    }
    let all = key_values(explanation)?;
    let found: Vec<&KeyValue> = members.iter().map(|m| all.get(&(m.layer, m.group)).ok_or_else(|| format!("no key-value group {}.{}", m.layer, m.group))).collect::<Result<_, _>>()?;
    let owner = found[0];
    let width = owner.heads.len();
    for (m, group) in members.iter().zip(&found) {
        let mut seen = m.queries.clone();
        seen.sort_unstable();
        if !group.own_key || group.heads.len() != width || group.rotary != owner.rotary || seen != (0..width).collect::<Vec<_>>() {
            return Err(format!("key-value group {}.{} cannot share the query-key function of {}.{}", m.layer, m.group, members[0].layer, members[0].group));
        }
    }
    if members[0].queries != (0..width).collect::<Vec<_>>() {
        return Err("the owner's query heads read their own maps".into());
    }
    let mut artifact = explanation.artifact.clone();
    let program = &artifact.program;
    let (owner_queries, owner_key) = (owner.queries(), owner.key);
    let name = |op: usize| program.operators[op].name.clone();
    // Each member keeps its native owners, now read through the shared maps; a member's query
    // through its head's scale. Where `M` norms each head's query (Qwen3) a scale would act after
    // the norm, as a gain does, which no native query map can carry: the member reads the shared
    // query map unscaled (`c ≡ 1`), and the query map itself is the owner's.
    let mut moves: Vec<(String, String, Option<String>)> = Vec::new();
    for (m, group) in members.iter().zip(&found).skip(1) {
        for ((h, head), &j) in group.heads.iter().zip(&m.queries) {
            let rule = &program.rules[head.rule];
            let normed = !matches!(rule.nodes[rule.output], Node::Attend { query: 1, .. });
            let scale = (!normed).then(|| format!("library.l{}.h{h}.q_shared_scale", m.layer));
            moves.push((name(head.query), name(owner_queries[j]), scale));
        }
        moves.push((name(group.key), name(owner_key), None));
    }
    for owner in &mut artifact.owners {
        if let Some((_, to, scale)) = moves.iter().find(|(from, _, _)| *from == owner.operator) {
            let (rows, cols) = (owner.rows.clone(), owner.cols.clone());
            owner.repoint(to, rows, cols, scale.as_slice(), &[]);
        }
    }
    let program = &mut artifact.program;
    let values = |op: usize| program.operators[op].matrix();
    let (q1, k1): (Vec<Array2<f64>>, Array2<f64>) = (owner_queries.iter().map(|&q| values(q)).collect(), values(owner_key));
    let planes = planes(k1.nrows(), owner.rotary);
    let (mut q_sum, mut k_sum) = (q1.clone(), k1.clone());
    for (m, group) in members.iter().zip(&found).skip(1) {
        let (q, k): (Vec<Array2<f64>>, Array2<f64>) = (group.queries().iter().map(|&q| values(q)).collect(), values(group.key));
        if q.iter().zip(&q1).any(|(a, b)| a.dim() != b.dim()) || k.dim() != k1.dim() {
            return Err("the members' query and key maps differ in shape".into());
        }
        // Member head `i` faces the owner's head `queries[i]`.
        let facing: Vec<&Array2<f64>> = m.queries.iter().map(|&j| &q1[j]).collect();
        let allowed = symmetry(program, &[owner, *group], &planes)?;
        let gauge = gauge(&facing, &k1, &q.iter().collect::<Vec<_>>(), &k, &planes, &allowed);
        for (i, &j) in m.queries.iter().enumerate() {
            q_sum[j] += &turn(&q[i], &planes, &gauge, true, false);
        }
        k_sum += &turn(&k, &planes, &gauge, false, false);
    }
    let n = members.len() as f64;
    let provenance = Provenance::derived(&[&program.operators[owner_key].provenance], "shared query-key function".into());
    for (op, values) in owner_queries.iter().zip(q_sum).chain([(&owner_key, k_sum)]) {
        let source = &program.operators[*op];
        program.operators[*op] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values / n, provenance.clone())?);
    }
    // Every other member's query head reads the shared maps, its query scaled by its own `c` (a
    // 1 × 1 operator, broadcast over the query's coordinates by a fixed column of ones); a head
    // that norms its query reads them unscaled (above).
    let coordinates = program.operators[owner_key].rows.clone();
    let one = Interface::uniform(1, 1, LabelKind::Unit, 0).map_err(error)?;
    let mut retired = Vec::new();
    let mut parts = Vec::new();
    for (m, group) in members.iter().zip(&found).skip(1) {
        retired.push(group.key);
        for ((h, head), &j) in group.heads.iter().zip(&m.queries) {
            retired.push(head.query);
            let normed = !matches!(program.rules[head.rule].nodes[program.rules[head.rule].output], Node::Attend { query: 1, .. });
            if normed {
                let rule = &mut program.rules[head.rule];
                rule.nodes[1] = Node::Affine { terms: vec![(0, owner_queries[j])], bias: None };
                rule.nodes[2] = Node::Affine { terms: vec![(0, owner_key)], bias: None };
                continue;
            }
            let scale = program.operators.len();
            let name = format!("library.l{}.h{h}.q_shared_scale", m.layer);
            program.operators.push(Arc::new(dense(name.clone(), one.clone(), Interface::constant(), Array2::ones((1, 1)), provenance.clone())?));
            program.operators.push(Arc::new(dense(format!("library.l{}.h{h}.q_shared_ones", m.layer), coordinates.clone(), one.clone(), Array2::ones((coordinates.width(), 1)), provenance.clone())?));
            parts.push((name, scale));
            let rule = &mut program.rules[head.rule];
            rule.nodes[1] = Node::Affine { terms: vec![(0, owner_queries[j])], bias: None };
            rule.nodes[2] = Node::Affine { terms: vec![(0, owner_key)], bias: None };
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
    }
    program.interfaces().map_err(error)?;
    // The retired maps leave the library with their groups (every plane group of a member holds
    // its key's rows); each part is one group.
    let mut index = BTreeMap::new();
    let (mut groups, mut reference) = (Vec::new(), Vec::new());
    for (g, group) in explanation.groups.iter().enumerate() {
        if group.cells.iter().any(|c| retired.contains(&c.operator)) {
            continue;
        }
        index.insert(g, groups.len());
        groups.push(group.clone());
        reference.push(explanation.reference[g]);
    }
    for (name, scale) in &parts {
        groups.push(Group { name: name.clone(), cells: vec![Cells { operator: *scale, rows: vec![0], cols: 0..1 }] });
        reference.push(SCALE_REFERENCE);
    }
    let mut trainable: Vec<usize> = explanation.trainable.iter().copied().filter(|op| !retired.contains(op)).chain(parts.iter().map(|p| p.1)).collect();
    trainable.sort_unstable();
    let owner_planes: Vec<usize> = explanation.layers[members[0].layer].heads[owner.heads[0].0].0.iter().filter_map(|g| index.get(g).copied()).collect();
    let member_heads: Vec<(usize, usize)> = members.iter().zip(&found).skip(1).flat_map(|(m, group)| group.heads.iter().map(move |(h, _)| (m.layer, *h))).collect();
    let mut layers = explanation.layers.clone();
    for (l, layer) in layers.iter_mut().enumerate() {
        for (h, (planes, values)) in layer.heads.iter_mut().enumerate() {
            *planes = if member_heads.contains(&(l, h)) { owner_planes.clone() } else { planes.iter().filter_map(|g| index.get(g).copied()).collect() };
            *values = values.iter().filter_map(|g| index.get(g).copied()).collect();
        }
        for function in &mut layer.functions {
            *function = function.iter().filter_map(|g| index.get(g).copied()).collect();
        }
    }
    let removed = explanation.removed.iter().filter_map(|g| index.get(g).copied()).collect();
    Ok(Explanation { artifact, trainable, groups, layers, removed, fixed_nats: explanation.fixed_nats, reference, reads: explanation.reads.clone(), shares: explanation.shares.clone() })
}

/// The value transport from one key-value group to another (module note): `T`, the source query
/// head each target query head faces, and the share of the source projections' squared norm the
/// transport leaves, `Σ_i ‖O_{t,i} T − O_{s,π(i)}‖² / Σ_j ‖O_{s,j}‖²`: zero for a symmetry, else
/// the transport is an approximation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Transport {
    pub(crate) matrix: Array2<f64>,
    pub(crate) assignment: Vec<usize>,
    pub(crate) residual: f64,
}

/// The assignment `π` maximizing `‖Σ_i M[i][π(i)]‖²`, by depth-first branch and bound from the
/// assignment `start`: a partial assignment with sum `S` is dropped when
/// `(‖S‖ + Σ_{i unassigned} max_{j free} ‖M[i][j]‖)²`, which bounds every completion by the
/// triangle inequality, does not exceed the best found. Exact for every size.
fn best_assignment(m: &[Vec<Array2<f64>>], start: Vec<usize>) -> Vec<usize> {
    let h = m.len();
    let value = |assignment: &[usize]| -> f64 {
        let mut sum = m[0][assignment[0]].clone();
        for (i, &j) in assignment.iter().enumerate().skip(1) {
            sum += &m[i][j];
        }
        sum.iter().map(|v| v * v).sum()
    };
    let norms: Vec<Vec<f64>> = m.iter().map(|row| row.iter().map(norm).collect()).collect();
    let mut best = (value(&start), start);
    let mut partial = Vec::with_capacity(h);
    let mut used = vec![false; h];
    fn search(m: &[Vec<Array2<f64>>], norms: &[Vec<f64>], sum: Option<&Array2<f64>>, partial: &mut Vec<usize>, used: &mut [bool], best: &mut (f64, Vec<usize>)) {
        let (h, i) = (m.len(), partial.len());
        let current = sum.map_or(0.0, norm);
        if i == h {
            if current * current > best.0 {
                *best = (current * current, partial.clone());
            }
            return;
        }
        let bound = current + (i..h).map(|r| (0..h).filter(|j| !used[*j]).map(|j| norms[r][j]).fold(0.0, f64::max)).sum::<f64>();
        if bound * bound <= best.0 {
            return;
        }
        let mut order: Vec<(f64, usize, Array2<f64>)> = (0..h)
            .filter(|j| !used[*j])
            .map(|j| {
                let next = match sum {
                    Some(s) => s + &m[i][j],
                    None => m[i][j].clone(),
                };
                (norm(&next), j, next)
            })
            .collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, j, next) in order {
            used[j] = true;
            partial.push(j);
            search(m, norms, Some(&next), partial, used, best);
            partial.pop();
            used[j] = false;
        }
    }
    search(m, &norms, None, &mut partial, &mut used, &mut best);
    best.1
}

/// Each query head's native output projection `O_h` (`blocks.{layer}.o{h}`, `d × width`), which the
/// library keeps, for key-value group `group` of layer `layer`.
fn output_projections(explanation: &Explanation, groups: &BTreeMap<(usize, usize), KeyValue>, (layer, group): (usize, usize)) -> Result<Vec<Array2<f64>>, String> {
    let found = groups.get(&(layer, group)).ok_or_else(|| format!("no key-value group {layer}.{group}"))?;
    let program = &explanation.artifact.program;
    found.heads.iter().map(|(h, _)| Ok(program.operators[operator_index(program, &format!("blocks.{layer}.o{h}"))?].matrix())).collect()
}

/// The value transports of key-value group `target` from each group of `sources` (module note):
/// `(π, T)` minimize `Σ_i ‖O_{t,i} T − O_{s,π(i)}‖²` jointly. For an assignment `π` the least-squares
/// transport is `T = A⁺ B_π`, `A` and `B_π` the stacked projections, and its residual is
/// `‖B_π‖² − ‖Q_Aᵀ B_π‖²` with `Q_A` an orthonormal basis of `A`'s columns; `‖B_π‖²` does not depend on
/// `π`, so the best assignment maximizes `‖Σ_i Q_{A,i}ᵀ O_{s,π(i)}‖²` (`Q_{A,i}` the rows of `Q_A`
/// for target head `i`), found exactly by branch and bound ([`best_assignment`]) from the
/// assignment that minimizes each source projection's residual outside each target projection's
/// column space.
pub(crate) fn transports(explanation: &Explanation, target: (usize, usize), sources: &[(usize, usize)]) -> Result<Vec<Transport>, String> {
    use rayon::prelude::*;
    let groups = key_values(explanation)?;
    let own = output_projections(explanation, &groups, target)?;
    let stacked = |ms: &[&Array2<f64>]| ndarray::concatenate(ndarray::Axis(0), &ms.iter().map(|m| m.view()).collect::<Vec<_>>()).map_err(error);
    let a = stacked(&own.iter().collect::<Vec<_>>())?;
    let inverse = gam_linalg::decompose::pseudo_inverse(a.view()).map_err(error)?;
    // An orthonormal basis of `A`'s columns, split into the target heads' row blocks.
    let basis = {
        let d = gam_linalg::decompose::svd(a.view(), false).map_err(error)?;
        let resolved: Vec<usize> = (0..d.singular_values.len()).filter(|&i| d.singular_values[i] > d.band).collect();
        d.u.select(ndarray::Axis(1), &resolved)
    };
    let height = own.first().map_or(0, Array2::nrows);
    let blocks: Vec<Array2<f64>> = (0..own.len()).map(|i| basis.slice(ndarray::s![i * height..(i + 1) * height, ..]).to_owned()).collect();
    // Each target projection's column space, orthonormal.
    let bases = own
        .iter()
        .map(|o| -> Result<Array2<f64>, String> {
            let d = gam_linalg::decompose::svd(o.view(), false).map_err(error)?;
            let resolved: Vec<usize> = (0..d.singular_values.len()).filter(|&i| d.singular_values[i] > d.band).collect();
            Ok(d.u.select(ndarray::Axis(1), &resolved))
        })
        .collect::<Result<Vec<_>, _>>()?;
    sources
        .par_iter()
        .map(|&source| {
            let theirs = output_projections(explanation, &groups, source)?;
            if theirs.len() != own.len() || theirs.iter().zip(&own).any(|(s, t)| s.nrows() != t.nrows()) {
                return Err(format!("key-value group {source:?} cannot stand for {target:?}"));
            }
            let assignment = if own.len() == 1 {
                vec![0]
            } else {
                let cost = Array2::from_shape_fn((own.len(), theirs.len()), |(i, j)| {
                    let inside = bases[i].t().dot(&theirs[j]);
                    theirs[j].iter().map(|v| v * v).sum::<f64>() - inside.iter().map(|v| v * v).sum::<f64>()
                });
                let start = crate::library_bodies::hungarian(&cost)?;
                let projected: Vec<Vec<Array2<f64>>> = blocks.iter().map(|q| theirs.iter().map(|o| q.t().dot(o)).collect()).collect();
                best_assignment(&projected, start)
            };
            let b = stacked(&assignment.iter().map(|&j| &theirs[j]).collect::<Vec<_>>())?;
            let matrix = inverse.dot(&b);
            let total: f64 = b.iter().map(|v| v * v).sum();
            let left: f64 = (a.dot(&matrix) - &b).iter().map(|v| v * v).sum();
            Ok(Transport { matrix, assignment, residual: if total > 0.0 { left / total } else { 0.0 } })
        })
        .collect()
}

/// `explanation` with the value map of key-value group `target` made `scale T V_s`, `V_s` the value
/// map of group `source` of an earlier layer and `T` the transport between them (`transports`):
/// every query head of the target reads `V_s`, moves it by `T` (fixed, from `M`'s output
/// projections) and scales it by `scale` (a 1 × 1 operator, one prior group). The target's own
/// value map leaves the library with its groups; `V_s` is stored once, its gradient summing both
/// uses; each native owner of the target's value reads `V_s` through `scale · T`.
pub fn share_value(explanation: &Explanation, target: (usize, usize), source: (usize, usize), scale: f64) -> Result<Explanation, String> {
    if source.0 >= target.0 {
        return Err(format!("the value map of layer {} cannot stand for an earlier layer {}'s", source.0, target.0));
    }
    let groups = key_values(explanation)?;
    let (own, other) = (groups.get(&target).ok_or("no target group")?, groups.get(&source).ok_or("no source group")?);
    if !own.own_value || !other.own_value {
        return Err("a shared value map is a group's own".into());
    }
    let transport = transports(explanation, target, &[source])?.pop().ok_or("no transport")?;
    let mut out = explanation.clone();
    let program = &mut out.artifact.program;
    let (mine, theirs) = (program.operators[own.value].clone(), program.operators[other.value].clone());
    if mine.cols != theirs.cols {
        return Err("the two value maps read different streams".into());
    }
    let name = format!("library.l{}.kv{}.v_from_l{}_kv{}", target.0, target.1, source.0, source.1);
    let provenance = Provenance::derived(&[&mine.provenance, &theirs.provenance], "shared value map".into());
    let one = Interface::uniform(1, 1, LabelKind::Unit, 0).map_err(error)?;
    let base = program.operators.len();
    program.operators.push(Arc::new(dense(format!("{name}.transport"), mine.rows.clone(), theirs.rows.clone(), transport.matrix, provenance.clone())?));
    program.operators.push(Arc::new(dense(format!("{name}.scale"), one.clone(), Interface::constant(), Array2::from_elem((1, 1), scale), provenance.clone())?));
    program.operators.push(Arc::new(dense(format!("{name}.ones"), mine.rows.clone(), one, Array2::ones((mine.rows.width(), 1)), provenance)?));
    for (_, head) in &own.heads {
        let rule = &mut program.rules[head.rule];
        rule.nodes[3] = Node::Affine { terms: vec![(0, other.value)], bias: None };
        let output = rule.output;
        rule.nodes.splice(
            output..output,
            [
                Node::Affine { terms: vec![(3, base)], bias: None },
                Node::Constant { operator: base + 1 },
                Node::Affine { terms: vec![(output + 1, base + 2)], bias: None },
                Node::Hadamard { left: output, right: output + 2 },
            ],
        );
        rule.output = output + 4;
        let Node::Attend { value, .. } = &mut rule.nodes[output + 4] else {
            return Err(format!("{}: the output is not an attention node", rule.name));
        };
        *value = output + 3;
    }
    program.interfaces().map_err(error)?;
    let factors = [format!("{name}.scale"), format!("{name}.transport")];
    for owner in &mut out.artifact.owners {
        if owner.operator == mine.name {
            owner.repoint(&theirs.name, 0..theirs.rows.width(), owner.cols.clone(), &factors, &[]);
        }
    }
    // The target's value map leaves with its groups; its heads' value groups are the source's.
    let retired = own.value;
    let mut index = BTreeMap::new();
    let (mut kept, mut reference) = (Vec::new(), Vec::new());
    for (g, group) in out.groups.iter().enumerate() {
        if group.cells.iter().any(|c| c.operator == retired) {
            continue;
        }
        index.insert(g, kept.len());
        kept.push(group.clone());
        reference.push(out.reference[g]);
    }
    kept.push(Group { name: format!("{name}.scale"), cells: vec![Cells { operator: base + 1, rows: vec![0], cols: 0..1 }] });
    reference.push(SCALE_REFERENCE);
    let source_values: Vec<usize> = out.layers[source.0].heads[other.heads[0].0].1.iter().filter_map(|g| index.get(g).copied()).collect();
    let members: Vec<usize> = own.heads.iter().map(|(h, _)| *h).collect();
    for (l, layer) in out.layers.iter_mut().enumerate() {
        for (h, (planes, values)) in layer.heads.iter_mut().enumerate() {
            *planes = planes.iter().filter_map(|g| index.get(g).copied()).collect();
            *values = if l == target.0 && members.contains(&h) { source_values.clone() } else { values.iter().filter_map(|g| index.get(g).copied()).collect() };
        }
        for function in &mut layer.functions {
            *function = function.iter().filter_map(|g| index.get(g).copied()).collect();
        }
    }
    out.removed = out.removed.iter().filter_map(|g| index.get(g).copied()).collect();
    out.groups = kept;
    out.reference = reference;
    out.trainable = out.trainable.iter().copied().filter(|op| *op != retired).chain([base + 1]).collect();
    out.trainable.sort_unstable();
    Ok(out)
}

/// `explanation` with the native block `owner` stands for edited by `delta` (its shape) at
/// `owner`'s site alone: every other site, and every other use of a block the site shares, computes
/// as before. An MLP site's own operator for the part stays applied there whether its block is its
/// own, tied or a tie's source, so the edit joins the node applying it as a term of its own (a
/// bias, which no sharing reads, is edited in place); a head's query or key edit joins its map's
/// node, divided by the site's scalar factors, which act after it; a value edit is added to the
/// head's value after its factors. A block inside a rule body (`library_bodies`) is the bodies' to
/// translate.
pub fn edit_native(explanation: &Explanation, owner: &crate::artifact::Owner, delta: &Array2<f64>) -> Result<Explanation, String> {
    if delta.dim() != (owner.native_rows.len(), owner.native_cols.len()) {
        return Err(format!("an edit of {} is {:?}, not {:?}", owner.native, (owner.native_rows.len(), owner.native_cols.len()), delta.dim()));
    }
    if owner.body != owner.site {
        return Err(format!("{} is computed inside the body {}", owner.native, owner.body));
    }
    let mut out = explanation.clone();
    let program = &mut out.artifact.program;
    let rule = program.rules.iter().position(|r| r.name == owner.site).ok_or_else(|| format!("no site {}", owner.site))?;
    let provenance = Provenance::derived(&[], format!("edit of {}", owner.native));
    // The edit as an operator from `cols` to `rows`, `values` at the native block.
    let edit = |program: &mut OperatorProgram, rows: Interface, cols: Interface, values: &Array2<f64>| -> Result<usize, String> {
        let mut full = Array2::zeros((rows.width(), cols.width()));
        full.slice_mut(ndarray::s![owner.native_rows.clone(), owner.native_cols.clone()]).assign(values);
        let name = format!("edit.{}.{}.{}", owner.site, owner.role, program.operators.len());
        program.operators.push(Arc::new(dense(name, rows, cols, full, provenance.clone())?));
        Ok(program.operators.len() - 1)
    };
    let shape = |program: &OperatorProgram, op: usize| (program.operators[op].rows.clone(), program.operators[op].cols.clone());
    match owner.role.as_str() {
        "gate_bias" | "up_bias" => {
            let index = operator_index(program, &format!("{}.{}", owner.site, owner.role))?;
            let source = program.operators[index].clone();
            let mut values = source.matrix();
            let mut block = values.slice_mut(ndarray::s![owner.native_rows.clone(), owner.native_cols.clone()]);
            block += delta;
            program.operators[index] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, source.provenance.clone())?);
        }
        "gate" | "up" | "out" => {
            let own = operator_index(program, &format!("{}.{}", owner.site, owner.role))?;
            let (rows, cols) = shape(program, own);
            let added = edit(program, rows, cols, delta)?;
            let mut applied = false;
            for node in &mut program.rules[rule].nodes {
                if let Node::Affine { terms, .. } = node
                    && let Some(input) = terms.iter().find(|t| t.1 == own).map(|t| t.0)
                {
                    terms.push((input, added));
                    applied = true;
                    break;
                }
            }
            if !applied {
                return Err(format!("{}: no node applies its {}", owner.site, owner.role));
            }
        }
        "q" | "k" => {
            let at = if owner.role == "q" { 1 } else { 2 };
            let Node::Affine { terms, .. } = &program.rules[rule].nodes[at] else {
                return Err(format!("{}: node {at} is not its {} map", owner.site, owner.role));
            };
            let like = terms.first().ok_or("an empty map")?.1;
            // The site's scalar factors act after the map.
            let mut factor = 1.0;
            for name in owner.left.iter().chain(&owner.right) {
                let value = program.operators[operator_index(program, name)?].matrix();
                if value.dim() != (1, 1) || value[[0, 0]] == 0.0 {
                    return Err(format!("{}: the factor {name} is not a nonzero scalar", owner.site));
                }
                factor *= value[[0, 0]];
            }
            let (rows, cols) = shape(program, like);
            let added = edit(program, rows, cols, &(delta / factor))?;
            if let Node::Affine { terms, .. } = &mut program.rules[rule].nodes[at] {
                terms.push((0, added));
            }
        }
        "v" => {
            let Node::Affine { terms, .. } = &program.rules[rule].nodes[3] else {
                return Err(format!("{}: node 3 is not its value map", owner.site));
            };
            let like = terms.first().ok_or("an empty map")?.1;
            // The head's value coordinates: the outermost matrix factor's rows (a shared value
            // map's transport), else the map's own.
            let outer = owner.left.iter().map(|name| operator_index(program, name)).collect::<Result<Vec<_>, _>>()?.into_iter().find(|&op| program.operators[op].rows.width() * program.operators[op].cols.width() > 1);
            let rows = shape(program, outer.unwrap_or(like)).0;
            let added = edit(program, rows.clone(), shape(program, like).1, delta)?;
            program.operators.push(Arc::new(Operator::identity(format!("edit.{}.v.identity.{}", owner.site, program.operators.len()), rows)));
            let identity = program.operators.len() - 1;
            let body = &mut program.rules[rule];
            let output = body.output;
            let Node::Attend { value, .. } = body.nodes[output] else {
                return Err(format!("{}: the output is not an attention node", owner.site));
            };
            body.nodes.insert(output, Node::Affine { terms: vec![(value, identity), (0, added)], bias: None });
            body.output = output + 1;
            if let Node::Attend { value, .. } = &mut body.nodes[output + 1] {
                *value = output;
            }
        }
        other => return Err(format!("{}: no edit of a part {other:?}", owner.site)),
    }
    program.interfaces().map_err(error)?;
    Ok(out)
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
            out.reference.push(SCALE_REFERENCE);
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
    out.reference.push(SCALE_REFERENCE);
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
    out.reference.push(SCALE_REFERENCE);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{grouped, same_native_blocks};
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
        let member = |layer: usize, group: usize| Member { layer, group, queries: vec![0] };
        let shared = share_query_key(&start, &[member(0, 0), member(1, 0)]).expect("shared");
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
        assert!(share_query_key(&start, &[member(0, 0), member(0, 1)]).is_err());
    }

    #[test]
    fn a_key_value_group_is_shared_whole_with_its_query_heads_assigned() {
        let imported = grouped("library_sharing_grouped");
        let native = split_sites(&imported.program).expect("split");
        let mut start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        let groups = key_values(&start).unwrap();
        assert_eq!(groups.len(), 2, "one key-value group per layer");
        assert!(groups.values().all(|g| g.heads.len() == 2 && g.own_key && g.own_value));
        // Layer 1's group is layer 0's with its two query heads swapped.
        let (first, second) = (&groups[&(0, 0)], &groups[&(1, 0)]);
        let copies = [(first.heads[1].1.query, second.heads[0].1.query), (first.heads[0].1.query, second.heads[1].1.query), (first.key, second.key)];
        let program = &mut start.artifact.program;
        for (from, to) in copies {
            program.operators[to] = Arc::new(dense(program.operators[to].name.clone(), program.operators[to].rows.clone(), program.operators[to].cols.clone(), program.operators[from].matrix(), Provenance::default()).unwrap());
        }
        let owner = Member { layer: 0, group: 0, queries: vec![0, 1] };
        let shared = share_query_key(&start, &[owner.clone(), Member { layer: 1, group: 0, queries: vec![1, 0] }]).expect("shared");
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), shared.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[shared.artifact.program.output]);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs())), "the assigned group keeps the explanation's output");
        same_native_blocks(&start, &shared);
        // The member's planes (its key and both queries) leave; each query head's scale is a group.
        let planes = start.layers[1].heads[0].0.len();
        assert_eq!(shared.groups.len(), start.groups.len() - planes + 2);
        assert_eq!(shared.trainable.len(), start.trainable.len() - 3 + 2);
        assert_eq!(shared.layers[1].heads[0].0, shared.layers[0].heads[0].0, "the member's heads read the owner's planes");
        Posterior::new(&shared, 2 * 6 * 12).expect("every shared entry in one group");
        let after = key_values(&shared).unwrap();
        assert!(!after[&(1, 0)].own_key && after[&(1, 0)].own_value, "the member reads the shared key and keeps its value");
        assert!(share_query_key(&start, &[owner.clone(), Member { layer: 1, group: 0, queries: vec![0, 0] }]).is_err(), "an assignment is a permutation");
        assert!(share_query_key(&shared, &[owner, Member { layer: 1, group: 0, queries: vec![0, 1] }]).is_err(), "a member is no member twice");
    }

    #[test]
    fn a_value_map_moved_through_the_output_projections_is_stored_once_and_keeps_the_outputs() {
        let plain = || {
            let dir = crate::test_support::tiny_export("library_value_share", 2);
            let imported = import_language_model(&dir, 6, 12).expect("import");
            std::fs::remove_dir_all(dir).unwrap();
            imported
        };
        for imported in [plain(), grouped("library_value_share_grouped")] {
            let native = split_sites(&imported.program).expect("split");
            let mut start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
            // Layer 1's first group writes through its heads what layer 0's first group writes,
            // scaled by 0.7: its value map is 0.7 T V_s.
            let transport = transports(&start, (1, 0), &[(0, 0)]).unwrap().pop().unwrap();
            let groups = key_values(&start).unwrap();
            let (to, from) = (groups[&(1, 0)].value, groups[&(0, 0)].value);
            let program = &mut start.artifact.program;
            let values = transport.matrix.dot(&program.operators[from].matrix()) * 0.7;
            program.operators[to] = Arc::new(dense(program.operators[to].name.clone(), program.operators[to].rows.clone(), program.operators[to].cols.clone(), values, Provenance::default()).unwrap());
            let shared = share_value(&start, (1, 0), (0, 0), 0.7).expect("shared");
            let (before, after) = (start.artifact.execute(&imported.family).unwrap(), shared.artifact.execute(&imported.family).unwrap());
            let (a, b) = (&before.values[start.artifact.program.output], &after.values[shared.artifact.program.output]);
            let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the shared value map keeps the outputs");
            same_native_blocks(&start, &shared);
            // The target's value coordinates leave; its scale is one group; its heads read the source's.
            let width = start.artifact.program.operators[to].rows.width();
            assert_eq!(shared.groups.len(), start.groups.len() - width + 1);
            assert_eq!(shared.trainable.len(), start.trainable.len());
            let after = key_values(&shared).unwrap();
            assert!(!after[&(1, 0)].own_value && after[&(1, 0)].own_key);
            assert_eq!(shared.layers[1].heads[after[&(1, 0)].heads[0].0].1, shared.layers[0].heads[after[&(0, 0)].heads[0].0].1);
            Posterior::new(&shared, 2 * 6 * 12).expect("every shared entry in one group");
            assert!(share_value(&start, (0, 0), (1, 0), 1.0).is_err(), "a later value map cannot stand for an earlier one");
            assert!(share_value(&shared, (1, 0), (0, 0), 1.0).is_err(), "a shared value map is shared once");
        }
    }

    #[test]
    fn a_native_edit_translates_to_its_site_alone_through_every_sharing() {
        let dir = crate::test_support::tiny_export("library_native_edit", 2);
        let imported = import_language_model(&dir, 6, 12).expect("import");
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).expect("split");
        let mut start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        // Exact copies: layer 0's output 3 writes 2.5 times layer 1's gate 5; layer 1's function 7
        // is layer 0's function 4 (gate times 2, output times 0.5); head 0 of layer 1 attends as
        // head 0 of layer 0; layer 1's group 1 writes 0.7 times what layer 0's group 1 writes.
        let set = |start: &mut Explanation, name: &str, values: Array2<f64>| {
            let program = &mut start.artifact.program;
            let at = operator_index(program, name).unwrap();
            let source = &program.operators[at];
            program.operators[at] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, Provenance::default()).unwrap());
        };
        let get = |start: &Explanation, name: &str| start.artifact.program.operators[operator_index(&start.artifact.program, name).unwrap()].matrix();
        let mut out0 = get(&start, "library.l0.mlp.out");
        out0.column_mut(3).assign(&(&get(&start, "library.l1.mlp.gate").row(5) * 2.5));
        set(&mut start, "library.l0.mlp.out", out0);
        let mut gate1 = get(&start, "library.l1.mlp.gate");
        gate1.row_mut(7).assign(&(&get(&start, "library.l0.mlp.gate").row(4) * 2.0));
        set(&mut start, "library.l1.mlp.gate", gate1);
        let mut out1 = get(&start, "library.l1.mlp.out");
        out1.column_mut(7).assign(&(&get(&start, "library.l0.mlp.out").column(4) * 0.5));
        set(&mut start, "library.l1.mlp.out", out1);
        for part in ["h0.q", "kv0.k"] {
            let values = get(&start, &format!("library.l0.{part}"));
            set(&mut start, &format!("library.l1.{part}"), values);
        }
        let transport = transports(&start, (1, 1), &[(0, 1)]).unwrap().pop().unwrap();
        let moved = transport.matrix.dot(&get(&start, "library.l0.kv1.v")) * 0.7;
        set(&mut start, "library.l1.kv1.v", moved);
        let tied = tie(&start, &[Tie { source: (0, 3), target: (1, 5), scale: 2.5 }]).unwrap();
        let row = tie_row(&tied, "gate", (1, 7), RowSource::Row { layer: 0, part: "gate", function: 4 }, 2.0).unwrap();
        let column = tie_column(&row, (1, 7), (0, 4), 0.5).unwrap();
        let heads = share_query_key(&column, &[Member { layer: 0, group: 0, queries: vec![0] }, Member { layer: 1, group: 0, queries: vec![0] }]).unwrap();
        let shared = share_value(&heads, (1, 1), (0, 1), 0.7).unwrap();
        same_native_blocks(&start, &shared);
        // Every native block a sharing moved, or a sharing reads, edited at its own site: the same
        // as editing it in the unshared library, where each site holds its own parameters.
        let output = |e: &Explanation| e.artifact.execute(&imported.family).unwrap().values[e.artifact.program.output].clone();
        let base = output(&start);
        let scale = base.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let picked = |o: &crate::artifact::Owner| match (o.site.as_str(), o.role.as_str()) {
            ("library.l0.mlp", "gate") => o.native_rows == (4..5),
            ("library.l1.mlp", "gate") => o.native_rows == (5..6) || o.native_rows == (7..8),
            ("library.l0.mlp", "out") => o.native_cols == (3..4) || o.native_cols == (4..5),
            ("library.l1.mlp", "out") => o.native_cols == (7..8),
            ("library.l0.h0" | "library.l1.h0", "q" | "k") | ("library.l0.h1" | "library.l1.h1", "v") => true,
            _ => false,
        };
        let mut edited = 0;
        for owner in shared.artifact.owners.iter().filter(|o| picked(o)) {
            let delta = Array2::from_shape_fn((owner.native_rows.len(), owner.native_cols.len()), |(i, j)| 0.1 * ((3 * i + 7 * j + edited) as f64).sin());
            let own = start.artifact.owners.iter().find(|o| o.native == owner.native && o.native_rows == owner.native_rows && o.native_cols == owner.native_cols && o.site == owner.site).unwrap();
            let mut reference = start.clone();
            let mut values = get(&reference, &own.operator);
            let mut block = values.slice_mut(ndarray::s![own.rows.clone(), own.cols.clone()]);
            block += &delta;
            set(&mut reference, &own.operator, values);
            let (expected, found) = (output(&reference), output(&edit_native(&shared, owner, &delta).unwrap()));
            assert!(expected.iter().zip(&base).any(|(x, y)| (x - y).abs() > 1e-6 * scale), "{} at {} changes the output", owner.native, owner.site);
            assert!(expected.iter().zip(found.iter()).all(|(x, y)| (x - y).abs() <= 1e-10 * scale), "{} edited at {} alone", owner.native, owner.site);
            edited += 1;
        }
        assert_eq!(edited, 12, "every chosen native block was edited");
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

    /// `(γ_q ⊙ N(q))ᵀ R (γ_k ⊙ N(k))`, `N` the RMS norm (ε = 0) and `R` turning each plane of
    /// `planes` by `angle`.
    fn normed_score(q: &[f64], k: &[f64], (gq, gk): (&[f64], &[f64]), planes: &[Vec<usize>], angle: f64) -> f64 {
        let normed = |x: &[f64], g: &[f64]| -> Vec<f64> {
            let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
            x.iter().zip(g).map(|(v, g)| g * v / rms).collect()
        };
        let (q, mut k) = (normed(q, gq), normed(k, gk));
        let (sine, cosine) = angle.sin_cos();
        for rows in planes.iter().filter(|r| r.len() == 2) {
            let (a, b) = (k[rows[0]], k[rows[1]]);
            (k[rows[0]], k[rows[1]]) = (cosine * a - sine * b, sine * a + cosine * b);
        }
        q.iter().zip(&k).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn the_gauges_of_normed_heads_keep_their_normed_scores() {
        let planes = [vec![0, 1], vec![2, 3]];
        let ones = [1.0; 4];
        // A plane's reciprocal scaling keeps the raw score and moves the normed one (4 to 3.2).
        let (q, k) = ([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]);
        let (qs, ks) = ([2.0, 0.0, 1.0, 0.0], [0.5, 0.0, 1.0, 0.0]);
        let raw = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        assert_eq!(raw(&q, &k), raw(&qs, &ks));
        assert!((normed_score(&q, &k, (&ones, &ones), &planes, 0.0) - 4.0).abs() < 1e-12);
        assert!((normed_score(&qs, &ks, (&ones, &ones), &planes, 0.0) - 3.2).abs() < 1e-12);
        // Gains unequal on plane 0, equal on plane 1: plane 0 takes the identity or the half turn,
        // plane 1 any rotation, and no plane a scale.
        let (gq, gk) = ([0.7, 1.3, 0.9, 0.9], [1.1, 0.6, 1.2, 1.2]);
        let symmetry = Symmetry { scale: false, turn: vec![false, true] };
        let q1 = Array2::from_shape_fn((4, 3), |(i, j)| ((i * 3 + j) as f64 * 0.37).sin() + 0.2);
        let k1 = Array2::from_shape_fn((4, 3), |(i, j)| ((i + 2 * j) as f64 * 0.53).cos() - 0.1);
        let (sine, cosine) = 0.9_f64.sin_cos();
        let mut g = Array2::<f64>::zeros((4, 4));
        g[[0, 0]] = -1.0;
        g[[1, 1]] = -1.0;
        g.slice_mut(ndarray::s![2..4, 2..4]).assign(&ndarray::array![[cosine, -sine], [sine, cosine]]);
        let (q, k) = (g.dot(&q1), g.dot(&k1));
        let gauge = gauge(&[&q1], &k1, &[&q], &k, &planes, &symmetry);
        assert!(gauge.iter().all(|(_, s)| *s == 1.0), "normed heads take no scale");
        let (qa, ka) = (turn(&q, &planes, &gauge, true, false), turn(&k, &planes, &gauge, false, false));
        assert!((&qa - &q1).iter().chain((&ka - &k1).iter()).all(|d| d.abs() < 1e-12), "the gauge brings the maps back");
        // Every gauge it can take keeps the normed scores of the maps it moves, at any rotary turn.
        let x = [0.3, -1.2, 0.8];
        let y = [-0.4, 0.9, 1.1];
        let apply = |m: &Array2<f64>, v: &[f64]| m.dot(&ndarray::Array1::from(v.to_vec())).to_vec();
        for angle in [0.0, 0.4, 2.1] {
            let before = normed_score(&apply(&q, &x), &apply(&k, &y), (&gq, &gk), &planes, angle);
            let after = normed_score(&apply(&qa, &x), &apply(&ka, &y), (&gq, &gk), &planes, angle);
            assert!((before - after).abs() < 1e-12, "the gauge keeps the normed score: {before} against {after}");
        }
    }

    #[test]
    fn the_gauges_of_a_qwen3_library_take_no_scale() {
        let dir = crate::test_support::tiny_qwen3_export("library_sharing_symmetry", 2);
        let imported = import_language_model(&dir, 6, 12).expect("import");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("split");
        let start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        let groups = key_values(&start).expect("groups");
        let group = &groups[&(0, 0)];
        let planes = planes(start.artifact.program.operators[group.key].rows.width(), group.rotary);
        let found = symmetry(&start.artifact.program, &[group], &planes).expect("symmetry");
        assert!(!found.scale, "heads with query and key norms take no scale");
        assert_eq!(found.turn.len(), planes.len());
    }

    #[test]
    fn the_best_assignment_is_found_jointly_with_the_transport() {
        // Scalar projections (1, 2) of the target heads and (2, 1) of the source's: each pair alone
        // fits exactly, the identity's common transport leaves a residual (0.8 with 1.8 left) and
        // the swap none.
        let m = |v: f64| Array2::from_elem((1, 1), v);
        let norm = (1.0_f64 + 4.0).sqrt();
        let projected = vec![vec![m(2.0 / norm), m(1.0 / norm)], vec![m(4.0 / norm), m(2.0 / norm)]];
        assert_eq!(best_assignment(&projected, vec![0, 1]), vec![1, 0]);
        // A permutation of five heads, found from the identity.
        let blocks: Vec<Array2<f64>> = (0..5).map(|i| Array2::from_shape_fn((3, 2), |(r, c)| ((i * 7 + r * 3 + c) as f64 * 0.61).sin())).collect();
        let truth = [3, 0, 4, 1, 2];
        let projected: Vec<Vec<Array2<f64>>> = (0..5).map(|i| (0..5).map(|j| if truth[i] == j { blocks[i].clone() } else { -&blocks[i] * 0.3 + 0.05 * (i * 5 + j) as f64 }).collect()).collect();
        let found = best_assignment(&projected, (0..5).collect());
        let value = |a: &[usize]| {
            let mut sum = projected[0][a[0]].clone();
            for (i, &j) in a.iter().enumerate().skip(1) {
                sum += &projected[i][j];
            }
            sum.iter().map(|v| v * v).sum::<f64>()
        };
        // Exhaustively: no permutation scores higher.
        let mut best = 0.0_f64;
        let mut order: Vec<usize> = (0..5).collect();
        fn each(k: usize, order: &mut Vec<usize>, visit: &mut dyn FnMut(&[usize])) {
            if k == order.len() {
                visit(order);
                return;
            }
            for i in k..order.len() {
                order.swap(k, i);
                each(k + 1, order, visit);
                order.swap(k, i);
            }
        }
        each(0, &mut order, &mut |a| best = best.max(value(a)));
        assert!((value(&found) - best).abs() <= 1e-12 * best, "branch and bound finds the best assignment");
    }

    #[test]
    fn alignment_undoes_a_plane_rotation_and_scale() {
        let q1 = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f64 + 1.0);
        let k1 = Array2::from_shape_fn((2, 3), |(i, j)| ((i + 2 * j) as f64).sin() + 2.0);
        let (sine, cosine) = 0.7_f64.sin_cos();
        let r = ndarray::array![[cosine, -sine], [sine, cosine]];
        let (q, k) = (r.dot(&q1) * 3.0, r.dot(&k1) / 3.0);
        // Two query heads of one key: one rotation and one scale of the plane for the whole group.
        let q2 = Array2::from_shape_fn((2, 3), |(i, j)| ((3 * i + j) as f64).cos() - 0.5);
        let planes = [vec![0, 1]];
        let gauge = gauge(&[&q1, &q2], &k1, &[&(r.dot(&q1) * 3.0), &(r.dot(&q2) * 3.0)], &k, &planes, &Symmetry { scale: true, turn: vec![true] });
        let (qa, q2a, ka) = (turn(&q, &planes, &gauge, true, false), turn(&(r.dot(&q2) * 3.0), &planes, &gauge, true, false), turn(&k, &planes, &gauge, false, false));
        assert!((&qa - &q1).iter().chain((&q2a - &q2).iter()).chain((&ka - &k1).iter()).all(|d| d.abs() < 1e-12));
        // The transpose takes a cotangent back: <turn(x), y> = <x, turn^T(y)>.
        let y = Array2::from_shape_fn((2, 3), |(i, j)| (i as f64 - j as f64) * 0.3 + 0.1);
        let (forward, back) = (turn(&q, &planes, &gauge, true, false), turn(&y, &planes, &gauge, true, true));
        assert!(((&forward * &y).sum() - (&q * &back).sum()).abs() < 1e-12);
    }
}
