//! Shared functions of a library explanation (#2951).
//!
//! Several heads of a library ([`crate::library_mdl`]) may compute one attention pattern (VPD's
//! previous-token and induction behaviours appear in more than one head). A shared query–key
//! function replaces the members' query and key maps by one pair `(Q, K)` that every member reads:
//! the first member attends with `q = Q x`, `k = K x`, and every other member `m` with
//! `q = G_m Q x`, `k = K x`, where `G_m` (head dimension square, starting at the identity) is its
//! own part. Every member keeps its value map. The shared maps carry the prior groups of one head
//! (a group per rotary plane, as in the library), paid once in `KL(q ‖ p)`; each `G_m` is one
//! group. Members sit in different layers, so each layer's read of `Q x` stays its own variable,
//! and each reads a key of its own: a key the query heads of one key-value group share stays
//! that group's.
//! The move is accepted only if the code length `F` of the re-converged fit falls.
//!
//! # Candidates
//!
//! A head's scores are `qᵀ R k` with `R` the rotary turn between the two positions; the score map
//! `B = Qᵀ K` (`d × d`) is unchanged by every rotation and scaling of a plane that leaves the
//! scores unchanged. Pairs of heads in different layers are ranked by the cosine of their score
//! maps.
//!
//! # The start
//!
//! The shared maps start at the members' mean after each member is brought to the first member's
//! gauge, plane by plane: the rotation `R` that minimizes `‖R Q_m − Q_1‖² + ‖R K_m − K_1‖²` over a
//! plane's two rows (the orthogonal Procrustes solution in two dimensions; a sign on a coordinate
//! no rotary plane holds), and the scale `s` with `(s R Q_m, R K_m / s)` in the first member's
//! ratio of query to key norm. Both leave the member's scores unchanged, so the start is close to
//! the current fit.

use crate::{
    library_mdl::{Cells, Explanation, Group},
    operator_program::{Interface, Node, Operator, OperatorProgram, Provenance, Rotary, exact_precision},
};
use ndarray::Array2;
use std::{collections::BTreeMap, sync::Arc};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Two heads `(layer, head)` and the cosine of their score maps.
#[derive(Clone, Debug, PartialEq)]
pub struct Pair {
    pub first: (usize, usize),
    pub second: (usize, usize),
    pub cosine: f64,
}

/// A head's block in the explanation's program: its rule's index, its query, key and value
/// operators, and its rotary planes.
struct Head {
    rule: usize,
    query: usize,
    key: usize,
    rotary: Option<Rotary>,
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
fn heads(explanation: &Explanation) -> Result<BTreeMap<(usize, usize), Head>, String> {
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

/// The pairs of heads in different layers by decreasing cosine of their score maps `Qᵀ K` at the
/// explanation's operator values.
pub fn query_key_pairs(explanation: &Explanation) -> Result<Vec<Pair>, String> {
    let program = &explanation.artifact.program;
    let maps: Vec<((usize, usize), Array2<f64>)> = heads(explanation)?
        .into_iter()
        .map(|(at, h)| (at, program.operators[h.query].matrix().t().dot(&program.operators[h.key].matrix())))
        .collect();
    let norms: Vec<f64> = maps.iter().map(|(_, b)| b.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let mut out = Vec::new();
    for i in 0..maps.len() {
        for j in i + 1..maps.len() {
            if maps[i].0.0 == maps[j].0.0 || norms[i] == 0.0 || norms[j] == 0.0 {
                continue;
            }
            let inner = (&maps[i].1 * &maps[j].1).sum();
            out.push(Pair { first: maps[i].0, second: maps[j].0, cosine: inner / (norms[i] * norms[j]) });
        }
    }
    out.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
    Ok(out)
}

/// The row sets the score map's gauge acts on: each rotary plane's two rows, then each coordinate
/// no plane holds.
fn planes(width: usize, rotary: Option<Rotary>) -> Vec<Vec<usize>> {
    let pairs = rotary.map(|r| r.pairs()).unwrap_or_default();
    let rotated: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    pairs.iter().map(|&(a, b)| vec![a, b]).chain((0..width).filter(|c| !rotated.contains(c)).map(|c| vec![c])).collect()
}

fn norm(x: &Array2<f64>) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `(q, k)` of a member brought to the gauge of `(q1, k1)` plane by plane (module note).
fn aligned(q1: &Array2<f64>, k1: &Array2<f64>, q: &Array2<f64>, k: &Array2<f64>, planes: &[Vec<usize>]) -> (Array2<f64>, Array2<f64>) {
    let (mut q_out, mut k_out) = (q.clone(), k.clone());
    for rows in planes {
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
        for (i, &row) in rows.iter().enumerate() {
            q_out.row_mut(row).assign(&(&r_q.row(i) * scale));
            k_out.row_mut(row).assign(&(&r_k.row(i) / scale));
        }
    }
    (q_out, k_out)
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
    // Every other member reads the shared maps, its query through its own part `G_m`.
    let coordinates = program.operators[owner.query].rows.clone();
    let mut retired = Vec::new();
    let mut parts = Vec::new();
    for (m, member) in members.iter().zip(&found).skip(1) {
        retired.extend([member.query, member.key]);
        let gain = program.operators.len();
        let name = format!("library.l{}.h{}.q_shared_part", m.0, m.1);
        let identity = Array2::eye(coordinates.width());
        program.operators.push(Arc::new(dense(name.clone(), coordinates.clone(), coordinates.clone(), identity, provenance.clone())?));
        parts.push((name, gain));
        let rule = &mut program.rules[member.rule];
        rule.nodes[1] = Node::Affine { terms: vec![(0, owner.query)], bias: None };
        rule.nodes[2] = Node::Affine { terms: vec![(0, owner.key)], bias: None };
        let output = rule.output;
        let Node::Attend { query, .. } = rule.nodes[output] else {
            return Err(format!("{}: the output is not an attention node", rule.name));
        };
        rule.nodes.insert(output, Node::Affine { terms: vec![(query, gain)], bias: None });
        rule.output = output + 1;
        if let Node::Attend { query, .. } = &mut rule.nodes[output + 1] {
            *query = output;
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
    let width = coordinates.width();
    for (name, gain) in &parts {
        groups.push(Group { name: name.clone(), cells: vec![Cells { operator: *gain, rows: (0..width).collect(), cols: 0..width }] });
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
    Ok(Explanation { artifact, trainable, groups, layers })
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
        let pairs = query_key_pairs(&start).expect("pairs");
        assert!(pairs.iter().all(|p| p.first.0 != p.second.0));
        assert!((pairs[0].cosine - 1.0).abs() < 1e-12 && pairs[0].first == (0, 0) && pairs[0].second == (1, 0), "{:?}", pairs[0]);
        let shared = share_query_key(&start, &[(0, 0), (1, 0)]).expect("shared");
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), shared.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[shared.artifact.program.output]);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs())), "identical members keep the explanation's output");
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
        assert!(query_key_pairs(&start).unwrap().is_empty(), "every head's key is its group's");
        assert!(share_query_key(&start, &[(0, 0), (1, 0)]).is_err());
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
