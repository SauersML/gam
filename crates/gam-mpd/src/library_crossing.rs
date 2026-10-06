//! Bodies that read through a head: computations across the attention/MLP boundary (#2951).
//!
//! A head `h` of layer `l′ ≤ l` writes `O_h z_h` into the residual stream, and layer `l`'s MLP reads
//! the normed stream `x̂ = γ ⊙ x / rms(x)`, so the head's writes reach the MLP's reads along the
//! columns of `H = diag(γ) O_h` (`d × e`, `e` the head's width). MLP functions that read only what
//! one head writes are a unit of computation that spans the two blocks: a head and the functions
//! it feeds. A body call of such functions (`library_bodies`) reads through the head:
//! its read binding is `R = P Hᵀ`, `P` (`k × e`) its own and `H` the head's, so the binding costs
//! `k e` values instead of `k d` and names the head it reads (`ln` of the number of heads it may
//! read, in `Explanation::fixed_nats`). Merges align and transform `P` as they would `R`
//! (`library_bodies::merge`), so a computation fed by different heads at different layers can be
//! one body.
//!
//! # Regions through a head
//!
//! [`regions_through`] takes, among an MLP's functions in the explanation, those whose reads lie in
//! the write directions of a set of heads up to the posterior's resolution (a body of several
//! inputs). The reads are whitened in the separable row-times-column form that keeps rank
//! (`library_bodies`), the heads' directions with the same column scales; a function's rows lie in
//! them when the largest singular value of what is left is within that of same-shaped unit noise in
//! the complement, `√rows + √(d − rank)`. A function's set grows one head at a time, the head
//! taking the most of what is left joining, until what is left is within noise; a set as wide as
//! the stream would read everything and save nothing, so it is not formed. Functions with one set
//! form one region. [`read_through`] then makes an existing call read through the head:
//! `P = R H (Hᵀ H)⁺`, the least-squares factor, exact when `R`'s rows lie in `H`'s columns. Both are
//! proposals; a rewrite is accepted only if the code length `F` falls on the fixed experiments.

use crate::{
    library_bodies::Call,
    library_mdl::{Explanation, Posterior},
    operator_program::{Interface, LabelKind, Node, Operator, OperatorProgram, Provenance, exact_precision, remap_node},
    run_check::LayerNodes,
};
use gam_linalg::decompose::{pseudo_inverse, svd};
use ndarray::{Array1, Array2, Axis, s};
use std::sync::Arc;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn operator_index(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(format!("no unique operator {name}")),
    }
}

fn dense(name: String, rows: Interface, cols: Interface, values: Array2<f64>, provenance: Provenance) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, provenance).map_err(error)
}

/// A head's writes as one MLP reads them: the head (its layer and index), its output projection's
/// name in `M`, and `H = diag(γ) O_h` (`d × e`).
#[derive(Clone, Debug)]
pub struct Writer {
    pub layer: usize,
    pub head: usize,
    pub operator: String,
    pub writes: Array2<f64>,
}

/// The gain of the RMS norm whose gain node is `normed` in `native`.
fn gain(native: &OperatorProgram, normed: usize) -> Result<Array1<f64>, String> {
    let Node::Affine { terms, .. } = &native.nodes[normed] else { return Err(format!("node {normed} is not a normed stream")) };
    let [(rms, gain)] = terms[..] else { return Err(format!("node {normed} is not one gain of a norm")) };
    if !matches!(native.nodes[rms], Node::RmsNorm { .. }) {
        return Err(format!("node {normed} does not read an RMS norm"));
    }
    Ok(native.operators[gain].matrix().diag().to_owned())
}

/// The writers layer `layer`'s MLP reads: every head of layers `0..=layer` (`native` the split
/// native program, `layers` its sites).
pub fn writers(native: &OperatorProgram, layers: &[LayerNodes], layer: usize) -> Result<Vec<Writer>, String> {
    let gamma = gain(native, layers.get(layer).ok_or_else(|| format!("no layer {layer}"))?.normed)?;
    let mut out = Vec::new();
    for (l, site) in layers.iter().enumerate().take(layer + 1) {
        let Node::Affine { terms, .. } = &native.nodes[site.attention] else { return Err(format!("layer {l}: the attention output is not a map")) };
        for (h, read) in site.reads.iter().enumerate() {
            let op = terms.iter().find(|(n, _)| n == read).map(|(_, op)| *op).ok_or_else(|| format!("layer {l} head {h}: no output columns"))?;
            let o = native.operators[op].matrix();
            out.push(Writer { layer: l, head: h, operator: native.operators[op].name.clone(), writes: &o * &gamma.view().insert_axis(Axis(1)) });
        }
    }
    Ok(out)
}

/// The columns of `m`'s resolved left singular vectors (above the decomposition's rounding band).
fn basis(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    let decomposition = svd(m.view(), false).map_err(error)?;
    let rank = decomposition.singular_values.iter().filter(|s| **s > decomposition.band).count();
    Ok(decomposition.u.slice(s![.., ..rank]).to_owned())
}

/// The regions of layer `layer`'s MLP that read through heads (module note), among its functions
/// in `pool` whose groups are in the explanation at `posterior`: per set of writers (indices into
/// `writers`) the functions reading exactly through it, wherever there are at least two.
pub fn regions_through(explanation: &Explanation, posterior: &Posterior, writers: &[Writer], layer: usize, pool: &[usize]) -> Result<Vec<(Vec<usize>, Vec<usize>)>, String> {
    let program = &explanation.artifact.program;
    let mlp = format!("library.l{layer}.mlp");
    let maps: Vec<usize> = ["gate", "up"].iter().filter_map(|part| program.operators.iter().position(|op| op.name == format!("{mlp}.{part}"))).collect();
    let at = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("{mlp}: operator {op} is not trainable"));
    let known = &explanation.layers.get(layer).ok_or_else(|| format!("no layer {layer}"))?.functions;
    let live: Vec<usize> = pool.iter().copied().filter(|i| known.get(*i).is_some_and(|groups| groups.iter().all(|g| posterior.active[*g]))).collect();
    if live.is_empty() || maps.is_empty() {
        return Ok(Vec::new());
    }
    let parts = maps.len();
    let positions = maps.iter().map(|op| at(*op)).collect::<Result<Vec<_>, _>>()?;
    let d = posterior.mean[positions[0]].ncols();
    // The live functions' reads and log deviations, stacked (`parts` rows per function), whitened
    // in the separable form.
    let rows = live.len() * parts;
    let reads = Array2::from_shape_fn((rows, d), |(r, c)| posterior.mean[positions[r % parts]][[live[r / parts], c]]);
    let log_sd = Array2::from_shape_fn((rows, d), |(r, c)| posterior.log_sd[positions[r % parts]][[live[r / parts], c]]);
    let row_scales = log_sd.mean_axis(Axis(1)).ok_or("no reads")?;
    let columns = (&log_sd - &row_scales.view().insert_axis(Axis(1))).mean_axis(Axis(0)).ok_or("no reads")?;
    let whitened = Array2::from_shape_fn((rows, d), |(r, c)| reads[[r, c]] * (-row_scales[r] - columns[c]).exp());
    let column_scale = columns.mapv(|c| (-c).exp());
    // Per writer its whitened directions. Per function, the heads it reads: the head capturing the
    // most of what is left of its whitened reads joins, one at a time, until what is left is within
    // noise; a set as wide as the stream reads everything and saves nothing, so a function needing
    // one is taken by no set.
    let scaled: Vec<Array2<f64>> = writers.iter().map(|w| &w.writes * &column_scale.view().insert_axis(Axis(1))).collect();
    let mut taken: std::collections::BTreeMap<Vec<usize>, Vec<usize>> = std::collections::BTreeMap::new();
    for (f, &i) in live.iter().enumerate() {
        let block = whitened.slice(s![f * parts..(f + 1) * parts, ..]).to_owned();
        let mut set: Vec<usize> = Vec::new();
        loop {
            let span = if set.is_empty() {
                Array2::zeros((d, 0))
            } else {
                let views: Vec<_> = set.iter().map(|w| scaled[*w].view()).collect();
                basis(&ndarray::concatenate(Axis(1), &views).map_err(error)?)?
            };
            let residual = &block - &block.dot(&span).dot(&span.t());
            let largest = svd(residual.view(), false).map_err(error)?.singular_values.first().copied().unwrap_or(0.0);
            if largest <= (parts as f64).sqrt() + ((d - span.ncols()) as f64).sqrt() {
                if !set.is_empty() {
                    set.sort_unstable();
                    taken.entry(set).or_default().push(i);
                }
                break;
            }
            let next = (0..writers.len())
                .filter(|w| !set.contains(w))
                .map(|w| {
                    let inside = residual.dot(&basis(&scaled[w])?);
                    Ok((inside.iter().map(|v| v * v).sum::<f64>(), w))
                })
                .collect::<Result<Vec<(f64, usize)>, String>>()?
                .into_iter()
                .max_by(|a, b| a.0.total_cmp(&b.0));
            let Some((_, w)) = next else { break };
            set.push(w);
            if set.iter().map(|w| writers[*w].writes.ncols()).sum::<usize>() >= d {
                break;
            }
        }
    }
    Ok(taken.into_iter().filter(|(_, f)| f.len() > 1).collect())
}

/// `explanation` with `call`'s read binding made to read through `writers` (module note): `R = P Hᵀ`
/// with `H` their writes side by side, `P = R H (Hᵀ H)⁺` its own trainable binding and `Hᵀ` a
/// constant of the call; the binding's groups now span `P`'s rows, the owners of the call's reads
/// gain `Hᵀ` as a factor after `P`, and the choice of the heads among the `choices` the MLP may read
/// costs their enumerative subset code.
pub fn read_through(explanation: &Explanation, call: &Call, writers: &[&Writer], choices: usize) -> Result<Explanation, String> {
    let mut out = explanation.clone();
    let program = &mut out.artifact.program;
    let read = operator_index(program, &format!("{}.read", call.name))?;
    let mlp = format!("library.l{}.mlp", call.layer);
    let rule = program.rules.iter().position(|r| r.name == mlp).ok_or_else(|| format!("no rule {mlp}"))?;
    let z = program.rules[rule]
        .nodes
        .iter()
        .position(|n| matches!(n, Node::Affine { terms, .. } if terms.len() == 1 && terms[0].1 == read))
        .ok_or_else(|| format!("{}: no node applies its read binding", call.name))?;
    let source = Arc::clone(&program.operators[read]);
    let views: Vec<_> = writers.iter().map(|w| w.writes.view()).collect();
    let h = &ndarray::concatenate(Axis(1), &views).map_err(error)?;
    let names: Vec<&str> = writers.iter().map(|w| w.operator.as_str()).collect();
    if source.cols.width() != h.nrows() {
        return Err(format!("{}: reads {} coordinates, the head writes {}", call.name, source.cols.width(), h.nrows()));
    }
    let p = source.matrix().dot(h).dot(&pseudo_inverse(h.t().dot(h).view()).map_err(error)?);
    let width = h.ncols();
    let head = Interface::uniform(width, 1, LabelKind::Unit, 0).map_err(error)?;
    let provenance = Provenance::derived(&[&source.provenance], format!("{} read through {}", call.name, names.join(", ")));
    let through = format!("{}.through", call.name);
    program.operators[read] = Arc::new(dense(source.name.clone(), source.rows.clone(), head.clone(), p, provenance.clone())?);
    program.operators.push(Arc::new(dense(through.clone(), head, source.cols.clone(), h.t().to_owned(), provenance)?));
    let constant = program.operators.len() - 1;
    // `t = Hᵀ x̂` before the read binding's node, which then reads `t`.
    let (ops, bases, rules): (Vec<usize>, Vec<usize>, Vec<usize>) = ((0..program.operators.len()).collect(), (0..program.bases.len()).collect(), (0..program.rules.len()).collect());
    let r = &mut program.rules[rule];
    let Node::Affine { terms, .. } = &r.nodes[z] else { return Err("a read binding's node".into()) };
    let input = terms[0].0;
    let map: Vec<usize> = (0..r.nodes.len()).map(|n| if n < z { n } else { n + 1 }).collect();
    let mut nodes = Vec::with_capacity(r.nodes.len() + 1);
    for (n, node) in r.nodes.iter().enumerate() {
        if n == z {
            nodes.push(Node::Affine { terms: vec![(input, constant)], bias: None });
        }
        let mut node = node.clone();
        remap_node(&mut node, &map, &ops, &bases, &rules);
        if n == z
            && let Node::Affine { terms, .. } = &mut node
        {
            terms[0].0 = z;
        }
        nodes.push(node);
    }
    r.nodes = nodes;
    r.output = map[r.output];
    program.interfaces().map_err(error)?;
    for group in &mut out.groups {
        for cell in group.cells.iter_mut().filter(|c| c.operator == read) {
            cell.cols = 0..width;
        }
    }
    let read_name = format!("{}.read", call.name);
    for owner in &mut out.artifact.owners {
        if let Some(at) = owner.right.iter().position(|f| *f == read_name) {
            owner.right.insert(at + 1, through.clone());
        }
    }
    // Which heads, among the `choices` the MLP may read (the enumerative subset code).
    out.fixed_nats += crate::codec::subset_code_len_bits(choices, writers.len()).map_err(error)? as f64 * std::f64::consts::LN_2;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        library_bodies::rewrite,
        library_mdl::explanation,
        operator_program::{FamilyInputs, SlotValues},
        run_check::{layer_nodes, split_sites},
    };
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn tiny(tag: &str) -> (OperatorProgram, Vec<LayerNodes>, FamilyInputs) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(_) = &imported.family.slots[0] else { panic!("token slot") };
        (native, layers, imported.family)
    }

    #[test]
    fn functions_reading_one_head_are_found_and_read_through_it_exactly() {
        let (native, layers, family) = tiny("crossing");
        let mut start = explanation(&native, &layers).unwrap();
        // Layer 1's functions 2, 5 and 9 read only what layer 0's head 1 writes.
        let all = writers(&native, &layers, 1).unwrap();
        let w = all.iter().position(|w| (w.layer, w.head) == (0, 1)).unwrap();
        let mut rng = StdRng::seed_from_u64(4);
        let program = &mut start.artifact.program;
        let gate = operator_index(program, "library.l1.mlp.gate").unwrap();
        let mut values = program.operators[gate].matrix();
        for f in [2, 5, 9] {
            let coefficients = Array1::from_shape_fn(all[w].writes.ncols(), |_| rng.random::<f64>() - 0.5);
            values.row_mut(f).assign(&all[w].writes.dot(&coefficients));
        }
        let source = Arc::clone(&program.operators[gate]);
        program.operators[gate] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, source.provenance.clone()).unwrap());
        let posterior = Posterior::new(&start, 1_000_000).unwrap();
        let found = regions_through(&start, &posterior, &all, 1, &(0..16).collect::<Vec<_>>()).unwrap();
        assert_eq!(found, vec![(vec![w], vec![2, 5, 9])]);
        // Rewritten and read through the head, the explanation computes what it computed.
        let (rewritten, call) = rewrite(&start, 1, &[2, 5, 9]).unwrap();
        let through = read_through(&rewritten, &call, &[&all[w]], all.len()).unwrap();
        let outputs = |e: &Explanation| e.artifact.execute(&family).unwrap().values[e.artifact.program.output].clone();
        let (a, b) = (outputs(&start), outputs(&through));
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-9 * scale), "reading through the head keeps the outputs");
        // The binding is `k × e` and its groups and owners follow.
        let program = &through.artifact.program;
        let read = &program.operators[operator_index(program, &format!("{}.read", call.name)).unwrap()];
        assert_eq!(read.cols.width(), all[w].writes.ncols());
        Posterior::new(&through, 72).unwrap();
        let natives = |e: &Explanation| -> Vec<Array2<f64>> { e.artifact.owners.iter().filter(|o| o.site == call.name && o.role == "gate").map(|o| e.artifact.native_block(o).unwrap()).collect() };
        for (x, y) in natives(&rewritten).iter().zip(natives(&through)) {
            assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() <= 1e-9 * (1.0 + p.abs())), "each native gate row is stated exactly");
        }
        let subset = crate::codec::subset_code_len_bits(all.len(), 1).unwrap() as f64 * std::f64::consts::LN_2;
        assert!((through.fixed_nats - rewritten.fixed_nats - subset).abs() < 1e-12);
        // A native edit of a gate row read through the head (in any direction, also outside the
        // head's writes) is the native library's edit.
        let owner = through.artifact.owners.iter().find(|o| o.site == call.name && o.role == "gate" && o.native_rows == (5..6)).unwrap().clone();
        let delta = Array2::from_shape_fn((1, owner.native_cols.len()), |_| rng.random::<f64>() - 0.5);
        let edited = crate::library_bodies::edit_native(&through, &owner, &delta).unwrap();
        let mut reference = start.clone();
        let program = &mut reference.artifact.program;
        let mut values = program.operators[gate].matrix();
        values.row_mut(5).scaled_add(1.0, &delta.row(0));
        let source = Arc::clone(&program.operators[gate]);
        program.operators[gate] = Arc::new(dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, source.provenance.clone()).unwrap());
        let (a, b) = (outputs(&reference), outputs(&edited));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-9 * scale), "the edit is the native edit");
    }
}
