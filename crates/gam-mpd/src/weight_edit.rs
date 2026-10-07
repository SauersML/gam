//! Native weight edits compiled into the model and an explanation (#2951): `D(E_e(P)) = e(D(P))`.
//!
//! A native edit `ΔW` of one of `M`'s matrices `W` (by its operator's name) is an always-on term
//! `ΔW·x` at every use of `W`, `x` the model's own input to that use, at every row. `M` then computes
//! with `W + ΔW` wherever it applies `W`, its tied uses included (an embedding and the readout that
//! reads it transposed, a key map a group's query heads share). The explanation `P` computes with its
//! decoded `W` plus `ΔW`:
//!
//! * where `P` applies `W` itself (an operator of `M`'s it keeps by name), with `W + ΔW`, as `M` does
//!   (and, where it also owns blocks of `W` elsewhere, at those too);
//! * where one of `P`'s operators owns a block of `W` (`Artifact::owners`: `P`'s block `B` stands for
//!   `M`'s block, up to scalar factors and a transpose), with the matching block of `ΔW` as a term of
//!   its own at every node applying that operator, on that node's input. The term is an operator
//!   apart from `B`, so a fit that writes `P`'s samples into `B` keeps the edit, and every node
//!   applying `B` (a key map several heads read) takes it.
//!
//! * where `P` computes a block of `W` as a sum of slices no single operator holds (`Σ_i U_i V_iᵀ`),
//!   through the block's use (`Owner::uses`: the rule body's nodes holding the map's input and its
//!   output): the block of `ΔW` as a term of the output node on the input node.
//!
//! An explanation that holds no copy of some edited `W`, neither by name nor through an owner, cannot
//! take the edit: [`compile`] reports it as not applicable (`None`). Where `P` owns only some blocks
//! of `W` (a simplified decomposition that dropped units), the rest of `ΔW` reaches `M` alone, and
//! [`Compiled::owned`] says how much of the edit `P` took.
//!
//! A component edit scales one component `b` of `P`'s decomposition by `α` (0 removes it): on every
//! map `W` its slices span, `ΔW = (α − 1) D_W(b)`, `D_W(b)` the sum of its blocks as `M`'s
//! coordinates hold them ([`component`]). Compiled, `M` computes with those slices scaled and `P` with
//! its own component's blocks scaled, so on an exact copy of `M` the two agree exactly.
//!
//! Two edits of one matrix add: `W + ΔW₁ + ΔW₂`.

use crate::{
    artifact::{Artifact, Owner},
    operator_program::{Node, Operator, OperatorProgram, Provenance, exact_precision},
};
use ndarray::{Array2, s};
use std::{collections::BTreeMap, sync::Arc};

fn error(e: impl std::fmt::Display) -> String {
    format!("weight edit: {e}")
}

/// An edit `delta` (its full shape) of `M`'s operator named `native`.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightEdit {
    pub native: String,
    pub delta: Array2<f64>,
}

/// `M` and `P` with an edit compiled into both.
#[derive(Clone, Debug)]
pub struct Compiled {
    /// `M` computing with `W + ΔW` for every edited `W`.
    pub model: OperatorProgram,
    /// `P` computing with its decoded `W` plus `ΔW`.
    pub explanation: Artifact,
    /// The share of the edit's squared size `Σ ‖ΔW‖²` that `P` took (1 where it holds every edited
    /// entry, by name or through owners).
    pub owned: f64,
}

/// The index of `program`'s operator named `name`, if exactly one is.
fn named(program: &OperatorProgram, name: &str) -> Result<Option<usize>, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(i), None) => Ok(Some(i)),
        (None, _) => Ok(None),
        _ => Err(error(format!("two operators named {name}"))),
    }
}

/// Every node of `program`, its top level and every rule body.
fn every_node(program: &OperatorProgram) -> impl Iterator<Item = &Node> {
    program.nodes.iter().chain(program.rules.iter().flat_map(|r| r.nodes.iter()))
}

/// Whether some node of `program` reads operator `op`.
fn applied(program: &OperatorProgram, op: usize) -> bool {
    every_node(program).any(|node| match node {
        Node::Affine { terms, bias } => terms.iter().any(|t| t.1 == op) || *bias == Some(op),
        Node::Transposed { operator, .. } | Node::Constant { operator } => *operator == op,
        _ => false,
    })
}

/// `op` with `delta` added to its values (a dense operator of the same name, interfaces and
/// provenance).
fn plus(op: &Operator, delta: &Array2<f64>) -> Result<Operator, String> {
    let values = op.matrix() + delta;
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(op.name.clone(), op.rows.clone(), op.cols.clone(), values, precision, op.provenance.clone()).map_err(error)
}

/// The product of an owner's factors, which a compiled edit divides by; a matrix factor is refused
/// (its block of `ΔW` need not lie in the factors' range).
fn scalar_factor(explanation: &Artifact, owner: &Owner) -> Result<f64, String> {
    let mut factor = 1.0;
    for name in owner.left.iter().chain(&owner.right) {
        let op = named(&explanation.program, name)?.ok_or_else(|| error(format!("{}: no factor {name}", owner.operator)))?;
        let value = explanation.program.operators[op].matrix();
        if value.dim() != (1, 1) || value[[0, 0]] == 0.0 {
            return Err(error(format!("{}: the factor {name} is not a nonzero scalar", owner.operator)));
        }
        factor *= value[[0, 0]];
    }
    Ok(factor)
}

/// `M` and `P` with `edits` compiled into both (module note), or `None` where `P` holds no copy of
/// some edited matrix (not applicable). `native` is `M`'s program; edits of one matrix add.
pub fn compile(native: &OperatorProgram, explanation: &Artifact, edits: &[WeightEdit]) -> Result<Option<Compiled>, String> {
    let mut summed: BTreeMap<&str, Array2<f64>> = BTreeMap::new();
    for e in edits {
        match summed.get_mut(e.native.as_str()) {
            Some(total) if total.dim() == e.delta.dim() => *total += &e.delta,
            Some(_) => return Err(error(format!("two edits of {} of different shapes", e.native))),
            None => {
                summed.insert(&e.native, e.delta.clone());
            }
        }
    }
    let mut model = native.clone();
    let mut out = explanation.clone();
    let (mut total, mut taken) = (0.0, 0.0);
    // Per operator of P owning an edited block, its term's values; per use of a summed block
    // (`Owner::uses`), its rule body, input and output nodes, its place there and its values.
    let mut terms: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
    let mut summed_uses: Vec<(String, (usize, usize), (std::ops::Range<usize>, std::ops::Range<usize>), Array2<f64>)> = Vec::new();
    for (name, delta) in &summed {
        let w = named(native, name)?.ok_or_else(|| error(format!("M has no operator {name}")))?;
        let source = &native.operators[w];
        if delta.dim() != (source.rows.width(), source.cols.width()) {
            return Err(error(format!("an edit of {name} of shape {:?}, not {:?}", delta.dim(), (source.rows.width(), source.cols.width()))));
        }
        let edited = Arc::new(plus(source, delta)?);
        model.operators[w] = Arc::clone(&edited);
        let size = delta.iter().map(|v| v * v).sum::<f64>();
        total += size;
        // P's own use of W, by name: the same edited operator (one resident copy for both). An
        // explanation may also own blocks of W elsewhere (M's own block at the attention sink's
        // position, the library's functions at the others): both take the edit.
        let kept = named(&out.program, name)?.filter(|op| applied(&out.program, *op));
        if let Some(op) = kept {
            out.program.operators[op] = edited;
        }
        // P's owners of W's blocks: each distinct record once (a key map several heads read has one
        // record per head, all alike), every entry of W owned at most once.
        let mut records: Vec<&Owner> = Vec::new();
        for o in out.owners.iter().filter(|o| o.native == *name) {
            let same = |r: &&Owner| {
                r.operator == o.operator && r.rows == o.rows && r.cols == o.cols && r.native_rows == o.native_rows && r.native_cols == o.native_cols && r.left == o.left && r.right == o.right && r.transposed == o.transposed && r.uses == o.uses && (o.uses.is_none() || r.body == o.body)
            };
            if !records.iter().any(same) {
                records.push(o);
            }
        }
        if records.is_empty() {
            if kept.is_none() {
                return Ok(None);
            }
            taken += size;
            continue;
        }
        // Every entry of W is owned at most once by P's operators, and computed at most once in each
        // body that sums slices of it (several bodies may each compute it: heads reading one map).
        let mut covered: BTreeMap<Option<&str>, Array2<bool>> = BTreeMap::new();
        for o in &records {
            if o.native_rows.end > delta.nrows() || o.native_cols.end > delta.ncols() {
                return Err(error(format!("{}: an owned block {:?} × {:?} outside {name}", o.operator, o.native_rows, o.native_cols)));
            }
            let place = covered.entry(o.uses.map(|_| o.body.as_str())).or_insert_with(|| Array2::from_elem(delta.dim(), false));
            let mut mask = place.slice_mut(s![o.native_rows.clone(), o.native_cols.clone()]);
            if mask.iter().any(|c| *c) {
                return Err(error(format!("{name}: an entry owned by two blocks of P (a summed decomposition needs its uses recorded, not its blocks)")));
            }
            mask.fill(true);
            if let Some(at) = o.uses {
                if !o.left.is_empty() || !o.right.is_empty() || o.transposed {
                    return Err(error(format!("{}: a summed block's use with factors", o.body)));
                }
                let block = delta.slice(s![o.native_rows.clone(), o.native_cols.clone()]).to_owned();
                summed_uses.push((o.body.clone(), at, (o.rows.clone(), o.cols.clone()), block));
                continue;
            }
            let op = named(&out.program, &o.operator)?.ok_or_else(|| error(format!("P has no operator {}", o.operator)))?;
            // The same block of P standing for another native block at another site (a shared body)
            // cannot take one site's edit through its operator.
            if out.owners.iter().any(|r| r.operator == o.operator && r.rows == o.rows && r.cols == o.cols && (r.native != o.native || r.native_rows != o.native_rows || r.native_cols != o.native_cols)) {
                return Err(error(format!("{}: a block shared by sites that own different native blocks", o.operator)));
            }
            let factor = scalar_factor(&out, o)?;
            let block = delta.slice(s![o.native_rows.clone(), o.native_cols.clone()]).mapv(|v| v / factor);
            let block = if o.transposed { block.t().to_owned() } else { block };
            if block.dim() != (o.rows.len(), o.cols.len()) {
                return Err(error(format!("{}: a block {:?} × {:?} for a native block of {:?}", o.operator, o.rows, o.cols, block.dim())));
            }
            let target = &out.program.operators[op];
            let term = terms.entry(op).or_insert_with(|| Array2::zeros((target.rows.width(), target.cols.width())));
            let mut at = term.slice_mut(s![o.rows.clone(), o.cols.clone()]);
            at += &block;
        }
        let owned: f64 = delta.indexed_iter().filter(|(at, _)| covered.values().any(|c| c[*at])).map(|(_, v)| v * v).sum();
        taken += if kept.is_some() { size } else { owned };
    }
    // Each owning operator's term joins every node applying it, on that node's input.
    for (op, values) in terms {
        let target = Arc::clone(&out.program.operators[op]);
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let term = Operator::dense(format!("edit.{}", target.name), target.rows.clone(), target.cols.clone(), values, precision, Provenance::derived(&[&target.provenance], format!("native edit of {}", target.name))).map_err(error)?;
        out.program.operators.push(Arc::new(term));
        let added = out.program.operators.len() - 1;
        let mut uses = 0;
        let rules = out.program.rules.iter_mut().flat_map(|r| r.nodes.iter_mut());
        for node in out.program.nodes.iter_mut().chain(rules) {
            if let Node::Transposed { operator, .. } | Node::Constant { operator } = node
                && *operator == op
            {
                return Err(error(format!("{}: an owned block read transposed or as a constant is not compiled", target.name)));
            }
            if let Node::Affine { terms, bias } = node {
                if *bias == Some(op) {
                    return Err(error(format!("{}: an edit of a bias is not compiled", target.name)));
                }
                let inputs: Vec<usize> = terms.iter().filter(|t| t.1 == op).map(|t| t.0).collect();
                uses += inputs.len();
                terms.extend(inputs.into_iter().map(|input| (input, added)));
            }
        }
        if uses == 0 {
            return Err(error(format!("{}: owns an edited block but no node applies it", target.name)));
        }
    }
    // Each use of a summed block takes the edit's block as a term of the output node on the input
    // node, in the nodes' own coordinates (the output's rows as its affine terms', the input's as a
    // map reading it in the body has them).
    for (body, (input, output), (rows, cols), block) in summed_uses {
        let rule = out.program.rules.iter().position(|r| r.name == body).ok_or_else(|| error(format!("no rule body {body}")))?;
        let nodes = &out.program.rules[rule].nodes;
        let Some(Node::Affine { terms: written, .. }) = nodes.get(output) else {
            return Err(error(format!("{body}: node {output} is not an affine node")));
        };
        let first = written.first().ok_or_else(|| error(format!("{body}: an empty affine node")))?.1;
        let reader = nodes
            .iter()
            .find_map(|n| if let Node::Affine { terms, .. } = n { terms.iter().find(|t| t.0 == input).map(|t| t.1) } else { None })
            .ok_or_else(|| error(format!("{body}: no map reads node {input}")))?;
        let (row_interface, col_interface) = (out.program.operators[first].rows.clone(), out.program.operators[reader].cols.clone());
        let mut values = Array2::zeros((row_interface.width(), col_interface.width()));
        if rows.end > values.nrows() || cols.end > values.ncols() || block.dim() != (rows.len(), cols.len()) {
            return Err(error(format!("{body}: a block {rows:?} × {cols:?} outside its nodes")));
        }
        values.slice_mut(s![rows, cols]).assign(&block);
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let term = Operator::dense(format!("edit.{body}.{output}"), row_interface, col_interface, values, precision, Provenance::derived(&[], format!("native edit at {body}"))).map_err(error)?;
        out.program.operators.push(Arc::new(term));
        let added = out.program.operators.len() - 1;
        if let Node::Affine { terms: written, .. } = &mut out.program.rules[rule].nodes[output] {
            written.push((input, added));
        }
    }
    out.program.interfaces().map_err(error)?;
    Ok(Some(Compiled { model, explanation: out, owned: if total > 0.0 { taken / total } else { 1.0 } }))
}

/// The native edits that scale one component of `P`'s decomposition by `alpha` (0 removes it):
/// `owners` are the component's blocks (its slices in each map it spans), and per map `W` the edit
/// is `(α − 1) D_W(b)`, `D_W(b)` the sum of the component's blocks at their native places
/// (`Artifact::native_block`). `native` is `M`'s program, which gives each map's shape.
pub fn component(native: &OperatorProgram, explanation: &Artifact, owners: &[Owner], alpha: f64) -> Result<Vec<WeightEdit>, String> {
    let mut out: BTreeMap<String, Array2<f64>> = BTreeMap::new();
    for o in owners {
        let w = named(native, &o.native)?.ok_or_else(|| error(format!("M has no operator {}", o.native)))?;
        let shape = (native.operators[w].rows.width(), native.operators[w].cols.width());
        let block = explanation.native_block(o).map_err(error)?;
        let delta = out.entry(o.native.clone()).or_insert_with(|| Array2::zeros(shape));
        let mut at = delta.slice_mut(s![o.native_rows.clone(), o.native_cols.clone()]);
        if at.dim() != block.dim() {
            return Err(error(format!("{}: a block of {:?} at a native block of {:?}", o.operator, block.dim(), at.dim())));
        }
        at.scaled_add(alpha - 1.0, &block);
    }
    Ok(out.into_iter().map(|(native, delta)| WeightEdit { native, delta }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        interchange::{Batch, Experiment, Interchange},
        library_mdl,
        operator_program::SlotValues,
        run_check::{LayerNodes, layer_nodes, split_sites},
    };
    use gam_gpu::tensor::Device;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    struct Setup {
        native: OperatorProgram,
        blocks: Vec<LayerNodes>,
        explanation: library_mdl::Explanation,
        batch: Batch,
    }

    /// The tiny Qwen3 export (tied embeddings, one key-value head read by both query heads) and its
    /// starting library, an exact copy of `M` whose owners name every q, k, v, gate, up and out block.
    fn setup(tag: &str) -> Setup {
        let dir = crate::test_support::tiny_qwen3_export(tag, 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::scoped(&library_mdl::explanation(&native, &layers).expect("the library"), &[0, 1, 2, 3]).expect("scoped");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let blocks = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        Setup { native, blocks, explanation, batch }
    }

    /// `KL(M_e ‖ P_e)` in bits summed over every token of the three base sequences, `M_e` running
    /// `model` and `P_e` (autonomous) `explanation` (no read patches, so no read variables).
    fn bits(s: &Setup, model: &OperatorProgram, explanation: &Artifact) -> f64 {
        let device = Device::host();
        let ic = Interchange::new(&device, model, &s.blocks, explanation, &s.explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
        let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
        ic.evaluate(&s.batch, &clean, false).expect("evaluate").bits.iter().flatten().sum()
    }

    fn random(rng: &mut StdRng, shape: (usize, usize), scale: f64) -> Array2<f64> {
        Array2::from_shape_fn(shape, |_| scale * (rng.random::<f64>() - 0.5))
    }

    fn shape(program: &OperatorProgram, name: &str) -> (usize, usize) {
        let op = &program.operators[named(program, name).unwrap().unwrap()];
        (op.rows.width(), op.cols.width())
    }

    fn native_of(s: &Setup, site: &str, role: &str) -> String {
        s.explanation.artifact.owners.iter().find(|o| o.site == site && o.role == role).map(|o| o.native.clone()).expect("an owner")
    }

    /// Two edits of one matrix (layer 0's down map, owned column by column by P's MLP): `M` computes
    /// with `W + ΔW₁ + ΔW₂`, the compile equals that of the summed edit, `P` (an exact copy) scores
    /// 0 bits against `M_e` (1e-9), and the edit moves `M` (`P` left unedited scores above 1e-4).
    #[test]
    fn two_edits_of_one_matrix_add_and_an_exact_copy_takes_them() {
        let s = setup("weight_edit_two");
        let mut rng = StdRng::seed_from_u64(3);
        let down = native_of(&s, "library.l0.mlp", "out");
        let (d1, d2) = (random(&mut rng, shape(&s.native, &down), 1.0), random(&mut rng, shape(&s.native, &down), 1.0));
        let two = [WeightEdit { native: down.clone(), delta: d1.clone() }, WeightEdit { native: down.clone(), delta: d2.clone() }];
        let compiled = compile(&s.native, &s.explanation.artifact, &two).expect("compiles").expect("applicable");
        let summed = compile(&s.native, &s.explanation.artifact, &[WeightEdit { native: down.clone(), delta: &d1 + &d2 }]).expect("compiles").expect("applicable");
        let w = named(&s.native, &down).unwrap().unwrap();
        let expected = s.native.operators[w].matrix() + &d1 + &d2;
        assert!((compiled.model.operators[w].matrix() - &expected).iter().all(|v| v.abs() <= 1e-12), "M computes with W + ΔW₁ + ΔW₂");
        assert_eq!(compiled.model.operators[w].matrix(), summed.model.operators[w].matrix());
        assert!((compiled.owned - 1.0).abs() < 1e-12, "P owns every column of the down map");
        let edited = bits(&s, &compiled.model, &compiled.explanation);
        assert!(edited.abs() <= 1e-9, "the exact copy scores {edited} bits under the edit");
        assert!((bits(&s, &summed.model, &summed.explanation) - edited).abs() <= 1e-9);
        let ignored = bits(&s, &compiled.model, &s.explanation.artifact);
        assert!(ignored > 1e-4, "the edit moves M: P unedited scores {ignored}");
    }

    /// Tied weights: the embedding `wte` (read again, transposed, by the readout) and the key map
    /// the two query heads of layer 1 share (one owner per head, one operator of P). Each model
    /// computes with the edited matrix at every use, and the exact copy scores 0 (1e-9). The key
    /// map's edit moves `M`; the embedding's changes the readout the two models share, so `P` left
    /// unedited is no longer comparable (the experiments refuse two heads), and `P` holds `M`'s one
    /// edited operator instead.
    #[test]
    fn an_edit_of_a_tied_weight_reaches_every_use() {
        let s = setup("weight_edit_tied");
        let mut rng = StdRng::seed_from_u64(5);
        let key = native_of(&s, "library.l1.h0", "k");
        assert_eq!(key, native_of(&s, "library.l1.h1", "k"), "both query heads read one key map");
        for name in ["wte".to_string(), key] {
            let edit = [WeightEdit { native: name.clone(), delta: random(&mut rng, shape(&s.native, &name), 0.5) }];
            let compiled = compile(&s.native, &s.explanation.artifact, &edit).expect("compiles").expect("applicable");
            let edited = bits(&s, &compiled.model, &compiled.explanation);
            assert!(edited.abs() <= 1e-9, "{name}: the exact copy scores {edited} bits under the edit");
            if name == "wte" {
                let (m, p) = (named(&compiled.model, &name).unwrap().unwrap(), named(&compiled.explanation.program, &name).unwrap().unwrap());
                assert!(Arc::ptr_eq(&compiled.model.operators[m], &compiled.explanation.program.operators[p]), "P holds M's edited embedding");
                assert!(every_node(&compiled.model).filter(|n| matches!(n, Node::Transposed { operator, .. } if *operator == m) || matches!(n, Node::Affine { terms, .. } if terms.iter().any(|t| t.1 == m))).count() == 2, "M reads the embedding twice");
            } else {
                let ignored = bits(&s, &compiled.model, &s.explanation.artifact);
                assert!(ignored > 1e-4, "{name}: the edit moves M ({ignored})");
            }
        }
    }

    /// A summed block's use (`Owner::uses`): layer 1's down map recorded not by its owning operator
    /// but by its use in the MLP's body (the gated activations, node 4, read into the output, node
    /// 5), as a decomposition into slices records it. The edit joins the output node as a term on
    /// the activations, and the exact copy scores 0 bits (1e-9) while the edit moves `M`.
    #[test]
    fn an_edit_reaches_a_summed_block_through_its_use() {
        let s = setup("weight_edit_use");
        let mut rng = StdRng::seed_from_u64(9);
        let site = "library.l1.mlp";
        let down = native_of(&s, site, "out");
        let (d, n) = shape(&s.native, &down);
        let program = &s.explanation.artifact.program;
        let body = &program.rules[program.rules.iter().position(|r| r.name == site).expect("the MLP's body")];
        assert!(matches!(&body.nodes[5], Node::Affine { terms, .. } if terms.iter().any(|t| t.0 == 4)), "node 5 writes the activations of node 4");
        let mut explanation = s.explanation.artifact.clone();
        explanation.owners.retain(|o| !(o.site == site && o.role == "out"));
        explanation.owners.push(Owner { rows: 0..d, cols: 0..n, body: site.into(), site: site.into(), native: down.clone(), native_rows: 0..d, native_cols: 0..n, role: "out".into(), uses: Some((4, 5)), ..Owner::default() });
        let edit = [WeightEdit { native: down.clone(), delta: random(&mut rng, (d, n), 1.0) }];
        let compiled = compile(&s.native, &explanation, &edit).expect("compiles").expect("applicable");
        assert!((compiled.owned - 1.0).abs() < 1e-12);
        let edited = bits(&s, &compiled.model, &compiled.explanation);
        assert!(edited.abs() <= 1e-9, "the exact copy scores {edited} bits under the edit");
        let ignored = bits(&s, &compiled.model, &s.explanation.artifact);
        assert!(ignored > 1e-4, "the edit moves M ({ignored})");
    }

    /// The tiny export decomposed exactly into gated slices (`library_vpd`; VPD's factors `V = I`,
    /// `U = Wᵀ` for every map, every gate always on): per layer one component at the attention's
    /// input (every q, k, v and o slice) and one at the MLP's input (every c_fc and down slice), or
    /// with `split` two of each listing each other as candidates (gate sharing).
    fn exact_slices(tag: &str, split: bool) -> (Setup, Vec<LayerNodes>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("export.json")).expect("export.json")).expect("the record");
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        std::fs::create_dir_all(&factors).expect("the factors' directory");
        let (mut files, mut sites) = (serde_json::Map::new(), Vec::new());
        let mut write = |name: String, values: Array2<f64>| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(factors.join(format!("{name}.f64")), bytes).expect("a factor");
            files.insert(name, serde_json::json!({"shape": [values.nrows(), values.ncols()]}));
        };
        let mut components = Vec::new();
        for l in 0..2 {
            let mut slices = [Vec::new(), Vec::new()];
            for (k, name) in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"].into_iter().enumerate() {
                let shape = &record["files"][format!("blocks.{l}.{name}")]["shape"];
                let (r, c) = (shape[0].as_u64().expect("rows") as usize, shape[1].as_u64().expect("columns") as usize);
                let w = crate::import::read_f64_shaped(&dir.join(format!("blocks.{l}.{name}.f64")), r, c).expect("a map");
                let site = format!("h.{l}.{name}");
                write(format!("{site}.U"), w.t().to_owned());
                write(format!("{site}.V"), Array2::eye(c));
                sites.push(site);
                slices[usize::from(k >= 4)].extend((0..c).map(|i| [6 * l + k, i]));
            }
            // Width 10⁻³: the gate Φ(z/w) at z ≥ 1 is 1 in float64, so the decomposition is exact
            // (a shared own gate's width on the squared norm is 2·10⁻⁶, its threshold −1).
            let [attention, mlp] = slices;
            if split {
                // Per stage two components, each listing the other: q, k and o slices with v; the
                // first half of c_fc and of down with the second.
                let (a1, a2): (Vec<[usize; 2]>, Vec<[usize; 2]>) = attention.into_iter().partition(|[site, _]| site % 6 != 2);
                let (m1, m2): (Vec<[usize; 2]>, Vec<[usize; 2]>) = mlp.into_iter().partition(|[site, i]| *i < if site % 6 == 4 { 4 } else { 8 });
                let first = components.len();
                for (k, (read, slices)) in [(0, a1), (2, a2), (4, m1), (4, m2)].into_iter().enumerate() {
                    let other = first + (k ^ 1);
                    components.push(serde_json::json!({"read": {"own": [6 * l + read, 0]}, "tau": -1.0, "width": 1e-3, "slices": slices, "candidates": [other]}));
                }
            } else {
                for (stage, slices) in [attention, mlp].into_iter().enumerate() {
                    components.push(serde_json::json!({"read": {"own": [6 * l + 4 * stage, 0]}, "tau": -1.0, "width": 1e-3, "slices": slices}));
                }
            }
        }
        std::fs::write(factors.join("export.json"), serde_json::json!({"config": {"sites": sites}, "files": files}).to_string()).expect("the factors' record");
        let start = factors.join("start.json");
        std::fs::write(&start, serde_json::json!([{"arm": "exact", "components": components}]).to_string()).expect("the start");
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = crate::library_vpd::explanation(&native, &layers, &factors, &start, "exact").expect("the decomposition");
        std::fs::remove_dir_all(&factors).expect("the factors are removed");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let blocks = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let s = Setup { native, blocks, explanation, batch };
        (s, layers)
    }

    /// Gate sharing (`library_vpd`, components that list candidates): on the exact decomposition
    /// with two components per stage, each on its own gate, `P` scores 0 bits, and each shared
    /// stage's record lists its components' candidates, own first. With the second component's
    /// gate shut (threshold −10⁹ on the squared norm) and that component assigned to it, `P` loses
    /// the component; assigning it to the first component's gate (a part of their summed rank,
    /// gated on the norm of both reads) makes `P` exact again.
    #[test]
    fn a_shared_stage_moves_components_between_gates() {
        let (s, _) = exact_slices("weight_edit_share", true);
        assert_eq!(s.explanation.shares.len(), 4, "two shared stages per layer");
        assert!(s.explanation.shares.iter().all(|share| share.candidates == vec![vec![0, 1], vec![1, 0]]), "{:?}", s.explanation.shares);
        let exact = bits(&s, &s.native, &s.explanation.artifact);
        assert!(exact.abs() <= 1e-9, "the shared decomposition scores {exact} bits");
        let set = |artifact: &mut Artifact, name: &str, values: Array2<f64>| {
            let at = named(&artifact.program, name).unwrap().expect("an operator");
            let old = Arc::clone(&artifact.program.operators[at]);
            let precision = exact_precision(values.iter().copied()).expect("a precision");
            artifact.program.operators[at] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).expect("an operator"));
        };
        let mut shut = s.explanation.artifact.clone();
        set(&mut shut, "library.l1.mlp.fc.threshold", ndarray::array![[-1.0], [-1e9]]);
        let lost = bits(&s, &s.native, &shut);
        assert!(lost > 1e-4, "the second component on its shut gate is off ({lost} bits)");
        set(&mut shut, "library.l1.mlp.fc.assign", ndarray::array![[1.0, 1.0], [0.0, 0.0]]);
        let moved = bits(&s, &s.native, &shut);
        assert!(moved.abs() <= 1e-9, "both components on the first gate: {moved} bits");
    }

    /// An exact decomposition into gated slices (`library_vpd` on the tiny export, VPD's factors set
    /// to every column slice of each map, `V = I` and `U = Wᵀ`, one component per layer at the
    /// attention's input carrying every q, k, v and o slice and one at the MLP's input carrying every
    /// c_fc and down slice, each always on): `P` computes `M` (0 bits, 1e-9), and no operator of `P`
    /// holds a block of `M`'s maps. Native edits compile through the maps' uses
    /// (`library_vpd::uses`): edits of layer 1's head-0 query, head 1's block of o, c_fc and down each
    /// score 0 bits on the decomposition and move `M`; without the uses each is not applicable.
    #[test]
    fn edits_reach_an_exact_gated_slice_decomposition_through_its_uses() {
        let (s, layers) = exact_slices("weight_edit_vpd", false);
        let exact = bits(&s, &s.native, &s.explanation.artifact);
        assert!(exact.abs() <= 1e-9, "the decomposition scores {exact} bits");
        let mut artifact = s.explanation.artifact.clone();
        artifact.owners = crate::library_vpd::uses(&s.native, &layers, &artifact).expect("its uses");
        assert!(!artifact.owners.is_empty() && artifact.owners.iter().all(|o| o.uses.is_some() && o.operator.is_empty()));
        let mut rng = StdRng::seed_from_u64(13);
        for (body, role) in [("library.l1.h0", "q"), ("library.l1.o", "o"), ("library.l1.mlp", "gate"), ("library.l1.mlp", "out")] {
            let owners: Vec<&Owner> = artifact.owners.iter().filter(|o| o.body == body && o.role == role).collect();
            let native = owners.last().expect("a use").native.clone();
            let edit = [WeightEdit { native: native.clone(), delta: random(&mut rng, shape(&s.native, &native), 1.0) }];
            assert!(compile(&s.native, &s.explanation.artifact, &edit).expect("compiles").is_none(), "{native}: not applicable without its uses");
            let compiled = compile(&s.native, &artifact, &edit).expect("compiles").expect("applicable");
            assert!((compiled.owned - 1.0).abs() < 1e-12, "{native}: P takes the whole edit");
            let edited = bits(&s, &compiled.model, &compiled.explanation);
            assert!(edited.abs() <= 1e-9, "{native}: the decomposition scores {edited} bits under the edit");
            let ignored = bits(&s, &compiled.model, &s.explanation.artifact);
            assert!(ignored > 1e-4, "{native}: the edit moves M ({ignored})");
        }
    }

    /// A component spanning the gate, up and out maps of layer 1's MLP (units 2, 5 and 11), removed
    /// (α = 0) and doubled (α = 2): on `M` the matching slices of the three maps, on `P` its own
    /// blocks. The exact copy scores 0 bits (1e-9), the edit moves `M`, and an explanation whose
    /// owners name no block of the down map reports the down map's edit as not applicable.
    #[test]
    fn a_cross_matrix_component_edit_scores_zero_on_an_exact_copy() {
        let s = setup("weight_edit_component");
        let units = [2usize, 5, 11];
        let site = "library.l1.mlp";
        let owners: Vec<Owner> = s
            .explanation
            .artifact
            .owners
            .iter()
            .filter(|o| o.site == site && units.iter().any(|u| if o.role == "out" { o.cols == (*u..u + 1) } else { o.rows == (*u..u + 1) }))
            .cloned()
            .collect();
        let roles: std::collections::BTreeSet<&str> = owners.iter().map(|o| o.role.as_str()).collect();
        assert_eq!(roles.into_iter().collect::<Vec<_>>(), vec!["gate", "out", "up"], "the component spans three maps");
        for alpha in [0.0, 2.0] {
            let edits = component(&s.native, &s.explanation.artifact, &owners, alpha).expect("the component's edits");
            assert_eq!(edits.len(), 3);
            let compiled = compile(&s.native, &s.explanation.artifact, &edits).expect("compiles").expect("applicable");
            let edited = bits(&s, &compiled.model, &compiled.explanation);
            assert!(edited.abs() <= 1e-9, "α = {alpha}: the exact copy scores {edited} bits");
            let ignored = bits(&s, &compiled.model, &s.explanation.artifact);
            assert!(ignored > 1e-4, "α = {alpha}: the edit moves M ({ignored})");
        }
        let down = native_of(&s, site, "out");
        let mut unowned = s.explanation.artifact.clone();
        unowned.owners.retain(|o| o.native != down);
        let edit = [WeightEdit { native: down.clone(), delta: Array2::ones(shape(&s.native, &down)) }];
        assert!(compile(&s.native, &unowned, &edit).expect("compiles").is_none(), "no copy of the down map: not applicable");
    }
}
