#![cfg(test)]
//! Interchange experiments on the device against an independent host computation, and their
//! gradient against central differences, on a small rotary language model (two layers, tied
//! unembedding) whose explanation `P` is the model with perturbed query, key, value and MLP input
//! maps.
//!
//! The host reference runs each model's program with `OperatorProgram::execute_edited`: a hybrid
//! runs block by block, each block's model entering at the stream the previous block left; a patch
//! writes the source run's entries of the patched variables' nodes into the base run's row (not
//! the device's masked rows); and the divergence is formed from full softmax distributions (not
//! the compact head statistics). Both sides evaluate the same expressions in float64 in different orders; each
//! value then differs by a few units of rounding of its widest sum (at most `d + V` terms here,
//! `V` classes), far below the `1e-9` bits the comparison allows and far below any effect of a
//! patch.

use super::artifact::Artifact;
use super::device_program::DeviceProgram;
use super::device_program_tests::{devices, fixture_sized, noise};
use super::interchange::{Batch, Experiment, FixedHead, Interchange, Model, Patch, ReadVariable, Site, Value, census, evaluate, reads, sample, sites, targets, values};
use super::interchange::{BlockEngine, Edit, Edits, FACTORS, Part};
use super::device_program::DeviceTrace;
use super::operator_program::{FamilyInputs, Operator, OperatorProgram, SequenceLayout, SlotValues, exact_precision};
use super::resident_causal_fit::fixed_head_target::Head;
use super::run_check::{layer_nodes, split_sites};
use gam_gpu::tensor::Device;
use gam_math::categorical::categorical_kl_from_logits;
use gam_linalg::decompose::{QrMode, qr};
use ndarray::Array2;
use rand::SeedableRng;
use std::sync::Arc;

const D: usize = 24;
const VOCAB: usize = 13;
const LENGTH: usize = 6;
const LAYERS: usize = 2;

/// One model's flat program through its hidden node and its sites.
struct Host {
    prefix: OperatorProgram,
    flat: OperatorProgram,
    entries: Vec<usize>,
    reads: Vec<usize>,
}

/// The native model `M`, its explanation `P` (perturbed maps), the read variables, where both
/// models hold their values (the same nodes: `P` is `M`'s program with other values), `P`'s
/// trainable operators, and the shared head.
struct Fixture {
    m: Host,
    p: Host,
    variables: Vec<ReadVariable>,
    values: Vec<Value>,
    trainable: Vec<usize>,
    head: Head,
    batch: Batch,
}

fn fixture() -> Fixture {
    fixture_at(D)
}

/// [`fixture`] at stream width `width`.
fn fixture_at(width: usize) -> Fixture {
    let (program, family) = fixture_sized(width, VOCAB, LENGTH, 6);
    let native = split_sites(&program).expect("split");
    let layers = layer_nodes(&native, LAYERS).expect("layers");
    let (flat, entries, reads) = sites(&Artifact::native(&native).expect("native artifact"), &layers).expect("sites");
    let head = Head::of(&flat).expect("head");
    let m = Host { prefix: head.prefix(&flat), flat: flat.clone(), entries: entries.clone(), reads: reads.clone() };
    // P: every query, key, value and MLP input map moved off M's.
    let mut p_flat = flat.clone();
    let mut variables = Vec::new();
    let mut trainable = Vec::new();
    for (index, op) in flat.operators.iter().enumerate() {
        let Some(rest) = op.name.strip_prefix("blocks.") else { continue };
        let (layer, part) = rest.split_once('.').expect("layer.part");
        let layer: usize = layer.parse().expect("layer");
        let block = match part.chars().next() {
            Some('q' | 'k' | 'v') if part[1..].parse::<usize>().is_ok() => 2 * layer,
            _ if part == "c_fc" => 2 * layer + 1,
            _ => continue,
        };
        let values = op.matrix();
        let moved = Array2::from_shape_fn(values.dim(), |(i, j)| values[[i, j]] * (1.0 + 0.3 * noise(7919 * index + 31 * i + j)));
        let precision = exact_precision(moved.iter().copied()).expect("finite");
        p_flat.operators[index] = Arc::new(Operator::dense(op.name.clone(), op.rows.clone(), op.cols.clone(), moved, precision, op.provenance.clone()).expect("dense"));
        trainable.push(index);
        if block % 2 == 0 {
            variables.push(ReadVariable { block, parts: vec![(index, 0..op.rows.width())] });
        } else {
            variables.extend((0..op.rows.width()).map(|i| ReadVariable { block, parts: vec![(index, i..i + 1)] }));
        }
    }
    // A variable read through two operator parts, as a gated MLP function's is.
    let up = flat.operators.iter().position(|op| op.name == "blocks.0.c_fc").expect("layer 0 MLP input");
    variables.push(ReadVariable { block: 1, parts: vec![(up, 1..2), (up, 4..6)] });
    let p = Host { prefix: head.prefix(&p_flat), flat: p_flat, entries, reads };
    let held = values(&native, &Artifact::native(&native).expect("native artifact"), &layers, &variables).expect("the variables' values");
    let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("tokens") };
    let base: Vec<Vec<u32>> = tokens.chunks(LENGTH).map(<[u32]>::to_vec).collect();
    let source: Vec<Vec<u32>> = base.iter().map(|s| s.iter().map(|t| (t + 5) % VOCAB as u32).rev().collect()).collect();
    Fixture { m, p, variables, values: held, trainable, head, batch: Batch::new(base, source).expect("batch") }
}

fn one(tokens: &[u32]) -> FamilyInputs {
    FamilyInputs {
        rows: tokens.len(),
        slots: vec![SlotValues::Tokens(tokens.to_vec())],
        layout: Some(SequenceLayout { sequence: vec![0; tokens.len()], position: (0..tokens.len() as u32).collect() }),
    }
}

/// A patch as the reference applies it: its block, the sites of its variables, the source run's
/// values of that block (every node), and the row it replaces.
type HostPatch<'a> = (usize, &'a [Site], &'a [Array2<f64>], usize);

/// The hybrid running `P`'s version of the blocks `explained` marks and `M`'s of the others, on
/// `tokens`, under `patch`: its hidden rows and every node's value of each block's run. Each block
/// is one run of its model's whole program with the stream entering the block replaced by the
/// previous block's output, of which only the block's own nodes are kept; a patched node's entries
/// at the patched row take the source run's.
fn hybrid(f: &Fixture, tokens: &[u32], explained: &[bool], patch: Option<HostPatch<'_>>) -> (Array2<f64>, Vec<Vec<Array2<f64>>>) {
    let family = one(tokens);
    let blocks = 2 * LAYERS;
    let (mut state, mut runs) = (None::<Array2<f64>>, Vec::with_capacity(blocks));
    for b in 0..blocks {
        let host = if explained[b] { &f.p } else { &f.m };
        let end = if b + 1 < blocks { host.entries[b + 1] } else { host.prefix.output };
        let run = host
            .prefix
            .execute_edited(&family, |node, value, _| {
                if let Some(entering) = &state
                    && node == host.entries[b]
                {
                    *value = entering.clone();
                }
                if let Some((block, sites, source, at)) = patch
                    && block == b
                {
                    for site in sites.iter().filter(|s| s.node == node) {
                        for c in site.columns.clone() {
                            value[[at, c]] = source[node][[at, c]];
                        }
                    }
                }
                Ok(())
            })
            .expect("block");
        state = Some(run.values[end].clone());
        runs.push(run.values);
    }
    (state.expect("a block ran"), runs)
}

/// `M` alone (the hybrid with no block of `P`).
fn native(f: &Fixture, tokens: &[u32], patch: Option<HostPatch<'_>>) -> (Array2<f64>, Vec<Vec<Array2<f64>>>) {
    hybrid(f, tokens, &[false; 2 * LAYERS], patch)
}

/// Per row `KL(softmax(E a) ‖ softmax(E b))` in bits.
fn kl_bits(head: &Head, a: &Array2<f64>, b: &Array2<f64>) -> Vec<f64> {
    let (za, zb) = (a.dot(&head.embedding().t()), b.dot(&head.embedding().t()));
    za.outer_iter()
        .zip(zb.outer_iter())
        .map(|(ra, rb)| categorical_kl_from_logits(&ra.to_vec(), &rb.to_vec()).expect("finite logits") / std::f64::consts::LN_2)
        .collect()
}

/// The reference bits of experiment `e`.
fn reference(f: &Fixture, e: &Experiment) -> Vec<f64> {
    let (base, source) = (&f.batch.base[e.base], &f.batch.source[e.source]);
    let Some(patch) = &e.patch else {
        return kl_bits(&f.head, &native(f, base, None).0, &hybrid(f, base, &e.explained, None).0)[e.position..].to_vec();
    };
    let block = f.variables[patch.variables()[0]].block;
    let sites: Vec<Site> = patch.variables().iter().flat_map(|v| f.values[*v].sites.clone()).collect();
    let (m_source, p_source) = (native(f, source, None).1, hybrid(f, source, &e.explained, None).1);
    let m = native(f, base, Some((block, &sites, &m_source[block], e.position))).0;
    let p = hybrid(f, base, &e.explained, Some((block, &sites, &p_source[block], e.position))).0;
    kl_bits(&f.head, &m, &p)[e.position..].to_vec()
}

/// Experiments covering every kind of patch on both sides of the hybrids' switches, under prefix
/// and non-prefix hybrids.
fn experiments(f: &Fixture) -> Vec<Experiment> {
    let read = |block: usize, nth: usize| f.variables.iter().enumerate().filter(|(_, v)| v.block == block).nth(nth).map(|(i, _)| i).expect("variable");
    let (t, n) = (true, false);
    let e = |base: usize, source: usize, explained: [bool; 2 * LAYERS], patch: Option<Patch>, position: usize| Experiment { base, source, explained: explained.to_vec(), patch, position };
    vec![
        e(0, 0, [t, t, n, n], None, 0),
        e(1, 1, [t, t, t, t], None, 0),
        e(4, 4, [n, t, t, n], None, 2),
        e(5, 5, [n, n, n, t], None, 0),
        e(2, 3, [t, t, n, n], Some(Patch::Read { variable: read(0, 0) }), 0),
        e(3, 2, [t, t, t, t], Some(Patch::Read { variable: read(2, 2) }), 3),
        e(4, 5, [t, t, n, n], Some(Patch::Read { variable: read(3, 5) }), 5),
        e(5, 4, [t, n, t, t], Some(Patch::Read { variable: read(1, 3) }), 1),
        e(1, 2, [n, t, n, t], Some(Patch::Read { variable: read(2, 1) }), 2),
        e(3, 0, [t, t, n, n], Some(Patch::Read { variable: f.variables.len() - 1 }), 4),
        // Joint read patches of several variables of one block.
        e(2, 4, [t, t, n, n], Some(Patch::Reads { variables: vec![read(3, 0), read(3, 2), read(3, 7)] }), 2),
        e(5, 3, [n, t, t, n], Some(Patch::Reads { variables: vec![read(2, 0), read(2, 2)] }), 1),
        e(0, 4, [t, t, t, t], Some(Patch::Reads { variables: vec![read(1, 0), read(1, 1), read(1, 6)] }), 0),
        e(1, 5, [n, n, t, t], Some(Patch::Reads { variables: vec![read(0, 1), read(0, 3)] }), 3),
    ]
}

struct Programs {
    m: DeviceProgram,
    p: DeviceProgram,
    head: FixedHead,
}

fn programs(device: &Device, f: &Fixture) -> Programs {
    let m = DeviceProgram::compile_values(device, &f.m.prefix).expect("M");
    let mut p = DeviceProgram::compile_values(device, &f.p.prefix).expect("P");
    p.prepare_dense_parameters(&f.trainable).expect("parameters");
    let head = FixedHead::new(device, &f.m.flat, &f.p.flat, 5).expect("head");
    Programs { m, p, head }
}

fn models<'a>(f: &Fixture, programs: &'a Programs) -> (Model<'a>, Model<'a>) {
    (
        Model::new(&programs.m, &f.m.flat, f.m.entries.clone(), f.m.reads.clone(), &[], f.values.clone()).expect("M model"),
        Model::new(&programs.p, &f.p.flat, f.p.entries.clone(), f.p.reads.clone(), &f.trainable, f.values.clone()).expect("P model"),
    )
}

#[test]
fn every_experiment_matches_the_host_reference() {
    let f = fixture();
    let experiments = experiments(&f);
    for device in devices() {
        let programs = programs(&device, &f);
        let (m, p) = models(&f, &programs);
        let targets = targets(&m, &programs.head, &f.batch, &experiments).expect("targets");
        let evaluation = evaluate(&m, &p, &programs.head, &f.batch, &targets, &experiments, false).expect("evaluate");
        for (e, bits) in experiments.iter().zip(&evaluation.bits) {
            let expected = reference(&f, e);
            assert_eq!(bits.len(), LENGTH - e.position);
            for (a, b) in bits.iter().zip(&expected) {
                assert!((a - b).abs() <= 1e-9, "{e:?}: device {a} bits, host {b} bits");
            }
            if e.patch.is_some() {
                assert!(expected.iter().any(|v| *v > 1e-6), "{e:?}: the patch has no visible effect");
            }
        }
    }
}

#[test]
fn the_model_explains_itself_exactly() {
    // With P = M every hybrid is M and every patch is the same patch: no divergence, no gradient.
    let f = fixture();
    let device = Device::host();
    let mut p = DeviceProgram::compile_values(&device, &f.m.prefix).expect("P");
    p.prepare_dense_parameters(&f.trainable).expect("parameters");
    let m = DeviceProgram::compile_values(&device, &f.m.prefix).expect("M");
    let head = FixedHead::new(&device, &f.m.flat, &f.m.flat, 4).expect("head");
    let mm = Model::new(&m, &f.m.flat, f.m.entries.clone(), f.m.reads.clone(), &[], f.values.clone()).expect("M model");
    let pm = Model::new(&p, &f.m.flat, f.m.entries.clone(), f.m.reads.clone(), &f.trainable, f.values.clone()).expect("P model");
    let experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(5), f.batch.base.len(), &f.variables, 2 * LAYERS, LENGTH).expect("sample");
    let targets = targets(&mm, &head, &f.batch, &experiments).expect("targets");
    let evaluation = evaluate(&mm, &pm, &head, &f.batch, &targets, &experiments, true).expect("evaluate");
    assert!(evaluation.bits.iter().flatten().all(|b| b.abs() <= 1e-12), "{:?}", evaluation.bits);
    for g in evaluation.gradient.values() {
        assert!(device.download(g).expect("gradient").iter().all(|v| v.abs() <= 1e-9));
    }
}

/// The total bits at `P`'s operator `op` moved to `value`.
fn total_at(device: &Device, f: &Fixture, programs: &mut Programs, op: usize, value: &Array2<f64>, experiments: &[Experiment]) -> f64 {
    programs.p.replace_dense_parameter(op, device.upload(value.view()).expect("upload")).expect("replace");
    let (m, p) = models(f, programs);
    let targets = targets(&m, &programs.head, &f.batch, experiments).expect("targets");
    let evaluation = evaluate(&m, &p, &programs.head, &f.batch, &targets, experiments, false).expect("evaluate");
    evaluation.bits.iter().flatten().sum()
}

#[test]
fn the_gradient_matches_central_differences() {
    // Central differences with step `δ` err by `O(δ²)` (the third derivative) plus `O(u/δ)`
    // rounding of the total; `δ = 1e-5` balances the two near `u^{1/3}`, and both stay below the
    // `1e-6` relative band the comparison allows on these O(1) totals.
    let f = fixture();
    let device = Device::host();
    let experiments = experiments(&f);
    let mut programs = programs(&device, &f);
    let gradient = {
        let (m, p) = models(&f, &programs);
        let targets = targets(&m, &programs.head, &f.batch, &experiments).expect("targets");
        evaluate(&m, &p, &programs.head, &f.batch, &targets, &experiments, true).expect("evaluate").gradient
    };
    let delta = 1e-5;
    for (k, &op) in f.trainable.iter().enumerate() {
        let start = f.p.flat.operators[op].matrix();
        let (rows, cols) = start.dim();
        let (i, j) = ((7 * k + 1) % rows, (3 * k + 2) % cols);
        let analytic = device.download(&gradient[&op]).expect("gradient")[[i, j]];
        let mut up = start.clone();
        up[[i, j]] += delta;
        let mut down = start.clone();
        down[[i, j]] -= delta;
        let numeric = (total_at(&device, &f, &mut programs, op, &up, &experiments) - total_at(&device, &f, &mut programs, op, &down, &experiments)) / (2.0 * delta);
        total_at(&device, &f, &mut programs, op, &start, &experiments);
        let scale = analytic.abs().max(numeric.abs()).max(1.0);
        assert!((analytic - numeric).abs() <= 1e-6 * scale, "operator {} entry ({i}, {j}): analytic {analytic}, numeric {numeric}", f.p.flat.operators[op].name);
    }
}

#[test]
fn sampling_draws_each_family_with_its_stated_weight() {
    // Six blocks of three read variables each. Per base: P alone with probability 1/2; the block
    // uniform; then one variable (probability 1/2) or a joint patch of a subset whose size is
    // uniform in 1..=3.
    let variables: Vec<ReadVariable> = (0..6).flat_map(|b| (0..3).map(move |i| ReadVariable { block: b, parts: vec![(b, i..i + 1)] })).collect();
    let n = 6000;
    let experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(11), n, &variables, 6, 9).expect("sample");
    assert_eq!(experiments.len(), 2 * n);
    let mut joint_sizes = [0usize; 4];
    for (i, pair) in experiments.chunks(2).enumerate() {
        assert_eq!((pair[0].base, pair[1].base, pair[1].source), (i, i, i));
        assert!(pair[0].patch.is_none() && pair[1].patch.is_some());
        // The patched experiment shares its base's hybrid; the clean one is scored everywhere.
        assert_eq!(pair[0].explained, pair[1].explained);
        assert!(pair[0].position == 0 && pair[1].position < 9);
        assert!(pair[0].explained.len() == 6 && pair[0].explained.contains(&true));
        if let Some(Patch::Reads { variables: chosen }) = &pair[1].patch {
            let block = variables[chosen[0]].block;
            assert!(!chosen.is_empty() && chosen.windows(2).all(|w| w[0] < w[1]) && chosen.iter().all(|v| variables[*v].block == block));
            joint_sizes[chosen.len()] += 1;
        }
    }
    // Each count within five standard deviations of its binomial expectation.
    let within = |count: f64, trials: f64, p: f64| {
        let (mean, sd) = (trials * p, (trials * p * (1.0 - p)).sqrt());
        assert!((count - mean).abs() <= 5.0 * sd, "{count} against {mean} ± {sd}");
    };
    let counts = census(&experiments, &variables);
    // A hybrid draw is P alone with probability 1/6 (its size uniform in 1..=6).
    within(counts["clean_alone"] as f64, n as f64, 0.5 + 0.5 / 6.0);
    within(counts["read_attention"] as f64, n as f64, 0.25);
    within(counts["read_mlp"] as f64, n as f64, 0.25);
    within(counts["read_joint"] as f64, n as f64, 0.5);
    // A joint subset of three: each size 1, 2, 3 with probability 1/3.
    let joints = counts["read_joint"] as f64;
    assert_eq!(joint_sizes[0], 0);
    for size in 1..=3 {
        within(joint_sizes[size] as f64, joints, 1.0 / 3.0);
    }
    // Every size 1..=6 appears, and a non-prefix set does: the hybrids are not only cuts.
    let sizes: std::collections::BTreeSet<usize> = experiments.iter().map(|e| e.explained.iter().filter(|x| **x).count()).collect();
    assert_eq!(sizes, (1..=6).collect());
    assert!(experiments.iter().any(|e| e.explained.windows(2).any(|w| !w[0] && w[1])));
    let positions: std::collections::BTreeSet<usize> = experiments.iter().filter(|e| e.patch.is_some()).map(|e| e.position).collect();
    assert_eq!(positions, (0..9).collect());
    // A block without a read variable is never patched (block 0's variables left out).
    let partial = sample(&mut rand::rngs::StdRng::seed_from_u64(11), 600, &variables[3..], 6, 9).expect("sample");
    assert_eq!(partial.len(), 1200);
    assert!(partial.iter().filter_map(|e| e.patch.as_ref()).flat_map(|p| p.variables()).all(|v| variables[*v + 3].block > 0));
    // No block holding a read variable: only the clean experiments.
    let clean = sample(&mut rand::rngs::StdRng::seed_from_u64(11), 5, &[], 6, 9).expect("sample");
    assert!(clean.len() == 5 && clean.iter().all(|e| e.patch.is_none()));
}

#[test]
fn many_dropped_reads_that_matter_little_alone_matter_jointly() {
    // M's last MLP gets n functions whose gate rows are one row g₀ putting every base read at -L
    // and the source's read at the patched position at +L (GELU leaves them off on the base, on at
    // the source), and whose outputs are a small common ε u (u the stream's first coordinate). P
    // drops them, so only M's values change under their patches. A single patch turns one function
    // on (output change ε L u), the joint patch all n (n ε L u): to second order the joint
    // divergence is n² times a single one, while the base never sees them.
    let f = fixture();
    let (n, epsilon, at, big) = (8, 1e-4, 3, 10.0);
    let host = Device::host();
    let base = f.batch.base[0].clone();
    let source = f.batch.source[1].clone();
    let reads = native(&f, &base, None).1[3][f.m.reads[3]].clone();
    let source_read = native(&f, &source, None).1[3][f.m.reads[3]].row(at).to_owned();
    let mut rows = reads.clone();
    rows.push_row(source_read.view()).expect("rows");
    let rhs: Vec<f64> = (0..rows.nrows()).map(|r| if r + 1 == rows.nrows() { big } else { -big }).collect();
    // A = rowsᵀ = Q R: the least-norm g₀ = Q₁ R₁⁻ᵀ rhs.
    let decomposition = qr(rows.t(), QrMode::Full).expect("qr");
    let (q, r) = (decomposition.q.expect("q"), decomposition.r);
    let k = rows.nrows();
    let mut y = vec![0.0; k];
    for i in 0..k {
        y[i] = (rhs[i] - (0..i).map(|j| r[[j, i]] * y[j]).sum::<f64>()) / r[[i, i]];
    }
    let g0 = (0..k).fold(ndarray::Array1::<f64>::zeros(D), |acc, i| acc + &(q.column(i).to_owned() * y[i]));
    let up = f.m.flat.operators.iter().position(|op| op.name == "blocks.1.c_fc").expect("layer 1 MLP input");
    let down = f.m.flat.operators.iter().position(|op| op.name == "blocks.1.down_proj").expect("layer 1 MLP output");
    let mut gates = f.m.flat.operators[up].matrix();
    let mut outputs = f.m.flat.operators[down].matrix();
    for unit in 0..n {
        gates.row_mut(unit).assign(&g0);
        outputs.column_mut(unit).fill(0.0);
        outputs[[0, unit]] = epsilon;
    }
    let replaced = |flat: &OperatorProgram, op: usize, values: Array2<f64>| -> OperatorProgram {
        let mut out = flat.clone();
        let old = &flat.operators[op];
        let precision = exact_precision(values.iter().copied()).expect("finite");
        out.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).expect("dense"));
        out
    };
    let m_flat = replaced(&replaced(&f.m.flat, up, gates.clone()), down, outputs);
    let mut dropped = gates.clone();
    dropped.rows_mut().into_iter().take(n).for_each(|mut row| row.fill(0.0));
    let p_flat = replaced(&m_flat, up, dropped);
    let head = Head::of(&m_flat).expect("head");
    let m_program = DeviceProgram::compile_values(&host, &head.prefix(&m_flat)).expect("M");
    let mut p_program = DeviceProgram::compile_values(&host, &head.prefix(&p_flat)).expect("P");
    p_program.prepare_dense_parameters(&[up]).expect("parameters");
    let fixed_head = FixedHead::new(&host, &m_flat, &p_flat, 5).expect("head");
    // The functions' values are where the fixture's units of the same rows are.
    let units: Vec<Value> = (0..n).map(|unit| f.values[f.variables.iter().position(|v| v.block == 3 && v.parts == [(up, unit..unit + 1)]).expect("a unit")].clone()).collect();
    let m = Model::new(&m_program, &m_flat, f.m.entries.clone(), f.m.reads.clone(), &[], units.clone()).expect("M model");
    let p = Model::new(&p_program, &p_flat, f.m.entries.clone(), f.m.reads.clone(), &[up], units).expect("P model");
    let batch = Batch::new(vec![base], vec![source]).expect("batch");
    let e = |patch: Option<Patch>| Experiment { base: 0, source: 0, explained: vec![true; 2 * LAYERS], patch, position: at };
    let mut experiments: Vec<Experiment> = (0..n).map(|v| e(Some(Patch::Read { variable: v }))).collect();
    experiments.push(e(Some(Patch::Reads { variables: (0..n).collect() })));
    experiments.push(e(None));
    let targets = targets(&m, &fixed_head, &batch, &experiments).expect("targets");
    let bits: Vec<f64> = evaluate(&m, &p, &fixed_head, &batch, &targets, &experiments, false).expect("evaluate").bits.iter().map(|b| b.iter().sum()).collect();
    let (single, joint, clean) = (bits[..n].iter().copied().fold(0.0, f64::max), bits[n], bits[n + 1]);
    assert!(clean.abs() <= 1e-12, "the dropped functions are off on the base: {clean} bits");
    assert!(single <= 1e-3, "one dropped read alone: {single} bits");
    let ratio = joint / bits[..n].iter().sum::<f64>() * n as f64;
    assert!(ratio >= 0.9 * (n * n) as f64 && ratio <= 1.1 * (n * n) as f64, "joint {joint} bits against {n}² times a mean single, ratio {ratio}");
}

#[test]
fn a_device_model_refuses_a_layer_reading_before_its_stream() {
    let f = fixture();
    let device = Device::host();
    let m = DeviceProgram::compile_values(&device, &f.m.prefix).expect("M");
    let mut reads = f.m.reads.clone();
    reads.swap(0, 2);
    assert!(Model::new(&m, &f.m.flat, f.m.entries.clone(), reads, &[], Vec::new()).is_err());
    let mut entries = f.m.entries.clone();
    entries[2] = f.m.reads[3];
    assert!(Model::new(&m, &f.m.flat, entries, f.m.reads.clone(), &[], Vec::new()).is_err());
}

#[test]
fn an_interchange_loaded_with_p_scores_as_the_host_reference() {
    // The owner starts at M (the native artifact, its values at M's nodes) and loads P's maps: it
    // scores every experiment as the host reference does.
    let f = fixture();
    let device = Device::host();
    let (program, _) = fixture_sized(D, VOCAB, LENGTH, 6);
    let native = split_sites(&program).expect("split");
    let layers = layer_nodes(&native, LAYERS).expect("layers");
    let explanation = Artifact::native(&native).expect("native artifact");
    let mut x = Interchange::new(&device, &native, &layers, &explanation, &f.trainable, f.variables.clone(), usize::MAX, 5).expect("interchange");
    let loaded: Vec<Array2<f64>> = f.trainable.iter().map(|op| f.p.flat.operators[*op].matrix()).collect();
    x.load(&loaded).expect("load");
    let experiments = experiments(&f);
    let scored = x.evaluate(&f.batch, &experiments, false).expect("evaluate").bits;
    for (e, bits) in experiments.iter().zip(&scored) {
        for (a, b) in bits.iter().zip(reference(&f, e)) {
            assert!((a - b).abs() <= 1e-9, "{e:?}: device {a} bits, host {b} bits");
        }
    }
}

/// The starting library of the tiny export `dir` (`library_mdl::explanation`) under the native
/// questions: its values sit at its rules' nodes (`Artifact::owners`), not at `M`'s, yet every
/// patched experiment scores zero, since the library computes what `M` computes, while each patch
/// changes `M`'s own prediction.
fn the_starting_library_explains_m_under_every_patch(dir: std::path::PathBuf) {
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let device = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let held = values(&native, &explanation.artifact, &layers, &variables).expect("the library's values");
    assert!(held.iter().all(|v| !v.sites.is_empty()), "the library computes every function of M");
    let x = Interchange::new(&device, &native, &layers, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
    let experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(9), 3, x.variables(), 4, 12).expect("the draw");
    let bits = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    let made = x.targets(&batch, &experiments).expect("targets").host(&device).expect("host");
    for (i, pair) in experiments.chunks(2).enumerate() {
        let (clean, patched) = (&made[2 * i].1, &made[2 * i + 1].1);
        let at = pair[1].position;
        assert!(clean[at..].iter().zip(patched).any(|(a, b)| (a - b).abs() > 1e-9), "{:?}: the patch changes nothing in M", pair[1]);
    }
}

#[test]
fn the_starting_gelu_library_explains_m_under_every_patch() {
    the_starting_library_explains_m_under_every_patch(crate::test_support::tiny_export("interchange_library_gelu", 2));
}

#[test]
fn the_starting_qwen3_library_explains_m_under_every_patch() {
    the_starting_library_explains_m_under_every_patch(crate::test_support::tiny_qwen3_export("interchange_library_qwen3", 2));
}

#[test]
fn a_body_rewrite_explains_m_under_every_patch() {
    // Four of layer 1's MLP functions rewritten as a call of a new body (`library_bodies::rewrite`,
    // exact: every singular value of their rows is kept). Their values now sit at the body's units
    // at the call, the other functions' at the MLP's rule, and every patch, of the region's
    // functions alone, with others, or elsewhere, still scores zero against M.
    let dir = crate::test_support::tiny_export("interchange_body_rewrite", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let start = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let region = [1, 4, 6, 9];
    let (rewritten, call) = crate::library_bodies::rewrite(&start, 1, &region).expect("the rewrite");
    assert_eq!(call.discarded, [0.0, 0.0], "the rewrite keeps every direction");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let variables = reads(&native, &layers).expect("the reads");
    let unit = |u: usize| variables.iter().enumerate().filter(|(_, v)| v.block == 3).nth(u).map(|(i, _)| i).expect("a unit of layer 1");
    let (before, after) = (values(&native, &start.artifact, &layers, &variables).expect("the start's values"), values(&native, &rewritten.artifact, &layers, &variables).expect("the rewrite's values"));
    for u in 0..16 {
        let (was, is) = (&before[unit(u)].sites, &after[unit(u)].sites);
        assert_eq!(is.len(), was.len(), "unit {u}: one site per part");
        assert_eq!(region.contains(&u), is.iter().zip(was).all(|(a, b)| a.node != b.node), "unit {u}: the region's functions move to the body");
    }
    let device = Device::host();
    let x = Interchange::new(&device, &native, &layers, &rewritten.artifact, &rewritten.trainable, variables.clone(), 1 << 30, 64).expect("the experiments");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
    let mut experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(9), 3, x.variables(), 4, 12).expect("the draw");
    let e = |base: usize, patch: Patch, position: usize| Experiment { base, source: base, explained: vec![true, false, true, true], patch: Some(patch), position };
    experiments.push(e(0, Patch::Read { variable: unit(4) }, 3));
    experiments.push(e(1, Patch::Reads { variables: vec![unit(1), unit(2), unit(9)] }, 5));
    experiments.push(e(2, Patch::Reads { variables: region.iter().map(|u| unit(*u)).collect() }, 0));
    let bits = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
}

/// An engine counting the rows each forward call runs, around the reference.
struct Counting<'a> {
    model: Model<'a>,
    rows: std::cell::Cell<usize>,
    /// The most bytes its forward passes' tapes may hold, within the reference's own budget.
    budget: usize,
}

impl BlockEngine for Counting<'_> {
    type Tape = DeviceTrace;

    fn device(&self) -> &Device {
        BlockEngine::device(&self.model)
    }

    fn width(&self) -> usize {
        BlockEngine::width(&self.model)
    }

    fn blocks(&self) -> usize {
        BlockEngine::blocks(&self.model)
    }

    fn arithmetic(&self) -> gam_gpu::tensor::Arithmetic {
        BlockEngine::arithmetic(&self.model)
    }

    fn values(&self) -> &[Value] {
        BlockEngine::values(&self.model)
    }

    fn forward(
        &self,
        block: usize,
        stream: &mut gam_gpu::tensor::Tensor,
        ranges: &[std::ops::Range<usize>],
        tokens: &[&[u32]],
        edits: Option<&Edits>,
        keep: bool,
    ) -> Result<Option<DeviceTrace>, String> {
        self.rows.set(self.rows.get() + ranges.iter().map(ExactSizeIterator::len).sum::<usize>());
        self.model.forward(block, stream, ranges, tokens, edits, keep)
    }

    fn reverse(
        &self,
        block: usize,
        tape: &DeviceTrace,
        cotangent: &mut gam_gpu::tensor::Tensor,
        ranges: &[std::ops::Range<usize>],
        edits: Option<&Edits>,
        sums: (&mut std::collections::BTreeMap<usize, gam_gpu::tensor::Tensor>, gam_gpu::tensor::Arithmetic),
    ) -> Result<(), String> {
        self.model.reverse(block, tape, cotangent, ranges, edits, sums)
    }

    fn tape_bytes(tape: &DeviceTrace) -> usize {
        Model::tape_bytes(tape)
    }

    fn gradient_bytes(&self) -> Result<usize, String> {
        self.model.gradient_bytes()
    }

    fn tape_budget(&self, rows: usize) -> Result<usize, String> {
        Ok(self.budget.min(self.model.tape_budget(rows)?))
    }
}

#[test]
fn a_patched_experiment_reuses_its_base_below_the_patched_block() {
    // Base 0 clean under hybrid h runs blocks 0..4; its patched experiment (the same hybrid,
    // patched at block 2) runs only blocks 2..4; a second clean experiment on base 0 under h runs
    // nothing of its own; base 1 runs 0..4 and the source (another sequence) 0..3: 13 block runs of
    // a sequence, against 19 without sharing. The scores equal the reference engine's.
    let f = fixture();
    let read = f.variables.iter().position(|v| v.block == 2).expect("an attention variable of layer 1");
    let hybrid = vec![true, false, true, true];
    let e = |base: usize, source: usize, patch: Option<Patch>, position: usize| Experiment { base, source, explained: hybrid.clone(), patch, position };
    let experiments = vec![e(0, 0, None, 0), e(0, 1, Some(Patch::Read { variable: read }), 3), e(0, 0, None, 2), e(1, 1, None, 0)];
    let device = Device::host();
    let programs = programs(&device, &f);
    let (m, p) = models(&f, &programs);
    let targets = targets(&m, &programs.head, &f.batch, &experiments).expect("targets");
    let reference = evaluate(&m, &p, &programs.head, &f.batch, &targets, &experiments, true).expect("evaluate");
    let (cm, cp) = models(&f, &programs);
    let (cm, cp) = (Counting { model: cm, rows: std::cell::Cell::new(0), budget: usize::MAX }, Counting { model: cp, rows: std::cell::Cell::new(0), budget: usize::MAX });
    let counted = evaluate(&cm, &cp, &programs.head, &f.batch, &targets, &experiments, true).expect("evaluate");
    assert_eq!(cm.rows.get() + cp.rows.get(), 13 * LENGTH);
    assert_eq!(counted.bits, reference.bits);
    for (op, g) in &reference.gradient {
        assert_eq!(device.download(g).expect("gradient"), device.download(&counted.gradient[op]).expect("gradient"));
    }
}

#[test]
fn blocks_run_again_in_the_reverse_pass_give_the_kept_tapes_gradient() {
    // The experiments of the sharing test (lanes forking at a patch, a source run to its patched
    // block): with no budget for tapes every call keeps only the rows entering it and runs its
    // block again in the reverse pass, so the 13 block runs of a sequence run twice; the scores
    // and the gradient equal those of the kept tapes bit for bit.
    let f = fixture();
    let read = f.variables.iter().position(|v| v.block == 2).expect("an attention variable of layer 1");
    let hybrid = vec![true, false, true, true];
    let e = |base: usize, source: usize, patch: Option<Patch>, position: usize| Experiment { base, source, explained: hybrid.clone(), patch, position };
    let experiments = vec![e(0, 0, None, 0), e(0, 1, Some(Patch::Read { variable: read }), 3), e(0, 0, None, 2), e(1, 1, None, 0)];
    let device = Device::host();
    let programs = programs(&device, &f);
    let (m, p) = models(&f, &programs);
    let targets = targets(&m, &programs.head, &f.batch, &experiments).expect("targets");
    let kept = evaluate(&m, &p, &programs.head, &f.batch, &targets, &experiments, true).expect("evaluate");
    let (cm, cp) = models(&f, &programs);
    let (cm, cp) = (Counting { model: cm, rows: std::cell::Cell::new(0), budget: 0 }, Counting { model: cp, rows: std::cell::Cell::new(0), budget: 0 });
    let again = evaluate(&cm, &cp, &programs.head, &f.batch, &targets, &experiments, true).expect("evaluate");
    assert_eq!(cm.rows.get() + cp.rows.get(), 2 * 13 * LENGTH);
    assert_eq!(again.bits, kept.bits);
    assert_eq!(again.gradient.len(), kept.gradient.len());
    for (op, g) in &kept.gradient {
        assert_eq!(device.download(g).expect("gradient"), device.download(&again.gradient[op]).expect("gradient"));
    }
}

#[test]
fn m_runs_the_same_whatever_explanation_it_is_compared_with() {
    // M's targets do not depend on the explanation P they are compared with (review item 7). The
    // same M is compared with its starting library and with a body rewrite of it (layer 1's
    // functions 1, 4, 6 and 9 called through a new body): its targets agree bit for bit.
    let dir = crate::test_support::tiny_export("interchange_teacher", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let start = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let (rewritten, _) = crate::library_bodies::rewrite(&start, 1, &[1, 4, 6, 9]).expect("the rewrite");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let device = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
    let experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(9), 3, &variables, 4, 12).expect("the draw");
    let mut made = Vec::new();
    for explanation in [&start, &rewritten] {
        let x = Interchange::new(&device, &native, &layers, &explanation.artifact, &explanation.trainable, variables.clone(), 1 << 30, 64).expect("the experiments");
        made.push(x.targets(&batch, &experiments).expect("the targets").host(&device).expect("on the host"));
    }
    assert_eq!(made[0].len(), made[1].len());
    for (a, b) in made[0].iter().zip(&made[1]) {
        assert!(a.0 == b.0 && a.1 == b.1, "M's targets depend on the explanation");
    }
}

/// Random ReLU parts of the tiny export's two MLP blocks (1 and 3) at width `width`: per block one
/// that fires on nearly every row (bias 1, small read) and one that never fires (bias −100).
fn random_parts(width: usize, seed: u64) -> Vec<Part> {
    use rand::RngExt;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut vector = |scale: f64| (0..width).map(|_| scale * (rng.random::<f64>() - 0.5)).collect::<Vec<f64>>();
    let mut out = Vec::new();
    for block in [1, 3] {
        for (index, bias) in [1.0, -100.0].into_iter().enumerate() {
            out.push(Part { block, index, read: vector(0.2), bias, write: vector(2.0) });
        }
    }
    out
}

/// Edits of parts (`Patch::Part`) on the starting library of the tiny export `dir`, an exact copy
/// of `M`: each part at each factor in `FACTORS`, under `P` alone and under a hybrid running `M`
/// at the edited block, scores zero against `M` (both models add the same write, computed from
/// the same read, to the same stream), with the gradient's reverse through the edits too; an edit
/// of a firing part changes `M`'s own prediction, one of a part that never fires does not.
fn the_starting_library_explains_m_under_every_part_edit(dir: std::path::PathBuf) {
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let mut devices = vec![Device::host()];
    devices.extend(Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("a device probe"));
    for device in devices {
        let variables = reads(&native, &layers).expect("the reads");
        let mut x = Interchange::new(&device, &native, &layers, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
        let width = BlockEngine::width(&x.models().0);
        x.set_parts(random_parts(width, 3)).expect("the parts");
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
        let mut experiments = Vec::new();
        for part in 0..4 {
            for factor in 0..FACTORS.len() {
                let base = (part + factor) % 3;
                let block = x.parts()[part].block;
                let hybrid: Vec<bool> = (0..4).map(|b| b != block).collect();
                experiments.push(Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 });
                experiments.push(Experiment { base, source: base, explained: vec![true; 4], patch: Some(Patch::Part { part, factor }), position: 1 + (part + 3 * factor) % 11 });
                experiments.push(Experiment { base, source: base, explained: hybrid, patch: Some(Patch::Part { part, factor }), position: 2 });
            }
        }
        let counts = census(&experiments, x.variables());
        assert_eq!((counts["remove_part"], counts["amplify_part"]), (8, 24), "{counts:?}");
        for gradient in [false, true] {
            let bits = x.evaluate(&batch, &experiments, gradient).expect("evaluate").bits;
            assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{}: {bits:?}", device.name());
        }
        let made = x.targets(&batch, &experiments).expect("targets").host(&device).expect("host");
        for (i, triple) in experiments.chunks(3).enumerate() {
            let (clean, edited) = (&made[3 * i].1, &made[3 * i + 1].1);
            let at = triple[1].position;
            let Some(Patch::Part { part, factor }) = triple[1].patch else { panic!("an edit") };
            let changed = clean[at..].iter().zip(edited).any(|(a, b)| (a - b).abs() > 1e-9);
            assert_eq!(changed, part % 2 == 0 && FACTORS[factor] != 1.0, "{:?}: M changes only under an edit of a firing part", triple[1]);
        }
        // With P applying no edit, an edit scores KL(M_e ‖ M): positive for a firing part where
        // P_e runs P's block, zero for a silent one, under a hybrid running M's block there (M's
        // block applies the edit in either run) and for the clean experiments.
        x.unedited_explanation();
        let effects = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
        for (e, bits) in experiments.iter().zip(&effects) {
            let sum: f64 = bits.iter().sum();
            match e.patch {
                Some(Patch::Part { part, .. }) if part % 2 == 0 && e.explained[x.parts()[part].block] => assert!(sum > 1e-9, "{e:?}: {sum}"),
                _ => assert!(sum.abs() <= 1e-9, "{e:?}: {sum}"),
            }
        }
    }
}

#[test]
fn the_starting_gelu_library_explains_m_under_every_part_edit() {
    the_starting_library_explains_m_under_every_part_edit(crate::test_support::tiny_export("interchange_parts_gelu", 2));
}

#[test]
fn the_starting_qwen3_library_explains_m_under_every_part_edit() {
    the_starting_library_explains_m_under_every_part_edit(crate::test_support::tiny_qwen3_export("interchange_parts_qwen3", 2));
}

/// An edit of a part on `M` is the hand computation: through layer 0's MLP block of the tiny
/// Qwen3 export, the stream after the block moves by exactly `(α − 1)·relu(g·x + c)·u` at the
/// edited row (`x` `M`'s own read there) and nowhere else, for every factor.
#[test]
fn an_edit_of_a_part_on_m_adds_its_scaled_write() {
    let dir = crate::test_support::tiny_qwen3_export("interchange_parts_hand", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequence: Vec<u32> = tokens[..12].to_vec();
    let d = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
    let width = BlockEngine::width(&x.models().0);
    x.set_parts(random_parts(width, 4)).expect("the parts");
    let (m, _) = x.models();
    let sites = m.part_sites().expect("M's part sites");
    let (read, _) = sites.nodes(1).expect("layer 0's MLP holds parts");
    let part = &x.parts()[0];
    let (ranges, tok) = (vec![0..12], vec![sequence.as_slice()]);
    let mut entering = d.zeros(12, width).expect("zeros");
    m.forward(0, &mut entering, &ranges, &tok, None, false).expect("attention");
    let mut plain = d.copy(&entering).expect("copy");
    let trace = m.forward(1, &mut plain, &ranges, &tok, None, true).expect("the MLP").expect("a tape");
    let normed = d.download(trace.value(read).expect("the read")).expect("download");
    let plain = d.download(&plain).expect("download");
    for (factor, alpha) in FACTORS.iter().enumerate() {
        let row = 3 + factor;
        let edits = Edits::with_parts(&d, &[], m.values(), &[(row, Edit::Part { part: 0, factor })], Some(sites), None).expect("the edits");
        let mut edited = d.copy(&entering).expect("copy");
        m.forward(1, &mut edited, &ranges, &tok, Some(&edits), false).expect("the edited MLP");
        let moved = d.download(&edited).expect("download") - &plain;
        let pre: f64 = normed.row(row).iter().zip(&part.read).map(|(a, b)| a * b).sum::<f64>() + part.bias;
        assert!(pre > 0.0, "the part fires at row {row}");
        let scale = moved.iter().chain(plain.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
        for r in 0..12 {
            for c in 0..width {
                let want = if r == row { (alpha - 1.0) * pre * part.write[c] } else { 0.0 };
                assert!((moved[[r, c]] - want).abs() <= 1e-12 * scale, "factor {alpha}, row {r}, column {c}: moved {} against {want}", moved[[r, c]]);
            }
        }
    }
}

/// On the tiny Qwen3 decoder with layer 1's MLP a transcoder's 64 features (every fourth never
/// fires): the parts are the features with their rows; the active parts at a row are the host's
/// positive pre-activations of `M`'s read there; an edit of a part on `P`'s transcoder block is the
/// hand computation `Σ_{j≠i} relu(g_j·x + c_j) u_j + α relu(g_i·x + c_i) u_i + b` at the edited
/// row (`x` `P`'s read there) and leaves the other rows; and the drawn edits are of firing parts
/// or the one silent part drawn with them, every one scored finite and nonnegative.
#[test]
fn edits_of_a_transcoder_block_are_the_hand_computation() {
    use super::interchange::{Family, parts_of};
    let export = crate::test_support::tiny_qwen3_export("interchange_parts_transcoder", 2);
    let imported = crate::import::import_language_model(&export, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(&export).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let dir = std::env::temp_dir().join(format!("interchange_parts_transcoder_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let (full, kept) = (dir.join("layer_1.safetensors"), dir.join("kept_1.safetensors"));
    crate::test_support::transcoder_file(&full, 64, 8, 3);
    crate::library_transcoder::Transcoder::open(&full).expect("the file").write_kept(&(0..64).collect::<Vec<_>>(), &[0.0; 8], &kept).expect("kept");
    let explanation = crate::library_mdl::explanation_with(&native, &layers, &std::collections::BTreeMap::from([(1, kept)])).expect("the library");
    std::fs::remove_dir_all(&dir).expect("the directory is removed");
    let program = &explanation.artifact.program;
    let operator = |name: &str| program.operators.iter().find(|op| op.name == name).expect("an operator").matrix();
    let (gate, bias, write, out_bias) = (operator("library.l1.mlp.gate"), operator("library.l1.mlp.gate_bias"), operator("library.l1.mlp.out"), operator("library.l1.mlp.bias"));
    let parts = parts_of(program, 2).expect("the parts");
    assert_eq!(parts.len(), 64);
    for (i, p) in parts.iter().enumerate() {
        assert_eq!((p.block, p.bias), (3, bias[[i, 0]]));
        assert!(p.read.iter().zip(gate.row(i)).all(|(a, b)| a == b) && p.write.iter().zip(write.column(i)).all(|(a, b)| a == b));
    }
    let d = Device::host();
    let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, explanation.reads.clone(), 1 << 30, 64).expect("the experiments");
    x.set_parts(parts.clone()).expect("the parts");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    // Activity against the host's pre-activations of M's read.
    let family = crate::library_mdl::sequence_family(&batch.base.iter().map(Vec::as_slice).collect::<Vec<_>>()).expect("the family");
    let read = native.execute(&family, false).expect("M on the host").values[layers[1].normed].clone();
    let pre = read.dot(&gate.t()) + &bias.column(0);
    let wanted: Vec<(usize, usize, usize)> = (0..3).flat_map(|n| (1..12).map(move |t| (n, 3, t))).collect();
    let active = x.active_parts(&batch, &wanted).expect("the activity");
    for ((n, _, t), found) in wanted.iter().zip(&active) {
        let host: Vec<usize> = (0..64).filter(|j| pre[[n * 12 + t, *j]] > 0.0).collect();
        assert_eq!(found, &host, "base {n} row {t}");
    }
    // An edit on P's transcoder block against the hand computation.
    let (_, p) = x.models();
    let sites = p.part_sites().expect("P's part sites");
    let (read_node, out_node) = sites.nodes(3).expect("layer 1's MLP holds parts");
    let (ranges, tok) = (vec![0..12], vec![batch.base[0].as_slice()]);
    let mut entering = d.zeros(12, 8).expect("zeros");
    for b in 0..3 {
        p.forward(b, &mut entering, &ranges, &tok, None, false).expect("a block");
    }
    let plain = p.forward(3, &mut d.copy(&entering).expect("copy"), &ranges, &tok, None, true).expect("the MLP").expect("a tape");
    let (xp, plain_out) = (d.download(plain.value(read_node).expect("the read")).expect("download"), d.download(plain.value(out_node).expect("the output")).expect("download"));
    let row = 5;
    let i = (0..64).find(|j| (xp.row(row).dot(&gate.row(*j)) + bias[[*j, 0]]) > 0.0).expect("a firing feature");
    for (factor, alpha) in FACTORS.iter().enumerate() {
        let edits = Edits::with_parts(&d, &[], p.values(), &[(row, Edit::Part { part: i, factor })], Some(sites), None).expect("the edits");
        let edited = p.forward(3, &mut d.copy(&entering).expect("copy"), &ranges, &tok, Some(&edits), true).expect("the edited MLP").expect("a tape");
        let edited = d.download(edited.value(out_node).expect("the output")).expect("download");
        let mut hand = out_bias.column(0).to_owned();
        for j in 0..64 {
            let a = (xp.row(row).dot(&gate.row(j)) + bias[[j, 0]]).max(0.0) * if j == i { *alpha } else { 1.0 };
            hand.scaled_add(a, &write.column(j));
        }
        let scale = hand.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for r in 0..12 {
            for c in 0..8 {
                let want = if r == row { hand[c] } else { plain_out[[r, c]] };
                assert!((edited[[r, c]] - want).abs() <= 1e-12 * scale, "factor {alpha}, row {r}, column {c}: {} against {want}", edited[[r, c]]);
            }
        }
    }
    // Drawn edits: of firing parts, or the silent part drawn with them; scored finite.
    let experiments = x.sample_edits(&mut rand::rngs::StdRng::seed_from_u64(2), &batch, &[Family::RemovePart, Family::AmplifyPart], 6, false).expect("the draw");
    let counts = census(&experiments, x.variables());
    assert_eq!((counts["clean_alone"], counts["remove_part"] + counts["amplify_part"]), (3, 18), "{counts:?}");
    let silent = experiments.iter().filter(|e| matches!(e.patch, Some(Patch::Part { part, .. }) if pre[[e.base * 12 + e.position, part]] <= 0.0)).count();
    assert!(silent <= 6, "{silent} edits of silent parts");
    let bits = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(bits.iter().flatten().all(|b| b.is_finite() && *b >= -1e-12), "{bits:?}");
}

/// A head's removal (`Patch::Head`) zeroes the head's attention output at the edited row, in both
/// models alike: with `P` = `M` (`Artifact::native`, every head held by both) on the tiny Qwen3
/// export, every head's removal under `P` alone scores zero and changes `M`'s prediction; the
/// edited forward of layer 0's attention holds zeros at the head's row and the plain values
/// elsewhere; drawn removals name held heads after the first token; and with `P` applying no
/// edit, each removal scores `KL(M_e ‖ M)` above zero.
#[test]
fn a_head_removal_zeroes_its_output_in_both_models() {
    use super::interchange::Family;
    let dir = crate::test_support::tiny_qwen3_export("interchange_heads", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let d = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let mut x = Interchange::new(&d, &native, &layers, &Artifact::native(&native).expect("native"), &[], variables, 1 << 30, 64).expect("the experiments");
    let heads = x.heads();
    assert!(!heads.is_empty() && heads.iter().all(|(b, held)| *held && b % 2 == 0), "{heads:?}");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    let experiments: Vec<Experiment> = (0..heads.len()).map(|head| Experiment { base: head % 3, source: head % 3, explained: vec![true; 4], patch: Some(Patch::Head { head }), position: 1 + head % 11 }).collect();
    assert_eq!(census(&experiments, x.variables())["remove_head"], heads.len());
    let bits = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    // The edited forward of layer 0's attention: the head's node is zero at the row.
    let (m, _) = x.models();
    let sites = m.part_sites().expect("M's edit sites");
    let (ranges, tok) = (vec![0..12], vec![sequences[0].as_slice()]);
    let width = BlockEngine::width(&m);
    let plain = m.forward(0, &mut d.zeros(12, width).expect("zeros"), &ranges, &tok, None, true).expect("attention").expect("a tape");
    let edits = Edits::with_parts(&d, &[], m.values(), &[(4, Edit::Head { head: 0 })], Some(sites), None).expect("the edits");
    let edited = m.forward(0, &mut d.zeros(12, width).expect("zeros"), &ranges, &tok, Some(&edits), true).expect("attention").expect("a tape");
    let node = *edits.nodes().iter().next().expect("one edited node");
    let (plain, edited) = (d.download(plain.value(node).expect("plain")).expect("download"), d.download(edited.value(node).expect("edited")).expect("download"));
    for r in 0..12 {
        for c in 0..plain.ncols() {
            let want = if r == 4 { 0.0 } else { plain[[r, c]] };
            assert_eq!(edited[[r, c]], want, "row {r}, column {c}");
        }
    }
    assert!(plain.row(4).iter().any(|v| *v != 0.0), "the head writes at the row");
    let drawn = x.sample_edits(&mut rand::rngs::StdRng::seed_from_u64(4), &batch, &[Family::RemoveHead], 4, false).expect("the draw");
    assert!(drawn.iter().all(|e| match e.patch {
        Some(Patch::Head { head }) => heads[head].1 && e.position >= 1,
        None => true,
        _ => false,
    }));
    // From the position on: zero against M as well, with and without the reverse pass.
    let from: Vec<Experiment> = (0..heads.len()).map(|head| Experiment { patch: Some(Patch::HeadFrom { head }), ..experiments[head].clone() }).collect();
    assert_eq!(census(&from, x.variables())["remove_head_from"], heads.len());
    for gradient in [false, true] {
        let bits = x.evaluate(&batch, &from, gradient).expect("evaluate").bits;
        assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    }
    x.unedited_explanation();
    let effects = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(effects.iter().all(|bits| bits.iter().sum::<f64>() > 1e-9), "{effects:?}");
    let longer = x.evaluate(&batch, &from, false).expect("evaluate").bits;
    assert!(longer.iter().zip(&effects).all(|(a, b)| a.iter().sum::<f64>() > 1e-9 && a.iter().sum::<f64>() >= b[..1].iter().sum::<f64>()), "{longer:?}");
}

/// A cut connection (`Patch::Cut`) on the starting library of the tiny Qwen3 export, an exact copy
/// of `M`, from a firing part of layer 0's MLP to one of layer 1's, its source another sequence:
/// it scores zero against `M` (both models record the same activations and recompute the same
/// norm), it changes `M`'s prediction, and with `P` applying no edit it scores `KL(M_e ‖ M)` above
/// zero; drawn cuts run from an earlier block's part to a later one's, with another source; its
/// reverse pass runs.
#[test]
fn a_cut_connection_is_the_same_path_patch_in_both_models() {
    use super::interchange::Family;
    let dir = crate::test_support::tiny_qwen3_export("interchange_cuts", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let d = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
    let width = BlockEngine::width(&x.models().0);
    // Large writes, so that the source's activation moves layer 1's read.
    let mut parts = random_parts(width, 5);
    parts[0].write.iter_mut().for_each(|u| *u *= 20.0);
    x.set_parts(parts).expect("the parts");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    let cut = |base: usize, position: usize| Experiment { base, source: (base + 1) % 3, explained: vec![true; 4], patch: Some(Patch::Cut { from: 0, to: 2 }), position };
    let clean = |base: usize| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 };
    let experiments = vec![clean(0), cut(0, 3), clean(1), cut(1, 7), clean(2), cut(2, 11)];
    assert_eq!(census(&experiments, x.variables())["cut_connection"], 3);
    let bits = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    let bits = x.evaluate(&batch, &experiments, true).expect("the reverse pass through the cuts").bits;
    assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    let made = x.targets(&batch, &experiments).expect("targets").host(&d).expect("host");
    for pair in 0..3 {
        let (plain, edited, at) = (&made[2 * pair].1, &made[2 * pair + 1].1, experiments[2 * pair + 1].position);
        assert!(plain[at..].iter().zip(edited).any(|(a, b)| (a - b).abs() > 1e-9), "{:?}: the cut changes nothing in M", experiments[2 * pair + 1]);
    }
    let drawn = x.sample_edits(&mut rand::rngs::StdRng::seed_from_u64(6), &batch, &[Family::CutConnection], 4, false).expect("the draw");
    for e in &drawn {
        if let Some(Patch::Cut { from, to }) = e.patch {
            assert!(x.parts()[from].block < x.parts()[to].block && e.source != e.base && e.position >= 1, "{e:?}");
        }
    }
    x.unedited_explanation();
    let effects = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    assert!(effects.iter().skip(1).step_by(2).all(|bits| bits.iter().sum::<f64>() > 1e-9), "{effects:?}");
}

/// A joint removal of parts (`Patch::Parts`) is the removal of each at the same row: on the tiny
/// Qwen3 decoder with layer 1's MLP a transcoder's 64 features, removing a set of firing features
/// at a row moves `P`'s and `M`'s predictions as removing them one edit at a time on the same path
/// does (the same scores, bit for bit), and drawn joint removals are subsets of the parts firing on
/// `M`'s read there.
#[test]
fn a_joint_removal_of_parts_is_each_removal_at_once() {
    use super::interchange::{Family, parts_of};
    let export = crate::test_support::tiny_qwen3_export("interchange_joint_parts", 2);
    let imported = crate::import::import_language_model(&export, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(&export).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let dir = std::env::temp_dir().join(format!("interchange_joint_parts_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let (full, kept) = (dir.join("layer_1.safetensors"), dir.join("kept_1.safetensors"));
    crate::test_support::transcoder_file(&full, 64, 8, 3);
    crate::library_transcoder::Transcoder::open(&full).expect("the file").write_kept(&(0..64).collect::<Vec<_>>(), &[0.0; 8], &kept).expect("kept");
    let explanation = crate::library_mdl::explanation_with(&native, &layers, &std::collections::BTreeMap::from([(1, kept)])).expect("the library");
    std::fs::remove_dir_all(&dir).expect("the directory is removed");
    let d = Device::host();
    let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, explanation.reads.clone(), 1 << 30, 64).expect("the experiments");
    x.set_parts(parts_of(&explanation.artifact.program, 2).expect("the parts")).expect("the parts");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    let firing = x.active_parts(&batch, &[(1, 3, 6)]).expect("the activity").remove(0);
    assert!(firing.len() >= 3, "{firing:?}");
    let chosen: Vec<usize> = firing[..3].to_vec();
    let joint = Experiment { base: 1, source: 1, explained: vec![true; 4], patch: Some(Patch::Parts { parts: chosen.clone(), factor: 0 }), position: 6 };
    let bits = x.evaluate(&batch, std::slice::from_ref(&joint), false).expect("evaluate").bits;
    assert!(bits[0].iter().all(|b| b.is_finite()) && bits[0].iter().sum::<f64>() > 0.0, "{bits:?}");
    // The joint removal of one part is that part's removal.
    let single = Experiment { patch: Some(Patch::Parts { parts: vec![chosen[0]], factor: 0 }), ..joint.clone() };
    let part = Experiment { patch: Some(Patch::Part { part: chosen[0], factor: 0 }), ..joint.clone() };
    let pair = x.evaluate(&batch, &[single, part], false).expect("evaluate").bits;
    assert_eq!(pair[0], pair[1], "a joint removal of one part is its removal");
    let drawn = x.sample_edits(&mut rand::rngs::StdRng::seed_from_u64(8), &batch, &[Family::RemoveParts], 4, false).expect("the draw");
    for e in &drawn {
        if let Some(Patch::Parts { parts, factor }) = &e.patch {
            let firing = x.active_parts(&batch, &[(e.base, 3, e.position)]).expect("the activity").remove(0);
            assert!(*factor == 0 && !parts.is_empty() && (parts.iter().all(|p| firing.contains(p)) || (firing.is_empty() && parts.len() == 1)), "{e:?} against {firing:?}");
        }
    }
}

/// A swap of a part's activation (`Patch::Swap`) on the starting library of the tiny Qwen3 export,
/// an exact copy of `M`, its source another sequence: it scores zero against `M` with and without
/// the gradient's reverse, and changes `M`'s prediction exactly for a firing part.
#[test]
fn a_swap_of_a_part_is_the_same_on_both_models() {
    let dir = crate::test_support::tiny_qwen3_export("interchange_swaps", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let d = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
    let width = BlockEngine::width(&x.models().0);
    x.set_parts(random_parts(width, 7)).expect("the parts");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    let swap = |base: usize, part: usize, position: usize| Experiment { base, source: (base + 1) % 3, explained: vec![true; 4], patch: Some(Patch::Swap { part }), position };
    let clean = |base: usize| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 };
    let experiments = vec![clean(0), swap(0, 0, 3), clean(1), swap(1, 2, 7), clean(2), swap(2, 1, 5)];
    assert_eq!(census(&experiments, x.variables())["swap_part"], 3);
    for gradient in [false, true] {
        let bits = x.evaluate(&batch, &experiments, gradient).expect("evaluate").bits;
        assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    }
    let made = x.targets(&batch, &experiments).expect("targets").host(&d).expect("host");
    for pair in 0..3 {
        let (plain, edited, e) = (&made[2 * pair].1, &made[2 * pair + 1].1, &experiments[2 * pair + 1]);
        let Some(Patch::Swap { part }) = e.patch else { panic!("a swap") };
        let changed = plain[e.position..].iter().zip(edited).any(|(a, b)| (a - b).abs() > 1e-9);
        assert_eq!(changed, part % 2 == 0, "{e:?}: M changes only under a swap of a firing part");
    }
}

/// A part removed at every row from the position on (`Patch::PartFrom`) on the starting library of
/// the tiny Qwen3 export, an exact copy of `M`: it scores zero against `M` with and without the
/// gradient's reverse, and moves `M` at least as much over the later tokens as the removal at the
/// position alone does.
#[test]
fn a_part_removed_from_a_position_on_is_the_same_on_both_models() {
    let dir = crate::test_support::tiny_qwen3_export("interchange_part_from", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let d = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
    let width = BlockEngine::width(&x.models().0);
    x.set_parts(random_parts(width, 9)).expect("the parts");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    let e = |patch: Patch| Experiment { base: 1, source: 1, explained: vec![true; 4], patch: Some(patch), position: 4 };
    let experiments = vec![e(Patch::PartFrom { part: 0, factor: 0 }), e(Patch::Part { part: 0, factor: 0 }), e(Patch::PartFrom { part: 2, factor: 0 })];
    assert_eq!(census(&experiments, x.variables())["remove_part_from"], 2);
    for gradient in [false, true] {
        let bits = x.evaluate(&batch, &experiments, gradient).expect("evaluate").bits;
        assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    }
    x.unedited_explanation();
    let effects = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    let (from, at): (f64, f64) = (effects[0].iter().sum(), effects[1].iter().sum());
    assert!(from > 1e-9 && from >= at, "removed from the position on {from}, at the position {at}");
}

/// The reverse pass of every edit of parts is the transpose of its forward tangent: on the tiny
/// Qwen3 export's part sites, with random reads, streams, tangents and cotangents (host, float64),
/// `⟨tangent, ḡ⟩` equals `⟨direction, reverse(ḡ)⟩` to 1e-12 relative for an amplification, a swap
/// (read at another row), a removal at several rows, a head's removal at two rows, and a cut
/// connection (probes at layer 0's MLP on a base and a source row, the cut at layer 1's: the
/// tangent through the recomputed norm `N`, the reverse through its transpose, the entering
/// stream's share and the record carried between the blocks). The tangents are written from the
/// edits' definitions, apart from the edits' code.
#[test]
fn the_reverse_of_every_edit_is_the_transpose_of_its_tangent() {
    use ndarray::Array2 as A;
    use rand::RngExt;
    let dir = crate::test_support::tiny_qwen3_export("interchange_adjoint", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let d = Device::host();
    let variables = reads(&native, &layers).expect("the reads");
    let mut x = Interchange::new(&d, &native, &layers, &Artifact::native(&native).expect("native"), &[], variables, 1 << 30, 64).expect("the experiments");
    let width = BlockEngine::width(&x.models().0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut random = |rows: usize, cols: usize, scale: f64| A::from_shape_fn((rows, cols), |_| scale * (rng.random::<f64>() - 0.5));
    // Parts that fire on the random reads (bias 1, small reads) and a large write for the cut's source.
    let parts: Vec<Part> = [1usize, 1, 3].iter().enumerate().map(|(index, b)| Part { block: *b, index, read: random(1, width, 0.2).row(0).to_vec(), bias: 1.0, write: random(1, width, 2.0).row(0).to_vec() }).collect();
    x.set_parts(parts.clone()).expect("the parts");
    let (m, _) = x.models();
    let sites = m.part_sites().expect("M's edit sites");
    let dot = |a: &A<f64>, b: &A<f64>| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-12), "{what}: ⟨tangent, ḡ⟩ {a}, ⟨direction, reverse⟩ {b}");
    let rows = 6;
    // Writes: an amplification by 3 at row 1, a swap at row 2 read at row 4, a removal at rows 3 and 5.
    {
        let (read, out) = sites.nodes(1).expect("layer 0's MLP");
        let edits = [(1, Edit::Part { part: 0, factor: 3 }), (2, Edit::SwapAt { part: 1, from: 4 }), (3, Edit::PartAt { part: 0, factor: 0, at: 3 }), (5, Edit::PartAt { part: 0, factor: 0, at: 5 })];
        let e = Edits::with_parts(&d, &[], m.values(), &edits, Some(sites), None).expect("the edits");
        let (xs, dx, g) = (random(rows, width, 2.0), random(rows, width, 1.0), random(rows, width, 1.0));
        let xt = d.upload(xs.view()).expect("upload");
        let mut value = d.zeros(rows, width).expect("zeros");
        e.write(&d, out, &mut value, |n| if n == read { Ok(&xt) } else { Err("another node".into()) }).expect("the forward");
        // The tangent of the added rows: Σ s 1[g·x + c > 0] (g·dx[from]) u at each edited row.
        let mut tangent = A::zeros((rows, width));
        let pre = |p: &Part, r: usize| xs.row(r).iter().zip(&p.read).map(|(a, b)| a * b).sum::<f64>() + p.bias;
        let mut add = |row: usize, from: usize, p: &Part, s: f64| {
            if pre(p, from) > 0.0 {
                let c = s * dx.row(from).iter().zip(&p.read).map(|(a, b)| a * b).sum::<f64>();
                tangent.row_mut(row).iter_mut().zip(&p.write).for_each(|(t, u)| *t += c * u);
            }
        };
        add(1, 1, &parts[0], FACTORS[3] - 1.0);
        add(2, 4, &parts[1], 1.0);
        add(2, 2, &parts[1], -1.0);
        add(3, 3, &parts[0], -1.0);
        add(5, 5, &parts[0], -1.0);
        let mut gout = d.upload(g.view()).expect("upload");
        e.transpose(&d, out, &mut gout).expect("the output's transpose");
        assert_eq!(d.download(&gout).expect("download"), g, "the output's own cotangent passes");
        let mut gread = d.zeros(rows, width).expect("zeros");
        e.transpose(&d, read, &mut gread).expect("the read's transpose");
        close(dot(&tangent, &g), dot(&dx, &d.download(&gread).expect("download")), "writes");
    }
    // A head's removal at rows 1 and 4: the masks' transpose.
    {
        let head = x.heads().iter().position(|(_, held)| *held).expect("a head");
        let e = Edits::with_parts(&d, &[], m.values(), &[(1, Edit::HeadAt { head, at: 1 }), (4, Edit::HeadAt { head, at: 4 })], Some(sites), None).expect("the edits");
        let (node, w) = sites.head(head).expect("the head's node");
        let (dv, g) = (random(rows, w, 1.0), random(rows, w, 1.0));
        let mut t = d.upload(dv.view()).expect("upload");
        e.apply(&d, node, &mut t).expect("the masks");
        let mut gt = d.upload(g.view()).expect("upload");
        e.transpose(&d, node, &mut gt).expect("the masks' transpose");
        close(dot(&d.download(&t).expect("download"), &g), dot(&dv, &d.download(&gt).expect("download")), "head removal");
    }
    // A cut from part 0 (layer 0's MLP; base row 1, source row 4 there) to part 2 (layer 1's MLP, row 2).
    {
        let records: super::interchange::Records = std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeMap::new()));
        let ((read_a, out_a), (_, out_b)) = (sites.nodes(1).expect("layer 0's MLP"), sites.nodes(3).expect("layer 1's MLP"));
        let norm = sites.norm(3).expect("layer 1's input norm");
        let probes = [(1, Edit::Probe { cut: 0, part: 0, role: 0, at: 1 }), (4, Edit::Probe { cut: 0, part: 0, role: 1, at: 4 })];
        let ea = Edits::with_parts(&d, &[], m.values(), &probes, Some(sites), Some(std::rc::Rc::clone(&records))).expect("the probes");
        let eb = Edits::with_parts(&d, &[], m.values(), &[(2, Edit::Cut { cut: 0, from: 0, to: 2 })], Some(sites), Some(std::rc::Rc::clone(&records))).expect("the cut");
        let (xa, dxa, s, ds, g) = (random(rows, width, 2.0), random(rows, width, 1.0), random(rows, width, 4.0), random(rows, width, 1.0), random(rows, width, 1.0));
        let (xat, st) = (d.upload(xa.view()).expect("upload"), d.upload(s.view()).expect("upload"));
        let mut value = d.zeros(rows, width).expect("zeros");
        ea.write(&d, out_a, &mut value, |n| if n == read_a { Ok(&xat) } else { Err("another node".into()) }).expect("the probes' forward");
        let mut value = d.zeros(rows, width).expect("zeros");
        eb.write(&d, out_b, &mut value, |n| if n == norm.entry() { Ok(&st) } else { Err("another node".into()) }).expect("the cut's forward");
        // The tangent from the definition: y = s + (a′ − a) u_A, out += (relu(g_B·N(y) + c) − relu(g_B·N(s) + c)) u_B.
        let (pa, pb) = (&parts[0], &parts[2]);
        let lin = |v: &[f64], w: &[f64]| v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>();
        let act = |r: usize| lin(xa.row(r).as_slice().unwrap(), &pa.read) + pa.bias;
        let dact = |r: usize| if act(r) > 0.0 { lin(dxa.row(r).as_slice().unwrap(), &pa.read) } else { 0.0 };
        let (a, a2) = (act(1).max(0.0), act(4).max(0.0));
        let srow = s.row(2).to_vec();
        let y: Vec<f64> = srow.iter().zip(&pa.write).map(|(v, u)| v + (a2 - a) * u).collect();
        let dy: Vec<f64> = ds.row(2).iter().zip(&pa.write).map(|(v, u)| v + (dact(4) - dact(1)) * u).collect();
        let on = |z: &[f64]| lin(&norm.apply(z), &pb.read) + pb.bias > 0.0;
        let dchange = if on(&y) { lin(&norm.tangent(&y, &dy), &pb.read) } else { 0.0 } - if on(&srow) { lin(&norm.tangent(&srow, ds.row(2).as_slice().unwrap()), &pb.read) } else { 0.0 };
        let mut tangent = A::zeros((rows, width));
        tangent.row_mut(2).iter_mut().zip(&pb.write).for_each(|(t, u)| *t = dchange * u);
        let mut gout = d.upload(g.view()).expect("upload");
        eb.transpose(&d, out_b, &mut gout).expect("the cut's transpose");
        let mut gs = d.zeros(rows, width).expect("zeros");
        eb.add_entering(&d, &mut gs).expect("the entering stream's share");
        let mut ga = d.zeros(rows, width).expect("zeros");
        ea.transpose(&d, read_a, &mut ga).expect("the probes' transpose");
        let back = dot(&ds, &d.download(&gs).expect("download")) + dot(&dxa, &d.download(&ga).expect("download"));
        assert!(dchange != 0.0, "the cut's tangent is not zero");
        close(dot(&tangent, &g), back, "cut connection");
    }
}


/// Operations on shared sites ([`Patch::Ops`], [`Interchange::sample_ops`]) on the scoped starting
/// library of the tiny Qwen3 export (its MLPs P's, an exact copy of M's; its attention M's): every
/// family (swap from a donor, zeroing, scaling, pushing a direction) at one row, onward and at
/// every row, with one to many operations, scores zero against M with and without the reverse
/// pass, the draws reach several operations at once, and with P applying no edit most experiments
/// move M.
#[test]
fn operations_on_shared_sites_are_the_same_on_both_models() {
    use super::interchange::Family;
    let dir = crate::test_support::tiny_qwen3_export("interchange_ops", 2);
    let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
    std::fs::remove_dir_all(dir).expect("the tiny export is removed");
    let native = split_sites(&imported.program).expect("the native sites");
    let layers = layer_nodes(&native, 2).expect("the layers");
    let explanation = crate::library_mdl::scoped(&crate::library_mdl::explanation(&native, &layers).expect("the library"), &[1, 3]).expect("scoped");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let d = Device::host();
    let blocks: Vec<crate::run_check::LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
    let variables = reads(&native, &blocks).expect("the reads");
    let mut x = Interchange::new(&d, &native, &blocks, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
    let shared = x.shared_sites();
    assert!(shared.len() >= 4 + 4 + 2, "{shared:?}");
    let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
    x.set_directions(8, 1);
    x.measure_typical(&batch).expect("the typical norms");
    let families = [Family::Swap, Family::Zero, Family::Scale, Family::Push];
    let experiments = x.sample_ops(&mut rand::rngs::StdRng::seed_from_u64(3), &batch, &families, 12, &[1, 2, 0], false).expect("the draw");
    let counts = census(&experiments, x.variables());
    assert!(families.iter().all(|f| counts[format!("{f:?}").to_lowercase().as_str()] > 0), "{counts:?}");
    assert!(experiments.iter().any(|e| matches!(&e.patch, Some(Patch::Ops { ops, .. }) if ops.len() > 2)), "several operations at once");
    for gradient in [false, true] {
        let bits = x.evaluate(&batch, &experiments, gradient).expect("evaluate").bits;
        assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
    }
    x.unedited_explanation();
    let effects = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
    let moved = experiments.iter().zip(&effects).filter(|(e, bits)| e.patch.is_some() && bits.iter().sum::<f64>() > 1e-9).count();
    let edited = experiments.iter().filter(|e| e.patch.is_some()).count();
    assert!(moved * 10 >= edited * 8, "{moved} of {edited} operations move M");
}
