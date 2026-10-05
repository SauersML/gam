#![cfg(test)]
//! Interchange experiments on the device against an independent host computation, and their
//! gradient against central differences, on a small rotary language model (two layers, tied
//! unembedding) whose explanation `P` is the model with perturbed query, key, value and MLP input
//! maps.
//!
//! The host reference runs each model's program with `OperatorProgram::execute_edited`: a hybrid
//! runs block by block, each block's model entering at the stream the previous block left; a patch
//! replaces the read node's value, its projector formed from a QR decomposition of the read rows
//! (not the device's singular value decomposition); and the divergence is formed from full softmax
//! distributions (not the compact head statistics). Both sides evaluate the same expressions in float64 in different orders; each
//! value then differs by a few units of rounding of its widest sum (at most `d + V` terms here,
//! `V` classes), far below the `1e-9` bits the comparison allows and far below any effect of a
//! patch.

use super::artifact::Artifact;
use super::device_program::DeviceProgram;
use super::device_program_tests::{devices, fixture_sized, noise};
use super::interchange::{Batch, Design, Experiment, FixedHead, Interchange, Model, Patch, ReadVariable, Targets, census, design, evaluate, fingerprint, sample, sites, targets};
use super::operator_program::{FamilyInputs, Operator, OperatorProgram, SequenceLayout, SlotValues, exact_precision};
use super::resident_causal_fit::fixed_head_target::Head;
use super::run_check::{layer_nodes, split_sites};
use gam_gpu::tensor::Device;
use gam_math::categorical::categorical_kl_from_logits;
use gam_linalg::decompose::{QrMode, qr};
use ndarray::{Array2, Axis, s};
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

/// The native model `M`, its explanation `P` (perturbed maps), `P`'s read variables and trainable
/// operators, and the shared head.
struct Fixture {
    m: Host,
    p: Host,
    variables: Vec<ReadVariable>,
    trainable: Vec<usize>,
    head: Head,
    batch: Batch,
}

fn fixture() -> Fixture {
    let (program, family) = fixture_sized(D, VOCAB, LENGTH, 6);
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
    // A variable read through two operator parts, as a gated MLP function's is (at a block no
    // complement patch below spans, since its rows repeat other variables').
    let up = flat.operators.iter().position(|op| op.name == "blocks.0.c_fc").expect("layer 0 MLP input");
    variables.push(ReadVariable { block: 1, parts: vec![(up, 1..2), (up, 4..6)] });
    let p = Host { prefix: head.prefix(&p_flat), flat: p_flat, entries, reads };
    let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("tokens") };
    let base: Vec<Vec<u32>> = tokens.chunks(LENGTH).map(<[u32]>::to_vec).collect();
    let source: Vec<Vec<u32>> = base.iter().map(|s| s.iter().map(|t| (t + 5) % VOCAB as u32).rev().collect()).collect();
    Fixture { m, p, variables, trainable, head, batch: Batch::new(base, source).expect("batch") }
}

fn one(tokens: &[u32]) -> FamilyInputs {
    FamilyInputs {
        rows: tokens.len(),
        slots: vec![SlotValues::Tokens(tokens.to_vec())],
        layout: Some(SequenceLayout { sequence: vec![0; tokens.len()], position: (0..tokens.len() as u32).collect() }),
    }
}

/// The projector onto the span of `rows` (`complement` false) as the reference forms it: `Q Qᵀ`
/// from a QR decomposition of the rows' transpose, which here have full rank or span everything.
fn projector(rows: &Array2<f64>) -> Array2<f64> {
    let q = qr(rows.t(), QrMode::Economic).expect("qr").q.expect("q");
    q.dot(&q.t())
}

/// A patch as the reference applies it: block, projector, complement, the source's read, and the
/// one row it replaces.
type HostPatch<'a> = (usize, &'a Array2<f64>, bool, &'a Array2<f64>, usize);

fn patched(h: &Array2<f64>, (_, pi, complement, s, at): HostPatch<'_>) -> Array2<f64> {
    let (hr, sr) = (h.row(at).to_owned(), s.row(at).to_owned());
    let row = if complement { &sr + &(&hr - &sr).dot(pi) } else { &hr + &(&sr - &hr).dot(pi) };
    let mut out = h.clone();
    out.row_mut(at).assign(&row);
    out
}

/// The hybrid running `P`'s version of the blocks `explained` marks and `M`'s of the others, on
/// `tokens`, under `patches`: its hidden rows and its read at every block. Each block is one run of
/// its model's whole program with the stream entering the block replaced by the previous block's
/// output, of which only the block's own nodes are kept.
fn hybrid(f: &Fixture, tokens: &[u32], explained: &[bool], patches: &[HostPatch<'_>]) -> (Array2<f64>, Vec<Array2<f64>>) {
    let family = one(tokens);
    let blocks = 2 * LAYERS;
    let (mut state, mut reads) = (None::<Array2<f64>>, Vec::with_capacity(blocks));
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
                if let Some(patch) = patches.iter().find(|p| p.0 == b)
                    && node == host.reads[b]
                {
                    *value = patched(value, *patch);
                }
                Ok(())
            })
            .expect("block");
        reads.push(run.values[host.reads[b]].clone());
        state = Some(run.values[end].clone());
    }
    (state.expect("a block ran"), reads)
}

/// `M` alone (the hybrid with no block of `P`).
fn native(f: &Fixture, tokens: &[u32], patches: &[HostPatch<'_>]) -> (Array2<f64>, Vec<Array2<f64>>) {
    hybrid(f, tokens, &[false; 2 * LAYERS], patches)
}

/// Per row `KL(softmax(E a) ‖ softmax(E b))` in bits.
fn kl_bits(head: &Head, a: &Array2<f64>, b: &Array2<f64>) -> Vec<f64> {
    let (za, zb) = (a.dot(&head.embedding.t()), b.dot(&head.embedding.t()));
    za.outer_iter()
        .zip(zb.outer_iter())
        .map(|(ra, rb)| categorical_kl_from_logits(&ra.to_vec(), &rb.to_vec()).expect("finite logits") / std::f64::consts::LN_2)
        .collect()
}

/// The reference bits of experiment `e`.
fn reference(f: &Fixture, e: &Experiment) -> Vec<f64> {
    let (base, source) = (&f.batch.base[e.base], &f.batch.source[e.source]);
    let Some(patch) = &e.patch else {
        return kl_bits(&f.head, &native(f, base, &[]).0, &hybrid(f, base, &e.explained, &[]).0)[e.position..].to_vec();
    };
    let rows_of = |v: &ReadVariable| -> Vec<Array2<f64>> { v.parts.iter().map(|(op, rows)| f.p.flat.operators[*op].matrix().slice(s![rows.clone(), ..]).to_owned()).collect() };
    // Per patched block: its rows, and whether the patch takes their complement.
    let sites: Vec<(usize, Vec<Array2<f64>>, bool)> = match patch {
        Patch::Read { variable } => vec![(f.variables[*variable].block, rows_of(&f.variables[*variable]), false)],
        Patch::Complement { blocks } => blocks.iter().map(|&b| (b, f.variables.iter().filter(|v| v.block == b).flat_map(rows_of).collect(), true)).collect(),
    };
    let projectors: Vec<Array2<f64>> = sites
        .iter()
        .map(|(_, parts, _)| {
            let views: Vec<_> = parts.iter().map(|m| m.view()).collect();
            let rows = ndarray::concatenate(Axis(0), &views).expect("rows");
            // More rows than coordinates span everything: the projector is the identity.
            if rows.nrows() >= D { Array2::eye(D) } else { projector(&rows) }
        })
        .collect();
    let (m_reads, p_reads) = (native(f, source, &[]).1, hybrid(f, source, &e.explained, &[]).1);
    let m_patches: Vec<HostPatch<'_>> = sites.iter().zip(&projectors).map(|((b, _, c), pi)| (*b, pi, *c, &m_reads[*b], e.position)).collect();
    let p_patches: Vec<HostPatch<'_>> = sites.iter().zip(&projectors).map(|((b, _, c), pi)| (*b, pi, *c, &p_reads[*b], e.position)).collect();
    kl_bits(&f.head, &native(f, base, &m_patches).0, &hybrid(f, base, &e.explained, &p_patches).0)[e.position..].to_vec()
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
        e(0, 2, [t, t, n, n], Some(Patch::Complement { blocks: vec![0] }), 0),
        e(1, 3, [t, t, t, t], Some(Patch::Complement { blocks: vec![2] }), 2),
        e(2, 0, [t, t, n, n], Some(Patch::Complement { blocks: vec![3] }), 5),
        e(5, 1, [n, t, t, n], Some(Patch::Complement { blocks: vec![0] }), 3),
        // Joint complements at several blocks, on both sides of the switches.
        e(4, 0, [t, n, t, n], Some(Patch::Complement { blocks: vec![0, 3] }), 1),
        e(3, 5, [n, t, t, t], Some(Patch::Complement { blocks: vec![0, 2, 3] }), 0),
        e(0, 4, [t, t, t, t], Some(Patch::Complement { blocks: vec![2, 3] }), 4),
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
        Model::new(&programs.m, &f.m.flat, f.m.entries.clone(), f.m.reads.clone(), &[]).expect("M model"),
        Model::new(&programs.p, &f.p.flat, f.p.entries.clone(), f.p.reads.clone(), &f.trainable).expect("P model"),
    )
}

#[test]
fn every_experiment_matches_the_host_reference() {
    let f = fixture();
    let experiments = experiments(&f);
    for device in devices() {
        let programs = programs(&device, &f);
        let (m, p) = models(&f, &programs);
        let design = design(&p, &f.variables, &experiments).expect("design");
        let targets = targets(&m, &programs.head, &f.batch, &experiments, &design).expect("targets");
        let evaluation = evaluate(&m, &p, &programs.head, &f.batch, &targets, &experiments, &design, false).expect("evaluate");
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
    let mm = Model::new(&m, &f.m.flat, f.m.entries.clone(), f.m.reads.clone(), &[]).expect("M model");
    let pm = Model::new(&p, &f.m.flat, f.m.entries.clone(), f.m.reads.clone(), &f.trainable).expect("P model");
    let experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(5), f.batch.base.len(), &f.variables, &[true; 2 * LAYERS], LENGTH).expect("sample");
    let design = design(&pm, &f.variables, &experiments).expect("design");
    let targets = targets(&mm, &head, &f.batch, &experiments, &design).expect("targets");
    let evaluation = evaluate(&mm, &pm, &head, &f.batch, &targets, &experiments, &design, true).expect("evaluate");
    assert!(evaluation.bits.iter().flatten().all(|b| b.abs() <= 1e-12), "{:?}", evaluation.bits);
    for g in evaluation.gradient.values() {
        assert!(device.download(g).expect("gradient").iter().all(|v| v.abs() <= 1e-9));
    }
}

/// The total bits at `P`'s operator `op` moved to `value`, the design held fixed.
fn total_at(device: &Device, f: &Fixture, programs: &mut Programs, op: usize, value: &Array2<f64>, experiments: &[Experiment], design: &Design) -> f64 {
    programs.p.replace_dense_parameter(op, device.upload(value.view()).expect("upload")).expect("replace");
    let (m, p) = models(f, programs);
    let targets = targets(&m, &programs.head, &f.batch, experiments, design).expect("targets");
    let evaluation = evaluate(&m, &p, &programs.head, &f.batch, &targets, experiments, design, false).expect("evaluate");
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
    let (design, gradient) = {
        let (m, p) = models(&f, &programs);
        let design = design(&p, &f.variables, &experiments).expect("design");
        let targets = targets(&m, &programs.head, &f.batch, &experiments, &design).expect("targets");
        let evaluation = evaluate(&m, &p, &programs.head, &f.batch, &targets, &experiments, &design, true).expect("evaluate");
        (design, evaluation.gradient)
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
        let numeric = (total_at(&device, &f, &mut programs, op, &up, &experiments, &design) - total_at(&device, &f, &mut programs, op, &down, &experiments, &design)) / (2.0 * delta);
        total_at(&device, &f, &mut programs, op, &start, &experiments, &design);
        let scale = analytic.abs().max(numeric.abs()).max(1.0);
        assert!((analytic - numeric).abs() <= 1e-6 * scale, "operator {} entry ({i}, {j}): analytic {analytic}, numeric {numeric}", f.p.flat.operators[op].name);
    }
}

#[test]
fn sampling_draws_each_family_with_its_stated_weight() {
    // Six blocks, three read variables at each except block 1, which has none; blocks 1 and 3
    // leave a complement. Per base: P alone with probability 1/2; the block uniform; at block 1
    // the complement, at block 3 the complement with probability 1/2, elsewhere a read.
    let variables: Vec<ReadVariable> = (0..6).filter(|b| *b != 1).flat_map(|b| (0..3).map(move |i| ReadVariable { block: b, parts: vec![(b, i..i + 1)] })).collect();
    let complements = [false, true, false, true, false, false];
    let n = 6000;
    let experiments = sample(&mut rand::rngs::StdRng::seed_from_u64(11), n, &variables, &complements, 9).expect("sample");
    assert_eq!(experiments.len(), 2 * n);
    for (i, pair) in experiments.chunks(2).enumerate() {
        assert_eq!((pair[0].base, pair[1].base, pair[1].source), (i, i, i));
        assert!(pair[0].patch.is_none() && pair[1].patch.is_some());
        // The patched experiment shares its base's hybrid; the clean one is scored everywhere.
        assert_eq!(pair[0].explained, pair[1].explained);
        assert!(pair[0].position == 0 && pair[1].position < 9);
        assert!(pair[0].explained.len() == 6 && pair[0].explained.contains(&true));
        if let Some(Patch::Complement { blocks }) = &pair[1].patch {
            assert!(blocks.len() == 1 && complements[blocks[0]]);
        }
    }
    // Each count within five standard deviations of its binomial expectation.
    let within = |count: usize, p: f64| {
        let (mean, sd) = (n as f64 * p, (n as f64 * p * (1.0 - p)).sqrt());
        assert!((count as f64 - mean).abs() <= 5.0 * sd, "{count} against {mean} ± {sd}");
    };
    let counts = census(&experiments, &variables);
    // A hybrid draw is P alone with probability 1/6 (its size uniform in 1..=6).
    within(counts["clean_alone"], 0.5 + 0.5 / 6.0);
    within(counts["complement"], 1.0 / 6.0 + 0.5 / 6.0);
    within(counts["read_attention"], 3.0 / 6.0);
    within(counts["read_mlp"], 0.5 / 6.0 + 1.0 / 6.0);
    // Every size 1..=6 appears, and a non-prefix set does: the hybrids are not only cuts.
    let sizes: std::collections::BTreeSet<usize> = experiments.iter().map(|e| e.explained.iter().filter(|x| **x).count()).collect();
    assert_eq!(sizes, (1..=6).collect());
    assert!(experiments.iter().any(|e| e.explained.windows(2).any(|w| !w[0] && w[1])));
    let positions: std::collections::BTreeSet<usize> = experiments.iter().filter(|e| e.patch.is_some()).map(|e| e.position).collect();
    assert_eq!(positions, (0..9).collect());
}

#[test]
fn targets_written_and_read_score_alike() {
    let f = fixture();
    let device = Device::host();
    let experiments = experiments(&f);
    let programs = programs(&device, &f);
    let (m, p) = models(&f, &programs);
    let design = design(&p, &f.variables, &experiments).expect("design");
    let made = targets(&m, &programs.head, &f.batch, &experiments, &design).expect("targets");
    let path = std::env::temp_dir().join(format!("interchange_targets_{}.bin", std::process::id()));
    let key = fingerprint(&f.batch, &experiments);
    made.write(&device, &path, key).expect("write");
    let read = Targets::read(&device, &programs.head, &path, key).expect("read").expect("the same experiments");
    // Targets made for other experiments are not read back.
    assert!(Targets::read(&device, &programs.head, &path, fingerprint(&f.batch, &experiments[1..])).expect("read").is_none());
    std::fs::remove_file(&path).expect("remove");
    let a = evaluate(&m, &p, &programs.head, &f.batch, &made, &experiments, &design, false).expect("evaluate").bits;
    let b = evaluate(&m, &p, &programs.head, &f.batch, &read, &experiments, &design, false).expect("evaluate").bits;
    assert_eq!(a, b);
}

#[test]
fn a_device_model_refuses_a_layer_reading_before_its_stream() {
    let f = fixture();
    let device = Device::host();
    let m = DeviceProgram::compile_values(&device, &f.m.prefix).expect("M");
    let mut reads = f.m.reads.clone();
    reads.swap(0, 2);
    assert!(Model::new(&m, &f.m.flat, f.m.entries.clone(), reads, &[]).is_err());
    let mut entries = f.m.entries.clone();
    entries[2] = f.m.reads[3];
    assert!(Model::new(&m, &f.m.flat, entries, f.m.reads.clone(), &[]).is_err());
}

#[test]
fn directions_from_host_values_are_those_of_the_loaded_explanation() {
    // The owner starts at M and loads P's maps; directions read from the host values it was given
    // must be the device's own after the load, and score as the host reference does.
    let f = fixture();
    let device = Device::host();
    let (program, _) = fixture_sized(D, VOCAB, LENGTH, 6);
    let native = split_sites(&program).expect("split");
    let layers = layer_nodes(&native, LAYERS).expect("layers");
    let explanation = Artifact::native(&native).expect("native artifact");
    let mut x = Interchange::new(&device, &native, &layers, &explanation, &f.trainable, f.variables.clone(), usize::MAX, 5).expect("interchange");
    let values: Vec<Array2<f64>> = f.trainable.iter().map(|op| f.p.flat.operators[*op].matrix()).collect();
    let experiments = experiments(&f);
    let at_values = x.design_at(&f.variables, &experiments, &values).expect("design at values");
    x.load(&values).expect("load");
    let (m, p) = x.models();
    let loaded = design(&p, &f.variables, &experiments).expect("design");
    let at_targets = targets(&m, x.head(), &f.batch, &experiments, &at_values).expect("targets");
    let loaded_targets = targets(&m, x.head(), &f.batch, &experiments, &loaded).expect("targets");
    let from_values = evaluate(&m, &p, x.head(), &f.batch, &at_targets, &experiments, &at_values, false).expect("evaluate").bits;
    let from_device = evaluate(&m, &p, x.head(), &f.batch, &loaded_targets, &experiments, &loaded, false).expect("evaluate").bits;
    assert_eq!(from_values, from_device);
    for (e, bits) in experiments.iter().zip(&from_values) {
        for (a, b) in bits.iter().zip(reference(&f, e)) {
            assert!((a - b).abs() <= 1e-9, "{e:?}: device {a} bits, host {b} bits");
        }
    }
}
