#![cfg(test)]
//! The box claim's certificate against every point an adversary or a draw finds in the box, on small
//! programs with every node kind the certificate reads.

use super::certify::{Curve, GELU_CURVATURE, Gates, Relaxation, SILU_CURVATURE, adversary, certify, certify_branching, linearize};
use super::masked::{Library, Masked, Target, forward, matrix, sites};
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance,
    Rotary, Scale, SequenceLayout, Slot, SlotValues,
};
use super::precision::DeclaredPrecision;
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn precision() -> DeclaredPrecision {
    DeclaredPrecision::new(40).expect("a precision")
}

fn dense(name: &str, rows: &Interface, cols: &Interface, salt: usize, scale: f64) -> Arc<Operator> {
    let m = Array2::from_shape_fn((rows.width(), cols.width()), |(i, j)| scale * noise(salt + 31 * i + j));
    Arc::new(Operator::dense(name, rows.clone(), cols.clone(), m, precision(), Provenance::default()).expect("dense"))
}

/// Two tokens embedded, two residual MLP layers (`law`), logits over five classes.
fn mlp(law: Law) -> (OperatorProgram, FamilyInputs) {
    const P: usize = 5;
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(4).expect("interface");
    let units = Interface::uniform(6, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let program = OperatorProgram {
        declarations: Declarations {
            parameters: 0,
            domains: vec![Domain { size: P, cycle: None }],
            slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
        },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![
            dense("E", &residual, &tokens, 1, 1.0),
            dense("Win", &units, &residual, 100, 1.0),
            dense("Wmid", &residual, &units, 200, 0.7),
            dense("Wout", &classes, &residual, 300, 1.5),
            Arc::new(Operator::identity("I", residual.clone())),
        ],
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },
            Node::Feature { slot: 1, basis: 0 },
            Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
            Node::Affine { terms: vec![(2, 1)], bias: None },
            Node::Pointwise { input: 3, laws: vec![law; 6] },
            Node::Affine { terms: vec![(2, 4), (4, 2)], bias: None },
            Node::Affine { terms: vec![(5, 3)], bias: None },
            Node::Readout { input: 6, basis: 0 },
        ],
        output: 7,
    };
    let pairs: Vec<(u32, u32)> = (0..P as u32).flat_map(|a| (0..P as u32).map(move |b| (a, b))).step_by(3).collect();
    let family = FamilyInputs {
        layout: None,
        rows: pairs.len(),
        slots: vec![SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()), SlotValues::Tokens(pairs.iter().map(|q| q.1).collect())],
    };
    (program, family)
}

/// A one-block decoder over sequences: RMS norms, rotary causal attention, a tanh-GELU MLP, a tied readout.
fn decoder() -> (OperatorProgram, FamilyInputs) {
    const VOCAB: usize = 7;
    let tokens = Interface::uniform(VOCAB, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(6).expect("interface");
    let head = Interface::native(4).expect("interface");
    let units = Interface::uniform(8, 1, LabelKind::Unit, 0).expect("interface");
    let program = OperatorProgram {
        declarations: Declarations { parameters: 0, domains: vec![Domain { size: VOCAB, cycle: None }], slots: vec![Slot::Token { domain: 0 }] },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![
            dense("E", &residual, &tokens, 7, 1.0),
            dense("Wq", &head, &residual, 400, 1.0),
            dense("Wk", &head, &residual, 500, 1.0),
            dense("Wv", &head, &residual, 600, 1.0),
            dense("Wo", &residual, &head, 700, 0.8),
            dense("Wup", &units, &residual, 800, 1.0),
            dense("Wdown", &residual, &units, 900, 0.6),
            Arc::new(Operator::identity("I", residual.clone())),
        ],
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::RmsNorm { input: 1, epsilon: 1e-5 },
            Node::Affine { terms: vec![(2, 1)], bias: None },
            Node::Affine { terms: vec![(2, 2)], bias: None },
            Node::Affine { terms: vec![(2, 3)], bias: None },
            Node::Attend {
                query: 3,
                key: 4,
                value: 5,
                scale: Scale::InverseSqrt(4),
                rotary: Some(Rotary { base: 10000, dims: 4, half_split: true }),
                causal: true,
            },
            Node::Affine { terms: vec![(1, 7), (6, 4)], bias: None },
            Node::RmsNorm { input: 7, epsilon: 1e-5 },
            Node::Affine { terms: vec![(8, 5)], bias: None },
            Node::Pointwise { input: 9, laws: vec![Law::GeluTanh; 8] },
            Node::Affine { terms: vec![(7, 7), (10, 6)], bias: None },
            Node::RmsNorm { input: 11, epsilon: 1e-5 },
            Node::Transposed { input: 12, operator: 0 },
            Node::Readout { input: 13, basis: 0 },
        ],
        output: 14,
    };
    let (sequences, positions) = (2usize, 4usize);
    let ids: Vec<u32> = (0..sequences * positions).map(|i| ((noise(i + 50) + 1.0) * 3.5) as u32 % VOCAB as u32).collect();
    let family = FamilyInputs {
        rows: ids.len(),
        slots: vec![SlotValues::Tokens(ids)],
        layout: Some(SequenceLayout {
            sequence: (0..sequences * positions).map(|i| (i / positions) as u32).collect(),
            position: (0..sequences * positions).map(|i| (i % positions) as u32).collect(),
        }),
    };
    (program, family)
}

/// Attention unrolled over three token positions (bilinear scores, a softmax, a mix), then a SiLU MLP.
fn unrolled() -> (OperatorProgram, FamilyInputs) {
    const P: usize = 5;
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(4).expect("interface");
    let head = Interface::native(3).expect("interface");
    let units = Interface::uniform(5, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let constant = Interface::constant();
    let mut operators = vec![
        dense("E", &residual, &tokens, 11, 1.0),
        dense("Wq", &head, &residual, 1000, 1.2),
        dense("Wk", &head, &residual, 1100, 1.2),
        dense("Wv", &head, &residual, 1200, 1.0),
        dense("Wo", &residual, &head, 1300, 0.8),
        dense("Win", &units, &residual, 1400, 1.0),
        dense("Wout", &residual, &units, 1500, 0.7),
        dense("WU", &classes, &residual, 1600, 1.5),
        Arc::new(Operator::identity("I", residual.clone())),
    ];
    for j in 0..3 {
        operators.push(dense(&format!("pos{j}"), &residual, &constant, 1700 + 10 * j, 0.5));
    }
    let mut nodes = Vec::new();
    let mut push = |node: Node| {
        nodes.push(node);
        nodes.len() - 1
    };
    let mut x = Vec::new();
    for j in 0..3 {
        let feature = push(Node::Feature { slot: j, basis: 0 });
        x.push(push(Node::Affine { terms: vec![(feature, 0)], bias: Some(9 + j) }));
    }
    let q = push(Node::Affine { terms: vec![(x[2], 1)], bias: None });
    let keys: Vec<usize> = x.iter().map(|&xj| push(Node::Affine { terms: vec![(xj, 2)], bias: None })).collect();
    let values: Vec<usize> = x.iter().map(|&xj| push(Node::Affine { terms: vec![(xj, 3)], bias: None })).collect();
    let scores: Vec<usize> = keys.iter().map(|&k| push(Node::Bilinear { left: q, right: k, scale: Scale::InverseSqrt(3) })).collect();
    let weights = push(Node::Softmax { scores });
    let read = push(Node::Mix { weights, payloads: values.iter().enumerate().map(|(c, &v)| (c, v)).collect() });
    let attended = push(Node::Affine { terms: vec![(x[2], 8), (read, 4)], bias: None });
    let pre = push(Node::Affine { terms: vec![(attended, 5)], bias: None });
    let act = push(Node::Pointwise { input: pre, laws: vec![Law::Silu; 5] });
    let out = push(Node::Affine { terms: vec![(attended, 8), (act, 6)], bias: None });
    let logits = push(Node::Affine { terms: vec![(out, 7)], bias: None });
    let output = push(Node::Readout { input: logits, basis: 0 });
    let program = OperatorProgram {
        declarations: Declarations { parameters: 0, domains: vec![Domain { size: P, cycle: None }], slots: vec![Slot::Token { domain: 0 }; 3] },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators,
        rules: Vec::new(),
        nodes,
        output,
    };
    let rows = 9;
    let slot = |j: usize| SlotValues::Tokens((0..rows).map(|r| ((r * (j + 2) + j) % P) as u32).collect());
    let family = FamilyInputs { rows, slots: vec![slot(0), slot(1), slot(2)], layout: None };
    (program, family)
}

/// Every site replaced by its columns (`u_c = W_{:,c}`, `v_c = e_c`), masks drawn from `seed`, the native logits
/// and their radius.
fn masked(program: &OperatorProgram, family: &FamilyInputs, seed: usize) -> (Masked, Vec<Array2<f64>>, Target, Array2<f64>) {
    let chosen = sites(program);
    assert!(!chosen.is_empty());
    let libraries: Vec<Library> = chosen
        .iter()
        .map(|site| {
            let w = matrix(program, site).expect("site matrix");
            Library { v: Array2::eye(w.ncols()), u: w.t().to_owned(), mean: Array1::zeros(w.ncols()) }
        })
        .collect();
    let masked = Masked::build(program, chosen, libraries).expect("masked");
    let masks: Vec<Array2<f64>> = (0..masked.sites.len())
        .map(|k| Array2::from_shape_fn((family.rows, masked.blocks(k)), |(r, c)| if noise(seed + 97 * k + 13 * r + c) > 0.0 { 1.0 } else { 0.0 }))
        .collect();
    let trace = program.execute(family, true).expect("native");
    let target = Target::every_row(trace.values[program.output].clone());
    let radius = trace.band(program.output).expect("bands");
    (masked, masks, target, radius)
}

fn fixtures() -> Vec<(&'static str, OperatorProgram, FamilyInputs)> {
    let mut out = Vec::new();
    for (name, law) in [("relu mlp", Law::Relu), ("gelu mlp", Law::Gelu), ("silu mlp", Law::Silu)] {
        let (p, f) = mlp(law);
        out.push((name, p, f));
    }
    let (p, f) = decoder();
    out.push(("decoder", p, f));
    let (p, f) = unrolled();
    out.push(("unrolled attention", p, f));
    out
}

/// The certificate is at or above the KL of the masks, of draws inside the box and of every point an adversary
/// visits, with no symbol dropped, with most enclosed along principal directions, and with nearly every one dropped.
#[test]
fn certificates_contain_every_point_found_in_the_box() {
    for (name, program, family) in fixtures() {
        for seed in [3, 71] {
            let (masked, masks, target, radius) = masked(&program, &family, seed);
            let gates = Gates::claim(&masks);
            let found = adversary(&masked, &family, &target, &gates, None, 8, 6, seed as u64).expect("adversary");
            let mut worst = found.clone();
            for draw in 0..20 {
                let point: Vec<Array2<f64>> = masks
                    .iter()
                    .enumerate()
                    .map(|(k, m)| Array2::from_shape_fn(m.dim(), |(r, c)| if m[[r, c]] > 0.0 { 1.0 } else { 0.5 + 0.5 * noise(seed * 7 + draw * 1009 + k * 31 + r * 17 + c) }))
                    .collect();
                let (kl, _, _) = forward(&masked, &masked.family(&family, &point), &target).expect("forward");
                ndarray::Zip::from(&mut worst).and(&kl).for_each(|w, &v| *w = w.max(v));
            }
            for budget in [4096, 24, 2] {
                let bound = certify(&masked, &family, &target, Some(&radius), &gates, Relaxation::sound(budget)).expect("certificate");
                for r in 0..family.rows {
                    assert!(bound[r].is_finite(), "{name}, seed {seed}, budget {budget}: row {r} unbounded");
                    assert!(bound[r] >= worst[r], "{name}, seed {seed}, budget {budget}: row {r} certified {} below a found {}", bound[r], worst[r]);
                }
            }
        }
    }
}

/// With every gate pinned to one value the box is a point, and the certificate is that point's own divergence.
#[test]
fn a_pinned_box_certifies_its_own_divergence() {
    for (name, program, family) in fixtures() {
        let (masked, masks, target, radius) = masked(&program, &family, 5);
        let point: Vec<Array2<f64>> = masks
            .iter()
            .enumerate()
            .map(|(k, m)| Array2::from_shape_fn(m.dim(), |(r, c)| 0.5 + 0.5 * noise(400 + 31 * k + 7 * r + c)))
            .collect();
        let gates = Gates { lower: point.clone(), upper: point.clone() };
        let bound = certify(&masked, &family, &target, Some(&radius), &gates, Relaxation::sound(64)).expect("certificate");
        let (kl, _, _) = forward(&masked, &masked.family(&family, &point), &target).expect("forward");
        for r in 0..family.rows {
            assert!(bound[r] >= kl[r], "{name}: row {r} certified {} below its {}", bound[r], kl[r]);
            assert!(bound[r] <= kl[r] + 1e-9 * (1.0 + kl[r]), "{name}: row {r} certified {} for a point of KL {}", bound[r], kl[r]);
        }
    }
}

/// Branching only lowers the certificate, which still contains what the adversary finds.
#[test]
fn branching_tightens_and_stays_sound() {
    for (name, program, family) in fixtures() {
        let (masked, masks, target, radius) = masked(&program, &family, 11);
        let found = adversary(&masked, &family, &target, &Gates::claim(&masks), None, 8, 4, 11).expect("adversary");
        let branched = certify_branching(&masked, &family, &target, Some(&radius), Gates::claim(&masks), Relaxation::sound(4096), 9).expect("branched");
        for r in 0..family.rows {
            assert!(branched.kl[r] <= branched.root[r], "{name}: row {r} branched {} above its root {}", branched.kl[r], branched.root[r]);
            assert!(branched.kl[r] >= found[r], "{name}: row {r} branched {} below a found {}", branched.kl[r], found[r]);
        }
    }
}

/// Every line a relaxation draws holds its curve within its deviation over the whole interval.
#[test]
fn linearizations_contain_their_curves() {
    let curves = [
        Curve::Law(Law::Relu),
        Curve::Law(Law::Silu),
        Curve::Law(Law::Gelu),
        Curve::Law(Law::GeluTanh),
        Curve::Exp,
        Curve::InverseSqrt,
        Curve::Reciprocal,
    ];
    for (n, curve) in curves.into_iter().enumerate() {
        for trial in 0..60 {
            let a = 6.0 * noise(n * 1000 + 2 * trial);
            let w = 8.0 * (noise(n * 1000 + 2 * trial + 1) + 1.0).powi(3);
            let (l, h) = match curve {
                Curve::InverseSqrt | Curve::Reciprocal => (1e-3 + a.abs(), 1e-3 + a.abs() + w),
                _ => (a - w / 2.0, a + w / 2.0),
            };
            let (lambda, mu, delta) = linearize(curve, l, h).expect("a line");
            let exact = |t: f64| match curve {
                Curve::Law(law) => law.apply(t),
                Curve::Exp => t.exp(),
                Curve::InverseSqrt => 1.0 / t.sqrt(),
                Curve::Reciprocal => 1.0 / t,
            };
            for k in 0..=2000 {
                let t = l + (h - l) * k as f64 / 2000.0;
                let deviation = (exact(t) - (lambda * t + mu)).abs();
                assert!(deviation <= delta * (1.0 + 1e-12) + 1e-12 * exact(t).abs(), "{curve:?} on [{l}, {h}] at {t}: {deviation} > {delta}");
            }
        }
    }
}

/// The second-derivative bounds of SiLU and both GELUs, on a fine grid of central differences.
#[test]
fn second_derivative_bounds_hold() {
    let h = 1e-4;
    for (law, bound) in [(Law::Silu, SILU_CURVATURE), (Law::Gelu, GELU_CURVATURE), (Law::GeluTanh, GELU_CURVATURE)] {
        let mut largest = 0.0_f64;
        for k in 0..=24000 {
            let t = -12.0 + k as f64 * 1e-3;
            let second = (law.apply(t + h) - 2.0 * law.apply(t) + law.apply(t - h)) / (h * h);
            largest = largest.max(second.abs());
        }
        assert!(largest < bound, "{law:?}: |f''| reaches {largest}, bound {bound}");
    }
}
