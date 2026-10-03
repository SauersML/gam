#![cfg(test)]
//! Known answers for the traced path decomposition: on a routing program (softmax routing, an RMS
//! norm, gated units of every law, a Hadamard gate, a gain, a readout and a tied transposed read)
//! and on a causal rotary attend program, paths and remainders sum to the traced value within the
//! derived band at every budget, a remainder is exactly the paths below it and within its bound,
//! the hop flows into the target carry every path once, and a query node splits like any target.

use std::sync::Arc;
use super::*;
use crate::operator_program::{
    Basis, Coefficient, Declarations, Domain, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance,
    Rotary, Scale, SequenceLayout, Slot, SlotValues,
};
use crate::precision::DeclaredPrecision;
use crate::test_support::test_governor;
use gam_linalg::faer_ndarray::FaerSvd;
use gam_linalg::roundoff::factor_singular_band;

fn reals(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i * 13 + j * 7 + salt * 5) as f64 * 0.731).sin() * 1.1)
}

fn dense(name: &str, rows: &Interface, cols: &Interface, salt: usize) -> Operator {
    let precision = DeclaredPrecision::new(24).expect("precision");
    Operator::dense(name, rows.clone(), cols.clone(), reals(rows.width(), cols.width(), salt), precision, Provenance::native(name))
        .expect("a dense operator")
}

/// Two token slots routed by a softmax, a residual, an RMS norm, gated units (every law) times an
/// up read, a gain, logits and a tied transposed read of the embedding, concatenated.
fn routing() -> (OperatorProgram, FamilyInputs) {
    let declarations = Declarations {
        parameters: 0,
        domains: vec![Domain { size: 7, cycle: None }, Domain { size: 5, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
    };
    let tokens = Interface::uniform(7, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(4).expect("interface");
    let head = Interface::native(3).expect("interface");
    let units = Interface::uniform(5, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(5, 1, LabelKind::Token, 0).expect("interface");
    let constant = Interface::constant();
    let operators = vec![
        dense("E", &model, &tokens, 1),        // 0
        dense("pos0", &model, &constant, 2),   // 1
        dense("pos1", &model, &constant, 3),   // 2
        dense("Q", &head, &model, 4),          // 3
        dense("K", &head, &model, 5),          // 4
        dense("V", &head, &model, 6),          // 5
        dense("O", &model, &head, 7),          // 6
        Operator::identity("I", model.clone()), // 7
        dense("b_O", &model, &constant, 8),    // 8
        dense("gate", &units, &model, 9),      // 9
        dense("b_gate", &units, &constant, 10), // 10
        dense("up", &units, &model, 11),       // 11
        dense("down", &model, &units, 12),     // 12
        dense("U", &classes, &model, 13),      // 13
        dense("b_U", &classes, &constant, 14), // 14
    ];
    let nodes = vec![
        Node::Feature { slot: 0, basis: 0 },                                     // 0
        Node::Feature { slot: 1, basis: 0 },                                     // 1
        Node::Affine { terms: vec![(0, 0)], bias: Some(1) },                     // 2 x0
        Node::Affine { terms: vec![(1, 0)], bias: Some(2) },                     // 3 x1
        Node::Affine { terms: vec![(3, 3)], bias: None },                        // 4 q
        Node::Affine { terms: vec![(2, 4)], bias: None },                        // 5 k0
        Node::Affine { terms: vec![(3, 4)], bias: None },                        // 6 k1
        Node::Bilinear { left: 4, right: 5, scale: Scale::InverseSqrt(3) },      // 7
        Node::Bilinear { left: 4, right: 6, scale: Scale::InverseSqrt(3) },      // 8
        Node::Softmax { scores: vec![7, 8] },                                    // 9
        Node::Affine { terms: vec![(2, 5)], bias: None },                        // 10 v0
        Node::Affine { terms: vec![(3, 5)], bias: None },                        // 11 v1
        Node::Mix { weights: 9, payloads: vec![(0, 10), (1, 11)] },              // 12
        Node::Affine { terms: vec![(3, 7), (12, 6)], bias: Some(8) },            // 13 residual
        Node::RmsNorm { input: 13, epsilon: 1e-5 },                              // 14
        Node::Affine { terms: vec![(14, 9)], bias: Some(10) },                   // 15 gate pre
        Node::Pointwise { input: 15, laws: vec![Law::Relu, Law::Silu, Law::Gelu, Law::GeluTanh, Law::Identity] }, // 16
        Node::Affine { terms: vec![(14, 11)], bias: None },                      // 17 up
        Node::Hadamard { left: 16, right: 17 },                                  // 18
        Node::Affine { terms: vec![(13, 7), (18, 12)], bias: None },             // 19 residual
        Node::Gain { input: 19, coefficient: Coefficient::Number(0.75) },        // 20
        Node::Affine { terms: vec![(20, 13)], bias: Some(14) },                  // 21 logits
        Node::Readout { input: 21, basis: 1 },                                   // 22
        Node::Transposed { input: 20, operator: 0 },                             // 23 tied read
        Node::Concat { parts: vec![22, 23] },                                    // 24
    ];
    let program = OperatorProgram {
        rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.into_iter().map(Arc::new).collect(),
        nodes,
        output: 24,
    };
    let pairs: Vec<(u32, u32)> = (0..7).flat_map(|a| (0..7).map(move |b| (a, b))).collect();
    let inputs = FamilyInputs {
        rows: pairs.len(),
        slots: vec![
            SlotValues::Tokens(pairs.iter().map(|p| p.0).collect()),
            SlotValues::Tokens(pairs.iter().map(|p| p.1).collect()),
        ],
        layout: None,
    };
    (program, inputs)
}

/// Three sequences of four tokens: embed, RMS norm, a gain diagonal, causal rotary attention, the
/// residual, a tied transposed read and a readout.
fn attending() -> (OperatorProgram, FamilyInputs) {
    let declarations = Declarations {
        parameters: 0,
        domains: vec![Domain { size: 6, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }],
    };
    let tokens = Interface::uniform(6, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(4).expect("interface");
    let head = Interface::native(4).expect("interface");
    let operators = vec![
        dense("E", &model, &tokens, 21),        // 0
        dense("G", &model, &model, 22),         // 1
        dense("Q", &head, &model, 23),          // 2
        dense("K", &head, &model, 24),          // 3
        dense("V", &head, &model, 25),          // 4
        dense("O", &model, &head, 26),          // 5
        Operator::identity("I", model.clone()), // 6
    ];
    let rotary = Rotary { base: 10000, dims: 4, half_split: true };
    let nodes = vec![
        Node::Feature { slot: 0, basis: 0 },                   // 0
        Node::Affine { terms: vec![(0, 0)], bias: None },      // 1 x
        Node::RmsNorm { input: 1, epsilon: 1e-6 },             // 2
        Node::Affine { terms: vec![(2, 1)], bias: None },      // 3 h
        Node::Affine { terms: vec![(3, 2)], bias: None },      // 4 q
        Node::Affine { terms: vec![(3, 3)], bias: None },      // 5 k
        Node::Affine { terms: vec![(3, 4)], bias: None },      // 6 v
        Node::Attend { query: 4, key: 5, value: 6, scale: Scale::InverseSqrt(4), rotary: Some(rotary), causal: true }, // 7
        Node::Affine { terms: vec![(1, 6), (7, 5)], bias: None }, // 8 residual
        Node::Transposed { input: 8, operator: 0 },            // 9
        Node::Readout { input: 9, basis: 0 },                  // 10
    ];
    let program = OperatorProgram {
        rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: operators.into_iter().map(Arc::new).collect(),
        nodes,
        output: 10,
    };
    let (sequences, length) = (3u32, 4u32);
    let tokens: Vec<u32> = (0..sequences * length).map(|i| (i * 5 + 1) % 6).collect();
    let inputs = FamilyInputs {
        rows: tokens.len(),
        slots: vec![SlotValues::Tokens(tokens)],
        layout: Some(SequenceLayout {
            sequence: (0..sequences * length).map(|i| i / length).collect(),
            position: (0..sequences * length).map(|i| i % length).collect(),
        }),
    };
    (program, inputs)
}

fn run(program: &OperatorProgram, inputs: &FamilyInputs, target: usize, options: &PathOptions) -> PathDecomposition {
    let trace = program.execute(inputs, false).expect("trace");
    decompose(test_governor(), program, inputs, &trace, target, options).expect("decomposition")
}

fn distance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    frobenius(&(a - b))
}

fn assert_identity(result: &PathDecomposition, context: &str) {
    let gap = distance(&result.items_sum(), &result.value);
    assert!(gap <= result.rounding_band, "{context}: {gap:e} vs band {:e}", result.rounding_band);
    let shares: f64 = result.paths.iter().map(|p| p.share).sum::<f64>() + result.remainders.iter().map(|r| r.share).sum::<f64>();
    assert!((shares - 1.0).abs() <= result.rounding_band / frobenius(&result.value), "{context}: shares {shares}");
}

#[test]
fn paths_sum_to_the_traced_value_at_every_budget() {
    // The attention program has one input and two routes to the output: the residual, and the
    // value read at the executed attention (query and key are conditions, not paths).
    for ((program, inputs), routes) in [(routing(), 29), (attending(), 2)] {
        let complete = run(&program, &inputs, program.output, &PathOptions::expansions(usize::MAX));
        assert!(complete.remainders.is_empty());
        assert_eq!(complete.paths.len(), routes);
        assert!(complete.paths.iter().all(|p| p.nodes.last() == Some(&program.output)));
        assert_identity(&complete, "complete");
        for expansions in [0, 1, 3, 10] {
            assert_identity(&run(&program, &inputs, program.output, &PathOptions::expansions(expansions)), "cut");
        }
    }
}

#[test]
fn a_remainder_is_the_paths_below_it_and_within_its_bound() {
    // Four expansions leave both of the attention program's routes unfinished.
    for ((program, inputs), expansions) in [(routing(), 4), (attending(), 1)] {
        let complete = run(&program, &inputs, program.output, &PathOptions::expansions(usize::MAX));
        let cut = run(&program, &inputs, program.output, &PathOptions::expansions(expansions));
        assert!(!cut.remainders.is_empty());
        for remainder in &cut.remainders {
            let below: Vec<&TracePath> =
                complete.paths.iter().filter(|p| p.nodes.starts_with(&remainder.nodes)).collect();
            let mut net = Array2::<f64>::zeros(remainder.net.dim());
            let mut mass = 0.0;
            for path in &below {
                net += &path.contribution;
                mass += path.mass();
            }
            assert!(distance(&net, &remainder.net) <= complete.rounding_band, "{:?}", remainder.nodes);
            assert!(mass <= remainder.mass_bound, "{:?}: mass {mass} vs bound {}", remainder.nodes, remainder.mass_bound);
        }
    }
}

#[test]
fn the_conditions_and_sources_are_reported_and_either_hadamard_side_is_exact() {
    let (program, inputs) = routing();
    let derived = run(&program, &inputs, program.output, &PathOptions::expansions(usize::MAX));
    let kinds: Vec<ConditionKind> = derived.conditions.iter().map(|c| c.kind).collect();
    for kind in [ConditionKind::MixWeights, ConditionKind::NormScale, ConditionKind::HadamardGate] {
        assert!(kinds.contains(&kind), "{kind:?}");
    }
    // The derived side carries the up read (node 17) and holds the gate (node 16).
    assert!(derived.conditions.contains(&Condition { consumer: 18, argument: 16, kind: ConditionKind::HadamardGate }));
    assert!(derived.sources.iter().any(|&(node, kind)| node == 0 && kind == SourceKind::Input));
    assert!(derived.sources.iter().any(|&(node, kind)| node == 21 && kind == SourceKind::Emitted));
    let mut options = PathOptions::expansions(usize::MAX);
    options.hadamard.insert(18, HadamardSide::Left);
    let declared = run(&program, &inputs, program.output, &options);
    assert!(declared.conditions.contains(&Condition { consumer: 18, argument: 17, kind: ConditionKind::HadamardGate }));
    // Carried through the gate side, the paths cross the gated units at their executed gates.
    assert!(declared.conditions.contains(&Condition { consumer: 16, argument: 15, kind: ConditionKind::PointwiseGate }));
    assert_identity(&declared, "gate side carried");
}

#[test]
fn hop_flows_into_the_target_carry_every_path_once() {
    let (program, inputs) = attending();
    let result = run(&program, &inputs, program.output, &PathOptions::expansions(usize::MAX));
    let flows = result.hop_flows();
    assert!(flows.windows(2).all(|pair| pair[0].mass >= pair[1].mass));
    let mut into = Array2::<f64>::zeros(result.value.dim());
    for flow in flows.iter().filter(|f| f.hop.reader == result.target) {
        into += &flow.net;
    }
    let listed = result.paths.iter().fold(Array2::<f64>::zeros(result.value.dim()), |acc, p| acc + &p.contribution);
    assert!(distance(&into, &listed) <= result.rounding_band);
    let hops: usize = result.paths.iter().map(|p| p.nodes.len() - 1).sum();
    assert_eq!(flows.iter().map(|f| f.paths).sum::<usize>(), hops);
    assert!(result.mass_count(0.99) <= result.paths.len());
}

#[test]
fn a_query_splits_like_any_target() {
    let (program, inputs) = attending();
    let output = run(&program, &inputs, program.output, &PathOptions::expansions(usize::MAX));
    assert!(output.conditions.contains(&Condition { consumer: 7, argument: 4, kind: ConditionKind::AttentionQuery }));
    assert!(output.conditions.contains(&Condition { consumer: 7, argument: 5, kind: ConditionKind::AttentionKey }));
    let query = run(&program, &inputs, 4, &PathOptions::expansions(usize::MAX));
    assert_identity(&query, "query");
    assert!(query.paths.iter().all(|p| p.nodes.first() == Some(&0)));
}

#[test]
fn a_target_outside_the_trace_is_refused() {
    let (program, inputs) = attending();
    let trace = program.execute(&inputs, false).expect("trace");
    assert!(matches!(
        decompose(test_governor(), &program, &inputs, &trace, 99, &PathOptions::expansions(1)),
        Err(PathError::Target { node: 99, .. })
    ));
}

/// The Sylvester Hadamard matrix of order `2^power`: entries `±1`, orthogonal
/// columns of squared norm `2^power`.
fn hadamard(power: u32) -> Array2<f64> {
    let order = 1_usize << power;
    Array2::from_shape_fn((order, order), |(row, col)| {
        if (row & col).count_ones() % 2 == 0 { 1.0 } else { -1.0 }
    })
}

/// `H_m[:, :k] diag(s) H_n[:, :k]ᵀ`, every entry an exact dyadic sum, so its
/// exact singular values are `|sᵢ|·√(mn)` and the matrix as stored is exact.
fn exact_spectrum(rows_power: u32, cols_power: u32, weights: &[f64]) -> Array2<f64> {
    let left = hadamard(rows_power);
    let right = hadamard(cols_power);
    let mut matrix = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    for (index, &weight) in weights.iter().enumerate() {
        for row in 0..left.nrows() {
            for col in 0..right.nrows() {
                matrix[[row, col]] += weight * left[[row, index]] * right[[col, index]];
            }
        }
    }
    matrix
}

/// The Gram-spectrum bounds bracket the exact `σ_max` of exactly representable
/// matrices with a known spectrum: full rank, rank deficient, wide and tall,
/// with a clustered top, and at both ends of the exponent range; the bracket is
/// as tight as its derivation says, and `formation` widens it by exactly itself.
#[test]
fn gram_spectrum_bounds_bracket_the_exact_largest_singular_value() {
    let cluster = 1.0 - (-40.0_f64).exp2();
    let cases: Vec<(&str, u32, u32, Vec<f64>)> = vec![
        ("full rank", 6, 6, (0..64).map(|index| 1.0 - index as f64 / 128.0).collect()),
        ("clustered top", 6, 6, vec![1.0, cluster, cluster, cluster, 0.5, 0.25]),
        ("rank deficient tall", 7, 5, vec![0.75, 0.5, 0.0, 0.125]),
        ("rank deficient wide", 5, 7, vec![0.75, 0.5, 0.0, 0.125]),
        ("rank one", 6, 4, vec![0.5]),
        ("negative weights", 5, 5, vec![-1.0, 0.875, -0.875]),
    ];
    for (name, rows_power, cols_power, weights) in cases {
        let matrix = exact_spectrum(rows_power, cols_power, &weights);
        let scale = ((matrix.nrows() * matrix.ncols()) as f64).sqrt();
        let exact = weights.iter().fold(0.0_f64, |largest, weight| largest.max(weight.abs())) * scale;
        for exponent in [-1000_i32, 0, 1000] {
            let power = f64::from(exponent).exp2();
            let scaled = matrix.mapv(|value| value * power);
            let truth = exact * power;
            let bounds = spectral_norm_bounds(test_governor(), &scaled, 0.0, "test").expect("bounds");
            assert!(
                bounds.lower <= truth && truth <= bounds.upper,
                "{name} at 2^{exponent}: {bounds:?} misses {truth:e}"
            );
            let long = scaled.nrows().max(scaled.ncols()) as f64;
            let short = scaled.nrows().min(scaled.ncols()) as f64;
            let width = (bounds.upper - bounds.lower) / truth;
            assert!(
                width <= 4.0 * (long + 1.0) * short * UNIT_ROUNDOFF,
                "{name} at 2^{exponent}: relative width {width:e}"
            );
        }
        let formation = 0.5 * exact;
        let widened = spectral_norm_bounds(test_governor(), &matrix, formation, "test").expect("bounds");
        let tight = spectral_norm_bounds(test_governor(), &matrix, 0.0, "test").expect("bounds");
        assert_eq!(widened.upper, tight.upper + formation, "{name}");
        assert_eq!(widened.lower, (tight.lower - formation).max(0.0), "{name}");
    }
    let zero = Array2::<f64>::zeros((3, 5));
    let bounds = spectral_norm_bounds(test_governor(), &zero, 0.25, "test").expect("bounds");
    assert_eq!((bounds.lower, bounds.upper), (0.0, 0.25));
    let mut infinite = Array2::<f64>::eye(3);
    infinite[[1, 2]] = f64::INFINITY;
    assert!(matches!(
        spectral_norm_bounds(test_governor(), &infinite, 0.0, "test"),
        Err(PathError::NonFiniteMatrix { .. })
    ));
}

/// On a seeded dense matrix the bracket agrees with the full SVD it replaces:
/// each interval contains the other's centre, both being certified.
#[test]
fn gram_spectrum_bounds_agree_with_the_singular_value_decomposition() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = StdRng::seed_from_u64(2951);
    for (rows, cols) in [(40, 40), (17, 90), (90, 17), (1, 30)] {
        let matrix = Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0));
        let (_, sigma, _) = matrix.svd(false, false).expect("svd");
        let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
        let band = factor_singular_band(rows, cols, sigma_max);
        let bounds = spectral_norm_bounds(test_governor(), &matrix, 0.0, "test").expect("bounds");
        assert!(
            bounds.lower <= sigma_max + band && sigma_max - band <= bounds.upper,
            "{rows}x{cols}: {bounds:?} against σ̂ {sigma_max:e} ± {band:e}"
        );
    }
}
