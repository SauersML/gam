#![cfg(test)]
//! Rule discovery on planted programs: one ReLU subroutine written into two residual layers under
//! different random orthogonal bases is recovered as one rule whose call is an orthogonal binding
//! (the body's own instance is the identity binding), and the library decodes back to the exact
//! operators; a random null yields no rule; duplicating or power-of-two rescaling the hidden units
//! leaves the library unchanged, and duplicating an operator of the null creates none; under an
//! invertible, non-orthogonal change of basis the subroutine is one rule with a linear binding. On the
//! ResidMLP toy, the ideal solution's layers are one rule called by every other layer through an
//! orthogonal binding, and random layers give none.

use std::sync::Arc;
use crate::antiunify::{BindingFamily, discover_rules};
use crate::egraph::normalize;
use crate::operator_program::{
    Declarations, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Slot, exact_precision,
    round_to_lattice,
};
use crate::precision::DeclaredPrecision;
use crate::test_support::planted_toys::dyadic;
use crate::test_support::{hidden_basis, test_governor};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

const WIDTH: usize = 6;
const HIDDEN: usize = 12;
/// The lattice the layers' reals are declared on.
const FRACTION_BITS: i32 = 20;

/// One layer's operators: read `A`, bias `b`, write `B`.
struct Layer {
    read: Array2<f64>,
    bias: Array2<f64>,
    write: Array2<f64>,
}

/// The planted subroutine: integer reads and writes over halves, distinct biases (so the
/// canonical unit order is determined).
fn subroutine() -> Layer {
    let mut rng = StdRng::seed_from_u64(0x2951_a1);
    let bias = Array2::from_shape_fn((HIDDEN, 1), |(i, _)| (i as f64 - 5.5) / 8.0);
    Layer { read: dyadic(&mut rng, HIDDEN, WIDTH, 6, 2.0), bias, write: dyadic(&mut rng, WIDTH, HIDDEN, 6, 2.0) }
}

fn on_lattice(m: &Array2<f64>) -> Array2<f64> {
    let lattice = DeclaredPrecision::new(FRACTION_BITS).expect("a lattice");
    m.mapv(|v| round_to_lattice(v, lattice).expect("on the lattice"))
}

/// The subroutine in the residual basis `Q`, on the declared lattice: `(A Qᵀ, b, Q B)`.
fn in_basis(layer: &Layer, q: &Array2<f64>) -> Layer {
    Layer { read: on_lattice(&layer.read.dot(&q.t())), bias: layer.bias.clone(), write: on_lattice(&q.dot(&layer.write)) }
}

/// `x → x + B relu(A x + b)` twice.
fn residual_program(layers: &[Layer]) -> OperatorProgram {
    let x = Interface::native(WIDTH).expect("residual interface");
    let mut operators = vec![Operator::identity("I", x.clone())];
    let mut nodes = vec![Node::Raw { slot: 0 }];
    let mut stream = 0;
    for (index, layer) in layers.iter().enumerate() {
        let hidden = layer.read.nrows();
        let units = Interface::uniform(hidden, 1, LabelKind::Unit, 0).expect("unit interface");
        let dense = |name: String, rows: &Interface, cols: &Interface, values: &Array2<f64>| {
            let lattice = exact_precision(values.iter().copied()).expect("an exact lattice");
            Operator::dense(name.clone(), rows.clone(), cols.clone(), values.clone(), lattice, Provenance::native(&name)).expect("dense")
        };
        operators.push(dense(format!("read{index}"), &units, &x, &layer.read));
        operators.push(dense(format!("bias{index}"), &units, &Interface::constant(), &layer.bias));
        operators.push(dense(format!("write{index}"), &x, &units, &layer.write));
        let base = operators.len() - 3;
        nodes.push(Node::Affine { terms: vec![(stream, base)], bias: Some(base + 1) });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Relu; hidden] });
        nodes.push(Node::Affine { terms: vec![(stream, 0), (nodes.len() - 1, base + 2)], bias: None });
        stream = nodes.len() - 1;
    }
    OperatorProgram {
        declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width: WIDTH }], parameters: 0 },
        bases: Vec::new(),
        rules: Vec::new(),
        operators: operators.into_iter().map(Arc::new).collect(),
        nodes,
        output: stream,
    }
}

fn bases() -> (Array2<f64>, Array2<f64>) {
    (hidden_basis(WIDTH, 0x2951_b1), hidden_basis(WIDTH, 0x2951_b2))
}

fn planted() -> OperatorProgram {
    let (q1, q2) = bases();
    let body = subroutine();
    residual_program(&[in_basis(&body, &q1), in_basis(&body, &q2)])
}

#[test]
fn one_subroutine_in_two_orthogonal_bases_is_one_rule() {
    let program = planted();
    let normalization = normalize(&program, test_governor()).expect("normalizes");
    assert!(normalization.saturation.report.saturated());
    let library = discover_rules(&normalization).expect("discovers");
    assert_eq!(library.rules.len(), 1, "{:?}", library.rules.iter().map(|r| &r.skeleton).collect::<Vec<_>>());
    let rule = &library.rules[0];
    assert_eq!(rule.holes.len(), 2, "the read and the write of the subroutine");
    assert_eq!(rule.calls.len(), 1, "the body is one layer, the call the other");
    let binding = &rule.calls[0].binding;
    assert_eq!(binding.family, BindingFamily::Orthogonal);
    assert_eq!(binding.maps.len(), 1, "the residual stream is one tied space");
    let map = &binding.maps[0].2;
    let gram = map.t().dot(map) - Array2::<f64>::eye(WIDTH);
    let defect = gram.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(defect <= 64.0 * f64::EPSILON * WIDTH as f64, "a Cayley map is orthogonal to rounding: {defect:e}");
    assert!(map.diag().iter().any(|v| (v.abs() - 1.0).abs() > 0.5), "the binding is a genuine rotation");
    let (q1, q2) = bases();
    let relative = [q2.dot(&q1.t()), q1.dot(&q2.t())];
    let distance = relative
        .iter()
        .map(|target| (map - target).iter().fold(0.0_f64, |m, v| m.max(v.abs())))
        .fold(f64::INFINITY, f64::min);
    assert!(distance < 2.0_f64.powi(-FRACTION_BITS / 2), "the map is the planted relative basis: {distance:e}");
    assert!(library.bits_after < library.bits_before, "{} against {}", library.bits_after, library.bits_before);
    for (index, matrix) in library.decode_instances().expect("the section decodes") {
        assert_eq!(matrix, library.program.operators[index].matrix(), "operator {index} decodes bit-exactly");
    }
}

#[test]
fn a_random_null_yields_no_rules() {
    let mut rng = StdRng::seed_from_u64(crate::test_support::planted_toys::RANDOM_NULL_SEED);
    let layer = |rng: &mut StdRng, first: f64| Layer {
        read: dyadic(rng, HIDDEN, WIDTH, 64, 32.0),
        bias: Array2::from_shape_fn((HIDDEN, 1), |(i, _)| first + i as f64 / 8.0),
        write: dyadic(rng, WIDTH, HIDDEN, 64, 32.0),
    };
    let null = residual_program(&[layer(&mut rng, -0.75), layer(&mut rng, -0.6875)]);
    let normalization = normalize(&null, test_governor()).expect("normalizes");
    let library = discover_rules(&normalization).expect("discovers");
    assert!(library.rules.is_empty(), "{:?}", library.rules.iter().map(|r| &r.skeleton).collect::<Vec<_>>());
    assert_eq!(library.bits_after, library.bits_before);
    // Duplicating a component of the null (a third layer equal to the second) creates no rule:
    // the copy's operators are the second layer's own leaves, already shared.
    let mut rng = StdRng::seed_from_u64(crate::test_support::planted_toys::RANDOM_NULL_SEED);
    let first = layer(&mut rng, -0.75);
    let second = layer(&mut rng, -0.6875);
    let copy = Layer { read: second.read.clone(), bias: second.bias.clone(), write: second.write.clone() };
    let duplicated = residual_program(&[first, second, copy]);
    let library = discover_rules(&normalize(&duplicated, test_governor()).expect("normalizes")).expect("discovers");
    assert!(library.rules.is_empty(), "{:?}", library.rules.iter().map(|r| &r.skeleton).collect::<Vec<_>>());
}

/// The planted program with its second layer's hidden units rescaled by powers of two, or with a
/// unit duplicated (read row copied, write column halved), has the same library.
#[test]
fn duplicating_or_rescaling_units_creates_no_fake_rules() {
    let reference = discover_rules(&normalize(&planted(), test_governor()).expect("normalizes")).expect("discovers");
    assert_eq!(reference.rules.len(), 1);
    let (q1, q2) = bases();
    let body = subroutine();
    let second = in_basis(&body, &q2);
    let (read, bias, write) = (second.read.clone(), second.bias.clone(), second.write.clone());
    let mut rescaled = Layer { read: read.clone(), bias: bias.clone(), write: write.clone() };
    for unit in 0..HIDDEN {
        let scale = 2.0_f64.powi(unit as i32 % 5 - 2);
        rescaled.read.row_mut(unit).mapv_inplace(|v| v * scale);
        rescaled.bias[[unit, 0]] *= scale;
        rescaled.write.column_mut(unit).mapv_inplace(|v| v / scale);
    }
    let mut duplicated = Layer {
        read: Array2::zeros((HIDDEN + 1, WIDTH)),
        bias: Array2::zeros((HIDDEN + 1, 1)),
        write: Array2::zeros((WIDTH, HIDDEN + 1)),
    };
    duplicated.read.slice_mut(ndarray::s![..HIDDEN, ..]).assign(&read);
    duplicated.read.row_mut(HIDDEN).assign(&read.row(4));
    duplicated.bias.slice_mut(ndarray::s![..HIDDEN, ..]).assign(&bias);
    duplicated.bias[[HIDDEN, 0]] = bias[[4, 0]];
    duplicated.write.slice_mut(ndarray::s![.., ..HIDDEN]).assign(&write);
    let halved = write.column(4).mapv(|v| v / 2.0);
    duplicated.write.column_mut(4).assign(&halved);
    duplicated.write.column_mut(HIDDEN).assign(&halved);
    for variant in [rescaled, duplicated] {
        let program = residual_program(&[in_basis(&body, &q1), variant]);
        let library = discover_rules(&normalize(&program, test_governor()).expect("normalizes")).expect("discovers");
        assert_eq!(library.rules.len(), 1);
        assert_eq!(library.rules[0].calls.len(), 1);
        assert_eq!(library.bits_after, reference.bits_after, "the canonical unit gauge removes the variant");
    }
}

/// The planted subroutine written into the second layer under an invertible, non-orthogonal change
/// of the residual basis `M` (unimodular, so every real stays exact): `(A M⁻¹, b, M B)`. Its
/// spectra differ from the first layer's, so only the linear family can bind it, and the binding
/// is `M` itself.
#[test]
fn one_subroutine_under_a_linear_change_of_basis_is_one_linear_rule() {
    let (m, m_inverse) = crate::test_support::planted_toys::unimodular_pair(WIDTH, 0x2951_d1);
    let body = subroutine();
    let moved = Layer { read: body.read.dot(&m_inverse), bias: body.bias.clone(), write: m.dot(&body.write) };
    let program = residual_program(&[subroutine(), moved]);
    let normalization = normalize(&program, test_governor()).expect("normalizes");
    let library = discover_rules(&normalization).expect("discovers");
    assert_eq!(library.rules.len(), 1, "{:?}", library.rules.iter().map(|r| &r.skeleton).collect::<Vec<_>>());
    let rule = &library.rules[0];
    assert_eq!(rule.holes.len(), 2, "the read and the write of the subroutine");
    assert_eq!(rule.calls.len(), 1);
    let binding = &rule.calls[0].binding;
    assert_eq!(binding.family, BindingFamily::Linear);
    assert_eq!(binding.maps.len(), 1, "the residual stream is one tied space");
    let map = &binding.maps[0].2;
    assert!(*map == m || *map == m_inverse, "the binding is the planted change of basis: {map:?}");
    let exact: u64 = rule.calls[0]
        .leaves
        .iter()
        .map(|leaf| {
            let lattice = normalization.saturation.egraph.analysis.leaves[*leaf as usize].lattice.expect("an exact call operator");
            crate::codec::signed_prefix_integer_len_bits(i64::from(lattice)).expect("a lattice code") + 1
        })
        .sum();
    assert_eq!(binding.residual_bits, exact, "an exact prediction: per hole only its lattice and a zero flag");
    assert!(library.bits_after < library.bits_before, "{} against {}", library.bits_after, library.bits_before);
    for (index, matrix) in library.decode_instances().expect("the section decodes") {
        assert_eq!(matrix, library.program.operators[index].matrix(), "operator {index} decodes bit-exactly");
    }
}

// ------------------------------------------------------------------------------ ResidMLP toys

/// Residual width of the ResidMLP toys.
const EMBED: usize = 8;
/// Features, and hidden units, per layer.
const PER_LAYER: usize = 2;

/// A ResidMLP (Braun et al. 2025, APD's toy): `r_0 = W_E x`, `r ← r + W_out relu(W_in r + b)` per
/// layer, `y = W_Eᵀ r`, on raw features `x`.
fn resid_mlp(embed: &Array2<f64>, layers: &[Layer]) -> OperatorProgram {
    let features = embed.ncols();
    let x = Interface::native(features).expect("feature interface");
    let r = Interface::native(EMBED).expect("residual interface");
    let dense = |name: String, rows: &Interface, cols: &Interface, values: &Array2<f64>| {
        let lattice = exact_precision(values.iter().copied()).expect("an exact lattice");
        Operator::dense(name.clone(), rows.clone(), cols.clone(), values.clone(), lattice, Provenance::native(&name)).expect("dense")
    };
    let mut operators = vec![Operator::identity("I", r.clone()), dense("W_E".into(), &r, &x, embed), dense("W_U".into(), &x, &r, &embed.t().to_owned())];
    let mut nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 1)], bias: None }];
    let mut stream = 1;
    for (index, layer) in layers.iter().enumerate() {
        let units = Interface::uniform(layer.read.nrows(), 1, LabelKind::Unit, 0).expect("unit interface");
        operators.push(dense(format!("W_in{index}"), &units, &r, &layer.read));
        operators.push(dense(format!("b_in{index}"), &units, &Interface::constant(), &layer.bias));
        operators.push(dense(format!("W_out{index}"), &r, &units, &layer.write));
        let base = operators.len() - 3;
        nodes.push(Node::Affine { terms: vec![(stream, base)], bias: Some(base + 1) });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Relu; layer.read.nrows()] });
        nodes.push(Node::Affine { terms: vec![(stream, 0), (nodes.len() - 1, base + 2)], bias: None });
        stream = nodes.len() - 1;
    }
    nodes.push(Node::Affine { terms: vec![(stream, 2)], bias: None });
    OperatorProgram {
        declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width: features }], parameters: 0 },
        bases: Vec::new(),
        rules: Vec::new(),
        operators: operators.into_iter().map(Arc::new).collect(),
        output: nodes.len() - 1,
        nodes,
    }
}

/// An exactly orthonormal dyadic basis of `R^EMBED`: two 4 × 4 Hadamard blocks scaled by `1/2`.
fn hadamard_basis() -> Array2<f64> {
    let h = [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]];
    Array2::from_shape_fn((EMBED, EMBED), |(i, j)| if i / 4 == j / 4 { 0.5 * h[i % 4][j % 4] } else { 0.0 })
}

/// The ideal ResidMLP of `layers` layers: orthonormal feature embeddings (the first columns of a
/// random orthogonal `Q`, on the lattice), and in layer `ℓ` one unit per feature `f` of its block,
/// reading `c_f e_fᵀ` and writing `e_f / c_f` with distinct gauges `c_f` (so the canonical unit order
/// is determined), zero bias. It computes `x + relu(x)` to the lattice. Layer `ℓ` is layer 0 under
/// the orthogonal change of residual basis `Q P Qᵀ` that moves block 0's features onto block `ℓ`'s.
fn ideal_resid_mlp(layers: usize) -> OperatorProgram {
    let q = hadamard_basis();
    let features = layers * PER_LAYER;
    let embed = on_lattice(&q.slice(ndarray::s![.., ..features]).to_owned());
    let gauge = [1.125, 1.5];
    let blocks: Vec<Layer> = (0..layers)
        .map(|layer| {
            let mut read = Array2::zeros((PER_LAYER, EMBED));
            let mut write = Array2::zeros((EMBED, PER_LAYER));
            for unit in 0..PER_LAYER {
                let e = q.column(layer * PER_LAYER + unit);
                read.row_mut(unit).assign(&e.mapv(|v| v * gauge[unit]));
                write.column_mut(unit).assign(&e.mapv(|v| v / gauge[unit]));
            }
            Layer { read: on_lattice(&read), bias: Array2::zeros((PER_LAYER, 1)), write: on_lattice(&write) }
        })
        .collect();
    resid_mlp(&embed, &blocks)
}

#[test]
fn an_ideal_resid_mlp_is_one_rule_called_by_every_other_layer() {
    for layers in [2, 3] {
        let program = ideal_resid_mlp(layers);
        let features = layers * PER_LAYER;
        let mut rng = StdRng::seed_from_u64(0x2951_c2);
        let inputs = crate::operator_program::FamilyInputs {
            rows: 16,
            slots: vec![crate::operator_program::SlotValues::Raw(dyadic(&mut rng, 16, features, 8, 4.0))],
            layout: None,
        };
        let trace = program.execute(&inputs, false).expect("executes");
        let crate::operator_program::SlotValues::Raw(x) = &inputs.slots[0] else { unreachable!() };
        let target = x + &x.mapv(|v| v.max(0.0));
        let error = (&trace.values[program.output] - &target).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(error < 2.0_f64.powi(-FRACTION_BITS / 2), "the planted ResidMLP computes x + relu(x): {error:e}");

        let normalization = normalize(&program, test_governor()).expect("normalizes");
        assert!(normalization.saturation.report.saturated());
        let library = discover_rules(&normalization).expect("discovers");
        assert_eq!(library.rules.len(), 1, "{layers} layers: {:?}", library.rules.iter().map(|r| &r.skeleton).collect::<Vec<_>>());
        let rule = &library.rules[0];
        assert_eq!(rule.holes.len(), 2, "W_in and W_out of a layer");
        assert_eq!(rule.calls.len(), layers - 1, "every other layer calls the body");
        assert!(rule.calls.iter().all(|call| call.binding.family == BindingFamily::Orthogonal));
        assert!(library.bits_after < library.bits_before, "{} against {}", library.bits_after, library.bits_before);
        for (index, matrix) in library.decode_instances().expect("the section decodes") {
            assert_eq!(matrix, library.program.operators[index].matrix(), "operator {index} decodes bit-exactly");
        }
    }
}

#[test]
fn a_random_resid_mlp_has_no_rules() {
    let mut rng = StdRng::seed_from_u64(crate::test_support::planted_toys::RANDOM_NULL_SEED);
    let embed = dyadic(&mut rng, EMBED, 2 * PER_LAYER, 64, 32.0);
    let layers: Vec<Layer> = (0..2)
        .map(|layer| Layer {
            read: dyadic(&mut rng, PER_LAYER, EMBED, 64, 32.0),
            bias: Array2::from_shape_fn((PER_LAYER, 1), |(i, _)| (layer * PER_LAYER + i) as f64 / 8.0 - 0.5),
            write: dyadic(&mut rng, EMBED, PER_LAYER, 64, 32.0),
        })
        .collect();
    let library = discover_rules(&normalize(&resid_mlp(&embed, &layers), test_governor()).expect("normalizes")).expect("discovers");
    assert!(library.rules.is_empty(), "{:?}", library.rules.iter().map(|r| &r.skeleton).collect::<Vec<_>>());
    assert_eq!(library.bits_after, library.bits_before);
}
