#![cfg(test)]
//! Rank-k gated blocks: their gates' derivatives, their maps' inner products, and the code's
//! merges and splits.

use super::blocks::{Blocked, Coded, Generic, balanced, block_cosine, block_inner, fit_blocks, measure, split_block};
use super::masked::{Library, Masked, Target, forward, mask_gradients, site_statistics, sites};
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Slot,
    SlotValues,
};
use super::precision::DeclaredPrecision;
use ndarray::{Array1, Array2, s};
use std::sync::Arc;

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

const P: usize = 5;
const WIDTH: usize = 4;
const UNITS: usize = 6;

/// Two tokens embedded, a ReLU layer, logits over five classes.
fn model() -> (OperatorProgram, FamilyInputs) {
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(WIDTH).expect("interface");
    let units = Interface::uniform(UNITS, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let op = |name: &str, rows: &Interface, cols: &Interface, salt: usize| {
        let m = Array2::from_shape_fn((rows.width(), cols.width()), |(i, j)| 2.0 * noise(salt + 31 * i + j));
        let precision = DeclaredPrecision::new(40).expect("a precision");
        Arc::new(Operator::dense(name, rows.clone(), cols.clone(), m, precision, Provenance::default()).expect("dense"))
    };
    let program = OperatorProgram {
        declarations: Declarations {
            parameters: 0,
            domains: vec![Domain { size: P }],
            slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
        },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![op("E", &residual, &tokens, 1), op("W_in", &units, &residual, 100), op("W_out", &classes, &units, 200)],
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },
            Node::Feature { slot: 1, basis: 0 },
            Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
            Node::Affine { terms: vec![(2, 1)], bias: None },
            Node::Pointwise { input: 3, laws: vec![Law::Relu; UNITS] },
            Node::Affine { terms: vec![(4, 2)], bias: None },
            Node::Readout { input: 5, basis: 0 },
        ],
        output: 6,
    };
    let pairs: Vec<(u32, u32)> = (0..P as u32).flat_map(|a| (0..P as u32).map(move |b| (a, b))).collect();
    let family = FamilyInputs {
        layout: None,
        rows: pairs.len(),
        slots: vec![SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()), SlotValues::Tokens(pairs.iter().map(|q| q.1).collect())],
    };
    (program, family)
}

/// The W_in site's map as `WIDTH` rank-one pieces (its rows of `Wᵀ` against unit outputs, so all
/// on is the map) on the uncentred read.
fn exact_library(program: &OperatorProgram) -> Library {
    let w = program.operators[1].matrix();
    Library { v: Array2::eye(WIDTH), u: w.t().to_owned(), mean: Array1::zeros(WIDTH) }
}

#[test]
fn a_blocks_gate_derivative_is_the_sum_of_its_columns() {
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let library = exact_library(&program);
    let masked = Masked::build_blocks(&program, vec![site], vec![library], vec![vec![2, 1, 1]]).expect("builds");
    assert_eq!(masked.blocks(0), 3);
    let masks = vec![Array2::from_shape_fn((family.rows, 3), |(r, c)| 0.3 + 0.2 * ((r + c) % 3) as f64)];
    let fam = masked.family(&family, &masks);
    let (kl, trace, cotangent) = forward(&masked, &fam, &target).expect("forward");
    let g = mask_gradients(&masked, &fam, &trace, cotangent).expect("gradients");
    let h = 1e-6;
    for (r, b) in [(0, 0), (7, 0), (11, 1), (19, 2)] {
        let mut shifted = masks.clone();
        shifted[0][[r, b]] += h;
        let (kl_h, _, _) = forward(&masked, &masked.family(&family, &shifted), &target).expect("forward");
        let numeric = (kl_h.sum() - kl.sum()) / h;
        assert!((numeric - g[0][[r, b]]).abs() <= 1e-4 * (1.0 + numeric.abs()), "gate ({r}, {b}): {numeric} against {}", g[0][[r, b]]);
    }
}

#[test]
fn rank_one_blocks_are_the_pieces_program() {
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let library = exact_library(&program);
    let pieces = Masked::build(&program, vec![site.clone()], vec![library.clone()]).expect("builds");
    let blocks = Masked::build_blocks(&program, vec![site], vec![library], vec![vec![1; WIDTH]]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, WIDTH), |(r, c)| f64::from((r + c) % 2 == 0))];
    let (a, _, _) = forward(&pieces, &pieces.family(&family, &masks), &target).expect("forward");
    let (b, _, _) = forward(&blocks, &blocks.family(&family, &masks), &target).expect("forward");
    assert_eq!(a, b);
}

#[test]
fn block_inner_products_are_the_maps_frobenius_products() {
    let library = Library {
        v: Array2::from_shape_fn((5, 7), |(i, j)| noise(10 + 7 * i + j)),
        u: Array2::from_shape_fn((5, 3), |(i, j)| noise(90 + 3 * i + j)),
        mean: Array1::zeros(7),
    };
    let map = |start: usize, k: usize| library.u.slice(s![start..start + k, ..]).t().dot(&library.v.slice(s![start..start + k, ..]));
    let (a, b) = (map(0, 2), map(2, 3));
    let explicit = (&a * &b).sum();
    assert!((block_inner(&library, (0, 2), (2, 3)) - explicit).abs() <= 1e-12 * (1.0 + explicit.abs()));
    assert!((block_cosine(&library, (0, 2), (0, 2)) - 1.0).abs() <= 1e-12);
}

/// The W_in site's generic description at `observations`.
fn generic(program: &OperatorProgram, family: &FamilyInputs, observations: f64) -> Generic {
    let site = sites(program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    Generic::new(&site_statistics(program, &[site], [family.clone()], 8, 7).expect("statistics"), observations)
}

fn coded<'a>(program: &'a OperatorProgram, family: &FamilyInputs, observations: f64, describe: &'a Generic) -> Coded<'a> {
    let target = Target::every_row(program.execute(family, false).expect("executes").values[program.output].clone());
    let site = sites(program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    Coded { model: program, sites: vec![site], batches: vec![(family.clone(), target)], observations, samples: 8, describe }
}

#[test]
fn balanced_factors_keep_the_map_with_equal_grams() {
    let u = Array2::from_shape_fn((3, 5), |(i, j)| noise(300 + 5 * i + j));
    let v = Array2::from_shape_fn((3, 4), |(i, j)| 3.0 * noise(400 + 4 * i + j));
    let (bu, bv) = balanced(u.view(), v.view()).expect("balances");
    let error = (&u.t().dot(&v) - &bu.t().dot(&bv)).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(error <= 1e-10, "{error}");
    let (gu, gv) = (bu.dot(&bu.t()), bv.dot(&bv.t()));
    let off = (&gu - &gv).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(off <= 1e-10 * (1.0 + gu[[0, 0]]), "{off}");
    assert!(gu[[0, 1]].abs() <= 1e-10 * gu[[0, 0]] && gu[[0, 0]] >= gu[[1, 1]] && gu[[1, 1]] >= gu[[2, 2]]);
    // A duplicated column collapses.
    let doubled_u = ndarray::concatenate(ndarray::Axis(0), &[u.view(), u.slice(s![..1, ..])]).expect("stack");
    let doubled_v = ndarray::concatenate(ndarray::Axis(0), &[v.view(), v.slice(s![..1, ..])]).expect("stack");
    assert_eq!(balanced(doubled_u.view(), doubled_v.view()).expect("balances").0.nrows(), 3);
}

#[test]
fn a_direction_small_on_one_side_and_large_on_the_other_is_kept() {
    // U = diag(1, 2^-40), V = diag(1, 2^40): U Vᵀ = I, rank two.
    let tiny = 2f64.powi(-40);
    let u = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, tiny]).expect("u");
    let v = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0 / tiny]).expect("v");
    let (bu, bv) = balanced(u.view(), v.view()).expect("balances");
    assert_eq!(bu.nrows(), 2);
    let error = (&bu.t().dot(&bv) - &Array2::<f64>::eye(2)).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(error <= 1e-12, "{error}");
    let library = Library { v, u, mean: Array1::zeros(2) };
    let blocked = Blocked::new(vec![library], vec![vec![2]], vec![vec![Array2::ones((3, 1))]]);
    let split = split_block(&blocked, 0, 0, &Array2::eye(2)).expect("splits");
    assert_eq!(split.ranks[0], vec![1, 1]);
    let map = split.libraries[0].u.t().dot(&split.libraries[0].v);
    let error = (&map - &Array2::<f64>::eye(2)).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(error <= 1e-12, "the split moved the map by {error}");
}

#[test]
fn a_split_keeps_the_blocks_map() {
    let (program, family) = model();
    let describe = generic(&program, &family, 100.0);
    let coded = coded(&program, &family, 100.0, &describe);
    // One rank-WIDTH block, on everywhere.
    let whole = Blocked::new(vec![exact_library(&program)], vec![vec![WIDTH]], vec![vec![Array2::ones((family.rows, 1))]]);
    let moment = Array2::from_shape_fn((WIDTH, WIDTH), |(i, j)| if i == j { 1.0 + i as f64 } else { 0.1 });
    let split = split_block(&whole, 0, 0, &moment).expect("splits");
    assert_eq!(split.ranks[0], vec![1; WIDTH]);
    let map = |b: &Blocked| b.libraries[0].u.t().dot(&b.libraries[0].v);
    let (before, after) = (map(&whole), map(&split));
    let error = (&before - &after).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(error <= 1e-10, "the split moved the map by {error}");
    let (a, _) = measure(&coded, &whole).expect("measures");
    let (b, _) = measure(&coded, &split).expect("measures");
    assert!((a.kl_nats - b.kl_nats).abs() <= 1e-9, "all on: {} against {}", a.kl_nats, b.kl_nats);
}

#[test]
fn always_cofiring_pieces_merge_by_the_code() {
    let (program, family) = model();
    let describe = generic(&program, &family, 1000.0);
    let coded = coded(&program, &family, 1000.0, &describe);
    let blocked = Blocked::rank_one(vec![exact_library(&program)], vec![vec![Array2::ones((family.rows, WIDTH))]]);
    let (start, _) = measure(&coded, &blocked).expect("measures");
    let (fitted, bits) = fit_blocks(&coded, blocked, false).expect("fits");
    assert!(bits.total() < start.total(), "{} against {}", bits.total(), start.total());
    assert!(fitted.ranks[0].len() < WIDTH, "no merge: {:?}", fitted.ranks[0]);
    assert_eq!(fitted.ranks[0].iter().sum::<usize>(), WIDTH);
    // Merging only regroups columns: the library's map is unchanged.
    let map = |l: &Library| l.u.t().dot(&l.v);
    let error = (&map(&*fitted.libraries[0]) - &map(&exact_library(&program))).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(error <= 1e-12, "{error}");
}
