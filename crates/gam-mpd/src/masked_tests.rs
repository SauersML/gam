#![cfg(test)]
//! The masked program's derivatives against finite differences of its own KL.

use super::masked::{Library, Masked, Target, forward, gradients, sites, split, step_pieces};
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Slot,
    SlotValues,
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

const P: usize = 5;
const WIDTH: usize = 4;
const UNITS: usize = 6;

fn precision() -> DeclaredPrecision {
    DeclaredPrecision::new(40).expect("a precision")
}

/// Two tokens embedded, a ReLU layer, logits over five classes.
fn model() -> (OperatorProgram, FamilyInputs) {
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(WIDTH).expect("interface");
    let units = Interface::uniform(UNITS, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let op = |name: &str, rows: &Interface, cols: &Interface, salt: usize| {
        let m = Array2::from_shape_fn((rows.width(), cols.width()), |(i, j)| noise(salt + 31 * i + j));
        Arc::new(Operator::dense(name, rows.clone(), cols.clone(), m, precision(), Provenance::default()).expect("dense"))
    };
    let program = OperatorProgram {
        declarations: Declarations {
            parameters: 0,
            domains: vec![Domain { size: P, cycle: None }],
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

#[test]
fn the_masked_programs_gradients_are_its_own_kls_derivatives() {
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let all = sites(&program);
    let site = all.iter().find(|s| s.name == "W_in").expect("the W_in site").clone();
    let pieces = 3;
    let v = Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j));
    let u = Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j));
    let mean = Array1::from_shape_fn(WIDTH, |i| 0.1 * noise(900 + i));
    let mut masked = Masked::build(&program, vec![site], vec![Library { v: v.clone(), u: u.clone(), mean: mean.clone() }]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })];
    let fam = masked.family(&family, &masks);
    let (kl, trace, cotangent) = forward(&masked, &fam, &target).expect("forward");
    let grads = gradients(&masked, &fam, &trace, &masks, cotangent).expect("gradients");
    let total = kl.sum();
    let h = 1e-6;
    for (i, j) in [(0, 0), (1, 2), (2, 3)] {
        let mut shifted = v.clone();
        shifted[[i, j]] += h;
        masked.set_library(0, Library { v: shifted, u: u.clone(), mean: mean.clone() }).expect("set");
        let (kl_v, _, _) = forward(&masked, &fam, &target).expect("forward");
        let numeric = (kl_v.sum() - total) / h;
        let analytic = grads[0].1[[i, j]];
        assert!((numeric - analytic).abs() <= 1e-4 * (1.0 + analytic.abs()), "V[{i},{j}]: {numeric} against {analytic}");
    }
    for (i, j) in [(0, 0), (1, 4), (2, 5)] {
        let mut shifted = u.clone();
        shifted[[i, j]] += h;
        masked.set_library(0, Library { v: v.clone(), u: shifted, mean: mean.clone() }).expect("set");
        let (kl_u, _, _) = forward(&masked, &fam, &target).expect("forward");
        let numeric = (kl_u.sum() - total) / h;
        let analytic = grads[0].2[[i, j]];
        assert!((numeric - analytic).abs() <= 1e-4 * (1.0 + analytic.abs()), "U[{i},{j}]: {numeric} against {analytic}");
    }
}

#[test]
fn a_step_of_the_pieces_lowers_the_masked_kl() {
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    // The fixed U side must have a nullspace for a nonzero sum-preserving V step.
    let pieces = UNITS + 2;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let mut masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })];
    let fam = masked.family(&family, &masks);
    let before = forward(&masked, &fam, &target).expect("forward").0.sum();
    assert_eq!(super::masked::score_only(&masked, &fam, &target).expect("score only").sum(), before);
    let mut running = super::masked::Running::default();
    assert!(step_pieces(&mut masked, &family, &target, &masks, 4, 7, &mut running, super::masked::Claim::Corner).expect("steps").is_some());
    let after = forward(&masked, &fam, &target).expect("forward").0.sum();
    assert!(after < before, "{after} against {before}");
}

#[test]
fn a_full_row_rank_factor_has_no_sum_preserving_direction() {
    for width in [3, 7, 256] {
        let other = Array2::from_shape_fn((3, width), |(i, j)| if i == j { 1.0 + i as f64 } else { 0.0 });
        let g = Array2::from_shape_fn((3, 4), |(i, j)| noise(100 * i + j));
        let direction = super::masked::keep_sum(&g, &other).expect("projection");
        assert!(direction.iter().all(|x| *x == 0.0), "full rank must not leave a roundoff direction");
    }
}

#[test]
fn failed_or_nonfinite_piece_trials_restore_the_original_operators() {
    let (program, _) = model();
    let mut masked = Masked::build(&program, vec![], vec![]).expect("build");
    let originals = masked.program.operators.clone();
    let moves: Vec<_> = originals.iter().enumerate().map(|(i, op)| (i, Array2::ones(op.matrix().dim()))).collect();
    let result = super::masked::backtrack_pieces(&mut masked, &moves, 1.0, 1.0, |trial| {
        assert!(!Arc::ptr_eq(&trial.program.operators[0], &originals[0]), "the trial was installed");
        Err("injected evaluation error".to_string())
    });
    assert_eq!(result, Err("injected evaluation error".to_string()));
    assert!(masked.program.operators.iter().zip(&originals).all(|(a, b)| Arc::ptr_eq(a, b)));
    for loss in [f64::NAN, f64::NEG_INFINITY, 2.0] {
        assert_eq!(super::masked::backtrack_pieces(&mut masked, &moves, 1.0, 1.0, |_| Ok(loss)).expect("reject"), None);
        assert!(masked.program.operators.iter().zip(&originals).all(|(a, b)| Arc::ptr_eq(a, b)));
    }
    assert_eq!(super::masked::backtrack_pieces(&mut masked, &moves, 1.0, 1.0, |_| Ok(0.5)).expect("keep"), Some((1.0, 0.5)));
    assert!(!Arc::ptr_eq(&masked.program.operators[0], &originals[0]));
}

#[test]
fn malformed_libraries_are_rejected_without_partial_edits() {
    let (program, _) = model();
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("site");
    let valid = Library { v: Array2::ones((3, WIDTH)), u: Array2::ones((3, UNITS)), mean: Array1::zeros(WIDTH) };
    assert!(Masked::build(&program, vec![site.clone()], vec![]).is_err());
    assert!(Masked::build(&program, vec![], vec![valid.clone()]).is_err());
    for (v, u) in [((3, WIDTH - 1), (3, UNITS)), ((3, WIDTH + 1), (3, UNITS)), ((3, WIDTH), (2, UNITS))] {
        let invalid = Library { v: Array2::ones(v), u: Array2::ones(u), mean: Array1::zeros(WIDTH) };
        assert!(Masked::build(&program, vec![site.clone()], vec![invalid]).is_err());
    }
    let mut masked = Masked::build(&program, vec![site], vec![valid.clone()]).expect("build");
    let originals = masked.program.operators.clone();
    assert!(masked.set_library(1, valid.clone()).is_err());
    let mut invalid = valid.clone();
    invalid.v.fill(2.0);
    invalid.u[[0, 0]] = f64::NAN;
    assert!(masked.set_library(0, invalid).is_err(), "invalid U after valid V must fail atomically");
    let mut invalid = valid;
    invalid.u = Array2::ones((3, UNITS - 1));
    assert!(masked.set_library(0, invalid).is_err());
    assert!(masked.program.operators.iter().zip(&originals).all(|(a, b)| Arc::ptr_eq(a, b)));
}

#[test]
fn score_only_kl_matches_derivatives_with_underflow_and_large_offsets() {
    let target = Target { logits: ndarray::array![[0.0, 1.0], [0.0, -1000.0], [0.0, 1.0]], scored: Some(vec![true, true, false]) };
    let logits = ndarray::array![[0.0, -1000.0], [0.0, -1000.0], [0.0, -1000.0]];
    let (values, gradient) = super::masked::kl(&target, &logits);
    assert_eq!(super::masked::kl_score_only(&target, &logits), values);
    let p = 1.0 / (1.0 + (-1.0_f64).exp());
    let expected = p * (1000.0 + p.ln()) + (1.0 - p) * (1.0 - p).ln();
    assert!((values[0] - expected).abs() < 1e-10);
    assert_eq!(values[1], 0.0);
    assert_eq!(values[2], 0.0);
    assert!((gradient[[0, 1]] + p).abs() < 1e-14);
    assert!(gradient.row(2).iter().all(|g| *g == 0.0));
    let shifted = Target { logits: target.logits.mapv(|x| x + 1e15), scored: target.scored.clone() };
    assert_eq!(super::masked::kl(&shifted, &logits.mapv(|x| x - 1e15)), (values, gradient));
}

/// A behaviour scored on some rows: an unscored row adds no KL and no cotangent, and the selection
/// leaves its masks as given while it changes the scored rows'.
#[test]
fn an_unscored_row_adds_no_kl_and_keeps_its_masks() {
    let (program, family) = model();
    let logits = program.execute(&family, false).expect("executes").values[program.output].clone();
    let scored: Vec<bool> = (0..family.rows).map(|r| r % 2 == 0).collect();
    let target = Target { logits, scored: Some(scored.clone()) };
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = 3;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let masks = vec![Array2::<f64>::ones((family.rows, pieces))];
    let (kl, _, cotangent) = forward(&masked, &masked.family(&family, &masks), &target).expect("forward");
    for r in (0..family.rows).filter(|r| !scored[*r]) {
        assert_eq!(kl[r], 0.0);
        assert!(cotangent.row(r).iter().all(|x| *x == 0.0));
    }
    assert!((0..family.rows).any(|r| scored[r] && kl[r] > 0.0));
    // Nearly free KL and two rarely listed pieces: every scored row drops one.
    let mut context = super::masked::Context::new(&[pieces]);
    context.new[0][0] = 100.0;
    let coder = context.coder(vec![None; family.rows]);
    let (selected, _) = super::masked::select(&masked, &family, &target, masks, &coder, 1e-3, 2).expect("selects");
    for r in 0..family.rows {
        let unchanged = selected[0].row(r).iter().all(|x| *x == 1.0);
        assert!(unchanged != scored[r], "row {r} (scored {})", scored[r]);
    }
}

#[test]
fn splitting_a_library_keeps_its_sum_and_lists_both_halves_where_the_piece_was_on() {
    let pieces = 3;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::from_shape_fn(WIDTH, |i| 0.1 * noise(900 + i)),
    };
    let rows = 20;
    let x = Array2::from_shape_fn((rows, WIDTH), |(t, j)| noise(1000 + 5 * t + j));
    let mask = Array2::from_shape_fn((rows, pieces), |(t, c)| if c == 2 && t > 0 { 0.0 } else { 1.0 });
    let (grown, masks, origin) = split(&library, &x, &mask);
    assert_eq!(origin, vec![0, 0, 1, 1, 2]);
    // Pieces 0 and 1 are listed by every input and split; piece 2 by one input and kept whole.
    assert_eq!(grown.v.nrows(), 5);
    assert_eq!(masks.dim(), (rows, 5));
    let sum = |l: &Library| l.v.t().dot(&l.u);
    let error = (&sum(&grown) - &sum(&library)).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(error < 1e-12, "{error}");
    assert_eq!(masks.column(0), masks.column(1));
}

#[test]
fn a_diagonal_gain_is_a_column_scale_forward_and_backward() {
    let (mut program, family) = model();
    let residual = Interface::uniform(WIDTH, 1, LabelKind::Unit, 0).expect("interface");
    let gains = Array1::from_shape_fn(WIDTH, |i| 0.5 + noise(1200 + i));
    let mut present = Array2::from_elem((WIDTH, WIDTH), false);
    for i in 0..WIDTH {
        present[[i, i]] = true;
    }
    let gain = Operator::blocks("gain", residual.clone(), residual.clone(), Array2::from_diag(&gains), present, precision(), Provenance::default())
        .expect("blocks");
    assert!(gain.diagonal().is_some());
    // The embedding writes the unit-labelled residual, then the gain reads it.
    let e = program.operators[0].matrix();
    program.operators[0] = Arc::new(Operator::dense("E", residual.clone(), program.operators[0].cols.clone(), e, precision(), Provenance::default()).expect("dense"));
    let w_in = program.operators[1].matrix();
    program.operators[1] = Arc::new(Operator::dense("W_in", program.operators[1].rows.clone(), residual, w_in, precision(), Provenance::default()).expect("dense"));
    program.operators.push(Arc::new(gain));
    let gain_op = program.operators.len() - 1;
    program.nodes.insert(3, Node::Affine { terms: vec![(2, gain_op)], bias: None });
    program.nodes[4] = Node::Affine { terms: vec![(3, 1)], bias: None };
    program.nodes[5] = Node::Pointwise { input: 4, laws: vec![Law::Relu; UNITS] };
    program.nodes[6] = Node::Affine { terms: vec![(5, 2)], bias: None };
    program.nodes[7] = Node::Readout { input: 6, basis: 0 };
    program.output = 7;
    let trace = program.execute(&family, true).expect("executes");
    let rounded = program.operators[gain_op].diagonal().expect("a diagonal");
    let expected = &trace.values[2] * &rounded;
    let error = (&trace.values[3] - &expected).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(error == 0.0, "{error}");
    let cotangent = Array2::from_shape_fn(trace.values[7].dim(), |(r, c)| noise(1300 + 7 * r + c));
    let back = super::derivatives::vjp(&program, &family, &trace, cotangent.clone()).expect("reverse");
    let tangent = Array2::from_shape_fn((WIDTH, WIDTH), |(i, j)| if i == j { noise(1400 + i) } else { 0.0 });
    let tangents = [(gain_op, tangent.clone())].into_iter().collect();
    let forward = super::derivatives::jvp(&program, &family, &trace, &tangents).expect("forward");
    let left: f64 = cotangent.iter().zip(forward.iter()).map(|(a, b)| a * b).sum();
    let moved = trace.values[2].dot(&tangent.t());
    let right: f64 = back[3].as_ref().expect("a cotangent").iter().zip(moved.iter()).map(|(a, b)| a * b).sum();
    assert!((left - right).abs() <= 1e-9 * left.abs().max(1.0), "{left} against {right}");
}

#[test]
fn a_set_that_carries_over_from_the_previous_input_is_cheap_to_explain() {
    use super::masked::Context;
    // One site of 16 pieces; inputs 0..8 form one sequence whose set {2, 5, 11} never changes.
    let rows = 8;
    let mask = Array2::from_shape_fn((rows, 16), |(_, c)| if [2, 5, 11].contains(&c) { 1.0 } else { 0.0 });
    let previous: Vec<Option<usize>> = (0..rows).map(|r| r.checked_sub(1)).collect();
    let mut context = Context::new(&[16]);
    context.absorb(std::slice::from_ref(&mask), &previous);
    let coder = context.coder(previous);
    let bits = coder.bits(std::slice::from_ref(&mask));
    // The first input lists its set; every later one only confirms that three pieces stay on.
    assert!(bits[1] < bits[0], "{bits:?}");
    assert!(bits[1] < 1.0, "{bits:?}");
}

#[test]
fn pieces_grown_from_what_selection_leaves_out_recover_its_kl() {
    use super::masked::{Running, dropped_atoms, with_pieces};
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = 3;
    // Three pieces: the site map's leading three singular directions; the rest of the map is in
    // no piece.
    let w = super::masked::matrix(&program, &site).expect("matrix");
    let decomposed = super::dense::svd(w.view(), false).expect("svd");
    let mut v = Array2::<f64>::zeros((pieces, WIDTH));
    let mut u = Array2::<f64>::zeros((pieces, UNITS));
    for c in 0..pieces.min(decomposed.singular_values.len()) {
        let s = decomposed.singular_values[c].sqrt();
        v.row_mut(c).assign(&(&decomposed.vt.row(c) * s));
        u.row_mut(c).assign(&(&decomposed.u.column(c) * s));
    }
    let library = Library { v, u, mean: Array1::zeros(WIDTH) };
    let mut masked = Masked::build(&program, vec![site.clone()], vec![library]).expect("builds");
    // Piece 2 is dropped everywhere.
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(_, c)| if c == 2 { 0.0 } else { 1.0 })];
    let mut running = Running::default();
    step_pieces(&mut masked, &family, &target, &masks, 4, 7, &mut running, super::masked::Claim::Corner).expect("steps");
    let fam = masked.family(&family, &masks);
    let (before, trace, _) = forward(&masked, &fam, &target).expect("forward");
    let (v, u) = dropped_atoms(&masked, 0, &trace, &masks[0], &running, 1000.0, 0.0).expect("atoms");
    assert!(v.nrows() >= 1, "no atom");
    let grown = with_pieces(&masked.library(0).expect("library"), &v, &u).expect("grown");
    let added = v.nrows();
    let bigger = Masked::build(&program, vec![site], vec![grown]).expect("builds");
    let on = vec![Array2::from_shape_fn((family.rows, pieces + added), |(_, c)| if c == 2 { 0.0 } else { 1.0 })];
    let (after, _, _) = forward(&bigger, &bigger.family(&family, &on), &target).expect("forward");
    assert!(after.sum() < before.sum(), "{} against {}", after.sum(), before.sum());
}

#[test]
fn sites_are_the_hidden_maps_and_their_statistics_build_an_exact_library() {
    use super::masked::site_statistics;
    use super::pieces::fisher_svd;
    let (program, family) = model();
    // The embedding `E` reads tokens and `W_out` writes token logits: only `W_in` is a site.
    let all = sites(&program);
    assert_eq!(all.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["W_in"]);
    let half = family.rows / 2;
    let batches = [family.select(&(0..half).collect::<Vec<_>>()), family.select(&(half..family.rows).collect::<Vec<_>>())];
    let measured = site_statistics(&program, &all, batches, 4, 3).expect("statistics");
    let trace = program.execute(&family, false).expect("executes");
    let x = super::masked::read_values(&trace, &all[0]).expect("reads");
    let mean = x.mean_axis(ndarray::Axis(0)).expect("rows");
    assert!((&measured[0].mean - &mean).iter().all(|d| d.abs() < 1e-12));
    assert!(measured[0].fisher.diag().iter().all(|f| *f >= 0.0) && measured[0].fisher.diag().sum() > 0.0);
    let library = fisher_svd(&measured[0]).expect("library");
    assert!(library.exactness(&measured[0].w) < 1e-9, "{}", library.exactness(&measured[0].w));
}

/// The box claim's error is the KL expected over every off gate drawn uniform: against the mean
/// KL of sampled gates on small pieces (where second order is accurate).
#[test]
fn the_box_expectation_is_the_kl_expected_over_uniform_off_gates() {
    use super::masked::{box_excess, fisher};
    let (mut program, family) = model();
    // The second-order comparison needs a smooth neighborhood. ReLU gate crossings are not
    // captured by the local Fisher, even when the factors themselves are small.
    program.nodes[4] = Node::Pointwise { input: 3, laws: vec![Law::Identity; UNITS] };
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = 3;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| 0.03 * noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| 0.03 * noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 2 == 0 { 0.0 } else { 1.0 })];
    let fam = masked.family(&family, &masks);
    // Anchor at the masked state so adding off gates has nonnegative excess. Against an
    // unrelated teacher it may instead improve KL, making a positivity assertion invalid.
    let target = Target::every_row(masked.program.execute(&fam, false).expect("anchor").values[masked.program.output].clone());
    let (kl, trace, cotangent) = forward(&masked, &fam, &target).expect("forward");
    let fishers: Vec<Array2<f64>> =
        fisher(&masked, &fam, &trace, &target, 256, 11, true).expect("fisher").into_iter().map(|(_, f)| f.expect("written")).collect();
    let (excess, _) = box_excess(&masked, &fam, &trace, &masks, cotangent, &fishers, false).expect("excess");
    let draws = 400;
    let mut sampled = 0.0;
    for d in 0..draws {
        let gates = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| {
            if masks[0][[r, c]] > 0.0 { 1.0 } else { 0.5 * (noise(31 * d + 7 * r + c + 100_000) + 1.0) }
        })];
        sampled += forward(&masked, &masked.family(&family, &gates), &target).expect("forward").0.sum();
    }
    let sampled_excess = sampled / draws as f64 - kl.sum();
    let predicted = excess.sum();
    assert!(predicted > 0.0 && sampled_excess > 0.0, "{predicted} against {sampled_excess}");
    assert!((predicted - sampled_excess).abs() <= 0.3 * sampled_excess, "{predicted} against {sampled_excess}");
}

/// Two-input sequences through causal attention: the second input reads the first's values, so a
/// mask on the first changes the second's KL. Selection accepts per sequence, on exactly the masks
/// it commits, so its result never codes worse than its start and its KL is that of its masks.
#[test]
fn selection_never_commits_masks_it_did_not_evaluate_across_attention() {
    use super::masked::{Coder, select};
    use super::operator_program::{Rotary, Scale, SequenceLayout};
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(4).expect("interface");
    let op = |name: &str, rows: &Interface, cols: &Interface, salt: usize| {
        let m = Array2::from_shape_fn((rows.width(), cols.width()), |(i, j)| 1.5 * noise(salt + 31 * i + j));
        Arc::new(Operator::dense(name, rows.clone(), cols.clone(), m, precision(), Provenance::default()).expect("dense"))
    };
    let program = OperatorProgram {
        declarations: Declarations { parameters: 0, domains: vec![Domain { size: P, cycle: None }], slots: vec![Slot::Token { domain: 0 }] },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![op("E", &model, &tokens, 1), op("Q", &model, &model, 2), op("K", &model, &model, 3), op("V", &model, &model, 4), op("O", &tokens, &model, 5)],
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },              // 0
            Node::Affine { terms: vec![(0, 0)], bias: None }, // 1 x
            Node::Affine { terms: vec![(1, 1)], bias: None }, // 2 q
            Node::Affine { terms: vec![(1, 2)], bias: None }, // 3 k
            Node::Affine { terms: vec![(1, 3)], bias: None }, // 4 v
            Node::Attend { query: 2, key: 3, value: 4, scale: Scale::InverseSqrt(4), rotary: Some(Rotary { base: 10000, dims: 4, half_split: true }), causal: true },
            Node::Affine { terms: vec![(5, 4)], bias: None }, // 6 logits
            Node::Readout { input: 6, basis: 0 },             // 7
        ],
        output: 7,
    };
    let (sequences, length) = (6u32, 2u32);
    let ids: Vec<u32> = (0..sequences * length).map(|i| (i * 3 + 1) % P as u32).collect();
    let family = FamilyInputs {
        rows: ids.len(),
        slots: vec![SlotValues::Tokens(ids)],
        layout: Some(SequenceLayout {
            sequence: (0..sequences * length).map(|i| i / length).collect(),
            position: (0..sequences * length).map(|i| i % length).collect(),
        }),
    };
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "V").expect("the value site");
    let w = super::masked::matrix(&program, &site).expect("matrix");
    let decomposed = super::dense::svd(w.view(), false).expect("svd");
    let pieces = decomposed.singular_values.len();
    let mut v = Array2::<f64>::zeros((pieces, 4));
    let mut u = Array2::<f64>::zeros((pieces, 4));
    for c in 0..pieces {
        let s = decomposed.singular_values[c].sqrt();
        v.row_mut(c).assign(&(&decomposed.vt.row(c) * s));
        u.row_mut(c).assign(&(&decomposed.u.column(c) * s));
    }
    let masked = Masked::build(&program, vec![site], vec![Library { v, u, mean: Array1::zeros(4) }]).expect("builds");
    let start = vec![Array2::<f64>::ones((family.rows, pieces))];
    let coder = Coder::ran(vec![Array1::from_elem(pieces, 0.5)], family.rows);
    let observations = 20.0;
    let code_of = |masks: &[Array2<f64>]| {
        let kl = forward(&masked, &masked.family(&family, masks), &target).expect("forward").0;
        coder.bits(masks).sum() + kl.sum() * observations / std::f64::consts::LN_2
    };
    let before = code_of(&start);
    let (selected, kl) = select(&masked, &family, &target, start, &coder, observations, 8).expect("selects");
    let fresh = forward(&masked, &masked.family(&family, &selected), &target).expect("forward").0;
    assert!((&kl - &fresh).iter().all(|d| d.abs() <= 1e-12), "the returned KL is not the committed masks'");
    let after = code_of(&selected);
    assert!(after <= before + 1e-9, "{after} against {before}");
}

/// A mask is a weight intervention on the read itself: with `W = diag(2, 3)`, a library measured
/// about `μ = (1, −2)`, and only the first piece on (`B_m = diag(2, 0)`), the masked site maps `0` to
/// `0` and `(1, 1)` to `(2, 0)`: no bias appears at a bias-free site.
#[test]
fn a_masked_site_reads_uncentred_so_no_bias_appears() {
    let width = Interface::native(2).expect("interface");
    let w = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).expect("shape");
    let program = OperatorProgram {
        declarations: Declarations { parameters: 0, domains: Vec::new(), slots: vec![Slot::Raw { width: 2 }] },
        bases: Vec::new(),
        operators: vec![Arc::new(Operator::dense("W", width.clone(), width.clone(), w, precision(), Provenance::default()).expect("dense"))],
        rules: Vec::new(),
        nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }],
        output: 1,
    };
    let site = sites(&program).into_iter().next().expect("the site");
    let library = Library {
        v: Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape"),
        u: Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).expect("shape"),
        mean: Array1::from_vec(vec![1.0, -2.0]),
    };
    let masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let inputs = FamilyInputs {
        rows: 2,
        slots: vec![SlotValues::Raw(Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("shape"))],
        layout: None,
    };
    let masks = vec![Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 0.0]).expect("shape")];
    let out = masked.program.execute(&masked.family(&inputs, &masks), false).expect("executes").values[masked.program.output].clone();
    let expected = [[0.0, 0.0], [2.0, 0.0]];
    for r in 0..2 {
        for c in 0..2 {
            assert!((out[[r, c]] - expected[r][c]).abs() < 1e-12, "{out}");
        }
    }
}

/// Every piece on stays the library's map: steps on either side keep `Σ_c u_c v_cᵀ` to rounding
/// while they lower the masked KL.
#[test]
fn steps_keep_every_piece_on_the_same_map() {
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = 9;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let sum = |l: &Library| l.u.t().dot(&l.v);
    let anchor = sum(&library);
    let scale = anchor.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + 2 * c) % 3 == 0 { 0.0 } else { 1.0 })];
    let mut running = super::masked::Running::default();
    let mut stepped = 0;
    for seed in 0..4u64 {
        if step_pieces(&mut masked, &family, &target, &masks, 4, seed, &mut running, super::masked::Claim::Corner).expect("steps").is_some() {
            stepped += 1;
        }
        let drift = (&sum(&masked.library(0).expect("library")) - &anchor).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(drift <= 1e-9 * scale, "seed {seed}: the sum moved by {drift:e}");
    }
    assert!(stepped > 0, "no step lowered the KL");
}

#[test]
fn shrunk_solve_matches_spectral_preconditioning_and_solves_the_shifted_system() {
    use super::masked::{shrunk_direction, shrunk_inverse};
    for rank in [1, 7, 16] {
        let x = Array2::from_shape_fn((rank, 16), |(i, j)| noise(31 * i + j + 7));
        let m = x.t().dot(&x);
        let g = Array2::from_shape_fn((5, 16), |(i, j)| noise(71 * i + j + 91));
        let actual = shrunk_direction(&m, &g).expect("solve");
        let expected = g.dot(&shrunk_inverse(&m).expect("spectral"));
        assert!((&actual - &expected).iter().all(|v| v.abs() < 1e-11));
        let lambda = m.diag().sum() / 16.0;
        let mut shifted = m;
        shifted.diag_mut().mapv_inplace(|v| v + lambda);
        assert!((&actual.dot(&shifted) - &g).iter().all(|v| v.abs() < 1e-11));
    }
    assert_eq!(shrunk_direction(&Array2::<f64>::zeros((3, 3)), &Array2::<f64>::ones((2, 3))).expect("zero"), Array2::<f64>::zeros((2, 3)));
}

#[test]
fn shared_cpu_fisher_matches_full_vocabulary_reverse_passes_for_grouped_masks() {
    use super::masked::fisher;
    let (program, base) = model();
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("site");
    let library = Library {
        v: Array2::from_shape_fn((3, WIDTH), |(i, j)| noise(71 * i + j)),
        u: Array2::from_shape_fn((3, UNITS), |(i, j)| noise(31 * i + j + 17)),
        mean: Array1::zeros(WIDTH),
    };
    let masked = Masked::build_blocks(&program, vec![site], vec![library], vec![vec![2, 1]]).expect("blocks");
    let family = masked.family(&base, &[Array2::from_shape_fn((base.rows, 2), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })]);
    let trace = masked.program.execute(&family, false).expect("trace");
    let logits = &trace.values[masked.program.output];
    let target = Target { logits: logits.clone(), scored: Some((0..base.rows).map(|r| r % 3 != 1).collect()) };
    let actual = fisher(&masked, &family, &trace, &target, 4, 73, true).expect("fisher");
    let mut h = Array2::zeros((base.rows, 2));
    let mut f = Array2::zeros((UNITS, UNITS));
    let mut rng = 73u64;
    for _ in 0..4 {
        let mut seed = Array2::zeros(logits.dim());
        // The reference constructs q - e_y at the full vocabulary and differentiates every node.
        for r in 0..base.rows {
            if !target.scores(r) { seed.row_mut(r).fill(0.0); continue; }
            let max = logits.row(r).iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut q = logits.row(r).mapv(|v| (v - max).exp());
            q /= q.sum();
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let mut pick = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let mut label = q.len() - 1;
            for (c, p) in q.iter().enumerate() { if pick < *p { label = c; break; } pick -= p; }
            seed.row_mut(r).assign(&q);
            seed[[r, label]] -= 1.0;
        }
        let back = super::derivatives::vjp(&masked.program, &family, &trace, seed).expect("reference reverse");
        let g = masked.to_blocks(0, &(back[masked.masked[0]].as_ref().expect("mask cotangent") * &trace.values[masked.z[0]]));
        h += &(&g * &g);
        let written = back[masked.sites[0].writes[0]].as_ref().expect("written");
        f += &written.t().dot(written);
    }
    h /= 4.0;
    f /= 4.0 * target.scored_rows() as f64;
    assert!((&actual[0].0 - &h).iter().all(|v| v.abs() < 1e-11));
    assert!((actual[0].1.as_ref().expect("Fisher") - &f).iter().all(|v| v.abs() < 1e-11));
}

#[test]
fn piece_space_projection_matches_width_space_and_preserves_the_native_sum() {
    let left = Array2::from_shape_fn((20, 8), |(i, j)| noise(31 * i + j + 1));
    let right = Array2::from_shape_fn((8, 256), |(i, j)| noise(17 * i + j + 9));
    let other = left.dot(&right);
    let g = Array2::from_shape_fn((20, 7), |(i, j)| noise(43 * i + j + 3));
    let direction = super::masked::keep_sum(&g, &other).expect("piece projection");
    assert!(other.t().dot(&direction).iter().all(|v| v.abs() < 1e-10), "the native sum moved");
    let gram = other.t().dot(&other);
    // Assemble the reference symmetrically, just as the historical implementation did.
    let mut sym = gram;
    for i in 0..sym.nrows() { for j in i + 1..sym.ncols() { let x = 0.5 * (sym[[i, j]] + sym[[j, i]]); sym[[i, j]] = x; sym[[j, i]] = x; } }
    let spectrum = super::dense::eigh(sym.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).expect("width spectrum");
    let mut scaled = spectrum.vectors.clone();
    for (k, value) in spectrum.values.iter().enumerate() { scaled.column_mut(k).mapv_inplace(|x| if *value > spectrum.band { x / value } else { 0.0 }); }
    let reference = &g - &other.dot(&scaled.dot(&spectrum.vectors.t()).dot(&other.t().dot(&g)));
    assert!((&direction - &reference).iter().all(|v| v.abs() < 1e-10));
}

/// A step under the box claim is judged on the claim's error, the worst over the box points
/// ([`super::masked::box_excess_at`]), not on the uniform expectation that only steers it: the
/// totals a step compares are the masks' KL plus that worst case, before and after.
#[test]
fn a_box_step_is_judged_on_the_claims_worst_case_error() {
    use super::masked::{Claim, Running, box_excess_at, expected_box_excess_at, score_only};
    let (program, family) = model();
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = UNITS + 2;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let mut masked = Masked::build(&program, vec![site.clone()], vec![library]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })];
    let claim_error = |m: &Masked, fishers: &[Array2<f64>]| -> f64 {
        score_only(m, &m.family(&family, &masks), &target).expect("kl").sum() + box_excess_at(m, &family, &target, &masks, fishers).expect("box").sum()
    };
    let mut running = Running::default();
    let mut judged = 0;
    for seed in 0..6u64 {
        let before = Masked::build(&program, vec![site.clone()], vec![masked.library(0).expect("library")]).expect("rebuilds");
        let Some((total, trial)) = step_pieces(&mut masked, &family, &target, &masks, 4, seed, &mut running, Claim::Box).expect("steps") else { continue };
        // The worst case is not the expectation here, so the test tells the two apart.
        let expected = score_only(&before, &before.family(&family, &masks), &target).expect("kl").sum()
            + expected_box_excess_at(&before, &family, &target, &masks, &running.fishers).expect("expected").sum();
        let worst = claim_error(&before, &running.fishers);
        assert!(worst > expected, "the box points add nothing over the expectation here: {worst} against {expected}");
        assert!((total - worst).abs() <= 1e-12 * worst.abs(), "the step's total {total} is not the claim's error {worst}");
        let after = claim_error(&masked, &running.fishers);
        assert!((trial - after).abs() <= 1e-12 * after.abs(), "the step's trial {trial} is not the claim's error {after}");
        assert!(after < worst, "{after} against {worst}");
        judged += 1;
    }
    assert!(judged > 0, "no step was taken");
}

/// The box claim's error is the worst over the box points it evaluates
/// ([`super::masked::box_excess_at`]), so it is never below the uniform expectation
/// ([`super::masked::box_excess`], the previous test's term) nor below zero (the masks themselves
/// are a point), and never below any layer's vertex (that layer's sites at the masks, the other
/// layers' gates all on), whose exact KL is evaluated here independently. With every gate on the
/// box is one point, the masks, and the error is zero.
#[test]
fn the_box_claims_error_is_its_worst_point_at_least_the_expectation() {
    use super::masked::{box_excess_at, expected_box_excess_at, fisher, matrix, score_only};
    let (mut program, family) = model();
    // A second hidden map, `W_mid` on the residual stream before `W_in`: two sites, two layers.
    let residual = Interface::native(WIDTH).expect("interface");
    let mid = Array2::from_shape_fn((WIDTH, WIDTH), |(i, j)| (if i == j { 1.0 } else { 0.0 }) + 0.3 * noise(300 + 31 * i + j));
    program.operators.push(Arc::new(Operator::dense("W_mid", residual.clone(), residual, mid, precision(), Provenance::default()).expect("dense")));
    let w_mid = program.operators.len() - 1;
    program.nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
        Node::Affine { terms: vec![(2, w_mid)], bias: None },
        Node::Affine { terms: vec![(3, 1)], bias: None },
        Node::Pointwise { input: 4, laws: vec![Law::Relu; UNITS] },
        Node::Affine { terms: vec![(5, 2)], bias: None },
        Node::Readout { input: 6, basis: 0 },
    ];
    program.output = 7;
    let chosen: Vec<_> = sites(&program).into_iter().filter(|s| s.name == "W_in" || s.name == "W_mid").collect();
    assert_eq!(chosen.len(), 2, "two sites in two layers");
    let pieces = 3;
    let libraries: Vec<Library> = chosen
        .iter()
        .enumerate()
        .map(|(k, site)| {
            let (d_out, d_in) = matrix(&program, site).expect("map").dim();
            Library {
                v: Array2::from_shape_fn((pieces, d_in), |(i, j)| 0.5 * noise(900 + 100 * k + 7 * i + j)),
                u: Array2::from_shape_fn((pieces, d_out), |(i, j)| 0.5 * noise(1900 + 100 * k + 7 * i + j)),
                mean: Array1::zeros(d_in),
            }
        })
        .collect();
    let masked = Masked::build(&program, chosen, libraries).expect("builds");
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let masks: Vec<Array2<f64>> = (0..2).map(|k| Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c + k) % 2 == 0 { 0.0 } else { 1.0 })).collect();
    let fam = masked.family(&family, &masks);
    let (_, trace, _) = forward(&masked, &fam, &target).expect("forward");
    let fishers: Vec<Array2<f64>> =
        fisher(&masked, &fam, &trace, &target, 64, 11, true).expect("fisher").into_iter().map(|(_, f)| f.expect("written")).collect();
    let corner = score_only(&masked, &fam, &target).expect("kl");
    let expected = expected_box_excess_at(&masked, &family, &target, &masks, &fishers).expect("expected");
    let error = box_excess_at(&masked, &family, &target, &masks, &fishers).expect("error");
    // Without a layout every input is its own sequence, so each row is charged its own worst point.
    let close = |a: f64, b: f64| a >= b - 1e-12 * (1.0 + b.abs());
    let mut above = 0;
    for r in 0..family.rows {
        assert!(error[r] >= expected[r].max(0.0), "row {r}: error {} below the expectation {} or zero", error[r], expected[r]);
        above += usize::from(error[r] > expected[r].max(0.0));
    }
    for layer in 0..2 {
        let vertex: Vec<Array2<f64>> = (0..2).map(|k| if k == layer { masks[k].clone() } else { Array2::ones(masks[k].dim()) }).collect();
        let kl_vertex = score_only(&masked, &masked.family(&family, &vertex), &target).expect("vertex");
        for r in 0..family.rows {
            assert!(close(error[r], kl_vertex[r] - corner[r]), "row {r}: error {} below layer {layer}'s vertex {}", error[r], kl_vertex[r] - corner[r]);
        }
    }
    assert!(above > 0, "no box point beyond the expectation: the test would not tell the worst case from it");
    let on: Vec<Array2<f64>> = masks.iter().map(|m| Array2::ones(m.dim())).collect();
    let none = box_excess_at(&masked, &family, &target, &on, &fishers).expect("all on");
    // Zero up to the rounding of the points' own forwards (they run by other routes than the masks').
    assert!(none.iter().zip(corner.iter()).all(|(e, k)| e.abs() <= 1e-12 * (1.0 + k.abs())), "with every gate on the error is the masks' own: {none:?}");
}

/// Under the box claim too, a selection resumed between any two rounds from the state it showed
/// its checkpoint (the masks' excess among it) ends exactly as the uninterrupted one.
#[test]
fn a_resumed_box_selection_is_the_uninterrupted_one() {
    use super::masked::{Coder, Progress, Resume, Round, fisher, select_resumable};
    let (program, family) = model();
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = UNITS + 2;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let target = Target::every_row(program.execute(&family, false).expect("executes").values[program.output].clone());
    let start = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })];
    let fam = masked.family(&family, &start);
    let (_, trace, _) = forward(&masked, &fam, &target).expect("forward");
    let fishers: Vec<Array2<f64>> =
        fisher(&masked, &fam, &trace, &target, 64, 11, true).expect("fisher").into_iter().map(|(_, f)| f.expect("written")).collect();
    let coder = Coder::ran(vec![Array1::from_elem(pieces, 3.0)], family.rows);
    let run = |resume: Option<Resume>| -> (Vec<Array2<f64>>, Array1<f64>, Vec<Resume>) {
        let mut states = Vec::new();
        let mut checkpoint = |progress: &Progress<'_>| -> Result<(), String> {
            states.push(progress.to_resume());
            Ok(())
        };
        let (masks, kl) =
            select_resumable(&masked, &family, &target, start.clone(), resume, &coder, 256.0, 2, Some(&fishers), &mut |_: &Round<'_>| Ok(()), &mut checkpoint)
                .expect("selection");
        (masks, kl, states)
    };
    let (masks, kl, states) = run(None);
    assert!(states.len() >= 3, "the selection took too few rounds to resume mid-way: {}", states.len());
    for k in 1..states.len() {
        let (resumed_masks, resumed_kl, resumed_states) = run(Some(states[k].clone()));
        assert_eq!(resumed_masks, masks, "resumed at round {k}: other masks");
        assert_eq!(resumed_kl, kl, "resumed at round {k}: another KL");
        assert_eq!(resumed_states, states[k..], "resumed at round {k}: other states on the way");
    }
}
