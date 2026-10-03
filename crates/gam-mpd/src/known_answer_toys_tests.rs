#![cfg(test)]
//! The known-answer toys of `test_support::known_answer_toys` against the owners that can
//! read their answers today: each toy computes its task, its written-down decomposition
//! reproduces it, and the attention owners recover the planted routing structure. A decomposition engine is held to the same `*Truth` values.

use crate::attention::AttentionExecution;
use crate::joint_operators::{attention_letters, query_key_operators, routing_laws};
use crate::test_support::known_answer_toys::{
    INDUCTION_VOCAB, RESID_FEATURES, ResidMlp, dyadic_features, fourier_modadd, induction_toy, repeated_sequence,
    resid_feature_duplicated, resid_mlp,
};
use crate::test_support::test_governor;
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use ndarray::{Array2, s};
use std::f64::consts::PI;

const RESID_SEED: u64 = 0x2951_00a1;
const MODADD_SEED: u64 = 0x2951_00a2;

/// The ResidMLP toys compute `y = x + relu(x)` exactly, on dense inputs, at every depth.
#[test]
fn resid_mlp_toys_compute_x_plus_relu_x_exactly() {
    let inputs = dyadic_features(RESID_SEED, 64);
    for layers in 1..=3 {
        let (model, _) = resid_mlp(layers, RESID_SEED + layers as u64);
        for x in inputs.rows() {
            assert_eq!(model.forward(x), x.mapv(|value| value + value.max(0.0)), "{layers} layers");
        }
    }
}

/// The written-down components are the model: they sum to every weight exactly, and
/// removing one feature's component removes exactly that feature's `relu`, on every input.
#[test]
fn resid_mlp_components_sum_to_the_weights_and_each_carries_one_feature() {
    let inputs = dyadic_features(RESID_SEED ^ 1, 32);
    for layers in 1..=3 {
        let (model, truth) = resid_mlp(layers, RESID_SEED + layers as u64);
        assert_eq!(truth.components.len(), RESID_FEATURES);
        for (layer, weights) in model.layers.iter().enumerate() {
            let pieces: Vec<_> = truth.components.iter().filter(|component| component.layer == layer).collect();
            let w_in = pieces.iter().fold(Array2::<f64>::zeros(weights.w_in.dim()), |sum, piece| sum + &piece.w_in);
            let w_out = pieces.iter().fold(Array2::<f64>::zeros(weights.w_out.dim()), |sum, piece| sum + &piece.w_out);
            assert_eq!(w_in, weights.w_in, "{layers} layers, layer {layer}");
            assert_eq!(w_out, weights.w_out, "{layers} layers, layer {layer}");
        }
        for component in &truth.components {
            assert_eq!(component.layer, component.feature % layers);
            assert_eq!(component.units.len(), if resid_feature_duplicated(component.feature) { 2 } else { 1 });
            let mut ablated: ResidMlp = model.clone();
            ablated.layers[component.layer].w_in -= &component.w_in;
            ablated.layers[component.layer].w_out -= &component.w_out;
            for x in inputs.rows() {
                let mut expected = x.mapv(|value| value + value.max(0.0));
                expected[component.feature] = x[component.feature];
                assert_eq!(ablated.forward(x), expected, "{layers} layers, feature {}", component.feature);
            }
        }
    }
}

const MODULUS: usize = 17;
const KEY_FREQUENCIES: [usize; 3] = [2, 5, 6];
const PHASES: usize = 9;

/// Every `(a, b)` of `ℤ_p²` is added correctly, by at least the analytic margin less the
/// logits' rounding: `units` products of entries within `2u(1 + 2π)` of their cosines, summed.
#[test]
fn fourier_modadd_adds_every_pair() {
    let (model, truth) = fourier_modadd(MODULUS, &KEY_FREQUENCIES, PHASES, MODADD_SEED);
    assert!(truth.margin_floor > 0.0);
    let units = model.w_in.nrows() as f64;
    let entry = 2.0 * UNIT_ROUNDOFF * (1.0 + 2.0 * PI);
    // Each pre-activation is 4 products of entries each within `entry`, each logit `units`
    // products of a pre-activation (at most 2) and a write (at most 1).
    let rounding = 2.0 * units * (2.0 * entry + 4.0 * 2.0 * entry + accumulation_growth(model.w_in.nrows() + 4) * 4.0);
    let modulus = model.modulus;
    for a in 0..modulus {
        for b in 0..modulus {
            let logits = model.logits(a, b);
            let answer = (a + b) % modulus;
            let runner_up = (0..modulus).filter(|&c| c != answer).map(|c| logits[c]).fold(f64::NEG_INFINITY, f64::max);
            assert!(
                logits[answer] - runner_up >= truth.margin_floor - rounding,
                "({a}, {b}): margin {} below {}",
                logits[answer] - runner_up,
                truth.margin_floor
            );
        }
    }
}

/// Whether head `head` scores `key` above every other admissible key of `query` by `gap`:
/// each computed score is within its radius of the exact one.
fn attends(executed: &AttentionExecution, head: usize, query: usize, key: usize, gap: f64) -> bool {
    let (scores, radius) = (executed.scores.slice(s![head, query, ..=query]), executed.score_radius.slice(s![head, query, ..=query]));
    (0..=query).filter(|&other| other != key).all(|other| scores[key] - scores[other] + radius[key] + radius[other] >= gap)
}

/// Layer 0 attends from every position to the previous one in both heads, layer 1 from
/// each second-copy token to the position after its first occurrence, both by their planted
/// gaps, and the logits name the next token everywhere it is determined.
#[test]
fn induction_toy_predicts_the_repeated_half() {
    let (toy, truth) = induction_toy();
    let n = INDUCTION_VOCAB - 1;
    for seed in 0..8 {
        let tokens = repeated_sequence(seed);
        let run = toy.forward(test_governor(), &tokens);
        for head in 0..2 {
            for t in 1..tokens.len() {
                assert!(attends(&run.layers[0], head, t, t - 1, truth.previous_token_gap), "seed {seed}: head {head} at {t}");
            }
        }
        for t in n + 1..2 * n {
            assert!(attends(&run.layers[1], 0, t, t - n + 1, truth.induction_gap), "seed {seed}: induction at {t}");
            let logits = run.logits.row(t);
            let answer = tokens[t + 1];
            let best = (0..INDUCTION_VOCAB).filter(|&c| c != answer).map(|c| logits[c]).fold(f64::NEG_INFINITY, f64::max);
            assert!(logits[answer] > best, "seed {seed}: position {t} predicts {answer}");
        }
    }
}

/// The routing laws are the planted ones: both previous-token heads route as one law and
/// the induction head alone. Each law's transport is its planted copy, and the induction
/// score form reads the current token against the previous-token subspace layer 0 writes.
#[test]
fn induction_toy_laws_transports_and_composition_are_planted() {
    let (toy, truth) = induction_toy();
    let governor = test_governor();
    let expected = [&truth.previous_token_laws, &truth.induction_laws];
    let transports = [&truth.previous_token_transport, &truth.copy_transport];
    for (layer, native) in toy.layers.iter().enumerate() {
        let operators = query_key_operators(native, None).expect("operators");
        assert_eq!(&routing_laws(governor, native, &operators).expect("laws").laws, expected[layer]);
        let letters = attention_letters(governor, native, None, None).expect("letters");
        assert_eq!(letters.transports().len(), 1);
        let distance = (&letters.transports()[0] - transports[layer]).iter().map(|entry| entry * entry).sum::<f64>().sqrt();
        assert!(distance <= letters.bands()[0], "layer {layer}: {distance:e} beyond {:e}", letters.bands()[0]);
    }
    let operators = query_key_operators(&toy.layers[1], None).expect("operators");
    let form = operators.pass_through(0).expect("the induction head has no rotary plane");
    assert_eq!(form.left().dot(&form.right().t()), truth.induction_score_form);
    // K-composition: the key reads what layer 0 writes, so the score form through layer 0's
    // transport is the gain on the token subspace alone: the current token against the
    // token one position back.
    let composed = truth.induction_score_form.dot(&truth.previous_token_transport);
    let gain = composed[[0, 0]];
    assert!(gain > 0.0);
    let expected = Array2::from_shape_fn(composed.dim(), |(row, column)| if row == column && row < INDUCTION_VOCAB { gain } else { 0.0 });
    assert_eq!(composed, expected);
}
