//! GREEN regression contract for issue #227: `GatedSAEDecoder` follows
//! canonical Gated-SAE Heaviside gating — `gate_i = 1` iff `(W_gate x)_i > 0`,
//! and 0 otherwise (including exact zero and all negative logits).
//!
//! The strict `if logit > 0.0` predicate lives at
//! `src/terms/gated_decoder.rs:59`. These tests guard against any regression
//! back to a non-strict comparison (e.g. `logit != 0.0`), which would wrongly
//! activate the gate for negative or exactly-zero logits.

use gam::terms::decoders::gated_decoder::GatedSAEDecoder;
use ndarray::{Array1, Array2, array};

#[test]
fn negative_logits_zero_the_output() {
    // W_gate = -I, x = [1,1,1] ⇒ every logit is strictly negative.
    let w_gate: Array2<f64> = -Array2::eye(3);
    let w_amp: Array2<f64> = Array2::eye(3);
    let decoder = GatedSAEDecoder::new(w_gate, w_amp).expect("construct");

    let x: Array1<f64> = array![1.0, 1.0, 1.0];
    let out = decoder.decode_row(x.view()).expect("decode");
    let expected: Array1<f64> = Array1::zeros(3);
    assert_eq!(out, expected, "negative gate logits must yield zero output");
}

#[test]
fn zero_logits_are_inactive() {
    // W_gate = 0 ⇒ every logit is exactly zero. Heaviside(0) := 0.
    let w_gate: Array2<f64> = Array2::zeros((2, 2));
    let w_amp: Array2<f64> = Array2::eye(2);
    let decoder = GatedSAEDecoder::new(w_gate, w_amp).expect("construct");

    let x: Array1<f64> = array![2.0, -3.0];
    let out = decoder.decode_row(x.view()).expect("decode");
    let expected: Array1<f64> = Array1::zeros(2);
    assert_eq!(out, expected, "zero gate logits must be inactive");
}

#[test]
fn positive_logits_pass_through() {
    let w_gate: Array2<f64> = Array2::eye(2);
    let w_amp: Array2<f64> = Array2::eye(2);
    let decoder = GatedSAEDecoder::new(w_gate, w_amp).expect("construct");

    let x: Array1<f64> = array![1.5, 2.5];
    let out = decoder.decode_row(x.view()).expect("decode");
    assert_eq!(out, x, "positive gate logits must pass x through W_amp");
}

#[test]
fn mixed_sign_logits_select_only_positive_rows() {
    // W_gate diag(1, -1): logits = (x_0, -x_1). For x = (1, 1): logits = (1, -1).
    let w_gate: Array2<f64> = array![[1.0, 0.0], [0.0, -1.0]];
    let w_amp: Array2<f64> = Array2::eye(2);
    let decoder = GatedSAEDecoder::new(w_gate, w_amp).expect("construct");

    let x: Array1<f64> = array![1.0, 1.0];
    let out = decoder.decode_row(x.view()).expect("decode");
    let expected: Array1<f64> = array![1.0, 0.0];
    assert_eq!(
        out, expected,
        "only coordinates with logit > 0 must contribute"
    );
}
