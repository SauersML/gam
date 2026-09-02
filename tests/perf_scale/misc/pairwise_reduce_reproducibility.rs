//! Reproducibility, association-stability, and accuracy tests for the
//! deterministic pairwise-tree reduction primitive (#973).
//!
//! All inputs are fixed and deterministic — no clock, no randomness.

use gam::linalg::pairwise_reduce::{BASE_CHUNK, pairwise_reduce, pairwise_sum};

/// A deterministic, fixed sequence with a wide dynamic range: one large value
/// followed by many tiny values. Naively summed left to right (large first),
/// the tiny values are repeatedly swamped and lost to rounding; the exact sum
/// is `BIG + n_small * SMALL`. This is the classic case pairwise summation is
/// built to improve.
fn ill_conditioned_input() -> Vec<f64> {
    const BIG: f64 = 1.0e16;
    const SMALL: f64 = 1.0;
    // Comfortably exceeds several base chunks so the full tree is exercised.
    let n_small = 5 * BASE_CHUNK + 37;
    let mut v = Vec::with_capacity(n_small + 1);
    v.push(BIG);
    for _ in 0..n_small {
        v.push(SMALL);
    }
    v
}

/// Naive sequential left-to-right fold — the baseline pairwise summation must
/// beat on accuracy.
fn naive_sum(xs: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &x in xs {
        acc += x;
    }
    acc
}

/// A fixed, varied input used for the reproducibility / association tests.
/// Mixed magnitudes and signs, length deliberately not a multiple of the base
/// chunk so the non-power-of-two tail path is exercised.
fn varied_input() -> Vec<f64> {
    let n = 4 * BASE_CHUNK + 53;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        // Deterministic, spread across many magnitudes and both signs.
        let k = i as f64;
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mag = ((i % 17) as f64 - 8.0) * 3.0; // exponent in [-24, 24]
        v.push(sign * (1.0 + k * 0.5) * 2.0_f64.powf(mag));
    }
    v
}

#[test]
fn bit_reproducible_same_slice_twice() {
    let xs = varied_input();
    let a = pairwise_sum(&xs);
    let b = pairwise_sum(&xs);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "pairwise_sum must be bit-identical across repeated calls on the same slice"
    );

    // Same guarantee for the generic monoid-style entry point.
    let ga = pairwise_reduce(&xs, |p, q| p + q, 0.0);
    let gb = pairwise_reduce(&xs, |p, q| p + q, 0.0);
    assert_eq!(
        ga.to_bits(),
        gb.to_bits(),
        "pairwise_reduce must be bit-identical across repeated calls"
    );
    assert_eq!(
        a.to_bits(),
        ga.to_bits(),
        "pairwise_sum and pairwise_reduce(+) must agree bit-for-bit"
    );
}

#[test]
fn pairwise_more_accurate_than_naive_on_ill_conditioned_input() {
    let xs = ill_conditioned_input();

    // Exact reference: BIG + n_small * SMALL. The true mathematical sum is the
    // integer 10^16 + n_small. Note this exact value is NOT representable in
    // f64 (1e16 > 2^53, so the ULP there is 2.0 and the running magnitude is at
    // the spike), which is precisely why naive accumulation is lossy. We carry
    // the reference as the exact integer and measure each method's error against
    // it in f64 arithmetic.
    let n_small = xs.len() - 1;
    let exact = 1.0e16_f64 + n_small as f64; // closest-f64 reference value

    let pair = pairwise_sum(&xs);
    let naive = naive_sum(&xs);

    let err_pair = (pair - exact).abs();
    let err_naive = (naive - exact).abs();

    // The naive large-first fold accumulates the whole array on top of the
    // spike: every tiny value is added to a running sum already at ~1e16 (ULP
    // 2.0), so essentially all of the small mass is lost. Its error is on the
    // order of the total small mass.
    assert!(
        err_naive > 0.5 * n_small as f64,
        "test setup invalid: naive fold should lose most of the small mass \
         (err_naive={err_naive}, n_small={n_small})"
    );

    // Pairwise summation only co-locates the spike with the small values inside
    // a single base block of BASE_CHUNK elements; every other base block sums
    // pure small values exactly and contributes its full mass across the tree.
    // The only loss is within that one spike-containing base block, bounding the
    // error well below BASE_CHUNK and far below the naive whole-array loss.
    assert!(
        err_pair < BASE_CHUNK as f64,
        "pairwise error ({err_pair}) must be bounded by the single spike-containing \
         base block (< BASE_CHUNK = {BASE_CHUNK})"
    );

    // The headline guarantee: pairwise is strictly — and here substantially —
    // more accurate than the naive sequential fold on ill-conditioned input.
    assert!(
        err_pair < err_naive,
        "pairwise sum (err={err_pair}) must be strictly more accurate than naive (err={err_naive})"
    );
}
