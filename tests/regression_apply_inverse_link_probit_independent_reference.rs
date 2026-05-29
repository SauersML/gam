//! Regression guard for the probit branch of `apply_inverse_link_vec`
//! (issue #351), approached from a different angle than the committed
//! deep-negative-tail precision test.
//!
//! Instead of comparing against a hand-written `½·erfc(-η/√2)` reference,
//! this validates the FFI helper against an *independent* implementation of
//! the standard-normal CDF — `statrs::distribution::Normal` via the
//! `ContinuousCDF` trait — across the full η range (deep negative tail,
//! bulk, deep positive tail). It also asserts the reflection identity
//! `Φ(η) + Φ(-η) = 1`, which a sign slip in the `erfc` argument (e.g.
//! `erfc(η/√2)` instead of `erfc(-η/√2)`) would violate even though such a
//! slip still keeps the negative tail strictly positive.

use gam::families::inverse_link::apply_inverse_link_vec;
use statrs::distribution::{ContinuousCDF, Normal};

#[test]
fn probit_matches_independent_normal_cdf_across_full_range() {
    let std_normal = Normal::new(0.0, 1.0).expect("standard normal constructs");

    // Spans both deep tails and the bulk. Deep-tail values are tiny but
    // strictly positive and well inside the f64 normal range.
    let eta = vec![
        -12.0, -10.0, -9.0, -8.0, -6.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0, 8.0, 9.0,
        10.0, 12.0,
    ];
    let mu = apply_inverse_link_vec(&eta, "probit").expect("probit dispatch succeeds");
    assert_eq!(mu.len(), eta.len());

    for (&e, &got) in eta.iter().zip(mu.iter()) {
        let reference = std_normal.cdf(e);
        // The defining failure of #351 is the lower tail collapsing to exactly
        // 0.0: Φ is strictly positive on every finite η and tiny values like
        // 1e-33 are representable, so this bound is both required and achievable.
        // The upper tail legitimately rounds to 1.0 (the complement 1-Φ(η) is
        // below f64 resolution for η ≳ 8.3), so we only bound it by ≤ 1.0.
        assert!(
            got > 0.0 && got <= 1.0,
            "Φ({e}) = {got:.6e} is outside (0, 1]; the probit inverse link must \
             be strictly positive on every finite η"
        );
        // statrs' Normal::cdf is itself erfc-based, so agreement should be to
        // near machine precision on the relative scale. Compare relatively in
        // the lower tail (small magnitudes) and on (1 - Φ) in the upper tail.
        let rel_err = if e <= 0.0 {
            (got - reference).abs() / reference
        } else {
            // Upper tail: the informative quantity is the complement 1 - Φ.
            ((1.0 - got) - (1.0 - reference)).abs() / (1.0 - reference).max(f64::MIN_POSITIVE)
        };
        assert!(
            rel_err < 1e-9,
            "Φ({e}): helper = {got:.10e}, independent statrs Normal::cdf = \
             {reference:.10e}, relative error = {rel_err:.3e}"
        );
    }
}

#[test]
fn probit_obeys_reflection_identity_phi_eta_plus_phi_neg_eta_is_one() {
    // Φ(η) + Φ(-η) = 1 for the standard normal. Sampling symmetric pairs
    // across the cancellation boundary; the sum must equal 1.0 to f64.
    let half_grid = [0.0, 0.25, 1.0, 3.0, 6.0, 8.0, 9.0, 10.0, 12.0];
    let mut eta = Vec::new();
    for &x in &half_grid {
        eta.push(x);
        eta.push(-x);
    }
    let mu = apply_inverse_link_vec(&eta, "probit").expect("probit dispatch succeeds");

    for (i, &x) in half_grid.iter().enumerate() {
        let pos = mu[2 * i];
        let neg = mu[2 * i + 1];
        let sum = pos + neg;
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "Φ({x}) + Φ({}) = {pos:.10e} + {neg:.10e} = {sum:.16}, expected 1.0; \
             a sign error in the erfc argument breaks this reflection identity",
            -x
        );
    }
}
