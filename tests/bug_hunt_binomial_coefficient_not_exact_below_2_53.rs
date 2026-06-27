//! Bug hunt: `binomial_coefficient_f64` returns non-exact (off-by-one, and
//! even non-integer) values for binomial coefficients that fit exactly in an
//! `f64`, violating its own documented contract.
//!
//! `crates/gam-math/src/special.rs` documents the function as:
//!
//!   "Numerically stable `C(n,k) = n! / (k!·(n−k)!)` as `f64`. … Returns `0.0`
//!    for `k > n` and **exact integer results within `2^53`**."
//!
//! Every integer up to `2^53 = 9_007_199_254_740_992` is representable exactly
//! as an `f64`, so the contract is well-posed: whenever the true `C(n,k)` is at
//! or below `2^53`, the returned `f64` must equal it exactly.
//!
//! The implementation evaluates the multiplicative recurrence
//!   `out *= (n − j) / (j + 1)`   for `j = 0 .. min(k, n−k)`
//! in `f64`. Although every partial product is mathematically an integer
//! binomial coefficient, the intermediate *divisions* are not exact in binary
//! floating point, so rounding accumulates and the final value drifts off the
//! true integer. Observed failures, all far below `2^53`:
//!
//!   * C(54, 24) = 1_402_659_561_581_460  ->  returns 1_402_659_561_581_459.0  (off by 1)
//!   * C(55, 25) = 3_085_851_035_479_212  ->  returns 3_085_851_035_479_210.5  (off by 1.5, not even an integer)
//!   * C(56, 24) = 4_355_031_703_297_275  ->  returns 4_355_031_703_297_273.0  (off by 2)
//!
//! The exact reference is computed here with the same recurrence in `u128`
//! (where each `num * (n−j) / (j+1)` step is exact because the running product
//! is always divisible by `j+1`). The check sweeps every `(n, k)` whose true
//! value is at or below `2^53` and asserts the `f64` result is bit-equal.
//!
//! This is a contract bug in a shared numeric primitive (the function backs the
//! binomial log-likelihood normalizer re-exported as
//! `gam::inference::probability::binomial_coefficient_f64`, plus the spline
//! closed-form-penalty and Matérn-kernel combinatorics). When the
//! implementation is made exact within `2^53` (e.g. an integer recurrence, or
//! computing in `u128`/`u64` while the result fits), this test passes with no
//! further edits.

use gam::inference::probability::binomial_coefficient_f64;

const TWO_POW_53: u128 = 9_007_199_254_740_992;

/// Exact `C(n, k)` in `u128`. The running product is always divisible by the
/// next denominator, so each integer division is exact.
fn exact_binomial(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    for j in 0..k {
        num = num * u128::from(n - j) / u128::from(j + 1);
    }
    num
}

#[test]
fn binomial_coefficient_f64_is_exact_for_values_at_or_below_2_pow_53() {
    let mut offenders: Vec<(u64, u64, u128, f64)> = Vec::new();
    for n in 0u64..=80 {
        for k in 0u64..=n {
            let exact = exact_binomial(n, k);
            if exact > TWO_POW_53 {
                // Beyond the exactly-representable range; the contract does not
                // apply here, so do not test these.
                continue;
            }
            let got = binomial_coefficient_f64(n as usize, k as usize);
            if got != exact as f64 {
                offenders.push((n, k, exact, got));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "binomial_coefficient_f64 is not exact for {} coefficient(s) at or below 2^53 \
         (contract: 'exact integer results within 2^53'). First few: {:?}",
        offenders.len(),
        &offenders[..offenders.len().min(5)]
    );
}
