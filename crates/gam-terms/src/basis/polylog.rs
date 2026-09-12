//! Polylogarithm primitives on the unit interval.
//!
//! `dilog_unit` (Li₂) and `trilog_unit` (Li₃) evaluate the dilogarithm and
//! trilogarithm for real `z ∈ [0, 1]` by direct power series with early exit.
//! They are pure scalar math with no dependency on the rest of the basis
//! machinery, used by the closed-form Sobolev Wahba sphere kernels.
//!
//! Both use the direct power series on `z ∈ [0, 0.5]`, summed until a term no
//! longer changes the rounded partial sum, and a reflection on `(0.5, 1)`. The reflections
//! differ, because the two functions have different ones available:
//!
//! * `Li₂` uses `Li₂(z) = π²/6 − ln z·ln(1−z) − Li₂(1−z)`, which maps the slow
//!   regime back onto the fast one.
//! * `Li₃` has no such two-term reflection on `(0, 1)`. The Landen identity
//!   `Li₃(z) + Li₃(1−z) + Li₃(z/(z−1)) = ζ(3) + …` does not help, because
//!   `z/(z−1) ∈ (−∞, 0)` leaves the direct series' radius of convergence; and
//!   a previous Landen-shifted attempt (using `−(1−z)/z`) was outright wrong,
//!   with errors of order 1 at `z = 0.7..0.9`. `Li₃` therefore uses the
//!   expansion in `μ = ln z` (Wood 1992 §4; the `s → 3` limit of
//!   `Li_s(e^μ) = Γ(1−s)(−μ)^{s−1} + Σ_{k≥0} ζ(s−k) μ^k/k!`, where the head
//!   and the `k = 2` term both diverge and their sum does not).
//!
//! ## Why `Li₃` is not a raw series on `(0.5, 1)` (#2475 lane)
//!
//! It used to be: a direct series under a 5000-term cap. That cap is a hard
//! error floor near `z = 1`, because the tail of `Σ z^k/k³` there is `≈ K/2`
//! times its own last term, so the per-term early-exit test underestimates it
//! by a factor of thousands. Measured against a 50-digit reference, the old
//! path gave 2.9e-11 relative at `z = 0.999` — the module previously claimed
//! ≳13 digits there — degrading to a **1.7e-8 plateau** for `z ≳ 0.9999` once
//! the cap bound. That is not a corner: `Li₃(u)` with `u = (1−cos γ)/2` is the
//! Sobolev `m = 3` sphere kernel, and `u → 1` is the ANTIPODAL pair, which
//! farthest-point center selection actively seeks out. So the antipodal block
//! of the Gram matrix carried ~1.7e-8 while the rest of it carried ~1e-15.
//!
//! The `μ` expansion is uniformly ≤5.0e-16 relative over the whole branch,
//! needs 10 terms rather than 5000, and has no tail near `z = 1` at all — the
//! leading term is just `ζ(3)`. Verified against 50-digit `mpmath.polylog`
//! values in this module's tests.

/// `Σ_{k≥1} z^k / k^power` for `z ∈ (0, 0.5]`, summed until a term no longer
/// changes the rounded partial sum.
///
/// The terms decrease at least geometrically (ratio `≤ z ≤ 1/2`), so once a
/// term is below half an ulp of the partial sum every later term is too, and
/// the whole tail cannot change the result. The loop terminates because the
/// terms tend to zero and the partial sum is at least `z > 0`.
fn unit_polylog_series(z: f64, power: i32) -> f64 {
    let mut sum = 0.0_f64;
    let mut zk = z;
    let mut k = 1.0_f64;
    loop {
        let next = sum + zk / k.powi(power);
        if next == sum {
            return sum;
        }
        sum = next;
        zk *= z;
        k += 1.0;
    }
}

/// Coefficients `ζ(3−k) / k!` of the `μ = ln z` expansion of `Li₃`, for the
/// `k ≥ 3` at which they do not vanish.
///
/// `ζ(3−k)` is `ζ(0) = −1/2` at `k = 3` and `ζ(−n) = −B_{n+1}/(n+1)` for
/// `n = k−3 ≥ 1`; since `B_odd>1 = 0`, every `ζ(−2n)` with `n ≥ 1` is zero and
/// the odd `k ≥ 5` drop out entirely. Truncating after `k = 14` holds the whole
/// `z ∈ [0.5, 1)` branch to 5.0e-16 relative (measured), so no further term is
/// representable.
const TRILOG_MU_SERIES: [(i32, f64); 7] = [
    (3, -1.0 / 12.0),                      // ζ(0)/3!    = (−1/2)/6
    (4, -1.0 / 288.0),                     // ζ(−1)/4!   = (−1/12)/24
    (6, 1.0 / 86_400.0),                   // ζ(−3)/6!   = (1/120)/720
    (8, -1.0 / 10_160_640.0),              // ζ(−5)/8!   = (−1/252)/40320
    (10, 1.0 / 870_912_000.0),             // ζ(−7)/10!  = (1/240)/3628800
    (12, -1.0 / 63_228_211_200.0),         // ζ(−9)/12!  = (−1/132)/479001600
    (14, 691.0 / 2_855_960_819_712_000.0), // ζ(−11)/14! = (691/32760)/14!
];

/// Dilogarithm `Li₂(z) = Σ_{k≥1} z^k / k²` for real `z ∈ [0, 1]`.
///
/// Direct series for `z ≤ 0.5`; for `z ∈ (0.5, 1]` the reflection
/// `Li₂(z) = π²/6 − ln(z)·ln(1−z) − Li₂(1−z)` keeps the series in its
/// fast-converging regime. Returns `NaN` for non-finite input.
#[inline]
pub(crate) fn dilog_unit(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    let z = z.clamp(0.0, 1.0);
    if z == 0.0 {
        return 0.0;
    }
    if z >= 1.0 {
        return std::f64::consts::PI * std::f64::consts::PI / 6.0;
    }
    if z <= 0.5 {
        unit_polylog_series(z, 2)
    } else {
        let one_minus_z = 1.0 - z;
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        pi2_6 - z.ln() * one_minus_z.ln() - dilog_unit(one_minus_z)
    }
}

/// Trilogarithm `Li₃(z) = Σ_{k≥1} z^k / k³` for real `z ∈ [0, 1]`.
///
/// Direct series for `z ≤ 0.5`; for `z ∈ (0.5, 1)` the expansion in
/// `μ = ln z`, which is the reflection the direct series needs and Landen
/// cannot supply (see the module docs):
///
/// ```text
/// Li₃(e^μ) = ζ(3) + ζ(2)·μ + (μ²/2)·(3/2 − ln(−μ)) + Σ_{k≥3} ζ(3−k)·μ^k/k!
/// ```
///
/// The `μ²·ln(−μ)` term is the `s = 3` degeneration of the general
/// `Γ(1−s)(−μ)^{s−1}` head of `Li_s(e^μ)`. Every term is bounded as `μ → 0⁻`,
/// so the accuracy near `z = 1` is set by `ζ(3)` rather than by a truncation
/// tail — which is the whole point, since `z → 1` is the ANTIPODAL entry
/// (`u = (1−cos γ)/2 → 1`) of every Sobolev `m = 3` sphere Gram matrix.
///
/// Returns `NaN` for non-finite input, `ζ(3)` at `z = 1`.
#[inline]
pub(crate) fn trilog_unit(z: f64) -> f64 {
    const ZETA3: f64 = 1.2020569031595942853997381615114499907649862923404988817922;
    const ZETA2: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    if !z.is_finite() {
        return f64::NAN;
    }
    let z = z.clamp(0.0, 1.0);
    if z == 0.0 {
        return 0.0;
    }
    if z >= 1.0 {
        return ZETA3;
    }
    if z > 0.5 {
        // `μ = ln z ∈ (−ln 2, 0)`; `μ → 0⁻` as `z → 1⁻`.
        let mu = z.ln();
        if mu == 0.0 {
            // `z < 1` but `ln z` underflowed to `-0.0`: the whole expansion
            // past `ζ(3)` is then below one ulp of `ζ(3)` anyway.
            return ZETA3;
        }
        let mut sum = ZETA3 + ZETA2 * mu + 0.5 * mu * mu * (1.5 - (-mu).ln());
        for (k, coeff) in TRILOG_MU_SERIES {
            sum += coeff * mu.powi(k);
        }
        return sum;
    }
    unit_polylog_series(z, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── dilog_unit ────────────────────────────────────────────────────────────

    #[test]
    fn dilog_unit_zero_is_zero() {
        assert_eq!(dilog_unit(0.0), 0.0);
    }

    #[test]
    fn dilog_unit_at_one_is_pi_squared_over_six() {
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((dilog_unit(1.0) - expected).abs() < 1e-15);
    }

    #[test]
    fn dilog_unit_nan_input_returns_nan() {
        assert!(dilog_unit(f64::NAN).is_nan());
    }

    #[test]
    fn dilog_unit_at_half_matches_reflection_identity() {
        // Li₂(1/2) = π²/12 − (ln 2)²/2  (reflection at z = 1/2)
        let expected = std::f64::consts::PI.powi(2) / 12.0 - (2.0_f64).ln().powi(2) / 2.0;
        assert!((dilog_unit(0.5) - expected).abs() < 1e-14);
    }

    #[test]
    fn dilog_unit_is_positive_on_open_unit_interval() {
        for z in [0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!(dilog_unit(z) > 0.0, "dilog_unit({z}) should be positive");
        }
    }

    #[test]
    fn dilog_unit_clamps_below_zero() {
        assert_eq!(dilog_unit(-1.0), dilog_unit(0.0));
    }

    #[test]
    fn dilog_unit_clamps_above_one() {
        assert!((dilog_unit(2.0) - dilog_unit(1.0)).abs() < 1e-15);
    }

    // ── trilog_unit ───────────────────────────────────────────────────────────

    #[test]
    fn trilog_unit_zero_is_zero() {
        assert_eq!(trilog_unit(0.0), 0.0);
    }

    #[test]
    fn trilog_unit_at_one_is_zeta3() {
        const ZETA3: f64 = 1.2020569031595942853997381615114499907649862923404988817922;
        assert!((trilog_unit(1.0) - ZETA3).abs() < 1e-14);
    }

    #[test]
    fn trilog_unit_nan_input_returns_nan() {
        assert!(trilog_unit(f64::NAN).is_nan());
    }

    #[test]
    fn trilog_unit_at_half_matches_known_value() {
        // Li₃(1/2) = 7ζ(3)/8 − π²·ln2/12 + (ln 2)³/6 ≈ 0.5372131936080403
        // (the previous literal 0.5372131936432659 was a stale, mistyped oracle;
        //  it disagreed with the true closed form by ~3.5e-11. The series
        //  Σ z^k/k³ this code computes matches the closed form to machine eps.)
        let expected = 0.5372131936080403_f64;
        assert!((trilog_unit(0.5) - expected).abs() < 1e-13);
    }

    #[test]
    fn trilog_unit_is_positive_on_open_unit_interval() {
        for z in [0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!(trilog_unit(z) > 0.0, "trilog_unit({z}) should be positive");
        }
    }

    #[test]
    fn trilog_unit_clamps_below_zero() {
        assert_eq!(trilog_unit(-0.5), trilog_unit(0.0));
    }

    #[test]
    fn trilog_unit_clamps_above_one() {
        assert!((trilog_unit(2.0) - trilog_unit(1.0)).abs() < 1e-14);
    }

    /// `Li₃(z)` from `mpmath.polylog(3, z)` at 40 decimal digits, rounded to
    /// `f64`. Spans both branches and the antipodal approach `z → 1⁻`.
    const TRILOG_REFERENCE: [(f64, f64); 16] = [
        (0.05, 5.031_722_986_057_436_7e-2),
        (0.125, 1.270_295_409_793_486_0e-1),
        (0.25, 2.584_613_957_965_732_9e-1),
        (0.5, 5.372_131_936_080_402_1e-1),
        // First point on the `μ`-expansion side of the branch cut.
        (0.500_000_000_000_000_1, 5.372_131_936_080_403_2e-1),
        (0.6, 6.560_025_136_329_806_8e-1),
        (0.75, 8.444_258_088_622_044_2e-1),
        (0.9, 1.049_658_950_186_439_9),
        (0.95, 1.123_574_584_279_198_9),
        (0.99, 1.185_832_933_645_036_8),
        (0.999, 1.200_415_353_995_464_3),
        (0.9999, 1.201_892_455_084_581_5),
        (0.999_999, 1.202_055_258_232_362_7),
        (0.999_999_999, 1.202_056_901_514_660_3),
        (0.999_999_999_999, 1.202_056_903_157_949_3),
        (0.999_999_999_999_999, 1.202_056_903_159_592_7),
    ];

    #[test]
    fn trilog_unit_matches_high_precision_reference_across_both_branches() {
        // The old 5000-term direct series plateaued at 1.7e-8 relative for
        // `z ≳ 0.9999` (the antipodal end of the Sobolev m=3 sphere kernel).
        // The `μ` expansion is uniform to a few ulps across the whole range.
        let mut worst = 0.0_f64;
        let mut worst_z = f64::NAN;
        println!(
            "\n{:>22} {:>24} {:>24} {:>10}",
            "z", "reference Li3", "trilog_unit", "rel"
        );
        for (z, want) in TRILOG_REFERENCE {
            let got = trilog_unit(z);
            let rel = (got - want).abs() / want.abs();
            if rel > worst {
                worst = rel;
                worst_z = z;
            }
            println!("{z:>22.17} {want:>24.17e} {got:>24.17e} {rel:>10.2e}");
        }
        println!("\n  worst: {worst:.3e} at z = {worst_z}\n");
        assert!(
            worst < 4.0 * f64::EPSILON,
            "trilog_unit is off by {worst:.3e} relative at z = {worst_z}; the \
             mu expansion should hold the whole interval to a few ulps"
        );
    }

    #[test]
    fn trilog_unit_is_continuous_across_the_branch_switch() {
        // `z = 0.5` takes the direct series, the next representable value takes
        // the `μ` expansion. Two different algorithms must agree to roundoff or
        // the Gram matrix inherits a step.
        let below = trilog_unit(0.5);
        let above = trilog_unit(f64::from_bits(0.5_f64.to_bits() + 1));
        let jump = (above - below).abs() / below;
        println!("\n  branch switch at z=0.5: {below:.17e} -> {above:.17e}, rel step {jump:.2e}\n");
        assert!(
            jump < 4.0 * f64::EPSILON,
            "branch switch at z = 0.5 steps by {jump:.3e} relative"
        );
    }

    #[test]
    fn trilog_unit_is_monotone_increasing_up_to_zeta3() {
        // `Li₃` is strictly increasing on [0,1]. A truncation-tail artefact
        // near `z = 1` shows up here as a non-monotone approach to ζ(3).
        const ZETA3: f64 = 1.202_056_903_159_594_3;
        let mut prev = 0.0_f64;
        for i in 0..=2000 {
            let z = i as f64 / 2000.0;
            let v = trilog_unit(z);
            assert!(v >= prev, "trilog_unit decreased at z = {z}: {prev} -> {v}");
            assert!(
                v <= ZETA3 + 4.0 * f64::EPSILON,
                "trilog_unit({z}) = {v} > zeta(3)"
            );
            prev = v;
        }
        // And the approach to ζ(3) is monotone right up to the boundary. This
        // is the assertion the old 5000-term series could not make: its tail
        // error grew as `z → 1`, so `Li₃` crept back DOWN toward the plateau.
        let mut prev_tail = 0.0_f64;
        for k in 1..=15 {
            let z = 1.0 - 10.0_f64.powi(-k); // 0.9, 0.99, … increasing to 1⁻
            let v = trilog_unit(z);
            assert!(
                v >= prev_tail,
                "trilog_unit decreased approaching z = 1 at 1 - 1e-{k}: {prev_tail} -> {v}"
            );
            assert!(v <= ZETA3, "trilog_unit(1 - 1e-{k}) = {v} exceeds zeta(3)");
            prev_tail = v;
        }
        // The approach is not just monotone, it has the right SLOPE:
        // `Li₃'(z) = Li₂(z)/z`, so `Li₃(1−δ) = ζ(3) − ζ(2)·δ + O(δ² ln δ)`.
        // δ = 1e-12 keeps the difference ~1.6e-12, four orders above the ulp of
        // ζ(3), so the gap is resolved to ~1e-4 despite the cancellation. This
        // is the assertion that pins the boundary behaviour: the old series
        // approached a plateau 1.7e-8 BELOW ζ(3), which fails it by six orders.
        const ZETA2: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        let delta = 1e-12_f64;
        let gap = ZETA3 - trilog_unit(1.0 - delta);
        let predicted = ZETA2 * delta;
        let rel = (gap - predicted).abs() / predicted;
        println!(
            "\n  zeta(3) - Li3(1-1e-12) = {gap:.6e}, predicted zeta(2)*delta = {predicted:.6e}, rel {rel:.2e}\n"
        );
        assert!(
            rel < 1e-3,
            "the approach to zeta(3) has the wrong slope: gap {gap:.6e} vs \
             zeta(2)*delta {predicted:.6e} (rel {rel:.3e})"
        );
    }

    #[test]
    fn dilog_unit_matches_high_precision_reference() {
        // `mpmath.polylog(2, z)` at 40 dps, rounded to f64. The reflection
        // branch was already sound; this pins it alongside Li₃.
        const REFERENCE: [(f64, f64); 8] = [
            (0.05, 5.063_929_246_449_602_7e-2),
            (0.25, 2.676_526_390_827_326_2e-1),
            (0.5, 5.822_405_264_650_124_5e-1),
            (0.75, 9.784_693_929_303_061_0e-1),
            (0.9, 1.299_714_723_004_958_8),
            (0.99, 1.588_625_448_076_375_3),
            (0.999, 1.637_022_605_276_117_7),
            (0.999_999_999, 1.644_934_045_124_961_2),
        ];
        let mut worst = 0.0_f64;
        for (z, want) in REFERENCE {
            let rel = (dilog_unit(z) - want).abs() / want.abs();
            worst = worst.max(rel);
        }
        println!("\n  dilog_unit worst relative error: {worst:.3e}\n");
        assert!(
            worst < 4.0 * f64::EPSILON,
            "dilog_unit is off by {worst:.3e} relative"
        );
    }
}
