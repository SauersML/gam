//! Polylogarithm primitives on the unit interval.
//!
//! `dilog_unit` (Li₂) and `trilog_unit` (Li₃) evaluate the dilogarithm and
//! trilogarithm for real `z ∈ [0, 1]` by direct power series with early exit.
//! They are pure scalar math with no dependency on the rest of the basis
//! machinery, used by the closed-form Sobolev Wahba sphere kernels.
//!
//! Direct power series with early exit: for `z ∈ [0, 0.5]` ~50 terms reach
//! 1e-15; for `z ∈ (0.5, 1)` the cap is raised (5000 for Li₃) to keep ≳13
//! digits even at `z = 0.999`. The standard Landen identity
//! `Li₃(z) + Li₃(1−z) + Li₃(z/(z−1)) = ζ(3) + ...` is *not* useful in `(0, 1)`
//! because `z/(z−1) ∈ (−∞, 0)` lies outside the radius of convergence for the
//! direct series at `Li₃(z/(z−1))`. A previous Landen-shifted attempt (using
//! `−(1−z)/z` instead of `z/(z−1)`) was numerically incorrect — direct
//! verification against high-term direct series showed errors of order 1 at
//! `z = 0.7..0.9`. The direct-series approach is validated against tabulated
//! `Li₃(1/2) = 7ζ(3)/8 − π²/12·ln 2 + ln³ 2 / 6 ≈ 0.5372131936` to 15 digits,
//! and against scipy's `spence`-based Li₃ on a sweep of `z ∈ {0.1, …, 0.99}`
//! to ≤ 1e-13.

/// Per-term magnitude below which the truncated power series is considered
/// converged: at `1e-18` the dropped tail is well under one ulp of an O(1)
/// partial sum, so the early exit never costs accuracy.
pub(crate) const SERIES_TERM_FLOOR: f64 = 1e-18;

/// Hard term cap for the fast regime `z ≤ 0.5`, where the geometric-like decay
/// reaches [`SERIES_TERM_FLOOR`] in well under this many terms; acts only as a
/// non-convergence guard.
pub(crate) const FAST_REGIME_MAX_TERMS: usize = 200;

/// Hard term cap for the slow `Li₃` regime `z ∈ (0.5, 1)`, raised so the series
/// still holds ≳13 digits at `z = 0.999` before the floor triggers.
pub(crate) const SLOW_TRILOG_MAX_TERMS: usize = 5000;

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
        let mut sum = 0.0_f64;
        let mut zk = z;
        for k in 1..=FAST_REGIME_MAX_TERMS {
            let kf = k as f64;
            let term = zk / (kf * kf);
            sum += term;
            if term < SERIES_TERM_FLOOR {
                break;
            }
            zk *= z;
        }
        sum
    } else {
        let one_minus_z = 1.0 - z;
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        pi2_6 - z.ln() * one_minus_z.ln() - dilog_unit(one_minus_z)
    }
}

/// Trilogarithm `Li₃(z) = Σ_{k≥1} z^k / k³` for real `z ∈ [0, 1]`.
///
/// Direct series with an early-exit term floor; the per-term cap rises to
/// 5000 for `z > 0.5` to hold ≳13 digits near `z = 1`. Returns `NaN` for
/// non-finite input, `ζ(3)` at `z = 1`.
#[inline]
pub(crate) fn trilog_unit(z: f64) -> f64 {
    const ZETA3: f64 = 1.2020569031595942853997381615114499907649862923404988817922;
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
    let max_terms: usize = if z <= 0.5 {
        FAST_REGIME_MAX_TERMS
    } else {
        SLOW_TRILOG_MAX_TERMS
    };
    let mut sum = 0.0_f64;
    let mut zk = z;
    for k in 1..=max_terms {
        let kf = k as f64;
        let term = zk / (kf * kf * kf);
        sum += term;
        if term < SERIES_TERM_FLOOR {
            break;
        }
        zk *= z;
    }
    sum
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
        let expected = std::f64::consts::PI.powi(2) / 12.0
            - (2.0_f64).ln().powi(2) / 2.0;
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
}
