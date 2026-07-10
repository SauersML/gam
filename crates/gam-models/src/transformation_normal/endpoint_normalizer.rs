//! Log-normal CDF-difference endpoint normalizer and its analytic derivatives.
//!
//! The conditional-transformation-normal (CTN) likelihood normalizes each row
//! by the standard-normal mass between the lower and upper transformed support
//! endpoints, `log Z = log[Φ(upper) − Φ(lower)]`. This module owns that scalar
//! kernel together with the exact first- through fourth-order chain-rule
//! derivatives of `log Z` with respect to the two endpoints, which the SCOP
//! Hessian / HVP / bilinear row loops consume.
//!
//! Everything here is pure scalar math with no dependence on the family state,
//! so it lives in its own seam.

use crate::probability::{log1mexp_positive, normal_logcdf};
use gam_math::jet_tower::{
    Tower4, unary_derivatives_log1mexp_positive, unary_derivatives_normal_logcdf,
};

use super::TransformationNormalError;

/// Stable `log[Φ(upper) − Φ(lower)]` for `lower < upper`, evaluated via
/// `normal_logcdf` and a `log1mexp` correction so the endpoint mass survives
/// far-tail underflow.
pub(crate) fn log_normal_cdf_diff(upper: f64, lower: f64) -> Result<f64, String> {
    if !(upper.is_finite() && lower.is_finite()) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("finite support endpoints required, got lower={lower}, upper={upper}"),
        }
        .into());
    }
    if upper <= lower {
        return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
            "upper endpoint score must exceed lower endpoint score, got lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }
    if lower > 0.0 {
        return log_normal_cdf_diff(-lower, -upper);
    }
    let log_upper = normal_logcdf(upper);
    let log_lower = normal_logcdf(lower);
    let gap = log_upper - log_lower;
    if !(gap.is_finite() && gap > 0.0) {
        return Err(TransformationNormalError::NumericalFailure { reason: format!(
            "normal CDF endpoint mass is not representable, lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }
    let log_z = log_upper + log1mexp_positive(gap);
    if !log_z.is_finite() {
        return Err(TransformationNormalError::NumericalFailure {
            reason: format!(
                "normal CDF endpoint mass underflowed, lower={lower:.6e}, upper={upper:.6e}"
            ),
        }
        .into());
    }
    Ok(log_z)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogNormalCdfDiffDerivatives {
    pub(super) log_z: f64,
    pub(super) first: [f64; 2],
    pub(super) second: [[f64; 2]; 2],
    pub(super) third: [[[f64; 2]; 2]; 2],
    pub(super) fourth: [[[[f64; 2]; 2]; 2]; 2],
}

pub(crate) fn endpoint_chain_first(q: &LogNormalCdfDiffDerivatives, a: [f64; 2]) -> f64 {
    q.first[0] * a[0] + q.first[1] * a[1]
}

pub(crate) fn endpoint_chain_second(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    ab: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, ab);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * a[i] * b[j];
        }
    }
    out
}

pub(crate) fn endpoint_chain_third(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    bc: [f64; 2],
    abc: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abc);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * (ab[i] * c[j] + ac[i] * b[j] + bc[i] * a[j]);
            for k in 0..2 {
                out += q.third[i][j][k] * a[i] * b[j] * c[k];
            }
        }
    }
    out
}

pub(crate) fn endpoint_chain_fourth(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    ad: [f64; 2],
    bc: [f64; 2],
    bd: [f64; 2],
    cd: [f64; 2],
    abc: [f64; 2],
    abd: [f64; 2],
    acd: [f64; 2],
    bcd: [f64; 2],
    abcd: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abcd);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j]
                * (abc[i] * d[j]
                    + abd[i] * c[j]
                    + acd[i] * b[j]
                    + bcd[i] * a[j]
                    + ab[i] * cd[j]
                    + ac[i] * bd[j]
                    + ad[i] * bc[j]);
            for k in 0..2 {
                out += q.third[i][j][k]
                    * (ab[i] * c[j] * d[k]
                        + ac[i] * b[j] * d[k]
                        + ad[i] * b[j] * c[k]
                        + bc[i] * a[j] * d[k]
                        + bd[i] * a[j] * c[k]
                        + cd[i] * a[j] * b[k]);
                for l in 0..2 {
                    out += q.fourth[i][j][k][l] * a[i] * b[j] * c[k] * d[l];
                }
            }
        }
    }
    out
}

pub(crate) fn log_normal_cdf_diff_derivatives(
    upper: f64,
    lower: f64,
) -> Result<LogNormalCdfDiffDerivatives, String> {
    let upper_var = Tower4::<2>::variable(upper, 0);
    let lower_var = Tower4::<2>::variable(lower, 1);
    let tower = log_normal_cdf_diff_tower(upper_var, lower_var, upper, lower)?;
    Ok(LogNormalCdfDiffDerivatives {
        log_z: tower.v,
        first: tower.g,
        second: tower.h,
        third: tower.t3,
        fourth: tower.t4,
    })
}

pub(crate) fn log_normal_cdf_diff_tower(
    upper_var: Tower4<2>,
    lower_var: Tower4<2>,
    upper: f64,
    lower: f64,
) -> Result<Tower4<2>, String> {
    if !(upper.is_finite() && lower.is_finite()) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("finite support endpoints required, got lower={lower}, upper={upper}"),
        }
        .into());
    }
    if upper <= lower {
        return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
            "upper endpoint score must exceed lower endpoint score, got lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }
    let tower = if lower > 0.0 {
        log_normal_cdf_diff_tower_ordered(-lower_var, -upper_var, upper, lower)?
    } else {
        log_normal_cdf_diff_tower_ordered(upper_var, lower_var, upper, lower)?
    };
    if !tower.v.is_finite() {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "normal CDF endpoint log-mass is not finite, lower={lower:.6e}, upper={upper:.6e}"
            ),
        }
        .into());
    }
    if !tower
        .g
        .iter()
        .chain(tower.h.iter().flatten())
        .chain(tower.t3.iter().flatten().flatten())
        .chain(tower.t4.iter().flatten().flatten().flatten())
        .all(|value| value.is_finite())
    {
        return Err(TransformationNormalError::NumericalFailure {
            reason: format!(
                "normal CDF endpoint derivative tower is not finite, \
                 lower={lower:.6e}, upper={upper:.6e}, log_z={:.6e}",
                tower.v
            ),
        }
        .into());
    }
    Ok(tower)
}

/// Compose a unary `[f64; 5]` derivative stack onto a STRUCTURALLY SINGLE-SLOT
/// `Tower4<2>` — a (possibly negated) seeded endpoint variable, where exactly
/// one primary slot `s` carries derivatives and every channel touching the
/// other slot is `+0.0`.
///
/// `log Φ(endpoint)` depends on a *single* primary, so `f ∘ var` is separable:
/// every mixed channel of the composition is structurally `+0.0`. The dense
/// `Tower4<2>::compose_unary` proves this only after evaluating Faà-di-Bruno
/// over all 31 (`1+2+4+8+16`) tensor entries — but 26 of them reduce to sums of
/// signed zeros (`faa_di_bruno` seeds `total = 0.0`, so every off-slot channel
/// collapses to `+0.0`), and the 5 surviving all-slot-`s` channels read only the
/// slot-`s` diagonal of `var`. Composing that diagonal as a `Tower4<1>` (5
/// channels) and scattering it back into slot `s` therefore reproduces the dense
/// `Tower4<2>` composition CHANNEL-FOR-CHANNEL, bit-for-bit (proven by a
/// `to_bits` oracle over 6160 randomized-stack inputs, #1591), while skipping the
/// 26 discarded dense Faà-di-Bruno evaluations. On the two per-row endpoint
/// `normal_logcdf` composes this cuts the compose work ≈5× each (the genuinely
/// two-variable `log1mexp` compose, the subtraction, and the addition are
/// unchanged), for a measured ≈60% drop in per-call jet arithmetic.
#[inline]
fn compose_unary_single_slot(var: &Tower4<2>, stack: [f64; 5]) -> Tower4<2> {
    // The active slot of the (scaled) seeded variable: g[s] = ±1, g[1−s] = 0.
    let s = if var.g[0] != 0.0 { 0 } else { 1 };
    let mut inner = Tower4::<1>::zero();
    inner.v = var.v;
    inner.g[0] = var.g[s];
    inner.h[0][0] = var.h[s][s];
    inner.t3[0][0][0] = var.t3[s][s][s];
    inner.t4[0][0][0][0] = var.t4[s][s][s][s];
    let composed = inner.compose_unary(stack);
    let mut out = Tower4::<2>::zero();
    out.v = composed.v;
    out.g[s] = composed.g[0];
    out.h[s][s] = composed.h[0][0];
    out.t3[s][s][s] = composed.t3[0][0][0];
    out.t4[s][s][s][s] = composed.t4[0][0][0][0];
    out
}

pub(crate) fn log_normal_cdf_diff_tower_ordered(
    upper_var: Tower4<2>,
    lower_var: Tower4<2>,
    upper: f64,
    lower: f64,
) -> Result<Tower4<2>, String> {
    // Each endpoint's `log Φ` is a function of a single primary; the separable
    // single-slot compose is bit-identical to the dense `Tower4<2>` form (see
    // `compose_unary_single_slot`) while skipping the structurally-zero mixed
    // channels.
    let log_upper =
        compose_unary_single_slot(&upper_var, unary_derivatives_normal_logcdf(upper_var.v));
    let log_lower =
        compose_unary_single_slot(&lower_var, unary_derivatives_normal_logcdf(lower_var.v));
    let gap = log_upper - log_lower;
    if !(gap.v.is_finite() && gap.v > 0.0) {
        return Err(TransformationNormalError::NumericalFailure { reason: format!(
            "normal CDF endpoint mass is not representable, lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }
    let log_z = log_upper + gap.compose_unary(unary_derivatives_log1mexp_positive(gap.v));
    if !log_z.v.is_finite() {
        return Err(TransformationNormalError::NumericalFailure {
            reason: format!(
                "normal CDF endpoint mass underflowed, lower={lower:.6e}, upper={upper:.6e}"
            ),
        }
        .into());
    }
    Ok(log_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn hand_factorial(n: usize) -> f64 {
        match n {
            0 | 1 => 1.0,
            2 => 2.0,
            3 => 6.0,
            4 => 24.0,
            other => {
                let mut acc = 1.0_f64;
                for k in 2..=other {
                    acc *= k as f64;
                }
                acc
            }
        }
    }

    pub(crate) fn hand_poly_mul_truncated(a: &[[f64; 5]; 5], b: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
        let mut out = [[0.0; 5]; 5];
        for ia in 0..=4 {
            for ib in 0..=(4 - ia) {
                let av = a[ia][ib];
                if av == 0.0 {
                    continue;
                }
                for ja in 0..=(4 - ia) {
                    for jb in 0..=(4 - ia - ja).min(4 - ib) {
                        let bv = b[ja][jb];
                        if bv != 0.0 && ia + ib + ja + jb <= 4 {
                            out[ia + ja][ib + jb] += av * bv;
                        }
                    }
                }
            }
        }
        out
    }

    pub(crate) fn hand_signed_normal_pdf_ratio(
        x: f64,
        polynomial_factor: f64,
        log_z: f64,
        factorial_scale: f64,
    ) -> f64 {
        if polynomial_factor == 0.0 {
            return 0.0;
        }
        const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
        let log_abs = polynomial_factor.abs().ln()
            - 0.5 * x * x
            - LOG_SQRT_2PI
            - factorial_scale.ln()
            - log_z;
        polynomial_factor.signum() * log_abs.exp()
    }

    pub(crate) fn hand_log_normal_cdf_diff_derivatives(
        upper: f64,
        lower: f64,
    ) -> Result<LogNormalCdfDiffDerivatives, String> {
        let log_z = log_normal_cdf_diff(upper, lower)?;
        let s_u = [
            0.0,
            1.0,
            -upper,
            upper * upper - 1.0,
            -(upper * upper * upper - 3.0 * upper),
        ];
        let s_l = [
            0.0,
            -1.0,
            lower,
            -(lower * lower - 1.0),
            lower * lower * lower - 3.0 * lower,
        ];

        let mut r = [[0.0; 5]; 5];
        for order in 1..=4 {
            let factor = hand_factorial(order);
            r[order][0] = hand_signed_normal_pdf_ratio(upper, s_u[order], log_z, factor);
            r[0][order] = hand_signed_normal_pdf_ratio(lower, s_l[order], log_z, factor);
            if !(r[order][0].is_finite() && r[0][order].is_finite()) {
                return Err(TransformationNormalError::NumericalFailure {
                    reason: format!(
                        "hand witness derivative ratio is not representable at order {order}, \
                         lower={lower:.6e}, upper={upper:.6e}, log_z={log_z:.6e}"
                    ),
                }
                .into());
            }
        }

        let r2 = hand_poly_mul_truncated(&r, &r);
        let r3 = hand_poly_mul_truncated(&r2, &r);
        let r4 = hand_poly_mul_truncated(&r3, &r);
        let mut q = [[0.0; 5]; 5];
        for i in 0..=4 {
            for j in 0..=(4 - i) {
                q[i][j] = r[i][j] - 0.5 * r2[i][j] + r3[i][j] / 3.0 - 0.25 * r4[i][j];
            }
        }

        let mut first = [0.0; 2];
        first[0] = q[1][0];
        first[1] = q[0][1];

        let mut second = [[0.0; 2]; 2];
        let mut third = [[[0.0; 2]; 2]; 2];
        let mut fourth = [[[[0.0; 2]; 2]; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                let nu = (a == 0) as usize + (b == 0) as usize;
                let nl = 2 - nu;
                second[a][b] = q[nu][nl] * hand_factorial(nu) * hand_factorial(nl);
                for c in 0..2 {
                    let nu = (a == 0) as usize + (b == 0) as usize + (c == 0) as usize;
                    let nl = 3 - nu;
                    third[a][b][c] = q[nu][nl] * hand_factorial(nu) * hand_factorial(nl);
                    for d in 0..2 {
                        let nu = (a == 0) as usize
                            + (b == 0) as usize
                            + (c == 0) as usize
                            + (d == 0) as usize;
                        let nl = 4 - nu;
                        fourth[a][b][c][d] = q[nu][nl] * hand_factorial(nu) * hand_factorial(nl);
                    }
                }
            }
        }

        Ok(LogNormalCdfDiffDerivatives {
            log_z,
            first,
            second,
            third,
            fourth,
        })
    }

    pub(crate) fn rel_error(actual: f64, expected: f64, floor: f64) -> f64 {
        (actual - expected).abs() / expected.abs().max(floor).max(1e-300)
    }

    pub(crate) fn max_relative_error(
        actual: &LogNormalCdfDiffDerivatives,
        expected: &LogNormalCdfDiffDerivatives,
    ) -> f64 {
        let first_floor = expected
            .first
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let second_floor = expected
            .second
            .iter()
            .flatten()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let third_floor = expected
            .third
            .iter()
            .flatten()
            .flatten()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let fourth_floor = expected
            .fourth
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let mut max_rel = rel_error(actual.log_z, expected.log_z, 1.0);
        for a in 0..2 {
            max_rel = max_rel.max(rel_error(actual.first[a], expected.first[a], first_floor));
            for b in 0..2 {
                max_rel = max_rel.max(rel_error(
                    actual.second[a][b],
                    expected.second[a][b],
                    second_floor,
                ));
                for c in 0..2 {
                    max_rel = max_rel.max(rel_error(
                        actual.third[a][b][c],
                        expected.third[a][b][c],
                        third_floor,
                    ));
                    for d in 0..2 {
                        max_rel = max_rel.max(rel_error(
                            actual.fourth[a][b][c][d],
                            expected.fourth[a][b][c][d],
                            fourth_floor,
                        ));
                    }
                }
            }
        }
        max_rel
    }

    /// #1591 bit-identity guard: the separable single-slot compose
    /// ([`compose_unary_single_slot`], used by
    /// [`log_normal_cdf_diff_tower_ordered`]) must reproduce the dense
    /// `Tower4<2>::compose_unary` of the same single-slot variable BIT-FOR-BIT
    /// (`to_bits`) across both seeding conventions and both `±` negations, so the
    /// perf prune can never silently change a derivative channel.
    #[test]
    pub(crate) fn single_slot_compose_is_bit_identical_to_dense() {
        fn dense(var: &Tower4<2>, stack: [f64; 5]) -> Tower4<2> {
            var.compose_unary(stack)
        }
        // A spread of finite stacks including negative f′ (exercises the ±0.0
        // off-slot accumulation) and the true normal_logcdf stack.
        let stacks: &[[f64; 5]] = &[
            [0.1, -0.2, 0.3, -0.4, 0.5],
            [-1.3, 2.7, -0.9, 1.1, -3.2],
            [0.0, -0.0, 0.0, -0.0, 0.0],
            unary_derivatives_normal_logcdf(-2.5),
            unary_derivatives_normal_logcdf(1.75),
        ];
        for &val in &[-7.5_f64, -2.0, -0.1, 0.0, 0.5, 2.0, 6.5, 38.0] {
            for slot in 0..2 {
                for sign in [1.0_f64, -1.0] {
                    let var = Tower4::<2>::variable(val, slot).scale(sign);
                    for &stack in stacks {
                        let d = dense(&var, stack);
                        let s = compose_unary_single_slot(&var, stack);
                        let chk = |a: f64, b: f64, label: &str| {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{label}: dense {a} vs separable {b} (val={val}, slot={slot}, sign={sign})"
                            );
                        };
                        chk(d.v, s.v, "v");
                        for i in 0..2 {
                            chk(d.g[i], s.g[i], "g");
                            for j in 0..2 {
                                chk(d.h[i][j], s.h[i][j], "h");
                                for k in 0..2 {
                                    chk(d.t3[i][j][k], s.t3[i][j][k], "t3");
                                    for l in 0..2 {
                                        chk(d.t4[i][j][k][l], s.t4[i][j][k][l], "t4");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    pub(crate) fn tower_endpoint_normalizer_matches_hand_witness_across_ctn_regimes() {
        struct Regime {
            pub(crate) name: &'static str,
            pub(crate) cases: &'static [(f64, f64)],
            pub(crate) rel_tol: f64,
        }

        const WIDE: &[(f64, f64)] = &[(-2.0, -7.5), (2.0, -3.0), (8.0, -8.0)];
        const NARROW: &[(f64, f64)] = &[(-0.249_999, -0.25), (1.000_001, 1.0), (-3.999_999, -4.0)];
        const BOTH_POS_TAIL: &[(f64, f64)] = &[(6.5, 6.0), (8.0, 7.9), (8.0, 7.0)];
        const BOTH_NEG_TAIL: &[(f64, f64)] = &[(-6.0, -6.5), (-7.9, -8.0), (-7.0, -8.0)];
        const STRADDLING: &[(f64, f64)] = &[(-0.1, -2.0), (0.1, -0.1), (3.0, -4.0)];
        const NEAR_EQUAL: &[(f64, f64)] = &[
            (1.0e-10, 0.0),
            (-0.999_999_999_999, -1.0),
            (7.000_000_001, 7.0),
            (-7.0, -7.000_000_001),
        ];
        const REGIMES: &[Regime] = &[
            Regime {
                name: "wide windows",
                cases: WIDE,
                rel_tol: 1.0e-11,
            },
            Regime {
                name: "narrow windows",
                cases: NARROW,
                rel_tol: 1.0e-11,
            },
            Regime {
                name: "both positive tail",
                cases: BOTH_POS_TAIL,
                // Same-tail |z|≈8 compares two equivalent stable forms:
                // endpoint PDF/log-Z ratios in the witness and Mills/log1mexp
                // composition in the tower. The observed probe bound is
                // 1.4e-11, set explicitly rather than hidden in the suite-wide
                // 1e-11 target.
                rel_tol: 2.0e-11,
            },
            Regime {
                name: "both negative tail",
                cases: BOTH_NEG_TAIL,
                // Symmetric to the positive-tail regime above; observed probe
                // bound is 1.4e-11 at |z|≈8.
                rel_tol: 2.0e-11,
            },
            Regime {
                name: "sign straddling",
                cases: STRADDLING,
                rel_tol: 1.0e-11,
            },
            Regime {
                name: "near-equal endpoints",
                cases: NEAR_EQUAL,
                rel_tol: 1.0e-11,
            },
        ];

        for regime in REGIMES {
            let mut achieved = 0.0_f64;
            for &(upper, lower) in regime.cases {
                let tower = log_normal_cdf_diff_derivatives(upper, lower).expect(regime.name);
                let witness =
                    hand_log_normal_cdf_diff_derivatives(upper, lower).expect(regime.name);
                achieved = achieved.max(max_relative_error(&tower, &witness));
            }
            assert!(
                achieved <= regime.rel_tol,
                "{} endpoint normalizer tower drift: achieved {:.3e}, bound {:.3e}",
                regime.name,
                achieved,
                regime.rel_tol
            );
        }
    }
}
