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
//! so it lives in its own seam. The companion polynomial / factorial primitives
//! (`factorial`, `poly_mul_truncated`, `signed_normal_pdf_ratio`) exist only to
//! assemble `log_normal_cdf_diff_derivatives` and are kept private to the module.

use crate::probability::{log1mexp_positive, normal_logcdf};

use super::TransformationNormalError;

/// Stable `log[Φ(upper) − Φ(lower)]` for `lower < upper`, evaluated via
/// `normal_logcdf` and a `log1mexp` correction so the endpoint mass survives
/// far-tail underflow.
pub(super) fn log_normal_cdf_diff(upper: f64, lower: f64) -> Result<f64, String> {
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

fn signed_normal_pdf_ratio(
    x: f64,
    polynomial_factor: f64,
    log_z: f64,
    factorial_scale: f64,
) -> f64 {
    if polynomial_factor == 0.0 {
        return 0.0;
    }
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    let log_abs =
        polynomial_factor.abs().ln() - 0.5 * x * x - LOG_SQRT_2PI - factorial_scale.ln() - log_z;
    polynomial_factor.signum() * log_abs.exp()
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LogNormalCdfDiffDerivatives {
    pub(super) log_z: f64,
    pub(super) first: [f64; 2],
    pub(super) second: [[f64; 2]; 2],
    pub(super) third: [[[f64; 2]; 2]; 2],
    pub(super) fourth: [[[[f64; 2]; 2]; 2]; 2],
}

pub(super) fn endpoint_chain_first(q: &LogNormalCdfDiffDerivatives, a: [f64; 2]) -> f64 {
    q.first[0] * a[0] + q.first[1] * a[1]
}

pub(super) fn endpoint_chain_second(
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

pub(super) fn endpoint_chain_third(
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

pub(super) fn endpoint_chain_fourth(
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

fn factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        // CTN normalizer derivatives only need order <= 4; compute generically
        // as a safe fallback for any unexpected higher orders.
        other => {
            let mut acc = 24.0_f64;
            let mut k = 5usize;
            while k <= other {
                acc *= k as f64;
                k += 1;
            }
            acc
        }
    }
}

fn poly_mul_truncated(a: &[[f64; 5]; 5], b: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
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

pub(super) fn log_normal_cdf_diff_derivatives(
    upper: f64,
    lower: f64,
) -> Result<LogNormalCdfDiffDerivatives, String> {
    let log_z = log_normal_cdf_diff(upper, lower)?;
    if !log_z.is_finite() {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "normal CDF endpoint log-mass is not finite, lower={lower:.6e}, upper={upper:.6e}"
            ),
        }
        .into());
    }

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
        let factor = factorial(order);
        r[order][0] = signed_normal_pdf_ratio(upper, s_u[order], log_z, factor);
        r[0][order] = signed_normal_pdf_ratio(lower, s_l[order], log_z, factor);
        if !(r[order][0].is_finite() && r[0][order].is_finite()) {
            return Err(TransformationNormalError::NumericalFailure {
                reason: format!(
                    "normal CDF endpoint derivative ratio is not representable at order {order}, \
                 lower={lower:.6e}, upper={upper:.6e}, log_z={log_z:.6e}"
                ),
            }
            .into());
        }
    }

    let r2 = poly_mul_truncated(&r, &r);
    let r3 = poly_mul_truncated(&r2, &r);
    let r4 = poly_mul_truncated(&r3, &r);
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
            second[a][b] = q[nu][nl] * factorial(nu) * factorial(nl);
            for c in 0..2 {
                let nu = (a == 0) as usize + (b == 0) as usize + (c == 0) as usize;
                let nl = 3 - nu;
                third[a][b][c] = q[nu][nl] * factorial(nu) * factorial(nl);
                for d in 0..2 {
                    let nu = (a == 0) as usize
                        + (b == 0) as usize
                        + (c == 0) as usize
                        + (d == 0) as usize;
                    let nl = 4 - nu;
                    fourth[a][b][c][d] = q[nu][nl] * factorial(nu) * factorial(nl);
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
