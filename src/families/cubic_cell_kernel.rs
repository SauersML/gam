use crate::probability::normal_cdf;

// De-nested cubic transport kernel.
//
// This module implements the de-nested flexible-link/score-warp model
//
//   eta(z) = a + b*z + b*delta_h(z) + delta_w(a + b*z)
//
// where delta_h is the score warp and delta_w is the link deviation.
// This is not the literal nested composition L(a + b*H(z)); it is an
// additive-correction model around the affine core a + b*z.
//
// On each partition cell, both deviations are cubic polynomials, so eta is
// at most sextic in z and q(z) = 0.5*(z^2 + eta^2) is at most degree 12.
// The integral of exp(-q(z)) is evaluated by transporting from the affine
// anchor (c2=c3=0, where q is Gaussian and the integral reduces to BVN)
// to the target non-affine cell via the polynomial moment recurrence.
//
// The partition covers (-∞, +∞) with:
//   • two semi-infinite affine TAIL cells (outside all deviation support),
//   • finitely many interior cells (each a sextic microcell).
// Because tail cells have constant deviations (c2=c3=0), their bounds
// are parameter-independent, so no Leibniz boundary-motion corrections
// appear in the derivatives.
//
// Shared by bernoulli_marginal_slope and survival_marginal_slope families.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalSpanCubic {
    pub left: f64,
    pub right: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

impl LocalSpanCubic {
    #[inline]
    pub fn evaluate(self, x: f64) -> f64 {
        let t = x - self.left;
        self.c0 + self.c1 * t + self.c2 * t * t + self.c3 * t * t * t
    }

    #[inline]
    pub fn first_derivative(self, x: f64) -> f64 {
        let t = x - self.left;
        self.c1 + 2.0 * self.c2 * t + 3.0 * self.c3 * t * t
    }

    #[inline]
    pub fn second_derivative(self, x: f64) -> f64 {
        let t = x - self.left;
        2.0 * self.c2 + 6.0 * self.c3 * t
    }
}

pub const ANCHORED_DEVIATION_KERNEL: &str = "DenestedCubicTransport";
pub const NORMALIZED_CELL_BRANCH_TOL: f64 = 1e-10;
const INV_TWO_PI: f64 = 1.0 / std::f64::consts::TAU;
const RECIP_FACTORIALS_0_TO_10: [f64; 11] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
];
const RECIP_POW2_0_TO_10: [f64; 11] = [
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
];
const ALTERNATING_SIGNS_0_TO_10: [f64; 11] =
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactCellBranch {
    Affine,
    Quartic,
    Sextic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenestedCubicCell {
    pub left: f64,
    pub right: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

impl DenestedCubicCell {
    #[inline]
    pub fn eta(self, z: f64) -> f64 {
        self.c0 + self.c1 * z + self.c2 * z * z + self.c3 * z * z * z
    }

    #[inline]
    pub fn q(self, z: f64) -> f64 {
        let eta = self.eta(z);
        0.5 * (z * z + eta * eta)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenestedPartitionCell {
    pub cell: DenestedCubicCell,
    pub score_span: LocalSpanCubic,
    pub link_span: LocalSpanCubic,
}

impl DenestedPartitionCell {}

#[derive(Clone, Debug, PartialEq)]
pub struct CellMomentState {
    pub branch: ExactCellBranch,
    pub value: f64,
    pub moments: Vec<f64>,
}

/// 20-point Gauss–Legendre nodes on [-1, 1] for the Drezner–Wesolowsky
/// bivariate normal CDF representation.  20 points give >30-digit accuracy
/// for the smooth arcsin-transformed integrand, ensuring the BVN value is
/// exact to f64 precision for all (h, k, ρ).
const BVN_GL_NODES_20: [f64; 20] = [
    -0.993_128_599_185_094_9,
    -0.963_971_927_277_913_8,
    -0.912_234_428_251_325_9,
    -0.839_116_971_822_218_8,
    -0.746_331_906_460_150_8,
    -0.636_053_680_726_515_0,
    -0.510_867_001_950_827_1,
    -0.373_706_088_715_419_6,
    -0.227_785_851_141_645_1,
    -0.076_526_521_133_497_33,
    0.076_526_521_133_497_33,
    0.227_785_851_141_645_1,
    0.373_706_088_715_419_6,
    0.510_867_001_950_827_1,
    0.636_053_680_726_515_0,
    0.746_331_906_460_150_8,
    0.839_116_971_822_218_8,
    0.912_234_428_251_325_9,
    0.963_971_927_277_913_8,
    0.993_128_599_185_094_9,
];

const BVN_GL_WEIGHTS_20: [f64; 20] = [
    0.017_614_007_139_152_12,
    0.040_601_429_800_386_94,
    0.062_672_048_334_109_06,
    0.083_276_741_576_704_75,
    0.101_930_119_817_240_4,
    0.118_194_531_961_518_4,
    0.131_688_638_449_176_6,
    0.142_096_109_318_382_1,
    0.149_172_986_472_603_7,
    0.152_753_387_130_725_9,
    0.152_753_387_130_725_9,
    0.149_172_986_472_603_7,
    0.142_096_109_318_382_1,
    0.131_688_638_449_176_6,
    0.118_194_531_961_518_4,
    0.101_930_119_817_240_4,
    0.083_276_741_576_704_75,
    0.062_672_048_334_109_06,
    0.040_601_429_800_386_94,
    0.017_614_007_139_152_12,
];

fn dedup_sorted_breakpoints(points: &mut Vec<f64>) {
    points.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    points.dedup_by(|lhs, rhs| {
        if *lhs == *rhs {
            true
        } else if lhs.is_finite() && rhs.is_finite() {
            (*lhs - *rhs).abs() <= 1e-12
        } else {
            false
        }
    });
}

#[inline]
pub fn interval_probe_point(left: f64, right: f64) -> Result<f64, String> {
    if !(left < right) {
        return Err(format!(
            "interval probe requires ordered bounds, got [{left}, {right}]"
        ));
    }
    if left.is_finite() && right.is_finite() {
        Ok(0.5 * (left + right))
    } else if left == f64::NEG_INFINITY && right == f64::INFINITY {
        Ok(0.0)
    } else if left == f64::NEG_INFINITY && right.is_finite() {
        Ok(right - 1.0)
    } else if left.is_finite() && right == f64::INFINITY {
        Ok(left + 1.0)
    } else {
        Err(format!(
            "interval probe requires finite bounds or full infinities, got [{left}, {right}]"
        ))
    }
}

#[inline]
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k_eff = k.min(n - k);
    let mut out = 1.0;
    for j in 0..k_eff {
        out *= (n - j) as f64 / (j + 1) as f64;
    }
    out
}

#[inline]
pub fn quartic_qprime_coefficients(c0: f64, c1: f64, c2: f64) -> [f64; 4] {
    [
        c0 * c1,
        1.0 + c1 * c1 + 2.0 * c0 * c2,
        3.0 * c1 * c2,
        2.0 * c2 * c2,
    ]
}

#[inline]
pub fn sextic_qprime_coefficients(c0: f64, c1: f64, c2: f64, c3: f64) -> [f64; 6] {
    [
        c0 * c1,
        1.0 + c1 * c1 + 2.0 * c0 * c2,
        3.0 * c0 * c3 + 3.0 * c1 * c2,
        4.0 * c1 * c3 + 2.0 * c2 * c2,
        5.0 * c2 * c3,
        3.0 * c3 * c3,
    ]
}

#[inline]
pub fn moment_boundary_term(cell: DenestedCubicCell, n: usize) -> f64 {
    let left_term = if cell.left.is_infinite() {
        0.0
    } else {
        cell.left.powi(n as i32) * (-cell.q(cell.left)).exp()
    };
    let right_term = if cell.right.is_infinite() {
        0.0
    } else {
        cell.right.powi(n as i32) * (-cell.q(cell.right)).exp()
    };
    right_term - left_term
}

pub fn reduce_quartic_moments(
    cell: DenestedCubicCell,
    base_m0_m2: [f64; 3],
    max_degree: usize,
) -> Result<Vec<f64>, String> {
    if max_degree <= 2 {
        return Ok(base_m0_m2[..=max_degree].to_vec());
    }
    let d = quartic_qprime_coefficients(cell.c0, cell.c1, cell.c2);
    let lead = d[3];
    if !lead.is_finite() || lead.abs() <= 1e-18 {
        return Err(format!(
            "quartic moment reduction requires nonzero leading coefficient, got {lead:.3e}"
        ));
    }
    let mut moments = vec![0.0; max_degree + 1];
    moments[0] = base_m0_m2[0];
    moments[1] = base_m0_m2[1];
    moments[2] = base_m0_m2[2];
    for n in 0..=(max_degree - 3) {
        let b_n = moment_boundary_term(cell, n);
        let mut numer = if n == 0 {
            0.0
        } else {
            (n as f64) * moments[n - 1]
        };
        for j in 0..=2 {
            numer -= d[j] * moments[n + j];
        }
        numer -= b_n;
        moments[n + 3] = numer / lead;
    }
    Ok(moments)
}

pub fn reduce_sextic_moments(
    cell: DenestedCubicCell,
    base_m0_m4: [f64; 5],
    max_degree: usize,
) -> Result<Vec<f64>, String> {
    if max_degree <= 4 {
        return Ok(base_m0_m4[..=max_degree].to_vec());
    }
    let d = sextic_qprime_coefficients(cell.c0, cell.c1, cell.c2, cell.c3);
    let lead = d[5];
    if !lead.is_finite() || lead.abs() <= 1e-18 {
        return Err(format!(
            "sextic moment reduction requires nonzero leading coefficient, got {lead:.3e}"
        ));
    }
    let mut moments = vec![0.0; max_degree + 1];
    for (idx, value) in base_m0_m4.into_iter().enumerate() {
        moments[idx] = value;
    }
    for n in 0..=(max_degree - 5) {
        let b_n = moment_boundary_term(cell, n);
        let mut numer = if n == 0 {
            0.0
        } else {
            (n as f64) * moments[n - 1]
        };
        for j in 0..=4 {
            numer -= d[j] * moments[n + j];
        }
        numer -= b_n;
        moments[n + 5] = numer / lead;
    }
    Ok(moments)
}

#[cfg(test)]
#[inline]
pub fn polynomial_value(coefficients: &[f64], z: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |acc, &coeff| acc * z + coeff)
}

#[inline]
pub fn cell_first_derivative_from_moments(
    derivative_coefficients: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let value = moment_dot_with_coefficients(derivative_coefficients, moments, "first derivative")?;
    Ok(value * INV_TWO_PI)
}

#[inline]
pub fn cell_polynomial_integral_from_moments(
    polynomial_coefficients: &[f64],
    moments: &[f64],
    label: &str,
) -> Result<f64, String> {
    let value = moment_dot_with_coefficients(polynomial_coefficients, moments, label)?;
    Ok(value * INV_TWO_PI)
}

pub fn cell_second_derivative_from_moments(
    cell: DenestedCubicCell,
    first_coefficients_r: &[f64],
    first_coefficients_s: &[f64],
    second_coefficients_rs: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let second_degree = second_coefficients_rs.len().saturating_sub(1);
    let product_degree = first_coefficients_r.len().saturating_sub(1)
        + first_coefficients_s.len().saturating_sub(1)
        + 3;
    let needed = second_degree.max(product_degree) + 1;
    if needed > moments.len() {
        return Err(format!(
            "insufficient reduced moments for second derivative: need {}, have {}",
            needed,
            moments.len()
        ));
    }
    let second_term = moment_dot_with_coefficients_unchecked(second_coefficients_rs, moments);
    let cubic = [cell.c0, cell.c1, cell.c2, cell.c3];
    let mut eta_term = 0.0;
    for (e, &eta_coeff) in cubic.iter().enumerate() {
        for (i, &lhs) in first_coefficients_r.iter().enumerate() {
            for (j, &rhs) in first_coefficients_s.iter().enumerate() {
                eta_term += eta_coeff * lhs * rhs * moments[e + i + j];
            }
        }
    }
    Ok((second_term - eta_term) * INV_TWO_PI)
}

#[inline]
fn moment_dot_with_coefficients(
    coefficients: &[f64],
    moments: &[f64],
    label: &str,
) -> Result<f64, String> {
    if coefficients.len() > moments.len() {
        return Err(format!(
            "insufficient reduced moments for {label}: need {}, have {}",
            coefficients.len(),
            moments.len()
        ));
    }
    Ok(moment_dot_with_coefficients_unchecked(
        coefficients,
        moments,
    ))
}

#[inline]
fn moment_dot_with_coefficients_unchecked(coefficients: &[f64], moments: &[f64]) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(idx, coeff)| coeff * moments[idx])
        .sum::<f64>()
}

#[inline]
fn require_moments_degree(
    required_degree: usize,
    moments: &[f64],
    label: &str,
) -> Result<(), String> {
    if required_degree >= moments.len() {
        return Err(format!(
            "insufficient reduced moments for {label}: need {}, have {}",
            required_degree + 1,
            moments.len()
        ));
    }
    Ok(())
}

pub fn cell_third_derivative_from_moments(
    cell: DenestedCubicCell,
    first_coefficients_r: &[f64],
    first_coefficients_s: &[f64],
    first_coefficients_t: &[f64],
    second_coefficients_rs: &[f64],
    second_coefficients_rt: &[f64],
    second_coefficients_st: &[f64],
    third_coefficients_rst: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
    let second_sum_degree = [
        second_coefficients_rs.len() + first_coefficients_t.len(),
        second_coefficients_rt.len() + first_coefficients_s.len(),
        second_coefficients_st.len() + first_coefficients_r.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .saturating_sub(1);
    let triple_product_degree =
        first_coefficients_r.len() + first_coefficients_s.len() + first_coefficients_t.len() - 3;
    let needed = (third_coefficients_rst.len().saturating_sub(1))
        .max(3 + second_sum_degree)
        .max(6 + triple_product_degree);
    require_moments_degree(needed, moments, "third derivative")?;

    let third_term = moment_dot_with_coefficients_unchecked(third_coefficients_rst, moments);
    let mut eta_second_term = 0.0;
    for (e, &eta_coeff) in eta.iter().enumerate() {
        for (i, &rs) in second_coefficients_rs.iter().enumerate() {
            for (j, &t) in first_coefficients_t.iter().enumerate() {
                eta_second_term += eta_coeff * rs * t * moments[e + i + j];
            }
        }
        for (i, &rt) in second_coefficients_rt.iter().enumerate() {
            for (j, &s) in first_coefficients_s.iter().enumerate() {
                eta_second_term += eta_coeff * rt * s * moments[e + i + j];
            }
        }
        for (i, &st) in second_coefficients_st.iter().enumerate() {
            for (j, &r) in first_coefficients_r.iter().enumerate() {
                eta_second_term += eta_coeff * st * r * moments[e + i + j];
            }
        }
    }

    let mut eta_sq_minus_one = [0.0; 7];
    for (i, &eta_i) in eta.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            eta_sq_minus_one[i + j] += eta_i * eta_j;
        }
    }
    eta_sq_minus_one[0] -= 1.0;
    let mut cubic_coeff_term = 0.0;
    for (e, &weight) in eta_sq_minus_one.iter().enumerate() {
        for (i, &r) in first_coefficients_r.iter().enumerate() {
            for (j, &s) in first_coefficients_s.iter().enumerate() {
                for (k, &t) in first_coefficients_t.iter().enumerate() {
                    cubic_coeff_term += weight * r * s * t * moments[e + i + j + k];
                }
            }
        }
    }

    Ok((third_term - eta_second_term + cubic_coeff_term) * INV_TWO_PI)
}

pub fn cell_fourth_derivative_from_moments(
    cell: DenestedCubicCell,
    first_coefficients_r: &[f64],
    first_coefficients_s: &[f64],
    first_coefficients_t: &[f64],
    first_coefficients_u: &[f64],
    second_coefficients_rs: &[f64],
    second_coefficients_rt: &[f64],
    second_coefficients_ru: &[f64],
    second_coefficients_st: &[f64],
    second_coefficients_su: &[f64],
    second_coefficients_tu: &[f64],
    third_coefficients_rst: &[f64],
    third_coefficients_rsu: &[f64],
    third_coefficients_rtu: &[f64],
    third_coefficients_stu: &[f64],
    fourth_coefficients_rstu: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
    let linear_sum_degree = [
        third_coefficients_rst.len() + first_coefficients_u.len(),
        third_coefficients_rsu.len() + first_coefficients_t.len(),
        third_coefficients_rtu.len() + first_coefficients_s.len(),
        third_coefficients_stu.len() + first_coefficients_r.len(),
        second_coefficients_rs.len() + second_coefficients_tu.len(),
        second_coefficients_rt.len() + second_coefficients_su.len(),
        second_coefficients_ru.len() + second_coefficients_st.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .saturating_sub(1);
    let quad_sum_degree = [
        second_coefficients_rs.len() + first_coefficients_t.len() + first_coefficients_u.len(),
        second_coefficients_rt.len() + first_coefficients_s.len() + first_coefficients_u.len(),
        second_coefficients_ru.len() + first_coefficients_s.len() + first_coefficients_t.len(),
        second_coefficients_st.len() + first_coefficients_r.len() + first_coefficients_u.len(),
        second_coefficients_su.len() + first_coefficients_r.len() + first_coefficients_t.len(),
        second_coefficients_tu.len() + first_coefficients_r.len() + first_coefficients_s.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .saturating_sub(2);
    let quartic_product_degree = first_coefficients_r.len()
        + first_coefficients_s.len()
        + first_coefficients_t.len()
        + first_coefficients_u.len()
        - 4;
    let needed = (fourth_coefficients_rstu.len().saturating_sub(1))
        .max(3 + linear_sum_degree)
        .max(6 + quad_sum_degree)
        .max(9 + quartic_product_degree);
    require_moments_degree(needed, moments, "fourth derivative")?;

    let fourth_term = moment_dot_with_coefficients_unchecked(fourth_coefficients_rstu, moments);
    let mut eta_linear_term = 0.0;
    for (e, &eta_coeff) in eta.iter().enumerate() {
        for (i, &rst) in third_coefficients_rst.iter().enumerate() {
            for (j, &u_coeff) in first_coefficients_u.iter().enumerate() {
                eta_linear_term += eta_coeff * rst * u_coeff * moments[e + i + j];
            }
        }
        for (i, &rsu) in third_coefficients_rsu.iter().enumerate() {
            for (j, &t_coeff) in first_coefficients_t.iter().enumerate() {
                eta_linear_term += eta_coeff * rsu * t_coeff * moments[e + i + j];
            }
        }
        for (i, &rtu) in third_coefficients_rtu.iter().enumerate() {
            for (j, &s_coeff) in first_coefficients_s.iter().enumerate() {
                eta_linear_term += eta_coeff * rtu * s_coeff * moments[e + i + j];
            }
        }
        for (i, &stu) in third_coefficients_stu.iter().enumerate() {
            for (j, &r_coeff) in first_coefficients_r.iter().enumerate() {
                eta_linear_term += eta_coeff * stu * r_coeff * moments[e + i + j];
            }
        }
        for (i, &rs) in second_coefficients_rs.iter().enumerate() {
            for (j, &tu) in second_coefficients_tu.iter().enumerate() {
                eta_linear_term += eta_coeff * rs * tu * moments[e + i + j];
            }
        }
        for (i, &rt) in second_coefficients_rt.iter().enumerate() {
            for (j, &su) in second_coefficients_su.iter().enumerate() {
                eta_linear_term += eta_coeff * rt * su * moments[e + i + j];
            }
        }
        for (i, &ru) in second_coefficients_ru.iter().enumerate() {
            for (j, &st) in second_coefficients_st.iter().enumerate() {
                eta_linear_term += eta_coeff * ru * st * moments[e + i + j];
            }
        }
    }

    let mut eta_sq_minus_one = [0.0; 7];
    for (i, &eta_i) in eta.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            eta_sq_minus_one[i + j] += eta_i * eta_j;
        }
    }
    eta_sq_minus_one[0] -= 1.0;
    let mut quad_coeff_term = 0.0;
    for (e, &weight) in eta_sq_minus_one.iter().enumerate() {
        for (i, &rs) in second_coefficients_rs.iter().enumerate() {
            for (j, &t_coeff) in first_coefficients_t.iter().enumerate() {
                for (k, &u_coeff) in first_coefficients_u.iter().enumerate() {
                    quad_coeff_term += weight * rs * t_coeff * u_coeff * moments[e + i + j + k];
                }
            }
        }
        for (i, &rt) in second_coefficients_rt.iter().enumerate() {
            for (j, &s_coeff) in first_coefficients_s.iter().enumerate() {
                for (k, &u_coeff) in first_coefficients_u.iter().enumerate() {
                    quad_coeff_term += weight * rt * s_coeff * u_coeff * moments[e + i + j + k];
                }
            }
        }
        for (i, &ru) in second_coefficients_ru.iter().enumerate() {
            for (j, &s_coeff) in first_coefficients_s.iter().enumerate() {
                for (k, &t_coeff) in first_coefficients_t.iter().enumerate() {
                    quad_coeff_term += weight * ru * s_coeff * t_coeff * moments[e + i + j + k];
                }
            }
        }
        for (i, &st) in second_coefficients_st.iter().enumerate() {
            for (j, &r_coeff) in first_coefficients_r.iter().enumerate() {
                for (k, &u_coeff) in first_coefficients_u.iter().enumerate() {
                    quad_coeff_term += weight * st * r_coeff * u_coeff * moments[e + i + j + k];
                }
            }
        }
        for (i, &su) in second_coefficients_su.iter().enumerate() {
            for (j, &r_coeff) in first_coefficients_r.iter().enumerate() {
                for (k, &t_coeff) in first_coefficients_t.iter().enumerate() {
                    quad_coeff_term += weight * su * r_coeff * t_coeff * moments[e + i + j + k];
                }
            }
        }
        for (i, &tu) in second_coefficients_tu.iter().enumerate() {
            for (j, &r_coeff) in first_coefficients_r.iter().enumerate() {
                for (k, &s_coeff) in first_coefficients_s.iter().enumerate() {
                    quad_coeff_term += weight * tu * r_coeff * s_coeff * moments[e + i + j + k];
                }
            }
        }
    }

    let mut eta_sq = [0.0; 7];
    for (i, &eta_i) in eta.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            eta_sq[i + j] += eta_i * eta_j;
        }
    }
    let mut cubic_weight = [0.0; 10];
    for (i, &eta_sq_i) in eta_sq.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            cubic_weight[i + j] -= eta_sq_i * eta_j;
        }
    }
    for (idx, &eta_coeff) in eta.iter().enumerate() {
        cubic_weight[idx] += 3.0 * eta_coeff;
    }
    let mut quartic_coeff_term = 0.0;
    for (e, &weight) in cubic_weight.iter().enumerate() {
        for (i, &r_coeff) in first_coefficients_r.iter().enumerate() {
            for (j, &s_coeff) in first_coefficients_s.iter().enumerate() {
                for (k, &t_coeff) in first_coefficients_t.iter().enumerate() {
                    for (l, &u_coeff) in first_coefficients_u.iter().enumerate() {
                        quartic_coeff_term += weight
                            * r_coeff
                            * s_coeff
                            * t_coeff
                            * u_coeff
                            * moments[e + i + j + k + l];
                    }
                }
            }
        }
    }

    Ok((fourth_term - eta_linear_term + quad_coeff_term + quartic_coeff_term) * INV_TWO_PI)
}

#[inline]
pub fn global_cubic_from_local(span: LocalSpanCubic) -> (f64, f64, f64, f64) {
    let left = span.left;
    let q0 = span.c0 - span.c1 * left + span.c2 * left * left - span.c3 * left * left * left;
    let q1 = span.c1 - 2.0 * span.c2 * left + 3.0 * span.c3 * left * left;
    let q2 = span.c2 - 3.0 * span.c3 * left;
    let q3 = span.c3;
    (q0, q1, q2, q3)
}

#[inline]
pub fn transformed_link_cubic(link_span: LocalSpanCubic, a: f64, b: f64) -> (f64, f64, f64, f64) {
    let shift = a - link_span.left;
    let d0 = link_span.c0
        + link_span.c1 * shift
        + link_span.c2 * shift * shift
        + link_span.c3 * shift * shift * shift;
    let d1 = b * (link_span.c1 + 2.0 * link_span.c2 * shift + 3.0 * link_span.c3 * shift * shift);
    let d2 = b * b * (link_span.c2 + 3.0 * link_span.c3 * shift);
    let d3 = link_span.c3 * b * b * b;
    (d0, d1, d2, d3)
}

#[inline]
pub fn denested_cell_coefficients(
    score_span: LocalSpanCubic,
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> [f64; 4] {
    let (h0, h1, h2, h3) = global_cubic_from_local(score_span);
    let (d0, d1, d2, d3) = transformed_link_cubic(link_span, a, b);
    [a + b * h0 + d0, b + b * h1 + d1, b * h2 + d2, b * h3 + d3]
}

#[inline]
pub fn denested_cell_coefficient_partials(
    score_span: LocalSpanCubic,
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4]) {
    let (h0, h1, h2, h3) = global_cubic_from_local(score_span);
    let shift = a - link_span.left;
    let alpha1 = link_span.c1;
    let alpha2 = link_span.c2;
    let alpha3 = link_span.c3;
    let dc_da = [
        1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
        3.0 * alpha3 * b * b,
        0.0,
    ];
    let dc_db = [
        h0,
        1.0 + h1 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        h2 + 2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
        h3 + 3.0 * alpha3 * b * b,
    ];
    (dc_da, dc_db)
}

#[inline]
pub fn denested_cell_second_partials(
    score_span: LocalSpanCubic,
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let score_left = score_span.left;
    if !score_left.is_finite() {
        return ([f64::NAN; 4], [f64::NAN; 4], [f64::NAN; 4]);
    }
    let shift = a - link_span.left;
    let alpha2 = link_span.c2;
    let alpha3 = link_span.c3;
    let dc_daa = [
        2.0 * alpha2 + 6.0 * alpha3 * shift,
        6.0 * alpha3 * b,
        0.0,
        0.0,
    ];
    let dc_dab = [
        0.0,
        2.0 * alpha2 + 6.0 * alpha3 * shift,
        6.0 * alpha3 * b,
        0.0,
    ];
    let dc_dbb = [
        0.0,
        0.0,
        2.0 * (alpha2 + 3.0 * alpha3 * shift),
        6.0 * alpha3 * b,
    ];
    (dc_daa, dc_dab, dc_dbb)
}

#[inline]
pub fn denested_cell_third_partials(
    link_span: LocalSpanCubic,
) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let alpha3 = link_span.c3;
    (
        [6.0 * alpha3, 0.0, 0.0, 0.0],
        [0.0, 6.0 * alpha3, 0.0, 0.0],
        [0.0, 0.0, 6.0 * alpha3, 0.0],
        [0.0, 0.0, 0.0, 6.0 * alpha3],
    )
}

#[inline]
pub fn score_basis_cell_coefficients(score_basis_span: LocalSpanCubic, b: f64) -> [f64; 4] {
    let (h0, h1, h2, h3) = global_cubic_from_local(score_basis_span);
    [b * h0, b * h1, b * h2, b * h3]
}

#[inline]
pub fn link_basis_cell_coefficients(link_basis_span: LocalSpanCubic, a: f64, b: f64) -> [f64; 4] {
    let (d0, d1, d2, d3) = transformed_link_cubic(link_basis_span, a, b);
    [d0, d1, d2, d3]
}

#[inline]
pub fn link_basis_cell_coefficient_partials(
    link_basis_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4]) {
    let shift = a - link_basis_span.left;
    let alpha1 = link_basis_span.c1;
    let alpha2 = link_basis_span.c2;
    let alpha3 = link_basis_span.c3;
    let dc_da = [
        alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
        3.0 * alpha3 * b * b,
        0.0,
    ];
    let dc_db = [
        0.0,
        alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
        3.0 * alpha3 * b * b,
    ];
    (dc_da, dc_db)
}

#[inline]
pub fn link_basis_cell_second_partials(
    link_basis_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let shift = a - link_basis_span.left;
    let alpha2 = link_basis_span.c2;
    let alpha3 = link_basis_span.c3;
    let dc_daa = [
        2.0 * alpha2 + 6.0 * alpha3 * shift,
        6.0 * alpha3 * b,
        0.0,
        0.0,
    ];
    let dc_dab = [
        0.0,
        2.0 * alpha2 + 6.0 * alpha3 * shift,
        6.0 * alpha3 * b,
        0.0,
    ];
    let dc_dbb = [
        0.0,
        0.0,
        2.0 * (alpha2 + 3.0 * alpha3 * shift),
        6.0 * alpha3 * b,
    ];
    (dc_daa, dc_dab, dc_dbb)
}

#[inline]
pub fn link_basis_cell_third_partials(
    link_basis_span: LocalSpanCubic,
) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let alpha3 = link_basis_span.c3;
    (
        [6.0 * alpha3, 0.0, 0.0, 0.0],
        [0.0, 6.0 * alpha3, 0.0, 0.0],
        [0.0, 0.0, 6.0 * alpha3, 0.0],
        [0.0, 0.0, 0.0, 6.0 * alpha3],
    )
}

pub fn build_denested_partition_cells<FS, FL>(
    a: f64,
    b: f64,
    score_breaks: &[f64],
    link_breaks: &[f64],
    score_span_at: FS,
    link_span_at: FL,
) -> Result<Vec<DenestedPartitionCell>, String>
where
    FS: FnMut(f64) -> Result<LocalSpanCubic, String>,
    FL: FnMut(f64) -> Result<LocalSpanCubic, String>,
{
    build_denested_partition_cells_with_tails(
        a,
        b,
        score_breaks,
        link_breaks,
        score_span_at,
        link_span_at,
    )
}

/// Build a partition covering `(-∞, +∞)` with parameter-independent outer
/// bounds.  Interior cells use the same finite-cell polynomial algebra.
/// The two tail cells are guaranteed affine (c2=c3=0) because both
/// deviations saturate to constants outside their knot support.
///
/// The tail cells' score/link spans come from the same closures evaluated
/// at a representative point in the tail region — the closures must return
/// constant (c1=c2=c3=0) cubics for points outside support.
pub fn build_denested_partition_cells_with_tails<FS, FL>(
    a: f64,
    b: f64,
    score_breaks: &[f64],
    link_breaks: &[f64],
    mut score_span_at: FS,
    mut link_span_at: FL,
) -> Result<Vec<DenestedPartitionCell>, String>
where
    FS: FnMut(f64) -> Result<LocalSpanCubic, String>,
    FL: FnMut(f64) -> Result<LocalSpanCubic, String>,
{
    // Collect all INTERNAL split points (finite).
    let mut split_points = score_breaks.to_vec();
    if b.abs() > 1e-12 {
        for &tau in link_breaks {
            let z = (tau - a) / b;
            if z.is_finite() {
                split_points.push(z);
            }
        }
    }
    dedup_sorted_breakpoints(&mut split_points);

    let mut out = Vec::new();

    if split_points.is_empty() {
        let score_span = score_span_at(0.0)?;
        let link_span = link_span_at(a)?;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        return Ok(vec![DenestedPartitionCell {
            cell: DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: f64::INFINITY,
                c0: coeffs[0],
                c1: coeffs[1],
                c2: 0.0,
                c3: 0.0,
            },
            score_span,
            link_span,
        }]);
    }

    // ── Left tail cell: (-∞, leftmost_split] ──
    let leftmost = split_points[0];
    // Evaluate spans at a point just left of the leftmost split.  The
    // closures return constant tail cubics for this region.
    let left_probe = interval_probe_point(f64::NEG_INFINITY, leftmost)?;
    let left_score_span = score_span_at(left_probe)?;
    let left_link_span = link_span_at(a + b * left_probe)?;
    let left_coeffs = denested_cell_coefficients(left_score_span, left_link_span, a, b);
    if left_coeffs[2].abs() > NORMALIZED_CELL_BRANCH_TOL
        || left_coeffs[3].abs() > NORMALIZED_CELL_BRANCH_TOL
    {
        return Err(format!(
            "left tail cell must be affine (deviations constant outside support), \
             got c2={:.3e}, c3={:.3e}",
            left_coeffs[2], left_coeffs[3]
        ));
    }
    out.push(DenestedPartitionCell {
        cell: DenestedCubicCell {
            left: f64::NEG_INFINITY,
            right: leftmost,
            c0: left_coeffs[0],
            c1: left_coeffs[1],
            c2: 0.0,
            c3: 0.0,
        },
        score_span: left_score_span,
        link_span: left_link_span,
    });

    // ── Interior cells (all finite) ──
    for window in split_points.windows(2) {
        let left = window[0];
        let right = window[1];
        if !left.is_finite() || !right.is_finite() || right - left <= 1e-12 {
            continue;
        }
        let mid = interval_probe_point(left, right)?;
        let score_span = score_span_at(mid)?;
        let link_span = link_span_at(a + b * mid)?;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        out.push(DenestedPartitionCell {
            cell: DenestedCubicCell {
                left,
                right,
                c0: coeffs[0],
                c1: coeffs[1],
                c2: coeffs[2],
                c3: coeffs[3],
            },
            score_span,
            link_span,
        });
    }

    // ── Right tail cell: [rightmost_split, +∞) ──
    let rightmost = *split_points.last().unwrap();
    let right_probe = interval_probe_point(rightmost, f64::INFINITY)?;
    let right_score_span = score_span_at(right_probe)?;
    let right_link_span = link_span_at(a + b * right_probe)?;
    let right_coeffs = denested_cell_coefficients(right_score_span, right_link_span, a, b);
    if right_coeffs[2].abs() > NORMALIZED_CELL_BRANCH_TOL
        || right_coeffs[3].abs() > NORMALIZED_CELL_BRANCH_TOL
    {
        return Err(format!(
            "right tail cell must be affine (deviations constant outside support), \
             got c2={:.3e}, c3={:.3e}",
            right_coeffs[2], right_coeffs[3]
        ));
    }
    out.push(DenestedPartitionCell {
        cell: DenestedCubicCell {
            left: rightmost,
            right: f64::INFINITY,
            c0: right_coeffs[0],
            c1: right_coeffs[1],
            c2: 0.0,
            c3: 0.0,
        },
        score_span: right_score_span,
        link_span: right_link_span,
    });

    Ok(out)
}

#[inline]
pub fn normalized_non_affine_coefficients(
    left: f64,
    right: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
) -> Result<(f64, f64), String> {
    let width = right - left;
    if !width.is_finite() || width <= 0.0 {
        return Err(format!(
            "normalized cubic coefficients require a positive finite cell width, got left={left}, right={right}"
        ));
    }
    let anchor_scale = c0.abs() + c1.abs();
    if !anchor_scale.is_finite() {
        return Err(format!(
            "normalized cubic coefficients require finite affine coefficients, got c0={c0}, c1={c1}"
        ));
    }
    let mid = 0.5 * (left + right);
    let half = 0.5 * width;
    let k2 = half * half * (c2 + 3.0 * c3 * mid);
    let k3 = c3 * half * half * half;
    Ok((k2, k3))
}

#[inline]
pub fn branch_cell(cell: DenestedCubicCell) -> Result<ExactCellBranch, String> {
    if !cell.left.is_finite() || !cell.right.is_finite() {
        if cell.c2.abs() <= NORMALIZED_CELL_BRANCH_TOL
            && cell.c3.abs() <= NORMALIZED_CELL_BRANCH_TOL
        {
            return Ok(ExactCellBranch::Affine);
        }
        return Err(format!(
            "non-affine cells require finite bounds, got [{}, {}] with c2={:.6e}, c3={:.6e}",
            cell.left, cell.right, cell.c2, cell.c3
        ));
    }
    let (k2, k3) = normalized_non_affine_coefficients(
        cell.left, cell.right, cell.c0, cell.c1, cell.c2, cell.c3,
    )?;
    if k2.abs() <= NORMALIZED_CELL_BRANCH_TOL && k3.abs() <= NORMALIZED_CELL_BRANCH_TOL {
        Ok(ExactCellBranch::Affine)
    } else if k3.abs() <= NORMALIZED_CELL_BRANCH_TOL {
        Ok(ExactCellBranch::Quartic)
    } else {
        Ok(ExactCellBranch::Sextic)
    }
}

pub fn bivariate_normal_cdf(h: f64, k: f64, rho: f64) -> Result<f64, String> {
    if !h.is_finite() && !h.is_infinite() {
        return Err("bivariate normal cdf requires finite or infinite h".to_string());
    }
    if !k.is_finite() && !k.is_infinite() {
        return Err("bivariate normal cdf requires finite or infinite k".to_string());
    }
    if !rho.is_finite() {
        return Err(format!(
            "bivariate normal cdf requires finite correlation, got {rho}"
        ));
    }
    if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if h == f64::INFINITY {
        return Ok(normal_cdf(k));
    }
    if k == f64::INFINITY {
        return Ok(normal_cdf(h));
    }

    let rho_clamped = rho.clamp(-1.0, 1.0);
    if rho_clamped >= 1.0 - 1e-12 {
        return Ok(normal_cdf(h.min(k)));
    }
    if rho_clamped <= -1.0 + 1e-12 {
        return Ok((normal_cdf(h) - normal_cdf(-k)).clamp(0.0, 1.0));
    }

    let hs = 0.5 * (h * h + k * k);
    let asr = rho_clamped.asin();
    let mut sum = 0.0;
    for (&node, &weight) in BVN_GL_NODES_20.iter().zip(BVN_GL_WEIGHTS_20.iter()) {
        let sn = (0.5 * asr * (node + 1.0)).sin();
        let one_minus = 1.0 - sn * sn;
        let expo = ((sn * h * k) - hs) / one_minus;
        sum += weight * expo.exp();
    }
    Ok((normal_cdf(h) * normal_cdf(k) + asr * sum / (4.0 * std::f64::consts::PI)).clamp(0.0, 1.0))
}

fn exp_neg_half_square(x: f64) -> f64 {
    if x.is_infinite() {
        0.0
    } else {
        (-0.5 * x * x).exp()
    }
}

fn truncated_gaussian_moment_raw(a: f64, b: f64, order: usize) -> f64 {
    match order {
        0 => {
            let cdf = |x: f64| {
                if x.is_infinite() {
                    if x.is_sign_positive() { 1.0 } else { 0.0 }
                } else {
                    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
                }
            };
            (2.0 * std::f64::consts::PI).sqrt() * (cdf(b) - cdf(a))
        }
        1 => exp_neg_half_square(a) - exp_neg_half_square(b),
        n => {
            let left = if a.is_infinite() {
                0.0
            } else {
                a.powi((n - 1) as i32) * exp_neg_half_square(a)
            };
            let right = if b.is_infinite() {
                0.0
            } else {
                b.powi((n - 1) as i32) * exp_neg_half_square(b)
            };
            left - right + (n as f64 - 1.0) * truncated_gaussian_moment_raw(a, b, n - 2)
        }
    }
}

pub fn affine_anchor_moment_vector(
    alpha: f64,
    beta: f64,
    left: f64,
    right: f64,
    max_degree: usize,
) -> Vec<f64> {
    let s = (1.0 + beta * beta).sqrt();
    let mu = -alpha * beta / (1.0 + beta * beta);
    let y_left = if left.is_infinite() {
        if left.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        s * (left - mu)
    };
    let y_right = if right.is_infinite() {
        if right.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        s * (right - mu)
    };
    let anchor = (-alpha * alpha / (2.0 * s * s)).exp() / s;
    let mut t = vec![0.0; max_degree + 1];
    for (k, tk) in t.iter_mut().enumerate() {
        *tk = truncated_gaussian_moment_raw(y_left, y_right, k);
    }
    let mut out = vec![0.0; max_degree + 1];
    for n in 0..=max_degree {
        let mut acc = 0.0;
        for k in 0..=n {
            acc +=
                binomial_coefficient(n, k) * mu.powi((n - k) as i32) * s.powi(-(k as i32)) * t[k];
        }
        out[n] = anchor * acc;
    }
    out
}

fn affine_zero_moment(alpha: f64, beta: f64, left: f64, right: f64) -> f64 {
    let s = (1.0 + beta * beta).sqrt();
    let mu = -alpha * beta / (1.0 + beta * beta);
    let y_left = if left.is_infinite() {
        if left.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        s * (left - mu)
    };
    let y_right = if right.is_infinite() {
        if right.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        s * (right - mu)
    };
    let anchor = (-alpha * alpha / (2.0 * s * s)).exp() / s;
    let span_mass =
        (2.0 * std::f64::consts::PI).sqrt() * (normal_cdf(y_right) - normal_cdf(y_left));
    anchor * span_mass
}

fn affine_value_from_moment_primitive(alpha: f64, beta: f64, left: f64, right: f64) -> f64 {
    let mut acc = 0.0;
    for (&node, &weight) in BVN_GL_NODES_20.iter().zip(BVN_GL_WEIGHTS_20.iter()) {
        let s = 0.5 * (node + 1.0);
        let angle = std::f64::consts::FRAC_PI_2 * s;
        let tail = angle.tan();
        let alpha_prime = alpha - tail;
        let jacobian = std::f64::consts::FRAC_PI_2 * (1.0 + tail * tail);
        let deriv = INV_TWO_PI * affine_zero_moment(alpha_prime, beta, left, right);
        acc += weight * deriv * jacobian;
    }
    (0.5 * acc).clamp(0.0, 1.0)
}

/// Evaluate an affine cell (c2=c3=0) with a value/moment-consistent primitive.
///
/// Value and moments are now generated from the same affine moment primitive.
/// The zero-moment derivative is exact, and `value` is reconstructed by
/// integrating `d value / d alpha = INV_TWO_PI * moments[0]` over `alpha`
/// on a transformed semi-infinite domain.
pub fn evaluate_affine_cell_state(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    let alpha = cell.c0;
    let beta = cell.c1;
    let value = affine_value_from_moment_primitive(alpha, beta, cell.left, cell.right);
    let moments = affine_anchor_moment_vector(alpha, beta, cell.left, cell.right, max_degree);
    Ok(CellMomentState {
        branch: ExactCellBranch::Affine,
        value,
        moments,
    })
}

#[derive(Clone, Debug)]
struct TransportCellState {
    lambda: f64,
    cell: DenestedCubicCell,
    value: f64,
    basis_moments: Vec<f64>,
}

#[inline]
fn recip_factorial(n: usize) -> f64 {
    if let Some(&value) = RECIP_FACTORIALS_0_TO_10.get(n) {
        value
    } else {
        1.0 / (1..=n).fold(1.0, |acc, k| acc * k as f64)
    }
}

#[inline]
fn recip_pow2(n: usize) -> f64 {
    if let Some(&value) = RECIP_POW2_0_TO_10.get(n) {
        value
    } else {
        2.0_f64.powi(-(n as i32))
    }
}

#[inline]
fn alternating_sign(n: usize) -> f64 {
    if let Some(&value) = ALTERNATING_SIGNS_0_TO_10.get(n) {
        value
    } else if n % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn poly_mul(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; lhs.len() + rhs.len() - 1];
    for (i, &lv) in lhs.iter().enumerate() {
        for (j, &rv) in rhs.iter().enumerate() {
            out[i + j] += lv * rv;
        }
    }
    out
}

#[inline]
fn add_scaled_poly_product(target: &mut [f64], lhs: &[f64], rhs: &[f64], scale: f64) {
    for (i, &lv) in lhs.iter().enumerate() {
        for (j, &rv) in rhs.iter().enumerate() {
            target[i + j] += scale * lv * rv;
        }
    }
}

fn build_transport_series_polynomials(
    current: DenestedCubicCell,
    delta_c2: f64,
    delta_c3: f64,
    order: usize,
) -> Vec<Vec<f64>> {
    let eta = [current.c0, current.c1, current.c2, current.c3];
    let delta = [0.0, 0.0, delta_c2, delta_c3];
    let a_poly = poly_mul(&eta, &delta);
    let b_poly = poly_mul(&delta, &delta);

    let mut a_powers = Vec::with_capacity(order + 1);
    a_powers.push(vec![1.0]);
    for p in 1..=order {
        a_powers.push(poly_mul(&a_powers[p - 1], &a_poly));
    }

    let max_q = order / 2;
    let mut b_powers = Vec::with_capacity(max_q + 1);
    b_powers.push(vec![1.0]);
    for q in 1..=max_q {
        b_powers.push(poly_mul(&b_powers[q - 1], &b_poly));
    }

    let mut coeffs = Vec::with_capacity(order + 1);
    coeffs.push(vec![1.0]);
    for m in 1..=order {
        let mut poly = vec![0.0; 6 * m + 1];
        for q in 0..=m / 2 {
            let p = m - 2 * q;
            let scale =
                alternating_sign(p + q) * recip_factorial(p) * recip_factorial(q) * recip_pow2(q);
            add_scaled_poly_product(&mut poly, &a_powers[p], &b_powers[q], scale);
        }
        coeffs.push(poly);
    }
    coeffs
}

fn non_affine_required_degree(branch: ExactCellBranch, order: usize) -> usize {
    let basis_max = match branch {
        ExactCellBranch::Quartic => 2,
        ExactCellBranch::Sextic => 4,
        ExactCellBranch::Affine => 4,
    };
    basis_max.max(3) + 6 * order
}

fn reduced_moments_from_basis(
    cell: DenestedCubicCell,
    branch: ExactCellBranch,
    basis_moments: &[f64],
    max_degree: usize,
) -> Result<Vec<f64>, String> {
    match branch {
        ExactCellBranch::Affine => Ok(affine_anchor_moment_vector(
            cell.c0, cell.c1, cell.left, cell.right, max_degree,
        )),
        ExactCellBranch::Quartic => {
            if basis_moments.len() < 3 {
                return Err("quartic transported state is missing M0..M2".to_string());
            }
            reduce_quartic_moments(
                cell,
                [basis_moments[0], basis_moments[1], basis_moments[2]],
                max_degree,
            )
        }
        ExactCellBranch::Sextic => {
            if basis_moments.len() < 5 {
                return Err("sextic transported state is missing M0..M4".to_string());
            }
            reduce_sextic_moments(
                cell,
                [
                    basis_moments[0],
                    basis_moments[1],
                    basis_moments[2],
                    basis_moments[3],
                    basis_moments[4],
                ],
                max_degree,
            )
        }
    }
}

fn evaluate_non_affine_transport_step(
    state: &TransportCellState,
    target_branch: ExactCellBranch,
    delta_c2: f64,
    delta_c3: f64,
    step: f64,
    order: usize,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<Option<TransportCellState>, String> {
    let required_degree = non_affine_required_degree(target_branch, order);
    let current_full = if state.lambda == 0.0 {
        affine_anchor_moment_vector(
            state.cell.c0,
            state.cell.c1,
            state.cell.left,
            state.cell.right,
            required_degree,
        )
    } else {
        reduced_moments_from_basis(
            state.cell,
            target_branch,
            &state.basis_moments,
            required_degree,
        )?
    };
    let basis_len = match target_branch {
        ExactCellBranch::Quartic => 3,
        ExactCellBranch::Sextic => 5,
        ExactCellBranch::Affine => 5,
    };
    let n_eval = basis_len.max(4);
    let series_len = order + 1;
    let transport_polys = build_transport_series_polynomials(state.cell, delta_c2, delta_c3, order);
    let mut moment_series = vec![0.0; n_eval * series_len];
    for n in 0..n_eval {
        for m in 0..=order {
            let poly = &transport_polys[m];
            let mut coeff = 0.0;
            for (deg, poly_coeff) in poly.iter().enumerate() {
                coeff += poly_coeff * current_full[n + deg];
            }
            moment_series[n * series_len + m] = coeff;
        }
    }

    let mut next_value = state.value;
    let mut step_pow = step;
    let mut value_tail_prev = if order == 1 { state.value.abs() } else { 0.0 };
    let mut value_tail_last = if order == 0 { state.value.abs() } else { 0.0 };
    for m in 0..order {
        let coeff = (delta_c2 * moment_series[2 * series_len + m]
            + delta_c3 * moment_series[3 * series_len + m])
            * INV_TWO_PI
            / (m as f64 + 1.0);
        next_value += coeff * step_pow;
        if m + 1 == order.saturating_sub(1) {
            value_tail_prev = coeff.abs() * step_pow;
        }
        if m + 1 == order {
            value_tail_last = coeff.abs() * step_pow;
        }
        step_pow *= step;
    }

    let mut next_basis = vec![0.0; basis_len];
    let mut basis_tail_prev = [0.0; 5];
    let mut basis_tail_last = [0.0; 5];
    for n in 0..basis_len {
        let row = &moment_series[(n * series_len)..((n + 1) * series_len)];
        let mut acc = 0.0;
        let mut pow = 1.0;
        for (m, coeff) in row.iter().enumerate() {
            acc += coeff * pow;
            if m == order.saturating_sub(1) {
                basis_tail_prev[n] = coeff.abs() * pow;
            }
            if m == order {
                basis_tail_last[n] = coeff.abs() * pow;
            }
            pow *= step;
        }
        next_basis[n] = acc;
    }

    let mut err_est = 0.0_f64;
    let mut scale = next_value.abs().max(1.0);
    err_est = err_est.max(value_tail_last);
    if order > 0 {
        err_est = err_est.max(value_tail_prev);
    }
    for n in 0..basis_len {
        scale = scale.max(next_basis[n].abs());
        err_est = err_est.max(basis_tail_last[n]);
        if order > 0 {
            err_est = err_est.max(basis_tail_prev[n]);
        }
    }
    let tol = abs_tol.max(rel_tol * scale);
    if err_est > tol {
        return Ok(None);
    }

    Ok(Some(TransportCellState {
        lambda: state.lambda + step,
        cell: DenestedCubicCell {
            left: state.cell.left,
            right: state.cell.right,
            c0: state.cell.c0,
            c1: state.cell.c1,
            c2: state.cell.c2 + step * delta_c2,
            c3: state.cell.c3 + step * delta_c3,
        },
        value: next_value,
        basis_moments: next_basis,
    }))
}

fn evaluate_non_affine_cell_state(
    cell: DenestedCubicCell,
    branch: ExactCellBranch,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    let order = 10usize;
    let abs_tol = 1e-14;
    let rel_tol = 1e-12;
    let affine_cell = DenestedCubicCell {
        left: cell.left,
        right: cell.right,
        c0: cell.c0,
        c1: cell.c1,
        c2: 0.0,
        c3: 0.0,
    };
    let affine_state = evaluate_affine_cell_state(
        affine_cell,
        match branch {
            ExactCellBranch::Quartic => 2,
            ExactCellBranch::Sextic => 4,
            ExactCellBranch::Affine => max_degree,
        },
    )?;
    let basis_len = match branch {
        ExactCellBranch::Quartic => 3,
        ExactCellBranch::Sextic => 5,
        ExactCellBranch::Affine => affine_state.moments.len(),
    };
    let mut state = TransportCellState {
        lambda: 0.0,
        cell: affine_cell,
        value: affine_state.value,
        basis_moments: affine_state.moments[..basis_len].to_vec(),
    };

    let delta_c2 = cell.c2;
    let delta_c3 = cell.c3;
    let mut step = 1.0f64;
    while state.lambda < 1.0 - 1e-15 {
        let remaining = 1.0 - state.lambda;
        let trial = step.min(remaining);
        match evaluate_non_affine_transport_step(
            &state, branch, delta_c2, delta_c3, trial, order, abs_tol, rel_tol,
        )? {
            Some(next_state) => {
                state = next_state;
                if state.lambda < 1.0 - 1e-15 {
                    step = (trial * 2.0).min(1.0 - state.lambda);
                }
            }
            None => {
                if trial <= 2.0_f64.powi(-24) {
                    return Err(format!(
                        "non-affine transported cell failed to satisfy Taylor tolerance on [{:.6}, {:.6}] with c2={:.6e}, c3={:.6e}",
                        cell.left, cell.right, cell.c2, cell.c3
                    ));
                }
                step = trial * 0.5;
            }
        }
    }

    let moments = reduced_moments_from_basis(cell, branch, &state.basis_moments, max_degree)?;
    Ok(CellMomentState {
        branch,
        value: state.value,
        moments,
    })
}

/// De-nested cubic cell evaluator.
///
/// Affine cells use the closed-form affine anchor. Quartic and sextic cells
/// are transported from that anchor in the non-affine coefficients `(c2,c3)`
/// with an adaptive Taylor integrator over the homotopy parameter.
pub fn evaluate_cell_moments(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    let left_inf = !cell.left.is_finite();
    let right_inf = !cell.right.is_finite();
    if left_inf || right_inf {
        // Semi-infinite tail cells must be affine: the deviation saturates
        // to a constant outside support, so c2=c3=0.  Both the BVN CDF
        // and the truncated-Gaussian moment vector handle infinite bounds.
        if cell.c2.abs() > NORMALIZED_CELL_BRANCH_TOL || cell.c3.abs() > NORMALIZED_CELL_BRANCH_TOL
        {
            return Err(format!(
                "semi-infinite cell [{}, {}] must be affine (c2=c3=0), got c2={:.3e}, c3={:.3e}",
                cell.left, cell.right, cell.c2, cell.c3
            ));
        }
        return evaluate_affine_cell_state(cell, max_degree);
    }
    if cell.right <= cell.left {
        return Err(format!(
            "finite cell must have left < right, got [{}, {}]",
            cell.left, cell.right
        ));
    }
    let branch = branch_cell(cell)?;
    if branch == ExactCellBranch::Affine {
        return evaluate_affine_cell_state(cell, max_degree);
    }
    evaluate_non_affine_cell_state(cell, branch, max_degree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::normal_pdf;

    fn simpson_integral<F>(left: f64, right: f64, steps: usize, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let n = if steps % 2 == 0 { steps } else { steps + 1 };
        let h = (right - left) / n as f64;
        let mut acc = f(left) + f(right);
        for k in 1..n {
            let x = left + h * k as f64;
            let w = if k % 2 == 0 { 2.0 } else { 4.0 };
            acc += w * f(x);
        }
        acc * h / 3.0
    }

    #[test]
    fn global_transform_preserves_local_span_polynomial() {
        let span = LocalSpanCubic {
            left: -1.2,
            right: 0.8,
            c0: 0.3,
            c1: -0.25,
            c2: 0.11,
            c3: -0.04,
        };
        let (g0, g1, g2, g3) = global_cubic_from_local(span);
        for &x in &[-1.2, -0.7, -0.1, 0.4, 0.8] {
            let local = span.evaluate(x);
            let global = g0 + g1 * x + g2 * x * x + g3 * x * x * x;
            assert!((local - global).abs() < 1e-12);
        }
    }

    #[test]
    fn bivariate_normal_cdf_independent_factorizes() {
        let h = -0.35;
        let k = 0.8;
        let out = bivariate_normal_cdf(h, k, 0.0).expect("bvn");
        let target = normal_cdf(h) * normal_cdf(k);
        assert!((out - target).abs() < 1e-12);
    }

    #[test]
    fn evaluate_affine_cell_state_matches_numeric_integrals() {
        let cell = DenestedCubicCell {
            left: -0.9,
            right: 0.8,
            c0: 0.15,
            c1: -0.35,
            c2: 0.0,
            c3: 0.0,
        };
        let state = evaluate_affine_cell_state(cell, 6).expect("affine cell");
        let value_numeric = simpson_integral(cell.left, cell.right, 4000, |z| {
            super::normal_cdf(cell.eta(z)) * normal_pdf(z)
        });
        assert_eq!(state.branch, ExactCellBranch::Affine);
        assert!((state.value - value_numeric).abs() < 1e-9);
        for degree in 0..=6 {
            let target = simpson_integral(cell.left, cell.right, 4000, |z| {
                z.powi(degree as i32) * (-cell.q(z)).exp()
            });
            assert!((state.moments[degree] - target).abs() < 1e-9);
        }
    }

    #[test]
    fn affine_cell_value_matches_zero_moment_derivative() {
        let cell = DenestedCubicCell {
            left: -1.1,
            right: 0.7,
            c0: 0.23,
            c1: -0.41,
            c2: 0.0,
            c3: 0.0,
        };
        let h = 1e-6;
        let plus = evaluate_affine_cell_state(
            DenestedCubicCell {
                c0: cell.c0 + h,
                ..cell
            },
            0,
        )
        .expect("affine plus");
        let minus = evaluate_affine_cell_state(
            DenestedCubicCell {
                c0: cell.c0 - h,
                ..cell
            },
            0,
        )
        .expect("affine minus");
        let center = evaluate_affine_cell_state(cell, 0).expect("affine center");
        let d_value = (plus.value - minus.value) / (2.0 * h);
        let target = INV_TWO_PI * center.moments[0];
        assert!((d_value - target).abs() < 1e-8);
    }

    #[test]
    fn coefficient_partials_match_exact_span_derivatives() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        for &z in &[-0.75, -0.4, -0.1, 0.2] {
            let u = a + b * z;
            let eta_a = 1.0 + link_span.first_derivative(u);
            let eta_b = z + score_span.evaluate(z) + z * link_span.first_derivative(u);
            assert!((polynomial_value(&dc_da, z) - eta_a).abs() < 1e-12);
            assert!((polynomial_value(&dc_db, z) - eta_b).abs() < 1e-12);
        }
    }

    #[test]
    fn second_coefficient_partials_match_exact_span_derivatives() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let second_partials = denested_cell_second_partials(score_span, link_span, a, b);
        let dc_daa = second_partials.0;
        let dc_dab = second_partials.1;
        let dc_dbb = second_partials.2;
        for &z in &[-0.75, -0.4, -0.1, 0.2] {
            let u = a + b * z;
            let eta_aa = link_span.second_derivative(u);
            let eta_ab = z * link_span.second_derivative(u);
            let eta_bb = z * z * link_span.second_derivative(u);
            assert!((polynomial_value(&dc_daa, z) - eta_aa).abs() < 1e-12);
            assert!((polynomial_value(&dc_dab, z) - eta_ab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbb, z) - eta_bb).abs() < 1e-12);
        }
    }

    #[test]
    fn score_and_link_basis_cell_coefficients_match_direct_construction() {
        let score_basis_span = LocalSpanCubic {
            left: -0.7,
            right: 0.4,
            c0: 0.2,
            c1: -0.04,
            c2: 0.03,
            c3: -0.01,
        };
        let link_basis_span = LocalSpanCubic {
            left: -0.5,
            right: 1.1,
            c0: -0.03,
            c1: 0.05,
            c2: -0.02,
            c3: 0.01,
        };
        let a = 0.25;
        let b = -0.8;
        let score_coeffs = score_basis_cell_coefficients(score_basis_span, b);
        let link_coeffs = link_basis_cell_coefficients(link_basis_span, a, b);
        for &z in &[-0.7, -0.1, 0.2, 0.4] {
            let score_poly = polynomial_value(&score_coeffs, z);
            let link_poly = polynomial_value(&link_coeffs, z);
            assert!((score_poly - b * score_basis_span.evaluate(z)).abs() < 1e-12);
            assert!((link_poly - link_basis_span.evaluate(a + b * z)).abs() < 1e-12);
        }
    }

    #[test]
    fn link_basis_partials_match_exact_span_derivatives() {
        let link_basis_span = LocalSpanCubic {
            left: -0.5,
            right: 1.1,
            c0: -0.03,
            c1: 0.05,
            c2: -0.02,
            c3: 0.01,
        };
        let a = 0.25;
        let b = -0.8;
        let (dc_da, dc_db) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (dc_daa, dc_dab, dc_dbb) = link_basis_cell_second_partials(link_basis_span, a, b);
        for &z in &[-0.6, -0.2, 0.15, 0.5] {
            let u = a + b * z;
            let eta_a = link_basis_span.first_derivative(u);
            let eta_b = z * link_basis_span.first_derivative(u);
            let eta_aa = link_basis_span.second_derivative(u);
            let eta_ab = z * link_basis_span.second_derivative(u);
            let eta_bb = z * z * link_basis_span.second_derivative(u);
            assert!((polynomial_value(&dc_da, z) - eta_a).abs() < 1e-12);
            assert!((polynomial_value(&dc_db, z) - eta_b).abs() < 1e-12);
            assert!((polynomial_value(&dc_daa, z) - eta_aa).abs() < 1e-12);
            assert!((polynomial_value(&dc_dab, z) - eta_ab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbb, z) - eta_bb).abs() < 1e-12);
        }
    }

    #[test]
    fn denested_third_partials_match_exact_span_derivatives() {
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = denested_cell_third_partials(link_span);
        let link_third = 6.0 * link_span.c3;
        for &z in &[-0.75, -0.4, -0.1, 0.2] {
            let eta_aaa = link_third;
            let eta_aab = z * link_third;
            let eta_abb = z * z * link_third;
            let eta_bbb = z * z * z * link_third;
            assert!((polynomial_value(&dc_daaa, z) - eta_aaa).abs() < 1e-12);
            assert!((polynomial_value(&dc_daab, z) - eta_aab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dabb, z) - eta_abb).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbbb, z) - eta_bbb).abs() < 1e-12);
        }
    }

    #[test]
    fn link_basis_third_partials_match_exact_span_derivatives() {
        let link_basis_span = LocalSpanCubic {
            left: -0.5,
            right: 1.1,
            c0: -0.03,
            c1: 0.05,
            c2: -0.02,
            c3: 0.01,
        };
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = link_basis_cell_third_partials(link_basis_span);
        let link_third = 6.0 * link_basis_span.c3;
        for &z in &[-0.6, -0.2, 0.15, 0.5] {
            let eta_aaa = link_third;
            let eta_aab = z * link_third;
            let eta_abb = z * z * link_third;
            let eta_bbb = z * z * z * link_third;
            assert!((polynomial_value(&dc_daaa, z) - eta_aaa).abs() < 1e-12);
            assert!((polynomial_value(&dc_daab, z) - eta_aab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dabb, z) - eta_abb).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbbb, z) - eta_bbb).abs() < 1e-12);
        }
    }

    #[test]
    fn branch_selection_uses_normalized_non_affine_coefficients() {
        let affine = DenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.1,
            c1: -0.4,
            c2: 1e-13,
            c3: -1e-13,
        };
        let quartic = DenestedCubicCell {
            c2: 2e-4,
            c3: 1e-13,
            ..affine
        };
        let sextic = DenestedCubicCell {
            c2: 2e-4,
            c3: 5e-3,
            ..affine
        };
        assert_eq!(branch_cell(affine).unwrap(), ExactCellBranch::Affine);
        assert_eq!(branch_cell(quartic).unwrap(), ExactCellBranch::Quartic);
        assert_eq!(branch_cell(sextic).unwrap(), ExactCellBranch::Sextic);
    }

    #[test]
    fn affine_anchor_moments_match_whole_line_closed_forms() {
        let out = affine_anchor_moment_vector(0.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, 4);
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        assert!((out[0] - sqrt_2pi).abs() < 1e-12);
        assert!(out[1].abs() < 1e-12);
        assert!((out[2] - sqrt_2pi).abs() < 1e-12);
    }

    #[test]
    fn affine_anchor_moments_match_shifted_gaussian_whole_line() {
        let alpha = 0.7;
        let beta = -0.4;
        let out = affine_anchor_moment_vector(alpha, beta, f64::NEG_INFINITY, f64::INFINITY, 4);
        let s = (1.0 + beta * beta).sqrt();
        let mu = -alpha * beta / (1.0 + beta * beta);
        let scale = (-alpha * alpha / (2.0 * s * s)).exp() / s;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        assert!((out[0] - scale * sqrt_2pi).abs() < 1e-12);
        assert!((out[1] - scale * sqrt_2pi * mu).abs() < 1e-12);
        assert!((out[2] - scale * sqrt_2pi * (mu * mu + 1.0 / (s * s))).abs() < 1e-10);
    }

    #[test]
    fn quartic_recurrence_reduces_higher_moments() {
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 0.9,
            c0: 0.2,
            c1: -0.3,
            c2: 0.18,
            c3: 0.0,
        };
        let exact = |k: usize| {
            simpson_integral(cell.left, cell.right, 2000, |z| {
                z.powi(k as i32) * (-cell.q(z)).exp()
            })
        };
        let reduced = reduce_quartic_moments(cell, [exact(0), exact(1), exact(2)], 6)
            .expect("quartic reduction");
        for k in 0..=6 {
            let target = exact(k);
            assert!(
                (reduced[k] - target).abs() < 1e-7,
                "quartic reduced moment M{k} mismatch: {} vs {}",
                reduced[k],
                target
            );
        }
    }

    #[test]
    fn sextic_recurrence_reduces_higher_moments() {
        let cell = DenestedCubicCell {
            left: -0.8,
            right: 0.7,
            c0: -0.1,
            c1: 0.25,
            c2: -0.14,
            c3: 0.22,
        };
        let exact = |k: usize| {
            simpson_integral(cell.left, cell.right, 3000, |z| {
                z.powi(k as i32) * (-cell.q(z)).exp()
            })
        };
        let reduced =
            reduce_sextic_moments(cell, [exact(0), exact(1), exact(2), exact(3), exact(4)], 9)
                .expect("sextic reduction");
        for k in 0..=9 {
            let target = exact(k);
            assert!(
                (reduced[k] - target).abs() < 1e-7,
                "sextic reduced moment M{k} mismatch: {} vs {}",
                reduced[k],
                target
            );
        }
    }

    #[test]
    fn moment_reduced_first_and_second_derivatives_match_numeric_integrals() {
        let cell = DenestedCubicCell {
            left: -0.9,
            right: 0.6,
            c0: 0.15,
            c1: -0.2,
            c2: 0.08,
            c3: 0.17,
        };
        let moments = reduce_sextic_moments(
            cell,
            [
                simpson_integral(cell.left, cell.right, 3000, |z| (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| z * (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| z * z * (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| {
                    z.powi(3) * (-cell.q(z)).exp()
                }),
                simpson_integral(cell.left, cell.right, 3000, |z| {
                    z.powi(4) * (-cell.q(z)).exp()
                }),
            ],
            9,
        )
        .expect("reduced moments");

        let r = [0.7, -0.1, 0.3];
        let s = [0.2, 0.5];
        let second = [0.4, -0.2, 0.1];
        let exact_first = cell_first_derivative_from_moments(&r, &moments).expect("first");
        let exact_second =
            cell_second_derivative_from_moments(cell, &r, &s, &second, &moments).expect("second");

        let numeric_first = simpson_integral(cell.left, cell.right, 3000, |z| {
            polynomial_value(&r, z) * (-cell.q(z)).exp() / (2.0 * std::f64::consts::PI)
        });
        let numeric_second = simpson_integral(cell.left, cell.right, 3000, |z| {
            let eta = cell.eta(z);
            (polynomial_value(&second, z) - eta * polynomial_value(&r, z) * polynomial_value(&s, z))
                * (-cell.q(z)).exp()
                / (2.0 * std::f64::consts::PI)
        });

        assert!((exact_first - numeric_first).abs() < 1e-7);
        assert!((exact_second - numeric_second).abs() < 1e-7);
    }

    #[test]
    fn moment_reduced_third_derivative_matches_numeric_integral() {
        let cell = DenestedCubicCell {
            left: -0.85,
            right: 0.7,
            c0: -0.12,
            c1: 0.18,
            c2: 0.09,
            c3: -0.11,
        };
        let moments = evaluate_cell_moments(cell, 12).expect("cell moments");
        let r = [0.35, -0.12, 0.08];
        let s = [0.17, 0.09];
        let t = [-0.21, 0.14, -0.04];
        let rs = [0.11, -0.07, 0.05];
        let rt = [-0.06, 0.03];
        let st = [0.08, -0.02, 0.01];
        let rst = [0.04, -0.05, 0.02];

        let exact_third = cell_third_derivative_from_moments(
            cell,
            &r,
            &s,
            &t,
            &rs,
            &rt,
            &st,
            &rst,
            &moments.moments,
        )
        .expect("third derivative");
        let numeric_third = simpson_integral(cell.left, cell.right, 4000, |z| {
            let eta = cell.eta(z);
            let rz = polynomial_value(&r, z);
            let sz = polynomial_value(&s, z);
            let tz = polynomial_value(&t, z);
            let rsz = polynomial_value(&rs, z);
            let rtz = polynomial_value(&rt, z);
            let stz = polynomial_value(&st, z);
            let rstz = polynomial_value(&rst, z);
            (rstz - eta * (rsz * tz + rtz * sz + stz * rz) + (eta * eta - 1.0) * rz * sz * tz)
                * (-cell.q(z)).exp()
                / (2.0 * std::f64::consts::PI)
        });

        assert!((exact_third - numeric_third).abs() < 1e-7);
    }

    #[test]
    fn moment_reduced_fourth_derivative_matches_numeric_integral() {
        let cell = DenestedCubicCell {
            left: -0.8,
            right: 0.65,
            c0: 0.11,
            c1: -0.22,
            c2: 0.07,
            c3: 0.13,
        };
        let moments = evaluate_cell_moments(cell, 16).expect("cell moments");
        let r = [0.21, -0.13, 0.06];
        let s = [-0.18, 0.04];
        let t = [0.09, 0.07, -0.03];
        let u = [-0.14, 0.05];
        let rs = [0.08, -0.03, 0.02];
        let rt = [-0.05, 0.01];
        let ru = [0.04, -0.02, 0.01];
        let st = [0.03, 0.02];
        let su = [-0.02, 0.05, -0.01];
        let tu = [0.07, -0.04];
        let rst = [0.03, -0.01, 0.02];
        let rsu = [-0.02, 0.04];
        let rtu = [0.01, 0.02, -0.01];
        let stu = [-0.03, 0.02];
        let rstu = [0.02, -0.01, 0.01];

        let exact_fourth = cell_fourth_derivative_from_moments(
            cell,
            &r,
            &s,
            &t,
            &u,
            &rs,
            &rt,
            &ru,
            &st,
            &su,
            &tu,
            &rst,
            &rsu,
            &rtu,
            &stu,
            &rstu,
            &moments.moments,
        )
        .expect("fourth derivative");
        let numeric_fourth = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let rz = polynomial_value(&r, z);
            let sz = polynomial_value(&s, z);
            let tz = polynomial_value(&t, z);
            let uz = polynomial_value(&u, z);
            let rsz = polynomial_value(&rs, z);
            let rtz = polynomial_value(&rt, z);
            let ruz = polynomial_value(&ru, z);
            let stz = polynomial_value(&st, z);
            let suz = polynomial_value(&su, z);
            let tuz = polynomial_value(&tu, z);
            let rstz = polynomial_value(&rst, z);
            let rsuz = polynomial_value(&rsu, z);
            let rtuz = polynomial_value(&rtu, z);
            let stuz = polynomial_value(&stu, z);
            let rstuz = polynomial_value(&rstu, z);
            let linear =
                rstz * uz + rsuz * tz + rtuz * sz + stuz * rz + rsz * tuz + rtz * suz + ruz * stz;
            let quadratic = rsz * tz * uz
                + rtz * sz * uz
                + ruz * sz * tz
                + stz * rz * uz
                + suz * rz * tz
                + tuz * rz * sz;
            let quartic = rz * sz * tz * uz;
            (rstuz - eta * linear
                + (eta * eta - 1.0) * quadratic
                + (-eta * eta * eta + 3.0 * eta) * quartic)
                * (-cell.q(z)).exp()
                / (2.0 * std::f64::consts::PI)
        });

        assert!((exact_fourth - numeric_fourth).abs() < 2e-7);
    }

    #[test]
    fn denested_cell_parameter_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa, dc_dab, dc_dbb) = denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = denested_cell_third_partials(link_span);
        let zero = [0.0; 4];
        let link_third = 6.0 * link_span.c3;

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_aa = |z: f64| link_span.second_derivative(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_bb = |z: f64| z * z * link_span.second_derivative(a + b * z);
        let eta_aaa = |z: f64| link_third + 0.0 * z;
        let eta_aab = |z: f64| z * link_third;
        let eta_abb = |z: f64| z * z * link_third;
        let eta_bbb = |z: f64| z * z * z * link_third;

        let exact_a = cell_first_derivative_from_moments(&dc_da, &state.moments).expect("a");
        let exact_b = cell_first_derivative_from_moments(&dc_db, &state.moments).expect("b");
        let exact_aa =
            cell_second_derivative_from_moments(cell, &dc_da, &dc_da, &dc_daa, &state.moments)
                .expect("aa");
        let exact_ab =
            cell_second_derivative_from_moments(cell, &dc_da, &dc_db, &dc_dab, &state.moments)
                .expect("ab");
        let exact_bb =
            cell_second_derivative_from_moments(cell, &dc_db, &dc_db, &dc_dbb, &state.moments)
                .expect("bb");
        let exact_aaa = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daaa,
            &state.moments,
        )
        .expect("aaa");
        let exact_aab = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_db,
            &dc_daa,
            &dc_dab,
            &dc_dab,
            &dc_daab,
            &state.moments,
        )
        .expect("aab");
        let exact_abb = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &dc_db,
            &dc_dab,
            &dc_dab,
            &dc_dbb,
            &dc_dabb,
            &state.moments,
        )
        .expect("abb");
        let exact_bbb = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbbb,
            &state.moments,
        )
        .expect("bbb");
        let exact_aaaa = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daaa,
            &dc_daaa,
            &dc_daaa,
            &dc_daaa,
            &zero,
            &state.moments,
        )
        .expect("aaaa");
        let exact_aaab = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_db,
            &dc_daa,
            &dc_daa,
            &dc_dab,
            &dc_daa,
            &dc_dab,
            &dc_dab,
            &dc_daaa,
            &dc_daab,
            &dc_daab,
            &dc_daab,
            &zero,
            &state.moments,
        )
        .expect("aaab");
        let exact_aabb = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_db,
            &dc_db,
            &dc_daa,
            &dc_dab,
            &dc_dab,
            &dc_dab,
            &dc_dab,
            &dc_dbb,
            &dc_daab,
            &dc_daab,
            &dc_dabb,
            &dc_dabb,
            &zero,
            &state.moments,
        )
        .expect("aabb");
        let exact_abbb = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_dab,
            &dc_dab,
            &dc_dab,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dabb,
            &dc_dabb,
            &dc_dabb,
            &dc_dbbb,
            &zero,
            &state.moments,
        )
        .expect("abbb");
        let exact_bbbb = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbbb,
            &dc_dbbb,
            &dc_dbbb,
            &dc_dbbb,
            &zero,
            &state.moments,
        )
        .expect("bbbb");

        let numeric_a = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_a(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_b = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_b(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aa = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_aa(z) - cell.eta(z) * eta_a(z) * eta_a(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ab = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_ab(z) - cell.eta(z) * eta_a(z) * eta_b(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bb = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_bb(z) - cell.eta(z) * eta_b(z) * eta_b(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aaa = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (eta_aaa(z) - 3.0 * eta * eta_aa(z) * eta_a(z) + (eta * eta - 1.0) * eta_a(z).powi(3))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aab = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            (eta_aab(z) - eta * (eta_aa(z) * b_z + 2.0 * eta_ab(z) * a_z)
                + (eta * eta - 1.0) * a_z * a_z * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            (eta_abb(z) - eta * (2.0 * eta_ab(z) * b_z + eta_bb(z) * a_z)
                + (eta * eta - 1.0) * a_z * b_z * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (eta_bbb(z) - 3.0 * eta * eta_bb(z) * eta_b(z) + (eta * eta - 1.0) * eta_b(z).powi(3))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aaaa = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let eta_a_z = eta_a(z);
            let eta_aa_z = eta_aa(z);
            let eta_aaa_z = eta_aaa(z);
            (-eta * (4.0 * eta_aaa_z * eta_a_z + 3.0 * eta_aa_z * eta_aa_z)
                + (eta * eta - 1.0) * (6.0 * eta_aa_z * eta_a_z * eta_a_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_a_z.powi(4))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aaab = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let aa_z = eta_aa(z);
            let ab_z = eta_ab(z);
            let aaa_z = eta_aaa(z);
            let aab_z = eta_aab(z);
            (-eta * (aaa_z * b_z + 3.0 * aab_z * a_z + 3.0 * aa_z * ab_z)
                + (eta * eta - 1.0) * (3.0 * aa_z * a_z * b_z + 3.0 * ab_z * a_z * a_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z.powi(3) * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aabb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let aa_z = eta_aa(z);
            let ab_z = eta_ab(z);
            let bb_z = eta_bb(z);
            let aab_z = eta_aab(z);
            let abb_z = eta_abb(z);
            (-eta * (2.0 * aab_z * b_z + 2.0 * abb_z * a_z + aa_z * bb_z + 2.0 * ab_z * ab_z)
                + (eta * eta - 1.0)
                    * (aa_z * b_z * b_z + 4.0 * ab_z * a_z * b_z + bb_z * a_z * a_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * b_z * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abbb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let ab_z = eta_ab(z);
            let bb_z = eta_bb(z);
            let abb_z = eta_abb(z);
            let bbb_z = eta_bbb(z);
            (-eta * (3.0 * abb_z * b_z + bbb_z * a_z + 3.0 * ab_z * bb_z)
                + (eta * eta - 1.0) * (3.0 * ab_z * b_z * b_z + 3.0 * bb_z * a_z * b_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * b_z.powi(3))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbbb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let eta_b_z = eta_b(z);
            let eta_bb_z = eta_bb(z);
            let eta_bbb_z = eta_bbb(z);
            (-eta * (4.0 * eta_bbb_z * eta_b_z + 3.0 * eta_bb_z * eta_bb_z)
                + (eta * eta - 1.0) * (6.0 * eta_bb_z * eta_b_z * eta_b_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_b_z.powi(4))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_a - numeric_a).abs() < 1e-8);
        assert!((exact_b - numeric_b).abs() < 1e-8);
        assert!((exact_aa - numeric_aa).abs() < 1e-8);
        assert!((exact_ab - numeric_ab).abs() < 1e-8);
        assert!((exact_bb - numeric_bb).abs() < 1e-8);
        assert!((exact_aaa - numeric_aaa).abs() < 2e-7);
        assert!((exact_aab - numeric_aab).abs() < 2e-7);
        assert!((exact_abb - numeric_abb).abs() < 2e-7);
        assert!((exact_bbb - numeric_bbb).abs() < 2e-7);
        assert!((exact_aaaa - numeric_aaaa).abs() < 2e-6);
        assert!((exact_aaab - numeric_aaab).abs() < 2e-6);
        assert!((exact_aabb - numeric_aabb).abs() < 2e-6);
        assert!((exact_abbb - numeric_abbb).abs() < 2e-6);
        assert!((exact_bbbb - numeric_bbbb).abs() < 2e-6);
    }

    #[test]
    fn link_basis_cell_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let link_basis_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: 0.02,
            c1: -0.01,
            c2: 0.03,
            c3: -0.02,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let second_partials = denested_cell_second_partials(score_span, link_span, a, b);
        let dc_daa = second_partials.0;
        let dc_dab = second_partials.1;
        let dc_dbb = second_partials.2;
        let denested_third = denested_cell_third_partials(link_span);
        let dc_daaa = denested_third.0;
        let dc_dbbb = denested_third.3;

        let coeff_w = link_basis_cell_coefficients(link_basis_span, a, b);
        let (coeff_aw, coeff_bw) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (coeff_aaw, coeff_abw, coeff_bbw) =
            link_basis_cell_second_partials(link_basis_span, a, b);
        let link_basis_third = link_basis_cell_third_partials(link_basis_span);
        let coeff_aaaw = link_basis_third.0;
        let coeff_bbbw = link_basis_third.3;
        let zero = [0.0; 4];
        let basis_third = 6.0 * link_basis_span.c3;

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_aa = |z: f64| link_span.second_derivative(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_bb = |z: f64| z * z * link_span.second_derivative(a + b * z);
        let eta_w = |z: f64| link_basis_span.evaluate(a + b * z);
        let eta_aw = |z: f64| link_basis_span.first_derivative(a + b * z);
        let eta_bw = |z: f64| z * link_basis_span.first_derivative(a + b * z);
        let eta_aaw = |z: f64| link_basis_span.second_derivative(a + b * z);
        let eta_abw = |z: f64| z * link_basis_span.second_derivative(a + b * z);
        let eta_bbw = |z: f64| z * z * link_basis_span.second_derivative(a + b * z);
        let eta_aaaw = |z: f64| basis_third + 0.0 * z;
        let eta_bbbw = |z: f64| z * z * z * basis_third;

        let exact_w = cell_first_derivative_from_moments(&coeff_w, &state.moments).expect("w");
        let exact_aw =
            cell_second_derivative_from_moments(cell, &dc_da, &coeff_w, &coeff_aw, &state.moments)
                .expect("aw");
        let exact_bw =
            cell_second_derivative_from_moments(cell, &dc_db, &coeff_w, &coeff_bw, &state.moments)
                .expect("bw");
        let exact_ww =
            cell_second_derivative_from_moments(cell, &coeff_w, &coeff_w, &zero, &state.moments)
                .expect("ww");
        let exact_aaw = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_w,
            &dc_daa,
            &coeff_aw,
            &coeff_aw,
            &coeff_aaw,
            &state.moments,
        )
        .expect("aaw");
        let exact_abw = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_w,
            &dc_dab,
            &coeff_aw,
            &coeff_bw,
            &coeff_abw,
            &state.moments,
        )
        .expect("abw");
        let exact_bbw = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_w,
            &dc_dbb,
            &coeff_bw,
            &coeff_bw,
            &coeff_bbw,
            &state.moments,
        )
        .expect("bbw");
        let exact_www = cell_third_derivative_from_moments(
            cell,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("www");
        let exact_aaaw = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &coeff_w,
            &dc_daa,
            &dc_daa,
            &coeff_aw,
            &dc_daa,
            &coeff_aw,
            &coeff_aw,
            &dc_daaa,
            &coeff_aaw,
            &coeff_aaw,
            &coeff_aaw,
            &coeff_aaaw,
            &state.moments,
        )
        .expect("aaaw");
        let exact_aaww = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_w,
            &coeff_w,
            &dc_daa,
            &coeff_aw,
            &coeff_aw,
            &coeff_aw,
            &coeff_aw,
            &zero,
            &coeff_aaw,
            &coeff_aaw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("aaww");
        let exact_abww = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_w,
            &coeff_w,
            &dc_dab,
            &coeff_aw,
            &coeff_aw,
            &coeff_bw,
            &coeff_bw,
            &zero,
            &coeff_abw,
            &coeff_abw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("abww");
        let exact_bbww = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_w,
            &coeff_w,
            &dc_dbb,
            &coeff_bw,
            &coeff_bw,
            &coeff_bw,
            &coeff_bw,
            &zero,
            &coeff_bbw,
            &coeff_bbw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bbww");
        let exact_bbbw = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &coeff_w,
            &dc_dbb,
            &dc_dbb,
            &coeff_bw,
            &dc_dbb,
            &coeff_bw,
            &coeff_bw,
            &dc_dbbb,
            &coeff_bbw,
            &coeff_bbw,
            &coeff_bbw,
            &coeff_bbbw,
            &state.moments,
        )
        .expect("bbbw");
        let exact_wwww = cell_fourth_derivative_from_moments(
            cell,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("wwww");

        let numeric_w = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_w(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aw = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_aw(z) - cell.eta(z) * eta_a(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bw = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_bw(z) - cell.eta(z) * eta_b(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ww = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_w(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aaw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            let a_z = eta_a(z);
            (eta_aaw(z) - eta * (eta_aa(z) * w_z + 2.0 * eta_aw(z) * a_z)
                + (eta * eta - 1.0) * a_z * a_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            (eta_abw(z) - eta * (eta_ab(z) * w_z + eta_aw(z) * b_z + eta_bw(z) * a_z)
                + (eta * eta - 1.0) * a_z * b_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            let b_z = eta_b(z);
            (eta_bbw(z) - eta * (eta_bb(z) * w_z + 2.0 * eta_bw(z) * b_z)
                + (eta * eta - 1.0) * b_z * b_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_www = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            ((eta * eta - 1.0) * w_z * w_z * w_z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aaaw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let w_z = eta_w(z);
            let aa_z = eta_aa(z);
            let aw_z = eta_aw(z);
            (eta_aaaw(z)
                - eta * ((dc_daaa[0] + 0.0 * z) * w_z + 3.0 * eta_aaw(z) * a_z + 3.0 * aa_z * aw_z)
                + (eta * eta - 1.0) * (3.0 * aa_z * a_z * w_z + 3.0 * aw_z * a_z * a_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * a_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aaww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let w_z = eta_w(z);
            let aw_z = eta_aw(z);
            (-(2.0 * eta * (eta_aaw(z) * w_z + aw_z * aw_z))
                + (eta * eta - 1.0) * (eta_aa(z) * w_z * w_z + 4.0 * aw_z * a_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let w_z = eta_w(z);
            let aw_z = eta_aw(z);
            let bw_z = eta_bw(z);
            (-(2.0 * eta * (eta_abw(z) * w_z + aw_z * bw_z))
                + (eta * eta - 1.0)
                    * (eta_ab(z) * w_z * w_z + 2.0 * aw_z * b_z * w_z + 2.0 * bw_z * a_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * b_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let w_z = eta_w(z);
            let bw_z = eta_bw(z);
            (-(2.0 * eta * (eta_bbw(z) * w_z + bw_z * bw_z))
                + (eta * eta - 1.0) * (eta_bb(z) * w_z * w_z + 4.0 * bw_z * b_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbbw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let w_z = eta_w(z);
            let bb_z = eta_bb(z);
            let bw_z = eta_bw(z);
            (eta_bbbw(z)
                - eta
                    * ((dc_dbbb[3] * z * z * z) * w_z + 3.0 * eta_bbw(z) * b_z + 3.0 * bb_z * bw_z)
                + (eta * eta - 1.0) * (3.0 * bb_z * b_z * w_z + 3.0 * bw_z * b_z * b_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * b_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_wwww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            ((-eta * eta * eta + 3.0 * eta) * w_z * w_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_w - numeric_w).abs() < 1e-8);
        assert!((exact_aw - numeric_aw).abs() < 1e-7);
        assert!((exact_bw - numeric_bw).abs() < 1e-7);
        assert!((exact_ww - numeric_ww).abs() < 1e-7);
        assert!((exact_aaw - numeric_aaw).abs() < 2e-6);
        assert!((exact_abw - numeric_abw).abs() < 2e-6);
        assert!((exact_bbw - numeric_bbw).abs() < 2e-6);
        assert!((exact_www - numeric_www).abs() < 2e-6);
        assert!((exact_aaaw - numeric_aaaw).abs() < 3e-6);
        assert!((exact_aaww - numeric_aaww).abs() < 3e-6);
        assert!((exact_abww - numeric_abww).abs() < 3e-6);
        assert!((exact_bbww - numeric_bbww).abs() < 3e-6);
        assert!((exact_bbbw - numeric_bbbw).abs() < 3e-6);
        assert!((exact_wwww - numeric_wwww).abs() < 3e-6);
    }

    #[test]
    fn score_basis_cell_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let score_basis_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: -0.04,
            c1: 0.06,
            c2: -0.01,
            c3: 0.02,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let second_partials = denested_cell_second_partials(score_span, link_span, a, b);
        let dc_daa = second_partials.0;
        let dc_dab = second_partials.1;
        let dc_dbb = second_partials.2;
        let denested_third = denested_cell_third_partials(link_span);
        let dc_dbbb = denested_third.3;

        let coeff_h = score_basis_cell_coefficients(score_basis_span, b);
        let coeff_bh = score_basis_cell_coefficients(score_basis_span, 1.0);
        let zero = [0.0; 4];

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_bb = |z: f64| z * z * link_span.second_derivative(a + b * z);
        let eta_h = |z: f64| b * score_basis_span.evaluate(z);
        let eta_bh = |z: f64| score_basis_span.evaluate(z);

        let exact_h = cell_first_derivative_from_moments(&coeff_h, &state.moments).expect("h");
        let exact_ah =
            cell_second_derivative_from_moments(cell, &dc_da, &coeff_h, &zero, &state.moments)
                .expect("ah");
        let exact_bh =
            cell_second_derivative_from_moments(cell, &dc_db, &coeff_h, &coeff_bh, &state.moments)
                .expect("bh");
        let exact_hh =
            cell_second_derivative_from_moments(cell, &coeff_h, &coeff_h, &zero, &state.moments)
                .expect("hh");
        let exact_abh = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_h,
            &dc_dab,
            &zero,
            &coeff_bh,
            &zero,
            &state.moments,
        )
        .expect("abh");
        let exact_bbh = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_h,
            &dc_dbb,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &state.moments,
        )
        .expect("bbh");
        let exact_bhh = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhh");
        let exact_hhh = cell_third_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhh");
        let exact_bbbh = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &coeff_h,
            &dc_dbb,
            &dc_dbb,
            &coeff_bh,
            &dc_dbb,
            &coeff_bh,
            &coeff_bh,
            &dc_dbbb,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bbbh");
        let exact_aahh = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_h,
            &coeff_h,
            &dc_daa,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("aahh");
        let exact_abhh = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &dc_dab,
            &zero,
            &zero,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("abhh");
        let exact_bbhh = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &dc_dbb,
            &coeff_bh,
            &coeff_bh,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bbhh");
        let exact_bhhh = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &coeff_bh,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhhh");
        let exact_hhhh = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhhh");

        let numeric_h = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_h(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ah = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_a(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bh = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_bh(z) - cell.eta(z) * eta_b(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_hh = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_h(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_abh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * (eta_ab(z) * eta_h(z) + eta_bh(z) * eta_a(z)))
                + (eta * eta - 1.0) * eta_a(z) * eta_b(z) * eta_h(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * (eta_bb(z) * eta_h(z) + 2.0 * eta_bh(z) * eta_b(z)))
                + (eta * eta - 1.0) * eta_b(z) * eta_b(z) * eta_h(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(2.0 * eta * eta_bh(z) * eta_h(z))
                + (eta * eta - 1.0) * eta_b(z) * eta_h(z) * eta_h(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((eta * eta - 1.0) * eta_h(z) * eta_h(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bbbh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let h_z = eta_h(z);
            let bb_z = eta_bb(z);
            let bh_z = eta_bh(z);
            (-(eta * ((dc_dbbb[3] * z * z * z) * h_z + 3.0 * bb_z * bh_z))
                + (eta * eta - 1.0) * (3.0 * bb_z * b_z * h_z + 3.0 * bh_z * b_z * b_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * b_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aahh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let h_z = eta_h(z);
            ((eta * eta - 1.0) * polynomial_value(&dc_daa, z) * h_z * h_z
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let h_z = eta_h(z);
            ((eta * eta - 1.0) * (eta_ab(z) * h_z * h_z + 2.0 * eta_bh(z) * a_z * h_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * b_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let h_z = eta_h(z);
            let bh_z = eta_bh(z);
            (-(2.0 * eta * bh_z * bh_z)
                + (eta * eta - 1.0) * (eta_bb(z) * h_z * h_z + 4.0 * bh_z * b_z * h_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let h_z = eta_h(z);
            (-(eta * (3.0 * eta_bh(z) * h_z * h_z))
                + (eta * eta - 1.0) * (3.0 * eta_bh(z) * h_z * h_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_b(z) * h_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let h_z = eta_h(z);
            ((-eta * eta * eta + 3.0 * eta) * h_z * h_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_h - numeric_h).abs() < 1e-8);
        assert!((exact_ah - numeric_ah).abs() < 1e-7);
        assert!((exact_bh - numeric_bh).abs() < 1e-7);
        assert!((exact_hh - numeric_hh).abs() < 1e-7);
        assert!((exact_abh - numeric_abh).abs() < 2e-6);
        assert!((exact_bbh - numeric_bbh).abs() < 2e-6);
        assert!((exact_bhh - numeric_bhh).abs() < 2e-6);
        assert!((exact_hhh - numeric_hhh).abs() < 2e-6);
        assert!((exact_bbbh - numeric_bbbh).abs() < 3e-6);
        assert!((exact_aahh - numeric_aahh).abs() < 3e-6);
        assert!((exact_abhh - numeric_abhh).abs() < 3e-6);
        assert!((exact_bbhh - numeric_bbhh).abs() < 3e-6);
        assert!((exact_bhhh - numeric_bhhh).abs() < 3e-6);
        assert!((exact_hhhh - numeric_hhhh).abs() < 3e-6);
    }

    #[test]
    fn cross_basis_cell_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let score_basis_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: -0.04,
            c1: 0.06,
            c2: -0.01,
            c3: 0.02,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let link_basis_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: 0.02,
            c1: -0.01,
            c2: 0.03,
            c3: -0.02,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa, dc_dab, _) = denested_cell_second_partials(score_span, link_span, a, b);

        let coeff_h = score_basis_cell_coefficients(score_basis_span, b);
        let coeff_bh = score_basis_cell_coefficients(score_basis_span, 1.0);
        let coeff_w = link_basis_cell_coefficients(link_basis_span, a, b);
        let (coeff_aw, coeff_bw) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (coeff_aaw, coeff_abw, _) = link_basis_cell_second_partials(link_basis_span, a, b);
        let zero = [0.0; 4];

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_h = |z: f64| b * score_basis_span.evaluate(z);
        let eta_bh = |z: f64| score_basis_span.evaluate(z);
        let eta_w = |z: f64| link_basis_span.evaluate(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_aw = |z: f64| link_basis_span.first_derivative(a + b * z);
        let eta_bw = |z: f64| z * link_basis_span.first_derivative(a + b * z);

        let exact_hw =
            cell_second_derivative_from_moments(cell, &coeff_h, &coeff_w, &zero, &state.moments)
                .expect("hw");
        let exact_ahw = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &coeff_h,
            &coeff_w,
            &zero,
            &coeff_aw,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("ahw");
        let exact_bhw = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_w,
            &coeff_bh,
            &coeff_bw,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhw");
        let exact_hhw = cell_third_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhw");
        let exact_hww = cell_third_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hww");
        let exact_aahw = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_h,
            &coeff_w,
            &dc_daa,
            &zero,
            &coeff_aw,
            &zero,
            &coeff_aw,
            &zero,
            &zero,
            &coeff_aaw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("aahw");
        let exact_hhww = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhww");
        let exact_hhhw = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhhw");
        let exact_abhw = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_h,
            &coeff_w,
            &dc_dab,
            &zero,
            &coeff_aw,
            &coeff_bh,
            &coeff_bw,
            &zero,
            &zero,
            &coeff_abw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("abhw");
        let exact_ahww = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &zero,
            &coeff_aw,
            &coeff_aw,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("ahww");
        let exact_bhww = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &coeff_bh,
            &coeff_bw,
            &coeff_bw,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhww");
        let exact_hwww = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hwww");

        let numeric_hw = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_h(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ahw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * eta_aw(z) * eta_h(z)) + (eta * eta - 1.0) * eta_a(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * (eta_bh(z) * eta_w(z) + eta_bw(z) * eta_h(z)))
                + (eta * eta - 1.0) * eta_b(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((eta * eta - 1.0) * eta_h(z) * eta_h(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_hww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((eta * eta - 1.0) * eta_h(z) * eta_w(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aahw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * polynomial_value(&coeff_aaw, z) * eta_h(z))
                + (eta * eta - 1.0)
                    * (polynomial_value(&dc_daa, z) * eta_h(z) * eta_w(z)
                        + 2.0 * eta_aw(z) * eta_a(z) * eta_h(z))
                + (-eta * eta * eta + 3.0 * eta) * eta_a(z) * eta_a(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((-eta * eta * eta + 3.0 * eta) * eta_h(z) * eta_h(z) * eta_w(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((-eta * eta * eta + 3.0 * eta) * eta_h(z) * eta_h(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * polynomial_value(&coeff_abw, z) * eta_h(z) + eta * eta_aw(z) * eta_bh(z))
                + (eta * eta - 1.0)
                    * (eta_ab(z) * eta_h(z) * eta_w(z)
                        + eta_aw(z) * eta_b(z) * eta_h(z)
                        + eta_bh(z) * eta_a(z) * eta_w(z)
                        + eta_bw(z) * eta_a(z) * eta_h(z))
                + (-eta * eta * eta + 3.0 * eta) * eta_a(z) * eta_b(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_ahww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (2.0 * (eta * eta - 1.0) * eta_aw(z) * eta_h(z) * eta_w(z)
                + (-eta * eta * eta + 3.0 * eta) * eta_a(z) * eta_h(z) * eta_w(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let h_z = eta_h(z);
            let w_z = eta_w(z);
            ((eta * eta - 1.0) * (eta_bh(z) * w_z * w_z + 2.0 * eta_bw(z) * h_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_b(z) * h_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hwww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((-eta * eta * eta + 3.0 * eta) * eta_h(z) * eta_w(z) * eta_w(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_hw - numeric_hw).abs() < 1e-7);
        assert!((exact_ahw - numeric_ahw).abs() < 2e-6);
        assert!((exact_bhw - numeric_bhw).abs() < 2e-6);
        assert!((exact_hhw - numeric_hhw).abs() < 2e-6);
        assert!((exact_hww - numeric_hww).abs() < 2e-6);
        assert!((exact_aahw - numeric_aahw).abs() < 3e-6);
        assert!((exact_hhww - numeric_hhww).abs() < 3e-6);
        assert!((exact_hhhw - numeric_hhhw).abs() < 3e-6);
        assert!((exact_abhw - numeric_abhw).abs() < 3e-6);
        assert!((exact_ahww - numeric_ahww).abs() < 3e-6);
        assert!((exact_bhww - numeric_bhww).abs() < 3e-6);
        assert!((exact_hwww - numeric_hwww).abs() < 3e-6);
    }

    #[test]
    fn evaluate_cell_moments_matches_numeric_integrals() {
        let cell = DenestedCubicCell {
            left: -0.9,
            right: 0.8,
            c0: 0.15,
            c1: -0.35,
            c2: 0.11,
            c3: -0.07,
        };
        let state = evaluate_cell_moments(cell, 6).expect("cell moments");
        let value_numeric = simpson_integral(cell.left, cell.right, 4000, |z| {
            super::normal_cdf(cell.eta(z)) * normal_pdf(z)
        });
        assert!((state.value - value_numeric).abs() < 1e-9);
        for degree in 0..=6 {
            let target = simpson_integral(cell.left, cell.right, 4000, |z| {
                z.powi(degree as i32) * (-cell.q(z)).exp()
            });
            assert!((state.moments[degree] - target).abs() < 1e-9);
        }
    }

    #[test]
    fn partition_builder_moves_link_preimages_with_intercept() {
        let score_breaks = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let link_breaks = [-1.5, -0.5, 0.5, 1.5];
        let score_span = |z: f64| {
            let left = if z < -1.0 {
                -2.0
            } else if z < 0.0 {
                -1.0
            } else if z < 1.0 {
                0.0
            } else {
                1.0
            };
            Ok(LocalSpanCubic {
                left,
                right: left + 1.0,
                c0: 0.1,
                c1: 0.2,
                c2: 0.0,
                c3: 0.0,
            })
        };
        let link_span = |u: f64| {
            let left = if u < -0.5 {
                -1.5
            } else if u < 0.5 {
                -0.5
            } else {
                0.5
            };
            Ok(LocalSpanCubic {
                left,
                right: left + 1.0,
                c0: -0.05,
                c1: 0.1,
                c2: 0.0,
                c3: 0.0,
            })
        };
        let cells_a0 = build_denested_partition_cells(
            0.25,
            0.9,
            &score_breaks,
            &link_breaks,
            score_span,
            link_span,
        )
        .expect("cells a0");
        let cells_a1 = build_denested_partition_cells(
            0.55,
            0.9,
            &score_breaks,
            &link_breaks,
            score_span,
            link_span,
        )
        .expect("cells a1");
        assert!(cells_a0.len() >= score_breaks.len() - 1);
        assert!(
            cells_a0
                .windows(2)
                .all(|w| (w[0].cell.right - w[1].cell.left).abs() <= 1e-12)
        );
        assert!(
            cells_a0
                .iter()
                .zip(cells_a1.iter())
                .any(|(lhs, rhs)| (lhs.cell.left - rhs.cell.left).abs() > 1e-10)
        );
        assert!(cells_a0.first().unwrap().cell.left.is_infinite());
        assert!(cells_a0.last().unwrap().cell.right.is_infinite());
    }

    #[test]
    fn partition_builder_without_breaks_returns_single_global_cell() {
        let cells = build_denested_partition_cells_with_tails(
            0.3,
            -0.4,
            &[],
            &[],
            |z| {
                if z.is_nan() {
                    return Err("probe z is NaN".to_string());
                }
                Ok(LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                })
            },
            |u| {
                if u.is_nan() {
                    return Err("probe u is NaN".to_string());
                }
                Ok(LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                })
            },
        )
        .expect("global cell");
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].cell.left, f64::NEG_INFINITY);
        assert_eq!(cells[0].cell.right, f64::INFINITY);
        assert!(cells[0].cell.c2.abs() < 1e-12);
        assert!(cells[0].cell.c3.abs() < 1e-12);
    }

    #[test]
    fn polynomial_integral_helper_matches_moment_sum() {
        let cell = DenestedCubicCell {
            left: -1.5,
            right: 1.25,
            c0: 0.2,
            c1: -0.4,
            c2: 0.15,
            c3: 0.03,
        };
        let state = evaluate_cell_moments(cell, 8).expect("cell moments");
        let coeffs = [1.5, -0.25, 0.75, 0.1];
        let expected = INV_TWO_PI
            * coeffs
                .iter()
                .enumerate()
                .map(|(idx, coeff)| coeff * state.moments[idx])
                .sum::<f64>();
        let got = cell_polynomial_integral_from_moments(&coeffs, &state.moments, "test poly")
            .expect("poly integral");
        assert!((got - expected).abs() < 1e-14);
    }
}
