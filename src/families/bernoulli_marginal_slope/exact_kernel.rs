use crate::probability::normal_cdf;

// Internal exact de-nested cubic cell kernel for Bernoulli marginal-slope.
// This is shared by fit-time family code and saved-model prediction, but it is
// not a standalone family surface and should not be part of the public crate API.

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

pub const ANCHORED_DEVIATION_KERNEL: &str = "ExactDenestedCubicV1";
pub const LEGACY_ANCHORED_DEVIATION_KERNEL: &str = "ExactDenestedCubic";
pub const NORMALIZED_CELL_BRANCH_TOL: f64 = 1e-10;

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

impl DenestedPartitionCell {
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellMomentState {
    pub branch: ExactCellBranch,
    pub value: f64,
    pub moments: Vec<f64>,
}

const BVN_GL_NODES_10: [f64; 10] = [
    -0.973_906_528_517_171_7,
    -0.865_063_366_688_984_5,
    -0.679_409_568_299_024_4,
    -0.433_395_394_129_247_2,
    -0.148_874_338_981_631_2,
    0.148_874_338_981_631_2,
    0.433_395_394_129_247_2,
    0.679_409_568_299_024_4,
    0.865_063_366_688_984_5,
    0.973_906_528_517_171_7,
];

const BVN_GL_WEIGHTS_10: [f64; 10] = [
    0.066_671_344_308_688_1,
    0.149_451_349_150_580_6,
    0.219_086_362_515_982,
    0.269_266_719_309_996_3,
    0.295_524_224_714_752_9,
    0.295_524_224_714_752_9,
    0.269_266_719_309_996_3,
    0.219_086_362_515_982,
    0.149_451_349_150_580_6,
    0.066_671_344_308_688_1,
];

fn dedup_sorted_breakpoints(points: &mut Vec<f64>) {
    points.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    points.dedup_by(|lhs, rhs| (*lhs - *rhs).abs() <= 1e-12);
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
    [c0 * c1, 1.0 + c1 * c1 + 2.0 * c0 * c2, 3.0 * c1 * c2, 2.0 * c2 * c2]
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
    if derivative_coefficients.len() > moments.len() {
        return Err(format!(
            "insufficient reduced moments for first derivative: need {}, have {}",
            derivative_coefficients.len(),
            moments.len()
        ));
    }
    Ok(derivative_coefficients
        .iter()
        .enumerate()
        .map(|(idx, coeff)| coeff * moments[idx])
        .sum::<f64>()
        / (2.0 * std::f64::consts::PI))
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
    let second_term = second_coefficients_rs
        .iter()
        .enumerate()
        .map(|(idx, coeff)| coeff * moments[idx])
        .sum::<f64>();
    let mut product_coefficients = vec![0.0; product_degree - 3 + 1];
    for (i, &lhs) in first_coefficients_r.iter().enumerate() {
        for (j, &rhs) in first_coefficients_s.iter().enumerate() {
            product_coefficients[i + j] += lhs * rhs;
        }
    }
    let cubic = [cell.c0, cell.c1, cell.c2, cell.c3];
    let mut eta_product = vec![0.0; product_degree + 1];
    for (i, &eta_coeff) in cubic.iter().enumerate() {
        for (j, &prod_coeff) in product_coefficients.iter().enumerate() {
            eta_product[i + j] += eta_coeff * prod_coeff;
        }
    }
    let eta_term = eta_product
        .iter()
        .enumerate()
        .map(|(idx, coeff)| coeff * moments[idx])
        .sum::<f64>();
    Ok((second_term - eta_term) / (2.0 * std::f64::consts::PI))
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
    let d1 =
        b * (link_span.c1 + 2.0 * link_span.c2 * shift + 3.0 * link_span.c3 * shift * shift);
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
    let dc_dab = [0.0, 2.0 * alpha2 + 6.0 * alpha3 * shift, 6.0 * alpha3 * b, 0.0];
    let dc_dbb = [
        0.0,
        0.0,
        2.0 * (alpha2 + 3.0 * alpha3 * shift),
        6.0 * alpha3 * b,
    ];
    (dc_daa, dc_dab, dc_dbb)
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

#[cfg(test)]
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
    let dc_dab = [0.0, 2.0 * alpha2 + 6.0 * alpha3 * shift, 6.0 * alpha3 * b, 0.0];
    let dc_dbb = [0.0, 0.0, 2.0 * (alpha2 + 3.0 * alpha3 * shift), 6.0 * alpha3 * b];
    (dc_daa, dc_dab, dc_dbb)
}

pub fn build_denested_partition_cells<FS, FL>(
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
    if score_breaks.len() < 2 {
        return Err("de-nested partition requires at least two score breakpoints".to_string());
    }
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
    if split_points.len() < 2 {
        return Err("de-nested partition did not produce a valid breakpoint set".to_string());
    }

    let mut out = Vec::new();
    for window in split_points.windows(2) {
        let left = window[0];
        let right = window[1];
        if !left.is_finite() || !right.is_finite() || right - left <= 1e-12 {
            continue;
        }
        let mid = 0.5 * (left + right);
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
    if out.is_empty() {
        return Err("de-nested partition produced no active cells".to_string());
    }
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
        return Err(format!("bivariate normal cdf requires finite correlation, got {rho}"));
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
    for (&node, &weight) in BVN_GL_NODES_10.iter().zip(BVN_GL_WEIGHTS_10.iter()) {
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
            acc += binomial_coefficient(n, k) * mu.powi((n - k) as i32) * s.powi(-(k as i32)) * t[k];
        }
        out[n] = anchor * acc;
    }
    out
}

pub fn evaluate_affine_cell_state(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    let alpha = cell.c0;
    let beta = cell.c1;
    let s = (1.0 + beta * beta).sqrt();
    let second = alpha / s;
    let rho = -beta / s;
    let right = bivariate_normal_cdf(cell.right, second, rho)?;
    let left = bivariate_normal_cdf(cell.left, second, rho)?;
    let value = (right - left).clamp(0.0, 1.0);
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
fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * k as f64)
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
            let term = poly_mul(&a_powers[p], &b_powers[q]);
            let sign = if (p + q) % 2 == 0 { 1.0 } else { -1.0 };
            let scale = sign / (factorial(p) * factorial(q) * 2.0_f64.powi(q as i32));
            for (idx, value) in term.iter().enumerate() {
                poly[idx] += scale * value;
            }
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
        reduced_moments_from_basis(state.cell, target_branch, &state.basis_moments, required_degree)?
    };
    let basis_len = match target_branch {
        ExactCellBranch::Quartic => 3,
        ExactCellBranch::Sextic => 5,
        ExactCellBranch::Affine => 5,
    };
    let n_eval = basis_len.max(4);
    let transport_polys =
        build_transport_series_polynomials(state.cell, delta_c2, delta_c3, order);
    let mut moment_series = vec![vec![0.0; order + 1]; n_eval];
    for n in 0..n_eval {
        for m in 0..=order {
            let poly = &transport_polys[m];
            let mut coeff = 0.0;
            for (deg, poly_coeff) in poly.iter().enumerate() {
                coeff += poly_coeff * current_full[n + deg];
            }
            moment_series[n][m] = coeff;
        }
    }

    let mut value_series = vec![0.0; order + 1];
    value_series[0] = state.value;
    for m in 0..order {
        value_series[m + 1] =
            (delta_c2 * moment_series[2][m] + delta_c3 * moment_series[3][m])
                / (2.0 * std::f64::consts::PI * (m as f64 + 1.0));
    }

    let mut next_value = 0.0;
    let mut step_pow = 1.0;
    for coeff in &value_series {
        next_value += coeff * step_pow;
        step_pow *= step;
    }

    let mut next_basis = vec![0.0; basis_len];
    for n in 0..basis_len {
        let mut acc = 0.0;
        let mut pow = 1.0;
        for coeff in &moment_series[n] {
            acc += coeff * pow;
            pow *= step;
        }
        next_basis[n] = acc;
    }

    let mut err_est = 0.0_f64;
    let mut scale = next_value.abs().max(1.0);
    let last_pow = step.powi(order as i32);
    let prev_pow = if order > 0 {
        step.powi((order - 1) as i32)
    } else {
        1.0
    };
    err_est = err_est.max(value_series[order].abs() * last_pow);
    if order > 0 {
        err_est = err_est.max(value_series[order - 1].abs() * prev_pow);
    }
    for n in 0..basis_len {
        scale = scale.max(next_basis[n].abs());
        err_est = err_est.max(moment_series[n][order].abs() * last_pow);
        if order > 0 {
            err_est = err_est.max(moment_series[n][order - 1].abs() * prev_pow);
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
            &state,
            branch,
            delta_c2,
            delta_c3,
            trial,
            order,
            abs_tol,
            rel_tol,
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

/// Exact de-nested cubic cell evaluator.
///
/// Affine cells use the closed-form affine anchor. Quartic and sextic cells
/// are transported from that anchor in the non-affine coefficients `(c2,c3)`
/// with an adaptive Taylor integrator over the homotopy parameter.
pub fn evaluate_cell_moments(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    if !cell.left.is_finite() || !cell.right.is_finite() || cell.right <= cell.left {
        return Err(format!(
            "cell moment evaluation requires finite ordered bounds, got [{}, {}]",
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
        let value_numeric =
            simpson_integral(cell.left, cell.right, 4000, |z| super::normal_cdf(cell.eta(z)) * normal_pdf(z));
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
    fn coefficient_partials_match_finite_differences() {
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
        let eps = 1e-6;
        let coeffs = |aa: f64, bb: f64| denested_cell_coefficients(score_span, link_span, aa, bb);
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let plus_a = coeffs(a + eps, b);
        let minus_a = coeffs(a - eps, b);
        let plus_b = coeffs(a, b + eps);
        let minus_b = coeffs(a, b - eps);
        for j in 0..4 {
            let fd_a = (plus_a[j] - minus_a[j]) / (2.0 * eps);
            let fd_b = (plus_b[j] - minus_b[j]) / (2.0 * eps);
            assert!((dc_da[j] - fd_a).abs() < 1e-6);
            assert!((dc_db[j] - fd_b).abs() < 1e-6);
        }
    }

    #[test]
    fn second_coefficient_partials_match_finite_differences() {
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
        let eps = 1e-6;
        let coeffs = |aa: f64, bb: f64| denested_cell_coefficients(score_span, link_span, aa, bb);
        let (dc_daa, dc_dab, dc_dbb) = denested_cell_second_partials(score_span, link_span, a, b);
        let (da_plus, db_plus) = denested_cell_coefficient_partials(score_span, link_span, a + eps, b);
        let (da_minus, db_minus) = denested_cell_coefficient_partials(score_span, link_span, a - eps, b);
        let (da_bplus, db_bplus) = denested_cell_coefficient_partials(score_span, link_span, a, b + eps);
        let (da_bminus, db_bminus) = denested_cell_coefficient_partials(score_span, link_span, a, b - eps);
        let plus_aa = coeffs(a + eps, b);
        let base = coeffs(a, b);
        let minus_aa = coeffs(a - eps, b);
        let plus_bb = coeffs(a, b + eps);
        let minus_bb = coeffs(a, b - eps);
        for j in 0..4 {
            let fd_aa = (plus_aa[j] - 2.0 * base[j] + minus_aa[j]) / (eps * eps);
            let fd_ab = (da_bplus[j] - da_bminus[j]) / (2.0 * eps);
            let fd_ab_alt = (db_plus[j] - db_minus[j]) / (2.0 * eps);
            let fd_bb = (plus_bb[j] - 2.0 * base[j] + minus_bb[j]) / (eps * eps);
            assert!((dc_daa[j] - fd_aa).abs() < 2e-5);
            assert!((dc_dab[j] - fd_ab).abs() < 2e-5);
            assert!((dc_dab[j] - fd_ab_alt).abs() < 2e-5);
            assert!((dc_dbb[j] - fd_bb).abs() < 2e-5);
            assert!((da_plus[j] - da_minus[j]).is_finite());
            assert!((db_bplus[j] - db_bminus[j]).is_finite());
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
    fn link_basis_partials_match_finite_differences() {
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
        let eps = 1e-6;
        let coeffs = |aa: f64, bb: f64| link_basis_cell_coefficients(link_basis_span, aa, bb);
        let (dc_da, dc_db) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (dc_daa, dc_dab, dc_dbb) = link_basis_cell_second_partials(link_basis_span, a, b);
        let plus_a = coeffs(a + eps, b);
        let minus_a = coeffs(a - eps, b);
        let plus_b = coeffs(a, b + eps);
        let minus_b = coeffs(a, b - eps);
        let (da_bplus, db_plus) = link_basis_cell_coefficient_partials(link_basis_span, a, b + eps);
        let (da_bminus, db_minus) = link_basis_cell_coefficient_partials(link_basis_span, a, b - eps);
        let (da_plus, _) = link_basis_cell_coefficient_partials(link_basis_span, a + eps, b);
        let (da_minus, _) = link_basis_cell_coefficient_partials(link_basis_span, a - eps, b);
        let base = coeffs(a, b);
        for j in 0..4 {
            let fd_a = (plus_a[j] - minus_a[j]) / (2.0 * eps);
            let fd_b = (plus_b[j] - minus_b[j]) / (2.0 * eps);
            let fd_aa = (plus_a[j] - 2.0 * base[j] + minus_a[j]) / (eps * eps);
            let fd_ab = (da_bplus[j] - da_bminus[j]) / (2.0 * eps);
            let fd_ab_alt = (db_plus[j] - db_minus[j]) / (2.0 * eps);
            let fd_bb = (plus_b[j] - 2.0 * base[j] + minus_b[j]) / (eps * eps);
            let sanity = da_plus[j] - da_minus[j];
            assert!((dc_da[j] - fd_a).abs() < 1e-6);
            assert!((dc_db[j] - fd_b).abs() < 1e-6);
            assert!((dc_daa[j] - fd_aa).abs() < 2e-5);
            assert!((dc_dab[j] - fd_ab).abs() < 2e-5);
            assert!((dc_dab[j] - fd_ab_alt).abs() < 2e-5);
            assert!((dc_dbb[j] - fd_bb).abs() < 2e-5);
            assert!(sanity.is_finite());
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
            simpson_integral(cell.left, cell.right, 2000, |z| z.powi(k as i32) * (-cell.q(z)).exp())
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
            simpson_integral(cell.left, cell.right, 3000, |z| z.powi(k as i32) * (-cell.q(z)).exp())
        };
        let reduced = reduce_sextic_moments(
            cell,
            [exact(0), exact(1), exact(2), exact(3), exact(4)],
            9,
        )
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
                simpson_integral(cell.left, cell.right, 3000, |z| z.powi(3) * (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| z.powi(4) * (-cell.q(z)).exp()),
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
        let value_numeric =
            simpson_integral(cell.left, cell.right, 4000, |z| super::normal_cdf(cell.eta(z)) * normal_pdf(z));
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
    }
}
