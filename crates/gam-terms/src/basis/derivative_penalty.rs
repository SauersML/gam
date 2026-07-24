//! Exact B-spline roughness penalties: the Gram matrix of basis derivatives.
//!
//! SPEC rule 5 requires penalties on the represented FUNCTION, never on the
//! model coefficients. For a spline `f(x) = Σ_i β_i B_i(x)` the order-`m`
//! roughness functional is
//!
//! ```text
//! J_m(f) = ∫ (f^{(m)}(x))² dx = βᵀ S β,   S_ij = ∫ B_i^{(m)}(x) B_j^{(m)}(x) dx,
//! ```
//!
//! and `S` is assembled here in closed form. The order-`m` derivative of a
//! degree-`p` B-spline is a spline of degree `p − m`, so on every knot span the
//! integrand `B_i^{(m)} B_j^{(m)}` is a polynomial of degree `≤ 2(p − m)`.
//! Gauss–Legendre with `p − m + 1` points per span integrates polynomials of
//! degree `≤ 2(p − m) + 1` exactly, so the span-by-span accumulation below is
//! exact up to roundoff — closed-form assembly, not approximate quadrature.
//!
//! Unlike the classical P-spline coefficient-difference penalty `Σ (Δᵐβ)²`,
//! this Gram is a functional of `f` itself: it is invariant under basis
//! reparameterization, stable under knot refinement (representing the SAME
//! function on a denser knot grid leaves `βᵀSβ` unchanged), and exactly
//! annihilates the polynomials of degree `< m` (constants, for the cyclic
//! basis), so the REML null space is the true polynomial space rather than a
//! knot-geometry-dependent rotation of it.
//!
//! A finite order-`m` Sobolev penalty requires `m ≤ p` and continuity through
//! derivative `m − 1`. Accordingly, an interior knot may have multiplicity at
//! most `p − m + 1`. Higher orders or multiplicities would put Dirac masses in
//! the weak derivative; silently integrating only the almost-everywhere part
//! would no longer be `∫(f⁽ᵐ⁾)²` and would create spurious null directions, so
//! those basis specifications are rejected. Repeated boundary knots and valid
//! repeated interior knots are handled exactly. Both entry points return a
//! symmetric PSD matrix.

use super::*;
use gam_math::special::gauss_legendre;

/// Exact open/clamped B-spline roughness penalty
/// `S_ij = ∫ B_i^{(order)} B_j^{(order)} dx` over the modeling interval
/// `[t_degree, t_{num_basis}]`.
///
/// This is the function-space replacement for
/// [`create_difference_penalty_matrix`] in the P-spline builder: same shape
/// (`num_basis × num_basis`, `num_basis = knots.len() − degree − 1`), same
/// null-space dimension (`order`), but an exact functional of the represented
/// spline rather than of its coefficient sequence.
pub fn bspline_derivative_penalty_matrix(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    let factor = bspline_derivative_penalty_factor(knot_vector, degree, order)?;
    materialize_derivative_gram(&factor, "open B-spline roughness penalty")
}

/// Constructive energy factor for the exact open B-spline roughness.
///
/// Each row is one weighted derivative-evaluation functional from the exact
/// span quadrature, so `S = AᵀA` without first materializing a dense Gram.
pub fn bspline_derivative_penalty_factor(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_knots_for_degree(knot_vector, degree)?;
    let num_basis = knot_vector.len() - degree - 1;
    if order == 0 || order >= num_basis {
        return Err(BasisError::InvalidPenaltyOrder { order, num_basis });
    }
    if order > degree {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: order,
            minimum_degree: order,
        });
    }

    validate_sobolev_knot_multiplicity(knot_vector, degree, order, num_basis)?;
    let (normalized_knots, domain_scale) =
        normalized_open_knot_vector(knot_vector, degree, num_basis)?;

    // The modeling interval is covered by spans `[t_k, t_{k+1}]` for
    // `k = degree .. num_basis`; clamped boundary knots make the exterior
    // spans degenerate and they carry no integral mass.
    let mut factor = derivative_energy_factor_spans(
        normalized_knots.view(),
        degree,
        order,
        degree..num_basis,
        num_basis,
        num_basis,
        |col| col,
    )?;
    rescale_derivative_factor(&mut factor, domain_scale, order)?;
    Ok(factor)
}

/// Exact function-space penalties for the anchored I-spline basis.
///
/// `roughness` is the derivative Gram of the represented value function and
/// `nullspace_shrinkage`, when requested and structurally non-empty, is the
/// exact L² metric restricted to `null(roughness)`.  The latter is a separate
/// REML coordinate; it never adds a coefficient-space ridge to directions the
/// primary roughness already controls.
#[derive(Clone, Debug)]
pub struct IsplineFunctionPenalties {
    pub roughness: Array2<f64>,
    pub roughness_nullspace_dim: usize,
    pub nullspace_shrinkage: Option<Array2<f64>>,
}

/// Exact I-spline roughness and optional function-space null shrinkage.
///
/// The `ispline_degree` argument has the same meaning as
/// [`BasisOptions::i_spline`]: the represented value basis has per-span degree
/// `q = ispline_degree + 1`.  Writing that basis as
///
/// ```text
/// I(x) = B_q(x) C - I(left),
/// C[r,j] = 1{r >= j + 1},
/// ```
///
/// gives, for every positive derivative order `m`,
///
/// ```text
/// ∫ I^(m)(x) I^(m)(x)ᵀ dx = Cᵀ [∫ B_q^(m)(x) B_q^(m)(x)ᵀ dx] C.
/// ```
///
/// The middle Gram is assembled exactly span by span by
/// [`bspline_derivative_penalty_matrix`], including nonuniform knot widths.
/// Thus this is an exact quadratic functional of the represented function,
/// not a P-spline difference approximation on its coefficients.
///
/// I-splines are anchored at the left endpoint, so the order-`m` polynomial
/// null space loses its constant direction and has structural dimension
/// `m - 1`.  When `include_nullspace_shrinkage` is true, that component alone
/// is penalized in the exact function Gram via
/// `G Z (Zᵀ G Z)⁻¹ Zᵀ G`.
pub fn ispline_function_penalties(
    knot_vector: ArrayView1<f64>,
    ispline_degree: usize,
    derivative_order: usize,
    include_nullspace_shrinkage: bool,
) -> Result<IsplineFunctionPenalties, BasisError> {
    if ispline_degree < 1 {
        return Err(BasisError::InvalidDegree(ispline_degree));
    }
    let value_degree = ispline_degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    if derivative_order == 0 {
        let num_basis = knot_vector.len().saturating_sub(value_degree + 2);
        return Err(BasisError::InvalidPenaltyOrder {
            order: derivative_order,
            num_basis,
        });
    }

    let bspline_roughness =
        bspline_derivative_penalty_matrix(knot_vector, value_degree, derivative_order)?;
    let num_bspline_basis = bspline_roughness.nrows();
    let num_ispline_basis = num_bspline_basis.checked_sub(1).ok_or_else(|| {
        BasisError::InvalidKnotVector(
            "I-spline roughness requires at least two value B-spline columns".to_string(),
        )
    })?;
    if num_ispline_basis == 0 {
        return Err(BasisError::InvalidKnotVector(
            "I-spline roughness has no represented columns".to_string(),
        ));
    }

    let mut cumulative = Array2::<f64>::zeros((num_bspline_basis, num_ispline_basis));
    for column in 0..num_ispline_basis {
        cumulative.slice_mut(s![column + 1.., column]).fill(1.0);
    }
    let mut roughness = cumulative.t().dot(&bspline_roughness).dot(&cumulative);
    symmetrize_in_place(&mut roughness);

    let roughness_nullspace_dim = derivative_order - 1;
    let nullspace_shrinkage = if include_nullspace_shrinkage && roughness_nullspace_dim > 0 {
        let function_gram = ispline_function_gram(knot_vector, ispline_degree)?;
        Some(
            function_space_nullspace_shrinkage(&roughness, &function_gram)?.ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "order-{derivative_order} I-spline roughness has structural nullity {roughness_nullspace_dim}, but its function-space null frame was not resolved"
                ))
            })?,
        )
    } else {
        None
    };

    Ok(IsplineFunctionPenalties {
        roughness,
        roughness_nullspace_dim,
        nullspace_shrinkage,
    })
}

/// Exact L² Gram `G_ij = ∫ I_i(x) I_j(x) dx` for an anchored I-spline basis.
///
/// The value functions have per-span degree `ispline_degree + 1`, so
/// `ispline_degree + 2` Gauss–Legendre nodes integrate every product exactly.
pub fn ispline_function_gram(
    knot_vector: ArrayView1<f64>,
    ispline_degree: usize,
) -> Result<Array2<f64>, BasisError> {
    if ispline_degree < 1 {
        return Err(BasisError::InvalidDegree(ispline_degree));
    }
    let value_degree = ispline_degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    validate_knots_for_degree(knot_vector, value_degree)?;
    let knot_count = knot_vector.len();
    if knot_count < 2 * (value_degree + 1) {
        crate::bail_invalid_basis!(
            "I-spline function Gram requires at least {} knots for value degree {value_degree}, got {knot_count}",
            2 * (value_degree + 1)
        );
    }
    let mut breaks = Vec::<f64>::with_capacity(knot_count - 2 * value_degree);
    for index in value_degree..=(knot_count - 1 - value_degree) {
        let knot = knot_vector[index];
        if breaks.last().is_none_or(|&previous| knot > previous) {
            breaks.push(knot);
        }
    }
    piecewise_polynomial_function_gram(&breaks, value_degree + 1, &mut |points| {
        let (basis, _) = create_basis::<Dense>(
            points,
            KnotSource::Provided(knot_vector),
            ispline_degree,
            BasisOptions::i_spline(),
        )?;
        Ok((*basis).clone())
    })
}

/// Exact cyclic (periodic) B-spline roughness penalty over one full period:
/// `S_ab = ∮ B_a^{(order)}(θ) B_b^{(order)}(θ) dθ` for the wrapped uniform
/// basis of `num_basis` cardinal translates with spacing `h = period /
/// num_basis`.
///
/// This is the exact function-space penalty for cyclic B-splines (the retired
/// coefficient-difference operator shared its `num_basis × num_basis` shape
/// and constant-only null space but penalized coefficients, violating SPEC 5):
/// an exact circulant functional of the represented function on the circle.
/// The Gram is independent of the knot anchor (integration over the whole
/// circle), so no origin argument is needed.
pub fn cyclic_bspline_derivative_penalty_matrix(
    degree: usize,
    num_basis: usize,
    period: f64,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    let factor = cyclic_bspline_derivative_penalty_factor(degree, num_basis, period, order)?;
    materialize_derivative_gram(&factor, "cyclic B-spline roughness penalty")
}

/// Constructive energy factor for the exact cyclic B-spline roughness.
pub fn cyclic_bspline_derivative_penalty_factor(
    degree: usize,
    num_basis: usize,
    period: f64,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }
    if num_basis <= degree {
        crate::bail_invalid_basis!(
            "cyclic roughness penalty requires more basis functions ({num_basis}) than degree ({degree})"
        );
    }
    if !period.is_finite() || period <= 0.0 {
        crate::bail_invalid_basis!(
            "cyclic roughness penalty requires a finite positive period, got {period}"
        );
    }
    if order == 0 || order >= num_basis {
        return Err(BasisError::InvalidPenaltyOrder { order, num_basis });
    }
    if order > degree {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: order,
            minimum_degree: order,
        });
    }

    // The wrapped basis is `B_a(θ) = Σ_k C(θ/h − a − k·num_basis)` for the
    // cardinal degree-`p` B-spline `C`. Realize it as the OPEN uniform basis on
    // the extended knot line (the exact construction the cyclic design
    // evaluator folds, see `bspline_raw_row_chunk`) and fold columns modulo
    // `num_basis` while accumulating: with `num_basis > degree` no translate
    // overlaps its own wrap, so the fold is an exact identification. The Gram
    // is anchor-invariant, so the extended knots are anchored at zero.
    // Assemble in a dimensionless unit-period coordinate and apply the exact
    // covariance `S_x = period^(1-2m) S_u` afterwards. Besides making the unit
    // behavior explicit, this avoids interpreting a perfectly valid small
    // physical period as a numerically degenerate knot grid.
    let knots = cyclic_uniform_knot_vector(0.0, 1.0, degree, num_basis);
    let num_basis_extended = knots.len() - degree - 1;

    // One period = the `num_basis` spans `[t_k, t_{k+1}]`,
    // `k = degree .. degree + num_basis`, of the extended knot line.
    let mut factor = derivative_energy_factor_spans(
        knots.view(),
        degree,
        order,
        degree..degree + num_basis,
        num_basis_extended,
        num_basis,
        |col| col % num_basis,
    )?;
    rescale_derivative_factor(&mut factor, period, order)?;
    Ok(factor)
}

/// Maps the open spline's modeling interval to `[0, 1]`. Assembly in this
/// dimensionless coordinate keeps the exact assembly independent of the user's
/// physical units.
fn normalized_open_knot_vector(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    num_basis: usize,
) -> Result<(Array1<f64>, f64), BasisError> {
    let left = knot_vector[degree];
    let right = knot_vector[num_basis];
    let domain_scale = right - left;
    if !domain_scale.is_finite() || domain_scale <= 0.0 {
        return Err(BasisError::InvalidKnotVector(format!(
            "B-spline roughness domain [{left}, {right}] must have finite positive width"
        )));
    }
    let normalized = knot_vector.mapv(|knot| (knot - left) / domain_scale);
    validate_knot_spans_nondegenerate(normalized.view(), degree)?;
    Ok((normalized, domain_scale))
}

/// Ensures every spline in the basis has an order-`m` weak derivative in L².
/// At an interior knot of multiplicity `r`, a degree-`p` spline is `C^(p-r)`;
/// membership in `H^m` therefore requires `r ≤ p-m+1`.
fn validate_sobolev_knot_multiplicity(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    order: usize,
    num_basis: usize,
) -> Result<(), BasisError> {
    let left = knot_vector[degree];
    let right = knot_vector[num_basis];
    let max_multiplicity = degree - order + 1;
    let mut i = 0usize;
    while i < knot_vector.len() {
        let knot = knot_vector[i];
        let mut end = i + 1;
        while end < knot_vector.len() && knot_vector[end] == knot {
            end += 1;
        }
        let multiplicity = end - i;
        if knot > left && knot < right && multiplicity > max_multiplicity {
            return Err(BasisError::InvalidKnotVector(format!(
                "interior knot {knot} has multiplicity {multiplicity}, but an order-{order} L2 derivative of a degree-{degree} spline requires multiplicity <= {max_multiplicity}"
            )));
        }
        i = end;
    }
    Ok(())
}

/// Materialize `AᵀA` for a derivative energy factor without allowing an
/// otherwise-finite factor to escape as a non-finite dense penalty.
fn materialize_derivative_gram(
    factor: &Array2<f64>,
    context: &str,
) -> Result<Array2<f64>, BasisError> {
    let mut penalty = fast_ata(factor);
    if penalty.iter().any(|value| !value.is_finite()) {
        return Err(BasisError::InvalidInput(format!(
            "{context} is not representable as a finite f64 Gram"
        )));
    }
    symmetrize_in_place(&mut penalty);
    Ok(penalty)
}

/// Under `x = a + c·u`, an order-`m` derivative Gram transforms as
/// `S_x = c^(1-2m) S_u`.
fn derivative_gram_coordinate_scale(domain_scale: f64, order: usize) -> Result<f64, BasisError> {
    let twice_order = order
        .checked_mul(2)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "derivative order {order} is too large to scale the roughness Gram"
            ))
        })?;
    let scale = domain_scale.powi(1 - twice_order);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(BasisError::InvalidInput(format!(
            "order-{order} roughness scaling over domain width {domain_scale} is not representable"
        )));
    }
    Ok(scale)
}

/// Apply exact coordinate covariance to the energy factor and reject a scale
/// whose floating representation would erase any basis-function energy.
fn rescale_derivative_factor(
    factor: &mut Array2<f64>,
    domain_scale: f64,
    order: usize,
) -> Result<(), BasisError> {
    let root_scale = derivative_gram_coordinate_scale(domain_scale, order)?.sqrt();
    factor.mapv_inplace(|value| value * root_scale);
    if factor.iter().any(|value| !value.is_finite()) {
        return Err(BasisError::InvalidInput(format!(
            "order-{order} roughness factor over domain width {domain_scale} is not representable"
        )));
    }
    let preserves_all_basis_energies = (0..factor.ncols()).all(|column| {
        factor
            .column(column)
            .iter()
            .any(|value| *value != 0.0)
    });
    if !preserves_all_basis_energies {
        return Err(BasisError::InvalidInput(format!(
            "order-{order} roughness factor over domain width {domain_scale} lost a basis-function energy"
        )));
    }
    Ok(())
}

/// Span-by-span exact Gauss–Legendre accumulation of
/// `∫ B_i^{(order)} B_j^{(order)}` into `s[fold(i), fold(j)]`.
///
/// `spans` indexes knot intervals `[t_k, t_{k+1}]`; `out_len` is the raw
/// (pre-fold) basis dimension of `knot_vector`. On each span only the
/// `degree + 1` basis functions `k − degree ..= k` are supported, so the
/// inner accumulation is restricted to that window.
fn derivative_energy_factor_spans(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    order: usize,
    spans: std::ops::Range<usize>,
    out_len: usize,
    output_dim: usize,
    fold: impl Fn(usize) -> usize,
) -> Result<Array2<f64>, BasisError> {
    // Integrand degree per span is 2(p − m); `p − m + 1` Gauss points are
    // exact through degree 2(p − m) + 1.
    let quad_points = degree - order + 1;
    let (nodes, weights) = gauss_legendre(quad_points);
    let active_span_count = spans
        .clone()
        .filter(|&k| knot_vector[k + 1] > knot_vector[k])
        .count();
    let mut factor = Array2::<f64>::zeros((active_span_count * quad_points, output_dim));
    let mut row = vec![0.0_f64; out_len];
    let mut workspace = BsplineDerivativeWorkspace::new();

    for (active_span, k) in spans
        .filter(|&k| knot_vector[k + 1] > knot_vector[k])
        .enumerate()
    {
        let left = knot_vector[k];
        let right = knot_vector[k + 1];
        let width = right - left;
        let mid = 0.5 * (left + right);
        let half = 0.5 * width;
        for (quadrature_node, (node, weight)) in nodes.iter().zip(weights.iter()).enumerate() {
            let factor_row = active_span * quad_points + quadrature_node;
            let x = mid + half * node;
            evaluate_bspline_derivative_recurrence_into(
                order,
                x,
                knot_vector,
                degree,
                &mut row,
                &mut workspace,
                0,
            )?;
            let root_weight = (weight * half).sqrt();
            let support_start = k - degree;
            for i in support_start..=k {
                let vi = row[i];
                if vi == 0.0 {
                    continue;
                }
                let fi = fold(i);
                factor[[factor_row, fi]] += root_weight * vi;
            }
        }
    }
    Ok(factor)
}

/// Exact symmetrization: the accumulation is symmetric in exact arithmetic;
/// this removes the last-ulp asymmetry from floating-point summation order so
/// downstream eigen/Cholesky consumers see a bit-exact symmetric matrix.
fn symmetrize_in_place(s: &mut Array2<f64>) {
    let n = s.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (s[[i, j]] + s[[j, i]]);
            s[[i, j]] = avg;
            s[[j, i]] = avg;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn binomial(n: usize, k: usize) -> f64 {
        let mut coefficient = 1.0_f64;
        for i in 0..k {
            coefficient = coefficient * (n - i) as f64 / (i + 1) as f64;
        }
        coefficient
    }

    /// Blossom (polar form) coefficients representing the monomial `x^r`
    /// (`r ≤ degree`) exactly in the B-spline basis:
    /// `β_i = e_r(t_{i+1}, …, t_{i+degree}) / C(degree, r)`, the normalized
    /// elementary symmetric polynomial of the interior knot window.
    fn monomial_coefficients(knots: ArrayView1<f64>, degree: usize, r: usize) -> Array1<f64> {
        let num_basis = knots.len() - degree - 1;
        let mut beta = Array1::<f64>::zeros(num_basis);
        for i in 0..num_basis {
            let window: Vec<f64> = (1..=degree).map(|k| knots[i + k]).collect();
            // e_r via the standard DP over the window.
            let mut e = vec![0.0_f64; r + 1];
            e[0] = 1.0;
            for &t in &window {
                for j in (1..=r).rev() {
                    e[j] += t * e[j - 1];
                }
            }
            beta[i] = e[r] / binomial(degree, r);
        }
        beta
    }

    fn clamped_knots(interior: &[f64], degree: usize, a: f64, b: f64) -> Array1<f64> {
        let mut v = vec![a; degree + 1];
        v.extend_from_slice(interior);
        v.extend(std::iter::repeat_n(b, degree + 1));
        Array1::from(v)
    }

    /// Convert B-spline coefficients for an anchored function (`b[0] = 0`)
    /// into the cumulative I-spline chart `b = C alpha`.
    fn anchored_bspline_to_ispline_coefficients(b: &Array1<f64>) -> Array1<f64> {
        assert!(b.len() >= 2);
        assert!(b[0].abs() < 1e-12, "anchored function must vanish at left");
        Array1::from_iter((0..b.len() - 1).map(|index| b[index + 1] - b[index]))
    }

    fn assert_symmetric_psd_with_nullity(s: &Array2<f64>, expected_nullity: usize) {
        assert_eq!(s.nrows(), s.ncols());
        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                assert_eq!(s[[i, j]], s[[j, i]], "penalty must be bit-symmetric");
            }
        }
        let (eigenvalues, _) = s.eigh(Side::Lower).expect("symmetric eigendecomposition");
        let spectral_scale = eigenvalues
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()))
            .max(1.0);
        let psd_tolerance =
            default_rrqr_rank_alpha() * f64::EPSILON * s.nrows().max(1) as f64 * spectral_scale;
        assert!(
            eigenvalues.iter().all(|&value| value >= -psd_tolerance),
            "penalty must be PSD; eigenvalues={eigenvalues:?}, tolerance={psd_tolerance}"
        );
        let (_, rank) =
            rrqr_nullspace_basis(s, default_rrqr_rank_alpha()).expect("penalty RRQR rank");
        assert_eq!(
            rank,
            s.nrows() - expected_nullity,
            "unexpected penalty nullity"
        );
    }

    /// Inserts one new, distinct interior knot and transforms coefficients via
    /// the exact Boehm identity, preserving the represented spline pointwise.
    fn insert_knot_once(
        knots: &Array1<f64>,
        coefficients: &Array1<f64>,
        degree: usize,
        knot: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let num_basis = coefficients.len();
        assert_eq!(knots.len(), num_basis + degree + 1);
        let span = (degree..num_basis)
            .find(|&k| knots[k] < knot && knot < knots[k + 1])
            .expect("new knot lies strictly inside one span");

        let mut refined_knots = knots.to_vec();
        refined_knots.insert(span + 1, knot);
        let mut refined = Array1::<f64>::zeros(num_basis + 1);
        for i in 0..=span - degree {
            refined[i] = coefficients[i];
        }
        for i in (span - degree + 1)..=span {
            let alpha = (knot - knots[i]) / (knots[i + degree] - knots[i]);
            refined[i] = alpha * coefficients[i] + (1.0 - alpha) * coefficients[i - 1];
        }
        for i in (span + 1)..=num_basis {
            refined[i] = coefficients[i - 1];
        }
        (Array1::from(refined_knots), refined)
    }

    /// The penalty is a functional of the FUNCTION: representing the fixed
    /// cubic `f(x) = x³` on [0,1] must give exactly
    /// `∫₀¹ (6x)² dx = 12` for every knot vector — uniform, quantile-like,
    /// dense, or sparse. The retired coefficient-difference penalty changes
    /// with knot density here; the exact Gram must not.
    #[test]
    fn open_penalty_of_fixed_cubic_is_knot_invariant() {
        let degree = 3usize;
        let knot_sets: Vec<Array1<f64>> = vec![
            clamped_knots(&[0.5], degree, 0.0, 1.0),
            clamped_knots(&[0.2, 0.4, 0.6, 0.8], degree, 0.0, 1.0),
            clamped_knots(&[0.05, 0.1, 0.35, 0.4, 0.41, 0.8, 0.97], degree, 0.0, 1.0),
            clamped_knots(
                &Array1::linspace(0.025, 0.975, 39).to_vec(),
                degree,
                0.0,
                1.0,
            ),
        ];
        for knots in &knot_sets {
            let s = bspline_derivative_penalty_matrix(knots.view(), degree, 2).unwrap();
            let beta = monomial_coefficients(knots.view(), degree, 3);
            let j = beta.dot(&s.dot(&beta));
            assert!(
                (j - 12.0).abs() < 1e-9,
                "∫(f'')² for f=x³ must be 12 on every knot grid; got {j} for {} knots",
                knots.len()
            );
            // Quadratic: ∫ (2)² = 4.
            let beta2 = monomial_coefficients(knots.view(), degree, 2);
            let j2 = beta2.dot(&s.dot(&beta2));
            assert!(
                (j2 - 4.0).abs() < 1e-9,
                "∫(f'')² for f=x² must be 4, got {j2}"
            );
        }
    }

    /// Closed-form polynomial oracle over a translated, non-unit domain. For
    /// `f(x)=(x-a)^r`,
    ///
    /// `integral (f^(m))^2 = (r!/(r-m)!)^2 (b-a)^(2(r-m)+1)/(2(r-m)+1)`.
    ///
    /// Exercising every supported derivative order through degree four checks
    /// both the recurrence coefficients and the physical-coordinate Jacobian,
    /// independently of a second quadrature implementation.
    #[test]
    fn open_penalty_matches_closed_form_polynomial_energies() {
        let (a, b) = (-1.7_f64, 2.4_f64);
        let width = b - a;
        let interior = [
            a + 0.11 * width,
            a + 0.37 * width,
            a + 0.52 * width,
            a + 0.86 * width,
        ];
        for degree in 1..=4usize {
            let knots = clamped_knots(&interior, degree, a, b);
            let shifted_knots = knots.mapv(|knot| knot - a);
            for order in 1..=degree {
                let s = bspline_derivative_penalty_matrix(knots.view(), degree, order).unwrap();
                for polynomial_degree in order..=degree {
                    let beta =
                        monomial_coefficients(shifted_knots.view(), degree, polynomial_degree);
                    let derivative_factor = ((polynomial_degree - order + 1)..=polynomial_degree)
                        .map(|factor| factor as f64)
                        .product::<f64>();
                    let residual_degree = polynomial_degree - order;
                    let expected = derivative_factor.powi(2)
                        * width.powi((2 * residual_degree + 1) as i32)
                        / (2 * residual_degree + 1) as f64;
                    let observed = beta.dot(&s.dot(&beta));
                    let relative_error = (observed - expected).abs() / expected.max(1.0);
                    // 1e-9: the degree-4/order-4 corner accumulates ~4.5e-10
                    // relative roundoff through the span-by-span Gauss–Legendre
                    // sums on energies O(2e3) — the quadrature is EXACT for the
                    // polynomial integrand, so the bound is f64 accumulation,
                    // not method error.
                    assert!(
                        relative_error < 1.0e-9,
                        "degree={degree}, order={order}, polynomial degree={polynomial_degree}: expected {expected}, observed {observed}, relative error {relative_error}"
                    );
                }
            }
        }
    }

    /// Independent closed-form oracle for the I-spline cumulative chart on a
    /// strongly nonuniform knot vector.  Every anchored monomial
    /// `(x-a)^r`, `r>=1`, belongs to the I-spline span.  Its order-`m` energy is
    /// known analytically and cannot depend on the coefficient geometry.
    #[test]
    fn ispline_penalty_matches_closed_form_on_nonuniform_knots() {
        let (a, b) = (-1.3_f64, 2.6_f64);
        let width = b - a;
        let value_degree = 3usize;
        let ispline_degree = value_degree - 1;
        let knots = clamped_knots(
            &[
                a + 0.03 * width,
                a + 0.21 * width,
                a + 0.22 * width,
                a + 0.68 * width,
                a + 0.94 * width,
            ],
            value_degree,
            a,
            b,
        );
        let shifted_knots = knots.mapv(|knot| knot - a);

        for order in 1..=value_degree {
            let built =
                ispline_function_penalties(knots.view(), ispline_degree, order, false).unwrap();
            assert_eq!(built.roughness_nullspace_dim, order - 1);
            assert!(built.nullspace_shrinkage.is_none());
            assert_symmetric_psd_with_nullity(&built.roughness, order - 1);

            for polynomial_degree in 1..=value_degree {
                let b_coefficients =
                    monomial_coefficients(shifted_knots.view(), value_degree, polynomial_degree);
                let alpha = anchored_bspline_to_ispline_coefficients(&b_coefficients);
                let observed = alpha.dot(&built.roughness.dot(&alpha));
                let expected = if polynomial_degree < order {
                    0.0
                } else {
                    let derivative_factor = ((polynomial_degree - order + 1)..=polynomial_degree)
                        .map(|factor| factor as f64)
                        .product::<f64>();
                    let residual_degree = polynomial_degree - order;
                    derivative_factor.powi(2) * width.powi((2 * residual_degree + 1) as i32)
                        / (2 * residual_degree + 1) as f64
                };
                if expected == 0.0 {
                    // This is a cancellation test, not an absolute-value test:
                    // the derivative Gram can be large on narrowly separated
                    // knots even though the represented polynomial is in its
                    // exact null space. Use the same source-quadratic backward-
                    // error envelope as generalized null classification.
                    let penalty_scale = built
                        .roughness
                        .rows()
                        .into_iter()
                        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
                        .fold(0.0_f64, f64::max);
                    let backward_error = default_rrqr_rank_alpha()
                        * f64::EPSILON
                        * built.roughness.nrows().max(1) as f64
                        * penalty_scale
                        * alpha.dot(&alpha);
                    assert!(
                        observed.abs() <= backward_error,
                        "I-spline degree={value_degree}, order={order}, polynomial degree={polynomial_degree}: expected exact zero, observed {observed}, backward-error envelope {backward_error}"
                    );
                } else {
                    let relative_error = (observed - expected).abs() / expected.abs();
                    assert!(
                        relative_error < 2e-10,
                        "I-spline degree={value_degree}, order={order}, polynomial degree={polynomial_degree}: expected {expected}, observed {observed}, relative error {relative_error}"
                    );
                }
            }
        }
    }

    /// The cumulative-map assembly and the public I-spline derivative
    /// evaluator are independent routes to the same exact integral.  An
    /// deliberately over-resolved Gauss rule must reproduce the matrix energy
    /// on every unequal knot span to roundoff.
    #[test]
    fn ispline_penalty_matches_independent_span_quadrature() {
        let value_degree = 4usize;
        let ispline_degree = value_degree - 1;
        let order = 3usize;
        let knots = clamped_knots(&[0.04, 0.19, 0.2, 0.61, 0.91], value_degree, 0.0, 1.0);
        let built = ispline_function_penalties(knots.view(), ispline_degree, order, false).unwrap();
        let alpha = Array1::from_iter(
            (0..built.roughness.nrows()).map(|index| (0.37 + 1.91 * index as f64).sin()),
        );
        let exact = alpha.dot(&built.roughness.dot(&alpha));

        let (nodes, weights) = gauss_legendre(2 * value_degree + 3);
        let mut points = Vec::<f64>::new();
        let mut quadrature_weights = Vec::<f64>::new();
        for span in knots.windows(2) {
            let (left, right) = (span[0], span[1]);
            if right <= left {
                continue;
            }
            let half = 0.5 * (right - left);
            let mid = 0.5 * (left + right);
            for (&node, &weight) in nodes.iter().zip(weights.iter()) {
                points.push(mid + half * node);
                quadrature_weights.push(half * weight);
            }
        }
        let derivative = create_ispline_derivative_dense(
            Array1::from(points).view(),
            &knots,
            ispline_degree,
            order,
        )
        .unwrap()
        .dot(&alpha);
        let oracle = derivative
            .iter()
            .zip(quadrature_weights.iter())
            .map(|(&value, &weight)| weight * value * value)
            .sum::<f64>();
        let relative_error = (exact - oracle).abs() / exact.abs().max(1.0);
        assert!(
            relative_error < 2e-11,
            "exact cumulative Gram {exact} differs from span quadrature {oracle}; relative error {relative_error}"
        );
    }

    /// Knot insertion only changes coordinates.  The same anchored spline in
    /// the coarse and Boehm-refined I-spline charts must agree pointwise and
    /// carry identical function roughness.
    #[test]
    fn ispline_penalty_is_invariant_under_exact_knot_insertion() {
        let value_degree = 3usize;
        let ispline_degree = value_degree - 1;
        let order = 2usize;
        let coarse_knots = clamped_knots(&[0.16, 0.53, 0.88], value_degree, 0.0, 1.0);
        let coarse_b = array![0.0, 0.7, -0.4, 1.6, 0.2, 1.1, -0.3];
        let (fine_knots, fine_b) = insert_knot_once(&coarse_knots, &coarse_b, value_degree, 0.37);
        let coarse_alpha = anchored_bspline_to_ispline_coefficients(&coarse_b);
        let fine_alpha = anchored_bspline_to_ispline_coefficients(&fine_b);

        let points = Array1::linspace(0.0, 1.0, 137);
        let (coarse_basis, _) = create_basis::<Dense>(
            points.view(),
            KnotSource::Provided(coarse_knots.view()),
            ispline_degree,
            BasisOptions::i_spline(),
        )
        .unwrap();
        let (fine_basis, _) = create_basis::<Dense>(
            points.view(),
            KnotSource::Provided(fine_knots.view()),
            ispline_degree,
            BasisOptions::i_spline(),
        )
        .unwrap();
        let value_error = (&coarse_basis.dot(&coarse_alpha) - &fine_basis.dot(&fine_alpha))
            .iter()
            .fold(0.0_f64, |error, value| error.max(value.abs()));
        assert!(
            value_error < 2e-12,
            "knot insertion changed f by {value_error}"
        );

        let coarse =
            ispline_function_penalties(coarse_knots.view(), ispline_degree, order, false).unwrap();
        let fine =
            ispline_function_penalties(fine_knots.view(), ispline_degree, order, false).unwrap();
        let coarse_energy = coarse_alpha.dot(&coarse.roughness.dot(&coarse_alpha));
        let fine_energy = fine_alpha.dot(&fine.roughness.dot(&fine_alpha));
        let relative_error = (coarse_energy - fine_energy).abs() / coarse_energy.abs().max(1.0);
        assert!(
            relative_error < 2e-12,
            "knot insertion changed roughness: coarse={coarse_energy}, fine={fine_energy}, relative error {relative_error}"
        );
    }

    /// Double penalty acts only on the primary derivative null space in the
    /// function metric.  It is covariant under a dense change of basis and is
    /// exactly zero on the G-orthogonal primary range, unlike `eye(p)`.
    #[test]
    fn ispline_null_shrinkage_is_metric_exact_and_reparameterization_covariant() {
        let value_degree = 3usize;
        let ispline_degree = value_degree - 1;
        let knots = clamped_knots(&[0.08, 0.31, 0.73, 0.95], value_degree, 0.0, 1.0);
        let built = ispline_function_penalties(knots.view(), ispline_degree, 2, true).unwrap();
        let ridge = built
            .nullspace_shrinkage
            .as_ref()
            .expect("order-two anchored spline has one linear null direction");
        let gram = ispline_function_gram(knots.view(), ispline_degree).unwrap();

        let linear_b = monomial_coefficients(knots.view(), value_degree, 1);
        let linear = anchored_bspline_to_ispline_coefficients(&linear_b);
        let roughness_scale = built
            .roughness
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()))
            .max(1.0);
        let null_residual = built
            .roughness
            .dot(&linear)
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        assert!(null_residual < 2e-11 * roughness_scale);
        let null_function_energy = linear.dot(&gram.dot(&linear));
        let null_ridge_energy = linear.dot(&ridge.dot(&linear));
        assert!(
            (null_ridge_energy - null_function_energy).abs()
                < 2e-11 * null_function_energy.max(1.0),
            "ridge must equal the exact L2 norm on null(S): ridge={null_ridge_energy}, L2={null_function_energy}"
        );

        let mut range =
            Array1::from_iter((0..linear.len()).map(|index| (0.2 + index as f64 * 1.7).cos()));
        let projection = linear.dot(&gram.dot(&range)) / null_function_energy;
        range -= &(projection * &linear);
        let range_ridge_energy = range.dot(&ridge.dot(&range));
        let range_roughness_energy = range.dot(&built.roughness.dot(&range));
        assert!(range_roughness_energy > 1e-8 * roughness_scale);
        assert!(
            range_ridge_energy.abs() < 2e-11 * null_function_energy.max(1.0),
            "null ridge leaked onto the function-metric range: {range_ridge_energy}"
        );

        let p = built.roughness.nrows();
        let mut map = Array2::<f64>::eye(p);
        for index in 0..p {
            map[[index, index]] = 0.7 + 0.13 * index as f64;
        }
        if p >= 3 {
            map[[0, 1]] = 0.31;
            map[[1, 2]] = -0.22;
        }
        let roughness_mapped = map.t().dot(&built.roughness).dot(&map);
        let gram_mapped = map.t().dot(&gram).dot(&map);
        let ridge_mapped = function_space_nullspace_shrinkage(&roughness_mapped, &gram_mapped)
            .unwrap()
            .expect("mapped null direction");
        let expected_mapped = map.t().dot(ridge).dot(&map);
        let map_scale = expected_mapped
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()))
            .max(1.0);
        let map_error = (&ridge_mapped - &expected_mapped)
            .iter()
            .fold(0.0_f64, |error, value| error.max(value.abs()));
        assert!(
            map_error < 2e-10 * map_scale,
            "function-space ridge failed basis covariance: error={map_error}, scale={map_scale}"
        );
    }

    /// Exact polynomial null space: monomials of degree < order are
    /// annihilated, degree = order is not; the Gram is symmetric PSD.
    #[test]
    fn open_penalty_null_space_is_exact_polynomials() {
        let degree = 3usize;
        let knots = clamped_knots(&[0.13, 0.4, 0.55, 0.72, 0.9], degree, 0.0, 1.0);
        for order in 1..=degree {
            let s = bspline_derivative_penalty_matrix(knots.view(), degree, order).unwrap();
            for r in 0..order {
                let beta = monomial_coefficients(knots.view(), degree, r);
                let energy = beta.dot(&s.dot(&beta));
                let residual = s
                    .dot(&beta)
                    .iter()
                    .fold(0.0_f64, |norm, value| norm.max(value.abs()));
                let scale = s.iter().fold(0.0_f64, |norm, value| norm.max(value.abs()));
                assert!(
                    residual < 1e-11 * scale.max(1.0),
                    "x^{r} must lie in the order-{order} null space; |Sβ|∞={residual}, energy={energy}"
                );
            }
            let beta = monomial_coefficients(knots.view(), degree, order);
            assert!(
                beta.dot(&s.dot(&beta)) > 1e-6,
                "x^{order} must be penalized at order {order}"
            );
            assert_symmetric_psd_with_nullity(&s, order);
        }
    }

    /// Knot insertion is an exact basis reparameterization. A non-polynomial
    /// coarse spline and its Boehm-refined representation must agree both
    /// pointwise and in roughness energy.
    #[test]
    fn open_penalty_is_invariant_under_exact_knot_insertion() {
        let degree = 3usize;
        let order = 2usize;
        let coarse_knots = clamped_knots(&[0.2, 0.55, 0.8], degree, 0.0, 1.0);
        let coarse_beta = array![0.3, -1.2, 0.7, 2.1, -0.4, 0.9, -0.2];
        let (fine_knots, fine_beta) = insert_knot_once(&coarse_knots, &coarse_beta, degree, 0.37);

        let points = Array1::linspace(0.0, 1.0, 101);
        let (coarse_basis, _) = create_basis::<Dense>(
            points.view(),
            KnotSource::Provided(coarse_knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let (fine_basis, _) = create_basis::<Dense>(
            points.view(),
            KnotSource::Provided(fine_knots.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let coarse_values = coarse_basis.dot(&coarse_beta);
        let fine_values = fine_basis.dot(&fine_beta);
        let value_error = (&coarse_values - &fine_values)
            .iter()
            .fold(0.0_f64, |error, value| error.max(value.abs()));
        assert!(
            value_error < 1e-12,
            "Boehm insertion changed f by {value_error}"
        );

        let coarse_s =
            bspline_derivative_penalty_matrix(coarse_knots.view(), degree, order).unwrap();
        let fine_s = bspline_derivative_penalty_matrix(fine_knots.view(), degree, order).unwrap();
        let coarse_energy = coarse_beta.dot(&coarse_s.dot(&coarse_beta));
        let fine_energy = fine_beta.dot(&fine_s.dot(&fine_beta));
        let relative_error = (coarse_energy - fine_energy).abs() / coarse_energy.abs().max(1.0);
        assert!(
            relative_error < 1e-12,
            "exact refinement changed roughness: coarse={coarse_energy}, fine={fine_energy}, rel={relative_error}"
        );
    }

    #[test]
    fn open_penalty_handles_valid_repeats_and_rejects_non_sobolev_multiplicity() {
        let degree = 3usize;
        let valid = clamped_knots(&[0.25, 0.5, 0.5, 0.75], degree, 0.0, 1.0);
        let s = bspline_derivative_penalty_matrix(valid.view(), degree, 2).unwrap();
        assert_symmetric_psd_with_nullity(&s, 2);
        let cubic = monomial_coefficients(valid.view(), degree, 3);
        assert!((cubic.dot(&s.dot(&cubic)) - 12.0).abs() < 1e-9);

        let invalid = clamped_knots(&[0.25, 0.5, 0.5, 0.5, 0.75], degree, 0.0, 1.0);
        let error = bspline_derivative_penalty_matrix(invalid.view(), degree, 2).unwrap_err();
        assert!(matches!(error, BasisError::InvalidKnotVector(_)));
    }

    #[test]
    fn open_penalty_is_affine_coordinate_covariant() {
        let degree = 3usize;
        let order = 2usize;
        let knots = clamped_knots(&[0.15, 0.4, 0.8], degree, 0.0, 1.0);
        let unit = bspline_derivative_penalty_matrix(knots.view(), degree, order).unwrap();
        for (translation, coordinate_scale) in [(-8.25_f64, 3.75_f64), (4.5, 1.0), (0.0, 1e-13)] {
            let transformed_knots = knots.mapv(|knot| translation + coordinate_scale * knot);
            let transformed =
                bspline_derivative_penalty_matrix(transformed_knots.view(), degree, order).unwrap();
            let expected_scale = coordinate_scale.powi(1 - 2 * order as i32);
            let max_relative_error = unit
                .iter()
                .zip(transformed.iter())
                .map(|(&base, &observed)| {
                    (observed - expected_scale * base).abs()
                        / observed.abs().max((expected_scale * base).abs()).max(1.0)
                })
                .fold(0.0_f64, f64::max);
            assert!(
                max_relative_error < 1e-12,
                "affine coordinate covariance failed for translation={translation}, scale={coordinate_scale}: rel={max_relative_error}"
            );
        }
    }

    /// The stated Gauss point count is EXACT: doubling the per-span points
    /// must reproduce the same matrix to roundoff.
    #[test]
    fn quadrature_point_count_is_exact_not_approximate() {
        let degree = 3usize;
        let knots = clamped_knots(&[0.1, 0.42, 0.43, 0.7], degree, 0.0, 1.0);
        for order in 1..=degree {
            let s = bspline_derivative_penalty_matrix(knots.view(), degree, order).unwrap();
            let num_basis = knots.len() - degree - 1;
            let mut s_over = Array2::<f64>::zeros((num_basis, num_basis));
            // Re-accumulate with a deliberately excessive rule by treating the
            // integrand as if it were of much higher degree.
            let (nodes, weights) = gauss_legendre(2 * (degree - order + 1) + 5);
            let mut row = vec![0.0_f64; num_basis];
            let mut ws = BsplineDerivativeWorkspace::new();
            for k in degree..num_basis {
                let (a, b) = (knots[k], knots[k + 1]);
                if b <= a {
                    continue;
                }
                for (node, weight) in nodes.iter().zip(weights.iter()) {
                    let x = 0.5 * (a + b) + 0.5 * (b - a) * node;
                    evaluate_bspline_derivative_recurrence_into(
                        order,
                        x,
                        knots.view(),
                        degree,
                        &mut row,
                        &mut ws,
                        0,
                    )
                    .unwrap();
                    let w = weight * 0.5 * (b - a);
                    for i in 0..num_basis {
                        for j in 0..num_basis {
                            s_over[[i, j]] += w * row[i] * row[j];
                        }
                    }
                }
            }
            let max_err = (&s - &s_over).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let scale = s.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            assert!(
                max_err < 1e-11 * scale,
                "order {order}: minimal rule differs from oversampled rule by {max_err}"
            );
        }
    }

    /// Orders with no finite function-level roughness are typed errors.
    #[test]
    fn order_above_degree_is_rejected() {
        let degree = 1usize;
        let knots = clamped_knots(&[0.2, 0.4, 0.6, 0.8], degree, 0.0, 1.0);
        let err = bspline_derivative_penalty_matrix(knots.view(), degree, 2).unwrap_err();
        assert!(matches!(
            err,
            BasisError::InsufficientDegreeForDerivative { .. }
        ));
        let err = cyclic_bspline_derivative_penalty_matrix(1, 8, 1.0, 2).unwrap_err();
        assert!(matches!(
            err,
            BasisError::InsufficientDegreeForDerivative { .. }
        ));
    }

    /// Degree-one periodic splines are ordinary hat functions. Their exact
    /// first-derivative Gram is the circular finite-element stiffness stencil:
    /// diagonal `2/h`, immediate circular neighbours `-1/h`, and zero
    /// elsewhere. In particular the `(0,n-1)` entry is a closed-form seam
    /// oracle for the wrapped-support assembly.
    #[test]
    fn cyclic_linear_first_derivative_matches_exact_wrapped_stencil() {
        let (degree, order, n, period) = (1usize, 1usize, 9usize, 2.75_f64);
        let h = period / n as f64;
        let s = cyclic_bspline_derivative_penalty_matrix(degree, n, period, order).unwrap();
        let matrix_scale = 2.0 / h;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    2.0 / h
                } else if j == (i + 1) % n || i == (j + 1) % n {
                    -1.0 / h
                } else {
                    0.0
                };
                let error = (s[[i, j]] - expected).abs();
                assert!(
                    error <= 1e-12 * matrix_scale,
                    "wrapped stiffness mismatch at ({i},{j}): expected {expected}, observed {}, error {error}",
                    s[[i, j]],
                );
            }
        }
        assert!(s[[0, n - 1]] < 0.0, "the seam neighbours must overlap");
    }

    /// Default cubic/order-two closed form. For cardinal cubic splines the
    /// circular stiffness stencil at lags `0,1,2,3` is
    /// `(8/3,-3/2,0,1/6)/h^3`; larger circular separations have disjoint
    /// derivative support and therefore an exactly zero integral.
    #[test]
    fn cyclic_cubic_second_derivative_matches_exact_compact_stencil() {
        let (degree, order, n, period) = (3usize, 2usize, 11usize, 3.1_f64);
        let h = period / n as f64;
        let inverse_h_cubed = h.powi(-3);
        let s = cyclic_bspline_derivative_penalty_matrix(degree, n, period, order).unwrap();
        let matrix_scale = (8.0 / 3.0) * inverse_h_cubed;
        for i in 0..n {
            for j in 0..n {
                let linear_distance = i.abs_diff(j);
                let circular_distance = linear_distance.min(n - linear_distance);
                let expected = match circular_distance {
                    0 => (8.0 / 3.0) * inverse_h_cubed,
                    1 => (-3.0 / 2.0) * inverse_h_cubed,
                    2 => 0.0,
                    3 => (1.0 / 6.0) * inverse_h_cubed,
                    _ => 0.0,
                };
                let error = (s[[i, j]] - expected).abs();
                assert!(
                    error <= 1e-12 * matrix_scale,
                    "cubic wrapped stiffness mismatch at ({i},{j}), circular lag {circular_distance}: expected {expected}, observed {}, error {error}",
                    s[[i, j]],
                );
            }
        }
    }

    #[test]
    fn unrepresentable_coordinate_scaling_is_rejected() {
        let underflow = cyclic_bspline_derivative_penalty_matrix(3, 8, 1e200, 2).unwrap_err();
        assert!(matches!(underflow, BasisError::InvalidInput(_)));

        // `period^-3` is finite here, but multiplying it by the unit-period
        // cubic stiffness entries overflows. The assembled operator must be a
        // typed error, never an infinite or silently rank-deficient matrix.
        let post_multiply_overflow =
            cyclic_bspline_derivative_penalty_matrix(3, 64, 1e-102, 2).unwrap_err();
        assert!(matches!(
            post_multiply_overflow,
            BasisError::InvalidInput(_)
        ));
    }

    /// The cyclic Gram must be circulant (no privileged knot), annihilate
    /// exactly the constants, and be PSD.
    #[test]
    fn cyclic_penalty_is_circulant_with_constant_null_space() {
        let (degree, n, period) = (3usize, 8usize, std::f64::consts::TAU);
        for order in 1..=degree {
            let s = cyclic_bspline_derivative_penalty_matrix(degree, n, period, order).unwrap();
            let scale = s.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for i in 0..n {
                for j in 0..n {
                    let d = (s[[i, j]] - s[[(i + 1) % n, (j + 1) % n]]).abs();
                    assert!(d < 1e-12 * scale, "cyclic Gram must be circulant; Δ={d}");
                }
            }
            let ones = Array1::<f64>::ones(n);
            let c = s.dot(&ones).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(c < 1e-10 * scale, "constants must be null; |S·1|={c}");
            // Any non-constant direction is penalized (nullity is exactly 1).
            let mut beta = Array1::<f64>::zeros(n);
            beta[0] = 1.0;
            assert!(beta.dot(&s.dot(&beta)) > 1e-8 * scale);
            assert_symmetric_psd_with_nullity(&s, 1);
        }
    }

    /// Cardinal B-splines satisfy an exact dyadic two-scale relation. Folding
    /// it around the circle refines a periodic spline from `n` to `2n`
    /// coefficients without changing the represented function or its energy.
    #[test]
    fn cyclic_penalty_is_invariant_under_exact_dyadic_refinement() {
        let (degree, order, n, period) = (3usize, 2usize, 7usize, 2.3_f64);
        let coarse =
            Array1::from_iter((0..n).map(|i| (0.7 + 1.9 * i as f64).sin() + 0.2 * i as f64));
        let mut fine = Array1::<f64>::zeros(2 * n);
        let refinement_scale = 2.0_f64.powi(-(degree as i32));
        for i in 0..n {
            for k in 0..=degree + 1 {
                fine[(2 * i + k) % (2 * n)] +=
                    refinement_scale * binomial(degree + 1, k) * coarse[i];
            }
        }

        let points = Array1::linspace(0.0, period, 137);
        let (coarse_basis, _) = crate::basis::cyclic::create_cyclic_bspline_basis_dense(
            points.view(),
            0.0,
            period,
            degree,
            n,
        )
        .unwrap();
        let (fine_basis, _) = crate::basis::cyclic::create_cyclic_bspline_basis_dense(
            points.view(),
            0.0,
            period,
            degree,
            2 * n,
        )
        .unwrap();
        let value_error = (&coarse_basis.dot(&coarse) - &fine_basis.dot(&fine))
            .iter()
            .fold(0.0_f64, |error, value| error.max(value.abs()));
        assert!(
            value_error < 1e-12,
            "dyadic refinement changed f by {value_error}"
        );

        let coarse_s = cyclic_bspline_derivative_penalty_matrix(degree, n, period, order).unwrap();
        let fine_s =
            cyclic_bspline_derivative_penalty_matrix(degree, 2 * n, period, order).unwrap();
        let coarse_energy = coarse.dot(&coarse_s.dot(&coarse));
        let fine_energy = fine.dot(&fine_s.dot(&fine));
        let relative_error = (coarse_energy - fine_energy).abs() / coarse_energy.abs().max(1.0);
        assert!(
            relative_error < 1e-12,
            "dyadic refinement changed roughness: coarse={coarse_energy}, fine={fine_energy}, rel={relative_error}"
        );
    }

    /// Independent oracle: for a random periodic spline, βᵀSβ must equal the
    /// numerically integrated squared second derivative of the function, with
    /// the derivative taken by finite differences of the VALUE basis (the
    /// value evaluator is a separate code path from the derivative
    /// recurrence used in assembly).
    #[test]
    fn cyclic_penalty_matches_value_basis_finite_difference_integral() {
        let (degree, n) = (3usize, 7usize);
        let period = 2.0_f64;
        let order = 2usize;
        let s = cyclic_bspline_derivative_penalty_matrix(degree, n, period, order).unwrap();
        let beta = Array1::from_iter(
            (0..n).map(|i| ((i as f64) * 2.4 + 0.7).sin() * (1.0 + i as f64 * 0.1)),
        );

        let eval = |x: f64| -> f64 {
            let pts = Array1::from(vec![crate::basis::cyclic::wrap_to_period(x, 0.0, period)]);
            let (b, _) = crate::basis::cyclic::create_cyclic_bspline_basis_dense(
                pts.view(),
                0.0,
                period,
                degree,
                n,
            )
            .unwrap();
            (0..n).map(|j| b[[0, j]] * beta[j]).sum()
        };
        // Midpoint rule over a fine grid; central second difference of the value.
        let grid = 4000usize;
        let hg = period / grid as f64;
        let fd_h = 1e-4_f64;
        let mut integral = 0.0_f64;
        for g in 0..grid {
            let x = (g as f64 + 0.5) * hg;
            let d2 = (eval(x + fd_h) - 2.0 * eval(x) + eval(x - fd_h)) / (fd_h * fd_h);
            integral += d2 * d2 * hg;
        }
        let exact = beta.dot(&s.dot(&beta));
        let rel = (exact - integral).abs() / exact.max(1e-12);
        assert!(
            rel < 1e-3,
            "cyclic ∮(f'')²: Gram {exact} vs FD integral {integral} (rel {rel})"
        );
    }

    /// Covariant scaling: stretching the period by `c` scales the order-`m`
    /// roughness Gram by `c^{1−2m}` (so after the builder's Frobenius
    /// normalization the shipped penalty is unit-invariant). Compare in matrix
    /// max-norm: analytically zero lags carry quadrature-roundoff residues whose
    /// entrywise relative error is undefined.
    #[test]
    fn cyclic_penalty_scales_covariantly_with_period() {
        let (degree, n, order) = (3usize, 9usize, 2usize);
        let s1 = cyclic_bspline_derivative_penalty_matrix(degree, n, 1.0, order).unwrap();
        for c in [3.5_f64, 1e-13] {
            let s2 = cyclic_bspline_derivative_penalty_matrix(degree, n, c, order).unwrap();
            let factor = c.powi(1 - 2 * order as i32);
            let expected_scale = s1
                .iter()
                .map(|&base| (base * factor).abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let max_absolute_error = s1
                .iter()
                .zip(s2.iter())
                .map(|(&base, &observed)| (base * factor - observed).abs())
                .fold(0.0_f64, f64::max);
            let relative_max_norm_error = max_absolute_error / expected_scale;
            assert!(
                relative_max_norm_error < 1e-12,
                "period scaling must be c^(1-2m) for c={c}; max-norm rel={relative_max_norm_error}"
            );
        }
    }
}
