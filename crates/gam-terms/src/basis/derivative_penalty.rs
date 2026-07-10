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

    let mut s = Array2::<f64>::zeros((num_basis, num_basis));
    // The modeling interval is covered by spans `[t_k, t_{k+1}]` for
    // `k = degree .. num_basis`; clamped boundary knots make the exterior
    // spans degenerate and they carry no integral mass.
    accumulate_derivative_gram_spans(
        normalized_knots.view(),
        degree,
        order,
        degree..num_basis,
        num_basis,
        |col| col,
        &mut s,
    )?;
    rescale_derivative_gram(&mut s, domain_scale, order)?;
    symmetrize_in_place(&mut s);
    Ok(s)
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

    let mut s = Array2::<f64>::zeros((num_basis, num_basis));
    // One period = the `num_basis` spans `[t_k, t_{k+1}]`,
    // `k = degree .. degree + num_basis`, of the extended knot line.
    accumulate_derivative_gram_spans(
        knots.view(),
        degree,
        order,
        degree..degree + num_basis,
        num_basis_extended,
        |col| col % num_basis,
        &mut s,
    )?;
    rescale_derivative_gram(&mut s, period, order)?;
    symmetrize_in_place(&mut s);
    Ok(s)
}

/// Maps the open spline's modeling interval to `[0, 1]`. Assembly in this
/// dimensionless coordinate keeps the evaluator's numerical knot-span floor
/// relative to the modeled domain rather than to the user's physical units.
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
    for k in degree..num_basis {
        let width = normalized[k + 1] - normalized[k];
        if width > 0.0 && width <= KNOT_SPAN_DEGENERACY_FLOOR {
            return Err(BasisError::InvalidKnotVector(format!(
                "positive knot span [{}, {}] has relative width {width:.3e}, too small for stable exact roughness assembly",
                knot_vector[k],
                knot_vector[k + 1]
            )));
        }
    }
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

/// Applies the exact coordinate covariance to a unit-domain derivative Gram
/// and rejects scales whose floating-point representation would change the
/// operator's rank. A finite scale factor alone is not sufficient: multiplying
/// a large unit-domain entry can still overflow, while a positive subnormal
/// factor can underflow one or more basis-function energies to zero.
fn rescale_derivative_gram(
    gram: &mut Array2<f64>,
    domain_scale: f64,
    order: usize,
) -> Result<(), BasisError> {
    let coordinate_scale = derivative_gram_coordinate_scale(domain_scale, order)?;
    gram.mapv_inplace(|value| value * coordinate_scale);
    let entries_are_finite = gram.iter().all(|value| value.is_finite());
    let preserves_all_basis_diagonals = (0..gram.nrows()).all(|i| gram[[i, i]] > 0.0);
    if !entries_are_finite || !preserves_all_basis_diagonals {
        return Err(BasisError::InvalidInput(format!(
            "order-{order} roughness Gram over domain width {domain_scale} is not representable"
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
fn accumulate_derivative_gram_spans(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    order: usize,
    spans: std::ops::Range<usize>,
    out_len: usize,
    fold: impl Fn(usize) -> usize,
    s: &mut Array2<f64>,
) -> Result<(), BasisError> {
    // Integrand degree per span is 2(p − m); `p − m + 1` Gauss points are
    // exact through degree 2(p − m) + 1.
    let quad_points = degree - order + 1;
    let (nodes, weights) = gauss_legendre(quad_points);
    let mut row = vec![0.0_f64; out_len];
    let mut workspace = BsplineDerivativeWorkspace::new();

    for k in spans {
        let left = knot_vector[k];
        let right = knot_vector[k + 1];
        let width = right - left;
        if width <= 0.0 {
            // Validated knots are non-decreasing, so a non-positive width is a
            // zero-length span with no quadrature mass.
            continue;
        }
        let mid = 0.5 * (left + right);
        let half = 0.5 * width;
        for (node, weight) in nodes.iter().zip(weights.iter()) {
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
            let w = weight * half;
            let support_start = k - degree;
            for i in support_start..=k {
                let vi = row[i];
                if vi == 0.0 {
                    continue;
                }
                let fi = fold(i);
                for j in support_start..=k {
                    let vj = row[j];
                    if vj != 0.0 {
                        s[[fi, fold(j)]] += w * vi * vj;
                    }
                }
            }
        }
    }
    Ok(())
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
                    assert!(
                        relative_error < 1e-10,
                        "degree={degree}, order={order}, polynomial degree={polynomial_degree}: expected {expected}, observed {observed}, relative error {relative_error}"
                    );
                }
            }
        }
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
                if b - a <= KNOT_SPAN_DEGENERACY_FLOOR {
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
    /// roughness Gram by exactly `c^{1−2m}` (so after the builder's Frobenius
    /// normalization the shipped penalty is unit-invariant).
    #[test]
    fn cyclic_penalty_scales_covariantly_with_period() {
        let (degree, n, order) = (3usize, 9usize, 2usize);
        let s1 = cyclic_bspline_derivative_penalty_matrix(degree, n, 1.0, order).unwrap();
        for c in [3.5_f64, 1e-13] {
            let s2 = cyclic_bspline_derivative_penalty_matrix(degree, n, c, order).unwrap();
            let factor = c.powi(1 - 2 * order as i32);
            let max_relative_error = s1
                .iter()
                .zip(s2.iter())
                .map(|(&base, &observed)| {
                    (base * factor - observed).abs()
                        / observed.abs().max((base * factor).abs()).max(1.0)
                })
                .fold(0.0_f64, f64::max);
            assert!(
                max_relative_error < 1e-12,
                "period scaling must be c^(1-2m) for c={c}; rel={max_relative_error}"
            );
        }
    }
}
