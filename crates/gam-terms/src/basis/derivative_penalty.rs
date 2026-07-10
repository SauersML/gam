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
//! For `m = p + 1` the classical difference penalty has no `∫ (f^{(m)})²`
//! analogue (the distributional derivative is a sum of Dirac masses with
//! infinite L² energy). The parameterization-invariant functional that fills
//! this slot is the squared-jump seminorm
//!
//! ```text
//! J_{p+1}(f) = Σ_t ( [f^{(p)}]_t )²,
//! ```
//!
//! the sum over breakpoints `t` of the squared jump of the highest existing
//! derivative. It is a functional of `f` (not of the coefficients), its null
//! space is exactly the polynomials of degree `≤ p` (dimension `m`, matching
//! the difference penalty's), and adding knots that the function does not bend
//! at contributes zero. This covers e.g. degree-1 bases with the default
//! order-2 penalty: the penalty is the squared change in slope at each
//! breakpoint. Orders `m > p + 1` are rejected as a typed error — no finite
//! function-level roughness of that order exists for splines of degree `p`.
//!
//! Repeated interior knots lower the continuity class; where the true
//! `∫ (f^{(m)})²` acquires Dirac contributions (knot multiplicity
//! `> p − m + 1`) the Gram integrates the absolutely continuous part, the
//! standard convention for exact spline penalties. Both entry points return a
//! symmetric PSD matrix.

use super::*;
use gam_math::special::gauss_legendre;

/// Exact open/clamped B-spline roughness penalty
/// `S_ij = ∫ B_i^{(order)} B_j^{(order)} dx` over the modeling interval
/// `[t_degree, t_{num_basis}]`, with the `order = degree + 1` squared-jump
/// seminorm extension described in the module docs.
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
    if order > degree + 1 {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: order,
            minimum_degree: order - 1,
        });
    }

    let mut s = Array2::<f64>::zeros((num_basis, num_basis));
    // The modeling interval is covered by spans `[t_k, t_{k+1}]` for
    // `k = degree .. num_basis`; clamped boundary knots make the exterior
    // spans degenerate and they carry no integral mass.
    if order <= degree {
        accumulate_derivative_gram_spans(
            knot_vector,
            degree,
            order,
            degree..num_basis,
            num_basis,
            |col| col,
            &mut s,
        )?;
    } else {
        accumulate_jump_seminorm_open(knot_vector, degree, num_basis, &mut s)?;
    }
    symmetrize_in_place(&mut s);
    Ok(s)
}

/// Exact cyclic (periodic) B-spline roughness penalty over one full period:
/// `S_ab = ∮ B_a^{(order)}(θ) B_b^{(order)}(θ) dθ` for the wrapped uniform
/// basis of `num_basis` cardinal translates with spacing `h = period /
/// num_basis`, with the `order = degree + 1` squared-jump seminorm extension
/// (jumps at all `num_basis` knots on the circle, seam included).
///
/// This is the function-space replacement for
/// [`create_cyclic_difference_penalty_matrix`]: same `num_basis × num_basis`
/// shape, same single-dimensional null space (the constants — the only
/// periodic polynomials), but an exact circulant functional of the function on
/// the circle. The Gram is independent of the knot anchor (integration over
/// the whole circle), so no origin argument is needed.
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
    if order > degree + 1 {
        return Err(BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order: order,
            minimum_degree: order - 1,
        });
    }

    // The wrapped basis is `B_a(θ) = Σ_k C(θ/h − a − k·num_basis)` for the
    // cardinal degree-`p` B-spline `C`. Realize it as the OPEN uniform basis on
    // the extended knot line (the exact construction the cyclic design
    // evaluator folds, see `bspline_raw_row_chunk`) and fold columns modulo
    // `num_basis` while accumulating: with `num_basis > degree` no translate
    // overlaps its own wrap, so the fold is an exact identification. The Gram
    // is anchor-invariant, so the extended knots are anchored at zero.
    let knots = cyclic_uniform_knot_vector(0.0, period, degree, num_basis);
    let num_basis_extended = knots.len() - degree - 1;

    let mut s = Array2::<f64>::zeros((num_basis, num_basis));
    // One period = the `num_basis` spans `[t_k, t_{k+1}]`,
    // `k = degree .. degree + num_basis`, of the extended knot line.
    if order <= degree {
        accumulate_derivative_gram_spans(
            knots.view(),
            degree,
            order,
            degree..degree + num_basis,
            num_basis_extended,
            |col| col % num_basis,
            &mut s,
        )?;
    } else {
        accumulate_jump_seminorm_cyclic(knots.view(), degree, num_basis, &mut s)?;
    }
    symmetrize_in_place(&mut s);
    Ok(s)
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
        if width <= KNOT_SPAN_DEGENERACY_FLOOR {
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

/// Piecewise-constant values of `B_i^{(degree)}` on each listed span,
/// evaluated at span midpoints (the derivative of order `degree` is constant
/// per span, so the midpoint value IS the span value).
fn top_derivative_span_values(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    spans: &[usize],
    out_len: usize,
) -> Result<Vec<Vec<f64>>, BasisError> {
    let mut workspace = BsplineDerivativeWorkspace::new();
    let mut values = Vec::with_capacity(spans.len());
    for &k in spans {
        let mut row = vec![0.0_f64; out_len];
        let x = 0.5 * (knot_vector[k] + knot_vector[k + 1]);
        evaluate_bspline_derivative_recurrence_into(
            degree,
            x,
            knot_vector,
            degree,
            &mut row,
            &mut workspace,
            0,
        )?;
        values.push(row);
    }
    Ok(values)
}

/// `order = degree + 1` squared-jump seminorm for the open/clamped basis:
/// `S = Σ_t j_t j_tᵀ` over interior breakpoints `t`, where
/// `(j_t)_i = [B_i^{(degree)}]_t` is the jump across `t` (coincident interior
/// knots form a single breakpoint whose jump spans the whole group).
fn accumulate_jump_seminorm_open(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    num_basis: usize,
    s: &mut Array2<f64>,
) -> Result<(), BasisError> {
    let spans: Vec<usize> = (degree..num_basis)
        .filter(|&k| knot_vector[k + 1] - knot_vector[k] > KNOT_SPAN_DEGENERACY_FLOOR)
        .collect();
    let values = top_derivative_span_values(knot_vector, degree, &spans, num_basis)?;
    for pair in values.windows(2) {
        accumulate_jump_outer(&pair[0], &pair[1], |col| col, s);
    }
    Ok(())
}

/// `order = degree + 1` squared-jump seminorm on the circle: jumps of the
/// folded `B_a^{(degree)}` at all `num_basis` knots, including the seam
/// (last span wraps to the first).
fn accumulate_jump_seminorm_cyclic(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    num_basis: usize,
    s: &mut Array2<f64>,
) -> Result<(), BasisError> {
    let num_basis_extended = knot_vector.len() - degree - 1;
    let spans: Vec<usize> = (degree..degree + num_basis).collect();
    let values = top_derivative_span_values(knot_vector, degree, &spans, num_basis_extended)?;
    for k in 0..num_basis {
        let next = (k + 1) % num_basis;
        accumulate_jump_outer(&values[k], &values[next], |col| col % num_basis, s);
    }
    Ok(())
}

/// Adds the outer product of the (folded) jump vector `after − before` to `s`.
fn accumulate_jump_outer(
    before: &[f64],
    after: &[f64],
    fold: impl Fn(usize) -> usize,
    s: &mut Array2<f64>,
) {
    let n = s.nrows();
    let mut jump = vec![0.0_f64; n];
    for (col, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
        let d = a - b;
        if d != 0.0 {
            jump[fold(col)] += d;
        }
    }
    for (i, &ji) in jump.iter().enumerate() {
        if ji == 0.0 {
            continue;
        }
        for (j, &jj) in jump.iter().enumerate() {
            if jj != 0.0 {
                s[[i, j]] += ji * jj;
            }
        }
    }
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
            beta[i] = e[r] / crate::basis::cyclic::binomial(degree, r);
        }
        beta
    }

    fn clamped_knots(interior: &[f64], degree: usize, a: f64, b: f64) -> Array1<f64> {
        let mut v = vec![a; degree + 1];
        v.extend_from_slice(interior);
        v.extend(std::iter::repeat_n(b, degree + 1));
        Array1::from(v)
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
            assert!((j2 - 4.0).abs() < 1e-9, "∫(f'')² for f=x² must be 4, got {j2}");
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
                assert!(
                    energy.abs() < 1e-10,
                    "x^{r} must lie in the order-{order} null space; energy {energy}"
                );
            }
            let beta = monomial_coefficients(knots.view(), degree, order);
            assert!(
                beta.dot(&s.dot(&beta)) > 1e-6,
                "x^{order} must be penalized at order {order}"
            );
            // Symmetric PSD on random directions.
            for seed in 0..8u64 {
                let n = s.nrows();
                let beta = Array1::from_iter(
                    (0..n).map(|i| (((seed + 1) as f64 * 37.0 + i as f64 * 17.3).sin())),
                );
                let e = beta.dot(&s.dot(&beta));
                assert!(e >= -1e-12, "Gram must be PSD; got βᵀSβ = {e}");
            }
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
                        order, x, knots.view(), degree, &mut row, &mut ws, 0,
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
            let max_err = (&s - &s_over)
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            let scale = s.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            assert!(
                max_err < 1e-11 * scale,
                "order {order}: minimal rule differs from oversampled rule by {max_err}"
            );
        }
    }

    /// `order = degree + 1` (the auto-shrink degree-1 default-order-2 case):
    /// the squared-jump seminorm. For hat functions with values v_k at the
    /// breakpoints, the penalty is Σ (slope change at interior breakpoints)².
    #[test]
    fn jump_seminorm_matches_hand_computed_slope_changes() {
        let degree = 1usize;
        let interior = [0.3, 0.7];
        let knots = clamped_knots(&interior, degree, 0.0, 1.0);
        let s = bspline_derivative_penalty_matrix(knots.view(), degree, 2).unwrap();
        // Degree-1 basis: coefficients are the function values at 0, .3, .7, 1.
        let v = array![0.0_f64, 1.0, -0.5, 2.0];
        let x = [0.0_f64, 0.3, 0.7, 1.0];
        let slope = |a: usize| (v[a + 1] - v[a]) / (x[a + 1] - x[a]);
        let expected = (slope(1) - slope(0)).powi(2) + (slope(2) - slope(1)).powi(2);
        let got = v.dot(&s.dot(&v));
        assert!(
            (got - expected).abs() < 1e-10,
            "jump seminorm {got} != hand-computed {expected}"
        );
        // Null space: exactly the globally linear functions (dimension 2).
        let lin = array![0.0_f64, 0.3, 0.7, 1.0];
        assert!(lin.dot(&s.dot(&lin)).abs() < 1e-12);
        let ones = Array1::<f64>::ones(4);
        assert!(ones.dot(&s.dot(&ones)).abs() < 1e-12);
    }

    /// Orders with no finite function-level roughness are typed errors.
    #[test]
    fn order_beyond_degree_plus_one_is_rejected() {
        let degree = 1usize;
        let knots = clamped_knots(&[0.2, 0.4, 0.6, 0.8], degree, 0.0, 1.0);
        let err = bspline_derivative_penalty_matrix(knots.view(), degree, 3).unwrap_err();
        assert!(matches!(
            err,
            BasisError::InsufficientDegreeForDerivative { .. }
        ));
        let err = cyclic_bspline_derivative_penalty_matrix(1, 8, 1.0, 3).unwrap_err();
        assert!(matches!(
            err,
            BasisError::InsufficientDegreeForDerivative { .. }
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
        }
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
        let beta =
            Array1::from_iter((0..n).map(|i| ((i as f64) * 2.4 + 0.7).sin() * (1.0 + i as f64 * 0.1)));

        let eval = |x: f64| -> f64 {
            let pts = Array1::from(vec![crate::basis::cyclic::wrap_to_period(x, 0.0, period)]);
            let (b, _) =
                crate::basis::cyclic::create_cyclic_bspline_basis_dense(pts.view(), 0.0, period, degree, n)
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
        let c = 3.5_f64;
        let s2 = cyclic_bspline_derivative_penalty_matrix(degree, n, c, order).unwrap();
        let factor = c.powi(1 - 2 * order as i32);
        let max_err = s1
            .iter()
            .zip(s2.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a * factor - b).abs()));
        let scale = s2.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            max_err < 1e-12 * scale.max(1.0),
            "period scaling must be exactly c^(1-2m); err {max_err}"
        );
    }

    /// Cyclic jump seminorm (`order = degree + 1`) on a degree-1 ring:
    /// hand-check against slope changes of the periodic piecewise-linear
    /// interpolant, seam included.
    #[test]
    fn cyclic_jump_seminorm_matches_hand_computed() {
        let (degree, n, period) = (1usize, 4usize, 4.0_f64);
        let s = cyclic_bspline_derivative_penalty_matrix(degree, n, period, 2).unwrap();
        // Degree-1 wrapped basis: coefficients are values at knots 0,1,2,3 (h=1).
        let v = array![1.0_f64, 3.0, 0.0, -2.0];
        let slope = |k: usize| v[(k + 1) % n] - v[k]; // h = 1
        let expected: f64 = (0..n).map(|k| (slope((k + 1) % n) - slope(k)).powi(2)).sum();
        let got = v.dot(&s.dot(&v));
        assert!(
            (got - expected).abs() < 1e-10,
            "cyclic jump seminorm {got} != hand value {expected}"
        );
        let ones = Array1::<f64>::ones(n);
        assert!(ones.dot(&s.dot(&ones)).abs() < 1e-12);
    }
}
