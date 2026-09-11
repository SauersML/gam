//! Cumulative-sum / derivative-control coefficient transforms for the
//! shape-constrained smooth arm.

use crate::basis::BasisError;
use ndarray::{Array1, Array2};

pub(crate) fn cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += values[i].exp();
        out[i] = sign * run;
    }
    out
}

pub(crate) fn second_cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let first = cumulative_exp(values, sign);
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += first[i];
        out[i] = run;
    }
    out
}

pub(crate) fn cumulative_sum_transform_matrix(dim: usize, order: usize, sign: f64) -> Array2<f64> {
    // The loop form `t = next^order` where `next` is the unit lower-triangular
    // matrix of ones (`next[i,j] = 1` iff `i >= j`) has the closed form
    //     T[i,j] = C(i - j + order - 1, order - 1)   for i >= j, else 0.
    // Order 0 yields the identity (consistent with the loop, which leaves
    // t = I when `order == 0`).
    let mut t = Array2::<f64>::zeros((dim, dim));
    if order == 0 {
        for i in 0..dim {
            t[[i, i]] = 1.0;
        }
    } else {
        let k = order - 1;
        for i in 0..dim {
            for j in 0..=i {
                t[[i, j]] = binomial(i - j + k, k) as f64;
            }
        }
    }
    if sign < 0.0 {
        t.mapv_inplace(|v| -v);
    }
    t
}

/// Knot-span-scaled second-order cumulative transform for the convex/concave
/// box reparameterization on a B-spline coefficient vector `θ` (raw control
/// points; the shape-constrained B-spline arm forces
/// `BSplineIdentifiability::None`, so the design columns are the raw basis
/// functions and `θ` carries the control-polygon geometry).
///
/// The plain integer second-difference cone `θ_{i+2} − 2θ_{i+1} + θ_i ≥ 0`
/// only certifies convexity of the *function* when the Greville abscissae are
/// evenly spaced. gam's B-splines are clamped (boundary knots repeated
/// `degree + 1` times), so even on a uniform breakpoint grid the Greville
/// abscissae `ξ_j = (1/d)·Σ_{k=1}^{d} t_{j+k}` cluster toward the ends and are
/// **not** uniform (and `knot_placement="quantile"` makes them more skewed
/// still). The geometrically-correct convexity cone is that the control-polygon
/// *slopes* `m_i = (θ_{i+1} − θ_i)/(ξ_{i+1} − ξ_i)` are non-decreasing, i.e. the
/// second *divided* differences `[D²θ]_i = (m_{i+1} − m_i)/(ξ_{i+2} − ξ_i) ≥ 0`.
/// This is the exact same divided-difference correction the difference-penalty
/// path applies (see `create_difference_penalty_matrix` /
/// `penalty_greville_abscissae_for_knots`): a coefficient sequence linear in
/// `x` (`θ_j = a + b·ξ_j`, the unpenalized affine null space, which must be a
/// boundary of both the convex and concave cones) has zero second divided
/// difference but a *non-zero* plain second difference under non-uniform ξ, so
/// the plain cone silently mis-orients the constraint.
///
/// Returns `T` (`p × p`) such that `θ = T·γ`. For `i ≥ 2`, `γ_i` is a positive
/// common rescaling of `sign · [D²θ]_{i−2}`; the common scale comes from
/// normalizing the supplied spans by their maximum so the coefficient chart is
/// invariant to physical covariate units. Pairing this `T` with the lower bounds
/// `γ_i ≥ 0` (`i ≥ 2`) from [`shape_lower_bounds_local`] therefore enforces
/// convexity (`sign = +1`) or concavity (`sign = −1`) exactly for arbitrary
/// (clamped / quantile) knot geometry. `γ_0` is the level and `γ_1` the initial
/// slope, both unconstrained. When ξ is uniform this reduces (column-scaled) to
/// `cumulative_sum_transform_matrix(p, 2, sign)`, recovering the original path.
pub(crate) fn convex_derivative_control_transform_matrix(
    first_span: &Array1<f64>,
    sign: f64,
) -> Result<Array2<f64>, BasisError> {
    let p = first_span.len() + 1;
    if first_span.is_empty() {
        crate::bail_invalid_basis!(
            "convex/concave box reparameterization requires at least 2 basis functions; found {p}"
        );
    }
    // These spans come directly from knot-window differences through
    // `bspline_first_derivative_control_spans`, rather than by subtracting two
    // separately averaged Greville abscissae. Revalidate at this numeric seam so
    // no degenerate transform can be constructed by a future caller.
    let span_scale = first_span.iter().copied().fold(0.0_f64, f64::max);
    for (i, &span) in first_span.iter().enumerate() {
        if !span.is_finite() || span <= 0.0 {
            crate::bail_invalid_basis!(
                "convex/concave box reparameterization requires positive finite derivative-control spans; span[{i}]={span:.3e}"
            );
        }
    }
    // A common positive rescaling changes only the free coordinate units, not
    // the represented cone. Normalize once to keep T finite and invariant to a
    // change of physical covariate units; every penalty is transformed by the
    // matching congruence downstream.
    let first_span = first_span.mapv(|span| span / span_scale);

    // Build T column by column: T[:, c] = θ for γ = e_c. Forward-accumulate
    // m_0 = γ_1, m_{i+1} = m_i + (ξ_{i+2} − ξ_i)·γ_{i+2}, θ_0 = γ_0,
    // θ_{i+1} = θ_i + (ξ_{i+1} − ξ_i)·m_i.
    let mut t = Array2::<f64>::zeros((p, p));
    for c in 0..p {
        // γ = e_c
        let gamma1 = if c == 1 { 1.0 } else { 0.0 };
        let mut m = Array1::<f64>::zeros(p - 1);
        m[0] = gamma1;
        for i in 0..p - 2 {
            // contribution of γ_{i+2} to the divided second difference
            let gamma_ip2 = if c == i + 2 { 1.0 } else { 0.0 };
            let second_span = first_span[i] + first_span[i + 1];
            m[i + 1] = m[i] + second_span * gamma_ip2;
        }
        let theta0 = if c == 0 { 1.0 } else { 0.0 };
        t[[0, c]] = theta0;
        for i in 0..p - 1 {
            t[[i + 1, c]] = t[[i, c]] + first_span[i] * m[i];
        }
    }
    if sign < 0.0 {
        t.mapv_inplace(|v| -v);
    }
    Ok(t)
}

/// Small integer binomial coefficient C(n, k). Used to build the
/// cumulative-sum transform; n stays small (dim plus a tiny order).
fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u64 = 1;
    for i in 0..k {
        num = num * (n - i) as u64 / (i + 1) as u64;
    }
    num
}

#[cfg(test)]
mod cumulative_sum_transform_tests {
    use super::cumulative_sum_transform_matrix;
    use ndarray::Array2;

    fn reference(dim: usize, order: usize, sign: f64) -> Array2<f64> {
        let mut t = Array2::<f64>::eye(dim);
        for _ in 0..order {
            let mut next = Array2::<f64>::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..=i {
                    next[[i, j]] = 1.0;
                }
            }
            t = t.dot(&next);
        }
        if sign < 0.0 {
            t.mapv_inplace(|v| -v);
        }
        t
    }

    #[test]
    fn closed_form_matches_loop() {
        for &dim in &[5usize, 10] {
            for &order in &[1usize, 2, 3, 4] {
                for &sign in &[1.0_f64, -1.0] {
                    let got = cumulative_sum_transform_matrix(dim, order, sign);
                    let want = reference(dim, order, sign);
                    for i in 0..dim {
                        for j in 0..dim {
                            assert!(
                                (got[[i, j]] - want[[i, j]]).abs() < 1e-12,
                                "mismatch at dim={dim} order={order} sign={sign} ({i},{j}): \
                                 got {} want {}",
                                got[[i, j]],
                                want[[i, j]],
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod convex_divided_difference_transform_tests {
    use super::{convex_derivative_control_transform_matrix, cumulative_sum_transform_matrix};
    use crate::smooth::{
        ShapeConstraint,
        shape_constraints::{
            bspline_first_derivative_control_spans, bspline_shape_linear_constraints,
        },
    };
    use ndarray::Array1;

    fn first_spans(g: &Array1<f64>) -> Array1<f64> {
        Array1::from_iter((0..g.len() - 1).map(|i| g[i + 1] - g[i]))
    }

    /// Second *divided* difference of `theta` over Greville abscissae `g`:
    /// `(m_{i+1} − m_i)/(g_{i+2} − g_i)` with `m_i = (θ_{i+1}−θ_i)/(g_{i+1}−g_i)`.
    fn second_divided_difference(theta: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        let p = theta.len();
        let mut m = Array1::<f64>::zeros(p - 1);
        for i in 0..p - 1 {
            m[i] = (theta[i + 1] - theta[i]) / (g[i + 1] - g[i]);
        }
        let mut d = Array1::<f64>::zeros(p - 2);
        for i in 0..p - 2 {
            d[i] = (m[i + 1] - m[i]) / (g[i + 2] - g[i]);
        }
        d
    }

    #[test]
    fn uniform_greville_matches_integer_transform_up_to_column_scale() {
        // Uniformly spaced abscissae must reproduce the integer second-difference
        // transform on the *cone coordinates* up to a positive per-column scale,
        // so the γ ≥ 0 convexity cone (columns c ≥ 2) is identical — guaranteeing
        // no regression on the common uniform case.
        //
        // The two *affine* columns (c ∈ {0, 1}) are NOT individually column-scaled
        // multiples of the plain transform's first two columns: the divided-
        // difference transform parameterizes the affine null space as
        // θ_j = γ_0 + γ_1·(ξ_j − ξ_0) (a genuine level + slope), whereas the plain
        // double-cumulative-sum's column 0 is the discrete second integral of e_0,
        // a non-constant ramp θ_{i,0} = i + 1. Both bases span the *same* 2-D space
        // of vectors affine in ξ, so the cone (which lives entirely on c ≥ 2) is
        // unchanged. The correct uniform-reduction identity is therefore: per-column
        // positive scale on the cone columns, and affine-subspace equality on the
        // first two columns — asserting a per-column scale there would be wrong.
        let p = 7;
        let g = Array1::from_iter((0..p).map(|i| i as f64));
        for &sign in &[1.0_f64, -1.0] {
            let t = convex_derivative_control_transform_matrix(&first_spans(&g), sign).unwrap();
            let plain = cumulative_sum_transform_matrix(p, 2, sign);

            // Cone coordinates (c ≥ 2): identical up to a positive per-column scale.
            for c in 2..p {
                // Find a non-zero reference entry in the plain column to fix the scale.
                let mut scale: Option<f64> = None;
                for i in 0..p {
                    if plain[[i, c]].abs() > 1e-9 {
                        scale = Some(t[[i, c]] / plain[[i, c]]);
                        break;
                    }
                }
                let s = scale.expect("plain cone column must have a non-zero entry");
                assert!(s > 0.0, "column {c} scale must be positive, got {s}");
                for i in 0..p {
                    assert!(
                        (t[[i, c]] - s * plain[[i, c]]).abs() < 1e-9,
                        "uniform reduction mismatch at ({i},{c})"
                    );
                }
            }

            // Affine columns (c ∈ {0, 1}): each column of *both* transforms must be
            // exactly affine in the abscissae ξ (i.e. lie in span{1, ξ}). With p ≥ 3
            // the affine fit is overdetermined, so an exact fit on all p rows proves
            // the column lies in the 2-D affine null space — hence the two transforms
            // share that null space and the convexity cone is identical.
            for c in 0..2usize {
                for col in [&t, &plain] {
                    // Closed-form least-squares fit of θ_i = a + b·g_i, then assert
                    // the residual is zero (an exact affine relation).
                    let n = p as f64;
                    let sum_g: f64 = g.iter().copied().sum();
                    let sum_gg: f64 = g.iter().map(|&x| x * x).sum();
                    let sum_y: f64 = (0..p).map(|i| col[[i, c]]).sum();
                    let sum_gy: f64 = (0..p).map(|i| g[i] * col[[i, c]]).sum();
                    let det = n * sum_gg - sum_g * sum_g;
                    let b = (n * sum_gy - sum_g * sum_y) / det;
                    let a = (sum_y - b * sum_g) / n;
                    for i in 0..p {
                        let fit = a + b * g[i];
                        assert!(
                            (col[[i, c]] - fit).abs() < 1e-9,
                            "affine column {c} entry {i} is not affine in ξ: \
                             got {}, affine fit {fit}",
                            col[[i, c]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nonneg_gamma_certifies_convexity_for_nonuniform_greville() {
        // Clustered (clamped/quantile-like) abscissae: γ_{≥2} ≥ 0 must certify a
        // non-negative second *divided* difference (true function convexity).
        let g = Array1::from(vec![0.0, 0.1, 0.3, 0.7, 1.4, 2.6, 4.5]);
        let spans = first_spans(&g);
        let span_scale = spans.iter().copied().fold(0.0_f64, f64::max);
        let t = convex_derivative_control_transform_matrix(&spans, 1.0).unwrap();
        // γ_0, γ_1 arbitrary (level/slope); γ_{≥2} ≥ 0 (convex cone interior).
        let gamma = Array1::from(vec![-2.0, 1.5, 0.4, 0.0, 1.2, 0.7, 0.9]);
        let theta = t.dot(&gamma);
        let d2 = second_divided_difference(&theta, &g);
        for (i, &v) in d2.iter().enumerate() {
            assert!(
                v >= -1e-9,
                "convex cone violated: second divided difference d2[{i}] = {v:.3e} < 0"
            );
            // The transform normalizes all spans by their maximum. In the
            // original physical x units this makes the second divided
            // difference γ/span_scale²: a positive coordinate rescaling that
            // leaves the cone unchanged.
            let expected = gamma[i + 2] / (span_scale * span_scale);
            assert!(
                (v - expected).abs() < 1e-9,
                "cone coordinate mismatch at {i}: d2 = {v:.3e}, expected = {expected:.3e}",
            );
        }
    }

    #[test]
    fn affine_coefficients_are_a_cone_boundary() {
        // A control vector linear in the abscissae (θ_j = a + b·g_j) is the
        // unpenalized affine null space and must lie on the boundary of BOTH the
        // convex and concave cones: every second divided difference is zero. The
        // PLAIN integer second difference would be NON-zero here under non-uniform
        // abscissae — exactly the silent mis-orientation this transform fixes.
        let g = Array1::from(vec![0.0, 0.1, 0.3, 0.7, 1.4, 2.6, 4.5]);
        let theta = g.mapv(|x| 3.0 - 2.0 * x);
        let d2 = second_divided_difference(&theta, &g);
        for (i, &v) in d2.iter().enumerate() {
            assert!(
                v.abs() < 1e-9,
                "affine control polygon must have zero second divided difference at {i}, got {v:.3e}"
            );
        }
        // Sanity: the plain integer second difference is genuinely non-zero,
        // confirming the divided-difference correction is load-bearing.
        let mut any_nonzero_plain = false;
        for i in 0..theta.len() - 2 {
            let plain = theta[i + 2] - 2.0 * theta[i + 1] + theta[i];
            if plain.abs() > 1e-6 {
                any_nonzero_plain = true;
            }
        }
        assert!(
            any_nonzero_plain,
            "non-uniform abscissae should make the plain second difference non-zero on affine data"
        );
    }

    #[test]
    fn box_transforms_realize_the_canonical_linear_cones() {
        let knots = Array1::from(vec![
            0.0, 0.0, 0.0, 0.0, 0.08, 0.37, 0.62, 1.0, 1.0, 1.0, 1.0,
        ]);
        let spans = bspline_first_derivative_control_spans(knots.view(), 3).unwrap();
        let p = spans.len() + 1;
        for shape in [
            ShapeConstraint::MonotoneIncreasing,
            ShapeConstraint::MonotoneDecreasing,
            ShapeConstraint::Convex,
            ShapeConstraint::Concave,
        ] {
            let (order, sign) = match shape {
                ShapeConstraint::MonotoneIncreasing => (1, 1.0),
                ShapeConstraint::MonotoneDecreasing => (1, -1.0),
                ShapeConstraint::Convex => (2, 1.0),
                ShapeConstraint::Concave => (2, -1.0),
                ShapeConstraint::None => unreachable!(),
            };
            let transform = if order == 1 {
                cumulative_sum_transform_matrix(p, order, sign)
            } else {
                convex_derivative_control_transform_matrix(&spans, sign).unwrap()
            };
            let constraints = bspline_shape_linear_constraints(knots.view(), 3, shape)
                .unwrap()
                .unwrap();
            let mapped = constraints.a.dot(&transform);
            for row in 0..mapped.nrows() {
                for col in 0..mapped.ncols() {
                    if col == row + order {
                        assert!(
                            mapped[[row, col]] > 0.0,
                            "shape={shape:?} mapped cone diagonal must be positive"
                        );
                    } else {
                        assert!(
                            mapped[[row, col]].abs() <= 64.0 * f64::EPSILON,
                            "shape={shape:?} A*T must isolate cone coordinate; ({row},{col})={} ",
                            mapped[[row, col]],
                        );
                    }
                }
            }
        }
    }
}
