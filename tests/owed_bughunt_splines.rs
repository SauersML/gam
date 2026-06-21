//! Bug-hunt lane regression gates for the spline / basis / penalty subsystem.
//!
//! Each test pins a property derived from the math the code is supposed to
//! implement, so a regression of a landed correctness fix fails CI. Tests use
//! only the public crate API.
//!
//! ## Pinned: divided-difference penalty scale-equivariance (#1364)
//!
//! `create_difference_penalty_matrix` normalizes each order's Greville spans by
//! their geometric mean so the penalty `S = DᵀD` is invariant to a global
//! rescaling of the covariate `x → c·x`. The raw divided-difference divisor
//! `g[i+o] − g[i]` carries the covariate's units, so without normalization `S`
//! scales as `c^(−2·order)`; REML's λ-search then drifts purely with the
//! covariate magnitude (the wine-scale underfit class, #1392). Dividing each
//! order's spans by their geometric mean makes the divisor a unitless
//! local/typical ratio, so `S(c·g)` is identical to `S(g)`. This is the
//! library-level mechanism `tests/owed_1392.rs` pins through the full fit path;
//! here we pin it directly on the penalty kernel across many decades of scale.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use gam::basis::{compute_greville_abscissae, create_difference_penalty_matrix};
use ndarray::{Array1, Array2};

/// A clamped cubic knot vector over `[a, a + width]` with `n_internal` interior
/// knots, i.e. degree-3 boundary knots repeated 4× at each end. This is the same
/// shape the P-spline builder produces; its Greville abscissae are non-uniform at
/// the clamped ends (so the divided-difference scaling path engages and the
/// scale-equivariance normalization is exercised).
fn clamped_cubic_knots(a: f64, width: f64, n_internal: usize) -> Array1<f64> {
    let degree = 3usize;
    let mut knots = Vec::new();
    for _ in 0..=degree {
        knots.push(a);
    }
    for i in 1..=n_internal {
        knots.push(a + width * (i as f64) / ((n_internal + 1) as f64));
    }
    for _ in 0..=degree {
        knots.push(a + width);
    }
    Array1::from_vec(knots)
}

/// Build the order-2 difference penalty from a clamped cubic knot vector of the
/// given `width` (covariate range), returned with the number of basis functions.
fn penalty_for_width(width: f64, n_internal: usize) -> (Array2<f64>, usize) {
    let degree = 3usize;
    let knots = clamped_cubic_knots(0.0, width, n_internal);
    let n_basis = knots.len() - degree - 1;
    let greville = compute_greville_abscissae(&knots, degree).expect("greville abscissae");
    let s = create_difference_penalty_matrix(n_basis, 2, Some(greville.view()))
        .expect("difference penalty must build");
    (s, n_basis)
}

/// The order-2 difference penalty must be invariant to the covariate rescaling
/// `x → c·x` (the #1364 scale-equivariance property): `S(c·g)` is bit-for-bit
/// (to a tight numerical floor) equal to `S(g)` across many decades of `c`.
/// Before #1364 the penalty scaled as `c^(−4)` at order 2, so REML selected a
/// different λ purely from the covariate's units.
#[test]
fn difference_penalty_value_is_scale_invariant_across_decades() {
    let (s_ref, n_basis) = penalty_for_width(1.0, 6);
    let frob_ref = s_ref.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
    assert!(
        frob_ref > 0.0 && s_ref.iter().all(|v| v.is_finite()),
        "reference penalty must be finite and non-trivial"
    );

    // Stay within the range where the Greville-abscissa degeneracy guard
    // (an absolute `g_max−g_min >= 1e-10` floor in `compute_greville_abscissae`)
    // is satisfied, so we isolate the divided-difference NORMALIZATION rather
    // than that separate guard: widths from 1e-6 to 1e6 all clear it.
    for &c in &[1.0e-6_f64, 1.0e-3, 0.5, 7.0, 1.0e3, 1.0e6] {
        let (s_c, n_c) = penalty_for_width(c, 6);
        assert_eq!(n_c, n_basis, "basis dimension is scale-independent");
        let mut worst = 0.0_f64;
        for i in 0..n_basis {
            for j in 0..n_basis {
                worst = worst.max((s_c[[i, j]] - s_ref[[i, j]]).abs());
            }
        }
        // Geometric-mean-span normalization cancels the units exactly; only
        // floating-point round-off in the log/exp of the reference span remains.
        // The pre-#1364 penalty drifted by `c^(±4)` — astronomically far above
        // this floor at c = 1e6.
        assert!(
            worst < 1.0e-9 * frob_ref,
            "order-2 difference penalty is NOT scale-invariant at width c={c:.0e}: \
             worst entry drift {worst:.3e} vs |S|_F {frob_ref:.3e} \
             (the divided-difference spans must be normalized by their \
             geometric mean, #1364)"
        );
    }
}

/// The order-2 difference penalty must annihilate the constant vector and the
/// Greville-abscissa ramp: `null(S) = {1, g}`. The divided-difference operator
/// of order 2 kills polynomials of degree < 2 evaluated at the Greville
/// abscissae `g` (NOT the integer coefficient index — once the non-uniform
/// per-row span scaling is applied, the linear null direction is `g` itself).
/// Per-row span scaling does not change which vectors `D` annihilates, so the
/// geometric-mean normalization (#1364) preserves this exact 2-dimensional null
/// space — the property that lets a P-spline collapse to a straight line under
/// heavy smoothing. A regression that mis-scaled or mis-built the divided
/// difference would lift `{1, g}` out of the null space.
#[test]
fn difference_penalty_annihilates_constant_and_greville_linear() {
    let degree = 3usize;
    let width = 3.7_f64;
    let n_internal = 6usize;
    let knots = clamped_cubic_knots(0.0, width, n_internal);
    let n_basis = knots.len() - degree - 1;
    let greville = compute_greville_abscissae(&knots, degree).expect("greville abscissae");
    let s = create_difference_penalty_matrix(n_basis, 2, Some(greville.view()))
        .expect("difference penalty must build");

    // The penalty acts in COEFFICIENT space. The 2nd-order divided-difference
    // operator annihilates any coefficient vector that is affine in the Greville
    // abscissae: the constant `1` and the abscissae `g` themselves.
    let ones = Array1::<f64>::from_elem(n_basis, 1.0);
    let g_lin = greville.clone();

    let quad = |v: &Array1<f64>| -> f64 { v.dot(&s.dot(v)) };
    let scale = s.iter().map(|x| x.abs()).fold(0.0_f64, f64::max).max(1e-300);

    let q_const = quad(&ones);
    let q_lin = quad(&g_lin);
    let g_norm_sq = g_lin.dot(&g_lin).max(1e-300);
    assert!(
        q_const.abs() < 1e-9 * scale,
        "2nd-difference penalty must annihilate the constant vector: βᵀSβ = {q_const:.3e} \
         (penalty max entry {scale:.3e})"
    );
    assert!(
        q_lin.abs() < 1e-9 * scale * g_norm_sq,
        "2nd-difference penalty must annihilate the Greville-abscissa ramp: \
         βᵀSβ = {q_lin:.3e} (penalty max entry {scale:.3e}, ‖g‖² = {g_norm_sq:.3e})"
    );

    // And it must NOT be the zero matrix (it genuinely penalizes curvature):
    // some quadratic-in-g coefficient vector has strictly positive energy.
    let quad_coef = Array1::<f64>::from_iter(greville.iter().map(|&v| v * v));
    assert!(
        quad(&quad_coef) > 1e-6 * scale,
        "2nd-difference penalty must penalize a quadratic-in-g coefficient vector \
         (otherwise it is degenerate); got βᵀSβ = {:.3e}",
        quad(&quad_coef)
    );
}

/// A genuinely collapsed divided-difference span (two coincident Greville
/// abscissae) must be REJECTED, not silently normalized into a singular penalty.
#[test]
fn difference_penalty_rejects_collapsed_span() {
    // Degree-1 (linear) basis: the order-1 Greville abscissae are the interior
    // knots, so a duplicated interior knot makes one order-1 span exactly zero
    // while the others are O(1). Clamped linear knots on [0,1] with a repeated
    // interior knot at 0.5 give greville = [0, 0.5, 0.5, 1.0] → a zero span.
    let degree = 1usize;
    let knots = Array1::from_vec(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
    let n_basis = knots.len() - degree - 1;
    let greville = compute_greville_abscissae(&knots, degree).expect("greville");
    let result = create_difference_penalty_matrix(n_basis, 1, Some(greville.view()));
    assert!(
        result.is_err(),
        "a collapsed (zero) divided-difference span must be rejected; got Ok with \
         greville = {greville:?}"
    );
}
