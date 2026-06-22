//! Property + discrimination gate for the hyperbolic-SAE roughness operator
//! `poincare::conformal_dirichlet_penalty` (the conformal-reweighted Dirichlet
//! Gram that turns a flat tangent patch into a hyperbolic patch).
//!
//! The operator is the single source of truth for hyperbolic atom smoothness.
//! These tests pin the three properties that make it a *correct* and *useful*
//! roughness penalty, with truth we construct ourselves (no reference tool):
//!
//! 1. **Symmetric PSD** — a roughness Gram must be a valid quadratic form: it
//!    is symmetric and `βᵀSβ ≥ 0` for every coefficient vector.
//! 2. **Conformal-invariance limit (`d = 2`)** — 2-D Dirichlet energy is
//!    conformally invariant, so at `d = 2` the hyperbolic weight `λ^{d−2} = 1`
//!    and the hyperbolic Gram must equal the *flat* Dirichlet Gram exactly. A
//!    penalty that claimed a `d = 2` hyperbolic correction would be wrong.
//! 3. **Boundary up-weighting (`d = 1`)** — for `d = 1` the weight is
//!    `λ(p)^{−1} = (1 + c‖p‖²)/2`, which *decreases* toward the boundary
//!    (`c < 0`). The hyperbolic penalty therefore charges *less* per unit
//!    tangent-coordinate wiggle near the boundary than the flat penalty does —
//!    i.e. a unit step in the tangent coordinate covers more hyperbolic
//!    distance away from the boundary and less near it. We assert the exact
//!    monotone relationship between the per-row weights and the boundary
//!    radius, so the geometry (not just the shape) is pinned.

use gam::geometry::poincare;
use gam::terms::basis::{
    DuchonNullspaceOrder, duchon_polynomial_first_derivative_nd, monomial_exponents,
};
use ndarray::{Array1, Array2, Array3};

const CURV: f64 = -1.0;

/// Deterministic uniform in [0, 1) keyed purely by index (no clock / RNG).
fn idx_uniform(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Monomial design jacobian `∂Φ_k/∂t_a` for total degree ≤ `max_degree`.
fn monomial_jet(coords: &Array2<f64>, max_degree: usize) -> Array3<f64> {
    let order = match max_degree {
        0 => DuchonNullspaceOrder::Zero,
        1 => DuchonNullspaceOrder::Linear,
        k => DuchonNullspaceOrder::Degree(k),
    };
    duchon_polynomial_first_derivative_nd(coords.view(), order)
}

#[test]
fn conformal_dirichlet_penalty_is_symmetric_and_psd() {
    let max_degree = 3;
    let n = 64;
    // 1-D tangent coords spread across the ball radius.
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| -1.6 + 3.2 * idx_uniform(i as u64));
    let jet = monomial_jet(&coords, max_degree);
    let gram = poincare::conformal_dirichlet_penalty(coords.view(), jet.view(), CURV)
        .expect("penalty builds");
    let m = gram.ncols();

    // Symmetric to round-off.
    for i in 0..m {
        for j in 0..m {
            assert!(
                (gram[[i, j]] - gram[[j, i]]).abs() <= 1e-12 * (1.0 + gram[[i, j]].abs()),
                "Gram not symmetric at ({i},{j}): {} vs {}",
                gram[[i, j]],
                gram[[j, i]]
            );
        }
    }
    // PSD: a battery of random coefficient vectors must give βᵀSβ ≥ 0.
    for trial in 0..40u64 {
        let beta = Array1::from_shape_fn(m, |k| idx_uniform(trial * 131 + k as u64) - 0.5);
        let sb = gram.dot(&beta);
        let q = beta.dot(&sb);
        assert!(
            q >= -1e-10,
            "conformal Dirichlet Gram is not PSD: βᵀSβ = {q:.3e} < 0 on trial {trial}"
        );
    }
}

/// Flat Euclidean Dirichlet Gram `Σ_n Φ'(t_n)ᵀ Φ'(t_n)` (weight ≡ 1) — the
/// `c → 0⁻` / `d = 2` limit the hyperbolic penalty must reproduce when the
/// conformal weight is unity.
fn flat_dirichlet_gram(coords: &Array2<f64>, jet: &Array3<f64>) -> Array2<f64> {
    let n = coords.nrows();
    let d = coords.ncols();
    let m = jet.shape()[1];
    let mut gram = Array2::<f64>::zeros((m, m));
    for row in 0..n {
        for axis in 0..d {
            for i in 0..m {
                let gi = jet[[row, i, axis]];
                if gi == 0.0 {
                    continue;
                }
                for j in 0..m {
                    gram[[i, j]] += gi * jet[[row, j, axis]];
                }
            }
        }
    }
    gram
}

#[test]
fn d2_hyperbolic_penalty_equals_flat_dirichlet_conformal_invariance() {
    // d = 2: λ^{d−2} = λ^0 = 1 everywhere, so the hyperbolic Gram must equal
    // the flat Dirichlet Gram to round-off. This is the discrete witness of the
    // conformal invariance of 2-D Dirichlet energy — a hyperbolic penalty that
    // deviated here would be geometrically wrong.
    let max_degree = 2;
    let n = 50;
    let coords = Array2::from_shape_fn((n, 2), |(i, a)| {
        // Keep points strictly inside the ball after exp₀: tangent radius < 1.
        0.7 * (idx_uniform((i * 2 + a) as u64) - 0.5)
    });
    let jet = monomial_jet(&coords, max_degree);
    let hyp = poincare::conformal_dirichlet_penalty(coords.view(), jet.view(), CURV)
        .expect("penalty builds");
    let flat = flat_dirichlet_gram(&coords, &jet);
    let m = hyp.ncols();
    let mut max_abs = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            max_abs = max_abs.max((hyp[[i, j]] - flat[[i, j]]).abs());
            scale = scale.max(flat[[i, j]].abs());
        }
    }
    assert!(
        max_abs <= 1e-10 * (1.0 + scale),
        "d=2 hyperbolic penalty must equal the flat Dirichlet Gram (conformal \
         invariance), but max|hyp−flat| = {max_abs:.3e} (scale {scale:.3e})"
    );
}

#[test]
fn d1_weight_is_exact_inverse_conformal_factor_monotone_in_radius() {
    // d = 1: the per-row weight the operator applies is λ(p)^{d−2} = λ^{−1}.
    // We reconstruct the implied weights from the rank-1 single-monomial design
    // Φ = [t] (degree 1, so Φ' ≡ 1 and S = Σ_n w_n is a scalar accumulation),
    // and assert each row's weight equals the closed-form (1 + c‖p‖²)/2 and is
    // monotone *decreasing* in the boundary radius ‖p‖ (c < 0). This pins the
    // geometry: wiggle near the boundary is charged less per tangent unit
    // because a tangent step there spans more hyperbolic length.
    let radii_in = [0.05_f64, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8];
    let mut prev_weight = f64::INFINITY;
    let mut prev_radius = -1.0_f64;
    for &t in &radii_in {
        // Single-row, single linear monomial: Φ'(t) = d/dt (t) = 1.
        let coords = Array2::from_shape_fn((1, 1), |_| t);
        // Build only the linear monomial column so Φ' is exactly 1.
        let exponents = monomial_exponents(1, 1); // [[0],[1]] -> [1, t]
        assert_eq!(exponents.len(), 2);
        let full_jet = monomial_jet(&coords, 1);
        // Linear column index is 1 (constant is 0); slice it to a 1×1×1 jet.
        let mut lin_jet = Array3::<f64>::zeros((1, 1, 1));
        lin_jet[[0, 0, 0]] = full_jet[[0, 1, 0]];
        assert!(
            (lin_jet[[0, 0, 0]] - 1.0).abs() < 1e-12,
            "linear monomial derivative must be 1"
        );

        let gram =
            poincare::conformal_dirichlet_penalty(coords.view(), lin_jet.view(), CURV).unwrap();
        let weight = gram[[0, 0]]; // = w_n · 1² = λ(p)^{-1}

        // Closed-form check: p = exp₀(t), λ = 2/(1 + c‖p‖²), weight = λ^{-1}.
        let p = poincare::exp_origin(coords.row(0), CURV).unwrap();
        let p_sq: f64 = p.iter().map(|x| x * x).sum();
        let lambda = 2.0 / (1.0 + CURV * p_sq);
        let expected = 1.0 / lambda;
        assert!(
            (weight - expected).abs() <= 1e-10 * (1.0 + expected.abs()),
            "d=1 weight at t={t}: got {weight:.6e}, expected λ^-1 = {expected:.6e}"
        );

        let radius = p_sq.sqrt();
        // Monotone decreasing in boundary radius (strictly, for c < 0).
        if prev_radius >= 0.0 {
            assert!(
                weight < prev_weight + 1e-12 && radius > prev_radius,
                "weight must decrease as the ball radius grows: t={t} radius={radius:.4} \
                 weight={weight:.6e} (prev radius={prev_radius:.4} weight={prev_weight:.6e})"
            );
        }
        prev_weight = weight;
        prev_radius = radius;
    }
}
