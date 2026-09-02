//! Property + discrimination gate for the hyperbolic-SAE roughness operator
//! `poincare::conformal_dirichlet_penalty` (the conformal-reweighted Dirichlet
//! Gram that turns a flat tangent patch into a hyperbolic patch).
//!
//! The operator is the single source of truth for hyperbolic atom smoothness.
//! These tests pin the properties that make it a *correct* and *useful*
//! roughness penalty, with truth we construct ourselves (no reference tool):
//!
//! 1. **Symmetric PSD** — a roughness Gram must be a valid quadratic form: it
//!    is symmetric and `βᵀSβ ≥ 0` for every coefficient vector.
//! 2. **Closed-form pullback (`d = 2`, `c = −1`)** — the penalised field is a
//!    function of the *tangent* coordinate `t`, and `p = exp₀(t)` is not a
//!    conformal chart, so the energy must integrate against the pullback metric
//!    `h(t) = J(t)ᵀ λ² J(t)`, which in polar tangent coordinates is
//!    `4 dr² + sinh²(2r) dθ²`. The assembled Gram must equal
//!    `Σ_n Φ'(t_n)ᵀ (√det h · h⁻¹) Φ'(t_n)` built from that closed form — and it
//!    must **not** equal the flat tangent-coordinate Dirichlet Gram (the old
//!    implementation, which conflated the ball and tangent charts).
//! 3. **`d = 1` half-speed tangent coordinate** — a 1-D hyperbolic manifold is
//!    intrinsically flat, but the tangent coordinate runs at half arc-length
//!    (`geodesic dist = 2‖t‖`), so the per-row weight the operator applies is the
//!    exact constant `1/2` at every radius, independent of the boundary
//!    proximity. We pin that constant.

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

/// Flat *tangent-coordinate* Dirichlet Gram `Σ_n Φ'(t_n)ᵀ Φ'(t_n)` (weight ≡ 1).
/// This is the OLD (incorrect) implementation, kept here only as the negative
/// control: for `d = 2` the correct pullback Gram must *differ* from it.
fn flat_tangent_dirichlet_gram(coords: &Array2<f64>, jet: &Array3<f64>) -> Array2<f64> {
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

/// Reference Gram for `d = 2`, `c = −1`, assembled from the closed-form pullback
/// metric `4 dr² + sinh²(2r) dθ²` (`r = ‖t‖`). In Cartesian tangent coordinates
/// this line element is `h = 4·t̂t̂ᵀ + (sinh(2r)²/r²)·(I − t̂t̂ᵀ)` (the radial
/// direction gets the `dr²` weight, the perpendicular direction the `dθ²` weight
/// through `r dθ = t̂⊥·dt`). The metric weight is `G = √det h · h⁻¹`, and since
/// `h = a·t̂t̂ᵀ + b·(I − t̂t̂ᵀ)` with `a = 4`, `b = sinh(2r)²/r²`,
/// `G = √(b/a)·t̂t̂ᵀ + √(a/b)·(I − t̂t̂ᵀ)`. This route is independent of the
/// implementation's eigen-decomposition (it starts from the polar line element).
fn closed_form_pullback_gram_d2(coords: &Array2<f64>, jet: &Array3<f64>) -> Array2<f64> {
    let n = coords.nrows();
    let m = jet.shape()[1];
    let mut gram = Array2::<f64>::zeros((m, m));
    for row in 0..n {
        let tx = coords[[row, 0]];
        let ty = coords[[row, 1]];
        let r = (tx * tx + ty * ty).sqrt();
        // G = g_par·t̂t̂ᵀ + g_perp·(I − t̂t̂ᵀ), with the closed-form eigenvalues.
        let (g_par, g_perp, that) = if r <= 1e-15 {
            // r → 0: a = 4, b = sinh(2r)²/r² → 4, so G → I (isotropic).
            (1.0, 1.0, [0.0, 0.0])
        } else {
            let a = 4.0_f64;
            let b = (2.0 * r).sinh().powi(2) / (r * r);
            ((b / a).sqrt(), (a / b).sqrt(), [tx / r, ty / r])
        };
        // 2×2 metric weight G.
        let mut gmat = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let pij = that[i] * that[j];
                let iso = if i == j { 1.0 } else { 0.0 };
                gmat[i][j] = g_par * pij + g_perp * (iso - pij);
            }
        }
        // S += Φ'ᵀ G Φ' with Φ'[k, axis] = jet[row, k, axis].
        for k in 0..m {
            // Gk[axis] = Σ_b G[axis, b] Φ'[k, b].
            let gk = [
                gmat[0][0] * jet[[row, k, 0]] + gmat[0][1] * jet[[row, k, 1]],
                gmat[1][0] * jet[[row, k, 0]] + gmat[1][1] * jet[[row, k, 1]],
            ];
            for l in 0..m {
                gram[[k, l]] += jet[[row, l, 0]] * gk[0] + jet[[row, l, 1]] * gk[1];
            }
        }
    }
    gram
}

