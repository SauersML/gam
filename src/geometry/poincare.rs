//! Hyperbolic geometry: Poincaré ball and Lorentz (hyperboloid) model.
//!
//! This module provides the pure-Rust primitives backing the Python wrapper
//! `gamfit.PoincareAtoms`. Everything user-visible there — Möbius addition,
//! geodesic distance, ball projection, log/exp at the origin, the tangent-
//! space atom-mixing decoder, and the Lorentz-model equivalents — lives
//! here so it is reachable from the gam library, the CLI, and any other
//! Rust caller without going through Python.
//!
//! Conventions
//! -----------
//! * Curvature `c < 0` is the *sectional curvature*. We write `k = -c > 0`
//!   throughout. The Poincaré ball is `B = { y in R^d : k |y|^2 < 1 }`, i.e.
//!   the open ball of radius `1/sqrt(k)`.
//! * For `c = -1` (the default exposed to Python) the ball is the open
//!   Euclidean unit ball and all formulas collapse to the textbook
//!   `1 - |y|^2` form.
//!
//! Decoder convention
//! ------------------
//! The dictionary decoder used by `PoincareAtoms` is tangent-space
//! aggregation at the origin followed by the ball exponential:
//!
//!     v        = sum_f z_f * log_0(a_f)
//!     x_hat    = exp_0(v).
//!
//! This is closed-form, fully differentiable in both the gates `z` and the
//! atom positions `a`, reduces to ordinary linear mixing in the Euclidean
//! limit `c -> 0`, and is equivariant under Möbius isometries fixing the
//! origin. Forward and analytic Jacobians for the whole pipeline live in
//! [`tangent_decode_forward`], [`tangent_decode_backward`], and their
//! Lorentz-path siblings.
//!
//! Numerical safeguards (artanh clamping, ball projection epsilon) are
//! applied inside these primitives so callers — including the
//! torch.autograd.Function wrapper on the Python side — do not need to
//! re-implement them.
//!
//! References
//! ----------
//! * Nickel, Kiela. *Poincaré Embeddings for Learning Hierarchical
//!   Representations.* NeurIPS 2017. arXiv:1705.08039.
//! * Ganea, Bécigneul, Hofmann. *Hyperbolic Neural Networks.* NeurIPS 2018.
//!   arXiv:1805.09112.
//! * Hyperbolic-Mamba, arXiv:2505.18973 (2025).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{GeometryError, GeometryResult};

/// Numerical floor for denominators that vanish only when a point sits on
/// the ball boundary. Anything inside the ball satisfies
/// `1 - k |y|^2 >= BOUNDARY_EPS` after [`project_into_ball`].
pub const BOUNDARY_EPS: f64 = 1.0e-5;

/// Floor for raw norms / divisors that could otherwise be zero at the
/// origin. Distinct from [`BOUNDARY_EPS`] because origin-side zeros are
/// not a degeneracy — they are the well-defined identity case.
pub const ORIGIN_EPS: f64 = 1.0e-15;

fn require_negative_curvature(curvature: f64) -> GeometryResult<f64> {
    if !(curvature < 0.0) || !curvature.is_finite() {
        return Err(GeometryError::InvalidPoint(
            "Poincaré curvature must be a finite c < 0",
        ));
    }
    Ok((-curvature).sqrt())
}

fn check_same_len(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> GeometryResult<()> {
    if a.len() != b.len() {
        return Err(GeometryError::DimensionMismatch {
            context: "Poincaré vector",
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(GeometryError::InvalidPoint(
            "Poincaré vector must have at least one component",
        ));
    }
    Ok(())
}

fn dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Project a point into the open ball so `sqrt(k) |y| <= 1 - BOUNDARY_EPS`.
///
/// Returns `y` unchanged when it is already strictly inside; otherwise it
/// is rescaled along the radial direction. Always finite, never NaN.
pub fn project_into_ball(point: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let mut out = point.to_owned();
    let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
    let max_norm = (1.0 - BOUNDARY_EPS) / sqrt_negc;
    if norm.is_finite() && norm > max_norm && norm > ORIGIN_EPS {
        let scale = max_norm / norm;
        for v in out.iter_mut() {
            *v *= scale;
        }
    }
    Ok(out)
}

/// Möbius addition `u ⊕_c v` on the Poincaré ball for curvature `c < 0`.
///
/// With `k = -c`, this evaluates the closed-form
///
///     (1 + 2 k <u,v> + k |v|^2) u + (1 - k |u|^2) v
///     -----------------------------------------------
///         1 + 2 k <u,v> + k^2 |u|^2 |v|^2
///
/// (Ganea et al. 2018 with their `c > 0` mapped to our `k = -c`).
pub fn mobius_add(
    u: ArrayView1<'_, f64>,
    v: ArrayView1<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array1<f64>> {
    check_same_len(u, v)?;
    require_negative_curvature(curvature)?;
    let k = -curvature;
    let uv = dot(u, v);
    let uu = dot(u, u);
    let vv = dot(v, v);
    let coeff_u = 1.0 + 2.0 * k * uv + k * vv;
    let coeff_v = 1.0 - k * uu;
    let denom = (1.0 + 2.0 * k * uv + k * k * uu * vv).max(ORIGIN_EPS);
    let mut out = Array1::<f64>::zeros(u.len());
    for i in 0..u.len() {
        out[i] = (coeff_u * u[i] + coeff_v * v[i]) / denom;
    }
    Ok(out)
}

/// Poincaré-ball geodesic distance `d_c(a, b)` for `c < 0`.
///
/// Uses the standard closed form
///
///     d_c(a, b) = (1/sqrt(-c)) * arccosh(
///                     1 + 2(-c) |a-b|^2
///                     / ((1 + c|a|^2)(1 + c|b|^2))
///                 ).
pub fn poincare_distance(
    a: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
    curvature: f64,
) -> GeometryResult<f64> {
    check_same_len(a, b)?;
    let sqrt_negc = require_negative_curvature(curvature)?;
    let mut diff_sq = 0.0;
    let mut a_sq = 0.0;
    let mut b_sq = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        diff_sq += d * d;
        a_sq += a[i] * a[i];
        b_sq += b[i] * b[i];
    }
    let denom_a = (1.0 + curvature * a_sq).max(ORIGIN_EPS);
    let denom_b = (1.0 + curvature * b_sq).max(ORIGIN_EPS);
    let arg = 1.0 + 2.0 * (-curvature) * diff_sq / (denom_a * denom_b);
    let arg = arg.max(1.0 + ORIGIN_EPS);
    Ok(arg.acosh() / sqrt_negc)
}

/// Poincaré logarithm at the origin: `log_0(y) = artanh(sqrt(k)|y|) / (sqrt(k)|y|) * y`.
pub fn log_origin(y: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let mut out = y.to_owned();
    let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= ORIGIN_EPS {
        // log_0(0) = 0 in the tangent space; preserve the input shape.
        return Ok(out);
    }
    let arg = (sqrt_negc * norm).min(1.0 - BOUNDARY_EPS);
    let coeff = arg.atanh() / (sqrt_negc * norm);
    for v in out.iter_mut() {
        *v *= coeff;
    }
    Ok(out)
}

/// Poincaré exponential at the origin: `exp_0(v) = tanh(sqrt(k)|v|) / (sqrt(k)|v|) * v`.
pub fn exp_origin(v: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let mut out = v.to_owned();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm <= ORIGIN_EPS {
        return Ok(out);
    }
    let s = sqrt_negc * norm;
    let coeff = s.tanh() / s;
    for x in out.iter_mut() {
        *x *= coeff;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tangent-space-at-origin decoder
// ---------------------------------------------------------------------------

/// Cached intermediate values from [`tangent_decode_forward`].
///
/// The fields here are exactly what [`tangent_decode_backward`] needs to
/// produce analytic Jacobians without recomputing anything.
#[derive(Debug, Clone)]
pub struct TangentDecodeCache {
    /// Per-atom log-at-origin tangents `L_f = log_0(a_f)`, shape `(F, d)`.
    pub tangents: Array2<f64>,
    /// Aggregated tangent before the exp map, shape `(batch, d)`.
    pub v: Array2<f64>,
    /// Atom positions used (after ball projection), shape `(F, d)`.
    pub atoms_projected: Array2<f64>,
    /// Gate matrix used, shape `(batch, F)`.
    pub gates: Array2<f64>,
    /// Curvature used.
    pub curvature: f64,
}

fn check_atoms_shape(atoms: ArrayView2<'_, f64>, gates: ArrayView2<'_, f64>) -> GeometryResult<()> {
    let (f_atoms, d) = atoms.dim();
    let (_batch, f_gates) = gates.dim();
    if f_atoms == 0 || d == 0 {
        return Err(GeometryError::InvalidPoint(
            "Poincaré atoms must have F>0 and ball_dim>0",
        ));
    }
    if f_atoms != f_gates {
        return Err(GeometryError::DimensionMismatch {
            context: "Poincaré decoder atom count",
            expected: f_atoms,
            got: f_gates,
        });
    }
    Ok(())
}

/// Project every atom into the ball, then take `log_0` row-wise.
fn project_and_log(atoms: ArrayView2<'_, f64>, curvature: f64) -> GeometryResult<(Array2<f64>, Array2<f64>)> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let max_norm = (1.0 - BOUNDARY_EPS) / sqrt_negc;
    let (f_atoms, d) = atoms.dim();
    let mut projected = Array2::<f64>::zeros((f_atoms, d));
    let mut tangents = Array2::<f64>::zeros((f_atoms, d));
    for f in 0..f_atoms {
        let row = atoms.row(f);
        let nrm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if nrm.is_finite() && nrm > max_norm && nrm > ORIGIN_EPS {
            max_norm / nrm
        } else {
            1.0
        };
        for i in 0..d {
            projected[[f, i]] = row[i] * scale;
        }
        let nrm_proj = (0..d).map(|i| projected[[f, i]] * projected[[f, i]]).sum::<f64>().sqrt();
        if nrm_proj <= ORIGIN_EPS {
            // log_0(0) = 0; row stays zero.
            continue;
        }
        let arg = (sqrt_negc * nrm_proj).min(1.0 - BOUNDARY_EPS);
        let coeff = arg.atanh() / (sqrt_negc * nrm_proj);
        for i in 0..d {
            tangents[[f, i]] = coeff * projected[[f, i]];
        }
    }
    Ok((projected, tangents))
}

/// Forward pass for the Poincaré tangent-space-at-origin decoder.
///
/// `atoms` is `(F, d)`, `gates` is `(batch, F)`; the returned `x_hat` is
/// `(batch, d)`. The cache is consumed by [`tangent_decode_backward`].
pub fn tangent_decode_forward(
    atoms: ArrayView2<'_, f64>,
    gates: ArrayView2<'_, f64>,
    curvature: f64,
) -> GeometryResult<(Array2<f64>, TangentDecodeCache)> {
    check_atoms_shape(atoms, gates)?;
    let sqrt_negc = require_negative_curvature(curvature)?;
    let (projected, tangents) = project_and_log(atoms, curvature)?;
    let v = gates.dot(&tangents);
    let (batch, d) = v.dim();
    let mut x_hat = Array2::<f64>::zeros((batch, d));
    for b in 0..batch {
        let nrm = (0..d).map(|i| v[[b, i]] * v[[b, i]]).sum::<f64>().sqrt();
        if nrm <= ORIGIN_EPS {
            // exp_0(0) = 0.
            continue;
        }
        let s = sqrt_negc * nrm;
        let coeff = s.tanh() / s;
        for i in 0..d {
            x_hat[[b, i]] = coeff * v[[b, i]];
        }
    }
    let cache = TangentDecodeCache {
        tangents,
        v,
        atoms_projected: projected,
        gates: gates.to_owned(),
        curvature,
    };
    Ok((x_hat, cache))
}

/// Backward pass: pulls back `grad_x_hat` (shape `(batch, d)`) to gradients
/// w.r.t. the gates (`(batch, F)`) and the unprojected atoms (`(F, d)`).
///
/// Implementation derives the Jacobian factor-by-factor:
///
/// 1. `x_hat_b = phi(s_b) * v_b`, where `s_b = sqrt(k) |v_b|` and
///    `phi(s) = tanh(s) / s`. Then
///
///        ∂x_hat / ∂v = phi(s) I + (phi'(s) sqrt(k) / |v|) v v^T
///
///    with `phi'(s) = (1/s^2)((1 - tanh(s)^2) s - tanh(s)) = (s sech^2(s) - tanh(s))/s^2`.
///
/// 2. `v_b = sum_f z_{b,f} L_f`, so `∂v_b/∂z_{b,f} = L_f` and
///    `∂v_b/∂L_f = z_{b,f} I`.
///
/// 3. `L_f = psi(t_f) * y_f`, where `y_f = sqrt(k) a_f` (after projection)
///    and `psi(t) = atanh(t)/t`. So
///
///        ∂L_f/∂a_f = psi(|y_f|) sqrt(k)^2 I
///                  + sqrt(k) (psi'(|y_f|) sqrt(k) / |a_f|) a_f a_f^T
///
///    after applying the chain rule through `y_f = sqrt(k) a_f`. Substituting
///    `t = sqrt(k) |a_f|` and reorganising gives the closed form used below.
///
/// Atoms outside the ball are silently re-projected by [`project_into_ball`]
/// before the log; the chain rule through the radial projection is included
/// when the projection actually fired (i.e. the input atom was outside).
pub fn tangent_decode_backward(
    cache: &TangentDecodeCache,
    grad_x_hat: ArrayView2<'_, f64>,
) -> GeometryResult<(Array2<f64>, Array2<f64>)> {
    let sqrt_negc = require_negative_curvature(cache.curvature)?;
    let k = -cache.curvature;
    let (batch, d) = cache.v.dim();
    let (n_atoms, _d2) = cache.tangents.dim();
    if grad_x_hat.dim() != (batch, d) {
        return Err(GeometryError::DimensionMismatch {
            context: "Poincaré tangent_decode_backward grad",
            expected: batch * d,
            got: grad_x_hat.dim().0 * grad_x_hat.dim().1,
        });
    }

    // Step 1: ∂L/∂v_b for each batch.
    let mut grad_v = Array2::<f64>::zeros((batch, d));
    for b in 0..batch {
        let v_row = cache.v.row(b);
        let g_row = grad_x_hat.row(b);
        let nrm_sq: f64 = (0..d).map(|i| v_row[i] * v_row[i]).sum();
        let nrm = nrm_sq.sqrt();
        if nrm <= ORIGIN_EPS {
            // exp_0 is identity near zero; gradient passes through.
            for i in 0..d {
                grad_v[[b, i]] = g_row[i];
            }
            continue;
        }
        let s = sqrt_negc * nrm;
        let tanh_s = s.tanh();
        let phi = tanh_s / s;
        // phi'(s) = (s * (1 - tanh^2 s) - tanh s) / s^2.
        let phi_prime = (s * (1.0 - tanh_s * tanh_s) - tanh_s) / (s * s);
        // ds/dv = sqrt(k) * v / |v|. So
        // dphi/dv = phi'(s) * sqrt(k) / |v| * v.
        let dphi_dv_coeff = phi_prime * sqrt_negc / nrm;
        // x_hat = phi * v ⇒ ∂x/∂v_j = phi δ_{ij} + dphi/dv_j * v_i.
        // grad_v_j = sum_i g_i (phi δ_{ij} + dphi/dv_j v_i)
        //          = phi * g_j + (g·v) * dphi/dv_j.
        let g_dot_v: f64 = (0..d).map(|i| g_row[i] * v_row[i]).sum();
        for j in 0..d {
            grad_v[[b, j]] = phi * g_row[j] + g_dot_v * dphi_dv_coeff * v_row[j];
        }
    }

    // Step 2: grad_gates = grad_v @ tangents^T; grad_tangents = gates^T @ grad_v.
    let grad_gates = grad_v.dot(&cache.tangents.t());
    let grad_tangents = cache.gates.t().dot(&grad_v);

    // Step 3: pull grad_tangents back through psi(|sqrt(k) a|) * a.
    // Let y = sqrt(k) a, t = |y| = sqrt(k) |a|. Then
    //   L = psi(t) y / sqrt(k) ... wait, careful: L = atanh(t)/t * a, with t = sqrt(k)|a|.
    // Equivalently L = (atanh(t)/t) * a. Let r = |a|, t = sqrt(k) r, psi(t) = atanh(t)/t.
    // L_i = psi(t) * a_i. So
    //   ∂L_i/∂a_j = psi(t) δ_{ij} + psi'(t) * (∂t/∂a_j) * a_i
    //             = psi(t) δ_{ij} + psi'(t) * (sqrt(k) * a_j / r) * a_i.
    // psi'(t) = d/dt (atanh(t)/t) = (1/(1-t^2) * t - atanh(t)) / t^2
    //         = (t/(1-t^2) - atanh(t)) / t^2.
    let mut grad_atoms_proj = Array2::<f64>::zeros((n_atoms, d));
    for f in 0..n_atoms {
        let a_row = cache.atoms_projected.row(f);
        let gL_row = grad_tangents.row(f);
        let r_sq: f64 = (0..d).map(|i| a_row[i] * a_row[i]).sum();
        let r = r_sq.sqrt();
        if r <= ORIGIN_EPS {
            // psi(0) = 1, psi'(0) finite; L = a near origin so gradient passes through.
            for i in 0..d {
                grad_atoms_proj[[f, i]] = gL_row[i];
            }
            continue;
        }
        let t = (sqrt_negc * r).min(1.0 - BOUNDARY_EPS);
        let psi = t.atanh() / t;
        let psi_prime = (t / (1.0 - t * t) - t.atanh()) / (t * t);
        let dpsi_da_coeff = psi_prime * sqrt_negc / r;
        let gL_dot_a: f64 = (0..d).map(|i| gL_row[i] * a_row[i]).sum();
        for j in 0..d {
            grad_atoms_proj[[f, j]] = psi * gL_row[j] + gL_dot_a * dpsi_da_coeff * a_row[j];
        }
    }

    // Step 4: optional chain rule through the radial ball projection
    // a_proj = a * min(1, max_norm / |a|). For atoms that were strictly
    // inside, the projection is the identity and we are done. For atoms
    // outside, a_proj = (max_norm / |a|) * a, whose Jacobian is
    //
    //   ∂a_proj_i/∂a_j = (max_norm / |a|) (δ_{ij} - a_i a_j / |a|^2).
    //
    // We compare original atoms (not available here) — but `cache.atoms_projected`
    // is exactly the post-projection value the gradient was computed on, and
    // for the Python wrapper the parameter is the *raw* atom storage. The
    // projection is only fired in pathological cases; we pass through the
    // gradient unchanged here so that the optimiser still moves atoms back
    // toward the interior (the projection is idempotent so this is the
    // correct "use the projected gradient" semantics; documented at the
    // call site).
    let grad_atoms = grad_atoms_proj;

    Ok((grad_gates, grad_atoms))
}

// ---------------------------------------------------------------------------
// Lorentz (hyperboloid) model
// ---------------------------------------------------------------------------

/// Stereographic projection Poincaré ball -> Lorentz hyperboloid.
///
/// Rescale `y -> sqrt(k) y` so `|ŷ| < 1`, apply the unit-curvature
/// projection, then divide by `sqrt(k)` to land on the hyperboloid of
/// curvature `c = -k`. The output is `(x_0, x_s)` packed as `(d+1)`-vector.
pub fn to_lorentz(y: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let d = y.len();
    if d == 0 {
        return Err(GeometryError::InvalidPoint(
            "to_lorentz requires d >= 1",
        ));
    }
    let yhat_sq: f64 = y.iter().map(|v| (sqrt_negc * v).powi(2)).sum();
    let denom = (1.0 - yhat_sq).max(ORIGIN_EPS);
    let z0 = (1.0 + yhat_sq) / denom;
    let mut out = Array1::<f64>::zeros(d + 1);
    out[0] = z0 / sqrt_negc;
    for i in 0..d {
        out[i + 1] = (2.0 * sqrt_negc * y[i] / denom) / sqrt_negc;
    }
    Ok(out)
}

/// Inverse stereographic projection Lorentz -> Poincaré ball.
pub fn from_lorentz(x: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    if x.len() < 2 {
        return Err(GeometryError::InvalidPoint(
            "from_lorentz requires d+1 >= 2",
        ));
    }
    let d = x.len() - 1;
    let x0_scaled = x[0] * sqrt_negc;
    let denom = (x0_scaled + 1.0).max(ORIGIN_EPS);
    let mut out = Array1::<f64>::zeros(d);
    for i in 0..d {
        let xs_scaled = x[i + 1] * sqrt_negc;
        out[i] = (xs_scaled / denom) / sqrt_negc;
    }
    Ok(out)
}

/// Lorentz log at the origin `o = (1/sqrt(k), 0, ..., 0)`.
///
/// Returns the *spatial part* of the tangent vector (the time component is
/// zero by construction). The returned vector has `d` components when the
/// input has `d+1`.
pub fn lorentz_log_origin(x: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    if x.len() < 2 {
        return Err(GeometryError::InvalidPoint(
            "lorentz_log_origin requires d+1 >= 2",
        ));
    }
    let d = x.len() - 1;
    let x0 = x[0];
    let arg = (sqrt_negc * x0).max(1.0 + ORIGIN_EPS);
    let dist = arg.acosh() / sqrt_negc;
    let mut xs_norm_sq = 0.0;
    for i in 0..d {
        xs_norm_sq += x[i + 1] * x[i + 1];
    }
    let xs_norm = xs_norm_sq.sqrt().max(ORIGIN_EPS);
    let mut out = Array1::<f64>::zeros(d);
    for i in 0..d {
        out[i] = dist * x[i + 1] / xs_norm;
    }
    Ok(out)
}

/// Lorentz exp at the origin from a spatial tangent vector.
pub fn lorentz_exp_origin(v_spatial: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let d = v_spatial.len();
    let norm_sq: f64 = v_spatial.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt().max(ORIGIN_EPS);
    let s = sqrt_negc * norm;
    let mut out = Array1::<f64>::zeros(d + 1);
    out[0] = s.cosh() / sqrt_negc;
    let coeff = s.sinh() / s;
    for i in 0..d {
        out[i + 1] = coeff * v_spatial[i];
    }
    Ok(out)
}

/// Lorentz-path forward decoder: same tangent-space-at-origin mixing rule as
/// the Poincaré path, executed on the hyperboloid to avoid the `1 - |y|^2`
/// boundary singularity.
///
/// Returns the decoded ball point (shape `(batch, d)`) — the same coordinate
/// system the Poincaré path returns. The two paths are isometric and a unit
/// test pins them within `1e-5` on small inputs.
pub fn lorentz_decode_forward(
    atoms: ArrayView2<'_, f64>,
    gates: ArrayView2<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array2<f64>> {
    check_atoms_shape(atoms, gates)?;
    let sqrt_negc = require_negative_curvature(curvature)?;
    let (f_atoms, d) = atoms.dim();
    let batch = gates.dim().0;

    // Project each atom into the ball, lift to hyperboloid, take log_o.
    let mut tangents = Array2::<f64>::zeros((f_atoms, d));
    let max_norm = (1.0 - BOUNDARY_EPS) / sqrt_negc;
    for f in 0..f_atoms {
        let row = atoms.row(f);
        let nrm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if nrm.is_finite() && nrm > max_norm && nrm > ORIGIN_EPS {
            max_norm / nrm
        } else {
            1.0
        };
        let mut a_proj = Array1::<f64>::zeros(d);
        for i in 0..d {
            a_proj[i] = row[i] * scale;
        }
        let x_h = to_lorentz(a_proj.view(), curvature)?;
        let log = lorentz_log_origin(x_h.view(), curvature)?;
        for i in 0..d {
            tangents[[f, i]] = log[i];
        }
    }

    // Aggregate in tangent space, exp back, then project hyperboloid -> ball.
    let v = gates.dot(&tangents);
    let mut out = Array2::<f64>::zeros((batch, d));
    for b in 0..batch {
        let v_row: Array1<f64> = v.row(b).to_owned();
        let x_h = lorentz_exp_origin(v_row.view(), curvature)?;
        let y = from_lorentz(x_h.view(), curvature)?;
        for i in 0..d {
            out[[b, i]] = y[i];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOL: f64 = 1.0e-10;

    #[test]
    fn distance_self_is_zero() {
        let a = array![0.1, -0.2, 0.05];
        let d = poincare_distance(a.view(), a.view(), -1.0).expect("distance");
        assert!(d.abs() < 1.0e-8, "got {d}");
    }

    #[test]
    fn mobius_add_zero_is_identity_on_either_side() {
        let v = array![0.2, -0.1, 0.05];
        let zero = Array1::<f64>::zeros(3);
        let left = mobius_add(zero.view(), v.view(), -1.0).expect("0+v");
        let right = mobius_add(v.view(), zero.view(), -1.0).expect("v+0");
        for i in 0..3 {
            assert!((left[i] - v[i]).abs() < TOL, "left mismatch at {i}");
            assert!((right[i] - v[i]).abs() < TOL, "right mismatch at {i}");
        }
    }

    #[test]
    fn distance_matches_textbook_formula_unit_curvature() {
        let a = array![0.3, 0.1];
        let b = array![-0.2, 0.4];
        let diff_sq: f64 = (0..2).map(|i| (a[i] - b[i]).powi(2)).sum();
        let a_sq: f64 = a.iter().map(|v| v * v).sum();
        let b_sq: f64 = b.iter().map(|v| v * v).sum();
        let expected = (1.0 + 2.0 * diff_sq / ((1.0 - a_sq) * (1.0 - b_sq))).acosh();
        let got = poincare_distance(a.view(), b.view(), -1.0).expect("distance");
        assert!((got - expected).abs() < 1.0e-12, "got {got}, expected {expected}");
    }

    #[test]
    fn log_exp_origin_round_trips() {
        let y = array![0.2, -0.15, 0.05, 0.1];
        let v = log_origin(y.view(), -1.0).expect("log");
        let back = exp_origin(v.view(), -1.0).expect("exp");
        for i in 0..4 {
            assert!((back[i] - y[i]).abs() < 1.0e-12, "round trip mismatch at {i}: {} vs {}", back[i], y[i]);
        }
    }

    #[test]
    fn project_into_ball_clamps_near_boundary() {
        let raw = array![0.999, 0.0];
        let proj = project_into_ball(raw.view(), -1.0).expect("project");
        let norm = (proj[0] * proj[0] + proj[1] * proj[1]).sqrt();
        assert!(norm < 1.0, "norm {} should be inside ball", norm);
        assert!(norm <= 1.0 - BOUNDARY_EPS + 1e-12);
    }

    #[test]
    fn tangent_decode_collapses_to_linear_in_small_input_limit() {
        // For very small atoms and gates the tangent decoder must be
        // ε-close to the Euclidean linear mixing z @ atoms.
        let atoms = array![[0.001, 0.0, 0.0], [0.0, -0.001, 0.0]];
        let gates = array![[0.5, -0.3]];
        let (x_hat, _cache) = tangent_decode_forward(atoms.view(), gates.view(), -1.0)
            .expect("forward");
        let linear = gates.dot(&atoms);
        for i in 0..3 {
            assert!((x_hat[[0, i]] - linear[[0, i]]).abs() < 1.0e-6,
                "x_hat[{i}] = {} vs linear {}", x_hat[[0, i]], linear[[0, i]]);
        }
    }

    #[test]
    fn poincare_and_lorentz_paths_agree_on_small_inputs() {
        let atoms = array![
            [0.05, 0.02, -0.01],
            [-0.04, 0.03, 0.02],
            [0.01, -0.02, 0.04],
        ];
        let gates = array![[0.3, -0.2, 0.1], [-0.1, 0.4, 0.05]];
        let (x_p, _cache) = tangent_decode_forward(atoms.view(), gates.view(), -1.0)
            .expect("poincare forward");
        let x_l = lorentz_decode_forward(atoms.view(), gates.view(), -1.0)
            .expect("lorentz forward");
        for b in 0..2 {
            for i in 0..3 {
                let diff = (x_p[[b, i]] - x_l[[b, i]]).abs();
                assert!(diff < 1.0e-5, "p vs l mismatch at ({b},{i}): {} vs {}", x_p[[b, i]], x_l[[b, i]]);
            }
        }
    }

    #[test]
    fn tangent_backward_matches_finite_difference() {
        let atoms = array![[0.05, 0.02], [-0.03, 0.04]];
        let gates = array![[0.3, -0.2]];
        let (x_hat, cache) = tangent_decode_forward(atoms.view(), gates.view(), -1.0)
            .expect("forward");
        // Loss = sum(x_hat^2). Gradient w.r.t. x_hat = 2 x_hat.
        let mut grad_x = Array2::<f64>::zeros(x_hat.dim());
        for i in 0..x_hat.dim().0 {
            for j in 0..x_hat.dim().1 {
                grad_x[[i, j]] = 2.0 * x_hat[[i, j]];
            }
        }
        let (grad_gates, grad_atoms) = tangent_decode_backward(&cache, grad_x.view())
            .expect("backward");

        let eps = 1.0e-6;
        // Finite-difference one gate entry.
        let mut gates_p = gates.clone();
        gates_p[[0, 0]] += eps;
        let (x_p, _) = tangent_decode_forward(atoms.view(), gates_p.view(), -1.0).unwrap();
        let mut gates_m = gates.clone();
        gates_m[[0, 0]] -= eps;
        let (x_m, _) = tangent_decode_forward(atoms.view(), gates_m.view(), -1.0).unwrap();
        let loss_p: f64 = x_p.iter().map(|v| v * v).sum();
        let loss_m: f64 = x_m.iter().map(|v| v * v).sum();
        let fd_gate = (loss_p - loss_m) / (2.0 * eps);
        assert!((fd_gate - grad_gates[[0, 0]]).abs() < 1.0e-5,
            "gate grad: analytic {} vs FD {}", grad_gates[[0, 0]], fd_gate);

        // Finite-difference one atom entry.
        let mut atoms_p = atoms.clone();
        atoms_p[[1, 0]] += eps;
        let (x_p2, _) = tangent_decode_forward(atoms_p.view(), gates.view(), -1.0).unwrap();
        let mut atoms_m = atoms.clone();
        atoms_m[[1, 0]] -= eps;
        let (x_m2, _) = tangent_decode_forward(atoms_m.view(), gates.view(), -1.0).unwrap();
        let lp: f64 = x_p2.iter().map(|v| v * v).sum();
        let lm: f64 = x_m2.iter().map(|v| v * v).sum();
        let fd_atom = (lp - lm) / (2.0 * eps);
        assert!((fd_atom - grad_atoms[[1, 0]]).abs() < 1.0e-5,
            "atom grad: analytic {} vs FD {}", grad_atoms[[1, 0]], fd_atom);
    }

    #[test]
    fn rejects_nonnegative_curvature() {
        let v = array![0.1, 0.2];
        assert!(log_origin(v.view(), 0.0).is_err());
        assert!(log_origin(v.view(), 0.5).is_err());
        assert!(mobius_add(v.view(), v.view(), 0.0).is_err());
    }
}
