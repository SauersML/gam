//! Hyperbolic geometry: Poincaré ball and Lorentz (hyperboloid) model.
//!
//! Inside this disc there is *room to spare*: volume grows exponentially with
//! radius, so a tree's worth of children fits near the rim with distances
//! barely distorted. That is exactly why hyperbolic embeddings flatter
//! hierarchies — the boundary is infinitely far away even though it looks a
//! finger's width off. Mind the rim: every point with `k |y|² → 1` is racing
//! off to infinity, which is why we project strictly *inside* the ball and
//! never let a coordinate touch the edge.
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

/// Largest radial exp-map argument `s = sqrt(k)|v|` worth evaluating
/// hyperbolic functions at. Above this, `tanh(s)` is already
/// `1 - BOUNDARY_EPS` in f64 (and `cosh`/`sinh` would eventually overflow to
/// `inf` near `s ≈ 710`), so the Lorentz lift caps `s` here to stay finite
/// while landing on the same clamped boundary as the Poincaré path. Derived
/// from [`BOUNDARY_EPS`] — no magic literal: `atanh(1 - BOUNDARY_EPS) ≈ 6.1`.
const EXP_SATURATION_CAP: f64 = {
    // `atanh` is not const-evaluable; the cap is fixed by BOUNDARY_EPS as
    // `atanh(1 - BOUNDARY_EPS)` and is pinned by
    // `exp_saturation_cap_matches_boundary_eps`.
    6.1030338227611125
};

/// Poincaré exp radial coefficient `tanh(s)/s` with the open-ball boundary
/// clamp baked in. Because `sqrt(k)|exp_0(v)| = tanh(s)`, clamping `tanh(s)`
/// to `1 - BOUNDARY_EPS` makes the output norm exactly `max_norm`, i.e.
/// strictly interior and consistent with [`project_into_ball`]. Callers must
/// guard `s > ORIGIN_EPS` before calling (norms at/under the origin floor
/// short-circuit to the identity), so no divide-by-zero is introduced.
fn exp_coeff(s: f64) -> f64 {
    (s.tanh().min(1.0 - BOUNDARY_EPS)) / s
}

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

/// Reject points that are not in the open Poincaré ball `{x : √k·|x| < 1}`.
///
/// The closed forms below (`distance`, `mobius_add`, `log_origin`) are only
/// defined on the open ball; for a point outside it the denominators
/// `1 + c|x|²` go non-positive and the `.max(ORIGIN_EPS)` floors would turn an
/// off-manifold input into a finite but meaningless number. Callers that
/// genuinely need to accept arbitrary coordinates must
/// [`project_into_ball`] first (as the decoder/Lorentz paths do). Tangent maps
/// such as [`exp_origin`] take an unbounded tangent vector, not a ball point,
/// and so must NOT use this check.
fn require_in_ball(point: ArrayView1<'_, f64>, sqrt_negc: f64) -> GeometryResult<()> {
    let mut norm_sq = 0.0_f64;
    for v in point.iter() {
        if !v.is_finite() {
            return Err(GeometryError::InvalidPoint(
                "Poincaré point contains NaN or infinity",
            ));
        }
        norm_sq += v * v;
    }
    if sqrt_negc * norm_sq.sqrt() >= 1.0 {
        return Err(GeometryError::InvalidPoint(
            "Poincaré point lies outside the open ball (√k·|x| ≥ 1)",
        ));
    }
    Ok(())
}

/// Project a point into the open ball so `sqrt(k) |y| <= 1 - BOUNDARY_EPS`.
///
/// Returns `y` unchanged when it is already strictly inside; otherwise it is
/// rescaled along the radial direction. The output is always finite and never
/// NaN — which requires rejecting a non-finite *input* up front, since a NaN or
/// infinite coordinate cannot be radially rescaled into the ball (a `NaN` norm
/// fails the `is_finite` guard and would otherwise pass straight through).
pub fn project_into_ball(
    point: ArrayView1<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    if !point.iter().all(|v| v.is_finite()) {
        return Err(GeometryError::InvalidPoint(
            "Poincaré projection input contains NaN or infinity",
        ));
    }
    let mut out = point.to_owned();
    let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
    let max_norm = (1.0 - BOUNDARY_EPS) / sqrt_negc;
    if norm > max_norm && norm > ORIGIN_EPS {
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
    let sqrt_negc = require_negative_curvature(curvature)?;
    require_in_ball(u, sqrt_negc)?;
    require_in_ball(v, sqrt_negc)?;
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
    require_in_ball(a, sqrt_negc)?;
    require_in_ball(b, sqrt_negc)?;
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
    // Geodesic distance via the cosh half-angle identity
    //   arccosh(1 + 2δ) = 2·arcsinh(√δ),
    //   δ = (-c)|a-b|² / ((1 + c|a|²)(1 + c|b|²)).
    // The textbook `arccosh(1 + 2δ)` form cancels catastrophically for nearby
    // points: `1 + 2δ` rounds away everything in δ below ~1e-16 relative to 1,
    // and since arccosh(1+2δ) ≈ 2√δ at the branch point, a relative argument
    // perturbation ~eps/(2δ) becomes a ~eps/δ ~ eps/sep² error in the distance
    // (≈4% at sep=1e-8). Forming √δ straight from |a-b|² and applying arcsinh
    // (which does not cancel for small argument) keeps full relative accuracy.
    // δ ≥ 0 always (−c>0, |a-b|²≥0, denoms>0); arcsinh(0)=0 still gives exact
    // zero for identical points, so the self-distance contract is preserved
    // without any near-1 clamp.
    let delta = (-curvature) * diff_sq / (denom_a * denom_b);
    Ok(2.0 * delta.max(0.0).sqrt().asinh() / sqrt_negc)
}

/// Poincaré logarithm at the origin: `log_0(y) = artanh(sqrt(k)|y|) / (sqrt(k)|y|) * y`.
pub fn log_origin(y: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    require_in_ball(y, sqrt_negc)?;
    let mut out = y.to_owned();
    let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= ORIGIN_EPS {
        // log_0(0) = 0 in the tangent space; preserve the input shape.
        return Ok(out);
    }
    // y is validated in-ball, so sqrt(k)·|y| < 1; the clamp only guards the
    // last-ulp approach to the boundary, never an out-of-domain artanh.
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
    let coeff = exp_coeff(s);
    for x in out.iter_mut() {
        *x *= coeff;
    }
    Ok(out)
}

/// Conformal factor `lambda_p = 2 / (1 + curvature * |p|^2)` of the Poincaré
/// ball at base point `p` (curvature `< 0`, so the denominator is `1 - |c||p|^2`).
///
/// The metric is the closed-form diagonal `g_p = lambda_p^2 I`; this is the
/// single source of truth for that factor across all front-ends.
pub fn conformal_factor(p: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<f64> {
    require_negative_curvature(curvature)?;
    let sq = p.iter().map(|x| x * x).sum::<f64>();
    Ok(2.0 / (1.0 + curvature * sq))
}

/// Conformal-reweighted Dirichlet roughness Gram of a smooth latent field over
/// the Poincaré ball — the hyperbolic analogue of the flat
/// `∫ Φ'(t)ᵀ Φ'(t) dt` patch penalty.
///
/// The atom's latent coordinate `t_n ∈ ℝ^d` is read as a *tangent vector at the
/// ball origin*; its ball point is `p_n = exp₀(t_n)` (the wrapped / tangent
/// parameterisation of Nagano et al. 2019, Mathieu et al. 2019). For a decoded
/// field `f(t) = β·Φ(t)` the hyperbolic Dirichlet energy is
///
/// ```text
/// E_g[f] = ∫ gᵃᵇ ∂_a f ∂_b f dμ_g
///        = ∫ λ(p)^{d−2} ‖∇_t f‖² dt           (g_ab = λ² δ_ab, dμ_g = λ^d dt)
/// ```
///
/// with `λ(p) = 2 / (1 + c‖p‖²)` the conformal factor ([`conformal_factor`],
/// `c < 0`). Discretising the integral against the empirical row density gives
/// the coefficient-space Gram
///
/// ```text
/// S = Σ_n w_n Φ'(t_n)ᵀ Φ'(t_n),   w_n = λ(p_n)^{d−2},
/// ```
/// which this function assembles from the basis first-derivative jet
/// `basis_jacobian[n, k, a] = ∂Φ_k/∂t_a (t_n)` and the latent coordinates
/// `coords[n, a] = t_{n,a}`. The flat patch is the `c → 0⁻` / `d = 2` limit
/// where `w_n ≡ 1` (2-D Dirichlet energy is conformally invariant), so the
/// hyperbolic penalty differs from the Euclidean one exactly when the data has
/// the boundary-concentrated (exponential-volume) structure hyperbolic geometry
/// is for. The returned Gram is symmetric PSD by construction.
///
/// `curvature` must be strictly negative. Returns the `(M, M)` Gram, `M` the
/// number of basis columns.
pub fn conformal_dirichlet_penalty(
    coords: ArrayView2<'_, f64>,
    basis_jacobian: ndarray::ArrayView3<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array2<f64>> {
    require_negative_curvature(curvature)?;
    let n = coords.nrows();
    let d = coords.ncols();
    let jet_shape = basis_jacobian.shape();
    if jet_shape[0] != n {
        return Err(GeometryError::DimensionMismatch {
            context: "conformal_dirichlet_penalty: basis_jacobian row count vs coords rows",
            expected: n,
            got: jet_shape[0],
        });
    }
    if jet_shape[2] != d {
        return Err(GeometryError::DimensionMismatch {
            context: "conformal_dirichlet_penalty: basis_jacobian latent-axis count vs coords cols",
            expected: d,
            got: jet_shape[2],
        });
    }
    let m = jet_shape[1];
    let mut gram = Array2::<f64>::zeros((m, m));
    if n == 0 || m == 0 {
        return Ok(gram);
    }
    let exponent = d as f64 - 2.0;
    let mut grad = vec![0.0_f64; m];
    for row in 0..n {
        // Ball point p_n = exp₀(t_n) and its conformal weight w_n = λ(p_n)^{d−2}.
        let p = exp_origin(coords.row(row), curvature)?;
        let lambda = conformal_factor(p.view(), curvature)?;
        let w = lambda.powf(exponent);
        if !(w.is_finite() && w > 0.0) {
            continue;
        }
        // Accumulate w_n · ∇f outer products one latent axis at a time:
        // S += Σ_a w_n · (∂Φ/∂t_a)(∂Φ/∂t_a)ᵀ.
        for axis in 0..d {
            for k in 0..m {
                grad[k] = basis_jacobian[[row, k, axis]];
            }
            for i in 0..m {
                let gi = grad[i];
                if gi == 0.0 {
                    continue;
                }
                let scaled = w * gi;
                for j in 0..m {
                    gram[[i, j]] += scaled * grad[j];
                }
            }
        }
    }
    Ok(gram)
}

/// Poincaré exponential map at an arbitrary base point `p`:
/// `exp_p(v) = p ⊕ exp_0(0.5 * lambda_p * v)`.
///
/// This is the single source of truth for the gyrovector composition of the
/// conformal factor with the origin-anchored map; front-ends (CLI, library,
/// `gamfit`) must call it rather than recomposing `lambda_p` themselves.
pub fn exp_map(
    p: ArrayView1<'_, f64>,
    v: ArrayView1<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array1<f64>> {
    check_same_len(p, v)?;
    let lam = conformal_factor(p, curvature)?;
    let scaled = v.mapv(|x| 0.5 * lam * x);
    let exp0 = exp_origin(scaled.view(), curvature)?;
    mobius_add(p, exp0.view(), curvature)
}

/// Poincaré logarithm map at an arbitrary base point `p`:
/// `log_p(q) = (2 / lambda_p) * log_0((-p) ⊕ q)`.
///
/// Inverse of [`exp_map`] and the single source of truth for the base-point
/// logarithm; front-ends must call it rather than recomposing `lambda_p`.
pub fn log_map(
    p: ArrayView1<'_, f64>,
    q: ArrayView1<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array1<f64>> {
    check_same_len(p, q)?;
    let lam = conformal_factor(p, curvature)?;
    let neg_p = p.mapv(|x| -x);
    let shifted = mobius_add(neg_p.view(), q, curvature)?;
    let log0 = log_origin(shifted.view(), curvature)?;
    Ok(log0.mapv(|x| (2.0 / lam) * x))
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
    /// Per-atom radial ball-projection factor `s_f = min(1, max_norm/|a_raw|)`,
    /// shape `(F,)`. `s_f == 1` for atoms that were already inside the ball;
    /// `s_f < 1` marks the atoms that were clamped and so carry a non-identity
    /// projection Jacobian in [`tangent_decode_backward`] step 4.
    pub proj_scale: Array1<f64>,
    /// Gate matrix used, shape `(batch, F)`.
    pub gates: Array2<f64>,
    /// Curvature used.
    pub curvature: f64,
}

fn check_atoms_shape(atoms: ArrayView2<'_, f64>, gates: ArrayView2<'_, f64>) -> GeometryResult<()> {
    let (f_atoms, d) = atoms.dim();
    let f_gates = gates.dim().1;
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
///
/// Returns `(projected, tangents, proj_scale)` where `proj_scale[f]` is the
/// radial ball-projection factor `s_f = min(1, max_norm / |a_raw_f|)` that was
/// applied to atom `f`. `s_f == 1` means the raw atom was already inside the
/// ball (projection is the identity); `s_f < 1` means it was clamped and the
/// projection Jacobian is non-trivial — [`tangent_decode_backward`] needs
/// `s_f` to chain the gradient back to the *raw* atom storage.
fn project_and_log(
    atoms: ArrayView2<'_, f64>,
    curvature: f64,
) -> GeometryResult<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let max_norm = (1.0 - BOUNDARY_EPS) / sqrt_negc;
    let (f_atoms, d) = atoms.dim();
    let mut projected = Array2::<f64>::zeros((f_atoms, d));
    let mut tangents = Array2::<f64>::zeros((f_atoms, d));
    let mut proj_scale = Array1::<f64>::ones(f_atoms);
    for f in 0..f_atoms {
        let row = atoms.row(f);
        let nrm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = if nrm.is_finite() && nrm > max_norm && nrm > ORIGIN_EPS {
            max_norm / nrm
        } else {
            1.0
        };
        proj_scale[f] = scale;
        for i in 0..d {
            projected[[f, i]] = row[i] * scale;
        }
        let nrm_proj = (0..d)
            .map(|i| projected[[f, i]] * projected[[f, i]])
            .sum::<f64>()
            .sqrt();
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
    Ok((projected, tangents, proj_scale))
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
    let (projected, tangents, proj_scale) = project_and_log(atoms, curvature)?;
    // V = gates · tangents (batch×F · F×d): the dominant decode GEMM over the
    // whole observation batch. Row-tile gates across ALL GPUs (each device runs
    // one cuBLAS call over its batch-row tile with tangents broadcast); single
    // device / small batch falls back to the auto-dispatch shim.
    let v = crate::geometry::manifold::fast_ab_rows_multi_gpu(gates, tangents.view());
    let (batch, d) = v.dim();
    let mut x_hat = Array2::<f64>::zeros((batch, d));
    for b in 0..batch {
        let nrm = (0..d).map(|i| v[[b, i]] * v[[b, i]]).sum::<f64>().sqrt();
        if nrm <= ORIGIN_EPS {
            // exp_0(0) = 0.
            continue;
        }
        let s = sqrt_negc * nrm;
        let coeff = exp_coeff(s);
        for i in 0..d {
            x_hat[[b, i]] = coeff * v[[b, i]];
        }
    }
    let cache = TangentDecodeCache {
        tangents,
        v,
        atoms_projected: projected,
        proj_scale,
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
    let (batch, d) = cache.v.dim();
    let n_atoms = cache.tangents.dim().0;
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

    // Step 2: grad_gates = grad_v @ tangentsᵀ (batch×d · d×F); grad_tangents =
    // gatesᵀ @ grad_v (F×batch · batch×d). Both are full-batch GEMMs,
    // GPU-dispatched via fast_abt / fast_atb.
    use crate::linalg::faer_ndarray::{fast_abt, fast_atb};
    let grad_gates = fast_abt(&grad_v, &cache.tangents);
    let grad_tangents = fast_atb(&cache.gates, &grad_v);

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
        let g_l_row = grad_tangents.row(f);
        let r_sq: f64 = (0..d).map(|i| a_row[i] * a_row[i]).sum();
        let r = r_sq.sqrt();
        if r <= ORIGIN_EPS {
            // psi(0) = 1, psi'(0) finite; L = a near origin so gradient passes through.
            for i in 0..d {
                grad_atoms_proj[[f, i]] = g_l_row[i];
            }
            continue;
        }
        let t = (sqrt_negc * r).min(1.0 - BOUNDARY_EPS);
        let psi = t.atanh() / t;
        let psi_prime = (t / (1.0 - t * t) - t.atanh()) / (t * t);
        let dpsi_da_coeff = psi_prime * sqrt_negc / r;
        let g_l_dot_a: f64 = (0..d).map(|i| g_l_row[i] * a_row[i]).sum();
        for j in 0..d {
            grad_atoms_proj[[f, j]] = psi * g_l_row[j] + g_l_dot_a * dpsi_da_coeff * a_row[j];
        }
    }

    // Step 4: chain rule through the radial ball projection
    // a_proj = a_raw * s_f, with s_f = min(1, max_norm / |a_raw|).
    //
    // For atoms that were strictly inside (s_f == 1) the projection is the
    // identity and grad_a_raw == grad_a_proj. For atoms that were clamped
    // (s_f < 1), a_proj = (max_norm / |a_raw|) a_raw, whose Jacobian is
    //
    //   ∂a_proj_i/∂a_raw_j = s_f (δ_{ij} - â_i â_j),   â = a_raw / |a_raw|.
    //
    // Since a_proj is parallel to a_raw, â = a_proj / |a_proj|, and the
    // transpose-applied VJP is grad_a_raw = s_f (I - â âᵀ) grad_a_proj. The
    // radial component of grad_a_proj is annihilated (moving the raw atom
    // radially outward does not change the clamped projection) — exactly the
    // missing factor the previous pass-through dropped.
    let mut grad_atoms = grad_atoms_proj;
    for f in 0..n_atoms {
        let s_f = cache.proj_scale[f];
        if s_f >= 1.0 {
            continue;
        }
        let a_row = cache.atoms_projected.row(f);
        let a_norm_sq: f64 = (0..d).map(|i| a_row[i] * a_row[i]).sum();
        if a_norm_sq <= ORIGIN_EPS * ORIGIN_EPS {
            // Clamp toward the origin cannot fire with a zero-norm projection;
            // nothing radial to remove.
            for j in 0..d {
                grad_atoms[[f, j]] *= s_f;
            }
            continue;
        }
        // â âᵀ projector onto the radial direction.
        let g_row: Vec<f64> = (0..d).map(|j| grad_atoms[[f, j]]).collect();
        let g_dot_a: f64 = (0..d).map(|i| g_row[i] * a_row[i]).sum();
        let radial_coeff = g_dot_a / a_norm_sq;
        for j in 0..d {
            grad_atoms[[f, j]] = s_f * (g_row[j] - radial_coeff * a_row[j]);
        }
    }

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
///
/// The input is first run through [`project_into_ball`] so the boundary-
/// vanishing denominator `1 - |ŷ|^2` is bounded below by `~2·BOUNDARY_EPS`.
/// A point on (or outside) the ideal boundary would otherwise drive the
/// denominator to the `ORIGIN_EPS` floor (`1e-15`) and blow the output up by
/// `~1e15`; projecting first keeps the map well-conditioned and makes the
/// `from_lorentz ∘ to_lorentz` round-trip equal `project_into_ball(y)`.
pub fn to_lorentz(y: ArrayView1<'_, f64>, curvature: f64) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let d = y.len();
    if d == 0 {
        return Err(GeometryError::InvalidPoint("to_lorentz requires d >= 1"));
    }
    let y_proj = project_into_ball(y, curvature)?;
    let yhat_sq: f64 = y_proj.iter().map(|v| (sqrt_negc * v).powi(2)).sum();
    // After `project_into_ball`, `sqrt(k)|y| <= 1 - BOUNDARY_EPS`, so
    // `1 - yhat_sq >= 2·BOUNDARY_EPS - BOUNDARY_EPS^2 > 0`; the floor is a
    // defensive no-op held at `BOUNDARY_EPS` (never the `1e15` `ORIGIN_EPS`).
    let denom = (1.0 - yhat_sq).max(BOUNDARY_EPS);
    let z0 = (1.0 + yhat_sq) / denom;
    let mut out = Array1::<f64>::zeros(d + 1);
    out[0] = z0 / sqrt_negc;
    for i in 0..d {
        out[i + 1] = (2.0 * sqrt_negc * y_proj[i] / denom) / sqrt_negc;
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
    // Symmetric with `to_lorentz`, which projects its input into the ball:
    // enforce the open-ball invariant `sqrt(k)|out| <= 1 - BOUNDARY_EPS` here
    // too. `tanh(s/2)` saturates to exactly 1.0 for large hyperboloid points,
    // which would land `out` ON the ideal boundary; the projection nudges it
    // strictly interior so every `from_lorentz` consumer is boundary-safe.
    project_into_ball(out.view(), curvature)
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
    // Same exact-1.0 clamp as `poincare_distance`: at the hyperboloid
    // origin, `sqrt_negc * x0 == 1.0` and the log must be the zero tangent.
    let arg = (sqrt_negc * x0).max(1.0);
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
pub fn lorentz_exp_origin(
    v_spatial: ArrayView1<'_, f64>,
    curvature: f64,
) -> GeometryResult<Array1<f64>> {
    let sqrt_negc = require_negative_curvature(curvature)?;
    let d = v_spatial.len();
    let norm_sq: f64 = v_spatial.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt().max(ORIGIN_EPS);
    let s = sqrt_negc * norm;
    // Cap the argument fed to `cosh`/`sinh`. For `s` beyond ~710 these
    // overflow to `inf`, giving an `inf/(inf+1) = NaN` ball point downstream,
    // so `s_eval` must be bounded well below that. Choosing the right cap also
    // fixes where near-boundary embeddings saturate. By this module's identity
    // `y^{Lorentz}(v) = exp_0^{Poincare}(v / 2)`, the downstream `from_lorentz`
    // maps this hyperboloid point to ball radius
    // `sinh(s_eval)/(cosh(s_eval)+1) = tanh(s_eval / 2)` — the EFFECTIVE
    // Poincaré argument is HALF of `s_eval`. To land the decoded radius on the
    // same open-ball boundary `1 - BOUNDARY_EPS` that the Poincaré path
    // (`exp_coeff`, which clamps `tanh(s)`) produces, we need
    // `tanh(s_eval / 2) = 1 - BOUNDARY_EPS`, i.e. `s_eval / 2 =
    // EXP_SATURATION_CAP`, hence the cap is `2 * EXP_SATURATION_CAP`
    // (≈ 12.206 — still far below the ~710 cosh/sinh overflow threshold).
    // Capping at `EXP_SATURATION_CAP` instead would saturate at
    // `tanh(EXP_SATURATION_CAP / 2) ≈ 0.9955`, a factor of ~2 too early.
    // `from_lorentz` then projects strictly interior, so the decode never NaNs.
    let s_eval = s.min(2.0 * EXP_SATURATION_CAP);
    let mut out = Array1::<f64>::zeros(d + 1);
    out[0] = s_eval.cosh() / sqrt_negc;
    // Radial scaling stays in the original (uncapped) tangent direction: the
    // spatial part is `(sinh(s_eval)/s) * v`, i.e. the capped hyperbolic
    // magnitude distributed along `v`. `from_lorentz`'s ratio depends only on
    // `cosh`/`sinh` of `s_eval`, so the cap and projection together fix the
    // output norm at `max_norm` exactly as the Poincaré path does.
    let coeff = s_eval.sinh() / s;
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

    // Aggregate in tangent space (batch×F · F×d GEMM over the whole batch,
    // row-tiled across ALL GPUs), exp back, then project hyperboloid -> ball.
    let v = crate::geometry::manifold::fast_ab_rows_multi_gpu(gates, tangents.view());
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

/// Analytic backward for the Lorentz-path decoder.
///
/// Lorentz vs Poincaré identity
/// ----------------------------
/// The tangent-space-at-origin decoder is *intrinsically defined* on the
/// hyperbolic manifold: it only sees the manifold's log/exp at the origin,
/// not any particular chart. Working through the Lorentz-path composition
/// (Poincaré atom → stereographic lift to hyperboloid → Lorentz log_o →
/// linear mix → Lorentz exp_o → stereographic projection back to ball)
/// produces, after using
///
///     acosh((1+q)/(1-q)) = 2 artanh(sqrt(q))     (for q in [0,1)),
///     cosh(σ) + 1        = 2 cosh^2(σ/2),
///     sinh(σ)            = 2 sinh(σ/2) cosh(σ/2),
///
/// the algebraic identity
///
///     L_f^{Lorentz}      = 2 * log_0^{Poincare}(a_f),
///     y^{Lorentz}(v)     = exp_0^{Poincare}( v / 2 ).
///
/// The factors of 2 cancel inside the linear-mix step, so:
///
///     y_Lorentz(z; A) === y_Poincare(z; A),
///
/// not just isometrically — they are the *same function* of the inputs.
/// The two forward implementations are numerically distinct routes (Lorentz
/// has no `1 - |y|^2` denominator and so survives near-boundary atoms
/// better) but mathematically equal in exact arithmetic. Their Jacobians
/// are therefore also equal, and the analytic backward derived for the
/// Poincaré path is the exact backward for the Lorentz path.
///
/// This function exists to make that exactness *demonstrable* — it shares
/// the implementation of [`tangent_decode_backward`] but has its own unit
/// test that finite-differences the Lorentz forward (not the Poincaré
/// forward) and checks the same analytic Jacobian against it.
pub fn lorentz_decode_backward(
    cache: &TangentDecodeCache,
    grad_x_hat: ArrayView2<'_, f64>,
) -> GeometryResult<(Array2<f64>, Array2<f64>)> {
    tangent_decode_backward(cache, grad_x_hat)
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
        assert_eq!(d, 0.0, "self-distance must be exactly 0.0, got {d:e}");
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
        let a = array![0.3_f64, 0.1];
        let b = array![-0.2_f64, 0.4];
        let diff_sq: f64 = (0..2).map(|i| (a[i] - b[i]).powi(2)).sum();
        let a_sq: f64 = a.iter().map(|v| v * v).sum();
        let b_sq: f64 = b.iter().map(|v| v * v).sum();
        let expected = (1.0 + 2.0 * diff_sq / ((1.0 - a_sq) * (1.0 - b_sq))).acosh();
        let got = poincare_distance(a.view(), b.view(), -1.0).expect("distance");
        assert!(
            (got - expected).abs() < 1.0e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn log_exp_origin_round_trips() {
        let y = array![0.2, -0.15, 0.05, 0.1];
        let v = log_origin(y.view(), -1.0).expect("log");
        let back = exp_origin(v.view(), -1.0).expect("exp");
        for i in 0..4 {
            assert!(
                (back[i] - y[i]).abs() < 1.0e-12,
                "round trip mismatch at {i}: {} vs {}",
                back[i],
                y[i]
            );
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
    fn lorentz_round_trip_interior_point() {
        // For a point strictly inside the ball, `project_into_ball` is the
        // identity, so `from_lorentz(to_lorentz(y)) == y` exactly.
        let y = array![0.2, -0.15, 0.05];
        let z = to_lorentz(y.view(), -1.0).expect("to_lorentz");
        assert!(
            z.iter().all(|v| v.is_finite()),
            "lorentz image must be finite"
        );
        let back = from_lorentz(z.view(), -1.0).expect("from_lorentz");
        for i in 0..y.len() {
            assert!(
                (back[i] - y[i]).abs() < 1.0e-12,
                "round trip mismatch at {i}: {} vs {}",
                back[i],
                y[i]
            );
        }
    }

    #[test]
    fn lorentz_round_trip_near_boundary_equals_projection() {
        // A point on/outside the ideal boundary must not blow up: the
        // composition equals `project_into_ball(y)` (the boundary clamp),
        // and the intermediate hyperboloid point stays finite.
        for curvature in [-1.0, -0.25, -4.0] {
            let y = array![1.5, 0.0, 0.0];
            let z = to_lorentz(y.view(), curvature).expect("to_lorentz");
            assert!(
                z.iter().all(|v| v.is_finite()),
                "boundary point must yield a finite hyperboloid image, got {z:?}"
            );
            let back = from_lorentz(z.view(), curvature).expect("from_lorentz");
            let expected = project_into_ball(y.view(), curvature).expect("project");
            for i in 0..y.len() {
                assert!(
                    (back[i] - expected[i]).abs() < 1.0e-9,
                    "near-boundary round trip must equal projection at {i} \
                     (curvature {curvature}): {} vs {}",
                    back[i],
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn tangent_decode_collapses_to_linear_in_small_input_limit() {
        // For very small atoms and gates the tangent decoder must be
        // ε-close to the Euclidean linear mixing z @ atoms.
        let atoms = array![[0.001, 0.0, 0.0], [0.0, -0.001, 0.0]];
        let gates = array![[0.5, -0.3]];
        let (x_hat, cache) =
            tangent_decode_forward(atoms.view(), gates.view(), -1.0).expect("forward");
        // Touch the cache so the binding is observed by the type system
        // (Rust forbids underscore-prefixed lets in this crate).
        assert_eq!(cache.tangents.dim(), (2, 3));
        let linear = gates.dot(&atoms);
        for i in 0..3 {
            assert!(
                (x_hat[[0, i]] - linear[[0, i]]).abs() < 1.0e-6,
                "x_hat[{i}] = {} vs linear {}",
                x_hat[[0, i]],
                linear[[0, i]]
            );
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
        let (x_p, cache) =
            tangent_decode_forward(atoms.view(), gates.view(), -1.0).expect("poincare forward");
        assert_eq!(cache.tangents.dim(), (3, 3));
        let x_l =
            lorentz_decode_forward(atoms.view(), gates.view(), -1.0).expect("lorentz forward");
        for b in 0..2 {
            for i in 0..3 {
                let diff = (x_p[[b, i]] - x_l[[b, i]]).abs();
                assert!(
                    diff < 1.0e-5,
                    "p vs l mismatch at ({b},{i}): {} vs {}",
                    x_p[[b, i]],
                    x_l[[b, i]]
                );
            }
        }
    }

    #[test]
    fn tangent_backward_matches_finite_difference() {
        let atoms = array![[0.05, 0.02], [-0.03, 0.04]];
        let gates = array![[0.3, -0.2]];
        let (x_hat, cache) =
            tangent_decode_forward(atoms.view(), gates.view(), -1.0).expect("forward");
        // Loss = sum(x_hat^2). Gradient w.r.t. x_hat = 2 x_hat.
        let mut grad_x = Array2::<f64>::zeros(x_hat.dim());
        for i in 0..x_hat.dim().0 {
            for j in 0..x_hat.dim().1 {
                grad_x[[i, j]] = 2.0 * x_hat[[i, j]];
            }
        }
        let (grad_gates, grad_atoms) =
            tangent_decode_backward(&cache, grad_x.view()).expect("backward");

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
        assert!(
            (fd_gate - grad_gates[[0, 0]]).abs() < 1.0e-5,
            "gate grad: analytic {} vs FD {}",
            grad_gates[[0, 0]],
            fd_gate
        );

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
        assert!(
            (fd_atom - grad_atoms[[1, 0]]).abs() < 1.0e-5,
            "atom grad: analytic {} vs FD {}",
            grad_atoms[[1, 0]],
            fd_atom
        );
    }

    #[test]
    fn tangent_backward_includes_ball_projection_jacobian() {
        // Atom 0 starts strictly OUTSIDE the unit ball (|a| ≈ 2.24 > 1 for
        // curvature -1) so `project_into_ball` clamps it. The backward must
        // then chain grad through the radial projection a_proj = s·a_raw,
        // i.e. left-multiply by s·(I - â âᵀ). Finite-differencing the forward
        // w.r.t. the *raw* atom storage is the only correct guardrail: the
        // forward consumes raw atoms and re-projects internally, so the FD
        // gradient is exactly the projected gradient.
        let atoms = array![[2.0, 1.0], [-0.03, 0.04]];
        let gates = array![[0.7, -0.5], [0.2, 0.9]];
        let (x_hat, cache) =
            tangent_decode_forward(atoms.view(), gates.view(), -1.0).expect("forward");
        // Atom 0 must have actually been clamped (s_0 < 1); otherwise this
        // test would silently exercise the identity branch and prove nothing.
        assert!(
            cache.proj_scale[0] < 1.0,
            "atom 0 should have been clamped, got proj_scale {}",
            cache.proj_scale[0]
        );
        assert!(
            (cache.proj_scale[1] - 1.0).abs() < 1.0e-12,
            "atom 1 should be interior (proj_scale == 1), got {}",
            cache.proj_scale[1]
        );

        let mut grad_x = Array2::<f64>::zeros(x_hat.dim());
        for i in 0..x_hat.dim().0 {
            for j in 0..x_hat.dim().1 {
                grad_x[[i, j]] = 2.0 * x_hat[[i, j]];
            }
        }
        let (_grad_gates, grad_atoms) =
            tangent_decode_backward(&cache, grad_x.view()).expect("backward");

        let eps = 1.0e-6;
        // Finite-difference BOTH raw components of the clamped atom 0.
        for comp in 0..2usize {
            let mut atoms_p = atoms.clone();
            atoms_p[[0, comp]] += eps;
            let (x_p, _) = tangent_decode_forward(atoms_p.view(), gates.view(), -1.0).unwrap();
            let mut atoms_m = atoms.clone();
            atoms_m[[0, comp]] -= eps;
            let (x_m, _) = tangent_decode_forward(atoms_m.view(), gates.view(), -1.0).unwrap();
            let lp: f64 = x_p.iter().map(|v| v * v).sum();
            let lm: f64 = x_m.iter().map(|v| v * v).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - grad_atoms[[0, comp]]).abs() < 1.0e-5,
                "clamped-atom grad comp {comp}: analytic {} vs FD {}",
                grad_atoms[[0, comp]],
                fd
            );
        }

        // The radial component of the analytic gradient on the clamped atom
        // must be (numerically) annihilated: moving the raw atom further out
        // radially does not change its projection, so the projected gradient
        // is orthogonal to â. Verify directly.
        let a0 = cache.atoms_projected.row(0);
        let a0_norm = (a0[0] * a0[0] + a0[1] * a0[1]).sqrt();
        let radial = (grad_atoms[[0, 0]] * a0[0] + grad_atoms[[0, 1]] * a0[1]) / a0_norm;
        assert!(
            radial.abs() < 1.0e-8,
            "clamped-atom gradient should have ~zero radial component, got {radial}"
        );
    }

    #[test]
    fn lorentz_backward_matches_finite_difference_of_lorentz_forward() {
        // Same tolerance as the Poincaré FD test, but the FD probes the
        // Lorentz forward (`lorentz_decode_forward`) rather than the
        // Poincaré one — verifying that the analytic backward produced by
        // [`lorentz_decode_backward`] really is the Jacobian of the Lorentz
        // forward, not just of its Poincaré sibling.
        let atoms = array![[0.05, 0.02], [-0.03, 0.04]];
        let gates = array![[0.3, -0.2]];

        // Cache from the Poincaré forward: the two paths are algebraically
        // equal so it does not matter which one we differentiate analytically.
        let (x_hat_p, cache) = tangent_decode_forward(atoms.view(), gates.view(), -1.0)
            .expect("poincare forward (for cache)");
        let x_hat_l =
            lorentz_decode_forward(atoms.view(), gates.view(), -1.0).expect("lorentz forward");
        // Sanity: forward outputs of the two paths must agree to fp slack.
        for b in 0..x_hat_l.dim().0 {
            for i in 0..x_hat_l.dim().1 {
                assert!(
                    (x_hat_p[[b, i]] - x_hat_l[[b, i]]).abs() < 1.0e-10,
                    "lorentz/poincare forward mismatch at ({b},{i})"
                );
            }
        }

        // Loss = sum(x_hat^2), grad_x = 2 x_hat (evaluate on Lorentz output
        // since that is what we are differentiating against).
        let mut grad_x = Array2::<f64>::zeros(x_hat_l.dim());
        for i in 0..x_hat_l.dim().0 {
            for j in 0..x_hat_l.dim().1 {
                grad_x[[i, j]] = 2.0 * x_hat_l[[i, j]];
            }
        }
        let (grad_gates, grad_atoms) =
            lorentz_decode_backward(&cache, grad_x.view()).expect("lorentz backward");

        let eps = 1.0e-6;
        // FD against the *Lorentz* forward, one gate entry.
        let mut gates_p = gates.clone();
        gates_p[[0, 0]] += eps;
        let x_p = lorentz_decode_forward(atoms.view(), gates_p.view(), -1.0).unwrap();
        let mut gates_m = gates.clone();
        gates_m[[0, 0]] -= eps;
        let x_m = lorentz_decode_forward(atoms.view(), gates_m.view(), -1.0).unwrap();
        let lp: f64 = x_p.iter().map(|v| v * v).sum();
        let lm: f64 = x_m.iter().map(|v| v * v).sum();
        let fd_gate = (lp - lm) / (2.0 * eps);
        assert!(
            (fd_gate - grad_gates[[0, 0]]).abs() < 1.0e-5,
            "lorentz gate grad: analytic {} vs FD {}",
            grad_gates[[0, 0]],
            fd_gate
        );

        // FD against the *Lorentz* forward, one atom entry.
        let mut atoms_p = atoms.clone();
        atoms_p[[1, 0]] += eps;
        let x_p2 = lorentz_decode_forward(atoms_p.view(), gates.view(), -1.0).unwrap();
        let mut atoms_m = atoms.clone();
        atoms_m[[1, 0]] -= eps;
        let x_m2 = lorentz_decode_forward(atoms_m.view(), gates.view(), -1.0).unwrap();
        let lp2: f64 = x_p2.iter().map(|v| v * v).sum();
        let lm2: f64 = x_m2.iter().map(|v| v * v).sum();
        let fd_atom = (lp2 - lm2) / (2.0 * eps);
        assert!(
            (fd_atom - grad_atoms[[1, 0]]).abs() < 1.0e-5,
            "lorentz atom grad: analytic {} vs FD {}",
            grad_atoms[[1, 0]],
            fd_atom
        );
    }

    #[test]
    fn exp_saturation_cap_matches_boundary_eps() {
        // The named cap must equal `atanh(1 - BOUNDARY_EPS)` so the Lorentz
        // lift caps exactly where `tanh` first reads `1 - BOUNDARY_EPS` in f64.
        let expected = (1.0 - BOUNDARY_EPS).atanh();
        assert!(
            (EXP_SATURATION_CAP - expected).abs() < 1.0e-12,
            "cap {EXP_SATURATION_CAP} vs atanh(1-eps) {expected}"
        );
    }

    #[test]
    fn exp_origin_large_tangent_stays_strictly_interior() {
        // Regression for #354: a large tangent must not saturate ONTO the ball
        // boundary. `tanh(s)` rounds to exactly 1.0 for s ≳ 19, so the
        // unclamped map would land |x| == 1; the clamp keeps it interior.
        for curvature in [-1.0_f64, -0.25, -4.0] {
            let sqrt_negc = (-curvature).sqrt();
            let max_norm = (1.0 - BOUNDARY_EPS) / sqrt_negc;
            let v = array![1.0e3, -5.0e2, 2.0e2, 7.0e1];
            let x = exp_origin(v.view(), curvature).expect("exp");
            assert!(x.iter().all(|q| q.is_finite()), "exp must be finite: {x:?}");
            let norm = x.iter().map(|q| q * q).sum::<f64>().sqrt();
            assert!(
                sqrt_negc * norm <= 1.0 - BOUNDARY_EPS + 1.0e-12,
                "exp must stay strictly interior (curvature {curvature}): \
                 sqrt(k)|x| = {}",
                sqrt_negc * norm
            );
            assert!(
                (norm - max_norm).abs() < 1.0e-9,
                "saturated exp must sit at max_norm {max_norm}, got {norm}"
            );
            // log/distance must remain finite for the clamped point.
            let origin = Array1::<f64>::zeros(x.len());
            let dist = poincare_distance(origin.view(), x.view(), curvature).expect("dist");
            assert!(dist.is_finite(), "distance must be finite, got {dist}");
        }
    }

    #[test]
    fn lorentz_exp_origin_large_tangent_is_finite_and_interior() {
        // Companion regression for #354 on the Lorentz path: large tangents
        // must not overflow cosh/sinh to inf (which becomes NaN downstream),
        // and the round-tripped ball point must stay strictly interior.
        for curvature in [-1.0_f64, -0.25, -4.0] {
            let sqrt_negc = (-curvature).sqrt();
            // Spatial tangent with norm ~1e3 — far past both the tanh-1.0
            // saturation (s ≳ 19) and the cosh/sinh overflow (s ≳ 710) regimes.
            let v = array![1.0e3, -5.0e2, 2.0e2];
            let x_h = lorentz_exp_origin(v.view(), curvature).expect("lorentz exp");
            assert!(
                x_h.iter().all(|q| q.is_finite()),
                "lorentz hyperboloid point must be finite, got {x_h:?}"
            );
            let y = from_lorentz(x_h.view(), curvature).expect("from_lorentz");
            assert!(
                y.iter().all(|q| q.is_finite()),
                "ball point must be finite, got {y:?}"
            );
            let norm = y.iter().map(|q| q * q).sum::<f64>().sqrt();
            assert!(
                sqrt_negc * norm <= 1.0 - BOUNDARY_EPS + 1.0e-12,
                "lorentz decode must stay strictly interior (curvature \
                 {curvature}): sqrt(k)|y| = {}",
                sqrt_negc * norm
            );
        }
    }

    #[test]
    fn lorentz_exp_saturates_at_same_boundary_as_poincare() {
        // Regression for #1349. The Lorentz decode path used to cap its exp
        // argument at `EXP_SATURATION_CAP`, but `from_lorentz` maps the
        // hyperboloid point to ball radius `tanh(s_eval / 2)` (the module's
        // identity `y^{Lorentz}(v) = exp_0^{Poincare}(v / 2)`). So capping at
        // `EXP_SATURATION_CAP` saturated the radius at `tanh(cap / 2) ≈ 0.9955`
        // — a factor of ~2 too early — instead of `1 - BOUNDARY_EPS`. The cap is
        // now `2 * EXP_SATURATION_CAP` so both decoders land on the same
        // boundary.
        let curvature = -1.0_f64;
        let sqrt_negc = (-curvature).sqrt();
        // A fixed unit direction in R^4, scaled to several large norms. With
        // curvature -1 the radial exp argument is `s = norm`; the saturation
        // cap is `2 * EXP_SATURATION_CAP ≈ 12.206`, so every norm here sits
        // deep in the saturated (capped) regime where the radius must pin to
        // `1 - BOUNDARY_EPS`. The OLD cap of `EXP_SATURATION_CAP ≈ 6.103` would
        // instead saturate at `tanh(6.103 / 2) ≈ 0.9955`, failing this test.
        let raw = array![0.3_f64, -0.5, 0.7, 0.2];
        let raw_norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        for &target_norm in &[20.0_f64, 50.0, 100.0] {
            let v: Array1<f64> = raw.mapv(|x| x * target_norm / raw_norm);

            // (a) The Lorentz decode is the SAME function as the Poincaré exp at
            // half the tangent: `from_lorentz(lorentz_exp_origin(v))` must match
            // `exp_origin(v / 2)` to ~1e-9 (radius times sqrt(-c)). This holds
            // because both saturate identically once their (shared) effective
            // argument exceeds the cap.
            let x_h = lorentz_exp_origin(v.view(), curvature).expect("lorentz exp");
            let y_lorentz = from_lorentz(x_h.view(), curvature).expect("from_lorentz");
            let v_half: Array1<f64> = v.mapv(|x| x * 0.5);
            let y_poincare = exp_origin(v_half.view(), curvature).expect("exp");
            let r_lorentz = sqrt_negc * y_lorentz.iter().map(|q| q * q).sum::<f64>().sqrt();
            let r_poincare = sqrt_negc * y_poincare.iter().map(|q| q * q).sum::<f64>().sqrt();
            assert!(
                (r_lorentz - r_poincare).abs() < 1.0e-9,
                "lorentz vs poincare(v/2) radius mismatch at norm {target_norm}: \
                 {r_lorentz} vs {r_poincare}"
            );

            // (b) The saturated radius reaches `1 - BOUNDARY_EPS`, NOT ~0.9955.
            assert!(
                (r_lorentz - (1.0 - BOUNDARY_EPS)).abs() < 1.0e-9,
                "saturated lorentz radius must reach 1 - BOUNDARY_EPS \
                 ({}), got {r_lorentz} at norm {target_norm}",
                1.0 - BOUNDARY_EPS
            );
            assert!(
                r_lorentz > 0.999,
                "saturated radius {r_lorentz} must be near 1, not the old ~0.9955 \
                 (norm {target_norm})"
            );
        }
    }

    #[test]
    fn decode_large_gates_stay_strictly_interior_both_paths() {
        // End-to-end #354 repro analog: large gates drive a large aggregated
        // tangent. Both the Poincaré and Lorentz decoders must return points
        // strictly inside the ball (not on the boundary).
        let atoms = array![
            [0.3, 0.1, -0.2, 0.05, 0.0, 0.1],
            [-0.1, 0.2, 0.05, -0.15, 0.1, 0.0],
            [0.05, -0.05, 0.2, 0.1, -0.1, 0.15],
            [0.1, 0.1, -0.1, 0.2, 0.05, -0.05],
        ];
        let gates = Array2::<f64>::from_elem((3, 4), 1.0e3);
        let curvature = -1.0;

        let (x_p, _) =
            tangent_decode_forward(atoms.view(), gates.view(), curvature).expect("poincare");
        let x_l = lorentz_decode_forward(atoms.view(), gates.view(), curvature).expect("lorentz");
        for x in [&x_p, &x_l] {
            for b in 0..x.dim().0 {
                let norm = (0..x.dim().1)
                    .map(|i| x[[b, i]] * x[[b, i]])
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    norm.is_finite() && norm <= 1.0 - BOUNDARY_EPS + 1.0e-12,
                    "decode row {b} must be strictly interior, got norm {norm}"
                );
            }
        }
    }

    #[test]
    fn exp_origin_is_radial_isometry_in_riemannian_metric() {
        // The origin exp map is a radial isometry w.r.t. the RIEMANNIAN metric,
        // not the bare Euclidean tangent norm. With the conformal factor
        // λ_x = 2/(1 - k|x|²), the tangent norm at the origin is λ_0·|v| = 2|v|
        // (λ_0 = 2 since |0| = 0), and the geodesic distance to exp_0(v) equals
        // that Riemannian norm:
        //
        //     |y| = tanh(s), δ = k|y|²/(1-k|y|²) = sinh²(s), s = √k·|v|
        //     d_c(0, y) = 2·asinh(√δ)/√k = 2s/√k = 2|v|.
        //
        // This pins exp_origin, poincare_distance and the metric together
        // against the independently-known scalar 2|v|. The factor of 2 is the
        // standard Poincaré-ball convention `acosh(1 + 2δ)` (matches geomstats'
        // `PoincareBallMetric.dist` and `distance_matches_textbook_formula_*`);
        // a missing/extra √k or a tanh/atanh swap (errors of order 0.1–1) trips
        // this bound.
        for &curvature in &[-1.0_f64, -0.25, -4.0] {
            let sqrt_k = (-curvature).sqrt();
            for &target_norm in &[1.0e-6_f64, 1.0e-3, 1.0e-2, 0.1, 0.5, 0.9] {
                // A fixed direction scaled to the prescribed Euclidean norm.
                let raw = array![0.3_f64, -0.5, 0.7, 0.2];
                let raw_norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
                let v: Array1<f64> = raw.mapv(|x| x * target_norm / raw_norm);
                let vnorm = v.iter().map(|x| x * x).sum::<f64>().sqrt();

                let y = exp_origin(v.view(), curvature).expect("exp");
                let origin = Array1::<f64>::zeros(v.len());
                let d = poincare_distance(origin.view(), y.view(), curvature).expect("dist");
                // Riemannian isometry: d_c(0, exp_0(v)) == λ_0·|v| == 2|v|.
                assert!(
                    (d - 2.0 * vnorm).abs() < 1.0e-12,
                    "radial isometry d_c(0,exp(v))==2|v| broke (c={curvature}, |v|={vnorm}): \
                     d={d}, 2|v|={}",
                    2.0 * vnorm
                );

                // Internal consistency cross-check: |y| == tanh(√k|v|)/√k.
                let ynorm = y.iter().map(|x| x * x).sum::<f64>().sqrt();
                let expected_ynorm = (sqrt_k * vnorm).tanh() / sqrt_k;
                assert!(
                    (ynorm - expected_ynorm).abs() < 1.0e-13,
                    "exp norm |y| must be tanh(√k|v|)/√k (c={curvature}): \
                     |y|={ynorm}, expected={expected_ynorm}"
                );

                // Round-trip log_0(exp_0(v)) == v to f64 precision.
                let back = log_origin(y.view(), curvature).expect("log");
                for i in 0..v.len() {
                    assert!(
                        (back[i] - v[i]).abs() < 1.0e-12,
                        "log∘exp round-trip broke at {i} (c={curvature}): {} vs {}",
                        back[i],
                        v[i]
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_nonnegative_curvature() {
        let v = array![0.1, 0.2];
        assert!(log_origin(v.view(), 0.0).is_err());
        assert!(log_origin(v.view(), 0.5).is_err());
        assert!(mobius_add(v.view(), v.view(), 0.0).is_err());
    }
}
