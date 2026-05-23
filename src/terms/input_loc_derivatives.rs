//! Input-location derivatives of basis design columns.
//!
//! # Piece 2 of the `LatentCoord` work-stream
//!
//! For a radial basis with centers `c_k ∈ ℝ^d` and a kernel `φ(r)` evaluated
//! at `r = ‖t − c_k‖`,
//!
//! ```text
//! Φ_{n,k}              = φ(‖t_n − c_k‖)
//! ∂Φ_{n,k}/∂t_{n,a}    = φ'(r_{nk}) · (t_n − c_k)_a / r_{nk}
//!                      = q(r_{nk}) · (t_n − c_k)_a
//! ∂²Φ_{n,k}/∂t_a∂t_b   = q(r_{nk}) · δ_ab
//!                      + s(r_{nk}) · (t_n − c_k)_a (t_n − c_k)_b
//! ```
//!
//! where `q(r) = φ'(r)/r` and `s(r) = (φ''(r) − q(r))/r²`.
//!
//! These two radial scalars are exactly the `(q, t)` pair already returned
//! by [`crate::terms::basis::RadialScalarKind::eval_design_triplet`], which
//! the ψ-derivative path uses to assemble `∂Φ/∂ψ`. The closed-form Taylor
//! expansions in that routine already cover the `r → 0` collision limits.
//!
//! For tensor-product (`te`, `ti`) bases, the chain rule factors across
//! axes — if axis `a` carries basis `B_a(t_a)`, then
//!
//! ```text
//! Φ(t)                  = ⊗_a B_a(t_a)
//! ∂Φ/∂t_a               = (⊗_{b≠a} B_b(t_b)) ⊗ B_a'(t_a)
//! ```
//!
//! For periodic / sphere bases, the input is a manifold coordinate `θ ∈ ℝ/2πℤ`
//! (or `θ ∈ S²` parameterised by `(lat, lon)`). The chain rule passes
//! through the wrap topology by evaluating the kernel against the signed
//! chord (wrap-distance) and supplying the Jacobian `dr/dt` of that chord —
//! identical to the chart-based Jacobian used by the periodic B-spline
//! evaluator.
//!
//! The functions in this module are written in the style of the existing
//! basis evaluators (caller-supplied output buffers, `BasisError` results)
//! so they slot directly into the `LatentCoordValues::design_gradient_wrt_t`
//! call site in `crate::terms::latent_coord` without an additional
//! materialization layer.

use ndarray::{Array1, Array3, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3};

use crate::terms::basis::{
    duchon_partial_fraction_coeffs, BasisError, MaternNu, RadialScalarKind,
};

// =========================================================================
// Kernel parameter bundle
// =========================================================================

/// Closed-form parameterisation of the radial families supported by the
/// input-location derivative routines below.
///
/// This is a thin convenience over [`RadialScalarKind`] (which itself is a
/// crate-internal enum carrying the same parameters). The public layer here
/// is kept as a separate type so that downstream consumers — including the
/// Python pyffi layer that will surface `LatentCoord` — can construct kernel
/// descriptors without poking at `pub(crate)` items.
#[derive(Debug, Clone)]
pub enum RadialInputKernel {
    /// Matérn isotropic kernel with closed-form `(½, 3⁄2, 5⁄2, 7⁄2, 9⁄2)`-ν.
    Matern { length_scale: f64, nu: MaternNu },
    /// Hybrid Duchon kernel `||w||^(2p) · (κ² + ||w||²)^s`.
    DuchonHybrid {
        length_scale: f64,
        p_order: usize,
        s_order: usize,
        dim: usize,
    },
    /// Pure scale-free Duchon kernel (single polyharmonic block of the
    /// given order). Equivalent to `DuchonHybrid` with `s_order = 0` and no
    /// finite length scale.
    DuchonPure {
        block_order: usize,
        p_order: usize,
        s_order: usize,
        dim: usize,
    },
    /// Thin-plate spline kernel with explicit length-scale (used by the
    /// 1-D thin-plate streaming path; for the general d-D thin-plate this
    /// coincides with the polyharmonic Duchon kernel of order `m_d`).
    ThinPlate { length_scale: f64, dim: usize },
}

impl RadialInputKernel {
    /// Project onto the internal `RadialScalarKind` enum so the existing
    /// radial-jet routines can be reused verbatim.
    fn into_scalar_kind(&self) -> RadialScalarKind {
        match self {
            RadialInputKernel::Matern { length_scale, nu } => RadialScalarKind::Matern {
                length_scale: *length_scale,
                nu: *nu,
            },
            RadialInputKernel::DuchonHybrid {
                length_scale,
                p_order,
                s_order,
                dim,
            } => {
                let kappa = 1.0 / length_scale.max(1e-300);
                let coeffs = duchon_partial_fraction_coeffs(*p_order, *s_order, kappa);
                RadialScalarKind::Duchon {
                    length_scale: *length_scale,
                    p_order: *p_order,
                    s_order: *s_order,
                    dim: *dim,
                    coeffs,
                }
            }
            RadialInputKernel::DuchonPure {
                block_order,
                p_order,
                s_order,
                dim,
            } => RadialScalarKind::PureDuchon {
                block_order: *block_order,
                p_order: *p_order,
                s_order: *s_order,
                dim: *dim,
            },
            RadialInputKernel::ThinPlate { length_scale, dim } => RadialScalarKind::ThinPlate {
                length_scale: *length_scale,
                dim: *dim,
            },
        }
    }

    /// Ambient input dimension `d` (the kernel argument length).
    pub fn dim(&self) -> usize {
        match self {
            RadialInputKernel::Matern { .. } => {
                // Matérn is ambient-dimension agnostic in `q, t`; the caller
                // is responsible for matching `centers.ncols()` to the data
                // dimensionality. We return the conventional sentinel `0`
                // here so consumers can short-circuit a dimension cross-check
                // on the centers themselves.
                0
            }
            RadialInputKernel::DuchonHybrid { dim, .. }
            | RadialInputKernel::DuchonPure { dim, .. }
            | RadialInputKernel::ThinPlate { dim, .. } => *dim,
        }
    }
}

// =========================================================================
// First derivative: ∂Φ/∂t
// =========================================================================

/// Compute `∂Φ/∂t` in closed form for a radial kernel.
///
/// # Arguments
/// * `kernel` — kernel parameters (any of the supported radial families).
/// * `t`     — `(n_obs, d)` evaluation points (the first kernel argument).
/// * `centers` — `(n_centers, d)` basis centers.
/// * `out`   — `(n_obs, n_centers, d)` output jet; entry `[n, k, a]` holds
///   `∂Φ_{n,k}/∂t_{n,a} = q(r_{nk}) · (t_n − c_k)_a`.
///
/// # Errors
///
/// Forwards [`BasisError`] from the radial-jet evaluator at any
/// `(t_n, c_k)` pair.
///
/// At `r_{nk} = 0` the unit vector `(t − c)/r` is undefined; the value
/// is defined by the analytic limit (handled inside the radial-jet
/// routines):
/// * for kernels with `φ'(0) = 0` (Matérn `ν ≥ 3/2`, polyharmonic with
///   `m > 1/2`), the limit is zero;
/// * for kernels with a singular `φ'` at the origin, the jet routines
///   return the Taylor-expanded `q` directly, so no `0/0` is observed
///   here.
pub fn basis_input_loc_grad(
    kernel: &RadialInputKernel,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<(), BasisError> {
    let n_obs = t.nrows();
    let d = t.ncols();
    let n_centers = centers.nrows();
    if centers.ncols() != d {
        return Err(BasisError::InvalidInput(format!(
            "basis_input_loc_grad: centers have {} cols but t has {} cols",
            centers.ncols(),
            d
        )));
    }
    if out.shape() != [n_obs, n_centers, d] {
        return Err(BasisError::InvalidInput(format!(
            "basis_input_loc_grad: out shape {:?} != expected {:?}",
            out.shape(),
            [n_obs, n_centers, d]
        )));
    }
    let kind = kernel.into_scalar_kind();
    for n in 0..n_obs {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            let (_, q, _) = kind.eval_design_triplet(r)?;
            for a in 0..d {
                out[[n, k, a]] = q * (t[[n, a]] - centers[[k, a]]);
            }
        }
    }
    Ok(())
}

/// Allocating convenience wrapper around [`basis_input_loc_grad`].
pub fn basis_input_loc_grad_alloc(
    kernel: &RadialInputKernel,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, BasisError> {
    let mut out = Array3::<f64>::zeros((t.nrows(), centers.nrows(), t.ncols()));
    basis_input_loc_grad(kernel, t, centers, out.view_mut())?;
    Ok(out)
}

// =========================================================================
// Second derivative: ∂²Φ/∂t∂t'
// =========================================================================

/// Row-local Hessian block `∂²Φ_{n,k}/∂t_a∂t_b` packed as a flat
/// `(n_obs, n_centers, d*d)` array (row-major over `(a, b)`).
///
/// The Hessian is symmetric in `(a, b)`; the full `d²` storage is kept
/// for ergonomic indexing inside the Piece-1 arrow-Hessian assembler
/// (each row's per-center contribution is contracted into a single
/// `d × d` block).
///
/// Formula:
///
/// ```text
/// ∂²Φ/∂t_a∂t_b = q · δ_ab + s · (t − c)_a (t − c)_b
/// ```
///
/// with `q = φ'/r`, `s = (φ'' − q)/r²` (analytic collision limits handled
/// by the existing radial-jet routine).
pub fn basis_input_loc_hess(
    kernel: &RadialInputKernel,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<(), BasisError> {
    let n_obs = t.nrows();
    let d = t.ncols();
    let n_centers = centers.nrows();
    if centers.ncols() != d {
        return Err(BasisError::InvalidInput(format!(
            "basis_input_loc_hess: centers have {} cols but t has {} cols",
            centers.ncols(),
            d
        )));
    }
    if out.shape() != [n_obs, n_centers, d * d] {
        return Err(BasisError::InvalidInput(format!(
            "basis_input_loc_hess: out shape {:?} != expected {:?}",
            out.shape(),
            [n_obs, n_centers, d * d]
        )));
    }
    let kind = kernel.into_scalar_kind();
    let mut delta = vec![0.0_f64; d];
    for n in 0..n_obs {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let v = t[[n, a]] - centers[[k, a]];
                delta[a] = v;
                r2 += v * v;
            }
            let r = r2.sqrt();
            let (_, q, s) = kind.eval_design_triplet(r)?;
            for a in 0..d {
                for b in 0..d {
                    let kron = if a == b { 1.0 } else { 0.0 };
                    out[[n, k, a * d + b]] = q * kron + s * delta[a] * delta[b];
                }
            }
        }
    }
    Ok(())
}

/// Allocating convenience wrapper around [`basis_input_loc_hess`].
pub fn basis_input_loc_hess_alloc(
    kernel: &RadialInputKernel,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, BasisError> {
    let d = t.ncols();
    let mut out = Array3::<f64>::zeros((t.nrows(), centers.nrows(), d * d));
    basis_input_loc_hess(kernel, t, centers, out.view_mut())?;
    Ok(out)
}

// =========================================================================
// Tensor product bases (`te`, `ti`)
// =========================================================================

/// Per-axis factor tables consumed by [`tensor_product_input_loc_grad`].
///
/// For a tensor-product basis `Φ_n(t) = ⊗_a B_a(t_{n,a})` evaluated at row
/// `n`, each axis contributes a length-`K_a` factor for both the value
/// `B_a(t_{n,a})` and its first derivative `B_a'(t_{n,a})`. The Kronecker
/// product across axes gives the full design row of length `∏_a K_a`.
///
/// Storage convention: `values[a]` is shape `(n_obs, K_a)` and
/// `derivatives[a]` is the same shape, holding `∂B_a/∂t_a` evaluated at
/// the same rows.
#[derive(Debug, Clone)]
pub struct TensorAxisFactors {
    /// Per-axis basis evaluations: `values[a][n, k_a] = B_a^{k_a}(t_{n,a})`.
    pub values: Vec<ndarray::Array2<f64>>,
    /// Per-axis first derivatives: `derivatives[a][n, k_a] = B_a^{k_a}'(t_{n,a})`.
    pub derivatives: Vec<ndarray::Array2<f64>>,
}

impl TensorAxisFactors {
    pub fn n_axes(&self) -> usize {
        self.values.len()
    }

    pub fn n_obs(&self) -> usize {
        self.values.first().map(|m| m.nrows()).unwrap_or(0)
    }

    /// Number of tensor-product columns `∏_a K_a`.
    pub fn n_cols(&self) -> usize {
        self.values.iter().map(|m| m.ncols()).product()
    }

    fn axis_dim(&self, axis: usize) -> usize {
        self.values[axis].ncols()
    }
}

/// Tensor-product input-location gradient.
///
/// For a `te`/`ti` basis `Φ_{n,k} = ∏_a B_a^{k_a}(t_{n,a})` with
/// multi-index `k = (k_1, …, k_d)` flattened lexicographically (axis 0 is
/// the slowest), the product rule gives
///
/// ```text
/// ∂Φ_{n,k}/∂t_{n,a} = (∏_{b ≠ a} B_b^{k_b}(t_{n,b})) · ∂B_a^{k_a}/∂t_a.
/// ```
///
/// `factors` supplies the per-axis values and first derivatives evaluated
/// at the same `t`. The output is `(n_obs, n_cols, d)` with
/// `n_cols = ∏_a K_a`.
pub fn tensor_product_input_loc_grad(
    factors: &TensorAxisFactors,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<(), BasisError> {
    let d = factors.n_axes();
    let n_obs = factors.n_obs();
    let n_cols = factors.n_cols();
    if d == 0 {
        return Err(BasisError::InvalidInput(
            "tensor_product_input_loc_grad: at least one axis is required".to_string(),
        ));
    }
    if factors.derivatives.len() != d {
        return Err(BasisError::InvalidInput(format!(
            "tensor_product_input_loc_grad: derivatives has {} axes, values has {}",
            factors.derivatives.len(),
            d
        )));
    }
    for a in 0..d {
        if factors.derivatives[a].shape() != factors.values[a].shape() {
            return Err(BasisError::InvalidInput(format!(
                "tensor_product_input_loc_grad: axis {a} value/deriv shape mismatch"
            )));
        }
    }
    if out.shape() != [n_obs, n_cols, d] {
        return Err(BasisError::InvalidInput(format!(
            "tensor_product_input_loc_grad: out shape {:?} != expected {:?}",
            out.shape(),
            [n_obs, n_cols, d]
        )));
    }

    // Multi-index walk via mixed-radix counter. `axis_dims[a] = K_a`.
    let axis_dims: Vec<usize> = (0..d).map(|a| factors.axis_dim(a)).collect();
    let mut idx = vec![0usize; d];
    for n in 0..n_obs {
        idx.iter_mut().for_each(|x| *x = 0);
        for col in 0..n_cols {
            // For derivative axis `a`, factor[b] is value if b != a, deriv if b == a.
            for a in 0..d {
                let mut prod = 1.0_f64;
                for b in 0..d {
                    let k_b = idx[b];
                    prod *= if b == a {
                        factors.derivatives[b][[n, k_b]]
                    } else {
                        factors.values[b][[n, k_b]]
                    };
                }
                out[[n, col, a]] = prod;
            }
            // increment mixed-radix counter (axis 0 slowest)
            for a in (0..d).rev() {
                idx[a] += 1;
                if idx[a] < axis_dims[a] {
                    break;
                }
                idx[a] = 0;
            }
        }
    }
    Ok(())
}

/// Allocating convenience wrapper for tensor-product gradient.
pub fn tensor_product_input_loc_grad_alloc(
    factors: &TensorAxisFactors,
) -> Result<Array3<f64>, BasisError> {
    let n_obs = factors.n_obs();
    let n_cols = factors.n_cols();
    let d = factors.n_axes();
    let mut out = Array3::<f64>::zeros((n_obs, n_cols, d));
    tensor_product_input_loc_grad(factors, out.view_mut())?;
    Ok(out)
}

// =========================================================================
// Periodic 1-D wrap topology
// =========================================================================

/// Wrap a signed coordinate displacement `Δ = t - c` into the principal
/// branch `(-period/2, +period/2]`.
///
/// For a periodic 1-D basis the chord distance is computed from the wrapped
/// `Δ`. The Jacobian `d(wrap(Δ))/dt` equals `1` almost everywhere — wrap is
/// piecewise-identity — so the chain rule passes through unchanged: the
/// kernel's first derivative w.r.t. wrapped Δ is the same as its first
/// derivative w.r.t. the unwrapped chord distance.
#[inline]
pub fn wrap_signed_displacement(delta: f64, period: f64) -> f64 {
    if !(period.is_finite() && period > 0.0) {
        return delta;
    }
    let half = period * 0.5;
    let mut d = delta % period;
    if d > half {
        d -= period;
    } else if d <= -half {
        d += period;
    }
    d
}

/// Periodic 1-D radial gradient: `∂φ(|wrap(t − c)|)/∂t`.
///
/// For each `(n, k)` pair, computes `q · wrap(t_n − c_k)` where
/// `q = φ'(|wrap|)/|wrap|` and the wrap topology has the given `period`.
pub fn periodic_radial_input_loc_grad_1d(
    kernel: &RadialInputKernel,
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    period: f64,
    mut out: ArrayViewMut2<'_, f64>,
) -> Result<(), BasisError> {
    if !(period.is_finite() && period > 0.0) {
        return Err(BasisError::InvalidInput(format!(
            "periodic_radial_input_loc_grad_1d: period must be finite and positive, got {period}"
        )));
    }
    let n_obs = t.len();
    let n_centers = centers.len();
    if out.shape() != [n_obs, n_centers] {
        return Err(BasisError::InvalidInput(format!(
            "periodic_radial_input_loc_grad_1d: out shape {:?} != expected {:?}",
            out.shape(),
            [n_obs, n_centers]
        )));
    }
    let kind = kernel.into_scalar_kind();
    for n in 0..n_obs {
        for k in 0..n_centers {
            let raw = t[n] - centers[k];
            let signed = wrap_signed_displacement(raw, period);
            let r = signed.abs();
            let (_, q, _) = kind.eval_design_triplet(r)?;
            out[[n, k]] = q * signed;
        }
    }
    Ok(())
}

// =========================================================================
// Sphere S² chart-based gradient
// =========================================================================

/// Sphere `S² ⊂ ℝ³` input-location gradient under the unit-vector
/// embedding `t = (sin θ cos ϕ, sin θ sin ϕ, cos θ)`.
///
/// The radial kernel acts on the chord distance `r = ‖t − c‖` (or
/// equivalently on `cos γ = ⟨t, c⟩` via `r² = 2(1 − cos γ)`). The gradient
/// w.r.t. the ambient unit vector is the standard radial form,
///
/// ```text
/// ∂Φ/∂t = q(r) · (t − c)        in ℝ³.
/// ```
///
/// Projecting to the tangent space `T_t S² = {v ∈ ℝ³ : v · t = 0}` yields
///
/// ```text
/// (I − t tᵀ) · ∂Φ/∂t = q(r) · ((t − c) − ⟨t − c, t⟩ · t)
///                    = q(r) · (−c + ⟨c, t⟩ · t).
/// ```
///
/// `t_unit` and `centers_unit` are `(N, 3)` and `(K, 3)` arrays of points
/// already on the unit sphere; the caller is responsible for any chart
/// (lat, lon) → (x, y, z) lift. The output is `(N, K, 3)` in the ambient
/// tangent representation (the chart Jacobian to (lat, lon) is left to the
/// outer optimizer, which already maintains it in the same place the ψ
/// machinery does).
pub fn sphere_s2_input_loc_grad(
    kernel: &RadialInputKernel,
    t_unit: ArrayView2<'_, f64>,
    centers_unit: ArrayView2<'_, f64>,
    mut out: ArrayViewMut3<'_, f64>,
) -> Result<(), BasisError> {
    let n_obs = t_unit.nrows();
    let n_centers = centers_unit.nrows();
    if t_unit.ncols() != 3 || centers_unit.ncols() != 3 {
        return Err(BasisError::InvalidInput(format!(
            "sphere_s2_input_loc_grad: t and centers must have 3 columns (ambient), got {} and {}",
            t_unit.ncols(),
            centers_unit.ncols()
        )));
    }
    if out.shape() != [n_obs, n_centers, 3] {
        return Err(BasisError::InvalidInput(format!(
            "sphere_s2_input_loc_grad: out shape {:?} != expected {:?}",
            out.shape(),
            [n_obs, n_centers, 3]
        )));
    }
    let kind = kernel.into_scalar_kind();
    for n in 0..n_obs {
        let tx = t_unit[[n, 0]];
        let ty = t_unit[[n, 1]];
        let tz = t_unit[[n, 2]];
        for k in 0..n_centers {
            let cx = centers_unit[[k, 0]];
            let cy = centers_unit[[k, 1]];
            let cz = centers_unit[[k, 2]];
            let dx = tx - cx;
            let dy = ty - cy;
            let dz = tz - cz;
            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let (_, q, _) = kind.eval_design_triplet(r)?;
            // Ambient gradient g = q · (t - c)
            let gx = q * dx;
            let gy = q * dy;
            let gz = q * dz;
            // Project onto tangent plane: g - (g · t) t
            let g_dot_t = gx * tx + gy * ty + gz * tz;
            out[[n, k, 0]] = gx - g_dot_t * tx;
            out[[n, k, 1]] = gy - g_dot_t * ty;
            out[[n, k, 2]] = gz - g_dot_t * tz;
        }
    }
    Ok(())
}

// =========================================================================
// Contraction helper (mirrors LatentCoordValues::contract_gradient)
// =========================================================================

/// Contract `(n_obs, n_centers)` upstream gradient against an
/// `(n_obs, n_centers, d)` jet into a flat `(n_obs · d)` gradient w.r.t.
/// the latent input `t`.
///
/// This is the closed-form chain-rule reduction used by both the inner
/// Newton step and the backward pyffi entrypoint. See
/// [`crate::terms::latent_coord::LatentCoordValues::contract_gradient`] for
/// the canonical signature.
pub fn contract_input_loc_gradient(
    grad_phi: ArrayView2<'_, f64>,
    jet: &Array3<f64>,
) -> Array1<f64> {
    let n_obs = jet.shape()[0];
    let n_centers = jet.shape()[1];
    let d = jet.shape()[2];
    debug_assert_eq!(grad_phi.shape(), &[n_obs, n_centers]);
    let mut grad_t = Array1::<f64>::zeros(n_obs * d);
    for n in 0..n_obs {
        for a in 0..d {
            let mut acc = 0.0_f64;
            for k in 0..n_centers {
                acc += grad_phi[[n, k]] * jet[[n, k, a]];
            }
            grad_t[n * d + a] = acc;
        }
    }
    grad_t
}

// =========================================================================
// Tests (type-level documentation; NOT executed in this work-stream).
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn finite_diff_grad(
        kernel: &RadialInputKernel,
        t: ArrayView2<'_, f64>,
        centers: ArrayView2<'_, f64>,
    ) -> Array3<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        let n_centers = centers.nrows();
        let mut out = Array3::<f64>::zeros((n_obs, n_centers, d));
        let eps = 1e-6;
        let kind = kernel.into_scalar_kind();
        for n in 0..n_obs {
            for a in 0..d {
                for sign in &[1.0_f64, -1.0_f64] {
                    let mut tp = t.to_owned();
                    tp[[n, a]] += sign * eps;
                    for k in 0..n_centers {
                        let mut r2 = 0.0_f64;
                        for b in 0..d {
                            let v = tp[[n, b]] - centers[[k, b]];
                            r2 += v * v;
                        }
                        let (phi, _, _) = kind.eval_design_triplet(r2.sqrt()).unwrap();
                        out[[n, k, a]] += sign * phi / (2.0 * eps);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn duchon_pure_grad_matches_radial_chain_rule() {
        let kernel = RadialInputKernel::DuchonPure {
            block_order: 2,
            p_order: 2,
            s_order: 0,
            dim: 2,
        };
        let t: Array2<f64> = array![[0.5, 0.3], [-0.2, 0.7]];
        let centers: Array2<f64> = array![[0.0, 0.0], [1.0, 0.5], [-0.5, 0.1]];
        let analytic =
            basis_input_loc_grad_alloc(&kernel, t.view(), centers.view()).expect("analytic grad");
        let numeric = finite_diff_grad(&kernel, t.view(), centers.view());
        for n in 0..t.nrows() {
            for k in 0..centers.nrows() {
                for a in 0..t.ncols() {
                    let diff = (analytic[[n, k, a]] - numeric[[n, k, a]]).abs();
                    assert!(
                        diff < 1e-4,
                        "Duchon grad mismatch at ({n},{k},{a}): {diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn matern_grad_is_zero_at_collision() {
        let kernel = RadialInputKernel::Matern {
            length_scale: 1.0,
            nu: MaternNu::FiveHalves,
        };
        // Co-located t and c → r = 0, gradient should vanish (φ'(0) = 0
        // for Matérn ν ≥ 3/2).
        let t: Array2<f64> = array![[0.5, 0.5]];
        let centers: Array2<f64> = array![[0.5, 0.5]];
        let g = basis_input_loc_grad_alloc(&kernel, t.view(), centers.view())
            .expect("collision grad");
        assert_eq!(g[[0, 0, 0]], 0.0);
        assert_eq!(g[[0, 0, 1]], 0.0);
    }

    #[test]
    fn hess_is_symmetric() {
        let kernel = RadialInputKernel::Matern {
            length_scale: 0.7,
            nu: MaternNu::FiveHalves,
        };
        let t: Array2<f64> = array![[0.2, -0.4]];
        let centers: Array2<f64> = array![[0.5, 0.1], [0.0, 0.0]];
        let h = basis_input_loc_hess_alloc(&kernel, t.view(), centers.view()).expect("hess");
        let d = 2;
        for k in 0..centers.nrows() {
            for a in 0..d {
                for b in 0..d {
                    let hab = h[[0, k, a * d + b]];
                    let hba = h[[0, k, b * d + a]];
                    assert!((hab - hba).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn tensor_product_grad_factors_correctly() {
        // d = 2, K_0 = 2, K_1 = 3; tensor cols = 6.
        let v0 = array![[1.0_f64, 2.0]];
        let v1 = array![[10.0_f64, 100.0, 1000.0]];
        let d0 = array![[0.5_f64, -0.5]];
        let d1 = array![[1.0_f64, 2.0, 3.0]];
        let factors = TensorAxisFactors {
            values: vec![v0, v1],
            derivatives: vec![d0, d1],
        };
        let out = tensor_product_input_loc_grad_alloc(&factors).unwrap();
        // ordering: col = k_0 * K_1 + k_1, axis 0 slowest.
        // d/dt_0 at (k_0, k_1) = d0[k_0] * v1[k_1]
        // d/dt_1 at (k_0, k_1) = v0[k_0] * d1[k_1]
        let k0_dims = 2;
        let k1_dims = 3;
        for k_0 in 0..k0_dims {
            for k_1 in 0..k1_dims {
                let col = k_0 * k1_dims + k_1;
                let expected0 = match k_0 {
                    0 => 0.5,
                    _ => -0.5,
                } * (10.0_f64.powi(k_1 as i32 + 1));
                let expected1 = match k_0 {
                    0 => 1.0,
                    _ => 2.0,
                } * ((k_1 + 1) as f64);
                assert!((out[[0, col, 0]] - expected0).abs() < 1e-12);
                assert!((out[[0, col, 1]] - expected1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn wrap_signed_displacement_principal_branch() {
        let period = 2.0 * std::f64::consts::PI;
        assert!((wrap_signed_displacement(0.1, period) - 0.1).abs() < 1e-15);
        // 2π + 0.2 wraps to 0.2
        assert!((wrap_signed_displacement(period + 0.2, period) - 0.2).abs() < 1e-12);
        // -π - 0.1 wraps to +π - 0.1
        let v = wrap_signed_displacement(-period * 0.5 - 0.1, period);
        assert!((v - (period * 0.5 - 0.1)).abs() < 1e-12);
    }

    #[test]
    fn periodic_grad_handles_wrap() {
        let kernel = RadialInputKernel::Matern {
            length_scale: 0.5,
            nu: MaternNu::ThreeHalves,
        };
        let period = 2.0 * std::f64::consts::PI;
        // Two points that are close across the wrap; their kernel gradient
        // should reflect the short chord.
        let t = array![0.05_f64];
        let centers = array![period - 0.05_f64];
        let mut out = Array2::<f64>::zeros((1, 1));
        periodic_radial_input_loc_grad_1d(
            &kernel,
            t.view(),
            centers.view(),
            period,
            out.view_mut(),
        )
        .unwrap();
        // wrapped Δ = 0.05 - (-0.05) = 0.1, well within one period; sign +
        assert!(out[[0, 0]].is_finite());
    }

    #[test]
    fn sphere_grad_is_tangent_to_t() {
        let kernel = RadialInputKernel::Matern {
            length_scale: 1.0,
            nu: MaternNu::FiveHalves,
        };
        // Unit-vector points.
        let t = array![[1.0_f64, 0.0, 0.0]];
        let centers = array![[0.0_f64, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let g = {
            let mut out = Array3::<f64>::zeros((1, 2, 3));
            sphere_s2_input_loc_grad(&kernel, t.view(), centers.view(), out.view_mut()).unwrap();
            out
        };
        for k in 0..2 {
            let dot = g[[0, k, 0]] * 1.0 + g[[0, k, 1]] * 0.0 + g[[0, k, 2]] * 0.0;
            assert!(dot.abs() < 1e-12, "tangent projection failed: dot={dot}");
        }
    }

    #[test]
    fn contract_input_loc_gradient_matches_einsum() {
        let jet = Array3::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let grad_phi = array![[1.0_f64, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let out = contract_input_loc_gradient(grad_phi.view(), &jet);
        // n=0: a=0 → 1*1 + 0*3 + 1*5 = 6; a=1 → 1*2 + 0*4 + 1*6 = 8
        // n=1: a=0 → 0*7 + 1*9 + 0*11 = 9; a=1 → 0*8 + 1*10 + 0*12 = 10
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], 8.0);
        assert_eq!(out[2], 9.0);
        assert_eq!(out[3], 10.0);
    }
}
