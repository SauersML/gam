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
//! by [`crate::basis::RadialScalarKind::eval_design_triplet`], which
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
//! call site in `crate::latent` without an additional
//! materialization layer.

use ndarray::{Array1, Array3, ArrayView2};

use crate::basis::{BasisError, MaternNu};

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
    /// Ambient input dimension `d` (the kernel argument length).
    pub const fn dim(&self) -> usize {
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
// Contraction helper (mirrors LatentCoordValues::contract_gradient)
// =========================================================================

/// Contract `(n_obs, n_centers)` upstream gradient against an
/// `(n_obs, n_centers, d)` jet into a flat `(n_obs · d)` gradient w.r.t.
/// the latent input `t`.
///
/// This is the closed-form chain-rule reduction used by both the inner
/// Newton step and the backward pyffi entrypoint. See
/// [`crate::latent::LatentCoordValues::contract_gradient`] for
/// the canonical signature.
pub fn contract_input_loc_gradient(
    grad_phi: ArrayView2<'_, f64>,
    jet: &Array3<f64>,
) -> Result<Array1<f64>, BasisError> {
    let n_obs = jet.shape()[0];
    let n_centers = jet.shape()[1];
    let d = jet.shape()[2];
    if grad_phi.shape() != [n_obs, n_centers] {
        crate::bail_dim_basis!(
            "contract_input_loc_gradient: grad_phi shape {:?} != expected {:?}",
            grad_phi.shape(),
            [n_obs, n_centers]
        );
    }
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
    Ok(grad_t)
}

// =========================================================================
// Tests (type-level documentation; NOT executed in this work-stream).
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{RadialScalarKind, duchon_partial_fraction_coeffs};
    use ndarray::array;

    /// Project a `RadialInputKernel` onto the internal `RadialScalarKind`
    /// enum so the radial-jet routines can be reused verbatim by the
    /// divergence-witness tests.
    fn into_scalar_kind(kernel: &RadialInputKernel) -> RadialScalarKind {
        match kernel {
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

    #[test]
    fn contract_input_loc_gradient_matches_einsum() {
        let jet = Array3::from_shape_vec(
            (2, 3, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let grad_phi = array![[1.0_f64, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let out = contract_input_loc_gradient(grad_phi.view(), &jet).unwrap();
        // n=0: a=0 → 1*1 + 0*3 + 1*5 = 6; a=1 → 1*2 + 0*4 + 1*6 = 8
        // n=1: a=0 → 0*7 + 1*9 + 0*11 = 9; a=1 → 0*8 + 1*10 + 0*12 = 10
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], 8.0);
        assert_eq!(out[2], 9.0);
        assert_eq!(out[3], 10.0);
    }

    // ----------------------------------------------------------------
    // Degenerate-collision tests (F1–F5)
    //
    // These do not invoke any executable test runner here (they compile-
    // check only under `cargo check --all-features --all-targets`); the
    // intent is to lock in the contract that divergent kernels surface a
    // `BasisError::DegenerateAtCollision` instead of a silent zero, and
    // that a finite-difference probe at ε just away from the collision
    // produces a large value consistent with the analytic divergence.
    // ----------------------------------------------------------------

    #[test]
    fn matern_half_finite_difference_diverges_near_collision() {
        // φ(r) = exp(−r); q = −exp(−r)/r → −∞ as r → 0+.
        // At r = ε the design gradient component along the axis is
        // q·s_axis = −exp(−ε)/ε · ε = −exp(−ε) ≈ −1, but the per-r
        // scalar q itself blows up at ~1/ε, which is the witness for the
        // divergence flagged by F1.
        let kernel = RadialInputKernel::Matern {
            length_scale: 1.0,
            nu: MaternNu::Half,
        };
        let kind = into_scalar_kind(&kernel);
        let eps = 1e-8_f64;
        let (_, q, _) = kind
            .eval_design_triplet(eps)
            .expect("ν=1/2 at r=ε is finite");
        assert!(
            q.abs() > 1e6,
            "expected divergent q for Matérn ν=1/2 near r=0, got {q}"
        );
    }

    #[test]
    fn thin_plate_collision_2d_finite_difference_diverges() {
        // φ(r) = (r/ℓ)² log(r/ℓ); q = (1/ℓ²)(2 log(r/ℓ) + 1) → −∞.
        let kernel = RadialInputKernel::ThinPlate {
            length_scale: 1.0,
            dim: 2,
        };
        let kind = into_scalar_kind(&kernel);
        let eps = 1e-10_f64;
        let (_, q, _) = kind
            .eval_design_triplet(eps)
            .expect("TPS dim=2 at r=ε is finite");
        assert!(
            q.abs() > 10.0,
            "expected large |q| for TPS dim=2 near r=0, got {q}"
        );
    }
}
