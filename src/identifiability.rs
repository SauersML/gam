//! Identifiability primitives — frontier 2026 unification.
//!
//! Two new analytic primitives that compose with the existing manifold-SAE
//! infrastructure, formalising the empirical recipe from `auto_exp_38`
//! (3 supervised HSV axes + 3 free axes that unsupervisedly align with
//! name-semantics):
//!
//! * `MechanismSparsityJacobian` — column-2-norm penalty on the decoder
//!   Jacobian: `Σ_k ||J_dec[:, k]||_2`. For an affine decoder `x = W t + b`,
//!   `J_dec = W` so this reduces to `Σ_k ||W[:, k]||_2`. Implements the
//!   mechanism-sparsity-identifiability theorem of Lachapelle (2401.04890).
//!   The penalty is smoothed by an `epsilon` to keep gradients defined at the
//!   origin: `ψ_k(W) = √(||W[:, k]||² + ε²) − ε`.
//!
//! * `ConditionalPriorIvae` — auxiliary-conditional Gaussian log-prior on the
//!   latent (Khemakhem iVAE, 2107.10098). Given per-row mean `μ_i(u_n)` and
//!   scale `σ_i(u_n)` evaluated at auxiliaries `u_n`, evaluates
//!   `−log p(t_n | u_n) = ½ Σ_{n,i} [ ((t_{n,i} − μ_{n,i}) / σ_{n,i})²
//!                                    + 2 log σ_{n,i} + log 2π ]`
//!   and returns its analytic gradient w.r.t. `t`.
//!
//! Both primitives are pure compute helpers — they take dense arrays and
//! return `(value, grad)` so they can be summed into any external optimiser
//! (PyTorch, JAX, the existing gam solver via a custom term, etc.) without
//! plumbing through the full dispatch registry. They are intentionally
//! standalone to keep the patch surface tight while parallel agents are
//! editing `analytic_penalties.rs`.

use ndarray::{Array2, ArrayView1, ArrayView2};

/// Smoothed column-2-norm of the decoder Jacobian.
///
/// Returns `(value, grad)` where `value = Σ_k √(Σ_d W[d,k]² + ε²) − ε`
/// scaled by `weight`, and `grad[d, k] = weight · W[d, k] / √(Σ_d W[d,k]² + ε²)`.
#[derive(Debug, Clone)]
pub struct MechanismSparsityJacobian {
    pub weight: f64,
    pub epsilon: f64,
}

impl MechanismSparsityJacobian {
    pub fn new(weight: f64, epsilon: f64) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "MechanismSparsityJacobian: weight must be finite and >0, got {weight}"
            ));
        }
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(format!(
                "MechanismSparsityJacobian: epsilon must be finite and >0, got {epsilon}"
            ));
        }
        Ok(Self { weight, epsilon })
    }

    /// Evaluate value and gradient on a (d_obs, k_latent) decoder weight matrix.
    pub fn value_and_grad(&self, w: ArrayView2<f64>) -> (f64, Array2<f64>) {
        let (d, k) = w.dim();
        let eps2 = self.epsilon * self.epsilon;
        let mut grad = Array2::<f64>::zeros((d, k));
        let mut value = 0.0;
        for col in 0..k {
            let mut sq = 0.0;
            for row in 0..d {
                sq += w[[row, col]] * w[[row, col]];
            }
            let denom = (sq + eps2).sqrt();
            value += denom - self.epsilon;
            let factor = self.weight / denom;
            for row in 0..d {
                grad[[row, col]] = factor * w[[row, col]];
            }
        }
        (self.weight * value, grad)
    }

    /// Diagonal of the Hessian wrt vec(W). Used as a Newton preconditioner.
    pub fn hessian_diag(&self, w: ArrayView2<f64>) -> Array2<f64> {
        let (d, k) = w.dim();
        let eps2 = self.epsilon * self.epsilon;
        let mut out = Array2::<f64>::zeros((d, k));
        for col in 0..k {
            let mut sq = 0.0;
            for row in 0..d {
                sq += w[[row, col]] * w[[row, col]];
            }
            let denom = (sq + eps2).sqrt();
            let inv = 1.0 / denom;
            let inv3 = inv * inv * inv;
            for row in 0..d {
                // ∂² / ∂W[d,k]² of √(||·||²+ε²) = 1/r − W[d,k]²/r³
                out[[row, col]] = self.weight * (inv - w[[row, col]] * w[[row, col]] * inv3);
            }
        }
        out
    }
}

/// iVAE-style auxiliary-conditional Gaussian log-prior on the latent block.
///
/// Stores per-row conditional means `μ` of shape `(n_rows, latent_dim)` and
/// scales `σ` of shape `(n_rows, latent_dim)`, where `(μ_{n,i}, σ_{n,i})` are
/// presumed evaluated by some external Smooth at the auxiliary `u_n`. The
/// negative log-prior contribution to the latent objective is
///
///   `½ Σ_n Σ_i [ ((t_{n,i} − μ_{n,i}) / σ_{n,i})²
///                + 2 log σ_{n,i} + log 2π ]`
///
/// scaled by `weight`. The gradient w.r.t. `t` is `(t − μ) / σ²` (times
/// `weight`); the gradient w.r.t. `μ` is its negative. Per-row scales make
/// this strictly more general than a fixed `N(0, I)`, which is recovered by
/// `μ ≡ 0`, `σ ≡ 1`.
#[derive(Debug, Clone)]
pub struct ConditionalPriorIvae {
    pub mean: Array2<f64>,
    pub scale: Array2<f64>,
    pub weight: f64,
}

impl ConditionalPriorIvae {
    pub fn new(mean: Array2<f64>, scale: Array2<f64>, weight: f64) -> Result<Self, String> {
        if mean.dim() != scale.dim() {
            return Err(format!(
                "ConditionalPriorIvae: mean shape {:?} != scale shape {:?}",
                mean.dim(),
                scale.dim()
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ConditionalPriorIvae: weight must be finite and >0, got {weight}"
            ));
        }
        for &v in scale.iter() {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!(
                    "ConditionalPriorIvae: every scale must be finite and >0, got {v}"
                ));
            }
        }
        for &v in mean.iter() {
            if !v.is_finite() {
                return Err("ConditionalPriorIvae: mean contains non-finite entry".to_string());
            }
        }
        Ok(Self {
            mean,
            scale,
            weight,
        })
    }

    /// Evaluate negative-log-prior value and gradient w.r.t. latent t.
    pub fn value_and_grad(&self, t: ArrayView2<f64>) -> (f64, Array2<f64>) {
        assert_eq!(
            t.dim(),
            self.mean.dim(),
            "ConditionalPriorIvae: t/mean shape mismatch"
        );
        let (n, d) = t.dim();
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let mut grad = Array2::<f64>::zeros((n, d));
        let mut value = 0.0;
        for row in 0..n {
            for col in 0..d {
                let mu = self.mean[[row, col]];
                let sigma = self.scale[[row, col]];
                let z = (t[[row, col]] - mu) / sigma;
                value += 0.5 * (z * z + 2.0 * sigma.ln() + log_2pi);
                grad[[row, col]] = self.weight * z / sigma;
            }
        }
        (self.weight * value, grad)
    }

    /// Evaluate value only — useful when only the loss is needed.
    pub fn value(&self, t: ArrayView2<f64>) -> f64 {
        self.value_and_grad(t).0
    }
}

/// Helper: evaluate a piecewise-linear "smooth" `f(u)` columnwise, given a
/// (k_centres, latent_dim) coefficient table and a (n_rows,) auxiliary vector
/// `u`. Used by the Python wrapper to back the iVAE per-latent (μ_i(u), σ_i(u))
/// without having to round-trip through gam's full Smooth machinery for the
/// minimal experiments. Centres are assumed evenly spaced in [u_min, u_max].
pub fn piecewise_linear_eval(
    u: ArrayView1<f64>,
    coeffs: ArrayView2<f64>,
    u_min: f64,
    u_max: f64,
) -> Array2<f64> {
    let (k, d) = coeffs.dim();
    assert!(k >= 2, "piecewise_linear_eval: need ≥2 centres");
    let n = u.len();
    let mut out = Array2::<f64>::zeros((n, d));
    let step = (u_max - u_min) / (k - 1) as f64;
    for (row, &val) in u.iter().enumerate() {
        let pos = ((val - u_min) / step).clamp(0.0, (k - 1) as f64 - 1e-12);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(k - 1);
        let frac = pos - lo as f64;
        for col in 0..d {
            out[[row, col]] = coeffs[[lo, col]] * (1.0 - frac) + coeffs[[hi, col]] * frac;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    #[test]
    fn mechanism_sparsity_jacobian_value_matches_closed_form() {
        let w = array![[3.0_f64, 0.0], [4.0, 0.0]]; // col0 norm=5, col1 norm=0
        let pen = MechanismSparsityJacobian::new(1.0, 1.0e-8).unwrap();
        let (v, _g) = pen.value_and_grad(w.view());
        assert!((v - 5.0).abs() < 1e-6, "value {v} expected ≈5");
    }

    #[test]
    fn mechanism_sparsity_jacobian_grad_matches_finite_diff() {
        let w = array![[0.5_f64, -1.2, 0.3], [1.1, 0.4, -0.7]];
        let pen = MechanismSparsityJacobian::new(2.5, 1.0e-6).unwrap();
        let (_, g) = pen.value_and_grad(w.view());
        let h = 1.0e-5;
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp[[i, j]] += h;
                wm[[i, j]] -= h;
                let (vp, _) = pen.value_and_grad(wp.view());
                let (vm, _) = pen.value_and_grad(wm.view());
                let fd = (vp - vm) / (2.0 * h);
                assert!(
                    (g[[i, j]] - fd).abs() < 1e-4,
                    "grad[{i},{j}] = {} vs fd {}",
                    g[[i, j]],
                    fd
                );
            }
        }
    }

    #[test]
    fn mechanism_sparsity_jacobian_rejects_bad_input() {
        assert!(MechanismSparsityJacobian::new(-1.0, 1e-6).is_err());
        assert!(MechanismSparsityJacobian::new(1.0, 0.0).is_err());
    }

    #[test]
    fn conditional_prior_ivae_zero_mean_unit_scale_matches_standard_gaussian() {
        let n = 4;
        let d = 3;
        let mean = Array2::<f64>::zeros((n, d));
        let scale = Array2::<f64>::ones((n, d));
        let pen = ConditionalPriorIvae::new(mean, scale, 1.0).unwrap();
        let t = Array2::<f64>::from_elem((n, d), 0.5);
        let (v, g) = pen.value_and_grad(t.view());
        // Expected ½ Σ t² + (n*d/2) log 2π
        let expected_quad: f64 = 0.5 * t.iter().map(|x| x * x).sum::<f64>();
        let expected = expected_quad + 0.5 * (n * d) as f64 * (2.0 * std::f64::consts::PI).ln();
        assert!((v - expected).abs() < 1e-9);
        for &gv in g.iter() {
            assert!((gv - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn conditional_prior_ivae_grad_matches_finite_diff() {
        let mean = array![[0.1_f64, -0.2], [0.3, 0.0]];
        let scale = array![[0.5_f64, 2.0], [1.0, 0.7]];
        let t = array![[0.4_f64, -0.1], [1.2, 0.6]];
        let pen = ConditionalPriorIvae::new(mean, scale, 1.7).unwrap();
        let (_, g) = pen.value_and_grad(t.view());
        let h = 1.0e-5;
        for i in 0..t.nrows() {
            for j in 0..t.ncols() {
                let mut tp = t.clone();
                let mut tm = t.clone();
                tp[[i, j]] += h;
                tm[[i, j]] -= h;
                let vp = pen.value(tp.view());
                let vm = pen.value(tm.view());
                let fd = (vp - vm) / (2.0 * h);
                assert!((g[[i, j]] - fd).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn conditional_prior_ivae_rejects_nonpositive_scale() {
        let mean = Array2::<f64>::zeros((2, 2));
        let mut scale = Array2::<f64>::ones((2, 2));
        scale[[0, 0]] = -0.1;
        assert!(ConditionalPriorIvae::new(mean, scale, 1.0).is_err());
    }

    #[test]
    fn piecewise_linear_eval_endpoints_and_midpoint() {
        let coeffs = array![[0.0_f64, 10.0], [1.0, 20.0], [2.0, 30.0]];
        let u = Array1::from(vec![0.0, 0.5, 1.0]);
        let out = piecewise_linear_eval(u.view(), coeffs.view(), 0.0, 1.0);
        assert!((out[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((out[[1, 0]] - 1.0).abs() < 1e-12);
        assert!((out[[2, 0]] - 2.0).abs() < 1e-12);
        assert!((out[[1, 1]] - 20.0).abs() < 1e-12);
    }
}
