//! A product of Stiefel manifolds `St(n₁, k₁) × … × St(n_B, k_B)` under the **embedded**
//! metric, with the **polar** retraction: the geometry of orthonormal frames learned at
//! scale, where each block is a matrix with orthonormal columns stored row-major.
//!
//! [`StiefelManifold`](crate::StiefelManifold) carries the canonical metric and the QR
//! retraction, which is first order, so a trust region on it drops its Hessian (#956). Here:
//!
//! * the metric is the ambient Frobenius one, so the Riemannian gradient is the tangent
//!   projection `P_X(E) = E − X·sym(XᵀE)`;
//! * the retraction is the polar factor `R_X(ξ) = (X + ξ)((X + ξ)ᵀ(X + ξ))^{-1/2}`, the
//!   metric projection of `X + ξ` onto the manifold and therefore second order: the
//!   trust region's quadratic model with the Riemannian Hessian is a valid second-order
//!   model along it. For a tangent `ξ`, `(X + ξ)ᵀ(X + ξ) = I + ξᵀξ` has every eigenvalue
//!   at least one, so the inverse square root is always well conditioned;
//! * the Riemannian Hessian comes from the objective's Euclidean derivatives as
//!   `Hess f(X)[ξ] = P_X(∇²f(X)[ξ] − ξ·sym(Xᵀ∇f(X)))`
//!   ([`RiemannianManifold::riemannian_hessian`]).
//!
//! The blocks' dimensions can be in the thousands, so nothing here forms an ambient-sized
//! matrix: the exponential and logarithm maps, parallel transport, a dense tangent basis and
//! curvature are refused as unsupported (no solver here needs them); every other operation
//! costs `O(n k²)` per block. Blocks are read as views of the flat vectors, never copied, and
//! every product is a CPU GEMM: these run inside a trust region whose objective already
//! occupies the device, and shipping `3072 × 3072` blocks across for each product measured
//! as most of a support fit's wall time.
use faer::linalg::matmul::matmul;
use faer::{Accum, Side};
use gam_linalg::faer_ndarray::{FaerArrayView, FaerEigh, array2_to_matmut, matmul_parallelism};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::manifold::{GeometryError, GeometryResult, RiemannianManifold, check_len, identity};


/// `St(n₁, k₁) × …` under the embedded metric with the polar retraction (module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiefelFrames {
    /// `(n, k)` per block: `n` rows, `k ≤ n` orthonormal columns.
    blocks: Vec<(usize, usize)>,
}

impl StiefelFrames {
    /// Blocks in the order their row-major entries are concatenated.
    pub fn new(blocks: Vec<(usize, usize)>) -> GeometryResult<Self> {
        if blocks.is_empty() || blocks.iter().any(|&(n, k)| k == 0 || k > n) {
            return Err(GeometryError::InvalidPoint("StiefelFrames requires blocks with 1 <= k <= n"));
        }
        Ok(Self { blocks })
    }

    pub fn blocks(&self) -> &[(usize, usize)] {
        &self.blocks
    }

    /// Apply `f` to every block of the given same-shaped ambient vectors (as `n × k` views,
    /// no copy), writing its `n × k` result into the output's block.
    fn per_block<F>(&self, parts: &[ArrayView1<'_, f64>], mut f: F) -> GeometryResult<Array1<f64>>
    where
        F: FnMut(&[ArrayView2<'_, f64>]) -> GeometryResult<Array2<f64>>,
    {
        let ambient = self.ambient_dim();
        for part in parts {
            check_len("StiefelFrames vector", part.len(), ambient)?;
        }
        let mut out = Array1::<f64>::zeros(ambient);
        let mut offset = 0;
        // A part that is not contiguous (a strided view) is compacted once for all blocks.
        let owned: Vec<Option<Array1<f64>>> =
            parts.iter().map(|p| if p.as_slice().is_some() { None } else { Some(p.to_owned()) }).collect();
        let flat: Vec<ArrayView1<'_, f64>> =
            parts.iter().zip(&owned).map(|(p, o)| o.as_ref().map_or(p.view(), |o| o.view())).collect();
        for &(n, k) in &self.blocks {
            let size = n * k;
            let mats = flat
                .iter()
                .map(|p| {
                    p.slice(s![offset..offset + size])
                        .into_shape_with_order((n, k))
                        .map_err(|_| GeometryError::InvalidPoint("StiefelFrames: a block is not contiguous"))
                })
                .collect::<GeometryResult<Vec<_>>>()?;
            let result = f(&mats)?;
            out.slice_mut(s![offset..offset + size])
                .assign(&result.into_shape_with_order(size).map_err(|_| GeometryError::InvalidPoint("StiefelFrames: result layout"))?);
            offset += size;
        }
        Ok(out)
    }
}

/// `alpha · op(A) · op(B)` accumulated into `out` (`Accum::Add`) or replacing it, as one CPU GEMM
/// on views (no copy of either operand).
fn gemm(out: &mut Array2<f64>, accumulate: bool, alpha: f64, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) {
    let (m, k) = a.dim();
    let n = b.ncols();
    let (va, vb) = (FaerArrayView::new(&a), FaerArrayView::new(&b));
    let accum = if accumulate { Accum::Add } else { Accum::Replace };
    matmul(array2_to_matmut(out), accum, va.as_ref(), vb.as_ref(), alpha, matmul_parallelism(m, n, k));
}

/// `sym(XᵀZ)`, `k × k`.
fn sym_xtz(x: ArrayView2<'_, f64>, z: ArrayView2<'_, f64>) -> Array2<f64> {
    let k = x.ncols();
    let mut xtz = Array2::<f64>::zeros((k, k));
    gemm(&mut xtz, false, 1.0, x.t(), z);
    let transposed = xtz.t().to_owned();
    (xtz + transposed) * 0.5
}

/// `Z − X·sym(XᵀZ)`: two GEMMs, the second accumulating into a copy of `Z`.
fn project(x: ArrayView2<'_, f64>, z: ArrayView2<'_, f64>) -> Array2<f64> {
    let s = sym_xtz(x, z);
    let mut out = z.to_owned();
    gemm(&mut out, true, -1.0, x, s.view());
    out
}

impl RiemannianManifold for StiefelFrames {
    fn dim(&self) -> usize {
        self.blocks.iter().map(|&(n, k)| n * k - k * (k + 1) / 2).sum()
    }

    fn ambient_dim(&self) -> usize {
        self.blocks.iter().map(|&(n, k)| n * k).sum()
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("StiefelFrames point", point.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported("StiefelFrames: no dense tangent basis (ambient × dim doubles)"))
    }

    fn exp_map(&self, point: ArrayView1<'_, f64>, tangent_vec: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        check_len("StiefelFrames point", point.len(), self.ambient_dim())?;
        check_len("StiefelFrames tangent", tangent_vec.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported(
            "StiefelFrames: the embedded-metric exponential needs a 2k × 2k matrix exponential per block; use retract",
        ))
    }

    fn log_map(&self, p_from: ArrayView1<'_, f64>, p_to: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        check_len("StiefelFrames source", p_from.len(), self.ambient_dim())?;
        check_len("StiefelFrames target", p_to.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported("StiefelFrames: no logarithm map"))
    }

    fn parallel_transport(&self, point_along: ArrayView2<'_, f64>, vec: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        if point_along.nrows() > 0 {
            check_len("StiefelFrames path width", point_along.ncols(), self.ambient_dim())?;
        }
        check_len("StiefelFrames transported vector", vec.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported("StiefelFrames: no parallel transport"))
    }

    /// The embedded metric is the ambient identity.
    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("StiefelFrames metric point", point.len(), self.ambient_dim())?;
        Ok(identity(self.ambient_dim()))
    }

    fn metric_product(&self, point: ArrayView1<'_, f64>, tangent: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        check_len("StiefelFrames metric point", point.len(), self.ambient_dim())?;
        check_len("StiefelFrames metric tangent", tangent.len(), self.ambient_dim())?;
        Ok(tangent.to_owned())
    }

    fn sectional_curvature(&self, point: ArrayView1<'_, f64>, tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>)) -> GeometryResult<f64> {
        check_len("StiefelFrames curvature point", point.len(), self.ambient_dim())?;
        check_len("StiefelFrames curvature tangent u", tangent_pair.0.len(), self.ambient_dim())?;
        check_len("StiefelFrames curvature tangent v", tangent_pair.1.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported("StiefelFrames: no sectional curvature"))
    }

    fn project_tangent(&self, point: ArrayView1<'_, f64>, vec: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        self.per_block(&[point, vec], |m| Ok(project(m[0], m[1])))
    }

    /// Under the embedded metric the Riesz representative is the tangent projection.
    fn riemannian_gradient(&self, point: ArrayView1<'_, f64>, euclidean_grad: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        self.project_tangent(point, euclidean_grad)
    }

    /// The polar factor of `X + P_X(ξ)` per block (module docs).
    fn retract(&self, point: ArrayView1<'_, f64>, tangent_vec: ArrayView1<'_, f64>) -> GeometryResult<Array1<f64>> {
        self.per_block(&[point, tangent_vec], |m| {
            let y = &m[0] + &project(m[0], m[1]);
            let gram = sym_xtz(y.view(), y.view());
            let (values, vectors) = gram
                .eigh(Side::Lower)
                .map_err(|_| GeometryError::InvalidPoint("StiefelFrames retraction: eigendecomposition failed"))?;
            if values.iter().any(|&v| !(v > 0.0)) {
                return Err(GeometryError::Singular("StiefelFrames retraction: X + ξ lost rank"));
            }
            let scaled = &vectors * &values.mapv(|v| 1.0 / v.sqrt());
            let k = values.len();
            let mut inverse_root = Array2::<f64>::zeros((k, k));
            gemm(&mut inverse_root, false, 1.0, scaled.view(), vectors.t());
            let mut out = Array2::<f64>::zeros(y.dim());
            gemm(&mut out, false, 1.0, y.view(), inverse_root.view());
            Ok(out)
        })
    }

    fn retraction_is_second_order(&self) -> bool {
        true
    }

    fn exp_map_vjp(&self, point: ArrayView1<'_, f64>, tangent_vec: ArrayView1<'_, f64>, grad_output: ArrayView1<'_, f64>) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        check_len("StiefelFrames exp_map_vjp point", point.len(), self.ambient_dim())?;
        check_len("StiefelFrames exp_map_vjp tangent", tangent_vec.len(), self.ambient_dim())?;
        check_len("StiefelFrames exp_map_vjp grad_output", grad_output.len(), self.ambient_dim())?;
        Err(GeometryError::Unsupported("StiefelFrames: no exponential map to differentiate"))
    }

    /// `P_X(∇²f[ξ] − ξ·sym(Xᵀ∇f))` per block: the Euclidean Hessian's tangent part plus the
    /// Weingarten term of the embedding.
    fn riemannian_hessian(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
        euclidean_hessian_product: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.per_block(&[point, euclidean_grad, euclidean_hessian_product, tangent], |m| {
            let (x, g, h, xi) = (m[0], m[1], m[2], m[3]);
            // h − ξ·sym(Xᵀg), then its tangent part
            let mut inner = h.to_owned();
            gemm(&mut inner, true, -1.0, xi, sym_xtz(x, g).view());
            Ok(project(x, inner.view()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{flatten, from_flat};
    use crate::optimizer::{RiemannianObjective, RiemannianTrustRegion};
    use gam_linalg::faer_ndarray::{fast_ab, fast_atb};

    /// A deterministic `n × k` matrix with orthonormal columns and a symmetric `A`.
    fn fixture(n: usize, k: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let raw = Array2::from_shape_fn((n, k), |_| next());
        let gram = fast_atb(&raw, &raw);
        let (values, vectors) = gram.eigh(Side::Lower).unwrap();
        let x = fast_ab(&raw, &fast_ab(&(&vectors * &values.mapv(|v| 1.0 / v.sqrt())), &vectors.t().to_owned()));
        let a = Array2::from_shape_fn((n, n), |_| next());
        (x, (&a + &a.t()) * 0.5)
    }

    #[test]
    fn a_retraction_stays_on_the_manifold() {
        let (x, a) = fixture(7, 3, 1);
        let m = StiefelFrames::new(vec![(7, 3)]).unwrap();
        let xi = m.project_tangent(flatten(&x).view(), flatten(&a.slice(s![.., 0..3]).to_owned()).view()).unwrap();
        let r = from_flat(m.retract(flatten(&x).view(), xi.view()).unwrap().view(), 7, 3).unwrap();
        let err = (&fast_atb(&r, &r) - &identity(3)).mapv(f64::abs).fold(0.0_f64, |a, &b| a.max(b));
        assert!(err < 1e-13, "{err}");
    }

    /// Second order: `R_X(tξ) − X − tξ` is normal to the manifold up to `O(t³)`, so its
    /// tangent part shrinks a thousandfold when `t` shrinks tenfold.
    #[test]
    fn the_polar_retraction_is_second_order() {
        let (x, a) = fixture(8, 3, 2);
        let m = StiefelFrames::new(vec![(8, 3)]).unwrap();
        let xi = m.project_tangent(flatten(&x).view(), flatten(&a.slice(s![.., 0..3]).to_owned()).view()).unwrap();
        let tangent_error = |t: f64| {
            let step = xi.mapv(|v| t * v);
            let r = m.retract(flatten(&x).view(), step.view()).unwrap();
            let residual = &(&r - &flatten(&x)) - &step;
            let tangent = m.project_tangent(flatten(&x).view(), residual.view()).unwrap();
            tangent.dot(&tangent).sqrt()
        };
        let ratio = tangent_error(1e-2) / tangent_error(1e-3);
        assert!((ratio / 1e3 - 1.0).abs() < 0.05, "{ratio}");
    }

    /// For `f(X) = ½ tr(XᵀAX)` the second derivative along the polar retraction is exactly
    /// `tr(ξᵀAξ) − tr(XᵀAX ξᵀξ)` (expand `(I + t²ξᵀξ)^{-1/2}`), which a second-order
    /// retraction equates with `⟨Hess f[ξ], ξ⟩`.
    #[test]
    fn the_riemannian_hessian_matches_the_second_derivative_along_the_retraction() {
        let (x, a) = fixture(9, 4, 3);
        let m = StiefelFrames::new(vec![(9, 4)]).unwrap();
        let xi = from_flat(
            m.project_tangent(flatten(&x).view(), flatten(&a.slice(s![.., 1..5]).to_owned()).view()).unwrap().view(),
            9,
            4,
        )
        .unwrap();
        let g = fast_ab(&a, &x);
        let h = fast_ab(&a, &xi);
        let hess = m
            .riemannian_hessian(flatten(&x).view(), flatten(&g).view(), flatten(&h).view(), flatten(&xi).view())
            .unwrap();
        let along = hess.dot(&flatten(&xi));
        let trace = |m: &Array2<f64>| (0..m.nrows()).map(|i| m[[i, i]]).sum::<f64>();
        let exact = trace(&fast_atb(&xi, &h)) - trace(&fast_ab(&fast_atb(&x, &g), &fast_atb(&xi, &xi)));
        assert!((along - exact).abs() < 1e-12 * exact.abs().max(1.0), "{along} vs {exact}");
    }

    /// The trust region with this geometry and the exact Riemannian Hessian finds
    /// `min ½ tr(XᵀAX)` over two blocks: half the sum of each `A`'s smallest `k` eigenvalues.
    #[test]
    fn the_trust_region_reaches_the_brockett_minimum_on_two_blocks() {
        let (x1, a1) = fixture(6, 2, 4);
        let (x2, a2) = fixture(5, 3, 5);
        let m = StiefelFrames::new(vec![(6, 2), (5, 3)]).unwrap();
        struct Brockett<'a> {
            m: &'a StiefelFrames,
            a: Vec<Array2<f64>>,
        }
        impl Brockett<'_> {
            fn split(&self, v: ArrayView1<'_, f64>) -> Vec<Array2<f64>> {
                let mut out = Vec::new();
                let mut offset = 0;
                for &(n, k) in self.m.blocks() {
                    out.push(from_flat(v.slice(s![offset..offset + n * k]), n, k).unwrap());
                    offset += n * k;
                }
                out
            }
            fn join(parts: &[Array2<f64>]) -> Array1<f64> {
                Array1::from_iter(parts.iter().flat_map(|p| flatten(p).to_vec()))
            }
        }
        impl RiemannianObjective for Brockett<'_> {
            fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)> {
                let xs = self.split(point);
                let grads: Vec<_> = xs.iter().zip(&self.a).map(|(x, a)| fast_ab(a, x)).collect();
                let value = xs.iter().zip(&grads).map(|(x, g)| 0.5 * (x * g).sum()).sum();
                Ok((value, Self::join(&grads)))
            }
            fn hessian_vector_product(&mut self, point: ArrayView1<'_, f64>, tangent: ArrayView1<'_, f64>) -> GeometryResult<Option<Array1<f64>>> {
                let xs = self.split(point);
                let grad = Self::join(&xs.iter().zip(&self.a).map(|(x, a)| fast_ab(a, x)).collect::<Vec<_>>());
                let euclidean = Self::join(&self.split(tangent).iter().zip(&self.a).map(|(t, a)| fast_ab(a, t)).collect::<Vec<_>>());
                Ok(Some(self.m.riemannian_hessian(point, grad.view(), euclidean.view(), tangent)?))
            }
        }
        let mut objective = Brockett { m: &m, a: vec![a1.clone(), a2.clone()] };
        let start = Brockett::join(&[x1, x2]);
        // The trust region compares values, so in float64 it resolves a minimizer to a relative
        // gradient of about √u ≈ 1e-8 (it reached 3e-9 here); certify well inside that.
        let solver = RiemannianTrustRegion { radius: 0.5, max_iter: 200, grad_tol: 1e-6, ..RiemannianTrustRegion::default() };
        let result = solver.minimize_reporting_termination(&m, &mut objective, start.view()).unwrap();
        assert!(result.residual <= result.tolerance, "{} > {}", result.residual, result.tolerance);
        let smallest = |a: &Array2<f64>, k: usize| {
            let (mut values, _) = a.eigh(Side::Lower).unwrap();
            values.as_slice_mut().unwrap().sort_by(f64::total_cmp);
            0.5 * values.iter().take(k).sum::<f64>()
        };
        let (value, _) = objective.value_gradient(result.point.view()).unwrap();
        let expected = smallest(&a1, 2) + smallest(&a2, 3);
        assert!((value - expected).abs() < 1e-9, "{value} vs {expected}");
    }
}
