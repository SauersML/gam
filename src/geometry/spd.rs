use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, cholesky_spd, dot, flatten,
    from_flat, inverse, spectral_map_spd, spectral_map_symmetric, sym,
    tangent_basis_metric_orthonormal,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpdManifold {
    n: usize,
}

impl SpdManifold {
    /// Relative tolerance on the asymmetry `max|P_ij − P_ji|` for accepting a
    /// flattened matrix as a symmetric SPD point.
    const SYM_REL_TOL: f64 = 1.0e-9;

    pub const fn new(n: usize) -> Self {
        Self { n }
    }

    fn matrix(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        let raw = from_flat(point, self.n, self.n)?;
        // An SPD point must be symmetric. Reject a non-symmetric input rather
        // than silently replacing it with (P+Pᵀ)/2 — that would accept an
        // off-manifold matrix as a *different* valid point and quietly move the
        // base of exp/log. Only residual float asymmetry (within tolerance) is
        // then cleaned by `sym` before the positive-definiteness check.
        let mut max_abs = 0.0_f64;
        let mut max_asym = 0.0_f64;
        for i in 0..self.n {
            for j in 0..self.n {
                max_abs = max_abs.max(raw[[i, j]].abs());
                max_asym = max_asym.max((raw[[i, j]] - raw[[j, i]]).abs());
            }
        }
        if !max_asym.is_finite() || max_asym > Self::SYM_REL_TOL * max_abs.max(1.0) {
            return Err(GeometryError::InvalidPoint(
                "SPD point must be a symmetric matrix",
            ));
        }
        let p = sym(&raw);
        cholesky_spd(&p)?;
        Ok(p)
    }

    fn affine_inner(
        &self,
        p: &Array2<f64>,
        u: &Array2<f64>,
        v: &Array2<f64>,
    ) -> GeometryResult<f64> {
        use crate::linalg::faer_ndarray::fast_ab;
        let pinv = inverse(p)?;
        // Affine-invariant inner product tr(P⁻¹U P⁻¹V): a chain of dense n×n
        // products that the auto-dispatch fast_ab shim offloads to the GPU for
        // large ambient dimension (and runs on faer otherwise).
        let a = fast_ab(&fast_ab(&fast_ab(&pinv, u), &pinv), v);
        let mut trace = 0.0;
        for i in 0..self.n {
            trace += a[[i, i]];
        }
        Ok(trace)
    }
}

impl RiemannianManifold for SpdManifold {
    fn dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    fn ambient_dim(&self) -> usize {
        self.n * self.n
    }

    /// Basis of the symmetric tangent space, orthonormal under the
    /// **affine-invariant metric** `⟨U,V⟩_P = tr(P⁻¹U P⁻¹V)` (i.e. `Qᵀ W Q = I`
    /// with `W = metric_tensor(point) = P⁻¹ ⊗ P⁻¹`). The hand-rolled
    /// Frobenius-orthonormal basis used previously is orthonormal only under the
    /// embedded `tr(UV)` inner product, which is *not* the SPD metric off the
    /// identity point, so it produced a basis that did not satisfy `Qᵀ W Q = I`.
    /// We Gram–Schmidt the projected symmetric standard basis under `W` instead.
    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("SPD point", point.len(), self.ambient_dim())?;
        tangent_basis_metric_orthonormal(self, point, self.n, self.n)
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        use crate::linalg::faer_ndarray::fast_ab;
        let p = self.matrix(point)?;
        let u = sym(&from_flat(tangent_vec, self.n, self.n)?);
        let sqrt_p = spectral_map_spd(&p, |x| Ok(x.sqrt()))?;
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        // The spectral conjugations P^{±1/2} · M · P^{±1/2} are dense n×n matmul
        // chains; route them through the GPU-dispatched fast_ab shim.
        let middle = fast_ab(&fast_ab(&inv_sqrt_p, &u), &inv_sqrt_p);
        let exp_middle = spectral_map_symmetric(&middle, |x| Ok(x.exp()))?;
        Ok(flatten(&sym(&fast_ab(
            &fast_ab(&sqrt_p, &exp_middle),
            &sqrt_p,
        ))))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        use crate::linalg::faer_ndarray::fast_ab;
        let p = self.matrix(p_from)?;
        let q = self.matrix(p_to)?;
        let sqrt_p = spectral_map_spd(&p, |x| Ok(x.sqrt()))?;
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        // Dense n×n spectral conjugations, GPU-dispatched via fast_ab.
        let middle = fast_ab(&fast_ab(&inv_sqrt_p, &q), &inv_sqrt_p);
        let log_middle = spectral_map_spd(&middle, |x| Ok(x.ln()))?;
        Ok(flatten(&sym(&fast_ab(
            &fast_ab(&sqrt_p, &log_middle),
            &sqrt_p,
        ))))
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("SPD transported vector", vec.len(), self.ambient_dim())?;
        if point_along.nrows() < 2 {
            return Ok(flatten(&sym(&from_flat(vec, self.n, self.n)?)));
        }
        let p = self.matrix(point_along.row(0))?;
        let q = self.matrix(point_along.row(point_along.nrows() - 1))?;
        use crate::linalg::faer_ndarray::{fast_ab, fast_abt};
        let u = sym(&from_flat(vec, self.n, self.n)?);
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        let middle = fast_ab(&fast_ab(&inv_sqrt_p, &q), &inv_sqrt_p);
        let e = spectral_map_spd(&middle, |x| Ok(x.sqrt()))?;
        let sqrt_p = spectral_map_spd(&p, |x| Ok(x.sqrt()))?;
        // Transport operator A = P^{1/2} E P^{-1/2} and the congruence A U Aᵀ,
        // both dense n×n matmul chains GPU-dispatched via fast_ab / fast_abt.
        let a = fast_ab(&fast_ab(&sqrt_p, &e), &inv_sqrt_p);
        Ok(flatten(&sym(&fast_abt(&fast_ab(&a, &u), &a))))
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        let p = self.matrix(point)?;
        let pinv = inverse(&p)?;
        let ambient = self.ambient_dim();
        let mut g = Array2::<f64>::zeros((ambient, ambient));
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    for l in 0..self.n {
                        g[[i * self.n + j, k * self.n + l]] = pinv[[i, k]] * pinv[[l, j]];
                    }
                }
            }
        }
        Ok(g)
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        let p = self.matrix(point)?;
        let pinv = inverse(&p)?;
        let ambient = self.ambient_dim();
        let mut gamma = (0..ambient)
            .map(|_| Array2::<f64>::zeros((ambient, ambient)))
            .collect::<Vec<_>>();
        for a in 0..ambient {
            let ai = a / self.n;
            let aj = a % self.n;
            for b in 0..ambient {
                let bi = b / self.n;
                let bj = b % self.n;
                let mut u = Array2::<f64>::zeros((self.n, self.n));
                let mut v = Array2::<f64>::zeros((self.n, self.n));
                u[[ai, aj]] = 1.0;
                v[[bi, bj]] = 1.0;
                let c = -0.5 * (u.dot(&pinv).dot(&v) + v.dot(&pinv).dot(&u));
                for r in 0..self.n {
                    for s in 0..self.n {
                        gamma[r * self.n + s][[a, b]] = c[[r, s]];
                    }
                }
            }
        }
        Ok(gamma)
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        let p = self.matrix(point)?;
        let u = sym(&from_flat(tangent_pair.0, self.n, self.n)?);
        let v = sym(&from_flat(tangent_pair.1, self.n, self.n)?);
        use crate::linalg::faer_ndarray::fast_ab;
        let inv_sqrt_p = spectral_map_spd(&p, |x| Ok(1.0 / x.sqrt()))?;
        // Whitened tangents Ã = P^{-1/2} U P^{-1/2} and their commutator [Ã,B̃]:
        // dense n×n matmul chains GPU-dispatched via fast_ab.
        let a = fast_ab(&fast_ab(&inv_sqrt_p, &u), &inv_sqrt_p);
        let b = fast_ab(&fast_ab(&inv_sqrt_p, &v), &inv_sqrt_p);
        let comm = &fast_ab(&a, &b) - &fast_ab(&b, &a);
        let comm_norm = dot(flatten(&comm).view(), flatten(&comm).view());
        let uu = self.affine_inner(&p, &u, &u)?;
        let vv = self.affine_inner(&p, &v, &v)?;
        let uv = self.affine_inner(&p, &u, &v)?;
        let denom = uu * vv - uv * uv;
        if denom.abs() <= 1.0e-14 {
            return Err(GeometryError::Singular(
                "SPD sectional curvature plane is degenerate",
            ));
        }
        Ok(-0.25 * comm_norm / denom)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("SPD projection point", point.len(), self.ambient_dim())?;
        Ok(flatten(&sym(&from_flat(vec, self.n, self.n)?)))
    }

    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        let m = self.ambient_dim();
        check_len("SPD exp_map_vjp point", point.len(), m)?;
        check_len("SPD exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("SPD exp_map_vjp grad", grad_output.len(), m)?;
        // The affine-invariant SPD exponential VJP requires differentiating
        // the symmetric matrix exponential / Fréchet derivative; no closed
        // form is wired up. Refuse rather than inherit the identity default.
        Err(GeometryError::Unsupported(
            "SPD exp_map_vjp: no analytic backward implemented",
        ))
    }
}

#[cfg(test)]
mod tangent_basis_tests {
    use super::SpdManifold;
    use crate::geometry::manifold::RiemannianManifold;
    use ndarray::Array1;

    /// The SPD `tangent_basis` must be orthonormal under the affine-invariant
    /// metric `⟨U,V⟩_P = tr(P⁻¹U P⁻¹V)`, i.e. `Qᵀ W Q = I` with
    /// `W = metric_tensor(P)`. At a non-identity point the old hand-rolled
    /// Frobenius-orthonormal basis fails this; the metric Gram–Schmidt fixes it.
    #[test]
    fn spd_tangent_basis_metric_orthonormal() {
        let spd = SpdManifold::new(2);
        // P = [[2, 0.5], [0.5, 1]] (SPD), row-major flatten.
        let p = Array1::from(vec![2.0, 0.5, 0.5, 1.0]);
        let q = spd.tangent_basis(p.view()).expect("tangent basis");
        let w = spd.metric_tensor(p.view()).expect("metric tensor");
        let d = spd.dim();
        assert_eq!(q.ncols(), d, "basis must have dim() columns");
        let wq = w.dot(&q);
        let gram = q.t().dot(&wq);
        for i in 0..d {
            for j in 0..d {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[[i, j]] - want).abs() <= 1.0e-10,
                    "QᵀWQ != I at ({i},{j}): got {}",
                    gram[[i, j]]
                );
            }
        }
    }
}
