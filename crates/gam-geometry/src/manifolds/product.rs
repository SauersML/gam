use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, quad_form,
};

pub struct ProductManifold {
    components: Vec<Box<dyn RiemannianManifold>>,
}

impl ProductManifold {
    pub fn new(components: Vec<Box<dyn RiemannianManifold>>) -> Self {
        Self { components }
    }

    pub fn components(&self) -> &[Box<dyn RiemannianManifold>] {
        &self.components
    }
}

impl RiemannianManifold for ProductManifold {
    fn dim(&self) -> usize {
        self.components.iter().map(|c| c.dim()).sum()
    }

    fn ambient_dim(&self) -> usize {
        self.components.iter().map(|c| c.ambient_dim()).sum()
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Product point", point.len(), self.ambient_dim())?;
        let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.dim()));
        let mut row_off = 0usize;
        let mut col_off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let d = component.dim();
            let q = component.tangent_basis(point.slice(s![row_off..row_off + m]))?;
            for i in 0..m {
                for j in 0..d {
                    out[[row_off + i, col_off + j]] = q[[i, j]];
                }
            }
            row_off += m;
            col_off += d;
        }
        Ok(out)
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product point", point.len(), self.ambient_dim())?;
        check_len("Product tangent", tangent_vec.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part = component.exp_map(
                point.slice(s![off..off + m]),
                tangent_vec.slice(s![off..off + m]),
            )?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        let ambient = self.ambient_dim();
        check_len("Product exp_map_vjp point", point.len(), ambient)?;
        check_len("Product exp_map_vjp tangent", tangent_vec.len(), ambient)?;
        check_len("Product exp_map_vjp grad", grad_output.len(), ambient)?;
        // exp on a product acts block-wise, so its Jacobian is block-diagonal:
        // dispatch each component's analytic VJP on its own slice. A Sphere
        // (or any curved factor) thus uses its real backward, never the flat
        // identity; a factor with no closed form propagates its error.
        let mut grad_point = Array1::<f64>::zeros(ambient);
        let mut grad_tangent = Array1::<f64>::zeros(ambient);
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let (gp, gt) = component.exp_map_vjp(
                point.slice(s![off..off + m]),
                tangent_vec.slice(s![off..off + m]),
                grad_output.slice(s![off..off + m]),
            )?;
            for i in 0..m {
                grad_point[off + i] = gp[i];
                grad_tangent[off + i] = gt[i];
            }
            off += m;
        }
        Ok((grad_point, grad_tangent))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product source", p_from.len(), self.ambient_dim())?;
        check_len("Product target", p_to.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part =
                component.log_map(p_from.slice(s![off..off + m]), p_to.slice(s![off..off + m]))?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len(
            "Product path width",
            point_along.ncols(),
            self.ambient_dim(),
        )?;
        check_len("Product transported vector", vec.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let mut path = Array2::<f64>::zeros((point_along.nrows(), m));
            for row in 0..point_along.nrows() {
                for col in 0..m {
                    path[[row, col]] = point_along[[row, off + col]];
                }
            }
            let part = component.parallel_transport(path.view(), vec.slice(s![off..off + m]))?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Product metric point", point.len(), self.ambient_dim())?;
        let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.ambient_dim()));
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let g = component.metric_tensor(point.slice(s![off..off + m]))?;
            for i in 0..m {
                for j in 0..m {
                    out[[off + i, off + j]] = g[[i, j]];
                }
            }
            off += m;
        }
        Ok(out)
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Product Christoffel point", point.len(), self.ambient_dim())?;
        let ambient = self.ambient_dim();
        let mut out = (0..ambient)
            .map(|_| Array2::<f64>::zeros((ambient, ambient)))
            .collect::<Vec<_>>();
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            // The product connection is block-diagonal: Γ assembles only from
            // factor Christoffels. If a factor cannot provide chart-valid
            // symbols (e.g. a curved embedded sphere/Grassmann/Stiefel returns
            // Unsupported), propagate that rather than silently leaving its
            // block a false flat zero — so the `?` here is deliberate.
            let gamma = component.christoffel_symbols(point.slice(s![off..off + m]))?;
            for k in 0..m {
                for i in 0..m {
                    for j in 0..m {
                        out[off + k][[off + i, off + j]] = gamma[k][[i, j]];
                    }
                }
            }
            off += m;
        }
        Ok(out)
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        check_len("Product curvature point", point.len(), self.ambient_dim())?;
        check_len(
            "Product curvature tangent u",
            tangent_pair.0.len(),
            self.ambient_dim(),
        )?;
        check_len(
            "Product curvature tangent v",
            tangent_pair.1.len(),
            self.ambient_dim(),
        )?;
        // A Riemannian product M = ∏_r M_r carries the block-diagonal product
        // metric g = ⊕_r g_r and the block-diagonal curvature tensor
        // R = ⊕_r R_r (mixed-factor components vanish). For tangent vectors
        // U = (U_r), V = (V_r) the curvature numerator and the Gram denominator
        // therefore split across factors:
        //
        //   ⟨R(U,V)V,U⟩ = Σ_r ⟨R_r(U_r,V_r)V_r,U_r⟩_r,
        //   |U|²|V|² − ⟨U,V⟩² with |·|, ⟨·,·⟩ the product metric.
        //
        // Each factor exposes its sectional curvature K_r, from which the
        // factor curvature numerator is recovered as
        //   num_r = K_r · (|U_r|²_r|V_r|²_r − ⟨U_r,V_r⟩²_r),
        // using that factor's own metric g_r (SphereManifold returns K_r = 1,
        // EuclideanManifold 0, and SpdManifold its affine-invariant value).
        // The product metric inner products are the sums of the per-factor
        // ones; the whole product's curvature is then
        //   K_M(U,V) = (Σ_r num_r) / (|U|²|V|² − ⟨U,V⟩²).
        let (u, v) = tangent_pair;
        let mut numerator = 0.0;
        let mut uu_total = 0.0;
        let mut vv_total = 0.0;
        let mut uv_total = 0.0;
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let u_r = u.slice(s![off..off + m]);
            let v_r = v.slice(s![off..off + m]);
            // Inner products under the factor's own metric g_r; this is the
            // ambient identity for Sphere/Euclidean/etc. and the
            // affine-invariant metric for SPD, so the Gram terms are computed
            // consistently with each factor's curvature definition.
            let g_r = component.metric_tensor(point.slice(s![off..off + m]))?;
            let uu_r = quad_form(g_r.view(), u_r, u_r);
            let vv_r = quad_form(g_r.view(), v_r, v_r);
            let uv_r = quad_form(g_r.view(), u_r, v_r);
            let gram_r = uu_r * vv_r - uv_r * uv_r;
            // Skip factors whose tangent pair spans no area (collinear or zero
            // within this factor): their curvature numerator is identically
            // zero, and calling the factor's `sectional_curvature` on a
            // degenerate plane may legitimately error (e.g. SPD), so a zero
            // contribution must not be allowed to abort the product as a whole.
            if gram_r > GEOMETRY_EPS {
                let k_r =
                    component.sectional_curvature(point.slice(s![off..off + m]), (u_r, v_r))?;
                numerator += k_r * gram_r;
            }
            uu_total += uu_r;
            vv_total += vv_r;
            uv_total += uv_r;
            off += m;
        }
        let denom = uu_total * vv_total - uv_total * uv_total;
        if denom <= GEOMETRY_EPS {
            return Err(GeometryError::Singular(
                "Product sectional curvature plane is degenerate",
            ));
        }
        Ok(numerator / denom)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product projection point", point.len(), self.ambient_dim())?;
        check_len("Product projection vector", vec.len(), self.ambient_dim())?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part = component
                .project_tangent(point.slice(s![off..off + m]), vec.slice(s![off..off + m]))?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }

    /// The product metric is block-diagonal across the factors, so the
    /// Riemannian gradient raises **independently within each block**: a factor
    /// with a genuine (non-identity) metric — an affine-invariant SPD or
    /// canonical Stiefel component — must use *its own* metric-raising, not a
    /// global tangent projection. Delegating per block keeps every factor's
    /// gradient first-order correct (issue #955) rather than silently applying
    /// the embedded projection across the whole product.
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        check_len("Product gradient point", point.len(), self.ambient_dim())?;
        check_len(
            "Product gradient vector",
            euclidean_grad.len(),
            self.ambient_dim(),
        )?;
        let mut out = Array1::<f64>::zeros(self.ambient_dim());
        let mut off = 0usize;
        for component in &self.components {
            let m = component.ambient_dim();
            let part = component.riemannian_gradient(
                point.slice(s![off..off + m]),
                euclidean_grad.slice(s![off..off + m]),
            )?;
            for i in 0..m {
                out[off + i] = part[i];
            }
            off += m;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::RiemannianManifold;
    use crate::manifolds::euclidean::EuclideanManifold;
    use ndarray::array;

    fn two_euclidean() -> ProductManifold {
        ProductManifold::new(vec![
            Box::new(EuclideanManifold::new(2)),
            Box::new(EuclideanManifold::new(3)),
        ])
    }

    #[test]
    fn dim_is_sum_of_component_dims() {
        assert_eq!(two_euclidean().dim(), 5);
    }

    #[test]
    fn ambient_dim_equals_dim_for_euclidean_factors() {
        assert_eq!(two_euclidean().ambient_dim(), 5);
    }

    #[test]
    fn exp_map_euclidean_product_is_componentwise_add() {
        let m = two_euclidean();
        let p = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let v = array![10.0_f64, 20.0, 30.0, 40.0, 50.0];
        let q = m.exp_map(p.view(), v.view()).unwrap();
        assert_eq!(q.len(), 5);
        for i in 0..5 {
            assert!((q[i] - (p[i] + v[i])).abs() < 1e-12, "index {i}: {}", q[i]);
        }
    }

    #[test]
    fn log_map_euclidean_product_is_componentwise_sub() {
        let m = two_euclidean();
        let p = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let q = array![4.0_f64, 2.0, 1.0, 9.0, 5.0];
        let v = m.log_map(p.view(), q.view()).unwrap();
        let expected = array![3.0_f64, 0.0, -2.0, 5.0, 0.0];
        for i in 0..5 {
            assert!((v[i] - expected[i]).abs() < 1e-12, "index {i}: {}", v[i]);
        }
    }

    #[test]
    fn metric_tensor_is_block_identity_for_euclidean_factors() {
        let m = two_euclidean();
        let p = Array1::<f64>::zeros(5);
        let g = m.metric_tensor(p.view()).unwrap();
        assert_eq!(g.dim(), (5, 5));
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((g[[i, j]] - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let m = two_euclidean();
        let p = array![1.0_f64, 2.0]; // wrong size (2 vs 5)
        let v = array![0.0_f64, 0.0];
        assert!(m.exp_map(p.view(), v.view()).is_err());
    }

    #[test]
    fn single_factor_product_behaves_like_that_manifold() {
        let m = ProductManifold::new(vec![Box::new(EuclideanManifold::new(3))]);
        assert_eq!(m.dim(), 3);
        let p = array![1.0_f64, 0.0, -1.0];
        let v = array![2.0_f64, 3.0, 4.0];
        let q = m.exp_map(p.view(), v.view()).unwrap();
        assert!((q[0] - 3.0).abs() < 1e-12);
        assert!((q[1] - 3.0).abs() < 1e-12);
        assert!((q[2] - 3.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod parallel_transport_tests {
    use super::*;
    use crate::manifold::quad_form;
    use crate::manifolds::euclidean::EuclideanManifold;
    use crate::manifolds::sphere::SphereManifold;
    use ndarray::array;

    /// A product with one curved factor (`S^2`, ambient 3) and one flat
    /// factor (`R^2`), so `parallel_transport`'s per-component splitting and
    /// re-stitching (see [`ProductManifold::parallel_transport`]) is
    /// exercised against a genuinely non-trivial transport, not the
    /// componentwise-identity case a two-Euclidean-factor fixture would
    /// give. The fixture points mirror `sphere.rs`'s own
    /// `parallel_transport_tests::fixture` for the curved half.
    fn fixture() -> (ProductManifold, Array1<f64>, Array1<f64>) {
        let product = ProductManifold::new(vec![
            Box::new(SphereManifold::new(2)),
            Box::new(EuclideanManifold::new(2)),
        ]);
        let sphere_p = array![1.0_f64, 0.0, 0.0];
        let raw = array![1.0_f64, 1.0, 1.0];
        let sphere_q = &raw / (raw.dot(&raw)).sqrt();
        let p = concat_1d(&[sphere_p.view(), array![0.5_f64, -1.0].view()]);
        let q = concat_1d(&[sphere_q.view(), array![2.0_f64, 0.5].view()]);
        (product, p, q)
    }

    fn concat_1d(parts: &[ArrayView1<'_, f64>]) -> Array1<f64> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let mut out = Array1::<f64>::zeros(total);
        let mut off = 0usize;
        for part in parts {
            for &x in part.iter() {
                out[off] = x;
                off += 1;
            }
        }
        out
    }

    fn path2(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((2, a.len()));
        out.row_mut(0).assign(a);
        out.row_mut(1).assign(b);
        out
    }

    /// Parallel transport on a product manifold is the block-diagonal
    /// concatenation of each factor's own transport, so it must still be a
    /// linear isometry of the *product* metric — `⟨Γ(U),Γ(V)⟩_Q = ⟨U,V⟩_P` —
    /// even though the two blocks have unrelated (curved vs. flat)
    /// geometries. `quad_form` against `metric_tensor` (block identity here,
    /// since both factors carry the embedded/ambient metric) is the
    /// manifold-agnostic inner product, so this genuinely checks the
    /// splitting/re-stitching in `ProductManifold::parallel_transport`
    /// rather than re-deriving each factor's own formula.
    #[test]
    fn parallel_transport_preserves_product_inner_product() {
        let (product, p, q) = fixture();
        let path = path2(&p, &q);
        // Tangent at p: sphere block orthogonal to (1,0,0) (zero first
        // coordinate), Euclidean block unconstrained.
        let u = array![0.0_f64, 1.0, 0.4, 3.0, -2.0];
        let v = array![0.0_f64, -0.3, 1.2, -1.0, 0.5];

        let tu = product
            .parallel_transport(path.view(), u.view())
            .expect("Γ(U)");
        let tv = product
            .parallel_transport(path.view(), v.view())
            .expect("Γ(V)");

        let g_p = product.metric_tensor(p.view()).expect("G(P)");
        let g_q = product.metric_tensor(q.view()).expect("G(Q)");
        let before = quad_form(g_p.view(), u.view(), v.view());
        let after = quad_form(g_q.view(), tu.view(), tv.view());
        assert!(
            (before - after).abs() <= 1e-10 * before.abs().max(1.0),
            "product parallel transport is not an isometry: ⟨U,V⟩_P={before:.12e}, ⟨ΓU,ΓV⟩_Q={after:.12e}"
        );
    }

    /// Same geodesic-velocity sign identity checked per-factor elsewhere
    /// (`sphere.rs`, `spd.rs`): `Γ_{P→Q}(log_P Q) = −log_Q P`, now across the
    /// whole product vector at once — a bug that swapped or misaligned a
    /// component's offset while splitting `point_along`/`vec` would show up
    /// here even if each factor's own transport were correct in isolation.
    #[test]
    fn parallel_transport_matches_geodesic_velocity_identity() {
        let (product, p, q) = fixture();
        let forward = path2(&p, &q);
        let v_p_to_q = product.log_map(p.view(), q.view()).expect("log_P(Q)");
        let v_q_to_p = product.log_map(q.view(), p.view()).expect("log_Q(P)");

        let transported = product
            .parallel_transport(forward.view(), v_p_to_q.view())
            .expect("Γ(log_P Q)");
        for (i, (&t, &v)) in transported.iter().zip(v_q_to_p.iter()).enumerate() {
            assert!(
                (t + v).abs() <= 1e-9 * v.abs().max(1.0),
                "component {i}: Γ(log_P Q)={t:.12e}, −log_Q P={:.12e}",
                -v
            );
        }
    }

    /// Transporting forward `P→Q` then back `Q→P` must recover the original
    /// tangent exactly, block by block.
    #[test]
    fn parallel_transport_round_trip_is_identity() {
        let (product, p, q) = fixture();
        let forward = path2(&p, &q);
        let backward = path2(&q, &p);
        let u = array![0.0_f64, 0.6, -0.2, 1.5, -0.7];

        let out = product
            .parallel_transport(forward.view(), u.view())
            .expect("Γ_{P→Q}(U)");
        let back = product
            .parallel_transport(backward.view(), out.view())
            .expect("Γ_{Q→P}(Γ_{P→Q}(U))");

        for (i, (&b, &orig)) in back.iter().zip(u.iter()).enumerate() {
            assert!(
                (b - orig).abs() <= 1e-9 * orig.abs().max(1.0),
                "component {i}: round-trip {b:.12e} vs original {orig:.12e}"
            );
        }
    }
}
