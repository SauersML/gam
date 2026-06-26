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
