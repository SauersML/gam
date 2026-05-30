use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, dot, flatten,
    from_flat, identity, inverse, jacobi_symmetric, qr_thin, zero_christoffel,
};
use crate::geometry::sphere::SphereManifold;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrassmannManifold {
    k: usize,
    n: usize,
}

impl GrassmannManifold {
    /// Construct the Grassmannian `Gr(k, n)`, the set of `k`-dimensional
    /// subspaces of `ℝⁿ`. This object exists only for `1 ≤ k ≤ n`: with
    /// `k > n` there is no `k`-dimensional subspace of `ℝⁿ`, the dimension
    /// `k(n − k)` would be negative (and `n − k` underflows in `usize`), and
    /// the QR orthonormalization cannot produce a rank-`k` basis. The domain is
    /// rejected here, before any dimension, projection, exponential, or
    /// curvature computation can run on a nonexistent manifold.
    pub fn new(k: usize, n: usize) -> GeometryResult<Self> {
        if k == 0 || n == 0 || k > n {
            return Err(GeometryError::InvalidPoint(
                "Grassmann Gr(k, n) requires 1 <= k <= n",
            ));
        }
        Ok(Self { k, n })
    }

    fn orthonormalize(&self, y: &Array2<f64>) -> Array2<f64> {
        let (q, _) = qr_thin(y);
        q
    }

    /// For `k == 1` the Grassmannian `Gr(1, n)` is real projective space
    /// `ℝP^{n-1}`, whose orientation double cover is the unit sphere
    /// `S^{n-1}` (a single unit column is a point of the sphere, and the flat
    /// ambient coordinates coincide). Within the injectivity radius `π/2` the
    /// two share the same geodesics, exponential, logarithm, parallel
    /// transport, and (constant `+1`) sectional curvature, so we reuse the
    /// [`SphereManifold`] formulas — exactly as `St(n, 1)` does in
    /// [`StiefelManifold`](crate::geometry::stiefel::StiefelManifold). This is
    /// essential at the principal-angle-`π/2` cut-locus boundary, where the
    /// `(YᵀZ)⁻¹` form used by the general-`k` `log_map` is singular but the
    /// sphere logarithm (denominator `1 + Y·Z`) is well defined, so e.g.
    /// transporting `e₂` from `e₁` to `e₂` correctly yields `-e₁` instead of
    /// failing.
    fn as_sphere(&self) -> Option<SphereManifold> {
        (self.k == 1).then(|| SphereManifold::new(self.n - 1))
    }

    fn compact_svd_from_tangent(
        &self,
        tangent: &Array2<f64>,
    ) -> GeometryResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let gram = tangent.t().dot(tangent);
        let (evals, v) = jacobi_symmetric(&gram)?;
        let mut sigma = Array1::<f64>::zeros(self.k);
        let mut u = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            sigma[j] = evals[j].max(0.0).sqrt();
            if sigma[j] > GEOMETRY_EPS {
                let col = tangent.dot(&v.column(j).to_owned()) / sigma[j];
                for i in 0..self.n {
                    u[[i, j]] = col[i];
                }
            }
        }
        Ok((u, sigma, v))
    }
}

impl RiemannianManifold for GrassmannManifold {
    fn dim(&self) -> usize {
        self.k * (self.n - self.k)
    }

    fn ambient_dim(&self) -> usize {
        self.n * self.k
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        from_flat(point, self.n, self.k).map(|_| ())?;
        let mut columns: Vec<Array1<f64>> = Vec::with_capacity(self.dim());
        for col in 0..self.k {
            for row in 0..self.n {
                let mut e = Array2::<f64>::zeros((self.n, self.k));
                e[[row, col]] = 1.0;
                let p = self.project_tangent(point, flatten(&e).view())?;
                let mut v = p;
                for q in &columns {
                    let proj = dot(q.view(), v.view());
                    v -= &(q * proj);
                }
                let nrm = dot(v.view(), v.view()).sqrt();
                if nrm > 1.0e-10 {
                    columns.push(v / nrm);
                }
                if columns.len() == self.dim() {
                    let mut out = Array2::<f64>::zeros((self.ambient_dim(), self.dim()));
                    for j in 0..columns.len() {
                        for i in 0..self.ambient_dim() {
                            out[[i, j]] = columns[j][i];
                        }
                    }
                    return Ok(out);
                }
            }
        }
        Ok(Array2::<f64>::zeros((self.ambient_dim(), columns.len())))
    }

    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.exp_map(point, tangent_vec);
        }
        let y = from_flat(point, self.n, self.k)?;
        let tangent = from_flat(
            self.project_tangent(point, tangent_vec)?.view(),
            self.n,
            self.k,
        )?;
        let (u, sigma, v) = self.compact_svd_from_tangent(&tangent)?;
        let mut cos_d = Array2::<f64>::zeros((self.k, self.k));
        let mut sin_d = Array2::<f64>::zeros((self.k, self.k));
        for i in 0..self.k {
            cos_d[[i, i]] = sigma[i].cos();
            sin_d[[i, i]] = sigma[i].sin();
        }
        let next = y.dot(&v).dot(&cos_d).dot(&v.t()) + u.dot(&sin_d).dot(&v.t());
        Ok(flatten(&self.orthonormalize(&next)))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.log_map(p_from, p_to);
        }
        let y = from_flat(p_from, self.n, self.k)?;
        let z = from_flat(p_to, self.n, self.k)?;
        let yt_z = y.t().dot(&z);
        let inv = inverse(&yt_z)?;
        let normal = z - y.dot(&yt_z);
        let m = normal.dot(&inv);
        let gram = m.t().dot(&m);
        let (evals, v) = jacobi_symmetric(&gram)?;
        let mut sigma = Array1::<f64>::zeros(self.k);
        let mut u = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            let tan_sigma = evals[j].max(0.0).sqrt();
            sigma[j] = tan_sigma.atan();
            if tan_sigma > GEOMETRY_EPS {
                let col = m.dot(&v.column(j).to_owned()) / tan_sigma;
                for i in 0..self.n {
                    u[[i, j]] = col[i];
                }
            }
        }
        let mut diag = Array2::<f64>::zeros((self.k, self.k));
        for i in 0..self.k {
            diag[[i, i]] = sigma[i];
        }
        Ok(flatten(&u.dot(&diag).dot(&v.t())))
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.parallel_transport(point_along, vec);
        }
        check_len("Grassmann path width", point_along.ncols(), self.ambient_dim())?;
        check_len(
            "Grassmann transported vector",
            vec.len(),
            self.ambient_dim(),
        )?;
        if point_along.nrows() == 0 {
            return Ok(vec.to_owned());
        }
        if point_along.nrows() == 1 {
            // A degenerate one-point path is the identity geodesic; the vector
            // stays in the tangent space at that single point.
            return self.project_tangent(point_along.row(0), vec);
        }
        // Levi-Civita parallel transport along the canonical Grassmann geodesic
        // from `from` to `to`. Endpoint projection (the previous implementation)
        // is *not* parallel transport: it can collapse the norm to zero (e.g.
        // transporting e₂ from e₁ to e₂ in Gr(1,n) projects e₂ - e₂(e₂ᵀe₂) = 0;
        // that k=1 case is handled by the `as_sphere` delegation above, whose
        // `1 + Y·Z` denominator stays well defined at the π/2 cut locus).
        //
        // The geodesic is determined by its initial direction Δ = Log_Y(Z),
        // whose thin SVD Δ = U Σ Vᵀ (U: n×k orthonormal, Σ: k×k diagonal of
        // principal angles, V: k×k orthogonal) gives the closed-form transport
        // operator of Edelman–Arias–Smith (1998, eq. 2.66) at unit time:
        //
        //   τ(H) = ( -Y V sin(Σ) Uᵀ + U cos(Σ) Uᵀ + (I - U Uᵀ) ) H,
        //
        // which preserves the canonical (Frobenius) inner product and maps the
        // horizontal tangent space at Y to the horizontal tangent space at Z.
        let from = point_along.row(0);
        let to = point_along.row(point_along.nrows() - 1);
        let y = from_flat(from, self.n, self.k)?;
        let direction = from_flat(self.log_map(from, to)?.view(), self.n, self.k)?;
        let (u, sigma, v) = self.compact_svd_from_tangent(&direction)?;
        let h = from_flat(self.project_tangent(from, vec)?.view(), self.n, self.k)?;

        let mut cos_d = Array2::<f64>::zeros((self.k, self.k));
        let mut sin_d = Array2::<f64>::zeros((self.k, self.k));
        for i in 0..self.k {
            cos_d[[i, i]] = sigma[i].cos();
            sin_d[[i, i]] = sigma[i].sin();
        }
        // Coordinates of H in the U-frame: ut_h = Uᵀ H (k×k).
        let ut_h = u.t().dot(&h);
        // Geodesic-aligned components: -Y V sin(Σ) Uᵀ H + U cos(Σ) Uᵀ H.
        let aligned = u.dot(&cos_d).dot(&ut_h) - y.dot(&v).dot(&sin_d).dot(&ut_h);
        // Component of H orthogonal to the geodesic 2-plane: (I - U Uᵀ) H.
        let orthogonal = &h - &u.dot(&ut_h);
        Ok(flatten(&(aligned + orthogonal)))
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Grassmann metric point", point.len(), self.ambient_dim())?;
        Ok(identity(self.ambient_dim()))
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len(
            "Grassmann Christoffel point",
            point.len(),
            self.ambient_dim(),
        )?;
        Ok(zero_christoffel(self.ambient_dim()))
    }

    fn sectional_curvature(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
    ) -> GeometryResult<f64> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.sectional_curvature(point, tangent_pair);
        }
        check_len("Grassmann curvature point", point.len(), self.ambient_dim())?;
        check_len(
            "Grassmann curvature tangent u",
            tangent_pair.0.len(),
            self.ambient_dim(),
        )?;
        check_len(
            "Grassmann curvature tangent v",
            tangent_pair.1.len(),
            self.ambient_dim(),
        )?;
        // Grassmann sectional curvature for the canonical (Frobenius) metric.
        // Gr(k,n) = O(n)/(O(k)×O(n-k)) is a symmetric space, so the curvature
        // of horizontal tangents X, Y (PᵀX = PᵀY = 0, viewed as n×k matrices)
        // is R(X,Y)Z = -[[Ω(X),Ω(Y)],Ω(Z)] in the embedding into 𝔬(n). Working
        // out the brackets gives, with the Gram matrices Gxx=XᵀX, Gyy=YᵀY,
        // Gxy=XᵀY, Gyx=YᵀX,
        //
        //   ⟨R(X,Y)Y, X⟩ = tr(Gxx·Gyy) + ‖Gxy‖²_F - 2·tr(Gyx·Gyx),
        //
        // and the sectional curvature divides by the area of the 2-plane,
        // ⟨X,X⟩⟨Y,Y⟩ - ⟨X,Y⟩² with ⟨·,·⟩ = tr(·ᵀ·). This expression matches the
        // projector-model curvature tensor R(a,b)c = [[a,b],c] (verified against
        // geomstats across Gr(2,4), Gr(2,5), Gr(3,7)); for Gr(2,4) it ranges over
        // [0, 2] as expected, so the manifold is not constant-curvature for k ≥ 2.
        // The previous constant 0.0 is only correct for a flat manifold, which
        // Grassmannians are not. The k = 1 case (Gr(1,n) = ℝP^{n-1}, constant
        // sectional curvature +1) is delegated to `as_sphere` above.
        let x = from_flat(self.project_tangent(point, tangent_pair.0)?.view(), self.n, self.k)?;
        let y = from_flat(self.project_tangent(point, tangent_pair.1)?.view(), self.n, self.k)?;
        let gxx = x.t().dot(&x);
        let gyy = y.t().dot(&y);
        let gxy = x.t().dot(&y);
        let gyx = y.t().dot(&x);
        let trace_product = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            let mut acc = 0.0;
            for i in 0..self.k {
                for j in 0..self.k {
                    acc += a[[i, j]] * b[[j, i]];
                }
            }
            acc
        };
        let frob_sq = |a: &Array2<f64>| -> f64 {
            let mut acc = 0.0;
            for value in a.iter() {
                acc += value * value;
            }
            acc
        };
        let numerator = trace_product(&gxx, &gyy) + frob_sq(&gxy) - 2.0 * trace_product(&gyx, &gyx);
        // Frobenius inner products: tr(Gxx) = ⟨X,X⟩, tr(Gyy) = ⟨Y,Y⟩,
        // tr(Gxy) = ⟨X,Y⟩.
        let trace = |a: &Array2<f64>| -> f64 {
            let mut acc = 0.0;
            for i in 0..self.k {
                acc += a[[i, i]];
            }
            acc
        };
        let xx = trace(&gxx);
        let yy = trace(&gyy);
        let xy = trace(&gxy);
        let denom = xx * yy - xy * xy;
        if denom.abs() <= 1.0e-14 {
            return Err(GeometryError::Singular(
                "Grassmann sectional curvature plane is degenerate",
            ));
        }
        Ok(numerator / denom)
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(vec, self.n, self.k)?;
        let projected = &z - y.dot(&y.t().dot(&z));
        Ok(flatten(&projected))
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let tangent = from_flat(
            self.project_tangent(point, tangent_vec)?.view(),
            self.n,
            self.k,
        )?;
        Ok(flatten(&self.orthonormalize(&(y + tangent))))
    }

    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.exp_map_vjp(point, tangent_vec, grad_output);
        }
        let m = self.ambient_dim();
        check_len("Grassmann exp_map_vjp point", point.len(), m)?;
        check_len("Grassmann exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("Grassmann exp_map_vjp grad", grad_output.len(), m)?;
        // The Grassmann geodesic VJP requires the SVD-Jacobi-field
        // differential; no closed form is wired up. Refuse rather than
        // inherit the flat identity default, which would be silently wrong.
        Err(GeometryError::Unsupported(
            "Grassmann exp_map_vjp: no analytic backward implemented",
        ))
    }
}
