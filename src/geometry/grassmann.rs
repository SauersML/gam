use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, dot, flatten,
    from_flat, identity, inverse, jacobi_symmetric, projected_standard_basis_tangent, qr_thin,
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
        use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
        // ΔᵀΔ (k×n · n×k) and the left singular vectors U = Δ·V·Σ⁻¹ (n×k · k×k)
        // both carry the large ambient dimension n; GPU-dispatch via fast_atb/ab.
        let gram = fast_atb(tangent, tangent);
        let (evals, v) = jacobi_symmetric(&gram)?;
        let mut sigma = Array1::<f64>::zeros(self.k);
        // U = Δ·V first (n×k), then scale each column j by 1/σ_j (skipping the
        // numerically-zero singular values, exactly as the per-column form did).
        let tangent_v = fast_ab(tangent, &v);
        let mut u = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            sigma[j] = evals[j].max(0.0).sqrt();
            if sigma[j] > GEOMETRY_EPS {
                let inv_sigma = 1.0 / sigma[j];
                for i in 0..self.n {
                    u[[i, j]] = tangent_v[[i, j]] * inv_sigma;
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
        projected_standard_basis_tangent(self, point, self.n, self.k)
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
        // Geodesic frame Y·V·cos(Σ)·Vᵀ + U·sin(Σ)·Vᵀ: dense products carrying the
        // large ambient dimension n, GPU-dispatched via fast_ab/fast_abt.
        use crate::linalg::faer_ndarray::{fast_ab, fast_abt};
        let yv_cos = fast_ab(&fast_ab(&y, &v), &cos_d);
        let u_sin = fast_ab(&u, &sin_d);
        let next = &fast_abt(&yv_cos, &v) + &fast_abt(&u_sin, &v);
        Ok(flatten(&self.orthonormalize(&next)))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            // Gr(1,n) = ℝP^{n-1}: the line span(p_to) is represented equally by
            // ±p_to, so before applying the sphere logarithm we pick the
            // representative in p_from's hemisphere (p_from·q ≥ 0). Without this
            // the sphere would report distance π−ε for two nearly-identical
            // lines that are ε apart. At the projective cut locus p_from·p_to=0
            // (principal angle π/2) the log is not unique, so we reject it —
            // mirroring the (YᵀZ)⁻¹ singularity of the general-k branch below.
            // At the projective cut locus p_from·p_to = 0 (principal angle
            // pi/2) the minimal geodesic is non-unique, but the cut-locus
            // distance is exactly pi/2 and at least one minimal geodesic always
            // exists: it leaves p_from along the unit direction in span(p_to)
            // orthogonal to p_from. Rather than abort — which strands an
            // otherwise well-defined Frechet mean whenever two responses happen
            // to be orthogonal lines — return that canonical length-pi/2
            // tangent. It is a genuine Riemannian log (correct magnitude,
            // tangent at p_from), so exp∘log round-trips and the Karcher descent
            // converges; only the arbitrary choice among equivalent minimizers
            // is fixed.
            let c = dot(p_from, p_to);
            if c.abs() <= GEOMETRY_EPS {
                let mut dir = p_to.to_owned();
                dir.scaled_add(-c, &p_from);
                let norm = dir.dot(&dir).sqrt();
                if norm <= GEOMETRY_EPS {
                    return Ok(Array1::<f64>::zeros(p_from.len()));
                }
                dir.mapv_inplace(|x| x * (std::f64::consts::FRAC_PI_2 / norm));
                return Ok(dir);
            }
            if c < 0.0 {
                let aligned = -&p_to.to_owned();
                return sphere.log_map(p_from, aligned.view());
            }
            return sphere.log_map(p_from, p_to);
        }
        use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
        let y = from_flat(p_from, self.n, self.k)?;
        let z = from_flat(p_to, self.n, self.k)?;
        // YᵀZ (k×n · n×k), the normal Z − Y(YᵀZ) and M = normal·(YᵀZ)⁻¹ (n×k · k×k),
        // and MᵀM (k×n · n×k): all carry n, GPU-dispatched via fast_atb/fast_ab.
        let yt_z = fast_atb(&y, &z);
        let inv = inverse(&yt_z)?;
        let normal = z - fast_ab(&y, &yt_z);
        let m = fast_ab(&normal, &inv);
        let gram = fast_atb(&m, &m);
        let (evals, v) = jacobi_symmetric(&gram)?;
        let mut sigma = Array1::<f64>::zeros(self.k);
        // U = M·V scaled column-wise by 1/tan(σ_j) (M·V is n×k · k×k, carrying n).
        let m_v = fast_ab(&m, &v);
        let mut u = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            let tan_sigma = evals[j].max(0.0).sqrt();
            sigma[j] = tan_sigma.atan();
            if tan_sigma > GEOMETRY_EPS {
                let inv_tan = 1.0 / tan_sigma;
                for i in 0..self.n {
                    u[[i, j]] = m_v[[i, j]] * inv_tan;
                }
            }
        }
        let mut diag = Array2::<f64>::zeros((self.k, self.k));
        for i in 0..self.k {
            diag[[i, i]] = sigma[i];
        }
        // Δ = U·Σ·Vᵀ: n×k · k×k · k×k, GPU-dispatched.
        Ok(flatten(&crate::linalg::faer_ndarray::fast_abt(
            &fast_ab(&u, &diag),
            &v,
        )))
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            // Gr(1,n) = ℝP^{n-1}: align the final representative's sign to the
            // first point's hemisphere so the transport runs along the minimal
            // RP geodesic (the sphere geodesic to the aligned ±endpoint), rather
            // than the antipodal great circle. The tangent space at a line is
            // the same for ±q, so negating the endpoint representative is the
            // correct lift.
            let last = point_along.nrows().saturating_sub(1);
            if point_along.nrows() >= 2 && dot(point_along.row(0), point_along.row(last)) < 0.0 {
                let mut aligned = point_along.to_owned();
                aligned.row_mut(last).mapv_inplace(|x| -x);
                return sphere.parallel_transport(aligned.view(), vec);
            }
            return sphere.parallel_transport(point_along, vec);
        }
        check_len(
            "Grassmann path width",
            point_along.ncols(),
            self.ambient_dim(),
        )?;
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
        // Coordinates of H in the U-frame: ut_h = Uᵀ H (k×n · n×k). The transport
        // operator's three dense terms all carry the large ambient dimension n;
        // GPU-dispatch via fast_ab/fast_atb.
        use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
        let ut_h = fast_atb(&u, &h);
        // Geodesic-aligned components: U cos(Σ) Uᵀ H − Y V sin(Σ) Uᵀ H.
        let aligned = &fast_ab(&fast_ab(&u, &cos_d), &ut_h)
            - &fast_ab(&fast_ab(&fast_ab(&y, &v), &sin_d), &ut_h);
        // Component of H orthogonal to the geodesic 2-plane: (I - U Uᵀ) H.
        let orthogonal = &h - &fast_ab(&u, &ut_h);
        Ok(flatten(&(aligned + orthogonal)))
    }

    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Grassmann metric point", point.len(), self.ambient_dim())?;
        Ok(identity(self.ambient_dim()))
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
        let x = from_flat(
            self.project_tangent(point, tangent_pair.0)?.view(),
            self.n,
            self.k,
        )?;
        let y = from_flat(
            self.project_tangent(point, tangent_pair.1)?.view(),
            self.n,
            self.k,
        )?;
        // Tangent Gram matrices (each k×n · n×k, carrying the large ambient
        // dimension n), GPU-dispatched via fast_atb.
        use crate::linalg::faer_ndarray::fast_atb;
        let gxx = fast_atb(&x, &x);
        let gyy = fast_atb(&y, &y);
        let gxy = fast_atb(&x, &y);
        let gyx = fast_atb(&y, &x);
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
        use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(vec, self.n, self.k)?;
        // Z − Y(YᵀZ): YᵀZ (k×n · n×k) and Y·(YᵀZ) (n×k · k×k) both carry n,
        // GPU-dispatched via fast_atb/fast_ab.
        let projected = &z - &fast_ab(&y, &fast_atb(&y, &z));
        Ok(flatten(&projected))
    }

    /// The Grassmannian carries the canonical metric `⟨Δ₁,Δ₂⟩ = tr(Δ₁ᵀΔ₂)`,
    /// which is the *embedded* Frobenius inner product restricted to the
    /// horizontal tangent space. The Riemannian gradient is therefore the
    /// horizontal (Frobenius-orthogonal) projection of the ambient gradient —
    /// exactly [`project_tangent`] — not the dense metric-raising default.
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.project_tangent(point, euclidean_grad)
    }

    /// QR retraction `R_Y(Δ) = qf(Y + Δ)`. This is a first-order retraction,
    /// distinct from the Riemannian [`exp_map`](Self::exp_map); the two agree
    /// only to first order in `Δ`.
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

    /// The QR retraction `qf(Y + Δ)` is only a FIRST-ORDER retraction, so
    /// `D²(f∘R_Y)(0) ≠ Hess f(Y)` in general. The trust region must therefore
    /// not score the Riemannian-Hessian quadratic term against this retraction;
    /// it falls back to the first-order-correct Cauchy model (issue #956).
    fn retraction_is_second_order(&self) -> bool {
        false
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Row-major flatten of an `n×k` frame into the `vec[r*k + c] = M[r, c]`
    /// layout `from_flat`/`flatten` use.
    fn flat(m: &Array2<f64>) -> Array1<f64> {
        let (rows, cols) = m.dim();
        let mut v = Array1::<f64>::zeros(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                v[r * cols + c] = m[[r, c]];
            }
        }
        v
    }

    /// Build two orthonormal `n×k` frames whose principal angles are EXACTLY the
    /// supplied `angles`, by rotating column `j` of the identity frame inside the
    /// disjoint coordinate 2-plane `(e_j, e_{k+j})`. With the `k` rotation planes
    /// pairwise orthogonal, `Y = [e_0 … e_{k-1}]` and
    /// `Z = [cosθ_j e_j + sinθ_j e_{k+j}]_j` satisfy `YᵀZ = diag(cosθ_j)`, so the
    /// principal angles of `span(Y), span(Z)` are precisely `θ_j` — analytic
    /// ground truth, no SVD/`arccos` conditioning and no external tool. Requires
    /// `n ≥ 2k` so the rotation planes do not overlap.
    fn frames_with_angles(n: usize, k: usize, angles: &[f64]) -> (Array2<f64>, Array2<f64>) {
        assert!(n >= 2 * k, "disjoint rotation planes need n >= 2k");
        assert_eq!(angles.len(), k);
        let mut y = Array2::<f64>::zeros((n, k));
        let mut z = Array2::<f64>::zeros((n, k));
        for (j, &theta) in angles.iter().enumerate() {
            y[[j, j]] = 1.0;
            z[[j, j]] = theta.cos();
            z[[k + j, j]] = theta.sin();
        }
        (y, z)
    }

    #[test]
    fn geodesic_distance_equals_analytic_principal_angle_arc_length() {
        // Gr(2, 6): two subspaces with KNOWN principal angles spanning the whole
        // injectivity-radius range, including an angle essentially at the π/2 cut
        // (where the `(YᵀZ)⁻¹` form is most stressed). The geodesic distance must
        // equal sqrt(Σ θ_j²) — the exact arc-length — to f64 linear-algebra noise.
        let gr = GrassmannManifold::new(2, 6).expect("Gr(2,6)");
        let cases: [[f64; 2]; 4] = [
            [0.1, 0.7],
            [0.9, 1.4],
            [0.3, 1.5705], // one angle a hair below π/2 ≈ 1.5708
            [1.2, 1.2],    // degenerate (repeated) angle: V-block is an arbitrary rotation
        ];
        for angles in cases {
            let (y, z) = frames_with_angles(6, 2, &angles);
            let log = gr
                .log_map(flat(&y).view(), flat(&z).view())
                .expect("log_map between known-angle frames");
            let dist: f64 = log.iter().map(|x| x * x).sum::<f64>().sqrt();
            let analytic: f64 = angles.iter().map(|t| t * t).sum::<f64>().sqrt();
            assert!(
                (dist - analytic).abs() < 1e-12,
                "geodesic distance {dist:.16} != analytic arc-length {analytic:.16} for \
                 angles {angles:?}"
            );
        }
    }

    #[test]
    fn exp_log_roundtrip_recovers_tangent_to_machine_precision() {
        // exp_P(v) then log back must return v componentwise, and the recovered
        // tangent's singular spectrum must equal the input principal angles — at
        // both tiny (sub-microradian) and near-π/2 scales. This pins gam's exp/log
        // involution against analytic truth (atan-recovered, well-conditioned),
        // independent of the arccos-near-1 endpoint extraction the e2e test uses.
        let gr = GrassmannManifold::new(3, 9).expect("Gr(3,9)");
        let scales: [f64; 5] = [1e-7, 1e-4, 0.3, 1.0, 1.5];
        let dirs: [[f64; 3]; 1] = [[0.4, 0.7, 1.0]]; // distinct so V is well separated
        for s in scales {
            for d in dirs {
                let angles = [d[0] * s, d[1] * s, d[2] * s];
                // Tangent matrix Δ = U Σ Vᵀ with U the rotation-plane image axes,
                // V = I, Σ = diag(angles): a horizontal tangent at Y whose compact
                // SVD spectrum is exactly `angles`.
                let (y, _z) = frames_with_angles(9, 3, &angles);
                let mut tangent = Array2::<f64>::zeros((9, 3));
                for (j, &theta) in angles.iter().enumerate() {
                    tangent[[3 + j, j]] = theta; // e_{k+j} direction, magnitude θ_j
                }
                let y_flat = flat(&y);
                let v_flat = flat(&tangent);
                // The tangent is horizontal (YᵀΔ = 0 by construction).
                let endpoint = gr
                    .exp_map(y_flat.view(), v_flat.view())
                    .expect("exp_map of horizontal tangent");
                let v_rec = gr
                    .log_map(y_flat.view(), endpoint.view())
                    .expect("log_map of geodesic endpoint");
                let mut max_abs = 0.0_f64;
                for (a, b) in v_rec.iter().zip(v_flat.iter()) {
                    max_abs = max_abs.max((a - b).abs());
                }
                assert!(
                    max_abs < 1e-10,
                    "exp/log roundtrip error {max_abs:.3e} at scale {s:.1e} (angles {angles:?})"
                );
                // Isometry: ‖log(exp v)‖_F == ‖v‖_F == ‖angles‖₂.
                let rec_norm: f64 = v_rec.iter().map(|x| x * x).sum::<f64>().sqrt();
                let truth_norm: f64 = angles.iter().map(|t| t * t).sum::<f64>().sqrt();
                assert!(
                    (rec_norm - truth_norm).abs() < 1e-10,
                    "isometry error {:.3e} at scale {s:.1e}",
                    (rec_norm - truth_norm).abs()
                );
            }
        }
    }
}
