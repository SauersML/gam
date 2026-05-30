use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, dot, flatten, from_flat,
    identity, matrix_exp, qr_thin, sym, zero_christoffel,
};
use crate::geometry::sphere::SphereManifold;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiefelManifold {
    k: usize,
    n: usize,
}

impl StiefelManifold {
    pub const fn new(k: usize, n: usize) -> Self {
        Self { k, n }
    }

    /// QR-based *retraction* `R_Y(Δ) = qf(Y + Δ)` with the sign convention that
    /// makes the diagonal of `R` non-negative (so the retraction is a smooth
    /// map agreeing with the exponential to first order). This is a retraction,
    /// not the Riemannian exponential, and is exposed only through
    /// [`retract`](RiemannianManifold::retract).
    fn qr_retraction(&self, y: &Array2<f64>) -> Array2<f64> {
        let (mut q, r) = qr_thin(y);
        for j in 0..self.k {
            if r[[j, j]] < 0.0 {
                for i in 0..self.n {
                    q[[i, j]] = -q[[i, j]];
                }
            }
        }
        q
    }

    /// For `k == 1` the Stiefel manifold `St(n, 1)` is exactly the unit sphere
    /// `S^{n-1}` (a single unit column is a point on the sphere), and the flat
    /// ambient coordinates coincide. Reuse the [`SphereManifold`] formulas so
    /// the exponential, logarithm, parallel transport, and curvature are the
    /// genuine Riemannian objects rather than re-derived approximations.
    fn as_sphere(&self) -> Option<SphereManifold> {
        (self.k == 1).then(|| SphereManifold::new(self.n - 1))
    }
}

impl RiemannianManifold for StiefelManifold {
    fn dim(&self) -> usize {
        self.n * self.k - self.k * (self.k + 1) / 2
    }

    fn ambient_dim(&self) -> usize {
        self.n * self.k
    }

    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Stiefel point", point.len(), self.ambient_dim())?;
        let mut columns: Vec<Array1<f64>> = Vec::with_capacity(self.dim());
        for col in 0..self.k {
            for row in 0..self.n {
                let mut e = Array2::<f64>::zeros((self.n, self.k));
                e[[row, col]] = 1.0;
                let mut v = self.project_tangent(point, flatten(&e).view())?;
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

    /// Riemannian exponential under the **canonical metric**
    /// `⟨Δ₁, Δ₂⟩ = tr(Δ₁ᵀ(I − ½YYᵀ)Δ₂)`. For `k == 1` this is the sphere
    /// exponential. For general `k`, with `A = YᵀΔ` (skew-symmetric on the
    /// tangent space), compact QR `(I − YYᵀ)Δ = QR`, the geodesic is the
    /// Edelman–Arias–Smith closed form
    ///
    /// ```text
    ///   Exp_Y(Δ) = [Y  Q] · exp([[A, −Rᵀ], [R, 0]]) · [[I_k], [0]].
    /// ```
    fn exp_map(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.exp_map(point, tangent_vec);
        }
        let y = from_flat(point, self.n, self.k)?;
        let delta = from_flat(
            self.project_tangent(point, tangent_vec)?.view(),
            self.n,
            self.k,
        )?;
        let a = y.t().dot(&delta); // k×k skew-symmetric
        let normal = &delta - &y.dot(&a); // (I − YYᵀ)Δ
        let (q, r) = qr_thin(&normal); // n×k, k×k

        // Block generator [[A, −Rᵀ], [R, 0]] of size 2k×2k.
        let two_k = 2 * self.k;
        let mut block = Array2::<f64>::zeros((two_k, two_k));
        for i in 0..self.k {
            for j in 0..self.k {
                block[[i, j]] = a[[i, j]];
                block[[i, self.k + j]] = -r[[j, i]];
                block[[self.k + i, j]] = r[[i, j]];
            }
        }
        let exp_block = matrix_exp(&block)?;

        // Result = [Y Q] · exp_block[:, 0..k]; only the first k columns of the
        // exponential survive against the [[I_k], [0]] selector.
        let mut result = Array2::<f64>::zeros((self.n, self.k));
        for col in 0..self.k {
            for row in 0..self.n {
                let mut acc = 0.0;
                for s in 0..self.k {
                    acc += y[[row, s]] * exp_block[[s, col]];
                    acc += q[[row, s]] * exp_block[[self.k + s, col]];
                }
                result[[row, col]] = acc;
            }
        }
        Ok(flatten(&result))
    }

    fn log_map(
        &self,
        p_from: ArrayView1<'_, f64>,
        p_to: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.log_map(p_from, p_to);
        }
        check_len("Stiefel source", p_from.len(), self.ambient_dim())?;
        check_len("Stiefel target", p_to.len(), self.ambient_dim())?;
        // The Stiefel logarithm under the canonical metric has no elementary
        // closed form for k > 1 (it is the solution of an iterative algebraic
        // Riccati / matrix-log iteration). Refuse rather than return the
        // projected ambient difference, which is *not* the inverse of the
        // geodesic exponential and would silently violate Exp∘Log = id.
        Err(GeometryError::Unsupported(
            "Stiefel log_map: no closed-form Riemannian logarithm for k > 1",
        ))
    }

    fn parallel_transport(
        &self,
        point_along: ArrayView2<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.parallel_transport(point_along, vec);
        }
        check_len("Stiefel transported vector", vec.len(), self.ambient_dim())?;
        // Parallel transport along a Stiefel geodesic under the canonical
        // connection has no elementary closed form for k > 1, and endpoint
        // tangent projection is *not* parallel transport (it does not preserve
        // the canonical inner product and can annihilate nonzero vectors).
        // Refuse rather than return a mathematically false value.
        Err(GeometryError::Unsupported(
            "Stiefel parallel_transport: no closed-form transport for k > 1",
        ))
    }

    /// Gram matrix of the **canonical metric**
    /// `⟨Δ₁, Δ₂⟩ = tr(Δ₁ᵀ(I − ½YYᵀ)Δ₂)`, expressed in the flattened ambient
    /// basis so that `quad_form(G, vec(Δ₁), vec(Δ₂))` reproduces this inner
    /// product. This is the *same* metric whose geodesic is implemented by
    /// [`exp_map`](Self::exp_map); returning the embedded/Euclidean identity
    /// here would contradict the geodesic for `k ≥ 2` (the two metrics differ
    /// off the `YᵀΔ = 0` subspace).
    ///
    /// With the row-major flatten `vec(Δ)[i·k + j] = Δ[i, j]`
    /// (see [`flatten`](crate::geometry::manifold)), the metric factorizes as
    /// the Kronecker product `(I − ½YYᵀ) ⊗ I_k`: entry `M[i, p]` of the n×n
    /// matrix `M = I − ½YYᵀ` scales the `k×k` identity block coupling rows `i`
    /// and `p`, i.e. `G[i·k + j, p·k + q] = M[i, p] · δ_{j, q}`.
    ///
    /// For `k == 1` the Stiefel manifold is the unit sphere; dispatch to
    /// [`SphereManifold`], whose embedded metric coincides with the canonical
    /// metric on the (one-dimensional-codimension) tangent space `YᵀΔ = 0` and
    /// whose [`exp_map`](SphereManifold::exp_map) is likewise the genuine
    /// Riemannian exponential, so metric and geodesic remain consistent.
    fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        if let Some(sphere) = self.as_sphere() {
            return sphere.metric_tensor(point);
        }
        let y = from_flat(point, self.n, self.k)?;
        // M = I_n − ½ Y Yᵀ (n×n, symmetric positive definite for Yᵀ Y = I_k).
        let mut m = identity(self.n);
        for i in 0..self.n {
            for p in 0..self.n {
                let mut yyt = 0.0;
                for s in 0..self.k {
                    yyt += y[[i, s]] * y[[p, s]];
                }
                m[[i, p]] -= 0.5 * yyt;
            }
        }
        // G = M ⊗ I_k in the row-major flattened basis.
        let ambient = self.ambient_dim();
        let mut g = Array2::<f64>::zeros((ambient, ambient));
        for i in 0..self.n {
            for p in 0..self.n {
                let block = m[[i, p]];
                for j in 0..self.k {
                    g[[i * self.k + j, p * self.k + j]] = block;
                }
            }
        }
        Ok(g)
    }

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Stiefel Christoffel point", point.len(), self.ambient_dim())?;
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
        check_len("Stiefel curvature point", point.len(), self.ambient_dim())?;
        check_len(
            "Stiefel curvature tangent u",
            tangent_pair.0.len(),
            self.ambient_dim(),
        )?;
        check_len(
            "Stiefel curvature tangent v",
            tangent_pair.1.len(),
            self.ambient_dim(),
        )?;
        // The canonical-metric Stiefel sectional curvature for k > 1 is a
        // nontrivial expression in the horizontal/vertical components of the
        // tangent pair; returning 0.0 (flat) is simply wrong (St(n, 1) is the
        // curvature-+1 sphere, handled above). Until the full curvature tensor
        // is wired up, refuse rather than report a false flat value.
        Err(GeometryError::Unsupported(
            "Stiefel sectional_curvature: no closed-form value for k > 1",
        ))
    }

    fn project_tangent(
        &self,
        point: ArrayView1<'_, f64>,
        vec: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(vec, self.n, self.k)?;
        let correction = y.dot(&sym(&y.t().dot(&z)));
        Ok(flatten(&(z - correction)))
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
        Ok(flatten(&self.qr_retraction(&(y + tangent))))
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
        check_len("Stiefel exp_map_vjp point", point.len(), m)?;
        check_len("Stiefel exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("Stiefel exp_map_vjp grad", grad_output.len(), m)?;
        // The Stiefel geodesic VJP requires differentiating the matrix
        // exponential of the canonical block form; no closed form is wired
        // up. Refuse rather than inherit the flat identity default.
        Err(GeometryError::Unsupported(
            "Stiefel exp_map_vjp: no analytic backward implemented",
        ))
    }
}
