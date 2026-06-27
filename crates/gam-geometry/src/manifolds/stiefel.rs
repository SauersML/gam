use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, flatten, from_flat, identity,
    matrix_exp, qr_thin, sym, tangent_basis_metric_orthonormal,
};
use crate::manifolds::sphere::SphereManifold;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiefelManifold {
    k: usize,
    n: usize,
}

impl StiefelManifold {
    /// Construct the Stiefel manifold `St(n, k) = {Y ∈ ℝ^{n×k} : YᵀY = I_k}`
    /// of `k`-frames in `ℝⁿ`. This object exists only for `1 ≤ k ≤ n`: with
    /// `k > n` there cannot be `k` orthonormal columns in `ℝⁿ`, the dimension
    /// `nk − k(k+1)/2` ceases to describe a frame manifold, and the QR
    /// retraction cannot produce `k` orthonormal columns. The domain is
    /// rejected here, before any dimension, projection, exponential, or
    /// curvature computation can run on a nonexistent manifold.
    pub fn new(k: usize, n: usize) -> GeometryResult<Self> {
        if k == 0 || n == 0 || k > n {
            return Err(GeometryError::InvalidPoint(
                "Stiefel St(n, k) requires 1 <= k <= n",
            ));
        }
        Ok(Self { k, n })
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

    /// Basis of the tangent space, orthonormal under the **canonical metric**
    /// `⟨Δ₁,Δ₂⟩ = tr(Δ₁ᵀ(I−½YYᵀ)Δ₂)` — i.e. `Qᵀ W Q = I` with
    /// `W = metric_tensor(point)`. A Euclidean-orthonormal basis would be wrong
    /// here because the canonical metric differs from the embedded inner product
    /// off the `YᵀΔ = 0` subspace (e.g. the vertical tangent has canonical norm²
    /// 1 but Euclidean norm² 2).
    fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
        check_len("Stiefel point", point.len(), self.ambient_dim())?;
        tangent_basis_metric_orthonormal(self, point, self.n, self.k)
    }

    /// Riemannian exponential under the **canonical metric**
    /// `⟨Δ₁, Δ₂⟩ = tr(Δ₁ᵀ(I − ½YYᵀ)Δ₂)`. For `k == 1` this is the sphere
    /// exponential. For general `k`, the canonical-metric geodesic at `Y`
    /// with tangent `Δ` is `exp(W) · Y`, where
    ///
    /// ```text
    ///   W = Δ Yᵀ − Y Δᵀ − Y (YᵀΔ) Yᵀ      (n×n, skew)
    /// ```
    ///
    /// is the unique skew matrix satisfying `W·Y = Δ` (using `YᵀΔ + ΔᵀY = 0`
    /// for a canonical-metric tangent). Since `W` is skew, `exp(W)` is
    /// orthogonal and `exp(W)·Y` lies on the Stiefel manifold for any
    /// `n ≥ k`. This avoids the Edelman–Arias–Smith `2k`-block form, whose
    /// thin QR of `(I − YYᵀ)Δ` is structurally rank-deficient when `n < 2k`
    /// and admits no canonical extension over the rank-deficient subspace.
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
        // Route the four large-n products (n×k · k×n and n×n · n×k) through
        // the GPU-dispatched shims; small frames stay on faer.
        use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
        let a = fast_atb(&y, &delta); // k×k: A = YᵀΔ (skew on the tangent space)
        let delta_yt = fast_abt(&delta, &y); // n×n: Δ Yᵀ
        let y_dt = fast_abt(&y, &delta); // n×n: Y Δᵀ = (Δ Yᵀ)ᵀ
        let yayt = fast_abt(&fast_ab(&y, &a), &y); // n×n: (Y A) Yᵀ
        let w = &(&delta_yt - &y_dt) - &yayt; // skew n×n
        let expw = matrix_exp(&w)?; // n×n orthogonal
        let result = fast_ab(&expw, &y); // n×k point on St(n, k)
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
    /// (see [`flatten`](crate::manifold)), the metric factorizes as
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
        // Y Yᵀ (n×k · k×n) carries the large ambient dimension n, GPU-dispatched
        // via fast_abt.
        let yyt = gam_linalg::faer_ndarray::fast_abt(&y, &y);
        let mut m = identity(self.n);
        for i in 0..self.n {
            for p in 0..self.n {
                m[[i, p]] -= 0.5 * yyt[[i, p]];
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
        use gam_linalg::faer_ndarray::{fast_ab, fast_atb};
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(vec, self.n, self.k)?;
        // Tangent projection z − Y·sym(Yᵀz): YᵀZ (k×n · n×k) and Y·S (n×k · k×k)
        // both carry the large ambient dimension n, GPU-dispatched via
        // fast_atb/fast_ab.
        let correction = fast_ab(&y, &sym(&fast_atb(&y, &z)));
        Ok(flatten(&(z - correction)))
    }

    /// Riemannian gradient under the **canonical metric**
    /// `⟨Δ₁,Δ₂⟩ = tr(Δ₁ᵀ(I−½YYᵀ)Δ₂)`. For a scalar `f` with ambient
    /// differential `E` (the `n×k` matrix of partials), the Riesz representative
    /// is the Edelman–Arias–Smith closed form
    ///
    /// ```text
    ///   grad f(Y) = E − Y Eᵀ Y.
    /// ```
    ///
    /// It is tangent (`Yᵀgrad` is skew: `Yᵀgrad = YᵀE − EᵀY = −(Yᵀgrad)ᵀ`) and
    /// satisfies `⟨grad, Δ⟩_canonical = tr(Eᵀ Δ) = ⟨E, Δ⟩` for every tangent `Δ`
    /// (the half-trace corrections contract a *symmetric* matrix against the
    /// skew `YᵀΔ` and vanish). This is the metric-raising default specialized to
    /// the canonical metric — computed directly to avoid forming the `nk×nk`
    /// metric tensor. The *embedded* projection `E − Y·sym(YᵀE)`
    /// ([`project_tangent`]) is the Euclidean-metric gradient and is wrong off
    /// the `YᵀΔ = 0` subspace for `k ≥ 2` (issue #955).
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        // For k == 1 the Stiefel manifold is the unit sphere, whose canonical
        // metric is the embedded one; the gradient is the tangent projection.
        if let Some(sphere) = self.as_sphere() {
            return sphere.riemannian_gradient(point, euclidean_grad);
        }
        use gam_linalg::faer_ndarray::{fast_ab, fast_atb};
        let y = from_flat(point, self.n, self.k)?;
        let e = from_flat(euclidean_grad, self.n, self.k)?;
        // grad = E − Y (Eᵀ Y): Eᵀ Y is k×k, Y·(EᵀY) carries the ambient n.
        let correction = fast_ab(&y, &fast_atb(&e, &y));
        Ok(flatten(&(e - correction)))
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

    /// The QR retraction `qf(Y + Δ)` is only a FIRST-ORDER retraction (its
    /// acceleration at `Δ = 0` is not normal to the manifold), so
    /// `D²(f∘R_Y)(0) ≠ Hess f(Y)` in general. The trust region must therefore
    /// not score the Riemannian-Hessian quadratic term against this retraction;
    /// it falls back to the first-order-correct Cauchy model (issue #956).
    fn retraction_is_second_order(&self) -> bool {
        false
    }

    /// Reverse-mode (vector–Jacobian product) of [`exp_map`](Self::exp_map).
    ///
    /// Given the output cotangent `Ḡ = ∂L/∂result` (n×k), returns
    /// `(∂L/∂point, ∂L/∂tangent_vec)` flattened. The derivation is the exact
    /// adjoint of the five forward steps (project → A → W → matrix-exp →
    /// multiply by Y), with the matrix-exponential adjoint obtained from
    /// the Mathias augmented identity `adj(dexp_W)·E = dexp_{Wᵀ}(E)`. No
    /// approximations: every intermediate is recomputed exactly as the
    /// forward produced it.
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

        // ── Recompute the forward intermediates exactly as `exp_map` does. ──
        // Every dense product below either contracts or carries the large
        // ambient dimension n (k×n · n×k, n×k · k×n, or n×n · n×k); route
        // them all through the GPU-dispatched fast_ab/fast_atb/fast_abt shims.
        use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(tangent_vec, self.n, self.k)?; // raw (unprojected) input
        let s_proj = sym(&fast_atb(&y, &z)); // S = sym(Yᵀz)
        let delta = &z - &fast_ab(&y, &s_proj); // Δ = z − Y·S
        let a = fast_atb(&y, &delta); // A = YᵀΔ
        let delta_yt = fast_abt(&delta, &y); // ΔYᵀ
        let y_dt = fast_abt(&y, &delta); // YΔᵀ
        let yayt = fast_abt(&fast_ab(&y, &a), &y); // (YA)Yᵀ
        let w = &(&delta_yt - &y_dt) - &yayt; // W (n×n skew)
        let expw = matrix_exp(&w)?; // n×n orthogonal

        let grad = from_flat(grad_output, self.n, self.k)?; // Ḡ (n×k)

        // ── Step 5 (result = expW · Y). ──
        //   ⟨Ḡ, dexpW · Y⟩    ⇒  W̄ ← Ḡ · Yᵀ      via matrix_exp_vjp.
        //   ⟨Ḡ, expW · dY⟩    ⇒  Ȳ += expWᵀ · Ḡ.
        let expw_bar = fast_abt(&grad, &y); // Ḡ·Yᵀ (n×n)
        let mut y_bar = fast_atb(&expw, &grad); // expWᵀ·Ḡ (n×k)

        // ── Step 4 (W → expW): W̄ = adjoint of dexp_W applied to Ḡ·Yᵀ. ──
        let w_bar = matrix_exp_vjp(&w, &expw_bar)?; // n×n

        // ── Step 3 (W = ΔYᵀ − YΔᵀ − Y A Yᵀ): split W̄ across (Δ, Y, A). ──
        //   ΔYᵀ      → Δ̄ += W̄·Y          and  Ȳ += W̄ᵀ·Δ.
        //   −YΔᵀ     → Δ̄ += −W̄ᵀ·Y         and  Ȳ += −W̄·Δ.
        //   −Y A Yᵀ  → Ȳ += −W̄·Y·Aᵀ − W̄ᵀ·Y·A     and   Ā = −Yᵀ·W̄·Y.
        let wb_y = fast_ab(&w_bar, &y); // W̄·Y (n×k)
        let wbt_y = fast_atb(&w_bar, &y); // W̄ᵀ·Y (n×k)
        let mut delta_bar = &wb_y - &wbt_y; // n×k
        y_bar = y_bar + &fast_atb(&w_bar, &delta); // + W̄ᵀ·Δ
        y_bar = y_bar - &fast_ab(&w_bar, &delta); // − W̄·Δ
        y_bar = y_bar - &fast_abt(&wb_y, &a); // − W̄·Y·Aᵀ
        y_bar = y_bar - &fast_ab(&wbt_y, &a); // − W̄ᵀ·Y·A
        let a_bar = -fast_ab(&fast_atb(&y, &w_bar), &y); // −Yᵀ·W̄·Y (k×k)

        // ── Step 2 (A = Yᵀ·Δ): Ā → (Y, Δ) via dA = dYᵀ·Δ + Yᵀ·dΔ. ──
        y_bar = y_bar + &fast_abt(&delta, &a_bar); // Ȳ += Δ·Āᵀ
        delta_bar = delta_bar + &fast_ab(&y, &a_bar); // Δ̄ += Y·Ā

        // ── Step 1 (Δ = z − Y·sym(Yᵀz)): Δ̄ → (Y, z). ──
        //   z̄ = Δ̄ − Y·sym(Yᵀ·Δ̄).
        //   Ȳ += −Δ̄·S − z·sym(Yᵀ·Δ̄).
        let sym_yt_db = sym(&fast_atb(&y, &delta_bar));
        let z_bar = &delta_bar - &fast_ab(&y, &sym_yt_db);
        y_bar = y_bar - &fast_ab(&delta_bar, &s_proj) - &fast_ab(&z, &sym_yt_db);

        Ok((flatten(&y_bar), flatten(&z_bar)))
    }
}

/// Adjoint of the Fréchet derivative of the matrix exponential at `b`, applied
/// to the cotangent `M̄` (`cotangent`). Uses the Mathias / Van Loan augmented
/// block identity: the adjoint of `dexp_B` equals `dexp_{Bᵀ}`, and the
/// Fréchet derivative of `expm` is read off the top-right block of the
/// exponential of the `2m × 2m` matrix `[[Bᵀ, M̄], [0, Bᵀ]]`. Concretely
///
/// ```text
///   exp([[Bᵀ, M̄], [0, Bᵀ]]) = [[exp(Bᵀ), dexp_{Bᵀ}(M̄)], [0, exp(Bᵀ)]],
/// ```
///
/// so `B̄ = dexp_{Bᵀ}(M̄)` is exactly the requested adjoint applied to `M̄`.
fn matrix_exp_vjp(b: &Array2<f64>, cotangent: &Array2<f64>) -> GeometryResult<Array2<f64>> {
    let m = b.nrows();
    if b.ncols() != m || cotangent.nrows() != m || cotangent.ncols() != m {
        return Err(GeometryError::InvalidPoint(
            "matrix_exp_vjp requires square matrices of equal size",
        ));
    }
    // Build the augmented 2m×2m matrix [[Bᵀ, M̄], [0, Bᵀ]].
    let two_m = 2 * m;
    let mut aug = Array2::<f64>::zeros((two_m, two_m));
    for i in 0..m {
        for j in 0..m {
            let bt = b[[j, i]]; // Bᵀ[i, j]
            aug[[i, j]] = bt;
            aug[[m + i, m + j]] = bt;
            aug[[i, m + j]] = cotangent[[i, j]];
        }
    }
    let exp_aug = matrix_exp(&aug)?;
    // Top-right block is dexp_{Bᵀ}(M̄) = adjoint(dexp_B)(M̄).
    Ok(exp_aug.slice(ndarray::s![0..m, m..two_m]).to_owned())
}

#[cfg(test)]
mod tangent_basis_tests {
    use super::StiefelManifold;
    use crate::manifold::RiemannianManifold;
    use ndarray::Array1;

    /// The Stiefel `tangent_basis` must be orthonormal under the canonical
    /// metric: `Qᵀ W Q = I` with `W = metric_tensor(Y)`. A Euclidean-orthonormal
    /// basis (the old shared routine) would give `Qᵀ W Q ≠ I` because the
    /// canonical metric `tr(Δᵀ(I−½YYᵀ)Δ)` differs from the embedded inner
    /// product off the `YᵀΔ = 0` subspace.
    #[test]
    fn stiefel_tangent_basis_metric_orthonormal() {
        // St(3, 2) at Y = [e1, e2] (row-major flatten of the 3×2 frame).
        let st = StiefelManifold::new(2, 3).expect("St(3,2) exists");
        let y = Array1::from(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let q = st.tangent_basis(y.view()).expect("tangent basis");
        let w = st.metric_tensor(y.view()).expect("metric tensor");
        let d = st.dim();
        assert_eq!(q.ncols(), d, "basis must have dim() columns");
        // QᵀWQ
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

    /// Sanity check the metric scaling the basis must capture: the vertical
    /// tangent Δ = Y·[[0,−1],[1,0]] has canonical-metric norm² 1, not the
    /// Euclidean 2. (This is the audit's discriminating case.)
    #[test]
    fn stiefel_vertical_tangent_canonical_norm() {
        let st = StiefelManifold::new(2, 3).expect("St(3,2) exists");
        let y = Array1::from(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // Δ = Y · [[0,-1],[1,0]] = columns (e2, -e1) ⇒ 3×2 with rows:
        //   row0 = (0, -1), row1 = (1, 0), row2 = (0, 0). Row-major flatten.
        let delta = Array1::from(vec![0.0, -1.0, 1.0, 0.0, 0.0, 0.0]);
        let w = st.metric_tensor(y.view()).expect("metric tensor");
        let wd = w.dot(&delta);
        let mut norm_sq = 0.0;
        for i in 0..delta.len() {
            norm_sq += delta[i] * wd[i];
        }
        assert!(
            (norm_sq - 1.0).abs() <= 1.0e-12,
            "canonical-metric norm² of vertical tangent must be 1, got {norm_sq}"
        );
    }
}
