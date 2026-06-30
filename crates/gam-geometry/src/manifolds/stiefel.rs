use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, flatten, from_flat, identity,
    matrix_det, matrix_exp, orthonormal_completion, qr_thin, skew_log_orthogonal, sym,
    tangent_basis_metric_orthonormal,
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
        let y = from_flat(p_from, self.n, self.k)?;
        let y_target = from_flat(p_to, self.n, self.k)?;
        stiefel_canonical_log(&y, &y_target, self.n, self.k)
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

/// Riemannian logarithm on `St(n, k)` under the **canonical metric** for
/// `k ≥ 2`: the tangent `Δ` at `Y` with `Exp_Y(Δ) = Ỹ`, computed by
/// Zimmermann's matrix-algebraic algorithm (Zimmermann, *A matrix-algebraic
/// algorithm for the Riemannian logarithm on the Stiefel manifold under the
/// canonical metric*, SIAM J. Matrix Anal. Appl. 38(2):322–342, 2017).
///
/// It is the exact inverse of [`StiefelManifold::exp_map`]: both are the
/// canonical-metric geodesic, the exponential written in the single `n×n`
/// matrix-exponential form `exp(W)·Y` and the logarithm in the equivalent
/// Edelman–Arias–Smith block form. Let `[Y Y⊥] ∈ SO(n)` complete `Y` with an
/// orthonormal basis `Y⊥` (`n×(n−k)`) of the normal space. In that basis the
/// canonical geodesic is
///
/// ```text
///   Exp_Y(Δ) = [Y Y⊥] · expm(Ω) · [I_k; 0],
///   Ω = [[A, −Bᵀ], [B, 0]]  (n×n skew),  A = YᵀΔ (k×k),  B = Y⊥ᵀΔ ((n−k)×k),
/// ```
///
/// the **zero** lower-right block being exactly what distinguishes a
/// canonical-metric (horizontal) geodesic. Writing the target in the same
/// basis, `[Y Y⊥]ᵀ Ỹ = [M; B₀]` with `M = YᵀỸ`, `B₀ = Y⊥ᵀỸ`, gives an `n×k`
/// matrix with orthonormal columns; complete it to `V ∈ SO(n)`. Then `log(V)`
/// has blocks `[[A, −Bᵀ], [B, C]]`, and the algorithm repeatedly
/// right-multiplies `V` by `diag(I_k, expm(−C))` — which leaves the first `k`
/// columns (`= [M; B₀]`) fixed — to drive `C → 0`, after which the generator
/// has the horizontal form and `Δ = Y A + Y⊥ B`.
///
/// Using the *full* `n×n` complement (rather than the economical `2k×2k`
/// block, which is structurally rank-deficient when `n < 2k`, e.g. `St(3, 2)`)
/// makes the routine correct for every `n ≥ k`: when `n − k < k` the lower
/// block `C` is small (a `1×1` skew block is identically zero) and the
/// iteration terminates at once.
///
/// Convergence is guaranteed when `Y` and `Ỹ` lie within the injectivity
/// radius; for frames at or beyond the cut locus the matrix logarithm or the
/// iteration reports the failure rather than returning a non-minimizing
/// geodesic, surfacing as an `Unsupported` error.
fn stiefel_canonical_log(
    y: &Array2<f64>,
    y_target: &Array2<f64>,
    n: usize,
    k: usize,
) -> GeometryResult<Array1<f64>> {
    use gam_linalg::faer_ndarray::{fast_ab, fast_atb};

    let c_dim = n - k; // dimension of the normal space spanned by Y⊥

    // Orthonormal frame `[Y | Y⊥] ∈ SO(n)`; `Y⊥` is the last `c_dim` columns.
    let frame_y = orthonormal_completion(y); // n×n
    let mut y_perp = Array2::<f64>::zeros((n, c_dim));
    for j in 0..c_dim {
        for i in 0..n {
            y_perp[[i, j]] = frame_y[[i, k + j]];
        }
    }

    // Build the matching frame `[Ỹ | Ỹ⊥] ∈ SO(n)` whose complement `Ỹ⊥` is the
    // image of `Y⊥` projected off `Ỹ` and re-orthonormalized. Completing `Ỹ`
    // with *standard axes* (as a generic completion would) yields an arbitrary
    // basis that can turn `V₀` into a reflection (a −1 eigenvalue / π rotation)
    // even for nearby frames; anchoring the completion to `Y⊥` instead makes
    // `V₀ = [Y Y⊥]ᵀ [Ỹ Ỹ⊥] → I` as `Ỹ → Y`, so its principal logarithm is the
    // small geodesic generator the iteration expects.
    let yt_yperp = fast_atb(y_target, &y_perp); // k×c_dim
    let mut y_perp_t = &y_perp - &fast_ab(y_target, &yt_yperp); // (I − ỸỸᵀ)Y⊥
    // Modified Gram–Schmidt to re-orthonormalize the projected columns,
    // re-projecting against Ỹ each pass for numerical safety.
    for j in 0..c_dim {
        for _pass in 0..2 {
            // Orthogonalize column j against Ỹ.
            for col in 0..k {
                let mut dot = 0.0_f64;
                for i in 0..n {
                    dot += y_target[[i, col]] * y_perp_t[[i, j]];
                }
                for i in 0..n {
                    y_perp_t[[i, j]] -= dot * y_target[[i, col]];
                }
            }
            // Orthogonalize against earlier complement columns.
            for prev in 0..j {
                let mut dot = 0.0_f64;
                for i in 0..n {
                    dot += y_perp_t[[i, prev]] * y_perp_t[[i, j]];
                }
                for i in 0..n {
                    y_perp_t[[i, j]] -= dot * y_perp_t[[i, prev]];
                }
            }
        }
        let mut nrm = 0.0_f64;
        for i in 0..n {
            nrm += y_perp_t[[i, j]] * y_perp_t[[i, j]];
        }
        let nrm = nrm.sqrt();
        if nrm > 1.0e-12 {
            for i in 0..n {
                y_perp_t[[i, j]] /= nrm;
            }
        }
    }

    // Assemble the full frame [Ỹ | Ỹ⊥] and force it into SO(n) so that
    // V₀ = [Y Y⊥]ᵀ [Ỹ Ỹ⊥] has det +1 (no spurious −1 eigenvalue). Flipping the
    // last complement column flips the determinant and leaves V₀'s first k
    // columns (= [M; B₀], independent of Ỹ⊥) unchanged.
    let mut frame_yt = Array2::<f64>::zeros((n, n));
    for j in 0..k {
        for i in 0..n {
            frame_yt[[i, j]] = y_target[[i, j]];
        }
    }
    for j in 0..c_dim {
        for i in 0..n {
            frame_yt[[i, k + j]] = y_perp_t[[i, j]];
        }
    }
    if c_dim >= 1 && matrix_det(&frame_yt) < 0.0 {
        for i in 0..n {
            frame_yt[[i, n - 1]] = -frame_yt[[i, n - 1]];
        }
    }
    let mut v = fast_atb(&frame_y, &frame_yt); // n×n, first k columns = [M; B₀]

    const MAX_ITER: usize = 100;
    const TOL: f64 = 1.0e-13;
    let mut a_block = Array2::<f64>::zeros((k, k));
    let mut b_block = Array2::<f64>::zeros((c_dim, k));
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let log_v = skew_log_orthogonal(&v)?; // n×n skew
        let mut c_norm_sq = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                a_block[[i, j]] = log_v[[i, j]];
            }
        }
        for i in 0..c_dim {
            for j in 0..k {
                b_block[[i, j]] = log_v[[k + i, j]];
            }
            for j in 0..c_dim {
                let c = log_v[[k + i, k + j]];
                c_norm_sq += c * c;
            }
        }
        if c_norm_sq.sqrt() <= TOL {
            converged = true;
            break;
        }
        // Φ = expm(−C); V ← V · diag(I_k, Φ) — right-multiply the last c_dim
        // columns, leaving the first k (= P) untouched.
        let mut neg_c = Array2::<f64>::zeros((c_dim, c_dim));
        for i in 0..c_dim {
            for j in 0..c_dim {
                neg_c[[i, j]] = -log_v[[k + i, k + j]];
            }
        }
        let phi = matrix_exp(&neg_c)?;
        let mut v_new = v.clone();
        for r in 0..n {
            for j in 0..c_dim {
                let mut acc = 0.0_f64;
                for t in 0..c_dim {
                    acc += v[[r, k + t]] * phi[[t, j]];
                }
                v_new[[r, k + j]] = acc;
            }
        }
        v = v_new;
    }
    if !converged {
        return Err(GeometryError::Unsupported(
            "Stiefel log_map: iteration did not converge \
             (frames beyond the injectivity radius / near the cut locus)",
        ));
    }

    // Δ = Y A + Y⊥ B.
    let delta = &fast_ab(y, &a_block) + &fast_ab(&y_perp, &b_block);
    Ok(flatten(&delta))
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

#[cfg(test)]
mod stiefel_tests {
    use super::StiefelManifold;
    use crate::manifold::{GeometryError, RiemannianManifold, from_flat};
    use ndarray::{Array1, Array2};

    #[test]
    fn constructor_rejects_invalid_args() {
        assert!(StiefelManifold::new(3, 2).is_err());
        assert!(StiefelManifold::new(0, 3).is_err());
        assert!(StiefelManifold::new(1, 0).is_err());
        assert!(StiefelManifold::new(2, 2).is_ok());
        assert!(StiefelManifold::new(1, 5).is_ok());
    }

    #[test]
    fn dim_and_ambient_dim_are_correct() {
        // St(2, 3): dim = 3*2 − 2*3/2 = 6 − 3 = 3, ambient = 6
        let st = StiefelManifold::new(2, 3).unwrap();
        assert_eq!(st.dim(), 3);
        assert_eq!(st.ambient_dim(), 6);
        // St(1, 4): dim = 4*1 − 1*2/2 = 4 − 1 = 3, ambient = 4
        let st14 = StiefelManifold::new(1, 4).unwrap();
        assert_eq!(st14.dim(), 3);
        assert_eq!(st14.ambient_dim(), 4);
    }

    /// Round-trip identity `Log_Y(Exp_Y(Δ)) = Δ` for `k = 2`: the canonical
    /// Stiefel logarithm must invert the canonical exponential exactly (to
    /// solver precision) on a tangent of moderate size. This is the property
    /// the old `Unsupported` stub could not provide.
    #[test]
    fn log_inverts_exp_k2() {
        let st = StiefelManifold::new(2, 4).unwrap();
        // Y = [e0, e1] in St(4, 2), row-major 4×2 flatten.
        let y = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // A moderate raw tangent; project it onto the tangent space first.
        let raw = Array1::from(vec![0.0_f64, -0.30, 0.30, 0.0, 0.15, -0.05, 0.10, 0.20]);
        let delta = st.project_tangent(y.view(), raw.view()).unwrap();
        let target = st.exp_map(y.view(), delta.view()).unwrap();
        let recovered = st.log_map(y.view(), target.view()).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..delta.len() {
            worst = worst.max((recovered[i] - delta[i]).abs());
        }
        assert!(worst < 1e-9, "Log∘Exp != id: max|Δ̂ − Δ| = {worst:.3e}");
    }

    /// Round-trip the other way, `Exp_Y(Log_Y(Ỹ)) = Ỹ`, on a nearby St(3, 2)
    /// frame produced by a small rotation — the use case the Fréchet-mean
    /// initializer exercises.
    #[test]
    fn exp_inverts_log_k2() {
        let st = StiefelManifold::new(2, 3).unwrap();
        let y = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // Ỹ = small canonical-metric step away from Y.
        let raw = Array1::from(vec![0.0_f64, 0.12, -0.12, 0.0, 0.08, 0.05]);
        let step = st.project_tangent(y.view(), raw.view()).unwrap();
        let y_target = st.exp_map(y.view(), step.view()).unwrap();
        let lg = st.log_map(y.view(), y_target.view()).unwrap();
        let back = st.exp_map(y.view(), lg.view()).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..y_target.len() {
            worst = worst.max((back[i] - y_target[i]).abs());
        }
        assert!(worst < 1e-9, "Exp∘Log != id: max|Ŷ − Ỹ| = {worst:.3e}");
    }

    /// Exhaustive `Log_Y(Exp_Y(Δ)) = Δ` sweep across `(n, k)` regimes and a
    /// range of tangent magnitudes, with deterministic pseudo-random tangents.
    /// This stresses the genuinely iterative `C → 0` loop (for `c_dim ≥ 2` the
    /// lower-right block is non-trivially skew and several iterations run),
    /// not just the near-identity one-shot, and covers both `n < 2k` (St(3,2),
    /// St(5,3)) and `n ≥ 2k` (St(4,2), St(6,2), St(7,3)).
    #[test]
    fn log_inverts_exp_sweep_all_regimes() {
        // Tiny deterministic LCG so the test is reproducible without `rand`.
        let mut state: u64 = 0x9e3779b97f4a7c15;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0 // ∈ (−1, 1)
        };
        for &(k, n) in &[(2usize, 3usize), (2, 4), (2, 5), (2, 6), (3, 5), (3, 7)] {
            let st = StiefelManifold::new(k, n).unwrap();
            // Base frame: first k standard axes, row-major n×k flatten.
            let mut y = Array1::<f64>::zeros(n * k);
            for j in 0..k {
                y[j * k + j] = 1.0;
            }
            for &scale in &[0.05_f64, 0.3, 0.7, 1.1] {
                let raw: Array1<f64> = (0..n * k).map(|_| next()).collect();
                let mut delta = st.project_tangent(y.view(), raw.view()).unwrap();
                // Normalize to the requested canonical magnitude.
                let g = st.metric_tensor(y.view()).unwrap();
                let gd = g.dot(&delta);
                let nrm = (0..delta.len()).map(|i| delta[i] * gd[i]).sum::<f64>().sqrt();
                if nrm > 1e-12 {
                    delta.mapv_inplace(|x| x * scale / nrm);
                }
                let target = st.exp_map(y.view(), delta.view()).unwrap();
                // Target must be a valid frame.
                let yt = from_flat(target.view(), n, k).unwrap();
                let gram = yt.t().dot(&yt);
                for a in 0..k {
                    for b in 0..k {
                        let want = if a == b { 1.0 } else { 0.0 };
                        assert!(
                            (gram[[a, b]] - want).abs() < 1e-10,
                            "St({n},{k}) exp off-manifold"
                        );
                    }
                }
                let recovered = st.log_map(y.view(), target.view()).unwrap();
                let mut worst = 0.0_f64;
                for i in 0..delta.len() {
                    worst = worst.max((recovered[i] - delta[i]).abs());
                }
                assert!(
                    worst < 1e-8,
                    "St({n},{k}) Log∘Exp != id at scale {scale}: max err {worst:.3e}"
                );
            }
        }
    }

    /// `Log_Y(Y) = 0`: the logarithm of a point to itself is the zero tangent.
    #[test]
    fn log_of_self_is_zero_k2() {
        let st = StiefelManifold::new(2, 5).unwrap();
        let y = Array1::from(vec![
            1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let lg = st.log_map(y.view(), y.view()).unwrap();
        let worst = lg.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
        assert!(worst < 1e-12, "Log_Y(Y) != 0: max = {worst:.3e}");
    }

    /// The recovered logarithm must itself be a tangent vector at `Y`
    /// (`YᵀΔ + ΔᵀY = 0`), and its canonical norm must equal the geodesic
    /// distance — here the norm of the tangent we exponentiated.
    #[test]
    fn log_is_tangent_and_isometric_k2() {
        let st = StiefelManifold::new(2, 4).unwrap();
        let y = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let raw = Array1::from(vec![0.0_f64, -0.2, 0.2, 0.0, 0.25, -0.1, 0.05, 0.15]);
        let delta = st.project_tangent(y.view(), raw.view()).unwrap();
        let target = st.exp_map(y.view(), delta.view()).unwrap();
        let recovered = st.log_map(y.view(), target.view()).unwrap();
        // Tangency: YᵀΔ̂ skew ⇒ projecting it leaves it unchanged.
        let proj = st.project_tangent(y.view(), recovered.view()).unwrap();
        let mut tan_err = 0.0_f64;
        for i in 0..recovered.len() {
            tan_err = tan_err.max((proj[i] - recovered[i]).abs());
        }
        assert!(tan_err < 1e-9, "Log not tangent: max|P Δ̂ − Δ̂| = {tan_err:.3e}");
        // Isometry: ‖Log_Y(Exp_Y(Δ))‖ = ‖Δ‖ under the canonical metric.
        let g = st.metric_tensor(y.view()).unwrap();
        let canon_norm = |d: &Array1<f64>| -> f64 {
            let gd = g.dot(d);
            let mut acc = 0.0_f64;
            for i in 0..d.len() {
                acc += d[i] * gd[i];
            }
            acc.sqrt()
        };
        let d_norm = canon_norm(&delta);
        let r_norm = canon_norm(&recovered);
        assert!(
            (d_norm - r_norm).abs() < 1e-9,
            "geodesic distance not preserved: ‖Δ‖={d_norm:.6}, ‖Δ̂‖={r_norm:.6}"
        );
    }

    #[test]
    fn parallel_transport_k_gt_1_returns_unsupported() {
        let st = StiefelManifold::new(2, 3).unwrap();
        let path = Array2::from_shape_vec((1, 6), vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let v = Array1::from(vec![0.0_f64, -1.0, 1.0, 0.0, 0.0, 0.0]);
        match st.parallel_transport(path.view(), v.view()) {
            Err(GeometryError::Unsupported(_)) => {}
            other => panic!("expected Unsupported for k>1, got {other:?}"),
        }
    }

    #[test]
    fn sectional_curvature_k_gt_1_returns_unsupported() {
        let st = StiefelManifold::new(2, 3).unwrap();
        let y = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let u = Array1::from(vec![0.0_f64, -1.0, 1.0, 0.0, 0.0, 0.0]);
        let v = Array1::from(vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0]);
        match st.sectional_curvature(y.view(), (u.view(), v.view())) {
            Err(GeometryError::Unsupported(_)) => {}
            other => panic!("expected Unsupported for k>1, got {other:?}"),
        }
    }

    #[test]
    fn project_tangent_makes_ytz_skew_symmetric() {
        // Y = [e0, e1] as 3×2 row-major: rows (1,0), (0,1), (0,0)
        let st = StiefelManifold::new(2, 3).unwrap();
        let y = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let v = Array1::from(vec![0.5_f64, 1.0, 0.0, 0.5, 1.0, 0.0]);
        let h = st.project_tangent(y.view(), v.view()).unwrap();
        // YᵀH where Y=[e0,e1] (3×2 standard frame): YᵀH[a,b] = H[a,b],
        // i.e. h[0..4] encodes the 2×2 block. Skew requires diagonal = 0
        // and off-diagonal sum = 0.
        assert!(h[0].abs() < 1e-12, "YᵀH[0,0] = {}", h[0]);
        assert!(h[3].abs() < 1e-12, "YᵀH[1,1] = {}", h[3]);
        assert!((h[1] + h[2]).abs() < 1e-12, "YᵀH not skew: h[1]={}, h[2]={}", h[1], h[2]);
    }

    #[test]
    fn retract_stays_on_stiefel_manifold() {
        // St(2, 4): QR retraction must return a frame with QᵀQ = I₂.
        let st = StiefelManifold::new(2, 4).unwrap();
        // Y = [e0, e1] as 4×2 row-major: [1,0, 0,1, 0,0, 0,0]
        let y = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let delta = Array1::from(vec![0.0_f64, -0.2, 0.2, 0.0, 0.1, 0.0, 0.0, 0.05]);
        let q_flat = st.retract(y.view(), delta.view()).unwrap();
        let n = 4usize;
        let k = 2usize;
        let mut qtq = [[0.0_f64; 2]; 2];
        for r in 0..n {
            for a in 0..k {
                for b in 0..k {
                    qtq[a][b] += q_flat[r * k + a] * q_flat[r * k + b];
                }
            }
        }
        for i in 0..k {
            for j in 0..k {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[i][j] - want).abs() < 1e-12, "QᵀQ[{i},{j}] = {}", qtq[i][j]);
            }
        }
    }

    #[test]
    fn exp_map_k1_sphere_half_pi_rotation() {
        // St(1, 3) = S²: exp at e1 along π/2·e2 reaches e2.
        let st = StiefelManifold::new(1, 3).unwrap();
        let p = Array1::from(vec![1.0_f64, 0.0, 0.0]);
        let v = Array1::from(vec![0.0_f64, std::f64::consts::FRAC_PI_2, 0.0]);
        let q = st.exp_map(p.view(), v.view()).unwrap();
        assert!(q[0].abs() < 1e-12, "q[0] = {}", q[0]);
        assert!((q[1] - 1.0).abs() < 1e-12, "q[1] = {}", q[1]);
        assert!(q[2].abs() < 1e-12, "q[2] = {}", q[2]);
    }
}
