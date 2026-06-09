use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::geometry::manifold::{
    GEOMETRY_EPS, GeometryError, GeometryResult, RiemannianManifold, check_len, flatten, from_flat,
    identity, matrix_exp, qr_thin, sym, tangent_basis_metric_orthonormal,
};
use crate::geometry::sphere::SphereManifold;

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
        // YᵀΔ (k×n · n×k) and Y·A (n×k · k×k) both carry the large ambient
        // dimension n, so route them through the GPU-dispatched fast_atb/fast_ab
        // shims; small frames stay on faer.
        use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
        let a = fast_atb(&y, &delta); // k×k skew-symmetric
        let normal = &delta - &fast_ab(&y, &a); // (I − YYᵀ)Δ
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
        // exponential survive against the [[I_k], [0]] selector, so this splits
        // into Y·exp_block[0..k, 0..k] + Q·exp_block[k.., 0..k] — two n×k · k×k
        // products carrying the large ambient dimension n, GPU-dispatched via
        // fast_ab.
        let top = exp_block.slice(ndarray::s![0..self.k, 0..self.k]);
        let bot = exp_block.slice(ndarray::s![self.k..two_k, 0..self.k]);
        let result = &fast_ab(&y, &top) + &fast_ab(&q, &bot);
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
        // Y Yᵀ (n×k · k×n) carries the large ambient dimension n, GPU-dispatched
        // via fast_abt.
        let yyt = crate::linalg::faer_ndarray::fast_abt(&y, &y);
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

    fn christoffel_symbols(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Vec<Array2<f64>>> {
        check_len("Stiefel Christoffel point", point.len(), self.ambient_dim())?;
        // The Stiefel manifold under the canonical metric is curved (its
        // geodesics Y(t)=Y·exp(tA) have ambient acceleration YA²≠0), so a
        // zero ambient Christoffel tensor would assert a false flat geometry.
        // No flat global chart exists; refuse rather than mislead callers.
        Err(GeometryError::Unsupported(
            "Christoffel symbols of the embedded Stiefel manifold require a local chart",
        ))
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
        use crate::linalg::faer_ndarray::{fast_ab, fast_atb};
        let y = from_flat(point, self.n, self.k)?;
        let z = from_flat(vec, self.n, self.k)?;
        // Tangent projection z − Y·sym(Yᵀz): YᵀZ (k×n · n×k) and Y·S (n×k · k×k)
        // both carry the large ambient dimension n, GPU-dispatched via
        // fast_atb/fast_ab.
        let correction = fast_ab(&y, &sym(&fast_atb(&y, &z)));
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

    /// Reverse-mode (vector–Jacobian product) of [`exp_map`](Self::exp_map).
    ///
    /// Given the output cotangent `Ḡ = ∂L/∂result` (n×k), returns
    /// `(∂L/∂point, ∂L/∂tangent_vec)` flattened. The derivation is the exact
    /// adjoint of the seven forward steps (project → A → normal → thin-QR →
    /// block → matrix-exp → assemble), with the matrix-exponential adjoint
    /// obtained from the Mathias augmented identity
    /// `adj(dexp_B)·M̄ = dexp_{Bᵀ}(M̄)` and the thin-QR adjoint from the
    /// standard `copyltu` formula (`Q` full column rank, `n ≥ k`). No
    /// approximations: every intermediate is recomputed exactly as the forward
    /// produced it.
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
        let k = self.k;
        let two_k = 2 * k;

        // ── Recompute the forward intermediates exactly as `exp_map` does. ──
        // Every dense product below either contracts or carries the large
        // ambient dimension n (k×n · n×k or n×k · k×k); route them all through
        // the GPU-dispatched fast_ab/fast_atb/fast_abt shims.
        use crate::linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
        let y = from_flat(point, self.n, k)?;
        let z = from_flat(tangent_vec, self.n, k)?; // raw (unprojected) input
        let s_proj = sym(&fast_atb(&y, &z)); // S = sym(Yᵀz)
        let delta = &z - &fast_ab(&y, &s_proj); // Δ = z − Y·S
        let a = fast_atb(&y, &delta); // A = YᵀΔ (skew)
        let normal = &delta - &fast_ab(&y, &a); // (I − YYᵀ)Δ
        let (q, r) = qr_thin(&normal); // n×k, k×k upper-triangular

        let mut block = Array2::<f64>::zeros((two_k, two_k));
        for i in 0..k {
            for j in 0..k {
                block[[i, j]] = a[[i, j]];
                block[[i, k + j]] = -r[[j, i]];
                block[[k + i, j]] = r[[i, j]];
            }
        }
        let exp_block = matrix_exp(&block)?;

        let grad = from_flat(grad_output, self.n, k)?; // Ḡ (n×k)

        // ── Step 7 (assemble): result = Y·M_tl + Q·M_bl. ──
        // M_tl = exp_block[0:k, 0:k], M_bl = exp_block[k:2k, 0:k].
        let m_tl = exp_block.slice(ndarray::s![0..k, 0..k]).to_owned();
        let m_bl = exp_block.slice(ndarray::s![k..two_k, 0..k]).to_owned();
        let mut y_bar = fast_abt(&grad, &m_tl); // Ȳ += Ḡ·M_tlᵀ
        let q_bar = fast_abt(&grad, &m_bl); // Q̄ = Ḡ·M_blᵀ

        // M̄ (2k×2k): top-left = Yᵀ·Ḡ, bottom-left = Qᵀ·Ḡ, rest zero.
        let mut m_bar = Array2::<f64>::zeros((two_k, two_k));
        let yt_g = fast_atb(&y, &grad);
        let qt_g = fast_atb(&q, &grad);
        for i in 0..k {
            for j in 0..k {
                m_bar[[i, j]] = yt_g[[i, j]];
                m_bar[[k + i, j]] = qt_g[[i, j]];
            }
        }

        // ── Step 6 (matrix-exp): B̄ = adjoint of dexp at B applied to M̄. ──
        let b_bar = matrix_exp_vjp(&block, &m_bar)?;

        // ── Step 5 (block assembly B = [[A, −Rᵀ], [R, 0]]). ──
        let mut a_bar = b_bar.slice(ndarray::s![0..k, 0..k]).to_owned();
        let mut r_bar = b_bar.slice(ndarray::s![k..two_k, 0..k]).to_owned();
        // R̄ += −(B̄[0:k, k:2k])ᵀ.
        let br_tr = b_bar.slice(ndarray::s![0..k, k..two_k]).to_owned();
        for i in 0..k {
            for j in 0..k {
                r_bar[[i, j]] -= br_tr[[j, i]];
            }
        }

        // ── Step 4 (thin-QR): normal̄ from (Q̄, R̄). ──
        let normal_bar = qr_thin_vjp(&q, &r, &q_bar, &r_bar)?;

        // ── Step 3 (normal = Δ − Y·A). ──
        let mut delta_bar = normal_bar.clone();
        y_bar = y_bar - &fast_abt(&normal_bar, &a); // Ȳ += −normal̄·Aᵀ
        a_bar = a_bar - &fast_atb(&y, &normal_bar); // Ā += −(Yᵀ·normal̄)

        // ── Step 2 (A = Yᵀ·Δ). ──
        y_bar = y_bar + &fast_abt(&delta, &a_bar); // Ȳ += Δ·Āᵀ
        delta_bar = delta_bar + &fast_ab(&y, &a_bar); // Δ̄ += Y·Ā

        // ── Step 1 (Δ = z − Y·sym(Yᵀz)). ──
        // z̄ = Δ̄ − Y·sym(Yᵀ·Δ̄)
        // Ȳ += −Δ̄·S − z·sym(Yᵀ·Δ̄)
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

/// Adjoint (VJP) of the thin/compact QR factorization `normal = Q·R` for a
/// full-column-rank `n×k` input (`n ≥ k`, `R` invertible upper-triangular).
/// Given the output cotangents `Q̄` and `R̄`, returns `normal̄`:
///
/// ```text
///   M      = R·R̄ᵀ − Q̄ᵀ·Q
///   normal̄ = (Q̄ + Q·copyltu(M)) · R⁻ᵀ
/// ```
///
/// where `copyltu(M)` is the symmetric matrix built from the lower triangle of
/// `M` (lower triangle incl. diagonal, plus the strictly-lower part reflected
/// into the upper triangle). The trailing `R⁻ᵀ` is realized by forward
/// substitution solving `normal̄·Rᵀ = RHS` (`R` upper-triangular).
///
/// This is the Walter / Seeger formula for the thin-QR adjoint. The asymmetric
/// pairing (`R·R̄ᵀ`, not `Rᵀ·R̄`) matters because `copyltu` extracts the lower
/// triangle: `copyltu(M) ≠ copyltu(Mᵀ)` in general, so transposing the factors
/// produces a genuinely different symmetric matrix and the wrong `normal̄`.
fn qr_thin_vjp(
    q: &Array2<f64>,
    r: &Array2<f64>,
    q_bar: &Array2<f64>,
    r_bar: &Array2<f64>,
) -> GeometryResult<Array2<f64>> {
    let k = r.nrows();
    // Mqr = R·R̄ᵀ − Q̄ᵀ·Q  (k×k).
    use crate::linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
    let mqr = fast_abt(r, r_bar) - fast_atb(q_bar, q);
    // copyltu: symmetric matrix from the lower triangle of Mqr.
    let mut sym_low = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            if i > j {
                sym_low[[i, j]] = mqr[[i, j]];
                sym_low[[j, i]] = mqr[[i, j]];
            } else if i == j {
                sym_low[[i, j]] = mqr[[i, j]];
            }
        }
    }
    // RHS = Q̄ + Q·copyltu(Mqr)  (n×k).
    let rhs = q_bar + &fast_ab(q, &sym_low);
    // Solve normal̄·Rᵀ = RHS for normal̄ (so normal̄ = RHS·R⁻ᵀ). `R` is upper
    // triangular, hence `Rᵀ` is lower triangular, so column `j` of the product
    // couples columns `l ≥ j` of normal̄:
    //   (normal̄·Rᵀ)[row, j] = Σ_{l ≥ j} normal̄[row, l]·R[j, l] = RHS[row, j].
    // This is a back substitution in the column index `j` (descending):
    //   normal̄[row, j]·R[j, j] = RHS[row, j] − Σ_{l > j} normal̄[row, l]·R[j, l].
    let n = rhs.nrows();
    let mut out = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        for j in (0..k).rev() {
            let mut acc = rhs[[row, j]];
            for l in (j + 1)..k {
                acc -= out[[row, l]] * r[[j, l]];
            }
            let diag = r[[j, j]];
            if diag.abs() <= GEOMETRY_EPS {
                return Err(GeometryError::Singular(
                    "qr_thin_vjp requires full-column-rank input (R invertible)",
                ));
            }
            out[[row, j]] = acc / diag;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tangent_basis_tests {
    use super::StiefelManifold;
    use crate::geometry::manifold::RiemannianManifold;
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
