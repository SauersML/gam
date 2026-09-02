use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
    GeometryError, GeometryResult, RiemannianManifold, check_len, cholesky_spd, dot, flatten,
    from_flat, inverse, jacobi_symmetric, spectral_map_spd, spectral_map_symmetric, sym,
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
        use gam_linalg::faer_ndarray::fast_ab;
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
        use gam_linalg::faer_ndarray::fast_ab;
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
        use gam_linalg::faer_ndarray::fast_ab;
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
        use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
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
        use gam_linalg::faer_ndarray::fast_ab;
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

    /// Riemannian gradient under the affine-invariant metric
    /// `⟨U,V⟩_P = tr(P⁻¹U P⁻¹V)`. For a scalar `f` with ambient differential
    /// `E` (so `Df_P[ξ] = ⟨E, ξ⟩ = tr(Eᵀξ)`), the Riesz representative is the
    /// closed form
    ///
    /// ```text
    ///   grad f(P) = P · sym(E) · P,
    /// ```
    ///
    /// which is symmetric (a genuine SPD tangent) and satisfies the defining
    /// relation: for any symmetric tangent `ξ`,
    /// `⟨grad, ξ⟩_P = tr(P⁻¹·P sym(E) P·P⁻¹·ξ) = tr(sym(E) ξ) = tr(Eᵀ ξ)`,
    /// since the antisymmetric part of `E` contracts to zero against symmetric
    /// `ξ`. This is the metric-raising default specialized to the affine metric —
    /// computed directly here to avoid forming the `n²×n²` metric tensor, and to
    /// stay exact. Merely projecting `E` to `sym(E)` ([`Self::project_tangent`]) is the
    /// *Euclidean*-metric gradient and is wrong off the identity (issue #955).
    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        euclidean_grad: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        use gam_linalg::faer_ndarray::fast_ab;
        let p = self.matrix(point)?;
        let e = sym(&from_flat(euclidean_grad, self.n, self.n)?);
        // P · sym(E) · P (dense n×n chain, GPU-dispatched via fast_ab).
        let grad = fast_ab(&fast_ab(&p, &e), &p);
        Ok(flatten(&sym(&grad)))
    }

    /// Analytic vector–Jacobian product of the affine-invariant exponential
    /// [`exp_map`](RiemannianManifold::exp_map), hand-derived via the
    /// Daleckii–Krein theorem.
    ///
    /// The forward map is the composition
    ///
    /// ```text
    ///   U = sym(T),  S = P^{1/2},  S⁻ = P^{-1/2},
    ///   M = S⁻ U S⁻,  E = exp(M),  Y = S E S,
    /// ```
    ///
    /// and every non-linear stage is a primary matrix function of a symmetric
    /// argument, whose Fréchet derivative at `A = Q Λ Qᵀ` is the Daleckii–Krein
    /// divided-difference form `Df(A)[H] = Q (Φ_f ∘ (Qᵀ H Q)) Qᵀ` with
    /// `Φ_f[i,j] = f[λ_i, λ_j]` (first divided difference; `f'(λ_i)` on the
    /// diagonal and for clustered eigenvalues). That map is self-adjoint under
    /// the Frobenius pairing, so each cotangent pulls back through the SAME
    /// divided-difference conjugation, and the product-rule terms of
    /// `Y = S E S`, `M = S⁻ U S⁻` transpose in closed form. The three divided
    /// differences involved are evaluated in cancellation-free closed forms:
    ///
    /// * `exp`: `e^max(a,b)·[-expm1(-|a−b|)]/|a−b|` (`= e^a` at
    ///   equality);
    /// * `√x`: `1/(√a + √b)`;
    /// * `x^{-1/2}`: `−1/(√a·√b·(√a + √b))`;
    ///
    /// so repeated/clustered eigenvalues need no branch beyond the exact
    /// `h → 0` limit of `sinh(h)/h`. The returned pair is
    /// `(∂⟨G, Y⟩/∂point, ∂⟨G, Y⟩/∂tangent)` for the raw flattened inputs; the
    /// `sym` projections of the forward map are their own adjoints and are
    /// applied to both outputs.
    fn exp_map_vjp(
        &self,
        point: ArrayView1<'_, f64>,
        tangent_vec: ArrayView1<'_, f64>,
        grad_output: ArrayView1<'_, f64>,
    ) -> GeometryResult<(Array1<f64>, Array1<f64>)> {
        use gam_linalg::faer_ndarray::fast_ab;
        let m = self.ambient_dim();
        check_len("SPD exp_map_vjp point", point.len(), m)?;
        check_len("SPD exp_map_vjp tangent", tangent_vec.len(), m)?;
        check_len("SPD exp_map_vjp grad", grad_output.len(), m)?;

        // Forward quantities, recomputed from the eigendecompositions the
        // divided-difference pullbacks need anyway.
        let p = self.matrix(point)?;
        let u = sym(&from_flat(tangent_vec, self.n, self.n)?);
        let (p_evals, p_vecs) = jacobi_symmetric(&p)?;
        for &lam in p_evals.iter() {
            if !(lam.is_finite() && lam > 0.0) {
                return Err(GeometryError::InvalidPoint(
                    "SPD eigenvalue is not positive",
                ));
            }
        }
        let sqrt_p = spectral_reconstruct(&p_vecs, &p_evals, f64::sqrt);
        let inv_sqrt_p = spectral_reconstruct(&p_vecs, &p_evals, |x| 1.0 / x.sqrt());
        let middle = sym(&fast_ab(&fast_ab(&inv_sqrt_p, &u), &inv_sqrt_p));
        let (m_evals, m_vecs) = jacobi_symmetric(&middle)?;
        let exp_middle = spectral_reconstruct(&m_vecs, &m_evals, f64::exp);

        // Adjoint of the trailing `flatten(sym(·))`.
        let g_y = sym(&from_flat(grad_output, self.n, self.n)?);

        // Y = S E S: Ḡ_E = S Ḡ_Y S, Ḡ_S = Ḡ_Y S E + E S Ḡ_Y.
        let g_e = fast_ab(&fast_ab(&sqrt_p, &g_y), &sqrt_p);
        let g_s = &fast_ab(&fast_ab(&g_y, &sqrt_p), &exp_middle)
            + &fast_ab(&fast_ab(&exp_middle, &sqrt_p), &g_y);

        // E = exp(M): the Daleckii–Krein map is self-adjoint, so
        // Ḡ_M = Q (Φ_exp ∘ (Qᵀ Ḡ_E Q)) Qᵀ.
        let g_m = daleckii_krein_pullback(&m_vecs, &m_evals, exp_divided_difference, &sym(&g_e));

        // M = S⁻ U S⁻: Ḡ_U = S⁻ Ḡ_M S⁻, Ḡ_{S⁻} = Ḡ_M S⁻ U + U S⁻ Ḡ_M.
        let g_u = fast_ab(&fast_ab(&inv_sqrt_p, &g_m), &inv_sqrt_p);
        let g_s_inv =
            &fast_ab(&fast_ab(&g_m, &inv_sqrt_p), &u) + &fast_ab(&fast_ab(&u, &inv_sqrt_p), &g_m);

        // S = P^{1/2} and S⁻ = P^{-1/2} pull back through their own
        // divided-difference conjugations on P's eigendecomposition.
        let g_p = &daleckii_krein_pullback(&p_vecs, &p_evals, sqrt_divided_difference, &sym(&g_s))
            + &daleckii_krein_pullback(
                &p_vecs,
                &p_evals,
                inv_sqrt_divided_difference,
                &sym(&g_s_inv),
            );

        // Adjoints of the leading `sym` projections of point and tangent.
        Ok((flatten(&sym(&g_p)), flatten(&sym(&g_u))))
    }
}

/// `V · diag(f(λ)) · Vᵀ` from an eigendecomposition already in hand (the VJP
/// needs the factors themselves, so it cannot use `spectral_map_spd`, which
/// re-decomposes internally and discards them).
fn spectral_reconstruct(
    vecs: &Array2<f64>,
    evals: &Array1<f64>,
    f: impl Fn(f64) -> f64,
) -> Array2<f64> {
    use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
    let n = evals.len();
    let mut diag = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        diag[[i, i]] = f(evals[i]);
    }
    fast_abt(&fast_ab(vecs, &diag), vecs)
}

/// Pull a symmetric cotangent `c` back through the Fréchet derivative of a
/// primary matrix function at `A = Q Λ Qᵀ`: the Daleckii–Krein map
/// `H ↦ Q (Φ ∘ (Qᵀ H Q)) Qᵀ` is self-adjoint under the Frobenius pairing
/// (`Φ` is symmetric), so the pullback applies the same conjugation to `c`.
fn daleckii_krein_pullback(
    vecs: &Array2<f64>,
    evals: &Array1<f64>,
    divided_difference: impl Fn(f64, f64) -> f64,
    c: &Array2<f64>,
) -> Array2<f64> {
    use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
    let n = evals.len();
    let mut inner = fast_ab(&fast_atb(vecs, c), vecs);
    for i in 0..n {
        for j in 0..n {
            inner[[i, j]] *= divided_difference(evals[i], evals[j]);
        }
    }
    fast_abt(&fast_ab(vecs, &inner), vecs)
}

/// First divided difference of `exp`: `(e^a − e^b)/(a − b)`. Factoring
/// out the larger exponential gives the cancellation-free form
/// `e^hi·[-expm1(-gap)]/gap`, where `gap = |a−b|`. Besides resolving the
/// clustered limit analytically, this avoids the indeterminate `0·∞` produced
/// by the equivalent midpoint/sinh identity when both eigenvalues are very
/// negative but far apart.
fn exp_divided_difference(a: f64, b: f64) -> f64 {
    if a == b {
        return a.exp();
    }
    let hi = a.max(b);
    let gap = (a - b).abs();
    hi.exp() * (-(-gap).exp_m1() / gap)
}

/// First divided difference of `√x` on the positive axis: the subtraction-free
/// closed form `1/(√a + √b)` (`= 1/(2√a)`, the derivative, at `a = b`).
fn sqrt_divided_difference(a: f64, b: f64) -> f64 {
    1.0 / (a.sqrt() + b.sqrt())
}

/// First divided difference of `x^{-1/2}` on the positive axis:
/// `−1/(√a·√b·(√a + √b))` (`= −1/(2a^{3/2})` at `a = b`), also subtraction-free.
fn inv_sqrt_divided_difference(a: f64, b: f64) -> f64 {
    let (sa, sb) = (a.sqrt(), b.sqrt());
    let (lo, hi) = if sa <= sb { (sa, sb) } else { (sb, sa) };
    -((1.0 / hi) / (hi + lo)) / lo
}

#[cfg(test)]
mod tangent_basis_tests {
    use super::SpdManifold;
    use crate::manifold::RiemannianManifold;
    use ndarray::Array1;

    /// The affine-invariant metric raise must satisfy the defining Riesz
    /// identity and must not collapse to Euclidean tangent projection away
    /// from the identity matrix.
    #[test]
    fn spd_riemannian_gradient_is_affine_metric_riesz_representative() {
        let spd = SpdManifold::new(2);
        let p = Array1::from(vec![2.0, 0.0, 0.0, 1.0]);
        let differential = Array1::from(vec![1.0, 0.0, 0.0, 1.0]);
        let tangent = Array1::from(vec![0.7, 0.2, 0.2, -0.3]);

        let gradient = spd
            .riemannian_gradient(p.view(), differential.view())
            .expect("affine-invariant metric raise");
        let metric = spd.metric_tensor(p.view()).expect("SPD metric tensor");
        let lhs = gradient.dot(&metric.dot(&tangent));
        let rhs = differential.dot(&tangent);
        assert!(
            (lhs - rhs).abs() <= 1.0e-12,
            "Riesz identity failed: g_P(grad, xi)={lhs}, <E, xi>={rhs}"
        );

        let expected = Array1::from(vec![4.0, 0.0, 0.0, 1.0]);
        for (got, want) in gradient.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12);
        }
        let projected = spd
            .project_tangent(p.view(), differential.view())
            .expect("Euclidean tangent projection");
        assert!(
            (&gradient - &projected).dot(&(&gradient - &projected)) > 1.0,
            "affine metric raise unexpectedly equals Euclidean projection"
        );
    }

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

#[cfg(test)]
mod exp_map_vjp_tests {
    use super::{SpdManifold, exp_divided_difference};
    use crate::manifold::RiemannianManifold;
    use ndarray::{Array1, Array2};

    /// Row-major flatten matching `flatten`/`from_flat`.
    fn flat(m: &Array2<f64>) -> Array1<f64> {
        Array1::from_iter(m.iter().copied())
    }

    /// `R diag(d) Rᵀ` with `R` a Givens-style 3-D rotation, giving an SPD
    /// matrix with EXACTLY the prescribed eigenvalues (repeated ones included).
    fn spd_with_eigs(d: [f64; 3], theta: f64, phi: f64) -> Array2<f64> {
        let (c1, s1) = (theta.cos(), theta.sin());
        let (c2, s2) = (phi.cos(), phi.sin());
        let g1 =
            Array2::from_shape_vec((3, 3), vec![c1, -s1, 0.0, s1, c1, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let g2 =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, c2, -s2, 0.0, s2, c2]).unwrap();
        let r = g1.dot(&g2);
        let mut dm = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            dm[[i, i]] = d[i];
        }
        r.dot(&dm).dot(&r.t())
    }

    /// Central finite-difference oracle (TEST-ONLY, per SPEC 2) for the scalar
    /// `f(P, T) = ⟨G, exp_P(T)⟩`: checks the analytic VJP pair against the FD
    /// directional derivative along every symmetric coordinate direction of `P`
    /// and every raw coordinate direction of `T`.
    fn assert_vjp_matches_fd(p: &Array2<f64>, t: &Array2<f64>, g: &Array2<f64>) {
        let spd = SpdManifold::new(3);
        let (pf, tf, gf) = (flat(p), flat(t), flat(g));
        let (grad_p, grad_t) = spd
            .exp_map_vjp(pf.view(), tf.view(), gf.view())
            .expect("SPD exp_map_vjp");
        let scalar = |pv: &Array1<f64>, tv: &Array1<f64>| -> f64 {
            let y = spd.exp_map(pv.view(), tv.view()).expect("exp_map");
            y.dot(&gf)
        };
        let eps = 1.0e-6;
        // Point directions: symmetric (the SPD chart rejects asymmetric points).
        for i in 0..3 {
            for j in i..3 {
                let mut h = Array2::<f64>::zeros((3, 3));
                h[[i, j]] = 1.0;
                h[[j, i]] = 1.0;
                let hf = flat(&h);
                let fd = (scalar(&(&pf + &(&hf * eps)), &tf) - scalar(&(&pf - &(&hf * eps)), &tf))
                    / (2.0 * eps);
                let analytic = grad_p.dot(&hf);
                assert!(
                    (fd - analytic).abs() <= 1.0e-5 * (1.0 + fd.abs()),
                    "grad_point mismatch along sym e({i},{j}): fd {fd:.9e} vs vjp {analytic:.9e}"
                );
            }
        }
        // Tangent directions: raw (the forward symmetrizes internally; the VJP
        // must carry that projection's adjoint).
        for idx in 0..9 {
            let mut hf = Array1::<f64>::zeros(9);
            hf[idx] = 1.0;
            let fd = (scalar(&pf, &(&tf + &(&hf * eps))) - scalar(&pf, &(&tf - &(&hf * eps))))
                / (2.0 * eps);
            let analytic = grad_t.dot(&hf);
            assert!(
                (fd - analytic).abs() <= 1.0e-5 * (1.0 + fd.abs()),
                "grad_tangent mismatch along e{idx}: fd {fd:.9e} vs vjp {analytic:.9e}"
            );
        }
    }

    #[test]
    fn spd_exp_map_vjp_matches_fd_generic_spectrum() {
        let p = spd_with_eigs([3.0, 1.2, 0.4], 0.7, 1.1);
        let t =
            Array2::from_shape_vec((3, 3), vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2, -0.3, 0.6, 0.1])
                .unwrap();
        let g =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.4, -0.3, 0.2, -0.8, 0.5, 0.7, -0.1, 0.9])
                .unwrap();
        assert_vjp_matches_fd(&p, &t, &g);
    }

    #[test]
    fn spd_exp_map_vjp_matches_fd_clustered_point_spectrum() {
        // Exactly repeated eigenvalues of P: the √/x^{-1/2} divided differences
        // must hit their analytic diagonal limit, not a 0/0 subtraction.
        let p = spd_with_eigs([2.0, 2.0, 0.5], 0.9, 0.3);
        let t =
            Array2::from_shape_vec((3, 3), vec![0.2, 0.1, -0.3, 0.1, -0.1, 0.4, -0.3, 0.4, 0.3])
                .unwrap();
        let g =
            Array2::from_shape_vec((3, 3), vec![0.5, -0.6, 0.2, -0.6, 0.3, 0.8, 0.2, 0.8, -0.4])
                .unwrap();
        assert_vjp_matches_fd(&p, &t, &g);
    }

    #[test]
    fn spd_exp_map_vjp_matches_fd_degenerate_exp_spectrum() {
        // T ∝ P makes the whitened middle M = c·I: EVERY eigenvalue of the exp
        // stage coincides, exercising the exp divided-difference limit e^a.
        let p = spd_with_eigs([1.5, 0.8, 2.5], 0.4, 1.3);
        let t = &p * 0.35;
        let g =
            Array2::from_shape_vec((3, 3), vec![0.9, 0.1, -0.2, 0.1, -0.5, 0.3, -0.2, 0.3, 0.6])
                .unwrap();
        assert_vjp_matches_fd(&p, &t, &g);
    }

    #[test]
    fn spd_exp_map_vjp_zero_tangent_reduces_to_identity_pullback() {
        // At T = 0 the exponential is exp_P(0) = P, so grad_point must be the
        // symmetrized cotangent exactly and grad_tangent must equal the
        // whitened-DK pullback (finite, symmetric).
        let spd = SpdManifold::new(3);
        let p = spd_with_eigs([2.0, 1.0, 0.5], 0.2, 0.8);
        let g = Array2::from_shape_vec((3, 3), vec![1.0, 0.3, 0.0, 0.3, -0.7, 0.2, 0.0, 0.2, 0.4])
            .unwrap();
        let zeros = Array1::<f64>::zeros(9);
        let (grad_p, grad_t) = spd
            .exp_map_vjp(flat(&p).view(), zeros.view(), flat(&g).view())
            .expect("VJP at zero tangent");
        let gs = crate::manifold::sym(&g);
        for (a, b) in grad_p.iter().zip(gs.iter()) {
            assert!(
                (a - b).abs() <= 1.0e-12,
                "grad_point at T=0 must be sym(G): {a} vs {b}"
            );
        }
        for (a, b) in grad_t.iter().zip(gs.iter()) {
            assert!(
                (a - b).abs() <= 1.0e-12,
                "grad_tangent at T=0 must be sym(G): {a} vs {b}"
            );
        }
    }

    #[test]
    fn exp_divided_difference_stays_finite_across_underflow_range() {
        // The midpoint/sinh identity is mathematically equivalent but evaluates
        // this case as `exp(-750.5) * sinh(749.5) = 0 * inf = NaN`.
        let got = exp_divided_difference(-1.0, -1500.0);
        let expected = (-1.0_f64).exp() / 1499.0;
        assert!(got.is_finite());
        assert!((got - expected).abs() <= f64::EPSILON * expected);
    }
}

#[cfg(test)]
mod parallel_transport_tests {
    use super::SpdManifold;
    use crate::manifold::{RiemannianManifold, from_flat, sym};
    use ndarray::{Array1, Array2};

    /// `R(θ) diag(a,b) R(θ)ᵀ` as a flat row-major 2×2 SPD point.
    fn rotated_diag(theta: f64, a: f64, b: f64) -> Array1<f64> {
        let (c, s) = (theta.cos(), theta.sin());
        let m00 = c * c * a + s * s * b;
        let m01 = c * s * (a - b);
        let m11 = s * s * a + c * c * b;
        Array1::from(vec![m00, m01, m01, m11])
    }

    /// Non-commuting fixture: `P` and `Q` have distinct eigenbases, so the
    /// affine-invariant geodesic between them genuinely curves — not the
    /// trivial commuting case, where the transport congruence collapses to a
    /// diagonal rescaling and cannot exercise the general formula.
    fn fixture() -> (SpdManifold, Array1<f64>, Array1<f64>) {
        let spd = SpdManifold::new(2);
        let p = rotated_diag(0.3, 3.0, 0.5);
        let q = rotated_diag(-0.5, 1.2, 4.0);
        (spd, p, q)
    }

    /// Stack two flat `n×n` points into the `2×n²` path `parallel_transport`
    /// reads its endpoints from (only `point_along.row(0)` and the last row
    /// matter — see [`SpdManifold::parallel_transport`]).
    fn path2(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((2, a.len()));
        for (col, &x) in a.iter().enumerate() {
            m[[0, col]] = x;
        }
        for (col, &x) in b.iter().enumerate() {
            m[[1, col]] = x;
        }
        m
    }

    /// Parallel transport under the Levi-Civita connection is, by
    /// definition, a linear ISOMETRY between tangent spaces:
    /// `⟨Γ(U), Γ(V)⟩_Q = ⟨U, V⟩_P` for every pair of tangents `U, V`. This is
    /// the defining property of the affine-invariant congruence
    /// `Γ(U) = A U Aᵀ`, `A = (Q P⁻¹)^{1/2}`, implemented above, which had no
    /// direct test coverage in this file (unlike, e.g.,
    /// `constant_curvature.rs`'s `parallel_transport_preserves_riemannian_norm`).
    #[test]
    fn parallel_transport_preserves_affine_inner_product() {
        let (spd, p, q) = fixture();
        let path = path2(&p, &q);
        let u = Array1::from(vec![1.0, 0.4, 0.4, -0.7]);
        let v = Array1::from(vec![-0.3, 0.9, 0.9, 1.6]);

        let tu = spd.parallel_transport(path.view(), u.view()).expect("Γ(U)");
        let tv = spd.parallel_transport(path.view(), v.view()).expect("Γ(V)");

        let pm = spd.matrix(p.view()).expect("P");
        let qm = spd.matrix(q.view()).expect("Q");
        let um = sym(&from_flat(u.view(), 2, 2).expect("U"));
        let vm = sym(&from_flat(v.view(), 2, 2).expect("V"));
        let tum = sym(&from_flat(tu.view(), 2, 2).expect("ΓU"));
        let tvm = sym(&from_flat(tv.view(), 2, 2).expect("ΓV"));

        let before = spd.affine_inner(&pm, &um, &vm).expect("⟨U,V⟩_P");
        let after = spd.affine_inner(&qm, &tum, &tvm).expect("⟨ΓU,ΓV⟩_Q");
        assert!(
            (before - after).abs() <= 1e-10 * before.abs().max(1.0),
            "parallel transport is not an isometry: ⟨U,V⟩_P={before:.12e}, ⟨ΓU,ΓV⟩_Q={after:.12e}"
        );
    }

    /// Manifold-agnostic sign check: transporting the initial velocity of
    /// the `P→Q` geodesic gives the negative of the `Q→P` geodesic's initial
    /// velocity, `Γ_{P→Q}(log_P Q) = −log_Q P` — the reverse-parametrized
    /// geodesic runs backward through the same tangent line. This is exactly
    /// the kind of sign/order error the affine-metric formula above is
    /// prone to (see the `#955`/`#693` regression comments elsewhere in this
    /// file for the class of bug), and was likewise untested.
    #[test]
    fn parallel_transport_matches_geodesic_velocity_identity() {
        let (spd, p, q) = fixture();
        let forward = path2(&p, &q);
        let v_p_to_q = spd.log_map(p.view(), q.view()).expect("log_P(Q)");
        let v_q_to_p = spd.log_map(q.view(), p.view()).expect("log_Q(P)");

        let transported = spd
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

    /// Transporting forward `P→Q` and then back `Q→P` along the same
    /// geodesic must recover the original tangent exactly (the two
    /// congruence operators `A_{P→Q}` and `A_{Q→P}` are mutual inverses).
    #[test]
    fn parallel_transport_round_trip_is_identity() {
        let (spd, p, q) = fixture();
        let forward = path2(&p, &q);
        let backward = path2(&q, &p);
        let u = Array1::from(vec![0.6, -0.2, -0.2, 1.1]);

        let out = spd
            .parallel_transport(forward.view(), u.view())
            .expect("Γ_{P→Q}(U)");
        let back = spd
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

#[cfg(test)]
mod christoffel_tests {
    use super::SpdManifold;
    use crate::manifold::{RiemannianManifold, flatten, from_flat};
    use ndarray::{Array1, Array2};

    /// Symmetric basis of `n×n` symmetric matrices, dimension `n(n+1)/2`:
    /// `E_ii = e_i e_iᵀ`, `E_ij (i<j) = e_i e_jᵀ + e_j e_iᵀ`. Perturbing the
    /// base point along these directions keeps it symmetric (unlike a raw
    /// single-entry ambient perturbation, which `SpdManifold::matrix` would
    /// reject as off-manifold), so this is a genuine local chart.
    fn symmetric_basis(n: usize) -> Vec<Array2<f64>> {
        let mut basis = Vec::with_capacity(n * (n + 1) / 2);
        for i in 0..n {
            let mut m = Array2::<f64>::zeros((n, n));
            m[[i, i]] = 1.0;
            basis.push(m);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let mut m = Array2::<f64>::zeros((n, n));
                m[[i, j]] = 1.0;
                m[[j, i]] = 1.0;
                basis.push(m);
            }
        }
        basis
    }

    /// A fixed, genuinely non-diagonal SPD base point (small, distinct
    /// off-diagonal entries on top of a well-separated diagonal), so the
    /// check exercises the general affine-invariant tensor rather than the
    /// degenerate identity/diagonal case.
    fn base_point(n: usize) -> Array2<f64> {
        let mut p = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            p[[i, i]] = 1.0 + i as f64;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.05 * (i as f64 + 1.0) - 0.03 * (j as f64 + 1.0) + 0.1;
                p[[i, j]] = v;
                p[[j, i]] = v;
            }
        }
        p
    }

    /// `⟨Γ(∂_a,∂_b), ∂_c⟩ = ½(∂_a g_{bc} + ∂_b g_{ac} − ∂_c g_{ab})` — the
    /// Levi-Civita identity for coordinate vector fields (zero Lie bracket),
    /// lowered by the metric. `christoffel_symbols` must be the connection
    /// generated by `metric_tensor`, not merely an independently hand-derived
    /// formula that happens to look right. `constant_curvature.rs`'s
    /// `christoffel_matches_fd_of_metric` pins the same identity for the
    /// conformal (diagonal) metric; this generalizes it to SPD's full
    /// non-diagonal ambient tensor along a genuine symmetric chart, which had
    /// no test anywhere — every existing test in this file exercises
    /// `exp_map`/`log_map`/`parallel_transport`/the Fréchet-mean solver, none
    /// of `christoffel_symbols` or `sectional_curvature`.
    #[test]
    fn christoffel_matches_fd_of_metric_on_symmetric_chart() {
        let n = 3;
        let m = SpdManifold::new(n);
        let p0 = base_point(n);
        let basis = symmetric_basis(n);
        let basis_flat: Vec<Array1<f64>> = basis.iter().map(flatten).collect();
        let d = basis.len();
        assert_eq!(d, n * (n + 1) / 2);
        let ambient = m.ambient_dim();

        let point_at = |x: &[f64]| -> Array1<f64> {
            let mut p = p0.clone();
            for (a, &xa) in x.iter().enumerate() {
                if xa != 0.0 {
                    p = &p + &(&basis[a] * xa);
                }
            }
            flatten(&p)
        };
        let contract = |g: &Array2<f64>, b: usize, c: usize| -> f64 {
            basis_flat[b].dot(&g.dot(&basis_flat[c]))
        };

        let x0 = vec![0.0_f64; d];
        let h = 1e-6;

        // ∂_a g_{bc} via central finite differences of `metric_tensor`,
        // caching one `metric_tensor` evaluation per perturbed point rather
        // than recomputing it inside the `(b, c)` loop.
        let mut dg = vec![vec![vec![0.0_f64; d]; d]; d]; // dg[a][b][c]
        for a in 0..d {
            let mut xp = x0.clone();
            xp[a] += h;
            let mut xn = x0.clone();
            xn[a] -= h;
            let gp = m.metric_tensor(point_at(&xp).view()).expect("G(x+h e_a)");
            let gn = m.metric_tensor(point_at(&xn).view()).expect("G(x-h e_a)");
            for b in 0..d {
                for c in 0..d {
                    dg[a][b][c] = (contract(&gp, b, c) - contract(&gn, b, c)) / (2.0 * h);
                }
            }
        }

        let point0 = point_at(&x0);
        let gamma = m.christoffel_symbols(point0.view()).expect("Γ tensor");
        let connection_matrix = |a: usize, b: usize| -> Array2<f64> {
            // Γ(E_a, E_b) as an ambient n×n matrix, contracted out of the
            // full ambient-indexed tensor `gamma[out][[in_a, in_b]]`.
            let mut gamma_vec = Array1::<f64>::zeros(ambient);
            for out in 0..ambient {
                let mut acc = 0.0;
                for p_idx in 0..ambient {
                    let coeff = basis_flat[a][p_idx];
                    if coeff == 0.0 {
                        continue;
                    }
                    for q_idx in 0..ambient {
                        acc += coeff * gamma[out][[p_idx, q_idx]] * basis_flat[b][q_idx];
                    }
                }
                gamma_vec[out] = acc;
            }
            from_flat(gamma_vec.view(), n, n).expect("Γ(E_a,E_b) as n×n")
        };

        for a in 0..d {
            for b in 0..d {
                let gamma_mat = connection_matrix(a, b);
                for c in 0..d {
                    let lhs = m
                        .affine_inner(&p0, &gamma_mat, &basis[c])
                        .expect("⟨Γ(E_a,E_b), E_c⟩");
                    let rhs = 0.5 * (dg[a][b][c] + dg[b][a][c] - dg[c][a][b]);
                    assert!(
                        (lhs - rhs).abs() <= 1e-6 * rhs.abs().max(1.0),
                        "a={a} b={b} c={c}: ⟨Γ,E_c⟩_analytic={lhs:.10e} vs FD-of-metric={rhs:.10e}"
                    );
                }
            }
        }
    }

    /// Two commuting symmetric directions at a *diagonal* base point (e.g.
    /// `diag(1,0)` and `diag(0,1)`) span a totally geodesic flat torus: the
    /// affine-invariant SPD geometry restricted to simultaneously
    /// diagonalizable matrices is exactly Euclidean in log-coordinates.
    /// `sectional_curvature` on that plane must be (numerically) zero — the
    /// one closed-form value the whitened-commutator formula
    /// `-¼‖[Ã,B̃]‖²/denom` predicts trivially (`[Ã,B̃] = 0`) and that every
    /// other test in this file leaves unchecked.
    #[test]
    fn sectional_curvature_vanishes_on_commuting_diagonal_plane() {
        let m = SpdManifold::new(2);
        let p = Array1::from(vec![2.0_f64, 0.0, 0.0, 3.0]); // diag(2,3)
        let u = Array1::from(vec![1.0_f64, 0.0, 0.0, 0.0]); // diag(1,0)
        let v = Array1::from(vec![0.0_f64, 0.0, 0.0, 1.0]); // diag(0,1)
        let k = m
            .sectional_curvature(p.view(), (u.view(), v.view()))
            .expect("sectional curvature on commuting plane");
        assert!(
            k.abs() <= 1e-12,
            "expected flat commuting plane, got κ={k:.3e}"
        );
    }

    /// The affine-invariant SPD metric is a symmetric space of non-compact
    /// type (`GL(n)/O(n)`), whose sectional curvature is non-positive
    /// everywhere — never spuriously positive from a sign slip in the
    /// commutator/denominator formula. Check on a genuinely non-commuting
    /// plane (distinct eigenbases), where curvature is strictly negative.
    #[test]
    fn sectional_curvature_is_nonpositive_on_noncommuting_plane() {
        let m = SpdManifold::new(2);
        let p = Array1::from(vec![1.0_f64, 0.0, 0.0, 1.0]); // identity
        let u = Array1::from(vec![1.0_f64, 0.0, 0.0, -1.0]); // diag(1,-1)
        let v = Array1::from(vec![0.0_f64, 1.0, 1.0, 0.0]); // off-diagonal
        let k = m
            .sectional_curvature(p.view(), (u.view(), v.view()))
            .expect("sectional curvature on non-commuting plane");
        assert!(
            k < -1e-6,
            "expected strictly negative curvature, got κ={k:.3e}"
        );
    }
}
