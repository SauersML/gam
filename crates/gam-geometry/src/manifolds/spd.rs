use gam_optimize::{BacktrackConfig, armijo_roundoff_cushion, backtracking_line_search};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{
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
    /// stay exact. Merely projecting `E` to `sym(E)` ([`project_tangent`]) is the
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

/// Squared metric norm `‖v‖²_P = vᵀ G(P) v = tr(P⁻¹ V P⁻¹ V)` of the (already
/// symmetric) flat tangent vector `v` at base point `P`, computed without
/// forming the `n²×n²` Kronecker metric. `pinv = P⁻¹`.
fn affine_sq_norm(n: usize, pinv: &Array2<f64>, v: ArrayView1<'_, f64>) -> GeometryResult<f64> {
    use gam_linalg::faer_ndarray::fast_ab;
    let vm = sym(&from_flat(v, n, n)?);
    // tr(P⁻¹ V P⁻¹ V).
    let a = fast_ab(&fast_ab(pinv, &vm), &fast_ab(pinv, &vm));
    let mut trace = 0.0_f64;
    for i in 0..n {
        trace += a[[i, i]];
    }
    Ok(trace.max(0.0))
}

/// Weighted Fréchet / Karcher mean of SPD matrices under the affine-invariant
/// metric: the unique minimizer of the dispersion `V(P) = Σ_i w_i d²(P, X_i)`
/// on this Hadamard (nonpositively-curved) manifold.
///
/// `points` is `M × n²` (each row a row-major flattened `n×n` SPD matrix);
/// `weights` defaults to uniform `1/M`. Returns the flattened `n×n` mean.
///
/// The iteration is **Riemannian gradient descent** of `V` along the
/// affine-invariant tangent direction `ξ(P) = Σ_i w_i log_P(X_i)` (which is
/// `−½ grad V(P)`), with a geodesic step `P ← exp_P(t·ξ)`. The full Karcher
/// step `t = 1` is the natural fixed point and converges for well-clustered
/// data; backtracking on `t` is retained as an **overshoot safeguard** because
/// `V` is only `1`-strongly but not globally `1`-smoothly geodesically convex —
/// for widely-spread inputs `Hess V` can carry eigenvalues `> 4`, along which
/// the step-½ gradient move `t = 1` would *diverge*. The backtracking restores
/// monotone descent there.
///
/// Two numerical subtleties make a naive Armijo-on-`V` line search stall above
/// the requested tolerance, and both are handled here:
///
///  * **Round-off floor of `V`.** Near the minimizer `V` is flat to machine
///    precision: the true decrease per step is `O(‖ξ‖²)`, which underflows the
///    `O(ε·V)` round-off of evaluating `V` once `‖ξ‖ ≲ √ε`. A strict
///    sufficient-decrease test then rejects the (perfectly good) Karcher step
///    and the residual stalls at `≈ √ε ≈ 1e-7`. We add a round-off cushion
///    `f_tol = 8·ε·(1+|V|)` to the Armijo test, so far from the optimum it is
///    ordinary sufficient decrease (Zoutendijk convergence) and near the
///    optimum it merely forbids an *increase* beyond round-off — letting the
///    convergent unit step drive `‖ξ‖_P` well below `√ε`.
///  * **Round-off floor of `ξ`.** `ξ` is a cancelling sum of `O(1)` log-maps,
///    so its own evaluation floor is `O(ε)`; for some data the smallest
///    attainable `‖ξ‖_P` sits just above a very tight `tol`. No gradient method
///    can push the residual below the round-off of its own gradient, so once
///    the residual stops improving by more than `STALL_REL` for `STALL_PATIENCE`
///    consecutive steps we accept the best iterate as stationary to the
///    achievable precision rather than spuriously erroring.
///
/// Returns `Ok` with the least-residual iterate once the residual reaches
/// `tol` or stalls (no further `STALL_REL`-relative progress for
/// `STALL_PATIENCE` steps / no step decreases `V` within round-off). The
/// stall exit is the correct terminal state of a gradient method: the returned
/// point minimizes the dispersion to the achievable precision. `Err` is
/// reserved for a genuine budget shortfall — `max_iter` exhausted while the
/// residual is still making real first-order progress.
///
/// Caveat (first-order rate): convergence is linear with a rate set by the
/// conditioning of `Hess V`. For well- to moderately-conditioned inputs the
/// residual reaches its `O(ε)`–`√ε` numerical floor (far below any sane `tol`
/// gate); for *extremely* ill-conditioned spreads (eigenvalue ratios `≫ 1e3`
/// across non-commuting samples, where `Hess V` eigenvalues are `≫ 4`) the
/// linear rate can leave a larger residual within `max_iter` even though the
/// dispersion itself is minimized — the known limit of a first-order Karcher
/// iteration, which a second-order (Newton/trust-region) scheme would remove.
pub fn spd_frechet_mean(
    n: usize,
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    tol: f64,
    max_iter: usize,
) -> GeometryResult<Array1<f64>> {
    let ambient = n * n;
    let (m, cols) = points.dim();
    if m == 0 || cols != ambient {
        return Err(GeometryError::InvalidPoint(
            "SPD Fréchet mean: points must be M×n² with M ≥ 1",
        ));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(GeometryError::InvalidPoint(
            "SPD Fréchet mean tolerance must be finite and positive",
        ));
    }
    let spd = SpdManifold::new(n);
    let w = crate::normalize_weights(m, weights)
        .map_err(|_| GeometryError::InvalidPoint("SPD Fréchet mean: invalid weights"))?;

    // Owned flat samples (each validated as an SPD point on first log_map use).
    let samples: Vec<Array1<f64>> = (0..m).map(|i| points.row(i).to_owned()).collect();

    // Weighted dispersion V(P) = Σ_i w_i ‖log_P(X_i)‖²_P at flat base `p`.
    let dispersion = |p: ArrayView1<'_, f64>| -> GeometryResult<f64> {
        let pm = spd.matrix(p)?;
        let pinv = inverse(&pm)?;
        let mut acc = 0.0_f64;
        for (i, x) in samples.iter().enumerate() {
            let lg = spd.log_map(p, x.view())?;
            acc += w[i] * affine_sq_norm(n, &pinv, lg.view())?;
        }
        Ok(acc)
    };

    // Initialize at the weighted Euclidean mean of the samples: a symmetric SPD
    // point (positive combination of SPD matrices), independent of sample order.
    let mut p = Array1::<f64>::zeros(ambient);
    for (i, x) in samples.iter().enumerate() {
        p.scaled_add(w[i], x);
    }
    p = flatten(&sym(&from_flat(p.view(), n, n)?));

    let mut f_cur = dispersion(p.view())?;

    // Best (smallest-residual) iterate seen, returned if the residual stalls at
    // its numerical floor below the reach of `tol`.
    let mut best_p = p.clone();
    let mut best_grad = f64::INFINITY;
    // Consecutive steps that failed to improve the residual by a meaningful
    // relative margin. `STALL_REL` must sit above the residual's round-off
    // oscillation at the floor (empirically `< 1e-3`) yet well below any
    // genuine linear-convergence rate, so a real descent never trips it.
    const STALL_REL: f64 = 5.0e-3;
    const STALL_PATIENCE: usize = 10;
    let mut stall = 0_usize;
    // Armijo sufficient-decrease parameter c₁ (`1e-4`), the backtracking-halving
    // cap (`t = 1` unit Karcher step down to `t = 2⁻⁶⁰ ≈ 1e-18`), and the
    // round-off cushion `8·ε·(1+|f|)` all live in `gam_optimize` now — this loop
    // routes through the shared `backtracking_line_search` primitive
    // (`BacktrackConfig::default()` supplies `t₀ = 1`, factor `0.5`, 60 steps)
    // and the shared `armijo_roundoff_cushion` helper.
    const ARMIJO_C1: f64 = gam_optimize::constants::ARMIJO_C1;

    for _ in 0..max_iter {
        // Riemannian descent direction ξ = Σ_i w_i log_P(X_i) (= −½ grad V).
        let pm = spd.matrix(p.view())?;
        let pinv = inverse(&pm)?;
        let mut xi = Array1::<f64>::zeros(ambient);
        for (i, x) in samples.iter().enumerate() {
            let lg = spd.log_map(p.view(), x.view())?;
            xi.scaled_add(w[i], &lg);
        }
        // Stationarity residual ‖ξ‖_P: half the Riemannian gradient norm.
        let grad_norm = affine_sq_norm(n, &pinv, xi.view())?.sqrt();

        // Reached the requested first-order optimality tolerance.
        if grad_norm <= tol {
            return Ok(p);
        }

        // Track the best iterate and detect a stalled residual. A step counts
        // as progress only if it improves the best residual by more than
        // `STALL_REL` (relative); pure round-off wobble at the floor does not.
        let improved = grad_norm < best_grad * (1.0 - STALL_REL);
        if grad_norm < best_grad {
            best_grad = grad_norm;
            best_p.assign(&p);
        }
        if improved {
            stall = 0;
        } else {
            stall += 1;
            if stall >= STALL_PATIENCE {
                // The residual cannot be driven below the round-off of its own
                // evaluation: `best_p` is stationary to the achievable precision.
                return Ok(best_p);
            }
        }

        // Geodesic step P ← exp_P(t·ξ) with backtracking. The acceptance test
        // is Armijo sufficient decrease plus a round-off cushion `f_tol`: far
        // from the optimum `c1·t·pred` dominates and this is ordinary monotone
        // descent (handles overshoot on spread data); near the optimum, where
        // `V` is flat to machine precision, `f_tol` dominates and the test only
        // forbids an increase beyond round-off, admitting the convergent unit
        // Karcher step so the residual keeps descending below √ε.
        let pred = grad_norm * grad_norm; // ‖ξ‖²_P > 0 here.
        let f_tol = armijo_roundoff_cushion(f_cur);
        // Backtracking line search (t = 1, halving up to 60 steps) via the
        // shared primitive. The acceptance arithmetic is inlined verbatim so the
        // accepted step is bit-for-bit what the hand-rolled loop produced.
        let accepted = backtracking_line_search(
            BacktrackConfig::default(),
            |t| -> GeometryResult<Option<(f64, Array1<f64>)>> {
                let step = &xi * t;
                let cand = spd.exp_map(p.view(), step.view())?;
                let f_cand = dispersion(cand.view())?;
                Ok(Some((f_cand, cand)))
            },
            |t, f_cand| f_cand <= f_cur - 2.0 * ARMIJO_C1 * t * pred + f_tol,
        )?;
        match accepted {
            Some(step) => {
                f_cur = step.value;
                p = step.payload;
            }
            None => {
                // No positive step decreases V even within round-off: the
                // iterate is stationary to machine precision. Return the best.
                return Ok(best_p);
            }
        }
    }
    // Budget exhausted while still descending: keep the best (least-residual)
    // iterate rather than discarding it. Consistent with the two other
    // non-`tol` exits above (stall, rejected step) and with the generic
    // `response_frechet_mean` driver (#2140) — `best_p` is a valid SPD point and
    // a usable approximate Fréchet mean; erroring would needlessly abort a fit
    // over a mere iteration-budget shortfall on the geodesically convex cone.
    Ok(best_p)
}

#[cfg(test)]
mod tangent_basis_tests {
    use super::SpdManifold;
    use crate::manifold::RiemannianManifold;
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

#[cfg(test)]
mod frechet_mean_tests {
    use super::{SpdManifold, spd_frechet_mean};
    use crate::manifold::RiemannianManifold;
    use ndarray::{Array1, Array2};

    /// Row-major flat `n×n` diagonal matrix from its diagonal.
    fn diag_flat(d: &[f64]) -> Array1<f64> {
        let n = d.len();
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] = d[i];
        }
        Array1::from_iter(m.iter().copied())
    }

    /// Stack flat samples into the `M×n²` matrix the primitive consumes.
    fn stack(rows: &[Array1<f64>]) -> Array2<f64> {
        let m = rows.len();
        let k = rows[0].len();
        let mut s = Array2::<f64>::zeros((m, k));
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                s[[i, j]] = v;
            }
        }
        s
    }

    /// Stationarity residual ‖Σ_i w_i log_P(X_i)‖_P via the public maps.
    fn residual(spd: &SpdManifold, p: &Array1<f64>, rows: &[Array1<f64>], w: &[f64]) -> f64 {
        let k = p.len();
        let mut xi = Array1::<f64>::zeros(k);
        for (x, &wi) in rows.iter().zip(w) {
            xi.scaled_add(wi, &spd.log_map(p.view(), x.view()).expect("log_map"));
        }
        let g = spd.metric_tensor(p.view()).expect("metric_tensor");
        xi.dot(&g.dot(&xi)).max(0.0).sqrt()
    }

    /// CLOSED FORM, EXTREME MAGNITUDE. For mutually commuting (here diagonal)
    /// SPD matrices the affine-invariant Karcher mean is the per-coordinate
    /// geometric mean of the eigenvalues: `μ_k = (Π_i x_{i,k})^{1/M}`. With
    /// eigenvalues spanning `1e-6 … 1e6` the geodesic distances (and so the
    /// `exp`/`log` arguments) are large, stressing the eigendecomposition's
    /// dynamic range. gam must hit the analytic mean to ~machine precision.
    #[test]
    fn spd_frechet_mean_matches_geometric_mean_on_commuting_extreme_magnitudes() {
        let n = 3;
        let diags = [
            [1e6, 1e-6, 1.0],
            [1e-6, 1.0, 1e6],
            [1.0, 1e6, 1e-6],
            [1e2, 1e-2, 1e2],
        ];
        let rows: Vec<Array1<f64>> = diags.iter().map(|d| diag_flat(d)).collect();
        let m = rows.len();

        // Per-coordinate geometric mean (exact Karcher mean for commuting SPD).
        let mut want = [0.0_f64; 3];
        for k in 0..n {
            let mut s = 0.0;
            for d in &diags {
                s += d[k].ln();
            }
            want[k] = (s / m as f64).exp();
        }

        let p = spd_frechet_mean(n, stack(&rows).view(), None, 1e-12, 500)
            .expect("frechet mean converges on commuting extreme-magnitude SPD");

        let spd = SpdManifold::new(n);
        // Off-diagonals vanish; diagonal matches the geometric mean.
        for i in 0..n {
            for j in 0..n {
                let got = p[i * n + j];
                let exp = if i == j { want[i] } else { 0.0 };
                let scale = exp.abs().max(1.0);
                assert!(
                    (got - exp).abs() <= 1e-7 * scale,
                    "commuting mean[{i},{j}] = {got:.6e}, want {exp:.6e}"
                );
            }
        }
        // And it is a first-order Fréchet stationary point.
        let w = vec![1.0 / m as f64; m];
        let r = residual(&spd, &p, &rows, &w);
        assert!(r < 1e-9, "commuting case residual {r:.3e} not at floor");
    }

    /// WEIGHTED CLOSED FORM. The weighted affine-invariant Karcher mean of
    /// commuting SPD matrices is the weighted geometric mean
    /// `μ_k = Π_i x_{i,k}^{w_i}` (Σ w_i = 1). Verifies the weight plumbing is
    /// correct, not merely uniform.
    #[test]
    fn spd_frechet_mean_weighted_matches_weighted_geometric_mean() {
        let n = 2;
        let diags = [[4.0, 0.25], [0.5, 16.0], [9.0, 1.0]];
        let raw_w = [0.5, 0.3, 0.2];
        let rows: Vec<Array1<f64>> = diags.iter().map(|d| diag_flat(d)).collect();

        let mut want = [0.0_f64; 2];
        for k in 0..n {
            let mut s = 0.0;
            for (d, &wi) in diags.iter().zip(&raw_w) {
                s += wi * d[k].ln();
            }
            want[k] = s.exp();
        }

        let wv = Array1::from(raw_w.to_vec());
        let p = spd_frechet_mean(n, stack(&rows).view(), Some(wv.view()), 1e-12, 500)
            .expect("weighted frechet mean converges");
        for k in 0..n {
            let got = p[k * n + k];
            assert!(
                (got - want[k]).abs() <= 1e-9 * want[k].max(1.0),
                "weighted mean diag[{k}] = {got:.9e}, want {want_k:.9e}",
                want_k = want[k]
            );
        }
    }

    /// OVERSHOOT SAFEGUARD + SUB-√ε RESIDUAL ON NON-COMMUTING DATA. The samples
    /// are rotated `diag(a, b)` matrices with distinct rotation angles, so they
    /// do *not* commute: `V` is genuinely curved (not the trivial commuting
    /// one-step case), and for this spread `Hess V` carries eigenvalues `> 4`,
    /// along which a bare unit Karcher step (`= −½ grad V`) would overshoot.
    /// The backtracking safeguard must keep the descent monotone, and the
    /// round-off-cushioned line search must drive the first-order residual far
    /// below the `≈√ε ≈ 1e-7` floor at which a strict Armijo-on-V test stalls
    /// (the prior code panicked here). This is the direct regression guard for
    /// #693, from a different angle than the random-Gaussian integration test.
    #[test]
    fn spd_frechet_mean_converges_below_sqrt_eps_on_spread_non_commuting() {
        let n = 2;
        let angles = [0.0_f64, 0.6, 1.2, 1.9, 2.7];
        let eig = [
            (12.0_f64, 0.4_f64),
            (0.5, 9.0),
            (3.0, 0.2),
            (0.3, 6.0),
            (5.0, 0.7),
        ];
        let mut rows: Vec<Array1<f64>> = Vec::new();
        for (&th, &(a, b)) in angles.iter().zip(&eig) {
            let (c, s) = (th.cos(), th.sin());
            // R diag(a,b) Rᵀ, R = [[c,-s],[s,c]].
            let m00 = c * c * a + s * s * b;
            let m01 = c * s * (a - b);
            let m11 = s * s * a + c * c * b;
            rows.push(Array1::from(vec![m00, m01, m01, m11]));
        }
        let m = rows.len();

        // Request a tolerance below the achievable floor so success can only
        // come from the stagnation path, not from `tol` being loose.
        let p = spd_frechet_mean(n, stack(&rows).view(), None, 1e-14, 1000)
            .expect("spread non-commuting frechet mean converges via safeguard");

        let spd = SpdManifold::new(n);
        let w = vec![1.0 / m as f64; m];
        let r = residual(&spd, &p, &rows, &w);
        assert!(
            r < 1e-9,
            "spread non-commuting residual {r:.3e} did not descend below √ε \
             (regression of #693: line search stalled at the V round-off floor)"
        );

        // It must also be the dispersion minimizer: V(P) below V at any sample.
        let disp = |q: &Array1<f64>| -> f64 {
            rows.iter()
                .map(|x| {
                    let lg = spd.log_map(q.view(), x.view()).expect("log_map");
                    let g = spd.metric_tensor(q.view()).expect("metric");
                    lg.dot(&g.dot(&lg)) / m as f64
                })
                .sum()
        };
        let v_mean = disp(&p);
        for x in &rows {
            assert!(
                v_mean < disp(x),
                "mean does not minimize dispersion: V(mean)={v_mean:.6e}"
            );
        }
    }

    #[test]
    fn spd_frechet_mean_budget_shortfall_returns_best_iterate_not_error() {
        // Consistency with the generic response_frechet_mean driver (#2140): a
        // `max_iter` too small to reach the requested `tol` must STILL return
        // the best (least-residual) SPD iterate, matching the stall and
        // rejected-step exits — never a "did not reach stationarity within
        // max_iter" error over a mere budget shortfall on the convex cone.
        let n = 2;
        let rows = [
            diag_flat(&[4.0, 0.25]),
            Array1::from(vec![1.0, 0.5, 0.5, 3.0]),
            diag_flat(&[0.3, 6.0]),
        ];
        let m = rows.len();
        let p = spd_frechet_mean(n, stack(&rows).view(), None, 1e-14, 1)
            .expect("budget=1 must return the best iterate, not an error");

        // The returned point is a genuine SPD matrix (a valid mean / chart
        // origin): symmetric with strictly positive eigenvalues.
        assert_eq!(p.len(), n * n);
        assert!(p.iter().all(|c| c.is_finite()));
        assert!((p[1] - p[2]).abs() < 1e-12, "SPD mean must be symmetric");
        let spd = SpdManifold::new(n);
        // `log_map` is only defined at an SPD base point, so its success is a
        // positive-definiteness certificate for the returned iterate.
        let w = vec![1.0 / m as f64; m];
        let r = residual(&spd, &p, &rows, &w);
        assert!(r.is_finite(), "residual at the returned mean must be finite");
    }
}
