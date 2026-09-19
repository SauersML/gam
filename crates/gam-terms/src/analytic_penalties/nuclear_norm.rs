use super::*;

// ---------------------------------------------------------------------------
// Nuclear norm penalty
// ---------------------------------------------------------------------------

/// Basis-free low-rank penalty for a row-major `(n_eff, d)` latent block.
///
/// Lives on the extension-coordinate tier. The target is viewed as
/// `T ∈ ℝ^{n_eff × d}` and penalized by the smoothed nuclear norm
///
/// ```text
///   P(T) = w · Σ_{i < r} (sqrt(σ_i(T)^2 + ε^2) - ε),
/// ```
///
/// where `σ_i(T)` are singular values and `r` is either the full thin-SVD rank
/// or `max_rank` when a spectral cap is supplied. The penalty is basis-free:
/// it selects the rank of the decoder/latent embedding used by SAE wiring
/// without first committing to a canonical coordinate axis.
///
/// In the SAE objective this is the decoder embedding-rank selection lever
/// (#672): it shrinks unused singular directions of the matrix-valued latent
/// block while allowing the active subspace to rotate. It complements ARD
/// (axis-wise pruning after a gauge fix) and orthogonality/isometry terms
/// (basis and metric identification).
///
/// Gotchas:
///
/// * The Hessian is spectral and dense; callers should use the analytic HVP,
///   not a row-block diagonal shortcut.
/// * `max_rank` truncates the active singular spectrum. If the cutoff splits a
///   tied smoothed Gram eigenvalue, the HVP is undefined and the implementation
///   treats that as a caller contract violation.
/// * `ε > 0` smooths the zero singular-value kink. Very small `ε` makes rank
///   selection sharper but increases curvature around nearly-null directions.
#[derive(Debug, Clone)]
pub struct NuclearNormPenalty {
    pub target: PsiSlice,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub smoothing_eps: f64,
    /// Optional spectrum cap. The implementation computes faer's full thin SVD
    /// and retains the leading `max_rank` singular triplets when present.
    pub max_rank: Option<usize>,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

/// Relative eigenvalue gap `|Δλ|/λ̄` below which the Fréchet coefficient of the
/// filter `f(λ) = λ^{-1/2}` is taken as the midpoint derivative `½(f′ᵢ + f′ⱼ)`
/// instead of the divided difference `(fᵢ − fⱼ)/Δλ`.
///
/// Each `f` and each regularized `λ = σ² + ε²` rounds once, so the divided
/// difference carries about `3u·f` in its numerator and `2u·λ̄` in `Δλ`;
/// against `|f′| = ½λ̄^{-3/2}` that is a relative `4ε·λ̄/|Δλ|`. The midpoint
/// derivative misses `f‴Δλ²/12`, a relative `(5/16)(Δλ/λ̄)²`. The two errors
/// are equal at `|Δλ|/λ̄ = ∛(64ε/5)`, and each route is the more accurate one
/// on its own side of that gap.
fn frechet_route_balance() -> f64 {
    // Derived (#2469): `64/5` and the cube root come from equating the two error
    // bounds above, so the crossover follows from the rounding counts, not a choice.
    (64.0 / 5.0 * f64::EPSILON).cbrt()
}

/// Orthonormal basis `S` of the joint row space of `T` and `V`, by reorthogonalized
/// modified Gram–Schmidt over their stacked rows in order. It is deterministic.
///
/// A row joins the basis when `gam_linalg::gram_schmidt::ReorthogonalizedRowBasis`
/// resolves its residual: above its own Gram–Schmidt band plus the error the
/// directions already collected carry (#2469). A zero row is skipped, and the
/// first nonzero row always joins.
pub(crate) fn joint_row_space_basis(t: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Vec<Array1<f64>> {
    let mut basis = gam_linalg::gram_schmidt::ReorthogonalizedRowBasis::new();
    for source in [&t, &v] {
        for row in source.rows() {
            basis.admit(row);
        }
    }
    basis.into_directions()
}

struct NuclearSvdCache {
    u: Array2<f64>,
    singular: Array1<f64>,
    vt: Array2<f64>,
}

impl NuclearNormPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        weight: f64,
        n_eff: usize,
        smoothing_eps: f64,
        max_rank: Option<usize>,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("NuclearNormPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "NuclearNormPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("NuclearNormPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "NuclearNormPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        if let Some(latent_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(latent_dim).ok_or_else(|| {
                "NuclearNormPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "NuclearNormPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    latent_dim
                ));
            }
        }
        if !(smoothing_eps.is_finite() && smoothing_eps > 0.0) {
            return Err(format!(
                "NuclearNormPenalty::new requires finite smoothing_eps > 0, got {smoothing_eps}"
            ));
        }
        // The HVP filter's curvature `−½(σ²+ε²)^{-3/2}` is largest at `σ = 0`, where
        // it is `−½ε^{-3}`: an ε whose `ε^{-3}` is not representable has no finite
        // curvature to report.
        if !(smoothing_eps * smoothing_eps).powf(-1.5).is_finite() {
            return Err(format!(
                "NuclearNormPenalty::new requires smoothing_eps with a representable ε^-3, \
                 the HVP filter's curvature at σ = 0; got {smoothing_eps}"
            ));
        }
        if matches!(max_rank, Some(0)) {
            return Err("NuclearNormPenalty::new requires max_rank > 0".to_string());
        }
        Ok(Self {
            target,
            weight,
            n_eff,
            smoothing_eps,
            max_rank,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            validated_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn rank_limit(&self, thin_rank: usize) -> usize {
        self.max_rank.unwrap_or(thin_rank).min(thin_rank)
    }

    /// Shift-regularized squared smoothed singular value `σ² + ε²`.
    ///
    /// This is the single regularized spectrum shared by `value`, `grad_target`
    /// and the HVP's right-Gram filter, so that the smoothed nuclear norm
    /// `Σ(√(σ²+ε²) − ε)`, its gradient `σ/√(σ²+ε²)`, and the Fréchet
    /// inverse-square-root filter `(σ²+ε²)^{-1/2}` are all evaluated on the
    /// *same* eigenvalue, keeping the value↔gradient↔Hessian consistency that
    /// REML evidence and the Newton curvature block rely on (#737). Every `σ²`
    /// that reaches it is non-negative — a squared singular value, or a
    /// right-Gram eigenvalue after the PSD refusal and zero clamp — and `new`
    /// refuses an `ε` whose filter curvature `ε^{-3}` is not representable, so
    /// the shifted spectrum is positive with finite filters and needs no floor.
    /// The shift is affine in `σ²`, so every consumer differentiates it exactly.
    fn regularized_sigma_sq(&self, sigma_sq: f64) -> f64 {
        sigma_sq + self.smoothing_eps * self.smoothing_eps
    }

    /// Number of leading right-Gram eigen-directions (top singular values) the
    /// HVP keeps active, identical to the rank `value`/`grad` sum over.
    ///
    /// The right Gram `TᵀT` is `d×d` but has at most `thin_rank = min(n_rows, d)`
    /// nonzero eigenvalues (the squared singular values); the remaining
    /// `d − thin_rank` are an exact, *tied* zero subspace. Capping the active
    /// count with the Gram width `d` (or any value `> thin_rank`) would push the
    /// active/inactive cutoff *inside* that tied zero subspace, where the split
    /// is ill-defined — for a wide block (`n_rows < d`) with
    /// `max_rank > thin_rank` this previously panicked. We therefore cap with the
    /// true SVD length `thin_rank`, matching `rank_limit`, so the cutoff always
    /// lands either between the zero subspace and the nonzero singular values, or
    /// between distinct nonzero singular values, never bisecting the zero block.
    fn right_filter_active_count(&self, n_rows: usize, n_cols: usize) -> usize {
        let thin_rank = n_rows.min(n_cols);
        match self.max_rank {
            // No cap: keep every right-Gram direction. The `d − thin_rank` exact
            // zero directions get a finite smoothed `(0+ε²)^(-1/2)` filter and
            // contribute nothing to `G(T)` (since `T` has no projection onto
            // them), so this is consistent with `value`/`grad`'s full sum.
            None => n_cols,
            // A cap that does not bite (`max_rank ≥ thin_rank`) is likewise a
            // no-op: keep every direction.
            Some(max_rank) if max_rank >= thin_rank => n_cols,
            // A genuine cap keeps only the top `max_rank` singular directions —
            // never more than `thin_rank`, so the active/inactive cutoff lands
            // strictly inside the nonzero singular block and never bisects the
            // tied zero subspace of the `d×d` Gram.
            Some(max_rank) => max_rank,
        }
    }

    /// Apply the right-spectral filter pair directly: returns `(V·R, T·dR[V])`,
    /// each `(n_rows, d)`, where `R = (TᵀT + ε²I)^{-1/2}` (regularized, active-
    /// windowed) and `dR[V]` is its Fréchet derivative along `V` — the two
    /// pieces [`Self::hvp`] sums.
    ///
    /// Cost structure: the right Gram `TᵀT` is `d×d` but `rank(G) ≤ n_rows`,
    /// and the tangent Gram `TᵀV + VᵀT` is supported on the joint row space
    /// `S = rowspace(T) ∪ rowspace(V)` with `dim S ≤ 2·n_rows`. Every pair of
    /// eigen-directions with either side in `S⊥` contributes `0` to `dR`
    /// (the tangent Gram annihilates them), and `R` acts on `S⊥` as the
    /// constant `f₀ = regularized(0)^{-1/2}` (a tied eigen-class, so the
    /// filter there is the basis-independent `f₀·(I − SSᵀ)` — active only
    /// when the window covers the full Gram, i.e. no biting `max_rank`).
    /// So the whole computation collapses to an `s×s` eigenproblem plus
    /// `O(n_rows·s·d)` products — replacing the former dense `d×d` eigh and
    /// two `O(d⁴)` basis rotations PER HVP CALL, which at decoder-block
    /// orientation (`d = p` in the thousands) was measured eating >99% of
    /// whole-fit wall time. The dense route is kept below for small `d`
    /// (no asymptotic win) and remains the defining oracle.
    pub(crate) fn right_spectral_filters_applied(
        &self,
        t: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let m = t.nrows();
        let d = t.ncols();
        if d <= 2 * m + 8 {
            let (rf, rfd) = self.right_spectral_inverse_sqrt_derivative(t, v)?;
            return Ok((v.dot(&rf), t.dot(&rfd)));
        }
        let basis = joint_row_space_basis(t, v);
        let s_dim = basis.len();
        if s_dim == 0 {
            // T = V = 0: R is the constant f₀ filter and dR vanishes.
            let active_count = self.right_filter_active_count(m, d);
            if active_count != d {
                // The full right-Gram spectrum is one d-fold tied zero class.
                // A biting max_rank would choose an arbitrary subspace of that
                // class, matching the dense oracle's tie-split rejection.
                return Err(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     right-Gram eigenvalue at the active/inactive cutoff (0.0e0, 0.0e0)"
                        .to_string(),
                );
            }
            let f0 = self.regularized_sigma_sq(0.0).powf(-0.5);
            let vr = v.to_owned() * f0;
            return Ok((vr, Array2::<f64>::zeros((m, d))));
        }
        let mut s = Array2::<f64>::zeros((d, s_dim));
        for (j, b) in basis.iter().enumerate() {
            s.column_mut(j).assign(b);
        }
        let ts = t.dot(&s); // m × s
        let vs = v.dot(&s); // m × s
        let gh = ts.t().dot(&ts); // Sᵀ G S
        let dgh = ts.t().dot(&vs) + vs.t().dot(&ts); // Sᵀ dG S
        let (evals, q) = gh.eigh(Side::Lower).map_err(|err| {
            format!("NuclearNormPenalty right-Gram eigendecomposition failed: {err}")
        })?;
        // `T S` forms each entry as a `d`-term inner product and `Sᵀ G S` forms an
        // `m`-term one over those, so entrywise `|δ(SᵀGS)| ≤ (γ_m + 2γ_d)·(BᵀB)` with
        // `B = |T||S|`. That Frobenius norm plus the eigensolver's own
        // `s·ε·‖SᵀGS‖₂` bounds how far each computed eigenvalue sits from the
        // spectrum of the Gram it stands for (#2469).
        let abs_ts = t.mapv(f64::abs).dot(&s.mapv(f64::abs));
        let formation = (gam_linalg::roundoff::accumulation_growth(m)
            + 2.0 * gam_linalg::roundoff::accumulation_growth(d))
            * abs_ts.t().dot(&abs_ts).iter().map(|value| value * value).sum::<f64>().sqrt();
        let band =
            gam_linalg::roundoff::symmetric_spectrum_rounding_band(&evals.to_vec()) + formation;
        let mut raw_evals = Array1::<f64>::zeros(s_dim);
        for i in 0..s_dim {
            let lambda = evals[i];
            if !lambda.is_finite() {
                return Err(format!(
                    "NuclearNormPenalty expected finite right-Gram eigenvalue; got {lambda}"
                ));
            }
            if lambda < -band {
                return Err(format!(
                    "NuclearNormPenalty expected PSD right Gram; eigenvalue {lambda:.3e} \
                     is below its rounding band {band:.3e}"
                ));
            }
            raw_evals[i] = lambda.max(0.0);
        }
        // Active window over the FULL ascending d-spectrum, which is the
        // (d − s)-fold tied zero class of S⊥ followed by `raw_evals`
        // (ascending). Mirrors the dense path's windowing and its tie-split
        // guard exactly.
        let active_count = self.right_filter_active_count(m, d);
        let zero_class_active = active_count == d;
        if !zero_class_active && active_count > s_dim {
            // The cutoff would bisect the tied zero class of S⊥ — the same
            // condition the dense path rejects via its adjacent-eigenvalue
            // guard (both neighbors are exact zeros).
            return Err(
                "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                 right-Gram eigenvalue at the active/inactive cutoff (0.0e0, 0.0e0)"
                    .to_string(),
            );
        }
        // Index of the first ACTIVE entry within the s-block.
        let active_start_s = s_dim.saturating_sub(active_count.min(s_dim));
        if self.max_rank.is_some() && !zero_class_active {
            // Tie guard at the cutoff, on RAW eigenvalues as in the dense path.
            // Left neighbor is inside the s-block when the window is strictly
            // interior; when the window covers the whole s-block the left
            // neighbor is the top of the S⊥ zero class.
            let (left, right) = if active_start_s > 0 {
                (evals[active_start_s - 1], evals[active_start_s])
            } else {
                (0.0, evals[0])
            };
            // Two computed eigenvalues whose exact values tie can sit up to twice
            // the band apart. The top of the S⊥ zero class is exact and adds none.
            let computed_endpoints = if active_start_s > 0 { 2.0 } else { 1.0 };
            if (right - left).abs() <= computed_endpoints * band {
                return Err(format!(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     right-Gram eigenvalue at the active/inactive cutoff \
                     ({left:.3e}, {right:.3e})"
                ));
            }
        }
        let mut regularized_evals = Array1::<f64>::zeros(s_dim);
        let mut f = Array1::<f64>::zeros(s_dim);
        let mut df = Array1::<f64>::zeros(s_dim);
        for i in 0..s_dim {
            regularized_evals[i] = self.regularized_sigma_sq(raw_evals[i]);
            if i >= active_start_s {
                let lambda = regularized_evals[i];
                f[i] = lambda.powf(-0.5);
                df[i] = -0.5 * lambda.powf(-1.5);
            }
        }
        // B̂ = Q̂ᵀ (Sᵀ dG S) Q̂, then the divided-difference Hadamard product —
        // identical pair rules to the dense path. All pairs touching S⊥ have
        // B = 0 (dG is supported on S), so they need no representation.
        let b_basis = q.t().dot(&dgh).dot(&q);
        let balance = frechet_route_balance();
        let mut deriv_basis = Array2::<f64>::zeros((s_dim, s_dim));
        for i in 0..s_dim {
            for j in 0..s_dim {
                let denom = regularized_evals[i] - regularized_evals[j];
                let midpoint = 0.5 * (regularized_evals[i] + regularized_evals[j]);
                let both_active = i >= active_start_s && j >= active_start_s;
                let divided_difference = if both_active && denom.abs() <= balance * midpoint {
                    0.5 * (df[i] + df[j])
                } else if denom == 0.0 {
                    0.0
                } else {
                    (f[i] - f[j]) / denom
                };
                deriv_basis[[i, j]] = divided_difference * b_basis[[i, j]];
            }
        }
        // V·R = f₀·V·(I − SSᵀ) [zero-class active only] + (V S) Q̂ f̂ Q̂ᵀ Sᵀ.
        let qf = {
            let mut qf = q.clone();
            for i in 0..s_dim {
                let fi = f[i];
                qf.column_mut(i).mapv_inplace(|x| x * fi);
            }
            qf.dot(&q.t()) // Q̂ diag(f̂) Q̂ᵀ, s×s
        };
        let mut vr = vs.dot(&qf).dot(&s.t());
        if zero_class_active {
            let f0 = self.regularized_sigma_sq(0.0).powf(-0.5);
            // V − (V S) Sᵀ is V's S⊥ component.
            let v_perp = v.to_owned() - vs.dot(&s.t());
            vr += &(v_perp * f0);
        }
        // T·dR = (T S) Q̂ (Δf̂ ∘ B̂) Q̂ᵀ Sᵀ.
        let w = q.dot(&deriv_basis).dot(&q.t());
        let tdr = ts.dot(&w).dot(&s.t());
        Ok((vr, tdr))
    }

    fn compute_svd_cached(&self, t: ArrayView2<'_, f64>) -> NuclearSvdCache {
        // Existing faer wrapper calls `faer::linalg::svd::svd(..., Thin, Thin, ...)`.
        let owned = t.to_owned();
        let (u, singular, vt) = owned
            .svd(true, true)
            .expect("NuclearNormPenalty SVD failed to converge");
        NuclearSvdCache {
            u: u.expect("NuclearNormPenalty requested left singular vectors"),
            singular,
            vt: vt.expect("NuclearNormPenalty requested right singular vectors"),
        }
    }

    pub(crate) fn right_spectral_inverse_sqrt_derivative(
        &self,
        t: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        // HVP for spectral matrix functions (matrix-derivative-with-singular-values):
        // G(T)=T(TᵀT+ε²I)^(-1/2), so dG[V]=V R + T dR[V].
        // The Fréchet derivative dR uses divided differences in the right
        // singular-vector basis, avoiding any dense Hessian materialization.
        let d = t.ncols();
        let active_count = self.right_filter_active_count(t.nrows(), d);
        let active_start = d.saturating_sub(active_count);
        let mut gram = Array2::<f64>::zeros((d, d));
        let mut tangent_gram = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut g = 0.0;
                let mut dg = 0.0;
                for n in 0..t.nrows() {
                    g += t[[n, a]] * t[[n, b]];
                    dg += t[[n, a]] * v[[n, b]] + v[[n, a]] * t[[n, b]];
                }
                gram[[a, b]] = g;
                tangent_gram[[a, b]] = dg;
            }
        }

        let (evals, q) = gram.eigh(Side::Lower).map_err(|err| {
            format!("NuclearNormPenalty right-Gram eigendecomposition failed: {err}")
        })?;
        // Each Gram entry is an `m`-term inner product, so `|δG| ≤ γ_m·(|T|ᵀ|T|)`
        // entrywise. That Frobenius norm plus the eigensolver's own `d·ε·‖G‖₂`
        // bounds how far each computed eigenvalue sits from the exact spectrum
        // (#2469).
        let abs_t = t.mapv(f64::abs);
        let formation = gam_linalg::roundoff::accumulation_growth(t.nrows())
            * abs_t.t().dot(&abs_t).iter().map(|value| value * value).sum::<f64>().sqrt();
        let band =
            gam_linalg::roundoff::symmetric_spectrum_rounding_band(&evals.to_vec()) + formation;
        let mut raw_evals = Array1::<f64>::zeros(d);
        for i in 0..d {
            let lambda = evals[i];
            if !lambda.is_finite() {
                return Err(format!(
                    "NuclearNormPenalty expected finite right-Gram eigenvalue; got {lambda}"
                ));
            }
            if lambda < -band {
                return Err(format!(
                    "NuclearNormPenalty expected PSD right Gram; eigenvalue {lambda:.3e} \
                     is below its rounding band {band:.3e}"
                ));
            }
            raw_evals[i] = lambda.max(0.0);
        }
        if self.max_rank.is_some() && active_count < d && active_start > 0 {
            let left = evals[active_start - 1];
            let right = evals[active_start];
            // Two computed eigenvalues whose exact values tie can sit up to twice
            // the band apart.
            if (right - left).abs() <= 2.0 * band {
                return Err(format!(
                    "NuclearNormPenalty HVP is undefined: max_rank splits a tied \
                     right-Gram eigenvalue at the active/inactive cutoff \
                     ({left:.3e}, {right:.3e})"
                ));
            }
        }
        let mut regularized_evals = Array1::<f64>::zeros(d);
        let mut f = Array1::<f64>::zeros(d);
        let mut df = Array1::<f64>::zeros(d);
        for i in 0..d {
            // Same shared shift used by `value`/`grad_target` (#737): the
            // right-Gram eigenvalue `raw_evals[i]` is the squared singular value
            // `σ²`, so `regularized_sigma_sq(σ²) = σ² + ε²` keeps the filter on
            // the identical regularized spectrum.
            regularized_evals[i] = self.regularized_sigma_sq(raw_evals[i]);
            if i >= active_start {
                // Keep the value filter and Fréchet derivative on the same
                // regularized spectrum, so divided differences never observe
                // stale raw eigenvalues near zero.
                let lambda = regularized_evals[i];
                f[i] = lambda.powf(-0.5);
                df[i] = -0.5 * lambda.powf(-1.5);
            }
        }

        let mut right_filter = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += q[[a, i]] * f[i] * q[[b, i]];
                }
                right_filter[[a, b]] = s;
            }
        }

        let mut b_basis = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for a in 0..d {
                    for b in 0..d {
                        s += q[[a, i]] * tangent_gram[[a, b]] * q[[b, j]];
                    }
                }
                b_basis[[i, j]] = s;
            }
        }

        let balance = frechet_route_balance();
        let mut derivative_basis = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let denom = regularized_evals[i] - regularized_evals[j];
                let midpoint = 0.5 * (regularized_evals[i] + regularized_evals[j]);
                let both_active = i >= active_start && j >= active_start;
                let divided_difference = if both_active && denom.abs() <= balance * midpoint {
                    0.5 * (df[i] + df[j])
                } else if denom == 0.0 {
                    0.0
                } else {
                    (f[i] - f[j]) / denom
                };
                derivative_basis[[i, j]] = divided_difference * b_basis[[i, j]];
            }
        }

        let mut right_filter_derivative = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        s += q[[a, i]] * derivative_basis[[i, j]] * q[[b, j]];
                    }
                }
                right_filter_derivative[[a, b]] = s;
            }
        }

        Ok((right_filter, right_filter_derivative))
    }
}

impl AnalyticPenalty for NuclearNormPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let svd = self.compute_svd_cached(t);
        let rank = self.rank_limit(svd.singular.len());
        let eps = self.smoothing_eps;
        let mut acc = 0.0;
        for i in 0..rank {
            let sigma = svd.singular[i];
            // Floored on the shared regularized spectrum so the value matches the
            // HVP's right-Gram filter (see `regularized_sigma_sq`).
            acc += self.regularized_sigma_sq(sigma * sigma).sqrt() - eps;
        }
        self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let svd = self.compute_svd_cached(t);
        let rank = self.rank_limit(svd.singular.len());
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for i in 0..rank {
            let sigma = svd.singular[i];
            // d/dσ (√(σ²+ε²) − ε) = σ/√(σ²+ε²), floored on the shared regularized
            // spectrum so grad↔value↔HVP stay mutually consistent (#737).
            let spectral_grad = sigma / self.regularized_sigma_sq(sigma * sigma).sqrt();
            for n in 0..t.nrows() {
                for a in 0..t.ncols() {
                    grad[[n, a]] += weight * svd.u[[n, i]] * spectral_grad * svd.vt[[i, a]];
                }
            }
        }
        super::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        // `AnalyticPenalty::hvp_target` has no Result channel; decomposition
        // or active-rank cutoff failures from the spectral helper are upstream
        // contract violations that must surface loudly.
        let (vr, tdr) = self
            .right_spectral_filters_applied(t.view(), v_mat.view())
            // SAFETY: error path is a caller contract violation; the upstream
            // helper already formatted a diagnostic message.
            .unwrap_or_else(|message| panic!("{}", message));
        let weight = self.resolved_weight(rho);
        let out = (vr + tdr) * weight;
        super::flatten_matrix(&out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();
    impl_learnable_weight_domain!(weight);

    fn name(&self) -> &str {
        "nuclear_norm"
    }

    impl_scalar_apply_schedule!(weight);
}
