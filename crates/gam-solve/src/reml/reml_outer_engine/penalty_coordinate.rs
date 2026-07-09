use super::*;
pub use gam_problem::PenaltyCoordinate;

/// Exact pseudo-logdeterminant log|S|₊ and its derivatives with respect to ρ.
///
/// # Exact pseudo-logdet on the positive eigenspace
///
/// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace
/// N(S) = ∩_k N(S_k) is structurally fixed (independent of ρ).
/// No eigenvalue of S crosses zero during optimization, so the
/// pseudo-logdet L = Σ_{σ_i > 0} log σ_i is C∞ in ρ.
///
/// ## Computation
///
/// Eigendecompose S, identify positive eigenvalues σ_i > ε (where ε is a
/// relative threshold for numerical zero detection), then:
///
///   L(S)     = Σ_{positive} log σ_i
///   ∂_k L    = tr(S⁺ A_k)            where A_k = λ_k S_k
///   ∂²_kl L  = δ_{kl} ∂_k L − tr(S⁺ A_l S⁺ A_k)
///
/// S⁺ is the Moore-Penrose pseudoinverse restricted to the positive
/// eigenspace. These are the exact derivatives of L — no δ-regularization,
/// no nullity metadata, no chain-rule inconsistencies.
#[derive(Clone, Debug)]
pub struct PenaltyLogdetDerivs {
    /// L(S) = log|S|₊ — the exact pseudo-logdeterminant on the positive eigenspace.
    ///
    /// L(S) = Σ_{σ_i > ε} log σ_i, where ε is a relative threshold that
    /// identifies the structural nullspace directly from the eigenspectrum.
    pub value: f64,
    /// ∂/∂ρₖ L(S) — first derivatives (one per smoothing parameter).
    ///
    /// ∂_k L = tr(S⁺ Aₖ) where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse
    /// restricted to the positive eigenspace.
    pub first: Array1<f64>,
    /// ∂²/(∂ρₖ∂ρₗ) L(S) — second derivatives (for outer Hessian).
    ///
    /// ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ).
    pub second: Option<Array2<f64>>,
}

/// Unified representation of a single smoothing-parameter penalty coordinate.
///

// PenaltyLogdetEigenspace, build_penalty_logdet_eigenspace,
// scaled_penalty_logdet_nullspace_leakage, and frobenius_inner_same_shape
// have been replaced by the canonical PenaltyPseudologdet in
// super::super::penalty_logdet. All callers now use that module directly.

/// Reduced trace kernel `K = U · M · Uᵀ` for pseudo-logdet REML/LAML
/// criteria: an orthonormal column basis `u_s` (p × r) plus the r × r
/// symmetric reduced kernel `h_proj_inverse`, with `tr(K · A)` evaluated as
/// `tr(M · Uᵀ A U)` so contractions run on the r-dimensional subspace.
///
/// Two producers install it, with different (documented) exactness domains:
///
/// 1. **Intrinsic spectral form (#901, the GLM dense paths in runtime.rs —
///    `intrinsic_hessian_pseudo_logdet_parts`):** `u_s = U_H`, the kept
///    eigenvectors of the penalized Hessian `H_pen`, and `h_proj_inverse =
///    diag(1/σ_a)`. Then `K = H_pen⁺` exactly, and `tr(K · Ḣ)` is the exact
///    first derivative of the cost's `log|H_pen|₊` along **every** drift
///    direction — penalty-supported or not, moving-subspace ψ drifts
///    included — because on a constant-rank stratum first-order eigenvector
///    motion cancels out of the pseudo-logdet derivative. This object can be
///    traced against the GLM IFT correction `D_β H[v] = X' diag(c ⊙ X v) X`
///    (which leaks onto `null(S)` via the intercept column) without error.
///
/// 2. **Range(Sλ) Schur block (#752, `joint_penalty_subspace_trace_parts`
///    in custom_family.rs):** `u_s` spans `range(Sλ)` and `h_proj_inverse =
///    U_Sᵀ (H+Sλ)⁺ U_S`. For penalty-supported `A` (`A = ∂Sλ/∂ρ`), the
///    identity `U_S U_Sᵀ A U_S U_Sᵀ = A` gives `tr(K · A) = tr((H+Sλ)⁺ A) =
///    d log|H+Sλ|₊/dρ` — exact for the ρ family. It is **not** exact for
///    drifts with `null(Sλ)` support (GLM cubic corrections, ψ basis
///    drifts); paths that carry such drifts must install form 1.
///
/// Historically this struct carried a third reading — `(U_Sᵀ H U_S)⁻¹`, the
/// plain projected inverse paired with the projected cost `log|U_Sᵀ H U_S|₊`.
/// That object is WRONG as a REML determinant term: splitting `H` over
/// `range(S) ⊕ ker(S)` as `[[A,B],[Bᵀ,C]]`, the projected logdet is
/// `log det A`, dropping the θ-dependent Schur curvature
/// `log det(C − BᵀA⁻¹B)` of the likelihood-identified, penalty-null block
/// (sign-flipped ρ-gradients, ~1e5 ψ blow-ups vs FD — #901). No producer
/// builds it anymore.
#[derive(Clone, Debug)]
pub struct PenaltySubspaceTrace {
    pub u_s: Array2<f64>,
    pub h_proj_inverse: Array2<f64>,
}

impl PenaltySubspaceTrace {
    /// Compute `tr(K · A)` where `K = U_S · h_proj_inverse · U_Sᵀ` — the
    /// pseudo-logdet trace kernel (see the struct doc for the two producer
    /// forms and their exactness domains).
    ///
    /// Uses the identity `tr(K · A) = tr(h_proj_inverse · U_Sᵀ A U_S)` so the
    /// reduction runs on the r × r subspace rather than materializing K.
    pub fn trace_projected_logdet(&self, a: &Array2<f64>) -> f64 {
        gam_terms::construction::trace_penalty_covariance_in_orthogonal_basis(
            a,
            &self.u_s,
            &self.h_proj_inverse,
        )
    }

    /// Reduce a p × p matrix `A` to its r × r projection `U_Sᵀ · A · U_S`.
    ///
    /// Exposed so callers that need the same reduced matrix for both the
    /// single-trace `tr(K · A)` and the cross-trace `tr(K · A · K · B)`
    /// can avoid repeating the p × p · p × r matmuls.  Routes through
    /// faer's parallel SIMD GEMM (`fast_atb` / `fast_ab`) so the p-large
    /// contraction axis amortizes across all cores.
    pub fn reduce(&self, a: &Array2<f64>) -> Array2<f64> {
        let u_s_t_a = gam_linalg::faer_ndarray::fast_atb(&self.u_s, a);
        gam_linalg::faer_ndarray::fast_ab(&u_s_t_a, &self.u_s)
    }

    /// Compute `tr(H_proj⁻¹ · R)` given an already-reduced `R = U_Sᵀ A U_S`.
    pub fn trace_projected_logdet_reduced(&self, r_mat: &Array2<f64>) -> f64 {
        gam_terms::construction::trace_reduced_penalty_covariance(r_mat, &self.h_proj_inverse)
    }

    /// Cross-trace given pre-reduced blocks `R_A = U_Sᵀ A U_S`, `R_B = U_Sᵀ B U_S`.
    pub fn trace_projected_logdet_cross_reduced(&self, ra: &Array2<f64>, rb: &Array2<f64>) -> f64 {
        // left = H_proj⁻¹ · R_A ;  right = H_proj⁻¹ · R_B ;  tr(left · right).
        let left = self.h_proj_inverse.dot(ra);
        let right = self.h_proj_inverse.dot(rb);
        dense::trace_product(&left, &right)
    }

    /// Reduce a `HyperOperator` `A` to its `r × r` projection
    /// `U_Sᵀ · A · U_S` without materializing the dense `p × p` block.
    /// Uses `A.mul_mat(U_S)` so an Hv-only operator is probed in `r` matvecs
    /// (each `O(work_of_A)`), then a single `r × p × r` reduction routed
    /// through faer's parallel SIMD GEMM (`fast_atb`).
    pub fn reduce_operator<O>(&self, a: &O) -> Array2<f64>
    where
        O: HyperOperator + ?Sized,
    {
        let au = a.mul_mat(&self.u_s);
        gam_linalg::faer_ndarray::fast_atb(&self.u_s, &au)
    }

    /// `tr(K · A)` for `A` exposed only as a `HyperOperator`.  Mirrors
    /// [`Self::trace_projected_logdet`] without forcing dense materialization
    /// of `A`.
    pub fn trace_operator<O>(&self, a: &O) -> f64
    where
        O: HyperOperator + ?Sized,
    {
        self.trace_projected_logdet_reduced(&self.reduce_operator(a))
    }

    /// Projected leverage `h^{G,proj}_i = Xᵢᵀ · K · Xᵢ` for every row of `x`.
    ///
    /// Computed in bulk as `Z = X · U_S` (`n × r`) then
    /// `h^{G,proj}_i = (Z H_proj⁻¹ Zᵀ)_{ii} = Σ_{a,b} Z_{ia} (H_proj⁻¹)_{ab} Z_{ib}`,
    /// total cost `O(n · p · r + n · r²)` — strictly cheaper than `n` calls
    /// to [`Self::apply`] because the `n × p · p × r` GEMM streams the
    /// `p`-axis once.  Streams `X` through `try_row_chunk` so operator-backed
    /// (Lazy) designs at large scale never densify the full `(n × p)` block.
    pub fn xt_projected_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let r = self.u_s.ncols();
        assert_eq!(self.u_s.nrows(), p);
        assert_eq!(self.h_proj_inverse.nrows(), r);
        assert_eq!(self.h_proj_inverse.ncols(), r);

        let block = {
            const TARGET_CHUNK_FLOATS: usize = 1 << 16;
            (TARGET_CHUNK_FLOATS / p.max(1)).clamp(1, n.max(1))
        };

        let mut h = Array1::<f64>::zeros(n);
        let mut start = 0usize;
        while start < n {
            let end = (start + block).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                // SAFETY: `start..end` is constructed from
                // `0..n = 0..x.nrows()` with `end = (start+block).min(n)`,
                // so it is always a valid sub-range of `x`. Failure means
                // the operator broke its row-chunk contract.
                // SAFETY: row range built from 0..x.nrows(); failure means operator broke its contract.
                reml_contract_panic(format!(
                    "xt_projected_kernel_x_diagonal: row chunk failed: {err}"
                ))
            });
            // Z_chunk = rows · U_S  ((end-start) × r).
            let z_chunk = gam_linalg::faer_ndarray::fast_ab(&rows, &self.u_s);
            // h_i = Σ_{a,b} Z_{ia} (H_proj⁻¹)_{ab} Z_{ib}.
            for (i, row_z) in z_chunk.outer_iter().enumerate() {
                let mut acc = 0.0;
                for (z_a, h_row) in row_z
                    .iter()
                    .copied()
                    .zip(self.h_proj_inverse.rows().into_iter())
                {
                    let mut inner = 0.0;
                    for (h_value, z_b) in h_row.iter().copied().zip(row_z.iter().copied()) {
                        inner += h_value * z_b;
                    }
                    acc += z_a * inner;
                }
                h[start + i] = acc;
            }
            start = end;
        }
        h
    }

    /// Projected bilinear pseudo-inverse `aᵀ · K⁺ · b` where
    /// `K⁺ = U_S · H_proj⁻¹ · U_Sᵀ`.
    ///
    /// Used by the rank-deficient LAML IFT correction path: when `b ∈
    /// col(S_k) ⊂ range(S_+)`, applying the projected pseudo-inverse
    /// instead of the full `H⁻¹` strips spurious null-space noise from
    /// `a` (≈ the outer-stationarity residual `r`) before the inverse,
    /// without biasing the numerator. Costs `O(p·r + r²)` versus the
    /// `O(p²·r)` full solve.
    pub fn bilinear_pseudo_inverse(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let proj_a = gam_linalg::faer_ndarray::fast_atv(&self.u_s, a);
        let proj_b = gam_linalg::faer_ndarray::fast_atv(&self.u_s, b);
        let h_proj_inv_b = self.h_proj_inverse.dot(&proj_b);
        proj_a.dot(&h_proj_inv_b)
    }

    /// Euclidean projection onto the retained penalty/Hessian range used by
    /// this projected kernel: `P_S a = U_S U_Sᵀ a`.
    pub fn project_onto_subspace(&self, a: &Array1<f64>) -> Array1<f64> {
        let proj_a = gam_linalg::faer_ndarray::fast_atv(&self.u_s, a);
        gam_linalg::faer_ndarray::fast_av(&self.u_s, &proj_a)
    }

    /// Apply the projected pseudo-inverse `K = U_S · H_proj⁻¹ · U_Sᵀ` to a
    /// vector `a`, returning the minimum-norm solution `v = K · a` of the
    /// system `H v = a` restricted to `range(S₊)`.
    ///
    /// This is the correct stand-in for `H⁻¹ · a` in all per-coordinate
    /// outer-gradient/Hessian formulas when the rank-deficient LAML fix is
    /// active (`penalty_subspace_trace = Some`). The full `H⁻¹ · a` solve
    /// amplifies any component of `a` outside `range(H_free)` by
    /// `1/σ_min(H_active_normal)` — which on large-scale survival
    /// marginal-slope is ~10¹² and propagates into outer gradients of
    /// magnitude 10¹⁴, suppressed by the envelope tripwire downstream and
    /// killing every seed before the fit can take a step. Dropping the
    /// `ker(M)` component of `a` is exactly what the cost `½ log|M|₊` demands —
    /// that subspace is the cost's gauge, so the drop is bit-invariant for the
    /// correction regardless of the residual's gauge-direction magnitude;
    /// `ProjectedKktResidual::projected_into_reduced_range` certifies only that
    /// the dropped component is genuinely invisible to this kernel (no retained-
    /// range leakage / orthonormal `U_S`) before the IFT correction uses it. The
    /// returned gradient lives on the identifiable manifold, matching the
    /// projected `½ log|M|₊` term.
    ///
    /// Costs `O(p·r + r²)` for the two `U_S`-contractions plus the `r × r`
    /// solve — strictly cheaper than the `O(p²)` full `hop.solve_multi`
    /// when `r ≪ p`, and bounded regardless of `σ_min(H)`.
    pub fn apply_pseudo_inverse(&self, a: &Array1<f64>) -> Array1<f64> {
        // The one sensitivity operator (#935): the projected inverse action
        // `U_S · H_proj⁻¹ · U_Sᵀ · a` has a single spelling, shared with every
        // other consumer of `FittedInverse::Projected`.
        self.sensitivity().apply(a)
    }

    /// View this projected trace kernel as the unified [`FitSensitivity`]
    /// (#935) over the rank-deficient LAML convention `K = U_S · H_proj⁻¹ ·
    /// U_Sᵀ`. The trace machinery stays here; the *inverse action* is the
    /// shared operator, so no site can disagree about what `H⁻¹` means.
    pub fn sensitivity(&self) -> crate::sensitivity::FitSensitivity<'_> {
        crate::sensitivity::FitSensitivity::from_projected(&self.u_s, &self.h_proj_inverse)
    }

    /// Build the **constrained pseudo-inverse kernel**
    /// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`
    /// from this penalty-projected kernel `K_S` and the *active* row block
    /// `A_act` of the joint linear inequality constraint matrix.
    ///
    /// `K_T` is the **Moore-Penrose pseudo-inverse of `H` restricted to
    /// `T = range(S₊) ∩ ker(A_act)`** — the smooth manifold the inner
    /// solver actually moves on at a constrained-stationary point. It is
    /// exactly the kernel that solves the per-coordinate saddle-point
    /// IFT system
    ///
    /// ```text
    ///   [ H   Aᵀ_act ] [ ∂β/∂ρ_k ]   [ −a_k ]
    ///   [ A_act  0   ] [ ∂λ/∂ρ_k ] = [   0  ]
    /// ```
    ///
    /// with `∂β/∂ρ_k = −K_T · a_k`. Using `K_T` for the per-coordinate
    /// mode response `v_k` makes the outer gradient the *exact* derivative
    /// of the projected Laplace cost `log|U_Tᵀ H U_T|`, where `U_T` is an
    /// orthonormal basis of `T` — the marginal-likelihood determinant the
    /// inner is actually drawing on.
    ///
    /// Returns a [`ConstrainedSubspaceKernel`] handle that caches the
    /// small `k_active × k_active` Schur complement so subsequent
    /// `apply_pseudo_inverse` calls for different RHS reuse it. When the
    /// active set is empty the handle degrades to a pass-through over
    /// `self` (no extra work).
    ///
    /// Total precompute cost: `k_active` calls to
    /// [`Self::apply_pseudo_inverse`] (one per active row) plus a
    /// `k_active × k_active` Cholesky/QR. Per-vector `apply` cost: one
    /// `K_S` apply + one `k_active × p` matvec + one small triangular
    /// solve + one `p × k_active` matvec.
    pub fn with_active_constraints<'a>(
        &'a self,
        a_act: ndarray::ArrayView2<'a, f64>,
    ) -> ConstrainedSubspaceKernel<'a> {
        let k_active = a_act.nrows();
        if k_active == 0 {
            return ConstrainedSubspaceKernel {
                kernel: self,
                z: Array2::zeros((0, self.u_s.nrows())),
                a_act,
                m_inv: Array2::zeros((0, 0)),
                k_active: 0,
            };
        }
        // Z = K_S · Aᵀ_act,  shape (p × k_active).
        let p = self.u_s.nrows();
        let mut z = Array2::<f64>::zeros((p, k_active));
        for j in 0..k_active {
            let a_row = a_act.row(j).to_owned();
            let k_s_a_row = self.apply_pseudo_inverse(&a_row);
            z.column_mut(j).assign(&k_s_a_row);
        }
        // M = A_act · Z   (shape k_active × k_active, symmetric PSD on
        // range(K_S) ∩ image(A_actᵀ); on a rank-deficient overlap we
        // add a tiny diagonal regulariser so the inversion remains
        // bounded — same noise-floor strategy as elsewhere in this
        // module).
        let mut m = a_act.dot(&z);
        // Symmetrise (numerical noise from the matmul leaves small skew).
        for i in 0..k_active {
            for j in 0..i {
                let avg = 0.5 * (m[[i, j]] + m[[j, i]]);
                m[[i, j]] = avg;
                m[[j, i]] = avg;
            }
        }
        // Eigendecomposition-based Moore-Penrose pseudo-inverse with a
        // relative spectral cutoff. This is the principled treatment of
        // rank deficiency in `A_act` when restricted to `range(S₊)`:
        // some active constraint rows may be linearly dependent after
        // projection (e.g. several monotonicity rows pinning the same
        // flat region all reduce to the same row in `range(S₊)`).
        // A plain `M⁻¹` then amplifies near-null directions; the
        // pseudo-inverse drops them at a relative threshold
        // `tol = eps · k_active · σ_max(M)`, which is the standard
        // NumPy/LAPACK convention and exactly what Codex flagged as
        // necessary in the math review.
        let (evals, evecs) = m
            .eigh(faer::Side::Lower)
            .unwrap_or_else(|_| (Array1::<f64>::zeros(k_active), Array2::<f64>::eye(k_active)));
        let sigma_max = evals.iter().copied().fold(0.0_f64, f64::max).max(0.0);
        let tol = f64::EPSILON * (k_active as f64) * sigma_max.max(1.0);
        let mut m_inv = Array2::<f64>::zeros((k_active, k_active));
        let mut dropped = 0usize;
        for q in 0..k_active {
            if evals[q] > tol {
                let inv_sigma = 1.0 / evals[q];
                // Outer product u_q u_qᵀ scaled by 1/σ_q.
                for i in 0..k_active {
                    for j in 0..k_active {
                        m_inv[[i, j]] += inv_sigma * evecs[[i, q]] * evecs[[j, q]];
                    }
                }
            } else {
                dropped += 1;
            }
        }
        if dropped > 0 {
            log::debug!(
                "[constrained-subspace kernel] dropped {} of {} active-constraint directions \
                 (rank-deficient on range(S₊)); pseudo-inverse threshold = {:.3e}",
                dropped,
                k_active,
                tol,
            );
        }
        ConstrainedSubspaceKernel {
            kernel: self,
            z,
            a_act,
            m_inv,
            k_active,
        }
    }
}

/// Per-evaluation handle that combines a penalty-projected
/// [`PenaltySubspaceTrace`] with an active inequality-constraint block,
/// producing the constraint-aware pseudo-inverse
/// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. See
/// [`PenaltySubspaceTrace::with_active_constraints`] for the math.
///
/// Caches the small `k_active × k_active` Schur inverse so subsequent
/// per-coordinate `apply` calls only do `O(p · k_active)` work each.
pub struct ConstrainedSubspaceKernel<'a> {
    pub(crate) kernel: &'a PenaltySubspaceTrace,
    /// `Z = K_S · Aᵀ_act`, shape `(p × k_active)`.
    pub(crate) z: Array2<f64>,
    /// Active-row block of the joint constraint matrix.
    pub(crate) a_act: ndarray::ArrayView2<'a, f64>,
    /// `(A_act · K_S · Aᵀ_act)⁻¹`, shape `(k_active × k_active)`.
    pub(crate) m_inv: Array2<f64>,
    pub(crate) k_active: usize,
}

impl<'a> ConstrainedSubspaceKernel<'a> {
    /// Apply `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S` to `a`. The result
    /// lies in `range(S₊) ∩ ker(A_act)` — the smooth manifold the inner
    /// solver actually moves on at a constrained-stationary point.
    pub fn apply_pseudo_inverse(&self, a: &Array1<f64>) -> Array1<f64> {
        let v_s = self.kernel.apply_pseudo_inverse(a);
        if self.k_active == 0 {
            return v_s;
        }
        // mu = M_inv · (A_act · v_s)
        let t = self.a_act.dot(&v_s);
        let mu = self.m_inv.dot(&t);
        // v = v_s - Z · mu
        let correction = self.z.dot(&mu);
        v_s - &correction
    }

    /// Whether any active constraints contribute (when false this kernel
    /// is identical to the bare [`PenaltySubspaceTrace::apply_pseudo_inverse`]).
    pub fn has_active_constraints(&self) -> bool {
        self.k_active > 0
    }
}

/// Tangency self-audit gate for the constrained mode-response arm: the
/// emitted `v = K_T · rhs` must lie in `ker(A_act)` by construction, so
/// `|A_act · v|` is compared against this fraction of the cancellation
/// scale `|A_act| · |v|` (per active row). Generous enough that legitimate
/// rank-deficient active sets (whose dropped Schur directions leave
/// ε-level residue, see [`PenaltySubspaceTrace::with_active_constraints`])
/// never trip it; the historical failure mode it guards (the d6b17a7f
/// `1/σ_min ≈ 10¹²` null-space amplification) exceeds it by six orders.
pub(crate) const THETA_MODE_RESPONSE_TANGENCY_GATE: f64 = 1e-6;

/// #931 migration pass 2 — the ThetaDirection shared-drift pass: the ONE
/// per-evaluation selection of the IFT mode-response kernel behind every
/// `dβ̂/dθ = −K · ∂g/∂θ` solve in the outer gradient/Hessian assembly.
///
/// Before this object existed, four sites (the gradient solve stack in
/// `reml_laml_evaluate`, the ρ- and ext-coordinate standalone fallbacks in
/// `compute_outer_hessian`, and the standalone fallback in
/// `build_outer_hessian_operator`) each re-implemented the same selection
/// rule by hand, with comments warning each other to "mirror the
/// selection exactly, otherwise the operator-form Hessian and dense
/// materialization disagree on every entry". A hand-copied convention every
/// caller must remember is precisely the objective↔gradient desync surface
/// (#748/#752/#901 class) the criterion-as-atoms architecture (#931)
/// removes. Now the rule is DECIDED in exactly one constructor and every
/// consumer is a contraction of the same kernel object — the gradient and
/// both Hessian representations structurally cannot pick different
/// inverses for the same evaluation point:
///
///   * Active inequality constraints recorded on the inner solution → the
///     lifted constrained kernel
///     `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. The inner SCOP solver
///     clamps β̂(θ) onto `T = range(S₊) ∩ ker(A_act)`, so the true IFT
///     derivative lives in T and the lifted kernel gives the minimum-norm
///     solution there; the full solve would amplify any RHS component
///     outside `range(H_free)` by `1/σ_min(H_active_normal)` — ~10¹² on
///     large-scale survival marginal-slope (commit d6b17a7f).
///   * Otherwise → the FULL Hessian solve `v = H⁻¹ · rhs`, even when the
///     LAML cost surface uses the projected logdet `½ log|U_Sᵀ H U_S|`:
///     the inner solver converges β̂ ∈ R^p in the unconstrained full
///     space, so the IFT identity demands the full inverse, and the
///     penalty-subspace projection acts on the TRACE contraction side
///     only. Routing through bare `K_S` here would discard the
///     `null(S₊)` component of dβ̂/dθ — the near-separable ψ-gradient
///     blow-up pinned by `duchon_probit_per_row_dnu_dpsi_fd_vs_analytic`.
///
/// The two emission shapes (`respond_one` per-vector, `respond_stack`
/// batched) exist because the call sites have different RHS layouts and
/// their solve shapes must stay bit-identical to the pre-port assembly
/// (per-column GEMV vs blocked GEMM sum in different orders) — NOT because
/// a site may choose a different kernel. Both shapes dispatch on the same
/// stored decision.
///
/// This is the `Sensitivity`-operator half of the `ThetaDirection`
/// calculus sketched in `atoms.rs`: the direction's `β̇` channel is a
/// contraction of this kernel, so atoms borrowing the shared drift can no
/// longer see a different chain rule than their neighbors.
pub(crate) struct ThetaModeResponseKernel<'s> {
    pub(crate) hop: &'s dyn HessianFactorization,
    /// `Some` exactly when the selection rule chose the lifted constrained
    /// kernel. Built once per evaluation point (one Schur-complement
    /// factorization), shared by every gradient/Hessian consumer — the
    /// pre-port code rebuilt it per consumer site.
    pub(crate) constrained: Option<ConstrainedSubspaceKernel<'s>>,
}

impl<'s> ThetaModeResponseKernel<'s> {
    /// The ONE place the mode-response kernel selection rule lives.
    pub(crate) fn select(
        subspace: Option<&'s PenaltySubspaceTrace>,
        active_constraints: Option<&'s ActiveLinearConstraintBlock>,
        hop: &'s dyn HessianFactorization,
    ) -> Self {
        let constrained = match (subspace, active_constraints) {
            (Some(kernel), Some(block)) => {
                let ck = kernel.with_active_constraints(block.a.view());
                ck.has_active_constraints().then_some(ck)
            }
            _ => None,
        };
        Self { hop, constrained }
    }

    /// Mode response for one right-hand side: `K_T · rhs` under active
    /// constraints, `H⁻¹ · rhs` (single-RHS `solve`) otherwise. Used by the
    /// per-coordinate fallbacks whose pre-port assembly solved one vector at
    /// a time — the single-RHS shape is preserved bit-identically.
    pub(crate) fn respond_one(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self.constrained.as_ref() {
            Some(ck) => {
                let v = ck.apply_pseudo_inverse(rhs);
                self.certify_tangency(ck, &v);
                v
            }
            None => self.hop.solve(rhs),
        }
    }

    /// Mode responses for a column-stacked RHS block: per-column `K_T`
    /// applies under active constraints (the lifted kernel has no blocked
    /// form), one batched `solve_multi` otherwise (BLAS-3 / GPU batched
    /// route) — exactly the shapes the stacked call sites used pre-port.
    /// Zero RHS columns (box-masked ρ coordinates) emit exact zeros through
    /// either arm, since both kernels are linear.
    pub(crate) fn respond_stack(&self, rhs_stack: &Array2<f64>) -> Array2<f64> {
        match self.constrained.as_ref() {
            Some(ck) => {
                let mut out = Array2::<f64>::zeros(rhs_stack.raw_dim());
                for (j, col) in rhs_stack.columns().into_iter().enumerate() {
                    let v = ck.apply_pseudo_inverse(&col.to_owned());
                    self.certify_tangency(ck, &v);
                    out.column_mut(j).assign(&v);
                }
                out
            }
            None => self.hop.solve_multi(rhs_stack),
        }
    }

    /// Per-atom certify body (#934 FD-self-audit pattern, applied as an
    /// exact structural invariant): every constrained emission must lie in
    /// `ker(A_act)` — `A_act · v = 0` is the defining property of `K_T`'s
    /// range, so a violation can only mean the kernel object and the
    /// emission desynced. Checked on every constrained response (cost
    /// `O(k_active · p)`, negligible next to the apply itself) against the
    /// row-wise cancellation scale `|A_act| · |v|`; a violation does not
    /// fail the fit — it names the atom loudly in the `[CERTIFICATE]`
    /// stream, exactly like the outer-optimum criterion audit. The
    /// unconstrained arm carries no separate certify: its coherence with
    /// the criterion VALUE is audited end-to-end by the #934
    /// `OuterCriterionCertificate` at every returned optimum.
    pub(crate) fn certify_tangency(&self, ck: &ConstrainedSubspaceKernel<'_>, v: &Array1<f64>) {
        let residual = ck.a_act.dot(v);
        for (row, r) in residual.iter().enumerate() {
            let scale: f64 = ck
                .a_act
                .row(row)
                .iter()
                .zip(v.iter())
                .map(|(a, x)| (a * x).abs())
                .sum();
            if r.abs() > THETA_MODE_RESPONSE_TANGENCY_GATE * (scale + f64::EPSILON) {
                log::warn!(
                    "[CERTIFICATE warning] atom \"theta_mode_response\": constrained IFT \
                     mode response left ker(A_act) — active row {row} residual {:.3e} \
                     exceeds gate {:.1e}·{:.3e}; the lifted kernel K_T and its emission \
                     have desynced (#931 pass-2 invariant)",
                    r.abs(),
                    THETA_MODE_RESPONSE_TANGENCY_GATE,
                    scale,
                );
            }
        }
    }
}

impl ProjectedKktResidual {
    pub(crate) fn projected_into_reduced_range(
        &self,
        kernel: &PenaltySubspaceTrace,
    ) -> Result<Self, String> {
        match self.subspace {
            KktResidualSubspace::ReducedRange => Ok(self.clone()),
            KktResidualSubspace::ActiveProjected => {
                let reduced_residual = kernel.project_onto_subspace(&self.residual);
                let dropped = &self.residual - &reduced_residual;
                // Validity invariant for reducing `r_A → r_R = U_S U_Sᵀ r_A`
                // before the projected IFT correction `−½ rᵀ K r`, `K = U_S
                // H_proj⁻¹ U_Sᵀ`.
                //
                // Since #901 the kernel's `U_S` is the ORTHONORMAL eigenbasis of
                // the full LAML Hessian `M = H + Sλ (+ H_Φ)` over its identifiable
                // range (`|σ| > positive_eigenvalue_threshold`), the SAME kept-set
                // that defines the cost's pseudo-logdet `½ log|M|₊`
                // (`joint_penalty_subspace_trace_parts`). The dropped component
                // `r_A − r_R = (I − U_S U_Sᵀ) r_A` therefore lives in `ker(M)` —
                // the genuine gauge of the cost surface — and is ANNIHILATED by the
                // kernel: `K (r_A − r_R) = U_S H_proj⁻¹ (U_Sᵀ(I − U_S U_Sᵀ) r_A) = 0`
                // because `U_Sᵀ(I − U_S U_Sᵀ) = 0` for orthonormal `U_S`. Dropping
                // it leaves the correction `−½ rᵀ K r` (and every `tr(K·)` trace)
                // bit-identical, so a gauge-direction residual of ANY magnitude is
                // valid to reduce.
                //
                // The magnitude the inner solver leaves in `ker(M)` is NOT bounded
                // by the KKT stationarity tolerance: those are the weakly-/un-
                // identified directions where `M`'s curvature is below threshold, so
                // a Newton step to resolve their residual is `r/σ` — far outside the
                // trust region — and the inner correctly leaves them unmoved
                // (gam#1040 gauge/plateau exit), the outer projected pseudo-inverse
                // correctly discards them, and the cost `½ log|M|₊` never included
                // them. Comparing that gauge mass to `residual_tol` (the previous
                // gate) is a category error inherited from the pre-#901 `range(Sλ)`
                // kernel, whose dropped subspace `null(Sλ)` DID contain likelihood-
                // identified curvature; it rejected every oversmoothed marginal-slope
                // fit as a false "unresolved mass" contract violation.
                //
                // The one property that IS required — and that this reduction must
                // certify — is that the dropped component is genuinely invisible to
                // the kernel, i.e. carries no RETAINED-range mass. For an orthonormal
                // `U_S` the leakage `U_Sᵀ(r_A − r_R)` is identically zero; a nonzero
                // value can only mean `U_S` is not orthonormal / the kernel and
                // residual desynced (the actual failure class the guard exists to
                // catch), in which case reducing WOULD alter the correction.
                let retained_leak = gam_linalg::faer_ndarray::fast_atv(&kernel.u_s, &dropped);
                let leak_inf = retained_leak
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max);
                let residual_inf = self
                    .residual
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max);
                // Orthonormal-projection round-off floor: `U_Sᵀ(I − U_S U_Sᵀ) r`
                // accumulates only `O(r_cols · ε · ‖r‖)` numerical noise, so a
                // relative floor of `1e-8·(1 + ‖r‖∞)` clears every well-formed
                // kernel by orders of magnitude while a genuinely non-orthonormal
                // `U_S` leaks `O(‖r‖)` and trips it.
                let leak_floor = 1e-8 * (1.0 + residual_inf);
                if leak_inf > leak_floor {
                    return Err(format!(
                        "projected KKT residual reduction leaks retained-range mass: \
                         |U_Sᵀ(r_A − r_R)|∞={leak_inf:.3e} > floor={leak_floor:.3e}; the \
                         reduced-range kernel U_S is not orthonormal or is desynced from the \
                         residual, so projecting r_A onto range(U_S) would change the IFT \
                         correction −½ rᵀ U_S H_proj⁻¹ U_Sᵀ r instead of leaving it invariant"
                    ));
                }
                let mut reduced = Self::from_reduced_range(reduced_residual);
                reduced.residual_tol = self.residual_tol;
                reduced.free_rank = self.free_rank;
                Ok(reduced)
            }
        }
    }

    /// The KKT-stationarity tolerance the inner solver applied at the
    /// producing iterate. Returns `None` when the residual was built
    /// from a legacy site that hasn't been threaded yet; downstream
    /// consumers should substitute `f64::NAN` in that case.
    pub fn residual_tol(&self) -> Option<f64> {
        self.residual_tol
    }

    /// Dimensionality of the free subspace: `total_p - active_set_size`
    /// at the producing iterate. `None` from legacy construction sites.
    pub fn free_rank(&self) -> Option<usize> {
        self.free_rank
    }
}
