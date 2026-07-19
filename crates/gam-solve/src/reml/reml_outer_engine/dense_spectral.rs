use super::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Dense spectral HessianFactorization implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Dense spectral Hessian operator using eigendecomposition.
///
/// Computes logdet, trace, and solve from a single eigendecomposition,
/// guaranteeing spectral consistency. Indefinite or near-singular eigenvalues
/// are handled via smooth spectral regularization `r_ε(σ)` rather than hard
/// clamping, ensuring that logdet and inverse use the same smooth mapping.
pub struct DenseSpectralOperator {
    /// Regularized eigenvalues: `r_ε(σ_i)` for each raw eigenvalue `σ_i`.
    pub(crate) reg_eigenvalues: Vec<f64>,
    /// Per-eigenvalue mask: `true` if the eigenpair participates in all
    /// traces, solves, and logdet contributions.  Under
    /// [`PseudoLogdetMode::Smooth`] every entry is `true`.  Under
    /// [`PseudoLogdetMode::HardPseudo`] entries with `σ_j ≤ ε` are `false`,
    /// so the numerical null space is excluded consistently from
    /// `log|H|_+`, its gradient, its cross-traces, AND `H⁻¹` solves
    /// (`H⁺` on the active subspace).
    pub(crate) active_mask: Vec<bool>,
    /// Eigenvectors of H (columns).
    pub(crate) eigenvectors: Array2<f64>,
    /// Precomputed: W = U diag(1/√r_ε(σ)) for efficient traces.
    /// trace(H⁻¹ A) = Σ (AW ⊙ W)
    pub(crate) w_factor: Array2<f64>,
    /// Precomputed kernel K_ab = 1 / (r_a r_b) for exact H⁻¹ cross traces in
    /// the eigenbasis.
    pub(crate) hinv_cross_kernel: Array2<f64>,
    /// Precomputed: G = U diag(1/√(√(σ² + 4ε²))) for logdet gradient traces.
    /// trace(G_ε(H) A) = Σ (AG ⊙ G) where G_ε uses φ'(σ) = 1/√(σ² + 4ε²).
    pub(crate) g_factor: Array2<f64>,
    /// Precomputed divided-difference kernel Γ for exact logdet Hessian cross traces
    /// in the eigenbasis.
    pub(crate) logdet_hessian_kernel: Array2<f64>,
    /// Precomputed log-determinant: Σ ln(r_ε(σ_i)).
    pub(crate) cached_logdet: f64,
    pub(crate) projected_factor_cache: ProjectedFactorCache,
    /// Full dimension.
    pub(crate) n_dim: usize,
    /// Raw (unregularized) eigenvalues σ_i of H, in eigenpair order. The fused
    /// second-order reductions need `σ` itself (not `r_ε(σ)`) to reassociate
    /// the logdet-Hessian diagonal cancellation-free.
    pub(crate) raw_eigenvalues: Vec<f64>,
    /// The spectral regularization scale ε used to build every kernel.
    pub(crate) epsilon: f64,
}

impl DenseSpectralOperator {
    /// Create from a symmetric matrix (may be indefinite or singular).
    ///
    /// The eigendecomposition is computed once. Eigenvalues are smoothly
    /// regularized via `r_ε(σ)`. All subsequent operations (logdet, trace,
    /// solve) use the regularized spectrum, ensuring mathematical consistency.
    pub fn from_symmetric(h: &Array2<f64>) -> Result<Self, String> {
        Self::from_symmetric_with_mode(h, PseudoLogdetMode::Smooth)
    }

    /// Variant of [`from_symmetric`](Self::from_symmetric) that selects the
    /// log-determinant convention.
    ///
    /// See [`PseudoLogdetMode`] for the derivation and the exact set of
    /// kernels that differ between the two modes.  At a high level:
    /// `Smooth` keeps every eigenpair in play with a soft floor, whereas
    /// `HardPseudo` masks out `σ_j ≤ ε` consistently across logdet,
    /// gradient traces, cross-traces, and the H⁻¹ kernels.
    pub fn from_symmetric_with_mode(
        h: &Array2<f64>,
        mode: PseudoLogdetMode,
    ) -> Result<Self, String> {
        use faer::Side;

        let n = h.nrows();
        if n != h.ncols() {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "HessianFactorization: expected square matrix, got {}×{}",
                    n,
                    h.ncols()
                ),
            }
            .into());
        }

        let (eigenvalues, eigenvectors) = h
            .eigh(Side::Lower)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        let epsilon = spectral_epsilon(eigenvalues.as_slice().unwrap());

        // `active[j]` selects which eigenpairs participate in every trace
        // and in the cached logdet.
        //
        // `Smooth` is the regularized full-spectrum mode: every eigenpair stays
        // active and singular directions are handled only through
        // `r_ε(σ)`. This is the documented default semantics used by the
        // unified REML/LAML objective.
        //
        // `HardPseudo` is the identified-subspace mode: eigenpairs with
        // `σ_j ≤ ε` are excluded consistently from logdet, traces, and solves.
        // Families that need exact pseudo-determinant behaviour opt into this
        // mode explicitly through `pseudo_logdet_mode()`.
        let active: Vec<bool> = match mode {
            PseudoLogdetMode::Smooth => vec![true; n],
            PseudoLogdetMode::HardPseudo => eigenvalues.iter().map(|&s| s > epsilon).collect(),
        };

        // Apply smooth regularization to all eigenvalues (even inactive ones:
        // `reg_eigenvalues[j]` is still consulted by `trace_hinv_product`
        // when using `w_factor[:, j]`, but we zero-out `w_factor[:, j]` for
        // inactive eigenpairs so those entries never enter any sum).
        let reg_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .map(|&sigma| spectral_regularize(sigma, epsilon))
            .collect();

        // Build W factor for traces: W[:, j] = u_j / sqrt(r_ε(σ_j)) on
        // active eigenpairs, 0 otherwise.
        let mut w_factor = Array2::zeros((n, n));
        for j in 0..n {
            if !active[j] {
                continue;
            }
            let scale = 1.0 / reg_eigenvalues[j].sqrt();
            for row in 0..n {
                w_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        let mut hinv_cross_kernel = Array2::zeros((n, n));
        for a in 0..n {
            if !active[a] {
                continue;
            }
            let inv_ra = 1.0 / reg_eigenvalues[a];
            for b in 0..n {
                if !active[b] {
                    continue;
                }
                hinv_cross_kernel[[a, b]] = inv_ra / reg_eigenvalues[b];
            }
        }

        // Build G factor for logdet gradient traces: G[:, j] = u_j / sqrt(√(σ_j² + 4ε²))
        // φ'(σ) = 1/√(σ² + 4ε²), so we need 1/√(φ'(σ)) = (σ² + 4ε²)^{1/4}
        // Actually: tr(G_ε A) = Σ_j φ'(σ_j) u_jᵀ A u_j = Σ (AG ⊙ G)
        // where G[:, j] = u_j · √(φ'(σ_j)) = u_j / (σ_j² + 4ε²)^{1/4}
        let four_eps_sq = 4.0 * epsilon * epsilon;
        let mut g_factor = Array2::zeros((n, n));
        for j in 0..n {
            if !active[j] {
                continue;
            }
            let sigma = eigenvalues[j];
            let phi_prime = 1.0 / (sigma * sigma + four_eps_sq).sqrt();
            let scale = phi_prime.sqrt();
            for row in 0..n {
                g_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        let mut logdet_hessian_kernel = Array2::zeros((n, n));
        let sqrt_disc: Vec<f64> = eigenvalues
            .iter()
            .map(|&s| (s * s + four_eps_sq).sqrt())
            .collect();
        for a in 0..n {
            if !active[a] {
                continue;
            }
            let sigma_a = eigenvalues[a];
            let sqrt_a = sqrt_disc[a];
            for b in 0..n {
                if !active[b] {
                    continue;
                }
                logdet_hessian_kernel[[a, b]] = if a == b {
                    -sigma_a / (sqrt_a * sqrt_a * sqrt_a)
                } else {
                    let sigma_b = eigenvalues[b];
                    let sqrt_b = sqrt_disc[b];
                    -(sigma_a + sigma_b) / (sqrt_a * sqrt_b * (sqrt_a + sqrt_b))
                };
            }
        }

        // Precompute logdet: Σ_{active} ln(r_ε(σ_i)).
        let cached_logdet: f64 = reg_eigenvalues
            .iter()
            .zip(active.iter())
            .filter_map(|(&v, &act)| if act { Some(v.ln()) } else { None })
            .sum();

        Ok(Self {
            reg_eigenvalues,
            active_mask: active,
            eigenvectors,
            w_factor,
            hinv_cross_kernel,
            g_factor,
            logdet_hessian_kernel,
            cached_logdet,
            projected_factor_cache: ProjectedFactorCache::default(),
            n_dim: n,
            raw_eigenvalues: eigenvalues.to_vec(),
            epsilon,
        })
    }

    #[inline]
    pub(crate) fn rotate_to_eigenbasis(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = gam_linalg::faer_ndarray::fast_atb(&self.eigenvectors, matrix);
        gam_linalg::faer_ndarray::fast_ab(&left, &self.eigenvectors)
    }

    /// Factor `F` satisfying `trace(G_epsilon(H) A) = trace(F^T A F)`.
    ///
    /// Structured row-local operators use this to contract the logdet-gradient
    /// trace directly in row space without forming `A F` in coefficient space.
    pub fn logdet_gradient_factor(&self) -> &Array2<f64> {
        &self.g_factor
    }

    /// Cancellation-free logdet-gradient reduction for a SQUARE FULL-RANK block
    /// penalty: computes `tr(G_ε(H)·λ_k S_k) − rank(S_k)` with the `−rank`
    /// subtraction distributed across eigenpairs.
    ///
    /// The ρ_k-gradient of the LAML `½·log|H|` cost carries the term
    /// `½·(tr(G_ε(H)·λ_k S_k) − ∂_{ρ_k} log|S(λ)|₊)`.  For a penalty coordinate
    /// whose block is the SOLE penalty of its span the det derivative is the
    /// exact integer `rank(S_k)`, and at the over-smoothing rail (λ_k → ∞,
    /// H ≈ λ_k S_k) the trace `tr(G_ε(H)·λ_k S_k) → rank(S_k)` so the difference
    /// of the two aggregate quantities catastrophically cancels — the surviving
    /// O(1/λ_k) gradient is then set by the last few bits of a `rank`-sized sum,
    /// which drift with the host's FMA/SIMD summation order.  Reforming the
    /// subtraction eigenpair-by-eigenpair keeps every intermediate at the true
    /// O(1/λ_k) scale, so the result is host-arithmetic-independent.
    ///
    /// Uses the identity, valid ONLY when the block is square and full rank
    /// (`end − start == rank(S_k)`), so the range projection `P_k` of `S_k` is
    /// the block-coordinate identity:
    ///
    /// ```text
    ///   tr(G_ε λ_k S_k) − rank = Σ_j [ λ_k · (g_jᵀ S_k g_j) − Σ_{i∈block} u_j[i]² ]
    /// ```
    ///
    /// where `g_j = g_factor[:, j] = √φ'(σ_j)·u_j` and `u_j = eigenvectors[:, j]`.
    /// The aggregate `Σ_j Σ_{i∈block} u_j[i]² = Σ_{i∈block} ‖U[i,:]‖² = width =
    /// rank` holds for ANY eigenbasis (the rows of the full orthogonal `U` are
    /// unit-norm, mask or no mask), so this is valid under a masked numerical
    /// null space too (`active_rank() < dim()`, `HardPseudo`): the `−rank`
    /// distribution runs over the complete (unmasked) `eigenvectors`, while the
    /// `+` trace reads `g_factor`, which is zeroed on the masked eigenpairs, so
    /// the fused value is `trace_active − rank` — exactly the active-subspace
    /// trace minus the integer rank the masked naive path pairs, and still
    /// cancellation-free (the masked pairs add only the non-negative lump
    /// `0 − Σ_{i∈block} u_j[i]²`). The `first ≈ rank` gate at the call site is
    /// what certifies the block is a proportional singleton whose det derivative
    /// is the integer rank.
    pub(crate) fn fused_logdet_gradient_minus_rank_full_block(
        &self,
        s_block: &Array2<f64>,
        start: usize,
        end: usize,
        scale: f64,
    ) -> f64 {
        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let u_block = self.eigenvectors.slice(ndarray::s![start..end, ..]);
        // S_k · g_block once: (width × n), column j is S_k g_j[block].
        let sg = s_block.dot(&g_block);
        let mut fused = 0.0;
        for j in 0..self.n_dim {
            let s_term: f64 = sg
                .column(j)
                .iter()
                .zip(g_block.column(j).iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let p_term: f64 = u_block.column(j).iter().map(|&u| u * u).sum();
            fused += scale * s_term - p_term;
        }
        fused
    }

    /// Cancellation-free logdet-gradient reduction for a RANK-DEFICIENT singleton
    /// penalty block: computes `tr(G_ε(H)·λ_k S_k) − rank(S_k)` for a block whose
    /// penalty `S_k` has a structural null space (`rank(S_k) < end − start`, e.g.
    /// a spline curvature / difference penalty).
    ///
    /// [`fused_logdet_gradient_minus_rank_full_block`] distributes the `−rank`
    /// subtraction over the block-coordinate identity `P_block`, whose trace is
    /// the block WIDTH.  That equals the det derivative `∂_{ρ_k} log|λ_k S_k|₊ =
    /// rank(S_k)` ONLY when the block is full rank (`width = rank`).  For a
    /// rank-deficient block the det derivative is still the integer `rank(S_k)`
    /// (a proportional singleton `log|λ_k S_k|₊ = rank·ρ_k + const`) but
    /// `tr(P_block) = width > rank`, so the full-block fusion over-subtracts.
    ///
    /// The correct per-eigenpair `−rank` distribution is the ORTHOGONAL
    /// PROJECTOR `P_{S_k} = Q Qᵀ` onto `range(S_k)` (with `Q` the orthonormal
    /// range basis, `QᵀQ = I_r`), because
    /// `Σ_j (u_j^{blk})ᵀ P_{S_k} (u_j^{blk}) = tr(P_{S_k}·I_block) = rank(S_k)`
    /// over the complete eigenbasis.  So per eigenpair
    ///
    /// ```text
    ///   fused += λ_k·(g_jᵀ S_k g_j) − ‖Qᵀ u_j^{blk}‖²
    /// ```
    ///
    /// is `tr(G_ε λ_k S_k) − rank(S_k)` reassociated with the subtraction matched
    /// eigenpair-by-eigenpair.  At the over-smoothing rail the `+` term (mass of
    /// `G_ε λ_k S_k` on eigendirection `j`) and the `−` term (mass of `u_j` in
    /// `range(S_k)`) each approach the same per-pair value, so their difference
    /// stays at the true O(1/λ_k) scale and never emerges from the last bits of a
    /// rank-sized aggregate — exactly as in the full-rank fusion, of which this
    /// is the strict generalization (`P_{S_k} = P_block` when `width = rank`).
    ///
    /// The `Σ_j ‖Qᵀ u_j^blk‖² = tr(P_{S_k}) = rank` identity holds for ANY
    /// eigenbasis (`Σ_j u_j u_jᵀ = I` over the complete, unmasked `eigenvectors`),
    /// so this is valid under a masked numerical null space too
    /// (`active_rank() < dim()`, `HardPseudo`): the `+` trace reads the masked
    /// `g_factor`, so the fused value is the active-subspace trace minus the
    /// integer rank, matching the masked naive pairing and cancellation-free.
    /// Callers gate only on the det derivative being the integer rank
    /// (proportional singleton, so `−rank` is the exact det pairing).
    pub(crate) fn fused_logdet_gradient_minus_rank_deficient_block(
        &self,
        s_block: &Array2<f64>,
        start: usize,
        end: usize,
        scale: f64,
    ) -> f64 {
        use faer::Side;
        let width = end - start;
        // Orthonormal basis `Q` (width × r) of `range(S_k)` from the block
        // eigenspectrum. The rank tolerance mirrors `penalty_matrix_root` so
        // `r` matches the coordinate's own `rank()` (the det derivative the
        // caller certified as the integer `−rank`).
        let (evals, evecs) = s_block
            .eigh(Side::Lower)
            .expect("rank-deficient penalty block eigendecomposition");
        let max_ev = evals.iter().copied().fold(0.0_f64, f64::max);
        let tol = (width.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);
        let active: Vec<usize> = evals
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > tol)
            .map(|(i, _)| i)
            .collect();
        let r = active.len();

        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let u_block = self.eigenvectors.slice(ndarray::s![start..end, ..]);
        // S_k · g_block once: (width × n), column j is S_k g_j[block].
        let sg = s_block.dot(&g_block);
        // Range coordinates of every eigenvector's block restriction:
        // `qt_u[:, j] = Qᵀ u_j^{blk}` (r × n), so `‖qt_u[:, j]‖²` is the mass of
        // `u_j^{blk}` inside `range(S_k)` — the per-eigenpair `−rank` share.
        let mut qt_u = Array2::<f64>::zeros((r, self.n_dim));
        for (out_row, &idx) in active.iter().enumerate() {
            for j in 0..self.n_dim {
                let mut acc = 0.0;
                for local in 0..width {
                    acc += evecs[[local, idx]] * u_block[[local, j]];
                }
                qt_u[[out_row, j]] = acc;
            }
        }

        let mut fused = 0.0;
        for j in 0..self.n_dim {
            let s_term: f64 = sg
                .column(j)
                .iter()
                .zip(g_block.column(j).iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let p_term: f64 = qt_u.column(j).iter().map(|&v| v * v).sum();
            fused += scale * s_term - p_term;
        }
        fused
    }

    /// General cancellation-free logdet-gradient reduction for a SINGLETON penalty
    /// coordinate whose det derivative `det1[k] = λ_k·tr(S_λ⁺ S_k)` is FRACTIONAL —
    /// the joint-normalizer case (`log|Σ_l λ_l S_l|₊`) that arises when penalty
    /// blocks overlap (a full-span stabilization ridge, coalesced same-span pairs,
    /// or any post-reparam coupling). Neither the block-indicator fusion
    /// ([`fused_logdet_gradient_minus_rank_full_block`]) nor the range-projector
    /// fusion ([`fused_logdet_gradient_minus_rank_deficient_block`]) applies there,
    /// because both distribute an INTEGER `−rank`, whereas the joint det derivative
    /// is not the integer rank of `S_k`.
    ///
    /// The correct per-H-eigenpair `−det1[k]` distribution uses the range chart of
    /// the JOINT penalty `S_λ = Σ_l λ_l S_l`, supplied as its whitening factor
    /// `W_S` (p × rank(S_λ), with `W_S W_Sᵀ = S_λ⁺`):
    ///
    /// ```text
    ///   w_jk = λ_k · u_jᵀ S_λ⁺ S_k u_j = λ_k · (W_Sᵀ u_j)·(W_Sᵀ S_k u_j)
    ///   fused_k = Σ_j [ φ'(σ_j)·(u_jᵀ λ_k S_k u_j) − w_jk ]
    /// ```
    ///
    /// `Σ_j w_jk = λ_k·tr(S_λ⁺ S_k) = det1[k]` identically over the complete
    /// eigenbasis, so `fused_k = tr(G_ε λ_k S_k) − det1[k]` — the same quantity the
    /// naive `trace − det1` path forms, reassociated with the subtraction matched
    /// eigenpair-by-eigenpair. At the over-smoothing rail `H → S_λ`, so
    /// `u_jᵀ[φ'(σ_j)I − S_λ⁺] → 0` per direction and each term stays at the true
    /// O(1/λ_k) scale — cancellation-free, exactly as the two special-case fusions
    /// (of which this is the strict generalization: `W_S W_Sᵀ = P_{S_k}/λ_k` when
    /// `S_λ = λ_k S_k`).
    ///
    /// Returns `(fused_k, Σ_j w_jk)`. The caller compares the returned weight sum
    /// against the cost's own `det1[k]` and only trusts the fused value when they
    /// agree — the runtime self-consistency gate that keeps this off any lane
    /// whose `det1` is not this exact joint quantity.
    ///
    /// Valid under a masked numerical null space too (`active_rank() < dim()`,
    /// `HardPseudo`): the weight sum runs over the complete (unmasked)
    /// `eigenvectors`, so `Σ_j w_jk = det1[k]` still holds (and the
    /// self-consistency gate still passes), while the `+` trace reads the masked
    /// `g_factor`, giving `trace_active − det1[k]` — the masked naive pairing,
    /// cancellation-free.
    pub(crate) fn fused_logdet_gradient_weighted_block(
        &self,
        s_block: &Array2<f64>,
        start: usize,
        end: usize,
        scale: f64,
        penalty_whitening: &Array2<f64>,
    ) -> (f64, f64) {
        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let u_block = self.eigenvectors.slice(ndarray::s![start..end, ..]);
        // Trace-term factor: sg[:,j] = S_k g_j[block]; s_k_u[:,j] = S_k u_j[block].
        let sg = s_block.dot(&g_block);
        let s_k_u_block = s_block.dot(&u_block);
        // W_Sᵀ u_j for every H-eigenvector (r × n), and W_Sᵀ (S_k u_j) using only
        // the block rows of S_k u_j (nonzero only there).
        let wt_u = gam_linalg::faer_ndarray::fast_atb(penalty_whitening, &self.eigenvectors);
        let ws_block = penalty_whitening.slice(ndarray::s![start..end, ..]);
        let wt_sk_u = gam_linalg::faer_ndarray::fast_atb(&ws_block.to_owned(), &s_k_u_block);

        let mut fused = 0.0;
        let mut weight_sum = 0.0;
        for j in 0..self.n_dim {
            let trace_term: f64 = scale
                * sg.column(j)
                    .iter()
                    .zip(g_block.column(j).iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
            let w_jk: f64 = scale
                * wt_u
                    .column(j)
                    .iter()
                    .zip(wt_sk_u.column(j).iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>();
            fused += trace_term - w_jk;
            weight_sum += w_jk;
        }
        (fused, weight_sum)
    }

    /// Cancellation-free logdet-HESSIAN diagonal reduction for a block-local
    /// penalty coordinate: computes `tr(G_ε Ḧ_kk) + Γ-cross(Ḣ_k, Ḣ_k)` — the
    /// full `2·L_kk` logdet contribution of `∂²_{ρ_k} ½log|H|` — with the
    /// second-order trace pair reassociated eigenpair-by-eigenpair.
    ///
    /// For a pure penalty coordinate (`Ḣ_k = Ḧ_kk = A = λ_k S_k`, no family
    /// curvature correction) the naive assembly forms
    ///
    /// ```text
    ///   base  =  tr(G_ε A)          = Σ_a φ'_a Ã_aa            → rank  (rail)
    ///   cross =  Σ_{ab} Γ_ab Ã_ab²                              → −rank (rail)
    /// ```
    ///
    /// (`Ã = Uᵀ A U`, `φ'(σ) = 1/√(σ²+4ε²)`, `Γ` the divided-difference kernel
    /// with `Γ_aa = −σ_a/(σ_a²+4ε²)^{3/2}`).  At the over-smoothing rail
    /// (`H ≈ λ_k S_k`) each aggregate approaches `rank(S_k)` and their sum's
    /// true value is the O(1/λ_k) tail curvature `+c·e^{−ρ}` — which the naive
    /// difference of two rank-sized sums surrenders to summation-order noise.
    /// This is the SECOND-ORDER instance of the #2298 rail trace−rank
    /// cancellation (first order fixed by the fused gradient reductions above);
    /// it is why deep-λ outer Hessian diagonals came back `≈ g` with the
    /// gradient's own sign (`hessian_psd=NO` at every deep-λ point, #2348).
    ///
    /// The reassociation pairs each diagonal eigenterm with its own cross
    /// share.  Exactly, per active eigenpair `a`:
    ///
    /// ```text
    ///   φ'_a Ã_aa + Γ_aa Ã_aa²  =  φ'_a · Ã_aa · (σ_a·b_a + 4ε²) / (σ_a²+4ε²),
    ///   b_a = σ_a − Ã_aa
    /// ```
    ///
    /// (algebraic identity: substitute `Γ_aa = −σ_a φ'_a/(σ_a²+4ε²)`).  The
    /// residual `b_a` is the mass of `H − A` on eigendirection `a` — O(1) at
    /// the rail while `σ_a, Ã_aa` are O(λ_k) — so every term stays at the true
    /// O(1/λ_k) scale.  Off-diagonal cross terms `Γ_ab Ã_ab²` (a ≠ b) are kept
    /// as-is: for in-block pairs `Ã_ab = −u_aᵀ(H−A)u_b` is already O(1) and
    /// `Γ_ab = O(1/λ²)`, so they are single products with no cancellation.
    ///
    /// Value-identical to `trace_logdet_block_local(S_k, λ_k, ..) +
    /// trace_logdet_hessian_cross(A, A)`; callers gate on
    /// `active_rank() == dim()` and on the coordinate having no drift
    /// correction (`Ḣ_k = A_k`), mirroring the fused-gradient gating.
    pub(crate) fn fused_logdet_hessian_diagonal_block(
        &self,
        s_block: &Array2<f64>,
        start: usize,
        end: usize,
        scale: f64,
    ) -> f64 {
        let u_block = self.eigenvectors.slice(ndarray::s![start..end, ..]);
        // Ã = scale · u_blkᵀ (S · u_blk): n × n, symmetric.
        let su = s_block.dot(&u_block);
        let mut a_tilde = u_block.t().dot(&su);
        a_tilde.mapv_inplace(|v| v * scale);

        let four_eps_sq = 4.0 * self.epsilon * self.epsilon;
        let mut fused = 0.0;
        for a in 0..self.n_dim {
            if !self.active_mask[a] {
                continue;
            }
            let sigma = self.raw_eigenvalues[a];
            let disc = sigma * sigma + four_eps_sq;
            let phi_prime = 1.0 / disc.sqrt();
            let t = a_tilde[[a, a]];
            let residual = sigma - t;
            fused += phi_prime * t * (sigma * residual + four_eps_sq) / disc;
            for b in 0..self.n_dim {
                if b == a || !self.active_mask[b] {
                    continue;
                }
                let cross = a_tilde[[a, b]];
                fused += self.logdet_hessian_kernel[[a, b]] * cross * cross;
            }
        }
        fused
    }

    #[inline]
    pub(crate) fn trace_hinv_product_cross_rotated(
        &self,
        a_rot: &Array2<f64>,
        b_rot: &Array2<f64>,
    ) -> f64 {
        let mut result = 0.0;
        for ((kernel_row, a_row), b_col) in self
            .hinv_cross_kernel
            .rows()
            .into_iter()
            .zip(a_rot.rows().into_iter())
            .zip(b_rot.columns().into_iter())
        {
            for ((kernel, a_value), b_value) in kernel_row
                .iter()
                .copied()
                .zip(a_row.iter().copied())
                .zip(b_col.iter().copied())
            {
                result += kernel * a_value * b_value;
            }
        }
        result
    }

    #[inline]
    pub(crate) fn trace_hinv_product_cross_dense(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let a_rot = self.rotate_to_eigenbasis(a);
        if std::ptr::eq(a, b) {
            return self.trace_hinv_product_cross_rotated(&a_rot, &a_rot);
        }
        let b_rot = self.rotate_to_eigenbasis(b);
        self.trace_hinv_product_cross_rotated(&a_rot, &b_rot)
    }

    #[inline]
    pub(crate) fn projected_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = gam_linalg::faer_ndarray::fast_atb(&self.w_factor, matrix);
        gam_linalg::faer_ndarray::fast_ab(&left, &self.w_factor)
    }

    #[inline]
    pub(crate) fn projected_operator(
        &self,
        factor: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> Array2<f64> {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result = op.projected_matrix_cached(factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::projected_operator dim={} rank={} implicit={}",
                self.n_dim,
                factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.projected_matrix_cached(factor, &self.projected_factor_cache)
        }
    }

    #[inline]
    pub(crate) fn trace_projected_cross(&self, left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        let mut result = 0.0;
        for (left_row, right_col) in left.rows().into_iter().zip(right.columns().into_iter()) {
            for (left_value, right_value) in left_row.iter().copied().zip(right_col.iter().copied())
            {
                result += left_value * right_value;
            }
        }
        result
    }

    #[inline]
    pub(crate) fn trace_logdet_hessian_cross_rotated(
        &self,
        h_i_rot: &Array2<f64>,
        h_j_rot: &Array2<f64>,
    ) -> f64 {
        let mut result = 0.0;
        for ((kernel_row, h_i_row), h_j_col) in self
            .logdet_hessian_kernel
            .rows()
            .into_iter()
            .zip(h_i_rot.rows().into_iter())
            .zip(h_j_rot.columns().into_iter())
        {
            for ((kernel, h_i_value), h_j_value) in kernel_row
                .iter()
                .copied()
                .zip(h_i_row.iter().copied())
                .zip(h_j_col.iter().copied())
            {
                result += kernel * h_i_value * h_j_value;
            }
        }
        result
    }
}

/// Coalesce repeated identical `[STAGE]` log lines from `DenseSpectralOperator`
/// methods. First occurrence of a (method, dims, implicit-flags) signature
/// logs immediately; identical consecutive repeats are silenced and accrue
/// into a counter, emitting heartbeat summaries at doubling cadence
/// (2, 4, 8, 16, …) and a final summary when the signature changes.
pub(crate) fn dense_spectral_stage_log(signature: &str, elapsed_s: f64) {
    use std::sync::Mutex;
    struct Repeat {
        pub(crate) signature: String,
        pub(crate) count: u64,
        pub(crate) total: f64,
        pub(crate) min: f64,
        pub(crate) max: f64,
        pub(crate) next_heartbeat: u64,
    }
    static REPEAT: Mutex<Option<Repeat>> = Mutex::new(None);

    let mut guard = match REPEAT.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some(state) = guard.as_mut() {
        if state.signature == signature {
            state.count += 1;
            state.total += elapsed_s;
            if elapsed_s < state.min {
                state.min = elapsed_s;
            }
            if elapsed_s > state.max {
                state.max = elapsed_s;
            }
            if state.count >= state.next_heartbeat {
                log::info!(
                    "[STAGE] {} (×{} so far, total={:.3}s min={:.3}s max={:.3}s avg={:.3}s)",
                    state.signature,
                    state.count,
                    state.total,
                    state.min,
                    state.max,
                    state.total / state.count as f64,
                );
                state.next_heartbeat = state.next_heartbeat.saturating_mul(2);
            }
            return;
        }
        // Signature changed — flush a final summary for the previous one
        // when it ran more than once (the first occurrence already logged
        // its own line, so a count of 1 needs no follow-up).
        if state.count > 1 {
            log::info!(
                "[STAGE] {} final ×{} total={:.3}s min={:.3}s max={:.3}s avg={:.3}s",
                state.signature,
                state.count,
                state.total,
                state.min,
                state.max,
                state.total / state.count as f64,
            );
        }
    }

    log::info!("[STAGE] {} elapsed={:.3}s", signature, elapsed_s);
    *guard = Some(Repeat {
        signature: signature.to_string(),
        count: 1,
        total: elapsed_s,
        min: elapsed_s,
        max: elapsed_s,
        next_heartbeat: 2,
    });
}

impl HessianFactorization for DenseSpectralOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        Ok(assemble_h_raw_dense(self))
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(H_reg⁻¹ A) = Σ_j (1/r_ε(σ_j)) uⱼᵀAuⱼ
        // Computed as Σ (AW ⊙ W) where W = U diag(1/√r_ε(σ)).
        let aw = a.dot(&self.w_factor);
        aw.iter()
            .zip(self.w_factor.iter())
            .map(|(&a, &w)| a * w)
            .sum()
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // H_reg⁻¹ v = Σ_j (1/r_ε(σ_j)) (uⱼᵀv) uⱼ.  Inactive eigenpairs
        // (σ_j ≤ ε under `HardPseudo`) are skipped so the returned vector
        // lives entirely in the active subspace — otherwise v_k picks up a
        // huge spurious component along the numerical null space direction
        // (coefficient ~ 1/r_ε(σ_j) for σ_j ≈ 0) that is not part of the
        // IFT mode response `dβ̂/dρ` and would leak into the REML gradient.
        let mut result = Array1::zeros(self.n_dim);
        for j in 0..self.n_dim {
            if !self.active_mask[j] {
                continue;
            }
            let u = self.eigenvectors.column(j);
            let coeff = u.dot(rhs) / self.reg_eigenvalues[j];
            for row in 0..self.n_dim {
                result[row] += coeff * u[row];
            }
        }
        result
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut projected = self.eigenvectors.t().dot(rhs);
        for j in 0..self.n_dim {
            if self.active_mask[j] {
                let scale = 1.0 / self.reg_eigenvalues[j];
                projected.row_mut(j).mapv_inplace(|value| value * scale);
            } else {
                // Zero out inactive eigendirections so `H⁺` acts on the
                // active subspace only (mirroring the single-vector `solve`).
                projected.row_mut(j).fill(0.0);
            }
        }
        self.eigenvectors.dot(&projected)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.trace_hinv_product_cross_dense(a, b)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result =
                op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::trace_hinv_operator dim={} rank={} implicit={}",
                self.n_dim,
                self.w_factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache)
        }
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let left = self.w_factor.t().dot(matrix).dot(&self.w_factor);
        let right = self.projected_operator(&self.w_factor, op);
        self.trace_projected_cross(&left, &right)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let left_proj = self.projected_operator(&self.w_factor, left);
            let result = if std::ptr::addr_eq(left, right) {
                self.trace_projected_cross(&left_proj, &left_proj)
            } else {
                let right_proj = self.projected_operator(&self.w_factor, right);
                self.trace_projected_cross(&left_proj, &right_proj)
            };
            let signature = format!(
                "DenseSpectralOperator::trace_hinv_operator_cross dim={} rank={} left_implicit={} right_implicit={}",
                self.n_dim,
                self.w_factor.ncols(),
                left.is_implicit(),
                right.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            let left_proj = self.projected_operator(&self.w_factor, left);
            if std::ptr::addr_eq(left, right) {
                self.trace_projected_cross(&left_proj, &left_proj)
            } else {
                let right_proj = self.projected_operator(&self.w_factor, right);
                self.trace_projected_cross(&left_proj, &right_proj)
            }
        }
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) A) = Σ_j φ'(σ_j) uⱼᵀAuⱼ
        // where φ'(σ) = 1/√(σ² + 4ε²).
        // Computed as Σ (AG ⊙ G) where G = U diag(√φ'(σ)).
        let ag = a.dot(&self.g_factor);
        ag.iter()
            .zip(self.g_factor.iter())
            .map(|(&a, &g)| a * g)
            .sum()
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        // h^G_i = ‖(X G)_{i,:}‖² where G_ε = G Gᵀ and G = self.g_factor.
        // The dominant cost at large scale is the (n × p)·(p × rank) matmul
        // — for matern60 with n=320K, p=101 that's ~3.3 GFLOPs and the
        // ndarray default `.dot()` runs single-threaded (no BLAS feature
        // enabled in this crate's build), so we route through faer's parallel
        // SIMD GEMM. For operator-backed (Lazy) designs we additionally
        // stream by row chunk so we never materialize the full (n×p) block
        // at large scale.
        let n = x.nrows();
        let p = x.ncols();
        let rank = self.g_factor.ncols();
        let mut h = Array1::<f64>::zeros(n);
        if n == 0 || p == 0 || rank == 0 {
            return h;
        }
        // Issue #922: offload this n-dependent pass to the device pool when a
        // GPU was probed and n·p² clears the dispatch floor. The result is the
        // same f64 arithmetic (X·G then row-wise ‖·‖²), just relocated across
        // every device via `scatter_batched`; any failure falls through to the
        // faer CPU stream below so the REML criterion is byte-for-byte
        // unchanged on machines without a GPU.
        if let Some(gpu) =
            gam_gpu::linalg_dispatch::try_fast_spectral_leverage_diagonal(x, self.g_factor.view())
        {
            return gpu;
        }
        let chunk_rows = byte_balanced_row_chunk(p + rank, n);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                // SAFETY: `try_row_chunk` only fails on operator implementation
                // bugs — the `start..end` range is constructed from
                // `0..n = 0..x.nrows()` with `end = (start+block).min(n)`,
                // so it is always a valid sub-range of `x`. A failure here
                // means the operator violated its row-chunk contract.
                // SAFETY: row range built from 0..x.nrows(); failure means operator broke its contract.
                reml_contract_panic(format!(
                    "xt_logdet_kernel_x_diagonal: row chunk failed: {err}"
                ))
            });
            let xg = gam_linalg::faer_ndarray::fast_ab(&rows, &self.g_factor);
            for (local, row) in xg.outer_iter().enumerate() {
                h[start + local] = row.iter().map(|v| v * v).sum();
            }
            start = end;
        }
        h
    }

    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(G_ε A) = Σ (A·G ⊙ G) for block-local A.
        // Only needs G[start..end, :] — O(block² × rank) instead of O(p² × rank).
        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let ag = block.dot(&g_block);
        scale
            * ag.iter()
                .zip(g_block.iter())
                .map(|(&a, &g)| a * g)
                .sum::<f64>()
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result =
                op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::trace_logdet_operator dim={} rank={} implicit={}",
                self.n_dim,
                self.g_factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache)
        }
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        if std::ptr::eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.rotate_to_eigenbasis(h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.projected_operator(&self.eigenvectors, h_i);
        if std::ptr::addr_eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn active_rank(&self) -> usize {
        self.active_mask.iter().filter(|&&active| active).count()
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }
}
