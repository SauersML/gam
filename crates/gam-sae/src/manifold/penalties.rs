use super::*;
use gam_linalg::faer_ndarray::FaerSvd;

/// #1610 / Jeffreys — one co-firing connected component of the SAE decoder
/// Jeffreys prior. The anti-collapse penalty is the Jeffreys prior on the
/// dictionary, `π(B) ∝ √det F(B)`, i.e. `−½·log det F`, where `F = Q ∘ O` is the
/// data-weighted Fisher information of the co-active atom directions:
///
/// ```text
///   O[j,k] = ⟨B_jᵀB_j, B_kᵀB_k⟩_F / (‖B_jᵀB_j‖_F·‖B_kᵀB_k‖_F) = o_jk   (decoder
///            subspace overlap, a genuine Frobenius cosine of the PSD self-Grams,
///            so O is a PSD correlation matrix with unit diagonal),
///   Q[j,k] = q_jk  (normalized routing coactivation = cosine of the atoms'
///            activation-mass vectors, also PSD with unit diagonal, frozen per
///            assembly), and `∘` is the Hadamard product.
/// ```
///
/// `F = Q ∘ O` is PSD with unit diagonal by the Schur product theorem, so
/// `det F ∈ (0, 1]` and `−½·log det F ≥ 0`, vanishing exactly when the co-active
/// atoms are mutually orthogonal and diverging as any co-firing pair aligns
/// (`det F → 0`). This is the honest multi-atom object of which the historical
/// pairwise barrier `−μ·q·w(o)·log(1−o+ε)` was only the `K = 2` shadow
/// (`det[[1,r],[r,1]] = 1 − r²`, `r = q·o`): the Jeffreys exponent `½` is fixed —
/// there is no free strength `μ_C` — and it is the exact reparametrization-
/// invariant counter-term to the Laplace evidence's `+½·log(volume)` collapse
/// reward. Because a pair that never co-fires has `q_jk = 0`, `Q` is block
/// diagonal across co-firing components and `det F` factorizes over them: atoms
/// that never fire together contribute a determinant factor of exactly `1` (zero
/// interaction — automatic sharing at `K ≫ p`). We therefore assemble the
/// penalty per co-firing connected component (the "routed-support blocks"), never
/// a dense `K × K` Gram.
///
/// SAMPLE-SIZE FACTORIZATION.  For a component of dimension `s`, total Fisher
/// information is `I_N = N_eff F`, hence
/// `log det I_N = s log N_eff + log det F`.  The first term is independent of
/// decoder overlap and must not multiply the second.  The Jeffreys barrier is
/// therefore exactly
///
/// ```text
///   P = -1/2 sum_C log det(F_C + eps_C I),
/// ```
///
/// with no `N_eff` multiplier.  Occupancy enters only through the precision of
/// the estimated coactivation matrix and therefore through the resolution shift
/// `eps_C` below.  Multiplying `log det F` by `N_eff` would turn a fixed-dimensional
/// Jeffreys volume term into an artificial O(N) force.
///
/// DATA-DERIVED SOFTENING `ε_C` (no magic constant). `F`'s off-diagonals `q·o`
/// carry the sampling noise of the occupancy cosine `q̂` estimated from the
/// component's co-fired rows: the Fisher variance of a correlation-type
/// estimator from `N` effective samples is `(1−q²)²/N ≤ 1/N`. An `s × s`
/// symmetric perturbation with independent entry noise of std `σ = 1/√N` has
/// spectral norm concentrating at the Wigner/MP bulk edge `2σ√s`, and by Weyl
/// `|λ̂_i − λ_i| ≤ ‖E‖₂`, so an eigenvalue of `F` below
///
/// ```text
///   ε_C = 2·√(s / min_{k∈C} N_eff,k)
/// ```
///
/// is statistically indistinguishable from an exactly-collapsed direction. The
/// interior-point shift `F + ε_C·I` therefore saturates the barrier precisely at
/// the data's own resolution limit — the same MP-bulk-edge construction the rank
/// charge uses for its reconstruction spectrum (`R·(1+√(p/N_eff))²`), with
/// dispersion `R ≡ 1` because `F` is unit-diagonal (a self-normalized
/// correlation; the residual-dispersion seam the rank charge needs for its
/// data-unit spectrum has no unit to contribute here). `min` (not mean) bounds
/// the worst-case entry noise, so `ε_C` is an honest resolution floor for every
/// edge in the component. With no effective co-fired data (`N_eff → 0`) the
/// floor exceeds the whole spectrum and the identity-referenced log determinant
/// tends to zero: the barrier honestly abstains.
///
/// Local edge indices `jl, kl` index into the owning component's `atoms`.
struct BarrierComponent {
    /// Global atom indices spanned by this component.
    atoms: Vec<usize>,
    /// Co-firing edges among `atoms`.
    edges: Vec<BarrierEdge>,
    /// Data-derived interior-point softening `ε_C = 2·√(s / min_{k∈C} N_eff,k)`,
    /// the Wigner/MP bulk edge of the coactivation estimation noise (see above).
    eps: f64,
}

/// Per-assembly FROZEN separation-barrier support (see
/// [`SaeManifoldTerm::refresh_barrier_coactivation_gate`]): the co-firing pairs
/// `(j, k, q_jk)` AND the per-atom effective sample sizes
/// `N_eff,k = Σ_{i∈J_k} a_ik²`, both read from ONE truncated-support scan of the
/// routing so the Jeffreys Fisher's weights `Q` and its softening `ε_C` are
/// mutually consistent and both frozen at the same
/// chokepoint (lagged diffusivity — the gradient treats all three as constants,
/// so the line-search value must too).
#[derive(Clone, Debug)]
pub(crate) struct BarrierCoactivationGate {
    /// Co-firing pairs `(j, k, q_jk)`, `j < k`, `q_jk ∈ (0, 1]`.
    pub(crate) pairs: Vec<(usize, usize, f64)>,
    /// Per-atom `N_eff,k` over the SAME truncated active support as `q`'s
    /// denominators (length `k_atoms`).
    pub(crate) atom_neff: Vec<f64>,
}

/// One co-firing edge of a [`BarrierComponent`].
struct BarrierEdge {
    /// Global atom indices of the two endpoints (`j < k`).
    j: usize,
    k: usize,
    /// Local indices into the owning component's `atoms`.
    jl: usize,
    kl: usize,
    /// Frozen normalized coactivation `q_jk ∈ [0, 1]`.
    q: f64,
    /// Live rank-aware decoder subspace overlap `o_jk ∈ [0, 1]`.
    o: f64,
}

impl SaeManifoldTerm {
    pub(crate) fn live_decoder_incoherence_penalty(
        &self,
        base: &Arc<DecoderIncoherencePenalty>,
    ) -> Option<DecoderIncoherencePenalty> {
        let k_atoms = self.k_atoms();
        if k_atoms < 2 {
            return None;
        }
        let p = self.output_dim();
        let block_sizes: Vec<usize> = self.atoms.iter().map(|atom| atom.basis_size()).collect();
        let m_total: usize = block_sizes.iter().sum();
        let gates = self.assignment.assignments();
        let n = gates.nrows();
        let inv_n = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        // Coactivation `W = Gᵀ·G / n`. Build it over the per-row ACTIVE support
        // only: a pair `(j,k)` contributes to row `row` exactly when BOTH gates
        // are nonzero there, so iterating each row's active atoms and accumulating
        // their outer product yields the IDENTICAL matrix as the dense `k²·n`
        // triple loop — every skipped `(j,k,row)` term had a zero gate factor and
        // contributed nothing. For the sparse IBP/Softmax routing this term exists
        // to regularize, the per-row active set is `≪ K`, so the cost collapses
        // from `O(K²·N)` (35e12 products at K=32768) to `O(N·active²)` with no
        // change to the result — the same coactive-pairs reduction proven exact
        // for the separation barrier (#1026, `barrier_coactive_pairs`).
        // Build the SPARSE symmetrized pair list directly, never the dense
        // `K×K` matrix (8 GiB at K=32768). For a co-active pair `(j,k)` with
        // `j<k` the operator's symmetrized weight is
        //   ½·(W[j,k] + W[k,j]) = W[j,k]   (W is symmetric by construction here)
        // with `W[j,k] = (Σ_row gj·gk)·inv_n`. Accumulating the numerator only
        // over the per-row active support and visiting rows in increasing order
        // reproduces the dense `Gᵀ·G/n` entry bit-for-bit; pairs that never
        // co-fire have weight 0 and are simply absent (the dense operator skipped
        // them). Cost collapses from `O(K²·N)` to `O(N·active²)`.
        let mut num: std::collections::BTreeMap<(usize, usize), f64> =
            std::collections::BTreeMap::new();
        let mut active: Vec<(usize, f64)> = Vec::with_capacity(k_atoms.min(64));
        for row in 0..n {
            active.clear();
            for j in 0..k_atoms {
                let g = gates[[row, j]];
                if g != 0.0 {
                    active.push((j, g));
                }
            }
            for ai in 0..active.len() {
                let (j, gj) = active[ai];
                for &(k, gk) in &active[ai + 1..] {
                    let (lo, hi) = if j < k { (j, k) } else { (k, j) };
                    *num.entry((lo, hi)).or_insert(0.0) += gj * gk;
                }
            }
        }
        let pairs: Vec<(usize, usize, f64)> = num
            .into_iter()
            .filter_map(|((j, k), s)| {
                let w = s * inv_n;
                (w != 0.0).then_some((j, k, w))
            })
            .collect();
        DecoderIncoherencePenalty::new_sparse(
            PsiSlice {
                range: 0..m_total * p,
                latent_dim: Some(m_total),
            },
            block_sizes,
            p,
            pairs,
            base.weight,
            base.learnable_weight,
        )
        .ok()
        .map(|mut per_fit| {
            per_fit.rho_index = base.rho_index;
            per_fit.weight_schedule = base.weight_schedule.clone();
            per_fit
        })
    }

    /// #1026 — refresh the frozen per-assembly decoder-repulsion gate from the
    /// current decoder state. Lagged-diffusivity discipline (mirrors
    /// [`SaeManifoldAtom::refresh_intrinsic_smooth_penalty`]): the gate WEIGHT is
    /// frozen here at assembly entry so the assembly's gradient/curvature and the
    /// line-search value path use the same gate even as trial decoders move.
    ///
    /// The per-pair weight is
    /// `decoder_repulsion_strength() · gate(s_jk) / (‖B_j‖²_F·‖B_k‖²_F)` with
    /// the normalized collinearity score
    /// `s_jk = ‖B_jB_kᵀ‖²_F / (‖B_j‖²_F·‖B_k‖²_F)` and a C1 smoothstep gate that
    /// is exactly 0 below [`SAE_DECODER_REPULSION_COLLINEARITY_GATE`]. The gate is
    /// stored as the symmetric matrix the [`DecoderIncoherencePenalty`] operator
    /// reads as `coactivation` (its `pair_weight` is `½(W[j,k]+W[k,j])`, and a
    /// symmetric matrix makes that exactly `W[j,k]`). When `K < 2`, or no pair is
    /// near-collinear, the gate is `None` — the strict no-op.
    pub(crate) fn refresh_decoder_repulsion_gate(&mut self) {
        let k_atoms = self.k_atoms();
        if k_atoms < 2 {
            self.decoder_repulsion_gate = None;
            return;
        }
        // Per-atom squared Frobenius decoder norms.
        let norm_sq: Vec<f64> = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients.iter().map(|v| v * v).sum::<f64>())
            .collect();
        // Candidate pair set: only co-active atom pairs. The repulsion is the
        // anti-collapse cure for two atoms that drift onto the SAME decoder
        // direction WHILE co-firing on the same rows; an atom pair that never
        // co-fires cannot drive the per-row `H_tt` near-singular, so it does not
        // need repulsion. Restricting the gate scan to the co-active pairs keeps
        // the whole path sparse — `O(N·active²)` candidate pairs instead of the
        // dense `O(K²)` all-pairs scan (fatal at K=32768) — while the C1
        // smoothstep below still zeroes out every pair that is not near-collinear,
        // so the engaged gate is the same near-collinear, co-firing pair set.
        let candidates = self.barrier_coactive_pairs();
        let mut gate: Vec<(usize, usize, f64)> = Vec::new();
        let s0 = SAE_DECODER_REPULSION_COLLINEARITY_GATE;
        // #1610 data-derived strength, hoisted (μ_C is a per-dictionary scalar,
        // not per-pair); the gate is refreshed once per assembly.
        let repulsion_strength = self.decoder_repulsion_strength();
        for (j, k, _qjk) in candidates {
            // Both decoders need a usable scale; a ~zero decoder has no
            // direction to be collinear with, so leave the pair at 0 (the
            // decoder-norm / mass guards own that degeneracy, not this term).
            if !(norm_sq[j] > 0.0 && norm_sq[k] > 0.0) {
                continue;
            }
            // #2 fix — the collinearity GATE keys on the TRUE rank-aware decoder
            // subspace overlap `o_jk = ‖B_jB_kᵀ‖²_F/(‖B_jB_jᵀ‖_F·‖B_kB_kᵀ‖_F)`
            // (the squared cosine between the OUTPUT Grams `B_jᵀB_j`, `B_kᵀB_k` in
            // the Frobenius inner product; see `decoder_gram_cosine_sq`), NOT the
            // Frobenius-NORM-normalized `‖B_jB_kᵀ‖²_F/(‖B_j‖²_F‖B_k‖²_F)`. The
            // latter is a true cosine² only for rank-1 blocks and reads `1/r` for
            // two IDENTICAL rank-r subspaces (e.g. `1/2` for identical 2-row
            // decoders), which sits at/below the `0.5` gate and so left multi-row
            // co-collapse undetected. `o_jk ∈ [0,1]`, `= 1` iff the two output
            // Grams are proportional (genuine co-collapse), so identical subspaces
            // of any rank now engage the repulsion. (The `norm_sq > 0` guard above
            // and the Frobenius energy the repulsion OPERATOR then penalizes are
            // unchanged — only the on/off GATE quantity is corrected.)
            if self.atoms[j].decoder_coefficients.ncols()
                != self.atoms[k].decoder_coefficients.ncols()
            {
                continue;
            }
            let s_jk = self.decoder_gram_cosine_sq(j, k);
            // C1 smoothstep gate: 0 below s0, smooth ramp to 1 at o=1.
            let gate_value = if s_jk <= s0 {
                0.0
            } else {
                let t = ((s_jk - s0) / (1.0 - s0)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            };
            if gate_value > 0.0 {
                // #1610 — energy-NORMALIZED per-pair weight. The repulsion operator
                // weights the un-normalized cross-Gram energy
                // `‖B_jB_kᵀ‖²_F = c_jk²·‖B_j‖²_F·‖B_k‖²_F`, so dividing the weight by
                // `‖B_j‖²_F·‖B_k‖²_F` makes the realized per-pair penalty
                // `½·STRENGTH·gate·c_jk²` — a function of the DIMENSIONLESS
                // collinearity `c_jk² ∈ [0,1]` alone, identical in form to the
                // separation barrier and invariant under a global corpus rescaling
                // (every `‖B_k‖²_F` scales by `s²`, exactly cancelling the `s⁴` in
                // the cross-Gram energy). `norm_sq[j], norm_sq[k] > 0` is guaranteed
                // by the scale guard above. The strength is a derived dimensionless
                // fraction of the primary separation-barrier strength
                // (`decoder_repulsion_strength`), not an independent magic
                // constant; at unit decoder scale it reduces to the historical `1e-3`.
                let w = repulsion_strength * gate_value / (norm_sq[j] * norm_sq[k]);
                gate.push((j, k, w));
            }
        }
        self.decoder_repulsion_gate = if gate.is_empty() { None } else { Some(gate) };
    }

    /// #1625 — freeze the SEPARATION barrier's normalized-coactivation weights
    /// `q_jk` at assembly entry, the analog of [`Self::refresh_decoder_repulsion_gate`]
    /// for the #1522 collapse-prevention barrier.
    ///
    /// The Jeffreys barrier energy `−½·log det(Q ∘ O)` (see [`BarrierComponent`])
    /// weights each co-firing edge of its data-weighted Fisher `F = Q ∘ O` by the
    /// routing coactivation `q_jk = (Σ_i a_ij a_ik)/√(Σa_ij²·Σa_ik²)`. That
    /// coactivation is a function of the assignment masses `a_ik` (hence the logits
    /// the inner Newton solve moves), but the barrier's gradient assembly
    /// differentiates ONLY the decoder overlaps `o_jk` — it consumes `q_jk` as a
    /// constant weight in `F`. Recomputing `q_jk` from the trial logits in the
    /// line-search VALUE while the GRADIENT held it fixed is a value/gradient
    /// desync: the value sees a logit force the Newton step never modelled, so the
    /// inner solve cannot reach KKT stationarity in the logit block and the undamped
    /// evidence solve refuses to rank an off-optimum Laplace criterion (#1625). It
    /// is also the WRONG semantics for a collapse-prevention barrier — an atom pair
    /// must separate its decoder SHAPES, not merely route apart to dodge the
    /// measurement. (Routing-hidden duplicates — near-identical decoders that route
    /// APART to escape a coactivation-weighted force — are reclassified as
    /// redundancy under the structure search's fusion move; see the report.)
    ///
    /// Freezing `q_jk` here (lagged-diffusivity, at the SAME chokepoint as the
    /// smoothness Gram and the repulsion gate) makes the barrier a pure function of
    /// the decoder overlaps within a Newton step: value, gradient, and curvature all
    /// read this frozen weight, so they stay mutually consistent across the line
    /// search while the decoder Grams (hence `O`) still move with the trial. The
    /// weight is refreshed every assembly, so across outer iterations it tracks the
    /// converging routing exactly (a self-consistent fixed point), never lagging by
    /// more than the one in-flight step the repulsion gate also lags. `None` when no
    /// pair co-fires (the strict no-op); the value/gradient seams fall back to the
    /// live coactivation in that case so standalone (non-line-search) calls are
    /// unaffected.
    pub(crate) fn refresh_barrier_coactivation_gate(&mut self) {
        if self.k_atoms() < 2 {
            self.barrier_coactivation_gate = None;
            return;
        }
        let (pairs, atom_neff) = self.barrier_coactive_support();
        if pairs.is_empty() {
            self.barrier_coactivation_gate = None;
            return;
        }
        // The Jeffreys barrier freezes the ROUTING-derived quantities only: the
        // coactivation weights `q_jk` AND the per-atom effective sample sizes
        // `N_eff,k` (which set the softening `ε_C` of every component — see
        // [`BarrierComponent`]). The whole strength is
        // the fixed Jeffreys exponent `½` (no evidence-derived `μ_jk` to freeze).
        // The decoder overlaps `o_jk` are deliberately NOT frozen — they are the
        // shapes the barrier is actively separating, so they stay LIVE and
        // differentiated in the value/gradient/curvature.
        self.barrier_coactivation_gate = Some(BarrierCoactivationGate { pairs, atom_neff });
    }

    /// #1625 — the SEPARATION barrier's routing support: the coactivation pairs
    /// `(j, k, q_jk)` and the per-atom effective sample sizes `N_eff,k`,
    /// preferring the per-assembly FROZEN gate ([`Self::barrier_coactivation_gate`])
    /// when present so the value and gradient seams read the SAME `q_jk` and the
    /// SAME softening `ε_C` across a Newton step
    /// (see [`Self::refresh_barrier_coactivation_gate`]). Falls back to the LIVE
    /// [`Self::barrier_coactive_support`] for standalone calls made outside an
    /// inner-solve assembly (e.g. the #1522 prevention-vs-bandaid test and the
    /// owed-1026 FD battery), which evaluate value and gradient at one and the
    /// same state and so are self-consistent either way.
    pub(crate) fn barrier_coactivation_pairs(&self) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
        match &self.barrier_coactivation_gate {
            Some(gate) => (gate.pairs.clone(), gate.atom_neff.clone()),
            None => {
                // Standalone (non-line-search) call: recompute the coactivation live
                // from the current routing. Value and gradient are evaluated at the
                // same state here, so they stay self-consistent.
                self.barrier_coactive_support()
            }
        }
    }

    /// #1026 — build the [`DecoderIncoherencePenalty`] operator for the frozen
    /// repulsion gate, or `None` when no repulsion is active. Reuses the existing
    /// analytic gradient + PSD majorizer; only the gate (fed as `coactivation`)
    /// and a fixed non-learnable strength differ from a user incoherence penalty.
    pub(crate) fn live_decoder_repulsion_penalty(&self) -> Option<DecoderIncoherencePenalty> {
        let gate = self.decoder_repulsion_gate.as_ref()?;
        let k_atoms = self.k_atoms();
        if k_atoms < 2 || gate.is_empty() {
            return None;
        }
        let p = self.output_dim();
        let block_sizes: Vec<usize> = self.atoms.iter().map(|atom| atom.basis_size()).collect();
        let m_total: usize = block_sizes.iter().sum();
        if m_total == 0 || p == 0 {
            return None;
        }
        // The operator multiplies its quadratic by `weight·pair_weight`; we want
        // the effective per-pair weight to be exactly the gate weight (which
        // already folds in the #1610 energy-normalized
        // `decoder_repulsion_strength()/(‖B_j‖²_F·‖B_k‖²_F)`), so pass weight=1
        // and feed the frozen gate directly as the sparse symmetrized pair list.
        DecoderIncoherencePenalty::new_sparse(
            PsiSlice {
                range: 0..m_total * p,
                latent_dim: Some(m_total),
            },
            block_sizes,
            p,
            gate.clone(),
            1.0,
            false,
        )
        .ok()
    }

    /// #1026 — accumulate the frozen-gate decoder repulsion's gradient into
    /// `sys.gb` and its PSD curvature into `sys.hbb`, in the full-`B` β layout
    /// (so the frame transform, if engaged, picks it up exactly like the analytic
    /// β penalties). No-op (returns `false`, nothing written) when no repulsion is
    /// active. Mirrors the `DecoderIncoherence` branch of `add_sae_beta_penalty`;
    /// the penalty is non-learnable so the (empty) rho slice is inert.
    pub(crate) fn add_sae_decoder_repulsion(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        let Some(per_fit) = self.live_decoder_repulsion_penalty() else {
            return false;
        };
        let beta_dim = self.beta_dim();
        let target_beta = self.flatten_beta();
        let rho_local = Array1::<f64>::zeros(0);
        let grad = per_fit.grad_target(target_beta.view(), rho_local.view());
        for j in 0..beta_dim {
            sys.gb[j] += penalty_scale * grad[j];
        }
        if !dense_beta_curvature {
            return true;
        }
        // The repulsion's PSD (Gauss-Newton) curvature is pair-local: scatter it
        // straight into `sys.hbb` block-by-block over the frozen gate pairs rather
        // than reconstructing the dense `β × β` matrix with `β` unit-probe HVPs.
        // At `β = K·M·p` with `O(K)` collinear gate pairs the probe loop is `O(K²)`
        // (minutes in debug at K≈512); the direct scatter is `O(K)`.
        per_fit.accumulate_psd_majorizer_dense(
            target_beta.view(),
            rho_local.view(),
            penalty_scale,
            &mut sys.hbb,
        );
        true
    }

    /// #1026 — penalized-objective contribution of the frozen-gate decoder
    /// repulsion at the current decoders. Reads the SAME frozen gate the
    /// assembly used (via [`Self::live_decoder_repulsion_penalty`]), so the
    /// line-search value is consistent with the step's gradient/curvature. 0 when
    /// no repulsion is active.
    pub(crate) fn decoder_repulsion_value(&self, penalty_scale: f64) -> f64 {
        let Some(per_fit) = self.live_decoder_repulsion_penalty() else {
            return 0.0;
        };
        let target_beta = self.flatten_beta();
        let rho_local = Array1::<f64>::zeros(0);
        use gam_terms::analytic_penalties::AnalyticPenalty;
        penalty_scale * per_fit.value(target_beta.view(), rho_local.view())
    }

    // ── #1026/#1522 COLLAPSE-PREVENTION interior-point barriers ──────────────
    //
    // These two log-barriers are the anti-collapse core. Unlike the
    // collinearity-gated repulsion they have a genuine restoring force at the
    // zero-decoder collapse point and at the alignment limit. They operate
    // directly on the per-atom decoders `B_k` (shape `M_k × p`, row-major in the
    // flat-β layout: index `beta_offsets[k] + a*p + o`), so the gradient /
    // curvature written into `sys.gb` / `sys.hbb` are in EXACTLY the same
    // full-`B` β coordinates as `add_sae_decoder_repulsion` and the analytic β
    // penalties, and the matching value (`*_barrier_value`) reads the same
    // decoders — so value / gradient / Hessian never desync across the line
    // search.

    /// #1026/#1522 — SPARSE normalized coactivation: the list of `(j, k, q_jk)`
    /// for `j < k` whose truncated-support normalized coactivation `q_jk > 0`,
    /// i.e. the atom pairs that actually co-fire on at least one row. This is the
    /// separation barrier's only input — every pair that never co-fires is a strict
    /// no-op — so enumerating the co-active pairs directly avoids the `O(K²·N)`
    /// dense build that is fatal at large K (K=32768).
    ///
    /// #3 fix — the score is the TRUNCATED-SUPPORT normalized coactivation
    ///   `q_jk = (Σ_{i∈J∩K} a_ij a_ik) / sqrt(Σ_{i∈J} a_ij² · Σ_{i∈K} a_ik²)`,
    /// where `J = {i : |a_ij| > floor_i}` is atom `j`'s active support (`floor_i =
    /// SAE_COACTIVE_RELATIVE_MASS_FLOOR·peak_i`). The relative-mass floor exists for
    /// a real performance reason (SOFTMAX leaves every atom a tiny strictly-positive
    /// tail mass, so a bare `a ≠ 0` test would mark all `K` atoms active on every
    /// row and degenerate this scan to the dense `O(N·K²)` all-pairs cost, minutes
    /// at `K = 10⁴`). The DENOMINATOR energies are therefore accumulated over the
    /// SAME per-atom active support as the numerator — NOT the full-row energies —
    /// so the floor is applied consistently to both and the score is a well-defined
    /// truncated-support cosine `∈ [0, 1]` (`{i∈J∩K} ⊆ J` and `⊆ K` ⇒
    /// Cauchy–Schwarz bounds it by 1). The PRIOR implementation was a hybrid
    /// (truncated numerator over `J∩K` but full-row denominators `Σ_all a_ik²`),
    /// which was NOT the full normalized coactivation NOR the truncated one — it
    /// systematically UNDER-weighted the barrier for dense-tail (softmax)
    /// assignments (the full-row denominator inflates the divisor). For structurally
    /// sparse assignments (JumpReLU hard gate / IBP-MAP) every sub-floor entry is a
    /// hard zero, so `J` is the full nonzero support, the truncated and full sums
    /// coincide, and this is EXACTLY the full normalized coactivation to the last
    /// bit (unchanged from before). Cost is `O(N·K)` to read the gates plus
    /// `O(Σ_row active_row²)` over the per-row support.
    pub(crate) fn barrier_coactive_pairs(&self) -> Vec<(usize, usize, f64)> {
        self.barrier_coactive_support().0
    }

    /// The full LIVE routing support of the separation barrier from ONE
    /// truncated-support scan: the co-firing pairs `(j, k, q_jk)` (see
    /// [`Self::barrier_coactive_pairs`]) AND the per-atom effective sample sizes
    /// `N_eff,k = Σ_{i∈J_k} a_ik²` — the q-denominator energies themselves, i.e.
    /// the SAME summed-squared-gate currency the rank charge's
    /// `per_atom_effective_sample_size` (`fisher_n = Σ w²`) uses, restricted to
    /// the same relative-mass active support as the numerator so numerator,
    /// denominator and softening `ε_C` are one measure
    /// (for hard-gated routings — JumpReLU/IBP/TopK — the truncated and full sums
    /// coincide exactly; for softmax the sub-floor tail is dropped from ALL of
    /// them consistently). Returned together so the frozen gate can pin both at
    /// the same chokepoint.
    pub(crate) fn barrier_coactive_support(&self) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
        let k_atoms = self.k_atoms();
        if k_atoms < 2 {
            return (Vec::new(), vec![0.0; k_atoms]);
        }
        let gates = self.assignment.assignments();
        let n = gates.nrows();
        // Sparse numerator `Σ_{i∈J∩K} a_ij a_ik` and truncated-support denominator
        // energies `Σ_{i∈J} a_ij²` accumulated over the SAME active support (the
        // `#3` consistency fix): both respect the per-row relative-mass floor, so
        // the score is a well-defined truncated cosine rather than a hybrid.
        //
        // "Co-firing" on a row means carrying NON-NEGLIGIBLE mass relative to that
        // row's peak (`SAE_COACTIVE_RELATIVE_MASS_FLOOR`), not merely `a ≠ 0`. For
        // structurally sparse modes (JumpReLU/IBP) the active atoms sit far above
        // the floor and the hard zeros are excluded either way, so the support is
        // the full nonzero support and the score is the exact full normalized
        // coactivation. For SOFTMAX the sub-floor tail is dropped from BOTH sums,
        // keeping the scan `O(N·active²)` and the score consistent with the compact
        // row layout's cutoff.
        let mut energy = vec![0.0_f64; k_atoms];
        let mut num: std::collections::BTreeMap<(usize, usize), f64> =
            std::collections::BTreeMap::new();
        let mut active: Vec<usize> = Vec::new();
        for row in 0..n {
            active.clear();
            let mut peak = 0.0_f64;
            for k in 0..k_atoms {
                let a = gates[[row, k]].abs();
                if a > peak {
                    peak = a;
                }
            }
            let floor = SAE_COACTIVE_RELATIVE_MASS_FLOOR * peak;
            for k in 0..k_atoms {
                if gates[[row, k]].abs() > floor {
                    active.push(k);
                }
            }
            // Truncated-support energies: only rows where the atom is active.
            for &k in &active {
                let a = gates[[row, k]];
                energy[k] += a * a;
            }
            for ai in 0..active.len() {
                let j = active[ai];
                let gj = gates[[row, j]];
                for &k in &active[ai + 1..] {
                    *num.entry((j, k)).or_insert(0.0) += gj * gates[[row, k]];
                }
            }
        }
        let mut pairs = Vec::with_capacity(num.len());
        for ((j, k), s) in num {
            let denom = (energy[j] * energy[k]).sqrt();
            let qjk = if denom > 0.0 { s / denom } else { 0.0 };
            if qjk > 0.0 {
                pairs.push((j, k, qjk));
            }
        }
        (pairs, energy)
    }

    /// #1610 — data-derived (scale-invariant) decoder-norm-squared floor below
    /// which an atom is shape-undefined for the separation barrier. The floor is
    /// `SAE_BARRIER_ACTIVE_NORM_REL_FLOOR² · max_k ‖B_k‖²_F`, a fixed fraction of
    /// the live dictionary's largest decoder energy, so under a global rescaling
    /// of the corpus (hence the decoders) by `s²` both an atom's `‖B_k‖²_F` and
    /// this floor scale by `s²` and the abstain set is unchanged — unlike the old
    /// absolute `1e-6²` floor, which silently disabled collapse prevention on a
    /// corpus whose natural decoder scale fell below it. `norm_sq[k] = ‖B_k‖²_F`.
    /// Returns 0 when the whole dictionary is ~0 (no live atom to be a shape for,
    /// so every pair abstains via the exactly-`0` self-norm check either way).
    /// Both [`Self::separation_barrier_value`] and
    /// [`Self::add_sae_separation_barrier`] source the floor here so value and
    /// gradient/curvature use the identical abstain set across the line search.
    pub(crate) fn barrier_norm_floor_sq(norm_sq: &[f64]) -> f64 {
        let max_norm_sq = norm_sq.iter().copied().fold(0.0_f64, f64::max);
        if !(max_norm_sq > 0.0) {
            return 0.0;
        }
        let rel = SAE_BARRIER_ACTIVE_NORM_REL_FLOOR;
        rel * rel * max_norm_sq
    }

    /// #1610 — the EVIDENCE-DERIVED data-fit INSEPARABILITY `γ_jk ∈ [0, 1]` of a
    /// coactive atom pair: the largest canonical correlation between the two atoms'
    /// coactivation-weighted chart-design column spaces. This is the quantity that
    /// decides whether the SAE's joint inner (Laplace/REML) Hessian stays positive
    /// definite — i.e. whether the model evidence is even DEFINED — so it is read
    /// straight off the reconstruction objective, not from a rank-count heuristic.
    ///
    /// DERIVATION. The Gaussian reconstruction NLL is quadratic in the stacked
    /// decoders `B`. Its Hessian (Gauss–Newton) block for atoms `(j, k)` is
    /// `H_jj = Σ_i w_i a_ij² φ_j(i)ᵀφ_j(i) ⊗ I_p`, `H_jk = Σ_i w_i a_ij a_ik φ_j(i)ᵀφ_k(i) ⊗ I_p`,
    /// where `φ_·(i)` is the atom's chart-basis row, `a_i·` the (frozen) routing
    /// mass, and `w_i` the per-row reconstruction weight (the #991 design-honesty
    /// weight [`Self::row_loss_weights`]; `1` when unweighted). Writing
    /// `G_j = Σ_i w_i a_ij² φ_jᵀφ_j`, `G_k` likewise, and
    /// `C = Σ_i w_i a_ij a_ik φ_jᵀφ_k`, the two-atom data Hessian is PD in the
    /// aligning direction iff the whitened cross operator `G_j^{-1/2} C G_k^{-1/2}`
    /// has spectral norm `< 1`. That spectral norm IS `γ_jk` (a canonical
    /// correlation), so `γ_jk → 1` is exactly the data-fit becoming unable to tell
    /// the two atoms apart — the co-collapse the barrier exists to prevent — and
    /// `γ_jk → 0` is a pair the data-fit already separates on its own.
    ///
    /// #4 fix — the per-row weight `w_i` is threaded through `G_j, G_k, C` so the
    /// canonical correlation is read from the SAME weighted β-Hessian the fit
    /// actually assembles (the arrow-Schur assembly applies `√w_i` to the residual,
    /// latent Jacobian, and β basis load, so each block carries exactly one factor
    /// of `w_i`). Previously these sums were UNWEIGHTED, so on a weighted fit the
    /// barrier strength was derived from a Hessian the fit does not use. `w_i` is a
    /// SCALAR row metric on the reconstruction channel (it reduces to `G_·⊗I_p`
    /// exactly), so this is the exact weighted inseparability. If a row-DEPENDENT
    /// output whitening metric `M_i` is additionally active (`row_metric` with
    /// [`gam_problem::RowMetric::whitens_likelihood`]), the true β-Hessian block is
    /// `Σ_i w_i a_ij a_ik φ_jᵀφ_k ⊗ M_i`, which does NOT reduce to `G_·⊗I_p`; this
    /// function then computes the weighted-but-unwhitened `γ_jk` as an EXPLICIT,
    /// documented approximation (the barrier is a conditioning SAFEGUARD, not the
    /// objective, and the scalar-weighted blocks capture the dominant per-row
    /// scaling; the exact row-dependent-metric canonical correlation would require
    /// the full `M_j·p × M_k·p` generalized eigenproblem).
    ///
    /// SCALE. `γ_jk` is read from the chart design `φ`, the routing masses `a`, and
    /// the row weights `w` only — NOT from the decoder magnitudes — so it is
    /// invariant under a global decoder rescale `B_k → s·B_k` and under a global
    /// weight rescale `w → c·w` (both `G` and `C` scale by `c`, cancelling in the
    /// whitened cross), preserving the barrier's decoder-scale invariance while
    /// keying the strength to the actual evidence objective.
    ///
    /// Uses the shared spectral cutoff [`SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF`] to
    /// whiten each `G` on its live range (a rank-deficient / never-firing atom
    /// whitens to `0` in its null directions), and clamps to `[0, 1]` against
    /// round-off. `gates` is the `n × K` routing matrix (passed in so a whole
    /// dictionary's pair strengths share one read of it).
    pub(crate) fn design_inseparability_with_gates(
        &self,
        gates: ArrayView2<'_, f64>,
        j: usize,
        k: usize,
    ) -> f64 {
        let phij = &self.atoms[j].basis_values;
        let phik = &self.atoms[k].basis_values;
        let n = phij.nrows();
        let mj = phij.ncols();
        let mk = phik.ncols();
        if n == 0 || mj == 0 || mk == 0 || gates.nrows() != n {
            return 0.0;
        }
        // #4 — per-row reconstruction weights `w_i` (the same #991 design-honesty
        // weights the assembly applies as `√w_i`), or the exact unweighted path
        // when none are installed. A mismatched length falls back to unweighted.
        let row_w = self.row_loss_weights.as_deref().filter(|w| w.len() == n);
        // Coactivation-weighted design Grams and cross-Gram (small: M_· × M_·).
        let mut gj = Array2::<f64>::zeros((mj, mj));
        let mut gk = Array2::<f64>::zeros((mk, mk));
        let mut cross = Array2::<f64>::zeros((mj, mk));
        for i in 0..n {
            let wi = row_w.map_or(1.0, |w| w[i]);
            let aj = gates[[i, j]];
            let ak = gates[[i, k]];
            let aj2 = wi * aj * aj;
            let ak2 = wi * ak * ak;
            let ajk = wi * aj * ak;
            if aj2 > 0.0 {
                for a in 0..mj {
                    let pja = phij[[i, a]];
                    for b in 0..mj {
                        gj[[a, b]] += aj2 * pja * phij[[i, b]];
                    }
                }
            }
            if ak2 > 0.0 {
                for a in 0..mk {
                    let pka = phik[[i, a]];
                    for b in 0..mk {
                        gk[[a, b]] += ak2 * pka * phik[[i, b]];
                    }
                }
            }
            if ajk != 0.0 {
                for a in 0..mj {
                    let pja = phij[[i, a]];
                    for b in 0..mk {
                        cross[[a, b]] += ajk * pja * phik[[i, b]];
                    }
                }
            }
        }
        // γ = σ_max( G_j^{-1/2} C G_k^{-1/2} ), the largest canonical correlation.
        let wj = Self::symmetric_psd_inv_sqrt(&gj);
        let wk = Self::symmetric_psd_inv_sqrt(&gk);
        let whitened = wj.dot(&cross).dot(&wk);
        let sigma_max = match whitened.svd(false, false) {
            Ok((_, sv, _)) => sv.iter().copied().fold(0.0_f64, f64::max),
            Err(_) => return 0.0,
        };
        sigma_max.clamp(0.0, 1.0)
    }

    /// Symmetric inverse square root `G^{-1/2}` of a small symmetric PSD matrix via
    /// its SVD (eigendecomposition for a symmetric PSD `G`), with directions whose
    /// singular value is below [`SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF`] · σ_max mapped
    /// to `0` (Moore–Penrose whitening). Used only on the tiny `M_· × M_·` design
    /// Grams in [`Self::design_inseparability_with_gates`].
    fn symmetric_psd_inv_sqrt(g: &Array2<f64>) -> Array2<f64> {
        let m = g.nrows();
        if m == 0 {
            return Array2::<f64>::zeros((0, 0));
        }
        let (_u, sv, vt_opt) = match g.svd(false, true) {
            Ok(parts) => parts,
            Err(_) => return Array2::<f64>::zeros((m, m)),
        };
        let Some(vt) = vt_opt else {
            return Array2::<f64>::zeros((m, m));
        };
        let sigma_max = sv.iter().copied().fold(0.0_f64, f64::max);
        if !(sigma_max > 0.0) {
            return Array2::<f64>::zeros((m, m));
        }
        let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * sigma_max;
        // G^{-1/2} = Σ_r s_r^{-1/2} v_r v_rᵀ, v_r = row r of Vt (= eigenvector of
        // the symmetric PSD G). Directions below the cutoff contribute nothing.
        let mut out = Array2::<f64>::zeros((m, m));
        for (r, &s) in sv.iter().enumerate() {
            if s <= tol {
                continue;
            }
            let inv_root = 1.0 / s.sqrt();
            for a in 0..m {
                let va = vt[[r, a]];
                if va == 0.0 {
                    continue;
                }
                for b in 0..m {
                    out[[a, b]] += inv_root * va * vt[[r, b]];
                }
            }
        }
        out
    }

    /// #1610 — the EVIDENCE-DERIVED per-pair data-fit inseparability strength
    /// `μ_jk`, the stiffness of the subdominant decoder-repulsion CONDITIONER.
    ///
    /// NOTE. This is NO LONGER the primary separation barrier's strength — that
    /// barrier is now the parameter-free Jeffreys prior `−½ log det F` (see
    /// [`BarrierComponent`]), which has no per-pair `μ`. `μ_jk` survives only to
    /// scale the collinearity-gated decoder-repulsion conditioner
    /// ([`Self::decoder_repulsion_strength`], a fixed fraction of the worst-case
    /// `μ_C`), a subdominant Gauss–Newton nudge on near-collinear co-firing pairs.
    ///
    /// The repulsion is an interior-point SAFEGUARD whose job is to help keep the
    /// joint inner (Laplace/REML) Hessian positive definite; it is NOT statistical
    /// shrinkage. The principled strength is the reciprocal-margin to the data-fit's
    /// co-collapse boundary, the data-fit inseparability `γ_jk`
    /// ([`Self::design_inseparability_with_gates`]): the whitened data Hessian loses
    /// positive definiteness in the aligning direction as `γ_jk → 1`, so the
    /// conditioner stiffens without bound there and vanishes as `γ_jk → 0`. Hence
    ///
    /// ```text
    ///   μ_jk = γ_jk / max(1 - γ_jk, ε_barrier),
    /// ```
    ///
    /// (a) EVIDENCE-DERIVED — read from the reconstruction Hessian's own
    /// conditioning; and (b) DECODER-SCALE-INVARIANT — `γ_jk` depends on the chart
    /// design + routing, not the decoder magnitudes. The softening `ε_barrier`
    /// reuses [`SAE_SEPARATION_BARRIER_EPS`], so a perfectly data-degenerate pair
    /// (`γ_jk = 1`) gets the largest finite strength `1/ε_barrier`.
    ///
    /// The per-fit [`Self::separation_barrier_strength_override`] takes precedence:
    /// when set it is the absolute conditioner strength for every pair, and `0.0`
    /// stays a legitimate "conditioner off" value.
    pub(crate) fn barrier_pair_strength_with_gates(
        &self,
        gates: ArrayView2<'_, f64>,
        j: usize,
        k: usize,
    ) -> f64 {
        if let Some(over) = self.separation_barrier_strength_override {
            return over;
        }
        let gamma = self.design_inseparability_with_gates(gates, j, k);
        gamma / (1.0 - gamma).max(SAE_SEPARATION_BARRIER_EPS)
    }

    /// #1610 — a single representative dictionary stiffness `μ_C`: the WORST-CASE
    /// (largest) evidence-derived per-pair data-fit inseparability over the
    /// co-active pairs. Used ONLY as the scale of the subdominant decoder-repulsion
    /// conditioner (a fixed fraction of it) and by external callers/tests that want
    /// one scalar. The primary Jeffreys separation barrier does NOT read this — it
    /// is parameter-free (the fixed Jeffreys exponent `½`, no `μ`). The runtime
    /// override takes precedence (absolute strength); a term with no co-active pair
    /// has no collapse geometry, so the strength is `0`.
    pub(crate) fn separation_barrier_strength(&self) -> f64 {
        if let Some(over) = self.separation_barrier_strength_override {
            return over;
        }
        if self.k_atoms() < 2 {
            return 0.0;
        }
        let gates = self.assignment.assignments();
        self.barrier_coactive_pairs()
            .iter()
            .map(|&(j, k, _q)| self.barrier_pair_strength_with_gates(gates.view(), j, k))
            .fold(0.0_f64, f64::max)
    }

    /// #1610 derived decoder-repulsion strength: a dimensionless fraction
    /// [`SAE_DECODER_REPULSION_BARRIER_RATIO`] of the data-derived (or per-fit)
    /// separation-barrier strength `μ_C`. Single source for
    /// the energy-normalized per-pair repulsion weight, so the subdominant
    /// conditioner stays a fixed fraction of the primary barrier under any sweep
    /// of `μ_C`, rather than a frozen independent magic number.
    pub(crate) fn decoder_repulsion_strength(&self) -> f64 {
        SAE_DECODER_REPULSION_BARRIER_RATIO * self.separation_barrier_strength()
    }

    /// Jeffreys prior support: partition the co-firing atom pairs into connected
    /// components and attach each edge's FROZEN coactivation `q` and LIVE decoder
    /// overlap `o`. An atom below the shape-undefined norm floor (a ~zero decoder
    /// has no direction to collapse onto) is dropped from every edge, so it joins no
    /// component. Pairs that never co-fire are absent, so the Fisher `F = Q ∘ O` is
    /// block diagonal across components and `−½ log det F` factorizes over them
    /// (an atom pair that never co-fires contributes a determinant factor of exactly
    /// `1` — zero interaction). Both the value ([`Self::separation_barrier_value`])
    /// and the gradient/curvature ([`Self::add_sae_separation_barrier`]) build `F`
    /// from THIS partition, so they read the identical `Q`, `O`, and support across
    /// the line search. See [`BarrierComponent`].
    fn barrier_components(&self, norm_sq: &[f64], floor2: f64) -> Vec<BarrierComponent> {
        let (support_pairs, atom_neff) = self.barrier_coactivation_pairs();
        // Co-firing edges with a defined decoder shape at BOTH endpoints.
        let raw: Vec<(usize, usize, f64, f64)> = support_pairs
            .into_iter()
            .filter(|&(j, k, _q)| norm_sq[j] > floor2 && norm_sq[k] > floor2)
            .map(|(j, k, q)| (j, k, q, self.decoder_gram_cosine_sq(j, k)))
            .collect();
        if raw.is_empty() {
            return Vec::new();
        }
        // Union–find over the atoms appearing in an edge → co-firing components.
        let k_atoms = self.k_atoms();
        let mut parent: Vec<usize> = (0..k_atoms).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while parent[r] != r {
                r = parent[r];
            }
            let mut c = x;
            while parent[c] != r {
                let n = parent[c];
                parent[c] = r;
                c = n;
            }
            r
        }
        let mut present = vec![false; k_atoms];
        for &(j, k, _, _) in &raw {
            present[j] = true;
            present[k] = true;
            let rj = find(&mut parent, j);
            let rk = find(&mut parent, k);
            if rj != rk {
                parent[rj] = rk;
            }
        }
        // Assign each present atom a component index and a local position.
        use std::collections::BTreeMap;
        let mut local = vec![usize::MAX; k_atoms];
        let mut root_to_comp: BTreeMap<usize, usize> = BTreeMap::new();
        let mut comps: Vec<BarrierComponent> = Vec::new();
        for a in 0..k_atoms {
            if !present[a] {
                continue;
            }
            let r = find(&mut parent, a);
            let ci = *root_to_comp.entry(r).or_insert_with(|| {
                comps.push(BarrierComponent {
                    atoms: Vec::new(),
                    edges: Vec::new(),
                    eps: f64::INFINITY,
                });
                comps.len() - 1
            });
            local[a] = comps[ci].atoms.len();
            comps[ci].atoms.push(a);
        }
        for &(j, k, q, o) in &raw {
            let r = find(&mut parent, j);
            let ci = root_to_comp[&r];
            comps[ci].edges.push(BarrierEdge {
                j,
                k,
                jl: local[j],
                kl: local[k],
                q,
                o,
            });
        }
        // Data-derived softening per component (see [`BarrierComponent`]):
        // `ε_C = 2·√(s / min_{k∈C} N_eff,k)`, from the SAME (frozen-or-live)
        // truncated-support energies as the edge weights `q`, so value, gradient,
        // and curvature all read one measure. An atom index outside the support
        // vector (a stale gate across a structural edit — the assembly resets the
        // gate on structure changes, so this is defensive) reads `N_eff = 0`,
        // which blows up `ε_C` and makes the consumers abstain
        // rather than index out of bounds.
        for comp in comps.iter_mut() {
            let s = comp.atoms.len();
            let mut min = f64::INFINITY;
            for &a in &comp.atoms {
                let ne = atom_neff.get(a).copied().unwrap_or(0.0);
                min = min.min(ne);
            }
            comp.eps = if min > 0.0 {
                2.0 * (s as f64 / min).sqrt()
            } else {
                f64::INFINITY
            };
        }
        comps
    }

    /// SEPARATION barrier value — the SAE decoder Jeffreys prior
    /// `P_sep = −½ · Σ_components log det(F_C + ε_C·I)`, `F = Q ∘ O`.
    /// Exactly `0` on a
    /// mutually-orthogonal co-active set (`F = I`) and `0` for `K < 2` or a fully
    /// disjoint routing. The Jeffreys exponent `½` is fixed — there is no strength
    /// `μ_C`. `penalty_scale = 0` disables it (the "no prevention" arm).
    pub(crate) fn separation_barrier_value(&self, penalty_scale: f64) -> f64 {
        if penalty_scale == 0.0 {
            return 0.0;
        }
        if self.k_atoms() < 2 {
            return 0.0;
        }
        let norm_sq: Vec<f64> = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients.iter().map(|v| v * v).sum::<f64>())
            .collect();
        let floor2 = Self::barrier_norm_floor_sq(&norm_sq);
        let mut acc = 0.0_f64;
        for comp in &self.barrier_components(&norm_sq, floor2) {
            let s = comp.atoms.len();
            if s < 2 {
                continue;
            }
            // No effective co-fired data ⇒ the barrier honestly abstains (the
            // gradient/curvature seam applies the identical guard).
            if !comp.eps.is_finite() {
                continue;
            }
            let eps = comp.eps;
            // F = Q ∘ O over the component: unit diagonal, off-diagonal `q·o` on the
            // co-firing edges (the LIVE decoder overlap `o`, the FROZEN routing `q`).
            let mut f = Array2::<f64>::eye(s);
            for e in &comp.edges {
                let v = e.q * e.o;
                f[[e.jl, e.kl]] = v;
                f[[e.kl, e.jl]] = v;
            }
            let sv = match f.svd(false, false) {
                Ok((_, sv, _)) => sv,
                Err(_) => continue,
            };
            // −½·log det(F + ε_C·I), identity-referenced. Shifting (rather than flooring)
            // keeps the pole bounded while `(F + ε_C·I)⁻¹` is the exact derivative
            // channel of this value everywhere
            // — no sub-floor value/gradient mismatch, no Armijo conservatism at
            // the pole. The per-eigenvalue −ln(1+ε_C) reference keeps a
            // mutually-orthogonal co-active set (F = I) at exactly 0.
            let mut logdet = 0.0_f64;
            for &lam in sv.iter() {
                logdet += (lam + eps).ln() - (1.0 + eps).ln();
            }
            acc += -0.5 * logdet;
        }
        penalty_scale * acc
    }

    /// Squared cross-Gram shape energy `‖B_j B_kᵀ‖²_F = Σ_{a,b}(Σ_o B_j[a,o]B_k[b,o])²`
    /// for atoms `j, k` (the un-normalized numerator of `c_jk²`).
    fn barrier_cross_shape_energy(&self, j: usize, k: usize) -> f64 {
        let bj = &self.atoms[j].decoder_coefficients;
        let bk = &self.atoms[k].decoder_coefficients;
        let (m_j, p) = (bj.nrows(), bj.ncols());
        let m_k = bk.nrows();
        if bk.ncols() != p {
            return 0.0;
        }
        let mut cross = 0.0_f64;
        for a in 0..m_j {
            for b in 0..m_k {
                let mut c = 0.0_f64;
                for o in 0..p {
                    c += bj[[a, o]] * bk[[b, o]];
                }
                cross += c * c;
            }
        }
        cross
    }

    /// Frobenius norm `‖BBᵀ‖_F = sqrt(Σ_{a,a'}(Σ_o B[a,o]B[a',o])²)` of a decoder
    /// block's own row-Gram. Equal to `‖BᵀB‖_F` (same nonzero spectrum) and to
    /// `sqrt(Σ_r σ_r⁴)` in the singular values of `B`; `= ‖B‖²_F` only for a
    /// rank-1 block. The rank-aware normalizer of [`Self::decoder_gram_cosine_sq`].
    fn decoder_self_gram_frobenius_norm(b: &Array2<f64>) -> f64 {
        let (m, p) = (b.nrows(), b.ncols());
        let mut s = 0.0_f64;
        for a in 0..m {
            for a2 in 0..m {
                let mut c = 0.0_f64;
                for o in 0..p {
                    c += b[[a, o]] * b[[a2, o]];
                }
                s += c * c;
            }
        }
        s.sqrt()
    }

    /// #2 fix — the TRUE rank-aware decoder subspace overlap `∈ [0, 1]`: the
    /// squared cosine between the two atoms' OUTPUT Gram operators `B_jᵀB_j` and
    /// `B_kᵀB_k` in the Frobenius inner product,
    ///
    /// ```text
    ///   o_jk = ‖B_jB_kᵀ‖²_F / (‖B_jB_jᵀ‖_F · ‖B_kB_kᵀ‖_F).
    /// ```
    ///
    /// Because `‖B_jB_kᵀ‖²_F = ⟨B_jᵀB_j, B_kᵀB_k⟩_F`, Cauchy–Schwarz gives
    /// `o_jk ∈ [0, 1]`, with `o_jk = 1` iff `B_jᵀB_j ∝ B_kᵀB_k` — the two atoms
    /// occupy the SAME output subspace with a proportional spectrum, i.e. genuine
    /// co-collapse — and `o_jk = cos²θ` for rank-1 blocks. This REPLACES the
    /// Frobenius-NORM-normalized `‖B_jB_kᵀ‖²_F/(‖B_j‖²_F‖B_k‖²_F)` as the
    /// collinearity GATE quantity: that older score is a true cosine² only for
    /// rank-1 blocks and reads `1/r` for two IDENTICAL rank-r orthonormal
    /// subspaces (e.g. `B_j = B_k = I₂` → `1/2`), sitting at/below the `0.5` gate
    /// and so leaving multi-row co-collapse UNDETECTED. `o_jk` scores `1` for
    /// identical subspaces of any rank while preserving the rank-1 special case
    /// (and the rank-1 co-collapse limit, where it also → 1). Returns `0` if
    /// either self-Gram is ~0 (a shapeless / vanishing decoder). It is the LIVE,
    /// differentiated collinearity scalar of the separation barrier and repulsion
    /// gate (its analytic gradient is derived in `add_sae_separation_barrier`).
    pub(crate) fn decoder_gram_cosine_sq(&self, j: usize, k: usize) -> f64 {
        let bj = &self.atoms[j].decoder_coefficients;
        let bk = &self.atoms[k].decoder_coefficients;
        let p = bj.ncols();
        if bk.ncols() != p || p == 0 {
            return 0.0;
        }
        let cross_sq = self.barrier_cross_shape_energy(j, k);
        let dj = Self::decoder_self_gram_frobenius_norm(bj);
        let dk = Self::decoder_self_gram_frobenius_norm(bk);
        if !(dj > 0.0 && dk > 0.0) {
            return 0.0;
        }
        (cross_sq / (dj * dk)).min(1.0)
    }

    /// Accumulate the SEPARATION barrier's analytic gradient into `sys.gb` and a
    /// PSD majorizer of its curvature into `sys.hbb` (dense path) or the
    /// `atom_curv` / `sep_rank1` carriers (matrix-free / framed path), in the
    /// full-`B` β layout. Returns `true` iff anything was written.
    ///
    /// The barrier is the SAE decoder Jeffreys prior
    /// `P = −½ Σ_comp log det(F + ε_C·I)`, `F = Q ∘ O` (see
    /// [`BarrierComponent`] for the data-derived softening `ε_C`). Per component
    /// (`G ≜ (F + ε_C·I)⁻¹`):
    ///
    /// GRADIENT. `∂P/∂o_e = −G[jₑ,kₑ]·q_e` (edge `e = (j,k)`, since
    /// `F[j,k] = q_e·o_e` and `ε_C` is a frozen routing constant), and
    /// `∂o_e/∂B` is the historical rank-aware carrier
    /// `v_e`: with `M = B_jB_kᵀ`, `S_· = B_·B_·ᵀ`, `D_· = ‖S_·‖_F`,
    /// `o_e = ‖M‖²_F/(D_jD_k)`,
    ///   `∂o_e/∂B_j = 2[ (M B_k)/(D_jD_k) − (o_e/D_j²) S_j B_j ]`,
    ///   `∂o_e/∂B_k = 2[ (Mᵀ B_j)/(D_jD_k) − (o_e/D_k²) S_k B_k ]`,
    /// so `∂P/∂B = Σ_e α_e·v_e`, `α_e = penalty_scale·(−G[jₑ,kₑ]·q_e)`. For
    /// the `K = 2` component `F = [[1,r],[r,1]]`, `r = q·o`, this is
    /// `α = q²o/((1+ε)²−q²o²)·penalty_scale ≥ 0` — the same repulsive
    /// `∂o/∂B` force as the historical pairwise barrier, with the Jeffreys
    /// `½` fixing its strength and no smoothstep gate:
    /// the force vanishes as `O(o)` for separated atoms (so it cannot drag a
    /// healthy fit off the data optimum, the #1625 concern) and diverges as
    /// `det F → 0`, an automatic soft gate.
    ///
    /// CURVATURE. `F` is LINEAR in the overlaps `o_e`, so the overlap-space Hessian
    /// is exactly Gauss–Newton and PSD:
    ///   `M[a,b] = ∂²P/∂o_a∂o_b = q_a q_b (G[jₐ,m_b]G[kₐ,l_b] + G[jₐ,l_b]G[kₐ,m_b])`
    /// (`a = (jₐ,kₐ)`, `b = (l_b,m_b)`), and the β-Hessian's PSD part is
    /// `Σ_{a,b} M[a,b] v_a v_bᵀ`. Eigendecomposing `M = Σ_r λ_r e_r e_rᵀ` gives the
    /// exact rank-1 carriers `(λ_r, w_r)`, `w_r = Σ_a e_r[a] v_a`, each PSD. For a
    /// single-edge component this reduces to one rank-1 `∂²P/∂o²·v vᵀ`,
    /// bit-compatible with the historical self-concordant rank-1. The remaining
    /// indefinite `Σ_e (∂P/∂o_e)·∂²o_e/∂B²` part is handled by the per-atom
    /// Levenberg ridge `2|α_e|·o_e/D_·`, which
    /// dominates its NEGATIVE part: the
    /// negative curvature of the cosine² overlap only appears past `o > ½` and
    /// scales like `2(2o−1)⁺·|α_e|/D_· ≤ 2o·|α_e|/D_·` (at small `o` the overlap
    /// sits at its minimum, so the dropped term is PSD and needs no domination —
    /// the metric merely under-counts positive curvature there, which the line
    /// search absorbs). The total metric GN + ridge is PSD by construction.
    /// Value (`−½·Σ ln(λ+ε_C)`), gradient (`G`), and curvature (GN plus the
    /// `|α|` ridge) all read `ε_C` from the same [`BarrierComponent`], so the
    /// three seams cannot desync.
    pub(crate) fn add_sae_separation_barrier(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty_scale: f64,
        dense_beta_curvature: bool,
        atom_curv: &mut [f64],
        sep_rank1: &mut Vec<(f64, Vec<(usize, f64)>)>,
    ) -> bool {
        if penalty_scale == 0.0 {
            return false;
        }
        let k_atoms = self.k_atoms();
        if k_atoms < 2 {
            return false;
        }
        let p = self.output_dim();
        let offsets = self.beta_offsets();
        let norm_sq: Vec<f64> = self
            .atoms
            .iter()
            .map(|atom| atom.decoder_coefficients.iter().map(|v| v * v).sum::<f64>())
            .collect();
        let floor2 = Self::barrier_norm_floor_sq(&norm_sq);
        let mut wrote = false;
        for comp in &self.barrier_components(&norm_sq, floor2) {
            let s = comp.atoms.len();
            let ne = comp.edges.len();
            if s < 2 || ne == 0 {
                continue;
            }
            // No effective co-fired data ⇒ abstain, exactly like the value seam.
            if !comp.eps.is_finite() {
                continue;
            }
            let eps = comp.eps;
            // F = Q ∘ O over the component and its ε_C-shifted spectral inverse G.
            let mut f = Array2::<f64>::eye(s);
            for e in &comp.edges {
                let v = e.q * e.o;
                f[[e.jl, e.kl]] = v;
                f[[e.kl, e.jl]] = v;
            }
            let (sv, vt) = match f.svd(false, true) {
                Ok((_, sv, Some(vt))) => (sv, vt),
                _ => continue,
            };
            // G = (F + εI)⁻¹ = Σ_i vᵢ vᵢᵀ/(λ_i + ε) — the EXACT derivative of the
            // ε-shifted value −½·Σ ln(λ+ε), sharing one factorization with it.
            // The shift keeps the interior-point desideratum the old floor bought
            // (a bounded restoring force ≤ 1/ε at the pole — a collapsed state is
            // still pushed out of, never flat-lined) with zero value/gradient
            // mismatch, so the line search's Armijo contract is exact at the pole
            // too.
            let mut g = Array2::<f64>::zeros((s, s));
            for (i, &lam) in sv.iter().enumerate() {
                let inv = 1.0 / (lam + eps);
                for a in 0..s {
                    let va = vt[[i, a]];
                    if va == 0.0 {
                        continue;
                    }
                    for b in 0..s {
                        g[[a, b]] += inv * va * vt[[i, b]];
                    }
                }
            }
            // Per-edge ∂o/∂B carrier `v_e` and force scalar `α_e`; scatter the
            // gradient and the bounded Levenberg majorizer as we go.
            let mut edge_v: Vec<Vec<(usize, f64)>> = Vec::with_capacity(ne);
            for e in &comp.edges {
                let bj = &self.atoms[e.j].decoder_coefficients;
                let bk = &self.atoms[e.k].decoder_coefficients;
                let (m_j, pj) = (bj.nrows(), bj.ncols());
                let m_k = bk.nrows();
                if pj != p || bk.ncols() != p {
                    edge_v.push(Vec::new());
                    continue;
                }
                let mut cross = Array2::<f64>::zeros((m_j, m_k));
                for a in 0..m_j {
                    for b in 0..m_k {
                        let mut c = 0.0_f64;
                        for o in 0..p {
                            c += bj[[a, o]] * bk[[b, o]];
                        }
                        cross[[a, b]] = c;
                    }
                }
                let s_j = bj.dot(&bj.t());
                let s_k = bk.dot(&bk.t());
                let d_j = s_j.iter().map(|v| v * v).sum::<f64>().sqrt();
                let d_k = s_k.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !(d_j > 0.0 && d_k > 0.0) {
                    edge_v.push(Vec::new());
                    continue;
                }
                let inv_dd = 1.0 / (d_j * d_k);
                let o_overlap = e.o;
                let sh_j = o_overlap / (d_j * d_j);
                let sh_k = o_overlap / (d_k * d_k);
                // `α_e = penalty_scale·(∂P/∂o_e) =
                // penalty_scale·(−G[jl,kl]·q_e)`.
                let alpha = penalty_scale * (-g[[e.jl, e.kl]] * e.q);
                let off_j = offsets[e.j];
                let off_k = offsets[e.k];
                let mut v: Vec<(usize, f64)> = Vec::with_capacity(m_j * p + m_k * p);
                for a in 0..m_j {
                    for o in 0..p {
                        let mut mb = 0.0_f64;
                        for b in 0..m_k {
                            mb += cross[[a, b]] * bk[[b, o]];
                        }
                        let mut sjb = 0.0_f64;
                        for a2 in 0..m_j {
                            sjb += s_j[[a, a2]] * bj[[a2, o]];
                        }
                        let do_j = 2.0 * (mb * inv_dd - sh_j * sjb);
                        sys.gb[off_j + a * p + o] += alpha * do_j;
                        v.push((off_j + a * p + o, do_j));
                    }
                }
                for b in 0..m_k {
                    for o in 0..p {
                        let mut mtb = 0.0_f64;
                        for a in 0..m_j {
                            mtb += cross[[a, b]] * bj[[a, o]];
                        }
                        let mut skb = 0.0_f64;
                        for b2 in 0..m_k {
                            skb += s_k[[b, b2]] * bk[[b2, o]];
                        }
                        let do_k = 2.0 * (mtb * inv_dd - sh_k * skb);
                        sys.gb[off_k + b * p + o] += alpha * do_k;
                        v.push((off_k + b * p + o, do_k));
                    }
                }
                if alpha != 0.0 {
                    wrote = true;
                }
                // Bounded Levenberg majorizer for the indefinite `α_e·∂²o_e/∂B²`
                // part. `|α_e|` keeps it PSD regardless of the sign of `G[jl,kl]`
                // (a frustrated component can carry either sign). On the dense path
                // scatter `lev·I` onto the atom block's `hbb` diagonal; on the
                // matrix-free/framed path hand the per-atom scalar back (folded into
                // `smooth_scaled_s`, the single source for the CPU op and device
                // smooth blocks).
                let lev_j = 2.0 * alpha.abs() * o_overlap / d_j;
                let lev_k = 2.0 * alpha.abs() * o_overlap / d_k;
                if dense_beta_curvature {
                    if lev_j > 0.0 {
                        for idx in 0..(m_j * p) {
                            let gi = off_j + idx;
                            sys.hbb[[gi, gi]] += lev_j;
                        }
                    }
                    if lev_k > 0.0 {
                        for idx in 0..(m_k * p) {
                            let gi = off_k + idx;
                            sys.hbb[[gi, gi]] += lev_k;
                        }
                    }
                } else {
                    if lev_j > 0.0 {
                        atom_curv[e.j] += lev_j;
                    }
                    if lev_k > 0.0 {
                        atom_curv[e.k] += lev_k;
                    }
                }
                edge_v.push(v);
            }
            // Exact PSD Gauss–Newton curvature: the overlap-space Hessian `M`
            // (`F` linear in `o` ⇒ no `∂²F/∂o²` term), eigendecomposed into rank-1
            // carriers `(penalty_scale·λ_r, w_r)`, `w_r = Σ_a e_r[a] v_a`.
            let mut mm = Array2::<f64>::zeros((ne, ne));
            for a in 0..ne {
                let ea = &comp.edges[a];
                for b in 0..ne {
                    let eb = &comp.edges[b];
                    mm[[a, b]] = ea.q
                        * eb.q
                        * (g[[ea.jl, eb.kl]] * g[[ea.kl, eb.jl]]
                            + g[[ea.jl, eb.jl]] * g[[ea.kl, eb.kl]]);
                }
            }
            let (sm, vm) = match mm.svd(false, true) {
                Ok((_, sm, Some(vm))) => (sm, vm),
                _ => continue,
            };
            for (r_i, &lam) in sm.iter().enumerate() {
                let scale = penalty_scale * lam;
                if !(scale > 0.0) {
                    continue;
                }
                // Aggregate `w_r = Σ_a e_r[a] v_a` over global β indices (edges can
                // share an atom's decoder coefficients, so accumulate into a map).
                use std::collections::BTreeMap;
                let mut agg: BTreeMap<usize, f64> = BTreeMap::new();
                for a in 0..ne {
                    let coef = vm[[r_i, a]];
                    if coef == 0.0 {
                        continue;
                    }
                    for &(idx, val) in &edge_v[a] {
                        *agg.entry(idx).or_insert(0.0) += coef * val;
                    }
                }
                let carrier: Vec<(usize, f64)> =
                    agg.into_iter().filter(|&(_, v)| v != 0.0).collect();
                if carrier.is_empty() {
                    continue;
                }
                if dense_beta_curvature {
                    // Scatter `scale·w wᵀ` into `hbb` (diagonal + true cross coupling).
                    for &(gi, vi) in &carrier {
                        let dvi = scale * vi;
                        for &(gj, vj) in &carrier {
                            sys.hbb[[gi, gj]] += dvi * vj;
                        }
                    }
                } else {
                    sep_rank1.push((scale, carrier));
                }
                wrote = true;
            }
        }
        wrote
    }

    pub(crate) fn live_mechanism_sparsity_penalties(
        &self,
        base: &Arc<MechanismSparsityPenalty>,
    ) -> Vec<(MechanismSparsityPenalty, usize, usize)> {
        let beta_offsets = self.beta_offsets();
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.atoms.len());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let start = beta_offsets[atom_idx];
            let end = start + m * p;
            let mut per_atom: MechanismSparsityPenalty = (**base).clone();
            per_atom.target = PsiSlice {
                range: start..end,
                latent_dim: Some(m),
            };
            out.push((per_atom, start, end));
        }
        out
    }

    pub(crate) fn live_nuclear_norm_penalties(
        &self,
        base: &Arc<NuclearNormPenalty>,
    ) -> Vec<(NuclearNormPenalty, usize, usize)> {
        let beta_offsets = self.beta_offsets();
        let p = self.output_dim();
        let mut out = Vec::with_capacity(self.atoms.len());
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let start = beta_offsets[atom_idx];
            let end = start + m * p;
            let mut per_atom: NuclearNormPenalty = (**base).clone();
            per_atom.n_eff = m;
            per_atom.target = PsiSlice {
                range: start..end,
                latent_dim: Some(p),
            };
            out.push((per_atom, start, end));
        }
        out
    }

    pub(crate) fn add_sae_beta_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        // MechanismSparsityPenalty is a group-lasso over a single
        // (latent_dim, p) decoder matrix and indexes its target via
        // `target.range.start + latent * p + feature`, treating its range as
        // one contiguous (M, p) block. The flat SAE β layout concatenates the
        // per-atom decoder blocks `[B_1 (M_1×p), B_2 (M_2×p), …]`, so for K≥2
        // (and in general for K=1, where it collapses to the same single
        // block) the penalty must operate per atom on its own
        // `[beta_offsets[k] .. beta_offsets[k+1])` slice with `latent_dim = M_k`.
        // Build a per-atom view of the penalty (cloning only the cheap
        // descriptor: range + latent_dim) and accumulate each atom's
        // contribution into the corresponding β segment. This removes the
        // K≥2 limitation (#240) at root rather than guarding it away.
        // DecoderIncoherencePenalty (#671) is a cross-atom decoder
        // column-space incoherence term restricted to co-activating atom pairs.
        // Its descriptor carries only placeholder shape/co-activation: the live
        // M_k (per-atom basis sizes), p_out, β target span, and the per-pair
        // co-activation weights `W[j,k] = mean_n gate[n,j]·gate[n,k]` are all
        // injected here from the current SAE state before the penalty's
        // gradient / PSD curvature are accumulated into the β-tier system.
        if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
            let Some(per_fit) = self.live_decoder_incoherence_penalty(base) else {
                return false;
            };
            let beta_dim = self.beta_dim();
            let grad = per_fit.grad_target(target_beta, rho_local);
            for j in 0..beta_dim {
                sys.gb[j] += penalty_scale * grad[j];
            }
            if !dense_beta_curvature {
                return true;
            }
            // `hbb` is the PSD Newton / PIRLS curvature block. The Gauss-Newton
            // (PSD) majorizer is pair-local, so scatter it directly into `sys.hbb`
            // over the co-active atom pairs instead of probing all `β` unit columns:
            // the cross-atom incoherence term has `O(K)` co-active pairs but the
            // probe loop is `O(β·pairs) = O(K²)`, the high-K assembly cliff.
            per_fit.accumulate_psd_majorizer_dense(
                target_beta,
                rho_local,
                penalty_scale,
                &mut sys.hbb,
            );
            return true;
        }
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            let mut any = false;
            for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                any |= self.add_sae_mech_sparsity_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                    dense_beta_curvature,
                );
            }
            return any;
        }
        // NuclearNormPenalty is a smoothed sum of singular values of a single
        // (n_eff, latent_dim) matrix. The flat SAE β layout concatenates the
        // per-atom decoder blocks `[B_1 (M_1×p), B_2 (M_2×p), …]`, so it must
        // operate per atom on that atom's own `[beta_offsets[k] .. +M_k*p)`
        // slice as an `M_k × p` matrix (`n_eff = M_k`, `latent_dim = p`). This
        // penalizes the embedding rank of each atom's decoder independently.
        if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
            let mut any = false;
            for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                any |= self.add_sae_nuclear_norm_atom(
                    sys,
                    &per_atom,
                    target_beta,
                    rho_local,
                    start,
                    end,
                    penalty_scale,
                    dense_beta_curvature,
                );
            }
            return any;
        }
        let k = self.beta_dim();
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            sys.gb[j] += penalty_scale * grad[j];
        }
        if !dense_beta_curvature {
            return true;
        }
        // `hbb` is the PSD Newton / PIRLS curvature block for the β tier:
        // accumulate the PSD majorizer (exact for convex penalties), not the
        // indefinite exact Hessian, so the solve stays positive-definite.
        if let Some(diag) = penalty.psd_majorizer_diag(target_beta, rho_local) {
            for j in 0..k {
                sys.hbb[[j, j]] += penalty_scale * diag[j];
            }
            return true;
        }
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = penalty.psd_majorizer_hvp(target_beta, rho_local, probe.view());
            for i in 0..k {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
        true
    }

    /// Accumulate one atom's MechanismSparsity contribution into `sys`. The
    /// `per_atom` penalty has its `target.range` set to that atom's β segment
    /// `[start, end)` and `latent_dim = M_k`, so `grad_target` / `hvp` return
    /// full-length β vectors whose nonzero support lies inside `[start, end)`.
    /// The Hessian probe only needs to sweep that segment, and its support is
    /// likewise confined to `[start, end)`, so the inner accumulation is
    /// quadratic in the atom's block size rather than the full β dimension.
    pub(crate) fn add_sae_mech_sparsity_atom(
        &self,
        sys: &mut ArrowSchurSystem,
        per_atom: &MechanismSparsityPenalty,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        start: usize,
        end: usize,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        let grad = per_atom.grad_target(target_beta, rho_local);
        for j in start..end {
            sys.gb[j] += penalty_scale * grad[j];
        }
        if !dense_beta_curvature {
            return true;
        }
        let k = self.beta_dim();
        let mut probe = Array1::<f64>::zeros(k);
        for j in start..end {
            probe.fill(0.0);
            probe[j] = 1.0;
            // `hbb` is the PSD Newton / PIRLS curvature block, so probe the PSD
            // majorizer. The group-lasso Hessian `factor·(I − ŵŵᵀ)/‖w‖` is
            // already PSD, so its majorizer equals the exact Hessian (the trait
            // default delegates), but we use the majorizer name to honor the
            // curvature-block contract uniformly with the other SAE penalties.
            let hv = per_atom.psd_majorizer_hvp(target_beta, rho_local, probe.view());
            for i in start..end {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
        true
    }

    /// Accumulate one atom's NuclearNorm contribution into `sys`. The
    /// `per_atom` penalty has `n_eff = M_k` and `latent_dim = Some(p)`, so it
    /// treats this atom's β segment `[start, end)` as an `M_k × p` matrix and
    /// shrinks its singular spectrum (embedding rank).
    ///
    /// Unlike MechanismSparsity, `NuclearNormPenalty::grad_target` / `hvp`
    /// reshape the *entire* `target` argument they are given (they do not use
    /// `self.target.range` to slice), so the local `M_k × p` block is passed
    /// directly and the returned vectors are local (length `M_k*p`). The PSD
    /// curvature is probed via `psd_majorizer_hvp`, which for NuclearNorm has
    /// no diagonal majorizer and delegates to its analytic spectral HVP.
    pub(crate) fn add_sae_nuclear_norm_atom(
        &self,
        sys: &mut ArrowSchurSystem,
        per_atom: &NuclearNormPenalty,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        start: usize,
        end: usize,
        penalty_scale: f64,
        dense_beta_curvature: bool,
    ) -> bool {
        let block = target_beta.slice(s![start..end]);
        let block_len = end - start;
        let grad = per_atom.grad_target(block, rho_local);
        for local in 0..block_len {
            sys.gb[start + local] += penalty_scale * grad[local];
        }
        if !dense_beta_curvature {
            return true;
        }
        let mut probe = Array1::<f64>::zeros(block_len);
        for local in 0..block_len {
            probe.fill(0.0);
            probe[local] = 1.0;
            let hv = per_atom.psd_majorizer_hvp(block, rho_local, probe.view());
            for i in 0..block_len {
                sys.hbb[[start + i, start + local]] += penalty_scale * hv[i];
            }
        }
        true
    }

    /// #1026/#1522 — PUBLIC test inspector: the SEPARATION barrier value and its
    /// analytic decoder gradient (length `beta_dim`, full-`B` layout) at the
    /// current decoders, scaled by `penalty_scale`. Hermetic seam so the owed-1026
    /// FD battery can certify `∂P_sep/∂B` in isolation, and so the #1522
    /// prevention-vs-bandaid pinning test can read the barrier ON (`scale = 1`)
    /// against barrier OFF (`scale = 0`, the local "no prevention" arm —
    /// `penalty_scale = 0` writes nothing — without changing the term's per-fit
    /// configuration). Returns `(value, grad)`.
    pub fn separation_barrier_value_and_grad_for_test(
        &self,
        penalty_scale: f64,
    ) -> (f64, Array1<f64>) {
        let mut sys = ArrowSchurSystem::new(0, 0, self.beta_dim());
        sys.gb = Array1::<f64>::zeros(self.beta_dim());
        sys.hbb = Array2::<f64>::zeros((0, 0));
        let mut atom_curv = vec![0.0_f64; self.k_atoms()];
        let mut sep_rank1 = Vec::new();
        self.add_sae_separation_barrier(
            &mut sys,
            penalty_scale,
            false,
            &mut atom_curv,
            &mut sep_rank1,
        );
        (self.separation_barrier_value(penalty_scale), sys.gb)
    }
}

impl SaeManifoldTerm {
    /// Returns whether Beta-tier analytic curvature was accumulated into the
    /// dense `sys.hbb` block or deferred for exact factored-space probing.
    pub(crate) fn add_sae_analytic_penalty_contributions(
        &self,
        sys: &mut ArrowSchurSystem,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
        row_layout: Option<&SaeRowLayout>,
        dense_beta_curvature: bool,
        factored_row_projection: Option<&FrameProjection>,
    ) -> Result<SaeBetaPenaltyAssembly, ArrowSchurError> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let beta = self.flatten_beta();
        let mut beta_assembly = SaeBetaPenaltyAssembly::default();
        for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            // The coordinate ARD prior is owned by the built-in `ArdAxisPrior`
            // path (the unconditional row-block gradient/curvature write above,
            // and `ard_value`/`loss.ard` for the energy). That path uses the
            // smooth von-Mises energy `V(t) = (α/κ²)(1−cos κt)` on periodic
            // (Circle) axes, whose value, gradient (`α/κ·sin κt`), and curvature
            // (`α·cos κt`) are mutually FD-consistent and continuous across the
            // branch cut. The registry `ARDPenalty` is the legacy Euclidean
            // Gaussian (`½λt²`, grad `λt`, curvature `λ`): adding it here would
            // (a) double-count the coordinate prior in both gradient and Newton
            // curvature, and (b) reintroduce the period-discontinuous `½λt²`
            // energy — its grad `λt` is continuous but its value jumps by
            // `½λ(t_after²−t_before²)` across the cut, so a near-zero Newton step
            // crossing the cut changes the line-search objective discontinuously
            // and Armijo rejects it. Skip it on every SAE path so the von-Mises
            // built-in is the single source of truth (matching the REML criterion,
            // which already scores only `loss.ard`).
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            match tier {
                PenaltyTier::Psi => {
                    if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) {
                        // NuclearNorm is a Psi-tier penalty but it targets each
                        // atom's decoder (β) matrix singular spectrum, not the
                        // coord "t" row block. Route it to the β tier so it
                        // shrinks each atom's embedding rank.
                        if self.add_sae_beta_penalty(
                            sys,
                            penalty,
                            beta.view(),
                            rho_local,
                            penalty_scale,
                            dense_beta_curvature,
                        ) {
                            beta_assembly.record_curvature(dense_beta_curvature);
                        }
                    } else {
                        // Every other Psi-tier penalty here is row-block
                        // supported with a coord-shape that matches each
                        // atom — `validate_analytic_penalty_registry`
                        // refused everything else upfront, so this branch
                        // is total and the K=1 vs K>=2 path is the same
                        // loop. Row-block coord penalties (ARD,
                        // BlockOrthogonality, Sparsity/TopK/JumpReLU,
                        // RowPrecisionPrior, ScadMcp, Isometry) target the
                        // "t" latent block (n_obs × d) and apply per atom
                        // — accumulate into the corresponding row offsets.
                        assert!(
                            sae_penalty_is_row_block_supported(penalty),
                            "validate_analytic_penalty_registry should have \
                             refused non-row-block Psi-tier penalty {:?} \
                             (registry layout name {name:?})",
                            penalty.name()
                        );
                        let offsets = self.assignment.coord_offsets();
                        for atom_idx in 0..self.k_atoms() {
                            let off = offsets[atom_idx];
                            let coord = &self.assignment.coords[atom_idx];
                            if let AnalyticPenaltyKind::Isometry(iso) = penalty {
                                let corrected_kind =
                                    self.corrected_isometry_penalty(iso, atom_idx, coord)?;
                                self.add_sae_coord_penalty(
                                    sys,
                                    atom_idx,
                                    off,
                                    coord,
                                    &corrected_kind,
                                    rho_local,
                                    row_layout,
                                    factored_row_projection,
                                );
                                // The isometry penalty value depends on the
                                // decoder B as well as the latent coords, through
                                // the pullback metric `g = JᵀWJ` with the model
                                // Jacobian `J = (∂Φ/∂t)·B`. `add_sae_coord_penalty`
                                // only routes `∂P/∂t` into `gt`; the matching
                                // `∂P/∂B` must be accumulated into `gb`, or the
                                // assembled gradient disagrees with the penalized
                                // objective on the β block (value path counts the
                                // isometry energy, which moves with B).
                                if let AnalyticPenaltyKind::Isometry(corrected) = &corrected_kind {
                                    self.add_sae_isometry_beta_penalty(
                                        sys,
                                        atom_idx,
                                        coord,
                                        corrected,
                                        rho_local,
                                        dense_beta_curvature,
                                    );
                                    // #1783 — record a DENSE β-tier curvature for the
                                    // isometry gauge ONLY on the dense path, never the
                                    // factored/deferred one. `add_sae_isometry_beta_
                                    // penalty` writes a real dense `H_ββ` block into
                                    // `sys.hbb` (see its `if !dense_beta_curvature {
                                    // return; }` guard) EXACTLY when
                                    // `dense_beta_curvature == true`; on the framed
                                    // path it writes only the per-row `H_tt` curvature
                                    // and the `H_tβ` cross-block (into
                                    // `sys.rows[i].htbeta`), and
                                    // `build_factored_beta_penalty_curvature`
                                    // deliberately ignores Isometry (ZERO factored
                                    // `hbb_c`).
                                    //
                                    // So the honest flag is `dense_written` iff a
                                    // dense `hbb` was actually written:
                                    //   * dense path (`dense_beta_curvature == true`):
                                    //     `record_curvature(true)` sets `dense_written`
                                    //     so the assembly applies `sys.hbb` (full-`B`
                                    //     `DensePenaltyOp`, or frame-projected `Φᵀ hbb
                                    //     Φ`). Dropping it here loses the #795 decoder-
                                    //     side isometry curvature.
                                    //   * factored large-`K` path
                                    //     (`dense_beta_curvature == false`, the
                                    //     reporter's d_atom=1 regime): record NOTHING.
                                    //     The original code called
                                    //     `record_curvature(false)`, which set
                                    //     `deferred_factored` and spuriously flipped
                                    //     `has_dense_beta_penalty` at the framed
                                    //     assembly site — SKIPPING
                                    //     `set_device_sae_pcg_data` so `device_sae_pcg
                                    //     == None` and every curved (`d_atom = 1`) fit
                                    //     silently declined the device-resident SAE
                                    //     PCG (GPU 0%). Not recording keeps the device
                                    //     data installed; the CPU dense reference and
                                    //     the device framed kernel already both carry
                                    //     the isometry `H_tt`/`H_tβ`, so the numerics
                                    //     are unchanged — only WHERE the reduced-Schur
                                    //     matvec runs changes.
                                    if dense_beta_curvature {
                                        beta_assembly.record_curvature(true);
                                    }
                                }
                            } else {
                                self.add_sae_coord_penalty(
                                    sys,
                                    atom_idx,
                                    off,
                                    coord,
                                    penalty,
                                    rho_local,
                                    row_layout,
                                    factored_row_projection,
                                );
                            }
                        }
                    }
                }
                PenaltyTier::Beta => {
                    // β-tier analytic penalties are global (B-only); minibatch-
                    // scaled so per-chunk sums reconstruct one global copy.
                    if self.add_sae_beta_penalty(
                        sys,
                        penalty,
                        beta.view(),
                        rho_local,
                        penalty_scale,
                        dense_beta_curvature,
                    ) {
                        beta_assembly.record_curvature(dense_beta_curvature);
                    }
                }
                PenaltyTier::Rho => {}
            }
        }
        Ok(beta_assembly)
    }

    pub(crate) fn corrected_isometry_penalty(
        &self,
        iso: &Arc<IsometryPenalty>,
        atom_idx: usize,
        coord: &LatentCoordValues,
    ) -> Result<AnalyticPenaltyKind, ArrowSchurError> {
        // Isometry requires per-step cache refresh from the atom's second jet
        // before value / grad_target / hvp are live. The registry-held
        // IsometryPenalty was constructed with p_out equal to the latent dim
        // from the JSON latent spec; clone it and correct p_out to the atom's
        // true decoder output dimension before refreshing caches.
        let atom = &self.atoms[atom_idx];
        let p = atom.decoder_coefficients.ncols();
        let mut corrected: IsometryPenalty = (**iso).clone();
        corrected.p_out = p;
        // Single-source-of-truth gauge metric: the isometry pullback weight is
        // taken from the SAME RowMetric the reconstruction likelihood whitens
        // through. There is no independent gauge-weight setter, so a
        // likelihood-metric ≠ gauge-metric state is unrepresentable. When the
        // term carries no RowMetric (Euclidean default) the gauge weight stays
        // Identity, matching the isotropic likelihood exactly. The metric's
        // p_out must agree with the atom's true decoder output dimension.
        if let Some(metric) = self.row_metric.as_ref() {
            // Only a metric that actually drives the gauge installs a non-identity
            // pullback weight: any non-Euclidean provenance (OutputFisher or the
            // #974 WhitenedStructured) pulls the isometry penalty back through its
            // per-row inner product. A Euclidean metric reduces the gauge to the
            // bare `J_nᵀ J_n` (Identity weight), so it is left untouched and the
            // gauge is bit-for-bit the historical isotropic pullback.
            if metric.drives_gauge() {
                if metric.p_out() == p {
                    corrected.weight = metric.to_weight_field();
                } else {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "corrected_isometry_penalty: RowMetric p_out {} disagrees with atom \
                             {} decoder output dim {p}; the gauge metric must match the likelihood \
                             metric",
                            metric.p_out(),
                            atom_idx
                        ),
                    });
                }
            }
        }
        let coords_mat = coord.as_matrix();
        let second_jet_installed =
            refresh_isometry_caches_from_atom(&corrected, atom, coords_mat.view())
                .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        if !second_jet_installed {
            match atom
                .basis_evaluator
                .as_ref()
                .and_then(|e| e.second_jet_dyn(coords_mat.view()))
            {
                Some(Ok(hess)) => {
                    let n_obs = coords_mat.nrows();
                    let d = atom.latent_dim;
                    let m = atom.basis_size();
                    if hess.dim() != (n_obs, m, d, d) {
                        return Err(ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "SAE Isometry atom '{}': second_jet_dyn returned shape {:?}, \
                                 expected ({n_obs}, {m}, {d}, {d})",
                                atom.name,
                                hess.dim()
                            ),
                        });
                    }
                    let b = &atom.decoder_coefficients;
                    let mut jac2 = Array2::<f64>::zeros((n_obs, p * d * d));
                    for n in 0..n_obs {
                        for i in 0..p {
                            for a in 0..d {
                                for c in 0..d {
                                    let mut acc = 0.0;
                                    for mm in 0..m {
                                        acc += hess[[n, mm, a, c]] * b[[mm, i]];
                                    }
                                    jac2[[n, (i * d + a) * d + c]] = acc;
                                }
                            }
                        }
                    }
                    corrected.set_jacobian_second_cache(Some(Arc::new(jac2)));
                }
                Some(Err(reason)) => {
                    return Err(ArrowSchurError::SchurFactorFailed { reason });
                }
                None => {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "IsometryPenalty requested for SAE atom '{}' (basis kind {:?}) but \
                             this evaluator does not expose an analytic second jet; use \
                             AffineCoordinateEvaluator, SphereChartEvaluator, \
                             PeriodicHarmonicEvaluator, or TorusHarmonicEvaluator for \
                             SAE-Isometry",
                            atom.name, atom.basis_kind
                        ),
                    });
                }
            }
        }
        Ok(AnalyticPenaltyKind::Isometry(Arc::new(corrected)))
    }

    pub(crate) fn add_sae_coord_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        atom_idx: usize,
        dense_off: usize,
        coord: &LatentCoordValues,
        penalty: &AnalyticPenaltyKind,
        rho_local: ArrayView1<'_, f64>,
        row_layout: Option<&SaeRowLayout>,
        factored_row_projection: Option<&FrameProjection>,
    ) {
        let n = coord.n_obs();
        let d = coord.latent_dim();
        // Origin-anchored magnitude shrinkage (SCAD/MCP) is restricted to the
        // Euclidean axes: a periodic chart axis has no origin, and folding a
        // raw-|t| energy there is period-discontinuous and breaks the joint
        // Newton solve (issue #795). Evaluate the axis-separable penalty on the
        // Euclidean-only compacted coordinate and scatter its gradient / PSD
        // curvature back to those axis slots — periodic axes get nothing. This
        // mirrors the value accounting in `analytic_penalty_value_total` exactly,
        // so the assembled gradient stays the gradient of the line-search
        // objective.
        if sae_coord_penalty_is_origin_anchored_magnitude(penalty) {
            if let Some((euclidean_axes, compacted)) =
                sae_coord_penalty_euclidean_restriction(coord)
            {
                let de = euclidean_axes.len();
                let grad = penalty.grad_target(compacted.view(), rho_local);
                let diag = penalty.psd_majorizer_diag(compacted.view(), rho_local);
                for row in 0..n {
                    if let Some(row_off) =
                        sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
                    {
                        for (j, &axis) in euclidean_axes.iter().enumerate() {
                            sys.rows[row].gt[row_off + axis] += grad[row * de + j];
                            if let Some(diag) = diag.as_ref() {
                                sys.rows[row].htt[[row_off + axis, row_off + axis]] +=
                                    diag[row * de + j];
                            }
                        }
                    }
                }
                return;
            }
        }
        let target = coord.as_flat().view();
        let grad = penalty.grad_target(target, rho_local);
        for row in 0..n {
            if let Some(row_off) = sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx) {
                for axis in 0..d {
                    sys.rows[row].gt[row_off + axis] += grad[row * d + axis];
                }
            }
        }
        if let AnalyticPenaltyKind::Isometry(corrected) = penalty {
            self.add_sae_isometry_metric_gn_blocks(
                sys,
                atom_idx,
                dense_off,
                coord,
                corrected,
                rho_local,
                row_layout,
                factored_row_projection,
            );
            return;
        }
        // `htt` is the PSD Newton / PIRLS curvature block: accumulate the PSD
        // majorizer (exact for convex penalties), not the indefinite exact
        // Hessian, for the same PSD-Newton reason used throughout the analytic
        // penalty assembly.
        if let Some(diag) = penalty.psd_majorizer_diag(target, rho_local) {
            for row in 0..n {
                if let Some(row_off) =
                    sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
                {
                    for axis in 0..d {
                        sys.rows[row].htt[[row_off + axis, row_off + axis]] += diag[row * d + axis];
                    }
                }
            }
            return;
        }
        let mut probe = Array1::<f64>::zeros(n * d);
        for axis in 0..d {
            probe.fill(0.0);
            for row in 0..n {
                probe[row * d + axis] = 1.0;
            }
            let hv = penalty.psd_majorizer_hvp(target, rho_local, probe.view());
            for row in 0..n {
                if let Some(row_off) =
                    sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
                {
                    for b in 0..d {
                        sys.rows[row].htt[[row_off + b, row_off + axis]] += hv[row * d + b];
                    }
                }
            }
        }
    }

    pub(crate) fn add_sae_isometry_metric_gn_blocks(
        &self,
        sys: &mut ArrowSchurSystem,
        atom_idx: usize,
        dense_off: usize,
        coord: &LatentCoordValues,
        corrected: &Arc<IsometryPenalty>,
        rho_local: ArrayView1<'_, f64>,
        row_layout: Option<&SaeRowLayout>,
        factored_row_projection: Option<&FrameProjection>,
    ) {
        let n_obs = coord.n_obs();
        let d = coord.latent_dim();
        let atom = &self.atoms[atom_idx];
        let p = atom.decoder_coefficients.ncols();
        let m = atom.basis_size();
        let Some(jac) = corrected.jacobian_cache() else {
            return;
        };
        if jac.dim() != (n_obs, p * d) {
            return;
        }
        let Some(jac2) = corrected.jacobian_second_cache() else {
            return;
        };
        if jac2.dim() != (n_obs, p * d * d) {
            return;
        }
        let beta_off = self.beta_offsets()[atom_idx];
        let beta_block = m * p;
        let jet = &atom.basis_jacobian;
        // Resolve the learnable isometry strength `scalar_weight · exp(rho)` in
        // log-space with a clamped exponent: the naive `scalar_weight *
        // rho.exp()` overflows to `inf` for `rho ≳ 709`, and the downstream
        // `inf · jacobian` / `inf · 0.0` then injects NaN into the GN curvature
        // block and β-penalty, poisoning the joint solve (#742, Issue 4).
        //
        // #795 scale-invariant curvature: the value / gradient paths penalize
        // the SCALE-INVARIANT residual `R_n = g_n/gbar − g^ref_n` (normalized by
        // the shared `gbar = mean tr(g)/d`), so the assembled gradient is free
        // of the decoder magnitude. This Gauss-Newton block, however, is built
        // below from the RAW weighted Jacobian `wj ∝ ‖B‖`, so `AᵀA ∝ ‖B‖⁴` — it
        // is the GN block of the UN-normalized `½μ‖g_n − g^ref‖²`. Pairing a
        // scale-free gradient with a `‖B‖⁴` curvature collapses the joint Newton
        // step (`H⁻¹g ∝ ‖B‖⁻⁴`) as the decoder grows, the proximal ridge
        // saturates at 1e15, and every trial step is rejected — the exact #795
        // failure. The Gauss-Newton block of the NORMALIZED residual is the raw
        // block times `1/gbar²` (freezing the shared normalizer, the same
        // convention `normalized_metric_state`'s majorizer uses): a positive
        // scalar on an already-PSD Gram block, so the Schur complement stays PSD,
        // and `1/gbar² ∝ ‖B‖⁻⁴` exactly cancels the raw `‖B‖⁴`. Fold it into `mu`
        // so every `mu * acc` write below carries it. `gbar` is read from the
        // penalty (the single source of truth shared with the gradient); if it
        // is unavailable/degenerate we skip the block rather than write a
        // mis-scaled one.
        let mu_raw =
            resolve_learnable_weight(corrected.scalar_weight, rho_local[corrected.rho_index]);
        let Some(gbar) = corrected.metric_normalizer(d) else {
            return;
        };
        let mu = mu_raw / (gbar * gbar);
        // A negligible (or non-finite) effective isometry weight contributes a
        // zero curvature block; writing zeros would still flip the solver onto
        // the dense-supplement Schur path (and invalidate caches) for no model
        // change. Skip entirely so `isometry_weight≈0` is bit-identical to the
        // no-isometry assembly. (`isometry_weight=0` never constructs the
        // penalty at all; this guards the ρ-sweep driving `exp(ρ)` to ~0.)
        if !(mu.is_finite() && mu > 0.0) {
            return;
        }
        // Coherence invariant for the coupled Gauss-Newton block. The isometry
        // residual `r_{ab} = (JᵀWJ − G_ref)_{ab}` yields one residual Jacobian
        // `A = [A_t | A_β]`, so `[[htt,cross],[crossᵀ,hbb]] = μ AᵀA` is PSD *as a
        // whole* and its Schur complement is PSD — but ONLY while all three
        // blocks stay that exact pullback. After assembly the latent blocks pass
        // through `apply_riemannian_latent_geometry`, which on a *curved* chart
        // rewrites `htt` with the (indefinite) Riemannian connection term and
        // column-projects the `htbeta` cross-block to `T_tM`, while the shared
        // `hbb` is left untouched. That projection breaks the `μ AᵀA` coherence:
        // the cross-block is then a nonzero coupling NOT paired with diagonals
        // from the same Jacobian, and the Schur complement
        // `hbb − Σ crossᵀ htt⁻¹ cross` can go indefinite (the #681 sphere
        // failure mode flagged in the math review).
        //
        // The decision must therefore key on whether the geometry transform is
        // the IDENTITY for this chart, NOT on `is_euclidean()`. A flat periodic
        // chart (`Circle`/`Torus`) is non-Euclidean yet transforms as the exact
        // identity — its tangent projection is the identity, it carries no
        // connection term, and it adds no normal pinning — so the coupled block
        // survives exactly and the full cross-coupling must be kept. Keying on
        // `is_euclidean()` instead wrongly dropped the cross-block for the
        // single-circle fit, leaving a block-diagonal Hessian that misses the
        // strong isometry `t`↔`B` coupling; the joint Newton step then never
        // reaches the KKT stationarity the REML criterion now requires, and the
        // arrow-Schur proximal ridge saturates at 1e15 (issue #795, a regression
        // of #681). For a genuinely curved chart (Sphere, an active Interval
        // boundary) we contribute only the PSD `htt` diagonal block and DROP the
        // cross-block: a block-diagonal `diag(μ A_tᵀA_t, μ A_βᵀA_β)` of two PSD
        // blocks is still PSD, so the Schur stays PSD by construction while the
        // gradient (which alone fixes the stationary point) is unchanged.
        let couple_cross_block = coord.manifold().preserves_isometry_cross_block_coherence();
        let mut metric_coord_jac = Array2::<f64>::zeros((d * d, d));
        let mut metric_beta_jac = Array2::<f64>::zeros((d * d, beta_block));
        let mut wrote_dense_cross = false;
        for row in 0..n_obs {
            let Some(row_off) = sae_coord_penalty_offset(row_layout, dense_off, row, atom_idx)
            else {
                continue;
            };
            let Some(wj) = Self::sae_isometry_weighted_jacobian_row(corrected, &jac, row, p, d)
            else {
                return;
            };
            metric_coord_jac.fill(0.0);
            for a in 0..d {
                for b in 0..d {
                    let metric_row = a * d + b;
                    for c in 0..d {
                        let mut acc = 0.0;
                        for i in 0..p {
                            acc += jac2[[row, (i * d + a) * d + c]] * wj[[i, b]];
                            acc += wj[[i, a]] * jac2[[row, (i * d + b) * d + c]];
                        }
                        metric_coord_jac[[metric_row, c]] = acc;
                    }
                }
            }
            if couple_cross_block {
                metric_beta_jac.fill(0.0);
                for a in 0..d {
                    for b in 0..d {
                        let metric_row = a * d + b;
                        for basis_col in 0..m {
                            let jet_a = jet[[row, basis_col, a]];
                            let jet_b = jet[[row, basis_col, b]];
                            for output in 0..p {
                                metric_beta_jac[[metric_row, basis_col * p + output]] =
                                    jet_a * wj[[output, b]] + wj[[output, a]] * jet_b;
                            }
                        }
                    }
                }
            }
            for c in 0..d {
                for e in 0..d {
                    let mut acc = 0.0;
                    for metric_row in 0..(d * d) {
                        acc +=
                            metric_coord_jac[[metric_row, c]] * metric_coord_jac[[metric_row, e]];
                    }
                    sys.rows[row].htt[[row_off + c, row_off + e]] += mu * acc;
                }
                if !couple_cross_block {
                    continue;
                }
                for beta_col in 0..beta_block {
                    let mut acc = 0.0;
                    for metric_row in 0..(d * d) {
                        acc += metric_coord_jac[[metric_row, c]]
                            * metric_beta_jac[[metric_row, beta_col]];
                    }
                    if let Some(projection) = factored_row_projection {
                        let basis_col = beta_col / p;
                        let output = beta_col % p;
                        let c_base = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx];
                        let mut hrow = sys.rows[row].htbeta.row_mut(row_off + c);
                        let hrow_slice = hrow.as_slice_mut().expect("htbeta row is contiguous");
                        projection.accumulate_output_project(
                            atom_idx,
                            c_base,
                            output,
                            mu * acc,
                            hrow_slice,
                        );
                    } else {
                        sys.rows[row].htbeta[[row_off + c, beta_off + beta_col]] += mu * acc;
                    }
                    wrote_dense_cross = true;
                }
            }
        }
        if wrote_dense_cross {
            sys.activate_dense_htbeta_supplement();
        }
    }

    pub(crate) fn sae_isometry_weighted_jacobian_row(
        corrected: &IsometryPenalty,
        jac: &Array2<f64>,
        row: usize,
        p: usize,
        d: usize,
    ) -> Option<Array2<f64>> {
        match &corrected.weight {
            WeightField::Identity => {
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        out[[i, a]] = jac[[row, i * d + a]];
                    }
                }
                Some(out)
            }
            WeightField::Factored { u, rank, p_out } => {
                if *p_out != p || u.nrows() != jac.nrows() || u.ncols() != p * *rank {
                    return None;
                }
                let mut projected = Array2::<f64>::zeros((*rank, d));
                for weight_axis in 0..*rank {
                    for a in 0..d {
                        let mut acc = 0.0;
                        for i in 0..p {
                            acc += u[[row, i * *rank + weight_axis]] * jac[[row, i * d + a]];
                        }
                        projected[[weight_axis, a]] = acc;
                    }
                }
                let mut out = Array2::<f64>::zeros((p, d));
                for i in 0..p {
                    for a in 0..d {
                        let mut acc = 0.0;
                        for weight_axis in 0..*rank {
                            acc += u[[row, i * *rank + weight_axis]] * projected[[weight_axis, a]];
                        }
                        out[[i, a]] = acc;
                    }
                }
                Some(out)
            }
        }
    }

    /// Accumulate the isometry penalty's decoder-block gradient `∂P/∂B` into the
    /// β-tier `gb` and its decoder-block Gauss-Newton majorizer into `hbb`. The
    /// isometry value
    ///   `P = ½ μ Σ_n ‖J_nᵀ W J_n − G_ref‖²_F`
    /// is a function of the model Jacobian `J_n[i, a] = Σ_m (∂Φ/∂t)[n, m, a]·B[m, i]`,
    /// so it depends on the decoder `B` as well as the latent coords `t`. The
    /// penalty exposes `∂P/∂J` (shape `(n_obs, p·d)`, layout `[n, i·d + a]`) via
    /// [`IsometryPenalty::grad_jacobian`]; the chain rule through
    /// `∂J[n, i·d + a]/∂B[m, i] = (∂Φ/∂t)[n, m, a]` gives
    ///   `∂P/∂B[m, i] = Σ_n Σ_a (∂P/∂J)[n, i·d + a] · (∂Φ/∂t)[n, m, a]`.
    /// Since `J` is linear in `B`, the PSD decoder curvature is the exact
    /// pullback of the J-space Gauss-Newton block:
    ///   `Σ_n jet[n,m,a] · B_GN^J[n,(i,a),(i',a')] · jet[n,m',a']`.
    /// This drops only the indefinite residual-curvature term, matching the
    /// file-wide PSD-majorizer convention for Newton / Arrow-Schur blocks.
    /// The flat β layout is `β[beta_offsets[k] + m·p + i] = B_k[m, i]`, so each
    /// atom's contribution lands in its own decoder span. The isometry penalty is
    /// unscaled at the row-block (Psi) tier — mirroring its coord-block routing
    /// and `analytic_penalty_value_total` — so no `penalty_scale` is applied here.
    pub(crate) fn add_sae_isometry_beta_penalty(
        &self,
        sys: &mut ArrowSchurSystem,
        atom_idx: usize,
        coord: &LatentCoordValues,
        corrected: &Arc<IsometryPenalty>,
        rho_local: ArrayView1<'_, f64>,
        dense_beta_curvature: bool,
    ) {
        let atom = &self.atoms[atom_idx];
        let d = coord.latent_dim();
        let p = atom.decoder_coefficients.ncols();
        let m = atom.basis_size();
        let n_obs = coord.n_obs();
        let grad_jac = corrected.grad_jacobian(coord.as_flat().view(), rho_local);
        if grad_jac.dim() != (n_obs, p * d) {
            return;
        }
        let jet = &atom.basis_jacobian;
        let beta_off = self.beta_offsets()[atom_idx];
        for basis_col in 0..m {
            for i in 0..p {
                let mut acc = 0.0;
                for n in 0..n_obs {
                    for a in 0..d {
                        acc += grad_jac[[n, i * d + a]] * jet[[n, basis_col, a]];
                    }
                }
                sys.gb[beta_off + basis_col * p + i] += acc;
            }
        }
        if !dense_beta_curvature {
            return;
        }
        let Some(jac) = corrected.jacobian_cache() else {
            return;
        };
        if jac.dim() != (n_obs, p * d) {
            return;
        }
        let mut weighted_jacobian_rows = Vec::with_capacity(n_obs);
        for n in 0..n_obs {
            let Some(wj) = Self::sae_isometry_weighted_jacobian_row(corrected, &jac, n, p, d)
            else {
                return;
            };
            weighted_jacobian_rows.push(wj);
        }
        // Resolve the learnable isometry strength `scalar_weight · exp(rho)` in
        // log-space with a clamped exponent: the naive `scalar_weight *
        // rho.exp()` overflows to `inf` for `rho ≳ 709`, and the downstream
        // `inf · jacobian` / `inf · 0.0` then injects NaN into the GN curvature
        // block and β-penalty, poisoning the joint solve (#742, Issue 4).
        //
        // #795 scale-invariant curvature (decoder β block): the `gb` gradient
        // accumulated above routes through the gbar-normalized
        // `grad_jacobian`, so it is free of the decoder magnitude. This dense
        // `hbb` Gauss-Newton block is the decoder-side pullback of the SAME raw
        // `wj`-built block as the coord-side `htt` (`add_sae_isometry_metric_gn_blocks`),
        // so it scales ∝‖B‖⁴ and would re-introduce the #795 step collapse on
        // the β tier. Fold the same `1/gbar²` frozen-normalizer factor in here
        // so the decoder curvature matches its scale-free gradient. PSD-
        // preserving (positive scalar on a PSD Gram block); skip on a degenerate
        // normalizer rather than write a mis-scaled block.
        let mu_raw =
            resolve_learnable_weight(corrected.scalar_weight, rho_local[corrected.rho_index]);
        let Some(gbar) = corrected.metric_normalizer(d) else {
            return;
        };
        let mu = mu_raw / (gbar * gbar);
        if !(mu.is_finite() && mu > 0.0) {
            return;
        }
        let mut metric_jvp = Array2::<f64>::zeros((d, d));
        let mut jac_hvp = Array2::<f64>::zeros((p, d));
        let mut beta_hvp = Array2::<f64>::zeros((m, p));
        for probe_basis_col in 0..m {
            for probe_output in 0..p {
                beta_hvp.fill(0.0);
                for n in 0..n_obs {
                    let wj = &weighted_jacobian_rows[n];
                    metric_jvp.fill(0.0);
                    for a in 0..d {
                        let probe_jet_a = jet[[n, probe_basis_col, a]];
                        for b in 0..d {
                            metric_jvp[[a, b]] = probe_jet_a * wj[[probe_output, b]]
                                + wj[[probe_output, a]] * jet[[n, probe_basis_col, b]];
                        }
                    }
                    jac_hvp.fill(0.0);
                    for i in 0..p {
                        for c in 0..d {
                            let mut acc = 0.0;
                            for b in 0..d {
                                acc += metric_jvp[[c, b]] * wj[[i, b]];
                            }
                            for a in 0..d {
                                acc += metric_jvp[[a, c]] * wj[[i, a]];
                            }
                            jac_hvp[[i, c]] = mu * acc;
                        }
                    }
                    for basis_row in 0..m {
                        for i in 0..p {
                            let mut acc = 0.0;
                            for a in 0..d {
                                acc += jac_hvp[[i, a]] * jet[[n, basis_row, a]];
                            }
                            beta_hvp[[basis_row, i]] += acc;
                        }
                    }
                }
                let beta_col = beta_off + probe_basis_col * p + probe_output;
                for basis_row in 0..m {
                    for i in 0..p {
                        sys.hbb[[beta_off + basis_row * p + i, beta_col]] +=
                            beta_hvp[[basis_row, i]];
                    }
                }
            }
        }
    }
}

pub(crate) fn sae_coord_penalty_offset(
    row_layout: Option<&SaeRowLayout>,
    dense_off: usize,
    row: usize,
    atom_idx: usize,
) -> Option<usize> {
    match row_layout {
        Some(layout) => {
            let active = &layout.active_atoms[row];
            let starts = &layout.coord_starts[row];
            active
                .iter()
                .zip(starts.iter())
                .find_map(|(&active_atom, &coord_start)| {
                    if active_atom == atom_idx {
                        Some(coord_start)
                    } else {
                        None
                    }
                })
        }
        None => Some(dense_off),
    }
}

pub(crate) fn sae_penalty_is_row_block_supported(penalty: &AnalyticPenaltyKind) -> bool {
    matches!(
        penalty,
        AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::JumpReLU(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::RowPrecisionPrior(_)
            | AnalyticPenaltyKind::ParametricRowPrecisionPrior(_)
            | AnalyticPenaltyKind::ScadMcp(_)
            | AnalyticPenaltyKind::BlockOrthogonality(_)
            | AnalyticPenaltyKind::Isometry(_)
    )
}

/// Whether a row-block coordinate penalty **composes over a
/// coordinate-heterogeneous dictionary** — i.e. its energy is a sum of
/// independent per-atom contributions, each evaluated on that atom's own
/// `(n_obs × d_k)` latent block with the atom's *own* latent dim `d_k`, so a
/// single registry entry dispatches cleanly across atoms whose coordinate dims
/// differ (issue F6). For these the assembled value / gradient / curvature is
/// *exactly* the sum of the per-atom energies — the same additive
/// decomposition the Laplace/REML evidence log-det already sums per atom — so
/// admitting them on a mixed `{d=1 circle, d=2 patch, linear}` dictionary keeps
/// the evidence exact with zero padding or truncation.
///
/// The composing penalties are exactly those with **no fixed cross-axis
/// structure tied to one shared `d`**:
/// * [`AnalyticPenaltyKind::ScadMcp`] / [`AnalyticPenaltyKind::Sparsity`] —
///   coordinate-separable, evaluated element-wise over the flat block; length-
///   and dim-agnostic (they never read a stored `latent_dim` at eval time).
/// * [`AnalyticPenaltyKind::Ard`] — the native von-Mises/Gaussian coordinate
///   ARD is summed per atom over `d_k` axes with a per-atom `log_ard[k]` of
///   length `d_k` (see `ard_value`), so heterogeneous dims are the native shape,
///   not a special case.
/// * [`AnalyticPenaltyKind::Isometry`] — rebuilt per atom by
///   `corrected_isometry_penalty`, which corrects `p_out` to the atom's true
///   decoder output dim and refreshes its caches from the atom's own second jet.
///
/// The **non-composing** row-block penalties carry a fixed per-axis structure
/// bound to one shared `d` and would silently misinterpret an atom of a
/// different dim (reshape to the wrong `(n_eff × d)`, index an out-of-range
/// axis, or misalign a per-axis threshold / precision):
/// [`AnalyticPenaltyKind::BlockOrthogonality`] (reshapes to `(n_eff × d)` and
/// partitions a fixed axis set into groups), [`AnalyticPenaltyKind::JumpReLU`]
/// and [`AnalyticPenaltyKind::TopKActivation`] (per-axis thresholds / top-k
/// across a fixed `latent_dim`), and the row-precision priors
/// ([`AnalyticPenaltyKind::RowPrecisionPrior`],
/// [`AnalyticPenaltyKind::ParametricRowPrecisionPrior`], which hold a
/// `(n_eff × d × d)` precision stack). These still require a uniform `atom_dim`
/// (or an explicit per-dim-group configuration, which is a separate feature).
pub(crate) fn sae_row_block_penalty_composes_over_heterogeneous_coord_dims(
    penalty: &AnalyticPenaltyKind,
) -> bool {
    matches!(
        penalty,
        AnalyticPenaltyKind::ScadMcp(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::Isometry(_)
    )
}

/// Whether a row-block coordinate penalty is an *origin-anchored, axis-separable
/// magnitude shrinkage* — its energy is `Σ_axis Σ_row f(|t|)` with a fixed zero,
/// evaluated independently per flat entry.
///
/// Such a penalty is only well-posed on a **Euclidean** chart axis, which has a
/// distinguished origin. On a **periodic** chart axis (a `Circle`/`Torus`
/// coordinate) the latent is a homogeneous angle defined only modulo its period:
/// there is no rotation-invariant "zero" to shrink toward, and the raw `|t|` is
/// *discontinuous across the retraction branch cut* (a coordinate just below the
/// period wraps to just above zero). Folding such an energy into the joint
/// Newton objective makes the line-search value jump by an `O(1)` amount under a
/// near-zero coordinate step, so Armijo rejects otherwise-valid steps and the
/// inner solve never reaches stationarity (issue #795; the same failure mode the
/// ARD prior avoids by switching to the periodic von-Mises energy on these axes).
///
/// For these penalties [`sae_coord_penalty_euclidean_restriction`] restricts the
/// energy to the Euclidean axes, where it is both meaningful and continuous;
/// periodic axes contribute nothing. The restriction is exact only because the
/// energy is axis-separable, so this matcher is deliberately narrow: non-separable
/// shrinkage (e.g. the Hoyer ℓ¹/ℓ² ratio) is excluded.
pub(crate) fn sae_coord_penalty_is_origin_anchored_magnitude(
    penalty: &AnalyticPenaltyKind,
) -> bool {
    matches!(penalty, AnalyticPenaltyKind::ScadMcp(_))
}

/// Restrict an origin-anchored, axis-separable coordinate shrinkage penalty (see
/// [`sae_coord_penalty_is_origin_anchored_magnitude`]) to the **Euclidean**
/// (non-periodic) axes of a latent coordinate block.
///
/// Returns `Some((euclidean_axes, compacted_target))` where `compacted_target` is
/// the row-major `(n_obs × euclidean_axes.len())` flat vector holding only the
/// Euclidean-axis coordinates, in the axis order given by `euclidean_axes`. The
/// caller evaluates the (axis-separable) penalty on this compacted target and
/// scatters its per-entry gradient / curvature back to the Euclidean axis slots,
/// leaving every periodic axis untouched (zero contribution). Because the penalty
/// is a sum of independent per-entry terms, evaluating it on the compacted target
/// is *exactly* the full energy with the periodic axes dropped — value, gradient,
/// and curvature stay mutually consistent.
///
/// Returns `None` when every axis is Euclidean: there is nothing to restrict, so
/// the caller uses the full target unchanged (zero overhead on the common path).
/// When every axis is periodic the returned `euclidean_axes` is empty and the
/// compacted target has length zero, so the penalty contributes nothing at all.
pub(crate) fn sae_coord_penalty_euclidean_restriction(
    coord: &LatentCoordValues,
) -> Option<(Vec<usize>, Array1<f64>)> {
    let periods = coord.effective_axis_periods();
    let d = periods.len();
    let euclidean_axes: Vec<usize> = (0..d).filter(|&axis| periods[axis].is_none()).collect();
    if euclidean_axes.len() == d {
        return None;
    }
    let n = coord.n_obs();
    let de = euclidean_axes.len();
    let flat = coord.as_flat();
    let mut compacted = Array1::<f64>::zeros(n * de);
    for row in 0..n {
        for (j, &axis) in euclidean_axes.iter().enumerate() {
            compacted[row * de + j] = flat[row * d + axis];
        }
    }
    Some((euclidean_axes, compacted))
}

/// The JSON descriptor `kind` strings for the SAE row-block analytic penalties
/// this build supports (i.e. those `sae_penalty_is_row_block_supported`
/// accepts). Co-located with that matcher so the two cannot drift. The FFI
/// `build_info` advertises this list so the Python wrapper can detect a stale
/// extension that predates a given penalty and raise a clear `NotImplementedError`
/// instead of forwarding a descriptor the binary will reject with a cryptic
/// Schur error (issue #338).
pub fn sae_row_block_penalty_kinds() -> &'static [&'static str] {
    &[
        "ard",
        "top_k_activation",
        "jumprelu",
        "sparsity",
        "row_precision_prior",
        "parametric_row_precision_prior",
        "scad_mcp",
        "block_orthogonality",
        "isometry",
    ]
}

#[cfg(test)]
mod tests_findings_234 {
    use super::*;
    use crate::manifold::tests::{
        TestPeriodicEvaluator, periodic_basis, small_two_atom_periodic_term,
    };
    use ndarray::array;

    /// Build a co-firing (softmax) two-atom term with EXPLICIT decoder blocks of
    /// shape `(3, p)`, so a test can plant an arbitrary decoder subspace geometry.
    fn two_atom_term_with_decoders(dec0: Array2<f64>, dec1: Array2<f64>) -> SaeManifoldTerm {
        let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
        let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let (phi1, jet1) = periodic_basis(&coords1);
        let atom0 = SaeManifoldAtom::new(
            "a0",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            dec0,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let atom1 = SaeManifoldAtom::new(
            "a1",
            SaeAtomBasisKind::Periodic,
            1,
            phi1,
            jet1,
            dec1,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        // softmax ⇒ every atom carries strictly positive mass on every row ⇒ the
        // pair co-fires (q_01 > 0) so the barrier / repulsion actually engage.
        let logits = array![
            [0.7, -0.2],
            [0.1, 0.4],
            [-0.3, 0.5],
            [0.6, -0.1],
            [0.2, 0.3]
        ];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords0, coords1],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    }

    /// The OLD Frobenius-NORM-normalized collinearity score
    /// `‖B_jB_kᵀ‖²_F / (‖B_j‖²_F·‖B_k‖²_F)` — the metric the #2 fix replaces.
    fn old_frobenius_cosine_sq(b0: &Array2<f64>, b1: &Array2<f64>) -> f64 {
        let (m0, p) = (b0.nrows(), b0.ncols());
        let m1 = b1.nrows();
        let mut cross = 0.0_f64;
        for a in 0..m0 {
            for b in 0..m1 {
                let mut c = 0.0_f64;
                for o in 0..p {
                    c += b0[[a, o]] * b1[[b, o]];
                }
                cross += c * c;
            }
        }
        let n0: f64 = b0.iter().map(|v| v * v).sum();
        let n1: f64 = b1.iter().map(|v| v * v).sum();
        cross / (n0 * n1)
    }

    /// #2 — the collinearity metric is now a TRUE rank-aware subspace overlap: two
    /// atoms sharing an IDENTICAL rank-2 output subspace score 1.0 (and trip BOTH
    /// the separation barrier and the decoder-repulsion gate), whereas the old
    /// Frobenius-norm score read exactly 1/2 = the gate threshold and gave them
    /// ZERO anti-collapse force. Orthogonal subspaces score 0 (strict no-op).
    #[test]
    fn true_subspace_overlap_detects_multirow_co_collapse_finding2() {
        // Identical rank-2 decoders spanning the same 2D output subspace.
        let ident = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let term = two_atom_term_with_decoders(ident.clone(), ident.clone());

        // Old Frobenius-norm score is exactly 1/2 for identical rank-2 blocks — at
        // the 0.5 gate, so the old gate returned 0 (the undetected-collapse hole).
        let old = old_frobenius_cosine_sq(&ident, &ident);
        assert!(
            (old - 0.5).abs() < 1e-12,
            "old Frobenius-norm score must read 1/2 for identical rank-2 subspaces (the hole), got {old}"
        );
        // New true overlap scores 1.0 → gate fully engaged.
        let o = term.decoder_gram_cosine_sq(0, 1);
        assert!(
            (o - 1.0).abs() < 1e-12,
            "true subspace overlap must be 1.0 for identical rank-2 subspaces, got {o}"
        );

        // Separation barrier ENGAGES (positive value + separating gradient) where
        // the old Frobenius gate (0.5 ≤ s0) would have been a strict no-op.
        let (value, grad) = term.separation_barrier_value_and_grad_for_test(1.0);
        assert!(
            value > 0.0,
            "identical rank-2 subspaces must carry a positive separation barrier, got {value}"
        );
        assert!(
            grad.iter().any(|&g| g != 0.0),
            "identical rank-2 subspaces must produce a separating gradient"
        );

        // Decoder-repulsion gate also engages on the pair.
        let mut t2 = two_atom_term_with_decoders(ident.clone(), ident.clone());
        t2.refresh_decoder_repulsion_gate();
        let gate = t2
            .decoder_repulsion_gate
            .as_ref()
            .expect("identical rank-2 subspaces must ENGAGE the repulsion gate");
        assert!(
            gate.iter().any(|&(j, k, w)| j == 0 && k == 1 && w > 0.0),
            "engaged repulsion gate must carry a positive weight on pair (0,1): {gate:?}"
        );

        // Orthogonal output subspaces: overlap 0, barrier + repulsion strict no-op.
        let b_ortho0 = array![[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let b_ortho1 = array![[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let ortho = two_atom_term_with_decoders(b_ortho0, b_ortho1);
        assert!(
            ortho.decoder_gram_cosine_sq(0, 1).abs() < 1e-12,
            "orthogonal output subspaces must have zero overlap"
        );
        let (vo, go) = ortho.separation_barrier_value_and_grad_for_test(1.0);
        assert_eq!(
            vo, 0.0,
            "orthogonal subspaces must carry zero barrier value"
        );
        assert!(
            go.iter().all(|&g| g == 0.0),
            "orthogonal subspaces must produce no separating force"
        );
    }

    /// #2 — analytic ∂P_sep/∂B on the new overlap `o` matches a finite difference of
    /// its OWN value on MULTI-ROW (rank-2) decoders, on the smoothstep interior. The
    /// existing #1625 FD test covers only single-row (rank-1) blocks — where the new
    /// barrier reduces bit-for-bit to the historical Frobenius one — so this test
    /// certifies the genuinely NEW multi-row gradient (the `S_·B_·` self-shrink and
    /// the `‖B_·B_·ᵀ‖_F` normalizers).
    #[test]
    fn separation_barrier_gradient_matches_fd_multirow_finding2() {
        // Two rank-2 decoders overlapping partially (interior of the gate ramp).
        let dec0 = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let dec1 = array![[0.9, 0.2], [0.1, 0.8], [0.0, 0.0]];
        let base = two_atom_term_with_decoders(dec0.clone(), dec1.clone());
        // Precondition: strictly on the smoothstep interior (0.5 < o < 1).
        let o = base.decoder_gram_cosine_sq(0, 1);
        assert!(
            o > 0.5 && o < 1.0,
            "fixture must sit on the interior (materially collinear, not collapsed), got o={o}"
        );
        let (_v, grad) = base.separation_barrier_value_and_grad_for_test(1.0);
        let offsets = base.beta_offsets();
        let p = base.output_dim();
        let h = 1.0e-7;
        let mut max_rel = 0.0_f64;
        // FD every coefficient of atom1's decoder block against the value.
        for a in 0..dec1.nrows() {
            for col in 0..p {
                let mut plus = two_atom_term_with_decoders(dec0.clone(), dec1.clone());
                let mut minus = two_atom_term_with_decoders(dec0.clone(), dec1.clone());
                plus.atoms[1].decoder_coefficients[[a, col]] += h;
                minus.atoms[1].decoder_coefficients[[a, col]] -= h;
                let vp = plus.separation_barrier_value(1.0);
                let vm = minus.separation_barrier_value(1.0);
                let fd = (vp - vm) / (2.0 * h);
                let analytic = grad[offsets[1] + a * p + col];
                let rel = (fd - analytic).abs() / (1.0 + fd.abs().max(analytic.abs()));
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel < 1.0e-5,
            "multi-row analytic ∂P_sep/∂B must match FD of the value: max rel err {max_rel:.3e}"
        );
    }

    /// #3 — the coactivation score is now a CONSISTENT truncated-support cosine: the
    /// denominator energies are summed over the SAME per-atom active support as the
    /// numerator, not over all rows. On a crafted softmax dense-tail routing (one
    /// row where atom 1 is sub-floor) the impl matches the truncated-support formula
    /// to machine precision and DIFFERS materially from the old truncated-numerator /
    /// full-denominator hybrid.
    #[test]
    fn coactivation_is_consistent_truncated_support_finding3() {
        // Row 0 drives atom 1 just BELOW the relative-mass floor (softmax tail);
        // the other rows keep atom 1 WEAK but active (a small energy E1). Because
        // the dropped tail mass is a MATERIAL fraction of that weak energy, the
        // truncated-support denominator differs visibly from the full-row one (the
        // gap is otherwise bounded by the floor² ≈ 1e-6 and hides on a strong atom).
        let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
        let coords1 = array![[0.15], [0.30], [0.65], [0.90]];
        let (phi0, jet0) = periodic_basis(&coords0);
        let (phi1, jet1) = periodic_basis(&coords1);
        let atom0 = SaeManifoldAtom::new(
            "a0",
            SaeAtomBasisKind::Periodic,
            1,
            phi0,
            jet0,
            array![[0.25], [-0.35], [0.15]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let atom1 = SaeManifoldAtom::new(
            "a1",
            SaeAtomBasisKind::Periodic,
            1,
            phi1,
            jet1,
            array![[-0.10], [0.20], [0.30]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let logits = array![[5.6, 0.0], [5.0, 0.04], [5.0, 0.05], [5.0, 0.03]];
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords0, coords1],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

        // Reference formulas evaluated directly on the realized gate matrix.
        let gates = term.assignment.assignments();
        let n = gates.nrows();
        let mut num = 0.0_f64; // Σ_{i∈J∩K} a0 a1
        let mut e0_trunc = 0.0_f64; // Σ_{i∈J} a0²
        let mut e1_trunc = 0.0_f64; // Σ_{i∈K} a1²
        let mut e0_all = 0.0_f64; // Σ_all a0²
        let mut e1_all = 0.0_f64; // Σ_all a1²
        for i in 0..n {
            let a0 = gates[[i, 0]];
            let a1 = gates[[i, 1]];
            let peak = a0.abs().max(a1.abs());
            let floor = SAE_COACTIVE_RELATIVE_MASS_FLOOR * peak;
            let act0 = a0.abs() > floor;
            let act1 = a1.abs() > floor;
            if act0 {
                e0_trunc += a0 * a0;
            }
            if act1 {
                e1_trunc += a1 * a1;
            }
            if act0 && act1 {
                num += a0 * a1;
            }
            e0_all += a0 * a0;
            e1_all += a1 * a1;
        }
        let q_truncated = num / (e0_trunc * e1_trunc).sqrt();
        let q_hybrid = num / (e0_all * e1_all).sqrt();

        let pairs = term.barrier_coactive_pairs();
        let q_impl = pairs
            .iter()
            .find(|&&(j, k, _)| j == 0 && k == 1)
            .map(|&(_, _, q)| q)
            .expect("pair (0,1) must co-fire");

        assert!(
            (q_impl - q_truncated).abs() < 1e-12,
            "impl must equal the consistent truncated-support score: impl={q_impl} trunc={q_truncated}"
        );
        assert!(
            (q_truncated - q_hybrid).abs() > 1e-6,
            "the crafted dense-tail case must expose a material hybrid-vs-consistent gap: \
             trunc={q_truncated} hybrid={q_hybrid}"
        );
        // The old hybrid under-weighted the barrier (full-row denominator inflates
        // the divisor), so the consistent score is strictly larger here.
        assert!(
            q_truncated > q_hybrid,
            "consistent truncated denominator must not under-weight vs the old hybrid"
        );
    }

    /// #4 — the data-fit inseparability `γ_jk` is now read from the WEIGHTED
    /// β-Hessian blocks (per-row weights `w_i` threaded into `G_j, G_k, C`, matching
    /// the `√w_i` the assembly applies). With single-basis (M=1) atoms the canonical
    /// correlation has a closed form we check exactly; a global weight rescale leaves
    /// it invariant, and a non-uniform weighting moves it away from the unweighted
    /// value.
    #[test]
    fn inseparability_uses_weighted_hessian_finding4() {
        let (mut term, _target, _rho) = small_two_atom_periodic_term();
        let n = term.n_obs();
        // Collapse each atom's chart design to a single (M=1) known column so the
        // canonical correlation reduces to a scalar closed form.
        let phi0 = array![[0.5], [-0.3], [0.8], [0.2], [-0.6]];
        let phi1 = array![[0.4], [0.7], [-0.2], [0.9], [0.1]];
        term.atoms[0].basis_values = phi0.clone();
        term.atoms[1].basis_values = phi1.clone();
        let gates = term.assignment.assignments();

        // Closed-form weighted γ = |Σ w a0 a1 φ0 φ1| / sqrt(Σ w a0²φ0² · Σ w a1²φ1²).
        let gamma_closed = |w: &[f64]| -> f64 {
            let (mut c, mut g0, mut g1) = (0.0_f64, 0.0_f64, 0.0_f64);
            for i in 0..n {
                let a0 = gates[[i, 0]];
                let a1 = gates[[i, 1]];
                let (p0, p1) = (phi0[[i, 0]], phi1[[i, 0]]);
                c += w[i] * a0 * a1 * p0 * p1;
                g0 += w[i] * a0 * a0 * p0 * p0;
                g1 += w[i] * a1 * a1 * p1 * p1;
            }
            (c.abs() / (g0 * g1).sqrt()).clamp(0.0, 1.0)
        };

        // Unweighted baseline.
        term.row_loss_weights = None;
        let gamma_none = term.design_inseparability_with_gates(gates.view(), 0, 1);
        assert!(
            (gamma_none - gamma_closed(&vec![1.0; n])).abs() < 1e-9,
            "unweighted γ must match the closed form"
        );

        // Global rescale (all weights = 3.0) leaves γ invariant.
        term.row_loss_weights = Some(vec![3.0; n]);
        let gamma_uniform = term.design_inseparability_with_gates(gates.view(), 0, 1);
        assert!(
            (gamma_uniform - gamma_none).abs() < 1e-9,
            "a global weight rescale must not change γ: {gamma_uniform} vs {gamma_none}"
        );

        // Non-uniform weights: matches the WEIGHTED closed form and differs from the
        // unweighted value (proving the weights are actually threaded).
        let w = vec![2.5_f64, 0.2, 1.7, 0.4, 3.1];
        term.row_loss_weights = Some(w.clone());
        let gamma_w = term.design_inseparability_with_gates(gates.view(), 0, 1);
        assert!(
            (gamma_w - gamma_closed(&w)).abs() < 1e-9,
            "weighted γ must match the weighted closed form: impl={gamma_w} closed={}",
            gamma_closed(&w)
        );
        assert!(
            (gamma_w - gamma_none).abs() > 1e-4,
            "non-uniform weights must move γ off the unweighted value: w={gamma_w} none={gamma_none}"
        );
    }
}
