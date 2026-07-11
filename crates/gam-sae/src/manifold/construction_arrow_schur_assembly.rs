//! Arrow-Schur bordered-Hessian assembly for `SaeManifoldTerm`, moved verbatim
//! out of construction.rs to keep it under the 10k-line ban gate. Pure code move,
//! no logic change.
//!
//! This is the curvature side of the term: [`SaeManifoldTerm::assemble_arrow_schur`]
//! and its scaled / streaming entry points materialize the enlarged `(logits, t)`
//! row-local Gauss-Newton bordered Hessian in the audit-revised layout, and the
//! factored β-penalty helpers (`project_dense_penalty_to_factored`,
//! `build_factored_beta_penalty_curvature`, `add_factored_repulsion_curvature`)
//! fold the analytic decoder penalties into that same arrow structure.
use super::*;

/// #2144 — PSD Loewner majorizer of the raw ordered Beta--Bernoulli assignment-prior diagonal
/// curvature `raw = w·(s'·J² + s·c)` at one logit slot, for the low-rank-metric
/// PD-repair path.
///
/// The exact ordered Beta--Bernoulli column-`k` Hessian block is
/// `H_p = w·s'·J Jᵀ + diag(w·s·c)`. Its rank-one coefficient is always
/// negative because `s' = -ψ₁(M+a)-ψ₁(N-M+1) < 0`; the concrete
/// second-Jacobian diagonal `w·s·c` can have either sign. Under a low-rank
/// whitening metric the data Gauss--Newton block need not dominate that
/// negative curvature, so an undamped exact-Hessian log determinant need not
/// exist. The rank-one block's zero matrix is already a PSD Loewner majorizer
/// because `w·s' < 0`; the row-local term is majorized by its positive part.
/// Thus the assembled block is simply `diag(max(w·s·c,0))`. The gradient
/// remains the exact derivative of the integrated scalar; only the
/// Newton/Laplace curvature uses this declared majorizer.
pub(super) fn ordered_beta_bernoulli_psd_majorized_hdiag(
    channels: &OrderedBetaBernoulliHessianDiagThirdChannels,
    row: usize,
    k_atoms: usize,
    atom: usize,
    raw_hdiag: f64,
) -> f64 {
    ordered_beta_bernoulli_psd_majorized_hdiag_derivative(
        channels,
        row,
        k_atoms,
        atom,
        raw_hdiag,
        channels.mass_hessian_coefficient[atom],
    )
}

/// Log-concentration derivative of the same ordered Beta--Bernoulli PSD
/// majorizer assembled by [`ordered_beta_bernoulli_psd_majorized_hdiag`].
pub(super) fn ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
    channels: &OrderedBetaBernoulliHessianDiagThirdChannels,
    row: usize,
    k_atoms: usize,
    atom: usize,
    raw_log_alpha_hdiag: f64,
) -> f64 {
    ordered_beta_bernoulli_psd_majorized_hdiag_derivative(
        channels,
        row,
        k_atoms,
        atom,
        raw_log_alpha_hdiag,
        channels.mass_hessian_log_alpha_derivative[atom],
    )
}

fn ordered_beta_bernoulli_psd_majorized_hdiag_derivative(
    channels: &OrderedBetaBernoulliHessianDiagThirdChannels,
    row: usize,
    k_atoms: usize,
    atom: usize,
    raw_hdiag_derivative: f64,
    mass_hessian_derivative: f64,
) -> f64 {
    let index = row * k_atoms + atom;
    if channels.diagonal_term[index] <= 0.0 {
        return 0.0;
    }
    let j = channels.z_jac[index];
    raw_hdiag_derivative - mass_hessian_derivative * j * j
}

impl SaeManifoldTerm {
    /// Build the per-row dense gate maps `a_{n,·}` at `rho` for all `n` rows.
    ///
    /// `try_assignments_row` is a pure read-only per-row computation
    /// (softmax / ordered Beta--Bernoulli sigmoid / TopK over that row's routing logits — no shared
    /// mutable state, no faer GEMM), so the rows are independent. Above the
    /// `SAE_LOSS_PARALLEL_ROW_MIN` floor (and when not already inside a rayon
    /// worker) they are computed in parallel and collected in row order; the
    /// order-preserving `collect` reproduces the serial push order bit-for-bit
    /// (deterministic — each row computed exactly once, no cross-row reduction).
    pub(crate) fn assignments_all_parallel(&self, n: usize) -> Result<Vec<Array1<f64>>, String> {
        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|row| self.assignment.try_assignments_row(row))
                .collect::<Result<Vec<_>, String>>()
        } else {
            let mut assignments_all = Vec::with_capacity(n);
            for row in 0..n {
                assignments_all.push(self.assignment.try_assignments_row(row)?);
            }
            Ok(assignments_all)
        }
    }

    /// Assemble the enlarged `(logits, t)` row-local Arrow-Schur system.
    ///
    /// Full-batch entry point: a single chunk covering all rows, with the
    /// β-tier penalties (decoder smoothness, ARD, analytic β penalties) carrying
    /// their full strength. The streaming driver calls
    /// [`Self::assemble_arrow_schur_scaled`] directly with a `penalty_scale`
    /// equal to the minibatch fraction `n_chunk / N`, so that the sum of the
    /// per-chunk β-tier contributions over a full pass reconstructs exactly the
    /// single global β penalty (the smoothness/ARD/β terms are functions of `B`
    /// and the global coordinates, not of the chunk's rows).
    pub fn assemble_arrow_schur(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled(target, rho, analytic_penalties, 1.0)
    }

    /// Assemble the row-local Arrow-Schur system with a `penalty_scale` applied
    /// to the β-tier (decoder smoothness, ARD prior, analytic β penalties).
    ///
    /// `penalty_scale == 1.0` recovers the full-batch assembly. The streaming
    /// driver passes the minibatch fraction `n_chunk / N` so that the β-tier
    /// reduced-Schur and gradient contributions of the chunks sum to exactly one
    /// global copy across a full pass (data-fit, assignment-prior, and per-row
    /// coord/logit analytic terms are *not* scaled — they are genuine per-row
    /// sums).
    pub fn assemble_arrow_schur_scaled(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target,
            rho,
            analytic_penalties,
            penalty_scale,
            SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM,
        )
    }

    pub(crate) fn assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
        dense_beta_penalty_probe_max_dim: usize,
    ) -> Result<ArrowSchurSystem, String> {
        self.assemble_arrow_schur_inner(
            target,
            rho,
            analytic_penalties,
            penalty_scale,
            dense_beta_penalty_probe_max_dim,
            None,
        )
    }

    /// Innermost assembly entry. `forced_layout` overrides the TopK-derived
    /// active-set layout so a test can pin the dense (`Forced(None)`) or a
    /// specific compact (`Forced(Some(layout))`) path — used by the
    /// compact-vs-dense Riemannian-geometry equality regression test to drive
    /// both layouts on identical data. The production layout is exact TopK
    /// support; smooth modes never use a compact layout.
    pub(crate) fn assemble_arrow_schur_inner(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        penalty_scale: f64,
        dense_beta_penalty_probe_max_dim: usize,
        forced_layout: ForcedRowLayout,
    ) -> Result<ArrowSchurSystem, String> {
        if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur_scaled: penalty_scale must be finite and positive; got {penalty_scale}"
            ));
        }
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: Z must be ({}, {}); got {:?}",
                self.n_obs(),
                self.output_dim(),
                target.dim()
            ));
        }
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: log_ard length {} != K {}",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        // `lambda_smooth` is indexed per-atom in the smoothness gradient/curvature
        // assembly (`lambda_smooth[atom_idx]`); a too-short vector (e.g. a growth
        // move that grew `k_atoms()` without extending ρ — #1556) would panic deep
        // in the assembly loop with an opaque index-out-of-bounds. Validate it here
        // alongside `log_ard` so the contract violation surfaces as a clear Err.
        if rho.log_lambda_smooth.len() != self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::assemble_arrow_schur: log_lambda_smooth length {} != K {}",
                rho.log_lambda_smooth.len(),
                self.k_atoms()
            ));
        }
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let ard_len = rho.log_ard[atom_idx].len();
            let d = coord.latent_dim();
            if ard_len != 0 && ard_len != d {
                return Err(format!(
                    "SaeManifoldTerm::assemble_arrow_schur: log_ard atom {atom_idx} \
                     has len {ard_len}; expected 0 (disabled) or atom dim {d}"
                ));
            }
        }
        // `smooth_penalty` is the validated reference-function Gram and is fixed
        // for the lifetime of this objective. Basis reparameterizations
        // transport it by congruence; ordinary Newton and penalized-LAML steps
        // all differentiate the same declared quadratic form.
        // #1026 — freeze the decoder-repulsion collinearity gate at the SAME
        // assembly chokepoint as the smoothness Gram, so the repulsion's
        // gradient/curvature (assembled below) and its value (read by the
        // line-search `penalized_objective_total`) share one frozen gate.
        //
        // #1801 — EXCEPT under a streaming fit, which freezes both collapse-
        // prevention gates ONCE per outer iteration from the FULL resident routing
        // and carries them onto every chunk (`streaming_gates_frozen`). The gates'
        // per-pair strength `μ_jk` inverts the coactivation-weighted design Grams
        // `G_j`, which are near-singular on a single small chunk, so a per-chunk
        // refresh makes `μ_jk = γ/(1−γ)` blow up as `γ→1` and the reduced β-Newton
        // step depend on `chunk_size`. Skipping the per-chunk refresh here keeps the
        // carried global gate, so the streaming fit is chunk-size invariant (pinned
        // by `sae_streaming_arrow_schur_contract::streaming_full_fit_is_chunk_size_invariant`).
        if !self.streaming_gates_frozen {
            self.refresh_decoder_repulsion_gate();
            // #1625 — freeze the SEPARATION barrier's normalized-coactivation `q_jk`
            // at the same chokepoint. The barrier weights its decoder-shape repulsion
            // by the routing coactivation, but its gradient treats that weight as a
            // constant; recomputing it from the trial logits in the line-search value
            // desyncs value vs gradient in the logit block and stalls the inner solve
            // (#1625). Freezing it here makes value/gradient/curvature consistent.
            self.refresh_barrier_coactivation_gate();
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let k_atoms = self.k_atoms();
        let assignment_dim = self.assignment.assignment_coord_dim();
        let q = self.assignment.row_block_dim();
        let beta_dim = self.beta_dim();
        let frame_projection = FrameProjection::new(self);
        let beta_offsets = frame_projection.beta_offsets.clone();
        let coord_offsets = self.assignment.coord_offsets();
        // β-tier decoder smoothness is a global (B-only) penalty; under a
        // minibatch pass it is scaled by the chunk fraction so the per-chunk
        // contributions sum to one global copy.
        // Per-atom decoder-smoothness strengths (#1556): atom k's penalty `S_k`
        // is scaled by `λ_smooth[k]·penalty_scale`. The minibatch `penalty_scale`
        // multiplies every atom uniformly.
        let lambda_smooth: Vec<f64> = rho
            .lambda_smooth_vec()
            .iter()
            .map(|&l| l * penalty_scale)
            .collect();
        // #991 — each differentiable assignment prior's per-(row, atom) gradient
        // (and its declared curvature majorizer) is design-weighted by
        // `w_i` here so `gt`/`htt` estimate the same target as the `√w`-weighted
        // data likelihood. The softmax curvature written to `htt` below is the
        // per-row Gershgorin/`row_psd_majorizer` block, weighted by folding
        // `w_row` into its `scale` at each site (no double application on the
        // gradient, which is already weighted here).
        let (assignment_grad, assignment_hdiag) =
            crate::assignment::assignment_prior_grad_hdiag_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?;

        // #1038 softmax entropy: the exact per-row Hessian in logits is dense
        // (`H_kj = (λ/τ²) a_k[δ_kj(m−L_k−1)+a_j(L_k+L_j+1−2m)]`), not just the
        // `assignment_hdiag` diagonal. Build the shared penalty + `scale = λ/τ²`
        // once here so the dense row block written into `block.htt` below, the
        // criterion's `log|H|`, and the #1006 θ-adjoint all differentiate the
        // SAME operator. Threshold-gate and ordered Beta--Bernoulli modes keep
        // their separate diagonal majorizers and leave this `None`. The block is gauge-null in
        // isolation (`H·𝟙 = 0`); it is only ever summed onto the gauge-breaking
        // data-fit row block before the Cholesky factor, never factored alone.
        let softmax_dense: Option<(
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty,
            f64,
        )> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                let scale = rho.lambda_sparse() * sparsity * inv_tau * inv_tau;
                Some((
                    gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                        k_atoms,
                        temperature,
                    ),
                    scale,
                ))
            }
            _ => None,
        };

        // Decoder smoothness penalty: build one KroneckerPenaltyOp per atom
        // (structure = λ·S_k ⊗ I_p, offset = beta_offsets[k]) instead of
        // materialising the dense K×K block.  The gradient is a dense K-vector
        // accumulated into `smooth_grad_gb` and written into sys.gb after sys
        // is constructed (#296).
        let mut smooth_ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len());
        // #972 / #977 T1: retain each atom's symmetrised `λ S_k` (`M_k × M_k`) so
        // the frame transform can rebuild the smooth penalty in the factored
        // coordinate space as `λ S_k ⊗ I_{r_k}` (the `tr(C_kᵀ S_k C_k)` form,
        // using `U_kᵀU_k = I`). Unused — and not even read — on the full-`B`
        // path, so this is a zero-cost capture there.
        let mut smooth_scaled_s: Vec<Array2<f64>> = Vec::with_capacity(self.atoms.len());
        let mut smooth_grad_gb = vec![0.0_f64; beta_dim];
        // #1117 — rank deficiency is handled at the basis layer: any
        // rank-deficient atom was reparametrized onto its data-supported subspace
        // at fit entry (`reduce_atoms_to_data_supported_rank`), so the β-tier here
        // always sees a full-rank design and needs no step-time data-null
        // deflation operator. The well-conditioned (full-rank) path is unchanged.
        // Per-atom smoothness-gradient GEMMs `½(S_k+S_kᵀ)·B_k` are independent
        // across atoms; batch them across ALL GPUs (uniform-shape tiles) and
        // scale by `lambda_smooth` below. `symmetrize = true` reproduces the
        // per-atom symmetrised `scaled_s/λ` used by the Kronecker op. Exact CPU
        // fallback per atom keeps the result bit-for-bit with the all-CPU path.
        let sym_sb_inputs: Vec<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> = self
            .atoms
            .iter()
            .map(|atom| (atom.smooth_penalty.view(), atom.decoder_coefficients.view()))
            .collect();
        let sym_sb_all = batched_smooth_sb(&sym_sb_inputs, true);
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let off = beta_offsets[atom_idx];
            // Symmetrise and scale the smoothness penalty matrix.
            let mut scaled_s = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    let s_ij = 0.5 * (atom.smooth_penalty[[i, j]] + atom.smooth_penalty[[j, i]]);
                    scaled_s[[i, j]] = lambda_smooth[atom_idx] * s_ij;
                }
            }
            // Gradient: g[beta_i] += (λ_k S_k B_k)[i, out_col]. The (m×m)·(m×p)
            // GEMM `½(S+Sᵀ)·B_k` was computed in the multi-GPU batch above; here
            // we only apply atom k's `lambda_smooth[atom_idx]`.
            let sb = &sym_sb_all[atom_idx] * lambda_smooth[atom_idx];
            for out_col in 0..p {
                for i in 0..m {
                    let beta_i = off + i * p + out_col;
                    smooth_grad_gb[beta_i] += sb[[i, out_col]];
                }
            }
            // IdentityRightKroneckerPenaltyOp: factor_a = λ·S_k (m×m), factor_b = I_p.
            smooth_ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
                factor_a: scaled_s.clone(),
                p,
                global_offset: off,
                k: beta_dim,
            }));
            // Retain `λ S_k` for the factored rebuild (no-op cost on full-`B`).
            smooth_scaled_s.push(scaled_s);
        }

        // Only the explicit hard-TopK model has a compact row layout. Every
        // smooth assignment family has nonzero full-support derivatives and
        // therefore remains exact; silently truncating those rows would make
        // the assembled gradient/Hessian derivatives of a different forward
        // objective.
        let coord_dims: Vec<usize> = self
            .assignment
            .coords
            .iter()
            .map(|c| c.latent_dim())
            .collect();
        let forced_compact = matches!(forced_layout, Some(Some(_)));
        let row_layout: Option<SaeRowLayout> = match forced_layout {
            Some(layout) => layout,
            None => match self.assignment.mode {
                AssignmentMode::TopK { k } => {
                    // The support IS the layout: TopK gates are exactly {0, 1},
                    // so the compact row block is precisely the k support atoms —
                    // no cutoff heuristics, no near-threshold population, and the
                    // per-token block is bounded at `k·d` BY CONSTRUCTION
                    // (the #2071 block-size contract holds with equality).
                    // Independent read-only per-row builds → order-preserving
                    // parallel collect is bit-identical to the serial push.
                    let assignments_all = self.assignments_all_parallel(n)?;
                    Some(SaeRowLayout::from_topk_gates(
                        &assignments_all,
                        k,
                        coord_dims.clone(),
                        self.assignment.coord_offsets(),
                    )?)
                }
                AssignmentMode::Softmax { .. }
                | AssignmentMode::OrderedBetaBernoulli { .. }
                | AssignmentMode::ThresholdGate { .. } => None,
            },
        };
        if !matches!(self.assignment.mode, AssignmentMode::TopK { .. }) && forced_compact {
            return Err(
                "compact row layouts are valid only for AssignmentMode::TopK; smooth assignment modes require exact full-support assembly"
                    .to_string(),
            );
        }

        // SPEC's never-OOM rule: refuse before allocating an exact dense system
        // whose two irreducible resident blocks exceed the host in-core budget.
        // The row tier stores one q×q block per row, while the full-support
        // decoder data Gram stores m_total² scalar entries (the `⊗ I_p`
        // structure avoids an additional p² factor). TopK has an exact sparse
        // representation and never enters this check.
        if row_layout.is_none() {
            let budget_bytes = sae_host_in_core_budget_bytes().0;
            self.require_exact_dense_assignment_budget(budget_bytes)?;
        }
        // #974 likelihood-whitening seam. The single per-row decision: when the
        // installed `RowMetric` is a genuinely estimated noise model
        // (`whitens_likelihood()` — only `WhitenedStructured`), the
        // reconstruction data-fit, its t-block Gauss-Newton row block, AND the
        // β-tier data-fit gradient are all assembled through the SAME per-row
        // metric `M_n = U_n U_nᵀ = Σ_n^{-1}`. There is exactly ONE construction
        // site (the `whiten_rows` closure below), so the value the line-search
        // sums and the gradient/Hessian the Newton step solves cannot drift apart
        // (the objective↔gradient-desync cure). For Euclidean / OutputFisher /
        // no-metric the closure is the identity and every downstream loop is
        // byte-identical to the historical isotropic path.
        let whitens_likelihood = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        // #972 / #977 T1: engage the FACTORED Grassmann-coordinate β-tier when
        // any atom has an active decoder frame. The closed-form factorization
        // `Φᵀ(G ⊗ I_p)Φ = G ⊗ (U_iᵀU_j)` is EXACT only for the isotropic
        // likelihood; under an active whitening metric (`whitens_likelihood()`)
        // the per-row output factor is `U_iᵀ M_n U_j` and does NOT factor out of
        // the basis Gram. #974 closes that gap: when `whitens_likelihood`, the
        // factored data-fit β-Hessian is built as the exact per-row sandwich
        // `Σ_n Φ_nᵀ M_n Φ_n` ([`WhitenedFactoredFrameOp`]) and the cross-block
        // `H_tβ` slab is whitened at write time (`L_i M_n J_β^framed`), so frames
        // now engage under whitening too — the memory-wall fix on the production
        // (whitened) composed path. The isotropic Euclidean / OutputFisher /
        // no-metric case keeps the separable `G ⊗ (U_iᵀU_j)` operator bit-for-bit.
        // When `frames_engaged` is false, every β-tier object below is assembled
        // bit-for-bit as the historical full-`B` path.
        let frames_engaged = self.any_frame_active();
        // #1407: fixed-decoder mode skips the entire β decoder tier (G/gb/htbeta
        // operator/hbb/β-penalties); only per-row htt/gt are produced.
        let fixed_decoder = self.fixed_decoder_assembly;
        let admission_plan = self
            .streaming_plan()
            .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
            .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        // #1407: fixed-decoder builds NO dense β-Hessian (hbb) — force the
        // empty-hbb system constructor so no `beta_dim × beta_dim` workspace is
        // taken (the early return skips `reclaim_border_hbb_workspace`).
        let dense_beta_curvature = !fixed_decoder
            && admission_plan.direct_admitted
            && !(frames_engaged && beta_dim > dense_beta_penalty_probe_max_dim);
        // #1406: the dense per-row cross-block slab `block.htbeta` is only WRITTEN
        // (line ~4243) and READ by the solver when `frames_engaged` (the factored
        // full-B path, which installs NO matrix-free row operator → the solver's
        // `sys_htbeta_apply_row` falls back to the dense slab). On the
        // `!frames_engaged` path the cross block is carried entirely by the
        // matrix-free Kronecker operator (`set_row_htbeta_operator`, ~line 4491);
        // `activate_dense_htbeta_supplement` is never called, so the solver never
        // touches `block.htbeta`. Allocating it at `beta_dim = K·M·p` there is the
        // ~6 TiB high-K leak (#1405/#1406): allocate ZERO columns instead. Frames
        // still use the (much smaller) factored border width.
        // #795/#1406/#1407: the non-frames matrix-free path normally holds a
        // ZERO-width per-row cross-block slab — the data-fit `H_tβ` is carried by
        // the Kronecker row operator (`set_row_htbeta_operator`), and allocating
        // the dense slab at `beta_dim = K·M·p` is the high-K memory leak. But an
        // ISOMETRY penalty on a coherence-preserving (flat) chart scatters an
        // ADDITIONAL Gauss-Newton cross-block into the dense per-row `htbeta`
        // slab and flips on `activate_dense_htbeta_supplement` — dropping it would
        // leave the Newton system block-diagonal and forfeit the strong `t↔B`
        // isometry coupling the circle fit needs to reach KKT stationarity (#795).
        // So on the non-frames path widen the slab to `beta_dim` exactly when that
        // dense supplement will be written, and keep zero width otherwise.
        let dense_isometry_cross_block = !fixed_decoder
            && analytic_penalties
                .map(|registry| self.registry_writes_dense_isometry_cross_block(registry))
                .unwrap_or(false);
        let row_htbeta_dim = if fixed_decoder {
            // Fixed-decoder mode skips the β tier entirely.
            0
        } else if frames_engaged {
            self.factored_border_dim()
        } else if dense_isometry_cross_block {
            // Matrix-free data-fit cross-block + dense isometry supplement: the
            // supplement is written/read in the full-`B` β coordinate system.
            beta_dim
        } else {
            // Matrix-free path with no dense cross-block supplement.
            0
        };
        // Build the Arrow-Schur system: heterogeneous row dims when a compact
        // layout is active, uniform `q` otherwise. Successive nonlinear
        // iterations take the row/gradient allocations returned by the driver;
        // `new_with_assembly_buffers` zeroes every numerical entry before this
        // iterate refills it, so residency never becomes stale factor reuse.
        let per_row_dims: Vec<usize> = match row_layout.as_ref() {
            Some(layout) => (0..n).map(|row| layout.row_q_active(row)).collect(),
            None => vec![q; n],
        };
        let hbb_workspace = if dense_beta_curvature {
            self.take_border_hbb_workspace(beta_dim)
        } else {
            self.border_hbb_workspace = Array2::<f64>::zeros((0, 0));
            Array2::<f64>::zeros((0, 0))
        };
        let (rows_workspace, gb_workspace) = self.take_arrow_assembly_buffers();
        let mut sys = ArrowSchurSystem::new_with_assembly_buffers(
            per_row_dims,
            beta_dim,
            row_htbeta_dim,
            hbb_workspace,
            rows_workspace,
            gb_workspace,
        );
        // Apply accumulated smoothness-penalty gradients into sys.gb.
        for (i, g) in smooth_grad_gb.iter().enumerate() {
            sys.gb[i] += g;
        }
        // `w_dim` is the whitened output dimension: `rank` of the metric factor
        // when whitening, else `p` (identity). `error_white` is the whitened
        // residual `U_nᵀ r_n ∈ ℝ^{w_dim}` whose squared norm is `r_nᵀ M_n r_n`,
        // shared by the value path, the t-block GN, and (lifted back to p-space)
        // the β-tier gradient.
        let w_dim = match self.row_metric.as_ref() {
            Some(metric) if whitens_likelihood => metric.metric_rank(),
            _ => p,
        };
        // #974 — a genuinely rank-deficient whitening metric (`rank < p`, e.g. an
        // `s`-probe BehavioralFisher sketch with `s < p`). In that regime the
        // per-row t-block Gauss-Newton curvature `H_tt = J_t Mₙ J_tᵀ` is
        // rank-limited: in directions where the reconstruction Jacobian row lies
        // in the metric's null space it carries NO data curvature, so the
        // (indefinite) assignment/ARD prior curvature — which the full-rank
        // isotropic data curvature normally dominates — is exposed and can drive
        // the evidence-mode `H_tt` Cholesky slightly non-PD. This flag gates the
        // spectral-deflation opt-in for that block below; the identity-metric
        // (`rank == p`) and no-metric paths keep `low_rank_whiten == false` and
        // are bit-for-bit unchanged.
        let low_rank_whiten = whitens_likelihood && w_dim < p;
        // PSD-majorize the ordered Beta--Bernoulli prior curvature on every
        // path. Its exact mass rank-one coefficient is strictly negative, so
        // zero is its PSD Loewner majorizer; the row-local concrete-Jacobian
        // term is replaced by its positive part. The exact prior gradient is
        // untouched, while assembly, rho traces, and theta adjoints all
        // differentiate this same declared curvature operator.
        let ordered_beta_bernoulli_majorizer =
            ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?;
        // Data-fit Gauss-Newton β-Hessian is block-diagonal across the `p`
        // output channels and identical in each: with the flat β layout
        // `β[μ·p + oc] = B[μ, oc]` (μ enumerating (atom, basis_col)) the GN
        // outer product `Jβᵀ Jβ` couples only equal `oc`, with the same
        // `(M_total × M_total)` block `G[μ, μ'] = Σ_rows (a_k φ_k[m])(a_{k'} φ_{k'}[m'])`
        // for every channel. So `H_data = G ⊗ I_p`. The `μ` index of an `a_phi`
        // entry whose global β base is `beta_base` is `beta_base / p` (every
        // `beta_offset` and the `basis_col·p` stride are multiples of `p`).
        //
        // `G` is only non-zero on `(atom_i, atom_j)` pairs that co-occur in
        // some row's active set, so we accumulate it as a sparse map of dense
        // per-atom-pair `(m_i × m_j)` blocks keyed by `(atom_i, atom_j)` rather
        // than as a dense `(m_total × m_total)` matrix. At `K = 100K` with
        // per-row active sets of size `k_active ≪ K`, only `O(N · k_active²)`
        // pairs are ever touched, so the data Gram (and every matvec /
        // diagonal pass over it via `SparseBlockKroneckerPenaltyOp`) tracks the
        // active atoms instead of `K²`. In the dense full-support layout the
        // map degenerates to every co-occurring pair, reproducing the dense
        // Gram exactly. A `BTreeMap` key order keeps the installed op's
        // fingerprint deterministic. The `μ`-space offset of atom `k` is
        // `beta_offsets[k] / p`.
        type SaeGBlocks = std::collections::BTreeMap<(usize, usize), Array2<f64>>;
        let m_total: usize = self.atoms.iter().map(|a| a.basis_size()).sum();
        let mu_offsets: Vec<usize> = beta_offsets.iter().map(|&off| off / p).collect();
        // #991 design honesty weights (mean-1 HT inclusion corrections); see
        // the seam comment at the per-row residual below.
        let row_loss_w = self.row_loss_weights.as_deref();
        // Dense full-support index `[0, k_atoms)`, used by the row loop when no
        // compact layout is engaged so the active-atom iteration is uniform.
        let all_atoms_index: Vec<usize> = (0..k_atoms).collect();
        // Per-atom per-axis periodicity, hoisted out of the row loop. Selects
        // the smooth von-Mises coordinate prior on wrapped (Circle) axes and
        // the Gaussian prior on Euclidean axes; see `ArdAxisPrior`.
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        struct SaeAssemblyRow {
            pub(crate) row: usize,
            pub(crate) block: ArrowRowBlock,
            pub(crate) gb_delta: Vec<(usize, f64)>,
            pub(crate) g_blocks: SaeGBlocks,
            pub(crate) kron_a_phi: Option<Vec<(usize, f64)>>,
            pub(crate) kron_jac: Option<Vec<f64>>,
            /// #974 per-row active support `(atom, basis, a·φ)` for the whitened
            /// factored β-Hessian operator. `Some` only on the frames+whitening
            /// path; `None` (and never allocated) otherwise.
            pub(crate) frame_support: Option<Vec<(usize, usize, f64)>>,
        }

        // Per-row scratch reused across all rows a rayon worker processes
        // (#1017). The assembly closure is re-run every inner Newton iteration ×
        // every outer ρ evaluation; allocating these eight loop-invariant-sized
        // buffers (`k_atoms·p`, several `p`, one `q·max(w_dim,p)`) once per
        // worker via `map_init` — rather than once per (row × assembly) inside
        // the closure — removes the dominant small-allocation traffic the
        // eu-stack profile attributed to allocator/barrier spin at the SAE LLM
        // shape (p≈5120). Every buffer is fully filled (or `.fill(0.0)`'d) before
        // it is read each row, so reuse is bit-identical to the fresh-alloc path;
        // `gb_delta`/`g_blocks` are NOT scratch (they move into the returned
        // `SaeAssemblyRow`) and stay allocated per row.
        struct RowScratch {
            pub(crate) decoded: Array2<f64>,
            pub(crate) dg_buf: Vec<f64>,
            pub(crate) fitted: Array1<f64>,
            pub(crate) error: Array1<f64>,
            pub(crate) error_white: Vec<f64>,
            pub(crate) error_metric: Array1<f64>,
            pub(crate) jac_white: Vec<f64>,
            pub(crate) decoded_scratch: Vec<f64>,
            // #1557 — per-worker scratch for the row assignment vector (filled via
            // `_into`, not allocated per row); full `k_atoms`, global-atom indexed.
            pub(crate) assignments: Array1<f64>,
        }
        // #1410: size the per-worker scratch by the COMPACT row dimensions, not
        // full `K`/`q`. With a compact layout the assembly only ever touches each
        // row's active atoms (≤ `max_active`) and its compact tangent block
        // (≤ `max_q_row`); allocating `decoded` at `k_atoms·p` and `jac_white` at
        // `q·max(w_dim,p)` was the per-worker `O(K)` blow-up (≈11 GiB/worker at
        // K=100k, p=5120 — and `map_init` gives every Rayon worker its own copy).
        // Without a layout the dense path needs full `k_atoms`/`q`. `decoded` rows
        // are addressed by COMPACT SLOT in the compact branch below (the dense
        // branch keeps global-atom rows), so the row count is the max active set.
        //
        // The compact branch is exclusively the exact TopK support. Smooth
        // assignment modes use the full dimensions and have already passed the
        // exact-memory admission check above.
        let (decoded_rows, scratch_q) = match row_layout.as_ref() {
            Some(layout) => {
                let max_active = (0..n)
                    .map(|r| layout.active_atoms[r].len())
                    .max()
                    .unwrap_or(0)
                    .max(1);
                let max_q_row = (0..n)
                    .map(|r| layout.row_q_active(r))
                    .max()
                    .unwrap_or(q)
                    .max(1);
                (max_active, max_q_row)
            }
            None => (k_atoms, q),
        };
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        // #1033 large-n: fold the per-row assembly results in row-ordered CHUNKS
        // rather than collecting all `n` `SaeAssemblyRow`s at once. The previous
        // path materialized the FULL `Vec<SaeAssemblyRow>` (every row's htt/gt
        // block + per-row `g_blocks` + `kron_a_phi`/`kron_jac`) AND the fold
        // destinations simultaneously — a ~2× transient peak over the resident
        // system during the fold, the assembly-side OOM cliff at large `n`. By
        // collecting one chunk, folding it into `sys.rows`/`g_blocks`/`kron_*`,
        // and dropping the chunk's `Vec` before the next chunk, the transient
        // intermediate is bounded to `O(chunk_size)` while the resident output is
        // unchanged. The fold stays STRICTLY row-ascending (chunk `[c0..c1)` then
        // `[c1..c2)`, rows in order within each chunk), so every `+=` into
        // `sys.gb`, the `g_blocks` BTreeMap, and the `kron_*` pushes lands in the
        // identical order as the single-pass fold — bit-for-bit the same system.
        // Chunk width is the admission plan's `chunk_size` (the same value
        // `streaming_plan` sizes for the matrix-free window), floored so a tiny
        // plan still makes forward progress.
        let assembly_chunk_rows = self
            .assembly_chunk_override
            .unwrap_or(admission_plan.chunk_size)
            .clamp(1, n.max(1));
        let mut g_blocks: SaeGBlocks = std::collections::BTreeMap::new();
        let mut kron_a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut kron_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
        // #974 whitened-frames per-row support `(atom, basis, a·φ)`, collected in
        // ascending row order for the `WhitenedFactoredFrameOp`. Stays empty off
        // the frames+whitening path.
        let mut frame_support_rows: Vec<Vec<(usize, usize, f64)>> =
            if frames_engaged && whitens_likelihood {
                Vec::with_capacity(n)
            } else {
                Vec::new()
            };
        let mut chunk_start = 0usize;
        while chunk_start < n {
            let chunk_end = (chunk_start + assembly_chunk_rows).min(n);
            let mut fold_offset_in_chunk = 0usize;
            let row_results: Vec<SaeAssemblyRow> = (chunk_start..chunk_end)
                .into_par_iter()
                .map_init(
                    || RowScratch {
                        decoded: Array2::<f64>::zeros((decoded_rows, p)),
                        dg_buf: vec![0.0_f64; p],
                        fitted: Array1::<f64>::zeros(p),
                        error: Array1::<f64>::zeros(p),
                        error_white: vec![0.0_f64; w_dim],
                        error_metric: Array1::<f64>::zeros(p),
                        jac_white: vec![0.0_f64; scratch_q * w_dim.max(p)],
                        decoded_scratch: vec![0.0_f64; p],
                        assignments: Array1::<f64>::zeros(k_atoms),
                    },
                    |scratch, row| -> Result<SaeAssemblyRow, String> {
                        // #1557 — mark this rayon row worker as a nested data-parallel
                        // region so any faer GEMM reached transitively from the per-row
                        // assembly (frame `Uᵀ` products, the per-row cross-block /
                        // Schur-accumulation matmuls, the Riemannian projections) pins to
                        // `Par::Seq` via `effective_global_parallelism` instead of
                        // re-fanning the global Rayon pool against this outer fan-out
                        // (the `spindle` barrier-spin). Serial vs parallel over these tiny
                        // per-row blocks is a single small product, so the result is
                        // bit-identical. The guard is held for the whole closure body
                        // including its `?`/`return` paths.
                        with_nested_parallel(|| {
                        let RowScratch {
                            decoded,
                            dg_buf,
                            fitted,
                            error,
                            error_white,
                            error_metric,
                            jac_white,
                            decoded_scratch,
                            assignments,
                        } = scratch;
                        let mut gb_delta: Vec<(usize, f64)> = Vec::new();
                        let mut g_blocks: SaeGBlocks = std::collections::BTreeMap::new();
                        // #1557 — fill per-worker scratch (bit-identical to alloc path).
                        let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
                        self.assignment
                            .try_assignments_row_into(row, a_scratch)?;
                        // Reconstruction uses the row's active support: for the dense
                        // full-support layout this is all atoms (exact); for a compact
                        // layout the dropped atoms carry negligible `O(a)` reconstruction
                        // mass and zero curvature, so excluding them keeps `fitted`,
                        // `error`, and the logit-JVP cross term `(decoded[k] − fitted)`
                        // mutually consistent with the curvature actually assembled.
                        fitted.fill(0.0);
                        let row_active_owned: Option<&[usize]> =
                            row_layout.as_ref().map(|l| l.active_atoms[row].as_slice());
                        match row_active_owned {
                            Some(active) => {
                                // #1410: `decoded` is a compact (max_active × p) buffer
                                // here; index it by the active-set SLOT `j` (the same
                                // index the compact tangent block / `coord_starts` use),
                                // NOT the global `atom_idx`.
                                for (j, &atom_idx) in active.iter().enumerate() {
                                    let a_k = assignments[atom_idx];
                                    self.atoms[atom_idx]
                                        .fill_decoded_row(row, decoded_scratch.as_mut_slice());
                                    for out_col in 0..p {
                                        decoded[[j, out_col]] = decoded_scratch[out_col];
                                        fitted[out_col] += a_k * decoded_scratch[out_col];
                                    }
                                }
                            }
                            None => {
                                for atom_idx in 0..k_atoms {
                                    let a_k = assignments[atom_idx];
                                    self.atoms[atom_idx]
                                        .fill_decoded_row(row, decoded_scratch.as_mut_slice());
                                    for out_col in 0..p {
                                        decoded[[atom_idx, out_col]] = decoded_scratch[out_col];
                                        fitted[out_col] += a_k * decoded_scratch[out_col];
                                    }
                                }
                            }
                        }
                        for out_col in 0..p {
                            error[out_col] = fitted[out_col] - target[[row, out_col]];
                        }
                        // #991 design-honesty seam: a per-row scalar weight `w_row` on the
                        // reconstruction channel is exactly the metric `w_row · I_p`, so it
                        // is realized as a `√w_row` scaling of the THREE row-local data
                        // quantities at their construction sites — this residual, the
                        // latent Jacobian (below), and the β basis load `a·φ` (below).
                        // Every downstream data object then carries exactly one factor of
                        // `w_row` (gt, htt, htbeta, the β Gram `G`, and the β gradient),
                        // matching the `w_row`-weighted value `loss_scaled` sums; the
                        // per-row latent priors (assignment / ARD, added to `gt`/`htt`
                        // further down) are deliberately unweighted — see the
                        // `row_loss_weights` field docs. `None` ⇒ `sqrt_row_w == 1.0` and
                        // no multiply is applied (bit-identical unweighted path).
                        let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
                        if sqrt_row_w != 1.0 {
                            for out_col in 0..p {
                                error[out_col] *= sqrt_row_w;
                            }
                        }
                        // #974 seam (step 1/2): whiten the per-row residual ONCE.
                        //   * not whitening ⇒ `error_white == error` (length p) and
                        //     `error_metric == error`; every downstream loop is the
                        //     historical isotropic path bit-for-bit.
                        //   * whitening ⇒ `error_white = U_nᵀ r_n ∈ ℝ^{w_dim}` (its squared
                        //     norm is `r_nᵀ M_n r_n`, the value the data-fit sums) and
                        //     `error_metric = U_n (U_nᵀ r_n) = M_n r_n ∈ ℝ^p` (the p-space
                        //     metric-applied residual the β-tier gradient contracts).
                        match self.row_metric.as_ref() {
                            Some(metric) if whitens_likelihood => {
                                let wr = metric.whiten_residual_row(row, error.view());
                                for (slot, &v) in error_white.iter_mut().zip(wr.iter()) {
                                    *slot = v;
                                }
                                let mr = metric.apply_metric_row(row, error.view());
                                for (slot, &v) in error_metric.iter_mut().zip(mr.iter()) {
                                    *slot = v;
                                }
                            }
                            _ => {
                                for out_col in 0..p {
                                    error_white[out_col] = error[out_col];
                                    error_metric[out_col] = error[out_col];
                                }
                            }
                        }

                        // Determine whether this row uses the compact hard-TopK
                        // support layout. Every differentiable assignment mode
                        // uses the dense layout because its nonzero derivatives
                        // must not be truncated.
                        let (q_row, mut local_jac_row) = if let Some(layout) = row_layout.as_ref() {
                            let active = &layout.active_atoms[row];
                            let starts = &layout.coord_starts[row];
                            let q_active = layout.row_q_active(row);
                            let mut jac_compact = Array2::<f64>::zeros((q_active, p));
                            // Coordinate JVP rows for active atoms only.
                            for (j, &k) in active.iter().enumerate() {
                                let d = self.atoms[k].latent_dim;
                                let a_k = assignments[k];
                                let coord_start = starts[j];
                                for axis in 0..d {
                                    self.atoms[k].fill_decoded_derivative_row(
                                        row,
                                        axis,
                                        dg_buf.as_mut_slice(),
                                    );
                                    for out_col in 0..p {
                                        jac_compact[[coord_start + axis, out_col]] =
                                            a_k * dg_buf[out_col];
                                    }
                                }
                            }
                            (q_active, jac_compact)
                        } else {
                            // Fresh per-row Jacobian, structurally identical to the
                            // ThresholdGate branch: every (q × p) element is unconditionally
                            // overwritten below (assignment-chart JVP rows + coordinate rows), so the
                            // `Array2::zeros` allocation needs no separate `fill(0.0)` and
                            // the populated buffer is returned by move without a clone.
                            let mut jac_row = Array2::<f64>::zeros((q, p));
                            fill_assignment_logit_jvp_rows(
                                self.assignment.mode,
                                self.assignment.logits.row(row),
                                assignments.view(),
                                decoded.view(),
                                fitted.view(),
                                // #1026/#1033: zero logit-JVP rows for FIXED-logit atoms
                                // (ungated, and all atoms under frozen routing).
                                &self.assignment.fixed_logit_mask(),
                                &mut jac_row,
                            );
                            // Coordinate columns for all atoms.
                            for atom_idx in 0..k_atoms {
                                let d = self.atoms[atom_idx].latent_dim;
                                let off = coord_offsets[atom_idx];
                                let a_k = assignments[atom_idx];
                                for axis in 0..d {
                                    self.atoms[atom_idx].fill_decoded_derivative_row(
                                        row,
                                        axis,
                                        dg_buf.as_mut_slice(),
                                    );
                                    for out_col in 0..p {
                                        jac_row[[off + axis, out_col]] = a_k * dg_buf[out_col];
                                    }
                                }
                            }
                            (q, jac_row)
                        };

                        // #991 design-honesty seam, Jacobian leg: scale the row's latent
                        // Jacobian by `√w_row` BEFORE the whitening / Kronecker capture so
                        // htt (= J̃J̃ᵀ), the data part of gt (= J̃ẽ, the residual already
                        // carries its own √w_row), and the htbeta cross block (J paired
                        // with the √w_row-scaled β load below) each carry exactly one
                        // factor of `w_row`. No-op on the unweighted path.
                        if sqrt_row_w != 1.0 {
                            for a in 0..q_row {
                                for out_col in 0..p {
                                    local_jac_row[[a, out_col]] *= sqrt_row_w;
                                }
                            }
                        }

                        // #974 seam (step 2/2): whiten the per-row Jacobian through the SAME
                        // metric the residual was whitened by. `jac_white[a*w_dim + k]` holds
                        // `J̃[a, k] = Σ_out U_n[out, k] · J_n[a, out]` so the t-block
                        // Gauss-Newton row block is `htt = J̃ J̃ᵀ = J_n M_n J_nᵀ` and
                        // `gt = J̃ ẽ = J_nᵀ M_n r_n`. When not whitening, `w_dim == p` and the
                        // whitened jac equals the raw Jacobian, so htt/gt are byte-identical
                        // to the historical isotropic assembly. Because the SAME `error_white`
                        // feeds both the value-path data-fit (Σ½ ẽ²) and this gradient
                        // (J̃ ẽ), the objective and its t-block gradient share one whitening
                        // — they cannot desync.
                        if whitens_likelihood {
                            if let Some(metric) = self.row_metric.as_ref() {
                                for a in 0..q_row {
                                    for k in 0..w_dim {
                                        let mut acc = 0.0;
                                        // U_n[out, k] read through the metric's factor layout.
                                        for out_col in 0..p {
                                            acc += metric.factor_entry(row, out_col, k)
                                                * local_jac_row[[a, out_col]];
                                        }
                                        jac_white[a * w_dim + k] = acc;
                                    }
                                }
                            }
                        } else {
                            for a in 0..q_row {
                                for out_col in 0..p {
                                    jac_white[a * w_dim + out_col] = local_jac_row[[a, out_col]];
                                }
                            }
                        }

                        // Build the per-row Arrow-Schur block at the row's active dim.
                        let mut block = ArrowRowBlock::new(q_row, row_htbeta_dim);
                        for a in 0..q_row {
                            let jac_a = &jac_white[a * w_dim..(a + 1) * w_dim];
                            let g = jac_a
                                .iter()
                                .zip(error_white.iter())
                                .map(|(&j, &e)| j * e)
                                .sum::<f64>();
                            block.gt[a] += g;
                            for b in 0..q_row {
                                let jac_b = &jac_white[b * w_dim..(b + 1) * w_dim];
                                let h = jac_a
                                    .iter()
                                    .zip(jac_b.iter())
                                    .map(|(&ja, &jb)| ja * jb)
                                    .sum::<f64>();
                                block.htt[[a, b]] += h;
                            }
                        }

                        // Assignment prior in logit space.
                        // For compact layout: position `j` = active_atoms index.
                        // For dense layout: position `atom_idx` directly.
                        //
                        // H-consistency note (#1006 audit / #1416 update). This
                        // `assignment_hdiag` is the assignment channel's diagonal
                        // curvature. Smooth-threshold curvature is exact; the
                        // nonconvex assignment modes use their declared PSD
                        // Loewner majorizers:
                        //
                        //   * softmax entropy has dense within-row Hessian
                        //     H_kj = (λ/τ²) a_k[δ_kj(m-L_k-1) + a_j(L_k+L_j+1-2m)];
                        //     this diagonal stores its Gershgorin Loewner majorizer (#1419).
                        //   * the integrated ordered Beta--Bernoulli marginal has
                        //     `s'=-ψ₁(M+a)-ψ₁(N-M+1)<0`, so its cross-row
                        //     rank-one block `w·s'·J Jᵀ` is negative semidefinite.
                        //     Its PSD Loewner majorizer is therefore exactly zero.
                        //     The remaining row-local concrete-Jacobian term is
                        //     isolated and clamped by
                        //     `ordered_beta_bernoulli_psd_majorized_hdiag`.
                        let assignment_base = row * k_atoms;
                        if row_layout.is_none() {
                            for free_idx in 0..assignment_dim {
                                block.gt[free_idx] += assignment_grad[assignment_base + free_idx];
                            }
                            if let Some((penalty, scale)) = softmax_dense.as_ref() {
                                // #1419: write the genuine Gershgorin Loewner majorizer
                                // `D = diag(Σ_j|H_kj|)` of the exact entropy Hessian onto the
                                // row's logit block in place of the EXACT entropy Hessian. The
                                // entropy Hessian is INDEFINITE (concave directions on
                                // long-tailed rows), which drove the per-row evidence block
                                // non-PD and forced the downstream Faddeev–Popov deflation to
                                // flatten data-relevant logit directions (under-identifying the
                                // atoms). `D` is a nonnegative diagonal, hence exactly PSD and
                                // PD-preserving like the previous Fisher surrogate, so the block
                                // stays PD and the deflation no longer fires on the entropy
                                // block. Unlike the Fisher metric `G = scale·(diag(a) − a aᵀ)`,
                                // which is PSD but NOT a majorizer (`G − H_entropy` can be
                                // indefinite — K=2, a=(0.95,0.05): G₁₁=0.0475 < H₁₁=0.0784,
                                // #1419), `D` actually satisfies `D ⪰ H_entropy` and `D ⪰ 0`,
                                // so it is a true MM/Loewner curvature majorizer. Because the
                                // entropy penalty is a FIXED prior whose stationary point is set
                                // by its (unchanged) EXACT gradient, replacing its curvature
                                // with the majorizer only conditions the Newton step and the
                                // Laplace normalizer's curvature operator — it does NOT move the
                                // optimum.
                                //
                                // Softmax uses the REDUCED K−1 free-logit chart (the last
                                // reference logit is fixed at 0, `assignment_coord_dim() = K−1`).
                                // Holding z_{K-1} fixed, the reduced curvature over the free
                                // logits 0..K−1 is exactly the top-left (K−1)×(K−1) submatrix of
                                // the full K×K majorizer (the fixed logit contributes no
                                // row/column to the free curvature). The criterion's `log|H|`
                                // and the #1006 θ-adjoint differentiate this SAME `D` (see the
                                // `row_psd_majorizer_logit_derivative` site below), so value and
                                // adjoint stay on one exact branch.
                                let row_logits: Vec<f64> = (0..k_atoms)
                                    .map(|k| self.assignment.logits[[row, k]])
                                    .collect();
                                // #991 — fold this row's design weight into the
                                // majorizer strength (the block is not sourced from
                                // the weighted `assignment_hdiag`); the θ-adjoint at
                                // `row_psd_majorizer_logit_derivative` carries the
                                // same `w_row` so value and adjoint stay on one branch.
                                let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                                let h_dense = penalty.row_psd_majorizer(&row_logits, *scale * w_row);
                                for ki in 0..assignment_dim {
                                    for kj in 0..assignment_dim {
                                        block.htt[[ki, kj]] += h_dense[[ki, kj]];
                                    }
                                }
                            } else {
                                for free_idx in 0..assignment_dim {
                                    let raw = assignment_hdiag[assignment_base + free_idx];
                                    // #2144: PSD-majorize the ordered Beta--Bernoulli diagonal under a
                                    // low-rank whitening metric (no-op otherwise).
                                    let val = match ordered_beta_bernoulli_majorizer.as_ref() {
                                        Some(ch) => ordered_beta_bernoulli_psd_majorized_hdiag(
                                            ch, row, k_atoms, free_idx, raw,
                                        ),
                                        None => raw,
                                    };
                                    block.htt[[free_idx, free_idx]] += val;
                                }
                            }
                        }

                        // ARD on each on-atom coordinate.
                        // For compact layout: only active atoms; coord positions use compact starts.
                        // For dense layout: all atoms; coord positions use coord_offsets.
                        //
                        // HT row weighting: the data channel is scaled by `√w_row`
                        // (curvature ⇒ `w_row`; see the residual seam above), so this
                        // per-row ARD prior — a genuine per-row prior that shares the
                        // criterion's `ard_value` energy — must carry the SAME full
                        // `w_row` on BOTH its gradient (inner-MAP stationarity vs the
                        // weight-aware `ard_value`) and its curvature (the ½log|H_tt|
                        // block and its ρ-trace). `None` ⇒ `w_row = 1`, bit-for-bit.
                        let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                        if let Some(layout) = row_layout.as_ref() {
                            let active = &layout.active_atoms[row];
                            let starts = &layout.coord_starts[row];
                            for (j, &k) in active.iter().enumerate() {
                                let coord = &self.assignment.coords[k];
                                let d = coord.latent_dim();
                                if rho.log_ard[k].is_empty() {
                                    continue;
                                }
                                if rho.log_ard[k].len() != d {
                                    return Err(format!(
                                        "ARD rho atom {k} has len {} but atom dim is {d}",
                                        rho.log_ard[k].len()
                                    ));
                                }
                                let row_t = coord.row(row);
                                let periods = &ard_axis_periods[k];
                                for axis in 0..d {
                                    // ARD on coords is a genuine per-row prior (each row
                                    // contributes the per-axis prior energy), so it is NOT
                                    // minibatch-scaled — the per-chunk row sums already
                                    // reconstruct the full coordinate prior across a pass.
                                    // The value (`ard_value`/`loss.ard`) and the gradient
                                    // both come from the SAME `ArdAxisPrior` energy, so they
                                    // stay FD-consistent on periodic axes. The exact
                                    // von-Mises curvature `V'' = α·cos(κt)` is INDEFINITE —
                                    // it goes negative for |t| past a quarter period — so
                                    // writing it raw into the Newton/Schur `htt` diagonal
                                    // makes that PSD curvature block indefinite and the Schur
                                    // Cholesky (used both for the Newton step and the exact
                                    // log-det) fails on a non-PD pivot. Accumulate the PSD
                                    // majorizer `max(V'', 0)` instead, exactly as
                                    // `add_sae_coord_penalty` does for the registry coord
                                    // penalties: the positive part keeps `htt` PSD so the
                                    // factorization succeeds, and majorizing the curvature of
                                    // a fixed prior only damps the Newton step — it does not
                                    // move the stationary point (the gradient, which sets the
                                    // fixed point, stays the exact `V'`).
                                    let alpha =
                                        SaeManifoldRho::stable_exp_strength(rho.log_ard[k][axis]);
                                    let prior =
                                        ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                                    block.gt[starts[j] + axis] += w_row * prior.grad;
                                    block.htt[[starts[j] + axis, starts[j] + axis]] +=
                                        w_row * prior.hess.max(0.0);
                                }
                            }
                        } else {
                            for atom_idx in 0..k_atoms {
                                let coord = &self.assignment.coords[atom_idx];
                                let d = coord.latent_dim();
                                if rho.log_ard[atom_idx].is_empty() {
                                    continue;
                                }
                                if rho.log_ard[atom_idx].len() != d {
                                    return Err(format!(
                                        "ARD rho atom {atom_idx} has len {} but atom dim is {d}",
                                        rho.log_ard[atom_idx].len()
                                    ));
                                }
                                let off = coord_offsets[atom_idx];
                                let row_t = coord.row(row);
                                let periods = &ard_axis_periods[atom_idx];
                                for axis in 0..d {
                                    // PSD-majorize the (possibly negative) von-Mises curvature
                                    // into the Newton/Schur `htt` block; see the compact-layout
                                    // branch above for why `max(V'', 0)` is required to keep
                                    // `htt` PD (the exact `V'' = α·cos κt` is indefinite past a
                                    // quarter period and breaks the Schur/log-det Cholesky).
                                    let alpha = SaeManifoldRho::stable_exp_strength(
                                        rho.log_ard[atom_idx][axis],
                                    );
                                    let prior =
                                        ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                                    block.gt[off + axis] += w_row * prior.grad;
                                    block.htt[[off + axis, off + axis]] += w_row * prior.hess.max(0.0);
                                }
                            }
                        }

                        // Beta gradient/Hessian — Kronecker form J_β = φᵀ ⊗ I_p.
                        //
                        // The per-row beta Jacobian is
                        //   J_β[out_col, beta_idx] = a_k · phi_k[basis_col]   if out_col == out_col(beta_idx)
                        //                            0                         otherwise
                        // so the data-fit Gauss-Newton beta-Hessian factors as a rank-`p`
                        // sum of outer products. We pre-compute the per-(atom, basis_col)
                        // scalar `a_k · phi_k` once and reuse it across the `out_col`
                        // and inner `(atom_j, basis_col2)` loops.
                        //
                        // Full-B rows keep the matrix-free Kronecker path below. Factored
                        // rows write the `q_i × Σ M_k r_k` C-space cross slab directly by
                        // folding each output-channel contribution through the atom frame,
                        // so no `q_i × β_dim` slab is ever materialized.
                        //
                        // Only the row's active atoms contribute `a_phi` support and data
                        // curvature: in a compact layout (TopK support)
                        // top-`k_active` truncation) the inactive atoms carry zero (gated)
                        // or sub-cutoff assignment mass and are excluded — this is what
                        // keeps both the htbeta support and the `G` accumulation
                        // `O(k_active)` rather than `O(K)`. In the dense full-support
                        // layout `row_active` spans all atoms.
                        let row_active: &[usize] = match row_layout.as_ref() {
                            Some(layout) => layout.active_atoms[row].as_slice(),
                            None => &all_atoms_index,
                        };
                        // #1407: in fixed-decoder mode the β tier is not assembled at
                        // all — leave gb_delta/g_blocks empty and kron None. htt/gt
                        // (built above) are the only outputs the frozen-decoder step
                        // consumes.
                        let mut a_phi: Vec<(usize, f64)> = Vec::with_capacity(row_active.len() * 4);
                        // Per-active-atom weighted basis row `a_k · φ_k[·]`, retained so the
                        // data Gram blocks can be accumulated as clean per-atom-pair outer
                        // products `(a_k φ_k) (a_{k'} φ_{k'})ᵀ`.
                        let mut weighted_phi: Vec<(usize, Vec<f64>)> =
                            Vec::with_capacity(row_active.len());
                        if !fixed_decoder {
                            for &atom_idx in row_active {
                                let atom = &self.atoms[atom_idx];
                                let atom_beta_off = beta_offsets[atom_idx];
                                let m = atom.basis_size();
                                let a_k = assignments[atom_idx];
                                let mut wphi = Vec::with_capacity(m);
                                for basis_col in 0..m {
                                    let phi = atom.basis_values[[row, basis_col]];
                                    // #991 design-honesty seam, β leg: the `√w_row` here pairs
                                    // with the `√w_row` on the residual (β gradient =
                                    // `a·φ · M r` ⇒ w_row) and with itself (β Gram `G` and the
                                    // htbeta Kronecker capture ⇒ w_row). `1.0` when unweighted.
                                    let w = a_k * phi * sqrt_row_w;
                                    a_phi.push((atom_beta_off + basis_col * p, w));
                                    wphi.push(w);
                                }
                                weighted_phi.push((atom_idx, wphi));
                            }
                            // β data-fit gradient `gᵦ += J_βᵀ M_n r_n`. The β-Jacobian is
                            // `J_β = φ_nᵀ ⊗ I_p`, so `J_βᵀ M_n r_n = φ_n ⊗ (M_n r_n)` —
                            // contract the basis weight `a·φ` against the p-space metric-applied
                            // residual `error_metric` (= `M_n r_n`), the SAME whitening the value
                            // path and t-block share. When not whitening, `error_metric == error`
                            // and this is byte-identical to the historical `J_βᵀ r`.
                            for &(beta_base_i, j_beta_i) in a_phi.iter() {
                                if j_beta_i == 0.0 {
                                    continue;
                                }
                                for out_col in 0..p {
                                    gb_delta.push((
                                        beta_base_i + out_col,
                                        j_beta_i * error_metric[out_col],
                                    ));
                                    // No dense hbb write — the sparse `G ⊗ I_p` op installed
                                    // after the loop carries the data-fit GN β-Hessian.
                                }
                            }
                            if frames_engaged {
                                // #974: under whitening the frames cross-block is
                                // `H_tβ = L_i M_n J_β^framed`, so whiten each t-row's
                                // p-vector `L_i[c, :] ← M_n L_i[c, :]` ONCE per row
                                // before projecting through the frames below. Off the
                                // whitening path this is `None` and the raw
                                // `local_jac_row` is used bit-for-bit.
                                let ljr_white: Option<Array2<f64>> = if whitens_likelihood {
                                    let metric = self
                                        .row_metric
                                        .as_ref()
                                        .expect("whitens_likelihood ⇒ metric present");
                                    let mut w = Array2::<f64>::zeros((q_row, p));
                                    for c in 0..q_row {
                                        let rowvec = local_jac_row.row(c).to_owned();
                                        let mr = metric.apply_metric_row(row, rowvec.view());
                                        for j in 0..p {
                                            w[[c, j]] = mr[j];
                                        }
                                    }
                                    Some(w)
                                } else {
                                    None
                                };
                                let jac_rows: ArrayView2<'_, f64> = match &ljr_white {
                                    Some(w_jac) => w_jac.view(),
                                    None => local_jac_row.view(),
                                };
                                for &atom_idx in row_active {
                                    let atom = &self.atoms[atom_idx];
                                    let m = atom.basis_size();
                                    let a_k = assignments[atom_idx];
                                    // The frame projection of the Jacobian rows is
                                    // basis-column independent: hoist it to ONE
                                    // `q×p · p×r` GEMM per (row, atom) and reduce the
                                    // basis scan to rank-length axpys, instead of
                                    // re-deriving it through m·q·p scalar
                                    // `accumulate_output_project` calls (a top-two
                                    // profile cost of the joint fit).
                                    let projected =
                                        frame_projection.project_jacobian_rows(atom_idx, jac_rows);
                                    let rank = frame_projection.ranks[atom_idx];
                                    for basis_col in 0..m {
                                        let phi = atom.basis_values[[row, basis_col]];
                                        let w = a_k * phi * sqrt_row_w;
                                        if w == 0.0 {
                                            continue;
                                        }
                                        let c_base = frame_projection.border_offsets[atom_idx]
                                            + basis_col * rank;
                                        for c in 0..q_row {
                                            let mut hrow = block.htbeta.row_mut(c);
                                            let hrow_slice = hrow
                                                .as_slice_mut()
                                                .expect("htbeta row is contiguous");
                                            match &projected {
                                                Some(proj) => {
                                                    let dst =
                                                        &mut hrow_slice[c_base..c_base + rank];
                                                    for (slot, &value) in
                                                        dst.iter_mut().zip(proj.row(c).iter())
                                                    {
                                                        *slot += w * value;
                                                    }
                                                }
                                                // Unframed atom: identity frame, the
                                                // border block IS the raw output row.
                                                None => {
                                                    let dst = &mut hrow_slice[c_base..c_base + p];
                                                    for (slot, &value) in
                                                        dst.iter_mut().zip(jac_rows.row(c).iter())
                                                    {
                                                        *slot += w * value;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Data-fit GN β-Hessian: accumulate the channel-independent block
                            // `G[μ_i, μ_j] += (a_k φ_k)[μ_i] (a_{k'} φ_{k'})[μ_j]` into the
                            // sparse per-atom-pair map (the `out_col` dimension is carried by
                            // `I_p`). Only co-occurring `(atom_i, atom_j)` pairs are touched.
                            for ai in 0..weighted_phi.len() {
                                let (atom_i, ref wphi_i) = weighted_phi[ai];
                                let m_i = wphi_i.len();
                                for aj in 0..weighted_phi.len() {
                                    let (atom_j, ref wphi_j) = weighted_phi[aj];
                                    let m_j = wphi_j.len();
                                    let blk = g_blocks
                                        .entry((atom_i, atom_j))
                                        .or_insert_with(|| Array2::<f64>::zeros((m_i, m_j)));
                                    for li in 0..m_i {
                                        let wi = wphi_i[li];
                                        if wi == 0.0 {
                                            continue;
                                        }
                                        for lj in 0..m_j {
                                            blk[[li, lj]] += wi * wphi_j[lj];
                                        }
                                    }
                                }
                            }
                        } // #1407 end `if !fixed_decoder` β-tier accumulation
                        let (kron_a_phi, kron_jac) = if !frames_engaged && !fixed_decoder {
                            // Flatten local_jac_row row-major into a plain Vec<f64> (q_row * p entries).
                            let mut jac_flat = vec![0.0_f64; q_row * p];
                            for c in 0..q_row {
                                for j in 0..p {
                                    jac_flat[c * p + j] = local_jac_row[[c, j]];
                                }
                            }
                            (Some(a_phi), Some(jac_flat))
                        } else {
                            (None, None)
                        };
                        // #974 whitened-frames support: flatten `weighted_phi`
                        // (atom, per-basis a·φ) into `(atom, basis, weight)` for
                        // the per-row `Φ_nᵀ M_n Φ_n` operator. Built only on the
                        // frames+whitening path (else `None`, never allocated).
                        let frame_support = if frames_engaged && whitens_likelihood && !fixed_decoder
                        {
                            let mut sup: Vec<(usize, usize, f64)> =
                                Vec::with_capacity(weighted_phi.iter().map(|(_, w)| w.len()).sum());
                            for (atom_idx, wphi) in weighted_phi.iter() {
                                for (basis_col, &w) in wphi.iter().enumerate() {
                                    if w != 0.0 {
                                        sup.push((*atom_idx, basis_col, w));
                                    }
                                }
                            }
                            Some(sup)
                        } else {
                            None
                        };
                        Ok(SaeAssemblyRow {
                            row,
                            block,
                            gb_delta,
                            g_blocks,
                            kron_a_phi,
                            kron_jac,
                            frame_support,
                        })
                        }) // #1557 with_nested_parallel
                    },
                )
                .collect::<Result<Vec<_>, String>>()?;

            // Fold THIS chunk's rows (ascending) into the global accumulators.
            // The parallel collect preserves index order within the chunk and
            // chunks are visited in ascending `chunk_start` order, so the overall
            // fold order is `0,1,2,…,n-1` — identical to the former single-pass
            // fold. The `row == chunk_start + fold_offset_in_chunk` assert pins
            // that strict sequential arrival (the invariant the `kron_*`
            // row-aligned pushes depend on).
            for row_result in row_results.into_iter() {
                let row = row_result.row;
                assert_eq!(
                    row,
                    chunk_start + fold_offset_in_chunk,
                    "parallel SAE row assembly returned rows out of order"
                );
                fold_offset_in_chunk += 1;
                for (idx, value) in row_result.gb_delta {
                    sys.gb[idx] += value;
                }
                for ((atom_i, atom_j), data) in row_result.g_blocks {
                    let m_i = data.nrows();
                    let m_j = data.ncols();
                    let blk = g_blocks
                        .entry((atom_i, atom_j))
                        .or_insert_with(|| Array2::<f64>::zeros((m_i, m_j)));
                    for li in 0..m_i {
                        for lj in 0..m_j {
                            blk[[li, lj]] += data[[li, lj]];
                        }
                    }
                }
                if !frames_engaged && !fixed_decoder {
                    // Rows arrive in ascending order across chunks, so pushing
                    // here yields `kron_*[row]` aligned to the row index exactly
                    // as the single-pass `push` did.
                    kron_a_phi.push(
                        row_result
                            .kron_a_phi
                            .expect("full-B SAE row assembly must return a_phi rows"),
                    );
                    kron_jac.push(
                        row_result
                            .kron_jac
                            .expect("full-B SAE row assembly must return local Jacobian rows"),
                    );
                }
                if let Some(sup) = row_result.frame_support {
                    // Ascending row arrival ⇒ `frame_support[row]` aligns to `row`.
                    frame_support_rows.push(sup);
                }
                sys.rows[row] = row_result.block;
            }
            chunk_start = chunk_end;
        }
        // #1407: fixed-decoder early return. The per-row htt/gt are now fully
        // assembled (data GN + assignment/ARD prior). Apply only the htt/gt
        // Riemannian projection (the decoder/β tier is intentionally absent), then
        // return the block-diagonal system. `fixed_decoder_step_from_rows` reads
        // only `rows[*].htt`/`gt` + `row_offsets`, so no β-tier object is needed.
        if fixed_decoder {
            match row_layout.as_ref() {
                None => {
                    // Dense uniform-q: project htt/gt (and the 0-width htbeta, a
                    // no-op) through the ext-coord manifold.
                    self.apply_sae_riemannian_geometry(&mut sys);
                }
                Some(layout) => {
                    // Compact heterogeneous-q: project each row's htt/gt at its
                    // own ext-coord point, mirroring the full path's compact
                    // Riemannian block (htbeta is 0-width here, so skipped).
                    if !self.ext_coord_manifold().is_euclidean() {
                        // Each row rebuilds its own compact ext-manifold from
                        // immutable `&self`/`layout` and writes ONLY its own
                        // `sys.rows[row_idx].{gt,htt}` (disjoint), so the row-parallel
                        // path is bit-identical to the serial sweep.
                        let this = &*self;
                        let project_fixed_row = |row_idx: usize, row: &mut ArrowRowBlock| {
                            let (manifold_i, point_i) =
                                this.compact_row_ext_manifold_and_point(row_idx, layout);
                            let t_i = point_i.view();
                            let gt_e = row.gt.clone();
                            let htt_e = row.htt.clone();
                            row.gt = manifold_i.project_gradient_to_tangent(t_i, gt_e.view());
                            row.htt = manifold_i.riemannian_hessian_matrix(
                                t_i,
                                gt_e.view(),
                                htt_e.view(),
                            );
                        };
                        let parallel = n >= SAE_LOSS_PARALLEL_ROW_MIN
                            && rayon::current_thread_index().is_none();
                        if parallel {
                            use rayon::prelude::*;
                            // #1557 — pin the projector's faer GEMM to Par::Seq.
                            sys.rows
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(row_idx, row)| {
                                    with_nested_parallel(|| project_fixed_row(row_idx, row));
                                });
                        } else {
                            for row_idx in 0..n {
                                let row = &mut sys.rows[row_idx];
                                project_fixed_row(row_idx, row);
                            }
                        }
                    }
                }
            }
            if let Some(deflation) = self
                .row_gauge_deflation_for_layout(row_layout.as_ref())
                .or_else(|| {
                    // #974 — see the main-path site: enable spectral discovery of
                    // the rank-deficient-metric-null `H_tt` directions on the
                    // fixed-decoder path too. No-op when the metric is full-rank.
                    low_rank_whiten.then(|| Self::empty_row_gauge_deflation(n))
                })
            {
                sys.set_row_gauge_deflation(deflation);
            }
            self.last_row_layout = row_layout;
            self.last_frames_active = frames_engaged;
            return Ok(sys);
        }
        // Apply Riemannian geometry to the per-row row blocks (htt, gt) and
        // also to the per-row Kronecker local Jacobians stored in kron_jac.
        // When the SAE ext-coord manifold is non-Euclidean (any atom latent
        // on sphere / circle / interval), the local Jacobian rows that map
        // into the t-block tangent space must be projected via the per-row
        // tangent projector P_i.  This mirrors what
        // `apply_riemannian_latent_geometry` does to `row.htbeta`, applied
        // here to the (q × p) kron_jac so the Kronecker htbeta_matvec uses
        // the Riemannian-projected form.
        // Dense rows share one product manifold. Exact TopK rows may have
        // heterogeneous supports, so the compact arm rebuilds the corresponding
        // per-row product manifold before applying the same geometry.
        match row_layout.as_ref() {
            None => {
                let raw_gt_rows: Vec<Array1<f64>> =
                    sys.rows.iter().map(|row| row.gt.clone()).collect();
                self.apply_sae_riemannian_geometry(&mut sys);
                let manifold = self.ext_coord_manifold();
                if !frames_engaged && !manifold.is_euclidean() {
                    let ext = self.ext_coord_matrix();
                    // Project the local Jacobian columns onto the tangent space at
                    // each row's ext-coord point. Each column `j` of the row's
                    // (q_row × p) Jacobian is an ambient-space vector of length
                    // `q_row`; the manifold projector acts on one such column at a
                    // time. Working directly on the row-major `jac_flat` storage via
                    // a single reusable `col_buf` avoids the two dense (q × p) copies
                    // (flatten→Array2, project, unflatten→Vec) that previously fired
                    // per row. `t_buf` still holds the row's ext-coord vector.
                    // Per-row Jacobian column projection: each row writes ONLY its own
                    // `kron_jac[row_idx]` and reads immutable `ext`/`raw_gt_rows`/`manifold`,
                    // so the row-parallel path is bit-identical to the serial sweep
                    // (disjoint-writes determinism). Per-row `t_buf`/`col_buf` scratch.
                    let project_row =
                        |row_idx: usize,
                         jac_flat: &mut Vec<f64>,
                         t_buf: &mut [f64],
                         col_buf: &mut Array1<f64>| {
                            let ext_row = ext.row(row_idx);
                            for (slot, &v) in t_buf.iter_mut().zip(ext_row.iter()) {
                                *slot = v;
                            }
                            let t_i = ArrayView1::from(&*t_buf);
                            let raw_gt = raw_gt_rows[row_idx].view();
                            let q_row = jac_flat.len() / p;
                            for j in 0..p {
                                for c in 0..q_row {
                                    col_buf[c] = jac_flat[c * p + j];
                                }
                                let projected_col = manifold.project_vector_to_gradient_tangent(
                                    t_i,
                                    raw_gt.slice(ndarray::s![..q_row]),
                                    col_buf.slice(ndarray::s![..q_row]),
                                );
                                for c in 0..q_row {
                                    jac_flat[c * p + j] = projected_col[c];
                                }
                            }
                        };
                    let parallel =
                        n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
                    if parallel {
                        use rayon::prelude::*;
                        // #1557 — pin the projector's faer GEMM to Par::Seq.
                        kron_jac
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(row_idx, jac_flat)| {
                                with_nested_parallel(|| {
                                    let mut t_buf = vec![0.0_f64; q];
                                    let mut col_buf = Array1::<f64>::zeros(q);
                                    project_row(row_idx, jac_flat, &mut t_buf, &mut col_buf);
                                });
                            });
                    } else {
                        let mut t_buf = vec![0.0_f64; q];
                        let mut col_buf = Array1::<f64>::zeros(q);
                        for row_idx in 0..n {
                            project_row(row_idx, &mut kron_jac[row_idx], &mut t_buf, &mut col_buf);
                        }
                    }
                }
            }
            Some(layout) => {
                // Compact active-set layout (#1117 follow-up): the dense
                // `ext_coord_manifold()` is keyed to the uniform full-`q` block
                // ordering, so it cannot be applied to the heterogeneous compact
                // rows directly. Instead we rebuild, PER ROW, the product manifold
                // and ext-coord point in that row's compact column order (see
                // `compact_row_ext_manifold_and_point`) and apply the SAME three
                // per-row Riemannian operations the dense
                // `apply_riemannian_latent_geometry` applies — gradient tangent
                // projection of `gt`, the Riemannian Hessian correction of `htt`,
                // and the column tangent projection of `htbeta` — plus the
                // identical Kronecker `kron_jac` column projection. On the shared
                // active support this is byte-identical to slicing the dense
                // product manifold.
                //
                // Euclidean ext manifolds still skip all of this (every
                // per-row manifold is a product of Euclidean parts whose
                // projector is the identity); we early-out so those rows stay
                // byte-for-byte the historical compact path.
                if !self.ext_coord_manifold().is_euclidean() {
                    // Each row rebuilds its own compact ext-manifold from immutable
                    // `&self`/`layout` and writes ONLY its own `sys.rows[row_idx]` (and,
                    // on the matrix-free path, its own `kron_jac[row_idx]`) — both disjoint
                    // per row — so the row-parallel path is bit-identical to the serial
                    // sweep. The frames path touches `htbeta` (no `kron_jac`); the
                    // matrix-free path touches `kron_jac` (0-width `htbeta` skipped), so
                    // the two arms parallelize over exactly the live Vec(s).
                    let this = &*self;
                    let parallel =
                        n >= SAE_LOSS_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
                    // gt / htt projection shared by both arms, exactly as
                    // `apply_riemannian_latent_geometry` does for dense uniform-q rows.
                    // Returns the row's `(manifold_i, point_i, gt_e)` so the caller can
                    // apply the htbeta / kron_jac leg with the SAME pre-projection state.
                    let project_gt_htt =
                        |row_idx: usize,
                         row: &mut ArrowRowBlock|
                         -> (LatentManifold, Array1<f64>, Array1<f64>) {
                            let (manifold_i, point_i) =
                                this.compact_row_ext_manifold_and_point(row_idx, layout);
                            let t_i = point_i.view();
                            let gt_e = row.gt.clone();
                            let htt_e = row.htt.clone();
                            row.gt = manifold_i.project_gradient_to_tangent(t_i, gt_e.view());
                            row.htt = manifold_i.riemannian_hessian_matrix(
                                t_i,
                                gt_e.view(),
                                htt_e.view(),
                            );
                            (manifold_i, point_i, gt_e)
                        };
                    // Frames arm: `htbeta` column projection with the SAME pre-projection
                    // gradient `gt_e`.
                    let frames_row = |row_idx: usize, row: &mut ArrowRowBlock| {
                        let (manifold_i, point_i, gt_e) = project_gt_htt(row_idx, row);
                        let t_i = point_i.view();
                        let htbeta_e = row.htbeta.clone();
                        row.htbeta = manifold_i.project_matrix_columns_to_gradient_tangent(
                            t_i,
                            gt_e.view(),
                            htbeta_e.view(),
                        );
                    };
                    // Matrix-free arm: Kronecker local-Jacobian column projection with the
                    // SAME pre-projection gradient `gt_e` so the cross-block geometry
                    // matches the dense branch.
                    let matrix_free_row =
                        |row_idx: usize, row: &mut ArrowRowBlock, jac_flat: &mut Vec<f64>| {
                            let (manifold_i, point_i, gt_e) = project_gt_htt(row_idx, row);
                            let t_i = point_i.view();
                            let q_row = jac_flat.len() / p;
                            let mut col_buf = Array1::<f64>::zeros(q_row);
                            for j in 0..p {
                                for c in 0..q_row {
                                    col_buf[c] = jac_flat[c * p + j];
                                }
                                let projected_col = manifold_i.project_vector_to_gradient_tangent(
                                    t_i,
                                    gt_e.view(),
                                    col_buf.view(),
                                );
                                for c in 0..q_row {
                                    jac_flat[c * p + j] = projected_col[c];
                                }
                            }
                        };
                    if frames_engaged {
                        if parallel {
                            use rayon::prelude::*;
                            // #1557 — pin the projector's faer GEMM to Par::Seq.
                            sys.rows
                                .par_iter_mut()
                                .enumerate()
                                .for_each(|(row_idx, row)| {
                                    with_nested_parallel(|| frames_row(row_idx, row));
                                });
                        } else {
                            for row_idx in 0..n {
                                frames_row(row_idx, &mut sys.rows[row_idx]);
                            }
                        }
                    } else if parallel {
                        use rayon::prelude::*;
                        // Disjoint per-row writes to BOTH `sys.rows` and `kron_jac`;
                        // zip the two indexed parallel iterators so each worker owns one
                        // aligned `(row, jac_flat)` pair. #1557 GEMM guard as above.
                        sys.rows
                            .par_iter_mut()
                            .zip(kron_jac.par_iter_mut())
                            .enumerate()
                            .for_each(|(row_idx, (row, jac_flat))| {
                                with_nested_parallel(|| matrix_free_row(row_idx, row, jac_flat));
                            });
                    } else {
                        for row_idx in 0..n {
                            matrix_free_row(
                                row_idx,
                                &mut sys.rows[row_idx],
                                &mut kron_jac[row_idx],
                            );
                        }
                    }
                }
            }
        }
        // Build and install the full-B Kronecker htbeta_matvec.
        //
        // `SaeKroneckerRows` holds per-row `(a_phi, local_jac)` and implements
        // the cross-block operator without ever materialising the dense
        // `(q × K·p)` slab.  The cross-block factorises as `H_tβ = L · J_β`,
        // where `J_β = φᵀ ⊗ I_p` projects a length-`K` β vector onto the
        // `p`-dimensional decoded output space (`apply_jbeta`) and `L_i` is
        // the per-row `(q_i × p)` assignment+coordinate Jacobian that lifts
        // that p-vector into the row's `q_i`-dim tangent block (`apply_l`).
        // Both factors are required: the contract of `set_row_htbeta_operator`
        // is `out.len() == d` (= `q_i`), so writing `apply_jbeta`'s p-vector
        // output directly into a length-`q_i` buffer overflows whenever
        // `p > q_i` (the common case once `p` reflects real feature width).
        // Symmetric for the transpose: `H_βt = J_βᵀ · Lᵀ`, so apply `Lᵀ`
        // first to map the q_i-vector back to p-space, then scatter through
        // the support.
        // #1017/#1026: the legacy full-B device PCG assumes `G ⊗ I_p`, while
        // framed systems carry `G_ij ⊗ W_ij` with rank-r atom blocks. Feeding a
        // framed system to that kernel would silently return the wrong Newton
        // step. Framed device PCG therefore needs the dedicated factored kernel.
        // #1033 large-n: the per-row support `kron_a_phi` and local Jacobians
        // `kron_jac` are consumed by BOTH the host matrix-free row operator
        // (`SaeKroneckerRows`) and the solver's `DeviceSaePcgData`. Previously
        // each took its own full `O(n·q·p)` / `O(n·k_active)` clone, so the
        // always-resident footprint of the CPU non-frames path carried TWO copies
        // of the dominant Jacobian slab. Promote each to a single `Arc<[…]>` once
        // and hand both consumers a refcount bump (`O(1)`) — the backing
        // allocation is shared, halving the resident per-row Jacobian memory.
        // Reads are identical (`&arc[row]`, `.len()`), so the assembled system and
        // every matvec are bit-for-bit unchanged.
        let device_rows = if frames_engaged {
            None
        } else {
            let a_phi_shared: Arc<[Vec<(usize, f64)>]> =
                Arc::from(std::mem::take(&mut kron_a_phi).into_boxed_slice());
            let jac_shared: Arc<[Vec<f64>]> =
                Arc::from(std::mem::take(&mut kron_jac).into_boxed_slice());
            Some((a_phi_shared, jac_shared))
        };
        // #974 likelihood-whitening: the per-row output metric `M_n = U_n U_nᵀ`
        // installed when the fit whitens the reconstruction likelihood
        // (`BehavioralFisher` / `WhitenedStructured`). Threaded into the
        // matrix-free cross-block and β-Gram operators so they carry the SAME
        // metric the residual/gradient (`error_metric`) and the t-block
        // (`jac_white`, `htt = J M Jᵀ`) already apply — closing the isotropic
        // `G ⊗ I_p` / raw-`L` Hessian gap. `None` on the isotropic path, where
        // every operator apply stays bit-for-bit the historical path.
        let output_metric: Option<gam_problem::RowMetric> = if whitens_likelihood {
            self.row_metric.clone()
        } else {
            None
        };
        // Hoisted so the whitening branch of the β-tier install below can build
        // the whitened β-Gram operator from the SAME `SaeKroneckerRows`
        // (support + metric) the cross-block operator uses.
        let mut whitened_gram_kron: Option<Arc<SaeKroneckerRows>> = None;
        if !frames_engaged {
            let (a_phi_shared, jac_shared) = device_rows
                .clone()
                .expect("non-frames path always populates device_rows");
            let kron = Arc::new(
                SaeKroneckerRows::new(p, a_phi_shared, jac_shared)
                    .with_output_metric(output_metric.clone()),
            );
            if whitens_likelihood {
                whitened_gram_kron = Some(Arc::clone(&kron));
            }
            let kron_t = Arc::clone(&kron);
            let p_dim = p;
            sys.set_row_htbeta_operator(
                move |row_idx, x, out| {
                    // out = L_i · M_n · (J_β · x). Allocate a length-p scratch
                    // buffer for the intermediate decoded-output vector; both
                    // factors overwrite their output buffers (`apply_jbeta` zeroes
                    // before accumulating, `apply_l` writes per-row), so no
                    // pre-zeroing of `u_p`/`out` is needed. #974: the metric
                    // `M_n` is applied to the p-space intermediate — a no-op on
                    // the isotropic path (`M_n = I_p`), giving the exact whitened
                    // cross-block `H_tβ = L_i M_n J_β` where whitening is active.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(xs) = x.as_slice() {
                        kron.apply_jbeta(row_idx, xs, &mut u_p);
                    } else {
                        let x_vec: Vec<f64> = x.iter().copied().collect();
                        kron.apply_jbeta(row_idx, &x_vec, &mut u_p);
                    }
                    kron.apply_output_metric_row(row_idx, &mut u_p);
                    kron.apply_l(row_idx, &u_p, out_slice);
                },
                move |row_idx, v, out| {
                    // out += J_βᵀ · M_n · (Lᵀ · v). `apply_l_t` accumulates into a
                    // zero-initialised length-p buffer to produce the p-vector
                    // `Lᵀ v`; #974 applies the (symmetric) metric `M_n` to it —
                    // a no-op isotropically — so the transpose is
                    // `H_βt = J_βᵀ M_n Lᵀ = H_tβᵀ`; `scatter_jbeta_t` then adds
                    // φ_i[s] · (M_n Lᵀ v)[j] into the length-K β accumulator at
                    // each active `(s, j)`.
                    let out_slice = out.as_slice_mut().expect("out is always standard-layout");
                    let mut u_p = vec![0.0_f64; p_dim];
                    if let Some(vs) = v.as_slice() {
                        kron_t.apply_l_t(row_idx, vs, &mut u_p);
                    } else {
                        let v_vec: Vec<f64> = v.iter().copied().collect();
                        kron_t.apply_l_t(row_idx, &v_vec, &mut u_p);
                    }
                    kron_t.apply_output_metric_row(row_idx, &mut u_p);
                    kron_t.scatter_jbeta_t(row_idx, &u_p, out_slice);
                },
            );
        }
        let mut beta_penalty_assembly = SaeBetaPenaltyAssembly::default();
        let factored_row_projection = if frames_engaged && analytic_penalties.is_some() {
            Some(&frame_projection)
        } else {
            None
        };
        if let Some(registry) = analytic_penalties {
            // Upfront validation: refuse penalty kinds the SAE row layout
            // cannot host, and refuse mixed-d row-block configurations.
            // This makes the dispatch loop below total — no runtime
            // "unsupported penalty" fallthrough, no K-gating.
            self.validate_analytic_penalty_registry(registry)
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
            beta_penalty_assembly = self
                .add_sae_analytic_penalty_contributions(
                    &mut sys,
                    registry,
                    penalty_scale,
                    row_layout.as_ref(),
                    dense_beta_curvature,
                    factored_row_projection,
                )
                .map_err(|err| format!("SaeManifoldTerm::assemble_arrow_schur: {err}"))?;
        }
        // #1026 — decoder repulsion (collinearity-gated, registry-independent):
        // accumulate into the full-`B` β-tier here, BEFORE the frame transform,
        // so a framed system carries it identically to the analytic β penalties.
        // No-op unless two atoms are near-collinear (the frozen gate is `None`).
        if self.add_sae_decoder_repulsion(&mut sys, penalty_scale, dense_beta_curvature) {
            beta_penalty_assembly.record_curvature(dense_beta_curvature);
        }
        // #1026/#1522 — interior-point collapse-prevention barriers. The amplitude
        // barrier supplies the OUTWARD radial force at the zero-decoder collapse
        // point (the principal failure state the threshold repulsion skips), and
        // the separation barrier supplies the alignment-divergent separating
        // curvature on normalized shapes weighted by coactivation. Both accumulate
        // into the full-`B` β-tier here, BEFORE the frame transform, so a framed
        // system carries them identically to the analytic β penalties.
        // #1610 — on the dense path the barrier's Levenberg majorizer scatters
        // onto `sys.hbb`; on the matrix-free / framed production path `sys.hbb` is
        // unused, so the barrier hands back a per-atom scalar ridge which we fold
        // into `smooth_scaled_s` (the single source for the CPU composite penalty
        // op AND the device smooth blocks), restoring the collapse-prevention
        // curvature the operator was silently dropping there.
        let mut sep_atom_curv = vec![0.0_f64; self.atoms.len()];
        // #1038 — full-`B` self-concordant rank-1 carriers `(d2, ∂o/∂B)` the barrier
        // hands back on the matrix-free / framed path (empty on the dense path, which
        // scatters the rank-1 straight into `hbb`). Installed as `SparseRankOnePenaltyOp`
        // on the structured penalty op below, projected to factored coords when framed.
        let mut sep_rank1: Vec<(f64, Vec<(usize, f64)>)> = Vec::new();
        if self.add_sae_separation_barrier(
            &mut sys,
            penalty_scale,
            dense_beta_curvature,
            &mut sep_atom_curv,
            &mut sep_rank1,
        ) {
            if dense_beta_curvature {
                beta_penalty_assembly.record_curvature(true);
            } else {
                // Fold the per-atom majorizer `lev_k·I_{M_k}` into the smooth
                // penalty factor `λ S_k`. With `⊗ I_p` (full-`B`) or `⊗ I_{r_k}`
                // (factored, `U_kᵀU_k = I`) this is exactly the `lev_k·I` block
                // diagonal the dense path writes — and it now flows through the
                // structured penalty op and the device smooth blocks. No
                // `deferred_factored` mark: the curvature is in the smooth op, not
                // a deferred dense block, so the device path stays engaged.
                for atom_idx in 0..self.atoms.len() {
                    let c = sep_atom_curv[atom_idx];
                    if c > 0.0 {
                        let m = smooth_scaled_s[atom_idx].nrows();
                        for i in 0..m {
                            smooth_scaled_s[atom_idx][[i, i]] += c;
                        }
                        smooth_ops[atom_idx] = Arc::new(IdentityRightKroneckerPenaltyOp {
                            factor_a: smooth_scaled_s[atom_idx].clone(),
                            p,
                            global_offset: beta_offsets[atom_idx],
                            k: beta_dim,
                        });
                    }
                }
            }
        }
        if frames_engaged {
            // ── #972 / #977 T1 — FACTORED β-tier transform ──────────────────
            //
            // The entire β-tier above was assembled in the full-`B` (p-wide)
            // layout: `sys.gb` is `g_B` (length `beta_dim`), `sys.hbb` carries
            // any analytic Beta-tier penalty, and `g_blocks` is the
            // FRAME-INDEPENDENT basis Gram. We now rebuild the β-tier in the
            // factored coordinate space `C` (width `factored_border_dim`), the
            // full-`B` system sandwiched by `Φ = blkdiag(I_{M_k} ⊗ U_k)`:
            //   * gradient   `g_C = Φᵀ g_B`              (per atom `(g_B U_k)`),
            //   * data H      `Φᵀ(G⊗I_p)Φ = G_{ij}⊗(U_iᵀU_j)`,
            //   * smooth      `λ S_k ⊗ I_{r_k}`          (since `U_kᵀU_k = I`),
            //   * analytic    `Φᵀ hbb Φ`                 (dense, only if written).
            // Un-framed atoms ride the `r_k = p, U_k = I_p` identity special case.
            let off_c = &frame_projection.border_offsets;
            let ranks = &frame_projection.ranks;
            let basis_sizes = &frame_projection.basis_sizes;
            let border_dim = frame_projection.border_dim();
            let gb_c = frame_projection.project_border_vec(sys.gb.view());

            // Data β-Hessian: `G_{ij} ⊗ W_{ij}` with `W_{ij} = U_iᵀU_j`. The
            // basis Gram `g_blocks` is unchanged; only the output factor is the
            // per-pair frame overlap (`I_{r_k}` within a framed atom, `I_p` for
            // un-framed).
            // #974: under a likelihood-whitening metric the separable
            // `G_{ij} ⊗ (U_iᵀU_j)` is WRONG (the per-row `U_iᵀ M_n U_j` does not
            // factor out of the basis Gram). Build the exact per-row sandwich
            // `Σ_n Φ_nᵀ M_n Φ_n` ([`WhitenedFactoredFrameOp`]) instead, and drop
            // the device frame blocks (the device PCG kernel assumes the isotropic
            // frame Gram → CPU fallback). Off the whitening path the separable
            // isotropic operator is built bit-for-bit as before.
            let (data_op, device_frame_blocks): (
                Arc<dyn BetaPenaltyOp>,
                Option<Vec<FactoredFrameGBlock>>,
            ) = if whitens_likelihood {
                let metric = self
                    .row_metric
                    .clone()
                    .expect("whitens_likelihood ⇒ metric present");
                let support: Arc<[Vec<(usize, usize, f64)>]> =
                    Arc::from(std::mem::take(&mut frame_support_rows).into_boxed_slice());
                let wop = WhitenedFactoredFrameOp::new(
                    p,
                    border_dim,
                    off_c.clone(),
                    ranks.clone(),
                    basis_sizes.clone(),
                    frame_projection.frames_owned(),
                    support,
                    metric,
                );
                (Arc::new(wop), None)
            } else {
                let mut frame_blocks: Vec<FactoredFrameGBlock> = Vec::with_capacity(g_blocks.len());
                for ((atom_i, atom_j), data) in g_blocks.into_iter() {
                    if data.iter().all(|&v| v == 0.0) {
                        continue;
                    }
                    // `W_{ij} = U_iᵀ U_j` from the precomputed per-atom frames.
                    let w = self.frame_cross_factor(atom_i, atom_j);
                    frame_blocks.push(FactoredFrameGBlock {
                        atom_i,
                        atom_j,
                        g: data,
                        w,
                    });
                }
                // #1017/#1026 — snapshot the factored data-fit blocks for the
                // frames-engaged device PCG BEFORE `FactoredFrameKroneckerOp::new`
                // consumes them. Cheap clone (co-occurring blocks only).
                let device_frame_blocks = frame_blocks.clone();
                let op = FactoredFrameKroneckerOp::new(
                    ranks.clone(),
                    basis_sizes.clone(),
                    frame_blocks,
                )?;
                (
                    Arc::new(op) as Arc<dyn BetaPenaltyOp>,
                    Some(device_frame_blocks),
                )
            };

            // Smooth penalty in factored space: `λ S_k ⊗ I_{r_k}` at `off_C[k]`.
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(self.atoms.len() + 2);
            for k in 0..self.atoms.len() {
                let r = ranks[k];
                ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
                    factor_a: smooth_scaled_s[k].clone(),
                    p: r,
                    global_offset: off_c[k],
                    k: border_dim,
                }));
            }
            ops.push(data_op);
            // Analytic Beta-tier penalty: project the dense full-`B` `hbb` block
            // `Φᵀ hbb Φ` into the factored space. Only present when a Beta-tier
            // penalty actually wrote `hbb` (else `hbb` is all-zero and the dense
            // `(border_dim)²` op is skipped entirely, exactly as full-`B`).
            if beta_penalty_assembly.dense_written {
                let hbb_c =
                    self.project_dense_penalty_to_factored(sys.hbb.view(), &frame_projection);
                ops.push(Arc::new(DensePenaltyOp(hbb_c)));
            } else if beta_penalty_assembly.deferred_factored {
                // Registry Beta-tier curvature deferred to factored-space probing.
                // The registry may be absent when `deferred_factored` was set ONLY
                // by the frozen-gate decoder repulsion (which is
                // registry-independent), so start from a zero factored block in
                // that case instead of unwrapping.
                let mut hbb_c = match analytic_penalties {
                    Some(registry) => self.build_factored_beta_penalty_curvature(
                        registry,
                        penalty_scale,
                        &frame_projection,
                    ),
                    None => Array2::<f64>::zeros((
                        frame_projection.border_dim(),
                        frame_projection.border_dim(),
                    )),
                };
                // #1610 — the frozen-gate decoder repulsion's PSD majorizer was
                // dropped on this matrix-free/framed path (only its gradient was
                // applied). Project it into the factored block via the same
                // `psd_majorizer_hvp` + frame-projection probe pattern the registry
                // DecoderIncoherence uses, so the collapse-prevention curvature
                // reaches the operator here too. No-op when no repulsion is active.
                self.add_factored_repulsion_curvature(&mut hbb_c, penalty_scale, &frame_projection);
                ops.push(Arc::new(DensePenaltyOp(hbb_c)));
            }

            // Re-point the system's β-tier to the factored width. The t-tier
            // (per-row `htt`, `gt`) is frame-independent and untouched; row
            // cross-block slabs were allocated and assembled directly in
            // factored coordinates, so analytic row supplements and data-fit
            // cross terms already share shape `(q_i × factored_border_dim)`.
            sys.k = border_dim;
            sys.gb = gb_c;
            self.reclaim_border_hbb_workspace(&mut sys);
            // Factored per-atom block ranges for the block-Jacobi Schur
            // preconditioner: `[off_C[k] .. off_C[k] + M_k·r_k]`.
            let mut block_ranges: Vec<std::ops::Range<usize>> =
                Vec::with_capacity(self.atoms.len());
            for k in 0..self.atoms.len() {
                let start = off_c[k];
                block_ranges.push(start..start + basis_sizes[k] * ranks[k]);
            }
            sys.set_block_offsets(Arc::from(block_ranges.into_boxed_slice()));
            // #1038 — install the barrier's exact self-concordant rank-1 curvature
            // in FACTORED coords. A full-`B` rank-1 `v vᵀ` projects to `(Φᵀv)(Φᵀv)ᵀ`
            // (still rank-1 since `Φ` is linear), so project each carrier `v` through
            // `project_border_vec` (= `Φᵀ`) and add a `SparseRankOnePenaltyOp`. This
            // is the curvature the scalar `smooth_scaled_s` ridge cannot represent.
            for (scale, carrier) in &sep_rank1 {
                let mut full = ndarray::Array1::<f64>::zeros(frame_projection.beta_dim());
                for &(idx, v) in carrier {
                    full[idx] += v;
                }
                let vc = frame_projection.project_border_vec(full.view());
                let sparse: Vec<(usize, f64)> = (0..vc.len())
                    .filter(|&i| vc[i] != 0.0)
                    .map(|i| (i, vc[i]))
                    .collect();
                if !sparse.is_empty() {
                    ops.push(Arc::new(SparseRankOnePenaltyOp {
                        k: border_dim,
                        scale: *scale,
                        carrier: sparse,
                    }));
                }
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: border_dim, ops }));
            // #1017/#1026 — install the frames-engaged device SAE PCG data. Skipped
            // (CPU fallback) when a dense analytic Beta-tier penalty fired (the
            // device kernel does not model that extra dense term). Builder:
            // `crate::frames::build_framed_device_sae_data`.
            // #1038 — also skip (CPU fallback) when the barrier installed a rank-1
            // curvature carrier: the device kernel folds only the per-atom scalar
            // smooth blocks, so it would silently drop the cross-atom rank-1 and
            // diverge from the CPU operator. The rank-1 fires only on a genuinely
            // co-collapsing dictionary (gated), so healthy fits keep the device path.
            let has_dense_beta_penalty = beta_penalty_assembly.dense_written
                || beta_penalty_assembly.deferred_factored
                || !sep_rank1.is_empty();
            // #974: `device_frame_blocks` is `None` on the whitening path (the
            // device kernel assumes the isotropic frame Gram), forcing the CPU
            // reduced-Schur matvec which routes `H_ββ` through the metric-aware
            // `WhitenedFactoredFrameOp` and `H_tβ` through the whitened `htbeta`
            // slab. On the isotropic path it is `Some`, keeping the device PCG.
            if !has_dense_beta_penalty {
                if let Some(device_frame_blocks) = device_frame_blocks {
                    let recycled = self.arrow_assembly_workspace.device_sae_pcg.take();
                    let device = crate::frames::build_framed_device_sae_data_reusing(
                        crate::frames::FramedDeviceArgs {
                            p,
                            border_dim,
                            border_offsets: off_c.as_slice(),
                            ranks: ranks.as_slice(),
                            basis_sizes: basis_sizes.as_slice(),
                            smooth_scaled_s: &smooth_scaled_s,
                            frame_blocks: device_frame_blocks,
                            rows: &sys.rows,
                        },
                        recycled,
                    );
                    sys.set_device_sae_pcg_allocation(device);
                }
            }
        } else if whitens_likelihood {
            // #974 whitening (non-frames): the collapsed `G ⊗ I_p` factorization
            // is invalid because the data-fit GN β-Hessian is `Σ_n (φ_n φ_nᵀ) ⊗ M_n`
            // with a per-row output metric `M_n`, which does NOT factor out of the
            // basis Gram. Install the matrix-free `WhitenedRowGramPenaltyOp` (per-row
            // gather → apply `M_n` → scatter, sharing the cross-block's
            // `SaeKroneckerRows` support + metric) instead of the isotropic
            // `SparseBlockKroneckerPenaltyOp`, and DO NOT install the device SAE PCG
            // data: the device kernel (`DeviceSaePcgData`) hard-codes the `G ⊗ I_p`
            // gather and the raw `local_jac` cross-block, so it cannot represent the
            // per-row metric. Declining it routes the solve to the CPU row-procedural
            // reduced-Schur matvec, which drives `H_tβ` through the metric-aware
            // `sys.htbeta_matvec` closure, `H_ββ` through `sys.effective_penalty_op()`
            // (this op), and the t-block through the already-metric-aware per-row
            // `htt`. `g_blocks` (the isotropic collapsed Gram) is intentionally
            // unused here.
            sys.set_block_offsets(self.beta_block_offsets());
            let gram_kron = whitened_gram_kron
                .expect("whitening non-frames path always populates whitened_gram_kron");
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = smooth_ops;
            ops.push(Arc::new(WhitenedRowGramPenaltyOp::new(gram_kron, beta_dim)));
            if beta_penalty_assembly.dense_written {
                ops.push(Arc::new(DensePenaltyOp(sys.hbb.clone())));
            }
            // #1038 — the barrier's exact self-concordant rank-1, full-`B` layout
            // (no frame projection on the non-frames path). Already CPU (no device
            // data installed on the whitening path), so no extra fallback guard.
            for (scale, carrier) in &sep_rank1 {
                ops.push(Arc::new(SparseRankOnePenaltyOp {
                    k: beta_dim,
                    scale: *scale,
                    carrier: carrier.clone(),
                }));
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: beta_dim, ops }));
            self.reclaim_border_hbb_workspace(&mut sys);
        } else {
            let (device_a_phi, device_local_jac) =
                device_rows.expect("full-beta SAE PCG rows are cloned before row operator install");
            // Wire per-atom β block ranges so the Jacobi preconditioner builds one
            // dense Schur sub-block per atom (block-Jacobi) instead of scalar-diagonal
            // inversion.  Each atom's decoder coefficients form a natural block:
            // `[beta_offsets[k] .. beta_offsets[k] + basis_size[k] * p_out]`.
            sys.set_block_offsets(self.beta_block_offsets());
            // Install the composite BetaPenaltyOp (#296): smoothness contributions
            // via per-atom KroneckerPenaltyOp (avoid dense K×K materialisation), the
            // data-fit Gauss-Newton β-Hessian as the structured `G ⊗ I_p`
            // SparseBlockKroneckerPenaltyOp (block-sparse over co-occurring
            // `(atom, atom')` pairs, block-diagonal across the `p` output channels,
            // identical per channel), plus — only when a Beta-tier analytic penalty
            // was written — the dense `sys.hbb` residual contribution. When no beta
            // penalty fired, `sys.hbb` is all-zero and the dense `(K·p)²` operator
            // is skipped entirely. The sparse data op tracks only the active-atom
            // couplings, so its storage and matvec cost scale with `k_active`, not
            // `K`, at `K = 100K`.
            // Convert the per-atom-pair coupling map into `SparseGBlock`s keyed
            // by μ-space offsets. Empty blocks (no co-occurrence) are simply
            // absent from the map.
            let g_sparse_blocks: Vec<SparseGBlock> = g_blocks
                .into_iter()
                .filter_map(|((atom_i, atom_j), data)| {
                    if data.iter().all(|&v| v == 0.0) {
                        None
                    } else {
                        Some(SparseGBlock {
                            row_off: mu_offsets[atom_i],
                            col_off: mu_offsets[atom_j],
                            data,
                        })
                    }
                })
                .collect();
            let device_smooth_blocks = smooth_scaled_s
                .iter()
                .enumerate()
                .map(|(atom_idx, factor_a)| {
                    // #1117 — rank deficiency is removed at the basis layer, so the
                    // device PCG smooth block is just `λ S_k ⊗ I_p` (full-rank
                    // design); no data-null deflation is folded in here.
                    DeviceSaeSmoothBlock {
                        global_offset: beta_offsets[atom_idx],
                        factor_a: factor_a.clone(),
                    }
                })
                .collect();
            // #1038 — the device SAE PCG kernel folds only the per-atom scalar smooth
            // blocks and the `G ⊗ I_p` data Gram; it cannot represent the barrier's
            // cross-atom rank-1 curvature. When that rank-1 fires (a co-collapsing
            // dictionary), skip the device install so the solve falls back to the CPU
            // reduced-Schur matvec, which routes `H_ββ` through the composite op below
            // (rank-1 included). Healthy fits install no rank-1 and keep the device PCG.
            if sep_rank1.is_empty() {
                self.install_device_sae_pcg_data(
                    &mut sys,
                    DeviceSaePcgData {
                        p,
                        beta_dim,
                        a_phi: device_a_phi,
                        local_jac: device_local_jac,
                        smooth_blocks: device_smooth_blocks,
                        sparse_g_blocks: g_sparse_blocks.clone(),
                        frame: None,
                    },
                );
            }
            let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = smooth_ops;
            ops.push(Arc::new(SparseBlockKroneckerPenaltyOp {
                p,
                dim_a: m_total,
                k: beta_dim,
                blocks: g_sparse_blocks,
            }));
            if beta_penalty_assembly.dense_written {
                ops.push(Arc::new(DensePenaltyOp(sys.hbb.clone())));
            }
            // #1038 — barrier's exact self-concordant rank-1 (full-`B`, no projection).
            for (scale, carrier) in &sep_rank1 {
                ops.push(Arc::new(SparseRankOnePenaltyOp {
                    k: beta_dim,
                    scale: *scale,
                    carrier: carrier.clone(),
                }));
            }
            sys.set_penalty_op(Arc::new(CompositePenaltyOp { k: beta_dim, ops }));
            self.reclaim_border_hbb_workspace(&mut sys);
        }
        if let Some(deflation) = self
            .row_gauge_deflation_for_layout(row_layout.as_ref())
            .or_else(|| {
                // #974 — enable evidence-mode spectral discovery of the
                // metric-null / indefinite quotient directions a rank-deficient
                // whitening metric creates in `H_tt`, even for Euclidean atoms
                // that supply no rotation/phase gauge (so `row_gauge_deflation_
                // for_layout` returns `None`). An empty per-row gauge routes the
                // factor through the spectral-deflation path
                // (`allow_spectral_deflation`), which deflates such a genuine flat
                // direction to unit stiffness (`log 1 = 0`, ρ-independent, so the
                // evidence value and its ρ-adjoint stay consistent) instead of
                // refusing the block. No-op when the metric is full-rank.
                low_rank_whiten.then(|| Self::empty_row_gauge_deflation(n))
            })
        {
            sys.set_row_gauge_deflation(deflation);
        }
        // Store the active-set layout for `apply_newton_step`.
        self.last_row_layout = row_layout;
        // Record whether `delta_beta` from this system is a factored ΔC (needs a
        // frame lift) or a full-`B` ΔB. Read by `apply_newton_step_impl`.
        self.last_frames_active = frames_engaged;
        Ok(sys)
    }

    /// Project a dense full-`B` Beta-tier penalty Hessian `hbb` (`beta_dim ×
    /// beta_dim`, the analytic `∂²P/∂B∂B` block) into the factored coordinate
    /// space `Φᵀ hbb Φ` (`border_dim × border_dim`) for the #972 / #977 T1
    /// frame transform. `Φ = blkdiag(I_{M_k} ⊗ U_k)` maps C-space → B-space, so
    /// the projected block contracts both index legs through the per-atom frames.
    ///
    /// The projection is done in two passes to stay `O(beta_dim · border_dim +
    /// border_dim²)` instead of forming the dense `Φ` explicitly: first
    /// `T = hbb · Φ` (right multiply, columns fold `U`), then `Φᵀ · T` (left
    /// multiply, rows fold `U`). Analytic Beta-tier penalties are rare and small,
    /// so this only fires when one is actually installed.
    pub(crate) fn project_dense_penalty_to_factored(
        &self,
        hbb: ArrayView2<'_, f64>,
        projection: &FrameProjection,
    ) -> Array2<f64> {
        projection.project_block(hbb)
    }

    pub(crate) fn build_factored_beta_penalty_curvature(
        &self,
        registry: &AnalyticPenaltyRegistry,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) -> Array2<f64> {
        let rho_global = Array1::<f64>::zeros(registry.total_rho_count());
        let layout = registry.rho_layout();
        let target_beta = self.flatten_beta();
        let mut hbb_c = Array2::<f64>::zeros((projection.border_dim(), projection.border_dim()));
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            if matches!(penalty, AnalyticPenaltyKind::Ard(_)) {
                continue;
            }
            let rho_local = rho_global.slice(s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi if matches!(penalty, AnalyticPenaltyKind::NuclearNorm(_)) => {
                    self.add_factored_beta_penalty_curvature_for_penalty(
                        &mut hbb_c,
                        penalty,
                        target_beta.view(),
                        rho_local,
                        penalty_scale,
                        projection,
                    );
                }
                PenaltyTier::Beta => {
                    self.add_factored_beta_penalty_curvature_for_penalty(
                        &mut hbb_c,
                        penalty,
                        target_beta.view(),
                        rho_local,
                        penalty_scale,
                        projection,
                    );
                }
                _ => {}
            }
        }
        hbb_c
    }

    pub(crate) fn add_factored_beta_penalty_curvature_for_penalty(
        &self,
        hbb_c: &mut Array2<f64>,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) {
        let p = self.output_dim();
        if let AnalyticPenaltyKind::DecoderIncoherence(base) = penalty {
            let Some(per_fit) = self.live_decoder_incoherence_penalty(base) else {
                return;
            };
            let beta_dim = self.beta_dim();
            let mut probe = Array1::<f64>::zeros(beta_dim);
            for k in 0..self.atoms.len() {
                for basis_col in 0..projection.basis_sizes[k] {
                    for frame_col in 0..projection.ranks[k] {
                        probe.fill(0.0);
                        projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                        let col = projection.border_offsets[k]
                            + basis_col * projection.ranks[k]
                            + frame_col;
                        let hv = per_fit.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                        projection
                            .project_border_vec(hv.view())
                            .iter()
                            .enumerate()
                            .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                    }
                }
            }
            return;
        }
        if let AnalyticPenaltyKind::MechanismSparsity(base) = penalty {
            for (per_atom, start, end) in self.live_mechanism_sparsity_penalties(base) {
                let atom_idx = projection
                    .beta_offsets
                    .iter()
                    .position(|&offset| offset == start)
                    .expect("live mechanism-sparsity offset must match an SAE atom");
                let block_len = end - start;
                let mut local_penalty = per_atom.clone();
                local_penalty.target = PsiSlice {
                    range: 0..block_len,
                    latent_dim: Some(projection.basis_sizes[atom_idx]),
                };
                let block = target_beta.slice(s![start..end]);
                let mut probe = Array1::<f64>::zeros(block_len);
                for basis_col in 0..projection.basis_sizes[atom_idx] {
                    for frame_col in 0..projection.ranks[atom_idx] {
                        probe.fill(0.0);
                        projection.lift_local_axis_into(&mut probe, atom_idx, basis_col, frame_col);
                        let col = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx]
                            + frame_col;
                        let hv = local_penalty.psd_majorizer_hvp(block, rho_local, probe.view());
                        projection.project_local_atom_vec_into(
                            atom_idx,
                            hv.view(),
                            hbb_c.column_mut(col),
                            penalty_scale,
                        );
                    }
                }
            }
            return;
        }
        if let AnalyticPenaltyKind::NuclearNorm(base) = penalty {
            for (per_atom, start, end) in self.live_nuclear_norm_penalties(base) {
                let atom_idx = projection
                    .beta_offsets
                    .iter()
                    .position(|&offset| offset == start)
                    .expect("live nuclear-norm offset must match an SAE atom");
                let block = target_beta.slice(s![start..end]);
                let block_len = end - start;
                let mut probe = Array1::<f64>::zeros(block_len);
                for basis_col in 0..projection.basis_sizes[atom_idx] {
                    for frame_col in 0..projection.ranks[atom_idx] {
                        probe.fill(0.0);
                        projection.lift_local_axis_into(&mut probe, atom_idx, basis_col, frame_col);
                        let col = projection.border_offsets[atom_idx]
                            + basis_col * projection.ranks[atom_idx]
                            + frame_col;
                        let hv = per_atom.psd_majorizer_hvp(block, rho_local, probe.view());
                        projection.project_local_atom_vec_into(
                            atom_idx,
                            hv.view(),
                            hbb_c.column_mut(col),
                            penalty_scale,
                        );
                    }
                }
            }
            return;
        }
        let beta_dim = self.beta_dim();
        let mut probe = Array1::<f64>::zeros(beta_dim);
        for k in 0..self.atoms.len() {
            for basis_col in 0..projection.basis_sizes[k] {
                for frame_col in 0..projection.ranks[k] {
                    probe.fill(0.0);
                    projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                    let col =
                        projection.border_offsets[k] + basis_col * projection.ranks[k] + frame_col;
                    let hv = penalty.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                    projection
                        .project_border_vec(hv.view())
                        .iter()
                        .enumerate()
                        .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                }
            }
        }
        assert_eq!(p, self.output_dim());
    }

    /// #1610 — project the frozen-gate decoder-repulsion PSD majorizer into the
    /// factored β block `hbb_c`. Mirrors the `DecoderIncoherence` arm of
    /// [`Self::add_factored_beta_penalty_curvature_for_penalty`] but sources the
    /// penalty from [`Self::live_decoder_repulsion_penalty`] (registry-independent,
    /// collinearity-gated), so the repulsion curvature reaches the operator on the
    /// matrix-free/framed path where the dense `sys.hbb` write is unused. No-op
    /// when no repulsion is active.
    pub(crate) fn add_factored_repulsion_curvature(
        &self,
        hbb_c: &mut Array2<f64>,
        penalty_scale: f64,
        projection: &FrameProjection,
    ) {
        let Some(per_fit) = self.live_decoder_repulsion_penalty() else {
            return;
        };
        let beta_dim = self.beta_dim();
        let target_beta = self.flatten_beta();
        // The repulsion penalty is non-learnable; its strength is already folded
        // into the frozen gate (see `live_decoder_repulsion_penalty`), so the rho
        // slice is empty/inert.
        let rho_local = Array1::<f64>::zeros(0);
        let mut probe = Array1::<f64>::zeros(beta_dim);
        for k in 0..self.atoms.len() {
            for basis_col in 0..projection.basis_sizes[k] {
                for frame_col in 0..projection.ranks[k] {
                    probe.fill(0.0);
                    projection.lift_axis_into(&mut probe, k, basis_col, frame_col);
                    let col =
                        projection.border_offsets[k] + basis_col * projection.ranks[k] + frame_col;
                    let hv = per_fit.psd_majorizer_hvp(
                        target_beta.view(),
                        rho_local.view(),
                        probe.view(),
                    );
                    projection
                        .project_border_vec(hv.view())
                        .iter()
                        .enumerate()
                        .for_each(|(row, &v)| hbb_c[[row, col]] += penalty_scale * v);
                }
            }
        }
    }
}
