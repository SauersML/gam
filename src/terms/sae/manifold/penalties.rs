use super::*;

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
        let mut coactivation = Array2::<f64>::zeros((k_atoms, k_atoms));
        for j in 0..k_atoms {
            for k in 0..k_atoms {
                let mut s = 0.0;
                for row in 0..n {
                    s += gates[[row, j]] * gates[[row, k]];
                }
                coactivation[[j, k]] = s * inv_n;
            }
        }
        let mut per_fit: DecoderIncoherencePenalty = (**base).clone();
        per_fit.block_sizes = block_sizes;
        per_fit.p_out = p;
        per_fit.target = PsiSlice {
            range: 0..m_total * p,
            latent_dim: Some(m_total),
        };
        per_fit.coactivation = coactivation;
        Some(per_fit)
    }

    /// #1026 — refresh the frozen per-assembly decoder-repulsion gate from the
    /// current decoder state. Lagged-diffusivity discipline (mirrors
    /// [`SaeManifoldAtom::refresh_intrinsic_smooth_penalty`]): the gate WEIGHT is
    /// frozen here at assembly entry so the assembly's gradient/curvature and the
    /// line-search value path use the same gate even as trial decoders move.
    ///
    /// The per-pair weight is `SAE_DECODER_REPULSION_STRENGTH · gate(s_jk)` with
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
        let mut gate = Array2::<f64>::zeros((k_atoms, k_atoms));
        let mut any_active = false;
        let s0 = SAE_DECODER_REPULSION_COLLINEARITY_GATE;
        for j in 0..k_atoms {
            for k in (j + 1)..k_atoms {
                // Both decoders need a usable scale; a ~zero decoder has no
                // direction to be collinear with, so leave the pair at 0 (the
                // decoder-norm / mass guards own that degeneracy, not this term).
                if !(norm_sq[j] > 0.0 && norm_sq[k] > 0.0) {
                    continue;
                }
                // Cross-Gram Frobenius energy ‖B_jB_kᵀ‖²_F = Σ_{a,b} C[a,b]² with
                // C[a,b] = Σ_o B_j[a,o]·B_k[b,o]; normalized by the two norms it
                // is the squared cosine of the decoder row-spaces ∈ [0, 1].
                let bj = &self.atoms[j].decoder_coefficients;
                let bk = &self.atoms[k].decoder_coefficients;
                let (m_j, p) = (bj.nrows(), bj.ncols());
                let m_k = bk.nrows();
                if bk.ncols() != p {
                    continue;
                }
                let mut cross_sq = 0.0_f64;
                for a in 0..m_j {
                    for b in 0..m_k {
                        let mut c = 0.0_f64;
                        for o in 0..p {
                            c += bj[[a, o]] * bk[[b, o]];
                        }
                        cross_sq += c * c;
                    }
                }
                let s_jk = cross_sq / (norm_sq[j] * norm_sq[k]);
                // C1 smoothstep gate: 0 below s0, smooth ramp to 1 at s=1.
                let gate_value = if s_jk <= s0 {
                    0.0
                } else {
                    let t = ((s_jk - s0) / (1.0 - s0)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };
                if gate_value > 0.0 {
                    let w = SAE_DECODER_REPULSION_STRENGTH * gate_value;
                    gate[[j, k]] = w;
                    gate[[k, j]] = w;
                    any_active = true;
                }
            }
        }
        self.decoder_repulsion_gate = if any_active { Some(gate) } else { None };
    }

    /// #1026 — build the [`DecoderIncoherencePenalty`] operator for the frozen
    /// repulsion gate, or `None` when no repulsion is active. Reuses the existing
    /// analytic gradient + PSD majorizer; only the gate (fed as `coactivation`)
    /// and a fixed non-learnable strength differ from a user incoherence penalty.
    pub(crate) fn live_decoder_repulsion_penalty(&self) -> Option<DecoderIncoherencePenalty> {
        let gate = self.decoder_repulsion_gate.as_ref()?;
        let k_atoms = self.k_atoms();
        if k_atoms < 2 || gate.dim() != (k_atoms, k_atoms) {
            return None;
        }
        let p = self.output_dim();
        let block_sizes: Vec<usize> = self.atoms.iter().map(|atom| atom.basis_size()).collect();
        let m_total: usize = block_sizes.iter().sum();
        if m_total == 0 || p == 0 {
            return None;
        }
        // The operator multiplies its quadratic by `weight·pair_weight`; we want
        // the effective per-pair weight to be exactly `gate[j,k]` (which already
        // folds in SAE_DECODER_REPULSION_STRENGTH), so pass weight=1 and feed the
        // gate as the (non-negative, symmetric) coactivation matrix.
        DecoderIncoherencePenalty::new(
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
        let mut probe = Array1::<f64>::zeros(beta_dim);
        for j in 0..beta_dim {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = per_fit.psd_majorizer_hvp(target_beta.view(), rho_local.view(), probe.view());
            for i in 0..beta_dim {
                sys.hbb[[i, j]] += penalty_scale * hv[i];
            }
        }
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
        use crate::terms::analytic_penalties::AnalyticPenalty;
        penalty_scale * per_fit.value(target_beta.view(), rho_local.view())
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
            // `hbb` is the PSD Newton / PIRLS curvature block: probe the PSD
            // majorizer (the Gauss-Newton Hessian, which is already PSD here).
            let mut probe = Array1::<f64>::zeros(beta_dim);
            for j in 0..beta_dim {
                probe.fill(0.0);
                probe[j] = 1.0;
                let hv = per_fit.psd_majorizer_hvp(target_beta, rho_local, probe.view());
                for i in 0..beta_dim {
                    sys.hbb[[i, j]] += penalty_scale * hv[i];
                }
            }
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
                                    beta_assembly.record_curvature(dense_beta_curvature);
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
        let mu = resolve_learnable_weight(corrected.scalar_weight, rho_local[corrected.rho_index]);
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
        let mu = resolve_learnable_weight(corrected.scalar_weight, rho_local[corrected.rho_index]);
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
