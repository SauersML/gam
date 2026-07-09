// [#780] Exact stationarity-Jacobian correction (`apply_exact_hessian_minus_b`),
// the exact inner-fit Hessian apply (`apply_exact_hessian`), and the exact
// stationarity solve (`solve_exact_stationarity`) were extracted verbatim from
// `construction.rs` into this sibling file to keep that file under the #780
// per-file line-count gate. It is `include!`d back into the parent module in
// `construction.rs`, so these methods share that module's scope exactly as
// before (same `impl SaeManifoldTerm`, same `use super::*` imports).

impl SaeManifoldTerm {
    /// #1418: apply the EXACT stationarity-Jacobian correction `ΔC·v = (A − B)·v`
    /// to a joint `(t, β)` vector, matrix-free and per row.
    ///
    /// `A = ∇²_θθ L` is the true inner-fit Hessian; `B` is the assembled
    /// evidence/Newton operator the solver factors. They differ ONLY by the three
    /// curvature substitutions the assembly makes for stability:
    ///   1. data: `B` uses Gauss-Newton `J̃J̃ᵀ`, dropping the residual curvature
    ///      `R[a,b] = Σ_out r_out·∂²f_out/∂θ_a∂θ_b` (t–t via `jets.second`, t–β via
    ///      `jets.beta_deriv`; the decoder is linear in β so the β–β block is 0);
    ///   2. softmax: `B` uses the Gershgorin majorizer `D = diag(Σ_j|H_kj|)`,
    ///      dropping `H_entropy − D` (#1419);
    ///   3. periodic ARD: `B` uses `max(V'',0)`, dropping the negative part
    ///      `min(V'',0)` (the indefinite tail past a quarter period).
    /// `ΔC` is the sum of exactly these three deltas, each built from the SAME
    /// jets / penalty curvatures the assembly and the θ-adjoint use, so
    /// `A = B + ΔC` is the one true Hessian. Exact on BOTH the isotropic and the
    /// whitened-metric paths: the data fit is `½ r_nᵀ M_n r_n`, so the residual
    /// curvature is `Σ_out (M_n r_n)_out·∂²f_out/∂θ_a∂θ_b` — contract the
    /// metric-applied √w-scaled residual `error_metric = √w·M_n r_n` (the SAME
    /// quantity the assembly's β-tier gradient uses) against the RAW second jets
    /// `jets.second`/`jets.beta_deriv` (the same raw-jet convention the whole
    /// θ-adjoint and the Gauss-Newton `htt = J̃J̃ᵀ = J M Jᵀ` assembly use). On the
    /// isotropic path `M_n = I` so `error_metric = √w·r` and `J M Jᵀ = JJᵀ`,
    /// recovering the plain case. The softmax / ARD deltas are logit/coord-space
    /// prior curvatures and carry no output metric, so they are path-independent.
    fn apply_exact_hessian_minus_b(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_t = cache.delta_t_len();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();

        // Optional softmax exact-entropy-minus-majorizer delta operator (#1419).
        let softmax_delta: Option<(
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

        let mut out = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        // #1557 — reuse one K-sized scratch row across all N rows (alias-free).
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        // #932 SIMD: jets are built in aligned 4-row SIMD batches through a
        // bounded (≤4-row) look-ahead window; unaligned / non-softmax / remainder
        // rows fall back to the scalar per-row path (bit-identical either way).
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment
                .try_assignments_row_for_rho_into(row, rho, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    rho,
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());

            // √w-scaled metric-applied per-row residual `error_metric = √w·M_n r_n`
            // (the SAME object the assembly's β-tier gradient contracts). The
            // data-fit `½ r_nᵀ M_n r_n` has residual curvature `Σ (M_n r_n)·∂²f`,
            // so this is exactly the residual contracted against the raw `∂²f`
            // jets. `M_n = I` on the isotropic path ⇒ `error_metric = √w·r`.
            fitted.fill(0.0);
            let active_atoms = self
                .last_row_layout
                .as_ref()
                .map(|layout| layout.active_atoms[row].as_slice());
            for k in 0..k_atoms {
                if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                    continue;
                }
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let a_k = assignments[k];
                for out_col in 0..p {
                    fitted[out_col] += a_k * decoded[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
            }
            let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                _ => error.to_vec(),
            };

            // Local t-slice of `v` for this row.
            let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();

            // (1a) residual curvature, t–t: ΔC_tt[a,b] = ⟨r, ∂²f_ab⟩.
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = sae_dot(&error_metric, &jets.second[a][b]);
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            // (1b) residual curvature, t–β and β–t: ΔC_tβ[a,β] = ⟨r, ∂²f_aβ⟩.
            //      `jets.beta_deriv[a][β]` = ∂(∂f/∂β_β)/∂θ_a (the mixed second jet).
            for a in 0..q {
                for (beta_pos, channel) in border.iter().enumerate() {
                    let r_ab = sae_dot(&error_metric, &jets.beta_deriv[a][beta_pos]);
                    // t row picks up β leg of v; β row picks up t leg of v.
                    out.t[base + a] += r_ab * v.beta[channel.index];
                    out.beta[channel.index] += r_ab * v_t[a];
                }
            }

            // (2) softmax: ΔC_logit = (H_entropy − D) over the free logits, where
            // `D = diag(Σ_j|H_kj|)` is the Gershgorin majorizer the assembled `B`
            // wrote into the logit block (#1419). Adding `H_entropy − D` recovers the
            // EXACT entropy curvature `A = B + ΔC`, so the solver's exact-Hessian
            // correction differentiates the SAME operator the assembly installed.
            if let Some((_penalty, scale)) = softmax_delta.as_ref() {
                let assignment_dim = self.assignment.assignment_coord_dim();
                // #1410: the correction only contracts the ACTIVE logit slots
                // (`jets.vars` carries the row's `≤ top_k` active atoms on the
                // compact layout), so build only the active sub-block of
                // `ΔC = H_entropy − D` ENTRY-WISE rather than materialising the
                // full `K×K` `row_dense_hessian` / `row_psd_majorizer` matrices per
                // row (an `O(K²)`-per-row allocation that defeated the compact
                // contract at the LLM shape). `D` is diagonal, so it subtracts only
                // on `ka == kb`; the off-diagonal `H_entropy` entries come from the
                // shared `(a, l, m)` algebra. The softmax row `a_soft` is the one
                // irreducible `O(K)` term, computed once per row.
                // #1557 — reuse this iteration's `assignments` (bit-identical).
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    let mut acc = 0.0_f64;
                    for (b, vb) in jets.vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, *scale);
                        // `D` is the diagonal Gershgorin majorizer (#1419), so it
                        // contributes only on the diagonal `ka == kb`.
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, *scale)
                        } else {
                            h_entropy
                        };
                        acc += delta * v_t[b];
                    }
                    out.t[base + a] += acc;
                }
            }

            // (3) periodic ARD: ΔC_coord = (V'' − max(V'',0)) = min(V'',0), diagonal.
            // The assembly writes the mean-one design-weighted majorizer
            // `w_row·max(V'',0)`, so the dropped-curvature correction must carry
            // that same `w_row`: `A = B + ΔC` then recovers `w_row·V''` exactly.
            // The prior is weighted directly, not through the √w data-jet seam.
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in jets.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.hess.min(0.0);
                if neg != 0.0 {
                    out.t[base + a] += w_row * neg * v_t[a];
                }
            }
        }
        Ok(out)
    }

    /// #1418: matrix-free apply of the EXACT stationarity Jacobian `A = ∇²_θθ L`:
    /// `A v = B v + ΔC v`, the assembled arrow Hessian apply
    /// ([`apply_cached_arrow_hessian`]) plus the matrix-free dropped-curvature
    /// correction `ΔC = A − B` ([`Self::apply_exact_hessian_minus_b`]).
    fn apply_exact_hessian(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let b_v = apply_cached_arrow_hessian(cache, v.t.view(), v.beta.view())?;
        let dc_v = self.apply_exact_hessian_minus_b(rho, target, cache, v)?;
        Ok(SaeArrowVector {
            t: &b_v.t + &dc_v.t,
            beta: &b_v.beta + &dc_v.beta,
        })
    }

    /// #1418: solve `A x = rhs` for the EXACT stationarity Jacobian `A = ∇²_θθ L`
    /// via left-`B`-preconditioned GMRES ([`solve_b_preconditioned_gmres`]) with the
    /// matrix-free `A v = B v + ΔC v` apply ([`Self::apply_exact_hessian`]). The
    /// IFT step `θ̂_ρ = −A⁻¹ g_ρ` must invert the EXACT `A`, not the surrogate `B`;
    /// GMRES does not require the exact stationarity Jacobian to be SPD; it
    /// refuses non-convergence instead of returning a negative-curvature CG
    /// iterate as though it were an inverse solve.
    pub(crate) fn solve_exact_stationarity(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        solve_b_preconditioned_gmres(solver, rhs, |v| {
            self.apply_exact_hessian(rho, target, cache, v)
        })
    }

    /// Analytic SAE REML outer-ρ gradient components at the already converged
    /// inner state represented by `loss` and `cache`.
    ///
    /// The returned gradient is the assembled analytic outer derivative:
    /// explicit penalty terms, direct logdet traces, Occam terms, and the #1006
    /// implicit-state third-order correction.
    pub(crate) fn analytic_outer_rho_gradient_components(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        self.analytic_outer_rho_gradient_components_with_bundle(
            target, rho, loss, cache, solver, None,
        )
    }

    /// #2080 forward plumbing — the analytic outer-ρ gradient with an OPTIONAL
    /// shared selected-inverse probe bundle `(z_j, S⁻¹ z_j)`.
    ///
    /// When `inverse_probe_bundle` is `Some`, the THREE selected-inverse channels
    /// that have matrix-free siblings — the per-atom decoder smoothness EDF
    /// `tr(H⁻¹ M_k)`, the per-(atom,axis) ARD log-precision Hessian trace
    /// `½tr(H⁻¹ ∂H/∂logα)`, and the #1006 envelope Γ = tr(H⁻¹ ∂H/∂θ) — are evaluated
    /// off that bundle (`decoder_smoothness_effective_dof_per_atom_from_probes` /
    /// `ard_log_precision_hessian_trace_from_probes` / `logdet_theta_adjoint_from_probes`)
    /// instead of the dense `DeflatedArrowSolver` selected inverse. They convert
    /// together as ONE all-or-nothing cluster on the single `Some` (invariant #1):
    /// never a partial mix within a single eval. Each from-probes channel hard-refuses
    /// deflated rows (the plain-S⁻¹ bundle cannot reconstruct the Daleckii–Krein
    /// correction), routing those fits to the dense channel.
    ///
    /// The analytic-gradient cluster is DENSE-ONLY today (invariant #3): every
    /// production caller passes `None`, so this `Some` branch is dormant forward
    /// plumbing that the eventual routing flip (once the surrogate lane owns the
    /// analytic gradient, not just the EFS lane) will exercise. Flipping any
    /// caller to `Some` still requires matrix-free siblings for the one remaining
    /// solver-bound channel — the assignment/learnable-IBP log-strength traces
    /// (`assignment_log_strength_hessian_trace` / `learnable_ibp_data_logdet_alpha_trace`,
    /// `logdet_trace[0]`) — so the `solver` argument is still required here and the
    /// flip stays off until that last gap closes.
    pub(crate) fn analytic_outer_rho_gradient_components_with_bundle(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        inverse_probe_bundle: Option<(&[Array1<f64>], &[Array1<f64>])>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);

        explicit[0] = assignment_prior_log_strength_derivative(&self.assignment, rho)
            + self
                .learnable_ibp_forward_alpha_data_derivative(rho, target)
                .map_err(OuterGradientError::internal)?;
        // #1417: the FULL `½ tr(H⁻¹ ∂H/∂logα)` for the assignment coordinate.
        // For LEARNABLE IBP alpha the forward assignments `a_ik = σ(ℓ/τ)·π_k(α)`
        // carry an explicit α-dependence (`∂logπ_k/∂logα = k/(α+1)`), so BOTH the
        // assignment-prior Hessian AND the data Gauss-Newton blocks
        // `H_ββ`, `H_tβ`, `H_tt` depend on logα. We assemble both traces:
        //   • prior:  `assignment_log_strength_hessian_trace`,
        //   • data:   `learnable_ibp_data_logdet_alpha_trace` (#1417), using the
        //             exact `(k_a+k_b)/(α+1)` block-scaling identity.
        // For FIXED alpha (and non-IBP modes) the data term is identically zero,
        // so the fixed-alpha gradient is unchanged and exact.
        logdet_trace[0] = self
            .assignment_log_strength_hessian_trace(rho, cache, solver)
            .map_err(OuterGradientError::internal)?
            + self
                .learnable_ibp_data_logdet_alpha_trace(rho, cache, solver)
                .map_err(OuterGradientError::internal)?;

        // #1556: λ_smooth is per-atom, so the smoothness gradient block occupies
        // flat indices `1..1+K` (one per atom), not a single index 1. Each atom
        // `k` carries its own explicit penalty-energy derivative, log|H| trace,
        // and Occam-normalizer derivative.
        let k_smooth = rho.log_lambda_smooth.len();
        let lambda_smooth_vec = rho.lambda_smooth_vec();
        // Explicit `∂loss.smoothness/∂log λ_k = 0.5·λ_k·<B_k, S_k B_k>` (the
        // per-atom split). Its sum is the λ-scaled penalty energy; renormalize to
        // `loss.smoothness` so the total matches the criterion's reported energy
        // bit-for-bit (folding in any minibatch `penalty_scale` baked into it).
        let mut smooth_explicit = self.decoder_smoothness_value_per_atom(&lambda_smooth_vec);
        let smooth_explicit_sum: f64 = smooth_explicit.iter().sum();
        if smooth_explicit_sum.abs() > 0.0 {
            let renorm = loss.smoothness / smooth_explicit_sum;
            for v in smooth_explicit.iter_mut() {
                *v *= renorm;
            }
        }
        // #2080: the per-atom smoothness EDF `tr(H⁻¹ M_k)` off the shared
        // selected-inverse bundle when the surrogate lane supplied it; the dense
        // `DeflatedArrowSolver` selected inverse otherwise (all callers today).
        let smooth_logdet = match inverse_probe_bundle {
            Some((probes, sinv)) => self
                .decoder_smoothness_effective_dof_per_atom_from_probes(
                    probes,
                    sinv,
                    &lambda_smooth_vec,
                )
                .map_err(|err| OuterGradientError::InternalInvariant {
                    reason: format!(
                        "analytic_outer_rho_gradient_components: smooth dof (matrix-free): {err}"
                    ),
                })?,
            None => self
                .decoder_smoothness_effective_dof_with_solver_per_atom(
                    cache,
                    solver,
                    &lambda_smooth_vec,
                )
                .map_err(|err| OuterGradientError::InternalInvariant {
                    reason: format!("analytic_outer_rho_gradient_components: {err}"),
                })?,
        };
        let smooth_occam = self
            .reml_occam_log_lambda_smooth_derivative(rho)
            .map_err(OuterGradientError::internal)?;
        for atom_idx in 0..k_smooth {
            explicit[1 + atom_idx] = smooth_explicit[atom_idx];
            logdet_trace[1 + atom_idx] = 0.5 * smooth_logdet[atom_idx];
            occam[1 + atom_idx] = -smooth_occam[atom_idx];
        }

        let ard_explicit = self
            .ard_log_precision_explicit_derivatives(rho)
            .map_err(OuterGradientError::internal)?;
        // #2080: the per-(atom,axis) ARD log-precision Hessian trace
        // `½tr(H⁻¹ ∂H/∂logα)` off the SAME shared selected-inverse bundle (the
        // all-or-nothing cluster's second channel) when present; the dense
        // deflated selected inverse otherwise. The from-probes channel HARD-REFUSES
        // any row carrying gauge/rotation deflation (the plain-S⁻¹ bundle cannot
        // reconstruct the Daleckii–Krein correction), routing that fit to the dense
        // channel rather than silently dropping the correction.
        let ard_trace = match inverse_probe_bundle {
            Some((probes, sinv)) => self
                .ard_log_precision_hessian_trace_from_probes(rho, cache, probes, sinv)
                .map_err(|err| OuterGradientError::InternalInvariant {
                    reason: format!(
                        "analytic_outer_rho_gradient_components: ARD logdet trace \
                         (matrix-free): {err}"
                    ),
                })?,
            None => self
                .ard_log_precision_hessian_trace(rho, cache, solver)
                .map_err(|err| OuterGradientError::InternalInvariant {
                    reason: format!("analytic_outer_rho_gradient_components: {err}"),
                })?,
        };
        // #1026 shared-ARD: `ard_flat_index` maps `(k, axis)` onto the flat outer
        // coordinate for BOTH parameterizations. In `Shared` mode several atoms
        // alias one axis coordinate `1+K+axis`, and the outer derivative there is
        // `∂/∂log α_axis = Σ_{k owns axis} ∂/∂log α_{k,axis}` (chain rule through
        // the broadcast), so we ACCUMULATE. In `PerAtom` mode each `(k, axis)` has
        // a unique coordinate, so `+=` is identical to the historical `=`. Walking
        // a raw per-atom cursor in `Shared` mode would index past the flat length
        // `1+K+max_d` (OOB) and split one shared strength across phantom slots.
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                let idx = rho.ard_flat_index(k, axis);
                explicit[idx] += ard_explicit[k][axis];
                logdet_trace[idx] += ard_trace[k][axis];
            }
        }

        // #2080: the envelope Γ = tr(H⁻¹ ∂H/∂θ) off the SAME shared selected-inverse
        // bundle (the all-or-nothing cluster's third channel) when present; the dense
        // selected inverse otherwise. The border-only bundle reconstructs the NO-SELF
        // base inverse `(H₀')⁻¹`, so `logdet_theta_adjoint_from_probes` HARD-REFUSES
        // (routes to dense) any cache carrying a T-space rank-R correction the border
        // cannot span — per-row gauge/rotation deflation OR an IBP cross-row Woodbury —
        // and otherwise owns the softmax / euclidean / non-cross-row regimes exactly.
        // This completes the matrix-free selected-inverse cluster (smoothness EDF + ARD
        // Hessian trace + θ-adjoint); the assignment/learnable-IBP log-strength traces
        // (`logdet_trace[0]`) plus the θ-adjoint's IBP-refused fits remain solver-bound
        // — the last gaps before the routing flip (see the docstring).
        let gamma = match inverse_probe_bundle {
            Some((probes, sinv)) => self
                .logdet_theta_adjoint_from_probes(rho, cache, probes, sinv)
                .map_err(OuterGradientError::internal)?,
            None => self
                .logdet_theta_adjoint(rho, cache, solver)
                .map_err(OuterGradientError::internal)?,
        };
        // #1418: the implicit-function correction is `−½·Γᵀ·θ̂_ρ` with
        // `θ̂_ρ = −A⁻¹ g_ρ`, where `A = ∇²_θθ L` is the EXACT stationarity
        // Jacobian of the inner fit — data residual curvature, exact softmax
        // entropy Hessian, exact periodic ARD curvature. The matrix the `solver`
        // factors is `B` (Gauss-Newton data curvature, softmax Fisher metric,
        // `max(V'',0)` ARD majorizers): the `½log|B|` Laplace term is consistent
        // with `Γ = ½tr(B⁻¹ ∂B/∂θ)`, but the implicit step is governed by `A`.
        // `solve_exact_stationarity` applies the TRUE `A⁻¹` with left-`B`
        // preconditioned GMRES on `A = B + ΔC`, where
        // `ΔC = apply_exact_hessian_minus_b`, so the correction is no longer
        // biased by `(B⁻¹ − A⁻¹)` and does not assume `A` is SPD.
        //
        // A numerical stopping tolerance does not change the mathematical
        // objective.  At the exact inner optimum the envelope theorem cancels
        // the penalized-loss response, but the Laplace term still contributes
        // `-1/2 Gamma' theta_hat_rho`.  Dropping this term differentiates a
        // fictitious criterion in which the fitted state is held fixed.  The
        // exact stationarity solve above supplies the required implicit response.
        for coord in 0..n_params {
            let rhs = self
                .outer_rho_gradient_ift_rhs(rho, target, coord, cache)
                .map_err(OuterGradientError::internal)?;
            let solved = self
                .solve_exact_stationarity(rho, target, cache, solver, &rhs)
                .map_err(OuterGradientError::internal)?;
            let mut dot = 0.0_f64;
            for idx in 0..gamma.t.len() {
                dot += gamma.t[idx] * solved.t[idx];
            }
            for idx in 0..gamma.beta.len() {
                dot += gamma.beta[idx] * solved.beta[idx];
            }
            third_order_correction[coord] = -0.5 * dot;
        }

        Ok(SaeOuterRhoGradientComponents {
            explicit,
            logdet_trace,
            occam,
            third_order_correction,
        })
    }
}
