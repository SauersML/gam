// [#780] Exact stationarity-Jacobian correction (`apply_exact_hessian_minus_b`),
// the exact inner-fit Hessian apply (`apply_exact_hessian`), and the exact
// stationarity solve (`solve_exact_stationarity`) were extracted verbatim from
// `construction.rs` into this sibling file to keep that file under the #780
// per-file line-count gate. It is `include!`d back into the parent module in
// `construction.rs`, so these methods share that module's scope exactly as
// before (same `impl SaeManifoldTerm`, same `use super::*` imports).

/// Dimensionless numerical-rank floor for the exact-stationarity IFT solve
/// (#2080 defect 4). `B` is the positive-definite scale/preconditioner for the
/// exact stationarity Hessian `A`; the generalized Rayleigh quotient
/// `μ(v) = vᵀAv/vᵀBv` therefore measures exact curvature relative to its own
/// solver scale. The floor is `√ε_machine`, the standard boundary below which
/// a double-precision curvature ratio is not numerically identifiable; it is
/// derived from the scalar type rather than tuned to a fixture. A direction
/// below this floor (a saturated IBP gate logit has data
/// curvature `∝ σ'(ℓ)² → 0`) is numerically curvature-free — the inner
/// optimizer cannot resolve the iterate's position along it, so the IFT
/// response `θ̂_ρ = −A⁻¹g_ρ` there is an unidentifiable `1/μ` amplification,
/// not a real derivative. That amplification is what flipped the analytic
/// λ-gradient's sign against the criterion it differentiates (the #931
/// objective↔gradient desync. The former outer-objective numerical safeguard
/// has been removed: deflating these directions keeps the envelope term
/// value-consistent at its analytic source.
fn sae_ift_min_curvature_fraction() -> f64 {
    f64::EPSILON.sqrt()
}

/// Apply a raw arrow operator on the closed-form gauge quotient represented by
/// `solver`: `M_Q v = M v + κ Q Qᵀ v`.
fn apply_gauge_fixed_arrow_operator<F>(
    solver: &DeflatedArrowSolver<'_>,
    v: &SaeArrowVector,
    apply_raw: &F,
) -> Result<SaeArrowVector, String>
where
    F: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    let mut out = apply_raw(v)?;
    solver.add_gauge_stiffness(v, &mut out)?;
    Ok(out)
}

/// Exact-stationarity Krylov and numerical-null refinement on one coherent
/// gauge-fixed pencil `(A_Q, B_Q)`, where both raw operators receive the same
/// `κ Q Qᵀ` action installed in `solver`.
///
/// Keeping this seam operator-generic makes the quotient invariant directly
/// testable with deterministic matrices while production supplies the real
/// matrix-free exact Hessian `A` and cached majorizer `B`. The helper owns every
/// Krylov, Rayleigh, normalization, and inverse-power apply so none can
/// accidentally regress to a raw operator while using the gauge-fixed inverse.
fn solve_exact_stationarity_on_gauge_quotient<A, B>(
    solver: &DeflatedArrowSolver<'_>,
    rhs: &SaeArrowVector,
    apply_raw_a: &A,
    apply_raw_b: &B,
) -> Result<SaeArrowVector, String>
where
    A: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    B: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    let apply_a_q = |v: &SaeArrowVector| apply_gauge_fixed_arrow_operator(solver, v, apply_raw_a);
    let apply_b_q = |v: &SaeArrowVector| apply_gauge_fixed_arrow_operator(solver, v, apply_raw_b);
    solve_exact_stationarity_preconditioned(rhs, &apply_a_q, &apply_b_q, |vector| {
        solver.solve(vector.t.view(), vector.beta.view())
    })
}

/// Shared exact-stationarity solve on an already identified operator. Dense
/// evidence supplies a gauge-fixed direct inverse; matrix-free evidence supplies
/// a quotient-aware reduced-Schur inverse. Both paths run the identical GMRES,
/// generalized-Rayleigh, and numerical-null certificate below.
fn solve_exact_stationarity_preconditioned<A, B, P>(
    rhs: &SaeArrowVector,
    apply_a: &A,
    apply_b: &B,
    precondition: P,
) -> Result<SaeArrowVector, String>
where
    A: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    B: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    P: Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
{
    let mut x = solve_b_preconditioned_gmres_with(rhs, |v| apply_a(v), |v| precondition(v))?;
    // #2080 defect 4 — deflate unidentifiable near-null pencil directions.
    //
    // The generalized Rayleigh quotient `μ(x) = xᵀAx / xᵀBx` of the
    // SOLUTION is a detector: expanding
    // `x = Σ (vᵢᵀrhs/μᵢ) vᵢ` in the B-orthonormal
    // `(A, B)`-eigenbasis, any near-null component present in `rhs` enters
    // `x` with weight `1/μᵢ`, so `μ(x)` collapses to `≈ μ_min` exactly
    // when the solve was amplified. A healthy solve (`rhs` B-orthogonal to
    // the flat directions, or no flat directions) leaves `μ(x)` above the
    // floor and pays only one extra `A`/`B` apply.
    //
    // Deflation is EXACT in that eigenbasis with no re-solve: the
    // amplified term of `x` along a B-normalized eigendirection `v` is
    // `v·(vᵀBx)` (since `vᵀBx = vᵀrhs/μ_v`), so subtracting the
    // B-projection removes precisely the unidentifiable component while
    // leaving every resolved direction untouched.
    let dim = x.t.len() + x.beta.len();
    let rank_floor = sae_ift_min_curvature_fraction();
    for _ in 0..dim {
        let ax = apply_a(&x)?;
        let bx = apply_b(&x)?;
        let x_b_norm_sq = sae_inner(&x, &bx);
        if x_b_norm_sq == 0.0 && sae_inner(&x, &x) == 0.0 {
            return Ok(x);
        }
        if !(x_b_norm_sq.is_finite() && x_b_norm_sq > 0.0) {
            return Err(format!(
                "solve_exact_stationarity: invalid B-norm squared {x_b_norm_sq:.6e}"
            ));
        }
        let mu = sae_inner(&x, &ax) / x_b_norm_sq;
        if !mu.is_finite() {
            return Err("solve_exact_stationarity: non-finite generalized curvature".into());
        }
        // #2253 — accept the solve when the solution's generalized curvature is
        // RESOLVED, i.e. `|μ| >= rank_floor`, NOT only when `μ >= rank_floor`.
        // `μ(x) ≈ μ_min` (the smallest-magnitude pencil eigenvalue excited by the
        // rhs), so `μ < 0` with `|μ|` well above the floor is a genuinely
        // NEGATIVE-curvature but fully IDENTIFIED direction (the exact Hessian
        // `A = B + ΔC` is marginally indefinite at a nonzero-residual fit — the
        // measured K=1-circle μ = −1.66e-3). Its `A⁻¹` response is a REAL, finite
        // part of `dθ̂/dρ = −A⁻¹ λSθ̂`, and the criterion VALUE's undamped inner
        // solve moves θ̂ along it identically — so the θ-adjoint −½Γᵀθ̂_ρ MUST keep
        // it or the analytic outer gradient desyncs from d(value)/dρ (the #2253
        // non-stationary stall: the adjoint collapsed ~19×, so steepest descent
        // could not decrease the criterion at its own minimum). Only a genuinely
        // SINGULAR direction (`|μ| < rank_floor`, spurious `1/μ` amplification of
        // an unidentified near-null) is deflated below — that one the evidence
        // factor also stiffens to unit curvature, so its outer-gradient
        // contribution is ρ-independent and must be projected out.
        if mu.abs() >= rank_floor {
            return Ok(x);
        }
        // Reaching here means `|μ| < rank_floor`: the solution is dominated by a
        // genuinely SINGULAR (numerically curvature-free) pencil direction, whose
        // `1/μ` amplification is an unidentifiable artifact, not a derivative. A
        // resolved indefinite direction (`μ < 0`, `|μ| ≥ rank_floor`) was already
        // returned above and is NOT deflated: the criterion value's `½log|B|`
        // uses the majorized joint factor `B`, which is fully PD along it (the
        // undamped inner solve SUCCEEDED, so `factor_spectral_deflated_evidence_
        // row` — which only stiffens non-PD PER-ROW blocks — never fired), so the
        // value genuinely depends on that direction and its `A⁻¹` IFT response is
        // a real part of the θ-adjoint. Only the singular direction handled below
        // is one the evidence factor would stiffen to unit curvature, so only its
        // response is spurious and must be projected out.
        // Sharpen the offending direction by inverse power iteration on
        // the pencil (`v ← A⁻¹(B v)`, B-normalized); the corrupted `x` is
        // already dominated by it, so it is the natural seed. Convergence
        // is certified by successive B-normalized direction alignment;
        // exhaustion or a failed inner solve propagates instead of silently
        // projecting with `v=x` (which would delete the entire response).
        let mut v = x.clone();
        let normalize_b = |v: &mut SaeArrowVector| -> Result<(), String> {
            let bv = apply_b(v)?;
            let norm_sq = sae_inner(v, &bv);
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
                return Err(format!(
                    "solve_exact_stationarity: inverse-power direction has invalid \
                     B-norm squared {norm_sq:.6e}"
                ));
            }
            let inv_norm = 1.0 / norm_sq.sqrt();
            v.t.mapv_inplace(|val| val * inv_norm);
            v.beta.mapv_inplace(|val| val * inv_norm);
            Ok(())
        };
        normalize_b(&mut v)?;
        let mut direction_converged = false;
        for _ in 0..dim {
            let bv = apply_b(&v)?;
            // #2253 — A⁻¹(Bv) is ILL-POSED along a near-null/indefinite pencil
            // direction (that is exactly the direction we are isolating), so the
            // refinement GMRES can legitimately exhaust its budget without
            // reaching tolerance. That is not a fatal error: the seed `v` is
            // already the B-normalized corrupted solution `x`, which — because
            // μ(x) collapsed onto μ_min — is ALREADY aligned with the offending
            // direction. Keep the best `v` and let the alignment/μ checks below
            // decide, instead of aborting the whole outer gradient.
            let refined = match solve_b_preconditioned_gmres_with(
                &bv,
                |w| apply_a(w),
                |w| precondition(w),
            ) {
                Ok(mut refined) => {
                    normalize_b(&mut refined)?;
                    refined
                }
                Err(_) => {
                    // Refinement stalled — the current `v` is our best isolate.
                    direction_converged = true;
                    break;
                }
            };
            let b_refined = apply_b(&refined)?;
            let alignment = sae_inner(&v, &b_refined).abs();
            if !alignment.is_finite() {
                return Err("solve_exact_stationarity: non-finite inverse-power alignment".into());
            }
            v = refined;
            // The discriminator asks whether the response's near-zero aggregate
            // Rayleigh quotient came from a numerical null or cancellation among
            // resolved pencil directions.  One inverse step amplifies smaller-|μ|
            // components relative to larger ones.  Therefore a refined direction
            // whose own curvature is already resolved proves the latter case; it
            // is unnecessary (and generally much slower) to wait for full
            // eigenvector alignment before keeping the original finite response.
            // Strict alignment remains mandatory below before a direction may be
            // projected as a numerical null.
            let av = apply_a(&v)?;
            let bv = apply_b(&v)?;
            let norm_sq = sae_inner(&v, &bv);
            if !(norm_sq.is_finite() && norm_sq > 0.0) {
                return Err(format!(
                    "solve_exact_stationarity: refined inverse-power direction has invalid \
                     B-norm squared {norm_sq:.6e}"
                ));
            }
            let refined_mu = sae_inner(&v, &av) / norm_sq;
            if !refined_mu.is_finite() {
                return Err(
                    "solve_exact_stationarity: refined inverse-power direction has non-finite \
                     generalized curvature"
                        .into(),
                );
            }
            if refined_mu.abs() >= rank_floor {
                return Ok(x);
            }
            if 1.0 - alignment.min(1.0) <= rank_floor {
                direction_converged = true;
                break;
            }
        }
        if !direction_converged {
            return Err(format!(
                "solve_exact_stationarity: inverse-power direction did not converge in the \
                 derived Krylov dimension {dim}"
            ));
        }
        // #2253 — deflate the isolated direction only when it is UNRESOLVED under
        // the exact pencil: `|μ|` below the numerical-null floor. A resolved
        // direction of either sign is a genuine finite part of the IFT response.
        // It can reach this branch when positive and negative resolved components
        // cancel in the solution's aggregate Rayleigh quotient; inverse iteration
        // then proves that no numerical null was present. In that case keep the
        // original exact solve instead of either deleting the resolved component
        // or turning benign Rayleigh cancellation into a typed failure.
        let av = apply_a(&v)?;
        let bv = apply_b(&v)?;
        let v_b_norm_sq = sae_inner(&v, &bv);
        if !(v_b_norm_sq.is_finite() && v_b_norm_sq > 0.0) {
            return Err(format!(
                "solve_exact_stationarity: converged inverse-power direction has invalid \
                 B-norm squared {v_b_norm_sq:.6e}"
            ));
        }
        let v_mu = sae_inner(&v, &av) / v_b_norm_sq;
        if !v_mu.is_finite() {
            return Err(format!(
                "solve_exact_stationarity: inverse power produced non-finite \
                 generalized curvature μ={v_mu:.6e}"
            ));
        }
        if v_mu.abs() >= rank_floor {
            return Ok(x);
        }
        let proj = sae_inner(&v, &bx);
        if proj == 0.0 || !proj.is_finite() {
            return Err(format!(
                "solve_exact_stationarity: invalid near-null B-projection {proj:.6e}"
            ));
        }
        x.t.scaled_add(-proj, &v.t);
        x.beta.scaled_add(-proj, &v.beta);
        log::debug!(
            "[SAE/#2080-d4] IFT solve deflated a near-null pencil direction \
             (μ={mu:.3e} < {rank_floor:.1e}, |proj|={:.3e})",
            proj.abs(),
        );
    }
    Err(format!(
        "solve_exact_stationarity: numerical-null deflation exhausted the derived \
         dimension {dim} without an identifiable IFT response"
    ))
}

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
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
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
                // #991 — the assembled `B` wrote the design-weighted majorizer
                // `w_row·D` into the logit block (see the assembly), and the exact
                // prior curvature is `w_row·H_entropy`, so this dropped-curvature
                // correction `ΔC = A − B = w_row·(H_entropy − D)` carries the SAME
                // `w_row`. The prior is weighted directly, not via the √w data seam.
                let w_row = row_loss_w.map_or(1.0, |w| w[row]);
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
                        acc += w_row * delta * v_t[b];
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
    /// on the closed-form gauge quotient via right-`B_Q`-preconditioned GMRES
    /// ([`solve_b_preconditioned_gmres`]) with the matrix-free
    /// `A_Q v = B v + ΔC v + κ Q Qᵀv` apply owned by
    /// [`solve_exact_stationarity_on_gauge_quotient`]. The
    /// IFT step `θ̂_ρ = −A⁻¹ g_ρ` (the code contracts `−½·⟨Γ, A⁻¹ g_ρ⟩` with rhs `= +∂g/∂ρ`, i.e. `+½·Γᵀθ̂_ρ` of the response — the sign lives in the −0.5 factor) must invert the EXACT `A`, not the surrogate `B`;
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
        let apply_raw_a = |v: &SaeArrowVector| self.apply_exact_hessian(rho, target, cache, v);
        let apply_raw_b =
            |v: &SaeArrowVector| apply_cached_arrow_hessian(cache, v.t.view(), v.beta.view());
        solve_exact_stationarity_on_gauge_quotient(solver, rhs, &apply_raw_a, &apply_raw_b)
    }

    /// Matrix-free exact-stationarity sibling used by the wide-border REML
    /// assignment-strength residual. `system` is the reassembled undamped
    /// bordered operator at the converged inner state; `cache` supplies the same
    /// row factors and H_tbeta operator whose rational log-determinant and shared
    /// inverse-probe bundle were consumed by the value/trace lanes.
    ///
    /// The reduced beta solve is quotient-aware and matrix-free. Per-row
    /// spectral deflation is refused by the selected-inverse channels before
    /// this seam is reached: a border-only probe bundle cannot differentiate
    /// the Daleckii-Krein deflation map, so proceeding would be a false exactness
    /// claim rather than a usable fallback.
    fn solve_exact_stationarity_matrix_free(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let (t, beta) = matrix_free_arrow_operator_apply(
                system,
                cache,
                vector.t.view(),
                vector.beta.view(),
            )
            .map_err(|error| format!("matrix-free evidence operator: {error}"))?;
            Ok(SaeArrowVector { t, beta })
        };
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let base = apply_b(vector)?;
            let correction = self.apply_exact_hessian_minus_b(rho, target, cache, vector)?;
            Ok(SaeArrowVector {
                t: &base.t + &correction.t,
                beta: &base.beta + &correction.beta,
            })
        };
        let precondition = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            // The outer exact-stationarity residual is certified to 1e-10 in
            // `solve_b_preconditioned_gmres`; drive its deterministic SPD
            // reduced preconditioner to the same relative accuracy. In exact
            // arithmetic CG terminates in at most the reduced dimension, so the
            // dimension itself is the non-arbitrary iteration bound.
            let (t, beta) = matrix_free_arrow_inverse_apply(
                system,
                cache,
                vector.t.view(),
                vector.beta.view(),
                1.0e-10,
                cache.k.max(1),
            )
            .map_err(|error| format!("matrix-free evidence inverse: {error}"))?;
            Ok(SaeArrowVector { t, beta })
        };
        solve_exact_stationarity_preconditioned(rhs, &apply_a, &apply_b, precondition)
    }

    fn combine_assignment_strength_gradient(
        &self,
        rho: &SaeManifoldRho,
        logdet_trace: f64,
        gamma: &SaeArrowVector,
        response: &SaeArrowVector,
        lane: &str,
    ) -> Result<f64, OuterGradientError> {
        let explicit = crate::assignment::assignment_prior_log_strength_derivative_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        );
        let correction = -0.5 * sae_inner(gamma, response);
        let gradient = explicit + logdet_trace + correction;
        if !gradient.is_finite() {
            return Err(OuterGradientError::internal(format!(
                "{lane} assignment-strength gradient is non-finite: explicit={explicit}, \
                 logdet_trace={logdet_trace}, IFT_correction={correction}"
            )));
        }
        Ok(gradient)
    }

    /// Dense-cache sibling of
    /// [`Self::analytic_assignment_strength_gradient_matrix_free`]. Hybrid-EFS
    /// uses this when the full streaming working set is not resident but the
    /// reduced Schur itself still fits and the returned cache therefore owns an
    /// exact dense factor. Computing only this coordinate avoids the O(K) IFT
    /// solves of the complete outer gradient while retaining identical math.
    pub(crate) fn analytic_assignment_strength_gradient_dense(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<f64, OuterGradientError> {
        let sparse_index = rho.sparse_flat_index().ok_or_else(|| {
            OuterGradientError::internal(
                "dense assignment-strength gradient requested for a rho with no sparse coordinate",
            )
        })?;
        let joint_trace = self
            .assignment_log_strength_hessian_trace(rho, cache, solver)
            .map_err(OuterGradientError::internal)?;
        let coordinate_trace = self
            .coordinate_block_assignment_log_strength_hessian_trace(rho, cache)
            .map_err(OuterGradientError::internal)?;
        let logdet_trace = joint_trace - coordinate_trace;
        let loss = self
            .loss(target, rho)
            .map_err(OuterGradientError::internal)?;
        let rank_charge = self
            .hard_rank_charge_derivative(target, rho, &loss, cache)
            .map_err(OuterGradientError::internal)?;
        let mut gamma = self
            .logdet_theta_adjoint(rho, cache, solver)
            .map_err(OuterGradientError::internal)?;
        let coordinate_gamma = self
            .coordinate_block_logdet_theta_adjoint(rho, cache, solver)
            .map_err(OuterGradientError::internal)?;
        gamma.t -= &coordinate_gamma.t;
        gamma.beta -= &coordinate_gamma.beta;
        gamma.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
        let rhs = self
            .outer_rho_gradient_ift_rhs(rho, sparse_index, cache)
            .map_err(OuterGradientError::internal)?;
        let response = self
            .solve_exact_stationarity(rho, target, cache, solver, &rhs)
            .map_err(|error| {
                OuterGradientError::classify_arrow_solver_error(
                    &error,
                    OuterGradientError::NonIdentifiable {
                        reason: error.clone(),
                    },
                )
            })?;
        self.combine_assignment_strength_gradient(rho, logdet_trace, &gamma, &response, "dense")
    }

    /// Exact non-IBP assignment-strength penalized-LAML gradient on the matrix-free
    /// evidence path. This is the one coordinate softmax entropy and gated L1
    /// cannot update through a Fellner-Schall equation:
    ///
    /// `dV/drho_sparse = explicit_prior + 0.5 tr(B^-1 dB/drho_sparse)
    ///                    - 0.5 Gamma^T A^-1 dg/drho_sparse`.
    ///
    /// Every selected-inverse contraction uses the same `(z, S^-1 z)` bundle
    /// the rational log-determinant emitted, and the implicit response uses the
    /// matrix-free exact-stationarity solve above. No dense Schur, finite
    /// difference, held-zero surrogate, or degraded derivative is involved.
    pub(crate) fn analytic_assignment_strength_gradient_matrix_free(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        probes: &[Array1<f64>],
        inverse_probes: &[Array1<f64>],
    ) -> Result<f64, OuterGradientError> {
        let sparse_index = rho.sparse_flat_index().ok_or_else(|| {
            OuterGradientError::internal(
                "matrix-free assignment-strength gradient requested for a rho with no sparse coordinate",
            )
        })?;
        let joint_trace = self
            .assignment_log_strength_hessian_trace_from_probes(rho, cache, probes, inverse_probes)
            .map_err(OuterGradientError::internal)?;
        let coordinate_trace = self
            .coordinate_block_assignment_log_strength_hessian_trace(rho, cache)
            .map_err(OuterGradientError::internal)?;
        let logdet_trace = joint_trace - coordinate_trace;
        let plain_solver = DeflatedArrowSolver::plain(cache);
        let loss = self
            .loss(target, rho)
            .map_err(OuterGradientError::internal)?;
        let rank_charge = self
            .hard_rank_charge_derivative(target, rho, &loss, cache)
            .map_err(OuterGradientError::internal)?;
        let mut gamma = self
            .logdet_theta_adjoint_from_probes(rho, cache, probes, inverse_probes)
            .map_err(OuterGradientError::internal)?;
        let coordinate_gamma = self
            .coordinate_block_logdet_theta_adjoint(rho, cache, &plain_solver)
            .map_err(OuterGradientError::internal)?;
        gamma.t -= &coordinate_gamma.t;
        gamma.beta -= &coordinate_gamma.beta;
        gamma.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
        let rhs = self
            .outer_rho_gradient_ift_rhs(rho, sparse_index, cache)
            .map_err(OuterGradientError::internal)?;
        let response = self
            .solve_exact_stationarity_matrix_free(rho, target, cache, system, &rhs)
            .map_err(|error| {
                OuterGradientError::classify_arrow_solver_error(
                    &error,
                    OuterGradientError::NonIdentifiable {
                        reason: error.clone(),
                    },
                )
            })?;
        self.combine_assignment_strength_gradient(
            rho,
            logdet_trace,
            &gamma,
            &response,
            "matrix-free",
        )
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
    /// The complete all-coordinate assembler remains dense-solver-bound: its IFT
    /// correction requires one exact-stationarity solve per active outer
    /// coordinate and still accepts a [`DeflatedArrowSolver`]. Production
    /// Hybrid-EFS deliberately does not route the scalable fit through that O(K)
    /// surface. It evaluates the sole non-FS assignment-strength coordinate with
    /// [`Self::analytic_assignment_strength_gradient_matrix_free`] and leaves the
    /// simultaneous smoothness/ARD block on Fellner-Schall updates. The `Some`
    /// branch here is retained for exact dense/bundle parity of the complete
    /// derivative, not as the production matrix-free route.
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
        let rank_charge = self
            .hard_rank_charge_derivative(target, rho, loss, cache)
            .map_err(OuterGradientError::internal)?;

        if let Some(sparse_index) = rho.sparse_flat_index() {
            explicit[sparse_index] =
                crate::assignment::assignment_prior_log_strength_derivative_weighted(
                    &self.assignment,
                    rho,
                    self.row_loss_weights.as_deref(),
                );
            // IBP concentration controls only the Beta--Bernoulli prior. The
            // final posterior-mean gate is `sigmoid(logit/tau)`, so the data
            // likelihood and its Gauss--Newton blocks have no direct alpha
            // derivative. Structurally fixed assignments have no sparse index
            // and skip this channel entirely.
            let joint_trace = match inverse_probe_bundle {
                Some((probes, sinv)) => self
                    .assignment_log_strength_hessian_trace_from_probes(rho, cache, probes, sinv)
                    .map_err(OuterGradientError::internal)?,
                None => self
                    .assignment_log_strength_hessian_trace(rho, cache, solver)
                    .map_err(OuterGradientError::internal)?,
            };
            let coordinate_trace = self
                .coordinate_block_assignment_log_strength_hessian_trace(rho, cache)
                .map_err(OuterGradientError::internal)?;
            logdet_trace[sparse_index] = joint_trace - coordinate_trace;
        }

        // #1556: λ_smooth is per-atom, so the smoothness gradient block occupies
        // the K layout-derived smooth indices (one per atom). Each atom
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
            let index = rho.smooth_flat_index(atom_idx);
            explicit[index] = smooth_explicit[atom_idx];
            logdet_trace[index] = 0.5 * smooth_logdet[atom_idx];
            occam[index] = -smooth_occam[atom_idx];
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
        let ard_joint_trace = match inverse_probe_bundle {
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
        let ard_coordinate_trace = self
            .coordinate_block_ard_log_precision_hessian_trace(rho, cache)
            .map_err(|err| OuterGradientError::InternalInvariant {
                reason: format!(
                    "analytic_outer_rho_gradient_components: coordinate-block ARD trace: {err}"
                ),
            })?;
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
                logdet_trace[idx] += ard_joint_trace[k][axis] - ard_coordinate_trace[k][axis];
            }
        }

        // The scalar criterion replaces `½ log|H_tt|` with the realised-rank
        // charge. Its direct rho differential belongs alongside the explicit
        // penalty channels and is present on every layout (dense or probes).
        explicit += &rank_charge.direct_rho;

        // #2080: the envelope Γ = tr(H⁻¹ ∂H/∂θ) off the SAME shared selected-inverse
        // bundle (the all-or-nothing cluster's third channel) when present; the dense
        // selected inverse otherwise. The border-only bundle reconstructs the NO-SELF
        // base inverse `(H₀')⁻¹`, so `logdet_theta_adjoint_from_probes` HARD-REFUSES
        // (routes to dense) any cache carrying a T-space rank-R correction the border
        // cannot span — per-row gauge/rotation deflation OR an IBP cross-row Woodbury —
        // and otherwise owns the softmax / euclidean / non-cross-row regimes exactly.
        // This completes the matrix-free selected-inverse cluster (smoothness EDF + ARD
        // Hessian trace + θ-adjoint); the assignment/learnable-IBP log-strength traces
        // (when that coordinate exists) plus the θ-adjoint's IBP-refused fits remain
        // solver-bound
        // — the last gaps before the routing flip (see the docstring).
        let mut gamma = match inverse_probe_bundle {
            Some((probes, sinv)) => self
                .logdet_theta_adjoint_from_probes(rho, cache, probes, sinv)
                .map_err(OuterGradientError::internal)?,
            None => self
                .logdet_theta_adjoint(rho, cache, solver)
                .map_err(OuterGradientError::internal)?,
        };
        let coordinate_gamma = self
            .coordinate_block_logdet_theta_adjoint(rho, cache, solver)
            .map_err(OuterGradientError::internal)?;
        gamma.t -= &coordinate_gamma.t;
        gamma.beta -= &coordinate_gamma.beta;
        // `½ Γ_joint·theta_hat - ½ Γ_tt·theta_hat + ∇R·theta_hat`
        // is represented by one effective logdet adjoint
        // `Γ_eff = Γ_joint - Γ_tt + 2∇R`, preserving the existing
        // `-½ <Γ_eff, A^-1 g_rho>` contraction convention below.
        gamma.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
        // #1418: the implicit-function correction is `−½·Γᵀ·θ̂_ρ` with
        // `θ̂_ρ = −A⁻¹ g_ρ` (the code contracts `−½·⟨Γ, A⁻¹ g_ρ⟩` with rhs `= +∂g/∂ρ`, i.e. `+½·Γᵀθ̂_ρ` of the response — the sign lives in the −0.5 factor), where `A = ∇²_θθ L` is the EXACT stationarity
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
        // #2231 — the trailing `L−1` flat coordinates are the crosscoder block
        // relevances `log λ_ℓ` (`SaeManifoldRho::to_flat` appends them last).
        // Their inner-gradient dependence enters through the λ-scaled target, so
        // their RHS is `−½·Jᵀ_M Z̃^{(ℓ)}` (`crosscoder_block_ift_rhs`), NOT the
        // penalty/prior channels `outer_rho_gradient_ift_rhs` owns. The adjoint
        // contraction below then completes the block gradient with the same
        // `−½·Γᵀθ̂_ρ` channel every other coordinate carries; the explicit data
        // + Jacobian parts stay with the eval lane's `block_log_lambda_gradient`.
        let block_tail_start = n_params - rho.log_lambda_block.len();
        for coord in 0..n_params {
            let rhs = if coord >= block_tail_start && !rho.log_lambda_block.is_empty() {
                let &(p_x, ref block_dims) =
                    self.crosscoder_pricing_spans.as_ref().ok_or_else(|| {
                        OuterGradientError::internal(
                            "analytic_outer_rho_gradient_components: rho carries block \
                             coordinates but no crosscoder pricing spans are installed"
                                .to_string(),
                        )
                    })?;
                let block = coord - block_tail_start;
                let start = p_x + block_dims[..block].iter().sum::<usize>();
                self.crosscoder_block_ift_rhs(cache, target, start..start + block_dims[block])
                    .map_err(OuterGradientError::internal)?
            } else {
                self.outer_rho_gradient_ift_rhs(rho, coord, cache)
                    .map_err(OuterGradientError::internal)?
            };
            let solved = self
                .solve_exact_stationarity(rho, target, cache, solver, &rhs)
                .map_err(|err| {
                    OuterGradientError::classify_arrow_solver_error(
                        &err,
                        OuterGradientError::NonIdentifiable {
                            reason: err.clone(),
                        },
                    )
                })?;
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
