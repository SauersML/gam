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
/// below this floor (a saturated ordered Beta--Bernoulli gate logit has data
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
        // is one the criterion factor would stiffen to unit curvature, so only its
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
            let refined =
                match solve_b_preconditioned_gmres_with(&bv, |w| apply_a(w), |w| precondition(w)) {
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
    /// to a joint `(t, β)` vector, matrix-free via row-local work and ordered
    /// prior column reductions.
    ///
    /// `A = ∇²_θθ L` is the true inner-fit Hessian; `B` is the assembled
    /// evidence/Newton operator the solver factors. They differ only by the four
    /// curvature substitutions the assembly makes for stability:
    ///   1. data: `B` uses Gauss-Newton `J̃J̃ᵀ`, dropping the residual curvature
    ///      `R[a,b] = Σ_out r_out·∂²f_out/∂θ_a∂θ_b` (t–t via `jets.second`, t–β via
    ///      `jets.beta_deriv`; the decoder is linear in β so the β–β block is 0);
    ///   2. softmax: `B` uses the Gershgorin majorizer `D = diag(Σ_j|H_kj|)`,
    ///      dropping `H_entropy − D` (#1419);
    ///   3. periodic ARD: `B` uses `max(V'',0)`, dropping the negative part
    ///      `min(V'',0)` (the indefinite tail past a quarter period).
    ///   4. ordered Beta--Bernoulli: `B` uses the positive row-local diagonal
    ///      majorizer and drops both the exact negative active-mass rank-one term
    ///      and every nonpositive row-local diagonal contribution.
    /// `ΔC` is the sum of exactly these four deltas, each built from the same
    /// jets / penalty curvatures the assembly and the θ-adjoint use, so
    /// `A = B + ΔC` is the one true Hessian. Exact on BOTH the isotropic and the
    /// whitened-metric paths: the data fit is `½ r_nᵀ M_n r_n`, so the residual
    /// curvature is `Σ_out (M_n r_n)_out·∂²f_out/∂θ_a∂θ_b` — contract the
    /// metric-applied √w-scaled residual `error_metric = √w·M_n r_n` (the SAME
    /// quantity the assembly's β-tier gradient uses) against the RAW second jets
    /// `jets.second`/`jets.beta_deriv` (the same raw-jet convention the whole
    /// θ-adjoint and the Gauss-Newton `htt = J̃J̃ᵀ = J M Jᵀ` assembly use). On the
    /// isotropic path `M_n = I` so `error_metric = √w·r` and `J M Jᵀ = JJᵀ`,
    /// recovering the plain case. The softmax, ordered Beta--Bernoulli, and ARD
    /// deltas are logit/coord-space prior curvatures and carry no output metric,
    /// so they are path-independent.
    pub(crate) fn apply_exact_hessian_minus_b(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        self.assignment.validate_rho_domain(rho)?;
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
        let ard_precisions = self.validated_ard_precisions(rho)?;

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
                let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
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
        // Ordered Beta--Bernoulli's exact prior Hessian couples all rows within
        // each atom column. Gather the logit slice of `v` while visiting the
        // row-local cache layout, then apply the analytic column reductions once
        // after the row loop. This remains O(NK) memory/time and constructs no
        // dense cross-row matrix or persistent carrier.
        let mut ordered_logit_direction = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        )
        .then(|| Array1::<f64>::zeros(n * k_atoms));
        // #932 complete schedule: softmax rows are built in memory-ledgered
        // CPU/CUDA tiles through one bounded look-ahead window; non-softmax
        // gates use their distinct dynamic row program.
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
            if let Some(direction) = ordered_logit_direction.as_mut() {
                for (local, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        direction[row * k_atoms + atom] = v_t[local];
                    }
                }
            }

            // (1a) residual curvature, t–t: ΔC_tt[a,b] = ⟨r, ∂²f_ab⟩.
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = sae_dot(&error_metric, jets.second(a, b));
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            // (1b) residual curvature, t–β and β–t: ΔC_tβ[a,β] = ⟨r, ∂²f_aβ⟩.
            //      `jets.beta_deriv[a][β]` = ∂(∂f/∂β_β)/∂θ_a (the mixed second jet).
            for a in 0..q {
                for (beta_pos, channel) in border.iter().enumerate() {
                    let r_ab = sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
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
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    out.t[base + a] += w_row * neg * v_t[a];
                }
            }
        }

        // (4) ordered Beta--Bernoulli: exact integrated-marginal Hessian minus
        // the diagonal PSD majorizer written into B. The helper evaluates the
        // negative within-column rank-one action by column reductions and the
        // row-local diagonal remainder directly, then we scatter its flat logit
        // result back into the cache's row-local coordinates.
        if let Some(direction) = ordered_logit_direction {
            let delta = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                &self.assignment,
                rho,
                row_loss_w,
                direction.view(),
            )?;
            for row in 0..n {
                let base = cache.row_offsets[row];
                let vars = self.row_vars_for_cache_row(row, cache)?;
                for (local, var) in vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        out.t[base + local] += delta[row * k_atoms + atom];
                    }
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

    /// Matrix-free exact-stationarity sibling used by the wide-border penalized quasi-Laplace
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

    /// Analytic SAE penalized quasi-Laplace outer-ρ gradient components at the already converged
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
            target, rho, loss, cache, solver, None, None,
        )
    }

    /// #2080 forward plumbing — the analytic outer-ρ gradient with an OPTIONAL
    /// low-rank representation of the reduced-logdet derivative.
    ///
    /// When `logdet_derivative_bundle` is `Some`, the THREE reduced-logdet channels
    /// that have matrix-free siblings — the per-atom decoder smoothness EDF
    /// `tr(H⁻¹ M_k)`, the per-(atom,axis) ARD log-precision Hessian trace
    /// `½tr(H⁻¹ ∂H/∂logα)`, and the #1006 envelope Γ = tr(H⁻¹ ∂H/∂θ) — are evaluated
    /// off that bundle (`decoder_smoothness_effective_dof_per_atom_from_probes` /
    /// `ard_log_precision_hessian_trace_from_probes` / `logdet_theta_adjoint_from_probes`)
    /// instead of the dense `DeflatedArrowSolver` selected inverse. For the
    /// rational route the two slices are the identical weighted vectors emitted
    /// by `RationalLogdetPlan::into_directional_derivative_bundle`, so every
    /// contraction is the derivative of the SAME shifted rational value, not a
    /// separately sampled `S^-1`. They convert
    /// together as ONE all-or-nothing cluster on the single `Some` (invariant #1):
    /// never a partial mix within a single eval. Each from-probes channel hard-refuses
    /// deflated rows (the plain-S⁻¹ bundle cannot reconstruct the Daleckii–Krein
    /// correction), routing those fits to the dense channel.
    ///
    /// The complete all-coordinate assembler is single-adjoint (#2080-A): the IFT
    /// correction `−½·⟨Γ, A⁺ g_ρ_l⟩` over every outer coordinate collapses to ONE
    /// exact-stationarity solve `a = A⁺Γ` plus O(K) cheap `⟨a, g_ρ_l⟩`
    /// contractions (self-adjointness of `A⁺`; see the collapse below). That
    /// single adjoint solve is the ONLY solver-bound step, so the whole assembler
    /// runs matrix-free at massive K: pass `matrix_free_system = Some(system)` to
    /// route it through [`Self::solve_exact_stationarity_matrix_free`] (the
    /// reduced-Schur CG on the reassembled undamped operator) with
    /// `solver = DeflatedArrowSolver::plain(cache)` for the cheap per-row
    /// `coordinate_block_*` subtractions — the K≥4096, direct-logdet-not-admitted
    /// route, mirroring the matrix-free branch of this complete assembler.
    /// Pass `matrix_free_system = None` to use the dense [`DeflatedArrowSolver`]
    /// adjoint (the direct-logdet-admitted route). Both produce the same complete
    /// derivative; the from-probes trace channels and the matrix-free adjoint
    /// convert together as one all-or-nothing matrix-free cluster (invariant #1).
    pub(crate) fn analytic_outer_rho_gradient_components_with_bundle(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        logdet_derivative_bundle: Option<(&[Array1<f64>], &[Array1<f64>])>,
        matrix_free_system: Option<&ArrowSchurSystem>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(OuterGradientError::internal)?;
        let n_params = rho.to_flat().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);
        let rank_charge = self
            .production_rank_charge_derivative(target, rho, loss, cache)
            .map_err(OuterGradientError::internal)?;

        if let Some(sparse_index) = rho.sparse_flat_index() {
            explicit[sparse_index] =
                crate::assignment::assignment_prior_log_strength_derivative_weighted(
                    &self.assignment,
                    rho,
                    self.row_loss_weights.as_deref(),
                )
                .map_err(OuterGradientError::internal)?;
            // ordered Beta--Bernoulli concentration controls only the Beta--Bernoulli prior. The
            // final reconstruction gate is `sigmoid(logit/tau)`, so the data
            // likelihood and its Gauss--Newton blocks have no direct alpha
            // derivative. Structurally fixed assignments have no sparse index
            // and skip this channel entirely.
            let joint_trace = match logdet_derivative_bundle {
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
        let lambda_smooth_vec = rho
            .lambda_smooth_vec()
            .map_err(OuterGradientError::internal)?;
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
        // #2080: the per-atom smoothness logdet derivative off the shared
        // low-rank derivative representation when the rational lane supplied it;
        // the dense `DeflatedArrowSolver` selected inverse otherwise.
        let smooth_logdet = match logdet_derivative_bundle {
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
        // #2080: the per-(atom,axis) ARD log-precision Hessian derivative off the
        // SAME shared low-rank representation (the all-or-nothing cluster's
        // second channel) when present; the dense
        // deflated selected inverse otherwise. The from-probes channel HARD-REFUSES
        // any row carrying gauge/rotation deflation (the plain-S⁻¹ bundle cannot
        // reconstruct the Daleckii–Krein correction), routing that fit to the dense
        // channel rather than silently dropping the correction.
        let ard_joint_trace = match logdet_derivative_bundle {
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

        // #2080: the envelope Γ off the SAME shared low-rank logdet derivative
        // representation (the all-or-nothing cluster's third channel) when
        // present; the dense
        // selected inverse otherwise. The border-only bundle reconstructs the NO-SELF
        // base derivative on the undeflated row chart, so
        // `logdet_theta_adjoint_from_probes` hard-refuses
        // (routes to dense) a cache carrying a T-space gauge/rotation deflation
        // that the border probes cannot span. Ordered Beta--Bernoulli uses its
        // row-local PSD majorizer and shared-mass derivative directly.
        // This completes the matrix-free selected-inverse cluster (smoothness EDF + ARD
        // Hessian trace + θ-adjoint); assignment log-strength traces remain
        // solver-bound
        // — the last gaps before the routing flip (see the docstring).
        let mut gamma = match logdet_derivative_bundle {
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
        // entropy Hessian, exact ordered Beta--Bernoulli marginal curvature, and
        // exact periodic ARD curvature. The matrix the `solver`
        // factors is `B` (Gauss-Newton data curvature, the softmax Gershgorin
        // majorizer, the ordered Beta--Bernoulli row-local PSD majorizer, and
        // `max(V'',0)` ARD curvature): the `½log|B|` Laplace term is consistent
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
        // #2080(A): collapse the per-coordinate IFT solves into ONE adjoint solve.
        // The implicit correction is `−½·⟨Γ, A⁺ g_ρ_l⟩` for every outer coordinate
        // `l`. The exact θθ-Hessian `A = ∇²_θθ L` is symmetric and its near-null
        // deflation is a symmetric `B`-orthogonal projection, so `A⁺` is
        // self-adjoint and `⟨Γ, A⁺ g_ρ_l⟩ = ⟨A⁺Γ, g_ρ_l⟩ = ⟨a, g_ρ_l⟩` with the
        // adjoint `a = A⁺Γ` solved ONCE. A near-null pencil direction contributes
        // `g_i r_i / μ_i` only when BOTH Γ and `g_ρ_l` excite it, in which case the
        // forward (per-coordinate) and this adjoint solve deflate it identically —
        // so the collapse is EXACT, not an approximation, while dropping the outer
        // IFT cost from `O(P_ρ)` solves to one. `solve_exact_stationarity_is_self_adjoint_2080`
        // pins the self-adjointness this identity rests on.
        // The single adjoint solve `a = A⁺Γ` — the only solver-bound step. At
        // massive K (`matrix_free_system = Some`) it rides the reduced-Schur CG on
        // the reassembled undamped operator; otherwise the dense deflated arrow
        // solver. Both realize the same self-adjoint `A⁺` action.
        let adjoint = match matrix_free_system {
            Some(system) => {
                self.solve_exact_stationarity_matrix_free(rho, target, cache, system, &gamma)
            }
            None => self.solve_exact_stationarity(rho, target, cache, solver, &gamma),
        }
        .map_err(|err| {
            OuterGradientError::classify_arrow_solver_error(
                &err,
                OuterGradientError::NonIdentifiable {
                    reason: err.clone(),
                },
            )
        })?;
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
            let mut dot = 0.0_f64;
            for idx in 0..adjoint.t.len() {
                dot += adjoint.t[idx] * rhs.t[idx];
            }
            for idx in 0..adjoint.beta.len() {
                dot += adjoint.beta[idx] * rhs.beta[idx];
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

    /// PATH C channel — exact fixed-stratum second derivative of the SOLVER-FREE
    /// explicit outer-gradient channels: the decoder-smoothness penalty energy
    /// (with its Occam renormalization to `loss.smoothness`) and the ARD
    /// log-precision prior. The rank-charge `direct_rho`, assignment
    /// log-strength, log-determinant traces, and third-order IFT channels are
    /// each assembled by their own methods; this one covers only the two
    /// channels that are closed forms of ρ at a frozen inner state (`atoms`,
    /// `assignment`) and touch no `H⁻¹`/`A⁺` solve, so it needs no cache.
    ///
    /// Math (all at fixed stratum, `s = log α`, `f_k = ⟨B_k, S_k B_k⟩` frozen):
    /// * Smoothness. The gradient renormalizes the per-atom penalty energy
    ///   `se_k = ½ λ_k f_k` to the frozen scalar `C = loss.smoothness`, i.e.
    ///   `g_k = C · se_k / Σ_m se_m`. With `∂se_k/∂ρ_j = δ_{jk} se_k`, holding `C`
    ///   frozen gives the symmetric rank-structured block
    ///   `∂²/∂ρ_i∂ρ_j = (C/Σ)·(δ_{ij} se_i − se_i se_j / Σ)` — NOT diagonal: the
    ///   shared normalizer couples every pair of smoothing atoms. (A zero `Σ`
    ///   leaves the energy unrenormalized, `g_k = se_k`, second derivative the
    ///   plain diagonal `δ_{ij} se_i`.)
    /// * ARD. Per `(atom, axis)` the gradient is `energy_deriv + normalizer_deriv`
    ///   with `energy_deriv = Σ_i w_i · V(α, t_i)` (degree-one in `α`, so its own
    ///   `∂/∂s` is itself) and a normalizer that is `−½ n_eff` (constant → zero)
    ///   on a Euclidean axis and `n_eff · d1(log η)` on a periodic axis,
    ///   `log η = log α + 2(log p − log τ)`. The periodic second derivative is
    ///   `energy_deriv + n_eff · c''(log η)` with `c''` the stable
    ///   [`gam_math::special::bessel_i0_centered_second_log_derivative_from_log_abs`].
    ///   ARD axes are independent (diagonal); a shared-ARD coordinate owned by
    ///   several atoms accumulates their diagonals, matching the gradient's `+=`.
    /// * Occam. `reml_occam_log_lambda_smooth_derivative` is ρ-independent → zero.
    ///
    /// `frozen_smoothness_energy` is the criterion's reported `loss.smoothness`
    /// at the fixed stratum (`Σ_m se_m` on the full-batch path; a minibatch
    /// `penalty_scale` folded into it is preserved by the `C/Σ` renormalization).
    pub(crate) fn outer_explicit_smoothness_ard_hessian(
        &self,
        rho: &SaeManifoldRho,
        frozen_smoothness_energy: f64,
    ) -> Result<Array2<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n_params = rho.to_flat().len();
        let mut hessian = Array2::<f64>::zeros((n_params, n_params));

        // Decoder-smoothness penalty energy with its Occam renormalization.
        let lambda_smooth = rho.lambda_smooth_vec()?;
        let smooth_energy = self.decoder_smoothness_value_per_atom(&lambda_smooth);
        let energy_sum: f64 = smooth_energy.iter().sum();
        let k_smooth = rho.log_lambda_smooth.len();
        if energy_sum.abs() > 0.0 {
            let renorm = frozen_smoothness_energy / energy_sum;
            for a in 0..k_smooth {
                let ia = rho.smooth_flat_index(a);
                for b in 0..k_smooth {
                    let ib = rho.smooth_flat_index(b);
                    let diagonal = if a == b { smooth_energy[a] } else { 0.0 };
                    hessian[[ia, ib]] +=
                        renorm * (diagonal - smooth_energy[a] * smooth_energy[b] / energy_sum);
                }
            }
        } else {
            for a in 0..k_smooth {
                let ia = rho.smooth_flat_index(a);
                hessian[[ia, ia]] += smooth_energy[a];
            }
        }

        // ARD log-precision prior (diagonal per coordinate; shared axes sum).
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs() as f64;
        let n_eff = row_w.map_or(n, |w| w.iter().sum::<f64>());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            if rho.log_ard[atom_idx].is_empty() {
                continue;
            }
            let periods = coord.effective_axis_periods();
            for axis in 0..coord.latent_dim() {
                let alpha = ard_precisions[atom_idx][axis];
                let log_alpha = rho.log_ard[atom_idx][axis];
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for row in 0..coord.n_obs() {
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let t = coord.row(row)[axis];
                    energy_deriv += w_row * ArdAxisPrior::eval(alpha, t, period).value;
                }
                let normalizer_second = match period {
                    None => 0.0,
                    Some(p) => {
                        let log_eta =
                            log_alpha + 2.0 * (p.ln() - std::f64::consts::TAU.ln());
                        n_eff
                            * gam_math::special::bessel_i0_centered_second_log_derivative_from_log_abs(
                                log_eta,
                            )
                    }
                };
                let idx = rho.ard_flat_index(atom_idx, axis);
                hessian[[idx, idx]] += energy_deriv + normalizer_second;
            }
        }

        // Sparse (assignment log-strength). For softmax the gradient's
        // `explicit[sparse]` is `assignment_prior_log_strength_derivative_weighted`
        // = the prior VALUE = `λ_sparse · E(logits)` (assignment.rs:1690), which is
        // degree-one in `λ_sparse = e^ρ_sparse` (the concentration multiplies the
        // logit penalty linearly). So `∂²/∂ρ_sparse² = ∂/∂ρ_sparse(λ_sparse·E) =
        // λ_sparse·E` — the SAME scalar the gradient reports — and there is no cross
        // term (it depends only on `λ_sparse` and the frozen logits, not on
        // smooth/ARD). K=1 softmax and frozen routing return 0, so the diagonal is
        // correctly zero there.
        if let Some(sparse_index) = rho.sparse_flat_index() {
            match self.assignment.mode {
                AssignmentMode::Softmax { .. } => {
                    hessian[[sparse_index, sparse_index]] +=
                        crate::assignment::assignment_prior_log_strength_derivative_weighted(
                            &self.assignment,
                            rho,
                            self.row_loss_weights.as_deref(),
                        )?;
                }
                _ => {
                    return Err(
                        "outer_explicit_smoothness_ard_hessian: rho carries a sparse \
                         log-strength coordinate under a non-softmax assignment prior, whose \
                         explicit second derivative this channel does not yet model; refusing \
                         to assemble a Hessian with a silently-zero sparse explicit term"
                            .to_string(),
                    );
                }
            }
        }

        Ok(hessian)
    }

    /// PATH C channel 4 — exact fixed-stratum second derivative of the outer
    /// gradient's log-determinant Daleckii–Krein trace channel (`logdet_trace`).
    ///
    /// The gradient's `logdet_trace` component is, per outer coordinate `i`,
    /// `logdet_trace_i = ½·[tr(G Cᵢ) − tr(H_bd⁻¹ Cᵢ)]`, where `Cᵢ = ∂H/∂ρ_i` is
    /// the penalty curvature the coordinate scales, `G = H⁻¹` is the FULL joint
    /// arrow inverse (the `ard_joint` / smoothness-EDF selected inverse), and
    /// `H_bd⁻¹` is the block-diagonal per-row `H_tt` inverse the rank-charge
    /// coordinate block subtracts (`ard_coordinate` trace). The smoothing channel
    /// touches only `H_ββ`, so its `H_bd⁻¹` leg is identically zero; the periodic
    /// ARD channel touches only the row-local `t`-slots, so both legs contribute.
    ///
    /// Every operator `Cᵢ` is degree-one in `exp(ρ_i)` at a frozen inner state —
    /// `λ_k·S_k ⊗ I` on the β-block for smoothing; `w_row·max(α cos κt, 0)` on the
    /// active `t`-rows for periodic ARD (`w_row·α` for a Euclidean axis). The
    /// `max(·,0)` majorizer active set is invariant under a ρ perturbation because
    /// ρ scales only `α`, never the frozen coordinate `t`. Hence
    /// `∂Cᵢ/∂ρ_j = δ_{ij} Cᵢ` and, with the Daleckii–Krein differential
    /// `∂G/∂ρ_j = −G C_j G` for each inverse `G`,
    /// `block[i,j] = ½·δ_{ij}·(tr(G Cᵢ) − tr(H_bd⁻¹ Cᵢ))
    ///              − ½·(tr(G C_j G Cᵢ) − tr(H_bd⁻¹ C_j H_bd⁻¹ Cᵢ))`.
    /// The diagonal `δ` term is exactly the coordinate's own `logdet_trace_i`
    /// value (the "self-term equals the operator" identity). A smoothing `C_j`
    /// vanishes on `H_bd⁻¹` (t-only) and an ARD `C_i` couples to a smoothing `C_j`
    /// only through the FULL inverse's `t`–β block, matching the gradient's
    /// construction.
    ///
    /// Small-dense materialization: build `G` dense by solving the arrow system
    /// against each unit arrow basis vector (`DeflatedArrowSolver::plain`), and
    /// `H_bd⁻¹` from the per-row undamped Cholesky factors — the same two inverses
    /// the gradient's `ard_joint` / `ard_coordinate` legs use, so value, gradient,
    /// and this Hessian share one (deflation-free interior) selected inverse.
    /// Shared-ARD axes accumulate their per-atom operators into one flat
    /// coordinate, matching the gradient's chain-rule `+=`.
    pub(crate) fn logdet_daleckii_krein_hessian(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n_params = rho.to_flat().len();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let solver = DeflatedArrowSolver::plain(cache);

        // Full joint inverse G = H⁻¹ (dim×dim), materialized column by column by
        // solving the arrow system against each unit arrow basis vector.
        let mut g = Array2::<f64>::zeros((dim, dim));
        let mut rhs_t = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(k);
        for col in 0..total_t {
            rhs_t[col] = 1.0;
            let sol = solver.solve(rhs_t.view(), rhs_beta_zero.view())?;
            rhs_t[col] = 0.0;
            for r in 0..total_t {
                g[[r, col]] = sol.t[r];
            }
            for r in 0..k {
                g[[total_t + r, col]] = sol.beta[r];
            }
        }
        let rhs_t_zero = Array1::<f64>::zeros(total_t);
        let mut rhs_beta = Array1::<f64>::zeros(k);
        for col in 0..k {
            rhs_beta[col] = 1.0;
            let sol = solver.solve(rhs_t_zero.view(), rhs_beta.view())?;
            rhs_beta[col] = 0.0;
            for r in 0..total_t {
                g[[r, total_t + col]] = sol.t[r];
            }
            for r in 0..k {
                g[[total_t + r, total_t + col]] = sol.beta[r];
            }
        }
        // H⁻¹ is self-adjoint; symmetrize away solver round-off asymmetry.
        for a in 0..dim {
            for b in (a + 1)..dim {
                let avg = 0.5 * (g[[a, b]] + g[[b, a]]);
                g[[a, b]] = avg;
                g[[b, a]] = avg;
            }
        }

        // Block-diagonal row-local t-inverse H_bd⁻¹ (dim×dim; β block zero) — the
        // inverse the rank-charge coordinate-block trace subtracts, built from the
        // same per-row undamped Cholesky factors `coordinate_block_ard_...` uses.
        let mut h_bd = Array2::<f64>::zeros((dim, dim));
        for row in 0..self.n_obs() {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let factor = cache.undamped_factor(row);
            let mut unit = Array1::<f64>::zeros(q);
            for col in 0..q {
                unit.fill(0.0);
                unit[col] = 1.0;
                let solved = cholesky_solve_vector(factor, unit.view());
                for r in 0..q {
                    h_bd[[base + r, base + col]] = solved[r];
                }
            }
        }

        // ∂H/∂ρ operators Cᵢ for the smoothing (β-block) and ARD (t-diagonal)
        // coordinates, keyed by flat outer index (shared-ARD axes accumulate).
        let mut c_by_flat: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();

        // Smoothing: Cₐ = (λ_a·½(Sₐ+Sₐᵀ)) ⊗ I on atom a's β-block, the exact
        // operator `decoder_smoothness_effective_dof_with_solver_per_atom` traces.
        let lambda_smooth = rho.lambda_smooth_vec()?;
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (beta_offsets, beta_out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) =
            if frames_active {
                let ranks: Vec<usize> =
                    self.atoms.iter().map(|a| a.border_frame_rank()).collect();
                (
                    self.factored_beta_offsets(),
                    Box::new(move |kk: usize| ranks[kk]),
                )
            } else {
                (self.beta_offsets(), Box::new(move |_kk: usize| p))
            };
        for a in 0..rho.log_lambda_smooth.len() {
            let atom = &self.atoms[a];
            let s = atom.smooth_penalty();
            let m = atom.basis_size();
            let off = beta_offsets[a];
            let r = beta_out_dim(a);
            let lambda = lambda_smooth[a];
            let flat = rho.smooth_flat_index(a);
            let c = c_by_flat
                .entry(flat)
                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
            for mu in 0..m {
                for nu in 0..m {
                    let s_sym = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                    let val = lambda * s_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        c[[total_t + off + nu * r + oc, total_t + off + mu * r + oc]] += val;
                    }
                }
            }
        }

        // ARD: C_{k,axis} = w_row·max(α cos κt, 0) (periodic) / w_row·α (Euclidean)
        // on the row-local t-slot for (atom k, axis) — the exact PSD-majorizer
        // curvature `ard_log_precision_hessian_trace` differentiates. The slot
        // layout mirrors that trace (compact top-k vs dense per-atom offsets).
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let coord_offsets = self.assignment.coord_offsets();
        let periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(LatentCoordValues::effective_axis_periods)
            .collect();
        for row in 0..self.n_obs() {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let base = cache.row_offsets[row];
            match self.last_row_layout {
                Some(ref layout) => {
                    for (pos, &kk) in layout.active_atoms[row].iter().enumerate() {
                        if rho.log_ard[kk].is_empty() {
                            continue;
                        }
                        let start = layout.coord_starts[row][pos];
                        let coord = &self.assignment.coords[kk];
                        for axis in 0..coord.latent_dim() {
                            let alpha = ard_precisions[kk][axis];
                            let t = coord.row(row)[axis];
                            let hess = w_row
                                * ArdAxisPrior::eval(alpha, t, periods[kk][axis])
                                    .psd_majorizer_hess();
                            if hess == 0.0 {
                                continue;
                            }
                            let flat = rho.ard_flat_index(kk, axis);
                            let c = c_by_flat
                                .entry(flat)
                                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                            let g_idx = base + start + axis;
                            c[[g_idx, g_idx]] += hess;
                        }
                    }
                }
                None => {
                    for kk in 0..self.k_atoms() {
                        if rho.log_ard[kk].is_empty() {
                            continue;
                        }
                        let coord = &self.assignment.coords[kk];
                        for axis in 0..coord.latent_dim() {
                            let alpha = ard_precisions[kk][axis];
                            let t = coord.row(row)[axis];
                            let hess = w_row
                                * ArdAxisPrior::eval(alpha, t, periods[kk][axis])
                                    .psd_majorizer_hess();
                            if hess == 0.0 {
                                continue;
                            }
                            let flat = rho.ard_flat_index(kk, axis);
                            let c = c_by_flat
                                .entry(flat)
                                .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                            let g_idx = base + coord_offsets[kk] + axis;
                            c[[g_idx, g_idx]] += hess;
                        }
                    }
                }
            }
        }

        // Sparse (assignment log-strength): C_sparse = the softmax Gershgorin PSD
        // majorizer `w_row · D`, `D = diag(Σ_j|H_kj|)` at `scale = λ_sparse·s/τ²`,
        // written into H_tt's logit slots by the assembly — the SAME operator
        // `assignment_log_strength_hessian_trace` traces. `|scale·H_kj| = scale·|H_kj|`
        // for `scale > 0`, so `D` is degree-one in `λ_sparse = e^ρ` exactly like the
        // smoothing and ARD operators, and `∂C_sparse/∂ρ_sparse = C_sparse`. Its
        // `sign(H_kj)` kink lives in the LOGITS, which a ρ perturbation never moves,
        // so the active branch is invariant at the fixed stratum. The `H_bd⁻¹` leg
        // then reproduces the gradient's `coordinate_block_assignment_...` subtraction
        // with no extra math, and the cross terms against smooth/ARD fall out of the
        // same uniform formula.
        if let Some(sparse_flat) = rho.sparse_flat_index() {
            let k_atoms = self.k_atoms();
            match self.assignment.mode {
                AssignmentMode::Softmax {
                    temperature,
                    sparsity,
                } if k_atoms > 1 => {
                    if self.last_row_layout.is_some() {
                        return Err(
                            "logdet_daleckii_krein_hessian: the compact top-k softmax row \
                             layout is not covered by the sparse log-strength operator; \
                             refusing to assemble a Hessian with an unmodelled sparse row"
                                .to_string(),
                        );
                    }
                    let inv_tau = 1.0 / temperature;
                    let scale = rho.lambda_sparse()? * sparsity * inv_tau * inv_tau;
                    let penalty =
                        gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                            k_atoms,
                            temperature,
                        );
                    let assignment_dim = self.assignment.assignment_coord_dim();
                    let c = c_by_flat
                        .entry(sparse_flat)
                        .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                    for row in 0..self.n_obs() {
                        let w_row = row_w.map_or(1.0, |w| w[row]);
                        let base = cache.row_offsets[row];
                        let q = cache.row_dims[row];
                        let logit_dim = assignment_dim.min(q);
                        let row_logits: Vec<f64> = (0..k_atoms)
                            .map(|atom| self.assignment.logits[[row, atom]])
                            .collect();
                        let d = penalty.psd_majorizer_abs_row_sums(&row_logits, scale);
                        for atom in 0..logit_dim {
                            c[[base + atom, base + atom]] += w_row * d[atom];
                        }
                    }
                }
                // K ≤ 1 softmax has no free logit: the gradient's sparse logdet trace
                // is identically zero, so a zero row here is the CORRECT curvature.
                AssignmentMode::Softmax { .. } => {}
                _ => {
                    return Err(
                        "logdet_daleckii_krein_hessian: rho carries a sparse log-strength \
                         coordinate under a non-softmax assignment prior, whose ∂H/∂ρ_sparse \
                         majorizer operator this channel does not yet model; refusing to \
                         assemble a Hessian with a silently-zero sparse row"
                            .to_string(),
                    );
                }
            }
        }

        // Precompute G·Cᵢ, H_bd⁻¹·Cᵢ, and their traces for each flat coordinate.
        let flats: Vec<usize> = c_by_flat.keys().copied().collect();
        let mut gc: Vec<Array2<f64>> = Vec::with_capacity(flats.len());
        let mut hc: Vec<Array2<f64>> = Vec::with_capacity(flats.len());
        let mut tr_g: Vec<f64> = Vec::with_capacity(flats.len());
        let mut tr_h: Vec<f64> = Vec::with_capacity(flats.len());
        for &flat in &flats {
            let c = &c_by_flat[&flat];
            let gci = g.dot(c);
            let hci = h_bd.dot(c);
            tr_g.push((0..dim).map(|d| gci[[d, d]]).sum());
            tr_h.push((0..dim).map(|d| hci[[d, d]]).sum());
            gc.push(gci);
            hc.push(hci);
        }

        // block[i,j] = ½·δ_{ij}·(tr(G Cᵢ) − tr(H_bd⁻¹ Cᵢ))
        //            − ½·(tr(G Cᵢ G C_j) − tr(H_bd⁻¹ Cᵢ H_bd⁻¹ C_j)).
        let mut hessian = Array2::<f64>::zeros((n_params, n_params));
        for (ii, &fi) in flats.iter().enumerate() {
            for (jj, &fj) in flats.iter().enumerate() {
                let (gi, gj) = (&gc[ii], &gc[jj]);
                let (hi, hj) = (&hc[ii], &hc[jj]);
                let mut cross_g = 0.0_f64;
                let mut cross_h = 0.0_f64;
                for a in 0..dim {
                    for b in 0..dim {
                        cross_g += gi[[a, b]] * gj[[b, a]];
                        cross_h += hi[[a, b]] * hj[[b, a]];
                    }
                }
                let diag = if ii == jj {
                    0.5 * (tr_g[ii] - tr_h[ii])
                } else {
                    0.0
                };
                hessian[[fi, fj]] += diag - 0.5 * (cross_g - cross_h);
            }
        }
        Ok(hessian)
    }

    /// PATH C (#2253) — assemble the exact fixed-stratum dense outer Hessian for
    /// the small-dense ARC route from its analytic channels, and the production
    /// consumer of the per-channel methods as they land.
    ///
    /// UNDER CONSTRUCTION: the solver-free explicit channel
    /// ([`Self::outer_explicit_smoothness_ard_hessian`]), the rank-charge
    /// `direct_rho` channel, and the deflated log-determinant Daleckii–Krein trace
    /// ([`Self::logdet_daleckii_krein_hessian`]) are implemented; only the
    /// third-order forward-sensitivity channel is still pending. Until every
    /// channel is assembled the curvature is incomplete, so this REFUSES rather
    /// than hand back a partial Hessian — the objective keeps
    /// `DeclaredHessianForm::Unavailable` and its `eval` returns
    /// `HessianValue::Unavailable`, so nothing steers on partial curvature. Each
    /// landed channel extends the block assembled here; the final one turns the
    /// refusal into `Ok` and flips `capability()` to `Dense`. `cache` is threaded
    /// now for the pending solver-bound channels.
    pub(crate) fn exact_fixed_stratum_outer_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        // #2231 crosscoder block relevances (`log_lambda_block`, the trailing flat
        // coordinates): the gradient prices them (`crosscoder_block_ift_rhs`), but no
        // Hessian channel writes their rows/columns yet. Emitting a Dense Hessian with
        // those rows identically zero while their gradient is live would hand ARC a
        // singular system — strictly worse than declaring the curvature unavailable.
        // Refuse until a block channel lands. (Empty on the circle-mint route.)
        if !rho.log_lambda_block.is_empty() {
            return Err(format!(
                "exact_fixed_stratum_outer_hessian: rho carries {} crosscoder block \
                 relevance coordinate(s) that no Hessian channel models; refusing to \
                 advertise a curvature block with unmodelled (zero) rows",
                rho.log_lambda_block.len()
            ));
        }
        let mut hessian = self.outer_explicit_smoothness_ard_hessian(rho, loss.smoothness)?;
        hessian += &self.rank_charge_direct_rho_hessian(target, rho, loss, cache)?;
        hessian += &self.logdet_daleckii_krein_hessian(rho, cache)?;
        // Landed: explicit smoothness/ARD + rank-charge direct + deflated
        // log-determinant Daleckii–Krein trace channels. Pending: the third-order
        // forward-sensitivity channel. Refuse partial curvature until it lands.
        Err(format!(
            "PATH C exact fixed-stratum outer Hessian is incomplete: the {}×{} explicit \
             smoothness/ARD + rank-charge + log-determinant blocks are landed, but the \
             third-order forward-sensitivity channel is still pending; refusing to \
             advertise partial curvature",
            hessian.nrows(),
            hessian.ncols()
        ))
    }
}
