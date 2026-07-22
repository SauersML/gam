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

/// PATH C (#2253) CH5 — which subset of the joint θ-derivative operator
/// `K_w = ∂H/∂θ_w` a dense θ-adjoint contraction assembles. The FULL set
/// reconstructs `Γ_w = tr(inv·K_w)` (self-checked against the production
/// `logdet_theta_adjoint`); the two MIXED subsets isolate the single ρ-scaled
/// term of `K_w` whose `∂/∂ρ_i` is nonzero (both are degree-one in `e^{ρ_i}`,
/// so `∂K_w/∂ρ_i` equals the term itself). Keeping them as distinct channels
/// makes a failing finite-difference gate localize to ONE formula.
#[derive(Clone, Copy)]
enum ThetaAdjointDhChannel {
    /// Every `∂H/∂θ_w` contribution: data residual curvature, the softmax
    /// data-weight logit factor, the softmax entropy Gershgorin majorizer, and
    /// the periodic ARD majorizer diagonal.
    All,
    /// ONLY the softmax entropy Gershgorin majorizer θ-derivative (logit–logit,
    /// same atom). This is the `∝ λ_sparse` term, so its `∂/∂ρ_sparse` equals
    /// itself — the part-(b) mixed channel for the sparse coordinate.
    SoftmaxSparseMixed,
    /// ONLY the periodic ARD majorizer diagonal `w_row·(−ακ sin κt)` for the
    /// coordinate slots whose `ard_flat_index` matches `target_flat`. This is
    /// `∝ α = e^{ρ_ard}`, so its `∂/∂ρ_ard` equals itself — the part-(b) mixed
    /// channel for one ARD coordinate.
    ArdMixed { target_flat: usize },
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
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            // #2304 resident path for the residual-curvature blocks (1a)+(1b):
            // the raw second/mixed jets are contracted on device (when the plan
            // admits it) against the metric-applied √w-scaled residual and the
            // direction's (t, β) coefficients — the packed channel tensors are
            // never materialized. Blocks (2)-(3) below are logit/coord-space
            // prior curvatures with no channel tensors involved and stay on
            // the host.
            {
                let mut probe_assignments = Array1::<f64>::zeros(k_atoms);
                let probe_for_row = |row: usize| -> Result<Vec<f64>, String> {
                    self.assignment.try_assignments_row_into(
                        row,
                        probe_assignments.as_slice_mut().ok_or_else(|| {
                            "apply_exact_hessian_minus_b: assignment scratch is not contiguous"
                                .to_string()
                        })?,
                    )?;
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
                        let a_k = probe_assignments[k];
                        for out_col in 0..p {
                            fitted[out_col] += a_k * decoded[out_col];
                        }
                    }
                    let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
                    for out_col in 0..p {
                        error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
                    }
                    Ok(match self.row_metric.as_ref() {
                        Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                        _ => error.to_vec(),
                    })
                };
                let v_t_for_row = |row: usize, q: usize| -> Result<Vec<f64>, String> {
                    let base = cache.row_offsets[row];
                    Ok((0..q).map(|c| v.t[base + c]).collect())
                };
                let v_beta_row: Vec<f64> =
                    border.iter().map(|channel| v.beta[channel.index]).collect();
                let out_ref = &mut out;
                self.contracted_softmax_bilinear_hvp(
                    cache,
                    &second_jets,
                    &border,
                    probe_for_row,
                    v_t_for_row,
                    &v_beta_row,
                    |row, _q, t_row, beta_row| {
                        let base = cache.row_offsets[row];
                        for (a, &value) in t_row.iter().enumerate() {
                            out_ref.t[base + a] += value;
                        }
                        for (channel, &value) in border.iter().zip(beta_row) {
                            out_ref.beta[channel.index] += value;
                        }
                        Ok(())
                    },
                )?;
            }
            // (2) softmax entropy-minus-majorizer and (3) periodic-ARD deltas,
            // per row with the layout rebuilt from the cache (no jets needed).
            for row in 0..n {
                let q = cache.row_dims[row];
                let base = cache.row_offsets[row];
                self.assignment.try_assignments_row_into(
                    row,
                    assignments.as_slice_mut().ok_or_else(|| {
                        "apply_exact_hessian_minus_b: assignment scratch is not contiguous"
                            .to_string()
                    })?,
                )?;
                let vars = self.row_vars_for_cache_row(row, cache)?;
                let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
                let w_row = row_loss_w.map_or(1.0, |w| w[row]);
                if let Some((_penalty, scale)) = softmax_delta.as_ref() {
                    let assignment_dim = self.assignment.assignment_coord_dim();
                    let a_soft = assignments
                        .as_slice()
                        .expect("softmax assignments row must be contiguous");
                    let m = softmax_majorizer_log_mean(a_soft);
                    for (a, va) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: ka } = *va else {
                            continue;
                        };
                        if ka >= assignment_dim {
                            continue;
                        }
                        let mut acc = 0.0_f64;
                        for (b, vb) in vars.iter().enumerate() {
                            let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                                continue;
                            };
                            if kb >= assignment_dim {
                                continue;
                            }
                            let h_entropy =
                                softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, *scale);
                            let delta = if ka == kb {
                                h_entropy
                                    - active_softmax_gershgorin_majorizer_entry(
                                        a_soft, ka, m, *scale,
                                    )
                            } else {
                                h_entropy
                            };
                            acc += w_row * delta * v_t[b];
                        }
                        out.t[base + a] += acc;
                    }
                }
                for (a, va) in vars.iter().enumerate() {
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
            return Ok(out);
        }
        // #932 complete schedule: non-softmax gates use their distinct dynamic
        // row program through the bounded look-ahead window.
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

            // (2) softmax entropy-minus-majorizer: softmax gates return through
            // the resident contracted branch above (#1419 algebra preserved
            // there verbatim, including the #1410 active-slot contraction and
            // the #991 `w_row` convention), so no softmax delta arises here.

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

    /// PATH C (#2253) — the per-flat-coordinate penalty curvature operators
    /// `M_i = ∂H/∂ρ_i` at a frozen inner state, keyed by flat outer coordinate.
    /// Extracted from [`Self::logdet_daleckii_krein_hessian`] (ch4) so ch4's
    /// Daleckii–Krein trace and ch5's forward-sensitivity twist read ONE
    /// operator map (value/gradient/Hessian never differentiate divergent
    /// curvatures). Each `M_i` is degree-one in `exp(ρ_i)`: `λ_k·½(S_k+S_kᵀ)⊗I`
    /// on atom `k`'s β-block for smoothing; `w_row·max(α cos κt,0)` on the active
    /// row-local t-slots for periodic ARD (`w_row·α` Euclidean); the softmax
    /// Gershgorin majorizer `w_row·diag(Σ_j|H_kj|)` on the logit slots for the
    /// sparse coordinate. The sparse refusals (compact top-k layout, non-softmax
    /// prior) match ch4's so both channels decline the same unmodelled cases.
    pub(crate) fn penalty_curvature_operators_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut c_by_flat: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();

        // Smoothing: Cₐ = (λ_a·½(Sₐ+Sₐᵀ)) ⊗ I on atom a's β-block.
        let lambda_smooth = rho.lambda_smooth_vec()?;
        let p = self.output_dim();
        let frames_active = self.frames_active();
        let (beta_offsets, beta_out_dim): (Vec<usize>, Box<dyn Fn(usize) -> usize>) =
            if frames_active {
                let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
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
        // on the row-local t-slot for (atom k, axis).
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

        // Sparse (assignment log-strength): the softmax Gershgorin PSD majorizer
        // `w_row·diag(Σ_j|H_kj|)` at `scale = λ_sparse·s/τ²`, written into the
        // logit slots — degree-one in `λ_sparse = e^ρ` exactly like smoothing/ARD.
        if let Some(sparse_flat) = rho.sparse_flat_index() {
            let k_atoms = self.k_atoms();
            match self.assignment.mode {
                AssignmentMode::Softmax {
                    temperature,
                    sparsity,
                } if k_atoms > 1 => {
                    if self.last_row_layout.is_some() {
                        return Err(
                            "penalty_curvature_operators_by_flat: the compact top-k softmax row \
                             layout is not covered by the sparse log-strength operator; refusing \
                             to assemble a curvature operator with an unmodelled sparse row"
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
                AssignmentMode::Softmax { .. } => {}
                AssignmentMode::OrderedBetaBernoulli { .. } => {
                    // #2330: the ordered-Beta–Bernoulli sparse ∂A/∂ρ_sparse is the
                    // EXACT integrated-marginal logit Hessian (cross-row), supplied
                    // by `dense_exact_a_ordered_bb_sparse_trace`, NOT a diagonal
                    // majorizer operator this map can assemble. Emit nothing here
                    // (the dense-A gradient adds that coordinate's trace directly)
                    // rather than a wrong diagonal-only operator.
                }
                _ => {
                    return Err(
                        "penalty_curvature_operators_by_flat: rho carries a sparse log-strength \
                         coordinate under an assignment prior whose ∂H/∂ρ_sparse operator this \
                         map does not model; refusing to assemble a silently-zero sparse operator"
                            .to_string(),
                    );
                }
            }
        }

        Ok(c_by_flat)
    }

    /// PATH C (#2253) CH5 — the ρ-derivative of the EXACT-minus-majorizer
    /// stationarity correction, `∂(ΔC)/∂ρ_i` where `ΔC = A − B`
    /// ([`Self::apply_exact_hessian_minus_b`]), keyed by flat coordinate. The IFT
    /// sensitivity `∂a/∂ρ_i = A⁺(∂Γ/∂ρ_i − (∂A/∂ρ_i)a)` differentiates the EXACT
    /// stationarity Hessian `A = B + ΔC`, not the majorized solver operator `B = H`
    /// (`penalty_curvature_operators_by_flat` = `∂B/∂ρ`). So the `M_i·a` term must
    /// use `∂A/∂ρ_i = ∂B/∂ρ_i + ∂(ΔC)/∂ρ_i` — this map supplies the second piece.
    ///
    /// Both deltas are degree-one in their ρ (so `∂(ΔC)/∂ρ_i` is the delta itself)
    /// and mirror `apply_exact_hessian_minus_b`'s deltas exactly:
    /// * periodic ARD: `w_row·min(α cos κt, 0)` (the negative-part remainder the
    ///   `max(·,0)` majorizer drops) on the coord slot, ALL rows — nonzero only on
    ///   the inactive half `cos κt < 0`. This is the term the ARD-perturbed
    ///   `H3[ard,·]` rows need (the transposed smooth-perturbed rows, where
    ///   `∂A = ∂B`, are already exact).
    /// * softmax sparse: the exact entropy Hessian minus the Gershgorin majorizer
    ///   on the row's logit block (dense, off-diagonal + diagonal), `∝ λ_sparse`.
    /// Smooth is unmajorized (`ΔC` has no smooth part), so its delta is zero and it
    /// is absent from the map. Covered config only (softmax, dense row layout).
    pub(crate) fn exact_stationarity_penalty_derivative_delta_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let k_atoms = self.k_atoms();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let softmax_delta: Option<(usize, f64)> = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } if k_atoms > 1 => {
                let inv_tau = 1.0 / temperature;
                match rho.sparse_flat_index() {
                    Some(sparse_flat) => Some((
                        sparse_flat,
                        rho.lambda_sparse()? * sparsity * inv_tau * inv_tau,
                    )),
                    None => None,
                }
            }
            _ => None,
        };
        let mut deltas: std::collections::BTreeMap<usize, Array2<f64>> =
            std::collections::BTreeMap::new();
        let mut assignments = Array1::<f64>::zeros(k_atoms);
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            self.assignment.try_assignments_row_into(
                row,
                assignments
                    .as_slice_mut()
                    .expect("assignment scratch is contiguous"),
            )?;
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let w_row = row_w.map_or(1.0, |w| w[row]);
            // Softmax entropy-minus-majorizer delta on the logit block.
            if let Some((sparse_flat, scale)) = softmax_delta {
                let assignment_dim = self.assignment.assignment_coord_dim();
                let a_soft = assignments
                    .as_slice()
                    .expect("softmax assignments row must be contiguous");
                let m = softmax_majorizer_log_mean(a_soft);
                let c = deltas
                    .entry(sparse_flat)
                    .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom: ka } = *va else {
                        continue;
                    };
                    if ka >= assignment_dim {
                        continue;
                    }
                    for (b, vb) in vars.iter().enumerate() {
                        let SaeLocalRowVar::Logit { atom: kb } = *vb else {
                            continue;
                        };
                        if kb >= assignment_dim {
                            continue;
                        }
                        let h_entropy =
                            softmax_dense_entropy_hessian_entry(a_soft, ka, kb, m, scale);
                        let delta = if ka == kb {
                            h_entropy
                                - active_softmax_gershgorin_majorizer_entry(a_soft, ka, m, scale)
                        } else {
                            h_entropy
                        };
                        c[[base + a, base + b]] += w_row * delta;
                    }
                }
            }
            // Periodic-ARD negative-part remainder on the coord slots.
            for (a, va) in vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let neg = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis])
                    .negative_hessian_remainder();
                if neg != 0.0 {
                    let flat = rho.ard_flat_index(atom, axis);
                    let c = deltas
                        .entry(flat)
                        .or_insert_with(|| Array2::<f64>::zeros((dim, dim)));
                    c[[base + a, base + a]] += w_row * neg;
                }
            }
        }
        Ok(deltas)
    }

    /// PATH C (#2253) — the full joint arrow inverse `G = H⁻¹` (dim×dim),
    /// materialized column by column against each unit arrow basis vector and
    /// symmetrized. Shared by ch4 and ch5's small-dense (circle-mint scale)
    /// route; `solver` must be [`DeflatedArrowSolver::plain`].
    pub(crate) fn materialize_joint_inverse(
        &self,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
    ) -> Result<Array2<f64>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
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
        for a in 0..dim {
            for b in (a + 1)..dim {
                let avg = 0.5 * (g[[a, b]] + g[[b, a]]);
                g[[a, b]] = avg;
                g[[b, a]] = avg;
            }
        }
        Ok(g)
    }

    /// PATH C (#2253) — the block-diagonal row-local t-inverse `H_bd⁻¹` (dim×dim;
    /// β block zero) built from the per-row undamped Cholesky factors, the same
    /// inverse the rank-charge coordinate-block trace subtracts. Shared by ch4
    /// and ch5.
    pub(crate) fn materialize_block_diag_t_inverse(&self, cache: &ArrowFactorCache) -> Array2<f64> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
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
        h_bd
    }

    /// PATH C (#2253) CH5 — dense reconstruction of the θ-adjoint contraction
    /// `Γ_w = tr(inv · K_w)`, `K_w = ∂H/∂θ_w`, for an ARBITRARY dense joint
    /// inverse `inv` (dim×dim over the `(t, β)` blocks) and a chosen subset of
    /// the `K_w` operator ([`ThetaAdjointDhChannel`]).
    ///
    /// With `inv = G` and `ThetaAdjointDhChannel::All` this reproduces the
    /// production [`Self::logdet_theta_adjoint`] (self-checked by the FD gate);
    /// with `inv = h_bd` it reproduces [`Self::coordinate_block_logdet_theta_adjoint`].
    /// Feeding the TWISTED inverse `−G M_i G` gives the part-(a) term
    /// `−tr(G M_i G K_w)` of `dΓ/dρ_i`; the two MIXED channels give part-(b).
    ///
    /// Covered config ONLY (validated by the caller): softmax assignment, dense
    /// per-atom row layout (`last_row_layout = None`), no per-row deflation, no
    /// border frames, no ordered Beta--Bernoulli. The `dh` assembly mirrors the
    /// production builder's inner loop for exactly that config; the softmax
    /// diagonal `assignment_prior_hdiag_derivative_entry` is 0 for softmax and is
    /// omitted here for the same reason.
    fn logdet_theta_adjoint_dense(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        inv: &Array2<f64>,
        channel: ThetaAdjointDhChannel,
        skip_deflation_dk: bool,
        exact_a: bool,
    ) -> Result<SaeArrowVector, String> {
        // #2330 — `skip_deflation_dk` drops the Daleckii–Krein deflation
        // correction, leaving the raw trace contraction. The split probe uses it
        // to attribute the g3 cross non-conservation to the trace vs the
        // frozen-DK piece of the twist. Production callers pass `false`.
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let k_atoms = self.k_atoms();
        let n = self.n_obs();
        let mut gamma_t = Array1::<f64>::zeros(total_t);
        let mut gamma_beta = Array1::<f64>::zeros(k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let whiten_row_jets = self.whiten_logdet_row_jets();
        let want_data = matches!(channel, ThetaAdjointDhChannel::All);
        let want_entropy = matches!(
            channel,
            ThetaAdjointDhChannel::All | ThetaAdjointDhChannel::SoftmaxSparseMixed
        );
        let want_ard = matches!(
            channel,
            ThetaAdjointDhChannel::All | ThetaAdjointDhChannel::ArdMixed { .. }
        );
        // `1/τ` (always, for the softmax data-weight logit factor) and the
        // entropy Gershgorin majorizer scale `λ_sparse·s/τ²` (only a live free
        // logit, i.e. `k_atoms > 1`, carries the sparsity penalty).
        let (entropy_scale, inv_tau) = match self.assignment.mode {
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            } => {
                let inv_tau = 1.0 / temperature;
                let scale = if k_atoms > 1 {
                    rho.lambda_sparse()? * sparsity * inv_tau * inv_tau
                } else {
                    0.0
                };
                (scale, inv_tau)
            }
            _ => (0.0, 0.0),
        };
        // #2330 — ordered-Beta--Bernoulli gate flag + inverse temperature for the
        // OBB gate-logit factor below (the softmax factor is inert for OBB).
        let patchd_is_obb = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        );
        let patchd_obb_inv_tau = match self.assignment.mode {
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => 1.0 / temperature,
            _ => 0.0,
        };
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        let mut assignments = Array1::<f64>::zeros(k_atoms);
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
            let mut jets = jet_window
                .pop_front()
                .ok_or_else(|| "logdet_theta_adjoint_dense: empty jet window".to_string())?;
            if whiten_row_jets {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }
            let a_soft = assignments
                .as_slice()
                .expect("softmax assignments row must be contiguous");
            let m_log_mean = softmax_majorizer_log_mean(a_soft);
            let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            // #2308 — per-row spectral/gauge deflation the criterion factor applied.
            // It is FROZEN at the fixed stratum (the radial-gauge / ARD-inactive-half
            // null is ρ-invariant), so contracting the DEFLATED inverse `inv` and
            // subtracting the SAME Daleckii–Krein correction the production θ-adjoint
            // subtracts makes `Γ(inv)` — and its twist `Γ(−G Mᵢ G)` — match the
            // gradient on the deflated circle route (where deflation is the norm, not
            // an error). `deflation_block_correction` is linear in `inv`, so the twist
            // rides through it exactly.
            let defl_dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let defl_spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            let inv_vv_block = if defl_dirs.is_empty() {
                Array2::<f64>::zeros((0, 0))
            } else {
                inv.slice(s![base..base + q, base..base + q]).to_owned()
            };
            for w in 0..q {
                let logit_w = match jets.vars[w] {
                    SaeLocalRowVar::Logit { atom } => Some(atom),
                    SaeLocalRowVar::Coord { .. } => None,
                };
                let mut gamma = 0.0_f64;
                let mut dh_mat = if defl_dirs.is_empty() {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = 0.0_f64;
                        if want_data {
                            dh += match (logit_w, jets.vars[a], jets.vars[b]) {
                                (
                                    Some(atom_w),
                                    SaeLocalRowVar::Coord { atom: atom_a, .. },
                                    SaeLocalRowVar::Coord { atom: atom_b, .. },
                                ) => {
                                    sae_dot(jets.first(a), jets.first(b))
                                        * (Self::softmax_data_weight_product_logit_factor(
                                            a_soft, atom_a, atom_b, atom_w, inv_tau,
                                        ) + if patchd_is_obb
                                            && atom_w == atom_a
                                            && atom_w == atom_b
                                        {
                                            // #2330 — ordered-Beta--Bernoulli gate
                                            // gradient of the GN curvature. `B[a,b] ∝
                                            // gate^2`, so `∂B/∂ℓ_w = 2·(gate'/gate)·B =
                                            // 2(1-gate)/τ·B`. Same-atom only: the OBB
                                            // gates are INDEPENDENT per atom, so the
                                            // cross-atom derivative is exactly zero
                                            // (unlike softmax's shared normalization).
                                            // The softmax factor above is 0 here
                                            // (`inv_tau` is 0 for non-softmax modes), so
                                            // the softmax path is bitwise unchanged.
                                            2.0 * (1.0 - a_soft[atom_w]) * patchd_obb_inv_tau
                                        } else {
                                            0.0
                                        })
                                }
                                _ => {
                                    sae_dot(jets.second(a, w), jets.first(b))
                                        + sae_dot(jets.first(a), jets.second(b, w))
                                }
                            };
                        }
                        if want_entropy {
                            if let (
                                Some(atom_w),
                                SaeLocalRowVar::Logit { atom: atom_a },
                                SaeLocalRowVar::Logit { atom: atom_b },
                            ) = (logit_w, jets.vars[a], jets.vars[b])
                            {
                                if atom_a == atom_b {
                                    dh += w_row
                                        * active_softmax_majorizer_logit_derivative_entry(
                                            a_soft,
                                            atom_a,
                                            atom_w,
                                            m_log_mean,
                                            entropy_scale,
                                            inv_tau,
                                        );
                                }
                            }
                        }
                        if want_ard && a == b && a == w {
                            if let SaeLocalRowVar::Coord { atom, axis } = jets.vars[a] {
                                if !ard_precisions[atom].is_empty() {
                                    let include = match channel {
                                        ThetaAdjointDhChannel::ArdMixed { target_flat } => {
                                            rho.ard_flat_index(atom, axis) == target_flat
                                        }
                                        _ => true,
                                    };
                                    if include {
                                        dh += if exact_a {
                                            self.ard_exact_hessian_derivative(
                                                ard_precisions[atom][axis],
                                                row,
                                                atom,
                                                axis,
                                            )
                                        } else {
                                            self.ard_majorized_hessian_derivative(
                                                ard_precisions[atom][axis],
                                                row,
                                                atom,
                                                axis,
                                            )
                                        };
                                    }
                                }
                            }
                        }
                        if !defl_dirs.is_empty() {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv[[base + b, base + a]] * dh;
                    }
                }
                if !defl_dirs.is_empty() && !skip_deflation_dk {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv_block,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                if want_data {
                    for a in 0..q {
                        for (beta_pos, ch) in border.iter().enumerate() {
                            let dh = sae_dot(jets.second(a, w), jets.beta(beta_pos))
                                + sae_dot(jets.first(a), jets.beta_deriv(w, beta_pos));
                            gamma += 2.0 * inv[[base + a, total_t + ch.index]] * dh;
                        }
                    }
                    for (beta_i, ch_i) in border.iter().enumerate() {
                        for (beta_j, ch_j) in border.iter().enumerate() {
                            let dh = sae_dot(jets.beta_deriv(w, beta_i), jets.beta(beta_j))
                                + sae_dot(jets.beta(beta_i), jets.beta_deriv(w, beta_j));
                            gamma += inv[[total_t + ch_i.index, total_t + ch_j.index]] * dh;
                        }
                    }
                }
                gamma_t[base + w] = gamma;
            }
            if want_data {
                for (w_beta_pos, w_channel) in border.iter().enumerate() {
                    let mut gamma = 0.0_f64;
                    let mut dh_mat = if defl_dirs.is_empty() {
                        Array2::<f64>::zeros((0, 0))
                    } else {
                        Array2::<f64>::zeros((q, q))
                    };
                    for a in 0..q {
                        for b in 0..q {
                            let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                                + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
                            if !defl_dirs.is_empty() {
                                dh_mat[[a, b]] = dh;
                            }
                            gamma += inv[[base + b, base + a]] * dh;
                        }
                    }
                    if !defl_dirs.is_empty() && !skip_deflation_dk {
                        gamma -= Self::deflation_block_correction(
                            &inv_vv_block,
                            &dh_mat,
                            defl_dirs,
                            defl_spectrum,
                        );
                    }
                    for a in 0..q {
                        for (beta_pos, ch) in border.iter().enumerate() {
                            let dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                            gamma += 2.0 * inv[[base + a, total_t + ch.index]] * dh;
                        }
                    }
                    gamma_beta[w_channel.index] += gamma;
                }
            }
        }
        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
    }

    /// PATH C (#2253) CH5 — the fixed-stratum ρ-derivative of the rank-charge
    /// θ-adjoint `∇R = production_rank_charge_derivative().theta`, for ONE smooth
    /// coordinate `smooth_flat`. `∇R` depends on ρ only through the per-atom
    /// penalized Gram `A = G + λ S` (`λ = e^{ρ_smooth}`), and the θ-assembly is
    /// LINEAR in each atom's differential blocks (`gram`, `occupancy`), so the
    /// derivative reruns the SAME assembly with those blocks replaced by their
    /// λ-derivatives (and zeroed for every other atom). With `A⁻¹ = inv`,
    /// `S = smooth_penalty`, `dλ/dρ = λ`, `dA⁻¹/dλ = −A⁻¹SA⁻¹`:
    /// `d(inv − inv G inv)/dρ = λ(−inv S inv + inv S inv G inv + inv G inv S inv)`
    /// and `d tr(inv G)/dρ = −λ tr(inv S inv G)`. Non-interior-EDF atoms are on a
    /// locally constant branch (zero derivative), matching the gradient.
    fn rank_charge_theta_rho_derivative(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        smooth_flat: usize,
    ) -> Result<SaeArrowVector, String> {
        let target_atom = smooth_flat - rho.smooth_flat_start();
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion = self.reconstruction_dispersion(loss, cache, rho, Some(residual.view()))?;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams)?;
        let n_eff = self.per_atom_effective_sample_size();
        let lambda_vec = rho.lambda_smooth_vec()?;
        let p = self.output_dim() as f64;

        // Per-atom differential BLOCKS (gram, occupancy), zero except the target
        // atom, whose blocks are the ρ_smooth-derivatives of the gradient's.
        let mut atom_differentials: Vec<ProductionRankChargeAtomDifferential> =
            Vec::with_capacity(self.k_atoms());
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let m = atom.basis_size();
            if atom_idx != target_atom || m == 0 {
                atom_differentials.push(ProductionRankChargeAtomDifferential {
                    gram: Array2::<f64>::zeros((m, m)),
                    occupancy: 0.0,
                });
                continue;
            }
            let gram = &grams[atom_idx];
            let n_atom = n_eff[atom_idx];
            let lambda = lambda_vec[atom_idx];
            let spectrum = super::wbic_audit::recon_spectrum(
                gram,
                &atom.decoder_coefficients,
                n_atom,
                p,
                dispersion,
                lambda,
                Some(atom.smooth_penalty()),
            )?;
            let rank = spectrum.production_chargeable_rank() as f64;
            if !(rank > 0.0) {
                return Err(format!(
                    "rank_charge_theta_rho_derivative: atom {atom_idx} is on the rank-zero \
                     Laplace-invalid branch (vanished decoder)"
                ));
            }
            let log_n = n_atom.max(1.0).ln();
            if log_n == 0.0 {
                atom_differentials.push(ProductionRankChargeAtomDifferential {
                    gram: Array2::<f64>::zeros((m, m)),
                    occupancy: 0.0,
                });
                continue;
            }
            let s = atom.smooth_penalty();
            let mut penalized_gram = gram.clone();
            for r in 0..m {
                for c in 0..m {
                    penalized_gram[[r, c]] += lambda * s[[r, c]];
                }
            }
            let factor = penalized_gram.cholesky(Side::Lower).map_err(|error| {
                format!(
                    "rank_charge_theta_rho_derivative: atom {atom_idx} penalized Gram \
                     factorization failed: {error}"
                )
            })?;
            let inverse = factor.solve_mat(&Array2::<f64>::eye(m));
            let edf_matrix = factor.solve_mat(gram);
            let raw_edf = (0..m).map(|i| edf_matrix[[i, i]]).sum::<f64>();
            let edf = super::construction::certified_basis_edf(
                raw_edf,
                m,
                "rank_charge_theta_rho_derivative",
            )?;
            let edf_is_interior = edf > 0.0 && edf < m as f64;
            // Reused products (all m×m): inv S inv, inv G inv, inv S inv G inv,
            // inv G inv S inv, and inv S inv G (for the EDF trace).
            let inv_s_inv = inverse.dot(s).dot(&inverse);
            let inv_g_inv = inverse.dot(gram).dot(&inverse);
            let inv_s_inv_g_inv = inv_s_inv.dot(gram).dot(&inverse);
            let inv_g_inv_s_inv = inv_g_inv.dot(s).dot(&inverse);
            let mut gram_prime = Array2::<f64>::zeros((m, m));
            if edf_is_interior {
                let coeff = lambda * 0.5 * rank * log_n;
                for r in 0..m {
                    for c in 0..m {
                        gram_prime[[r, c]] = coeff
                            * (-inv_s_inv[[r, c]]
                                + inv_s_inv_g_inv[[r, c]]
                                + inv_g_inv_s_inv[[r, c]]);
                    }
                }
            }
            let occupancy_prime = if n_atom > 1.0 {
                let edf_prime = if edf_is_interior {
                    let inv_s_inv_g = inv_s_inv.dot(gram);
                    -lambda * (0..m).map(|i| inv_s_inv_g[[i, i]]).sum::<f64>()
                } else {
                    0.0
                };
                0.5 * rank * edf_prime / n_atom
            } else {
                0.0
            };
            atom_differentials.push(ProductionRankChargeAtomDifferential {
                gram: gram_prime,
                occupancy: occupancy_prime,
            });
        }

        // The SAME linear θ-assembly as `production_rank_charge_derivative`, now
        // driven by the differential-of-the-differential blocks.
        let mut theta_t = Array1::<f64>::zeros(cache.delta_t_len());
        let theta_beta = Array1::<f64>::zeros(cache.k);
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        for row in 0..self.n_obs() {
            self.assignment.try_assignments_row_into(
                row,
                assignments
                    .as_slice_mut()
                    .expect("rank-charge assignment scratch is contiguous"),
            )?;
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let base = cache.row_offsets[row];
            for (slot, var) in vars.into_iter().enumerate() {
                theta_t[base + slot] = match var {
                    SaeLocalRowVar::Coord { atom, axis } => {
                        let a = assignments[atom];
                        if a == 0.0 {
                            0.0
                        } else {
                            let phi = self.atoms[atom].basis_values.row(row);
                            let dphi = self.atoms[atom].basis_jacobian.slice(s![row, .., axis]);
                            2.0 * a * a * dphi.dot(&atom_differentials[atom].gram.dot(&phi))
                        }
                    }
                    SaeLocalRowVar::Logit { atom: wrt_atom } => {
                        let mut derivative = 0.0_f64;
                        for atom in 0..self.k_atoms() {
                            let da = self.rank_charge_assignment_derivative(
                                wrt_atom,
                                atom,
                                assignments
                                    .as_slice()
                                    .expect("rank-charge assignment scratch is contiguous"),
                            );
                            if da == 0.0 {
                                continue;
                            }
                            let a = assignments[atom];
                            let phi = self.atoms[atom].basis_values.row(row);
                            let gram_quadratic = phi.dot(&atom_differentials[atom].gram.dot(&phi));
                            derivative += 2.0
                                * a
                                * da
                                * (gram_quadratic + atom_differentials[atom].occupancy);
                        }
                        derivative
                    }
                };
            }
        }
        Ok(SaeArrowVector {
            t: theta_t,
            beta: theta_beta,
        })
    }

    /// PATH C (#2253) CH5 — the exact fixed-stratum second derivative of the
    /// outer gradient's third-order forward-sensitivity channel
    /// `g3[j] = −½⟨a, g_ρ,j⟩`, `a = A⁺Γ_eff`.
    ///
    /// `H3[i,j] = ∂g3[j]/∂ρ_i = −½( ⟨dΓ_eff/dρ_i − M_i·a, b_j⟩ + δ_ij⟨a, g_ρ,j⟩ )`
    /// with `b_j = A⁺ g_ρ,j` (self-adjointness of `A⁺`). `Γ_eff = Γ_joint − Γ_tt
    /// + 2∇R` — the SAME effective adjoint the gradient assembles. Each
    /// `dΓ_·/dρ_i` splits into part-(a) `−tr(inv M_i inv K_w)` (twisted inverse)
    /// and part-(b) `tr(inv ∂K_w/∂ρ_i)` (the ARD / softmax-sparse mixed
    /// channels), and `d∇R/dρ` is nonzero only on the smooth coordinates. The
    /// returned block is `∂g3[j]/∂ρ_i` verbatim (validated by the FD gate);
    /// the caller may symmetrize.
    fn third_order_forward_sensitivity_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        // Covered config: the small-dense softmax route with no deflation,
        // frames, compact layout, or ordered Beta--Bernoulli. Outside it the
        // dense `dh` reconstruction and the twist are not the exact operator, so
        // refuse rather than advertise wrong curvature.
        if !matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            return Err(
                "third_order_forward_sensitivity_hessian: only the softmax assignment route is \
                 modelled by the dense θ-adjoint reconstruction"
                    .to_string(),
            );
        }
        if self.last_row_layout.is_some() {
            return Err(
                "third_order_forward_sensitivity_hessian: the compact top-k softmax row layout is \
                 not covered by the dense θ-adjoint reconstruction"
                    .to_string(),
            );
        }
        if self.frames_active() {
            return Err(
                "third_order_forward_sensitivity_hessian: border-frame smoothness offsets are not \
                 covered by this channel"
                    .to_string(),
            );
        }
        let solver = DeflatedArrowSolver::plain(cache);
        // Per-row spectral/gauge deflation IS modelled — the dense θ-adjoint
        // subtracts the same frozen Daleckii–Krein correction the production
        // builder does (#2308), and the plain deflated inverse is what `a`/`b_j`
        // and the twist all ride. What the plain solver CANNOT reconstruct is the
        // rank-R β-Schur Woodbury GAUGE correction: there the materialized inverse
        // would omit it, so refuse rather than assemble a wrong twist.
        if !solver.plain_selected_inverse_available() {
            return Err(
                "third_order_forward_sensitivity_hessian: a β-Schur Woodbury gauge deflation is \
                 active; the plain selected inverse omits its rank-R correction, so the \
                 twisted-inverse reconstruction is not the exact operator"
                    .to_string(),
            );
        }

        let n_params = rho.to_flat().len();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let flatten = |v: &SaeArrowVector| -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(dim);
            out.slice_mut(s![..total_t]).assign(&v.t);
            out.slice_mut(s![total_t..]).assign(&v.beta);
            out
        };

        let g = self.materialize_joint_inverse(cache, &solver)?;
        let h_bd = self.materialize_block_diag_t_inverse(cache);
        let operators = self.penalty_curvature_operators_by_flat(rho, cache)?;
        // `∂A/∂ρᵢ = ∂H/∂ρᵢ (operators) + ∂(ΔC)/∂ρᵢ (this delta)`. BOTH the twist
        // inverse ∂G/∂ρ = −G(∂A/∂ρ)G and the IFT `Mᵢ·a` term differentiate the
        // EXACT stationarity Hessian A, so both add this delta (#2330).
        let exact_deltas = self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)?;

        // Effective adjoint Γ_eff = Γ_joint − Γ_tt + 2∇R, assembled EXACTLY as
        // the gradient does (construction_exact_hessian.rs analytic assembler).
        let rank_charge = self.production_rank_charge_derivative(target, rho, loss, cache)?;
        let mut gamma_eff = self.logdet_theta_adjoint(rho, cache, &solver)?;
        let gamma_tt = self.coordinate_block_logdet_theta_adjoint(rho, cache, &solver)?;
        gamma_eff.t -= &gamma_tt.t;
        gamma_eff.beta -= &gamma_tt.beta;
        gamma_eff.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma_eff.beta.scaled_add(2.0, &rank_charge.theta.beta);

        // Adjoints: a = A⁺Γ_eff (once) and b_j = A⁺ g_ρ,j (per coordinate).
        let a_vec = self.solve_exact_stationarity(rho, target, cache, &solver, &gamma_eff)?;
        let a_flat = flatten(&a_vec);
        let flats: Vec<usize> = operators.keys().copied().collect();
        let mut b_flat: std::collections::BTreeMap<usize, Array1<f64>> =
            std::collections::BTreeMap::new();
        let mut g_rho_flat: std::collections::BTreeMap<usize, Array1<f64>> =
            std::collections::BTreeMap::new();
        for &j in &flats {
            let g_rho = self.outer_rho_gradient_ift_rhs(rho, j, cache)?;
            let b_j = self.solve_exact_stationarity(rho, target, cache, &solver, &g_rho)?;
            g_rho_flat.insert(j, flatten(&g_rho));
            b_flat.insert(j, flatten(&b_j));
        }

        let smooth_range =
            rho.smooth_flat_start()..rho.smooth_flat_start() + rho.log_lambda_smooth.len();
        let sparse_index = rho.sparse_flat_index();

        let mut hessian = Array2::<f64>::zeros((n_params, n_params));
        for &i in &flats {
            let m_i = &operators[&i];
            // Twisted inverses G_i = −G (∂A/∂ρ_i) G, h_bd_i = −h_bd (∂A/∂ρ_i) h_bd.
            // The Laplace logdet is logdet(A_exact), so ∂G/∂ρ_i differentiates the
            // EXACT stationarity Hessian ∂A/∂ρ_i = M_i + ΔC-delta_i — NOT the
            // majorized M_i alone, which is one-sided on ARD (delta ≠ 0 only for
            // ARD/softmax) and breaks g3 smooth↔ARD cross-conservation (#2330).
            let twist_op = match exact_deltas.get(&i) {
                Some(delta_i) => m_i + delta_i,
                None => m_i.clone(),
            };
            let g_i = -g.dot(&twist_op).dot(&g);
            let h_bd_i = -h_bd.dot(&twist_op).dot(&h_bd);

            // dΓ_joint/dρ_i and dΓ_tt/dρ_i = part(a) twist + part(b) mixed.
            let mut d_gamma_joint = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &g_i,
                ThetaAdjointDhChannel::All,
                false,
                false,
            )?;
            let mut d_gamma_tt = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &h_bd_i,
                ThetaAdjointDhChannel::All,
                false,
                false,
            )?;
            if smooth_range.contains(&i) {
                // Smooth part(b) = 0; the only smooth ρ-derivative of Γ_eff is
                // through the rank-charge adjoint.
                let d_rank = self.rank_charge_theta_rho_derivative(target, rho, loss, cache, i)?;
                d_gamma_joint.t.scaled_add(2.0, &d_rank.t);
                d_gamma_joint.beta.scaled_add(2.0, &d_rank.beta);
            } else if sparse_index == Some(i) {
                let mixed_joint = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &g,
                    ThetaAdjointDhChannel::SoftmaxSparseMixed,
                    false,
                    false,
                )?;
                let mixed_tt = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &h_bd,
                    ThetaAdjointDhChannel::SoftmaxSparseMixed,
                    false,
                    false,
                )?;
                d_gamma_joint.t += &mixed_joint.t;
                d_gamma_joint.beta += &mixed_joint.beta;
                d_gamma_tt.t += &mixed_tt.t;
                d_gamma_tt.beta += &mixed_tt.beta;
            } else {
                // ARD coordinate: part(b) mixed channel for this flat index.
                let mixed_joint = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &g,
                    ThetaAdjointDhChannel::ArdMixed { target_flat: i },
                    false,
                    false,
                )?;
                let mixed_tt = self.logdet_theta_adjoint_dense(
                    rho,
                    cache,
                    &h_bd,
                    ThetaAdjointDhChannel::ArdMixed { target_flat: i },
                    false,
                    false,
                )?;
                d_gamma_joint.t += &mixed_joint.t;
                d_gamma_joint.beta += &mixed_joint.beta;
                d_gamma_tt.t += &mixed_tt.t;
                d_gamma_tt.beta += &mixed_tt.beta;
            }

            // dΓ_eff/dρ_i = dΓ_joint − dΓ_tt (+2∇R' folded into joint above).
            let mut d_gamma = flatten(&d_gamma_joint);
            d_gamma -= &flatten(&d_gamma_tt);
            // resid_i = dΓ_eff/dρ_i − (∂A/∂ρ_i)·a, with ∂A/∂ρ_i = M_i + ΔC-delta_i
            // (the IFT term differentiates the EXACT A, not the majorized H).
            let mut a_op_i_a = m_i.dot(&a_flat);
            if let Some(delta_i) = exact_deltas.get(&i) {
                a_op_i_a += &delta_i.dot(&a_flat);
            }
            let resid_i = &d_gamma - &a_op_i_a;

            for &j in &flats {
                let b_j = &b_flat[&j];
                let mut term = resid_i.dot(b_j);
                if i == j {
                    term += a_flat.dot(&g_rho_flat[&j]);
                }
                hessian[[i, j]] = -0.5 * term;
            }
        }
        Ok(hessian)
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
        let mut smooth_explicit = self
            .decoder_smoothness_value_per_atom(&lambda_smooth_vec)
            .map_err(OuterGradientError::internal)?;
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
        // #2330 Phase-2: on the dense direct-logdet route (no probe bundle, no
        // matrix-free system) the ranked value is ½log|A|, so the logdet channels
        // must be A-based. Overwrite the B-majorizer logdet_trace + Γ assembled
        // above with the exact-A ones; explicit / occam / rank-charge-direct
        // channels are majorizer-independent and stay. The matrix-free / bundle
        // route keeps ½log|B| until Phase-2b (streaming signed-LDLᵀ A-factor).
        if logdet_derivative_bundle.is_none() && matrix_free_system.is_none() {
            let (exact_logdet_trace, exact_gamma) = self
                .dense_exact_a_logdet_channels(target, rho, loss, cache)
                .map_err(OuterGradientError::internal)?;
            logdet_trace = exact_logdet_trace;
            gamma = exact_gamma;
        }

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
    ///   `se_k = ½ λ_k f_k` to `C = loss.smoothness`, i.e. `g_k = C · se_k / Σse`.
    ///   But `C = penalty_scale · Σse` (construction.rs:4995), so the renormalizer
    ///   `renorm = C/Σse = penalty_scale` is ρ-INVARIANT — the `Σse` cancels — and
    ///   `g_k = renorm · se_k`. With `∂se_k/∂ρ_j = δ_{jk} se_k` the block is the
    ///   plain DIAGONAL `∂²/∂ρ_i∂ρ_j = renorm · δ_{ij} se_i`. (Holding `C` frozen
    ///   while `Σse` moves manufactures a spurious Occam cross term the
    ///   full-gradient FD reports as zero — the frozen-cache false-green genus.)
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
        let smooth_energy = self.decoder_smoothness_value_per_atom(&lambda_smooth)?;
        let energy_sum: f64 = smooth_energy.iter().sum();
        let k_smooth = rho.log_lambda_smooth.len();
        // The gradient's explicit smooth term is `g_k = C·se_k/Σse` with
        // `C = loss.smoothness = penalty_scale·Σse` (construction.rs:4995 — the
        // criterion energy IS the λ-scaled per-atom penalty times the minibatch
        // `penalty_scale`). So the renormalizer `renorm = C/Σse = penalty_scale`
        // is ρ-INVARIANT — the `Σse` in `C` cancels the denominator — and
        // `g_k = renorm·se_k`. Hence `∂g_k/∂ρ_j = renorm·δ_jk·se_k`: the block is
        // DIAGONAL. Holding `C` frozen while `Σse` moves (the old code) manufac-
        // tured a spurious Occam cross term `−renorm·se_a·se_b/Σse` that the
        // full-gradient FD (which recomputes `C` at each ρ) correctly reports as
        // zero. This is the frozen-cache false-green genus — the renormalizer must
        // be differentiated, not held constant.
        if energy_sum.abs() > 0.0 {
            let renorm = frozen_smoothness_energy / energy_sum;
            for a in 0..k_smooth {
                let ia = rho.smooth_flat_index(a);
                hessian[[ia, ia]] += renorm * smooth_energy[a];
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
                        let log_eta = log_alpha + 2.0 * (p.ln() - std::f64::consts::TAU.ln());
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
                let ranks: Vec<usize> = self.atoms.iter().map(|a| a.border_frame_rank()).collect();
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

    /// PATH C (#2253) — assemble the COMPLETE exact fixed-stratum dense outer
    /// Hessian for the small-dense ARC route from all four analytic channels
    /// (ch1 explicit smoothness/ARD, ch2 rank-charge direct, ch4 log-determinant
    /// Daleckii–Krein, ch5 third-order forward-sensitivity), enforce the
    /// coordinate-coverage invariant, and return `Ok(block)`.
    ///
    /// ch5 refuses for any config outside the covered small-dense softmax route
    /// (compact top-k layout, per-row deflation, border frames, non-softmax
    /// priors), and the crosscoder-block guard / coverage invariant refuse an
    /// unmodelled coordinate — those refusals propagate as `Err`, so this only
    /// returns `Ok` when the full block is assembled AND validated. The public
    /// [`Self::exact_fixed_stratum_outer_hessian`] currently wraps this in a
    /// staged `Err` (see its doc); the finite-difference gates call THIS assembler
    /// directly to validate the block.
    pub(crate) fn assemble_exact_fixed_stratum_outer_hessian(
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
        let n_params = rho.to_flat().len();
        let mut hessian = self.outer_explicit_smoothness_ard_hessian(rho, loss.smoothness)?;
        hessian += &self.rank_charge_direct_rho_hessian(target, rho, loss, cache)?;
        hessian += &self.logdet_daleckii_krein_hessian(rho, cache)?;
        // CH5 — the third-order forward-sensitivity channel completes the exact
        // fixed-stratum curvature. It refuses (propagated here) for any config
        // outside the covered small-dense softmax route, so a Dense Hessian is
        // never advertised where a sub-channel is unmodelled.
        hessian += &self.third_order_forward_sensitivity_hessian(target, rho, loss, cache)?;

        // Coordinate-coverage invariant (#2253), checked at assembly time on
        // EVERY call: every flat coordinate the outer gradient prices must own a
        // non-zero Hessian row. The priced set is assembled from the SAME
        // channels the gradient uses (per-atom smoothness, ARD axes, and the
        // softmax sparse log-strength coordinate when it is structurally live).
        // A live-gradient coordinate with an identically-zero Hessian row would
        // hand ARC a singular system, so refuse (naming the gap) rather than
        // advertise partial curvature. For the covered route ch1+ch4 already
        // fill every such row, so this passes; it is a guard against an
        // unhandled coordinate slipping through, not an expected refusal.
        let mut priced: Vec<usize> = Vec::new();
        for a in 0..rho.log_lambda_smooth.len() {
            priced.push(rho.smooth_flat_index(a));
        }
        for k in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[k].len() {
                let idx = rho.ard_flat_index(k, axis);
                if !priced.contains(&idx) {
                    priced.push(idx);
                }
            }
        }
        if let Some(sparse) = rho.sparse_flat_index() {
            if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) && self.k_atoms() > 1
            {
                priced.push(sparse);
            }
        }
        for &c in &priced {
            let row_is_live = (0..n_params).any(|j| hessian[[c, j]] != 0.0);
            if !row_is_live {
                return Err(format!(
                    "exact_fixed_stratum_outer_hessian: flat coordinate {c} carries a live \
                     outer-gradient component but an identically-zero Hessian row; refusing \
                     to advertise a curvature block with an unmodelled coordinate"
                ));
            }
        }
        Ok(hessian)
    }

    /// PATH C (#2253) — production entry for the exact fixed-stratum outer
    /// Hessian. COMMIT 1 (this): assemble AND validate the full block
    /// ([`Self::assemble_exact_fixed_stratum_outer_hessian`]) — exercising the
    /// config guards, all four channels, and the coordinate-coverage invariant —
    /// then keep returning `Err` so `eval` yields `HessianValue::Unavailable` and
    /// production stays on the analytic-gradient BFGS route during the blind
    /// window. The finite-difference gates validate the assembly by calling the
    /// assembler directly. COMMIT 2 (once the FD gate is green on MSI) replaces
    /// this body with the assembler's `Ok` result and flips `capability()` to
    /// `Dense` for the covered softmax config — a tiny separately-validated
    /// change that carries the wrong-curvature-steering risk out of this window.
    pub(crate) fn exact_fixed_stratum_outer_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        let hessian = self.assemble_exact_fixed_stratum_outer_hessian(target, rho, loss, cache)?;
        Err(format!(
            "PATH C exact fixed-stratum outer Hessian is assembled and validated \
             ({}×{}) but intentionally not advertised in commit 1: the Err→Ok + \
             capability→Dense flip lands as a separately-validated commit once the \
             finite-difference gate is green",
            hessian.nrows(),
            hessian.ncols()
        ))
    }

    /// Shared PD-classification floor for the exact observed information
    /// `A = B + ΔC` (#2330 / #2336). A converged inner mode is a genuine
    /// exact-Laplace maximum iff every eigenvalue of `A` is `≥ −floor`, with
    /// `floor = SAE_EXACT_A_PD_FLOOR_REL · max(max_eig, 1)`. The band
    /// `[−floor, floor]` is the radial-gauge quotient null (an exact ρ-invariant
    /// null of `A`, unit-pinned ⇒ `log 1 = 0`, `1/λ → 0`). #2330 ACCEPTS
    /// `min_eig > −floor`; #2336's saddle-escape TRIGGERS on `min_eig < −floor` —
    /// the same constant, so the two features cannot disagree in the band.
    pub(crate) const SAE_EXACT_A_PD_FLOOR_REL: f64 = 1.0e-9;

    /// #2330 Phase-2 — the EXACT observed-information Laplace log-determinants
    /// `(log|A|, log|A_tt|)` at the converged fixed-θ̂ mode, `A = ∇²_θθ L = B + ΔC`.
    /// One symmetric eigendecomposition per block; kept eigenvalues (`λ > floor`)
    /// contribute `ln λ`, the gauge quotient (`|λ| ≤ floor`) contributes 0, and a
    /// strictly negative eigenvalue (`λ < −floor`) is a saddle ⇒ typed
    /// `IndefiniteObservedInformation` refusal. `A_tt` drops the β border.
    pub(crate) fn exact_observed_information_log_dets(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(f64, f64), SaeCriterionError> {
        let total_t = cache.delta_t_len();
        let a = self
            .materialize_exact_hessian_dense(rho, target, cache)
            .map_err(SaeCriterionError::Numerical)?;
        let quotient_log_det =
            |m: &Array2<f64>, block: &'static str| -> Result<f64, SaeCriterionError> {
                let (eigs, _vecs) = m.eigh(Side::Lower).map_err(|e| {
                    SaeCriterionError::Numerical(format!(
                        "exact_observed_information_log_dets: {block} eigendecomposition failed: {e:?}"
                    ))
                })?;
                let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let floor = Self::SAE_EXACT_A_PD_FLOOR_REL * max_eig.max(1.0);
                let mut log_det = 0.0_f64;
                for &lambda in eigs.iter() {
                    if lambda < -floor {
                        return Err(SaeCriterionError::IndefiniteObservedInformation { block });
                    }
                    if lambda > floor {
                        log_det += lambda.ln();
                    }
                    // |lambda| <= floor: radial-gauge quotient null ⇒ log 1 = 0.
                }
                Ok(log_det)
            };
        let log_a = quotient_log_det(&a, "joint")?;
        let a_tt = a.slice(s![..total_t, ..total_t]).to_owned();
        let log_a_tt = quotient_log_det(&a_tt, "coordinate")?;
        Ok((log_a, log_a_tt))
    }

    /// #2330 Phase-2 — the quotient pseudo-inverses `(A⁺, A_tt⁺)` used by the
    /// exact-A outer-ρ gradient, from the SAME spectral classification the value
    /// uses: `A⁺ = Σ_{λ>floor} (1/λ) uᵀu`, dropping the `|λ|≤floor` gauge null.
    /// Both are returned as dense `dim×dim` operators (`A_tt⁺` has a zero β
    /// border) so they can feed `logdet_theta_adjoint_dense`'s border indexing
    /// exactly as `materialize_block_diag_t_inverse` does. Refuses an indefinite
    /// `A` (`λ < −floor`): the gradient must never be assembled at a saddle.
    pub(crate) fn materialize_exact_hessian_quotient_inverse(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let a = self.materialize_exact_hessian_dense(rho, target, cache)?;
        let pinv = |m: &Array2<f64>| -> Result<Array2<f64>, String> {
            let (eigs, vecs) = m.eigh(Side::Lower).map_err(|e| {
                format!("materialize_exact_hessian_quotient_inverse: eigh failed: {e:?}")
            })?;
            let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let floor = Self::SAE_EXACT_A_PD_FLOOR_REL * max_eig.max(1.0);
            let mut w = Array1::<f64>::zeros(eigs.len());
            for (idx, &lambda) in eigs.iter().enumerate() {
                if lambda < -floor {
                    return Err(format!(
                        "materialize_exact_hessian_quotient_inverse: indefinite A \
                         (λ={lambda:.3e}); the outer gradient must not be assembled at a saddle"
                    ));
                }
                w[idx] = if lambda > floor { 1.0 / lambda } else { 0.0 };
            }
            Ok(vecs.dot(&Array2::from_diag(&w)).dot(&vecs.t()))
        };
        let a_pinv = pinv(&a)?;
        let a_tt_block = a.slice(s![..total_t, ..total_t]).to_owned();
        let a_tt_pinv_small = pinv(&a_tt_block)?;
        let mut a_tt_pinv = Array2::<f64>::zeros((dim, dim));
        a_tt_pinv
            .slice_mut(s![..total_t, ..total_t])
            .assign(&a_tt_pinv_small);
        Ok((a_pinv, a_tt_pinv))
    }

    /// PATH C / #2330 — dense symmetric materialization of the EXACT stationarity
    /// Hessian `A = ∇²_θθ L = B + ΔC` (`dim×dim`, `dim = total_t + k`), built
    /// column by column via [`Self::apply_exact_hessian`] and symmetrized. The
    /// small-dense (circle-mint) scale this route already pays for
    /// [`Self::materialize_joint_inverse`]; shared by the observed-information
    /// log-determinant (VALUE) and its `A⁻¹` selected inverse (GRADIENT) so both
    /// factor one identical operator. `test_support`-scoped until Phase 2 wiring
    /// (see [`Self::exact_observed_information_log_dets`]).
    pub(crate) fn materialize_exact_hessian_dense(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut unit = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(k),
        };
        for col in 0..dim {
            if col < total_t {
                unit.t[col] = 1.0;
            } else {
                unit.beta[col - total_t] = 1.0;
            }
            let av = self.apply_exact_hessian(rho, target, cache, &unit)?;
            if col < total_t {
                unit.t[col] = 0.0;
            } else {
                unit.beta[col - total_t] = 0.0;
            }
            for r in 0..total_t {
                a[[r, col]] = av.t[r];
            }
            for r in 0..k {
                a[[total_t + r, col]] = av.beta[r];
            }
        }
        // The matrix-free apply is symmetric only up to round-off; symmetrize
        // so downstream Cholesky / selected-inverse factors see an exactly
        // symmetric operand.
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
                a[[r, c]] = avg;
                a[[c, r]] = avg;
            }
        }
        Ok(a)
    }

    /// #2330 Phase-2 — the A-based logdet gradient channels on the dense direct
    /// route: the direct trace vector `logdet_trace_i = ½tr(A⁺ ∂A/∂ρ_i)
    /// − ½tr(A_tt⁺ ∂A/∂ρ_i)` and the effective θ-adjoint
    /// `Γ_eff = tr(A⁺ ∂A/∂θ) − tr(A_tt⁺ ∂A_tt/∂θ) + 2∇R` (fed to the unchanged
    /// single-adjoint IFT collapse `a = A⁺Γ_eff`, `−½⟨a, g_ρ⟩`). `∂A/∂ρ_i =
    /// ∂B/∂ρ_i (penalty_curvature_operators_by_flat) + ∂ΔC/∂ρ_i
    /// (exact_stationarity_penalty_derivative_delta_by_flat)`, already exact. The
    /// θ-adjoint rides `exact_a = true` (ARD clamp-free) with `skip_deflation_dk
    /// = true` (the exact A carries only the ρ-invariant gauge null, handled by
    /// the quotient pseudo-inverse — no B-style Daleckii–Krein correction).
    ///
    /// EXACT-MINUS-PATCH-D: the two `logdet_theta_adjoint_dense` calls emit
    /// `∂B/∂θ + ∂ΔC_ard/∂θ` but NOT the residual-curvature / softmax-entropy legs
    /// of `∂ΔC/∂θ` (Patch D). Until D lands, Γ_eff — hence the IFT correction — is
    /// missing that term and the conservation bisection stays red by exactly it.
    pub(crate) fn dense_exact_a_logdet_channels(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<(Array1<f64>, SaeArrowVector), String> {
        let n_params = rho.to_flat().len();
        let (a_pinv, a_tt_pinv) =
            self.materialize_exact_hessian_quotient_inverse(rho, target, cache)?;
        let m = self.penalty_curvature_operators_by_flat(rho, cache)?;
        let d = self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)?;
        let frob = |x: &Array2<f64>, y: &Array2<f64>| -> f64 { (x * y).sum() };
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        for (&i, m_i) in m.iter() {
            let da = match d.get(&i) {
                Some(d_i) => m_i + d_i,
                None => m_i.clone(),
            };
            // A_tt⁺ has a zero β border, so frobbing it against the full ∂A/∂ρ_i
            // restricts to the t–t block automatically.
            logdet_trace[i] = 0.5 * frob(&a_pinv, &da) - 0.5 * frob(&a_tt_pinv, &da);
        }
        // Ordered-Beta–Bernoulli sparse coordinate: its ∂A/∂ρ_sparse is the exact
        // integrated-marginal logit Hessian (cross-row), absent from the operator
        // map above (softmax-only). Add its ½log|A| trace directly.
        if let Some(sparse) = rho.sparse_flat_index() {
            if matches!(
                self.assignment.mode,
                AssignmentMode::OrderedBetaBernoulli { .. }
            ) {
                logdet_trace[sparse] = self
                    .dense_exact_a_ordered_bb_sparse_trace(rho, cache, &a_pinv, &a_tt_pinv)?;
            }
        }
        let mut gamma = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &a_pinv,
            ThetaAdjointDhChannel::All,
            true,
            true,
        )?;
        let gamma_tt = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &a_tt_pinv,
            ThetaAdjointDhChannel::All,
            true,
            true,
        )?;
        gamma.t -= &gamma_tt.t;
        gamma.beta -= &gamma_tt.beta;
        let rank_charge = self.production_rank_charge_derivative(target, rho, loss, cache)?;
        gamma.t.scaled_add(2.0, &rank_charge.theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge.theta.beta);
        Ok((logdet_trace, gamma))
    }

    /// #2330 — the ordered-Beta–Bernoulli (non-softmax) sparse-coordinate ½log|A|
    /// trace `½[tr(A⁺ ∂A/∂ρ_sparse) − tr(A_tt⁺ ∂A/∂ρ_sparse)]`. For the
    /// non-learnable prior `∂A/∂ρ_sparse` is the EXACT integrated-marginal logit
    /// Hessian `H_obb` (linear-in-`weight` proof on the parent issue): its column
    /// `H_obb·e_j = ΔC_obb·e_j (cross-row HVP) + hdiag[j]·e_j (majorizer diagonal)`.
    /// The operator lives on logit t-slots only (no β border), so the coordinate
    /// block reuses the same columns against `A_tt⁺`. Learnable α (nonlinear
    /// concentration derivative) is refused, not silently mispriced.
    pub(crate) fn dense_exact_a_ordered_bb_sparse_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        a_pinv: &Array2<f64>,
        a_tt_pinv: &Array2<f64>,
    ) -> Result<f64, String> {
        if self.assignment.effective_alpha_is_learnable() {
            return Err(
                "dense_exact_a_ordered_bb_sparse_trace: learnable-α ordered-Beta–Bernoulli \
                 ∂A/∂ρ_sparse (nonlinear concentration derivative) is not yet modelled; refusing \
                 rather than emitting a wrong sparse ½log|A| trace"
                    .to_string(),
            );
        }
        let k_atoms = self.k_atoms();
        let n = self.n_obs();
        let row_weights = self.row_loss_weights.as_deref();
        // Global t-index of each (row, atom) logit slot in the cache layout.
        let mut logit_gindex: Vec<Vec<Option<usize>>> = vec![vec![None; k_atoms]; n];
        for row in 0..n {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            for (local, var) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *var {
                    if atom < k_atoms {
                        logit_gindex[row][atom] = Some(base + local);
                    }
                }
            }
        }
        // ∂B/∂ρ_sparse: the majorizer's diagonal log-strength derivative on the
        // logit slots — the SAME builder the B-majorizer trace uses.
        let mut hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if hdiag.is_empty() {
            // Inert / frozen prior: ∂B and ΔC are both zero.
            return Ok(0.0);
        }
        let channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if let Some(ch) = channels.as_ref() {
            for row in 0..n {
                for atom in 0..k_atoms {
                    let slot = row * k_atoms + atom;
                    hdiag[slot] =
                        super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_hdiag(
                            ch, row, k_atoms, atom, hdiag[slot],
                        );
                }
            }
        }
        // ½[tr(A⁺ ∂A/∂ρ_sparse) − tr(A_tt⁺ ∂A/∂ρ_sparse)], column by column over
        // the flat logit basis: ∂A/∂ρ_sparse·e_j = ΔC_obb·e_j + hdiag[j]·e_j.
        let n_logits = n * k_atoms;
        let mut e = Array1::<f64>::zeros(n_logits);
        let mut tr_joint = 0.0_f64;
        let mut tr_coord = 0.0_f64;
        for jrow in 0..n {
            for jatom in 0..k_atoms {
                let Some(gj) = logit_gindex[jrow][jatom] else {
                    continue;
                };
                let jflat = jrow * k_atoms + jatom;
                e[jflat] = 1.0;
                let dc = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                    &self.assignment,
                    rho,
                    row_weights,
                    e.view(),
                )?;
                e[jflat] = 0.0;
                for irow in 0..n {
                    for iatom in 0..k_atoms {
                        let val = dc[irow * k_atoms + iatom];
                        if val == 0.0 {
                            continue;
                        }
                        if let Some(gi) = logit_gindex[irow][iatom] {
                            tr_joint += a_pinv[[gi, gj]] * val;
                            tr_coord += a_tt_pinv[[gi, gj]] * val;
                        }
                    }
                }
                tr_joint += a_pinv[[gj, gj]] * hdiag[jflat];
                tr_coord += a_tt_pinv[[gj, gj]] * hdiag[jflat];
            }
        }
        Ok(0.5 * (tr_joint - tr_coord))
    }

}

#[cfg(test)]
mod test_support {
    use super::{
        ArrowFactorCache, DeflatedArrowSolver, SaeArrowVector, SaeManifoldRho,
        ThetaAdjointDhChannel,
    };
    use ndarray::{Array1, s};

    impl super::SaeManifoldTerm {
        /// PATH C (#2253) CH5 test-support — the max `|dense − production|` of the
        /// θ-adjoint reconstruction over the `(t, β)` blocks, for the joint
        /// (`inv = G`) and coordinate-block (`inv = h_bd`) legs. A failing FD gate
        /// uses this to separate a bug in the dense `dh` + Daleckii–Krein
        /// reproduction (this diverges from the trusted production builder) from a
        /// bug in the twist / rank-charge assembly (this is ~0 but the FD still
        /// reds). Both should be at solver noise.
        pub(crate) fn ch5_dense_theta_adjoint_selfcheck(
            &self,
            rho: &SaeManifoldRho,
            cache: &ArrowFactorCache,
        ) -> Result<(f64, f64), String> {
            let solver = DeflatedArrowSolver::plain(cache);
            let g = self.materialize_joint_inverse(cache, &solver)?;
            let h_bd = self.materialize_block_diag_t_inverse(cache);
            let dense_joint =
                self.logdet_theta_adjoint_dense(rho, cache, &g, ThetaAdjointDhChannel::All, false, false)?;
            let dense_tt = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &h_bd,
                ThetaAdjointDhChannel::All,
                false,
                false,
            )?;
            let prod_joint = self.logdet_theta_adjoint(rho, cache, &solver)?;
            let prod_tt = self.coordinate_block_logdet_theta_adjoint(rho, cache, &solver)?;
            let max_diff = |a: &SaeArrowVector, b: &SaeArrowVector| -> f64 {
                let t =
                    a.t.iter()
                        .zip(b.t.iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0_f64, f64::max);
                let beta = a
                    .beta
                    .iter()
                    .zip(b.beta.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                t.max(beta)
            };
            Ok((
                max_diff(&dense_joint, &prod_joint),
                max_diff(&dense_tt, &prod_tt),
            ))
        }

        /// #2330 split probe — the g3 cross non-conservation attributed to the
        /// trace vs the frozen-DK piece of `dΓ_joint/dρ_i`, per leg. Returns
        /// `⟨leg_i, b_j⟩` and `⟨leg_j, b_i⟩` for the (i,j) cross pair so the caller
        /// can assert cross-symmetry of each leg: part-a (twist `−G Mᵢ G`) trace,
        /// part-a DK, part-b (`∂Kw/∂ρ`) trace, part-b DK. The asymmetric leg is the
        /// leak. `with_dk` legs include `deflation_block_correction`; `_tr` legs
        /// pass `skip_deflation_dk = true`.
        pub(crate) fn ch5_twist_leg_cross(
            &self,
            rho: &SaeManifoldRho,
            target: ndarray::ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            i: usize,
            j: usize,
        ) -> Result<[(f64, f64); 4], String> {
            let solver = DeflatedArrowSolver::plain(cache);
            let g = self.materialize_joint_inverse(cache, &solver)?;
            let operators = self.penalty_curvature_operators_by_flat(rho, cache)?;
            // Mirror production: the twist inverse rides the EXACT ∂A/∂ρ = M_c + Δ.
            let exact_deltas = self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)?;
            let total_t = cache.delta_t_len();
            let dim = total_t + cache.k;
            let flatten = |v: &SaeArrowVector| -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(dim);
                out.slice_mut(s![..total_t]).assign(&v.t);
                out.slice_mut(s![total_t..]).assign(&v.beta);
                out
            };
            let smooth_range =
                rho.smooth_flat_start()..rho.smooth_flat_start() + rho.log_lambda_smooth.len();
            let sparse_index = rho.sparse_flat_index();
            // part-a (twist) and part-b (Kw ρ-deriv) legs of dΓ_joint/dρ_c, each in
            // trace-only and full (trace − DK) form, contracted against b_other.
            let leg = |c: usize, skip_dk: bool, part_a: bool| -> Result<Array1<f64>, String> {
                if part_a {
                    let twist_op = match exact_deltas.get(&c) {
                        Some(delta_c) => &operators[&c] + delta_c,
                        None => operators[&c].clone(),
                    };
                    let g_c = -g.dot(&twist_op).dot(&g);
                    Ok(flatten(&self.logdet_theta_adjoint_dense(
                        rho,
                        cache,
                        &g_c,
                        ThetaAdjointDhChannel::All,
                        skip_dk,
                        false,
                    )?))
                } else if smooth_range.contains(&c) {
                    Ok(Array1::<f64>::zeros(dim)) // smooth part-b is 0
                } else {
                    let channel = if sparse_index == Some(c) {
                        ThetaAdjointDhChannel::SoftmaxSparseMixed
                    } else {
                        ThetaAdjointDhChannel::ArdMixed { target_flat: c }
                    };
                    Ok(flatten(&self.logdet_theta_adjoint_dense(
                        rho, cache, &g, channel, skip_dk,
                        false,
                    )?))
                }
            };
            let b = |c: usize| -> Result<Array1<f64>, String> {
                let g_rho = self.outer_rho_gradient_ift_rhs(rho, c, cache)?;
                let solver = DeflatedArrowSolver::plain(cache);
                Ok(flatten(&self.solve_exact_stationarity(
                    rho, target, cache, &solver, &g_rho,
                )?))
            };
            let bi = b(i)?;
            let bj = b(j)?;
            // part_a_tr, part_a_dk, part_b_tr, part_b_dk cross pairs.
            let pa_full_i = leg(i, false, true)?;
            let pa_tr_i = leg(i, true, true)?;
            let pa_full_j = leg(j, false, true)?;
            let pa_tr_j = leg(j, true, true)?;
            let pb_full_i = leg(i, false, false)?;
            let pb_tr_i = leg(i, true, false)?;
            let pb_full_j = leg(j, false, false)?;
            let pb_tr_j = leg(j, true, false)?;
            let dot = |x: &Array1<f64>, y: &Array1<f64>| x.dot(y);
            Ok([
                (dot(&pa_tr_i, &bj), dot(&pa_tr_j, &bi)),
                (
                    dot(&(&pa_full_i - &pa_tr_i), &bj),
                    dot(&(&pa_full_j - &pa_tr_j), &bi),
                ),
                (dot(&pb_tr_i, &bj), dot(&pb_tr_j, &bi)),
                (
                    dot(&(&pb_full_i - &pb_tr_i), &bj),
                    dot(&(&pb_full_j - &pb_tr_j), &bi),
                ),
            ])
        }
    }
}
