
/// Reduce one contiguous device tile's rows into a private `-Σ leftᵀ·right`
/// partial (`k×k`).
///
/// The tile stacks its per-row `left_i` / `right_i` factors (each `d×k`) into
/// two `(Σ_i d_i × k)` matrices and tries a single per-ordinal `AᵀB` device
/// GEMM (`crate::gpu::try_fast_atb_on_ordinal`), which runs on the device this
/// worker thread already bound — one big GPU GEMM per tile rather than `n` small
/// CPU ones. When the device primitive declines (no GPU, shape below policy,
/// transient failure) the tile reduces with the exact CPU `block_gemm_subtract`
/// loop, so the result is unchanged. The partial is negated so the caller's
/// `schur += partial` reproduces the serial `schur -= Σ contribution`.
fn tile_schur_partial<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    ordinal: usize,
    range: Range<usize>,
) -> Array2<f64> {
    let k = sys.k;

    // Build the per-row contribution factors once; both the GPU stacked-GEMM
    // and the CPU fallback consume them.
    let mut factors: Vec<(Array2<f64>, Array2<f64>)> = Vec::with_capacity(range.len());
    let mut total_d = 0usize;
    for i in range.clone() {
        let (left, right) = row_schur_contribution_factors(
            sys,
            i,
            &sys.rows[i],
            htt_factors.factor(i),
            backend,
            kind,
        );
        total_d += left.nrows();
        factors.push((left, right));
    }

    // Stack into (total_d × k) left/right matrices for one device AᵀB GEMM on
    // this tile's bound ordinal. `try_fast_atb_on_ordinal` returns leftᵀ·right
    // (k×k); negate into the partial. At an SAE-shaped whole-fit tile with
    // n=2000 rows, k=2048 shared columns, M=12 local rows per observation, and
    // K=8 candidate/atom batches, the stacked GEMM is
    // 2*(n*M)*k^2 = 201_326_592_000 flops per batch, or
    // 1_610_612_736_000 flops across K=8, so the policy work gate is cleared
    // even though the observation count is far below the old row floor.
    if total_d > 0 && k > 0 {
        let mut left_stack = Array2::<f64>::zeros((total_d, k));
        let mut right_stack = Array2::<f64>::zeros((total_d, k));
        let mut base = 0usize;
        for (left, right) in &factors {
            let di = left.nrows();
            left_stack
                .slice_mut(ndarray::s![base..base + di, ..])
                .assign(left);
            right_stack
                .slice_mut(ndarray::s![base..base + di, ..])
                .assign(right);
            base += di;
        }
        if let Some(product) =
            crate::gpu::try_fast_atb_on_ordinal(ordinal, left_stack.view(), right_stack.view())
        {
            return product.mapv(|v| -v);
        }
    }

    // CPU fallback: exact per-row block_gemm_subtract into a zero-seeded partial.
    let mut partial = Array2::<f64>::zeros((k, k));
    for (left, right) in &factors {
        backend.block_gemm_subtract(&mut partial, left, right);
    }
    partial
}


/// Reduce the per-row Schur contributions `Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`
/// out of `schur` (seeded with `H_ββ + ρ_β·I`).
///
/// The per-row contributions are independent — exactly the "sum over independent
/// arrow-tip blocks" axis the device pool partitions. When more than one GPU is
/// usable, [`crate::gpu::pool::balanced_partition`] splits the `0..n` rows into
/// per-device contiguous tiles; each tile is reduced on its own scoped thread
/// (binding that ordinal's context so the per-row GEMM-subtract offloads to its
/// device) into a private `k×k` partial, and the partials are summed back into
/// `schur` in tile order. The tiles are contiguous, ordered to cover `0..n`, and
/// folded back in that same order, so within each tile the per-row accumulation
/// order is preserved and the only departure from the serial loop is the
/// inter-tile reassociation of the reduction sum — the established
/// reduction-order equivalence the device pool already operates under, well
/// inside the Newton solve's tolerance.
///
/// With a single device (or no GPU) the row loop runs serially in place, which
/// is bit-for-bit the original behaviour.
fn reduce_row_schur_contributions<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
) {
    let n = sys.rows.len();
    let k = sys.k;

    let tiles = crate::gpu::runtime::GpuRuntime::global()
        .map(|rt| crate::gpu::pool::balanced_partition(rt, n))
        .filter(|tiles| tiles.len() > 1);

    let Some(tiles) = tiles else {
        // Single-device / CPU: reduce serially in place (original order).
        for (i, row) in sys.rows.iter().enumerate() {
            subtract_row_schur_contribution(
                sys,
                i,
                row,
                htt_factors.factor(i),
                backend,
                kind,
                schur,
            );
        }
        return;
    };

    // Multi-GPU: one private `-Σ leftᵀ·right` partial per contiguous device
    // tile. Each tile runs on its own scoped worker thread that binds its
    // ordinal's context and issues a single stacked AᵀB GEMM on that device, so
    // the tiles' GEMMs overlap across the pool. Folding the partials back into
    // the H_ββ-seeded `schur` reproduces the serial reduction (up to inter-tile
    // reassociation).
    let partials: Vec<Array2<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = tiles
            .iter()
            .map(|(ordinal, range)| {
                let ordinal = *ordinal;
                let range = range.clone();
                scope.spawn(move || {
                    // Bind this ordinal's CUDA context on this worker thread so
                    // the per-row GPU GEMM shims issued from `tile_schur_partial`
                    // offload to that device. A missing context or bind failure
                    // is intentionally consumed without escalation — the shims
                    // no-op back to CPU and the math is unchanged. Off Linux
                    // `GpuRuntime::global()` is always `None`, so this branch
                    // is unreachable and the bind is omitted entirely.
                    #[cfg(target_os = "linux")]
                    {
                        if let Some(ctx) = crate::gpu::runtime::cuda_context_for(ordinal) {
                            if ctx.bind_to_thread().is_err() {
                                // Fall through: this tile reduces on the CPU.
                            }
                        }
                    }
                    tile_schur_partial(sys, htt_factors, backend, kind, ordinal, range)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("schur-reduction tile thread panicked"))
            .collect()
    });

    // Fold partials into `schur` in tile order (contiguous, covering 0..n) so
    // the per-tile and inter-tile accumulation order is the row order; each
    // partial holds `-Σ contribution` over its rows, so `schur += partial`
    // reproduces `schur -= Σ contribution`.
    for partial in &partials {
        for a in 0..k {
            for b in 0..k {
                schur[[a, b]] += partial[[a, b]];
            }
        }
    }
}


fn build_dense_schur_direct<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;
    // Materialise H_ββ via the BetaPenaltyOp trait (#296): DensePenaltyOp
    // for the legacy dense path, structured ops for SAE / Kronecker smooths.
    let op = sys.effective_penalty_op();
    if op.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Direct BA requires a K×K shared H_ββ penalty operator".to_string(),
        });
    }
    let mut schur = op.to_dense();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    reduce_row_schur_contributions(
        sys,
        htt_factors,
        backend,
        SchurReductionKind::Direct,
        &mut schur,
    );
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}


fn build_dense_schur_sqrt_ba<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;
    // Materialise H_ββ via the BetaPenaltyOp trait (#296).
    let op = sys.effective_penalty_op();
    if op.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Square-Root BA direct solve requires a K×K shared H_ββ penalty operator"
                .to_string(),
        });
    }
    let mut schur = op.to_dense();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    reduce_row_schur_contributions(
        sys,
        htt_factors,
        backend,
        SchurReductionKind::SqrtBa,
        &mut schur,
    );
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}


/// Certified Carson–Higham mixed-precision solve of the reduced dense Schur
/// system `S Δβ = rhs` (#1014), specialized to the streaming/residency path.
///
/// Returns `Some(Δβ)` when certified mixed precision is enabled AND the κ gate
/// admits the f32 factorization AND the f64 backward-error certificate closes;
/// `None` in every other case so the caller falls back to the exact f64
/// triangular solve. The f64 `factor` (whose diagonal carries the exact
/// `log|S|`) is supplied by the caller and never re-derived here — the logdet
/// the evidence path reads stays f64 by construction.
///
/// Method: store the f64 Cholesky factor as f32, solve in f32, then refine with
/// residuals `r = rhs − S·x` computed in f64 against the f64 `S`. With
/// `κ(S)·u_f32 < margin` the refinement contracts at rate `κ·u`, and the
/// terminating certificate is the normwise backward error
/// `‖r‖∞ / (‖S‖∞‖x‖∞ + ‖rhs‖∞) ≤ tol`. A non-decreasing residual or an
/// unmet certificate after `max_refinement_steps` returns `None`.
fn mixed_precision_reduced_beta(
    schur: &Array2<f64>,
    factor: &Array2<f64>,
    rhs: &Array1<f64>,
    options: &ArrowSolveOptions,
) -> Option<Array1<f64>> {
    let MixedPrecisionPolicy::Certified {
        max_refinement_steps,
        residual_relative_tolerance,
        kappa_unit_roundoff_margin,
    } = options.mixed_precision
    else {
        return None;
    };
    // The reduced-system mixed-precision path is the dense reduced solve only;
    // a trust-region-truncated step takes the Steihaug branch below in f64.
    if options.trust_region.radius.is_finite() {
        return None;
    }
    let n = schur.nrows();
    if n == 0 {
        return None;
    }

    // κ gate: the f32 factorization is only admissible when κ(S)·u_f32 leaves
    // the refinement contraction headroom the certificate needs.
    let kappa = cholesky_factor_kappa_estimate(factor);
    if !kappa.is_finite() || kappa * F32_UNIT_ROUNDOFF >= kappa_unit_roundoff_margin {
        return None;
    }

    let factor_f32 = factor.mapv(|v| v as f32);
    let s_inf = matrix_inf_norm(schur);
    let rhs_inf = rhs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let certificate_tol = residual_relative_tolerance
        .max(MIXED_PRECISION_CERTIFICATE_EPSILON_MULTIPLIER * f64::EPSILON);

    // f32 solve of the seed system, then f64-residual refinement steps.
    let mut x = cholesky_solve_lower_f32(&factor_f32, &rhs.mapv(|v| v as f32)).mapv(|v| v as f64);
    let mut last_residual = f64::INFINITY;
    for _ in 0..=max_refinement_steps {
        // Residual r = rhs − S·x in f64 against the f64 model.
        let sx = schur.dot(&x);
        let mut r = rhs.clone();
        r -= &sx;
        let r_inf = r.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let x_inf = x.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        let denom = s_inf * x_inf + rhs_inf;
        let backward_error = if denom > 0.0 { r_inf / denom } else { 0.0 };
        if backward_error <= certificate_tol {
            return Some(x);
        }
        // Refinement must make monotone progress, else hand back to f64.
        if !(r_inf < last_residual) {
            return None;
        }
        last_residual = r_inf;
        // Correction solve in f32 against the f32 factor: S·δ = r.
        let delta = cholesky_solve_lower_f32(&factor_f32, &r.mapv(|v| v as f32)).mapv(|v| v as f64);
        x += &delta;
    }
    None
}


/// Infinity norm (max absolute row sum) of a dense matrix.
fn matrix_inf_norm(a: &Array2<f64>) -> f64 {
    let mut max_row = 0.0_f64;
    for row in a.rows() {
        let s: f64 = row.iter().map(|v| v.abs()).sum();
        if s > max_row {
            max_row = s;
        }
    }
    max_row
}


fn solve_dense_reduced_system(
    schur: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    options: &ArrowSolveOptions,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, Option<Array2<f64>>, PcgDiagnostics), ArrowSchurError> {
    let factor = match cholesky_lower(schur) {
        Ok(factor) => factor,
        Err(e) => {
            // Evidence/log-det-only callers must not die on a genuinely non-PD
            // reduced Schur complement (#1118 β-block analogue). On a
            // rank-deficient multi-atom dictionary the per-row H_tt blocks are
            // unit-stiffness deflated to stay PD, but the Schur subtraction
            // `Σ H_tβᵀ H_tt⁻¹ H_tβ` can still drive a β-complement pivot negative
            // off the inner optimum (the reported `-0.064 at index 256` on the
            // OLMo K=8 capstone). Condition the offending eigen-directions to
            // unit stiffness exactly as the per-row evidence path does: the
            // deflated directions contribute a ρ-independent `log 1 = 0` to
            // `log|S|`, so the evidence value stays consistent with the analytic
            // ρ-gradient and the EV≥0 / finite-normaliser guarantee is preserved.
            // The discarded Δβ is solved against the conditioned factor (harmless
            // — evidence mode ignores it). Non-evidence (step-accuracy) callers
            // still surface the hard `SchurFactorFailed` so the outer LM loop can
            // lift `ridge_beta` and re-form a genuinely PD complement.
            if options.tolerate_ill_conditioning {
                if let Some(deflated) = factor_spectral_deflated_evidence_dense(schur) {
                    let delta_beta = cholesky_solve_vector(&deflated, rhs_beta);
                    return Ok((delta_beta, Some(deflated), PcgDiagnostics::default()));
                }
            }
            return Err(ArrowSchurError::SchurFactorFailed { reason: e });
        }
    };
    // Ill-conditioned-but-PD Schur guard. The per-row factor checks reject
    // any single barely-PD H_tt^(i) block, but the reduced Schur complement
    //     S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
    // accumulates the (H_tt^(i))⁻¹ contributions of every row in finite
    // precision. With many weak-but-admissible rows those terms can sum to a
    // Schur matrix whose Cholesky succeeds yet whose condition number is far
    // past the safe inversion regime, so `cholesky_solve_vector` yields an
    // inaccurate Δβ that is silently propagated to the Newton step. Apply the
    // same diagonal-ratio κ proxy used per-row to the reduced factor and treat
    // an over-threshold estimate as a Schur-stability failure: `SchurFactorFailed`
    // is already recoverable in `solve_with_lm_escalation_inner`, so this lifts
    // `ridge_beta` and re-forms a better-conditioned Schur. This guard is
    // exclusive to the dense Direct / SqrtBA path (the only caller of this
    // function); the inexact-PCG path tolerates higher κ(S) and is unaffected.
    // Evidence/log-det-only callers (`tolerate_ill_conditioning`) skip this
    // rejection: the factor is genuinely PD (Cholesky above succeeded), so its
    // diagonal still yields an exact `log|S|`, and an inaccurate Δβ is harmless
    // because the step is discarded.
    if !options.tolerate_ill_conditioning {
        let schur_kappa = cholesky_factor_kappa_estimate(&factor);
        if !schur_kappa.is_finite() || schur_kappa > safe_spd_kappa_max(schur.nrows()) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "reduced Schur complement Cholesky succeeded but is ill-conditioned \
                     (kappa_estimate={schur_kappa:e}); accumulated per-row \
                     (H_tt)⁻¹ contamination would yield an inaccurate Δβ"
                ),
            });
        }
    }
    // Reduced-system solve. The f64 `factor` is always retained and returned —
    // its diagonal is the EXACT `log|S|` the evidence path reads, so the logdet
    // stays f64 regardless of how Δβ is computed (#1014 invariant). When the
    // streaming/residency path enabled certified mixed precision, the Δβ solve
    // itself runs f32-then-f64-refined (κ-gated, with the f64 triangular solve
    // as the automatic fallback); the certificate is the f64 backward error.
    let direct = mixed_precision_reduced_beta(schur, &factor, rhs_beta, options)
        .unwrap_or_else(|| cholesky_solve_vector(&factor, rhs_beta));
    if step_inside_trust_region(direct.view(), options.trust_region.radius, metric_weights) {
        return Ok((direct, Some(factor), PcgDiagnostics::default()));
    }

    // Ceres-style trust-region correction: once the dense BA solve proposes a
    // step outside the trust ball, Steihaug-CG returns the boundary point
    // without requiring a second dense factorization.
    let identity = IdentityPreconditioner;
    let (delta, diag) = steihaug_dense_system(
        schur,
        rhs_beta,
        &identity,
        &ArrowPcgOptions {
            max_iterations: options.trust_region.max_iterations,
            relative_tolerance: options.trust_region.steihaug_relative_tolerance,
        },
        &options.trust_region,
        metric_weights,
    )?;
    Ok((delta, Some(factor), diag))
}


/// Solve an externally accumulated dense reduced β system
/// `S Δβ = rhs_β` with the same LM-style ridge escalation the full-batch
/// driver applies: on a `SchurFactorFailed` (non-PD or ill-conditioned `S`),
/// geometrically grow a proximal ridge on `S`'s diagonal and retry.
///
/// Used by the SAE streaming joint fit, which accumulates `S` and `rhs_β` over
/// re-materialized row chunks (via [`StreamingArrowSchur::take_accumulators`])
/// and must solve the single global reduced system without a per-row
/// `ArrowSchurSystem`. `S` is symmetrized from its lower triangle before each
/// factorization. `base_ridge_beta` is folded into the caller's `S` already;
/// this routine only adds the *escalation* ridge on top.
pub fn solve_streaming_reduced_beta(
    s_acc: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    options: &ArrowSolveOptions,
) -> Result<Array1<f64>, ArrowSchurError> {
    let mut proximal_ridge = 0.0_f64;
    let mut last_err: Option<ArrowSchurError> = None;
    for attempt in 0..=DEFAULT_PROXIMAL_MAX_ATTEMPTS {
        let mut schur = s_acc.clone();
        symmetrize_upper_from_lower(&mut schur);
        if proximal_ridge > 0.0 {
            for j in 0..schur.nrows() {
                schur[[j, j]] += proximal_ridge;
            }
        }
        // Reduced K-system on device: Jacobi-preconditioned CG over the dense
        // symmetric `S`. The `O(K²)` `S·p` matvec runs device-side; only the
        // K-vectors cross the boundary per CG iteration. This is the dominant
        // cost of the streaming SAE joint fit at `K = 100K`. Any device-side
        // failure (`Unavailable`, non-PD Jacobi diagonal) falls through to the
        // CPU `solve_dense_reduced_system`, which then drives the same proximal
        // ridge escalation. A genuine device PD failure is non-recoverable for
        // this attempt's `schur`, so we let the CPU path re-confirm and escalate.
        if crate::gpu::runtime::GpuRuntime::is_available() {
            match crate::gpu::arrow_schur::solve_reduced_beta_pcg(
                &schur,
                rhs_beta,
                options.trust_region.max_iterations,
                options.trust_region.steihaug_relative_tolerance,
            ) {
                Ok(delta_beta) => return Ok(delta_beta),
                Err(crate::gpu::arrow_schur::ArrowSchurGpuFailure::Unavailable) => {}
                Err(_) => {
                    // Device declined this `schur` (e.g. non-PD Jacobi diag);
                    // let the CPU path confirm and escalate the proximal ridge.
                }
            }
        }
        match solve_dense_reduced_system(&schur, rhs_beta, options, None) {
            Ok((delta_beta, _factor, _diag)) => return Ok(delta_beta),
            Err(err) => {
                let recoverable = matches!(
                    err,
                    ArrowSchurError::SchurFactorFailed { .. } | ArrowSchurError::PcgFailed { .. }
                );
                last_err = Some(err);
                if !recoverable || attempt == DEFAULT_PROXIMAL_MAX_ATTEMPTS {
                    break;
                }
                proximal_ridge = if proximal_ridge == 0.0 {
                    DEFAULT_PROXIMAL_INITIAL_RIDGE
                } else {
                    proximal_ridge * DEFAULT_PROXIMAL_RIDGE_GROWTH
                };
            }
        }
    }
    Err(last_err.expect("escalation loop set last_err on failure"))
}


fn step_inside_trust_region(
    step: ArrayView1<'_, f64>,
    radius: f64,
    metric_weights: Option<&MetricWeights>,
) -> bool {
    !radius.is_finite() || metric_norm(step, metric_weights) <= radius
}


/// Below this row count the per-row Schur loop stays sequential: the rayon
/// fan-out (chunk dispatch + the deterministic per-chunk length-`K` reduction)
/// costs more than it saves for the handful-of-rows arrow systems that dominate
/// the non-SAE callers. Above it — the SAE LLM shape (`n` in the thousands,
/// wide border `k`) that issue #1017 names — the per-row `H_βt (H_tt)⁻¹ H_tβ x`
/// contributions are the matvec's whole cost and parallelize cleanly.
const SCHUR_MATVEC_PARALLEL_ROW_MIN: usize = 256;


/// Below this border width `k` the dense `H_ββ` penalty-prologue GEMV stays
/// sequential: parallelizing a `k×k` matvec only pays once `k²` is large enough
/// to dwarf the rayon fan-out, which for the arrow callers with narrow borders
/// it never is. At the SAE LLM border (`k` in the low thousands) the `O(k²)`
/// prologue is ≈4M flops/CG-iteration and was the serial Amdahl ceiling on the
/// otherwise per-row-parallel matvec (#1017), so it crosses this threshold and
/// fans out. 512 keeps the prologue serial for every non-SAE arrow system while
/// engaging it for the wide SAE/Qwen borders the issue targets.
const SCHUR_PROLOGUE_PARALLEL_K_MIN: usize = 512;


/// Device-residency CPU analogue for the SAE reduced-Schur matvec (#1017).
///
/// In the production SAE joint fit the per-row cross-block factors as
/// `H_tβ^(i) = L_i P_i`, where `L_i` (`q_i × p`) is the row's local
/// assignment/coordinate Jacobian and `P_i` (`p × K`, sparse) gathers the
/// active atoms' decoder blocks (`P_i x = Σ_s φ_s · x[base_s .. base_s+p]`).
/// The reduced-Schur point-elimination contribution of one row is therefore
///
/// ```text
/// S_i x = H_βt^(i) (H_tt^(i)+ρ_t I)⁻¹ H_tβ^(i) x
///       = P_iᵀ · [ L_iᵀ (H_tt^(i)+ρ_t I)⁻¹ L_i ] · P_i x
///       = P_iᵀ G_i (P_i x),      G_i := L_iᵀ (H_tt^(i)+ρ_t I)⁻¹ L_i   (p×p).
/// ```
///
/// The block `G_i = L_iᵀ Y_i` depends only on the assembled per-row blocks and
/// the (already-computed, solve-stable) `H_tt` factor — NOT on the CG iterate
/// `x`. The generic [`schur_matvec`] re-walks `apply_jbeta → apply_l →
/// solve(d×d) → apply_l_t → scatter` on every CG iteration; this object **stages
/// the factors `(L_i, Y_i)` once per CG solve** (the "upload X once" residency
/// mechanism, applied on CPU to the matvec rather than a dense factorization),
/// turning each subsequent matvec into a sparse gather → two `di×p` GEMVs →
/// sparse scatter, with no per-iteration triangular solve and no operator-closure
/// re-walk. It never materialises the dense `p×p` product: `di ≪ p` for SAE
/// rows, so the factored apply is `2·di·p` flops/row (vs `p²`) and `O(n·di·p)`
/// memory (vs `O(n·p²)` ≈ 67 GB at the Qwen shape — the dense form is OOM).
///
/// Numerically identical to the generic path up to floating-point reassociation
/// (it differentiates and accumulates the SAME quotient), so the criterion
/// ranking across topology candidates cannot move — the #1017 verification gate.
pub(crate) struct SaeResidentReducedSchur {
    /// Decoder output dimension `p` (the side length of every `G_i = L_iᵀ Y_i`).
    p: usize,
    /// Per-row **factored** residency: `(L_i, Y_i)`, each stored row-major as a
    /// `di × p` slab (`L_i` = local Jacobian, `Y_i = (H_tt^(i)+ρ_t I)⁻¹ L_i`).
    /// The reduced block is `G_i = L_iᵀ Y_i` (`p×p`, symmetric PSD), but it has
    /// rank ≤ `di` and `di ≪ p` for SAE rows (the per-row latent dim is 1–2
    /// while `p` is the decoder block width, ~2048). Materialising the dense
    /// `p×p` block would cost `O(n·p²)` memory (≈67 GB at the Qwen shape) and
    /// `p²` flops per matvec/row; the factored form costs `O(n·di·p)` memory and
    /// `2·di·p` flops/row, applying `G_i v = L_iᵀ (Y_i v)` (gather → `di`-length
    /// GEMV → `p`-length GEMV → scatter). A row with empty active support /
    /// degenerate dims gets `di = 0` and is skipped.
    /// `(di, L_i, Y_i)` per row; `L_i`/`Y_i` are `di·p`-length row-major buffers.
    rows: Vec<ResidentRowFactor>,
    /// Per-row active atom support `(β-block base index, φ weight)`, shared with
    /// the assembler's [`DeviceSaePcgData`] (no re-clone of the index lists).
    a_phi: Arc<[Vec<(usize, f64)>]>,
}


/// Factored per-row residency block: `G_i = L_iᵀ Y_i` kept as its `di×p` factors
/// so the matvec never materialises the dense `p×p` product. See
/// [`SaeResidentReducedSchur`].
struct ResidentRowFactor {
    /// Row latent dimension `di` (the inner contraction width). `0` ⇒ skipped.
    di: usize,
    /// `L_i` row-major `di × p` (`di·p` entries). Empty when `di == 0`.
    l: Vec<f64>,
    /// `Y_i = (H_tt^(i)+ρ_t I)⁻¹ L_i` row-major `di × p`. Empty when `di == 0`.
    y: Vec<f64>,
}


impl SaeResidentReducedSchur {
    /// Stage the per-row `G_i = L_iᵀ (H_tt^(i)+ρ_t I)⁻¹ L_i` blocks once, from
    /// the SAE structure (`DeviceSaePcgData`: `p`, per-row `a_phi`, per-row
    /// row-major `local_jac` = `L_i`) and the already-factored `H_tt` slab.
    ///
    /// Returns `None` when the structure does not match (degenerate `p`, row
    /// count mismatch) so the caller falls back to the generic matvec. Row
    /// builds are independent and run under the same deterministic rayon
    /// discipline as the matvec (each `G_i` is self-contained — no cross-row
    /// reduction — so there is no ordering subtlety).
    /// `ridge_t` is NOT a parameter: it is already folded into the factored
    /// blocks `htt_factors` carry (they factor `H_tt^(i) + ridge_t·I` — see
    /// `factor_blocks`), so solving against the factor yields `(H_tt^(i)+ρ_t I)⁻¹`
    /// exactly. The residency block is a pure function of the factor and `L_i`.
    fn build<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        backend: &B,
    ) -> Option<Self> {
        let data = sys.device_sae_pcg.as_ref()?;
        let p = data.p;
        let n = sys.rows.len();
        if p == 0 || data.a_phi.len() != n || data.local_jac.len() != n {
            return None;
        }
        let empty = || ResidentRowFactor {
            di: 0,
            l: Vec::new(),
            y: Vec::new(),
        };
        let build_row = |row: usize| -> ResidentRowFactor {
            let di = sys.row_dims[row];
            let jac = &data.local_jac[row];
            // q_i = len/p; must match the row's latent dimension di.
            if p == 0 || jac.len() != di * p || di == 0 {
                return empty();
            }
            // L_i as a (di × p) matrix (row-major in `local_jac`).
            let l_i = match ArrayView2::from_shape((di, p), jac.as_slice()) {
                Ok(v) => v.to_owned(),
                Err(_) => return empty(),
            };
            // Solve (H_tt+ρ_t I) Y = L_i for Y (di × p): one batched back-solve
            // over the p columns against the cached factor. Stage `(L_i, Y_i)`
            // — NOT the dense `p×p` product `G_i = L_iᵀ Y_i` — so storage and the
            // matvec stay `O(di·p)` instead of `O(p²)` (`di ≪ p` for SAE rows).
            let y = backend.solve_block_matrix(htt_factors.factor(row), l_i.view());
            // Flatten both factors to `di × p` row-major buffers (iteration over
            // a standard-layout view is row-major regardless of the source
            // strides, so the hot loop can index `r*p + c` directly).
            let l_flat: Vec<f64> = l_i.iter().copied().collect();
            let y_flat: Vec<f64> = y.iter().copied().collect();
            ResidentRowFactor {
                di,
                l: l_flat,
                y: y_flat,
            }
        };
        let rows: Vec<ResidentRowFactor> =
            if n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
                use rayon::prelude::*;
                (0..n).into_par_iter().map(build_row).collect()
            } else {
                (0..n).map(build_row).collect()
            };
        Some(Self {
            p,
            rows,
            a_phi: data.a_phi_shared(),
        })
    }

    /// Accumulate one row's `S_i x = P_iᵀ G_i (P_i x) = P_iᵀ L_iᵀ Y_i (P_i x)`
    /// into `acc` (length `K`). `gather`/`prod` are caller-owned length-`p`
    /// buffers and `w` a caller-owned `≥ max_i di`-length buffer, all reused
    /// across rows to keep the hot loop allocation-free. The matvec applies the
    /// factored block: `w = Y_i·(P_i x)` (`di`-length, `di·p` flops) then
    /// `prod = L_iᵀ·w` (`p`-length, `di·p` flops) — `2·di·p` total, never the
    /// dense `p²` product.
    #[inline]
    fn row_into(
        &self,
        row: usize,
        x: &Array1<f64>,
        acc: &mut Array1<f64>,
        gather: &mut [f64],
        prod: &mut [f64],
        w: &mut [f64],
    ) {
        let rf = &self.rows[row];
        let di = rf.di;
        if di == 0 {
            return;
        }
        let p = self.p;
        let support = &self.a_phi[row];
        if support.is_empty() {
            return;
        }
        // P_i x = Σ_s φ_s · x[base_s .. base_s+p]   (length p).
        for v in gather.iter_mut() {
            *v = 0.0;
        }
        for &(base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            for j in 0..p {
                gather[j] += phi * x[base + j];
            }
        }
        // w = Y_i · (P_i x)   (di × p GEMV → length di).  Y_i row-major di×p.
        for r in 0..di {
            let yrow = &rf.y[r * p..r * p + p];
            let mut s = 0.0_f64;
            for c in 0..p {
                s += yrow[c] * gather[c];
            }
            w[r] = s;
        }
        // prod = L_iᵀ · w   (p × di GEMV → length p).  L_i row-major di×p, so
        // L_iᵀ[j,r] = L_i[r,j]; accumulate column-by-column over the di rows.
        for v in prod.iter_mut().take(p) {
            *v = 0.0;
        }
        for r in 0..di {
            let lrow = &rf.l[r * p..r * p + p];
            let wr = w[r];
            for j in 0..p {
                prod[j] += lrow[j] * wr;
            }
        }
        // acc += P_iᵀ prod = scatter φ_s · prod into base_s blocks.
        for &(base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            for j in 0..p {
                acc[base + j] += phi * prod[j];
            }
        }
    }

    /// Max row latent dim `di` across resident rows — the size of the `w`
    /// scratch the matvec needs for the inner `Y_i·(P_i x)` GEMV.
    fn max_di(&self) -> usize {
        self.rows.iter().map(|r| r.di).max().unwrap_or(0)
    }
}


/// Reduced-Schur matvec `out = S·x` with an optional pre-staged SAE residency
/// operator. When `resident` is `Some`, the per-row point-elimination term is
/// applied through the resident `p×p` blocks (#1017 CPU residency); otherwise it
/// falls back to the generic per-row `apply → solve → transpose` path. Both
/// routes accumulate the SAME reduced operator
/// `S = H_ββ + ρ_β I − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)`.
fn schur_matvec<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
) {
    // `steihaug_cg` reuses one output buffer across iterations and requires
    // `matvec` to ASSIGN every entry of `out` (the contract `dense_matvec`
    // upholds). This routine builds `S·x` purely by accumulation
    // (`penalty_matvec_add`, `out[a] += ridge·x`, `out[a] -= neg_contrib`), so it
    // MUST clear `out` first. Without this, iteration n>0 returns `S·x` plus the
    // previous call's `S·p`, the PCG solves a corrupted reduced system, and the
    // resulting Newton step is inconsistent with the assembled gradient
    // (g·δ ≈ 0 — a non-descent direction that defeats the line search).
    out.fill(0.0);
    let k = sys.k;
    // Top-level (not nested in a rayon worker) and big enough to amortize the
    // fan-out: the single gate that authorizes BOTH the dense penalty-prologue
    // GEMV and the per-row point-elimination loop to go parallel. The topology
    // race fans candidates with `run_topology_race_parallel`, so inside a worker
    // both stay sequential (no nested-rayon oversubscription).
    let parallel =
        sys.rows.len() >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    // Route the penalty-side (H_ββ + ridge·I) x product through the prologue:
    // no Arc-clone hot-path cost when penalty_op is None (falls back to hbb
    // inline); the dense fallback fans across cores at the wide SAE border (#1017).
    {
        let x_slice = x.as_slice().expect("x must be contiguous");
        let out_slice = out.as_slice_mut().expect("out must be contiguous");
        sys.penalty_ridge_prologue_into(x_slice, ridge_beta, out_slice, parallel);
    }
    // The reduced-Schur point-elimination term: `out -= Σ_i H_βt^(i) (H_tt^(i))⁻¹
    // H_tβ^(i) x`. Each row contributes an independent length-`K` vector, so for
    // the SAE LLM shape (#1017) this is the matvec's whole cost and is
    // embarrassingly parallel. Run it under rayon over fixed row chunks, summing
    // the per-chunk partials in chunk order so the f64 reduction is bit-identical
    // run-to-run regardless of thread scheduling (the #1017 verification gate:
    // the criterion ranking across topology candidates must not move). Stay
    // sequential when already inside a rayon worker (the topology race fans
    // candidates with `run_topology_race_parallel`) to avoid nested-rayon
    // oversubscription — the same guard `HyperOperator::mul_mat` uses. The
    // `parallel` gate above authorizes this loop too.
    let p = resident.map(|r| r.p).unwrap_or(0);
    if parallel {
        use rayon::prelude::*;
        const CHUNK: usize = 64;
        let n = sys.rows.len();
        let partials: Vec<Array1<f64>> = (0..n)
            .into_par_iter()
            .chunks(CHUNK)
            .map(|idxs| {
                let mut acc = Array1::<f64>::zeros(k);
                if let Some(res) = resident {
                    // Resident path: each matvec is gather → factored di×p GEMVs
                    // → scatter, reading only the pre-staged `(L_i, Y_i)` (no
                    // per-iteration solve, no dense p×p block).
                    let mut gather = vec![0.0_f64; p];
                    let mut prod = vec![0.0_f64; p];
                    let mut w = vec![0.0_f64; res.max_di()];
                    for i in idxs {
                        res.row_into(i, x, &mut acc, &mut gather, &mut prod, &mut w);
                    }
                } else {
                    let mut local = Array1::<f64>::zeros(sys.d);
                    for i in idxs {
                        schur_matvec_row_into(
                            sys,
                            htt_factors,
                            x,
                            backend,
                            i,
                            &mut local,
                            &mut acc,
                        );
                    }
                }
                acc
            })
            .collect();
        // Deterministic ordered reduction: fold chunk partials left-to-right.
        for acc in &partials {
            for a in 0..k {
                out[a] -= acc[a];
            }
        }
    } else if let Some(res) = resident {
        let mut acc = Array1::<f64>::zeros(k);
        let mut gather = vec![0.0_f64; p];
        let mut prod = vec![0.0_f64; p];
        let mut w = vec![0.0_f64; res.max_di()];
        for i in 0..sys.rows.len() {
            res.row_into(i, x, &mut acc, &mut gather, &mut prod, &mut w);
        }
        for a in 0..k {
            out[a] -= acc[a];
        }
    } else {
        // Allocate scratch at max_d; per-row slice is `..di`.
        let mut local = Array1::<f64>::zeros(sys.d);
        let mut neg_contrib = Array1::<f64>::zeros(k);
        for i in 0..sys.rows.len() {
            neg_contrib.fill(0.0);
            schur_matvec_row_into(
                sys,
                htt_factors,
                x,
                backend,
                i,
                &mut local,
                &mut neg_contrib,
            );
            for a in 0..k {
                out[a] -= neg_contrib[a];
            }
        }
    }
}


/// Accumulate one row's reduced-Schur point-elimination contribution
/// `H_βt^(i) (H_tt^(i))⁻¹ H_tβ^(i) x` (length `K`) into `acc`.
///
/// `local` is caller-owned `≥ sys.d`-length scratch (reused across rows to keep
/// the hot loop allocation-free); only `..di` is touched. `acc` is **added to**,
/// never cleared, so the caller controls whether contributions sum into a chunk
/// partial (parallel path) or a per-row buffer (sequential path).
#[inline]
fn schur_matvec_row_into<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    x: &Array1<f64>,
    backend: &B,
    i: usize,
    local: &mut Array1<f64>,
    acc: &mut Array1<f64>,
) {
    let row = &sys.rows[i];
    let di = sys.row_dims[i];
    // H_tβ^(i) · x → local[..di], routed through sys.htbeta_matvec
    // when the dense block is absent.
    let mut local_i = local.slice_mut(ndarray::s![..di]).to_owned();
    local_i.fill(0.0);
    sys_htbeta_apply_row(sys, i, row, x.view(), &mut local_i);
    let solved = backend.solve_block_vector(htt_factors.factor(i), local_i.view());
    // H_βt^(i) · solved accumulates into acc (length k).  Routed through
    // sys.htbeta_matvec when needed.
    sys_htbeta_accumulate_transpose(sys, i, row, solved.view(), acc);
}


/// One per-term block factor for the block-Jacobi Schur preconditioner.
///
/// Carries either a dense Cholesky factor (for PD blocks ≤ 256 columns) or
/// the scalar inverses for that block's diagonal as a fallback.
#[derive(Clone)]
enum BlockFactor {
    /// Cholesky L stored column-major via faer. `range` identifies the
    /// columns in the full K-vector this block covers.
    Chol {
        factor: FaerLlt<f64>,
        range: Range<usize>,
    },
    /// Scalar fallback: per-element `1/s_aa` for each column in `range`.
    Scalar {
        inv: Array1<f64>,
        range: Range<usize>,
    },
}


impl std::fmt::Debug for BlockFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFactor::Chol { range, .. } => {
                write!(f, "BlockFactor::Chol {{ range: {:?} }}", range)
            }
            BlockFactor::Scalar { inv, range } => {
                write!(
                    f,
                    "BlockFactor::Scalar {{ inv.len: {}, range: {:?} }}",
                    inv.len(),
                    range
                )
            }
        }
    }
}


/// Block-Jacobi Schur preconditioner for BA's inexact reduced-system PCG.
///
/// When [`ArrowSchurSystem::block_offsets`] is populated (via
/// [`ArrowSchurSystem::set_block_offsets`]) and the largest block has ≤ 256
/// columns, builds one small dense Schur block per term, factors it with
/// Cholesky (faer LLT), and applies the preconditioner as per-block
/// triangular solves.  Non-PD blocks fall back to scalar diagonal inversion
/// for that block only.  When `block_offsets` is empty or the largest block
/// exceeds 256 columns the preconditioner reduces to pure scalar-diagonal
/// Jacobi (pre-#283 behaviour), so callers that have not called
/// `set_block_offsets` are unaffected.
///
/// The `block_offsets` plumbing is compatible with issue #287 (custom
/// `ParameterBlockSpec` families): those callers supply ranges derived from
/// their own block layout.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    blocks: Vec<BlockFactor>,
}


/// Maximum block size for which we attempt dense block-Jacobi factorization.
const BLOCK_JACOBI_MAX_BLOCK: usize = 256;


/// Positive-definiteness floor on a Schur-complement Jacobi diagonal entry.
/// A diagonal at or below this value (or non-finite) signals a non-PD reduced
/// system: the preconditioner cannot invert it, so the PCG solve fails loudly
/// and demands operator regularization rather than returning a garbage scale.
const JACOBI_DIAGONAL_PD_FLOOR: f64 = 1e-18;


impl JacobiPreconditioner {
    /// Build the block-Jacobi (or scalar fallback) preconditioner from the
    /// Arrow-Schur system without materializing the full dense Schur
    /// complement.
    ///
    /// When `sys.block_offsets` is non-empty and `max(block_size) ≤ 256`,
    /// each block gets a dense `b×b` Schur sub-matrix formed, factored, and
    /// stored.  Otherwise every column gets its own scalar entry.
    pub(crate) fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        resident: Option<&SaeResidentReducedSchur>,
    ) -> Result<Self, ArrowSchurError> {
        let use_block = !sys.block_offsets.is_empty()
            && sys
                .block_offsets
                .iter()
                .map(|r| r.end.saturating_sub(r.start))
                .max()
                .unwrap_or(0)
                <= BLOCK_JACOBI_MAX_BLOCK;
        if use_block {
            Self::build_block_jacobi(sys, htt_factors, ridge_beta, backend)
        } else if let Some(res) = resident {
            // #1017 — SAE residency scalar Jacobi. The generic scalar build
            // probes `H_tβ^(i) e_a` and re-solves `(H_tt^(i))⁻¹` once for EVERY
            // (row, β-column) pair: `O(n·K)` triangular solves and `O(n·K·p)`
            // operator-probe work per Newton step, with `K = K_atoms·p` in the
            // tens of thousands at LLM shapes. The reduced-Schur diagonal is the
            // same quotient the resident `(L_i, Y_i)` factors already carry, so
            // read the diagonal straight off them in one support-sparse pass —
            // no probe, no per-column solve.
            Self::build_scalar_jacobi_resident(sys, ridge_beta, res)
        } else {
            Self::build_scalar_jacobi(sys, htt_factors, ridge_beta, backend)
        }
    }

    /// Build scalar-diagonal Jacobi: one `BlockFactor::Scalar` of length 1
    /// per column.  Matches pre-#283 semantics.
    ///
    /// When `sys.htbeta_matvec` is set and per-row `htbeta` slabs are absent,
    /// each column is probed via the matvec (one call per column per row).
    fn build_scalar_jacobi<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        // Extract diagonal of H_ββ via penalty_diagonal_add (#296):
        // no Arc-clone; falls back to hbb_diag or hbb[[a,a]] inline.
        let mut diag = Array1::<f64>::zeros(k);
        {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            sys.penalty_diagonal_add(diag_slice);
        }
        for a in 0..k {
            diag[a] += ridge_beta;
        }
        // For each column a, extract H_tβ^(i) e_a via matvec probe when
        // dense slab is absent, then compute the scalar Schur diagonal.
        // Allocate scratch at max_d; per-row slice is ..di.
        let mut col = Array1::<f64>::zeros(sys.d);
        let mut e_a = Array1::<f64>::zeros(k);
        for (i, row) in sys.rows.iter().enumerate() {
            let di = sys.row_dims[i];
            let mut col_i = col.slice_mut(ndarray::s![..di]).to_owned();
            for a in 0..k {
                if sys.htbeta_matvec.is_some() || row.htbeta.dim() != (di, k) {
                    // Kronecker / matrix-free path: probe column a.
                    e_a.fill(0.0);
                    e_a[a] = 1.0;
                    col_i.fill(0.0);
                    sys_htbeta_apply_row(sys, i, row, e_a.view(), &mut col_i);
                } else {
                    for c in 0..di {
                        col_i[c] = row.htbeta[[c, a]];
                    }
                }
                let solved = backend.solve_block_vector(htt_factors.factor(i), col_i.view());
                let mut acc = 0.0;
                for c in 0..di {
                    acc += col_i[c] * solved[c];
                }
                diag[a] -= acc;
            }
        }
        let mut blocks = Vec::with_capacity(k);
        for a in 0..k {
            let v = diag[a];
            if !v.is_finite() || v <= JACOBI_DIAGONAL_PD_FLOOR {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "invalid Schur Jacobi diagonal at index {a}: {v}; \
                         operator regularization is required"
                    ),
                });
            }
            blocks.push(BlockFactor::Scalar {
                inv: Array1::from_elem(1, 1.0 / v),
                range: a..a + 1,
            });
        }
        Ok(Self { blocks })
    }

    /// Build scalar-diagonal Jacobi from the pre-staged SAE residency factors
    /// `(L_i, Y_i)` (#1017).
    ///
    /// The generic [`Self::build_scalar_jacobi`] forms each reduced-Schur
    /// diagonal entry `S_aa = H_ββ,aa + ρ − Σ_i (H_tβ^(i) e_a)ᵀ(H_tt^(i))⁻¹(H_tβ^(i) e_a)`
    /// by probing the cross-block operator with the unit vector `e_a` and
    /// re-solving `(H_tt^(i))⁻¹` for every `(row, column)` pair — `O(n·K)`
    /// triangular solves per Newton step. For the SAE Kronecker cross-block the
    /// `a`-th column lives on exactly one active support entry: `a = beta_base + j`
    /// for some `(beta_base, φ) ∈ a_phi[i]` and output channel `j ∈ 0..p`, with
    /// `H_tβ^(i) e_a = φ · L_i[:, j]`. The point-elimination quotient is then
    ///
    /// ```text
    /// (H_tβ^(i) e_a)ᵀ (H_tt^(i))⁻¹ (H_tβ^(i) e_a)
    ///     = φ² · L_i[:, j]ᵀ (H_tt^(i))⁻¹ L_i[:, j]
    ///     = φ² · (L_i[:, j] · Y_i[:, j]),          Y_i := (H_tt^(i))⁻¹ L_i.
    /// ```
    ///
    /// so the whole diagonal is accumulated in ONE support-sparse pass over the
    /// resident factors — no probe, no per-column solve, the staged `Y_i` reused
    /// from the matvec residency. The result is the SAME quotient the generic
    /// path computes (up to float reassociation of the row sum), so the PCG
    /// preconditioner — and therefore the criterion ranking — is unchanged.
    fn build_scalar_jacobi_resident(
        sys: &ArrowSchurSystem,
        ridge_beta: f64,
        resident: &SaeResidentReducedSchur,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        let p = resident.p;
        // Seed with diag(H_ββ) + ridge — same penalty source the generic path
        // reads, so the only difference is how the point-elimination term is
        // gathered.
        let mut diag = Array1::<f64>::zeros(k);
        {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            sys.penalty_diagonal_add(diag_slice);
        }
        for a in 0..k {
            diag[a] += ridge_beta;
        }
        // Per-row point-elimination diagonal: for each active support entry
        // `(beta_base, φ)` and channel `j`, subtract `φ² · L_i[:, j]·Y_i[:, j]`
        // into `diag[beta_base + j]`. `L_i`/`Y_i` are row-major `di × p`, so the
        // `j`-th column dot is `Σ_r L_i[r·p + j]·Y_i[r·p + j]`.
        for (row, rf) in resident.rows.iter().enumerate() {
            let di = rf.di;
            if di == 0 {
                continue;
            }
            let support = &resident.a_phi[row];
            if support.is_empty() {
                continue;
            }
            for &(beta_base, phi) in support {
                if phi == 0.0 {
                    continue;
                }
                let phi2 = phi * phi;
                for j in 0..p {
                    let mut col_dot = 0.0_f64;
                    for r in 0..di {
                        let idx = r * p + j;
                        col_dot += rf.l[idx] * rf.y[idx];
                    }
                    diag[beta_base + j] -= phi2 * col_dot;
                }
            }
        }
        let mut blocks = Vec::with_capacity(k);
        for a in 0..k {
            let v = diag[a];
            if !v.is_finite() || v <= JACOBI_DIAGONAL_PD_FLOOR {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "invalid SAE-resident Schur Jacobi diagonal at index {a}: {v}; \
                         operator regularization is required"
                    ),
                });
            }
            blocks.push(BlockFactor::Scalar {
                inv: Array1::from_elem(1, 1.0 / v),
                range: a..a + 1,
            });
        }
        Ok(Self { blocks })
    }

    /// Build term-block Jacobi: one dense `b×b` Schur block per term in
    /// `sys.block_offsets`.
    fn build_block_jacobi<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let block_offsets = &sys.block_offsets;

        // Initialise every b×b Schur sub-block from H_ββ + ridge·I via
        // penalty_block_add (#296): routes to penalty_op or falls back to
        // hbb / hbb_diag inline without Arc-clone per loop iteration. These are
        // the block-diagonal restrictions of the reduced Schur complement; the
        // per-row cross-block contributions are accumulated in the row sweep
        // below.
        let mut schur_blocks: Vec<Array2<f64>> = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let mut schur_block = Array2::<f64>::zeros((b, b));
            sys.penalty_block_add(
                BetaBlockId(block_idx),
                block_offsets.as_ref(),
                &mut schur_block,
            );
            for bi in 0..b {
                schur_block[[bi, bi]] += ridge_beta;
            }
            schur_blocks.push(schur_block);
        }

        // Subtract Schur contributions:
        // S_kk -= H_βt_k^(i) (H_tt^(i))^{-1} H_tβ_k^(i)
        //
        // Materialize each row's (d_i × K) cross-block ONCE and scatter its
        // contribution into every block-diagonal sub-block — mirroring the
        // row-outer structure of `build_dense_schur_direct`. The previous
        // block-outer form re-materialized every row for each β-block
        // (O(n_blocks · n · K) probes); for the matrix-free softmax cross-block
        // each materialize is itself O(K²), so that nesting made the
        // preconditioner build quadratically more expensive than the direct
        // dense Schur it preconditions. sys_htbeta_materialize_row handles the
        // Kronecker / htbeta_matvec path transparently.
        for (i, row) in sys.rows.iter().enumerate() {
            let di = sys.row_dims[i];
            let htbeta_full = sys_htbeta_materialize_row(sys, i, row);
            for (block_idx, range) in block_offsets.iter().enumerate() {
                let b = range.end - range.start;
                let mut solved_cols = Array2::<f64>::zeros((di, b));
                for bj in 0..b {
                    let gj = range.start + bj;
                    let rhs = htbeta_full.column(gj).to_owned();
                    let solved = backend.solve_block_vector(htt_factors.factor(i), rhs.view());
                    for c in 0..di {
                        solved_cols[[c, bj]] = solved[c];
                    }
                }
                let schur_block = &mut schur_blocks[block_idx];
                for bi in 0..b {
                    let gi = range.start + bi;
                    for bj in 0..b {
                        let mut acc = 0.0;
                        for c in 0..di {
                            acc += htbeta_full[[c, gi]] * solved_cols[[c, bj]];
                        }
                        schur_block[[bi, bj]] -= acc;
                    }
                }
            }
        }

        // Factor each accumulated block: LLT, with scalar-diagonal fallback for
        // a block that comes out non-PD at this ridge.
        let mut blocks = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let schur_block = &schur_blocks[block_idx];
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(schur_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                blocks.push(BlockFactor::Chol {
                    factor: llt,
                    range: range.clone(),
                });
            } else {
                // Non-PD block: fall back to scalar diagonal for this block.
                let mut inv = Array1::<f64>::zeros(b);
                for bi in 0..b {
                    let v = schur_block[[bi, bi]];
                    if !v.is_finite() || v <= JACOBI_DIAGONAL_PD_FLOOR {
                        return Err(ArrowSchurError::PcgFailed {
                            reason: format!(
                                "block Jacobi scalar fallback: non-PD diagonal at \
                                 global index {}: {v}; regularization required",
                                range.start + bi
                            ),
                        });
                    }
                    inv[bi] = 1.0 / v;
                }
                blocks.push(BlockFactor::Scalar {
                    inv,
                    range: range.clone(),
                });
            }
        }
        Ok(Self { blocks })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for block in &self.blocks {
            match block {
                BlockFactor::Scalar { inv, range } => {
                    for (local, gi) in range.clone().enumerate() {
                        out[gi] = inv[local] * r[gi];
                    }
                }
                BlockFactor::Chol { factor, range } => {
                    let b = range.end - range.start;
                    let mut rhs = Array1::<f64>::zeros(b);
                    for (local, gi) in range.clone().enumerate() {
                        rhs[local] = r[gi];
                    }
                    use faer::linalg::solvers::Solve;
                    let stride = rhs.strides()[0];
                    let len = rhs.len();
                    // SAFETY: rhs is a uniquely-borrowed contiguous Array1
                    // with positive stride (standard layout).
                    let rhs_mat =
                        unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
                    let solved = factor.solve(rhs_mat);
                    for (local, gi) in range.clone().enumerate() {
                        out[gi] = solved[(local, 0)];
                    }
                }
            }
        }
        out
    }
}


// ---------------------------------------------------------------------------
// Preconditioner ladder: SchurPreconditionerKind, ClusterJacobi,
// AdditiveSchwarz  (issue #299)
// ---------------------------------------------------------------------------

/// Which Schur preconditioner to use in the inexact-PCG path.
///
/// Ladder ordered by cost / effectiveness:
/// - `Diagonal`: scalar Jacobi (pre-#283 behaviour).
/// - `BetaBlockJacobi`: block-Jacobi per `block_offsets` term (#287).
/// - `ClusterJacobi`: one dense block per beta-graph connected component.
/// - `AdditiveSchwarz { overlap }`: component + `overlap`-hop expansion,
///   overlapping columns averaged by partition-of-unity weights.
///
/// ```text
/// Future variants (not yet wired, see #299):
///   DiagAssembledSchwarz { overlap: usize },
///   SparseIncompleteCholesky,
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurPreconditionerKind {
    Diagonal,
    BetaBlockJacobi,
    ClusterJacobi,
    AdditiveSchwarz { overlap: usize },
}


/// Escalate beyond BetaBlockJacobi only when K exceeds this value and PCG
/// exhausted `max_iterations`.
const PRECOND_ESCALATE_K_THRESHOLD: usize = 100;


/// Cholesky or scalar factor for one cluster of the beta-coefficient graph.
#[derive(Clone)]
enum ClusterFactor {
    Chol {
        cols: Vec<usize>,
        factor: FaerLlt<f64>,
    },
    Scalar {
        cols: Vec<usize>,
        inv: Vec<f64>,
    },
}


impl std::fmt::Debug for ClusterFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterFactor::Chol { cols, .. } => {
                write!(f, "ClusterFactor::Chol {{ cols.len: {} }}", cols.len())
            }
            ClusterFactor::Scalar { cols, inv } => write!(
                f,
                "ClusterFactor::Scalar {{ cols.len: {}, inv.len: {} }}",
                cols.len(),
                inv.len()
            ),
        }
    }
}


/// Maximum columns per cluster before scalar fallback.
const CLUSTER_JACOBI_MAX_CLUSTER: usize = 512;


/// Dense Schur block per connected component of the beta-coupling graph.
///
/// Nodes = beta blocks (`block_offsets`); edges = rows where two blocks
/// co-occur with nonzero `H_t_beta` entries. One Cholesky factor per
/// connected component; applied as a triangular solve.
#[derive(Debug, Clone)]
pub struct ClusterJacobiPreconditioner {
    clusters: Vec<ClusterFactor>,
}


impl ClusterJacobiPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            return Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &[cols]);
        }
        let graph = BetaCouplingGraph::build(
            &sys.block_offsets,
            &sys.rows
                .iter()
                .map(|r| r.htbeta.clone())
                .collect::<Vec<_>>(),
        );
        let col_groups: Vec<Vec<usize>> = graph
            .component_partition()
            .iter()
            .map(|comp_blocks| {
                let mut cols: Vec<usize> = comp_blocks
                    .iter()
                    .flat_map(|&b| sys.block_offsets[b].clone())
                    .collect();
                cols.sort_unstable();
                cols
            })
            .collect();
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    fn build_from_column_groups<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        col_groups: &[Vec<usize>],
    ) -> Result<Self, ArrowSchurError> {
        let d = sys.d;
        let mut clusters = Vec::with_capacity(col_groups.len());
        for cols in col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            if b > CLUSTER_JACOBI_MAX_CLUSTER {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
                continue;
            }
            let mut s_block = Array2::<f64>::zeros((b, b));
            // Initialise from H_ββ via penalty_subblock_add (#296): routes
            // through penalty_op or falls back to hbb / hbb_diag inline.
            sys.penalty_subblock_add(cols, &mut s_block);
            for bi in 0..b {
                s_block[[bi, bi]] += ridge_beta;
            }
            let mut col_vec = Array1::<f64>::zeros(d);
            let mut solved_cols = Array2::<f64>::zeros((d, b));
            for (row_idx, row) in sys.rows.iter().enumerate() {
                for bj in 0..b {
                    let gj = cols[bj];
                    for c in 0..d {
                        col_vec[c] = row.htbeta[[c, gj]];
                    }
                    let solved =
                        backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
                    for c in 0..d {
                        solved_cols[[c, bj]] = solved[c];
                    }
                }
                for bi in 0..b {
                    let gi = cols[bi];
                    for bj in 0..b {
                        let mut acc = 0.0;
                        for c in 0..d {
                            acc += row.htbeta[[c, gi]] * solved_cols[[c, bj]];
                        }
                        s_block[[bi, bj]] -= acc;
                    }
                }
            }
            symmetrize_upper_from_lower(&mut s_block);
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(&s_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                clusters.push(ClusterFactor::Chol {
                    cols: cols.clone(),
                    factor: llt,
                });
            } else {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
            }
        }
        Ok(Self { clusters })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster(cluster, r, &mut out, &ClusterApplyMode::Overwrite);
        }
        out
    }
}


/// Additive Schwarz: base components expanded by `overlap` graph-hops;
/// overlapping columns averaged by partition-of-unity weights.
#[derive(Debug, Clone)]
pub struct AdditiveSchwarzPreconditioner {
    clusters: Vec<ClusterFactor>,
    weights: Vec<f64>,
}


impl AdditiveSchwarzPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        overlap: usize,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            let inner = ClusterJacobiPreconditioner::build_from_column_groups(
                sys,
                htt_factors,
                ridge_beta,
                backend,
                &[cols],
            )?;
            return Ok(Self {
                clusters: inner.clusters,
                weights: vec![1.0f64; sys.k],
            });
        }
        let graph = BetaCouplingGraph::build(
            &sys.block_offsets,
            &sys.rows
                .iter()
                .map(|r| r.htbeta.clone())
                .collect::<Vec<_>>(),
        );
        let col_groups: Vec<Vec<usize>> = graph
            .component_partition()
            .iter()
            .map(|seed| {
                let mut current = seed.clone();
                for _ in 0..overlap {
                    current = graph.expand_one_hop(&current);
                }
                let mut cols: Vec<usize> = current
                    .iter()
                    .flat_map(|&b| sys.block_offsets[b].clone())
                    .collect();
                cols.sort_unstable();
                cols.dedup();
                cols
            })
            .collect();
        let mut counts = vec![0u32; sys.k];
        for cols in &col_groups {
            for &gi in cols {
                counts[gi] += 1;
            }
        }
        let weights: Vec<f64> = counts
            .iter()
            .map(|&c| if c == 0 { 1.0 } else { 1.0 / c as f64 })
            .collect();
        let inner = ClusterJacobiPreconditioner::build_from_column_groups(
            sys,
            htt_factors,
            ridge_beta,
            backend,
            &col_groups,
        )?;
        Ok(Self {
            clusters: inner.clusters,
            weights,
        })
    }

    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster(
                cluster,
                r,
                &mut out,
                &ClusterApplyMode::Accumulate {
                    weights: &self.weights,
                },
            );
        }
        out
    }
}


/// How a cluster factor's contribution is written into the output vector.
///
/// `Overwrite` assigns `out[gi] = value` (non-overlapping clusters, each global
/// column touched by exactly one cluster). `Accumulate` adds the partition-of-unity
/// weighted contribution `out[gi] += weights[gi] * value` (overlapping Schwarz
/// clusters, where a column may belong to several clusters).
enum ClusterApplyMode<'w> {
    Overwrite,
    Accumulate { weights: &'w [f64] },
}


impl ClusterApplyMode<'_> {
    #[inline]
    fn write(&self, out: &mut Array1<f64>, gi: usize, value: f64) {
        match self {
            ClusterApplyMode::Overwrite => out[gi] = value,
            ClusterApplyMode::Accumulate { weights } => out[gi] += weights[gi] * value,
        }
    }
}


/// Apply a single cluster factor to the residual `r`, writing into `out`
/// according to `mode` (overwrite for non-overlapping clusters, weighted
/// accumulate for overlapping Schwarz clusters).
fn apply_cluster(
    cluster: &ClusterFactor,
    r: &Array1<f64>,
    out: &mut Array1<f64>,
    mode: &ClusterApplyMode<'_>,
) {
    match cluster {
        ClusterFactor::Scalar { cols, inv } => {
            for (local, &gi) in cols.iter().enumerate() {
                mode.write(out, gi, inv[local] * r[gi]);
            }
        }
        ClusterFactor::Chol { cols, factor } => {
            let b = cols.len();
            let mut rhs = Array1::<f64>::zeros(b);
            for (local, &gi) in cols.iter().enumerate() {
                rhs[local] = r[gi];
            }
            use faer::linalg::solvers::Solve;
            let stride = rhs.strides()[0];
            let len = rhs.len();
            // SAFETY: rhs is uniquely-borrowed contiguous Array1 with positive stride.
            let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
            let solved = factor.solve(rhs_mat);
            for (local, &gi) in cols.iter().enumerate() {
                mode.write(out, gi, solved[(local, 0)]);
            }
        }
    }
}


/// Build scalar diagonal inverses for a set of global column indices.
///
/// Used when a cluster is non-PD or exceeds `CLUSTER_JACOBI_MAX_CLUSTER`.
fn build_schur_scalar_inv<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    cols: &[usize],
) -> Result<Vec<f64>, ArrowSchurError> {
    let d = sys.d;
    let mut result = Vec::with_capacity(cols.len());
    let mut col_vec = Array1::<f64>::zeros(d);
    // Extract the penalty diagonal for all K columns once, then index per-column.
    let mut full_diag = Array1::<f64>::zeros(sys.k);
    {
        let fd_slice = full_diag.as_slice_mut().expect("full_diag contiguous");
        sys.penalty_diagonal_add(fd_slice);
    }
    for &gi in cols {
        let mut s = full_diag[gi] + ridge_beta;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            for c in 0..d {
                col_vec[c] = row.htbeta[[c, gi]];
            }
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
            let mut acc = 0.0;
            for c in 0..d {
                acc += col_vec[c] * solved[c];
            }
            s -= acc;
        }
        if !s.is_finite() || s <= JACOBI_DIAGONAL_PD_FLOOR {
            return Err(ArrowSchurError::PcgFailed {
                reason: format!(
                    "cluster Schur scalar fallback: non-PD diagonal at index {gi}: {s}"
                ),
            });
        }
        result.push(1.0 / s);
    }
    Ok(result)
}


/// Inexact PCG with automatic preconditioner-ladder escalation.
///
/// Starts with `JacobiPreconditioner` (Diagonal or BetaBlockJacobi).
/// If PCG hits `MaxIter` and `k > PRECOND_ESCALATE_K_THRESHOLD`,
/// escalates to `ClusterJacobi`; if still `MaxIter`, escalates to
/// `AdditiveSchwarz { overlap: 1 }`.
fn steihaug_pcg_auto<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    rhs: &Array1<f64>,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    // #1017 CPU residency: stage the per-row reduced-Schur factors `(L_i, Y_i)`
    // (NOT the dense `p×p` block — `di ≪ p`, so the factored form is `O(n·di·p)`
    // memory and `2·di·p` flops/row) once, up
    // front, when the SAE structure is installed and the matvec runs on host
    // (CPU). The GPU matvec carries its own residency, so skip when it is engaged.
    // The same staged operator is reused across the whole preconditioner ladder
    // (Jacobi → ClusterJacobi → AdditiveSchwarz) — built once, not per tier.
    let resident = if gpu_matvec.is_none() {
        SaeResidentReducedSchur::build(sys, htt_factors, backend)
    } else {
        None
    };
    let jacobi = JacobiPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident.as_ref(),
    )?;
    let (x0, diag0) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        ridge_beta,
        rhs,
        |r| jacobi.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
        resident.as_ref(),
    )?;
    if sys.k <= PRECOND_ESCALATE_K_THRESHOLD || diag0.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x0, diag0));
    }
    let cluster =
        ClusterJacobiPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend)?;
    let (x1, diag1) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        ridge_beta,
        rhs,
        |r| cluster.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
        resident.as_ref(),
    )?;
    if diag1.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x1, diag1));
    }
    let schwarz =
        AdditiveSchwarzPreconditioner::from_arrow_schur(sys, htt_factors, ridge_beta, backend, 1)?;
    let (x2, diag2) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        ridge_beta,
        rhs,
        |r| schwarz.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
        resident.as_ref(),
    )?;
    // All three preconditioner tiers (Jacobi -> ClusterJacobi ->
    // AdditiveSchwarz) exhausted their iteration budget without driving the
    // residual below tolerance. Returning the truncated AdditiveSchwarz iterate
    // as `Ok` would feed an arbitrarily-large-residual step into the Newton
    // driver, where the PCG diagnostics are discarded. Surface a recoverable
    // failure instead so `solve_with_lm_escalation_inner` escalates the
    // proximal ridge: better conditioning is precisely what a stalled PCG on
    // an ill-conditioned reduced system needs.
    if diag2.stopping_reason == PcgStopReason::MaxIter {
        return Err(ArrowSchurError::PcgFailed {
            reason: format!(
                "Schur PCG exhausted all preconditioner tiers (Jacobi, ClusterJacobi, \
                 AdditiveSchwarz) at MaxIter; final relative residual = {:e}",
                diag2.final_relative_residual
            ),
        });
    }
    Ok((x2, diag2))
}


/// Run Steihaug-CG with a generic preconditioner closure.
/// Routes matvec through GPU when `gpu_matvec` is set.
fn run_pcg_with_preconditioner<ApplyPrec, B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    rhs: &Array1<f64>,
    apply_prec: ApplyPrec,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    metric_weights: Option<&MetricWeights>,
    resident: Option<&SaeResidentReducedSchur>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError>
where
    ApplyPrec: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let max_iters = pcg.max_iterations.min(trust.max_iterations);
    let tol = pcg
        .relative_tolerance
        .max(trust.steihaug_relative_tolerance);
    if let Some(gpu_mv) = gpu_matvec {
        let gpu_mv = Arc::clone(gpu_mv);
        steihaug_cg(
            rhs,
            move |p, out| gpu_mv(p, out),
            apply_prec,
            max_iters,
            tol,
            trust.radius,
            metric_weights,
        )
    } else {
        steihaug_cg(
            rhs,
            |p, out| schur_matvec(sys, htt_factors, ridge_beta, p, out, backend, resident),
            apply_prec,
            max_iters,
            tol,
            trust.radius,
            metric_weights,
        )
    }
}


#[derive(Debug, Clone, Copy)]
struct IdentityPreconditioner;


impl IdentityPreconditioner {
    fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        r.clone()
    }
}


fn steihaug_dense_system(
    schur: &Array2<f64>,
    rhs: &Array1<f64>,
    preconditioner: &IdentityPreconditioner,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError> {
    steihaug_cg(
        rhs,
        |p, out| dense_matvec(schur, p, out),
        |r| preconditioner.apply(r),
        pcg.max_iterations,
        pcg.relative_tolerance,
        trust.radius,
        metric_weights,
    )
}


fn steihaug_cg<MatVec, ApplyPrec>(
    rhs: &Array1<f64>,
    mut matvec: MatVec,
    mut apply_preconditioner: ApplyPrec,
    max_iterations: usize,
    relative_tolerance: f64,
    trust_radius: f64,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, PcgDiagnostics), ArrowSchurError>
where
    MatVec: FnMut(&Array1<f64>, &mut Array1<f64>),
    ApplyPrec: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let n = rhs.len();
    if let Some(weights) = metric_weights {
        assert_eq!(
            weights.len(),
            n,
            "Steihaug-CG metric weight length must match solve dimension"
        );
    }
    let radius = if trust_radius.is_finite() && trust_radius > 0.0 {
        trust_radius
    } else {
        f64::INFINITY
    };
    let rhs_norm = metric_norm(rhs.view(), metric_weights);
    if rhs_norm == 0.0 {
        return Ok((Array1::<f64>::zeros(n), PcgDiagnostics::default()));
    }
    let tol = (relative_tolerance.max(0.0) * rhs_norm).max(PCG_ABSOLUTE_TOLERANCE_FLOOR);
    let mut x = Array1::<f64>::zeros(n);
    let mut r = rhs.clone();
    let mut z = apply_preconditioner(&r);
    let mut diag = PcgDiagnostics {
        precond_apply_calls: 1,
        ..PcgDiagnostics::default()
    };
    let mut p = z.clone();
    let mut rz = metric_dot(&r, &z, metric_weights);
    if rz <= 0.0 || !rz.is_finite() {
        if radius.is_finite() {
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &r, radius, metric_weights), diag));
        }
        return Err(ArrowSchurError::PcgFailed {
            reason: "non-positive preconditioned residual in Schur PCG".to_string(),
        });
    }
    if metric_norm(r.view(), metric_weights) <= tol {
        diag.final_relative_residual = 0.0;
        diag.stopping_reason = PcgStopReason::Converged;
        return Ok((x, diag));
    }
    let mut ap = Array1::<f64>::zeros(n);
    // Reused candidate scratch — avoid per-iteration clone of x.
    let mut candidate = Array1::<f64>::zeros(n);
    for _ in 0..max_iterations {
        matvec(&p, &mut ap);
        diag.matvec_calls += 1;
        diag.iterations += 1;
        let pap = metric_dot(&p, &ap, metric_weights);
        if pap <= 0.0 || !pap.is_finite() {
            if radius.is_finite() {
                diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
                diag.stopping_reason = PcgStopReason::TrustRegion;
                return Ok((step_to_trust_boundary(&x, &p, radius, metric_weights), diag));
            }
            return Err(ArrowSchurError::PcgFailed {
                reason: "negative curvature in unbounded Schur PCG".to_string(),
            });
        }
        let alpha = rz / pap;
        for i in 0..n {
            candidate[i] = x[i] + alpha * p[i];
        }
        if radius.is_finite() && metric_norm(candidate.view(), metric_weights) >= radius {
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &p, radius, metric_weights), diag));
        }
        x.assign(&candidate);
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }
        if metric_norm(r.view(), metric_weights) <= tol {
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::Converged;
            return Ok((x, diag));
        }
        z = apply_preconditioner(&r);
        diag.precond_apply_calls += 1;
        let rz_next = metric_dot(&r, &z, metric_weights);
        if rz_next <= 0.0 || !rz_next.is_finite() {
            return Err(ArrowSchurError::PcgFailed {
                reason: "non-positive or non-finite PCG residual".to_string(),
            });
        }
        let beta = rz_next / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_next;
    }
    diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
    diag.stopping_reason = PcgStopReason::MaxIter;
    Ok((x, diag))
}


fn step_to_trust_boundary(
    x: &Array1<f64>,
    p: &Array1<f64>,
    radius: f64,
    metric_weights: Option<&MetricWeights>,
) -> Array1<f64> {
    let pp = metric_dot(p, p, metric_weights);
    if pp == 0.0 {
        return x.clone();
    }
    let xp = metric_dot(x, p, metric_weights);
    let xx = metric_dot(x, x, metric_weights);
    let disc = (xp * xp + pp * (radius * radius - xx)).max(0.0);
    let tau = (-xp + disc.sqrt()) / pp;
    let mut out = x.clone();
    for i in 0..out.len() {
        out[i] += tau * p[i];
    }
    out
}


fn dense_matvec(a: &Array2<f64>, x: &Array1<f64>, out: &mut Array1<f64>) {
    let n = a.nrows();
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[[i, j]] * x[j];
        }
        out[i] = acc;
    }
}


fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}


fn metric_dot(a: &Array1<f64>, b: &Array1<f64>, metric_weights: Option<&MetricWeights>) -> f64 {
    assert_eq!(a.len(), b.len());
    match metric_weights {
        Some(weights) => {
            assert_eq!(weights.len(), a.len());
            let mut acc = 0.0;
            for i in 0..a.len() {
                acc += weights[i] * a[i] * b[i];
            }
            acc
        }
        None => dot(a, b),
    }
}


fn metric_norm(v: ArrayView1<'_, f64>, metric_weights: Option<&MetricWeights>) -> f64 {
    let mut acc = 0.0;
    match metric_weights {
        Some(weights) => {
            assert_eq!(weights.len(), v.len());
            for i in 0..v.len() {
                acc += weights[i] * v[i] * v[i];
            }
        }
        None => {
            for x in v.iter() {
                acc += x * x;
            }
        }
    }
    acc.sqrt()
}


fn symmetrize_upper_from_lower(a: &mut Array2<f64>) {
    let n = a.nrows().min(a.ncols());
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
}


/// Errors raised by [`ArrowSchurSystem::solve`].
#[derive(Debug, Clone)]
pub enum ArrowSchurError {
    /// A per-row `H_tt^(i)` block was not positive-definite at the
    /// supplied ridge. Indicates an under-regularized latent block —
    /// typically a gauge-free fit without an identifiability penalty.
    PerRowFactorFailed { row: usize, reason: String },
    /// A per-row `H_tt^(i)` block factored, but the Cholesky factor failed
    /// the safe-inversion guard for the Schur reduction. This can be either
    /// an excessive diagonal-ratio condition-number estimate or a numerically
    /// tiny pivot relative to the row block scale. Cholesky technically
    /// succeeded, but the inverse used in
    /// `S = H_ββ − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)` is contaminated
    /// by spectral terms on the order of `κ_i`; functionally
    /// equivalent to a PSD-fail for Schur stability. The LM outer
    /// wrapper escalates `ridge_t` identically to `PerRowFactorFailed`.
    PerRowFactorIllConditioned { row: usize, kappa_estimate: f64 },
    /// The Schur complement was not positive-definite. Indicates a
    /// near-collinear decoder or a degenerate weighting; the LM outer
    /// wrapper should escalate `ridge_beta` and retry.
    SchurFactorFailed { reason: String },
    /// The BA inexact-step PCG solve failed before producing a usable
    /// Steihaug trust-region step.
    PcgFailed { reason: String },
    /// Adaptive proximal damping could not produce an Armijo-accepted
    /// nonlinear step.
    AdaptiveCorrectionFailed { reason: String },
}


impl std::fmt::Display for ArrowSchurError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrowSchurError::PerRowFactorFailed { row, reason } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky failed: {reason}"
            ),
            ArrowSchurError::PerRowFactorIllConditioned {
                row,
                kappa_estimate,
            } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky succeeded but failed \
                 the safe-inversion guard (kappa_estimate={kappa_estimate:e}); \
                 Schur reduction would be numerically contaminated"
            ),
            ArrowSchurError::SchurFactorFailed { reason } => {
                write!(f, "arrow-Schur: Schur complement Cholesky failed: {reason}")
            }
            ArrowSchurError::PcgFailed { reason } => {
                write!(f, "arrow-Schur: Schur PCG failed: {reason}")
            }
            ArrowSchurError::AdaptiveCorrectionFailed { reason } => {
                write!(
                    f,
                    "arrow-Schur: adaptive proximal correction failed: {reason}"
                )
            }
        }
    }
}


impl std::error::Error for ArrowSchurError {}


// ---------------------------------------------------------------------------
// Cholesky helpers (kept local to avoid a new public-API dependency on the
// linalg crate. The systems here are tiny per-row (d × d, d ∈ {1..16}) and
// modest at the Schur level (K × K, K ∈ {basis size}). For production SAE
// scales the Schur factor should switch to faer; this module's `cholesky_lower`
// is the obvious replacement site.)
// ---------------------------------------------------------------------------

fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(format!("cholesky_lower: non-square {}×{}", n, a.ncols()));
    }
    if let Some((idx, _)) = a.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "cholesky_lower: non-finite entry at linear index {idx}"
        ));
    }

    let mut maybe_device = a.clone();
    if crate::gpu::try_cholesky_lower_inplace(&mut maybe_device).is_some() {
        return Ok(maybe_device);
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(format!(
                        "non-PD pivot {sum} at index {i} (matrix is not positive definite)"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// `SparseBlockKroneckerPenaltyOp` must reproduce the dense
    /// `KroneckerPenaltyOp { factor_a: G, factor_b: I_p }` on every interface
    /// (matvec, gradient, diagonal, to_dense) when the sparse block set covers
    /// the same `(atom, atom')` couplings — this is the equivalence that makes
    /// the sparse op a drop-in replacement for the dense data Gram.
    #[test]
    fn sparse_block_kronecker_matches_dense_kronecker() {
        // Two atoms: atom 0 has m_0 = 2 basis cols (μ offset 0), atom 1 has
        // m_1 = 3 (μ offset 2). p = 2 output channels ⇒ dim_a = 5, k = 10.
        let p = 2usize;
        let dim_a = 5usize;
        let k = dim_a * p;
        // Dense G (5×5) with non-zero (0,0), (0,1), (1,0), (1,1) atom blocks.
        let g_dense = array![
            [3.0_f64, 0.5, 0.2, -0.1, 0.0],
            [0.5, 4.0, 0.0, 0.3, 0.1],
            [0.2, 0.0, 2.0, 0.4, -0.2],
            [-0.1, 0.3, 0.4, 5.0, 0.6],
            [0.0, 0.1, -0.2, 0.6, 1.5],
        ];
        let dense = KroneckerPenaltyOp {
            factor_a: g_dense.clone(),
            factor_b: Array2::<f64>::eye(p),
            global_offset: 0,
            k,
        };
        // Sparse: atom 0 block = G[0..2, 0..2], cross blocks G[0..2,2..5] and
        // its transpose, atom 1 block = G[2..5, 2..5].
        let block_00 = g_dense.slice(ndarray::s![0..2, 0..2]).to_owned();
        let block_01 = g_dense.slice(ndarray::s![0..2, 2..5]).to_owned();
        let block_10 = g_dense.slice(ndarray::s![2..5, 0..2]).to_owned();
        let block_11 = g_dense.slice(ndarray::s![2..5, 2..5]).to_owned();
        let sparse = SparseBlockKroneckerPenaltyOp {
            p,
            dim_a,
            k,
            blocks: vec![
                SparseGBlock {
                    row_off: 0,
                    col_off: 0,
                    data: block_00,
                },
                SparseGBlock {
                    row_off: 0,
                    col_off: 2,
                    data: block_01,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 0,
                    data: block_10,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 2,
                    data: block_11,
                },
            ],
        };

        // to_dense parity.
        let d_dense = dense.to_dense();
        let d_sparse = sparse.to_dense();
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (d_dense[[i, j]] - d_sparse[[i, j]]).abs() < 1e-12,
                    "to_dense mismatch at ({i},{j}): {} vs {}",
                    d_dense[[i, j]],
                    d_sparse[[i, j]]
                );
            }
        }

        // matvec / gradient parity on an arbitrary vector.
        let x: Vec<f64> = (0..k).map(|i| 0.1 * (i as f64) - 0.3).collect();
        let mut y_dense = vec![0.0_f64; k];
        let mut y_sparse = vec![0.0_f64; k];
        dense.matvec(&x, &mut y_dense);
        sparse.matvec(&x, &mut y_sparse);
        for i in 0..k {
            assert!(
                (y_dense[i] - y_sparse[i]).abs() < 1e-12,
                "matvec mismatch at {i}: {} vs {}",
                y_dense[i],
                y_sparse[i]
            );
        }

        // diagonal parity.
        let mut diag_dense = vec![0.0_f64; k];
        let mut diag_sparse = vec![0.0_f64; k];
        dense.diagonal(&mut diag_dense);
        sparse.diagonal(&mut diag_sparse);
        for i in 0..k {
            assert!(
                (diag_dense[i] - diag_sparse[i]).abs() < 1e-12,
                "diagonal mismatch at {i}: {} vs {}",
                diag_dense[i],
                diag_sparse[i]
            );
        }

        // block parity: probe the per-atom β block ranges.
        let offsets = [0..(2 * p), (2 * p)..k];
        for id in 0..offsets.len() {
            let b = offsets[id].end - offsets[id].start;
            let mut blk_dense = Array2::<f64>::zeros((b, b));
            let mut blk_sparse = Array2::<f64>::zeros((b, b));
            dense.block(BetaBlockId(id), &offsets, &mut blk_dense);
            sparse.block(BetaBlockId(id), &offsets, &mut blk_sparse);
            for i in 0..b {
                for j in 0..b {
                    assert!(
                        (blk_dense[[i, j]] - blk_sparse[[i, j]]).abs() < 1e-12,
                        "block {id} mismatch at ({i},{j})"
                    );
                }
            }
        }
    }

    /// Hand-built dense reference for the frame-factored Gram
    /// `H[(i,li,a),(j,lj,b)] = g_ij[li,lj]·(U_iᵀU_j)[a,b]`, with the variable
    /// per-atom width `r_k`.
    fn factored_reference_dense(
        ranks: &[usize],
        basis_sizes: &[usize],
        blocks: &[FactoredFrameGBlock],
    ) -> Array2<f64> {
        let n_atoms = ranks.len();
        let mut offsets = vec![0usize; n_atoms + 1];
        for k in 0..n_atoms {
            offsets[k + 1] = offsets[k] + basis_sizes[k] * ranks[k];
        }
        let dim = offsets[n_atoms];
        let mut h = Array2::<f64>::zeros((dim, dim));
        for blk in blocks {
            let (r_i, r_j) = (ranks[blk.atom_i], ranks[blk.atom_j]);
            let (off_i, off_j) = (offsets[blk.atom_i], offsets[blk.atom_j]);
            let (m_i, m_j) = blk.g.dim();
            for li in 0..m_i {
                for lj in 0..m_j {
                    for a in 0..r_i {
                        for b in 0..r_j {
                            h[[off_i + li * r_i + a, off_j + lj * r_j + b]] +=
                                blk.g[[li, lj]] * blk.w[[a, b]];
                        }
                    }
                }
            }
        }
        h
    }

    /// `FactoredFrameKroneckerOp` must equal its dense `g ⊗ (UᵀU)` reference on
    /// every interface, with VARIABLE per-atom rank (`r_0 = 2`, `r_1 = 3`) and a
    /// genuine cross-atom output factor `U_0ᵀU_1 ≠ 0`.
    #[test]
    fn factored_frame_kronecker_matches_dense_reference() {
        // Atom 0: M_0 = 2, r_0 = 2. Atom 1: M_1 = 3, r_1 = 3. dim = 4 + 9 = 13.
        let ranks = vec![2usize, 3];
        let basis_sizes = vec![2usize, 3];
        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4, -0.2], [0.4, 5.0, 0.6], [-0.2, 0.6, 1.5]];
        let g01 = array![[0.2_f64, -0.1, 0.0], [0.3, 0.1, -0.2]];
        let g10 = g01.t().to_owned();
        // Within-atom frame factors are identity (orthonormal U); the cross
        // factor U_0ᵀU_1 (2×3) is a generic dense principal-angle matrix.
        let w00 = Array2::<f64>::eye(2);
        let w11 = Array2::<f64>::eye(3);
        let w01 = array![[0.8_f64, 0.1, -0.05], [0.0, 0.7, 0.2]];
        let w10 = w01.t().to_owned();
        let blocks = vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00.clone(),
                w: w00.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11.clone(),
                w: w11.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01.clone(),
                w: w01.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10.clone(),
                w: w10.clone(),
            },
        ];
        let op = FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), blocks.clone())
            .expect("op");
        assert_eq!(op.dim(), 13);
        let reference = factored_reference_dense(&ranks, &basis_sizes, &blocks);

        // to_dense.
        let dense = op.to_dense();
        for i in 0..13 {
            for j in 0..13 {
                assert!(
                    (dense[[i, j]] - reference[[i, j]]).abs() < 1e-12,
                    "to_dense mismatch at ({i},{j}): {} vs {}",
                    dense[[i, j]],
                    reference[[i, j]]
                );
            }
        }
        // matvec == reference·x.
        let x: Vec<f64> = (0..13).map(|i| 0.13 * (i as f64) - 0.4).collect();
        let mut y = vec![0.0_f64; 13];
        op.matvec(&x, &mut y);
        for i in 0..13 {
            let mut expect = 0.0;
            for j in 0..13 {
                expect += reference[[i, j]] * x[j];
            }
            assert!(
                (y[i] - expect).abs() < 1e-10,
                "matvec mismatch at {i}: {} vs {expect}",
                y[i]
            );
        }
        // diagonal.
        let mut diag = vec![0.0_f64; 13];
        op.diagonal(&mut diag);
        for i in 0..13 {
            assert!(
                (diag[i] - reference[[i, i]]).abs() < 1e-12,
                "diagonal mismatch at {i}"
            );
        }
        // block over each atom's β range.
        let offsets_ranges = [0..4usize, 4..13usize];
        for id in 0..2 {
            let b = offsets_ranges[id].end - offsets_ranges[id].start;
            let mut blk = Array2::<f64>::zeros((b, b));
            op.block(BetaBlockId(id), &offsets_ranges, &mut blk);
            for bi in 0..b {
                for bj in 0..b {
                    let gi = offsets_ranges[id].start + bi;
                    let gj = offsets_ranges[id].start + bj;
                    assert!(
                        (blk[[bi, bj]] - reference[[gi, gj]]).abs() < 1e-12,
                        "block {id} mismatch at ({bi},{bj})"
                    );
                }
            }
        }
    }

    /// Strict-generalization pin: with every `r_k = p` and `U_k = I_p` (so all
    /// frame factors are identity), `FactoredFrameKroneckerOp` reproduces
    /// `SparseBlockKroneckerPenaltyOp` (the `G ⊗ I_p` data Gram) bit-for-bit on
    /// matvec — i.e. the full-`B` border is the `r = p` special case of the
    /// factored op, not a separate path.
    #[test]
    fn factored_frame_kronecker_reduces_to_sparse_block_at_full_rank() {
        let p = 2usize;
        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4], [0.4, 5.0]];
        let g01 = array![[0.2_f64, -0.1], [0.3, 0.1]];
        let g10 = g01.t().to_owned();
        // Factored op with r_k = p, U = I_p (w = I_p everywhere).
        let ident = Array2::<f64>::eye(p);
        let factored = FactoredFrameKroneckerOp::new(
            vec![p, p],
            vec![2, 2],
            vec![
                FactoredFrameGBlock {
                    atom_i: 0,
                    atom_j: 0,
                    g: g00.clone(),
                    w: ident.clone(),
                },
                FactoredFrameGBlock {
                    atom_i: 1,
                    atom_j: 1,
                    g: g11.clone(),
                    w: ident.clone(),
                },
                FactoredFrameGBlock {
                    atom_i: 0,
                    atom_j: 1,
                    g: g01.clone(),
                    w: ident.clone(),
                },
                FactoredFrameGBlock {
                    atom_i: 1,
                    atom_j: 0,
                    g: g10.clone(),
                    w: ident.clone(),
                },
            ],
        )
        .expect("factored op");
        // Equivalent SparseBlockKroneckerPenaltyOp (μ-major / oc-minor, p=2).
        let sparse = SparseBlockKroneckerPenaltyOp {
            p,
            dim_a: 4,
            k: 8,
            blocks: vec![
                SparseGBlock {
                    row_off: 0,
                    col_off: 0,
                    data: g00,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 2,
                    data: g11,
                },
                SparseGBlock {
                    row_off: 0,
                    col_off: 2,
                    data: g01,
                },
                SparseGBlock {
                    row_off: 2,
                    col_off: 0,
                    data: g10,
                },
            ],
        };
        assert_eq!(factored.dim(), sparse.dim());
        let x: Vec<f64> = (0..8).map(|i| 0.2 * (i as f64) - 0.5).collect();
        let mut yf = vec![0.0_f64; 8];
        let mut ys = vec![0.0_f64; 8];
        factored.matvec(&x, &mut yf);
        sparse.matvec(&x, &mut ys);
        for i in 0..8 {
            assert!(
                (yf[i] - ys[i]).abs() < 1e-12,
                "full-rank factored op must equal SparseBlockKronecker at {i}: {} vs {}",
                yf[i],
                ys[i]
            );
        }
    }

    /// Modified Gram–Schmidt orthonormalization of the columns of a `p × r`
    /// matrix (`r ≤ p`), used by the frame-constructor tests to build genuine
    /// `St(p, r)` representatives. Returns the orthonormal `Q` (`p × r`).
    fn mgs_orthonormalize(a: &Array2<f64>) -> Array2<f64> {
        let (p, r) = a.dim();
        let mut q = a.clone();
        for j in 0..r {
            // Subtract projections onto the already-orthonormalized columns.
            for i in 0..j {
                let mut dot = 0.0;
                for c in 0..p {
                    dot += q[[c, i]] * q[[c, j]];
                }
                for c in 0..p {
                    q[[c, j]] -= dot * q[[c, i]];
                }
            }
            let mut nrm = 0.0;
            for c in 0..p {
                nrm += q[[c, j]] * q[[c, j]];
            }
            let nrm = nrm.sqrt();
            assert!(nrm > 1e-9, "mgs column {j} degenerate");
            for c in 0..p {
                q[[c, j]] /= nrm;
            }
        }
        q
    }

    /// `frame_output_gram` of an orthonormal frame with itself is the identity.
    #[test]
    fn frame_output_gram_orthonormal_is_identity() {
        let p = 5usize;
        let r = 3usize;
        // A deterministic-but-generic p×r seed, then orthonormalize.
        let mut seed = Array2::<f64>::zeros((p, r));
        for c in 0..p {
            for a in 0..r {
                seed[[c, a]] = ((c as f64) * 0.37 + (a as f64) * 1.31).sin() + 0.1 * (a as f64);
            }
        }
        let u = mgs_orthonormalize(&seed);
        let g = frame_output_gram(u.view(), u.view());
        assert_eq!(g.dim(), (r, r));
        for a in 0..r {
            for b in 0..r {
                let expect = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (g[[a, b]] - expect).abs() < 1e-12,
                    "UᵀU not identity at ({a},{b}): {}",
                    g[[a, b]]
                );
            }
        }
    }

    /// `from_frames_and_blocks` with two genuinely orthonormal frames must
    /// reproduce the hand-built dense `g ⊗ (UᵀU)` reference on every interface,
    /// computing the `W_ij` factors itself from the supplied frames.
    #[test]
    fn from_frames_and_blocks_matches_dense_reference() {
        let p = 4usize;
        // Atom 0: M_0 = 2, r_0 = 2. Atom 1: M_1 = 3, r_1 = 3.
        let basis_sizes = vec![2usize, 3];
        // Build two generic seeds and orthonormalize into St(p, r) frames.
        let mut seed0 = Array2::<f64>::zeros((p, 2));
        let mut seed1 = Array2::<f64>::zeros((p, 3));
        for c in 0..p {
            for a in 0..2 {
                seed0[[c, a]] = ((c as f64) * 0.91 - (a as f64) * 0.5).cos() + 0.2 * (c as f64);
            }
            for a in 0..3 {
                seed1[[c, a]] = ((c as f64) * 0.23 + (a as f64) * 1.7).sin() - 0.3 * (a as f64);
            }
        }
        let u0 = mgs_orthonormalize(&seed0);
        let u1 = mgs_orthonormalize(&seed1);

        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4, -0.2], [0.4, 5.0, 0.6], [-0.2, 0.6, 1.5]];
        let g01 = array![[0.2_f64, -0.1, 0.0], [0.3, 0.1, -0.2]];
        let g10 = g01.t().to_owned();

        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
        g_blocks.insert((0, 0), g00.clone());
        g_blocks.insert((1, 1), g11.clone());
        g_blocks.insert((0, 1), g01.clone());
        g_blocks.insert((1, 0), g10.clone());

        let frames = vec![Some(u0.clone()), Some(u1.clone())];
        let op =
            FactoredFrameKroneckerOp::from_frames_and_blocks(&frames, &basis_sizes, p, &g_blocks)
                .expect("from_frames_and_blocks");
        // dim = M_0·r_0 + M_1·r_1 = 2·2 + 3·3 = 13.
        assert_eq!(op.dim(), 13);

        // Hand-built dense reference: W_ij = U_iᵀ U_j computed independently.
        let ranks = vec![2usize, 3];
        let w00 = frame_output_gram(u0.view(), u0.view());
        let w11 = frame_output_gram(u1.view(), u1.view());
        let w01 = frame_output_gram(u0.view(), u1.view());
        let w10 = frame_output_gram(u1.view(), u0.view());
        let ref_blocks = vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00,
                w: w00,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11,
                w: w11,
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01,
                w: w01,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10,
                w: w10,
            },
        ];
        let reference = factored_reference_dense(&ranks, &basis_sizes, &ref_blocks);

        let dense = op.to_dense();
        for i in 0..13 {
            for j in 0..13 {
                assert!(
                    (dense[[i, j]] - reference[[i, j]]).abs() < 1e-12,
                    "to_dense mismatch at ({i},{j}): {} vs {}",
                    dense[[i, j]],
                    reference[[i, j]]
                );
            }
        }
        // matvec == reference·x.
        let x: Vec<f64> = (0..13).map(|i| 0.17 * (i as f64) - 0.6).collect();
        let mut y = vec![0.0_f64; 13];
        op.matvec(&x, &mut y);
        for i in 0..13 {
            let mut expect = 0.0;
            for j in 0..13 {
                expect += reference[[i, j]] * x[j];
            }
            assert!(
                (y[i] - expect).abs() < 1e-10,
                "matvec mismatch at {i}: {} vs {expect}",
                y[i]
            );
        }
    }

    /// Mixed framed/unframed case: atom 0 framed (`r_0 = 2 < p = 4`), atom 1
    /// unframed (`None → r_1 = p = 4`). The constructor must stand `I_p` in for
    /// the missing frame, so the within-atom-1 block is exactly `g_11 ⊗ I_4`.
    #[test]
    fn from_frames_and_blocks_mixed_framed_unframed() {
        let p = 4usize;
        let basis_sizes = vec![2usize, 2]; // M_0 = 2, M_1 = 2.
        // Atom 0 gets a genuine orthonormal 4×2 frame; atom 1 stays full-B.
        let mut seed0 = Array2::<f64>::zeros((p, 2));
        for c in 0..p {
            for a in 0..2 {
                seed0[[c, a]] = ((c as f64) * 0.61 + (a as f64) * 0.9).cos() - 0.15 * (c as f64);
            }
        }
        let u0 = mgs_orthonormalize(&seed0);

        let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
        let g11 = array![[2.0_f64, 0.4], [0.4, 5.0]];
        let g01 = array![[0.2_f64, -0.1], [0.3, 0.1]];
        let g10 = g01.t().to_owned();

        let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
            std::collections::BTreeMap::new();
        g_blocks.insert((0, 0), g00.clone());
        g_blocks.insert((1, 1), g11.clone());
        g_blocks.insert((0, 1), g01.clone());
        g_blocks.insert((1, 0), g10.clone());

        let frames = vec![Some(u0.clone()), None];
        let op =
            FactoredFrameKroneckerOp::from_frames_and_blocks(&frames, &basis_sizes, p, &g_blocks)
                .expect("from_frames_and_blocks mixed");

        // dim = M_0·r_0 + M_1·r_1 = 2·2 + 2·4 = 12.
        assert_eq!(op.ranks, vec![2usize, 4]);
        assert_eq!(op.dim(), 12);

        // The within-unframed-atom block (atom 1) must be exactly g_11 ⊗ I_4.
        // Atom 1's β range starts at offset M_0·r_0 = 4 and spans M_1·r_1 = 8.
        let dense = op.to_dense();
        let off1 = 4usize;
        for li in 0..2 {
            for lj in 0..2 {
                for a in 0..4 {
                    for b in 0..4 {
                        let gi = off1 + li * 4 + a;
                        let gj = off1 + lj * 4 + b;
                        let expect = if a == b { g11[[li, lj]] } else { 0.0 };
                        assert!(
                            (dense[[gi, gj]] - expect).abs() < 1e-12,
                            "g_11 ⊗ I_4 mismatch at ({gi},{gj}): {} vs {expect}",
                            dense[[gi, gj]]
                        );
                    }
                }
            }
        }

        // Full dense reference: W computed with U_1 = I_p for the unframed atom.
        let ranks = vec![2usize, 4];
        let ident_p = Array2::<f64>::eye(p);
        let w00 = frame_output_gram(u0.view(), u0.view());
        let w11 = frame_output_gram(ident_p.view(), ident_p.view());
        let w01 = frame_output_gram(u0.view(), ident_p.view());
        let w10 = frame_output_gram(ident_p.view(), u0.view());
        let ref_blocks = vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00,
                w: w00,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11.clone(),
                w: w11,
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01,
                w: w01,
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10,
                w: w10,
            },
        ];
        let reference = factored_reference_dense(&ranks, &basis_sizes, &ref_blocks);

        // matvec == reference·x.
        let x: Vec<f64> = (0..12).map(|i| 0.11 * (i as f64) - 0.4).collect();
        let mut y = vec![0.0_f64; 12];
        op.matvec(&x, &mut y);
        for i in 0..12 {
            let mut expect = 0.0;
            for j in 0..12 {
                expect += reference[[i, j]] * x[j];
            }
            assert!(
                (y[i] - expect).abs() < 1e-10,
                "mixed matvec mismatch at {i}: {} vs {expect}",
                y[i]
            );
        }
    }

    /// Verify the arrow-Schur solve against a small dense reference.
    /// Build the joint bordered system as a single dense (K + N·d)² matrix,
    /// solve it with the local cholesky_lower path, and compare to the
    /// arrow-Schur output.
    #[test]
    fn arrow_schur_matches_dense_reference_2x2() {
        // N = 2 rows, d = 2 latent, K = 3 β.
        let n = 2;
        let d = 2;
        let k = 3;
        let mut sys = ArrowSchurSystem::new(n, d, k);

        // Row 0: H_tt = [[2, 0.1],[0.1, 3]], H_tβ = [[1, 0, 0.5],[0.2, 1, 0]],
        //         g_t = [0.3, -0.2].
        sys.rows[0].htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.0, 0.5], [0.2, 1.0, 0.0]];
        sys.rows[0].gt = array![0.3_f64, -0.2];

        // Row 1.
        sys.rows[1].htt = array![[1.5_f64, -0.1], [-0.1, 2.0]];
        sys.rows[1].htbeta = array![[0.1_f64, 0.5, 0.0], [0.0, 0.3, 1.0]];
        sys.rows[1].gt = array![-0.1_f64, 0.4];

        // β-block.
        sys.hbb = array![[4.0_f64, 0.2, 0.0], [0.2, 5.0, 0.1], [0.0, 0.1, 6.0],];
        sys.gb = array![0.5_f64, -0.3, 0.2];

        let (delta_t, delta_beta, _diag) = sys.solve(0.0, 0.0).expect("arrow-schur solve");
        let streaming_options = ArrowSolveOptions::direct().with_streaming_chunk_size(Some(1));
        let (delta_t_stream, delta_beta_stream, _diag_stream) = sys
            .solve_with_options(0.0, 0.0, &streaming_options)
            .expect("streaming arrow-schur solve");
        assert_eq!(delta_beta, delta_beta_stream);
        assert_eq!(delta_t, delta_t_stream);

        // Build dense reference: order is [β; t_0; t_1] = K + N·d entries.
        let total = k + n * d;
        let mut hjoint = Array2::<f64>::zeros((total, total));
        let mut gjoint = Array1::<f64>::zeros(total);
        // β-β block.
        for a in 0..k {
            for b in 0..k {
                hjoint[[a, b]] = sys.hbb[[a, b]];
            }
            gjoint[a] = sys.gb[a];
        }
        // t-blocks and cross-blocks.
        for i in 0..n {
            let toff = k + i * d;
            for a in 0..d {
                for b in 0..d {
                    hjoint[[toff + a, toff + b]] = sys.rows[i].htt[[a, b]];
                }
                gjoint[toff + a] = sys.rows[i].gt[a];
                for a2 in 0..k {
                    hjoint[[toff + a, a2]] = sys.rows[i].htbeta[[a, a2]];
                    hjoint[[a2, toff + a]] = sys.rows[i].htbeta[[a, a2]];
                }
            }
        }
        // Solve hjoint · x = -gjoint via cholesky.
        let lj = cholesky_lower(&hjoint).expect("dense ref PD");
        let neg_g = gjoint.mapv(|v| -v);
        let xref = cholesky_solve_vector(&lj, &neg_g);
        // Compare β.
        for a in 0..k {
            assert!(
                (xref[a] - delta_beta[a]).abs() < 1e-10,
                "β[{a}] mismatch: dense {} vs arrow {}",
                xref[a],
                delta_beta[a]
            );
        }
        // Compare t.
        for i in 0..n {
            for a in 0..d {
                let dense = xref[k + i * d + a];
                let arrow = delta_t[i * d + a];
                assert!(
                    (dense - arrow).abs() < 1e-10,
                    "t[{i},{a}] mismatch: dense {dense} vs arrow {arrow}"
                );
            }
        }
    }

    fn diagonal_arrow_fixture(row_min: f64, schur_min: f64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(2, 2, 2);
        sys.rows[0].htt = array![[row_min, 0.0], [0.0, row_min + 1.0]];
        sys.rows[1].htt = array![[row_min + 2.0, 0.0], [0.0, row_min + 3.0]];
        for row in sys.rows.iter_mut() {
            row.htbeta.fill(0.0);
            row.gt.fill(0.0);
        }
        sys.hbb = array![[schur_min, 0.0], [0.0, schur_min + 1.0]];
        sys.gb.fill(0.0);
        sys
    }

    fn diagonal_fixture_dense_lambda_min(sys: &ArrowSchurSystem) -> f64 {
        let mut out = f64::INFINITY;
        for row in &sys.rows {
            for axis in 0..row.htt.nrows() {
                out = out.min(row.htt[[axis, axis]]);
            }
        }
        for axis in 0..sys.hbb.nrows() {
            out = out.min(sys.hbb[[axis, axis]]);
        }
        out
    }

    #[test]
    fn arrow_factor_min_pivot_matches_dense_lambda_min_ordering() {
        let weak = diagonal_arrow_fixture(0.2, 0.8);
        let strong = diagonal_arrow_fixture(0.7, 1.2);
        let options = ArrowSolveOptions::direct();
        let (_dt_w, _db_w, weak_cache) =
            solve_arrow_newton_step_with_options(&weak, 0.0, 0.0, &options)
                .expect("weak diagonal fixture should factor");
        let (_dt_s, _db_s, strong_cache) =
            solve_arrow_newton_step_with_options(&strong, 0.0, 0.0, &options)
                .expect("strong diagonal fixture should factor");

        let weak_lambda = diagonal_fixture_dense_lambda_min(&weak);
        let strong_lambda = diagonal_fixture_dense_lambda_min(&strong);
        assert!(weak_lambda < strong_lambda);

        let weak_pivot = arrow_factor_min_pivot(&weak_cache)
            .min_pivot
            .expect("weak pivot");
        let strong_pivot = arrow_factor_min_pivot(&strong_cache)
            .min_pivot
            .expect("strong pivot");
        assert_abs_diff_eq!(weak_pivot, weak_lambda, epsilon = 1.0e-14);
        assert_abs_diff_eq!(strong_pivot, strong_lambda, epsilon = 1.0e-14);
        assert!(weak_pivot < strong_pivot);
    }

    fn quartic_counterexample_value(t: f64) -> f64 {
        0.25 * t.powi(4) - t * t + 2.0 * t
    }

    fn quartic_counterexample_system(t: f64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(1, 1, 0);
        sys.rows[0].gt = array![t.powi(3) - 2.0 * t + 2.0];
        sys.rows[0].htt = array![[3.0 * t * t - 2.0]];
        sys
    }

    #[test]
    fn proximal_correction_breaks_scalar_newton_cycle() {
        let options = ArrowSolveOptions::direct();
        let correction = ArrowProximalCorrectionOptions {
            initial_ridge: 1e-8,
            ridge_growth: 10.0,
            max_attempts: 16,
            armijo_c1: 1e-4,
            gradient_tolerance: 1e-12,
            convergence_objective_rel_tol: DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL,
        };
        let mut t = 0.0_f64;
        let mut previous_value = quartic_counterexample_value(t);

        for _ in 0..32 {
            let sys = quartic_counterexample_system(t);
            let accepted = solve_arrow_newton_step_with_proximal_correction(
                &sys,
                0.0,
                0.0,
                previous_value,
                &options,
                &correction,
                |delta_t, _delta_beta| quartic_counterexample_value(t + delta_t[0]),
            )
            .expect("proximal correction should accept a descent step");
            assert!(
                accepted.trial_objective_value <= previous_value,
                "accepted step must not increase the objective"
            );
            t += accepted.delta_t[0];
            previous_value = accepted.trial_objective_value;
        }

        let final_grad = t.powi(3) - 2.0 * t + 2.0;
        assert!(
            final_grad.abs() < 1e-7,
            "corrected iteration should reach the scalar critical point; t={t}, g={final_grad}"
        );
    }

    /// Issue #195 / gam#578: a per-row block that is barely-PD (smallest
    /// pivot on the order of ε·trace — a rank-deficient / over-parameterized
    /// decoder atom) factors successfully but is unsafe to use raw in the
    /// Schur reduction. The κ proxy is folded INTO the per-row ridge
    /// escalation loop: rather than reject such a block outright (which made
    /// the advertised Arrow-Schur ridge never actually run and aborted the
    /// whole SAE fit, gam#578), `factor_one_row` lifts this row's ridge until
    /// the block is BOTH positive-definite and well-conditioned, then returns
    /// a genuinely conditioned factor safe to plug into
    /// `S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`.
    /// Only a block that cannot be conditioned even at `ridge_cap` errors.
    #[test]
    fn factor_one_row_conditions_barely_pd_block_via_ridge() {
        let d = 2;
        let k = 2;
        let mut row = ArrowRowBlock::new(d, k);
        // Matrix from the issue body: PD by an exact ε along the second
        // direction. Cholesky succeeds at ridge 0, but κ ≈ 1e14 — far past
        // the safe inversion regime. This is exactly the rank-deficient
        // decoder-atom block gam#578 advertised the ridge would stabilize.
        row.htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
        row.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row.gt = array![0.0_f64, 0.0];

        // The fix: instead of rejecting, the escalation loop lifts this
        // row's ridge until the factor is well-conditioned. The returned
        // factor must satisfy the κ ceiling that a raw barely-PD block fails.
        let factor = factor_one_row(&row, 0.0, d, 0, false).expect(
            "barely-PD H_tt must be CONDITIONED by per-row ridge escalation, not rejected (gam#578)",
        );
        let kappa = cholesky_factor_kappa_estimate(&factor);
        assert!(
            kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
            "conditioned factor must be within the safe-inversion κ ceiling; got κ={kappa:e}"
        );
        // The factor is a genuine Cholesky of the ridge-lifted block
        // H_tt + ridge_eff·I (ridge_eff ≥ 0), so reconstructing L Lᵀ must
        // match H_tt up to a nonnegative diagonal shift (never below).
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += factor[[i, kk]] * factor[[j, kk]];
                }
                if i == j {
                    assert!(
                        acc >= row.htt[[i, j]] - 1e-12,
                        "diagonal of L Lᵀ must be H_tt + (nonneg ridge) at ({i},{j}): \
                         {acc} vs {}",
                        row.htt[[i, j]]
                    );
                } else {
                    assert!(
                        (acc - row.htt[[i, j]]).abs() < 1e-9,
                        "off-diagonal of L Lᵀ must equal H_tt at ({i},{j}): {acc} vs {}",
                        row.htt[[i, j]]
                    );
                }
            }
        }

        // Evidence/log-det mode (`tolerate_ill_conditioning = true`) must
        // accept the same barely-PD block and return its genuine Cholesky
        // factor — the diagonal gives an exact log-determinant.
        let factor = factor_one_row(&row, 0.0, d, 0, true)
            .expect("tolerate_ill_conditioning must accept a barely-PD-but-PD block");
        // L Lᵀ must reproduce the original block (the factor is real, not a
        // damped surrogate).
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += factor[[i, kk]] * factor[[j, kk]];
                }
                assert!(
                    (acc - row.htt[[i, j]]).abs() < 1e-12,
                    "tolerated factor must satisfy L Lᵀ = H_tt at ({i},{j})"
                );
            }
        }

        // A genuinely non-PD block must STILL error even under tolerance —
        // the flag lifts only the κ rejection, not the PD requirement.
        let mut row_npd = ArrowRowBlock::new(d, k);
        row_npd.htt = array![[1.0_f64, 2.0], [2.0, 1.0]]; // indefinite (eigvals 3, -1)
        row_npd.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_npd.gt = array![0.0_f64, 0.0];
        let npd = factor_one_row(&row_npd, 0.0, d, 0, true);
        assert!(
            matches!(npd, Err(ArrowSchurError::PerRowFactorFailed { .. })),
            "non-PD block must error even with tolerate_ill_conditioning; got {npd:?}"
        );

        // Sanity: a well-conditioned block at the same dimension still
        // factors successfully.
        let mut row_ok = ArrowRowBlock::new(d, k);
        row_ok.htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
        row_ok.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_ok.gt = array![0.0_f64, 0.0];
        factor_one_row(&row_ok, 0.0, d, 0, false)
            .expect("well-conditioned block must still factor at ridge_t=0");

        // A block that cannot be conditioned at all — a non-finite entry —
        // is genuinely broken: no finite ridge shift repairs it, so the
        // escalation loop must still surface a typed `PerRowFactorFailed`
        // for the outer loop rather than loop forever or return garbage.
        let mut row_nan = ArrowRowBlock::new(d, k);
        row_nan.htt = array![[f64::NAN, 0.0], [0.0, 1.0]];
        row_nan.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        row_nan.gt = array![0.0_f64, 0.0];
        let nan = factor_one_row(&row_nan, 1.0e-6, d, 0, false);
        assert!(
            matches!(nan, Err(ArrowSchurError::PerRowFactorFailed { .. })),
            "non-finite block must surface PerRowFactorFailed, not loop or condition; got {nan:?}"
        );
    }

    #[test]
    fn factor_one_row_conditions_scalar_tiny_pivot_via_ridge() {
        let d = 1;
        let k = 1;
        let mut row = ArrowRowBlock::new(d, k);
        row.htt = array![[1.0e-20_f64]];
        row.htbeta = array![[1.0_f64]];
        row.gt = array![0.0_f64];

        let factor = factor_one_row(&row, 0.0, d, 0, false)
            .expect("tiny positive scalar pivot must be ridge-conditioned");
        let pivot = factor[[0, 0]] * factor[[0, 0]];
        assert!(
            pivot >= safe_spd_pivot_min(1.0),
            "scalar pivot must be lifted above the absolute safe floor; got {pivot:e}"
        );
        assert!(
            pivot > row.htt[[0, 0]],
            "scalar block must not be accepted at the raw tiny pivot"
        );

        let tolerated = factor_one_row(&row, 0.0, d, 0, true)
            .expect("tolerated log-det path must accept a positive scalar block");
        let raw_pivot = tolerated[[0, 0]] * tolerated[[0, 0]];
        assert!(
            (raw_pivot - row.htt[[0, 0]]).abs() < 1.0e-30,
            "tolerated factor must remain the raw scalar Cholesky"
        );
    }

    /// #1117/#1118: a per-row `H_tt` that is gauge-flat AND genuinely indefinite
    /// off the gauge orbit (the K>1 IBP/softmax row-sharing state) must be
    /// conditioned by the undamped evidence factor through **unit-stiffness
    /// spectral deflation** — `factor_spectral_deflated_evidence_row` discovers
    /// the negative/flat eigen-direction the closed-form gauge deflation cannot
    /// rescue and stiffens it to eigenvalue `+1` (a ρ-independent `log 1 = 0`
    /// evidence contribution), NOT a ρ-dependent `+ridge·I` bias. And the
    /// STATIONARY version of the same block (the indefinite direction now
    /// positive, i.e. genuinely PD) must factor through the undamped evidence
    /// path to the EXACT Cholesky `L Lᵀ = H_tt` with NO bias. This pins the
    /// contract the `converge_inner_for_undamped_logdet` path relies on:
    /// finite-and-bias-free pre-stationarity (so the outer REML value and its
    /// analytic ρ-gradient agree), exact-and-unbiased at the optimum.
    #[test]
    fn evidence_row_spectral_deflates_indefinite_non_gauge_block_at_unit_stiffness() {
        let d = 3usize;
        let k = 2usize;

        // Pre-stationarity block: e_1 is a near-null GAUGE direction (curvature
        // 1e-10, far below GAUGE_RAYLEIGH_EPS·max_diag = 1e-8·4 = 4e-8, so it
        // qualifies for Faddeev-Popov deflation), e_2 is GENUINELY indefinite
        // (eigenvalue −1.0 — real negative curvature, NOT a gauge orbit). The
        // gauge deflation lifts only e_1 (→ +1), leaving the −1.0 along e_2, so
        // the closed-form gauge deflation alone cannot make the block PD.
        let mut indef = ArrowRowBlock::new(d, k);
        indef.htt = array![
            [4.0_f64, 0.0, 0.0],
            [0.0, 1.0e-10, 0.0],
            [0.0, 0.0, -1.0],
        ];
        indef.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0], [0.5, 0.5]];
        indef.gt = array![0.0_f64, 0.0, 0.0];
        let gauge_e1 = array![0.0_f64, 1.0, 0.0];

        // Gauge deflation cannot manufacture a PD block: the −1.0 along e_2 is
        // genuine indefiniteness, not a near-null orbit, so deflating e_1 leaves
        // it negative and the closed-form deflation returns None.
        assert!(
            factor_gauge_deflated_evidence_row(&indef, d, std::slice::from_ref(&gauge_e1)).is_none(),
            "gauge deflation must NOT rescue a genuinely-indefinite non-gauge direction"
        );

        // Spectral deflation DISCOVERS the negative e_2 direction (and the flat
        // e_1) from the block's own eigendecomposition and stiffens BOTH to +1,
        // producing an SPD block. The two sub-floor eigenvalues (−1.0 and 1e-10
        // vs floor = 1e-8·4) are counted; the genuine e_0 (eigenvalue 4.0) is
        // preserved exactly.
        let spectral = factor_spectral_deflated_evidence_row(&indef, d)
            .expect("spectral deflation must condition the indefinite non-gauge block");
        assert_eq!(
            spectral.gauge_deflated_directions, 2,
            "the two sub-floor eigen-directions (−1.0 and 1e-10) must be unit-deflated"
        );
        // Reconstruct L Lᵀ: e_0 keeps 4.0; the two deflated axes each carry +1.
        let ls = &spectral.factor;
        let mut recon = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += ls[[i, kk]] * ls[[j, kk]];
                }
                recon[[i, j]] = acc;
            }
        }
        assert!(
            (recon[[0, 0]] - 4.0).abs() < 1.0e-9,
            "genuine direction e_0 must be preserved exactly; got {}",
            recon[[0, 0]]
        );
        assert!(
            (recon[[2, 2]] - 1.0).abs() < 1.0e-9,
            "the genuinely-indefinite direction e_2 must be deflated to unit \
             stiffness +1 (log 1 = 0, ρ-independent), NOT ridge-damped; got {}",
            recon[[2, 2]]
        );

        // The undamped evidence factor (tolerate_ill_conditioning, ridge_t = 0,
        // gauge passed in) now SUCCEEDS on this block via spectral deflation
        // rather than refusing — so the SAE driver gets a finite, BIAS-FREE
        // evidence cache and never falls back to a ρ-dependent ridge.
        let factored = factor_one_row_result(&indef, 0.0, d, 0, true, std::slice::from_ref(&gauge_e1))
            .expect("undamped evidence factor must condition the indefinite block by deflation");
        for a in 0..d {
            assert!(
                factored.factor[[a, a]].is_finite() && factored.factor[[a, a]] > 0.0,
                "deflated evidence factor must have a finite positive pivot at {a}; got {}",
                factored.factor[[a, a]]
            );
        }

        // Stationary block: the previously-indefinite e_2 direction is now
        // positive (genuine PD), the gauge direction e_1 stays near-null. The
        // undamped evidence factor must SUCCEED and return the EXACT Cholesky of
        // the block (with the unit-stiffness deflation on the gauge direction
        // contributing exactly +1 there, log(1) = 0 to the evidence) — NO ridge
        // bias. This is the converged state whose value/gradient must be
        // bit-identical to today's.
        let mut pd = ArrowRowBlock::new(d, k);
        pd.htt = array![
            [4.0_f64, 0.0, 0.0],
            [0.0, 1.0e-10, 0.0],
            [0.0, 0.0, 2.0],
        ];
        pd.htbeta = indef.htbeta.clone();
        pd.gt = array![0.0_f64, 0.0, 0.0];

        let result = factor_one_row_result(&pd, 0.0, d, 0, true, std::slice::from_ref(&gauge_e1))
            .expect("undamped evidence factor must succeed on the genuinely-PD stationary block");
        // Exactly one gauge direction deflated; the non-gauge spectrum is
        // factored as-is (no ridge), so L Lᵀ reproduces H_tt on the two genuine
        // directions and the deflated gauge direction carries the +1 stiffness.
        assert_eq!(
            result.gauge_deflated_directions, 1,
            "exactly the single near-null gauge direction must be deflated"
        );
        let l = &result.factor;
        let mut reconstructed = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += l[[i, kk]] * l[[j, kk]];
                }
                reconstructed[[i, j]] = acc;
            }
        }
        // Genuine directions: exact, no ridge bias.
        assert!(
            (reconstructed[[0, 0]] - 4.0).abs() < 1.0e-12,
            "stationary factor must be the EXACT Cholesky on the genuine direction e_0; got {}",
            reconstructed[[0, 0]]
        );
        assert!(
            (reconstructed[[2, 2]] - 2.0).abs() < 1.0e-12,
            "stationary factor must be the EXACT Cholesky on the genuine direction e_2; got {}",
            reconstructed[[2, 2]]
        );
        // Gauge direction: raw curvature 1e-10 + unit Faddeev-Popov stiffness 1.0.
        assert!(
            (reconstructed[[1, 1]] - (1.0 + 1.0e-10)).abs() < 1.0e-9,
            "deflated gauge direction must carry exactly the +1 unit stiffness; got {}",
            reconstructed[[1, 1]]
        );
    }

    /// #1117 flicker guard: a per-row evidence block carrying ONE genuinely
    /// indefinite direction (so spectral deflation runs) plus a small POSITIVE
    /// eigenvalue parked right at the relative cutoff `floor = REL_FLOOR·max|λ|`
    /// must report the SAME deflation count at two infinitesimally different
    /// "ρ values" that straddle the bare floor. Without the hysteresis band the
    /// positive near-floor eigenvalue would be counted as deflated on one side
    /// (`λ ≤ floor`) and live on the other (`λ > floor`), flipping the per-row
    /// count and tripping the quotient-dimension guard
    /// (`record_evidence_gauge_deflation_count`) mid-optimization — the slow
    /// seed/homotopy cascade. The genuine indefinite direction (the true
    /// quotient null) is deflated on BOTH sides, so the count is stable.
    #[test]
    fn evidence_row_spectral_deflation_count_is_stable_across_the_cutoff() {
        let d = 3usize;
        let k = 1usize;
        // max|λ| = 4.0 ⇒ floor = SPECTRAL_DEFLATION_REL_FLOOR·4 = 4e-8. Place the
        // small positive eigenvalue just BELOW and just ABOVE the bare floor at
        // two ρ-walk iterates; the third direction is genuinely indefinite
        // (−1.0) so spectral deflation runs on both.
        let floor = SPECTRAL_DEFLATION_REL_FLOOR * 4.0;

        // The bare cutoff is the knife-edge: `λ ≤ floor` would deflate the lo
        // iterate and keep the hi iterate, flipping the count. The hysteresis
        // floor is `floor·(1−1e-2) = floor·0.99`, so picking both iterates
        // strictly ABOVE it (0.995·floor and 1.05·floor) keeps them on the same
        // (KEEP) side of the banded decision while still straddling the BARE
        // floor — exactly the flicker regime the fix removes.
        let near_floor_lo = floor * 0.995; // bare cutoff: deflated; banded: kept
        let near_floor_hi = floor * 1.05; // bare cutoff: live; banded: kept

        let mut block_lo = ArrowRowBlock::new(d, k);
        block_lo.htt = array![
            [4.0_f64, 0.0, 0.0],
            [0.0, near_floor_lo, 0.0],
            [0.0, 0.0, -1.0],
        ];
        block_lo.htbeta = array![[1.0_f64], [0.0], [0.5]];
        block_lo.gt = array![0.0_f64, 0.0, 0.0];

        let mut block_hi = block_lo.clone();
        block_hi.htt[[1, 1]] = near_floor_hi;

        let lo = factor_spectral_deflated_evidence_row(&block_lo, d)
            .expect("indefinite block must spectrally deflate (lo iterate)");
        let hi = factor_spectral_deflated_evidence_row(&block_hi, d)
            .expect("indefinite block must spectrally deflate (hi iterate)");

        // The genuine −1.0 quotient direction is deflated on both sides; the
        // small positive near-floor direction is KEPT on both sides thanks to
        // the hysteresis band, so the count does NOT flicker.
        assert_eq!(
            lo.gauge_deflated_directions, 1,
            "lo iterate: only the genuine indefinite direction is deflated"
        );
        assert_eq!(
            hi.gauge_deflated_directions, lo.gauge_deflated_directions,
            "deflation count must be STABLE across an eigenvalue straddling the \
             bare cutoff — the quotient-dimension guard must not trip mid-walk"
        );

        // Sanity: the bare (non-hysteresis) cutoff WOULD have split these two
        // iterates, confirming the test actually exercises the flicker regime.
        let bare_count = |lambda: f64| -> usize {
            let mut c = 0usize;
            for &l in &[4.0_f64, lambda, -1.0] {
                if !(l.is_finite() && l > floor) {
                    c += 1;
                }
            }
            c
        };
        assert_ne!(
            bare_count(near_floor_lo),
            bare_count(near_floor_hi),
            "test must straddle the bare cutoff (else it proves nothing): the \
             un-banded decision flips the count, the banded one does not"
        );
    }

    /// #1118 (β-block analogue): a genuinely indefinite REDUCED SCHUR complement
    /// — the state the OLMo K=8 capstone hits, where the per-row H_tt blocks are
    /// deflated PD but the Schur subtraction drives a β-pivot negative (the
    /// reported `-0.064 at index 256`) — must be conditioned by the evidence
    /// dense factor through unit-stiffness spectral deflation rather than failing
    /// the whole fit. The negative direction is stiffened to eigenvalue `+1`
    /// (ρ-independent `log 1 = 0`), the genuine positive spectrum is preserved
    /// exactly, and the result is PD so its Cholesky and `log|S|` are finite.
    #[test]
    fn evidence_dense_schur_deflates_indefinite_complement_at_unit_stiffness() {
        // A 3×3 symmetric Schur complement with one genuinely NEGATIVE eigenvalue
        // (−0.5 along e_1) and two healthy positive ones (4.0 along e_0, 2.0 along
        // e_2). The plain Cholesky must refuse it; the evidence deflation must
        // condition it to PD.
        let schur = array![
            [4.0_f64, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 2.0],
        ];
        assert!(
            cholesky_lower(&schur).is_err(),
            "an indefinite Schur complement must be refused by the plain Cholesky"
        );

        let factor = factor_spectral_deflated_evidence_dense(&schur)
            .expect("indefinite Schur complement must spectrally deflate to a PD factor");

        // Reconstruct L Lᵀ and check the spectrum: genuine directions exact, the
        // deflated negative direction carries the +1 unit stiffness.
        let d = 3usize;
        let mut reconstructed = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0_f64;
                for kk in 0..d {
                    acc += factor[[i, kk]] * factor[[j, kk]];
                }
                reconstructed[[i, j]] = acc;
            }
        }
        assert!(
            (reconstructed[[0, 0]] - 4.0).abs() < 1.0e-9,
            "genuine positive direction e_0 must be exact; got {}",
            reconstructed[[0, 0]]
        );
        assert!(
            (reconstructed[[2, 2]] - 2.0).abs() < 1.0e-9,
            "genuine positive direction e_2 must be exact; got {}",
            reconstructed[[2, 2]]
        );
        assert!(
            (reconstructed[[1, 1]] - 1.0).abs() < 1.0e-9,
            "deflated negative direction must carry exactly the +1 unit stiffness; got {}",
            reconstructed[[1, 1]]
        );
    }

    #[test]
    fn sys_htbeta_materialize_row_sums_operator_and_dense_slab() {
        let mut sys = ArrowSchurSystem::new(1, 1, 3);
        sys.rows[0].htbeta = array![[0.25_f64, 0.5, 0.75]];
        sys.activate_dense_htbeta_supplement();
        sys.set_row_htbeta_operator(
            |row_idx, x, out| {
                assert_eq!(row_idx, 0);
                out[0] += 2.0 * x[0] - x[1] + 0.5 * x[2];
            },
            |row_idx, v, out| {
                assert_eq!(row_idx, 0);
                out[0] += 2.0 * v[0];
                out[1] -= v[0];
                out[2] += 0.5 * v[0];
            },
        );

        let htbeta = sys_htbeta_materialize_row(&sys, 0, &sys.rows[0]);
        assert_eq!(htbeta, array![[2.25_f64, -0.5, 1.25]]);
    }

    /// Issue #195 / gam#578 / gam#845: when the per-row block is barely-PD at
    /// `ridge_t = 0` (a rank-deficient atom), the per-row factor must
    /// CONDITION it through the folded ridge escalation, and the full
    /// `solve_with_lm_escalation_inner` must produce a finite Newton step
    /// rather than aborting the whole fit.
    ///
    /// Note (gam#845): per-row κ-conditioning bounds each block's inverse
    /// spectrum, but it cannot on its own guarantee the *dense Schur
    /// complement* `S = H_ββ − Σ_i H_tβᵀ(H_tt+ridge)⁻¹H_tβ` stays PD: the
    /// per-row ceiling still admits a ~`1/κ_ceiling`-scale smallest pivot, so
    /// `(H_tt+ridge)⁻¹` retains a ~`κ_ceiling`-scale eigenvalue that, after the
    /// Schur subtraction, can drive `S` strongly indefinite when
    /// `‖H_tβ‖²·κ_ceiling ≫ ‖H_ββ‖`. Outer LM ridge escalation is the correct,
    /// principled recovery for that regime. The achievable invariant is
    /// therefore: a finite, well-conditioned Newton step is produced (via a
    /// bounded number of outer ridge escalations), NOT zero escalations.
    #[test]
    fn lm_escalation_recovers_from_ill_conditioned_row() {
        let n = 1;
        let d = 2;
        let k = 2;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        // Same barely-PD row as the issue body.
        sys.rows[0].htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
        sys.rows[0].gt = array![0.1_f64, -0.2];
        sys.hbb = array![[4.0_f64, 0.2], [0.2, 5.0]];
        sys.gb = array![0.3_f64, -0.1];

        // Direct factor at ridge_t=0 CONDITIONS the barely-PD block via the
        // folded per-row ridge escalation (gam#578: the advertised ridge
        // genuinely stabilizes the deficient direction instead of rejecting
        // it) and returns a well-conditioned factor satisfying the κ ceiling.
        let factor = factor_one_row(&sys.rows[0], 0.0, d, 0, false)
            .expect("barely-PD row must be conditioned, not rejected (gam#578)");
        let kappa = cholesky_factor_kappa_estimate(&factor);
        assert!(
            kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
            "conditioned per-row factor must satisfy the κ ceiling; got κ={kappa:e}"
        );

        // The full LM-escalating wrapper produces a finite, well-conditioned
        // Newton step. Per-row conditioning alone cannot keep the dense Schur
        // complement PD here (κ_ceiling × ‖H_tβ‖² ≫ ‖H_ββ‖), so the proximal
        // wrapper escalates the outer ridge a bounded number of times — this
        // is the correct recovery (gam#845), not a failure.
        let options = ArrowSolveOptions::direct();
        let (delta_t, delta_beta, diag) = solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
            .expect("LM escalation must recover from a barely-PD per-row block");
        for v in delta_t.iter().chain(delta_beta.iter()) {
            assert!(v.is_finite(), "recovered step must be finite: {v}");
        }
        assert!(
            diag.ridge_escalations <= DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            "recovery must use a bounded number of outer ridge escalations; got {}",
            diag.ridge_escalations
        );
    }

    /// `latent_block_inverse_diagonal` must reproduce the `t`-block diagonal of
    /// the dense bordered-arrow inverse `(H⁻¹)_tt` to machine precision.
    ///
    /// Build a small `(N=3, d=2, K=2)` arrow system, factor it through the
    /// real solve to obtain an [`ArrowFactorCache`], then assemble the full
    /// dense `(N·d + K) × (N·d + K)` Hessian from the same per-row blocks,
    /// invert it via dense Cholesky, and compare diagonals.
    #[test]
    fn latent_block_inverse_diagonal_matches_dense() {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);

        // Distinct, well-conditioned per-row blocks and cross-blocks.
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        // SPD shared block; the full bordered H must stay PD.
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.0_f64, 0.0];

        let options = ArrowSolveOptions::direct();
        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("direct arrow solve should factor this SPD system");

        // Assemble the dense bordered-arrow Hessian H (t-coords first, then β).
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            // H_tt^(i) block.
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
            }
            // H_tβ^(i) (d×K) and its transpose into the β border.
            for r in 0..d {
                for c in 0..k {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        // H_ββ.
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
            }
        }

        // Dense inverse via Cholesky against the identity.
        let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

        let diag = cache
            .latent_block_inverse_diagonal()
            .expect("dense Schur cache must support the selected-inverse diagonal");
        assert_eq!(diag.len(), n * d);
        for i in 0..n {
            for j in 0..d {
                let idx = i * d + j; // homogeneous system ⇒ row_offsets[i] == i*d.
                let expected = h_inv[[idx, idx]];
                let got = diag[idx];
                assert!(
                    (got - expected).abs() < 1e-9,
                    "row {i} axis {j}: selected-inverse diag {got} vs dense {expected}"
                );
            }
        }

        // The per-(atom, axis) trace is a sum over the relevant indices; e.g.
        // tr[(H⁻¹)_tt] over all latent coords equals the dense t-block trace.
        let trace_selected: f64 = diag.iter().sum();
        let trace_dense: f64 = (0..n * d).map(|idx| h_inv[[idx, idx]]).sum();
        assert!(
            (trace_selected - trace_dense).abs() < 1e-9,
            "full latent trace {trace_selected} vs dense {trace_dense}"
        );
    }

    /// `full_inverse_apply` (#1006 IFT/adjoint back-solve) must reproduce the dense
    /// bordered-arrow inverse applied to an arbitrary arrow-layout RHS, and
    /// solving against the system's own gradient must reproduce the Newton
    /// step the solver itself returned (`Δ = H⁻¹g`) — both to near machine
    /// precision on the ridge-0 Direct factor.
    #[test]
    fn full_inverse_apply_matches_dense_inverse_and_newton_step() {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[0].gt = array![0.4_f64, -0.7];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[1].gt = array![-0.2_f64, 0.9];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        sys.rows[2].gt = array![1.1_f64, 0.3];
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.5_f64, -0.8];

        let options = ArrowSolveOptions::direct();
        let (delta_t, delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("direct arrow solve should factor this SPD system");

        // (a) The solver returns the DESCENT step Δ = −H⁻¹g; full_inverse_apply is the
        // bare inverse application H⁻¹g, so u must equal −Δ exactly.
        let mut g_t = Array1::<f64>::zeros(n * d);
        for i in 0..n {
            for j in 0..d {
                g_t[i * d + j] = sys.rows[i].gt[j];
            }
        }
        let (u_t, u_beta) = cache
            .full_inverse_apply(g_t.view(), sys.gb.view())
            .expect("full_inverse_apply on the ridge-0 Direct cache");
        for idx in 0..n * d {
            assert!(
                (u_t[idx] + delta_t[idx]).abs() < 1e-10,
                "t[{idx}]: full_inverse_apply {} vs −(Newton step) {}",
                u_t[idx],
                -delta_t[idx]
            );
        }
        for c in 0..k {
            assert!(
                (u_beta[c] + delta_beta[c]).abs() < 1e-10,
                "beta[{c}]: full_inverse_apply {} vs −(Newton step) {}",
                u_beta[c],
                -delta_beta[c]
            );
        }

        // (b) Arbitrary RHS vs the dense bordered inverse.
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
                for c in 0..k {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
            }
        }
        let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let mut w_full = Array1::<f64>::zeros(dim);
        for (idx, v) in w_full.iter_mut().enumerate() {
            *v = 0.3 + 0.17 * (idx as f64) * (if idx % 2 == 0 { 1.0 } else { -1.0 });
        }
        let dense_u = cholesky_solve_vector(&l, &w_full);
        let (u_t2, u_beta2) = cache
            .full_inverse_apply(
                w_full.slice(ndarray::s![..n * d]),
                w_full.slice(ndarray::s![n * d..]),
            )
            .expect("full_inverse_apply on arbitrary RHS");
        for idx in 0..n * d {
            assert!(
                (u_t2[idx] - dense_u[idx]).abs() < 1e-10,
                "t[{idx}]: full_inverse_apply {} vs dense {}",
                u_t2[idx],
                dense_u[idx]
            );
        }
        for c in 0..k {
            assert!(
                (u_beta2[c] - dense_u[n * d + c]).abs() < 1e-10,
                "beta[{c}]: full_inverse_apply {} vs dense {}",
                u_beta2[c],
                dense_u[n * d + c]
            );
        }
    }

    /// `schur_inverse_apply` / `schur_inverse_block` must reproduce the
    /// β-block of the dense bordered-arrow inverse `(H⁻¹)_ββ = S_β⁻¹`, and a
    /// caller-assembled `tr(S_β⁻¹ M)` must match the dense Kron-block trace —
    /// the β-side analogue used by the SAE λ_smooth Fellner-Schall step.
    #[test]
    fn schur_inverse_beta_block_matches_dense() {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.0_f64, 0.0];

        let options = ArrowSolveOptions::direct();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("direct arrow solve should factor this SPD system");

        // Dense bordered H and its inverse (same assembly as the t-block test).
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
            }
            for r in 0..d {
                for c in 0..k {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
            }
        }
        let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

        // The β-block of H⁻¹ is the bottom-right K×K corner.
        let beta_off = n * d;

        // schur_inverse_apply against each unit column reproduces the full
        // β-block (every entry, not just the diagonal).
        for col in 0..k {
            let mut e = Array1::<f64>::zeros(k);
            e[col] = 1.0;
            let x = cache
                .schur_inverse_apply(e.view())
                .expect("dense Schur cache must support schur_inverse_apply");
            for r in 0..k {
                let expected = h_inv[[beta_off + r, beta_off + col]];
                assert!(
                    (x[r] - expected).abs() < 1e-9,
                    "S_β⁻¹[{r},{col}] {} vs dense {expected}",
                    x[r]
                );
            }
        }

        // Caller-assembled Kron trace tr(S_β⁻¹ M) for a single atom block
        // M = A_k ⊗ I_p with K = M_k · p. Here M_k = 1, p = 2 ⇒ K = 2, so
        // A_k is 1×1 = [a] and M = a·I_2. tr(S_β⁻¹ M) = a·tr(S_β⁻¹).
        let a_scalar = 0.75_f64;
        let mut trace = 0.0_f64;
        for col in 0..k {
            // (A_k ⊗ I_p) e_col = a_scalar · e_col for this M_k=1 block.
            let mut m_col = Array1::<f64>::zeros(k);
            m_col[col] = a_scalar;
            let z = cache
                .schur_inverse_apply(m_col.view())
                .expect("schur_inverse_apply");
            trace += z[col];
        }
        let trace_dense: f64 = a_scalar
            * (0..k)
                .map(|j| h_inv[[beta_off + j, beta_off + j]])
                .sum::<f64>();
        assert!(
            (trace - trace_dense).abs() < 1e-9,
            "Kron-block trace {trace} vs dense {trace_dense}"
        );

        // schur_inverse_block must reproduce a contiguous dense sub-block of
        // (H⁻¹)_ββ — both the full β-block and an interior single-coordinate
        // window — and be exactly symmetric.
        let full = cache
            .schur_inverse_block(0..k)
            .expect("dense Schur cache must support schur_inverse_block");
        assert_eq!(full.dim(), (k, k));
        for r in 0..k {
            for c in 0..k {
                let expected = h_inv[[beta_off + r, beta_off + c]];
                assert!(
                    (full[[r, c]] - expected).abs() < 1e-9,
                    "block[{r},{c}] {} vs dense {expected}",
                    full[[r, c]]
                );
                assert!(
                    (full[[r, c]] - full[[c, r]]).abs() < 1e-12,
                    "schur_inverse_block must be symmetric at [{r},{c}]"
                );
            }
        }
        let sub = cache
            .schur_inverse_block(1..k)
            .expect("interior block must be supported");
        assert_eq!(sub.dim(), (k - 1, k - 1));
        assert!(
            (sub[[0, 0]] - h_inv[[beta_off + 1, beta_off + 1]]).abs() < 1e-9,
            "interior block [1,1] {} vs dense {}",
            sub[[0, 0]],
            h_inv[[beta_off + 1, beta_off + 1]]
        );
        // Out-of-range block must error rather than panic.
        assert!(cache.schur_inverse_block(0..(k + 1)).is_err());
    }

    /// Evidence/log-det mode: a per-row `H_tt` that is PD but ill-conditioned
    /// (κ above the safe-Schur ceiling) is handled differently by the two
    /// solve paths. The default `direct()` path conditions each row to the
    /// safe-Schur κ ceiling; when that per-row conditioning is insufficient to
    /// keep the *dense Schur complement* PD (gam#845), the single-shot solve
    /// correctly reports a recoverable factorization error and the
    /// LM-escalating wrapper recovers it with a finite, well-conditioned step.
    ///
    /// `with_ill_conditioning_tolerated()` accepts the RAW (undamped) blocks.
    /// Its contract has two sides, pinned on two fixtures:
    ///   * row-PD but assembled-INDEFINITE H (strong coupling into near-null
    ///     t-directions) → honest refusal. Per-row PD does not imply bordered-
    ///     system PD, and an exact `log|H|` does not exist on the Cholesky
    ///     branch — fabricating one would corrupt the evidence.
    ///   * row κ ≈ 1e9 but assembled H genuinely PD (coupling subordinate to
    ///     the weak curvature) → a usable cache whose log-determinant equals
    ///     the exact dense `log|H|`, undistorted by any κ-ceiling ridge. This
    ///     is the SAE evidence path under a wide ARD α sweep.
    #[test]
    fn ill_conditioning_tolerated_returns_cache_with_exact_logdet() {
        let n = 2usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        // Barely-PD rows: second pivot ~1e-9 of the first ⇒ κ ≈ 1e9, above
        // the safe-Schur ceiling but genuinely PD (Cholesky succeeds).
        sys.rows[0].htt = array![[1.0_f64, 0.0], [0.0, 1e-9]];
        sys.rows[0].htbeta = array![[0.3_f64, 0.1], [0.05, 0.2]];
        sys.rows[1].htt = array![[2.0_f64, 0.0], [0.0, 2e-9]];
        sys.rows[1].htbeta = array![[0.2_f64, -0.1], [0.1, 0.15]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        sys.hbb = array![[5.0_f64, 0.3], [0.3, 4.0]];
        sys.gb = array![0.0_f64, 0.0];

        // factor_one_row conditions each barely-PD per-row block to the
        // safe-Schur κ ceiling (gam#578): the raw block fails the ceiling but
        // the ridge-lifted factor satisfies it. Verify the per-row contract
        // directly — this is what per-row conditioning genuinely guarantees.
        for i in 0..n {
            let factor = factor_one_row(&sys.rows[i], 0.0, d, i, false)
                .expect("barely-PD row must be conditioned, not rejected (gam#578)");
            let kappa = cholesky_factor_kappa_estimate(&factor);
            assert!(
                kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
                "conditioned per-row factor {i} must satisfy the safe-Schur κ ceiling; got κ={kappa:e}"
            );
        }

        // Per-row conditioning alone cannot keep the dense Schur complement PD
        // for these inputs (κ_ceiling × ‖H_tβ‖² ≫ ‖H_ββ‖, gam#845), so the
        // single-shot strict solve reports a recoverable factorization error
        // rather than a finite step.
        let single_shot =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &ArrowSolveOptions::direct());
        assert!(
            matches!(
                single_shot,
                Err(ArrowSchurError::SchurFactorFailed { .. })
                    | Err(ArrowSchurError::PerRowFactorIllConditioned { .. })
                    | Err(ArrowSchurError::PcgFailed { .. })
            ),
            "single-shot strict direct() cannot keep the dense Schur PD with per-row \
             conditioning alone; expected a recoverable factorization error, got {single_shot:?}"
        );

        // The LM-escalating wrapper is the correct recovery: a bounded number
        // of outer ridge escalations yields a finite, well-conditioned step.
        let (strict_dt, strict_db, strict_diag) =
            solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &ArrowSolveOptions::direct())
                .expect("LM escalation must recover the ill-conditioned strict solve (gam#845)");
        for v in strict_dt.iter().chain(strict_db.iter()) {
            assert!(v.is_finite(), "recovered strict step must be finite: {v}");
        }
        assert!(
            strict_diag.ridge_escalations <= DEFAULT_PROXIMAL_MAX_ATTEMPTS,
            "recovery must use a bounded number of outer ridge escalations; got {}",
            strict_diag.ridge_escalations
        );

        // Evidence mode accepts the RAW (undamped) blocks. For THIS system the
        // honest answer is refusal: each per-row `H_tt` is PD in isolation, but
        // the strong coupling into the near-null t-directions makes the
        // assembled bordered H indefinite (its true Schur complement has a
        // ≈ −7.5e6 leading pivot; the full spectrum has two negative
        // eigenvalues). An exact log|H| does not exist on the Cholesky branch,
        // and tolerating ill-CONDITIONING must never fabricate a determinant
        // for an in-DEFINITE system — the SchurFactorFailed refusal is the
        // contract, not a defect.
        let opts = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
        let tolerate_indefinite = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &opts);
        assert!(
            matches!(
                tolerate_indefinite,
                Err(ArrowSchurError::SchurFactorFailed { .. })
            ),
            "tolerate mode must refuse the indefinite assembled H rather than fabricate \
             a log-determinant; got {tolerate_indefinite:?}"
        );

        // The regime the tolerate flag exists for: per-row κ ≈ 1e9 (above the
        // safe-Schur ceiling, so the strict path would ridge-condition the row
        // and distort the determinant) yet the assembled H is genuinely PD
        // because the coupling into the near-null t-directions is subordinate
        // to their curvature (‖H_tβ row‖² ≲ λ_min(H_tt)·λ_min(H_ββ)). Evidence
        // mode must factor the RAW blocks and report the EXACT dense log|H|,
        // undistorted by any κ-ceiling ridge.
        let mut pd_sys = ArrowSchurSystem::new(n, d, k);
        pd_sys.rows[0].htt = array![[1.0_f64, 0.0], [0.0, 1e-9]];
        pd_sys.rows[0].htbeta = array![[0.3_f64, 0.1], [3e-6, 1e-6]];
        pd_sys.rows[1].htt = array![[2.0_f64, 0.0], [0.0, 2e-9]];
        pd_sys.rows[1].htbeta = array![[0.2_f64, -0.1], [2e-6, 4e-6]];
        for row in pd_sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        pd_sys.hbb = array![[5.0_f64, 0.3], [0.3, 4.0]];
        pd_sys.gb = array![0.0_f64, 0.0];

        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&pd_sys, 0.0, 0.0, &opts)
            .expect("tolerate mode must factor the ill-conditioned-but-PD system");

        // Cache log-determinant (Σ log|H_tt^i| + log|S_β|) must equal the exact
        // dense log|H|, regardless of conditioning — the whole point.
        let (log_det_tt, log_det_schur) = cache.arrow_log_det();
        let log_det_cache = log_det_tt + log_det_schur.expect("dense Schur factor present");

        // Dense reference: assemble H and take log|H| = 2 Σ log L_ii.
        let dim = n * d + k;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for r in 0..d {
                for c in 0..d {
                    h[[base + r, base + c]] = pd_sys.rows[i].htt[[r, c]];
                }
            }
            for r in 0..d {
                for c in 0..k {
                    let v = pd_sys.rows[i].htbeta[[r, c]];
                    h[[base + r, n * d + c]] = v;
                    h[[n * d + c, base + r]] = v;
                }
            }
        }
        for r in 0..k {
            for c in 0..k {
                h[[n * d + r, n * d + c]] = pd_sys.hbb[[r, c]];
            }
        }
        let lh = cholesky_lower(&h).expect("assembled bordered H must be SPD");
        let log_det_dense: f64 = 2.0 * (0..dim).map(|i| lh[[i, i]].ln()).sum::<f64>();

        assert!(
            (log_det_cache - log_det_dense).abs() < 1e-6,
            "tolerated-cache log|H| {log_det_cache} vs dense {log_det_dense}"
        );

        // Selected-inverse traces must still be available from the cache.
        let tdiag = cache
            .latent_block_inverse_diagonal()
            .expect("tolerated cache must support latent_block_inverse_diagonal");
        assert_eq!(tdiag.len(), n * d);
        assert!(tdiag.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn arrow_factor_slab_accessor_matches_array_blocks_bitwise() {
        let blocks = vec![
            array![[1.0_f64]],
            array![[2.0_f64, 0.0], [0.25, 3.0]],
            array![[4.0_f64, 0.0, 0.0], [0.5, 5.0, 0.0], [-0.25, 0.75, 6.0]],
        ];
        let slab = ArrowFactorSlab::from_blocks(blocks.clone());
        assert_eq!(slab.len(), blocks.len());
        for row in 0..blocks.len() {
            let view = slab.factor(row);
            assert_eq!(view.dim(), blocks[row].dim());
            for r in 0..blocks[row].nrows() {
                for c in 0..blocks[row].ncols() {
                    assert_eq!(view[[r, c]].to_bits(), blocks[row][[r, c]].to_bits());
                }
            }
        }
    }

    fn fixed_row_kernel_fixture<const D: usize>() -> (ArrowRowBlock, Array1<f64>) {
        let mut row = ArrowRowBlock::new(D, 0);
        for r in 0..D {
            for c in 0..D {
                row.htt[[r, c]] = if r == c {
                    4.0 + r as f64
                } else {
                    0.03125 * ((r + c + 1) as f64)
                };
            }
        }
        let rhs = Array1::from_iter((0..D).map(|i| 0.5 + i as f64 * 0.25));
        (row, rhs)
    }

    fn assert_fixed_row_kernels_match_dynamic<const D: usize>() -> usize {
        let (row, rhs) = fixed_row_kernel_fixture::<D>();
        let ridge = 0.125_f64;
        let fixed = factor_row_block_cholesky_fixed::<D>(&row, ridge).expect("fixed factor");
        let dynamic = factor_row_block_cholesky_dynamic(&row, ridge, D).expect("dynamic factor");
        for r in 0..D {
            for c in 0..D {
                assert_eq!(
                    fixed[[r, c]].to_bits(),
                    dynamic[[r, c]].to_bits(),
                    "factor mismatch at D={D} ({r},{c})"
                );
            }
        }

        let fixed_solve = cholesky_solve_vector_fixed::<D>(fixed.view(), rhs.view());
        let dynamic_solve = cholesky_solve_vector(dynamic.view(), rhs.view());
        for i in 0..D {
            assert_eq!(
                fixed_solve[i].to_bits(),
                dynamic_solve[i].to_bits(),
                "solve mismatch at D={D} index {i}"
            );
        }
        D
    }

    #[test]
    fn fixed_row_kernels_match_dynamic_path_bitwise() {
        let checked = assert_fixed_row_kernels_match_dynamic::<1>()
            + assert_fixed_row_kernels_match_dynamic::<2>()
            + assert_fixed_row_kernels_match_dynamic::<3>()
            + assert_fixed_row_kernels_match_dynamic::<4>();
        assert_eq!(checked, 10);
    }

    /// Build a small, well-conditioned dense Direct arrow system: `n` rows of
    /// `d×d` PD blocks, small `d×k` cross blocks, a diagonally-dominant `k×k`
    /// border. Used to exercise the #1017 production device-routing seam on the
    /// host (where the device declines, so the CPU path must answer unchanged).
    fn dense_direct_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(n, d, k);
        for (i, row) in sys.rows.iter_mut().enumerate() {
            for r in 0..d {
                for c in 0..d {
                    row.htt[[r, c]] = if r == c { 4.0 + (i % 3) as f64 } else { 0.1 };
                }
                row.gt[r] = 0.05 * ((i + r + 1) as f64).sin();
                for c in 0..k {
                    row.htbeta[[r, c]] = 0.01 * (((i + 1) * (c + 1)) as f64).cos();
                }
            }
        }
        for r in 0..k {
            sys.gb[r] = 0.02 * ((r + 1) as f64).cos();
            for c in 0..k {
                sys.hbb[[r, c]] = if r == c { 6.0 } else { 0.0 };
            }
        }
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    /// The #1017 work-based dispatch predicate must admit LLM/SAE shapes (few
    /// rows, wide border) and reject tiny shapes where launch latency wins.
    #[test]
    fn device_dispatch_predicate_gates_on_work_not_rows() {
        let policy = crate::gpu::policy::GpuDispatchPolicy::default();
        // Tiny: below the DEVICE_LOOP_MIN_P border floor → never on device.
        assert!(!policy.dense_hessian_work_target_is_gpu(300, 8));
        // LLM/SAE: 2000 rows × a few-thousand-wide border clears both the
        // min-p floor and the 2·n·p² flop threshold.
        assert!(policy.dense_hessian_work_target_is_gpu(2_000, 4_096));
    }

    /// #1017 Phase-1 call-site re-key: the live matvec-injection gate
    /// (`maybe_inject_gpu_schur_matvec`) now keys on the CG-amortised
    /// `reduced_schur_matvec_should_offload(rows, k, sys.d, cg_iters)` predicate
    /// rather than the dense-Direct `(rows, k)` floor. This asserts the predicate
    /// the gate consults — with the exact `cg_iters` the gate derives from the
    /// options (`pcg.max_iterations.min(trust_region.max_iterations)`) — fires for
    /// the SAE LLM shape (n~2000 rows × k~2048 border × d~8 frame depth) while
    /// staying off for tiny shapes where launch latency dominates. The gate's
    /// device-presence short-circuit (`GpuRuntime::global()?`) makes the helper
    /// itself return `None` on a CPU-only host, so the routing logic is asserted
    /// through the predicate it consults (the device==CPU 1e-10 numeric parity is
    /// asserted by the box harness).
    #[test]
    fn matvec_gate_engages_for_llm_shape_off_for_tiny() {
        let policy = crate::gpu::policy::GpuDispatchPolicy::default();
        // The cg_iters the live gate derives from default options is exactly the
        // budget the PCG loop launches with.
        let options = ArrowSolveOptions::inexact_pcg();
        let cg_iters = options
            .pcg
            .max_iterations
            .min(options.trust_region.max_iterations);
        assert!(cg_iters > 0);

        // SAE LLM shape: few row blocks, wide border, modest frame depth. The
        // dense-Direct `(rows, k)` floor that the gate used to consult ignores the
        // frame depth `d` and the CG amortisation — assert the NEW predicate the
        // re-keyed gate consults admits it.
        let (n_llm, k_llm, d_llm) = (2_000_usize, 2_048_usize, 8_usize);
        assert!(policy.reduced_schur_matvec_should_offload(n_llm, k_llm, d_llm, cg_iters));

        // Tiny shape: narrow border below the device-loop floor → the gate stays
        // off regardless of the CG budget (launch latency dominates).
        assert!(!policy.reduced_schur_matvec_should_offload(30, 8, 2, cg_iters));
        // CPU-canary `(300, 8)` shape from the dense floor's own tests: still off.
        assert!(!policy.reduced_schur_matvec_should_offload(300, 8, 4, cg_iters));
    }

    /// On a host without a CUDA device the production seam must decline (return
    /// `None`), so `solve_arrow_newton_step_core` runs the unchanged CPU path
    /// and the result equals the direct CPU artifacts solve bit-for-bit.
    #[test]
    fn device_seam_declines_without_gpu_and_matches_cpu() {
        if crate::gpu::runtime::GpuRuntime::global().is_some() {
            // On a CUDA host the device may legitimately serve the step; this
            // host-only invariant does not apply. The box harness asserts the
            // device==CPU 1e-10 parity instead.
            return;
        }
        let sys = dense_direct_system(6, 2, 4);
        let options = ArrowSolveOptions::direct();

        // The seam helpers both decline when no device is present.
        assert!(try_device_arrow_direct(&sys, 0.0, 0.0, &options).is_none());
        assert!(maybe_inject_gpu_schur_matvec(&sys, 0.0, 0.0, &options).is_none());

        // The public core entry therefore equals the direct CPU artifacts solve.
        let (dt_core, db_core, diag) =
            solve_arrow_newton_step_core(&sys, 0.0, 0.0, &options).expect("core solve");
        assert!(
            !diag.used_device_arrow,
            "no device present, so the solve must not be flagged device-served"
        );
        let artifacts =
            solve_arrow_newton_step_artifacts(&sys, 0.0, 0.0, &options).expect("artifacts solve");
        for (a, b) in dt_core.iter().zip(artifacts.delta_t.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "Δt must be bit-identical to CPU");
        }
        for (a, b) in db_core.iter().zip(artifacts.delta_beta.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "Δβ must be bit-identical to CPU");
        }
    }

    /// #1014: the streaming reduced solve under certified mixed precision must
    /// agree with the f64 solve to the backward-error certificate, and — the
    /// load-bearing invariant — the evidence log-determinant must be UNCHANGED
    /// (bit-for-bit) because it is read from the f64 reduced-Schur factor, never
    /// the f32 solve.
    #[test]
    fn streaming_mixed_precision_matches_f64_and_keeps_logdet_f64() {
        let sys = dense_direct_system(40, 3, 6);

        let f64_options = ArrowSolveOptions::direct().with_streaming_chunk_size(Some(8));
        let mp_options = f64_options
            .clone()
            .with_mixed_precision_policy(MixedPrecisionPolicy::certified());
        assert!(matches!(
            f64_options.mixed_precision,
            MixedPrecisionPolicy::Off
        ));

        let mut s_f64 = StreamingArrowSchur::from_system(&sys, 8);
        let (_, db_f64, _) = s_f64
            .solve(0.0, 0.0, &f64_options)
            .expect("f64 streaming solve");
        let mut s_mp = StreamingArrowSchur::from_system(&sys, 8);
        let (_, db_mp, _) = s_mp
            .solve(0.0, 0.0, &mp_options)
            .expect("mp streaming solve");

        // The mixed-precision Δβ matches the f64 Δβ to the certified tolerance.
        let mut max_abs = 0.0_f64;
        for (a, b) in db_f64.iter().zip(db_mp.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-7,
            "mixed-precision Δβ deviates from f64 by {max_abs:e}, above the certified tolerance"
        );

        // Evidence log-determinant: f64 regardless of the Δβ precision policy.
        let mut ld_f64 = StreamingArrowSchur::from_system(&sys, 8);
        let logdet_f64 = ld_f64
            .exact_arrow_log_det(0.0, 0.0, &f64_options)
            .expect("f64 logdet");
        let mut ld_mp = StreamingArrowSchur::from_system(&sys, 8);
        let logdet_mp = ld_mp
            .exact_arrow_log_det(0.0, 0.0, &mp_options)
            .expect("mp logdet");
        assert_eq!(
            logdet_f64.to_bits(),
            logdet_mp.to_bits(),
            "evidence log|H| must stay bit-for-bit f64 under the mixed-precision policy"
        );
    }

    /// The streaming dispatch turns mixed precision ON by default (#1014) but
    /// honors an explicit caller policy.
    #[test]
    fn streaming_mixed_precision_default_upgrades_only_off() {
        let off = ArrowSolveOptions::direct();
        assert!(matches!(
            off.with_streaming_mixed_precision_default().mixed_precision,
            MixedPrecisionPolicy::Certified { .. }
        ));
        let pinned =
            ArrowSolveOptions::direct().with_mixed_precision_policy(MixedPrecisionPolicy::Off);
        // An explicit Off is still upgraded (it is the inherited default), but a
        // caller that pinned Certified keeps its own parameters.
        let custom = ArrowSolveOptions::direct().with_mixed_precision_policy(
            MixedPrecisionPolicy::Certified {
                max_refinement_steps: 1,
                residual_relative_tolerance: 1e-6,
                kappa_unit_roundoff_margin: 0.25,
            },
        );
        match custom
            .with_streaming_mixed_precision_default()
            .mixed_precision
        {
            MixedPrecisionPolicy::Certified {
                max_refinement_steps,
                ..
            } => assert_eq!(max_refinement_steps, 1, "explicit policy preserved"),
            MixedPrecisionPolicy::Off => panic!("explicit Certified must not be downgraded"),
        }
        // `pinned` documents that Off is the upgrade trigger.
        assert!(matches!(pinned.mixed_precision, MixedPrecisionPolicy::Off));
    }

    // ----------------------------------------------------------------------
    // #1038 cross-row IBP Woodbury: value + log-determinant + adjoint must all
    // describe the SAME dense `H_full = H₀' + U D Uᵀ`. These checks build the
    // dense bordered `H_full` explicitly (the i≠j cross-row terms layered onto
    // the assembled self-term `H₀`) and assert the cache reproduces its
    // log-determinant, its full inverse, its latent inverse diagonal, and the
    // Newton step `H_full⁻¹(−g)` exactly.
    // ----------------------------------------------------------------------

    /// Build a small `(N, d, K_beta)` system with `R` IBP atom columns whose
    /// logit slots are the first `R` latent coords of every row. Returns the
    /// system (with the self term `d_k·z'_ik²` already on the logit diagonals,
    /// as the assembly writes it), the source, and the per-(row, atom) `z'_ik`.
    fn build_ibp_woodbury_fixture() -> (ArrowSchurSystem, IbpCrossRowSource, Vec<Vec<f64>>) {
        let n = 3usize;
        let d = 2usize;
        let k_beta = 2usize;
        let r = 2usize; // two atom columns, supported on logit slots 0 and 1.
        let mut sys = ArrowSchurSystem::new(n, d, k_beta);
        // Base (no-self) per-row latent blocks + cross-blocks + gradient.
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[0].gt = array![0.4_f64, -0.7];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[1].gt = array![-0.2_f64, 0.9];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        sys.rows[2].gt = array![1.1_f64, 0.3];
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.5_f64, -0.8];

        // IBP source: d_k coefficients (one positive, one negative — exercise the
        // indefinite-capacitance LU path) and z'_ik per (row, atom).
        let d_coef = array![0.6_f64, -0.35];
        let zprime = vec![
            vec![0.9_f64, 0.4], // row 0: z'_00, z'_01
            vec![-0.5, 0.8],    // row 1
            vec![0.7, -0.6],    // row 2
        ];
        let mut entries = Vec::new();
        for i in 0..n {
            for k in 0..r {
                // logit slot for atom k is latent coord k of row i.
                let g = sys.row_offsets[i] + k;
                entries.push((g, k, zprime[i][k]));
            }
        }
        // Write the self term d_k·z'_ik² onto the logit diagonals (as assembly does).
        for i in 0..n {
            for k in 0..r {
                sys.rows[i].htt[[k, k]] += d_coef[k] * zprime[i][k] * zprime[i][k];
            }
        }
        let source = IbpCrossRowSource {
            r,
            d: d_coef,
            entries,
        };
        (sys, source, zprime)
    }

    /// Assemble the dense bordered `H_full` (with the i≠j cross-row terms) from
    /// the self-term system + source.
    fn dense_h_full(
        sys: &ArrowSchurSystem,
        source: &IbpCrossRowSource,
        zprime: &[Vec<f64>],
    ) -> Array2<f64> {
        let n = sys.rows.len();
        let d = sys.d;
        let k_beta = sys.k;
        let dim = n * d + k_beta;
        let mut h = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            for rr in 0..d {
                for cc in 0..d {
                    h[[base + rr, base + cc]] = sys.rows[i].htt[[rr, cc]];
                }
                for cc in 0..k_beta {
                    let v = sys.rows[i].htbeta[[rr, cc]];
                    h[[base + rr, n * d + cc]] = v;
                    h[[n * d + cc, base + rr]] = v;
                }
            }
        }
        for rr in 0..k_beta {
            for cc in 0..k_beta {
                h[[n * d + rr, n * d + cc]] = sys.hbb[[rr, cc]];
            }
        }
        // Cross-row i≠j terms: H[g_i, g_j] += d_k·z'_ik·z'_jk (the self i=j part
        // is already on the diagonal via the assembled self term).
        for k in 0..source.r {
            let dk = source.d[k];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let gi = i * d + k;
                    let gj = j * d + k;
                    h[[gi, gj]] += dk * zprime[i][k] * zprime[j][k];
                }
            }
        }
        h
    }

    #[test]
    fn ibp_cross_row_woodbury_logdet_matches_dense() {
        let (mut sys, source, zprime) = build_ibp_woodbury_fixture();
        sys.set_ibp_cross_row_source(source.clone());
        let options = ArrowSolveOptions::direct();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("IBP Woodbury cache should factor");
        assert!(
            cache.cross_row_woodbury.is_some(),
            "the cache must carry the cross-row Woodbury"
        );

        let h_full = dense_h_full(&sys, &source, &zprime);
        let l = cholesky_lower(&h_full).expect("H_full must be SPD for this fixture");
        let mut dense_logdet = 0.0_f64;
        for i in 0..l.nrows() {
            dense_logdet += 2.0 * l[[i, i]].ln();
        }

        let (tt, schur) = cache.arrow_log_det();
        let cache_logdet = tt + schur.expect("direct mode has a Schur factor");
        assert!(
            (cache_logdet - dense_logdet).abs() < 1e-9,
            "cache log det H_full {cache_logdet} vs dense {dense_logdet}"
        );

        // The Woodbury correction is exactly log det H_full − log det H₀', where
        // the factored base `H₀' = H_full − U D Uᵀ` has the WHOLE rank-`R` update
        // removed — both the `i=j` self diagonal `d_k·z'_ik²` AND the `i≠j`
        // cross-row off-diagonals `d_k·z'_ik·z'_jk`. (The per-row latent blocks the
        // cache factors never carry cross-row coupling, so its base is exactly this
        // `H₀'`; subtracting only the self diagonal would leave the cross terms in
        // and compare the lemma correction against a different base.)
        let mut h0prime = h_full.clone();
        for k in 0..source.r {
            for i in 0..sys.rows.len() {
                let gi = i * sys.d + k;
                for j in 0..sys.rows.len() {
                    let gj = j * sys.d + k;
                    h0prime[[gi, gj]] -= source.d[k] * zprime[i][k] * zprime[j][k];
                }
            }
        }
        let l0 = cholesky_lower(&h0prime).expect("H₀' SPD");
        let mut logdet_h0prime = 0.0_f64;
        for i in 0..l0.nrows() {
            logdet_h0prime += 2.0 * l0[[i, i]].ln();
        }
        let correction = cache.cross_row_woodbury_log_det();
        assert!(
            (correction - (dense_logdet - logdet_h0prime)).abs() < 1e-9,
            "Woodbury log det correction {correction} vs (logdet H_full − logdet H₀') {}",
            dense_logdet - logdet_h0prime
        );
    }

    #[test]
    fn ibp_cross_row_woodbury_full_inverse_and_newton_match_dense() {
        let (mut sys, source, zprime) = build_ibp_woodbury_fixture();
        sys.set_ibp_cross_row_source(source.clone());
        let options = ArrowSolveOptions::direct();
        let (delta_t, delta_beta, cache) =
            solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
                .expect("IBP Woodbury cache should factor");

        let n = sys.rows.len();
        let d = sys.d;
        let k_beta = sys.k;
        let dim = n * d + k_beta;
        let h_full = dense_h_full(&sys, &source, &zprime);
        let l = cholesky_lower(&h_full).expect("H_full SPD");

        // (a) Newton step Δ = −H_full⁻¹ g.
        let mut g = Array1::<f64>::zeros(dim);
        for i in 0..n {
            for j in 0..d {
                g[i * d + j] = sys.rows[i].gt[j];
            }
        }
        for c in 0..k_beta {
            g[n * d + c] = sys.gb[c];
        }
        let dense_step = cholesky_solve_vector(&l, &g); // H_full⁻¹ g
        for idx in 0..n * d {
            assert!(
                (delta_t[idx] + dense_step[idx]).abs() < 1e-9,
                "Δt[{idx}] {} vs −H_full⁻¹g {}",
                delta_t[idx],
                -dense_step[idx]
            );
        }
        for c in 0..k_beta {
            assert!(
                (delta_beta[c] + dense_step[n * d + c]).abs() < 1e-9,
                "Δβ[{c}] {} vs −H_full⁻¹g {}",
                delta_beta[c],
                -dense_step[n * d + c]
            );
        }

        // (b) full_inverse_apply on an arbitrary RHS = dense H_full⁻¹ w.
        let mut w_full = Array1::<f64>::zeros(dim);
        for (idx, v) in w_full.iter_mut().enumerate() {
            *v = 0.25 + 0.13 * (idx as f64) * (if idx % 2 == 0 { 1.0 } else { -1.0 });
        }
        let dense_u = cholesky_solve_vector(&l, &w_full);
        let (u_t, u_beta) = cache
            .full_inverse_apply(
                w_full.slice(ndarray::s![..n * d]),
                w_full.slice(ndarray::s![n * d..]),
            )
            .expect("full_inverse_apply on the Woodbury cache");
        for idx in 0..n * d {
            assert!(
                (u_t[idx] - dense_u[idx]).abs() < 1e-9,
                "H_full⁻¹w t[{idx}] {} vs dense {}",
                u_t[idx],
                dense_u[idx]
            );
        }
        for c in 0..k_beta {
            assert!(
                (u_beta[c] - dense_u[n * d + c]).abs() < 1e-9,
                "H_full⁻¹w beta[{c}] {} vs dense {}",
                u_beta[c],
                dense_u[n * d + c]
            );
        }

        // (c) latent_block_inverse_diagonal = diag((H_full⁻¹)_tt).
        let mut h_full_inv = Array2::<f64>::zeros((dim, dim));
        let mut e = Array1::<f64>::zeros(dim);
        for col in 0..dim {
            e.fill(0.0);
            e[col] = 1.0;
            let sol = cholesky_solve_vector(&l, &e);
            for rrow in 0..dim {
                h_full_inv[[rrow, col]] = sol[rrow];
            }
        }
        let diag = cache
            .latent_block_inverse_diagonal()
            .expect("latent_block_inverse_diagonal on the Woodbury cache");
        for idx in 0..n * d {
            assert!(
                (diag[idx] - h_full_inv[[idx, idx]]).abs() < 1e-9,
                "diag (H_full⁻¹)_tt[{idx}] {} vs dense {}",
                diag[idx],
                h_full_inv[[idx, idx]]
            );
        }
    }

    /// Value↔gradient consistency: the log-determinant the evidence reports and
    /// the Hessian the Newton/adjoint solve inverts must be the SAME `H_full`.
    /// A finite-difference of `½ log det H(ε)` along the gradient direction
    /// `g = ∂(½ wᵀ H_full w)/∂...` is overkill here; instead we verify the
    /// cross-row correction's own internal coherence: removing the source must
    /// recover the bare-`H₀'` log-determinant (no double-count), and the
    /// rank-`R` capacitance LU determinant matches the dense ratio. (Covered by
    /// `ibp_cross_row_woodbury_logdet_matches_dense`.) Here we additionally
    /// check that a system WITHOUT the source yields no Woodbury carrier and an
    /// unchanged (bare) log-determinant, so the path is a strict no-op off-IBP.
    #[test]
    fn ibp_cross_row_woodbury_absent_is_strict_noop() {
        let (sys, _source, zprime) = build_ibp_woodbury_fixture();
        // No set_ibp_cross_row_source call: the source is absent.
        let options = ArrowSolveOptions::direct();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("bare cache should factor");
        assert!(
            cache.cross_row_woodbury.is_none(),
            "no source ⇒ no Woodbury carrier"
        );
        assert_eq!(
            cache.cross_row_woodbury_log_det(),
            0.0,
            "absent Woodbury contributes exactly zero to the log-determinant"
        );
        // The bare cache's log det is that of the assembled self-term `H₀` (the
        // fixture's rows already carry the self term), with no cross-row terms.
        let dim = sys.rows.len() * sys.d + sys.k;
        let mut h0 = Array2::<f64>::zeros((dim, dim));
        let n = sys.rows.len();
        let d = sys.d;
        for i in 0..n {
            let base = i * d;
            for rr in 0..d {
                for cc in 0..d {
                    h0[[base + rr, base + cc]] = sys.rows[i].htt[[rr, cc]];
                }
                for cc in 0..sys.k {
                    let v = sys.rows[i].htbeta[[rr, cc]];
                    h0[[base + rr, n * d + cc]] = v;
                    h0[[n * d + cc, base + rr]] = v;
                }
            }
        }
        for rr in 0..sys.k {
            for cc in 0..sys.k {
                h0[[n * d + rr, n * d + cc]] = sys.hbb[[rr, cc]];
            }
        }
        let l = cholesky_lower(&h0).expect("H₀ SPD");
        let mut dense_logdet = 0.0_f64;
        for i in 0..l.nrows() {
            dense_logdet += 2.0 * l[[i, i]].ln();
        }
        let (tt, schur) = cache.arrow_log_det();
        let cache_logdet = tt + schur.expect("direct Schur");
        assert!(
            (cache_logdet - dense_logdet).abs() < 1e-9,
            "bare cache log det {cache_logdet} vs dense H₀ {dense_logdet}"
        );
        // `zprime` is part of the shared fixture; touch it so the helper's third
        // return stays meaningful for readers and is not dead in this arm.
        assert_eq!(zprime.len(), n);
    }

    /// The streaming log-det path must REFUSE an IBP-active system rather than
    /// silently drop the cross-row correction (a value↔gradient desync).
    #[test]
    fn ibp_cross_row_streaming_logdet_refuses() {
        let (mut sys, source, _zprime) = build_ibp_woodbury_fixture();
        sys.set_ibp_cross_row_source(source);
        let mut streaming = StreamingArrowSchur::from_system(&sys, 2);
        let options = ArrowSolveOptions::direct();
        let err = streaming.reduced_schur_and_log_det_tt(0.0, 0.0, &options);
        assert!(
            err.is_err(),
            "streaming arrow log-det must refuse an IBP-active system"
        );
    }

    /// Build a dense-`htbeta` arrow system at an SAE-LLM-flavoured shape
    /// (`n` row blocks × `d` latent coords × wide border `k`), with
    /// deterministic well-conditioned per-row blocks and cross-blocks. This is
    /// the shape the reduced-Schur matvec (#1017) walks O(cg_iters) times.
    fn dense_arrow_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(n, d, k);
        // Deterministic diagonally-dominant per-row H_tt and modest H_tβ.
        for i in 0..n {
            let mut htt = Array2::<f64>::zeros((d, d));
            for r in 0..d {
                for c in 0..d {
                    let s = ((i + 1) * (r + 2) * (c + 3)) as f64;
                    htt[[r, c]] = if r == c {
                        4.0 + (s % 7.0)
                    } else {
                        0.1 * ((s % 5.0) - 2.0)
                    };
                }
            }
            // Symmetrize and ensure SPD by diagonal dominance.
            let mut sym = &htt + &htt.t();
            for r in 0..d {
                sym[[r, r]] = sym[[r, r]].abs() + (d as f64) + 2.0;
            }
            sys.rows[i].htt = sym;
            let mut htb = Array2::<f64>::zeros((d, k));
            for r in 0..d {
                for c in 0..k {
                    let s = ((i + 1) * (r + 1) + 3 * (c + 1)) as f64;
                    htb[[r, c]] = 0.05 * ((s % 11.0) - 5.0);
                }
            }
            sys.rows[i].htbeta = htb;
            sys.rows[i].gt = Array1::<f64>::zeros(d);
        }
        // SPD H_ββ: diagonally dominant.
        let mut hbb = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                let s = ((r + 1) * (c + 1)) as f64;
                hbb[[r, c]] = if r == c {
                    (k as f64) + 6.0 + (s % 3.0)
                } else {
                    0.02 * ((s % 7.0) - 3.0)
                };
            }
        }
        sys.hbb = hbb;
        sys.gb = Array1::<f64>::zeros(k);
        sys
    }

    /// Sequential reference for the reduced-Schur matvec: the exact per-row fold
    /// the `schur_matvec` sequential branch performs (used to compare the
    /// parallel path against). Mirrors the production routine's H_ββ + ridge
    /// prologue, then the per-row point-elimination subtraction in row order.
    fn schur_matvec_sequential_ref<B: BatchedBlockSolver>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        x: &Array1<f64>,
        backend: &B,
    ) -> Array1<f64> {
        let k = sys.k;
        let mut out = Array1::<f64>::zeros(k);
        {
            let xs = x.as_slice().unwrap();
            let os = out.as_slice_mut().unwrap();
            sys.penalty_matvec_add(xs, os);
            for a in 0..k {
                os[a] += ridge_beta * xs[a];
            }
        }
        let mut local = Array1::<f64>::zeros(sys.d);
        let mut neg = Array1::<f64>::zeros(k);
        for i in 0..sys.rows.len() {
            neg.fill(0.0);
            schur_matvec_row_into(sys, htt_factors, x, backend, i, &mut local, &mut neg);
            for a in 0..k {
                out[a] -= neg[a];
            }
        }
        out
    }

    /// The parallel reduced-Schur matvec (rows ≥ `SCHUR_MATVEC_PARALLEL_ROW_MIN`)
    /// must be (a) DETERMINISTIC run-to-run — bit-identical across repeated
    /// invocations regardless of thread scheduling, the #1017 verification gate
    /// that the criterion ranking across candidates cannot move; and (b)
    /// numerically equal to the sequential per-row fold up to the ULP-level
    /// reordering of an otherwise-identical sum (the chunk-partial reduction
    /// reassociates the same row contributions, so it agrees with the per-row
    /// fold to a tight relative tolerance, not bit-for-bit).
    #[test]
    fn parallel_schur_matvec_deterministic_and_matches_sequential() {
        let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // trips the parallel path
        let d = 6usize;
        let k = 96usize;
        let sys = dense_arrow_system(n, d, k);
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, d, false)
            .expect("SPD per-row blocks must factor");
        let ridge_beta = 1e-6;
        let x = Array1::from_iter((0..k).map(|a| 0.3 * (a as f64).sin() - 0.1));

        // (a) Determinism: two independent invocations of the live (parallel)
        // path must be bit-identical.
        let mut out_a = Array1::<f64>::zeros(k);
        let mut out_b = Array1::<f64>::zeros(k);
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_a,
            &backend,
            None,
        );
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_b,
            &backend,
            None,
        );
        for a in 0..k {
            assert_eq!(
                out_a[a].to_bits(),
                out_b[a].to_bits(),
                "parallel Schur matvec must be deterministic run-to-run at index {a}"
            );
        }

        // (b) Equivalence with the sequential per-row fold within ULP-scale
        // reassociation error.
        let out_seq = schur_matvec_sequential_ref(&sys, &htt_factors, ridge_beta, &x, &backend);
        let scale = out_seq
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        for a in 0..k {
            let rel = (out_a[a] - out_seq[a]).abs() / scale;
            assert!(
                rel < 1e-12,
                "parallel vs sequential Schur matvec must agree to reassociation error \
                 at index {a}: {} vs {} (rel {rel:e})",
                out_a[a],
                out_seq[a]
            );
        }
    }

    /// The dense `H_ββ` penalty-prologue GEMV parallelized over output rows at
    /// the wide SAE border (`k ≥ SCHUR_PROLOGUE_PARALLEL_K_MIN`, #1017) must be
    /// **bit-identical** to the serial prologue — unlike the per-row reduction,
    /// the GEMV carries no reassociation: each `y[a] = Σ_b hbb[a,b]·x[b] + ridge·x[a]`
    /// is computed in its entirety by one thread in the same `b` order whether
    /// one core or many run, so distributing the `a`-rows across threads cannot
    /// move a single bit. This pins the determinism/parity gate exactly at the
    /// border width where the prologue stops being serial.
    #[test]
    fn parallel_penalty_prologue_bit_identical_to_serial() {
        let k = 576usize; // ≥ SCHUR_PROLOGUE_PARALLEL_K_MIN: trips the parallel GEMV
        assert!(
            k >= SCHUR_PROLOGUE_PARALLEL_K_MIN,
            "test border must exceed the prologue parallel threshold"
        );
        let d = 4usize;
        // A handful of rows: small enough that the per-row loop stays sequential
        // (rows < SCHUR_MATVEC_PARALLEL_ROW_MIN), isolating the prologue as the
        // only parallelized stage so the bit-parity claim is about it alone.
        let n = 8usize;
        assert!(n < SCHUR_MATVEC_PARALLEL_ROW_MIN);
        let sys = dense_arrow_system(n, d, k);
        let ridge = 7.5e-3;
        let x = Array1::from_iter((0..k).map(|a| 0.4 * (a as f64 * 0.31).cos() - 0.17));
        let xs = x.as_slice().unwrap();

        // Serial reference: penalty_matvec_add + ridge axpy into a zeroed buffer.
        let mut serial = vec![0.0_f64; k];
        sys.penalty_matvec_add(xs, &mut serial);
        for a in 0..k {
            serial[a] += ridge * xs[a];
        }

        // Parallel prologue (parallel=true engages the rayon dense GEMV at this k).
        let mut par = vec![0.0_f64; k];
        sys.penalty_ridge_prologue_into(xs, ridge, &mut par, true);
        // And the serial branch of the same fn (parallel=false) for completeness.
        let mut ser_branch = vec![0.0_f64; k];
        sys.penalty_ridge_prologue_into(xs, ridge, &mut ser_branch, false);

        for a in 0..k {
            assert_eq!(
                par[a].to_bits(),
                serial[a].to_bits(),
                "parallel penalty prologue must be bit-identical to serial at index {a}"
            );
            assert_eq!(
                ser_branch[a].to_bits(),
                serial[a].to_bits(),
                "serial prologue branch must match the reference at index {a}"
            );
        }
    }

    /// Wall-clock benchmark of the reduced-Schur matvec at an SAE-LLM-flavoured
    /// shape (#1017): sequential per-row fold vs the rayon-parallel chunked path.
    /// Runs as an ordinary test (the ban gate forbids `#[ignore]`), so the shape
    /// and call count are sized to stay fast in a debug CI build while still
    /// tripping the parallel path. Run with `--release --nocapture` on a quiet
    /// multicore box to read the per-call wall-clock and the parallel speedup at
    /// the inner-CG matvec cost the production InexactPCG loop pays O(cg_iters)
    /// times:
    ///
    /// ```text
    /// cargo test -p gam --lib --release \
    ///   solver::arrow_schur::tests::bench_reduced_schur_matvec_parallel_speedup \
    ///   -- --nocapture
    /// ```
    #[test]
    fn bench_reduced_schur_matvec_parallel_speedup() {
        // SAE-arm-flavoured shape from the issue: many row blocks, wide border
        // k, modest frame depth d. Sized so the debug build stays quick.
        let n = 1500usize;
        let d = 6usize;
        let k = 1024usize;
        let sys = dense_arrow_system(n, d, k);
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, d, false)
            .expect("SPD per-row blocks must factor");
        let ridge_beta = 1e-6;
        let x = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.017).sin() - 0.1));

        // A representative inner-CG budget: the matvec is paid once per CG iter.
        let calls = 30usize;
        let mut sink = 0.0_f64;

        // Warm up (factor caches, allocator, rayon pool) before timing.
        let warm = schur_matvec_sequential_ref(&sys, &htt_factors, ridge_beta, &x, &backend);
        sink += warm[0];

        let t_seq = std::time::Instant::now();
        for _ in 0..calls {
            let out = schur_matvec_sequential_ref(&sys, &htt_factors, ridge_beta, &x, &backend);
            sink += out[0];
        }
        let seq_elapsed = t_seq.elapsed();

        let mut out_par = Array1::<f64>::zeros(k);
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_par,
            &backend,
            None,
        ); // warm
        sink += out_par[0];
        let t_par = std::time::Instant::now();
        for _ in 0..calls {
            schur_matvec(
                &sys,
                &htt_factors,
                ridge_beta,
                &x,
                &mut out_par,
                &backend,
                None,
            );
            sink += out_par[0];
        }
        let par_elapsed = t_par.elapsed();

        let seq_per = seq_elapsed.as_secs_f64() / calls as f64;
        let par_per = par_elapsed.as_secs_f64() / calls as f64;
        let speedup = seq_per / par_per;
        println!(
            "[#1017 reduced-Schur matvec, n={n} d={d} k={k}, {calls} calls, \
             {} rayon threads]\n  sequential: {:.3} ms/call\n  parallel:   {:.3} ms/call\n  \
             speedup:    {:.2}x  (sink {:.3e})",
            rayon::current_num_threads(),
            seq_per * 1e3,
            par_per * 1e3,
            speedup,
            sink,
        );
        // Loose floor so a single-core or heavily-loaded box does not flap the
        // benchmark; the real signal is the printed numbers.
        assert!(par_per > 0.0 && seq_per > 0.0, "timings must be positive");
    }

    /// Build an SAE-structured arrow system exercising the residency path: per
    /// row a `q×q` SPD `H_tt`, a `q×p` local Jacobian `L_i`, and `m_i` active
    /// atoms over `n_atoms` decoder blocks of width `p` (border `k = n_atoms·p`).
    /// Installs BOTH the matrix-free Kronecker cross-block operator (the generic
    /// matvec path: `H_tβ = L_i P_i`) AND the matching `DeviceSaePcgData` (the
    /// residency path), so the two routes see the identical operator.
    fn sae_structured_system(
        n: usize,
        q: usize,
        p: usize,
        n_atoms: usize,
        m_active: usize,
    ) -> (ArrowSchurSystem, Vec<Vec<(usize, f64)>>, Vec<Vec<f64>>) {
        let k = n_atoms * p;
        let mut sys = ArrowSchurSystem::new(n, q, k);
        let mut a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut local_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            // SPD H_tt: diagonally dominant.
            let mut htt = Array2::<f64>::zeros((q, q));
            for r in 0..q {
                for c in 0..q {
                    let s = ((i + 1) * (r + 2) * (c + 3)) as f64;
                    htt[[r, c]] = 0.1 * ((s % 5.0) - 2.0);
                }
            }
            let mut sym = &htt + &htt.t();
            for r in 0..q {
                sym[[r, r]] = sym[[r, r]].abs() + (q as f64) + 3.0;
            }
            sys.rows[i].htt = sym;
            sys.rows[i].gt = Array1::<f64>::zeros(q);
            // L_i (q×p), row-major.
            let mut jac = vec![0.0_f64; q * p];
            for c in 0..q {
                for j in 0..p {
                    let s = ((i + 1) + 2 * (c + 1) + 3 * (j + 1)) as f64;
                    jac[c * p + j] = 0.1 * ((s % 7.0) - 3.0);
                }
            }
            local_jac.push(jac);
            // m_active atoms per row, deterministic spread over n_atoms.
            let mut support = Vec::with_capacity(m_active);
            for s in 0..m_active {
                let atom = ((i * 3 + s * 5) % n_atoms).min(n_atoms - 1);
                let phi = 0.5 + 0.25 * (((i + s) % 4) as f64);
                support.push((atom * p, phi));
            }
            a_phi.push(support);
        }
        // SPD H_ββ.
        let mut hbb = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            hbb[[r, r]] = (k as f64) + 4.0;
        }
        sys.hbb = hbb;
        sys.gb = Array1::<f64>::zeros(k);
        // Install the matrix-free Kronecker operator (H_tβ = L_i · P_i): forward
        // gathers active atoms into a length-p vector then applies L_i; transpose
        // is the exact adjoint. Mirrors src/terms/sae_manifold.rs:6028.
        let a_phi_f = a_phi.clone();
        let jac_f = local_jac.clone();
        let a_phi_t = a_phi.clone();
        let jac_t = local_jac.clone();
        let p_f = p;
        sys.set_row_htbeta_operator(
            move |row, x, out| {
                let mut u_p = vec![0.0_f64; p_f];
                for &(base, phi) in &a_phi_f[row] {
                    for j in 0..p_f {
                        u_p[j] += phi * x[base + j];
                    }
                }
                let jac = &jac_f[row];
                let qi = jac.len() / p_f;
                for c in 0..qi {
                    let mut acc = 0.0;
                    for j in 0..p_f {
                        acc += jac[c * p_f + j] * u_p[j];
                    }
                    out[c] = acc;
                }
            },
            move |row, v, out| {
                let jac = &jac_t[row];
                let qi = jac.len() / p_f;
                let mut u_p = vec![0.0_f64; p_f];
                for c in 0..qi {
                    let vc = v[c];
                    for j in 0..p_f {
                        u_p[j] += jac[c * p_f + j] * vc;
                    }
                }
                for &(base, phi) in &a_phi_t[row] {
                    for j in 0..p_f {
                        out[base + j] += phi * u_p[j];
                    }
                }
            },
        );
        sys.set_device_sae_pcg_data(DeviceSaePcgData {
            p,
            beta_dim: k,
            a_phi: a_phi.clone(),
            local_jac: local_jac.clone(),
            smooth_blocks: Vec::new(),
            sparse_g_blocks: Vec::new(),
        });
        (sys, a_phi, local_jac)
    }

    /// The CPU-resident SAE reduced-Schur matvec (#1017) must compute the SAME
    /// `S·x` as the generic per-row `apply → solve → transpose` path, up to f64
    /// reassociation. This is the residency correctness gate: a resident matvec
    /// that changed the reduced operator would change the Newton step and the
    /// criterion ranking — a correctness regression, not a speedup.
    #[test]
    fn resident_sae_matvec_matches_generic() {
        let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 96; // trips the parallel path
        let q = 4usize;
        let p = 6usize;
        let n_atoms = 32usize;
        let m_active = 5usize;
        let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
        let k = sys.k;
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, q, false)
            .expect("SPD per-row blocks must factor");
        let ridge_beta = 1e-6;
        let x = Array1::from_iter((0..k).map(|a| 0.2 * ((a as f64) * 0.013).cos() - 0.05));

        // Generic path (no resident operator).
        let mut out_generic = Array1::<f64>::zeros(k);
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_generic,
            &backend,
            None,
        );

        // Resident path: stage G_i once, then matvec.
        let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
            .expect("SAE structure must yield a resident operator");
        let mut out_resident = Array1::<f64>::zeros(k);
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_resident,
            &backend,
            Some(&resident),
        );

        let scale = out_generic
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        for a in 0..k {
            let rel = (out_resident[a] - out_generic[a]).abs() / scale;
            assert!(
                rel < 1e-10,
                "resident vs generic SAE Schur matvec must agree at index {a}: \
                 {} vs {} (rel {rel:e})",
                out_resident[a],
                out_generic[a]
            );
        }

        // Determinism: rebuilding + re-applying is bit-identical run-to-run.
        let resident2 = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend).unwrap();
        let mut out_resident2 = Array1::<f64>::zeros(k);
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_resident2,
            &backend,
            Some(&resident2),
        );
        for a in 0..k {
            assert_eq!(
                out_resident[a].to_bits(),
                out_resident2[a].to_bits(),
                "resident SAE matvec must be deterministic run-to-run at index {a}"
            );
        }
    }

    /// The #1017 SAE-resident scalar Jacobi (built from the staged `(L_i, Y_i)`
    /// factors in one support-sparse pass) must produce the SAME reduced-Schur
    /// diagonal — hence the SAME `BlockFactor::Scalar` inverses — as the generic
    /// per-column probe-and-solve `build_scalar_jacobi`. A diverging
    /// preconditioner would change the PCG iterate and the criterion ranking.
    #[test]
    fn resident_scalar_jacobi_matches_generic() {
        let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64;
        let q = 4usize;
        let p = 5usize;
        let n_atoms = 20usize;
        let m_active = 4usize;
        let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, q, false)
            .expect("SPD per-row blocks must factor");
        let ridge_beta = 1e-6;

        let generic = JacobiPreconditioner::build_scalar_jacobi(&sys, &htt_factors, ridge_beta, &backend)
            .expect("generic scalar Jacobi must build");
        let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
            .expect("SAE structure must yield a resident operator");
        let resident_jac =
            JacobiPreconditioner::build_scalar_jacobi_resident(&sys, ridge_beta, &resident)
                .expect("resident scalar Jacobi must build");

        // Probe both preconditioners with the same residual and compare the
        // applied (diagonal-scaled) output: identical diagonals ⇒ identical apply.
        let k = sys.k;
        let r = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.021).sin() + 0.07));
        let out_generic = generic.apply(&r);
        let out_resident = resident_jac.apply(&r);
        let scale = out_generic
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        for a in 0..k {
            let rel = (out_resident[a] - out_generic[a]).abs() / scale;
            assert!(
                rel < 1e-9,
                "resident vs generic SAE scalar Jacobi must agree at index {a}: \
                 {} vs {} (rel {rel:e})",
                out_resident[a],
                out_generic[a]
            );
        }
    }

    /// The factored residency (storing `(L_i, Y_i)` and applying `G_i v =
    /// L_iᵀ(Y_i v)`) must reproduce the dense `p×p` block `G_i = L_iᵀ Y_i`
    /// exactly — this is the #1017 memory/compute win (`O(n·di·p)` vs `O(n·p²)`)
    /// and must not perturb the operator. Asserts, per row, that the factored
    /// `row_into` applied to a unit-support probe equals the explicit dense
    /// `G_i · (P_i x)` to rel < 1e-10.
    #[test]
    fn factored_residency_matches_dense_g_block() {
        let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 40;
        let q = 3usize;
        let p = 7usize;
        let n_atoms = 24usize;
        let m_active = 4usize;
        let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, q, false)
            .expect("SPD per-row blocks must factor");
        let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
            .expect("SAE structure must yield a resident operator");

        for row in 0..n {
            let rf = &resident.rows[row];
            if rf.di == 0 {
                continue;
            }
            let di = rf.di;
            // Reconstruct the dense block G_i = L_iᵀ Y_i (p×p) from the stored
            // factors and check the factored GEMV chain against a direct G_i·g.
            let l = ArrayView2::from_shape((di, p), &rf.l).unwrap();
            let y = ArrayView2::from_shape((di, p), &rf.y).unwrap();
            let g_dense = l.t().dot(&y); // p×p

            // A non-trivial gather vector g (length p).
            let g_vec: Vec<f64> = (0..p)
                .map(|j| 0.4 * ((row + j) as f64 * 0.11).sin() - 0.07)
                .collect();
            // Dense reference: prod_ref = G_i · g.
            let mut prod_ref = vec![0.0_f64; p];
            for r in 0..p {
                let mut s = 0.0;
                for c in 0..p {
                    s += g_dense[(r, c)] * g_vec[c];
                }
                prod_ref[r] = s;
            }
            // Factored chain: w = Y_i·g, prod = L_iᵀ·w.
            let mut w = vec![0.0_f64; di];
            for r in 0..di {
                let yrow = &rf.y[r * p..r * p + p];
                w[r] = (0..p).map(|c| yrow[c] * g_vec[c]).sum();
            }
            let mut prod = vec![0.0_f64; p];
            for r in 0..di {
                let lrow = &rf.l[r * p..r * p + p];
                for j in 0..p {
                    prod[j] += lrow[j] * w[r];
                }
            }
            let scale = prod_ref
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()))
                .max(1.0);
            for j in 0..p {
                let rel = (prod[j] - prod_ref[j]).abs() / scale;
                assert!(
                    rel < 1e-10,
                    "factored G_i apply must match dense G_i at row {row} idx {j}: \
                     {} vs {} (rel {rel:e})",
                    prod[j],
                    prod_ref[j]
                );
            }
        }
        // Storage check: the factored form keeps di·p (not p²) per row.
        let factored_entries: usize = resident.rows.iter().map(|r| r.l.len() + r.y.len()).sum();
        let dense_entries: usize = resident.rows.iter().filter(|r| r.di > 0).count() * p * p;
        assert!(
            factored_entries < dense_entries,
            "factored residency must store fewer entries than the dense p×p form \
             ({factored_entries} vs {dense_entries})"
        );
    }

    /// Wall-clock benchmark: generic per-row matvec vs the CPU-resident SAE
    /// matvec (#1017) at an SAE-flavoured shape, amortised over a representative
    /// CG-iteration count (the residency build is paid once, then N matvecs).
    /// Ordinary test (ban gate forbids `#[ignore]`); run `--release --nocapture`.
    #[test]
    fn bench_resident_sae_matvec_speedup() {
        // SAE shape: small per-row latent dim `q = di` (1–2 in production) and a
        // wider per-atom decoder block `p` — the regime where the factored
        // residency (`2·di·p` flops/row, `O(n·di·p)` memory) beats both the
        // generic per-iteration solve AND a dense `p×p` residency (`p²` /
        // `O(n·p²)`). Here di=2, p=64 ⇒ ~16× fewer matvec flops/row than dense.
        let n = 1500usize;
        let q = 2usize;
        let p = 64usize;
        let n_atoms = 32usize; // border k = n_atoms·p = 2048
        let m_active = 6usize;
        let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
        let k = sys.k;
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, q, false)
            .expect("SPD per-row blocks must factor");
        let ridge_beta = 1e-6;
        let x = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.017).sin() - 0.1));
        let cg_iters = 30usize;
        let mut sink = 0.0_f64;

        // Generic: matvec re-walks apply/solve/transpose every iteration.
        let mut out = Array1::<f64>::zeros(k);
        schur_matvec(&sys, &htt_factors, ridge_beta, &x, &mut out, &backend, None); // warm
        sink += out[0];
        let t_gen = std::time::Instant::now();
        for _ in 0..cg_iters {
            schur_matvec(&sys, &htt_factors, ridge_beta, &x, &mut out, &backend, None);
            sink += out[0];
        }
        let gen_elapsed = t_gen.elapsed();

        // Resident: stage once (timed into the total — honest amortisation),
        // then cg_iters cheap matvecs.
        let t_res = std::time::Instant::now();
        let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
            .expect("resident operator");
        let mut outr = Array1::<f64>::zeros(k);
        for _ in 0..cg_iters {
            schur_matvec(
                &sys,
                &htt_factors,
                ridge_beta,
                &x,
                &mut outr,
                &backend,
                Some(&resident),
            );
            sink += outr[0];
        }
        let res_elapsed = t_res.elapsed();

        let gen_total = gen_elapsed.as_secs_f64();
        let res_total = res_elapsed.as_secs_f64();
        // Residency footprint: factored `(L_i, Y_i)` = `2·di·p` f64/row vs the
        // dense `p×p` block = `p²` f64/row.
        let factored_f64: usize = resident.rows.iter().map(|r| r.l.len() + r.y.len()).sum();
        let dense_f64: usize = resident.rows.iter().filter(|r| r.di > 0).count() * p * p;
        println!(
            "[#1017 SAE resident matvec, n={n} q={q} p={p} k={k} m={m_active}, \
             {cg_iters} CG matvecs incl. 1 residency build, {} rayon threads]\n  \
             generic:  {:.3} ms total ({:.3} ms/matvec)\n  resident: {:.3} ms total \
             (build + {cg_iters} matvecs)\n  speedup:  {:.2}x  (sink {:.3e})\n  \
             residency mem: factored {:.2} MiB vs dense p×p {:.2} MiB ({:.1}× smaller)",
            rayon::current_num_threads(),
            gen_total * 1e3,
            gen_total / cg_iters as f64 * 1e3,
            res_total * 1e3,
            gen_total / res_total,
            sink,
            factored_f64 as f64 * 8.0 / (1024.0 * 1024.0),
            dense_f64 as f64 * 8.0 / (1024.0 * 1024.0),
            dense_f64 as f64 / factored_f64.max(1) as f64,
        );
        assert!(
            gen_total > 0.0 && res_total > 0.0,
            "timings must be positive"
        );
    }
}
