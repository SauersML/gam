//! The reduced `K x K` shared-system solve: dense Schur assembly (direct and
//! square-root BA), the Schur matvec, the Jacobi/cluster/Schwarz
//! preconditioners, Steihaug-PCG, and the [`ArrowSchurError`] type.

use super::*;

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
pub(crate) fn tile_schur_partial<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    ordinal: usize,
    range: Range<usize>,
) -> Result<Array2<f64>, ArrowSchurError> {
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
        )?;
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
            return Ok(product.mapv(|v| -v));
        }
    }

    // CPU fallback: exact per-row block_gemm_subtract into a zero-seeded partial.
    let mut partial = Array2::<f64>::zeros((k, k));
    for (left, right) in &factors {
        backend.block_gemm_subtract(&mut partial, left, right);
    }
    Ok(partial)
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
pub(crate) fn reduce_row_schur_contributions<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
) -> Result<(), ArrowSchurError> {
    let n = sys.rows.len();
    let k = sys.k;

    let tiles = crate::gpu::device_runtime::GpuRuntime::global()
        .map(|rt| crate::gpu::pool::balanced_partition(rt, n))
        .filter(|tiles| tiles.len() > 1);

    let Some(tiles) = tiles else {
        // Single-device / CPU. The per-row contributions `-Σ_i leftᵀ·right` fold
        // into the `k×k` `schur` independently — the same dense-assembly axis the
        // multi-GPU tile path partitions, and the dense-Direct analog of the
        // per-row matvec / streaming `accumulate_chunk` loops already parallelized
        // for #1017. At the SAE Direct-solve shape (`n` in the thousands, wide
        // border `k`) this O(n·d·k²) reduction is the dense assembly's whole cost
        // and was the last serial CPU step on the dense-Schur build.
        //
        // Fan it across rayon over fixed row chunks: each chunk reduces its rows
        // (in row order) into a private zero-seeded `k×k` partial, then the
        // partials are folded into `schur` in CHUNK order. The per-chunk row order
        // and the inter-chunk fold order are both fixed independent of thread
        // scheduling, so the f64 reduction is **bit-identical run-to-run** (the
        // #1017 determinism gate). NOTE: bit-identical run-to-run does NOT make
        // it bit-identical to the in-place serial loop — the chunk-boundary
        // reassociation of the reduction sum is a genuine f64 departure (the
        // established equivalence `accumulate_chunk` / the per-row matvec operate
        // under, well inside the Newton solve's tolerance). It bounds candidate-
        // to-candidate drift to that reassociation margin, so the criterion
        // ranking is stable EXCEPT for candidates tying within the margin, where
        // the winner can flip; it is not an exact no-move guarantee (#1211). For
        // an exact-order guarantee, take the serial path. Stay in-place serial
        // below the row floor and when already inside a rayon worker (the topology
        // race fans candidates with `run_topology_race_parallel`) to avoid
        // nested-rayon oversubscription — the same guard the matvec uses.
        let n_rows = sys.rows.len();
        let parallel =
            n_rows >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let partials: Result<Vec<Array2<f64>>, ArrowSchurError> = (0..n_rows)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut partial = Array2::<f64>::zeros((k, k));
                    for i in idxs {
                        subtract_row_schur_contribution(
                            sys,
                            i,
                            &sys.rows[i],
                            htt_factors.factor(i),
                            backend,
                            kind,
                            &mut partial,
                        )?;
                    }
                    Ok(partial)
                })
                .collect();
            // Deterministic ordered fold: chunk partials hold `-Σ contribution`
            // over their rows, so `schur += partial` reproduces the serial
            // `schur -= Σ contribution` in fixed (chunk, a, b) order.
            for partial in &partials? {
                for a in 0..k {
                    for b in 0..k {
                        schur[[a, b]] += partial[[a, b]];
                    }
                }
            }
            return Ok(());
        }
        // Serial in-place reduction (original order) — bit-for-bit reference.
        for (i, row) in sys.rows.iter().enumerate() {
            subtract_row_schur_contribution(
                sys,
                i,
                row,
                htt_factors.factor(i),
                backend,
                kind,
                schur,
            )?;
        }
        return Ok(());
    };

    // Multi-GPU: one private `-Σ leftᵀ·right` partial per contiguous device
    // tile. Each tile runs on its own scoped worker thread that binds its
    // ordinal's context and issues a single stacked AᵀB GEMM on that device, so
    // the tiles' GEMMs overlap across the pool. Folding the partials back into
    // the H_ββ-seeded `schur` reproduces the serial reduction (up to inter-tile
    // reassociation).
    let partials: Result<Vec<Array2<f64>>, ArrowSchurError> = std::thread::scope(|scope| {
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
                        if let Some(ctx) = crate::gpu::device_runtime::cuda_context_for(ordinal) {
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
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| ArrowSchurError::SchurFactorFailed {
                        reason: "schur-reduction tile thread panicked".to_string(),
                    })?
            })
            .collect()
    });
    let partials = partials?;

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
    Ok(())
}

pub(crate) fn build_dense_schur_direct<B: BatchedBlockSolver + Sync>(
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
    )?;
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}

pub(crate) fn build_dense_schur_sqrt_ba<B: BatchedBlockSolver + Sync>(
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
    )?;
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
pub(crate) fn mixed_precision_reduced_beta(
    schur: &Array2<f64>,
    factor: &Array2<f64>,
    rhs: &Array1<f64>,
    options: &ArrowSolveOptions,
) -> Option<Array1<f64>> {
    let ArrowSolvePrecisionPolicy::CertifiedMixed {
        max_refinement_steps,
        residual_relative_tolerance,
        kappa_unit_roundoff_margin,
    } = options.solve_precision
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
pub(crate) fn matrix_inf_norm(a: &Array2<f64>) -> f64 {
    let mut max_row = 0.0_f64;
    for row in a.rows() {
        let s: f64 = row.iter().map(|v| v.abs()).sum();
        if s > max_row {
            max_row = s;
        }
    }
    max_row
}

pub(crate) fn solve_dense_reduced_system(
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
        if crate::gpu::device_runtime::GpuRuntime::is_available() {
            match crate::gpu::kernels::arrow_schur::solve_reduced_beta_pcg(
                &schur,
                rhs_beta,
                options.trust_region.max_iterations,
                options.trust_region.steihaug_relative_tolerance,
            ) {
                Ok(delta_beta) => return Ok(delta_beta),
                Err(crate::gpu::kernels::arrow_schur::ArrowSchurGpuFailure::Unavailable) => {}
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

pub(crate) fn step_inside_trust_region(
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
pub(crate) const SCHUR_MATVEC_PARALLEL_ROW_MIN: usize = 256;

/// Below this border width `k` the dense `H_ββ` penalty-prologue GEMV stays
/// sequential: parallelizing a `k×k` matvec only pays once `k²` is large enough
/// to dwarf the rayon fan-out, which for the arrow callers with narrow borders
/// it never is. At the SAE LLM border (`k` in the low thousands) the `O(k²)`
/// prologue is ≈4M flops/CG-iteration and was the serial Amdahl ceiling on the
/// otherwise per-row-parallel matvec (#1017), so it crosses this threshold and
/// fans out. 512 keeps the prologue serial for every non-SAE arrow system while
/// engaging it for the wide SAE/Qwen borders the issue targets.
pub(crate) const SCHUR_PROLOGUE_PARALLEL_K_MIN: usize = 512;

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
/// rows, so the factored apply is `2·support_i·p + 2·di·p` flops/row — the two
/// `di·p` GEMVs PLUS the `support_i·p` sparse gather (`P_i x`) and `support_i·p`
/// sparse scatter (`P_iᵀ prod`) — versus the dense `p²` block apply, and
/// `O(n·di·p)` memory (vs `O(n·p²)` ≈ 67 GB at the Qwen shape — the dense form
/// is OOM). For dense/full active support `support_i` can scale with the active
/// β-columns, so the gather/scatter term is NOT negligible and is counted here.
///
/// Numerically identical to the generic path up to floating-point reassociation
/// (it differentiates and accumulates the SAME quotient). It is deterministic
/// run-to-run and within the reassociation margin of the serial path, so the
/// criterion ranking across topology candidates is stable except for candidates
/// separated by less than that f64 margin, where reassociation can flip the
/// near-tie winner — it is NOT an exact no-move guarantee (#1211).
pub(crate) struct SaeResidentReducedSchur {
    /// Decoder output dimension `p` (the side length of every `G_i = L_iᵀ Y_i`).
    pub(crate) p: usize,
    /// Per-row **factored** residency: `(L_i, Y_i)`, each stored row-major as a
    /// `di × p` slab (`L_i` = local Jacobian, `Y_i = (H_tt^(i)+ρ_t I)⁻¹ L_i`).
    /// The reduced block is `G_i = L_iᵀ Y_i` (`p×p`, symmetric PSD), but it has
    /// rank ≤ `di` and `di ≪ p` for SAE rows (the per-row latent dim is 1–2
    /// while `p` is the decoder block width, ~2048). Materialising the dense
    /// `p×p` block would cost `O(n·p²)` memory (≈67 GB at the Qwen shape) and
    /// `p²` flops per matvec/row; the factored form costs `O(n·di·p)` memory and
    /// `2·support_i·p + 2·di·p` flops/row, applying `G_i v = L_iᵀ (Y_i v)`
    /// (sparse gather over `support_i` atoms → `di`-length GEMV → `p`-length
    /// GEMV → sparse scatter over `support_i` atoms). The `2·support_i·p`
    /// gather/scatter term is part of the per-row cost — for dense/full support
    /// `support_i` scales with active β-columns — and is not dropped. A row with
    /// empty active support / degenerate dims gets `di = 0` and is skipped.
    /// `(di, L_i, Y_i)` per row; `L_i`/`Y_i` are `di·p`-length row-major buffers.
    pub(crate) rows: Vec<ResidentRowFactor>,
    /// Per-row active atom support `(β-block base index, φ weight)`, shared with
    /// the assembler's [`DeviceSaePcgData`] (no re-clone of the index lists).
    pub(crate) a_phi: Arc<[Vec<(usize, f64)>]>,
}

/// Factored per-row residency block: `G_i = L_iᵀ Y_i` kept as its `di×p` factors
/// so the matvec never materialises the dense `p×p` product. See
/// [`SaeResidentReducedSchur`].
pub(crate) struct ResidentRowFactor {
    /// Row latent dimension `di` (the inner contraction width). `0` ⇒ skipped.
    pub(crate) di: usize,
    /// `L_i` row-major `di × p` (`di·p` entries). Empty when `di == 0`.
    pub(crate) l: Vec<f64>,
    /// `Y_i = (H_tt^(i)+ρ_t I)⁻¹ L_i` row-major `di × p`. Empty when `di == 0`.
    pub(crate) y: Vec<f64>,
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
    pub(crate) fn build<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        backend: &B,
    ) -> Option<Self> {
        let data = sys.device_sae_pcg.as_ref()?;
        let p = data.p;
        let n = sys.rows.len();
        if p == 0
            || sys.htbeta_dense_supplement
            || data.a_phi.len() != n
            || data.local_jac.len() != n
        {
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
    /// factored block in four steps: sparse gather `P_i x = Σ_s φ_s·x[base_s..]`
    /// (`support_i·p` flops), `w = Y_i·(P_i x)` (`di`-length, `di·p` flops),
    /// `prod = L_iᵀ·w` (`p`-length, `di·p` flops), and sparse scatter
    /// `acc += P_iᵀ prod` (`support_i·p` flops) — `2·support_i·p + 2·di·p`
    /// total, never the dense `p²` product. The gather/scatter `2·support_i·p`
    /// term is counted: it is not dominated by the GEMVs when the active support
    /// is wide.
    #[inline]
    pub(crate) fn row_into(
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
    pub(crate) fn max_di(&self) -> usize {
        self.rows.iter().map(|r| r.di).max().unwrap_or(0)
    }
}

/// Reduced-Schur matvec `out = S·x` with an optional pre-staged SAE residency
/// operator. When `resident` is `Some`, the per-row point-elimination term is
/// applied through the resident `p×p` blocks (#1017 CPU residency); otherwise it
/// falls back to the generic per-row `apply → solve → transpose` path. Both
/// routes accumulate the SAME reduced operator
/// `S = H_ββ + ρ_β I − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)`.
pub(crate) fn schur_matvec<B: BatchedBlockSolver + Sync>(
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
    // run-to-run regardless of thread scheduling (the #1017 verification gate).
    // This is deterministic and within the chunk-reassociation margin of serial,
    // so the criterion ranking is stable except for candidates that tie inside
    // that f64 margin — not an exact no-move guarantee (#1211). Stay
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
pub(crate) fn schur_matvec_row_into<B: BatchedBlockSolver>(
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
pub(crate) enum BlockFactor {
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
    pub(crate) blocks: Vec<BlockFactor>,
}

/// Maximum block size for which we attempt dense block-Jacobi factorization.
pub(crate) const BLOCK_JACOBI_MAX_BLOCK: usize = 256;

/// Positive-definiteness floor on a Schur-complement Jacobi diagonal entry.
/// A diagonal at or below this value (or non-finite) signals a non-PD reduced
/// system: the preconditioner cannot invert it, so the PCG solve fails loudly
/// and demands operator regularization rather than returning a garbage scale.
pub(crate) const JACOBI_DIAGONAL_PD_FLOOR: f64 = 1e-18;

impl JacobiPreconditioner {
    /// Build the block-Jacobi (or scalar fallback) preconditioner from the
    /// Arrow-Schur system without materializing the full dense Schur
    /// complement.
    ///
    /// When `sys.block_offsets` is non-empty and `max(block_size) ≤ 256`,
    /// each block gets a dense `b×b` Schur sub-matrix formed, factored, and
    /// stored.  Otherwise every column gets its own scalar entry.
    pub(crate) fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
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
            if let Some(res) = resident {
                Self::build_block_jacobi_resident(sys, ridge_beta, res)
            } else {
                Self::build_block_jacobi(sys, htt_factors, ridge_beta, backend)
            }
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
    pub(crate) fn build_scalar_jacobi<B: BatchedBlockSolver + Sync>(
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
        // Per-row body: subtract this row's `Σ_a (H_tβ^(i)e_a)ᵀ(H_tt^(i))⁻¹
        // (H_tβ^(i)e_a)` contribution into a caller-provided length-`K` diagonal
        // accumulator (`-=`). For each column `a`, probe the cross-block (or read
        // the dense slab) and compute the scalar point-elimination quotient. The
        // `O(K)` solves per row are the build's whole cost; the row contributions
        // are independent length-`K` vectors, so a worker sums a chunk into a
        // private `diag_part` and the caller folds the partials back in chunk
        // order — bit-identical run-to-run (the #1017 preconditioner gate).
        let row_into = |i: usize, row: &ArrowRowBlock, diag_part: &mut Array1<f64>| {
            let di = sys.row_dims[i];
            let mut col_i = Array1::<f64>::zeros(di);
            let mut e_a = Array1::<f64>::zeros(k);
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
                diag_part[a] -= acc;
            }
        };
        let n = sys.rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let partials: Vec<Array1<f64>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut diag_part = Array1::<f64>::zeros(k);
                    for i in idxs {
                        row_into(i, &sys.rows[i], &mut diag_part);
                    }
                    diag_part
                })
                .collect();
            // Deterministic ordered reduction: fold chunk partials left-to-right.
            for part in &partials {
                for a in 0..k {
                    diag[a] += part[a];
                }
            }
        } else {
            for (i, row) in sys.rows.iter().enumerate() {
                row_into(i, row, &mut diag);
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
    /// preconditioner is unchanged up to that f64 margin. Since the preconditioner
    /// only steers the iterate (which still terminates at the PCG tolerance), the
    /// criterion ranking is stable except for candidates within that margin,
    /// where the near-tie winner can flip — not an exact no-move guarantee (#1211).
    pub(crate) fn build_scalar_jacobi_resident(
        sys: &ArrowSchurSystem,
        ridge_beta: f64,
        resident: &SaeResidentReducedSchur,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        let p = resident.p;
        let n = resident.rows.len();
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
        //
        // The accumulation is into a SHARED `diag` (rows scatter into overlapping
        // `beta_base + j` columns), so — like the generic `build_scalar_jacobi`
        // and the `schur_matvec` row loop (#1017) — parallelism uses worker-private
        // length-`K` partials folded back in chunk order: each chunk is a
        // contiguous ascending row range and rows within it stay ascending, so the
        // chunk-ordered fold reproduces the serial `row = 0..n` subtraction order
        // bit-for-bit run-to-run (the #1017 determinism gate). Run-to-run
        // bit-identity does not extend to bit-identity with the in-place serial
        // accumulation, so the preconditioner — and any criterion ranking it
        // steers — is stable only up to the chunk-reassociation margin; a near-tie
        // winner inside that margin can flip (#1211).
        // This build runs once per inexact-PCG solve = O(inner-Newton-iters)
        // per fit; at the SAE LLM shape (thousands of rows, wide border `k`) the
        // per-row support sweep is the build's whole cost and was on one core.
        // The per-channel column dot `col_dot[j] = Σ_r L_i[r·p+j]·Y_i[r·p+j]`
        // (the diagonal of `G_i = L_iᵀ(H_tt)⁻¹L_i`) depends ONLY on the row `i`,
        // not on the support entry `(beta_base, φ)`. The previous loop recomputed
        // it once per support entry — a row with `m` active atoms paid `m·p`
        // column dots over `di`. Hoist it: compute the `p` column dots once per
        // row into reusable `col_dot` scratch, then each support entry is a pure
        // scatter `diag[beta_base+j] -= φ²·col_dot[j]`. Bit-for-bit identical:
        // each `col_dot[j]` is the same `r`-ascending sum, and `φ²·col_dot[j]`
        // yields identical bits whether `col_dot[j]` was just computed or cached.
        let row_into = |row: usize, diag_part: &mut [f64], col_dot: &mut [f64]| {
            let rf = &resident.rows[row];
            let di = rf.di;
            if di == 0 {
                return;
            }
            let support = &resident.a_phi[row];
            if support.is_empty() {
                return;
            }
            for (j, slot) in col_dot.iter_mut().enumerate().take(p) {
                let mut acc = 0.0_f64;
                for r in 0..di {
                    let idx = r * p + j;
                    acc += rf.l[idx] * rf.y[idx];
                }
                *slot = acc;
            }
            for &(beta_base, phi) in support {
                if phi == 0.0 {
                    continue;
                }
                let phi2 = phi * phi;
                for j in 0..p {
                    diag_part[beta_base + j] -= phi2 * col_dot[j];
                }
            }
        };
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let partials: Vec<Array1<f64>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut diag_part = Array1::<f64>::zeros(k);
                    let mut col_dot = vec![0.0_f64; p];
                    let slice = diag_part
                        .as_slice_mut()
                        .expect("diag_part must be contiguous");
                    for i in idxs {
                        row_into(i, slice, &mut col_dot);
                    }
                    diag_part
                })
                .collect();
            // Deterministic ordered reduction: fold chunk partials left-to-right
            // (each partial already holds the per-row terms subtracted, so add
            // them into `diag` in chunk order to mirror the serial subtraction).
            for part in &partials {
                for a in 0..k {
                    diag[a] += part[a];
                }
            }
        } else {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            let mut col_dot = vec![0.0_f64; p];
            for row in 0..n {
                row_into(row, diag_slice, &mut col_dot);
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

    /// Build block-Jacobi from the pre-staged SAE residency factors `(L_i, Y_i)`.
    ///
    /// This is the block analogue of [`Self::build_scalar_jacobi_resident`].
    /// When SAE block offsets are small enough to select BetaBlockJacobi (for
    /// example per-atom decoder blocks with `basis_size·p <= 256`), the generic
    /// block builder materializes every row's dense `(d_i × K)` `H_tβ` by probing
    /// the matrix-free operator, then re-solves `(H_tt)⁻¹` for each block column.
    /// The resident factors already carry `G_i = L_iᵀ(H_tt)⁻¹L_i`, so each block
    /// is assembled by scattering only the active support pairs inside that block:
    ///
    /// ```text
    /// S_block -= Σ_i Σ_(s,t in block support) φ_s φ_t · G_i[channel_s, channel_t]
    /// ```
    ///
    /// It computes the same block-diagonal restriction as the generic path, but
    /// avoids the full-row `H_tβ` materialization and per-column triangular solves.
    pub(crate) fn build_block_jacobi_resident(
        sys: &ArrowSchurSystem,
        ridge_beta: f64,
        resident: &SaeResidentReducedSchur,
    ) -> Result<Self, ArrowSchurError> {
        let block_offsets = &sys.block_offsets;
        let p = resident.p;
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

        let row_into = |row: usize, blocks: &mut [Array2<f64>]| {
            let rf = &resident.rows[row];
            let di = rf.di;
            if di == 0 {
                return;
            }
            let support = &resident.a_phi[row];
            if support.is_empty() {
                return;
            }
            for (block_idx, range) in block_offsets.iter().enumerate() {
                let block = &mut blocks[block_idx];
                for &(base_left, phi_left) in support {
                    if phi_left == 0.0 {
                        continue;
                    }
                    let left_start = base_left.max(range.start);
                    let left_end = (base_left + p).min(range.end);
                    if left_start >= left_end {
                        continue;
                    }
                    for &(base_right, phi_right) in support {
                        if phi_right == 0.0 {
                            continue;
                        }
                        let right_start = base_right.max(range.start);
                        let right_end = (base_right + p).min(range.end);
                        if right_start >= right_end {
                            continue;
                        }
                        let phi = phi_left * phi_right;
                        for gi in left_start..left_end {
                            let li = gi - range.start;
                            let ch_i = gi - base_left;
                            for gj in right_start..right_end {
                                let lj = gj - range.start;
                                let ch_j = gj - base_right;
                                let mut gij = 0.0_f64;
                                for r in 0..di {
                                    gij += rf.l[r * p + ch_i] * rf.y[r * p + ch_j];
                                }
                                block[[li, lj]] -= phi * gij;
                            }
                        }
                    }
                }
            }
        };

        let n = resident.rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let n_blocks = block_offsets.len();
            let block_dims: Vec<usize> = block_offsets.iter().map(|r| r.end - r.start).collect();
            let partials: Vec<Vec<Array2<f64>>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut local: Vec<Array2<f64>> = block_dims
                        .iter()
                        .map(|&b| Array2::<f64>::zeros((b, b)))
                        .collect();
                    for i in idxs {
                        row_into(i, &mut local);
                    }
                    local
                })
                .collect();
            for local in &partials {
                for bidx in 0..n_blocks {
                    schur_blocks[bidx] += &local[bidx];
                }
            }
        } else {
            for row in 0..n {
                row_into(row, &mut schur_blocks);
            }
        }

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
                let mut inv = Array1::<f64>::zeros(b);
                for bi in 0..b {
                    let v = schur_block[[bi, bi]];
                    if !v.is_finite() || v <= JACOBI_DIAGONAL_PD_FLOOR {
                        return Err(ArrowSchurError::PcgFailed {
                            reason: format!(
                                "SAE-resident block Jacobi scalar fallback: non-PD diagonal at \
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

    /// Build term-block Jacobi: one dense `b×b` Schur block per term in
    /// `sys.block_offsets`.
    pub(crate) fn build_block_jacobi<B: BatchedBlockSolver + Sync>(
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
        // Per-row body: materialize the row's `(d_i × K)` cross-block once and
        // subtract its `H_βt_k^(i)(H_tt^(i))⁻¹H_tβ_k^(i)` contribution into EACH
        // block-diagonal sub-block. Writes INTO a caller-provided `blocks`
        // accumulator (`-=`) so a rayon worker can subtract a chunk's rows into
        // a worker-private zero-seeded `Vec<Array2>` and the caller folds the
        // chunk partials back in chunk order — bit-identical run-to-run
        // regardless of thread scheduling (the #1017 verification gate). This
        // is deterministic and within the chunk-reassociation margin of serial,
        // so the preconditioner, hence the criterion ranking, is stable except
        // for near-tie candidates inside that f64 margin — not an exact no-move
        // guarantee (#1211).
        let row_into = |i: usize,
                        row: &ArrowRowBlock,
                        blocks: &mut [Array2<f64>]|
         -> Result<(), ArrowSchurError> {
            let di = sys.row_dims[i];
            let htbeta_full = sys_htbeta_materialize_row(sys, i, row)?;
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
                let schur_block = &mut blocks[block_idx];
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
            Ok(())
        };
        // Each row materializes an `O(K²)` cross-block (Kronecker) plus `Σ_k b_k`
        // triangular solves — the preconditioner build's whole per-row cost at
        // the SAE LLM shape (#1017), and the rows are independent. Fan over fixed
        // row chunks above the threshold, staying serial for the handful-of-rows
        // non-SAE callers and inside a rayon worker (topology-race nesting guard)
        // — the same gate `schur_matvec` uses.
        let n = sys.rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let n_blocks = block_offsets.len();
            let block_dims: Vec<usize> = block_offsets.iter().map(|r| r.end - r.start).collect();
            let partials: Vec<Vec<Array2<f64>>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut local: Vec<Array2<f64>> = block_dims
                        .iter()
                        .map(|&b| Array2::<f64>::zeros((b, b)))
                        .collect();
                    for i in idxs {
                        row_into(i, &sys.rows[i], &mut local)?;
                    }
                    Ok::<_, ArrowSchurError>(local)
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Deterministic ordered reduction: fold chunk partials left-to-right.
            for local in &partials {
                for bidx in 0..n_blocks {
                    schur_blocks[bidx] += &local[bidx];
                }
            }
        } else {
            for (i, row) in sys.rows.iter().enumerate() {
                row_into(i, row, &mut schur_blocks)?;
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

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
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
pub(crate) const PRECOND_ESCALATE_K_THRESHOLD: usize = 100;

/// Cholesky or scalar factor for one cluster of the beta-coefficient graph.
#[derive(Clone)]
pub(crate) enum ClusterFactor {
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
pub(crate) const CLUSTER_JACOBI_MAX_CLUSTER: usize = 512;

/// Dense Schur block per connected component of the beta-coupling graph.
///
/// Nodes = beta blocks (`block_offsets`); edges = rows where two blocks
/// co-occur with nonzero `H_t_beta` entries. One Cholesky factor per
/// connected component; applied as a triangular solve.
#[derive(Debug, Clone)]
pub struct ClusterJacobiPreconditioner {
    pub(crate) clusters: Vec<ClusterFactor>,
}

impl ClusterJacobiPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
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

    pub(crate) fn build_from_column_groups<B: BatchedBlockSolver + Sync>(
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
            // Per-row Schur contribution `-= H_tβ[cols]ᵀ (H_tt)⁻¹ H_tβ[cols]`,
            // subtracted into a (possibly thread-local) `b×b` accumulator. The
            // rows are independent, so this is the per-cluster analogue of the
            // already row-parallel `build_block_jacobi` body (#1017): at the SAE
            // LLM shape the `Σ_i di·b` triangular solves plus the `b²·di` cross
            // product are the cluster build's whole per-row cost.
            let cluster_row_into = |row_idx: usize, row: &ArrowRowBlock, acc: &mut Array2<f64>| {
                let mut col_vec = Array1::<f64>::zeros(d);
                let mut solved_cols = Array2::<f64>::zeros((d, b));
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
                        let mut dot = 0.0;
                        for c in 0..d {
                            dot += row.htbeta[[c, gi]] * solved_cols[[c, bj]];
                        }
                        acc[[bi, bj]] -= dot;
                    }
                }
            };
            // Fan over fixed 64-row chunks above the threshold, staying serial for
            // the handful-of-rows non-SAE callers and inside a rayon worker
            // (topology-race nesting guard). Chunk partials are folded
            // left-to-right so the result is bit-identical to the serial path.
            let n = sys.rows.len();
            let parallel =
                n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
            if parallel {
                use rayon::prelude::*;
                const CHUNK: usize = 64;
                let partials: Vec<Array2<f64>> = (0..n)
                    .into_par_iter()
                    .chunks(CHUNK)
                    .map(|idxs| {
                        let mut local = Array2::<f64>::zeros((b, b));
                        for i in idxs {
                            cluster_row_into(i, &sys.rows[i], &mut local);
                        }
                        local
                    })
                    .collect();
                for local in &partials {
                    s_block += local;
                }
            } else {
                for (row_idx, row) in sys.rows.iter().enumerate() {
                    cluster_row_into(row_idx, row, &mut s_block);
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

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
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
    pub(crate) clusters: Vec<ClusterFactor>,
    pub(crate) weights: Vec<f64>,
}

impl AdditiveSchwarzPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
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

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
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
pub(crate) enum ClusterApplyMode<'w> {
    Overwrite,
    Accumulate { weights: &'w [f64] },
}

impl ClusterApplyMode<'_> {
    #[inline]
    pub(crate) fn write(&self, out: &mut Array1<f64>, gi: usize, value: f64) {
        match self {
            ClusterApplyMode::Overwrite => out[gi] = value,
            ClusterApplyMode::Accumulate { weights } => out[gi] += weights[gi] * value,
        }
    }
}

/// Apply a single cluster factor to the residual `r`, writing into `out`
/// according to `mode` (overwrite for non-overlapping clusters, weighted
/// accumulate for overlapping Schwarz clusters).
pub(crate) fn apply_cluster(
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
pub(crate) fn build_schur_scalar_inv<B: BatchedBlockSolver>(
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
        let diag_slice = full_diag.as_slice_mut().expect("full_diag contiguous");
        sys.penalty_diagonal_add(diag_slice);
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
pub(crate) fn steihaug_pcg_auto<B: BatchedBlockSolver + Sync>(
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
    // memory and `2·support_i·p + 2·di·p` flops/row including the sparse
    // gather/scatter over the active support) once, up
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
pub(crate) fn run_pcg_with_preconditioner<ApplyPrec, B: BatchedBlockSolver + Sync>(
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
pub(crate) struct IdentityPreconditioner;

impl IdentityPreconditioner {
    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        r.clone()
    }
}

pub(crate) fn steihaug_dense_system(
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

pub(crate) fn steihaug_cg<MatVec, ApplyPrec>(
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

pub(crate) fn step_to_trust_boundary(
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

pub(crate) fn dense_matvec(a: &Array2<f64>, x: &Array1<f64>, out: &mut Array1<f64>) {
    let n = a.nrows();
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[[i, j]] * x[j];
        }
        out[i] = acc;
    }
}

pub(crate) fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

pub(crate) fn metric_dot(
    a: &Array1<f64>,
    b: &Array1<f64>,
    metric_weights: Option<&MetricWeights>,
) -> f64 {
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

pub(crate) fn metric_norm(v: ArrayView1<'_, f64>, metric_weights: Option<&MetricWeights>) -> f64 {
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

pub(crate) fn symmetrize_upper_from_lower(a: &mut Array2<f64>) {
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

pub(crate) fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>, String> {
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
