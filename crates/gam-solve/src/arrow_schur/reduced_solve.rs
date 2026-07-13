//! The reduced `K x K` shared-system solve: dense Schur assembly (direct and
//! square-root BA), the Schur matvec, the Jacobi/cluster/Schwarz
//! preconditioners, Steihaug-PCG, and the [`ArrowSchurError`] type.

use super::*;

/// Host budget for a dense reduced Schur `k × k` f64 matrix (#1017). Above this
/// the dense assembly is refused with a loud `SchurFactorFailed` rather than
/// OOM-killing the host. 8 GiB ⇒ `k ≈ 32768`; every currently-feasible SAE border
/// (k ≤ 5120 ⇒ 0.2 GiB) is well under it, while the qwen LLM border (k = 98304 ⇒
/// 77 GiB) is correctly rejected as matrix-free-only.
pub(crate) const DENSE_SCHUR_BYTES_BUDGET: u128 = 8 * 1024 * 1024 * 1024;

/// Reduce one contiguous device tile's rows into a private `-Σ leftᵀ·right`
/// partial (`k×k`).
///
/// The tile stacks its per-row `left_i` / `right_i` factors (each `d×k`) into
/// two `(Σ_i d_i × k)` matrices and tries a single per-ordinal `AᵀB` device
/// GEMM (`gam_gpu::try_fast_atb_on_ordinal`), which runs on the device this
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
            gam_gpu::try_fast_atb_on_ordinal(ordinal, left_stack.view(), right_stack.view())
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
/// usable, [`gam_gpu::pool::balanced_partition`] splits the `0..n` rows into
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

    // Size gate BEFORE the device probe (startup-tax ordering fix): the
    // multi-GPU tile path exists to overlap the per-row `leftᵀ·right` GEMMs
    // (≈ `2·d·k²` flops each, `2·n·d·k²` total) across the pool, and each
    // tile's GEMMs still pass through the policy-gated dispatch shims — which
    // refuse every op when the WHOLE assembly is below
    // `MIN_CALIBRATABLE_GEMM_FLOPS`, the smallest floor any reachable policy
    // can carry. Such a shape would only inherit the tile split's inter-tile
    // reassociation (the documented, tolerance-bounded departure) while doing
    // 100% CPU work, so route it to the serial/rayon reference path below
    // WITHOUT calling `GpuRuntime::global()` (whose first call creates a CUDA
    // primary context on every GPU). Shapes clearing the floor probe and tile
    // exactly as before.
    let assembly_work = 2u128 * (n as u128) * (sys.d as u128) * (k as u128) * (k as u128);
    let tiles = if assembly_work < gam_gpu::GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS {
        None
    } else {
        gam_gpu::device_runtime::GpuRuntime::global().and_then(|rt| {
            let tiles = gam_gpu::pool::balanced_partition(rt, n);
            // Engage the device stacked-GEMM reduction when a MULTI-GPU pool can
            // overlap tiles, OR — the single-GPU gap this closes — when the one
            // stacked `(total_d×k)ᵀ(total_d×k)` GEMM clears the runtime's own
            // `gemm_min_flops`, so `try_fast_atb_on_ordinal` will actually offload
            // it instead of declining back to CPU. This reduction is the dense
            // build's O(n·d·k²) cost (measured on an H100 as ~28% of the fit in
            // `block_gemm_subtract`), and on a single GPU it previously always ran
            // on the CPU because the tile path required `len() > 1` — the device
            // sat idle. `assembly_work` IS the stacked GEMM's flop count (2·k²·Σd),
            // so this is exactly `try_fast_atb`'s own offload predicate; below the
            // GEMM floor the launch/staging tax loses to the CPU, so we keep the
            // deterministic CPU rayon fold there. Small K (e.g. K=8) never clears
            // the floor and stays on the CPU — magic-by-default crossover, no flag.
            let engage = tiles.len() > 1 || assembly_work >= rt.policy().gemm_min_flops as u128;
            (engage && !tiles.is_empty()).then_some(tiles)
        })
    };

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
                        if let Some(ctx) = gam_gpu::device_runtime::cuda_context_for(ordinal) {
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
    // Fail LOUD, never OOM-kill (#1017): the dense reduced Schur is `k × k` f64.
    // At SAE LLM borders (qwen `k = 98304` ⇒ 77 GiB) materialising it would crash
    // the host. The matrix-free device PCG already solves the *step* without it
    // (`try_device_arrow_direct_sae_pcg`); only the joint-Hessian log-det still
    // routes here. A matrix-free determinant-lemma log-det (the proper follow-up)
    // is not yet wired, so refuse the allocation with an actionable error rather
    // than degrading silently into an OOM. The budget is generous so every
    // currently-feasible border (k ≤ 5120 ⇒ 0.2 GiB) is unaffected.
    let dense_bytes = (k as u128).saturating_mul(k as u128).saturating_mul(8);
    if dense_bytes > DENSE_SCHUR_BYTES_BUDGET {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "dense reduced Schur is {k}×{k} f64 = {} MiB, exceeding the {} MiB host budget; \
                 this border is matrix-free-only (the device PCG solves the step without the dense \
                 Schur) and a matrix-free determinant-lemma log-det is the required follow-up",
                dense_bytes / (1024 * 1024),
                DENSE_SCHUR_BYTES_BUDGET / (1024 * 1024),
            ),
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
    // Same fail-loud host-memory contract as the Direct reduction (#1017).  The
    // square-root BA route still materialises the same dense `k×k` reduced
    // Schur; letting this path bypass the budget would preserve an OOM-class
    // fallback even after Direct learned to refuse matrix-free-only borders.
    let dense_bytes = (k as u128).saturating_mul(k as u128).saturating_mul(8);
    if dense_bytes > DENSE_SCHUR_BYTES_BUDGET {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "square-root BA dense reduced Schur is {k}×{k} f64 = {} MiB, exceeding the \
                 {} MiB host budget; this border is matrix-free-only",
                dense_bytes / (1024 * 1024),
                DENSE_SCHUR_BYTES_BUDGET / (1024 * 1024),
            ),
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

/// Spectral positive-definiteness floor for the reduced Schur complement
/// `S` (#1026 SAE co-collapse SOLVE-path cure).
///
/// Reached only after the genuine Cholesky of `S` has REFUSED it (an indefinite
/// reduced Schur: collapsed atoms drive a per-row `H_tt` near-singular, so the
/// accumulated `Σ_i H_tβᵀ (H_tt)⁻¹ H_tβ` over-subtracts `H_ββ + ridge_β·I` into a
/// matrix with a non-positive eigenvalue). Rather than reject and let the LM
/// loop inflate `ridge_β` over EVERY β direction (the #1026 "crawl"), we
/// symmetric-eigendecompose `S` and clamp every eigenvalue UP to
/// `floor·max(λ)`. This is Levenberg–Marquardt restricted to exactly the
/// indefinite/collapsed subspace: a well-separated positive direction
/// (`λ ≫ floor·max λ`) keeps its EXACT eigenvalue (`λ.max(floor·max λ) = λ`), so
/// the Newton step in the healthy β subspace is unchanged, while only the
/// collapsed directions get the minimal positive stiffness needed for a PD
/// solve. Returns the floored, symmetric, strictly-PD matrix, or `None` if `S`
/// has no usable scale (non-finite / all-zero spectrum), in which case the
/// caller keeps the strict refusal.
///
/// Mirrors the per-row evidence floor
/// [`super::factorization::factor_spectral_deflated_criterion_row`]; the only
/// difference is the floored VALUE — a small positive `floor·max λ` (Tikhonov,
/// for an accurate solve) here, vs unit stiffness `+1` (`log 1 = 0`) there (for
/// the quotient log-det).
pub(crate) fn spectral_pd_floored_schur(
    schur: &Array2<f64>,
    relative_floor: f64,
) -> Option<(Array2<f64>, Array2<f64>)> {
    spectral_pd_floored_schur_with_factor(schur, relative_floor)
}

/// Shared body for [`spectral_pd_floored_schur`]: symmetrise, eigendecompose,
/// condition the spectrum, and return BOTH
/// the conditioned matrix `Σ λ̃_i v_i v_iᵀ` (consumed by Steihaug / matvec /
/// mixed-precision refinement) and its lower Cholesky factor.
///
/// The factor is built DIRECTLY from the conditioned spectral form — QR of
/// `W = diag(√λ̃)·Vᵀ` gives `A = WᵀW = RᵀR`, so `L = Rᵀ` — never by
/// re-factorising the reconstructed matrix. Reconstruct-then-refactor fails
/// under extreme eigenvalue spread: with `λ_max ~ 1e57` the `Σ λ̃ v vᵀ`
/// reconstruction carries `O(ε·λ_max)` round-off, which swamps unit-deflated
/// (`λ̃ = 1`) and floored (`λ̃ = floor·λ_max`) directions and re-poisons the
/// second Cholesky — the #2230 "spectral PD-floor reconstruction still non-PD"
/// refusal at a ρ whose conditioned evidence is perfectly well-defined. The QR
/// route factors the exact conditioned spectrum, so it succeeds whenever the
/// policy produced strictly positive `λ̃` (always, by construction).
fn spectral_pd_floored_schur_with_factor(
    schur: &Array2<f64>,
    relative_floor: f64,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let n = schur.nrows();
    if n == 0 || schur.ncols() != n || !(relative_floor.is_finite() && relative_floor > 0.0) {
        return None;
    }
    // Symmetrise defensively (the assembled Schur is symmetric up to reduction
    // order; the eig routine assumes exact symmetry).
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (schur[[i, j]] + schur[[j, i]]);
            if !v.is_finite() {
                return None;
            }
            sym[[i, j]] = v;
        }
    }
    let (evals, evecs) = sym.eigh(Side::Lower).ok()?;
    let max_abs = evals.iter().fold(
        0.0_f64,
        |acc, &v| if v.is_finite() { acc.max(v.abs()) } else { acc },
    );
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let floor = relative_floor * max_abs;
    // Newton-step policy (LM): clamp every eigenvalue UP to a strictly positive
    // `floor` — healthy positive directions (`λ ≫ floor`) keep their EXACT
    // eigenvalue, collapsed/indefinite directions get the minimal stiffness for
    // a stable `Δβ`.
    let mut conditioned = Array2::<f64>::zeros((n, n));
    let mut weighted_vt = Array2::<f64>::zeros((n, n));
    for eig_idx in 0..evals.len() {
        let lambda = evals[eig_idx];
        let lambda_conditioned = if lambda.is_finite() {
            lambda.max(floor)
        } else {
            floor
        };
        let sqrt_lambda = lambda_conditioned.sqrt();
        for i in 0..n {
            let vi = evecs[[i, eig_idx]];
            weighted_vt[[eig_idx, i]] = sqrt_lambda * vi;
            if vi == 0.0 {
                continue;
            }
            for j in 0..n {
                conditioned[[i, j]] += lambda_conditioned * vi * evecs[[j, eig_idx]];
            }
        }
    }
    let factor =
        spectral_qr_cholesky_factor(&weighted_vt).or_else(|| cholesky_lower(&conditioned).ok())?;
    Some((conditioned, factor))
}

/// Original-coordinate unit-deflation for an evidence reduced Schur.
///
/// The rank decision and unit pin are made in the caller's β coordinates. A
/// Jacobi congruence is appropriate for a Newton solve but would turn a unit
/// eigenvalue in scaled coordinates into a scale-dependent stiffness after
/// unscaling, corrupting both `log 1 = 0` and the cached null-space metadata.
fn factor_evidence_unit_deflated_schur(
    schur: &Array2<f64>,
    relative_floor: f64,
) -> Option<DenseReducedSchurFactorization> {
    let n = schur.nrows();
    if n == 0 || schur.ncols() != n || !(relative_floor.is_finite() && relative_floor > 0.0) {
        return None;
    }
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let value = 0.5 * (schur[[i, j]] + schur[[j, i]]);
            if !value.is_finite() {
                return None;
            }
            sym[[i, j]] = value;
        }
    }
    let (raw_evals, evecs) = sym.eigh(Side::Lower).ok()?;
    let max_abs = raw_evals.iter().fold(0.0_f64, |acc, &value| {
        if value.is_finite() {
            acc.max(value.abs())
        } else {
            acc
        }
    });
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let deflate_floor = relative_floor * max_abs * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
    let deflated: Vec<bool> = raw_evals
        .iter()
        .map(|&value| !value.is_finite() || value < deflate_floor)
        .collect();

    // Preserve the ordinary equilibrated-Cholesky bit path in the interior.
    // If Cholesky alone is numerically unable to factor a spectrally healthy
    // operator, the spectral QR below still factors the identical raw spectrum.
    if !deflated.iter().any(|&is_deflated| is_deflated)
        && let Ok(interior) = factor_dense_reduced_schur(schur, ReducedSchurPolicy::StrictNewton)
    {
        return Some(interior);
    }

    let mut cond_evals = raw_evals.clone();
    let mut conditioned = Array2::<f64>::zeros((n, n));
    let mut weighted_vt = Array2::<f64>::zeros((n, n));
    for eig_idx in 0..n {
        if deflated[eig_idx] {
            cond_evals[eig_idx] = 1.0;
        }
        let lambda = cond_evals[eig_idx];
        if !(lambda.is_finite() && lambda > 0.0) {
            return None;
        }
        let sqrt_lambda = lambda.sqrt();
        for i in 0..n {
            let vi = evecs[[i, eig_idx]];
            weighted_vt[[eig_idx, i]] = sqrt_lambda * vi;
            if vi != 0.0 {
                for j in 0..n {
                    conditioned[[i, j]] += lambda * vi * evecs[[j, eig_idx]];
                }
            }
        }
    }
    let factor = spectral_qr_cholesky_factor(&weighted_vt)?;
    let beta_deflation =
        deflated
            .iter()
            .any(|&is_deflated| is_deflated)
            .then(|| BetaSchurDeflationSpectrum {
                evecs,
                raw_evals,
                cond_evals,
                deflated: deflated.into(),
            });
    Some(DenseReducedSchurFactorization {
        factor,
        conditioned_schur: beta_deflation.as_ref().map(|_| conditioned),
        beta_deflation,
    })
}

/// Lower Cholesky factor of `A = WᵀW` computed from `W` itself: QR gives
/// `W = QR ⇒ A = RᵀR`, so the factor is `L = Rᵀ` (rows sign-fixed to a positive
/// diagonal). `W` here is `diag(√λ̃)·Vᵀ` with every `λ̃ > 0`, so `W` has full
/// rank and the factor exists exactly; returns `None` only if the QR itself
/// declines or produces a non-finite / zero pivot, in which case the caller
/// falls back to factoring the reconstructed matrix (the historical path).
fn spectral_qr_cholesky_factor(weighted_vt: &Array2<f64>) -> Option<Array2<f64>> {
    let n = weighted_vt.nrows();
    let (_q, r) = weighted_vt.qr().ok()?;
    if r.nrows() != n || r.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let d = r[[i, i]];
        if !d.is_finite() || d == 0.0 {
            return None;
        }
        let s = if d < 0.0 { -1.0 } else { 1.0 };
        for j in i..n {
            let v = s * r[[i, j]];
            if !v.is_finite() {
                return None;
            }
            l[[j, i]] = v;
        }
    }
    Some(l)
}

/// Jacobi/Van der Sluis diagonal equilibration scale for a symmetric matrix
/// (#2015): `d_a = sqrt(schur[a,a])`, floored at `√JACOBI_DIAGONAL_PD_FLOOR` so
/// a numerically-empty diagonal entry never divides by ~0. This is a PURE
/// numerical-conditioning aid for [`factor_dense_reduced_schur`] below — it is
/// never returned or exposed, and it changes no value any caller of that
/// function sees, only the accuracy of computing it.
fn jacobi_diagonal_scale(schur: &Array2<f64>) -> Array1<f64> {
    let n = schur.nrows();
    let floor_sqrt = JACOBI_DIAGONAL_PD_FLOOR.sqrt();
    let mut d = Array1::<f64>::zeros(n);
    for a in 0..n {
        let diag = schur[[a, a]];
        d[a] = if diag.is_finite() && diag > JACOBI_DIAGONAL_PD_FLOOR {
            diag.sqrt()
        } else {
            floor_sqrt
        };
    }
    d
}

/// Factor the dense reduced Schur complement `S`, returning its lower Cholesky
/// factor, the conditioned operator when policy changed it, and authoritative
/// β-null metadata for evidence unit deflation.
///
/// #2015 — SOLVER-LEVEL conditioning fix (design: issue 2015 comment
/// 4949898801). A real activation+behavior augmented target can carry output
/// column-norm spreads of ~1e4 (joint Hessian condition number ≈ 1e8), which a
/// PLAIN `cholesky_lower(schur)` is not designed to survive: the recursive
/// `L_ii = sqrt(S_ii − Σ_{j<i} L_ij²)` step loses precision (or falsely
/// refuses a genuinely PD matrix) when the diagonal spans many orders of
/// magnitude. Equilibrate FIRST: `D = diag(d)` with `d_a = sqrt(S_aa)`
/// ([`jacobi_diagonal_scale`] — Van der Sluis equilibration, provably within a
/// factor of `n` of the OPTIMAL diagonal preconditioner for a symmetric
/// matrix), factor `S̃ = D⁻¹SD⁻¹` (unit diagonal by construction) with the
/// EXACT SAME Cholesky/spectral-floor logic below, then undo the equilibration
/// on the way out.
///
/// This is NOT a reparametrization of any objective or estimand (contrast the
/// REVERTED #2015 attempt that divided the FIT TARGET's columns, which
/// changed what "best fit" means for a homoscedastic residual). `D` is
/// diagonal, so `L := D·L̃` is STILL lower-triangular, and
/// `L·Lᵀ = D·S̃·Dᵀ = D·(D⁻¹SD⁻¹)·D = S` exactly — `L` is a bit-exact valid
/// Cholesky factor of the CALLER'S ORIGINAL `schur`, just computed via a
/// numerically superior route. Undoing the scale is one exact elementwise
/// multiply (`factor[i,j] *= d[i]`, `floored[i,j] *= d[i]*d[j]`) — no further
/// precision is lost recovering original units. Evidence unit deflation
/// deliberately bypasses this congruence and works in the original β
/// coordinates so a unit-pinned null contributes exactly `log 1`.
///
/// GPU cross-reference: the device/GPU dense-reference path
/// (`gam_solve::gpu_kernels::arrow_schur::solve_arrow_newton_step_dense_reference`)
/// factors the full joint `(t, β)` system independently of this function and
/// does NOT yet get this equilibration. Both paths are exact; the GPU path is
/// simply not yet as well-conditioned on an ill-scaled system. Porting the
/// same technique there is a deliberate follow-up, not part of this change.
///
/// Newton-step damping and evidence quotient deflation are deliberately
/// different policies: Tikhonov directions retain a small positive curvature
/// for a stable step, while evidence-null directions are pinned to unit
/// stiffness so their log-determinant contribution is exactly zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReducedSchurPolicy {
    StrictNewton,
    NewtonTikhonov { relative_floor: f64 },
    EvidenceUnitDeflation { relative_floor: f64 },
}

impl ReducedSchurPolicy {
    pub(crate) fn newton(relative_floor: Option<f64>) -> Self {
        match relative_floor {
            Some(relative_floor) => Self::NewtonTikhonov { relative_floor },
            None => Self::StrictNewton,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DenseReducedSchurFactorization {
    pub(crate) factor: Array2<f64>,
    pub(crate) conditioned_schur: Option<Array2<f64>>,
    pub(crate) beta_deflation: Option<BetaSchurDeflationSpectrum>,
}

pub(crate) fn factor_dense_reduced_schur(
    schur: &Array2<f64>,
    policy: ReducedSchurPolicy,
) -> Result<DenseReducedSchurFactorization, ArrowSchurError> {
    let newton_relative_floor = match policy {
        ReducedSchurPolicy::StrictNewton => None,
        ReducedSchurPolicy::NewtonTikhonov { relative_floor } => Some(relative_floor),
        ReducedSchurPolicy::EvidenceUnitDeflation { relative_floor } => {
            return factor_evidence_unit_deflated_schur(schur, relative_floor).ok_or_else(|| {
                ArrowSchurError::SchurFactorFailed {
                    reason: "evidence reduced Schur unit-deflation declined (no usable spectrum)"
                        .to_string(),
                }
            });
        }
    };
    let n = schur.nrows();
    let d = jacobi_diagonal_scale(schur);
    let mut schur_scaled = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            schur_scaled[[i, j]] = schur[[i, j]] / (d[i] * d[j]);
        }
    }
    let (factor_scaled, floored_scaled) = match cholesky_lower(&schur_scaled) {
        Ok(factor) => (factor, None),
        Err(e) => {
            // #1026/#1038 — every dense reduced-Schur factorization in the SAE
            // path must honor the same opt-in spectral floor. Otherwise
            // auxiliary entry points (mixed precision and cross-row ordered Beta--Bernoulli
            // preconditioning) can reject the collapsed dead-atom subspace even
            // though the main direct solve would floor it and continue.
            //
            // #1803 — Newton-step callers use the Levenberg-Marquardt PD floor
            // (`spectral_pd_floored_schur`) so `Δβ` is stable. Evidence/log-det
            // callers (`unit_deflate_null_directions`) instead deflate
            // quotient/null directions to unit stiffness so they contribute the
            // ρ-independent `log 1 = 0` to the Laplace normaliser rather than a
            // ρ-dependent Occam reward for collapsed decoders.
            //
            // #2015 — this spectral floor runs on the EQUILIBRATED `schur_scaled`,
            // so `relative_floor` (a FRACTION of the operator's own max
            // eigenvalue) reads a numerically trustworthy spectrum instead of one
            // dominated by the raw column-scale spread; the floored
            // reconstruction is undone back to original units below exactly like
            // the plain factor.
            match newton_relative_floor {
                Some(relative_floor) => {
                    match spectral_pd_floored_schur(&schur_scaled, relative_floor) {
                        Some((floored, floored_factor)) => (floored_factor, Some(floored)),
                        None => {
                            return Err(ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "reduced Schur non-PD ({e}); spectral PD-floor declined \
                                 (no usable spectrum)"
                                ),
                            });
                        }
                    }
                }
                None => {
                    return Err(ArrowSchurError::SchurFactorFailed { reason: e });
                }
            }
        }
    };
    // Undo the equilibration exactly: L = D·L̃ (row i scaled by d_i); the
    // floored reconstruction (when present) scales back as D·S̃_floor·D.
    let mut factor = factor_scaled;
    for i in 0..n {
        let di = d[i];
        for j in 0..=i {
            factor[[i, j]] *= di;
        }
    }
    let floored_schur = floored_scaled.map(|mut floored| {
        for i in 0..n {
            for j in 0..n {
                floored[[i, j]] *= d[i] * d[j];
            }
        }
        floored
    });
    Ok(DenseReducedSchurFactorization {
        factor,
        conditioned_schur: floored_schur,
        beta_deflation: None,
    })
}

pub(crate) fn solve_dense_reduced_system(
    schur: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    options: &ArrowSolveOptions,
    metric_weights: Option<&MetricWeights>,
) -> Result<(Array1<f64>, Option<Array2<f64>>, ArrowPcgDiagnostics), ArrowSchurError> {
    let policy = ReducedSchurPolicy::newton(options.newton_schur_tikhonov_rel_floor);
    let DenseReducedSchurFactorization {
        factor,
        conditioned_schur: floored_schur,
        beta_deflation: _,
    } = factor_dense_reduced_schur(schur, policy)?;
    if let Some(floored) = floored_schur {
        let direct = mixed_precision_reduced_beta(&floored, &factor, rhs_beta, options)
            .unwrap_or_else(|| cholesky_solve_vector(&factor, rhs_beta));
        if step_inside_trust_region(direct.view(), options.trust_region.radius, metric_weights) {
            return Ok((direct, Some(factor), ArrowPcgDiagnostics::default()));
        }
        let identity = IdentityPreconditioner;
        let (delta, diag) = steihaug_dense_system(
            &floored,
            rhs_beta,
            &identity,
            &ArrowPcgOptions {
                max_iterations: options.trust_region.max_iterations,
                relative_tolerance: options.trust_region.steihaug_relative_tolerance,
            },
            &options.trust_region,
            metric_weights,
        )?;
        return Ok((delta, Some(factor), diag));
    }
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
    let schur_kappa = cholesky_factor_kappa_estimate(&factor);
    if !schur_kappa.is_finite() || schur_kappa > safe_spd_kappa_max(schur.nrows()) {
        // #1026 — over-complete SAE dictionaries park surplus atoms dead
        // (β_k → 0), so the reduced Schur is PD (the Cholesky above succeeded)
        // but ILL-CONDITIONED: the dead decoder subspace carries near-zero
        // eigenvalues while the live subspace is healthy. The kappa gate's
        // concern is an inaccurate Δβ from accumulated (H_tt)⁻¹ contamination —
        // but on the dead subspace the correct Δβ IS ≈0 (those atoms have no
        // signal), so the only "inaccuracy" is in directions whose true step is
        // zero. When the spectral PD-floor is enabled (the SAE solve path),
        // clamp exactly those collapsed directions up to `floor·max(λ)` and
        // solve against the floored Schur: the live subspace keeps its EXACT
        // Newton component, the dead subspace is damped to ≈0, and κ is bounded
        // so Δβ is accurate where it matters. This is the same conditioning the
        // non-PD branch above applies; here it also covers the PD-but-ill-
        // conditioned case so the LM loop does not exhaust `ridge_β` trying to
        // (futilely) lift a fundamentally rank-deficient dead-atom subspace.
        // Without the floor (BA / non-SAE callers) the strict refusal stands.
        if let Some(relative_floor) = options.newton_schur_tikhonov_rel_floor
            && let Some((floored, floored_factor)) =
                spectral_pd_floored_schur(schur, relative_floor)
        {
            let direct = mixed_precision_reduced_beta(&floored, &floored_factor, rhs_beta, options)
                .unwrap_or_else(|| cholesky_solve_vector(&floored_factor, rhs_beta));
            if step_inside_trust_region(direct.view(), options.trust_region.radius, metric_weights)
            {
                return Ok((direct, Some(floored_factor), ArrowPcgDiagnostics::default()));
            }
            let identity = IdentityPreconditioner;
            let (delta, diag) = steihaug_dense_system(
                &floored,
                rhs_beta,
                &identity,
                &ArrowPcgOptions {
                    max_iterations: options.trust_region.max_iterations,
                    relative_tolerance: options.trust_region.steihaug_relative_tolerance,
                },
                &options.trust_region,
                metric_weights,
            )?;
            return Ok((delta, Some(floored_factor), diag));
        }
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "reduced Schur complement Cholesky succeeded but is ill-conditioned \
                     (kappa_estimate={schur_kappa:e}); accumulated per-row \
                     (H_tt)⁻¹ contamination would yield an inaccurate Δβ"
            ),
        });
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
        return Ok((direct, Some(factor), ArrowPcgDiagnostics::default()));
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
        if gam_gpu::device_runtime::GpuRuntime::is_available() {
            match crate::gpu_kernels::arrow_schur::solve_reduced_beta_pcg(
                &schur,
                rhs_beta,
                options.trust_region.max_iterations,
                options.trust_region.steihaug_relative_tolerance,
            ) {
                Ok(delta_beta) => return Ok(delta_beta),
                Err(crate::gpu_kernels::arrow_schur::ArrowSchurGpuFailure::Unavailable) => {}
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
                    ArrowSchurError::SchurFactorFailed { .. }
                        | ArrowSchurError::PcgFailed { .. }
                        | ArrowSchurError::UnboundedNegativeCurvature { .. }
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
pub struct SaeResidentReducedSchur {
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
    /// #1033: per-row local Jacobian `L_i` (row-major `di × p`), SHARED via `Arc`
    /// with the assembler's [`DeviceSaePcgData`] rather than copied into each
    /// `ResidentRowFactor`. The staged factor previously held its own verbatim
    /// row-major copy of `data.local_jac[row]` — a second full `O(n·di·p)` slab
    /// for zero benefit (the bytes and the `di × p` layout are identical). The
    /// matvec now reads `L_i = &self.local_jac[row]` directly; only the SOLVED
    /// factor `Y_i = (H_tt+ρI)⁻¹ L_i` (genuinely new data) stays per-row. Reads
    /// are byte-for-byte the former `rf.l` (same slab, same `r·p + c` indexing),
    /// so the matvec/preconditioner output is bit-identical.
    pub(crate) local_jac: Arc<[Vec<f64>]>,
}

/// Factored per-row residency block: `G_i = L_iᵀ Y_i` kept as its `di×p` factors
/// so the matvec never materialises the dense `p×p` product. The local Jacobian
/// factor `L_i` is NOT stored here — it is shared via
/// [`SaeResidentReducedSchur::local_jac`] (`&local_jac[row]`); only the solved
/// `Y_i` is per-row. See [`SaeResidentReducedSchur`].
pub(crate) struct ResidentRowFactor {
    /// Row latent dimension `di` (the inner contraction width). `0` ⇒ skipped.
    pub(crate) di: usize,
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
            // Flatten the SOLVED factor to a `di × p` row-major buffer (iteration
            // over a standard-layout view is row-major regardless of the source
            // strides, so the hot loop can index `r*p + c` directly). `L_i` is NOT
            // copied — the matvec reads it from the shared `local_jac` slab (it is
            // byte-for-byte `data.local_jac[row]`).
            let y_flat: Vec<f64> = y.iter().copied().collect();
            ResidentRowFactor { di, y: y_flat }
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
            local_jac: data.local_jac_shared(),
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
        // Slice `x`/`acc` ONCE so the per-support gather/scatter (the dominant
        // `support·p` terms for wide active support) run over contiguous `f64`
        // slices — the compiler can prove unit stride and emit vectorized FMA,
        // where the former `x[base+j]`/`acc[base+j]` ndarray element indexing
        // forced a per-element strided lookup + bounds check that blocked
        // autovectorization. Every accumulation order is unchanged, so the
        // result is bit-identical to the ndarray-indexed form.
        let x_slice = x.as_slice().expect("resident matvec x must be contiguous");
        // P_i x = Σ_s φ_s · x[base_s .. base_s+p]   (length p).
        let gather = &mut gather[..p];
        for v in gather.iter_mut() {
            *v = 0.0;
        }
        for &(base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            let xrow = &x_slice[base..base + p];
            for (g, &xv) in gather.iter_mut().zip(xrow) {
                *g += phi * xv;
            }
        }
        // w = Y_i · (P_i x)   (di × p GEMV → length di).  Y_i row-major di×p.
        for r in 0..di {
            let yrow = &rf.y[r * p..r * p + p];
            let mut s = 0.0_f64;
            for (&yv, &gv) in yrow.iter().zip(gather.iter()) {
                s += yv * gv;
            }
            w[r] = s;
        }
        // prod = L_iᵀ · w   (p × di GEMV → length p).  L_i row-major di×p, so
        // L_iᵀ[j,r] = L_i[r,j]; accumulate column-by-column over the di rows.
        // `L_i` is the shared `local_jac[row]` slab (#1033) — byte-for-byte the
        // former per-row `rf.l` copy.
        let l_i = &self.local_jac[row];
        let prod = &mut prod[..p];
        for v in prod.iter_mut() {
            *v = 0.0;
        }
        for r in 0..di {
            let lrow = &l_i[r * p..r * p + p];
            let wr = w[r];
            for (pj, &lj) in prod.iter_mut().zip(lrow) {
                *pj += lj * wr;
            }
        }
        // acc += P_iᵀ prod = scatter φ_s · prod into base_s blocks.
        let acc_slice = acc
            .as_slice_mut()
            .expect("resident matvec acc must be contiguous");
        for &(base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            let arow = &mut acc_slice[base..base + p];
            for (a, &pv) in arow.iter_mut().zip(prod.iter()) {
                *a += phi * pv;
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
    // embarrassingly parallel — reduced below through the deterministic pairwise
    // tree (see the block-fold comment) rather than a chunk-order fold.
    let p = resident.map(|r| r.p).unwrap_or(0);
    // #2228 determinism: the per-row length-`k` contributions
    // (`Σ_i H_βt^(i)(H_tt^(i))⁻¹ H_tβ^(i) x`) are reduced through the length-only
    // pairwise tree so the result is bit-identical across thread count AND to the
    // sequential fold — parallel and nested-serial evaluation agree to the last
    // bit, removing the #1017/#1211 chunk-reassociation margin that let the
    // criterion ranking depend on the driver. The tree self-serializes below
    // `BASE_CHUNK` rows (a base block is folded directly with no `rayon::join`),
    // so small systems and nested topology-race calls stay single-threaded
    // without a separate branch that could associate the round-off differently.
    // The resident path gathers → factored `di×p` GEMVs → scatter; the direct
    // path does a per-row block solve — both ADD their row's contribution into a
    // block-local accumulator, so splitting the row sum across the tree is exact.
    let n_rows = sys.rows.len();
    let contribution = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
        n_rows,
        |range: core::ops::Range<usize>| {
            let mut acc = Array1::<f64>::zeros(k);
            if let Some(res) = resident {
                let mut gather = vec![0.0_f64; p];
                let mut prod = vec![0.0_f64; p];
                let mut w = vec![0.0_f64; res.max_di()];
                for i in range {
                    res.row_into(i, x, &mut acc, &mut gather, &mut prod, &mut w);
                }
            } else {
                let mut local = Array1::<f64>::zeros(sys.d);
                for i in range {
                    schur_matvec_row_into(sys, htt_factors, x, backend, i, &mut local, &mut acc);
                }
            }
            acc
        },
        |mut a: Array1<f64>, b: Array1<f64>| {
            a += &b;
            a
        },
    );
    if let Some(acc) = contribution {
        for a in 0..k {
            out[a] -= acc[a];
        }
    }
}

/// #1017: the reduced-Schur operator `v ↦ S·v` staged ONCE per criterion
/// evaluation and reused across EVERY shifted / warm-started solve of the
/// rational-logdet (and SLQ) ladder — the widened-lifetime residency the #1017
/// device design calls for.
///
/// The rational-logdet criterion (`matrix_free_arrow_evidence_log_det_surrogate`)
/// walks SEVERAL shift ladders inside ONE evaluation: the `λ_max` power iteration
/// ([`reduced_schur_lambda_max`]), the pilot / deflation-derived plan build
/// ([`rational_reduced_schur_plan_derived`]), the value [`RationalLogdetPlan::
/// evaluate`], and the `(probes, S⁻¹·probes)` gradient bundle
/// ([`reduced_schur_inverse_probe_solves`]). Each formerly re-captured its own
/// inline `schur_matvec` closure over `(sys, htt_factors, ρ_β, backend,
/// resident)`. On CPU those captures are free; on the device lane they are the
/// per-solve FLATTEN — every ladder would re-marshal and re-upload the
/// ridge-independent operands (the factored `H_tt` slab, the framed `G ⊗ W`, the
/// dense per-row cross blocks) that are INVARIANT across the whole evaluation.
///
/// This object is the single operator every ladder borrows: the invariant state
/// (`sys`, the factored `H_tt` slab, the `ρ_β` border, the pre-staged CPU
/// [`SaeResidentReducedSchur`] frame, and — when engaged — a device-resident
/// [`GpuSchurMatvec`] whose per-row factors upload ONCE) lives for the whole
/// evaluation, so a shifted solve reuses the resident operator instead of
/// re-staging it. Every `apply` accumulates the SAME reduced operator
/// `S = (H_ββ + ρ_β I) − Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹H_tβ^(i)` regardless of
/// lane. With `gpu_matvec == None` (every current construction) the result is
/// byte-for-byte the pre-context inline `schur_matvec` closure; the `gpu_matvec`
/// seam is where a device operator, built once per evaluation, is threaded through
/// the ladder (the reported #1017 next increment).
pub(crate) struct ReducedSchurOperator<'a, B: BatchedBlockSolver + Sync> {
    sys: &'a ArrowSchurSystem,
    htt_factors: &'a ArrowFactorSlab,
    ridge_beta: f64,
    backend: &'a B,
    resident: Option<&'a SaeResidentReducedSchur>,
    gpu_matvec: Option<&'a GpuSchurMatvec>,
}

impl<'a, B: BatchedBlockSolver + Sync> ReducedSchurOperator<'a, B> {
    /// The CPU/host operator — the byte-identical default. Every shifted solve in
    /// the evaluation reuses the same pre-staged `resident` frame (or the generic
    /// per-row `apply → solve → transpose` when `resident` is `None`).
    pub(crate) fn new(
        sys: &'a ArrowSchurSystem,
        htt_factors: &'a ArrowFactorSlab,
        ridge_beta: f64,
        backend: &'a B,
        resident: Option<&'a SaeResidentReducedSchur>,
    ) -> Self {
        Self {
            sys,
            htt_factors,
            ridge_beta,
            backend,
            resident,
            gpu_matvec: None,
        }
    }

    /// Attach a device-resident [`GpuSchurMatvec`] (built ONCE per evaluation) so
    /// the whole ladder applies `S·v` on device without a per-solve re-upload.
    /// #1017 next increment: the caller that owns the device operand upload builds
    /// the operator once and calls this; until then every construction is CPU
    /// (`gpu_matvec == None`), so the lane stays byte-identical.
    pub(crate) fn with_gpu_matvec(mut self, gpu_matvec: Option<&'a GpuSchurMatvec>) -> Self {
        self.gpu_matvec = gpu_matvec;
        self
    }

    /// `out = S·x`. Both lanes CLEAR and fully assign `out`, so a fresh zeroed
    /// buffer per apply is correct (and the shift-ladder CG contract is upheld).
    #[inline]
    pub(crate) fn apply_into(&self, x: &Array1<f64>, out: &mut Array1<f64>) {
        let Some(quotient) = self.sys.beta_gauge_quotient.as_ref() else {
            if let Some(gpu) = self.gpu_matvec {
                gpu(x, out);
            } else {
                schur_matvec(
                    self.sys,
                    self.htt_factors,
                    self.ridge_beta,
                    x,
                    out,
                    self.backend,
                    self.resident,
                );
            }
            return;
        };

        // Evidence operator on the quotient: `P S P + Q Q^T`.  Apply the
        // original reduced Schur only to `P x`, project its result once more,
        // then add the unit Faddeev--Popov pin. The same arithmetic is used by
        // dense `pin_reduced_schur`, so SLQ/rational-logdet values and dense
        // Cholesky values represent the identical operator.
        let projected_x = quotient.project_complement(x.view());
        if let Some(gpu) = self.gpu_matvec {
            gpu(&projected_x, out);
        } else {
            schur_matvec(
                self.sys,
                self.htt_factors,
                self.ridge_beta,
                &projected_x,
                out,
                self.backend,
                self.resident,
            );
        }
        let mut projected_out = quotient.project_complement(out.view());
        for direction in quotient.directions.iter() {
            projected_out.scaled_add(direction.dot(x), direction);
        }
        out.assign(&projected_out);
    }

    /// `S·v` into a fresh length-`k` vector — the shift-ladder matvec-closure form
    /// (`|v: ArrayView1| op.apply(v)`). Byte-for-byte the inline
    /// `let x = v.to_owned(); schur_matvec(…, &x, &mut zeros(k), …)` it replaces.
    #[inline]
    pub(crate) fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let x = v.to_owned();
        let mut out = Array1::<f64>::zeros(self.sys.k);
        self.apply_into(&x, &mut out);
        out
    }

    /// `S·x` into a fresh vector from an already-owned `&Array1` (no redundant copy
    /// of a vector the caller already owns) — the power-iteration / CG-solve form.
    #[inline]
    pub(crate) fn apply_owned(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.sys.k);
        self.apply_into(x, &mut out);
        out
    }
}

/// Matrix-free reduced-Schur log-determinant `log|S|` via Stochastic Lanczos
/// Quadrature on the exact `schur_matvec` apply `v ↦ S·v`, where
/// `S = (H_ββ + ρ_β I) − Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹H_tβ^(i)` is the SPD
/// reduced Schur. **The dense `k×k` `S` is NEVER formed.**
///
/// This is the memory-matrix-free evidence path for the massive-K manifold SAE.
/// The dense evidence routes assemble `S` explicitly (`O(k²)` ≈ 8 GB at the
/// K=32k border) and Cholesky-factor it (`O(k³/3)`) purely to read `Σ 2·log Lᵢᵢ`;
/// that dense assembly + factor is the massive-K wall (both dense evidence
/// routes REFUSE above the in-core budget). Here peak memory is `O(k)` — the SLQ
/// Rademacher probe and Lanczos basis vectors — and the cost is
/// `O(num_probes·lanczos_steps · matvec)`, each matvec the same `O(n·d·k)`
/// reduced-Schur apply the PCG hot loop already runs. Deterministic for a fixed
/// `(sys, htt_factors, ρ_β, resident, num_probes, lanczos_steps, seed)` so the
/// REML evidence outer loop stays reproducible.
///
/// `htt_factors` are the per-row `(H_tt^(i)+ρ_t I)` Cholesky factors; `resident`
/// is the optional pre-staged SAE residency operator (`None` for the framed /
/// closure `H_tβ` path). SLQ is an ESTIMATE — the same accuracy contract the
/// device seam already accepts for `k ≥ SCHUR_SLQ_LOGDET_MIN_DIM`; callers that
/// need the exact dense log-det at small `k` must stay on the dense route.
///
/// Crate-internal because the `resident` parameter carries the `pub(crate)`
/// [`SaeResidentReducedSchur`] operator; cross-crate callers use the
/// [`matrix_free_arrow_evidence_log_det`] convenience, which stages residency
/// internally and exposes no crate-private type.
pub(crate) fn slq_reduced_schur_log_det<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> SlqLogDet {
    let k = sys.k;
    // Stage the reduced-Schur operator ONCE; every probe/Lanczos apply reuses the
    // pre-staged residency (no per-apply operator re-capture). The probes fan
    // across rayon workers (in `slq_logdet`), and `schur_matvec`'s own row
    // parallelism is guarded off inside a worker, so there is no nested
    // oversubscription. When `gpu_matvec` is `Some` (the #1017 Phase-3 device
    // seam, built once for the whole evidence evaluation), EVERY Rademacher-probe
    // Lanczos apply runs through the single resident device `S·v`; when `None`
    // the byte-identical CPU `schur_matvec` lane is taken.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    slq_logdet(k, |v| op.apply(v), num_probes, lanczos_steps, seed)
}

/// One-call matrix-free arrow evidence log-determinant for an assembled system.
///
/// Factors the per-row `H_tt^(i)+ρ_t I` blocks (accumulating
/// `log_det_tt = Σ_i Σ_axis 2·log Lᵢᵢ` from the Cholesky diagonals — the cheap
/// `O(n·d³)` t-tier term), stages the SAE residency operator when the system
/// carries `device_sae_pcg` full-`B` data, and estimates `log|S|` via
/// [`slq_reduced_schur_log_det`] with NO dense `k×k` Schur formed at any point.
///
/// Returns `(log_det_tt, log|S| SLQ estimate)`; the undamped joint evidence
/// log-det the Laplace normaliser needs is their sum. Uses the identical
/// [`factor_blocks_for_system`] the dense Direct evidence path uses (same gauge
/// deflation), so `log_det_tt` matches the dense convention exactly and only the
/// `k×k` Schur term is replaced by its matrix-free SLQ estimate.
pub fn matrix_free_arrow_evidence_log_det(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Result<(f64, SlqLogDet), ArrowSchurError> {
    let backend = CpuBatchedBlockSolver;
    let factorization = factor_blocks_for_system(
        sys,
        ridge_t,
        options.evidence_policy.factors_undamped_evidence(),
        &backend,
    )?;
    let htt_factors = factorization.factors;
    let mut log_det_tt = 0.0_f64;
    for row in 0..htt_factors.len() {
        let factor = htt_factors.factor(row);
        for axis in 0..factor.nrows() {
            log_det_tt += 2.0 * factor[[axis, axis]].ln();
        }
    }
    // #1017 Phase-3: build the reduced-Schur device `S·v` ONCE for the whole SLQ
    // evaluation. Every Rademacher-probe Lanczos apply then rides that single
    // resident operator (uploaded/pre-factored once) instead of re-capturing the
    // CPU `schur_matvec` per apply. The device operator carries its own residency,
    // so the CPU `SaeResidentReducedSchur` frame is only staged on the CPU lane.
    let device_matvec = maybe_build_evidence_gpu_matvec(
        sys,
        ridge_t,
        ridge_beta,
        options,
        num_probes.saturating_mul(lanczos_steps),
    );
    let gpu_matvec: Option<&GpuSchurMatvec> =
        options.gpu_matvec.as_ref().or(device_matvec.as_ref());
    let resident = if gpu_matvec.is_none() {
        SaeResidentReducedSchur::build(sys, &htt_factors, &backend)
    } else {
        None
    };
    let slq = slq_reduced_schur_log_det(
        sys,
        &htt_factors,
        ridge_beta,
        &backend,
        resident.as_ref(),
        gpu_matvec,
        num_probes,
        lanczos_steps,
        seed,
    );
    Ok((log_det_tt, slq))
}

/// #1017 Phase-3: build the reduced-Schur device matvec ONCE for a matrix-free
/// evidence log-det evaluation, so the whole rational-logdet + SLQ ladder applies
/// `S·v` through a single device-resident operator (uploaded / pre-factored once)
/// rather than re-capturing the CPU `schur_matvec` per probe / shifted solve. The
/// PCG numerics are identical whether the matvec runs on host or device (same
/// reduced Schur operator, same f64 accumulation), so engaging it changes only
/// where the `Σ_i H_βt(H_tt)⁻¹H_tβ` flops execute.
///
/// Same admission contract as the PCG matvec offload ([`maybe_inject_gpu_schur_matvec`]):
/// declines (returns `None`, so every apply stays on the byte-identical CPU lane)
/// when cross-row penalties or streaming are present, the work predicate rejects
/// the shape, or no live device is present. `apply_budget` is the amortising apply
/// count for the shape predicate — the reduced-Schur matvec is `O(n·d·k)` per
/// apply and the evidence ladder runs that apply across every probe / Lanczos /
/// shifted-CG step, so a large budget is the honest amortisation the offload
/// break-even is measured against.
pub(crate) fn maybe_build_evidence_gpu_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    apply_budget: usize,
) -> Option<GpuSchurMatvec> {
    // A caller-supplied operator (threaded through `options.gpu_matvec`) already
    // owns its residency; the caller passes it directly, so never double-build.
    if options.gpu_matvec.is_some() {
        return None;
    }
    if !sys.cross_row_penalties.is_empty() || options.streaming_chunk_size.is_some() {
        return None;
    }
    // Size gate BEFORE the device probe (startup-tax ordering): the predicate
    // reads only associated constants, so a shape it rejects skips
    // `GpuRuntime::global()` (whose first call creates a CUDA primary context on
    // every GPU); an admitted shape probes exactly as the PCG seam does.
    if !gam_gpu::GpuDispatchPolicy::default().reduced_schur_matvec_should_offload(
        sys.rows.len(),
        sys.k,
        sys.d,
        apply_budget.max(1),
    ) {
        return None;
    }
    gam_gpu::device_runtime::GpuRuntime::global()?;
    // #1017: framed matrix-free system with resident device operands — prefer the
    // device-resident DETERMINISTIC reduced-Schur apply (upload operands once,
    // cross only x/out per apply, atomics-free so the SLQ log|S| determinism
    // contract holds) over the CPU row-procedural closure `gpu_schur_matvec_backend`
    // returns for `htbeta_matvec` systems. Declines (no device / shape / non-PD at
    // this ridge) fall through to the backend/CPU path. Non-Linux/CPU: this always
    // returns `None` (no `device_sae_pcg`), so the lane is byte-identical.
    if sys.device_sae_pcg.is_some() {
        if let Some(matvec) = crate::gpu_kernels::arrow_schur::build_framed_resident_evidence_matvec(
            sys,
            ridge_t,
            ridge_beta,
            apply_budget.max(1),
        ) {
            return Some(matvec);
        }
    }
    crate::gpu_kernels::arrow_schur::gpu_schur_matvec_backend(sys, ridge_t, ridge_beta).ok()
}

/// Fixed configuration for the #2080 rational-surrogate evidence lane: the probe
/// count, seeds, quadrature/CG tolerances, and derived-rank deflation budget the
/// [`SurrogateLaneState`] plan is (re)built with. The caller (the SAE streaming
/// criterion) supplies these once; `deflation_target_std_err_rel` is the derived
/// bar `0.1 · STALL_REL_TOL` (see [`rational_reduced_schur_plan_derived`]).
#[derive(Clone)]
pub struct SurrogateLaneConfig {
    pub num_probes: usize,
    pub seed: u64,
    pub rel_tol: f64,
    pub power_iters: usize,
    pub cg_rel_tol: f64,
    pub cg_max_iters: usize,
    pub deflation_max_rank: usize,
    pub deflation_subspace_iters: usize,
    pub deflation_target_std_err_rel: f64,
}

/// Per-outer-solve state for the #2080 rational-surrogate evidence lane. Holds
/// the FROZEN derived-rank plan — probes, bracket-centred quadrature, and Hutch++
/// `Q`, all fixed once at the entry ρ so value and gradient stay a single
/// functional across the ρ sweep — plus the config to (re)build it when the
/// reduced-Schur dimension changes (a basin mutation between outer solves).
/// Threaded as `Option<&mut _>` through the streaming criterion; `None` keeps the
/// bit-identical SLQ path.
pub struct SurrogateLaneState {
    plan: Option<RationalLogdetPlan>,
    cfg: SurrogateLaneConfig,
    /// When set, the next matrix-free evidence eval also computes the shared
    /// `(probes, S⁻¹·probes)` bundle for EFS/MacKay proposal traces and stashes
    /// it in `inverse_probes`. It is never an outer gradient artifact: the fixed
    /// rational value's derivative is `logdet_derivative_bundle` below.
    request_inverse_probes: bool,
    /// The last-computed shared bundle: the FROZEN plan's probes `v_j` and their
    /// `S⁻¹ v_j` (t = 0) solves at the current operator. One bundle drives every
    /// selected-inverse trace `tr(S⁻¹·M) ≈ (1/m)Σ_j (S⁻¹v_j)ᵀ(M v_j)` off the
    /// same frozen raw probes as the value plan. This is useful for EFS trace
    /// proposals but is not the derivative of the shifted rational value.
    inverse_probes: Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)>,
    /// Request/stash the lossless weighted derivative representation emitted by
    /// the next rational value evaluation.  Unlike `inverse_probes`, this is the
    /// derivative of the fixed rational surrogate itself (all shifted solves and
    /// frozen-Q columns), and is the only bundle admissible for its outer
    /// gradient.
    request_logdet_derivative_bundle: bool,
    logdet_derivative_bundle: Option<RationalLogdetDerivativeBundle>,
    /// The previous ρ's `S⁻¹ v_j` solves, kept as the CG warm-start for the next
    /// bundle solve. `S⁻¹` is smooth in ρ, so a neighbouring-ρ solution is a near
    /// seed (common-random-numbers reuse — the discipline that makes the
    /// surrogate's shifted ladder cheap); the converged solve is unchanged to
    /// `cg_rel_tol`, only its iteration count drops. Cleared when the plan rebuilds
    /// (basin border change ⇒ the old-dim seeds are meaningless).
    warm_inverse_probes: Option<Vec<Array1<f64>>>,
}

impl SurrogateLaneState {
    /// A lane with no plan yet — the first evaluation builds and freezes it.
    pub fn new(cfg: SurrogateLaneConfig) -> Self {
        Self {
            plan: None,
            cfg,
            request_inverse_probes: false,
            inverse_probes: None,
            request_logdet_derivative_bundle: false,
            logdet_derivative_bundle: None,
            warm_inverse_probes: None,
        }
    }

    /// The frozen plan, once built (for the gradient lane, which contracts
    /// against the SAME `Q` the value used).
    pub fn plan(&self) -> Option<&RationalLogdetPlan> {
        self.plan.as_ref()
    }

    /// Ask the next matrix-free evidence eval to also emit the shared
    /// `(probes, S⁻¹·probes)` bundle. Clears any stale bundle so a failed or
    /// skipped eval cannot hand back last call's solves.
    pub fn request_inverse_probes(&mut self) {
        self.request_inverse_probes = true;
        self.inverse_probes = None;
    }

    /// Take the shared bundle produced by the most recent eval, if requested and
    /// computed. Consumes it so a later gradient read cannot reuse stale solves.
    pub fn take_inverse_probes(&mut self) -> Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)> {
        self.request_inverse_probes = false;
        self.inverse_probes.take()
    }

    /// Ask the next rational value evaluation to retain its complete weighted
    /// derivative representation. Clears stale output eagerly so a failed value
    /// cannot be paired with a previous operator's gradient.
    pub fn request_logdet_derivative_bundle(&mut self) {
        self.request_logdet_derivative_bundle = true;
        self.logdet_derivative_bundle = None;
    }

    /// Consume the derivative representation produced by the most recent
    /// requested rational value evaluation.
    pub fn take_logdet_derivative_bundle(&mut self) -> Option<RationalLogdetDerivativeBundle> {
        self.request_logdet_derivative_bundle = false;
        self.logdet_derivative_bundle.take()
    }
}

/// Split arrow-Schur evidence `log|H| = Σ log|H_tt| + log|S|` where the reduced
/// Schur term is estimated by the #2080 rational surrogate rather than SLQ, on
/// ONE shared factorization. The build-once companion to
/// [`matrix_free_arrow_evidence_log_det`]:
///
/// - `lane = None` runs the identical [`slq_reduced_schur_log_det`] path — a
///   bit-for-bit fallback so a caller that has not opted in is unchanged.
/// - `lane = Some(state)` builds (or, when the reduced-Schur dimension is
///   unchanged, reuses) the frozen derived-rank [`RationalLogdetPlan`] and
///   evaluates it against the current operator. The plan's `Q`/probes/quadrature
///   are fixed at first build, so only the matrix-free `S·v` apply moves with ρ —
///   the value and its [`RationalLogdetPlan::directional_derivative`] gradient
///   remain one functional.
///
/// Returns `(log_det_tt, log_det_schur)`; the caller adds them for the evidence.
pub fn matrix_free_arrow_evidence_log_det_surrogate(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    slq_num_probes: usize,
    slq_lanczos_steps: usize,
    slq_seed: u64,
    lane: Option<&mut SurrogateLaneState>,
) -> Result<(f64, f64), ArrowSchurError> {
    let backend = CpuBatchedBlockSolver;
    let factorization = factor_blocks_for_system(
        sys,
        ridge_t,
        options.evidence_policy.factors_undamped_evidence(),
        &backend,
    )?;
    let htt_factors = factorization.factors;
    let mut log_det_tt = 0.0_f64;
    for row in 0..htt_factors.len() {
        let factor = htt_factors.factor(row);
        for axis in 0..factor.nrows() {
            log_det_tt += 2.0 * factor[[axis, axis]].ln();
        }
    }
    // #1017 Phase-3: one device-resident reduced-Schur `S·v` for the WHOLE
    // evaluation — the surrogate value ladder (two-sided deflation: block-power on
    // S + inverse subspace iteration on S⁻¹ via matrix-free CG), the λ_max bracket
    // power iteration, the SLQ probes, AND the S⁻¹·probe bundle all ride this
    // single operator (uploaded / pre-factored once). Sized against the surrogate's
    // per-evaluation apply budget (probe count × shifted-CG ladder depth). The
    // device operator carries its own residency, so the CPU `SaeResidentReducedSchur`
    // frame is only staged on the CPU lane.
    let cfg_apply_budget = lane
        .as_ref()
        .map(|s| s.cfg.num_probes.saturating_mul(s.cfg.cg_max_iters))
        .unwrap_or_else(|| slq_num_probes.saturating_mul(slq_lanczos_steps));
    let device_matvec =
        maybe_build_evidence_gpu_matvec(sys, ridge_t, ridge_beta, options, cfg_apply_budget);
    let gpu_matvec: Option<&GpuSchurMatvec> =
        options.gpu_matvec.as_ref().or(device_matvec.as_ref());
    let resident = if gpu_matvec.is_none() {
        SaeResidentReducedSchur::build(sys, &htt_factors, &backend)
    } else {
        None
    };

    let log_det_schur = match lane {
        None => {
            let slq = slq_reduced_schur_log_det(
                sys,
                &htt_factors,
                ridge_beta,
                &backend,
                resident.as_ref(),
                gpu_matvec,
                slq_num_probes,
                slq_lanczos_steps,
                slq_seed,
            );
            slq.estimate
        }
        Some(state) => {
            let dim = sys.k;
            // (Re)build the frozen plan when absent or dimension-mismatched (a
            // basin mutation changed the border); otherwise reuse the frozen Q.
            let need_build = state.plan.as_ref().map_or(true, |p| p.dim != dim);
            if need_build {
                let cfg = state.cfg.clone();
                let plan = rational_reduced_schur_plan_derived(
                    sys,
                    &htt_factors,
                    ridge_beta,
                    &backend,
                    resident.as_ref(),
                    gpu_matvec,
                    cfg.num_probes,
                    cfg.seed,
                    cfg.rel_tol,
                    cfg.power_iters,
                    cfg.cg_rel_tol,
                    cfg.cg_max_iters,
                    cfg.deflation_max_rank,
                    cfg.deflation_subspace_iters,
                    cfg.deflation_target_std_err_rel,
                )
                .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "rational log-det surrogate plan build failed for reduced Schur dim {dim}"
                    ),
                })?;
                state.plan = Some(plan);
                // The old-dim S⁻¹·probes are meaningless against the new border.
                state.warm_inverse_probes = None;
            }
            let plan = state
                .plan
                .as_ref()
                .expect("plan installed just above when absent");
            let want_bundle = state.request_inverse_probes;
            let want_logdet_derivative = state.request_logdet_derivative_bundle;
            // Value, its lossless shifted derivative representation, and any
            // EFS-only `(probes, S⁻¹·probes)` trace bundle are computed under one
            // borrow of the frozen plan and stashed after that borrow ends. The
            // EFS bundle uses raw probes; the outer gradient consumes only the
            // weighted shifted derivative bundle.
            let (estimate, derivative_bundle, bundle) = {
                // #1017: ONE reduced-Schur operator for the whole value ladder —
                // the frozen plan walks its shift ladder through this single
                // resident apply instead of re-capturing a `schur_matvec` closure
                // per shifted solve. When `gpu_matvec` is `Some` (Phase-3 device
                // seam, built once above) every shifted apply runs on device; when
                // `None` the byte-identical CPU `schur_matvec` lane is taken.
                let op = ReducedSchurOperator::new(
                    sys,
                    &htt_factors,
                    ridge_beta,
                    &backend,
                    resident.as_ref(),
                )
                .with_gpu_matvec(gpu_matvec);
                let matvec = |v: ArrayView1<f64>| -> Array1<f64> { op.apply(v) };
                let eval = plan
                    .evaluate(&matvec, state.cfg.cg_rel_tol, state.cfg.cg_max_iters)
                    .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                        reason: "rational log-det surrogate evaluation returned non-finite"
                            .to_string(),
                    })?;
                let estimate = eval.estimate;
                let derivative_bundle = if want_logdet_derivative {
                    Some(
                        plan.into_directional_derivative_bundle(eval)
                            .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                                reason: "rational log-det derivative bundle assembly failed"
                                    .to_string(),
                            })?,
                    )
                } else {
                    None
                };
                let bundle = if want_bundle {
                    let sinv = reduced_schur_inverse_probe_solves(
                        sys,
                        &htt_factors,
                        ridge_beta,
                        &backend,
                        resident.as_ref(),
                        gpu_matvec,
                        &plan.probes,
                        state.warm_inverse_probes.as_deref(),
                        state.cfg.cg_rel_tol,
                        state.cfg.cg_max_iters,
                    )
                    .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                        reason: "rational surrogate inverse-probe bundle solve failed".to_string(),
                    })?;
                    Some((plan.probes.clone(), sinv))
                } else {
                    None
                };
                (estimate, derivative_bundle, bundle)
            };
            if want_logdet_derivative {
                state.logdet_derivative_bundle = derivative_bundle;
                state.request_logdet_derivative_bundle = false;
            }
            if want_bundle {
                // Keep the fresh solves as the next ρ's warm-start seed (CRN),
                // then hand the bundle to the gradient lane.
                if let Some((_, sinv)) = &bundle {
                    state.warm_inverse_probes = Some(sinv.clone());
                }
                state.inverse_probes = bundle;
                state.request_inverse_probes = false;
            }
            estimate
        }
    };
    Ok((log_det_tt, log_det_schur))
}

/// Power-iteration estimate of the largest eigenvalue `λ_max` of the SPD reduced
/// Schur `S` through the matrix-free [`schur_matvec`] apply — the upper end of
/// the spectral bracket the #2080 rational log-det surrogate
/// ([`RationalLogdetPlan`]) needs to size its bracket-centred DE quadrature.
///
/// Deterministic: the start vector is a fixed SplitMix64 Rademacher draw from
/// `seed`, so a given `(sys, htt_factors, ρ_β, resident, iters, seed)` always
/// returns the same estimate — the surrogate bracket must be reproducible for the
/// REML outer loop, exactly like the SLQ probes. `iters` power steps refine the
/// Rayleigh quotient `vᵀ S v` (each step is one `schur_matvec`); a handful
/// suffice because the surrogate only needs a bracket good to a factor, not a
/// converged eigenvalue (the quadrature window is padded two decades each side).
///
/// Returns `None` for a degenerate operator (`k == 0`) or a non-finite /
/// non-positive Rayleigh quotient (an SPD operator forbids the latter, so it
/// signals a caller bug or a non-finite operator, both of which must surface
/// rather than be silently bracketed).
pub fn reduced_schur_lambda_max<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    iters: usize,
    seed: u64,
) -> Option<f64> {
    let k = sys.k;
    if k == 0 {
        return None;
    }
    // Deterministic Rademacher start (same stream discipline as the surrogate
    // probes): a ±1 vector never lands orthogonal to the top eigenspace.
    let mut v = Array1::<f64>::zeros(k);
    {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut bits: u64 = 0;
        let mut remaining: u32 = 0;
        for value in v.iter_mut() {
            if remaining == 0 {
                bits = gam_linalg::utils::splitmix64(&mut state);
                remaining = 64;
            }
            *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
            bits >>= 1;
            remaining -= 1;
        }
    }
    let inv_norm0 = v.dot(&v).sqrt().recip();
    if !inv_norm0.is_finite() {
        return None;
    }
    v.mapv_inplace(|x| x * inv_norm0);
    // One resident operator reused across every power-iteration apply — device
    // seam threaded so the bracket estimate rides the SAME resident `S·v` the
    // ladder/probes use.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let apply = |x: &Array1<f64>| -> Array1<f64> { op.apply_owned(x) };
    for _ in 0..iters.max(1) {
        let sv = apply(&v);
        let norm = sv.dot(&sv).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            break;
        }
        v = sv / norm;
    }
    // Rayleigh quotient on the converged iterate (v stays unit).
    let sv = apply(&v);
    let lambda = v.dot(&sv);
    (lambda.is_finite() && lambda > 0.0).then_some(lambda)
}

/// Matrix-free reduced-Schur log-determinant `log|S|` via the #2080 fixed
/// rational surrogate ([`RationalLogdetPlan`]) on the exact [`schur_matvec`]
/// apply — the desync-safe companion to [`slq_reduced_schur_log_det`]. **The
/// dense `k×k` `S` is NEVER formed.**
///
/// Returns the built plan and its evaluation so the caller can (a) read
/// `eval.estimate` = the surrogate value `L̃ ≈ log|S|` (with `eval.std_err` the
/// honest Hutchinson error bar), and (b) later contract the SAME shifted-solve
/// bundle against any per-ρ-coordinate Schur-derivative operator `∂S` via
/// [`rational_reduced_schur_directional`]. Because both the value and that
/// derivative are the exact value / gradient of the ONE deterministic function
/// `L̃(ρ)` (fixed probes, fixed quadrature), the outer optimiser descends a
/// function whose gradient is its own — the objective↔gradient desync class the
/// bare SLQ value re-opened (a stochastic value paired with the analytic exact
/// gradient) is closed by construction, not by tolerance tuning.
///
/// The spectral bracket is estimated matrix-free: `λ_max` by power iteration
/// ([`reduced_schur_lambda_max`]), `λ_min` from the deflation-floor convention
/// `SPECTRAL_DEFLATION_REL_FLOOR·λ_max` (the operative lower bound of the
/// unit-deflated spectrum). Deterministic for a fixed
/// `(sys, htt_factors, ρ_β, resident, num_probes, seed, rel_tol, power_iters,
/// cg_rel_tol, cg_max_iters)`.
///
/// `None` when `k == 0`, the bracket estimate is degenerate, the plan cannot be
/// built, or a shifted CG solve breaks down on a non-finite operator.
pub fn rational_reduced_schur_log_det<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    num_probes: usize,
    seed: u64,
    rel_tol: f64,
    power_iters: usize,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<(RationalLogdetPlan, RationalLogdetEval)> {
    let k = sys.k;
    if k == 0 {
        return None;
    }
    let lambda_max = reduced_schur_lambda_max(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        power_iters,
        seed,
    )?;
    // λ_min from the deflation floor: after unit-deflation the operative spectrum
    // is bounded below by `SPECTRAL_DEFLATION_REL_FLOOR·λ_max` (or 1.0), so this
    // is a sound lower bracket for the quadrature window sizing. The window is
    // padded two decades below `λ_min` inside `RationalLogdetPlan::build`, so a
    // conservative (too-small) floor only widens the resolved range, never biases
    // the estimate.
    let lambda_min = (SPECTRAL_DEFLATION_REL_FLOOR * lambda_max).max(f64::MIN_POSITIVE);
    let plan = RationalLogdetPlan::build(k, num_probes, seed, lambda_min, lambda_max, rel_tol)?;
    // One resident operator; the plan's shift ladder reuses it across every
    // shifted solve. The probes fan across rayon workers (in `evaluate`), and
    // `schur_matvec`'s own row parallelism is guarded off inside a worker, so
    // there is no nested oversubscription.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let matvec = |v: ArrayView1<f64>| -> Array1<f64> { op.apply(v) };
    let eval = plan.evaluate(&matvec, cg_rel_tol, cg_max_iters)?;
    Some((plan, eval))
}

/// Build the FROZEN #2080 surrogate plan for one outer solve, with the Hutch++
/// deflation rank DERIVED from a pilot evaluation — the build-once companion to
/// per-ρ [`RationalLogdetPlan::evaluate`]. Returns just the plan (probes +
/// quadrature + frozen Hutch++ `Q`); the caller evaluates it at each ρ, so the
/// expensive rank derivation (several re-solves) is paid ONCE per outer solve,
/// not per criterion evaluation.
///
/// Derived rank (the #2080 lead ruling): a rank-0 pilot fixes the log-det scale,
/// the target bar is `deflation_target_std_err_rel · (|log|S|_pilot| + 1)` — one
/// order under the smallest tolerance the criterion feeds (the caller passes
/// `0.1 · STALL_REL_TOL`; `log|S|` is the criterion's dominant term at wide `k`
/// so `|log|S||+1` is the right objective scale to `O(1)` and the `0.1` margin
/// absorbs the loss/Occam remainder). The peel rank grows on a doubling schedule
/// until the Hutchinson error bar clears the target. `deflation_max_rank` is a
/// resource-admission ceiling, not permission to return an under-certified
/// estimate: exhausting it before the bar clears returns `None` and the caller
/// surfaces a typed evidence failure. `deflation_max_rank == 0` explicitly
/// requests the bare-Hutchinson plan; a pilot already under target also returns
/// it. Deterministic for fixed inputs (`Q` and probes are seed-derived). The
/// returned plan's `Q` is FROZEN, so
/// [`RationalLogdetPlan::directional_derivative`] on its evaluations is the exact
/// surrogate gradient.
pub fn rational_reduced_schur_plan_derived<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    num_probes: usize,
    seed: u64,
    rel_tol: f64,
    power_iters: usize,
    cg_rel_tol: f64,
    cg_max_iters: usize,
    deflation_max_rank: usize,
    deflation_subspace_iters: usize,
    deflation_target_std_err_rel: f64,
) -> Option<RationalLogdetPlan> {
    let k = sys.k;
    if k == 0
        || !(cg_rel_tol.is_finite() && cg_rel_tol > 0.0 && cg_rel_tol < 1.0)
        || !(deflation_target_std_err_rel.is_finite() && deflation_target_std_err_rel >= 0.0)
    {
        return None;
    }
    let lambda_max = reduced_schur_lambda_max(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        power_iters,
        seed,
    )?;
    let lambda_min = (SPECTRAL_DEFLATION_REL_FLOOR * lambda_max).max(f64::MIN_POSITIVE);
    let base_plan =
        RationalLogdetPlan::build(k, num_probes, seed, lambda_min, lambda_max, rel_tol)?;
    // One resident operator across the pilot, every deflation re-solve, and the
    // subspace-iteration `with_two_sided_deflation` applies — the whole rank-derivation
    // ladder (the two-sided deflation: block-power on S + inverse subspace
    // iteration on S⁻¹) reuses the same staged residency / device `S·v`.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let matvec = |v: ArrayView1<f64>| -> Array1<f64> { op.apply(v) };
    // Rank-0 pilot: fixes the |log|S|| scale and is the answer outright when no
    // deflation is requested or the bare bar already clears the target.
    let pilot = base_plan.evaluate(&matvec, cg_rel_tol, cg_max_iters)?;
    if deflation_max_rank == 0 {
        return Some(base_plan);
    }
    let target = deflation_target_std_err_rel * (pilot.estimate.abs() + 1.0);
    if pilot.std_err <= target {
        return Some(base_plan);
    }
    // Grow from the smallest nonzero peel rank (doubling ⇒ log-many re-solves)
    // until the bar clears. The caller's cap is a resource ceiling; reaching it
    // with an over-target bar refuses the surrogate rather than silently
    // weakening the requested statistical-accuracy contract.
    let cap = deflation_max_rank.min(k);
    let mut rank = 1usize;
    // Basis iteration only steers Q for variance reduction. Derive its looser
    // true-residual tolerance from the evaluation solve's tolerance instead of
    // carrying an unrelated fixed knob: √tol is strictly looser while still
    // converging as the bottom-tail builder now requires.
    let basis_cg_rel_tol = cg_rel_tol.sqrt();
    loop {
        let r = rank.min(cap);
        // Split the peel budget across BOTH spectral tails at equal total rank:
        // the Hutchinson bar rides on ‖offdiag(P log(S/c) P)‖_F, whose mass sits
        // symmetrically on the λ_max AND λ_min tails (|log(λ/c)| peaks equally at
        // both ends of the bracket since c is its geometric midpoint), so top-only
        // deflation stalls at ~½ the removable variance
        // (`two_sided_deflation_drops_wide_kappa_std_err_below_two_percent`).
        // The bottom-tail basis comes from inverse iteration — CG on the UNSHIFTED
        // operator at full κ — so it gets its own LOOSE budget, not the
        // evaluation-grade `cg_rel_tol`: an approximate bottom `Q` only relaxes
        // the variance reduction, never biases the value (the split is exact for
        // any orthonormal `Q`), while an evaluation-grade solve there would burn
        // √κ-scale iterations per basis column for no accuracy in return.
        let plan = base_plan.clone().with_two_sided_deflation(
            &matvec,
            r.div_ceil(2),
            r / 2,
            deflation_subspace_iters,
            seed,
            (basis_cg_rel_tol, cg_max_iters),
        )?;
        let eval = plan.evaluate(&matvec, cg_rel_tol, cg_max_iters)?;
        if eval.std_err <= target {
            return Some(plan);
        }
        if r >= cap {
            return None;
        }
        rank = rank.saturating_mul(2);
    }
}

/// Contract the surrogate's shifted-solve bundle from
/// [`rational_reduced_schur_log_det`] against a reduced-Schur derivative operator
/// `∂S` (supplied through its matvec `dmatvec(v) = (∂S)·v`) to obtain the EXACT
/// ρ-derivative of the surrogate value:
/// `∂L̃ = (1/m)·Σ_{j,ℓ} w_ℓ · y_{jℓ}ᵀ (∂S) y_{jℓ}`, `y_{jℓ} = (S+t_ℓ I)⁻¹ v_j`.
///
/// This is the true gradient of the SAME function the value came from — value
/// and gradient can never desync. Thin reduced-Schur wrapper over
/// [`RationalLogdetPlan::directional_derivative`]; the `∂S` matvec is the
/// per-ρ-coordinate Schur-derivative operator the SAE trace channels assemble
/// row-locally (`(∂S)·y = (∂H_ββ)y − Σ_i[ (∂H_βt^(i))(H_tt⁻¹H_tβ y) −
/// H_βt H_tt⁻¹(∂H_tt^(i))H_tt⁻¹H_tβ y + H_βt H_tt⁻¹(∂H_tβ^(i))y ]`).
pub fn rational_reduced_schur_directional(
    plan: &RationalLogdetPlan,
    eval: &RationalLogdetEval,
    dmatvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
) -> Option<f64> {
    plan.directional_derivative(eval, dmatvec)
}

/// Plain CG solve `S y = b` on the SPD reduced Schur through the matrix-free
/// [`schur_matvec`] apply (the `t = 0`, unshifted companion to the surrogate's
/// shifted solves), warm-started from `y0`. Yields `y = S⁻¹ b` — the operator
/// every `tr(S⁻¹·M)` gradient / adjoint channel contracts against at massive K.
/// `None` on a non-finite breakdown (SPD `S` ⇒ that signals a caller bug or a
/// non-finite operator, both of which must surface rather than be swallowed).
fn reduced_schur_cg_solve<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    b: &Array1<f64>,
    y0: &Array1<f64>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<Array1<f64>> {
    // One resident operator reused across every CG apply of this solve — device
    // seam threaded so the inverse-subspace S⁻¹·probe solves ride the resident op.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let apply = |v: &Array1<f64>| -> Array1<f64> { op.apply_owned(v) };
    let quotient = sys.beta_gauge_quotient.as_ref();
    let b = match quotient {
        Some(quotient) => quotient.project_complement(b.view()),
        None => b.clone(),
    };
    let mut y = match quotient {
        Some(quotient) => quotient.project_complement(y0.view()),
        None => y0.clone(),
    };
    let mut r = &b - &apply(&y);
    let b_norm = b.dot(&b).sqrt().max(f64::MIN_POSITIVE);
    let mut p = r.clone();
    let mut rs = r.dot(&r);
    if !rs.is_finite() {
        return None;
    }
    let tol = cg_rel_tol * b_norm;
    let mut iters = 0usize;
    while rs.sqrt() > tol && iters < cg_max_iters {
        let ap = apply(&p);
        let denom = p.dot(&ap);
        if !(denom.is_finite() && denom > 0.0) {
            return None;
        }
        let alpha = rs / denom;
        y.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        let rs_new = r.dot(&r);
        if !rs_new.is_finite() {
            return None;
        }
        p = &r + &(&p * (rs_new / rs));
        rs = rs_new;
        iters += 1;
    }
    Some(match quotient {
        Some(quotient) => quotient.project_complement(y.view()),
        None => y,
    })
}

/// Matrix-free single-rhs reduced-Schur solve `S⁻¹ rhs` (`t = 0`) via CG on
/// [`schur_matvec`], warm-started from `warm` (or cold). The base primitive for
/// the selected-inverse gradient channels whose `S⁻¹` argument is NOT the fixed
/// probe family but a per-call probe-derived vector (e.g. `(H⁻¹)_tt`'s
/// `H_βt(H_tt)⁻¹z` term in the ARD latent-block diagonal, and the per-row
/// `(H⁻¹)_tβ` blocks the θ-adjoint / assignment-strength traces contract) — those
/// cannot reuse the `(probes, S⁻¹·probes)` bundle, so they solve `S⁻¹` on demand
/// through this. `None` on a CG breakdown (SPD `S` forbids it, so it signals a
/// non-finite operator or caller bug).
pub fn reduced_schur_inverse_apply<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    rhs: &Array1<f64>,
    warm: Option<&Array1<f64>>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<Array1<f64>> {
    let zero = Array1::<f64>::zeros(sys.k);
    let y0 = warm.unwrap_or(&zero);
    reduced_schur_cg_solve(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        rhs,
        y0,
        cg_rel_tol,
        cg_max_iters,
    )
}

fn matrix_free_cache_factor_slab(cache: &ArrowFactorCache) -> &ArrowFactorSlab {
    match &cache.htt_factors_undamped {
        ArrowUndampedFactors::SameAsDamped => &cache.htt_factors,
        ArrowUndampedFactors::Owned(factors) => factors,
    }
}

fn validate_matrix_free_arrow_pair(
    sys: &ArrowSchurSystem,
    cache: &ArrowFactorCache,
    operation: &str,
) -> Result<(), ArrowSchurError> {
    if cache.ridge_t != 0.0 || cache.ridge_beta != 0.0 || !cache.schur_factor_is_undamped {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} requires an undamped evidence cache; got ridge_t={}, \
                 ridge_beta={}, schur_factor_is_undamped={}",
                cache.ridge_t, cache.ridge_beta, cache.schur_factor_is_undamped
            ),
        });
    }
    if sys.k != cache.k
        || sys.rows.len() != cache.n_rows()
        || sys.row_dims.as_ref() != cache.row_dims.as_ref()
        || sys.row_offsets.as_ref() != cache.row_offsets.as_ref()
    {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} system/cache layout mismatch: system (rows={}, k={}, offsets={:?}) \
                 vs cache (rows={}, k={}, offsets={:?})",
                sys.rows.len(),
                sys.k,
                sys.row_offsets,
                cache.n_rows(),
                cache.k,
                cache.row_offsets,
            ),
        });
    }
    if sys.row_hessian_fingerprint != cache.row_hessian_fingerprint
        || sys.manifold_mode_fingerprint != cache.manifold_mode_fingerprint
    {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} refuses a stale matrix-free system/cache pair \
                 (row fingerprint {} vs {}, manifold fingerprint {} vs {})",
                sys.row_hessian_fingerprint,
                cache.row_hessian_fingerprint,
                sys.manifold_mode_fingerprint,
                cache.manifold_mode_fingerprint,
            ),
        });
    }
    if !sys.cross_row_penalties.is_empty() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} supports the row-block bordered arrow only; cross-row latent \
                 curvature requires its own matrix-free inverse carrier"
            ),
        });
    }
    if !cache.htbeta_available() && cache.k > 0 {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!("{operation} requires the cached H_tbeta operator"),
        });
    }
    Ok(())
}

fn cholesky_factor_operator_apply(
    factor: ArrayView2<'_, f64>,
    vector: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let n = factor.nrows();
    let mut transposed = Array1::<f64>::zeros(n);
    for col in 0..n {
        let mut value = 0.0_f64;
        for row in col..n {
            value += factor[[row, col]] * vector[row];
        }
        transposed[col] = value;
    }
    let mut out = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut value = 0.0_f64;
        for col in 0..=row {
            value += factor[[row, col]] * transposed[col];
        }
        out[row] = value;
    }
    out
}

/// Apply the undamped full bordered-arrow evidence operator without forming its
/// dense reduced Schur complement.
///
/// The cache supplies the authoritative conditioned row factors and `H_tbeta`
/// operator. The system supplies the matrix-free shared block. Rather than read
/// raw `H_betabeta` directly, this reconstructs it from
/// `S + H_betat A^-1 H_tbeta`, where `S` is applied through the same quotient-
/// aware reduced operator used by the matrix-free log-determinant. Value,
/// selected-inverse traces, and this IFT operator therefore describe one `B`.
pub fn matrix_free_arrow_operator_apply(
    sys: &ArrowSchurSystem,
    cache: &ArrowFactorCache,
    vector_t: ArrayView1<'_, f64>,
    vector_beta: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    validate_matrix_free_arrow_pair(sys, cache, "matrix_free_arrow_operator_apply")?;
    if vector_t.len() != cache.delta_t_len() || vector_beta.len() != cache.k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "matrix_free_arrow_operator_apply vector shapes (t={}, beta={}) != ({}, {})",
                vector_t.len(),
                vector_beta.len(),
                cache.delta_t_len(),
                cache.k,
            ),
        });
    }

    let factors = matrix_free_cache_factor_slab(cache);
    let backend = CpuBatchedBlockSolver;
    let reduced = ReducedSchurOperator::new(sys, factors, 0.0, &backend, None);
    let mut out_beta = reduced.apply(vector_beta);
    let mut out_t = Array1::<f64>::zeros(cache.delta_t_len());
    for row in 0..cache.n_rows() {
        let dim = cache.row_dims[row];
        let start = cache.row_offsets[row];
        let row_vector = vector_t.slice(ndarray::s![start..start + dim]);
        let factor = cache.undamped_factor(row);
        let row_applied = cholesky_factor_operator_apply(factor, row_vector);
        for axis in 0..dim {
            out_t[start + axis] = row_applied[axis];
        }

        if cache.k == 0 {
            continue;
        }
        let mut cross = Array1::<f64>::zeros(dim);
        if !cache.apply_htbeta_row(row, vector_beta, &mut cross) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!("matrix_free_arrow_operator_apply H_tbeta row {row} apply failed"),
            });
        }
        for axis in 0..dim {
            out_t[start + axis] += cross[axis];
        }
        if !cache.apply_htbeta_row_transpose(row, row_vector, &mut out_beta, None) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!("matrix_free_arrow_operator_apply H_betat row {row} apply failed"),
            });
        }

        // `out_beta` already contains `S * vector_beta`; add the eliminated
        // `H_betat A^-1 H_tbeta * vector_beta` term to recover H_betabeta.
        let solved_cross = cholesky_solve_vector(factor, cross.view());
        if !cache.apply_htbeta_row_transpose(row, solved_cross.view(), &mut out_beta, None) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "matrix_free_arrow_operator_apply Schur reconstruction row {row} failed"
                ),
            });
        }
    }
    Ok((out_t, out_beta))
}

/// Solve the undamped full bordered-arrow evidence system for an arbitrary RHS
/// using the matrix-free reduced-Schur CG primitive and exact row backsolves.
///
/// This is the matrix-free sibling of `ArrowFactorCache::full_inverse_apply`.
/// It never materializes `S` or `S^-1`; the beta solve uses the same
/// quotient-aware `S` operator as the rational log-determinant, then the latent
/// block is recovered by standard arrow back-substitution.
pub fn matrix_free_arrow_inverse_apply(
    sys: &ArrowSchurSystem,
    cache: &ArrowFactorCache,
    rhs_t: ArrayView1<'_, f64>,
    rhs_beta: ArrayView1<'_, f64>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    validate_matrix_free_arrow_pair(sys, cache, "matrix_free_arrow_inverse_apply")?;
    if rhs_t.len() != cache.delta_t_len() || rhs_beta.len() != cache.k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "matrix_free_arrow_inverse_apply rhs shapes (t={}, beta={}) != ({}, {})",
                rhs_t.len(),
                rhs_beta.len(),
                cache.delta_t_len(),
                cache.k,
            ),
        });
    }
    if !(cg_rel_tol.is_finite() && cg_rel_tol > 0.0) || cg_max_iters == 0 {
        return Err(ArrowSchurError::PcgFailed {
            reason: format!(
                "matrix_free_arrow_inverse_apply requires positive finite CG tolerance and \
                 iteration count; got rel_tol={cg_rel_tol}, max_iters={cg_max_iters}"
            ),
        });
    }

    let factors = matrix_free_cache_factor_slab(cache);
    let backend = CpuBatchedBlockSolver;
    let mut latent_forward = Array1::<f64>::zeros(cache.delta_t_len());
    let mut eliminated = Array1::<f64>::zeros(cache.k);
    for row in 0..cache.n_rows() {
        let dim = cache.row_dims[row];
        let start = cache.row_offsets[row];
        let solved = cholesky_solve_vector(
            cache.undamped_factor(row),
            rhs_t.slice(ndarray::s![start..start + dim]),
        );
        for axis in 0..dim {
            latent_forward[start + axis] = solved[axis];
        }
        if cache.k > 0
            && !cache.apply_htbeta_row_transpose(row, solved.view(), &mut eliminated, None)
        {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!("matrix_free_arrow_inverse_apply H_betat row {row} apply failed"),
            });
        }
    }
    // The transpose helper accumulates the eliminated term positively.
    let mut reduced_rhs = rhs_beta.to_owned();
    reduced_rhs -= &eliminated;

    let solved_beta = if cache.k == 0 {
        Array1::<f64>::zeros(0)
    } else {
        reduced_schur_inverse_apply(
            sys,
            factors,
            0.0,
            &backend,
            None,
            None,
            &reduced_rhs,
            None,
            cg_rel_tol,
            cg_max_iters,
        )
        .ok_or_else(|| ArrowSchurError::PcgFailed {
            reason: format!(
                "matrix_free_arrow_inverse_apply reduced-Schur solve failed \
                 (dim={}, rel_tol={cg_rel_tol}, max_iters={cg_max_iters})",
                cache.k
            ),
        })?
    };

    let mut solved_t = latent_forward;
    for row in 0..cache.n_rows() {
        let dim = cache.row_dims[row];
        let start = cache.row_offsets[row];
        if cache.k == 0 {
            continue;
        }
        let mut cross = Array1::<f64>::zeros(dim);
        if !cache.apply_htbeta_row(row, solved_beta.view(), &mut cross) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!("matrix_free_arrow_inverse_apply H_tbeta row {row} apply failed"),
            });
        }
        let correction = cholesky_solve_vector(cache.undamped_factor(row), cross.view());
        for axis in 0..dim {
            solved_t[start + axis] -= correction[axis];
        }
    }
    Ok((solved_t, solved_beta))
}

/// The `S⁻¹ v_j` bundle for a fixed probe set: solves `S y_j = v_j` (`t = 0`) on
/// the matrix-free reduced Schur for each probe `v_j`, warm-started per-probe
/// from `warm` when supplied (e.g. the surrogate's smallest-shift solves, which
/// already sit close to `S⁻¹ v_j`). Computed ONCE per outer solve and reused
/// across every `tr(S⁻¹·M)` channel, so the whole massive-K ρ-gradient +
/// θ-adjoint rides on one probe family — one functional, desync closed.
///
/// `probes` are the surrogate plan's Rademacher probes (`RationalLogdetPlan::
/// probes`); pass the SAME set the value used so the trace estimates are
/// consistent with it. `None` on any CG breakdown.
pub fn reduced_schur_inverse_probe_solves<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    probes: &[Array1<f64>],
    warm: Option<&[Array1<f64>]>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<Vec<Array1<f64>>> {
    let k = sys.k;
    let zero = Array1::<f64>::zeros(k);
    let mut out = Vec::with_capacity(probes.len());
    for (j, v) in probes.iter().enumerate() {
        let y0 = warm.and_then(|w| w.get(j)).unwrap_or(&zero);
        let y = reduced_schur_cg_solve(
            sys,
            htt_factors,
            ridge_beta,
            backend,
            resident,
            gpu_matvec,
            v,
            y0,
            cg_rel_tol,
            cg_max_iters,
        )?;
        out.push(y);
    }
    Some(out)
}

/// Hutchinson estimate `tr(S⁻¹ M) ≈ (1/m) Σ_j (S⁻¹ v_j)ᵀ (M v_j)` for the reduced
/// Schur `S` and a SYMMETRIC channel operator `M` supplied by its matvec
/// `m_matvec(v) = M·v`. `sinv_probes[j] = S⁻¹ v_j` is the bundle from
/// [`reduced_schur_inverse_probe_solves`] and `probes` the matching probe set.
///
/// The general umbrella (#2080): every dense-`S⁻¹` consumer in the SAE outer
/// gradient — the per-row selected-inverse deflation corrections
/// (`M = Σ_i G_iᵀ C_i G_i`), the direct β–β contractions (`M = ∂H_ββ` channel),
/// and the θ-adjoint — is ultimately a `tr(S⁻¹·M)` with `M·v` computable
/// row-locally without forming `M`. Estimating them all from the SAME
/// `(probes, S⁻¹ v_j)` pair keeps the value, ρ-gradient, and θ-adjoint one
/// functional. Unbiased for the ±1 Rademacher probes (`E[vᵀ S⁻¹ M v] =
/// tr(S⁻¹ M)`). `None` on a length mismatch or a non-finite accumulation.
pub fn hutchinson_reduced_schur_inverse_trace(
    probes: &[Array1<f64>],
    sinv_probes: &[Array1<f64>],
    m_matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
) -> Option<f64> {
    let m = probes.len();
    if m == 0 || sinv_probes.len() != m {
        return None;
    }
    let mut acc = 0.0_f64;
    for (v, y) in probes.iter().zip(sinv_probes) {
        let mv = m_matvec(v.view());
        acc += y.dot(&mv);
    }
    acc /= m as f64;
    acc.is_finite().then_some(acc)
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
            // Dense-slab fast path (#1017): when the per-row cross-block is a
            // materialized `di × k` slab (no matrix-free operator), the entire
            // reduced-Schur diagonal contribution for this row is
            // `Σ_c H_tβ[c,a] · ((H_tt)⁻¹ H_tβ)[c,a]`. The generic loop below
            // re-solved `(H_tt)⁻¹` once PER COLUMN — `O(k)` block solves + `O(k)`
            // allocations per row, i.e. `O(n·k)` tiny solves per Newton step
            // (the dominant fixed per-solve cost at the SAE wide-border shape,
            // k in the tens of thousands). Solve all `k` columns in ONE batched
            // block solve instead, then take the column dots. Reassociates the
            // diagonal within the documented #1211 preconditioner margin (same as
            // the resident no-probe path), and the preconditioner only steers the
            // PCG iterate, which still terminates at the PCG tolerance.
            if sys.htbeta_matvec.is_none() && row.htbeta.dim() == (di, k) {
                let solved = backend.solve_block_matrix(htt_factors.factor(i), row.htbeta.view());
                for a in 0..k {
                    let mut acc = 0.0;
                    for c in 0..di {
                        acc += row.htbeta[[c, a]] * solved[[c, a]];
                    }
                    diag_part[a] -= acc;
                }
                return;
            }
            // Matrix-free path: probe column a. `e_a` stays all-zero between
            // columns — set the single active entry and reset it after the probe,
            // so we never pay the `O(k)` `e_a.fill(0.0)` per column (that fill was
            // `O(n·k²)`). `sys_htbeta_apply_row` zeroes `col_i` internally.
            let mut col_i = Array1::<f64>::zeros(di);
            let mut e_a = Array1::<f64>::zeros(k);
            for a in 0..k {
                e_a[a] = 1.0;
                sys_htbeta_apply_row(sys, i, row, e_a.view(), &mut col_i);
                e_a[a] = 0.0;
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
            // `L_i` is the shared `local_jac[row]` slab (#1033) — byte-for-byte
            // the former per-row `rf.l` copy.
            let l_i = &resident.local_jac[row];
            for (j, slot) in col_dot.iter_mut().enumerate().take(p) {
                let mut acc = 0.0_f64;
                for r in 0..di {
                    let idx = r * p + j;
                    acc += l_i[idx] * rf.y[idx];
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
            // `L_i` is the shared `local_jac[row]` slab (#1033) — byte-for-byte
            // the former per-row `rf.l` copy.
            let l_i = &resident.local_jac[row];
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
                                    gij += l_i[r * p + ch_i] * rf.y[r * p + ch_j];
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
///   overlapping columns averaged by partition-of-unity weights (full dense
///   local-inverse apply per subdomain).
/// - `DiagAssembledSchwarz { overlap }`: the cheap Schwarz variant (#299) —
///   same overlapping decomposition, but each subdomain contributes only the
///   diagonal of its local inverse `(A_k⁻¹)_ii`, assembled additively with
///   partition-of-unity weights into a single `O(K)`-apply diagonal.
/// - `BlockIncompleteCholesky`: level-0 incomplete Cholesky (#299). Within each
///   connected component of the β-coupling graph the dense reduced-Schur block
///   `S[C,C]` is assembled once, its structural-nonzero pattern is taken as the
///   level-0 fill pattern, and a no-fill incomplete Cholesky `S ≈ L̃ L̃ᵀ` is
///   formed keeping ONLY that pattern (Saad, *Iterative Methods*, IC(0)). Apply
///   is a sparse triangular forward/back solve over `nnz(S[C,C])`, so for a
///   large component with internal sparsity it is far cheaper to build and apply
///   than `ClusterJacobi`'s full dense Cholesky (which fills the whole `b×b`
///   factor) while retaining the inter-block coupling that ClusterJacobi keeps
///   but the diagonal/Schwarz tiers discard. A non-PD incomplete pivot degrades
///   that component to the scalar reciprocal diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurPreconditionerKind {
    Diagonal,
    BetaBlockJacobi,
    ClusterJacobi,
    /// Cluster-Jacobi whose blocks come from the bounded co-visibility PARTITION
    /// (`BetaCouplingGraph::covisibility_cluster_partition`) rather than the
    /// connected-component partition. At real over-complete widths the co-firing
    /// graph is a single giant component, so plain `ClusterJacobi` exceeds the
    /// size cap and degrades to scalar Jacobi; this tier splits that component
    /// into bounded strongly-co-firing clusters so the dense per-cluster factor
    /// conditions the cross-atom coupling scalar Jacobi cannot see.
    CoVisibilityClusterJacobi,
    AdditiveSchwarz {
        overlap: usize,
    },
    DiagAssembledSchwarz {
        overlap: usize,
    },
    BlockIncompleteCholesky,
}

/// Escalate beyond BetaBlockJacobi only when K exceeds this value and PCG
/// exhausted `max_iterations`.
pub(crate) const PRECOND_ESCALATE_K_THRESHOLD: usize = 100;

/// #1026 matrix-free Schur curvature-floor (the unbounded-PCG analogue of the
/// dense `spectral_pd_floored_schur`). On `pᵀSp ≤ 0` in the unbounded SAE inner
/// PCG, the operator ridge is lifted by the minimal amount that restores
/// positive curvature along the offending direction, plus this fractional
/// margin (so the next CG iterate sits strictly inside the positive cone, not on
/// the `0` knife-edge).
pub(crate) const SCHUR_CURVATURE_FLOOR_MARGIN: f64 = 1.0e-2;
/// Lower bound on the curvature-floor ridge bump, relative to the rhs scale, so
/// a `pᵀSp` that rounds to exactly `0` still gets a strictly positive bump.
pub(crate) const SCHUR_CURVATURE_FLOOR_REL_FLOOR: f64 = 1.0e-12;
/// Ceiling on the accumulated curvature-floor ridge, relative to the rhs scale.
/// Beyond this the operator is treated as un-conditionable by a minimal floor
/// and the recoverable failure is handed to the outer LM loop (which re-forms
/// the whole system at a heavier ridge). Generous so that a large collapsed
/// over-subtraction `(H_tβ)²/H_tt` is still reachable.
pub(crate) const SCHUR_CURVATURE_FLOOR_REL_CEILING: f64 = 1.0e12;
/// Multiplicative growth for the DIAGONAL-refusal ridge escalation (no
/// `(curvature, ‖p‖²)` deficit is available there), matching the per-row
/// `factor_one_row_result` `RIDGE_GROWTH_FACTOR`.
pub(crate) const SCHUR_CURVATURE_FLOOR_DIAG_GROWTH: f64 = 10.0;
/// Max curvature-floor ridge-lift attempts before deferring to the outer LM
/// loop. The diagonal-refusal path grows ×10 per attempt, so this bounds the
/// reachable ridge at `rhs_scale · 10^(attempts)` — ample for any realistic
/// over-subtraction while still bounded.
pub(crate) const SCHUR_CURVATURE_FLOOR_MAX_ATTEMPTS: usize = 24;

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

/// Host-memory budget for ONE cluster's dense reduced-Schur Cholesky factor
/// (the `b×b` f64 `L` the cluster-Jacobi preconditioner stores and applies).
///
/// The co-visibility cluster partition caps a cluster's total column count `b`
/// at the largest value whose factor fits this budget, `b_max = ⌊√(budget/8)⌋`
/// (`8b²` bytes for an `f64` `b×b` factor). This DERIVES the cluster-size cap
/// from the factor's memory footprint rather than asserting a bare number:
/// beyond `b_max` the dense factor's `O(b²)` apply also throttles the CG
/// iteration budget, so the cap is the point past which a single dense block
/// stops being the right preconditioner and the partition must split instead.
/// 2 MiB ⇒ `b_max = 512`, pinned equal to [`CLUSTER_JACOBI_MAX_CLUSTER`] by
/// [`tests::covisibility_cap_is_derived_from_factor_budget`] so the co-visibility
/// partition and the legacy scalar-fallback ceiling agree by construction.
pub(crate) const CLUSTER_SCHUR_FACTOR_BYTES_BUDGET: u128 = 2 * 1024 * 1024;

/// Derived co-visibility cluster-size cap (columns): the largest `b` whose dense
/// `b×b` f64 Cholesky factor fits [`CLUSTER_SCHUR_FACTOR_BYTES_BUDGET`]. See that
/// constant for the memory justification. Never below 1.
pub(crate) fn covisibility_cluster_max_cols() -> usize {
    let b = ((CLUSTER_SCHUR_FACTOR_BYTES_BUDGET / 8) as f64)
        .sqrt()
        .floor() as usize;
    b.max(1)
}

/// Maximum columns in a single connected component for which the IC(0)
/// preconditioner assembles the dense `S[C,C]` to derive its sparsity pattern.
/// IC(0) is cheap to APPLY at any size, but the pattern is read from the dense
/// assembly, which is `O(b²)` memory; beyond this the component falls back to
/// the scalar reciprocal diagonal (the same ceiling concern as
/// `CLUSTER_JACOBI_MAX_CLUSTER`, lifted because the IC(0) FACTOR is sparse).
pub(crate) const IC0_MAX_COMPONENT: usize = 4096;

/// Relative threshold below which an assembled `S[i,j]` is treated as a
/// structural zero when deriving the IC(0) level-0 pattern. Scaled by
/// `sqrt(|S_ii|·|S_jj|)` so it is invariant to column scaling; this prunes
/// entries that are pure FMA round-off (a genuinely decoupled `(i,j)` pair
/// assembles to ~0) so they do not enter the kept fill pattern.
pub(crate) const IC0_PATTERN_REL_DROP: f64 = 1.0e-13;

/// Assemble the dense `b×b` reduced-Schur block for the column set `cols`:
/// `S[cols, cols] = H_ββ[cols, cols] + ridge·I − Σ_i H_tβ[cols]ᵀ (H_tt^i)⁻¹ H_tβ[cols]`.
///
/// Shared by `ClusterJacobiPreconditioner::build_from_column_groups` (which
/// Cholesky-factors the returned block) and `DiagAssembledSchwarzPreconditioner`
/// (which inverts each subdomain block and keeps only its diagonal). The result
/// is the LOWER triangle filled by the row reduction; callers that need the full
/// symmetric block must `symmetrize_upper_from_lower`.
///
/// The per-row Schur contribution is fanned over fixed 64-row chunks above
/// `SCHUR_MATVEC_PARALLEL_ROW_MIN` and folded left-to-right so the assembly is
/// bit-identical to the serial path (and run-to-run deterministic), exactly as
/// in `build_block_jacobi` (#1017).
pub(crate) fn assemble_local_schur_block<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    cols: &[usize],
) -> Array2<f64> {
    let b = cols.len();
    let mut s_block = Array2::<f64>::zeros((b, b));
    // Initialise from H_ββ via penalty_subblock_add (#296): routes through
    // penalty_op or falls back to hbb / hbb_diag inline.
    sys.penalty_subblock_add(cols, &mut s_block);
    for bi in 0..b {
        s_block[[bi, bi]] += ridge_beta;
    }
    let cluster_row_into = |row_idx: usize, row: &ArrowRowBlock, acc: &mut Array2<f64>| {
        // Materialize the b needed cross-block columns through the ROUTED
        // `H_tβ` convention (`sys_htbeta_apply_row`: matrix-free operator plus
        // any dense supplement) at the row's OWN width `di` — never a raw
        // `row.htbeta` read at the global `sys.d`: matvec-backed rows carry
        // absent/zero-sized slabs by contract (a raw read is wrong or panics),
        // and per-row widths vary.
        let di = sys.row_dims[row_idx];
        let mut e_g = Array1::<f64>::zeros(sys.k);
        let mut col_i = Array1::<f64>::zeros(di);
        let mut cols_mat = Array2::<f64>::zeros((di, b));
        let mut solved_cols = Array2::<f64>::zeros((di, b));
        for bj in 0..b {
            let gj = cols[bj];
            e_g[gj] = 1.0;
            sys_htbeta_apply_row(sys, row_idx, row, e_g.view(), &mut col_i);
            e_g[gj] = 0.0;
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_i.view());
            for c in 0..di {
                cols_mat[[c, bj]] = col_i[c];
                solved_cols[[c, bj]] = solved[c];
            }
        }
        for bi in 0..b {
            for bj in 0..b {
                let mut dot = 0.0;
                for c in 0..di {
                    dot += cols_mat[[c, bi]] * solved_cols[[c, bj]];
                }
                acc[[bi, bj]] -= dot;
            }
        }
    };
    let n = sys.rows.len();
    let parallel = n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
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
    s_block
}

/// Column groups for the bounded co-visibility cluster preconditioner.
///
/// Builds the weighted co-firing graph over `sys.block_offsets` and returns the
/// column sets of its bounded co-visibility partition
/// (`BetaCouplingGraph::covisibility_cluster_partition`), each capped at
/// [`covisibility_cluster_max_cols`] columns. With no registered block offsets
/// there is no block structure to cluster, so the whole `0..k` border is one
/// group (identical to the component-partition builders' `block_offsets`-empty
/// case). Each group's columns are sorted ascending.
pub(crate) fn covisibility_column_groups(sys: &ArrowSchurSystem) -> Vec<Vec<usize>> {
    if sys.block_offsets.is_empty() {
        return vec![(0..sys.k).collect()];
    }
    let graph = BetaCouplingGraph::build(
        &sys.block_offsets,
        &sys.rows
            .iter()
            .map(|r| r.htbeta.clone())
            .collect::<Vec<_>>(),
    );
    graph
        .covisibility_cluster_partition(&sys.block_offsets, covisibility_cluster_max_cols())
        .iter()
        .map(|blocks| {
            let mut cols: Vec<usize> = blocks
                .iter()
                .flat_map(|&b| sys.block_offsets[b].clone())
                .collect();
            cols.sort_unstable();
            cols
        })
        .collect()
}

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

    /// Cluster-Jacobi from the bounded CO-VISIBILITY partition (Kushal & Agarwal,
    /// CVPR 2012) — the default above the size cap.
    ///
    /// [`Self::from_arrow_schur`] groups β-blocks by CONNECTED COMPONENT of the
    /// co-firing graph. At real over-complete SAE widths that graph is a single
    /// giant component (transitive co-firing), so the lone component's column
    /// count exceeds [`CLUSTER_JACOBI_MAX_CLUSTER`] and
    /// [`Self::build_from_column_groups`] degrades the whole tier to the scalar
    /// reciprocal diagonal — the scaling ceiling (cross-atom coupling through
    /// co-activating atoms with overlapping ambient subspaces is dropped, and PCG
    /// iteration counts blow up). This builder instead partitions the co-firing
    /// graph into clusters bounded by [`covisibility_cluster_max_cols`], keeping
    /// the strongest co-firing edges inside a cluster, so each cluster's dense
    /// Cholesky conditions the strong cross-atom coupling the scalar diagonal
    /// misses while staying inside the per-factor memory budget.
    ///
    /// With no registered `block_offsets` (or a graph that fits the cap in one
    /// piece) the partition is a single group and this coincides with
    /// [`Self::from_arrow_schur`]. Because the preconditioner only steers the CG
    /// iterate over the SAME reduced operator, the solve converges to the SAME
    /// reduced-system solution regardless of the partition — REML-neutral.
    pub(crate) fn from_arrow_schur_covisibility<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let col_groups = covisibility_column_groups(sys);
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    pub(crate) fn build_from_column_groups<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        col_groups: &[Vec<usize>],
    ) -> Result<Self, ArrowSchurError> {
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
            let mut s_block =
                assemble_local_schur_block(sys, htt_factors, ridge_beta, backend, cols);
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

/// Diagonal-assembled additive Schwarz (#299).
///
/// The cheap Schwarz variant the domain-decomposition literature recommends as
/// the default for sparse-coupling β-graphs: instead of storing and applying a
/// dense Cholesky factor per overlapping subdomain (as
/// [`AdditiveSchwarzPreconditioner`] does), it inverts each overlapping
/// subdomain Schur block ONCE at build time and keeps only the **diagonal of the
/// local inverse** `(A_k⁻¹)_ii`. Those per-subdomain diagonal contributions are
/// then assembled additively across overlapping subdomains with partition-of-
/// unity weights into a single global diagonal `m`, applied as `out[i] = m[i]·r[i]`.
///
/// This is strictly richer than scalar Jacobi (`1/S_ii`): the local inverse
/// diagonal `(A_k⁻¹)_ii` folds in the off-diagonal coupling WITHIN the subdomain,
/// so a strongly-coupled column gets a smaller (better-damped) effective scale
/// than its bare reciprocal diagonal would give — while the apply stays `O(K)`
/// (one multiply per column), unlike the `O(Σ b_k²)` triangular solves of dense
/// Schwarz. For `overlap = 0` and one column per subdomain it reduces exactly to
/// scalar Jacobi.
#[derive(Debug, Clone)]
pub struct DiagAssembledSchwarzPreconditioner {
    /// Global per-column multiplier `m[i]`; `out[i] = m[i] · r[i]`.
    pub(crate) inv_diag: Vec<f64>,
}

impl DiagAssembledSchwarzPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        overlap: usize,
    ) -> Result<Self, ArrowSchurError> {
        // Build the overlapping subdomain column groups exactly like
        // AdditiveSchwarz (component partition + `overlap` graph-hop expansion),
        // so the two Schwarz variants decompose the β space identically and
        // differ only in how each subdomain's local inverse is applied.
        let col_groups: Vec<Vec<usize>> = if sys.block_offsets.is_empty() {
            vec![(0..sys.k).collect()]
        } else {
            let graph = BetaCouplingGraph::build(
                &sys.block_offsets,
                &sys.rows
                    .iter()
                    .map(|r| r.htbeta.clone())
                    .collect::<Vec<_>>(),
            );
            graph
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
                .collect()
        };
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    pub(crate) fn build_from_column_groups<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        col_groups: &[Vec<usize>],
    ) -> Result<Self, ArrowSchurError> {
        // Partition-of-unity weights: a column shared by `c` subdomains gets each
        // of its `c` diagonal contributions scaled by `1/c`, so the assembled
        // diagonal is a convex combination (and reduces to a single contribution
        // for non-overlapping columns).
        let mut counts = vec![0u32; sys.k];
        for cols in col_groups {
            for &gi in cols {
                counts[gi] += 1;
            }
        }
        let mut accum = vec![0.0f64; sys.k];
        for cols in col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            // For large subdomains, the dense inverse is too costly; fall back to
            // the global scalar Schur diagonal inverse `1/S_ii` for those columns
            // (the diag-assembled variant then coincides with scalar Jacobi over
            // that subdomain, which is exactly the intended cheap degradation).
            if b > CLUSTER_JACOBI_MAX_CLUSTER {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                for (local, &gi) in cols.iter().enumerate() {
                    let w = if counts[gi] == 0 {
                        1.0
                    } else {
                        1.0 / counts[gi] as f64
                    };
                    accum[gi] += w * inv[local];
                }
                continue;
            }
            let mut s_block =
                assemble_local_schur_block(sys, htt_factors, ridge_beta, backend, cols);
            symmetrize_upper_from_lower(&mut s_block);
            // Diagonal of the local inverse `(A_k⁻¹)_ii`, obtained by solving
            // `A_k X = I` through the same faer Cholesky used elsewhere; on a
            // non-PD local block, degrade to the scalar reciprocal diagonal.
            let local_inv_diag = match local_inverse_diagonal(&s_block) {
                Some(diag) => diag,
                None => {
                    let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                    inv
                }
            };
            for (local, &gi) in cols.iter().enumerate() {
                let w = if counts[gi] == 0 {
                    1.0
                } else {
                    1.0 / counts[gi] as f64
                };
                accum[gi] += w * local_inv_diag[local];
            }
        }
        // A column never covered by any subdomain (only possible for `k` columns
        // with no block_offsets coverage) keeps a neutral unit scale.
        for (gi, &c) in counts.iter().enumerate() {
            if c == 0 {
                accum[gi] = 1.0;
            }
        }
        for (gi, m) in accum.iter().enumerate() {
            if !m.is_finite() || *m <= 0.0 {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "diag-assembled Schwarz: non-positive assembled diagonal at index {gi}: {m}"
                    ),
                });
            }
        }
        Ok(Self { inv_diag: accum })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for (gi, &m) in self.inv_diag.iter().enumerate() {
            out[gi] = m * r[gi];
        }
        out
    }
}

/// Diagonal of `A⁻¹` for a small dense SPD block `A`, via the same faer
/// Cholesky used by the cluster/Schwarz factors. Returns `None` if `A` is not
/// positive-definite (caller degrades to the scalar reciprocal diagonal).
pub(crate) fn local_inverse_diagonal(a: &Array2<f64>) -> Option<Vec<f64>> {
    let b = a.nrows();
    let llt = {
        use faer::Side;
        let view = FaerArrayView::new(a);
        FaerLlt::new(view.as_ref(), Side::Lower).ok()?
    };
    use faer::linalg::solvers::Solve;
    let mut diag = Vec::with_capacity(b);
    for col in 0..b {
        // Solve `A x = e_col`; the `col`-th entry of `x` is `(A⁻¹)_{col,col}`.
        let mut rhs = Array1::<f64>::zeros(b);
        rhs[col] = 1.0;
        let stride = rhs.strides()[0];
        let len = rhs.len();
        // SAFETY: `rhs` is a uniquely-borrowed contiguous `Array1<f64>` of `len`
        // elements with positive row stride; a single column never dereferences
        // the column stride, so `0` is sound.
        let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
        let solved = llt.solve(rhs_mat);
        diag.push(solved[(col, 0)]);
    }
    Some(diag)
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

/// One connected-component factor of the block IC(0) preconditioner.
///
/// `IncompleteChol` holds a sparse lower-triangular `L̃` in column-compressed
/// form over the component's local indices: `col_ptr[j]..col_ptr[j+1]` indexes
/// into `(row_idx, val)` for column `j` (rows `>= j`, diagonal first). `cols`
/// maps a local index back to its global β column. `Scalar` is the non-PD /
/// oversized degradation, identical in meaning to [`ClusterFactor::Scalar`].
#[derive(Clone)]
pub(crate) enum Ic0Factor {
    IncompleteChol {
        cols: Vec<usize>,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        val: Vec<f64>,
    },
    Scalar {
        cols: Vec<usize>,
        inv: Vec<f64>,
    },
}

impl std::fmt::Debug for Ic0Factor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ic0Factor::IncompleteChol { cols, val, .. } => write!(
                f,
                "Ic0Factor::IncompleteChol {{ cols.len: {}, nnz: {} }}",
                cols.len(),
                val.len()
            ),
            Ic0Factor::Scalar { cols, .. } => {
                write!(f, "Ic0Factor::Scalar {{ cols.len: {} }}", cols.len())
            }
        }
    }
}

/// Level-0 incomplete-Cholesky Schur preconditioner (#299).
///
/// One sparse incomplete-Cholesky factor per connected component of the
/// β-coupling graph. Within a component the dense `S[C,C]` is assembled, its
/// structural-nonzero pattern `P = { (i,j) : |S_ij| > drop·sqrt(S_ii S_jj) }`
/// is taken as the level-0 fill set, and the no-fill incomplete Cholesky
/// `S ≈ L̃ L̃ᵀ` is formed keeping only `P` (drop any update landing outside it).
/// See [`SchurPreconditionerKind::BlockIncompleteCholesky`].
#[derive(Debug, Clone)]
pub struct BlockIncompleteCholeskyPreconditioner {
    pub(crate) components: Vec<Ic0Factor>,
}

impl BlockIncompleteCholeskyPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        // Column grouping mirrors ClusterJacobi: one group per connected
        // component of the β-coupling graph (whole-K single group when no
        // block_offsets are registered), so IC(0) preconditions exactly the
        // coupling ClusterJacobi keeps, but with a sparse (no-fill) factor.
        let col_groups: Vec<Vec<usize>> = if sys.block_offsets.is_empty() {
            vec![(0..sys.k).collect()]
        } else {
            let graph = BetaCouplingGraph::build(
                &sys.block_offsets,
                &sys.rows
                    .iter()
                    .map(|r| r.htbeta.clone())
                    .collect::<Vec<_>>(),
            );
            graph
                .component_partition()
                .iter()
                .map(|comp| {
                    let mut cols: Vec<usize> = comp
                        .iter()
                        .flat_map(|&blk| sys.block_offsets[blk].clone())
                        .collect();
                    cols.sort_unstable();
                    cols.dedup();
                    cols
                })
                .collect()
        };

        let mut components = Vec::with_capacity(col_groups.len());
        for cols in &col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            if b > IC0_MAX_COMPONENT {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                components.push(Ic0Factor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
                continue;
            }
            let mut s_block =
                assemble_local_schur_block(sys, htt_factors, ridge_beta, backend, cols);
            symmetrize_upper_from_lower(&mut s_block);
            match incomplete_cholesky_level0(&s_block) {
                Some((col_ptr, row_idx, val)) => components.push(Ic0Factor::IncompleteChol {
                    cols: cols.clone(),
                    col_ptr,
                    row_idx,
                    val,
                }),
                None => {
                    // Non-PD incomplete pivot: degrade this component to the
                    // scalar reciprocal diagonal (mirrors the ClusterJacobi
                    // non-PD fallback), which is always applicable for a
                    // PD-floored Schur diagonal.
                    let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                    components.push(Ic0Factor::Scalar {
                        cols: cols.clone(),
                        inv,
                    });
                }
            }
        }
        Ok(Self { components })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for comp in &self.components {
            match comp {
                Ic0Factor::Scalar { cols, inv } => {
                    for (local, &gi) in cols.iter().enumerate() {
                        out[gi] = inv[local] * r[gi];
                    }
                }
                Ic0Factor::IncompleteChol {
                    cols,
                    col_ptr,
                    row_idx,
                    val,
                } => {
                    let b = cols.len();
                    // Gather the local residual, solve `L̃ L̃ᵀ z = r_local` by a
                    // sparse forward solve (`L̃ y = r`) then a sparse back solve
                    // (`L̃ᵀ z = y`), then scatter `z` back to global columns.
                    let mut z = vec![0.0f64; b];
                    for (local, &gi) in cols.iter().enumerate() {
                        z[local] = r[gi];
                    }
                    // Forward solve `L̃ y = r` (overwrite z with y). Column-major
                    // CSC: row_idx[col_ptr[j]] == j (diagonal stored first).
                    for j in 0..b {
                        let dstart = col_ptr[j];
                        let diag = val[dstart];
                        z[j] /= diag;
                        let yj = z[j];
                        for k in (dstart + 1)..col_ptr[j + 1] {
                            z[row_idx[k]] -= val[k] * yj;
                        }
                    }
                    // Back solve `L̃ᵀ z = y` (overwrite z). Walk columns in
                    // reverse; the below-diagonal entries of column j are the
                    // off-diagonal entries of row j of L̃ᵀ.
                    for j in (0..b).rev() {
                        let dstart = col_ptr[j];
                        let mut acc = z[j];
                        for k in (dstart + 1)..col_ptr[j + 1] {
                            acc -= val[k] * z[row_idx[k]];
                        }
                        z[j] = acc / val[dstart];
                    }
                    for (local, &gi) in cols.iter().enumerate() {
                        out[gi] = z[local];
                    }
                }
            }
        }
        out
    }
}

/// Level-0 incomplete Cholesky of a dense SPD-ish block `a` (`b×b`, symmetric).
///
/// Returns the lower factor `L̃` in column-compressed (CSC) form
/// `(col_ptr, row_idx, val)` where each column lists its diagonal entry FIRST
/// followed by the strictly-below-diagonal entries, in increasing row order.
/// The kept pattern is the level-0 set `P` = structural nonzeros of `a` (a
/// relative drop threshold prunes round-off). IC(0) computes the standard
/// Cholesky recurrence but DROPS any value at a position outside `P`, so the
/// factor has exactly `nnz(tril(P))` entries — no fill. Returns `None` on a
/// non-positive pivot (caller degrades to scalar diagonal).
///
/// Reference: Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
/// §10.3.2 (IC(0)). This is the left-looking, pattern-restricted variant.
pub(crate) fn incomplete_cholesky_level0(
    a: &Array2<f64>,
) -> Option<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let b = a.nrows();
    assert_eq!(a.ncols(), b, "incomplete Cholesky needs a square block");

    // ---- derive the level-0 lower-triangular pattern from `a` --------------
    // Per column j, the kept below-or-on-diagonal rows i>=j with a structurally
    // nonzero a[i,j]. The diagonal is always kept.
    let mut col_ptr = vec![0usize; b + 1];
    let mut row_idx: Vec<usize> = Vec::new();
    // value buffer, parallel to row_idx, initialised from tril(a) on the pattern
    let mut val: Vec<f64> = Vec::new();
    // For O(1) "is (i,j) in pattern + where" lookups during the recurrence, keep
    // a per-column map from global row -> position in that column's value slice.
    let mut col_pos: Vec<std::collections::HashMap<usize, usize>> = Vec::with_capacity(b);
    for j in 0..b {
        let ajj = a[[j, j]];
        let scale_j = ajj.abs().max(0.0).sqrt();
        let mut map = std::collections::HashMap::new();
        // diagonal first
        map.insert(j, val.len());
        row_idx.push(j);
        val.push(ajj);
        for i in (j + 1)..b {
            let aij = a[[i, j]];
            let scale_i = a[[i, i]].abs().sqrt();
            let thresh = IC0_PATTERN_REL_DROP * scale_i * scale_j;
            if aij.abs() > thresh {
                map.insert(i, val.len());
                row_idx.push(i);
                val.push(aij);
            }
        }
        col_pos.push(map);
        col_ptr[j + 1] = val.len();
    }

    // ---- IC(0) recurrence, left-looking over columns -----------------------
    // For column j: subtract the contributions of all prior columns k<j that
    // have BOTH a nonzero at row j (so they touch the diagonal/the column) — the
    // multiplier L[j,k] — and a nonzero at the rows i of column j's pattern.
    // Any update whose target (i,j) is OUTSIDE the kept pattern is dropped.
    for j in 0..b {
        // Diagonal: a[j,j] - Σ_{k<j} L[j,k]². Each prior column k<j contributes
        // its row-j entry L[j,k] (looked up by row, so the column index is not
        // needed); columns without a row-j entry contribute nothing.
        let dpos = col_ptr[j];
        let mut diag = val[dpos];
        for mapk in &col_pos[..j] {
            if let Some(&pjk) = mapk.get(&j) {
                let ljk = val[pjk];
                diag -= ljk * ljk;
            }
        }
        if !diag.is_finite() || diag <= JACOBI_DIAGONAL_PD_FLOOR {
            return None;
        }
        let ljj = diag.sqrt();
        val[dpos] = ljj;
        // Below-diagonal of column j: L[i,j] = (a[i,j] - Σ_{k<j} L[i,k] L[j,k]) / L[j,j]
        for p in (dpos + 1)..col_ptr[j + 1] {
            let i = row_idx[p];
            let mut s = val[p];
            for mapk in &col_pos[..j] {
                if let (Some(&pik), Some(&pjk)) = (mapk.get(&i), mapk.get(&j)) {
                    s -= val[pik] * val[pjk];
                }
            }
            val[p] = s / ljj;
        }
    }
    Some((col_ptr, row_idx, val))
}

/// One row of the #299 preconditioner-ladder iteration study: the converged
/// PCG iteration count and stop reason for a single preconditioner tier.
#[derive(Debug, Clone, Copy)]
pub struct PrecondLadderRow {
    /// PCG iterations to convergence (or to the `MaxIter` cutoff).
    pub iterations: usize,
    /// Whether the PCG converged (vs hit `MaxIter` / negative curvature).
    pub converged: bool,
    /// Final relative residual reported by the PCG.
    pub final_relative_residual: f64,
}

/// Full #299 ladder iteration study on one reduced-Schur system: run the SAME
/// preconditioned CG (same `rhs`, tolerances, trust radius) once per ladder tier
/// and report the iteration count of each. This is the public seam the
/// `tests/owed_299.rs` iteration-reduction gate drives — it keeps the internal
/// `run_pcg_with_preconditioner` / preconditioner constructors `pub(crate)`
/// while exposing exactly the per-tier measurement the issue asks for.
///
/// Tiers (in escalation order): scalar `Diagonal`, `BetaBlockJacobi`,
/// `ClusterJacobi`, `AdditiveSchwarz{overlap:1}`, `DiagAssembledSchwarz{1}`, and
/// `BlockIncompleteCholesky`. A tier whose build fails (e.g. non-PD reduced
/// Schur with no curvature floor) reports `None` for that entry; every healthy
/// SPD reduced system populates all six.
pub fn arrow_precond_ladder_iteration_study(
    sys: &ArrowSchurSystem,
    ridge_beta: f64,
    rhs: &Array1<f64>,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
) -> Result<Vec<(SchurPreconditionerKind, Option<PrecondLadderRow>)>, ArrowSchurError> {
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend.factor_blocks(&sys.rows, 0.0, sys.d, false)?;

    let run = |apply: &dyn Fn(&Array1<f64>) -> Array1<f64>| -> Option<PrecondLadderRow> {
        let (_sol, diag) = run_pcg_with_preconditioner(
            sys,
            &htt_factors,
            ridge_beta,
            rhs,
            |r| apply(r),
            pcg,
            trust,
            &backend,
            None,
            None,
            None,
        )
        .ok()?;
        Some(PrecondLadderRow {
            iterations: diag.iterations,
            converged: matches!(diag.stopping_reason, PcgStopReason::Converged),
            final_relative_residual: diag.final_relative_residual,
        })
    };

    let mut out: Vec<(SchurPreconditionerKind, Option<PrecondLadderRow>)> = Vec::with_capacity(7);

    // Scalar Diagonal Jacobi: force the scalar path by clearing block_offsets on
    // a clone so the build does not pick up the per-block dense Schur blocks.
    let diag_row = {
        let mut bare = sys.clone();
        bare.set_block_offsets(std::sync::Arc::from([] as [Range<usize>; 0]));
        let bare_factors = backend.factor_blocks(&bare.rows, 0.0, bare.d, false)?;
        JacobiPreconditioner::from_arrow_schur(&bare, &bare_factors, ridge_beta, &backend, None)
            .ok()
            .and_then(|p| {
                run_pcg_with_preconditioner(
                    &bare,
                    &bare_factors,
                    ridge_beta,
                    rhs,
                    |r| p.apply(r),
                    pcg,
                    trust,
                    &backend,
                    None,
                    None,
                    None,
                )
                .ok()
                .map(|(_s, diag)| PrecondLadderRow {
                    iterations: diag.iterations,
                    converged: matches!(diag.stopping_reason, PcgStopReason::Converged),
                    final_relative_residual: diag.final_relative_residual,
                })
            })
    };
    out.push((SchurPreconditionerKind::Diagonal, diag_row));

    let block_row =
        JacobiPreconditioner::from_arrow_schur(sys, &htt_factors, ridge_beta, &backend, None)
            .ok()
            .and_then(|p| run(&|r| p.apply(r)));
    out.push((SchurPreconditionerKind::BetaBlockJacobi, block_row));

    let cluster_row =
        ClusterJacobiPreconditioner::from_arrow_schur(sys, &htt_factors, ridge_beta, &backend)
            .ok()
            .and_then(|p| run(&|r| p.apply(r)));
    out.push((SchurPreconditionerKind::ClusterJacobi, cluster_row));

    let covis_row = ClusterJacobiPreconditioner::from_arrow_schur_covisibility(
        sys,
        &htt_factors,
        ridge_beta,
        &backend,
    )
    .ok()
    .and_then(|p| run(&|r| p.apply(r)));
    out.push((
        SchurPreconditionerKind::CoVisibilityClusterJacobi,
        covis_row,
    ));

    let schwarz_row =
        AdditiveSchwarzPreconditioner::from_arrow_schur(sys, &htt_factors, ridge_beta, &backend, 1)
            .ok()
            .and_then(|p| run(&|r| p.apply(r)));
    out.push((
        SchurPreconditionerKind::AdditiveSchwarz { overlap: 1 },
        schwarz_row,
    ));

    let diag_schwarz_row = DiagAssembledSchwarzPreconditioner::from_arrow_schur(
        sys,
        &htt_factors,
        ridge_beta,
        &backend,
        1,
    )
    .ok()
    .and_then(|p| run(&|r| p.apply(r)));
    out.push((
        SchurPreconditionerKind::DiagAssembledSchwarz { overlap: 1 },
        diag_schwarz_row,
    ));

    let ic0_row = BlockIncompleteCholeskyPreconditioner::from_arrow_schur(
        sys,
        &htt_factors,
        ridge_beta,
        &backend,
    )
    .ok()
    .and_then(|p| run(&|r| p.apply(r)));
    out.push((SchurPreconditionerKind::BlockIncompleteCholesky, ic0_row));

    Ok(out)
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
    let mut result = Vec::with_capacity(cols.len());
    // Extract the penalty diagonal for all K columns once, then index per-column.
    let mut full_diag = Array1::<f64>::zeros(sys.k);
    {
        let diag_slice = full_diag.as_slice_mut().expect("full_diag contiguous");
        sys.penalty_diagonal_add(diag_slice);
    }
    // Probe each needed column through the ROUTED `H_tβ` convention at each
    // row's own width (see `assemble_local_schur_block` for why a raw
    // `row.htbeta` read at the global `sys.d` is wrong here).
    let mut e_g = Array1::<f64>::zeros(sys.k);
    for &gi in cols {
        let mut s = full_diag[gi] + ridge_beta;
        e_g[gi] = 1.0;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let di = sys.row_dims[row_idx];
            let mut col_vec = Array1::<f64>::zeros(di);
            sys_htbeta_apply_row(sys, row_idx, row, e_g.view(), &mut col_vec);
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
            let mut acc = 0.0;
            for c in 0..di {
                acc += col_vec[c] * solved[c];
            }
            s -= acc;
        }
        e_g[gi] = 0.0;
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
    curvature_floor: Option<f64>,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
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
    // #1026 — curvature-floor retry on the Jacobi tier. The unbounded SAE inner
    // PCG (trust radius = ∞) fails on `pᵀSp ≤ 0` when the reduced Schur is
    // indefinite (K≥4 co-collapse: a near-singular per-row `H_tt` over-subtracts
    // `S`). Instead of letting that failure propagate to the outer LM loop —
    // which inflates `ridge_β` over EVERY β direction and makes the inner Newton
    // crawl — floor the OPERATOR by the minimal ridge `δ = |pᵀSp|/‖p‖² · (1+ε)`
    // that restores positive curvature along the offending direction, rebuild the
    // Jacobi preconditioner at the lifted ridge, and retry. This is the
    // matrix-free analogue of the dense `spectral_pd_floored_schur`: the healthy
    // β subspace (where curvature is already positive) is essentially untouched
    // by a tiny `δ`, while the collapsed direction gets exactly the stiffness it
    // needs to make a real descent step. A PD reduced Schur never hits `pᵀSp ≤ 0`,
    // so this loop is a strict no-op there (bit-for-bit unchanged). Bounded by a
    // small attempt cap and a relative ridge ceiling; on exhaustion the original
    // recoverable failure still reaches the outer LM loop.
    let mut effective_ridge = ridge_beta;
    let mut x0_diag0: Option<(Array1<f64>, ArrowPcgDiagnostics)> = None;
    let mut last_curvature_err: Option<ArrowSchurError> = None;
    let rhs_scale = metric_norm(rhs.view(), metric_weights).max(1.0);
    let ridge_ceiling = ridge_beta.max(SCHUR_CURVATURE_FLOOR_REL_CEILING * rhs_scale);
    for _attempt in 0..=SCHUR_CURVATURE_FLOOR_MAX_ATTEMPTS {
        // The Jacobi preconditioner build itself refuses a non-PD Schur diagonal
        // (`PcgFailed: invalid Schur Jacobi diagonal`) — the SAME co-collapse
        // signature reached BEFORE the CG loop, since `S_ii = H_ββ,ii − Σ …` goes
        // negative. Treat that build failure as a curvature deficit too: when the
        // floor is enabled, lift the ridge and retry; otherwise propagate.
        let jacobi = match JacobiPreconditioner::from_arrow_schur(
            sys,
            htt_factors,
            effective_ridge,
            backend,
            resident.as_ref(),
        ) {
            Ok(jacobi) => jacobi,
            Err(err @ ArrowSchurError::PcgFailed { .. }) => {
                if curvature_floor.is_none() {
                    return Err(err);
                }
                // A diagonal refusal carries no `(curvature, ‖p‖²)` deficit, and
                // the over-subtraction magnitude `Σ H_tβᵀ(H_tt)⁻¹H_tβ` is
                // unbounded relative to `rhs_scale`, so a small additive bump
                // would crawl. Escalate the ridge MULTIPLICATIVELY (×10, matching
                // the per-row `factor_one_row_result` RIDGE_GROWTH_FACTOR), seeded
                // at `rhs_scale`, so even a large deficit (the collapsed
                // `(H_tβ)²/H_tt` over-subtraction) is reached in a handful of
                // attempts. The ceiling + attempt cap still bound it; on
                // exhaustion the recoverable failure reaches the outer LM loop.
                // Jump straight to a meaningful scale on the FIRST refusal rather
                // than crawling ×10 from a tiny `ridge_beta`: each rebuild is a full
                // block-Jacobi factorization (the massive-K preconditioner hotspot),
                // and a large collapsed deficit (`Σ H_tβᵀ(H_tt)⁻¹H_tβ` over-subtraction,
                // O(1)-scale) otherwise costs ~log10(deficit / ridge_beta) rebuilds.
                // Seeding the first bump at `rhs_scale` covers it in one or two, then
                // escalates multiplicatively; the ceiling + attempt cap still bound it.
                let next = if effective_ridge > 0.0 {
                    (effective_ridge * SCHUR_CURVATURE_FLOOR_DIAG_GROWTH).max(rhs_scale)
                } else {
                    rhs_scale
                };
                last_curvature_err = Some(err);
                if !next.is_finite() || next > ridge_ceiling {
                    break;
                }
                effective_ridge = next;
                continue;
            }
            Err(other) => return Err(other),
        };
        match run_pcg_with_preconditioner(
            sys,
            htt_factors,
            effective_ridge,
            rhs,
            |r| jacobi.apply(r),
            pcg,
            trust,
            backend,
            gpu_matvec,
            metric_weights,
            resident.as_ref(),
        ) {
            Ok(result) => {
                x0_diag0 = Some(result);
                break;
            }
            Err(ArrowSchurError::UnboundedNegativeCurvature {
                curvature,
                direction_norm_sq,
            }) => {
                // Only floor when the caller opted in (SAE solve path); otherwise
                // propagate the raw negative-curvature signal so BA / non-SAE
                // unbounded solves keep their existing failure contract.
                let Some(relative_floor) = curvature_floor else {
                    return Err(ArrowSchurError::UnboundedNegativeCurvature {
                        curvature,
                        direction_norm_sq,
                    });
                };
                // Minimal ridge to make `pᵀ(S+δI)p = |curvature| + δ·‖p‖² > 0`,
                // with a margin so the next CG iterate has strictly positive
                // curvature rather than sitting on the `0` knife-edge.
                let deficit = if direction_norm_sq > 0.0 {
                    curvature.abs() / direction_norm_sq
                } else {
                    0.0
                };
                let bump = (deficit * (1.0 + SCHUR_CURVATURE_FLOOR_MARGIN))
                    .max(relative_floor.max(SCHUR_CURVATURE_FLOOR_REL_FLOOR) * rhs_scale);
                let next = (effective_ridge + bump).max(effective_ridge * 2.0);
                last_curvature_err = Some(ArrowSchurError::UnboundedNegativeCurvature {
                    curvature,
                    direction_norm_sq,
                });
                if !next.is_finite() || next > ridge_ceiling {
                    break;
                }
                effective_ridge = next;
            }
            Err(other) => return Err(other),
        }
    }
    let (x0, diag0) = match x0_diag0 {
        Some(result) => result,
        None => {
            // The curvature floor could not condition the operator within the
            // ceiling; hand the recoverable failure to the outer LM loop, which
            // re-forms the system at a heavier ridge.
            return Err(last_curvature_err.unwrap_or(ArrowSchurError::PcgFailed {
                reason: "unbounded Schur PCG negative curvature unresolved by curvature floor"
                    .to_string(),
            }));
        }
    };
    if sys.k <= PRECOND_ESCALATE_K_THRESHOLD || diag0.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x0, diag0));
    }
    // Escalation tiers reuse the curvature-floored `effective_ridge` so the
    // operator they precondition is the SAME (PD-floored) one the Jacobi tier
    // settled on; a still-negative-curvature signal here is handed to the outer
    // LM loop (it only arises if the floored Jacobi tier merely ran out of
    // iterations yet a coarser preconditioner still finds an indefinite
    // direction — rare; the LM loop re-forms at a heavier ridge).
    // Default cluster tier: the bounded CO-VISIBILITY partition, not the
    // connected-component partition. At the SAE widths this ladder targets the
    // co-firing graph is one giant component, so the component partition exceeds
    // the size cap and `from_arrow_schur` degrades to scalar Jacobi (the ceiling
    // this tier exists to lift). `from_arrow_schur_covisibility` splits that
    // component into bounded strongly-co-firing clusters whose dense factors
    // condition the cross-atom coupling scalar Jacobi drops. The component
    // partition stays selectable via `from_arrow_schur` (used by the ladder
    // study and its regression gates). Both precondition the SAME operator, so
    // the converged step — and the REML optimum — is unchanged.
    let cluster = ClusterJacobiPreconditioner::from_arrow_schur_covisibility(
        sys,
        htt_factors,
        effective_ridge,
        backend,
    )?;
    let (x1, diag1) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
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
    let schwarz = AdditiveSchwarzPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        effective_ridge,
        backend,
        1,
    )?;
    let (x2, diag2) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| schwarz.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
        resident.as_ref(),
    )?;
    if diag2.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x2, diag2));
    }
    // Final tier — diagonal-assembled additive Schwarz (#299), the cheap-apply
    // Schwarz variant. When the dense-block AdditiveSchwarz still ran out of
    // iterations its O(Σ b_k²) apply may have throttled the iteration budget on
    // a wide subdomain; the diag-assembled variant keeps Schwarz's overlapping
    // local-inverse conditioning but applies in O(K), so it can take more CG
    // iterations within the same wall budget. Same overlap (1) and same
    // curvature-floored ridge as the dense-block tier.
    let diag_schwarz = DiagAssembledSchwarzPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        effective_ridge,
        backend,
        1,
    )?;
    let (x3, diag3) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| diag_schwarz.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
        resident.as_ref(),
    )?;
    if diag3.stopping_reason != PcgStopReason::MaxIter {
        return Ok((x3, diag3));
    }
    // Richest tier — level-0 incomplete Cholesky (#299). ClusterJacobi keeps the
    // full DENSE Cholesky of each component (so on a single large connected
    // component it fills the whole `b×b` factor and its `O(b²)` apply throttles
    // the CG iteration budget), while the diagonal/Schwarz tiers drop most
    // inter-block coupling. IC(0) keeps the component's full structural coupling
    // but only the level-0 (no-fill) pattern, so its sparse triangular apply is
    // `O(nnz(S[C,C]))` — it can take more CG iterations within the same wall
    // budget AND conditions the off-diagonal coupling the cheap tiers discard.
    // Last in the ladder so it is only paid when every cheaper tier stalled.
    let ic0 = BlockIncompleteCholeskyPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        effective_ridge,
        backend,
    )?;
    let (x4, diag4) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| ic0.apply(r),
        pcg,
        trust,
        backend,
        gpu_matvec,
        metric_weights,
        resident.as_ref(),
    )?;
    // All five preconditioner tiers (Jacobi -> ClusterJacobi -> AdditiveSchwarz
    // -> DiagAssembledSchwarz -> BlockIncompleteCholesky) exhausted their
    // iteration budget without driving the residual below tolerance. Returning a
    // truncated iterate as `Ok` would feed an arbitrarily-large-residual step
    // into the Newton driver, where the PCG diagnostics are discarded. Surface a
    // recoverable failure instead so `solve_with_lm_escalation_inner` escalates
    // the proximal ridge: better conditioning is precisely what a stalled PCG on
    // an ill-conditioned reduced system needs.
    if diag4.stopping_reason == PcgStopReason::MaxIter {
        return Err(ArrowSchurError::PcgFailed {
            reason: format!(
                "Schur PCG exhausted all preconditioner tiers (Jacobi, ClusterJacobi, \
                 AdditiveSchwarz, DiagAssembledSchwarz, BlockIncompleteCholesky) at MaxIter; \
                 final relative residual = {:e}",
                diag4.final_relative_residual
            ),
        });
    }
    Ok((x4, diag4))
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
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError>
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
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
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
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError>
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
        return Ok((Array1::<f64>::zeros(n), ArrowPcgDiagnostics::default()));
    }
    let tol = (relative_tolerance.max(0.0) * rhs_norm).max(PCG_ABSOLUTE_TOLERANCE_FLOOR);
    let mut x = Array1::<f64>::zeros(n);
    let mut r = rhs.clone();
    let mut z = apply_preconditioner(&r);
    let mut diag = ArrowPcgDiagnostics {
        precond_apply_calls: 1,
        ..ArrowPcgDiagnostics::default()
    };
    let mut p = z.clone();
    let mut rz = metric_dot(&r, &z, metric_weights);
    if rz <= 0.0 || !rz.is_finite() {
        if radius.is_finite() {
            diag.final_relative_residual = metric_norm(r.view(), metric_weights) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &r, radius, metric_weights), diag));
        }
        // Unbounded (radius = ∞) non-positive preconditioned residual: the
        // reduced Schur is indefinite at the very first direction. Surface the
        // typed curvature-floor signal so `steihaug_pcg_auto` floors the
        // operator minimally and retries, instead of failing into a global
        // `ridge_β` ramp. `rz = rᵀM⁻¹r` is a preconditioner-metric curvature;
        // report it with the residual norm² as the direction scale.
        return Err(ArrowSchurError::UnboundedNegativeCurvature {
            curvature: rz,
            direction_norm_sq: metric_dot(&r, &r, metric_weights),
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
            // Unbounded negative curvature `pᵀSp ≤ 0`: the reduced Schur is
            // indefinite along `p` (the #1026 co-collapse direction). Surface
            // the typed signal carrying `pᵀSp` and `‖p‖²` so the caller floors
            // the operator by the minimal ridge `δ = |pᵀSp|/‖p‖²` (which makes
            // `pᵀ(S+δI)p = 0⁺`) plus a margin, and retries.
            return Err(ArrowSchurError::UnboundedNegativeCurvature {
                curvature: pap,
                direction_norm_sq: metric_dot(&p, &p, metric_weights),
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
    /// The UNBOUNDED (trust-radius = ∞) Schur PCG encountered negative
    /// curvature `pᵀSp ≤ 0` (or a non-positive preconditioned residual): the
    /// reduced Schur is indefinite, the #1026 K≥4 co-collapse signature where
    /// a near-singular per-row `H_tt` over-subtracts `S`. With no trust radius
    /// there is no boundary to step to, so CG cannot proceed. `curvature` is
    /// the offending `pᵀSp` and `direction_norm_sq` the `‖p‖²` of the
    /// negative-curvature direction; the caller floors the operator with the
    /// minimal ridge `δ = (|curvature|/‖p‖² )·(1+ε)` that restores positive
    /// curvature along `p` and retries (matrix-free analogue of the dense
    /// `spectral_pd_floored_schur`), rather than blindly inflating `ridge_β`.
    UnboundedNegativeCurvature {
        curvature: f64,
        direction_norm_sq: f64,
    },
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
            ArrowSchurError::UnboundedNegativeCurvature {
                curvature,
                direction_norm_sq,
            } => write!(
                f,
                "arrow-Schur: unbounded Schur PCG hit negative curvature pᵀSp={curvature:e} \
                 (‖p‖²={direction_norm_sq:e}); reduced Schur is indefinite (co-collapse), \
                 retry with a curvature-floor ridge"
            ),
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

    // Only clone for the device attempt when a GPU is actually selected;
    // `try_cholesky_lower_inplace` returns `None` on every CPU-only host, so the
    // former unconditional `k×k` clone (≈32 MB at the SAE border) was pure waste
    // on the CPU path that now dominates. `cuda_selected()` is a cheap policy +
    // runtime-presence probe.
    if gam_gpu::cuda_selected() {
        let mut maybe_device = a.clone();
        if gam_gpu::try_cholesky_lower_inplace(&mut maybe_device).is_some() {
            return Ok(maybe_device);
        }
    }

    // CPU fast path (#1017): at the SAE border width the reduced Schur is a
    // dense `k×k` (k≈2k–4k) whose scalar triple-loop factorization is O(k³/3)
    // and neither blocked nor SIMD-vectorized — the dominant per-Newton-step
    // cost on a CPU-only host. faer's blocked LLT computes the SAME `A = L Lᵀ`
    // (to O(κ·ε), the slack the reduced solve/log-det already tolerate) an order
    // of magnitude faster. Restrict it to `k ≥ FAER_CHOLESKY_MIN` so the many
    // tiny per-row `d×d` blocks (d≤~8, factorization.rs) and the small dense
    // test fixtures keep the exact scalar loop — bit-for-bit their historical
    // factor — where faer's setup overhead would not pay off anyway. If faer
    // declines (a non-PD blocked pivot) fall through to the scalar loop so the
    // PD/non-PD verdict and its typed error stay exactly the historical ones
    // (`factor_dense_reduced_schur`'s spectral-floor fallback keys only on Ok vs
    // Err, so the boundary behavior is unchanged).
    const FAER_CHOLESKY_MIN: usize = 128;
    if n >= FAER_CHOLESKY_MIN {
        let view = gam_linalg::faer_ndarray::FaerArrayView::new(a);
        if let Ok(llt) = gam_linalg::faer_ndarray::FaerLlt::new(view.as_ref(), faer::Side::Lower) {
            let l_faer = llt.L();
            let mut l = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..=i {
                    l[[i, j]] = l_faer[(i, j)];
                }
            }
            return Ok(l);
        }
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
