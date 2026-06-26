//! Per-row Hessian block factorization (Cholesky, gauge/spectral deflation,
//! safe-inversion gates) and the system/penalty fingerprinting that keys the
//! per-row factor cache.

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct ArrowRowFactorResult {
    pub(crate) factor: Array2<f64>,
    pub(crate) gauge_deflated_directions: usize,
}

/// Attempt the per-row block factorization as one device batch spread across
/// every usable GPU.
///
/// The `n` per-row blocks `H_tt^(i) + ridge_t·I` are independent SPD systems of
/// the uniform shape `d×d`; `gam_gpu::try_cholesky_batched_lower_inplace`
/// factors the whole batch with a batched POTRF that the shared device pool
/// tiles across all ordinals. Returns `Some(factors)` only when:
///   * every row really is the uniform `(d, d)` shape with a length-`d` `g_t`
///     (heterogeneous systems keep the per-row CPU loop), and
///   * a device is available and EVERY block is positive-definite at the base
///     ridge (a non-PD block makes the batched POTRF return `None`), and
///   * unless `tolerate_ill_conditioning`, every resulting factor passes the
///     same diagonal-ratio κ ceiling `factor_one_row` enforces.
///
/// Any of those failing returns `None`, so the caller runs the exact per-row
/// CPU path (which performs minimal per-block ridge escalation and the κ check).
/// The factor a device POTRF produces is the lower Cholesky of the identical
/// SPD matrix the CPU `cholesky_lower` would, with the strict upper triangle
/// zeroed — bit-for-bit equivalent modulo IEEE-754 reduction order.
pub(crate) fn try_factor_blocks_batched(
    rows: &[ArrowRowBlock],
    ridge_t: f64,
    d: usize,
    tolerate_ill_conditioning: bool,
) -> Option<ArrowFactorSlab> {
    if d == 0 || rows.is_empty() {
        return None;
    }
    // Uniform-shape gate: a heterogeneous row defeats the single-shape batched
    // POTRF and deliberately falls through to per-row CPU escalation.
    if rows
        .iter()
        .any(|row| row.htt.dim() != (d, d) || row.gt.len() != d)
    {
        return None;
    }
    // No device → let the CPU path own the work (it is the exact fallback).
    if !gam_gpu::device_runtime::GpuRuntime::is_available() {
        return None;
    }

    // Assemble the damped blocks `H_tt^(i) + ridge_t·I` for the batched POTRF.
    let mut blocks: Vec<Array2<f64>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut block = row.htt.clone();
        for a in 0..d {
            block[[a, a]] += ridge_t;
        }
        blocks.push(block);
    }

    // Batched lower Cholesky over ALL usable GPUs. `None` ⇒ either no device
    // accepted the workload or some block was not PD at the base ridge; either
    // way the per-row CPU path must own escalation.
    gam_gpu::try_cholesky_batched_lower_inplace(&mut blocks)?;

    // Re-apply the κ-conditioning rejection so a barely-PD block forces the
    // whole batch back to the per-row path (where its ridge lifts), matching
    // `factor_one_row` semantics exactly. Evidence/log-det-only callers
    // tolerate ill-conditioning and skip this, as on the CPU path.
    if !tolerate_ill_conditioning {
        for (row, factor) in rows.iter().zip(blocks.iter()) {
            let diag_scale = row_block_diag_scale(row, d);
            let kappa_est = cholesky_factor_kappa_estimate(factor);
            if !cholesky_factor_passes_safe_inversion(factor, d, diag_scale, kappa_est) {
                return None;
            }
        }
    }
    Some(ArrowFactorSlab::from_blocks(blocks))
}

pub(crate) fn row_block_diag_scale(row: &ArrowRowBlock, d: usize) -> f64 {
    (0..d)
        .map(|a| row.htt[[a, a]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0)
}

/// Diagonal-ratio condition-number proxy for an SPD matrix from its lower
/// Cholesky factor `L` (where `A = L Lᵀ`):
///     κ(A) ≈ (max_i L_ii / min_i L_ii)².
///
/// (Golub & Van Loan, "Matrix Computations" 4th ed., §4.2.4 — the ratio of
/// diagonal entries of the Cholesky factor bounds the 2-norm condition number
/// of the SPD matrix.) Returns `f64::INFINITY` when the factor has a
/// non-positive or non-finite diagonal pivot, which the callers treat as a
/// hard ill-conditioning signal.
pub(crate) fn cholesky_factor_kappa_estimate(factor: &Array2<f64>) -> f64 {
    let d = factor.nrows();
    let mut min_diag = f64::INFINITY;
    let mut max_diag = 0.0_f64;
    for a in 0..d {
        let v = factor[[a, a]];
        if v < min_diag {
            min_diag = v;
        }
        if v > max_diag {
            max_diag = v;
        }
    }
    if min_diag > 0.0 && max_diag.is_finite() {
        let ratio = max_diag / min_diag;
        ratio * ratio
    } else {
        f64::INFINITY
    }
}

/// Smallest Cholesky pivot estimate for `A = L Lᵀ`, using `L_ii²`.
///
/// The diagonal-ratio κ proxy is blind for scalar blocks: every positive
/// `1×1` factor has κ=1 even when the pivot is tiny. This pivot floor catches
/// absolute near-singularity relative to the row block scale.
pub(crate) fn cholesky_factor_min_pivot_estimate(factor: &Array2<f64>) -> f64 {
    let d = factor.nrows();
    if d == 0 {
        return 0.0;
    }
    let mut min_pivot = f64::INFINITY;
    for a in 0..d {
        let v = factor[[a, a]];
        if !(v > 0.0 && v.is_finite()) {
            return 0.0;
        }
        let pivot = v * v;
        if pivot < min_pivot {
            min_pivot = pivot;
        }
    }
    min_pivot
}

pub(crate) fn safe_spd_pivot_min(diag_scale: f64) -> f64 {
    f64::EPSILON.sqrt() * diag_scale.max(1.0)
}

pub(crate) fn cholesky_factor_passes_safe_inversion(
    factor: &Array2<f64>,
    dim: usize,
    diag_scale: f64,
    kappa_est: f64,
) -> bool {
    kappa_est.is_finite()
        && kappa_est <= safe_spd_kappa_max(dim)
        && cholesky_factor_min_pivot_estimate(factor) >= safe_spd_pivot_min(diag_scale)
}

/// Near-singularity condition-number ceiling for double precision at dimension
/// `dim`: κ_max = 1 / (sqrt(DBL_EPS) · max(dim, 1)).
///
/// Classic Higham rule (Higham, "Accuracy and Stability of Numerical
/// Algorithms" 2nd ed., §10.1): a system is treated as numerically
/// rank-deficient once κ · ε approaches 1/sqrt(ε), scaled by problem dimension.
pub(crate) fn safe_spd_kappa_max(dim: usize) -> f64 {
    let d_scale = (dim as f64).max(1.0);
    1.0 / (f64::EPSILON.sqrt() * d_scale)
}

pub(crate) fn factor_row_block_cholesky(
    row: &ArrowRowBlock,
    ridge_eff: f64,
    d: usize,
) -> Result<Array2<f64>, String> {
    match d {
        1 => factor_row_block_cholesky_fixed::<1>(row, ridge_eff),
        2 => factor_row_block_cholesky_fixed::<2>(row, ridge_eff),
        3 => factor_row_block_cholesky_fixed::<3>(row, ridge_eff),
        4 => factor_row_block_cholesky_fixed::<4>(row, ridge_eff),
        _ => factor_row_block_cholesky_dynamic(row, ridge_eff, d),
    }
}

pub(crate) fn factor_row_block_cholesky_dynamic(
    row: &ArrowRowBlock,
    ridge_eff: f64,
    d: usize,
) -> Result<Array2<f64>, String> {
    let mut block = row.htt.clone();
    for a in 0..d {
        block[[a, a]] += ridge_eff;
    }
    cholesky_lower(&block)
}

pub(crate) fn factor_row_block_cholesky_fixed<const D: usize>(
    row: &ArrowRowBlock,
    ridge_eff: f64,
) -> Result<Array2<f64>, String> {
    for i in 0..D {
        for j in 0..D {
            let value = if i == j {
                row.htt[[i, j]] + ridge_eff
            } else {
                row.htt[[i, j]]
            };
            if !value.is_finite() {
                let idx = i * D + j;
                return Err(format!(
                    "cholesky_lower: non-finite entry at linear index {idx}"
                ));
            }
        }
    }

    let mut l = [[0.0_f64; D]; D];
    for i in 0..D {
        for j in 0..=i {
            let mut sum = if i == j {
                row.htt[[i, j]] + ridge_eff
            } else {
                row.htt[[i, j]]
            };
            for kk in 0..j {
                sum -= l[i][kk] * l[j][kk];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(format!(
                        "non-PD pivot {sum} at index {i} (matrix is not positive definite)"
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut out = Array2::<f64>::zeros((D, D));
    for i in 0..D {
        for j in 0..=i {
            out[[i, j]] = l[i][j];
        }
    }
    Ok(out)
}

pub(crate) fn row_gauge_curvature(
    row: &ArrowRowBlock,
    d: usize,
    gauge: &Array1<f64>,
) -> Option<f64> {
    if gauge.len() != d {
        return None;
    }
    let mut acc = 0.0_f64;
    for i in 0..d {
        let gi = gauge[i];
        for j in 0..d {
            acc += gi * row.htt[[i, j]] * gauge[j];
        }
    }
    if acc.is_finite() { Some(acc) } else { None }
}

pub(crate) fn factor_gauge_deflated_evidence_row(
    row: &ArrowRowBlock,
    d: usize,
    gauges: &[Array1<f64>],
) -> Option<ArrowRowFactorResult> {
    const GAUGE_RAYLEIGH_EPS: f64 = 1.0e-8;
    if gauges.is_empty() {
        return None;
    }
    let max_diag = row_block_diag_scale(row, d);
    if !(max_diag.is_finite() && max_diag > 0.0) {
        return None;
    }
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for gauge in gauges {
        if gauge.len() != d {
            continue;
        }
        let norm_sq = gauge.iter().map(|&v| v * v).sum::<f64>();
        if !(norm_sq.is_finite() && norm_sq > 1.0e-24) {
            continue;
        }
        let curvature = row_gauge_curvature(row, d, gauge)?;
        // Two-sided gauge qualification: a true orbit direction has Rayleigh
        // quotient ~ 0 from EITHER side (the observed failures sit at ~ -1e-10).
        // Strongly NEGATIVE curvature is genuine indefiniteness (e.g. the raw
        // assignment-prior logit curvature off-optimum), not a gauge — deflating
        // it would mask a real non-PD block and bias the evidence. Only
        // |g^T H g| <= eps * scale * |g|^2 qualifies (the absolute value is what
        // makes this two-sided: a large-magnitude curvature of EITHER sign is
        // disqualified, so only a genuine near-null orbit is deflated).
        if curvature.abs() > GAUGE_RAYLEIGH_EPS * max_diag * norm_sq {
            continue;
        }
        let mut direction = gauge.clone();
        for existing in &basis {
            let coeff = direction.dot(existing);
            for idx in 0..d {
                direction[idx] -= coeff * existing[idx];
            }
        }
        let residual_norm_sq = direction.iter().map(|&v| v * v).sum::<f64>();
        if !(residual_norm_sq.is_finite() && residual_norm_sq > 1.0e-24) {
            continue;
        }
        let inv_norm = residual_norm_sq.sqrt().recip();
        for value in direction.iter_mut() {
            *value *= inv_norm;
        }
        basis.push(direction);
    }
    if basis.is_empty() {
        return None;
    }

    // Faddeev-Popov stiffening of the orbit, at UNIT stiffness kappa = 1.0
    // (NOT max_diag). The direction is already unit-normalized, so each deflated
    // direction contributes exactly +1 to that eigenvalue of `deflated`, hence
    // log(1) = 0 to log|H|. This is the codebase's quotient PSEUDO-DETERMINANT
    // convention (cf. `PenaltyPseudologdet`, which evaluates log|S| over the
    // non-degenerate subspace and drops the kernel): the gauge orbit is a
    // genuine null direction of the criterion, so it must contribute NOTHING to
    // the Laplace normalizer. A theta/rho-dependent kappa (e.g. max_diag) would
    // inject a spurious log(kappa(theta,rho)) into the evidence and bias the
    // REML rho-gradient whenever a deflated direction survives to the optimum —
    // holding the deflated COUNT fixed across the solve does NOT make the VALUE
    // theta/rho-constant. kappa = 1.0 is exactly zero-derivative by
    // construction. The quotient-complement solve is identical either way, so
    // evidence-mode exactness on the non-degenerate subspace is preserved.
    // `max_diag` stays ONLY in the qualification threshold above, where it is
    // the curvature unit the orbit's near-zero Rayleigh quotient is measured
    // against — never in the stiffness. (d <= 3 blocks; kappa = 1 against large
    // other-direction curvature is a condition number the Cholesky handles
    // trivially.)
    let mut deflated = row.htt.clone();
    for direction in &basis {
        for i in 0..d {
            for j in 0..d {
                deflated[[i, j]] += direction[i] * direction[j];
            }
        }
    }
    let factor = cholesky_lower(&deflated).ok()?;
    Some(ArrowRowFactorResult {
        factor,
        gauge_deflated_directions: basis.len(),
    })
}

/// Relative spectral floor (vs the block's largest-magnitude eigenvalue) below
/// which a per-row `H_tt` eigen-direction is treated as non-identified and
/// unit-stiffness deflated rather than ridge-damped. Matches the magnitude of
/// the gauge Rayleigh qualifier and the `SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF`
/// data-null detection so the three deflation paths agree on what "flat" means.
pub(crate) const SPECTRAL_DEFLATION_REL_FLOOR: f64 = 1.0e-8;

/// Hysteresis half-width (as a fraction of `SPECTRAL_DEFLATION_REL_FLOOR`)
/// applied to the spectral-deflation decision for *positive* near-floor
/// eigenvalues, to stop the per-row deflation COUNT from flickering as a small
/// positive curvature direction wanders across the cutoff over a ρ/θ-walk
/// (#1117). The quotient-dimension guard (`record_evidence_gauge_deflation_count`)
/// correctly refuses to compare Laplace normalizers across different deflated
/// dimensions, so a single eigenvalue oscillating around the bare floor would
/// otherwise toggle the count 6↔7 within one optimization and trip the guard
/// spuriously, forcing a slow seed/homotopy cascade.
///
/// The decision is split by the only physically meaningful distinction at the
/// inner optimum: a NON-POSITIVE (or non-finite) eigenvalue is a genuine null /
/// indefinite quotient direction and is ALWAYS deflated — that boundary sits at
/// exact zero, far from where live curvature lives, so it does not flicker (a
/// curvature direction genuinely crossing zero IS a structural event the guard
/// must still catch). Only a *positive* eigenvalue near `floor` is ambiguous,
/// and for it we use the LOWER band edge `floor·(1−ε)`: a positive eigenvalue
/// parked at the bare floor is `> floor·(1−ε)` and is therefore consistently
/// KEPT on both sides of the walk, so the count is stable by construction. A
/// direction that is genuinely numerically flat sits orders of magnitude below
/// `floor` (a true rank deficiency, `λ ≪ floor·(1−ε)`), so it is still deflated
/// exactly as before — the converged result is unchanged wherever the old path
/// already deflated a clearly-flat or indefinite direction.
pub(crate) const SPECTRAL_DEFLATION_HYSTERESIS_FRACTION: f64 = 1.0e-2;

/// Unit-stiffness **spectral** Faddeev-Popov conditioning of a per-row evidence
/// block `H_tt` that the undamped Cholesky refused because it is genuinely
/// indefinite or numerically flat off the closed-form gauge orbit.
///
/// This is the spectral sibling of [`factor_gauge_deflated_evidence_row`]: that
/// one deflates a SUPPLIED orbit direction (the circle rotation gauge, etc.);
/// this one DISCOVERS the offending directions from the block's own symmetric
/// eigendecomposition, for the case (#1117/#1118, K>1 IBP/softmax row-sharing)
/// where the logit×coordinate Gauss-Newton cross term drives an eigenvalue of
/// `H_tt` negative (or to a numerically-flat near-zero) at a direction that is
/// NOT a known gauge vector. `d ≤ 3` here so the eigendecomposition is trivial.
///
/// Each eigenvalue at or below `floor = SPECTRAL_DEFLATION_REL_FLOOR · max|λ|`
/// (this INCLUDES every negative eigenvalue) is replaced by exactly `+1` while
/// its eigenvector is preserved; the strictly-positive, well-separated
/// directions are reconstructed bit-for-bit (`Σ λ_i v_i v_iᵀ`). The result is
/// SPD by construction, so the Cholesky succeeds and the evidence log-det is
/// finite.
///
/// The stiffness is UNIT (`+1`), not `max|λ|` and not `ridge·I`: a deflated
/// direction therefore contributes exactly `log 1 = 0` to `log|H|`, with ZERO
/// θ/ρ dependence — the same quotient pseudo-determinant convention the gauge
/// deflation (κ=1) and the #1117 data-null projector use. This is what makes
/// the value and the analytic outer ρ-gradient consistent: a ridge fallback
/// (`+ridge·I`) injects a ρ-dependent bias `½·log|I + ridge·H_tt⁻¹|` into the
/// VALUE that the analytic gradient (built for the undamped Laplace log-det)
/// never sees, desyncing the outer line-search; unit-stiffness deflation has no
/// such bias because the deflated direction's contribution is the ρ-independent
/// constant `0`. Returns `None` only if the block is non-finite or the
/// eigendecomposition fails (the caller then surfaces the hard refusal).
pub(crate) fn factor_spectral_deflated_evidence_row(
    row: &ArrowRowBlock,
    d: usize,
) -> Option<ArrowRowFactorResult> {
    if d == 0 || row.htt.dim() != (d, d) {
        return None;
    }
    // Symmetrise defensively before the eigendecomposition (the assembled
    // block is symmetric up to reduction order; the eig routine assumes exact
    // symmetry).
    let mut sym = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let v = 0.5 * (row.htt[[i, j]] + row.htt[[j, i]]);
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
    let floor = SPECTRAL_DEFLATION_REL_FLOOR * max_abs;
    // Hysteresis-banded deflation floor for *positive* near-cutoff eigenvalues.
    // The bare `floor` is a knife-edge: a small positive curvature direction
    // parked at ~`floor` toggles deflated/not-deflated as ρ/θ move, flipping the
    // per-row count and tripping the quotient-dimension guard spuriously
    // (#1117). We deflate a positive eigenvalue only once it drops below the
    // LOWER band edge `floor·(1−ε)`, so a value oscillating around the bare
    // floor stays consistently KEPT. Non-positive / non-finite eigenvalues
    // (genuine null / indefinite quotient directions) are still always deflated
    // at the exact-zero boundary, which the guard must continue to honour.
    let deflate_floor = floor * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
    // Reconstruct `Σ_i λ̃_i v_i v_iᵀ`, replacing every deflated eigenvalue
    // (every non-positive/non-finite one, plus any positive one that has
    // dropped below the hysteresis floor) with unit stiffness `+1` and keeping
    // the genuine positive spectrum untouched.
    //
    // PD GUARANTEE (#1118). This function is reached ONLY after the genuine
    // undamped Cholesky has already REFUSED the block (`factor_one_row_result`
    // Err arm), so we are committed to delivering a PD factor — declining here
    // surfaces the hard "non-PD per-row H_tt" refusal that kills the whole K>1
    // fit. The previous code declined in two silent ways that violated that
    // contract on a barely-(non)-PD knife-edge: (1) `deflated_count == 0`, when
    // the symmetric `eigh` rounds the offending direction to a tiny-POSITIVE
    // eigenvalue just above `deflate_floor` while the unrolled scalar Cholesky
    // underflowed its pivot to `≤ 0`; and (2) the reconstruction's own Cholesky
    // failing because a kept eigenvalue was positive but `≪ floor`, so the
    // assembled `Σ λ v vᵀ` was numerically indefinite. Both routed a genuinely
    // non-PD block to the hard refusal even though a valid quotient factor
    // exists. We instead FLOOR every reconstructed eigenvalue to a strictly
    // positive `floor`: a direction at or below the hysteresis edge is deflated
    // to unit stiffness `+1` (the ρ-independent `log 1 = 0` quotient convention),
    // and any other near-floor positive direction is clamped UP to `floor` so
    // the assembled block is PD by construction and the Cholesky cannot fail.
    // The genuine, well-separated positive spectrum (`λ ≫ floor`) is untouched,
    // so every block the old path already conditioned is bit-for-bit unchanged.
    let mut conditioned = Array2::<f64>::zeros((d, d));
    let mut deflated_count = 0usize;
    for eig_idx in 0..evals.len() {
        let lambda = evals[eig_idx];
        let lambda_tilde = if lambda.is_finite() && lambda > deflate_floor {
            // Genuine positive direction: keep it, but clamp UP to the positive
            // `floor` so a tiny-but-kept eigenvalue cannot make the reconstructed
            // block numerically non-PD (it never lowers a healthy `λ ≫ floor`).
            lambda.max(floor)
        } else {
            // Null / indefinite / numerically-flat quotient direction: unit
            // stiffness `+1`, contributing `log 1 = 0` to the evidence log-det.
            deflated_count += 1;
            1.0
        };
        for i in 0..d {
            let vi = evecs[[i, eig_idx]];
            for j in 0..d {
                conditioned[[i, j]] += lambda_tilde * vi * evecs[[j, eig_idx]];
            }
        }
    }
    if deflated_count == 0 {
        // The hysteresis band kept every direction (the offending eigenvalue
        // rounded just above `deflate_floor`), yet the genuine Cholesky still
        // refused the block — a barely-non-PD knife-edge. We are on the refused
        // path, so we must not decline: deflate the single smallest-eigenvalue
        // direction to unit stiffness, which removes the marginal pivot while
        // leaving the rest of the (now positive-floored) spectrum exact.
        let mut min_idx = 0usize;
        let mut min_lambda = f64::INFINITY;
        for eig_idx in 0..evals.len() {
            let lambda = evals[eig_idx];
            if lambda < min_lambda {
                min_lambda = lambda;
                min_idx = eig_idx;
            }
        }
        // Subtract the kept (floored) contribution of `min_idx` and add unit
        // stiffness in its place: `conditioned += (1 − λ̃_min) v_min v_minᵀ`.
        let kept = min_lambda.max(floor);
        let delta = 1.0 - kept;
        for i in 0..d {
            let vi = evecs[[i, min_idx]];
            for j in 0..d {
                conditioned[[i, j]] += delta * vi * evecs[[j, min_idx]];
            }
        }
        deflated_count = 1;
    }
    let factor = cholesky_lower(&conditioned).ok()?;
    Some(ArrowRowFactorResult {
        factor,
        gauge_deflated_directions: deflated_count,
    })
}

pub(crate) fn cholesky_solve_vector_fixed<const D: usize>(
    l: ArrayView2<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> Array1<f64> {
    // Precondition: `l` is a Cholesky factor whose diagonals are strictly
    // positive and finite (every f64 factor in this module is produced by
    // `cholesky_lower`, which rejects `!is_finite() || sum <= 0.0` pivots). The
    // back/forward substitution below divides by `l[[i, i]]` with no per-row
    // guard; a future caller that hands an unvalidated factor here would emit a
    // silent `NaN` into the Schur reduction (#1038). Catch that loudly —
    // always, release included — rather than letting it flow into the
    // evidence/gradient. The check is O(D) over a small fixed-size factor, so
    // it is negligible next to the substitution it guards.
    assert!(
        (0..D).all(|i| l[[i, i]].is_finite() && l[[i, i]].abs() >= f64::MIN_POSITIVE),
        "cholesky_solve_vector_fixed: factor diagonal must be finite and non-subnormal"
    );
    let mut y = [0.0_f64; D];
    for i in 0..D {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }

    let mut x = [0.0_f64; D];
    for i in (0..D).rev() {
        let mut sum = y[i];
        for k in (i + 1)..D {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }

    let mut out = Array1::<f64>::zeros(D);
    for i in 0..D {
        out[i] = x[i];
    }
    out
}

pub(crate) fn factor_one_row(
    row: &ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    row_idx: usize,
    tolerate_ill_conditioning: bool,
) -> Result<Array2<f64>, ArrowSchurError> {
    // Generic / non-evidence callers (CPU/GPU `factor_blocks`, the system.rs
    // assembly loops) supply no gauge directions AND do not install a row-gauge
    // deflation, so they must NOT spectrally discover-and-deflate a flat
    // direction — a genuinely non-PD block has to surface as a typed error for
    // their outer ridge/rebuild handling. Only the SAE evidence path that
    // installs a `row_gauge_deflation` (via `factor_blocks_for_system`) opts
    // into spectral deflation.
    factor_one_row_result(
        row,
        ridge_t,
        d,
        row_idx,
        tolerate_ill_conditioning,
        &[],
        false,
    )
    .map(|result| result.factor)
}

pub(crate) fn factor_one_row_result(
    row: &ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    row_idx: usize,
    tolerate_ill_conditioning: bool,
    row_gauges: &[Array1<f64>],
    allow_spectral_deflation: bool,
) -> Result<ArrowRowFactorResult, ArrowSchurError> {
    // Dimension mismatches in caller-supplied row blocks must surface as a
    // typed error rather than aborting the process. The BA/SAE assembler can
    // mis-size a row (for instance when latent_dim disagrees between the
    // design and the term that materialized the block), and downstream code
    // — including the LM outer loop — needs to recover by escalating ridge
    // or rebuilding the system, not by panicking.
    if row.htt.dim() != (d, d) {
        return Err(ArrowSchurError::PerRowFactorFailed {
            row: row_idx,
            reason: format!(
                "row {row_idx} H_tt shape {:?} does not match per_point_hessian_block dimension ({d}, {d})",
                row.htt.dim()
            ),
        });
    }
    if row.gt.len() != d {
        return Err(ArrowSchurError::PerRowFactorFailed {
            row: row_idx,
            reason: format!(
                "row {row_idx} g_t length {} does not match latent dimension {d}",
                row.gt.len()
            ),
        });
    }
    // Per-row adaptive Tikhonov ridge. A non-convex objective (e.g. softmax
    // assignment) can leave an individual token's latent Hessian H_tt^(i)
    // indefinite, so `H_tt + ridge_t·I` has a negative Cholesky pivot. Rather
    // than fail and force the OUTER LM loop to lift `ridge_t` for EVERY row
    // (over-damping the well-conditioned tokens), damp only this block by the
    // minimal amount it needs: escalate this row's ridge geometrically from the
    // caller's base `ridge_t` until the factor is positive-definite. A
    // positive-definite block factors at the base ridge with zero escalation,
    // so the common case is bit-for-bit unchanged. The escalation is capped
    // relative to the block's diagonal scale, so a genuinely broken block
    // (non-finite, or unboundedly indefinite) still surfaces as
    // `PerRowFactorFailed` for the outer loop to handle rather than looping.
    // Per-row ridge escalation policy. The escalation starts at the caller's
    // base ridge (or, if that is zero, a tiny seed scaled by the block's
    // diagonal magnitude), multiplies geometrically each rejection, and is
    // capped at a large multiple of the base scale so a genuinely broken block
    // surfaces as an error instead of looping forever.
    const RIDGE_GROWTH_FACTOR: f64 = 10.0;
    const RIDGE_SEED_DIAG_FRACTION: f64 = 1.0e-10;
    const RIDGE_CAP_DIAG_FRACTION: f64 = 1.0e-12;
    const RIDGE_CAP_SCALE: f64 = 1.0e12;
    let diag_scale = row_block_diag_scale(row, d);
    let ridge_cap = ridge_t.max(RIDGE_CAP_DIAG_FRACTION * diag_scale) * RIDGE_CAP_SCALE;
    let mut ridge_eff = ridge_t;
    // Escalate the per-row ridge until the block is BOTH positive-definite AND
    // well-conditioned. Previously the escalation only fired on a *failed*
    // Cholesky (indefinite block); a barely-PD but ill-conditioned block
    // (pivots ~ε·trace — e.g. a rank-deficient / over-parameterized decoder
    // atom) factored successfully and was then rejected outright as
    // `PerRowFactorIllConditioned`, so the ridge the SAE audit advertises
    // ("the Arrow-Schur ridge will regularise the deficient directions") never
    // got the chance to. Folding the κ proxy into the loop lets the ridge lift
    // just enough to regularise the deficient directions, as advertised,
    // instead of aborting the whole fit (gam#578). A genuinely PD,
    // well-conditioned block factors at the base ridge with zero escalation and
    // is bit-for-bit unchanged; only a block that cannot be conditioned even at
    // `ridge_cap` (1e12 × base) still surfaces an error for the outer loop.
    let factor = loop {
        match factor_row_block_cholesky(row, ridge_eff, d) {
            Ok(factor) => {
                // Evidence/log-det-only callers tolerate ill-conditioning: the
                // factor is genuinely PD, so its diagonal gives an exact log|S|
                // and an inaccurate Δβ would be discarded anyway.
                if tolerate_ill_conditioning {
                    if ridge_t == 0.0
                        && !row_gauges.is_empty()
                        && let Some(deflated) =
                            factor_gauge_deflated_evidence_row(row, d, row_gauges)
                    {
                        return Ok(deflated);
                    }
                    // #1377 — route-independent intrinsic-dimension-flat recovery.
                    // At `ridge_t = 0` the undamped per-row Cholesky of a #1273
                    // intrinsic-dimension-flat block sits on a knife-edge: the
                    // marginal pivot of the flat direction rounds to a *tiny
                    // positive* value on one route (Cholesky succeeds → this `Ok`
                    // arm) and to `≤ 0` on the other (Cholesky fails → the `Err`
                    // arm below). The `Err` arm already deflates that flat
                    // direction to UNIT stiffness via
                    // `factor_spectral_deflated_evidence_row` (`log 1 = 0`
                    // contribution), but this `Ok` arm previously returned the RAW
                    // barely-PD factor whose tiny pivot contributes a large
                    // `2·ln(√ε)` instead. The two memory-budget routes (dense
                    // `factor_blocks_for_system` vs streaming
                    // `reduced_schur_and_log_det_tt`) then disagreed on the
                    // per-row log-det, breaking the streaming-plan identical-logdet
                    // invariant and surfacing as a NON-PD `H_tt` on whichever route
                    // landed on the failing side of the edge.
                    //
                    // Unify the two arms: when this is the SAE evidence path
                    // (`allow_spectral_deflation`) and the just-computed factor is
                    // NOT safely invertible (a near-flat / rank-deficient block, by
                    // the SAME κ / min-pivot proxy used for the Δβ-accuracy gate
                    // below), deflate it through the identical spectral recovery the
                    // `Err` arm uses. A genuinely well-conditioned PD block
                    // (`λ ≫ floor`) passes the safe-inversion proxy and is returned
                    // bit-for-bit unchanged, so #1273's recovery and every healthy
                    // block are untouched — only the marginal flat direction is now
                    // deflated identically regardless of which side of the pivot
                    // knife-edge a route lands on.
                    if ridge_t == 0.0 && allow_spectral_deflation {
                        let kappa_est = cholesky_factor_kappa_estimate(&factor);
                        if !cholesky_factor_passes_safe_inversion(&factor, d, diag_scale, kappa_est)
                            && let Some(deflated) = factor_spectral_deflated_evidence_row(row, d)
                        {
                            return Ok(deflated);
                        }
                    }
                    break ArrowRowFactorResult {
                        factor,
                        gauge_deflated_directions: 0,
                    };
                }
                // Diagonal-ratio condition-number proxy κ(LLᵀ) ≈
                // (max L_ii / min L_ii)², vs the dimension-scaled Higham
                // near-singularity ceiling. A barely-PD inverse plugged into
                //   S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
                // contaminates S by spectral terms scaled by κ_i, so an
                // over-threshold block is regularised further rather than used.
                let kappa_est = cholesky_factor_kappa_estimate(&factor);
                if cholesky_factor_passes_safe_inversion(&factor, d, diag_scale, kappa_est) {
                    break ArrowRowFactorResult {
                        factor,
                        gauge_deflated_directions: 0,
                    };
                }
                let next = if ridge_eff > 0.0 {
                    ridge_eff * RIDGE_GROWTH_FACTOR
                } else {
                    RIDGE_SEED_DIAG_FRACTION * diag_scale
                };
                if !next.is_finite() || next > ridge_cap {
                    return Err(ArrowSchurError::PerRowFactorIllConditioned {
                        row: row_idx,
                        kappa_estimate: kappa_est,
                    });
                }
                ridge_eff = next;
            }
            Err(e) => {
                // Evidence/log-det callers (`tolerate_ill_conditioning = true`)
                // consume the returned factor's diagonal as the exact
                // log|H_tt + ridge_t·I|. Silently lifting ridge past the
                // caller's base would shift that determinant by Σ d·log(1+δ/λ)
                // while returning Ok, corrupting the reported evidence. A
                // genuinely non-PD block at the base ridge must surface as
                // an error here, not be quietly conditioned.
                if tolerate_ill_conditioning {
                    if ridge_t == 0.0 {
                        if let Some(deflated) =
                            factor_gauge_deflated_evidence_row(row, d, row_gauges)
                        {
                            // Faddeev-Popov row-gauge deflation: only the
                            // closed-form orbit direction is stiffened, at UNIT
                            // stiffness kappa = 1.0, so each deflated direction
                            // contributes log(1) = 0 to log|H| — the quotient
                            // pseudo-determinant convention (the gauge orbit is a
                            // criterion null direction, contributing nothing to
                            // the Laplace normalizer). Zero theta/rho dependence,
                            // so criterion derivatives stay exact on the quotient.
                            return Ok(deflated);
                        }
                        // #1117/#1118/#1273 — the offending direction is NOT a
                        // supplied gauge vector. Two distinct geometries reach
                        // here: (a) under K>1 IBP/softmax row-sharing the
                        // logit×coordinate Gauss-Newton cross term drives an
                        // eigenvalue of this row's H_tt negative at a direction the
                        // closed-form gauge orbit does not span; and (b) — the
                        // #1273 circle/torus case — a single atom whose data is
                        // intrinsically LOWER-dimensional than its chart (a 1-D ring
                        // embedded in a 2-D torus harmonic basis) has a genuine FLAT
                        // tangent direction: H_tt is rank-deficient even though the
                        // REML cost is finite and valid. In (b) the supplied row
                        // gauge is only the rotation/phase orbit, which does NOT span
                        // the intrinsic-dimension flat direction, so that row's gauge
                        // list can be empty (or non-spanning) yet a valid quotient
                        // factor exists. In BOTH cases DISCOVER the offending
                        // direction from the block's own symmetric eigendecomposition
                        // and deflate it at the SAME unit stiffness (eigenvalue → +1),
                        // so its evidence contribution is the ρ-independent constant
                        // log 1 = 0. This replaces the previous ridge-damped evidence
                        // fallback, whose ½·log|I + ridge·H_tt⁻¹| bias was
                        // ρ-DEPENDENT and therefore desynced the outer REML value
                        // (which saw it) from the analytic ρ-gradient (built for
                        // the undamped Laplace log-det, which did not) — the
                        // multi-atom outer line-search non-convergence (#1117).
                        //
                        // The undamped exact Cholesky still owns every genuinely PD
                        // block (this arm is reached only on a refused factor), and
                        // only the SAE evidence path opts into spectral discovery
                        // (`allow_spectral_deflation`); generic callers keep the
                        // strict non-PD refusal. The previous `!row_gauges.is_empty()`
                        // gate spuriously withheld this recovery from a row whose
                        // flat direction was intrinsic-dimension deficiency rather
                        // than a supplied gauge — exactly the #1273 abort.
                        if allow_spectral_deflation
                            && let Some(deflated) = factor_spectral_deflated_evidence_row(row, d)
                        {
                            return Ok(deflated);
                        }
                    }
                    return Err(ArrowSchurError::PerRowFactorFailed {
                        row: row_idx,
                        reason: format!(
                            "row {row_idx} H_tt is non-PD at base ridge {ridge_t:e}; \
                             evidence mode preserves the genuine Cholesky of \
                             H_tt and does not condition non-PD blocks: {e}"
                        ),
                    });
                }
                let next = if ridge_eff > 0.0 {
                    ridge_eff * RIDGE_GROWTH_FACTOR
                } else {
                    RIDGE_SEED_DIAG_FRACTION * diag_scale
                };
                if !next.is_finite() || next > ridge_cap {
                    return Err(ArrowSchurError::PerRowFactorFailed {
                        row: row_idx,
                        reason: format!(
                            "row {row_idx} H_tt remained non-PD up to ridge {ridge_eff:e} \
                             (base ridge_t={ridge_t}); last cholesky error: {e}"
                        ),
                    });
                }
                ridge_eff = next;
            }
        }
    };
    Ok(factor)
}

pub(crate) fn manifold_mode_fingerprint(latent: &LatentCoordValues) -> u64 {
    let manifold = latent.manifold();
    if manifold.is_euclidean() {
        return EUCLIDEAN_MANIFOLD_MODE_FINGERPRINT;
    }

    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-manifold-mode-v1");
    hasher.write_usize(latent.n_obs());
    hasher.write_usize(latent.latent_dim());
    write_latent_manifold(&mut hasher, manifold);
    let mut metric_weights = Vec::new();
    append_latent_metric_weights(&mut metric_weights, manifold);
    hasher.write_usize(metric_weights.len());
    for weight in metric_weights {
        hasher.write_f64(weight);
    }
    hasher.finish_u64()
}

pub(crate) fn row_hessian_fingerprint_for_system(sys: &ArrowSchurSystem) -> u64 {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-row-hessian-v2");
    hasher.write_usize(sys.rows.len());
    hasher.write_usize(sys.d);
    hasher.write_usize(sys.k);
    // When htbeta_matvec is installed (Kronecker / matrix-free path),
    // row.htbeta is usually a zero slab that does not capture the operator
    // state. Hash the Arc pointer address as a proxy: a new Arc is allocated
    // per assemble call, so the fingerprint is invalidated each time the
    // system is rebuilt with a fresh Kronecker operator. Analytic penalties may
    // opt into a dense supplemental slab; when active, hash it as well.
    // SAFETY: We cast the fat pointer to a thin *const () to extract the data
    // pointer address as a fingerprint proxy. No dereference occurs; the only
    // use is as a usize hash input, which is sound for any aligned pointer.
    let htbeta_op_addr: Option<usize> = sys
        .htbeta_matvec
        .as_ref()
        .map(|op| Arc::as_ptr(op) as *const () as usize);
    for row in sys.rows.iter() {
        hasher.write_f64_array2(&row.htt);
        match htbeta_op_addr {
            Some(addr) => {
                hasher.write_usize(addr);
                if sys.htbeta_dense_supplement {
                    hasher.write_f64_array2(&row.htbeta);
                }
            }
            None => hasher.write_f64_array2(&row.htbeta),
        }
    }
    // Hash the β-block operator's defining state. When a structured
    // `penalty_op` is installed (e.g. the SAE composite carrying the data-fit
    // Gauss-Newton block as `G ⊗ I_p`), hashing the operator captures the full
    // β-block content cheaply; the dense `sys.hbb` no longer holds it. When no
    // `penalty_op` is installed, fall back to hashing the dense accumulator.
    match sys.penalty_op.as_ref() {
        Some(op) => {
            hasher.write_bool(true);
            op.fingerprint(&mut hasher);
        }
        None => {
            hasher.write_bool(false);
            hasher.write_f64_array2(&sys.hbb);
        }
    }
    match sys.hbb_diag.as_ref() {
        Some(diag) => {
            hasher.write_bool(true);
            hasher.write_usize(diag.len());
            for &value in diag.iter() {
                hasher.write_f64(value);
            }
        }
        None => hasher.write_bool(false),
    }
    hasher.finish_u64()
}

pub(crate) fn combine_row_and_registry_fingerprints(row: u64, registry: u64) -> u64 {
    if registry == 0 {
        return row;
    }
    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-row-hessian-with-penalties-v1");
    hasher.write_u64(row);
    hasher.write_u64(registry);
    hasher.finish_u64()
}

pub(crate) fn analytic_penalty_row_hessian_fingerprint(
    penalty: &AnalyticPenaltyKind,
    target_t: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
) -> Option<u64> {
    if penalty.tier() != PenaltyTier::Psi || !analytic_penalty_is_row_block_diagonal(penalty) {
        return None;
    }

    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-analytic-row-hessian-v1");
    hasher.write_str(penalty.name());
    hasher.write_usize(target_t.len());
    hasher.write_usize(rho_local.len());
    for &rho in rho_local.iter() {
        hasher.write_f64(rho);
    }

    match penalty {
        AnalyticPenaltyKind::RowPrecisionPrior(p) => {
            let (n, rows, cols) = p.lambda_per_row.dim();
            hasher.write_str("row-precision-fixed");
            hasher.write_usize(n);
            hasher.write_usize(rows);
            hasher.write_usize(cols);
            hasher.write_f64(p.weight);
            hasher.write_bool(p.learnable_weight);
            if p.learnable_weight {
                hasher.write_usize(p.rho_index);
                hasher.write_f64(p.weight * rho_local[p.rho_index].exp());
            }
            for &value in p.lambda_per_row.iter() {
                hasher.write_f64(value);
            }
        }
        AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
            let (aux_n, aux_dim) = p.aux.dim();
            let (mu_rows, mu_cols) = p.mu.dim();
            let weight_offset = p.log_alpha.len() + p.raw_beta.len() + p.mu.len();
            hasher.write_str("row-precision-parametric");
            hasher.write_usize(aux_n);
            hasher.write_usize(aux_dim);
            hasher.write_usize(mu_rows);
            hasher.write_usize(mu_cols);
            hasher.write_f64(p.weight);
            hasher.write_bool(p.learnable_weight);
            for &value in p.aux.iter() {
                hasher.write_f64(value);
            }
            for k in 0..p.log_alpha.len() {
                let active_log_alpha = p.log_alpha[k] + rho_local[k];
                hasher.write_f64(p.log_alpha[k]);
                hasher.write_f64(active_log_alpha);
                hasher.write_f64(active_log_alpha.exp());
            }
            let raw_beta_offset = p.log_alpha.len();
            for k in 0..p.raw_beta.len() {
                let active_raw_beta = p.raw_beta[k] + rho_local[raw_beta_offset + k];
                hasher.write_f64(p.raw_beta[k]);
                hasher.write_f64(active_raw_beta);
                hasher.write_f64(gam_linalg::utils::stable_softplus(active_raw_beta));
            }
            let mu_offset = p.log_alpha.len() + p.raw_beta.len();
            for k in 0..p.mu.nrows() {
                for a in 0..p.mu.ncols() {
                    let idx = mu_offset + k * p.aux.ncols() + a;
                    hasher.write_f64(p.mu[[k, a]]);
                    hasher.write_f64(p.mu[[k, a]] + rho_local[idx]);
                }
            }
            if p.learnable_weight {
                hasher.write_usize(weight_offset);
                hasher.write_f64(p.weight * rho_local[weight_offset].exp());
            }
        }
        _ => {
            hasher.write_str("row-block-diagonal");
            if let Some(diag) = penalty.hessian_diag(target_t, rho_local) {
                hasher.write_usize(diag.len());
                for &value in diag.iter() {
                    hasher.write_f64(value);
                }
            } else {
                hasher.write_usize(0);
            }
        }
    }

    Some(hasher.finish_u64())
}

/// Structural/value fingerprint for a cross-row (non-row-block-diagonal)
/// Psi-tier analytic penalty.
///
/// Unlike [`analytic_penalty_row_hessian_fingerprint`], which can read a
/// closed-form per-row diagonal, a cross-row penalty's curvature only surfaces
/// through its Hessian-vector product. We probe the penalty's PSD majorizer
/// against the *current latent vector itself* — a deterministic, penalty- and
/// state-dependent probe — and hash the resulting vector together with the
/// penalty name, target length, and local ρ. Any change to the operator that
/// matters for the Newton solve (different ρ, different smoothing geometry,
/// different latent linearization point) perturbs this probe, correctly
/// invalidating any factor cache keyed on the row-Hessian fingerprint.
pub(crate) fn cross_row_penalty_fingerprint(
    penalty: &AnalyticPenaltyKind,
    target_t: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
) -> u64 {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("arrow-schur-analytic-cross-row-hessian-v1");
    hasher.write_str(penalty.name());
    hasher.write_usize(target_t.len());
    hasher.write_usize(rho_local.len());
    for &rho in rho_local.iter() {
        hasher.write_f64(rho);
    }
    let probe = penalty.psd_majorizer_hvp(target_t, rho_local, target_t);
    hasher.write_usize(probe.len());
    for &value in probe.iter() {
        hasher.write_f64(value);
    }
    hasher.finish_u64()
}

pub(crate) fn write_latent_manifold(hasher: &mut Fingerprinter, manifold: &LatentManifold) {
    match manifold {
        LatentManifold::Euclidean => {
            hasher.write_str("euclidean");
        }
        LatentManifold::Circle { period } => {
            hasher.write_str("circle");
            hasher.write_f64(*period);
        }
        LatentManifold::Sphere { dim } => {
            hasher.write_str("sphere");
            hasher.write_usize(*dim);
        }
        LatentManifold::Interval { lo, hi } => {
            hasher.write_str("interval");
            hasher.write_f64(*lo);
            hasher.write_f64(*hi);
        }
        LatentManifold::Product(parts) => {
            hasher.write_str("product");
            hasher.write_usize(parts.len());
            for part in parts {
                write_latent_manifold(hasher, part);
            }
        }
        LatentManifold::ProductWithMetric { manifolds, weights } => {
            hasher.write_str("product-with-metric");
            hasher.write_usize(manifolds.len());
            for part in manifolds {
                write_latent_manifold(hasher, part);
            }
            hasher.write_usize(weights.len());
            for weight in weights {
                hasher.write_f64(*weight);
            }
        }
    }
}

pub(crate) fn append_latent_metric_weights(out: &mut Vec<f64>, manifold: &LatentManifold) {
    match manifold {
        LatentManifold::Euclidean => out.push(1.0),
        LatentManifold::Circle { period } => {
            out.push(1.0 / (period * period));
        }
        LatentManifold::Sphere { dim } => {
            let scale = std::f64::consts::PI;
            for _ in 0..*dim {
                out.push(1.0 / (scale * scale));
            }
        }
        LatentManifold::Interval { lo, hi } => {
            let scale = hi - lo;
            out.push(1.0 / (scale * scale));
        }
        LatentManifold::Product(parts) => {
            for part in parts {
                append_latent_metric_weights(out, part);
            }
        }
        LatentManifold::ProductWithMetric {
            manifolds: _,
            weights,
        } => {
            out.extend(weights.iter().copied());
        }
    }
}
