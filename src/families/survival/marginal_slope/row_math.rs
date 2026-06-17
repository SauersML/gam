//! Closed-form per-row marginal-slope kernel mathematics: the rigid/shared/
//! vector closed forms, the analytic derivative helpers (`c_derivatives`,
//! `neglog_derivatives`), the pilot-eta geometry solvers, the public vector
//! NLL/eta/scale helpers, latent-z standardization, and the eval cache.

use super::*;

// ── Closed-form row kernel ─────────────────────────────────────────────
//
// The survival marginal-slope NLL for row i is:
//
//   ℓ_i = w_i [ (1-d)·neglogΦ(-η₁) + logΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
//
// with η₀ = q₀c + s_f g z, η₁ = q₁c + s_f g z, a'₁ = qd₁·c,
// c = √(1 + (s_f g)²), s_f = 1/√(1+σ²).
//
// All derivatives w.r.t. the 4 primary scalars (q₀, q₁, qd₁, g) are
// closed-form scalar formulas. No jets, no per-row heap allocation.

#[inline]
pub(crate) fn rigid_observed_logslope(g: f64, probit_scale: f64) -> f64 {
    probit_scale * g
}

#[inline]
pub(crate) fn rigid_observed_scale(g: f64, probit_scale: f64) -> f64 {
    let observed_g = rigid_observed_logslope(g, probit_scale);
    (1.0 + observed_g * observed_g).sqrt()
}

#[inline]
pub(crate) fn rigid_observed_eta(q: f64, g: f64, z: f64, probit_scale: f64) -> f64 {
    q * rigid_observed_scale(g, probit_scale) + rigid_observed_logslope(g, probit_scale) * z
}

/// Survival row Hessian row metric at a β-independent pilot η, used to
/// build the W inner product for cross-block orthogonalisation of the
/// score-warp and link-deviation flex bases. The previous implementation
/// copied the Bernoulli probit IRLS row weight
/// `w · φ(η)² / (Φ(η)·(1 − Φ(η)))` from BMS verbatim, but that is the row
/// curvature for a Bernoulli probit likelihood, not for the survival
/// marginal-slope likelihood. The survival row neg-log Hessian wrt η₁
/// (the dominant linear-predictor channel — both event and censored rows
/// contribute through it) at fixed β is
///
///   u2_eta1[i] = (1 − d_i) · w_i · d²/dη²[−log Φ(−η_i)]
///              + d_i · w_i
///
/// where the first term is the censored-row Mills-ratio second derivative
/// computed via `signed_probit_neglog_derivatives_up_to_fourth(-η, w·(1−d))`
/// (matching `row_primary_closed_form` at ~3483 column `u2_eta1`) and the
/// second term is the event-row contribution from `−log φ(η) = η²/2 + …`,
/// whose η-Hessian is `w·d`. This is exactly the curvature the joint
/// penalised Hessian sees along the dominant linear-predictor direction at
/// the pilot, so `Aᵀ W C̃ = 0` after the cross-block reparam is preserved in
/// the inner product PIRLS uses (modulo β-dependent drift between pilot and
/// running η — second-order in the off-anchor direction and bounded by the
/// Mills-ratio curvature scaling shared between both branches).
///
/// The pilot η is β-independent so this remains a one-shot construction-time
/// step. See the long comment block in bernoulli_marginal_slope.rs:19180 for
/// the BMS analogue and the failure mode the metric prevents (REML can
/// otherwise collapse the alias eigenvalue when `Aᵀ W_pirls C̃ ≠ 0`).
///
/// # Residual approximation
///
/// This is the η₁-channel row curvature. The full survival row Hessian is
/// 4×4 in primary scalars `(q₀, q₁, qd₁, g)` and chains differently into
/// each block (time/marginal share `dη₁/dq = c`; logslope chains through
/// `dη₁/dg = q₁·c₁ + s_f·z`; flex bases chain through `dη₁/dη = 1`). The
/// cross-block orthogonalisation here is therefore exact in the η₁
/// direction and approximate along the η₀ and ad₁ directions and between
/// blocks whose chain factors differ. In practice η₁ dominates because
/// both event and censored rows contribute through it while only entry
/// contributes to η₀; the alias structure the audit reported on the
/// large-scale fit is along the η₁ channel (time ↔ marginal ↔ logslope
/// ↔ score_warp ↔ link_dev all share constant + low-order columns that
/// project onto η₁). A fully chain-corrected metric is exactly what
/// `identifiability::families::compiler` (Phase 3, family-agnostic
/// `RowJacobianOperator` / `RowHessian` driver)
/// provides as the canonical home; this SMGS pre-PIRLS pilot reparam is
/// the principled scope for *alias killing only*. Two blocks with the
/// same chain factor (time ↔ marginal here, both `dη₁/dq = c`) produce
/// identical η contributions iff their bare-design columns are linearly
/// dependent — killed by bare-design orthogonality regardless of chain.
/// Different-chain aliases (marginal ↔ logslope) require alias structure
/// that exactly matches the chain ratio `(q₁·c₁ + s_f·z)/c`, a
/// degenerate case not observed. The large-scale alias chain the audit
/// reported (`time ↔ marginal ↔ logslope ↔ score_warp ↔ link_dev`) is
/// driven by shared constant + low-order columns — chain-independent and
/// killed by this scalar W. The reparam is one-shot and pre-PIRLS by
/// necessity: PIRLS itself cannot proceed on the aliased design.
pub(crate) fn survival_pilot_irls_row_metric_at_eta(
    eta_pilot: &Array1<f64>,
    sample_weights: &Array1<f64>,
    event: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let n = eta_pilot.len();
    if sample_weights.len() != n || event.len() != n {
        return Err(format!(
            "survival cross-block W metric: length mismatch eta={}, weights={}, event={}",
            n,
            sample_weights.len(),
            event.len(),
        ));
    }
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta = eta_pilot[i];
        let d = event[i];
        let weight = sample_weights[i];
        let (_, k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-eta, weight * (1.0 - d))?;
        let phi_part = weight * d;
        w[i] = k2 + phi_part;
    }
    Ok(w)
}

/// Build the SMGS rigid pooled-probit pilot η at training rows from the
/// time-block offsets, marginal/logslope offsets, baseline slope and z. This
/// is the survival analog of BMS `rigid_pooled_probit_pilot_eta`; the basis
/// is intentionally β-independent so the resulting W metric depends only on
/// data + spec offsets and the cross-block orthogonalisation remains a one-
/// shot construction-time step. Used to weight the W-metric inner product
/// in `install_compiled_flex_block_into_runtime` (both flex paths).
pub(crate) fn survival_rigid_pilot_eta(
    n: usize,
    z_primary: &Array1<f64>,
    offset_exit: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    baseline_slope: f64,
    probit_scale: f64,
) -> Array1<f64> {
    Array1::from_iter((0..n).map(|row| {
        let q_exit = offset_exit[row] + marginal_offset[row];
        let slope = baseline_slope + logslope_offset[row];
        rigid_observed_eta(q_exit, slope, z_primary[row], probit_scale)
    }))
}

/// One-step IRLS refinement of the rigid pilot η along the dominant η₁ row
/// channel. Starts from `survival_rigid_pilot_eta` (offset+baseline), runs
/// a single weighted Newton step over the joint location-anchor + logslope
/// design `X = [T_exit | M | G]`, and returns the refined per-row η₁ pilot
/// that the cross-block W metric uses. Prevents the flex-anchor bases
/// (`link_dev`, `score_warp`) from collapsing onto the same constant scalar
/// path that the offset-only seed would produce when training offsets are
/// uniform — the failure mode the original large-scale audit pinned to a
/// 13-dimensional unidentified quotient. The η₁ direction matches the doc
/// scope of `survival_pilot_irls_row_metric_at_eta` (see its long block);
/// the chain factor `dη₁/dq = c(g)` is absorbed into a per-row scaling of
/// the location anchor before the solve.
///
/// Returns `(eta1, beta_logslope)`: the per-row observed index `eta1` (used by
/// the cross-block W metric, unchanged from the legacy scalar return) AND the
/// one-step IRLS estimate of the logslope-surface coefficients `beta_logslope`
/// (the `G`-block portion of the joint Newton step). The latter is the #808
/// operating-point WARM START for the logslope block's `initial_beta`: on
/// clustered-PC designs the logslope block is EXACTLY W-null at the `g = 0`
/// seed (the slope-channel IRLS weight vanishes at the null slope), so the
/// inner joint-Newton cannot take its first step and freezes; seeding the
/// block at the pilot's `g ≈ 0.3` operating point (where the slope channel
/// carries information and the block is full-rank) breaks the chicken-and-egg
/// and lets the inner converge to the true data optimum — preserving the
/// log-slope estimand rather than dropping/pinning it. Self-correcting: it is
/// just a warm start, so the converged fit is the data optimum (zero bias).
pub(crate) fn survival_nonrigid_pilot_eta(
    n: usize,
    location_anchor_design: &DesignMatrix,
    logslope_design: &DesignMatrix,
    z_primary: &Array1<f64>,
    offset_exit: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    baseline_slope: f64,
    sample_weights: &Array1<f64>,
    event: &Array1<f64>,
    probit_scale: f64,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    if location_anchor_design.nrows() != n
        || logslope_design.nrows() != n
        || z_primary.len() != n
        || offset_exit.len() != n
        || marginal_offset.len() != n
        || logslope_offset.len() != n
        || sample_weights.len() != n
        || event.len() != n
    {
        return Err(format!(
            "survival_nonrigid_pilot_eta: row-count mismatch (n={n}, location={}, logslope={}, \
             z={}, offset_exit={}, marginal_offset={}, logslope_offset={}, weights={}, event={})",
            location_anchor_design.nrows(),
            logslope_design.nrows(),
            z_primary.len(),
            offset_exit.len(),
            marginal_offset.len(),
            logslope_offset.len(),
            sample_weights.len(),
            event.len(),
        ));
    }
    let p_loc = location_anchor_design.ncols();
    let p_g = logslope_design.ncols();
    let p_joint = p_loc + p_g;
    if p_joint == 0 {
        return Ok((
            survival_rigid_pilot_eta(
                n,
                z_primary,
                offset_exit,
                marginal_offset,
                logslope_offset,
                baseline_slope,
                probit_scale,
            ),
            Array1::<f64>::zeros(p_g),
        ));
    }
    // Starting pilot (offset-only). Decompose into q_exit and slope so the
    // chain rule below can attribute the η₁ Newton step back to each piece.
    let mut q_exit = Array1::<f64>::zeros(n);
    let mut slope = Array1::<f64>::zeros(n);
    let mut eta1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        q_exit[i] = offset_exit[i] + marginal_offset[i];
        slope[i] = baseline_slope + logslope_offset[i];
        eta1[i] = rigid_observed_eta(q_exit[i], slope[i], z_primary[i], probit_scale);
    }
    // Per-row chain factors and IRLS gradient/Hessian along η₁.
    //
    //   η₁ = q·c(g) + s(g)·z   with c(g) = sqrt(1 + s(g)²), s(g) = observed_logslope(g)
    //   ∂η₁/∂q = c(g)
    //   ∂η₁/∂g = q·c'(g) + s'(g)·z
    //
    // The chain factors come from `rigid_observed_eta` via numerical
    // finite-difference on a tiny step (the parametric closed-form
    // derivatives live further down in `c_derivatives`; reusing it would
    // couple this helper to private internals, while finite-difference at
    // 1e-7 is well within the IRLS pilot's accuracy budget — the result is
    // just used to weight the W metric, not propagated into a final
    // coefficient).
    let mut chain_q = Array1::<f64>::zeros(n);
    let mut chain_g = Array1::<f64>::zeros(n);
    let mut grad_eta1 = Array1::<f64>::zeros(n);
    let mut hess_eta1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let g_i = slope[i];
        let z_i = z_primary[i];
        // FD-OK: non-propagated IRLS pilot W-metric weight (bounded central FD, never enters a final coefficient)
        let h_fd: f64 = 1.0e-7;
        chain_q[i] = (rigid_observed_eta(q_exit[i] + h_fd, g_i, z_i, probit_scale)
            - rigid_observed_eta(q_exit[i] - h_fd, g_i, z_i, probit_scale))
            / (2.0 * h_fd);
        chain_g[i] = (rigid_observed_eta(q_exit[i], g_i + h_fd, z_i, probit_scale)
            - rigid_observed_eta(q_exit[i], g_i - h_fd, z_i, probit_scale))
            / (2.0 * h_fd);
        // END-FD-OK
        // Row gradient and Hessian along η₁ (mirror
        // `survival_pilot_irls_row_metric_at_eta`'s formula):
        //   censored:  d(-log Φ(-η))/dη at weight w·(1-d). The primitive
        //              `signed_probit_neglog_derivatives_up_to_fourth(-η, c)`
        //              returns derivatives wrt its first argument; the chain
        //              rule wrt η is therefore  d/dη = -d/d(-η).
        //   event:     d(η²/2)/dη · w·d = w·d·η, Hessian = w·d.
        let (k1, k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(
            -eta1[i],
            sample_weights[i] * (1.0 - event[i]),
        )
        .map_err(|e| format!("survival_nonrigid_pilot_eta: row {i}: {e}"))?;
        let event_w = sample_weights[i] * event[i];
        grad_eta1[i] = -k1 + event_w * eta1[i];
        hess_eta1[i] = k2 + event_w;
        if !hess_eta1[i].is_finite() || hess_eta1[i] < 0.0 {
            // Defensive: any non-PSD row would corrupt the joint Gram. Clamp
            // to a tiny positive so the IRLS step degenerates to a small
            // proximal update rather than producing NaN.
            hess_eta1[i] = hess_eta1[i].max(0.0);
        }
        if !grad_eta1[i].is_finite() {
            grad_eta1[i] = 0.0;
        }
    }
    // Normal equations: (Xᵀ W X + λI) β = -Xᵀ g, where W = diag(hess_eta1)
    // along η₁, g = grad_eta1, and X is the η₁ chain-corrected joint design
    // (each location column scaled by chain_q, each logslope column by
    // chain_g). X is never materialized at full height: a one-shot dense
    // `(n, p_joint)` build was ~0.7 GiB at biobank scale (n=320k) and sat
    // co-resident with the construction phase's other dense transients —
    // one of the contributors to the #979 large-scale OOM. Instead the Gram
    // and rhs accumulate over fixed-height row chunks, so peak extra memory
    // is `chunk × p_joint` regardless of n.
    let mut gram = Array2::<f64>::zeros((p_joint, p_joint));
    let mut rhs = Array1::<f64>::zeros(p_joint);
    const PILOT_ROW_CHUNK: usize = 4096;
    let mut x_chunk = Array2::<f64>::zeros((PILOT_ROW_CHUNK.min(n), p_joint));
    let mut chunk_start = 0usize;
    while chunk_start < n {
        let chunk_end = (chunk_start + PILOT_ROW_CHUNK).min(n);
        let rows = chunk_end - chunk_start;
        let loc_rows = location_anchor_design
            .try_row_chunk(chunk_start..chunk_end)
            .map_err(|e| format!("survival_nonrigid_pilot_eta: location anchor rows: {e}"))?;
        let g_rows = logslope_design
            .try_row_chunk(chunk_start..chunk_end)
            .map_err(|e| format!("survival_nonrigid_pilot_eta: logslope rows: {e}"))?;
        {
            let mut x_view = x_chunk.slice_mut(s![..rows, ..]);
            for local in 0..rows {
                let i = chunk_start + local;
                for j in 0..p_loc {
                    x_view[[local, j]] = chain_q[i] * loc_rows[[local, j]];
                }
                for j in 0..p_g {
                    x_view[[local, p_loc + j]] = chain_g[i] * g_rows[[local, j]];
                }
            }
        }
        let h_chunk = hess_eta1.slice(s![chunk_start..chunk_end]).to_owned();
        let mut neg_g_chunk = Array1::<f64>::zeros(rows);
        for local in 0..rows {
            neg_g_chunk[local] = -grad_eta1[chunk_start + local];
        }
        if rows == x_chunk.nrows() {
            gram += &fast_xt_diag_x(&x_chunk, &h_chunk);
            rhs += &fast_atv(&x_chunk, &neg_g_chunk);
        } else {
            let x_tail = x_chunk.slice(s![..rows, ..]).to_owned();
            gram += &fast_xt_diag_x(&x_tail, &h_chunk);
            rhs += &fast_atv(&x_tail, &neg_g_chunk);
        }
        chunk_start = chunk_end;
    }
    // Adaptive ridge: 1e-6 × average diagonal, floored at 1e-12. Keeps the
    // Cholesky well-conditioned even when the rigid design has near-null
    // directions (which it often does at construction — the whole point of
    // the eventual cross-block reparam).
    let avg_diag = if p_joint > 0 {
        (0..p_joint).map(|j| gram[[j, j]]).sum::<f64>() / (p_joint as f64)
    } else {
        0.0
    };
    let ridge_eff = (1.0e-6 * avg_diag).max(1.0e-12);
    for j in 0..p_joint {
        gram[[j, j]] += ridge_eff;
    }
    let factor = gram
        .cholesky(faer::Side::Lower)
        .map_err(|e| format!("survival_nonrigid_pilot_eta: Cholesky failed: {e:?}"))?;
    let beta_step = factor.solvevec(&rhs);
    // Apply the Newton update: pilot q_exit ← q_exit + chain_q · β_loc,
    // pilot slope ← slope + chain_g · β_g — but the chain factors were
    // already folded into `x_chain` above so the row delta along η₁ is
    // simply `x_chain[i,:] · β_step`. We split it back into q and g by
    // re-projecting through the bare designs (without chain factors).
    let mut beta_loc = Array1::<f64>::zeros(p_loc);
    let mut beta_g = Array1::<f64>::zeros(p_g);
    for j in 0..p_loc {
        beta_loc[j] = beta_step[j];
    }
    for j in 0..p_g {
        beta_g[j] = beta_step[p_loc + j];
    }
    let q_delta = location_anchor_design.apply(&beta_loc);
    let g_delta = logslope_design.apply(&beta_g);
    // Trust-region cap to prevent a runaway first step on ill-conditioned
    // pilots: limit |Δη₁| per row to 4·σ_η (σ_η ≈ 1 under probit), measured
    // by the rigid pilot's η₁ standard deviation. This keeps the pilot in
    // the regime where the second-order Taylor approximation is meaningful;
    // the cross-block W metric only needs a per-row varying η₁, not the
    // converged β.
    let mut step_cap: f64 = 4.0;
    {
        let mean: f64 = eta1.iter().sum::<f64>() / (n as f64).max(1.0);
        let mut var: f64 = 0.0;
        for i in 0..n {
            let d = eta1[i] - mean;
            var += d * d;
        }
        let sd = (var / (n as f64).max(1.0)).sqrt();
        if sd.is_finite() && sd > 0.0 {
            step_cap = (4.0_f64).max(4.0 * sd);
        }
    }
    let mut pilot_eta = Array1::<f64>::zeros(n);
    for i in 0..n {
        let q_new = q_exit[i] + q_delta[i];
        let g_new = slope[i] + g_delta[i];
        let proposed = rigid_observed_eta(q_new, g_new, z_primary[i], probit_scale);
        let delta = proposed - eta1[i];
        let capped = if delta.abs() > step_cap {
            eta1[i] + step_cap.copysign(delta)
        } else {
            proposed
        };
        pilot_eta[i] = if capped.is_finite() { capped } else { eta1[i] };
    }
    // Logslope-surface warm start (#808): the `G`-block portion of the joint
    // Newton step, used to seed the logslope block's `initial_beta` off the
    // `g = 0` seed where the block is W-null. Sanitise to finite values; the
    // per-row logslope value `baseline_slope + logslope_offset + G·β_g` is the
    // operating point the inner refines from, so a non-finite coefficient
    // (degenerate one-step solve) falls back to the zero warm start rather than
    // poisoning the seed.
    let beta_logslope = if beta_g.iter().all(|v| v.is_finite()) {
        beta_g
    } else {
        Array1::<f64>::zeros(p_g)
    };
    Ok((pilot_eta, beta_logslope))
}

pub fn survival_marginal_slope_vector_scale(
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<f64, String> {
    marginal_slope_preserving_scale(slopes, covariance, probit_scale)
}

pub fn survival_marginal_slope_vector_eta(
    q: f64,
    z: &[f64],
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<f64, String> {
    if z.len() != covariance.dim() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector eta: score/covariance dimension mismatch: z={}, covariance={}",
                z.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    marginal_slope_probit_eta(q, z, slopes, covariance, probit_scale)
        .map_err(|err| format!("survival marginal-slope vector eta: {err}"))
}

/// Splice `at row {row}` between the reason prefix and the `:` details
/// separator so that errors from the scalar `row_primary_closed_form*`
/// kernels (which are intentionally row-agnostic) match the canonical
/// `<reason> at row N: <details>` shape that downstream consumers match on.
pub(crate) fn with_row_context(err: String, row: usize) -> String {
    if let Some(colon) = err.find(':') {
        let (head, tail) = err.split_at(colon);
        format!("{head} at row {row}{tail}")
    } else {
        format!("{err} at row {row}")
    }
}

pub fn survival_marginal_slope_vector_neglog(
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    covariance: &MarginalSlopeCovariance,
    weight: f64,
    event: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<f64, String> {
    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }
    let c = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale)?;
    let eta0 = survival_marginal_slope_vector_eta(q0, z, slopes, covariance, probit_scale)?;
    let eta1 = survival_marginal_slope_vector_eta(q1, z, slopes, covariance, probit_scale)?;
    let ad1 = qd1 * c;
    if !(ad1.is_finite() && ad1 > 0.0) {
        return Err(SurvivalMarginalSlopeError::NumericalFailure {
            reason: format!(
                "survival marginal-slope transformed derivative must be positive, got {ad1}"
            ),
        }
        .into());
    }

    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    // Same survival row density as the scalar closed-form path, with only
    // eta and d eta/dt changed by the vector marginal-preserving map:
    //
    //   ell = w[(1-d)(-log Phi(-eta1)) + log Phi(-eta0)
    //           - d log phi(eta1) - d log(qd1 c)].
    //
    // There is no extra baseline -log phi(q1) or -log qd1 factor; adding
    // either would make K=1 diverge from `row_primary_closed_form`.
    Ok(weight
        * ((1.0 - event) * (-logcdf_neg_eta1) + logcdf_neg_eta0
            - event * log_phi_eta1
            - event * ad1.ln()))
}

pub(crate) fn marginal_slope_covariance_matvec(
    covariance: &MarginalSlopeCovariance,
    vector: &[f64],
) -> Result<Vec<f64>, String> {
    covariance.validate("survival marginal-slope covariance matvec")?;
    if vector.len() != covariance.dim() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope covariance matvec dimension mismatch: vector={}, covariance={}",
                vector.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    Ok(match covariance {
        MarginalSlopeCovariance::Diagonal(diag) => vector
            .iter()
            .zip(diag.iter())
            .map(|(&v, &sigma)| sigma * v)
            .collect(),
        MarginalSlopeCovariance::Full(cov) => {
            let mut out = vec![0.0; cov.nrows()];
            for i in 0..cov.nrows() {
                for j in 0..cov.ncols() {
                    out[i] += cov[[i, j]] * vector[j];
                }
            }
            out
        }
        MarginalSlopeCovariance::LowRank(factor) => {
            let mut projected = vec![0.0; factor.ncols()];
            for r in 0..factor.ncols() {
                for k in 0..factor.nrows() {
                    projected[r] += factor[[k, r]] * vector[k];
                }
            }
            let mut out = vec![0.0; factor.nrows()];
            for k in 0..factor.nrows() {
                for r in 0..factor.ncols() {
                    out[k] += factor[[k, r]] * projected[r];
                }
            }
            out
        }
    })
}

pub(crate) fn row_primary_closed_form_vector(
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    covariance: &MarginalSlopeCovariance,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let k = slopes.len();
    if z.len() != k || covariance.dim() != k {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector row dimension mismatch: slopes={}, z={}, covariance={}",
                k,
                z.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    let c = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale)?;
    let sigma_g = marginal_slope_covariance_matvec(covariance, slopes)?;
    let s2 = probit_scale * probit_scale;
    let mut c1 = vec![0.0; k];
    for a in 0..k {
        c1[a] = s2 * sigma_g[a] / c;
    }
    let mut c2 = Array2::<f64>::zeros((k, k));
    for a in 0..k {
        for b in 0..k {
            let sigma_ab = match covariance {
                MarginalSlopeCovariance::Diagonal(diag) => {
                    if a == b {
                        diag[a]
                    } else {
                        0.0
                    }
                }
                MarginalSlopeCovariance::Full(cov) => cov[[a, b]],
                MarginalSlopeCovariance::LowRank(factor) => {
                    let mut value = 0.0;
                    for r in 0..factor.ncols() {
                        value += factor[[a, r]] * factor[[b, r]];
                    }
                    value
                }
            };
            c2[[a, b]] = s2 * sigma_ab / c - (s2 * sigma_g[a]) * (s2 * sigma_g[b]) / (c * c * c);
        }
    }

    let linear = probit_scale
        * slopes
            .iter()
            .zip(z.iter())
            .map(|(&g, &zi)| g * zi)
            .sum::<f64>();
    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let ad1 = qd1 * c;
    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }
    if !(ad1.is_finite() && ad1 > 0.0) {
        return Err(SurvivalMarginalSlopeError::NumericalFailure {
            reason: format!(
                "survival marginal-slope transformed derivative must be positive, got {ad1}"
            ),
        }
        .into());
    }

    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * ad1.ln());

    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;
    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    let dim = 3 + k;
    let mut grad = Array1::<f64>::zeros(dim);
    let mut hess = Array2::<f64>::zeros((dim, dim));
    grad[0] = u1_eta0 * c;
    grad[1] = u1_eta1 * c;
    grad[2] = u1_ad1 * c;
    hess[[0, 0]] = u2_eta0 * c * c;
    hess[[1, 1]] = u2_eta1 * c * c;
    hess[[2, 2]] = u2_ad1 * c * c;
    for a in 0..k {
        let idx = 3 + a;
        let dlin = probit_scale * z[a];
        let deta0 = q0 * c1[a] + dlin;
        let deta1 = q1 * c1[a] + dlin;
        let dad1 = qd1 * c1[a];
        grad[idx] = u1_eta0 * deta0 + u1_eta1 * deta1 + u1_ad1 * dad1;
        hess[[0, idx]] = u2_eta0 * c * deta0 + u1_eta0 * c1[a];
        hess[[idx, 0]] = hess[[0, idx]];
        hess[[1, idx]] = u2_eta1 * c * deta1 + u1_eta1 * c1[a];
        hess[[idx, 1]] = hess[[1, idx]];
        hess[[2, idx]] = u2_ad1 * c * dad1 + u1_ad1 * c1[a];
        hess[[idx, 2]] = hess[[2, idx]];
        for b in 0..k {
            let jdx = 3 + b;
            let dlin_b = probit_scale * z[b];
            let deta0_b = q0 * c1[b] + dlin_b;
            let deta1_b = q1 * c1[b] + dlin_b;
            let dad1_b = qd1 * c1[b];
            hess[[idx, jdx]] = u2_eta0 * deta0 * deta0_b
                + u1_eta0 * q0 * c2[[a, b]]
                + u2_eta1 * deta1 * deta1_b
                + u1_eta1 * q1 * c2[[a, b]]
                + u2_ad1 * dad1 * dad1_b
                + u1_ad1 * qd1 * c2[[a, b]];
        }
    }
    Ok((nll, grad, hess))
}

pub(crate) fn standardize_latent_z_matrix_with_policy(
    z: &Array2<f64>,
    weights: &Array1<f64>,
    context: &str,
    policy: &LatentZPolicy,
) -> Result<(Array2<f64>, LatentZNormalization), String> {
    if z.ncols() == 0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!("{context} requires at least one z column"),
        }
        .into());
    }
    let mut out = Array2::<f64>::zeros(z.raw_dim());
    let mut first_norm = LatentZNormalization { mean: 0.0, sd: 1.0 };
    for col in 0..z.ncols() {
        let input = z.column(col).to_owned();
        let (standardized, normalization) =
            standardize_latent_z_with_policy(&input, weights, context, policy)?;
        if col == 0 {
            first_norm = normalization;
        }
        out.column_mut(col).assign(&standardized);
    }
    Ok((out, first_norm))
}

/// Derivatives of c(g) = √(1 + (s_f g)^2) up to 4th order in the raw slope g.
#[inline]
pub(crate) fn c_derivatives(g: f64, probit_scale: f64) -> (f64, f64, f64, f64, f64) {
    let observed_g = rigid_observed_logslope(g, probit_scale);
    let g2 = observed_g * observed_g;
    let s2 = probit_scale * probit_scale;
    let s4 = s2 * s2;
    let c = (1.0 + g2).sqrt();
    let c2 = c * c;
    let c3 = c2 * c;
    let c5 = c3 * c2;
    let c7 = c5 * c2;
    let c1 = s2 * g / c;
    let c2d = s2 / c3;
    let c3d = -3.0 * s4 * g / c5;
    let c4d = s4 * (12.0 * g2 - 3.0) / c7;
    (c, c1, c2d, c3d, c4d)
}

/// Derivatives of neglog(x) = -log(x): [-1/x, 1/x², -2/x³, 6/x⁴].
#[inline]
pub(crate) fn neglog_derivatives(x: f64) -> (f64, f64, f64, f64) {
    let x1 = x.max(1e-300);
    let inv = 1.0 / x1;
    let inv2 = inv * inv;
    (-inv, inv2, -2.0 * inv2 * inv, 6.0 * inv2 * inv2)
}

/// Row-level primary gradient (4-vector) and Hessian (4×4 symmetric)
/// computed entirely from closed-form scalar formulas.
///
/// Returns (nll, gradient[4], hessian[4][4]) on the stack.
#[inline]
pub(crate) fn row_primary_closed_form(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    let (c, c1, c2, ..) = c_derivatives(g, probit_scale);
    let observed_g = rigid_observed_logslope(g, probit_scale);

    // Linear predictors
    let eta0 = q0 * c + observed_g * z;
    let eta1 = q1 * c + observed_g * z;
    let ad1 = qd1 * c;

    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }

    // ── NLL terms ──
    // Entry survival: -neglogΦ(-η₀) = logΦ(-η₀)
    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    // Exit survival: (1-d)·neglogΦ(-η₁)
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    // Event density: d·logφ(η₁)
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    // Time derivative: d·log(ad1)
    let log_ad1 = ad1.max(1e-300).ln();

    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * log_ad1);

    // ── First and second derivatives of each NLL component ──
    // signed_probit_neglog_derivatives gives derivatives with respect to m for
    // -weight * logΦ(m). Here m = -η, so odd derivatives flip sign when mapped
    // back to derivatives with respect to η.
    // For entry: m = -η₀, weight = -w because the NLL contains +w logΦ(-η₀)
    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
    // For exit: m = -η₁, weight = w(1-d)
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
    // Event density: -d·logφ(η₁) = d·(η₁²/2 + const).
    // d/dη₁ = d·w·η₁, d²/dη₁² = d·w.
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    // Time derivative: -d·log(ad1).
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    // ── Chain rule to primary space ──
    // η₀ depends on (q₀, g): ∂η₀/∂q₀ = c, ∂η₀/∂g = q₀c₁ + s_f z
    // η₁ depends on (q₁, g): ∂η₁/∂q₁ = c, ∂η₁/∂g = q₁c₁ + s_f z
    // ad1 depends on (qd1, g): ∂ad1/∂qd1 = c, ∂ad1/∂g = qd1·c₁
    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + probit_scale * z;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + probit_scale * z;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    // Combined first derivatives of total NLL:
    // u1 for η₀ terms = -e0_k1 (chain rule through m = -η₀)
    // u1 for η₁ terms = -e1_k1 + phi_u1 (chain rule through m = -η₁)
    // u1 for ad1 term = td_u1 (time derivative)
    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;

    let mut grad = [0.0_f64; N_PRIMARY];
    grad[0] = u1_eta0 * deta0_dq0; // ∂ℓ/∂q₀
    grad[1] = u1_eta1 * deta1_dq1; // ∂ℓ/∂q₁
    grad[2] = u1_ad1 * dad1_dqd1; // ∂ℓ/∂qd₁
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg; // ∂ℓ/∂g

    // Combined second derivatives:
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    // Second mixed derivatives of η w.r.t. primary scalars:
    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    // d²η₀/dg² = q₀·c₂ (z is linear in g, so its second derivative is 0)
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; N_PRIMARY]; N_PRIMARY];

    // (q0, q0)
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    // (q1, q1)
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    // (qd1, qd1)
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    // (q0, q1) = 0 (η₀ and η₁ share no primary scalars except g)
    hess[0][1] = 0.0;
    hess[1][0] = 0.0;
    // (q0, qd1) = 0
    hess[0][2] = 0.0;
    hess[2][0] = 0.0;
    // (q1, qd1) = 0
    hess[1][2] = 0.0;
    hess[2][1] = 0.0;
    // (q0, g) = u2_η₀ · (∂η₀/∂q₀)(∂η₀/∂g) + u1_η₀ · (∂²η₀/∂q₀∂g)
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    // (q1, g)
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    // (qd1, g)
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    // (g, g) = Σ_terms [u2·(dterm/dg)² + u1·(d²term/dg²)]
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg
        + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg
        + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg
        + u1_ad1 * d2ad1_dg2;

    Ok((nll, grad, hess))
}

/// Crate-visible wrapper around `row_primary_closed_form` so the
/// identifiability-compiler sibling module
/// (`survival_marginal_slope_identifiability`) can build its 4×4
/// `SurvivalRowHessian` without exposing the closed-form kernel publicly.
pub(crate) fn row_primary_for_compiler(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    row_primary_closed_form(q0, q1, qd1, g, z, w, d, derivative_guard, probit_scale)
}

/// Shared-slope multi-z reduction for the rigid 4-primary row calculus.
///
/// When K observed scores share one raw log-slope value `g`, the probit
/// gradient vector is `r(g) = s_f g 1_K` and the row index is
///
///     eta_j = q_j sqrt(1 + r(g)' Sigma r(g)) + r(g)' z_i
///           = q_j sqrt(1 + g^2 s_f^2 1' Sigma 1) + g s_f (1'z_i).
///
/// The exact row kernel can therefore remain four-dimensional
/// `(q0, q1, qd1, g)` if, and only if, the log-slope surface is shared across
/// z coordinates.  The derivatives below are the scalar chain rule with
/// `c(g) = sqrt(1 + g^2 s_f^2 1' Sigma 1)` and
/// `d(r'z_i)/dg = s_f 1'z_i`.  K=1 with `1' Sigma 1 = 1` is exactly the
/// existing scalar path handled by `row_primary_closed_form`.
#[inline]
pub(crate) fn row_primary_closed_form_shared_score(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z_sum: f64,
    covariance_ones: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    if !(covariance_ones.is_finite() && covariance_ones >= 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival marginal-slope shared-score covariance scale must be finite and non-negative, got {covariance_ones}"
            ),
        }
        .into());
    }
    let effective_scale = probit_scale * covariance_ones.sqrt();
    let (c, c1, c2, ..) = c_derivatives(g, effective_scale);
    let linear = rigid_observed_logslope(g, probit_scale) * z_sum;
    let linear_dg = probit_scale * z_sum;

    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let ad1 = qd1 * c;

    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }

    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let log_ad1 = ad1.max(1e-300).ln();

    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * log_ad1);

    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + linear_dg;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + linear_dg;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;

    let mut grad = [0.0_f64; N_PRIMARY];
    grad[0] = u1_eta0 * deta0_dq0;
    grad[1] = u1_eta1 * deta1_dq1;
    grad[2] = u1_ad1 * dad1_dqd1;
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg;

    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; N_PRIMARY]; N_PRIMARY];
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg
        + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg
        + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg
        + u1_ad1 * d2ad1_dg2;

    Ok((nll, grad, hess))
}

// ── Eval cache ────────────────────────────────────────────────────────
//
// Third and fourth order contracted derivatives for the outer REML path
// continue to use the MultiDirJet engine via row_neglog_directional.
// That path is called O(n_rho²) times, not O(n × inner_iters) times,
// so the jet overhead is acceptable there.

#[derive(Clone)]
pub(crate) struct RowPrimaryBase {
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Array2<f64>,
}

pub(crate) struct EvalCache {
    pub(crate) row_bases: Vec<RowPrimaryBase>,
}
