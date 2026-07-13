//! Per-row marginal-slope mathematics: the canonical rigid row-program
//! lowering, vector score geometry, analytic primitive helpers, pilot-eta
//! solvers, latent-z standardization, and the evaluation cache.

use super::*;

use gam_math::jet_scalar::{
    DynamicJetArena, DynamicOrder2, FixedRuntimeJet, JetScalar, Order2, RuntimeJetScalar,
    RuntimeValue,
};
use gam_math::order2_graph::{Order2Graph, Order2GraphWorkspace};

// ── Closed-form row kernel ─────────────────────────────────────────────
//
// The survival marginal-slope NLL for row i is:
//
//   ℓ_i = w_i [ (1-d)·neglogΦ(-η₁) + logΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
//
// with η₀ = q₀c + s_f g z, η₁ = q₁c + s_f g z, a'₁ = qd₁·c,
// c = √(1 + (s_f g)²), s_f = 1/√(1+σ²).
//
// The K=1 value/gradient/Hessian is lowered symbolically from the same
// `row_program!` SSA graph as `rigid_row_nll` and CUDA. It remains stack-only
// without constructing a forward jet or duplicating the mathematical program.

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
    //   η₁ = q·c(g) + s(g)·z   with c(g) = √(1 + s(g)²), s(g) = observed_logslope(g)
    //   ∂η₁/∂q = c(g)
    //   ∂η₁/∂g = q·c'(g) + s'(g)·z
    //
    // The chain factors are the EXACT closed forms from `c_derivatives`
    // (which returns `(c, c', c'', …)`) and `rigid_observed_logslope`
    // (`s(g) = probit_scale·g`, so `s'(g) = probit_scale`). They weight the
    // W metric for the pilot IRLS step and are not propagated into a final
    // coefficient, but the analytic chain is bit-stable and avoids the
    // step-size/round-off error of a finite difference.
    let mut chain_q = Array1::<f64>::zeros(n);
    let mut chain_g = Array1::<f64>::zeros(n);
    let mut grad_eta1 = Array1::<f64>::zeros(n);
    let mut hess_eta1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let g_i = slope[i];
        let z_i = z_primary[i];
        // ∂η₁/∂q = c(g); ∂η₁/∂g = q·c'(g) + s'(g)·z with s'(g) = probit_scale.
        let (c, c1, ..) = c_derivatives(g_i, probit_scale);
        chain_q[i] = c;
        chain_g[i] = q_exit[i] * c1 + probit_scale * z_i;
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
    let dimension = 3 + slopes.len();
    let mut vars = Vec::with_capacity(dimension);
    for (axis, value) in [q0, q1, qd1]
        .into_iter()
        .chain(slopes.iter().copied())
        .enumerate()
    {
        vars.push(RuntimeValue::variable(value, axis, dimension, &()));
    }
    let inputs = RigidRowInputs {
        row: 0,
        wi: weight,
        di: event,
        z_sum: 0.0,
        covariance_ones: 0.0,
        probit_scale,
        qd1_lower: derivative_guard,
    };
    Ok(rigid_vector_row_nll(&vars, z, covariance, &inputs, &())?.value())
}

#[cfg(test)]
mod vector_hand_oracle_tests {
    use super::*;

    /// Derivatives of neglog(x) = -log(x): [-1/x, 1/x², -2/x³, 6/x⁴].
    #[inline]
    pub(super) fn neglog_derivatives(x: f64) -> (f64, f64, f64, f64) {
        let x1 = x.max(1e-300);
        let inv = 1.0 / x1;
        let inv2 = inv * inv;
        (-inv, inv2, -2.0 * inv2 * inv, 6.0 * inv2 * inv2)
    }

    fn marginal_slope_covariance_matvec(
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

    pub(super) fn row_primary_closed_form_vector_hand_reference(
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
                c2[[a, b]] =
                    s2 * sigma_ab / c - (s2 * sigma_g[a]) * (s2 * sigma_g[b]) / (c * c * c);
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
        let nll = w
            * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * ad1.ln());

        let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
        let (e1_k1, e1_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
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
}

/// Runtime-width rigid row program for independent score slopes.
///
/// The first three primaries are `(q0, q1, qd1)` and the remaining primaries
/// are the score-specific slopes. The covariance representation changes only
/// how the exact quadratic form `r' Sigma r` is lowered; every value, gradient,
/// and Hessian channel then flows through the same stable probit/log-density
/// leaf stacks as [`rigid_row_nll`]. There is no separately maintained chain
/// rule for score cross-blocks.
fn rigid_vector_row_nll<'arena, S>(
    vars: &[S],
    z: &[f64],
    covariance: &MarginalSlopeCovariance,
    inputs: &RigidRowInputs,
    workspace: &'arena S::Workspace,
) -> Result<S, String>
where
    S: RuntimeJetScalar<'arena>,
{
    let dimension = vars.len();
    let k = dimension.checked_sub(3).ok_or_else(|| {
        SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector row needs three index primaries, got {dimension}"
            ),
        }
        .to_string()
    })?;
    if z.len() != k || covariance.dim() != k {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector row dimension mismatch: slopes={k}, z={}, covariance={}",
                z.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    covariance.validate("survival marginal-slope vector row program")?;
    if !inputs.probit_scale.is_finite() {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival marginal-slope probit scale must be finite, got {}",
                inputs.probit_scale
            ),
        }
        .into());
    }
    if z.iter().any(|value| !value.is_finite())
        || vars[3..].iter().any(|slope| !slope.value().is_finite())
    {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival marginal-slope vector scores and slopes must be finite".to_string(),
        }
        .into());
    }

    let linear =
        S::linear_combination(&vars[3..], z, dimension, workspace).scale(inputs.probit_scale);

    // One semantic primitive owns the mechanically derived V/G/H channels for
    // `g' Sigma g`; representation-specific multiplication stays matrix free.
    let mut variance = S::symmetric_quadratic_form(&vars[3..], covariance, dimension, workspace);
    let probit_variance_scale = inputs.probit_scale * inputs.probit_scale;
    let variance_value = variance.value() * probit_variance_scale;
    if !(variance_value.is_finite()
        && variance_value >= crate::bms::gradient_paths::COVARIANCE_QUADRATIC_FORM_PSD_TOL)
    {
        return Err(SurvivalMarginalSlopeError::NumericalFailure {
            reason: format!(
                "survival marginal-slope covariance quadratic form must be non-negative, got {variance_value}"
            ),
        }
        .into());
    }
    // At a mathematical zero, floating accumulation may land a few ulps below
    // zero. Shift only the primal constant; derivative channels are unchanged.
    if variance_value < 0.0 {
        variance = variance.add_constant(-variance.value(), workspace);
    }

    let correction_argument = 1.0 + probit_variance_scale * variance.value();
    let correction = variance.affine_compose(
        probit_variance_scale,
        1.0,
        unary_derivatives_sqrt(correction_argument),
        workspace,
    );
    let correction_value = correction.value();
    let linear_value = linear.value();
    let eta0_value = vars[0].value() * correction_value + linear_value;
    let eta1_value = vars[1].value() * correction_value + linear_value;
    let adjusted_derivative_value = vars[2].value() * correction_value;
    let neg_eta0 = -eta0_value;
    let neg_eta1 = -eta1_value;

    validate_rigid_row_admission(
        vars[2].value(),
        inputs,
        neg_eta0,
        neg_eta1,
        adjusted_derivative_value,
    )?;

    let mut entry_stack = unary_derivatives_neglog_phi(neg_eta0, inputs.wi);
    for derivative in &mut entry_stack {
        *derivative = -*derivative;
    }
    let exit_stack = unary_derivatives_neglog_phi(neg_eta1, inputs.wi * (1.0 - inputs.di));
    if inputs.di > 0.0 {
        let scale = (-inputs.wi) * inputs.di;
        let mut event_stack = unary_derivatives_log_normal_pdf(eta1_value);
        let mut time_stack = unary_derivatives_log(adjusted_derivative_value);
        for derivative in event_stack.iter_mut().chain(&mut time_stack) {
            *derivative *= scale;
        }
        let zero = S::constant(0.0, dimension, workspace);
        Ok(S::multiply_add_affine_composed_sum(
            &[
                vars[1].clone(),
                vars[0].clone(),
                vars[1].clone(),
                vars[2].clone(),
            ],
            &[
                correction.clone(),
                correction.clone(),
                correction.clone(),
                correction,
            ],
            &[linear.clone(), linear.clone(), linear, zero],
            &[-1.0, -1.0, 1.0, 1.0],
            &[exit_stack, entry_stack, event_stack, time_stack],
            dimension,
            workspace,
        ))
    } else {
        Ok(S::multiply_add_affine_composed_sum(
            &[vars[1].clone(), vars[0].clone()],
            &[correction.clone(), correction],
            &[linear.clone(), linear],
            &[-1.0, -1.0],
            &[exit_stack, entry_stack],
            dimension,
            workspace,
        ))
    }
}

fn row_primary_closed_form_vector_dynamic(
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
    arena: &mut DynamicJetArena,
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
    let dim = 3 + k;
    arena.reset();
    let primary_values = arena.alloc_slice_fill_with(dim, |axis| match axis {
        0 => q0,
        1 => q1,
        2 => qd1,
        _ => slopes[axis - 3],
    });
    let vars = arena.alloc_slice_fill_with(dim, |axis| {
        DynamicOrder2::variable(primary_values[axis], axis, dim, arena)
    });
    let inputs = RigidRowInputs {
        row: 0,
        wi: w,
        di: d,
        z_sum: 0.0,
        covariance_ones: 0.0,
        probit_scale,
        qd1_lower: derivative_guard,
    };
    let row = rigid_vector_row_nll(vars, z, covariance, &inputs, arena)?;
    let gradient = Array1::from_vec(row.g().to_vec());
    let hessian = Array2::from_shape_fn((dim, dim), |(a, b)| row.h_at(a, b));
    Ok((row.v, gradient, hessian))
}

fn row_primary_closed_form_vector_fixed<const DIM: usize>(
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
    if DIM != 3 + slopes.len() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "fixed runtime-vector row dimension mismatch: DIM={DIM}, slopes={}",
                slopes.len()
            ),
        }
        .into());
    }
    let values: [f64; DIM] = std::array::from_fn(|axis| match axis {
        0 => q0,
        1 => q1,
        2 => qd1,
        _ => slopes[axis - 3],
    });
    let vars: [FixedRuntimeJet<Order2<DIM>, DIM>; DIM] = std::array::from_fn(|axis| {
        FixedRuntimeJet::from_inner(Order2::variable(values[axis], axis))
    });
    let inputs = RigidRowInputs {
        row: 0,
        wi: w,
        di: d,
        z_sum: 0.0,
        covariance_ones: 0.0,
        probit_scale,
        qd1_lower: derivative_guard,
    };
    let row = rigid_vector_row_nll(&vars, z, covariance, &inputs, &())?.into_inner();
    let gradient = Array1::from_vec(row.g().to_vec());
    let hessian = Array2::from_shape_fn((DIM, DIM), |(a, b)| row.h()[a][b]);
    Ok((row.0.v, gradient, hessian))
}

enum RigidVectorRowBackend {
    Fixed,
    Graph(Order2GraphWorkspace),
    Dynamic(DynamicJetArena),
}

/// Width-bound storage for exactly one production row backend. Fixed widths
/// carry no workspace, graph widths own one boxed tape, and dynamic widths own
/// one arena; a fold can never silently reuse storage configured for another
/// score dimension.
pub(crate) struct RigidVectorRowWorkspace {
    score_dimension: usize,
    backend: RigidVectorRowBackend,
}

impl RigidVectorRowWorkspace {
    pub(crate) fn new(score_dimension: usize) -> Result<Self, String> {
        let backend = match score_dimension {
            0 => {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: "survival marginal-slope vector row requires at least one score slope"
                        .to_string(),
                }
                .into());
            }
            1..=6 => RigidVectorRowBackend::Fixed,
            7..=13 => RigidVectorRowBackend::Graph(Order2GraphWorkspace::new()),
            14.. => RigidVectorRowBackend::Dynamic(DynamicJetArena::new()),
        };
        Ok(Self {
            score_dimension,
            backend,
        })
    }
}

fn row_primary_closed_form_vector_graph<const DIM: usize>(
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
    workspace: &mut Order2GraphWorkspace,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    if DIM != 3 + slopes.len() || z.len() != slopes.len() || covariance.dim() != slopes.len() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "compiled vector row dimension mismatch: DIM={DIM}, slopes={}, z={}, covariance={}",
                slopes.len(),
                z.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    workspace.reset(DIM);
    let primary_values: [f64; DIM] = std::array::from_fn(|axis| match axis {
        0 => q0,
        1 => q1,
        2 => qd1,
        _ => slopes[axis - 3],
    });
    let vars: [Order2Graph<'_, DIM>; DIM] = std::array::from_fn(|axis| {
        Order2Graph::variable(primary_values[axis], axis, DIM, workspace)
    });
    let inputs = RigidRowInputs {
        row: 0,
        wi: w,
        di: d,
        z_sum: 0.0,
        covariance_ones: 0.0,
        probit_scale,
        qd1_lower: derivative_guard,
    };
    let row = rigid_vector_row_nll(&vars, z, covariance, &inputs, workspace)?.into_order2();
    let gradient = Array1::from_vec(row.g().to_vec());
    let hessian = Array2::from_shape_fn((DIM, DIM), |(a, b)| row.h()[a][b]);
    Ok((row.0.v, gradient, hessian))
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
    workspace: &mut RigidVectorRowWorkspace,
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
    if workspace.score_dimension != k {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector workspace width mismatch: configured={}, row={k}",
                workspace.score_dimension
            ),
        }
        .into());
    }

    macro_rules! graph_row {
        ($dimension:literal, $graph:expr) => {
            row_primary_closed_form_vector_graph::<$dimension>(
                q0,
                q1,
                qd1,
                slopes,
                z,
                covariance,
                w,
                d,
                derivative_guard,
                probit_scale,
                $graph,
            )
        };
    }

    macro_rules! fixed_row {
        ($dimension:literal) => {
            row_primary_closed_form_vector_fixed::<$dimension>(
                q0,
                q1,
                qd1,
                slopes,
                z,
                covariance,
                w,
                d,
                derivative_guard,
                probit_scale,
            )
        };
    }

    match (&mut workspace.backend, k) {
        (RigidVectorRowBackend::Fixed, 1) => fixed_row!(4),
        (RigidVectorRowBackend::Fixed, 2) => fixed_row!(5),
        (RigidVectorRowBackend::Fixed, 3) => fixed_row!(6),
        (RigidVectorRowBackend::Fixed, 4) => fixed_row!(7),
        (RigidVectorRowBackend::Fixed, 5) => fixed_row!(8),
        (RigidVectorRowBackend::Fixed, 6) => fixed_row!(9),
        (RigidVectorRowBackend::Graph(graph), 7) => graph_row!(10, graph),
        (RigidVectorRowBackend::Graph(graph), 8) => graph_row!(11, graph),
        (RigidVectorRowBackend::Graph(graph), 9) => graph_row!(12, graph),
        (RigidVectorRowBackend::Graph(graph), 10) => graph_row!(13, graph),
        (RigidVectorRowBackend::Graph(graph), 11) => graph_row!(14, graph),
        (RigidVectorRowBackend::Graph(graph), 12) => graph_row!(15, graph),
        (RigidVectorRowBackend::Graph(graph), 13) => graph_row!(16, graph),
        (RigidVectorRowBackend::Dynamic(arena), 14..) => row_primary_closed_form_vector_dynamic(
            q0,
            q1,
            qd1,
            slopes,
            z,
            covariance,
            w,
            d,
            derivative_guard,
            probit_scale,
            arena,
        ),
        _ => Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector workspace backend does not support score width {k}"
            ),
        }
        .into()),
    }
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

/// Row-level primary value, gradient, and Hessian lowered from the canonical
/// [`rigid_row_nll`] program.
///
/// Every live K=1 caller (pilot initialization, sigma evaluation,
/// identifiability compilation, KKT refusal, and the `RowKernel`) therefore
/// executes the same direct scalar schedule, including its leaf curvature and
/// cross terms.
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
    let inputs = RigidRowInputs {
        row: 0,
        wi: w,
        di: d,
        z_sum: z,
        covariance_ones: 1.0,
        probit_scale,
        qd1_lower: derivative_guard,
    };
    rigid_row_order2(&[q0, q1, qd1, g], &inputs)
}

#[cfg(test)]
mod test_support {
    use super::*;

    /// Historical hand-expanded K=1 V/G/H schedule, retained only as an
    /// independent strongest-hand test and performance witness for the generated
    /// [`row_primary_closed_form`] lowering. Production must never call this body.
    #[inline]
    pub(super) fn row_primary_closed_form_hand_reference(
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
        let (e1_k1, e1_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
        // Event density: -d·logφ(η₁) = d·(η₁²/2 + const).
        // d/dη₁ = d·w·η₁, d²/dη₁² = d·w.
        let phi_u1 = w * d * eta1;
        let phi_u2 = w * d;
        // Time derivative: -d·log(ad1).
        let (nl_u1, nl_u2, _, _) = super::vector_hand_oracle_tests::neglog_derivatives(ad1);
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

// ── Eval cache ────────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct RowPrimaryBase {
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Array2<f64>,
}

pub(crate) struct EvalCache {
    pub(crate) row_bases: Vec<RowPrimaryBase>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::SymmetricQuadraticCoefficients;

    impl RigidVectorRowWorkspace {
        fn backend_name(&self) -> &'static str {
            match &self.backend {
                RigidVectorRowBackend::Fixed => "fixed",
                RigidVectorRowBackend::Graph(_) => "graph",
                RigidVectorRowBackend::Dynamic(_) => "dynamic",
            }
        }
    }

    /// #1440 cutover oracle: the pilot W-metric chain factors are now the EXACT
    /// closed forms `∂η₁/∂q = c(g)` and `∂η₁/∂g = q·c'(g) + s'(g)·z`
    /// (`s'(g) = probit_scale`). They used to be a central finite difference of
    /// `rigid_observed_eta`; this oracle pins that the analytic substitution agrees
    /// with that retired FD to ~1e-7 across a grid of (q, g, z, probit_scale), so
    /// the production FD removal is behaviour-preserving.
    #[test]
    fn pilot_chain_factors_match_retired_central_difference() {
        // The retired finite-difference step (was `h_fd = 1e-7`).
        let h: f64 = 1.0e-7;
        let grid_q = [-2.5_f64, -0.4, 0.0, 0.7, 3.1];
        let grid_g = [-1.8_f64, -0.3, 0.0, 0.9, 2.4];
        let grid_z = [-1.2_f64, 0.0, 0.5, 2.0];
        let grid_scale = [0.25_f64, 0.6, 1.0, 1.7];

        let mut max_err_q = 0.0_f64;
        let mut max_err_g = 0.0_f64;
        for &probit_scale in &grid_scale {
            for &q in &grid_q {
                for &g in &grid_g {
                    for &z in &grid_z {
                        // Analytic chain (production path).
                        let (c, c1, ..) = c_derivatives(g, probit_scale);
                        let chain_q = c;
                        let chain_g = q * c1 + probit_scale * z;

                        // Retired central difference of `rigid_observed_eta`.
                        let fd_q = (rigid_observed_eta(q + h, g, z, probit_scale)
                            - rigid_observed_eta(q - h, g, z, probit_scale))
                            / (2.0 * h);
                        let fd_g = (rigid_observed_eta(q, g + h, z, probit_scale)
                            - rigid_observed_eta(q, g - h, z, probit_scale))
                            / (2.0 * h);

                        // Relative comparison (the FD truncation/round-off floor at
                        // h = 1e-7 is ~1e-7 of the magnitude of the derivative).
                        let scale_q = chain_q.abs().max(1.0);
                        let scale_g = chain_g.abs().max(1.0);
                        max_err_q = max_err_q.max((chain_q - fd_q).abs() / scale_q);
                        max_err_g = max_err_g.max((chain_g - fd_g).abs() / scale_g);
                    }
                }
            }
        }
        assert!(
            max_err_q < 1.0e-6,
            "∂η₁/∂q analytic vs retired FD mismatch: max rel err {max_err_q:.3e}"
        );
        assert!(
            max_err_g < 1.0e-6,
            "∂η₁/∂g analytic vs retired FD mismatch: max rel err {max_err_g:.3e}"
        );
    }

    /// Sanity: `c_derivatives` itself is the analytic derivative of
    /// `rigid_observed_scale` (the `c(g)` the chain reuses), and
    /// `rigid_observed_logslope` is linear in g with slope `probit_scale`. This
    /// guards the two ingredients the chain depends on, independent of the chain
    /// assembly above.
    #[test]
    fn c_and_logslope_derivatives_match_central_difference() {
        let h: f64 = 1.0e-7;
        for &probit_scale in &[0.3_f64, 0.8, 1.5] {
            for &g in &[-1.4_f64, -0.2, 0.0, 0.6, 2.0] {
                let (c, c1, ..) = c_derivatives(g, probit_scale);
                assert!(
                    (c - rigid_observed_scale(g, probit_scale)).abs() < 1e-12,
                    "c(g) must equal rigid_observed_scale"
                );
                let fd_c1 = (rigid_observed_scale(g + h, probit_scale)
                    - rigid_observed_scale(g - h, probit_scale))
                    / (2.0 * h);
                assert!(
                    (c1 - fd_c1).abs() / c1.abs().max(1.0) < 1e-6,
                    "c'(g) analytic vs FD mismatch at g={g}, scale={probit_scale}: {c1} vs {fd_c1}"
                );
                let fd_s = (rigid_observed_logslope(g + h, probit_scale)
                    - rigid_observed_logslope(g - h, probit_scale))
                    / (2.0 * h);
                assert!(
                    (probit_scale - fd_s).abs() < 1e-6,
                    "s'(g) must equal probit_scale"
                );
            }
        }
    }

    #[test]
    fn covariance_quadratic_primitive_matches_scalar_program_all_forms_932() {
        use gam_math::jet_scalar::{JetScalar, Order2};
        use gam_math::nested_dual::JetField;

        const DIM: usize = 5;
        let values = [0.4, -0.7, 1.1, 0.25, -0.3];
        let vars: [Order2<DIM>; DIM] =
            std::array::from_fn(|axis| Order2::variable(values[axis], axis));
        let inputs = [
            vars[0].mul(&vars[1]).add(&vars[3]),
            vars[1].exp().add(&vars[2].scale(0.4)),
            vars[2].mul(&vars[4]).sub(&vars[0]),
        ];
        let covariances = [
            MarginalSlopeCovariance::Diagonal(Array1::from_vec(vec![1.2, 0.8, 1.5])),
            MarginalSlopeCovariance::Full(
                Array2::from_shape_vec(
                    (3, 3),
                    vec![1.2, 0.3, -0.2, 0.3, 0.8, 0.15, -0.2, 0.15, 1.5],
                )
                .expect("3x3 covariance"),
            ),
            MarginalSlopeCovariance::LowRank(
                Array2::from_shape_vec((3, 2), vec![0.8, -0.1, 0.25, 0.7, -0.45, 0.35])
                    .expect("3x2 factor"),
            ),
        ];

        for covariance in &covariances {
            let direct = Order2::symmetric_quadratic_form(&inputs, covariance);
            let mut scalar = Order2::constant(0.0);
            for row in 0..inputs.len() {
                scalar = scalar.add(
                    &inputs[row]
                        .mul(&inputs[row])
                        .scale(covariance.coefficient(row, row)),
                );
                for column in row + 1..inputs.len() {
                    scalar = scalar.add(
                        &inputs[row]
                            .mul(&inputs[column])
                            .scale(2.0 * covariance.coefficient(row, column)),
                    );
                }
            }
            let tolerance = 2.0e-13;
            let close = |actual: f64, expected: f64| {
                assert!(
                    (actual - expected).abs()
                        <= tolerance * actual.abs().max(expected.abs()).max(1.0),
                    "quadratic primitive mismatch: direct={actual:+.16e}, scalar={expected:+.16e}"
                );
            };
            close(direct.value(), scalar.value());
            for primary_a in 0..DIM {
                close(direct.g()[primary_a], scalar.g()[primary_a]);
                for primary_b in 0..DIM {
                    close(
                        direct.h()[primary_a][primary_b],
                        scalar.h()[primary_a][primary_b],
                    );
                }
            }
        }
    }

    /// #932 runtime-width cutover gate. The production row program derives the
    /// complete V/G/H tower through the scheduled fixed/graph backends for
    /// `k <= 13` and the same-expression dynamic backend above it. This compares every
    /// channel with the retired strongest-hand schedule for all covariance
    /// forms. The explicit score-score assertions pin the mixed blocks that a
    /// scalar or block-diagonal check would miss.
    #[test]
    fn runtime_vector_row_program_matches_strongest_hand_mixed_score_vgh_932() {
        let diagonal = MarginalSlopeCovariance::Diagonal(Array1::from_vec(vec![1.2, 0.9, 0.7]));
        let full = MarginalSlopeCovariance::Full(
            Array2::from_shape_vec(
                (3, 3),
                vec![1.2, 0.18, -0.11, 0.18, 0.9, 0.24, -0.11, 0.24, 0.7],
            )
            .expect("3x3 full covariance"),
        );
        let low_rank = MarginalSlopeCovariance::LowRank(
            Array2::from_shape_vec((3, 2), vec![0.8, -0.1, 0.25, 0.7, -0.45, 0.35])
                .expect("3x2 covariance factor"),
        );
        let cases = [
            (&diagonal, 0.0, -0.4, 0.65, 0.95, 0.8),
            (&diagonal, 1.0, 0.15, -0.55, 1.35, 0.94),
            (&full, 0.0, -0.6, 0.45, 0.85, 0.72),
            (&full, 1.0, 0.3, -0.75, 1.25, 1.0),
            (&low_rank, 0.35, -1.1, 0.8, 0.55, 0.63),
            (&low_rank, 1.0, 0.9, -0.2, 1.6, 0.91),
        ];
        let slopes = [0.55, -0.8, 0.35];
        let scores = [-1.2, 0.65, 1.4];
        let mut workspace = RigidVectorRowWorkspace::new(3).expect("k=3 production workspace");
        let mut dynamic_arena = DynamicJetArena::new();
        let close = |label: &str, actual: f64, expected: f64| {
            let tolerance = 5.0e-11 * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= tolerance,
                "{label}: runtime={actual:+.16e}, hand={expected:+.16e}, tolerance={tolerance:.3e}",
            );
        };

        for (case, &(covariance, event, q0, q1, qd1, probit_scale)) in cases.iter().enumerate() {
            let production = row_primary_closed_form_vector(
                q0,
                q1,
                qd1,
                &slopes,
                &scores,
                covariance,
                1.3,
                event,
                1.0e-8,
                probit_scale,
                &mut workspace,
            )
            .expect("compiled vector row program");
            let dynamic = row_primary_closed_form_vector_dynamic(
                q0,
                q1,
                qd1,
                &slopes,
                &scores,
                covariance,
                1.3,
                event,
                1.0e-8,
                probit_scale,
                &mut dynamic_arena,
            )
            .expect("dynamic vector row program");
            let fixed = row_primary_closed_form_vector_fixed::<6>(
                q0,
                q1,
                qd1,
                &slopes,
                &scores,
                covariance,
                1.3,
                event,
                1.0e-8,
                probit_scale,
            )
            .expect("fixed vector row program");
            let hand = vector_hand_oracle_tests::row_primary_closed_form_vector_hand_reference(
                q0,
                q1,
                qd1,
                &slopes,
                &scores,
                covariance,
                1.3,
                event,
                1.0e-8,
                probit_scale,
            )
            .expect("strongest-hand vector row");

            close(&format!("case {case} value"), production.0, hand.0);
            close(&format!("case {case} dynamic value"), dynamic.0, hand.0);
            close(&format!("case {case} fixed value"), fixed.0, hand.0);
            for axis_a in 0..production.1.len() {
                close(
                    &format!("case {case} gradient[{axis_a}]"),
                    production.1[axis_a],
                    hand.1[axis_a],
                );
                close(
                    &format!("case {case} dynamic gradient[{axis_a}]"),
                    dynamic.1[axis_a],
                    hand.1[axis_a],
                );
                close(
                    &format!("case {case} fixed gradient[{axis_a}]"),
                    fixed.1[axis_a],
                    hand.1[axis_a],
                );
                for axis_b in 0..production.1.len() {
                    close(
                        &format!("case {case} Hessian[{axis_a},{axis_b}]"),
                        production.2[[axis_a, axis_b]],
                        hand.2[[axis_a, axis_b]],
                    );
                    close(
                        &format!("case {case} dynamic Hessian[{axis_a},{axis_b}]"),
                        dynamic.2[[axis_a, axis_b]],
                        hand.2[[axis_a, axis_b]],
                    );
                    close(
                        &format!("case {case} fixed Hessian[{axis_a},{axis_b}]"),
                        fixed.2[[axis_a, axis_b]],
                        hand.2[[axis_a, axis_b]],
                    );
                }
            }

            let mixed_score_mass = production.2[[3, 4]]
                .abs()
                .max(production.2[[3, 5]].abs())
                .max(production.2[[4, 5]].abs());
            assert!(
                mixed_score_mass > 1.0e-6,
                "case {case}: fixture must exercise a nonzero cross-score Hessian block"
            );
        }
    }

    /// Permanent backend-dispatch gate. Every scheduled compiled width evaluates
    /// the same generic row expression through the graph, eager fixed, eager
    /// dynamic, and retired strongest-hand authority. All covariance
    /// representations and both survival branches are covered.
    #[test]
    fn compiled_graph_schedule_matches_all_backends_every_width_932() {
        fn check_width<const DIM: usize>() {
            let k = DIM - 3;
            let slopes: Vec<f64> = (0..k)
                .map(|axis| {
                    let magnitude = 0.24 + 0.07 * axis as f64;
                    if axis % 2 == 0 { magnitude } else { -magnitude }
                })
                .collect();
            let scores: Vec<f64> = (0..k)
                .map(|axis| -0.9 + 1.8 * (axis + 1) as f64 / (k + 1) as f64)
                .collect();
            let diagonal = MarginalSlopeCovariance::Diagonal(Array1::from_shape_fn(k, |axis| {
                0.75 + 0.08 * axis as f64
            }));
            let full =
                MarginalSlopeCovariance::Full(Array2::from_shape_fn((k, k), |(row, col)| {
                    if row == col {
                        1.0 + 0.06 * row as f64
                    } else {
                        0.025 / (1.0 + row.abs_diff(col) as f64)
                    }
                }));
            let rank = k.min(3);
            let low_rank = MarginalSlopeCovariance::LowRank(Array2::from_shape_fn(
                (k, rank),
                |(row, column)| {
                    let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (0.18 + 0.03 * row as f64 + 0.05 * column as f64)
                },
            ));
            let covariances = [diagonal, full, low_rank];
            let q0 = -0.55 + 0.035 * k as f64;
            let q1 = 0.72 - 0.025 * k as f64;
            let qd1 = 0.85 + 0.04 * k as f64;
            let probit_scale = 0.71 + 0.021 * k as f64;
            let close = |backend: &str, channel: &str, actual: f64, expected: f64| {
                let tolerance = 8.0e-11 * actual.abs().max(expected.abs()).max(1.0);
                assert!(
                    actual.is_finite()
                        && expected.is_finite()
                        && (actual - expected).abs() <= tolerance,
                    "k={k} {backend} {channel}: actual={actual:+.16e}, hand={expected:+.16e}, tolerance={tolerance:.3e}"
                );
            };

            let mut production_workspace =
                RigidVectorRowWorkspace::new(k).expect("production width workspace");
            let mut graph_workspace = Order2GraphWorkspace::new();
            let mut dynamic_arena = DynamicJetArena::new();
            for (shape_index, covariance) in covariances.iter().enumerate() {
                for event in [0.0, 0.35, 1.0] {
                    let production = row_primary_closed_form_vector(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &scores,
                        covariance,
                        1.17,
                        event,
                        1.0e-8,
                        probit_scale,
                        &mut production_workspace,
                    )
                    .expect("production vector row");
                    let value = survival_marginal_slope_vector_neglog(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &scores,
                        covariance,
                        1.17,
                        event,
                        1.0e-8,
                        probit_scale,
                    )
                    .expect("zero-order canonical vector row");
                    if shape_index == 0 && event == 0.0 {
                        let expected_backend = if k <= 6 { "fixed" } else { "graph" };
                        assert_eq!(production_workspace.backend_name(), expected_backend);
                    }
                    let graph = row_primary_closed_form_vector_graph::<DIM>(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &scores,
                        covariance,
                        1.17,
                        event,
                        1.0e-8,
                        probit_scale,
                        &mut graph_workspace,
                    )
                    .expect("compiled graph vector row");
                    let dynamic = row_primary_closed_form_vector_dynamic(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &scores,
                        covariance,
                        1.17,
                        event,
                        1.0e-8,
                        probit_scale,
                        &mut dynamic_arena,
                    )
                    .expect("dynamic vector row");
                    let fixed = row_primary_closed_form_vector_fixed::<DIM>(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &scores,
                        covariance,
                        1.17,
                        event,
                        1.0e-8,
                        probit_scale,
                    )
                    .expect("fixed vector row");
                    let hand =
                        vector_hand_oracle_tests::row_primary_closed_form_vector_hand_reference(
                            q0,
                            q1,
                            qd1,
                            &slopes,
                            &scores,
                            covariance,
                            1.17,
                            event,
                            1.0e-8,
                            probit_scale,
                        )
                        .expect("strongest-hand vector row");

                    close(
                        "zero-order",
                        &format!("shape={shape_index} event={event} value"),
                        value,
                        hand.0,
                    );

                    for (backend, actual) in [
                        ("production", &production),
                        ("graph", &graph),
                        ("dynamic", &dynamic),
                        ("fixed", &fixed),
                    ] {
                        close(
                            backend,
                            &format!("shape={shape_index} event={event} value"),
                            actual.0,
                            hand.0,
                        );
                        for primary in 0..DIM {
                            close(
                                backend,
                                &format!("shape={shape_index} event={event} gradient[{primary}]"),
                                actual.1[primary],
                                hand.1[primary],
                            );
                            for other in 0..DIM {
                                close(
                                    backend,
                                    &format!(
                                        "shape={shape_index} event={event} Hessian[{primary},{other}]"
                                    ),
                                    actual.2[[primary, other]],
                                    hand.2[[primary, other]],
                                );
                            }
                        }
                    }
                }
            }
        }

        check_width::<4>();
        check_width::<5>();
        check_width::<6>();
        check_width::<7>();
        check_width::<8>();
        check_width::<9>();
        check_width::<10>();
        check_width::<11>();
        check_width::<12>();
        check_width::<13>();
        check_width::<14>();
        check_width::<15>();
        check_width::<16>();
    }

    /// First width beyond the compiled graph's exact u16 axis-mask domain.
    /// Production must select the dynamic packed lowering without touching the
    /// graph, while preserving the canonical row expression's complete V/G/H.
    #[test]
    fn dynamic_schedule_boundary_k14_matches_strongest_hand_vgh_932() {
        const K: usize = 14;
        const DIM: usize = K + 3;
        let slopes: Vec<f64> = (0..K)
            .map(|axis| {
                let magnitude = 0.19 + 0.045 * axis as f64;
                if axis % 2 == 0 { magnitude } else { -magnitude }
            })
            .collect();
        let scores: Vec<f64> = (0..K)
            .map(|axis| -1.1 + 2.2 * (axis + 1) as f64 / (K + 1) as f64)
            .collect();
        let diagonal = MarginalSlopeCovariance::Diagonal(Array1::from_shape_fn(K, |axis| {
            0.8 + 0.055 * axis as f64
        }));
        let full = MarginalSlopeCovariance::Full(Array2::from_shape_fn((K, K), |(row, col)| {
            if row == col {
                1.0 + 0.04 * row as f64
            } else {
                0.018 / (1.0 + row.abs_diff(col) as f64)
            }
        }));
        let low_rank =
            MarginalSlopeCovariance::LowRank(Array2::from_shape_fn((K, 3), |(row, column)| {
                let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
                sign * (0.14 + 0.018 * row as f64 + 0.035 * column as f64)
            }));
        let covariances = [diagonal, full, low_rank];
        let mut production_workspace =
            RigidVectorRowWorkspace::new(K).expect("k=14 production workspace");
        let mut dynamic_arena = DynamicJetArena::new();

        let close = |shape: usize, event: f64, channel: &str, actual: f64, expected: f64| {
            let tolerance = 8.0e-11 * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= tolerance,
                "k={K} shape={shape} event={event} {channel}: actual={actual:+.16e}, hand={expected:+.16e}, tolerance={tolerance:.3e}"
            );
        };

        for (shape, covariance) in covariances.iter().enumerate() {
            for event in [0.0, 0.35, 1.0] {
                let production = row_primary_closed_form_vector(
                    -0.31,
                    0.47,
                    1.09,
                    &slopes,
                    &scores,
                    covariance,
                    1.17,
                    event,
                    1.0e-8,
                    0.83,
                    &mut production_workspace,
                )
                .expect("production dynamic-boundary row");
                let value = survival_marginal_slope_vector_neglog(
                    -0.31, 0.47, 1.09, &slopes, &scores, covariance, 1.17, event, 1.0e-8, 0.83,
                )
                .expect("zero-order dynamic-boundary row");
                let dynamic = row_primary_closed_form_vector_dynamic(
                    -0.31,
                    0.47,
                    1.09,
                    &slopes,
                    &scores,
                    covariance,
                    1.17,
                    event,
                    1.0e-8,
                    0.83,
                    &mut dynamic_arena,
                )
                .expect("direct dynamic-boundary row");
                let hand = vector_hand_oracle_tests::row_primary_closed_form_vector_hand_reference(
                    -0.31, 0.47, 1.09, &slopes, &scores, covariance, 1.17, event, 1.0e-8, 0.83,
                )
                .expect("strongest-hand dynamic-boundary row");

                close(shape, event, "production value", production.0, hand.0);
                close(shape, event, "zero-order value", value, hand.0);
                close(shape, event, "dynamic value", dynamic.0, hand.0);
                for primary in 0..DIM {
                    close(
                        shape,
                        event,
                        &format!("production gradient[{primary}]"),
                        production.1[primary],
                        hand.1[primary],
                    );
                    close(
                        shape,
                        event,
                        &format!("dynamic gradient[{primary}]"),
                        dynamic.1[primary],
                        hand.1[primary],
                    );
                    for other in 0..DIM {
                        close(
                            shape,
                            event,
                            &format!("production Hessian[{primary},{other}]"),
                            production.2[[primary, other]],
                            hand.2[[primary, other]],
                        );
                        close(
                            shape,
                            event,
                            &format!("dynamic Hessian[{primary},{other}]"),
                            dynamic.2[[primary, other]],
                            hand.2[[primary, other]],
                        );
                    }
                }
            }
        }

        assert_eq!(production_workspace.backend_name(), "dynamic");
    }

    #[test]
    fn vector_workspace_is_compact_and_rejects_cross_width_reuse_932() {
        assert!(
            std::mem::size_of::<RigidVectorRowWorkspace>() <= 128,
            "backend selector must not inline the fixed graph tape"
        );
        assert!(RigidVectorRowWorkspace::new(0).is_err());

        let mut workspace = RigidVectorRowWorkspace::new(3).expect("k=3 workspace");
        let covariance = MarginalSlopeCovariance::Diagonal(Array1::ones(2));
        let error = row_primary_closed_form_vector(
            -0.2,
            0.4,
            1.1,
            &[0.3, -0.5],
            &[0.7, -1.2],
            &covariance,
            1.0,
            1.0,
            1.0e-8,
            0.9,
            &mut workspace,
        )
        .expect_err("a row cannot reuse workspace configured for another score width");
        assert!(error.contains("workspace width mismatch"), "{error}");
    }

    /// Temporary #932 release sweep for the fused fixed/graph cutover candidates.
    /// This compares every canonical backend directly so the production cutoff
    /// cannot hide a larger-width stack or lowering cost.
    #[test]
    fn release_measure_packed_widths_k1_to_k8_vs_strongest_hand_932() {
        use std::hint::black_box;
        use std::time::Instant;

        fn best_ns<T, F: FnMut() -> T>(iterations: usize, mut evaluate: F) -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(evaluate());
                }
                best = best.min(start.elapsed().as_nanos() as f64 / iterations as f64);
            }
            best
        }

        macro_rules! measure_width {
            ($k:literal, $dim:literal) => {{
                let slopes: Vec<f64> = (0..$k)
                    .map(|axis| {
                        let magnitude = 0.27 + 0.08 * axis as f64;
                        if axis % 2 == 0 { magnitude } else { -magnitude }
                    })
                    .collect();
                let scores: Vec<f64> = (0..$k)
                    .map(|axis| -1.25 + 2.5 * (axis + 1) as f64 / ($k + 1) as f64)
                    .collect();
                let diagonal =
                    MarginalSlopeCovariance::Diagonal(Array1::from_shape_fn($k, |axis| {
                        0.8 + 0.07 * axis as f64
                    }));
                let full =
                    MarginalSlopeCovariance::Full(Array2::from_shape_fn(($k, $k), |(row, col)| {
                        if row == col {
                            1.0 + 0.05 * row as f64
                        } else {
                            0.02 / (1.0 + row.abs_diff(col) as f64)
                        }
                    }));
                let low_rank = MarginalSlopeCovariance::LowRank(Array2::from_shape_fn(
                    ($k, $k.min(3)),
                    |(row, column)| {
                        let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
                        sign * (0.17 + 0.025 * row as f64 + 0.04 * column as f64)
                    },
                ));
                let cases = [
                    ("diagonal", &diagonal),
                    ("full", &full),
                    ("low_rank", &low_rank),
                ];

                for &(label, covariance) in &cases {
                    for event in [0.0, 1.0] {
                        let mut workspace = RigidVectorRowWorkspace::new($k)
                            .expect("packed production workspace");
                        let mut graph_workspace = Order2GraphWorkspace::new();
                        let mut dynamic_arena = DynamicJetArena::new();
                        let evaluate_production = |workspace: &mut RigidVectorRowWorkspace| {
                            row_primary_closed_form_vector(
                                -0.28, 0.53, 1.18, &slopes, &scores, covariance, 1.21, event,
                                1.0e-8, 0.87, workspace,
                            )
                            .expect("packed production width")
                        };
                        black_box(evaluate_production(&mut workspace));

                        let production_ns =
                            best_ns(5_000, || evaluate_production(&mut workspace));
                        let fixed_ns = best_ns(5_000, || {
                            row_primary_closed_form_vector_fixed::<$dim>(
                                -0.28, 0.53, 1.18, &slopes, &scores, covariance, 1.21, event,
                                1.0e-8, 0.87,
                            )
                            .expect("direct fixed width")
                        });
                        let graph_ns = best_ns(5_000, || {
                            row_primary_closed_form_vector_graph::<$dim>(
                                -0.28,
                                0.53,
                                1.18,
                                &slopes,
                                &scores,
                                covariance,
                                1.21,
                                event,
                                1.0e-8,
                                0.87,
                                &mut graph_workspace,
                            )
                            .expect("direct graph width")
                        });
                        let dynamic_ns = best_ns(5_000, || {
                            row_primary_closed_form_vector_dynamic(
                                -0.28,
                                0.53,
                                1.18,
                                &slopes,
                                &scores,
                                covariance,
                                1.21,
                                event,
                                1.0e-8,
                                0.87,
                                &mut dynamic_arena,
                            )
                            .expect("direct dynamic width")
                        });
                        let hand_ns = best_ns(5_000, || {
                            vector_hand_oracle_tests::row_primary_closed_form_vector_hand_reference(
                                -0.28,
                                0.53,
                                1.18,
                                &slopes,
                                &scores,
                                covariance,
                                1.21,
                                event,
                                1.0e-8,
                                0.87,
                            )
                            .expect("strongest-hand width")
                        });
                        let fastest_canonical_ns = fixed_ns.min(graph_ns).min(dynamic_ns);
                        eprintln!(
                            "G932_PACKED_WIDTH_RELEASE covariance={label} event={event:.0} \
                             k={} dim={} production_ns={production_ns:.3} fixed_ns={fixed_ns:.3} \
                             graph_ns={graph_ns:.3} dynamic_ns={dynamic_ns:.3} hand_ns={hand_ns:.3} \
                             production_over_fastest_canonical={:.6} hand_over_production={:.6}",
                            $k,
                            $dim,
                            production_ns / fastest_canonical_ns,
                            hand_ns / production_ns,
                        );
                    }
                }
            }};
        }

        measure_width!(1, 4);
        measure_width!(2, 5);
        measure_width!(3, 6);
        measure_width!(4, 7);
        measure_width!(5, 8);
        measure_width!(6, 9);
        measure_width!(7, 10);
        measure_width!(8, 11);
    }

    /// #932 cutover gate: every live K=1 scalar caller now uses the packed
    /// lowering of `rigid_row_nll`. Pin every returned channel against the
    /// retired strongest hand schedule over both event branches, ordinary
    /// rows, and opposite probability tails.
    #[test]
    fn canonical_rigid_order2_matches_strongest_hand_schedule_932() {
        let cases = [
            (-0.7, 0.4, 0.8, -0.3, 0.6, 1.0, 0.0, 0.75),
            (0.2, -0.5, 1.4, 0.9, -1.1, 0.8, 1.0, 1.0),
            (7.0, 6.2, 0.15, 1.7, -2.0, 1.3, 0.0, 0.55),
            (-7.5, -6.8, 2.1, -1.4, 1.8, 0.7, 1.0, 0.9),
        ];
        let close = |label: &str, actual: f64, expected: f64| {
            let tolerance = 2.0e-11 * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= tolerance,
                "{label}: canonical={actual:+.16e}, hand={expected:+.16e}, tolerance={tolerance:.3e}",
            );
        };

        for (case, &(q0, q1, qd1, g, z, w, d, scale)) in cases.iter().enumerate() {
            let canonical = row_primary_closed_form(q0, q1, qd1, g, z, w, d, 1.0e-8, scale)
                .expect("canonical rigid row");
            let hand = test_support::row_primary_closed_form_hand_reference(
                q0, q1, qd1, g, z, w, d, 1.0e-8, scale,
            )
            .expect("strongest hand rigid row");
            close(&format!("case {case} value"), canonical.0, hand.0);
            for a in 0..N_PRIMARY {
                close(
                    &format!("case {case} gradient[{a}]"),
                    canonical.1[a],
                    hand.1[a],
                );
                for b in 0..N_PRIMARY {
                    close(
                        &format!("case {case} Hessian[{a},{b}]"),
                        canonical.2[a][b],
                        hand.2[a][b],
                    );
                }
            }
        }
    }
}
