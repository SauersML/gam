//! Empirical KKT-refusal report for the survival marginal-slope joint
//! penalized Hessian (gam#979).
//!
//! Background. The residual #979 pathology is an ill-posed survival
//! marginal-slope shape whose joint penalized Hessian
//! `M = Jᵀ H J + S` carries a quadratically-flat near-null direction
//! (tiny `λ_min`, huge condition number, `nullity@tol = 0`). When that
//! direction is left unreduced and handed to the constrained joint Newton,
//! the inner solve cannot certify stationarity, seed screening escalates,
//! and the fit exhausts its bounded (deterministic) iteration budget without
//! ever certifying convergence.
//!
//! The temptation is to simply shrink or delete every near-null direction.
//! That is reward-hacking: a near-null direction can also be a *genuine*
//! non-stationarity (a near-separating direction the model legitimately
//! wants to push), and silently projecting it would hide a real failure to
//! converge. A naive nullspace-shrinkage fix of exactly this kind already
//! broke the `n ≥ 1000` spatial path.
//!
//! This module produces an **empirical, measured** report instead. It
//! assembles `M` and the joint score `g` at the pilot operating point,
//! eigendecomposes `M`, and for every near-null direction `v` measures the
//! gradient residual `r = |vᵀ g|`. A direction is classified as a
//! projectable PHANTOM only when its residual stays at or below the gate
//! `λ · step` (equivalently, its unconstrained Newton step `r/λ` stays
//! within the trust radius `step`): the optimizer would converge that
//! direction in under one step, so projecting it cannot change the optimum
//! and cannot hide non-stationarity. A direction whose residual EXCEEDS the
//! gate is flagged as real non-stationarity and is deliberately left alone.
//!
//! The joint Hessian here is small (`p_total ≲ 30`); the eigendecomposition
//! and gating are deterministic and cheap, so the whole report is a
//! crate-level diagnostic that does not require running the full outer fit.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_problem::diagnostics::KktRefusalDiagnosis;
use ndarray::{Array1, Array2, Array3};

use super::identifiability::{
    LogslopeBlockOperator, QChannelBlockOperator, SurvivalRowHessian, TimeBlockOperator,
};
use super::{LogslopeLayout, MarginalSlopeCovariance};
use gam_identifiability::families::compiler::RowJacobianOperator;

/// Default coefficient-space trust radius for the phantom gate. A near-null
/// direction is a projectable phantom only when its unconstrained Newton
/// step `r/λ` stays within this radius. It is the convergence step scale of
/// the inner solver: a direction already stationary to within this radius is
/// "converged", so projecting it is identity on the optimum. Deliberately
/// small so the gate is conservative — projecting fewer directions risks no
/// reward-hacking, only a missed cleanup (the inner solver still terminates at
/// its bounded iteration cap).
pub(crate) const KKT_PHANTOM_TRUST_RADIUS: f64 = 1.0e-3;

/// Relative tolerance applied to `λ_max` when counting eigenvalues as
/// structurally zero (`nullity@tol`) and when flooring the gate so a
/// genuinely zero-curvature direction is still allowed its numerical slop.
pub(crate) const KKT_RANK_REL_TOL: f64 = 1.0e-10;

/// Relative threshold (against `λ_max`) below which an eigenvalue is
/// considered "near-null" and is examined by the gate. Captures the
/// "tiny `λ_min`, huge condition number" regime up to condition `~1e8`.
pub(crate) const KKT_NEAR_NULL_REL_TOL: f64 = 1.0e-8;

/// One near-null eigen-direction of the joint penalized Hessian, with the
/// measured quantities the phantom gate decides on.
#[derive(Clone, Debug)]
pub(crate) struct NearNullDirection {
    /// Eigenvalue `λ` (curvature along this direction).
    pub(crate) eigenvalue: f64,
    /// Measured gradient residual `r = |vᵀ g|` along the eigenvector.
    pub(crate) gradient_residual: f64,
    /// Unconstrained Newton step magnitude `r / max(λ, floor)`.
    pub(crate) newton_step: f64,
    /// Gate threshold `max(λ, rank_floor·λ_max) · step`.
    pub(crate) gate_threshold: f64,
    /// `true` iff `gradient_residual ≤ gate_threshold` — the measured,
    /// non-reward-hacking signal that this direction is a phantom that may be
    /// projected away.
    pub(crate) phantom_projectable: bool,
    /// The eigenvector in full joint-coefficient coordinates.
    pub(crate) eigenvector: Array1<f64>,
}

/// Empirical refusal report on the joint penalized Hessian `M = Jᵀ H J + S`
/// and joint score `g` at a fixed operating point.
#[derive(Clone, Debug)]
pub(crate) struct SurvivalKktRefusalReport {
    /// Eigenvalues of `M`, ascending.
    pub(crate) eigenvalues_ascending: Vec<f64>,
    pub(crate) lambda_min: f64,
    pub(crate) lambda_max: f64,
    /// `λ_max / λ_min` (`+inf` when `λ_min == 0`).
    pub(crate) condition_number: f64,
    /// Count of eigenvalues `≤ KKT_RANK_REL_TOL · λ_max`.
    pub(crate) nullity_at_tol: usize,
    pub(crate) rank_tol: f64,
    /// The coefficient-space trust radius the gate used.
    pub(crate) step: f64,
    /// Every near-null direction, with its gate decision.
    pub(crate) near_null: Vec<NearNullDirection>,
    pub(crate) diagnosis: KktRefusalDiagnosis,
}

impl SurvivalKktRefusalReport {
    /// Number of near-null directions the gate classified as projectable
    /// phantoms (`r ≤ λ·step`).
    pub(crate) fn phantom_projectable_count(&self) -> usize {
        self.near_null
            .iter()
            .filter(|d| d.phantom_projectable)
            .count()
    }

    /// Number of near-null directions that FAILED the gate — measured real
    /// non-stationarity that must NOT be projected away.
    pub(crate) fn real_nonstationary_count(&self) -> usize {
        self.near_null
            .iter()
            .filter(|d| !d.phantom_projectable)
            .count()
    }

    /// `true` iff there is at least one near-null direction and EVERY
    /// near-null direction is a measured projectable phantom. This is the
    /// gate the design-reduction fallback consults before accepting a
    /// channel-collapsing reduction: the collapse is legitimate
    /// identifiability cleanup only when no collapsed direction carries real
    /// non-stationarity.
    pub(crate) fn all_near_null_are_phantom(&self) -> bool {
        !self.near_null.is_empty() && self.real_nonstationary_count() == 0
    }

    /// Compact one-line summary for structured logging. Reads every measured
    /// field so the empirical report is fully surfaced in the production log
    /// (eigenvalue spectrum extent, rank/gate scales, and the carrying
    /// near-null direction's residual / Newton step / eigenvector extent).
    pub(crate) fn summary(&self) -> String {
        let p = self.eigenvalues_ascending.len();
        let carrying = self.near_null.first();
        let (car_lambda, car_resid, car_newton, car_gate, car_vinf, car_phantom) = match carrying {
            Some(d) => (
                d.eigenvalue,
                d.gradient_residual,
                d.newton_step,
                d.gate_threshold,
                d.eigenvector.iter().fold(0.0_f64, |m, &v| m.max(v.abs())),
                d.phantom_projectable,
            ),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, false),
        };
        format!(
            "p={p} lambda_min={:.4e} lambda_max={:.4e} cond={:.4e} \
             nullity@tol={} rank_tol={:.4e} step={:.4e} near_null={} \
             phantom={} real={} diagnosis={} | carrying: lambda={:.4e} \
             resid={:.4e} newton_step={:.4e} gate={:.4e} v_inf={:.4e} phantom={}",
            self.lambda_min,
            self.lambda_max,
            self.condition_number,
            self.nullity_at_tol,
            self.rank_tol,
            self.step,
            self.near_null.len(),
            self.phantom_projectable_count(),
            self.real_nonstationary_count(),
            self.diagnosis.as_str(),
            car_lambda,
            car_resid,
            car_newton,
            car_gate,
            car_vinf,
            car_phantom,
        )
    }
}

/// The effective per-row designs that chain the stacked block layouts into
/// the survival primary state `(q0, q1, qd1, g_0, …, g_{K-1})`: the time columns
/// (`dq0`/`dq1`/`dqd1` driving `q0`/`q1`/`qd1`), the marginal columns (`m_dq`
/// driving `q0` and `q1`, `m_dqd1` driving `qd1`), and the canonical logslope
/// layout driving every physical score channel.
#[derive(Clone, Copy)]
pub(crate) struct SurvivalEffectiveDesigns<'a> {
    pub(crate) dq0: &'a Array2<f64>,
    pub(crate) dq1: &'a Array2<f64>,
    pub(crate) dqd1: &'a Array2<f64>,
    pub(crate) m_dq: &'a Array2<f64>,
    pub(crate) m_dqd1: &'a Array2<f64>,
    pub(crate) logslope_layout: &'a LogslopeLayout,
}

/// Per-row pilot operating point: the linearisation primary state
/// `(q0, q1, qd1, g_0, …, g_{K-1})` paired with the vector score, covariance,
/// weights, and event columns the survival row kernel needs.
#[derive(Clone, Copy)]
pub(crate) struct SurvivalPilotRows<'a> {
    pub(crate) q0: &'a Array1<f64>,
    pub(crate) q1: &'a Array1<f64>,
    pub(crate) qd1: &'a Array1<f64>,
    pub(crate) slopes: &'a Array2<f64>,
    pub(crate) z: &'a Array2<f64>,
    pub(crate) covariance: &'a MarginalSlopeCovariance,
    pub(crate) weights: &'a Array1<f64>,
    pub(crate) event: &'a Array1<f64>,
}

/// Link-kernel scalars shared by the survival row primary evaluation.
#[derive(Clone, Copy)]
pub(crate) struct SurvivalLinkParams {
    pub(crate) derivative_guard: f64,
    pub(crate) probit_scale: f64,
}

/// Assemble the joint penalized Hessian `M = Σ_i J_iᵀ H_i J_i + S` and the
/// joint score `g = Σ_i J_iᵀ grad_i` at a fixed operating point.
///
/// The per-row effective primary Jacobian `J_i` (`(3+K) × p_total`) chains the
/// stacked block designs into `(q0, q1, qd1, g_0, …, g_{K-1})`:
/// time columns drive `(q0, q1, qd1)`; marginal columns drive `(q0, q1)`
/// equally and `qd1` via the marginal derivative design; logslope columns
/// drive their physical score channels. This is the same effective metric the
/// identifiability compiler residualises in.
///
/// `row_hess` is the PSD per-row `(n × (3+K) × (3+K))` Hessian and `row_grad`
/// the matching gradient. `s_total`
/// is the unit-weight block-diagonal penalty (its NULLSPACE is what makes a
/// confounded direction unidentifiable for ALL smoothing parameters, so unit
/// weights suffice to expose the phantom).
pub(crate) fn assemble_joint_penalized_hessian_and_score(
    designs: SurvivalEffectiveDesigns<'_>,
    row_hess: &Array3<f64>,
    row_grad: &Array2<f64>,
    s_total: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    let SurvivalEffectiveDesigns {
        dq0,
        dq1,
        dqd1,
        m_dq,
        m_dqd1,
        logslope_layout,
    } = designs;
    let n = dq0.nrows();
    let p_time = dq0.ncols();
    let p_marg = m_dq.ncols();
    let primary_width = *row_hess
        .shape()
        .get(1)
        .ok_or_else(|| "kkt_refusal assembly: row_hess has no primary axis".to_string())?;
    let score_dim = primary_width.checked_sub(3).ok_or_else(|| {
        format!(
            "kkt_refusal assembly: row_hess primary width {primary_width} omits required q channels"
        )
    })?;
    if score_dim == 0 {
        return Err("kkt_refusal assembly requires at least one score channel".to_string());
    }
    logslope_layout.validate_for(score_dim)?;
    let p_log = logslope_layout.coefficient_design().ncols();
    let p_total = p_time + p_marg + p_log;
    for (name, rows, cols, want_rows, want_cols) in [
        ("dq1", dq1.nrows(), dq1.ncols(), n, p_time),
        ("dqd1", dqd1.nrows(), dqd1.ncols(), n, p_time),
        ("m_dqd1", m_dqd1.nrows(), m_dqd1.ncols(), n, p_marg),
    ] {
        if rows != want_rows || cols != want_cols {
            return Err(format!(
                "kkt_refusal assembly: {name} is {rows}x{cols}, expected {want_rows}x{want_cols}"
            ));
        }
    }
    if row_hess.shape() != [n, primary_width, primary_width] {
        return Err(format!(
            "kkt_refusal assembly: row_hess is {:?}, expected [{n}, {primary_width}, {primary_width}]",
            row_hess.shape()
        ));
    }
    if row_grad.shape() != [n, primary_width] {
        return Err(format!(
            "kkt_refusal assembly: row_grad is {:?}, expected [{n}, {primary_width}]",
            row_grad.shape()
        ));
    }
    if logslope_layout.coefficient_design().nrows() != n {
        return Err(format!(
            "kkt_refusal assembly: logslope rows {} do not match n={n}",
            logslope_layout.coefficient_design().nrows(),
        ));
    }
    if s_total.shape() != [p_total, p_total] {
        return Err(format!(
            "kkt_refusal assembly: s_total is {:?}, expected [{p_total}, {p_total}]",
            s_total.shape()
        ));
    }

    let mut m = s_total.clone();
    let mut g = Array1::<f64>::zeros(p_total);
    let time_operator = TimeBlockOperator::new(dq0.clone(), dq1.clone(), dqd1.clone(), score_dim);
    let marginal_operator = QChannelBlockOperator::new(m_dq.clone(), m_dqd1.clone(), score_dim);
    let logslope_operator = LogslopeBlockOperator::new(logslope_layout.clone(), score_dim)?;
    let mut time_rows = Array2::<f64>::zeros((n * primary_width, p_time));
    let mut marginal_rows = Array2::<f64>::zeros((n * primary_width, p_marg));
    let mut logslope_rows = Array2::<f64>::zeros((n * primary_width, p_log));
    time_operator.channel_flattened_rows(0..n, &mut time_rows);
    marginal_operator.channel_flattened_rows(0..n, &mut marginal_rows);
    logslope_operator.channel_flattened_rows(0..n, &mut logslope_rows);
    // Per-row `(3+K) × p_total` effective Jacobian and product, reused.
    let mut j_i = Array2::<f64>::zeros((primary_width, p_total));
    let mut hj = Array2::<f64>::zeros((primary_width, p_total));
    let marg_off = p_time;
    let log_off = p_time + p_marg;
    for i in 0..n {
        j_i.fill(0.0);
        for c in 0..p_time {
            for channel in 0..primary_width {
                j_i[[channel, c]] = time_rows[[i * primary_width + channel, c]];
            }
        }
        for c in 0..p_marg {
            let gc = marg_off + c;
            for channel in 0..primary_width {
                j_i[[channel, gc]] = marginal_rows[[i * primary_width + channel, c]];
            }
        }
        for c in 0..p_log {
            for channel in 0..primary_width {
                j_i[[channel, log_off + c]] = logslope_rows[[i * primary_width + channel, c]];
            }
        }
        // H_i J_i  (`(3+K) × p_total`).
        hj.fill(0.0);
        for a in 0..primary_width {
            for col in 0..p_total {
                let mut acc = 0.0;
                for b in 0..primary_width {
                    acc += row_hess[[i, a, b]] * j_i[[b, col]];
                }
                hj[[a, col]] = acc;
            }
        }
        // M += J_iᵀ (H_i J_i);  g += J_iᵀ grad_i
        for col in 0..p_total {
            for row in 0..p_total {
                let mut acc = 0.0;
                for a in 0..primary_width {
                    acc += j_i[[a, row]] * hj[[a, col]];
                }
                m[[row, col]] += acc;
            }
            let mut gacc = 0.0;
            for a in 0..primary_width {
                gacc += j_i[[a, col]] * row_grad[[i, a]];
            }
            g[col] += gacc;
        }
    }
    // Symmetrize M defensively (the accumulation is symmetric up to rounding).
    for row in 0..p_total {
        for col in (row + 1)..p_total {
            let avg = 0.5 * (m[[row, col]] + m[[col, row]]);
            m[[row, col]] = avg;
            m[[col, row]] = avg;
        }
    }
    Ok((m, g))
}

/// Build the empirical refusal report from an already-assembled joint
/// penalized Hessian `M` and joint score `g`, gating each near-null
/// direction at trust radius `step`.
pub(crate) fn build_refusal_report_from_hessian(
    m: &Array2<f64>,
    g: &Array1<f64>,
    step: f64,
) -> Result<SurvivalKktRefusalReport, String> {
    let p = m.nrows();
    if m.ncols() != p {
        return Err(format!(
            "kkt_refusal: M is {}x{}, must be square",
            p,
            m.ncols()
        ));
    }
    if g.len() != p {
        return Err(format!("kkt_refusal: g len {} != M dim {p}", g.len()));
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(format!(
            "kkt_refusal: step must be finite and positive, got {step}"
        ));
    }
    let (evals, evecs) = m
        .eigh(Side::Lower)
        .map_err(|e| format!("kkt_refusal: eigendecomposition of M failed: {e:?}"))?;
    // `eigh` yields ascending eigenvalues with eigenvectors in matching
    // columns, but do not rely on that ordering — sort an index permutation.
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| {
        evals[a]
            .partial_cmp(&evals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues_ascending: Vec<f64> = order.iter().map(|&i| evals[i]).collect();
    let lambda_min = eigenvalues_ascending.first().copied().unwrap_or(0.0);
    let lambda_max = eigenvalues_ascending.last().copied().unwrap_or(0.0);
    let condition_number = if lambda_min > 0.0 {
        lambda_max / lambda_min
    } else {
        f64::INFINITY
    };
    let rank_tol = KKT_RANK_REL_TOL * lambda_max.max(0.0);
    let nullity_at_tol = eigenvalues_ascending
        .iter()
        .filter(|&&v| v <= rank_tol)
        .count();
    // Gate floor: a structurally-zero direction is still allowed the numerical
    // residual a rank-floor curvature would produce over one step.
    let gate_floor_curvature = rank_tol;
    let near_null_threshold = KKT_NEAR_NULL_REL_TOL * lambda_max.max(0.0);

    let mut near_null: Vec<NearNullDirection> = Vec::new();
    for &idx in &order {
        let eigenvalue = evals[idx];
        if eigenvalue > near_null_threshold {
            // Ascending order: once above the near-null band, stop.
            break;
        }
        let v = evecs.column(idx);
        let mut r = 0.0;
        for k in 0..p {
            r += v[k] * g[k];
        }
        let gradient_residual = r.abs();
        let gate_curvature = eigenvalue.max(gate_floor_curvature);
        let gate_threshold = gate_curvature * step;
        let newton_step = gradient_residual / gate_curvature;
        let phantom_projectable = gradient_residual <= gate_threshold;
        near_null.push(NearNullDirection {
            eigenvalue,
            gradient_residual,
            newton_step,
            gate_threshold,
            phantom_projectable,
            eigenvector: v.to_owned(),
        });
    }

    // Classify by the H spectrum, matching the shared diagnosis carrier.
    // The per-direction `phantom_projectable` flags carry the actual gate
    // decision; the diagnosis is the spectrum-level label.
    let diagnosis = if nullity_at_tol > 0 {
        KktRefusalDiagnosis::RankDeficientHPen
    } else {
        // nullity@tol == 0 but a near-null band exists with a huge condition
        // number: the #979 "phantom multiplier with well-conditioned H" case.
        KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH
    };

    Ok(SurvivalKktRefusalReport {
        eigenvalues_ascending,
        lambda_min,
        lambda_max,
        condition_number,
        nullity_at_tol,
        rank_tol,
        step,
        near_null,
        diagnosis,
    })
}

/// End-to-end report: assemble `M`/`g` from the effective designs + per-row
/// derivatives + unit penalty, then gate. Convenience wrapper used by the
/// fit-entry fallback and the unit tests.
pub(crate) fn survival_kkt_refusal_report_from_designs(
    designs: SurvivalEffectiveDesigns<'_>,
    row_hess: &Array3<f64>,
    row_grad: &Array2<f64>,
    s_total: &Array2<f64>,
    step: f64,
) -> Result<SurvivalKktRefusalReport, String> {
    let (m, g) = assemble_joint_penalized_hessian_and_score(designs, row_hess, row_grad, s_total)?;
    build_refusal_report_from_hessian(&m, &g, step)
}

/// Per-row gradient `(n × (3+K))` of the survival neg-log-likelihood at the
/// pilot primary state — the score companion to
/// [`SurvivalRowHessian::from_pilot_primary_state`].
pub(crate) fn survival_row_gradient_from_pilot_primary_state(
    rows: SurvivalPilotRows<'_>,
    link: SurvivalLinkParams,
) -> Result<Array2<f64>, String> {
    let SurvivalPilotRows {
        q0,
        q1,
        qd1,
        slopes,
        z,
        covariance,
        weights,
        event,
    } = rows;
    let SurvivalLinkParams {
        derivative_guard,
        probit_scale,
    } = link;
    let n = q0.len();
    for (name, len) in [
        ("q1", q1.len()),
        ("qd1", qd1.len()),
        ("slopes", slopes.nrows()),
        ("z", z.nrows()),
        ("weights", weights.len()),
        ("event", event.len()),
    ] {
        if len != n {
            return Err(format!(
                "survival_row_gradient: length mismatch q0={n}, {name}={len}"
            ));
        }
    }
    covariance.validate("survival KKT pilot gradient covariance")?;
    let score_dim = covariance.dim();
    if slopes.ncols() != score_dim || z.ncols() != score_dim {
        return Err(format!(
            "survival_row_gradient: score dimension mismatch slopes={}, z={}, covariance={score_dim}",
            slopes.ncols(),
            z.ncols(),
        ));
    }
    let primary_width = 3 + score_dim;
    let mut out = Array2::<f64>::zeros((n, primary_width));
    let mut workspace = super::RigidVectorRowWorkspace::new(covariance)?;
    for i in 0..n {
        super::row_primary_closed_form_vector_into(
            q0[i],
            q1[i],
            qd1[i],
            slopes
                .row(i)
                .as_slice()
                .ok_or_else(|| "survival_row_gradient: slope row is not contiguous".to_string())?,
            z.row(i)
                .as_slice()
                .ok_or_else(|| "survival_row_gradient: score row is not contiguous".to_string())?,
            weights[i],
            event[i],
            derivative_guard,
            probit_scale,
            &mut workspace,
        )?;
        let (gradient, _) = workspace.derivatives();
        out.row_mut(i).assign(&gradient);
    }
    Ok(out)
}

/// Build the unit-weight block-diagonal total penalty `S` (dimension
/// `p_total = p_time + p_marg + p_log`) from per-block penalty matrices given
/// as full-block dense matrices. Each `*_block` matrix is square at its block
/// width and is embedded at its block offset. Penalties act block-locally, so
/// their nullspaces — the directions no smoothing parameter can ever bound —
/// are exactly the cross-block-confounded directions the report hunts.
pub(crate) fn assemble_unit_block_penalty(
    p_time: usize,
    p_marg: usize,
    p_log: usize,
    time_block: &Array2<f64>,
    marg_block: &Array2<f64>,
    log_block: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let p_total = p_time + p_marg + p_log;
    let mut s = Array2::<f64>::zeros((p_total, p_total));
    for (name, blk, off, width) in [
        ("time", time_block, 0usize, p_time),
        ("marginal", marg_block, p_time, p_marg),
        ("logslope", log_block, p_time + p_marg, p_log),
    ] {
        if blk.nrows() != width || blk.ncols() != width {
            return Err(format!(
                "assemble_unit_block_penalty: {name} block is {}x{}, expected {width}x{width}",
                blk.nrows(),
                blk.ncols(),
            ));
        }
        for r in 0..width {
            for c in 0..width {
                s[[off + r, off + c]] += blk[[r, c]];
            }
        }
    }
    Ok(s)
}

/// Sum a list of full-block dense penalty matrices (the `TimeBlockInput`
/// penalty representation, `Vec<Array2<f64>>`) into one `width × width`
/// block penalty. Each entry must already be `width × width`.
pub(crate) fn dense_block_penalty_from_dense_list(
    pens: &[Array2<f64>],
    width: usize,
) -> Result<Array2<f64>, String> {
    let mut s = Array2::<f64>::zeros((width, width));
    for (k, p) in pens.iter().enumerate() {
        if p.nrows() != width || p.ncols() != width {
            return Err(format!(
                "dense_block_penalty_from_dense_list: penalty {k} is {}x{}, expected {width}x{width}",
                p.nrows(),
                p.ncols(),
            ));
        }
        s += p;
    }
    Ok(s)
}

/// Embed a list of block-local [`super::BlockwisePenalty`] (the marginal /
/// logslope design penalty representation) into one `width × width` block
/// penalty by summing each `local` at its `col_range` offset.
pub(crate) fn dense_block_penalty_from_blockwise(
    pens: &[super::BlockwisePenalty],
    width: usize,
) -> Result<Array2<f64>, String> {
    let mut s = Array2::<f64>::zeros((width, width));
    for (k, p) in pens.iter().enumerate() {
        let r = p.col_range.clone();
        if r.end > width {
            return Err(format!(
                "dense_block_penalty_from_blockwise: penalty {k} col_range {}..{} exceeds width {width}",
                r.start, r.end,
            ));
        }
        let bl = r.len();
        if p.local.nrows() != bl || p.local.ncols() != bl {
            return Err(format!(
                "dense_block_penalty_from_blockwise: penalty {k} local is {}x{} but col_range width is {bl}",
                p.local.nrows(),
                p.local.ncols(),
            ));
        }
        for i in 0..bl {
            for j in 0..bl {
                s[[r.start + i, r.start + j]] += p.local[[i, j]];
            }
        }
    }
    Ok(s)
}

/// Build the empirical refusal report at a pilot operating point straight from
/// the survival design pieces, the PSD row Hessian, the pilot primary state,
/// and an already-assembled unit block penalty `s_total`. Computes the per-row
/// score internally (the gradient companion to the row Hessian) and gates each
/// near-null direction at trust radius `step`.
pub(crate) fn survival_kkt_refusal_report_at_pilot(
    designs: SurvivalEffectiveDesigns<'_>,
    row_hess: &SurvivalRowHessian,
    rows: SurvivalPilotRows<'_>,
    link: SurvivalLinkParams,
    s_total: &Array2<f64>,
    step: f64,
) -> Result<SurvivalKktRefusalReport, String> {
    use gam_identifiability::families::compiler::RowHessian;
    let row_grad = survival_row_gradient_from_pilot_primary_state(rows, link)?;
    let h_tensor = row_hess.evaluate_full();
    survival_kkt_refusal_report_from_designs(designs, &h_tensor, &row_grad, s_total, step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const K_SURVIVAL: usize = 4;

    fn shared_logslope_layout(design: Array2<f64>) -> LogslopeLayout {
        let n = design.nrows();
        super::super::LogslopeTopology::shared()
            .materialize_identity(
                super::super::DesignMatrix::from(design),
                &Array1::<f64>::zeros(n),
            )
            .unwrap()
    }

    /// Build a per-row Hessian tensor where every row is the same 4×4 PSD
    /// matrix `h`.
    fn constant_row_hess(n: usize, h: &Array2<f64>) -> Array3<f64> {
        let mut out = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for a in 0..K_SURVIVAL {
                for b in 0..K_SURVIVAL {
                    out[[i, a, b]] = h[[a, b]];
                }
            }
        }
        out
    }

    /// Rank-1-coupled primary Hessian on the (q1, g) sub-block:
    /// `[[1,1],[1,1]]`. The marginal channel drives q1 and the logslope
    /// channel drives g, so this q1↔g coupling is exactly what makes a
    /// SHARED marginal/logslope basis collinear in the effective metric —
    /// the real #979 confound. The (q1, g) block is PSD (eigenvalues 0, 2)
    /// with null direction `(q1 − g)`, so the joint (marginal − logslope)
    /// direction is flat. With independent bases the same coupling yields a
    /// well-conditioned Gram, so it doubles as the control geometry.
    fn coupled_q1_g_hessian() -> Array2<f64> {
        let mut h = Array2::<f64>::zeros((K_SURVIVAL, K_SURVIVAL));
        h[[1, 1]] = 1.0; // q1
        h[[3, 3]] = 1.0; // g
        h[[1, 3]] = 1.0; // q1↔g coupling
        h[[3, 1]] = 1.0;
        h
    }

    #[test]
    fn two_score_kkt_assembly_uses_all_physical_logslope_channels_932() {
        let n = 2;
        let logslope_design = ndarray::array![[1.0, 10.0], [2.0, 20.0]];
        let logslope_layout = super::super::LogslopeTopology::per_score(vec![0..1, 1..2], 2)
            .unwrap()
            .materialize_identity(
                super::super::DesignMatrix::from(logslope_design),
                &Array1::<f64>::zeros(n),
            )
            .unwrap();
        let empty = Array2::<f64>::zeros((n, 0));
        let mut row_hess = Array3::<f64>::zeros((n, 5, 5));
        for row in 0..n {
            for channel in 0..5 {
                row_hess[[row, channel, channel]] = 1.0;
            }
        }
        let row_grad = Array2::<f64>::zeros((n, 5));
        let penalty = Array2::<f64>::zeros((2, 2));
        let (hessian, score) = assemble_joint_penalized_hessian_and_score(
            SurvivalEffectiveDesigns {
                dq0: &empty,
                dq1: &empty,
                dqd1: &empty,
                m_dq: &empty,
                m_dqd1: &empty,
                logslope_layout: &logslope_layout,
            },
            &row_hess,
            &row_grad,
            &penalty,
        )
        .unwrap();

        assert_eq!(hessian[[0, 0]], 5.0);
        assert_eq!(hessian[[1, 1]], 500.0);
        assert_eq!(hessian[[0, 1]], 0.0);
        assert_eq!(hessian[[1, 0]], 0.0);
        assert_eq!(score, Array1::<f64>::zeros(2));
    }

    /// A confounded design: the marginal column and the logslope column are
    /// the SAME basis evaluated on the same rows, so in the q1/g effective
    /// metric the combined direction `(marginal, -logslope)` is flat. With
    /// NO independent gradient along it, it is a phantom: the gate must mark
    /// it projectable.
    #[test]
    fn confounded_flat_undriven_direction_is_a_projectable_phantom() {
        let n = 64;
        let p_time = 0;
        let p_marg = 1;
        let p_log = 1;
        // Shared basis values across rows.
        let mut basis = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            basis[[i, 0]] = (i as f64 / n as f64) - 0.5;
        }
        let dq0 = Array2::<f64>::zeros((n, p_time));
        let dq1 = Array2::<f64>::zeros((n, p_time));
        let dqd1 = Array2::<f64>::zeros((n, p_time));
        let m_dq = basis.clone();
        let m_dqd1 = Array2::<f64>::zeros((n, p_marg));
        let g_dg = basis.clone();
        let logslope_layout = shared_logslope_layout(g_dg);
        let row_hess = constant_row_hess(n, &coupled_q1_g_hessian());
        // Gradient driving BOTH channels equally and in the SAME sign as the
        // shared basis, so the score lies along (marginal + logslope) — the
        // RANGE direction — and is orthogonal to the flat (marginal - logslope)
        // null direction. The null direction therefore carries no residual.
        let mut row_grad = Array2::<f64>::zeros((n, K_SURVIVAL));
        for i in 0..n {
            row_grad[[i, 1]] = basis[[i, 0]]; // q1 score
            row_grad[[i, 3]] = basis[[i, 0]]; // g score
        }
        let s_total = Array2::<f64>::zeros((p_marg + p_log, p_marg + p_log));
        let report = survival_kkt_refusal_report_from_designs(
            SurvivalEffectiveDesigns {
                dq0: &dq0,
                dq1: &dq1,
                dqd1: &dqd1,
                m_dq: &m_dq,
                m_dqd1: &m_dqd1,
                logslope_layout: &logslope_layout,
            },
            &row_hess,
            &row_grad,
            &s_total,
            KKT_PHANTOM_TRUST_RADIUS,
        )
        .expect("report builds");

        // Confound ⇒ tiny λ_min, huge condition number.
        assert!(
            report.lambda_min <= 1e-9 * report.lambda_max.max(1.0),
            "expected a near-null λ_min, got {:.3e} (λ_max={:.3e})",
            report.lambda_min,
            report.lambda_max
        );
        assert!(
            report.condition_number >= 1e8,
            "expected huge condition number, got {:.3e}",
            report.condition_number
        );
        assert!(
            !report.near_null.is_empty(),
            "expected at least one near-null direction"
        );
        // The flat direction is undriven ⇒ measured phantom ⇒ projectable.
        assert!(
            report.all_near_null_are_phantom(),
            "flat + undriven confound must be a projectable phantom; report: {}",
            report.summary()
        );
    }

    /// Same flat geometry, but now inject a gradient ALONG the flat
    /// (marginal - logslope) direction: a near-separating direction the model
    /// genuinely wants to push. The gate MUST refuse to project it — doing so
    /// would hide real non-stationarity (reward-hacking).
    #[test]
    fn flat_but_driven_direction_is_real_nonstationarity_not_projected() {
        let n = 64;
        let p_marg = 1;
        let p_log = 1;
        let mut basis = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            basis[[i, 0]] = (i as f64 / n as f64) - 0.5;
        }
        let dq0 = Array2::<f64>::zeros((n, 0));
        let dq1 = Array2::<f64>::zeros((n, 0));
        let dqd1 = Array2::<f64>::zeros((n, 0));
        let m_dq = basis.clone();
        let m_dqd1 = Array2::<f64>::zeros((n, p_marg));
        let g_dg = basis.clone();
        let logslope_layout = shared_logslope_layout(g_dg);
        let row_hess = constant_row_hess(n, &coupled_q1_g_hessian());
        // Score along the FLAT direction: marginal channel pushed up, logslope
        // channel pushed down (opposite signs) ⇒ projects onto (marginal -
        // logslope). Large magnitude ⇒ a separating pull.
        let mut row_grad = Array2::<f64>::zeros((n, K_SURVIVAL));
        for i in 0..n {
            row_grad[[i, 1]] = 50.0 * basis[[i, 0]]; // q1 score (marginal up)
            row_grad[[i, 3]] = -50.0 * basis[[i, 0]]; // g score (logslope down)
        }
        let s_total = Array2::<f64>::zeros((p_marg + p_log, p_marg + p_log));
        let report = survival_kkt_refusal_report_from_designs(
            SurvivalEffectiveDesigns {
                dq0: &dq0,
                dq1: &dq1,
                dqd1: &dqd1,
                m_dq: &m_dq,
                m_dqd1: &m_dqd1,
                logslope_layout: &logslope_layout,
            },
            &row_hess,
            &row_grad,
            &s_total,
            KKT_PHANTOM_TRUST_RADIUS,
        )
        .expect("report builds");

        assert!(
            !report.near_null.is_empty(),
            "expected a near-null direction"
        );
        // The flat direction is DRIVEN ⇒ NOT a phantom ⇒ must not be projected.
        assert!(
            report.real_nonstationary_count() >= 1,
            "driven flat direction must be flagged real non-stationarity; report: {}",
            report.summary()
        );
        assert!(
            !report.all_near_null_are_phantom(),
            "must refuse to project a driven flat direction; report: {}",
            report.summary()
        );
    }

    /// A well-conditioned, fully-identified problem: independent marginal and
    /// logslope bases, unit curvature. No near-null direction, so there is
    /// nothing to project and `all_near_null_are_phantom` is false (empty).
    #[test]
    fn well_conditioned_problem_has_no_near_null_direction() {
        let n = 64;
        let p_marg = 1;
        let p_log = 1;
        let mut marg = Array2::<f64>::zeros((n, 1));
        let mut logb = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let t = i as f64 / n as f64;
            marg[[i, 0]] = t - 0.5;
            logb[[i, 0]] = (2.0 * std::f64::consts::PI * t).sin();
        }
        let dq0 = Array2::<f64>::zeros((n, 0));
        let dq1 = Array2::<f64>::zeros((n, 0));
        let dqd1 = Array2::<f64>::zeros((n, 0));
        let m_dqd1 = Array2::<f64>::zeros((n, p_marg));
        let logslope_layout = shared_logslope_layout(logb);
        let row_hess = constant_row_hess(n, &coupled_q1_g_hessian());
        let row_grad = Array2::<f64>::zeros((n, K_SURVIVAL));
        let s_total = Array2::<f64>::zeros((p_marg + p_log, p_marg + p_log));
        let report = survival_kkt_refusal_report_from_designs(
            SurvivalEffectiveDesigns {
                dq0: &dq0,
                dq1: &dq1,
                dqd1: &dqd1,
                m_dq: &marg,
                m_dqd1: &m_dqd1,
                logslope_layout: &logslope_layout,
            },
            &row_hess,
            &row_grad,
            &s_total,
            KKT_PHANTOM_TRUST_RADIUS,
        )
        .expect("report builds");
        assert!(
            report.condition_number < 1e6,
            "well-conditioned design must have modest condition number, got {:.3e}",
            report.condition_number
        );
        assert!(
            report.near_null.is_empty(),
            "well-conditioned design must have no near-null direction; report: {}",
            report.summary()
        );
        assert!(!report.all_near_null_are_phantom());
    }

    /// A penalty-null but data-curved direction must NOT register as near-null:
    /// the data curvature identifies it even though the penalty does not. This
    /// guards against re-introducing the broken naive nullspace shrinkage.
    #[test]
    fn penalty_null_but_data_curved_direction_is_not_near_null() {
        let n = 64;
        let p_marg = 1;
        let p_log = 1;
        let mut marg = Array2::<f64>::zeros((n, 1));
        let mut logb = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let t = i as f64 / n as f64;
            marg[[i, 0]] = t - 0.5;
            logb[[i, 0]] = (3.0 * t).cos();
        }
        let dq0 = Array2::<f64>::zeros((n, 0));
        let dq1 = Array2::<f64>::zeros((n, 0));
        let dqd1 = Array2::<f64>::zeros((n, 0));
        let m_dqd1 = Array2::<f64>::zeros((n, p_marg));
        let logslope_layout = shared_logslope_layout(logb);
        let row_hess = constant_row_hess(n, &coupled_q1_g_hessian());
        let row_grad = Array2::<f64>::zeros((n, K_SURVIVAL));
        // Zero penalty everywhere: every direction is penalty-null, yet the
        // data curvature (unit q1/g Hessian on independent bases) identifies
        // both, so none is near-null.
        let s_total = Array2::<f64>::zeros((p_marg + p_log, p_marg + p_log));
        let report = survival_kkt_refusal_report_from_designs(
            SurvivalEffectiveDesigns {
                dq0: &dq0,
                dq1: &dq1,
                dqd1: &dqd1,
                m_dq: &marg,
                m_dqd1: &m_dqd1,
                logslope_layout: &logslope_layout,
            },
            &row_hess,
            &row_grad,
            &s_total,
            KKT_PHANTOM_TRUST_RADIUS,
        )
        .expect("report builds");
        assert!(
            report.near_null.is_empty(),
            "data-identified directions must not be near-null; report: {}",
            report.summary()
        );
    }
}
