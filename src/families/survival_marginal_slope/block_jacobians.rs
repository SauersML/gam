//! Per-block effective Jacobians: the family scalars and the rigid / flex /
//! time-wiggle `BlockEffectiveJacobian` implementations (time, marginal,
//! logslope, score-warp, link-dev) plus the primary->joint row chain.

use super::*;

/// Per-row scalars for survival marginal-slope Jacobian evaluation at a given β.
///
/// Fields:
/// - `q0_i`: entry-time probit argument (per-row, length n)
/// - `q1_i`: exit-time probit argument (per-row, length n)
/// - `qd1_i`: derivative probit argument (per-row, length n)
/// - `g_i`: per-row log-slope value `g = logslope_design · β_logslope`
/// - `c_i`: `sqrt(1 + (s·g_i)²)` (per-row, length n)
/// - `s`: probit scale (scalar, = `probit_frailty_scale()`)
/// - `z_i`: per-row covariate score (length n)
/// - `observed_g_i`: per-row observed baseline g (length n, pre-β contribution)
pub struct SurvivalMarginalSlopeFamilyScalars {
    pub q0_i: Vec<f64>,
    pub q1_i: Vec<f64>,
    pub qd1_i: Vec<f64>,
    pub g_i: Vec<f64>,
    pub c_i: Vec<f64>,
    pub s: f64,
    pub z_i: Vec<f64>,
    pub observed_g_i: Vec<f64>,
}


impl SurvivalMarginalSlopeFamilyScalars {
    /// Construct with c_i computed from g_i and s.
    pub fn new(
        q0_i: Vec<f64>,
        q1_i: Vec<f64>,
        qd1_i: Vec<f64>,
        g_i: Vec<f64>,
        s: f64,
        z_i: Vec<f64>,
        observed_g_i: Vec<f64>,
    ) -> Self {
        let c_i: Vec<f64> = g_i
            .iter()
            .map(|&g| (1.0 + (s * g).powi(2)).sqrt())
            .collect();
        Self {
            q0_i,
            q1_i,
            qd1_i,
            g_i,
            c_i,
            s,
            z_i,
            observed_g_i,
        }
    }
}


/// n_outputs=3 stacked Jacobian for the logslope block.
///
/// The logslope block contributes `g_i = logslope_design[i] · β` to each row.
/// The three stacked output rows for row i are:
///
/// ```text
/// ∂η0[i]/∂β = (q0[i] · s²·g[i]/c[i] + s·z[i]) · G[i,:]
/// ∂η1[i]/∂β = (q1[i] · s²·g[i]/c[i] + s·z[i]) · G[i,:]
/// ∂ad1[i]/∂β = qd1[i] · s²·g[i]/c[i] · G[i,:]
/// ```
///
/// At g=0 (β=0 init): c=1, s²·g/c=0, so:
/// ```text
/// ∂η0[i]/∂β = s·z[i] · G[i,:]
/// ∂η1[i]/∂β = s·z[i] · G[i,:]
/// ∂ad1[i]/∂β = 0
/// ```
pub struct LogslopeBlockJacobian {
    /// The logslope basis design (n × p_logslope). Held behind an `Arc` so a
    /// materialized design is shared with its owner rather than deep-copied —
    /// at biobank scale each retained `n × p` copy in these construction-time
    /// callbacks was hundreds of MiB held for the whole fit (#979 OOM).
    pub(crate) design: Arc<Array2<f64>>,
    /// Per-row covariate score z_i (length n).
    pub(crate) z: Vec<f64>,
    /// Probit scale s.
    pub(crate) s: f64,
}


impl LogslopeBlockJacobian {
    pub fn new(design: impl Into<Arc<Array2<f64>>>, z: Vec<f64>, s: f64) -> Self {
        Self {
            design: design.into(),
            z,
            s,
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for LogslopeBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        // Read s_f from the linearization state so that outer-loop σ updates are
        // reflected without requiring the spec to be rebuilt.  Every construction
        // site sets probit_frailty_scale = 1.0 when it does not know the family's
        // σ; `self.s` carries the construction-time value as a fallback.  Use the
        // state value when positive and finite; fall back to self.s otherwise.
        // For the no-frailty case both are 1.0 so the choice is immaterial.
        let s = if state.probit_frailty_scale > 0.0 && state.probit_frailty_scale.is_finite() {
            state.probit_frailty_scale
        } else {
            self.s
        };

        // Compute per-row g_i = logslope_design[i,:] · β directly from state.beta.
        // This block owns the logslope design so g is always self-computable without
        // family_scalars.  Truncate to min(p, beta.len()) to handle the pre-fit
        // initialisation call where beta may be shorter or empty.
        let beta = state.beta;
        let p_use = p.min(beta.len());
        let mut g_rows = vec![0.0_f64; chunk];
        for i in rows.clone() {
            let local_i = i - rows.start;
            for j in 0..p_use {
                g_rows[local_i] += self.design[[i, j]] * beta[j];
            }
        }

        // Hard contract: when any g_i is nonzero the per-row primary scalars
        // (q0, q1, qd1) from the time/marginal blocks are required for the correct
        // hyperbolic formula (q·s²g/c + s·z).  Those scalars live in family_scalars.
        // A caller operating at non-init β must populate them.
        let scalars: Option<&SurvivalMarginalSlopeFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        let any_nonzero_g = g_rows.iter().any(|&gi| gi != 0.0);
        if any_nonzero_g && scalars.is_none() {
            return Err("survival marginal-slope logslope block requires \
                 SurvivalMarginalSlopeFamilyScalars when beta != 0 \
                 (g_i != 0 for at least one row); got family_scalars: None. \
                 The caller must compute per-row (q0, q1, qd1) at the current \
                 beta and pass them via FamilyLinearizationState::family_scalars."
                .to_string());
        }

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            // g_i computed from beta above; c_i from family_scalars when present,
            // otherwise computed from g_i.  q0/q1/qd1 from family_scalars -
            // guaranteed present by the contract check whenever g_i != 0.
            let g = g_rows[local_i];
            let (q0, q1, qd1, c) = match scalars {
                Some(sc) => (sc.q0_i[i], sc.q1_i[i], sc.qd1_i[i], sc.c_i[i]),
                None => {
                    // g == 0.0 here (enforced by contract above), so c = 1.
                    // The q terms vanish: q * s^2 * 0 / 1 = 0.
                    (0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64)
                }
            };
            let z_i = self.z[i];
            let sg_over_c = if g == 0.0 { 0.0 } else { s * s * g / c };
            let coeff_eta0 = q0 * sg_over_c + s * z_i;
            let coeff_eta1 = q1 * sg_over_c + s * z_i;
            let coeff_ad1 = qd1 * sg_over_c;

            for j in 0..p {
                let g_ij = self.design[[i, j]];
                jac[[local_i, j]] = coeff_eta0 * g_ij;
                jac[[chunk + local_i, j]] = coeff_eta1 * g_ij;
                jac[[2 * chunk + local_i, j]] = coeff_ad1 * g_ij;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// n_outputs=3 stacked Jacobian for the marginal block.
///
/// The marginal block contributes identically to q0 and q1 (both entry and
/// exit probit arguments) but not to ad1 (the derivative). The stacked Jacobian is:
///
/// ```text
/// ∂η0[i]/∂β = c[i] · M[i,:]
/// ∂η1[i]/∂β = c[i] · M[i,:]
/// ∂ad1[i]/∂β = 0
/// ```
///
/// At g=0 (β=0 init): c=1, so each row is just M[i,:].
pub struct MarginalBlockJacobian {
    /// The marginal basis design (n × p_marginal), `Arc`-shared with its
    /// owner (see [`LogslopeBlockJacobian::design`]).
    pub(crate) design: Arc<Array2<f64>>,
}


impl MarginalBlockJacobian {
    pub fn new(design: impl Into<Arc<Array2<f64>>>) -> Self {
        Self {
            design: design.into(),
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for MarginalBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;

        // c_i = sqrt(1 + (s * g_i)^2) depends on the logslope block's g at the
        // current beta.  This block does not own the logslope design so it cannot
        // compute c from beta alone.  Hard contract: when state.beta is non-empty
        // (post-init), family_scalars must carry SurvivalMarginalSlopeFamilyScalars
        // so the correct c_i is used.  At init (beta empty or all-zero), c_i = 1
        // exactly and family_scalars may be omitted.
        let scalars: Option<&SurvivalMarginalSlopeFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        let beta_nonzero = state.beta.iter().any(|&b| b != 0.0);
        if beta_nonzero && scalars.is_none() {
            return Err("survival marginal-slope marginal block requires \
                 SurvivalMarginalSlopeFamilyScalars when beta != 0 (c_i != 1 in general); \
                 got family_scalars: None. The caller must populate per-row c_i via \
                 FamilyLinearizationState::family_scalars."
                .to_string());
        }

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let c = match scalars {
                Some(sc) => sc.c_i[i],
                // beta is all-zero here (enforced above), so g = 0 and c = 1.
                None => 1.0_f64,
            };
            for j in 0..p {
                let m_ij = c * self.design[[i, j]];
                jac[[local_i, j]] = m_ij;
                jac[[chunk + local_i, j]] = m_ij;
                // jac[[2*n + i, j]] = 0 -- ad1 row stays zero
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// n_outputs=3 stacked Jacobian for the time block.
///
/// The time block contributes separately to η0 (entry), η1 (exit), and ad1
/// (derivative) via three distinct design matrices. The stacked Jacobian is:
///
/// ```text
/// ∂η0[i]/∂β = c[i] · T_entry[i,:]
/// ∂η1[i]/∂β = c[i] · T_exit[i,:]
/// ∂ad1[i]/∂β = c[i] · T_deriv[i,:]
/// ```
///
/// At g=0 (β=0 init): c=1.
pub struct TimeBlockJacobian {
    // `Arc`-shared with their owners (see [`LogslopeBlockJacobian::design`]).
    pub(crate) design_entry: Arc<Array2<f64>>,
    pub(crate) design_exit: Arc<Array2<f64>>,
    pub(crate) design_deriv: Arc<Array2<f64>>,
}


impl TimeBlockJacobian {
    pub fn new(
        design_entry: impl Into<Arc<Array2<f64>>>,
        design_exit: impl Into<Arc<Array2<f64>>>,
        design_deriv: impl Into<Arc<Array2<f64>>>,
    ) -> Self {
        Self {
            design_entry: design_entry.into(),
            design_exit: design_exit.into(),
            design_deriv: design_deriv.into(),
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for TimeBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design_entry.nrows();
        let p = self.design_entry.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;

        if self.design_exit.nrows() != n || self.design_deriv.nrows() != n {
            return Err(format!(
                "TimeBlockJacobian: design row count mismatch \
                 entry={n} exit={} deriv={}",
                self.design_exit.nrows(),
                self.design_deriv.nrows(),
            ));
        }
        if self.design_exit.ncols() != p || self.design_deriv.ncols() != p {
            return Err(format!(
                "TimeBlockJacobian: design col count mismatch \
                 entry={p} exit={} deriv={}",
                self.design_exit.ncols(),
                self.design_deriv.ncols(),
            ));
        }

        // c_i = sqrt(1 + (s * g_i)^2) depends on the logslope block's g.  This block
        // does not own the logslope design.  Hard contract: when beta is non-empty/nonzero,
        // family_scalars must carry SurvivalMarginalSlopeFamilyScalars with the correct c_i.
        // At init (beta empty or all-zero), c_i = 1 exactly.
        let scalars: Option<&SurvivalMarginalSlopeFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        let beta_nonzero = state.beta.iter().any(|&b| b != 0.0);
        if beta_nonzero && scalars.is_none() {
            return Err("survival marginal-slope time block requires \
                 SurvivalMarginalSlopeFamilyScalars when beta != 0 (c_i != 1 in general); \
                 got family_scalars: None. The caller must populate per-row c_i via \
                 FamilyLinearizationState::family_scalars."
                .to_string());
        }

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let c = match scalars {
                Some(sc) => sc.c_i[i],
                // beta is all-zero here (enforced above), so g = 0 and c = 1.
                None => 1.0_f64,
            };
            for j in 0..p {
                jac[[local_i, j]] = c * self.design_entry[[i, j]];
                jac[[chunk + local_i, j]] = c * self.design_exit[[i, j]];
                jac[[2 * chunk + local_i, j]] = c * self.design_deriv[[i, j]];
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


// ── Full flex-chain Jacobian support ─────────────────────────────────
//
// When score_warp (β_h) and/or link_dev (β_w) are non-zero the rigid
// hyperbolic formula η = q·c + s·z·g is no longer exact.  The calibrated
// intercept `a` is an implicit function of all primary coordinates and the
// correct linearization is the IFT chain:
//
//   ∂η/∂u_k = chi · a_u[k] + rho[k]
//
// computed by `compute_survival_timepoint_exact`.  `chain_primary_to_joint_row`
// contracts these per-primary-coordinate derivatives with the block-level
// primary-to-coefficient maps to produce the joint-β Jacobian row.

/// Per-row IFT-corrected derivatives from the full flex chain.
///
/// Populated once per fit at the current β by a higher-level driver and
/// stored in `FamilyLinearizationState::family_scalars`.  When present, the
/// block Jacobian callbacks (`LogslopeBlockJacobian`, etc.) switch from the
/// rigid hyperbolic formula to the exact IFT chain.
///
/// Field layout:
/// - `eta_u_entry[i]`:  `∂η0/∂u_k` for primary slot `k=0..p_primary`, row `i`
/// - `eta_u_exit[i]`:   `∂η1/∂u_k`, row `i`
/// - `chi_exit[i]`:      observed χ₁ at exit (the density factor, > 0)
/// - `chi_u_exit[i]`:   `∂χ₁/∂u_k`, row `i`
/// - `d_exit[i]`:        normalisation D₁ at exit (> 0)
/// - `d_u_exit[i]`:     `∂D₁/∂u_k`, row `i`
/// - `q1_i[i]`:          exit-time probit argument q₁
/// - `qd1_i[i]`:         exit-time derivative qd₁
/// - `dq0_time[i]`:     `∂q0/∂β_time` row vector (length p_time)
/// - `dq1_time[i]`:     `∂q1/∂β_time` row vector (length p_time)
/// - `dqd1_time[i]`:    `∂qd1/∂β_time` row vector (length p_time)
/// - `dq0_marginal[i]`: `∂q0/∂β_marginal` row vector (length p_marginal)
/// - `dq1_marginal[i]`: `∂q1/∂β_marginal` row vector (length p_marginal)
/// - `dqd1_marginal[i]`:`∂qd1/∂β_marginal` row vector (length p_marginal)
/// - `logslope_design`:  the logslope design matrix (n × p_logslope)
/// - `score_warp_design`:score_warp design matrix (n × p_h), or empty
/// - `link_dev_design`:  link_dev design matrix (n × p_w), or empty
/// - `p_primary`:        total number of primary coordinates
/// - `p_q0`, `p_q1`, `p_qd1`, `p_g`: primary slot indices (scalar)
/// - `p_h_start`, `p_h_len`: h-slot range in primary
/// - `p_w_start`, `p_w_len`: w-slot range in primary
pub struct SurvivalFlexFamilyScalars {
    /// Shape n × p_primary; row i = ∂η0/∂u
    pub eta_u_entry: Array2<f64>,
    /// Shape n × p_primary; row i = ∂η1/∂u
    pub eta_u_exit: Array2<f64>,
    /// Length n; χ₁ at exit
    pub chi_exit: Vec<f64>,
    /// Shape n × p_primary; row i = ∂χ₁/∂u
    pub chi_u_exit: Array2<f64>,
    /// Length n; D₁ at exit
    pub d_exit: Vec<f64>,
    /// Shape n × p_primary; row i = ∂D₁/∂u
    pub d_u_exit: Array2<f64>,
    /// Length n; exit-time probit argument q₁
    pub q1_i: Vec<f64>,
    /// Length n; exit-time derivative qd₁
    pub qd1_i: Vec<f64>,
    /// ∂q0/∂β_time per row; shape n × p_time
    pub dq0_time: Array2<f64>,
    /// ∂q1/∂β_time per row; shape n × p_time
    pub dq1_time: Array2<f64>,
    /// ∂qd1/∂β_time per row; shape n × p_time
    pub dqd1_time: Array2<f64>,
    /// ∂q0/∂β_marginal per row; shape n × p_marginal
    pub dq0_marginal: Array2<f64>,
    /// ∂q1/∂β_marginal per row; shape n × p_marginal
    pub dq1_marginal: Array2<f64>,
    /// ∂qd1/∂β_marginal per row; shape n × p_marginal
    pub dqd1_marginal: Array2<f64>,
    /// Logslope design (n × p_logslope)
    pub logslope_design: Array2<f64>,
    /// Score-warp design (n × p_h); zero-column if inactive
    pub score_warp_design: Array2<f64>,
    /// Link-dev design (n × p_w); zero-column if inactive
    pub link_dev_design: Array2<f64>,
    /// Total primary coordinate count
    pub p_primary: usize,
    /// Primary slot index for q0 (= 0)
    pub idx_q0: usize,
    /// Primary slot index for q1 (= 1)
    pub idx_q1: usize,
    /// Primary slot index for qd1 (= 2)
    pub idx_qd1: usize,
    /// Primary slot index for g (= 3)
    pub idx_g: usize,
    /// Start of h-slots in primary coordinate vector
    pub h_start: usize,
    /// Number of h-slots (0 if score_warp inactive)
    pub h_len: usize,
    /// Start of w-slots in primary coordinate vector
    pub w_start: usize,
    /// Number of w-slots (0 if link_dev inactive)
    pub w_len: usize,
}


/// Contract a primary-coordinate derivative vector `coeff_u` (length
/// `p_primary`) with the block-level primary-to-coefficient maps, writing the
/// joint-β Jacobian row into `out` (length = `slices.total`).
///
/// # Channel routing
///
/// | Block slice       | Primary slots touched            | Map                  |
/// |-------------------|----------------------------------|----------------------|
/// | `slices.time`     | idx_q0, idx_q1, idx_qd1          | `q.dq*_time`         |
/// | `slices.marginal` | idx_q0, idx_q1, idx_qd1          | `q.dq*_marginal`     |
/// | `slices.logslope` | idx_g                            | `logslope_row`       |
/// | `slices.score_warp`| h-slots                         | identity             |
/// | `slices.link_dev` | w-slots                          | identity             |
///
/// At rigid β=0 with `beta_h = beta_w = 0` this reduces to:
/// - time: `chi * dq*_time + rho_q* * dq*_time` which, for the rigid formula
///   (rho_q = 0 for off-diagonal primaries), equals `c * dq*_time`
/// - logslope: `(q*c1 + s*z) * logslope_row`
///
/// # Arguments
/// - `coeff_u`: `ArrayView1<f64>` of length `p_primary`; the ∂·/∂u vector
/// - `q`: per-row primary chain maps (dq*_time, dq*_marginal)
/// - `logslope_row`: the logslope design row for this observation
/// - `flex`: the flex scalars (carries slot indices and p_primary)
/// - `slices`: the block-β offset layout
/// - `out`: output buffer of length `slices.total`; zeroed on entry by contract
pub fn chain_primary_to_joint_row(
    coeff_u: ndarray::ArrayView1<'_, f64>,
    q: &SurvivalFlexPerRowChain,
    logslope_row: ndarray::ArrayView1<'_, f64>,
    flex: &SurvivalFlexFamilyScalars,
    slices: &SurvivalBlockSlicesPublic,
    out: &mut [f64],
) {
    assert_eq!(
        coeff_u.len(),
        flex.p_primary,
        "chain_primary_to_joint_row: coeff_u len {} != p_primary {}",
        coeff_u.len(),
        flex.p_primary
    );
    assert_eq!(
        out.len(),
        slices.total,
        "chain_primary_to_joint_row: out len {} != slices.total {}",
        out.len(),
        slices.total
    );

    let cu_q0 = coeff_u[flex.idx_q0];
    let cu_q1 = coeff_u[flex.idx_q1];
    let cu_qd1 = coeff_u[flex.idx_qd1];
    let cu_g = coeff_u[flex.idx_g];

    // Time block: Σ_k (cu_q0 * dq0_time[k] + cu_q1 * dq1_time[k] + cu_qd1 * dqd1_time[k])
    for (k, dst) in out[slices.time.clone()].iter_mut().enumerate() {
        *dst = cu_q0 * q.dq0_time[k] + cu_q1 * q.dq1_time[k] + cu_qd1 * q.dqd1_time[k];
    }

    // Marginal block: same with marginal maps
    for (k, dst) in out[slices.marginal.clone()].iter_mut().enumerate() {
        *dst = cu_q0 * q.dq0_marginal[k] + cu_q1 * q.dq1_marginal[k] + cu_qd1 * q.dqd1_marginal[k];
    }

    // Logslope block: cu_g * logslope_row[k]
    for (k, dst) in out[slices.logslope.clone()].iter_mut().enumerate() {
        *dst = cu_g * logslope_row[k];
    }

    // Score-warp block: identity map through h-slots
    if let Some(sw_range) = slices.score_warp.as_ref() {
        for (local_k, dst) in out[sw_range.clone()].iter_mut().enumerate() {
            let primary_idx = flex.h_start + local_k;
            if primary_idx < flex.p_primary {
                *dst = coeff_u[primary_idx];
            }
        }
    }

    // Link-dev block: identity map through w-slots
    if let Some(ld_range) = slices.link_dev.as_ref() {
        for (local_k, dst) in out[ld_range.clone()].iter_mut().enumerate() {
            let primary_idx = flex.w_start + local_k;
            if primary_idx < flex.p_primary {
                *dst = coeff_u[primary_idx];
            }
        }
    }
}


/// Per-row primary chain maps extracted from `SurvivalMarginalSlopeDynamicRow`.
/// Passed by value (slice references) to `chain_primary_to_joint_row`.
pub struct SurvivalFlexPerRowChain<'a> {
    pub dq0_time: &'a [f64],
    pub dq1_time: &'a [f64],
    pub dqd1_time: &'a [f64],
    pub dq0_marginal: &'a [f64],
    pub dq1_marginal: &'a [f64],
    pub dqd1_marginal: &'a [f64],
}


/// Public mirror of the private `BlockSlices` for use in `chain_primary_to_joint_row`.
pub struct SurvivalBlockSlicesPublic {
    pub time: std::ops::Range<usize>,
    pub marginal: std::ops::Range<usize>,
    pub logslope: std::ops::Range<usize>,
    pub score_warp: Option<std::ops::Range<usize>>,
    pub link_dev: Option<std::ops::Range<usize>>,
    pub total: usize,
}


// ── 6-output flex-chain Jacobian for the logslope block ─────────────────────
//
// n_outputs = 6: η0, η1, log_chi1, log_D1, q1_log_phi_q1, log_qd1.
// When flex state is absent this is a hard error (must use 3-output rigid path
// LogslopeBlockJacobian instead). When flex state is present (SurvivalFlexFamilyScalars
// in family_scalars), uses the full IFT chain.

/// 6-output stacked Jacobian for the **logslope** block using the IFT flex chain.
///
/// Output ordering (for row `i`, outputs stacked as `[0..n], [n..2n], ..., [5n..6n]`):
///
/// - Row 0·n + i: `∂η0[i]/∂β_logslope` (entry probit index)
/// - Row 1·n + i: `∂η1[i]/∂β_logslope` (exit probit index)
/// - Row 2·n + i: `∂log_chi1[i]/∂β_logslope` = `chi_u_exit[i] / chi_exit[i]` chained
/// - Row 3·n + i: `∂log_D1[i]/∂β_logslope`   = `d_u_exit[i] / d_exit[i]` chained
/// - Row 4·n + i: `∂(q1·log_phi_q1)[i]/∂β_logslope` = `q1_i[i] * ∂q1/∂β_logslope`
///               (zero for logslope block since q1 does not depend on β_logslope)
/// - Row 5·n + i: `∂log_qd1[i]/∂β_logslope` = (1/qd1_i[i]) * ∂qd1/∂β_logslope
///               (zero for logslope block since qd1 does not depend on β_logslope)
///
/// When `family_scalars` is `None` (rigid path, no flex), returns `Err` — use
/// `LogslopeBlockJacobian` (3-output) for the rigid hyperbolic path instead.
pub struct LogslopeFlexBlockJacobian {
    /// Logslope design (n × p_logslope); shared with the block spec.
    pub(crate) design: Array2<f64>,
}


impl LogslopeFlexBlockJacobian {
    pub fn new(design: Array2<f64>) -> Self {
        Self { design }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for LogslopeFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "LogslopeFlexBlockJacobian requires SurvivalFlexFamilyScalars in \
                 family_scalars (flex path); for the rigid path use LogslopeBlockJacobian"
                    .to_string()
            })?;

        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if flex.eta_u_entry.nrows() != n || flex.eta_u_exit.nrows() != n {
            return Err(format!(
                "LogslopeFlexBlockJacobian: flex scalars have {} rows but design has {n}",
                flex.eta_u_entry.nrows()
            ));
        }

        // For the logslope block: ∂u_g/∂β_j = design[i,j]; all other primary
        // coords are zero (time, marginal, h, w do not depend on β_logslope).
        // So: ∂output/∂β_j = output_u[idx_g] * design[i,j].
        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let g_idx = flex.idx_g;
            let cu_eta0_g = flex.eta_u_entry[[i, g_idx]];
            let cu_eta1_g = flex.eta_u_exit[[i, g_idx]];
            let chi1 = flex.chi_exit[i];
            let cu_logchi1_g = if chi1 > 0.0 && chi1.is_finite() {
                flex.chi_u_exit[[i, g_idx]] / chi1
            } else {
                0.0
            };
            let d1 = flex.d_exit[i];
            let cu_logd1_g = if d1 > 0.0 && d1.is_finite() {
                flex.d_u_exit[[i, g_idx]] / d1
            } else {
                0.0
            };
            // q1 and qd1 do not depend on β_logslope → rows 4 and 5 are zero.

            for j in 0..p {
                let x_ij = self.design[[i, j]];
                jac[[local_i, j]] = cu_eta0_g * x_ij;
                jac[[chunk + local_i, j]] = cu_eta1_g * x_ij;
                jac[[2 * chunk + local_i, j]] = cu_logchi1_g * x_ij;
                jac[[3 * chunk + local_i, j]] = cu_logd1_g * x_ij;
                // rows 4 and 5 remain zero
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **marginal** block using the IFT flex chain.
///
/// For the marginal block:
/// - ∂u_{q0}/∂β_m = dq0_marginal[i,k]  (row of the chain map)
/// - ∂u_{q1}/∂β_m = dq1_marginal[i,k]
/// - ∂u_{qd1}/∂β_m = dqd1_marginal[i,k]
/// - all other primary slots: zero
///
/// Outputs: same 6 as `LogslopeFlexBlockJacobian`.
pub struct MarginalFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for MarginalFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "MarginalFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.dq0_marginal.nrows();
        let p = flex.dq0_marginal.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, 0)));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        let q0_idx = flex.idx_q0;
        let q1_idx = flex.idx_q1;
        let qd1_idx = flex.idx_qd1;

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];
            let q1 = flex.q1_i[i];
            let qd1 = flex.qd1_i[i];

            for k in 0..p {
                let dq0 = flex.dq0_marginal[[i, k]];
                let dq1 = flex.dq1_marginal[[i, k]];
                let dqd1 = flex.dqd1_marginal[[i, k]];

                // ∂η0/∂β_m[k] = Σ_u eta_u_entry[i,u] * ∂u/∂β_m[k]
                //             = eta_u_entry[q0] * dq0 + eta_u_entry[q1]*dq1 + eta_u_entry[qd1]*dqd1
                // (only q-primary slots are non-zero for the marginal block)
                let deta0 = flex.eta_u_entry[[i, q0_idx]] * dq0
                    + flex.eta_u_entry[[i, q1_idx]] * dq1
                    + flex.eta_u_entry[[i, qd1_idx]] * dqd1;
                let deta1 = flex.eta_u_exit[[i, q0_idx]] * dq0
                    + flex.eta_u_exit[[i, q1_idx]] * dq1
                    + flex.eta_u_exit[[i, qd1_idx]] * dqd1;
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    (flex.chi_u_exit[[i, q0_idx]] * dq0
                        + flex.chi_u_exit[[i, q1_idx]] * dq1
                        + flex.chi_u_exit[[i, qd1_idx]] * dqd1)
                        / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    (flex.d_u_exit[[i, q0_idx]] * dq0
                        + flex.d_u_exit[[i, q1_idx]] * dq1
                        + flex.d_u_exit[[i, qd1_idx]] * dqd1)
                        / d1
                } else {
                    0.0
                };
                // ∂(log_phi(q1))/∂β_m[k] = -q1 * dq1  (d/dq1 of -0.5*q1^2 = -q1)
                let dq1_logphi = -q1 * dq1;
                // ∂log_qd1/∂β_m[k] = dqd1 / qd1
                let dlogqd1 = if qd1 > 0.0 { dqd1 / qd1 } else { 0.0 };

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
                jac[[4 * chunk + local_i, k]] = dq1_logphi;
                jac[[5 * chunk + local_i, k]] = dlogqd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **time** block using the IFT flex chain.
///
/// For the time block:
/// - ∂u_{q0}/∂β_t = dq0_time[i,k]
/// - ∂u_{q1}/∂β_t = dq1_time[i,k]
/// - ∂u_{qd1}/∂β_t = dqd1_time[i,k]
///
/// Outputs: same 6 as above.
pub struct TimeFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for TimeFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "TimeFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.dq0_time.nrows();
        let p = flex.dq0_time.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, 0)));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        let q0_idx = flex.idx_q0;
        let q1_idx = flex.idx_q1;
        let qd1_idx = flex.idx_qd1;

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];
            let q1 = flex.q1_i[i];
            let qd1 = flex.qd1_i[i];

            for k in 0..p {
                let dq0 = flex.dq0_time[[i, k]];
                let dq1 = flex.dq1_time[[i, k]];
                let dqd1 = flex.dqd1_time[[i, k]];

                let deta0 = flex.eta_u_entry[[i, q0_idx]] * dq0
                    + flex.eta_u_entry[[i, q1_idx]] * dq1
                    + flex.eta_u_entry[[i, qd1_idx]] * dqd1;
                let deta1 = flex.eta_u_exit[[i, q0_idx]] * dq0
                    + flex.eta_u_exit[[i, q1_idx]] * dq1
                    + flex.eta_u_exit[[i, qd1_idx]] * dqd1;
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    (flex.chi_u_exit[[i, q0_idx]] * dq0
                        + flex.chi_u_exit[[i, q1_idx]] * dq1
                        + flex.chi_u_exit[[i, qd1_idx]] * dqd1)
                        / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    (flex.d_u_exit[[i, q0_idx]] * dq0
                        + flex.d_u_exit[[i, q1_idx]] * dq1
                        + flex.d_u_exit[[i, qd1_idx]] * dqd1)
                        / d1
                } else {
                    0.0
                };
                // ∂(log_phi(q1))/∂β_t[k] = -q1 * dq1
                let dq1_logphi = -q1 * dq1;
                let dlogqd1 = if qd1 > 0.0 { dqd1 / qd1 } else { 0.0 };

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
                jac[[4 * chunk + local_i, k]] = dq1_logphi;
                jac[[5 * chunk + local_i, k]] = dlogqd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **score_warp** block using the IFT flex chain.
///
/// For score_warp: ∂u_{h_j}/∂β_h[k] = δ_{j,k} (identity map in h-slots).
/// All other primary slots are zero.
///
/// Outputs: same 6 as above.
pub struct ScoreWarpFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for ScoreWarpFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "ScoreWarpFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.score_warp_design.nrows();
        let p = flex.score_warp_design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 || flex.h_len == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, p)));
        }
        if flex.h_len != p {
            return Err(format!(
                "ScoreWarpFlexBlockJacobian: h_len={} but score_warp_design has {} cols",
                flex.h_len, p
            ));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];

            for k in 0..p {
                // h-slot k in primary coordinate space
                let h_idx = flex.h_start + k;
                if h_idx >= flex.p_primary {
                    continue;
                }
                // ∂η0/∂β_h[k] = eta_u_entry[i, h_idx]
                let deta0 = flex.eta_u_entry[[i, h_idx]];
                let deta1 = flex.eta_u_exit[[i, h_idx]];
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    flex.chi_u_exit[[i, h_idx]] / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    flex.d_u_exit[[i, h_idx]] / d1
                } else {
                    0.0
                };
                // rows 4 and 5 (q1_logphi, log_qd1) are zero: q1, qd1 do not depend on β_h

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


/// 6-output stacked Jacobian for the **link_dev** block using the IFT flex chain.
///
/// For link_dev: ∂u_{w_j}/∂β_w[k] = δ_{j,k} (identity map in w-slots).
///
/// Outputs: same 6 as above.
pub struct LinkDevFlexBlockJacobian;


impl crate::custom_family::BlockEffectiveJacobian for LinkDevFlexBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let flex: &SurvivalFlexFamilyScalars = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<SurvivalFlexFamilyScalars>())
            .ok_or_else(|| {
                "LinkDevFlexBlockJacobian requires SurvivalFlexFamilyScalars in family_scalars"
                    .to_string()
            })?;

        let n = flex.link_dev_design.nrows();
        let p = flex.link_dev_design.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if p == 0 || flex.w_len == 0 {
            return Ok(Array2::<f64>::zeros((6 * chunk, p)));
        }
        if flex.w_len != p {
            return Err(format!(
                "LinkDevFlexBlockJacobian: w_len={} but link_dev_design has {} cols",
                flex.w_len, p
            ));
        }

        const N_OUT: usize = 6;
        let mut jac = Array2::<f64>::zeros((N_OUT * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let chi1 = flex.chi_exit[i];
            let d1 = flex.d_exit[i];

            for k in 0..p {
                let w_idx = flex.w_start + k;
                if w_idx >= flex.p_primary {
                    continue;
                }
                let deta0 = flex.eta_u_entry[[i, w_idx]];
                let deta1 = flex.eta_u_exit[[i, w_idx]];
                let dlogchi1 = if chi1 > 0.0 && chi1.is_finite() {
                    flex.chi_u_exit[[i, w_idx]] / chi1
                } else {
                    0.0
                };
                let dlogd1 = if d1 > 0.0 && d1.is_finite() {
                    flex.d_u_exit[[i, w_idx]] / d1
                } else {
                    0.0
                };

                jac[[local_i, k]] = deta0;
                jac[[chunk + local_i, k]] = deta1;
                jac[[2 * chunk + local_i, k]] = dlogchi1;
                jac[[3 * chunk + local_i, k]] = dlogd1;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        6
    }
}


// ── Timewiggle-active Jacobians ───────────────────────────────────────
//
// When timewiggle is active, (q0, q1, qd1) are nonlinear functions of
// (β_time, β_marginal) through the composition:
//
//   h0 = X_entry_base[i] · β_t_base + offset_entry[i] + M[i] · β_m
//   q0 = h0 + B(h0) · β_tw           (B = monotone wiggle basis)
//
// and analogously for q1 and qd1.  The chain rule gives:
//
//   ∂q0/∂β_t[j < p_base] = (1 + B'(h0)·β_tw) · X_entry[i,j]
//                         = dq_dq0(h0) · X_entry[i,j]
//   ∂q0/∂β_t[p_base + k] = B_k(h0)
//   ∂q0/∂β_m[j]          = dq_dq0(h0) · M[i,j]
//
// Since η_r = c · q_r + … and ∂η_r/∂β_block = c · ∂q_r/∂β_block,
// the stacked Jacobian for each block is:
//
//   J[i,       j] = c_i · ∂q0/∂β_block[j]
//   J[n + i,   j] = c_i · ∂q1/∂β_block[j]
//   J[2*n + i, j] = c_i · ∂qd1/∂β_block[j]
//
// where c_i = sqrt(1 + (s · g_i)²) and g_i = G[i] · β_g.
//
// At β = 0: dq_dq0 = 1, d²q/dh² = 0, c_i = 1, so both timewiggle
// callbacks reduce to the rigid-path `TimeBlockJacobian` /
// `MarginalBlockJacobian` values.
//
// Joint β layout (same for both callbacks):
//   [β_t (p_time) | β_m (p_m) | β_g (p_g) | …]
//
// p_time = p_base + p_tw where p_tw = time_wiggle_ncols.

/// n_outputs = 3 stacked Jacobian for the **time** block when timewiggle
/// is active.  Computes `c_i` from the embedded logslope design and
/// joint β, so no `family_scalars` are required.
pub struct SmsTimewiggleTimeJacobian {
    pub(crate) design_entry: Arc<Array2<f64>>,
    pub(crate) design_exit: Arc<Array2<f64>>,
    pub(crate) design_deriv: Arc<Array2<f64>>,
    pub(crate) design_marginal: Arc<Array2<f64>>,
    pub(crate) design_logslope: Arc<Array2<f64>>,
    pub(crate) offset_entry: Arc<Array1<f64>>,
    pub(crate) offset_exit: Arc<Array1<f64>>,
    pub(crate) offset_deriv: Arc<Array1<f64>>,
    /// Fixed marginal-predictor offset. The full marginal predictor entering
    /// the entry/exit channels is `design_marginal·β_m + marginal_offset`
    /// (see `row_dynamic_q_values`); this is the β-independent part.
    pub(crate) marginal_offset: Arc<Array1<f64>>,
    pub(crate) time_wiggle_knots: Array1<f64>,
    pub(crate) time_wiggle_degree: usize,
    /// Full time block width (= design_entry.ncols()).
    pub(crate) p_time: usize,
    /// Wiggle tail width.
    pub(crate) p_tw: usize,
    /// Marginal block width (for joint β parsing).
    pub(crate) p_m: usize,
    /// Logslope block width (for joint β parsing).
    pub(crate) p_g: usize,
    /// Probit frailty scale s.
    pub(crate) probit_scale: f64,
}


impl SmsTimewiggleTimeJacobian {
    /// Construct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        design_entry: Arc<Array2<f64>>,
        design_exit: Arc<Array2<f64>>,
        design_deriv: Arc<Array2<f64>>,
        design_marginal: Arc<Array2<f64>>,
        design_logslope: Arc<Array2<f64>>,
        offset_entry: Arc<Array1<f64>>,
        offset_exit: Arc<Array1<f64>>,
        offset_deriv: Arc<Array1<f64>>,
        marginal_offset: Arc<Array1<f64>>,
        time_wiggle_knots: Array1<f64>,
        time_wiggle_degree: usize,
        p_tw: usize,
        p_m: usize,
        p_g: usize,
        probit_scale: f64,
    ) -> Self {
        let p_time = design_entry.ncols();
        Self {
            design_entry,
            design_exit,
            design_deriv,
            design_marginal,
            design_logslope,
            offset_entry,
            offset_exit,
            offset_deriv,
            marginal_offset,
            time_wiggle_knots,
            time_wiggle_degree,
            p_time,
            p_tw,
            p_m,
            p_g,
            probit_scale,
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for SmsTimewiggleTimeJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design_entry.nrows();
        let p = self.p_time;
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        let p_base = p.saturating_sub(self.p_tw);

        let beta = state.beta;
        // β_t = joint β[0 .. p_time]
        let beta_t = if beta.len() >= p { &beta[..p] } else { beta };
        let beta_t_base = &beta_t[..p_base.min(beta_t.len())];
        // β_tw must always be a length-`p_tw` vector. The timewiggle block
        // exists whenever `self.p_tw > 0`, independent of how many coefficients
        // the caller supplied: the identifiability canonicaliser calls this at
        // the β=0 linearisation point with `beta = &[]` (see
        // `BlockJacobianAsRowOp::from_callback`), so inferring "no wiggle block"
        // from an empty slice — the old behaviour — wrongly drove `beta_tw`
        // empty, made `sms_tw_first_order_geom` return `None`, and zeroed the
        // wiggle tail columns. That made the time block look structurally
        // aliased ("block 0 fully aliased") even though ∂q/∂β_tw[j] = B_j(h) ≠ 0
        // at β=0. Zero-pad to `self.p_tw` so the basis is always evaluated.
        let zero_tw: Vec<f64>;
        let beta_tw: &[f64] = if beta_t.len() >= p_base + self.p_tw {
            &beta_t[p_base..p_base + self.p_tw]
        } else {
            zero_tw = vec![0.0; self.p_tw];
            &zero_tw
        };
        // β_m = joint β[p_time .. p_time + p_m]
        let beta_m = {
            let s = p;
            let e = (s + self.p_m).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };
        // β_g = joint β[p_time + p_m .. p_time + p_m + p_g]
        let beta_g = {
            let s = p + self.p_m;
            let e = (s + self.p_g).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };

        let sc = self.probit_scale;
        let knots = &self.time_wiggle_knots;
        let degree = self.time_wiggle_degree;

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            // c_i computed directly from logslope design and joint β_g.
            let g_i: f64 = beta_g
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < self.design_logslope.ncols())
                .map(|(j, &b)| self.design_logslope[[i, j]] * b)
                .sum();
            let c_i = (1.0_f64 + (sc * g_i).powi(2)).sqrt();

            // Base marginal η contribution.
            let eta_m: f64 = beta_m
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < self.design_marginal.ncols())
                .map(|(j, &b)| self.design_marginal[[i, j]] * b)
                .sum();

            // The marginal predictor (coefficient part `eta_m` plus the fixed
            // `marginal_offset`) enters BOTH entry and exit channels but NOT
            // the derivative channel — see `row_dynamic_q_values`.
            let h0: f64 = self.offset_entry[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_entry.ncols()))
                    .map(|j| self.design_entry[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let h1: f64 = self.offset_exit[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_exit.ncols()))
                    .map(|j| self.design_exit[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let d_raw: f64 = self.offset_deriv[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_deriv.ncols()))
                    .map(|j| self.design_deriv[[i, j]] * beta_t_base[j])
                    .sum::<f64>();

            let beta_tw_view = ndarray::ArrayView1::from(beta_tw);
            let eg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h0][..]),
                beta_tw_view,
                knots,
                degree,
            )?;
            let xg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h1][..]),
                beta_tw_view,
                knots,
                degree,
            )?;

            let (entry_dq, exit_dq, exit_d2q, entry_basis, exit_basis, exit_basis_d1) =
                match (eg, xg) {
                    (Some(eg), Some(xg)) => (
                        eg.dq_dq0[0],
                        xg.dq_dq0[0],
                        xg.d2q_dq02[0],
                        Some(eg.basis),
                        Some(xg.basis),
                        Some(xg.basis_d1),
                    ),
                    _ => (1.0_f64, 1.0_f64, 0.0_f64, None, None, None),
                };

            // Base columns j < p_base.
            for j in 0..p_base.min(self.design_entry.ncols()) {
                let xe = self.design_entry[[i, j]];
                let xx = self.design_exit[[i, j]];
                let xd = self.design_deriv[[i, j]];
                jac[[local_i, j]] = c_i * entry_dq * xe;
                jac[[chunk + local_i, j]] = c_i * exit_dq * xx;
                jac[[2 * chunk + local_i, j]] = c_i * (exit_d2q * d_raw * xx + exit_dq * xd);
            }

            // Wiggle tail columns.
            for local_idx in 0..self.p_tw {
                let col = p_base + local_idx;
                let b0 = entry_basis.as_ref().map_or(0.0, |b| b[[0, local_idx]]);
                let b1 = exit_basis.as_ref().map_or(0.0, |b| b[[0, local_idx]]);
                let bd1 = exit_basis_d1.as_ref().map_or(0.0, |b| b[[0, local_idx]]);
                jac[[local_i, col]] = c_i * b0;
                jac[[chunk + local_i, col]] = c_i * b1;
                jac[[2 * chunk + local_i, col]] = c_i * bd1 * d_raw;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// n_outputs = 3 stacked Jacobian for the **marginal** block when timewiggle
/// is active.
pub struct SmsTimewiggleMarginalJacobian {
    pub(crate) design_entry: Arc<Array2<f64>>,
    pub(crate) design_exit: Arc<Array2<f64>>,
    pub(crate) design_deriv: Arc<Array2<f64>>,
    pub(crate) design_marginal: Arc<Array2<f64>>,
    pub(crate) design_logslope: Arc<Array2<f64>>,
    pub(crate) offset_entry: Arc<Array1<f64>>,
    pub(crate) offset_exit: Arc<Array1<f64>>,
    pub(crate) offset_deriv: Arc<Array1<f64>>,
    /// Fixed marginal-predictor offset (β-independent part of the marginal
    /// predictor entering the entry/exit channels; see `row_dynamic_q_values`).
    pub(crate) marginal_offset: Arc<Array1<f64>>,
    pub(crate) time_wiggle_knots: Array1<f64>,
    pub(crate) time_wiggle_degree: usize,
    pub(crate) p_time: usize,
    pub(crate) p_tw: usize,
    pub(crate) p_g: usize,
    pub(crate) probit_scale: f64,
}


impl SmsTimewiggleMarginalJacobian {
    /// Construct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        design_entry: Arc<Array2<f64>>,
        design_exit: Arc<Array2<f64>>,
        design_deriv: Arc<Array2<f64>>,
        design_marginal: Arc<Array2<f64>>,
        design_logslope: Arc<Array2<f64>>,
        offset_entry: Arc<Array1<f64>>,
        offset_exit: Arc<Array1<f64>>,
        offset_deriv: Arc<Array1<f64>>,
        marginal_offset: Arc<Array1<f64>>,
        time_wiggle_knots: Array1<f64>,
        time_wiggle_degree: usize,
        p_time: usize,
        p_tw: usize,
        p_g: usize,
        probit_scale: f64,
    ) -> Self {
        Self {
            design_entry,
            design_exit,
            design_deriv,
            design_marginal,
            design_logslope,
            offset_entry,
            offset_exit,
            offset_deriv,
            marginal_offset,
            time_wiggle_knots,
            time_wiggle_degree,
            p_time,
            p_tw,
            p_g,
            probit_scale,
        }
    }
}


impl crate::custom_family::BlockEffectiveJacobian for SmsTimewiggleMarginalJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design_marginal.nrows();
        let p_m = self.design_marginal.ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        let p_t = self.p_time;
        let p_base = p_t.saturating_sub(self.p_tw);

        let beta = state.beta;
        let beta_t = if beta.len() >= p_t {
            &beta[..p_t]
        } else {
            beta
        };
        let beta_t_base = &beta_t[..p_base.min(beta_t.len())];
        let beta_tw = if beta_t.len() > p_base {
            &beta_t[p_base..]
        } else {
            &[][..]
        };
        let beta_m = {
            let s = p_t;
            let e = (s + p_m).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };
        let beta_g = {
            let s = p_t + p_m;
            let e = (s + self.p_g).min(beta.len());
            if e > s { &beta[s..e] } else { &[][..] }
        };

        let sc = self.probit_scale;
        let knots = &self.time_wiggle_knots;
        let degree = self.time_wiggle_degree;

        let mut jac = Array2::<f64>::zeros((3 * chunk, p_m));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let g_i: f64 = beta_g
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < self.design_logslope.ncols())
                .map(|(j, &b)| self.design_logslope[[i, j]] * b)
                .sum();
            let c_i = (1.0_f64 + (sc * g_i).powi(2)).sqrt();

            let eta_m: f64 = beta_m
                .iter()
                .enumerate()
                .filter(|&(j, _)| j < p_m)
                .map(|(j, &b)| self.design_marginal[[i, j]] * b)
                .sum();

            // Marginal predictor (eta_m + fixed marginal_offset) enters entry
            // and exit channels alike (see `row_dynamic_q_values`).
            let h0: f64 = self.offset_entry[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_entry.ncols()))
                    .map(|j| self.design_entry[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let h1: f64 = self.offset_exit[i]
                + eta_m
                + self.marginal_offset[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_exit.ncols()))
                    .map(|j| self.design_exit[[i, j]] * beta_t_base[j])
                    .sum::<f64>();
            let d_raw: f64 = self.offset_deriv[i]
                + (0..p_base.min(beta_t_base.len()).min(self.design_deriv.ncols()))
                    .map(|j| self.design_deriv[[i, j]] * beta_t_base[j])
                    .sum::<f64>();

            let beta_tw_view = ndarray::ArrayView1::from(beta_tw);
            let eg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h0][..]),
                beta_tw_view,
                knots,
                degree,
            )?;
            let xg = sms_tw_first_order_geom(
                ndarray::ArrayView1::from(&[h1][..]),
                beta_tw_view,
                knots,
                degree,
            )?;

            let (entry_dq, exit_dq, exit_d2q) = match (eg, xg) {
                (Some(eg), Some(xg)) => (eg.dq_dq0[0], xg.dq_dq0[0], xg.d2q_dq02[0]),
                _ => (1.0_f64, 1.0_f64, 0.0_f64),
            };

            for j in 0..p_m {
                let m_ij = self.design_marginal[[i, j]];
                jac[[local_i, j]] = c_i * entry_dq * m_ij;
                jac[[chunk + local_i, j]] = c_i * exit_dq * m_ij;
                jac[[2 * chunk + local_i, j]] = c_i * exit_d2q * d_raw * m_ij;
            }
        }
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        3
    }
}


/// Compute timewiggle first-order geometry at a single evaluation point `h0`.
///
/// Returns `Ok(None)` when `beta_tw` is empty (no active wiggle columns).
/// This is a free-function mirror of
/// `SurvivalMarginalSlopeFamily::time_wiggle_first_order_geometry` for use in
/// `BlockEffectiveJacobian` impls that do not hold a family reference.
pub(crate) fn sms_tw_first_order_geom(
    h0: ndarray::ArrayView1<'_, f64>,
    beta_tw: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Option<SurvivalTimeWiggleFirstOrderGeometry>, String> {
    if beta_tw.is_empty() {
        return Ok(None);
    }
    let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
    let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
    let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
    if basis.ncols() != beta_tw.len()
        || basis_d1.ncols() != beta_tw.len()
        || basis_d2.ncols() != beta_tw.len()
    {
        return Err(format!(
            "sms_tw_first_order_geom: basis/beta_tw width mismatch \
             B/B'/B''={}/{}/{} beta_tw={}",
            basis.ncols(),
            basis_d1.ncols(),
            basis_d2.ncols(),
            beta_tw.len(),
        ));
    }
    let dq_dq0 = fast_av(&basis_d1, &beta_tw) + 1.0;
    let d2q_dq02 = fast_av(&basis_d2, &beta_tw);
    Ok(Some(SurvivalTimeWiggleFirstOrderGeometry {
        basis,
        basis_d1,
        basis_d2,
        dq_dq0,
        d2q_dq02,
    }))
}

