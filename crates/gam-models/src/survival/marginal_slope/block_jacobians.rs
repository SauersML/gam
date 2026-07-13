//! Per-block effective Jacobians: the family scalars and the rigid / flex /
//! time-wiggle `BlockEffectiveJacobian` implementations (time, marginal,
//! logslope, score-warp, link-dev) plus the primary->joint row chain.

use super::*;
use gam_math::jet_scalar::SymmetricQuadraticCoefficients;

/// Per-row scalars for survival marginal-slope Jacobian evaluation at a given β.
///
/// Fields:
/// - `q0_i`: entry-time probit argument (per-row, length n)
/// - `q1_i`: exit-time probit argument (per-row, length n)
/// - `qd1_i`: derivative probit argument (per-row, length n)
/// - `slopes`: per-row physical log-slope vector `(n × K)`
/// - `c_i`: `sqrt(1 + s² g_iᵀΣg_i)` (per-row, length n)
/// - `timewiggle_primary_rows`: canonical channel-major q-gradient rows when
///   the nonlinear timewiggle map is active
/// - `s`: probit scale (scalar, = `probit_frailty_scale()`)
pub struct SurvivalMarginalSlopeFamilyScalars {
    pub(crate) q0_i: Vec<f64>,
    pub(crate) q1_i: Vec<f64>,
    pub(crate) qd1_i: Vec<f64>,
    pub(crate) slopes: Array2<f64>,
    pub(crate) c_i: Vec<f64>,
    /// Exact channel-major `(3n × p_time, 3n × p_marginal)` q-Jacobian rows
    /// when timewiggle is active. These come from the family's canonical
    /// `row_dynamic_q_gradient`; callbacks must not reconstruct joint state
    /// from a block-local audit coefficient slice.
    pub(crate) timewiggle_primary_rows: Option<(Array2<f64>, Array2<f64>)>,
    pub(crate) s: f64,
}

fn scaled_channel_major_rows(
    source: &Array2<f64>,
    c_i: &[f64],
    n: usize,
    columns: usize,
    rows: std::ops::Range<usize>,
    label: &str,
) -> Result<Array2<f64>, String> {
    if c_i.len() != n {
        return Err(format!(
            "{label} c length {} does not match n={n}",
            c_i.len(),
        ));
    }
    if source.dim() != (3 * n, columns) {
        return Err(format!(
            "{label} primary rows are {}x{}, expected {}x{columns}",
            source.nrows(),
            source.ncols(),
            3 * n,
        ));
    }
    let chunk = rows.end - rows.start;
    let mut jacobian = Array2::<f64>::zeros((3 * chunk, columns));
    for row in rows.clone() {
        let local_row = row - rows.start;
        for channel in 0..3 {
            for column in 0..columns {
                jacobian[[channel * chunk + local_row, column]] =
                    c_i[row] * source[[channel * n + row, column]];
            }
        }
    }
    Ok(jacobian)
}

impl SurvivalMarginalSlopeFamilyScalars {
    /// Construct the exact current primary geometry. The score covariance is
    /// part of the construction contract, so `c_i` cannot drift away from the
    /// vector likelihood's `g_iᵀΣg_i` scale.
    pub fn new(
        q0_i: Vec<f64>,
        q1_i: Vec<f64>,
        qd1_i: Vec<f64>,
        slopes: Array2<f64>,
        timewiggle_primary_rows: Option<(Array2<f64>, Array2<f64>)>,
        s: f64,
        covariance: &MarginalSlopeCovariance,
    ) -> Result<Self, String> {
        if !(s.is_finite() && s > 0.0) {
            return Err(format!(
                "survival marginal-slope family scalars require a positive finite probit scale, got {s}"
            ));
        }
        covariance.validate("survival marginal-slope family scalars covariance")?;
        let n = q0_i.len();
        if q1_i.len() != n
            || qd1_i.len() != n
            || slopes.nrows() != n
            || slopes.ncols() != covariance.dim()
        {
            return Err(format!(
                "survival marginal-slope family scalar dimensions disagree: q0={n}, q1={}, qd1={}, slopes={}x{}, covariance K={}",
                q1_i.len(),
                qd1_i.len(),
                slopes.nrows(),
                slopes.ncols(),
                covariance.dim(),
            ));
        }
        if q0_i
            .iter()
            .chain(q1_i.iter())
            .chain(qd1_i.iter())
            .chain(slopes.iter())
            .any(|value| !value.is_finite())
        {
            return Err(
                "survival marginal-slope family scalars contain a non-finite primary value"
                    .to_string(),
            );
        }
        if let Some((time_rows, marginal_rows)) = &timewiggle_primary_rows {
            if time_rows.nrows() != 3 * n || marginal_rows.nrows() != 3 * n {
                return Err(format!(
                    "survival marginal-slope timewiggle primary rows must have 3n={} rows, got time={} marginal={}",
                    3 * n,
                    time_rows.nrows(),
                    marginal_rows.nrows(),
                ));
            }
            if time_rows
                .iter()
                .chain(marginal_rows.iter())
                .any(|value| !value.is_finite())
            {
                return Err(
                    "survival marginal-slope timewiggle primary rows contain a non-finite value"
                        .to_string(),
                );
            }
        }
        let mut c_i = Vec::with_capacity(n);
        for row in slopes.rows() {
            let variance = covariance.quadratic_form(row.as_slice().ok_or_else(|| {
                "survival marginal-slope slope row is not contiguous".to_string()
            })?)?;
            c_i.push((1.0 + s * s * variance).sqrt());
        }
        Ok(Self {
            q0_i,
            q1_i,
            qd1_i,
            slopes,
            c_i,
            timewiggle_primary_rows,
            s,
        })
    }
}

/// Three-output effective Jacobian for a shared or per-score log-slope layout.
///
/// For physical slopes `g ∈ R^K`, full-width channel rows `G_k`, covariance
/// `Σ`, and `c = sqrt(1 + s² gᵀΣg)`, this emits
/// `q0·dc + dlinear`, `q1·dc + dlinear`, and `qd1·dc`, where
/// `dc = s²(Σg)ᵀG/c` and `dlinear = s zᵀG`.
pub struct LogslopeBlockJacobian {
    pub(crate) layout: LogslopeLayout,
    pub(crate) z: Arc<Array2<f64>>,
    pub(crate) covariance: MarginalSlopeCovariance,
}

impl LogslopeBlockJacobian {
    pub(crate) fn new(
        layout: LogslopeLayout,
        z: Arc<Array2<f64>>,
        covariance: MarginalSlopeCovariance,
    ) -> Result<Self, String> {
        layout.validate_for(z.ncols())?;
        covariance.validate("logslope effective Jacobian covariance")?;
        if covariance.dim() != z.ncols() {
            return Err(format!(
                "logslope effective Jacobian covariance dimension {} does not match score dimension {}",
                covariance.dim(),
                z.ncols(),
            ));
        }
        Ok(Self {
            layout,
            z,
            covariance,
        })
    }
}

impl crate::custom_family::BlockEffectiveJacobian for LogslopeBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &crate::custom_family::FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.layout.coefficient_design().nrows();
        let p = self.layout.coefficient_design().ncols();
        let rows = rows.start.min(n)..rows.end.min(n);
        let chunk = rows.end - rows.start;
        if self.z.nrows() != n {
            return Err(format!(
                "logslope effective Jacobian score rows {} do not match layout rows {n}",
                self.z.nrows(),
            ));
        }
        let s = state.probit_frailty_scale;
        if !(s.is_finite() && s > 0.0) {
            return Err(format!(
                "logslope effective Jacobian requires a positive finite probit scale, got {s}"
            ));
        }
        if !state.beta.is_empty() && state.beta.len() != p {
            return Err(format!(
                "logslope effective Jacobian beta length {} does not match width {p}",
                state.beta.len(),
            ));
        }
        let structural_zero = state.beta.is_empty().then(|| Array1::<f64>::zeros(p));
        let beta = match structural_zero.as_ref() {
            Some(zero) => zero.view(),
            None => ArrayView1::from(state.beta),
        };
        let scalars = state
            .family_scalars
            .as_ref()
            .map(|value| {
                value
                    .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
                    .ok_or_else(|| {
                        "logslope effective Jacobian received the wrong family-scalar type"
                            .to_string()
                    })
            })
            .transpose()?;
        if let Some(scalars) = scalars {
            for (name, values) in [
                ("q0", &scalars.q0_i),
                ("q1", &scalars.q1_i),
                ("qd1", &scalars.qd1_i),
            ] {
                if values.len() != n {
                    return Err(format!(
                        "logslope effective Jacobian {name} scalar length {} does not match n={n}",
                        values.len(),
                    ));
                }
            }
        }
        let mut jac = Array2::<f64>::zeros((3 * chunk, p));
        let mut workspace = self.layout.row_workspace(self.z.ncols())?;
        let mut sigma_g = vec![0.0; self.z.ncols()];
        for i in rows.clone() {
            let local_i = i - rows.start;
            self.layout
                .fill_callback_row(i, beta.view(), &mut workspace)?;
            let values = workspace.values();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "logslope effective Jacobian row {i} contains a non-finite physical slope"
                ));
            }
            self.covariance.multiply(values, &mut sigma_g);
            let variance = values
                .iter()
                .zip(sigma_g.iter())
                .map(|(value, sigma_value)| value * sigma_value)
                .sum::<f64>();
            if !(variance.is_finite() && variance >= COVARIANCE_QUADRATIC_FORM_PSD_TOL) {
                return Err(format!(
                    "logslope effective Jacobian covariance quadratic form must be non-negative, got {variance} at row {i}"
                ));
            }
            let variance = variance.max(0.0);
            let c = (1.0 + s * s * variance).sqrt();
            let (q0, q1, qd1) = match scalars {
                Some(values) => (values.q0_i[i], values.q1_i[i], values.qd1_i[i]),
                // The pre-fit structural audit explicitly linearizes q at the
                // origin when no family state has yet been evaluated.
                None => (0.0, 0.0, 0.0),
            };
            let channel_rows = workspace.channel_rows();
            for j in 0..p {
                let mut covariance_direction = 0.0;
                let mut linear_direction = 0.0;
                for coordinate in 0..self.z.ncols() {
                    let channel_row = channel_rows[[coordinate, j]];
                    covariance_direction += sigma_g[coordinate] * channel_row;
                    linear_direction += self.z[[i, coordinate]] * channel_row;
                }
                let dc = s * s * covariance_direction / c;
                let dlinear = s * linear_direction;
                jac[[local_i, j]] = q0 * dc + dlinear;
                jac[[chunk + local_i, j]] = q1 * dc + dlinear;
                jac[[2 * chunk + local_i, j]] = qd1 * dc;
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
    /// owner rather than copied for the callback lifetime.
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
    // `Arc`-shared with their owners.
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
/// is active. Current `c_i` values come from the family-owned vector
/// log-slope state, so nonzero β requires `family_scalars`.
pub struct SmsTimewiggleTimeJacobian {
    pub(crate) design_entry: Arc<Array2<f64>>,
    pub(crate) design_exit: Arc<Array2<f64>>,
    pub(crate) design_deriv: Arc<Array2<f64>>,
    pub(crate) design_marginal: Arc<Array2<f64>>,
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
}

impl SmsTimewiggleTimeJacobian {
    /// Construct.
    pub fn new(
        design_entry: Arc<Array2<f64>>,
        design_exit: Arc<Array2<f64>>,
        design_deriv: Arc<Array2<f64>>,
        design_marginal: Arc<Array2<f64>>,
        offset_entry: Arc<Array1<f64>>,
        offset_exit: Arc<Array1<f64>>,
        offset_deriv: Arc<Array1<f64>>,
        marginal_offset: Arc<Array1<f64>>,
        time_wiggle_knots: Array1<f64>,
        time_wiggle_degree: usize,
        p_tw: usize,
        p_m: usize,
    ) -> Self {
        let p_time = design_entry.ncols();
        Self {
            design_entry,
            design_exit,
            design_deriv,
            design_marginal,
            offset_entry,
            offset_exit,
            offset_deriv,
            marginal_offset,
            time_wiggle_knots,
            time_wiggle_degree,
            p_time,
            p_tw,
            p_m,
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

        if let Some(values) = state
            .family_scalars
            .as_ref()
            .map(|value| {
                value
                    .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
                    .ok_or_else(|| {
                        "timewiggle time Jacobian received the wrong family-scalar type".to_string()
                    })
            })
            .transpose()?
        {
            let (time_rows, _) = values.timewiggle_primary_rows.as_ref().ok_or_else(|| {
                "timewiggle time Jacobian requires canonical current q-gradient rows".to_string()
            })?;
            return scaled_channel_major_rows(
                time_rows,
                &values.c_i,
                n,
                p,
                rows,
                "timewiggle time Jacobian",
            );
        }

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
        if beta.iter().any(|&value| value != 0.0) {
            return Err(
                "timewiggle time Jacobian requires current survival marginal-slope family scalars at nonzero beta"
                    .to_string(),
            );
        }
        let knots = &self.time_wiggle_knots;
        let degree = self.time_wiggle_degree;

        let mut jac = Array2::<f64>::zeros((3 * chunk, p));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let c_i = 1.0;

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

    fn locks_raw_width_reduction(&self) -> bool {
        // The time-wiggle Jacobian recomputes a fixed `p_tw`-column monotone
        // wiggle basis on every evaluation and the family's likelihood reads
        // the raw-width entry/exit/derivative designs (asserting
        // `beta.len() == design_derivative_exit.ncols()`). A canonicaliser
        // column-reduction of this block produces a reduced β the raw-width
        // family cannot consume, so it must be kept at full raw width.
        true
    }
}

/// n_outputs = 3 stacked Jacobian for the **marginal** block when timewiggle
/// is active.
pub struct SmsTimewiggleMarginalJacobian {
    pub(crate) design_entry: Arc<Array2<f64>>,
    pub(crate) design_exit: Arc<Array2<f64>>,
    pub(crate) design_deriv: Arc<Array2<f64>>,
    pub(crate) design_marginal: Arc<Array2<f64>>,
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
}

impl SmsTimewiggleMarginalJacobian {
    /// Construct.
    pub fn new(
        design_entry: Arc<Array2<f64>>,
        design_exit: Arc<Array2<f64>>,
        design_deriv: Arc<Array2<f64>>,
        design_marginal: Arc<Array2<f64>>,
        offset_entry: Arc<Array1<f64>>,
        offset_exit: Arc<Array1<f64>>,
        offset_deriv: Arc<Array1<f64>>,
        marginal_offset: Arc<Array1<f64>>,
        time_wiggle_knots: Array1<f64>,
        time_wiggle_degree: usize,
        p_time: usize,
        p_tw: usize,
    ) -> Self {
        Self {
            design_entry,
            design_exit,
            design_deriv,
            design_marginal,
            offset_entry,
            offset_exit,
            offset_deriv,
            marginal_offset,
            time_wiggle_knots,
            time_wiggle_degree,
            p_time,
            p_tw,
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

        if let Some(values) = state
            .family_scalars
            .as_ref()
            .map(|value| {
                value
                    .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
                    .ok_or_else(|| {
                        "timewiggle marginal Jacobian received the wrong family-scalar type"
                            .to_string()
                    })
            })
            .transpose()?
        {
            let (_, marginal_rows) = values.timewiggle_primary_rows.as_ref().ok_or_else(|| {
                "timewiggle marginal Jacobian requires canonical current q-gradient rows"
                    .to_string()
            })?;
            return scaled_channel_major_rows(
                marginal_rows,
                &values.c_i,
                n,
                p_m,
                rows,
                "timewiggle marginal Jacobian",
            );
        }

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
        if beta.iter().any(|&value| value != 0.0) {
            return Err(
                "timewiggle marginal Jacobian requires current survival marginal-slope family scalars at nonzero beta"
                    .to_string(),
            );
        }
        let knots = &self.time_wiggle_knots;
        let degree = self.time_wiggle_degree;

        let mut jac = Array2::<f64>::zeros((3 * chunk, p_m));

        for i in rows.clone() {
            let local_i = i - rows.start;
            let c_i = 1.0;

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn per_score_logslope_callback_uses_offsets_covariance_and_full_width_rows() {
        let topology = LogslopeTopology::per_score(vec![0..1, 1..2], 2).unwrap();
        let layout = topology
            .materialize_identity(DesignMatrix::from(array![[2.0, 3.0]]), &array![0.4])
            .unwrap();
        let covariance = MarginalSlopeCovariance::Diagonal(array![2.0, 0.5]);
        let callback =
            LogslopeBlockJacobian::new(layout, Arc::new(array![[1.5, -0.5]]), covariance.clone())
                .unwrap();
        assert_eq!(
            crate::custom_family::BlockEffectiveJacobian::n_outputs(&callback),
            3
        );
        let scalars: Arc<dyn std::any::Any + Send + Sync> = Arc::new(
            SurvivalMarginalSlopeFamilyScalars::new(
                vec![1.1],
                vec![1.3],
                vec![0.7],
                array![[0.0, 0.0]],
                None,
                0.8,
                &covariance,
            )
            .unwrap(),
        );
        let beta = [0.1, 0.2];
        let state = crate::custom_family::FamilyLinearizationState {
            beta: &beta,
            family_scalars: Some(scalars),
            channel_hessian: None,
            probit_frailty_scale: 0.8,
        };
        let jacobian = crate::custom_family::BlockEffectiveJacobian::effective_jacobian_rows(
            &callback,
            &state,
            0..1,
        )
        .unwrap();

        // g=(2*.1+.4, 3*.2+.4)=(.6,1), Sigma*g=(1.2,.5).
        let c = (1.0_f64 + 0.8_f64.powi(2) * 1.22).sqrt();
        let dc0 = 0.8_f64.powi(2) * (1.2 * 2.0) / c;
        let dc1 = 0.8_f64.powi(2) * (0.5 * 3.0) / c;
        let linear0 = 0.8 * 1.5 * 2.0;
        let linear1 = 0.8 * -0.5 * 3.0;
        let expected = array![
            [1.1 * dc0 + linear0, 1.1 * dc1 + linear1],
            [1.3 * dc0 + linear0, 1.3 * dc1 + linear1],
            [0.7 * dc0, 0.7 * dc1],
        ];
        for (actual, expected) in jacobian.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn timewiggle_current_audit_uses_family_q_gradient_rows_932() {
        let n = 2;
        let p_time = 2;
        let p_marginal = 1;
        let time_rows = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ];
        let marginal_rows = array![[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]];
        let covariance = MarginalSlopeCovariance::Diagonal(array![2.0]);
        let scalars = SurvivalMarginalSlopeFamilyScalars::new(
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            array![[0.5], [-0.25]],
            Some((time_rows.clone(), marginal_rows.clone())),
            0.8,
            &covariance,
        )
        .unwrap();
        let c_i = scalars.c_i.clone();
        let scalars: Arc<dyn std::any::Any + Send + Sync> = Arc::new(scalars);

        let design_time = Arc::new(Array2::<f64>::zeros((n, p_time)));
        let design_marginal = Arc::new(Array2::<f64>::zeros((n, p_marginal)));
        let offset = Arc::new(Array1::<f64>::zeros(n));
        let knots = array![0.0, 0.0, 1.0, 1.0];
        let time_callback = SmsTimewiggleTimeJacobian::new(
            design_time.clone(),
            design_time.clone(),
            design_time.clone(),
            design_marginal.clone(),
            offset.clone(),
            offset.clone(),
            offset.clone(),
            offset.clone(),
            knots.clone(),
            1,
            1,
            p_marginal,
        );
        let marginal_callback = SmsTimewiggleMarginalJacobian::new(
            design_time.clone(),
            design_time.clone(),
            design_time,
            design_marginal,
            offset.clone(),
            offset.clone(),
            offset.clone(),
            offset,
            knots,
            1,
            p_time,
            1,
        );

        // Each callback receives only its own block's nonzero coefficients.
        // The exact current rows must therefore come from the family scalar
        // snapshot, never from parsing this slice as a joint coefficient vector.
        let time_beta = [7.0, -3.0];
        let time_state = crate::custom_family::FamilyLinearizationState {
            beta: &time_beta,
            family_scalars: Some(scalars.clone()),
            channel_hessian: None,
            probit_frailty_scale: 0.8,
        };
        let marginal_beta = [11.0];
        let marginal_state = crate::custom_family::FamilyLinearizationState {
            beta: &marginal_beta,
            family_scalars: Some(scalars),
            channel_hessian: None,
            probit_frailty_scale: 0.8,
        };
        let actual_time = crate::custom_family::BlockEffectiveJacobian::effective_jacobian_rows(
            &time_callback,
            &time_state,
            0..n,
        )
        .unwrap();
        let actual_marginal =
            crate::custom_family::BlockEffectiveJacobian::effective_jacobian_rows(
                &marginal_callback,
                &marginal_state,
                0..n,
            )
            .unwrap();

        for row in 0..n {
            for channel in 0..3 {
                for column in 0..p_time {
                    let index = channel * n + row;
                    assert_eq!(
                        actual_time[[index, column]],
                        c_i[row] * time_rows[[index, column]]
                    );
                }
                let index = channel * n + row;
                assert_eq!(
                    actual_marginal[[index, 0]],
                    c_i[row] * marginal_rows[[index, 0]],
                );
            }
        }
    }
}
