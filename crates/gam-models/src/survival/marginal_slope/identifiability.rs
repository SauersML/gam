//! Survival marginal-slope concrete impls for the family-agnostic
//! identifiability compiler (`gam_identifiability::families::compiler`).
//!
//! Survival's row primary state is
//! `u_i = (q0, q1, qd1, g_0, …, g_{K-1})`, so its width is `3 + K` for a
//! K-dimensional latent score. The row Hessian is derived from the same vector
//! row program as the likelihood and PSD-projected at the pilot point.
//!
//! Each block exposes its row Jacobian as the contribution of `δβ_block`
//! to the row primary-state vector:
//!
//! - **TimeBlockOperator**: `(δq0, δq1, δqd1, 0_K)` from `design_entry`,
//!   `design_exit`, `design_derivative_exit` rows.
//! - **MarginalBlockOperator**: `(δq, δq, δqd_marginal, 0)` from the
//!   marginal design row (shared by q0 and q1; qd contribution zero unless
//!   timewiggle is active — captured by an explicit derivative row matrix).
//! - **LogslopeBlockOperator**: `(0, 0, 0, δg_0, …, δg_{K-1})` from the
//!   canonical physical-channel layout.
//! - **ScoreWarpBlockOperator**: `(δq, δq, δqd_warp, 0)` from the warp
//!   basis (shifts q at entry/exit; chain rule via dq0_seed/dt for qd1).
//! - **LinkDevBlockOperator**: `(δq, δq, δqd_link, 0)` from the link-dev
//!   basis on the rigid/pilot q-seed.
//!
//! Phase 4a delivery: trait impls + an input-builder helper. Phase 4b
//! threads these through SMGS's construction site and the migrated pilot
//! β; Phase 4c deletes the legacy
//! `install_compiled_flex_block_into_runtime` path.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use faer::Side;
use gam_identifiability::families::compiler::{
    BlockOrder, RowHessian, RowJacobianOperator, scale_jacobian_by_sqrt_h_with,
};
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::matrix::{CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix};
use gam_problem::gauge::assemble_block_triangular_t;
use gam_problem::{FamilyChannelHessian, PenaltyMatrix};

/// Per-row `(3 + K) × (3 + K)` row Hessian for the survival marginal-slope
/// likelihood at a pilot `β`.
pub struct SurvivalRowHessian {
    /// PSD-projected per-row Hessian, stored as `(n × (3+K) × (3+K))`.
    h: Array3<f64>,
    /// Immutable row data needed to refresh `h` exactly at a new primary
    /// state. These must not be replaced by unit weights/events: censoring and
    /// observation weights change the channel curvature and therefore the
    /// identifiability certificate.
    weights: Array1<f64>,
    event: Array1<f64>,
    z: Array2<f64>,
    covariance: crate::bms::MarginalSlopeCovariance,
    derivative_guard: f64,
}

impl SurvivalRowHessian {
    /// Construct from the full physical primary state. `slopes` and `z` are
    /// both `(n × K)`; no score coordinate is privileged or discarded.
    pub fn from_pilot_primary_state(
        q0: &Array1<f64>,
        q1: &Array1<f64>,
        qd1: &Array1<f64>,
        slopes: &Array2<f64>,
        z: &Array2<f64>,
        covariance: &crate::bms::MarginalSlopeCovariance,
        weights: &Array1<f64>,
        event: &Array1<f64>,
        derivative_guard: f64,
        probit_scale: f64,
    ) -> Result<Self, String> {
        let n = q0.len();
        if [
            q1.len(),
            qd1.len(),
            slopes.nrows(),
            z.nrows(),
            weights.len(),
            event.len(),
        ]
        .iter()
        .any(|&l| l != n)
        {
            return Err(format!(
                "SurvivalRowHessian: row mismatch \
                 q0={n}, q1={}, qd1={}, slopes={}, z={}, weights={}, event={}",
                q1.len(),
                qd1.len(),
                slopes.nrows(),
                z.nrows(),
                weights.len(),
                event.len()
            ));
        }
        let score_dim = covariance.dim();
        if slopes.ncols() != score_dim || z.ncols() != score_dim {
            return Err(format!(
                "SurvivalRowHessian: score dimension mismatch slopes={}, z={}, covariance={score_dim}",
                slopes.ncols(),
                z.ncols(),
            ));
        }
        let primary_width = 3 + score_dim;
        let mut h_full = Array3::<f64>::zeros((n, primary_width, primary_width));
        let mut workspace =
            crate::survival::marginal_slope::RigidVectorRowWorkspace::new(covariance)?;
        for i in 0..n {
            let clamped = evaluated_psd_row_hessian(
                q0[i],
                q1[i],
                qd1[i],
                slopes
                    .row(i)
                    .as_slice()
                    .ok_or_else(|| "SurvivalRowHessian slope row is not contiguous".to_string())?,
                z.row(i)
                    .as_slice()
                    .ok_or_else(|| "SurvivalRowHessian score row is not contiguous".to_string())?,
                weights[i],
                event[i],
                derivative_guard,
                probit_scale,
                &mut workspace,
            )
            .map_err(|reason| format!("SurvivalRowHessian: row {i}: {reason}"))?;
            for a in 0..primary_width {
                for b in 0..primary_width {
                    h_full[[i, a, b]] = clamped[[a, b]];
                }
            }
        }
        Ok(Self {
            h: h_full,
            weights: weights.clone(),
            event: event.clone(),
            z: z.clone(),
            covariance: covariance.clone(),
            derivative_guard,
        })
    }
}

impl RowHessian for SurvivalRowHessian {
    fn k(&self) -> usize {
        self.h.shape()[1]
    }
    fn nrows(&self) -> usize {
        self.h.shape()[0]
    }
    fn fill_row(&self, row: usize, out: &mut [f64]) {
        let k = self.k();
        assert_eq!(out.len(), k * k);
        for a in 0..k {
            for b in 0..k {
                out[a * k + b] = self.h[[row, a, b]];
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        self.h.clone()
    }
}

/// `FamilyChannelHessian` for survival marginal-slope. Both the pilot snapshot
/// and current-β refresh use the same `(3 + K)` vector row program.
///
/// # β-dependent W via `channel_hessian_at`
///
/// `channel_hessian_at` overrides the default β-independent path.  When
/// `family_scalars` carries `SurvivalMarginalSlopeFamilyScalars`, the
/// current per-row primary state `(q0_i, q1_i, qd1_i, g_{i,0}, …, g_{i,K-1})`
/// is read from those scalars and the `(3 + K) × (3 + K)` `W_i` is recomputed
/// by the same vector row program used by the fitted likelihood.
/// This makes `I(β) = J(β)^T W(β) J(β)` accurate at the current β instead of
/// at the frozen pilot β=0 state.
///
/// When `family_scalars` is `None` and `beta` is exactly zero, the frozen pilot
/// W is returned unchanged. When `family_scalars` is `None` and any `beta`
/// entry is non-zero, `Err` is returned — the caller must
/// supply scalars for a correct W at non-pilot β (same contract as T26's
/// Jacobian callbacks: scalars required when β affects the primary state).
impl FamilyChannelHessian for SurvivalRowHessian {
    fn n_outputs(&self) -> usize {
        self.h.shape()[1]
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        let k = self.h.shape()[1];
        assert_eq!(out.len(), k * k);
        for a in 0..k {
            for b in 0..k {
                out[a * k + b] = self.h[[i, a, b]];
            }
        }
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }

    fn channel_hessian_at(
        &self,
        beta: &[f64],
        family_scalars: Option<&Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Result<Arc<dyn FamilyChannelHessian>, String> {
        use crate::survival::marginal_slope::SurvivalMarginalSlopeFamilyScalars;

        if beta.iter().any(|b| !b.is_finite()) {
            return Err(
                "SurvivalRowHessian::channel_hessian_at: beta contains a non-finite value"
                    .to_string(),
            );
        }
        if beta.is_empty() && family_scalars.is_some() {
            return Err(
                "SurvivalRowHessian::channel_hessian_at: family_scalars supplied but beta is empty"
                    .to_string(),
            );
        }
        let scalars_opt = match family_scalars {
            None => None,
            Some(scalars) => Some(
                scalars
                    .downcast_ref::<SurvivalMarginalSlopeFamilyScalars>()
                    .ok_or_else(|| {
                        "SurvivalRowHessian::channel_hessian_at: family_scalars has the wrong type; expected SurvivalMarginalSlopeFamilyScalars"
                            .to_string()
                    })?,
            ),
        };

        let beta_nontrivial = beta.iter().any(|&b| b != 0.0);

        match scalars_opt {
            None if beta_nontrivial => {
                // β is non-zero in a way that would change W via the primary-state
                // coupling (g ≠ 0 → c ≠ 1 → W changes).  Scalars are required.
                Err(
                    "SurvivalRowHessian::channel_hessian_at: beta is non-trivial but \
                     family_scalars is None; supply SurvivalMarginalSlopeFamilyScalars \
                     via FamilyLinearizationState::family_scalars to evaluate W(β) \
                     correctly (same contract as T26 Jacobian callbacks)."
                        .to_string(),
                )
            }
            None => {
                // β = 0: return the exact stored pilot W unchanged.
                Ok(Arc::new(gam_problem::TensorChannelHessian {
                    h: self.h.clone(),
                }))
            }
            Some(sc) => {
                let n = self.h.shape()[0];
                if sc.q0_i.len() != n
                    || sc.q1_i.len() != n
                    || sc.qd1_i.len() != n
                    || sc.slopes.nrows() != n
                    || sc.slopes.ncols() != self.covariance.dim()
                {
                    return Err(format!(
                        "SurvivalRowHessian::channel_hessian_at: scalars length mismatch \
                         (expected n={n}, K={}, got q0={} q1={} qd1={} slopes={}x{})",
                        self.covariance.dim(),
                        sc.q0_i.len(),
                        sc.q1_i.len(),
                        sc.qd1_i.len(),
                        sc.slopes.nrows(),
                        sc.slopes.ncols(),
                    ));
                }
                let primary_width = self.h.shape()[1];
                let mut h_full = Array3::<f64>::zeros((n, primary_width, primary_width));
                let mut workspace = crate::survival::marginal_slope::RigidVectorRowWorkspace::new(
                    &self.covariance,
                )?;
                for i in 0..n {
                    let clamped = evaluated_psd_row_hessian(
                        sc.q0_i[i],
                        sc.q1_i[i],
                        sc.qd1_i[i],
                        sc.slopes.row(i).as_slice().ok_or_else(|| {
                            "SurvivalRowHessian::channel_hessian_at slope row is not contiguous"
                                .to_string()
                        })?,
                        self.z.row(i).as_slice().ok_or_else(|| {
                            "SurvivalRowHessian::channel_hessian_at score row is not contiguous"
                                .to_string()
                        })?,
                        self.weights[i],
                        self.event[i],
                        self.derivative_guard,
                        sc.s,
                        &mut workspace,
                    )
                    .map_err(|reason| {
                        format!("SurvivalRowHessian::channel_hessian_at: row {i}: {reason}")
                    })?;
                    for a in 0..primary_width {
                        for b in 0..primary_width {
                            h_full[[i, a, b]] = clamped[[a, b]];
                        }
                    }
                }
                Ok(Arc::new(gam_problem::TensorChannelHessian { h: h_full }))
            }
        }
    }
}

fn evaluated_psd_row_hessian(
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    weight: f64,
    event: f64,
    derivative_guard: f64,
    probit_scale: f64,
    workspace: &mut crate::survival::marginal_slope::RigidVectorRowWorkspace<'_>,
) -> Result<Array2<f64>, String> {
    crate::survival::marginal_slope::row_primary_closed_form_vector_into(
        q0,
        q1,
        qd1,
        slopes,
        z,
        weight,
        event,
        derivative_guard,
        probit_scale,
        workspace,
    )?;
    let (_, hessian) = workspace.derivatives();
    psd_clamp(&hessian.to_owned())
}

/// Project a symmetric matrix onto the PSD cone by zeroing negative
/// eigenvalues. A failed decomposition is an invalid curvature certificate,
/// so it is reported instead of being replaced by unrelated diagonal data.
fn psd_clamp(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    let k = m.nrows();
    if k == 0 || m.ncols() != k {
        return Err(format!(
            "survival row Hessian must be non-empty and square, got {}x{}",
            m.nrows(),
            m.ncols(),
        ));
    }
    if m.iter().any(|v| !v.is_finite()) {
        return Err("survival row Hessian contains a non-finite entry".to_string());
    }
    let (evals, evecs) = m
        .eigh(Side::Lower)
        .map_err(|_| "survival row Hessian symmetric eigendecomposition failed".to_string())?;
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += evecs[[i, l]] * evals[l].max(0.0) * evecs[[j, l]];
            }
            out[[i, j]] = acc;
        }
    }
    Ok(out)
}

/// Row Jacobian operator for the survival time block. Channels (q0, q1,
/// qd1) come from the three time designs; the g channel is zero.
pub struct TimeBlockOperator {
    dq0: Array2<f64>,
    dq1: Array2<f64>,
    dqd1: Array2<f64>,
    score_dim: usize,
}

impl TimeBlockOperator {
    pub fn new(dq0: Array2<f64>, dq1: Array2<f64>, dqd1: Array2<f64>, score_dim: usize) -> Self {
        assert_eq!(dq0.dim(), dq1.dim());
        assert_eq!(dq0.dim(), dqd1.dim());
        assert!(score_dim > 0);
        Self {
            dq0,
            dq1,
            dqd1,
            score_dim,
        }
    }
}

impl RowJacobianOperator for TimeBlockOperator {
    fn k(&self) -> usize {
        3 + self.score_dim
    }
    fn ncols(&self) -> usize {
        self.dq0.ncols()
    }
    fn nrows(&self) -> usize {
        self.dq0.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        let k = self.k();
        assert_eq!(out.len(), k);
        assert_eq!(delta_beta.len(), self.dq0.ncols());
        out.fill(0.0);
        for (j, &b) in delta_beta.iter().enumerate() {
            out[0] += self.dq0[[row, j]] * b;
            out[1] += self.dq1[[row, j]] * b;
            out[2] += self.dqd1[[row, j]] * b;
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dq0.nrows();
        let p = self.dq0.ncols();
        let mut out = Array3::<f64>::zeros((n, p, self.k()));
        for i in 0..n {
            for j in 0..p {
                out[[i, j, 0]] = self.dq0[[i, j]];
                out[[i, j, 1]] = self.dq1[[i, j]];
                out[[i, j, 2]] = self.dqd1[[i, j]];
            }
        }
        out
    }
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        // Scale straight out of the three compact `(n, p)` channel designs —
        // the compiler consumes the `(n·K, p)` sqrt(H)-scaled design, so the
        // dense `(n, p, K)` tensor (3 of its 4 channels held explicitly, the
        // 4th identically zero) that `evaluate_full()` builds is never needed.
        // (#738: a capability is not a representation.)
        let n = self.dq0.nrows();
        let p = self.dq0.ncols();
        scale_jacobian_by_sqrt_h_with(n, p, self.k(), h_full, |i, a, c| match c {
            0 => self.dq0[[i, a]],
            1 => self.dq1[[i, a]],
            2 => self.dqd1[[i, a]],
            _ => 0.0,
        })
    }
}

/// Row Jacobian operator for a block whose contribution flows into the
/// q-channels (q0 and q1 identically) and optionally the qd1 channel.
/// Covers the survival marginal, score-warp, and link-dev blocks (all
/// three share the structural property `δq0 = δq1 = basis·δβ`, `δg = 0`).
pub struct QChannelBlockOperator {
    dq: Array2<f64>,
    dqd1: Array2<f64>,
    score_dim: usize,
}

impl QChannelBlockOperator {
    pub fn new(dq: Array2<f64>, dqd1: Array2<f64>, score_dim: usize) -> Self {
        assert_eq!(dq.dim(), dqd1.dim());
        assert!(score_dim > 0);
        Self {
            dq,
            dqd1,
            score_dim,
        }
    }
}

impl RowJacobianOperator for QChannelBlockOperator {
    fn k(&self) -> usize {
        3 + self.score_dim
    }
    fn ncols(&self) -> usize {
        self.dq.ncols()
    }
    fn nrows(&self) -> usize {
        self.dq.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), self.k());
        assert_eq!(delta_beta.len(), self.dq.ncols());
        let mut dq_acc = 0.0;
        let mut dqd_acc = 0.0;
        for (j, &b) in delta_beta.iter().enumerate() {
            dq_acc += self.dq[[row, j]] * b;
            dqd_acc += self.dqd1[[row, j]] * b;
        }
        out[0] = dq_acc;
        out[1] = dq_acc;
        out[2] = dqd_acc;
        out[3..].fill(0.0);
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dq.nrows();
        let p = self.dq.ncols();
        let mut out = Array3::<f64>::zeros((n, p, self.k()));
        for i in 0..n {
            for j in 0..p {
                let v = self.dq[[i, j]];
                out[[i, j, 0]] = v;
                out[[i, j, 1]] = v;
                out[[i, j, 2]] = self.dqd1[[i, j]];
            }
        }
        out
    }
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        // q0 and q1 share `dq`; qd1 is `dqd1`; the g channel is identically
        // zero. Scale directly from the compact `(n, p)` designs, skipping the
        // dense `(n, p, K)` tensor `evaluate_full()` would build. (#738.)
        let n = self.dq.nrows();
        let p = self.dq.ncols();
        scale_jacobian_by_sqrt_h_with(n, p, self.k(), h_full, |i, a, c| match c {
            0 | 1 => self.dq[[i, a]],
            2 => self.dqd1[[i, a]],
            _ => 0.0,
        })
    }
}

/// Row Jacobian operator for the survival log-slope block. The canonical
/// [`crate::survival::marginal_slope::LogslopeLayout`] emits one physical
/// channel row per score coordinate; this operator validates and materialises
/// those rows once as a compact `(n K) × p` matrix, then serves every hot row
/// callback by borrowed reads. It never stores the mostly-zero `n × p × (3+K)`
/// tensor.
pub(crate) struct LogslopeBlockOperator {
    physical_rows: Array2<f64>,
    nrows: usize,
    score_dim: usize,
}

impl LogslopeBlockOperator {
    pub(crate) fn new(
        layout: crate::survival::marginal_slope::LogslopeLayout,
        score_dim: usize,
    ) -> Result<Self, String> {
        let width = layout.coefficient_design().ncols();
        Self::restricted(layout, score_dim, 0..width)
    }

    fn restricted(
        layout: crate::survival::marginal_slope::LogslopeLayout,
        score_dim: usize,
        columns: std::ops::Range<usize>,
    ) -> Result<Self, String> {
        layout.validate_for(score_dim)?;
        let width = layout.coefficient_design().ncols();
        if columns.end < columns.start || columns.end > width {
            return Err(format!(
                "logslope identifiability operator column range {:?} is outside 0..{width}",
                columns,
            ));
        }
        let nrows = layout.coefficient_design().nrows();
        let mut physical_rows = Array2::<f64>::zeros((nrows * score_dim, columns.len()));
        layout.fill_primary_jacobian_rows(score_dim, 0..nrows, columns, &mut physical_rows)?;
        Ok(Self {
            physical_rows,
            nrows,
            score_dim,
        })
    }
}

impl RowJacobianOperator for LogslopeBlockOperator {
    fn k(&self) -> usize {
        3 + self.score_dim
    }
    fn ncols(&self) -> usize {
        self.physical_rows.ncols()
    }
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), self.k());
        assert_eq!(delta_beta.len(), self.ncols());
        out.fill(0.0);
        for channel in 0..self.score_dim {
            for (column, &coefficient) in delta_beta.iter().enumerate() {
                out[3 + channel] +=
                    self.physical_rows[[row * self.score_dim + channel, column]] * coefficient;
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.nrows();
        let p = self.ncols();
        let mut out = Array3::<f64>::zeros((n, p, self.k()));
        for i in 0..n {
            for j in 0..p {
                for channel in 0..self.score_dim {
                    out[[i, j, 3 + channel]] =
                        self.physical_rows[[i * self.score_dim + channel, j]];
                }
            }
        }
        out
    }
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        let n = self.nrows();
        let p = self.ncols();
        let primary_width = self.k();
        assert_eq!(h_full.shape(), &[n, primary_width, primary_width]);
        scale_jacobian_by_sqrt_h_with(n, p, primary_width, h_full, |row, column, channel| {
            if channel < 3 {
                0.0
            } else {
                self.physical_rows[[row * self.score_dim + channel - 3, column]]
            }
        })
    }

    fn channel_flattened_rows(&self, rows: std::ops::Range<usize>, out: &mut Array2<f64>) {
        let start = rows.start.min(self.nrows());
        let end = rows.end.min(self.nrows());
        let chunk = end - start;
        assert_eq!(out.dim(), (chunk * self.k(), self.ncols()));
        out.fill(0.0);
        for local_row in 0..chunk {
            for channel in 0..self.score_dim {
                out.row_mut(local_row * self.k() + 3 + channel).assign(
                    &self
                        .physical_rows
                        .row((start + local_row) * self.score_dim + channel),
                );
            }
        }
    }
}

/// Inputs assembled for the survival fit driver to feed `compile()`. The
/// ordering follows `gauge_priority` descending (time=200 → marginal=150 →
/// logslope=120 → score_warp=80 → link_dev=60).
pub(crate) struct SurvivalCompilerInputs {
    pub(crate) operators: Vec<Arc<dyn RowJacobianOperator>>,
    pub(crate) ordering: Vec<BlockOrder>,
}

/// Per-block V reparameterisation matrices for the three parametric
/// survival blocks emitted by [`compile_survival_parametric_designs`].
/// Each `v_*` is a `(p_block_raw × p_block_kept)` selection-or-rotation
/// matrix that maps a `β_kept` coefficient vector to its `β_raw`
/// equivalent: `β_raw = V · β_kept`. The construction site applies these
/// to the raw block designs (`design_raw · V → design_compiled`) and
/// to the penalties (`Vᵀ S V`) before building `ParameterBlockSpec`s
/// and passing the compiled designs into `make_family`.
///
/// Phase-4b architecture: this is the seam where the family-agnostic
/// row-Jacobian compiler hands control back to the family-specific
/// construction site. Each `v_*` width equals the corresponding
/// `CompiledBlocks::blocks[i].t_lw.ncols()` — i.e., the kept-direction
/// count after sqrt-H-metric residualisation and post-walk RRQR
/// trailing-pivot drop.
pub(crate) struct SurvivalParametricCompiled {
    pub(crate) v_time: Array2<f64>,
    pub(crate) v_marginal: Array2<f64>,
    pub(crate) v_logslope: Array2<f64>,
    /// Per-block dropped raw-column count, indexed
    /// `(time_dropped, marginal_dropped, logslope_dropped)`. Equal to
    /// `(p_raw − v.ncols())` for each block. Useful for logging the
    /// gauge-attribution summary at the construction site.
    pub(crate) drops_by_block: (usize, usize, usize),
}

pub(crate) fn wrap_design_with_transform(
    raw: DesignMatrix,
    v: &Array2<f64>,
    context: &str,
) -> Result<DesignMatrix, String> {
    if raw.ncols() != v.nrows() {
        return Err(format!(
            "{context}: raw design has {} cols but V has {} rows (V is {}×{})",
            raw.ncols(),
            v.nrows(),
            v.nrows(),
            v.ncols(),
        ));
    }
    let inner_dense = match raw {
        DesignMatrix::Dense(d) => d,
        DesignMatrix::Sparse(_) => {
            let dense = raw
                .try_to_dense_by_chunks(&format!("{context} sparse→dense for V apply"))
                .map_err(|reason| format!("{context}: densify failed: {reason}"))?;
            DenseDesignMatrix::from(dense)
        }
    };
    let op = CoefficientTransformOperator::new(inner_dense, v.clone())
        .map_err(|reason| format!("{context}: CoefficientTransformOperator::new: {reason}"))?;
    Ok(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op))))
}

/// Per-term V reparameterisation matrices for the three parametric
/// survival blocks. Each block's full V is the block-diagonal assembly
/// of its per-term V's (one entry per element of the input
/// `*_partition`). Preserves per-term penalty structure: applying
/// `V_b = block_diag(V_term1, ..., V_termM)` to a per-term BlockwisePenalty
/// pulls each penalty back only via its OWN term's V, so what was a
/// per-term λ tunable in REML stays per-term tunable.
pub struct SurvivalParametricCompiledPerTerm {
    pub v_time_per_term: Vec<Array2<f64>>,
    pub v_marginal_per_term: Vec<Array2<f64>>,
    pub v_logslope_per_term: Vec<Array2<f64>>,
    /// Per-term residualised reparam `R_b = M_b · V_b` from the
    /// identifiability compiler, in the same global compile order
    /// (time terms, then marginal terms, then logslope terms). `None`
    /// for the very first compiled block (no anchor). Used by the
    /// V+M-exact apply path to emit residualised rows
    /// `C_b·V_b − A_{<b}·R_b` and to assemble the full triangular T.
    pub r_lw_per_term: Vec<Option<Array2<f64>>>,
    /// Per-block drops (raw_cols − sum(kept_cols across terms)).
    pub drops_by_block: (usize, usize, usize),
}

/// Per-term-aware compile: residualise each block's TERMS individually
/// in priority order so the emitted V is block-diagonal on term
/// boundaries. This preserves the per-term penalty structure that
/// REML's per-λ accounting depends on.
///
/// Each `*_partition` is a list of disjoint contiguous column ranges
/// covering `[0..p_block)`. For the marginal/logslope blocks the
/// natural source is the union of `BlockwisePenalty::col_range` values
/// (one per smoothness penalty / term) plus the complement
/// (unpenalised parametric columns).
///
/// Order of residualisation: time terms first (in their partition
/// order), then marginal terms, then logslope terms. Within each
/// block, terms are residualised against ALL prior anchor columns
/// (terms from earlier blocks + earlier terms within this block).
/// Aliased directions land in the lowest-priority block that contains
/// them, in the natural term order within that block — matching the
/// gauge-priority ownership contract.
pub(crate) fn compile_survival_parametric_designs_per_term(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    time_partition: &[std::ops::Range<usize>],
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    marginal_partition: &[std::ops::Range<usize>],
    logslope_layout: crate::survival::marginal_slope::LogslopeLayout,
    logslope_partition: &[std::ops::Range<usize>],
    row_hess: &dyn RowHessian,
    protect_time: bool,
) -> Result<SurvivalParametricCompiledPerTerm, String> {
    use gam_identifiability::families::compiler::compile_protected;

    let p_time = time_dq0.ncols();
    let p_marg = marginal_dq.ncols();
    let score_dim = row_hess.k().checked_sub(3).ok_or_else(|| {
        format!(
            "per-term compile row Hessian width {} is smaller than the three q channels",
            row_hess.k()
        )
    })?;
    if score_dim == 0 {
        return Err("per-term compile requires at least one score channel".to_string());
    }
    logslope_layout.validate_for(score_dim)?;
    let p_log = logslope_layout.coefficient_design().ncols();
    validate_partition(time_partition, p_time, "time")?;
    validate_partition(marginal_partition, p_marg, "marginal")?;
    validate_partition(logslope_partition, p_log, "logslope")?;

    // Build per-term operators. Each term gets its own RowJacobianOperator
    // restricted to its column slice; the operator type matches the
    // block's K-channel signature (Time, QChannel, Logslope).
    let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::new();
    let mut ordering: Vec<BlockOrder> = Vec::new();
    for range in time_partition {
        let dq0 = time_dq0.slice(ndarray::s![.., range.clone()]).to_owned();
        let dq1 = time_dq1.slice(ndarray::s![.., range.clone()]).to_owned();
        let dqd1 = time_dqd1.slice(ndarray::s![.., range.clone()]).to_owned();
        operators.push(Arc::new(TimeBlockOperator::new(dq0, dq1, dqd1, score_dim)));
        ordering.push(BlockOrder::Time);
    }
    for range in marginal_partition {
        let dq = marginal_dq.slice(ndarray::s![.., range.clone()]).to_owned();
        let dqd1 = marginal_dqd1
            .slice(ndarray::s![.., range.clone()])
            .to_owned();
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1, score_dim)));
        ordering.push(BlockOrder::Marginal);
    }
    for range in logslope_partition {
        operators.push(Arc::new(LogslopeBlockOperator::restricted(
            logslope_layout.clone(),
            score_dim,
            range.clone(),
        )?));
        ordering.push(BlockOrder::Logslope);
    }

    // The time block carries the monotone time-wiggle basis, whose effective
    // Jacobian is a fixed nonlinear functional basis rather than a linear
    // design. When `protect_time` is set it must be kept at full raw width: a
    // linear reparameterisation of it would desynchronise the raw-width
    // wiggle-basis chain rule (`SmsTimewiggleTimeJacobian`), which recomputes
    // the basis on every evaluation. Marginal/logslope still reduce against the
    // full time anchor. The time block spans operators `0..n_time` (pushed
    // first above); mark exactly those protected.
    let n_time = time_partition.len();
    let protected: Vec<bool> = if protect_time {
        (0..operators.len()).map(|i| i < n_time).collect()
    } else {
        Vec::new()
    };
    let compiled = compile_protected(&operators, row_hess, &ordering, &protected).map_err(|e| {
        format!("identifiability::families::compiler::compile (per-term) failed: {e}")
    })?;
    let blocks = compiled.blocks;
    let n_marg = marginal_partition.len();
    let n_log = logslope_partition.len();
    if blocks.len() != n_time + n_marg + n_log {
        return Err(format!(
            "per-term compile: expected {} compiled blocks (time={}, marg={}, log={}), got {}",
            n_time + n_marg + n_log,
            n_time,
            n_marg,
            n_log,
            blocks.len(),
        ));
    }
    let mut iter = blocks.into_iter();
    let mut v_time_per_term: Vec<Array2<f64>> = Vec::with_capacity(n_time);
    let mut r_time_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_time);
    for _ in 0..n_time {
        let blk = iter.next().unwrap();
        v_time_per_term.push(blk.t_lw);
        r_time_per_term.push(blk.r_lw);
    }
    let mut v_marginal_per_term: Vec<Array2<f64>> = Vec::with_capacity(n_marg);
    let mut r_marginal_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_marg);
    for _ in 0..n_marg {
        let blk = iter.next().unwrap();
        v_marginal_per_term.push(blk.t_lw);
        r_marginal_per_term.push(blk.r_lw);
    }
    let mut v_logslope_per_term: Vec<Array2<f64>> = Vec::with_capacity(n_log);
    let mut r_logslope_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_log);
    for _ in 0..n_log {
        let blk = iter.next().unwrap();
        v_logslope_per_term.push(blk.t_lw);
        r_logslope_per_term.push(blk.r_lw);
    }
    let mut r_lw_per_term: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_time + n_marg + n_log);
    r_lw_per_term.extend(r_time_per_term);
    r_lw_per_term.extend(r_marginal_per_term);
    r_lw_per_term.extend(r_logslope_per_term);
    let drops_time: usize = time_partition
        .iter()
        .zip(v_time_per_term.iter())
        .map(|(r, v)| r.len().saturating_sub(v.ncols()))
        .sum();
    let drops_marg: usize = marginal_partition
        .iter()
        .zip(v_marginal_per_term.iter())
        .map(|(r, v)| r.len().saturating_sub(v.ncols()))
        .sum();
    let drops_log: usize = logslope_partition
        .iter()
        .zip(v_logslope_per_term.iter())
        .map(|(r, v)| r.len().saturating_sub(v.ncols()))
        .sum();
    Ok(SurvivalParametricCompiledPerTerm {
        v_time_per_term,
        v_marginal_per_term,
        v_logslope_per_term,
        r_lw_per_term,
        drops_by_block: (drops_time, drops_marg, drops_log),
    })
}

fn validate_partition(
    partition: &[std::ops::Range<usize>],
    p_block: usize,
    label: &str,
) -> Result<(), String> {
    if partition.is_empty() {
        if p_block == 0 {
            return Ok(());
        }
        return Err(format!(
            "{label} partition empty but block has p={p_block} columns"
        ));
    }
    if partition[0].start != 0 {
        return Err(format!(
            "{label} partition must start at 0, got start={}",
            partition[0].start
        ));
    }
    if partition.last().unwrap().end != p_block {
        return Err(format!(
            "{label} partition must cover [0, {p_block}); last range ends at {}",
            partition.last().unwrap().end
        ));
    }
    for w in partition.windows(2) {
        if w[0].end != w[1].start {
            return Err(format!(
                "{label} partition has gap/overlap between [{}..{}) and [{}..{})",
                w[0].start, w[0].end, w[1].start, w[1].end
            ));
        }
        if w[0].is_empty() {
            return Err(format!(
                "{label} partition has empty range [{}..{})",
                w[0].start, w[0].end
            ));
        }
    }
    if partition.last().unwrap().is_empty() {
        return Err(format!("{label} partition's final range is empty",));
    }
    Ok(())
}

/// Derive a disjoint contiguous partition of `[0..p_block)` from a
/// list of BlockwisePenalty col_ranges. Distinct penalty ranges define
/// term boundaries; gaps between them (unpenalised columns) become
/// their own single-column partitions. Multiple penalties with the
/// SAME col_range (e.g. tensor anisotropy axes) coalesce to one term.
pub fn extract_term_partition_from_penalty_ranges(
    p_block: usize,
    penalty_ranges: &[std::ops::Range<usize>],
) -> Vec<std::ops::Range<usize>> {
    use std::collections::BTreeSet;
    let mut starts: BTreeSet<usize> = BTreeSet::new();
    starts.insert(0);
    starts.insert(p_block);
    for r in penalty_ranges {
        starts.insert(r.start.min(p_block));
        starts.insert(r.end.min(p_block));
    }
    let v: Vec<usize> = starts.into_iter().collect();
    v.windows(2)
        .filter_map(|w| if w[0] < w[1] { Some(w[0]..w[1]) } else { None })
        .collect()
}

/// Pull a single raw block-local [`BlockwisePenalty`] back through the
/// block's own diagonal reparameterisation `V_b` (the `(b, b)` block of
/// the triangular T), producing a per-block-width compiled penalty.
///
/// The penalty's `local` is `pen.col_range.len()` square and covers a
/// sub-region of the raw block at offset `pen.col_range.start` (which is
/// block-local, i.e. relative to the block's first raw column). It is
/// embedded into the full raw block width `v_block.nrows()` at that
/// offset, then pulled back as `V_bᵀ · embed(S) · V_b`, giving a
/// `(w_b_compiled × w_b_compiled)` symmetric `PenaltyMatrix::Dense`
/// where `w_b_compiled == v_block.ncols()`.
///
/// This is the penalty contract a per-block `ParameterBlockSpec`
/// requires: each block's penalty acts on that block's own compiled
/// coordinate `θ_b`. The cross-block residualisation `R_{a→b}` carried
/// in T's strict-upper triangle is absorbed into the *design* columns
/// (the residualised emitted design `C_b V_b − A_{<b} R_b`), not into
/// the penalty — exactly as the VM-exact compile-map path
/// [`apply_compiled_map_to_designs`] does. Pulling the penalty back
/// through the full joint T instead would yield a `(p_compiled × p_compiled)` dense
/// matrix that cannot live in a single block's `penalties` slot and
/// would violate the `p_b × p_b` block-spec validation.
pub fn pull_back_blockwise_penalty_through_block_v(
    pen: &gam_terms::smooth::BlockwisePenalty,
    v_block: &Array2<f64>,
) -> Result<PenaltyMatrix, String> {
    let raw_p = v_block.nrows();
    let compiled_p = v_block.ncols();
    let block_p = pen.col_range.len();
    let embed_start = pen.col_range.start;
    let embed_end = pen.col_range.end;
    if embed_end > raw_p {
        return Err(format!(
            "pull_back_blockwise_penalty_through_block_v: penalty col_range {embed_start}..{embed_end} \
             exceeds block raw width {raw_p}"
        ));
    }
    if pen.local.nrows() != block_p || pen.local.ncols() != block_p {
        return Err(format!(
            "pull_back_blockwise_penalty_through_block_v: penalty local is {}x{} but col_range \
             width is {block_p}",
            pen.local.nrows(),
            pen.local.ncols(),
        ));
    }
    let mut embedded = Array2::<f64>::zeros((raw_p, raw_p));
    if block_p > 0 {
        let mut dst =
            embedded.slice_mut(ndarray::s![embed_start..embed_end, embed_start..embed_end]);
        for i in 0..block_p {
            for j in 0..block_p {
                dst[[i, j]] = pen.local[[i, j]];
            }
        }
    }
    // V_bᵀ · embed(S) · V_b → (compiled_p × compiled_p).
    let temp = embedded.dot(v_block);
    let pulled = v_block.t().dot(&temp);
    let mut sym = Array2::<f64>::zeros((compiled_p, compiled_p));
    for i in 0..compiled_p {
        for j in 0..compiled_p {
            sym[[i, j]] = 0.5 * (pulled[[i, j]] + pulled[[j, i]]);
        }
    }
    Ok(PenaltyMatrix::Dense(sym))
}

/// Assemble a 3-block [`CompiledMap`] (time, marginal, logslope) from a
/// [`SurvivalParametricCompiledPerTerm`] produced by the full 4×4 row-Hessian
/// driver [`compile_survival_parametric_designs_per_term`].
///
/// The full global triangular `T` is built from the per-term `V`/`R` blocks
/// (diagonal `V_b`, strict-upper `−R_{a→b}` — identical to the matrix the
/// result-time lift [`Gauge::from_v_and_r`] uses), then partitioned
/// into the three *block* ranges (raw = summed per-term raw widths, compiled =
/// summed per-term kept widths). The resulting `CompiledMap` is interchangeable
/// with one from
/// [`gam_identifiability::families::compiler::compile_from_raw_grams`], so the
/// existing [`apply_compiled_map_to_designs`] +
/// [`Gauge::from_compiled_map`] machinery consumes it unchanged.
///
/// This is the seam that lets the survival closed-form fast path engage on the
/// *correct* identifiable quotient: the cheap η₁-only rawstack metric can
/// falsely collapse a whole channel (marginal/logslope share a PC surface in
/// the η₁ row curvature), but the full survival row Hessian is 4×4 in
/// `(q0, q1, qd1, g)` and chains differently into each block, so it keeps the
/// channels distinct when no *true* alias exists. The reduced basis it emits
/// goes to Newton in place of the rank-deficient raw basis.
pub fn compiled_map_from_per_term(
    compiled: &SurvivalParametricCompiledPerTerm,
) -> gam_identifiability::families::compiler::CompiledMap {
    // Per-term V's and R's in global compile order: time terms, then marginal,
    // then logslope — exactly the order `r_lw_per_term` is stored in.
    let mut v_all: Vec<Array2<f64>> = Vec::new();
    v_all.extend(compiled.v_time_per_term.iter().cloned());
    v_all.extend(compiled.v_marginal_per_term.iter().cloned());
    v_all.extend(compiled.v_logslope_per_term.iter().cloned());

    let t_full = assemble_block_triangular_t(&v_all, &compiled.r_lw_per_term);

    // Per-block raw / compiled widths = summed per-term widths within the block.
    let raw_w = |terms: &[Array2<f64>]| -> usize { terms.iter().map(|v| v.nrows()).sum() };
    let kept_w = |terms: &[Array2<f64>]| -> usize { terms.iter().map(|v| v.ncols()).sum() };
    let raw_time = raw_w(&compiled.v_time_per_term);
    let raw_marg = raw_w(&compiled.v_marginal_per_term);
    let raw_log = raw_w(&compiled.v_logslope_per_term);
    let kept_time = kept_w(&compiled.v_time_per_term);
    let kept_marg = kept_w(&compiled.v_marginal_per_term);
    let kept_log = kept_w(&compiled.v_logslope_per_term);

    let raw_block_ranges = vec![
        0..raw_time,
        raw_time..(raw_time + raw_marg),
        (raw_time + raw_marg)..(raw_time + raw_marg + raw_log),
    ];
    let compiled_block_ranges = vec![
        0..kept_time,
        kept_time..(kept_time + kept_marg),
        (kept_time + kept_marg)..(kept_time + kept_marg + kept_log),
    ];

    gam_identifiability::families::compiler::CompiledMap {
        raw_from_compiled: t_full,
        compiled_block_ranges,
        raw_block_ranges,
    }
}

/// Build a W-orthogonal **partial** reduced-logslope reparameterisation `T`
/// (`p_log × r`, `0 < r < p_log`) for the survival marginal↔logslope confound,
/// mirroring the proven-correct BMS effective-Schur-Gram construction
/// [`crate::bms::block_specs::reduced_logslope_transform_effective`] but in
/// survival's per-row 4×4 primary-state Hessian metric.
///
/// # Why this exists (#979)
///
/// The survival marginal and logslope channels share the SAME spatial basis
/// (e.g. `matern(PC1,PC2,PC3)`), so on clustered-PC data the full 4×4
/// row-Hessian identifiability compiler can attribute the *entire* shared
/// surface to the lowest-priority logslope block and collapse it to zero width
/// — which the `#741` required-channel guard rejects, forcing a fallback to the
/// UNREDUCED design + Jeffreys conditioning. That fallback leaves a
/// quadratically-flat near-null direction in the joint penalised Hessian
/// `M = JᵀHJ + S`, so the inner joint-Newton cannot certify stationarity and
/// runs to its bounded iteration cap without converging.
///
/// The BMS path never hits this because it does a *partial* reduction: it
/// removes from the logslope block ONLY the directions whose effective image is
/// W-explained by the marginal span (the confounded null space of the effective
/// Schur Gram), keeping every surviving logslope direction. The result is
/// full-rank `M` BY CONSTRUCTION — no runtime projection needed.
///
/// The metric is assembled from the true vector operators:
///
/// ```text
///     W_m = stack_i sqrt(H_i) J_m,i
///     W_g = stack_i sqrt(H_i) J_g,i
///     A = W_mᵀ W_m + εI
///     B = W_mᵀ W_g
///     C = W_gᵀ W_g
///     Gtt = C − Bᵀ A⁻¹ B                 (p_log × p_log, PSD Schur complement)
/// ```
///
/// This remains exact for shared and per-score layouts, arbitrary score
/// covariance, and coefficient transforms that mix raw physical channels.
pub(crate) fn survival_reduced_logslope_transform_effective(
    marginal_dq: ndarray::ArrayView2<'_, f64>,
    logslope_layout: &crate::survival::marginal_slope::LogslopeLayout,
    row_hess: &SurvivalRowHessian,
) -> Result<crate::bms::block_specs::ReducedLogslopeOutcome, String> {
    use crate::bms::block_specs::{LOGSLOPE_REDUCED_BASIS_RELATIVE_TOL, ReducedLogslopeOutcome};
    use gam_linalg::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback, fast_atb};

    let n = marginal_dq.nrows();
    let p_m = marginal_dq.ncols();
    let score_dim = row_hess.k().checked_sub(3).ok_or_else(|| {
        "survival reduced logslope row Hessian omits required q channels".to_string()
    })?;
    if score_dim == 0 {
        return Err("survival reduced logslope requires at least one score channel".to_string());
    }
    logslope_layout.validate_for(score_dim)?;
    let p_log = logslope_layout.coefficient_design().ncols();
    if p_m == 0 || p_log == 0 {
        return Ok(ReducedLogslopeOutcome::FullRank);
    }
    if logslope_layout.coefficient_design().nrows() != n || row_hess.h.shape()[0] != n {
        return Err(format!(
            "survival reduced logslope: row mismatch marginal={n}, logslope={}, row_hess={}",
            logslope_layout.coefficient_design().nrows(),
            row_hess.h.shape()[0],
        ));
    }

    let marg = marginal_dq.to_owned();
    let marginal_operator =
        QChannelBlockOperator::new(marg.clone(), Array2::<f64>::zeros(marg.dim()), score_dim);
    let logslope_operator = LogslopeBlockOperator::new(logslope_layout.clone(), score_dim)?;
    let marginal_scaled = marginal_operator.scaled_design_by_sqrt_h(&row_hess.h);
    let logslope_scaled = logslope_operator.scaled_design_by_sqrt_h(&row_hess.h);

    // C = G_effᵀ W G_eff (raw-coordinate effective logslope Gram); its diagonal
    // sets the energy scale for the relative kept-direction tolerance.
    let c_gram = fast_atb(&logslope_scaled, &logslope_scaled);
    let energy_scale = (0..p_log).map(|i| c_gram[[i, i]]).fold(0.0_f64, f64::max);
    if !energy_scale.is_finite() {
        return Err(
            "survival reduced logslope: non-finite effective logslope energy scale".to_string(),
        );
    }
    if energy_scale <= 0.0 {
        // Zero effective logslope energy: the block carries no curvature at
        // all — every direction is unidentified (#2245 finding 45 sibling).
        return Ok(ReducedLogslopeOutcome::FullyConfounded);
    }

    // A = M_effᵀ W M_eff + εI (ridge relative to the marginal effective energy
    // so the Schur solve is well-posed even when the marginal pilot Gram is
    // rank-soft; the ridge only under-removes, i.e. is conservative).
    let mut a_gram = fast_atb(&marginal_scaled, &marginal_scaled);
    let a_scale = (0..p_m).map(|i| a_gram[[i, i]]).fold(0.0_f64, f64::max);
    let a_ridge = (a_scale * LOGSLOPE_REDUCED_BASIS_RELATIVE_TOL).max(f64::EPSILON);
    for i in 0..p_m {
        a_gram[[i, i]] += a_ridge;
    }

    // B = M_effᵀ W G_eff (p_m × p_log);  Gtt = C − Bᵀ A⁻¹ B (p_log × p_log, PSD).
    let b_cross = fast_atb(&marginal_scaled, &logslope_scaled);
    let a_view = FaerArrayView::new(&a_gram);
    let a_factor = factorize_symmetricwith_fallback(a_view.as_ref(), Side::Lower).map_err(|e| {
        format!("survival reduced logslope: marginal effective Gram factorization failed: {e}")
    })?;
    let b_view = FaerArrayView::new(&b_cross);
    let solved = a_factor.solve(b_view.as_ref()); // A⁻¹ B  (p_m × p_log)
    let a_inv_b = Array2::from_shape_fn((p_m, p_log), |(i, j)| solved[(i, j)]);
    let schur = fast_atb(&b_cross, &a_inv_b); // Bᵀ A⁻¹ B  (p_log × p_log)
    let mut stt = &c_gram - &schur;
    stt = (&stt + &stt.t()) * 0.5;
    if stt.iter().any(|v| !v.is_finite()) {
        return Err(
            "survival reduced logslope: effective Schur Gram produced non-finite entries"
                .to_string(),
        );
    }

    let (evals, evecs) = stt
        .eigh(Side::Lower)
        .map_err(|e| format!("survival reduced logslope: eigendecomposition failed: {e:?}"))?;
    // A `Gtt` eigenvalue far below the effective logslope energy scale means that
    // direction's effective logslope column is W-explained by the effective
    // marginal span — exactly the joint-Hessian rank-soft confounded direction.
    let tol = energy_scale * LOGSLOPE_REDUCED_BASIS_RELATIVE_TOL;
    let mut kept: Vec<usize> = (0..evals.len()).filter(|&i| evals[i] > tol).collect();
    kept.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let r = kept.len();
    // r == p_log: no confounded direction to remove — keep the raw design.
    // r == 0: the whole effective logslope image is W-explained by the
    // marginal span — the block is unidentified. These are OPPOSITE cases and
    // must not share a signal (#2245 finding 45 sibling): keeping the raw
    // columns for a fully confounded block lets the penalty pick an arbitrary
    // decomposition and report it as an estimate.
    if r == p_log {
        return Ok(ReducedLogslopeOutcome::FullRank);
    }
    if r == 0 {
        return Ok(ReducedLogslopeOutcome::FullyConfounded);
    }
    let mut transform = Array2::<f64>::zeros((p_log, r));
    for (out_col, &src) in kept.iter().enumerate() {
        transform.column_mut(out_col).assign(&evecs.column(src));
    }
    if transform.iter().any(|v| !v.is_finite()) {
        return Err(
            "survival reduced logslope: reduced transform produced non-finite entries".to_string(),
        );
    }
    Ok(ReducedLogslopeOutcome::Reduced(transform))
}

/// Assemble a block-diagonal 3-block [`CompiledMap`] that passes the time and
/// marginal blocks through unchanged (identity) and reparameterises ONLY the
/// logslope block via `t_log` (`p_log × r`). Used by the survival `#979`
/// partial reduced-logslope confound removal
/// ([`survival_reduced_logslope_transform_effective`]): the marginal/time
/// channels are untouched, the logslope block drops only its confounded
/// directions, and the joint penalised Hessian is full-rank by construction.
///
/// The resulting `CompiledMap` is interchangeable with one from
/// [`compiled_map_from_per_term`] /
/// [`gam_identifiability::families::compiler::compile_from_raw_grams`], so the
/// existing [`apply_compiled_map_to_designs`] + [`Gauge::from_compiled_map`]
/// machinery consumes it unchanged. Because the map is block-diagonal there is
/// no strict-upper cross-block residual `R`, and `apply_compiled_map_to_designs`
/// reads only the per-block diagonal `V_b = T[raw_b, compiled_b]` — `V_time` and
/// `V_marg` are identities, `V_log = t_log`.
pub fn survival_block_diagonal_logslope_map(
    p_time: usize,
    p_marg: usize,
    t_log: &Array2<f64>,
) -> gam_identifiability::families::compiler::CompiledMap {
    let p_log = t_log.nrows();
    let r = t_log.ncols();
    let raw_total = p_time + p_marg + p_log;
    let compiled_total = p_time + p_marg + r;
    let mut t_full = Array2::<f64>::zeros((raw_total, compiled_total));
    for i in 0..p_time {
        t_full[[i, i]] = 1.0;
    }
    for i in 0..p_marg {
        t_full[[p_time + i, p_time + i]] = 1.0;
    }
    for ri in 0..p_log {
        for cj in 0..r {
            t_full[[p_time + p_marg + ri, p_time + p_marg + cj]] = t_log[[ri, cj]];
        }
    }
    gam_identifiability::families::compiler::CompiledMap {
        raw_from_compiled: t_full,
        compiled_block_ranges: vec![
            0..p_time,
            p_time..(p_time + p_marg),
            (p_time + p_marg)..compiled_total,
        ],
        raw_block_ranges: vec![
            0..p_time,
            p_time..(p_time + p_marg),
            (p_time + p_marg)..raw_total,
        ],
    }
}

/// Apply a global [`CompiledMap`] T directly to the three survival
/// parametric block designs (time/marginal/logslope). Slices the
/// per-block diagonal of T into `V_b = T[raw_range_b, compiled_range_b]`
/// (shape `p_b_raw × w_b_compiled`), wraps each channel's raw design via
/// [`wrap_design_with_transform`], and pulls each block's penalties back
/// through that block's OWN `V_b` via
/// [`pull_back_blockwise_penalty_through_block_v`], producing
/// per-block-width `(w_b_compiled × w_b_compiled)` penalties — the shape
/// a per-block `ParameterBlockSpec.penalties` slot requires.
///
/// `map.raw_block_ranges` must equal three contiguous ranges in the
/// order Time → Marginal → Logslope (matching the input designs).
/// `map.compiled_block_ranges` runs in the same order.
///
/// Penalties supplied to this function:
/// - `time_penalties` are `BlockwisePenalty`s whose `col_range` is in
///   the time block's local raw coords (e.g. `0..p_time`).
/// - `marginal_penalties` / `logslope_penalties` likewise — local to
///   their own channel's raw width.
///
/// Each penalty's block-local `col_range` is embedded into the block's
/// raw width and pulled back as `V_bᵀ S_b V_b`. The cross-block
/// residualisation `R_{a→b}` carried in T's strict-upper triangle is
/// absorbed into the residualised *design* columns, not the penalty, so
/// the per-block penalty model stays exact for the highest-priority
/// block (time, no anchor → `R = []`) and matches the sibling per-block
/// compile path for the rest.
pub fn apply_compiled_map_to_designs(
    map: &gam_identifiability::families::compiler::CompiledMap,
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    time_penalties: &[gam_terms::smooth::BlockwisePenalty],
    marginal_penalties: &[gam_terms::smooth::BlockwisePenalty],
    logslope_penalties: &[gam_terms::smooth::BlockwisePenalty],
) -> Result<CompiledSurvivalDesignsVMExact, String> {
    if map.raw_block_ranges.len() != 3 || map.compiled_block_ranges.len() != 3 {
        return Err(format!(
            "apply_compiled_map_to_designs: expected exactly 3 blocks (time, marginal, logslope), \
             got {} raw / {} compiled",
            map.raw_block_ranges.len(),
            map.compiled_block_ranges.len(),
        ));
    }

    let time_raw = map.raw_block_ranges[0].clone();
    let marg_raw = map.raw_block_ranges[1].clone();
    let log_raw = map.raw_block_ranges[2].clone();
    let time_compiled = map.compiled_block_ranges[0].clone();
    let marg_compiled = map.compiled_block_ranges[1].clone();
    let log_compiled = map.compiled_block_ranges[2].clone();

    let t = &map.raw_from_compiled;
    let raw_total = t.nrows();
    let compiled_total = t.ncols();
    let expected_raw_total = log_raw.end;
    if raw_total != expected_raw_total {
        return Err(format!(
            "apply_compiled_map_to_designs: T has {raw_total} raw rows but block ranges sum to \
             {expected_raw_total}"
        ));
    }
    let expected_compiled_total = log_compiled.end;
    if compiled_total != expected_compiled_total {
        return Err(format!(
            "apply_compiled_map_to_designs: T has {compiled_total} compiled cols but block ranges \
             sum to {expected_compiled_total}"
        ));
    }
    let validate_partition =
        |ranges: &[std::ops::Range<usize>], total: usize, coordinate: &str| -> Result<(), String> {
            let mut expected_start = 0usize;
            for (block_idx, range) in ranges.iter().enumerate() {
                if range.start != expected_start || range.end < range.start || range.end > total {
                    return Err(format!(
                        "apply_compiled_map_to_designs: malformed {coordinate} block range \
                     {block_idx}={range:?}; expected a contiguous range starting at \
                     {expected_start} within 0..{total}"
                    ));
                }
                expected_start = range.end;
            }
            if expected_start != total {
                return Err(format!(
                    "apply_compiled_map_to_designs: {coordinate} block ranges end at \
                 {expected_start}, expected total {total}"
                ));
            }
            Ok(())
        };
    validate_partition(&map.raw_block_ranges, raw_total, "raw")?;
    validate_partition(&map.compiled_block_ranges, compiled_total, "compiled")?;

    // The current survival runtime represents time, marginal, and logslope as
    // three semantically distinct coefficient blocks.  It can therefore apply
    // an independent coefficient-side transform inside each block, but it
    // cannot represent a compiled column that simultaneously changes an
    // earlier block's primary channels.  A general compiler map is block-upper
    // triangular: its strict-upper `-R` slabs make (for example) a logslope
    // coordinate also move the raw time/marginal coefficients.  Slicing only
    // the diagonal `V_b` slabs would silently drop that primary-state motion;
    // the corresponding exact penalty `Tᵀ S T` would also couple blocks.
    //
    // Until the family owns a joint compiled-primary layout and joint
    // penalties, accept exactly block-diagonal maps only.  This is an exact
    // structural gate, deliberately without a numerical tolerance: an
    // unrepresented nonzero coefficient is a model change, however small.
    for (raw_block_idx, raw_range) in map.raw_block_ranges.iter().enumerate() {
        for (compiled_block_idx, compiled_range) in map.compiled_block_ranges.iter().enumerate() {
            if raw_block_idx == compiled_block_idx {
                continue;
            }
            for raw_row in raw_range.clone() {
                for compiled_col in compiled_range.clone() {
                    let value = map.raw_from_compiled[[raw_row, compiled_col]];
                    if value != 0.0 {
                        return Err(format!(
                            "apply_compiled_map_to_designs: unsupported cross-block lift at \
                             raw block {raw_block_idx}, compiled block {compiled_block_idx}, \
                             T[{raw_row},{compiled_col}]={value}; survival's split-block runtime \
                             requires an exactly block-diagonal CompiledMap"
                        ));
                    }
                }
            }
        }
    }

    let v_time = t
        .slice(ndarray::s![time_raw.clone(), time_compiled.clone()])
        .to_owned();
    let v_marg = t
        .slice(ndarray::s![marg_raw.clone(), marg_compiled.clone()])
        .to_owned();
    let v_log = t
        .slice(ndarray::s![log_raw.clone(), log_compiled.clone()])
        .to_owned();

    let time_entry_out =
        wrap_design_with_transform(time_design_entry, &v_time, "compiled-map: time entry")?;
    let time_exit_out =
        wrap_design_with_transform(time_design_exit, &v_time, "compiled-map: time exit")?;
    let time_deriv_out = wrap_design_with_transform(
        time_design_derivative_exit,
        &v_time,
        "compiled-map: time derivative_exit",
    )?;
    let marg_out = wrap_design_with_transform(marginal_design, &v_marg, "compiled-map: marginal")?;
    let log_out = wrap_design_with_transform(logslope_design, &v_log, "compiled-map: logslope")?;

    // Pull each block's penalties back through that block's OWN diagonal
    // reparameterisation V_b (= the (b, b) block of T). This produces a
    // per-block-width `(w_b_compiled × w_b_compiled)` penalty — the only
    // shape a per-block `ParameterBlockSpec.penalties` slot accepts.
    //
    // The block-local penalty `V_bᵀ S_b V_b` is the correct per-block
    // penalty: in raw coords the model penalises `γ_bᵀ S_b γ_b` on block
    // b's own coefficients, and under the residualised reparameterisation
    // the cross-block carry `R_{a→b}` lives entirely in the *design*
    // columns (`C_b V_b − A_{<b} R_b`), not in the penalty.
    //
    // Pulling penalties back through the full joint triangular T instead
    // (`Tᵀ blkdiag(S_b) T`) yields a `(p_compiled × p_compiled)` dense
    // matrix whose off-diagonal couples θ_b to earlier blocks' θ_a;
    // jamming that joint-width matrix into a single block's `penalties`
    // produced the `block 0 penalty 0 must be 12x12, got 17x17` mismatch
    // that surfaced as the `assert_valid_blockspecs` FFI panic. The two
    // agree whenever the residualisation `R_{a→b}` lands in the null space
    // of S_a (the shared low-order / parametric directions the identifiable
    // quotient strips), which is the case the compiler targets.
    let pull_set = |pens: &[gam_terms::smooth::BlockwisePenalty],
                    v_block: &Array2<f64>,
                    channel: &str|
     -> Result<Vec<PenaltyMatrix>, String> {
        pens.iter()
            .map(|p| {
                pull_back_blockwise_penalty_through_block_v(p, v_block).map_err(|e| {
                    format!("apply_compiled_map_to_designs: {channel} penalty pullback: {e}")
                })
            })
            .collect()
    };

    let time_penalties = pull_set(time_penalties, &v_time, "time")?;
    let marginal_penalties = pull_set(marginal_penalties, &v_marg, "marginal")?;
    let logslope_penalties = pull_set(logslope_penalties, &v_log, "logslope")?;
    validate_block_penalty_shapes("time", time_exit_out.ncols(), &time_penalties)?;
    validate_block_penalty_shapes("marginal", marg_out.ncols(), &marginal_penalties)?;
    validate_block_penalty_shapes("logslope", log_out.ncols(), &logslope_penalties)?;

    Ok(CompiledSurvivalDesignsVMExact {
        time_design_entry: time_entry_out,
        time_design_exit: time_exit_out,
        time_design_derivative_exit: time_deriv_out,
        marginal_design: marg_out,
        logslope_design: log_out,
        logslope_current_from_raw: v_log,
        time_penalties,
        marginal_penalties,
        logslope_penalties,
    })
}

fn validate_block_penalty_shapes(
    block: &str,
    width: usize,
    penalties: &[PenaltyMatrix],
) -> Result<(), String> {
    for (idx, penalty) in penalties.iter().enumerate() {
        let shape = penalty.shape();
        if shape != (width, width) {
            return Err(format!(
                "apply_compiled_map_to_designs: {block} penalty {idx} must be {width}x{width}, got {}x{}",
                shape.0, shape.1
            ));
        }
    }
    Ok(())
}

/// Run the identifiability compiler on the three survival parametric
/// blocks (time, marginal, logslope) at a pilot β and return the per-
/// block V reparameterisation matrices.
///
/// `row_hess` must be a PSD per-row 4×4 Hessian of `−log L_i(u_i)` at
/// the pilot β (see [`SurvivalRowHessian::from_pilot_primary_state`]).
/// The compiler residualises blocks left-to-right in priority order
/// (time → marginal → logslope) in the sqrt-H-metric so any aliased
/// direction lands in the lower-priority block, then runs a post-walk
/// column-pivoted QR on the cumulative anchor and drops trailing
/// pivots from the latest block. The returned V matrices are ready to
/// be applied to each block's raw design and penalty before the
/// `ParameterBlockSpec` list is assembled.
///
/// On `FullyAliased` from `compile()` (a block fully absorbed by its
/// cumulative anchor) this returns `Err`. The construction site should
/// surface that as a structured user-facing diagnostic — the model is
/// asking the compiler to assign zero degrees of freedom to a named
/// parametric block, which is a model-spec bug not a numerical one.
///
/// Sibling Phase-4b wiring (`bernoulli_marginal_slope::install_compiled_flex_block_into_runtime`)
/// already calls `compile()` for the flex blocks. This helper extends
/// that contract to the parametric blocks by giving the SMGS
/// construction site a one-line entry point — it does NOT yet apply
/// the V transforms to the family's captured designs (the captured-
/// design update is the remaining integration step that touches the
/// family's row-Hessian assembly assertions).
pub(crate) fn compile_survival_parametric_designs(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    logslope_layout: crate::survival::marginal_slope::LogslopeLayout,
    row_hess: &dyn RowHessian,
) -> Result<SurvivalParametricCompiled, String> {
    use gam_identifiability::families::compiler::compile;

    let p_time_raw = time_dq0.ncols();
    let p_marg_raw = marginal_dq.ncols();
    let score_dim = row_hess.k().checked_sub(3).ok_or_else(|| {
        format!(
            "compile_survival_parametric_designs row Hessian width {} is smaller than the three q channels",
            row_hess.k()
        )
    })?;
    if score_dim == 0 {
        return Err(
            "compile_survival_parametric_designs requires at least one score channel".to_string(),
        );
    }
    logslope_layout.validate_for(score_dim)?;
    let p_log_raw = logslope_layout.coefficient_design().ncols();

    let inputs = build_survival_compiler_inputs(
        time_dq0,
        time_dq1,
        time_dqd1,
        marginal_dq,
        marginal_dqd1,
        logslope_layout,
        score_dim,
        None,
        None,
    )?;
    if inputs.operators.len() != 3 {
        return Err(format!(
            "compile_survival_parametric_designs: expected exactly 3 parametric operators \
             (time, marginal, logslope); got {}",
            inputs.operators.len(),
        ));
    }
    let compiled = compile(&inputs.operators, row_hess, &inputs.ordering)
        .map_err(|e| format!("identifiability::families::compiler::compile failed: {e}"))?;
    if compiled.blocks.len() != 3 {
        return Err(format!(
            "compile_survival_parametric_designs: compiler emitted {} blocks; expected 3",
            compiled.blocks.len(),
        ));
    }
    let v_time = compiled.blocks[0].t_lw.clone();
    let v_marginal = compiled.blocks[1].t_lw.clone();
    let v_logslope = compiled.blocks[2].t_lw.clone();
    let drops_by_block = (
        p_time_raw.saturating_sub(v_time.ncols()),
        p_marg_raw.saturating_sub(v_marginal.ncols()),
        p_log_raw.saturating_sub(v_logslope.ncols()),
    );
    Ok(SurvivalParametricCompiled {
        v_time,
        v_marginal,
        v_logslope,
        drops_by_block,
    })
}

/// Build the operator stack from already-materialised dense designs.
///
/// `time_dq0/dq1/dqd1` are the time block's three primary-state Jacobians
/// at training rows. `marginal_dq` and `marginal_dqd1` are the marginal
/// block's contributions to q (shared between q0 and q1) and to qd1
/// (typically zero unless timewiggle interacts). `logslope_dg` is the
/// logslope block's contribution to g.
///
/// `score_warp_(dq, dqd1)` / `link_dev_(dq, dqd1)` are present only when
/// the corresponding flex block is active. The returned `ordering` parallels
/// `operators` so the caller can route compiled outputs back to runtime slots.
pub(crate) fn build_survival_compiler_inputs(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    logslope_layout: crate::survival::marginal_slope::LogslopeLayout,
    score_dim: usize,
    score_warp_dq_dqd1: Option<(Array2<f64>, Array2<f64>)>,
    link_dev_dq_dqd1: Option<(Array2<f64>, Array2<f64>)>,
) -> Result<SurvivalCompilerInputs, String> {
    if score_dim == 0 {
        return Err("survival compiler inputs require at least one score channel".to_string());
    }
    logslope_layout.validate_for(score_dim)?;
    let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::with_capacity(5);
    let mut ordering: Vec<BlockOrder> = Vec::with_capacity(5);

    operators.push(Arc::new(TimeBlockOperator::new(
        time_dq0, time_dq1, time_dqd1, score_dim,
    )));
    ordering.push(BlockOrder::Time);

    operators.push(Arc::new(QChannelBlockOperator::new(
        marginal_dq,
        marginal_dqd1,
        score_dim,
    )));
    ordering.push(BlockOrder::Marginal);

    operators.push(Arc::new(LogslopeBlockOperator::new(
        logslope_layout,
        score_dim,
    )?));
    ordering.push(BlockOrder::Logslope);

    if let Some((dq, dqd1)) = score_warp_dq_dqd1 {
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1, score_dim)));
        ordering.push(BlockOrder::ScoreWarp);
    }
    if let Some((dq, dqd1)) = link_dev_dq_dqd1 {
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1, score_dim)));
        ordering.push(BlockOrder::LinkDev);
    }

    Ok(SurvivalCompilerInputs {
        operators,
        ordering,
    })
}

/// V+M-exact compiled designs + per-block penalties for the survival
/// time/marginal/logslope blocks, produced by
/// [`apply_compiled_map_to_designs`] from a `CompiledMap`. The
/// construction site swaps raw designs/penalties for these compiled
/// versions before building `ParameterBlockSpec`s.
///
/// The emitted designs carry the exact residualised `C_b·V_b − A_{<b}·R_b`
/// row form (via [`wrap_design_with_transform`] on `V_b = T[raw_b, comp_b]`):
/// the cross-block residualisation `R_{a→b}` lives in those design columns,
/// while each block's penalty is pulled back through that block's own
/// diagonal `V_b` as `V_bᵀ S_b V_b` (the `*_penalties` fields).
///
/// At fit result the joint compiled β is lifted back to raw via the
/// `gam_solve::gauge::Gauge` built from the *same* `CompiledMap`
/// (`β_raw = T · θ`, T block-upper-triangular with `V_b` on the diagonal
/// and `-R_{a→b}` off-diagonal). The full T therefore lives on that
/// `Gauge`, not on this struct — the caller holds the `CompiledMap` and
/// constructs both from it, so duplicating T here would be dead state.
pub struct CompiledSurvivalDesignsVMExact {
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    /// Exact block-local map from raw log-slope coefficients to the current
    /// compiled coordinates. Physical per-score channels use this same map to
    /// emit full-width current-coordinate Jacobian rows.
    pub logslope_current_from_raw: Array2<f64>,
    /// Per-block penalties, each pulled back through that block's OWN
    /// diagonal reparameterisation `V_b` as `V_bᵀ S_b V_b`. The result
    /// is a per-block-width `PenaltyMatrix::Dense`
    /// (`w_b_compiled × w_b_compiled`) — the shape a per-block
    /// `ParameterBlockSpec.penalties` slot requires. Cross-block
    /// residualisation `R_{a→b}` is carried by the residualised design
    /// columns, not the penalty.
    pub time_penalties: Vec<PenaltyMatrix>,
    pub marginal_penalties: Vec<PenaltyMatrix>,
    pub logslope_penalties: Vec<PenaltyMatrix>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survival::{LogslopeLayout, LogslopeTopology};
    use gam_problem::Gauge;
    use ndarray::array;

    const SINGLE_SCORE_PRIMARY_WIDTH: usize = 4;

    fn shared_logslope_layout(design: Array2<f64>) -> LogslopeLayout {
        let n = design.nrows();
        LogslopeTopology::shared()
            .materialize_identity(DesignMatrix::from(design), &Array1::<f64>::zeros(n))
            .unwrap()
    }

    /// Construct a synthetic tensor-backed `SurvivalRowHessian` for the unit
    /// tests below. Production instances must retain their real row data so a
    /// later `channel_hessian_at` refresh is exact — hence a test-module
    /// helper, not a production constructor.
    fn survival_row_hessian_from_full(h: Array3<f64>) -> SurvivalRowHessian {
        assert_eq!(h.shape()[1], h.shape()[2]);
        assert!(h.shape()[1] > 3);
        let n = h.shape()[0];
        let score_dim = h.shape()[1] - 3;
        SurvivalRowHessian {
            h,
            weights: Array1::ones(n),
            event: Array1::ones(n),
            z: Array2::zeros((n, score_dim)),
            covariance: crate::bms::MarginalSlopeCovariance::diagonal(Array1::ones(score_dim))
                .unwrap(),
            derivative_guard:
                crate::survival::marginal_slope::DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
        }
    }

    #[test]
    fn psd_clamp_zeros_negative_eigenvalues() {
        // Construct M = U diag(2, -1, 0.5, -0.25) Uᵀ for a fixed U from
        // a small rotation, verify the clamped matrix has eigenvalues
        // (2, 0, 0.5, 0).
        let mut m = Array2::<f64>::zeros((4, 4));
        // Diagonal with mixed signs is sufficient for the test: the
        // eigenvalues equal the diagonal and the eigenvectors are e_i.
        m[[0, 0]] = 2.0;
        m[[1, 1]] = -1.0;
        m[[2, 2]] = 0.5;
        m[[3, 3]] = -0.25;
        let clamped = psd_clamp(&m).expect("finite 4x4 eigendecomposition must succeed");
        assert!((clamped[[0, 0]] - 2.0).abs() < 1e-12);
        assert!(clamped[[1, 1]].abs() < 1e-12);
        assert!((clamped[[2, 2]] - 0.5).abs() < 1e-12);
        assert!(clamped[[3, 3]].abs() < 1e-12);
    }

    #[test]
    fn time_block_operator_evaluate_full_shape() {
        let n = 6;
        let p = 3;
        let dq0 = Array2::from_shape_fn((n, p), |(i, j)| (i + j) as f64);
        let dq1 = Array2::from_shape_fn((n, p), |(i, j)| (i as f64) * 2.0 + j as f64);
        let dqd1 = Array2::from_shape_fn((n, p), |(i, j)| 0.5 * ((i * j) as f64));
        let op = TimeBlockOperator::new(dq0.clone(), dq1.clone(), dqd1.clone(), 1);
        let full = op.evaluate_full();
        assert_eq!(full.shape(), &[n, p, SINGLE_SCORE_PRIMARY_WIDTH]);
        for i in 0..n {
            for j in 0..p {
                assert_eq!(full[[i, j, 0]], dq0[[i, j]]);
                assert_eq!(full[[i, j, 1]], dq1[[i, j]]);
                assert_eq!(full[[i, j, 2]], dqd1[[i, j]]);
                assert_eq!(full[[i, j, 3]], 0.0);
            }
        }
    }

    #[test]
    fn q_channel_block_apply_row_shares_q0_q1() {
        let n = 5;
        let p = 2;
        let dq = Array2::from_shape_fn((n, p), |(i, j)| (i as f64) * (j as f64 + 1.0));
        let dqd1 = Array2::from_shape_fn((n, p), |(i, j)| (j as f64) - (i as f64));
        let op = QChannelBlockOperator::new(dq.clone(), dqd1.clone(), 1);
        let mut out = [0.0_f64; SINGLE_SCORE_PRIMARY_WIDTH];
        let delta = [1.0_f64, -0.5];
        op.apply_row(3, &delta, &mut out);
        let want_q = dq[[3, 0]] * 1.0 + dq[[3, 1]] * (-0.5);
        let want_qd = dqd1[[3, 0]] * 1.0 + dqd1[[3, 1]] * (-0.5);
        assert!((out[0] - want_q).abs() < 1e-12);
        assert!((out[1] - want_q).abs() < 1e-12);
        assert!((out[2] - want_qd).abs() < 1e-12);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn logslope_block_writes_only_g_channel() {
        let n = 4;
        let p = 2;
        let dg = Array2::from_shape_fn((n, p), |(i, j)| (i as f64) + 0.1 * (j as f64));
        let op = LogslopeBlockOperator::new(shared_logslope_layout(dg.clone()), 1).unwrap();
        let mut out = [0.0_f64; SINGLE_SCORE_PRIMARY_WIDTH];
        let delta = [2.0_f64, -1.0];
        op.apply_row(1, &delta, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        let want = dg[[1, 0]] * 2.0 + dg[[1, 1]] * (-1.0);
        assert!((out[3] - want).abs() < 1e-12);
    }

    #[test]
    fn per_score_identifiability_uses_three_plus_k_primary_geometry_932() {
        let n = 3;
        let design = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let layout = LogslopeTopology::per_score(vec![0..1, 1..2], 2)
            .unwrap()
            .materialize_identity(DesignMatrix::from(design.clone()), &Array1::<f64>::zeros(n))
            .unwrap();
        let operator = LogslopeBlockOperator::new(layout, 2).unwrap();

        let mut out = [f64::NAN; 5];
        operator.apply_row(1, &[2.0, -1.0], &mut out);
        assert_eq!(operator.k(), 5);
        assert_eq!(&out[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(out[3], 2.0 * design[[1, 0]]);
        assert_eq!(out[4], -design[[1, 1]]);

        let mut h = Array3::<f64>::zeros((n, 5, 5));
        for row in 0..n {
            for channel in 0..5 {
                h[[row, channel, channel]] = 1.0;
            }
        }
        let row_hessian = survival_row_hessian_from_full(h);
        assert_eq!(row_hessian.k(), 5);
        assert_eq!(row_hessian.n_outputs(), 5);
    }

    #[test]
    fn extract_term_partition_simple_cases() {
        let full = 0..5usize;
        // No penalties: whole block is one term.
        let part = extract_term_partition_from_penalty_ranges(5, &[]);
        assert_eq!(part.as_slice(), std::slice::from_ref(&full));
        // One penalty covering the whole block.
        let part = extract_term_partition_from_penalty_ranges(5, std::slice::from_ref(&full));
        assert_eq!(part.as_slice(), std::slice::from_ref(&full));
        // Two penalties with a gap: produces three terms (pen1, gap, pen2).
        let part = extract_term_partition_from_penalty_ranges(10, &[0..3, 6..10]);
        assert_eq!(part, vec![0..3, 3..6, 6..10]);
        // Duplicate penalty ranges coalesce.
        let part = extract_term_partition_from_penalty_ranges(6, &[0..3, 0..3, 3..6]);
        assert_eq!(part, vec![0..3, 3..6]);
        // Empty block.
        let part = extract_term_partition_from_penalty_ranges(0, &[]);
        assert!(part.is_empty());
    }

    #[test]
    fn assemble_block_triangular_t_identity_when_v_eye_and_r_none() {
        let v_a = Array2::<f64>::eye(2);
        let v_b = Array2::<f64>::eye(2);
        let t = assemble_block_triangular_t(&[v_a, v_b], &[None, None]);
        assert_eq!(t.dim(), (4, 4));
        let eye4 = Array2::<f64>::eye(4);
        for i in 0..4 {
            for j in 0..4 {
                assert!((t[[i, j]] - eye4[[i, j]]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn assemble_block_triangular_t_with_drops_and_nonzero_r() {
        let mut v_a = Array2::<f64>::zeros((3, 2));
        v_a[[0, 0]] = 1.0;
        v_a[[1, 0]] = 0.5;
        v_a[[2, 1]] = 1.0;
        let v_b = Array2::<f64>::eye(2);
        let r_ab =
            Array2::<f64>::from_shape_fn((3, 2), |(i, j)| 1.0 + (i as f64) + 0.25 * (j as f64));
        let t =
            assemble_block_triangular_t(&[v_a.clone(), v_b.clone()], &[None, Some(r_ab.clone())]);
        assert_eq!(t.dim(), (5, 4));
        for i in 0..3 {
            for j in 0..2 {
                assert!((t[[i, j]] - v_a[[i, j]]).abs() < 1e-14);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((t[[3 + i, 2 + j]] - v_b[[i, j]]).abs() < 1e-14);
            }
        }
        for i in 0..3 {
            for j in 0..2 {
                assert!((t[[i, 2 + j]] + r_ab[[i, j]]).abs() < 1e-14);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(t[[3 + i, j]], 0.0);
            }
        }
    }

    #[test]
    fn validate_partition_rejects_bad_partitions() {
        let bad_start = 1..5usize;
        let short_cover = 0..3usize;
        let full_cover = 0..5usize;
        // Doesn't start at 0.
        assert!(validate_partition(std::slice::from_ref(&bad_start), 5, "test").is_err());
        // Doesn't cover the block.
        assert!(validate_partition(std::slice::from_ref(&short_cover), 5, "test").is_err());
        // Has a gap.
        assert!(validate_partition(&[0..2, 3..5], 5, "test").is_err());
        // Has overlap.
        assert!(validate_partition(&[0..3, 2..5], 5, "test").is_err());
        // Has empty range.
        assert!(validate_partition(&[0..0, 0..5], 5, "test").is_err());
        // Empty block + empty partition OK.
        assert!(validate_partition(&[], 0, "test").is_ok());
        // Valid partition.
        assert!(validate_partition(&[0..2, 2..5], 5, "test").is_ok());
        assert!(validate_partition(std::slice::from_ref(&full_cover), 5, "test").is_ok());
    }

    /// A nonzero cross-block residual cannot be represented by the split-block
    /// survival runtime: the likelihood needs the omitted `-A R` primary-state
    /// motion and the exact penalty `Tᵀ S T` couples coefficient blocks.  The
    /// old test falsely blessed this map by checking only output widths and the
    /// diagonal `Vᵀ S V` slabs.  The construction boundary must reject it.
    #[test]
    fn compiled_map_with_nonzero_cross_block_residual_is_rejected() {
        use gam_identifiability::families::compiler::CompiledMap;
        use gam_terms::smooth::BlockwisePenalty;

        let n = 10;
        // Time raw 3 → compiled 3 (block 0: no anchor, V pure, R=None).
        // Marginal raw 3 → compiled 2 (a real drop, with nonzero R against time).
        // Logslope raw 2 → compiled 2 (nonzero R against time+marginal).
        let v_time =
            Array2::<f64>::from_shape_fn(
                (3, 3),
                |(i, j)| {
                    if i == j { 1.0 } else { 0.1 * ((i + j) as f64) }
                },
            );
        let v_marg = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| {
            0.5 + 0.3 * (i as f64) - 0.2 * (j as f64)
        });
        let v_log = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| if i == j { 1.2 } else { 0.4 });
        // R_marg: rows = time raw width 3, cols = marginal compiled width 2.
        let r_marg = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| 0.7 - 0.1 * ((i + j) as f64));
        // R_log: rows = time+marg RAW width 6 (3 + 3), cols = logslope compiled
        // width 2. `assemble_block_triangular_t` stacks R_{a→b} over a<b, so the row
        // count is the sum of the RAW widths of the prior blocks (not their
        // compiled widths — marginal's compiled width is 2 but its raw width is 3).
        let r_log =
            Array2::<f64>::from_shape_fn((6, 2), |(i, j)| 0.3 + 0.05 * ((i * 2 + j) as f64));

        let t = assemble_block_triangular_t(
            &[v_time.clone(), v_marg.clone(), v_log.clone()],
            &[None, Some(r_marg.clone()), Some(r_log.clone())],
        );
        assert_eq!(t.dim(), (8, 7), "joint raw 8 × joint compiled 7");

        let map = CompiledMap {
            raw_from_compiled: t.clone(),
            compiled_block_ranges: vec![0..3, 3..5, 5..7],
            raw_block_ranges: vec![0..3, 3..6, 6..8],
        };

        // Raw designs (dense, n rows).
        let raw_time_entry = DesignMatrix::Dense(DenseDesignMatrix::from(
            Array2::<f64>::from_shape_fn((n, 3), |(i, j)| 1.0 + (i as f64) * 0.1 + (j as f64)),
        ));
        let raw_time_exit = raw_time_entry.clone();
        let raw_time_deriv = raw_time_entry.clone();
        let raw_marg = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::from_shape_fn(
            (n, 3),
            |(i, j)| 0.2 * (i as f64) - 0.3 * (j as f64),
        )));
        let raw_log = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::from_shape_fn(
            (n, 2),
            |(i, j)| 0.5 + (i as f64) * (j as f64 + 1.0),
        )));

        // Block-local penalties (col_range relative to each block's first col).
        let s_time =
            Array2::<f64>::from_shape_fn(
                (3, 3),
                |(i, j)| if i == j { (i + 2) as f64 } else { 0.3 },
            );
        let s_marg =
            Array2::<f64>::from_shape_fn(
                (3, 3),
                |(i, j)| if i == j { 1.5 + i as f64 } else { 0.2 },
            );
        let s_log = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| if i == j { 2.0 } else { 0.5 });
        let time_pens = vec![BlockwisePenalty::new(0..3, s_time.clone())];
        let marg_pens = vec![BlockwisePenalty::new(0..3, s_marg.clone())];
        let log_pens = vec![BlockwisePenalty::new(0..2, s_log.clone())];

        let error = match apply_compiled_map_to_designs(
            &map,
            raw_time_entry,
            raw_time_exit,
            raw_time_deriv,
            raw_marg,
            raw_log,
            &time_pens,
            &marg_pens,
            &log_pens,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a triangular map cannot be represented by split blocks"),
        };
        assert!(error.contains("unsupported cross-block lift"), "{error}");
    }

    #[test]
    fn compiled_map_with_malformed_ranges_is_rejected_without_indexing() {
        use gam_identifiability::families::compiler::CompiledMap;

        let map = CompiledMap {
            raw_from_compiled: Array2::<f64>::eye(3),
            raw_block_ranges: vec![0..1, 2..2, 2..3],
            compiled_block_ranges: vec![0..1, 1..2, 2..3],
        };
        let design = || DesignMatrix::from(Array2::<f64>::ones((2, 1)));
        let error = match apply_compiled_map_to_designs(
            &map,
            design(),
            design(),
            design(),
            design(),
            design(),
            &[],
            &[],
            &[],
        ) {
            Err(error) => error,
            Ok(_) => panic!("a non-contiguous raw partition must be rejected"),
        };
        assert!(error.contains("malformed raw block range"), "{error}");
    }

    /// Top-level Phase-4b API test for the SMGS parametric path:
    /// call `compile_survival_parametric_designs` on a shared-constant
    /// alias between time and marginal, with an identity row Hessian.
    /// Verify the returned `v_*` matrices have the expected widths
    /// (time keeps all 3, marginal loses 1, logslope keeps both) and
    /// `drops_by_block` reports `(0, 1, 0)`.
    #[test]
    fn compile_survival_parametric_designs_helper_attributes_drop_to_marginal() {
        let n = 24;
        let p_time = 3;
        let p_marginal = 3;
        let p_logslope = 2;
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let mut time_dq0 = Array2::<f64>::zeros((n, p_time));
        let mut time_dq1 = Array2::<f64>::zeros((n, p_time));
        let mut time_dqd1 = Array2::<f64>::zeros((n, p_time));
        let mut marg_dq = Array2::<f64>::zeros((n, p_marginal));
        let marg_dqd1 = Array2::<f64>::zeros((n, p_marginal));
        let mut log_dg = Array2::<f64>::zeros((n, p_logslope));
        for i in 0..n {
            time_dq0[[i, 0]] = 1.0;
            time_dq0[[i, 1]] = x[i];
            time_dq0[[i, 2]] = x[i] * x[i];
            time_dq1[[i, 0]] = 1.0;
            time_dq1[[i, 1]] = x[i];
            time_dq1[[i, 2]] = x[i] * x[i];
            time_dqd1[[i, 0]] = 0.0;
            time_dqd1[[i, 1]] = 1.0;
            time_dqd1[[i, 2]] = 2.0 * x[i];
            marg_dq[[i, 0]] = 1.0; // alias with time col 0
            marg_dq[[i, 1]] = x[i] * x[i] * x[i];
            marg_dq[[i, 2]] = x[i].sin();
            log_dg[[i, 0]] = (2.0 * x[i]).cos();
            log_dg[[i, 1]] = x[i].tanh();
        }
        let mut h_full =
            Array3::<f64>::zeros((n, SINGLE_SCORE_PRIMARY_WIDTH, SINGLE_SCORE_PRIMARY_WIDTH));
        for i in 0..n {
            for k in 0..SINGLE_SCORE_PRIMARY_WIDTH {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = survival_row_hessian_from_full(h_full);
        let out = compile_survival_parametric_designs(
            time_dq0,
            time_dq1,
            time_dqd1,
            marg_dq,
            marg_dqd1,
            shared_logslope_layout(log_dg),
            &row_hess,
        )
        .expect("Phase-4b parametric compile must succeed on single-direction alias");
        assert_eq!(out.v_time.ncols(), p_time, "time keeps all columns");
        assert_eq!(
            out.v_marginal.ncols(),
            p_marginal - 1,
            "marginal loses exactly the shared-constant direction"
        );
        assert_eq!(out.v_logslope.ncols(), p_logslope, "logslope is clean");
        assert_eq!(
            out.drops_by_block,
            (0, 1, 0),
            "attribution: zero from time/logslope, one from marginal",
        );
    }

    /// End-to-end Phase-4b smoke test: build the full 3-block survival
    /// parametric operator stack (time + marginal + logslope) with a
    /// shared-constant alias seeded between the time and marginal
    /// blocks, feed it into `compile()` with an identity 4×4 row
    /// Hessian on every row, and verify the compiler:
    ///
    ///   (1) returns a [`CompiledBlocks`] with one block per input;
    ///   (2) preserves all 3 columns of the highest-priority `Time`
    ///       block in `t_lw` (the time block enters first in the
    ///       ordering, so its full column span survives);
    ///   (3) drops exactly one direction from `Marginal` (the
    ///       constant aliased with the time intercept), leaving its
    ///       remaining columns in `t_lw`;
    ///   (4) reports `joint_rank` = (raw_total - 1).
    ///
    /// This validates the Phase-4b construction-time orthogonalisation
    /// path on the survival K=4 row primary state and then feeds the
    /// compiled per-block reduced bases through the SMGS lift [`Gauge`]
    /// (step 6), asserting the lift's reduced/raw block structure agrees
    /// with the compiled rank-drop — the construction contract end to end.
    #[test]
    fn compile_survival_three_block_with_shared_constant_drops_one_direction() {
        use gam_identifiability::families::compiler::compile;

        let n = 32;
        let p_time = 3;
        let p_marginal = 3;
        let p_logslope = 2;

        // Time block:
        //   col 0 = ones (the shared constant — aliases marginal col 0);
        //   col 1 = linear x;
        //   col 2 = quadratic x².
        // q0/q1 share the same design (so the alias surfaces in both
        // the entry and exit primary channels); qd1 is the derivative
        // of the design w.r.t. time at the exit point, which for the
        // constant column is exactly zero (the gauge identity that
        // makes the constant a true null direction under (q0, q1, qd1)
        // joint).
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let mut time_dq0 = Array2::<f64>::zeros((n, p_time));
        let mut time_dq1 = Array2::<f64>::zeros((n, p_time));
        let mut time_dqd1 = Array2::<f64>::zeros((n, p_time));
        for i in 0..n {
            time_dq0[[i, 0]] = 1.0;
            time_dq0[[i, 1]] = x[i];
            time_dq0[[i, 2]] = x[i] * x[i];
            time_dq1[[i, 0]] = 1.0;
            time_dq1[[i, 1]] = x[i];
            time_dq1[[i, 2]] = x[i] * x[i];
            // d/dt of a constant = 0; d/dt of x ≡ 1; d/dt of x² ≡ 2x.
            time_dqd1[[i, 0]] = 0.0;
            time_dqd1[[i, 1]] = 1.0;
            time_dqd1[[i, 2]] = 2.0 * x[i];
        }

        // Marginal block (q-channel only; qd1 contribution zero — no
        // timewiggle in this scenario):
        //   col 0 = ones (the shared constant);
        //   col 1 = x³;
        //   col 2 = sin(x).
        let mut marg_dq = Array2::<f64>::zeros((n, p_marginal));
        let marg_dqd1 = Array2::<f64>::zeros((n, p_marginal));
        for i in 0..n {
            marg_dq[[i, 0]] = 1.0;
            marg_dq[[i, 1]] = x[i] * x[i] * x[i];
            marg_dq[[i, 2]] = x[i].sin();
        }

        // Logslope block (g-channel only):
        //   col 0 = cos(2x);
        //   col 1 = tanh(x).  (no shared constant — logslope is clean)
        let mut log_dg = Array2::<f64>::zeros((n, p_logslope));
        for i in 0..n {
            log_dg[[i, 0]] = (2.0 * x[i]).cos();
            log_dg[[i, 1]] = x[i].tanh();
        }

        let inputs = build_survival_compiler_inputs(
            time_dq0,
            time_dq1,
            time_dqd1,
            marg_dq,
            marg_dqd1,
            shared_logslope_layout(log_dg),
            1,
            None,
            None,
        )
        .expect("valid shared logslope layout must build compiler inputs");

        // Identity 4×4 row Hessian on every row. With H_i = I the
        // sqrt-H metric collapses to the standard Frobenius metric,
        // so the compiler's residualisation is ordinary least-squares
        // projection — exactly what we want for verifying the
        // structural rank-deficiency attribution.
        let mut h_full =
            Array3::<f64>::zeros((n, SINGLE_SCORE_PRIMARY_WIDTH, SINGLE_SCORE_PRIMARY_WIDTH));
        for i in 0..n {
            for k in 0..SINGLE_SCORE_PRIMARY_WIDTH {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = survival_row_hessian_from_full(h_full);

        let compiled = compile(&inputs.operators, &row_hess, &inputs.ordering)
            .expect("survival 3-block compile must succeed; aliasing is single-direction");

        // (1) One CompiledBlock per input.
        assert_eq!(compiled.blocks.len(), 3, "expected 3 CompiledBlocks");

        // (2) Time enters first; under sqrt-I metric every column of
        // the time block is residual-vs-empty-anchor and therefore
        // survives the eigendecomposition with positive eigenvalue.
        // V_time has p_time columns.
        let v_time = &compiled.blocks[0].t_lw;
        assert_eq!(
            v_time.ncols(),
            p_time,
            "time block (first in ordering) must retain all {p_time} of its columns; V_time={:?}",
            v_time.dim(),
        );

        // (3) Marginal enters second. Its constant column is aliased
        // with time's constant column in (q0, q1) and contributes zero
        // to qd1. After residualising against the time anchor in the
        // K=4 stacked metric, the residual Gram has rank
        // p_marginal − 1 (one direction collapsed by the alias). So
        // V_marginal has exactly (p_marginal − 1) columns.
        let v_marg = &compiled.blocks[1].t_lw;
        assert_eq!(
            v_marg.ncols(),
            p_marginal - 1,
            "marginal block must lose exactly the shared-constant direction; \
             V_marginal cols = {}, expected {}",
            v_marg.ncols(),
            p_marginal - 1,
        );

        // (4) Logslope enters third and carries no shared direction
        // with time or marginal in the g-channel. Both columns survive.
        let v_log = &compiled.blocks[2].t_lw;
        assert_eq!(
            v_log.ncols(),
            p_logslope,
            "logslope block (no shared direction) must retain all {p_logslope} columns",
        );

        // (5) Joint rank consistency: sum of compiled column counts
        // equals raw_total minus the one aliased direction.
        let raw_total = p_time + p_marginal + p_logslope;
        let kept_total: usize = compiled.blocks.iter().map(|b| b.t_lw.ncols()).sum();
        assert_eq!(
            kept_total,
            raw_total - 1,
            "joint kept = raw_total − aliased; got {kept_total}, expected {}",
            raw_total - 1,
        );
        assert_eq!(
            compiled.joint_rank, kept_total,
            "CompiledBlocks::joint_rank must match the sum of per-block t_lw widths",
        );

        // (6) SMGS construction contract. Feed the compiled per-block reduced
        // bases (V_k = t_lw, shaped raw_k × kept_k) into the SMGS lift `Gauge`
        // and verify the lift's coordinate bookkeeping matches the compiler's
        // rank attribution: the reduced dimension equals `joint_rank`, the
        // reduced block boundaries advance by each block's kept width, and —
        // with R = None (no residualised cross-block reparam in this V-only
        // construction) — the raw block boundaries advance by each block's raw
        // width. This exercises the SMGS construction hook directly on the
        // compiled output rather than asserting against a hypothetical shape.
        let v_per_term: Vec<Array2<f64>> = compiled.blocks.iter().map(|b| b.t_lw.clone()).collect();
        let r_per_term: Vec<Option<Array2<f64>>> = vec![None; v_per_term.len()];
        let gauge = Gauge::from_v_and_r(&v_per_term, &r_per_term);

        let mut expected_reduced = vec![0usize];
        let mut expected_raw = vec![0usize];
        for b in &compiled.blocks {
            let prev_reduced = *expected_reduced.last().unwrap();
            expected_reduced.push(prev_reduced + b.t_lw.ncols());
            let prev_raw = *expected_raw.last().unwrap();
            expected_raw.push(prev_raw + b.t_lw.nrows());
        }
        assert_eq!(
            *gauge.block_starts_reduced.last().unwrap(),
            compiled.joint_rank,
            "SMGS lift reduced dimension must equal the compiled joint_rank",
        );
        assert_eq!(
            gauge.block_starts_reduced, expected_reduced,
            "SMGS lift reduced block boundaries must match the compiled kept widths",
        );
        assert_eq!(
            gauge.block_starts_raw, expected_raw,
            "SMGS lift raw block boundaries must match the compiled per-block raw widths",
        );

        // (7) Every kept direction is finite and non-degenerate. A retained
        // column with a zero or non-finite norm would be a spurious rank
        // contribution that the count-only checks above cannot catch, so verify
        // each compiled block's surviving directions directly.
        for (bi, block) in compiled.blocks.iter().enumerate() {
            for j in 0..block.t_lw.ncols() {
                let col = block.t_lw.column(j);
                assert!(
                    col.iter().all(|v| v.is_finite()),
                    "block {bi} kept direction {j} has a non-finite entry",
                );
                let norm = col.dot(&col).sqrt();
                assert!(
                    norm > 1e-10,
                    "block {bi} kept direction {j} is degenerate (norm {norm:.3e})",
                );
            }
        }
    }

    /// `T = I` case: per-block V = identity, R = None. The triangular
    /// lift must be the identity on each block.
    #[test]
    fn smgs_lift_via_t_identity_passes_through() {
        let v0 = Array2::<f64>::eye(3);
        let v1 = Array2::<f64>::eye(2);
        let v_per_term = vec![v0, v1];
        let r_per_term: Vec<Option<Array2<f64>>> = vec![None, None];
        let lift = Gauge::from_v_and_r(&v_per_term, &r_per_term);
        assert_eq!(lift.t_full.dim(), (5, 5));
        assert_eq!(lift.block_starts_reduced, vec![0, 3, 5]);
        assert_eq!(lift.block_starts_raw, vec![0, 3, 5]);
        for i in 0..5 {
            for j in 0..5 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((lift.t_full[[i, j]] - want).abs() < 1e-14);
            }
        }
        let theta_0 = Array1::from(vec![1.0_f64, -2.0, 3.5]);
        let theta_1 = Array1::from(vec![-0.5_f64, 7.0]);
        let lifted = lift.lift_block_betas(&[theta_0.clone(), theta_1.clone()]);
        assert_eq!(lifted.len(), 2);
        for (a, b) in theta_0.iter().zip(lifted[0].iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        for (a, b) in theta_1.iter().zip(lifted[1].iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    /// Two-block toy: V_a = I_3, V_b drops the middle column, R is a
    /// non-trivial residualised reparam. Verify β_a_raw = θ_a − R · θ_b
    /// and β_b_raw = V_b · θ_b.
    #[test]
    fn smgs_lift_via_t_two_block_with_residualisation() {
        let v_a = Array2::<f64>::eye(3);
        let mut v_b = Array2::<f64>::zeros((3, 2));
        v_b[[0, 0]] = 1.0;
        v_b[[2, 1]] = 1.0;
        let mut r_b = Array2::<f64>::zeros((3, 2));
        r_b[[0, 0]] = 0.4;
        r_b[[0, 1]] = -0.1;
        r_b[[1, 0]] = 0.7;
        r_b[[1, 1]] = 1.3;
        r_b[[2, 0]] = -0.2;
        r_b[[2, 1]] = 0.5;
        let lift = Gauge::from_v_and_r(&[v_a.clone(), v_b.clone()], &[None, Some(r_b.clone())]);
        assert_eq!(lift.t_full.dim(), (6, 5));
        assert_eq!(lift.block_starts_reduced, vec![0, 3, 5]);
        assert_eq!(lift.block_starts_raw, vec![0, 3, 6]);

        let theta_a = Array1::from(vec![1.0_f64, 2.0, -1.5]);
        let theta_b = Array1::from(vec![0.5_f64, -0.25]);
        let lifted = lift.lift_block_betas(&[theta_a.clone(), theta_b.clone()]);
        let r_theta_b = r_b.dot(&theta_b);
        let expected_a = &theta_a - &r_theta_b;
        assert_eq!(lifted[0].len(), 3);
        for (got, want) in lifted[0].iter().zip(expected_a.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
        assert_eq!(lifted[1].len(), 3);
        assert!((lifted[1][0] - theta_b[0]).abs() < 1e-12);
        assert!(lifted[1][1].abs() < 1e-12);
        assert!((lifted[1][2] - theta_b[1]).abs() < 1e-12);
    }

    /// Covariance pushforward `Σ_raw = T · Σ_θ · Tᵀ` must be the exact
    /// inference companion of the point-estimate lift. Two invariants:
    ///
    /// 1. Identity T (V = I, R = None): the lifted covariance equals the
    ///    input covariance — a true no-op for a rank-clean fit.
    /// 2. Rank-1 consistency with the β lift: for a degenerate posterior
    ///    `Σ_θ = θ θᵀ`, the pushforward must equal `(T θ)(T θ)ᵀ`, i.e.
    ///    lifting the covariance of a point mass agrees with lifting the
    ///    point itself. This couples `lift_covariance` to
    ///    `lift_block_betas` exactly, so the mean and its
    ///    uncertainty can never drift into inconsistent coordinates.
    #[test]
    fn smgs_lift_covariance_identity_and_rank1_consistency() {
        // ── Invariant 1: identity T leaves the covariance unchanged. ──
        let lift_id = Gauge::from_v_and_r(
            &[Array2::<f64>::eye(2), Array2::<f64>::eye(2)],
            &[None, None],
        );
        let mut cov = Array2::<f64>::zeros((4, 4));
        // An arbitrary symmetric PSD-ish covariance.
        for i in 0..4 {
            for j in 0..4 {
                cov[[i, j]] = 1.0 / (1.0 + (i as f64 - j as f64).abs());
            }
        }
        let lifted_id = lift_id.lift_covariance(&cov);
        assert_eq!(lifted_id.dim(), (4, 4));
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (lifted_id[[i, j]] - cov[[i, j]]).abs() < 1e-12,
                    "identity-T covariance lift must be a no-op at [{i},{j}]",
                );
            }
        }

        // ── Invariant 2: rank-1 Σ_θ = θθᵀ pushes to (Tθ)(Tθ)ᵀ. ──
        // Reuse the two-block-with-residualisation geometry: V_a = I_3,
        // V_b drops the middle raw column, R_b non-trivial → raw width 6,
        // compiled width 5.
        let v_a = Array2::<f64>::eye(3);
        let mut v_b = Array2::<f64>::zeros((3, 2));
        v_b[[0, 0]] = 1.0;
        v_b[[2, 1]] = 1.0;
        let mut r_b = Array2::<f64>::zeros((3, 2));
        r_b[[0, 0]] = 0.4;
        r_b[[0, 1]] = -0.1;
        r_b[[1, 0]] = 0.7;
        r_b[[1, 1]] = 1.3;
        r_b[[2, 0]] = -0.2;
        r_b[[2, 1]] = 0.5;
        let lift = Gauge::from_v_and_r(&[v_a, v_b], &[None, Some(r_b)]);

        let theta_a = Array1::from(vec![1.0_f64, 2.0, -1.5]);
        let theta_b = Array1::from(vec![0.5_f64, -0.25]);
        // Concatenated compiled θ (width 5).
        let theta_full = Array1::from(vec![
            theta_a[0], theta_a[1], theta_a[2], theta_b[0], theta_b[1],
        ]);
        // Σ_θ = θ θᵀ (rank-1).
        let mut cov_rank1 = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            for j in 0..5 {
                cov_rank1[[i, j]] = theta_full[i] * theta_full[j];
            }
        }
        let lifted_cov = lift.lift_covariance(&cov_rank1);
        // Reference: (T θ)(T θ)ᵀ via the point-estimate lift.
        let lifted_blocks = lift.lift_block_betas(&[theta_a, theta_b]);
        let beta_raw = Array1::from(
            lifted_blocks
                .iter()
                .flat_map(|b| b.iter().copied())
                .collect::<Vec<f64>>(),
        );
        assert_eq!(lifted_cov.dim(), (6, 6));
        assert_eq!(beta_raw.len(), 6);
        for i in 0..6 {
            for j in 0..6 {
                let want = beta_raw[i] * beta_raw[j];
                assert!(
                    (lifted_cov[[i, j]] - want).abs() < 1e-10,
                    "rank-1 covariance pushforward must equal (Tθ)(Tθ)ᵀ at [{i},{j}]: got {}, want {want}",
                    lifted_cov[[i, j]],
                );
            }
        }
        // Symmetry sanity.
        for i in 0..6 {
            for j in 0..6 {
                assert!((lifted_cov[[i, j]] - lifted_cov[[j, i]]).abs() < 1e-14);
            }
        }
    }

    /// When all R's are None, the triangular gauge lift must equal the
    /// strictly per-block `V_b · θ_b` lift.
    #[test]
    fn smgs_lift_via_t_zero_r_matches_per_block_v_lift() {
        let mut v_a = Array2::<f64>::zeros((3, 2));
        v_a[[0, 0]] = 0.6;
        v_a[[1, 0]] = -0.8;
        v_a[[1, 1]] = 0.3;
        v_a[[2, 1]] = 0.9;
        let mut v_b = Array2::<f64>::zeros((4, 3));
        v_b[[0, 0]] = 1.0;
        v_b[[1, 1]] = -0.4;
        v_b[[2, 0]] = 0.2;
        v_b[[2, 2]] = 0.7;
        v_b[[3, 2]] = -1.1;
        let v_per_term = vec![v_a.clone(), v_b.clone()];
        let lift = Gauge::from_v_and_r(&v_per_term, &[None, None]);
        let theta_a = Array1::from(vec![0.3_f64, -1.4]);
        let theta_b = Array1::from(vec![2.1_f64, 0.0, -0.7]);
        let via_t = lift.lift_block_betas(&[theta_a.clone(), theta_b.clone()]);
        let ref_a = v_a.dot(&theta_a);
        let ref_b = v_b.dot(&theta_b);
        assert_eq!(via_t[0].len(), ref_a.len());
        for (g, w) in via_t[0].iter().zip(ref_a.iter()) {
            assert!((g - w).abs() < 1e-12);
        }
        assert_eq!(via_t[1].len(), ref_b.len());
        for (g, w) in via_t[1].iter().zip(ref_b.iter()) {
            assert!((g - w).abs() < 1e-12);
        }
    }

    /// Recompile-after-first-PIRLS-accept refinement: under a structural
    /// (identity) row Hessian, a direction that is *only* identifiable
    /// through the q1 channel survives the per-term compile; under a
    /// data-adaptive row Hessian that happens to zero out the q1/qd1/g
    /// metric weight (everything except q0), the same direction collapses.
    /// This pins the diagnostic the production hook in
    /// `fit_survival_marginal_slope_terms` watches for: the two row
    /// Hessians produce different `drops_by_block` on identical raw
    /// designs.
    #[test]
    fn recompile_after_accept_diff_detection_pilot_curvature_trap() {
        let n = 6usize;
        // Time block: a single column that only contributes through q0
        // (entry-time channel). Both row Hessians see it identically on
        // the q0 axis.
        let time_dq0 = Array2::<f64>::from_elem((n, 1), 1.0);
        let time_dq1 = Array2::<f64>::zeros((n, 1));
        let time_dqd1 = Array2::<f64>::zeros((n, 1));
        // Marginal block: a single column whose q0 part is colinear with
        // the time block's q0 (both are ones-vectors). Its q-channel maps
        // into BOTH q0 and q1 under QChannelBlockOperator, so under a
        // metric that weighs q1 it carries a non-colinear component.
        let marg_dq = Array2::<f64>::from_elem((n, 1), 1.0);
        let marg_dqd1 = Array2::<f64>::zeros((n, 1));
        // A real logslope channel remains in both compiles. Its distinct
        // primary channel is weighted in both metrics so this fixture isolates
        // the intended q0-vs-q1 marginal rank change.
        let log_dg = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 + 1.0);
        let mut time_partition: Vec<std::ops::Range<usize>> = Vec::with_capacity(1);
        time_partition.push(0..1);
        let mut marg_partition: Vec<std::ops::Range<usize>> = Vec::with_capacity(1);
        marg_partition.push(0..1);
        let log_partition = vec![0..1];

        // Pass 1: structural identity row Hessian. q0/q1/qd1/g all weighted
        // equally → marg's q1 component is visible, so marg is identifiable
        // after residualising against the time block (drops_marg = 0).
        let mut h_ident =
            Array3::<f64>::zeros((n, SINGLE_SCORE_PRIMARY_WIDTH, SINGLE_SCORE_PRIMARY_WIDTH));
        for i in 0..n {
            for k in 0..SINGLE_SCORE_PRIMARY_WIDTH {
                h_ident[[i, k, k]] = 1.0;
            }
        }
        let row_hess_ident = survival_row_hessian_from_full(h_ident);
        let compiled_ident = compile_survival_parametric_designs_per_term(
            time_dq0.clone(),
            time_dq1.clone(),
            time_dqd1.clone(),
            &time_partition,
            marg_dq.clone(),
            marg_dqd1.clone(),
            &marg_partition,
            shared_logslope_layout(log_dg.clone()),
            &log_partition,
            &row_hess_ident,
            false,
        )
        .expect("identity-H compile must succeed");

        // Pass 2: data-adaptive row Hessian that only weighs q0 (all
        // other channel diagonals zero). Marg's q1 contribution is now
        // invisible → marg fully aliases with time on q0 → drops_marg = 1.
        let mut h_q0_only =
            Array3::<f64>::zeros((n, SINGLE_SCORE_PRIMARY_WIDTH, SINGLE_SCORE_PRIMARY_WIDTH));
        for i in 0..n {
            h_q0_only[[i, 0, 0]] = 1.0;
            h_q0_only[[i, 3, 3]] = 1.0;
        }
        let row_hess_q0 = survival_row_hessian_from_full(h_q0_only);
        let compiled_q0 = compile_survival_parametric_designs_per_term(
            time_dq0,
            time_dq1,
            time_dqd1,
            &time_partition,
            marg_dq,
            marg_dqd1,
            &marg_partition,
            shared_logslope_layout(log_dg),
            &log_partition,
            &row_hess_q0,
            false,
        )
        .expect("q0-only-H compile must succeed");

        // The two drops_by_block tuples disagree on the marginal block —
        // this is exactly the "pilot-curvature trap" the recompile-after-
        // accept hook is designed to surface.
        assert_ne!(
            compiled_ident.drops_by_block, compiled_q0.drops_by_block,
            "structural-H and data-adaptive-H compiles must produce different \
             drops_by_block on the constructed pilot-curvature-trap design; \
             identity={:?} q0-only={:?}",
            compiled_ident.drops_by_block, compiled_q0.drops_by_block,
        );
        // Under identity H, marg survives (no drop).
        assert_eq!(
            compiled_ident.drops_by_block.1, 0,
            "identity-H marg drops expected 0, got {:?}",
            compiled_ident.drops_by_block,
        );
        // Under q0-only H, marg fully aliases with time on q0.
        assert_eq!(
            compiled_q0.drops_by_block.1, 1,
            "q0-only-H marg drops expected 1, got {:?}",
            compiled_q0.drops_by_block,
        );
    }

    #[test]
    fn compiled_map_from_per_term_partitions_and_lift_round_trip() {
        // Build a per-term compile by hand: time has one term (raw 2, kept 2),
        // marginal one term (raw 2, kept 1 — a drop), logslope one term
        // (raw 1, kept 1). No required channel is fully collapsed.
        let v_time = Array2::<f64>::eye(2);
        let mut v_marg = Array2::<f64>::zeros((2, 1));
        v_marg[[0, 0]] = 1.0;
        v_marg[[1, 0]] = 0.5;
        let v_log = Array2::<f64>::eye(1);
        // R for the marginal block (anchor = time, raw width 2) and logslope
        // block (anchors = time + marginal, raw width 2 + 2 = 4).
        let r_marg = Array2::<f64>::from_shape_fn((2, 1), |(i, _)| 0.25 + i as f64);
        let r_log = Array2::<f64>::from_shape_fn((4, 1), |(i, _)| 0.1 * (i as f64 + 1.0));
        let per_term = SurvivalParametricCompiledPerTerm {
            v_time_per_term: vec![v_time.clone()],
            v_marginal_per_term: vec![v_marg.clone()],
            v_logslope_per_term: vec![v_log.clone()],
            r_lw_per_term: vec![None, Some(r_marg.clone()), Some(r_log.clone())],
            drops_by_block: (0, 1, 0),
        };

        let map = compiled_map_from_per_term(&per_term);

        // Raw block ranges: time 0..2, marginal 2..4, logslope 4..5.
        assert_eq!(map.raw_block_ranges, vec![0..2, 2..4, 4..5]);
        // Compiled block ranges: time 0..2, marginal 2..3, logslope 3..4.
        assert_eq!(map.compiled_block_ranges, vec![0..2, 2..3, 3..4]);
        assert_eq!(map.raw_from_compiled.dim(), (5, 4));

        // The block-diagonal slices recovered by apply_compiled_map_to_designs
        // must equal the per-term V's exactly.
        let v_time_slice = map
            .raw_from_compiled
            .slice(ndarray::s![0..2, 0..2])
            .to_owned();
        let v_marg_slice = map
            .raw_from_compiled
            .slice(ndarray::s![2..4, 2..3])
            .to_owned();
        let v_log_slice = map
            .raw_from_compiled
            .slice(ndarray::s![4..5, 3..4])
            .to_owned();
        for i in 0..2 {
            for j in 0..2 {
                assert!((v_time_slice[[i, j]] - v_time[[i, j]]).abs() < 1e-14);
            }
            assert!((v_marg_slice[[i, 0]] - v_marg[[i, 0]]).abs() < 1e-14);
        }
        assert!((v_log_slice[[0, 0]] - v_log[[0, 0]]).abs() < 1e-14);

        // The cross-block carry (-R) must sit in the strict upper triangle, so
        // the map agrees with the lift assembled directly from V and R.
        let ordering = [
            gam_identifiability::families::compiler::BlockOrder::Time,
            gam_identifiability::families::compiler::BlockOrder::Marginal,
            gam_identifiability::families::compiler::BlockOrder::Logslope,
        ];
        let lift_from_map = Gauge::from_compiled_map(&map, &ordering);
        let v_all = vec![v_time, v_marg, v_log];
        let lift_direct = Gauge::from_v_and_r(&v_all, &[None, Some(r_marg), Some(r_log)]);
        assert_eq!(lift_from_map.t_full.dim(), lift_direct.t_full.dim());
        for i in 0..lift_from_map.t_full.nrows() {
            for j in 0..lift_from_map.t_full.ncols() {
                assert!(
                    (lift_from_map.t_full[[i, j]] - lift_direct.t_full[[i, j]]).abs() < 1e-14,
                    "T mismatch at ({i},{j}): map={} direct={}",
                    lift_from_map.t_full[[i, j]],
                    lift_direct.t_full[[i, j]],
                );
            }
        }
    }

    // ----- #979 effective reduced-logslope confound removal -----------------
    //
    // Direct unit coverage of the two numerical routines added for #979,
    // mirroring the BMS reference cuts
    // (`bms::block_specs` `effective_reduction_*`): the scalar weight
    // contraction off the per-row 4×4 Hessian, and the block-diagonal map
    // assembly. The 900s end-to-end `survival_marginal_slope_converges_*`
    // guard exercises the same path but is slow and data-dependent; these pin
    // the distinguishing logic deterministically.

    /// Constant per-row 4×4 PSD Hessian carrying ONLY the (q0, g) coupling,
    /// channel order (q0, q1, qd1, g): `H[0,0]=h00`, `H[0,3]=H[3,0]=h03`,
    /// `H[3,3]=h33`, all else zero. The effective scalar weights the
    /// contraction reads are then `w_mm=h00`, `w_mg=h03`, `w_gg=h33`. The
    /// 2×2 (q0,g) block `[[h00,h03],[h03,h33]]` is PSD when `h00·h33 ≥ h03²`.
    fn const_row_hess_q0g(n: usize, h00: f64, h03: f64, h33: f64) -> SurvivalRowHessian {
        let mut h =
            Array3::<f64>::zeros((n, SINGLE_SCORE_PRIMARY_WIDTH, SINGLE_SCORE_PRIMARY_WIDTH));
        for i in 0..n {
            h[[i, 0, 0]] = h00;
            h[[i, 0, 3]] = h03;
            h[[i, 3, 0]] = h03;
            h[[i, 3, 3]] = h33;
        }
        survival_row_hessian_from_full(h)
    }

    #[test]
    fn survival_reduced_logslope_drops_confounded_keeps_free_979() {
        // p_m=1 marginal column m; p_log=2 logslope columns [l1, l2] with
        // l1 == m (an exact rank-1 (q0,g) confound: h00·h33 = h03²) so l1 is
        // fully marginal-explained, and l2 ⊥ m with 100× the energy so it is
        // unambiguously free. The effective Schur Gram must drop ONLY the
        // confounded direction: 0 < r == 1 < p_log == 2.
        let n = 4;
        let row_hess = const_row_hess_q0g(n, 2.0, 2.0, 2.0); // (q0,g) = [[2,2],[2,2]], rank-1
        let marg = Array2::from_shape_vec((n, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        // l1 = m (confounded); l2 = [10,-10,10,-10] (Euclidean-orthogonal to m,
        // ‖l2‖² = 400 ≫ ‖m‖² = 4, so Gtt's free eigenvalue ≫ tol).
        let log =
            Array2::from_shape_vec((n, 2), vec![1.0, 10.0, 1.0, -10.0, 1.0, 10.0, 1.0, -10.0])
                .unwrap();
        let layout = shared_logslope_layout(log);
        let t = match survival_reduced_logslope_transform_effective(marg.view(), &layout, &row_hess)
            .expect("contraction must succeed")
        {
            crate::bms::block_specs::ReducedLogslopeOutcome::Reduced(t) => t,
            other => panic!("a partial confound must yield a reduced transform, got {other:?}"),
        };
        assert_eq!(t.dim(), (2, 1), "exactly one logslope direction survives");
        // The kept eigenvector is the free column ≈ e2 (up to sign); the
        // confounded e1 component is dropped.
        assert!(
            t[[0, 0]].abs() < 1e-6,
            "confounded (e1) direction must be dropped, got {}",
            t[[0, 0]]
        );
        assert!(
            (t[[1, 0]].abs() - 1.0).abs() < 1e-6,
            "free (e2) direction must be kept as a unit vector, got {}",
            t[[1, 0]]
        );
    }

    #[test]
    fn survival_reduced_logslope_fully_confounded_is_distinct_signal_979() {
        // A single logslope column equal to the marginal column under the exact
        // rank-1 (q0,g) confound: the whole effective logslope image lies in the
        // marginal span. The conservative ridge floors the residual eigenvalue at
        // energy_scale·TOL/(1+TOL) < tol, so r == 0 → FullyConfounded — a
        // DISTINCT signal from the full-rank case, so the caller can delete the
        // unidentified channel deliberately (#2245 finding 45 sibling).
        let n = 4;
        let row_hess = const_row_hess_q0g(n, 2.0, 2.0, 2.0);
        let marg = Array2::from_shape_vec((n, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let log = marg.clone();
        let layout = shared_logslope_layout(log);
        let out = survival_reduced_logslope_transform_effective(marg.view(), &layout, &row_hess)
            .expect("contraction must succeed");
        assert!(
            matches!(
                out,
                crate::bms::block_specs::ReducedLogslopeOutcome::FullyConfounded
            ),
            "a fully marginal-explained logslope block must report FullyConfounded"
        );
    }

    #[test]
    fn survival_reduced_logslope_no_confound_is_full_rank_979() {
        // No marginal↔logslope cross weight (h03 = 0): the channels are
        // W-orthogonal, so every logslope direction is free (r == p_log) and
        // there is nothing to remove → FullRank (keep the raw design).
        let n = 4;
        let row_hess = const_row_hess_q0g(n, 2.0, 0.0, 2.0);
        let marg = Array2::from_shape_vec((n, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let log =
            Array2::from_shape_vec((n, 2), vec![1.0, 10.0, 1.0, -10.0, 1.0, 10.0, 1.0, -10.0])
                .unwrap();
        let layout = shared_logslope_layout(log);
        let out = survival_reduced_logslope_transform_effective(marg.view(), &layout, &row_hess)
            .expect("contraction must succeed");
        assert!(
            matches!(
                out,
                crate::bms::block_specs::ReducedLogslopeOutcome::FullRank
            ),
            "W-orthogonal channels need no reduction → keep raw"
        );
    }

    #[test]
    fn survival_block_diagonal_logslope_map_is_identity_on_time_and_marginal_979() {
        // Time (p=2) and marginal (p=3) blocks pass through as identities; only
        // the logslope block (raw p_log=4) is reparameterised by t_log (4×2).
        let p_time = 2;
        let p_marg = 3;
        let t_log = Array2::from_shape_fn((4, 2), |(i, j)| 1.0 + (i * 2 + j) as f64);
        let map = survival_block_diagonal_logslope_map(p_time, p_marg, &t_log);

        assert_eq!(map.raw_block_ranges, vec![0..2, 2..5, 5..9]);
        assert_eq!(map.compiled_block_ranges, vec![0..2, 2..5, 5..7]);
        assert_eq!(map.raw_from_compiled.dim(), (9, 7));

        let t = &map.raw_from_compiled;
        // V_time = I2.
        for i in 0..p_time {
            for j in 0..p_time {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((t[[i, j]] - want).abs() < 1e-14, "V_time[{i},{j}]");
            }
        }
        // V_marg = I3.
        for i in 0..p_marg {
            for j in 0..p_marg {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (t[[p_time + i, p_time + j]] - want).abs() < 1e-14,
                    "V_marg[{i},{j}]"
                );
            }
        }
        // V_log = t_log.
        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (t[[p_time + p_marg + i, p_time + p_marg + j]] - t_log[[i, j]]).abs() < 1e-14,
                    "V_log[{i},{j}]"
                );
            }
        }
        // No cross-block bleed: the only nonzeros are the two identities and the
        // t_log block (every t_log entry here is nonzero).
        let nnz = t.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            nnz,
            p_time + p_marg + t_log.iter().filter(|&&v| v != 0.0).count()
        );
    }
}
