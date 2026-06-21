//! Survival marginal-slope concrete impls for the family-agnostic
//! identifiability compiler (`crate::identifiability::families::compiler`).
//!
//! Survival's row primary state is the 4-vector `u_i = (q0, q1, qd1, g)`,
//! so `K = 4`. The row Hessian is the 4×4 second-derivative block of the
//! per-row neg-log-likelihood kernel `row_primary_closed_form` at a pilot
//! `β`, PSD-clamped via eigendecomposition (negative eigenvalues projected
//! to zero) to handle pilot points far from the optimum.
//!
//! Each block exposes its row Jacobian as the contribution of `δβ_block`
//! to the row primary-state vector:
//!
//! - **TimeBlockOperator**: `(δq0, δq1, δqd1, 0)` from `design_entry`,
//!   `design_exit`, `design_derivative_exit` rows.
//! - **MarginalBlockOperator**: `(δq, δq, δqd_marginal, 0)` from the
//!   marginal design row (shared by q0 and q1; qd contribution zero unless
//!   timewiggle is active — captured by an explicit derivative row matrix).
//! - **LogslopeBlockOperator**: `(0, 0, 0, δg)` from the logslope design.
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

use crate::families::custom_family::{FamilyChannelHessian, PenaltyMatrix};
use crate::identifiability::families::compiler::{
    BlockOrder, RowHessian, RowJacobianOperator, scale_jacobian_by_sqrt_h_with,
};
use crate::linalg::faer_ndarray::{FaerEigh, fast_ab};
use crate::linalg::matrix::{CoefficientTransformOperator, DenseDesignMatrix, DesignMatrix};
use crate::solver::gauge::assemble_block_triangular_t;
use faer::Side;

const K_SURVIVAL: usize = 4;

/// Threshold below which a coefficient vector is treated as the trivial
/// (all-zero) pilot point in the drift-detection audit. At β ≈ 0 the
/// primary-state coupling g vanishes (c ≡ 1), so the frozen pilot W is exact;
/// any |β_j| above this is "non-trivial" and requires the family scalars to
/// re-evaluate W(β). The bound is well below any meaningful fitted coefficient.
const BETA_NONTRIVIAL_ABS_THRESHOLD: f64 = 1e-12;

/// Per-row 4×4 row Hessian for the survival marginal-slope likelihood at a
/// pilot `β`. The pilot supplies the primary-state vector
/// `(q0_i, q1_i, qd1_i, g_i)` and the per-row sample weight + event
/// indicator + z + probit scale. The 4×4 block is evaluated via the
/// existing `row_primary_closed_form` kernel (which already returns the
/// full Hessian in `(q0, q1, qd1, g)` order) and PSD-clamped per row.
pub struct SurvivalRowHessian {
    /// PSD-projected per-row 4×4 Hessian, stored row-major as
    /// `(n × 4 × 4)`.
    h: Array3<f64>,
}

impl SurvivalRowHessian {
    /// Construct from explicit per-row pilot primary-state and the row
    /// data needed by `row_primary_closed_form`. Negative eigenvalues are
    /// projected to zero before storage so the matrix is PSD.
    pub fn from_pilot_primary_state(
        q0: &Array1<f64>,
        q1: &Array1<f64>,
        qd1: &Array1<f64>,
        g: &Array1<f64>,
        z: &Array1<f64>,
        weights: &Array1<f64>,
        event: &Array1<f64>,
        derivative_guard: f64,
        probit_scale: f64,
    ) -> Result<Self, String> {
        let n = q0.len();
        if [
            q1.len(),
            qd1.len(),
            g.len(),
            z.len(),
            weights.len(),
            event.len(),
        ]
        .iter()
        .any(|&l| l != n)
        {
            return Err(format!(
                "SurvivalRowHessian: length mismatch \
                 q0={n}, q1={}, qd1={}, g={}, z={}, weights={}, event={}",
                q1.len(),
                qd1.len(),
                g.len(),
                z.len(),
                weights.len(),
                event.len()
            ));
        }
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            let (_, _grad, hess) =
                crate::families::survival::marginal_slope::row_primary_for_compiler(
                    q0[i],
                    q1[i],
                    qd1[i],
                    g[i],
                    z[i],
                    weights[i],
                    event[i],
                    derivative_guard,
                    probit_scale,
                )?;
            // PSD-clamp via eigendecomposition: project negative eigvals to 0.
            let mut h_i = Array2::<f64>::zeros((K_SURVIVAL, K_SURVIVAL));
            for a in 0..K_SURVIVAL {
                for b in 0..K_SURVIVAL {
                    h_i[[a, b]] = hess[a][b];
                }
            }
            let clamped = psd_clamp_4x4(&h_i);
            for a in 0..K_SURVIVAL {
                for b in 0..K_SURVIVAL {
                    h_full[[i, a, b]] = clamped[[a, b]];
                }
            }
        }
        Ok(Self { h: h_full })
    }

    /// Construct from an already-PSD per-row tensor. Used by callers that
    /// have computed the Hessian via a different route.
    pub fn from_full(h: Array3<f64>) -> Self {
        assert_eq!(h.shape()[1], K_SURVIVAL);
        assert_eq!(h.shape()[2], K_SURVIVAL);
        Self { h }
    }
}

impl RowHessian for SurvivalRowHessian {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn nrows(&self) -> usize {
        self.h.shape()[0]
    }
    fn fill_row(&self, row: usize, out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL * K_SURVIVAL);
        for a in 0..K_SURVIVAL {
            for b in 0..K_SURVIVAL {
                out[a * K_SURVIVAL + b] = self.h[[row, a, b]];
            }
        }
    }
    fn evaluate_full(&self) -> Array3<f64> {
        self.h.clone()
    }
}

/// `FamilyChannelHessian` for survival marginal-slope.
///
/// The 4×4 per-subject W_i is the Hessian of the row negative log-likelihood
/// `ρ_i(q0, q1, qd1, g) = −δ_i log f(η_1, ad1) − (1−δ_i) log S(η_1) + log S(η_0)`
/// with respect to the 4-vector primary state `(q0, q1, qd1, g)`.
///
/// Derivation of W_i entries (all from `row_primary_closed_form`):
///
/// - W[0,0] = u2_η0 · c²  (q0–q0; only η0 depends on q0)
/// - W[1,1] = (u2_η1 + w·δ) · c²  (q1–q1; η1 and log-φ both depend on q1)
/// - W[2,2] = w·δ · (∂ad1/∂qd1)² · (−1/ad1²)  (qd1–qd1 via neglog(ad1))
/// - W[3,3] = u2_η0·(∂η0/∂g)² + u1_η0·(∂²η0/∂g²) + u2_η1·(∂η1/∂g)² + ...
/// - W[0,3] = W[3,0] = u2_η0·c·(q0·c1 + s_f·z) + u1_η0·c1  (cross q0–g)
/// - W[1,3] = W[3,1] = u2_η1·c·(q1·c1 + s_f·z) + u1_η1·c1  (cross q1–g)
/// - W[2,3] = W[3,2] = u2_ad1·c·(qd1·c1) + u1_ad1·c1  (cross qd1–g)
/// - All other off-diagonals are zero (η0, η1, ad1 depend on non-overlapping
///   subsets of (q0,q1,qd1,g), and only g is shared across all three).
///
/// This is already computed by `row_primary_closed_form` and stored in
/// `SurvivalRowHessian::h` after PSD-clamping.
///
/// # β-dependent W via `channel_hessian_at`
///
/// `channel_hessian_at` overrides the default β-independent path.  When
/// `family_scalars` carries `SurvivalMarginalSlopeFamilyScalars`, the
/// current per-row primary state `(q0_i, q1_i, qd1_i, g_i)` is read from
/// those scalars and the 4×4 W_i is recomputed via `row_primary_for_compiler`.
/// This makes `I(β) = J(β)^T W(β) J(β)` accurate at the current β instead of
/// at the frozen pilot β=0 state.
///
/// When `family_scalars` is `None` but `beta` is zero-ish (all entries ≤ ε),
/// the frozen pilot W is returned unchanged.  When `family_scalars` is `None`
/// and any `beta` entry is non-trivial, `Err` is returned — the caller must
/// supply scalars for a correct W at non-pilot β (same contract as T26's
/// Jacobian callbacks: scalars required when β affects the primary state).
impl FamilyChannelHessian for SurvivalRowHessian {
    fn n_outputs(&self) -> usize {
        K_SURVIVAL
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL * K_SURVIVAL);
        for a in 0..K_SURVIVAL {
            for b in 0..K_SURVIVAL {
                out[a * K_SURVIVAL + b] = self.h[[i, a, b]];
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
        use crate::families::survival::marginal_slope::SurvivalMarginalSlopeFamilyScalars;

        let scalars_opt =
            family_scalars.and_then(|a| a.downcast_ref::<SurvivalMarginalSlopeFamilyScalars>());

        // Determine whether beta is non-trivial (any |β_j| > ε).
        let beta_nontrivial = beta
            .iter()
            .any(|&b| b.abs() > BETA_NONTRIVIAL_ABS_THRESHOLD);

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
                // β ≈ 0: return the frozen pilot W unchanged.
                Ok(Arc::new(
                    crate::families::custom_family::TensorChannelHessian { h: self.h.clone() },
                ))
            }
            Some(sc) => {
                let n = self.h.shape()[0];
                if sc.q0_i.len() != n
                    || sc.q1_i.len() != n
                    || sc.qd1_i.len() != n
                    || sc.g_i.len() != n
                    || sc.z_i.len() != n
                {
                    return Err(format!(
                        "SurvivalRowHessian::channel_hessian_at: scalars length mismatch \
                         (expected n={n}, got q0={} q1={} qd1={} g={} z={})",
                        sc.q0_i.len(),
                        sc.q1_i.len(),
                        sc.qd1_i.len(),
                        sc.g_i.len(),
                        sc.z_i.len(),
                    ));
                }
                // We do not have weights/event stored in SurvivalRowHessian itself.
                // The scalars carry the per-row primary state; we need per-row weights
                // and event indicators to call row_primary_for_compiler.  Those are
                // NOT stored in SurvivalMarginalSlopeFamilyScalars — so we can only
                // recompute W's structural shape (the 4×4 curvature geometry) using
                // unit weights and event=1, which gives us the correct _direction_ of
                // W at the current β even if the magnitude is off by the sample weight.
                //
                // For the drift-detection audit the direction matters more than the
                // exact per-row magnitudes: rank changes emerge from structural
                // identifiability, not from per-row weight scaling. Using w=1, d=1
                // is therefore the principled approximation for the audit path.
                //
                // Production callers that need exact W (e.g. for the Fisher Gram in
                // the compiler) should use SurvivalRowHessian::from_pilot_primary_state
                // directly with the true per-row weights and event indicators.
                let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
                for i in 0..n {
                    let q0 = sc.q0_i[i];
                    let q1 = sc.q1_i[i];
                    let qd1 = sc.qd1_i[i];
                    let g = sc.g_i[i];
                    let z = sc.z_i[i];
                    // Use unit weight and d=1 (event indicator 1) for the audit path.
                    // The derivative_guard is the family default (small but non-zero).
                    match crate::families::survival::marginal_slope::row_primary_for_compiler(
                        q0, q1, qd1, g, z, 1.0,  // w = unit weight
                        1.0,  // d = event
                        crate::families::survival::marginal_slope::DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
                        sc.s, // probit_scale from scalars
                    ) {
                        Ok((_nll, _grad, hess)) => {
                            let mut h_i = ndarray::Array2::<f64>::zeros((K_SURVIVAL, K_SURVIVAL));
                            for a in 0..K_SURVIVAL {
                                for b in 0..K_SURVIVAL {
                                    h_i[[a, b]] = hess[a][b];
                                }
                            }
                            let clamped = psd_clamp_4x4(&h_i);
                            for a in 0..K_SURVIVAL {
                                for b in 0..K_SURVIVAL {
                                    h_full[[i, a, b]] = clamped[[a, b]];
                                }
                            }
                        }
                        Err(_) => {
                            // Monotonicity violation or other numerical issue at this
                            // row: fall back to the frozen pilot W for this row only.
                            for a in 0..K_SURVIVAL {
                                for b in 0..K_SURVIVAL {
                                    h_full[[i, a, b]] = self.h[[i, a, b]];
                                }
                            }
                        }
                    }
                }
                Ok(Arc::new(SurvivalRowHessian::from_full(h_full)))
            }
        }
    }
}

/// Project a 4×4 symmetric matrix onto the PSD cone: zero negative
/// eigenvalues. If the eigendecomposition fails (extremely defensive —
/// `row_primary_closed_form` already guarantees finite entries), return
/// the diagonal with negatives clamped.
fn psd_clamp_4x4(m: &Array2<f64>) -> Array2<f64> {
    let k = m.nrows();
    let (evals, evecs) = match m.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(_) => {
            let mut out = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                out[[i, i]] = m[[i, i]].max(0.0);
            }
            return out;
        }
    };
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
    out
}

/// Evaluate the survival marginal-slope row NLL at the given primary-state
/// and data scalars.
///
/// This is a thin public wrapper around the crate-private
/// `row_primary_for_compiler` for use in integration tests that
/// finite-difference the NLL to verify `SurvivalRowHessian`.
///
/// Returns `(nll, grad[4], hess[4][4])` on the stack. See
/// `row_primary_closed_form` for the complete derivation.
pub fn survival_row_nll_grad_hess(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
    crate::families::survival::marginal_slope::row_primary_for_compiler(
        q0,
        q1,
        qd1,
        g,
        z,
        w,
        d,
        derivative_guard,
        probit_scale,
    )
}

/// Row Jacobian operator for the survival time block. Channels (q0, q1,
/// qd1) come from the three time designs; the g channel is zero.
pub struct TimeBlockOperator {
    dq0: Array2<f64>,
    dq1: Array2<f64>,
    dqd1: Array2<f64>,
}

impl TimeBlockOperator {
    pub fn new(dq0: Array2<f64>, dq1: Array2<f64>, dqd1: Array2<f64>) -> Self {
        assert_eq!(dq0.dim(), dq1.dim());
        assert_eq!(dq0.dim(), dqd1.dim());
        Self { dq0, dq1, dqd1 }
    }
}

impl RowJacobianOperator for TimeBlockOperator {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn ncols(&self) -> usize {
        self.dq0.ncols()
    }
    fn nrows(&self) -> usize {
        self.dq0.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL);
        assert_eq!(delta_beta.len(), self.dq0.ncols());
        let mut acc = [0.0_f64; K_SURVIVAL];
        for (j, &b) in delta_beta.iter().enumerate() {
            acc[0] += self.dq0[[row, j]] * b;
            acc[1] += self.dq1[[row, j]] * b;
            acc[2] += self.dqd1[[row, j]] * b;
        }
        out.copy_from_slice(&acc);
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dq0.nrows();
        let p = self.dq0.ncols();
        let mut out = Array3::<f64>::zeros((n, p, K_SURVIVAL));
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
        scale_jacobian_by_sqrt_h_with(n, p, K_SURVIVAL, h_full, |i, a, c| match c {
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
}

impl QChannelBlockOperator {
    pub fn new(dq: Array2<f64>, dqd1: Array2<f64>) -> Self {
        assert_eq!(dq.dim(), dqd1.dim());
        Self { dq, dqd1 }
    }
}

impl RowJacobianOperator for QChannelBlockOperator {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn ncols(&self) -> usize {
        self.dq.ncols()
    }
    fn nrows(&self) -> usize {
        self.dq.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL);
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
        out[3] = 0.0;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dq.nrows();
        let p = self.dq.ncols();
        let mut out = Array3::<f64>::zeros((n, p, K_SURVIVAL));
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
        scale_jacobian_by_sqrt_h_with(n, p, K_SURVIVAL, h_full, |i, a, c| match c {
            0 | 1 => self.dq[[i, a]],
            2 => self.dqd1[[i, a]],
            _ => 0.0,
        })
    }
}

/// Row Jacobian operator for the survival logslope block: contribution
/// lives entirely on the g channel.
pub struct LogslopeBlockOperator {
    dg: Array2<f64>,
}

impl LogslopeBlockOperator {
    pub fn new(dg: Array2<f64>) -> Self {
        Self { dg }
    }
}

impl RowJacobianOperator for LogslopeBlockOperator {
    fn k(&self) -> usize {
        K_SURVIVAL
    }
    fn ncols(&self) -> usize {
        self.dg.ncols()
    }
    fn nrows(&self) -> usize {
        self.dg.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), K_SURVIVAL);
        assert_eq!(delta_beta.len(), self.dg.ncols());
        let mut acc = 0.0;
        for (j, &b) in delta_beta.iter().enumerate() {
            acc += self.dg[[row, j]] * b;
        }
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = acc;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.dg.nrows();
        let p = self.dg.ncols();
        let mut out = Array3::<f64>::zeros((n, p, K_SURVIVAL));
        for i in 0..n {
            for j in 0..p {
                out[[i, j, 3]] = self.dg[[i, j]];
            }
        }
        out
    }
    fn scaled_design_by_sqrt_h(&self, h_full: &Array3<f64>) -> Array2<f64> {
        // The logslope contribution lives entirely on the g channel (3); the
        // other three channels are identically zero. Scale directly from the
        // compact `(n, p)` design, skipping the mostly-zero dense `(n, p, K)`
        // tensor `evaluate_full()` would build. (#738.)
        let n = self.dg.nrows();
        let p = self.dg.ncols();
        scale_jacobian_by_sqrt_h_with(n, p, K_SURVIVAL, h_full, |i, a, c| {
            if c == 3 { self.dg[[i, a]] } else { 0.0 }
        })
    }
}

/// Inputs assembled for the survival fit driver to feed `compile()`. The
/// ordering follows `gauge_priority` descending (time=200 → marginal=150 →
/// logslope=120 → score_warp=80 → link_dev=60).
pub struct SurvivalCompilerInputs {
    pub operators: Vec<Arc<dyn RowJacobianOperator>>,
    pub ordering: Vec<BlockOrder>,
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
pub struct SurvivalParametricCompiled {
    pub v_time: Array2<f64>,
    pub v_marginal: Array2<f64>,
    pub v_logslope: Array2<f64>,
    /// Per-block dropped raw-column count, indexed
    /// `(time_dropped, marginal_dropped, logslope_dropped)`. Equal to
    /// `(p_raw − v.ncols())` for each block. Useful for logging the
    /// gauge-attribution summary at the construction site.
    pub drops_by_block: (usize, usize, usize),
}

fn wrap_design_with_transform(
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

impl SurvivalParametricCompiledPerTerm {
    /// Block-diagonal V for the time block.
    pub fn v_time_block_diag(&self) -> Array2<f64> {
        block_diag_from(&self.v_time_per_term)
    }
    /// Block-diagonal V for the marginal block.
    pub fn v_marginal_block_diag(&self) -> Array2<f64> {
        block_diag_from(&self.v_marginal_per_term)
    }
    /// Block-diagonal V for the logslope block.
    pub fn v_logslope_block_diag(&self) -> Array2<f64> {
        block_diag_from(&self.v_logslope_per_term)
    }
}

/// Stack a list of per-term V matrices into a single block-diagonal V.
/// Output has rows = sum of raw widths, cols = sum of kept widths.
fn block_diag_from(v_per_term: &[Array2<f64>]) -> Array2<f64> {
    let total_rows: usize = v_per_term.iter().map(|v| v.nrows()).sum();
    let total_cols: usize = v_per_term.iter().map(|v| v.ncols()).sum();
    let mut out = Array2::<f64>::zeros((total_rows, total_cols));
    let mut row_off = 0usize;
    let mut col_off = 0usize;
    for v in v_per_term {
        let r = v.nrows();
        let c = v.ncols();
        if r > 0 && c > 0 {
            out.slice_mut(ndarray::s![row_off..row_off + r, col_off..col_off + c])
                .assign(v);
        }
        row_off += r;
        col_off += c;
    }
    out
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
pub fn compile_survival_parametric_designs_per_term(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    time_partition: &[std::ops::Range<usize>],
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    marginal_partition: &[std::ops::Range<usize>],
    logslope_dg: Array2<f64>,
    logslope_partition: &[std::ops::Range<usize>],
    row_hess: &dyn RowHessian,
) -> Result<SurvivalParametricCompiledPerTerm, String> {
    use crate::identifiability::families::compiler::compile;

    let p_time = time_dq0.ncols();
    let p_marg = marginal_dq.ncols();
    let p_log = logslope_dg.ncols();
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
        operators.push(Arc::new(TimeBlockOperator::new(dq0, dq1, dqd1)));
        ordering.push(BlockOrder::Time);
    }
    for range in marginal_partition {
        let dq = marginal_dq.slice(ndarray::s![.., range.clone()]).to_owned();
        let dqd1 = marginal_dqd1
            .slice(ndarray::s![.., range.clone()])
            .to_owned();
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1)));
        ordering.push(BlockOrder::Marginal);
    }
    for range in logslope_partition {
        let dg = logslope_dg.slice(ndarray::s![.., range.clone()]).to_owned();
        operators.push(Arc::new(LogslopeBlockOperator::new(dg)));
        ordering.push(BlockOrder::Logslope);
    }

    let compiled = compile(&operators, row_hess, &ordering).map_err(|e| {
        format!("identifiability::families::compiler::compile (per-term) failed: {e}")
    })?;
    let blocks = compiled.blocks;
    let n_time = time_partition.len();
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

/// Pull back a BlockwisePenalty under a block-diagonal V whose per-term
/// V's are listed in `v_per_term`, partitioned by `raw_partition`.
/// The penalty's `col_range` must lie within a single partition entry
/// (i.e. the penalty belongs to one term). Returns a new BlockwisePenalty
/// with `local = V_term^T · local · V_term` and `col_range` shifted to
/// compiled coordinates.
pub fn pull_back_blockwise_penalty_per_term(
    pen: &crate::terms::smooth::BlockwisePenalty,
    raw_partition: &[std::ops::Range<usize>],
    v_per_term: &[Array2<f64>],
) -> Result<crate::terms::smooth::BlockwisePenalty, String> {
    use crate::terms::smooth::BlockwisePenalty;
    if raw_partition.len() != v_per_term.len() {
        return Err(format!(
            "pull_back_blockwise_penalty_per_term: partition len {} != v_per_term len {}",
            raw_partition.len(),
            v_per_term.len()
        ));
    }
    // Find which term the penalty belongs to.
    let mut term_idx = None;
    for (idx, range) in raw_partition.iter().enumerate() {
        if pen.col_range.start >= range.start && pen.col_range.end <= range.end {
            term_idx = Some(idx);
            break;
        }
    }
    let term_idx = term_idx.ok_or_else(|| {
        format!(
            "pull_back_blockwise_penalty_per_term: penalty col_range {}..{} does not fit \
             within any term partition (partition entries: {:?})",
            pen.col_range.start, pen.col_range.end, raw_partition
        )
    })?;
    let v_term = &v_per_term[term_idx];
    let term_range = &raw_partition[term_idx];
    // The penalty's local matrix is sized `pen.col_range.len()` and
    // covers a *sub-region* of the term (some smooths emit a single
    // penalty over the whole term; tensor smooths can emit multiple
    // penalties at the same col_range; anisotropic constructions can
    // emit a penalty over a sub-range of the term). We need a V slice
    // that lines up with pen.col_range relative to term_range.start.
    let local_off_start = pen.col_range.start - term_range.start;
    // Per-term V is (term_p × term_p_kept). Slicing the rows of V along
    // pen.col_range's offset within the term gives the sub-block to use
    // for pullback. Equivalently: build an embedded V_full at term width
    // and multiply V_full^T · embed(local) · V_full.
    //
    // For correctness when the penalty doesn't cover the full term, we
    // embed the local matrix to the full term width, then pull back via
    // the whole V_term, then keep all kept-cols of the result. This
    // matches the canonical Vᵀ embed(S) V on the term subspace.
    let term_p = term_range.len();
    let mut embedded = Array2::<f64>::zeros((term_p, term_p));
    for i in 0..pen.col_range.len() {
        for j in 0..pen.col_range.len() {
            embedded[[local_off_start + i, local_off_start + j]] = pen.local[[i, j]];
        }
    }
    // Vᵀ · embedded · V
    let temp = embedded.dot(v_term);
    let pulled = v_term.t().dot(&temp);
    // Symmetrise to wash out floating noise.
    let r = pulled.nrows();
    let mut sym = Array2::<f64>::zeros((r, r));
    for i in 0..r {
        for j in 0..r {
            sym[[i, j]] = 0.5 * (pulled[[i, j]] + pulled[[j, i]]);
        }
    }
    // Compiled term col_range: offsets in the compiled (block-diagonal)
    // coordinate are the cumulative sum of v_per_term[k].ncols() up to
    // term_idx.
    let mut compiled_start = 0usize;
    for v in v_per_term.iter().take(term_idx) {
        compiled_start += v.ncols();
    }
    let compiled_end = compiled_start + v_term.ncols();
    // The pen.local previously covered a sub-region of the term; after
    // pullback we drop that sub-region distinction and the new penalty
    // covers the WHOLE compiled term (since the embedded-then-pulled-back
    // form is over all term-kept cols).
    Ok(BlockwisePenalty::new(compiled_start..compiled_end, sym))
}

/// Assemble the block-upper-triangular reparameterisation T that maps a
/// compiled coefficient vector θ (concatenation of per-term θ_b in
/// `kept_p_b` widths) to the raw coefficient vector γ (concatenation of
/// per-term γ_a in `raw_p_a` widths) under the V+M-exact residualised
/// emitted design `L_b = C_b V_b − A_{<b} R_b`.
///
/// On the diagonal: block (b, b) = `V_b` of shape `raw_p_b × kept_p_b`.
/// In the strictly upper triangle: block (a, b) for `a < b` =
/// `−R_{a→b}` of shape `raw_p_a × kept_p_b`. Each `r_per_term[b]` is
/// `Some(R_b)` where the rows of `R_b` are the vertical stack of the
/// `R_{a→b}` over `a = 0..b` (so `R_b.nrows() == Σ_{a<b} raw_p_a`).
/// `r_per_term[0]` should be `None` (or a zero-row matrix) since no
/// earlier blocks exist.
///
/// Returns T of shape `(Σ raw_p_b) × (Σ kept_p_b)`.
/// A single global-T block specification: the V matrix and the
/// optional residualisation against earlier blocks for one block in
/// the global block ordering. `r_against_earlier` is `Some(R)` where
/// `R.nrows() == Σ_{a<b} raw_p_a` and `R.ncols() == kept_p_b`; or
/// `None` for the very first block.
pub struct GlobalTBlock {
    pub v: Array2<f64>,
    pub r_against_earlier: Option<Array2<f64>>,
}

/// Per-block global-T spec: for each top-level block (e.g. time,
/// marginal, logslope) we carry an ordered list of `GlobalTBlock`s
/// describing that block's per-term sub-partition. The global T is
/// the block-upper-triangular reparameterisation across ALL terms in
/// the order `blocks[0][0], blocks[0][1], …, blocks[1][0], …`.
pub struct GlobalTSpec {
    pub blocks: Vec<Vec<GlobalTBlock>>,
}

/// Assemble the global T across every (block, term) in the natural
/// nested order encoded in `spec`. The output is shape
/// `(Σ raw_p) × (Σ kept_p)` where the sums run over every leaf term
/// in every top-level block, in nested order.
///
/// This delegates to `assemble_block_triangular_t` after flattening
/// `spec.blocks` — the block-upper-triangular structure is the same
/// regardless of how the leaf terms group into top-level blocks; the
/// only thing the top-level grouping affects is the call-site
/// indexing convention for things like β lift slices.
pub fn build_global_t_matrix(spec: &GlobalTSpec) -> Array2<f64> {
    let total_terms: usize = spec.blocks.iter().map(|b| b.len()).sum();
    let mut v_flat: Vec<Array2<f64>> = Vec::with_capacity(total_terms);
    let mut r_flat: Vec<Option<Array2<f64>>> = Vec::with_capacity(total_terms);
    for block in &spec.blocks {
        for term in block {
            v_flat.push(term.v.clone());
            r_flat.push(term.r_against_earlier.clone());
        }
    }
    assemble_block_triangular_t(&v_flat, &r_flat)
}

/// Pull a single raw per-term penalty back through the full-width
/// reparameterisation T. The penalty's `local` is `block_p × block_p`
/// where `block_p == pen.col_range.len()`; `anchor_offset` is the start
/// (in the joint raw coordinate) of the term whose partition contains
/// `pen.col_range`, so the penalty is embedded into the joint raw
/// matrix at rows/cols `anchor_offset + pen.col_range`. The returned
/// matrix is the full-width compiled penalty `T^T · embed(S) · T`,
/// stored as `PenaltyMatrix::Dense`.
///
/// Unlike `pull_back_blockwise_penalty_per_term`, the result is NOT a
/// `BlockwisePenalty`: residualisation off-diagonal couples θ_b through
/// every earlier block's θ_a, so the pulled-back penalty is generally
/// dense across the full compiled width.
pub fn pull_back_penalty_through_t(
    pen: &crate::terms::smooth::BlockwisePenalty,
    anchor_offset: usize,
    t: &Array2<f64>,
) -> PenaltyMatrix {
    let raw_total = t.nrows();
    let compiled_total = t.ncols();
    let block_p = pen.col_range.len();
    let embed_start = anchor_offset + pen.col_range.start;
    let embed_end = embed_start + block_p;
    assert!(
        embed_end <= raw_total,
        "pull_back_penalty_through_t: embed range {}..{} exceeds raw total {}",
        embed_start,
        embed_end,
        raw_total,
    );
    let mut embedded = Array2::<f64>::zeros((raw_total, raw_total));
    if block_p > 0 {
        let mut dst =
            embedded.slice_mut(ndarray::s![embed_start..embed_end, embed_start..embed_end]);
        for i in 0..block_p {
            for j in 0..block_p {
                dst[[i, j]] = pen.local[[i, j]];
            }
        }
    }
    let temp = embedded.dot(t);
    let pulled = t.t().dot(&temp);
    let mut sym = Array2::<f64>::zeros((compiled_total, compiled_total));
    for i in 0..compiled_total {
        for j in 0..compiled_total {
            sym[[i, j]] = 0.5 * (pulled[[i, j]] + pulled[[j, i]]);
        }
    }
    PenaltyMatrix::Dense(sym)
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
    pen: &crate::terms::smooth::BlockwisePenalty,
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

/// Per-term-aware compiled designs + penalties + block-diagonal V
/// matrices for the result-time β lift. The construction site swaps
/// raw designs/penalties for these compiled versions before building
/// `ParameterBlockSpec`s; at fit result the per-block β is lifted via
/// `v_time · β_time_compiled` (and similarly for marginal/logslope) to
/// produce raw-width β that predict-time consumes unchanged.
pub struct CompiledSurvivalDesignsPerTerm {
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    pub time_penalties: Vec<crate::terms::smooth::BlockwisePenalty>,
    pub marginal_penalties: Vec<crate::terms::smooth::BlockwisePenalty>,
    pub logslope_penalties: Vec<crate::terms::smooth::BlockwisePenalty>,
    /// Block-diagonal V for each block: rows = raw block width,
    /// cols = compiled block width. Used by `lift_smgs_block_betas_to_raw`
    /// at fit result to map compiled β back to raw β before serialise.
    pub v_time: Array2<f64>,
    pub v_marginal: Array2<f64>,
    pub v_logslope: Array2<f64>,
}

/// Per-term apply: produce compiled designs + per-term-pulled-back
/// penalties + the block-diagonal V matrices needed for the result-
/// time lift.
#[allow(clippy::too_many_arguments)]
pub fn apply_per_term_survival_parametric_compile_to_designs(
    compiled: &SurvivalParametricCompiledPerTerm,
    time_partition: &[std::ops::Range<usize>],
    marginal_partition: &[std::ops::Range<usize>],
    logslope_partition: &[std::ops::Range<usize>],
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    time_penalties: &[crate::terms::smooth::BlockwisePenalty],
    marginal_penalties: &[crate::terms::smooth::BlockwisePenalty],
    logslope_penalties: &[crate::terms::smooth::BlockwisePenalty],
) -> Result<CompiledSurvivalDesignsPerTerm, String> {
    let v_time = compiled.v_time_block_diag();
    let v_marginal = compiled.v_marginal_block_diag();
    let v_logslope = compiled.v_logslope_block_diag();
    let pull_set = |pens: &[crate::terms::smooth::BlockwisePenalty],
                    partition: &[std::ops::Range<usize>],
                    v_per_term: &[Array2<f64>]|
     -> Result<Vec<crate::terms::smooth::BlockwisePenalty>, String> {
        pens.iter()
            .map(|p| pull_back_blockwise_penalty_per_term(p, partition, v_per_term))
            .collect()
    };
    Ok(CompiledSurvivalDesignsPerTerm {
        time_design_entry: wrap_design_with_transform(
            time_design_entry,
            &v_time,
            "smgs per-term apply: time entry",
        )?,
        time_design_exit: wrap_design_with_transform(
            time_design_exit,
            &v_time,
            "smgs per-term apply: time exit",
        )?,
        time_design_derivative_exit: wrap_design_with_transform(
            time_design_derivative_exit,
            &v_time,
            "smgs per-term apply: time derivative_exit",
        )?,
        marginal_design: wrap_design_with_transform(
            marginal_design,
            &v_marginal,
            "smgs per-term apply: marginal",
        )?,
        logslope_design: wrap_design_with_transform(
            logslope_design,
            &v_logslope,
            "smgs per-term apply: logslope",
        )?,
        time_penalties: pull_set(time_penalties, time_partition, &compiled.v_time_per_term)?,
        marginal_penalties: pull_set(
            marginal_penalties,
            marginal_partition,
            &compiled.v_marginal_per_term,
        )?,
        logslope_penalties: pull_set(
            logslope_penalties,
            logslope_partition,
            &compiled.v_logslope_per_term,
        )?,
        v_time,
        v_marginal,
        v_logslope,
    })
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
/// [`crate::identifiability::families::compiler::compile_from_raw_grams`], so the
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
) -> crate::identifiability::families::compiler::CompiledMap {
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

    crate::identifiability::families::compiler::CompiledMap {
        raw_from_compiled: t_full,
        compiled_block_ranges,
        raw_block_ranges,
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
#[allow(clippy::too_many_arguments)]
pub fn apply_compiled_map_to_designs(
    map: &crate::identifiability::families::compiler::CompiledMap,
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    time_penalties: &[crate::terms::smooth::BlockwisePenalty],
    marginal_penalties: &[crate::terms::smooth::BlockwisePenalty],
    logslope_penalties: &[crate::terms::smooth::BlockwisePenalty],
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
    // (`pull_back_penalty_through_t`) yields a `(p_compiled × p_compiled)`
    // dense matrix whose off-diagonal couples θ_b to earlier blocks' θ_a;
    // jamming that joint-width matrix into a single block's `penalties`
    // produced the `block 0 penalty 0 must be 12x12, got 17x17` mismatch
    // that surfaced as the `assert_valid_blockspecs` FFI panic.
    let pull_set = |pens: &[crate::terms::smooth::BlockwisePenalty],
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
        time_penalties,
        marginal_penalties,
        logslope_penalties,
        t_full: t.clone(),
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
pub fn compile_survival_parametric_designs(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    logslope_dg: Array2<f64>,
    row_hess: &dyn RowHessian,
) -> Result<SurvivalParametricCompiled, String> {
    use crate::identifiability::families::compiler::compile;

    let p_time_raw = time_dq0.ncols();
    let p_marg_raw = marginal_dq.ncols();
    let p_log_raw = logslope_dg.ncols();

    let inputs = build_survival_compiler_inputs(
        time_dq0,
        time_dq1,
        time_dqd1,
        marginal_dq,
        marginal_dqd1,
        logslope_dg,
        None,
        None,
    );
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
pub fn build_survival_compiler_inputs(
    time_dq0: Array2<f64>,
    time_dq1: Array2<f64>,
    time_dqd1: Array2<f64>,
    marginal_dq: Array2<f64>,
    marginal_dqd1: Array2<f64>,
    logslope_dg: Array2<f64>,
    score_warp_dq_dqd1: Option<(Array2<f64>, Array2<f64>)>,
    link_dev_dq_dqd1: Option<(Array2<f64>, Array2<f64>)>,
) -> SurvivalCompilerInputs {
    let mut operators: Vec<Arc<dyn RowJacobianOperator>> = Vec::with_capacity(5);
    let mut ordering: Vec<BlockOrder> = Vec::with_capacity(5);

    operators.push(Arc::new(TimeBlockOperator::new(
        time_dq0, time_dq1, time_dqd1,
    )));
    ordering.push(BlockOrder::Time);

    operators.push(Arc::new(QChannelBlockOperator::new(
        marginal_dq,
        marginal_dqd1,
    )));
    ordering.push(BlockOrder::Marginal);

    operators.push(Arc::new(LogslopeBlockOperator::new(logslope_dg)));
    ordering.push(BlockOrder::Logslope);

    if let Some((dq, dqd1)) = score_warp_dq_dqd1 {
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1)));
        ordering.push(BlockOrder::ScoreWarp);
    }
    if let Some((dq, dqd1)) = link_dev_dq_dqd1 {
        operators.push(Arc::new(QChannelBlockOperator::new(dq, dqd1)));
        ordering.push(BlockOrder::LinkDev);
    }

    SurvivalCompilerInputs {
        operators,
        ordering,
    }
}

/// Per-term-aware V+M-exact compiled designs + penalties + the full
/// triangular T matrix for result-time β lift. The construction site
/// swaps raw designs/penalties for these compiled versions before
/// building `ParameterBlockSpec`s; at fit result the joint compiled β
/// is lifted to raw via `T · θ` (where T is block-upper-triangular with
/// V_b on the diagonal and `-R_{a→b}` off-diagonal).
///
/// Difference from [`CompiledSurvivalDesignsPerTerm`]:
/// - emitted designs use [`ResidualisedDesignOperator`] (the exact
///   `C_b·V_b − A_{<b}·R_b` row form), not the V-only
///   `CoefficientTransformOperator`;
/// - per-term penalties are pulled back through the FULL triangular T
///   (not just V_b), so they are full-width `PenaltyMatrix::Dense`
///   matrices that may couple across blocks via the off-diagonal
///   `-R_{a→b}` entries of T;
/// - the joint result-time lift uses `t_full` rather than the per-block
///   V_b matrices on their own.
pub struct CompiledSurvivalDesignsVMExact {
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
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
    /// Full triangular T matrix: rows = raw joint width, cols = compiled
    /// joint width. Diagonal blocks are V_b; upper off-diagonal blocks
    /// at `(a, b)` for `a < b` are `-R_{a→b}` (residualised reparam
    /// against the earlier block). Used at result time to lift
    /// `θ_compiled → β_raw = T · θ_compiled`.
    pub t_full: Array2<f64>,
}

/// Project a raw-space warm start β_raw into compiled coordinates θ via the
/// Gram-aware least-squares formula
///
///     θ = (Tᵀ K^S T)^+ · Tᵀ K^S · β_raw
///
/// where K^S is the structural Gram on raw space. This is the correct
/// projection whenever T's columns are not Euclidean-orthonormal (e.g. the
/// V+M-exact compiled basis), because Vᵀ·β is only an orthogonal projector
/// in the Euclidean inner product, not in the K^S-induced inner product
/// the compiled coordinates live under.
pub fn project_raw_beta_to_compiled(
    t_full: &Array2<f64>,
    k_struct: &Array2<f64>,
    beta_raw: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let p_raw = t_full.nrows();
    let p_comp = t_full.ncols();
    if k_struct.nrows() != p_raw || k_struct.ncols() != p_raw {
        return Err(format!(
            "project_raw_beta_to_compiled: K^S shape {}x{} mismatches T rows {}",
            k_struct.nrows(),
            k_struct.ncols(),
            p_raw,
        ));
    }
    if beta_raw.len() != p_raw {
        return Err(format!(
            "project_raw_beta_to_compiled: beta_raw length {} mismatches T rows {}",
            beta_raw.len(),
            p_raw,
        ));
    }
    if p_comp == 0 {
        return Ok(Array1::<f64>::zeros(0));
    }
    if p_raw == 0 {
        return Ok(Array1::<f64>::zeros(p_comp));
    }

    // K^S · T  (p_raw × p_comp)
    let ks_t = fast_ab(k_struct, t_full);
    // M = Tᵀ K^S T  (p_comp × p_comp); compute as Tᵀ · (K^S T).
    let m = crate::linalg::faer_ndarray::fast_atb(t_full, &ks_t);
    // Symmetrize M before eigendecomposition.
    let mut m_sym = Array2::<f64>::zeros((p_comp, p_comp));
    for i in 0..p_comp {
        for j in 0..p_comp {
            m_sym[[i, j]] = 0.5 * (m[[i, j]] + m[[j, i]]);
        }
    }

    // Tᵀ K^S β_raw  (length p_comp): compute K^S β_raw, then Tᵀ · that.
    let ks_beta = crate::linalg::faer_ndarray::fast_av(k_struct, beta_raw);
    let rhs = crate::linalg::faer_ndarray::fast_atv(t_full, &ks_beta);

    let (evals, evecs) = m_sym
        .eigh(Side::Lower)
        .map_err(|e| format!("project_raw_beta_to_compiled: eigh failed: {e:?}"))?;

    let max_abs = evals.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (p_comp as f64) * f64::EPSILON * max_abs.max(1.0);

    // Apply M^+ via spectral decomposition: M^+ = U · diag(1/λ_i for λ_i>tol) · Uᵀ.
    let ut_rhs = crate::linalg::faer_ndarray::fast_atv(&evecs, &rhs);
    let mut scaled = Array1::<f64>::zeros(p_comp);
    for i in 0..p_comp {
        if evals[i].abs() > tol {
            scaled[i] = ut_rhs[i] / evals[i];
        }
    }
    let theta = crate::linalg::faer_ndarray::fast_av(&evecs, &scaled);
    Ok(theta)
}

/// Per-block, per-channel raw primary-state designs at block `a`. Each
/// channel design has shape `(n × raw_widths[a])` — the contribution of
/// block `a`'s raw coefficients to that primary-state channel's emitted
/// row. For blocks that don't contribute to a channel (e.g. a logslope
/// block contributes only to `g`), pass a zero design of the right shape.
#[derive(Debug, Clone)]
pub struct RawPrimaryBlockDesign {
    pub name: String,
    pub q0: DesignMatrix,
    pub q1: DesignMatrix,
    pub qd1: DesignMatrix,
    pub g: DesignMatrix,
}

/// Per-block, per-channel compiled primary-state designs at compiled
/// block `b`. Each channel design has shape `(n × p_b_compiled)` and is
/// the row-space realisation of the T-residualisation:
/// `compiled X_b^(c) = Σ_a raw X_a^(c) · T[raw_a_range, compiled_b_range]`.
/// For COMPILED logslope blocks the q0/q1/qd1 channels may be nonzero
/// because anchor subtraction (`−R_{a→b}` in the upper triangle of T)
/// can pull contributions from earlier time/marginal blocks into the
/// logslope block's coefficient column.
#[derive(Debug, Clone)]
pub struct CompiledPrimaryBlockDesign {
    pub name: String,
    pub q0: DesignMatrix,
    pub q1: DesignMatrix,
    pub qd1: DesignMatrix,
    pub g: DesignMatrix,
}

/// Build per-compiled-block, per-channel designs from per-raw-block,
/// per-channel raw designs and the full triangular reparam `t_full`
/// (rows = raw joint width, cols = compiled joint width).
///
/// For each compiled block `b` and channel `c`:
///   `compiled X_b^(c)  =  Σ_a raw X_a^(c) · T[raw_a_range, compiled_b_range]`
///
/// `n_rows` is the number of rows of every channel design — all raw
/// channel designs (including any zero-designs) must share this row
/// count.
pub fn materialise_compiled_primary_blocks(
    n_rows: usize,
    raw_channel_blocks: &[RawPrimaryBlockDesign],
    compiled_block_ranges: &[std::ops::Range<usize>],
    raw_block_ranges: &[std::ops::Range<usize>],
    t_full: &Array2<f64>,
) -> Result<Vec<CompiledPrimaryBlockDesign>, String> {
    let n_blocks = raw_channel_blocks.len();
    if compiled_block_ranges.len() != n_blocks {
        return Err(format!(
            "materialise_compiled_primary_blocks: compiled_block_ranges len {} != raw_channel_blocks len {}",
            compiled_block_ranges.len(),
            n_blocks,
        ));
    }
    if raw_block_ranges.len() != n_blocks {
        return Err(format!(
            "materialise_compiled_primary_blocks: raw_block_ranges len {} != raw_channel_blocks len {}",
            raw_block_ranges.len(),
            n_blocks,
        ));
    }

    let raw_joint_width: usize = raw_block_ranges.iter().map(|r| r.len()).sum();
    let compiled_joint_width: usize = compiled_block_ranges.iter().map(|r| r.len()).sum();
    if t_full.nrows() != raw_joint_width {
        return Err(format!(
            "materialise_compiled_primary_blocks: t_full has {} rows but raw joint width is {raw_joint_width}",
            t_full.nrows(),
        ));
    }
    if t_full.ncols() != compiled_joint_width {
        return Err(format!(
            "materialise_compiled_primary_blocks: t_full has {} cols but compiled joint width is {compiled_joint_width}",
            t_full.ncols(),
        ));
    }
    for (a, r) in raw_block_ranges.iter().enumerate() {
        if r.end > raw_joint_width {
            return Err(format!(
                "materialise_compiled_primary_blocks: raw_block_ranges[{a}] = {r:?} exceeds raw_joint_width {raw_joint_width}",
            ));
        }
    }
    for (b, r) in compiled_block_ranges.iter().enumerate() {
        if r.end > compiled_joint_width {
            return Err(format!(
                "materialise_compiled_primary_blocks: compiled_block_ranges[{b}] = {r:?} exceeds compiled_joint_width {compiled_joint_width}",
            ));
        }
    }

    let densify = |dm: &DesignMatrix,
                   raw_w: usize,
                   a: usize,
                   channel: &str|
     -> Result<Array2<f64>, String> {
        let dense = match dm {
            DesignMatrix::Dense(d) => d.to_dense(),
            DesignMatrix::Sparse(_) => dm
                .try_to_dense_by_chunks(&format!(
                    "materialise_compiled_primary_blocks: densify raw block {a} channel {channel}"
                ))
                .map_err(|e| {
                    format!(
                        "materialise_compiled_primary_blocks: densify raw block {a} channel {channel}: {e}"
                    )
                })?,
        };
        if dense.nrows() != n_rows {
            return Err(format!(
                "materialise_compiled_primary_blocks: raw block {a} channel {channel} has {} rows, expected {n_rows}",
                dense.nrows(),
            ));
        }
        if dense.ncols() != raw_w {
            return Err(format!(
                "materialise_compiled_primary_blocks: raw block {a} channel {channel} has {} cols, expected raw width {raw_w}",
                dense.ncols(),
            ));
        }
        Ok(dense)
    };

    let mut raw_q0: Vec<Array2<f64>> = Vec::with_capacity(n_blocks);
    let mut raw_q1: Vec<Array2<f64>> = Vec::with_capacity(n_blocks);
    let mut raw_qd1: Vec<Array2<f64>> = Vec::with_capacity(n_blocks);
    let mut raw_g: Vec<Array2<f64>> = Vec::with_capacity(n_blocks);
    for (a, blk) in raw_channel_blocks.iter().enumerate() {
        let raw_w = raw_block_ranges[a].len();
        raw_q0.push(densify(&blk.q0, raw_w, a, "q0")?);
        raw_q1.push(densify(&blk.q1, raw_w, a, "q1")?);
        raw_qd1.push(densify(&blk.qd1, raw_w, a, "qd1")?);
        raw_g.push(densify(&blk.g, raw_w, a, "g")?);
    }

    let mut out: Vec<CompiledPrimaryBlockDesign> = Vec::with_capacity(n_blocks);
    for (b, comp_range) in compiled_block_ranges.iter().enumerate() {
        let p_b = comp_range.len();
        let mut acc_q0 = Array2::<f64>::zeros((n_rows, p_b));
        let mut acc_q1 = Array2::<f64>::zeros((n_rows, p_b));
        let mut acc_qd1 = Array2::<f64>::zeros((n_rows, p_b));
        let mut acc_g = Array2::<f64>::zeros((n_rows, p_b));
        for (a, raw_range) in raw_block_ranges.iter().enumerate() {
            if raw_range.is_empty() || p_b == 0 {
                continue;
            }
            let t_ab = t_full
                .slice(ndarray::s![raw_range.clone(), comp_range.clone()])
                .to_owned();
            acc_q0 = acc_q0 + fast_ab(&raw_q0[a], &t_ab);
            acc_q1 = acc_q1 + fast_ab(&raw_q1[a], &t_ab);
            acc_qd1 = acc_qd1 + fast_ab(&raw_qd1[a], &t_ab);
            acc_g = acc_g + fast_ab(&raw_g[a], &t_ab);
        }
        out.push(CompiledPrimaryBlockDesign {
            name: raw_channel_blocks[b].name.clone(),
            q0: DesignMatrix::Dense(DenseDesignMatrix::from(acc_q0)),
            q1: DesignMatrix::Dense(DenseDesignMatrix::from(acc_q1)),
            qd1: DesignMatrix::Dense(DenseDesignMatrix::from(acc_qd1)),
            g: DesignMatrix::Dense(DenseDesignMatrix::from(acc_g)),
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gauge::Gauge;

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
        let clamped = psd_clamp_4x4(&m);
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
        let op = TimeBlockOperator::new(dq0.clone(), dq1.clone(), dqd1.clone());
        let full = op.evaluate_full();
        assert_eq!(full.shape(), &[n, p, K_SURVIVAL]);
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
        let op = QChannelBlockOperator::new(dq.clone(), dqd1.clone());
        let mut out = [0.0_f64; K_SURVIVAL];
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
        let op = LogslopeBlockOperator::new(dg.clone());
        let mut out = [0.0_f64; K_SURVIVAL];
        let delta = [2.0_f64, -1.0];
        op.apply_row(1, &delta, &mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        let want = dg[[1, 0]] * 2.0 + dg[[1, 1]] * (-1.0);
        assert!((out[3] - want).abs() < 1e-12);
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
    fn block_diag_from_assembles_correctly() {
        let v1 = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| (i * 2 + j + 1) as f64);
        let v2 = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| (10 + i * 2 + j) as f64);
        let bd = block_diag_from(&[v1.clone(), v2.clone()]);
        assert_eq!(bd.dim(), (5, 4));
        // Top-left = v1, bottom-right = v2, off-blocks = zero.
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(bd[[i, j]], v1[[i, j]]);
                assert_eq!(bd[[i, 2 + j]], 0.0);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(bd[[3 + i, 2 + j]], v2[[i, j]]);
                assert_eq!(bd[[3 + i, j]], 0.0);
            }
        }
    }

    #[test]
    fn pull_back_blockwise_penalty_per_term_full_term_identity_v() {
        use crate::terms::smooth::BlockwisePenalty;
        // Single-term partition, identity V → pullback returns input.
        let v_term = Array2::<f64>::eye(3);
        let v_per_term = vec![v_term];
        let term_range = 0..3usize;
        let partition = std::slice::from_ref(&term_range);
        let local = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (i + j) as f64);
        let pen = BlockwisePenalty::new(0..3, local.clone());
        let out = pull_back_blockwise_penalty_per_term(&pen, partition, &v_per_term)
            .expect("identity-V pullback must succeed");
        assert_eq!(out.col_range, 0..3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((out.local[[i, j]] - 0.5 * (local[[i, j]] + local[[j, i]])).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pull_back_blockwise_penalty_per_term_drops_one_column() {
        use crate::terms::smooth::BlockwisePenalty;
        // 3-column term, V drops one column (3 → 2): V is the 3×2
        // matrix with columns e_1 and e_2 (drop the constant).
        let mut v_term = Array2::<f64>::zeros((3, 2));
        v_term[[1, 0]] = 1.0;
        v_term[[2, 1]] = 1.0;
        let term_range = 0..3usize;
        let partition = std::slice::from_ref(&term_range);
        let local = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (i + j + 1) as f64);
        let pen = BlockwisePenalty::new(0..3, local.clone());
        let out = pull_back_blockwise_penalty_per_term(&pen, partition, &[v_term])
            .expect("selection-V pullback must succeed");
        assert_eq!(out.col_range, 0..2);
        // Expected: pullback is the (1..3, 1..3) sub-block of the
        // symmetric part of `local`.
        let sym = |i: usize, j: usize| 0.5 * (local[[i, j]] + local[[j, i]]);
        assert!((out.local[[0, 0]] - sym(1, 1)).abs() < 1e-12);
        assert!((out.local[[0, 1]] - sym(1, 2)).abs() < 1e-12);
        assert!((out.local[[1, 0]] - sym(2, 1)).abs() < 1e-12);
        assert!((out.local[[1, 1]] - sym(2, 2)).abs() < 1e-12);
    }

    #[test]
    fn pull_back_blockwise_penalty_per_term_routes_to_correct_term() {
        use crate::terms::smooth::BlockwisePenalty;
        // Two-term partition: term0 = 0..2, term1 = 2..5.
        // V_term0 = 2×2 identity; V_term1 drops 1 col (3 → 2).
        let v0 = Array2::<f64>::eye(2);
        let mut v1 = Array2::<f64>::zeros((3, 2));
        v1[[0, 0]] = 1.0;
        v1[[2, 1]] = 1.0;
        let partition = vec![0..2, 2..5];
        let v_per_term = vec![v0, v1];
        // Penalty over term1 only (col_range 2..5).
        let local1 = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (10 + i + j) as f64);
        let pen1 = BlockwisePenalty::new(2..5, local1.clone());
        let out1 = pull_back_blockwise_penalty_per_term(&pen1, &partition, &v_per_term)
            .expect("term1 pullback must succeed");
        // Compiled col_range for term1 = (0+2)..(0+2)+v1.ncols() = 2..4.
        assert_eq!(out1.col_range, 2..4);
        // Pullback expected: V1ᵀ · sym(local1) · V1, which picks rows/
        // cols (0, 2) of sym(local1).
        let sym = |i: usize, j: usize| 0.5 * (local1[[i, j]] + local1[[j, i]]);
        assert!((out1.local[[0, 0]] - sym(0, 0)).abs() < 1e-12);
        assert!((out1.local[[0, 1]] - sym(0, 2)).abs() < 1e-12);
        assert!((out1.local[[1, 0]] - sym(2, 0)).abs() < 1e-12);
        assert!((out1.local[[1, 1]] - sym(2, 2)).abs() < 1e-12);
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
    fn build_global_t_matrix_identity_when_all_v_eye_and_all_r_none() {
        // Three top-level blocks (time / marginal / logslope) with two
        // terms each, every V = I_2 and every R = None → global T = I.
        let mk_block = || GlobalTBlock {
            v: Array2::<f64>::eye(2),
            r_against_earlier: None,
        };
        let spec = GlobalTSpec {
            blocks: vec![
                vec![mk_block(), mk_block()],
                vec![mk_block(), mk_block()],
                vec![mk_block(), mk_block()],
            ],
        };
        let t = build_global_t_matrix(&spec);
        let total: usize = 2 * 6;
        assert_eq!(t.dim(), (total, total));
        let eye = Array2::<f64>::eye(total);
        for i in 0..total {
            for j in 0..total {
                assert!((t[[i, j]] - eye[[i, j]]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn build_global_t_matrix_block_upper_triangular_with_cross_block_r() {
        // Two top-level blocks, each with one term. Block 0: V_a = I_2,
        // R = None. Block 1: V_b = I_2, R_against_earlier is a 2x2 dense
        // block. Global T should have V_a on (0..2, 0..2), V_b on
        // (2..4, 2..4), and -R on (0..2, 2..4).
        let r_block = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| 1.0 + (i + 2 * j) as f64);
        let spec = GlobalTSpec {
            blocks: vec![
                vec![GlobalTBlock {
                    v: Array2::<f64>::eye(2),
                    r_against_earlier: None,
                }],
                vec![GlobalTBlock {
                    v: Array2::<f64>::eye(2),
                    r_against_earlier: Some(r_block.clone()),
                }],
            ],
        };
        let t = build_global_t_matrix(&spec);
        assert_eq!(t.dim(), (4, 4));
        for i in 0..2 {
            for j in 0..2 {
                let want_diag_a = if i == j { 1.0 } else { 0.0 };
                let want_diag_b = if i == j { 1.0 } else { 0.0 };
                assert!((t[[i, j]] - want_diag_a).abs() < 1e-14);
                assert!((t[[2 + i, 2 + j]] - want_diag_b).abs() < 1e-14);
                assert!((t[[i, 2 + j]] + r_block[[i, j]]).abs() < 1e-14);
                assert_eq!(t[[2 + i, j]], 0.0);
            }
        }
    }

    #[test]
    fn pull_back_penalty_through_t_identity_returns_zero_embedded_raw() {
        use crate::terms::smooth::BlockwisePenalty;
        let v_a = Array2::<f64>::eye(2);
        let v_b = Array2::<f64>::eye(2);
        let t = assemble_block_triangular_t(&[v_a, v_b], &[None, None]);
        let local = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| (i + 2 * j + 1) as f64);
        let pen = BlockwisePenalty::new(0..2, local.clone());
        let anchor_offset = 2usize;
        let out = pull_back_penalty_through_t(&pen, anchor_offset, &t);
        let PenaltyMatrix::Dense(dense) = out else {
            panic!("expected PenaltyMatrix::Dense");
        };
        assert_eq!(dense.dim(), (4, 4));
        let sym_local = |i: usize, j: usize| 0.5 * (local[[i, j]] + local[[j, i]]);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i >= 2 && j >= 2 {
                    sym_local(i - 2, j - 2)
                } else {
                    0.0
                };
                assert!(
                    (dense[[i, j]] - want).abs() < 1e-14,
                    "mismatch at ({i},{j}): got {}, want {}",
                    dense[[i, j]],
                    want,
                );
            }
        }
    }

    #[test]
    fn pull_back_penalty_through_t_nontrivial_t_has_off_block_coupling() {
        use crate::terms::smooth::BlockwisePenalty;
        let v_a = Array2::<f64>::eye(2);
        let v_b = Array2::<f64>::eye(2);
        let r_ab = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| 1.0 + (i + j) as f64);
        let t = assemble_block_triangular_t(&[v_a, v_b], &[None, Some(r_ab.clone())]);
        let mut local = Array2::<f64>::zeros((2, 2));
        local[[0, 0]] = 2.0;
        local[[1, 1]] = 3.0;
        let pen = BlockwisePenalty::new(0..2, local.clone());
        let out = pull_back_penalty_through_t(&pen, 0, &t);
        let PenaltyMatrix::Dense(dense) = out else {
            panic!("expected PenaltyMatrix::Dense");
        };
        assert_eq!(dense.dim(), (4, 4));
        let mut any_nonzero = false;
        for i in 0..2 {
            for j in 0..2 {
                if dense[[i, 2 + j]].abs() > 1e-10 {
                    any_nonzero = true;
                }
            }
        }
        assert!(any_nonzero, "expected nonzero off-block coupling");
        let want = local.dot(&r_ab).map(|v| -v);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (dense[[i, 2 + j]] - want[[i, j]]).abs() < 1e-12,
                    "off-block (a,b) mismatch at ({i},{j}): got {}, want {}",
                    dense[[i, 2 + j]],
                    want[[i, j]],
                );
                assert!((dense[[2 + j, i]] - want[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pull_back_penalty_through_t_round_trip_quadratic_form() {
        use crate::terms::smooth::BlockwisePenalty;
        let v_a = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| {
            ((i + 1) as f64).sin() + 0.3 * (j as f64)
        });
        let v_b = Array2::<f64>::from_shape_fn((2, 2), |(i, j)| 1.0 + 0.1 * ((i * 2 + j) as f64));
        let r_ab = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| {
            0.5 - 0.2 * (i as f64) + 0.1 * (j as f64)
        });
        let t =
            assemble_block_triangular_t(&[v_a.clone(), v_b.clone()], &[None, Some(r_ab.clone())]);
        let raw_local =
            Array2::<f64>::from_shape_fn(
                (2, 2),
                |(i, j)| {
                    if i == j { 2.5 + i as f64 } else { 0.4 }
                },
            );
        let pen = BlockwisePenalty::new(0..2, raw_local.clone());
        let anchor = 3usize;
        let out = pull_back_penalty_through_t(&pen, anchor, &t);
        let PenaltyMatrix::Dense(s_compiled) = out else {
            panic!("expected Dense");
        };
        let raw_total = t.nrows();
        let mut s_raw_emb = Array2::<f64>::zeros((raw_total, raw_total));
        for i in 0..2 {
            for j in 0..2 {
                s_raw_emb[[anchor + i, anchor + j]] = raw_local[[i, j]];
            }
        }
        let mut s_raw_sym = Array2::<f64>::zeros((raw_total, raw_total));
        for i in 0..raw_total {
            for j in 0..raw_total {
                s_raw_sym[[i, j]] = 0.5 * (s_raw_emb[[i, j]] + s_raw_emb[[j, i]]);
            }
        }
        let theta = Array1::<f64>::from_shape_fn(t.ncols(), |k| ((k as f64) * 0.7 - 0.3).cos());
        let gamma = t.dot(&theta);
        let lhs = theta.dot(&s_compiled.dot(&theta));
        let rhs = gamma.dot(&s_raw_sym.dot(&gamma));
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "round-trip mismatch: lhs={lhs}, rhs={rhs}",
        );
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

    /// Regression for #368: the phase-4b compiled-map penalty pullback must
    /// emit a PER-BLOCK-WIDTH penalty for every block (sized to that block's
    /// COMPILED design width), even when a block drops columns and the
    /// triangular T carries nonzero off-diagonal cross-block residualisation
    /// `R_{a→b}`. The original bug pulled penalties back through the full
    /// joint T (`Tᵀ S T`), producing joint-compiled-width penalties (e.g.
    /// 7×7) that did not fit a single per-block `ParameterBlockSpec.penalties`
    /// slot (e.g. time block compiled width 3), making `validate_blockspecs`
    /// fail and `assert_valid_blockspecs` panic across the FFI boundary on
    /// ordinary survival data.
    #[test]
    fn compiled_map_penalty_pullback_is_per_block_width_with_nonzero_residual() {
        use crate::identifiability::families::compiler::CompiledMap;
        use crate::terms::smooth::BlockwisePenalty;

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

        let out = apply_compiled_map_to_designs(
            &map,
            raw_time_entry,
            raw_time_exit,
            raw_time_deriv,
            raw_marg,
            raw_log,
            &time_pens,
            &marg_pens,
            &log_pens,
        )
        .expect("apply_compiled_map_to_designs must succeed");

        // Designs carry per-block compiled widths.
        assert_eq!(out.time_design_entry.ncols(), 3);
        assert_eq!(out.marginal_design.ncols(), 2);
        assert_eq!(out.logslope_design.ncols(), 2);

        // Core invariant the bug violated: every penalty is sized to ITS
        // OWN block's compiled width, NOT the joint compiled width (7).
        for s in &out.time_penalties {
            assert_eq!(
                s.as_dense_cow().dim(),
                (3, 3),
                "time penalty must be per-block 3×3, not joint-width"
            );
        }
        for s in &out.marginal_penalties {
            assert_eq!(
                s.as_dense_cow().dim(),
                (2, 2),
                "marginal penalty must match reduced compiled width 2, not joint 7"
            );
        }
        for s in &out.logslope_penalties {
            assert_eq!(s.as_dense_cow().dim(), (2, 2));
        }

        // For the time block (block 0, no anchor ⇒ R=None), the per-block
        // pullback is EXACT: θ_timeᵀ P_time θ_time == γ_timeᵀ S_time γ_time
        // with γ_time = V_time · θ_time. Verify the quadratic-form identity.
        let p_time_dense = out.time_penalties[0].as_dense_cow().into_owned();
        let theta_time = Array1::<f64>::from_shape_fn(3, |k| 0.4 + 0.7 * (k as f64));
        let gamma_time = v_time.dot(&theta_time);
        let lhs = theta_time.dot(&p_time_dense.dot(&theta_time));
        let rhs = gamma_time.dot(&s_time.dot(&gamma_time));
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "time-block per-block pullback must be exact: lhs={lhs}, rhs={rhs}"
        );

        // The marginal pullback must equal V_margᵀ S_marg V_marg exactly
        // (block-local; the cross-block R_marg lives in the design, not here).
        let p_marg_dense = out.marginal_penalties[0].as_dense_cow().into_owned();
        let want_marg = v_marg.t().dot(&s_marg.dot(&v_marg));
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (p_marg_dense[[i, j]] - want_marg[[i, j]]).abs() < 1e-12,
                    "marginal penalty must be V_margᵀ S_marg V_marg at ({i},{j})"
                );
            }
        }
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
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for k in 0..K_SURVIVAL {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = SurvivalRowHessian::from_full(h_full);
        let out = compile_survival_parametric_designs(
            time_dq0, time_dq1, time_dqd1, marg_dq, marg_dqd1, log_dg, &row_hess,
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
    /// path on the survival K=4 row primary state without the SMGS construction
    /// hook the active sibling integration provides. The shape of the result is
    /// the contract the SMGS wiring will assert against.
    #[test]
    fn compile_survival_three_block_with_shared_constant_drops_one_direction() {
        use crate::identifiability::families::compiler::compile;

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
            time_dq0, time_dq1, time_dqd1, marg_dq, marg_dqd1, log_dg, None, None,
        );

        // Identity 4×4 row Hessian on every row. With H_i = I the
        // sqrt-H metric collapses to the standard Frobenius metric,
        // so the compiler's residualisation is ordinary least-squares
        // projection — exactly what we want for verifying the
        // structural rank-deficiency attribution.
        let mut h_full = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for k in 0..K_SURVIVAL {
                h_full[[i, k, k]] = 1.0;
            }
        }
        let row_hess = SurvivalRowHessian::from_full(h_full);

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

    #[test]
    fn project_raw_beta_identity_t_and_k_is_identity() {
        let p = 5;
        let t_full = Array2::<f64>::eye(p);
        let k_struct = Array2::<f64>::eye(p);
        let beta_raw = Array1::from_vec(vec![0.3_f64, -1.1, 2.5, 0.0, 4.2]);
        let theta = project_raw_beta_to_compiled(&t_full, &k_struct, &beta_raw)
            .expect("identity projection should succeed");
        assert_eq!(theta.len(), p);
        for i in 0..p {
            assert!(
                (theta[i] - beta_raw[i]).abs() < 1e-12,
                "identity projection mismatch at {i}: got {} expected {}",
                theta[i],
                beta_raw[i]
            );
        }
    }

    #[test]
    fn project_raw_beta_column_selector_recovers_kept_indices() {
        // Selection-V case: T picks columns 0, 2, 4 from a p_raw=5 raw space.
        let p_raw = 5;
        let kept = [0usize, 2, 4];
        let p_comp = kept.len();
        let mut t_full = Array2::<f64>::zeros((p_raw, p_comp));
        for (j, &i) in kept.iter().enumerate() {
            t_full[[i, j]] = 1.0;
        }
        // Use a diagonal K^S with positive entries on the kept rows; off-kept
        // entries are irrelevant to the projection (selector zeroes them).
        let mut k_struct = Array2::<f64>::zeros((p_raw, p_raw));
        for i in 0..p_raw {
            k_struct[[i, i]] = (i as f64 + 1.0) * 0.7;
        }
        let beta_raw = Array1::from_vec(vec![1.5_f64, -0.4, 2.0, 9.9, -3.3]);
        let theta = project_raw_beta_to_compiled(&t_full, &k_struct, &beta_raw)
            .expect("selector projection should succeed");
        for (j, &i) in kept.iter().enumerate() {
            assert!(
                (theta[j] - beta_raw[i]).abs() < 1e-10,
                "selector projection at compiled idx {j} (raw {i}): got {} expected {}",
                theta[j],
                beta_raw[i]
            );
        }
    }

    #[test]
    fn project_raw_beta_roundtrip_on_structural_positive_subspace() {
        // T is a thin orthonormal-ish basis with non-trivial structure; build
        // β_raw = T·θ on a vector strictly inside the structural-positive
        // subspace, then project back and expect θ to be recovered.
        let p_raw = 6;
        let p_comp = 3;
        // Construct T from Gram-Schmidt on a deterministic data matrix.
        let raw_cols = Array2::from_shape_fn((p_raw, p_comp), |(i, j)| {
            ((i as f64) * 0.31 + (j as f64) * 1.7 + 0.5).sin() + (j as f64)
        });
        // Orthonormalize columns of raw_cols to get T (Euclidean QR-like).
        let mut t_full = raw_cols.clone();
        for j in 0..p_comp {
            for k in 0..j {
                let mut dot = 0.0;
                for i in 0..p_raw {
                    dot += t_full[[i, j]] * t_full[[i, k]];
                }
                for i in 0..p_raw {
                    t_full[[i, j]] -= dot * t_full[[i, k]];
                }
            }
            let mut nrm = 0.0;
            for i in 0..p_raw {
                nrm += t_full[[i, j]].powi(2);
            }
            nrm = nrm.sqrt();
            assert!(nrm > 1e-10, "column {j} collapsed in QR");
            for i in 0..p_raw {
                t_full[[i, j]] /= nrm;
            }
        }
        // K^S is positive on the column span of T (use K = T·diag(d)·Tᵀ + tiny
        // positive on its complement so the full Gram is PD).
        let d = Array1::from_vec(vec![2.0_f64, 0.7, 1.3]);
        let mut k_struct = Array2::<f64>::zeros((p_raw, p_raw));
        for i in 0..p_raw {
            for j in 0..p_raw {
                let mut s = 0.0;
                for r in 0..p_comp {
                    s += t_full[[i, r]] * d[r] * t_full[[j, r]];
                }
                k_struct[[i, j]] = s;
            }
            k_struct[[i, i]] += 1e-6;
        }
        let theta_true = Array1::from_vec(vec![0.7_f64, -1.4, 2.1]);
        // β_raw = T · θ_true lies entirely in T's range.
        let beta_raw = crate::linalg::faer_ndarray::fast_av(&t_full, &theta_true);
        let theta = project_raw_beta_to_compiled(&t_full, &k_struct, &beta_raw)
            .expect("round-trip projection should succeed");
        for i in 0..p_comp {
            assert!(
                (theta[i] - theta_true[i]).abs() < 1e-8,
                "round-trip mismatch at {i}: got {} expected {}",
                theta[i],
                theta_true[i]
            );
        }
    }

    fn make_raw_block(
        name: &str,
        n_rows: usize,
        raw_w: usize,
        q0: Option<Array2<f64>>,
        q1: Option<Array2<f64>>,
        qd1: Option<Array2<f64>>,
        g: Option<Array2<f64>>,
    ) -> RawPrimaryBlockDesign {
        let zero = || Array2::<f64>::zeros((n_rows, raw_w));
        let wrap =
            |a: Array2<f64>| -> DesignMatrix { DesignMatrix::Dense(DenseDesignMatrix::from(a)) };
        RawPrimaryBlockDesign {
            name: name.to_string(),
            q0: wrap(q0.unwrap_or_else(zero)),
            q1: wrap(q1.unwrap_or_else(zero)),
            qd1: wrap(qd1.unwrap_or_else(zero)),
            g: wrap(g.unwrap_or_else(zero)),
        }
    }

    fn dense_of(dm: &DesignMatrix) -> Array2<f64> {
        match dm {
            DesignMatrix::Dense(d) => d.to_dense(),
            DesignMatrix::Sparse(_) => panic!("expected Dense"),
        }
    }

    #[test]
    fn materialise_compiled_primary_blocks_block_diagonal_t_no_cross_block() {
        // 3 raw blocks: time (raw_w=2), marginal (raw_w=2), logslope (raw_w=2).
        // T is block-diagonal (identity) — no off-diag anchor subtraction.
        // Compiled q0 for block_marg should equal raw marg q0 directly.
        let n = 5;
        let raw_ranges: Vec<std::ops::Range<usize>> = vec![0..2, 2..4, 4..6];
        let compiled_ranges: Vec<std::ops::Range<usize>> = vec![0..2, 2..4, 4..6];

        let x_time_q0 = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64 + 1.0);
        let x_time_q1 = Array2::from_shape_fn((n, 2), |(i, j)| 2.0 * (i as f64) + j as f64);
        let x_time_qd1 = Array2::from_shape_fn((n, 2), |(i, j)| 0.5 * (i as f64) - j as f64);
        let x_marg_q0 = Array2::from_shape_fn((n, 2), |(i, j)| 0.3 + (i * 3 + j) as f64);
        let x_marg_q1 = x_marg_q0.clone();
        let x_log_g = Array2::from_shape_fn((n, 2), |(i, j)| 7.0 - (i as f64) + 0.1 * (j as f64));

        let raw_blocks = vec![
            make_raw_block(
                "time",
                n,
                2,
                Some(x_time_q0.clone()),
                Some(x_time_q1.clone()),
                Some(x_time_qd1.clone()),
                None,
            ),
            make_raw_block(
                "marginal",
                n,
                2,
                Some(x_marg_q0.clone()),
                Some(x_marg_q1.clone()),
                None,
                None,
            ),
            make_raw_block("logslope", n, 2, None, None, None, Some(x_log_g.clone())),
        ];

        let mut t_full = Array2::<f64>::zeros((6, 6));
        for i in 0..6 {
            t_full[[i, i]] = 1.0;
        }

        let out = materialise_compiled_primary_blocks(
            n,
            &raw_blocks,
            &compiled_ranges,
            &raw_ranges,
            &t_full,
        )
        .expect("materialise should succeed on block-diagonal T");
        assert_eq!(out.len(), 3);

        let comp_marg_q0 = dense_of(&out[1].q0);
        assert_eq!(comp_marg_q0.shape(), &[n, 2]);
        for i in 0..n {
            for j in 0..2 {
                assert!(
                    (comp_marg_q0[[i, j]] - x_marg_q0[[i, j]]).abs() < 1e-12,
                    "block-diag T: compiled marg q0 mismatch at ({i},{j}): got {} want {}",
                    comp_marg_q0[[i, j]],
                    x_marg_q0[[i, j]],
                );
            }
        }
        // No cross-block bleed: compiled time q0 equals raw time q0.
        let comp_time_q0 = dense_of(&out[0].q0);
        for i in 0..n {
            for j in 0..2 {
                assert!((comp_time_q0[[i, j]] - x_time_q0[[i, j]]).abs() < 1e-12);
            }
        }
        // Compiled logslope q0 is all zero.
        let comp_log_q0 = dense_of(&out[2].q0);
        for i in 0..n {
            for j in 0..2 {
                assert!(comp_log_q0[[i, j]].abs() < 1e-14);
            }
        }
    }

    #[test]
    fn materialise_compiled_primary_blocks_with_anchor_subtraction_pulls_q0_into_logslope() {
        // 3 raw blocks. T has identity on the diagonal and a nonzero −R
        // at T[time_range, logslope_range]. Raw logslope contributes nothing
        // to q0 directly, but compiled logslope's q0 design must equal
        // raw time q0 · (−R).
        let n = 4;
        let raw_ranges: Vec<std::ops::Range<usize>> = vec![0..2, 2..4, 4..6];
        let compiled_ranges: Vec<std::ops::Range<usize>> = vec![0..2, 2..4, 4..6];

        let x_time_q0 = Array2::from_shape_fn((n, 2), |(i, j)| (i + 2 * j + 1) as f64);
        let x_marg_q0 = Array2::from_shape_fn((n, 2), |(i, j)| 0.5 + (i as f64) - (j as f64));
        let x_log_g = Array2::from_shape_fn((n, 2), |(i, j)| 1.0 + 0.25 * (i + j) as f64);

        let raw_blocks = vec![
            make_raw_block("time", n, 2, Some(x_time_q0.clone()), None, None, None),
            make_raw_block("marginal", n, 2, Some(x_marg_q0.clone()), None, None, None),
            make_raw_block("logslope", n, 2, None, None, None, Some(x_log_g.clone())),
        ];

        let r_time_log = Array2::from_shape_vec((2, 2), vec![0.7, -0.3, 0.1, 0.9]).unwrap();
        let mut t_full = Array2::<f64>::zeros((6, 6));
        for i in 0..6 {
            t_full[[i, i]] = 1.0;
        }
        for i in 0..2 {
            for j in 0..2 {
                t_full[[i, 4 + j]] = -r_time_log[[i, j]];
            }
        }

        let out = materialise_compiled_primary_blocks(
            n,
            &raw_blocks,
            &compiled_ranges,
            &raw_ranges,
            &t_full,
        )
        .expect("materialise should succeed with anchor subtraction");

        let comp_log_q0 = dense_of(&out[2].q0);
        assert_eq!(comp_log_q0.shape(), &[n, 2]);
        let neg_r = -&r_time_log;
        let expected = fast_ab(&x_time_q0, &neg_r);
        let mut max_abs = 0.0_f64;
        for i in 0..n {
            for j in 0..2 {
                max_abs = max_abs.max(expected[[i, j]].abs());
            }
        }
        assert!(
            max_abs > 1e-6,
            "test construction error: expected compiled logslope q0 to be nonzero",
        );
        for i in 0..n {
            for j in 0..2 {
                assert!(
                    (comp_log_q0[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                    "compiled logslope q0 mismatch at ({i},{j}): got {} want {}",
                    comp_log_q0[[i, j]],
                    expected[[i, j]],
                );
            }
        }

        // Compiled logslope g equals raw logslope g (no off-diag in g-affecting cols).
        let comp_log_g = dense_of(&out[2].g);
        for i in 0..n {
            for j in 0..2 {
                assert!((comp_log_g[[i, j]] - x_log_g[[i, j]]).abs() < 1e-12);
            }
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
        // No logslope columns.
        let log_dg = Array2::<f64>::zeros((n, 0));
        let mut time_partition: Vec<std::ops::Range<usize>> = Vec::with_capacity(1);
        time_partition.push(0..1);
        let mut marg_partition: Vec<std::ops::Range<usize>> = Vec::with_capacity(1);
        marg_partition.push(0..1);
        let log_partition: Vec<std::ops::Range<usize>> = Vec::new();

        // Pass 1: structural identity row Hessian. q0/q1/qd1/g all weighted
        // equally → marg's q1 component is visible, so marg is identifiable
        // after residualising against the time block (drops_marg = 0).
        let mut h_ident = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            for k in 0..K_SURVIVAL {
                h_ident[[i, k, k]] = 1.0;
            }
        }
        let row_hess_ident = SurvivalRowHessian::from_full(h_ident);
        let compiled_ident = compile_survival_parametric_designs_per_term(
            time_dq0.clone(),
            time_dq1.clone(),
            time_dqd1.clone(),
            &time_partition,
            marg_dq.clone(),
            marg_dqd1.clone(),
            &marg_partition,
            log_dg.clone(),
            &log_partition,
            &row_hess_ident,
        )
        .expect("identity-H compile must succeed");

        // Pass 2: data-adaptive row Hessian that only weighs q0 (all
        // other channel diagonals zero). Marg's q1 contribution is now
        // invisible → marg fully aliases with time on q0 → drops_marg = 1.
        let mut h_q0_only = Array3::<f64>::zeros((n, K_SURVIVAL, K_SURVIVAL));
        for i in 0..n {
            h_q0_only[[i, 0, 0]] = 1.0;
        }
        let row_hess_q0 = SurvivalRowHessian::from_full(h_q0_only);
        let compiled_q0 = compile_survival_parametric_designs_per_term(
            time_dq0,
            time_dq1,
            time_dqd1,
            &time_partition,
            marg_dq,
            marg_dqd1,
            &marg_partition,
            log_dg,
            &log_partition,
            &row_hess_q0,
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
            crate::identifiability::families::compiler::BlockOrder::Time,
            crate::identifiability::families::compiler::BlockOrder::Marginal,
            crate::identifiability::families::compiler::BlockOrder::Logslope,
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
}
