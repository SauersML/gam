use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::{new_cell_moment_cache_stats, new_cell_moment_lru_cache};
use super::install_flex::validate_spec;
use super::*;
use crate::faer_ndarray::{FaerEigh, fast_ab, fast_atb, fast_xt_diag_x};
use crate::families::marginal_slope_orthogonal::INFLUENCE_ABSORBER_FIXED_LOG_LAMBDA;
use faer::Side;

/// Sup-norm of the FITTED marginal linear predictor `η = X·β` at which the
/// probit model becomes numerically degenerate. This is the decisive
/// separation quantity — raw coefficient magnitude is NOT, because the
/// thin-plate / Duchon marginal bases are non-orthonormal and ill-conditioned,
/// so a smooth, bounded fitted surface can carry large-and-cancelling
/// coefficients (e.g. `β=[60,-60]` on two collinear columns yields a bounded
/// `Xβ`). The probit information `φ(η)²/[Φ(η)(1−Φ(η))]` only collapses once
/// `|η|` is enormous: `Φ(−35) ≈ 1e−268` is still representable, but by `|η|≈38`
/// the tail probability underflows to exactly 0 in `f64` and the per-row
/// Fisher weight vanishes. We trip the guard at 35 — comfortably inside the
/// representable range yet far beyond any legitimately fitted predictor (a
/// converged penalized probit surface keeps `|η|` at single/low-double digits),
/// so a true separating direction (whose `|η|→∞`) still trips it while a
/// well-fitted ill-conditioned surface does not.
pub(crate) const BMS_PROBIT_SEPARATION_ETA_INF: f64 = 35.0;

// ── Canonical-gauge priority ladder (issue #322) ─────────────────────────────
//
// The priority-ordered RRQR in `canonicalize_for_identifiability` presents
// higher-priority blocks first and routes any shared cross-block alias drop
// into the lowest-priority block that still spans the aliased direction. The
// values below form a single ordered ladder so the relationships that the
// architecture depends on (anchors > parametric surfaces > flex deviations,
// marginal > logslope, score_warp > link_dev) are expressed once here rather
// than re-derived from comments at each `gauge_priority:` site. The ladder
// mirrors the survival marginal-slope entry (time=200 / marginal=150 /
// logslope=120 / score_warp=80 / link_dev=60).

/// Audit-only anchor blocks sit at the top of the ladder so the candidate
/// flex block always yields to them in the cross-block identifiability audit.
pub(super) const GAUGE_PRIORITY_ANCHOR: u8 = 200;
/// Marginal surface: strictly above the logslope surface so a shared affine
/// direction is demoted out of logslope, never out of marginal.
pub(super) const GAUGE_PRIORITY_MARGINAL: u8 = 150;
/// Logslope surface: one rung below the marginal surface.
pub(super) const GAUGE_PRIORITY_LOGSLOPE: u8 = 120;
/// Candidate flex block under audit: below every parametric anchor so the
/// audit demotes the candidate when it aliases an anchor.
pub(super) const GAUGE_PRIORITY_CANDIDATE_FLEX: u8 = 100;
/// `score_warp_dev`: above `link_dev` because in mixed-flex configurations
/// link_dev is the residualised block and should yield first.
pub(super) const GAUGE_PRIORITY_SCORE_WARP_DEV: u8 = 80;
/// Default for any deviation auxiliary block not otherwise named; below the
/// parametric default so shared affine directions never demote a parametric
/// block.
pub(super) const GAUGE_PRIORITY_DEVIATION_DEFAULT: u8 = 70;
/// `link_dev`: lowest rung, yields first among the flex deviation blocks.
pub(super) const GAUGE_PRIORITY_LINK_DEV: u8 = 60;

/// Floor on the relative outer tolerance used by the exact-joint spatial
/// length-scale optimiser. The user's `rel_tol` drives most of the fit, but
/// the exact spatial outer loop is a coarse 1-D search over a smooth profiled
/// objective; tightening it below this floor only burns cycles on noise in the
/// inner-solve-reported objective without moving the selected length scale.
pub(crate) const EXACT_SPATIAL_OUTER_TOL_FLOOR: f64 = 1e-6;

// ── BlockEffectiveJacobian impls for BMS ─────────────────────────────────────
//
// BMS has a single Bernoulli output per row (n_outputs = 1). The observed η is
//
//   η_i = q_i · c_i + s·g_i · z_i
//
// where
//   q_i   = marginal_design[i,:] · β_m + offset_m[i]      (marginal η)
//   g_i   = logslope_design[i,:] · β_s + offset_s[i]      (log-slope η)
//   s     = probit_frailty_scale(gaussian_frailty_sd)
//   c_i   = sqrt(1 + (s·g_i)²)
//
// Per-block Jacobians ∂η_i / ∂β_block:
//
//   Marginal block  → ∂η_i/∂β_m = c_i · M_i
//     (M_i = marginal_design row i; c_i is β-dependent but does not involve β_m)
//
//   Logslope block  → ∂η_i/∂β_s = (q_i · s²·g_i / c_i + s·z_i) · G_i
//     (G_i = logslope_design row i)
//
// score_warp_dev and link_dev blocks use IFT-corrected η, but their
// contribution to the identifiability audit is captured by the raw design
// columns (the IFT correction adds a direction already in the anchor span at
// compile time). These blocks leave jacobian_callback = None and rely on
// effective_design (= raw design) for the flat audit.

/// β-dependent Jacobian for the BMS marginal block.
///
/// ∂η_i/∂β_m = c_i · M[i,:]
/// where c_i = sqrt(1 + (s · g_i)²),
///       g_i = G[i,:] · β_s + offset_s[i],
///       s   = state.probit_frailty_scale.
///
/// `probit_frailty_scale` is read from the evaluation state at call time (not
/// captured at construction) so the callback remains correct across outer-loop
/// σ updates without rebuilding the block spec.
///
/// Designs are pre-densified at construction to avoid repeated materialisation.
pub struct BmsMarginalJacobian {
    /// Dense marginal design: n × p_m.
    pub marginal_dense: Arc<Array2<f64>>,
    /// Dense logslope design: n × p_s.
    pub logslope_dense: Arc<Array2<f64>>,
    pub offset_m: Array1<f64>,
    pub offset_s: Array1<f64>,
    /// Number of marginal columns (= size of β_m slice in the full β vector).
    pub p_marginal: usize,
}

impl BmsMarginalJacobian {
    pub fn new(
        marginal_dense: Arc<Array2<f64>>,
        logslope_dense: Arc<Array2<f64>>,
        offset_m: Array1<f64>,
        offset_s: Array1<f64>,
        p_marginal: usize,
    ) -> Self {
        Self {
            marginal_dense,
            logslope_dense,
            offset_m,
            offset_s,
            p_marginal,
        }
    }
}

impl BlockEffectiveJacobian for BmsMarginalJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let beta = state.beta;
        let s = state.probit_frailty_scale;
        let p_m = self.p_marginal;
        let p_s_block = self.logslope_dense.ncols();
        let beta_s_raw = if beta.len() > p_m {
            &beta[p_m..]
        } else {
            &[][..]
        };
        let p_s_use = p_s_block.min(beta_s_raw.len());
        let beta_s = &beta_s_raw[..p_s_use];
        let n = self.marginal_dense.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);
        let p_block = self.marginal_dense.ncols();

        // ∂η_i/∂β_m = c_i · M[i,:], with c_i = sqrt(1 + (s·g_i)²) and
        //   g_i = G[i, :p_s_use] · β_s + offset_s[i].
        //
        // This block owns the logslope design and offset, so g_i — and hence
        // c_i — is fully self-computable at the current β for every row,
        // including β = 0 (where g_i = offset_s[i], which carries the fitted
        // logslope baseline and is generically nonzero).  There is no external
        // scalar this block cannot reconstruct, so the Jacobian is evaluated
        // directly from owned data with no caller-supplied contract.
        let mut out = Array2::<f64>::zeros((rows.end - rows.start, p_block));
        for i in rows.clone() {
            let g_i = self.offset_s[i]
                + self
                    .logslope_dense
                    .row(i)
                    .slice(ndarray::s![..p_s_use])
                    .dot(&ArrayView1::from(beta_s));
            let sg = s * g_i;
            let c_i = (1.0 + sg * sg).sqrt();
            // J[i,:] = c_i · M[i,:]
            let m_row = self.marginal_dense.row(i);
            out.row_mut(i - rows.start).assign(&m_row.mapv(|x| c_i * x));
        }
        Ok(out)
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

/// β-dependent Jacobian for the BMS logslope block.
///
/// ∂η_i/∂β_s = (q_i · s²·g_i / c_i + s·z_i) · G[i,:]
/// where q_i = M[i,:] · β_m + offset_m[i],
///       g_i = G[i,:] · β_s + offset_s[i],
///       c_i = sqrt(1 + (s·g_i)²),
///       s   = state.probit_frailty_scale.
///
/// `probit_frailty_scale` is read from the evaluation state at call time.
///
/// Designs are pre-densified at construction to avoid repeated materialisation.
pub struct BmsLogslopeJacobian {
    /// Dense marginal design: n × p_m.
    pub marginal_dense: Arc<Array2<f64>>,
    /// Dense logslope design: n × p_s.
    pub logslope_dense: Arc<Array2<f64>>,
    pub offset_m: Array1<f64>,
    pub offset_s: Array1<f64>,
    pub z: Arc<Array1<f64>>,
    /// Number of marginal columns (= start of β_s in the full β vector).
    pub p_marginal: usize,
}

impl BmsLogslopeJacobian {
    pub fn new(
        marginal_dense: Arc<Array2<f64>>,
        logslope_dense: Arc<Array2<f64>>,
        offset_m: Array1<f64>,
        offset_s: Array1<f64>,
        z: Arc<Array1<f64>>,
        p_marginal: usize,
    ) -> Self {
        Self {
            marginal_dense,
            logslope_dense,
            offset_m,
            offset_s,
            z,
            p_marginal,
        }
    }
}

impl BlockEffectiveJacobian for BmsLogslopeJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let beta = state.beta;
        let s = state.probit_frailty_scale;
        let p_m = self.p_marginal;
        let p_m_use = p_m.min(beta.len());
        let beta_m = &beta[..p_m_use];
        let beta_s_raw = if beta.len() > p_m {
            &beta[p_m..]
        } else {
            &[][..]
        };
        let p_s_block = self.logslope_dense.ncols();
        let p_s_use = p_s_block.min(beta_s_raw.len());
        let beta_s = &beta_s_raw[..p_s_use];
        let n = self.logslope_dense.nrows();
        let rows = rows.start.min(n)..rows.end.min(n);

        // ∂η_i/∂β_s = (q_i · s²·g_i / c_i + s·z_i) · G[i,:] where
        //   q_i = M[i,:] · β_m + offset_m[i],
        //   g_i = G[i,:] · β_s + offset_s[i],
        //   c_i = sqrt(1 + (s·g_i)²),
        //   z_i = self.z[i].
        //
        // This block owns the marginal design, logslope design, both offsets,
        // and z, so q_i, g_i, c_i, z_i are all closed-form functions of the
        // current β and owned data — for every row at every β, including β = 0
        // (where g_i = offset_s[i] carries the nonzero fitted logslope
        // baseline).  The Jacobian is therefore evaluated directly from owned
        // data with no caller-supplied scalar contract.
        let mut out = Array2::<f64>::zeros((rows.end - rows.start, p_s_block));
        for i in rows.clone() {
            let q_i = self.offset_m[i]
                + self
                    .marginal_dense
                    .row(i)
                    .slice(ndarray::s![..p_m_use])
                    .dot(&ArrayView1::from(beta_m));
            let g_i = self.offset_s[i]
                + self
                    .logslope_dense
                    .row(i)
                    .slice(ndarray::s![..p_s_use])
                    .dot(&ArrayView1::from(beta_s));
            let sg = s * g_i;
            let c_i = (1.0 + sg * sg).sqrt();
            let z_i = self.z[i];
            // per-row scalar factor: q_i · s²·g_i / c_i + s·z_i
            let factor = q_i * s * s * g_i / c_i + s * z_i;
            // J[i,:] = factor · G[i,:]
            let g_row = self.logslope_dense.row(i);
            out.row_mut(i - rows.start)
                .assign(&g_row.mapv(|x| factor * x));
        }
        Ok(out)
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

/// Horizontally stack the absorbed influence columns `Z̃_infl` onto the raw
/// marginal design `M`, yielding the widened additive marginal-index design
/// `[M | Z̃_infl]` (#461). When `influence_columns` is `None` the original
/// dense design is returned unchanged. The influence columns shift the marginal
/// index `α(x)` additively, so the de-nested probit kernel — which reads the
/// marginal index from `block_states[0].eta` and reconstructs `∂q/∂β_m` from
/// `self.marginal_design` (a matched (design, β) pair) — picks them up with no
/// kernel-site change; the widened `p_marginal` keeps every per-row Jacobian /
/// gradient / Hessian projection consistent.
pub(crate) fn widen_marginal_dense_with_influence(
    marginal_dense: &Arc<Array2<f64>>,
    influence_columns: Option<&Array2<f64>>,
) -> Result<Arc<Array2<f64>>, String> {
    let Some(z_infl) = influence_columns else {
        return Ok(Arc::clone(marginal_dense));
    };
    let n = marginal_dense.nrows();
    if z_infl.nrows() != n {
        return Err(format!(
            "influence block: residualised columns have {} rows, marginal design has {n}",
            z_infl.nrows()
        ));
    }
    let p_m = marginal_dense.ncols();
    let p1 = z_infl.ncols();
    let mut widened = Array2::<f64>::zeros((n, p_m + p1));
    widened
        .slice_mut(s![.., ..p_m])
        .assign(marginal_dense.as_ref());
    widened.slice_mut(s![.., p_m..]).assign(z_infl);
    Ok(Arc::new(widened))
}

/// Tolerance (relative to the dominant retained eigenvalue) below which a
/// reduced-basis direction of the W-orthogonalised effective logslope Gram is
/// treated as a confounded null direction and dropped. Directions whose
/// effective weighted image is (near-)explained by the marginal span collapse to
/// ~0 eigenvalue in `Gtt` (see [`build_reduced_logslope_reparam`]); this keeps
/// the cut well above floating-point noise but well below any genuine surviving
/// logslope curvature.
pub(crate) const LOGSLOPE_REDUCED_BASIS_RELATIVE_TOL: f64 = 1.0e-6;

/// An exact reduced-basis reparameterization of the BMS logslope design through
/// the family's OWN internal `logslope_design` geometry, expressed as a single
/// linear map `T` (`p_logslope × r`, `r ≤ p_logslope`).
///
/// # Why a reduced basis (not a dense design swap)
///
/// The structural confound is that the score-weighted logslope channel
/// `diag(factor)·G·β_s` overlaps the effective marginal channel `diag(c)·M·β_m`
/// in the PIRLS row metric `W`, leaving the joint penalised Hessian rank-soft
/// along the shared direction. The shared-solver primitive
/// [`OrthogonalReparam`](crate::solver::orthogonal_reparam::OrthogonalReparam)
/// forms `C̃ = C − M·B`, exactly W-orthogonal to `span(M)` — but `C̃` is a dense
/// design the BMS family's row kernel does NOT consume: the family reads
/// `η_logslope = G·β_s` from its own `logslope_design` and reconstructs the
/// per-row Jacobian `factor_i · G_i` from that same matrix. A block-level design
/// swap is therefore ignored by the family, and feeding a rank-deficient `C̃` at
/// full width desynchronises the inner identifiable-subspace reduction from the
/// stored design width.
///
/// This builds instead a TRUE reparameterization the family consumes: a
/// full-rank reduced logslope design `G_reduced = G·T` (width `r`) plus the
/// penalty projection `S_reduced = Tᵀ S T`. The map is constructed so that the
/// directions of raw logslope coefficient space whose effective weighted image
/// is W-explained by the marginal span are removed (they carry ~zero curvature
/// in the W-orthogonalised effective Gram), and the surviving `r` directions are
/// full-rank.
///
/// # The math
///
/// At the rigid pilot, the effective Jacobians are
///
/// ```text
///     M_eff = diag(c) · M        (n × p_m),   c_i = sqrt(1 + (s·g_i)²)
///     G_eff = diag(f) · G        (n × p_g),   f_i = q_i·s²·g_i/c_i + s·z_i
/// ```
///
/// In the row metric `W` the component of the effective logslope design that is
/// W-orthogonal to `span(M_eff)` has the raw-coordinate Gram
///
/// ```text
///     Gtt = G_effᵀ W G_eff − (G_effᵀ W M_eff)(M_effᵀ W M_eff + εI)⁻¹(M_effᵀ W G_eff)
/// ```
///
/// (a `p_g × p_g` PSD matrix in the raw logslope coefficient coordinates). Its
/// range = the logslope directions that survive the confound removal; its null
/// space = the confounded directions absorbed by the marginal span. The reduced
/// transform `T` is the orthonormal eigenbasis of `Gtt` for eigenvalues above a
/// relative tolerance; `r = rank(Gtt)`. The new design `G_reduced = G·T`, the
/// reparameterized penalty `S_reduced = Tᵀ S T`, and the round-trip
/// `β_logslope = T·β'` make the family's geometry consistent at width `r` and
/// recover the original-basis logslope coefficients for prediction/reporting.
#[derive(Debug, Clone)]
pub(super) struct ReducedLogslopeReparam {
    /// Reduced transform `T` (`p_logslope × r`). `G_reduced = G·T`,
    /// `β_logslope = T·β'`, `S_reduced = Tᵀ S T`.
    transform: Array2<f64>,
}

impl ReducedLogslopeReparam {
    /// Original (full) logslope width `p_logslope`.
    #[inline]
    pub(super) fn original_cols(&self) -> usize {
        self.transform.nrows()
    }

    /// Reduced width `r`.
    #[inline]
    pub(super) fn reduced_cols(&self) -> usize {
        self.transform.ncols()
    }

    /// Map a reduced-basis logslope coefficient `β'` (length `r`) back to the
    /// original logslope basis `β_logslope = T·β'` (length `p_logslope`), so
    /// prediction/reporting are unchanged-in-meaning.
    pub(super) fn recover_original_logslope_beta(
        &self,
        beta_reduced: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if beta_reduced.len() != self.reduced_cols() {
            return Err(format!(
                "reduced logslope reparam: β' length ({}) != reduced width ({})",
                beta_reduced.len(),
                self.reduced_cols()
            ));
        }
        Ok(self.transform.dot(beta_reduced))
    }
}

/// Build the reduced-basis logslope reparameterization (see
/// [`ReducedLogslopeReparam`]) from the rigid-pilot EFFECTIVE Jacobian geometry,
/// in the PIRLS row metric `W`. Extracts the dense pilot designs and delegates
/// the geometry to [`reduced_logslope_transform_effective`]. Returns `Ok(None)`
/// when there is no logslope/marginal span, no effective-confounded direction to
/// remove (`r == p_g`), or the whole effective logslope image is in the effective
/// marginal span (`r == 0`); in those cases the caller keeps the raw design.
fn build_reduced_logslope_reparam(
    marginal_design: &TermCollectionDesign,
    logslope_design: &TermCollectionDesign,
    z: &Array1<f64>,
    row_metric: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    marginal_baseline: f64,
    logslope_baseline: f64,
    probit_scale: f64,
) -> Result<Option<ReducedLogslopeReparam>, String> {
    let marginal = marginal_design
        .design
        .try_to_dense_arc("build_reduced_logslope_reparam::marginal")?;
    let logslope = logslope_design
        .design
        .try_to_dense_arc("build_reduced_logslope_reparam::logslope")?;
    let n = marginal.nrows();
    if logslope.nrows() != n
        || z.len() != n
        || row_metric.len() != n
        || marginal_offset.len() != n
        || logslope_offset.len() != n
    {
        return Err(format!(
            "reduced logslope reparam row mismatch: marginal={}, logslope={}, z={}, row_metric={}, marginal_offset={}, logslope_offset={}",
            marginal.nrows(),
            logslope.nrows(),
            z.len(),
            row_metric.len(),
            marginal_offset.len(),
            logslope_offset.len(),
        ));
    }
    let p_m = marginal.ncols();
    let p_g = logslope.ncols();
    if p_m == 0 || p_g == 0 {
        return Ok(None);
    }
    if !marginal_baseline.is_finite()
        || !logslope_baseline.is_finite()
        || !probit_scale.is_finite()
        || probit_scale <= 0.0
        || z.iter().any(|v| !v.is_finite())
        || row_metric.iter().any(|v| !v.is_finite() || *v < 0.0)
        || marginal_offset.iter().any(|v| !v.is_finite())
        || logslope_offset.iter().any(|v| !v.is_finite())
    {
        return Err(
            "reduced logslope reparam requires finite pilot geometry and finite non-negative row metric"
                .to_string(),
        );
    }

    // The joint Hessian the inner solve factorises is built from the EFFECTIVE
    // BMS Jacobians, not the raw design. At the rigid pilot,
    //   ∂η_i/∂β_m = c_i · M_i,   c_i = sqrt(1 + (s·g_i)²)
    //   ∂η_i/∂β_s = f_i · G_i,   f_i = q_i·s²·g_i/c_i + s·z_i
    // so a raw logslope direction `v` is rank-soft in the joint Hessian iff its
    // EFFECTIVE image `diag(f)·G·v` is W-explained by `span(diag(c)·M)` — NOT iff
    // raw `G·v` is W-explained by `span(M)`. Auditing the raw design removes the
    // wrong directions; the reduced basis is built from the effective Schur Gram.
    // The pure-array geometry lives in `reduced_logslope_transform_effective` so
    // it can be unit-tested directly against the raw-vs-effective counterexample.
    match reduced_logslope_transform_effective(
        marginal.view(),
        logslope.view(),
        z,
        row_metric,
        marginal_offset,
        logslope_offset,
        marginal_baseline,
        logslope_baseline,
        probit_scale,
    )? {
        Some(transform) => Ok(Some(ReducedLogslopeReparam { transform })),
        None => Ok(None),
    }
}

/// Build the reduced logslope basis `T` (p_g × r) from the EFFECTIVE BMS pilot
/// geometry, in the PIRLS row metric `W`. `T`'s columns span the raw logslope
/// coefficient directions whose effective image `diag(f)·G·v` is NOT W-explained
/// by `span(diag(c)·M)` — i.e. the directions the joint Hessian retains real
/// curvature along. Returns `Ok(None)` when there is nothing to reduce
/// (`r == p_g`) or when the entire effective logslope image collapses into the
/// effective marginal span (`r == 0`); in both cases the caller keeps the raw
/// design (a zero-width reduction would silently delete the score-effect surface,
/// which is the estimand — the REML penalty regularises any residual softness).
///
/// At the rigid pilot the effective Jacobians are
///     M_eff = diag(c) · M,   c_i = sqrt(1 + (s·g_i)²)
///     G_eff = diag(f) · G,   f_i = q_i·s²·g_i/c_i + s·z_i
/// and the raw-coordinate Gram of the logslope component W-orthogonal to
/// `span(M_eff)` is the Schur complement
///     Gtt = G_effᵀ W G_eff − (G_effᵀ W M_eff)(M_effᵀ W M_eff + εI)⁻¹(M_effᵀ W G_eff).
/// `T` is the orthonormal eigenbasis of `Gtt` for eigenvalues above a tolerance
/// relative to the effective logslope energy scale.
pub(crate) fn reduced_logslope_transform_effective(
    marginal: ArrayView2<'_, f64>,
    logslope: ArrayView2<'_, f64>,
    z: &Array1<f64>,
    row_metric: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    marginal_baseline: f64,
    logslope_baseline: f64,
    probit_scale: f64,
) -> Result<Option<Array2<f64>>, String> {
    let n = marginal.nrows();
    let p_m = marginal.ncols();
    let p_g = logslope.ncols();
    if p_m == 0 || p_g == 0 {
        return Ok(None);
    }

    // Effective pilot Jacobians M_eff = diag(c)·M and G_eff = diag(f)·G.
    let mut m_eff = Array2::<f64>::zeros((n, p_m));
    let mut g_eff = Array2::<f64>::zeros((n, p_g));
    for i in 0..n {
        let q_i = marginal_offset[i] + marginal_baseline;
        let g_i = logslope_offset[i] + logslope_baseline;
        let sg = probit_scale * g_i;
        let c_i = (1.0 + sg * sg).sqrt();
        let f_i = q_i * probit_scale * probit_scale * g_i / c_i + probit_scale * z[i];
        for j in 0..p_m {
            m_eff[[i, j]] = c_i * marginal[[i, j]];
        }
        for j in 0..p_g {
            g_eff[[i, j]] = f_i * logslope[[i, j]];
        }
    }

    // C = G_effᵀ W G_eff (raw-coordinate effective logslope Gram); its diagonal
    // sets the energy scale for the relative kept-direction tolerance.
    let c_gram = fast_xt_diag_x(&g_eff, row_metric);
    let energy_scale = (0..p_g).map(|i| c_gram[[i, i]]).fold(0.0_f64, f64::max);
    if !energy_scale.is_finite() || energy_scale <= 0.0 {
        return Ok(None);
    }

    // A = M_effᵀ W M_eff + εI (ridge relative to the marginal effective energy so
    // the Schur solve is well-posed even when the marginal pilot Gram is
    // rank-soft; the ridge only under-removes, i.e. is conservative).
    let mut a_gram = fast_xt_diag_x(&m_eff, row_metric);
    let a_scale = (0..p_m).map(|i| a_gram[[i, i]]).fold(0.0_f64, f64::max);
    let a_ridge = (a_scale * LOGSLOPE_REDUCED_BASIS_RELATIVE_TOL).max(f64::EPSILON);
    for i in 0..p_m {
        a_gram[[i, i]] += a_ridge;
    }

    // B = M_effᵀ W G_eff (p_m × p_g);  Gtt = C − Bᵀ A⁻¹ B (p_g × p_g, PSD).
    let b_cross = crate::faer_ndarray::fast_xt_diag_y(&m_eff, row_metric, &g_eff);
    let a_view = crate::faer_ndarray::FaerArrayView::new(&a_gram);
    let a_factor =
        crate::faer_ndarray::factorize_symmetricwith_fallback(a_view.as_ref(), Side::Lower)
            .map_err(|e| {
                format!(
                    "reduced logslope reparam: effective marginal Gram factorization failed: {e}"
                )
            })?;
    let b_view = crate::faer_ndarray::FaerArrayView::new(&b_cross);
    let solved = a_factor.solve(b_view.as_ref()); // A⁻¹ B  (p_m × p_g)
    let a_inv_b = Array2::from_shape_fn((p_m, p_g), |(i, j)| solved[(i, j)]);
    let schur = fast_atb(&b_cross, &a_inv_b); // Bᵀ A⁻¹ B  (p_g × p_g)
    let mut stt = &c_gram - &schur;
    stt = (&stt + &stt.t()) * 0.5;
    if stt.iter().any(|v| !v.is_finite()) {
        return Err(
            "reduced logslope reparam: effective Schur Gram produced non-finite entries"
                .to_string(),
        );
    }

    let (evals, evecs) = stt
        .eigh(Side::Lower)
        .map_err(|e| format!("reduced logslope reparam: eigendecomposition failed: {e:?}"))?;
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
    // r == p_g: no effective-confounded direction to remove. r == 0: the whole
    // effective logslope image is in the effective marginal span. In both cases
    // install no transform (see fn-level doc) and keep the raw design.
    if r == p_g || r == 0 {
        return Ok(None);
    }
    let mut transform = Array2::<f64>::zeros((p_g, r));
    for (out_col, &src) in kept.iter().enumerate() {
        transform.column_mut(out_col).assign(&evecs.column(src));
    }
    if transform.iter().any(|v| !v.is_finite()) {
        return Err(
            "reduced logslope reparam: reduced transform produced non-finite entries".to_string(),
        );
    }
    Ok(Some(transform))
}

/// Apply a [`ReducedLogslopeReparam`] to a logslope `TermCollectionDesign`,
/// producing a new design at the reduced width `r`: the design becomes
/// `G_reduced = G·T`, and every blockwise penalty `S` is reparameterized to
/// `S_reduced = Tᵀ S T` over the full reduced column range `0..r`. The reduced
/// penalty's null space is recomputed from its numerical rank so the REML
/// log-determinant accounting stays consistent at the reduced width.
fn reparameterize_logslope_design_reduced(
    logslope_design: &TermCollectionDesign,
    reparam: &ReducedLogslopeReparam,
) -> Result<TermCollectionDesign, String> {
    let g = logslope_design
        .design
        .try_to_dense_arc("reparameterize_logslope_design_reduced::logslope")?;
    let p_g = g.ncols();
    if p_g != reparam.original_cols() {
        return Err(format!(
            "reduced logslope reparam width mismatch: design has {p_g} cols, transform expects {}",
            reparam.original_cols()
        ));
    }
    let t = &reparam.transform;
    let r = reparam.reduced_cols();
    // G_reduced = G·T   (n × r).
    let g_reduced = fast_ab(&g, t);

    // Reparameterize each penalty: embed its local block at full width p_g, then
    // form S_reduced = Tᵀ S T (r × r) over the whole reduced column range.
    let mut new_penalties: Vec<crate::terms::smooth::BlockwisePenalty> =
        Vec::with_capacity(logslope_design.penalties.len());
    let mut new_nullspace_dims: Vec<usize> = Vec::with_capacity(logslope_design.penalties.len());
    for bp in &logslope_design.penalties {
        let mut full = Array2::<f64>::zeros((p_g, p_g));
        full.slice_mut(s![bp.col_range.clone(), bp.col_range.clone()])
            .assign(&bp.local);
        // S_reduced = Tᵀ (S) T.
        let st = fast_ab(&full, t); // p_g × r
        let mut s_reduced = fast_atb(t, &st); // r × r
        s_reduced = (&s_reduced + &s_reduced.t()) * 0.5;
        // Null-space dimension of the reduced penalty = r − rank(S_reduced).
        let (evals, _) = s_reduced
            .eigh(Side::Lower)
            .map_err(|e| format!("reduced logslope penalty eigendecomposition failed: {e:?}"))?;
        let max_eval = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let pen_tol = (max_eval * 1.0e-12).max(f64::EPSILON);
        let rank = evals.iter().filter(|&&v| v.abs() > pen_tol).count();
        let nullspace_dim = r.saturating_sub(rank);
        new_penalties.push(crate::terms::smooth::BlockwisePenalty::new(0..r, s_reduced));
        new_nullspace_dims.push(nullspace_dim);
    }

    let new_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(g_reduced));
    // The reduced logslope block is a single dense smooth-like surface over the
    // reparameterized coordinates; it carries no parametric/random-effect/
    // intercept structure of its own (those live in the marginal block), so the
    // structural ranges collapse to empty and the smooth metadata is cleared.
    // The penalties + nullspace_dims above are what the joint REML consumes.
    Ok(TermCollectionDesign {
        design: new_design,
        penalties: new_penalties,
        nullspace_dims: new_nullspace_dims,
        penaltyinfo: Vec::new(),
        dropped_penaltyinfo: Vec::new(),
        coefficient_lower_bounds: None,
        linear_constraints: None,
        intercept_range: 0..0,
        linear_ranges: Vec::new(),
        random_effect_ranges: Vec::new(),
        random_effect_levels: Vec::new(),
        smooth: crate::terms::smooth::SmoothDesign {
            term_designs: Vec::new(),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            penaltyinfo: Vec::new(),
            dropped_penaltyinfo: Vec::new(),
            terms: Vec::new(),
            coefficient_lower_bounds: None,
            linear_constraints: None,
        },
    })
}

/// Re-embed the term-collection marginal penalties at the (possibly widened)
/// block dimension `p_m [+ p₁]`, then append the #461 fixed-ridge absorber:
///
///  (#461, only with influence columns) the fixed-ridge absorber identity on
///  the influence columns `p_m..p_m+p₁`.
///
/// The two former gam#754 pinned ridges — the marginal nullspace-shrinkage ridge
/// and the marginal↔logslope overlap ridge — are DELETED: robustness is now
/// unconditional, so the full-identifiable-span Jeffreys term (`Z_J = I`, see
/// `jeffreys_subspace_from_penalty`) supplies automatic O(n)-scaled curvature on
/// every under-identified direction (subsuming the nullspace ridge), and the
/// exact orthogonal reparameterization of the logslope design (now unconditional,
/// see `build_reduced_logslope_reparam`) resolves the marginal↔logslope confound
/// by construction (subsuming the overlap ridge).
///
/// The genuine marginal smooth penalties keep their `col_range` (marginal
/// columns stay in `0..p_m`). Returns `(penalties, nullspace_dims,
/// initial_log_lambdas)` to install on the marginal block. Fixed ridges remain
/// in this physical penalty layout and are removed from every REML/outer
/// coordinate vector by [`PenaltyMatrix::Fixed`].
pub(crate) fn marginal_penalties_with_influence_ridge(
    design: &TermCollectionDesign,
    rho_marginal: &Array1<f64>,
    influence_columns: Option<&Array2<f64>>,
    influence_ridge_log_lambda: f64,
) -> Result<(Vec<PenaltyMatrix>, Vec<usize>, Array1<f64>), String> {
    let p_m = design.design.ncols();
    let p1 = influence_columns.map(|z| z.ncols()).unwrap_or(0);
    let total_dim = p_m + p1;
    // Re-embed each marginal penalty at the (widened) total dimension (col_range
    // unchanged: marginal columns remain 0..p_m).
    let mut penalties: Vec<PenaltyMatrix> = design
        .penalties
        .iter()
        .map(|bp| bp.to_penalty_matrix(total_dim))
        .collect();
    let mut nullspace_dims = design.nullspace_dims.clone();
    let mut log_lambdas = rho_marginal.to_vec();

    // (#461) fixed-ridge absorber: identity on the influence columns only.
    // Full rank (nullspace 0); its log λ is pinned out of REML by a degenerate
    // ρ box.
    if p1 > 0 {
        penalties.push(
            PenaltyMatrix::Blockwise {
                local: Array2::<f64>::eye(p1),
                col_range: p_m..total_dim,
                total_dim,
            }
            .with_fixed_log_lambda(influence_ridge_log_lambda),
        );
        nullspace_dims.push(0);
        log_lambdas.push(influence_ridge_log_lambda);
    }

    Ok((penalties, nullspace_dims, Array1::from_vec(log_lambdas)))
}

/// Widen an optional β warm-start hint to the influence-widened marginal
/// dimension, zero-filling the absorber coefficients `γ` (#461).
pub(crate) fn widen_marginal_beta_hint(
    beta_hint: Option<Array1<f64>>,
    p_marginal_widened: usize,
) -> Option<Array1<f64>> {
    beta_hint.map(|hint| {
        if hint.len() == p_marginal_widened {
            hint
        } else {
            let mut widened = Array1::<f64>::zeros(p_marginal_widened);
            let copy = hint.len().min(p_marginal_widened);
            widened
                .slice_mut(s![..copy])
                .assign(&hint.slice(s![..copy]));
            widened
        }
    })
}

/// Sup-norm of the fitted marginal linear predictor `η = X·β` restricted to a
/// subset of the marginal design's columns. The mask selects which columns of
/// `design.design` (length `design.design.ncols()`) contribute; coefficients
/// beyond `ncols` (the fixed-ridge influence absorber) never enter the marginal
/// predictor and are excluded by construction. Returns `0.0` for an empty
/// design. This is the decisive separation quantity: the probit Fisher weight
/// collapses with `|η|`, not with `‖β‖` (an ill-conditioned non-orthonormal
/// Duchon/thin-plate basis carries large cancelling coefficients on a smooth,
/// bounded surface).
fn marginal_fitted_eta_sup_norm(design: &TermCollectionDesign, masked_beta: &Array1<f64>) -> f64 {
    let x = &design.design;
    let n = x.nrows();
    if n == 0 || x.ncols() == 0 {
        return 0.0;
    }
    let mut sup = 0.0_f64;
    for row in 0..n {
        let eta = x.dot_row_view(row, masked_beta.view());
        if eta.is_finite() {
            sup = sup.max(eta.abs());
        }
    }
    sup
}

/// Build a copy of the marginal block β truncated to `design.design.ncols()`
/// (drops any fixed-ridge absorber tail) so it can drive `X·β`.
fn marginal_design_beta(
    design: &TermCollectionDesign,
    block_beta: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let ncols = design.design.ncols();
    let mut masked = Array1::<f64>::zeros(ncols);
    let copy = ncols.min(block_beta.len());
    masked
        .slice_mut(s![..copy])
        .assign(&block_beta.slice(s![..copy]));
    masked
}

/// Zero every entry of `beta` outside the parametric (penalty-nullspace)
/// marginal columns — the intercept and the single-penalty linear terms. These
/// are the directions an unpenalized fit can genuinely separate along (no
/// smoothness penalty bounds them), so their fitted contribution is tested on
/// the same η scale.
fn mask_parametric_columns(
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    full: &Array1<f64>,
) -> Array1<f64> {
    let ncols = design.design.ncols();
    let mut masked = Array1::<f64>::zeros(ncols);
    if design.intercept_range.len() == 1 {
        let idx = design.intercept_range.start;
        if idx < ncols {
            masked[idx] = full[idx];
        }
    }
    for (linear, (_, range)) in spec.linear_terms.iter().zip(design.linear_ranges.iter()) {
        if linear.double_penalty {
            continue;
        }
        for col in range.clone() {
            if col < ncols {
                masked[col] = full[col];
            }
        }
    }
    masked
}

/// Decide whether the converged marginal fit has genuinely separated, using the
/// FITTED predictor sup-norm `|η|∞` (not raw `|β|∞`). Two arms share the
/// numerical-degeneracy threshold [`BMS_PROBIT_SEPARATION_ETA_INF`]:
///   - parametric arm: the penalty-nullspace columns' fitted contribution
///     (an unpenalized direction can run to infinity);
///   - full arm: the whole marginal surface's fitted predictor.
/// The raw `|β|∞` (and its term label) is reported only as diagnostic context;
/// it never gates the abort. When `|η|∞` is below threshold the converged
/// penalized fit is numerically trustworthy and this returns `None` — no error,
/// even if individual coefficients are large.
pub(crate) fn bernoulli_marginal_slope_runaway_error_from_beta(
    block_beta: ArrayView1<'_, f64>,
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    inner_converged: bool,
    eval_label: &str,
) -> Option<String> {
    let full_beta = marginal_design_beta(design, block_beta);
    let parametric_beta = mask_parametric_columns(design, spec, &full_beta);

    let eta_parametric = marginal_fitted_eta_sup_norm(design, &parametric_beta);
    let eta_full = marginal_fitted_eta_sup_norm(design, &full_beta);

    let (eta_inf, explanation) = if eta_parametric >= BMS_PROBIT_SEPARATION_ETA_INF {
        (
            eta_parametric,
            "an unpenalized parametric marginal direction has no stable finite probit optimum and its fitted predictor has run to the probit underflow scale",
        )
    } else if eta_full >= BMS_PROBIT_SEPARATION_ETA_INF {
        (
            eta_full,
            "a marginal direction is trading off against the logslope surface; this is the under-constrained marginal/logslope coupling that appears when the score is correlated with the shared surface covariates",
        )
    } else {
        // |η|∞ is bounded: even if raw coefficients are large (ill-conditioned
        // non-orthonormal basis with cancellation), the converged penalized
        // probit fit is numerically trustworthy. Do NOT abort.
        return None;
    };

    let inner_status = if inner_converged {
        "the inner solve reached a KKT certificate at this separation-scale predictor"
    } else {
        "the inner solve failed while already carrying a separation-scale predictor"
    };
    // Raw |β|∞ context (decisive quantity is |η|∞ above).
    let beta_abs = full_beta
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));

    Some(format!(
        "bernoulli marginal-slope probit marginal/logslope runaway detected in block \
         'marginal_surface' during {eval_label}: the fitted marginal predictor has \
         |η|∞={eta_inf:.3e} (numerical-degeneracy threshold \
         {BMS_PROBIT_SEPARATION_ETA_INF:.1}; raw |β|∞={beta_abs:.3e} is reported for \
         context only and does not gate this diagnostic). The joint design is \
         identifiable; {explanation}. {inner_status}. The robust Jeffreys curvature \
         path is already installed for this fit, so this diagnostic means the current \
         coupled surface still drives the linear predictor to the probit underflow \
         scale rather than a request for an external bias-reduction prior. Reduce or \
         reparameterize the coupled marginal/logslope surface, or use a \
         lower-dimensional logslope interaction. This is not a \
         Matérn/Duchon polynomial-nullspace or cross-block gauge-priority \
         failure."
    ))
}

pub(crate) fn bernoulli_marginal_slope_runaway_error(
    warm_start: &CustomFamilyWarmStart,
    design: &TermCollectionDesign,
    spec: &TermCollectionSpec,
    inner_converged: bool,
    eval_label: &str,
) -> Option<String> {
    let block_beta = warm_start.block_beta_view(0)?;
    bernoulli_marginal_slope_runaway_error_from_beta(
        block_beta,
        design,
        spec,
        inner_converged,
        eval_label,
    )
}

#[cfg(test)]
mod runaway_tests {
    use super::*;
    use crate::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback, fast_xt_diag_y};
    use crate::smooth::{LinearCoefficientGeometry, LinearTermSpec};

    // The marginal↔logslope overlap penalty is no longer installed as a pinned
    // ridge (subsumed by the now-unconditional exact logslope orthogonalisation in
    // `build_reduced_logslope_reparam`). The geometry helper is retained here under
    // the test module because the basis-independence/weight-orthogonality unit tests
    // below exercise it directly as the canonical overlap-direction reference.
    pub(crate) fn marginal_logslope_overlap_penalty(
        marginal_design: &DesignMatrix,
        logslope_design: &DesignMatrix,
        z: &Array1<f64>,
        row_metric: &Array1<f64>,
        marginal_offset: &Array1<f64>,
        logslope_offset: &Array1<f64>,
        marginal_baseline: f64,
        logslope_baseline: f64,
        probit_scale: f64,
    ) -> Result<Option<Array2<f64>>, String> {
        let marginal =
            marginal_design.try_to_dense_arc("marginal_logslope_overlap_penalty::marginal")?;
        let logslope =
            logslope_design.try_to_dense_arc("marginal_logslope_overlap_penalty::logslope")?;
        let n = marginal.nrows();
        if logslope.nrows() != n
            || z.len() != n
            || row_metric.len() != n
            || marginal_offset.len() != n
            || logslope_offset.len() != n
        {
            return Err(format!(
                "marginal/logslope overlap penalty row mismatch: marginal={}, logslope={}, z={}, row_metric={}, marginal_offset={}, logslope_offset={}",
                marginal.nrows(),
                logslope.nrows(),
                z.len(),
                row_metric.len(),
                marginal_offset.len(),
                logslope_offset.len(),
            ));
        }
        let p_m = marginal.ncols();
        let p_g = logslope.ncols();
        if p_m == 0 || p_g == 0 {
            return Ok(None);
        }
        if !marginal_baseline.is_finite()
            || !logslope_baseline.is_finite()
            || !probit_scale.is_finite()
            || probit_scale <= 0.0
            || z.iter().any(|v| !v.is_finite())
            || row_metric.iter().any(|v| !v.is_finite() || *v < 0.0)
            || marginal_offset.iter().any(|v| !v.is_finite())
            || logslope_offset.iter().any(|v| !v.is_finite())
        {
            return Err(
                "marginal/logslope overlap penalty requires finite pilot geometry and finite non-negative row metric"
                    .to_string(),
            );
        }

        let mut marginal_effective = Array2::<f64>::zeros((n, p_m));
        let mut effective_logslope = Array2::<f64>::zeros((n, p_g));
        for i in 0..n {
            let q_i = marginal_offset[i] + marginal_baseline;
            let g_i = logslope_offset[i] + logslope_baseline;
            let sg = probit_scale * g_i;
            let c_i = (1.0 + sg * sg).sqrt();
            let logslope_factor =
                q_i * probit_scale * probit_scale * g_i / c_i + probit_scale * z[i];
            for j in 0..p_m {
                marginal_effective[[i, j]] = c_i * marginal[[i, j]];
            }
            for j in 0..p_g {
                effective_logslope[[i, j]] = logslope_factor * logslope[[i, j]];
            }
        }
        if effective_logslope.iter().all(|v| v.abs() <= f64::EPSILON) {
            return Ok(None);
        }

        let mut gram = fast_xt_diag_x(&effective_logslope, row_metric);
        let gram_scale = gram.diag().iter().copied().fold(0.0_f64, f64::max);
        if !gram_scale.is_finite() || gram_scale <= 0.0 {
            return Ok(None);
        }
        let projection_ridge = (gram_scale * 1.0e-10).max(f64::EPSILON);
        for i in 0..p_g {
            gram[[i, i]] += projection_ridge;
        }
        let cross = fast_xt_diag_y(&effective_logslope, row_metric, &marginal_effective);
        let gram_view = FaerArrayView::new(&gram);
        let factor = factorize_symmetricwith_fallback(gram_view.as_ref(), Side::Lower)
            .map_err(|e| format!("marginal/logslope overlap Gram factorization failed: {e}"))?;
        let rhsview = FaerArrayView::new(&cross);
        let coeffs_mat = factor.solve(rhsview.as_ref());
        let coeffs = Array2::from_shape_fn((p_g, p_m), |(i, j)| coeffs_mat[(i, j)]);
        let projected_marginal = fast_ab(&effective_logslope, &coeffs);
        let mut penalty = fast_xt_diag_y(&marginal_effective, row_metric, &projected_marginal);
        penalty = (&penalty + &penalty.t()) * 0.5;
        let max_abs = penalty.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if !max_abs.is_finite() || max_abs <= 1.0e-12 {
            return Ok(None);
        }
        Ok(Some(penalty))
    }

    // The raw-vs-effective counterexample. With q=g=0, s=1: c_i=1 (so M_eff=M)
    // and f_i=z_i (so G_eff=diag(z)·G). Pick M=[1,1,1]ᵀ and a two-column logslope
    // G whose RAW columns are both linearly independent of M (raw orthogonalising
    // [M|G] would keep BOTH — the old code returned None, no reduction), but whose
    // first EFFECTIVE column diag(z)·G[:,0] equals M_eff exactly. The effective
    // audit must therefore drop exactly one direction (r=1), proving it removes
    // the joint-Hessian rank-soft direction the raw audit could not see.
    #[test]
    pub(crate) fn effective_reduction_drops_score_weighted_confound_raw_audit_misses() {
        // G col0 = [1,2,3], col1 = [1,2,9]  (row-major rows: [1,1],[2,2],[3,9]).
        let m = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let g = Array2::<f64>::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 9.0]).unwrap();
        let z = Array1::from_vec(vec![1.0, 0.5, 1.0 / 3.0]);
        let w = Array1::<f64>::ones(3);
        let zero = Array1::<f64>::zeros(3);

        // diag(z)·G[:,0] = [1·1, 0.5·2, (1/3)·3] = [1,1,1] = M_eff (fully aliased);
        // diag(z)·G[:,1] = [1·1, 0.5·2, (1/3)·9] = [1,1,3] (NOT in span([1,1,1])).
        let reparam = reduced_logslope_transform_effective(
            m.view(),
            g.view(),
            &z,
            &w,
            &zero,
            &zero,
            0.0,
            0.0,
            1.0,
        )
        .expect("effective reduction must succeed")
        .expect("effective audit must reduce the score-weighted confound (raw audit would not)");
        assert_eq!(
            reparam.ncols(),
            1,
            "exactly one effective-identifiable logslope direction should survive"
        );

        // The surviving raw direction's EFFECTIVE image diag(z)·G·t must carry the
        // non-constant ([1,1,3]) content — i.e. it is the identifiable direction,
        // not the [1,1,1] confound. Its row variance must be clearly positive.
        let g_eff = {
            let mut e = Array2::<f64>::zeros((3, 2));
            for i in 0..3 {
                for j in 0..2 {
                    e[[i, j]] = z[i] * g[[i, j]];
                }
            }
            e
        };
        let img = g_eff.dot(&reparam.column(0));
        let mean = img.iter().sum::<f64>() / 3.0;
        let var = img.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0;
        assert!(
            var > 1.0e-6,
            "kept direction must be the identifiable (non-constant) effective column, var={var}"
        );
    }

    // The single-column fully-confounded case: G=[1,2,3]ᵀ, z=[1,1/2,1/3] gives
    // G_eff=[1,1,1]=M_eff, so the entire effective logslope image is in the
    // effective marginal span (r==0). The helper returns None (keep the raw
    // design) rather than emitting a zero-width transform that would delete the
    // score-effect surface.
    #[test]
    pub(crate) fn effective_reduction_fully_confounded_single_column_returns_none() {
        let m = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let g = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let z = Array1::from_vec(vec![1.0, 0.5, 1.0 / 3.0]);
        let w = Array1::<f64>::ones(3);
        let zero = Array1::<f64>::zeros(3);
        let reparam = reduced_logslope_transform_effective(
            m.view(),
            g.view(),
            &z,
            &w,
            &zero,
            &zero,
            0.0,
            0.0,
            1.0,
        )
        .expect("effective reduction must succeed");
        assert!(
            reparam.is_none(),
            "fully effective-confounded logslope must keep raw design (None), not a 0-width block"
        );
    }

    // No effective confound: both effective logslope columns stay independent of
    // M_eff, so nothing is reduced (r==p_g ⇒ None) and healthy fits are untouched.
    #[test]
    pub(crate) fn effective_reduction_no_confound_returns_none() {
        let m = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        // diag(z)·col gives non-constant images for both columns under z below.
        let g = Array2::<f64>::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let z = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let w = Array1::<f64>::ones(3);
        let zero = Array1::<f64>::zeros(3);
        let reparam = reduced_logslope_transform_effective(
            m.view(),
            g.view(),
            &z,
            &w,
            &zero,
            &zero,
            0.0,
            0.0,
            1.0,
        )
        .expect("effective reduction must succeed");
        assert!(
            reparam.is_none(),
            "no effective confound ⇒ no reduction (raw design kept unchanged)"
        );
    }

    #[test]
    pub(crate) fn spatial_joint_setup_counts_only_learned_penalties_in_rho() {
        let data = Array2::<f64>::zeros((3, 1));
        let empty_terms = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        };
        let setup = joint_setup(
            data.view(),
            &empty_terms,
            &empty_terms,
            2,
            3,
            &[0.4],
            &SpatialLengthScaleOptimizationOptions::default(),
        );

        assert_eq!(
            setup.rho_dim(),
            6,
            "BMS spatial setup rho must contain only learned marginal/logslope/auxiliary penalties; fixed physical ridges are carried by PenaltyMatrix::Fixed"
        );
    }

    #[test]
    pub(crate) fn overlap_penalty_targets_score_weighted_logslope_span() {
        let marginal = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        ));
        let logslope = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        ));
        let z = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let row_metric = Array1::ones(4);
        let offsets = Array1::zeros(4);

        let penalty = marginal_logslope_overlap_penalty(
            &marginal,
            &logslope,
            &z,
            &row_metric,
            &offsets,
            &offsets,
            0.0,
            0.0,
            1.0,
        )
        .expect("overlap penalty should build")
        .expect("marginal signal lies in the pilot logslope Jacobian span");

        assert_eq!(penalty.dim(), (1, 1));
        assert!((penalty[[0, 0]] - 14.0).abs() < 1.0e-6);
    }

    #[test]
    pub(crate) fn overlap_penalty_skips_weight_orthogonal_channels() {
        let marginal = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_vec((4, 1), vec![-1.0, 1.0, -1.0, 1.0]).unwrap(),
        ));
        let logslope = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
        ));
        let z = Array1::ones(4);
        let row_metric = Array1::ones(4);
        let offsets = Array1::zeros(4);

        let penalty = marginal_logslope_overlap_penalty(
            &marginal,
            &logslope,
            &z,
            &row_metric,
            &offsets,
            &offsets,
            0.0,
            0.0,
            1.0,
        )
        .expect("overlap penalty should build");

        assert!(penalty.is_none());
    }

    // ── Fitted-η separation guard fixtures ───────────────────────────────
    //
    // The runaway guard tests the FITTED marginal predictor sup-norm
    // `|η|∞ = max_i |X[i,:]·β|`, not raw `|β|∞`. These helpers build minimal
    // `TermCollectionDesign` / `TermCollectionSpec` pairs from a dense design so
    // the criterion is exercised deterministically with no data files.

    fn dense_marginal_design(
        x: Array2<f64>,
        intercept_range: std::ops::Range<usize>,
        linear_ranges: Vec<(String, std::ops::Range<usize>)>,
    ) -> TermCollectionDesign {
        TermCollectionDesign {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            penaltyinfo: Vec::new(),
            dropped_penaltyinfo: Vec::new(),
            coefficient_lower_bounds: None,
            linear_constraints: None,
            intercept_range,
            linear_ranges,
            random_effect_ranges: Vec::new(),
            random_effect_levels: Vec::new(),
            smooth: crate::terms::smooth::SmoothDesign {
                term_designs: Vec::new(),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                penaltyinfo: Vec::new(),
                dropped_penaltyinfo: Vec::new(),
                terms: Vec::new(),
                coefficient_lower_bounds: None,
                linear_constraints: None,
            },
        }
    }

    fn linear_term(name: &str, feature_col: usize) -> LinearTermSpec {
        LinearTermSpec {
            name: name.to_string(),
            feature_col,
            feature_cols: vec![feature_col],
            categorical_levels: vec![],
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::default(),
            coefficient_min: None,
            coefficient_max: None,
        }
    }

    fn empty_spec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        }
    }

    /// Regression lock for the false-positive fix: an ill-conditioned
    /// non-orthonormal basis (two identical/collinear columns) with a large
    /// cancelling coefficient `β=[60,-60]` yields `Xβ ≡ 0` — a perfectly
    /// bounded fitted predictor. The guard MUST NOT fire even though raw
    /// `|β|∞=60` is far above the old `40.0` coefficient threshold. This is the
    /// exact pathology that aborted valid biobank fits.
    #[test]
    pub(crate) fn runaway_guard_silent_when_huge_beta_cancels_to_bounded_eta() {
        // Two identical columns ⇒ Xβ = (β0+β1)·col; β=[60,-60] ⇒ Xβ ≡ 0.
        let x = Array2::<f64>::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let design = dense_marginal_design(x, 0..0, Vec::new());
        let beta = Array1::from_vec(vec![60.0, -60.0]);

        let msg = bernoulli_marginal_slope_runaway_error_from_beta(
            beta.view(),
            &design,
            &empty_spec(),
            true,
            "regression-fixture",
        );
        assert!(
            msg.is_none(),
            "huge cancelling β with bounded fitted η must NOT trip the runaway guard; got {msg:?}"
        );
    }

    /// A genuinely separating full marginal surface: a single column with a
    /// large coefficient drives `|η|∞ = 40 ≥ 35`, so the guard fires and names
    /// the marginal/logslope coupling explanation.
    #[test]
    pub(crate) fn runaway_guard_fires_when_fitted_eta_exceeds_threshold() {
        let x = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let design = dense_marginal_design(x, 0..0, Vec::new());
        let beta = Array1::from_vec(vec![40.0]);

        let msg = bernoulli_marginal_slope_runaway_error_from_beta(
            beta.view(),
            &design,
            &empty_spec(),
            true,
            "separation-fixture",
        )
        .expect("fitted |η|∞=40 ≥ 35 must trip the runaway guard");

        assert!(msg.contains("marginal/logslope runaway"));
        assert!(msg.contains("|η|∞"));
        assert!(msg.contains("4.000e1"));
        assert!(msg.contains("score is correlated with the shared surface covariates"));
        assert!(msg.contains("not a Matérn/Duchon polynomial-nullspace"));
        assert!(msg.contains("KKT certificate"));
    }

    /// An unpenalized parametric direction can genuinely separate (no smoothness
    /// penalty bounding it). When its fitted contribution reaches the η scale
    /// the parametric arm fires first and names the parametric explanation.
    #[test]
    pub(crate) fn runaway_guard_names_unpenalized_parametric_direction_via_fitted_eta() {
        let x = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let design = dense_marginal_design(x, 0..0, vec![("sex".to_string(), 0..1)]);
        let mut spec = empty_spec();
        spec.linear_terms.push(linear_term("sex", 0));
        let beta = Array1::from_vec(vec![41.0]);

        let msg = bernoulli_marginal_slope_runaway_error_from_beta(
            beta.view(),
            &design,
            &spec,
            true,
            "parametric-fixture",
        )
        .expect("parametric fitted |η|∞=41 ≥ 35 must trip the runaway guard");

        assert!(msg.contains("unpenalized parametric marginal direction"));
        assert!(msg.contains("|η|∞"));
        assert!(msg.contains("robust Jeffreys curvature path is already installed"));
        assert!(msg.contains("not a Matérn/Duchon polynomial-nullspace"));
    }

    /// A non-converged inner solve with a BOUNDED fitted predictor must NOT
    /// surface the separation error — the non-convergence is reported through
    /// the existing downstream path, not as a runaway.
    #[test]
    pub(crate) fn runaway_guard_silent_for_nonconverged_but_bounded_eta() {
        let x = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let design = dense_marginal_design(x, 0..0, Vec::new());
        let beta = Array1::from_vec(vec![5.0]);

        let msg = bernoulli_marginal_slope_runaway_error_from_beta(
            beta.view(),
            &design,
            &empty_spec(),
            false,
            "nonconverged-fixture",
        );
        assert!(
            msg.is_none(),
            "bounded fitted η must not raise the separation error even when the inner solve did not converge; got {msg:?}"
        );
    }

    /// A genuinely separating fit that ALSO failed to converge still surfaces
    /// the runaway error, and reports the non-converged inner status.
    #[test]
    pub(crate) fn runaway_guard_fires_for_nonconverged_separating_eta() {
        let x = Array2::<f64>::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let design = dense_marginal_design(x, 0..0, Vec::new());
        let beta = Array1::from_vec(vec![50.0]);

        let msg = bernoulli_marginal_slope_runaway_error_from_beta(
            beta.view(),
            &design,
            &empty_spec(),
            false,
            "nonconverged-separating-fixture",
        )
        .expect("separating |η|∞ at non-convergence must still trip the guard");

        assert!(msg.contains(
            "the inner solve failed while already carrying a separation-scale predictor"
        ));
    }
}

pub(crate) fn build_marginal_blockspec_bms(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
    logslope_design: &TermCollectionDesign,
    logslope_offset: &Array1<f64>,
    logslope_baseline: f64,
    p_marginal: usize,
    influence_columns: Option<&Array2<f64>>,
    influence_ridge_log_lambda: f64,
) -> Result<ParameterBlockSpec, String> {
    let offset_m = offset + baseline;
    let offset_s = logslope_offset + logslope_baseline;
    let raw_marginal_dense = design
        .design
        .try_to_dense_arc("build_marginal_blockspec_bms::marginal")?;
    let marginal_dense =
        widen_marginal_dense_with_influence(&raw_marginal_dense, influence_columns)?;
    let logslope_dense = logslope_design
        .design
        .try_to_dense_arc("build_marginal_blockspec_bms::logslope")?;
    let callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(BmsMarginalJacobian {
        marginal_dense: Arc::clone(&marginal_dense),
        logslope_dense,
        offset_m: offset_m.clone(),
        offset_s,
        p_marginal,
    });
    let (penalties, nullspace_dims, initial_log_lambdas) = marginal_penalties_with_influence_ridge(
        design,
        &rho,
        influence_columns,
        influence_ridge_log_lambda,
    )?;
    Ok(ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            (*marginal_dense).clone(),
        )),
        offset: offset_m,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta: widen_marginal_beta_hint(beta_hint, p_marginal),
        // Canonical-gauge architecture (issue #322): give marginal_surface
        // strictly higher priority than logslope_surface so the priority-
        // ordered RRQR in `canonicalize_for_identifiability` presents
        // marginal columns first and routes any cross-block alias drop into
        // logslope.  Equal priorities (the previous default of 100/100)
        // produced a same-priority `hard_alias_pair` whenever a
        // high-dimensional smooth — e.g. `s(x, type=duchon, centers>=6)`
        // in the location block — accidentally spanned the logslope basis
        // direction, leaving the joint Hessian with a structural null and
        // the spectral Newton solve refusing to step.  The values mirror
        // the canonical-gauge entry for survival marginal-slope
        // (marginal=150, logslope=120).
        gauge_priority: GAUGE_PRIORITY_MARGINAL,
        jacobian_callback: Some(callback),
        stacked_design: None,
        stacked_offset: None,
    })
}

pub(crate) fn build_logslope_blockspec_bms(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
    marginal_design: &TermCollectionDesign,
    marginal_offset: &Array1<f64>,
    marginal_baseline: f64,
    z: Arc<Array1<f64>>,
    p_marginal: usize,
    influence_columns: Option<&Array2<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let offset_s = offset + baseline;
    let offset_m = marginal_offset + marginal_baseline;
    let raw_marginal_dense = marginal_design
        .design
        .try_to_dense_arc("build_logslope_blockspec_bms::marginal")?;
    // The logslope Jacobian reconstructs q_i = M·β_m + offset_m; with the
    // absorbed influence columns folded into the marginal index, the marginal
    // reference design and p_marginal MUST be the widened [M | Z̃] / (p_m+p₁)
    // so β_m slices to the absorber and q_i carries the Z̃·γ shift (#461).
    let marginal_dense =
        widen_marginal_dense_with_influence(&raw_marginal_dense, influence_columns)?;
    let logslope_dense = design
        .design
        .try_to_dense_arc("build_logslope_blockspec_bms::logslope")?;
    let callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(BmsLogslopeJacobian {
        marginal_dense,
        logslope_dense: Arc::clone(&logslope_dense),
        offset_m,
        offset_s: offset_s.clone(),
        z,
        p_marginal,
    });
    Ok(ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            (*logslope_dense).clone(),
        )),
        offset: offset_s,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        // Canonical-gauge architecture (issue #322): logslope is strictly
        // lower priority than marginal so the priority-ordered RRQR in
        // `canonicalize_for_identifiability` demotes a shared cross-block
        // direction here, not in marginal.  Mirrors the survival-mgs
        // value (marginal=150, logslope=120).  See the matching comment
        // on `build_marginal_blockspec_bms` for the failure mode this
        // resolves.
        gauge_priority: GAUGE_PRIORITY_LOGSLOPE,
        jacobian_callback: Some(callback),
        stacked_design: None,
        stacked_offset: None,
    })
}

pub(crate) fn build_deviation_aux_blockspec(
    name: &str,
    prepared: &DeviationPrepared,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let mut block = prepared.block.clone();
    block.initial_log_lambdas = Some(rho);
    let candidate_beta = beta_hint.or_else(|| Some(Array1::<f64>::zeros(block.design.ncols())));
    block.initial_beta = candidate_beta
        .map(|beta| {
            let zero = Array1::<f64>::zeros(beta.len());
            project_monotone_feasible_beta(&prepared.runtime, &zero, &beta, name)
        })
        .transpose()?;
    let mut spec = block.intospec(name)?;
    // Deviation auxiliary blocks (score_warp_dev, link_dev, and any
    // future flex block routed through this builder) model pure
    // shape modifications on top of parametric anchors. They must
    // never own a shared affine direction with the parametric
    // (time / marginal / logslope) blocks. The canonical-gauge
    // selector drops shared directions from blocks with lower
    // gauge_priority first; assigning a value below the parametric
    // default (GAUGE_PRIORITY_CANDIDATE_FLEX) realises that contract
    // automatically.
    spec.gauge_priority = match name {
        "link_dev" => GAUGE_PRIORITY_LINK_DEV,
        // score_warp_dev gets a slightly higher priority than link_dev
        // because in mixed-flex configurations (both blocks present)
        // link_dev is the residualised one (orthogonalised against the
        // parametric anchors PLUS the already-prepared score_warp
        // basis at construction time); link_dev should therefore yield
        // first when an alias still survives into the joint design.
        "score_warp_dev" => GAUGE_PRIORITY_SCORE_WARP_DEV,
        _ => GAUGE_PRIORITY_DEVIATION_DEFAULT,
    };
    Ok(spec)
}

pub(crate) fn push_deviation_aux_blockspecs(
    blocks: &mut Vec<ParameterBlockSpec>,
    rho: &Array1<f64>,
    cursor: &mut usize,
    score_warp_prepared: Option<&DeviationPrepared>,
    link_dev_prepared: Option<&DeviationPrepared>,
    score_warp_beta_hint: Option<Array1<f64>>,
    link_dev_beta_hint: Option<Array1<f64>>,
) -> Result<(), String> {
    if let Some(prepared) = score_warp_prepared {
        let rho_h = rho
            .slice(s![*cursor..*cursor + prepared.block.penalties.len()])
            .to_owned();
        *cursor += prepared.block.penalties.len();
        blocks.push(build_deviation_aux_blockspec(
            "score_warp_dev",
            prepared,
            rho_h,
            score_warp_beta_hint,
        )?);
    }
    if let Some(prepared) = link_dev_prepared {
        let rho_w = rho
            .slice(s![*cursor..*cursor + prepared.block.penalties.len()])
            .to_owned();
        blocks.push(build_deviation_aux_blockspec(
            "link_dev",
            prepared,
            rho_w,
            link_dev_beta_hint,
        )?);
    }
    Ok(())
}

fn inner_fit(
    family: &BernoulliMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    let mut options = options.clone();
    // BMS carries fixed physical ridge penalties that regularize coefficient
    // geometry but are not REML coordinates. The exact hyper-Hessian route can
    // stall after that projection; the family has a dedicated exact-gradient
    // path with full-data polish, so make it the primary nested smoother.
    options.use_outer_hessian = false;
    options.outer_tol = options.outer_tol.max(2.0e-5);
    fit_custom_family(family, blocks, &options).map_err(|e| e.to_string())
}

pub fn fit_bernoulli_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: BernoulliMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    policy: &crate::solver::resource::ResourcePolicy,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    let mut spec = spec;
    let data_view = data;
    validate_spec(data_view, &spec)?;
    // Freeze the measure-jet representer length-scale dial on the coupled
    // marginal + log-slope surfaces (#1116). A shared mjs basis feeds BOTH
    // blocks; a design-moving ℓ on those shared covariates lets the outer
    // search reach a sharp ℓ where a marginal smooth direction trades off
    // against the log-slope into a separation-scale runaway (|β|→1e3). The auto
    // (frozen) ℓ is well-conditioned here — the pre-#1116 behavior this
    // restores. ℓ-learning stays on for single-surface (e.g. Gaussian) fits,
    // where there is no marginal/log-slope coupling to destabilize.
    let mjs_frozen_marginal =
        crate::smooth::freeze_measure_jet_length_scale_learning(&mut spec.marginalspec);
    let mjs_frozen_logslope =
        crate::smooth::freeze_measure_jet_length_scale_learning(&mut spec.logslopespec);
    if mjs_frozen_marginal + mjs_frozen_logslope > 0 {
        log::info!(
            "[BMS spatial] froze measure-jet length-scale learning on {} marginal + {} log-slope \
             term(s): the coupled surface keeps ℓ at its conditioned auto value (#1116)",
            mjs_frozen_marginal,
            mjs_frozen_logslope
        );
    }
    let mut effective_kappa_options = kappa_options.clone();
    // Honor explicit `length_scale=X` in the user's formula: when every
    // spatial term in BOTH the marginal mean and log-slope blocks carries
    // a user-supplied scalar length scale and no per-axis anisotropy is
    // requested, there is nothing for the joint-spatial outer optimizer
    // to do. Routing through it anyway spends ~80 outer ARC iters stalled
    // at the user's chosen ρ (the n-block ARC's first proposed step lands
    // at the box corner and never recovers), then falls through to the
    // ρ-only "custom family" path which is what we wanted all along.
    // Short-circuit straight to the ρ-only path.
    let kappa_locked_marginal = crate::smooth::all_spatial_terms_kappa_fixed(&spec.marginalspec);
    let kappa_locked_logslope = crate::smooth::all_spatial_terms_kappa_fixed(&spec.logslopespec);
    if effective_kappa_options.enabled && kappa_locked_marginal && kappa_locked_logslope {
        log::info!(
            "[BMS spatial] disabling κ/ψ optimization: every spatial term has an \
             explicit length_scale and no anisotropy; user-supplied kernel scale is fixed"
        );
        effective_kappa_options.enabled = false;
    }
    let flex_spatial_pilot_path = (spec.score_warp.is_some() || spec.link_dev.is_some())
        && spec.y.len() >= BMS_FLEX_SPATIAL_OUTER_PILOT_ROW_THRESHOLD
        && effective_kappa_options.enabled;
    if flex_spatial_pilot_path {
        let marginal_terms = spatial_length_scale_term_indices(&spec.marginalspec);
        let logslope_terms = spatial_length_scale_term_indices(&spec.logslopespec);
        let marginal_updates = apply_spatial_anisotropy_pilot_initializer(
            data_view,
            &mut spec.marginalspec,
            &marginal_terms,
            effective_kappa_options.pilot_subsample_threshold,
            &effective_kappa_options,
        );
        let logslope_updates = apply_spatial_anisotropy_pilot_initializer(
            data_view,
            &mut spec.logslopespec,
            &logslope_terms,
            effective_kappa_options.pilot_subsample_threshold,
            &effective_kappa_options,
        );
        effective_kappa_options.enabled = false;
        log::info!(
            "[BMS spatial] n={} flex=true pilot_geometry_updates={} iterative_spatial_outer=false reason=large-flex-spatial-pilot",
            spec.y.len(),
            marginal_updates + logslope_updates,
        );
    }
    let (z_standardized, z_normalization) = standardize_latent_z_with_policy(
        &spec.z,
        &spec.weights,
        "bernoulli-marginal-slope",
        &spec.latent_z_policy,
    )?;
    spec.z = z_standardized;
    let sigma_learnable = matches!(
        &spec.frailty,
        FrailtySpec::GaussianShift { sigma_fixed: None }
    );
    let initial_sigma = match &spec.frailty {
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(s),
        } => Some(*s),
        FrailtySpec::GaussianShift { sigma_fixed: None } => Some(0.5),
        FrailtySpec::None => None,
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(
                "internal: validate_spec should have rejected unsupported marginal-slope frailty"
                    .to_string(),
            );
        }
    };
    let probit_scale = probit_frailty_scale(initial_sigma);
    let (_raw_joint_designs, mut joint_specs) = build_term_collection_designs_and_freeze_joint(
        data_view,
        &[spec.marginalspec.clone(), spec.logslopespec.clone()],
    )
    .map_err(|e| e.to_string())?;
    let marginalspec_boot = joint_specs.remove(0);
    let logslopespec_boot = joint_specs.remove(0);
    // Rebuild the probe designs from the frozen `*_boot` specs so the probe's
    // penalty topology matches the topology produced by every other build path
    // in this optimization. The spatial optimizer's own bootstrap
    // (`build_term_collection_designs_and_freeze_joint(data, &[marginalspec_boot,
    // logslopespec_boot])` inside `optimize_spatial_length_scale_exact_joint`)
    // and every subsequent kappa-driven rebuild feed the basis builder the
    // captured `FrozenTransform` identifiability. Applying that captured
    // transform to the same kernel can land the structural null-space block on
    // the other side of `build_nullspace_shrinkage_penalty`'s spectral
    // tolerance, so the raw and frozen builds disagree on whether the trend
    // ridge survives as an active penalty candidate. Without this rebuild,
    // `marginal_penalty_count` / `logslope_design.penalties.len()` are taken
    // from the raw build but every subsequent evaluator measures the frozen
    // build, and `evaluate_custom_family_joint_hyper` refuses with a
    // "joint hyper rho dimension mismatch". Mirrors the CTN-side fix in
    // `fit_transformation_normal`.
    let (mut joint_designs, _) = build_term_collection_designs_and_freeze_joint(
        data_view,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
    )
    .map_err(|e| format!("failed to rebuild frozen probe BMS joint designs: {e}"))?;
    let marginal_design = joint_designs.remove(0);
    let logslope_design = joint_designs.remove(0);
    // #905: the conditional `E[z|C]`/`Var(z|C)` Rao gate conditions on the
    // marginal-index span a(C) (= the marginal design columns), which is
    // exactly where the `b(C)·m(C)` leakage lives. It is engaged only on the
    // raw-z path (no CTN Stage-1 influence absorber); when an absorber is
    // active the conditional leakage is already absorbed (#461) and the
    // widened-marginal predict seam must not be perturbed by replacing z.
    let absorber_active = spec
        .score_influence_jacobian
        .as_ref()
        .is_some_and(|j| j.ncols() > 0);
    let conditioning_dense = if absorber_active {
        None
    } else {
        Some(
            marginal_design
                .design
                .try_to_dense_arc("bernoulli marginal-slope conditional latent-z gate")?,
        )
    };
    let (latent_measure, latent_z_calibration) = build_latent_measure_with_geometry(
        &spec.z,
        &spec.weights,
        &spec.latent_z_policy,
        conditioning_dense.as_ref().map(|d| d.view()),
    )?;
    if latent_measure.is_empirical() && sigma_learnable {
        return Err("empirical latent-measure marginal-slope calibration requires fixed GaussianShift sigma; learnable sigma derivatives must be fit under the standard-normal latent measure"
                    .to_string());
    }

    let y = Arc::new(spec.y.clone());
    let weights = Arc::new(spec.weights.clone());
    // Apply rank-INT calibration to training z before any downstream
    // consumer (pooled probit baseline, term-collection designs, the
    // family's PIRLS loops) sees it. The calibration is persisted on the
    // fit result so prediction applies the identical monotone map.
    let z = match &latent_z_calibration {
        LatentMeasureCalibration::None => Arc::new(spec.z.clone()),
        LatentMeasureCalibration::RankInverseNormal(cal) => {
            Arc::new(cal.apply_to_training(&spec.z)?)
        }
        LatentMeasureCalibration::ConditionalLocationScale(cal) => {
            // ζ = (z − m(C))/√v(C) on the marginal-index span. The conditioning
            // block was built above (raw-z path only), so it is present here.
            let a_block = conditioning_dense.as_ref().ok_or_else(|| {
                "conditional latent calibration requires the marginal conditioning block"
                    .to_string()
            })?;
            Arc::new(cal.apply(spec.z.view(), a_block.view())?)
        }
    };
    let z_train = z.as_ref();
    let pilot_baseline = pooled_probit_baseline(&spec.y, z_train, &spec.weights)?;
    let baseline = (
        bernoulli_marginal_slope_eta_from_probability(
            &spec.base_link,
            normal_cdf(pilot_baseline.0),
            "bernoulli marginal-slope baseline link inversion",
        )?,
        pilot_baseline.1 / probit_scale,
    );

    // Score-warp basis construction is β-independent (identifiability is
    // provided by the smoothness-null-space drop on the basis transform,
    // not by a data-distribution moment anchor at the rigid-pilot η₀), so
    // the standard-normal and empirical latent-measure branches build the
    // same block. There is no row-weight pilot to thread into the basis;
    // the latent-measure split is enforced upstream via the empirical
    // intercept solve in `build_row_exact_context_with_stats`, not in the
    // deviation basis.
    // Score-warp basis is built first, then immediately reparameterised
    // against the parametric span (marginal + logslope columns at the n
    // training rows) so its column span is orthogonal to span(X_marginal,
    // X_logslope) by construction. This is the first half of the joint-
    // design identifiability invariant; the second half (link-deviation
    // orthogonalised against parametric + the now-reparameterised score-
    // warp) runs inside the link-deviation closure below. Together they
    // ensure `[X_marginal | X_logslope | Φ_score_warp · T_sw |
    // Φ_link_dev · T_lw]` has full numerical column rank, structurally
    // bounding `σ_min(joint H + S) ≥ λ_min(S) > 0` regardless of how β
    // drifts the linear predictor distribution during PIRLS.
    // Cross-block W-metric pilot. The joint penalised Hessian during PIRLS
    // uses the probit-style data Hessian row metric
    //
    //   W_pirls[i] = spec.weights[i] · φ(η_i)² / (μ_i·(1−μ_i))
    //
    // which is the canonical IRLS row weight. The cross-block
    // orthogonalisation below must use this metric (not uniform
    // spec.weights) so that `Aᵀ W C̃ = 0` holds in the same inner product
    // the joint Hessian sees — otherwise A and C̃ are merely Euclidean-
    // orthogonal, `Aᵀ W_pirls C̃ ≠ 0`, the joint Hessian carries a near-
    // null direction along the W-metric alias, and REML can drive the
    // flex block's λ small enough that the alias direction's joint
    // Hessian eigenvalue collapses. β then runs away along the alias
    // (manifest as `rho≈2.0`, constant `step_inf`, growing `beta_inf`
    // during PIRLS, and the inner solve hitting `inner_max_cycles`
    // without satisfying the KKT residual).
    //
    // Use the rigid pooled-probit pilot η for score-warp (its basis is
    // β-independent in z, so the rigid pilot suffices) and the one-GN-
    // stepped pilot η for link-deviation (its basis is evaluated at the
    // same eta_pilot used here, so the orthogonalisation metric matches
    // the basis evaluation point exactly). Both are β-independent so the
    // orthogonalisation remains a one-shot construction-time step.
    let rigid_pilot_eta = rigid_pooled_probit_pilot_eta(
        &spec.base_link,
        z_train,
        &spec.marginal_offset,
        &spec.logslope_offset,
        baseline.0,
        baseline.1,
        probit_scale,
    )?;
    let cross_block_pilot_w_score_warp =
        pilot_irls_hessian_row_metric_at_eta(&rigid_pilot_eta, &spec.weights);

    // Absorbed Stage-1 influence columns (#461, design §3). When the workflow
    // chained a CTN Stage-1 into this marginal-slope fit, `spec.score_influence_
    // jacobian` carries the out-of-fold `J = ∂z/∂θ₁`; the realized leakage
    // directions `Z_infl = diag(s_f·β̂₀)·J` are residualised against the
    // marginal span (logslope-aligned component retained) and appended to the
    // additive marginal-index block as a fixed-ridge absorber, so the joint
    // penalised solve makes the (α,β) score orthogonal to span(Z_infl) — the
    // x-dependent realisation of `ψ − Π_η[ψ]`. `None` ⇒ raw z, and the free
    // score_warp spline below is the x-free-column fallback. β̂₀(x_i) is the
    // rigid-pilot logslope `baseline.1 + logslope_offset[i]`; s_f = probit_scale.
    let influence_columns = if let Some(jac) = spec
        .score_influence_jacobian
        .as_ref()
        .filter(|j| j.ncols() > 0)
    {
        let marginal_dense_for_proj = marginal_design
            .design
            .try_to_dense_arc("bernoulli marginal-slope influence-block marginal projection")?;
        let marginal_dense = marginal_dense_for_proj.as_ref();
        if jac.nrows() != marginal_dense.nrows() {
            return Err(format!(
                "influence block: Jacobian has {} rows, marginal design has {}",
                jac.nrows(),
                marginal_dense.nrows()
            ));
        }
        // Z̃ = residualize(diag(s_f·β̂₀)·J) against the marginal span in the
        // rigid-pilot W-metric, via the SHARED core entry point (single source
        // of truth across bms + survival; the two families differ ONLY in how
        // they install the returned Z̃ — bms widens [M | Z̃], survival adds a
        // dedicated η₁ channel). β̂₀(x_i) = baseline.1 + logslope_offset[i];
        // s_f = probit_scale; W = the rigid-pilot PIRLS row metric.
        let rigid_logslope_at_rows = &spec.logslope_offset + baseline.1;
        let residualized =
            crate::families::marginal_slope_orthogonal::residualized_influence_block(
                jac,
                z_train,
                &rigid_logslope_at_rows,
                probit_scale,
                marginal_dense.view(),
                &cross_block_pilot_w_score_warp,
            )?;
        Some(residualized)
    } else {
        None
    };
    let mut cross_block_warnings: Vec<CrossBlockIdentifiabilityWarning> = Vec::new();
    let score_warp_prepared = if let Some(cfg) = spec.score_warp.as_ref() {
        use super::deviation_runtime::ParametricAnchorBlock;
        let mut prepared = build_score_warp_deviation_block_from_seed(z_train, cfg)?;
        // `install_compiled_flex_block_into_runtime` now delegates
        // its math body to `identifiability::families::compiler::compile` (commit
        // 4e20b8dc8); the prior Phase-4a shadow compile here was a
        // duplicate of that internal call and has been removed.
        let outcome = install_compiled_flex_block_into_runtime(
            &mut prepared,
            z_train,
            cfg,
            &[
                (&marginal_design.design, ParametricAnchorBlock::Marginal),
                (&logslope_design.design, ParametricAnchorBlock::Logslope),
            ],
            &[],
            &cross_block_pilot_w_score_warp,
        )?;
        match outcome {
            FlexCompileOutcome::Reparameterised => Some(prepared),
            FlexCompileOutcome::FullyAliased { reason } => {
                // Record via the structured channel. Keep the original
                // (non-compiled) design so the unified audit sees score_warp_dev
                // and attributes the drop via dropped_columns (gauge_priority=80
                // is below marginal=150 / logslope=120, so RRQR correctly
                // demotes score_warp_dev when it aliases those blocks).
                cross_block_warnings.push(CrossBlockIdentifiabilityWarning {
                    candidate_label: "score_warp",
                    anchor_summary: "marginal+logslope".to_string(),
                    reason,
                });
                Some(prepared)
            }
        }
    } else {
        None
    };
    // Build the link-deviation block. The basis lives in η-space, and at
    // PIRLS time `runtime.design(η_current)` is re-evaluated at the
    // current β-dependent η, so the basis is genuinely β-dependent during
    // optimisation. The construction-time seed is used only for (a) knot
    // placement in η-space and (b) the cross-block identifiability check
    // that computes the basis-space transform `T` orthogonalising the
    // candidate against the parametric and score-warp anchors at training
    // rows.
    //
    // Using the rigid pooled probit pilot directly (`q0 = a₀·√(…) + s_f·
    // b₀·z`) is structurally degenerate: with zero per-row offsets it is
    // affine in z, so a degree-3 I-spline of `q0` spans the same column
    // space at training rows as a degree-3 I-spline of z, and the cross-
    // block check finds the candidate fully aliased by the score-warp
    // anchor even though at any non-rigid β the link-deviation carries
    // PC/age structure the score-warp cannot represent.
    //
    // Instead, seed both knot placement and the orthogonalisation pivot at
    // a non-rigid pilot η computed via one probit Gauss-Newton step from
    // the rigid pilot onto the full marginal design (see
    // `pilot_eta_for_link_dev_orthogonalisation`). The pilot is row-varying
    // in PCs/age and the resulting `T` drops only directions aliased
    // across all β. The score-warp basis at training rows is also threaded
    // in as a flex anchor when active so the kept directions are jointly
    // orthogonal to parametric ⊕ score-warp.
    let link_dev_prepared = if let Some(cfg) = spec.link_dev.as_ref() {
        let eta_pilot = pilot_eta_for_link_dev_orthogonalisation(
            &spec.base_link,
            &spec.y,
            z_train,
            &spec.weights,
            &marginal_design.design,
            &spec.marginal_offset,
            &spec.logslope_offset,
            baseline.0,
            baseline.1,
            probit_scale,
        )?;
        let link_dev_seed = padded_deviation_seed(&eta_pilot, 1.0, 0.5);
        let mut prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
            &link_dev_seed,
            &eta_pilot,
            cfg,
        )?;
        // Cross-block identifiability for the link-deviation basis. The
        // anchor union covers BOTH possible aliasing channels:
        //
        //  - Parametric: location and logslope designs evaluated at the n
        //    training rows. Columns of `Φ_link_dev(q0)` that reproduce
        //    parametric features become null-direction targets in the
        //    joint penalised Hessian since `S_link_dev` has no mass on
        //    them.
        //
        //  - Score-warp (when active): the now-reparameterised score-warp
        //    basis, also evaluated at training rows. Both flex bases are
        //    cubic I-spline cubic combinations of an η-pilot scalar, and
        //    even with each block's own smoothness-null-space drop their
        //    column spans can still overlap inside the orthogonal
        //    complement of `{1, η_pilot}`.
        //
        // After the orthogonalisation, `[X_marginal | X_logslope |
        // Φ_score_warp · T_sw | Φ_link_dev · T_lw]` has full numerical
        // column rank at training rows, so `σ_min(joint H+S) ≥ λ_min(S)
        // > 0` for every β. This is the standard GAM `gam.side`
        // convention generalised to multi-anchor unions (mgcv applies it
        // sequentially across smooths sharing a covariate).
        // When `install_compiled_flex_block_into_runtime`
        // reparameterised the score-warp runtime against the parametric
        // anchor union (marginal + logslope), it installed an
        // `anchor_residual` and cached the training-row parametric
        // anchor matrix on the runtime. `runtime.design()` on a
        // residualised runtime returns the *raw* basis evaluation,
        // which `assert`s the caller hasn't conflated with the
        // reparameterised basis — we want the reparameterised one
        // here, so go through `design_at_training_with_residual` so
        // the cached anchor rows are folded in. For score-warp
        // configurations where reparameterisation was a no-op (no
        // residual installed) the same call falls back to the raw
        // `design()` path, so the residual-vs-no-residual branches
        // converge on the right matrix.
        let score_warp_anchor_design = score_warp_prepared
            .as_ref()
            .map(|sw| sw.runtime.design_at_training_with_residual(z_train))
            .transpose()?;
        use super::deviation_runtime::ParametricAnchorBlock;
        let parametric_anchors: [(&DesignMatrix, ParametricAnchorBlock); 2] = [
            (&marginal_design.design, ParametricAnchorBlock::Marginal),
            (&logslope_design.design, ParametricAnchorBlock::Logslope),
        ];
        let flex_anchor_slot: Option<&Array2<f64>> = score_warp_anchor_design.as_ref();
        let flex_anchors: Vec<&Array2<f64>> = flex_anchor_slot.into_iter().collect();
        // W-metric for link-deviation orthogonalisation: same IRLS-style
        // probit Hessian row weight as the score-warp path, but evaluated at
        // `eta_pilot` (the one-GN-stepped pilot at which the link-dev basis
        // itself is anchored).
        let cross_block_pilot_w_link_dev =
            pilot_irls_hessian_row_metric_at_eta(&eta_pilot, &spec.weights);
        let outcome = install_compiled_flex_block_into_runtime(
            &mut prepared,
            &eta_pilot,
            cfg,
            &parametric_anchors,
            &flex_anchors,
            &cross_block_pilot_w_link_dev,
        )?;
        match outcome {
            FlexCompileOutcome::Reparameterised => Some(prepared),
            FlexCompileOutcome::FullyAliased { reason } => {
                // Record via the structured channel. Keep the original
                // (non-compiled) design so the unified audit sees link_dev
                // and attributes the drop via dropped_columns (gauge_priority=60
                // is below all parametric blocks so RRQR correctly demotes
                // link_dev when it aliases marginal / logslope / score_warp).
                cross_block_warnings.push(CrossBlockIdentifiabilityWarning {
                    candidate_label: "link_deviation",
                    anchor_summary: "marginal+logslope+score_warp".to_string(),
                    reason,
                });
                Some(prepared)
            }
        }
    } else {
        None
    };
    let extra_rho0 = {
        let mut out = Vec::new();
        if let Some(ref prepared) = score_warp_prepared {
            out.extend(std::iter::repeat_n(0.0, prepared.block.penalties.len()));
        }
        if let Some(ref prepared) = link_dev_prepared {
            out.extend(std::iter::repeat_n(0.0, prepared.block.penalties.len()));
        }
        out
    };
    // Reduced-basis orthogonalisation of the logslope design through the BMS
    // family's OWN internal `logslope_design` geometry (robust cure for the
    // marginal↔logslope structural confound). Robustness is unconditional, so we
    // always reparameterize the logslope coordinate space to a full-rank reduced
    // basis `T` whose effective weighted columns are W-orthogonal to the marginal
    // span at the rigid pilot — removing the rank-soft confounded direction the
    // former pinned overlap ridge merely penalised. The transform is
    // β/ρ-independent (pilot geometry only), so it is a one-shot construction-
    // time map applied to every per-iteration logslope design inside
    // `build_blocks` / `make_family`, and inverted at fit-result assembly so the
    // reported logslope β is in the original basis. `None` ⇒ nothing to reduce
    // (no rank-soft confounded direction) ⇒ raw design used everywhere.
    let logslope_reduced_reparam: Option<ReducedLogslopeReparam> = build_reduced_logslope_reparam(
        &marginal_design,
        &logslope_design,
        z.as_ref(),
        &cross_block_pilot_w_score_warp,
        &spec.marginal_offset,
        &spec.logslope_offset,
        baseline.0,
        baseline.1,
        probit_scale,
    )?;
    // Apply the reduced reparam to a logslope `TermCollectionDesign`, or return
    // the raw design clone when the reparam is absent (flag off / nothing to
    // reduce). Used by both `build_blocks` and `make_family` so the family's
    // internal design, the block design, β width, jacobian, penalty, and the
    // `validate_exact_block_state_shapes` check all agree at the reduced width.
    let reduce_logslope_design =
        |logslope_design: &TermCollectionDesign| -> Result<TermCollectionDesign, String> {
            match logslope_reduced_reparam.as_ref() {
                Some(reparam) => reparameterize_logslope_design_reduced(logslope_design, reparam),
                None => Ok(logslope_design.clone()),
            }
        };

    let marginal_penalty_count = marginal_design.penalties.len();
    let setup = joint_setup(
        data_view,
        &marginalspec_boot,
        &logslopespec_boot,
        marginal_penalty_count,
        logslope_design.penalties.len(),
        &extra_rho0,
        &effective_kappa_options,
    );
    let setup = if sigma_learnable {
        setup.with_auxiliary(
            Array1::from_vec(vec![initial_sigma.expect("learnable sigma seed").ln()]),
            Array1::from_vec(vec![0.01_f64.ln()]),
            Array1::from_vec(vec![5.0_f64.ln()]),
        )
    } else {
        setup
    };
    let final_sigma_cell = std::cell::Cell::new(initial_sigma);
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);
    let runaway_error = RefCell::new(None::<String>);
    // Outer ρ-cache β-seed staging slot. On a cache hit the spatial-joint
    // optimizer invokes `seed_inner_beta_fn` before the first eval at the
    // restored ρ: per-block column widths aren't known until the first
    // `build_blocks(rho, …)` runs, so we stash the flat β here and the eval
    // closures promote it into `exact_warm_start` (the slot the inner
    // PIRLS / Newton solve actually consumes) on their first invocation.
    let pending_beta_seed = RefCell::new(None::<Array1<f64>>);
    let hints = RefCell::new(ThetaHints::default());
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        // Reduced-basis orthogonalisation: replace the per-iteration logslope
        // design with its full-rank reduced reparameterization `G·T` (flag ON);
        // a no-op clone when off. The reduced design carries the SAME number of
        // penalties (each S → Tᵀ S T), so the `rho_logslope` slice width below
        // is unchanged. Every consumer (marginal jacobian's c_i, logslope
        // blockspec design/β/penalty/jacobian) now agrees at the reduced width.
        let logslope_design_reduced = reduce_logslope_design(logslope_design)?;
        let logslope_design = &logslope_design_reduced;
        // Fixed #754/#461 ridges are appended inside
        // `marginal_penalties_with_influence_ridge` as physical penalties and
        // are excluded from `rho`; only genuine REML-learned smooth penalties
        // appear in the spatial joint setup.
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        cursor += logslope_design.penalties.len();
        let p_m = marginal_design.design.ncols()
            + influence_columns.as_ref().map(|z| z.ncols()).unwrap_or(0);
        let mut blocks = vec![
            build_marginal_blockspec_bms(
                marginal_design,
                baseline.0,
                &spec.marginal_offset,
                rho_marginal,
                hints.marginal_beta.clone(),
                logslope_design,
                &spec.logslope_offset,
                baseline.1,
                p_m,
                influence_columns.as_ref(),
                INFLUENCE_ABSORBER_FIXED_LOG_LAMBDA,
            )?,
            build_logslope_blockspec_bms(
                logslope_design,
                baseline.1,
                &spec.logslope_offset,
                rho_logslope,
                hints.logslope_beta.clone(),
                marginal_design,
                &spec.marginal_offset,
                baseline.0,
                Arc::clone(&z),
                p_m,
                influence_columns.as_ref(),
            )?,
        ];
        push_deviation_aux_blockspecs(
            &mut blocks,
            rho,
            &mut cursor,
            score_warp_prepared.as_ref(),
            link_dev_prepared.as_ref(),
            hints.score_warp_beta.clone(),
            hints.link_dev_beta.clone(),
        )?;
        Ok(blocks)
    };

    let intercept_warm_starts = new_intercept_warm_start_cache(y.len());
    let cell_moment_lru = new_cell_moment_lru_cache(policy);
    let cell_moment_cache_stats = new_cell_moment_cache_stats();
    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign,
                       sigma: Option<f64>|
     -> BernoulliMarginalSlopeFamily {
        // The kernel reads the marginal index from a matched (self.marginal_
        // design, β_m) pair. When the Stage-1 influence absorber is active the
        // marginal β is widened to [β_m; γ], so the family's marginal design
        // MUST be the widened [M | Z̃] for every per-row projection to slice
        // correctly (#461). With no absorber it is the raw design unchanged.
        let kernel_marginal_design = match influence_columns.as_ref() {
            Some(z_infl) => {
                let raw = marginal_design
                    .design
                    .try_to_dense_arc("make_family::widened-marginal")
                    .expect("dense marginal design for influence widening");
                let widened = widen_marginal_dense_with_influence(&raw, Some(z_infl))
                    .expect("widen marginal design with influence columns");
                DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from((*widened).clone()))
            }
            None => marginal_design.design.clone(),
        };
        // The family's row kernel reconstructs η_logslope = G·β_s and the
        // logslope Jacobian factor_i·G_i from this matched (logslope_design,
        // β_s) pair, so it MUST be the SAME reduced design `G·T` the block specs
        // fit against — otherwise β_s (reduced width) and the family design
        // (full width) desync. A no-op clone when the reparam is absent.
        let kernel_logslope_design = reduce_logslope_design(logslope_design)
            .expect("reduce logslope design for family construction")
            .design;
        BernoulliMarginalSlopeFamily {
            y: Arc::clone(&y),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            latent_measure: latent_measure.clone(),
            gaussian_frailty_sd: sigma,
            base_link: spec.base_link.clone(),
            marginal_design: kernel_marginal_design,
            logslope_design: kernel_logslope_design,
            score_warp: score_warp_runtime.clone(),
            link_dev: link_dev_runtime.clone(),
            policy: policy.clone(),
            cell_moment_lru: Arc::clone(&cell_moment_lru),
            cell_moment_cache_stats: Arc::clone(&cell_moment_cache_stats),
            intercept_warm_starts: Some(Arc::clone(&intercept_warm_starts)),
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    };

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let marginal_has_spatial = !marginal_terms.is_empty();
    let logslope_has_spatial = !logslope_terms.is_empty();
    let analytic_joint_derivatives_available =
        marginal_has_spatial || logslope_has_spatial || setup.log_kappa_dim() == 0;
    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err("exact bernoulli marginal-slope spatial optimization requires analytic joint psi derivatives"
                    .to_string());
    }
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design, initial_sigma);
    let (joint_gradient, joint_hessian) =
        custom_family_outer_derivatives(&initial_family, &initial_blocks, options);
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_gradient,
            crate::solver::rho_optimizer::Derivative::Analytic
        );
    // Keep the analytic outer Hessian advertised at large scale. The
    // row-tensor terms below are represented through block-local
    // `HyperOperator`s and cached exact-Hessian workspaces, so ARC/trust-region
    // can consume exact HVPs without falling back to BFGS merely because the
    // realized problem is large.
    let analytic_joint_hessian_available =
        analytic_joint_derivatives_available && joint_hessian.is_analytic();
    let kappa_options_ref: &SpatialLengthScaleOptimizationOptions = &effective_kappa_options;
    let sigma_from_theta = |theta: &Array1<f64>| -> Option<f64> {
        if sigma_learnable {
            Some(theta[setup.rho_dim() + setup.log_kappa_dim()].exp())
        } else {
            initial_sigma
        }
    };
    let derivative_block_cache = RefCell::new(
        None::<(
            Array1<f64>,
            Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        )>,
    );
    let theta_matches = |left: &Array1<f64>, right: &Array1<f64>| -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12 * (1.0 + lhs.abs().max(rhs.abs())))
    };
    let get_derivative_blocks = |theta: &Array1<f64>,
                                 specs: &[TermCollectionSpec],
                                 designs: &[TermCollectionDesign]|
     -> Result<
        Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        String,
    > {
        if let Some((cached_theta, cached_blocks)) = derivative_block_cache.borrow().as_ref()
            && theta_matches(cached_theta, theta)
        {
            return Ok(Arc::clone(cached_blocks));
        }

        let built = |specs: &[TermCollectionSpec],
                     designs: &[TermCollectionDesign]|
         -> Result<
            Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
            String,
        > {
            let marginal_psi_derivs = if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data_view, &specs[0], &designs[0])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: marginal block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let logslope_psi_derivs = if logslope_has_spatial {
                build_block_spatial_psi_derivatives(data_view, &specs[1], &designs[1])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: logslope block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let mut derivative_blocks = vec![marginal_psi_derivs, logslope_psi_derivs];
            if score_warp_runtime.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if link_dev_runtime.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if sigma_learnable {
                derivative_blocks
                    .last_mut()
                    .expect("bernoulli derivative block list is non-empty")
                    .push(crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                        None,
                        Array2::zeros((0, 0)),
                        Array2::zeros((0, 0)),
                        None,
                        None,
                        None,
                        None,
                    ));
            }
            Ok(derivative_blocks)
        }(specs, designs)?;
        let built = Arc::new(built);
        derivative_block_cache.replace(Some((theta.clone(), Arc::clone(&built))));
        Ok(built)
    };

    // Bernoulli marginal-slope is a multi-block family with β-dependent
    // joint Hessian: EFS/HybridEFS fixed-point structural invariant fails,
    // so we disable fixed-point at plan time rather than burning cycles on
    // a stalled first attempt that silently falls back.
    let outer_policy = {
        let psi_dim = setup.theta0().len() - setup.rho_dim();
        initial_family.outer_derivative_policy(&initial_blocks, psi_dim, options)
    };
    let exact_spatial_outer_tol = kappa_options_ref.rel_tol.max(EXACT_SPATIAL_OUTER_TOL_FLOOR);
    let solved = optimize_spatial_length_scale_exact_joint(
        data_view,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms.clone(), logslope_terms.clone()],
        kappa_options_ref,
        &setup,
        crate::seeding::SeedRiskProfile::GeneralizedLinear,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        None,
        outer_policy,
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            if let Some(err) = runaway_error.borrow().as_ref().cloned() {
                return Err(err);
            }
            assert_eq!(
                specs.len(),
                designs.len(),
                "spatial joint optimizer must supply one spec per design",
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let fit = inner_fit(&family, &blocks, options)?;
            if let Some(block) = fit.block_states.first()
                && let Some(err) = bernoulli_marginal_slope_runaway_error_from_beta(
                    block.beta.view(),
                    &designs[0],
                    &specs[0],
                    fit.outer_converged,
                    "final fit",
                )
            {
                runaway_error.replace(Some(err.clone()));
                return Err(err);
            }
            let mut hints_mut = hints.borrow_mut();
            let mut bidx = 0usize;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if score_warp_prepared.is_some() {
                if let Some(block) = fit.block_states.get(bidx) {
                    hints_mut.score_warp_beta = Some(block.beta.clone());
                }
                bidx += 1;
            }
            if link_dev_prepared.is_some()
                && let Some(block) = fit.block_states.get(bidx)
            {
                hints_mut.link_dev_beta = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set: &crate::families::row_kernel::RowSet| {
            if let Some(err) = runaway_error.borrow().as_ref().cloned() {
                return Err(err);
            }
            use crate::reml_contracts::EvalMode;
            // One-shot row-measure waypoint. This closure runs on EVERY outer
            // objective evaluation (value/gradient/Hessian probes, line-search
            // cost-only probes, EFS evals), so an unconditional per-eval line
            // floods the biobank fit log with thousands of near-identical
            // entries. The bridge already emits a timed `[STAGE] outer eval`
            // marker per eval; this one records the row-measure exactly once.
            static BMS_OUTER_EVAL_ROWSET_LOGGED: std::sync::Once = std::sync::Once::new();
            BMS_OUTER_EVAL_ROWSET_LOGGED.call_once(|| {
                let row_set_rows = match row_set {
                    crate::families::row_kernel::RowSet::All => spec.y.len(),
                    crate::families::row_kernel::RowSet::Subsample { rows, .. } => rows.len(),
                };
                log::debug!(
                    "[BMS exact outer eval] mode={eval_mode:?} row_set_rows={row_set_rows}"
                );
            });
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            // Promote a staged β seed (deposited by the outer ρ-cache hit
            // before any eval ran) into the family warm-start slot now that
            // we know the per-block widths from the freshly built blocks.
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = blocks.iter().map(|b| b.design.ncols()).collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[BMS] outer ρ-cache β-warm-start rejected: {e}; falling back to cold β"
                        );
                    }
                }
            }
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            // Downgrade to ValueAndGradient when the caller asks for a
            // Hessian we can't provide; preserve ValueOnly probes for
            // line-search cost-only evaluation.
            let effective_mode = match eval_mode {
                EvalMode::ValueGradientHessian if !analytic_joint_hessian_available => {
                    EvalMode::ValueAndGradient
                }
                other => other,
            };
            let mut eval_options =
                joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol);
            if let crate::families::row_kernel::RowSet::Subsample { rows, n_full } = row_set {
                let subsample =
                    crate::solver::outer_subsample::OuterScoreSubsample::from_weighted_rows(
                        rows.as_ref().clone(),
                        *n_full,
                        0,
                    );
                eval_options.outer_score_subsample = Some(Arc::new(subsample));
                eval_options.auto_outer_subsample = false;
            }
            let eval = evaluate_custom_family_joint_hyper_shared(
                &family,
                &blocks,
                &eval_options,
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )?;
            if let Some(err) = bernoulli_marginal_slope_runaway_error(
                &eval.warm_start,
                &designs[0],
                &specs[0],
                eval.inner_converged,
                "exact outer evaluation",
            ) {
                runaway_error.replace(Some(err.clone()));
                return Err(err);
            }
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "exact bernoulli marginal-slope inner solve did not converge".to_string(),
                );
            }
            if matches!(eval_mode, EvalMode::ValueGradientHessian)
                && analytic_joint_hessian_available
                && !eval.outer_hessian.is_analytic()
            {
                return Err("exact bernoulli marginal-slope joint [rho, psi] objective did not return an outer Hessian"
                            .to_string());
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            if let Some(err) = runaway_error.borrow().as_ref().cloned() {
                return Err(err);
            }
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = blocks.iter().map(|b| b.design.ncols()).collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[BMS] outer ρ-cache β-warm-start rejected (efs): {e}; falling back to cold β"
                        );
                    }
                }
            }
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            let eval = evaluate_custom_family_joint_hyper_efs_shared(
                &family,
                &blocks,
                &joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol),
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )?;
            if let Some(err) = bernoulli_marginal_slope_runaway_error(
                &eval.warm_start,
                &designs[0],
                &specs[0],
                eval.inner_converged,
                "EFS outer evaluation",
            ) {
                runaway_error.replace(Some(err.clone()));
                return Err(err);
            }
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "exact bernoulli marginal-slope EFS inner solve did not converge".to_string(),
                );
            }
            Ok(eval.efs_eval)
        },
        crate::families::marginal_slope_shared::make_beta_seed_validator(&pending_beta_seed),
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    // Reduced-basis round-trip (robust cure). When the logslope design was
    // orthogonalised to a reduced basis `G·T`, the fitted logslope coefficient
    // `β'` lives in the reduced coordinates (width `r`). The returned
    // `logslope_design` / `logslopespec_resolved` are the ORIGINAL full-width
    // basis (prediction rebuilds full `G` from the resolved spec), so map the
    // reported logslope coefficients back to the original basis `β_logslope =
    // T·β'` (predictor-identical: `G·(T·β') = (G·T)·β'`). The marginal block,
    // aux blocks, and the internal reduced-width flat β/geometry are untouched;
    // only the per-block reported logslope coefficients (blocks[1] and
    // block_states[1]) — which prediction/reporting consume against the full
    // design — are lifted to full width.
    let mut solved_fit = solved.fit;
    if let Some(reparam) = logslope_reduced_reparam.as_ref() {
        let r = reparam.reduced_cols();
        if let Some(block) = solved_fit.blocks.get_mut(1)
            && block.beta.len() == r
        {
            block.beta = reparam.recover_original_logslope_beta(&block.beta)?;
        }
        if let Some(state) = solved_fit.block_states.get_mut(1)
            && state.beta.len() == r
        {
            state.beta = reparam.recover_original_logslope_beta(&state.beta)?;
        }
    }
    // #905 GENERATED-REGRESSOR (Murphy–Topel) SEAM. When the conditional
    // location-scale gate fired, the slope fit above treated the calibrated
    // score `ζ = (z − m̂(C))/√v̂(C)` as KNOWN, so `solved_fit.beta_covariance()`
    // is the naive second-stage covariance `V_β^naive = H_β⁻¹` that ignores the
    // first-stage estimation error in `θ₁ = (mean_coeffs, var_coeffs)`. The
    // honest two-stage covariance is
    //   `V_β = V_β^naive + (H_β⁻¹ G) V₁ (H_β⁻¹ G)ᵀ`,  `G = ∂(score_β)/∂θ₁`.
    // The closed-form first-stage covariance `V₁` and the per-row chain-rule
    // sensitivity `∂ζ_i/∂θ₁` are computed and stored on the calibration at fit
    // time (see `LatentZConditionalCalibration::{theta1_covariance,
    // zeta_theta1_jacobian_row, generated_regressor_term}`), so the correction
    // is consumable wherever the slope information `G` (the per-row
    // `∂score_β/∂ζ_i` of the marginal/logslope blocks) is available.
    //
    // ASSEMBLY READY, ONE ENGINE QUANTITY OUTSTANDING. The full correction is
    // assembled by `LatentZConditionalCalibration::generated_regressor_correction`
    // (mod.rs): given the per-row reduced-frame slope-score sensitivity to the
    // calibrated score `s_i = ∂score_β,i/∂ζ_i` (an `n × p_β` matrix), it
    //   1. builds `J_zeta` row-by-row via `zeta_theta1_jacobian_row` (exact-zero
    //      on floored rows, so `G`'s support is the gate-fired rows),
    //   2. accumulates `G = Σ_i s_i ⊗ (∂ζ_i/∂θ₁)` (`p_β × dim θ₁`),
    //   3. forms `Vb·G = solved_fit.beta_covariance()·G` (the naive reduced-frame
    //      covariance IS `H_β⁻¹`, so `H_β⁻¹ G = Vb·G`), and
    //   4. returns `(Vb·G)·V₁·(Vb·G)ᵀ` (PSD ⇒ corrected slope SE strictly ≥
    //      naive whenever the gate fires).
    // So `V₁`, `∂ζ/∂θ₁`, the `Vb` frame, and the whole congruence are all
    // available HERE — the only quantity the seam still lacks is `s_i`.
    //
    // `s_i = ∂²ℓ_i/∂β∂ζ_i = J_iᵀ·(∂²ℓ_i/∂η_i∂ζ_i)` is the mixed `(β, ζ)` second
    // derivative of the row kernel contracted through the slope Jacobian `J_i`.
    // The conditional location-scale gate ALWAYS selects the rigid standard-normal
    // measure (`build_latent_measure_with_geometry` returns
    // `LatentMeasureKind::StandardNormal` for `ConditionalLocationScale`), so the
    // per-row kernel is the closed-form `rigid_standard_normal` tower
    // `η = q·c(g) + g·(s·ζ)`. The mixed 2-vector `∂²ℓ_i/∂(q,g)∂ζ_i` is read off the
    // SAME `Tower4` the value/grad/Hessian path uses (#932 row-jet machinery) by
    // seeding `ζ` as a third jet axis
    // (`rigid_standard_normal_mixed_z_sensitivity`); contracting it through the
    // marginal+logslope design rows (the `J_iᵀ` the row kernel exposes via
    // `jacobian_transpose_action`) yields `s_i` in the SAME reduced frame as
    // `covariance_conditional` (`rigid_standard_normal_score_zeta_sensitivity`).
    let (latent_z_rank_int_calibration, latent_z_conditional_calibration) =
        match latent_z_calibration {
            LatentMeasureCalibration::None => (None, None),
            LatentMeasureCalibration::RankInverseNormal(cal) => (Some(cal), None),
            LatentMeasureCalibration::ConditionalLocationScale(cal) => (None, Some(cal)),
        };
    // #905/#1028: apply the Murphy–Topel generated-regressor correction now that
    // `s_i` is available. `covariance_conditional` (Vb) and `covariance_corrected`
    // (Vp) are in the reduced logslope frame (`p_m + r`), exactly the frame
    // `s_i`'s reduced-logslope contraction lives in, so add the PSD term
    // `(Vb·G)·V₁·(Vb·G)ᵀ` to each. Applied only for the canonical (non-flex)
    // standard-normal kernel: the rigid tower carries no score_warp/link_dev
    // z-dependence, so when aux deviation blocks widen β beyond `p_m + r` the
    // correction's deviation columns are not yet derived and the term is skipped
    // (the conditional gate's intended kernel has no such blocks).
    if let Some(cal) = latent_z_conditional_calibration.as_ref()
        && let Some(vb) = solved_fit.covariance_conditional.clone()
    {
        let p_beta = vb.nrows();
        let marginal_dense = marginal_design
            .design
            .try_to_dense_arc("bms generated-regressor marginal design")?;
        let logslope_reduced = reduce_logslope_design(&logslope_design)?;
        let logslope_reduced_dense = logslope_reduced
            .design
            .try_to_dense_arc("bms generated-regressor reduced logslope design")?;
        let p_m = marginal_dense.ncols();
        let r = logslope_reduced_dense.ncols();
        if p_beta != vb.ncols() {
            return Err(format!(
                "bms generated-regressor: covariance_conditional must be square, got {}×{}",
                vb.nrows(),
                vb.ncols()
            ));
        }
        // Skip when aux deviation (score_warp / link_dev) blocks are present:
        // β is wider than the marginal+reduced-logslope frame the rigid kernel's
        // z-channel covers. Equality ⇒ the canonical non-flex gate kernel.
        if p_beta == p_m + r {
            let marginal_eta = &solved_fit.block_states[0].eta;
            let slope_eta = &solved_fit.block_states[1].eta;
            let probit_scale = probit_frailty_scale(final_sigma_cell.get());
            let s = rigid_standard_normal_score_zeta_sensitivity(
                &spec.base_link,
                marginal_eta,
                slope_eta,
                z.as_ref(),
                y.as_ref(),
                weights.as_ref(),
                probit_scale,
                marginal_dense.view(),
                logslope_reduced_dense.view(),
                p_beta,
            )?;
            // `generated_regressor_correction` re-derives `∂ζ_i/∂θ₁` via
            // `zeta_theta1_jacobian_row(z_i, a_row)`, which expects the RAW
            // normalized latent score `z_i` (it recomputes `ζ_i = (z_i − m)/√v`
            // internally), and conditions on the marginal-index span
            // `a(C_i)` = the RAW marginal design rows (the basis the gate was fit
            // on). Feed `spec.z` (the standardized raw score, NOT the calibrated
            // ζ the kernel consumed) and the raw marginal dense design.
            let correction = cal.generated_regressor_correction(
                s.view(),
                spec.z.view(),
                marginal_dense.view(),
                vb.view(),
            )?;
            if let Some(cov) = solved_fit.covariance_conditional.as_mut() {
                *cov = &*cov + &correction;
            }
            if let Some(cov) = solved_fit.covariance_corrected.as_mut() {
                *cov = &*cov + &correction;
            }
            log::info!(
                "[BMS latent-z] Murphy–Topel generated-regressor SE correction applied: \
                 p_beta={p_beta} theta1_dim={} max_diag_inflation={:.3e}",
                cal.theta1_dim(),
                (0..p_beta)
                    .map(|i| correction[[i, i]])
                    .fold(0.0_f64, f64::max),
            );
        } else {
            log::info!(
                "[BMS latent-z] Murphy–Topel generated-regressor SE correction skipped: \
                 aux deviation blocks present (p_beta={p_beta} > marginal({p_m})+logslope({r})); \
                 rigid-kernel z-channel does not yet cover score_warp/link_dev deviations"
            );
        }
    }
    // #461: PREDICT SEAM — when the Stage-1 influence absorber is active
    // (spec.score_influence_jacobian.is_some()), `fit.block_states[0].beta` is
    // the WIDENED marginal coefficient `[β_m; γ]` (length p_m + p₁), but
    // `marginal_design` below is the RAW term-collection design (p_m columns):
    // the absorbed influence columns Z̃_infl are a TRAINING-only leakage
    // absorber and do NOT exist at predict rows (no Stage-1 fold there). The
    // orthogonalized β̂_m is a property of the training fit, so prediction must
    // use ONLY the first p_m entries of block_states[0].beta against this raw
    // marginal_design and DROP the trailing γ. The model-payload / predict
    // builder (src/main.rs run_fit_bernoulli_marginal_slope → inference) owns
    // that truncation; it must record p_m (= marginal_design.design.ncols())
    // and slice the persisted marginal β to it. Survival mirrors this seam.
    Ok(BernoulliMarginalSlopeFitResult {
        fit: solved_fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs.remove(0),
        logslope_design: designs.remove(0),
        baseline_marginal: baseline.0,
        baseline_logslope: baseline.1,
        z_normalization,
        latent_measure,
        score_warp_runtime,
        link_dev_runtime,
        gaussian_frailty_sd: final_sigma_cell.get(),
        cross_block_warnings,
        latent_z_rank_int_calibration,
        latent_z_conditional_calibration,
    })
}
