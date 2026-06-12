//! Measure-jet spline smooth: multiscale local-jet-residual energy of the
//! empirical measure (V0, center-quadratured).
//!
//! The term penalizes, at every quadrature point and every scale, the failure
//! of `f` to be locally affine *in the measure*:
//!
//! ```text
//!   Q = Σ_ℓ  w_ℓ · Σ_i  mass_i · q_i(ε_ℓ)^(1−2α) · R_{i,ℓ},
//!   w_ℓ = log_step · ε_ℓ^(−2s),
//! ```
//!
//! where `R_{i,ℓ}` is the residual quadratic form of the τ-ridged, weighted
//! local affine fit at center `i` and scale `ε_ℓ`: kernel weights
//! `w_j = mass_j · exp(−d_ij²/(2ε_ℓ²))`, kernel mass `q_i = Σ_j w_j`, and the
//! penalized fit `min_b ‖Cv − Φ̃b‖²_W + τ·q·‖b‖²` over weighted-centered
//! values `Cv = v − (uᵀv)·1` (`u = w/q`) and weighted-centered scaled
//! features `Φ̃` (rows `(c_j − c_i)/ε`, column means removed under `u`).
//!
//! # Contracts (each is load-bearing; tests pin them)
//!
//! - **Exact constant annihilation.** The constant is removed by the weighted
//!   mean projection `C`, never ridged: `Q·1 = 0` to machine precision at
//!   every scale, so the penalty carries NO mass term and the fit has no
//!   prior mean to revert to. This is the no-mean-reversion contract of the
//!   measure-jet design; ridging the constant would silently reintroduce
//!   mean reversion.
//! - **Ridge as noise floor / rank adaptation.** Only the slope block is
//!   ridged (`G + τ·I` on the dimensionless local Gram `G = Φ̃ᵀWΦ̃/q`). On a
//!   1-D filament in ambient dimension d the local Gram has numerical rank
//!   ~1: the resolved tangent slope is absorbed (not penalized) while slope
//!   variation in unresolved/below-noise directions is treated as roughness.
//!   Rank transitions (filament thickening into a sheet) are smooth
//!   crossovers governed by τ, not discrete decisions.
//! - **Mellin band.** Scales form a geometric grid from the center-spacing
//!   floor to the half-diameter; `w_ℓ = log_step · ε_ℓ^(−2s)` is the
//!   quadrature of `∫ ε^(−2s) (·) dε/ε`, giving a continuous smoothness
//!   order `s ∈ (0, 2)` with no preferred internal scale (default 1.5). On a
//!   flat stratum the symbol of the band-limited form is `≍ |ξ|^{2s}`
//!   (substitute `t = ε|ξ|` in the Mellin integral) — fractional Duchon on
//!   the web with learned order.
//! - **Density normalization.** The outer quadrature weight
//!   `mass_i · q_i^(1−2α)` realizes `dμ(x)/q_ε(x)^(2α−1)`; `α = 1` (default)
//!   removes the sampling-density dependence of the limiting energy, `α = 0`
//!   penalizes density-weighted roughness.
//! - **Frozen-quadrature replay.** The penalty depends on the FIT data
//!   through center masses and the realized band. The freeze step persists
//!   both ([`MeasureJetFrozenQuadrature`]) so predict-time (and future
//!   per-ψ-trial) rebuilds replay the exact fit-time penalty instead of
//!   recomputing it from predict rows.
//! - **Single assembly source.** Every quadratic form this module emits —
//!   the energy, its (s, α) jets, the per-scale spectrum — is produced by
//!   ONE workhorse ([`assemble_weighted_forms`]) that walks the local
//!   residual blocks exactly once per request and differs only in the
//!   scalar weights applied per block. Criterion value and criterion
//!   derivatives cannot drift apart (the objective↔gradient desync class is
//!   structurally excluded).
//!
//! # ψ-differentiability contract (what the ψ-channel stage consumes)
//!
//! Mirroring the constant-curvature κ-contract (#944): centers, masses, the
//! band, and the representer range ℓ are all deliberately hyperparameter-
//! FIXED at build time. Consequences, available TODAY from this module:
//!
//! - **Design drift is identically zero** for every measure-jet ψ
//!   coordinate: the Gaussian representer design depends on none of
//!   (s, α, τ), so `∂X/∂ψ ≡ 0` and the channels are penalty-only
//!   (`is_penalty_like` auto-derives true in the outer engine's
//!   `DirectionalHyperParam`).
//! - **Exact (s, α) penalty jets are shipped**:
//!   [`measure_jet_energy_form_with_jets`] returns `∂Q/∂s`, `∂²Q/∂s²`,
//!   `∂Q/∂α`, `∂²Q/∂α²`, `∂²Q/∂s∂α` in closed form — both dials enter only
//!   through the per-block log-weights (`∂ln w/∂s = −2 ln ε`,
//!   `∂ln w/∂α = −2 ln q`), so the jets are reweighted re-scatters of the
//!   SAME residual blocks, FD-gated in this module's tests.
//! - **Not yet shipped:** the τ-channel (a resolvent derivative of the
//!   ridged local Gram — analytic, heavier plumbing) and the outer-engine
//!   wiring itself (the `normalize_penaltywith_psi_derivatives` seam), which
//!   follow the Matérn iso-κ template once the #901 seam is green.
//!
//! # Cost shape (and the upgrade ladder above it)
//!
//! The outer sum is coarsened per scale to a deterministic ε/2-net (the
//! outer Riemann sum needs resolution ε, not the center-spacing floor), so
//! the band totals ~O(m²·d) instead of O(L·m³) — the V0 realization of the
//! pyramid principle that each scale interacts at its own level. The known
//! upgrade ladder (jet-MRA synthesis coordinates with a diagonal prior;
//! Hermite/fast-Gauss moment fusion at n-scale; screening-based sparse
//! factorization) replaces this module's REALIZATION without touching the
//! estimand: the analysis-form energy above is the definition, everything
//! else is certified quadrature of it.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use faer::Side;

use crate::linalg::faer_ndarray::FaerEigh;

use super::{
    BasisBuildResult, BasisError, BasisMetadata, CenterStrategy, PenaltyCandidate, PenaltySource,
    filter_active_penalty_candidates_with_ops, normalize_penalty, select_centers_by_strategy,
    weighted_coefficient_sum_to_zero_transform,
};

/// Truncation radius of the Gaussian profile in units of the scale ε: weights
/// beyond `3ε` are below `e^{-4.5} ≈ 1.1e-2` of the peak and are dropped from
/// the local fit, bounding each local residual form to the ε-neighborhood.
const MEASURE_JET_PROFILE_CUTOFF: f64 = 3.0;

/// Relative eigenvalue threshold for the unridged (`tau0 == 0`) pseudo-inverse
/// of the local slope Gram: directions with `λ ≤ threshold · λ_max` are
/// treated as unresolved and excluded from the affine fit.
const MEASURE_JET_PSEUDOINVERSE_RTOL: f64 = 1e-10;

/// Default continuous smoothness order `s` realized by the `0.0` auto
/// sentinel. Sits mid-band in the admissible `(0, 2)` for the affine-jet
/// (r = 2) energy: rough enough to stay pointwise-defined on filaments and
/// sheets (`s > p/2` for intrinsic `p ≤ 2`), smooth enough to bridge gaps
/// with attested trends.
const MEASURE_JET_DEFAULT_ORDER_S: f64 = 1.5;

/// Auto-band scale-count clamp: at least 3 octave-ish nodes so the energy is
/// genuinely multiscale, at most 8 so degenerate spacing cannot explode the
/// build.
const MEASURE_JET_MIN_AUTO_SCALES: usize = 3;
const MEASURE_JET_MAX_AUTO_SCALES: usize = 8;

/// Representer-range multiple of the median nearest-center spacing used by
/// the `0.0` auto sentinel: ×2 so adjacent representers overlap and thin
/// sampling gaps stay inside the basis span.
const MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR: f64 = 2.0;

/// Memory budget (in f64 entries) above which the multi-form assembly stops
/// parallelizing over scales: parallel scale partials cost
/// `L · n_forms · m²` doubles; past this budget the scales run sequentially
/// (same numbers — the per-scale loop and the ordered sum are deterministic
/// either way).
const MEASURE_JET_PARALLEL_FORM_BUDGET_DOUBLES: usize = 1 << 26;

/// Realized-design identifiability policy for the measure-jet smooth.
/// Mirrors [`super::ConstantCurvatureIdentifiability`] (#532): the fit-time
/// center-space sum-to-zero `z` gets the parametric orthogonalization
/// composed onto it by the global identifiability pipeline, and the composed
/// transform is frozen so predict-time (and per-ψ-trial) rebuilds replay it
/// verbatim instead of recomputing `z` from the centers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MeasureJetIdentifiability {
    /// Fit-time default: uniform-weight coefficient sum-to-zero over the
    /// centers (`Σ_j α_j = 0`), then global parametric residualization.
    #[default]
    CenterSumToZero,
    /// Predict-time replay: the frozen composed transform captured at fit
    /// time. `transform.nrows()` equals the number of centers.
    FrozenTransform { transform: Array2<f64> },
}

/// Fit-time quadrature of the empirical measure (center masses + realized
/// scale band), frozen onto the spec so predict-time rebuilds replay the
/// exact fit-time penalty. Recomputing either from predict rows would
/// silently change the penalty the coefficients were estimated under.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureJetFrozenQuadrature {
    /// Per-center masses `m_i` (nearest-center fractions of the FIT rows).
    pub masses: Array1<f64>,
    /// Realized geometric scale band `ε_0 < … < ε_{L−1}`.
    pub eps_band: Vec<f64>,
}

/// Measure-jet smooth configuration (`mjs(x0, …, xd)`).
///
/// The feature columns are ambient coordinates of data concentrated near an
/// unknown low-dimensional (possibly stratified) set; the term learns the
/// geometry from the empirical measure itself — centers as quadrature nodes,
/// masses as μ-weights, local jet residuals as the roughness carrier — with
/// no graph, mesh, or neighbor-set inside the statistical object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureJetBasisSpec {
    /// Center/knot selection strategy (deterministic; quadrature of μ).
    pub center_strategy: CenterStrategy,
    /// Continuous smoothness order `s ∈ (0, 2)`; `0.0` sentinel = auto
    /// ([`MEASURE_JET_DEFAULT_ORDER_S`]).
    pub order_s: f64,
    /// Density-normalization exponent α (outer weight `q^{1−2α}`).
    pub alpha: f64,
    /// Dimensionless jet-ridge floor τ on the local slope Gram. `0.0` selects
    /// the exact pseudo-inverse (test/oracle mode; the fitted default keeps a
    /// positive floor as the noise-floor regularizer).
    pub tau0: f64,
    /// Number of scale nodes; `0` sentinel = auto dyadic band.
    pub num_scales: usize,
    /// Representer (Gaussian RBF) range ℓ; `0.0` sentinel = auto
    /// (median nearest-center spacing × [`MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR`]).
    pub length_scale: f64,
    /// Add the ridge-like shrinkage penalty alongside the jet-energy penalty.
    pub double_penalty: bool,
    /// Realized-design identifiability policy (see type docs).
    #[serde(default)]
    pub identifiability: MeasureJetIdentifiability,
    /// Fit-time quadrature replay (see type docs). `None` at fit time;
    /// `Some` on the frozen predict/ψ-trial path.
    #[serde(default)]
    pub frozen_quadrature: Option<MeasureJetFrozenQuadrature>,
}

impl Default for MeasureJetBasisSpec {
    fn default() -> Self {
        Self {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 50 },
            order_s: 0.0,
            alpha: 1.0,
            tau0: 1e-3,
            num_scales: 0,
            length_scale: 0.0,
            double_penalty: true,
            identifiability: MeasureJetIdentifiability::CenterSumToZero,
            frozen_quadrature: None,
        }
    }
}

/// Realized geometric scale band: `eps` ascending, `log_step` the constant
/// log-spacing `ln(eps[ℓ+1]/eps[ℓ])` used as the Mellin quadrature weight.
pub struct MeasureJetBand {
    pub eps: Vec<f64>,
    pub log_step: f64,
}

/// The energy and its exact hyperparameter jets in all three dials —
/// `(s, α, ψ_τ = ln τ)` — ten symmetric m×m forms scattered from the SAME
/// local residual blocks in one pass. `s` and `α` enter only through
/// per-block log-weights; `ln τ` enters through the resolvent of the ridged
/// local Gram (`∂R/∂τ = B·M²·Bᵀ/q`, `∂²R/∂τ² = −2·B·M³·Bᵀ/q` with
/// `M = (G + τI)⁻¹`), also closed-form. All ten are exact by construction
/// (FD-gated in this module's tests), and the ψ-channel consumes them with
/// zero design drift.
pub struct MeasureJetEnergyJets {
    pub q: Array2<f64>,
    pub dq_ds: Array2<f64>,
    pub d2q_ds2: Array2<f64>,
    pub dq_dalpha: Array2<f64>,
    pub d2q_dalpha2: Array2<f64>,
    pub d2q_ds_dalpha: Array2<f64>,
    pub dq_dlogtau: Array2<f64>,
    pub d2q_dlogtau2: Array2<f64>,
    pub d2q_ds_dlogtau: Array2<f64>,
    pub d2q_dalpha_dlogtau: Array2<f64>,
}

fn validate_finite_points(points: ArrayView2<'_, f64>, what: &str) -> Result<(), BasisError> {
    for (i, row) in points.outer_iter().enumerate() {
        if row.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_basis!("measure-jet {what} row {i} has a non-finite coordinate");
        }
    }
    Ok(())
}

/// Squared Euclidean distances between all center pairs (m × m, symmetric).
fn center_pairwise_dist2(centers: ArrayView2<'_, f64>) -> Array2<f64> {
    let m = centers.nrows();
    let mut d2 = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in (i + 1)..m {
            let mut s = 0.0_f64;
            for k in 0..centers.ncols() {
                let dlt = centers[(i, k)] - centers[(j, k)];
                s += dlt * dlt;
            }
            d2[(i, j)] = s;
            d2[(j, i)] = s;
        }
    }
    d2
}

/// Median nearest-OTHER-center distance — the resolution floor of the center
/// quadrature, used for the band floor and the auto representer range.
fn median_nearest_center_spacing(dist2: &Array2<f64>) -> Result<f64, BasisError> {
    let m = dist2.nrows();
    if m < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: m });
    }
    let mut nearest: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        let mut best = f64::INFINITY;
        for j in 0..m {
            if j != i && dist2[(i, j)] < best {
                best = dist2[(i, j)];
            }
        }
        nearest.push(best.sqrt());
    }
    nearest.sort_by(|a, b| a.partial_cmp(b).expect("finite center spacings"));
    let median = nearest[nearest.len() / 2];
    if !(median.is_finite() && median > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet centers are degenerate (median nearest-center spacing = {median}); \
             duplicate centers cannot carry a scale band"
        );
    }
    Ok(median)
}

/// Build the realized geometric scale band from the center set: floor at the
/// median nearest-center spacing (below it the quadrature resolves nothing),
/// ceiling at half the bounding-box diagonal (above it every local fit is the
/// global affine fit). `num_scales == 0` requests the auto count
/// `clamp(⌈log2(ε_max/ε_min)⌉ + 1, 3, 8)`; a degenerate band (ceiling ≤
/// floor) collapses to the single floor scale with `log_step = ln 2`.
pub fn measure_jet_band(
    centers: ArrayView2<'_, f64>,
    num_scales: usize,
) -> Result<MeasureJetBand, BasisError> {
    validate_finite_points(centers, "centers")?;
    let dist2 = center_pairwise_dist2(centers);
    let eps_min = median_nearest_center_spacing(&dist2)?;
    // Half the bounding-box diagonal: a cheap, deterministic diameter proxy.
    let d = centers.ncols();
    let mut diag2 = 0.0_f64;
    for k in 0..d {
        let col = centers.column(k);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in col.iter() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        diag2 += (hi - lo) * (hi - lo);
    }
    let eps_max = 0.5 * diag2.sqrt();
    if !(eps_max.is_finite() && eps_max > eps_min) {
        return Ok(MeasureJetBand {
            eps: vec![eps_min],
            log_step: std::f64::consts::LN_2,
        });
    }
    let auto = ((eps_max / eps_min).log2().ceil() as usize + 1)
        .clamp(MEASURE_JET_MIN_AUTO_SCALES, MEASURE_JET_MAX_AUTO_SCALES);
    let count = if num_scales == 0 { auto } else { num_scales };
    if count == 1 {
        return Ok(MeasureJetBand {
            eps: vec![eps_min],
            log_step: std::f64::consts::LN_2,
        });
    }
    let ratio = (eps_max / eps_min).powf(1.0 / (count as f64 - 1.0));
    let mut eps = Vec::with_capacity(count);
    let mut e = eps_min;
    for _ in 0..count {
        eps.push(e);
        e *= ratio;
    }
    Ok(MeasureJetBand {
        eps,
        log_step: ratio.ln(),
    })
}

/// First-moment-exact quadrature of the empirical measure on the cell
/// partition induced by the seed centers: nearest-center assignment
/// (deterministic tie-break: lowest center index) yields per-cell masses,
/// and each cell's quadrature NODE is its mass-weighted barycenter — so the
/// lumped measure matches μ's zeroth AND first moments per cell, one order
/// better than placing the mass at the seed point. Empty cells keep their
/// seed coordinates with zero mass (the assembly skips them; their
/// representer columns remain valid).
pub fn measure_jet_quadrature_nodes(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "measure-jet mass assignment dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    validate_finite_points(data, "data")?;
    validate_finite_points(centers, "centers")?;
    let n = data.nrows();
    let m = centers.nrows();
    let d = centers.ncols();
    if n == 0 || m == 0 {
        crate::bail_invalid_basis!("measure-jet mass assignment needs nonempty data and centers");
    }
    let assignments: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = data.row(i);
            let mut best_j = 0usize;
            let mut best = f64::INFINITY;
            for (j, c) in centers.outer_iter().enumerate() {
                let mut s = 0.0_f64;
                for k in 0..row.len() {
                    let dlt = row[k] - c[k];
                    s += dlt * dlt;
                }
                if s < best {
                    best = s;
                    best_j = j;
                }
            }
            best_j
        })
        .collect();
    let mut masses = Array1::<f64>::zeros(m);
    let mut nodes = centers.to_owned();
    let mut sums = Array2::<f64>::zeros((m, d));
    let unit = 1.0 / n as f64;
    for (i, &j) in assignments.iter().enumerate() {
        masses[j] += unit;
        for k in 0..d {
            sums[(j, k)] += data[(i, k)];
        }
    }
    for j in 0..m {
        let count = masses[j] * n as f64;
        if count > 0.0 {
            for k in 0..d {
                nodes[(j, k)] = sums[(j, k)] / count;
            }
        }
    }
    Ok((nodes, masses))
}

/// Per-center masses of the empirical measure (the zeroth-moment half of
/// [`measure_jet_quadrature_nodes`]; single assignment source).
pub fn measure_jet_center_masses(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, BasisError> {
    measure_jet_quadrature_nodes(data, centers).map(|(_, masses)| masses)
}

/// THE single assembly source: walk every (scale, outer-net center) local
/// residual block exactly once and scatter it into `n_forms` accumulators
/// with caller-chosen scalar weights. The energy, its (s, α) jets, and the
/// per-scale spectrum are all this routine with different weight closures,
/// so a value/derivative desync is structurally impossible.
///
/// Per block the closure receives `(scale_idx, eps, q, base)` where `base`
/// is the fully-assembled outer weight
/// `log_step · ε^(−2s) · net_mass_i · q^(1−2α)`, and writes, per requested
/// form, one weight triple `[w_R, w_2, w_3]` against the three block
/// channels `R = CᵀWC − B·M·Bᵀ/q`, `B·M²·Bᵀ/q`, and `B·M³·Bᵀ/q`
/// (`M = (G + τI)⁻¹`) — the resolvent powers carrying the exact ln-τ jets.
/// Channels 2 and 3 are only computed when `channels` requests them.
///
/// The outer sum over centers is coarsened per scale to a deterministic
/// ε/2-net with nearest-member mass aggregation (the outer Riemann sum needs
/// resolution ε, not the center-spacing floor), so each scale's cost sits at
/// its own level and the band totals ~O(m²·d) instead of O(L·m³). The inner
/// (local-fit) quadrature always uses the full center set, so the local
/// residual identities (exact constant annihilation, PSD) are untouched.
fn assemble_weighted_forms<F>(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
    n_forms: usize,
    channels: usize,
    weights: &F,
) -> Result<Vec<Array2<f64>>, BasisError>
where
    F: Fn(usize, f64, f64, f64, &mut [[f64; 3]]) + Sync,
{
    let m = centers.nrows();
    let d = centers.ncols();
    if n_forms == 0 || !(1..=3).contains(&channels) {
        crate::bail_invalid_basis!(
            "measure-jet assembly needs at least one output form and 1..=3 block channels"
        );
    }
    if masses.len() != m {
        crate::bail_dim_basis!(
            "measure-jet energy mass/center mismatch: {} masses for {} centers",
            masses.len(),
            m
        );
    }
    if band.eps.is_empty() || band.eps.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        crate::bail_invalid_basis!("measure-jet energy needs a nonempty positive scale band");
    }
    if !(order_s.is_finite() && order_s > 0.0 && order_s < 2.0) {
        crate::bail_invalid_basis!(
            "measure-jet order s must lie in (0, 2) for the affine-jet energy; got {order_s}"
        );
    }
    if !(alpha.is_finite() && tau0.is_finite() && tau0 >= 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet energy needs finite alpha and finite tau0 >= 0; got alpha={alpha}, tau0={tau0}"
        );
    }
    if masses.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        crate::bail_invalid_basis!("measure-jet energy needs finite nonnegative center masses");
    }
    let dist2 = center_pairwise_dist2(centers);

    // One block of `n_forms` m×m accumulators per scale. Each scale's center
    // loop is sequential and the cross-scale sum below runs in band order,
    // so the result is bit-deterministic whether or not the scales
    // themselves run in parallel.
    let assemble_scale = |scale_idx: usize, eps: f64| -> Result<Vec<Array2<f64>>, BasisError> {
        let mut out: Vec<Array2<f64>> =
            (0..n_forms).map(|_| Array2::<f64>::zeros((m, m))).collect();
        let cutoff2 = (MEASURE_JET_PROFILE_CUTOFF * eps) * (MEASURE_JET_PROFILE_CUTOFF * eps);
        let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
        let scale_weight = band.log_step * eps.powf(-2.0 * order_s);
        // Outer-quadrature coarsening: greedy ε/2-net over the centers in
        // fixed index order (deterministic), with every center's mass
        // aggregated to its nearest net member (lowest-index tie break).
        let net_radius2 = 0.25 * eps * eps;
        let mut outer: Vec<usize> = Vec::new();
        for i in 0..m {
            if masses[i] <= 0.0 {
                continue;
            }
            let covered = outer.iter().any(|&o| dist2[(i, o)] <= net_radius2);
            if !covered {
                outer.push(i);
            }
        }
        let mut net_mass = vec![0.0_f64; m];
        for i in 0..m {
            if masses[i] <= 0.0 {
                continue;
            }
            let mut best = f64::INFINITY;
            let mut best_o = usize::MAX;
            for &o in &outer {
                if dist2[(i, o)] < best {
                    best = dist2[(i, o)];
                    best_o = o;
                }
            }
            if best_o != usize::MAX {
                net_mass[best_o] += masses[i];
            }
        }
        let mut wbuf = vec![[0.0_f64; 3]; n_forms];
        for &i in &outer {
            // Local neighbor set (always includes i itself).
            let mut idx: Vec<usize> = Vec::new();
            for j in 0..m {
                if dist2[(i, j)] <= cutoff2 {
                    idx.push(j);
                }
            }
            let ml = idx.len();
            // Kernel weights and mass.
            let mut w = Array1::<f64>::zeros(ml);
            let mut q = 0.0_f64;
            for (a, &j) in idx.iter().enumerate() {
                let wj = masses[j] * (-dist2[(i, j)] * inv_two_eps2).exp();
                w[a] = wj;
                q += wj;
            }
            if !(q > 0.0) {
                continue;
            }
            // Scaled local features Φ (ml × d) and weighted column means a.
            let mut phi = Array2::<f64>::zeros((ml, d));
            for (a, &j) in idx.iter().enumerate() {
                for k in 0..d {
                    phi[(a, k)] = (centers[(j, k)] - centers[(i, k)]) / eps;
                }
            }
            let a_mean = phi.t().dot(&w) / q;
            // B = WΦ − w·aᵀ and G = (ΦᵀWΦ)/q − a·aᵀ.
            let mut wphi = phi.clone();
            for (a, mut row) in wphi.outer_iter_mut().enumerate() {
                row.mapv_inplace(|v| v * w[a]);
            }
            let mut b = wphi.clone();
            for (a, mut row) in b.outer_iter_mut().enumerate() {
                for k in 0..d {
                    row[k] -= w[a] * a_mean[k];
                }
            }
            let mut g = phi.t().dot(&wphi);
            g.mapv_inplace(|v| v / q);
            for r in 0..d {
                for c in 0..d {
                    g[(r, c)] -= a_mean[r] * a_mean[c];
                }
            }
            // (G + τI)⁻¹ via symmetric eigendecomposition; pseudo-inverse
            // with a relative floor in the τ = 0 oracle mode.
            let (evals, evecs) = g.eigh(Side::Lower).map_err(|e| {
                BasisError::InvalidInput(format!(
                    "measure-jet local Gram eigendecomposition failed at center {i}: {e}"
                ))
            })?;
            let lam_max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let mut inv_diag = Array1::<f64>::zeros(d);
            for k in 0..d {
                let lam = evals[k].max(0.0);
                inv_diag[k] = if tau0 > 0.0 {
                    1.0 / (lam + tau0)
                } else if lam > MEASURE_JET_PSEUDOINVERSE_RTOL * lam_max {
                    1.0 / lam
                } else {
                    0.0
                };
            }
            // M = V·diag(inv)·Vᵀ, then the projected block B·M·Bᵀ/q.
            let mut vm = evecs.clone();
            for (k, mut col) in vm.axis_iter_mut(Axis(1)).enumerate() {
                col.mapv_inplace(|v| v * inv_diag[k]);
            }
            let m_inv = vm.dot(&evecs.t());
            let bm = b.dot(&m_inv);
            // Resolvent powers for the ln-τ jets, computed only on request:
            // B·M²·Bᵀ/q (first τ-derivative channel) and B·M³·Bᵀ/q (second).
            let bm2 = if channels >= 2 {
                let mut vm2 = evecs.clone();
                for (k, mut col) in vm2.axis_iter_mut(Axis(1)).enumerate() {
                    col.mapv_inplace(|v| v * inv_diag[k] * inv_diag[k]);
                }
                Some(b.dot(&vm2.dot(&evecs.t())))
            } else {
                None
            };
            let bm3 = if channels >= 3 {
                let mut vm3 = evecs.clone();
                for (k, mut col) in vm3.axis_iter_mut(Axis(1)).enumerate() {
                    col.mapv_inplace(|v| v * inv_diag[k] * inv_diag[k] * inv_diag[k]);
                }
                Some(b.dot(&vm3.dot(&evecs.t())))
            } else {
                None
            };
            let base = scale_weight * net_mass[i] * q.powf(1.0 - 2.0 * alpha);
            weights(scale_idx, eps, q, base, &mut wbuf);
            // Scatter-add Σ_k wbuf[k]·[R | B·M²·Bᵀ/q | B·M³·Bᵀ/q] into each form.
            for (a, &ja) in idx.iter().enumerate() {
                let bma = bm.row(a);
                for (c, &jc) in idx.iter().enumerate() {
                    let b_c = b.row(c);
                    let mut val_r = -w[a] * w[c] / q - bma.dot(&b_c) / q;
                    if a == c {
                        val_r += w[a];
                    }
                    let val_2 = bm2.as_ref().map_or(0.0, |m2| m2.row(a).dot(&b_c) / q);
                    let val_3 = bm3.as_ref().map_or(0.0, |m3| m3.row(a).dot(&b_c) / q);
                    for (k, out_k) in out.iter_mut().enumerate() {
                        let wk = wbuf[k];
                        out_k[(ja, jc)] += wk[0] * val_r + wk[1] * val_2 + wk[2] * val_3;
                    }
                }
            }
        }
        Ok(out)
    };

    let n_scales = band.eps.len();
    let parallel_ok =
        m.saturating_mul(m)
            .saturating_mul(n_scales)
            .saturating_mul(n_forms)
            <= MEASURE_JET_PARALLEL_FORM_BUDGET_DOUBLES;
    let per_scale: Vec<Vec<Array2<f64>>> = if parallel_ok {
        band.eps
            .par_iter()
            .enumerate()
            .map(|(scale_idx, &eps)| assemble_scale(scale_idx, eps))
            .collect::<Result<Vec<_>, BasisError>>()?
    } else {
        band.eps
            .iter()
            .enumerate()
            .map(|(scale_idx, &eps)| assemble_scale(scale_idx, eps))
            .collect::<Result<Vec<_>, BasisError>>()?
    };

    let mut totals: Vec<Array2<f64>> = (0..n_forms).map(|_| Array2::<f64>::zeros((m, m))).collect();
    for scale_forms in per_scale {
        for (total, part) in totals.iter_mut().zip(scale_forms) {
            *total += &part;
        }
    }
    // Numerical symmetrization (every analytic form here is symmetric).
    Ok(totals
        .into_iter()
        .map(|t| (&t + &t.t()) * 0.5)
        .collect())
}

/// The multiscale jet-residual energy `Q` (m × m, symmetric PSD) on the
/// center set. See the module docs for the formula and contracts; the local
/// residual form is assembled through the closed-form identities
///
/// ```text
///   CᵀWC          = W − w·wᵀ/q,
///   B = CᵀWΦ̃     = WΦ − w·aᵀ          (a = Φᵀw/q),
///   G = Φ̃ᵀWΦ̃/q  = (ΦᵀWΦ)/q − a·aᵀ,
///   R_loc         = CᵀWC − B·(G + τI)⁻¹·Bᵀ/q,
/// ```
///
/// with `(G + τI)⁻¹` realized through the symmetric eigendecomposition
/// (pseudo-inverse with relative threshold when `τ = 0`). One walk of
/// [`assemble_weighted_forms`] with the unit weight.
pub fn measure_jet_energy_form(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
) -> Result<Array2<f64>, BasisError> {
    let mut forms = assemble_weighted_forms(
        centers,
        masses,
        band,
        order_s,
        alpha,
        tau0,
        1,
        1,
        &|_, _, _, base, out: &mut [[f64; 3]]| out[0] = [base, 0.0, 0.0],
    )?;
    Ok(forms.swap_remove(0))
}

/// The energy together with its exact first and second jets in all three
/// dials `(s, α, ψ_τ = ln τ)` — the complete measure-jet ψ-channel
/// feedstock. With `g_s = −2 ln ε`, `g_α = −2 ln q`, `M = (G + τI)⁻¹`:
///
/// ```text
///   ∂Q/∂s   = Σ g_s·w·R,        ∂²Q/∂s²   = Σ g_s²·w·R,
///   ∂Q/∂α   = Σ g_α·w·R,        ∂²Q/∂α²   = Σ g_α²·w·R,
///   ∂²Q/∂s∂α = Σ g_s·g_α·w·R,
///   ∂Q/∂ψ_τ  = Σ w·τ·BM²Bᵀ/q,
///   ∂²Q/∂ψ_τ² = Σ w·(τ·BM²Bᵀ − 2τ²·BM³Bᵀ)/q,
///   ∂²Q/∂s∂ψ_τ = Σ g_s·w·τ·BM²Bᵀ/q,   ∂²Q/∂α∂ψ_τ = Σ g_α·w·τ·BM²Bᵀ/q,
/// ```
///
/// all scattered from the SAME local blocks as `Q` in one pass (no second
/// assembly that could drift). FD-gated in this module's tests. Requires
/// `tau0 > 0` (the ln-τ channel is undefined in the τ = 0 pseudo-inverse
/// oracle mode).
pub fn measure_jet_energy_form_with_jets(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
) -> Result<MeasureJetEnergyJets, BasisError> {
    if !(tau0.is_finite() && tau0 > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet jets need tau0 > 0 (the ln-τ channel is undefined at the τ = 0 \
             pseudo-inverse oracle mode); got {tau0}"
        );
    }
    let t = tau0;
    let mut forms = assemble_weighted_forms(
        centers,
        masses,
        band,
        order_s,
        alpha,
        tau0,
        10,
        3,
        &|_, eps: f64, q: f64, base: f64, out: &mut [[f64; 3]]| {
            let gs = -2.0 * eps.ln();
            let ga = -2.0 * q.max(f64::MIN_POSITIVE).ln();
            out[0] = [base, 0.0, 0.0];
            out[1] = [gs * base, 0.0, 0.0];
            out[2] = [gs * gs * base, 0.0, 0.0];
            out[3] = [ga * base, 0.0, 0.0];
            out[4] = [ga * ga * base, 0.0, 0.0];
            out[5] = [gs * ga * base, 0.0, 0.0];
            out[6] = [0.0, t * base, 0.0];
            out[7] = [0.0, t * base, -2.0 * t * t * base];
            out[8] = [0.0, gs * t * base, 0.0];
            out[9] = [0.0, ga * t * base, 0.0];
        },
    )?;
    let d2q_dalpha_dlogtau = forms.pop().expect("ten assembled forms");
    let d2q_ds_dlogtau = forms.pop().expect("ten assembled forms");
    let d2q_dlogtau2 = forms.pop().expect("ten assembled forms");
    let dq_dlogtau = forms.pop().expect("ten assembled forms");
    let d2q_ds_dalpha = forms.pop().expect("ten assembled forms");
    let d2q_dalpha2 = forms.pop().expect("ten assembled forms");
    let dq_dalpha = forms.pop().expect("ten assembled forms");
    let d2q_ds2 = forms.pop().expect("ten assembled forms");
    let dq_ds = forms.pop().expect("ten assembled forms");
    let q = forms.pop().expect("ten assembled forms");
    Ok(MeasureJetEnergyJets {
        q,
        dq_ds,
        d2q_ds2,
        dq_dalpha,
        d2q_dalpha2,
        d2q_ds_dalpha,
        dq_dlogtau,
        d2q_dlogtau2,
        d2q_ds_dlogtau,
        d2q_dalpha_dlogtau,
    })
}

/// Per-scale energy decomposition of center values `v`: element ℓ is
/// `vᵀ Q_ℓ v`, the detail energy charged at scale `ε_ℓ`. Sums exactly to
/// `vᵀQv` (same blocks, one-hot weights) and doubles as the scale spectrum
/// diagnostic of the fitted intensity field — where along the band the
/// signal lives, and the analytic carrier of `∂/∂s` reweightings.
pub fn measure_jet_scale_spectrum(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
    values: ArrayView1<'_, f64>,
) -> Result<Vec<f64>, BasisError> {
    if values.len() != centers.nrows() {
        crate::bail_dim_basis!(
            "measure-jet scale spectrum needs one value per center: {} values for {} centers",
            values.len(),
            centers.nrows()
        );
    }
    let n_scales = band.eps.len();
    let forms = assemble_weighted_forms(
        centers,
        masses,
        band,
        order_s,
        alpha,
        tau0,
        n_scales,
        1,
        &|scale_idx, _, _, base, out: &mut [[f64; 3]]| {
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = if k == scale_idx {
                    [base, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 0.0]
                };
            }
        },
    )?;
    let v = values.to_owned();
    Ok(forms.iter().map(|q_l| v.dot(&q_l.dot(&v))).collect())
}

/// The support diagnostic `ε ↦ q_ε(x★)`: kernel mass of the (frozen) center
/// quadrature seen from each query point at every band scale (n_query × L).
/// A query ON the web sees its strand's mass already at fine scales; a query
/// OFF the web accumulates mass only once ε reaches its distance to the
/// support. This is the on-web-ness statistic shipped alongside predictions
/// — smooth, multiresolution, derived from the measure with no neighbor
/// sets.
pub fn measure_jet_support_curve(
    queries: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    eps_band: &[f64],
) -> Result<Array2<f64>, BasisError> {
    if queries.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "measure-jet support curve dimension mismatch: queries d={} centers d={}",
            queries.ncols(),
            centers.ncols()
        );
    }
    if masses.len() != centers.nrows() {
        crate::bail_dim_basis!(
            "measure-jet support curve mass/center mismatch: {} masses for {} centers",
            masses.len(),
            centers.nrows()
        );
    }
    if eps_band.is_empty() || eps_band.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        crate::bail_invalid_basis!("measure-jet support curve needs a nonempty positive band");
    }
    validate_finite_points(queries, "queries")?;
    validate_finite_points(centers, "centers")?;
    let nq = queries.nrows();
    let nl = eps_band.len();
    let mut out = Array2::<f64>::zeros((nq, nl));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(qi, mut row)| {
            let x = queries.row(qi);
            for (li, &eps) in eps_band.iter().enumerate() {
                let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
                let mut acc = 0.0_f64;
                for (j, c) in centers.outer_iter().enumerate() {
                    let mut s = 0.0_f64;
                    for k in 0..c.len() {
                        let dlt = x[k] - c[k];
                        s += dlt * dlt;
                    }
                    acc += masses[j] * (-s * inv_two_eps2).exp();
                }
                row[li] = acc;
            }
        });
    Ok(out)
}

/// Gaussian representer features `exp(−‖x − c‖²/(2ℓ²))` (n × m).
pub fn measure_jet_design_matrix(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
) -> Result<Array2<f64>, BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "measure-jet design dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet design needs a positive finite length_scale; got {length_scale}"
        );
    }
    validate_finite_points(data, "data")?;
    validate_finite_points(centers, "centers")?;
    let inv_two_l2 = 1.0 / (2.0 * length_scale * length_scale);
    let mut out = Array2::<f64>::zeros((data.nrows(), centers.nrows()));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for (j, c) in centers.outer_iter().enumerate() {
                let mut s = 0.0_f64;
                for k in 0..c.len() {
                    let dlt = data[(i, k)] - c[k];
                    s += dlt * dlt;
                }
                row[j] = (-s * inv_two_l2).exp();
            }
        });
    Ok(out)
}

/// Resolve the realized representer range ℓ. An explicit positive
/// `spec_length_scale` is used verbatim; the `0.0` sentinel auto-initializes
/// from the median nearest-center spacing (doubled so neighboring
/// representers overlap across thin sampling gaps).
pub fn realized_measure_jet_length_scale(
    centers: ArrayView2<'_, f64>,
    spec_length_scale: f64,
) -> Result<f64, BasisError> {
    if spec_length_scale.is_finite() && spec_length_scale > 0.0 {
        return Ok(spec_length_scale);
    }
    if spec_length_scale != 0.0 {
        crate::bail_invalid_basis!(
            "measure-jet length_scale must be positive (or 0.0 for auto); got {spec_length_scale}"
        );
    }
    let dist2 = center_pairwise_dist2(centers);
    let spacing = median_nearest_center_spacing(&dist2)?;
    Ok(MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR * spacing)
}

/// Build the measure-jet smooth: Gaussian representer design `K(data,
/// centers)·z`, multiscale jet-residual penalty `zᵀ K_ccᵀ Q K_cc z`, and the
/// replayable [`BasisMetadata::MeasureJet`]. Structure mirrors the
/// constant-curvature builder; the geometry comes from the empirical measure
/// (centers + masses + band) rather than any declared chart.
pub fn build_measure_jet_basis(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("measure-jet smooth needs at least one feature column");
    }
    validate_finite_points(data, "data")?;
    let seed_centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let m = seed_centers.nrows();
    if m < 3 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: m });
    }
    let order_s = if spec.order_s == 0.0 {
        MEASURE_JET_DEFAULT_ORDER_S
    } else {
        spec.order_s
    };
    // Quadrature realization. Fit path: the realized nodes are the cell
    // BARYCENTERS of the seed partition (first-moment-exact lumping of μ —
    // see `measure_jet_quadrature_nodes`), so the metadata's `centers` are
    // already the realized nodes and the frozen path (predict / ψ-trial,
    // `CenterStrategy::UserProvided`) replays them verbatim with the frozen
    // masses + band.
    let (centers, masses, eps_band, log_step) = match &spec.frozen_quadrature {
        Some(frozen) => {
            if frozen.masses.len() != m {
                crate::bail_dim_basis!(
                    "frozen measure-jet quadrature mismatch: {} masses for {} centers",
                    frozen.masses.len(),
                    m
                );
            }
            if frozen.eps_band.is_empty() {
                crate::bail_invalid_basis!("frozen measure-jet quadrature has an empty band");
            }
            let log_step = if frozen.eps_band.len() >= 2 {
                (frozen.eps_band[1] / frozen.eps_band[0]).ln()
            } else {
                std::f64::consts::LN_2
            };
            (
                seed_centers,
                frozen.masses.clone(),
                frozen.eps_band.clone(),
                log_step,
            )
        }
        None => {
            let (nodes, masses) = measure_jet_quadrature_nodes(data, seed_centers.view())?;
            let band = measure_jet_band(nodes.view(), spec.num_scales)?;
            (nodes, masses, band.eps, band.log_step)
        }
    };
    let length_scale = realized_measure_jet_length_scale(centers.view(), spec.length_scale)?;
    let band = MeasureJetBand {
        eps: eps_band.clone(),
        log_step,
    };
    let q_form = measure_jet_energy_form(
        centers.view(),
        masses.view(),
        &band,
        order_s,
        spec.alpha,
        spec.tau0,
    )?;
    // Realized-design constraint transform: uniform coefficient sum-to-zero
    // at fit time; the frozen composed `z · z_parametric` at predict time
    // (#532 pattern — see MeasureJetIdentifiability).
    let z = match &spec.identifiability {
        MeasureJetIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != m {
                crate::bail_dim_basis!(
                    "frozen measure-jet identifiability transform mismatch: {} centers but transform has {} rows",
                    m,
                    transform.nrows()
                );
            }
            transform.clone()
        }
        MeasureJetIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(m);
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let k_cc = measure_jet_design_matrix(centers.view(), centers.view(), length_scale)?;
    let kz = k_cc.dot(&z);
    let penalty = kz.t().dot(&q_form).dot(&kz);
    let raw_design = measure_jet_design_matrix(data, centers.view(), length_scale)?;
    let design = crate::matrix::DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        raw_design.dot(&z),
    ));
    let (penalty_norm, c_primary) = normalize_penalty(&((&penalty + &penalty.t()) * 0.5));
    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty_norm,
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: c_primary,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        let ridge = Array2::<f64>::eye(design.ncols());
        let (ridge_norm, c_ridge) = normalize_penalty(&ridge);
        candidates.push(PenaltyCandidate {
            matrix: ridge_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_ridge,
            kronecker_factors: None,
            op: None,
        });
    }
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::MeasureJet {
            centers,
            input_scales: None,
            length_scale,
            eps_band,
            order_s,
            alpha: spec.alpha,
            tau0: spec.tau0,
            masses,
            constraint_transform: Some(z),
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Two clusters of 2-D centers, deterministic coordinates, plus uniform
    /// masses — small enough to inspect, irregular enough to be honest.
    fn two_cluster_centers() -> (Array2<f64>, Array1<f64>) {
        let centers = array![
            [0.00, 0.00],
            [0.31, 0.05],
            [0.58, -0.07],
            [0.93, 0.11],
            [1.22, 0.02],
            [1.49, -0.04],
            [3.10, 2.00],
            [3.42, 2.13],
            [3.71, 1.91],
            [4.05, 2.07],
            [4.33, 1.96],
            [4.61, 2.12],
        ];
        let m = centers.nrows();
        let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
        (centers, masses)
    }

    fn band_for(centers: &Array2<f64>) -> MeasureJetBand {
        measure_jet_band(centers.view(), 0).expect("band")
    }

    /// The no-mass contract: constants must be annihilated to machine
    /// precision at every scale (the constant is projected, never ridged).
    #[test]
    fn energy_form_annihilates_constants_exactly() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
            .expect("energy form");
        let m = q.nrows();
        let ones = Array1::<f64>::ones(m);
        let qv = q.dot(&ones);
        let scale = q.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 0.0, "energy form is identically zero");
        for (i, v) in qv.iter().enumerate() {
            assert!(
                v.abs() <= 1e-12 * scale,
                "Q·1 leak at row {i}: {v:.3e} vs scale {scale:.3e}"
            );
        }
        let vqv = ones.dot(&qv);
        assert!(
            vqv.abs() <= 1e-12 * scale,
            "constant carries energy: 1ᵀQ1 = {vqv:.3e}"
        );
    }

    /// Unridged (τ = 0) local fits annihilate ambient affine functions on
    /// well-spread centers; the fitted default τ = 1e-3 damps them to far
    /// below a rough vector's energy instead of exactly zero.
    #[test]
    fn energy_form_annihilates_affine_when_unridged() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let m = centers.nrows();
        // Affine values v = 0.7 + 1.3·x − 0.4·y, and a rough ±1 checkerboard.
        let mut affine = Array1::<f64>::zeros(m);
        let mut rough = Array1::<f64>::zeros(m);
        for i in 0..m {
            affine[i] = 0.7 + 1.3 * centers[(i, 0)] - 0.4 * centers[(i, 1)];
            rough[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let q0 = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 0.0)
            .expect("unridged energy form");
        let e_affine0 = affine.dot(&q0.dot(&affine));
        let e_rough0 = rough.dot(&q0.dot(&rough));
        assert!(e_rough0 > 0.0, "rough vector must pay energy");
        assert!(
            e_affine0.abs() <= 1e-10 * e_rough0,
            "unridged affine energy {e_affine0:.3e} vs rough {e_rough0:.3e}"
        );
        let q1 = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
            .expect("ridged energy form");
        let e_affine1 = affine.dot(&q1.dot(&affine));
        let e_rough1 = rough.dot(&q1.dot(&rough));
        assert!(
            e_affine1 <= 1e-2 * e_rough1,
            "ridged affine energy {e_affine1:.3e} not damped vs rough {e_rough1:.3e}"
        );
    }

    /// PSD: the energy is a sum of weighted least-squares residuals.
    #[test]
    fn energy_form_is_psd() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
            .expect("energy form");
        let m = q.nrows();
        for trial in 0..5usize {
            let v = Array1::<f64>::from_shape_fn(m, |i| {
                ((i * 7 + trial * 13) % 11) as f64 / 11.0 - 0.5
            });
            let e = v.dot(&q.dot(&v));
            assert!(e >= -1e-10, "vᵀQv = {e:.3e} < 0 on trial {trial}");
        }
    }

    /// A 1-D filament embedded in 2-D: high-frequency center values along the
    /// strand pay strictly more energy than a slow trend.
    #[test]
    fn rough_vector_pays_more_than_smooth() {
        let m = 24usize;
        let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
            let t = i as f64 / (m as f64 - 1.0);
            if k == 0 { t * 4.0 } else { 0.3 * (t * 4.0).sin() }
        });
        let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
        let band = band_for(&centers);
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
            .expect("energy form");
        let slow = Array1::<f64>::from_shape_fn(m, |i| (i as f64 / (m as f64 - 1.0)).powi(2));
        let fast = Array1::<f64>::from_shape_fn(m, |i| if i % 2 == 0 { 0.5 } else { -0.5 });
        let e_slow = slow.dot(&q.dot(&slow));
        let e_fast = fast.dot(&q.dot(&fast));
        assert!(
            e_fast > 10.0 * e_slow,
            "alternating values must pay >> a slow trend: fast {e_fast:.3e} vs slow {e_slow:.3e}"
        );
    }

    /// The exact (s, α) jets must match central finite differences of the
    /// energy — the FD gate the ψ-channel stage will inherit (the discipline
    /// whose absence is exactly the objective↔gradient desync bug class).
    #[test]
    fn energy_jets_match_finite_differences() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let (s0, a0, tau) = (1.3, 0.8, 1e-3);
        let jets =
            measure_jet_energy_form_with_jets(centers.view(), masses.view(), &band, s0, a0, tau)
                .expect("jets");
        let q_at = |s: f64, a: f64| {
            measure_jet_energy_form(centers.view(), masses.view(), &band, s, a, tau)
                .expect("energy form")
        };
        // Base form must equal the plain assembly bit-for-bit (same walk).
        let q_plain = q_at(s0, a0);
        for (a, b) in jets.q.iter().zip(q_plain.iter()) {
            assert!((a - b).abs() <= 1e-14 * (1.0 + b.abs()), "Q drift {a} vs {b}");
        }
        let lt0 = tau.ln();
        let q_at_lt = |lt: f64| {
            measure_jet_energy_form(centers.view(), masses.view(), &band, s0, a0, lt.exp())
                .expect("energy form")
        };
        let h = 1e-5;
        let checks: [(&str, &Array2<f64>, Array2<f64>); 9] = [
            ("dq_ds", &jets.dq_ds, {
                let (p, m_) = (q_at(s0 + h, a0), q_at(s0 - h, a0));
                (&p - &m_) / (2.0 * h)
            }),
            ("d2q_ds2", &jets.d2q_ds2, {
                let (p, c, m_) = (q_at(s0 + h, a0), q_at(s0, a0), q_at(s0 - h, a0));
                (&(&p + &m_) - &(&c * 2.0)) / (h * h)
            }),
            ("dq_dalpha", &jets.dq_dalpha, {
                let (p, m_) = (q_at(s0, a0 + h), q_at(s0, a0 - h));
                (&p - &m_) / (2.0 * h)
            }),
            ("d2q_dalpha2", &jets.d2q_dalpha2, {
                let (p, c, m_) = (q_at(s0, a0 + h), q_at(s0, a0), q_at(s0, a0 - h));
                (&(&p + &m_) - &(&c * 2.0)) / (h * h)
            }),
            ("d2q_ds_dalpha", &jets.d2q_ds_dalpha, {
                let pp = q_at(s0 + h, a0 + h);
                let pm = q_at(s0 + h, a0 - h);
                let mp = q_at(s0 - h, a0 + h);
                let mm = q_at(s0 - h, a0 - h);
                (&(&pp - &pm) - &(&mp - &mm)) / (4.0 * h * h)
            }),
            ("dq_dlogtau", &jets.dq_dlogtau, {
                let (p, m_) = (q_at_lt(lt0 + h), q_at_lt(lt0 - h));
                (&p - &m_) / (2.0 * h)
            }),
            ("d2q_dlogtau2", &jets.d2q_dlogtau2, {
                let (p, c, m_) = (q_at_lt(lt0 + h), q_at_lt(lt0), q_at_lt(lt0 - h));
                (&(&p + &m_) - &(&c * 2.0)) / (h * h)
            }),
            ("d2q_ds_dlogtau", &jets.d2q_ds_dlogtau, {
                let f = |s: f64, lt: f64| {
                    measure_jet_energy_form(
                        centers.view(),
                        masses.view(),
                        &band,
                        s,
                        a0,
                        lt.exp(),
                    )
                    .expect("energy form")
                };
                let pp = f(s0 + h, lt0 + h);
                let pm = f(s0 + h, lt0 - h);
                let mp = f(s0 - h, lt0 + h);
                let mm = f(s0 - h, lt0 - h);
                (&(&pp - &pm) - &(&mp - &mm)) / (4.0 * h * h)
            }),
            ("d2q_dalpha_dlogtau", &jets.d2q_dalpha_dlogtau, {
                let f = |a: f64, lt: f64| {
                    measure_jet_energy_form(
                        centers.view(),
                        masses.view(),
                        &band,
                        s0,
                        a,
                        lt.exp(),
                    )
                    .expect("energy form")
                };
                let pp = f(a0 + h, lt0 + h);
                let pm = f(a0 + h, lt0 - h);
                let mp = f(a0 - h, lt0 + h);
                let mm = f(a0 - h, lt0 - h);
                (&(&pp - &pm) - &(&mp - &mm)) / (4.0 * h * h)
            }),
        ];
        for (name, analytic, fd) in checks.iter() {
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (a, b) in analytic.iter().zip(fd.iter()) {
                assert!(
                    (a - b).abs() <= 5e-5 * scale,
                    "{name} jet mismatch: analytic {a:.6e} vs FD {b:.6e} (scale {scale:.3e})"
                );
            }
        }
    }

    /// The per-scale spectrum must sum exactly to the total energy (same
    /// blocks, one-hot weights) and concentrate rough content at fine
    /// scales.
    #[test]
    fn scale_spectrum_sums_to_total_and_localizes_roughness() {
        let m = 24usize;
        let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
            let t = i as f64 / (m as f64 - 1.0);
            if k == 0 { t * 4.0 } else { 0.0 }
        });
        let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
        let band = band_for(&centers);
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
            .expect("energy form");
        let fast = Array1::<f64>::from_shape_fn(m, |i| if i % 2 == 0 { 0.5 } else { -0.5 });
        let spec = measure_jet_scale_spectrum(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            1.0,
            1e-3,
            fast.view(),
        )
        .expect("spectrum");
        assert_eq!(spec.len(), band.eps.len());
        let total = fast.dot(&q.dot(&fast));
        let sum: f64 = spec.iter().sum();
        assert!(
            (sum - total).abs() <= 1e-10 * total.abs().max(1e-30),
            "spectrum must sum to vᵀQv: {sum:.6e} vs {total:.6e}"
        );
        // Alternating-sign content lives at the finest scale of the band.
        let finest = spec[0];
        let coarsest = *spec.last().expect("nonempty spectrum");
        assert!(
            finest > coarsest,
            "alternating values must charge fine scales hardest: fine {finest:.3e} vs coarse {coarsest:.3e}"
        );
    }

    /// The support curve separates on-web from off-web queries at fine
    /// scales and grows monotonically in ε for any query.
    #[test]
    fn support_curve_separates_on_web_from_off_web() {
        let m = 24usize;
        let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
            let t = i as f64 / (m as f64 - 1.0);
            if k == 0 { t * 4.0 } else { 0.0 }
        });
        let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
        let band = band_for(&centers);
        let queries = array![[2.0, 0.0], [2.0, 1.5]];
        let curves =
            measure_jet_support_curve(queries.view(), centers.view(), masses.view(), &band.eps)
                .expect("support curve");
        // On-web sees strictly more mass than off-web at the finest scale.
        assert!(
            curves[(0, 0)] > 10.0 * curves[(1, 0)],
            "fine-scale support must separate web from void: on {:.3e} vs off {:.3e}",
            curves[(0, 0)],
            curves[(1, 0)]
        );
        // Kernel mass is monotone in ε for every query.
        for qi in 0..2 {
            for li in 1..band.eps.len() {
                assert!(
                    curves[(qi, li)] >= curves[(qi, li - 1)] - 1e-15,
                    "support curve must be monotone in scale (query {qi}, level {li})"
                );
            }
        }
    }

    /// Quadrature nodes must be the mass-weighted cell barycenters
    /// (first-moment-exact lumping), with empty cells keeping their seed
    /// coordinates at zero mass.
    #[test]
    fn quadrature_nodes_are_cell_barycenters() {
        // Two tight groups around (0,0) and (10,10); a third seed far away
        // captures nothing.
        let data = array![
            [0.0, 0.2],
            [0.4, -0.2],
            [0.2, 0.0],
            [9.8, 10.1],
            [10.2, 9.9],
        ];
        let seeds = array![[0.1, 0.1], [10.0, 10.0], [-50.0, -50.0]];
        let (nodes, masses) =
            measure_jet_quadrature_nodes(data.view(), seeds.view()).expect("quadrature nodes");
        assert!((masses.sum() - 1.0).abs() <= 1e-15, "masses must sum to 1");
        assert!((masses[0] - 0.6).abs() <= 1e-15);
        assert!((masses[1] - 0.4).abs() <= 1e-15);
        assert_eq!(masses[2], 0.0);
        // Cell 0 barycenter = mean of the three assigned rows.
        assert!((nodes[(0, 0)] - 0.2).abs() <= 1e-12);
        assert!((nodes[(0, 1)] - 0.0).abs() <= 1e-12);
        // Cell 1 barycenter.
        assert!((nodes[(1, 0)] - 10.0).abs() <= 1e-12);
        assert!((nodes[(1, 1)] - 10.0).abs() <= 1e-12);
        // Empty cell keeps its seed coordinates.
        assert_eq!(nodes[(2, 0)], -50.0);
        assert_eq!(nodes[(2, 1)], -50.0);
    }

    /// Freeze→replay: rebuilding from the first build's frozen transform and
    /// frozen quadrature must reproduce design and penalty bit-for-bit (the
    /// predict-path contract).
    #[test]
    fn build_replay_roundtrip_reproduces_design_and_penalty() {
        // A bent filament with a side cluster, n = 40 deterministic rows.
        let n = 40usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 {
                t * 3.0
            } else {
                0.5 * (t * 3.0).cos() + if i % 9 == 0 { 0.8 } else { 0.0 }
            }
        });
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 14 },
            ..MeasureJetBasisSpec::default()
        };
        let first = build_measure_jet_basis(data.view(), &spec).expect("first build");
        let BasisMetadata::MeasureJet {
            centers,
            length_scale,
            eps_band,
            order_s,
            alpha,
            tau0,
            masses,
            constraint_transform,
            ..
        } = &first.metadata
        else {
            panic!("measure-jet build must return MeasureJet metadata");
        };
        let replay_spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            order_s: *order_s,
            alpha: *alpha,
            tau0: *tau0,
            num_scales: eps_band.len(),
            length_scale: *length_scale,
            double_penalty: spec.double_penalty,
            identifiability: MeasureJetIdentifiability::FrozenTransform {
                transform: constraint_transform.clone().expect("fit-time z"),
            },
            frozen_quadrature: Some(MeasureJetFrozenQuadrature {
                masses: masses.clone(),
                eps_band: eps_band.clone(),
            }),
        };
        let second = build_measure_jet_basis(data.view(), &replay_spec).expect("replay build");
        let x1 = first.design.to_dense();
        let x2 = second.design.to_dense();
        assert_eq!(x1.shape(), x2.shape());
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() <= 1e-12, "design replay drift: {a} vs {b}");
        }
        assert_eq!(first.penalties.len(), second.penalties.len());
        for (p1, p2) in first.penalties.iter().zip(second.penalties.iter()) {
            for (a, b) in p1.iter().zip(p2.iter()) {
                assert!((a - b).abs() <= 1e-12, "penalty replay drift: {a} vs {b}");
            }
        }
    }
}
