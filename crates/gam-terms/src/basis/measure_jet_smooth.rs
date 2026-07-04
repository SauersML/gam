//! Measure-jet spline smooth: multiscale local-jet-residual energy of the
//! empirical measure (center-quadratured current implementation).
//!
//! The term penalizes, at every quadrature point and every scale, the failure
//! of `f` to be locally affine *in the measure*:
//!
//! ```text
//!   Q = Σ_ℓ  w_ℓ · Σ_i  mass_i · q_i(ε_ℓ)^(1−2α) · R_{i,ℓ},
//!   w_ℓ = log_step · ε_ℓ^(−η),   η = 2s + d(2−2α),
//! ```
//!
//! where `R_{i,ℓ}` is the residual quadratic form of the exact weighted
//! local affine projection at center `i` and scale `ε_ℓ`: kernel weights
//! `w_j = mass_j · exp(−d_ij²/(2ε_ℓ²))`, kernel mass `q_i = Σ_j w_j`, and the
//! fit `min_b ‖Cv − Φ̃b‖²_W` over weighted-centered values
//! `Cv = v − (uᵀv)·1` (`u = w/q`) and weighted-centered scaled features
//! `Φ̃` (rows `(c_j − c_i)/ε`, column means removed under `u`). Rank-deficient
//! cells use the machine-precision pseudo-inverse of `Φ̃ᵀWΦ̃/q`, so ambient
//! affine values are projected away exactly instead of paying a ridge toll.
//!
//! # Contracts (each is load-bearing; tests pin them)
//!
//! - **Exact constant annihilation.** The constant is removed by the weighted
//!   mean projection `C`, never ridged: `Q·1 = 0` to machine precision at
//!   every scale, so the penalty carries NO mass term and the fit has no
//!   prior mean to revert to. This is the no-mean-reversion contract of the
//!   measure-jet design; ridging the constant would silently reintroduce
//!   mean reversion.
//! - **Exact affine projection / rank adaptation.** The slope block uses the
//!   rank-revealing pseudo-inverse of the dimensionless local Gram
//!   `G = Φ̃ᵀWΦ̃/q`, not a Tikhonov ridge. On a 1-D filament in ambient
//!   dimension d the resolved tangent slope is absorbed (not penalized);
//!   unresolved directions have no variation after weighted centering and do
//!   not create an affine toll. The retained rank is a numerical property of
//!   the weighted cell, not a smoothing dial.
//! - **Mellin band.** Scales form a geometric grid from the center-spacing
//!   floor to the half-diameter; `w_ℓ = log_step · ε_ℓ^(−η)`, with
//!   `η = 2s + d(2−2α)`, is the fixed-order quadrature weight used by this
//!   implementation. It keeps the advertised continuous smoothness order
//!   `s ∈ (0, 2)` from silently changing when `α` changes.
//! - **Density normalization.** The outer quadrature weight
//!   `mass_i · q_i^(1−2α)` realizes `dμ(x)/q_ε(x)^(2α−1)`. On a p-dimensional
//!   stratum with sampling density `ρ`, `q_ε ~ Cρ ε^p` and the local residual
//!   contributes an extra `ε^{p(2−2α)}` factor. The fixed-order scale weight
//!   cancels that factor using the available dimension parameter; without that
//!   correction, the symbol exponent would be `2s + 2p(α−1)`.
//! - **Frozen-quadrature replay.** The penalty and extrapolation diagnostic
//!   depend on the FIT data through center masses, the realized band, on-web
//!   support anchors, and penalty normalization scales. The freeze step
//!   persists all of them ([`MeasureJetFrozenQuadrature`]) so predict-time
//!   rebuilds replay the exact fit-time penalty instead of recomputing it from
//!   predict rows.
//! - **Single assembly source.** Every quadratic form this module emits —
//!   the energy, its (s, α) jets, the per-scale spectrum — is produced by
//!   ONE workhorse ([`assemble_weighted_forms`]) that walks the local
//!   residual blocks exactly once per request and differs only in the
//!   scalar weights applied per block. Criterion value and criterion
//!   derivatives cannot drift apart (the objective↔gradient desync class is
//!   structurally excluded).
//! - **single-scale/multiscale opt-in (#1039/#1116).** The per-scale spectrum
//!   and the `(α, ln τ)` ψ dials are the multiscale-mode realization, engaged
//!   ONLY when the spec opts in (`MeasureJetBasisSpec::multiscale = true`, the
//!   DSL `mjs(…, multiscale=true)`); see [`measure_jet_multiscale_mode`]. There
//!   is NO center-count auto-gate: at ANY center count the default is
//!   single-scale — one fused jet-energy penalty at the auto order with no ψ
//!   dials, the same one-λ outer footprint as Duchon/Matérn — so a fit pays
//!   Duchon-class cost unless the user explicitly asks for the per-scale
//!   spectrum. The flag is persisted on the spec, so freeze→replay re-enters
//!   the same mode verbatim.
//!
//! # ψ-differentiability contract (what the ψ-channel stage consumes)
//!
//! Mirroring the constant-curvature κ-contract (#944): centers, masses, the
//! band are deliberately hyperparameter-FIXED at build time; the representer
//! range ℓ is the ONE opt-in design-moving dial (#1116). Consequences:
//!
//! - **Penalty-dial design drift is identically zero**: the (s, α, τ) dials
//!   reweight only the jet-energy penalty, never the Gaussian representer
//!   design (`∂X/∂{s,α,τ} ≡ 0`), so those channels are penalty-only
//!   (`is_penalty_like` auto-derives true in the outer engine's
//!   `DirectionalHyperParam`).
//! - **The representer range ℓ is a design-moving dial** (matérn's `log_kappa`
//!   analog, #1116): `X = K(data, centers; ℓ)·z` depends on ℓ, so
//!   `∂X/∂lnℓ = (K ⊙ r²/ℓ²)·z ≠ 0`, while the jet-energy penalty does NOT
//!   depend on ℓ (`∂S/∂lnℓ ≡ 0`). When explicitly enabled, ℓ rebuilds the
//!   design per outer trial; it does not change the basis RANK (the Gaussian
//!   kernel is PD ∀ℓ > 0 → rank ≡ m centers) but it does change which m-dim
//!   subspace the representers span. FD-gated by
//!   `psi_producer_matches_fd_length_scale`.
//! - **Exact (s, α) penalty jets are shipped**:
//!   [`measure_jet_energy_form_with_jets`] returns `∂Q/∂s`, `∂²Q/∂s²`,
//!   `∂Q/∂α`, `∂²Q/∂α²`, `∂²Q/∂s∂α` in closed form — both dials enter only
//!   through the per-block log-weights (`∂ln w/∂s = −2 ln ε`,
//!   `∂ln w/∂α = −2 ln q`), so the jets are reweighted re-scatters of the
//!   SAME residual blocks, FD-gated in this module's tests.
//!   The retained τ coordinate is inert under the exact projection, so its
//!   derivative slots are identically zero.
//!
//! # Cost shape (and the upgrade ladder above it)
//!
//! The outer sum is coarsened per scale to a deterministic ε/2-net (the
//! outer Riemann sum needs resolution ε, not the center-spacing floor), so
//! the band totals ~O(m²·d) instead of O(L·m³) — the current realization of
//! the pyramid principle that each scale interacts at its own level. This is
//! mass-lumped quadrature of the displayed outer integral; it is first-
//! moment exact for the cell locations and carries the usual
//! `O(diam²/ε²)` relative scale for smooth Gaussian-weighted functionals,
//! not an estimand-preserving identity.
//! The long-form home for the ladder and the substrate contracts is the
//! frame notes (`docs/measure_jet_frame.md`); its §2 moment substrate is
//! `measure_jet_moments.rs`, its §5 extrapolation pricing
//! `measure_jet_predict.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use faer::Side;

use gam_linalg::faer_ndarray::FaerEigh;

use super::{
    AnisoBasisPsiDerivatives, AnisoPenaltyCrossProvider, BasisBuildResult, BasisError,
    BasisMetadata, CenterStrategy, PenaltyCandidate, PenaltySource,
    filter_active_penalty_candidates_with_ops, normalize_penalty,
    normalize_penalty_cross_psi_derivative, normalize_penaltywith_psi_derivatives,
    select_centers_by_strategy, trace_of_product,
};

/// Truncation radius of the Gaussian profile in units of the scale ε: weights
/// beyond `3ε` are below `e^{-4.5} ≈ 1.1e-2` of the peak and are dropped from
/// both the local fit and the `q^(1−2α)` outer weight. This is an absolute
/// kernel-weight cutoff; using the same truncated q keeps the discrete
/// functional self-consistent, but it is not a relative tail-error bound.
pub(crate) const MEASURE_JET_PROFILE_CUTOFF: f64 = 3.0;

/// Relative eigenvalue threshold for rank-revealing pseudo-inverses of local
/// Gram matrices. Directions at the roundoff floor are treated as unresolved
/// and excluded from the affine fit.
pub(crate) const MEASURE_JET_PSEUDOINVERSE_RTOL: f64 = 64.0 * f64::EPSILON;

/// Default continuous smoothness order `s` realized by the `0.0` auto
/// sentinel. Sits mid-band in the admissible `(0, 2)` for the affine-jet
/// (r = 2) energy: rough enough to stay pointwise-defined on filaments and
/// sheets (`s > p/2` for intrinsic `p ≤ 2`), smooth enough to bridge gaps
/// with attested trends.
pub(crate) const MEASURE_JET_DEFAULT_ORDER_S: f64 = 1.5;

/// Auto-band scale-count clamp: at least 3 octave-ish nodes so the energy is
/// genuinely multiscale, at most 8 so degenerate spacing cannot explode the
/// build.
pub(crate) const MEASURE_JET_MIN_AUTO_SCALES: usize = 3;
pub(crate) const MEASURE_JET_MAX_AUTO_SCALES: usize = 8;

/// Representer-range multiple of the median nearest-center spacing used by
/// the `0.0` auto sentinel.
///
/// Set to ×1: a Gaussian representer of range `ℓ = h` (the median
/// nearest-center spacing) already overlaps its neighbors at
/// `exp(−h²/(2ℓ²)) = exp(−1/2) ≈ 0.61`, so adjacent bumps blend smoothly while
/// each center keeps a *distinct* response. The old ×2 made every column
/// `exp(−1/8) ≈ 0.88` at its neighbor — the representers became nearly
/// collinear, which (a) over-smoothed the fitted surface (the #1116/#1041
/// accuracy deficit: measure-jet sat ~1.6× the matern/duchon truth-RMSE) and
/// (b) drove the design Gram toward rank deficiency, so the inner PIRLS /
/// outer REML conditioning degraded and the smoothing-parameter search cycled
/// for hundreds of seconds (the #1116 timeout). One spacing-width kernel fixes
/// both at the root without touching the energy penalty or the dials.
pub(crate) const MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR: f64 = 1.0;

/// Single-scale fused-mode weight of the affine-preserving nullspace ridge,
/// as a fraction of the primary energy penalty's Frobenius scale (#1116). The
/// ridge's role is identifiability — flooring the fused energy's near-nullspace
/// — not smoothing selection, so it is fused at a small FIXED fraction rather
/// than carrying its own REML λ (which would double the single-scale outer
/// search). 1e-2 lifts the unpenalized directions ~two orders below the
/// dominant penalized modes: a definite identifiability floor that leaves the
/// genuine spectrum (and the single primary λ) in control of the fit.
pub(crate) const MEASURE_JET_FUSED_RIDGE_FRACTION: f64 = 1e-2;

/// Memory budget (in f64 entries) above which the multi-form assembly stops
/// parallelizing over scales: parallel scale partials cost
/// `L · n_forms · m²` doubles; past this budget the scales run sequentially
/// (same numbers — the per-scale loop and the ordered sum are deterministic
/// either way).
pub(crate) const MEASURE_JET_PARALLEL_FORM_BUDGET_DOUBLES: usize = 1 << 26;

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
    /// Per-scale on-web support anchor
    /// `q̄_ℓ = (Σ_i m_i q_ℓ(c_i)) / (Σ_i m_i)`.
    pub support_means: Vec<f64>,
    /// Frobenius scales of the emitted per-level normalized penalties. Empty in
    /// fused mode, where the band emits one primary penalty instead.
    pub penalty_normalization_scales: Vec<f64>,
    /// Frobenius scales of the raw per-level forms before the arbitrary Mellin
    /// `log_step · ε_ℓ^(-2s0)` gauge is folded in.
    pub raw_penalty_normalization_scales: Vec<f64>,
    /// Frobenius scale of the single fused primary penalty. `None` in per-level
    /// mode.
    pub fused_penalty_normalization_scale: Option<f64>,
}

/// Serde default for [`MeasureJetBasisSpec::learn_length_scale`]: freeze ℓ at
/// the realized auto/user value unless a fit explicitly opts into the
/// design-moving outer coordinate.
fn measure_jet_learn_length_scale_default() -> bool {
    false
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
    /// Historical τ coordinate retained for frozen specs and ψ layout. The
    /// measure-jet energy itself uses the exact weighted affine projection and
    /// is independent of τ; the τ ψ derivatives are therefore zero.
    pub tau0: f64,
    /// Number of scale nodes; `0` sentinel = auto dyadic band.
    pub num_scales: usize,
    /// Representer (Gaussian RBF) range ℓ; `0.0` sentinel = auto
    /// (median nearest-center spacing × [`MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR`]).
    pub length_scale: f64,
    /// Add an affine-preserving shrinkage penalty alongside the jet-energy
    /// penalty.
    pub double_penalty: bool,
    /// REML-learn the representer range ℓ as a design-moving outer dial
    /// (opt-in), mirroring Matérn's `log_kappa`. The Gaussian kernel is
    /// strictly PD for every ℓ > 0, so ℓ does NOT change the basis rank (always
    /// `m` centers) — but it changes WHICH `m`-dim subspace the representers
    /// span, i.e. the span alignment with the true surface. The stable default
    /// freezes ℓ at the auto/user value; `true` enrolls the outer coordinate for
    /// experiments that need REML-selected representer range.
    #[serde(default = "measure_jet_learn_length_scale_default")]
    pub learn_length_scale: bool,
    /// Explicit opt-in for multiscale mode: the per-scale spectral penalty
    /// split plus the `(α, ln τ)` outer ψ dials and the affine-preserving
    /// ridge. `false` (default) keeps the term in single-scale mode at ANY
    /// center count — one fused jet-energy penalty, the one-λ Duchon/Matérn
    /// footprint. There is no center-count auto-gate; the user opts in via
    /// `mjs(…, multiscale=true)`. Persisted on the spec so freeze→replay
    /// re-enters the same mode.
    #[serde(default)]
    pub multiscale: bool,
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
            // Density-WEIGHTED Hessian energy (the module-header default): the
            // outer weight is q^{1−2α} = q^{−1} at α = 1. The density-free
            // variant α = 3/2 gives q^{−2}, which on a low-intrinsic-dimension
            // stratum (data on a 1-D/2-D manifold embedded in higher ambient d)
            // makes the local kernel mass q tiny AND spatially varying along
            // the manifold, so q^{−2} amplifies the penalty unevenly and
            // over-smooths the high-frequency signal there (MEASURED #1116: on
            // the 1-D-curve-in-3-D fixture α = 3/2 left mjs ~13× worse than
            // matérn). α = 1's q^{−1} weighting is far gentler and is the
            // header-derived default; an explicit `alpha=` still overrides for
            // genuinely density-free use on a full-dimensional stratum.
            alpha: 1.0,
            tau0: 1e-3,
            num_scales: 0,
            length_scale: 0.0,
            double_penalty: true,
            learn_length_scale: false,
            multiscale: false,
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

/// The energy and its exact hyperparameter jets in the live dials. `s` and
/// `α` enter only through per-block log-weights. The retained `ln τ` slots
/// are zero because the local fit is the exact weighted affine projection
/// and no longer depends on τ. All forms are scattered from the SAME local
/// residual blocks, and the ψ-channel consumes them with zero design drift.
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

/// Householder vector `u` for the uniform sum-to-zero constraint: the
/// reflection `H = I − 2uuᵀ` maps `c̄ = 1/√m·1` onto `e₁`, so columns 2..m
/// of `H` are an orthonormal basis of `1⊥` — the same model space as the
/// generic RRQR nullspace basis, but with O(rows·m) STRUCTURED application
/// (`X·z = (X − 2(Xu)uᵀ) minus column 1`) instead of the O(rows·m²)
/// constraint GEMM that the scale-smoke gate identified as the dominant
/// build cost.
pub(crate) fn householder_sum_to_zero_u(m: usize) -> Array1<f64> {
    let c = 1.0 / (m as f64).sqrt();
    let mut u = Array1::<f64>::from_elem(m, c);
    u[0] -= 1.0;
    let norm = u.dot(&u).sqrt();
    u.mapv_inplace(|v| v / norm);
    u
}

/// Materialize the Householder sum-to-zero basis `z` (m × (m−1)) — columns
/// 2..m of `H = I − 2uuᵀ` — for the frozen-replay metadata. O(m²), built
/// once per fit.
pub(crate) fn householder_sum_to_zero_z(u: &Array1<f64>) -> Array2<f64> {
    let m = u.len();
    let mut z = Array2::<f64>::zeros((m, m - 1));
    for j in 0..(m - 1) {
        for i in 0..m {
            let h = if i == j + 1 { 1.0 } else { 0.0 } - 2.0 * u[i] * u[j + 1];
            z[(i, j)] = h;
        }
    }
    z
}

pub(crate) fn symmetric_pseudoinverse(
    a: &Array2<f64>,
    label: &str,
) -> Result<Array2<f64>, BasisError> {
    let n = a.nrows();
    if a.ncols() != n {
        crate::bail_dim_basis!(
            "measure-jet pseudo-inverse `{label}` needs a square matrix, got {:?}",
            a.dim()
        );
    }
    let (evals, evecs) = a.eigh(Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!(
            "measure-jet pseudo-inverse `{label}` eigendecomposition failed: {e}"
        ))
    })?;
    let lam_max = evals.iter().fold(0.0_f64, |acc, v| acc.max((*v).max(0.0)));
    let rank_tol = MEASURE_JET_PSEUDOINVERSE_RTOL * (n.max(1) as f64) * lam_max;
    let mut scaled = evecs.clone();
    for (k, mut col) in scaled.axis_iter_mut(Axis(1)).enumerate() {
        let lam = evals[k].max(0.0);
        let inv = if lam > rank_tol { 1.0 / lam } else { 0.0 };
        col.mapv_inplace(|v| v * inv);
    }
    Ok(scaled.dot(&evecs.t()))
}

/// Affine-preserving shrinkage ridge `I − P_pseudoaffine` in the constrained
/// Gaussian-coefficient space, where `P_pseudoaffine` projects onto the
/// coefficient directions `b̂` that — pushed through the representer `K·z` — best
/// reproduce the affine center functions `[1, c₁…c_d]` under the mass-weighted
/// normal equations. The double penalty thus shrinks the wiggle space while
/// preserving the (pseudo-)affine component, the measure-jet analogue of leaving
/// a smooth's null space lightly damped.
pub(crate) fn affine_preserving_coefficient_ridge(
    kz: &Array2<f64>,
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let m = centers.nrows();
    let d = centers.ncols();
    let p = kz.ncols();
    if kz.nrows() != m || masses.len() != m {
        crate::bail_dim_basis!(
            "measure-jet affine-preserving ridge shape mismatch: kz {:?}, centers {:?}, masses {}",
            kz.dim(),
            centers.dim(),
            masses.len()
        );
    }
    if p == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let mut weighted_kz = kz.clone();
    for (i, mut row) in weighted_kz.outer_iter_mut().enumerate() {
        row.mapv_inplace(|v| v * masses[i]);
    }
    let normal = kz.t().dot(&weighted_kz);
    let normal_pinv = symmetric_pseudoinverse(&normal, "affine ridge normal")?;
    let mut affine = Array2::<f64>::ones((m, d + 1));
    for i in 0..m {
        for k in 0..d {
            affine[(i, k + 1)] = centers[(i, k)];
        }
    }
    let mut weighted_affine = affine.clone();
    for (i, mut row) in weighted_affine.outer_iter_mut().enumerate() {
        row.mapv_inplace(|v| v * masses[i]);
    }
    let rhs = kz.t().dot(&weighted_affine);
    let beta = normal_pinv.dot(&rhs);
    let beta_gram = beta.t().dot(&beta);
    let (evals, evecs) = beta_gram.eigh(Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!(
            "measure-jet affine ridge subspace eigendecomposition failed: {e}"
        ))
    })?;
    let lam_max = evals.iter().fold(0.0_f64, |acc, v| acc.max((*v).max(0.0)));
    let rank_tol = MEASURE_JET_PSEUDOINVERSE_RTOL * ((d + 1).max(1) as f64) * lam_max;
    let mut ridge = Array2::<f64>::eye(p);
    for k in 0..(d + 1) {
        let lam = evals[k].max(0.0);
        if lam <= rank_tol {
            continue;
        }
        let dir = beta.dot(&evecs.column(k).to_owned()) / lam.sqrt();
        for r in 0..p {
            for c in 0..p {
                ridge[(r, c)] -= dir[r] * dir[c];
            }
        }
    }
    Ok((&ridge + &ridge.t()) * 0.5)
}

/// Pairwise squared distances `‖a_i − b_j‖²` via the GEMM identity
/// `‖a − b‖² = ‖a‖² + ‖b‖² − 2·aᵀb`: one (n×d)·(d×m) matrix product carries
/// every FMA at tile speed instead of n·m scalar distance loops — the
/// machine-native form of this kernel, and the module's ONLY distance
/// source (representer design, support curve, and the center-pair geometry:
/// band floor, median spacing, ε/2-net, neighbor cutoffs). The cancellation
/// error near-coincident points pay is O(ε_f64·‖x‖²) ABSOLUTE, harmless
/// under a Gaussian profile (the kernel is flat at d ≈ 0); clamped at zero
/// so roundoff cannot emit tiny negatives (the a = b diagonal therefore
/// lands at roundoff scale, not an exact 0 — no caller pins it).
pub(crate) fn pairwise_sq_dists(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    let an: Vec<f64> = a.outer_iter().map(|r| r.dot(&r)).collect();
    let bn: Vec<f64> = b.outer_iter().map(|r| r.dot(&r)).collect();
    let mut g = a.dot(&b.t());
    g.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (an[i] + bn[j] - 2.0 * *v).max(0.0);
            }
        });
    g
}

/// Row-block size for streaming GEMM passes that must not materialize the
/// full n×m distance matrix (nearest-node assignment): 64Ki rows × m ≤ a
/// few hundred MB of transient per block, GEMM-speed throughout.
pub(crate) const MEASURE_JET_ASSIGN_BLOCK_ROWS: usize = 65_536;

pub(crate) fn validate_finite_points(
    points: ArrayView2<'_, f64>,
    what: &str,
) -> Result<(), BasisError> {
    for (i, row) in points.outer_iter().enumerate() {
        if row.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_basis!("measure-jet {what} row {i} has a non-finite coordinate");
        }
    }
    Ok(())
}

/// Median nearest-OTHER-center distance — the resolution floor of the center
/// quadrature, used for the band floor and the auto representer range.
pub(crate) fn median_nearest_center_spacing(dist2: &Array2<f64>) -> Result<f64, BasisError> {
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
/// ceiling at half the bounding-box diagonal (a deterministic diameter-scale
/// cap; local fits remain center-weighted and distinct there).
/// `num_scales == 0` requests the auto count `clamp(⌈log2(ε_max/ε_min)⌉ + 1,
/// 3, 8)`; a degenerate band (ceiling ≤ floor) collapses to the single floor
/// scale with `log_step = ln 2`.
pub fn measure_jet_band(
    centers: ArrayView2<'_, f64>,
    num_scales: usize,
) -> Result<MeasureJetBand, BasisError> {
    validate_finite_points(centers, "centers")?;
    let dist2 = pairwise_sq_dists(centers, centers);
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

/// First-moment-exact quadrature of the empirical measure on the cell partition
/// induced by the seed centers: nearest-center assignment (deterministic
/// tie-break: lowest center index) yields per-cell masses, and each non-empty
/// cell's quadrature node is its mass-weighted barycenter. Empty cells keep
/// their seed coordinates with zero mass (the assembly skips them; their
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
    // Nearest-node assignment in streamed GEMM blocks: argmin_j ‖x−c_j‖² =
    // argmin_j (‖c_j‖² − 2·xᵀc_j), so each block is one (rows×d)·(d×m)
    // product plus a row-wise argmin — tile-speed FMAs, O(block·m) transient
    // memory, deterministic ties to the lowest center index.
    let cn: Vec<f64> = centers.outer_iter().map(|r| r.dot(&r)).collect();
    let assignments: Vec<usize> = (0..n)
        .step_by(MEASURE_JET_ASSIGN_BLOCK_ROWS)
        .flat_map(|start| {
            let end = (start + MEASURE_JET_ASSIGN_BLOCK_ROWS).min(n);
            let g = data.slice(ndarray::s![start..end, ..]).dot(&centers.t());
            let block: Vec<usize> = g
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| {
                    let mut best_j = 0usize;
                    let mut best = f64::INFINITY;
                    for (j, &gij) in row.iter().enumerate() {
                        let s = cn[j] - 2.0 * gij;
                        if s < best {
                            best = s;
                            best_j = j;
                        }
                    }
                    best_j
                })
                .collect();
            block
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
    // Cell barycenters: the first moment of μ on each cell. These are the
    // realized nodes for first-moment-exact lumping.
    let mut barycenter = sums;
    for j in 0..m {
        let count = masses[j] * n as f64;
        if count > 0.0 {
            for k in 0..d {
                barycenter[(j, k)] /= count;
                nodes[(j, k)] = barycenter[(j, k)];
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
/// Per block the closure receives `(scale_idx, eps, q, base)` where `q` is
/// the truncated kernel sum used by the local residual and `base`
/// is the fully-assembled outer weight
/// `log_step · ε^(−η) · net_mass_i · q^(1−2α)`, with
/// `η = 2s + d(2−2α)` for the available dimension parameter, and writes, per requested
/// form, one weight triple `[w_R, w_2, w_3]`. Only `w_R` is live:
/// `R = CᵀWC − B·G⁺·Bᵀ/q`, with `G⁺` the rank-revealing pseudo-inverse.
/// The extra slots are retained for the ψ layout and receive zero local
/// channels because τ no longer changes the energy.
///
/// The outer sum over centers is coarsened per scale to a deterministic
/// ε/2-net with nearest-member mass aggregation (the outer Riemann sum needs
/// resolution ε, not the center-spacing floor), so each scale's cost sits at
/// its own level and the band totals ~O(m²·d) instead of O(L·m³). The inner
/// (local-fit) quadrature always uses the full center set, so the local
/// residual identities (exact constant annihilation, PSD) are untouched.
pub(crate) fn assemble_weighted_forms<F>(
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
    let dist2 = pairwise_sq_dists(centers, centers);

    // One block of `n_forms` m×m accumulators per scale. Each scale's center
    // loop is sequential and the cross-scale sum below runs in band order,
    // so the result is bit-deterministic whether or not the scales
    // themselves run in parallel.
    let assemble_scale = |scale_idx: usize, eps: f64| -> Result<Vec<Array2<f64>>, BasisError> {
        let mut out: Vec<Array2<f64>> =
            (0..n_forms).map(|_| Array2::<f64>::zeros((m, m))).collect();
        let cutoff2 = (MEASURE_JET_PROFILE_CUTOFF * eps) * (MEASURE_JET_PROFILE_CUTOFF * eps);
        let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
        let eta = 2.0 * order_s + (d as f64) * (2.0 - 2.0 * alpha);
        let scale_weight = band.log_step * eps.powf(-eta);
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
            let g_pinv = symmetric_pseudoinverse(&g, "local affine Gram")?;
            let bm = b.dot(&g_pinv);
            let base = scale_weight * net_mass[i] * q.powf(1.0 - 2.0 * alpha);
            weights(scale_idx, eps, q, base, &mut wbuf);
            // Scatter-add Σ_k wbuf[k]·R into each form. The τ channels are
            // zero because the exact projection is τ-independent.
            for (a, &ja) in idx.iter().enumerate() {
                let bma = bm.row(a);
                for (c, &jc) in idx.iter().enumerate() {
                    let b_c = b.row(c);
                    let mut val_r = -w[a] * w[c] / q - bma.dot(&b_c) / q;
                    if a == c {
                        val_r += w[a];
                    }
                    for (k, out_k) in out.iter_mut().enumerate() {
                        let wk = wbuf[k];
                        out_k[(ja, jc)] += wk[0] * val_r;
                    }
                }
            }
        }
        Ok(out)
    };

    let n_scales = band.eps.len();
    let parallel_ok = m
        .saturating_mul(m)
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
    Ok(totals.into_iter().map(|t| (&t + &t.t()) * 0.5).collect())
}

/// The multiscale jet-residual energy `Q` (m × m, symmetric PSD) on the
/// center set. See the module docs for the formula and contracts; the local
/// residual form is assembled through the closed-form identities
///
/// ```text
///   CᵀWC          = W − w·wᵀ/q,
///   B = CᵀWΦ̃     = WΦ − w·aᵀ          (a = Φᵀw/q),
///   G = Φ̃ᵀWΦ̃/q  = (ΦᵀWΦ)/q − a·aᵀ,
///   R_loc         = CᵀWC − B·G⁺·Bᵀ/q,
/// ```
///
/// with `G⁺` realized through the symmetric eigendecomposition and a
/// machine-precision rank cutoff. One walk of [`assemble_weighted_forms`]
/// with the unit weight.
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
    let q = forms.swap_remove(0);
    // The energy `Q = Σ wᵢ Rᵢ` is a nonnegative combination of analytically
    // PSD local residual forms, so it is PSD in exact arithmetic. The affine
    // span is annihilated to machine zero, where roundoff in the per-block
    // pseudo-inverse and the centering cancellation leaves the smallest
    // eigenvalue at ±ε_mach·‖Q‖. Project onto the PSD cone (floor negative
    // eigenvalues at 0) so `vᵀQv ≥ 0` holds exactly for every `v`, including
    // the affine directions the energy must annihilate.
    project_symmetric_psd(q, "measure-jet energy form")
}

/// Project a symmetric matrix onto the PSD cone by flooring its negative
/// eigenvalues at 0. Only sub-machine-precision negative eigenvalues are
/// expected here (the form is analytically PSD); a meaningfully negative
/// eigenvalue would indicate an assembly bug, so it is floored but the
/// reconstruction otherwise preserves the spectrum exactly.
pub(crate) fn project_symmetric_psd(
    a: Array2<f64>,
    label: &str,
) -> Result<Array2<f64>, BasisError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(a);
    }
    let (evals, evecs) = a.eigh(Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!(
            "measure-jet PSD projection `{label}` eigendecomposition failed: {e}"
        ))
    })?;
    if evals.iter().all(|&lam| lam >= 0.0) {
        return Ok(a);
    }
    let mut scaled = evecs.clone();
    for (k, mut col) in scaled.axis_iter_mut(Axis(1)).enumerate() {
        let lam = evals[k].max(0.0);
        col.mapv_inplace(|v| v * lam);
    }
    let psd = scaled.dot(&evecs.t());
    Ok((&psd + &psd.t()) * 0.5)
}

/// The energy together with its exact first and second jets in the live
/// dials, plus zero slots for the retained `ψ_τ = ln τ` coordinate. With
/// `g_s = −2 ln ε`, `g_α = −2 ln q`:
///
/// ```text
///   ∂Q/∂s   = Σ g_s·w·R,        ∂²Q/∂s²   = Σ g_s²·w·R,
///   ∂Q/∂α   = Σ g_α·w·R,        ∂²Q/∂α²   = Σ g_α²·w·R,
///   ∂²Q/∂s∂α = Σ g_s·g_α·w·R,
///   ∂Q/∂ψ_τ = ∂²Q/∂ψ_τ² = ∂²Q/∂s∂ψ_τ = ∂²Q/∂α∂ψ_τ = 0.
/// ```
///
/// all scattered from the SAME local blocks as `Q` in one pass (no second
/// assembly that could drift). FD-gated in this module's tests. Requires
/// `tau0 > 0` only because the retained coordinate is `ln τ`.
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
            "measure-jet jets need tau0 > 0 because the retained τ coordinate is ln τ; got {tau0}"
        );
    }
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
            let intrinsic_dim = centers.ncols() as f64;
            let ga = 2.0 * intrinsic_dim * eps.ln() - 2.0 * q.max(f64::MIN_POSITIVE).ln();
            out[0] = [base, 0.0, 0.0];
            out[1] = [gs * base, 0.0, 0.0];
            out[2] = [gs * gs * base, 0.0, 0.0];
            out[3] = [ga * base, 0.0, 0.0];
            out[4] = [ga * ga * base, 0.0, 0.0];
            out[5] = [gs * ga * base, 0.0, 0.0];
            out[6] = [0.0, 0.0, 0.0];
            out[7] = [0.0, 0.0, 0.0];
            out[8] = [0.0, 0.0, 0.0];
            out[9] = [0.0, 0.0, 0.0];
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
    let forms = measure_jet_energy_forms_per_scale(centers, masses, band, order_s, alpha, tau0)?;
    Ok(forms
        .iter()
        .map(|q_l| values.dot(&q_l.dot(&values)))
        .collect())
}

/// The per-scale energy forms `Q_ℓ` (each m × m, symmetric PSD), with
/// `Σ_ℓ Q_ℓ = Q` exactly (same blocks, one-hot weights). These are the
/// spectral-split carriers: emitted as separate penalty candidates they let
/// the multi-penalty REML engine learn per-level amplitudes λ_ℓ directly —
/// scale adaptivity at ρ-speed with no rebuild and no new optimizer code.
pub fn measure_jet_energy_forms_per_scale(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
) -> Result<Vec<Array2<f64>>, BasisError> {
    let n_scales = band.eps.len();
    assemble_weighted_forms(
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
    )
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
    // Distances once (GEMM), then every band scale reads the same d² row —
    // an L-fold saving over per-scale distance recomputation.
    let d2 = pairwise_sq_dists(queries, centers);
    let mut out = Array2::<f64>::zeros((nq, nl));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(qi, mut row)| {
            let d2_row = d2.row(qi);
            for (li, &eps) in eps_band.iter().enumerate() {
                let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
                let mut acc = 0.0_f64;
                for (j, &dd) in d2_row.iter().enumerate() {
                    acc += masses[j] * (-dd * inv_two_eps2).exp();
                }
                row[li] = acc;
            }
        });
    Ok(out)
}

pub(crate) fn measure_jet_support_means(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    eps_band: &[f64],
) -> Result<Vec<f64>, BasisError> {
    let total_mass = masses.sum();
    if !(total_mass.is_finite() && total_mass > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet support means need positive finite total mass; got {total_mass}"
        );
    }
    let support = measure_jet_support_curve(centers, centers, masses, eps_band)?;
    let mut means = vec![0.0_f64; eps_band.len()];
    for (i, row) in support.rows().into_iter().enumerate() {
        let mass = masses[i];
        for (mean, &q) in means.iter_mut().zip(row.iter()) {
            *mean += mass * q;
        }
    }
    for mean in &mut means {
        *mean /= total_mass;
        if !(*mean).is_finite() || *mean <= 0.0 {
            crate::bail_invalid_basis!(
                "measure-jet support mean must be positive and finite; got {mean}"
            );
        }
    }
    Ok(means)
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
    // One GEMM for every distance, then the Gaussian applied in place — the
    // n×m allocation IS the output, no transient copy.
    let mut out = pairwise_sq_dists(data, centers);
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            row.mapv_inplace(|d2| (-d2 * inv_two_l2).exp());
        });
    Ok(out)
}

/// Rank-revealing ambient-linear head lift `T` (d × head_rank) for the
/// extrapolation null space (#1845).
///
/// The measure-jet energy annihilates ambient-affine functions EXACTLY (the
/// no-mass contract), so the affine functions are the penalty's null space —
/// the directions the fit is free to extend across a training gap. But the
/// Gaussian representer design cannot REPRESENT a global affine function off
/// its support: a finite sum of decaying bumps reverts to the parametric
/// backbone away from the centers, so in a gap the fit collapses toward the
/// training mean instead of carrying the flank-attested trend. Completing the
/// smoothing-spline structure, the builder appends this ambient-linear null
/// space to the design as an UNPENALIZED head (the `{x_1..x_d}` head the frame
/// notes §1 pin as the property the representer basis lacked).
///
/// The head is data-derived and magic-free. Ambient coordinates of data on a
/// low intrinsic-dimension stratum are rank-deficient as linear trends, so the
/// coordinate columns are orthonormalized on the centers and the
/// numerically-degenerate directions dropped. Working in the mean-CENTERED
/// coordinate columns (the mass-weighted mean is the intercept's, not the
/// head's) makes the rank test measure the genuine spread of the centers along
/// each direction rather than its offset; the relative floor
/// [`MEASURE_JET_PSEUDOINVERSE_RTOL`] is the module's own numerical rank
/// tolerance (the same one the local Gram pseudo-inverses use). The returned
/// `T` satisfies `head(points) = points · T` (the mean-centering only informs
/// the keep/drop decision; the constant component of `points · T` is removed
/// downstream by the global parametric orthogonalization). `T` is a
/// deterministic function of the frozen centers + masses, so the frozen replay
/// path reconstructs the identical head with no persisted state.
fn measure_jet_affine_head_transform(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let m = centers.nrows();
    let d = centers.ncols();
    let total_mass = masses.sum();
    // Mass inner product on center values.
    let mdot = |u: &Array1<f64>, v: &Array1<f64>| -> f64 {
        let mut acc = 0.0;
        for i in 0..m {
            acc += masses[i] * u[i] * v[i];
        }
        acc
    };
    // Mean-centered coordinate columns: the mass-weighted mean is removed so the
    // residual mass-norm is the genuine spread of the centers along a direction,
    // not dominated by the coordinate's offset (which the intercept owns).
    let cols: Vec<Array1<f64>> = (0..d)
        .map(|k| {
            let col = centers.column(k).to_owned();
            let mean = if total_mass > 0.0 {
                mdot(&col, &Array1::ones(m)) / total_mass
            } else {
                0.0
            };
            col.mapv(|x| x - mean)
        })
        .collect();
    // Relative numerical rank floor from the centered coordinate-column scale.
    let max_norm = cols.iter().fold(0.0_f64, |acc, c| acc.max(mdot(c, c).sqrt()));
    let drop_below =
        (MEASURE_JET_PSEUDOINVERSE_RTOL * (d.max(1) as f64) * max_norm).max(f64::MIN_POSITIVE);
    // Mass-weighted modified Gram–Schmidt on the centered columns; `t`
    // accumulates the lift in the ORIGINAL coordinate basis, so every kept head
    // column is `points · t_r` (up to the intercept-owned constant).
    let mut q_cols: Vec<Array1<f64>> = Vec::new();
    let mut t_cols: Vec<Array1<f64>> = Vec::new();
    for k in 0..d {
        let mut v = cols[k].clone();
        let mut t = Array1::<f64>::zeros(d);
        t[k] = 1.0;
        for (q, tq) in q_cols.iter().zip(t_cols.iter()) {
            let proj = mdot(q, &v);
            v.scaled_add(-proj, q);
            t.scaled_add(-proj, tq);
        }
        let norm = mdot(&v, &v).sqrt();
        if norm > drop_below {
            v.mapv_inplace(|x| x / norm);
            t.mapv_inplace(|x| x / norm);
            q_cols.push(v);
            t_cols.push(t);
        }
    }
    let head_rank = t_cols.len();
    let mut t_mat = Array2::<f64>::zeros((d, head_rank));
    for (r, t) in t_cols.into_iter().enumerate() {
        t_mat.column_mut(r).assign(&t);
    }
    t_mat
}

/// Resolve the realized representer range ℓ. An explicit positive
/// `spec_length_scale` is used verbatim; the `0.0` sentinel auto-initializes
/// from the median nearest-center spacing (one spacing width: neighbors
/// overlap at exp(−1/2) ≈ 0.61, smooth blend without collinearity).
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
    let dist2 = pairwise_sq_dists(centers, centers);
    let spacing = median_nearest_center_spacing(&dist2)?;
    Ok(MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR * spacing)
}

/// The realized, ψ-FIXED geometry shared by the basis builder and the
/// ψ-derivative producer — ONE realization source, so the penalty the fit
/// uses and the penalty the ψ-channel differentiates can never drift apart
/// (the #901 desync class, excluded structurally).
pub(crate) struct RealizedMeasureJetGeometry {
    pub(crate) centers: Array2<f64>,
    pub(crate) masses: Array1<f64>,
    pub(crate) eps_band: Vec<f64>,
    pub(crate) log_step: f64,
    pub(crate) length_scale: f64,
    /// Assembly order for the energy weights: the realized default in
    /// per-level mode (absorbed per candidate by normalization), the
    /// explicit value in fused mode.
    pub(crate) order_s_eval: f64,
    /// Spectral-split mode marker (`order_s == 0.0` sentinel).
    pub(crate) per_level: bool,
    pub(crate) z: Array2<f64>,
    pub(crate) coefficient_gauge: gam_problem::Gauge,
    pub(crate) kz: Array2<f64>,
    /// Ambient-linear head lift `T` (d × head_rank): the extrapolation
    /// null-space basis appended to the representer design (#1845). The head
    /// columns evaluate as `points · T`; empty (`d × 0`) when the geometry
    /// resolves no supported linear direction. Deterministic in the frozen
    /// centers + masses, so predict-time replay rebuilds it verbatim.
    pub(crate) head_transform: Array2<f64>,
}

pub(crate) fn realize_measure_jet_geometry(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<RealizedMeasureJetGeometry, BasisError> {
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
    // masses, band, support anchors, and normalization scales.
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
    // Ambient-linear extrapolation head (#1845): the raw center space becomes
    // `[ m Gaussian representers | head_rank ambient-linear columns ]`. The head
    // carries the penalty's affine null space explicitly so the fit no longer
    // reverts to the parametric backbone (the training mean) across an
    // unsupported gap.
    // The extrapolation head is the single-scale (fused) gap-bridge path. In
    // multiscale mode the per-scale spectral penalties carry their own
    // structure and the design stays the pure representer basis (the per-level
    // replay + width contracts pin `m − 1` columns), so the head is added only
    // when the term is single-scale.
    let head_transform = if spec.multiscale {
        Array2::<f64>::zeros((centers.ncols(), 0))
    } else {
        measure_jet_affine_head_transform(centers.view(), masses.view())
    };
    let head_rank = head_transform.ncols();
    let m_aug = m + head_rank;
    // Realized-design constraint transform: block sum-to-zero on the
    // representer coefficients composed with an identity pass-through on the
    // head at fit time; the frozen composed `z · z_parametric` at predict time
    // (#532 pattern — see MeasureJetIdentifiability). The head rides the SAME
    // global parametric orthogonalization, so its constant component cannot
    // collide with the intercept.
    let (z, coefficient_gauge) = match &spec.identifiability {
        MeasureJetIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != m_aug {
                crate::bail_dim_basis!(
                    "frozen measure-jet identifiability transform mismatch: {} representers + {} head columns but transform has {} rows",
                    m,
                    head_rank,
                    transform.nrows()
                );
            }
            (
                transform.clone(),
                gam_problem::Gauge::from_block_transforms(&[transform.clone()]),
            )
        }
        MeasureJetIdentifiability::CenterSumToZero => {
            // Householder sum-to-zero on the representers, block-composed with
            // the identity on the head columns. The active projection is owned
            // by Gauge; the dense z is persisted for frozen replay metadata.
            let u = householder_sum_to_zero_u(m);
            let z_rbf = householder_sum_to_zero_z(&u);
            let mut z_block = Array2::<f64>::zeros((m_aug, (m - 1) + head_rank));
            z_block.slice_mut(ndarray::s![..m, ..m - 1]).assign(&z_rbf);
            for r in 0..head_rank {
                z_block[(m + r, (m - 1) + r)] = 1.0;
            }
            (
                z_block.clone(),
                gam_problem::Gauge::from_block_transforms(&[z_block]),
            )
        }
    };
    // Augmented raw center matrix `[K(centers, centers) | centers · T]`, so the
    // restricted `kz` maps constrained coefficients to center nodal values for
    // BOTH the representers and the head; the energy annihilates the head block
    // (affine) to machine precision, so it stays the unpenalized null space.
    let k_cc = measure_jet_design_matrix(centers.view(), centers.view(), length_scale)?;
    let mut k_aug = Array2::<f64>::zeros((m, m_aug));
    k_aug.slice_mut(ndarray::s![.., ..m]).assign(&k_cc);
    if head_rank > 0 {
        let head_cc = centers.dot(&head_transform);
        k_aug.slice_mut(ndarray::s![.., m..]).assign(&head_cc);
    }
    let kz = coefficient_gauge.restrict_design(&k_aug);
    Ok(RealizedMeasureJetGeometry {
        centers,
        masses,
        eps_band,
        log_step,
        length_scale,
        order_s_eval: order_s,
        // multiscale (per-scale spectral) mode is an EXPLICIT opt-in
        // (#1116): single-scale (one fused penalty, the Duchon/Matérn outer
        // footprint) at any center count unless the spec asks for it. No
        // center-count auto-gate.
        per_level: spec.multiscale,
        z,
        coefficient_gauge,
        kz,
        head_transform,
    })
}

/// Whether a measure-jet spec runs in multiscale mode (per-scale spectral
/// penalties + `(α, ln τ)` ψ dials + the affine-preserving ridge). The single
/// source of truth for the mode decision, shared by the builder and the
/// outer-engine enrollment predicates so the penalty count and the ψ-dimension
/// cannot disagree. Multiscale is an explicit opt-in (`spec.multiscale`); there
/// is no center-count auto-gate (#1116).
pub fn measure_jet_multiscale_mode(spec: &MeasureJetBasisSpec) -> bool {
    spec.multiscale
}

/// Build the measure-jet smooth: Gaussian representer design `K(data,
/// centers)·z`, multiscale jet-residual penalty (one candidate per scale in
/// spectral mode, one fused candidate in pinned-order mode), and the
/// replayable [`BasisMetadata::MeasureJet`]. The geometry comes from the
/// empirical measure (centers + masses + band) through the shared
/// realization helper — the same source the ψ-derivative producer uses.
pub fn build_measure_jet_basis(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let RealizedMeasureJetGeometry {
        centers,
        masses,
        eps_band,
        log_step,
        length_scale,
        order_s_eval: order_s,
        per_level,
        z,
        coefficient_gauge,
        kz,
        head_transform,
    } = realize_measure_jet_geometry(data, spec)?;
    let band = MeasureJetBand {
        eps: eps_band.clone(),
        log_step,
    };
    let m = centers.nrows();
    let head_rank = head_transform.ncols();
    let m_aug = m + head_rank;
    // Augmented raw design `[K(data, centers) | data · T]` (#1845): the head
    // columns are the ambient-linear extrapolation basis. The gauge restricts
    // BOTH blocks together, so the frozen composed transform replays the head
    // verbatim at predict time.
    let kernel_design = measure_jet_design_matrix(data, centers.view(), length_scale)?;
    let mut raw_design = Array2::<f64>::zeros((data.nrows(), m_aug));
    raw_design
        .slice_mut(ndarray::s![.., ..m])
        .assign(&kernel_design);
    if head_rank > 0 {
        let head_design = data.dot(&head_transform);
        raw_design
            .slice_mut(ndarray::s![.., m..])
            .assign(&head_design);
    }
    let constrained_design = coefficient_gauge.restrict_design(&raw_design);
    let design = gam_linalg::matrix::DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(constrained_design),
    );
    let support_means = measure_jet_support_means(centers.view(), masses.view(), &eps_band)?;
    // Spectral/geometric split. With the auto order sentinel (order_s == 0.0)
    // the term emits one candidate PER scale: the multi-penalty REML engine
    // then learns the level amplitudes λ_ℓ directly — scale adaptivity at
    // ρ-speed, dead scales REML-deselected (the Duchon-ARD pattern) — and the
    // fitted order is read off the spectrum (ŝ = −½ · slope of ln λ̂_ℓ on
    // ln ε_ℓ) instead of being optimized. An explicit s > 0 pins the Mellin
    // weights and fuses the band into one candidate. The Mellin prefactor
    // ε^(−η)·log_step inside each per-scale form is absorbed by the
    // per-candidate Frobenius normalization, so REML owns the amplitudes
    // outright. The sentinel itself is persisted in the metadata as the mode
    // marker: a replay MUST re-enter the same mode or the penalty count
    // desyncs (the gam#860 trap class).
    let mut candidates = Vec::new();
    let mut penalty_normalization_scales = Vec::new();
    let mut raw_penalty_normalization_scales = Vec::new();
    let mut fused_penalty_normalization_scale = None;
    if per_level {
        let forms = measure_jet_energy_forms_per_scale(
            centers.view(),
            masses.view(),
            &band,
            order_s,
            spec.alpha,
            spec.tau0,
        )?;
        for (level, q_l) in forms.into_iter().enumerate() {
            let s_l = kz.t().dot(&q_l).dot(&kz);
            let (s_norm, c_l) = normalize_penalty(&((&s_l + &s_l.t()) * 0.5));
            let intrinsic_dim = centers.ncols() as f64;
            let eta = 2.0 * order_s + intrinsic_dim * (2.0 - 2.0 * spec.alpha);
            let scale_weight = log_step * eps_band[level].powf(-eta);
            penalty_normalization_scales.push(c_l);
            raw_penalty_normalization_scales.push(c_l / scale_weight);
            candidates.push(PenaltyCandidate {
                matrix: s_norm,
                nullspace_dim_hint: 0,
                source: PenaltySource::Other(format!("measure_jet_scale_{level}")),
                normalization_scale: c_l,
                kronecker_factors: None,
                op: None,
            });
        }
    } else {
        let q_form = measure_jet_energy_form(
            centers.view(),
            masses.view(),
            &band,
            order_s,
            spec.alpha,
            spec.tau0,
        )?;
        let mut penalty = kz.t().dot(&q_form).dot(&kz);
        penalty = (&penalty + &penalty.t()) * 0.5;
        // Single-scale nullspace stabilization, FUSED (not a second REML λ).
        //
        // The fused jet-residual energy penalizes only the LOCAL jet residual,
        // so it has a large near-nullspace of
        // globally-smooth-but-not-locally-affine modes; on low-effective-rank
        // data those weakly-penalized directions are unidentified and the inner
        // solve / outer REML cycle (MEASURED #1116: with NO stabilizer the
        // bms-accuracy fit was 3s→59s and less accurate). The
        // affine-preserving ridge `I − P_affine` stabilizes exactly those
        // directions. But its job is IDENTIFIABILITY, not smoothing selection —
        // it does not need its own learned λ. As a separate candidate it
        // doubled the single-scale outer-REML search into 2-D (the #1116
        // perf-parity 2.8× vs Duchon's 1-λ fit). Instead fold it into the
        // primary at a small FIXED fraction of the primary's Frobenius scale:
        // the near-nullspace gets a definite floor (κ ≈ 1/ε_ridge bounded), the
        // smoothing strength stays owned by the single primary λ, and the outer
        // footprint is one λ — Duchon/Matérn parity. Multiscale mode keeps the
        // ridge as a genuine learned candidate (its per-scale spectrum IS the
        // estimand).
        if spec.double_penalty {
            // With the ambient-linear head carrying the affine null space
            // explicitly (#1845), the representer's own affine-reproducing
            // directions become redundant with the (unpenalized) head: leaving
            // them unpenalized would split the flank trend arbitrarily between a
            // decaying representer combination and the head, and only the head
            // survives across the gap. A plain identity floor on the representer
            // coefficients (`blockdiag(I_m, 0_head)` restricted through the same
            // gauge) breaks that tie deterministically toward the free head —
            // any affine mass placed on the representer now pays a strictly
            // positive toll, so the penalized-least-squares optimum drives it
            // into the head — while still flooring the energy's smooth
            // near-nullspace exactly as the affine-preserving ridge did. Without
            // a head the geometry keeps the original affine-preserving ridge.
            let ridge = if head_rank > 0 {
                let mut r_raw = Array2::<f64>::zeros((m_aug, m_aug));
                for i in 0..m {
                    r_raw[(i, i)] = 1.0;
                }
                coefficient_gauge.restrict_penalty(&r_raw)
            } else {
                affine_preserving_coefficient_ridge(&kz, centers.view(), masses.view())?
            };
            let primary_fro = trace_of_product(&penalty, &penalty).sqrt();
            let ridge_fro = trace_of_product(&ridge, &ridge).sqrt();
            if primary_fro.is_finite()
                && primary_fro > 0.0
                && ridge_fro.is_finite()
                && ridge_fro > 0.0
            {
                let w = MEASURE_JET_FUSED_RIDGE_FRACTION * primary_fro / ridge_fro;
                penalty = &penalty + &(&ridge * w);
            }
        }
        let (penalty_norm, c_primary) = normalize_penalty(&penalty);
        fused_penalty_normalization_scale = Some(c_primary);
        candidates.push(PenaltyCandidate {
            matrix: penalty_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::Primary,
            normalization_scale: c_primary,
            kronecker_factors: None,
            op: None,
        });
    }
    // Multiscale mode: the affine-preserving ridge is a genuine learned
    // candidate (the per-scale split leaves its own spectrum to estimate). In
    // single-scale mode it is fused into the primary above (a fixed
    // identifiability floor, not a second λ) — see that block.
    if spec.double_penalty && per_level {
        let ridge = affine_preserving_coefficient_ridge(&kz, centers.view(), masses.view())?;
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
            // The SPEC's order field, sentinel included: 0.0 marks per-level
            // (spectral) mode and must replay as per-level — persisting the
            // realized default here would silently flip the rebuild into
            // fused mode and desync the penalty count.
            order_s: spec.order_s,
            alpha: spec.alpha,
            tau0: spec.tau0,
            masses,
            support_means,
            penalty_normalization_scales,
            raw_penalty_normalization_scales,
            fused_penalty_normalization_scale,
            constraint_transform: Some(z),
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

/// Exact ψ-jets of the REALIZED measure-jet penalty candidates, adapted to
/// the anisotropic group-ψ carrier the spatial optimizer consumes.
///
/// Coordinates (the layout contract for the registration arm):
/// - per-level (spectral) mode, `order_s == 0.0`: `[α, ln τ]` — the order is
///   absorbed by the REML-learned per-scale amplitudes and is NOT a ψ
///   coordinate; `ln τ` is retained as an inert coordinate with zero
///   derivatives because the exact affine projection is τ-independent;
/// - fused (pinned-order) mode: `[s, α, ln τ]`.
///
/// Design drift is identically zero for every coordinate (the Gaussian
/// representer is ψ-fixed), so `design_first`/`design_second_diag` are
/// correctly-shaped zero matrices and there are no design cross terms.
/// Penalty derivatives are routed through the SAME constrained Frobenius
/// normalization as the fit-time candidates
/// (`normalize_penaltywith_psi_derivatives` + the cross rule), so criterion
/// value and criterion derivative share one normalization — the #901 lesson
/// made structural. The affine-preserving ridge candidate (when
/// `double_penalty` is on) carries identically-zero derivatives. The
/// per-candidate layout follows the builder's ORIGINAL candidate order
/// (scale candidates then ridge / primary then ridge); consumers align to
/// the FITTED penalty list via
/// `PenaltyInfo.original_index` when the active-candidate filter dropped
/// any.
pub fn build_measure_jet_basis_psi_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    if !(spec.tau0.is_finite() && spec.tau0 > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet ψ derivatives need tau0 > 0 because the retained τ coordinate is ln τ; got {}",
            spec.tau0
        );
    }
    let geom = realize_measure_jet_geometry(data, spec)?;
    let band = MeasureJetBand {
        eps: geom.eps_band.clone(),
        log_step: geom.log_step,
    };
    let n = data.nrows();
    let p = geom.kz.ncols(); // design width: constrained Gaussian-coefficient space
    let kz = &geom.kz;
    let sandwich = |j: &Array2<f64>| {
        let s = kz.t().dot(j).dot(kz);
        (&s + &s.t()) * 0.5
    };
    // Raw (pre-normalization) value + jet stacks per ORIGINAL candidate, in
    // coordinate order. `raw[cand] = (S, firsts, seconds, crosses)` with
    // crosses keyed by coordinate pairs in `pairs` order.
    let (n_coords, pairs, raw): (
        usize,
        Vec<(usize, usize)>,
        Vec<(
            Array2<f64>,
            Vec<Array2<f64>>,
            Vec<Array2<f64>>,
            Vec<Array2<f64>>,
        )>,
    ) = if geom.per_level {
        let l_count = band.eps.len();
        // Six forms per scale: value, ∂α, ∂α², and zero τ slots — same
        // blocks, one walk (single-source rule).
        let forms = assemble_weighted_forms(
            geom.centers.view(),
            geom.masses.view(),
            &band,
            geom.order_s_eval,
            spec.alpha,
            spec.tau0,
            6 * l_count,
            3,
            &|scale_idx, eps: f64, q: f64, base: f64, out: &mut [[f64; 3]]| {
                for slot in out.iter_mut() {
                    *slot = [0.0, 0.0, 0.0];
                }
                let intrinsic_dim = geom.centers.ncols() as f64;
                let ga = 2.0 * intrinsic_dim * eps.ln() - 2.0 * q.max(f64::MIN_POSITIVE).ln();
                let k0 = 6 * scale_idx;
                out[k0] = [base, 0.0, 0.0];
                out[k0 + 1] = [ga * base, 0.0, 0.0];
                out[k0 + 2] = [ga * ga * base, 0.0, 0.0];
                out[k0 + 3] = [0.0, 0.0, 0.0];
                out[k0 + 4] = [0.0, 0.0, 0.0];
                out[k0 + 5] = [0.0, 0.0, 0.0];
            },
        )?;
        let mut raw = Vec::with_capacity(l_count);
        for level in 0..l_count {
            let chunk = &forms[6 * level..6 * level + 6];
            raw.push((
                sandwich(&chunk[0]),
                vec![sandwich(&chunk[1]), sandwich(&chunk[3])],
                vec![sandwich(&chunk[2]), sandwich(&chunk[4])],
                vec![sandwich(&chunk[5])],
            ));
        }
        (2usize, vec![(0usize, 1usize)], raw)
    } else {
        // Single-scale mode (the default): the penalty dials (s, α, lnτ) are NOT
        // enrolled — the fused penalty is fixed and only the design-moving ℓ
        // coordinate (added below) varies. So there are ZERO penalty-derivative
        // COORDINATES (n_coords = 0), but the term still has ONE fitted penalty
        // candidate (the fused primary), so we emit one `raw` entry with EMPTY
        // per-coordinate derivative vecs. That keeps the candidate COUNT
        // (n_active = 1) aligned with the build's penalty list while contributing
        // no penalty ψ-derivative (the penalty is ℓ-independent). MUST match
        // `measure_jet_penalty_psi_dim` (= 0 coords) and the build's 1 candidate.
        let q_value = sandwich(&measure_jet_energy_form(
            geom.centers.view(),
            geom.masses.view(),
            &band,
            geom.order_s_eval,
            spec.alpha,
            spec.tau0,
        )?);
        let raw = vec![(q_value, Vec::new(), Vec::new(), Vec::new())];
        (0usize, Vec::new(), raw)
    };
    // Design-moving ℓ coordinate (#1116). The Gaussian representer design
    // `X = K(data, centers; ℓ)·z` is the ONLY ℓ-dependent object (the jet-energy
    // penalty uses the ε-band, not ℓ; the constraint transform `z` acts in
    // center-coefficient space). With `u = lnℓ` and `a = r²/ℓ²`:
    //   ∂K/∂u = K ⊙ a,     ∂²K/∂u² = K ⊙ (a² − 2a),
    // and `∂X/∂u = (∂K/∂u)·z`, `∂²X/∂u² = (∂²K/∂u²)·z`. Cross terms with the
    // (s, α, lnτ) penalty dials vanish: those carry zero design drift and the
    // penalty carries zero ℓ drift. FD-gated by `psi_producer_matches_fd_length_scale`.
    let length_scale_design = if spec.learn_length_scale {
        let ell = geom.length_scale;
        let k = measure_jet_design_matrix(data, geom.centers.view(), ell)?;
        let r2 = pairwise_sq_dists(data, geom.centers.view());
        let inv_l2 = 1.0 / (ell * ell);
        let mut dk = k.clone();
        let mut d2k = k.clone();
        for ((dk_v, d2k_v), &r2_v) in dk.iter_mut().zip(d2k.iter_mut()).zip(r2.iter()) {
            let a = r2_v * inv_l2;
            let kij = *dk_v;
            *dk_v = kij * a;
            *d2k_v = kij * (a * a - 2.0 * a);
        }
        // The ambient-linear head (#1845) is ℓ-independent, so its derivative
        // columns are exactly zero: pad the representer derivatives to the
        // augmented raw width before restricting through the SAME Gauge-owned
        // coefficient section the design uses.
        let m = geom.centers.nrows();
        let m_aug = m + geom.head_transform.ncols();
        let mut dk_aug = Array2::<f64>::zeros((data.nrows(), m_aug));
        let mut d2k_aug = Array2::<f64>::zeros((data.nrows(), m_aug));
        dk_aug.slice_mut(ndarray::s![.., ..m]).assign(&dk);
        d2k_aug.slice_mut(ndarray::s![.., ..m]).assign(&d2k);
        let dx_du = geom.coefficient_gauge.restrict_design(&dk_aug);
        let d2x_du2 = geom.coefficient_gauge.restrict_design(&d2k_aug);
        Some((dx_du, d2x_du2))
    } else {
        None
    };
    let n_active = raw.len();
    // Mirror the build-path gate (`build_measure_jet_basis`): the
    // affine-preserving ridge is a SEPARATE candidate only in multiscale mode.
    // In single-scale mode it is FUSED into the primary value above (#1116), so
    // no separate candidate — the ψ-derivative penalty count must match the
    // build path, count desync = the gam#860 trap class.
    let ridge = spec.double_penalty && geom.per_level;
    let n_cands = n_active + usize::from(ridge);
    let zero_p = || Array2::<f64>::zeros((p, p));
    let mut penalties_first: Vec<Vec<Array2<f64>>> =
        (0..n_coords).map(|_| Vec::with_capacity(n_cands)).collect();
    let mut penalties_second_diag: Vec<Vec<Array2<f64>>> =
        (0..n_coords).map(|_| Vec::with_capacity(n_cands)).collect();
    // Cross matrices per pair per candidate, precomputed eagerly (the
    // candidate count is the band length, not the data size) and served
    // through the on-demand provider.
    let mut crosses: Vec<Vec<Array2<f64>>> = (0..pairs.len()).map(|_| Vec::new()).collect();
    for (s_raw, firsts, seconds, cross_raw) in &raw {
        // ONE Frobenius scale per candidate, fixed up front from `s_raw`
        // alone: c anchors the value and every derivative of this candidate.
        // `normalize_penaltywith_psi_derivatives` recomputes the identical c
        // per coordinate (same trace_of_product + sqrt on the same `s_raw`),
        // and its degenerate convention is mirrored here: ‖S‖_F ≤ 1e-12 (or
        // non-finite) reports scale 1.0 — the value passes through unscaled,
        // and the cross helper receives that same 1.0, never a collapsed
        // near-zero scale.
        let fro = trace_of_product(s_raw, s_raw).sqrt();
        let c = if fro.is_finite() && fro > 1e-12 {
            fro
        } else {
            1.0
        };
        for coord in 0..n_coords {
            let (_, s_first, s_second, _) =
                normalize_penaltywith_psi_derivatives(s_raw, &firsts[coord], &seconds[coord]);
            penalties_first[coord].push(s_first);
            penalties_second_diag[coord].push(s_second);
        }
        for (pair_idx, &(a, b)) in pairs.iter().enumerate() {
            let cross_raw_mat = normalize_penalty_cross_psi_derivative(
                s_raw,
                &firsts[a],
                &firsts[b],
                &cross_raw[pair_idx],
                c,
            );
            crosses[pair_idx].push(cross_raw_mat);
        }
    }
    if ridge {
        for coord in 0..n_coords {
            penalties_first[coord].push(zero_p());
            penalties_second_diag[coord].push(zero_p());
        }
        for pair_crosses in crosses.iter_mut() {
            pair_crosses.push(zero_p());
        }
    }
    // Prepend the design-moving ℓ coordinate at index 0 when enrolled. ℓ
    // carries the nonzero `design_first/second`, ZERO penalty derivatives (the
    // penalty is ℓ-independent), and no cross terms with the penalty dials
    // (those carry zero design drift). Every penalty-dial coordinate and cross
    // pair therefore shifts up by one.
    let coord_offset = usize::from(length_scale_design.is_some());
    if coord_offset == 1 {
        penalties_first.insert(0, (0..n_cands).map(|_| zero_p()).collect());
        penalties_second_diag.insert(0, (0..n_cands).map(|_| zero_p()).collect());
    }
    let n_coords_total = n_coords + coord_offset;
    // Penalty-dial cross pairs, shifted past the ℓ coordinate.
    let mut all_pairs: Vec<(usize, usize)> = pairs
        .iter()
        .map(|&(a, b)| (a + coord_offset, b + coord_offset))
        .collect();
    let mut all_crosses: Vec<Vec<Array2<f64>>> = crosses;
    // ℓ (coord 0) × every penalty coordinate: the penalty is ℓ-independent and
    // the penalty dials carry zero design drift, so each cross block is zero —
    // but the pair MUST be registered (the consumer queries every coordinate
    // pair `(a, b)`), else the provider errors on an unregistered ℓ pair.
    if coord_offset == 1 {
        for c in 1..n_coords_total {
            all_pairs.push((0, c));
            all_crosses.push((0..n_cands).map(|_| zero_p()).collect());
        }
    }
    let pair_index: Vec<((usize, usize), Vec<Array2<f64>>)> = all_pairs
        .iter()
        .copied()
        .zip(all_crosses.into_iter())
        .collect();
    let shifted_pairs = all_pairs;
    let provider = AnisoPenaltyCrossProvider::new(move |a, b| {
        pair_index
            .iter()
            .find(|((pa, pb), _)| (*pa, *pb) == (a, b) || (*pa, *pb) == (b, a))
            .map(|(_, mats)| mats.clone())
            .ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "measure-jet ψ cross derivative requested for unknown pair ({a}, {b})"
                ))
            })
    });
    let mut design_first: Vec<Array2<f64>> = (0..n_coords_total)
        .map(|_| Array2::<f64>::zeros((n, p)))
        .collect();
    let mut design_second_diag: Vec<Array2<f64>> = (0..n_coords_total)
        .map(|_| Array2::<f64>::zeros((n, p)))
        .collect();
    if let Some((dx_du, d2x_du2)) = length_scale_design {
        design_first[0] = dx_du;
        design_second_diag[0] = d2x_du2;
    }
    Ok(AnisoBasisPsiDerivatives {
        design_first,
        design_second_diag,
        design_second_cross: Vec::new(),
        design_second_cross_pairs: Vec::new(),
        penalties_first,
        penalties_second_diag,
        penalties_cross_pairs: shifted_pairs,
        penalties_cross_provider: Some(provider),
        implicit_operator: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(crate) fn two_cluster_centers() -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
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
        let masses = ndarray::Array1::<f64>::from_elem(m, 1.0 / m as f64);
        (centers, masses)
    }
    use ndarray::array;

    pub(crate) fn band_for(centers: &Array2<f64>) -> MeasureJetBand {
        measure_jet_band(centers.view(), 0).expect("band")
    }

    /// The no-mass contract: constants must be annihilated to machine
    /// precision at every scale (the constant is projected, never ridged).
    #[test]
    pub(crate) fn energy_form_annihilates_constants_exactly() {
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

    /// The default local projection annihilates ambient affine functions
    /// exactly; τ is retained for ψ layout but no longer adds an affine toll.
    #[test]
    pub(crate) fn energy_form_annihilates_affine_at_default_tau() {
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
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
            .expect("energy form");
        let e_affine = affine.dot(&q.dot(&affine));
        let e_rough = rough.dot(&q.dot(&rough));
        assert!(e_rough > 0.0, "rough vector must pay energy");
        assert!(
            e_affine.abs() <= 1e-12 * e_rough,
            "default affine energy {e_affine:.3e} vs rough {e_rough:.3e}"
        );
    }

    /// PSD: the energy is a sum of weighted least-squares residuals.
    #[test]
    pub(crate) fn energy_form_is_psd() {
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
    pub(crate) fn rough_vector_pays_more_than_smooth() {
        let m = 24usize;
        let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
            let t = i as f64 / (m as f64 - 1.0);
            if k == 0 {
                t * 4.0
            } else {
                0.3 * (t * 4.0).sin()
            }
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

    /// The exact (s, α) jets and zero τ slots must match central finite
    /// differences of the energy — the FD gate the ψ-channel stage will
    /// inherit (the discipline whose absence is exactly the
    /// objective↔gradient desync bug class).
    #[test]
    pub(crate) fn energy_jets_match_finite_differences() {
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
            assert!(
                (a - b).abs() <= 1e-14 * (1.0 + b.abs()),
                "Q drift {a} vs {b}"
            );
        }
        let lt0 = tau.ln();
        let q_at_lt = |lt: f64| {
            measure_jet_energy_form(centers.view(), masses.view(), &band, s0, a0, lt.exp())
                .expect("energy form")
        };
        // FD step calibrated for the SECOND differences: their roundoff
        // floor is ~4·ε_f64·scale/h² (assembly noise amplified by 1/h²), so
        // h = 1e-4 ≈ ε^(1/4) balances it against the O(h²) truncation —
        // both land ≥3 orders below the unchanged 5e-5·scale gate. h = 1e-5
        // sits ON the roundoff floor and fails spuriously.
        let h = 1e-4;
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
                    measure_jet_energy_form(centers.view(), masses.view(), &band, s, a0, lt.exp())
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
                    measure_jet_energy_form(centers.view(), masses.view(), &band, s0, a, lt.exp())
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
    pub(crate) fn scale_spectrum_sums_to_total_and_localizes_roughness() {
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
    pub(crate) fn support_curve_separates_on_web_from_off_web() {
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

    /// The default is single-scale mode at ANY center count: ONE fused penalty,
    /// the same one-λ outer footprint as Duchon/Matérn. Multiscale (the
    /// per-scale spectral split + ψ dials + the affine-preserving ridge) is an
    /// EXPLICIT opt-in (`spec.multiscale`, the DSL `mjs(…, multiscale=true)`) —
    /// there is no center-count auto-gate (#1116). `measure_jet_multiscale_mode`
    /// is the single source for this decision.
    #[test]
    pub(crate) fn default_stays_single_scale_until_multiscale_opt_in() {
        let n = 200usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 {
                t * 3.0
            } else {
                0.4 * (t * 3.0).sin()
            }
        });
        // Default (multiscale = false) stays single-scale even at a LARGE center
        // count that, under the deleted auto-gate, would have flipped to
        // multiscale: exactly ONE candidate — the fused jet-energy penalty with
        // the affine-preserving nullspace ridge folded in at a fixed fraction
        // (#1116), the one-λ Duchon/Matérn footprint.
        let single = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 80 },
            ..MeasureJetBasisSpec::default()
        };
        assert!(
            !measure_jet_multiscale_mode(&single),
            "default must resolve to single-scale at any center count"
        );
        let built_single =
            build_measure_jet_basis(data.view(), &single).expect("single-scale build");
        assert_eq!(
            built_single.penalties.len(),
            1,
            "single-scale mode emits one fused penalty (ridge folded in, not a 2nd λ)"
        );
        // The explicit opt-in flips to multiscale at the SAME center count: the
        // per-scale spectral split (several candidates) + the ridge, strictly
        // more candidates than the single fused penalty.
        let multi = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 80 },
            multiscale: true,
            ..MeasureJetBasisSpec::default()
        };
        assert!(
            measure_jet_multiscale_mode(&multi),
            "multiscale=true must resolve to multiscale mode"
        );
        let built_multi = build_measure_jet_basis(data.view(), &multi).expect("multiscale build");
        assert!(
            built_multi.penalties.len() > built_single.penalties.len(),
            "multiscale mode emits the per-scale spectral split plus the ridge, got {} (vs single-scale {})",
            built_multi.penalties.len(),
            built_single.penalties.len()
        );
    }

    /// An explicit order pins the Mellin weights and fuses the band into a
    /// single Primary candidate. With multiscale off (the default) the mode is
    /// single-scale: one fused Primary penalty with the affine-preserving
    /// nullspace ridge folded in (#1116) = exactly one candidate.
    #[test]
    pub(crate) fn fused_mode_emits_single_primary_candidate() {
        let n = 40usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 {
                t * 3.0
            } else {
                0.4 * (t * 3.0).sin()
            }
        });
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 14 },
            order_s: 1.3,
            ..MeasureJetBasisSpec::default()
        };
        let built = build_measure_jet_basis(data.view(), &spec).expect("fused build");
        assert_eq!(
            built.penalties.len(),
            1,
            "fused single-scale mode emits exactly one Primary candidate (ridge folded in)"
        );
        let BasisMetadata::MeasureJet { order_s, .. } = &built.metadata else {
            panic!("measure-jet build must return MeasureJet metadata");
        };
        assert_eq!(*order_s, 1.3, "explicit order must persist verbatim");
    }

    /// The Householder basis must be orthonormal with sum-to-zero columns.
    #[test]
    pub(crate) fn householder_sum_to_zero_basis_is_orthonormal() {
        let m = 9usize;
        let u = householder_sum_to_zero_u(m);
        let z = householder_sum_to_zero_z(&u);
        for j in 0..(m - 1) {
            let col_j = z.column(j);
            assert!(col_j.sum().abs() <= 1e-12, "column {j} must sum to zero");
            for j2 in j..(m - 1) {
                let dot = col_j.dot(&z.column(j2));
                let want = if j == j2 { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() <= 1e-12,
                    "orthonormality failure at ({j}, {j2}): {dot}"
                );
            }
        }
    }

    /// Frozen-geometry fixture shared by the ψ-producer FD gates: build
    /// once, pin everything (nodes, masses, band, transform, realized ℓ),
    /// and return the pinned spec so dial-perturbed rebuilds move ONLY the
    /// dials — the per-trial contract the optimizer relies on.
    pub(crate) fn frozen_spec_fixture(
        order_s: f64,
        multiscale: bool,
    ) -> (Array2<f64>, MeasureJetBasisSpec) {
        // Multiscale (per-scale + ψ) mode is the explicit opt-in (#1116); the
        // per-level fixture passes `multiscale = true`, the fused fixture
        // `false`. A large center count is kept so the multiscale spectrum is
        // identifiable when opted in.
        let n = 140usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 {
                t * 3.0
            } else {
                0.5 * (t * 3.0).cos() + if i % 9 == 0 { 0.8 } else { 0.0 }
            }
        });
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 70 },
            order_s,
            multiscale,
            // These fixtures gate the PENALTY-dial derivatives; freeze ℓ so the
            // coordinate layout is exactly the penalty dials (the design-moving
            // ℓ dial has its own FD gate, `psi_producer_matches_fd_length_scale`).
            learn_length_scale: false,
            ..MeasureJetBasisSpec::default()
        };
        let first = build_measure_jet_basis(data.view(), &spec).expect("fixture build");
        let BasisMetadata::MeasureJet {
            centers,
            length_scale,
            eps_band,
            masses,
            support_means,
            penalty_normalization_scales,
            raw_penalty_normalization_scales,
            fused_penalty_normalization_scale,
            constraint_transform,
            ..
        } = &first.metadata
        else {
            panic!("measure-jet build must return MeasureJet metadata");
        };
        let frozen = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            order_s,
            alpha: spec.alpha,
            tau0: spec.tau0,
            num_scales: eps_band.len(),
            length_scale: *length_scale,
            double_penalty: spec.double_penalty,
            learn_length_scale: false,
            multiscale,
            identifiability: MeasureJetIdentifiability::FrozenTransform {
                transform: constraint_transform.clone().expect("fit-time z"),
            },
            frozen_quadrature: Some(MeasureJetFrozenQuadrature {
                masses: masses.clone(),
                eps_band: eps_band.clone(),
                support_means: support_means.clone(),
                penalty_normalization_scales: penalty_normalization_scales.clone(),
                raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
                fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            }),
        };
        (data, frozen)
    }

    /// ψ-producer vs central finite differences of the NORMALIZED fit-time
    /// candidates under frozen geometry — per-level mode (coords α, lnτ).
    /// This is the end-to-end gate #901 never had: the derivative is checked
    /// against the exact object the optimizer consumes.
    #[test]
    pub(crate) fn psi_producer_matches_fd_per_level_mode() {
        let (data, frozen) = frozen_spec_fixture(0.0, true);
        let derivs =
            build_measure_jet_basis_psi_derivatives(data.view(), &frozen).expect("psi derivatives");
        let l_count = frozen
            .frozen_quadrature
            .as_ref()
            .expect("frozen quadrature")
            .eps_band
            .len();
        assert_eq!(
            derivs.penalties_first.len(),
            2,
            "per-level coords are (α, lnτ)"
        );
        assert_eq!(derivs.penalties_first[0].len(), l_count + 1);
        assert_eq!(derivs.penalties_cross_pairs, vec![(0, 1)]);
        let pen_at = |alpha: f64, tau0: f64| {
            let trial = MeasureJetBasisSpec {
                alpha,
                tau0,
                ..frozen.clone()
            };
            build_measure_jet_basis(data.view(), &trial)
                .expect("trial build")
                .penalties
        };
        // Second-difference-optimal step (see the jets FD test): the 4-point
        // cross stencil shares the ~ε·scale/h² roundoff floor.
        let h = 1e-4;
        let (a0, t0) = (frozen.alpha, frozen.tau0);
        let ap = pen_at(a0 + h, t0);
        let am = pen_at(a0 - h, t0);
        let tp = pen_at(a0, t0 * h.exp());
        let tm = pen_at(a0, t0 * (-h).exp());
        assert_eq!(
            ap.len(),
            l_count + 1,
            "fixture must keep every scale active"
        );
        for level in 0..l_count {
            let fd_alpha = (&ap[level] - &am[level]) / (2.0 * h);
            let fd_tau = (&tp[level] - &tm[level]) / (2.0 * h);
            for (name, analytic, fd) in [
                ("alpha", &derivs.penalties_first[0][level], fd_alpha),
                ("ln_tau", &derivs.penalties_first[1][level], fd_tau),
            ] {
                let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
                for (x, y) in analytic.iter().zip(fd.iter()) {
                    assert!(
                        (x - y).abs() <= 5e-5 * scale,
                        "{name} jet of scale-candidate {level}: analytic {x:.6e} vs FD {y:.6e}"
                    );
                }
            }
        }
        // The ridge candidate carries identically-zero derivatives.
        for coord in 0..2 {
            assert!(
                derivs.penalties_first[coord][l_count]
                    .iter()
                    .all(|v| *v == 0.0),
                "ridge candidate must have zero ψ drift"
            );
        }
        // Cross derivative through the provider, against a 4-point FD.
        let provider = derivs
            .penalties_cross_provider
            .as_ref()
            .expect("cross provider");
        let cross = provider.evaluate(0, 1).expect("cross pair (α, lnτ)");
        let pp = pen_at(a0 + h, t0 * h.exp());
        let pm = pen_at(a0 + h, t0 * (-h).exp());
        let mp = pen_at(a0 - h, t0 * h.exp());
        let mm = pen_at(a0 - h, t0 * (-h).exp());
        for level in 0..l_count {
            let fd = (&(&pp[level] - &pm[level]) - &(&mp[level] - &mm[level])) / (4.0 * h * h);
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (x, y) in cross[level].iter().zip(fd.iter()) {
                assert!(
                    (x - y).abs() <= 5e-4 * scale,
                    "cross (α, lnτ) jet of scale-candidate {level}: analytic {x:.6e} vs FD {y:.6e}"
                );
            }
        }
    }

    /// Design-moving ℓ dial (#1116): the producer's `design_first[0]` /
    /// `design_second_diag[0]` must match central finite differences of the
    /// REBUILT design `X(ℓ)` under frozen geometry, and the ℓ coordinate must
    /// carry IDENTICALLY ZERO penalty derivatives (the jet-energy penalty is
    /// ℓ-independent). This is the FD gate matérn's log_kappa has and mjs lacked.
    #[test]
    pub(crate) fn psi_producer_matches_fd_length_scale() {
        // Single-scale with opt-in ℓ learning; frozen geometry so only ℓ moves
        // across the FD trials.
        let (data, mut frozen) = frozen_spec_fixture(0.0, false);
        frozen.learn_length_scale = true;
        let derivs =
            build_measure_jet_basis_psi_derivatives(data.view(), &frozen).expect("psi derivatives");
        // ℓ is the only coordinate in single-scale + learn_length_scale.
        assert_eq!(
            derivs.design_first.len(),
            1,
            "single-scale + learn_length_scale enrolls exactly the ℓ coordinate"
        );
        // The ℓ coordinate carries one (all-zero) penalty candidate: the penalty
        // is ℓ-independent, so its ψ-derivative is identically zero.
        assert_eq!(
            derivs.penalties_first[0].len(),
            1,
            "one fitted penalty candidate"
        );
        assert!(
            derivs.penalties_first[0][0].iter().all(|v| *v == 0.0)
                && derivs.penalties_second_diag[0][0].iter().all(|v| *v == 0.0),
            "the jet-energy penalty must not move with ℓ"
        );
        // Rebuild the DESIGN at ℓ·e^{±h} (perturb via spec.length_scale; the
        // realized resolver uses an explicit positive length_scale verbatim).
        let ell0 = frozen.length_scale;
        let design_at = |ell: f64| {
            let trial = MeasureJetBasisSpec {
                length_scale: ell,
                ..frozen.clone()
            };
            build_measure_jet_basis(data.view(), &trial)
                .expect("trial build")
                .design
                .to_dense()
        };
        let h: f64 = 1e-4;
        let x_plus = design_at(ell0 * h.exp());
        let x_minus = design_at(ell0 * (-h).exp());
        let x_0 = design_at(ell0);
        let fd_first = (&x_plus - &x_minus) / (2.0 * h);
        let fd_second = (&x_plus - &(&x_0 * 2.0) + &x_minus) / (h * h);
        let scale1 = fd_first.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
        for (x, y) in derivs.design_first[0].iter().zip(fd_first.iter()) {
            assert!(
                (x - y).abs() <= 5e-5 * scale1,
                "∂X/∂lnℓ: analytic {x:.6e} vs FD {y:.6e}"
            );
        }
        let scale2 = fd_second.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
        for (x, y) in derivs.design_second_diag[0].iter().zip(fd_second.iter()) {
            assert!(
                (x - y).abs() <= 1e-3 * scale2,
                "∂²X/∂lnℓ²: analytic {x:.6e} vs FD {y:.6e}"
            );
        }
    }

    /// Quadrature nodes must be the mass-weighted cell barycenters
    /// (first-moment-exact lumping), with empty cells keeping their seed
    /// coordinates at zero mass.
    #[test]
    pub(crate) fn quadrature_nodes_are_cell_barycenters() {
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
        // Cell 0 barycenter = (0.2, 0.0).
        assert_eq!(nodes[(0, 0)], 0.2);
        assert_eq!(nodes[(0, 1)], 0.0);
        // Cell 1 barycenter = (10.0, 10.0), which is not a sampled row.
        assert_eq!(nodes[(1, 0)], 10.0);
        assert_eq!(nodes[(1, 1)], 10.0);
        // Empty cell keeps its seed coordinates.
        assert_eq!(nodes[(2, 0)], -50.0);
        assert_eq!(nodes[(2, 1)], -50.0);
    }

    /// Freeze→replay: rebuilding from the first build's frozen transform and
    /// frozen quadrature must reproduce design and penalty bit-for-bit (the
    /// predict-path contract).
    #[test]
    pub(crate) fn build_replay_roundtrip_reproduces_design_and_penalty() {
        // A bent filament with a side cluster; multiscale opt-in so this
        // exercises the per-scale (spectral) replay path (#1116).
        let n = 140usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 {
                t * 3.0
            } else {
                0.5 * (t * 3.0).cos() + if i % 9 == 0 { 0.8 } else { 0.0 }
            }
        });
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 70 },
            multiscale: true,
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
            support_means,
            penalty_normalization_scales,
            raw_penalty_normalization_scales,
            fused_penalty_normalization_scale,
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
            learn_length_scale: spec.learn_length_scale,
            multiscale: spec.multiscale,
            identifiability: MeasureJetIdentifiability::FrozenTransform {
                transform: constraint_transform.clone().expect("fit-time z"),
            },
            frozen_quadrature: Some(MeasureJetFrozenQuadrature {
                masses: masses.clone(),
                eps_band: eps_band.clone(),
                support_means: support_means.clone(),
                penalty_normalization_scales: penalty_normalization_scales.clone(),
                raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
                fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            }),
        };
        // Per-level (auto-sentinel) mode: one candidate per band scale plus
        // the ridge, and the count must survive freeze→replay bit-for-bit.
        assert_eq!(
            first.penalties.len(),
            eps_band.len() + 1,
            "per-level mode must emit one candidate per scale + ridge"
        );
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
