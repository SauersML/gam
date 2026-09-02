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
//!   single-scale — one jet-energy Primary at the auto order with no energy ψ
//!   dials. The independent function-space null-component candidate requested
//!   by `double_penalty` is present in either mode and has its own REML λ. The
//!   flag is persisted on the spec, so freeze→replay re-enters the same mode
//!   verbatim.
//!
//! # ψ-differentiability contract (what the ψ-channel stage consumes)
//!
//! Mirroring the constant-curvature κ-contract (#944): centers, masses, the
//! band are deliberately hyperparameter-FIXED at build time; the representer
//! range ℓ is the ONE design-moving dial, and it is REML-selected by default
//! (#1116, restored in #2761) — λ shrinks inside a span and cannot move one, so
//! the range that decides WHICH span the representers occupy has to be chosen
//! by the criterion, exactly as the Matérn κ is. Consequences:
//!
//! - **Penalty-dial design drift is identically zero**: the (s, α, τ) dials
//!   reweight only the jet-energy penalty, never the Gaussian representer
//!   design (`∂X/∂{s,α,τ} ≡ 0`), so those channels are penalty-only
//!   (`is_penalty_like` auto-derives true in the outer engine's
//!   `DirectionalHyperParam`).
//! - **The representer range ℓ is a design-and-pullback-moving dial** (matérn's
//!   `log_kappa` analog, #1116): `X = K(data, centers; ℓ)·z` and the center
//!   evaluation map `E = K(centers, centers; ℓ)·z` both depend on ℓ. The
//!   center-value forms `Q` and `H₀` are ℓ-invariant, but their coefficient
//!   pullbacks `EᵀQE` and `EᵀH₀E` are not; exact product-rule jets are shipped
//!   alongside the design jets. ℓ rebuilds the design per outer trial; it does
//!   not change the frozen basis rank. FD-gated by
//!   `psi_producer_matches_fd_length_scale`. Frozen only where a design-moving
//!   kernel scale on covariates SHARED by two coupled blocks is an
//!   identifiability hazard — the BMS marginal/slope pair, at its own entry
//!   point (`freeze_measure_jet_length_scale_learning`, #1116/`a3afd17a2`) —
//!   and where the user pins `length_scale=` outright.
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

use gam_linalg::faer_ndarray::{
    FaerEigh, FaerSvd, default_rrqr_rank_alpha, rrqr_nullspace_basis,
};

use super::{
    AnisoBasisPsiDerivatives, AnisoPenaltyCrossProvider, BasisBuildResult, BasisError,
    BasisMetadata, CenterStrategy, ConstructiveQuadratic, PenaltyCandidate, PenaltySource,
    filter_penalty_candidates, normalize_penalty, normalize_penalty_cross_psi_derivative,
    normalize_penaltywith_psi_derivatives, select_centers_by_strategy, trace_of_product,
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

/// Representer-range multiple of the median nearest-center spacing used by the
/// `0.0` auto sentinel.
///
/// **This is the SEED of the ℓ outer coordinate, not the realized range.**
/// [`MeasureJetBasisSpec::learn_length_scale`] is on by default, so what a fit
/// ships is whatever REML certifies from here; this constant only has to put
/// the optimizer somewhere feasible and well-conditioned. It is a starting
/// point in exactly the sense the Matérn `MaternLengthScale::Auto` seed is one.
///
/// Set to ×1: a Gaussian representer of range `ℓ = h` (the median
/// nearest-center spacing) already overlaps its neighbors at
/// `exp(−h²/(2ℓ²)) = exp(−1/2) ≈ 0.61`, so adjacent bumps blend smoothly while
/// each center keeps a *distinct* response and the design Gram is far from
/// rank-deficient. The old ×2 seed made every column `exp(−1/8) ≈ 0.88` at its
/// neighbor, driving the Gram toward rank deficiency so the inner PIRLS / outer
/// REML conditioning degraded and the search cycled for hundreds of seconds
/// (the #1116 timeout). ×1 is the resolving, well-conditioned end of the range
/// axis — the right side to *start* a search from.
///
/// It is emphatically NOT the right place to *stop*. #1041 changed the factor
/// ×2 → ×1 as a replacement for the ℓ dial it was turning off, on the argument
/// that a spacing-width kernel "fixes both at the root". Measured on
/// `measure_jet_perf_parity` at ×1, the design's least-squares span floor
/// against the noiseless truth is `0.152` and REML moves ℓ 7.5× longer to a
/// floor of `1.4e-5` (#2761). No fixed multiple of the center spacing is the
/// answer, because the answer depends on the target's smoothness relative to
/// the center layout — which is data, not geometry.
pub(crate) const MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR: f64 = 1.0;

/// Memory budget (in f64 entries) above which the multi-form assembly stops
/// parallelizing over scales: parallel scale partials cost
/// `L · n_forms · m²` doubles; past this budget the scales run sequentially
/// (same numbers — the per-scale loop and the ordered sum are deterministic
/// either way).
pub(crate) const MEASURE_JET_PARALLEL_FORM_BUDGET_DOUBLES: usize = 1 << 26;

/// Realized-design identifiability policy for the measure-jet smooth.
/// Mirrors [`super::ConstantCurvatureIdentifiability`] (#532): the fit-time
/// section gets the parametric orthogonalization composed onto it by the global
/// identifiability pipeline, and the composed transform is frozen so
/// predict-time (and per-ψ-trial) rebuilds replay it verbatim.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MeasureJetIdentifiability {
    /// Fit-time default. With the single-scale affine head, the RBF center
    /// values are mass-orthogonalized against the affine value space and the
    /// head passes through exactly; without a head, the representer coefficient
    /// sum-to-zero section is used. Global parametric residualization follows.
    #[default]
    CenterSumToZero,
    /// Predict-time replay: the frozen composed transform captured at fit
    /// time. `transform.nrows()` equals representer count plus affine-head width.
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
    /// Ambient input-measurement-error scale `σ_coord` (issue #2225): the
    /// perpendicular off-manifold residual spread of the fit-time empirical
    /// measure, in the frozen centers' (standardized) coordinate frame. Frozen
    /// so the predict-time errors-in-variables variance term
    /// `Var_input = σ_coord²·‖∇f̂‖²` uses the same input-noise scale the fit
    /// saw. `None` when it could not be estimated (no cell spanned a tangent),
    /// leaving `Var_input` disabled. Defaults to `None` for models persisted
    /// before the term existed.
    #[serde(default)]
    pub sigma_coord: Option<f64>,
}

/// Serde default for [`MeasureJetBasisSpec::learn_length_scale`]: REML-select
/// ℓ, the same standing the Matérn κ has. A function (not a literal) because
/// `#[serde(default)]` on a `bool` deserializes a missing field as `false`,
/// which would silently freeze ℓ on every spec that predates the field — the
/// opposite of the default. See the field's own docs for the measurement.
fn measure_jet_learn_length_scale_default() -> bool {
    true
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
    /// (`MEASURE_JET_DEFAULT_ORDER_S`).
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
    /// (median nearest-center spacing × `MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR`).
    pub length_scale: f64,
    /// Add a separate function-space affine/null-component penalty alongside
    /// the jet-energy penalty. Its strength is independently REML-selected.
    pub double_penalty: bool,
    /// REML-select the representer range ℓ as a design-moving outer dial,
    /// mirroring Matérn's `log_kappa`. **Default `true`** — ℓ is a basis
    /// coordinate of the same kind as the Matérn κ, not a smoothing parameter,
    /// and a fitted model must choose it.
    ///
    /// # Why it cannot be a frozen geometric value (#2761)
    ///
    /// The Gaussian kernel is strictly PD for every ℓ > 0, so ℓ does NOT change
    /// the basis rank (always `m` centers) — but it changes WHICH `m`-dim
    /// subspace the representers span. λ can only shrink inside a span; it
    /// cannot move one. So a mis-set ℓ is an error no smoothing parameter can
    /// repair, and the size of that error is not small. Measured on
    /// `measure_jet_perf_parity`'s 1-D-curve-in-3-D Gaussian fixture
    /// (`n = 1500`, σ = 0.10, 16 centers, `p = 15`), where `span floor` is the
    /// least-squares projection residual of the NOISELESS truth onto the
    /// realized design's column span — the bound no λ can beat:
    ///
    /// ```text
    ///   arm                       ell      edf   span floor  unpen. LS  held-out
    ///   frozen (auto ell)      0.5144   14.684    0.152488   0.155484   0.155584
    ///   REML-selected ell      3.8813   14.006    0.000014   0.008155   0.009642
    ///   matern(k=16)                -   14.619    0.006077   0.011989   0.011639
    ///   duchon(k=16)                -   15.016    0.002443   0.011308   0.010521
    /// ```
    ///
    /// At the frozen range the fit is already at `edf/p = 0.98` and its held-out
    /// RMSE *is* the span floor: unpenalized least squares on the same design
    /// gives 0.1555, and dropping the null-component penalty moves the fourth
    /// decimal. Freeing ℓ drops the floor by four orders and the held-out RMSE
    /// by 16x, past both comparators, at LOWER edf — nothing is traded for it.
    ///
    /// # History (so a fourth flip needs new evidence)
    ///
    /// `299c83ffc` (#1116) introduced this dial default-ON precisely to remove
    /// this fixture's 13x. `a3afd17a2` then found the one place it is unsafe —
    /// a BMS fit shares ONE mjs basis between the marginal mean and the
    /// slope surface, and a design-moving kernel scale on shared covariates
    /// is an identifiability hazard that reached a separation runaway — and
    /// contained it AT THE BMS ENTRY POINT with
    /// [`crate::smooth::freeze_measure_jet_length_scale_learning`], which is
    /// still what runs there. `b1d94d1a5` (#1041) nevertheless flipped the
    /// GLOBAL default off, and the 13x returned as #2761. The scoped freeze is
    /// the correct containment; the global one buys nothing it does not already
    /// buy and costs every single-surface fit its span alignment.
    ///
    /// `false` freezes ℓ at the auto (or explicit) value with no outer
    /// enrollment. The term builder selects that automatically when the user
    /// pins `length_scale=` — an explicit range is a request, not a seed —
    /// mirroring the Matérn `all_spatial_terms_kappa_fixed` short-circuit.
    #[serde(default = "measure_jet_learn_length_scale_default")]
    pub learn_length_scale: bool,
    /// Explicit opt-in for multiscale mode: the per-scale spectral penalty
    /// split plus the `(α, ln τ)` outer ψ dials. `false` (default) keeps the
    /// energy in single-scale mode at ANY center count. The separate
    /// `double_penalty` null component is available in both modes. There is no
    /// center-count auto-gate; the user opts in via
    /// `mjs(…, multiscale=true)`. Persisted on the spec so freeze→replay enters
    /// the same mode.
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
            learn_length_scale: true,
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

/// Rank-adapted center values of the measure-jet energy's affine null space.
///
/// The first column is the constant. The remaining columns are the supported
/// ambient-linear directions returned by [`measure_jet_affine_head_transform`].
/// Using that transform makes the basis full-column-rank even when the centers
/// lie on a lower-dimensional affine stratum of the ambient coordinates.
fn measure_jet_affine_value_basis(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Array2<f64> {
    // The SAME object the design's head block is built from, evaluated at the
    // centers. Sharing one construction is what makes "the head spans exactly
    // the energy's null space" a property of the code rather than a comment
    // two call sites have to keep agreeing on (#2751).
    let lift = measure_jet_affine_head_lift(centers, masses);
    measure_jet_affine_head_block(centers, lift.view())
}

/// Mass-metric quadratic form selecting the affine/null component of center
/// function values:
///
/// `H₀ = W A (Aᵀ W A)⁺ Aᵀ W`.
///
/// This is a function-space object: `vᵀH₀v` is the squared mass norm of the
/// affine projection of the center values `v`. No coefficient metric enters.
fn affine_function_nullspace_form(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let m = centers.nrows();
    if masses.len() != m {
        crate::bail_dim_basis!(
            "measure-jet affine function-space form shape mismatch: centers {:?}, masses {}",
            centers.dim(),
            masses.len()
        );
    }
    let affine = measure_jet_affine_value_basis(centers, masses);
    let mut weighted_affine = affine.clone();
    for (i, mut row) in weighted_affine.outer_iter_mut().enumerate() {
        row.mapv_inplace(|v| v * masses[i]);
    }
    let affine_gram = affine.t().dot(&weighted_affine);
    let affine_gram_pinv = symmetric_pseudoinverse(&affine_gram, "affine function-space Gram")?;
    let form = weighted_affine
        .dot(&affine_gram_pinv)
        .dot(&weighted_affine.t());
    Ok((&form + &form.t()) * 0.5)
}

/// Pull a center-value quadratic form back through an evaluation map.
fn pullback_center_form(evaluation: &Array2<f64>, form: &Array2<f64>) -> Array2<f64> {
    let pulled = evaluation.t().dot(form).dot(evaluation);
    (&pulled + &pulled.t()) * 0.5
}

/// Energy factor `F` of a symmetric PSD center-value form: `FᵀF = H` on `H`'s
/// positive part, rows `√λ_k · u_kᵀ`.
///
/// Every eigenvalue that is positive is kept — no rank tolerance enters, so
/// this is an exact factorization of `H⁺` rather than a truncation. `H` here is
/// always the output of [`measure_jet_energy_form`] or the affine projector,
/// both of which are PSD by construction, so the dropped part is roundoff.
fn psd_energy_factor(form: &Array2<f64>, context: &str) -> Result<Array2<f64>, BasisError> {
    let m = form.nrows();
    if m == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let sym = (form + &form.t()) * 0.5;
    let (evals, evecs) = sym.eigh(Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!(
            "measure-jet energy factorization `{context}` eigendecomposition failed: {e}"
        ))
    })?;
    let kept: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value > 0.0).then_some(index))
        .collect();
    let mut factor = Array2::<f64>::zeros((kept.len(), m));
    for (row, index) in kept.into_iter().enumerate() {
        let scale = evals[index].sqrt();
        for column in 0..m {
            factor[[row, column]] = scale * evecs[[column, index]];
        }
    }
    Ok(factor)
}

/// Pull a PSD center-value form back through an evaluation map CONSTRUCTIVELY:
/// `EᵀHE = (F E)ᵀ (F E)` for `FᵀF = H`.
///
/// The dense route (`pullback_center_form` + `try_from_dense_psd`) is a false
/// refusal waiting to happen and #2761 measured it firing: the Gaussian
/// representers go collinear as the range ℓ grows, `E`'s condition number blows
/// up, and the triple product loses exactly the digits that keep the smallest
/// eigenvalue non-negative. On `measure_jet_perf_parity` the whole design then
/// refuses to build for every `ℓ ≳ 2.8` —
///
/// ```text
///   ell   2.15059  builds        reml -1279.00091366   (still descending)
///   ell   2.79577  REFUSED  min eigenvalue -9.266e-9
///   ell   7.98500  REFUSED  min eigenvalue -2.864e-5
/// ```
///
/// — while the criterion is still descending at the last `ℓ` that builds, which
/// is what the outer search reports as `StepSizeTooSmall after 50 attempt(s)`:
/// its descent direction points into a region where the objective cannot be
/// EVALUATED. `-4e-5` relative on a Frobenius-normalized matrix is cancellation,
/// not negative curvature, and the refusal's own guidance says so ("supply the
/// native energy factor for a PSD function penalty").
///
/// Going through the factor makes PSD-ness structural instead of a numerical
/// accident, and never squares `E`'s condition number in the middle product.
/// The module's null-component penalty already worked this way
/// ([`affine_function_nullspace_quadratic`]); the energy Primary was the
/// sibling that did not.
fn constructive_pullback_center_form(
    evaluation: &Array2<f64>,
    form: &Array2<f64>,
    context: &str,
) -> Result<ConstructiveQuadratic, BasisError> {
    let factor = psd_energy_factor(form, context)?;
    ConstructiveQuadratic::from_energy_factor(factor.dot(evaluation), context)
}

/// Congruence of a jet onto a frame: `F (Fᵀ J F) Fᵀ`.
///
/// This is the exact derivative of `R(ψ) = N M(ψ) Nᵀ`, `M = Nᵀ S(ψ) N`, for a
/// ψ-FIXED frame `N` — which is what a declared structural null frame is. An
/// empty frame gives an exact zero, matching a rebuild that declined.
fn restrict_jet_to_frame(jet: &Array2<f64>, frame: &Array2<f64>) -> Array2<f64> {
    if frame.ncols() == 0 {
        return Array2::<f64>::zeros(jet.dim());
    }
    let inner = frame.t().dot(jet).dot(frame);
    let restricted = frame.dot(&inner).dot(&frame.t());
    (&restricted + &restricted.t()) * 0.5
}

/// Frobenius scale of a constructive quadratic, with the same degenerate
/// convention `normalize_penalty` uses: a scale at or below `1e-12` reports
/// `1e-12` so the division can never blow up.
fn constructive_frobenius_scale(quadratic: &ConstructiveQuadratic) -> f64 {
    quadratic
        .dense()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1e-12)
}

/// First and diagonal-second `u = ln ℓ` derivatives of `E(u)ᵀ H E(u)` for a
/// `u`-invariant center-value form `H`.
fn pullback_center_form_log_length_jets(
    evaluation: &Array2<f64>,
    evaluation_first: &Array2<f64>,
    evaluation_second: &Array2<f64>,
    form: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let h_e = form.dot(evaluation);
    let h_e_first = form.dot(evaluation_first);
    let h_e_second = form.dot(evaluation_second);
    let first_raw = evaluation_first.t().dot(&h_e) + evaluation.t().dot(&h_e_first);
    let second_raw = evaluation_second.t().dot(&h_e)
        + evaluation.t().dot(&h_e_second)
        + evaluation_first.t().dot(&h_e_first) * 2.0;
    (
        (&first_raw + &first_raw.t()) * 0.5,
        (&second_raw + &second_raw.t()) * 0.5,
    )
}

/// Mixed derivative `∂²(EᵀH(ψ)E)/(∂lnℓ ∂ψ)` when only `E` depends on `ℓ`.
fn pullback_center_form_log_length_cross(
    evaluation: &Array2<f64>,
    evaluation_first: &Array2<f64>,
    form_first: &Array2<f64>,
) -> Array2<f64> {
    let h_e = form_first.dot(evaluation);
    let h_e_first = form_first.dot(evaluation_first);
    let cross_raw = evaluation_first.t().dot(&h_e) + evaluation.t().dot(&h_e_first);
    (&cross_raw + &cross_raw.t()) * 0.5
}

/// The Primary energy's structural null frame in whatever coefficient chart
/// `z` realizes: the coefficients whose REPRESENTER block vanishes, i.e. the
/// pure ambient-affine-head directions.
///
/// This is a theorem of the construction, not a measurement, and it is the
/// reason the double-penalty topology can be ψ-invariant (#2445's mechanism,
/// applied here for #2761):
///
/// * the energy annihilates ambient-affine center values EXACTLY (the module's
///   no-mass / exact-affine-projection contract), so a coefficient whose center
///   values are `head_cc·b_head` — pure affine — is annihilated for EVERY `ℓ`;
/// * the single-scale gauge restricts the representer block to `null(AᵀW K_cc)`,
///   so its center values are mass-orthogonal to the affine space. A nonzero
///   representer part therefore cannot land in the energy's null space:
///   `K_cc z_rbf b_rbf ∈ A ∩ A^⊥ = {0}` forces `z_rbf b_rbf = 0`.
///
/// So `null(Primary) = { b : (z·b)|representer rows = 0 }` exactly, in the
/// fit-time chart AND in any composed frozen chart, because both statements are
/// about `z` alone. Nothing here reads `ℓ`.
///
/// Without the declaration the topology is decided by a rank test on the
/// pullback `kzᵀ Q kz`, whose numerical rank falls as the range grows and the
/// representers go collinear (measured on a 1-D 50-center term: rank 47 of 48 at
/// the auto seed, 14 of 48 at 8x it). A design-moving `ℓ` then adds or removes
/// the double-penalty ridge between outer trials — the `incremental realizer
/// topology changed` abort, and the `#860` penalty-count desync class.
///
/// Returns `None` in multiscale mode, where there is no affine head: the
/// energy's null space is then a genuinely `ℓ`-dependent subspace of the
/// representer block, and declaring an EMPTY frame would assert the opposite of
/// the truth (that the Primary has no null space at all) and silently delete the
/// null component. Measuring is the honest fallback there.
fn measure_jet_primary_structural_null_frame(
    z: &Array2<f64>,
    representer_count: usize,
    head_rank: usize,
) -> Result<Option<Array2<f64>>, BasisError> {
    if head_rank == 0 || representer_count == 0 || z.ncols() == 0 {
        return Ok(None);
    }
    // `rrqr_nullspace_basis(B)` returns `null(Bᵀ)`, so pass the transpose of the
    // representer rows to get the coefficient-space null vectors.
    let representer_rows = z.slice(ndarray::s![..representer_count, ..]).to_owned();
    let (frame, _) =
        rrqr_nullspace_basis(&representer_rows.t().to_owned(), default_rrqr_rank_alpha())
            .map_err(BasisError::LinalgError)?;
    // An EMPTY frame is a declaration, not a missing one: it says the chart has
    // absorbed every affine-head direction, so the Primary has no null space
    // here. That is the answer in a composed frozen chart, where the global
    // parametric orthogonalization has already taken the head — and it is
    // exactly the chart every ψ trial rebuilds in. Returning `None` there would
    // send the topology decision back to the rank test the declaration exists
    // to replace, reinstating the ℓ-dependence on the one path that cannot
    // tolerate it. `None` is reserved for "this construction does not apply"
    // (multiscale, no head).
    Ok(Some(frame))
}

/// Fixed-rank constructive witness for the affine/null quadratic in center-value
/// space. Rank is decided here, where `H₀` is independent of `ℓ`, rather than
/// after its coefficient pullback has acquired `ℓ`-dependent roundoff modes.
fn affine_function_nullspace_center_quadratic(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Result<ConstructiveQuadratic, BasisError> {
    ConstructiveQuadratic::try_from_dense_psd(
        affine_function_nullspace_form(centers, masses)?,
        "measure-jet affine center-value form",
    )
}

fn affine_function_nullspace_quadratic(
    evaluation: &Array2<f64>,
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Result<ConstructiveQuadratic, BasisError> {
    if evaluation.nrows() != centers.nrows() {
        crate::bail_dim_basis!(
            "measure-jet affine function-space penalty shape mismatch: evaluation {:?}, centers {:?}",
            evaluation.dim(),
            centers.dim()
        );
    }
    let center_quadratic = affine_function_nullspace_center_quadratic(centers, masses)?;
    ConstructiveQuadratic::from_energy_factor(
        center_quadratic.factor().dot(evaluation),
        "measure-jet affine/null coefficient penalty",
    )
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

/// Make the representer coefficient section a BASIS of its span at the realized
/// range, not merely a spanning set (gam#2750).
///
/// ## What goes wrong without this
///
/// A Gaussian kernel's spectrum decays super-exponentially in `ℓ/spacing`, so
/// the center evaluation map `E = K_cc · Z_rbf` — through which BOTH the design
/// and every penalty pullback pass — loses conditioning fast as the representer
/// range grows. The raw section is chosen for head-orthogonality alone and
/// keeps every direction, so the shipped chart carries directions the criterion
/// cannot resolve, and the criterion's own Occam term is the first casualty:
///
/// ```text
///   ell     cond(S)     d log|S|+/d ln ell:  analytic       FD        gap
///   1.21    1.2e10                          -109.843    -109.829    -0.014
///   1.57    3.0e11                          -107.924    -107.919    -0.005
///   2.04    3.7e13                           -97.252     -96.331    -0.922
///   2.65    3.8e13                           -51.836     -64.765   +12.930
///   3.45    6.5e12  (rank 14 -> 13)          -26.621     -40.633   +14.012
/// ```
///
/// (16 centers on the `measure_jet_perf_parity` geometry, chart frozen so the
/// finite difference and the analytic jet differentiate the same object; the
/// jet itself agrees with a central difference of the shipped penalty to
/// `1e-8` throughout, so the producer is exact and it is `log|S|₊` that stops
/// being a function.) Once `cond(S)` reaches the rank-classification floor the
/// kept spectrum bottoms out in roundoff: `log|S|₊` is then a sum over
/// directions whose logs are noise, `tr(S⁺Ṡ)` divides by them, and the outer
/// search is handed a direction that is not a descent direction of its own
/// objective. That is the wall the `ln ℓ` coordinate has been hitting.
///
/// ## The section
///
/// Whiten against the section's own center evaluation map,
/// `E = K_cc Z = U Σ Vᵀ`: take `Z ← Z · V · diag(1/max(σ, √ε·‖K_cc‖))`, so the
/// realized section satisfies `EᵀE = I` wherever `E` is resolvable and is
/// *damped rather than deleted* below that. Two questions, two bars, and they
/// are not the same question — but they share one anchor, and the anchor is
/// `‖K_cc‖₂`, the norm of the operator whose product formed `E`, never `E`'s
/// own largest singular value (which collapses; see the table at the bars):
///
/// * **Does the direction exist?** `σ > ε·‖K_cc‖·max(dim)` — the backward-error
///   bar of forming `E = K_cc·Z`. Below it there is no direction, only the
///   roundoff of that product.
/// * **How far may it be amplified?** `1/σ` is also the factor by which
///   direction `i` amplifies the design's own roundoff, and the criterion
///   squares the design into `XᵀWX`, so the amplification survives that Gram
///   with significant digits exactly when `ε·(‖K_cc‖/σ_i)² < 1`. Hence the
///   scaling — not the membership — is floored at `√ε·‖K_cc‖`.
///
/// **Damping instead of deleting is what makes the width `ℓ`-invariant.** A
/// span is invariant under ANY invertible diagonal rescaling, so a damped
/// direction contributes exactly the same column space as an undamped one; it
/// simply enters the design with a small norm instead of an amplified one, and
/// carries its own roundoff in at that same small norm. The realized chart
/// therefore keeps every direction `E` has, at every `ℓ`, while
/// `cond(E·W) ≤ √ε·cond(E)` is bounded by construction.
///
/// * The rescaling is a pure change of coefficient chart. The profiled
///   criterion is invariant under an invertible reparameterization (`X → XT`,
///   `S → TᵀST` moves `log|XᵀWX + λS|` and `log|λS|₊` by the same
///   `2 ln|det T|`), so it changes no estimate — only the arithmetic.
/// * **The whitening removes the squaring from the PENALTY, exactly.** With
///   `EᵀE = I` the energy pullback is `S = UᵀQU` with `U` orthonormal, whose
///   spectrum is bounded by `Q`'s; and `Q` is `ℓ`-INVARIANT (a form on center
///   VALUES), so `cond(S)` stops being a function of the range at all.
///   Measured, re-realizing the chart at each `ℓ` on the #2761 fixture:
///   `cond(E) = 1.0`, `cond(S₊) = 21.1`, `log|S|₊ = 0.800` from `1×` to `16×`
///   the seed range. So "the energy pullback squares `E`" — the reason the
///   previous cut gave for being where it was — does not survive its own
///   remedy.
/// * **What the whitening does NOT remove is the squaring in `XᵀWX`.** The
///   chart lives at the CENTERS; the design lives at the DATA, and `1/σ_i` is
///   also the factor by which direction `i` amplifies the design's own
///   roundoff. The criterion squares the design, so a retained direction
///   survives with significant digits exactly when `ε·(σ_max/σ_i)² < 1`. That
///   is where the half-mantissa belongs, and it is one half-mantissa, not two.
/// * **The previous cut spent the half-mantissa twice, on an already-squared
///   quantity — and spent it by DELETING.** It whitened against `G = EᵀE` and
///   cut at `√ε·λ_max(G)`; since `λ = σ²` that is `σ > ε^{1/4}·σ_max`, i.e.
///   `cond(E) ≤ ε^{-1/4} ≈ 8·10³`. Measured on the #2761 fixture at 16 centers,
///   that deleted most of the span at the ranges REML actively selects:
///
/// ```text
///   ℓ/ℓ_seed  cond(E)   ε^{1/4} DELETE: p  span floor    √ε DAMP: p  span floor
///      1      3.0e+01         12            6.11e-2          12       6.11e-2
///      2      2.8e+04         11            2.43e-2          12       1.94e-2
///      4      4.2e+07          8            1.50e-2          12       2.10e-3
///      8      9.1e+09          6            1.67e-2          12       1.81e-4   <- 92x
///     16      2.7e+11          4            8.92e-2          12       3.54e-5
/// ```
///
///   `span floor` is the least-squares residual RMSE of the NOISELESS truth on
///   the realized design's own column span — the bound no `λ` can beat, since
///   `λ` shrinks inside a span and never moves one. Note the `ε^{1/4}` floor
///   going UP past `4×`: that chart is worse than the seed range it was meant
///   to improve on. The damped column keeps the whole span (`1.81e-4` at `8×`
///   reproduces an 80-digit projection of the same span to every digit) at a
///   width that does not move with the dial, which a DELETING bar cannot do at
///   any threshold — `12,11,8,6,4` for `ε^{1/4}` and `12,12,12,11,10` even for
///   `√ε`.
///
/// Realized per cold build from `K_cc(ℓ)`, so the chart tracks the dial it is a
/// chart for; a frozen-quadrature replay reuses the composed transform verbatim.
fn condition_representer_section(
    k_cc: &Array2<f64>,
    z_rbf: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    if z_rbf.ncols() == 0 {
        return Ok(z_rbf.clone());
    }
    let evaluation = k_cc.dot(z_rbf);
    let (_, singular, right) = evaluation.svd(false, true).map_err(BasisError::LinalgError)?;
    let leading = singular.iter().copied().fold(0.0_f64, f64::max);
    if !(leading.is_finite() && leading > 0.0) {
        return Ok(z_rbf.clone());
    }
    // `right` is `Vᵀ`: row `i` is `σ_i`'s right singular vector, in the
    // coefficient coordinates of `z_rbf`.
    let right = right.ok_or_else(|| {
        BasisError::LinalgError(gam_linalg::faer_ndarray::FaerLinalgError::SvdNoConvergence {
            context: "measure-jet representer section: right singular vectors were not returned",
        })
    })?;
    // Both bars are anchored on `‖K_cc‖₂`, NOT on `σ_max(E)`. `E = K_cc·Z` with
    // orthonormal `Z`, so the computed `E` carries a backward error
    // `O(ε·‖K_cc‖₂)` whatever `E`'s own size turns out to be — and `σ_max(E)`
    // is not a fixed fraction of `‖K_cc‖₂`: it COLLAPSES as `ℓ` grows, because
    // the constraint that makes `Z` head-orthogonal is exactly what annihilates
    // the flat limit `K_cc → 𝟙𝟙ᵀ`. Measured on the 1-D sweep fixture (50
    // centers, standardized `x`):
    //
    // ```text
    //   ℓ      ‖K_cc‖₂  σ_max(E)  σ_max/‖K‖  σ_min(E)
    //   0.05     2.4     2.29e+0   9.6e-1     5.5e-2
    //   0.55    17.2     8.79e+0   5.1e-1     3.2e-16   <- σ_min is the roundoff
    //   6.09    48.4     1.49e-2   3.1e-4     4.5e-17      floor, and it is FLAT
    //  49.73    50.0     3.48e-6   7.0e-8     3.5e-17
    // ```
    //
    // A bar relative to `σ_max(E)` therefore falls BELOW that flat roundoff
    // floor once the range is long, and admits the floor itself as signal: at
    // `ℓ = 11` a `ε·σ_max·dim` bar is `1.5e-17` against entries of `1.7e-17`,
    // so all 48 directions "pass" and the chart hands the fit 48 columns of
    // pure rounding noise. Those columns fit anything, so the criterion has a
    // spurious minimum out there — measured at `V = −447.8` on this fixture's
    // sweep case 1 against `≈ −44` in the honest region, which is precisely
    // where its outer search was terminating.
    let dimension_factor = evaluation.nrows().max(evaluation.ncols()) as f64;
    let (_, kernel_singular, _) = k_cc.svd(false, false).map_err(BasisError::LinalgError)?;
    let anchor = kernel_singular
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(leading);
    // Membership: the backward-error bar of the product that formed `E`. Below
    // it there is no direction, only the roundoff of `K_cc·Z`.
    let existence = anchor * f64::EPSILON * dimension_factor;
    // Amplification: ONE half-mantissa, spent ONCE. `1/σ_i` is the factor by
    // which direction `i` lifts the design's own roundoff — which is set by
    // `‖K_xc‖ ∼ ‖K_cc‖`, not by `σ_max(E)` — and the criterion squares the
    // design into `XᵀWX`, so the lift survives that Gram with significant
    // digits exactly when `ε·(‖K_cc‖/σ_i)² < 1`. Below `√ε·‖K_cc‖` the
    // direction is DAMPED to that scaling rather than dropped: the span is
    // unchanged (any invertible rescaling spans the same columns) and the
    // roundoff comes in at the damped norm instead of an amplified one.
    //
    // This also reproduces the range bracket's own physics for free. As `ℓ`
    // passes the node diameter, `σ_max(E)` falls under the floor and the WHOLE
    // representer block damps smoothly toward zero, leaving the affine head —
    // which is exactly `MeasureJetRangeBracket::ceiling`'s statement that past
    // there "the block is numerically one function plus the affine head and
    // there is no distinct model past it", now realized by the arithmetic
    // rather than asserted next to it.
    let amplification_floor = anchor * f64::EPSILON.sqrt();
    // Visibility: a damped direction enters the energy pullback
    // `S = (E·W)ᵀQ(E·W)` with squared weight `(σ_i/floor)²`, so if that weight
    // drops below the canonical penalty-spectrum rank cutoff the direction is
    // classified UNPENALIZED — an accidentally free design direction, which is
    // the opposite of conservative. It also makes `log|S|₊` a step function of
    // `ℓ`: measured on the 1-D sweep fixture, the primary's nullity flapping by
    // one moved the profiled criterion by `8.5` at fixed `λ`, which is what a
    // `ln ℓ` line search cannot cross.
    //
    // So the chart's retention bar is at least as strict as the penalty's own:
    // keep direction `i` only while `(σ_i/floor)² > tol`, with `tol` the same
    // `spectral_tolerance` convention (#1425's single classifier) every other
    // penalty-spectrum consumer reads. No second constant, and the two
    // decisions can no longer disagree about which directions are penalized.
    let rank_tolerance =
        z_rbf.ncols().max(1) as f64 * super::bspline_build::SPECTRAL_RANK_RELATIVE_TOLERANCE;
    let visibility = amplification_floor * rank_tolerance.sqrt();
    let retention = existence.max(visibility);
    let kept: Vec<usize> = (0..singular.len())
        .filter(|&i| singular[i] > retention)
        .collect();
    let kept = if kept.is_empty() {
        // Every representer direction is below the resolvable floor. Keep the
        // single strongest one rather than emitting an empty block: a term with
        // no representer columns is a different model, and that decision
        // belongs to the range screen, not to a conditioning step.
        vec![(0..singular.len()).fold(0usize, |best, i| {
            if singular[i] > singular[best] { i } else { best }
        })]
    } else {
        kept
    };
    let mut transform = Array2::<f64>::zeros((z_rbf.ncols(), kept.len()));
    for (column, &index) in kept.iter().enumerate() {
        let inverse = singular[index]
            .max(amplification_floor)
            .max(f64::MIN_POSITIVE)
            .recip();
        // Sign gauge: a singular vector is defined up to sign, and the sign a
        // decomposition happens to return is not a property of the geometry.
        // Pin it on the entry of largest magnitude so the realized chart is
        // reproducible across faer revisions and platforms.
        let mut pivot = 0usize;
        for row in 1..z_rbf.ncols() {
            if right[(index, row)].abs() > right[(index, pivot)].abs() {
                pivot = row;
            }
        }
        let sign = if right[(index, pivot)] < 0.0 { -1.0 } else { 1.0 };
        for row in 0..z_rbf.ncols() {
            transform[(row, column)] = sign * right[(index, row)] * inverse;
        }
    }
    Ok(z_rbf.dot(&transform))
}

/// Axis-aligned bounding-box diagonal of a point set — the deterministic
/// diameter proxy the scale band and the range bracket both measure the
/// configuration's extent with. `O(m·d)` and permutation-invariant, unlike a
/// max-pairwise-distance scan.
pub(crate) fn bounding_box_diagonal(points: ArrayView2<'_, f64>) -> f64 {
    let mut diag2 = 0.0_f64;
    for k in 0..points.ncols() {
        let col = points.column(k);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in col.iter() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo.is_finite() && hi.is_finite() {
            diag2 += (hi - lo) * (hi - lo);
        }
    }
    diag2.sqrt()
}

/// The deterministic bracket the representer range `ℓ` is SCREENED over before
/// the outer ψ search refines it, in the STANDARDIZED frame the basis is
/// realized in (gam#2750).
///
/// ## Why a bracket exists at all
///
/// `ℓ` is a design-moving coordinate: it decides WHICH span the representers
/// occupy, and the outer search reaches it by local descent from the seed. The
/// profiled criterion in `ln ℓ` is not unimodal — as `ℓ` grows past a few
/// center spacings the Gaussian columns become collinear, the rank-revealing
/// identifiability section drops columns, and the criterion steps. Measured on
/// `measure_jet_formula_fit_robustness_sweep` seed 1 (n = 200, one sine cycle):
/// a local minimum at the auto range `ℓ = 0.020` (`V = −234.5`), a barrier at
/// `ℓ = 0.035` (`V = −231.1`), and the GLOBAL minimum at `ℓ = 0.80`
/// (`V = −256.3`) — 21.7 log units deeper, with held-out RMSE `0.0084` against
/// `0.0175`, i.e. the same basis fitting 2.1× better and beating `tp` on both
/// the criterion and the truth instead of losing on both. A local descent
/// seeded inside the first basin cannot cross that barrier, and the frozen
/// coefficient chart the ψ trials rebuild in stops being evaluable ~1.6× past
/// the seed, so the search terminates essentially where it started. The λ that
/// comes back is then a faithful readout of a range nothing could move — which
/// is the "1-D fits select a too-large λ" this bracket exists to end.
///
/// ## Why THIS bracket
///
/// The nodes are the term's own realized scale band, verbatim. That is not a
/// coincidence of convenience: the energy's `ε` and the representer range `ℓ`
/// are the same physical quantity — a length in the chart — and the band is
/// already derived, not chosen: its floor is the median nearest-node spacing
/// (below it neighbouring representers stop overlapping and the design is a
/// bump-per-node indicator with no partition of unity) and its ceiling is half
/// the node bounding-box diagonal, at the band's own auto-clamped resolution.
/// So the screen introduces no length, no count and no step of its own.
///
/// ## Where a walk past the top node stops (#2761)
///
/// [`MeasureJetRangeBracket::feasibility_ceiling`] — the range at which the
/// closest node pair stops being distinguishable in the chart's own arithmetic,
/// [`measure_jet_range_feasibility_ceiling`]. It is the SAME wall
/// [`measure_jet_ln_range_window`] gives the outer search, and it is the same
/// wall for the same reason: a stopping rule may not be tighter than the model.
///
/// It used to be [`MeasureJetRangeBracket::node_diameter`], on the argument
/// that at `ℓ` that long every pair of representers overlaps at `≥ exp(−1/2)`
/// so "there is no distinct model past it". That argument is measurably wrong,
/// and the tree said so in two places before this: `measure_jet_ln_range_window`
/// records that *"the profiled criterion genuinely prefers a range AT or ABOVE
/// the node diameter"* on three fixtures, and
/// `the_search_window_reaches_past_where_the_screen_stops_walking` pins the
/// search window as strictly wider. Those reconcile only while something else
/// keeps searching past the stopping rule. On a term whose `ℓ` dial is FROZEN —
/// the BMS marginal/slope pair, or any `learn_length_scale=false` — nothing
/// does, and the stopping rule becomes the wall.
///
/// Measured on the #1041 parity fixture (`m = 10`, extent `[2.671, 2.726]`):
/// band `[1.08074, 1.43607, 1.90823]`, `log_step = 0.284265`, node diameter
/// `3.81645`. The screen's chosen range for the marginal surface was `3.36930`
/// — walk node 2 to every printed digit, with walk node 3 at `4.47708` past the
/// diameter. The walk pushes a node and only then breaks if it failed to
/// improve, so an argmin that IS the last pushed node improved: the walk left
/// through the ceiling test with the criterion still descending.
///
/// **What raising the stop was actually worth here, measured after the change,
/// because it is less than the shape of the defect suggests.** The walk now
/// scores `4.47708`, which does NOT improve — so the criterion has an interior
/// optimum on this fixture and the old ceiling happened to cut just past it.
/// What the extra node buys is the PARABOLIC REFINEMENT, which cannot fire on
/// an argmin that is the last element: with a neighbour on both sides the
/// refinement lands at `ℓ = 3.10543` with a better criterion value, and
/// held-out marginal RMSE goes `0.04185 → 0.04179`. A rule that cannot be
/// stepped past also cannot be refined at, and that is the part of the cost
/// that was invisible.
///
/// The larger held-out number on the same sweep — `0.03788` at `ℓ = 68.5`,
/// where the block still carries `edf = 7.47` and is not degenerate — is NOT
/// what this change recovers, and attributing it to the ceiling would be wrong:
/// the criterion does not want to go there. That gap is a statement about the
/// screening CRITERION (a profiled Gaussian REML of the term alone against a
/// binary response) disagreeing with held-out truth at long ranges, and it
/// belongs to whoever takes that question next.
///
/// Raising the stop is safe by the walk's own rule, which only continues while
/// the criterion improves: on the gam#2750 fixture, where the criterion drops
/// from `−256.3` to `−198.5` just past the diameter, the walk stops on the
/// first non-improving node exactly as it does today. The ceiling only ever
/// binds where the criterion is still descending — which is precisely the case
/// where stopping is wrong.
#[derive(Clone, Debug)]
pub struct MeasureJetRangeBracket {
    /// Geometric grid of candidate ranges, ascending. The realized scale band.
    pub nodes: Vec<f64>,
    /// The band's own log step, so an endpoint walk keeps its resolution.
    pub log_step: f64,
    /// The node bounding-box diagonal. A geometric fact about the cloud,
    /// reported because the band's own ceiling is half of it; NOT a stopping
    /// rule (see the type docs).
    pub node_diameter: f64,
    /// Hard upper end for any walk past the top node: the feasibility wall
    /// [`measure_jet_range_feasibility_ceiling`], the same one the outer
    /// search's [`measure_jet_ln_range_window`] stops at.
    pub feasibility_ceiling: f64,
}

/// The range at which a node pair separated by `spacing` stops being
/// DISTINGUISHABLE from a coincident one in `f64`.
///
/// `exp(−spacing²/2ℓ²)` has come within `√ε` of 1 at
/// `ℓ = spacing/√(2√ε)`; past it `K_cc` is the all-ones matrix to working
/// precision and the gauge annihilates exactly that (the affine span, constant
/// included), so no distinct model survives. `√ε` is the chart's own bar — the
/// same half-mantissa `condition_representer_section` spends, and for the same
/// reason: it is the point past which a direction cannot survive being squared
/// into a Gram and inverted back out.
///
/// ONE definition, used by both the outer search's window
/// ([`measure_jet_ln_range_window`]) and the response screen's walk stop
/// ([`MeasureJetRangeBracket::feasibility_ceiling`]), so the two cannot drift
/// into disagreeing about where the model ends (#2761).
pub fn measure_jet_range_feasibility_ceiling(spacing: f64) -> f64 {
    spacing / (2.0 * f64::EPSILON.sqrt()).sqrt()
}

/// Realize [`MeasureJetRangeBracket`] for a fresh (unfrozen) measure-jet spec.
///
/// `data` must already be in the standardized frame the basis is built in, and
/// the spec must be the FRESH one (no frozen quadrature): the bracket is a
/// seeding device and a frozen term has nothing left to seed.
pub fn measure_jet_range_bracket(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<MeasureJetRangeBracket, BasisError> {
    if spec.frozen_quadrature.is_some() {
        crate::bail_invalid_basis!(
            "measure-jet range bracket is a seeding device; a frozen-quadrature spec has no seed left to choose"
        );
    }
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("measure-jet range bracket needs at least one feature column");
    }
    validate_finite_points(data, "data")?;
    let seed_centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    if seed_centers.nrows() < 3 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: seed_centers.nrows(),
        });
    }
    let (nodes, _masses) = measure_jet_quadrature_nodes(data, seed_centers.view())?;
    let band = measure_jet_band(nodes.view(), spec.num_scales)?;
    // The band floor IS the median nearest-node spacing, so the feasibility
    // wall is read off the bracket's own first node rather than remeasured.
    let feasibility_ceiling = measure_jet_range_feasibility_ceiling(band.eps[0]);
    Ok(MeasureJetRangeBracket {
        nodes: band.eps,
        log_step: band.log_step,
        node_diameter: bounding_box_diagonal(nodes.view()),
        feasibility_ceiling,
    })
}

/// The `ln ℓ` SEARCH WINDOW for the design-moving representer range, in the
/// frame `spec.length_scale` is expressed in (gam#2750).
///
/// ## Why this is not an absolute interval
///
/// `ℓ` is a LENGTH in the chart the basis is realized in, so a window for it is
/// a statement about the node cloud, not about `f64`. Both ends here are the
/// same measured length — the median nearest-node spacing `s`, which is also
/// [`realized_measure_jet_length_scale`]'s auto value and
/// [`MeasureJetBand::eps`]'s floor — read at the two ranges where the kernel
/// stops saying anything about the pair it separates:
///
/// * **floor `ℓ = s`.** Neighbouring representers overlap at exactly
///   `exp(−1/2)`. Below it they stop overlapping, the design degenerates from a
///   partition of unity into a bump-per-node indicator, and rows between nodes
///   fall outside every representer's support.
/// * **ceiling `ℓ = s/√(2√ε)`.** The same neighbouring pair's kernel value
///   `exp(−s²/2ℓ²)` has come within `√ε` of 1, so the pair is no longer
///   DISTINGUISHABLE from a coincident one in the arithmetic the chart is built
///   in. Past it `K_cc` is the all-ones matrix to working precision, and the
///   gauge annihilates exactly that (the affine span, constant included), so
///   there is no distinct model left. `√ε` is the chart's own bar — the same
///   half-mantissa `condition_representer_section` spends, and for the same
///   reason: it is the point past which a direction cannot survive being
///   squared into a Gram and inverted back out.
///
/// The window is therefore `[ln s, ln s − ½ln(2√ε)]`: it TRANSLATES with the
/// chart (both ends are proportional to a measured length, so an isotropic
/// rescale by `c` shifts them both by `ln c`) and its WIDTH is `8.664`, a pure
/// function of `f64::EPSILON` rather than a number anybody picked.
///
/// ## What this deliberately is NOT
///
/// It is **not** [`MeasureJetRangeBracket::node_diameter`], the node bounding-box
/// diagonal. That is where the response SCREEN stops walking, which is a
/// stopping rule for a search over nodes, not a wall in the model: measured on
/// three fixtures (`measure_jet_formula_fit_robustness_sweep` seed 1,
/// `measure_jet_web_quality`, and the two probes that score them), the profiled
/// criterion genuinely prefers a range AT or ABOVE the node diameter, and a box
/// that stopped there railed the outer search and refused the fit. A long range
/// is a legitimate model — as `ℓ` grows the gauge-quotiented representer span
/// tends to a polynomial one, which is exactly the right basis for a smooth
/// target — so the upper end has to be a feasibility statement and nothing
/// weaker.
pub fn measure_jet_ln_range_window(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<(f64, f64), BasisError> {
    let spacing = match (&spec.center_strategy, &spec.frozen_quadrature) {
        (CenterStrategy::UserProvided(_), Some(frozen)) if !frozen.eps_band.is_empty() => {
            frozen.eps_band[0]
        }
        _ => measure_jet_range_bracket(data, spec)?.nodes[0],
    };
    if !(spacing.is_finite() && spacing > 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet ln-range window is degenerate: the node cloud reports a nearest-node \
             spacing of {spacing}, so it has no range scale to search over"
        );
    }
    // The range at which `1 - exp(-s^2/2l^2)` reaches the chart's half-mantissa
    // bar, from the single definition the screen's walk stop also reads.
    let ceiling = measure_jet_range_feasibility_ceiling(spacing);
    Ok((spacing.ln(), ceiling.ln()))
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
    let eps_max = 0.5 * bounding_box_diagonal(centers);
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
/// machine-precision rank cutoff. One walk of `assemble_weighted_forms`
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

/// The per-scale energy forms `Q_ℓ` (each m × m, symmetric PSD), with
/// `Σ_ℓ Q_ℓ = Q` to the PSD-projection floor (same blocks, one-hot weights).
/// These are the spectral-split carriers: emitted as separate penalty
/// candidates they let the multi-penalty REML engine learn per-level amplitudes
/// λ_ℓ directly — scale adaptivity at ρ-speed with no rebuild and no new
/// optimizer code.
///
/// Each level is projected onto the PSD cone for the same reason the fused
/// [`measure_jet_energy_form`] is, and the projection is load-bearing HERE in a
/// way it is not there — see the note at the return.
pub fn measure_jet_energy_forms_per_scale(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
) -> Result<Vec<Array2<f64>>, BasisError> {
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
    // PSD cone projection, per level. Every `Q_ℓ` is a NONNEGATIVE combination
    // of the same analytically-PSD local residual blocks the fused energy sums,
    // so it is PSD in exact arithmetic and only the per-block pseudo-inverse and
    // centering cancellation put a ±ε_mach·‖Q_ℓ‖ negative in the spectrum —
    // the identical situation `measure_jet_energy_form` floors on the cone.
    //
    // Skipping it here was NOT symmetric with the fused path, because the fused
    // path normalizes ONE matrix while the builder normalizes EVERY LEVEL BY ITS
    // OWN Frobenius scale. A level whose detail energy is numerically dead
    // carries only that roundoff, and dividing roundoff by its own tiny norm
    // rescales it to unit norm: a ±ε_mach relative negative becomes an O(1)
    // absolute one. `ConstructiveQuadratic::try_from_dense_psd` then rejects the
    // candidate and the whole multiscale BUILD fails — measured as
    // `IndefinitePenalty { context: "measure-jet scale penalty",
    // min_eigenvalue: -0.3039, tolerance: 1.486e-8 }`, where the tolerance is
    // √ε_mach against a max |λ| of ~1, i.e. the certified matrix is already
    // unit-normalized and the negative is 2e7× tolerance. No real detail
    // spectrum is 30% negative; that is normalized roundoff.
    //
    // Flooring restores the invariant the signature documents, leaves a dead
    // level as an exact-zero candidate for `filter_penalty_candidates`/REML to
    // deselect rather than a fatal build error, and preserves `Σ_ℓ Q_ℓ = Q` to
    // the same machine-precision floor the fused projection already accepts. It
    // also stops `measure_jet_scale_spectrum` from being able to report a
    // NEGATIVE detail energy `vᵀQ_ℓv`.
    forms
        .into_iter()
        .enumerate()
        .map(|(level, q_l)| {
            project_symmetric_psd(q_l, &format!("measure-jet per-scale energy form {level}"))
        })
        .collect()
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

/// Exact first and diagonal-second derivatives of the Gaussian representer
/// design with respect to `u = ln ℓ`.
fn measure_jet_design_log_length_jets(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let kernel = measure_jet_design_matrix(data, centers, length_scale)?;
    let squared_distances = pairwise_sq_dists(data, centers);
    let inv_l2 = 1.0 / (length_scale * length_scale);
    let mut first = kernel.clone();
    let mut second = kernel;
    for ((first_value, second_value), &distance_squared) in first
        .iter_mut()
        .zip(second.iter_mut())
        .zip(squared_distances.iter())
    {
        let a = distance_squared * inv_l2;
        let kernel_value = *first_value;
        *first_value = kernel_value * a;
        *second_value = kernel_value * (a * a - 2.0 * a);
    }
    Ok((first, second))
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
/// `MEASURE_JET_PSEUDOINVERSE_RTOL` is the module's own numerical rank
/// tolerance (the same one the local Gram pseudo-inverses use). The returned
/// `T` satisfies `linear_head(points) = points · T` (the mean-centering only
/// informs the keep/drop decision). `T` is a deterministic function of the
/// frozen centers + masses, so the frozen replay path reconstructs the
/// identical head with no persisted state.
///
/// This is the LINEAR half of the null space. The realized head block is the
/// whole affine null space `[1 | points·T]`; build it through
/// [`measure_jet_affine_head_lift`] + [`measure_jet_affine_head_block`], which
/// is what the design, the gauge and the null-component penalty all use. A
/// linear-only head is a defect, not an economy: the global parametric
/// orthogonalization removes ONE design direction, and if the term's null space
/// has no constant to give up, the direction it takes comes out of the null
/// space itself, leaving `d − 1` free linear directions instead of `d` (#2751).
pub fn measure_jet_affine_head_transform(
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
    let max_norm = cols
        .iter()
        .fold(0.0_f64, |acc, c| acc.max(mdot(c, c).sqrt()));
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

/// Affine head lift `T_aff` (`(d+1) × (1 + head_rank)`) acting on the augmented
/// point rows `[1 | x]`: column 0 is the constant, the rest are the supported
/// ambient-linear directions of [`measure_jet_affine_head_transform`].
///
/// This — not the linear lift alone — is the energy's null space. The energy
/// annihilates every AFFINE function of the centers exactly, constant included
/// (`affine_function_nullspace_form` projects onto exactly this span), so the
/// design block that carries the null space has to span the same thing.
///
/// The constant column looks redundant against the model intercept and is not.
/// The term-collection chokepoint residualizes every measure-jet design against
/// the parametric block and reparameterizes to `Z = null(1ᵀX)`, which removes
/// exactly one coefficient direction. The null space of the constrained penalty
/// is `{γ : Zγ ∈ null(S)}`, so that removal is charged to the null space unless
/// the null space contains the constraint's own direction. With a linear-only
/// head the term's null space is `span{x·T}`, the constant is nowhere in it,
/// and the centering deletes a LINEAR direction: on a 2-D fixture the surviving
/// direction is the accidental one with zero data-mean, and every REML fit that
/// selects a large energy λ collapses onto it (#2751, measured at Pearson
/// 0.705 = |cos 45°| against a planted `x1` plane). With the constant present
/// the centering consumes the constant — which the intercept re-supplies —
/// and all `head_rank` linear directions stay free. That is exactly how the
/// thin-plate/Duchon null space `{1, x_1..x_d}` behaves at the same chokepoint.
pub fn measure_jet_affine_head_lift(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let linear = measure_jet_affine_head_transform(centers, masses);
    let d = centers.ncols();
    let mut lift = Array2::<f64>::zeros((d + 1, linear.ncols() + 1));
    lift[(0, 0)] = 1.0;
    lift.slice_mut(ndarray::s![1.., 1..]).assign(&linear);
    lift
}

/// Realize the affine head block `[1 | points] · T_aff` for the lift returned
/// by [`measure_jet_affine_head_lift`]. A zero-column lift (multiscale mode,
/// which carries no head) yields a zero-column block.
pub fn measure_jet_affine_head_block(
    points: ArrayView2<'_, f64>,
    lift: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = points.nrows();
    let width = lift.ncols();
    if width == 0 {
        return Array2::<f64>::zeros((n, 0));
    }
    let d = points.ncols();
    assert_eq!(
        lift.nrows(),
        d + 1,
        "affine head lift must have d+1 rows for d ambient coordinates"
    );
    let mut augmented = Array2::<f64>::ones((n, d + 1));
    augmented.slice_mut(ndarray::s![.., 1..]).assign(&points);
    augmented.dot(&lift)
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
    /// Affine head lift `T_aff` ((d+1) × head_width): the energy's null space
    /// appended to the representer design (#1845), constant included (#2751).
    /// The head columns evaluate as `[1 | points] · T_aff`; empty
    /// (`(d+1) × 0`) in multiscale mode, which carries no head. Deterministic
    /// in the frozen centers + masses, so predict-time replay rebuilds it
    /// verbatim.
    pub(crate) head_lift: Array2<f64>,
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
    // Affine extrapolation head (#1845): the raw center space becomes
    // `[ m Gaussian representers | head_width affine columns ]`. The head
    // carries the penalty's affine null space explicitly — constant included
    // (#2751) — so the fit no longer reverts to the parametric backbone (the
    // training mean) across an unsupported gap, and so the collection's
    // parametric orthogonalization has the constant to consume instead of a
    // linear direction.
    // The extrapolation head is the single-scale (fused) gap-bridge path. In
    // multiscale mode the per-scale spectral penalties carry their own
    // structure and the design stays the pure representer basis (the per-level
    // replay + width contracts pin `m − 1` columns), so the head is added only
    // when the term is single-scale.
    let head_lift = if spec.multiscale {
        Array2::<f64>::zeros((centers.ncols() + 1, 0))
    } else {
        measure_jet_affine_head_lift(centers.view(), masses.view())
    };
    let head_width = head_lift.ncols();
    let m_aug = m + head_width;
    let k_cc = measure_jet_design_matrix(centers.view(), centers.view(), length_scale)?;
    let head_cc = measure_jet_affine_head_block(centers.view(), head_lift.view());
    // Realized-design constraint transform. In single-scale mode the explicit
    // affine head and Gaussian representers can otherwise carry the same affine
    // CENTER values in two different ways. That is a genuine gauge redundancy,
    // not a reason to ridge either coefficient block. At fit time remove it
    // exactly by restricting the RBF center values to the mass-orthogonal
    // complement of the supported affine space:
    //
    //   C = A^T W K_cc,       Z_rbf = null(C).
    //
    // The head then passes through as an identity block. The frozen composed
    // `z · z_parametric` is replayed verbatim at prediction/ψ trials (#532), so
    // the rank-revealed section never changes after fit-time realization. In
    // multiscale mode there is no explicit head, hence no affine duplication;
    // retain the existing representer sum-to-zero section there.
    let (z, coefficient_gauge) = match &spec.identifiability {
        MeasureJetIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != m_aug {
                crate::bail_dim_basis!(
                    "frozen measure-jet identifiability transform mismatch: {} representers + {} head columns but transform has {} rows",
                    m,
                    head_width,
                    transform.nrows()
                );
            }
            (
                transform.clone(),
                gam_problem::Gauge::from_block_transforms(&[transform.clone()]),
            )
        }
        MeasureJetIdentifiability::CenterSumToZero => {
            let z_rbf = if head_width > 0 {
                // `head_cc` IS the affine value basis A at the centers, by
                // construction (both come from `measure_jet_affine_head_lift`),
                // so the gauge constrains the representers against exactly the
                // span the head carries.
                let mut weighted_affine = head_cc.clone();
                for (i, mut row) in weighted_affine.outer_iter_mut().enumerate() {
                    row.mapv_inplace(|v| v * masses[i]);
                }
                // `rrqr_nullspace_basis(B)` returns null(B^T). Here
                // `B = K_cc^T W A = C^T`, hence the returned columns span
                // null(C), exactly the required RBF coefficient section.
                let constraint_cross = k_cc.t().dot(&weighted_affine);
                rrqr_nullspace_basis(&constraint_cross, default_rrqr_rank_alpha())
                    .map_err(BasisError::LinalgError)?
                    .0
            } else {
                let u = householder_sum_to_zero_u(m);
                householder_sum_to_zero_z(&u)
            };
            let z_rbf = condition_representer_section(&k_cc, &z_rbf)?;
            let rbf_rank = z_rbf.ncols();
            let mut z_block = Array2::<f64>::zeros((m_aug, rbf_rank + head_width));
            z_block
                .slice_mut(ndarray::s![..m, ..rbf_rank])
                .assign(&z_rbf);
            for r in 0..head_width {
                z_block[(m + r, rbf_rank + r)] = 1.0;
            }
            (
                z_block.clone(),
                gam_problem::Gauge::from_block_transforms(&[z_block]),
            )
        }
    };
    // Augmented raw center matrix `[K(centers, centers) | A]`, so the
    // restricted `kz` maps constrained coefficients to center nodal values for
    // BOTH the representers and the head; the energy annihilates the head block
    // (affine) to machine precision, so it stays the unpenalized null space.
    let mut k_aug = Array2::<f64>::zeros((m, m_aug));
    k_aug.slice_mut(ndarray::s![.., ..m]).assign(&k_cc);
    if head_width > 0 {
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
        // Multiscale (per-scale spectral) energy is an EXPLICIT opt-in (#1116):
        // one Primary energy at any center count unless the spec asks for the
        // scale split. The independent null-component candidate is orthogonal
        // to this mode decision. No center-count auto-gate.
        per_level: spec.multiscale,
        z,
        coefficient_gauge,
        kz,
        head_lift,
    })
}

/// Estimate the ambient input-measurement-error scale `σ_coord` — the
/// perpendicular off-manifold residual spread of the empirical measure — for
/// the errors-in-variables predictive-variance term `Var_input = ∇f̂ᵀΣ_x∇f̂`,
/// `Σ_x = σ_coord²·I` (issue #2225).
///
/// The measure-jet models data concentrated near an unknown low-intrinsic-
/// dimension set sampled with isotropic ambient coordinate noise. In a
/// neighborhood the set is locally affine, so the noise lives in the ambient
/// directions ORTHOGONAL to the local tangent — exactly the smallest principal
/// directions of the local data covariance. This is the standard local-PCA
/// noise floor: for each center's nearest-assignment cell with enough points to
/// span a tangent (`≥ d + 1`, the linear-algebra rank requirement — not a tuned
/// knob), the smallest eigenvalue of the cell-local covariance estimates the
/// perpendicular variance `σ_coord²`; averaging over cells (weighted by the
/// cell count) pools the estimate. No response values, no smoothing dial, and
/// no magic constant enter — it is a pure function of the ambient point cloud
/// and the frozen centers, in the centers' (standardized) coordinate frame.
///
/// Returns `None` when no cell can span a tangent (e.g. `d`-dimensional data
/// with fewer than `d + 1` points per cell, or a full-dimensional stratum with
/// no separable perpendicular direction) — the caller then leaves `Var_input`
/// disabled rather than invent a scale.
pub fn measure_jet_input_noise_scale(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<Option<f64>, BasisError> {
    let d = data.ncols();
    let m = centers.nrows();
    if d == 0 || m == 0 || data.nrows() == 0 {
        return Ok(None);
    }
    if centers.ncols() != d {
        crate::bail_dim_basis!(
            "measure-jet input-noise estimate: data d={d} disagrees with centers d={}",
            centers.ncols()
        );
    }
    validate_finite_points(data, "data")?;
    validate_finite_points(centers, "centers")?;
    // Nearest-center assignment (the same rule that lumps the quadrature
    // masses): the squared-distance Gram, argmin per row.
    let sq = pairwise_sq_dists(data, centers);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (j, row) in sq.axis_iter(Axis(0)).enumerate() {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &dij) in row.iter().enumerate() {
            if dij < best_d {
                best_d = dij;
                best = i;
            }
        }
        members[best].push(j);
    }
    let mut weighted_sum = 0.0_f64;
    let mut weight = 0.0_f64;
    for cell in &members {
        let n_i = cell.len();
        // A cell needs at least d + 1 points to define a full-rank local
        // covariance; otherwise its smallest eigenvalue is a spurious zero.
        if n_i < d + 1 {
            continue;
        }
        // Cell-local mean and covariance in ambient coordinates.
        let mut mean = Array1::<f64>::zeros(d);
        for &j in cell {
            mean += &data.row(j);
        }
        mean /= n_i as f64;
        let mut cov = Array2::<f64>::zeros((d, d));
        for &j in cell {
            let mut centered = data.row(j).to_owned();
            centered -= &mean;
            for a in 0..d {
                for b in 0..d {
                    cov[(a, b)] += centered[a] * centered[b];
                }
            }
        }
        cov /= n_i as f64;
        // Symmetrize against accumulation asymmetry, then read the smallest
        // eigenvalue = the perpendicular (noise) principal variance.
        let cov_sym = (&cov + &cov.t()) * 0.5;
        let (evals, _) = cov_sym.eigh(Side::Lower).map_err(|e| {
            BasisError::InvalidInput(format!(
                "measure-jet input-noise estimate: local covariance eigendecomposition failed: {e}"
            ))
        })?;
        let smallest = evals
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, v| acc.min(v))
            .max(0.0);
        if smallest.is_finite() {
            weighted_sum += n_i as f64 * smallest;
            weight += n_i as f64;
        }
    }
    if weight <= 0.0 {
        return Ok(None);
    }
    let sigma2 = weighted_sum / weight;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Ok(None);
    }
    Ok(Some(sigma2.sqrt()))
}

/// Whether a measure-jet spec runs in multiscale mode (per-scale spectral
/// energies + `(α, ln τ)` ψ dials). The separate `double_penalty`
/// affine/null-component candidate is available in both modes. This is the
/// single source of truth shared by the builder and outer enrollment predicates,
/// so the energy layout and ψ dimension cannot disagree. Multiscale is an
/// explicit opt-in (`spec.multiscale`); there is no center-count auto-gate
/// (#1116).
pub fn measure_jet_multiscale_mode(spec: &MeasureJetBasisSpec) -> bool {
    spec.multiscale
}

/// Build the measure-jet smooth: Gaussian representer design `K(data,
/// centers)·z`, multiscale jet-residual penalty (one candidate per scale in
/// spectral mode, one Primary in pinned-order mode), an optional separate
/// function-space null-component candidate, and the replayable
/// [`BasisMetadata::MeasureJet`]. The geometry comes from the
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
        head_lift,
    } = realize_measure_jet_geometry(data, spec)?;
    let band = MeasureJetBand {
        eps: eps_band.clone(),
        log_step,
    };
    let m = centers.nrows();
    let head_width = head_lift.ncols();
    let m_aug = m + head_width;
    // Augmented raw design `[K(data, centers) | [1 | data]·T_aff]` (#1845): the
    // head columns are the AFFINE extrapolation basis, which is the energy's
    // whole null space (#2751). The gauge restricts BOTH blocks together, so
    // the frozen composed transform replays the head verbatim at predict time.
    let kernel_design = measure_jet_design_matrix(data, centers.view(), length_scale)?;
    let mut raw_design = Array2::<f64>::zeros((data.nrows(), m_aug));
    raw_design
        .slice_mut(ndarray::s![.., ..m])
        .assign(&kernel_design);
    if head_width > 0 {
        let head_design = measure_jet_affine_head_block(data, head_lift.view());
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
            // Constructive pullback, not a dense triple product: the per-scale
            // form is PSD by construction and so is its pullback, and only the
            // arithmetic can lose that (#2761).
            let s_l = constructive_pullback_center_form(&kz, &q_l, "measure-jet scale penalty")?;
            let c_l = constructive_frobenius_scale(&s_l);
            let intrinsic_dim = centers.ncols() as f64;
            let eta = 2.0 * order_s + intrinsic_dim * (2.0 - 2.0 * spec.alpha);
            let scale_weight = log_step * eps_band[level].powf(-eta);
            penalty_normalization_scales.push(c_l);
            raw_penalty_normalization_scales.push(c_l / scale_weight);
            candidates.push(PenaltyCandidate {
                matrix: s_l.scaled(1.0 / c_l, "normalized measure-jet scale penalty")?,
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
        // The Primary is exactly the jet-energy functional pulled back through
        // the center evaluation map. It is independent of `double_penalty`:
        // statistical selection is a distinct REML component below, never a
        // fixed coefficient toll fused into this estimand.
        let penalty =
            constructive_pullback_center_form(&kz, &q_form, "measure-jet primary penalty")?;
        let c_primary = constructive_frobenius_scale(&penalty);
        fused_penalty_normalization_scale = Some(c_primary);
        // Declare the energy's structural null frame on the shipped Primary
        // (#2761, the #2445 mechanism): the affine head is null by theorem, and
        // the pullback's NUMERICAL rank falls as the representer range grows, so
        // a rank test on the shipped matrix would let a design-moving ℓ decide
        // the double-penalty topology between outer trials.
        let mut primary =
            penalty.scaled(1.0 / c_primary, "normalized measure-jet primary penalty")?;
        if let Some(frame) = measure_jet_primary_structural_null_frame(&z, m, head_width)? {
            primary = primary.with_structural_null_frame(
                frame,
                "measure-jet primary structural null declaration",
            )?;
        }
        candidates.push(PenaltyCandidate {
            matrix: primary,
            source: PenaltySource::Primary,
            normalization_scale: c_primary,
            kronecker_factors: None,
            op: None,
        });
    }
    // Explicit null recovery is a genuine statistical component: penalize the
    // affine/null FUNCTION projection under the empirical-measure mass metric,
    // and let REML select its strength independently in both modes. This is the
    // standard double-penalty decomposition (roughness + null component); no
    // coefficient identity and no hard-coded mixture changes the Primary.
    if spec.double_penalty {
        let null_penalty = affine_function_nullspace_quadratic(&kz, centers.view(), masses.view())?;
        let (_, c_null) = normalize_penalty(null_penalty.dense());
        candidates.push(PenaltyCandidate {
            matrix: null_penalty
                .scaled(1.0 / c_null, "normalized measure-jet null-function penalty")?,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_null,
            kronecker_factors: None,
            op: None,
        });
        // Decide the ridge's fate in THIS chart, the way the term-collection
        // chokepoint decides it in its own (#2433's repair, which periodic
        // Duchon already carries verbatim, extended here for #2761).
        //
        // The collection applies its global gauge and then rebuilds the ridge
        // from `null(Primary_constrained)`; a chart that has taken the last
        // structural null direction leaves nothing for the ridge to shrink and
        // the collection drops it. A frozen composed chart — which is what
        // every outer ψ trial and every predict-time replay rebuilds in — is
        // exactly such a chart for a 1-D measure-jet term, where the parametric
        // orthogonalization absorbs the whole affine head. Emitting the raw
        // ridge there produces a LOCAL topology of 2 against the collection's
        // cached 1, and the incremental realizer aborts the outer search with
        // `topology changed ... active_penalties=2, cached_penalties=1`.
        //
        // Running the same rebuild locally makes the two layers agree by
        // construction instead of by coincidence. In the cold chart the head is
        // still present, the rebuild keeps the ridge, and this is a no-op on
        // the shipped topology.
        let primary_physical = candidates
            .iter()
            .find(|candidate| matches!(candidate.source, PenaltySource::Primary))
            .map(|candidate| {
                candidate.matrix.scaled(
                    candidate.normalization_scale,
                    "physical measure-jet primary",
                )
            })
            .transpose()?;
        if let Some(primary_physical) = primary_physical {
            let width = primary_physical.nrows();
            for candidate in &mut candidates {
                if !matches!(candidate.source, PenaltySource::DoublePenaltyNullspace) {
                    continue;
                }
                let ridge_physical = candidate.matrix.scaled(
                    candidate.normalization_scale,
                    "physical measure-jet null-function penalty",
                )?;
                match super::rebuild_metric_consistent_ridge(&primary_physical, &ridge_physical)? {
                    Some(rebuilt) => {
                        let normalized = super::normalize_constructive_penalty_candidate(
                            rebuilt,
                            PenaltySource::DoublePenaltyNullspace,
                        )?;
                        candidate.matrix = normalized.matrix;
                        candidate.normalization_scale = normalized.normalization_scale;
                    }
                    None => {
                        candidate.matrix = ConstructiveQuadratic::zero(width);
                        candidate.normalization_scale = 1.0;
                    }
                }
                candidate.kronecker_factors = None;
                candidate.op = None;
            }
        }
    }
    let filtered = filter_penalty_candidates(candidates)?;
    // #2225: compute the errors-in-variables input-noise scale while `centers`
    // is still owned; it is moved into the metadata `centers` field below.
    let sigma_coord = measure_jet_input_noise_scale(data, centers.view())?;
    Ok(BasisBuildResult {
        design,
        affine_offset: None,
        active_penalties: filtered.active,
        dropped_penalties: filtered.dropped,
        metadata: BasisMetadata::MeasureJet {
            centers,
            input_scale: crate::IsotropicScale::ONE,
            // The realized range from `realize_measure_jet_geometry`, in the
            // same frame as `centers` and `eps_band`.  Unlike the other three
            // Euclidean families the term-collection wrapper does NOT restore
            // an original-units value over this, so the standardized tag
            // survives to every consumer (#2636).
            length_scale: crate::StandardizedUnits::new(length_scale),
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
            // Perpendicular off-manifold residual scale of the fit rows in the
            // centers' frame — the errors-in-variables input-noise scale (#2225).
            sigma_coord,
        },
        kronecker_factored: None,
        joint_null_rotation: None,
    })
}

/// Exact ψ-jets of the REALIZED measure-jet penalty candidates, adapted to
/// the anisotropic group-ψ carrier the spatial optimizer consumes.
///
/// Coordinates (the layout contract for the registration arm):
/// - per-level (spectral) mode: `[ln ℓ?, α, ln τ]` — order is absorbed by the
///   REML-learned scale amplitudes; `ln τ` is retained as an inert coordinate;
/// - single-scale mode: `[ln ℓ?]`, because its energy dials are fixed.
///
/// Only `ln ℓ` moves the design. It also moves every coefficient-space penalty
/// pullback through the center evaluation map `E(ℓ)`; `(α, ln τ)` move only the
/// per-scale center-value forms. Exact diagonal and mixed product-rule jets are
/// emitted before Frobenius normalization.
/// Penalty derivatives are routed through the SAME constrained Frobenius
/// normalization as the fit-time candidates
/// (`normalize_penaltywith_psi_derivatives` + the cross rule), so criterion
/// value and criterion derivative share one normalization — the #901 lesson
/// made structural. The function-space null candidate has nonzero `ln ℓ` jets
/// and zero `(α, ln τ)` jets. The per-candidate layout follows the builder's
/// ORIGINAL order (scale candidates or Primary, then null component); consumers
/// align to the FITTED penalty list via
/// `ActivePenaltyInfo.original_index` when the candidate filter dropped
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
    let p = geom.kz.ncols();
    let m = geom.centers.nrows();
    let m_aug = m + geom.head_lift.ncols();

    struct LengthScaleJets {
        evaluation_first: Array2<f64>,
        evaluation_second: Array2<f64>,
        design_first: Array2<f64>,
        design_second: Array2<f64>,
    }

    // The Gaussian representer range moves both the FIT design and the center
    // evaluation map `E = [K_cc | A_head] Z`. The affine head is ℓ-invariant,
    // so its raw derivative columns are exactly zero before applying the frozen
    // Gauge section. Keeping `Z` frozen is the replay contract: rank/gauge
    // realization happens once at fit time, then every ψ trial differentiates
    // the same coefficient chart.
    let length_scale_jets = if spec.learn_length_scale {
        let (dk_data, d2k_data) =
            measure_jet_design_log_length_jets(data, geom.centers.view(), geom.length_scale)?;
        let mut dk_data_aug = Array2::<f64>::zeros((n, m_aug));
        let mut d2k_data_aug = Array2::<f64>::zeros((n, m_aug));
        dk_data_aug.slice_mut(ndarray::s![.., ..m]).assign(&dk_data);
        d2k_data_aug
            .slice_mut(ndarray::s![.., ..m])
            .assign(&d2k_data);

        let (dk_centers, d2k_centers) = measure_jet_design_log_length_jets(
            geom.centers.view(),
            geom.centers.view(),
            geom.length_scale,
        )?;
        let mut dk_centers_aug = Array2::<f64>::zeros((m, m_aug));
        let mut d2k_centers_aug = Array2::<f64>::zeros((m, m_aug));
        dk_centers_aug
            .slice_mut(ndarray::s![.., ..m])
            .assign(&dk_centers);
        d2k_centers_aug
            .slice_mut(ndarray::s![.., ..m])
            .assign(&d2k_centers);

        Some(LengthScaleJets {
            evaluation_first: geom.coefficient_gauge.restrict_design(&dk_centers_aug),
            evaluation_second: geom.coefficient_gauge.restrict_design(&d2k_centers_aug),
            design_first: geom.coefficient_gauge.restrict_design(&dk_data_aug),
            design_second: geom.coefficient_gauge.restrict_design(&d2k_data_aug),
        })
    } else {
        None
    };

    let coord_offset = usize::from(length_scale_jets.is_some());
    let n_coords = coord_offset + if geom.per_level { 2 } else { 0 };
    let pairs: Vec<(usize, usize)> = (0..n_coords)
        .flat_map(|a| ((a + 1)..n_coords).map(move |b| (a, b)))
        .collect();
    let zero_p = || Array2::<f64>::zeros((p, p));

    struct RawPenaltyJets {
        value: Array2<f64>,
        first: Vec<Array2<f64>>,
        second_diag: Vec<Array2<f64>>,
        cross: Vec<Array2<f64>>,
    }

    let sandwich = |form: &Array2<f64>| pullback_center_form(&geom.kz, form);
    let length_diag = |form: &Array2<f64>| {
        let jets = length_scale_jets
            .as_ref()
            .expect("length-scale form jets require an enrolled length coordinate");
        pullback_center_form_log_length_jets(
            &geom.kz,
            &jets.evaluation_first,
            &jets.evaluation_second,
            form,
        )
    };
    let length_cross = |form_first: &Array2<f64>| {
        let jets = length_scale_jets
            .as_ref()
            .expect("length-scale cross jets require an enrolled length coordinate");
        pullback_center_form_log_length_cross(&geom.kz, &jets.evaluation_first, form_first)
    };

    // Raw (pre-normalization) value + exact jet stacks per ORIGINAL candidate.
    // Coordinate order is `[lnℓ?, α, lnτ]` in multiscale mode and `[lnℓ?]`
    // in single-scale mode. Candidate order exactly mirrors the value builder:
    // scale candidates or Primary first, then the optional null-component
    // candidate. Active filtering aligns through `ActivePenaltyInfo::original_index`.
    // The single-scale Primary, when there is one. `None` in per-level mode,
    // which emits scale candidates instead and therefore never rebuilds the
    // null component.
    let mut single_scale_primary: Option<ConstructiveQuadratic> = None;
    let mut raw: Vec<RawPenaltyJets> = if geom.per_level {
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
        let alpha_coord = coord_offset;
        let tau_coord = coord_offset + 1;
        let mut raw = Vec::with_capacity(l_count + usize::from(spec.double_penalty));
        for level in 0..l_count {
            let chunk = &forms[6 * level..6 * level + 6];
            let mut first: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
            let mut second_diag: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
            first[alpha_coord] = sandwich(&chunk[1]);
            first[tau_coord] = sandwich(&chunk[3]);
            second_diag[alpha_coord] = sandwich(&chunk[2]);
            second_diag[tau_coord] = sandwich(&chunk[4]);
            if coord_offset == 1 {
                let (ell_first, ell_second) = length_diag(&chunk[0]);
                first[0] = ell_first;
                second_diag[0] = ell_second;
            }
            let mut cross: Vec<Array2<f64>> = (0..pairs.len()).map(|_| zero_p()).collect();
            for (pair_idx, &(a, b)) in pairs.iter().enumerate() {
                cross[pair_idx] = if coord_offset == 1 && a == 0 && b == alpha_coord {
                    length_cross(&chunk[1])
                } else if coord_offset == 1 && a == 0 && b == tau_coord {
                    length_cross(&chunk[3])
                } else if a == alpha_coord && b == tau_coord {
                    sandwich(&chunk[5])
                } else {
                    zero_p()
                };
            }
            raw.push(RawPenaltyJets {
                value: sandwich(&chunk[0]),
                first,
                second_diag,
                cross,
            });
        }
        raw
    } else {
        // Single-scale mode enrolls no `(s, α, lnτ)` penalty dials. It still
        // emits the pure Primary and, when requested, a separate REML null
        // component; an opt-in `lnℓ` coordinate differentiates both pullbacks.
        let q_form = measure_jet_energy_form(
            geom.centers.view(),
            geom.masses.view(),
            &band,
            geom.order_s_eval,
            spec.alpha,
            spec.tau0,
        )?;
        let mut first: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
        let mut second_diag: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
        if coord_offset == 1 {
            let (ell_first, ell_second) = length_diag(&q_form);
            first[0] = ell_first;
            second_diag[0] = ell_second;
        }
        // Keep the Primary the builder would emit: the null component's shipped
        // matrix is a REBUILD off it, so the producer needs the same object to
        // differentiate the same thing (see below).
        single_scale_primary = Some(constructive_pullback_center_form(
            &geom.kz,
            &q_form,
            "measure-jet primary penalty",
        )?);
        if let (Some(primary), Some(frame)) = (
            single_scale_primary.as_mut(),
            measure_jet_primary_structural_null_frame(&geom.z, m, geom.head_lift.ncols())?,
        ) {
            *primary = primary.clone().with_structural_null_frame(
                frame,
                "measure-jet primary structural null declaration",
            )?;
        }
        vec![RawPenaltyJets {
            value: sandwich(&q_form),
            first,
            second_diag,
            cross: Vec::new(),
        }]
    };

    if spec.double_penalty {
        let null_center =
            affine_function_nullspace_center_quadratic(geom.centers.view(), geom.masses.view())?;
        let null_form = null_center.dense();
        let mut first: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
        let mut second_diag: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
        if coord_offset == 1 {
            let (ell_first, ell_second) = length_diag(null_form);
            first[0] = ell_first;
            second_diag[0] = ell_second;
        }
        let mut value = sandwich(null_form);
        // The builder does NOT ship this raw pullback when a Primary exists: it
        // ships `rebuild_metric_consistent_ridge`'s output, `R = N M Nᵀ` with
        // `M = Nᵀ (EᵀH₀E) N` and `N` the Primary's declared structural null
        // frame. Differentiating the raw pullback instead is an
        // objective↔gradient desync on the `ln ℓ` coordinate, and #2761
        // measured it as the WHOLE of that coordinate's gradient error:
        //
        //   arm                       analytic     Ridders FD     rel
        //   double_penalty = true    -1.124278e1  -1.149171e1   2.2e-2
        //   double_penalty = false   -1.1255398e1 -1.1255398e1   9e-10
        //
        // with the total's per-atom breakdown putting it in `logdet_S`
        // (−0.2369) and `fixed_beta` (+0.4858). λ_null being tiny does not
        // shrink it: on the directions the Primary annihilates, `S_λ` IS
        // `λ_null·S_null`, so `λ_null` cancels out of
        // `tr(S_λ⁺ ∂S_λ/∂ψ)` and a wrong `∂S_null/∂ψ` lands at full size.
        //
        // The exact jets of the rebuilt object are `N (Nᵀ ∂S_raw N) Nᵀ`, since
        // `N` is ψ-fixed by construction (it is a declaration, not a rank
        // test). They are numerically ZERO here — `N`'s columns carry no
        // representer coefficients, so `E·N` is ℓ-invariant and `M` cannot
        // move — but computing them rather than asserting them keeps the
        // producer correct if a future frame does move.
        if let Some(primary) = single_scale_primary.as_ref() {
            let ridge_physical = ConstructiveQuadratic::from_energy_factor(
                null_center.factor().dot(&geom.kz),
                "measure-jet affine/null coefficient penalty",
            )?;
            match super::rebuild_metric_consistent_ridge(primary, &ridge_physical)? {
                Some(rebuilt) => {
                    let frame = primary
                        .structural_null_frame()
                        .cloned()
                        .unwrap_or_else(|| Array2::<f64>::zeros((p, 0)));
                    for coord in 0..n_coords {
                        first[coord] = restrict_jet_to_frame(&first[coord], &frame);
                        second_diag[coord] = restrict_jet_to_frame(&second_diag[coord], &frame);
                    }
                    value = rebuilt.dense().clone();
                }
                None => {
                    // The rebuild declined, so the builder ships an exact zero
                    // and the candidate is dropped. A dropped candidate has no
                    // derivative.
                    for coord in 0..n_coords {
                        first[coord] = zero_p();
                        second_diag[coord] = zero_p();
                    }
                    value = zero_p();
                }
            }
        }
        raw.push(RawPenaltyJets {
            value,
            first,
            second_diag,
            // H₀ is independent of α and τ; its only moving object is E(ℓ),
            // so every mixed coordinate derivative is zero.
            cross: (0..pairs.len()).map(|_| zero_p()).collect(),
        });
    }

    let n_cands = raw.len();
    let mut penalties_first: Vec<Vec<Array2<f64>>> =
        (0..n_coords).map(|_| Vec::with_capacity(n_cands)).collect();
    let mut penalties_second_diag: Vec<Vec<Array2<f64>>> =
        (0..n_coords).map(|_| Vec::with_capacity(n_cands)).collect();
    // Cross matrices per pair per candidate, precomputed eagerly (the
    // candidate count is the band length, not the data size) and served
    // through the on-demand provider.
    let mut crosses: Vec<Vec<Array2<f64>>> = (0..pairs.len()).map(|_| Vec::new()).collect();
    for candidate in &raw {
        let s_raw = &candidate.value;
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
            let (_, s_first, s_second, _) = normalize_penaltywith_psi_derivatives(
                s_raw,
                &candidate.first[coord],
                &candidate.second_diag[coord],
            );
            penalties_first[coord].push(s_first);
            penalties_second_diag[coord].push(s_second);
        }
        for (pair_idx, &(a, b)) in pairs.iter().enumerate() {
            let cross_raw_mat = normalize_penalty_cross_psi_derivative(
                s_raw,
                &candidate.first[a],
                &candidate.first[b],
                &candidate.cross[pair_idx],
                c,
            );
            crosses[pair_idx].push(cross_raw_mat);
        }
    }

    let pair_index: Vec<((usize, usize), Vec<Array2<f64>>)> =
        pairs.iter().copied().zip(crosses.into_iter()).collect();
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
    let mut design_first: Vec<Array2<f64>> = (0..n_coords)
        .map(|_| Array2::<f64>::zeros((n, p)))
        .collect();
    let mut design_second_diag: Vec<Array2<f64>> = (0..n_coords)
        .map(|_| Array2::<f64>::zeros((n, p)))
        .collect();
    if let Some(jets) = &length_scale_jets {
        design_first[0] = jets.design_first.clone();
        design_second_diag[0] = jets.design_second.clone();
    }
    Ok(AnisoBasisPsiDerivatives {
        design_first,
        design_second_diag,
        design_second_cross: Vec::new(),
        design_second_cross_pairs: Vec::new(),
        penalties_first,
        penalties_second_diag,
        penalties_cross_pairs: pairs,
        penalties_cross_provider: Some(provider),
        implicit_operator: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Box–Muller standard normal from a 64-bit LCG state — a
    /// self-contained noise generator (no external RNG dependency).
    fn lcg_normal(state: &mut u64) -> f64 {
        let mut next = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 53 bits → uniform (0, 1).
            (((*state >> 11) as f64) + 0.5) / (1u64 << 53) as f64
        };
        let u1 = next();
        let u2 = next();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// The perpendicular off-manifold residual estimator recovers a KNOWN
    /// ambient noise scale on a 1-D manifold (a line) embedded in 2-D: points
    /// sampled along the tangent with isotropic-perpendicular Gaussian noise of
    /// scale σ, centers spaced along the line. The local-PCA smallest-eigenvalue
    /// floor must return ≈ σ (#2225).
    #[test]
    pub(crate) fn input_noise_scale_recovers_known_perpendicular_sigma() {
        // Line direction (unit) and its perpendicular in 2-D.
        let tang = [1.0 / 5f64.sqrt(), 2.0 / 5f64.sqrt()];
        let perp = [2.0 / 5f64.sqrt(), -1.0 / 5f64.sqrt()];
        let sigma = 0.05_f64;
        let n = 600usize;
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut data = Array2::<f64>::zeros((n, 2));
        for j in 0..n {
            // Tangential coordinate marches deterministically over [0, 3].
            let t = 3.0 * (j as f64) / (n as f64 - 1.0);
            let noise = sigma * lcg_normal(&mut state);
            for a in 0..2 {
                data[(j, a)] = t * tang[a] + noise * perp[a];
            }
        }
        // Centers along the line (on the noiseless manifold): plenty of points
        // per cell to span the tangent.
        let n_centers = 8usize;
        let mut centers = Array2::<f64>::zeros((n_centers, 2));
        for i in 0..n_centers {
            let t = 3.0 * (i as f64 + 0.5) / (n_centers as f64);
            for a in 0..2 {
                centers[(i, a)] = t * tang[a];
            }
        }
        let est = measure_jet_input_noise_scale(data.view(), centers.view())
            .expect("estimate ok")
            .expect("noise scale present");
        // Sample smallest-eigenvalue floor is mildly downward-biased; require it
        // within 40% of the truth (central estimate, not a tuned tolerance).
        assert!(
            (est - sigma).abs() <= 0.4 * sigma,
            "estimated σ_coord {est} far from true {sigma}"
        );
    }

    /// Too few points per cell (cannot span a d-dim tangent) ⇒ no estimate,
    /// so the caller leaves Var_input disabled rather than invent a scale.
    #[test]
    pub(crate) fn input_noise_scale_none_when_cells_too_small() {
        let data = array![[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]];
        let centers = array![[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]];
        // Each point is its own nearest center (1 point per cell < d + 1 = 3).
        assert!(
            measure_jet_input_noise_scale(data.view(), centers.view())
                .expect("estimate ok")
                .is_none()
        );
    }

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

    /// The default is single-scale mode at ANY center count: one Primary
    /// jet-energy candidate plus the independently REML-selected affine/null
    /// component requested by the default `double_penalty`. Multiscale (the
    /// per-scale spectral split + ψ dials) is an EXPLICIT opt-in
    /// (`spec.multiscale`, the DSL `mjs(…, multiscale=true)`) — there is no
    /// center-count auto-gate (#1116). `measure_jet_multiscale_mode` is the
    /// single source for this decision.
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
        // multiscale: one pure Primary plus one function-space null component.
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
            built_single.active_penalties.len(),
            2,
            "single-scale double-penalty mode emits Primary + affine/null component"
        );
        assert!(matches!(
            built_single.active_penalties[0].info.source,
            PenaltySource::Primary
        ));
        assert!(matches!(
            built_single.active_penalties[1].info.source,
            PenaltySource::DoublePenaltyNullspace
        ));
        // The explicit opt-in flips to multiscale at the SAME center count: the
        // per-scale spectral split (several candidates) plus the same explicit
        // null-component candidate, strictly more candidates than single-scale.
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
            built_multi.active_penalties.len() > built_single.active_penalties.len(),
            "multiscale mode emits the per-scale spectral split plus null selection, got {} (vs single-scale {})",
            built_multi.active_penalties.len(),
            built_single.active_penalties.len()
        );
    }

    /// An explicit order pins the Mellin weights and fuses the band into a
    /// single Primary candidate. Disabling explicit null recovery leaves exactly
    /// that candidate; enabling it must never alter the Primary itself.
    #[test]
    pub(crate) fn fused_mode_without_double_penalty_emits_single_primary_candidate() {
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
            double_penalty: false,
            ..MeasureJetBasisSpec::default()
        };
        let built = build_measure_jet_basis(data.view(), &spec).expect("fused build");
        assert_eq!(
            built.active_penalties.len(),
            1,
            "single-scale mode without null recovery emits exactly one Primary"
        );
        assert!(matches!(
            built.active_penalties[0].info.source,
            PenaltySource::Primary
        ));
        let BasisMetadata::MeasureJet { order_s, .. } = &built.metadata else {
            panic!("measure-jet build must return MeasureJet metadata");
        };
        assert_eq!(*order_s, 1.3, "explicit order must persist verbatim");
    }

    /// The single-scale affine head is a gauge-fixed decomposition, not a
    /// coefficient ridge: RBF center values are exactly mass-orthogonal to the
    /// supported affine space, and replacing those directions with the head
    /// keeps the RAW chart exactly `m` wide. The collection's parametric
    /// orthogonalization then removes the head's constant, landing the FIT
    /// chart at `m - 1` — the width this test asserted directly before #2751,
    /// when the head omitted the constant and the centering took a linear
    /// direction instead.
    #[test]
    pub(crate) fn single_scale_affine_head_gauge_annihilates_center_cross() {
        let n = 90usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 {
                3.0 * t
            } else {
                (2.0 * std::f64::consts::PI * t).sin() + 0.2 * t
            }
        });
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 18 },
            double_penalty: false,
            multiscale: false,
            ..MeasureJetBasisSpec::default()
        };
        let geom = realize_measure_jet_geometry(data.view(), &spec).expect("realized geometry");
        let m = geom.centers.nrows();
        let head_width = geom.head_lift.ncols();
        assert!(head_width > 0, "fixture must realize an affine head");
        assert_eq!(
            geom.z.ncols(),
            m,
            "the affine head replaces the RBF block's affine directions one for one: the RAW \
             chart is exactly m wide (m - head_width representers + head_width head columns). \
             The collection's parametric orthogonalization then removes the constant, landing \
             the FIT chart at m - 1 (#2751)"
        );
        let rbf_rank = m - head_width;
        let z_rbf = geom.z.slice(ndarray::s![..m, ..rbf_rank]).to_owned();
        let k_cc =
            measure_jet_design_matrix(geom.centers.view(), geom.centers.view(), geom.length_scale)
                .expect("center kernel");
        let affine = measure_jet_affine_value_basis(geom.centers.view(), geom.masses.view());
        assert_eq!(affine.ncols(), head_width);
        let mut weighted_affine = affine.clone();
        for (i, mut row) in weighted_affine.outer_iter_mut().enumerate() {
            row.mapv_inplace(|v| v * geom.masses[i]);
        }
        let constraint_cross = k_cc.t().dot(&weighted_affine);
        let residual = constraint_cross.t().dot(&z_rbf);
        let scale = constraint_cross
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            residual.iter().all(|value| value.abs() <= 1e-10 * scale),
            "A^T W Kcc Z_rbf must vanish; max residual {:.3e}",
            residual
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        );
    }

    /// `double_penalty` adds a distinct evidence-selected component and cannot
    /// mutate the jet-energy estimand carried by Primary.
    #[test]
    pub(crate) fn double_penalty_leaves_primary_matrix_unchanged() {
        let n = 64usize;
        let data = Array2::<f64>::from_shape_fn((n, 2), |(i, k)| {
            let t = i as f64 / (n as f64 - 1.0);
            if k == 0 { 2.5 * t } else { (4.0 * t).cos() }
        });
        let base = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 16 },
            order_s: 1.25,
            double_penalty: false,
            ..MeasureJetBasisSpec::default()
        };
        let without = build_measure_jet_basis(data.view(), &base).expect("primary-only build");
        let with = build_measure_jet_basis(
            data.view(),
            &MeasureJetBasisSpec {
                double_penalty: true,
                ..base.clone()
            },
        )
        .expect("double-penalty build");
        assert_eq!(without.active_penalties.len(), 1);
        assert_eq!(with.active_penalties.len(), 2);
        assert!(matches!(
            without.active_penalties[0].info.source,
            PenaltySource::Primary
        ));
        assert!(matches!(
            with.active_penalties[0].info.source,
            PenaltySource::Primary
        ));
        assert!(matches!(
            with.active_penalties[1].info.source,
            PenaltySource::DoublePenaltyNullspace
        ));
        assert!(
            without.active_penalties[0]
                .matrix
                .iter()
                .zip(with.active_penalties[0].matrix.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-13),
            "turning on null recovery must not modify Primary"
        );
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
            // MeasureJet freezes its range STANDARDIZED and replays it
            // verbatim; the tag is what records that it is the odd family out.
            length_scale: length_scale.standardized_value(),
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
                sigma_coord: None,
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
                .active_penalties
                .into_iter()
                .map(|penalty| penalty.matrix)
                .collect::<Vec<_>>()
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
        // The function-space null candidate is independent of α and τ.
        for coord in 0..2 {
            assert!(
                derivs.penalties_first[coord][l_count]
                    .iter()
                    .all(|v| *v == 0.0),
                "null-component candidate must have zero (α, lnτ) drift"
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

    /// Design-moving ℓ dial (#1116): the producer's design jets and every
    /// normalized penalty candidate's jets must match central differences of the
    /// REBUILT objects under frozen geometry. Although the center-value forms
    /// `Q` and `H₀` are ℓ-invariant, their coefficient pullbacks `E(ℓ)ᵀQ E(ℓ)`
    /// and `E(ℓ)ᵀH₀E(ℓ)` are not.
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
        assert_eq!(
            derivs.penalties_first[0].len(),
            2,
            "single-scale double penalty carries Primary + affine/null component"
        );
        // Rebuild design and normalized penalties at ℓ·e^{±h}; the explicit
        // positive length_scale is honored verbatim while the frozen transform
        // keeps the coefficient chart fixed.
        let ell0 = frozen.length_scale;
        let build_at = |ell: f64| {
            let trial = MeasureJetBasisSpec {
                length_scale: ell,
                ..frozen.clone()
            };
            build_measure_jet_basis(data.view(), &trial).expect("trial build")
        };
        let h: f64 = 1e-4;
        let plus = build_at(ell0 * h.exp());
        let minus = build_at(ell0 * (-h).exp());
        let at = build_at(ell0);
        assert_eq!(
            plus.active_penalties.len(),
            2,
            "fixture must keep both candidates active"
        );
        assert_eq!(
            minus.active_penalties.len(),
            2,
            "fixture must keep both candidates active"
        );
        assert_eq!(
            at.active_penalties.len(),
            2,
            "fixture must keep both candidates active"
        );

        let x_plus = plus.design.to_dense();
        let x_minus = minus.design.to_dense();
        let x_0 = at.design.to_dense();
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

        for candidate in 0..2 {
            let fd_penalty_first = (&plus.active_penalties[candidate].matrix
                - &minus.active_penalties[candidate].matrix)
                / (2.0 * h);
            let fd_penalty_second = (&plus.active_penalties[candidate].matrix
                - &(&at.active_penalties[candidate].matrix * 2.0)
                + &minus.active_penalties[candidate].matrix)
                / (h * h);
            // A central difference cannot resolve a derivative below its own
            // cancellation noise: differencing entries of size `E` at step `h`
            // leaves `~ε·E/h` in the first difference and `~ε·E/h²` in the
            // second, whatever the true derivative is. The null component's
            // shipped matrix (the rebuilt metric-consistent ridge) is EXACTLY
            // ℓ-invariant, so its analytic jets are exactly zero and its FD is
            // pure noise — measured at 3.5e-13 against a `1e-12` scale floor
            // that predates the exact answer. Grading that against a relative
            // tolerance alone asserts the ORACLE is exact, which it is not.
            // The factor 8 covers the handful of roundings between the two
            // rebuilds; it is not a fudge on the gradient, which is still
            // graded relatively wherever the FD resolves anything.
            let entry_scale = [&plus, &minus, &at]
                .iter()
                .flat_map(|built| built.active_penalties[candidate].matrix.iter())
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            let first_floor = 8.0 * f64::EPSILON * entry_scale / h;
            let second_floor = 8.0 * f64::EPSILON * entry_scale / (h * h);
            let first_scale = fd_penalty_first
                .iter()
                .fold(1e-12_f64, |acc, value| acc.max(value.abs()));
            let second_scale = fd_penalty_second
                .iter()
                .fold(1e-10_f64, |acc, value| acc.max(value.abs()));
            for (analytic, finite_difference) in derivs.penalties_first[0][candidate]
                .iter()
                .zip(fd_penalty_first.iter())
            {
                assert!(
                    (analytic - finite_difference).abs() <= 1e-4 * first_scale + first_floor,
                    "candidate {candidate} ∂S~/∂lnℓ: analytic {analytic:.6e} vs FD \
                     {finite_difference:.6e} (rel budget {:.3e}, oracle floor {first_floor:.3e})",
                    1e-4 * first_scale
                );
            }
            for (analytic, finite_difference) in derivs.penalties_second_diag[0][candidate]
                .iter()
                .zip(fd_penalty_second.iter())
            {
                assert!(
                    (analytic - finite_difference).abs() <= 5e-3 * second_scale + second_floor,
                    "candidate {candidate} ∂²S~/∂lnℓ²: analytic {analytic:.6e} vs FD \
                     {finite_difference:.6e} (rel budget {:.3e}, oracle floor {second_floor:.3e})",
                    5e-3 * second_scale
                );
            }
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
            // MeasureJet freezes its range STANDARDIZED and replays it
            // verbatim; the tag is what records that it is the odd family out.
            length_scale: length_scale.standardized_value(),
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
                sigma_coord: None,
            }),
        };
        // Per-level mode: one candidate per band scale plus the function-space
        // null component, and the count must survive replay bit-for-bit.
        assert_eq!(
            first.active_penalties.len(),
            eps_band.len() + 1,
            "per-level mode must emit one candidate per scale + null component"
        );
        let second = build_measure_jet_basis(data.view(), &replay_spec).expect("replay build");
        let x1 = first.design.to_dense();
        let x2 = second.design.to_dense();
        assert_eq!(x1.shape(), x2.shape());
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() <= 1e-12, "design replay drift: {a} vs {b}");
        }
        assert_eq!(first.active_penalties.len(), second.active_penalties.len());
        for (p1, p2) in first
            .active_penalties
            .iter()
            .zip(second.active_penalties.iter())
        {
            for (a, b) in p1.matrix.iter().zip(p2.matrix.iter()) {
                assert!((a - b).abs() <= 1e-12, "penalty replay drift: {a} vs {b}");
            }
        }
    }
}
