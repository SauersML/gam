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
//!   and the `α` ψ dial are the multiscale-mode realization, engaged
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
//! - **Penalty-dial design drift is identically zero**: the (s, α) dials
//!   reweight only the jet-energy penalty, never the Gaussian representer
//!   design (`∂X/∂{s,α} ≡ 0`), so those channels are penalty-only
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
//! frame notes (`docs/measure_jet_frame.md`); its §5 extrapolation pricing is
//! `measure_jet_predict.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use faer::Side;

use gam_linalg::faer_ndarray::{
    FaerEigh, FaerSvd, rrqr_nullspace_basis,
};

use super::{
    AnisoBasisPsiDerivatives, AnisoPenaltyCrossProvider, BasisBuildResult, BasisError,
    BasisMetadata, CenterStrategy, ConstructiveQuadratic, PenaltyCandidate, PenaltySource,
    filter_penalty_candidates, normalize_penalty, normalize_penalty_cross_psi_derivative,
    normalize_penaltywith_psi_derivatives, select_centers_by_strategy, stable_euclidean_norm,
    trace_of_product,
};

/// Truncation radius of the Gaussian profile in units of the scale ε: weights
/// beyond `3ε` are below `e^{-4.5} ≈ 1.1e-2` of the peak and are dropped from
/// both the local fit and the `q^(1−2α)` outer weight. This is an absolute
/// kernel-weight cutoff; using the same truncated q keeps the discrete
/// functional self-consistent, but it is not a relative tail-error bound.
pub(crate) const MEASURE_JET_PROFILE_CUTOFF: f64 = 3.0;

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
    /// mirroring how `spatial_term_supports_hyper_optimization` declines an
    /// explicitly-scaled Matérn or Duchon term (gam#3020).
    #[serde(default = "measure_jet_learn_length_scale_default")]
    pub learn_length_scale: bool,
    /// Explicit opt-in for multiscale mode: the per-scale spectral penalty
    /// split plus the `α` outer ψ dial. `false` (default) keeps the
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

/// Rank-revealing pseudo-inverse of a symmetric Gram.
///
/// `assembly_band` bounds, in spectral norm, the rounding the Gram's formation
/// left in `a`. An eigenvalue at or below that plus the eigensolver's own band
/// ([`gam_linalg::roundoff::resolved_eigenvalue_band`]) is not resolved from
/// zero, so its direction is excluded from the affine fit.
pub(crate) fn symmetric_pseudoinverse(
    a: &Array2<f64>,
    assembly_band: f64,
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
    let rank_tol = gam_linalg::roundoff::resolved_eigenvalue_band(&evals.to_vec(), assembly_band);
    let mut scaled = evecs.clone();
    for (k, mut col) in scaled.axis_iter_mut(Axis(1)).enumerate() {
        let lam = evals[k].max(0.0);
        let inv = if lam > rank_tol { 1.0 / lam } else { 0.0 };
        col.mapv_inplace(|v| v * inv);
    }
    Ok(scaled.dot(&evecs.t()))
}

/// Spectral-norm bound on the rounding in the local affine Gram
/// `G = (ΦᵀWΦ)/q − a·aᵀ`, `a = Φᵀw/q`, `q = Σw`, as the measure-jet energy
/// forms it from `ml` weighted neighbors (`w ≥ 0`).
///
/// Per entry, with `P = |Φ|ᵀW|Φ|/q` and `ã = |Φ|ᵀw/q`: the second-moment sum
/// rounds `ml + 1` deep (the weight product, the feature product, `ml − 1`
/// additions), `q` `ml − 1` deep and the quotient once, so `(ΦᵀWΦ)/q` errs by
/// `γ_{2ml+1}·P`. The mean's sum rounds `ml` deep and its quotient `ml`, so
/// `a·aᵀ` errs by `γ_{4ml+1}·ããᵀ`, and the subtraction adds one. Hence
/// `|E| ≤ γ_{4ml+2}·(P + ããᵀ)` entrywise. That majorant is PSD, so
/// `‖E‖₂ ≤ γ_{4ml+2}·(tr P + ‖ã‖²)`, and `ã_r² ≤ P_rr` (Jensen) gives
/// `‖E‖₂ ≤ 2·γ_{4ml+2}·tr P` with `tr P = Σ_a w_a‖φ_a‖²/q`. The uncentered
/// formula cancels, so this band scales with the second moment, not with `G`.
fn local_affine_gram_assembly_band(phi: &Array2<f64>, w: &Array1<f64>, q: f64) -> f64 {
    let ml = phi.nrows();
    let second_moment_trace: f64 = phi
        .outer_iter()
        .zip(w.iter())
        .map(|(row, wa)| wa * row.dot(&row))
        .sum::<f64>()
        / q;
    2.0 * gam_linalg::roundoff::accumulation_growth(4 * ml + 2) * second_moment_trace
}

/// Rank-adapted center values of the measure-jet energy's affine null space.
///
/// The first column is the constant. The remaining columns are the supported
/// ambient-linear directions returned by `measure_jet_affine_head_transform`.
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
pub fn affine_function_nullspace_form(
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
    // `AᵀWA` over the m centers, each summand rounding twice (the mass scaling
    // and the product) before the additions.
    let weighted_row_norm_sum: f64 = affine
        .outer_iter()
        .zip(masses.iter())
        .map(|(row, w)| w.abs() * row.dot(&row))
        .sum();
    let affine_gram_pinv = symmetric_pseudoinverse(
        &affine_gram,
        gam_linalg::roundoff::weighted_gram_assembly_band(m, 2, weighted_row_norm_sum),
        "affine function-space Gram",
    )?;
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

/// Frobenius scale of a constructive quadratic, with the degenerate convention
/// `normalize_penalty` uses: an exactly zero quadratic has no scale and reports
/// `1`.
fn constructive_frobenius_scale(quadratic: &ConstructiveQuadratic) -> f64 {
    let norm = stable_euclidean_norm(quadratic.dense().iter().copied());
    if norm == 0.0 { 1.0 } else { norm }
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
        rrqr_nullspace_basis(&representer_rows.t().to_owned())
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
    Ok(match representer_section_chart(k_cc, z_rbf)?.transform {
        Some(transform) => z_rbf.dot(&transform),
        None => z_rbf.clone(),
    })
}

/// The chart [`condition_representer_section`] realizes, with the decomposition
/// it is read off. [`representer_section_log_length_jets`] differentiates the
/// same decision, so both read it here.
struct RepresenterSectionChart {
    /// `z_rbf` coordinates to realized columns, `q × r`. `None` when the section
    /// passes through unchanged.
    transform: Option<Array2<f64>>,
    /// Every singular value of `E = K_cc·z_rbf`.
    singular: Array1<f64>,
    /// The right singular vectors of `E` as COLUMNS, `q × q`.
    right: Array2<f64>,
    /// The directions the retention bar keeps, in the transform's column order.
    kept: Vec<usize>,
    /// `sign/max(σ, floor)` for each kept column.
    column_scales: Vec<f64>,
}

fn representer_section_chart(
    k_cc: &Array2<f64>,
    z_rbf: &Array2<f64>,
) -> Result<RepresenterSectionChart, BasisError> {
    let width = z_rbf.ncols();
    let pass_through = |singular: Array1<f64>| RepresenterSectionChart {
        transform: None,
        singular,
        right: Array2::<f64>::eye(width),
        kept: (0..width).collect(),
        column_scales: vec![1.0; width],
    };
    if width == 0 {
        return Ok(pass_through(Array1::<f64>::zeros(0)));
    }
    let evaluation = k_cc.dot(z_rbf);
    let (_, singular, right) = evaluation.svd(false, true).map_err(BasisError::LinalgError)?;
    let leading = singular.iter().copied().fold(0.0_f64, f64::max);
    if !(leading.is_finite() && leading > 0.0) {
        return Ok(pass_through(singular));
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
    //
    // The bar is dimensionless — it scores `(σ_i/floor)²`, a ratio — so it is
    // taken at a unit spectrum, which is the convention's relative width at
    // this chart's dimension. Restating the constant's arithmetic here instead
    // would leave this decision behind when the rank cutoff moves, which is the
    // one thing the constant's own doc forbids (gam#4057).
    let rank_tolerance = super::bspline_build::spectral_tolerance_for_dim(
        z_rbf.ncols().max(1),
        &Array1::from_elem(1, 1.0),
    );
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
    let mut column_scales = Vec::with_capacity(kept.len());
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
        column_scales.push(sign * inverse);
    }
    Ok(RepresenterSectionChart {
        transform: Some(transform),
        singular,
        right: right.t().to_owned(),
        kept,
        column_scales,
    })
}

/// `(W·A, null(AᵀW·K_cc))`: the mass-weighted affine value basis at the centers
/// and the orthonormal representer coefficient section it leaves, before
/// conditioning.
///
/// `head_cc` IS the affine value basis `A` at the centers, by construction (both
/// come from `measure_jet_affine_head_lift`), so the gauge constrains the
/// representers against exactly the span the head carries.
fn representer_null_section(
    k_cc: &Array2<f64>,
    head_cc: &Array2<f64>,
    masses: ArrayView1<'_, f64>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let mut weighted_affine = head_cc.clone();
    for (i, mut row) in weighted_affine.outer_iter_mut().enumerate() {
        row.mapv_inplace(|v| v * masses[i]);
    }
    // `rrqr_nullspace_basis(B)` returns null(B^T). Here `B = K_cc^T W A = C^T`,
    // hence the returned columns span null(C), exactly the required RBF
    // coefficient section.
    let constraint_cross = k_cc.t().dot(&weighted_affine);
    let (section, _) = rrqr_nullspace_basis(&constraint_cross)
        .map_err(BasisError::LinalgError)?;
    Ok((weighted_affine, section))
}

/// First and second `u = ln ℓ` derivatives `(∂Z, ∂²Z)` of the representer section
/// a `CenterSumToZero` build realizes, on its representer rows and columns
/// (`m × r`).
///
/// A frozen transform replays one coefficient chart, so its jets hold `Z` fixed.
/// A `CenterSumToZero` build realizes the section at every range,
/// `Z(ℓ) = N(ℓ)·V(ℓ)·D(ℓ)`, and the design and penalty jets have to carry that
/// motion or they differentiate a function the builder never evaluates. On the
/// range-screen fixture the frozen-`Z` jet put `V′` 1.5e-3 relative off central
/// differences of the rebuilt criterion, while freezing the gauge on both sides
/// agreed to 5e-7 (#2902 row 5). Two factors move:
///
/// * `N(ℓ)`, an orthonormal basis of `null(C(ℓ))`, `C = AᵀW K_cc(ℓ)`, of constant
///   rank. Along `N(u) = P(u)N₀(N₀ᵀP(u)N₀)^{-1/2}` with `P = I − C⁺C` and
///   `CN₀ = 0`: `N′ = −C⁺C′N₀` and `N″ = −2(C⁺)′C′N₀ − C⁺C″N₀ + N₀N′ᵀN′`, where
///   `(C⁺)′ = −C⁺C′C⁺ + C⁺C⁺ᵀC′ᵀ(I − CC⁺) + (I − C⁺C)C′ᵀC⁺ᵀC⁺`.
/// * `V(ℓ)`, the right singular directions of `E = K_cc·N` the retention bar
///   keeps. Their span is the dominant invariant subspace of `F = EᵀE`. In the
///   graph chart `V_K(u) = V_K + V_D·X(u)` of that subspace, the invariance
///   equation gives, in `E`'s singular basis with `λ = σ²`,
///   `X′_jk·(λ_k − λ_j) = F′_jk` and
///   `X″_jk·(λ_k − λ_j) = F″_jk + 2(F′_DD·X′ − X′·F′_KK)_jk`.
///
/// `D`, the conditioning's damping, and every other chart freedom right-multiply
/// the section, which the profiled criterion does not see (`X → XT`,
/// `S → TᵀST`). So on any stretch of ranges where the constraint rank and the
/// retained count hold, this curve's jets are the realized criterion's jets.
/// Where either count changes the realized design jumps, and no derivative
/// describes a jump.
fn representer_section_log_length_jets(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    head_lift: ArrayView2<'_, f64>,
    kernel: [&Array2<f64>; 3],
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let [k_cc, dk_cc, d2k_cc] = kernel;
    let m = centers.nrows();
    let head_cc = measure_jet_affine_head_block(centers, head_lift);
    let (weighted_affine, section) = representer_null_section(k_cc, &head_cc, masses)?;
    let width = section.ncols();
    let constraint = weighted_affine.t().dot(k_cc);
    let constraint_first = weighted_affine.t().dot(dk_cc);
    let constraint_second = weighted_affine.t().dot(d2k_cc);
    // The pseudo-inverse on the rank the section was cut at.
    let constraint_rank = m - width;
    let (left, singular, right) = constraint
        .svd(true, true)
        .map_err(BasisError::LinalgError)?;
    let (Some(left), Some(right)) = (left, right) else {
        crate::bail_invalid_basis!(
            "measure-jet representer constraint decomposition returned no singular vectors"
        );
    };
    let mut order: Vec<usize> = (0..singular.len()).collect();
    order.sort_by(|&a, &b| singular[b].total_cmp(&singular[a]));
    if constraint_rank > order.len() {
        crate::bail_invalid_basis!(
            "measure-jet representer section of width {width} implies constraint rank \
             {constraint_rank}, but the constraint has {} singular values",
            order.len()
        );
    }
    let retained = &order[..constraint_rank];
    let range_left = left.select(Axis(1), retained);
    let range_right = right.select(Axis(0), retained).t().to_owned();
    let mut inverse_singular = Array1::<f64>::zeros(constraint_rank);
    for (slot, &index) in retained.iter().enumerate() {
        let value = singular[index];
        if !(value.is_finite() && value > 0.0) {
            crate::bail_invalid_basis!(
                "measure-jet representer constraint retains a singular value {value:e}"
            );
        }
        inverse_singular[slot] = value.recip();
    }
    let pseudo_inverse =
        (&range_right * &inverse_singular.view().insert_axis(Axis(0))).dot(&range_left.t());
    let moved = constraint_first.dot(&section);
    let pulled = pseudo_inverse.dot(&moved);
    let off_range_left = &moved - &range_left.dot(&range_left.t().dot(&moved));
    let back = constraint_first.t().dot(&pseudo_inverse.t().dot(&pulled));
    let off_range_right = &back - &range_right.dot(&range_right.t().dot(&back));
    let pseudo_inverse_moved = -pseudo_inverse.dot(&constraint_first.dot(&pulled))
        + pseudo_inverse.dot(
            &pseudo_inverse
                .t()
                .dot(&constraint_first.t().dot(&off_range_left)),
        )
        + off_range_right;
    let null_first = -pulled;
    let null_second = pseudo_inverse_moved * -2.0
        - pseudo_inverse.dot(&constraint_second.dot(&section))
        + section.dot(&null_first.t().dot(&null_first));

    let chart = representer_section_chart(k_cc, &section)?;
    let kept = &chart.kept;
    let dropped: Vec<usize> = (0..width).filter(|index| !kept.contains(index)).collect();
    // `F = EᵀE` and its `u = ln ℓ` jets in `E`'s singular basis. These are needed
    // whether or not any direction was dropped: the graph chart reads their
    // OFF-diagonal kept/dropped block, and the column scales below read their
    // DIAGONAL, which moves at every range.
    let evaluation = k_cc.dot(&section);
    let evaluation_first = dk_cc.dot(&section) + k_cc.dot(&null_first);
    let evaluation_second =
        d2k_cc.dot(&section) + dk_cc.dot(&null_first) * 2.0 + k_cc.dot(&null_second);
    let symmetric_product = |left: &Array2<f64>, right: &Array2<f64>| -> Array2<f64> {
        let half = left.t().dot(right);
        &half + &half.t()
    };
    let basis = &chart.right;
    let gram_first = basis
        .t()
        .dot(&symmetric_product(&evaluation_first, &evaluation))
        .dot(basis);
    let gram_second = basis
        .t()
        .dot(
            &(symmetric_product(&evaluation_second, &evaluation)
                + evaluation_first.t().dot(&evaluation_first) * 2.0),
        )
        .dot(basis);
    let (rotation_first, rotation_second) = if dropped.is_empty() {
        (
            Array2::<f64>::zeros((width, kept.len())),
            Array2::<f64>::zeros((width, kept.len())),
        )
    } else {
        let mut gaps = Array2::<f64>::zeros((dropped.len(), kept.len()));
        let mut graph_first = Array2::<f64>::zeros((dropped.len(), kept.len()));
        for (row, &lower) in dropped.iter().enumerate() {
            for (column, &upper) in kept.iter().enumerate() {
                let gap = chart.singular[upper].powi(2) - chart.singular[lower].powi(2);
                if !(gap.is_finite() && gap > 0.0) {
                    crate::bail_invalid_basis!(
                        "measure-jet representer section has no spectral gap ({gap:e}) between a \
                         kept and a dropped direction, so its range jet is undefined there"
                    );
                }
                gaps[(row, column)] = gap;
                graph_first[(row, column)] = gram_first[(lower, upper)] / gap;
            }
        }
        let coupling = gram_first
            .select(Axis(0), &dropped)
            .select(Axis(1), &dropped)
            .dot(&graph_first)
            - graph_first.dot(&gram_first.select(Axis(0), kept).select(Axis(1), kept));
        let mut graph_second = Array2::<f64>::zeros((dropped.len(), kept.len()));
        for (row, &lower) in dropped.iter().enumerate() {
            for (column, &upper) in kept.iter().enumerate() {
                graph_second[(row, column)] = (gram_second[(lower, upper)]
                    + 2.0 * coupling[(row, column)])
                    / gaps[(row, column)];
            }
        }
        let dropped_basis = basis.select(Axis(1), &dropped);
        (
            dropped_basis.dot(&graph_first),
            dropped_basis.dot(&graph_second),
        )
    };
    let scales = Array1::from_vec(chart.column_scales.clone());
    let scale_columns =
        |block: Array2<f64>| -> Array2<f64> { &block * &scales.view().insert_axis(Axis(0)) };
    let transform = chart
        .transform
        .clone()
        .unwrap_or_else(|| Array2::<f64>::eye(width));
    // The chart the builder realizes is `T = V_K·S`, `S = diag(sign/σ_k)` from
    // `representer_section_chart`. `S` MOVES: `σ_k` is a singular value of
    // `E = K_cc·Z`, and that spectrum collapses as `ℓ` grows — the chart's own doc
    // tabulates `σ_max(E)` falling from 2.29e0 to 3.48e-6 over the range sweep. So
    // `T′ = V_K′·S + V_K·S′` and `T″ = V_K″·S + 2V_K′·S′ + V_K·S″`, and an
    // implementation carrying only the `V_K` motion differentiates a chart nobody
    // realizes.
    //
    // The consumer is what makes the omission wrong rather than free. The section
    // jet is added to the DESIGN jet as `∂(K·Z)`, and a column rescaling of `Z` is
    // a column rescaling of `X` — visible in `log|XᵀWX + λS|` at `2·Σ_k ∂s_k/s_k`
    // even though the profiled criterion is invariant to applying one chart to
    // `X` and `S` TOGETHER. Measured on the #2959 run at 7573811d1a, dropping it
    // put `V′` at −1314.42 against a central difference of +9.47 at
    // `ln ℓ = −2.6275`: 140× and sign-inverted, an additive −1.32e3 with the
    // factor two of a log-determinant on it (#2902 rows 5 and 37, #2959).
    //
    // `s_k = sign_k/σ_k` gives `s′_k/s_k = −σ′_k/σ_k` and
    // `s″_k/s_k = 2(σ′_k/σ_k)² − σ″_k/σ_k`, and the σ jets come from the diagonal
    // of `F = EᵀE` already formed above: `F′_kk = 2σ_kσ′_k` and
    // `F″_kk = 2σ′_k² + 2σ_kσ″_k`.
    let mut scale_log_first = Array1::<f64>::zeros(kept.len());
    let mut scale_log_second = Array1::<f64>::zeros(kept.len());
    for (column, &index) in kept.iter().enumerate() {
        let sigma = chart.singular[index];
        // `representer_section_chart` CLAMPS `1/σ_k` at `1/(anchor·√ε)` for a
        // damped direction, and that floor rides on `‖K_cc‖₂`, whose own `ln ℓ`
        // derivative this function is not given. Refusing is the honest answer:
        // the alternative is to report a scale as frozen when it is not, which is
        // the defect this term exists to repair.
        // `|s_k| = 1/max(σ_k, floor)`, so `|s_k|·σ_k` is exactly 1 on an
        // UNDAMPED column and strictly below it on a damped one. The first form
        // of this guard read `<= 1 + 8ε`, which every column satisfies and which
        // would therefore have let the damped case through silently — the same
        // shape of defect as the term it is guarding.
        let unclamped = sigma.is_finite()
            && sigma > 0.0
            && (scales[column].abs() * sigma - 1.0).abs() <= 8.0 * f64::EPSILON;
        if !unclamped {
            crate::bail_invalid_basis!(
                "measure-jet representer section column {column} is damped (σ = {sigma:e},                  scale = {:e}), so its chart scale moves with the amplification floor and this                  jet cannot report it",
                scales[column]
            );
        }
        let sigma_first = gram_first[(index, index)] / (2.0 * sigma);
        let sigma_second =
            (gram_second[(index, index)] - 2.0 * sigma_first * sigma_first) / (2.0 * sigma);
        let relative_first = sigma_first / sigma;
        scale_log_first[column] = -relative_first;
        scale_log_second[column] = 2.0 * relative_first * relative_first - sigma_second / sigma;
    }
    let kept_basis = basis.select(Axis(1), kept);
    let weight_columns = |block: &Array2<f64>, weights: &Array1<f64>| -> Array2<f64> {
        block * &weights.view().insert_axis(Axis(0))
    };
    // `T′ = V_K′·S + V_K·S′`, `T″ = V_K″·S + 2V_K′·S′ + V_K·S″`, with `S′` and `S″`
    // written as `S` times the relative motions above so `scale_columns` applies
    // the one chart `S` exactly once on every term.
    let rotation_second = scale_columns(
        rotation_second
            + weight_columns(&rotation_first, &scale_log_first) * 2.0
            + weight_columns(&kept_basis, &scale_log_second),
    );
    let rotation_first =
        scale_columns(rotation_first + weight_columns(&kept_basis, &scale_log_first));
    let first = null_first.dot(&transform) + section.dot(&rotation_first);
    let second = null_second.dot(&transform)
        + null_first.dot(&rotation_first) * 2.0
        + section.dot(&rotation_second);
    Ok((first, second))
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

/// The deterministic nodes the representer range `ℓ` is SCREENED from before
/// the outer ψ search refines it, in the STANDARDIZED frame the basis is
/// realized in (gam#2750).
///
/// ## Why a screen exists at all
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
/// is the "1-D fits select a too-large λ" this screen exists to end.
///
/// ## Why THESE nodes
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
/// The screen (`screen_measure_jet_range` in gam-models) starts one certified
/// outer search from each node, on the exact `ln ℓ` jet of its criterion over
/// [`measure_jet_ln_range_window`], and keeps the lowest certified criterion. A
/// node is where a search STARTS; no node's own value is ever the answer
/// (#2902).
///
/// ## How far the screen may reach (#2761)
///
/// Up to the range at which the
/// closest node pair stops being distinguishable in the chart's own arithmetic,
/// [`measure_jet_range_feasibility_ceiling`]. It is the SAME wall
/// [`measure_jet_ln_range_window`] gives the outer search, for the same reason:
/// a search may not be caged tighter than the model.
///
/// The screen used to stop at the node bounding-box diameter, on the
/// argument that at `ℓ` that long every pair of representers overlaps at
/// `≥ exp(−1/2)`, so "there is no distinct model past it". That argument is
/// measurably wrong: `measure_jet_ln_range_window` records that *"the profiled
/// criterion genuinely prefers a range AT or ABOVE the node diameter"* on three
/// fixtures. On a term whose `ℓ` dial is FROZEN — the BMS marginal/slope pair,
/// or any `learn_length_scale=false` — nothing searches past the screen, so a
/// screen stop tighter than the model becomes the model's wall.
#[derive(Clone, Debug)]
pub struct MeasureJetRangeBracket {
    /// The realized scale band, ascending: where the screen's searches start.
    pub nodes: Vec<f64>,
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
/// ONE definition, read through the outer search's window
/// ([`measure_jet_ln_range_window`]), which the response screen searches too, so
/// the fit and the screen cannot drift into disagreeing about where the model
/// ends (#2761).
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
    Ok(MeasureJetRangeBracket {
        nodes: band.eps,
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
/// It is **not** the node bounding-box
/// diagonal. That is where the response screen USED to stop walking, a stopping
/// rule for a search over nodes rather than a wall in the model: measured on
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

/// The `α` SEARCH WINDOW for the density-normalization exponent, measured off
/// the node cloud `spec` realizes on `data` (#2902).
///
/// ## What `α` can move
///
/// At scale `ε` the energy weights the local jet residual of outer center `i` by
/// `ε^{−η}·net_mass_i·q_i^{1−2α}`, with `q_i` the kernel mass around `i` and
/// `η = 2s + d(2 − 2α)`. In multiscale mode every scale is its own penalty
/// candidate, Frobenius-normalized, so every factor common to a scale cancels,
/// `ε^{−η}` included. What `α` moves is the RELATIVE weight of the outer centers
/// within a scale, `(q_i/q_j)^{1−2α}`, and at `α = ½` that is flat.
///
/// ## Where it stops moving anything distinct
///
/// The relative weights span `(q_max/q_min)^{|1−2α|}` at the scale with the widest
/// kernel-mass spread. Past `1/√ε`, the lightest center's block sits below
/// half-mantissa of the heaviest in the assembled form. That is the same bar the
/// #2812 resolvability interval puts on a smoothing parameter against the data
/// curvature. Pushing `α` further only removes centers the arithmetic already
/// could not see. So the window is `½ ± ln(1/√ε) / (2·max_ε ln(q_max/q_min))`: its
/// centre is the flat weighting and its width is measured off the cloud.
///
/// A cloud whose kernel masses are equal at every scale gives `α` nothing to
/// move, and the window collapses onto the spec's own `α`.
pub(crate) fn measure_jet_alpha_window(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<(f64, f64), BasisError> {
    let geom = realize_measure_jet_geometry(data, spec)?;
    let band = MeasureJetBand {
        eps: geom.eps_band.clone(),
        log_step: geom.log_step,
    };
    // `(min ln q, max ln q)` per scale, read off the SAME walk the energy is
    // assembled by, so the window and the form cannot disagree about which outer
    // centers exist or what their kernel masses are.
    let spread = std::sync::Mutex::new(vec![(f64::INFINITY, f64::NEG_INFINITY); band.eps.len()]);
    assemble_weighted_forms(
        geom.centers.view(),
        geom.masses.view(),
        &band,
        geom.order_s_eval,
        spec.alpha,
        1,
        &|scale_idx, _: f64, q: f64, _: f64, out: &mut [f64]| {
            out[0] = 0.0;
            let ln_q = q.ln();
            let mut table = spread
                .lock()
                .expect("kernel-mass spread table poisoned by a panic during assembly");
            let (low, high) = &mut table[scale_idx];
            *low = low.min(ln_q);
            *high = high.max(ln_q);
        },
    )?;
    let table = spread
        .into_inner()
        .expect("kernel-mass spread table poisoned by a panic during assembly");
    let widest = table
        .iter()
        .filter(|(low, high)| low.is_finite() && high.is_finite())
        .map(|(low, high)| high - low)
        .fold(0.0_f64, f64::max);
    if !(widest > 0.0) {
        return Ok((spec.alpha, spec.alpha));
    }
    let half_width = -0.5 * f64::EPSILON.ln() / (2.0 * widest);
    Ok((0.5 - half_width, 0.5 + half_width))
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
/// form, one weight `w`; the form accumulates `w·R` with
/// `R = CᵀWC − B·G⁺·Bᵀ/q`, `G⁺` the rank-revealing pseudo-inverse.
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
    n_forms: usize,
    weights: &F,
) -> Result<Vec<Array2<f64>>, BasisError>
where
    F: Fn(usize, f64, f64, f64, &mut [f64]) + Sync,
{
    let m = centers.nrows();
    let d = centers.ncols();
    if n_forms == 0 {
        crate::bail_invalid_basis!("measure-jet assembly needs at least one output form");
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
    if !alpha.is_finite() {
        crate::bail_invalid_basis!("measure-jet energy needs a finite alpha; got alpha={alpha}");
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
        let mut wbuf = vec![0.0_f64; n_forms];
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
            let g_pinv = symmetric_pseudoinverse(
                &g,
                local_affine_gram_assembly_band(&phi, &w, q),
                "local affine Gram",
            )?;
            let bm = b.dot(&g_pinv);
            let base = scale_weight * net_mass[i] * q.powf(1.0 - 2.0 * alpha);
            weights(scale_idx, eps, q, base, &mut wbuf);
            // Scatter-add wbuf[k]·R into each form k.
            for (a, &ja) in idx.iter().enumerate() {
                let bma = bm.row(a);
                for (c, &jc) in idx.iter().enumerate() {
                    let b_c = b.row(c);
                    let mut val_r = -w[a] * w[c] / q - bma.dot(&b_c) / q;
                    if a == c {
                        val_r += w[a];
                    }
                    for (k, out_k) in out.iter_mut().enumerate() {
                        out_k[(ja, jc)] += wbuf[k] * val_r;
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
) -> Result<Array2<f64>, BasisError> {
    let mut forms = assemble_weighted_forms(
        centers,
        masses,
        band,
        order_s,
        alpha,
        1,
        &|_, _, _, base, out: &mut [f64]| out[0] = base,
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
pub(crate) fn measure_jet_energy_forms_per_scale(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
) -> Result<Vec<Array2<f64>>, BasisError> {
    let n_scales = band.eps.len();
    let forms = assemble_weighted_forms(
        centers,
        masses,
        band,
        order_s,
        alpha,
        n_scales,
        &|scale_idx, _, _, base, out: &mut [f64]| {
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = if k == scale_idx { base } else { 0.0 };
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
    // also keeps every per-scale detail energy `vᵀQ_ℓv` non-negative.
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
pub(crate) fn measure_jet_design_matrix(
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
/// each direction rather than its offset; a residual is dropped when it lies
/// inside the rounding band of the centering and projection arithmetic that
/// produced it (derived in the body). The returned
/// `T` satisfies `linear_head(points) = points · T` (the mean-centering only
/// informs the keep/drop decision). `T` is a deterministic function of the
/// frozen centers + masses, so the frozen replay path reconstructs the
/// identical head with no persisted state.
///
/// This is the LINEAR half of the null space. The realized head block is the
/// whole affine null space `[1 | points·T]`; build it through
/// [`measure_jet_affine_head_lift`] + `measure_jet_affine_head_block`, which
/// is what the design, the gauge and the null-component penalty all use. A
/// linear-only head is a defect, not an economy: the global parametric
/// orthogonalization removes ONE design direction, and if the term's null space
/// has no constant to give up, the direction it takes comes out of the null
/// space itself, leaving `d − 1` free linear directions instead of `d` (#2751).
pub(crate) fn measure_jet_affine_head_transform(
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
    let ones = Array1::ones(m);
    let (cols, raw_norms): (Vec<Array1<f64>>, Vec<f64>) = (0..d)
        .map(|k| {
            let col = centers.column(k).to_owned();
            let mean = if total_mass > 0.0 {
                mdot(&col, &ones) / total_mass
            } else {
                0.0
            };
            let raw_norm = mdot(&col, &col).sqrt();
            (col.mapv(|x| x - mean), raw_norm)
        })
        .unzip();
    // Mass-weighted modified Gram–Schmidt on the centered columns; `t`
    // accumulates the lift in the ORIGINAL coordinate basis, so every kept head
    // column is `points · t_r` (up to the intercept-owned constant).
    let mut q_cols: Vec<Array1<f64>> = Vec::new();
    let mut t_cols: Vec<Array1<f64>> = Vec::new();
    for k in 0..d {
        // A residual is kept only when it clears the rounding that produced it.
        // Centering: the mean's mass sum rounds `m + 1` deep, the total mass
        // `m − 1` and the quotient once, so `|δmean| ≤ γ_{2m+1}·Σw|x|/W`, and in
        // the mass norm `|δmean|·√W ≤ γ_{2m+1}·‖x‖_W` (Cauchy–Schwarz) — the
        // UNcentered norm, which is what makes a far-offset stratum's spurious
        // spread visible as rounding; each subtraction adds `u·‖c‖_W`. The mass
        // inner products of the single projection pass are `m + 1` deep. The
        // mass-metric projector is a contraction, so the centering error reaches
        // the residual undiminished and the two bands add.
        let centered_norm = mdot(&cols[k], &cols[k]).sqrt();
        let drop_below =
            gam_math::roundoff::gram_schmidt_residual_band(1, q_cols.len(), m + 1, centered_norm)
                + gam_math::roundoff::UNIT_ROUNDOFF * centered_norm
                + gam_math::roundoff::accumulation_growth(2 * m + 1) * raw_norms[k];
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
/// ambient-linear directions of `measure_jet_affine_head_transform`.
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
pub(crate) fn measure_jet_affine_head_block(
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
                representer_null_section(&k_cc, &head_cc, masses.view())?.1
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
/// knob), the smallest eigenvalue of the cell-local scatter `S_i = Σ (x−x̄)(x−x̄)ᵀ`
/// carries the perpendicular variance `σ_coord²`, and the cells are pooled over
/// their residual degrees of freedom:
///
///   σ̂² = Σ_i λ_min(S_i) / Σ_i (n_i − d).
///
/// Along the true normal `n`, `nᵀS_i n ~ σ²χ²_{n_i−1}` (the cell mean costs one
/// degree of freedom). The PCA minimum then rotates `n` toward each of the
/// `d − 1` tangent directions `t` and removes `(tᵀS_i n)²/(tᵀS_i t − nᵀS_i n) ~ σ²χ²₁`
/// per direction to first order in σ over the tangent spread, so
/// `E[λ_min(S_i)] = σ²(n_i − d)` on a codimension-1 stratum. Pooling over `n_i`
/// instead would shrink σ̂² by `Σ(n_i − d)/Σn_i`, which is `1/(d+1)` for the
/// smallest admitted cell (#3741). On a stratum of codimension `c > 1`, the
/// minimum over the `c` noise eigenvalues is an extreme order statistic and
/// stays biased low. No response values, no smoothing dial, and
/// no magic constant enter — it is a pure function of the ambient point cloud
/// and the frozen centers, in the centers' (standardized) coordinate frame.
///
/// Returns `None` when no cell can span a tangent (e.g. `d`-dimensional data
/// with fewer than `d + 1` points per cell, or a full-dimensional stratum with
/// no separable perpendicular direction) — the caller then leaves `Var_input`
/// disabled rather than invent a scale.
pub(crate) fn measure_jet_input_noise_scale(
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
    // Σ_i λ_min(S_i) and its residual degrees of freedom Σ_i (n_i − d).
    let mut scatter_minimum_sum = 0.0_f64;
    let mut residual_dof = 0.0_f64;
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
            // `smallest` is λ_min(S_i / n_i); `n_i ≥ d + 1` keeps the dof positive.
            scatter_minimum_sum += n_i as f64 * smallest;
            residual_dof += (n_i - d) as f64;
        }
    }
    if residual_dof <= 0.0 {
        return Ok(None);
    }
    let sigma2 = scatter_minimum_sum / residual_dof;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Ok(None);
    }
    Ok(Some(sigma2.sqrt()))
}

/// Whether a measure-jet spec runs in multiscale mode (per-scale spectral
/// energies + the `α` ψ dial). The separate `double_penalty`
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
        joint_null_rotation: None,
    })
}

/// Exact ψ-jets of the REALIZED measure-jet penalty candidates, adapted to
/// the anisotropic group-ψ carrier the spatial optimizer consumes.
///
/// Coordinates (the layout contract for the registration arm):
/// - per-level (spectral) mode: `[ln ℓ?, α]` — order is absorbed by the
///   REML-learned scale amplitudes;
/// - single-scale mode: `[ln ℓ?]`, because its energy dials are fixed.
///
/// Only `ln ℓ` moves the design. It also moves every coefficient-space penalty
/// pullback through the center evaluation map `E(ℓ)`; `α` moves only the
/// per-scale center-value forms. Exact diagonal and mixed product-rule jets are
/// emitted before Frobenius normalization.
/// Penalty derivatives are routed through the SAME constrained Frobenius
/// normalization as the fit-time candidates
/// (`normalize_penaltywith_psi_derivatives` + the cross rule), so criterion
/// value and criterion derivative share one normalization — the #901 lesson
/// made structural. The function-space null candidate has nonzero `ln ℓ` jets
/// and zero `α` jets. The per-candidate layout follows the builder's
/// ORIGINAL order (scale candidates or Primary, then null component); consumers
/// align to the FITTED penalty list via
/// `ActivePenaltyInfo.original_index` when the candidate filter dropped
/// any.
pub fn build_measure_jet_basis_psi_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &MeasureJetBasisSpec,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
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
    // so its raw derivative columns are exactly zero before applying the Gauge
    // section. A frozen transform is the replay contract: rank/gauge realization
    // happens once at fit time, then every ψ trial differentiates the same
    // coefficient chart. A `CenterSumToZero` build realizes the section at every
    // ℓ, so its jets carry the section's own motion as well (#2902 row 5).
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

        let mut jets = LengthScaleJets {
            evaluation_first: geom.coefficient_gauge.restrict_design(&dk_centers_aug),
            evaluation_second: geom.coefficient_gauge.restrict_design(&d2k_centers_aug),
            design_first: geom.coefficient_gauge.restrict_design(&dk_data_aug),
            design_second: geom.coefficient_gauge.restrict_design(&d2k_data_aug),
        };
        if matches!(
            spec.identifiability,
            MeasureJetIdentifiability::CenterSumToZero
        ) && geom.head_lift.ncols() > 0
        {
            let kernel_centers = measure_jet_design_matrix(
                geom.centers.view(),
                geom.centers.view(),
                geom.length_scale,
            )?;
            let (section_first, section_second) = representer_section_log_length_jets(
                geom.centers.view(),
                geom.masses.view(),
                geom.head_lift.view(),
                [&kernel_centers, &dk_centers, &d2k_centers],
            )?;
            let representer_columns = p - geom.head_lift.ncols();
            if section_first.dim() != (m, representer_columns) {
                crate::bail_dim_basis!(
                    "measure-jet representer section motion is {:?}, expected ({m}, {representer_columns})",
                    section_first.dim()
                );
            }
            let kernel_data =
                measure_jet_design_matrix(data, geom.centers.view(), geom.length_scale)?;
            // `∂(KZ) = ∂K·Z + K·∂Z` and `∂²(KZ) = ∂²K·Z + 2∂K·∂Z + K·∂²Z` on the
            // representer columns. The first term of each is already in place;
            // the head columns do not move.
            {
                let mut block = jets
                    .design_first
                    .slice_mut(ndarray::s![.., ..representer_columns]);
                block += &kernel_data.dot(&section_first);
            }
            {
                let mut block = jets
                    .design_second
                    .slice_mut(ndarray::s![.., ..representer_columns]);
                block += &(dk_data.dot(&section_first) * 2.0 + kernel_data.dot(&section_second));
            }
            {
                let mut block = jets
                    .evaluation_first
                    .slice_mut(ndarray::s![.., ..representer_columns]);
                block += &kernel_centers.dot(&section_first);
            }
            {
                let mut block = jets
                    .evaluation_second
                    .slice_mut(ndarray::s![.., ..representer_columns]);
                block +=
                    &(dk_centers.dot(&section_first) * 2.0 + kernel_centers.dot(&section_second));
            }
        }
        Some(jets)
    } else {
        None
    };

    let coord_offset = usize::from(length_scale_jets.is_some());
    let n_coords = coord_offset + usize::from(geom.per_level);
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
    // Coordinate order is `[lnℓ?, α]` in multiscale mode and `[lnℓ?]`
    // in single-scale mode. Candidate order exactly mirrors the value builder:
    // scale candidates or Primary first, then the optional null-component
    // candidate. Active filtering aligns through `ActivePenaltyInfo::original_index`.
    // The single-scale Primary, when there is one. `None` in per-level mode,
    // which emits scale candidates instead and therefore never rebuilds the
    // null component.
    let mut single_scale_primary: Option<ConstructiveQuadratic> = None;
    let mut raw: Vec<RawPenaltyJets> = if geom.per_level {
        let l_count = band.eps.len();
        // Three forms per scale: value, ∂α, ∂α² — same blocks, one walk
        // (single-source rule).
        let forms = assemble_weighted_forms(
            geom.centers.view(),
            geom.masses.view(),
            &band,
            geom.order_s_eval,
            spec.alpha,
            3 * l_count,
            &|scale_idx, eps: f64, q: f64, base: f64, out: &mut [f64]| {
                for slot in out.iter_mut() {
                    *slot = 0.0;
                }
                let intrinsic_dim = geom.centers.ncols() as f64;
                let ga = 2.0 * intrinsic_dim * eps.ln() - 2.0 * q.max(f64::MIN_POSITIVE).ln();
                let k0 = 3 * scale_idx;
                out[k0] = base;
                out[k0 + 1] = ga * base;
                out[k0 + 2] = ga * ga * base;
            },
        )?;
        let alpha_coord = coord_offset;
        let mut raw = Vec::with_capacity(l_count + usize::from(spec.double_penalty));
        for level in 0..l_count {
            let chunk = &forms[3 * level..3 * level + 3];
            let mut first: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
            let mut second_diag: Vec<Array2<f64>> = (0..n_coords).map(|_| zero_p()).collect();
            first[alpha_coord] = sandwich(&chunk[1]);
            second_diag[alpha_coord] = sandwich(&chunk[2]);
            if coord_offset == 1 {
                let (ell_first, ell_second) = length_diag(&chunk[0]);
                first[0] = ell_first;
                second_diag[0] = ell_second;
            }
            let mut cross: Vec<Array2<f64>> = (0..pairs.len()).map(|_| zero_p()).collect();
            for (pair_idx, &(a, b)) in pairs.iter().enumerate() {
                cross[pair_idx] = if coord_offset == 1 && a == 0 && b == alpha_coord {
                    length_cross(&chunk[1])
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
        // Single-scale mode enrolls no `(s, α)` penalty dials. It still
        // emits the pure Primary and, when requested, a separate REML null
        // component; an opt-in `lnℓ` coordinate differentiates both pullbacks.
        let q_form = measure_jet_energy_form(
            geom.centers.view(),
            geom.masses.view(),
            &band,
            geom.order_s_eval,
            spec.alpha,
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
            // H₀ is independent of α; its only moving object is E(ℓ),
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
        // and its degenerate convention is mirrored here: ‖S‖_F = 0 (or
        // non-finite) reports scale 1.0 — the value passes through unscaled,
        // and the cross helper receives that same 1.0, never a collapsed
        // near-zero scale.
        let fro = trace_of_product(s_raw, s_raw).sqrt();
        let c = if fro.is_finite() && fro > 0.0 {
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

    /// Small cells: the pooled local-PCA minimum must be charged its residual
    /// degrees of freedom `n_i − d`, not `n_i` (#3741). This test puts a line in 2-D
    /// with 200 centers over 800 points, so about 4 points fall in each cell.
    /// Pooling over `n_i` returns about `σ·√(400/800) ≈ 0.71σ`. Pooling over `n_i − d`
    /// has an unbiased σ̂², and its sampling spread is χ²₄₀₀/400, a relative sd of
    /// `√(2/400) ≈ 7%` on σ̂² and about 3.5% on σ̂. The 12% band is about 3.4 sd and
    /// still excludes the n_i-pooled value.
    #[test]
    pub(crate) fn input_noise_scale_is_unbiased_on_small_cells() {
        let tang = [1.0 / 5f64.sqrt(), 2.0 / 5f64.sqrt()];
        let perp = [2.0 / 5f64.sqrt(), -1.0 / 5f64.sqrt()];
        // Noise far below the within-cell tangent spread (~4e-3), so the
        // first-order dof count is the operative one.
        let sigma = 1.0e-4_f64;
        let n = 800usize;
        let mut state = 0x0fed_cba9_8765_4321u64;
        let mut data = Array2::<f64>::zeros((n, 2));
        for j in 0..n {
            let t = 3.0 * (j as f64) / (n as f64 - 1.0);
            let noise = sigma * lcg_normal(&mut state);
            for a in 0..2 {
                data[(j, a)] = t * tang[a] + noise * perp[a];
            }
        }
        let n_centers = 200usize;
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
        assert!(
            (est / sigma - 1.0).abs() <= 0.12,
            "small-cell σ_coord {est} is not unbiased for {sigma} (ratio {:.3}; n_i pooling gives ≈ 0.71)",
            est / sigma
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
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0)
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

    /// The local weighted affine projection annihilates ambient affine
    /// functions exactly: no affine direction pays energy.
    #[test]
    pub(crate) fn energy_form_annihilates_affine_exactly() {
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
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0)
            .expect("energy form");
        let e_affine = affine.dot(&q.dot(&affine));
        let e_rough = rough.dot(&q.dot(&rough));
        assert!(e_rough > 0.0, "rough vector must pay energy");
        assert!(
            e_affine.abs() <= 1e-12 * e_rough,
            "default affine energy {e_affine:.3e} vs rough {e_rough:.3e}"
        );
    }

    /// Frozen specs written before the τ field was deleted carry `"tau0"`.
    /// The spec does not deny unknown fields, so they still load and the
    /// stale key is dropped.
    #[test]
    pub(crate) fn frozen_spec_with_removed_tau0_still_deserializes() {
        let current = serde_json::to_value(MeasureJetBasisSpec::default()).expect("serialize");
        let mut legacy = current.clone();
        legacy
            .as_object_mut()
            .expect("the spec serializes as an object")
            .insert("tau0".to_string(), serde_json::json!(1e-3));
        let loaded: MeasureJetBasisSpec =
            serde_json::from_value(legacy).expect("a spec carrying tau0 loads");
        assert_eq!(serde_json::to_value(&loaded).expect("serialize"), current);
    }

    /// PSD: the energy is a sum of weighted least-squares residuals.
    #[test]
    pub(crate) fn energy_form_is_psd() {
        let (centers, masses) = two_cluster_centers();
        let band = band_for(&centers);
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0)
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
        let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0)
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
    /// candidates under frozen geometry — per-level mode (coordinate α).
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
            1,
            "the per-level coordinate is α alone"
        );
        assert_eq!(derivs.penalties_first[0].len(), l_count + 1);
        assert!(
            derivs.penalties_cross_pairs.is_empty(),
            "a single penalty dial has no cross pair"
        );
        let pen_at = |alpha: f64| {
            let trial = MeasureJetBasisSpec {
                alpha,
                ..frozen.clone()
            };
            build_measure_jet_basis(data.view(), &trial)
                .expect("trial build")
                .active_penalties
                .into_iter()
                .map(|penalty| penalty.matrix)
                .collect::<Vec<_>>()
        };
        let h = 1e-4;
        let a0 = frozen.alpha;
        let ap = pen_at(a0 + h);
        let am = pen_at(a0 - h);
        assert_eq!(
            ap.len(),
            l_count + 1,
            "fixture must keep every scale active"
        );
        for level in 0..l_count {
            let fd = (&ap[level] - &am[level]) / (2.0 * h);
            let analytic = &derivs.penalties_first[0][level];
            let scale = fd.iter().fold(1e-30_f64, |acc, v| acc.max(v.abs()));
            for (x, y) in analytic.iter().zip(fd.iter()) {
                assert!(
                    (x - y).abs() <= 5e-5 * scale,
                    "alpha jet of scale-candidate {level}: analytic {x:.6e} vs FD {y:.6e}"
                );
            }
        }
        // The function-space null candidate is independent of α.
        assert!(
            derivs.penalties_first[0][l_count]
                .iter()
                .all(|v| *v == 0.0),
            "null-component candidate must have zero α drift"
        );
    }

    /// #2902: the α window is measured off the node cloud. It is centred on the
    /// flat weighting `α = ½`, and an isotropic rescale of the data, which leaves
    /// every kernel mass unchanged, leaves the window unchanged.
    #[test]
    fn alpha_window_is_measured_off_the_node_cloud_2902() {
        let chart = |scale: f64| {
            Array2::from_shape_fn((240, 1), |(i, _)| {
                let t = i as f64 / 239.0;
                scale * (t * t + 0.04 * (7.0 * t).sin())
            })
        };
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 40 },
            multiscale: true,
            ..MeasureJetBasisSpec::default()
        };
        let (lower, upper) =
            measure_jet_alpha_window(chart(1.0).view(), &spec).expect("alpha window");
        assert!(
            lower < 0.5 && 0.5 < upper,
            "a density-varying cloud must leave α a window around the flat weighting: [{lower}, {upper}]"
        );
        assert!(
            ((0.5 - lower) - (upper - 0.5)).abs() <= 1e-12,
            "the window must be symmetric about α = ½: [{lower}, {upper}]"
        );
        let (scaled_lower, scaled_upper) =
            measure_jet_alpha_window(chart(10.0).view(), &spec).expect("rescaled alpha window");
        assert!(
            (scaled_lower - lower).abs() <= 1e-9 * (1.0 + lower.abs())
                && (scaled_upper - upper).abs() <= 1e-9 * (1.0 + upper.abs()),
            "an isotropic rescale moved the α window: [{lower}, {upper}] -> [{scaled_lower}, {scaled_upper}]"
        );
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

    /// The `CenterSumToZero` arm's `ln ℓ` design jet against central differences
    /// of the rebuilt design (#2902 rows 5 and 37, #2959).
    ///
    /// # Why this exists beside `psi_producer_matches_fd_length_scale`
    ///
    /// That test is the file's only finite-difference gate on the ℓ derivative,
    /// and it CANNOT reach this code. It builds from `frozen_spec_fixture`, whose
    /// spec carries `MeasureJetIdentifiability::FrozenTransform`, while
    /// `representer_section_log_length_jets` is gated on
    /// `MeasureJetIdentifiability::CenterSumToZero`. A frozen transform replays
    /// one coefficient chart and holds `Z` fixed; the default arm realizes `Z` at
    /// every range and has to carry its motion. The two arms are complementary,
    /// so a 302-line derivative path landed on the default arm — inside an archive
    /// of a dirty tree, `bf3398e863` — and the suite stayed green while the
    /// realized `V′` sat at −1314.42 against a central difference of +9.47.
    ///
    /// # The bar
    ///
    /// Two steps, `h` and `2h`. A central difference carries `g′ + C·h² + O(h⁴)`,
    /// so the gap between the two differences IS `3C·h²` and the truncation of the
    /// `h` difference is a third of a quantity this test MEASURES rather than
    /// names. The bar is that whole gap — three times the truncation, the factor
    /// being the `(2h)²/h²` scaling of the differencing error itself — plus the
    /// rounding the differencing amplifies by `1/(2h)`. This is the same rule the
    /// landed range-screen pin uses (`a606cd1754`), so both read one bar.
    #[test]
    fn center_sum_to_zero_design_jet_matches_central_differences_2902() {
        let n = 96usize;
        let data = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| {
            let t = i as f64 / (n as f64 - 1.0);
            t + 0.04 * (7.0 * t).sin()
        });
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
            learn_length_scale: true,
            ..MeasureJetBasisSpec::default()
        };
        assert!(
            matches!(
                spec.identifiability,
                MeasureJetIdentifiability::CenterSumToZero
            ),
            "this test exists to cover the CenterSumToZero arm; the default moved"
        );

        let derivs = build_measure_jet_basis_psi_derivatives(data.view(), &spec)
            .expect("psi derivatives on the CenterSumToZero arm");
        assert_eq!(
            derivs.design_first.len(),
            1,
            "learn_length_scale enrolls exactly the ln ℓ coordinate"
        );

        let ell0 = spec.length_scale;
        let build_at = |ell: f64| {
            let trial = MeasureJetBasisSpec {
                length_scale: ell,
                ..spec.clone()
            };
            build_measure_jet_basis(data.view(), &trial)
                .expect("trial build")
                .design
                .to_dense()
        };
        let h: f64 = 1e-4;
        let near = (build_at(ell0 * h.exp()) - build_at(ell0 * (-h).exp())) / (2.0 * h);
        let far =
            (build_at(ell0 * (2.0 * h).exp()) - build_at(ell0 * (-2.0 * h).exp())) / (4.0 * h);
        let plus = build_at(ell0 * h.exp());
        let minus = build_at(ell0 * (-h).exp());

        let analytic = &derivs.design_first[0];
        assert_eq!(
            analytic.dim(),
            near.dim(),
            "the analytic jet and the rebuilt difference must be the same block"
        );
        let rows = data.nrows();
        let columns = analytic.ncols();
        let mut worst = 0.0_f64;
        let mut worst_report = String::new();
        for ((index, value), difference) in analytic.indexed_iter().zip(near.iter()) {
            let truncation = (far[index] - difference).abs();
            let rounding = gam_linalg::roundoff::accumulation_growth(rows + columns)
                * (plus[index].abs() + minus[index].abs())
                / (2.0 * h);
            let band = truncation + rounding;
            let miss = (value - difference).abs();
            if miss > band && miss - band > worst {
                worst = miss - band;
                worst_report = format!(
                    "entry {index:?}: analytic {value:.9e} vs central difference                      {difference:.9e} (at 2h: {:.9e}), band {band:.3e}",
                    far[index]
                );
            }
        }
        assert!(
            worst_report.is_empty(),
            "the CenterSumToZero design jet left the band its own differences measure by \
             {worst:.3e}. Worst entry — {worst_report}"
        );
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

    /// #2469: centers on a tilted line far from the origin. Storing
    /// `y = 0.3·x + 10⁶` rounds each `y` by up to `u·10⁶`, so the centered `y`
    /// column keeps a residual of about `2.4e-9` after projecting out `x`. That
    /// is rounding, not spread: the replaced floor `64·ε·d·max‖c‖` (≈ 4e-14
    /// here) read it as a second linear direction and normalized it into a head
    /// column of size ~4e8. The derived band, which carries the centering's
    /// rounding at the uncentered scale, drops it.
    #[test]
    fn affine_head_drops_the_rounding_residual_of_a_far_offset_line_2469() {
        let m = 24usize;
        let mut centers = Array2::<f64>::zeros((m, 2));
        for i in 0..m {
            let x = i as f64 / (m - 1) as f64;
            centers[(i, 0)] = x;
            centers[(i, 1)] = 0.3 * x + 1.0e6;
        }
        let masses = Array1::<f64>::ones(m);
        let lift = measure_jet_affine_head_transform(centers.view(), masses.view());
        assert_eq!(
            lift.ncols(),
            1,
            "a line carries one linear direction, got {}",
            lift.ncols()
        );

        // Negative control: the same offset with a perpendicular spread of
        // 1e-4 — far above the `γ_{2m+1}·‖y‖_W` centering band (~3e-8) — is a
        // genuine plane and keeps both directions.
        let mut plane = centers.clone();
        for i in 0..m {
            plane[(i, 1)] += if i % 2 == 0 { 1.0e-4 } else { -1.0e-4 };
        }
        let plane_lift = measure_jet_affine_head_transform(plane.view(), masses.view());
        assert_eq!(
            plane_lift.ncols(),
            2,
            "a resolved plane keeps both linear directions"
        );
    }

    /// #2469: the local affine Gram is formed by the uncentered formula
    /// `(ΦᵀWΦ)/q − a·aᵀ`, which cancels when the neighbors sit far from the
    /// base point relative to their spread. On a cluster at `φ ≈ 2.9` spread
    /// along a line, the computed normal eigenvalue is cancellation noise
    /// (~1e-15) while the exact one is zero. The derived band covers the
    /// formation error (checked against the centered two-pass Gram) and drops
    /// the noise; the replaced floor `64·ε·d·λ_max` sat ~9 decades below that
    /// error and inverted the noise whenever it rounded positive.
    #[test]
    fn local_affine_gram_band_covers_uncentered_cancellation_2469() {
        let ml = 40usize;
        let mut phi = Array2::<f64>::zeros((ml, 2));
        let mut w = Array1::<f64>::zeros(ml);
        for a in 0..ml {
            let t = (1.7 * a as f64 + 0.3).sin();
            phi[(a, 0)] = 2.9 + 1.0e-5 * t;
            phi[(a, 1)] = 2.9 + 0.3e-5 * t;
            w[a] = (-0.5 * (1.0 + 0.01 * (a as f64).cos())).exp();
        }
        let q = w.sum();
        // The production formation, operation for operation.
        let a_mean = phi.t().dot(&w) / q;
        let mut wphi = phi.clone();
        for (a, mut row) in wphi.outer_iter_mut().enumerate() {
            row.mapv_inplace(|v| v * w[a]);
        }
        let mut g = phi.t().dot(&wphi);
        g.mapv_inplace(|v| v / q);
        for r in 0..2 {
            for c in 0..2 {
                g[(r, c)] -= a_mean[r] * a_mean[c];
            }
        }
        // Centered two-pass reference: no cancellation, error at the scale of
        // G itself.
        let mut centered = phi.clone();
        for mut row in centered.outer_iter_mut() {
            row -= &a_mean;
        }
        let mut wc = centered.clone();
        for (a, mut row) in wc.outer_iter_mut().enumerate() {
            row.mapv_inplace(|v| v * w[a]);
        }
        let g_ref = centered.t().dot(&wc) / q;
        let band = local_affine_gram_assembly_band(&phi, &w, q);
        let formation_error = (&g - &g_ref).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            formation_error <= band,
            "formation error {formation_error:e} escapes the assembly band {band:e}"
        );

        let pinv = symmetric_pseudoinverse(&g, band, "test local Gram").expect("pinv");
        let rank = pinv.dot(&g).diag().sum();
        assert!(
            (rank - 1.0).abs() < 0.5,
            "the cluster is 1-D: the pseudo-inverse must resolve one direction, got trace {rank}"
        );

        // Negative control: the replaced floor sits below the formation error,
        // so which side of it the noise eigenvalue lands on is the sign of a
        // rounding error (a positive one was inverted).
        let (evals, _) = g.eigh(Side::Lower).expect("eigh");
        let lam_max = evals.iter().fold(0.0_f64, |acc, v| acc.max(*v));
        let replaced_floor = 64.0 * f64::EPSILON * 2.0 * lam_max;
        assert!(
            formation_error > replaced_floor,
            "fixture must carry formation error {formation_error:e} above the replaced floor {replaced_floor:e}"
        );
        assert!(
            evals.iter().filter(|&&v| v > band).count() == 1,
            "the tangent eigenvalue must clear the derived band {band:e}: {evals:?}"
        );
    }
}
