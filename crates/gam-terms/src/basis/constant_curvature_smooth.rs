//! Constant-curvature (`M_κ`) smooth term: basis + penalty over the
//! κ-stereographic chart (#944, stage 3 step 1).
//!
//! The term is the κ-generic sibling of the intrinsic-S² Wahba smooth
//! (`sphere_spec.rs` / `build_spherical_spline_basis`): a reproducing-kernel
//! basis on a center set, with the kernel Gram on the centers as the RKHS
//! roughness penalty and a coefficient-space sum-to-zero constraint for
//! identifiability. Where the Wahba smooth hard-codes S² (lat/lon chart,
//! Legendre kernels), this term takes the geometry from
//! [`gam_geometry::constant_curvature::ConstantCurvature`] at an explicit
//! curvature κ, so one construction covers the whole interpolation
//! `S^d(1/√κ) → ℝ^d → H^d(1/√−κ)` through κ = 0.
//!
//! # Kernel
//!
//! `k_{κ,ℓ}(x, y) = ℓ·(exp(−d_κ(x, y)/ℓ) − 1)` — the geodesic-exponential
//! kernel in its CONTRAST gauge, where `d_κ` is the exact constant-curvature
//! geodesic distance in the κ-stereographic chart. The geodesic distance is a
//! kernel of conditionally negative type on all three constant-curvature space
//! forms (Schoenberg 1942 for `S^d`; classical CND of `‖·‖` on `ℝ^d`;
//! Faraut–Harzallah 1974 for `H^d`), so `exp(−c·d_κ)` is positive definite for
//! every `c > 0` and every κ, and the Gram on distinct centers is strictly PD
//! on the sum-to-zero frame — which is exactly what the RKHS penalty
//! construction needs. At κ = 0 the chart carries the doubled gauge
//! (`metric 4δ`, `d_0(x, y) = 2‖x − y‖`), so the κ = 0 term is the Euclidean
//! exponential (Matérn-½) kernel smooth with effective Euclidean range `ℓ/2`.
//!
//! The `ℓ` factor and the subtracted `1` are both invisible to the model —
//! `z` annihilates constants and `λ` absorbs a positive scale — and both are
//! load-bearing anyway, because the gauge is what decides whether the range
//! coordinate is confounded with `ρ` and whether the criterion survives its own
//! rounding at large `ℓ` (gam#2747). The derivation, with the measurements that
//! forced it, is on [`constant_curvature_kernel_matrix`]. Its consequence for
//! the family: `k → −d_κ` as `ℓ → ∞`, so the range's far face is the
//! geodesic-distance kernel — an ordinary member of the family's own closure,
//! not a degeneracy.
//!
//! # The exponent has TWO coordinates, and BOTH are estimated (gam#2747)
//!
//! `d_κ/ℓ` carries a curvature and a range, and they are strongly confounded:
//! to first order `d_κ = d_0·(1 + κ·a(x,y))`, so the MEAN of `a` over the
//! evaluated pairs acts exactly like a rescaling of `ℓ` and only the VARIATION
//! of `a` is genuine curvature. A construction that fits κ while pinning ℓ to a
//! heuristic therefore does not estimate curvature at all — it lets κ absorb
//! whatever range correction the heuristic got wrong, which is monotone in one
//! direction, so `V_p(κ)` rails.
//!
//! That was measured on truths planted inside the fitted span (gam#2747): with
//! `ℓ` pinned at the auto `ℓ_ref`, the criterion recovers the planted κ⋆ only
//! when the truth's own radial length scale happens to BE `ℓ_ref`. At half or
//! twice that range it rails at a box endpoint, reports the wrong SIGN, or —
//! on genuinely FLAT data — reports a confident interior `κ̂ = ∓0.94`. With
//! `ℓ` profiled it lands within 0.19 of κ⋆ in all nine cells of a
//! 3 curvatures × 3 ranges sweep (median 0.07), with no rails and no sign
//! inversions, and `ℓ̂` recovers the planted range to 3%.
//!
//! So the smooth exposes the ψ-vector `(κ, η)` with `η = ln ℓ` — the same
//! treatment the Matérn / Duchon / TPS siblings already give their range,
//! extended by the signed curvature coordinate that keeps κ = 0 interior.
//! Two prior attempts to remove the confounding by CONSTRAINT rather than by
//! estimation are retired by this: `#1059`'s mean-geodesic-distance
//! normalization and `#944`'s fill-invariant `L(κ)` both pin one scalar summary
//! of the design, which selects a one-dimensional curve through the `(κ, ℓ)`
//! plane a priori. On such a curve `dV/dκ = V_κ + V_ℓ·L′(κ)`, and the second
//! term vanishes only if `ℓ_ref` was already optimal; on the profile curve it
//! vanishes identically by the envelope theorem. `#1464`'s separate penalty
//! length `L_S(κ)` is retired with them: it made `S` the Gram of a DIFFERENT
//! kernel than the design's, so the penalty was no longer the RKHS roughness
//! of the function it penalized.
//!
//! # ψ-differentiability contract (what the outer stage consumes)
//!
//! Every ψ-moving piece of this construction is differentiable in `(κ, η)` in
//! closed form, and every ψ-FIXED piece is documented as such:
//!
//! - **Centers are ψ-fixed.** Center selection runs in chart coordinates
//!   (farthest-point / k-means / user-provided) and deliberately consults
//!   neither κ nor ℓ, so `∂(centers)/∂ψ ≡ 0` and the design moves with ψ only
//!   through the kernel. A ψ-dependent center rule would add an uncontrolled,
//!   non-smooth term to the design drift.
//! - **The constraint transform `z` is ψ-fixed.** Uniform coefficient
//!   weights; at fit time the global identifiability pipeline composes the
//!   parametric orthogonalization onto it and the result is FROZEN
//!   (mirroring `SphericalSplineIdentifiability::FrozenTransform`, #532), so
//!   the predict/ψ-trial rebuild replays the same `z` verbatim.
//! - **The kernel has exact ψ-jets.** With `q = d_κ(x,y)·e^{−η}` and the
//!   Tower4-exact, FD-gated `distance_kappa_jet` supplying `(d, d′, d″)`:
//!   `q_κ = d′/ℓ`, `q_κκ = d″/ℓ`, `q_η = −q`, `q_κη = −q_κ`, `q_ηη = q`, and
//!   `K = e^{−q}` gives `K_a = −q_a K`, `K_ab = (q_a q_b − q_ab) K` — see
//!   [`constant_curvature_kernel_psi_jets`]. The realized drifts follow by the
//!   ψ-fixed transforms: `∂X/∂ψ_a = (∂K_dc/∂ψ_a)·z` and
//!   `∂S/∂ψ_a = symm(zᵀ(∂K_cc/∂ψ_a)z)`, with no normalization quotient to
//!   propagate (the RKHS penalty ships raw, `normalization_scale = 1`).
//! - **Available but not yet consumed:** `log_map_kappa_jet` /
//!   `exp_map_kappa_jet` cover future geodesic/normal-coordinate basis
//!   variants (e.g. tangent-space designs); the distance jet is the only one
//!   this kernel construction needs.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use gam_geometry::constant_curvature::{ConstantCurvature, distance_kappa_jet};

use super::{
    ActivePenalty, BasisBuildResult, BasisError, BasisMetadata, BasisPsiDerivativeBundle,
    BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult, CenterStrategy, CenterStrategyKind,
    ConstructiveQuadratic, PenaltyCandidate, PenaltySource, center_strategy_kind,
    filter_penalty_candidates, normalize_penalty, select_centers_by_strategy,
    weighted_coefficient_sum_to_zero_transform,
};

/// Realized-design identifiability policy for the constant-curvature smooth.
/// Mirrors [`super::SphericalSplineIdentifiability`] (#532): the fit-time
/// center-space sum-to-zero `z` gets the parametric orthogonalization composed
/// onto it by the global identifiability pipeline, and the composed transform
/// is frozen here so predict-time (and future per-ψ-trial) rebuilds replay it
/// verbatim instead of recomputing `z` from the centers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ConstantCurvatureIdentifiability {
    /// Fit-time default: uniform-weight coefficient sum-to-zero over the
    /// centers (`Σ_j α_j = 0`), then global parametric residualization.
    #[default]
    CenterSumToZero,
    /// Predict-time replay: the frozen composed transform captured at fit
    /// time. `transform.nrows()` equals the number of centers.
    FrozenTransform { transform: Array2<f64> },
}

/// Constant-curvature smooth configuration (`curv(x, z, kappa = …)`).
///
/// The chart inputs are the raw feature columns interpreted as
/// κ-stereographic chart coordinates: any finite point for κ ≥ 0, the open
/// ball `‖x‖ < 1/√(−κ)` for κ < 0. The default κ = 0 reproduces a Euclidean
/// exponential-kernel smooth (in the doubled κ = 0 chart gauge), so the term
/// is safe to use as a drop-in flat smooth until κ becomes a fitted
/// ψ-coordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantCurvatureBasisSpec {
    /// Center/knot selection strategy in chart coordinates. Deliberately
    /// κ-independent (see the module-level κ-contract).
    pub center_strategy: CenterStrategy,
    /// Sectional curvature κ of the latent/feature geometry. Fixed at build
    /// time; when [`Self::kappa_fixed`] is `false` the later ψ-channel stage
    /// promotes it to a fitted outer coordinate consuming this module's exact
    /// κ-jets, and this field is only the seed. When `kappa_fixed` is `true`
    /// this value is the user's PINNED sectional curvature and the outer loop
    /// must hold it constant (never re-derive it).
    pub kappa: f64,
    /// Did the user explicitly pin the sectional curvature (`curv(.., kappa=K)`)?
    ///
    /// This is the mgcv-`sp=` convention applied to κ: an explicit `kappa=`
    /// selects a FIXED geometry (`Sᵈ` for κ>0, `ℝᵈ` for κ=0, `Hᵈ` for κ<0) and
    /// the fit builds/keeps the design and penalty at exactly that κ; an OMITTED
    /// `kappa=` leaves κ free, seeded at [`Self::kappa`] (default 0), for the
    /// #944/#1464 outer ψ-coordinate estimation to fit. The two paths must never
    /// be confused: honoring the pin is the whole contract of a fixed-curvature
    /// smooth (gam#2152), while the estimation path is the whole point of the
    /// κ-inference subsystem. Defaults to `false` (estimate) so the estimand
    /// machinery and every serialized pre-#2152 model keep their behaviour.
    #[serde(default)]
    pub kappa_fixed: bool,
    /// Geodesic kernel range ℓ in `K_κ = exp(−d_κ/ℓ)`. The `0.0` sentinel
    /// requests the κ-independent auto initialization
    /// ([`realized_constant_curvature_length_scale`]); the realized value is
    /// persisted in [`BasisMetadata::ConstantCurvature`] and frozen back into
    /// the spec for predict-time replay.
    ///
    /// When [`Self::length_scale_fixed`] is `false` this is only the SEED: the
    /// range is the smooth's second outer ψ-coordinate (`η = ln ℓ`) and is
    /// estimated jointly with κ, because a pinned range makes κ absorb the
    /// range error instead of measuring curvature (gam#2747, module docs).
    pub length_scale: f64,
    /// Did the user explicitly pin the kernel range (`curv(.., length_scale=L)`)?
    ///
    /// The mgcv-`sp=` convention that [`Self::kappa_fixed`] already applies to
    /// the curvature, applied to the range: an explicit `length_scale=` selects
    /// a FIXED kernel resolution and the fit honors it verbatim; an OMITTED
    /// `length_scale=` leaves `η = ln ℓ` free for the gam#2747 outer estimation,
    /// seeded at the auto rule. Defaults to `false` (estimate), so a spec that
    /// merely had its realized ℓ frozen back into it after a fit is not
    /// mistaken for a user pin on the next fit.
    #[serde(default)]
    pub length_scale_fixed: bool,
    /// Add the ridge-like shrinkage penalty alongside the RKHS Gram penalty.
    pub double_penalty: bool,
    /// Realized-design identifiability policy (see type docs).
    #[serde(default)]
    pub identifiability: ConstantCurvatureIdentifiability,
}

impl Default for ConstantCurvatureBasisSpec {
    fn default() -> Self {
        Self {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 50 },
            kappa: 0.0,
            kappa_fixed: false,
            length_scale: 0.0,
            length_scale_fixed: false,
            // No double-penalty ridge by default (#1464). The RKHS Gram penalty
            // zᵀKz is strictly PD/full-rank on distinct centers, so it already
            // regularizes every coefficient direction — the ridge `I` adds no
            // stability. Worse, `I` is curvature-BLIND: with its own λ it absorbs
            // the data fit independently of κ. Curvature-sign identification
            // remains the separate #1464 problem; an extra curvature-blind
            // ridge cannot resolve it.
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        }
    }
}

/// Validate that every row of `points` is finite and inside the
/// κ-stereographic chart: `1 + κ‖x‖² > 0`, the open ball `‖x‖ < 1/√(−κ)` for
/// κ < 0 and vacuous for κ ≥ 0.
///
/// **The vacuity on the κ ≥ 0 branch is not a gap, and this doc used to read as
/// if it were** (gam#2687 quoted the old wording — *"automatic for κ ≥ 0"* — as
/// evidence that the code did not implement its own comment). The two branches
/// have different constraints, of different ARITY, enforced at different sites:
///
/// * **κ < 0 — a PER-POINT gauge, enforced here.** `λ(p) = 1 + κ‖p‖²` is the
///   conformal factor's denominator; it vanishes when `p` reaches the Poincaré
///   ball's boundary. One point is enough to violate it, so a per-row scan is
///   the right check and this function is where it belongs. It runs on data
///   **and** centers, because both are points the kernel evaluates.
/// * **κ > 0 — a PER-PAIR fold, enforced by the κ box.** `λ` really cannot
///   vanish for κ ≥ 0, but the quantity that does is
///   `D = 1 + 2κ⟨x,c⟩ + κ²‖x‖²‖c‖²`, the Möbius denominator of
///   `w = (−x) ⊕_κ c`, which is `(1 − κ‖x‖‖c‖)²` for an anti-aligned pair and
///   vanishes at the antipodal fold `κ = 1/(‖x‖‖c‖)`. No per-point predicate can
///   see it — it is a property of a PAIR — so the retreat is taken upstream, by
///   [`constant_curvature_kappa_bounds`](crate::smooth::constant_curvature_kappa_bounds),
///   which caps `|κ|` at `F/R²` over `R = max‖p‖` on `data ∪ centers` (gam#2716:
///   over `data` alone, a user-provided center past `2·max‖x‖` put the box past
///   the fold and made it doubly covered).
///
/// So there is exactly one wall per branch and each is checked against its own
/// geometry; neither is the other mirrored. Gated by
/// `spherical_branch_folds_at_kappa_r2_one_so_the_kappa_window_is_symmetric_2687`
/// (gam-geometry) and by `crate::smooth::constant_curvature_kappa_box_tests`,
/// which drives both endpoints of the shipped box through this function and
/// through the shipped `distance`.
pub(crate) fn validate_chart_points(
    points: ArrayView2<'_, f64>,
    kappa: f64,
    what: &str,
) -> Result<(), BasisError> {
    for (i, row) in points.outer_iter().enumerate() {
        let mut nx2 = 0.0_f64;
        for &v in row.iter() {
            if !v.is_finite() {
                crate::bail_invalid_basis!(
                    "constant-curvature {what} row {i} has a non-finite coordinate"
                );
            }
            nx2 += v * v;
        }
        if 1.0 + kappa * nx2 <= 0.0 {
            crate::bail_invalid_basis!(
                "constant-curvature {what} row {i} lies outside the κ-stereographic chart \
                 (need 1 + κ·‖x‖² > 0; got κ = {kappa}, ‖x‖² = {nx2}); for κ < 0 the chart is \
                 the open ball ‖x‖ < 1/√(−κ)"
            );
        }
    }
    Ok(())
}

/// `φ(u) = e^{−u}(1 + u) − 1` — the `∂/∂η` shape factor of the normalized
/// kernel, evaluated without cancellation.
///
/// `φ` vanishes to SECOND order at `u = 0` (`φ = −u²/2 + u³/3 − …`) while both
/// of its terms are `O(1)`, so the direct form loses `2·log₁₀(1/u)` digits as
/// `u → 0` — which is exactly the regime the range coordinate walks into. The
/// series is `−Σ_{m≥2} (−1)^m (m−1) u^m / m!`, used below `u = 1/2` where its
/// terms are monotone and the truncation at `m = 20` is `< 2⁻²⁰/20! ≈ 5e-25`.
#[inline]
fn eta_shape(u: f64) -> f64 {
    if u >= 0.5 {
        return (-u).exp() * (1.0 + u) - 1.0;
    }
    let mut term = u * u; // u^m / m! at m = 2, times 2!
    let mut factorial = 2.0_f64;
    let mut sum = 0.0_f64;
    let mut sign = 1.0_f64; // (−1)^m at m = 2
    for m in 2..=20u32 {
        if m > 2 {
            term *= u;
            factorial *= f64::from(m);
            sign = -sign;
        }
        sum -= sign * f64::from(m - 1) * term / factorial;
    }
    sum
}

/// `χ(u) = e^{−u}(1 + u + u²) − 1` — the `∂²/∂η²` shape factor.
///
/// Same second-order zero at the origin (`χ = u²/2 − 2u³/3 + …`) and the same
/// series treatment, with coefficients `(m−1)²/m!`. Away from the origin the
/// direct form is used; `χ` has a genuine root near `u = 2.15`, where relative
/// precision is unavailable to any formula and irrelevant — the block is summed
/// over pairs, so absolute accuracy is what the Hessian consumes.
#[inline]
fn eta2_shape(u: f64) -> f64 {
    if u >= 0.5 {
        return (-u).exp() * (1.0 + u + u * u) - 1.0;
    }
    let mut term = u * u;
    let mut factorial = 2.0_f64;
    let mut sum = 0.0_f64;
    let mut sign = 1.0_f64;
    for m in 2..=20u32 {
        if m > 2 {
            term *= u;
            factorial *= f64::from(m);
            sign = -sign;
        }
        let weight = f64::from(m - 1) * f64::from(m - 1);
        sum += sign * weight * term / factorial;
    }
    sum
}

/// The model kernel at one pair: `k = ℓ·(e^{−d/ℓ} − 1)`, evaluated as
/// `ℓ·expm1(−d/ℓ)`.
#[inline]
pub(crate) fn constant_curvature_kernel_scalar(distance: f64, length_scale: f64) -> f64 {
    length_scale * (-distance / length_scale).exp_m1()
}

/// `k_κ(data, centers)` — the realized constant-curvature model kernel matrix
/// `ℓ·(e^{−d_κ(x_i,c_j)/ℓ} − 1)`, evaluated as `ℓ·expm1(−d_κ/ℓ)`.
///
/// # Why this is not `exp(−d_κ/ℓ)` (gam#2747)
///
/// The kernel is only ever consumed through the coefficient sum-to-zero frame
/// `z`: the realized design is `K z` and the realized penalty is `zᵀ K z`. `z`
/// annihilates constants, so the construction is invariant under `K → K + c1ᵀ`
/// for any per-row `c`, and multiplying `K` by a positive scalar is absorbed
/// exactly by the smoothing parameter (the prior on the fitted function is
/// `(1/λ)·X S⁻ Xᵀ`, invariant under `(X, S, λ) → (aX, aS, aλ)`). So `exp(−d/ℓ)`
/// and `ℓ·(e^{−d/ℓ} − 1)` are the SAME model in two gauges — and the gauge is
/// not free, because two things are decided by it.
///
/// **It decides whether the range coordinate is confounded with the smoothing
/// parameter.** `exp(−d/ℓ)z = −(1/ℓ)Dz + O(1/ℓ²)`, so the realized design and
/// penalty both collapse like `1/ℓ` and `λ̂` has to chase them: measured on the
/// κ = 1 sphere fixture, `ρ̂` falls one-for-one with `ln ℓ` over eleven decades
/// (`−5.49 → −18.91` from `ℓ = 1` to `10⁶`) while the criterion value is
/// unchanged to eight significant figures. A range search on an absolute `ρ`
/// box therefore walks `ρ̂` into `RHO_LOWER` for no statistical reason. In this
/// gauge `ρ̂` is FLAT (`−5.0978 ± 1e-4` over the same eleven decades).
///
/// **And it decides whether the criterion is a function of the data at all.**
/// All of the model's range information lives in `K − 1`; forming it by
/// subtracting `exp(−d/ℓ)` from an implicit `1` costs `log₁₀(ℓ/d)` significant
/// digits, and the Gram then squares what is left. Measured on the same
/// fixture, the `exp` gauge's REML value departs from the truth by **78.8 nats
/// at the derived box top** `ℓ_hi = d_min/√ε = 2.53e6` and by 476 nats at
/// `ℓ = 10⁸`, descending ~100 nats per decade into its own rounding with `edf`
/// railed at `p` — which is what a range search reads when it "converges to an
/// asymptote" at the box end. `expm1` forms `K − 1` directly and the departure
/// is zero to eight figures at `ℓ = 10⁹`.
///
/// # Consequences of the gauge, all of them intended
///
/// * `k ≤ 0`, with `k = 0` exactly on coincident points. The realized penalty
///   `zᵀkz = ℓ·zᵀe^{−d/ℓ}z` is still strictly positive definite on the frame —
///   `−k` is a conditionally negative definite kernel, which is what makes it
///   so — but the RAW `m × m` matrix is no longer PSD, so the builder forms the
///   penalty from the RESTRICTED Gram rather than restricting a raw PSD one.
/// * `k → −d_κ` as `ℓ → ∞`, exactly and with no cancellation. The `ℓ = ∞` face
///   is the geodesic-distance kernel, an ordinary non-degenerate member of the
///   family's own closure rather than a degenerate limit — which is what lets
///   the range coordinate be compactified instead of walled.
pub fn constant_curvature_kernel_matrix(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    length_scale: f64,
) -> Result<Array2<f64>, BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature kernel dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature kernel needs a positive finite length_scale; got {length_scale}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let mut out = Array2::<f64>::zeros((data.nrows(), centers.nrows()));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| -> Result<(), BasisError> {
            for (j, c) in centers.outer_iter().enumerate() {
                let d = manifold.distance(data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature distance failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                row[j] = constant_curvature_kernel_scalar(d, length_scale);
            }
            Ok(())
        })?;
    Ok(out)
}

/// The `(k, ∂k/∂ψ_a, ∂²k/∂ψ_a∂ψ_b)` tower of the raw (pre-constraint) model
/// kernel matrix in BOTH outer coordinates, `ψ = (κ, η)` with `η = ln ℓ`.
///
/// Exact. `distance_kappa_jet` (Tower4, FD-gated in
/// `geometry::constant_curvature`) supplies `(d, d′, d″)`; everything else is
/// the chain rule on `k = ℓ·(e^{−q} − 1)` with `q = d_κ(x,y)·e^{−η}` (see
/// [`constant_curvature_kernel_matrix`] for why the kernel carries the `ℓ`
/// factor and the annihilated `−1`):
///
/// ```text
///   k_κ  = −d′·e^{−q}                  k_κκ = e^{−q}·(d′²/ℓ − d″)
///   k_η  =  ℓ·φ(q),  φ(u) = e^{−u}(1+u) − 1
///   k_κη = −d′·q·e^{−q}
///   k_ηη =  ℓ·χ(q),  χ(u) = e^{−u}(1+u+u²) − 1
/// ```
///
/// Every block has a finite `ℓ → ∞` limit, and the two that vanish there do so
/// through `φ` and `χ`, both of which have a second-order zero at the origin —
/// so they are evaluated by series below `u = 1/2` rather than by the
/// two-term difference, which would lose `2·log₁₀(ℓ/d)` digits exactly where
/// the range coordinate is least identified. `k_κ`, `k_κκ` and `k_κη` are
/// cancellation-free as written.
///
/// The `η` channel is what makes the range an ESTIMAND rather than a heuristic
/// (gam#2747): with `ℓ` pinned, κ absorbs the range error and the profiled
/// criterion rails. The realized design/penalty drifts follow by the ψ-fixed
/// transforms `∂X/∂ψ_a = (∂K_dc/∂ψ_a)·z`,
/// `∂S/∂ψ_a = symm(zᵀ(∂K_cc/∂ψ_a)z)`; the RKHS penalty ships raw
/// (`normalization_scale = 1`), so no normalization quotient rule participates.
#[derive(Clone, Debug)]
pub struct ConstantCurvatureKernelPsiJets {
    /// `k`.
    pub value: Array2<f64>,
    /// `∂k/∂κ`.
    pub d_kappa: Array2<f64>,
    /// `∂k/∂η`, `η = ln ℓ`.
    pub d_eta: Array2<f64>,
    /// `∂²k/∂κ²`.
    pub d_kappa2: Array2<f64>,
    /// `∂²k/∂κ∂η`.
    pub d_kappa_eta: Array2<f64>,
    /// `∂²k/∂η²`.
    pub d_eta2: Array2<f64>,
}

/// Build [`ConstantCurvatureKernelPsiJets`] for one `(data, centers)` block.
pub fn constant_curvature_kernel_psi_jets(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    kappa: f64,
    length_scale: f64,
) -> Result<ConstantCurvatureKernelPsiJets, BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature kernel-jet dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    if !(length_scale.is_finite() && length_scale > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature kernel jets need a positive finite length_scale; got {length_scale}"
        );
    }
    validate_chart_points(data, kappa, "data")?;
    validate_chart_points(centers, kappa, "centers")?;
    let manifold = ConstantCurvature::new(data.ncols(), kappa);
    let n = data.nrows();
    let m = centers.nrows();
    let mut jets = ConstantCurvatureKernelPsiJets {
        value: Array2::<f64>::zeros((n, m)),
        d_kappa: Array2::<f64>::zeros((n, m)),
        d_eta: Array2::<f64>::zeros((n, m)),
        d_kappa2: Array2::<f64>::zeros((n, m)),
        d_kappa_eta: Array2::<f64>::zeros((n, m)),
        d_eta2: Array2::<f64>::zeros((n, m)),
    };
    let rows: Vec<(usize, Vec<[f64; 6]>)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(usize, Vec<[f64; 6]>), BasisError> {
            let mut row = Vec::with_capacity(m);
            for (j, c) in centers.outer_iter().enumerate() {
                let (d, d1, d2) = distance_kappa_jet(&manifold, data.row(i), c).map_err(|e| {
                    BasisError::InvalidInput(format!(
                        "constant-curvature distance κ-jet failed at (row {i}, center {j}): {e}"
                    ))
                })?;
                let q = d / length_scale;
                let decay = (-q).exp();
                row.push([
                    constant_curvature_kernel_scalar(d, length_scale),
                    -d1 * decay,
                    length_scale * eta_shape(q),
                    decay * (d1 * d1 / length_scale - d2),
                    -d1 * q * decay,
                    length_scale * eta2_shape(q),
                ]);
            }
            Ok((i, row))
        })
        .collect::<Result<Vec<_>, BasisError>>()?;
    for (i, row) in rows {
        for (j, entry) in row.into_iter().enumerate() {
            jets.value[(i, j)] = entry[0];
            jets.d_kappa[(i, j)] = entry[1];
            jets.d_eta[(i, j)] = entry[2];
            jets.d_kappa2[(i, j)] = entry[3];
            jets.d_kappa_eta[(i, j)] = entry[4];
            jets.d_eta2[(i, j)] = entry[5];
        }
    }
    Ok(jets)
}

/// Resolve the realized kernel range ℓ. An explicit positive `spec_length_scale`
/// is used verbatim; the `0.0` sentinel auto-initializes from the median
/// pairwise CHART distance among the centers, doubled to match the κ = 0
/// chart gauge (`d_0 = 2‖Δ‖`).
///
/// This is a SEED unless the user pinned it (`length_scale_fixed`): the range is
/// the smooth's second outer coordinate and the fit estimates it (gam#2747).
/// The auto rule reads chart coordinates only — it never consults κ — so the
/// seed and the derived search window
/// ([`constant_curvature_length_scale_bounds`]) are both κ-FIXED, and the outer
/// box does not move while the optimizer walks κ.
pub fn realized_constant_curvature_length_scale(
    centers: ArrayView2<'_, f64>,
    spec_length_scale: f64,
) -> Result<f64, BasisError> {
    if spec_length_scale.is_finite() && spec_length_scale > 0.0 {
        return Ok(spec_length_scale);
    }
    if spec_length_scale != 0.0 {
        crate::bail_invalid_basis!(
            "constant-curvature length_scale must be positive (or 0.0 for auto); got {spec_length_scale}"
        );
    }
    let dists = center_chart_gauge_distances(centers)?;
    let median = dists[dists.len() / 2];
    if !(median.is_finite() && median > 0.0) {
        crate::bail_invalid_basis!(
            "constant-curvature auto length_scale failed: centers are degenerate \
             (median pairwise chart distance = {median})"
        );
    }
    Ok(median)
}

/// The sorted multiset of pairwise center distances in the κ = 0 doubled chart
/// gauge (`d_0 = 2‖Δ‖`) — the single source both the auto `ℓ_ref` (its median)
/// and the derived range window (its ends) are read from.
fn center_chart_gauge_distances(centers: ArrayView2<'_, f64>) -> Result<Vec<f64>, BasisError> {
    let m = centers.nrows();
    if m < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: m });
    }
    let mut dists: Vec<f64> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            let mut s = 0.0_f64;
            for k in 0..centers.ncols() {
                let dlt = centers[(i, k)] - centers[(j, k)];
                s += dlt * dlt;
            }
            dists.push(2.0 * s.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).expect("finite chart distances"));
    Ok(dists)
}

/// The scales the kernel actually evaluates: `(d_min⁺, d_max)` over the
/// data→center **and** center→center pairs, in the κ = 0 doubled chart gauge,
/// excluding the exact zeros that self-pairs contribute.
///
/// This is the same pair set the chart guard validates and the κ box takes its
/// radius over (`data ∪ centers`, gam#2716) — one set, so a configuration that
/// moves one of the smooth's two outer boxes moves the other consistently.
pub fn constant_curvature_evaluated_scale_span(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(f64, f64), BasisError> {
    if data.ncols() != centers.ncols() {
        crate::bail_dim_basis!(
            "constant-curvature scale span dimension mismatch: data d={} centers d={}",
            data.ncols(),
            centers.ncols()
        );
    }
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    let mut observe = |a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>| {
        let mut sum = 0.0_f64;
        for k in 0..a.len() {
            let delta = a[k] - b[k];
            sum += delta * delta;
        }
        let d = 2.0 * sum.sqrt();
        if d.is_finite() && d > 0.0 {
            lo = lo.min(d);
            hi = hi.max(d);
        }
    };
    for x in data.outer_iter() {
        for c in centers.outer_iter() {
            observe(x, c);
        }
    }
    for i in 0..centers.nrows() {
        for j in (i + 1)..centers.nrows() {
            observe(centers.row(i), centers.row(j));
        }
    }
    if !(lo.is_finite() && lo > 0.0 && hi.is_finite() && hi >= lo) {
        crate::bail_invalid_basis!(
            "constant-curvature range window is undefined: the evaluated pairs carry no \
             positive chart distance (d_min = {lo}, d_max = {hi})"
        );
    }
    Ok((lo, hi))
}

/// DERIVED box `[ℓ_lo, ℓ_hi]` for the kernel range — the interval on which the
/// realized design's GRAM is still resolvable in double precision, which is the
/// only thing a box on this coordinate is entitled to enforce.
///
/// Two derivations were tried and measured before this one.
///
/// The first bounded `ℓ` by the scales the geometry contains, `[d_min⁺, d_max]`.
/// That is wrong because the criterion `V(κ⋆, ℓ)` is sharply unimodal with an
/// interior minimum that recovers the planted range and rises monotonically on
/// both sides across four log-units — it walls the range in by itself — while
/// its minimum sits OUTSIDE the center set's own span in a third of the planted
/// cells. A window at the geometry's scales rails a coordinate the criterion
/// handles perfectly well, trading one artificial constraint for another, which
/// is exactly the mistake `#944` and `#1464` made in the κ direction.
///
/// The second put the wall at REPRESENTABILITY: `ℓ_lo = d_max/ln(1/MIN_POSITIVE)`
/// and `ℓ_hi = d_min⁺/EPSILON`, the points at which `exp(−d_max/ℓ)` underflows
/// and `1 − exp(−d_min⁺/ℓ)` rounds away. That is 39× too permissive at the
/// bottom and seven orders too permissive at the top, because **the criterion
/// does not evaluate the kernel — it evaluates a Cholesky of `H = XᵀX + λS`**,
/// and the Gram SQUARES the design's dynamic range. At `ℓ_lo` so defined the
/// design entries span 300 orders of magnitude, `H` is numerically singular,
/// and the profile's derivatives come back at `10⁶`–`10⁸` — which is how a
/// bounded outer solve ends up reporting `|Pg| = 8.2e6` against a stationarity
/// bound of `1.1e-3` and failing its line search.
///
/// So the wall belongs where the linear algebra lives:
///
/// * **`ℓ_lo = d_max / (½·ln(1/ε))`** — with `q = d/ℓ`, the design's entries
///   span `e^{−q_max}` and `XᵀX` spans its square, so a Cholesky resolves the
///   Gram only while `e^{−2·q_max} ≥ ε`, i.e. `q_max ≤ ½·ln(1/ε) ≈ 18`.
/// * **`ℓ_hi = d_max / (2√ε)`** — the far end is not a numerical wall at all,
///   and the third derivation is what made that visible. `ℓ_hi` used to be
///   `d_min⁺/√ε`, the point at which `1 − e^{−q_min}` rounds away and the Gram
///   squares what is left. That loss was REAL, but it was a property of the
///   `exp(−d/ℓ)` gauge rather than of the model: the design is `Kz` and `z`
///   annihilates constants, so all the range information lived in `K − 1`, and
///   forming it by subtraction cost `log₁₀(ℓ/d)` digits. The criterion built on
///   it was measurably FALSE well before that wall — 78.8 nats of fabricated
///   descent AT `ℓ_hi`, ~100 nats per decade past it, `edf` railed at `p`
///   (gam#2747). The contrast gauge
///   ([`constant_curvature_kernel_matrix`]) forms `K − 1` directly and the loss
///   is gone, so nothing numerical bounds the range from above any more.
///
///   What bounds it is the MODEL: `k → −d_κ` as `ℓ → ∞`, so the far face of the
///   range is the geodesic-distance kernel, and the departure from it is
///   `d/(2ℓ)` relative, first order. Truncating the chart where the LARGEST
///   evaluated pair is within `√ε` of its limit puts every entry of the design
///   within the square root of machine precision of the limit design — past
///   that point the range is not identified because there is nothing left to
///   identify, and an estimate reported there is a statement about the model
///   (`the kernel IS the geodesic distance`) rather than about the box. That is
///   what [`RangeSolveOutcome`](../../../gam_models/index.html) declares rather
///   than leaves to be inferred.
///
/// Both ends are read in the κ = 0 doubled gauge, the same gauge the auto
/// `ℓ_ref` rule uses, so the box is κ-FIXED and does not move while the
/// optimizer walks κ. It is still wide — some eight orders — and deliberately
/// so: the criterion supplies the shape, this supplies only the wall.
/// Bracketing the inner search is a separate concern and uses
/// [`constant_curvature_evaluated_scale_span`] directly.
pub fn constant_curvature_length_scale_bounds(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
) -> Result<(f64, f64), BasisError> {
    let (d_min, d_max) = constant_curvature_evaluated_scale_span(data, centers)?;
    let gram_resolvable_efolds = 0.5 * -f64::EPSILON.ln();
    let lo = d_max / gram_resolvable_efolds;
    // The top end no longer reads `d_min` — the closest pair's contrast was the
    // RETIRED gauge's failure mode, not this one's — but the span is still read
    // as a pair because `d_min` guards the span's own validity and names the
    // geometry in the refusal below.
    let hi = d_max / (2.0 * f64::EPSILON.sqrt());
    if !(lo.is_finite() && lo > 0.0 && hi.is_finite() && hi > lo) {
        crate::bail_invalid_basis!(
            "constant-curvature range box collapsed: [{lo}, {hi}] from an evaluated span of \
             [{d_min}, {d_max}]"
        );
    }
    Ok((lo, hi))
}

/// Build the constant-curvature reproducing-kernel smooth: realized design
/// `K_κ(data, centers)·z`, RKHS penalty `zᵀ K_κ(centers, centers) z`, and the
/// replayable [`BasisMetadata::ConstantCurvature`]. Structure mirrors the
/// Wahba S² builder (`build_spherical_spline_basis`); geometry comes from
/// `ConstantCurvature` at the spec's fixed κ.
pub fn build_constant_curvature_basis(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("constant-curvature smooth needs at least one feature column");
    }
    if !spec.kappa.is_finite() {
        crate::bail_invalid_basis!("constant-curvature smooth needs a finite kappa");
    }
    validate_chart_points(data, spec.kappa, "data")?;
    let centers = select_constant_curvature_centers(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    validate_chart_points(centers.view(), spec.kappa, "centers")?;
    // ONE kernel, ONE range (gam#2747). The design and the penalty are the two
    // blocks of the SAME Gram — `X = k_{κ,ℓ}(data,C)z` and
    // `S = zᵀk_{κ,ℓ}(C,C)z` — which is what makes `S` the RKHS roughness of the
    // function `X` realizes, and the whole model the ordinary
    // subset-of-regressors GP with kernel `exp(−d_κ/ℓ)`, written in the
    // contrast gauge `k = ℓ·(e^{−d_κ/ℓ} − 1)` that
    // `constant_curvature_kernel_matrix` derives.
    //
    // Two earlier constructions evaluated them at DIFFERENT lengths: `#944`'s
    // fill-invariant `L(κ)` for the design and `#1464`'s `L_S(κ)` for the
    // penalty, each solved from its own fill target. Both were attempts to
    // remove the κ/ℓ confounding by CONSTRAINT — pinning one scalar summary of
    // the design so κ could not buy resolution — and both fail for the same
    // reason: pinning a summary selects a one-dimensional curve through the
    // `(κ, ℓ)` plane a priori, and on such a curve `dV/dκ = V_κ + V_ℓ·L′(κ)`
    // carries a range term that vanishes only if the heuristic `ℓ_ref` was
    // already optimal. Measured (gam#2747, truths planted inside the fitted
    // span): with `ℓ` pinned the criterion recovers κ⋆ only when the truth's own
    // range IS `ℓ_ref`, and at half or twice that range it rails, inverts the
    // sign, or invents `κ̂ = ∓0.94` from flat data. The confounding is removed
    // by ESTIMATING the range instead — `η = ln ℓ` is the smooth's second outer
    // coordinate — after which the envelope theorem kills the leak exactly.
    let length_scale = realized_constant_curvature_length_scale(centers.view(), spec.length_scale)?;
    let raw_penalty =
        constant_curvature_kernel_matrix(centers.view(), centers.view(), spec.kappa, length_scale)?;
    // Realized-design constraint transform: uniform coefficient sum-to-zero at
    // fit time; the frozen composed `z · z_parametric` at predict time (#532
    // pattern — see ConstantCurvatureIdentifiability).
    let z = match &spec.identifiability {
        ConstantCurvatureIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen constant-curvature identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        ConstantCurvatureIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(centers.nrows());
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);
    // The penalty is assembled AFTER the gauge, not before it (gam#2747). In the
    // `exp(−d/ℓ)` gauge the raw `m × m` Gram was itself PSD, so it could be
    // wrapped first and restricted second; the normalized kernel `ℓ·(e^{−d/ℓ}−1)`
    // is conditionally negative definite instead — `−k` is CND on every constant
    // -curvature space form — and is PSD only ON the sum-to-zero frame, where
    // `zᵀkz = ℓ·zᵀe^{−d/ℓ}z ≻ 0` exactly. Restricting first is therefore not a
    // convenience: it is where the quadratic becomes a penalty. It is also where
    // the cancellation the gauge exists to avoid would otherwise reappear —
    // `zᵀ(ℓK)z` formed from entries that all approach `ℓ` loses the same digits
    // `Kz` does, while `zᵀkz` formed from `expm1` loses none.
    let penalty = ConstructiveQuadratic::try_from_dense_psd(
        symmetrize(&gauge.restrict_penalty(&raw_penalty)),
        "constant-curvature restricted RKHS penalty",
    )?;
    let raw_design =
        constant_curvature_kernel_matrix(data, centers.view(), spec.kappa, length_scale)?;
    let design = gam_linalg::matrix::DesignMatrix::Dense(
        gam_linalg::matrix::DenseDesignMatrix::from(gauge.restrict_design(&raw_design)),
    );
    // Keep the RKHS penalty RAW (the symmetric kernel Gram zᵀKz) with
    // normalization_scale = 1, rather than Frobenius-normalizing it. The Gram's
    // eigenvalues ARE the physical RKHS roughness energies of each coefficient
    // direction: the smoothest functions (the low-degree / degree-1 signal) sit
    // in the genuinely tiny-eigenvalue directions, while wiggly functions sit in
    // the large ones — a spread of many orders of magnitude. Frobenius-
    // normalizing divides the whole operator by ‖·‖_F (dominated by the large
    // wiggly eigenvalues), which compresses that spread and inflates the
    // smallest eigenvalues relative to their natural scale. REML's scale-
    // sensitive λ heuristics then drive a single λ high enough to suppress the
    // wiggly directions and, because the smooth directions are no longer
    // proportionally tiny, over-shrink the recoverable low-degree signal
    // (planted degree-1 sphere harmonic recovered at only R²≈0.84). Keeping the
    // raw physical operator (scale = 1, matching the sphere-harmonic Laplace-
    // Beltrami penalty) lets REML act on true roughness, leaving the smooth
    // signal essentially unpenalized while still shrinking the wiggly tail —
    // raising recovery toward the unconstrained RKHS ceiling. The penalty stays
    // exactly proportional to zᵀKz, so the constrained-kernel-Gram contract is
    // unchanged.
    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        // #1531: the primary here is the RKHS kernel Gram zᵀKz, which is
        // strictly PD / full-rank on distinct centers. It has no unpenalized
        // function subspace, so an explicit second shrinkage coordinate must
        // target the whole coefficient chart. The full identity is therefore
        // intentional for this basis rather than a null-space penalty.
        // The regression test `constant_curvature_gram_is_full_rank_so_identity_is_the_only_double_penalty`
        // locks the full-rank fact that justifies this branch.
        let ridge = Array2::<f64>::eye(design.ncols());
        let (ridge_norm, c_ridge) = normalize_penalty(&ridge);
        candidates.push(PenaltyCandidate {
            matrix: ConstructiveQuadratic::try_from_dense_psd(
                ridge_norm,
                "constant-curvature whole-function ridge",
            )?,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_ridge,
            kronecker_factors: None,
            op: None,
        });
    }
    let filtered = filter_penalty_candidates(candidates)?;
    Ok(BasisBuildResult {
        design,
        affine_offset: None,
        active_penalties: filtered.active,
        dropped_penalties: filtered.dropped,
        metadata: BasisMetadata::ConstantCurvature {
            centers,
            kappa: spec.kappa,
            length_scale,
            constraint_transform: Some(z),
        },
        kronecker_factored: None,
        joint_null_rotation: None,
    })
}

/// Select constant-curvature centers.
///
/// Upper bound on `max‖c‖²`, the largest squared chart radius among the centers
/// `select_constant_curvature_centers` will return for `strategy` on `data` —
/// computed WITHOUT materializing them.
///
/// Every κ bound is denominated in a chart radius, and the set that radius must
/// be taken over is the one the kernel EVALUATES: `K_κ` calls
/// `ConstantCurvature::distance(x, c)` for each (data row, center) pair and
/// `validate_chart_points` checks data **and** centers. Taking the radius over
/// `data` alone is only correct while every center is inside the data hull
/// (gam#2716). Two strategies break that:
///
/// * [`CenterStrategy::UserProvided`] — left verbatim by
///   `select_constant_curvature_centers`, so a center may sit at any radius.
/// * [`CenterStrategy::UniformGrid`] — the Cartesian product of per-axis
///   linspaces over the data's *bounding box*, so a corner center sits at the
///   bounding box corner, radius up to `√d·max‖x‖`, outside the hull for `d ≥ 2`.
///
/// Every other strategy assigns either a data row verbatim (equal-mass leaves,
/// farthest point) or a convex combination of data rows (k-means centroids), so
/// `max‖c‖ ≤ max‖x‖` and this returns exactly the data radius — which is what
/// makes the κ box bit-identical to its pre-#2716 value on every data-driven
/// strategy. The origin-snap in `select_constant_curvature_centers` only ever
/// moves a center toward the origin, so it cannot invalidate an upper bound.
///
/// The match is exhaustive rather than wildcarded: a new strategy has to state
/// its own radius law instead of silently inheriting a wrong one.
pub fn constant_curvature_center_chart_radius2(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    strategy: &CenterStrategy,
) -> f64 {
    match strategy {
        CenterStrategy::Auto(inner) => {
            constant_curvature_center_chart_radius2(data, feature_cols, inner)
        }
        CenterStrategy::DuchonSpectral { knots, .. } => {
            constant_curvature_center_chart_radius2(data, feature_cols, knots)
        }
        CenterStrategy::UserProvided(centers) => {
            let mut max_r2 = 0.0_f64;
            for row in centers.outer_iter() {
                let mut r2 = 0.0_f64;
                for &v in row.iter() {
                    if v.is_finite() {
                        r2 += v * v;
                    }
                }
                max_r2 = max_r2.max(r2);
            }
            max_r2
        }
        CenterStrategy::UniformGrid { .. } => {
            // The grid spans `[min_c, max_c]` per axis, so the extreme center
            // radius is realized at the bounding-box corner whose coordinate is
            // `max(|min_c|, |max_c|)` on every axis simultaneously.
            let mut corner_r2 = 0.0_f64;
            for &c in feature_cols.iter() {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for row in data.outer_iter() {
                    if let Some(&v) = row.get(c)
                        && v.is_finite()
                    {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                if lo.is_finite() && hi.is_finite() {
                    let extreme = lo.abs().max(hi.abs());
                    corner_r2 += extreme * extreme;
                }
            }
            corner_r2
        }
        CenterStrategy::EqualMass { .. }
        | CenterStrategy::EqualMassCovarRepresentative { .. }
        | CenterStrategy::FarthestPoint { .. }
        | CenterStrategy::KMeans { .. } => {
            constant_curvature_data_chart_radius2(data, feature_cols)
        }
    }
}

/// `max‖x‖²` over the term's feature columns of `data`, the other half of the
/// evaluated pair set. Non-finite coordinates are skipped rather than poisoning
/// the maximum (the basis build refuses them separately, by row and name).
pub fn constant_curvature_data_chart_radius2(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
) -> f64 {
    let mut max_r2 = 0.0_f64;
    for row in data.outer_iter() {
        let mut r2 = 0.0_f64;
        for &c in feature_cols.iter() {
            if let Some(&v) = row.get(c)
                && v.is_finite()
            {
                r2 += v * v;
            }
        }
        max_r2 = max_r2.max(r2);
    }
    max_r2
}

/// The stereographic constant-curvature chart has a distinguished pole: the
/// chart origin.  Curvature sign is visible first in the radial geodesic map
/// from that pole (`2 atan(√κ r)/√κ` versus `2 atanh(√|κ| r)/√|κ|`).  A pure
/// farthest-point subset can miss the pole on disk-like clouds, leaving the
/// radial mode to be reconstructed indirectly from boundary centers; then the
/// positive chart's distance compression becomes a generic interpolation
/// advantage and the κ profile is sign-blind.  Keep the user's requested center
/// count, but make data-driven center sets pole-aware by replacing the center
/// closest to the origin with the exact origin.  User-provided centers are left
/// verbatim.
fn select_constant_curvature_centers(
    data: ArrayView2<'_, f64>,
    strategy: &CenterStrategy,
) -> Result<Array2<f64>, BasisError> {
    let mut centers = select_centers_by_strategy(data, strategy)?;
    match strategy {
        CenterStrategy::UserProvided(_) => return Ok(centers),
        CenterStrategy::Auto(inner) => {
            if matches!(inner.as_ref(), CenterStrategy::UserProvided(_)) {
                return Ok(centers);
            }
        }
        CenterStrategy::DuchonSpectral { knots, .. } => {
            if center_strategy_kind(knots) == CenterStrategyKind::UserProvided {
                return Ok(centers);
            }
        }
        // Every data-driven strategy picks its centers from the cloud and can
        // therefore miss the chart origin, so all of them get the pole-aware
        // replacement below. Enumerated rather than wildcarded so a new
        // strategy has to state whether its centers are user-authored.
        CenterStrategy::EqualMass { .. }
        | CenterStrategy::EqualMassCovarRepresentative { .. }
        | CenterStrategy::FarthestPoint { .. }
        | CenterStrategy::KMeans { .. }
        | CenterStrategy::UniformGrid { .. } => {}
    }
    if centers.nrows() == 0 || centers.ncols() == 0 {
        return Ok(centers);
    }
    let (closest, _) = centers
        .outer_iter()
        .enumerate()
        .map(|(i, row)| (i, row.dot(&row)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("centers has at least one row; the empty case returned above");
    for j in 0..centers.ncols() {
        centers[(closest, j)] = 0.0;
    }
    Ok(centers)
}

/// The REALIZED center set the builder will use for `spec` on `data` — the
/// pole-aware selection, not the raw strategy output.
///
/// Public because the outer ψ machinery has to derive the range seed and the
/// range window from the SAME centers the basis is built on: reading them from
/// a re-derived plain strategy would put the ψ box and the objective on two
/// different geometries.
pub fn constant_curvature_realized_centers(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<Array2<f64>, BasisError> {
    let centers = select_constant_curvature_centers(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    Ok(centers)
}

/// Symmetrize `M` in place to `(M + Mᵀ)/2` (the realized penalty is built from
/// the symmetric kernel Gram; the κ-derivative blocks inherit the same exact
/// symmetrization the value path applies before normalization).
pub(crate) fn symmetrize(m: &Array2<f64>) -> Array2<f64> {
    gam_linalg::matrix::symmetrize(m)
}

/// Map a single primary-penalty κ-derivative onto the active penalty list by
/// source — the constant-curvature analogue of the Matérn double-penalty
/// derivative selector. The RKHS Gram is the only κ-moving penalty; the
/// double-penalty ridge `I` is κ-independent, so its derivative is exactly
/// zero. Any other source would mean the basis grew a penalty whose κ-movement
/// is unaccounted for, so we refuse loudly rather than silently drop a term.
pub(crate) fn active_constant_curvature_penalty_derivatives(
    penalties: &[ActivePenalty],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penalties
        .iter()
        .map(|penalty| match &penalty.info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => {
                Ok(Array2::<f64>::zeros(primary_derivative.raw_dim()))
            }
            other => Err(BasisError::InvalidInput(format!(
                "unexpected constant-curvature penalty source in κ-derivative path: {other:?}"
            ))),
        })
        .collect()
}

/// Design and penalty jets of the realized constant-curvature smooth in BOTH
/// outer coordinates, `ψ = (κ, η)` with `η = ln ℓ` (gam#2747).
///
/// The realized construction is `X = K(data,C)·z`, `S = symm(zᵀK(C,C)z)` with
/// `z` and the centers ψ-FIXED, so every block below is the corresponding
/// kernel jet pushed through the same two ψ-fixed linear maps. The RKHS penalty
/// ships raw (`normalization_scale = 1`), so no normalization quotient rule
/// participates; the double-penalty ridge `I` is ψ-independent and its
/// derivative blocks are exactly zero.
#[derive(Clone, Debug)]
pub struct ConstantCurvaturePsiJets {
    /// `∂X/∂κ`.
    pub design_kappa: Array2<f64>,
    /// `∂X/∂η`.
    pub design_eta: Array2<f64>,
    /// `∂²X/∂κ²`.
    pub design_kappa2: Array2<f64>,
    /// `∂²X/∂κ∂η`.
    pub design_kappa_eta: Array2<f64>,
    /// `∂²X/∂η²`.
    pub design_eta2: Array2<f64>,
    /// `∂S_m/∂κ`, one per ACTIVE penalty in the realized basis's order.
    pub penalties_kappa: Vec<Array2<f64>>,
    /// `∂S_m/∂η`.
    pub penalties_eta: Vec<Array2<f64>>,
    /// `∂²S_m/∂κ²`.
    pub penalties_kappa2: Vec<Array2<f64>>,
    /// `∂²S_m/∂κ∂η`.
    pub penalties_kappa_eta: Vec<Array2<f64>>,
    /// `∂²S_m/∂η²`.
    pub penalties_eta2: Vec<Array2<f64>>,
}

/// Full `(κ, η)` derivative bundle for the constant-curvature smooth — the
/// outer-channel hook that lets the signed curvature AND the kernel range join
/// the REML optimization as two design-moving coordinates.
///
/// The curvature coordinate is the **raw, signed κ** (NOT `log κ` as for the
/// Matérn kernel scale): κ = 0 must be a reachable interior point of the
/// `S^d ← ℝ^d → H^d` family, which `log κ` cannot represent. The range
/// coordinate IS logarithmic (`η = ln ℓ`), matching every other kernel smooth in
/// the tree, because `ℓ > 0` and the criterion's curvature in `ℓ` is scale-free.
///
/// Mirrors [`build_constant_curvature_basis`] so the realized design and
/// penalties whose ψ-derivatives this returns are byte-for-byte the same
/// construction the value path produced (same centers, same ℓ, same `z`).
pub fn build_constant_curvature_basis_psi_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<ConstantCurvaturePsiJets, BasisError> {
    if data.ncols() == 0 {
        crate::bail_invalid_basis!("constant-curvature smooth needs at least one feature column");
    }
    if !spec.kappa.is_finite() {
        crate::bail_invalid_basis!("constant-curvature smooth needs a finite kappa");
    }
    validate_chart_points(data, spec.kappa, "data")?;
    // Pole-aware centers, IDENTICAL to `build_constant_curvature_basis` (#1464):
    // this bundle's whole contract is that the design/penalty whose ψ-derivatives
    // it returns are byte-for-byte the SAME construction the value path produced
    // (see the doc above). The value builder replaces the near-origin center with
    // the exact pole for sign identifiability; if this bundle re-derived plain
    // farthest-point centers instead, ∂X/∂ψ would be the derivative of a DIFFERENT
    // design than the frozen one the outer criterion is built on, desyncing the
    // analytic gradient from the finite difference of the cost.
    let centers = select_constant_curvature_centers(data, &spec.center_strategy)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    validate_chart_points(centers.view(), spec.kappa, "centers")?;
    let length_scale = realized_constant_curvature_length_scale(centers.view(), spec.length_scale)?;

    // ψ-fixed constraint transform `z`, resolved exactly as the value builder.
    let z = match &spec.identifiability {
        ConstantCurvatureIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen constant-curvature identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        ConstantCurvatureIdentifiability::CenterSumToZero => {
            let weights = Array1::<f64>::ones(centers.nrows());
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);

    // Design ψ-jets: X = K(data, centers)·z at the SAME ℓ the value path uses,
    // so the derivatives are the kernel ψ-jets right-multiplied by `z`.
    let dc = constant_curvature_kernel_psi_jets(data, centers.view(), spec.kappa, length_scale)?;
    // Penalty ψ-jets: S = symm(zᵀ K(centers,centers) z) at that same ℓ.
    let cc = constant_curvature_kernel_psi_jets(
        centers.view(),
        centers.view(),
        spec.kappa,
        length_scale,
    )?;

    // Align each primary-penalty derivative with the realized active penalty
    // list (primary always; ridge only when double_penalty, and ψ-independent).
    // Rebuild the realized basis once to read `penaltyinfo`.
    let base = build_constant_curvature_basis(data, spec)?;
    let penalty_block = |raw: &Array2<f64>| -> Result<Vec<Array2<f64>>, BasisError> {
        let restricted = symmetrize(&gauge.restrict_penalty(raw));
        active_constant_curvature_penalty_derivatives(&base.active_penalties, &restricted)
    };

    Ok(ConstantCurvaturePsiJets {
        design_kappa: gauge.restrict_design(&dc.d_kappa),
        design_eta: gauge.restrict_design(&dc.d_eta),
        design_kappa2: gauge.restrict_design(&dc.d_kappa2),
        design_kappa_eta: gauge.restrict_design(&dc.d_kappa_eta),
        design_eta2: gauge.restrict_design(&dc.d_eta2),
        penalties_kappa: penalty_block(&cc.d_kappa)?,
        penalties_eta: penalty_block(&cc.d_eta)?,
        penalties_kappa2: penalty_block(&cc.d_kappa2)?,
        penalties_kappa_eta: penalty_block(&cc.d_kappa_eta)?,
        penalties_eta2: penalty_block(&cc.d_eta2)?,
    })
}

/// The κ slice of [`build_constant_curvature_basis_psi_derivatives`], in the
/// generic [`BasisPsiDerivativeBundle`] shape the isotropic spatial ψ-channel
/// consumes for callers that hold the range fixed.
pub fn build_constant_curvature_basis_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &ConstantCurvatureBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let jets = build_constant_curvature_basis_psi_derivatives(data, spec)?;
    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: jets.design_kappa,
            penalties_derivative: jets.penalties_kappa,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: jets.design_kappa2,
            penaltiessecond_derivative: jets.penalties_kappa2,
            implicit_operator: None,
        },
        implicit_operator: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::faer_ndarray::FaerEigh;

    // Diagnostic (#1059 follow-up): show that a κ-FROZEN chart-scale length
    // makes the geodesic-exponential kernel COLLAPSE toward the constant
    // function as κ grows positive (sphere distances compress), which is the
    // degenerate optimum the REML criterion rails to. For a fixed center set we
    // print, per κ, the median geodesic distance and the kernel "spread"
    // 1 − mean(offdiag K). A collapsing kernel ⇒ spread → 0 as κ ↑.
    #[test]
    pub(crate) fn kernel_spread_collapses_with_kappa_at_frozen_length_scale() {
        // 8 centers in a disk of radius 0.45 (inside every κ∈[-2,2] chart).
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        // Frozen ℓ: the κ=0 chart-scale auto rule (median 2‖Δ‖).
        let ell_frozen = realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("fixture centers span a positive pairwise distance");

        let spread = |kappa: f64, ell: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            let m = k.nrows();
            let mut s = 0.0;
            let mut cnt = 0.0;
            for i in 0..m {
                for j in 0..m {
                    if i != j {
                        s += k[(i, j)];
                        cnt += 1.0;
                    }
                }
            }
            1.0 - s / cnt
        };

        let s_neg = spread(-2.0, ell_frozen);
        let s_zero = spread(0.0, ell_frozen);
        let s_pos = spread(2.0, ell_frozen);
        eprintln!(
            "[κ-collapse] frozen ℓ={ell_frozen:.4}: spread κ=-2 {s_neg:.4} | κ=0 {s_zero:.4} | κ=+2 {s_pos:.4}"
        );

        // The degenerate signature: positive κ collapses the kernel toward the
        // constant (spread shrinks), so the criterion can buy cheap EDF by
        // pushing κ up — this is the unidentifiability we are fixing.
        assert!(
            s_pos < s_zero && s_zero < s_neg,
            "expected kernel spread to shrink with κ at frozen ℓ: κ=-2 {s_neg} κ=0 {s_zero} κ=+2 {s_pos}"
        );

        // Decompose the κ-monotone REML Occam term. The realized penalty is the
        // Frobenius-normalized centered Gram S~ = S_raw/‖S_raw‖_F with
        // S_raw = symm(zᵀ K z); the REML evidence carries +½ log|S~|_+ over its
        // range. Print log det₊(S~) per κ to see whether the penalty-normalization
        // Occam term (not just the modest kernel-spread shift) is what rails κ.
        let weights = Array1::<f64>::ones(centers.nrows());
        let z = weighted_coefficient_sum_to_zero_transform(weights.view())
            .expect("fixture weights are positive, so the sum-to-zero transform exists");
        let logdet_norm_penalty = |kappa: f64, ell: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            let s_raw = symmetrize(&z.t().dot(&k).dot(&z));
            let (s_norm, _c) = normalize_penalty(&s_raw);
            let sym = symmetrize(&s_norm);
            let (evals, _v) = FaerEigh::eigh(&sym, faer::Side::Lower)
                .expect("the fixture Gram is symmetric, so eigh converges");
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let tol = max * 1e-9;
            evals
                .iter()
                .filter(|&&e| e > tol)
                .map(|&e| e.ln())
                .sum::<f64>()
        };
        let l_neg = logdet_norm_penalty(-2.0, ell_frozen);
        let l_zero = logdet_norm_penalty(0.0, ell_frozen);
        let l_pos = logdet_norm_penalty(2.0, ell_frozen);
        eprintln!(
            "[κ-collapse] log|S~|_+ (frozen ℓ): κ=-2 {l_neg:.4} | κ=0 {l_zero:.4} | κ=+2 {l_pos:.4}"
        );

        // GEODESIC-SCALED ℓ removes the κ-dependence of the kernel resolution:
        // set ℓ(κ) = median geodesic distance d_κ among centers. Then the spread
        // should be ~κ-invariant. Print the geodesic-ℓ spread per κ.
        let geo_median_ell = |kappa: f64| -> f64 {
            let m = centers.nrows();
            let manifold = ConstantCurvature::new(centers.ncols(), kappa);
            let mut dists = Vec::with_capacity(m * (m - 1) / 2);
            for i in 0..m {
                for j in (i + 1)..m {
                    dists.push(
                        manifold
                            .distance(centers.row(i), centers.row(j))
                            .expect("fixture centers lie on the manifold"),
                    );
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).expect("pairwise distances are finite"));
            dists[dists.len() / 2]
        };
        let gs_neg = spread(-2.0, geo_median_ell(-2.0));
        let gs_zero = spread(0.0, geo_median_ell(0.0));
        let gs_pos = spread(2.0, geo_median_ell(2.0));
        let gl_neg = logdet_norm_penalty(-2.0, geo_median_ell(-2.0));
        let gl_zero = logdet_norm_penalty(0.0, geo_median_ell(0.0));
        let gl_pos = logdet_norm_penalty(2.0, geo_median_ell(2.0));
        eprintln!(
            "[κ-collapse] geodesic ℓ: spread κ=-2 {gs_neg:.4} | κ=0 {gs_zero:.4} | κ=+2 {gs_pos:.4}"
        );
        eprintln!(
            "[κ-collapse] geodesic ℓ: log|S~|_+ κ=-2 {gl_neg:.4} | κ=0 {gl_zero:.4} | κ=+2 {gl_pos:.4}"
        );

        // CANDIDATE FIX: freeze the Frobenius normalization constant at κ=0 so
        // the REML Occam term log|S_λ|_+ carries only the GENUINE roughness
        // spectrum log|S_raw(κ)|_+ (minus a κ-independent constant), not the
        // spurious −r·log‖S_raw(κ)‖_F leak. Compare:
        //   (a) log|S_raw(κ)|_+        (un-normalized, true roughness Occam term)
        //   (b) log|S_raw(κ)/c₀|_+     (frozen-c₀ normalization at κ=0)
        // Both should be κ-IDENTIFYING (a real interior optimum), not monotone.
        let logdet_raw = |kappa: f64, ell: f64, c0: f64| -> f64 {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            let s_raw = symmetrize(&z.t().dot(&k).dot(&z));
            let scaled = s_raw.mapv(|v| v / c0);
            let (evals, _v) = FaerEigh::eigh(&scaled, faer::Side::Lower)
                .expect("the fixture Gram is symmetric, so eigh converges");
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let tol = max * 1e-9;
            evals
                .iter()
                .filter(|&&e| e > tol)
                .map(|&e| e.ln())
                .sum::<f64>()
        };
        // c₀ = ‖S_raw(κ=0)‖_F at frozen ℓ.
        let k0 = constant_curvature_kernel_matrix(centers.view(), centers.view(), 0.0, ell_frozen)
            .expect("fixture centers are distinct and the length scale is positive");
        let s_raw0 = symmetrize(&z.t().dot(&k0).dot(&z));
        let c0 = s_raw0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let r_neg = logdet_raw(-2.0, ell_frozen, c0);
        let r_zero = logdet_raw(0.0, ell_frozen, c0);
        let r_pos = logdet_raw(2.0, ell_frozen, c0);
        eprintln!(
            "[κ-collapse] frozen-c₀ log|S_raw/c₀|_+ (frozen ℓ): κ=-2 {r_neg:.4} | κ=0 {r_zero:.4} | κ=+2 {r_pos:.4}"
        );
        // Finer grid to see the shape of the un-normalized roughness Occam term.
        eprint!("[κ-collapse] frozen-c₀ grid:");
        for kk in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            eprint!(" κ={kk}:{:.4}", logdet_raw(kk, ell_frozen, c0));
        }
        eprintln!();
    }

    /// 8 data rows + 8 centers inside a disk of radius < 0.5 (valid in every
    /// κ ∈ [−3, 3] chart). Data ≠ centers so the data→center scale is nontrivial.
    pub(crate) fn oracle_disk_design_centers() -> (Array2<f64>, Array2<f64>) {
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        // Deterministic pseudo-random data on a slightly wider disk.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // map to (−0.42, 0.42)
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.84
        };
        let n = 60usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[(i, 0)] = next();
            data[(i, 1)] = next();
        }
        (data, centers)
    }

    /// Degenerate geometries must refuse the range box by name rather than
    /// returning a collapsed or infinite one: a box is a claim about where the
    /// design is evaluable, and there is no such claim to make when every
    /// evaluated pair is coincident.
    #[test]
    pub(crate) fn range_box_refuses_a_degenerate_geometry() {
        let coincident = ndarray::array![[0.3_f64, -0.2], [0.3, -0.2]];
        let error = constant_curvature_length_scale_bounds(coincident.view(), coincident.view())
            .expect_err("a cloud with no positive pairwise distance has no range box");
        let message = format!("{error}");
        assert!(
            message.contains("no") && message.contains("positive"),
            "the refusal must name what is missing; got {message}"
        );
        // TWO distinct points are enough. With one distance both ends are the
        // SAME `d`, so the box's width is a property of the FORMAT alone:
        // `(d/(2√ε)) / (d/(½ln(1/ε))) = ½ln(1/ε)/(2√ε) ≈ 6.0e8`.
        let pair = ndarray::array![[0.0_f64, 0.0], [0.2, 0.0]];
        let (lo, hi) = constant_curvature_length_scale_bounds(pair.view(), pair.view())
            .expect("one positive pairwise distance is enough");
        let format_width = (0.5 * -f64::EPSILON.ln()) / (2.0 * f64::EPSILON.sqrt());
        assert!(
            lo > 0.0 && (hi / lo - format_width).abs() <= 1.0e-9 * format_width,
            "a one-distance geometry's box width is the format's, {format_width}; got \
             [{lo}, {hi}] with ratio {}",
            hi / lo
        );
    }

    /// Every entry of the `(κ, η)` kernel tower must match a central finite
    /// difference of the value (first order) and of the first derivatives
    /// (second order), on BOTH branches and across the κ = 0 series/closed-form
    /// seam. This is the gate the outer ψ gradient and the stationarity
    /// certificate stand on: five blocks, so five FD comparisons, and the cross
    /// term is differenced along the OTHER coordinate than the one it names so
    /// a symmetric-by-construction bug cannot hide in it.
    #[test]
    pub(crate) fn kernel_psi_jets_match_central_differences_in_both_coordinates() {
        let (data, centers) = oracle_disk_design_centers();
        let ell0 = realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("fixture centers span a positive pairwise distance");
        let eta0 = ell0.ln();
        let at = |kappa: f64, eta: f64| {
            constant_curvature_kernel_psi_jets(data.view(), centers.view(), kappa, eta.exp())
                .expect("the fixture disk is inside every probed chart")
        };
        let rel = |exact: &Array2<f64>, fd: &Array2<f64>| -> f64 {
            let mut err = 0.0_f64;
            let mut scale = 0.0_f64;
            for (&a, &b) in exact.iter().zip(fd.iter()) {
                err = err.max((a - b).abs());
                scale = scale.max(a.abs()).max(b.abs());
            }
            err / scale.max(1.0)
        };
        let h = 1.0e-5_f64;
        for &kappa in &[-1.5_f64, -0.5, -1e-7, 0.0, 1e-7, 0.8, 1.7] {
            for &eta in &[eta0 - 0.7, eta0, eta0 + 0.7] {
                let jets = at(kappa, eta);
                let kp = at(kappa + h, eta);
                let km = at(kappa - h, eta);
                let ep = at(kappa, eta + h);
                let em = at(kappa, eta - h);
                let central = |plus: &Array2<f64>, minus: &Array2<f64>| -> Array2<f64> {
                    (plus - minus) / (2.0 * h)
                };
                let checks = [
                    ("∂K/∂κ", &jets.d_kappa, central(&kp.value, &km.value), 1e-6),
                    ("∂K/∂η", &jets.d_eta, central(&ep.value, &em.value), 1e-6),
                    (
                        "∂²K/∂κ²",
                        &jets.d_kappa2,
                        central(&kp.d_kappa, &km.d_kappa),
                        1e-5,
                    ),
                    (
                        // Differenced along η of ∂K/∂κ — the opposite order from
                        // the closed form's derivation, so equality is a real
                        // check of the mixed partial rather than a tautology.
                        "∂²K/∂κ∂η",
                        &jets.d_kappa_eta,
                        central(&ep.d_kappa, &em.d_kappa),
                        1e-5,
                    ),
                    ("∂²K/∂η²", &jets.d_eta2, central(&ep.d_eta, &em.d_eta), 1e-5),
                ];
                for (label, exact, fd, tol) in checks {
                    let error = rel(exact, &fd);
                    assert!(
                        error < tol,
                        "κ={kappa} η={eta}: {label} disagrees with its central difference: rel={error:.6e}"
                    );
                }
            }
        }
    }

    /// The κ = 0 build is the plain Euclidean exponential smooth in the doubled
    /// chart gauge, and the design and the penalty are two blocks of ONE Gram at
    /// ONE range at every κ (gam#2747). Reconstructing either at the metadata's
    /// own `length_scale` must reproduce the realized block exactly — which is
    /// the property the fill-invariant `L(κ)` / `L_S(κ)` pair broke, and the
    /// reason the penalty was not the RKHS roughness of its own design.
    #[test]
    pub(crate) fn design_and_penalty_are_one_gram_at_one_range() {
        let (data, centers) = oracle_disk_design_centers();
        for kappa in [-1.2_f64, -0.4, 0.0, 0.4, 1.2] {
            let spec = ConstantCurvatureBasisSpec {
                center_strategy: CenterStrategy::UserProvided(centers.clone()),
                kappa,
                length_scale: 1.3,
                ..Default::default()
            };
            let built = build_constant_curvature_basis(data.view(), &spec).expect("build");
            let BasisMetadata::ConstantCurvature {
                length_scale,
                constraint_transform,
                ..
            } = &built.metadata
            else {
                panic!("expected ConstantCurvature metadata");
            };
            assert_eq!(
                *length_scale, 1.3,
                "the realized range is the spec's range, not a κ-remapped one"
            );
            let z = constraint_transform.as_ref().expect("constraint transform");
            let k_dc =
                constant_curvature_kernel_matrix(data.view(), centers.view(), kappa, *length_scale)
                    .expect("design kernel");
            let k_cc = constant_curvature_kernel_matrix(
                centers.view(),
                centers.view(),
                kappa,
                *length_scale,
            )
            .expect("penalty kernel");
            let design = built.design.to_dense();
            for (a, b) in design.iter().zip(k_dc.dot(z).iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "κ={kappa}: design != K(ℓ)·z ({a} vs {b})"
                );
            }
            let gram = symmetrize(&z.t().dot(&k_cc).dot(z));
            let primary = built
                .active_penalties
                .iter()
                .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
                .expect("primary RKHS penalty");
            for (a, b) in gram.iter().zip(primary.matrix.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "κ={kappa}: penalty != zᵀK(ℓ)z at the SAME ℓ ({a} vs {b})"
                );
            }
        }
    }

    /// The range box is a CONDITIONING wall, not a statistical one: it must
    /// contain the coarse end of the geometry's scale span with room above,
    /// because the criterion's own minimum provably leaves that span
    /// (gam#2747). And it must sit where the GRAM stops being resolvable, not
    /// where the kernel stops being representable — the criterion evaluates a
    /// Cholesky of `H = XᵀX + λS`, and the Gram squares the design's dynamic
    /// range.
    #[test]
    pub(crate) fn range_box_is_the_gram_conditioning_wall_and_contains_the_scale_span() {
        let (data, centers) = oracle_disk_design_centers();
        let (span_lo, span_hi) =
            constant_curvature_evaluated_scale_span(data.view(), centers.view())
                .expect("the fixture carries positive evaluated distances");
        let (lo, hi) = constant_curvature_length_scale_bounds(data.view(), centers.view())
            .expect("box is derivable");
        let seed = realized_constant_curvature_length_scale(centers.view(), 0.0).expect("seed");
        assert!(
            span_hi < hi && lo < seed && seed < hi,
            "the box [{lo}, {hi}] must contain the coarse end of the evaluated span \
             [{span_lo}, {span_hi}] and the auto seed {seed}"
        );
        // AT `lo` the Gram's dynamic range is exactly one ε — the last one a
        // double-precision Cholesky can resolve; half an ℓ below, the Gram's
        // far entries have rounded into the diagonal.
        // "Resolvable" stated operationally, so the boundary case cannot turn
        // on which way a comparison against ε rounds: the quantity must still
        // PERTURB 1, and must stop perturbing it just outside the wall.
        let gram_range = |ell: f64| (-2.0 * span_hi / ell).exp();
        assert!(
            1.0 + gram_range(lo) != 1.0,
            "at ℓ_lo the Gram's far entries must still perturb the diagonal; got {}",
            gram_range(lo)
        );
        assert!(
            1.0 + gram_range(lo / 2.0) == 1.0,
            "half an ℓ_lo below, the Gram's far entries must round into the diagonal"
        );
        // AT `hi` the far end is a MODEL statement, not a numerical one
        // (gam#2747): the widest evaluated pair's kernel must have come within
        // `√ε` of its `ℓ → ∞` limit `−d`, and must NOT yet be that close an
        // order of magnitude below. Truncating the chart there is what makes an
        // estimate at the top mean "the kernel is the geodesic distance"
        // instead of "the box ended here".
        let limit_departure = |ell: f64| {
            let k = constant_curvature_kernel_scalar(span_hi, ell);
            ((k + span_hi) / span_hi).abs()
        };
        let root_eps = f64::EPSILON.sqrt();
        assert!(
            limit_departure(hi) <= root_eps * 1.01,
            "at ℓ_hi the widest evaluated pair must BE its own limit to √ε; departure {:.3e} \
             against {root_eps:.3e}",
            limit_departure(hi)
        );
        assert!(
            limit_departure(hi / 10.0) > root_eps,
            "an order of magnitude below ℓ_hi the limit must still be resolvable, or the \
             chart is being truncated earlier than the model justifies; departure {:.3e}",
            limit_departure(hi / 10.0)
        );
        // And the old top must be gone: `d_min/√ε` is where the RETIRED `exp`
        // gauge lost the design to cancellation, which is a property of a gauge
        // this basis no longer uses.
        let retired_top = span_lo / root_eps;
        assert!(
            (hi - retired_top).abs() > 0.1 * retired_top,
            "ℓ_hi must no longer be the retired cancellation wall {retired_top:.4e}; got {hi:.4e}"
        );
    }

    #[test]
    fn constant_curvature_gram_is_full_rank_so_identity_is_the_only_double_penalty() {
        // Centers inside every κ chart, several curvatures spanning sign.
        let centers = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        let weights = Array1::<f64>::ones(centers.nrows());
        let z = weighted_coefficient_sum_to_zero_transform(weights.view())
            .expect("fixture weights are positive, so the sum-to-zero transform exists");
        // Frozen auto length scale (the κ=0 chart-scale rule; 0.0 ⇒ auto), reused
        // across κ so the full-rank check is on the same resolution the basis uses.
        let ell = realized_constant_curvature_length_scale(centers.view(), 0.0)
            .expect("fixture centers span a positive pairwise distance");

        for &kappa in &[-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            let k = constant_curvature_kernel_matrix(centers.view(), centers.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            // Primary penalty exactly as the basis builder forms it: symmetrized
            // gauge-restricted kernel Gram.
            let raw = symmetrize(&z.t().dot(&k).dot(&z));

            // (a) The primary is full-rank PD: smallest eigenvalue is strictly
            // positive (well above the spectral tolerance), so there is no null
            // space for a Marra-Wood ridge to shrink.
            let (evals, _v) = FaerEigh::eigh(&raw, faer::Side::Lower)
                .expect("the fixture Gram is symmetric, so eigh converges");
            let max = evals.iter().cloned().fold(0.0_f64, f64::max);
            let min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                max > 0.0 && min > max * 1e-9,
                "constant-curvature Gram must be full-rank PD at κ={kappa}: \
                 min eig {min:e}, max eig {max:e}"
            );
        }
    }

    /// The geodesic-exponential kernel is Matérn-½, so it has a CUSP at every
    /// center — and that is what blocks a constant-curvature SAE ATOM.
    ///
    /// The distinction is between the two ways a basis gets used. A GAM smooth
    /// evaluates its design at FIXED data rows and never differentiates with
    /// respect to the input, so a cusp in `x` is invisible to it and this kernel
    /// is exactly right there. A SAE atom's latent coordinate is a FITTED
    /// parameter: its solve consumes `∂Φ/∂t` and `∂²Φ/∂t²` (the
    /// `SaeBasisSecondJet` contract feeding the Newton/Schur assembly). At a
    /// center the first derivative is direction-dependent and the second is
    /// unbounded, so a Newton step there has no curvature to trust.
    ///
    /// This measures it rather than citing it: the central second difference of
    /// `K` along a fixed direction through a center is compared against the same
    /// quantity a smooth distance away. If the kernel were `C²` both would
    /// converge; instead the at-center value grows like `1/h` while the offset
    /// one converges, and the test asserts a decade of separation.
    ///
    /// The kernel is not at fault. Geodesic distance is conditionally negative
    /// definite on all three space forms, so `exp(−c·d_κ)` is PD for EVERY κ —
    /// which is precisely why it was chosen, and smoother radial families lose
    /// that guarantee on spheres. The obstruction is real and belongs to the
    /// atom design, not to this module.
    #[test]
    fn geodesic_exponential_kernel_has_unbounded_curvature_at_a_center() {
        let kappa = 0.0_f64;
        let ell = 1.0_f64;
        let center = ndarray::arr2(&[[0.0_f64, 0.0]]);

        // Second difference of `h -> K(center + h·e_x, center)` at the center,
        // and at a point a fixed distance away where the kernel is smooth.
        let curvature_at = |base: [f64; 2], h: f64| -> f64 {
            let probe = ndarray::arr2(&[
                [base[0] - h, base[1]],
                [base[0], base[1]],
                [base[0] + h, base[1]],
            ]);
            let k = constant_curvature_kernel_matrix(probe.view(), center.view(), kappa, ell)
                .expect("fixture centers are distinct and the length scale is positive");
            (k[[0, 0]] - 2.0 * k[[1, 0]] + k[[2, 0]]) / (h * h)
        };

        let mut at_center = Vec::new();
        let mut off_center = Vec::new();
        for &h in &[1.0e-2_f64, 1.0e-3, 1.0e-4] {
            at_center.push(curvature_at([0.0, 0.0], h).abs());
            off_center.push(curvature_at([0.5, 0.0], h).abs());
        }

        // Away from the center the second difference converges: successive
        // refinements agree.
        assert!(
            (off_center[2] - off_center[1]).abs() <= 1.0e-3 * off_center[1].max(1.0),
            "the kernel must be C² away from its centers; got {off_center:?}"
        );
        // At the center it diverges like 1/h: each 10x refinement multiplies it
        // by ~10, so three decades of h separate by ~100x.
        assert!(
            at_center[2] > 10.0 * at_center[0],
            "a Matérn-½ cusp must make the second difference diverge as h -> 0; \
             got {at_center:?}"
        );
        assert!(
            at_center[2] > 100.0 * off_center[2],
            "the at-center curvature must dwarf the smooth-region curvature; \
             at-center {:?} vs off-center {:?}",
            at_center[2],
            off_center[2]
        );
    }

    /// #2458 — the second κ-derivatives shipped WITHOUT a reader.
    ///
    /// `build_constant_curvature_basis_kappa_derivatives` returns a
    /// `BasisPsiDerivativeBundle` whose `.second` carries the κ-second
    /// derivatives of the design and of each penalty block. Those are consumed
    /// in production (`spatial_optimization.rs` destructures and rotates them
    /// for the spatial ψ path), but nothing anywhere pinned their VALUES:
    /// grepping for readers of `designsecond_derivative` /
    /// `penaltiessecond_derivative` outside the destructuring sites returns
    /// nothing. A second derivative that is shipped and consumed but never
    /// checked is exactly the input #2458 proposes to build a stationarity
    /// CERTIFICATE on, and a wrong certificate is worse than an honestly
    /// missing one.
    ///
    /// This differences the ANALYTIC FIRST derivative, which the κ-gradient
    /// path already exercises end to end, so a failure localizes to the
    /// second-order construction rather than to the basis itself.
    ///
    /// The bound is the central-difference error budget, not a tuned number.
    /// Truncation is order h^2 times the third derivative and roundoff is
    /// order eps times the first derivative over h, so at h = 1e-4 on an
    /// order-one chart both sit near 1e-8 relative. Asserting 1e-6 leaves two
    /// orders of headroom while still failing a missing or mis-scaled term —
    /// which is scale-invariant and would NOT shrink with h, hence the
    /// h-halving arm below.
    #[test]
    fn kappa_second_derivatives_match_a_central_difference_of_the_first_2458() {
        let data = ndarray::array![
            [0.10, 0.05],
            [-0.20, 0.15],
            [0.30, -0.10],
            [-0.05, -0.25],
            [0.22, 0.20],
            [-0.30, -0.05],
            [0.05, 0.30],
            [-0.15, 0.10],
        ];
        let spec = ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
            ..Default::default()
        };
        let kappa0 = 0.35_f64;

        let first_at = |kappa: f64| {
            let mut probe = spec.clone();
            probe.kappa = kappa;
            let bundle = build_constant_curvature_basis_kappa_derivatives(data.view(), &probe)
                .expect("fixture points lie inside the chart for every probed kappa");
            (
                bundle.first.design_derivative,
                bundle.first.penalties_derivative,
            )
        };

        let mut exact_spec = spec.clone();
        exact_spec.kappa = kappa0;
        let analytic = build_constant_curvature_basis_kappa_derivatives(data.view(), &exact_spec)
            .expect("fixture points lie inside the chart at kappa0");
        let design_second = analytic.second.designsecond_derivative;
        let penalty_second = analytic.second.penaltiessecond_derivative;

        let max_rel_error_at = |h: f64| -> (f64, f64) {
            let (x_plus, s_plus) = first_at(kappa0 + h);
            let (x_minus, s_minus) = first_at(kappa0 - h);

            let mut design_error = 0.0_f64;
            let mut design_scale = 0.0_f64;
            for ((&plus, &minus), &exact) in
                x_plus.iter().zip(x_minus.iter()).zip(design_second.iter())
            {
                let fd = (plus - minus) / (2.0 * h);
                design_error = design_error.max((fd - exact).abs());
                design_scale = design_scale.max(exact.abs()).max(fd.abs());
            }

            assert_eq!(
                s_plus.len(),
                penalty_second.len(),
                "penalty block count must not depend on kappa"
            );
            let mut penalty_error = 0.0_f64;
            let mut penalty_scale = 0.0_f64;
            for ((block_plus, block_minus), block_exact) in
                s_plus.iter().zip(s_minus.iter()).zip(penalty_second.iter())
            {
                for ((&plus, &minus), &exact) in block_plus
                    .iter()
                    .zip(block_minus.iter())
                    .zip(block_exact.iter())
                {
                    let fd = (plus - minus) / (2.0 * h);
                    penalty_error = penalty_error.max((fd - exact).abs());
                    penalty_scale = penalty_scale.max(exact.abs()).max(fd.abs());
                }
            }
            (
                design_error / design_scale.max(1.0),
                penalty_error / penalty_scale.max(1.0),
            )
        };

        let h = 1.0e-4_f64;
        let (design_rel, penalty_rel) = max_rel_error_at(h);
        eprintln!(
            "[2458-second-fd] h={h:.1e}: design rel={design_rel:.3e} penalty rel={penalty_rel:.3e}"
        );
        assert!(
            design_rel < 1.0e-6,
            "d2X/dkappa2 disagrees with a central difference of dX/dkappa: rel={design_rel:.6e}"
        );
        assert!(
            penalty_rel < 1.0e-6,
            "d2S/dkappa2 disagrees with a central difference of dS/dkappa: rel={penalty_rel:.6e}"
        );

        // A MISSING term is scale-invariant: it does not shrink when h does.
        // Halving h must not inflate the disagreement, which it would if the
        // residual were a genuine missing contribution rather than truncation.
        let (design_rel_half, penalty_rel_half) = max_rel_error_at(0.5 * h);
        eprintln!(
            "[2458-second-fd] h={:.1e}: design rel={design_rel_half:.3e} penalty rel={penalty_rel_half:.3e}",
            0.5 * h
        );
        assert!(
            design_rel_half <= design_rel.max(1.0e-9) * 2.0,
            "halving h must not inflate the design disagreement (missing-term signature): \
             {design_rel:.6e} -> {design_rel_half:.6e}"
        );
        assert!(
            penalty_rel_half <= penalty_rel.max(1.0e-9) * 2.0,
            "halving h must not inflate the penalty disagreement (missing-term signature): \
             {penalty_rel:.6e} -> {penalty_rel_half:.6e}"
        );
    }

    /// The contrast gauge is the SAME model as `exp(−d_κ/ℓ)` — and the `exp`
    /// gauge stops being able to express it (gam#2747).
    ///
    /// The realized blocks are only ever `K z` and `zᵀ K z`, and `z` annihilates
    /// constants, so `K` and `ℓ·(K − 1)` differ by a per-row constant and a
    /// positive scale: the second block must equal `ℓ ×` the first, EXACTLY, in
    /// exact arithmetic, at every κ and every ℓ. That equality is the licence
    /// for the change of gauge, so it is asserted first and at a tolerance that
    /// leaves no room (1e-12 relative, mid-box).
    ///
    /// It is then asserted to FAIL — by a factor of `10⁵` in relative terms — at
    /// the top of the derived box, because that is the finding: the `exp` gauge
    /// forms `K − 1` by subtracting two numbers that agree to `log₁₀(ℓ/d)`
    /// digits, so past `ℓ ≈ 10⁶` the design it emits is not the model's. Only
    /// one of the two is still right there, and the reference direction says
    /// which: `expm1` is exact by construction and `exp(x) − 1` is not.
    #[test]
    fn the_contrast_gauge_is_the_same_model_and_the_exp_gauge_loses_it_2747() {
        let (data, centers) = oracle_disk_design_centers();
        let manifold = ConstantCurvature::new(2, 0.6);
        let exp_gauge_design = |ell: f64| -> Array2<f64> {
            let mut raw = Array2::<f64>::zeros((data.nrows(), centers.nrows()));
            for i in 0..data.nrows() {
                for j in 0..centers.nrows() {
                    let d = manifold
                        .distance(data.row(i), centers.row(j))
                        .expect("fixture disk is inside the κ = 0.6 chart");
                    raw[(i, j)] = (-d / ell).exp();
                }
            }
            let weights = Array1::<f64>::ones(centers.nrows());
            let z = weighted_coefficient_sum_to_zero_transform(weights.view())
                .expect("uniform sum-to-zero frame");
            raw.dot(&z).mapv(|value| value * ell)
        };
        let realized_design = |ell: f64| -> Array2<f64> {
            let spec = ConstantCurvatureBasisSpec {
                center_strategy: CenterStrategy::UserProvided(centers.clone()),
                kappa: 0.6,
                length_scale: ell,
                ..Default::default()
            };
            build_constant_curvature_basis(data.view(), &spec)
                .expect("build")
                .design
                .to_dense()
        };
        let disagreement = |ell: f64| -> f64 {
            let a = realized_design(ell);
            let b = exp_gauge_design(ell);
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                num += (x - y) * (x - y);
                den += x * x;
            }
            (num / den).sqrt()
        };
        for ell in [0.25_f64, 1.0, 4.0, 16.0] {
            let error = disagreement(ell);
            assert!(
                error < 1.0e-12,
                "the two gauges are one model: at ℓ={ell} they differ by {error:.3e}"
            );
        }
        // At the top of the derived box the `exp` gauge forms `K − 1` from
        // entries that agree to `d/ℓ`, so its relative error is `≈ ε·ℓ/d` — a
        // DERIVED prediction, bracketed by the geometry's own evaluated scale
        // span, not a threshold. The measurement has to sit inside that bracket:
        // above it would mean some other error source, below it would mean the
        // cancellation is not happening and the finding is wrong.
        let (_, ell_hi) = constant_curvature_length_scale_bounds(data.view(), centers.view())
            .expect("the fixture geometry has a range box");
        let (d_min, d_max) = constant_curvature_evaluated_scale_span(data.view(), centers.view())
            .expect("the fixture geometry has an evaluated scale span");
        let mildest = f64::EPSILON * ell_hi / d_max;
        let worst = f64::EPSILON * ell_hi / d_min;
        let at_wall = disagreement(ell_hi);
        assert!(
            at_wall >= mildest && at_wall <= worst,
            "at the box top ℓ={ell_hi:.4e} the `exp` gauge's error must be the cancellation \
             `ε·ℓ/d`, bracketed by the evaluated span [{d_min:.4e}, {d_max:.4e}] as \
             [{mildest:.3e}, {worst:.3e}]; measured {at_wall:.3e}"
        );
        assert!(
            at_wall > 1.0e3 * disagreement(16.0),
            "and it must GROW with the range: {at_wall:.3e} at the box top against \
             {:.3e} mid-box",
            disagreement(16.0)
        );
    }

    /// The `ℓ → ∞` face of the range coordinate is the geodesic-DISTANCE kernel,
    /// reached exactly rather than approached (gam#2747).
    ///
    /// `−d_κ` is conditionally positive definite on every constant-curvature
    /// space form, so `X = −D(data,C)z` and `S = −zᵀD(C,C)z` are an ordinary
    /// non-degenerate smooth. In the contrast gauge the realized blocks converge
    /// to it — with the `1/ℓ` collapse gone there is something LEFT to converge
    /// to — and the penalty stays strictly positive definite all the way, which
    /// is what makes the limit a point of the parameter space rather than a
    /// wall. `20bde053f` reverted the free-range enrollment for want of "a
    /// derived stopping rule for a criterion that converges rather than turning
    /// over"; a convergent criterion whose limit is a MODEL does not need one.
    #[test]
    fn the_range_limit_is_the_geodesic_distance_kernel_2747() {
        let (data, centers) = oracle_disk_design_centers();
        let weights = Array1::<f64>::ones(centers.nrows());
        let z = weighted_coefficient_sum_to_zero_transform(weights.view())
            .expect("uniform sum-to-zero frame");
        for kappa in [-1.1_f64, 0.0, 0.9] {
            let manifold = ConstantCurvature::new(2, kappa);
            let distances = |a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>| -> Array2<f64> {
                let mut out = Array2::<f64>::zeros((a.nrows(), b.nrows()));
                for i in 0..a.nrows() {
                    for j in 0..b.nrows() {
                        out[(i, j)] = manifold
                            .distance(a.row(i), b.row(j))
                            .expect("fixture disk is inside every probed chart");
                    }
                }
                out
            };
            let limit_design = distances(data.view(), centers.view())
                .mapv(|d| -d)
                .dot(&z);
            let limit_penalty = symmetrize(
                &z.t()
                    .dot(&distances(centers.view(), centers.view()).mapv(|d| -d))
                    .dot(&z),
            );
            let mut previous = f64::INFINITY;
            for ell in [1.0e3_f64, 1.0e5, 1.0e7, 1.0e9] {
                let spec = ConstantCurvatureBasisSpec {
                    center_strategy: CenterStrategy::UserProvided(centers.clone()),
                    kappa,
                    length_scale: ell,
                    ..Default::default()
                };
                let built = build_constant_curvature_basis(data.view(), &spec).expect("build");
                let design = built.design.to_dense();
                let penalty = built.active_penalties[0].matrix.clone();
                let gap = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
                    let mut num = 0.0_f64;
                    let mut den = 0.0_f64;
                    for (&x, &y) in a.iter().zip(b.iter()) {
                        num += (x - y) * (x - y);
                        den += y * y;
                    }
                    (num / den).sqrt()
                };
                let design_gap = gap(&design, &limit_design);
                let penalty_gap = gap(&penalty, &limit_penalty);
                // First order in `1/ℓ`, so each two decades must buy two orders.
                assert!(
                    design_gap < previous / 50.0,
                    "κ={kappa}: the design must converge to the distance kernel; \
                     at ℓ={ell:.0e} the gap is {design_gap:.3e} against {previous:.3e} before"
                );
                previous = design_gap;
                assert!(
                    penalty_gap < 1.0e-2,
                    "κ={kappa}: the penalty must converge too; gap {penalty_gap:.3e} at ℓ={ell:.0e}"
                );
                let (evals, _) = FaerEigh::eigh(&penalty, faer::Side::Lower).expect("penalty spectrum");
                let smallest = evals.iter().cloned().fold(f64::INFINITY, f64::min);
                let largest = evals.iter().cloned().fold(0.0_f64, f64::max);
                assert!(
                    smallest > 1.0e-10 * largest,
                    "κ={kappa}: the restricted Gram must stay strictly PD at ℓ={ell:.0e}; \
                     spectrum spans [{smallest:.3e}, {largest:.3e}]"
                );
            }
            assert!(
                previous < 1.0e-8,
                "κ={kappa}: at ℓ=1e9 the design must BE the distance kernel; gap {previous:.3e}"
            );
        }
    }
}
