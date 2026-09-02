//! Curvature-as-an-estimand: the inference layer on top of the κ-jets.
//!
//! #944 stage 3. The κ-jets (`distance_kappa_jet` / `log_map_kappa_jet` /
//! `exp_map_kappa_jet`) and the analytic Jacobi-field `exp_map_vjp` for the
//! [`ConstantCurvature`] chart are landed and
//! FD-gated. This module turns the fitted curvature `κ̂` from "we chose
//! hyperbolic space" into a reported estimate with a confidence interval and a
//! likelihood-ratio test of flatness — and exposes the κ-derivative of the
//! design-moving geometry quantity (geodesic normal coordinates) as the clean
//! seam the outer ψ-channel calls when κ joins the LAML/REML optimisation as one
//! signed design-moving coordinate.
//!
//! ## What lives here vs. what the caller supplies
//!
//! The *outer* optimisation (PIRLS inner fit + the LAML/REML profiling over the
//! smoothing hyperparameters ρ and any nuisance ψ) is owned by the outer-loop
//! machinery and is **deliberately not touched here**. The caller hands this
//! module the already-profiled criterion as a callable
//!
//! ```text
//! V_p(κ) = max_{ρ, other ψ} V(κ, ρ, ·)      (a 1-D function of κ)
//! ```
//!
//! and this module does the purely statistical work on top of it: the
//! profile-likelihood CI walk, the interior-point κ=0 LR test, and the
//! geometry-side κ-derivative API. None of the routines here re-enter the inner
//! fit; they only evaluate the `V_p` the caller provides.
//!
//! ## The `smooth.rs` seam (documented, not edited)
//!
//! When `ConstantCurvature` becomes a smooth term, its design block `X(κ)` is
//! built from geodesic normal coordinates `log_{x̄}(yᵢ)` of the latent points
//! about a base `x̄` (the intrinsic-S² Wahba smooth is the structural template).
//! The single quantity whose κ-movement the outer gradient consumes is therefore
//! `∂ log_{x̄}(y)/∂κ` (and `∂²/∂κ²` for the exact Wald curvature). The seam is:
//!
//! * In `terms/smooth.rs`, wherever the constant-curvature smooth builds its
//!   design from `manifold.log_map(x̄, yᵢ)` (the per-row normal coordinates),
//!   the ψ-channel variant must instead call
//!   [`design_coord_kappa_derivative`] to obtain the *same* coordinates together
//!   with their `∂/∂κ` and `∂²/∂κ²`. That triple feeds the outer assembly's
//!   ext-coord channel exactly as the Matérn-κ basis hyper-derivatives do
//!   (hyper.rs ext-coords → unified outer assembly, with `∂S/∂κ` handled by the
//!   penalty ψ-derivatives). κ then optimises as one more signed ψ-coordinate;
//!   no new outer machinery is introduced — this module only provides the
//!   geometry-side derivative the seam reads.
//!
//! The API here is intentionally allocation-light and stateless so the seam can
//! call it per row inside the design build without owning any outer state.

use ndarray::Array1;

use super::closure_family::inv_std_normal;

/// χ²₁ survival function `P(χ²₁ > t)` for `t ≥ 0` — the p-value of an
/// interior-point likelihood-ratio statistic on one degree of freedom.
///
/// Evaluated as the identity `P(χ²₁ > t) = P(|Z| > √t) = 2Φ(−√t) = erfc(√(t/2))`
/// rather than as the `2(1 − Φ(√t))` the definition is usually written in.
/// Those are the same function on paper and nothing alike in `f64`: `Φ(√t)`
/// closes on `1`, so the subtraction throws away every digit the answer is made
/// of. Measured against a 40-digit reference, it passed 1% relative error at
/// `t = 59.7`, reached 18% at `t = 67`, and returned **exactly zero** for every
/// `t ≥ 68.764` — the point where the true tail `1.11e−16` falls under
/// `ulp(1)/2` and `Φ(√t)` rounds to `1.0`. An LR statistic of 100 reported
/// `p = 0` where the truth is `1.5e−23`. That is not a rounding difference in a
/// diagnostic; it is a p-value of zero, and `flatness_lr_test` publishes it
/// through `FlatnessTest::p_value` into the Python `summary()`.
///
/// The `erfc` form subtracts nothing, so it holds full relative precision for
/// as long as its own result is representable: `t ≤ 1409` in the normal range,
/// and non-zero (subnormal, so with digits falling away) out to `t = 1482`.
/// That is 21x further out in `t` than the subtractive form reached, and the
/// range it gains is entirely the range where a p-value is worth computing at
/// all — below `t ≈ 60` the two forms agree to 14 digits.
///
/// The argument is routed through `erfcx` so the square is carried exactly:
/// `erfc(u) = erfcx(u)·exp(−u²)` with `u² = t/2`, and halving is exact in
/// binary. Handing `√(t/2)` to `erfc` directly instead would re-round the
/// square, and `erfc` converts a relative perturbation of its argument into
/// `2u²= t` times as much relative error in its result — `5.5e−14` at
/// `t = 1000`, against the `~ε` this form holds. It is the same correction
/// `probability::square_residual` exists for, available here for free because
/// `t/2` is exact and never has to be recovered.
fn chi2_1_sf(t: f64) -> f64 {
    if !(t > 0.0) {
        // Also catches NaN, for which no tail probability is defined.
        return if t.is_nan() { f64::NAN } else { 1.0 };
    }
    let half = 0.5 * t;
    gam_math::probability::erfcx_nonnegative(half.sqrt()) * (-half).exp()
}

/// `χ²₁(level)` two-sided quantile: `(Φ⁻¹((1+level)/2))²`.
fn chi2_1_quantile(level: f64) -> f64 {
    let z = inv_std_normal(0.5 * (1.0 + level));
    z * z
}

/// The geometric verdict implied by the sign of the κ confidence interval — a
/// topology-free, likelihood-based answer to "what curvature does my latent
/// space have?". Composes *within* a fixed topology candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureVerdict {
    /// CI lies strictly in `(0, ∞)` — spherical at the chosen level.
    Spherical,
    /// CI lies strictly in `(−∞, 0)` — hyperbolic at the chosen level.
    Hyperbolic,
    /// CI straddles `0` — indistinguishable from flat.
    Flat,
}

/// Where the point estimate `κ̂` sits relative to the chart-feasible search box
/// it was optimized in — the provenance the CI's `*_at_bound` flags give for the
/// interval's endpoints, and which `κ̂` itself did not carry (gam#2687).
///
/// `κ̂` is the argmin of `V_p` **over the box**. When the box constraint is
/// active, that argmin is a readout of the box rather than of the data: the
/// criterion never turned over inside it, and moving the box moves `κ̂` by
/// exactly as much. #2687 measured that directly — on the coverage fixture
/// `V_p` is strictly decreasing across the entire admissible interval, so
/// `κ̂ = 1.391` is the endpoint `0.5/max‖x‖²` to four figures in every
/// replicate, and widening the box to the resolution-derived end moves it to
/// 2.78 against a planted 1.5.
///
/// Consumers must not read a railed `κ̂` as an estimate. Two statistics
/// downstream of it change meaning when it is railed, and neither can detect it
/// on its own:
///
/// * the **profile CI** thresholds `2[V_p(κ) − V_p(κ̂)]` against `χ²₁`, which is
///   the Wilks region only when `κ̂` is an interior stationary point;
/// * the **flatness LR** compares `V_p(0)` against `V_p(κ̂)`, so a railed `κ̂`
///   understates the statistic (the true alternative optimum, if any, lies
///   outside the box) — an error in the conservative direction, but an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KappaEstimateSupport {
    /// `κ̂` is an interior stationary point of `V_p`: an estimate.
    Interior,
    /// `κ̂` IS `kappa_min`. The criterion was still descending toward the
    /// hyperbolic chart wall when the box stopped the search.
    RailedAtLowerBound,
    /// `κ̂` IS `kappa_max`. The criterion was still descending toward the
    /// antipodal fold when the box stopped the search.
    RailedAtUpperBound,
}

impl KappaEstimateSupport {

    /// Serialized provenance label, for the report surfaces.
    pub fn label(self) -> &'static str {
        match self {
            Self::Interior => "interior",
            Self::RailedAtLowerBound => "railed_at_lower_bound",
            Self::RailedAtUpperBound => "railed_at_upper_bound",
        }
    }
}

/// Where the profiled range `ℓ̂` sits relative to the range chart it was solved
/// in — the range coordinate's analogue of [`KappaEstimateSupport`], and for the
/// same reason (gam#2747).
///
/// `ℓ̂` is the argmin of `V(κ̂, ·)` over the chart, and the magnitude alone
/// cannot say whether the criterion turned over inside it. Unlike the κ box,
/// though, the range chart's two ends are not the same kind of place, and the
/// whole of gam#2747 on this coordinate is that conflating them hid a defect:
/// the bottom is an evaluability wall (below it the Gram's dynamic range exceeds
/// what a double-precision Cholesky resolves), while the TOP is the model's own
/// limit — `k → −d_κ`, the geodesic-distance kernel, which is conditionally
/// positive definite on all three space forms and therefore an ordinary smooth.
///
/// So a reader must be able to tell three things apart, not two:
///
/// * an interior minimum — `ℓ̂` is an estimate;
/// * an ARRIVAL at the distance-kernel face — `ℓ̂` is a lower bound with a
///   meaning, "the kernel is the geodesic distance", and nothing past it is a
///   different model;
/// * anything else — pinned by the user, stopped at the evaluability wall, or a
///   solve whose η-curvature does not identify the reduction.
///
/// The middle case is what `20bde053f` reported as "a readout of the box rather
/// than of the data" and reverted an enrollment over. It is not one, and saying
/// which of the three happened is how a consumer stops having to guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeEstimateSupport {
    /// `V_η = 0` with `V_ηη > 0` strictly inside the chart: an estimate, and the
    /// only case in which the profile's envelope/Schur reduction applies.
    Interior,
    /// `ℓ̂` is at the top of the chart, where the kernel has become `−d_κ` to
    /// within `√ε` in every design entry. An arrival, not a rail.
    DistanceKernelLimit,
    /// Pinned by an explicit `length_scale=`, parked at the evaluability wall,
    /// or otherwise not certified as an interior minimizer. `η̂` is locally
    /// constant in κ, so the profile takes the plain κ slice.
    LocallyFixed,
}

impl RangeEstimateSupport {

    /// Serialized provenance label, for the report surfaces.
    pub fn label(self) -> &'static str {
        match self {
            Self::Interior => "interior",
            Self::DistanceKernelLimit => "distance_kernel_limit",
            Self::LocallyFixed => "locally_fixed",
        }
    }
}

/// Profile-likelihood confidence interval for the fitted curvature `κ̂`.
///
/// The set is the Wilks region `{κ : 2[V_p(κ) − V_p(κ̂)] ≤ χ²_{1,1−α}}` where
/// `V_p` is the caller's profiled criterion (LAML/REML negative log-evidence,
/// already maximised over ρ and any nuisance ψ at each κ; `κ̂` is its minimiser,
/// so the profile *drop* `V_p(κ) − V_p(κ̂) ≥ 0` is what the CI walk thresholds at
/// `½χ²`). Because κ=0 is an
/// **interior** point of the smooth `S^d ← ℝ^d → H^d` family, no half-χ²
/// boundary correction is applied.
#[derive(Clone, Copy, Debug)]
pub struct KappaProfileCi {
    /// The profile minimiser κ̂ (caller-supplied; echoed for convenience).
    pub kappa_hat: f64,
    /// Lower CI endpoint.
    pub ci_lo: f64,
    /// Upper CI endpoint.
    pub ci_hi: f64,
    /// `true` if the lower walk hit the `kappa_min` chart bound before the
    /// profile drop reached the threshold (CI is left-open at the bound).
    pub lo_at_bound: bool,
    /// `true` if the upper walk hit the `kappa_max` chart bound before the
    /// profile drop reached the threshold (CI is right-open at the bound).
    pub hi_at_bound: bool,
    /// Whether the POINT ESTIMATE `κ̂` is itself a box endpoint (gam#2687).
    /// The two flags above answer "is this CI endpoint real?"; this one answers
    /// the prior question "is `κ̂` an estimate at all?", which nothing was
    /// answering. See [`KappaEstimateSupport`].
    pub kappa_hat_support: KappaEstimateSupport,
    /// Geometry verdict from the CI sign.
    pub verdict: CurvatureVerdict,
}

/// Result of the interior-point κ=0 likelihood-ratio test.
#[derive(Clone, Copy, Debug)]
pub struct FlatnessTest {
    /// LR statistic `2[V_p(0) − V_p(κ̂)] ≥ 0` (`V_p` is a negative log-evidence,
    /// so the constrained fit κ=0 has the larger `V_p`).
    pub lr_stat: f64,
    /// p-value against the **interior** χ²₁ reference (no boundary correction).
    pub p_value: f64,
    /// The fitted curvature, echoed.
    pub kappa_hat: f64,
}

/// Wald starting bracket `κ̂ ± z_{1−α/2} / √(∂²(−V_p)/∂κ²)` from the exact outer
/// curvature `v_pp = ∂²V_p/∂κ²` at κ̂ (which is `∂²(−V_p_evidence)` in the
/// negative-log-evidence convention used here, i.e. positive at a minimiser).
///
/// `v_pp` must be `> 0` (a genuine minimiser of `V_p`); the returned half-width
/// is `z / √v_pp`. Returns `None` when `v_pp` is non-positive (the Wald
/// approximation is undefined — fall back to a wider manual bracket).
pub fn wald_half_width(v_pp: f64, level: f64) -> Option<f64> {
    if !(v_pp.is_finite()) || v_pp <= 0.0 {
        return None;
    }
    let z = inv_std_normal(0.5 * (1.0 + level));
    Some(z / v_pp.sqrt())
}

/// Profile-likelihood CI for κ by walking out from κ̂ until the profile drop
/// `2[V_p(κ̂) − V_p(κ)]` crosses `χ²_{1,1−level}`, on each side independently.
///
/// * `v_p` — the caller's profiled criterion `V_p(κ)` (a **negative**
///   log-evidence: lower is better, so `κ̂` is its argmin). Each call may run a
///   full inner fit + ρ/ψ profile; this routine treats it as an opaque oracle.
/// * `kappa_hat` — the profile minimiser (the outer optimiser's fitted κ).
/// * `v_pp` — the exact outer curvature `∂²V_p/∂κ²` at κ̂ (from the κ-channel
///   LAML second derivative). Used only to size the initial Wald step; the CI
///   itself is the exact likelihood crossing, not the Wald ellipsoid. May be
///   non-positive, in which case a default initial step is used.
/// * `(kappa_min, kappa_max)` — chart-validity bounds on κ; the walk refuses to
///   step outside them and flags the corresponding endpoint as `*_at_bound`.
/// * `level` — two-sided coverage, e.g. `0.95`.
///
/// The walk does geometric step-growth to bracket each crossing, then bisects to
/// `tol` in κ. The threshold uses the full χ²₁ quantile (interior point).
pub fn profile_ci_walk<F>(
    mut v_p: F,
    kappa_hat: f64,
    v_pp: f64,
    kappa_min: f64,
    kappa_max: f64,
    level: f64,
    tol: f64,
) -> Result<KappaProfileCi, String>
where
    F: FnMut(f64) -> Result<f64, String>,
{
    if !(level > 0.0 && level < 1.0) {
        return Err("profile CI level must lie in (0, 1)".into());
    }
    if !(kappa_min < kappa_max) {
        return Err("kappa bounds must satisfy kappa_min < kappa_max".into());
    }
    if !(kappa_hat.is_finite()) || kappa_hat < kappa_min || kappa_hat > kappa_max {
        return Err("kappa_hat must be finite and inside [kappa_min, kappa_max]".into());
    }
    let tol = if tol > 0.0 { tol } else { 1e-6 };
    let half_thresh = 0.5 * chi2_1_quantile(level);
    let v_hat = v_p(kappa_hat)?;
    if !v_hat.is_finite() {
        return Err("V_p(kappa_hat) is non-finite".into());
    }

    // Initial step: Wald half-width if the curvature is usable, else a modest
    // default scaled to the bracket so the first probe is informative.
    let init_step = wald_half_width(v_pp, level)
        .filter(|h| h.is_finite() && *h > 0.0)
        .unwrap_or_else(|| 0.1 * (kappa_max - kappa_min).max(tol));

    // Profile drop relative to κ̂: `g(κ) = 2[V_p(κ) − V_p(κ̂)] ≥ 0`. The CI
    // endpoint is the κ where `g = χ²` (i.e. half_thresh on the raw `V_p` scale).
    let drop = |v: f64| v - v_hat;

    let cfg = WalkCfg {
        kappa_hat,
        init_step,
        half_thresh,
        tol,
    };
    let (ci_lo, lo_at_bound) = walk_one_side(&mut v_p, &cfg, -1.0, kappa_min, &drop)?;
    let (ci_hi, hi_at_bound) = walk_one_side(&mut v_p, &cfg, 1.0, kappa_max, &drop)?;

    // Is κ̂ itself a box endpoint? Denominated in the SAME `tol` the walk
    // bisects to, so "κ̂ is at the bound" and "this side returned zero width"
    // are one statement rather than two thresholds that can disagree.
    let kappa_hat_support = if (kappa_hat - kappa_min).abs() <= tol {
        KappaEstimateSupport::RailedAtLowerBound
    } else if (kappa_max - kappa_hat).abs() <= tol {
        KappaEstimateSupport::RailedAtUpperBound
    } else {
        KappaEstimateSupport::Interior
    };

    let verdict = if ci_lo > 0.0 {
        CurvatureVerdict::Spherical
    } else if ci_hi < 0.0 {
        CurvatureVerdict::Hyperbolic
    } else {
        CurvatureVerdict::Flat
    };

    Ok(KappaProfileCi {
        kappa_hat,
        ci_lo,
        ci_hi,
        lo_at_bound,
        hi_at_bound,
        kappa_hat_support,
        verdict,
    })
}

/// Shared scalar configuration for the two one-sided CI walks.
struct WalkCfg {
    kappa_hat: f64,
    init_step: f64,
    half_thresh: f64,
    tol: f64,
}

/// Walk in one direction (`sign = ±1`) from κ̂ until the profile-drop crossing,
/// returning `(endpoint, hit_bound)`.
fn walk_one_side<F, D>(
    v_p: &mut F,
    cfg: &WalkCfg,
    sign: f64,
    bound: f64,
    drop: &D,
) -> Result<(f64, bool), String>
where
    F: FnMut(f64) -> Result<f64, String>,
    D: Fn(f64) -> f64,
{
    let WalkCfg {
        kappa_hat,
        init_step,
        half_thresh,
        tol,
    } = *cfg;
    // Bracket: grow the step geometrically until `drop ≥ half_thresh` or we hit
    // the chart bound. `lo` is inside the CI (drop < thresh), `hi` is outside.
    let mut lo = kappa_hat;
    let mut step = init_step.max(tol);
    let span = (bound - kappa_hat) * sign; // ≥ 0 distance to the bound
    if span <= tol {
        // No room to move toward the bound: CI is open at the bound here.
        return Ok((bound, true));
    }
    let mut probe = step.min(span);
    loop {
        let kappa = kappa_hat + sign * probe;
        let v = v_p(kappa)?;
        if !v.is_finite() {
            return Err("V_p returned a non-finite value during the CI walk".into());
        }
        if drop(v) >= half_thresh {
            // Crossing bracketed in [lo, kappa]: bisect to tolerance.
            let mut a = lo; // drop < thresh
            let mut b = kappa; // drop ≥ thresh
            while (b - a).abs() > tol {
                let m = 0.5 * (a + b);
                let vm = v_p(m)?;
                if !vm.is_finite() {
                    return Err("V_p returned a non-finite value during bisection".into());
                }
                if drop(vm) >= half_thresh {
                    b = m;
                } else {
                    a = m;
                }
            }
            return Ok((0.5 * (a + b), false));
        }
        // Still inside: advance.
        lo = kappa;
        if (probe - span).abs() <= tol {
            // Reached the chart bound without crossing: CI open at the bound.
            return Ok((bound, true));
        }
        step *= 2.0;
        probe = (probe + step).min(span);
    }
}

/// Interior-point likelihood-ratio test of `κ = 0` (flatness).
///
/// `lr = 2[V_p(0) − V_p(κ̂)] ~ χ²₁` under H₀: κ=0. Because the constant-curvature
/// family interpolates `S^d ← ℝ^d → H^d` smoothly, κ=0 is an interior point —
/// the reference is the full χ²₁, **not** the ½χ²₀ + ½χ²₁ mixture of a
/// variance-component boundary test. The p-value is `P(χ²₁ > lr)`.
///
/// * `v_p` — the profiled criterion (negative log-evidence).
/// * `kappa_hat` — the unconstrained fitted curvature.
///
/// `lr` is clamped at `0` to absorb tiny numerical negativity when κ̂≈0.
pub fn flatness_lr_test<F>(mut v_p: F, kappa_hat: f64) -> Result<FlatnessTest, String>
where
    F: FnMut(f64) -> Result<f64, String>,
{
    let v_hat = v_p(kappa_hat)?;
    let v_zero = v_p(0.0)?;
    if !v_hat.is_finite() || !v_zero.is_finite() {
        return Err("V_p evaluated to a non-finite value in the flatness test".into());
    }
    let lr_stat = (2.0 * (v_zero - v_hat)).max(0.0);
    let p_value = chi2_1_sf(lr_stat);
    Ok(FlatnessTest {
        lr_stat,
        p_value,
        kappa_hat,
    })
}

/// The design-moving geometry quantity and its κ-derivatives, for one latent
/// row — the clean API the `smooth.rs` ψ-channel seam calls.
///
/// The constant-curvature smooth's design is built from geodesic normal
/// coordinates `coord = log_{base}(point)`. This returns that vector together
/// with `∂coord/∂κ` and `∂²coord/∂κ²` (exact, from `log_map_kappa_jet`), which
/// the outer assembly's ext-coord channel consumes when κ moves as a ψ-coordinate.
#[derive(Clone, Debug)]
pub struct DesignCoordKappaJet {
    /// The normal coordinate `log_{base}(point)` at the current κ.
    pub coord: Array1<f64>,
    /// `∂coord/∂κ`.
    pub d_kappa: Array1<f64>,
    /// `∂²coord/∂κ²`.
    pub d_kappa2: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // A synthetic profiled criterion with a known minimiser and curvature:
    //   V_p(κ) = v0 + 0.5 * a * (κ − κ⋆)²   (a > 0, minimiser at κ⋆).
    // Then 2[V_p(κ) − V_p(κ⋆)] = a (κ − κ⋆)², so the exact χ²-crossing CI is
    //   κ⋆ ± √(χ²₁(level) / a),  and ∂²V_p/∂κ² = a (the Wald bracket is EXACT
    // for a quadratic, so the walk must return the closed-form endpoints).
    fn quad(v0: f64, a: f64, k_star: f64) -> impl Fn(f64) -> Result<f64, String> {
        move |k: f64| Ok(v0 + 0.5 * a * (k - k_star) * (k - k_star))
    }

    #[test]
    fn wald_half_width_matches_closed_form() {
        let level = 0.95;
        let a = 3.0;
        let h = wald_half_width(a, level).expect("positive curvature");
        let z = inv_std_normal(0.5 * (1.0 + level));
        assert!((h - z / a.sqrt()).abs() < 1e-12);
        assert!(wald_half_width(0.0, level).is_none());
        assert!(wald_half_width(-1.0, level).is_none());
    }

    #[test]
    fn profile_ci_walk_recovers_quadratic_crossing() {
        let level = 0.95;
        let a = 2.5;
        let k_star = -0.7;
        let f = quad(0.3, a, k_star);
        let ci = profile_ci_walk(
            |k| f(k),
            k_star,
            a, // exact ∂²V_p/∂κ²
            -10.0,
            10.0,
            level,
            1e-9,
        )
        .expect("CI walk");
        let chi2 = chi2_1_quantile(level);
        let half = (chi2 / a).sqrt();
        // Exact closed-form endpoints for the quadratic profile.
        assert!((ci.ci_lo - (k_star - half)).abs() < 1e-6, "lo {}", ci.ci_lo);
        assert!((ci.ci_hi - (k_star + half)).abs() < 1e-6, "hi {}", ci.ci_hi);
        assert!(!ci.lo_at_bound && !ci.hi_at_bound);
        // κ̂ < 0 and the CI straddles… check the verdict against the sign.
        let expected = if ci.ci_lo > 0.0 {
            CurvatureVerdict::Spherical
        } else if ci.ci_hi < 0.0 {
            CurvatureVerdict::Hyperbolic
        } else {
            CurvatureVerdict::Flat
        };
        assert_eq!(ci.verdict, expected);
    }

    #[test]
    fn profile_ci_walk_verdict_hyperbolic_when_far_negative() {
        // Sharp, far-negative minimiser ⇒ CI strictly < 0 ⇒ Hyperbolic.
        let level = 0.95;
        let a = 50.0; // sharp ⇒ narrow CI
        let k_star = -2.0;
        let f = quad(0.0, a, k_star);
        let ci = profile_ci_walk(|k| f(k), k_star, a, -10.0, 10.0, level, 1e-9).unwrap();
        assert!(ci.ci_hi < 0.0, "ci_hi {}", ci.ci_hi);
        assert_eq!(ci.verdict, CurvatureVerdict::Hyperbolic);
    }

    #[test]
    fn profile_ci_walk_flags_bound_when_profile_too_flat() {
        // Very flat profile inside a tight bracket ⇒ never crosses ⇒ both
        // endpoints pinned to the bounds and flagged.
        let level = 0.95;
        let a = 1e-6;
        let k_star = 0.0;
        let f = quad(0.0, a, k_star);
        let ci = profile_ci_walk(|k| f(k), k_star, a, -0.01, 0.01, level, 1e-9).unwrap();
        assert!(ci.lo_at_bound && ci.hi_at_bound);
        assert!((ci.ci_lo + 0.01).abs() < 1e-12 && (ci.ci_hi - 0.01).abs() < 1e-12);
        assert_eq!(ci.verdict, CurvatureVerdict::Flat);
    }

    #[test]
    fn flatness_test_zero_when_minimiser_is_flat() {
        // κ̂ = 0 ⇒ lr = 0 ⇒ p = 1.
        let f = quad(1.0, 4.0, 0.0);
        let t = flatness_lr_test(|k| f(k), 0.0).unwrap();
        assert!(t.lr_stat.abs() < 1e-12);
        assert!((t.p_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn flatness_test_lr_and_pvalue_match_chi2_1() {
        // lr = 2[V_p(0) − V_p(κ̂)] = a κ⋆²; p = χ²₁ survival at that stat.
        let a = 3.0;
        let k_star = 0.8;
        let f = quad(0.5, a, k_star);
        let t = flatness_lr_test(|k| f(k), k_star).unwrap();
        let expected_lr = a * k_star * k_star;
        assert!((t.lr_stat - expected_lr).abs() < 1e-10, "lr {}", t.lr_stat);
        let expected_p = chi2_1_sf(expected_lr);
        assert!((t.p_value - expected_p).abs() < 1e-12);
        // No boundary correction: an interior χ²₁ p-value, NOT a half-χ². The
        // half-χ² mixture would give exactly half this tail; assert we did NOT
        // apply it.
        let half_chi2_p = 0.5 * expected_p;
        assert!((t.p_value - half_chi2_p).abs() > 1e-6);
    }

    #[test]
    fn chi2_1_sf_matches_known_quantiles() {
        // χ²₁(0.95) = 3.841459…; survival at that point is 0.05.
        let q = chi2_1_quantile(0.95);
        assert!((q - 3.841_458_820_694_124).abs() < 1e-6, "q {}", q);
        assert!((chi2_1_sf(q) - 0.05).abs() < 1e-9);
        assert!((chi2_1_sf(chi2_1_quantile(0.99)) - 0.01).abs() < 1e-9);
    }

    /// The tail is where an LR test actually reports, and it is the half of the
    /// domain the quantile check above cannot see: both of its probes sit at
    /// `t < 7`, where the discarded `2(1 − Φ(√t))` form was still correct to 16
    /// digits. References are `mpmath.erfc(sqrt(t/2))` at 40 digits.
    ///
    /// `t = 70` and beyond are the cases that used to return exactly `0.0`.
    #[test]
    fn chi2_1_sf_holds_relative_precision_into_the_deep_tail() {
        const CASES: [(f64, f64); 8] = [
            (4.0, 0.045_500_263_896_358_414),
            (25.0, 5.733_031_437_583_878_2e-7),
            (50.0, 1.537_459_794_428_034_9e-12),
            (67.0, 2.715_071_321_942_525_9e-16),
            (70.0, 5.930_445_850_082_486_8e-17),
            (100.0, 1.523_970_604_832_105_2e-23),
            (200.0, 2.088_487_583_762_544_8e-45),
            (1000.0, 1.795_832_784_800_726_2e-219),
        ];
        for (t, expected) in CASES {
            let got = chi2_1_sf(t);
            let relative = (got - expected).abs() / expected;
            assert!(
                relative < 1e-14,
                "chi2_1_sf({t}) = {got:e}, expected {expected:e}, rel {relative:e}"
            );
        }
    }

    /// A p-value is a probability: it must be finite, in `[0, 1]`, and
    /// monotonically non-increasing in the statistic. The subtractive form
    /// violated the last of these in the tail, where it quantized to a staircase
    /// of multiples of `2·ε` before flattening onto zero.
    #[test]
    fn chi2_1_sf_is_a_monotone_probability_across_the_whole_domain() {
        assert_eq!(chi2_1_sf(0.0), 1.0);
        assert_eq!(chi2_1_sf(-1.0), 1.0);
        assert!(chi2_1_sf(f64::NAN).is_nan());
        assert_eq!(chi2_1_sf(f64::INFINITY), 0.0);

        let mut previous = 1.0_f64;
        for step in 0..=14_000 {
            let t = 0.1 * f64::from(step);
            let p = chi2_1_sf(t);
            assert!(
                (0.0..=1.0).contains(&p),
                "chi2_1_sf({t}) = {p} is not a probability"
            );
            assert!(p <= previous, "chi2_1_sf rose at t={t}: {previous} -> {p}");
            assert!(p > 0.0, "chi2_1_sf({t}) underflowed to zero");
            previous = p;
        }
    }

    // The κ-derivative API must echo `log_map_kappa_jet` exactly (it is a thin,
    // intent-named adapter) and the derivatives must match a central finite
    // difference of the value channel in κ.
    #[test]
    fn design_coord_kappa_derivative_matches_jet_and_fd() {
        let dim = 3;
        let kappa = 0.6;
        let manifold = ConstantCurvature::new(dim, kappa);
        let base = array![0.05, -0.1, 0.07];
        let point = array![0.2, 0.15, -0.05];

        let jet = design_coord_kappa_derivative(&manifold, base.view(), point.view()).unwrap();
        let (val, dk, dkk) = log_map_kappa_jet(&manifold, base.view(), point.view()).unwrap();
        for i in 0..dim {
            assert!((jet.coord[i] - val[i]).abs() < 1e-14);
            assert!((jet.d_kappa[i] - dk[i]).abs() < 1e-14);
            assert!((jet.d_kappa2[i] - dkk[i]).abs() < 1e-14);
        }

        // Central FD of the value channel in κ for ∂/∂κ and ∂²/∂κ².
        let h = 1e-5;
        let coord_at = |k: f64| -> Array1<f64> {
            let m = ConstantCurvature::new(dim, k);
            log_map_kappa_jet(&m, base.view(), point.view()).unwrap().0
        };
        let cp = coord_at(kappa + h);
        let cm = coord_at(kappa - h);
        let c0 = jet.coord.clone();
        for i in 0..dim {
            let fd1 = (cp[i] - cm[i]) / (2.0 * h);
            let fd2 = (cp[i] - 2.0 * c0[i] + cm[i]) / (h * h);
            assert!((jet.d_kappa[i] - fd1).abs() < 1e-6, "d_kappa[{i}] vs FD");
            assert!((jet.d_kappa2[i] - fd2).abs() < 1e-4, "d_kappa2[{i}] vs FD");
        }
    }

    // Near the flat point κ=0 the adapter must still agree with the FD of the
    // value (the Taylor branch boundary of the underlying jet).
    #[test]
    fn design_coord_kappa_derivative_fd_through_flat() {
        let dim = 2;
        let kappa = 1e-6;
        let manifold = ConstantCurvature::new(dim, kappa);
        let base = array![0.1, -0.2];
        let point = array![0.25, 0.05];
        let jet = design_coord_kappa_derivative(&manifold, base.view(), point.view()).unwrap();
        let h = 1e-4;
        let coord_at = |k: f64| -> Array1<f64> {
            let m = ConstantCurvature::new(dim, k);
            log_map_kappa_jet(&m, base.view(), point.view()).unwrap().0
        };
        let cp = coord_at(kappa + h);
        let cm = coord_at(kappa - h);
        for i in 0..dim {
            let fd1 = (cp[i] - cm[i]) / (2.0 * h);
            assert!((jet.d_kappa[i] - fd1).abs() < 1e-5, "flat d_kappa[{i}]");
        }
    }

    /// gam#2687: a MONOTONE profiled criterion has no interior optimum, so κ̂ is
    /// the box endpoint the search stopped at and not an estimate. The walk must
    /// say so — and must keep saying so when the box moves, since the whole
    /// point is that κ̂ moves with it.
    ///
    /// This is exactly the shape measured on the #2687 fixture: `V_p` strictly
    /// decreasing across the entire admissible interval, `κ̂` equal to the cap to
    /// four figures in every replicate.
    #[test]
    fn a_monotone_criterion_rails_kappa_hat_and_the_walk_declares_it_2687() {
        // V_p(κ) = −κ: strictly decreasing, so the box's upper end is the argmin
        // wherever that end is put.
        let monotone = |kappa: f64| -> Result<f64, String> { Ok(-kappa) };
        for kappa_max in [1.389_f64, 2.78, 40.0] {
            let ci = profile_ci_walk(monotone, kappa_max, -1.0, -kappa_max, kappa_max, 0.95, 1e-8)
                .expect("a monotone profile is a legal input; it is a rail, not an error");
            assert_eq!(
                ci.kappa_hat_support,
                KappaEstimateSupport::RailedAtUpperBound,
                "κ̂ = {kappa_max} IS the box's upper end; the report must not call \
                 it an estimate"
            );
            assert!(ci.kappa_hat_support.is_railed());
            assert_eq!(ci.kappa_hat_support.label(), "railed_at_upper_bound");
            // And the CI's own right endpoint is open at the same bound, which
            // is the flag that already existed: the two answer different
            // questions and must both be available.
            assert!(
                ci.hi_at_bound,
                "the CI is right-open at the bound the estimate is railed against"
            );
        }
        // The mirrored case, so the declaration is not accidentally one-sided.
        let increasing = |kappa: f64| -> Result<f64, String> { Ok(kappa) };
        let ci = profile_ci_walk(increasing, -2.0, -1.0, -2.0, 2.0, 0.95, 1e-8)
            .expect("monotone increasing rails at the lower end");
        assert_eq!(
            ci.kappa_hat_support,
            KappaEstimateSupport::RailedAtLowerBound
        );
        assert_eq!(ci.kappa_hat_support.label(), "railed_at_lower_bound");
    }

    /// The other half of the contract: an interior stationary point must NOT be
    /// declared railed, however close the box happens to be. A declaration that
    /// fired on every fit would be as useless as one that never fired.
    #[test]
    fn an_interior_optimum_is_not_declared_railed_2687() {
        let kappa_star = -0.37_f64;
        let a = 16.0_f64;
        let quadratic = |kappa: f64| -> Result<f64, String> {
            Ok(7.0 + 0.5 * a * (kappa - kappa_star) * (kappa - kappa_star))
        };
        let ci = profile_ci_walk(quadratic, kappa_star, a, -3.0, 3.0, 0.95, 1e-8)
            .expect("quadratic profile CI");
        assert_eq!(
            ci.kappa_hat_support,
            KappaEstimateSupport::Interior,
            "an interior minimiser is an estimate"
        );
        assert!(!ci.kappa_hat_support.is_railed());
        assert!(!ci.lo_at_bound && !ci.hi_at_bound);

        // Squeeze the box until κ̂ sits ON its lower end. Same criterion, same
        // κ̂ — only the box moved — and the declaration flips. That is the
        // property #2687 needs: the flag reports the BOX's involvement, not a
        // shape of the criterion.
        let squeezed = profile_ci_walk(quadratic, kappa_star, a, kappa_star, 3.0, 0.95, 1e-8)
            .expect("boxed-at-the-optimum profile CI");
        assert_eq!(
            squeezed.kappa_hat_support,
            KappaEstimateSupport::RailedAtLowerBound
        );
    }
}
