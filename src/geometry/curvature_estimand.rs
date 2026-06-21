//! Curvature-as-an-estimand: the inference layer on top of the κ-jets.
//!
//! #944 stage 3. The κ-jets (`distance_kappa_jet` / `log_map_kappa_jet` /
//! `exp_map_kappa_jet`) and the analytic Jacobi-field `exp_map_vjp` for the
//! [`ConstantCurvature`](crate::geometry::ConstantCurvature) chart are landed and
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

use ndarray::{Array1, ArrayView1};

use super::constant_curvature::{ConstantCurvature, log_map_kappa_jet};
use super::manifold::GeometryResult;

/// Acklam rational inverse standard-normal CDF + one Halley refinement —
/// deterministic and dependency-free, the same construction the closure-family
/// CI driver uses. `inv_std_normal(p)` solves `Φ(x) = p`.
fn inv_std_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let e = 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

/// Standard-normal upper-tail / two-sided CDF: `Φ(x)`.
fn std_normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

/// χ²₁ survival function `P(χ²₁ > t) = 2(1 − Φ(√t))` for `t ≥ 0` — the p-value
/// of an interior-point likelihood-ratio statistic on one degree of freedom.
fn chi2_1_sf(t: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    2.0 * (1.0 - std_normal_cdf(t.sqrt()))
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

/// Geodesic normal coordinate `log_{base}(point)` and its `∂/∂κ`, `∂²/∂κ²` on
/// the constant-curvature chart — the per-row design quantity whose κ-movement
/// the outer ψ-channel consumes (see the module-level `smooth.rs` seam note).
///
/// This is a thin, allocation-light adapter over [`log_map_kappa_jet`] so the
/// seam has a single, intent-named entry point and does not re-derive which
/// geometric quantity moves the design.
pub fn design_coord_kappa_derivative(
    manifold: &ConstantCurvature,
    base: ArrayView1<'_, f64>,
    point: ArrayView1<'_, f64>,
) -> GeometryResult<DesignCoordKappaJet> {
    let (coord, d_kappa, d_kappa2) = log_map_kappa_jet(manifold, base, point)?;
    Ok(DesignCoordKappaJet {
        coord,
        d_kappa,
        d_kappa2,
    })
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
}
