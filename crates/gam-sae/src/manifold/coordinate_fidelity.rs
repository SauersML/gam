//! #2081 — per-atom chart coordinate-fidelity certificate + the seed-selection
//! tie-break that prices it.
//!
//! Reconstruction EV provably does NOT certify coordinate quality: a `K = 1`
//! circle chart can reconstruct its ring at EV 0.926 while reading an angle
//! coordinate at correlation 0.771 (the planted-ring case that motivated this
//! issue), and the weekday cyclic ordering collapses from 0.714 to 0.22 under a
//! rotation of the reading basis at unchanged EV. Every downstream claim we care
//! about (adjacency, dose-in-nats, identity-η², template transfer) consumes the
//! COORDINATE, not the reconstruction — so the coordinate must be a certified,
//! reported quantity, not an implicit by-product.
//!
//! This module reports two complementary, calibrated per-`d = 1`-atom quantities:
//!
//!  * a **circular-uniformity statistic** of the fitted coordinates against the
//!    atom's invariant (uniform) measure — Watson's `U²`
//!    ([`watson_u2_uniform`]). `U²` is rotation- AND reflection-invariant, so it
//!    is blind to the circle's residual `O(2)` gauge (base-point rotation +
//!    orientation reflection) and measures ONLY the coordinate distribution. It
//!    carries a closed-form asymptotic null p-value ([`watson_u2_pvalue`]) — no
//!    tabulated critical constant.
//!  * an **arc-length (unit-speed) defect**
//!    ([`crate::chart_canonicalization::chart_unit_speed_defect`]): the speed
//!    coefficient of variation of the decoded curve on a uniform latent grid — a
//!    pure property of the CHART parameterization, independent of the data.
//!    Reuses the isometry-gauge speed machinery (`speed_uniformity_defect`).
//!
//! The two separate the two failure modes: a non-uniform statistic with a LOW
//! arc-length defect means the DATA is genuinely non-uniform on an honest,
//! arc-length chart (no pathology); a HIGH arc-length defect means the chart
//! itself squishes arc length (the #2081 pathology), which EV cannot see.
//!
//! F2 — two-part split (chart honesty vs occupancy law). Watson's `U²` tests the
//! coordinates against the UNIFORM invariant measure, but uniformity is a
//! property of the data's OCCUPANCY, not the chart's honesty: a correct circle
//! whose data occupies seven points (weekdays) reads a highly non-uniform
//! coordinate and so fails a uniform-null test even though the chart is perfectly
//! honest. Reporting that as a fidelity failure conflates "dishonest chart" with
//! "discrete measure on an honest chart." The certificate therefore reports two
//! independent verdicts: `chart_honest` (a pure parameterization property — the
//! unit-speed / collapse verdict) and the `occupancy` law
//! ([`OccupancyLaw`]: `Uniform` / `Discrete{anchors}` / `Continuous`, adjudicated
//! by evidence via [`classify_occupancy`], NOT by the p-value). A discrete
//! measure on an honest chart passes chart-honesty and is reported as discrete
//! occupancy (`d_eff = anchors − 1`) — the finite-set alternative in the race.
//!
//! The seed-selection tie-break ([`prefer_candidate_basin`]) prices the
//! uniformity statistic: at (near-)equal reconstruction EV — "near" derived from
//! the existing #1026 EV negligibility band
//! [`crate::manifold::SAE_FINAL_EV_DEGRADATION_TOL`], not a fresh constant — the
//! more-uniform-coordinate basin wins, because EV alone provably cannot break
//! that tie.

use ndarray::{Array1, ArrayView1};

use crate::chart_canonicalization::{
    CanonicalChartTopology, ChartArcLengthReading, SAE_FLOW_DIFFEO_MIN_DET,
    UNIT_SPEED_INLOOP_DEFECT_TOL, chart_arclength_coordinates,
};

use super::SaeManifoldTerm;

/// #2081 — the certified verdict on whether a fitted `d = 1` atom carries an
/// honest angle/position coordinate. A downstream angle / dose-in-nats /
/// adjacency claim keys off this: read the raw `t` only under
/// [`Self::ArcLengthHonest`], read the canonical `u_arc` under
/// [`Self::RecoverableViaArcLength`], and REFUSE under [`Self::Degenerate`]
/// (the chart collapses, so no faithful coordinate exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleFidelityVerdict {
    /// The raw fitted coordinate is already arc-length (decoder-speed CV below
    /// the in-loop retraction tolerance [`UNIT_SPEED_INLOOP_DEFECT_TOL`]): the
    /// reported raw `t` IS the honest angle and `u_arc ≈ t`.
    ArcLengthHonest,
    /// The raw coordinate is NOT arc-length, but the arc-length
    /// reparameterization is a well-conditioned diffeomorphism (speed stays
    /// above the [`SAE_FLOW_DIFFEO_MIN_DET`] collapse floor everywhere), so the
    /// honest coordinate is recoverable: consumers must read `coords_u_arc`.
    RecoverableViaArcLength,
    /// The chart collapses — the decoder speed drops to a
    /// [`SAE_FLOW_DIFFEO_MIN_DET`] fraction of its mean somewhere, so `u_arc`
    /// has a flat spot and no faithful coordinate exists there. Refuse.
    Degenerate,
}

impl AngleFidelityVerdict {
    /// Lowercase label for the diagnostics payload.
    pub fn label(self) -> &'static str {
        match self {
            AngleFidelityVerdict::ArcLengthHonest => "arclength_honest",
            AngleFidelityVerdict::RecoverableViaArcLength => "recoverable_via_arclength",
            AngleFidelityVerdict::Degenerate => "degenerate",
        }
    }

    /// `true` when an honest coordinate is available (raw `t` under
    /// `ArcLengthHonest`, `coords_u_arc` under `RecoverableViaArcLength`).
    /// `false` only under `Degenerate`, where every coordinate consumer must
    /// refuse rather than read an arbitrary chart.
    pub fn certified(self) -> bool {
        !matches!(self, AngleFidelityVerdict::Degenerate)
    }
}

/// The certified angle-fidelity verdict from a chart's arc-length reading. The
/// two decision thresholds are the fit's OWN dimensionless invariants, not fresh
/// constants: the chart is a well-conditioned diffeomorphism iff its slowest
/// speed stays above the [`SAE_FLOW_DIFFEO_MIN_DET`] fraction of the mean (the
/// same fold floor the `d = 2` flow charts enforce on `det Dφ`), and the raw
/// coordinate is already the honest angle iff its speed CV is below
/// [`UNIT_SPEED_INLOOP_DEFECT_TOL`] (the same tolerance below which the in-loop
/// retraction treats a chart as already arc-length and skips it). A `None`
/// reading (arc length ill-defined) is `Degenerate`.
pub fn angle_fidelity_verdict(reading: Option<&ChartArcLengthReading>) -> AngleFidelityVerdict {
    match reading {
        Some(r) if r.min_speed_over_mean > SAE_FLOW_DIFFEO_MIN_DET => {
            if r.speed_cv < UNIT_SPEED_INLOOP_DEFECT_TOL {
                AngleFidelityVerdict::ArcLengthHonest
            } else {
                AngleFidelityVerdict::RecoverableViaArcLength
            }
        }
        _ => AngleFidelityVerdict::Degenerate,
    }
}

/// Watson's `U²` uniformity statistic and its asymptotic null p-value for a set
/// of coordinates already mapped onto the unit interval `[0, 1)` (a circle by
/// wrapping modulo its period, an interval by range-normalization). Larger `U²`
/// ⟺ farther from the uniform invariant measure.
#[derive(Debug, Clone, Copy)]
pub struct WatsonUniformity {
    /// Watson's `U²` statistic. Rotation- and reflection-invariant. Null mean
    /// `1/12 ≈ 0.0833`; the classical asymptotic upper-tail critical values are
    /// `0.187` (5%) and `0.267` (1%).
    pub statistic: f64,
    /// Closed-form asymptotic upper-tail p-value `P(U² ≥ statistic)` under the
    /// uniform null. Small ⟺ coordinates flagged non-uniform. `1.0` for
    /// `n < 2` (statistic undefined).
    pub p_value: f64,
    /// Number of coordinates the statistic was computed from.
    pub n: usize,
}

/// Closed-form asymptotic upper-tail p-value of Watson's `U²` under the uniform
/// null: `P(U² ≥ u) = 2 Σ_{j≥1} (−1)^{j−1} exp(−2 j² π² u)` (Watson 1961). This
/// is the exact limiting distribution — NOT a tabulated critical constant — so
/// the "flagged / not flagged" decision is derived, not tuned. As a check the
/// series returns `≈ 0.05` at the tabulated 5% point `u = 0.187` and `≈ 0.01` at
/// the 1% point `u = 0.267` (asserted in the tests). The alternating series
/// converges geometrically; terms below `1e-14` are negligible.
pub fn watson_u2_pvalue(u2: f64) -> f64 {
    if !(u2 > 0.0) {
        return 1.0;
    }
    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    let mut sum = 0.0_f64;
    for j in 1..=100_usize {
        let jf = j as f64;
        let term = (-two_pi_sq * jf * jf * u2).exp();
        sum += if j % 2 == 1 { term } else { -term };
        if term < 1.0e-14 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Watson's `U²` uniformity statistic of coordinates `u` on the unit interval
/// `[0, 1)` (values are folded into `[0, 1)` first, so a circle's wrapped
/// coordinate is handled directly). For sorted `u_(1) ≤ … ≤ u_(n)`,
///
/// ```text
///   W² = Σ_i (u_(i) − (2i−1)/(2n))² + 1/(12n)      (Cramér–von Mises)
///   U² = W² − n (ū − 1/2)²                         (Watson's rotation-invariant form)
/// ```
///
/// Subtracting `n(ū − 1/2)²` is exactly what makes `U²` invariant to a rotation
/// of the origin (and, being symmetric under `u ↦ 1 − u`, to reflection) — the
/// circle's residual `O(2)` gauge. Returns a zero statistic / unit p-value for
/// `n < 2`.
pub fn watson_u2_uniform(u: &[f64]) -> WatsonUniformity {
    let n = u.len();
    if n < 2 {
        return WatsonUniformity {
            statistic: 0.0,
            p_value: 1.0,
            n,
        };
    }
    // Fold into [0, 1) — a wrapped circle coordinate at exactly `period` folds to
    // `0`, and floating-point `1.0 − ε` folds cleanly.
    let mut v: Vec<f64> = u
        .iter()
        .map(|&x| {
            let f = x - x.floor();
            if f >= 1.0 { 0.0 } else { f }
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = n as f64;
    let mut cvm = 1.0 / (12.0 * nf);
    let mut mean = 0.0_f64;
    for (i, &ui) in v.iter().enumerate() {
        let expected = (2.0 * (i as f64 + 1.0) - 1.0) / (2.0 * nf);
        let d = ui - expected;
        cvm += d * d;
        mean += ui;
    }
    mean /= nf;
    let u2 = cvm - nf * (mean - 0.5) * (mean - 0.5);
    let p_value = watson_u2_pvalue(u2);
    WatsonUniformity {
        statistic: u2,
        p_value,
        n,
    }
}

/// Watson's `U²` uniformity of the fitted coordinates against the atom's
/// invariant (uniform) measure, per `d = 1` topology: a circle wraps modulo its
/// period; an interval is normalized by its fitted coordinate range. Returns
/// `None` (statistic undefined) for fewer than two coordinates, a non-finite
/// coordinate, a non-positive period, or a collapsed interval range.
pub fn coordinate_uniformity(
    coords: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Option<WatsonUniformity> {
    let n = coords.len();
    if n < 2 {
        return None;
    }
    if coords.iter().any(|t| !t.is_finite()) {
        return None;
    }
    let u: Vec<f64> = match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return None;
            }
            coords
                .iter()
                .map(|&t| t.rem_euclid(*period) / *period)
                .collect()
        }
        CanonicalChartTopology::Interval => {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &t in coords.iter() {
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let span = hi - lo;
            let scale = lo.abs().max(hi.abs()).max(1.0);
            if !(span > 1.0e-12 * scale) {
                return None;
            }
            coords.iter().map(|&t| (t - lo) / span).collect()
        }
    };
    Some(watson_u2_uniform(&u))
}

// ===========================================================================
// F2 — occupancy law: the SECOND half of the two-part certificate.
//
// Watson's `U²` tests the fitted coordinates against the atom's UNIFORM
// invariant measure. But uniformity is a property of the DATA's occupancy, NOT
// of the chart's honesty: a CORRECT circle whose data occupies only seven points
// (weekdays with cyclic adjacency) reads a highly non-uniform coordinate and so
// FAILS a uniform-null test — even though the chart is perfectly honest and the
// seven-point structure is exactly the thing we want to discover. Reporting that
// as a fidelity failure conflates "dishonest chart" with "discrete measure on an
// honest chart."
//
// The fix is to split the certificate:
//   * **chart honesty** — a pure property of the parameterization (unit-speed /
//     arc-length defect, the collapse floor): does the chart faithfully carry a
//     coordinate at all. Discrete occupancy does not touch this.
//   * **occupancy law** — WHAT measure the data draws from ON that honest chart:
//     `Uniform`, `Discrete{anchors}` (a finite set — the finite-set / cluster
//     alternative, `d_eff = anchors − 1`), or `Continuous` (a non-uniform but
//     spread density, e.g. a concentrated arc). This is adjudicated by evidence,
//     not by a p-value cut, so a circle-vs-clusters contest is raced per atom.
//
// The occupancy adjudication is a BIC (rank-aware Laplace-evidence) comparison
// across a small FIXED model-class enumeration — the SAME "discrete structure
// choice" pattern the topology / `K` / mixture ladders already use, not a grid
// search: the uniform density (0 free location parameters), a single wrapped
// Gaussian (the continuous unimodal / von-Mises-like alternative), and a
// `k`-anchor wrapped-Gaussian mixture for `k` on the anchor ladder. The winning
// class is the occupancy law; when a `k ≥ 2` anchor model wins, the atom carries
// a discrete measure of `k` anchors (`d_eff = k − 1`).
// ===========================================================================

/// The fixed anchor ladder swept for the discrete-occupancy rung. A discrete
/// structure choice (like [`MIXTURE_K_LADDER`](crate) / the topology ladder),
/// not a grid search — each `k` is priced by its own free-parameter count and
/// ranked by evidence. Includes `7` (weekday-cyclic) and `12` (month-cyclic).
pub const OCCUPANCY_ANCHOR_LADDER: &[usize] = &[2, 3, 4, 5, 6, 7, 9, 12];

/// The occupancy law of a fitted `d = 1` coordinate ON its honest chart: which
/// measure the data draws from. Adjudicated by evidence ([`classify_occupancy`]),
/// SEPARATELY from whether the chart itself is honest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLaw {
    /// The coordinate fills its manifold uniformly — the invariant measure. An
    /// occupied circle / interval.
    Uniform,
    /// The coordinate collapses onto a finite set of `anchors` points (a discrete
    /// measure — weekdays, categories). `d_eff = anchors − 1` is the rank charge
    /// the finite-set alternative carries into the race.
    Discrete { anchors: usize },
    /// The coordinate is non-uniform but continuously spread (a concentrated arc,
    /// a unimodal density) — neither uniform nor a finite anchor set.
    Continuous,
    /// Too few / degenerate coordinates to classify.
    Indeterminate,
}

impl OccupancyLaw {
    /// Lowercase label for the diagnostics payload.
    pub fn label(self) -> &'static str {
        match self {
            OccupancyLaw::Uniform => "uniform",
            OccupancyLaw::Discrete { .. } => "discrete",
            OccupancyLaw::Continuous => "continuous",
            OccupancyLaw::Indeterminate => "indeterminate",
        }
    }

    /// The number of anchors for a discrete occupancy (`0` otherwise).
    pub fn anchors(self) -> usize {
        match self {
            OccupancyLaw::Discrete { anchors } => anchors,
            _ => 0,
        }
    }

    /// The effective latent rank the occupancy contributes to the race charge:
    /// `anchors − 1` for a finite set (the categorical `t` has `anchors − 1`
    /// independent contrasts), `0` for the smooth / uniform laws whose rank the
    /// manifold dimension already carries.
    pub fn d_eff(self) -> usize {
        match self {
            OccupancyLaw::Discrete { anchors } => anchors.saturating_sub(1),
            _ => 0,
        }
    }
}

/// Classify the occupancy law of coordinates already folded onto the unit circle
/// `u ∈ [0, 1)` (a circle wraps modulo its period; an interval is range
/// normalized — both handled by [`coordinate_uniformity`]'s mapping) by a BIC
/// comparison across the fixed model-class enumeration `{uniform, one wrapped
/// Gaussian, k-anchor wrapped-Gaussian mixture for k on the anchor ladder}`. The
/// class with the lowest BIC (= highest rank-aware Laplace evidence) is the law:
/// a `k ≥ 2` anchor model ⟹ [`OccupancyLaw::Discrete`], a single wrapped Gaussian
/// ⟹ [`OccupancyLaw::Continuous`], the uniform density ⟹ [`OccupancyLaw::Uniform`].
pub fn classify_occupancy(u: &[f64]) -> OccupancyLaw {
    let n = u.len();
    if n < 4 {
        return OccupancyLaw::Indeterminate;
    }
    // Fold into [0, 1) defensively (callers already normalize, but a raw circle
    // coordinate may arrive un-wrapped).
    let mut pts: Vec<f64> = u
        .iter()
        .map(|&x| {
            let f = x - x.floor();
            if f >= 1.0 { 0.0 } else { f }
        })
        .collect();
    if pts.iter().any(|p| !p.is_finite()) {
        return OccupancyLaw::Indeterminate;
    }
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = n as f64;
    let ln_n = nf.ln();

    // Uniform density on the unit circle is `1`, so its per-point log-density is
    // `0` and its total loglik is `0`; it has no free location parameters.
    let bic_uniform = -2.0 * 0.0 + 0.0 * ln_n;

    // Resolution floor: with `n` points on the unit circle you cannot resolve a
    // cluster tighter than the mean spacing `1/n`, so a wrapped-Gaussian width is
    // floored at half that (a Nyquist-like, data-derived floor — not a knob).
    let sigma_floor = 1.0 / (2.0 * nf);

    // Single wrapped Gaussian: the continuous unimodal (von-Mises-like)
    // alternative. Two free parameters (mean, width).
    let single = wrapped_gaussian_mixture_bic(&pts, 1, sigma_floor, ln_n);

    let mut best_law = OccupancyLaw::Uniform;
    let mut best_bic = bic_uniform;
    if let Some(bic) = single {
        if bic < best_bic {
            best_bic = bic;
            best_law = OccupancyLaw::Continuous;
        }
    }
    for &k in OCCUPANCY_ANCHOR_LADDER {
        if k >= n {
            break;
        }
        if let Some(bic) = wrapped_gaussian_mixture_bic(&pts, k, sigma_floor, ln_n) {
            if bic < best_bic {
                best_bic = bic;
                best_law = OccupancyLaw::Discrete { anchors: k };
            }
        }
    }
    best_law
}

/// BIC of a `k`-anchor wrapped-Gaussian mixture fitted to sorted circle
/// coordinates `pts ∈ [0, 1)` by deterministic circular `k`-means (evenly spaced
/// init) plus a SHARED (pooled) wrapped-Gaussian width. Returns `None` when the
/// fit is degenerate. `BIC = −2·loglik + p·ln n`, `p = 2k` (`k` means, `k − 1`
/// weights, `1` shared width); lower is better.
fn wrapped_gaussian_mixture_bic(
    pts: &[f64],
    k: usize,
    sigma_floor: f64,
    ln_n: f64,
) -> Option<f64> {
    let n = pts.len();
    if k == 0 || k > n {
        return None;
    }
    // Deterministic circular k-means. Angles are the coordinates themselves on
    // the unit-circumference circle (period 1); circular distance wraps.
    let circ_dist = |a: f64, b: f64| -> f64 {
        let d = (a - b).rem_euclid(1.0);
        d.min(1.0 - d)
    };
    let mut means: Vec<f64> = (0..k).map(|j| (j as f64 + 0.5) / k as f64).collect();
    let mut assign = vec![0usize; n];
    for _ in 0..100 {
        let mut changed = false;
        for (i, &p) in pts.iter().enumerate() {
            let mut best_j = 0usize;
            let mut best_d = f64::INFINITY;
            for (j, &m) in means.iter().enumerate() {
                let d = circ_dist(p, m);
                if d < best_d {
                    best_d = d;
                    best_j = j;
                }
            }
            if assign[i] != best_j {
                assign[i] = best_j;
                changed = true;
            }
        }
        // Circular-mean update of each occupied cluster (resultant-vector angle).
        for (j, m) in means.iter_mut().enumerate() {
            let (mut sx, mut sy, mut cnt) = (0.0_f64, 0.0_f64, 0usize);
            for (i, &p) in pts.iter().enumerate() {
                if assign[i] == j {
                    let ang = std::f64::consts::TAU * p;
                    sx += ang.cos();
                    sy += ang.sin();
                    cnt += 1;
                }
            }
            if cnt > 0 && (sx * sx + sy * sy) > 0.0 {
                *m = (sy.atan2(sx) / std::f64::consts::TAU).rem_euclid(1.0);
            }
        }
        if !changed {
            break;
        }
    }

    // Per-cluster weight + a SHARED (pooled) wrapped-Gaussian width. A shared
    // width is essential: with a free PER-cluster variance, adding anchors beyond
    // the true count cheats — a spurious extra center splits a real cluster and
    // drives that sub-cluster's variance toward zero, buying unbounded likelihood
    // that the parameter penalty cannot claw back, so over-clustered `k` always
    // wins. Pooling the variance over ALL points means an extra anchor only
    // shaves the shared width slightly, so BIC stops at the `k` where genuine
    // gaps (not jitter) separate the anchors. Width floored at the resolution.
    let mut weights = vec![0.0_f64; k];
    let mut counts = vec![0usize; k];
    let mut total_ss = 0.0_f64;
    for (i, &p) in pts.iter().enumerate() {
        let j = assign[i];
        counts[j] += 1;
        let d = {
            let raw = (p - means[j]).rem_euclid(1.0);
            if raw > 0.5 { raw - 1.0 } else { raw }
        };
        total_ss += d * d;
    }
    for j in 0..k {
        weights[j] = counts[j] as f64 / n as f64;
    }
    let shared_sigma = (total_ss / n as f64).sqrt().max(sigma_floor);
    let sigmas = vec![shared_sigma; k];

    // Mixture loglik with a wrapped Gaussian per component (±1 wrap images are
    // ample for σ well below 0.5). Density is per unit circumference so it is
    // directly commensurable with the uniform density `1`.
    let inv_sqrt_2pi = 1.0 / (std::f64::consts::TAU).sqrt();
    let mut loglik = 0.0_f64;
    for &p in pts {
        let mut dens = 0.0_f64;
        for j in 0..k {
            if weights[j] <= 0.0 {
                continue;
            }
            let s = sigmas[j];
            let mut g = 0.0_f64;
            for m in -1..=1 {
                let d = p - means[j] + m as f64;
                g += (-0.5 * (d / s) * (d / s)).exp();
            }
            dens += weights[j] * inv_sqrt_2pi / s * g;
        }
        if !(dens > 0.0) {
            return None;
        }
        loglik += dens.ln();
    }
    if !loglik.is_finite() {
        return None;
    }
    // Free parameters: k means + (k−1) mixture weights + 1 shared variance = 2k.
    let p_free = (2 * k) as f64;
    Some(-2.0 * loglik + p_free * ln_n)
}

/// The per-atom coordinate-fidelity certificate: a reported, calibrated summary
/// of whether one fitted `d = 1` atom's latent coordinate is an honest reading
/// of its manifold. Produced by [`atom_coordinate_fidelity`]; `None` for atoms
/// without a `d = 1` circle/interval chart.
#[derive(Debug, Clone)]
pub struct AtomCoordinateFidelity {
    /// `"circle"` or `"interval"` — the invariant measure the uniformity is
    /// tested against.
    pub topology: &'static str,
    /// Watson's `U²` of the fitted coordinates against the uniform invariant
    /// measure (larger ⟺ less uniform). Rotation/reflection invariant.
    pub uniformity_statistic: f64,
    /// Closed-form asymptotic p-value of `uniformity_statistic`. Small ⟺ the
    /// coordinates are flagged non-uniform relative to the invariant measure.
    pub uniformity_p_value: f64,
    /// Arc-length (unit-speed) defect of the chart parameterization
    /// ([`crate::chart_canonicalization::chart_unit_speed_defect`]): speed
    /// coefficient of variation on a uniform latent grid, `0` ⟺ exactly
    /// arc-length. `NaN` when the chart-speed evaluation honest-skipped
    /// (degenerate chart).
    pub arclength_defect: f64,
    /// Number of fitted coordinates the uniformity statistic was computed from.
    pub n_coords: usize,
    /// The certified verdict on whether an honest coordinate is available and
    /// which one to read ([`AngleFidelityVerdict`]).
    pub verdict: AngleFidelityVerdict,
    /// `true` when the certificate provides an honest coordinate. `false` only
    /// for a collapsed / degenerate chart, where coordinate consumers must
    /// refuse rather than read the raw chart.
    pub certified: bool,
    /// The honest, pure-read arc-length coordinate `u_i = s(t_i)/L ∈ [0, 1)` for
    /// every fitted row, in atom-coordinate order — the coordinate every
    /// downstream angle/dose/adjacency claim should read in place of the
    /// gauge-arbitrary raw `t` (#2081). Computed regardless of whether the
    /// mutating canonicalization committed (it is a property of the fitted curve
    /// alone). `None` only when the chart is degenerate (arc length ill-defined).
    pub coords_u_arc: Option<Array1<f64>>,
    /// RMS over the fitted rows of the (circular, for a circle) distance between
    /// the raw normalized coordinate and its arc-length image `u_arc`, after the
    /// best rotation/reflection alignment of the residual `O(2)` gauge. `0` ⟺ the
    /// raw coordinate already IS the arc-length coordinate up to gauge; large ⟺
    /// the raw chart squishes arc length AT THE DATA ROWS (the #2081 pathology,
    /// measured on data rather than a grid). `NaN` when `u_arc` is unavailable.
    pub raw_arclength_defect_rms: f64,
    /// Max over the fitted rows of the same aligned raw-vs-`u_arc` distance.
    pub raw_arclength_defect_max: f64,
    /// `min ‖γ'‖ / mean ‖γ'‖` of the decoder curve on a uniform grid. Below the
    /// [`SAE_FLOW_DIFFEO_MIN_DET`] collapse floor drives the `Degenerate`
    /// verdict. `NaN` when the chart is degenerate.
    pub min_speed_over_mean: f64,
    /// `max ‖γ'‖ / mean ‖γ'‖` on the grid. `NaN` when degenerate.
    pub max_speed_over_mean: f64,
    /// RMS of `log(‖γ'‖/mean)` on the grid — scale-invariant log-speed spread.
    /// `NaN` when degenerate.
    pub log_speed_rms: f64,
    /// **Chart-honesty half of the certificate (F2):** `true` iff the chart
    /// itself faithfully carries a coordinate — a well-conditioned, non-collapsed
    /// parameterization (`verdict != Degenerate`). This is a property of the
    /// PARAMETERIZATION alone and is INDEPENDENT of how the data occupies it, so a
    /// correct circle whose data sits on seven points is still chart-honest.
    pub chart_honest: bool,
    /// **Occupancy-law half of the certificate (F2):** which measure the data
    /// draws from ON the honest chart — `"uniform"`, `"discrete"`, `"continuous"`,
    /// or `"indeterminate"` ([`OccupancyLaw`]). Adjudicated by evidence, NOT by
    /// the uniform-null p-value, so a discrete measure is reported as discrete
    /// occupancy rather than a chart failure.
    pub occupancy: &'static str,
    /// Number of anchors when `occupancy == "discrete"` (`0` otherwise) — the
    /// finite-set size the discrete measure collapses onto.
    pub occupancy_anchors: usize,
    /// The effective latent rank the occupancy contributes to the race charge:
    /// `anchors − 1` for a discrete measure, `0` for the smooth laws.
    pub occupancy_d_eff: usize,
}

/// Aggregate certificate adapter for the unified certificate ledger.
///
/// The full per-atom records remain in the typed `coordinate_fidelity` payload;
/// this adapter contributes the conservative dictionary-level claim to the
/// shared ledger: every eligible d=1 coordinate must have an honest reading.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateFidelityCertificate<'a> {
    pub atoms: &'a [Option<AtomCoordinateFidelity>],
}

impl<'a> CoordinateFidelityCertificate<'a> {
    pub fn new(atoms: &'a [Option<AtomCoordinateFidelity>]) -> Self {
        Self { atoms }
    }
}

/// Build the coordinate-fidelity certificate for one fitted atom, or `None` when
/// the atom has no `d = 1` circle/interval chart (higher-`d` / non-metric atoms,
/// a demoted homotopy, or a lost basis evaluator — the same gate the in-loop
/// unit-speed retraction uses, [`SaeManifoldTerm::d1_unit_speed_topology`]).
///
/// The row set mirrors the existing per-atom diagnostics (e.g. the curvature
/// bound): all of the atom's fitted coordinate rows,
/// `term.assignment.coords[atom_idx]`.
pub fn atom_coordinate_fidelity(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Option<AtomCoordinateFidelity>, String> {
    let Some(topology) = term.d1_unit_speed_topology(atom_idx) else {
        return Ok(None);
    };
    let coords = term.assignment.coords[atom_idx].as_matrix();
    if coords.ncols() != 1 {
        return Ok(None);
    }
    let row_coords = coords.column(0);
    let uniformity = coordinate_uniformity(row_coords, &topology);
    // Occupancy law (F2): classified from the SAME folded coordinates the
    // uniformity statistic reads, but adjudicated by evidence rather than the
    // uniform-null p-value. Reported separately from chart honesty so a discrete
    // measure on an honest chart is not read as a fidelity failure.
    let occupancy_law = fold_for_occupancy(row_coords, &topology)
        .map(|folded| classify_occupancy(&folded))
        .unwrap_or(OccupancyLaw::Indeterminate);
    let atom = &term.atoms[atom_idx];
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!("atom_coordinate_fidelity: atom {atom_idx} has no basis evaluator")
    })?;
    let defect = crate::chart_canonicalization::chart_unit_speed_defect(
        evaluator.as_ref(),
        atom.decoder_coefficients.view(),
        row_coords,
        &topology,
    )?;
    // The honest arc-length coordinate + speed profile, computed as a pure read
    // (ungated by the decoder-recomposition tolerance) — always reportable even
    // when the mutating canonicalization honestly refused.
    let reading = chart_arclength_coordinates(
        evaluator.as_ref(),
        atom.decoder_coefficients.view(),
        row_coords,
        &topology,
    )?;
    let topology_label = match topology {
        CanonicalChartTopology::Circle { .. } => "circle",
        CanonicalChartTopology::Interval => "interval",
    };
    let is_circle = matches!(topology, CanonicalChartTopology::Circle { .. });

    let (
        verdict,
        coords_u_arc,
        raw_arclength_defect_rms,
        raw_arclength_defect_max,
        min_speed_over_mean,
        max_speed_over_mean,
        log_speed_rms,
    ) = match reading {
        Some(r) if r.min_speed_over_mean > SAE_FLOW_DIFFEO_MIN_DET => {
            // A well-conditioned chart: raw t is honest iff already unit-speed,
            // otherwise the coordinate is recoverable via `u_arc`.
            let verdict = angle_fidelity_verdict(Some(&r));
            let (rms, max) =
                raw_vs_arclength_defect(row_coords, r.coords_u_arc.view(), &topology, is_circle);
            (
                verdict,
                Some(r.coords_u_arc),
                rms,
                max,
                r.min_speed_over_mean,
                r.max_speed_over_mean,
                r.log_speed_rms,
            )
        }
        // Collapsed chart (speed vanishes somewhere) or arc length ill-defined:
        // no faithful coordinate exists — refuse.
        Some(r) => (
            AngleFidelityVerdict::Degenerate,
            None,
            f64::NAN,
            f64::NAN,
            r.min_speed_over_mean,
            r.max_speed_over_mean,
            r.log_speed_rms,
        ),
        None => (
            AngleFidelityVerdict::Degenerate,
            None,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ),
    };

    Ok(Some(AtomCoordinateFidelity {
        topology: topology_label,
        uniformity_statistic: uniformity.as_ref().map(|u| u.statistic).unwrap_or(f64::NAN),
        uniformity_p_value: uniformity.as_ref().map(|u| u.p_value).unwrap_or(f64::NAN),
        arclength_defect: defect.unwrap_or(f64::NAN),
        n_coords: uniformity.as_ref().map(|u| u.n).unwrap_or(row_coords.len()),
        verdict,
        certified: verdict.certified(),
        coords_u_arc,
        raw_arclength_defect_rms,
        raw_arclength_defect_max,
        min_speed_over_mean,
        max_speed_over_mean,
        log_speed_rms,
        chart_honest: verdict.certified(),
        occupancy: occupancy_law.label(),
        occupancy_anchors: occupancy_law.anchors(),
        occupancy_d_eff: occupancy_law.d_eff(),
    }))
}

/// Fold the fitted `d = 1` coordinates onto the unit interval `[0, 1)` for the
/// occupancy classifier, using the SAME per-topology mapping as
/// [`coordinate_uniformity`] (circle wraps modulo its period; interval is range
/// normalized). Returns `None` when the mapping is ill-defined (non-finite
/// coordinate, non-positive period, collapsed interval range).
fn fold_for_occupancy(
    coords: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Option<Vec<f64>> {
    if coords.len() < 2 || coords.iter().any(|t| !t.is_finite()) {
        return None;
    }
    match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return None;
            }
            Some(coords.iter().map(|&t| t.rem_euclid(*period) / *period).collect())
        }
        CanonicalChartTopology::Interval => {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &t in coords.iter() {
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let span = hi - lo;
            let scale = lo.abs().max(hi.abs()).max(1.0);
            if !(span > 1.0e-12 * scale) {
                return None;
            }
            Some(coords.iter().map(|&t| (t - lo) / span).collect())
        }
    }
}

/// The (circular, for a circle) distance between the raw normalized coordinate
/// `t_i / span` and its arc-length image `u_i`, minimized over the residual
/// gauge — a base-point shift `c` and an orientation flip `s ∈ {+1, −1}` — and
/// summarized as `(rms, max)` over the rows. On a circle the residual gauge is
/// the full `O(2)` (rotation + reflection), so the best rotation is the circular
/// mean of `u_i − s·r_i`; on an interval it is reflection + translation, so the
/// best shift is the ordinary mean. `0` ⟺ the raw coordinate already equals the
/// arc-length coordinate up to that gauge (an honest, unit-speed chart at the
/// data rows).
fn raw_vs_arclength_defect(
    raw: ArrayView1<'_, f64>,
    u_arc: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
    is_circle: bool,
) -> (f64, f64) {
    let n = raw.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    // Raw coordinate normalized to `[0, 1)` (circle) / `[0, 1]` (interval),
    // matching the `u_arc` normalization.
    let r: Vec<f64> = match topology {
        CanonicalChartTopology::Circle { period } => {
            raw.iter().map(|&t| (t / period).rem_euclid(1.0)).collect()
        }
        CanonicalChartTopology::Interval => {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &t in raw.iter() {
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let span = hi - lo;
            if !(span > 0.0) {
                return (f64::NAN, f64::NAN);
            }
            raw.iter()
                .map(|&t| ((t - lo) / span).clamp(0.0, 1.0))
                .collect()
        }
    };

    let circ_dist = |a: f64, b: f64| -> f64 {
        let d = (a - b).rem_euclid(1.0);
        d.min(1.0 - d)
    };

    let mut best_rms = f64::INFINITY;
    let mut best_max = f64::INFINITY;
    for &s in &[1.0_f64, -1.0_f64] {
        // Best gauge offset c: circular mean of (u - s·r) on a circle, ordinary
        // mean on an interval.
        let c = if is_circle {
            let (mut sx, mut sy) = (0.0_f64, 0.0_f64);
            for (ui, ri) in u_arc.iter().zip(r.iter()) {
                let diff = ui - s * ri;
                let ang = std::f64::consts::TAU * diff;
                sx += ang.cos();
                sy += ang.sin();
            }
            sy.atan2(sx) / std::f64::consts::TAU
        } else {
            let mut acc = 0.0_f64;
            for (ui, ri) in u_arc.iter().zip(r.iter()) {
                acc += ui - s * ri;
            }
            acc / n as f64
        };
        let mut sum_sq = 0.0_f64;
        let mut max = 0.0_f64;
        for (ui, ri) in u_arc.iter().zip(r.iter()) {
            let aligned = s * ri + c;
            let d = if is_circle {
                circ_dist(*ui, aligned)
            } else {
                (ui - aligned).abs()
            };
            sum_sq += d * d;
            max = max.max(d);
        }
        let rms = (sum_sq / n as f64).sqrt();
        if rms < best_rms {
            best_rms = rms;
            best_max = max;
        }
    }
    (best_rms, best_max)
}

/// #2081 — basin preference at (near-)equal reconstruction EV: the seed-selection
/// tie-break that prices coordinate fidelity.
///
/// A candidate whose reconstruction EV is strictly better than the incumbent's
/// by more than `ev_tol` always wins on EV (and strictly worse always loses) —
/// EV remains the primary criterion, and this can never return a materially
/// worse-reconstructing basin. Within the `ev_tol` band the two basins are
/// EV-equivalent (`ev_tol` is the caller-supplied #1026 negligibility tolerance
/// [`crate::manifold::SAE_FINAL_EV_DEGRADATION_TOL`], a scale-invariant "0.1% of
/// variance" point — no fresh constant), so the tie is broken on the
/// coordinate-uniformity certificate: the candidate is preferred iff its
/// aggregate Watson `U²` is strictly LOWER (more uniform coordinates), because
/// EV provably does not certify coordinate fidelity. When either side has no
/// `d = 1` chart to compare (`None`), the tie-break is inert (the incumbent is
/// kept).
///
/// Lower `uniformity` = more uniform (Watson `U²`). Returns `false` for a
/// non-finite candidate EV.
pub fn prefer_candidate_basin(
    candidate_ev: f64,
    candidate_uniformity: Option<f64>,
    incumbent_ev: f64,
    incumbent_uniformity: Option<f64>,
    ev_tol: f64,
) -> bool {
    if !candidate_ev.is_finite() {
        return false;
    }
    if !incumbent_ev.is_finite() {
        // No finite incumbent to compare against: adopt any finite candidate.
        return true;
    }
    if candidate_ev > incumbent_ev + ev_tol {
        return true; // strictly better reconstruction
    }
    if incumbent_ev > candidate_ev + ev_tol {
        return false; // strictly worse reconstruction
    }
    // Near-equal EV: break the tie on the coordinate-uniformity certificate.
    match (candidate_uniformity, incumbent_uniformity) {
        (Some(candidate), Some(incumbent)) => candidate < incumbent,
        _ => false,
    }
}

impl SaeManifoldTerm {
    /// #2081 — aggregate chart-honesty score over the fit's `d = 1` atoms: the
    /// MEAN arc-length (unit-speed) DEFECT
    /// ([`crate::chart_canonicalization::chart_unit_speed_defect`]) across atoms
    /// that carry a `d = 1` circle/interval chart (LOWER ⟺ more arc-length-uniform
    /// parameterization). `None` when no atom yields a finite defect (no `d = 1`
    /// chart, or every such chart degenerate), which makes the seed-selection
    /// tie-break ([`prefer_candidate_basin`]) inert.
    ///
    /// It prices the arc-length defect — a PURE parameterization property measured
    /// on a uniform latent grid — rather than the raw-coordinate Watson `U²`
    /// occupancy statistic ([`coordinate_uniformity`]). The two are NOT
    /// interchangeable for seed selection (the F2 split): Watson `U²` conflates
    /// data occupancy with chart honesty, so a WARPED chart that spreads a
    /// genuinely clustered coordinate into a uniform-looking raw distribution reads
    /// a LOWER `U²` than the honest chart it should lose to — i.e. occupancy
    /// uniformity can prefer the dishonest chart at equal EV, the exact #2081
    /// failure. The arc-length defect isolates the pathology EV cannot see (a chart
    /// that squishes arc length at high reconstruction EV) independent of where the
    /// data falls, so it is the correct quantity for the tie-break to price. Lower
    /// is better for BOTH statistics, so the [`prefer_candidate_basin`] ordering
    /// (candidate `<` incumbent wins the tie) is unchanged.
    ///
    /// Evaluates each `d = 1` atom's basis on the arc-length quadrature grid, so it
    /// is heavier than the coordinate-only occupancy read; it is still called only
    /// at accepted-iterate incumbent-comparison boundaries (never inside a line
    /// search), where one band-limited grid evaluation per atom is negligible
    /// against the joint Newton assembly.
    pub(crate) fn coordinate_uniformity_aggregate(&self) -> Option<f64> {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for atom_idx in 0..self.atoms.len() {
            let Some(topology) = self.d1_unit_speed_topology(atom_idx) else {
                continue;
            };
            let coords = self.assignment.coords[atom_idx].as_matrix();
            if coords.ncols() != 1 {
                continue;
            }
            let atom = &self.atoms[atom_idx];
            let defect = atom.basis_evaluator.as_ref().and_then(|evaluator| {
                crate::chart_canonicalization::chart_unit_speed_defect(
                    evaluator.as_ref(),
                    atom.decoder_coefficients.view(),
                    coords.column(0),
                    &topology,
                )
                .ok()
                .flatten()
            });
            if let Some(d) = defect {
                if d.is_finite() {
                    sum += d;
                    count += 1;
                }
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }
}

#[cfg(test)]
mod coordinate_fidelity_tests {
    use super::*;
    use crate::manifold::{SAE_FINAL_EV_DEGRADATION_TOL, SaeBasisEvaluator};
    use ndarray::{Array1, Array2, Array3, Array4, Array5, ArrayView2};

    /// A minimal circle-harmonic evaluator for the arc-length-defect tests:
    /// `Φ(t) = [cos 2πt, sin 2πt, cos 4πt, sin 4πt, …]` up to `harmonics`
    /// frequencies (period `1.0`, fraction-of-period convention). Enough to build
    /// unit-speed and non-uniform-speed circle decoders without the production
    /// evaluators.
    #[derive(Debug)]
    struct CircleHarmonicEvaluator {
        harmonics: usize,
    }

    impl SaeBasisEvaluator for CircleHarmonicEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            let m = 2 * self.harmonics;
            let mut phi = Array2::<f64>::zeros((n, m));
            let mut jet = Array3::<f64>::zeros((n, m, 1));
            let tau = std::f64::consts::TAU;
            for i in 0..n {
                let t = coords[[i, 0]];
                for h in 1..=self.harmonics {
                    let w = tau * h as f64;
                    let c = 2 * (h - 1);
                    let s = c + 1;
                    phi[[i, c]] = (w * t).cos();
                    phi[[i, s]] = (w * t).sin();
                    jet[[i, c, 0]] = -w * (w * t).sin();
                    jet[[i, s, 0]] = w * (w * t).cos();
                }
            }
            Ok((phi, jet))
        }

        fn second_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<Array4<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "CircleHarmonicEvaluator::second_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<Array5<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "CircleHarmonicEvaluator::third_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }
    }

    #[derive(Debug)]
    struct IntervalLinearEvaluator;

    impl SaeBasisEvaluator for IntervalLinearEvaluator {
        fn evaluate(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Result<(Array2<f64>, Array3<f64>), String> {
            let n = coords.nrows();
            let mut phi = Array2::<f64>::zeros((n, 2));
            let mut jet = Array3::<f64>::zeros((n, 2, 1));
            for i in 0..n {
                phi[[i, 0]] = 1.0;
                phi[[i, 1]] = coords[[i, 0]];
                jet[[i, 1, 0]] = 1.0;
            }
            Ok((phi, jet))
        }

        fn second_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<Array4<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "IntervalLinearEvaluator::second_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }

        fn third_jet_dyn(
            &self,
            coords: ArrayView2<'_, f64>,
        ) -> Option<Result<Array5<f64>, String>> {
            if coords.ncols() != 1 {
                return Some(Err(format!(
                    "IntervalLinearEvaluator::third_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                )));
            }
            None
        }
    }

    fn circle() -> CanonicalChartTopology {
        CanonicalChartTopology::Circle { period: 1.0 }
    }

    fn interval() -> CanonicalChartTopology {
        CanonicalChartTopology::Interval
    }

    /// The closed-form Watson p-value must reproduce the classical tabulated
    /// critical values — this validates the derived flag against published
    /// statistics, not against a tuned constant.
    #[test]
    fn watson_pvalue_matches_tabulated_critical_values() {
        // 5% critical value 0.187, 1% critical value 0.267 (Stephens 1970).
        let p05 = watson_u2_pvalue(0.187);
        let p01 = watson_u2_pvalue(0.267);
        assert!(
            (p05 - 0.05).abs() < 5.0e-3,
            "p(U²=0.187) must be ≈0.05, got {p05}"
        );
        assert!(
            (p01 - 0.01).abs() < 5.0e-3,
            "p(U²=0.267) must be ≈0.01, got {p01}"
        );
        // Monotone decreasing in the statistic.
        assert!(watson_u2_pvalue(0.05) > watson_u2_pvalue(0.15));
        assert!(watson_u2_pvalue(0.15) > watson_u2_pvalue(0.30));
    }

    /// CALIBRATION: uniform planted angles land in the null range (not flagged);
    /// bunched angles are flagged. This is the item-3 calibration requirement.
    #[test]
    fn uniformity_statistic_is_calibrated() {
        let n = 240;
        // Uniform: equally-spaced coordinates on the circle — the invariant
        // measure. U² sits well below the 5% critical value; p is high.
        let uniform: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let uu = coordinate_uniformity(Array1::from(uniform.clone()).view(), &circle()).unwrap();
        assert!(
            uu.statistic < 0.187,
            "uniform angles must fall below the 5% critical value, got U²={}",
            uu.statistic
        );
        assert!(
            uu.p_value > 0.10,
            "uniform angles must not be flagged, p={}",
            uu.p_value
        );
        // Bunched: every angle compressed into a 5% arc — the #2081 pathology.
        let bunched: Vec<f64> = (0..n).map(|i| 0.05 * (i as f64 / n as f64)).collect();
        let bu = coordinate_uniformity(Array1::from(bunched).view(), &circle()).unwrap();
        assert!(
            bu.statistic > 0.267,
            "bunched angles must exceed the 1% critical value, got U²={}",
            bu.statistic
        );
        assert!(
            bu.p_value < 0.01,
            "bunched angles must be flagged, p={}",
            bu.p_value
        );
        // The bunched chart reads a MORE non-uniform coordinate than the honest one.
        assert!(bu.statistic > uu.statistic);
    }

    /// Watson's `U²` is invariant to the circle's residual `O(2)` gauge: a
    /// rotation of the base point and a reflection of orientation leave it
    /// unchanged (so the statistic is not an artifact of the reading convention —
    /// the exact fragility the weekday-basis data point is about).
    #[test]
    fn uniformity_is_rotation_and_reflection_invariant() {
        // A deterministic non-uniform sample so the invariance is non-trivial.
        let base: Vec<f64> = (0..97)
            .map(|i| {
                let x = (i as f64 * 0.61803398875).fract();
                // Squash toward 0 to make it genuinely non-uniform.
                x * x
            })
            .collect();
        let u0 = watson_u2_uniform(&base).statistic;
        let rotated: Vec<f64> = base.iter().map(|&x| (x + 0.37).rem_euclid(1.0)).collect();
        let reflected: Vec<f64> = base.iter().map(|&x| (1.0 - x).rem_euclid(1.0)).collect();
        let ur = watson_u2_uniform(&rotated).statistic;
        let uf = watson_u2_uniform(&reflected).statistic;
        assert!(
            (u0 - ur).abs() < 1e-9,
            "rotation must not change U²: {u0} vs {ur}"
        );
        assert!(
            (u0 - uf).abs() < 1e-9,
            "reflection must not change U²: {u0} vs {uf}"
        );
    }

    /// The arc-length defect is ≈0 for a unit-speed circle (pure first harmonic,
    /// constant speed) and strictly positive for a non-uniform-speed chart (a
    /// second harmonic mixed in) — the pure-parameterization signal EV cannot see.
    #[test]
    fn arclength_defect_flags_non_unit_speed_chart() {
        let ev = CircleHarmonicEvaluator { harmonics: 2 };
        // Pure first harmonic, radius R: γ(t) = R(cos 2πt, sin 2πt), speed 2πR.
        let mut unit = Array2::<f64>::zeros((4, 2));
        unit[[0, 0]] = 1.3; // cos → x
        unit[[1, 1]] = 1.3; // sin → y
        let row_coords = Array1::linspace(0.0, 1.0, 32);
        let d_unit = crate::chart_canonicalization::chart_unit_speed_defect(
            &ev,
            unit.view(),
            row_coords.view(),
            &circle(),
        )
        .unwrap()
        .expect("unit-speed circle must produce a defect");
        assert!(
            d_unit < 1e-6,
            "a constant-speed circle must have ~zero arc-length defect, got {d_unit}"
        );
        // Add a second-harmonic component: the speed field is no longer constant.
        let mut wobbly = unit.clone();
        wobbly[[2, 0]] = 0.6; // cos 4πt → x
        wobbly[[3, 1]] = 0.6; // sin 4πt → y
        let d_wobbly = crate::chart_canonicalization::chart_unit_speed_defect(
            &ev,
            wobbly.view(),
            row_coords.view(),
            &circle(),
        )
        .unwrap()
        .expect("wobbly circle must produce a defect");
        assert!(
            d_wobbly > 1e-2,
            "a non-unit-speed chart must have a positive arc-length defect, got {d_wobbly}"
        );
    }

    /// CONTRACT: the declining higher-jet impls are a *capability declaration*
    /// (`None` = "no analytic jet"), not a silent stub. A d = 1 evaluator must
    /// still validate its coordinate shape and surface a wrong-dimension call as
    /// an error rather than ignore the argument. This guards against the higher
    /// jets regressing back to an unused-`_coords` body (which the whole-workspace
    /// ban-scanner rejects, and which cold release builds fail on — #2092): if the
    /// argument were ignored, the malformed-shape probe below would silently
    /// return `None` instead of `Some(Err(..))`.
    #[test]
    fn declining_higher_jets_enforce_d1_coords_contract() {
        let ev = CircleHarmonicEvaluator { harmonics: 3 };
        // Well-formed d = 1 coords: both higher jets decline (no analytic form).
        let good = Array2::<f64>::zeros((5, 1));
        assert!(
            ev.second_jet_dyn(good.view()).is_none(),
            "d = 1 coords must decline the second jet with None"
        );
        assert!(
            ev.third_jet_dyn(good.view()).is_none(),
            "d = 1 coords must decline the third jet with None"
        );
        // Malformed coords (d = 2): the evaluator must consume the argument and
        // reject the contract violation, not silently decline.
        let bad = Array2::<f64>::zeros((5, 2));
        let second = ev
            .second_jet_dyn(bad.view())
            .expect("wrong-dimension coords must not silently decline the second jet");
        assert!(
            second.is_err(),
            "second_jet_dyn must reject d != 1 coords, got {second:?}"
        );
        let third = ev
            .third_jet_dyn(bad.view())
            .expect("wrong-dimension coords must not silently decline the third jet");
        assert!(
            third.is_err(),
            "third_jet_dyn must reject d != 1 coords, got {third:?}"
        );
    }

    /// TIE-BREAK: the raw EV comparison is preserved, and at (near-)equal EV the
    /// more-uniform-coordinate candidate is preferred.
    #[test]
    fn prefer_candidate_basin_prices_ev_then_uniformity() {
        let tol = SAE_FINAL_EV_DEGRADATION_TOL;
        // Strictly better EV always wins, regardless of uniformity.
        assert!(prefer_candidate_basin(
            0.90,
            Some(0.5),
            0.80,
            Some(0.01),
            tol
        ));
        // Strictly worse EV always loses, regardless of uniformity.
        assert!(!prefer_candidate_basin(
            0.80,
            Some(0.01),
            0.90,
            Some(0.5),
            tol
        ));
        // Near-equal EV: lower U² (more uniform) wins.
        assert!(prefer_candidate_basin(
            0.90,
            Some(0.02),
            0.9005,
            Some(0.20),
            tol
        ));
        // Near-equal EV: higher U² loses.
        assert!(!prefer_candidate_basin(
            0.90,
            Some(0.20),
            0.9005,
            Some(0.02),
            tol
        ));
        // Near-equal EV, equal uniformity: keep incumbent (no thrash).
        assert!(!prefer_candidate_basin(
            0.90,
            Some(0.05),
            0.90,
            Some(0.05),
            tol
        ));
        // No certificate on either side: tie-break inert.
        assert!(!prefer_candidate_basin(0.90, None, 0.90, Some(0.05), tol));
        // Non-finite candidate EV never preferred.
        assert!(!prefer_candidate_basin(
            f64::NAN,
            Some(0.0),
            0.5,
            Some(0.5),
            tol
        ));
    }

    /// PLANTED-CIRCLE tie-break: two seeds reach equal EV but read different
    /// angle fidelity — one reads a uniform (honest, arc-length) angle, the other
    /// a bunched (compressed) angle. The tie-break must pick the more uniform one.
    #[test]
    fn planted_circle_tie_break_picks_the_more_uniform_seed() {
        let n = 200;
        let tol = SAE_FINAL_EV_DEGRADATION_TOL;
        // True planted angles are uniform on the ring. Seed A reads them honestly
        // (uniform coordinate); seed B reads them through a squished chart that
        // compresses the same ring into a fraction of the coordinate span.
        let honest: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let squished: Vec<f64> = honest.iter().map(|&u| 0.5 * u * u + 0.25 * u).collect();
        let ua = coordinate_uniformity(Array1::from(honest).view(), &circle())
            .unwrap()
            .statistic;
        let ub = coordinate_uniformity(Array1::from(squished).view(), &circle())
            .unwrap()
            .statistic;
        assert!(
            ua < ub,
            "honest chart must read a more uniform angle: {ua} vs {ub}"
        );
        // Both seeds reconstruct the ring equally well (EV within the negligibility
        // band): the tie-break must prefer the honest (uniform) seed over the
        // squished incumbent, and never the reverse.
        let ev = 0.926;
        assert!(
            prefer_candidate_basin(ev, Some(ua), ev, Some(ub), tol),
            "the honest seed must be preferred at equal EV"
        );
        assert!(
            !prefer_candidate_basin(ev, Some(ub), ev, Some(ua), tol),
            "the squished seed must NOT displace the honest incumbent at equal EV"
        );
    }

    /// The honest arc-length coordinate is the pure-read complement to the raw
    /// chart: on an already-unit-speed circle it equals the raw coordinate, the
    /// speed profile is flat, the verdict certifies the raw reading is honest,
    /// and the raw-vs-`u_arc` defect is ~zero.
    #[test]
    fn arclength_reading_is_identity_on_a_unit_speed_circle() {
        use crate::chart_canonicalization::chart_arclength_coordinates;
        let ev = CircleHarmonicEvaluator { harmonics: 2 };
        let mut unit = Array2::<f64>::zeros((4, 2));
        unit[[0, 0]] = 1.3; // cos 2πt → x
        unit[[1, 1]] = 1.3; // sin 2πt → y
        let rows = Array1::linspace(0.0, 0.97, 40);
        let reading = chart_arclength_coordinates(&ev, unit.view(), rows.view(), &circle())
            .unwrap()
            .expect("unit-speed circle yields a reading");
        // Constant speed ⇒ u_arc(t) = t (mod 1) and the speed profile is flat.
        for (i, &t) in rows.iter().enumerate() {
            let d = (reading.coords_u_arc[i] - t).rem_euclid(1.0);
            let circ = d.min(1.0 - d);
            assert!(
                circ < 1e-6,
                "u_arc must equal raw t on a unit-speed circle: {circ}"
            );
        }
        assert!(
            reading.speed_cv < 1e-6,
            "flat speed ⇒ ~zero CV, got {}",
            reading.speed_cv
        );
        assert!((reading.min_speed_over_mean - 1.0).abs() < 1e-6);
        assert!((reading.max_speed_over_mean - 1.0).abs() < 1e-6);
        assert_eq!(
            angle_fidelity_verdict(Some(&reading)),
            AngleFidelityVerdict::ArcLengthHonest
        );
        let (rms, max) =
            raw_vs_arclength_defect(rows.view(), reading.coords_u_arc.view(), &circle(), true);
        assert!(
            rms < 1e-6 && max < 1e-6,
            "honest chart has ~zero raw defect: rms={rms} max={max}"
        );
    }

    /// The pure-read arclength coordinate also handles interval charts: for a
    /// linear decoded segment the speed is constant, so the reported coordinate
    /// is exactly the affine normalization of the fitted interval.
    #[test]
    fn arclength_reading_is_affine_on_a_linear_interval() {
        use crate::chart_canonicalization::chart_arclength_coordinates;
        let ev = IntervalLinearEvaluator;
        let mut decoder = Array2::<f64>::zeros((2, 2));
        decoder[[0, 0]] = 0.7;
        decoder[[0, 1]] = -0.2;
        decoder[[1, 0]] = 1.5;
        decoder[[1, 1]] = -0.5;
        let rows = Array1::linspace(-0.4, 1.3, 37);
        let reading = chart_arclength_coordinates(&ev, decoder.view(), rows.view(), &interval())
            .unwrap()
            .expect("linear interval yields a reading");
        let lo = rows[0];
        let span = rows[rows.len() - 1] - lo;
        for (i, &t) in rows.iter().enumerate() {
            let expected = (t - lo) / span;
            assert!(
                (reading.coords_u_arc[i] - expected).abs() < 1e-9,
                "linear interval u_arc must be affine: got {}, expected {}",
                reading.coords_u_arc[i],
                expected
            );
        }
        assert!(reading.speed_cv < 1e-9, "linear segment has constant speed");
        assert_eq!(
            angle_fidelity_verdict(Some(&reading)),
            AngleFidelityVerdict::ArcLengthHonest
        );
    }

    /// EV-INSUFFICIENCY (the #2081 headline): a wobbly (non-unit-speed) circle
    /// reconstructs its ring at high EV while reading a squished coordinate. The
    /// pure-read arc-length coordinate is computed regardless (it is a property
    /// of the fitted curve alone), the verdict flags the raw chart as
    /// recoverable-via-arclength rather than silently trusting the raw `t`, and
    /// `u_arc` materially differs from the raw coordinate at the data rows — the
    /// correction reconstruction EV provably cannot make.
    #[test]
    fn arclength_reading_recovers_and_certifies_a_wobbly_circle() {
        use crate::chart_canonicalization::chart_arclength_coordinates;
        let ev = CircleHarmonicEvaluator { harmonics: 2 };
        let mut wobbly = Array2::<f64>::zeros((4, 2));
        wobbly[[0, 0]] = 1.3;
        wobbly[[1, 1]] = 1.3;
        wobbly[[2, 0]] = 0.2; // cos 4πt → x (a mild, well-conditioned wobble)
        wobbly[[3, 1]] = 0.2; // sin 4πt → y
        let rows = Array1::linspace(0.0, 0.98, 64);
        let reading = chart_arclength_coordinates(&ev, wobbly.view(), rows.view(), &circle())
            .unwrap()
            .expect("wobbly circle yields a reading");
        assert!(
            reading.speed_cv > 1e-2,
            "wobbly chart must have a positive speed CV, got {}",
            reading.speed_cv
        );
        assert!(reading.min_speed_over_mean < 1.0 && reading.max_speed_over_mean > 1.0);
        // Stays a well-conditioned diffeomorphism ⇒ RECOVERABLE, not degenerate.
        assert!(reading.min_speed_over_mean > SAE_FLOW_DIFFEO_MIN_DET);
        assert_eq!(
            angle_fidelity_verdict(Some(&reading)),
            AngleFidelityVerdict::RecoverableViaArcLength
        );
        let (rms, _max) =
            raw_vs_arclength_defect(rows.view(), reading.coords_u_arc.view(), &circle(), true);
        assert!(
            rms > 1e-2,
            "u_arc must materially differ from raw t on a squished chart, got rms={rms}"
        );
    }

    /// A chart whose decoded speed COLLAPSES (a cusp where `‖γ'‖ → 0`) has no
    /// faithful coordinate: the arc-length map has a flat spot, so the verdict is
    /// `Degenerate` — a coordinate consumer must refuse rather than read it. Built
    /// from a real decoder: a second harmonic of equal amplitude to the first
    /// makes the tangent vanish at `t = 1/2`.
    #[test]
    fn arclength_reading_flags_a_cusped_chart_degenerate() {
        use crate::chart_canonicalization::chart_arclength_coordinates;
        let ev = CircleHarmonicEvaluator { harmonics: 2 };
        let mut cusped = Array2::<f64>::zeros((4, 2));
        cusped[[0, 0]] = 1.0; // R = 1
        cusped[[1, 1]] = 1.0;
        cusped[[2, 0]] = 0.5; // 4π·0.5 = 2π·1.0 ⇒ effective 2nd amp = R ⇒ cusp
        cusped[[3, 1]] = 0.5;
        let rows = Array1::linspace(0.0, 0.98, 64);
        let reading = chart_arclength_coordinates(&ev, cusped.view(), rows.view(), &circle())
            .unwrap()
            .expect("a cusped-but-finite chart still yields a reading");
        assert!(
            reading.min_speed_over_mean < SAE_FLOW_DIFFEO_MIN_DET,
            "a cusped chart must have a collapsing min speed, got {}",
            reading.min_speed_over_mean
        );
        assert_eq!(
            angle_fidelity_verdict(Some(&reading)),
            AngleFidelityVerdict::Degenerate
        );
    }

    /// The verdict keys off the fit's OWN dimensionless invariants, not fresh
    /// tuned constants: the diffeomorphism collapse floor `SAE_FLOW_DIFFEO_MIN_DET`
    /// and the in-loop retraction tolerance `UNIT_SPEED_INLOOP_DEFECT_TOL`.
    #[test]
    fn angle_fidelity_verdict_uses_derived_thresholds() {
        use crate::chart_canonicalization::{ChartArcLengthReading, UNIT_SPEED_INLOOP_DEFECT_TOL};
        let mk = |speed_cv: f64, min_over: f64, max_over: f64| ChartArcLengthReading {
            coords_u_arc: Array1::zeros(1),
            speed_cv,
            log_speed_rms: 0.0,
            min_speed_over_mean: min_over,
            max_speed_over_mean: max_over,
            total_arc_length: 1.0,
        };
        // Below the retraction tol ⇒ raw t already IS the arc-length coordinate.
        assert_eq!(
            angle_fidelity_verdict(Some(&mk(0.1 * UNIT_SPEED_INLOOP_DEFECT_TOL, 1.0, 1.0))),
            AngleFidelityVerdict::ArcLengthHonest
        );
        // Non-uniform speed but well above the collapse floor ⇒ recoverable.
        assert_eq!(
            angle_fidelity_verdict(Some(&mk(0.3, 2.0 * SAE_FLOW_DIFFEO_MIN_DET, 1.8))),
            AngleFidelityVerdict::RecoverableViaArcLength
        );
        // Min speed below the collapse floor ⇒ degenerate (refuse).
        assert_eq!(
            angle_fidelity_verdict(Some(&mk(0.3, 0.5 * SAE_FLOW_DIFFEO_MIN_DET, 3.0))),
            AngleFidelityVerdict::Degenerate
        );
        // No reading at all ⇒ degenerate.
        assert_eq!(
            angle_fidelity_verdict(None),
            AngleFidelityVerdict::Degenerate
        );
        assert!(AngleFidelityVerdict::ArcLengthHonest.certified());
        assert!(AngleFidelityVerdict::RecoverableViaArcLength.certified());
        assert!(!AngleFidelityVerdict::Degenerate.certified());
    }

    /// The raw-vs-`u_arc` defect is invariant to the circle's residual `O(2)`
    /// gauge (rotation + reflection) — it aligns before measuring — so it reports
    /// the genuine parameterization discrepancy, not the reading convention.
    #[test]
    fn raw_vs_arclength_defect_is_gauge_invariant() {
        // A deterministic non-uniform u_arc against a uniform raw grid.
        let n = 80;
        let raw = Array1::linspace(0.0, 1.0 - 1.0 / n as f64, n);
        let u_arc = Array1::from_iter(raw.iter().map(|&t| (0.5 * t * t + 0.5 * t).rem_euclid(1.0)));
        let (rms0, _) = raw_vs_arclength_defect(raw.view(), u_arc.view(), &circle(), true);
        // Rotate the raw base point and reflect its orientation: both are the
        // circle's residual gauge, so the aligned defect must not change.
        let rotated = Array1::from_iter(raw.iter().map(|&t| (t + 0.31).rem_euclid(1.0)));
        let reflected = Array1::from_iter(raw.iter().map(|&t| (1.0 - t).rem_euclid(1.0)));
        let (rms_rot, _) = raw_vs_arclength_defect(rotated.view(), u_arc.view(), &circle(), true);
        let (rms_ref, _) = raw_vs_arclength_defect(reflected.view(), u_arc.view(), &circle(), true);
        assert!(
            (rms0 - rms_rot).abs() < 1e-9,
            "rotation must not change the defect: {rms0} vs {rms_rot}"
        );
        assert!(
            (rms0 - rms_ref).abs() < 1e-9,
            "reflection must not change the defect: {rms0} vs {rms_ref}"
        );
    }

    // ---- F2: occupancy law (chart honesty vs occupancy split) ---------------

    /// A deterministic low-discrepancy sequence on `[0, 1)` (van der Corput,
    /// base 2) so the occupancy tests need no RNG.
    fn vdc(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let (mut x, mut denom, mut k) = (0.0_f64, 2.0_f64, i + 1);
                while k > 0 {
                    x += (k & 1) as f64 / denom;
                    denom *= 2.0;
                    k >>= 1;
                }
                x
            })
            .collect()
    }

    #[test]
    fn uniform_occupancy_is_classified_uniform() {
        // A uniformly-occupied circle: an even grid. The uniform density (0 free
        // params) must win the BIC race over any anchor mixture.
        let n = 400;
        let u: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        assert_eq!(classify_occupancy(&u), OccupancyLaw::Uniform);
    }

    #[test]
    fn seven_point_cyclic_is_classified_discrete() {
        // Weekday-cyclic occupancy: 7 anchors at k/7, each realized ~100 times
        // with sub-resolution jitter. The chart (a circle) is honest; the
        // OCCUPANCY is discrete — the k=7 anchor model must win by evidence, and
        // its rank charge d_eff must be 6.
        let per = 100;
        let jit = vdc(7 * per);
        let mut u = Vec::with_capacity(7 * per);
        for i in 0..(7 * per) {
            let anchor = (i % 7) as f64 / 7.0;
            // jitter within ±0.5% of the circle — far below the 1/7 spacing.
            u.push((anchor + 0.005 * (jit[i] - 0.5)).rem_euclid(1.0));
        }
        let law = classify_occupancy(&u);
        assert_eq!(law, OccupancyLaw::Discrete { anchors: 7 }, "{law:?}");
        assert_eq!(law.d_eff(), 6);
        assert_eq!(law.anchors(), 7);
    }

    #[test]
    fn concentrated_arc_is_classified_continuous() {
        // A single concentrated arc (unimodal, spread) is neither uniform nor a
        // finite anchor set: the single wrapped Gaussian must win, so the
        // occupancy law is Continuous.
        let n = 1200;
        let base = vdc(n);
        // A Bates(3) draw (mean of three low-discrepancy uniforms) is a genuinely
        // bell-shaped, single-moded density centered at 0.5 — a concentrated arc,
        // not a finite anchor set. Scale toward the center so it stays a spread
        // (not sub-resolution) continuous bump.
        let u: Vec<f64> = (0..n / 3)
            .map(|i| {
                let m = (base[3 * i] + base[3 * i + 1] + base[3 * i + 2]) / 3.0;
                (0.5 + 0.6 * (m - 0.5)).rem_euclid(1.0)
            })
            .collect();
        let law = classify_occupancy(&u);
        assert_eq!(law, OccupancyLaw::Continuous, "{law:?}");
    }

    #[test]
    fn discrete_occupancy_passes_chart_honesty_conceptually() {
        // The core F2 claim as a unit fact: a discrete occupancy is reported as
        // discrete WITHOUT implying the chart is dishonest. `chart_honest` keys
        // off the arc-length verdict only; `occupancy` keys off the measure. The
        // two are independent, so a discrete-occupancy verdict never forces
        // chart_honest=false. Here we assert the classifier isolates occupancy
        // (a data property) from any chart notion.
        let per = 80;
        let mut u = Vec::new();
        for i in 0..(3 * per) {
            u.push((i % 3) as f64 / 3.0 + 0.002 * ((i as f64).sin()));
        }
        let law = classify_occupancy(&u);
        assert!(
            matches!(law, OccupancyLaw::Discrete { anchors: 3 }),
            "3-point occupancy should be discrete, got {law:?}"
        );
        // d_eff of a 3-anchor discrete measure is 2 (three categories → two
        // contrasts).
        assert_eq!(law.d_eff(), 2);
    }
}
