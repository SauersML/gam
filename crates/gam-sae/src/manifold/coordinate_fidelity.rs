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
//! by evidence via [`classify_occupancy_weighted`], NOT by the p-value). A discrete
//! measure on an honest chart passes chart-honesty and is reported as discrete
//! occupancy (`d_eff = anchors − 1`) — the finite-set alternative in the race.
//!
//! The seed-selection tie-break ([`prefer_candidate_basin`]) prices the
//! uniformity statistic: at (near-)equal reconstruction EV — "near" derived from
//! the existing #1026 EV negligibility band
//! [`crate::manifold::SAE_FINAL_EV_DEGRADATION_TOL`], not a fresh constant — the
//! more-uniform-coordinate basin wins, because EV alone provably cannot break
//! that tie.

use gam_solve::exact_jet_objective::certified_newton_minimum;
use ndarray::{Array1, Array2, ArrayView1};
use opt::{Bounds, DecrementBands, ObjectiveEvalError, SecondOrderSample, accumulation_growth};

use crate::chart_canonicalization::{
    CanonicalChartTopology, ChartArcLengthReading, SAE_FLOW_DIFFEO_MIN_DET,
    UNIT_SPEED_INLOOP_DEFECT_TOL, chart_arclength_coordinates,
};

use super::{SaeManifoldTerm, SupportMeasure};

#[cfg(test)]
#[path = "coordinate_fidelity_recovery_tests.rs"]
mod recovered_collapse_tests;

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
pub(crate) fn angle_fidelity_verdict(reading: Option<&ChartArcLengthReading>) -> AngleFidelityVerdict {
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

/// Geometry-appropriate uniformity statistic for a one-dimensional chart.
/// Circles use rotation-invariant Watson `U²`; intervals use the ordinary
/// (non-wrapping) Kolmogorov--Smirnov distance on `[0, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct WatsonUniformity {
    /// Watson `U²` for a circle, or two-sided KS distance for an interval.
    pub statistic: f64,
    /// Closed-form Watson upper-tail p-value for circles. `None` for intervals:
    /// the interval endpoints are estimated from this same sample, so a
    /// known-boundary KS p-value would not be calibrated. The interval statistic
    /// remains a descriptive occupancy diagnostic until endpoint uncertainty is
    /// carried by the chart schema.
    pub p_value: Option<f64>,
    /// Number of coordinates the statistic was computed from.
    pub n: usize,
}

/// Non-circular KS distance of range-normalized interval coordinates. The value
/// `1.0` is retained as the right endpoint; it is never folded to zero.
fn interval_uniformity(
    u: &[f64],
    weights: Option<ArrayView1<'_, f64>>,
) -> Option<WatsonUniformity> {
    let mut pairs: Vec<(f64, f64)> = u
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, x)| {
            let w = weights.map_or(1.0, |wv| wv[i]);
            (x.is_finite() && w.is_finite() && w > 0.0).then_some((x.clamp(0.0, 1.0), w))
        })
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mass: f64 = pairs.iter().map(|(_, w)| *w).sum();
    if !(mass > 0.0) {
        return None;
    }
    let mut cumulative = 0.0_f64;
    let mut d = 0.0_f64;
    for (x, w) in pairs.iter().copied() {
        let before = cumulative / mass;
        cumulative += w;
        let after = cumulative / mass;
        d = d.max((x - before).abs()).max((after - x).abs());
    }
    Some(WatsonUniformity {
        statistic: d,
        p_value: None,
        n: pairs.len(),
    })
}

/// Closed-form asymptotic upper-tail p-value of Watson's `U²` under the uniform
/// null: `P(U² ≥ u) = 2 Σ_{j≥1} (−1)^{j−1} exp(−2 j² π² u)` (Watson 1961). This
/// is the exact limiting distribution — NOT a tabulated critical constant — so
/// the "flagged / not flagged" decision is derived, not tuned. As a check the
/// series returns `≈ 0.05` at the tabulated 5% point `u = 0.187` and `≈ 0.01` at
/// the 1% point `u = 0.267` (asserted in the tests).
///
/// With `a = 2π²u`, Poisson summation (Jacobi's `θ₄ ↔ θ₂` transform)
/// `Σ_{j∈ℤ} (−1)^j e^{−aj²} = √(π/a) Σ_{k∈ℤ} e^{−π²(k+½)²/a}` gives the dual form
/// `P = 1 − √(2/(πu)) Σ_{k≥0} e^{−(2k+1)²/(8u)}`. Each form's terms decay by at
/// least `e^{−π}` per step on its own side of `a = π` (`u = 1/(2π)`), so the
/// direct series is summed there and above and the dual below. That keeps a
/// small statistic, a coordinate more uniform than chance, at `P → 1`; the
/// direct series alone needs about `1/√u` terms to settle there.
///
/// Derived (#2469): each sum runs until a term no longer changes it in f64. The
/// terms decrease, so everything omitted is below half an ulp of the sum (the
/// direct series alternates, and the dual's positive tail is at most `e^{−2π}`
/// times its last term). The terms underflow to zero for every finite `u > 0`,
/// so both loops end.
pub fn watson_u2_pvalue(u2: f64) -> f64 {
    if !(u2 > 0.0) {
        return 1.0;
    }
    if !u2.is_finite() {
        return 0.0;
    }
    let pi = std::f64::consts::PI;
    if u2 >= 1.0 / (2.0 * pi) {
        let a = 2.0 * pi * pi * u2;
        let mut sum = 0.0_f64;
        let mut j = 1.0_f64;
        let mut sign = 1.0_f64;
        loop {
            let next = sum + sign * (-a * j * j).exp();
            if next == sum {
                break;
            }
            sum = next;
            j += 1.0;
            sign = -sign;
        }
        2.0 * sum
    } else {
        let mut sum = 0.0_f64;
        let mut odd = 1.0_f64;
        loop {
            let next = sum + (-(odd * odd) / (8.0 * u2)).exp();
            if next == sum {
                break;
            }
            sum = next;
            odd += 2.0;
        }
        if sum == 0.0 {
            // Every dual term underflowed: `P` is 1 to working precision, and the
            // prefactor may itself overflow at a subnormal `u`.
            return 1.0;
        }
        1.0 - (2.0 / (pi * u2)).sqrt() * sum
    }
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
            p_value: Some(1.0),
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
        p_value: Some(p_value),
        n,
    }
}

pub fn coordinate_uniformity_weighted(
    coords: ArrayView1<'_, f64>,
    support: &SupportMeasure,
    topology: &CanonicalChartTopology,
) -> Option<WatsonUniformity> {
    if support.len() != coords.len() {
        return None;
    }
    coordinate_uniformity_impl(coords, Some(support.weights()), topology)
}

fn coordinate_uniformity_impl(
    coords: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
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
            // Derived (#2469): `t − lo` and `hi − lo` are each one rounded subtraction
            // of exact inputs, so every normalized `(t − lo)/span` is within a few ulps
            // of its exact value for any positive finite span. Only an empty interval,
            // every coordinate equal, leaves nothing to normalize.
            if !(span > 0.0 && span.is_finite()) {
                return None;
            }
            coords.iter().map(|&t| (t - lo) / span).collect()
        }
    };
    match topology {
        CanonicalChartTopology::Circle { .. } => match weights {
            Some(w) => watson_u2_uniform_weighted(&u, w),
            None => Some(watson_u2_uniform(&u)),
        },
        CanonicalChartTopology::Interval => interval_uniformity(&u, weights),
    }
}

/// Weighted Watson `U²` against the uniform invariant measure. `weights` are the
/// unnormalised support masses for the same rows as `u`; zero-weight rows do not
/// contribute. For equal unit weights this reduces to [`watson_u2_uniform`].
pub(crate) fn watson_u2_uniform_weighted(
    u: &[f64],
    weights: ArrayView1<'_, f64>,
) -> Option<WatsonUniformity> {
    if u.len() != weights.len() {
        return None;
    }
    let mut pairs: Vec<(f64, f64)> = u
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter_map(|(x, w)| {
            if x.is_finite() && w.is_finite() && w > 0.0 {
                let f = x - x.floor();
                Some((if f >= 1.0 { 0.0 } else { f }, w))
            } else {
                None
            }
        })
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mass: f64 = pairs.iter().map(|(_, w)| *w).sum();
    let fisher_n: f64 = pairs.iter().map(|(_, w)| *w * *w).sum();
    if !(mass > 0.0 && fisher_n > 0.0) {
        return None;
    }
    let ess = (mass * mass) / fisher_n;
    let mut cumulative = 0.0_f64;
    let mut cvm_core = 0.0_f64;
    let mut mean = 0.0_f64;
    for (ui, wi_raw) in pairs.iter().copied() {
        let wi = wi_raw / mass;
        let midpoint = cumulative + 0.5 * wi;
        let d = ui - midpoint;
        cvm_core += wi * d * d;
        mean += wi * ui;
        cumulative += wi;
    }
    let u2 = ess * cvm_core + 1.0 / (12.0 * ess) - ess * (mean - 0.5) * (mean - 0.5);
    Some(WatsonUniformity {
        statistic: u2,
        p_value: Some(watson_u2_pvalue(u2)),
        n: pairs.len(),
    })
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
// across three model classes: the uniform density (0 free location parameters),
// a single wrapped Gaussian (the continuous unimodal / von-Mises-like
// alternative), and a `k`-anchor wrapped-Gaussian mixture. The anchor count is
// walked upward from `k = 2` one order at a time, and the walk ends at the first
// order whose evidence does not improve on the order below it, so every count
// the data can carry is reachable and no hand-picked ladder of counts is
// consulted (SPEC rule 18, #2902). The winning class is the occupancy law; when a
// `k ≥ 2` anchor model wins, the atom carries a discrete measure of `k` anchors
// (`d_eff = k − 1`).
//
// #4237 — BIC is `−2 ℓ̂ + p ln n` with `ℓ̂` the MAXIMISED log-likelihood, so each
// rung's `ℓ̂` is the certified maximum of the shared-width mixture likelihood
// over all `2k` parameters (means, weights, width), found by `opt`'s Newton
// trust region on the exact jet and read only where the Newton decrement lies
// inside the likelihood's own rounding band (SPEC: "A fit object must only ever
// come from a converged optimization."). A rung whose maximum does not certify
// has no evidence to compare, so the law is `Indeterminate` rather than a
// verdict read off an unconverged plug-in.
// ===========================================================================

/// The occupancy law of a fitted `d = 1` coordinate ON its honest chart: which
/// measure the data draws from. Adjudicated by evidence (`classify_occupancy_weighted`),
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
    /// #2691 — the coordinate does not resolve AT ALL: every row lies inside an
    /// arc narrower than the resolution floor `1/(2n)` this classifier already
    /// derives, so the chart assigns the rows one value and encodes nothing.
    ///
    /// This is NOT [`Self::Continuous`]. A collapsed coordinate is fit perfectly
    /// by a single wrapped Gaussian at the floor width, so on BIC alone it WINS
    /// the continuous rung and gets reported as "a concentrated arc" — a benign
    /// reading of a chart that carries no information. Measured on
    /// `sae_manifold_fit` (#2691): an exact planted circle returned a coordinate
    /// with std `1.06e-14`, one distinct value across 70 rows, and the fit
    /// certified. Reconstruction EV cannot discriminate it either — the collapsed
    /// arm's EV (0.0883) exceeded the recovering arm's on the same fixture.
    Collapsed,
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
            OccupancyLaw::Collapsed => "collapsed",
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

/// Weighted circular occupancy law. `weights` must be the same atom support
/// masses used by coordinate fidelity and persistence; zero-mass rows are absent.
/// Hard 0/1 support reproduces the unweighted occupancy law.
pub fn classify_occupancy_weighted(u: &[f64], weights: ArrayView1<'_, f64>) -> OccupancyLaw {
    classify_occupancy_weighted_impl(u, weights, true)
}

/// Occupancy law for an INTERVAL (non-wrapping) coordinate `u ∈ [0, 1]` with
/// per-row weights: the same evidence race, but on the LINE rather than the
/// circle, so the extreme values `0` and `1` are NOT cyclically adjacent. Use
/// this for interval-topology coordinates (a birth PCA seed, a bounded latent)
/// where a circular fold would wrongly merge a linear finite set's first and
/// last anchors and misread a range-filling uniform coordinate as non-uniform.
/// [`classify_occupancy_weighted`] is the circular counterpart for genuinely
/// cyclic (circle-chart) coordinates.
pub(crate) fn classify_occupancy_interval_weighted(
    u: &[f64],
    weights: ArrayView1<'_, f64>,
) -> OccupancyLaw {
    classify_occupancy_weighted_impl(u, weights, false)
}

/// Shared occupancy adjudicator. `circular` selects the geometry: `true` folds
/// onto the unit circle (wrapped distances, the periodic images
/// [`wrapped_image_count`] retains), `false` treats
/// `[0, 1]` as a line (linear distances, no wrap). The model race and BIC are
/// identical; only the metric differs.
/// #2691 — the extent of the smallest arc (circle) or interval (line) containing
/// every coordinate, with `pts` already folded into `[0, 1]` and SORTED.
///
/// On the circle this is `1 − (largest gap between cyclically adjacent points)`:
/// a coordinate concentrated near the wrap point occupies a short arc even though
/// its raw `min`/`max` span nearly the whole period, so a plain range would
/// mistake it for a spread-out coordinate.
fn occupied_extent(pts: &[f64], circular: bool) -> f64 {
    match (pts.first(), pts.last()) {
        (Some(&first), Some(&last)) if pts.len() >= 2 => {
            if !circular {
                return last - first;
            }
            let mut largest_gap = (first + 1.0) - last; // the wrap-around gap
            for pair in pts.windows(2) {
                largest_gap = largest_gap.max(pair[1] - pair[0]);
            }
            (1.0 - largest_gap).max(0.0)
        }
        _ => 0.0,
    }
}

fn classify_occupancy_weighted_impl(
    u: &[f64],
    weights: ArrayView1<'_, f64>,
    circular: bool,
) -> OccupancyLaw {
    if u.len() != weights.len() {
        return OccupancyLaw::Indeterminate;
    }
    let mut pairs: Vec<(f64, f64)> = u
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter_map(|(x, w)| {
            if x.is_finite() && w.is_finite() && w > 0.0 {
                let folded = if circular {
                    let f = x - x.floor();
                    if f >= 1.0 { 0.0 } else { f }
                } else {
                    x.clamp(0.0, 1.0)
                };
                Some((folded, w))
            } else {
                None
            }
        })
        .collect();
    if pairs.len() < 4 {
        return OccupancyLaw::Indeterminate;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let pts: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let w: Vec<f64> = pairs.iter().map(|(_, weight)| *weight).collect();
    let support = match SupportMeasure::from_weights(0, Array1::from_vec(w.clone())) {
        Ok(support) => support,
        Err(_) => return OccupancyLaw::Indeterminate,
    };
    let mass = support.mass();
    let effective_rows = support.ess();
    if !(mass > 0.0 && effective_rows >= 4.0) {
        return OccupancyLaw::Indeterminate;
    }
    // #4319 — every other side of the race is charged at the Kish effective
    // support: the admission gate above, the BIC penalty `p·ln n` below and the
    // width floor's resolution all read `n = ess`. The gate masses are
    // unnormalised, so `Σ w_i ln f(x_i)` is the log-likelihood of `mass` rows;
    // multiplying every gate by `c` scales it by `c` while leaving `ess`, the
    // penalty and the floor fixed, and the verdict would read the unit of the
    // masses rather than the coordinate. Renormalising the masses so they sum to
    // the effective support puts the likelihood on the rows the penalty counts.
    // For hard 0/1 support `mass == ess` exactly, the factor is exactly `1`, and
    // the unweighted values are reproduced bit for bit.
    let w: Vec<f64> = w
        .into_iter()
        .map(|weight| weight * (effective_rows / mass))
        .collect();
    let ln_n = effective_rows.ln();
    let bic_uniform = 0.0_f64;
    let sigma_floor = 1.0 / (2.0 * effective_rows);

    // #2691 — the same collapse guard on the weighted path, against the
    // effective-sample resolution floor (`pts` is already sorted and folded).
    if occupied_extent(&pts, circular) < sigma_floor {
        return OccupancyLaw::Collapsed;
    }

    let mixture_bic = |anchors: usize| {
        certified_mixture_log_likelihood(&pts, &w, anchors, sigma_floor, circular, effective_rows)
            .map(|log_likelihood| -2.0 * log_likelihood + (2 * anchors) as f64 * ln_n)
    };

    let Some(single) = mixture_bic(1) else {
        return OccupancyLaw::Indeterminate;
    };
    let mut best_law = OccupancyLaw::Uniform;
    let mut best_bic = bic_uniform;
    if single < best_bic {
        best_bic = single;
        best_law = OccupancyLaw::Continuous;
    }
    // #4323 — the winning law is the BIC minimiser over EVERY admissible order
    // (`k < rows`), not the order where the BIC first stops falling. BIC is not
    // unimodal in `k` under a shared width: below the true anchor count one width
    // must span merged clusters, so each added anchor costs `2 ln n` and buys
    // little likelihood, and the BIC rises until `k` reaches the true count, where
    // the width collapses and the BIC drops by orders of magnitude (the weekday
    // fixture's BIC rises from k = 2 to k = 5, then falls by 347 from k = 6
    // to k = 7).
    //
    // The walk ends on a certificate instead of a local rule. No mixture of any
    // order has a log-likelihood above `loglik_bound`, so
    // `BIC_k ≥ −2·loglik_bound + 2k·ln n`. Once that floor reaches the best
    // BIC so far, no higher order can win: the floor rises with `k`, and
    // `best_bic` never rises.
    let Some(loglik_bound) =
        order_free_mixture_loglik_bound(&pts, &w, sigma_floor, circular, effective_rows)
    else {
        return OccupancyLaw::Indeterminate;
    };
    for k in 2..pairs.len() {
        let bic_floor = -2.0 * loglik_bound + (2 * k) as f64 * ln_n;
        if bic_floor >= best_bic {
            break;
        }
        // An order whose evidence is not computable leaves the minimiser
        // unknown. Reading the law off the orders that happened to compute would
        // name a winner the race never established.
        let Some(bic) = mixture_bic(k) else {
            return OccupancyLaw::Indeterminate;
        };
        if bic < best_bic {
            best_bic = bic;
            best_law = OccupancyLaw::Discrete { anchors: k };
        }
    }
    best_law
}

/// #4323 — an upper bound on the weighted log-likelihood `Σ wᵢ ln f(uᵢ)` of
/// every density the anchor walk can score, at every order at once. Such a
/// density is `f = Σⱼ πⱼ κ_σ(· − μⱼ)` with `Σ πⱼ = 1` and a shared width
/// `σ ≥ σ_floor`. `κ_σ` is the Gaussian on the line. On the circle it is a
/// partial sum of the Gaussian's periodic images, which is at most the full
/// wrapped normal. The bound comes from three facts.
///
/// 1. **One width suffices.** Gaussians compose: `κ_σ = κ_{σ_floor} ∗ N(0,
///    σ² − σ_floor²)`, on the line and, for the wrapped normal, on the circle.
///    So every such `f` is a mixture `f_Q = ∫ κ_{σ_floor}(· − μ) dQ(μ)` for some
///    mixing law `Q`, whatever its order or width.
/// 2. **Jensen's inequality (Lindsay's bound).** Take any positive `gᵢ` and set
///    `D(μ) = Σᵢ (wᵢ/M) κ_{σ_floor}(uᵢ − μ)/gᵢ`, where `M` is the total mass.
///    Then for every `Q`:
///    `Σ wᵢ ln f_Q(uᵢ) ≤ Σ wᵢ ln gᵢ + M ln Σ (wᵢ/M) f_Q(uᵢ)/gᵢ
///      = Σ wᵢ ln gᵢ + M ln ∫ D dQ ≤ Σ wᵢ ln gᵢ + M ln sup D`.
/// 3. **The supremum of `D` over cells.** For every `μ` in a cell `[a, z]`,
///    `D(μ) ≤ Σ cᵢ κ(dist(uᵢ, [a, z]))`. On the line, `D` falls off outside
///    the rows' hull, so cells covering the hull suffice.
///
/// Choices:
/// - `gᵢ` is the kernel density of the rows at `σ_floor`. It is the value of
///   `f_Q` when `Q` is the empirical law, so the bound is tight when the rows
///   are the anchors (the weekday fixture: bound 187.8, seven-anchor fit
///   186.2). Any positive `gᵢ` keeps the bound valid, so image terms beyond the
///   reach below are simply left out of `gᵢ`.
/// - In `sup D`, each term beyond `reach` is charged at its largest possible
///   value, not dropped. `reach` is where one kernel term falls to one rounding
///   of the peak term.
/// - The cell width `σ_floor/4` sets only how early the walk's certificate
///   fires, never which order wins. Any width gives a valid bound.
fn order_free_mixture_loglik_bound(
    pts: &[f64],
    weights: &[f64],
    sigma_floor: f64,
    circular: bool,
    total_mass: f64,
) -> Option<f64> {
    let n = pts.len();
    if n == 0 || weights.len() != n || !(sigma_floor > 0.0) || !(total_mass > 0.0) {
        return None;
    }
    let sigma = sigma_floor;
    let peak = 1.0 / (sigma * std::f64::consts::TAU.sqrt());
    let inv_two_var = 0.5 / (sigma * sigma);
    let kernel = |d: f64| peak * (-(d * d) * inv_two_var).exp();
    let reach = sigma * (-2.0 * f64::EPSILON.ln()).sqrt();
    // On the circle, `reach ≤ 1/2` keeps every image beyond the −1, 0, +1
    // shifts outside the window: for `u, μ ∈ [0, 1]`, `|u + m − μ| ≥ 1` when
    // `|m| ≥ 2`.
    let reach = if circular { reach.min(0.5) } else { reach };
    let shifts: &[f64] = if circular { &[-1.0, 0.0, 1.0] } else { &[0.0] };
    // `pts` is sorted inside `[0, 1)` (circle) or `[0, 1]` (line), so the shifted
    // copies, concatenated in shift order, stay sorted.
    let mut images: Vec<(f64, usize)> = Vec::with_capacity(shifts.len() * n);
    for &shift in shifts {
        for (i, &u) in pts.iter().enumerate() {
            images.push((u + shift, i));
        }
    }
    let window = |lo: f64, hi: f64| {
        let start = images.partition_point(|&(x, _)| x < lo);
        let end = images.partition_point(|&(x, _)| x <= hi).max(start);
        &images[start..end]
    };

    let mut loglik_g = 0.0_f64;
    let mut coefficient = vec![0.0_f64; n];
    for (i, &u) in pts.iter().enumerate() {
        let g: f64 = window(u - reach, u + reach)
            .iter()
            .map(|&(x, j)| weights[j] / total_mass * kernel(x - u))
            .sum();
        if !(g > 0.0 && g.is_finite()) {
            return None;
        }
        loglik_g += weights[i] * g.ln();
        coefficient[i] = weights[i] / total_mass / g;
    }
    // Every term outside a cell's window lies more than `reach` from the cell.
    // On the line each row has one such term. On the circle a row's images
    // outside the window lie at `reach + j` or beyond on each side (`j ≥ 0`).
    // `κ(reach + j) ≤ κ(reach)·exp(−reach·j/σ²)`, so their sum is at most the
    // geometric series `2κ(reach)/(1 − exp(−reach/σ²))`.
    let far_per_row = if circular {
        2.0 * kernel(reach) / (1.0 - (-reach / (sigma * sigma)).exp())
    } else {
        kernel(reach)
    };
    let far = far_per_row * coefficient.iter().sum::<f64>();

    let (lo_mu, hi_mu) = if circular {
        (0.0, 1.0)
    } else {
        (pts[0], pts[n - 1])
    };
    let cell = 0.25 * sigma;
    let cells = ((hi_mu - lo_mu) / cell).ceil().max(1.0) as usize;
    let mut sup_d = 0.0_f64;
    for b in 0..cells {
        let a = lo_mu + b as f64 * cell;
        let z = (a + cell).min(hi_mu);
        let mut d = far;
        for &(x, j) in window(a - reach, z + reach) {
            let gap = if x < a {
                a - x
            } else if x > z {
                x - z
            } else {
                0.0
            };
            d += coefficient[j] * kernel(gap);
        }
        sup_d = sup_d.max(d);
    }
    let bound = loglik_g + total_mass * sup_d.ln();
    bound.is_finite().then_some(bound)
}

/// #4237 — the widest shared width the circular mixture carries: at
/// `σ ≥ σ_max` a wrapped normal is the uniform density to working precision.
///
/// Its Fourier series is `f(d) = 1 + 2 Σ_{q≥1} e^{−2π²q²σ²} cos 2πqd`, so
/// `|f − 1| ≤ 2e^{−2π²σ²} (1 + Σ_{q≥2} e^{−2π²(q²−1)σ²})`. At
/// `σ_max = √(ln(2/u) / (2π²))`, with `u` the unit roundoff, the leading term is
/// `u` and every later term is below `u²`: a wider width changes no density by
/// more than a rounding of `1`. The likelihood is flat in the width beyond it,
/// so it bounds the width from above and the uniform limit is a point the
/// search can reach and certify.
fn wrapped_width_ceiling() -> f64 {
    let unit_roundoff = gam_linalg::roundoff::UNIT_ROUNDOFF;
    ((2.0 / unit_roundoff).ln() / (2.0 * std::f64::consts::PI.powi(2))).sqrt()
}

/// #4237 — how many periodic images `m ∈ [−M, M]` the wrapped normal of width
/// `sigma` needs: the smallest `M` whose omitted images carry, even weighted by
/// the `1 + z⁴` the Hessian's fourth moment puts on them, at most a unit
/// roundoff of the retained central term.
///
/// With the offset `d ∈ [−½, ½)`, the central term has `|z| ≤ b = 1/(2σ)` and an
/// omitted image `|q| ≥ M + 1` has `|z| ≥ a = (M + ½)/σ`. On each side the
/// omitted `(1 + z⁴) e^{−z²/2}` fall at least geometrically, by
/// `ρ = ((M + 3/2)/(M + ½))⁴ e^{−(M+1)/σ²}` per image, so once `ρ < 1` their sum
/// relative to the central `e^{−b²/2}` is at most
/// `2 (1 + a⁴) e^{−(a² − b²)/2} / (1 − ρ)`.
fn wrapped_image_count(sigma: f64) -> usize {
    let unit_roundoff = gam_linalg::roundoff::UNIT_ROUNDOFF;
    let central = 0.5 / sigma;
    let mut images = 0usize;
    loop {
        let nearest = images as f64 + 0.5;
        let reach = nearest / sigma;
        let ratio =
            ((nearest + 1.0) / nearest).powi(4) * (-(nearest + 0.5) / (sigma * sigma)).exp();
        if ratio < 1.0 {
            let tail = 2.0 * (1.0 + reach.powi(4))
                * (-0.5 * (reach * reach - central * central)).exp()
                / (1.0 - ratio);
            if tail <= unit_roundoff {
                return images;
            }
        }
        images += 1;
    }
}

/// The signed offset of `x` from `mean`: on the circle the representative in
/// `[−½, ½)`, on the line the plain difference.
fn signed_offset(x: f64, mean: f64, circular: bool) -> f64 {
    let raw = x - mean;
    if circular {
        (raw + 0.5).rem_euclid(1.0) - 0.5
    } else {
        raw
    }
}

/// #4237 — the exact second-order jet of the negative mixture log-likelihood
/// `−Σ_i w_i ln f(x_i; θ)` of a `k`-anchor, shared-width Gaussian mixture
/// (wrapped on the circle), at `θ = [μ_1..μ_k, a_1..a_{k−1}, ln σ]` with the
/// weights `π = softmax(a_1, .., a_{k−1}, 0)`.
///
/// Every term is `h_{jm} = π_j φ(z_{jm}) / σ` with `z_{jm} = (d_j + m)/σ` over
/// the images [`wrapped_image_count`] retains (`m = 0` alone on the line), and
/// `ln f = ln Σ h` is taken by log-sum-exp. With responsibilities `r = h / f`,
/// `∇ ln f = Σ r ∇ ln h` and `∇² ln f = Σ r (∇² ln h + ∇ln h ∇ln hᵀ) − ∇ln f ∇ln fᵀ`,
/// where `∇ ln h` is `z/σ` in `μ_j`, `z² − 1` in `ln σ` and `δ_{jl} − π_l` in
/// `a_l`, and `∇² ln h` is `−1/σ²`, `−2z/σ`, `−2z²` and `−(diag π − ππᵀ)`.
///
/// The sample carries its rounding bands, so `opt`'s verdict on it is the
/// Newton decrement against them: each band charges the accumulation growth of
/// the longest sum (the rows, then every image of every anchor, then the `2k`
/// parameter terms) on the absolute values the sums accumulate.
fn mixture_negative_log_likelihood(
    pts: &[f64],
    weights: &[f64],
    anchors: usize,
    circular: bool,
    theta: &Array1<f64>,
) -> SecondOrderSample {
    let k = anchors;
    let p = 2 * k;
    let s_idx = p - 1;
    let log_sigma = theta[s_idx];
    let sigma = log_sigma.exp();
    let logits: Vec<f64> = (0..k)
        .map(|j| if j + 1 < k { theta[k + j] } else { 0.0 })
        .collect();
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_normaliser = top + logits.iter().map(|a| (a - top).exp()).sum::<f64>().ln();
    let log_pi: Vec<f64> = logits.iter().map(|a| a - log_normaliser).collect();
    let pi: Vec<f64> = log_pi.iter().map(|l| l.exp()).collect();
    let images = if circular { wrapped_image_count(sigma) } else { 0 };
    let span = 2 * images + 1;
    let log_density_offset = 0.5 * std::f64::consts::TAU.ln() + log_sigma;

    let mut value = 0.0_f64;
    let mut gradient = Array1::<f64>::zeros(p);
    let mut hessian = Array2::<f64>::zeros((p, p));
    let mut value_channel = 0.0_f64;
    let mut gradient_channel = Array1::<f64>::zeros(p);
    let mut hessian_channel = Array2::<f64>::zeros((p, p));

    let mut z = vec![0.0_f64; k * span];
    let mut log_terms = vec![0.0_f64; k * span];
    let mut g = Array1::<f64>::zeros(p);
    let mut g_abs = Array1::<f64>::zeros(p);
    let mut h = Array2::<f64>::zeros((p, p));
    let mut h_abs = Array2::<f64>::zeros((p, p));
    let mut dl = vec![0.0_f64; k.saturating_sub(1)];
    for (&x, &wi) in pts.iter().zip(weights) {
        for j in 0..k {
            let d = signed_offset(x, theta[j], circular);
            for m in 0..span {
                let zz = (d + m as f64 - images as f64) / sigma;
                z[j * span + m] = zz;
                log_terms[j * span + m] = log_pi[j] - 0.5 * zz * zz - log_density_offset;
            }
        }
        let peak = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = log_terms.iter().map(|l| (l - peak).exp()).sum();
        let ln_f = peak + total.ln();

        g.fill(0.0);
        g_abs.fill(0.0);
        h.fill(0.0);
        h_abs.fill(0.0);
        let mut second_moment_total = 0.0_f64;
        for j in 0..k {
            let (mut r0, mut z1, mut z2, mut z3, mut z4) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut a1, mut b1, mut ab, mut b2) = (0.0, 0.0, 0.0, 0.0);
            for m in 0..span {
                let r = (log_terms[j * span + m] - peak).exp() / total;
                let zz = z[j * span + m];
                let zsq = zz * zz;
                let excess = zsq - 1.0;
                r0 += r;
                z1 += r * zz;
                z2 += r * zsq;
                z3 += r * zsq * zz;
                z4 += r * zsq * zsq;
                a1 += r * zz.abs();
                b1 += r * excess.abs();
                ab += r * zz.abs() * excess.abs();
                b2 += r * excess * excess;
            }
            second_moment_total += z2;
            for (l, slot) in dl.iter_mut().enumerate() {
                *slot = if l == j { 1.0 } else { 0.0 } - pi[l];
            }
            g[j] += z1 / sigma;
            g[s_idx] += z2 - r0;
            g_abs[j] += a1 / sigma;
            g_abs[s_idx] += b1;
            h[[j, j]] += (z2 - r0) / (sigma * sigma);
            h_abs[[j, j]] += (r0 + z2) / (sigma * sigma);
            let mixed = (z3 - 3.0 * z1) / sigma;
            let mixed_abs = (2.0 * a1 + ab) / sigma;
            h[[j, s_idx]] += mixed;
            h[[s_idx, j]] += mixed;
            h_abs[[j, s_idx]] += mixed_abs;
            h_abs[[s_idx, j]] += mixed_abs;
            h[[s_idx, s_idx]] += z4 - 4.0 * z2 + r0;
            h_abs[[s_idx, s_idx]] += 2.0 * z2 + b2;
            for l in 0..dl.len() {
                let al = k + l;
                g[al] += r0 * dl[l];
                g_abs[al] += r0 * dl[l].abs();
                h[[j, al]] += z1 / sigma * dl[l];
                h[[al, j]] += z1 / sigma * dl[l];
                h_abs[[j, al]] += a1 / sigma * dl[l].abs();
                h_abs[[al, j]] += a1 / sigma * dl[l].abs();
                h[[s_idx, al]] += (z2 - r0) * dl[l];
                h[[al, s_idx]] += (z2 - r0) * dl[l];
                h_abs[[s_idx, al]] += b1 * dl[l].abs();
                h_abs[[al, s_idx]] += b1 * dl[l].abs();
                for q in 0..dl.len() {
                    let aq = k + q;
                    let covariance = if l == q { pi[l] } else { 0.0 } - pi[l] * pi[q];
                    h[[al, aq]] += r0 * (dl[l] * dl[q] - covariance);
                    h_abs[[al, aq]] += r0 * (dl[l].abs() * dl[q].abs() + covariance.abs());
                }
            }
        }
        for row in 0..p {
            for col in 0..p {
                h[[row, col]] -= g[row] * g[col];
                h_abs[[row, col]] += g_abs[row] * g_abs[col];
            }
        }
        value -= wi * ln_f;
        gradient.scaled_add(-wi, &g);
        hessian.scaled_add(-wi, &h);
        value_channel += wi * (ln_f.abs() + 1.0 + second_moment_total);
        gradient_channel.scaled_add(wi, &g_abs);
        hessian_channel.scaled_add(wi, &h_abs);
    }
    let growth = accumulation_growth(pts.len() + k * span + p);
    let hessian_band = growth * hessian_channel.iter().map(|v| v * v).sum::<f64>().sqrt();
    SecondOrderSample {
        value,
        gradient,
        hessian: Some(hessian),
        decrement_bands: Some(DecrementBands {
            objective: growth * value_channel,
            tolerance: growth * value_channel,
            gradient: gradient_channel * growth,
            hessian: hessian_band,
        }),
    }
}

/// #4237 — the certified maximum log-likelihood of the `anchors`-component,
/// shared-width Gaussian mixture (wrapped on the circle) of the weighted
/// points, or `None` when the maximisation does not certify.
///
/// `opt`'s Newton trust region runs on the exact jet
/// ([`mixture_negative_log_likelihood`]) over the width box
/// `σ ∈ [σ_floor, σ_max]` — the resolution floor below, and on the circle the
/// uniform limit [`wrapped_width_ceiling`] above — and starts from the weighted
/// quantile means with equal weights and the width of the rows' spread about
/// their nearest mean. The start only chooses the basin; the value read is the
/// certified stationary point's.
fn certified_mixture_log_likelihood(
    pts: &[f64],
    weights: &[f64],
    anchors: usize,
    sigma_floor: f64,
    circular: bool,
    total_mass: f64,
) -> Option<f64> {
    let k = anchors;
    let p = 2 * k;
    let means = weighted_quantile_initial_means(pts, weights, k, total_mass);
    let spread: f64 = pts
        .iter()
        .zip(weights)
        .map(|(&x, &wi)| {
            let nearest = means
                .iter()
                .map(|&m| signed_offset(x, m, circular).abs())
                .fold(f64::INFINITY, f64::min);
            wi * nearest * nearest
        })
        .sum();
    let lower_log_width = sigma_floor.ln();
    let upper_log_width = if circular {
        wrapped_width_ceiling().ln()
    } else {
        f64::INFINITY
    };
    let start_log_width = (spread / total_mass)
        .sqrt()
        .max(sigma_floor)
        .ln()
        .clamp(lower_log_width, upper_log_width);
    let mut start = Array1::<f64>::zeros(p);
    for (j, &m) in means.iter().enumerate() {
        start[j] = m;
    }
    start[p - 1] = start_log_width;
    let mut lower = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut upper = Array1::from_elem(p, f64::INFINITY);
    lower[p - 1] = lower_log_width;
    upper[p - 1] = upper_log_width;
    let bounds = Bounds::new(lower, upper, 0.0).ok()?;
    let solution = certified_newton_minimum(start, Some(bounds), None, |theta: &Array1<f64>| {
        Ok::<_, ObjectiveEvalError>(mixture_negative_log_likelihood(
            pts, weights, k, circular, theta,
        ))
    })
    .ok()?;
    Some(-solution.final_value)
}

fn weighted_quantile_initial_means(
    pts: &[f64],
    weights: &[f64],
    k: usize,
    total_mass: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    for j in 0..k {
        let target = (j as f64 / k as f64) * total_mass;
        let mut acc = 0.0_f64;
        let mut chosen = pts[0];
        for (&p, &w) in pts.iter().zip(weights.iter()) {
            acc += w;
            if acc >= target {
                chosen = p;
                break;
            }
        }
        out.push(chosen);
    }
    out
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
    pub uniformity_statistic: Option<f64>,
    /// Closed-form asymptotic p-value of the circle Watson statistic. `None` for
    /// interval charts whose endpoints were fitted from these same coordinates.
    pub uniformity_p_value: Option<f64>,
    /// Arc-length (unit-speed) defect of the chart parameterization
    /// ([`crate::chart_canonicalization::chart_unit_speed_defect`]): speed
    /// coefficient of variation on a uniform latent grid, `0` ⟺ exactly
    /// arc-length. `None` when the chart-speed evaluation honest-skipped
    /// (degenerate chart).
    pub arclength_defect: Option<f64>,
    /// Number of fitted coordinates the uniformity statistic was computed from.
    pub n_coords: usize,
    /// Soft occupancy mass `Σ_i w_i` from the shared atom support measure.
    pub support_mass: f64,
    /// Reconstruction-information effective count `Σ_i w_i²` from the shared
    /// atom support measure.
    pub effective_n: f64,
    /// Kish effective support `(Σ_i w_i)² / Σ_i w_i²`, the number of equally
    /// weighted rows represented by this atom's support distribution.
    pub support_ess: f64,
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
    /// measured on data rather than a grid). `None` when `u_arc` is unavailable.
    pub raw_arclength_defect_rms: Option<f64>,
    /// Max over the fitted rows of the same aligned raw-vs-`u_arc` distance.
    pub raw_arclength_defect_max: Option<f64>,
    /// `min ‖γ'‖ / mean ‖γ'‖` of the decoder curve on a uniform grid. Below the
    /// [`SAE_FLOW_DIFFEO_MIN_DET`] collapse floor drives the `Degenerate`
    /// verdict. `None` when the chart-speed reading is unavailable.
    pub min_speed_over_mean: Option<f64>,
    /// `max ‖γ'‖ / mean ‖γ'‖` on the grid. `None` when unavailable.
    pub max_speed_over_mean: Option<f64>,
    /// RMS of `log(‖γ'‖/mean)` on the grid — scale-invariant log-speed spread.
    /// `None` when unavailable.
    pub log_speed_rms: Option<f64>,
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
pub(crate) struct CoordinateFidelityCertificate<'a> {
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
/// unit-speed retraction uses, `SaeManifoldTerm::d1_unit_speed_topology`).
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
    let support = SupportMeasure::from_assignment(&term.assignment, atom_idx)?;
    let uniformity = coordinate_uniformity_weighted(row_coords, &support, &topology);
    // Occupancy law (F2): classified from the SAME folded coordinates the
    // uniformity statistic reads, but adjudicated by evidence rather than the
    // uniform-null p-value. Reported separately from chart honesty so a discrete
    // measure on an honest chart is not read as a fidelity failure.
    let occupancy_law = fold_for_occupancy_weighted(row_coords, support.weights(), &topology)
        .map(|(folded, folded_weights)| {
            if matches!(topology, CanonicalChartTopology::Circle { .. }) {
                classify_occupancy_weighted(&folded, folded_weights.view())
            } else {
                classify_occupancy_interval_weighted(&folded, folded_weights.view())
            }
        })
        .unwrap_or(OccupancyLaw::Indeterminate);
    let atom = &term.atoms[atom_idx];
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!("atom_coordinate_fidelity: atom {atom_idx} has no basis evaluator")
    })?;
    let defect = crate::chart_canonicalization::chart_unit_speed_defect(
        evaluator.as_ref(),
        atom.decoder_coefficients().view(),
        row_coords,
        &topology,
    )?;
    // The honest arc-length coordinate + speed profile, computed as a pure read
    // (ungated by the decoder-recomposition tolerance) — always reportable even
    // when the mutating canonicalization honestly refused.
    let reading = chart_arclength_coordinates(
        evaluator.as_ref(),
        atom.decoder_coefficients().view(),
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
            let (rms, max) = raw_vs_arclength_defect_weighted(
                row_coords,
                r.coords_u_arc.view(),
                support.weights(),
                &topology,
                is_circle,
            );
            (
                verdict,
                Some(r.coords_u_arc),
                Some(rms),
                Some(max),
                Some(r.min_speed_over_mean),
                Some(r.max_speed_over_mean),
                Some(r.log_speed_rms),
            )
        }
        // Collapsed chart (speed vanishes somewhere) or arc length ill-defined:
        // no faithful coordinate exists — refuse.
        Some(r) => (
            AngleFidelityVerdict::Degenerate,
            None,
            None,
            None,
            Some(r.min_speed_over_mean),
            Some(r.max_speed_over_mean),
            Some(r.log_speed_rms),
        ),
        None => (
            AngleFidelityVerdict::Degenerate,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    };

    Ok(Some(AtomCoordinateFidelity {
        topology: topology_label,
        uniformity_statistic: uniformity.as_ref().map(|u| u.statistic),
        uniformity_p_value: uniformity.as_ref().and_then(|u| u.p_value),
        arclength_defect: defect,
        n_coords: uniformity.as_ref().map(|u| u.n).unwrap_or(row_coords.len()),
        support_mass: support.mass(),
        effective_n: support.fisher_n(),
        support_ess: support.ess(),
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

fn fold_for_occupancy_weighted(
    coords: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
) -> Option<(Vec<f64>, Array1<f64>)> {
    if coords.len() != weights.len() {
        return None;
    }
    if coords.len() < 2 || coords.iter().any(|t| !t.is_finite()) {
        return None;
    }
    match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return None;
            }
            let mut folded = Vec::new();
            let mut folded_weights = Vec::new();
            for (&t, &w) in coords.iter().zip(weights.iter()) {
                if w > 0.0 {
                    folded.push(t.rem_euclid(*period) / *period);
                    folded_weights.push(w);
                }
            }
            Some((folded, Array1::from_vec(folded_weights)))
        }
        CanonicalChartTopology::Interval => {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for (&t, &w) in coords.iter().zip(weights.iter()) {
                if !(w > 0.0) {
                    continue;
                }
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let span = hi - lo;
            // Derived (#2469): `t − lo` and `hi − lo` are each one rounded subtraction
            // of exact inputs, so every normalized `(t − lo)/span` is within a few ulps
            // of its exact value for any positive finite span. Only an empty interval,
            // every coordinate equal, leaves nothing to normalize.
            if !(span > 0.0 && span.is_finite()) {
                return None;
            }
            let mut folded = Vec::new();
            let mut folded_weights = Vec::new();
            for (&t, &w) in coords.iter().zip(weights.iter()) {
                if w > 0.0 {
                    folded.push((t - lo) / span);
                    folded_weights.push(w);
                }
            }
            Some((folded, Array1::from_vec(folded_weights)))
        }
    }
}

/// The support-weighted (circular, for a circle) distance between the raw
/// normalized coordinate `t_i / span` and its arc-length image `u_i`, minimized
/// over the residual gauge — a base-point shift `c` and an orientation flip
/// `s ∈ {+1, −1}` — and summarized as `(rms, max)` over the rows.
fn raw_vs_arclength_defect_weighted(
    raw: ArrayView1<'_, f64>,
    u_arc: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    topology: &CanonicalChartTopology,
    is_circle: bool,
) -> (f64, f64) {
    let n = raw.len();
    if n == 0 || u_arc.len() != n || weights.len() != n {
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
            for (&t, &w) in raw.iter().zip(weights.iter()) {
                if !(w > 0.0) {
                    continue;
                }
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
            for ((ui, ri), wi) in u_arc.iter().zip(r.iter()).zip(weights.iter()) {
                if !(*wi > 0.0) {
                    continue;
                }
                let diff = ui - s * ri;
                let ang = std::f64::consts::TAU * diff;
                sx += *wi * ang.cos();
                sy += *wi * ang.sin();
            }
            sy.atan2(sx) / std::f64::consts::TAU
        } else {
            let mut acc = 0.0_f64;
            let mut mass = 0.0_f64;
            for ((ui, ri), wi) in u_arc.iter().zip(r.iter()).zip(weights.iter()) {
                if !(*wi > 0.0) {
                    continue;
                }
                acc += *wi * (ui - s * ri);
                mass += *wi;
            }
            if mass > 0.0 { acc / mass } else { 0.0 }
        };
        let mut sum_sq = 0.0_f64;
        let mut max = 0.0_f64;
        let mut mass = 0.0_f64;
        for ((ui, ri), wi) in u_arc.iter().zip(r.iter()).zip(weights.iter()) {
            if !(*wi > 0.0) {
                continue;
            }
            let aligned = s * ri + c;
            let d = if is_circle {
                circ_dist(*ui, aligned)
            } else {
                (ui - aligned).abs()
            };
            sum_sq += *wi * d * d;
            mass += *wi;
            max = max.max(d);
        }
        let rms = if mass > 0.0 {
            (sum_sq / mass).sqrt()
        } else {
            f64::NAN
        };
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
/// `crate::manifold::SAE_FINAL_EV_DEGRADATION_TOL`, a scale-invariant "0.1% of
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

/// #2230 — ONE-referee state preference for the inner-fit keep-best incumbent,
/// keyed on the PENALIZED OBJECTIVE (the exact scalar the inner Armijo lane
/// descends and the outer penalized quasi-Laplace score consumes), with the #2081
/// EV-then-uniformity ordering ([`prefer_candidate_basin`]) demoted to a
/// tie-break at (near-)equal objective.
///
/// Rationale: the inner walk at a probed ρ is objective-monotone (Armijo), so a
/// trajectory that ends at lower reconstruction EV has a LOWER penalized
/// objective — at that ρ the objective genuinely prefers the walked-to state.
/// An EV-keyed incumbent restore then installs a HIGHER-objective state, and
/// because the restored state is ρ-independent the outer criterion gets priced
/// at ≈ the same state for every probe: the outer objective flattens, the ρ
/// search loses its gradient, and the fit grinds `max_iter` restoring the same
/// incumbent after every evaluation (the #2230/#2134 churn signature). Keying
/// the incumbent on the objective makes the restore fire ONLY when the
/// non-monotone boundary hooks (collapse reseeds, gauge retraction/pin, frame
/// refresh) genuinely damaged the walk — never to veto legitimate descent.
///
/// `objective_rel_tol` is the numerical convergence tolerance of the penalized
/// objective itself; it must not be borrowed from the much coarser,
/// dimensionless EV negligibility band. Within the objective convergence band,
/// `ev_tol` controls the EV/uniformity tie-break.
pub(crate) fn prefer_candidate_state(
    candidate_objective: f64,
    candidate_ev: f64,
    candidate_uniformity: Option<f64>,
    incumbent_objective: f64,
    incumbent_ev: f64,
    incumbent_uniformity: Option<f64>,
    objective_rel_tol: f64,
    ev_tol: f64,
) -> bool {
    if !candidate_objective.is_finite() {
        return false;
    }
    if !incumbent_objective.is_finite() {
        return true;
    }
    let scale =
        objective_rel_tol * (1.0 + candidate_objective.abs().max(incumbent_objective.abs()));
    if candidate_objective < incumbent_objective - scale {
        return true; // strictly lower penalized objective — the walk's own referee
    }
    if candidate_objective > incumbent_objective + scale {
        return false; // strictly higher objective can never displace the incumbent
    }
    prefer_candidate_basin(
        candidate_ev,
        candidate_uniformity,
        incumbent_ev,
        incumbent_uniformity,
        ev_tol,
    )
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
    /// occupancy statistic ([`coordinate_uniformity_weighted`]). The two are NOT
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
    ///
    /// A degenerate chart is an honest `Ok(None)` skip inside
    /// [`crate::chart_canonicalization::chart_unit_speed_defect`]; an `Err`
    /// (basis evaluation failure, basis/decoder width mismatch, malformed jet) is
    /// a structural fault and propagates, exactly as it does in the per-atom
    /// report ([`atom_coordinate_fidelity`]). Swallowing it would silently drop
    /// the atom from the mean, so the candidate and incumbent of one tie-break
    /// could be averaged over DIFFERENT atom sets.
    pub(crate) fn coordinate_uniformity_aggregate(&self) -> Result<Option<f64>, String> {
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
            let defect = match atom.basis_evaluator.as_ref() {
                Some(evaluator) => crate::chart_canonicalization::chart_unit_speed_defect(
                    evaluator.as_ref(),
                    atom.decoder_coefficients().view(),
                    coords.column(0),
                    &topology,
                )
                .map_err(|err| {
                    format!("coordinate_uniformity_aggregate: atom {atom_idx}: {err}")
                })?,
                None => None,
            };
            if let Some(d) = defect {
                if d.is_finite() {
                    sum += d;
                    count += 1;
                }
            }
        }
        if count == 0 {
            Ok(None)
        } else {
            Ok(Some(sum / count as f64))
        }
    }
}

#[cfg(test)]
mod coordinate_fidelity_tests {
    use super::*;
    use crate::manifold::{
        SAE_FINAL_EV_DEGRADATION_TOL, SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL, SaeBasisEvaluator,
    };
    use ndarray::{Array1, Array2, Array3, Array4, ArrayView2};

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
        ) -> Result<crate::basis::SaeBasisThirdJetCapability, String> {
            if coords.ncols() != 1 {
                return Err(format!(
                    "CircleHarmonicEvaluator::third_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                ));
            }
            Ok(crate::basis::SaeBasisThirdJetCapability::Unavailable)
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
        ) -> Result<crate::basis::SaeBasisThirdJetCapability, String> {
            if coords.ncols() != 1 {
                return Err(format!(
                    "IntervalLinearEvaluator::third_jet_dyn: d = 1 evaluator got {} coords",
                    coords.ncols()
                ));
            }
            // `[1, t]` is affine, so every third partial vanishes identically.
            Ok(crate::basis::SaeBasisThirdJetCapability::CertifiedZero)
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

    #[test]
    fn support_metrics_are_shared_by_fidelity_occupancy_and_persistence_reads() {
        let weights = Array1::from_vec(vec![1.0, 1.0, 0.5, 0.0]);
        let support = SupportMeasure::from_weights(0, weights).unwrap();
        let coords = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.9]);
        let fidelity = coordinate_uniformity_weighted(coords.view(), &support, &circle()).unwrap();
        let (occupancy_rows, occupancy_weights) =
            fold_for_occupancy_weighted(coords.view(), support.weights(), &circle()).unwrap();
        let persistence_rows = support.positive_rows();

        assert_eq!(fidelity.n, occupancy_rows.len());
        assert_eq!(fidelity.n, occupancy_weights.len());
        assert_eq!(fidelity.n, persistence_rows.len());
        assert!((support.mass() - 2.5).abs() < 1e-12);
        assert!((support.fisher_n() - 2.25).abs() < 1e-12);
        assert!((support.ess() - (2.5_f64 * 2.5 / 2.25)).abs() < 1e-12);
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
            matches!(
                ev.third_jet_dyn(good.view()),
                Ok(crate::basis::SaeBasisThirdJetCapability::Unavailable)
            ),
            "d = 1 coords must declare the third jet Unavailable"
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
        let third = ev.third_jet_dyn(bad.view());
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

    /// #2230 ONE-referee ordering: the penalized objective is primary — a
    /// lower-objective candidate wins even at catastrophically worse EV (the
    /// walk's own preference at this ρ must never be vetoed), a higher-objective
    /// candidate loses even at much better EV (the exact churn mode: the
    /// high-EV incumbent must NOT displace a legitimately walked-to state), and
    /// only a numerical objective tie falls through to EV-then-uniformity.
    #[test]
    fn prefer_candidate_state_prices_objective_then_ev() {
        let objective_tol = SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL;
        let ev_tol = SAE_FINAL_EV_DEGRADATION_TOL;
        // Strictly lower objective wins despite much worse EV.
        assert!(prefer_candidate_state(
            100.0,
            0.13,
            Some(0.5),
            200.0,
            0.65,
            Some(0.01),
            objective_tol,
            ev_tol,
        ));
        // Strictly higher objective loses despite much better EV — the #2230
        // churn signature (EV 0.65 incumbent vetoing an EV 0.13 walked state).
        assert!(!prefer_candidate_state(
            200.0,
            0.65,
            Some(0.01),
            100.0,
            0.13,
            Some(0.5),
            objective_tol,
            ev_tol,
        ));
        // Numerically tied objective (within objective_tol·(1+scale)): EV decides.
        assert!(prefer_candidate_state(
            100.0,
            0.65,
            Some(0.5),
            100.0 + 0.5 * objective_tol,
            0.13,
            Some(0.01),
            objective_tol,
            ev_tol,
        ));
        // Tied objective AND near-equal EV: uniformity decides.
        assert!(prefer_candidate_state(
            100.0,
            0.65,
            Some(0.02),
            100.0,
            0.6502,
            Some(0.20),
            objective_tol,
            ev_tol,
        ));
        // Non-finite candidate objective never preferred.
        assert!(!prefer_candidate_state(
            f64::NAN,
            0.9,
            Some(0.0),
            100.0,
            0.1,
            Some(0.5),
            objective_tol,
            ev_tol,
        ));
        // Non-finite incumbent objective: any finite candidate adopted.
        assert!(prefer_candidate_state(
            100.0,
            0.1,
            None,
            f64::INFINITY,
            0.9,
            None,
            objective_tol,
            ev_tol,
        ));

        // The original #2230 patch used the 1e-3 EV tolerance for objective
        // ties. At the issue's O(1e5) criterion scale that made an O(1)
        // objective improvement look tied and allowed EV to restore the worse
        // state. Objective convergence is five orders tighter than that EV
        // reporting band, so the walked-to state must win here.
        assert!(prefer_candidate_state(
            83_999.0,
            0.13,
            Some(0.5),
            84_000.0,
            0.65,
            Some(0.01),
            objective_tol,
            ev_tol,
        ));
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
        let unit = Array1::<f64>::ones(rows.len());
        let (rms, max) = raw_vs_arclength_defect_weighted(
            rows.view(),
            reading.coords_u_arc.view(),
            unit.view(),
            &circle(),
            true,
        );
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
        let unit = Array1::<f64>::ones(rows.len());
        let (rms, _max) = raw_vs_arclength_defect_weighted(
            rows.view(),
            reading.coords_u_arc.view(),
            unit.view(),
            &circle(),
            true,
        );
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
        let unit = Array1::<f64>::ones(raw.len());
        let (rms0, _) = raw_vs_arclength_defect_weighted(
            raw.view(),
            u_arc.view(),
            unit.view(),
            &circle(),
            true,
        );
        // Rotate the raw base point and reflect its orientation: both are the
        // circle's residual gauge, so the aligned defect must not change.
        let rotated = Array1::from_iter(raw.iter().map(|&t| (t + 0.31).rem_euclid(1.0)));
        let reflected = Array1::from_iter(raw.iter().map(|&t| (1.0 - t).rem_euclid(1.0)));
        let (rms_rot, _) = raw_vs_arclength_defect_weighted(
            rotated.view(),
            u_arc.view(),
            unit.view(),
            &circle(),
            true,
        );
        let (rms_ref, _) = raw_vs_arclength_defect_weighted(
            reflected.view(),
            u_arc.view(),
            unit.view(),
            &circle(),
            true,
        );
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

    // ======================================================================
    // #2691 — the collapse guard. Three mechanisms in this crate LOOK like they
    // cover "the fitted chart coordinate encodes nothing" and none can express
    // it: `AngleFidelityVerdict::Degenerate` is a property of the chart MAP on a
    // uniform latent grid (blind to where the rows land), reconstruction EV is
    // measured non-discriminating on this exact defect, and the occupancy race
    // hands a constant to the single-Gaussian rung and calls it `Continuous`.
    // These tests pin the guard from BOTH sides: it must name the collapse, and
    // it must NOT swallow a genuinely narrow-but-resolvable arc — a guard that
    // rejected every concentrated coordinate would pass a one-sided test while
    // destroying the `Continuous` rung.
    // ======================================================================

    /// #2469: Watson's p-value sums each series until a term no longer changes
    /// it, and below `u = 1/(2π)` it sums the Poisson dual. Under the old
    /// `1e-14` term floor with its 100-term cap, the direct series stopped
    /// mid-oscillation at a small statistic: `p(10⁻⁶)` came out near `0.02`, so a
    /// coordinate far more uniform than chance was flagged. The dual form keeps
    /// it at 1, reproduces the tabulated 10% point, which lies on the dual side,
    /// and meets the direct series at the switch.
    #[test]
    fn watson_pvalue_sums_to_its_fixed_point_and_reads_a_small_statistic_as_uniform_2469() {
        assert_eq!(
            watson_u2_pvalue(1.0e-6),
            1.0,
            "a statistic far below its null mean must read P = 1"
        );
        // 10% critical value 0.152 (Stephens 1970), below the switch.
        let p10 = watson_u2_pvalue(0.152);
        assert!((p10 - 0.10).abs() < 5.0e-3, "p(U²=0.152) must be ≈0.10, got {p10}");
        // At the switch the two forms meet. `dP/du ≈ −1.7` there, so one ulp of
        // `u` moves `P` by `≈ 5e-17`; the rest is each form's few-ulp rounding.
        let switch = 1.0 / (2.0 * std::f64::consts::PI);
        let dual = watson_u2_pvalue(switch.next_down());
        let direct = watson_u2_pvalue(switch);
        assert!(
            (dual - direct).abs() <= 1.0e-15,
            "the dual ({dual:e}) and direct ({direct:e}) forms must meet at u = 1/(2π)"
        );
        // Positive control: the two forms are not trivially equal away from each other.
        assert!(watson_u2_pvalue(0.05) > dual && dual > watson_u2_pvalue(0.30));
    }

    /// #2469: an interval chart's coordinates normalize at any positive span.
    /// `1 + i·2⁻⁴⁵` spans `7·2⁻⁴⁵ ≈ 2e-13`, below the old `1e-12·max(|lo|, |hi|, 1)`
    /// floor, which refused it; `t − lo` is exact here, so its statistic equals
    /// that of `0, 1, …, 7` bit for bit. Equal coordinates are still refused.
    #[test]
    fn interval_uniformity_normalizes_any_positive_span_2469() {
        let support = SupportMeasure::from_weights(0, Array1::from_elem(8, 1.0)).unwrap();
        let reference = Array1::from_shape_fn(8, |i| i as f64);
        let narrow = Array1::from_shape_fn(8, |i| 1.0 + i as f64 * 2.0_f64.powi(-45));
        let wide = coordinate_uniformity_weighted(
            reference.view(),
            &support,
            &CanonicalChartTopology::Interval,
        )
        .expect("a unit-spaced interval coordinate has a span");
        let tight = coordinate_uniformity_weighted(
            narrow.view(),
            &support,
            &CanonicalChartTopology::Interval,
        )
        .expect("a span of 7·2⁻⁴⁵ must normalize");
        assert_eq!(
            tight.statistic.to_bits(),
            wide.statistic.to_bits(),
            "an exactly affine rescaling must leave the statistic bit-identical"
        );
        assert_eq!(tight.p_value, wide.p_value);
        let flat = Array1::from_elem(8, 1.0);
        assert!(
            coordinate_uniformity_weighted(flat.view(), &support, &CanonicalChartTopology::Interval)
                .is_none(),
            "equal coordinates have no span to normalize"
        );
    }

    // ---- #4237: the occupancy BIC reads a CERTIFIED mixture maximum ----------

    /// Normal scores `Φ⁻¹((i + ½)/m)`: a deterministic sample whose moments
    /// are those of a standard normal to `O(1/m)`.
    fn normal_scores(m: usize) -> Vec<f64> {
        (0..m)
            .map(|i| {
                gam_math::probability::standard_normal_quantile((i as f64 + 0.5) / m as f64)
                    .expect("an interior probability has a quantile")
            })
            .collect()
    }

    /// #4237 — the flip fixture: 48 rows at `0.40 + 0.03·z` and 12 at
    /// `0.48 + 0.05·z` on the line. The old hard-assignment plug-in scored the
    /// two-anchor rung at its Lloyd partition (a split near the midpoint of two
    /// overlapping clusters), not at its maximum, and read `Continuous`. At the
    /// certified maxima the BICs are `k = 1: −189.39`, `k = 2: −192.52`,
    /// `k = 3: −184.33`, so the evidence picks two anchors.
    #[test]
    fn occupancy_reads_the_certified_two_anchor_maximum_on_overlapping_clusters_4237() {
        let mut u: Vec<f64> = normal_scores(48).iter().map(|z| 0.40 + 0.03 * z).collect();
        u.extend(normal_scores(12).iter().map(|z| 0.48 + 0.05 * z));
        let weights = Array1::from_elem(u.len(), 1.0);
        assert_eq!(
            classify_occupancy_interval_weighted(&u, weights.view()),
            OccupancyLaw::Discrete { anchors: 2 },
            "the certified two-anchor maximum beats the single Gaussian"
        );
    }

    /// #4237 — on the line a single Gaussian's maximum is closed-form: the
    /// weighted mean and the (biased) weighted standard deviation, with
    /// `ℓ* = −(W/2)(ln 2πσ̂² + 1)`. The certified search must land on it: within
    /// one rounding band of the decrement it certified and one of the value.
    #[test]
    fn single_anchor_line_maximum_matches_the_closed_form_4237() {
        let mut pts: Vec<f64> = normal_scores(48).iter().map(|z| 0.40 + 0.03 * z).collect();
        pts.extend(normal_scores(12).iter().map(|z| 0.48 + 0.05 * z));
        pts.sort_by(f64::total_cmp);
        let w: Vec<f64> = (0..pts.len()).map(|i| 1.0 + (i % 3) as f64).collect();
        let mass: f64 = w.iter().sum();
        let mean = pts.iter().zip(&w).map(|(x, wi)| wi * x).sum::<f64>() / mass;
        let variance =
            pts.iter().zip(&w).map(|(x, wi)| wi * (x - mean).powi(2)).sum::<f64>() / mass;
        let closed_form = -0.5 * mass * ((std::f64::consts::TAU * variance).ln() + 1.0);
        let floor = 1.0 / (2.0 * pts.len() as f64);
        let certified = certified_mixture_log_likelihood(&pts, &w, 1, floor, false, mass)
            .expect("a single Gaussian on spread rows certifies");
        let at_closed_form = mixture_negative_log_likelihood(
            &pts,
            &w,
            1,
            false,
            &Array1::from_vec(vec![mean, 0.5 * variance.ln()]),
        );
        let band = at_closed_form
            .decrement_bands
            .as_ref()
            .expect("the mixture jet carries its bands")
            .objective;
        assert!(
            (-at_closed_form.value - closed_form).abs() <= band,
            "the jet's value at the closed-form maximum must be ℓ* within its band"
        );
        assert!(
            (certified - closed_form).abs() <= 2.0 * band,
            "certified ℓ̂ = {certified} vs closed-form ℓ* = {closed_form} (band {band:e})"
        );
    }

    /// #4237 — 84 evenly spaced points on the circle ARE the uniform law. A
    /// single wrapped Gaussian's likelihood rises with its width all the way to
    /// the uniform limit, so its certified maximum sits on the width ceiling at
    /// `ℓ̂ = 0` to within the band; the old plug-in scored it at the rows'
    /// Lloyd spread, `ℓ = −3.31`. The law is `Uniform`.
    #[test]
    fn uniform_circle_single_anchor_maximum_is_the_uniform_limit_4237() {
        let pts: Vec<f64> = (0..84).map(|r| r as f64 / 84.0).collect();
        let w = vec![1.0; pts.len()];
        let certified = certified_mixture_log_likelihood(&pts, &w, 1, 1.0 / 168.0, true, 84.0)
            .expect("the single wrapped Gaussian certifies at the width ceiling");
        let at_ceiling = mixture_negative_log_likelihood(
            &pts,
            &w,
            1,
            true,
            &Array1::from_vec(vec![0.5, wrapped_width_ceiling().ln()]),
        );
        let band = at_ceiling
            .decrement_bands
            .expect("the mixture jet carries its bands")
            .objective;
        assert!(
            certified.abs() <= 2.0 * band,
            "ℓ̂ = {certified} must be the uniform limit 0 (band {band:e})"
        );
        let weights = Array1::from_vec(w);
        assert_eq!(
            classify_occupancy_weighted(&pts, weights.view()),
            OccupancyLaw::Uniform
        );
    }
}
