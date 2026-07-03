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
            coords.iter().map(|&t| t.rem_euclid(*period) / *period).collect()
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
        uniformity_statistic: uniformity
            .as_ref()
            .map(|u| u.statistic)
            .unwrap_or(f64::NAN),
        uniformity_p_value: uniformity
            .as_ref()
            .map(|u| u.p_value)
            .unwrap_or(f64::NAN),
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
    }))
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
            raw.iter().map(|&t| ((t - lo) / span).clamp(0.0, 1.0)).collect()
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
    /// #2081 — aggregate coordinate-uniformity score over the fit's `d = 1`
    /// atoms: the MEAN Watson `U²` uniformity statistic across atoms that carry a
    /// `d = 1` circle/interval chart (LOWER ⟺ more uniform coordinates). `None`
    /// when no atom carries such a chart, which makes the seed-selection
    /// tie-break ([`prefer_candidate_basin`]) inert.
    ///
    /// Reads only the fitted coordinates + each atom's fixed topology — no basis
    /// evaluation — so it is cheap enough to call at every incumbent-comparison
    /// boundary in the fit loop.
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
            if let Some(uniformity) = coordinate_uniformity(coords.column(0), &topology) {
                if uniformity.statistic.is_finite() {
                    sum += uniformity.statistic;
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
    use crate::manifold::{SaeBasisEvaluator, SAE_FINAL_EV_DEGRADATION_TOL};
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
        assert!((u0 - ur).abs() < 1e-9, "rotation must not change U²: {u0} vs {ur}");
        assert!((u0 - uf).abs() < 1e-9, "reflection must not change U²: {u0} vs {uf}");
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
        assert!(prefer_candidate_basin(0.90, Some(0.5), 0.80, Some(0.01), tol));
        // Strictly worse EV always loses, regardless of uniformity.
        assert!(!prefer_candidate_basin(0.80, Some(0.01), 0.90, Some(0.5), tol));
        // Near-equal EV: lower U² (more uniform) wins.
        assert!(prefer_candidate_basin(0.90, Some(0.02), 0.9005, Some(0.20), tol));
        // Near-equal EV: higher U² loses.
        assert!(!prefer_candidate_basin(0.90, Some(0.20), 0.9005, Some(0.02), tol));
        // Near-equal EV, equal uniformity: keep incumbent (no thrash).
        assert!(!prefer_candidate_basin(0.90, Some(0.05), 0.90, Some(0.05), tol));
        // No certificate on either side: tie-break inert.
        assert!(!prefer_candidate_basin(0.90, None, 0.90, Some(0.05), tol));
        // Non-finite candidate EV never preferred.
        assert!(!prefer_candidate_basin(f64::NAN, Some(0.0), 0.5, Some(0.5), tol));
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
        assert!(ua < ub, "honest chart must read a more uniform angle: {ua} vs {ub}");
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
            assert!(circ < 1e-6, "u_arc must equal raw t on a unit-speed circle: {circ}");
        }
        assert!(reading.speed_cv < 1e-6, "flat speed ⇒ ~zero CV, got {}", reading.speed_cv);
        assert!((reading.min_speed_over_mean - 1.0).abs() < 1e-6);
        assert!((reading.max_speed_over_mean - 1.0).abs() < 1e-6);
        assert_eq!(
            angle_fidelity_verdict(Some(&reading)),
            AngleFidelityVerdict::ArcLengthHonest
        );
        let (rms, max) =
            raw_vs_arclength_defect(rows.view(), reading.coords_u_arc.view(), &circle(), true);
        assert!(rms < 1e-6 && max < 1e-6, "honest chart has ~zero raw defect: rms={rms} max={max}");
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
        use crate::chart_canonicalization::{
            ChartArcLengthReading, UNIT_SPEED_INLOOP_DEFECT_TOL,
        };
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
        assert_eq!(angle_fidelity_verdict(None), AngleFidelityVerdict::Degenerate);
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
        assert!((rms0 - rms_rot).abs() < 1e-9, "rotation must not change the defect: {rms0} vs {rms_rot}");
        assert!((rms0 - rms_ref).abs() < 1e-9, "reflection must not change the defect: {rms0} vs {rms_ref}");
    }
}
