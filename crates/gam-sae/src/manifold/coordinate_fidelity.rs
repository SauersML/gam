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

use ndarray::ArrayView1;

use crate::chart_canonicalization::CanonicalChartTopology;

use super::SaeManifoldTerm;

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
    let Some(uniformity) = coordinate_uniformity(row_coords, &topology) else {
        return Ok(None);
    };
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
    let topology_label = match topology {
        CanonicalChartTopology::Circle { .. } => "circle",
        CanonicalChartTopology::Interval => "interval",
    };
    Ok(Some(AtomCoordinateFidelity {
        topology: topology_label,
        uniformity_statistic: uniformity.statistic,
        uniformity_p_value: uniformity.p_value,
        arclength_defect: defect.unwrap_or(f64::NAN),
        n_coords: uniformity.n,
    }))
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

    fn circle() -> CanonicalChartTopology {
        CanonicalChartTopology::Circle { period: 1.0 }
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
}
