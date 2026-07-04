//! #2081 — chart angle-fidelity regression.
//!
//! A `K = 1` circle chart can reconstruct its ring at high EV while reading a
//! materially NON-uniform angle coordinate: the `n_basis = 4` Fourier chart has
//! the capacity to fit a warped (non-arc-length) parameterization that
//! reconstructs the ring almost as well as the honest arc-length chart, and
//! reconstruction EV cannot see the difference. The multi-start keep-best /
//! seed-selection tie-break therefore needs a coordinate-quality price that
//! reconstruction EV does not supply.
//!
//! This module pins the mechanism at the STATISTIC level (no `SaeManifoldTerm`
//! construction): it builds two candidate charts of the SAME clustered-circle
//! data — an honest arc-length chart and a warped chart — and asserts that the
//! statistic the seed-selection tie-break prices
//! ([`SaeManifoldTerm::coordinate_uniformity_aggregate`], the mean
//! [`chart_unit_speed_defect`]) ranks the honest chart above the warped one,
//! whereas the raw-coordinate Watson `U²` occupancy statistic used before the
//! fix INVERTS that ranking (it prefers the warped chart, whose warped raw
//! coordinate merely looks more uniformly spread). The honest chart the fixed
//! tie-break selects reads an angle coordinate that clears the 0.8 correlation
//! bar; the warped raw coordinate does not, and only the arc-length correction
//! recovers it.

use ndarray::{Array1, Array2, Array3, Array4, Array5, ArrayView2};

use crate::chart_canonicalization::{
    CanonicalChartTopology, chart_arclength_coordinates, chart_unit_speed_defect,
};
use crate::manifold::SaeBasisEvaluator;

use super::coordinate_fidelity::watson_u2_uniform;

/// A circle-harmonic evaluator: `Φ(t) = [cos 2πt, sin 2πt, cos 4πt, sin 4πt, …]`
/// up to `harmonics` frequencies (period `1.0`, fraction-of-period convention).
#[derive(Debug)]
struct CircleHarmonicEvaluator {
    harmonics: usize,
}

impl SaeBasisEvaluator for CircleHarmonicEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
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

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        if coords.ncols() != 1 {
            return Some(Err("CircleHarmonicEvaluator: d = 1 evaluator".to_string()));
        }
        None
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        if coords.ncols() != 1 {
            return Some(Err("CircleHarmonicEvaluator: d = 1 evaluator".to_string()));
        }
        None
    }
}

fn circle() -> CanonicalChartTopology {
    CanonicalChartTopology::Circle { period: 1.0 }
}

/// Least-squares decoder `B` (m×2) mapping `Φ(t_i) ↦ target_i` via the normal
/// equations `(ΦᵀΦ) B = Φᵀ target`, solved with a small SPD Gaussian
/// elimination (design widths are 2 or 4 here).
fn fit_decoder(evaluator: &CircleHarmonicEvaluator, t: &[f64], target: &Array2<f64>) -> Array2<f64> {
    let coords = Array2::from_shape_fn((t.len(), 1), |(i, _)| t[i]);
    let (phi, _) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    // Normal equations.
    let mut ata = vec![vec![0.0_f64; m]; m];
    let mut aty = vec![vec![0.0_f64; 2]; m];
    for r in 0..phi.nrows() {
        for i in 0..m {
            for j in 0..m {
                ata[i][j] += phi[[r, i]] * phi[[r, j]];
            }
            for c in 0..2 {
                aty[i][c] += phi[[r, i]] * target[[r, c]];
            }
        }
    }
    // Solve [ata | aty] by Gaussian elimination with partial pivoting.
    let mut aug: Vec<Vec<f64>> = (0..m)
        .map(|i| ata[i].iter().copied().chain(aty[i].iter().copied()).collect())
        .collect();
    for col in 0..m {
        let piv = (col..m).max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs())).unwrap();
        aug.swap(col, piv);
        let d = aug[col][col];
        for v in aug[col].iter_mut() {
            *v /= d;
        }
        for r in 0..m {
            if r != col {
                let f = aug[r][col];
                for k in 0..(m + 2) {
                    aug[r][k] -= f * aug[col][k];
                }
            }
        }
    }
    Array2::from_shape_fn((m, 2), |(i, c)| aug[i][m + c])
}

fn reconstruction_ev(evaluator: &CircleHarmonicEvaluator, t: &[f64], b: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let coords = Array2::from_shape_fn((t.len(), 1), |(i, _)| t[i]);
    let (phi, _) = evaluator.evaluate(coords.view()).unwrap();
    let pred = phi.dot(b);
    let mean: Vec<f64> = (0..2)
        .map(|c| target.column(c).sum() / target.nrows() as f64)
        .collect();
    let mut sse = 0.0;
    let mut tss = 0.0;
    for r in 0..target.nrows() {
        for c in 0..2 {
            sse += (pred[[r, c]] - target[[r, c]]).powi(2);
            tss += (target[[r, c]] - mean[c]).powi(2);
        }
    }
    1.0 - sse / tss
}

/// Circular correlation of two coordinates on the unit interval (period 1),
/// invariant to the residual `O(2)` gauge (base-point rotation + orientation
/// reflection): the mean resultant length of the angular differences, maximized
/// over the reflection sign. `1` ⟺ the two coordinates agree up to rotation and
/// reflection; `0` ⟺ unrelated. This is exactly the gauge the angle-fidelity
/// claim must be measured in (a raw Pearson correlation of a wrapping coordinate
/// is meaningless).
fn circular_corr(a: &[f64], b: &[f64]) -> f64 {
    let tau = std::f64::consts::TAU;
    let mut best = -1.0_f64;
    for sgn in [1.0_f64, -1.0] {
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for i in 0..a.len() {
            let d = tau * (sgn * a[i] - b[i]);
            re += d.cos();
            im += d.sin();
        }
        let r = (re * re + im * im).sqrt() / a.len() as f64;
        best = best.max(r);
    }
    best
}

/// The two candidate charts of the same clustered-circle data.
///
/// Data: unit-circle points at NON-uniform true angles `φ_i = 2π (i/N)^1.6`
/// (accelerating occupancy — genuinely clustered, not a pathology).
///
///  * HONEST chart: a pure first harmonic (`n_basis = 2`) with the raw coordinate
///    set to the true angle. Its parameterization is exactly arc-length (constant
///    speed), so raw `t` IS the honest angle — but the raw coordinate is CLUSTERED
///    (the data is), so its Watson `U²` is high.
///  * WARPED chart: a two-harmonic (`n_basis = 4`) least-squares fit with a
///    UNIFORM raw coordinate `t_i = i/N`. It reconstructs the same ring at high EV
///    but its parameterization squishes arc length (high arc-length defect), and
///    its uniform raw coordinate reads a LOW Watson `U²` even though it is
///    dishonest — the exact #2081 trap.
struct TwoCharts {
    honest_ev: CircleHarmonicEvaluator,
    honest_b: Array2<f64>,
    honest_t: Vec<f64>,
    warped_ev: CircleHarmonicEvaluator,
    warped_b: Array2<f64>,
    warped_t: Vec<f64>,
    truth: Vec<f64>,
}

fn build_two_charts() -> TwoCharts {
    let n = 300usize;
    let tau = std::f64::consts::TAU;
    let phi: Vec<f64> = (0..n).map(|i| tau * (i as f64 / n as f64).powf(1.6)).collect();
    let target = Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { phi[i].cos() } else { phi[i].sin() });
    let truth: Vec<f64> = phi.iter().map(|&p| (p / tau).rem_euclid(1.0)).collect();

    // Honest chart: pure first harmonic, raw coordinate = true angle. The
    // least-squares decoder is the 2×2 identity here (Φ(φ/2π) == the target), so
    // fit it directly to keep the honesty exact.
    let honest_ev = CircleHarmonicEvaluator { harmonics: 1 };
    let honest_t = truth.clone();
    let honest_b = fit_decoder(&honest_ev, &honest_t, &target);

    // Warped chart: two harmonics, UNIFORM raw coordinate.
    let warped_ev = CircleHarmonicEvaluator { harmonics: 2 };
    let warped_t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let warped_b = fit_decoder(&warped_ev, &warped_t, &target);

    TwoCharts { honest_ev, honest_b, honest_t, warped_ev, warped_b, warped_t, truth }
}

/// The fix's statistic (mean chart arc-length defect) prefers the honest chart,
/// while the pre-fix statistic (raw-coordinate Watson `U²`) prefers the WARPED
/// chart — the ranking inversion that let a warped `n_basis = 4` chart win the
/// seed-selection tie-break at equal EV and read a non-uniform angle (#2081).
#[test]
fn arclength_defect_prices_chart_honesty_where_watson_u2_inverts() {
    let c = build_two_charts();

    // Both charts reconstruct the ring well (the tie-break regime: EV cannot
    // separate them, so the coordinate-quality price must).
    let ev_honest = reconstruction_ev(&c.honest_ev, &c.honest_t, &c.honest_b, &{
        let n = c.truth.len();
        let tau = std::f64::consts::TAU;
        Array2::from_shape_fn((n, 2), |(i, col)| {
            let p = tau * c.truth[i];
            if col == 0 { p.cos() } else { p.sin() }
        })
    });
    // (Rebuild target once for the warped EV too.)
    let n = c.truth.len();
    let tau = std::f64::consts::TAU;
    let target = Array2::from_shape_fn((n, 2), |(i, col)| {
        let p = tau * c.truth[i];
        if col == 0 { p.cos() } else { p.sin() }
    });
    let ev_warped = reconstruction_ev(&c.warped_ev, &c.warped_t, &c.warped_b, &target);
    assert!(ev_honest > 0.99, "honest chart must reconstruct exactly, EV={ev_honest}");
    assert!(ev_warped > 0.90, "warped chart must still reconstruct the ring, EV={ev_warped}");

    // --- The fix's statistic: chart arc-length (unit-speed) defect. ---
    let honest_coords = Array1::from(c.honest_t.clone());
    let warped_coords = Array1::from(c.warped_t.clone());
    let defect_honest = chart_unit_speed_defect(
        &c.honest_ev,
        c.honest_b.view(),
        honest_coords.view(),
        &circle(),
    )
    .unwrap()
    .expect("honest chart defect");
    let defect_warped = chart_unit_speed_defect(
        &c.warped_ev,
        c.warped_b.view(),
        warped_coords.view(),
        &circle(),
    )
    .unwrap()
    .expect("warped chart defect");
    assert!(
        defect_honest < 0.05,
        "the arc-length chart must have ~zero defect, got {defect_honest}"
    );
    assert!(
        defect_warped > 0.20,
        "the warped chart must have a large arc-length defect (the #2081 pathology), got {defect_warped}"
    );
    // The seed-selection statistic (LOWER wins) correctly prefers the honest chart.
    assert!(
        defect_honest < defect_warped,
        "arc-length defect must rank honest ({defect_honest}) above warped ({defect_warped})"
    );

    // --- The pre-fix statistic: raw-coordinate Watson U² (LOWER wins). ---
    let watson_honest = watson_u2_uniform(&c.honest_t).statistic;
    let watson_warped = watson_u2_uniform(&c.warped_t).statistic;
    // The warped chart's UNIFORM raw coordinate reads a LOWER Watson U² than the
    // honest chart's clustered raw coordinate: the occupancy statistic INVERTS the
    // honesty ranking and would pick the warped chart — the bug this fix removes.
    assert!(
        watson_warped < watson_honest,
        "pre-fix Watson U² must invert the ranking (warped {watson_warped} < honest {watson_honest}) — \
         this is the #2081 trap the arc-length defect fixes"
    );
}

/// The honest chart the fixed tie-break selects reads an angle coordinate that
/// clears the 0.8 correlation bar; the warped raw coordinate reads a materially
/// worse angle, and only the arc-length correction recovers it.
#[test]
fn selected_honest_chart_reads_angle_correlation_above_bar() {
    let c = build_two_charts();

    // The honest (arc-length) chart's RAW coordinate is the honest angle.
    let honest_raw_corr = circular_corr(&c.honest_t, &c.truth);
    assert!(
        honest_raw_corr >= 0.8,
        "the arc-length chart the fix selects must read angle correlation ≥ 0.8, got {honest_raw_corr}"
    );

    // The warped chart's raw coordinate reads a strictly worse angle than the
    // honest one — reconstruction EV bought a non-uniform parameterization.
    let warped_raw_corr = circular_corr(&c.warped_t, &c.truth);
    assert!(
        warped_raw_corr < honest_raw_corr,
        "the warped raw coordinate must read a worse angle ({warped_raw_corr}) than the honest one ({honest_raw_corr})"
    );

    // The arc-length correction recovers the angle the warped raw coordinate lost:
    // reading the warped chart through its arc-length coordinate clears the bar.
    let warped_coords = Array1::from(c.warped_t.clone());
    let reading = chart_arclength_coordinates(
        &c.warped_ev,
        c.warped_b.view(),
        warped_coords.view(),
        &circle(),
    )
    .unwrap()
    .expect("warped chart arc-length reading");
    let u_arc: Vec<f64> = reading.coords_u_arc.to_vec();
    let warped_arc_corr = circular_corr(&u_arc, &c.truth);
    assert!(
        warped_arc_corr >= 0.8,
        "the arc-length-corrected warped coordinate must recover angle correlation ≥ 0.8, got {warped_arc_corr}"
    );
    assert!(
        warped_arc_corr > warped_raw_corr,
        "arc-length correction must improve the warped angle ({warped_raw_corr} → {warped_arc_corr})"
    );
}
