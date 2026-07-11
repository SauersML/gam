//! #1019 stage 1 — post-fit arc-length (unit-speed) chart canonicalization
//! for `d = 1` manifold atoms.
//!
//! The failure being cured: a planted circle fit can compress the full loop
//! into a sliver of coordinate span at near-perfect reconstruction EV — the
//! IMAGE is right, the CHART is arbitrary, and the gauge-invariant smoothness
//! penalty makes every chart equal-cost by design, so no reconstruction
//! metric can detect the dishonesty. For `d = 1` the cure is exact:
//! arc-length reparameterization picks the unit-speed representative of the
//! `Diff(S¹)` / `Diff([0, 1])` orbit, image-frozen, no refit.
//!
//! Three contracts, asserted against self-constructed truth (the planted
//! warped chart — #904 reference-as-truth):
//! * (a) a deliberately warped-chart planted circle has constant coordinate
//!   speed `‖γ̃'(t̃)‖` within 1% after canonicalization, and the canonical
//!   chart spans the full period within 10%;
//! * (b) image invariance — reconstruction EV unchanged within 1e-8;
//! * (c) the residual-gauge certificate reports the chart pinned with the
//!   `PinnedByCanonicalization` provenance and names the finite isometry
//!   group (`O(2)` for the circle).
//! Plus the interval topology (warped-chart line segment): constant speed,
//! unit-interval span, exact image invariance.

use faer::Side as FaerSide;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::terms::latent::LatentManifold;
use gam::terms::sae::identifiability::{GeneratorFamily, VerdictProvenance};
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator,
    sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, Array4, Array5, ArrayView2};
use std::sync::Arc;

const N: usize = 180;
const P: usize = 6;
/// Constant + 12 harmonics: enough for the planted warp's Jacobi–Anger tail
/// (`J_k(0.9)` for `k ≥ 12` is ~1e-13) to sit far below the recomposition
/// tolerance — the planted curve is an exact circle to ~1e-12.
const M_CIRCLE: usize = 25;
/// Warp strength of the planted dishonest chart: the honest angle is
/// `u(t) = t + (A/2π)·sin(2πt)`, a degree-1 circle diffeomorphism with
/// `u'(t) = 1 + A·cos(2πt) ∈ [1−A, 1+A]` — a 19× coordinate-speed swing.
const WARP_A: f64 = 0.9;

/// Deterministic orthonormal `P × 2` frame for the planted circle plane.
fn planted_frame() -> Array2<f64> {
    let mut raw = Array2::<f64>::zeros((P, 2));
    for j in 0..2 {
        for i in 0..P {
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((P, 2));
    for j in 0..2 {
        let mut v = raw.column(j).to_owned();
        for prev in 0..j {
            let qp = q.column(prev);
            let dot: f64 = qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for i in 0..P {
                v[i] -= dot * qp[i];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..P {
            q[[i, j]] = v[i] / nrm;
        }
    }
    q
}

/// Honest angle (fraction of one turn) at warped chart coordinate `t`.
fn honest_angle(t: f64) -> f64 {
    t + WARP_A * (std::f64::consts::TAU * t).sin() / std::f64::consts::TAU
}

/// Exact-circle point at honest angle `u`, embedded through the frame.
fn circle_point(u: f64, frame: &Array2<f64>) -> Vec<f64> {
    let ang = std::f64::consts::TAU * u;
    let (c, s) = (ang.cos(), ang.sin());
    (0..P)
        .map(|i| c * frame[[i, 0]] + s * frame[[i, 1]])
        .collect()
}

/// Exact least squares `B = argmin ‖Φ B − Y‖²` via normal equations (the
/// harmonic design on a dense uniform grid is essentially orthogonal, so the
/// Cholesky is rock-solid; the relative jitter only guards the factorization).
fn least_squares_decoder(phi: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let m = phi.ncols();
    let mut xtx = fast_ata(phi);
    let mut trace = 0.0_f64;
    for i in 0..m {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / m as f64).max(1.0) * 1.0e-12;
    for i in 0..m {
        xtx[[i, i]] += jitter;
    }
    let xty = fast_atb(phi, y);
    xtx.cholesky(FaerSide::Lower)
        .expect("decoder LS Cholesky")
        .solve_mat(&xty)
}

/// Per-row decoded coordinate speed `‖Φ'(t_i) B‖₂` straight off the atom's
/// stored jet — the quantity the unit-speed contract is about.
fn row_speeds(term: &SaeManifoldTerm) -> Vec<f64> {
    let atom = &term.atoms[0];
    (0..term.n_obs())
        .map(|row| {
            let tangent = atom.decoded_derivative_row(row, 0);
            tangent.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect()
}

/// Reconstruction EV of the atom-level decode against the planted target.
fn reconstruction_ev(term: &SaeManifoldTerm, z: &Array2<f64>) -> f64 {
    let n = z.nrows();
    let p = z.ncols();
    let mut mean = vec![0.0_f64; p];
    for i in 0..n {
        for j in 0..p {
            mean[j] += z[[i, j]] / n as f64;
        }
    }
    let mut resid_sq = 0.0_f64;
    let mut var_sq = 0.0_f64;
    for i in 0..n {
        let decoded = term.atoms[0].decoded_row(i);
        for j in 0..p {
            let r = z[[i, j]] - decoded[j];
            resid_sq += r * r;
            let c = z[[i, j]] - mean[j];
            var_sq += c * c;
        }
    }
    1.0 - resid_sq / var_sq
}

/// Build the planted warped-chart circle term: rows sit at chart coordinates
/// `t_i` (uniform — the chart looks innocuous), the decoder is LS-fitted on a
/// dense grid so the decoded curve is the exact circle traversed at the
/// dishonest, 19×-varying coordinate speed `u'(t)`.
fn planted_warped_circle() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let frame = planted_frame();
    let evaluator = PeriodicHarmonicEvaluator::new(M_CIRCLE).expect("evaluator");

    // Dense plant grid: decoder reproduces the warped-chart circle as a curve.
    let g = 2048usize;
    let grid = Array2::from_shape_fn((g, 1), |(j, _)| j as f64 / g as f64);
    let (grid_phi, grid_jet) = evaluator.evaluate(grid.view()).expect("grid basis");
    assert_eq!(grid_jet.dim(), (g, M_CIRCLE, 1));
    let mut grid_y = Array2::<f64>::zeros((g, P));
    for j in 0..g {
        let point = circle_point(honest_angle(grid[[j, 0]]), &frame);
        for i in 0..P {
            grid_y[[j, i]] = point[i];
        }
    }
    let decoder = least_squares_decoder(&grid_phi, &grid_y);

    // Rows: uniform in the WARPED chart, targets on the exact circle.
    let coords = Array2::from_shape_fn((N, 1), |(i, _)| (i as f64 + 0.5) / N as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("row basis");
    let mut z = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        let point = circle_point(honest_angle(coords[[i, 0]]), &frame);
        for j in 0..P {
            z[[i, j]] = point[j];
        }
    }

    let atom = SaeManifoldAtom::new(
        "warped-circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(M_CIRCLE),
    )
    .expect("atom")
    .with_basis_evaluator(Arc::new(evaluator));

    let logits = Array2::<f64>::zeros((N, 1));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.5, 1.0, false),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); 1]);
    (term, z, rho)
}

#[test]
fn warped_circle_chart_canonicalizes_to_unit_speed_full_span() {
    let (mut term, z, rho) = planted_warped_circle();

    // The planted chart is genuinely dishonest: coordinate speed swings by
    // more than 10× across rows (the test is vacuous otherwise).
    let speeds_before = row_speeds(&term);
    let max_before = speeds_before.iter().cloned().fold(0.0_f64, f64::max);
    let min_before = speeds_before.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_before / min_before > 10.0,
        "planted chart must be speed-dishonest; got ratio {}",
        max_before / min_before
    );

    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(
        term.atoms[0].chart_canonicalized,
        "the planted circle atom must be canonicalized (exact harmonic closure)"
    );

    // (a) Unit speed within 1%: the canonical chart traverses the circle at
    // constant coordinate speed.
    let speeds_after = row_speeds(&term);
    let max_after = speeds_after.iter().cloned().fold(0.0_f64, f64::max);
    let min_after = speeds_after.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_after / min_after < 1.01,
        "canonical chart speed must be constant within 1%; got ratio {}",
        max_after / min_after
    );

    // (a) Full chart span within 10% of one period (period = 1: the harmonic
    // evaluator's fraction-of-period convention — the codebase's native unit
    // for what the issue states as 2π radians).
    let coords_after = term.assignment.coords[0].as_matrix();
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for i in 0..N {
        t_min = t_min.min(coords_after[[i, 0]]);
        t_max = t_max.max(coords_after[[i, 0]]);
    }
    assert!(
        t_max - t_min > 0.9,
        "canonical chart must span the full period within 10%; got [{t_min}, {t_max}]"
    );
}

#[test]
fn circle_canonicalization_freezes_the_image() {
    let (mut term, z, rho) = planted_warped_circle();

    let ev_before = reconstruction_ev(&term, &z);
    assert!(
        ev_before > 0.99,
        "the plant must reconstruct the circle nearly perfectly; got EV {ev_before}"
    );

    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(term.atoms[0].chart_canonicalized);

    // (b) Image invariance: reconstruction EV unchanged within 1e-8 — the
    // canonicalization moved the chart, never the curve.
    let ev_after = reconstruction_ev(&term, &z);
    assert!(
        (ev_before - ev_after).abs() <= 1.0e-8,
        "canonicalization must be image-frozen: EV {ev_before} -> {ev_after}"
    );
}

/// #1227 regression: the hybrid-split report must be computed AFTER chart
/// canonicalization, not before. The split stores a per-atom linear image
/// `b0 + (t − t̄) b1` keyed to the coordinate `t` in use; canonicalization is a
/// (nonlinear) reparameterization of `t`, so a report computed before the
/// reparameterization records the line in a coordinate system that is then
/// replaced. The fix runs the reparameterization first, then the split — so the
/// stored report must equal a fresh report recomputed against the post-
/// canonicalization (current) coordinate. On the warped circle the canonical
/// coordinate differs materially from the fitted one (19× speed warp), so a
/// stale pre-canonicalization report would NOT match this recompute.
#[test]
fn hybrid_split_report_is_computed_after_chart_canonicalization() {
    let (mut term, z, rho) = planted_warped_circle();

    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(
        term.atoms[0].chart_canonicalized,
        "the warped circle must canonicalize for this regression to be meaningful"
    );

    let stored = term
        .hybrid_split_report()
        .cloned()
        .expect("post-fit canonicalization must store a hybrid-split report");

    // Recompute against the CURRENT (canonical) coordinate. If the stored report
    // had been computed before canonicalization (the #1227 bug), its
    // chart-dependent verdict fields would disagree with this recompute.
    let recomputed = term
        .compute_hybrid_split_report(&rho, Some(z.view()))
        .expect("recompute hybrid split against canonical coords")
        .expect("recomputed report present");

    assert_eq!(
        stored.verdicts.len(),
        recomputed.verdicts.len(),
        "stored vs recomputed verdict count"
    );
    for (s, r) in stored.verdicts.iter().zip(recomputed.verdicts.iter()) {
        assert_eq!(s.atom_name, r.atom_name);
        assert_eq!(
            s.kept_curved, r.kept_curved,
            "atom '{}' curved/linear verdict must match the canonical-coord recompute",
            s.atom_name
        );
        match (&s.linear_image, &r.linear_image) {
            (Some(si), Some(ri)) => assert!(
                (si.t_bar - ri.t_bar).abs() <= 1.0e-9,
                "atom '{}' linear-image t̄ must be in the canonical coordinate: \
                 stored {} vs recomputed {}",
                s.atom_name,
                si.t_bar,
                ri.t_bar
            ),
            (None, None) => {}
            _ => panic!(
                "atom '{}' linear-image presence must match the canonical recompute",
                s.atom_name
            ),
        }
    }
    assert_eq!(
        stored.selection.curved_atom_count, recomputed.selection.curved_atom_count,
        "curved-atom count must match the canonical-coord recompute"
    );
}

#[test]
fn certificate_reports_chart_pinned_by_canonicalization_with_finite_group() {
    let (mut term, z, rho) = planted_warped_circle();
    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(term.atoms[0].chart_canonicalized);

    // (c) The residual-gauge certificate downgrades the atom's chart freedom
    // to the finite isometry group, with the canonicalization provenance —
    // distinct from curvature/penalty pinning.
    let fitted = term
        .try_fitted_target_aware(z.view(), Some(&rho))
        .expect("target-aware reconstruction");
    let report = term
        .fit_diagnostics_report(None, false, None, fitted.view(), None)
        .expect("diagnostics")
        .residual_gauge;
    let chart = report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::ChartReparameterization)
        .expect("certificate must carry the chart-reparameterization record");
    assert!(
        !chart.unpinned,
        "the canonicalized chart freedom must be reported pinned"
    );
    assert_eq!(
        chart.provenance,
        VerdictProvenance::PinnedByCanonicalization,
        "the chart pin's provenance must be the canonicalization, not curvature"
    );
    assert!(
        chart.description.contains("O(2)"),
        "the record must name the finite isometry group O(2); got: {}",
        chart.description
    );
    assert!(
        report.summary.contains("pinned to arc length"),
        "the summary must surface the canonicalization; got: {}",
        report.summary
    );
    // Every other verdict still carries the curvature-test provenance.
    for g in &report.generators {
        if g.family != GeneratorFamily::ChartReparameterization {
            assert_eq!(g.provenance, VerdictProvenance::CurvatureTest);
        }
    }
}

// ─── Interval topology ──────────────────────────────────────────────────────

/// Monomial basis `{1, t, t², t³}` on the line — the minimal interval-chart
/// evaluator. Second/third jets are declared unavailable (the
/// canonicalization needs only `Φ` and `Φ'`).
#[derive(Debug)]
struct MonomialEvaluator;

impl SaeBasisEvaluator for MonomialEvaluator {
    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        let n = coords.nrows();
        if coords.ncols() != 1 {
            return Err(format!(
                "MonomialEvaluator: expected latent_dim 1, got {}",
                coords.ncols()
            ));
        }
        let mut phi = Array2::<f64>::zeros((n, 4));
        let mut jet = Array3::<f64>::zeros((n, 4, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = t;
            phi[[row, 2]] = t * t;
            phi[[row, 3]] = t * t * t;
            jet[[row, 1, 0]] = 1.0;
            jet[[row, 2, 0]] = 2.0 * t;
            jet[[row, 3, 0]] = 3.0 * t * t;
        }
        Ok((phi, jet))
    }

    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        let n = coords.nrows();
        let mut h = Array4::<f64>::zeros((n, 4, 1, 1));
        for row in 0..n {
            let t = coords[[row, 0]];
            h[[row, 2, 0, 0]] = 2.0;
            h[[row, 3, 0, 0]] = 6.0 * t;
        }
        Some(Ok(h))
    }

    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        let n = coords.nrows();
        let mut t3 = Array5::<f64>::zeros((n, 4, 1, 1, 1));
        for row in 0..n {
            t3[[row, 3, 0, 0, 0]] = 6.0;
        }
        Some(Ok(t3))
    }
}

#[test]
fn warped_interval_chart_canonicalizes_to_unit_speed_unit_span() {
    // Planted dishonest interval chart: the decoded curve is a straight
    // segment traversed at speed ∝ 3t² (the deliverable's `t = u³` warp read
    // from the chart side) — a 25× speed swing over the fitted range.
    let n = 120usize;
    let lo = 0.2_f64;
    let hi = 1.0_f64;
    let mut direction = vec![0.0_f64; P];
    let mut base = vec![0.0_f64; P];
    for i in 0..P {
        direction[i] = ((i as f64) * 0.71 + 0.3).sin() + 0.4;
        base[i] = 0.2 * (i as f64) - 0.5;
    }

    // Decoder: γ(t) = base + direction · t³ — rows {const, t, t², t³}.
    let mut decoder = Array2::<f64>::zeros((4, P));
    for i in 0..P {
        decoder[[0, i]] = base[i];
        decoder[[3, i]] = direction[i];
    }

    let coords = Array2::from_shape_fn((n, 1), |(i, _)| lo + (hi - lo) * i as f64 / (n - 1) as f64);
    let evaluator = MonomialEvaluator;
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("row basis");
    let mut z = Array2::<f64>::zeros((n, P));
    for row in 0..n {
        let t = coords[[row, 0]];
        for i in 0..P {
            z[[row, i]] = base[i] + direction[i] * t * t * t;
        }
    }

    let atom = SaeManifoldAtom::new(
        "warped-segment".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(4),
    )
    .expect("atom")
    .with_basis_evaluator(Arc::new(MonomialEvaluator));

    let logits = Array2::<f64>::zeros((n, 1));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::ibp_map(0.5, 1.0, false),
    )
    .expect("assignment");
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); 1]);

    let speeds_before = row_speeds(&term);
    let max_before = speeds_before.iter().cloned().fold(0.0_f64, f64::max);
    let min_before = speeds_before.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_before / min_before > 10.0,
        "planted interval chart must be speed-dishonest; got ratio {}",
        max_before / min_before
    );
    let ev_before = reconstruction_ev(&term, &z);

    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(
        term.atoms[0].chart_canonicalized,
        "the straight-segment atom must canonicalize exactly (monomial closure)"
    );

    // Constant speed within 1% (here: exactly, up to roundoff — a straight
    // line's arc length IS its linear coordinate).
    let speeds_after = row_speeds(&term);
    let max_after = speeds_after.iter().cloned().fold(0.0_f64, f64::max);
    let min_after = speeds_after.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_after / min_after < 1.01,
        "canonical interval chart speed must be constant within 1%; got ratio {}",
        max_after / min_after
    );

    // Canonical chart spans [0, 1] within 10%.
    let coords_after = term.assignment.coords[0].as_matrix();
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for i in 0..n {
        t_min = t_min.min(coords_after[[i, 0]]);
        t_max = t_max.max(coords_after[[i, 0]]);
    }
    assert!(
        t_min.abs() < 0.05 && (t_max - 1.0).abs() < 0.05 && t_max - t_min > 0.9,
        "canonical interval chart must span [0, 1] within 10%; got [{t_min}, {t_max}]"
    );

    // Image-frozen.
    let ev_after = reconstruction_ev(&term, &z);
    assert!(
        (ev_before - ev_after).abs() <= 1.0e-8,
        "interval canonicalization must be image-frozen: EV {ev_before} -> {ev_after}"
    );
}
