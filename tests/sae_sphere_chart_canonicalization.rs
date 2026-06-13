//! #1019 sphere arm — post-fit chart canonicalization for `d = 2` sphere (S²)
//! atoms, at the term-level acceptance boundary.
//!
//! The failure being cured: a planted sphere image can reconstruct perfectly
//! while living in an anisotropically warped `(lat, lon)` chart. The image is
//! right, the chart is dishonest, and gauge-invariant smoothness cannot tell
//! two diffeomorphic parameterizations apart.
//!
//! The cure (issue #1019, acceptance item 1 for the sphere): pin the chart
//! post-fit to the minimum-isometry-defect representative against the
//! round-sphere reference metric `g_ref = diag(1, cos²lat)`, image frozen. The
//! sphere has no global pole-free flow basis (hairy ball), so the flow family
//! is the three CONFORMAL-BOOST fields (gradients of the degree-1 harmonics
//! x, y, z) — the non-isometric part of the Möbius group. Minimizing the
//! defect over the boosts breaks the conformal moduli down to the round
//! sphere's isometry group O(3), the finite residual the certificate reports.
//!
//! Contracts (against self-constructed truth, #904 reference-as-truth):
//! * (a) a deliberately warped-chart sphere has a sizeable round-sphere
//!   isometry defect, and a round-isometric chart has ≈0 defect — the issue's
//!   acceptance quantity, computed by the same code the post-fit pass runs;
//! * (b) the post-fit pass PINS the warped chart: the defect collapses (within
//!   10% of the round-chart optimum) and the image is frozen (EV unchanged);
//! * (c) the residual-gauge certificate reports the sphere chart
//!   `PinnedByCanonicalization` with residual freedom `Isom(S², round) = O(3)`.

use faer::Side as FaerSide;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::sae_identifiability::{GeneratorFamily, VerdictProvenance};
use gam::terms::latent_coord::LatentManifold;
use gam::terms::sae_chart_canonicalization::sphere_chart_isometry_defect;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm, SphereChartEvaluator,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const LAT_GRID: usize = 12;
const LON_GRID: usize = 16;
const N: usize = LAT_GRID * LON_GRID;
const P: usize = 3;
const D: usize = 2;
const DECODER_LAT: usize = 28;
const DECODER_LON: usize = 36;
// Interior latitude band the data lives in (round coordinates), kept well off
// the poles so the conformal-boost flow (1/cos lat) is well-conditioned.
const LAT_LO: f64 = -1.05;
const LAT_HI: f64 = 1.05;

#[derive(Clone, Copy)]
enum Chart {
    /// Round-isometric `(lat, lon)` chart: pullback metric is exactly
    /// `diag(1, cos²lat)` up to a global scale, so the round-sphere isometry
    /// defect is ≈0.
    Round,
    /// Conformal-boost warped chart: the latitude is pushed by a smooth zonal
    /// boost `lat ↦ lat + a·sin(lat)·cos(lat)` (a non-isometric conformal
    /// squeeze toward the equator). The image is the SAME physical sphere
    /// surface; only the parameterization is warped, so reconstruction cannot
    /// see the dishonesty but the isometry defect can.
    Warped,
}

/// The latitude reparameterization defining each chart. `Round` is the
/// identity; `Warped` applies a smooth, strictly-monotone zonal squeeze that
/// fixes the equator and both poles.
fn warp_lat(chart: Chart, lat: f64) -> f64 {
    match chart {
        Chart::Round => lat,
        Chart::Warped => lat + 0.40 * lat.sin() * lat.cos(),
    }
}

/// Inverse of [`warp_lat`] by monotone Newton iteration — recovers the chart
/// coordinate whose round latitude is `lat_round`.
fn inv_warp_lat(chart: Chart, lat_round: f64) -> f64 {
    match chart {
        Chart::Round => lat_round,
        Chart::Warped => {
            let mut x = lat_round;
            for _ in 0..50 {
                // d/dx [x + 0.40 sin x cos x] = 1 + 0.40 cos(2x)
                let f = x + 0.40 * x.sin() * x.cos() - lat_round;
                let df = 1.0 + 0.40 * (2.0 * x).cos();
                x -= f / df;
            }
            x
        }
    }
}

/// Round-sphere embedding point for a `(lat, lon)` pair in ROUND coordinates.
fn sphere_point(lat_round: f64, lon: f64) -> [f64; P] {
    [
        lat_round.cos() * lon.cos(),
        lat_round.cos() * lon.sin(),
        lat_round.sin(),
    ]
}

/// Exact least squares `B = argmin ||Phi B - Y||²` via jittered normal
/// equations (mirrors the torus acceptance test's decoder fit).
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

/// Deterministic interior `(lat, lon)` sampling grid in CHART coordinates: the
/// same physical sphere band sampled in each chart's own parameterization.
fn chart_coords(chart: Chart) -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((N, D));
    for i in 0..LAT_GRID {
        let lat_round = LAT_LO + (LAT_HI - LAT_LO) * (i as f64 + 0.5) / LAT_GRID as f64;
        let lat_chart = inv_warp_lat(chart, lat_round);
        for j in 0..LON_GRID {
            let row = i * LON_GRID + j;
            let lon = std::f64::consts::TAU * (j as f64 + 0.5) / LON_GRID as f64
                - std::f64::consts::PI;
            coords[[row, 0]] = lat_chart;
            coords[[row, 1]] = lon;
        }
    }
    coords
}

fn atom_name(chart: Chart) -> &'static str {
    match chart {
        Chart::Round => "round-sphere",
        Chart::Warped => "warped-sphere",
    }
}

/// Build a planted sphere atom in the requested chart: decode the round-sphere
/// image through the [`SphereChartEvaluator`] harmonic basis, fitted by exact
/// LS on a dense grid, then evaluate at the interior sample rows.
fn planted_sphere(chart: Chart) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let evaluator = SphereChartEvaluator;

    let grid_rows = DECODER_LAT * DECODER_LON;
    let mut grid = Array2::<f64>::zeros((grid_rows, D));
    for i in 0..DECODER_LAT {
        let lat_round = LAT_LO + (LAT_HI - LAT_LO) * (i as f64 + 0.5) / DECODER_LAT as f64;
        let lat_chart = inv_warp_lat(chart, lat_round);
        for j in 0..DECODER_LON {
            let row = i * DECODER_LON + j;
            let lon = std::f64::consts::TAU * (j as f64 + 0.5) / DECODER_LON as f64
                - std::f64::consts::PI;
            grid[[row, 0]] = lat_chart;
            grid[[row, 1]] = lon;
        }
    }
    let (grid_phi, _grid_jet) = evaluator.evaluate(grid.view()).expect("grid basis");
    let m = grid_phi.ncols();
    let mut grid_y = Array2::<f64>::zeros((grid_rows, P));
    for row in 0..grid_rows {
        let lat_chart = grid[[row, 0]];
        let lon = grid[[row, 1]];
        let lat_round = warp_lat(chart, lat_chart);
        let pt = sphere_point(lat_round, lon);
        for col in 0..P {
            grid_y[[row, col]] = pt[col];
        }
    }
    let decoder = least_squares_decoder(&grid_phi, &grid_y);

    let coords = chart_coords(chart);
    let mut z = Array2::<f64>::zeros((N, P));
    for row in 0..N {
        let lat_chart = coords[[row, 0]];
        let lon = coords[[row, 1]];
        let lat_round = warp_lat(chart, lat_chart);
        let pt = sphere_point(lat_round, lon);
        for col in 0..P {
            z[[row, col]] = pt[col];
        }
    }
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("row basis");

    let atom = SaeManifoldAtom::new(
        atom_name(chart),
        SaeAtomBasisKind::Sphere,
        D,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("atom")
    .with_basis_evaluator(Arc::new(evaluator));

    let logits = Array2::<f64>::zeros((N, 1));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Product(vec![
            LatentManifold::Interval {
                lo: -std::f64::consts::FRAC_PI_2,
                hi: std::f64::consts::FRAC_PI_2,
            },
            LatentManifold::Circle {
                period: std::f64::consts::TAU,
            },
        ])],
        AssignmentMode::softmax(0.5),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(D); 1]);
    (term, z, rho)
}

/// Reconstruction EV of the atom decode against the planted sphere target.
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

/// The round-sphere isometry defect of an atom's CURRENT fitted chart — the
/// exact objective `canonicalize_charts_post_fit` descends for sphere atoms.
fn sphere_defect(term: &SaeManifoldTerm) -> f64 {
    let atom = &term.atoms[0];
    let evaluator = atom
        .basis_evaluator
        .as_ref()
        .cloned()
        .expect("sphere atom carries its evaluator");
    let coords = term.assignment.coords[0].as_matrix();
    sphere_chart_isometry_defect(
        evaluator.as_ref(),
        atom.decoder_coefficients.view(),
        coords.view(),
    )
    .expect("defect must evaluate")
    .expect("interior sphere band is non-degenerate, defect must be Some")
}

#[test]
fn warped_sphere_chart_has_large_defect_round_chart_near_zero() {
    let (round_term, round_z, _) = planted_sphere(Chart::Round);
    let (warped_term, warped_z, _) = planted_sphere(Chart::Warped);

    // Both plants reconstruct the same physical sphere surface nearly perfectly.
    let round_ev = reconstruction_ev(&round_term, &round_z);
    let warped_ev = reconstruction_ev(&warped_term, &warped_z);
    assert!(
        round_ev > 0.999,
        "round-chart plant must reconstruct the sphere; EV {round_ev}"
    );
    assert!(
        warped_ev > 0.999,
        "warped-chart plant must reconstruct the same sphere image; EV {warped_ev}"
    );

    // The acceptance quantity: round ≈ 0, warped sizeable — the chart dishonesty
    // reconstruction cannot see, exposed by the isometry defect.
    let round_defect = sphere_defect(&round_term);
    let warped_defect = sphere_defect(&warped_term);
    assert!(
        round_defect < 1.0e-3,
        "a round-isometric sphere chart must score ≈0 defect; got {round_defect:.6e}"
    );
    assert!(
        warped_defect > 1.0e-1,
        "an anisotropically warped sphere chart must register a sizeable round-sphere \
         isometry defect; got {warped_defect:.6e}"
    );
}

#[test]
fn warped_sphere_chart_canonicalizes_to_near_isometric() {
    let (mut warped_term, warped_z, warped_rho) = planted_sphere(Chart::Warped);
    let (round_term, _round_z, _) = planted_sphere(Chart::Round);

    let defect_before = sphere_defect(&warped_term);
    let round_optimum = sphere_defect(&round_term);
    let ev_before = reconstruction_ev(&warped_term, &warped_z);

    // The production post-fit pass pins the warped sphere chart to the
    // round-sphere conformal-boost minimum-defect representative.
    warped_term
        .canonicalize_charts_post_fit(warped_z.view(), &warped_rho, None)
        .expect("post-fit canonicalization pass");

    assert!(
        warped_term.atoms[0].chart_canonicalized,
        "the warped sphere atom must be canonicalized (pinned)"
    );

    // Acceptance item 1: the canonical chart recovers near-uniform (isometric)
    // coordinates — the defect collapses to within 10% of the round-chart
    // optimum (the issue's "within 10% of optimum" bar).
    let defect_after = sphere_defect(&warped_term);
    assert!(
        defect_after < 0.5 * defect_before,
        "canonicalization must substantially reduce the isometry defect; \
         {defect_before:.6e} -> {defect_after:.6e}"
    );
    assert!(
        defect_after <= round_optimum + 0.10 * defect_before,
        "the canonical sphere chart's defect must be within 10% of the round optimum; \
         after {defect_after:.6e}, round optimum {round_optimum:.6e}, before {defect_before:.6e}"
    );

    // Image-frozen: reconstruction unchanged (the recomposition gate enforces
    // ≤ 1e-9 relative drift, so EV moves only at round-off).
    let ev_after = reconstruction_ev(&warped_term, &warped_z);
    assert!(
        (ev_before - ev_after).abs() <= 1.0e-6,
        "sphere canonicalization must be image-frozen: EV {ev_before} -> {ev_after}"
    );
}

#[test]
fn certificate_reports_sphere_chart_pinned_by_canonicalization() {
    let (mut term, z, rho) = planted_sphere(Chart::Warped);
    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("post-fit canonicalization pass");
    assert!(
        term.atoms[0].chart_canonicalized,
        "the warped sphere chart must be pinned before the certificate check"
    );

    let report = term
        .fit_diagnostics_report(None, false, None)
        .expect("diagnostics")
        .residual_gauge;
    let chart = report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::ChartReparameterization)
        .expect("certificate must carry the chart-reparameterization record");
    assert!(
        !chart.unpinned,
        "the canonicalized sphere chart freedom must be reported pinned"
    );
    assert_eq!(
        chart.provenance,
        VerdictProvenance::PinnedByCanonicalization,
        "the sphere chart pin's provenance must be the canonicalization"
    );
    assert!(
        chart.description.contains("Isom(S²"),
        "the record must name Isom(S², round) = O(3); got: {}",
        chart.description
    );
    assert!(
        report.summary.contains("sphere chart(s) pinned"),
        "the summary must surface sphere chart canonicalization; got: {}",
        report.summary
    );
}
