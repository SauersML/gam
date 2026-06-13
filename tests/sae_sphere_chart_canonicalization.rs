//! #1019 sphere arm — post-fit chart honesty for `d = 2` sphere (S²) atoms,
//! at the term-level acceptance boundary.
//!
//! The failure being exposed: a planted sphere image can reconstruct perfectly
//! while living in an anisotropically warped `(lat, lon)` chart. The image is
//! right, the chart is dishonest, and gauge-invariant smoothness cannot tell
//! two diffeomorphic parameterizations apart.
//!
//! Unlike the torus and free-patch arms, the sphere has NO global pole-free
//! flow basis (hairy ball), so the issue's flow-PIN does not transfer and the
//! sphere chart is honestly **left as fitted**. What IS well-defined with no
//! flow basis at all is the issue's acceptance QUANTITY — the round-sphere
//! isometry defect `E = Σ_i ‖Ĝ_i − c·ĝ_ref,i‖²` (scale `c` analytically
//! profiled) — which a round-isometric chart drives to ≈0 and a warped chart
//! drives large. These contracts assert that quantity flows through the
//! production `canonicalize_charts_post_fit` pass (not just the module unit
//! tests) AND that the certificate stays HONEST about the sphere: it must NOT
//! claim `PinnedByCanonicalization` for a chart it never pinned.
//!
//! Contracts (against self-constructed truth, #904 reference-as-truth):
//! * (a) a deliberately warped-chart sphere has a sizeable round-sphere
//!   isometry defect, and a round-isometric chart has ≈0 defect — the issue's
//!   acceptance quantity, computed by the same code the post-fit pass runs;
//! * (b) the post-fit canonicalization pass leaves the sphere atom UNCHANGED
//!   (image-frozen and chart-unchanged — no false pin);
//! * (c) the residual-gauge certificate does NOT carry a
//!   `PinnedByCanonicalization` chart record for the sphere atom (honest
//!   "left as fitted"); the sphere atom's `chart_canonicalized` flag stays
//!   false.

use faer::Side as FaerSide;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::sae_identifiability::VerdictProvenance;
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

#[derive(Clone, Copy)]
enum Chart {
    /// Round-isometric `(lat, lon)` chart: pullback metric is exactly
    /// `diag(1, cos²lat)` up to a global scale, so the round-sphere isometry
    /// defect is ≈0.
    Round,
    /// Anisotropically warped chart: the lat coordinate is squeezed toward the
    /// equator by a smooth monotone reparameterization (a zonal warp), so the
    /// pullback metric is NOT a global rescale of `diag(1, cos²lat)` and the
    /// defect is strictly positive. The image is identical to the round chart's
    /// (same sphere surface), only the parameterization differs.
    Warped,
}

/// The latitude reparameterization defining each chart. `Round` is the
/// identity; `Warped` applies a smooth, strictly-monotone zonal squeeze
/// `lat ↦ lat + a·sin(2·lat)` (`a < 1/2` keeps it a diffeomorphism on
/// `[-π/2, π/2]`, fixing both poles and the equator).
fn warp_lat(chart: Chart, lat: f64) -> f64 {
    match chart {
        Chart::Round => lat,
        Chart::Warped => lat + 0.35 * (2.0 * lat).sin(),
    }
}

/// Inverse of [`warp_lat`] by monotone Newton iteration — recovers the chart
/// coordinate `lat_chart` whose round latitude is `lat_round`.
fn inv_warp_lat(chart: Chart, lat_round: f64) -> f64 {
    match chart {
        Chart::Round => lat_round,
        Chart::Warped => {
            let mut x = lat_round;
            for _ in 0..40 {
                let f = x + 0.35 * (2.0 * x).sin() - lat_round;
                let df = 1.0 + 0.70 * (2.0 * x).cos();
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

/// Deterministic interior `(lat, lon)` sampling grid in CHART coordinates,
/// kept clear of the poles so the round-sphere reference metric `cos²lat` is
/// well away from its singularity.
fn chart_coords(chart: Chart) -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((N, D));
    let lat_lo = -1.2_f64;
    let lat_hi = 1.2_f64;
    for i in 0..LAT_GRID {
        // Sample the ROUND latitude on a uniform interior band, then map to the
        // chart latitude so every chart sees the same physical sphere band.
        let lat_round =
            lat_lo + (lat_hi - lat_lo) * (i as f64 + 0.5) / LAT_GRID as f64;
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

    // Dense decoder grid in CHART coordinates spanning the same physical band.
    let grid_rows = DECODER_LAT * DECODER_LON;
    let mut grid = Array2::<f64>::zeros((grid_rows, D));
    let lat_lo = -1.2_f64;
    let lat_hi = 1.2_f64;
    for i in 0..DECODER_LAT {
        let lat_round = lat_lo + (lat_hi - lat_lo) * (i as f64 + 0.5) / DECODER_LAT as f64;
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
    // Two latent axes ⇒ one log-precision per axis in the ARD block.
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
/// exact quantity `canonicalize_charts_post_fit` measures and logs for sphere
/// atoms, called directly here so the contract asserts the production code path.
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
    assert!(
        warped_defect > 100.0 * round_defect.max(1.0e-12),
        "the warped chart's defect must dominate the round chart's; \
         warped {warped_defect:.6e} vs round {round_defect:.6e}"
    );
}

#[test]
fn sphere_post_fit_pass_leaves_chart_unchanged_and_image_frozen() {
    let (mut term, z, rho) = planted_sphere(Chart::Warped);

    let ev_before = reconstruction_ev(&term, &z);
    let defect_before = sphere_defect(&term);
    let coords_before = term.assignment.coords[0].as_matrix().to_owned();

    // The production post-fit pass: it MEASURES the sphere defect (read-only) but
    // must NOT pin/mutate the sphere chart (no global pole-free flow basis).
    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("post-fit canonicalization pass");

    assert!(
        !term.atoms[0].chart_canonicalized,
        "the sphere atom must NOT be flagged canonicalized — it is honestly left as fitted"
    );

    let ev_after = reconstruction_ev(&term, &z);
    let defect_after = sphere_defect(&term);
    let coords_after = term.assignment.coords[0].as_matrix().to_owned();

    assert!(
        (ev_before - ev_after).abs() <= 1.0e-12,
        "the sphere pass must be image-frozen: EV {ev_before} -> {ev_after}"
    );
    assert!(
        (defect_before - defect_after).abs() <= 1.0e-12,
        "the sphere chart must be untouched: defect {defect_before:.6e} -> {defect_after:.6e}"
    );
    let mut max_coord_drift = 0.0_f64;
    for (a, b) in coords_before.iter().zip(coords_after.iter()) {
        max_coord_drift = max_coord_drift.max((a - b).abs());
    }
    assert!(
        max_coord_drift <= 1.0e-12,
        "the sphere chart coordinates must be untouched; max drift {max_coord_drift:.3e}"
    );
}

#[test]
fn certificate_does_not_falsely_pin_the_sphere_chart() {
    let (mut term, z, rho) = planted_sphere(Chart::Warped);
    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("post-fit canonicalization pass");

    let report = term
        .fit_diagnostics_report(None, false, None)
        .expect("diagnostics")
        .residual_gauge;

    // HONESTY contract: no chart generator may claim PinnedByCanonicalization
    // for the sphere atom — the sphere arm is measurement-only, never pinned.
    let falsely_pinned = report.generators.iter().any(|g| {
        g.provenance == VerdictProvenance::PinnedByCanonicalization
            && g.description.contains(atom_name(Chart::Warped))
    });
    assert!(
        !falsely_pinned,
        "the sphere chart must NOT be reported PinnedByCanonicalization (it is left as \
         fitted — no global pole-free flow basis to pin); generators: {:#?}",
        report
            .generators
            .iter()
            .map(|g| format!("{:?}: {}", g.provenance, g.description))
            .collect::<Vec<_>>()
    );
}
