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
use gam::identifiability::sae::{GeneratorFamily, VerdictProvenance};
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::terms::latent::LatentManifold;
use gam::terms::sae::chart_canonicalization::sphere_chart_isometry_defect;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

/// Maximum spherical-harmonic degree of the test sphere basis. `L = 8` gives
/// `(L+1)² = 81` real harmonics — rich enough that the conformal-boost warped
/// image stays inside the span to the post-fit recomposition floor (`< 1e-9`
/// relative on the audit grid for `WARP_A ≤ 0.05`), so the sphere flow-pin can
/// actually COMMIT the canonical chart rather than honestly refusing for want of
/// a basis that can absorb the reparameterized image. The production
/// 7-column embedding stub `SphereChartEvaluator` is a jet-parity fixture, not a
/// fitting basis: it cannot represent a boosted sphere image (the conformal
/// factor escapes any fixed finite basis, and at 7 columns the residual is ~6e-3
/// — far above the 1e-9 freeze gate), so the pin would refuse on it.
const SPHERE_L: usize = 8;

/// Number of real spherical harmonics of degree `≤ L`: `(L+1)²`.
fn sphere_basis_size(l_max: usize) -> usize {
    (l_max + 1) * (l_max + 1)
}

/// Real spherical-harmonic sphere-chart basis up to degree `L`, evaluated on
/// `(lat, lon)` coordinates with its analytic first jet. This is the rich
/// sibling of the production 7-column [`gam::terms::SphereChartEvaluator`] stub:
/// `(L+1)²` real harmonics span enough of the boosted-image function space that
/// the conformal-boost canonicalization is exactly image-frozen (the production
/// stub is a jet-parity fixture and cannot absorb a warped image). The basis is
/// the column family `Y_l^m(lat, lon)` with the geodesy `4π`-normalization (the
/// normalization is irrelevant to the column SPAN, which is all the least-squares
/// decoder fit and the chart transport consume).
#[derive(Debug, Clone, Copy)]
struct RichSphereHarmonicEvaluator {
    l_max: usize,
}

impl RichSphereHarmonicEvaluator {
    fn new(l_max: usize) -> Self {
        Self { l_max }
    }
}

/// Associated Legendre `P_l^m(x)` for all `0 ≤ m ≤ l ≤ L` at a scalar `x`,
/// plus the `x`-derivative `dP_l^m/dx`, via the standard recurrences. Returned
/// as `(p, dp)` indexed `[l][m]`. The data band keeps `|x| = |sin lat| < 1`, so
/// the `(1 − x²)` derivative denominator is safely positive.
fn assoc_legendre_with_deriv(l_max: usize, x: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut p = vec![vec![0.0_f64; l_max + 1]; l_max + 1];
    let one_minus_x2 = (1.0 - x * x).max(f64::MIN_POSITIVE);
    let somx2 = one_minus_x2.sqrt();
    p[0][0] = 1.0;
    if l_max >= 1 {
        p[1][0] = x;
        p[1][1] = -somx2; // Condon–Shortley phase
    }
    for l in 2..=l_max {
        for m in 0..=l {
            p[l][m] = if m == l {
                -((2 * l - 1) as f64) * somx2 * p[l - 1][l - 1]
            } else if m == l - 1 {
                x * ((2 * l - 1) as f64) * p[l - 1][l - 1]
            } else {
                (((2 * l - 1) as f64) * x * p[l - 1][m] - ((l - 1 + m) as f64) * p[l - 2][m])
                    / ((l - m) as f64)
            };
        }
    }
    // dP_l^m/dx = [l·x·P_l^m − (l+m)·P_{l-1}^m] / (x² − 1), off the poles.
    let mut dp = vec![vec![0.0_f64; l_max + 1]; l_max + 1];
    for l in 0..=l_max {
        for m in 0..=l {
            let prev = if l >= 1 && m <= l - 1 {
                p[l - 1][m]
            } else {
                0.0
            };
            dp[l][m] = ((l as f64) * x * p[l][m] - ((l + m) as f64) * prev) / (x * x - 1.0);
        }
    }
    (p, dp)
}

impl SaeBasisEvaluator for RichSphereHarmonicEvaluator {
    fn evaluate(
        &self,
        coords: ndarray::ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array3<f64>), String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "RichSphereHarmonicEvaluator: expected latent_dim == 2, got {}",
                coords.ncols()
            ));
        }
        let l_max = self.l_max;
        let m_cols = sphere_basis_size(l_max);
        let n = coords.nrows();
        let mut phi = Array2::<f64>::zeros((n, m_cols));
        let mut jet = Array3::<f64>::zeros((n, m_cols, 2));
        for row in 0..n {
            let lat = coords[[row, 0]];
            let lon = coords[[row, 1]];
            let clat = lat.cos();
            let x = lat.sin(); // cos(colatitude)
            let (p, dp) = assoc_legendre_with_deriv(l_max, x);
            let mut col = 0usize;
            for l in 0..=l_max {
                for m in -(l as i64)..=(l as i64) {
                    let am = m.unsigned_abs() as usize;
                    let norm = {
                        // 4π geodesy normalization N_lm = √((2l+1)/(4π) · (l−|m|)!/(l+|m|)!).
                        let mut ratio = 1.0_f64;
                        for k in (l - am + 1)..=(l + am) {
                            ratio *= k as f64;
                        }
                        ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) / ratio).sqrt()
                    };
                    let leg = norm * p[l][am];
                    // dP/dlat = dP/dx · dx/dlat = dP/dx · cos(lat).
                    let dleg_dlat = norm * dp[l][am] * clat;
                    let (ang, dang_dlon) = match m.cmp(&0) {
                        std::cmp::Ordering::Greater => {
                            let a = am as f64;
                            (
                                std::f64::consts::SQRT_2 * (a * lon).cos(),
                                std::f64::consts::SQRT_2 * (-a) * (a * lon).sin(),
                            )
                        }
                        std::cmp::Ordering::Less => {
                            let a = am as f64;
                            (
                                std::f64::consts::SQRT_2 * (a * lon).sin(),
                                std::f64::consts::SQRT_2 * a * (a * lon).cos(),
                            )
                        }
                        std::cmp::Ordering::Equal => (1.0, 0.0),
                    };
                    phi[[row, col]] = leg * ang;
                    jet[[row, col, 0]] = dleg_dlat * ang;
                    jet[[row, col, 1]] = leg * dang_dlon;
                    col += 1;
                }
            }
        }
        Ok((phi, jet))
    }

    fn second_jet_dyn(
        &self,
        coords: ndarray::ArrayView2<'_, f64>,
    ) -> Option<Result<ndarray::Array4<f64>, String>> {
        // The post-fit sphere canonicalization consumes only `evaluate` (basis +
        // first jet); the exact isometry-penalty Hessian (which would need the
        // analytic second jet) is not exercised by these chart-pin tests. Surface
        // an explicit "not provided" rather than a fabricated zero curvature, but
        // still validate the coordinate width so a malformed call fails loudly.
        if coords.ncols() != 2 {
            return Some(Err(format!(
                "RichSphereHarmonicEvaluator::second_jet_dyn: expected latent_dim == 2, got {}",
                coords.ncols()
            )));
        }
        None
    }

    fn third_jet_dyn(
        &self,
        coords: ndarray::ArrayView2<'_, f64>,
    ) -> Option<Result<ndarray::Array5<f64>, String>> {
        if coords.ncols() != 2 {
            return Some(Err(format!(
                "RichSphereHarmonicEvaluator::third_jet_dyn: expected latent_dim == 2, got {}",
                coords.ncols()
            )));
        }
        None
    }
}

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
// Zonal-boost amplitude of the warped plant. On `|lat| ≤ 1.05` the map
// derivative `1 − a·sin(lat)` stays strictly positive, so the warp is a
// diffeomorphism; and being exactly the `K_z = cos(lat) ∂_lat` Euler map, it is
// the flow the canonicalizer descends, so the pin provably recovers the round
// chart up to O(3). `a = 0.05` already registers a sizeable round-sphere
// isometry defect (≈0.82, far above the 0.1 dishonesty bar) while keeping the
// boosted image inside the degree-`SPHERE_L` harmonic span to the post-fit
// recomposition floor (the LS decoder transport reproduces the image to ~1e-15
// on the audit grid), so the pin COMMITS instead of refusing. Larger amplitudes
// push more conformal-factor energy out of any fixed finite basis; `0.05` is the
// honest operating point where the d=2 sphere flow-pin is exactly image-frozen.
const WARP_A: f64 = 0.05;

#[derive(Clone, Copy)]
enum Chart {
    /// Round-isometric `(lat, lon)` chart: pullback metric is exactly
    /// `diag(1, cos²lat)` up to a global scale, so the defect is ≈0.
    Round,
    /// Conformal-boost warped chart: latitude pushed by the zonal-boost Euler
    /// map `lat ↦ lat + a·cos(lat)`. Same physical sphere surface, warped
    /// parameterization — reconstruction cannot see it, the isometry defect can.
    Warped,
}

/// The latitude reparameterization defining each chart. `Round` is the
/// identity; `Warped` is the zonal-boost Euler map `lat ↦ lat + a·cos(lat)`.
fn warp_lat(chart: Chart, lat: f64) -> f64 {
    match chart {
        Chart::Round => lat,
        Chart::Warped => lat + WARP_A * lat.cos(),
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
                // d/dx [x + a cos x] = 1 − a sin x
                let f = x + WARP_A * x.cos() - lat_round;
                let df = 1.0 - WARP_A * x.sin();
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
            let lon =
                std::f64::consts::TAU * (j as f64 + 0.5) / LON_GRID as f64 - std::f64::consts::PI;
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
/// image through the rich degree-`SPHERE_L` [`RichSphereHarmonicEvaluator`]
/// basis, fitted by exact LS on a dense grid, then evaluate at the interior
/// sample rows.
fn planted_sphere(chart: Chart) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let evaluator = RichSphereHarmonicEvaluator::new(SPHERE_L);

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
