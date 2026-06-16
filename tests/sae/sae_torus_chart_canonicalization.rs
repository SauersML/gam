//! #1019 stage 2 — post-fit chart canonicalization for `d = 2` torus atoms.
//!
//! The failure being cured: a planted flat torus can reconstruct perfectly
//! while living in an arbitrary smooth torus chart. The image is right, the
//! chart is dishonest, and gauge-invariant smoothness cannot distinguish two
//! diffeomorphic parameterizations. For `T²`, the post-fit cure picks the
//! minimum-isometry-defect representative inside the band-limited flow family:
//! frequencies `|a|, |b| <= 2`, no constant translations.
//!
//! Contracts asserted against self-constructed truth (#904 reference-as-truth):
//! * (a) a deliberately warped-chart flat torus becomes isometric up to a
//!   global scale after canonicalization;
//! * (b) image invariance — reconstruction EV unchanged within 1e-8;
//! * (c) the residual-gauge certificate reports the chart pinned with
//!   `PinnedByCanonicalization` and names
//!   `Isom(T², flat) = U(1)² ⋊ D₄`;
//! * (d) two different planted gauges canonicalize to the same chart, up to
//!   the residual group `U(1)² ⋊ D₄`.

use faer::Side as FaerSide;
use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::identifiability::sae::{GeneratorFamily, VerdictProvenance};
use gam::terms::latent_coord::LatentManifold;
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
    SaeManifoldRho, SaeManifoldTerm, TorusHarmonicEvaluator,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const GRID_N: usize = 14;
const N: usize = GRID_N * GRID_N;
const P: usize = 6;
const D: usize = 2;
const FRAME_COLS: usize = 4;
const H_TORUS: usize = 9;
const M_TORUS: usize = 19 * 19;
const DECODER_GRID: usize = 40;

#[derive(Clone, Copy)]
enum Gauge {
    A,
    B,
}

impl Gauge {
    fn atom_name(self) -> String {
        match self {
            Gauge::A => "warped-torus-a".to_string(),
            Gauge::B => "warped-torus-b".to_string(),
        }
    }

    fn psi(self, t: [f64; 2]) -> [f64; 2] {
        match self {
            Gauge::A => psi_a(t),
            Gauge::B => psi_b(t),
        }
    }

    fn dpsi(self, t: [f64; 2]) -> [[f64; 2]; 2] {
        match self {
            Gauge::A => dpsi_a(t),
            Gauge::B => dpsi_b(t),
        }
    }
}

/// Deterministic orthonormal `P x 4` frame for the planted flat-torus
/// embedding.
fn planted_frame() -> Array2<f64> {
    let mut raw = Array2::<f64>::zeros((P, FRAME_COLS));
    for j in 0..FRAME_COLS {
        for i in 0..P {
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((P, FRAME_COLS));
    for j in 0..FRAME_COLS {
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

fn torus_point(s: [f64; 2], frame: &Array2<f64>) -> Vec<f64> {
    let a0 = std::f64::consts::TAU * s[0];
    let a1 = std::f64::consts::TAU * s[1];
    let raw = [a0.cos(), a0.sin(), a1.cos(), a1.sin()];
    (0..P)
        .map(|i| raw.iter().enumerate().map(|(j, v)| frame[[i, j]] * v).sum())
        .collect()
}

fn psi_a(t: [f64; 2]) -> [f64; 2] {
    let tau = std::f64::consts::TAU;
    [
        t[0] + 0.020 * (tau * t[0]).sin()
            + 0.014 * (tau * t[1]).cos()
            + 0.012 * (tau * (t[0] + t[1])).sin(),
        t[1] - 0.016 * (tau * t[0]).cos()
            + 0.018 * (tau * t[1]).sin()
            + 0.010 * (tau * (t[0] - t[1])).cos(),
    ]
}

fn dpsi_a(t: [f64; 2]) -> [[f64; 2]; 2] {
    let tau = std::f64::consts::TAU;
    [
        [
            1.0 + 0.020 * tau * (tau * t[0]).cos() + 0.012 * tau * (tau * (t[0] + t[1])).cos(),
            -0.014 * tau * (tau * t[1]).sin() + 0.012 * tau * (tau * (t[0] + t[1])).cos(),
        ],
        [
            0.016 * tau * (tau * t[0]).sin() - 0.010 * tau * (tau * (t[0] - t[1])).sin(),
            1.0 + 0.018 * tau * (tau * t[1]).cos() + 0.010 * tau * (tau * (t[0] - t[1])).sin(),
        ],
    ]
}

fn psi_b(t: [f64; 2]) -> [f64; 2] {
    let tau = std::f64::consts::TAU;
    [
        t[0] - 0.018 * (tau * (t[0] - t[1])).cos()
            + 0.015 * (tau * t[1]).sin()
            + 0.011 * (tau * t[0]).cos(),
        t[1] + 0.019 * (tau * (t[0] + t[1])).sin() - 0.013 * (tau * t[1]).cos()
            + 0.012 * (tau * t[0]).sin(),
    ]
}

fn dpsi_b(t: [f64; 2]) -> [[f64; 2]; 2] {
    let tau = std::f64::consts::TAU;
    [
        [
            1.0 + 0.018 * tau * (tau * (t[0] - t[1])).sin() - 0.011 * tau * (tau * t[0]).sin(),
            -0.018 * tau * (tau * (t[0] - t[1])).sin() + 0.015 * tau * (tau * t[1]).cos(),
        ],
        [
            0.019 * tau * (tau * (t[0] + t[1])).cos() + 0.012 * tau * (tau * t[0]).cos(),
            1.0 + 0.019 * tau * (tau * (t[0] + t[1])).cos() + 0.013 * tau * (tau * t[1]).sin(),
        ],
    ]
}

fn wrapped_delta(x: f64) -> f64 {
    x - x.round()
}

fn invert_gauge(gauge: Gauge, s: [f64; 2]) -> [f64; 2] {
    let mut t = s;
    let mut iteration = 0usize;
    while iteration < 30 {
        let u = gauge.psi(t);
        let r = [wrapped_delta(u[0] - s[0]), wrapped_delta(u[1] - s[1])];
        let jac = gauge.dpsi(t);
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        let step = [
            (jac[1][1] * r[0] - jac[0][1] * r[1]) / det,
            (-jac[1][0] * r[0] + jac[0][0] * r[1]) / det,
        ];
        t[0] -= step[0];
        t[1] -= step[1];
        iteration += 1;
    }
    let u = gauge.psi(t);
    let r = [wrapped_delta(u[0] - s[0]), wrapped_delta(u[1] - s[1])];
    let residual_norm = (r[0] * r[0] + r[1] * r[1]).sqrt();
    assert!(
        residual_norm < 1.0e-12,
        "inverse chart Newton residual too large: {residual_norm}"
    );
    [t[0].rem_euclid(1.0), t[1].rem_euclid(1.0)]
}

fn honest_sample() -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((N, D));
    for i in 0..GRID_N {
        for j in 0..GRID_N {
            let row = i * GRID_N + j;
            coords[[row, 0]] = (i as f64 + 0.5) / GRID_N as f64;
            coords[[row, 1]] = (j as f64 + 0.5) / GRID_N as f64;
        }
    }
    coords
}

/// Exact least squares `B = argmin ||Phi B - Y||^2` via normal equations.
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

fn planted_warped_torus(
    gauge: Gauge,
    honest: &Array2<f64>,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let frame = planted_frame();
    let evaluator = TorusHarmonicEvaluator::new(D, H_TORUS).expect("torus evaluator");
    let m = evaluator.basis_size();
    assert_eq!(m, M_TORUS);

    let grid_rows = DECODER_GRID * DECODER_GRID;
    let mut grid = Array2::<f64>::zeros((grid_rows, D));
    for i in 0..DECODER_GRID {
        for j in 0..DECODER_GRID {
            let row = i * DECODER_GRID + j;
            grid[[row, 0]] = i as f64 / DECODER_GRID as f64;
            grid[[row, 1]] = j as f64 / DECODER_GRID as f64;
        }
    }
    let (grid_phi, grid_jet) = evaluator.evaluate(grid.view()).expect("grid basis");
    assert_eq!(grid_jet.dim(), (grid_rows, m, D));
    let mut grid_y = Array2::<f64>::zeros((grid_rows, P));
    for row in 0..grid_rows {
        let t = [grid[[row, 0]], grid[[row, 1]]];
        let point = torus_point(gauge.psi(t), &frame);
        for col in 0..P {
            grid_y[[row, col]] = point[col];
        }
    }
    let decoder = least_squares_decoder(&grid_phi, &grid_y);

    let mut coords = Array2::<f64>::zeros((N, D));
    let mut z = Array2::<f64>::zeros((N, P));
    for row in 0..N {
        let s = [honest[[row, 0]], honest[[row, 1]]];
        let t = invert_gauge(gauge, s);
        coords[[row, 0]] = t[0];
        coords[[row, 1]] = t[1];
        let point = torus_point(s, &frame);
        for col in 0..P {
            z[[row, col]] = point[col];
        }
    }
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("row basis");

    let atom = SaeManifoldAtom::new(
        gauge.atom_name(),
        SaeAtomBasisKind::Torus,
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
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ])],
        AssignmentMode::ibp_map(0.5, 1.0, false),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); 1]);
    (term, z, rho)
}

fn metric_summaries(term: &SaeManifoldTerm) -> Vec<(f64, f64)> {
    let atom = &term.atoms[0];
    let mut out = Vec::with_capacity(term.n_obs());
    for row in 0..term.n_obs() {
        let v0 = atom.decoded_derivative_row(row, 0);
        let v1 = atom.decoded_derivative_row(row, 1);
        let g00: f64 = v0.iter().map(|v| v * v).sum();
        let g11: f64 = v1.iter().map(|v| v * v).sum();
        let g01: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        let trace = g00 + g11;
        let disc = ((g00 - g11) * (g00 - g11) + 4.0 * g01 * g01).sqrt();
        let lambda_max = 0.5 * (trace + disc);
        let lambda_min = 0.5 * (trace - disc);
        let det = g00 * g11 - g01 * g01;
        out.push((lambda_max / lambda_min, det.sqrt()));
    }
    out
}

fn metric_extrema(term: &SaeManifoldTerm) -> (f64, f64, f64) {
    let summaries = metric_summaries(term);
    let mut max_ratio = 0.0_f64;
    let mut min_area = f64::INFINITY;
    let mut max_area = 0.0_f64;
    for (ratio, area) in summaries {
        max_ratio = max_ratio.max(ratio);
        min_area = min_area.min(area);
        max_area = max_area.max(area);
    }
    (max_ratio, min_area, max_area)
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

fn transformed_point(kind: usize, a: [f64; 2]) -> [f64; 2] {
    match kind {
        0 => [a[0], a[1]],
        1 => [a[1], a[0]],
        2 => [-a[0], a[1]],
        3 => [a[0], -a[1]],
        4 => [-a[0], -a[1]],
        5 => [-a[1], a[0]],
        6 => [a[1], -a[0]],
        7 => [-a[1], -a[0]],
        invalid => panic!("invalid D4 transform index {invalid}"),
    }
}

fn torus_alignment_distance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a.ncols(), D);
    let tau_const = std::f64::consts::TAU;
    let mut best = f64::INFINITY;
    for transform in 0..8 {
        let mut tau = [0.0_f64; D];
        for axis in 0..D {
            let mut sin_sum = 0.0_f64;
            let mut cos_sum = 0.0_f64;
            for row in 0..a.nrows() {
                let ra = transformed_point(transform, [a[[row, 0]], a[[row, 1]]]);
                let v = b[[row, axis]] - ra[axis];
                sin_sum += (tau_const * v).sin();
                cos_sum += (tau_const * v).cos();
            }
            tau[axis] = sin_sum.atan2(cos_sum) / tau_const;
        }
        let mut max_residual = 0.0_f64;
        for row in 0..a.nrows() {
            let ra = transformed_point(transform, [a[[row, 0]], a[[row, 1]]]);
            let mut row_residual = 0.0_f64;
            for axis in 0..D {
                let v = b[[row, axis]] - ra[axis];
                row_residual = row_residual.max(wrapped_delta(v - tau[axis]).abs());
            }
            max_residual = max_residual.max(row_residual);
        }
        best = best.min(max_residual);
    }
    best
}

fn canonical_coords(term: &SaeManifoldTerm) -> Array2<f64> {
    term.assignment.coords[0].as_matrix().to_owned()
}

#[test]
fn warped_torus_chart_canonicalizes_to_uniform_metric() {
    let honest = honest_sample();
    let (mut term, z, rho) = planted_warped_torus(Gauge::A, &honest);

    let (max_ratio_before, min_area_before, max_area_before) = metric_extrema(&term);
    assert!(
        max_ratio_before > 1.5 || max_area_before / min_area_before > 1.5,
        "planted torus chart must be dishonest; metric ratio {max_ratio_before}, area ratio {}",
        max_area_before / min_area_before
    );

    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(
        term.atoms[0].chart_canonicalized,
        "the planted torus atom must be canonicalized"
    );

    let (max_ratio_after, min_area_after, max_area_after) = metric_extrema(&term);
    assert!(
        max_ratio_after < 1.05,
        "canonical chart metric eigenvalue ratio must be < 1.05; got {max_ratio_after}"
    );
    assert!(
        max_area_after / min_area_after < 1.05,
        "canonical chart area density ratio must be < 1.05; got {}",
        max_area_after / min_area_after
    );
}

#[test]
fn torus_canonicalization_freezes_the_image() {
    let honest = honest_sample();
    let (mut term, z, rho) = planted_warped_torus(Gauge::A, &honest);

    let ev_before = reconstruction_ev(&term, &z);
    assert!(
        ev_before > 0.999,
        "the plant must reconstruct the torus nearly perfectly; got EV {ev_before}"
    );

    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(term.atoms[0].chart_canonicalized);

    let ev_after = reconstruction_ev(&term, &z);
    assert!(
        (ev_before - ev_after).abs() <= 1.0e-8,
        "canonicalization must be image-frozen: EV {ev_before} -> {ev_after}"
    );
}

#[test]
fn certificate_reports_torus_chart_pinned_by_canonicalization() {
    let honest = honest_sample();
    let (mut term, z, rho) = planted_warped_torus(Gauge::A, &honest);
    term.canonicalize_charts_post_fit(z.view(), &rho, None)
        .expect("canonicalization pass");
    assert!(term.atoms[0].chart_canonicalized);

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
        "the canonicalized torus chart freedom must be reported pinned"
    );
    assert_eq!(
        chart.provenance,
        VerdictProvenance::PinnedByCanonicalization,
        "the chart pin's provenance must be the canonicalization"
    );
    assert!(
        chart.description.contains("Isom(T²"),
        "the record must name Isom(T², flat); got: {}",
        chart.description
    );
    assert!(
        report.summary.contains("torus chart(s) pinned"),
        "the summary must surface torus chart canonicalization; got: {}",
        report.summary
    );
}

#[test]
fn two_random_gauges_canonicalize_to_the_same_chart() {
    let honest = honest_sample();
    let (mut term_a, z_a, rho_a) = planted_warped_torus(Gauge::A, &honest);
    let (mut term_b, z_b, rho_b) = planted_warped_torus(Gauge::B, &honest);

    let raw_a = canonical_coords(&term_a);
    let raw_b = canonical_coords(&term_b);
    let raw_distance = torus_alignment_distance(&raw_a, &raw_b);
    assert!(
        raw_distance > 0.01,
        "raw planted gauges must genuinely disagree; alignment distance {raw_distance}"
    );

    term_a
        .canonicalize_charts_post_fit(z_a.view(), &rho_a, None)
        .expect("canonicalization pass A");
    term_b
        .canonicalize_charts_post_fit(z_b.view(), &rho_b, None)
        .expect("canonicalization pass B");
    assert!(term_a.atoms[0].chart_canonicalized);
    assert!(term_b.atoms[0].chart_canonicalized);

    let canon_a = canonical_coords(&term_a);
    let canon_b = canonical_coords(&term_b);
    let canon_distance = torus_alignment_distance(&canon_a, &canon_b);
    assert!(
        canon_distance < 2.0e-3,
        "canonical gauges must agree up to Isom(T², flat); alignment distance {canon_distance}"
    );

    let truth_distance_a = torus_alignment_distance(&canon_a, &honest);
    let truth_distance_b = torus_alignment_distance(&canon_b, &honest);
    assert!(
        truth_distance_a < 2.0e-3,
        "gauge A canonical chart must recover the honest chart; alignment distance {truth_distance_a}"
    );
    assert!(
        truth_distance_b < 2.0e-3,
        "gauge B canonical chart must recover the honest chart; alignment distance {truth_distance_b}"
    );
}
