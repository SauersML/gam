//! Spectral decode of the learned graph atom: eigengap basis, Nyström
//! out-of-sample coordinate + analytic jet, penalty/Dirichlet-form identity, and
//! the reconstruction advantage of a `q`-dimensional spectral decode over a
//! single typed circle atom on a shape outside the typed zoo.

use crate::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator, SaeBasisSecondJet};
use crate::manifold::{GraphEdge, LearnedGraphAtom, graph_edge_rank_charge};
use crate::saebench_metrics::{ChartInterpObservation, chart_interp_score};
use gam_solve::gaussian_reml::gaussian_reml_multi_closed_form;
use ndarray::{Array2, Array3};

use std::f64::consts::TAU;

/// A clean `anchors`-vertex circle graph: unit-circle anchor embeddings whose
/// derived 2-NN candidate set is exactly the cycle, all edges kept.
fn circle_atom(anchors: usize) -> LearnedGraphAtom {
    let embeddings = Array2::<f64>::from_shape_fn((anchors, 2), |(i, j)| {
        let phase = TAU * i as f64 / anchors as f64;
        if j == 0 { phase.cos() } else { phase.sin() }
    });
    let rows: Vec<f64> = (0..anchors * 4)
        .map(|i| i as f64 / (anchors * 4) as f64)
        .collect();
    let edges = LearnedGraphAtom::knn_candidate_edges(embeddings.view()).expect("knn edges");
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let precisions = vec![1.0; edges.len()];
    let deltas = vec![charge * 2.0; edges.len()];
    LearnedGraphAtom::from_reml_candidate_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &edges,
        &precisions,
        &deltas,
    )
    .expect("circle graph atom")
}

/// A figure-eight (wedge of two tangent circles sharing vertex 0): `b0 = 1`,
/// `b1 = 2` — a shape absent from the typed zoo (circle / interval / torus /
/// sphere / cylinder / finite-set). Returns the atom, the vertex positions, and
/// a single-loop 1-D coordinate for the circle baseline.
fn figure_eight(m: usize) -> (LearnedGraphAtom, Array2<f64>, Vec<f64>) {
    let num_vertices = 2 * m - 1;
    let mut embeddings = Array2::<f64>::zeros((num_vertices, 2));
    // Lobe A: circle centered (-1, 0), vertex 0 at the tangent origin (0, 0).
    for j in 0..m {
        let angle = TAU * j as f64 / m as f64;
        embeddings[[j, 0]] = -1.0 + angle.cos();
        embeddings[[j, 1]] = angle.sin();
    }
    // Lobe B: circle centered (1, 0), tangent origin at angle π (shared vertex 0).
    for j in 1..m {
        let idx = m - 1 + j;
        let angle = std::f64::consts::PI + TAU * j as f64 / m as f64;
        embeddings[[idx, 0]] = 1.0 + angle.cos();
        embeddings[[idx, 1]] = angle.sin();
    }

    let mut edges: Vec<GraphEdge> = Vec::new();
    // Lobe A cycle 0-1-…-(m-1)-0.
    for j in 0..m {
        edges.push(GraphEdge::new(j, (j + 1) % m).expect("lobe A edge"));
    }
    // Lobe B cycle 0-m-…-(2m-2)-0.
    let lobe_b: Vec<usize> = std::iter::once(0).chain(m..=(2 * m - 2)).collect();
    for w in 0..lobe_b.len() {
        let a = lobe_b[w];
        let b = lobe_b[(w + 1) % lobe_b.len()];
        edges.push(GraphEdge::new(a, b).expect("lobe B edge"));
    }

    let rows: Vec<f64> = (0..num_vertices)
        .map(|i| i as f64 / num_vertices as f64)
        .collect();
    let n_eff = (num_vertices * 4) as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let precisions = vec![1.0; edges.len()];
    let deltas = vec![charge * 2.0; edges.len()];
    let atom = LearnedGraphAtom::from_reml_candidate_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &edges,
        &precisions,
        &deltas,
    )
    .expect("figure-eight graph atom");

    let circle_coord: Vec<f64> = (0..num_vertices)
        .map(|i| i as f64 / num_vertices as f64)
        .collect();
    (atom, embeddings, circle_coord)
}

/// Explained variance of a multi-output reconstruction `Ŷ` against `Y`.
fn explained_variance(y: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let n = y.nrows();
    let p = y.ncols();
    let mut mean = vec![0.0; p];
    for row in 0..n {
        for c in 0..p {
            mean[c] += y[[row, c]];
        }
    }
    for c in 0..p {
        mean[c] /= n as f64;
    }
    let mut sse = 0.0;
    let mut sst = 0.0;
    for row in 0..n {
        for c in 0..p {
            sse += (y[[row, c]] - fitted[[row, c]]).powi(2);
            sst += (y[[row, c]] - mean[c]).powi(2);
        }
    }
    1.0 - sse / sst
}

// (d) — eigengap q-selection recovers q = 2 for the circle, and the embedding
// dimension is deliberately NOT tied to the first Betti number.
#[test]
fn eigengap_selects_two_for_circle_and_decouples_from_betti() {
    let atom = circle_atom(24);
    let basis = atom.spectral_decode_basis().expect("spectral basis");
    // A circle needs a 2-D embedding (cos, sin) even though b1 = 1: the first
    // non-trivial Laplacian eigenvalue is doubly degenerate, so the eigengap
    // falls after the pair.
    assert_eq!(basis.selected_q(), 2, "circle decode dimension");
    assert_eq!(atom.topology_readout().b1, 1, "circle first Betti number");

    // Figure-eight (b1 = 2): the eigengap picks an embedding dimension >= 2. We
    // do NOT assert q == b1: embedding dimension and cycle rank are different
    // invariants (the circle already shows q = 2 > b1 = 1). The principled
    // statement is that a wedge of two circles is not 1-D-embeddable, so the
    // decode keeps at least two modes.
    let (fig, _pos, _coord) = figure_eight(16);
    assert_eq!(
        fig.topology_readout().b1,
        2,
        "figure-eight first Betti number"
    );
    let fig_basis = fig
        .spectral_decode_basis()
        .expect("figure-eight spectral basis");
    assert!(
        fig_basis.selected_q() >= 2,
        "figure-eight decode dimension {} should be >= 2",
        fig_basis.selected_q()
    );
}

// (1) — the decode penalty and the graph Dirichlet form are the SAME object:
// diag(λ) == Φᵀ L_W Φ, where L_W is the atom's own surviving Laplacian.
#[test]
fn spectral_penalty_is_the_graph_dirichlet_form() {
    let atom = circle_atom(20);
    let basis = atom.spectral_decode_basis().expect("spectral basis");
    let laplacian = atom.surviving_laplacian();
    let phi = basis.vertex_basis().to_owned();
    let gram = phi.t().dot(&laplacian).dot(&phi); // Φᵀ L_W Φ
    let penalty = basis.penalty();
    let q = basis.selected_q();
    for a in 0..q {
        for b in 0..q {
            let expected = if a == b { basis.eigenvalues()[a] } else { 0.0 };
            assert!(
                (gram[[a, b]] - expected).abs() < 1e-8,
                "Φᵀ L_W Φ[{a},{b}] = {} != {expected}",
                gram[[a, b]]
            );
            assert!(
                (penalty[[a, b]] - expected).abs() < 1e-12,
                "penalty[{a},{b}] mismatch"
            );
        }
    }
}

// (a) — planted noisy circle: Nyström spectral coordinates achieve
// orientation-quotiented circular correlation > 0.95 against the true angle.
#[test]
fn nystrom_recovers_noisy_circle_angle() {
    let atom = circle_atom(24);
    let basis = atom.spectral_decode_basis().expect("spectral basis");
    assert_eq!(basis.selected_q(), 2);

    let n_rows = 240usize;
    let mut z = Array2::<f64>::zeros((n_rows, 2));
    let mut true_turns = vec![0.0; n_rows];
    // Deterministic radial jitter via a full-period LCG (no RNG dependency).
    let mut state = 0x2545F4914F6CDD1Du64;
    for row in 0..n_rows {
        let t = row as f64 / n_rows as f64;
        let angle = TAU * t;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = (state >> 40) as f64 / (1u64 << 24) as f64; // [0, 1)
        let radius = 1.0 + 0.05 * (2.0 * unit - 1.0);
        z[[row, 0]] = radius * angle.cos();
        z[[row, 1]] = radius * angle.sin();
        true_turns[row] = t;
    }

    let (phi, _jet) = basis.nystrom_coordinates(z.view()).expect("nystrom coords");
    let obs: Vec<ChartInterpObservation> = (0..n_rows)
        .map(|row| {
            let recovered = phi[[row, 1]].atan2(phi[[row, 0]]) / TAU;
            ChartInterpObservation {
                recovered_turns: recovered.rem_euclid(1.0),
                label_turns: true_turns[row],
                weight: 1.0,
            }
        })
        .collect();
    let report = chart_interp_score(&obs).expect("chart interp");
    assert!(
        report.circular_correlation > 0.95,
        "noisy-circle circular correlation {} should exceed 0.95",
        report.circular_correlation
    );
}

// (c) — the analytic Nyström jet matches a central-difference check to 1e-5
// relative (finite differences are legal in test code, banned in src).
#[test]
fn nystrom_jet_matches_central_difference() {
    let atom = circle_atom(24);
    let basis = atom.spectral_decode_basis().expect("spectral basis");
    let q = basis.selected_q();

    for &z in &[[0.8, 0.3], [-0.4, 0.9], [0.1, -0.6]] {
        let (_phi, jac) = basis.nystrom_coordinate(&z).expect("nystrom coordinate");
        let h = 1e-6;
        for c in 0..2 {
            let mut zp = z;
            let mut zm = z;
            zp[c] += h;
            zm[c] -= h;
            let (php, _) = basis.nystrom_coordinate(&zp).expect("nystrom +h");
            let (phm, _) = basis.nystrom_coordinate(&zm).expect("nystrom -h");
            for k in 0..q {
                let fd = (php[k] - phm[k]) / (2.0 * h);
                let analytic = jac[[k, c]];
                let tol = 1e-5 * analytic.abs().max(1e-3);
                assert!(
                    (fd - analytic).abs() <= tol,
                    "jet mismatch at z={z:?} k={k} c={c}: analytic={analytic}, fd={fd}"
                );
            }
        }
    }
}

// (b) — planted figure-eight (outside the typed zoo): the q-dimensional spectral
// decode reconstructs the shape with EV clearly beating the best single typed
// circle atom fit, both scored through the same public closed-form REML fitter.
#[test]
fn spectral_decode_beats_single_circle_on_figure_eight() {
    let (atom, positions, circle_coord) = figure_eight(16);
    let basis = atom
        .spectral_decode_basis()
        .expect("figure-eight spectral basis");
    let n = positions.nrows();

    // Spectral decode: design = the q eigengap eigenvectors at the vertices,
    // penalty = diag(λ) (the graph Dirichlet form).
    let phi_spectral = basis.vertex_basis().to_owned();
    let penalty_spectral = basis.penalty();
    let spectral_fit = gaussian_reml_multi_closed_form(
        phi_spectral.view(),
        positions.view(),
        penalty_spectral.view(),
        None,
        None,
    )
    .expect("spectral REML fit");
    let ev_spectral = explained_variance(&positions, &spectral_fit.fitted);

    // Best single typed circle atom: the race's periodic basis (3 columns,
    // one harmonic) evaluated on a single-loop coordinate, with its own
    // curvature (second-jet) roughness Gram, fit through the same entry point.
    let circle_eval = PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator");
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| circle_coord[i]);
    let (phi_circle, _jet) = circle_eval
        .evaluate(coords.view())
        .expect("circle basis evaluate");
    let second = circle_eval
        .second_jet(coords.view())
        .expect("circle second jet");
    let m_cols = phi_circle.ncols();
    let mut s_circle = Array2::<f64>::zeros((m_cols, m_cols));
    for row in 0..n {
        for mu in 0..m_cols {
            let hmu = second[[row, mu, 0, 0]];
            if hmu == 0.0 {
                continue;
            }
            for nu in 0..m_cols {
                s_circle[[mu, nu]] += hmu * second[[row, nu, 0, 0]];
            }
        }
    }
    let circle_fit = gaussian_reml_multi_closed_form(
        phi_circle.view(),
        positions.view(),
        s_circle.view(),
        None,
        None,
    )
    .expect("circle REML fit");
    let ev_circle = explained_variance(&positions, &circle_fit.fitted);

    assert!(
        ev_spectral > ev_circle + 0.10,
        "spectral decode EV {ev_spectral} must beat single circle EV {ev_circle} by a clear margin"
    );
}

// The Nyström extension is a first-class basis evaluator: the trait `evaluate`
// path agrees with the batched coordinate call.
#[test]
fn nystrom_evaluator_matches_batched_coordinates() {
    let atom = circle_atom(18);
    let basis = atom.spectral_decode_basis().expect("spectral basis");
    let evaluator = basis.evaluator();
    let z = Array2::<f64>::from_shape_fn((5, 2), |(i, j)| {
        let a = TAU * i as f64 / 5.0;
        if j == 0 { 0.9 * a.cos() } else { 0.9 * a.sin() }
    });
    let (phi_direct, jet_direct): (Array2<f64>, Array3<f64>) =
        basis.nystrom_coordinates(z.view()).expect("direct");
    let (phi_eval, jet_eval) = evaluator.evaluate(z.view()).expect("evaluator");
    assert_eq!(phi_direct.dim(), phi_eval.dim());
    assert_eq!(jet_direct.dim(), jet_eval.dim());
    for (a, b) in phi_direct.iter().zip(phi_eval.iter()) {
        assert!((a - b).abs() < 1e-14);
    }
    for (a, b) in jet_direct.iter().zip(jet_eval.iter()) {
        assert!((a - b).abs() < 1e-14);
    }
    // No analytic second/third jet is declared.
    assert!(evaluator.second_jet_dyn(z.view()).is_none());
    assert!(evaluator.third_jet_dyn(z.view()).is_none());
}
