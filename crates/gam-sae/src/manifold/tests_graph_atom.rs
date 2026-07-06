use crate::manifold::{CycleGraphAtom, OccupancyLaw};
use ndarray::Array2;

const WEEKDAY_ANCHORS: usize = 7;
const ROWS_PER_ANCHOR: usize = 24;
const KEEP_MARGIN: f64 = 1.4;
const DROP_MARGIN: f64 = 0.4;

fn sine_anchor_embeddings(anchors: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((anchors, 1), |(i, _)| {
        (std::f64::consts::TAU * i as f64 / anchors as f64).sin()
    })
}

fn arc_anchor_embeddings(anchors: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((anchors, 1), |(i, _)| i as f64 / (anchors - 1) as f64)
}

fn continuous_circle_rows(anchors: usize, rows_per_anchor: usize) -> Vec<f64> {
    let n = anchors * rows_per_anchor;
    (0..n).map(|i| i as f64 / n as f64).collect()
}

fn weekday_rows(anchors: usize, rows_per_anchor: usize) -> Vec<f64> {
    let mut rows = Vec::with_capacity(anchors * rows_per_anchor);
    for anchor in 0..anchors {
        for repeat in 0..rows_per_anchor {
            let jitter_rank = repeat as f64 - (rows_per_anchor - 1) as f64 * 0.5;
            let jitter =
                jitter_rank / (rows_per_anchor as f64 * anchors as f64 * rows_per_anchor as f64);
            rows.push((anchor as f64 / anchors as f64 + jitter).rem_euclid(1.0));
        }
    }
    rows
}

fn edge_deltas(anchors: usize, charge: f64, broken_edge: Option<usize>) -> Vec<f64> {
    (0..anchors)
        .map(|edge| {
            if Some(edge) == broken_edge {
                charge * DROP_MARGIN
            } else {
                charge * KEEP_MARGIN
            }
        })
        .collect()
}

#[test]
fn graph_atom_reads_continuous_circle_as_one_loop() {
    let anchors = WEEKDAY_ANCHORS;
    let rows = continuous_circle_rows(anchors, ROWS_PER_ANCHOR);
    let embeddings = sine_anchor_embeddings(anchors);
    let n_eff = rows.len() as f64;
    let charge = crate::manifold::graph_edge_rank_charge(n_eff, 1);
    let precisions = vec![1.0; anchors];
    let deltas = edge_deltas(anchors, charge, None);

    let atom = CycleGraphAtom::from_reml_cycle_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &precisions,
        &deltas,
    )
    .expect("continuous circle graph atom");
    let readout = atom.topology_readout();
    let smoothness = atom.surviving_smoothness_value();

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 1);
    assert_eq!(readout.surviving_edges, anchors);
    assert!(
        matches!(atom.occupancy(), OccupancyLaw::Uniform | OccupancyLaw::Continuous),
        "continuous circle occupancy should remain continuous/uniform, got {:?}",
        atom.occupancy()
    );
    assert!(smoothness > 0.0);
}

#[test]
fn graph_atom_reads_weekdays_as_atomic_cycle() {
    let anchors = WEEKDAY_ANCHORS;
    let rows = weekday_rows(anchors, ROWS_PER_ANCHOR);
    let embeddings = sine_anchor_embeddings(anchors);
    let n_eff = rows.len() as f64;
    let charge = crate::manifold::graph_edge_rank_charge(n_eff, 1);
    let precisions = vec![1.0; anchors];
    let deltas = edge_deltas(anchors, charge, None);

    let atom = CycleGraphAtom::from_reml_cycle_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &precisions,
        &deltas,
    )
    .expect("weekday graph atom");
    let readout = atom.topology_readout();
    let laplacian = atom.surviving_laplacian();

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 1);
    assert_eq!(readout.surviving_edges, anchors);
    assert_eq!(atom.occupancy(), OccupancyLaw::Discrete { anchors });
    assert_eq!(laplacian.dim(), (anchors, anchors));
    for row in 0..anchors {
        assert!((laplacian.row(row).sum()).abs() < 1e-12);
        assert_eq!(laplacian[[row, row]], 2.0);
        assert_eq!(laplacian[[row, (row + 1) % anchors]], -1.0);
    }

    let offset = 3usize;
    let beta_dim = offset + anchors * atom.fiber_rank();
    let op = atom.surviving_penalty_op(offset, beta_dim);
    assert_eq!(op.dim(), beta_dim);
    assert_eq!(op.output_range(), Some(offset..beta_dim));

    let mut beta = vec![0.0; beta_dim];
    for anchor in 0..anchors {
        beta[offset + anchor] = embeddings[[anchor, 0]];
    }
    let mut h_beta = vec![0.0; beta_dim];
    op.matvec(&beta, &mut h_beta);
    for row in 0..anchors {
        let mut expected = 0.0;
        for col in 0..anchors {
            expected += laplacian[[row, col]] * embeddings[[col, 0]];
        }
        assert!((h_beta[offset + row] - expected).abs() < 1e-12);
    }
}

#[test]
fn graph_atom_retires_broken_closing_edge_and_reads_interval() {
    let anchors = WEEKDAY_ANCHORS;
    let rows = continuous_circle_rows(anchors - 1, ROWS_PER_ANCHOR);
    let embeddings = arc_anchor_embeddings(anchors);
    let n_eff = rows.len() as f64;
    let charge = crate::manifold::graph_edge_rank_charge(n_eff, 1);
    let precisions = vec![1.0; anchors];
    let deltas = edge_deltas(anchors, charge, Some(anchors - 1));

    let atom = CycleGraphAtom::from_reml_cycle_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &precisions,
        &deltas,
    )
    .expect("broken arc graph atom");
    let readout = atom.topology_readout();
    let laplacian = atom.surviving_laplacian();
    let closing_weight = -laplacian[[0, anchors - 1]];

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 0);
    assert_eq!(readout.surviving_edges, anchors - 1);
    assert!(!atom.surviving_edges()[anchors - 1]);
    assert_eq!(closing_weight, 0.0);
}
