use crate::manifold::{
    CycleGraphAtom, GraphCompressionKind, GraphEdge, OccupancyLaw, graph_edge_rank_charge,
};
use ndarray::Array2;

const WEEKDAY_ANCHORS: usize = 7;
const ROWS_PER_ANCHOR: usize = 24;
const KEEP_MARGIN: f64 = 1.4;
const DROP_MARGIN: f64 = 0.4;

fn unit_circle_anchor_embeddings(anchors: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((anchors, 2), |(i, j)| {
        let phase = std::f64::consts::TAU * i as f64 / anchors as f64;
        if j == 0 { phase.cos() } else { phase.sin() }
    })
}

fn arc_anchor_embeddings(anchors: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((anchors, 1), |(i, _)| i as f64 / (anchors - 1) as f64)
}

fn star_anchor_embeddings(leaves: usize) -> Array2<f64> {
    let anchors = leaves + 1;
    Array2::<f64>::from_shape_fn((anchors, 2), |(i, j)| {
        if i == 0 {
            0.0
        } else {
            let phase = std::f64::consts::TAU * (i - 1) as f64 / leaves as f64;
            if j == 0 { phase.cos() } else { phase.sin() }
        }
    })
}

fn two_cycle_anchor_embeddings() -> Array2<f64> {
    Array2::<f64>::from_shape_fn((8, 2), |(i, j)| {
        let cycle = i / 4;
        let local = i % 4;
        let center = if cycle == 0 { -4.0 } else { 4.0 };
        let phase = std::f64::consts::TAU * local as f64 / 4.0;
        if j == 0 {
            center + phase.cos()
        } else {
            phase.sin()
        }
    })
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

fn edge(a: usize, b: usize) -> GraphEdge {
    GraphEdge::new(a, b).expect("valid test edge")
}

fn path_edges(anchors: usize) -> Vec<GraphEdge> {
    (0..anchors - 1).map(|i| edge(i, i + 1)).collect()
}

fn cycle_edges(offset: usize, anchors: usize) -> Vec<GraphEdge> {
    (0..anchors)
        .map(|i| edge(offset + i, offset + ((i + 1) % anchors)))
        .collect()
}

fn keep_all(edges: usize, charge: f64) -> (Vec<f64>, Vec<f64>) {
    (vec![1.0; edges], vec![charge * KEEP_MARGIN; edges])
}

fn keep_with_extra_retired(
    edges: &[GraphEdge],
    keep: &[GraphEdge],
    charge: f64,
) -> (Vec<f64>, Vec<f64>) {
    let precisions = vec![1.0; edges.len()];
    let deltas = edges
        .iter()
        .map(|edge| {
            if keep.contains(edge) {
                charge * KEEP_MARGIN
            } else {
                charge * DROP_MARGIN
            }
        })
        .collect();
    (precisions, deltas)
}

#[test]
fn graph_atom_reads_continuous_circle_as_one_loop_from_knn_edges() {
    let anchors = WEEKDAY_ANCHORS;
    let rows = continuous_circle_rows(anchors, ROWS_PER_ANCHOR);
    let embeddings = unit_circle_anchor_embeddings(anchors);
    let edges = CycleGraphAtom::knn_candidate_edges(embeddings.view()).expect("knn edges");
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let (precisions, deltas) = keep_all(edges.len(), charge);

    let atom = CycleGraphAtom::from_reml_candidate_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &edges,
        &precisions,
        &deltas,
    )
    .expect("continuous circle graph atom");
    let readout = atom.topology_readout();
    let smoothness = atom.surviving_smoothness_value();
    let selection = atom.structure_selection();

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 1);
    assert_eq!(readout.surviving_edges, anchors);
    assert_eq!(selection.compression.kind, GraphCompressionKind::Circle);
    assert!(selection.selected);
    assert!(
        matches!(atom.occupancy(), OccupancyLaw::Uniform | OccupancyLaw::Continuous),
        "continuous circle occupancy should remain continuous/uniform, got {:?}",
        atom.occupancy()
    );
    assert!(smoothness > 0.0);
}

#[test]
fn graph_atom_reads_weekdays_as_atomic_cycle_without_fixed_menu_selection() {
    let anchors = WEEKDAY_ANCHORS;
    let rows = weekday_rows(anchors, ROWS_PER_ANCHOR);
    let embeddings = unit_circle_anchor_embeddings(anchors);
    let edges = CycleGraphAtom::knn_candidate_edges(embeddings.view()).expect("knn edges");
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let (precisions, deltas) = keep_all(edges.len(), charge);

    let atom = CycleGraphAtom::from_reml_knn_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &precisions,
        &deltas,
    )
    .expect("weekday graph atom");
    let readout = atom.topology_readout();
    let laplacian = atom.surviving_laplacian();
    let selection = atom.structure_selection();

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 1);
    assert_eq!(readout.surviving_edges, anchors);
    assert_eq!(atom.occupancy(), OccupancyLaw::Discrete { anchors });
    assert_eq!(selection.compression.kind, GraphCompressionKind::Circle);
    assert!(selection.total_edge_charge > 0.0);
    assert!(selection.margin > 0.0);
    assert_eq!(laplacian.dim(), (anchors, anchors));
    for row in 0..anchors {
        assert!((laplacian.row(row).sum()).abs() < 1e-12);
        assert_eq!(laplacian[[row, row]], 2.0);
    }

    let offset = 3usize;
    let beta_dim = offset + anchors * atom.fiber_rank();
    let op = atom.surviving_penalty_op(offset, beta_dim);
    assert_eq!(op.dim(), beta_dim);
    assert_eq!(op.output_range(), Some(offset..beta_dim));

    let mut beta = vec![0.0; beta_dim];
    for anchor in 0..anchors {
        for channel in 0..atom.fiber_rank() {
            beta[offset + anchor * atom.fiber_rank() + channel] = embeddings[[anchor, channel]];
        }
    }
    let mut h_beta = vec![0.0; beta_dim];
    op.matvec(&beta, &mut h_beta);
    for row in 0..anchors {
        for channel in 0..atom.fiber_rank() {
            let mut expected = 0.0;
            for col in 0..anchors {
                expected += laplacian[[row, col]] * embeddings[[col, channel]];
            }
            let idx = offset + row * atom.fiber_rank() + channel;
            assert!((h_beta[idx] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn learned_graph_reads_path_as_interval() {
    let anchors = WEEKDAY_ANCHORS;
    let rows = continuous_circle_rows(anchors - 1, ROWS_PER_ANCHOR);
    let embeddings = arc_anchor_embeddings(anchors);
    let edges = path_edges(anchors);
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let (precisions, deltas) = keep_all(edges.len(), charge);

    let atom = CycleGraphAtom::from_reml_candidate_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &edges,
        &precisions,
        &deltas,
    )
    .expect("path graph atom");
    let readout = atom.topology_readout();

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 0);
    assert_eq!(readout.surviving_edges, anchors - 1);
    assert_eq!(
        atom.certified_compression().kind,
        GraphCompressionKind::Interval
    );
}

#[test]
fn learned_graph_reads_two_disconnected_cycles() {
    let rows = weekday_rows(8, ROWS_PER_ANCHOR);
    let embeddings = two_cycle_anchor_embeddings();
    let mut edges = cycle_edges(0, 4);
    edges.extend(cycle_edges(4, 4));
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let (precisions, deltas) = keep_all(edges.len(), charge);

    let atom = CycleGraphAtom::from_reml_candidate_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &edges,
        &precisions,
        &deltas,
    )
    .expect("two-cycle graph atom");
    let readout = atom.topology_readout();

    assert_eq!(readout.b0, 2);
    assert_eq!(readout.b1, 2);
    assert_eq!(
        atom.certified_compression().kind,
        GraphCompressionKind::DisconnectedCycles
    );
}

#[test]
fn learned_graph_reads_branching_tree_and_detects_branch_vertex() {
    let leaves = 5usize;
    let embeddings = star_anchor_embeddings(leaves);
    let rows = continuous_circle_rows(leaves + 1, ROWS_PER_ANCHOR);
    let keep = (1..=leaves).map(|i| edge(0, i)).collect::<Vec<_>>();
    let mut edges = keep.clone();
    edges.extend((1..leaves).map(|i| edge(i, i + 1)));
    let n_eff = rows.len() as f64;
    let charge = graph_edge_rank_charge(n_eff, embeddings.ncols());
    let (precisions, deltas) = keep_with_extra_retired(&edges, &keep, charge);

    let atom = CycleGraphAtom::from_reml_candidate_edges(
        embeddings.view(),
        &rows,
        n_eff,
        &edges,
        &precisions,
        &deltas,
    )
    .expect("branching tree graph atom");
    let readout = atom.topology_readout();
    let degrees = atom.surviving_degrees();

    assert_eq!(readout.b0, 1);
    assert_eq!(readout.b1, 0);
    assert!(
        degrees.iter().any(|&degree| degree > 2),
        "branching tree must expose a degree>2 vertex: {degrees:?}"
    );
    assert_eq!(atom.certified_compression().kind, GraphCompressionKind::Tree);
}
