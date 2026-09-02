use crate::manifold::GraphEdge;
use crate::sparse_dict::BlockSparseFit;
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

fn circ_err(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

fn harmonic_code(spikes: &[(f64, f32)], h_count: usize) -> Vec<f32> {
    let mut code = Vec::with_capacity(2 * h_count);
    for h in 1..=h_count {
        let mut cos_sum = 0.0_f32;
        let mut sin_sum = 0.0_f32;
        for &(t, amp) in spikes {
            let phase = std::f64::consts::TAU * h as f64 * t;
            cos_sum += amp * phase.cos() as f32;
            sin_sum += amp * phase.sin() as f32;
        }
        code.push(cos_sum);
        code.push(sin_sum);
    }
    code
}

fn harmonic_fit_from_codes(rows: &[Vec<f32>], block_size: usize) -> BlockSparseFit {
    let n = rows.len();
    let mut x = Array2::<f32>::zeros((n, block_size));
    let mut codes = ndarray::Array3::<f32>::zeros((n, 1, block_size));
    for row in 0..n {
        for col in 0..block_size {
            x[[row, col]] = rows[row][col];
            codes[[row, 0, col]] = rows[row][col];
        }
    }
    BlockSparseFit {
        decoder: Array2::<f32>::eye(block_size),
        blocks: Array2::<u32>::zeros((n, 1)),
        gates: Array2::<f32>::ones((n, 1)),
        codes,
        gamma: 1.0,
        block_utilization: vec![1.0],
        block_stable_rank: vec![2.0],
        matryoshka_prefix_losses: Vec::new(),
        explained_variance: 1.0,
        epochs: 0,
        convergence: crate::sparse_dict::BlockSparseConvergence::trivially_converged(),
        block_topk: 1,
        block_size,
    }
}

