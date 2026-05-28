//! β-coefficient graph for Arrow-Schur preconditioner construction.
//!
//! Nodes are β coefficient blocks (indexed `0..num_blocks`); edges record
//! rows where two blocks co-occur with nonzero cross-block contributions.
//! Two blocks co-occur in a row when both have at least one nonzero column
//! entry in that row's `H_tβ` slab — i.e. both blocks actively couple to
//! the latent coordinate for that observation.
//!
//! The graph is used by [`ClusterJacobiPreconditioner`] (connected-component
//! partition) and [`AdditiveSchwarzPreconditioner`] (1-hop neighbourhood
//! expansion). Both preconditioners are built on top of this graph rather
//! than duplicating the connectivity scan.
//!
//! [`ClusterJacobiPreconditioner`]: super::arrow_schur::ClusterJacobiPreconditioner
//! [`AdditiveSchwarzPreconditioner`]: super::arrow_schur::AdditiveSchwarzPreconditioner

use std::ops::Range;

/// Edge in the β-coefficient coupling graph.
///
/// Presence of edge `(a, b)` with `a < b` means blocks `a` and `b`
/// co-occur in at least one observation row's `H_tβ` slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BetaEdge {
    /// Lower block index (always `< b`).
    pub a: usize,
    /// Higher block index.
    pub b: usize,
}

/// Sparse β-coefficient coupling graph over `num_blocks` nodes.
///
/// Each node corresponds to one entry in a `block_offsets` slice.
/// Edges are stored as a sorted, deduplicated list of `(a, b)` pairs.
/// Adjacency queries are O(degree) via a CSR-style row-start array.
#[derive(Debug, Clone)]
pub struct BetaCouplingGraph {
    /// Number of β blocks (nodes).
    pub num_blocks: usize,
    /// Sorted, deduplicated edge list.
    edges: Vec<BetaEdge>,
    /// `adj_start[b]` = first index in `adj_targets` for block `b`.
    adj_start: Vec<usize>,
    /// Adjacency lists: for each block `b`, the list of neighbouring blocks.
    adj_targets: Vec<usize>,
}

impl BetaCouplingGraph {
    /// Build the coupling graph from an Arrow-Schur system's row blocks.
    ///
    /// `block_offsets` gives the column ranges of each β block in the K-vector.
    /// `htbeta_rows` is a slice of `(d × K)` row-block matrices (one per
    /// observation); each matrix is indexed as `row[[c, col]]`.
    ///
    /// A column `col` in block `b` (i.e. `block_offsets[b].contains(&col)`) is
    /// "active" in row `i` when at least one entry `htbeta[[c, col]]` is nonzero.
    /// Two distinct blocks `a != b` share an edge when both are active in the
    /// same row.
    pub fn build<M>(block_offsets: &[Range<usize>], htbeta_rows: &[M]) -> Self
    where
        M: BlockHtbetaRow,
    {
        let num_blocks = block_offsets.len();
        if num_blocks == 0 {
            return Self {
                num_blocks: 0,
                edges: Vec::new(),
                adj_start: vec![0],
                adj_targets: Vec::new(),
            };
        }

        // Collect all edges as (min, max) pairs; deduplicate at the end.
        let mut edge_set: Vec<(usize, usize)> = Vec::new();

        for row in htbeta_rows {
            // Find which blocks have at least one nonzero column in this row.
            let mut active: Vec<usize> = Vec::new();
            for (b, range) in block_offsets.iter().enumerate() {
                if row.has_nonzero_in_range(range.clone()) {
                    active.push(b);
                }
            }
            // All pairs of active blocks are edges.
            for i in 0..active.len() {
                for j in (i + 1)..active.len() {
                    let lo = active[i].min(active[j]);
                    let hi = active[i].max(active[j]);
                    edge_set.push((lo, hi));
                }
            }
        }

        // Deduplicate.
        edge_set.sort_unstable();
        edge_set.dedup();

        let edges: Vec<BetaEdge> =
            edge_set.iter().map(|&(a, b)| BetaEdge { a, b }).collect();

        // Build CSR adjacency (undirected: add both directions).
        let mut degree = vec![0usize; num_blocks];
        for &BetaEdge { a, b } in &edges {
            degree[a] += 1;
            degree[b] += 1;
        }
        let mut adj_start = vec![0usize; num_blocks + 1];
        for i in 0..num_blocks {
            adj_start[i + 1] = adj_start[i] + degree[i];
        }
        let total_adj = adj_start[num_blocks];
        let mut adj_targets = vec![0usize; total_adj];
        let mut cursor = adj_start[..num_blocks].to_vec();
        for &BetaEdge { a, b } in &edges {
            adj_targets[cursor[a]] = b;
            cursor[a] += 1;
            adj_targets[cursor[b]] = a;
            cursor[b] += 1;
        }

        Self { num_blocks, edges, adj_start, adj_targets }
    }

    /// Iterator over block indices adjacent to `node`.
    pub fn neighbours(&self, node: usize) -> &[usize] {
        let start = self.adj_start[node];
        let end = self.adj_start[node + 1];
        &self.adj_targets[start..end]
    }

    /// All edges in the graph.
    pub fn edges(&self) -> &[BetaEdge] {
        &self.edges
    }

    /// Compute connected-component labels via union-find.
    ///
    /// Returns a vector of length `num_blocks` where `labels[b]` is the
    /// (canonical) component index for block `b`. Component indices are in
    /// `0..num_components`.  The second return value is the number of
    /// components.
    pub fn connected_components(&self) -> (Vec<usize>, usize) {
        let n = self.num_blocks;
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0u8; n];

        fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path-halving
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut Vec<usize>, rank: &mut Vec<u8>, x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx == ry {
                return;
            }
            if rank[rx] < rank[ry] {
                parent[rx] = ry;
            } else if rank[rx] > rank[ry] {
                parent[ry] = rx;
            } else {
                parent[ry] = rx;
                rank[rx] += 1;
            }
        }

        for &BetaEdge { a, b } in &self.edges {
            union(&mut parent, &mut rank, a, b);
        }

        // Compress and relabel in traversal order (deterministic).
        let mut label_map = vec![usize::MAX; n];
        let mut next_label = 0usize;
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let root = find(&mut parent, i);
            if label_map[root] == usize::MAX {
                label_map[root] = next_label;
                next_label += 1;
            }
            labels[i] = label_map[root];
        }
        (labels, next_label)
    }

    /// Partition the blocks into groups by connected component.
    ///
    /// Returns a `Vec<Vec<usize>>` where each inner vec is the sorted list of
    /// block indices in one component. Components are ordered by their
    /// smallest member index.
    pub fn component_partition(&self) -> Vec<Vec<usize>> {
        let (labels, num_comp) = self.connected_components();
        let mut parts: Vec<Vec<usize>> = vec![Vec::new(); num_comp];
        for (b, &comp) in labels.iter().enumerate() {
            parts[comp].push(b);
        }
        // Each part is already in ascending order because we iterate b in order.
        parts
    }

    /// Expand a set of block indices by 1-hop neighbours.
    ///
    /// For each block in `seed`, adds all direct neighbours. The result is
    /// deduplicated and sorted.
    pub fn expand_one_hop(&self, seed: &[usize]) -> Vec<usize> {
        let mut expanded: Vec<usize> = seed.to_vec();
        for &b in seed {
            for &nb in self.neighbours(b) {
                expanded.push(nb);
            }
        }
        expanded.sort_unstable();
        expanded.dedup();
        expanded
    }
}

/// Trait abstracting row-block access for graph construction.
///
/// The implementation for `ndarray::Array2<f64>` is provided below.
/// Custom callers can implement this for matrix-free row representations.
pub trait BlockHtbetaRow {
    /// Returns `true` when any entry `htbeta[[c, col]]` is nonzero
    /// for `col` in `range`.
    fn has_nonzero_in_range(&self, range: Range<usize>) -> bool;
}

impl BlockHtbetaRow for ndarray::Array2<f64> {
    fn has_nonzero_in_range(&self, range: Range<usize>) -> bool {
        let d = self.nrows();
        for col in range {
            for c in 0..d {
                if self[[c, col]] != 0.0 {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_htbeta(d: usize, k: usize, nonzeros: &[(usize, usize)]) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((d, k));
        for &(c, col) in nonzeros {
            m[[c, col]] = 1.0;
        }
        m
    }

    /// Three blocks; rows couple (0,1) and (1,2) but not (0,2) directly.
    /// Connected-components should give a single component.
    #[test]
    fn graph_three_blocks_one_component() {
        // block 0: cols 0..2, block 1: cols 2..4, block 2: cols 4..6
        let offsets: Vec<Range<usize>> = vec![0..2, 2..4, 4..6];
        let rows = vec![
            // row 0: blocks 0 and 1 active
            make_htbeta(1, 6, &[(0, 0), (0, 3)]),
            // row 1: blocks 1 and 2 active
            make_htbeta(1, 6, &[(0, 2), (0, 5)]),
        ];
        let g = BetaCouplingGraph::build(&offsets, &rows);
        assert_eq!(g.num_blocks, 3);
        assert_eq!(g.edges().len(), 2);
        let (_, nc) = g.connected_components();
        assert_eq!(nc, 1);
        let parts = g.component_partition();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], vec![0, 1, 2]);
    }

    /// Three blocks; no row couples across all of them → two components.
    #[test]
    fn graph_disconnected_two_components() {
        let offsets: Vec<Range<usize>> = vec![0..2, 2..4, 4..6];
        let rows = vec![
            // row 0: only block 0 active
            make_htbeta(1, 6, &[(0, 1)]),
            // row 1: blocks 1 and 2 active
            make_htbeta(1, 6, &[(0, 2), (0, 4)]),
        ];
        let g = BetaCouplingGraph::build(&offsets, &rows);
        assert_eq!(g.edges().len(), 1); // only edge (1,2)
        let (_, nc) = g.connected_components();
        assert_eq!(nc, 2);
        let parts = g.component_partition();
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn expand_one_hop_basic() {
        let offsets: Vec<Range<usize>> = vec![0..1, 1..2, 2..3, 3..4];
        let rows = vec![
            // 0-1 edge
            make_htbeta(1, 4, &[(0, 0), (0, 1)]),
            // 1-2 edge
            make_htbeta(1, 4, &[(0, 1), (0, 2)]),
            // 2-3 edge
            make_htbeta(1, 4, &[(0, 2), (0, 3)]),
        ];
        let g = BetaCouplingGraph::build(&offsets, &rows);
        // Seed = {1}: neighbours are {0, 2}. Expanded = {0, 1, 2}.
        let expanded = g.expand_one_hop(&[1]);
        assert_eq!(expanded, vec![0, 1, 2]);
    }
}
