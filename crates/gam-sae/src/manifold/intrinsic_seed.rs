//! Intrinsic-metric (Landmark-Isomap) seeder for the SAE manifold fit
//! (#2240 re-scope / #2280 stage-0).
//!
//! The PCA seed ([`super::sae_pca_seed_initial_coords`]) embeds the residual by
//! its leading LINEAR principal directions. On a manifold that is FOLDED in the
//! ambient space — the canonical example is a swiss roll, a flat 2-D sheet rolled
//! up so that ambient-near points are geodesically far — a linear projection
//! collapses the fold onto itself, and every downstream chart family (flat patch,
//! Duchon sheet, torus, …) inherits a seed whose coordinates are self-overlapping.
//! No amount of joint refit unfolds a seed that starts creased.
//!
//! This module builds the seed from the INTRINSIC (geodesic) metric instead, via
//! classical Landmark-Isomap (de Silva & Tenenbaum, 2003):
//!
//!   1. a deterministic symmetric kNN graph over the ambient rows (edge weight =
//!      Euclidean distance), bridged to full connectivity so geodesics are finite;
//!   2. deterministic farthest-point landmarks (the same greedy coverage pattern
//!      the encoder chart seeder and the Tier-1 seeder use) to bound the geodesic
//!      cost to `L` shortest-path trees rather than `n`;
//!   3. landmark-source Dijkstra shortest paths → geodesic distances from every
//!      landmark to every row;
//!   4. classical multidimensional scaling on the `L × L` landmark geodesic matrix
//!      — double-centering + a SELF-ADJOINT eigendecomposition, taking the top
//!      POSITIVE eigenvalues of the centered Gram (NOT SVD singular values); and
//!   5. the L-Isomap distance-based (Nyström-style) extension of the landmark
//!      coordinates to every row.
//!
//! [`intrinsic_geodesic_embedding`] returns those coordinates for every row: on
//! a fold it unrolls where PCA creases, and on a non-fold it is a near-isometric
//! embedding that ties the linear seed. `structure_harvest.rs` and the local
//! atlas read it directly.
//!
//! Determinism is fleet law: no RNG anywhere. kNN ties break by index, landmarks
//! by first-wins-lowest-index, Dijkstra by (distance, node) so equal-cost frontier
//! nodes pop in index order, and the eigendecomposition is faer's deterministic
//! self-adjoint solver. Same input ⇒ bit-identical coordinates run-to-run.

use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use faer::Side;

/// Neighborhood degree `k` for the intrinsic seed's kNN graph, derived from the
/// embedding dimension and the sample size rather than a tuned scalar — the SAME
/// principled rule the topology seed uses (`super::pca_seed::topology_seed_knn`):
///   * geometric faithfulness — a geodesic graph approximates the manifold metric
///     only if each node links its full local tangent star (`≥ 2d+1` neighbours
///     span a `d`-simplex neighborhood); and
///   * connectivity — a kNN graph on `n` points is connected with high probability
///     only for `k ≳ log₂ n` (Penrose, random geometric graphs).
/// Take the larger of the two floors so the graph is both faithful and (almost
/// surely) connected before the explicit bridging pass runs.
fn intrinsic_seed_knn(n_points: usize, d: usize) -> usize {
    let tangent_floor = 2 * d + 1;
    let connectivity_floor = (n_points.max(2) as f64).log2().ceil() as usize;
    tangent_floor.max(connectivity_floor).max(2)
}

/// Number of farthest-point landmarks the geodesic computation runs Dijkstra from.
///
/// Landmark-Isomap needs `L ≥ d + 1` landmarks to span a `d`-dimensional MDS
/// solution, and its embedding error decreases as more landmarks cover the
/// manifold. The cost is `L` shortest-path trees plus an `L × L` eigensolve, so
/// `L` is set to a coverage target that grows sublinearly in `n` — `⌈c·√n⌉`, the
/// standard covering-number scaling for a fixed-radius net — floored at `2(d+1)`
/// so even tiny samples over-determine the MDS solution, and capped at `n` (every
/// row is a landmark when the sample is smaller than the target). The constant `c`
/// is the coverage multiplier; `4` places ~16 landmarks around a 2-D sheet of
/// n = 16 and ~80 at n = 400, dense enough that the L-Isomap extension residual is
/// negligible while the eigensolve stays trivially small.
fn intrinsic_landmark_count(n_points: usize, d: usize) -> usize {
    const COVERAGE_MULTIPLIER: f64 = 4.0;
    let coverage = (COVERAGE_MULTIPLIER * (n_points as f64).sqrt()).ceil() as usize;
    let floor = 2 * (d + 1);
    coverage.max(floor).min(n_points)
}

fn squared_distance(z: ArrayView2<'_, f64>, a: usize, b: usize) -> f64 {
    let mut acc = 0.0;
    for c in 0..z.ncols() {
        let diff = z[[a, c]] - z[[b, c]];
        acc += diff * diff;
    }
    acc
}

/// Deterministic symmetric kNN graph as an adjacency list of `(neighbor, weight)`
/// with `weight` the Euclidean distance. Each node links its `k` nearest rows
/// (ties broken by ascending index); the graph is symmetrized by UNION (an edge is
/// present if either endpoint chose the other), then bridged to a single connected
/// component so every landmark reaches every row and no geodesic is infinite.
///
/// Bridging is deterministic: the added edges are the minimum spanning tree of
/// the component graph under the strict order `(squared distance, min_row,
/// max_row)`, so connectivity is restored through the true nearest
/// cross-component pairs, and a graph that is already connected is returned
/// untouched. Because that order is strict, the spanning tree is unique. Each
/// Borůvka pass joins every component to its own nearest other component, which
/// at least halves the component count. So `C` components cost `⌈log₂ C⌉`
/// all-pairs scans, rather than the `C − 1` scans of adding one global-minimum
/// edge at a time. Both methods yield that same unique tree, so the graph is
/// bit-identical to the one-edge-per-scan result.
pub(crate) fn deterministic_knn_graph(z: ArrayView2<'_, f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
    let n = z.nrows();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    if n == 0 {
        return adj;
    }
    let k = k.min(n.saturating_sub(1)).max(1);
    // Undirected edge set keyed by (min, max) so the UNION symmetrization never
    // double-inserts a mutually-chosen edge.
    let mut edges: std::collections::BTreeMap<(usize, usize), f64> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                dists.push((squared_distance(z, i, j), j));
            }
        }
        // Only the `k` nearest are kept: partition them out in linear time and
        // order just that prefix. `(distance, index)` is a strict total order,
        // so the prefix is the same set in the same order a full sort yields.
        let by_distance_then_index =
            |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1));
        let keep = k.min(dists.len());
        if keep > 0 && keep < dists.len() {
            dists.select_nth_unstable_by(keep - 1, by_distance_then_index);
        }
        dists[..keep].sort_unstable_by(by_distance_then_index);
        for &(dist2, j) in &dists[..keep] {
            let key = (i.min(j), i.max(j));
            edges.entry(key).or_insert_with(|| dist2.sqrt());
        }
    }
    // Bridge components so geodesics are finite. Union-find over the current edge
    // set, then repeatedly add the shortest inter-component edge.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &(a, b) in edges.keys() {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    // Borůvka passes: every component records its nearest outgoing pair under the
    // strict order `(d², i, j)`. Pairs are scanned in ascending `(i, j)`, and a
    // candidate replaces the incumbent only when strictly closer, so the first
    // minimum is kept and it is the lexicographic minimum. Under a strict edge
    // order every such pick is an edge of the unique minimum spanning tree, so
    // the whole batch is added at once. Passes ≤ ⌈log₂ #components⌉.
    loop {
        let roots: Vec<usize> = (0..n).map(|x| find(&mut parent, x)).collect();
        let mut nearest: Vec<Option<(f64, usize, usize)>> = vec![None; n];
        for i in 0..n {
            let ri = roots[i];
            for j in (i + 1)..n {
                let rj = roots[j];
                if rj == ri {
                    continue;
                }
                let d2 = squared_distance(z, i, j);
                for root in [ri, rj] {
                    let closer = match nearest[root] {
                        None => true,
                        Some((bd, _, _)) => d2.total_cmp(&bd) == Ordering::Less,
                    };
                    if closer {
                        nearest[root] = Some((d2, i, j));
                    }
                }
            }
        }
        let mut bridges: Vec<(f64, usize, usize)> = nearest.into_iter().flatten().collect();
        if bridges.is_empty() {
            break; // single component
        }
        bridges.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        bridges.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
        for (d2, i, j) in bridges {
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri != rj {
                edges.insert((i, j), d2.sqrt());
                parent[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    for (&(a, b), &w) in &edges {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    adj
}

/// Deterministic farthest-point landmark selection over the ambient rows: seed
/// from row 0, then repeatedly add the row maximally far (in Euclidean distance)
/// from the chosen set, first-wins on ties: coverage-maximizing and reproducible
/// run-to-run.
pub(crate) fn farthest_point_landmarks(z: ArrayView2<'_, f64>, count: usize) -> Vec<usize> {
    let n = z.nrows();
    if n == 0 {
        return Vec::new();
    }
    let target = count.max(1).min(n);
    let mut chosen: Vec<usize> = Vec::with_capacity(target);
    chosen.push(0);
    let mut nearest_sq: Vec<f64> = (0..n).map(|r| squared_distance(z, r, 0)).collect();
    while chosen.len() < target {
        let mut best = 0usize;
        let mut best_d = -1.0;
        for r in 0..n {
            if nearest_sq[r] > best_d {
                best_d = nearest_sq[r];
                best = r;
            }
        }
        if best_d <= 0.0 {
            break; // remaining rows coincide with a chosen landmark
        }
        chosen.push(best);
        for r in 0..n {
            let dr = squared_distance(z, r, best);
            if dr < nearest_sq[r] {
                nearest_sq[r] = dr;
            }
        }
    }
    chosen
}

/// Min-heap frontier node for Dijkstra. Orders by ASCENDING distance then
/// ASCENDING node index, so `BinaryHeap` (a max-heap) pops the smallest-distance,
/// smallest-index frontier node — the deterministic tie-break the seed doctrine
/// requires.
#[derive(PartialEq)]
struct DijkstraNode {
    dist: f64,
    node: usize,
}
impl Eq for DijkstraNode {}
impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the max-heap yields the minimum; distances here are finite
        // non-negative path costs, so total_cmp is a total order.
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Single-source Dijkstra shortest-path distances from `source` over the weighted
/// adjacency graph. Returns a length-`n` vector of geodesic distances (finite for
/// every node on a connected graph).
fn dijkstra(adj: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(DijkstraNode {
        dist: 0.0,
        node: source,
    });
    while let Some(DijkstraNode { dist: d, node }) = heap.pop() {
        if d > dist[node] {
            continue;
        }
        for &(nbr, w) in &adj[node] {
            let nd = d + w;
            if nd < dist[nbr] {
                dist[nbr] = nd;
                heap.push(DijkstraNode {
                    dist: nd,
                    node: nbr,
                });
            }
        }
    }
    dist
}

/// Geodesic distances from every landmark to every row: row `l` is
/// [`dijkstra`] from landmark `landmarks[l]`. Shape `(L, n)`.
pub(crate) fn landmark_geodesics(adj: &[Vec<(usize, f64)>], landmarks: &[usize]) -> Array2<f64> {
    let n = adj.len();
    let l = landmarks.len();
    let mut out = Array2::<f64>::zeros((l, n));
    for (li, &src) in landmarks.iter().enumerate() {
        let d = dijkstra(adj, src);
        for j in 0..n {
            out[[li, j]] = d[j];
        }
    }
    out
}

/// Classical Landmark-Isomap embedding of the ambient rows `z` into `d`
/// dimensions. Returns `(n, d)` intrinsic coordinates. The columns are ordered by
/// descending MDS eigenvalue (leading intrinsic axis first); a column beyond the
/// number of positive eigenvalues is left at zero (a genuinely lower-rank
/// manifold). Deterministic and RNG-free.
pub fn intrinsic_geodesic_embedding(
    z: ArrayView2<'_, f64>,
    d: usize,
) -> Result<Array2<f64>, String> {
    intrinsic_geodesic_embedding_on_graph(z, d, &intrinsic_knn_graph(z, d))
}

/// The neighbourhood graph the intrinsic embedding measures geodesics on: the
/// deterministic symmetric kNN graph at `intrinsic_seed_knn`'s degree, bridged to one
/// component. A caller that needs the same graph for another purpose (the local atlas
/// grows patch membership through it) builds it once here and hands it to
/// `intrinsic_geodesic_embedding_on_graph`.
pub(crate) fn intrinsic_knn_graph(z: ArrayView2<'_, f64>, d: usize) -> Vec<Vec<(usize, f64)>> {
    deterministic_knn_graph(z, intrinsic_seed_knn(z.nrows(), d))
}

/// `intrinsic_geodesic_embedding` on a neighbourhood graph the caller already built
/// with `intrinsic_knn_graph` from the same rows and dimension.
pub(crate) fn intrinsic_geodesic_embedding_on_graph(
    z: ArrayView2<'_, f64>,
    d: usize,
    adj: &[Vec<(usize, f64)>],
) -> Result<Array2<f64>, String> {
    let n = z.nrows();
    if d == 0 {
        return Ok(Array2::<f64>::zeros((n, 0)));
    }
    let mut out = Array2::<f64>::zeros((n, d));
    if n == 0 || z.ncols() == 0 {
        return Ok(out);
    }
    for ((row, col), &value) in z.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "intrinsic_seed: Z must be finite; Z[{row}, {col}] = {value}"
            ));
        }
    }
    if n == 1 {
        return Ok(out);
    }
    if n == 2 {
        // Two points have a unique centered one-dimensional MDS realization up
        // to sign: place them at ± half their full ambient distance. Copying a
        // leading raw feature (the old path) discarded separation carried by
        // any other ambient column and was neither centered nor isometric.
        let dist2 = squared_distance(z, 0, 1);
        if !dist2.is_finite() {
            return Err(
                "intrinsic_seed: pairwise distance overflowed; rescale Z before seeding"
                    .to_string(),
            );
        }
        let half_distance = 0.5 * dist2.sqrt();
        out[[0, 0]] = -half_distance;
        out[[1, 0]] = half_distance;
        return Ok(out);
    }
    let l_count = intrinsic_landmark_count(n, d);
    let landmarks = farthest_point_landmarks(z, l_count);
    let l = landmarks.len();
    if l < 2 {
        return Ok(out);
    }
    let geo = landmark_geodesics(adj, &landmarks);

    // Squared landmark-to-landmark geodesic matrix, symmetrized (independent
    // Dijkstra runs agree up to float noise; averaging makes it exactly symmetric
    // for the strict self-adjoint eigensolver).
    let mut d2 = Array2::<f64>::zeros((l, l));
    for a in 0..l {
        for b in 0..l {
            let g = geo[[a, landmarks[b]]];
            d2[[a, b]] = g * g;
        }
    }
    for a in 0..l {
        for b in (a + 1)..l {
            let avg = 0.5 * (d2[[a, b]] + d2[[b, a]]);
            d2[[a, b]] = avg;
            d2[[b, a]] = avg;
        }
    }

    // Double-centering: B = -1/2 J D2 J, J = I - 11ᵀ/L. Compute row means (=
    // column means, D2 symmetric) and the grand mean, then
    // B_ij = -1/2 (D2_ij - rowmean_i - rowmean_j + grand).
    let mut row_mean = vec![0.0_f64; l];
    let mut grand = 0.0_f64;
    for a in 0..l {
        let mut s = 0.0;
        for b in 0..l {
            s += d2[[a, b]];
        }
        row_mean[a] = s / l as f64;
        grand += s;
    }
    grand /= (l * l) as f64;
    let mut b_mat = Array2::<f64>::zeros((l, l));
    for a in 0..l {
        for b in 0..l {
            b_mat[[a, b]] = -0.5 * (d2[[a, b]] - row_mean[a] - row_mean[b] + grand);
        }
    }
    // `B` is symmetric in exact arithmetic, but `B_ab` and `B_ba` are accumulated
    // through independent float operations, leaving a ~1e-14 asymmetry that grows
    // with the data scale. Average each off-diagonal pair so the matrix is
    // BIT-symmetric before the self-adjoint eigensolve; the solver reads only the
    // lower triangle, so this also fixes which triangle it trusts.
    for a in 0..l {
        for b in (a + 1)..l {
            let avg = 0.5 * (b_mat[[a, b]] + b_mat[[b, a]]);
            b_mat[[a, b]] = avg;
            b_mat[[b, a]] = avg;
        }
    }

    let (evals, evecs) = b_mat
        .eigh(Side::Lower)
        .map_err(|err| format!("intrinsic_seed: MDS eigensolve failed: {err:?}"))?;
    // Select the top d POSITIVE eigenvalues (descending). faer returns ascending;
    // sort an index list by descending eigenvalue with an index tie-break.
    let leading = evals.iter().cloned().fold(0.0_f64, f64::max);
    if !(leading > 0.0) {
        return Ok(out); // degenerate (all rows coincide): zero embedding
    }
    // An axis counts only when it clears both the eigensolver's backward error
    // `l·ε·λ_max` and the metric's measured non-embeddability. Classical MDS on a
    // geodesic distance that is not exactly Euclidean spreads spurious eigenvalues
    // of both signs, and the most negative one reads their magnitude off this
    // spectrum. Such axes carry no coordinate signal and their `1/√λ` extension
    // weight blows up.
    let most_negative = evals.iter().cloned().fold(0.0_f64, f64::min);
    let floor = (evals.len() as f64 * f64::EPSILON * leading).max(-most_negative);
    let mut order: Vec<usize> = (0..evals.len()).collect();
    order.sort_by(|&i, &j| evals[j].total_cmp(&evals[i]).then_with(|| i.cmp(&j)));
    let axes: Vec<usize> = order
        .into_iter()
        .filter(|&c| evals[c] > floor)
        .take(d)
        .collect();

    // L-Isomap distance-based extension of every row. For row `r` with squared
    // geodesic distances δ_r to the landmarks,
    //   x_r[c] = -1/(2√λ_c) · Σ_a evec[a,c] · (δ_r[a] - rowmean_a).
    // For a landmark row this reproduces its MDS coordinate √λ_c·evec[·,c] exactly
    // (the eigenvectors are orthogonal to the constant vector B annihilates).
    for (col, &c) in axes.iter().enumerate() {
        let inv = -0.5 / evals[c].sqrt();
        for r in 0..n {
            let mut acc = 0.0_f64;
            for a in 0..l {
                let g = geo[[a, r]];
                let delta = g * g;
                acc += evecs[[a, c]] * (delta - row_mean[a]);
            }
            out[[r, col]] = inv * acc;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Classical MDS on the EXACT Euclidean distance matrix of a known planar
    /// configuration must reproduce that configuration up to a rigid motion. We
    /// check the recovered pairwise distances match the originals (rigid-motion
    /// invariant), which is the defining property of a correct MDS.
    ///
    /// The configuration is small enough (`n = 6 ≤ k + 1`) that the deterministic
    /// kNN graph is COMPLETE — every pair is a direct edge, so the graph geodesic
    /// equals the exact Euclidean distance and MDS recovers the plane to rounding.
    /// (A sparse grid graph would inject the Isomap "staircase" over-estimate of
    /// diagonal distances, which is a property of the discrete graph metric, not
    /// an MDS error — so this test isolates the MDS/extension math.)
    #[test]
    fn mds_recovers_known_configuration_up_to_rigid_motion() {
        // Six deterministic 2-D points (a 2×3 lattice) embedded in 3-D ambient
        // with a constant third coordinate. n = 6 and k = max(2d+1, ⌈log2 n⌉) = 5
        // ⇒ each node links the other five ⇒ complete graph ⇒ exact Euclidean
        // geodesics.
        let n = 6usize;
        let mut z = Array2::<f64>::zeros((n, 3));
        for i in 0..2 {
            for j in 0..3 {
                let r = i * 3 + j;
                z[[r, 0]] = i as f64;
                z[[r, 1]] = j as f64;
                z[[r, 2]] = 0.5; // constant ambient offset
            }
        }
        let embed = intrinsic_geodesic_embedding(z.view(), 2).unwrap();
        // Every embedded pairwise distance must match the ambient Euclidean
        // distance (rigid-motion invariant) to rounding.
        let mut max_rel = 0.0_f64;
        for a in 0..n {
            for b in (a + 1)..n {
                let amb = super::squared_distance(z.view(), a, b).sqrt();
                let mut e2 = 0.0;
                for c in 0..2 {
                    let diff = embed[[a, c]] - embed[[b, c]];
                    e2 += diff * diff;
                }
                let emb = e2.sqrt();
                if amb > 1e-9 {
                    max_rel = max_rel.max(((emb - amb) / amb).abs());
                }
            }
        }
        assert!(
            max_rel < 1e-6,
            "MDS on a complete-graph (exact-Euclidean) configuration must reproduce \
             its pairwise distances to rounding (max relative error {max_rel:.3e})"
        );
    }

    #[test]
    fn two_row_embedding_preserves_full_ambient_distance() {
        let z = Array2::from_shape_vec((2, 3), vec![4.0, -2.0, 1.0, 4.0, 4.0, 9.0]).unwrap();
        let embed = intrinsic_geodesic_embedding(z.view(), 1).unwrap();
        assert_eq!(embed[[0, 0]], -5.0);
        assert_eq!(embed[[1, 0]], 5.0);
        assert_eq!(embed[[0, 0]] + embed[[1, 0]], 0.0);
    }

    /// The Borůvka bridging must produce exactly the graph of the one-edge-per-scan
    /// reference (full sort, then add the global-minimum cross-component edge until
    /// connected). The data are well-separated clusters with a small `k`, so the
    /// kNN graph starts with many components.
    #[test]
    fn knn_graph_bridging_matches_one_edge_per_scan_reference() {
        fn reference(z: ArrayView2<'_, f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
            let n = z.nrows();
            let k = k.min(n.saturating_sub(1)).max(1);
            let mut edges = std::collections::BTreeMap::new();
            for i in 0..n {
                let mut dists: Vec<(f64, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (squared_distance(z, i, j), j))
                    .collect();
                dists.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                for &(dist2, j) in dists.iter().take(k) {
                    edges
                        .entry((i.min(j), i.max(j)))
                        .or_insert_with(|| dist2.sqrt());
                }
            }
            let mut comp: Vec<usize> = (0..n).collect();
            let relabel = |comp: &mut Vec<usize>, from: usize, to: usize| {
                for c in comp.iter_mut() {
                    if *c == from {
                        *c = to;
                    }
                }
            };
            for &(a, b) in edges.keys() {
                let (ca, cb) = (comp[a], comp[b]);
                if ca != cb {
                    relabel(&mut comp, ca.max(cb), ca.min(cb));
                }
            }
            loop {
                let mut best: Option<(f64, usize, usize)> = None;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if comp[i] == comp[j] {
                            continue;
                        }
                        let d2 = squared_distance(z, i, j);
                        if best.is_none_or(|(bd, _, _)| d2 < bd) {
                            best = Some((d2, i, j));
                        }
                    }
                }
                let Some((d2, i, j)) = best else { break };
                edges.insert((i, j), d2.sqrt());
                let (ci, cj) = (comp[i], comp[j]);
                relabel(&mut comp, ci.max(cj), ci.min(cj));
            }
            let mut adj = vec![Vec::new(); n];
            for (&(a, b), &w) in &edges {
                adj[a].push((b, w));
                adj[b].push((a, w));
            }
            adj
        }

        // Nine tight 4-point clusters on an irregular 3-D layout: k = 2 keeps
        // every cluster its own component, so bridging joins nine components.
        let centers = [
            [0.0, 0.0, 0.0],
            [7.0, 1.0, 0.5],
            [2.5, 9.0, -1.0],
            [-6.0, 3.5, 2.0],
            [11.0, 8.0, 3.0],
            [-3.0, -8.0, 1.5],
            [5.0, -4.5, -6.0],
            [-9.5, -2.0, -4.0],
            [1.0, 4.0, 10.0],
        ];
        let offsets = [
            [0.0, 0.0, 0.0],
            [0.11, 0.0, 0.02],
            [0.0, 0.13, 0.05],
            [0.07, 0.09, 0.17],
        ];
        let mut z = Array2::<f64>::zeros((centers.len() * offsets.len(), 3));
        for (ci, c) in centers.iter().enumerate() {
            for (oi, o) in offsets.iter().enumerate() {
                for axis in 0..3 {
                    z[[ci * offsets.len() + oi, axis]] = c[axis] + o[axis];
                }
            }
        }
        for k in 1..=4 {
            assert_eq!(
                deterministic_knn_graph(z.view(), k),
                reference(z.view(), k),
                "bridged kNN graph must equal the one-edge-per-scan reference at k = {k}"
            );
        }
        // A single row has no neighbour to keep and nothing to bridge.
        assert_eq!(
            deterministic_knn_graph(z.slice(ndarray::s![0..1, ..]), 3),
            vec![Vec::new()]
        );
    }

    /// Determinism doctrine: the embedding is bit-identical run-to-run.
    #[test]
    fn intrinsic_embedding_is_bit_identical_run_to_run() {
        let n = 40usize;
        let mut z = Array2::<f64>::zeros((n, 4));
        for r in 0..n {
            let t = r as f64 * 0.3;
            z[[r, 0]] = t.sin();
            z[[r, 1]] = t.cos();
            z[[r, 2]] = (0.5 * t).sin();
            z[[r, 3]] = 0.2 * t;
        }
        let a = intrinsic_geodesic_embedding(z.view(), 2).unwrap();
        let b = intrinsic_geodesic_embedding(z.view(), 2).unwrap();
        assert_eq!(a, b, "intrinsic embedding must be bit-identical run-to-run");
    }
}
