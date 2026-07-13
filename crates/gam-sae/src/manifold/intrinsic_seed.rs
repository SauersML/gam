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
//! The output of [`sae_intrinsic_seed_initial_coords`] has the SAME
//! `(K_atoms, n_obs, d_max)` shape and per-chart-kind conventions as
//! [`super::sae_pca_seed_initial_coords`], so any chart family consumes it
//! unchanged; on a fold it unrolls where PCA creases, and on a non-fold it is a
//! near-isometric embedding that ties the linear seed. The seed RACE (born-atom
//! topology adjudication in `structure_harvest.rs`) keeps PCA as the cheaper
//! default and only adopts the intrinsic seed when it earns higher REML evidence.
//!
//! Determinism is fleet law: no RNG anywhere. kNN ties break by index, landmarks
//! by first-wins-lowest-index, Dijkstra by (distance, node) so equal-cost frontier
//! nodes pop in index order, and the eigendecomposition is faer's deterministic
//! self-adjoint solver. Same input ⇒ bit-identical coordinates run-to-run.

use super::SaeAtomBasisKind;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array2, Array3, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use faer::Side;

/// An MDS eigenvalue counts as a genuine embedding axis only when it is POSITIVE
/// and clears this fraction of the leading (largest) eigenvalue. Classical MDS on
/// a geodesic (non-Euclidean) distance matrix produces small negative and
/// near-zero eigenvalues from the metric's slight non-embeddability; those axes
/// carry no coordinate signal and their `1/√λ` extension weight blows up, so they
/// are dropped rather than fed a vanishing denominator. `1e-9` sits ~7 orders
/// above the f64 eigensolver noise on a normalized Gram and ~all the way below any
/// axis carrying real spread.
const MDS_EIGENVALUE_FLOOR_FRAC: f64 = 1.0e-9;

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
/// Bridging is deterministic: while more than one component remains, the globally
/// shortest edge joining two distinct components (ties by `(min_row, max_row)`) is
/// added. This restores connectivity using the true nearest cross-component pair,
/// so a graph that is already connected is returned untouched.
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
        dists.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for &(dist2, j) in dists.iter().take(k) {
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
    loop {
        // Best bridging edge per unordered component pair does not need tracking;
        // a single global-minimum inter-component edge per pass suffices and is
        // deterministic (ties by (i, j)). Passes ≤ #components − 1.
        let mut best: Option<(f64, usize, usize)> = None;
        for i in 0..n {
            let ri = find(&mut parent, i);
            for j in (i + 1)..n {
                if find(&mut parent, j) == ri {
                    continue;
                }
                let d2 = squared_distance(z, i, j);
                let better = match best {
                    None => true,
                    Some((bd, _, _)) => d2 < bd,
                };
                if better {
                    best = Some((d2, i, j));
                }
            }
        }
        match best {
            None => break, // single component
            Some((d2, i, j)) => {
                edges.insert((i, j), d2.sqrt());
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
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
/// from the chosen set, first-wins on ties. The same coverage-maximizing greedy
/// pattern as `encode::data_driven_chart_centers`, reproducible run-to-run.
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
    if n < 3 {
        // Too few rows for a geodesic graph to differ from the ambient metric.
        // Fall back to a mean-centered raw-coordinate embedding (leading columns),
        // which for n ≤ 2 is already the exact MDS solution.
        for row in 0..n {
            for col in 0..d.min(z.ncols()) {
                out[[row, col]] = z[[row, col]];
            }
        }
        return Ok(out);
    }
    let k = intrinsic_seed_knn(n, d).min(n - 1);
    let adj = deterministic_knn_graph(z, k);
    let l_count = intrinsic_landmark_count(n, d);
    let landmarks = farthest_point_landmarks(z, l_count);
    let l = landmarks.len();
    if l < 2 {
        return Ok(out);
    }
    let geo = landmark_geodesics(&adj, &landmarks);

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
    let floor = leading * MDS_EIGENVALUE_FLOOR_FRAC;
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

fn min_max_normalize_into(out: &mut Array3<f64>, atom_idx: usize, axis: usize, values: &[f64]) {
    let (lo, hi) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let span = hi - lo;
    if span > 0.0 && span.is_finite() {
        for (row, &v) in values.iter().enumerate() {
            out[[atom_idx, row, axis]] = (v - lo) / span - 0.5;
        }
    }
}

/// Number of independent intrinsic embedding functions needed to seed a chart.
///
/// This is deliberately distinct from the chart's latent dimension. A periodic
/// coordinate is the phase of a two-function plane, a sphere's two coordinates
/// need a three-function frame, and every torus coordinate needs its own
/// two-function phase plane. Using only `latent_dim` embedding axes silently
/// collapsed sphere latitude and every torus axis after the first to zero.
fn intrinsic_chart_embedding_axes(kind: &SaeAtomBasisKind, latent_dim: usize) -> usize {
    match kind {
        SaeAtomBasisKind::Periodic => 2 + latent_dim.max(1).saturating_sub(1),
        SaeAtomBasisKind::Sphere => 3,
        SaeAtomBasisKind::Torus => 2 * latent_dim.max(1),
        _ => latent_dim.max(1),
    }
}

/// Intrinsic-metric seed with the SAME `(K_atoms, n_obs, d_max)` output contract
/// and per-chart-kind conventions as [`super::sae_pca_seed_initial_coords`], built
/// from a single geodesic embedding of `z` wide enough for every atom's chart
/// functions. The returned coordinate tensor still uses the maximum LATENT
/// dimension, matching the PCA seed contract. Each
/// atom reads its chart off the leading intrinsic axes:
///   * flat (Euclidean/Linear/other) — leading `d` axes, min-max normalized to
///     `[-0.5, 0.5]` (the flat PCA convention);
///   * periodic — `[0, 1)` phase off axes 0/1 via `atan2`, extra axes min-max;
///   * sphere — `(lat, lon)` off the unit-normalized leading 3 axes;
///   * torus — `[0, 1)` phase per axis off disjoint intrinsic-axis pairs.
///
/// A drop-in for the PCA seed: same shape, finite, ready for any chart family. Its
/// value is on folded manifolds, where the geodesic axes unroll a sheet the linear
/// PCA seed would crease.
pub fn sae_intrinsic_seed_initial_coords(
    z: ArrayView2<'_, f64>,
    basis_kinds: &[SaeAtomBasisKind],
    atom_dim: &[usize],
) -> Result<Array3<f64>, String> {
    let k_atoms = basis_kinds.len();
    let (n_obs, _p) = z.dim();
    let latent_d_max = atom_dim.iter().copied().max().unwrap_or(1).max(1);
    let embedding_axes = basis_kinds
        .iter()
        .zip(atom_dim.iter().copied())
        .map(|(kind, d)| intrinsic_chart_embedding_axes(kind, d))
        .max()
        .unwrap_or(1);
    let mut out = Array3::<f64>::zeros((k_atoms, n_obs, latent_d_max));
    if n_obs == 0 || z.ncols() == 0 || k_atoms == 0 {
        return Ok(out);
    }
    let embed = intrinsic_geodesic_embedding(z, embedding_axes)?;
    let two_pi = std::f64::consts::TAU;
    for atom_idx in 0..k_atoms {
        let d = atom_dim[atom_idx];
        if d == 0 {
            continue;
        }
        match &basis_kinds[atom_idx] {
            SaeAtomBasisKind::Periodic => {
                if embed.ncols() >= 2 {
                    for row in 0..n_obs {
                        let phase = embed[[row, 1]].atan2(embed[[row, 0]]) / two_pi;
                        out[[atom_idx, row, 0]] = phase - phase.floor();
                    }
                } else {
                    let col: Vec<f64> = (0..n_obs).map(|r| embed[[r, 0]]).collect();
                    // Single-axis fallback: min-max the lone geodesic axis to a
                    // phase in [0, 1).
                    let (lo, hi) = col
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                            (lo.min(v), hi.max(v))
                        });
                    let span = hi - lo;
                    if span > 0.0 {
                        for (row, &v) in col.iter().enumerate() {
                            out[[atom_idx, row, 0]] = (v - lo) / span;
                        }
                    }
                }
                for axis in 1..d {
                    if axis + 1 >= embed.ncols() {
                        break;
                    }
                    let vals: Vec<f64> = (0..n_obs).map(|r| embed[[r, axis + 1]]).collect();
                    min_max_normalize_into(&mut out, atom_idx, axis, &vals);
                }
            }
            SaeAtomBasisKind::Sphere => {
                for row in 0..n_obs {
                    let x = embed[[row, 0]];
                    let y = embed[[row, 1]];
                    let zz = embed[[row, 2]];
                    let norm = (x * x + y * y + zz * zz).sqrt().max(1.0e-24);
                    if d >= 1 {
                        out[[atom_idx, row, 0]] = (zz / norm).clamp(-1.0, 1.0).asin();
                    }
                    if d >= 2 {
                        out[[atom_idx, row, 1]] = y.atan2(x);
                    }
                }
            }
            SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    let (ca, cb) = (2 * axis, 2 * axis + 1);
                    if cb >= embed.ncols() {
                        break;
                    }
                    for row in 0..n_obs {
                        let phase = embed[[row, cb]].atan2(embed[[row, ca]]) / two_pi;
                        out[[atom_idx, row, axis]] = phase - phase.floor();
                    }
                }
            }
            _ => {
                for axis in 0..d {
                    if axis >= embed.ncols() {
                        break;
                    }
                    let vals: Vec<f64> = (0..n_obs).map(|r| embed[[r, axis]]).collect();
                    min_max_normalize_into(&mut out, atom_idx, axis, &vals);
                }
            }
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

    /// The Array3 contract wrapper matches the PCA seed's shape and produces
    /// finite coordinates for every chart kind.
    #[test]
    fn intrinsic_seed_matches_pca_contract_shape_and_finite() {
        let n = 30usize;
        let p = 5usize;
        let mut z = Array2::<f64>::zeros((n, p));
        for r in 0..n {
            for c in 0..p {
                z[[r, c]] = ((r * 3 + c) as f64 * 0.21).sin() + 0.1 * (r as f64 - c as f64);
            }
        }
        for (kind, d) in [
            (SaeAtomBasisKind::Linear, 2usize),
            (SaeAtomBasisKind::Periodic, 1usize),
            (SaeAtomBasisKind::Sphere, 2usize),
            (SaeAtomBasisKind::Torus, 2usize),
        ] {
            let k = 3;
            let kinds = vec![kind.clone(); k];
            let dims = vec![d; k];
            let seed = sae_intrinsic_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
            assert_eq!(seed.dim(), (k, n, d));
            for v in seed.iter() {
                assert!(
                    v.is_finite(),
                    "{kind:?}: non-finite intrinsic seed coord {v}"
                );
            }
        }
    }

    /// Chart coordinates and embedding functions are not the same dimension:
    /// sphere latitude needs a 3-frame, and a 2-torus needs two independent
    /// phase planes. Pin the production allocation rule so those coordinates
    /// cannot silently collapse back to zero.
    #[test]
    fn intrinsic_seed_allocates_every_chart_function_2240() {
        assert_eq!(
            intrinsic_chart_embedding_axes(&SaeAtomBasisKind::Periodic, 1),
            2
        );
        assert_eq!(
            intrinsic_chart_embedding_axes(&SaeAtomBasisKind::Periodic, 3),
            4
        );
        assert_eq!(
            intrinsic_chart_embedding_axes(&SaeAtomBasisKind::Sphere, 2),
            3
        );
        assert_eq!(
            intrinsic_chart_embedding_axes(&SaeAtomBasisKind::Torus, 2),
            4
        );
        assert_eq!(
            intrinsic_chart_embedding_axes(&SaeAtomBasisKind::Linear, 2),
            2
        );
    }
}
