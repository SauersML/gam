//! Spectral seeding for latent-coordinate (GP-LVM-style) models.
//!
//! Fitting a per-row latent coordinate `t_n` against a decoder `Ŷ_n = Φ(t_n)β`
//! is a non-convex problem: the inner `β | t` solve can only fit a `t` that is
//! already ordered along the data manifold, so a cold random start leaves the
//! outer optimizer with almost no gradient signal to *sort* the rows — it
//! settles in a poor local optimum (see issue #627, where a random init returns
//! R²≈0 while a near-correct init returns R²≈1).
//!
//! Laplacian eigenmaps supplies a principled warm start. It embeds the rows by
//! the leading non-trivial eigenvectors of the k-nearest-neighbour graph
//! Laplacian of the *responses*, recovering the intrinsic manifold coordinate up
//! to the model's own monotone/rotation gauge. Refining that seed with the
//! Riemannian outer optimizer then converges to the global configuration instead
//! of a random local one.
//!
//! Every stage keeps memory linear in the number of rows: the neighbour scan
//! keeps each row's `k` best, the graph is held as adjacency lists, and the modes
//! come from a block eigensolver that only applies the graph Laplacian. No stage
//! holds an `n × n` matrix, so the seed runs one method at every `n` (SPEC rules 6
//! and 10).

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::utils::splitmix64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Zip, concatenate, s};

/// Laplacian-eigenmaps embedding of `features` (`n × q`) into `latent_dim`
/// intrinsic coordinates, each axis affinely rescaled to `[0, 1]`.
///
/// The construction is the standard normalized Laplacian eigenmap:
///
/// 1. symmetric k-nearest-neighbour graph with Gaussian affinities
///    `w_ij = exp(-‖y_i − y_j‖² / ε)`, bandwidth `ε` = median squared k-NN
///    distance (a scale-free, data-driven choice);
/// 2. normalized Laplacian `L = I − D^{-1/2} W D^{-1/2}`;
/// 3. the `latent_dim` eigenvectors with the smallest *non-zero* eigenvalues
///    (skipping the trivial constant mode), mapped back through `D^{-1/2}` so
///    they are the generalized eigenvectors of `L v = λ D v`.
///
/// The returned coordinate recovers the manifold parameterization up to a
/// monotone reparameterization / axis rotation — exactly the gauge a
/// latent-coordinate decoder is free in — which is why it is a reliable seed for
/// the outer optimizer rather than a final answer.
///
/// `n_neighbors` is clamped to `[1, n − 1]`. Errors only on structurally
/// impossible requests (too few rows to expose `latent_dim` non-trivial modes,
/// non-finite inputs, or an eigensolver failure).
pub fn laplacian_eigenmap_coords(
    features: ArrayView2<'_, f64>,
    latent_dim: usize,
    n_neighbors: usize,
) -> Result<Array2<f64>, String> {
    let n = features.nrows();
    if latent_dim == 0 {
        return Err("laplacian_eigenmap_coords: latent_dim must be >= 1".to_string());
    }
    // Need the trivial mode plus `latent_dim` non-trivial modes.
    if n < latent_dim + 2 {
        return Err(format!(
            "laplacian_eigenmap_coords: need at least latent_dim + 2 = {} rows to expose \
             {latent_dim} non-trivial eigenvectors; got {n}",
            latent_dim + 2
        ));
    }
    if features.iter().any(|v| !v.is_finite()) {
        return Err("laplacian_eigenmap_coords: features contain non-finite values".to_string());
    }
    let k = n_neighbors.clamp(1, n - 1);

    // Per-row k nearest neighbours (excluding self) and the bandwidth: the
    // median of the retained k-NN squared distances. Median is robust to the
    // far tail and keeps the affinities well-scaled across datasets.
    let (knn, knn_d2) = nearest_neighbours(features, k);
    let mut retained: Vec<f64> = knn_d2.iter().copied().collect();
    let middle = retained.len() / 2;
    let (_, median, _) = retained.select_nth_unstable_by(middle, f64::total_cmp);
    let epsilon = median.max(f64::MIN_POSITIVE);

    let graph = NormalizedAffinity::from_neighbours(&knn, &knn_d2, epsilon);
    // Ascending modes of the normalized Laplacian. Column 0 is the smallest (the
    // trivial mode ∝ D^{1/2}·1 on a connected graph) and is skipped below.
    let modes = smallest_eigenvectors(&graph, latent_dim + 1)?;

    let mut coords = Array2::<f64>::zeros((n, latent_dim));
    for a in 0..latent_dim {
        for i in 0..n {
            // Generalized eigenvector y = D^{-1/2} v of L v = λ D v.
            coords[[i, a]] = graph.dinv_sqrt[i] * modes[[i, a + 1]];
        }
        // Affinely rescale this axis to [0, 1]; a degenerate (constant) axis
        // collapses to 0, which the outer optimizer then moves off.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..n {
            lo = lo.min(coords[[i, a]]);
            hi = hi.max(coords[[i, a]]);
        }
        let span = hi - lo;
        if span > 0.0 && span.is_finite() {
            for i in 0..n {
                coords[[i, a]] = (coords[[i, a]] - lo) / span;
            }
        } else {
            for i in 0..n {
                coords[[i, a]] = 0.0;
            }
        }
    }
    Ok(coords)
}

/// Each row's `k` nearest other rows by exact squared distance (ties to the
/// lower row index), with those distances. A row keeps only its `k` best, so the
/// scan holds `O(n·k)` results plus one `O(n)` workspace per worker, never the
/// `n × n` distance matrix.
fn nearest_neighbours(features: ArrayView2<'_, f64>, k: usize) -> (Array2<usize>, Array2<f64>) {
    let n = features.nrows();
    let mut knn = Array2::<usize>::zeros((n, k));
    let mut knn_d2 = Array2::<f64>::zeros((n, k));
    Zip::indexed(knn.rows_mut())
        .and(knn_d2.rows_mut())
        .par_for_each(|i, mut neighbour_row, mut distance_row| {
            let row = features.row(i);
            let mut candidates: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let squared: f64 = row
                        .iter()
                        .zip(features.row(j))
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum();
                    (squared, j)
                })
                .collect();
            let closer =
                |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
            if k < candidates.len() {
                candidates.select_nth_unstable_by(k, closer);
                candidates.truncate(k);
            }
            candidates.sort_unstable_by(closer);
            for (t, (squared, j)) in candidates.into_iter().enumerate() {
                neighbour_row[t] = j;
                distance_row[t] = squared;
            }
        });
    (knn, knn_d2)
}

/// `S = D^{-1/2} W D^{-1/2}` of the symmetric k-NN affinity graph, held as
/// adjacency lists: `O(n·k)` storage whatever `n` is.
struct NormalizedAffinity {
    row_start: Vec<usize>,
    neighbours: Vec<usize>,
    weights: Vec<f64>,
    dinv_sqrt: Array1<f64>,
}

impl NormalizedAffinity {
    /// The union of directed k-NN edges with `w_ij = exp(−d²_ij / ε)`.
    fn from_neighbours(knn: &Array2<usize>, knn_d2: &Array2<f64>, epsilon: f64) -> Self {
        let n = knn.nrows();
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(2 * knn.len());
        for ((i, t), &j) in knn.indexed_iter() {
            let affinity = (-knn_d2[[i, t]] / epsilon).exp();
            edges.push((i, j, affinity));
            edges.push((j, i, affinity));
        }
        // A mutual pair arrives twice with bitwise-equal affinity (the squared
        // distance is symmetric in the pair), so keeping one copy loses nothing.
        edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        edges.dedup_by(|later, kept| later.0 == kept.0 && later.1 == kept.1);
        let mut row_start = vec![0usize; n + 1];
        for edge in &edges {
            row_start[edge.0 + 1] += 1;
        }
        for i in 0..n {
            row_start[i + 1] += row_start[i];
        }
        let neighbours = edges.iter().map(|edge| edge.1).collect();
        let weights: Vec<f64> = edges.iter().map(|edge| edge.2).collect();
        // An isolated node (zero degree) gets a tiny floor so D^{-1/2} stays
        // finite; its row of L is then the identity and it contributes no
        // spurious coupling.
        let dinv_sqrt = (0..n)
            .map(|i| {
                let degree: f64 = weights[row_start[i]..row_start[i + 1]].iter().sum();
                1.0 / degree.max(f64::MIN_POSITIVE).sqrt()
            })
            .collect();
        Self {
            row_start,
            neighbours,
            weights,
            dinv_sqrt,
        }
    }

    /// `out = A·x` for `A = (I − S)/2`, half the normalized Laplacian, whose
    /// spectrum lies in `[0, 1]`.
    fn apply(&self, x: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        Zip::indexed(out).par_for_each(|i, slot| {
            let mut coupled = 0.0;
            for p in self.row_start[i]..self.row_start[i + 1] {
                let j = self.neighbours[p];
                coupled += self.weights[p] * self.dinv_sqrt[j] * x[j];
            }
            *slot = 0.5 * (x[i] - self.dinv_sqrt[i] * coupled);
        });
    }

    fn apply_block(&self, block: &Array2<f64>) -> Array2<f64> {
        let mut image = Array2::<f64>::zeros(block.raw_dim());
        for (column, target) in block.columns().into_iter().zip(image.columns_mut()) {
            self.apply(column, target);
        }
        image
    }
}

/// Orthonormal columns spanning `candidates` beyond `basis`, whose columns are
/// already orthonormal. Each candidate gets two Gram–Schmidt passes; one whose
/// remainder falls below √ε of its own norm already lies in the span numerically
/// and is dropped.
fn orthonormal_extension(basis: &Array2<f64>, candidates: &Array2<f64>) -> Array2<f64> {
    let mut accepted: Vec<Array1<f64>> = Vec::with_capacity(candidates.ncols());
    for candidate in candidates.columns() {
        let norm = candidate.dot(&candidate).sqrt();
        if !(norm > 0.0) {
            continue;
        }
        let mut v = candidate.mapv(|value| value / norm);
        for _pass in 0..2 {
            for column in basis.columns() {
                let projection = column.dot(&v);
                v.scaled_add(-projection, &column);
            }
            for column in &accepted {
                let projection = column.dot(&v);
                v.scaled_add(-projection, column);
            }
        }
        let remainder = v.dot(&v).sqrt();
        if remainder > f64::EPSILON.sqrt() {
            v.mapv_inplace(|value| value / remainder);
            accepted.push(v);
        }
    }
    let mut extension = Array2::<f64>::zeros((basis.nrows(), accepted.len()));
    for (mut target, column) in extension.columns_mut().into_iter().zip(&accepted) {
        target.assign(column);
    }
    extension
}

/// A pseudo-random start block from one SplitMix64 stream, so the seed needs no
/// RNG from the caller and repeats bit for bit.
fn deterministic_start_block(n: usize, block: usize) -> Array2<f64> {
    let mut state = 0u64;
    Array2::from_shape_simple_fn((n, block), || {
        // The top 53 bits as a uniform on [0, 1), centred on zero.
        (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

/// The `wanted` smallest eigenvectors of `A = (I − S)/2`, ascending, by LOBPCG
/// over `span[X, R, P]` with one guard vector beyond the wanted block.
///
/// It holds `O(n·block)` vectors and applies `A` only through the graph, so its
/// memory is linear in `n` at every size. The Rayleigh–Ritz step over the whole
/// search space keeps a wanted block that ends inside a degenerate eigenspace
/// converging. It stops when every wanted residual `‖A x − θ x‖` meets
/// `(n + 1)·ε`, the backward error a dense symmetric eigensolver guarantees at
/// this dimension and so the accuracy the dense seed had, or when no wanted Ritz
/// value moves by more than roundoff in an iteration: in exact arithmetic the
/// Ritz values strictly decrease until convergence, so the subspace is then as
/// converged as the arithmetic resolves.
fn smallest_eigenvectors(
    graph: &NormalizedAffinity,
    wanted: usize,
) -> Result<Array2<f64>, String> {
    let n = graph.dinv_sqrt.len();
    let block = (wanted + 1).min(n);
    let residual_bound = (n as f64 + 1.0) * f64::EPSILON;
    let mut x = orthonormal_extension(
        &Array2::<f64>::zeros((n, 0)),
        &deterministic_start_block(n, block),
    );
    if x.ncols() < block {
        return Err(format!(
            "laplacian_eigenmap_coords: the start block spans {} of {block} directions",
            x.ncols()
        ));
    }
    let mut ax = graph.apply_block(&x);
    let mut search = x.clone();
    let mut a_search = ax.clone();
    let mut previous_ritz: Option<Array1<f64>> = None;
    loop {
        let projected = search.t().dot(&a_search);
        let projected = (&projected + &projected.t()) * 0.5;
        let (values, vectors) = projected.eigh(Side::Lower).map_err(|e| {
            format!("laplacian_eigenmap_coords: Rayleigh-Ritz eigendecomposition failed: {e}")
        })?;
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let mut coefficients = Array2::<f64>::zeros((search.ncols(), block));
        for (target, &source) in order.iter().take(block).enumerate() {
            coefficients
                .column_mut(target)
                .assign(&vectors.column(source));
        }
        let ritz: Array1<f64> = order
            .iter()
            .take(block)
            .map(|&source| values[source])
            .collect();
        // The part of the new block drawn from the residuals and the previous
        // directions is LOBPCG's next search direction.
        let kept = x.ncols();
        let directions = if search.ncols() > kept {
            search
                .slice(s![.., kept..])
                .dot(&coefficients.slice(s![kept.., ..]))
        } else {
            Array2::<f64>::zeros((n, 0))
        };
        x = search.dot(&coefficients);
        ax = a_search.dot(&coefficients);
        let residual = &ax - &(&x * &ritz);
        let worst = residual
            .columns()
            .into_iter()
            .take(wanted)
            .map(|column| column.dot(&column).sqrt())
            .fold(0.0_f64, f64::max);
        let stationary = previous_ritz.as_ref().is_some_and(|last| {
            (0..wanted).all(|j| (last[j] - ritz[j]).abs() <= f64::EPSILON * (1.0 + ritz[j].abs()))
        });
        let extension = if worst <= residual_bound || stationary {
            Array2::<f64>::zeros((n, 0))
        } else {
            let candidates = concatenate(Axis(1), &[residual.view(), directions.view()])
                .expect("residuals and directions share the row count");
            orthonormal_extension(&x, &candidates)
        };
        if extension.ncols() == 0 {
            return Ok(x.slice(s![.., ..wanted]).to_owned());
        }
        previous_ritz = Some(ritz);
        let a_extension = graph.apply_block(&extension);
        search = concatenate(Axis(1), &[x.view(), extension.view()])
            .expect("the Ritz block and its extension share the row count");
        a_search = concatenate(Axis(1), &[ax.view(), a_extension.view()])
            .expect("the images of the Ritz block and its extension share the row count");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// The dense pipeline the seed used to run, kept as the oracle: dense
    /// pairwise distances, per-row sort, the union k-NN graph, the dense
    /// normalized Laplacian and a full eigendecomposition. Returns the
    /// generalized eigenvectors `D^{-1/2} v` of modes `1..=modes`.
    fn dense_eigenmap_modes(
        features: ArrayView2<'_, f64>,
        modes: usize,
        n_neighbors: usize,
    ) -> Array2<f64> {
        let n = features.nrows();
        let k = n_neighbors.clamp(1, n - 1);
        let d2 = Array2::from_shape_fn((n, n), |(i, j)| {
            features
                .row(i)
                .iter()
                .zip(features.row(j))
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        });
        let mut edges = Vec::with_capacity(n * k);
        let mut retained = Vec::with_capacity(n * k);
        for i in 0..n {
            let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            order.sort_by(|&a, &b| d2[[i, a]].total_cmp(&d2[[i, b]]).then(a.cmp(&b)));
            for &j in order.iter().take(k) {
                edges.push((i, j));
                retained.push(d2[[i, j]]);
            }
        }
        retained.sort_by(f64::total_cmp);
        let epsilon = retained[retained.len() / 2].max(f64::MIN_POSITIVE);
        let mut w = Array2::<f64>::zeros((n, n));
        for &(i, j) in &edges {
            let affinity = (-d2[[i, j]] / epsilon).exp();
            w[[i, j]] = affinity;
            w[[j, i]] = affinity;
        }
        let dinv: Array1<f64> = w
            .rows()
            .into_iter()
            .map(|row| 1.0 / row.sum().max(f64::MIN_POSITIVE).sqrt())
            .collect();
        let lap = Array2::from_shape_fn((n, n), |(i, j)| {
            let norm = dinv[i] * dinv[j] * w[[i, j]];
            if i == j { 1.0 - norm } else { -norm }
        });
        let (values, vectors) = lap.eigh(Side::Lower).unwrap();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        Array2::from_shape_fn((n, modes), |(i, a)| dinv[i] * vectors[[i, order[a + 1]]])
    }

    /// Share of the variance of `target` explained by the best affine fit from
    /// `predictors` (R²), by projection onto the orthonormalized centred
    /// predictors.
    fn explained_fraction(target: ArrayView1<'_, f64>, predictors: &[ArrayView1<'_, f64>]) -> f64 {
        let n = target.len() as f64;
        let mean = target.sum() / n;
        let centred = target.mapv(|value| value - mean);
        let mut basis: Vec<Array1<f64>> = Vec::new();
        for predictor in predictors {
            let predictor_mean = predictor.sum() / n;
            let mut v = predictor.mapv(|value| value - predictor_mean);
            for _pass in 0..2 {
                for column in &basis {
                    let projection = column.dot(&v);
                    v.scaled_add(-projection, column);
                }
            }
            let norm = v.dot(&v).sqrt();
            if norm > 0.0 {
                basis.push(v / norm);
            }
        }
        let captured: f64 = basis
            .iter()
            .map(|column| column.dot(&centred).powi(2))
            .sum();
        captured / centred.dot(&centred)
    }

    /// The block eigensolver reads the modes the dense eigendecomposition did.
    /// On the shuffled parabola the spectrum is simple, so the seed axis is an
    /// affine image of the dense mode.
    #[test]
    fn seed_axis_matches_the_dense_mode_on_a_simple_spectrum() {
        let n = 200usize;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut state = 7usize;
        for i in (1..n).rev() {
            state = (state.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
            let j = state % (i + 1);
            perm.swap(i, j);
        }
        let y = Array2::from_shape_fn((n, 2), |(row, c)| {
            let u = 2.0 * perm[row] as f64 / n as f64 - 1.0;
            if c == 0 { u } else { u * u - 0.33 }
        });
        let coords = laplacian_eigenmap_coords(y.view(), 1, 10).unwrap();
        let dense = dense_eigenmap_modes(y.view(), 1, 10);
        let r2 = explained_fraction(coords.column(0), &[dense.column(0)]);
        assert!(
            r2 > 1.0 - 1e-8,
            "seed axis departs from the dense mode: R²={r2}"
        );
    }

    /// Evenly spaced points on a circle give a degenerate cos/sin pair of modes,
    /// so with `latent_dim = 1` the wanted block ends inside that pair. The seed
    /// axis must still come out as an eigenvector of the pair: inside the span of
    /// the dense solver's two modes.
    #[test]
    fn a_wanted_block_ending_inside_a_degenerate_pair_still_converges() {
        let n = 120usize;
        let y = Array2::from_shape_fn((n, 2), |(i, c)| {
            let angle = std::f64::consts::TAU * i as f64 / n as f64;
            if c == 0 { angle.cos() } else { angle.sin() }
        });
        let coords = laplacian_eigenmap_coords(y.view(), 1, 6).unwrap();
        let dense = dense_eigenmap_modes(y.view(), 2, 6);
        let r2 = explained_fraction(coords.column(0), &[dense.column(0), dense.column(1)]);
        assert!(
            r2 > 1.0 - 1e-8,
            "seed axis leaves the degenerate eigenspace: R²={r2}"
        );
    }

    /// Spearman rank correlation magnitude between two equal-length samples.
    fn abs_spearman(a: &[f64], b: &[f64]) -> f64 {
        fn ranks(x: &[f64]) -> Vec<f64> {
            let n = x.len();
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
            let mut r = vec![0.0; n];
            for (rank, &i) in idx.iter().enumerate() {
                r[i] = rank as f64;
            }
            r
        }
        let ra = ranks(a);
        let rb = ranks(b);
        let n = a.len() as f64;
        let mean = (n - 1.0) / 2.0;
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for i in 0..a.len() {
            let da = ra[i] - mean;
            let db = rb[i] - mean;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        (cov / (va.sqrt() * vb.sqrt())).abs()
    }

    /// A shuffled parabola (a 1-D curve in 2-D) — the issue #627 geometry. The
    /// recovered 1-D coordinate must match the true arc position up to monotone
    /// gauge, i.e. |Spearman| ≈ 1, despite the random row order.
    #[test]
    fn recovers_shuffled_parabola_ordering() {
        let n = 200usize;
        // Deterministic permutation via a full-period LCG so the test needs no
        // RNG dependency.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut state = 7usize;
        for i in (1..n).rev() {
            state = (state.wrapping_mul(1103515245).wrapping_add(12345)) % 2147483648;
            let j = state % (i + 1);
            perm.swap(i, j);
        }
        let mut y = Array2::<f64>::zeros((n, 2));
        let mut true_t = vec![0.0; n];
        for (row, &p) in perm.iter().enumerate() {
            let t = p as f64 / n as f64;
            let u = 2.0 * t - 1.0;
            y[[row, 0]] = u;
            y[[row, 1]] = u * u - 0.33;
            true_t[row] = t;
        }
        let coords = laplacian_eigenmap_coords(y.view(), 1, 10).unwrap();
        let recovered: Vec<f64> = coords.column(0).to_vec();
        let rho = abs_spearman(&recovered, &true_t);
        assert!(
            rho > 0.98,
            "spectral seed should recover the parabola ordering; |spearman|={rho}"
        );
        // Output axis is normalized to [0, 1].
        let lo = recovered.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = recovered.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((lo - 0.0).abs() < 1e-9 && (hi - 1.0).abs() < 1e-9);
    }

    /// Two intrinsic coordinates of a flat 2-D sheet embedded in 3-D should be
    /// recovered (jointly) up to a rotation: the seed's 2-D span must explain the
    /// true 2-D coordinates. We check each true axis is well-predicted by the
    /// best linear combination of the two recovered axes (R² close to 1).
    #[test]
    fn recovers_two_dimensional_sheet() {
        let side = 16usize;
        let n = side * side;
        let mut y = Array2::<f64>::zeros((n, 3));
        let mut ta = vec![0.0; n];
        let mut tb = vec![0.0; n];
        for r in 0..side {
            for c in 0..side {
                let i = r * side + c;
                let a = r as f64 / (side - 1) as f64;
                let b = c as f64 / (side - 1) as f64;
                ta[i] = a;
                tb[i] = b;
                // Embed the unit square in 3-D with a mild tilt (still a graph
                // over (a, b), so the intrinsic geometry is the flat sheet).
                y[[i, 0]] = a;
                y[[i, 1]] = b;
                y[[i, 2]] = 0.15 * (a + b);
            }
        }
        let coords = laplacian_eigenmap_coords(y.view(), 2, 8).unwrap();
        // Best linear fit of each true axis from [1, c0, c1].
        let r2_of = |truth: &[f64]| -> f64 {
            let mut xtx = [[0.0f64; 3]; 3];
            let mut xty = [0.0f64; 3];
            for i in 0..n {
                let x = [1.0, coords[[i, 0]], coords[[i, 1]]];
                for p in 0..3 {
                    xty[p] += x[p] * truth[i];
                    for q in 0..3 {
                        xtx[p][q] += x[p] * x[q];
                    }
                }
            }
            // Solve 3x3 via Cramer-ish Gaussian elimination.
            let mut m = xtx;
            let mut v = xty;
            for col in 0..3 {
                let piv = m[col][col];
                for k in 0..3 {
                    m[col][k] /= piv;
                }
                v[col] /= piv;
                for r2 in 0..3 {
                    if r2 != col {
                        let f = m[r2][col];
                        for k in 0..3 {
                            m[r2][k] -= f * m[col][k];
                        }
                        v[r2] -= f * v[col];
                    }
                }
            }
            let mean: f64 = truth.iter().sum::<f64>() / n as f64;
            let (mut sse, mut sst) = (0.0, 0.0);
            for i in 0..n {
                let pred = v[0] + v[1] * coords[[i, 0]] + v[2] * coords[[i, 1]];
                sse += (truth[i] - pred).powi(2);
                sst += (truth[i] - mean).powi(2);
            }
            1.0 - sse / sst
        };
        assert!(
            r2_of(&ta) > 0.95,
            "true axis a not recovered: R²={}",
            r2_of(&ta)
        );
        assert!(
            r2_of(&tb) > 0.95,
            "true axis b not recovered: R²={}",
            r2_of(&tb)
        );
    }

    #[test]
    fn rejects_too_few_rows() {
        let y = Array2::<f64>::zeros((2, 2));
        assert!(laplacian_eigenmap_coords(y.view(), 1, 5).is_err());
    }
}
