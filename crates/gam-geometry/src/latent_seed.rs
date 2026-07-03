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

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView2};

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

    // Pairwise squared distances in response space. `n` is the number of fitted
    // rows; the seed is an O(n² q) preprocessing step, dominated by the O(n³)
    // dense eigensolve below — both are negligible next to the outer optimizer's
    // repeated inner REML solves.
    let mut d2 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for c in 0..features.ncols() {
                let diff = features[[i, c]] - features[[j, c]];
                s += diff * diff;
            }
            d2[[i, j]] = s;
            d2[[j, i]] = s;
        }
    }

    // Per-row k nearest neighbours (excluding self) and the bandwidth: the
    // median of the retained k-NN squared distances. Median is robust to the
    // far tail and keeps the affinities well-scaled across datasets.
    let mut order: Vec<usize> = (0..n).collect();
    let mut knn = vec![0usize; n * k];
    let mut knn_d2: Vec<f64> = Vec::with_capacity(n * k);
    for i in 0..n {
        order.sort_unstable_by(|&a, &b| {
            d2[[i, a]]
                .partial_cmp(&d2[[i, b]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // order[0] is i itself (distance 0); take the next k.
        let mut taken = 0usize;
        for &j in order.iter() {
            if j == i {
                continue;
            }
            knn[i * k + taken] = j;
            knn_d2.push(d2[[i, j]]);
            taken += 1;
            if taken == k {
                break;
            }
        }
    }
    let mut sorted_d2 = knn_d2.clone();
    sorted_d2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted_d2.is_empty() {
        1.0
    } else {
        sorted_d2[sorted_d2.len() / 2]
    };
    let epsilon = median.max(f64::MIN_POSITIVE);

    // Symmetric affinity matrix (union of directed k-NN edges).
    let mut w = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for t in 0..k {
            let j = knn[i * k + t];
            let aff = (-d2[[i, j]] / epsilon).exp();
            if aff > w[[i, j]] {
                w[[i, j]] = aff;
                w[[j, i]] = aff;
            }
        }
    }

    // Normalized Laplacian L = I − D^{-1/2} W D^{-1/2}. An isolated node (zero
    // degree) gets a tiny floor so D^{-1/2} stays finite; its row of L is then
    // effectively the identity and it contributes no spurious coupling.
    let mut dinv_sqrt = Array1::<f64>::zeros(n);
    for i in 0..n {
        let deg: f64 = w.row(i).sum();
        dinv_sqrt[i] = 1.0 / deg.max(f64::MIN_POSITIVE).sqrt();
    }
    let mut lap = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let norm = dinv_sqrt[i] * dinv_sqrt[j] * w[[i, j]];
            lap[[i, j]] = if i == j { 1.0 - norm } else { -norm };
        }
    }

    let (evals, evecs) = lap
        .eigh(Side::Lower)
        .map_err(|e| format!("laplacian_eigenmap_coords: eigendecomposition failed: {e}"))?;

    // Sort eigenpairs by ascending eigenvalue, then take modes 1..=latent_dim
    // (skip the trivial near-zero mode ∝ D^{1/2}·1).
    let mut idx: Vec<usize> = (0..evals.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        evals[a]
            .partial_cmp(&evals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut coords = Array2::<f64>::zeros((n, latent_dim));
    for a in 0..latent_dim {
        let col = idx[a + 1];
        for i in 0..n {
            // Generalized eigenvector y = D^{-1/2} v of L v = λ D v.
            coords[[i, a]] = dinv_sqrt[i] * evecs[[i, col]];
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
