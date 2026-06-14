use super::*;

#[derive(Debug, Clone)]
pub struct CollocationOperatorMatrices {
    pub d0: Array2<f64>,
    pub d1: Array2<f64>,
    pub d2: Array2<f64>,
    pub collocation_points: Array2<f64>,
    /// Kernel-constraint nullspace transform `Z` applied internally to the
    /// raw kernel-basis K×K operator matrices (Some for Duchon, None for
    /// Matérn which uses a different basis).
    pub kernel_nullspace_transform: Option<Array2<f64>>,
    /// Polynomial block columns appended after the kernel block (Duchon
    /// polynomial null space). Zero for Matérn.
    pub polynomial_block_cols: usize,
}


#[derive(Debug, Clone)]
pub struct DuchonOperatorPenaltyMatrices {
    pub mass: Array2<f64>,
    pub tension: Array2<f64>,
    pub stiffness: Array2<f64>,
}


#[derive(Debug, Clone)]
pub struct ThinPlatePenaltyMatrix {
    pub penalty: Array2<f64>,
}


pub(crate) fn default_normalization_scale() -> f64 {
    1.0
}


pub(crate) fn validate_center_count(num_centers: usize) -> Result<(), BasisError> {
    if num_centers == 0 {
        crate::bail_invalid_basis!("center count must be positive");
    }
    Ok(())
}


pub(crate) fn select_equal_mass_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_center_count(num_centers)?;
    let n = data.nrows();
    let d = data.ncols();
    if num_centers > n {
        crate::bail_invalid_basis!(
            "equal-mass center selection requested {num_centers} centers but data has {n} rows"
        );
    }
    if d == 0 {
        crate::bail_invalid_basis!("equal-mass center selection requires at least one column");
    }
    #[derive(Clone, Copy)]
    struct Leaf {
        start: usize,
        end: usize,
    }

    // Recursive equal-mass partition that always splits the leaf along its widest
    // coordinate dimension. This addresses the root cause of PC1-only slicing by
    // adapting splits to the local geometry of each partition. Keep all row indices
    // in a single buffer and sort subranges in-place so center selection stays exact
    // without allocating fresh index vectors at every split.
    let mut order: Vec<usize> = (0..n).collect();
    let mut leaves = vec![Leaf { start: 0, end: n }];

    let choose_split_dim = |slice: &[usize]| -> usize {
        // Score candidate split dimensions in parallel, but keep each dimension's
        // row scan in serial row order and use the same strict-`>` update rule
        // (with lowest-dimension tie breaking) as the original greedy splitter.
        (0..d)
            .into_par_iter()
            .map(|j| {
                let mut minv = f64::INFINITY;
                let mut maxv = f64::NEG_INFINITY;
                for &idx in slice {
                    let v = data[[idx, j]];
                    if v < minv {
                        minv = v;
                    }
                    if v > maxv {
                        maxv = v;
                    }
                }
                let span = maxv - minv;
                let span = if span.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    span
                };
                (j, span)
            })
            .reduce_with(|a, b| {
                if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(j, _)| j)
            .unwrap_or(0)
    };

    while leaves.len() < num_centers {
        let mut split_pos = None;
        let mut split_size = 0usize;
        for (i, leaf) in leaves.iter().enumerate() {
            let leaf_size = leaf.end - leaf.start;
            if leaf_size > split_size && leaf_size > 1 {
                split_size = leaf_size;
                split_pos = Some(i);
            }
        }
        let Some(pos) = split_pos else {
            break;
        };

        let leaf = leaves.swap_remove(pos);
        let split_dim = choose_split_dim(&order[leaf.start..leaf.end]);
        order[leaf.start..leaf.end].sort_by(|&a, &b| {
            let ord = data[[a, split_dim]].total_cmp(&data[[b, split_dim]]);
            if ord.is_eq() { a.cmp(&b) } else { ord }
        });
        let mid = leaf.start + (split_size / 2);

        if mid == leaf.start || mid == leaf.end {
            leaves.push(leaf);
            break;
        }

        leaves.push(Leaf {
            start: leaf.start,
            end: mid,
        });
        leaves.push(Leaf {
            start: mid,
            end: leaf.end,
        });
    }

    if leaves.len() < num_centers {
        crate::bail_invalid_basis!(
            "equal-mass partition produced {} leaves, expected {num_centers}",
            leaves.len()
        );
    }

    let mut centers = Array2::<f64>::zeros((num_centers, d));
    for (c, leaf) in leaves.iter().take(num_centers).enumerate() {
        let slice = &order[leaf.start..leaf.end];
        let m = slice.len() as f64;
        let mut centroid = vec![0.0_f64; d];
        for &idx in slice {
            for j in 0..d {
                centroid[j] += data[[idx, j]];
            }
        }
        for v in &mut centroid {
            *v /= m.max(1.0);
        }

        let best_idx = slice
            .par_iter()
            .filter_map(|&idx| {
                let mut d2 = 0.0;
                for j in 0..d {
                    let delta = data[[idx, j]] - centroid[j];
                    d2 += delta * delta;
                }
                if d2.is_finite() {
                    Some((idx, d2))
                } else {
                    None
                }
            })
            .reduce_with(|a, b| {
                if b.1 < a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(idx, _)| idx)
            .unwrap_or(slice[0]);
        centers.row_mut(c).assign(&data.row(best_idx));
    }
    Ok(centers)
}


pub(crate) fn select_equal_mass_covar_representative_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_center_count(num_centers)?;
    let n = data.nrows();
    let d = data.ncols();
    if num_centers > n {
        crate::bail_invalid_basis!(
            "equal-mass covariate-representative center selection requested {num_centers} centers but data has {n} rows"
        );
    }
    if d == 0 {
        crate::bail_invalid_basis!(
            "equal-mass covariate-representative center selection requires at least one column"
                .to_string(),
        );
    }

    let mut split_dim = 0usize;
    let mut best_span = f64::NEG_INFINITY;
    for j in 0..d {
        let mut minv = f64::INFINITY;
        let mut maxv = f64::NEG_INFINITY;
        for i in 0..n {
            let v = data[[i, j]];
            if v < minv {
                minv = v;
            }
            if v > maxv {
                maxv = v;
            }
        }
        let span = maxv - minv;
        if span > best_span {
            best_span = span;
            split_dim = j;
        }
    }

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| {
        let ord = data[[a, split_dim]].total_cmp(&data[[b, split_dim]]);
        if ord.is_eq() { a.cmp(&b) } else { ord }
    });

    let mut centers = Array2::<f64>::zeros((num_centers, d));
    for c in 0..num_centers {
        let lo = (c * n) / num_centers;
        let hi = ((c + 1) * n) / num_centers;
        let chunk = &sorted[lo..hi.max(lo + 1)];
        let mid = chunk[chunk.len() / 2];
        centers.row_mut(c).assign(&data.row(mid));
    }
    Ok(centers)
}


pub(crate) fn select_kmeans_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
    max_iter: usize,
) -> Result<Array2<f64>, BasisError> {
    validate_center_count(num_centers)?;
    let n = data.nrows();
    let d = data.ncols();
    if num_centers > n {
        crate::bail_invalid_basis!("kmeans requested {num_centers} centers but data has {n} rows");
    }
    const KMEANS_PILOT_MAX_ROWS: usize = 20_000;
    if n > KMEANS_PILOT_MAX_ROWS {
        let pilot_n = KMEANS_PILOT_MAX_ROWS.max(num_centers);
        // log::info! rather than warn! — this is a deliberate performance
        // choice (O(n·k·iter) kmeans scales badly past ~20K rows), not a
        // problem the user can act on. Surfacing it as a warning adds
        // noise to CI output and mislabels normal operation.
        log::info!(
            "kmeans center selection using {}-row pilot subsample instead of full {} rows",
            pilot_n,
            n
        );
        let pilot = select_equal_mass_covar_representative_centers(data, pilot_n)?;
        return select_kmeans_centers(pilot.view(), num_centers, max_iter);
    }
    let mut centers = select_thin_plate_knots(data, num_centers)?;
    let mut assign = vec![0usize; n];
    let iters = max_iter.max(1);

    // For large n (large-scale), parallelize the assignment step.
    // Each observation's nearest-center query is independent.
    let use_parallel = n >= 10_000;

    for _ in 0..iters {
        // Assignment: find nearest center for each observation.
        if use_parallel {
            const KMEANS_CHUNK: usize = 4096;
            assign
                .par_chunks_mut(KMEANS_CHUNK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * KMEANS_CHUNK;
                    for (local, slot) in chunk.iter_mut().enumerate() {
                        let i = base + local;
                        let mut best = 0usize;
                        let mut best_d2 = f64::INFINITY;
                        for k in 0..num_centers {
                            let mut d2 = 0.0;
                            for c in 0..d {
                                let delta = data[[i, c]] - centers[[k, c]];
                                d2 += delta * delta;
                            }
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best = k;
                            }
                        }
                        *slot = best;
                    }
                });
        } else {
            for i in 0..n {
                let mut best = 0usize;
                let mut best_d2 = f64::INFINITY;
                for k in 0..num_centers {
                    let mut d2 = 0.0;
                    for c in 0..d {
                        let delta = data[[i, c]] - centers[[k, c]];
                        d2 += delta * delta;
                    }
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = k;
                    }
                }
                assign[i] = best;
            }
        }
        // Update: recompute centroids from assignments.
        let mut sums = Array2::<f64>::zeros((num_centers, d));
        let mut counts = vec![0usize; num_centers];
        for i in 0..n {
            let k = assign[i];
            counts[k] += 1;
            for c in 0..d {
                sums[[k, c]] += data[[i, c]];
            }
        }
        for k in 0..num_centers {
            if counts[k] == 0 {
                continue;
            }
            let inv = 1.0 / counts[k] as f64;
            for c in 0..d {
                centers[[k, c]] = sums[[k, c]] * inv;
            }
        }
    }
    Ok(centers)
}


pub(crate) fn cartesian_grid_axes(axes: &[Array1<f64>]) -> Result<Array2<f64>, BasisError> {
    if axes.is_empty() {
        crate::bail_invalid_basis!("uniform grid requires at least one axis");
    }
    let d = axes.len();
    let total = axes.iter().try_fold(1usize, |acc, axis| {
        acc.checked_mul(axis.len())
            .ok_or_else(|| BasisError::DimensionMismatch("uniform grid is too large".to_string()))
    })?;
    let mut out = Array2::<f64>::zeros((total, d));
    for r in 0..total {
        let mut q = r;
        for c in (0..d).rev() {
            let len = axes[c].len();
            let idx = q % len;
            q /= len;
            out[[r, c]] = axes[c][idx];
        }
    }
    Ok(out)
}


pub(crate) fn select_uniform_grid_centers(
    data: ArrayView2<'_, f64>,
    points_per_dim: usize,
) -> Result<Array2<f64>, BasisError> {
    if points_per_dim == 0 {
        crate::bail_invalid_basis!("uniform-grid points_per_dim must be positive");
    }
    let d = data.ncols();
    if d == 0 {
        crate::bail_invalid_basis!("uniform-grid center selection requires at least one column");
    }
    let mut axes = Vec::with_capacity(d);
    for c in 0..d {
        let col = data.column(c);
        let minv = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let maxv = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        axes.push(Array::linspace(minv, maxv, points_per_dim));
    }
    cartesian_grid_axes(&axes)
}
