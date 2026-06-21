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
        pub(crate) start: usize,
        pub(crate) end: usize,
    }

    // Recursive equal-mass partition that splits each leaf along its PRINCIPAL
    // axis (the leading eigen-direction of the leaf covariance), rather than along
    // its widest *coordinate* axis. An axis-aligned k-d-tree split is NOT
    // rotation-equivariant: under a rigid rotation of the inputs the per-leaf
    // coordinate spans change, "the widest coordinate" can flip, and a different
    // center set results — which is the root cause of #1456 (default
    // `thinplate(x,z)` drifting under rotation). The principal axis rotates with
    // the data, so projecting onto it and splitting at the equal-mass median
    // along that axis selects the SAME points (up to the rotation), making the
    // low-rank center set rotation-equivariant while staying deterministic and
    // permutation-invariant. The 2-D principal direction is taken from the
    // closed-form covariance angle (continuous through near-isotropic leaves)
    // rather than from `eigh`, whose eigenvalue ordering swaps discontinuously at
    // near-degeneracy and would flip the split by 90° under rotation. Keep all row
    // indices in a single buffer and sort subranges in-place so center selection
    // stays exact without allocating fresh index vectors at every split.
    let mut order: Vec<usize> = (0..n).collect();
    let mut leaves = vec![Leaf { start: 0, end: n }];

    // Leading-eigenvector ("principal") axis of the leaf covariance. The sign of
    // an eigenvector is arbitrary; we canonicalise it deterministically (largest
    // |component| made positive, lowest index breaking magnitude ties) so the
    // sort order — and hence the chosen split — is reproducible. The median
    // split itself is sign-invariant, but canonicalisation also pins the
    // tie-break ordering used for points with equal projections. Returns `None`
    // when the leaf has no usable spread (all eigenvalues ~0), in which case the
    // caller falls back to a deterministic coordinate-lexicographic order.
    let principal_axis = |slice: &[usize]| -> Option<Vec<f64>> {
        let m = slice.len();
        if m < 2 {
            return None;
        }
        let mut centroid = vec![0.0_f64; d];
        for &idx in slice {
            for j in 0..d {
                centroid[j] += data[[idx, j]];
            }
        }
        let inv = 1.0 / m as f64;
        for v in &mut centroid {
            *v *= inv;
        }
        // Covariance (d×d, small): symmetric accumulation of centred outer
        // products. d is the covariate dimension (typically 2 for thinplate).
        let mut cov = Array2::<f64>::zeros((d, d));
        for &idx in slice {
            for a in 0..d {
                let da = data[[idx, a]] - centroid[a];
                for b in a..d {
                    let db = data[[idx, b]] - centroid[b];
                    cov[[a, b]] += da * db;
                }
            }
        }
        for a in 0..d {
            cov[[a, a]] *= inv;
            for b in (a + 1)..d {
                cov[[a, b]] *= inv;
                cov[[b, a]] = cov[[a, b]];
            }
        }
        if cov.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Leading principal direction of the leaf covariance. For the 2-D case
        // (the thin-plate `(x, z)` smooth of #1456) use the CLOSED-FORM principal
        // angle of the 2×2 symmetric covariance rather than `eigh`. `eigh` returns
        // eigenvectors ordered by eigenvalue, so when a leaf is near-isotropic
        // (the two eigenvalues nearly equal) an arbitrarily small perturbation —
        // such as the rounding introduced by rotating the inputs — can SWAP the
        // ordering and flip the chosen axis by 90°, producing a completely
        // different partition (the #1456 failure: residual ~1.5, a different
        // center set). The closed form
        //     θ = ½·atan2(2·S_xy, S_xx − S_yy)
        // is CONTINUOUS in the covariance entries, so it tracks the major axis
        // smoothly through near-degeneracy, and it is rotation-equivariant: under
        // a rotation by φ the pair (2·S_xy, S_xx − S_yy) rotates by 2φ, so θ
        // rotates by exactly φ. It is undefined only at EXACT isotropy
        // (S_xy == 0 and S_xx == S_yy), where there is no preferred axis and we
        // fall back to the coordinate-lexicographic order. Higher dimensions keep
        // the `eigh` path (the rotation-invariance regression is 2-D).
        let mut axis: Vec<f64> = if d == 2 {
            let sxx = cov[[0, 0]];
            let syy = cov[[1, 1]];
            let sxy = cov[[0, 1]];
            if sxy == 0.0 && sxx == syy {
                return None;
            }
            let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
            vec![angle.cos(), angle.sin()]
        } else {
            // `eigh` returns eigenvalues in ascending order, so the principal axis
            // is the LAST column. Fall back to coordinate order if it cannot factor.
            let (evals, evecs) = cov.eigh(Side::Lower).ok()?;
            let last = evals.len().checked_sub(1)?;
            if !(evals[last] > 0.0) {
                return None;
            }
            (0..d).map(|r| evecs[[r, last]]).collect()
        };
        if axis.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Canonical, rotation-EQUIVARIANT orientation: point the axis toward the
        // leaf member farthest from the centroid. Distance-to-centroid is
        // rotation-invariant and the lowest-index tie-break is rotation-stable, so
        // the SAME row is chosen for the rotated leaf and the axis SIGN is
        // therefore equivariant (it rotates with the data). A canonical
        // orientation — not merely a canonical line — is required because an
        // equal-mass split of an ODD-sized leaf is asymmetric (the extra point
        // falls on one side of the median): a bare sign flip of the axis would
        // reassign the median point and produce a different partition under
        // rotation. Orienting by the farthest point removes that ambiguity.
        let mut far_idx = slice[0];
        let mut far_d2 = f64::NEG_INFINITY;
        for &idx in slice {
            let mut d2 = 0.0_f64;
            for j in 0..d {
                let delta = data[[idx, j]] - centroid[j];
                d2 += delta * delta;
            }
            if d2 > far_d2 || (d2 == far_d2 && idx < far_idx) {
                far_d2 = d2;
                far_idx = idx;
            }
        }
        let mut proj = 0.0_f64;
        for j in 0..d {
            proj += (data[[far_idx, j]] - centroid[j]) * axis[j];
        }
        if proj < 0.0 {
            for v in &mut axis {
                *v = -*v;
            }
        } else if proj == 0.0 {
            // Farthest point sits orthogonal to the axis (no orientation cue):
            // fall back to the deterministic magnitude-pivot sign so the split is
            // still reproducible.
            let mut pivot = 0usize;
            for r in 1..d {
                if axis[r].abs() > axis[pivot].abs() {
                    pivot = r;
                }
            }
            if axis[pivot] < 0.0 {
                for v in &mut axis {
                    *v = -*v;
                }
            }
        }
        Some(axis)
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
        let axis = principal_axis(&order[leaf.start..leaf.end]);
        match axis {
            Some(axis) => {
                // Project each row onto the principal axis and sort by the scalar
                // projection (index tie-break for determinism). The projection
                // rotates with the data, so this split is rotation-equivariant.
                order[leaf.start..leaf.end].sort_by(|&a, &b| {
                    let mut pa = 0.0_f64;
                    let mut pb = 0.0_f64;
                    for j in 0..d {
                        pa += data[[a, j]] * axis[j];
                        pb += data[[b, j]] * axis[j];
                    }
                    let ord = pa.total_cmp(&pb);
                    if ord.is_eq() { a.cmp(&b) } else { ord }
                });
            }
            None => {
                // Degenerate leaf (no spread / non-finite covariance): fall back
                // to a deterministic coordinate-lexicographic order so the split
                // is still well defined and permutation-invariant.
                order[leaf.start..leaf.end].sort_by(|&a, &b| {
                    for j in 0..d {
                        let ord = data[[a, j]].total_cmp(&data[[b, j]]);
                        if !ord.is_eq() {
                            return ord;
                        }
                    }
                    a.cmp(&b)
                });
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 2-D scatter with a clear, off-axis anisotropy so the
    /// principal axis is well separated from both coordinate axes. A small
    /// lattice perturbed by a reproducible pseudo-random jitter; no RNG crate
    /// needed, fully deterministic across runs.
    fn make_points() -> Array2<f64> {
        let n_side = 11usize;
        let n = n_side * n_side;
        let mut pts = Array2::<f64>::zeros((n, 2));
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            // xorshift64* — deterministic, no external dependency.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut r = 0usize;
        for i in 0..n_side {
            for j in 0..n_side {
                let x = i as f64;
                // Shear the lattice so its spread is genuinely off both axes,
                // making the "widest coordinate" choice fragile under rotation.
                let y = 0.35 * i as f64 + 1.7 * j as f64;
                pts[[r, 0]] = x + 0.05 * (next() - 0.5);
                pts[[r, 1]] = y + 0.05 * (next() - 0.5);
                r += 1;
            }
        }
        pts
    }

    /// Assert two center sets are equal up to ordering, by greedily matching each
    /// row of `expected` to its nearest row of `actual` and requiring the match
    /// residual to be below `tol`. Both sets must have the same number of rows.
    fn assert_center_sets_match(expected: ArrayView2<'_, f64>, actual: ArrayView2<'_, f64>, tol: f64) {
        assert_eq!(expected.nrows(), actual.nrows(), "center counts differ");
        let k = expected.nrows();
        let mut used = vec![false; k];
        let mut worst = 0.0_f64;
        for ei in 0..k {
            let mut best = usize::MAX;
            let mut best_d2 = f64::INFINITY;
            for ai in 0..k {
                if used[ai] {
                    continue;
                }
                let dx = expected[[ei, 0]] - actual[[ai, 0]];
                let dy = expected[[ei, 1]] - actual[[ai, 1]];
                let d2 = dx * dx + dy * dy;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = ai;
                }
            }
            assert!(best != usize::MAX, "no unmatched center available");
            used[best] = true;
            worst = worst.max(best_d2.sqrt());
        }
        assert!(
            worst <= tol,
            "rotation-equivariance violated: worst center match residual {worst:.3e} > tol {tol:.3e}"
        );
    }

    /// Existing invariant we must preserve: center selection is invariant to row
    /// permutation of the inputs (the selected SET is unchanged when rows are
    /// reordered). Locks the determinism/permutation-invariance the fix must keep.
    #[test]
    fn equal_mass_centers_are_permutation_invariant() {
        let pts = make_points();
        let num_centers = 16usize;
        let base = select_equal_mass_centers(pts.view(), num_centers).unwrap();

        let n = pts.nrows();
        let mut perm: Vec<usize> = (0..n).collect();
        // Deterministic shuffle.
        let mut state: u64 = 0xD1B5_4A32_D192_ED03;
        for i in (1..n).rev() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let j = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        let mut permuted = Array2::<f64>::zeros((n, 2));
        for (new_r, &old_r) in perm.iter().enumerate() {
            permuted[[new_r, 0]] = pts[[old_r, 0]];
            permuted[[new_r, 1]] = pts[[old_r, 1]];
        }
        let permuted_centers = select_equal_mass_centers(permuted.view(), num_centers).unwrap();
        assert_center_sets_match(base.view(), permuted_centers.view(), 1e-13);
    }

    /// #1456: the low-rank equal-mass center selector must be rotation
    /// EQUIVARIANT — rigidly rotating the inputs about their centroid rotates
    /// the selected center SET by the same rotation (so the un-rotated centers
    /// coincide with the base selection). The axis-aligned k-d split was
    /// rotation-SENSITIVE (the "widest coordinate" flipped under rotation,
    /// picking a different equal-mass set and drifting the fitted thin-plate
    /// surface ~2% of the signal range); the principal-axis (closed-form
    /// covariance-angle) split restores the invariant. Exercises an exact 90°
    /// rotation (no floating rounding of its own) and a generic 0.7-rad angle.
    #[test]
    fn equal_mass_centers_are_rotation_equivariant() {
        // ~300 deterministic points on an anisotropic cloud (a sheared lattice
        // plus jitter), so leaves have a well-defined principal axis.
        let n = 300usize;
        let mut pts = Array2::<f64>::zeros((n, 2));
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for r in 0..n {
            let u = next();
            let v = next();
            // Shear to break isotropy (anisotropic principal axes per leaf).
            pts[[r, 0]] = 2.0 * u - 1.0 + 0.6 * (2.0 * v - 1.0);
            pts[[r, 1]] = 2.0 * v - 1.0;
        }
        let num_centers = 48usize;
        let base = select_equal_mass_centers(pts.view(), num_centers).unwrap();

        // Centroid (rotation pivot).
        let mut cx = 0.0;
        let mut cy = 0.0;
        for r in 0..n {
            cx += pts[[r, 0]];
            cy += pts[[r, 1]];
        }
        cx /= n as f64;
        cy /= n as f64;

        for &(ca, sa) in &[(0.0_f64, 1.0_f64), (0.7f64.cos(), 0.7f64.sin())] {
            let mut rot = Array2::<f64>::zeros((n, 2));
            for r in 0..n {
                let x = pts[[r, 0]] - cx;
                let y = pts[[r, 1]] - cy;
                rot[[r, 0]] = ca * x - sa * y + cx;
                rot[[r, 1]] = sa * x + ca * y + cy;
            }
            let rotated_centers = select_equal_mass_centers(rot.view(), num_centers).unwrap();
            // Un-rotate the centers selected in the rotated frame and require the
            // SET to coincide with the base selection (equivariance).
            let mut unrotated = Array2::<f64>::zeros((num_centers, 2));
            for r in 0..num_centers {
                let x = rotated_centers[[r, 0]] - cx;
                let y = rotated_centers[[r, 1]] - cy;
                unrotated[[r, 0]] = ca * x + sa * y + cx;
                unrotated[[r, 1]] = -sa * x + ca * y + cy;
            }
            assert_center_sets_match(base.view(), unrotated.view(), 1e-9);
        }
    }
}
