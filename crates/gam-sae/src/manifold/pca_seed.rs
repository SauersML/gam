//! Production PCA-based per-atom chart seeding for the SAE manifold fit.
//! Moved out of the (test-only) `tests` module so the production FFI seed
//! path can call it in release builds.

use super::SaeAtomBasisKind;
use faer::Side;
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use ndarray::{Array1, Array2, Array3, ArrayView2};

/// Residual-norm floor below which the surplus atom's second phase axis is
/// treated as collinear with the first (a degenerate 2-plane). Since both
/// candidate directions enter [`surplus_phase_plane`] unit-normalized, the
/// post-orthogonalization residual norm equals the sine of the angle between
/// them, so this is an absolute "how far from collinear" threshold in `[0, 1]`.
const SURPLUS_DIR_FLOOR: f64 = 1.0e-6;

/// Golden-ratio conjugate `φ⁻¹`. Additive step of a low-discrepancy (mod 1)
/// rotation that folds the #976 multi-start retry index into the periodic
/// `phase_offset`, so successive reseeds place the SAME surplus atom at a
/// well-spread DISTINCT circle phase. `pc_pair_offset == 0` contributes exactly
/// `0.0`, leaving the original `atom_idx / k_atoms` offset bit-for-bit.
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;

/// Maximum rows used by the topology-aware curved-atom seed.  The seed is only
/// a starting chart, so a deterministic stride subsample keeps the graph solve
/// bounded while preserving reproducibility and memory use on large inputs.
const TOPOLOGY_SEED_MAX_POINTS: usize = 4096;

/// k used for the local neighborhood graph in the topology-aware seed.  This is
/// large enough to connect ordinary manifold samples but small enough that the
/// dense Laplacian eigensolve is still dominated by one joint SAE iteration on
/// the subsample.
const TOPOLOGY_SEED_KNN: usize = 12;

fn is_curved_kind(kind: &SaeAtomBasisKind) -> bool {
    matches!(
        kind,
        SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus | SaeAtomBasisKind::Sphere
    )
}

fn topology_seed_subsample(n_obs: usize) -> Vec<usize> {
    if n_obs <= TOPOLOGY_SEED_MAX_POINTS {
        return (0..n_obs).collect();
    }
    let mut rows = Vec::with_capacity(TOPOLOGY_SEED_MAX_POINTS);
    for i in 0..TOPOLOGY_SEED_MAX_POINTS {
        rows.push(i * n_obs / TOPOLOGY_SEED_MAX_POINTS);
    }
    rows
}

fn squared_distance_rows(z: ArrayView2<'_, f64>, a: usize, b: usize) -> f64 {
    let mut acc = 0.0;
    for c in 0..z.ncols() {
        let d = z[[a, c]] - z[[b, c]];
        acc += d * d;
    }
    acc
}

/// Topology-aware deterministic initialization for curved atom charts.
///
/// The issue asks for persistent-cohomology harmonic coordinates.  In the core
/// build we avoid a heavyweight dependency and compute the same object needed by
/// the optimizer seed: low-energy harmonic coordinates on a symmetric kNN graph
/// built from a bounded deterministic subsample.  The first non-constant graph
/// Laplacian eigenfunctions are the discrete harmonic representatives; reading
/// their phases gives circle/torus coordinates, while normalizing the first
/// three gives a sphere chart.  If the graph is too small/degenerate this returns
/// `Ok(None)` and the caller falls back to the older PCA seed.
fn topology_curved_seed_initial_coords(
    z: ArrayView2<'_, f64>,
    basis_kinds: &[SaeAtomBasisKind],
    atom_dim: &[usize],
    pc_pair_offset: usize,
) -> Result<Option<Array3<f64>>, String> {
    if !basis_kinds.iter().any(is_curved_kind) || z.nrows() < 4 || z.ncols() == 0 {
        return Ok(None);
    }
    for ((row, col), &value) in z.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "sae_pca_seed: Z must be finite; Z[{row}, {col}] = {value}"
            ));
        }
    }
    let rows = topology_seed_subsample(z.nrows());
    let m = rows.len();
    if m < 4 {
        return Ok(None);
    }
    let k = TOPOLOGY_SEED_KNN.min(m - 1);
    let mut w = Array2::<f64>::zeros((m, m));
    for (ia, &ra) in rows.iter().enumerate() {
        let mut dists = Vec::with_capacity(m - 1);
        for (ib, &rb) in rows.iter().enumerate() {
            if ia != ib {
                dists.push((squared_distance_rows(z, ra, rb), ib));
            }
        }
        dists.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let scale = dists[k.saturating_sub(1)].0.max(1.0e-24);
        for &(dist2, ib) in dists.iter().take(k) {
            let wij = (-dist2 / scale).exp().max(1.0e-12);
            if wij > w[[ia, ib]] {
                w[[ia, ib]] = wij;
            }
            if wij > w[[ib, ia]] {
                w[[ib, ia]] = wij;
            }
        }
    }
    let mut lap = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        let deg: f64 = w.row(i).sum();
        if deg <= 0.0 || !deg.is_finite() {
            return Ok(None);
        }
        lap[[i, i]] = 1.0;
        let inv_sqrt = 1.0 / deg.sqrt();
        for j in 0..m {
            if i != j && w[[i, j]] != 0.0 {
                let deg_j: f64 = w.row(j).sum();
                lap[[i, j]] = -w[[i, j]] * inv_sqrt / deg_j.sqrt();
            }
        }
    }
    let (evals, evecs) = lap
        .eigh(Side::Lower)
        .map_err(|err| format!("topology_seed: graph Laplacian eigensolve failed: {err:?}"))?;
    if evals.len() < 3 {
        return Ok(None);
    }
    let d_max = atom_dim.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((basis_kinds.len(), z.nrows(), d_max));
    let interp = |sample_values: &Array1<f64>, row: usize| -> f64 {
        if let Some(pos) = rows.iter().position(|&r| r == row) {
            return sample_values[pos];
        }
        let mut best = [(f64::INFINITY, 0usize); 3];
        for (i, &r) in rows.iter().enumerate() {
            let d = squared_distance_rows(z, row, r);
            if d < best[2].0 {
                best[2] = (d, i);
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for (d, i) in best {
            let ww = 1.0 / d.max(1.0e-24);
            num += ww * sample_values[i];
            den += ww;
        }
        num / den
    };
    let component = |idx: usize| -> Option<Array1<f64>> {
        if idx >= evecs.ncols() {
            None
        } else {
            Some(evecs.column(idx).to_owned())
        }
    };
    for atom_idx in 0..basis_kinds.len() {
        let d = atom_dim[atom_idx];
        match basis_kinds[atom_idx] {
            SaeAtomBasisKind::Periodic => {
                let pair_count = (evecs.ncols().saturating_sub(1)) / 2;
                let base = if pair_count > 0 {
                    1 + 2 * ((atom_idx + pc_pair_offset) % pair_count)
                } else {
                    1
                };
                let Some(a) = component(base) else {
                    continue;
                };
                let Some(b) = component(base + 1) else {
                    continue;
                };
                for row in 0..z.nrows() {
                    let phase = interp(&b, row).atan2(interp(&a, row)) / std::f64::consts::TAU;
                    out[[atom_idx, row, 0]] = phase - phase.floor();
                }
            }
            SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    let pair_count = (evecs.ncols().saturating_sub(1)) / 2;
                    let pair = if pair_count > 0 {
                        (axis + pc_pair_offset) % pair_count
                    } else {
                        axis
                    };
                    let Some(a) = component(1 + 2 * pair) else {
                        break;
                    };
                    let Some(b) = component(2 + 2 * pair) else {
                        break;
                    };
                    for row in 0..z.nrows() {
                        let phase = interp(&b, row).atan2(interp(&a, row)) / std::f64::consts::TAU;
                        out[[atom_idx, row, axis]] = phase - phase.floor();
                    }
                }
            }
            SaeAtomBasisKind::Sphere => {
                let base = if evecs.ncols() > 3 {
                    1 + (2 * pc_pair_offset) % (evecs.ncols() - 3)
                } else {
                    1
                };
                let (Some(xv), Some(yv), Some(zv)) =
                    (component(base), component(base + 1), component(base + 2))
                else {
                    continue;
                };
                for row in 0..z.nrows() {
                    let x = interp(&xv, row);
                    let y = interp(&yv, row);
                    let zz = interp(&zv, row);
                    let norm = (x * x + y * y + zz * zz).sqrt().max(1.0e-24);
                    if d >= 1 {
                        out[[atom_idx, row, 0]] = (zz / norm).clamp(-1.0, 1.0).asin();
                    }
                    if d >= 2 {
                        out[[atom_idx, row, 1]] = y.atan2(x);
                    }
                }
            }
            _ => {
                for axis in 0..d {
                    let Some(values) = component(1 + axis) else {
                        break;
                    };
                    let (lo, hi) = values
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                            (lo.min(v), hi.max(v))
                        });
                    let span = hi - lo;
                    if span > 0.0 && span.is_finite() {
                        for row in 0..z.nrows() {
                            out[[atom_idx, row, axis]] = (interp(&values, row) - lo) / span - 0.5;
                        }
                    }
                }
            }
        }
    }
    Ok(Some(out))
}

/// Deterministic generic 2-plane for an OVERCOMPLETE (#1893) SURPLUS periodic
/// atom, i.e. one whose index outruns the available disjoint PC pairs. Builds
/// two pseudo-random combinations of ALL principal directions (a random 2-plane
/// in the PC span) and Gram-Schmidt orthogonalizes the second against the first.
///
/// `pc_pair_offset` (the #976 multi-start retry index) is folded into the
/// splitmix64 weight key via a base-`k_atoms` mix of `(atom_idx, pc_pair_offset)`,
/// so the SAME surplus atom lands on a DIFFERENT random plane on each reseed
/// retry (distinct basins by construction) while every `(atom, retry)` key stays
/// distinct and `pc_pair_offset == 0` reproduces the original key bit-for-bit.
///
/// Returns `(dir1, dir2, two_dimensional)`. When the orthogonalized residual
/// falls below [`SURPLUS_DIR_FLOOR`] the random combination was essentially
/// collinear (or the effective PC span is 1-D), so `two_dimensional` is `false`
/// and the caller must fall back to the 1-D span phase path rather than feed an
/// `atan2` a degenerate plane that collapses to two phase points.
fn surplus_phase_plane(
    vt: ArrayView2<'_, f64>,
    atom_idx: usize,
    pc_pair_offset: usize,
    k_atoms: usize,
) -> (Array1<f64>, Array1<f64>, bool) {
    let vt_rows = vt.nrows();
    let ncols = vt.ncols();
    // splitmix64-seeded weights keyed by (atom_idx, pc_pair_offset, pc):
    // reproducible, no RNG crate, distinct per atom and per multi-start retry.
    let mix = |mut z: u64| -> f64 {
        z = z.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut a = Array1::<f64>::zeros(ncols);
    let mut b = Array1::<f64>::zeros(ncols);
    for pc in 0..vt_rows {
        let row_pc = vt.row(pc);
        // Base-`k_atoms` mix of (atom_idx, pc_pair_offset) before the `<< 20`
        // spread. At `pc_pair_offset == 0` this is exactly `(atom_idx << 20) ^ pc`
        // — the original key, bit-for-bit — so the first attempt is unchanged.
        let key =
            (((atom_idx as u64) + (pc_pair_offset as u64) * (k_atoms as u64)) << 20) ^ (pc as u64);
        let wa = mix(key);
        let wb = mix(key ^ 0xD1B54A32D192ED03);
        for c in 0..ncols {
            a[c] += wa * row_pc[c];
            b[c] += wb * row_pc[c];
        }
    }
    let na = a.dot(&a).sqrt().max(1.0e-12);
    a.mapv_inplace(|v| v / na);
    let nb = b.dot(&b).sqrt().max(1.0e-12);
    b.mapv_inplace(|v| v / nb);
    // Gram-Schmidt: strip dir1's component out of dir2 so the two phase axes are
    // not near-collinear. Two independently normalized random combinations can be
    // near-parallel (likely when the effective spectrum is small), which would
    // make `atan2(z·dir2, z·dir1)` take only two values and kill the diversity
    // this branch exists to create. With unit `dir1` the residual norm equals the
    // sine of the angle between the raw directions.
    let proj = b.dot(&a);
    b.scaled_add(-proj, &a);
    let nb_res = b.dot(&b).sqrt();
    if nb_res > SURPLUS_DIR_FLOOR {
        b.mapv_inplace(|v| v / nb_res);
        (a, b, true)
    } else {
        (a, b, false)
    }
}

/// PCA-based seed for SAE atom latent coordinates. Centers `z`, takes its SVD,
/// and projects onto leading principal components to initialize each atom's
/// chart according to its [`SaeAtomBasisKind`]: periodic atoms read a `[0, 1)`
/// phase off the top-2 PCs (remaining axes min-max normalized to
/// `[-0.5, 0.5]`), sphere atoms read `(lat, lon)` off the unit-normalized top-3
/// PCs, torus axes read a `[0, 1)` phase off disjoint PC pairs, and
/// Euclidean/other atoms take score-scaled, min-max-normalized PC projections.
/// Returns a padded
/// `(K_atoms, n_obs, d_max)` coordinate array.
pub fn sae_pca_seed_initial_coords(
    z: ArrayView2<'_, f64>,
    basis_kinds: &[SaeAtomBasisKind],
    atom_dim: &[usize],
) -> Result<Array3<f64>, String> {
    sae_pca_seed_initial_coords_with_pc_offset(z, basis_kinds, atom_dim, 0)
}

/// PCA seed with a deterministic principal-component-pair ROTATION offset.
///
/// Identical to [`sae_pca_seed_initial_coords`] (which is this with
/// `pc_pair_offset = 0`) except every atom reads its chart off a PC subspace
/// shifted by `pc_pair_offset` pairs. This is the lever the #976 simultaneous
/// co-collapse multi-start uses to make successive reseeds explore GENUINELY
/// DIFFERENT basins: the residual a co-collapsed dictionary leaves is ≈ the
/// target on every retry, so a fixed-offset reseed re-reads the SAME leading
/// PCs and the joint LSQ relaxes back into the SAME degenerate basin — the
/// budget-N multi-start would then be N identical attempts. Shifting the PC
/// pairs by the retry index lands the atoms on a disjoint principal subspace
/// each attempt (top pairs on retry 0, next pairs on retry 1, …), so the basins
/// are distinct by construction. The offset is a pure deterministic function of
/// the retry count (no RNG), so the seed stays bit-reproducible run-to-run and
/// across thread/device counts.
pub fn sae_pca_seed_initial_coords_with_pc_offset(
    z: ArrayView2<'_, f64>,
    basis_kinds: &[SaeAtomBasisKind],
    atom_dim: &[usize],
    pc_pair_offset: usize,
) -> Result<Array3<f64>, String> {
    if let Some(seed) =
        topology_curved_seed_initial_coords(z, basis_kinds, atom_dim, pc_pair_offset)?
    {
        return Ok(seed);
    }
    let k_atoms = basis_kinds.len();
    let (n_obs, _p_out) = z.dim();
    let d_max = atom_dim.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((k_atoms, n_obs, d_max));
    if n_obs == 0 || z.ncols() == 0 {
        return Ok(out);
    }
    // Reject non-finite input up front so a clean error surfaces here rather
    // than a silent non-finite seed (or an opaque SVD failure) downstream.
    for ((row, col), &value) in z.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "sae_pca_seed: Z must be finite; Z[{row}, {col}] = {value}"
            ));
        }
    }
    // Accumulate the column mean with Welford's running update
    // `mean += (x − mean) / count` instead of a plain running sum. The plain
    // sum overflows to `±inf` for huge finite columns (e.g. two rows of
    // `1e308` sum to `2e308 = inf`), which poisons the centered matrix and the
    // SVD. Welford's update keeps the accumulator bounded by the column's data
    // range, so the mean is finite whenever the inputs are.
    let mut col_means = Array1::<f64>::zeros(z.ncols());
    for col in 0..z.ncols() {
        let mut mean = 0.0_f64;
        for (count, row) in (0..n_obs).enumerate() {
            let x = z[[row, col]];
            mean += (x - mean) / (count as f64 + 1.0);
        }
        col_means[col] = mean;
    }
    let mut centered = z.to_owned();
    for row in 0..n_obs {
        for col in 0..z.ncols() {
            centered[[row, col]] -= col_means[col];
        }
    }
    // Centering can still overflow if the data span itself is non-finite
    // (e.g. `+1e308` and `−1e308` in one column give a finite mean but an
    // `inf` deviation). Surface that as a clean error rather than feeding a
    // non-finite matrix to the SVD.
    for ((row, col), &value) in centered.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "sae_pca_seed: centered Z is non-finite at [{row}, {col}] \
                 (data span exceeds f64 range); rescale Z before seeding"
            ));
        }
    }
    let (u_opt, s_vals, vt_opt) = centered
        .svd(true, true)
        .map_err(|err| format!("sae_pca_seed: SVD failed: {err:?}"))?;
    let u = u_opt.ok_or_else(|| "sae_pca_seed: SVD returned no U".to_string())?;
    let vt = vt_opt.ok_or_else(|| "sae_pca_seed: SVD returned no Vt".to_string())?;
    let vt_rows = vt.nrows();
    let u_cols = u.ncols();
    let two_pi = std::f64::consts::TAU;
    for atom_idx in 0..k_atoms {
        let d = atom_dim[atom_idx];
        if d == 0 {
            continue;
        }
        match &basis_kinds[atom_idx] {
            SaeAtomBasisKind::Periodic => {
                if vt_rows >= 1 {
                    // Diversify the per-atom circle seed (issue #671). The
                    // previous scheme shared PC0 as the first phase axis for
                    // *every* atom, so all periodic atoms read off nearly the
                    // same phase coordinate, producing near-duplicate basis
                    // designs and a severely ill-conditioned joint decoder LSQ
                    // seed. Give each atom a disjoint pair of principal
                    // components `(PC_{2k}, PC_{2k+1})` when the spectrum is
                    // wide enough, wrapping around only when atoms outnumber the
                    // available PC pairs. This keeps distinct atoms' seed
                    // coordinates decorrelated so the decoder seed stays
                    // well-conditioned and the cross-atom Gram starts small.
                    let pc_pairs = vt_rows / 2;
                    let (pc1_row, pc2_row) = if pc_pairs >= 1 {
                        // Rotate the per-atom PC pair by the multi-start offset so
                        // a co-collapse reseed retry reads a DISJOINT principal
                        // subspace (the #976 distinct-basin lever).
                        let pair = (atom_idx + pc_pair_offset) % pc_pairs;
                        (2 * pair, 2 * pair + 1)
                    } else {
                        (0, 0)
                    };
                    // #1893 — OVERCOMPLETE (K ≫ p) generic seeding. Linear
                    // projection has only `pc_pairs ≈ p/2` distinct PC planes, so
                    // once atoms outnumber them the old scheme reused a plane with
                    // only a constant phase shift — a decoder-equivalent DUPLICATE
                    // that leaves the joint block rank-deficient and drives the
                    // co-collapse. For every SURPLUS atom (index ≥ pc_pairs) build a
                    // DISTINCT generic phase plane from a deterministic pseudo-random
                    // combination of ALL principal directions (a random 2-plane in
                    // the PC span), so K ≫ p atoms are pairwise distinct for any K.
                    // Atoms 0..pc_pairs keep their exact PC-pair seed (small-K path
                    // byte-for-byte unchanged); only wrap-victims are rerouted.
                    let surplus = pc_pairs >= 1 && k_atoms > pc_pairs && atom_idx >= pc_pairs;
                    let phase_offset = if pc_pairs > 0 && pc_pairs < k_atoms {
                        // Fold the #976 retry index into the circle offset with a
                        // golden-ratio additive rotation (low-discrepancy mod 1) so
                        // successive reseeds place the SAME atom at a distinct
                        // phase. The `phase - phase.floor()` wrap below reduces this
                        // mod 1; `pc_pair_offset == 0` adds exactly `0.0`, leaving
                        // the original `atom_idx / k_atoms` offset bit-for-bit.
                        atom_idx as f64 / k_atoms as f64
                            + pc_pair_offset as f64 * GOLDEN_RATIO_CONJUGATE
                    } else {
                        0.0
                    };
                    let s0 = s_vals.get(pc1_row).copied().unwrap_or(0.0).abs();
                    let s1 = s_vals.get(pc2_row).copied().unwrap_or(0.0).abs();
                    let has_two_dimensional_phase =
                        vt_rows >= 2 && pc2_row != pc1_row && s1 > 1.0e-10 * s0.max(1.0);
                    // `two_dimensional_phase` gates the atan2 (2-plane) read vs the
                    // min-max (1-span) read. Non-surplus atoms keep the rank check
                    // above unchanged; surplus atoms use the Gram-Schmidt residual
                    // flag so a near-collinear random 2-plane falls back to the 1-D
                    // path instead of collapsing atan2 to two phase points.
                    let mut two_dimensional_phase = has_two_dimensional_phase;
                    let dir1: Array1<f64>;
                    let dir2: Array1<f64>;
                    if surplus {
                        // #1893/#976 — distinct generic 2-plane per (atom, retry),
                        // Gram-Schmidt orthogonalized with a collinearity floor.
                        let (a, b, two_d) =
                            surplus_phase_plane(vt.view(), atom_idx, pc_pair_offset, k_atoms);
                        dir1 = a;
                        dir2 = b;
                        two_dimensional_phase = two_d;
                    } else {
                        dir1 = vt.row(pc1_row.min(vt_rows - 1)).to_owned();
                        dir2 = vt.row(pc2_row.min(vt_rows - 1)).to_owned();
                    }
                    let pc1 = dir1.view();
                    if two_dimensional_phase {
                        let pc2 = dir2.view();
                        for row in 0..n_obs {
                            let mut a = 0.0_f64;
                            let mut b = 0.0_f64;
                            for col in 0..centered.ncols() {
                                a += centered[[row, col]] * pc1[col];
                                b += centered[[row, col]] * pc2[col];
                            }
                            let phase = b.atan2(a) / two_pi + phase_offset;
                            out[[atom_idx, row, 0]] = phase - phase.floor();
                        }
                    } else {
                        let mut proj = Array1::<f64>::zeros(n_obs);
                        for row in 0..n_obs {
                            let mut acc = 0.0_f64;
                            for col in 0..centered.ncols() {
                                acc += centered[[row, col]] * pc1[col];
                            }
                            proj[row] = acc;
                        }
                        let (min_v, max_v) = proj
                            .iter()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                                (lo.min(v), hi.max(v))
                            });
                        let span = max_v - min_v;
                        if span > 0.0 {
                            for row in 0..n_obs {
                                let phase = (proj[row] - min_v) / span + phase_offset;
                                out[[atom_idx, row, 0]] = phase - phase.floor();
                            }
                        }
                    }
                }
                for axis in 1..d {
                    if axis >= vt_rows {
                        break;
                    }
                    let pc = vt.row(axis);
                    let mut proj = Array1::<f64>::zeros(n_obs);
                    for row in 0..n_obs {
                        let mut acc = 0.0_f64;
                        for col in 0..centered.ncols() {
                            acc += centered[[row, col]] * pc[col];
                        }
                        proj[row] = acc;
                    }
                    let (min_v, max_v) = proj
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                            (lo.min(v), hi.max(v))
                        });
                    let span = max_v - min_v;
                    if span > 0.0 {
                        for row in 0..n_obs {
                            out[[atom_idx, row, axis]] = (proj[row] - min_v) / span - 0.5;
                        }
                    }
                }
            }
            SaeAtomBasisKind::Sphere => {
                // Seed the sphere chart from the top-3 PCs: drop the centred
                // response onto (pc0, pc1, pc2), unit-normalise, and read off
                // (lat, lon). This places every row on the chart with
                // `lat ∈ (-π/2, π/2)` and `lon ∈ (-π, π]`.
                let n_pc = vt_rows.min(3);
                if n_pc == 0 {
                    continue;
                }
                // Rotate the sphere's leading-PC window by the multi-start offset
                // (in PC-pair units, mod the available PCs) so a reseed retry
                // reads a distinct 3-PC subspace (the #976 distinct-basin lever).
                let base = if vt_rows > 0 {
                    (2 * pc_pair_offset) % vt_rows
                } else {
                    0
                };
                let pcs: Vec<_> = (0..n_pc).map(|i| vt.row((base + i) % vt_rows)).collect();
                for row in 0..n_obs {
                    let mut amb = [0.0_f64; 3];
                    for (i, pc) in pcs.iter().enumerate() {
                        let mut acc = 0.0_f64;
                        for col in 0..centered.ncols() {
                            acc += centered[[row, col]] * pc[col];
                        }
                        amb[i] = acc;
                    }
                    let norm = (amb[0] * amb[0] + amb[1] * amb[1] + amb[2] * amb[2]).sqrt();
                    let (x, y, z) = if norm > 0.0 {
                        (amb[0] / norm, amb[1] / norm, amb[2] / norm)
                    } else {
                        (1.0, 0.0, 0.0)
                    };
                    let lat = z.clamp(-1.0, 1.0).asin();
                    let lon = y.atan2(x);
                    if d >= 1 {
                        out[[atom_idx, row, 0]] = lat;
                    }
                    if d >= 2 {
                        out[[atom_idx, row, 1]] = lon;
                    }
                }
            }
            SaeAtomBasisKind::Torus => {
                // Seed each torus axis from a disjoint pair of PCs: axis `a`
                // uses (pc_{2a}, pc_{2a+1}) projected onto the centred
                // response and read off as `atan2`, normalised to `[0, 1)`.
                let pc_pairs = vt_rows / 2;
                for axis in 0..d {
                    // Rotate each torus axis's PC pair by the multi-start offset
                    // (same #976 distinct-basin lever as the periodic arm). With
                    // `pc_pair_offset == 0` this is the identity (`pair == axis`)
                    // and the original `pc_b_idx >= vt_rows` break is preserved
                    // bit-for-bit; a nonzero offset wraps within the available
                    // pairs so a retry reads a disjoint pair.
                    let pair = if pc_pair_offset != 0 && pc_pairs > 0 {
                        (axis + pc_pair_offset) % pc_pairs
                    } else {
                        axis
                    };
                    let pc_a_idx = 2 * pair;
                    let pc_b_idx = 2 * pair + 1;
                    if pc_b_idx >= vt_rows {
                        break;
                    }
                    let pc_a = vt.row(pc_a_idx);
                    let pc_b = vt.row(pc_b_idx);
                    for row in 0..n_obs {
                        let mut a = 0.0_f64;
                        let mut b = 0.0_f64;
                        for col in 0..centered.ncols() {
                            a += centered[[row, col]] * pc_a[col];
                            b += centered[[row, col]] * pc_b[col];
                        }
                        // atan2 ∈ (-π, π]; map to phase ∈ [0, 1).
                        let phase = b.atan2(a) / two_pi;
                        let wrapped = phase - phase.floor();
                        out[[atom_idx, row, axis]] = wrapped;
                    }
                }
            }
            _ => {
                let avail = u_cols.min(s_vals.len());
                let k_cols = d.min(avail);
                // Rotate the score-column window by the multi-start offset (in
                // PC-pair units, mod the available components) so a reseed retry
                // reads distinct principal scores (the #976 distinct-basin lever).
                let base = if avail > 0 {
                    (2 * pc_pair_offset) % avail
                } else {
                    0
                };
                // Per-atom diversification (mirrors the #671 Periodic / Torus fix).
                // Without it EVERY Euclidean/Linear atom read the SAME leading
                // principal-score columns, so a K-atom dictionary seeded K
                // IDENTICAL atoms — a rank-deficient joint decoder whose undamped
                // Laplace factor is non-PD, which the seed-startup validation then
                // rejects with "no candidate seeds passed outer startup validation"
                // (the #1782 euclidean/linear failure; the #1094 multi-atom
                // euclidean numerical-fixed-point refusal is the same duplicate-atom
                // rank deficiency). Give atom `k` a DISJOINT window of principal
                // components `[k·d, k·d + d)` (wrapping when atoms outnumber the
                // available PCs), so distinct atoms read decorrelated scores and the
                // cross-atom Gram starts well-conditioned. `atom_idx == 0` keeps the
                // K=1 path byte-for-byte identical (offset 0).
                let atom_pc_offset = atom_idx.saturating_mul(d);
                let mut tmp = Array2::<f64>::zeros((n_obs, d));
                for col in 0..k_cols {
                    let src = if avail > 0 {
                        (base + atom_pc_offset + col) % avail
                    } else {
                        col
                    };
                    let s_col = s_vals[src];
                    for row in 0..n_obs {
                        tmp[[row, col]] = u[[row, src]] * s_col;
                    }
                }
                for col in 0..d {
                    let mut min_v = f64::INFINITY;
                    let mut max_v = f64::NEG_INFINITY;
                    for row in 0..n_obs {
                        let v = tmp[[row, col]];
                        if v < min_v {
                            min_v = v;
                        }
                        if v > max_v {
                            max_v = v;
                        }
                    }
                    let span = max_v - min_v;
                    if span > 0.0 {
                        for row in 0..n_obs {
                            out[[atom_idx, row, col]] = (tmp[[row, col]] - min_v) / span - 0.5;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// #2023 — DATA-ROW-anchored seed for flat (Euclidean/Linear) dead atoms.
///
/// The PCA reseed ([`sae_pca_seed_initial_coords_with_pc_offset`]) draws its
/// per-atom / per-retry diversity from principal-component pairs, of which there
/// are only `≈ min(n, p) / 2`. A co-collapsed dictionary leaves residual ≈
/// target, so every multi-start retry re-SVDs the same matrix and, once the
/// retry offset wraps the PC pool, re-reads the SAME leading components → the
/// joint LSQ relaxes into the SAME degenerate basin. That p/2 ceiling is the
/// co-collapse reseed-duplication spiral.
///
/// This anchors flat atom `slot` at a DISTINCT DATA ROW `anchor_rows[slot]`
/// (there are `n ≫ p` of them, so the diversity domain is unbounded) and seeds
/// each row `i`'s latent coordinate from that row's residual SIMILARITY to the
/// anchor row, `<residual_i, residual_anchor>`, min-max normalized to
/// `[-0.5, 0.5]` — the exact convention of the Euclidean/Linear branch of the
/// PCA seed, so a downstream refit sees a same-shaped (just more diverse) seed.
/// A `d`-dimensional atom spans `d` consecutive anchor rows (one per axis), so
/// its axes stay decorrelated. Returns the padded `(K, n, d_max)` coordinate
/// array; non-flat kinds are the caller's responsibility (it keeps them on the
/// PCA path). Rejects non-finite residuals up front.
pub fn sae_data_row_anchored_euclidean_coords(
    residual: ArrayView2<'_, f64>,
    atom_dim: &[usize],
    anchor_rows: &[usize],
) -> Result<Array3<f64>, String> {
    let k_atoms = atom_dim.len();
    let (n_obs, p_out) = residual.dim();
    let d_max = atom_dim.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((k_atoms, n_obs, d_max));
    if n_obs == 0 || p_out == 0 {
        return Ok(out);
    }
    if anchor_rows.len() != k_atoms {
        return Err(format!(
            "sae_data_row_anchored_euclidean_coords: anchor_rows len {} != atoms {k_atoms}",
            anchor_rows.len()
        ));
    }
    for ((row, col), &value) in residual.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "sae_data_row_anchored_euclidean_coords: residual must be finite; \
                 residual[{row}, {col}] = {value}"
            ));
        }
    }
    let mut sim = vec![0.0_f64; n_obs];
    for slot in 0..k_atoms {
        let d = atom_dim[slot];
        if d == 0 {
            continue;
        }
        let base = anchor_rows[slot] % n_obs;
        for axis in 0..d {
            // A distinct anchor row per axis so a d>1 atom spans d data rows and
            // its axes stay decorrelated (mirrors the PCA branch's disjoint-PC
            // window per axis).
            let anchor = (base + axis) % n_obs;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for i in 0..n_obs {
                let mut dot = 0.0_f64;
                for col in 0..p_out {
                    dot += residual[[i, col]] * residual[[anchor, col]];
                }
                sim[i] = dot;
                if dot < min_v {
                    min_v = dot;
                }
                if dot > max_v {
                    max_v = dot;
                }
            }
            let span = max_v - min_v;
            if span > 0.0 {
                for i in 0..n_obs {
                    out[[slot, i, axis]] = (sim[i] - min_v) / span - 0.5;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic `vt`-like matrix (rows = principal directions). The
    /// helper does not require orthonormal rows, only a non-degenerate span.
    fn make_vt(rows: usize, cols: usize) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                m[[r, c]] = ((r * 7 + c * 3 + 1) as f64).sin() + 0.1 * (r as f64 - c as f64);
            }
        }
        m
    }

    /// FIX #1: distinct `pc_pair_offset` ⇒ distinct random plane for a surplus
    /// atom, so successive #976 reseeds explore different basins. Also checks the
    /// FIX #2 Gram-Schmidt output is genuinely orthogonal.
    #[test]
    fn surplus_plane_differs_across_retries() {
        let vt = make_vt(6, 5);
        let k_atoms = 8;
        let atom_idx = 5;
        let (d1_0, d2_0, ok0) = surplus_phase_plane(vt.view(), atom_idx, 0, k_atoms);
        let (d1_1, _d2_1, ok1) = surplus_phase_plane(vt.view(), atom_idx, 1, k_atoms);
        assert!(ok0 && ok1, "well-conditioned vt should give a 2-D plane");
        let diff = d1_0
            .iter()
            .zip(d1_1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            diff > 1e-6,
            "distinct retry offsets must yield distinct dir1 (max diff {diff:.3e})"
        );
        let dot: f64 = d1_0.iter().zip(d2_0.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot.abs() < 1e-9,
            "dir2 must be orthogonal to dir1 (dot {dot:.3e})"
        );
        let n2: f64 = d2_0.dot(&d2_0).sqrt();
        assert!(
            (n2 - 1.0).abs() < 1e-9,
            "dir2 must be unit-normalized (norm {n2})"
        );
    }

    /// FIX #1: `pc_pair_offset == 0` reproduces the ORIGINAL splitmix64 key
    /// `((atom_idx) << 20) ^ pc` bit-for-bit, so the first attempt's primary
    /// phase axis is unchanged by the fix.
    #[test]
    fn surplus_plane_offset_zero_matches_original_key() {
        let vt = make_vt(5, 4);
        let k_atoms = 7;
        let atom_idx = 6;
        // Reference: replicate the pre-fix inline `dir1` computation exactly
        // (original key, same accumulation and normalization order).
        let mix = |mut z: u64| -> f64 {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        let ncols = vt.ncols();
        let mut a = Array1::<f64>::zeros(ncols);
        for pc in 0..vt.nrows() {
            let row_pc = vt.row(pc);
            let key = ((atom_idx as u64) << 20) ^ (pc as u64);
            let wa = mix(key);
            for c in 0..ncols {
                a[c] += wa * row_pc[c];
            }
        }
        let na = a.dot(&a).sqrt().max(1.0e-12);
        a.mapv_inplace(|v| v / na);
        let (d1, _d2, _ok) = surplus_phase_plane(vt.view(), atom_idx, 0, k_atoms);
        for (r, o) in d1.iter().zip(a.iter()) {
            assert!(
                (r - o).abs() < 1e-15,
                "offset-0 dir1 must match the original key bit-for-bit ({r} vs {o})"
            );
        }
    }

    /// FIX #2: a rank-1 PC span forces both random combinations collinear, so the
    /// orthogonalized residual is ~0 and the helper must report the 1-D fallback
    /// rather than a degenerate 2-plane that collapses atan2 to two phase points.
    #[test]
    fn surplus_plane_collinear_falls_back_to_1d() {
        let vt = make_vt(1, 4);
        let (d1, _d2, ok) = surplus_phase_plane(vt.view(), 3, 0, 5);
        assert!(!ok, "rank-1 span must report a 1-D (non-2-plane) result");
        let n1: f64 = d1.dot(&d1).sqrt();
        assert!(
            (n1 - 1.0).abs() < 1e-9 && n1.is_finite(),
            "dir1 must stay a finite unit vector (norm {n1})"
        );
    }

    /// End-to-end wiring: with more periodic atoms than PC pairs, offset 0 vs 1
    /// must move a SURPLUS atom's phase coords (distinct basins), every coord
    /// stays finite in `[0, 1)`, and offset 0 equals the no-offset entry point.
    #[test]
    fn surplus_periodic_seed_reproduces_and_diversifies() {
        // p = 4 ⇒ pc_pairs = 2; 5 periodic atoms ⇒ atoms 2..5 are surplus.
        let n_obs = 8;
        let p = 4;
        let mut zvals = Vec::with_capacity(n_obs * p);
        for r in 0..n_obs {
            for c in 0..p {
                zvals.push(
                    ((r as f64) * 0.9 + 1.0).sin() * ((c + 1) as f64)
                        + 0.3 * ((r * c) as f64).cos(),
                );
            }
        }
        let z = Array2::from_shape_vec((n_obs, p), zvals).unwrap();
        let kinds = vec![SaeAtomBasisKind::Periodic; 5];
        let dims = vec![1usize; 5];
        let s0 = sae_pca_seed_initial_coords_with_pc_offset(z.view(), &kinds, &dims, 0).unwrap();
        let s1 = sae_pca_seed_initial_coords_with_pc_offset(z.view(), &kinds, &dims, 1).unwrap();
        let plain = sae_pca_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
        assert_eq!(
            s0, plain,
            "offset-0 must equal the no-offset seed bit-for-bit"
        );
        for v in s0.iter().chain(s1.iter()) {
            assert!(
                v.is_finite() && *v >= 0.0 && *v < 1.0,
                "periodic phase must be finite in [0, 1): {v}"
            );
        }
        let surplus_atom = 3;
        let mut moved = 0.0_f64;
        for row in 0..n_obs {
            moved = moved.max((s0[[surplus_atom, row, 0]] - s1[[surplus_atom, row, 0]]).abs());
        }
        assert!(
            moved > 1e-6,
            "surplus atom must land on a distinct basin across retries (max move {moved:.3e})"
        );
    }

    /// #2023 — the data-row-anchored reseed must diversify BEYOND the ~p/2 PC
    /// pool: with `p` small (few PCs) but many distinct data-row anchors, the
    /// number of distinct seeds must exceed the PC-pair ceiling that caps the PCA
    /// reseed (the co-collapse cause). Coords stay in the `[-0.5, 0.5]` band.
    #[test]
    fn data_row_anchored_seed_diversifies_beyond_pc_pool() {
        let n = 64usize;
        let p = 4usize; // pc_pairs = min(n,p)/2 = 2 — the PCA reseed ceiling.
        // A low-rank-ish structured residual (the co-collapse regime).
        let mut residual = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                residual[[i, j]] = ((i * 7 + j) as f64).sin() + 0.25 * ((i + 3 * j) as f64).cos();
            }
        }
        let dims = vec![1usize];
        // p distinct anchors (> pc_pairs = 2) must yield > pc_pairs distinct seeds.
        let mut seeds = std::collections::HashSet::new();
        for anchor in 0..p {
            let s = sae_data_row_anchored_euclidean_coords(residual.view(), &dims, &[anchor])
                .expect("data-row seed");
            for v in s.iter() {
                assert!(
                    v.is_finite() && *v >= -0.5 - 1e-12 && *v <= 0.5 + 1e-12,
                    "seed coord must stay in [-0.5, 0.5]: {v}"
                );
            }
            let key: Vec<i64> = (0..n).map(|i| (s[[0, i, 0]] * 1e6).round() as i64).collect();
            seeds.insert(key);
        }
        assert!(
            seeds.len() >= p,
            "data-row anchors must give ≥ p distinct seeds; the PCA reseed caps at \
             pc_pairs = {} — got {} distinct",
            (n.min(p)) / 2,
            seeds.len()
        );
    }
}
