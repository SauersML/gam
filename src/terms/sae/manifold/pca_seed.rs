//! Production PCA-based per-atom chart seeding for the SAE manifold fit.
//! Moved out of the (test-only) `tests` module so the production FFI seed
//! path can call it in release builds.

use super::SaeAtomBasisKind;
use crate::linalg::faer_ndarray::FaerSvd;
use ndarray::{Array1, Array2, Array3, ArrayView2};

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
                if vt_rows >= 2 {
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
                        (0, 1)
                    };
                    let pc1 = vt.row(pc1_row.min(vt_rows - 1));
                    let pc2 = vt.row(pc2_row.min(vt_rows - 1));
                    for row in 0..n_obs {
                        let mut a = 0.0_f64;
                        let mut b = 0.0_f64;
                        for col in 0..centered.ncols() {
                            a += centered[[row, col]] * pc1[col];
                            b += centered[[row, col]] * pc2[col];
                        }
                        let phase = b.atan2(a) / two_pi;
                        out[[atom_idx, row, 0]] = phase - phase.floor();
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
                    // (same #976 distinct-basin lever as the periodic arm).
                    let pair = if pc_pairs > 0 {
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
                let mut tmp = Array2::<f64>::zeros((n_obs, d));
                for col in 0..k_cols {
                    let src = if avail > 0 { (base + col) % avail } else { col };
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
