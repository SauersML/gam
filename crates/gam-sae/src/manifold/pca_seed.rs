//! Production PCA-based per-atom chart seeding for the SAE manifold fit.
//! Moved out of the (test-only) `tests` module so the production FFI seed
//! path can call it in release builds.

use super::SaeAtomBasisKind;
use faer::Side;
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

/// Residual-norm floor below which the surplus atom's second phase axis is
/// treated as collinear with the first (a degenerate 2-plane). Since both
/// candidate directions enter [`surplus_phase_plane`] unit-normalized, the
/// post-orthogonalization residual norm equals the sine of the angle between
/// them, so this is an absolute "how far from collinear" threshold in `[0, 1]`.
///
/// PRICED (#2071): `sin θ = 1e-6` ⟺ `θ ≈ 1e-6 rad` (~6e-5°). A second axis within
/// a microradian of the first spans no genuine plane, and the 2-plane it forms
/// has conditioning `~ 1/sin θ ≈ 1e6`. The scale is bounded on both sides: the
/// residual can only be resolved above the f64 rounding of the unit-vector dot
/// product (`~1e-16`), so `1e-6` sits ~10 orders above that numerical-noise floor;
/// and it is ~6 orders below unity, so it rejects only planes that are collinear
/// for all practical purposes. What breaks at 10×: at `1e-5` genuinely-tight but
/// real 2-planes (θ ≈ 0.6°) start being discarded to the 1-D fallback; at `1e-7`
/// planes conditioned at `~1e7` enter the phase solve and degrade its accuracy.
const SURPLUS_DIR_FLOOR: f64 = 1.0e-6;

/// Golden-ratio conjugate `φ⁻¹`. Additive step of a low-discrepancy (mod 1)
/// rotation that folds the #976 multi-start retry index into the periodic
/// `phase_offset`, so successive reseeds place the SAME surplus atom at a
/// well-spread DISTINCT circle phase. `pc_pair_offset == 0` contributes exactly
/// `0.0`, leaving the original `atom_idx / k_atoms` offset bit-for-bit.
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;

/// Explicit memory budget for the topology-aware seed's dense graph-Laplacian
/// eigensolve. The subsample size is *derived* from this budget rather than
/// fixed to a tuned point count (#2065): the seed builds two dense `m×m` f64
/// matrices — the symmetric normalized Laplacian and its eigenvectors — so the
/// largest `m` that fits `B` bytes is `sqrt(B / 16)`. At the default 256 MiB
/// this yields `m = 4096` (the historical value), but now as a stated resource
/// choice whose O(m³) eigensolve cost is bounded by the budget, not by a fudge.
const TOPOLOGY_SEED_LAPLACIAN_BUDGET_BYTES: usize = 256 << 20;

/// Largest subsample size whose two dense f64 `m×m` matrices fit
/// [`TOPOLOGY_SEED_LAPLACIAN_BUDGET_BYTES`] (`16·m²` bytes total).
fn topology_seed_max_points() -> usize {
    let m = ((TOPOLOGY_SEED_LAPLACIAN_BUDGET_BYTES / 16) as f64).sqrt() as usize;
    m.max(4)
}

/// Neighborhood degree `k` for the topology seed's kNN graph, *derived* from the
/// atom's intrinsic dimension and the sample size rather than a tuned scalar
/// (#2065). Two independent requirements set the floor:
///   * geometric faithfulness — the graph Laplacian approximates the
///     Laplace–Beltrami operator only if each node links its full local tangent
///     star, i.e. `≥ 2d+1` neighbours span a `d`-simplex neighborhood; and
///   * connectivity — a kNN graph on `n` points is connected with high
///     probability only for `k ≳ log₂ n` (Penrose, random geometric graphs).
/// Take the larger so the harmonic coordinates are both faithful and connected.
/// (For the default full subsample `m = 4096`, `d ≤ 2` ⇒ `k = 12`, unchanged.)
fn topology_seed_knn(n_points: usize, d_atom: usize) -> usize {
    let tangent_floor = 2 * d_atom + 1;
    let connectivity_floor = (n_points.max(2) as f64).log2().ceil() as usize;
    tangent_floor.max(connectivity_floor).max(2)
}

fn is_curved_kind(kind: &SaeAtomBasisKind) -> bool {
    matches!(
        kind,
        SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus | SaeAtomBasisKind::Sphere
    )
}

fn topology_seed_subsample(n_obs: usize) -> Vec<usize> {
    let cap = topology_seed_max_points();
    if n_obs <= cap {
        return (0..n_obs).collect();
    }
    let mut rows = Vec::with_capacity(cap);
    for i in 0..cap {
        rows.push(i * n_obs / cap);
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
    let d_max = atom_dim.iter().copied().max().unwrap_or(1).max(1);
    // Neighborhood degree derived from intrinsic dimension + connectivity
    // (#2065), capped at the subsample size.
    let k = topology_seed_knn(m, d_max).min(m - 1);
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
    // Degrees once (O(m²)), not per nonzero entry (O(m) row-sum each, O(m²·k)
    // total): the Laplacian reads √(deg_i·deg_j) for every kNN edge.
    let mut deg = vec![0.0_f64; m];
    for i in 0..m {
        deg[i] = w.row(i).sum();
        if deg[i] <= 0.0 || !deg[i].is_finite() {
            return Ok(None);
        }
    }
    let mut lap = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        lap[[i, i]] = 1.0;
        let inv_sqrt = 1.0 / deg[i].sqrt();
        for j in 0..m {
            if i != j && w[[i, j]] != 0.0 {
                lap[[i, j]] = -w[[i, j]] * inv_sqrt / deg[j].sqrt();
            }
        }
    }
    let (evals, evecs) = lap
        .eigh(Side::Lower)
        .map_err(|err| format!("topology_seed: graph Laplacian eigensolve failed: {err:?}"))?;
    if evals.len() < 3 {
        return Ok(None);
    }
    let mut out = Array3::<f64>::zeros((basis_kinds.len(), z.nrows(), d_max));
    // Inverse-distance interpolation onto out-of-subsample rows uses the `d+1`
    // nearest subsample points — the vertices of a barycentric simplex for a
    // `d`-dimensional atom chart — derived from the atom dimension rather than a
    // hardcoded 3-NN (#2065). Bounded by the subsample size `m`.
    //
    // The neighbor set depends only on the ROW, not on which harmonic is being
    // interpolated, so the `O(m·p)` nearest-subsample scan is done ONCE per row
    // here rather than once per `(atom, chart function, row)` interp call — the
    // per-call rescan multiplied the dominant seed cost by `Σ_atoms need_fns`
    // (≈ 2K at K curved atoms) for bit-identical output.
    let interp_k = (d_max + 1).max(2).min(m);
    let mut pos_of_row = vec![usize::MAX; z.nrows()];
    for (pos, &r) in rows.iter().enumerate() {
        pos_of_row[r] = pos;
    }
    let mut row_neighbors: Vec<Vec<(f64, usize)>> = vec![Vec::new(); z.nrows()];
    for (row, neighbors) in row_neighbors.iter_mut().enumerate() {
        if pos_of_row[row] != usize::MAX {
            continue;
        }
        let mut best: Vec<(f64, usize)> = vec![(f64::INFINITY, 0usize); interp_k];
        for (i, &r) in rows.iter().enumerate() {
            let d = squared_distance_rows(z, row, r);
            if d < best[interp_k - 1].0 {
                best[interp_k - 1] = (d, i);
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }
        // Store the inverse-distance weights directly; the normalizing sum is
        // fn-independent too.
        *neighbors = best
            .into_iter()
            .map(|(d, i)| (1.0 / d.max(1.0e-24), i))
            .collect();
    }
    let interp = |sample_values: &Array1<f64>, row: usize| -> f64 {
        let pos = pos_of_row[row];
        if pos != usize::MAX {
            return sample_values[pos];
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for &(ww, i) in &row_neighbors[row] {
            num += ww * sample_values[i];
            den += ww;
        }
        num / den
    };
    // Harmonic chart functions: graph-Laplacian eigenvectors, excluding the
    // constant Fiedler-0 column. Each is a function over the `m` subsample rows.
    let harmonic: Vec<ArrayView1<'_, f64>> = (1..evecs.ncols()).map(|c| evecs.column(c)).collect();
    let n_harm = harmonic.len();
    let starts = topology_seed_harmonic_starts(basis_kinds, atom_dim);
    for atom_idx in 0..basis_kinds.len() {
        let d = atom_dim[atom_idx];
        let kind = &basis_kinds[atom_idx];
        let need = topology_seed_chart_need(kind, d);
        if need == 0 || n_harm == 0 {
            continue;
        }
        // #1893 UNIVERSAL overcomplete seeding on the topology-seed path (which
        // previously returned early with a wrap that gave every sphere / flat atom
        // an IDENTICAL chart, and every surplus circle / torus atom a duplicate
        // pair). An atom takes a DISJOINT harmonic window `[start, start+need)`
        // only when that window fits the available harmonics and no reseed rotation
        // is requested; otherwise it gets a DISTINCT atom-keyed generic combination
        // of ALL harmonics, so K ≫ p atoms never share a chart and a co-collapse
        // reseed (retry > 0) lands every atom on a different basin. Atom 0 at
        // retry 0 keeps its original leading-harmonic window bit-for-bit.
        let start = starts[atom_idx];
        let canonical = pc_pair_offset == 0 && start + need <= n_harm;
        let fns: Vec<Array1<f64>> = if canonical {
            (0..need).map(|i| harmonic[start + i].to_owned()).collect()
        } else {
            generic_ortho_combos(&harmonic, atom_idx, pc_pair_offset, basis_kinds.len(), need)
        };
        if fns.is_empty() {
            continue;
        }
        match kind {
            SaeAtomBasisKind::Periodic => {
                if fns.len() < 2 {
                    continue;
                }
                for row in 0..z.nrows() {
                    let phase =
                        interp(&fns[1], row).atan2(interp(&fns[0], row)) / std::f64::consts::TAU;
                    out[[atom_idx, row, 0]] = phase - phase.floor();
                }
                // Axes beyond the leading circle are flat coordinates, mirroring the
                // linear PCA Periodic seed's `1..d` min-max branch: a periodic atom of
                // intrinsic dim `d > 1` is one phase plane plus `d − 1` Euclidean axes.
                // The phase consumed `fns[0..2]`, so the flat axes read the NEXT
                // harmonics (`fns[axis + 1]`). Without this the topology seed left
                // every axis past 0 at zero, collapsing a `d > 1` seed to rank 1 (so a
                // higher latent dim was a bit-for-bit no-op).
                for axis in 1..d {
                    let Some(values) = fns.get(axis + 1) else {
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
                            out[[atom_idx, row, axis]] = (interp(values, row) - lo) / span - 0.5;
                        }
                    }
                }
            }
            SaeAtomBasisKind::Torus => {
                for axis in 0..d {
                    let (Some(a), Some(b)) = (fns.get(2 * axis), fns.get(2 * axis + 1)) else {
                        break;
                    };
                    for row in 0..z.nrows() {
                        let phase = interp(b, row).atan2(interp(a, row)) / std::f64::consts::TAU;
                        out[[atom_idx, row, axis]] = phase - phase.floor();
                    }
                }
            }
            SaeAtomBasisKind::Sphere => {
                if fns.len() < 3 {
                    continue;
                }
                for row in 0..z.nrows() {
                    let x = interp(&fns[0], row);
                    let y = interp(&fns[1], row);
                    let zz = interp(&fns[2], row);
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
                    let Some(values) = fns.get(axis) else {
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
                            out[[atom_idx, row, axis]] = (interp(values, row) - lo) / span - 0.5;
                        }
                    }
                }
            }
        }
    }
    Ok(Some(out))
}

// Chart functions needed by one atom in the topology-seed path: one phase plane
// per circle axis, a 3-frame for the sphere, and one coordinate per flat axis.
fn topology_seed_chart_need(kind: &SaeAtomBasisKind, d: usize) -> usize {
    match kind {
        // Two harmonics for the leading phase plane, plus one flat harmonic per
        // extra axis so a `d > 1` periodic atom's Euclidean axes are seeded (not
        // left at zero). `d == 1` keeps the historical need of 2.
        SaeAtomBasisKind::Periodic => 2 + d.max(1).saturating_sub(1),
        SaeAtomBasisKind::Torus => 2 * d.max(1),
        SaeAtomBasisKind::Sphere => 3,
        _ => d.max(1),
    }
}

fn topology_seed_harmonic_starts(
    basis_kinds: &[SaeAtomBasisKind],
    atom_dim: &[usize],
) -> Vec<usize> {
    let mut next = 0usize;
    let mut starts = Vec::with_capacity(basis_kinds.len());
    for (kind, &d) in basis_kinds.iter().zip(atom_dim.iter()) {
        starts.push(next);
        next = next.saturating_add(topology_seed_chart_need(kind, d));
    }
    starts
}

/// splitmix64 → pseudo-random weight in `[-1, 1]`, keyed deterministically. No
/// RNG crate: reproducible run-to-run and across thread/device counts.
fn splitmix_unit(mut z: u64) -> f64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z as f64 / u64::MAX as f64) * 2.0 - 1.0
}

/// Deterministic distinct XOR salt per chart-function slot. Slots 0 and 1 keep the
/// exact salts the original two-direction [`surplus_phase_plane`] used, so its
/// output (and every pinned offset-0 contract) is preserved bit-for-bit; higher
/// slots (needed for torus `2d`-planes, sphere 3-frames, and `d`-axis flat charts)
/// get a splitmix-derived salt so all `m` combinations stay mutually distinct.
fn slot_salt(slot: usize) -> u64 {
    match slot {
        0 => 0,
        1 => 0xD1B54A32D192ED03,
        _ => {
            let mut z = (slot as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xA0761D6478BD642F;
            z = (z ^ (z >> 32)).wrapping_mul(0xE7037ED1A0B428DB);
            z ^ (z >> 29)
        }
    }
}

/// Atom-keyed generic orthonormal combinations of `sources` — the universal
/// overcomplete (#1893, K ≫ p) seeding primitive shared by every basis kind and
/// every seed path (linear PCA + topology-curved) so surplus atoms never share a
/// chart. Each source is one equal-length direction (a `vt` principal-direction
/// row in feature space, or a graph-Laplacian eigenfunction over the topology
/// subsample). Forms `m` splitmix64-weighted combinations keyed by
/// `(atom_idx, retry, k_atoms)` — distinct per atom and per multi-start retry —
/// then Gram-Schmidt orthonormalizes them with the [`SURPLUS_DIR_FLOOR`]
/// collinearity floor. The returned `Vec` is shorter than `m` only when the source
/// span's rank is below `m` (a genuinely low-rank residual); callers treat a short
/// return as "this axis/frame is unavailable" and fall back or skip. Linear span,
/// so the Welch bound applies: for K ≫ p the planes cannot be mutually orthogonal,
/// but they are pairwise DISTINCT (the goal is non-duplication, not orthogonality).
fn generic_ortho_combos(
    sources: &[ArrayView1<'_, f64>],
    atom_idx: usize,
    retry: usize,
    k_atoms: usize,
    m: usize,
) -> Vec<Array1<f64>> {
    if sources.is_empty() || m == 0 {
        return Vec::new();
    }
    let len = sources[0].len();
    // Base-`k_atoms` mix of (atom_idx, retry) before the `<< 20` spread. At
    // `retry == 0` slot 0 reduces to `(atom_idx << 20) ^ pc` — the original
    // surplus-plane key, bit-for-bit — so the first attempt is unchanged.
    let base = ((atom_idx as u64) + (retry as u64) * (k_atoms as u64)) << 20;
    let mut out: Vec<Array1<f64>> = Vec::with_capacity(m);
    for slot in 0..m {
        let salt = slot_salt(slot);
        let mut v = Array1::<f64>::zeros(len);
        for (pc, src) in sources.iter().enumerate() {
            let w = splitmix_unit((base ^ (pc as u64)) ^ salt);
            v.scaled_add(w, src);
        }
        // Gram-Schmidt against the already-accepted directions so the m axes are
        // not near-collinear (a near-parallel plane makes `atan2` take only two
        // values and kills the diversity this exists to create).
        for u in &out {
            let proj = v.dot(u);
            v.scaled_add(-proj, u);
        }
        let nv = v.dot(&v).sqrt();
        if nv > SURPLUS_DIR_FLOOR {
            v.mapv_inplace(|x| x / nv);
            out.push(v);
        }
    }
    out
}

/// Two-direction specialization of [`generic_ortho_combos`] for a surplus periodic
/// atom's phase plane. Returns `(dir1, dir2, two_dimensional)`; when the source
/// span is effectively 1-D the second direction collapses below
/// [`SURPLUS_DIR_FLOOR`] and `two_dimensional` is `false`, so the caller falls back
/// to the 1-D span phase path rather than feeding `atan2` a degenerate plane.
fn surplus_phase_plane(
    vt: ArrayView2<'_, f64>,
    atom_idx: usize,
    pc_pair_offset: usize,
    k_atoms: usize,
) -> (Array1<f64>, Array1<f64>, bool) {
    let ncols = vt.ncols();
    let sources: Vec<ArrayView1<'_, f64>> = (0..vt.nrows()).map(|r| vt.row(r)).collect();
    let dirs = generic_ortho_combos(&sources, atom_idx, pc_pair_offset, k_atoms, 2);
    match dirs.len() {
        n if n >= 2 => {
            let mut it = dirs.into_iter();
            (it.next().unwrap(), it.next().unwrap(), true)
        }
        1 => (
            dirs.into_iter().next().unwrap(),
            Array1::zeros(ncols),
            false,
        ),
        _ => (Array1::zeros(ncols), Array1::zeros(ncols), false),
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
                // Seed the sphere chart from a 3-frame: drop the centred response
                // onto three ambient directions, unit-normalise, and read off
                // (lat, lon), placing every row on the chart with
                // `lat ∈ (-π/2, π/2)` and `lon ∈ (-π, π]`.
                let n_pc = vt_rows.min(3);
                if n_pc == 0 {
                    continue;
                }
                // #1893 — the pre-#1893 sphere seed read the leading 3 PCs for
                // EVERY atom (its window did not depend on `atom_idx`), so a K-atom
                // sphere dictionary was K IDENTICAL charts. Give atom `k` a DISJOINT
                // leading window `[3k, 3k+3)` when it fits and no reseed rotation is
                // asked; otherwise (K ≫ p surplus, or a #976 retry) a DISTINCT
                // atom-keyed generic 3-frame in the PC span. Atom 0 at offset 0 keeps
                // the original top-3-PC frame bit-for-bit.
                let atom_start = atom_idx.saturating_mul(3);
                let canonical = pc_pair_offset == 0 && atom_start + 3 <= vt_rows;
                let frame: Vec<Array1<f64>> = if canonical {
                    (0..n_pc)
                        .map(|i| vt.row(atom_start + i).to_owned())
                        .collect()
                } else {
                    let sources: Vec<ArrayView1<'_, f64>> =
                        (0..vt_rows).map(|r| vt.row(r)).collect();
                    let dirs = generic_ortho_combos(&sources, atom_idx, pc_pair_offset, k_atoms, 3);
                    if dirs.is_empty() {
                        continue;
                    }
                    dirs
                };
                for row in 0..n_obs {
                    let mut amb = [0.0_f64; 3];
                    for (i, pc) in frame.iter().enumerate().take(3) {
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
                // Seed each torus axis from a phase plane: project the centred
                // response onto two ambient directions and read `atan2`, normalised
                // to `[0, 1)`.
                //
                // #1893 — the pre-#1893 torus seed set `pair == axis` for EVERY
                // atom (no `atom_idx` dependence), so a K-atom torus dictionary was
                // K IDENTICAL charts. Give atom `k` a DISJOINT block of PC pairs
                // `[k·d, k·d + d)` when it fits and no reseed rotation is asked;
                // otherwise (K ≫ p surplus, or a #976 retry) DISTINCT atom-keyed
                // generic `2d`-plane. Atom 0 at offset 0 keeps its original
                // leading-PC-pair-per-axis seed bit-for-bit.
                let pc_pairs = vt_rows / 2;
                let atom_start = atom_idx.saturating_mul(d);
                let canonical = pc_pair_offset == 0 && pc_pairs > 0 && atom_start + d <= pc_pairs;
                let generic_dirs: Vec<Array1<f64>> = if canonical {
                    Vec::new()
                } else {
                    let sources: Vec<ArrayView1<'_, f64>> =
                        (0..vt_rows).map(|r| vt.row(r)).collect();
                    generic_ortho_combos(&sources, atom_idx, pc_pair_offset, k_atoms, 2 * d)
                };
                for axis in 0..d {
                    let (pc_a, pc_b): (Array1<f64>, Array1<f64>) = if canonical {
                        let pair = atom_start + axis;
                        let pc_b_idx = 2 * pair + 1;
                        if pc_b_idx >= vt_rows {
                            break;
                        }
                        (vt.row(2 * pair).to_owned(), vt.row(pc_b_idx).to_owned())
                    } else {
                        match (generic_dirs.get(2 * axis), generic_dirs.get(2 * axis + 1)) {
                            (Some(a), Some(b)) => (a.clone(), b.clone()),
                            // Ran out of independent generic directions (span rank
                            // < 2·axis): leave the remaining axes at 0.
                            _ => break,
                        }
                    };
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
                // Per-atom diversification: give atom `k` a DISJOINT window of
                // principal components `[k·d, k·d + d)`. Without it EVERY
                // Euclidean/Linear atom read the SAME leading principal-score
                // columns, so a K-atom dictionary seeded K IDENTICAL atoms — a
                // rank-deficient joint decoder whose undamped Laplace factor is
                // non-PD, which the seed-startup validation then rejects with "no
                // candidate seeds passed outer startup validation" (the #1782
                // euclidean/linear failure; the #1094 multi-atom refusal is the
                // same duplicate-atom rank deficiency).
                //
                // #1893 — the disjoint window only holds while `k·d + d ≤ avail`.
                // Once atoms outrun the ≈`min(n, p)` principal scores the old scheme
                // WRAPPED (`% avail`), re-reading the SAME score column and seeding a
                // DUPLICATE design (exact Hessian null). For every SURPLUS atom
                // (window overruns `avail`) or a #976 reseed retry, project the
                // centred response onto a DISTINCT atom-keyed generic frame in the PC
                // span instead, so K ≫ p flat atoms stay pairwise distinct. Atom 0 at
                // offset 0 keeps the K=1 path byte-for-byte identical.
                let atom_start = atom_idx.saturating_mul(d);
                let canonical = pc_pair_offset == 0 && avail > 0 && atom_start + d <= avail;
                if canonical {
                    let k_cols = d.min(avail);
                    let mut tmp = Array2::<f64>::zeros((n_obs, d));
                    for col in 0..k_cols {
                        let src = atom_start + col;
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
                } else {
                    let sources: Vec<ArrayView1<'_, f64>> =
                        (0..vt_rows).map(|r| vt.row(r)).collect();
                    let dirs = generic_ortho_combos(&sources, atom_idx, pc_pair_offset, k_atoms, d);
                    for (col, dir) in dirs.iter().enumerate().take(d) {
                        let mut proj = Array1::<f64>::zeros(n_obs);
                        for row in 0..n_obs {
                            let mut acc = 0.0_f64;
                            for c in 0..centered.ncols() {
                                acc += centered[[row, c]] * dir[c];
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
                                out[[atom_idx, row, col]] = (proj[row] - min_v) / span - 0.5;
                            }
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

/// Deterministic farthest-point landmark selection (#2023 Tier-1 pattern): the
/// first landmark is row 0, then each next landmark maximizes the minimum
/// ambient distance to the chosen set, ties broken by the smaller row index. No
/// RNG; the choice is a pure function of the data and the requested count.
fn farthest_point_landmark_rows(z: ArrayView2<'_, f64>, m: usize) -> Vec<usize> {
    let n = z.nrows();
    let p = z.ncols();
    let dist2 = |a: usize, b: usize| -> f64 {
        (0..p)
            .map(|c| {
                let x = z[[a, c]] - z[[b, c]];
                x * x
            })
            .sum::<f64>()
    };
    let mut chosen = Vec::with_capacity(m);
    chosen.push(0usize);
    let mut min_d2: Vec<f64> = (0..n).map(|i| dist2(i, 0)).collect();
    while chosen.len() < m {
        let mut best_row = usize::MAX;
        let mut best_val = -1.0_f64;
        for (row, &value) in min_d2.iter().enumerate() {
            if value > best_val {
                best_val = value;
                best_row = row;
            }
        }
        if best_row == usize::MAX {
            break;
        }
        chosen.push(best_row);
        for (row, slot) in min_d2.iter_mut().enumerate() {
            *slot = slot.min(dist2(row, best_row));
        }
    }
    chosen
}

/// Deterministic intrinsic-metric seed coordinates (issue #2240 reframe /
/// #2280 mechanism 1): a neighborhood-graph geodesic embedding — Isomap, i.e.
/// kNN graph → shortest-path geodesics → classical MDS — into `d` dimensions, to
/// be raced in the seed race against the global-linear PCA seed under the same
/// REML evidence.
///
/// A global linear projection (PCA) is NON-INJECTIVE on any embedding that folds
/// back on itself — a rolled sheet, a spiral, an entangled product — so distinct
/// manifold points collapse onto the same coordinate and the downstream fit can
/// only average them (held-out reconstruction collapses on the overlap). The
/// intrinsic geodesic metric unfolds the fold: graph-neighbors are manifold-
/// neighbors, so the embedding is injective exactly where PCA is not. On an
/// already-flat cluster the geodesic metric agrees with the Euclidean one, so
/// this ties PCA and PCA wins on evidence/cost; the intrinsic seed earns its
/// place only when folding breaks PCA. No shape-specific logic.
///
/// **Determinism (fleet law):** farthest-point landmarks from a fixed start, kNN
/// with index tie-breaks, Floyd–Warshall geodesics, the faer symmetric
/// eigendecomposition, and sign-pinned embedding axes are all deterministic, so
/// the seed is bit-reproducible run-to-run and across thread/device counts.
///
/// Cost is bounded by capping the geodesic/MDS core at `LANDMARK_CAP` landmarks
/// (`O(m³)` Floyd–Warshall + `O(m³)` eigendecomposition); at or below the cap
/// every row is its own landmark (exact Isomap). Above the cap, non-landmark
/// rows take their nearest landmark's coordinates — a coarse but deterministic
/// seed that the joint fit refines (a smooth Landmark-MDS out-of-sample
/// placement is the follow-up).
pub fn intrinsic_metric_seed_coords(
    z: ArrayView2<'_, f64>,
    d: usize,
    n_neighbors: usize,
) -> Result<Array2<f64>, String> {
    use std::cmp::Ordering;
    let n = z.nrows();
    let p = z.ncols();
    if d == 0 {
        return Err("intrinsic_metric_seed_coords: d must be >= 1".to_string());
    }
    if n_neighbors < 1 {
        return Err("intrinsic_metric_seed_coords: n_neighbors must be >= 1".to_string());
    }
    if n < d + 1 {
        return Err(format!(
            "intrinsic_metric_seed_coords: need at least d+1={} rows, got {n}",
            d + 1
        ));
    }
    for ((row, col), &value) in z.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "intrinsic_metric_seed_coords: Z must be finite; Z[{row}, {col}] = {value}"
            ));
        }
    }
    let dist2 = |a: usize, b: usize| -> f64 {
        (0..p)
            .map(|c| {
                let x = z[[a, c]] - z[[b, c]];
                x * x
            })
            .sum::<f64>()
    };

    const LANDMARK_CAP: usize = 700;
    let m = n.min(LANDMARK_CAP);
    let landmarks: Vec<usize> = if n <= LANDMARK_CAP {
        (0..n).collect()
    } else {
        farthest_point_landmark_rows(z, m)
    };
    let cmp = |x: f64, y: f64| x.partial_cmp(&y).unwrap_or(Ordering::Equal);

    // Symmetric kNN graph over the landmarks; edge weight = ambient distance.
    let mut geo = vec![vec![f64::INFINITY; m]; m];
    for (i, g) in geo.iter_mut().enumerate() {
        g[i] = 0.0;
    }
    for i in 0..m {
        let li = landmarks[i];
        let mut order: Vec<usize> = (0..m).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| {
            cmp(dist2(li, landmarks[a]), dist2(li, landmarks[b])).then(a.cmp(&b))
        });
        for &j in order.iter().take(n_neighbors) {
            let w = dist2(li, landmarks[j]).sqrt();
            if w < geo[i][j] {
                geo[i][j] = w;
                geo[j][i] = w;
            }
        }
    }

    // Bridge disconnected components by the nearest cross-component landmark
    // pair (deterministic) so the geodesic metric is finite everywhere.
    loop {
        // component labels via flood fill over finite edges
        let mut comp = vec![usize::MAX; m];
        let mut n_comp = 0usize;
        for start in 0..m {
            if comp[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            comp[start] = n_comp;
            while let Some(u) = stack.pop() {
                for v in 0..m {
                    if u != v && geo[u][v].is_finite() && comp[v] == usize::MAX {
                        comp[v] = n_comp;
                        stack.push(v);
                    }
                }
            }
            n_comp += 1;
        }
        if n_comp <= 1 {
            break;
        }
        // nearest cross-component pair
        let mut best = (usize::MAX, usize::MAX);
        let mut best_d = f64::INFINITY;
        for i in 0..m {
            for j in (i + 1)..m {
                if comp[i] != comp[j] {
                    let dd = dist2(landmarks[i], landmarks[j]);
                    if dd < best_d {
                        best_d = dd;
                        best = (i, j);
                    }
                }
            }
        }
        let (i, j) = best;
        let w = best_d.sqrt();
        geo[i][j] = w;
        geo[j][i] = w;
    }

    // Floyd–Warshall all-pairs shortest paths = geodesic distances.
    for k in 0..m {
        for i in 0..m {
            let dik = geo[i][k];
            if dik.is_finite() {
                for j in 0..m {
                    let alt = dik + geo[k][j];
                    if alt < geo[i][j] {
                        geo[i][j] = alt;
                    }
                }
            }
        }
    }

    // Classical MDS: double-center the squared geodesic matrix, eigendecompose.
    let mut row_mean = vec![0.0_f64; m];
    let mut grand = 0.0_f64;
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..m {
            s += geo[i][j] * geo[i][j];
        }
        row_mean[i] = s / m as f64;
        grand += row_mean[i];
    }
    grand /= m as f64;
    let mut b = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let g = geo[i][j];
            // symmetric double-centering; average i,j to kill rounding asymmetry
            b[[i, j]] = -0.5 * (g * g - row_mean[i] - row_mean[j] + grand);
        }
    }
    for i in 0..m {
        for j in (i + 1)..m {
            let a = 0.5 * (b[[i, j]] + b[[j, i]]);
            b[[i, j]] = a;
            b[[j, i]] = a;
        }
    }
    let (vals, vecs) = b
        .eigh(Side::Lower)
        .map_err(|error| format!("intrinsic_metric_seed_coords: eigh failed: {error:?}"))?;
    // Top-d eigenpairs by DESCENDING eigenvalue (classical MDS keeps the largest
    // positive eigenvalues — the metric directions), index tie-break.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b2| cmp(vals[b2], vals[a]).then(a.cmp(&b2)));
    let mut land_emb = Array2::<f64>::zeros((m, d));
    for (col, &ei) in idx.iter().take(d).enumerate() {
        let scale = vals[ei].max(0.0).sqrt();
        for i in 0..m {
            land_emb[[i, col]] = vecs[[i, ei]] * scale;
        }
    }

    let mut out = Array2::<f64>::zeros((n, d));
    if n <= LANDMARK_CAP {
        out.assign(&land_emb);
    } else {
        for row in 0..n {
            let mut best_idx = 0usize;
            let mut best_d = f64::INFINITY;
            for (li_idx, &li) in landmarks.iter().enumerate() {
                let dd = dist2(row, li);
                if dd < best_d {
                    best_d = dd;
                    best_idx = li_idx;
                }
            }
            for col in 0..d {
                out[[row, col]] = land_emb[[best_idx, col]];
            }
        }
    }

    // Sign-pin each axis (make its largest-magnitude entry positive) so the
    // eigenvector sign freedom does not make the seed run-dependent.
    for col in 0..d {
        let mut max_row = 0usize;
        let mut max_abs = 0.0_f64;
        for row in 0..n {
            let a = out[[row, col]].abs();
            if a > max_abs {
                max_abs = a;
                max_row = row;
            }
        }
        if out[[max_row, col]] < 0.0 {
            for row in 0..n {
                out[[row, col]] = -out[[row, col]];
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

    #[test]
    fn topology_harmonic_windows_are_cumulative_for_mixed_kinds() {
        let kinds = vec![
            SaeAtomBasisKind::Torus,
            SaeAtomBasisKind::Periodic,
            SaeAtomBasisKind::Sphere,
            SaeAtomBasisKind::Linear,
        ];
        let dims = vec![2usize, 1, 2, 3];
        assert_eq!(
            topology_seed_harmonic_starts(&kinds, &dims),
            vec![0, 4, 6, 9],
            "mixed atom kinds must allocate canonical harmonic windows cumulatively; \
             atom_idx * per-atom-need overlaps when need varies"
        );
    }

    /// #2240 reframe falsifier: on a genuinely ROLLED sheet (a swiss roll whose
    /// layers overlap in every 2-plane), the global-linear PCA seed is
    /// non-injective and its held-out reconstruction collapses, while the
    /// intrinsic-metric (geodesic Isomap) seed unfolds the roll and recovers it.
    /// This is the general seed-geometry fix — no roll-specific code — and the
    /// property the seed race exploits: race intrinsic vs PCA, intrinsic wins
    /// exactly when folding breaks PCA (an already-flat cluster ties, and PCA
    /// wins there on cost/evidence). Objective quality, no reference tool.
    #[test]
    fn intrinsic_seed_unfolds_swiss_roll_where_pca_collapses() {
        use gam_linalg::faer_ndarray::{fast_ata, fast_atb, FaerCholesky};

        // Deterministic ~2-turn swiss roll grid, embedded in R^3.
        let (n_t, n_h) = (45usize, 10usize);
        let n = n_t * n_h;
        let mut z = Array2::<f64>::zeros((n, 3));
        for ti in 0..n_t {
            let t = 1.2 * std::f64::consts::PI
                + (3.2 - 1.2) * std::f64::consts::PI * ti as f64 / (n_t - 1) as f64;
            for hi in 0..n_h {
                let h = 10.0 * hi as f64 / (n_h - 1) as f64;
                let row = ti * n_h + hi;
                z[[row, 0]] = t * t.cos();
                z[[row, 1]] = h;
                z[[row, 2]] = t * t.sin();
            }
        }

        // PCA-2 seed (the global-linear incumbent) via the same SVD path.
        let mut centered = z.clone();
        let mean = z.mean_axis(ndarray::Axis(0)).unwrap();
        for row in 0..n {
            for c in 0..3 {
                centered[[row, c]] -= mean[c];
            }
        }
        let (_, _, vt) = centered.svd(false, true).unwrap();
        let vt = vt.unwrap();
        let mut pca = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            for k in 0..2 {
                pca[[row, k]] = (0..3).map(|c| centered[[row, c]] * vt[[k, c]]).sum();
            }
        }

        // Intrinsic-metric seed.
        let intrinsic = intrinsic_metric_seed_coords(z.view(), 2, 8).unwrap();

        // Held-out thin-plate reconstruction R^2 from 2-D coordinates.
        let heldout_r2 = |coords: &Array2<f64>| -> f64 {
            // standardize coords
            let cmean = coords.mean_axis(ndarray::Axis(0)).unwrap();
            let mut cstd = [0.0_f64; 2];
            for k in 0..2 {
                cstd[k] = (coords.column(k).iter().map(|&v| (v - cmean[k]).powi(2)).sum::<f64>()
                    / n as f64)
                    .sqrt()
                    .max(1e-12);
            }
            let n_centers = 70usize;
            let centers: Vec<usize> =
                (0..n_centers).map(|i| i * (n - 1) / (n_centers - 1)).collect();
            let width = 3 + n_centers;
            let mut phi = Array2::<f64>::zeros((n, width));
            let cs = |row: usize, k: usize| (coords[[row, k]] - cmean[k]) / cstd[k];
            for row in 0..n {
                phi[[row, 0]] = 1.0;
                phi[[row, 1]] = cs(row, 0);
                phi[[row, 2]] = cs(row, 1);
                for (ci, &cr) in centers.iter().enumerate() {
                    let r2 = (0..2)
                        .map(|k| {
                            let d = cs(row, k) - cs(cr, k);
                            d * d
                        })
                        .sum::<f64>()
                        .max(1e-12);
                    phi[[row, 3 + ci]] = 0.5 * r2 * r2.ln();
                }
            }
            let train: Vec<usize> = (0..n).filter(|r| r % 4 != 0).collect();
            let test: Vec<usize> = (0..n).filter(|r| r % 4 == 0).collect();
            let phi_tr = phi.select(ndarray::Axis(0), &train);
            let z_tr = z.select(ndarray::Axis(0), &train);
            let mut gram = fast_ata(&phi_tr);
            let scale = gram.diag().iter().copied().fold(0.0_f64, f64::max);
            for dgn in gram.diag_mut().iter_mut() {
                *dgn += scale * 1e-8;
            }
            let rhs = fast_atb(&phi_tr, &z_tr);
            let decoder = gram.cholesky(Side::Lower).unwrap().solve_mat(&rhs);
            let mut mean_t = [0.0_f64; 3];
            for &row in &test {
                for c in 0..3 {
                    mean_t[c] += z[[row, c]];
                }
            }
            for c in 0..3 {
                mean_t[c] /= test.len() as f64;
            }
            let (mut resid, mut total) = (0.0_f64, 0.0_f64);
            for &row in &test {
                for c in 0..3 {
                    let mut fit = 0.0_f64;
                    for a in 0..width {
                        fit += phi[[row, a]] * decoder[[a, c]];
                    }
                    resid += (z[[row, c]] - fit).powi(2);
                    total += (z[[row, c]] - mean_t[c]).powi(2);
                }
            }
            1.0 - resid / total
        };

        let r2_intrinsic = heldout_r2(&intrinsic);
        let r2_pca = heldout_r2(&pca);
        assert!(
            r2_intrinsic > 0.99,
            "intrinsic geodesic seed must unfold the rolled sheet; intrinsic R²={r2_intrinsic}, pca R²={r2_pca}"
        );
        assert!(
            r2_pca < 0.9,
            "the global-linear PCA seed must collapse on a genuine fold (non-injective projection); \
             intrinsic R²={r2_intrinsic}, pca R²={r2_pca}"
        );
        assert!(
            r2_intrinsic > r2_pca + 0.05,
            "intrinsic seed must beat PCA on the fold; intrinsic R²={r2_intrinsic}, pca R²={r2_pca}"
        );
    }

    /// Determinism (fleet law): the intrinsic seed is a pure function of the data
    /// — bit-identical across repeated calls, no RNG, no thread/order dependence.
    #[test]
    fn intrinsic_seed_is_deterministic() {
        let n = 120usize;
        let mut z = Array2::<f64>::zeros((n, 4));
        for row in 0..n {
            let a = row as f64 * 0.11;
            z[[row, 0]] = a.sin();
            z[[row, 1]] = (2.0 * a).cos();
            z[[row, 2]] = (0.5 * a).sin() * 0.7;
            z[[row, 3]] = ((row % 7) as f64) * 0.05;
        }
        let first = intrinsic_metric_seed_coords(z.view(), 2, 6).unwrap();
        let second = intrinsic_metric_seed_coords(z.view(), 2, 6).unwrap();
        assert_eq!(first, second, "intrinsic seed must be bit-reproducible");
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
            let key: Vec<i64> = (0..n)
                .map(|i| (s[[0, i, 0]] * 1e6).round() as i64)
                .collect();
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

    /// Deterministic structured Z with `p` features, moderate rank.
    fn structured_z(n: usize, p: usize) -> Array2<f64> {
        let mut zvals = Vec::with_capacity(n * p);
        for r in 0..n {
            for c in 0..p {
                zvals.push(
                    ((r as f64) * 0.37 + (c as f64) * 1.1).sin() * ((c + 1) as f64)
                        + 0.3 * (((r * 3 + c * 5) as f64) * 0.21).cos()
                        + 0.05 * (r as f64 - c as f64),
                );
            }
        }
        Array2::from_shape_vec((n, p), zvals).unwrap()
    }

    /// Count atoms whose coordinate fibers are pairwise DISTINCT designs (rounded
    /// to `1e-6`). Two atoms sharing a fiber are a DUPLICATE design — the exact
    /// Hessian null that drives K≫p co-collapse (#1893).
    fn distinct_fibers(seed: &Array3<f64>) -> usize {
        let (k, n, dm) = seed.dim();
        let mut set = std::collections::HashSet::new();
        for atom in 0..k {
            let mut key: Vec<i64> = Vec::with_capacity(n * dm);
            for row in 0..n {
                for ax in 0..dm {
                    key.push((seed[[atom, row, ax]] * 1.0e6).round() as i64);
                }
            }
            set.insert(key);
        }
        set.len()
    }

    /// #1893 — the CURVED topology-seed path (the main path for circle/torus/sphere
    /// at n ≥ 4) must give K ≫ p atoms pairwise-DISTINCT charts. Before the fix the
    /// sphere/flat topology arms ignored `atom_idx` entirely (every atom identical)
    /// and the circle/torus arms wrapped once atoms outran the harmonic pairs.
    #[test]
    fn overcomplete_topology_seeds_pairwise_distinct_all_curved() {
        let n = 64usize;
        let p = 6usize;
        let z = structured_z(n, p);
        for (kind, d) in [
            (SaeAtomBasisKind::Periodic, 1usize),
            (SaeAtomBasisKind::Torus, 2usize),
            (SaeAtomBasisKind::Sphere, 2usize),
        ] {
            for mult in [4usize, 40usize] {
                let k = mult * p;
                let kinds = vec![kind.clone(); k];
                let dims = vec![d; k];
                let seed = sae_pca_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
                for v in seed.iter() {
                    assert!(v.is_finite(), "{kind:?} K={k}: non-finite seed coord {v}");
                }
                let distinct = distinct_fibers(&seed);
                assert_eq!(
                    distinct, k,
                    "{kind:?} K={k} (={mult}·p): every atom's chart must be a distinct \
                     design — got {distinct}/{k} distinct (duplicate designs ⇒ exact \
                     Hessian null ⇒ co-collapse)"
                );
            }
        }
    }

    /// #1893 — the FLAT (Euclidean) linear path: surplus atoms (K > available
    /// principal scores) previously WRAPPED `% avail`, re-reading the same score
    /// column and seeding exact-duplicate designs. K = 40·p ≫ avail must now be
    /// pairwise distinct via the generic-frame fallback.
    #[test]
    fn overcomplete_flat_linear_seeds_pairwise_distinct() {
        let n = 64usize;
        let p = 6usize; // avail ≈ 6 principal scores; 40·p = 240 ≫ 6.
        let z = structured_z(n, p);
        for mult in [4usize, 40usize] {
            let k = mult * p;
            let kinds = vec![SaeAtomBasisKind::Linear; k];
            let dims = vec![1usize; k];
            let seed = sae_pca_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
            let distinct = distinct_fibers(&seed);
            assert_eq!(
                distinct, k,
                "flat K={k} (={mult}·p): surplus atoms must not wrap onto duplicate \
                 principal-score designs — got {distinct}/{k} distinct"
            );
        }
    }

    /// #1893 — the linear CURVED fallback (n < 4 forces the topology path to return
    /// None): surplus torus/sphere atoms must route to distinct generic frames
    /// rather than reuse the same PC pairs. Exercises the linear Torus/Sphere
    /// surplus arms directly.
    #[test]
    fn overcomplete_linear_curved_fallback_distinct() {
        let n = 3usize; // < 4 ⇒ topology_curved_seed returns None ⇒ linear path.
        let p = 6usize;
        let z = structured_z(n, p);
        for (kind, d) in [
            (SaeAtomBasisKind::Periodic, 1usize),
            (SaeAtomBasisKind::Torus, 2usize),
            (SaeAtomBasisKind::Sphere, 2usize),
        ] {
            let k = 4 * p;
            let kinds = vec![kind.clone(); k];
            let dims = vec![d; k];
            let seed = sae_pca_seed_initial_coords(z.view(), &kinds, &dims).unwrap();
            for v in seed.iter() {
                assert!(v.is_finite(), "{kind:?} linear K={k}: non-finite {v}");
            }
            let distinct = distinct_fibers(&seed);
            assert_eq!(
                distinct, k,
                "{kind:?} linear-fallback K={k}: surplus atoms must be pairwise \
                 distinct — got {distinct}/{k}"
            );
        }
    }
}
