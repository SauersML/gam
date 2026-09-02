//! Fixed-margin (curveball) null for the binary support matrix that the
//! structure-move triggers read.
//!
//! # Why a fixed-margin null
//!
//! The fusion / fission proposal triggers (`crate::structure_harvest`) read
//! co-activation statistics off the per-token active-support masks
//! ([`crate::atom_codes::SparseAtomCodes`]). Those masks are produced by a
//! (hard or soft) top-`k` selection, and top-`k` selection puts MECHANICAL
//! structure into the co-activation that has nothing to do with real coupling:
//!
//!  * fixing the number active per token to (about) `k` induces a negative
//!    indicator covariance `≈ −k(G−k) / (G²(G−1))` between EVERY pair of atoms,
//!    purely because one atom firing leaves less room for the others; and
//!  * a hard top-`k` puts ZERO mass off the `k`-shell, so any coupling reading
//!    that assumes a positive base measure (an Ising / log-linear model) is
//!    reading an artifact of the constraint surface, not an interaction.
//!
//! A raw coupling trigger therefore fires on this mechanical structure. The fix
//! is to score each pair's observed co-activation against a null that PRESERVES
//! exactly the structure the mechanism forces — every row sum (each token's
//! support size, i.e. the top-`k` constraint) AND every column sum (each atom's
//! total activation) — and keep only the EXCEEDANCE over that null. Under pure
//! top-`k` noise the observed co-activation equals the null (zero exceedance);
//! only genuine, above-margin co-firing survives.
//!
//! # Curveball
//!
//! The null is sampled by the *curveball* algorithm (Strona et al., 2014): a
//! Markov chain over binary matrices with fixed row and column margins whose
//! elementary move ("trade") is O(row weight) and needs no rejection step. Pick
//! two rows; the atoms shared by both stay put; the atoms exclusive to one of
//! the two are pooled and re-dealt uniformly at random back to the two rows in
//! their original per-row counts. Each dealt atom still appears exactly once
//! across the pair, so both row sums and both column sums are preserved, and the
//! chain mixes toward the uniform distribution over the fixed-margin class.
//!
//! # Determinism
//!
//! The sampler is seeded from a content hash of the support matrix, so a harvest
//! that consumes it stays a deterministic function of the fitted state (the
//! `structure_harvest` purity contract): same codes → same seed → same stream →
//! same exceedances → same proposals.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::BTreeMap;

use crate::atom_codes::SparseAtomCodes;

/// Number of independent fixed-margin replicates whose joint-count spread defines
/// the null. A computational budget (magic-by-default, like the harvest's
/// per-round move caps), large enough that the per-pair null mean/standard
/// deviation are stable and small enough that the whole null is cheap next to a
/// fit. Not a statistical knob — the exceedance threshold is derived separately.
pub const NULL_REPLICATES: usize = 200;

/// Curveball state: each row's active-atom index list (kept sorted so a trade's
/// shared/exclusive split is a linear merge) plus the driving RNG. Preserves
/// every row sum and every column sum under [`Self::trade`].
pub struct CurveballSampler {
    rows: Vec<Vec<usize>>,
    rng: StdRng,
}

impl CurveballSampler {
    /// Seed a sampler from the discrete active supports of `codes`. The RNG seed
    /// is a content hash of the support matrix (dimensions + per-row supports) so
    /// the chain is reproducible for identical inputs.
    pub fn from_codes(codes: &SparseAtomCodes) -> Self {
        let n_atoms = codes.k_atoms();
        let mut rows: Vec<Vec<usize>> = Vec::with_capacity(codes.n_obs());
        // Content hash (SplitMix64 fold) of the support matrix → deterministic seed.
        let mut seed = gam_linalg::utils::splitmix64_hash(
            (codes.n_obs() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (n_atoms as u64),
        );
        for code in codes.iter() {
            let active: Vec<usize> = code.active_mask.iter_ones().collect();
            for &a in &active {
                seed = gam_linalg::utils::splitmix64_hash(seed ^ (a as u64).wrapping_add(1));
            }
            seed = gam_linalg::utils::splitmix64_hash(seed ^ 0xD1B5_4A32_D192_ED03);
            rows.push(active);
        }
        Self {
            rows,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Total number of active entries (ones) in the matrix — the natural scale for
    /// the mixing budget (each 1 gets ~one chance to move per sweep).
    pub fn n_ones(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// One curveball trade between two distinct random rows. Shared atoms stay;
    /// atoms exclusive to exactly one of the two rows are pooled and re-dealt
    /// uniformly at random, preserving both rows' sizes (row sums) and each atom's
    /// total count (column sums).
    pub fn trade(&mut self) {
        let n = self.rows.len();
        if n < 2 {
            return;
        }
        let i = self.rng.random_range(0..n);
        let mut j = self.rng.random_range(0..n - 1);
        if j >= i {
            j += 1;
        }
        // Split the two sorted rows into shared (kept) and exclusive (pooled).
        let (a, b) = (&self.rows[i], &self.rows[j]);
        let mut shared_i: Vec<usize> = Vec::new();
        let mut pool: Vec<usize> = Vec::new();
        let (mut p, mut q) = (0usize, 0usize);
        let mut n_from_i = 0usize;
        while p < a.len() && q < b.len() {
            match a[p].cmp(&b[q]) {
                std::cmp::Ordering::Equal => {
                    shared_i.push(a[p]);
                    p += 1;
                    q += 1;
                }
                std::cmp::Ordering::Less => {
                    pool.push(a[p]);
                    n_from_i += 1;
                    p += 1;
                }
                std::cmp::Ordering::Greater => {
                    pool.push(b[q]);
                    q += 1;
                }
            }
        }
        while p < a.len() {
            pool.push(a[p]);
            n_from_i += 1;
            p += 1;
        }
        while q < b.len() {
            pool.push(b[q]);
            q += 1;
        }
        // Nothing exclusive ⇒ the trade is a no-op (identical rows, or one nests
        // in the other with no swappable element). Leave the rows untouched.
        if pool.is_empty() || n_from_i == 0 || n_from_i == pool.len() {
            return;
        }
        // Deal `n_from_i` of the pool to row i (rest to row j), uniformly at random
        // via a partial Fisher–Yates over the pool.
        let m = pool.len();
        for t in 0..n_from_i {
            let swap = t + self.rng.random_range(0..(m - t));
            pool.swap(t, swap);
        }
        // Rebuild the two rows: shared + their dealt exclusive atoms, kept sorted.
        let build = |shared: &[usize], extra: &[usize]| -> Vec<usize> {
            let mut v: Vec<usize> = Vec::with_capacity(shared.len() + extra.len());
            v.extend_from_slice(shared);
            v.extend_from_slice(extra);
            v.sort_unstable();
            v
        };
        let new_i = build(&shared_i, &pool[..n_from_i]);
        let new_j = build(&shared_i, &pool[n_from_i..]);
        self.rows[i] = new_i;
        self.rows[j] = new_j;
    }

    /// Run `trades` elementary trades to advance the chain.
    pub fn mix(&mut self, trades: usize) {
        for _ in 0..trades {
            self.trade();
        }
    }

    /// Accumulate joint counts only for the requested unordered pairs. This is
    /// the sparse structure-search path: once proposal candidates are restricted
    /// to observed co-firing pairs, the fixed-margin null must not reintroduce a
    /// dense `K²` buffer just to score them.
    fn accumulate_selected(
        &self,
        pair_to_pos: &BTreeMap<(usize, usize), usize>,
        joint: &mut [f64],
    ) {
        for row in &self.rows {
            for (idx, &u) in row.iter().enumerate() {
                for &v in &row[idx + 1..] {
                    let key = if u < v { (u, v) } else { (v, u) };
                    if let Some(&pos) = pair_to_pos.get(&key) {
                        joint[pos] += 1.0;
                    }
                }
            }
        }
    }
}

/// Per-pair co-activation exceedance over the fixed-margin (curveball) null.
///
/// For each unordered atom pair it holds the observed joint activation count, the
/// null mean, and the standardized excess `z = (obs − mean) / sd`. A pair with
/// `z` near zero co-activates no more than the top-`k` margins mechanically force;
/// a large positive `z` is genuine above-margin co-firing.
#[derive(Clone, Debug)]
pub struct CoactivationExceedance {
    n_obs: usize,
}

impl CoactivationExceedance {

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }
}

/// Numerical floor guarding the exceedance division when a pair's null joint
/// count has (essentially) zero spread. A pair the fixed margins PIN (no
/// swappable configuration ever moves it — e.g. two columns locked together with
/// no mixing room) has every replicate equal to the observed value, so its null
/// mean equals the observation and the numerator is ~0: the ratio is a
/// well-defined ~0 exceedance, not a spurious spike. The floor only prevents a
/// literal divide-by-zero; it is deliberately tiny so a genuinely extreme pair
/// (observed at the boundary of the fixed-margin polytope, tiny but non-zero
/// spread) still reports a large exceedance rather than being clamped to noise.
const NULL_SD_FLOOR: f64 = 1e-9;

/// Sparse fixed-margin exceedance for a pre-indexed candidate pair set.
///
/// Returns standardized excess values aligned with `pairs`. Unlike
/// [`coactivation_exceedance`], this never allocates or scans a dense `K²` pair
/// table; every replicate accumulates only co-firing pairs that are present in
/// `pairs`.
pub fn coactivation_exceedance_for_pairs(
    codes: &SparseAtomCodes,
    pairs: &[(usize, usize)],
    replicates: usize,
) -> Vec<f64> {
    let g = codes.k_atoms();
    let n_obs = codes.n_obs();
    let mut pair_to_pos = BTreeMap::new();
    let mut canonical = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        if a == b || a >= g || b >= g {
            canonical.push(None);
            continue;
        }
        let key = if a < b { (a, b) } else { (b, a) };
        let next = pair_to_pos.len();
        let pos = *pair_to_pos.entry(key).or_insert(next);
        canonical.push(Some(pos));
    }
    let m = pair_to_pos.len();
    if m == 0 {
        return vec![0.0; pairs.len()];
    }

    let mut obs = vec![0.0_f64; m];
    let sampler = CurveballSampler::from_codes(codes);
    sampler.accumulate_selected(&pair_to_pos, &mut obs);
    if g < 2 || n_obs < 2 || replicates == 0 {
        return vec![0.0; pairs.len()];
    }

    let mut sampler = CurveballSampler::from_codes(codes);
    let sweep = sampler.n_ones().max(sampler.n_rows());
    sampler.mix(sweep);
    let mut mean = vec![0.0_f64; m];
    let mut m2 = vec![0.0_f64; m];
    let mut scratch = vec![0.0_f64; m];
    for r in 0..replicates {
        sampler.mix(sweep);
        for value in scratch.iter_mut() {
            *value = 0.0;
        }
        sampler.accumulate_selected(&pair_to_pos, &mut scratch);
        let count = (r + 1) as f64;
        for pos in 0..m {
            let x = scratch[pos];
            let delta = x - mean[pos];
            mean[pos] += delta / count;
            m2[pos] += delta * (x - mean[pos]);
        }
    }

    let denom = (replicates.saturating_sub(1)).max(1) as f64;
    let mut sparse_z = vec![0.0_f64; m];
    for pos in 0..m {
        let var = m2[pos] / denom;
        let sd = var.max(0.0).sqrt();
        sparse_z[pos] = if sd > NULL_SD_FLOOR {
            (obs[pos] - mean[pos]) / sd
        } else {
            0.0
        };
    }
    canonical
        .into_iter()
        .map(|pos| pos.map_or(0.0, |idx| sparse_z[idx]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A curveball trade must preserve every row sum and every column sum.
    #[test]
    fn curveball_preserves_both_margins() {
        let n = 60usize;
        let g = 12usize;
        let mut codes = SparseAtomCodes::empty(n, g);
        // A structured matrix: each row a contiguous run of 3, sliding.
        for row in 0..n {
            let start = (row * 5) % g;
            for off in 0..3 {
                codes.row_mut(row).assign((start + off) % g, 1.0);
            }
        }
        let row_sums: Vec<usize> = (0..n).map(|r| codes.row(r).n_active()).collect();
        let mut col_sums = vec![0usize; g];
        for r in 0..n {
            for c in codes.row(r).active_mask.iter_ones() {
                col_sums[c] += 1;
            }
        }

        let mut s = CurveballSampler::from_codes(&codes);
        s.mix(2000);

        // Row sums preserved exactly.
        for (r, &want) in row_sums.iter().enumerate() {
            assert_eq!(s.rows[r].len(), want, "row {r} sum changed");
            // Sorted + unique (a valid support).
            for w in s.rows[r].windows(2) {
                assert!(w[0] < w[1], "row {r} not sorted/unique");
            }
        }
        // Column sums preserved exactly.
        let mut got_col = vec![0usize; g];
        for row in &s.rows {
            for &c in row {
                got_col[c] += 1;
            }
        }
        assert_eq!(got_col, col_sums, "column sums changed");
    }

    /// The reviewer's core requirement (2): under PURE top-`k` noise (uniform
    /// random which `k` fire, no real coupling) the null-corrected exceedance fires
    /// on ~zero pairs even though the RAW dependence trigger fires on many, because
    /// the mechanical top-`k` structure is inside the fixed-margin null; a PLANTED
    /// block of genuinely co-firing atoms still exceeds the null.
    #[test]
    fn top_k_noise_gives_no_exceedance_but_planted_block_does() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;

        let n = 500usize;
        let g = 20usize;
        let k = 14usize; // DENSE top-k: mechanical co-firing is near-certain.

        // (a) Pure top-k noise: each row a uniform random k-subset (fixed row sums).
        let mut rng = StdRng::seed_from_u64(0xA11CE);
        let mut noise = SparseAtomCodes::empty(n, g);
        let mut atoms: Vec<usize> = (0..g).collect();
        for row in 0..n {
            atoms.shuffle(&mut rng);
            for &a in &atoms[..k] {
                noise.row_mut(row).assign(a, 1.0);
            }
        }
        // Raw dependence trigger fires on MANY pairs (dense top-k ⇒ P(a|b) ≈ 0.7).
        let raw_fires = {
            let mut c = 0usize;
            for a in 0..g {
                for b in (a + 1)..g {
                    if noise.coactivation(a, b).dependence() >= 0.6 {
                        c += 1;
                    }
                }
            }
            c
        };
        assert!(
            raw_fires > 50,
            "raw dependence must fire on many top-k-noise pairs; got {raw_fires}"
        );
        // Null-corrected exceedance fires on ~none (allow a small false-positive tail).
        let ex = coactivation_exceedance(&noise, NULL_REPLICATES);
        let null_fires = {
            let mut c = 0usize;
            for a in 0..g {
                for b in (a + 1)..g {
                    if ex.excess_z(a, b) >= 3.0 {
                        c += 1;
                    }
                }
            }
            c
        };
        let total_pairs = g * (g - 1) / 2;
        assert!(
            null_fires <= total_pairs / 20,
            "null-corrected exceedance must fire on ~zero top-k-noise pairs; got \
             {null_fires} of {total_pairs}"
        );

        // (b) Planted block: atoms {0,1,2,3} co-fire together far above the
        // mechanical rate. The block fires — always together — on a SPARSE subset
        // of rows (every fifth), and the block atoms fire NOWHERE else, so each has
        // a low margin (≈ n/5). Two low-margin columns are easily separated by the
        // fixed-margin null (expected joint ≈ m²/n ≪ m), so the observed all-together
        // joint sits far in the null's upper tail. The remaining k on block rows and
        // all k on the other rows are drawn from the NON-block atoms (top-k noise).
        let mut planted = SparseAtomCodes::empty(n, g);
        let block = [0usize, 1, 2, 3];
        let mut non_block: Vec<usize> = (block.len()..g).collect();
        for row in 0..n {
            if row % 5 == 0 {
                for &a in &block {
                    planted.row_mut(row).assign(a, 1.0);
                }
                non_block.shuffle(&mut rng);
                for &a in &non_block[..(k - block.len())] {
                    planted.row_mut(row).assign(a, 1.0);
                }
            } else {
                non_block.shuffle(&mut rng);
                for &a in &non_block[..k] {
                    planted.row_mut(row).assign(a, 1.0);
                }
            }
        }
        let ex_p = coactivation_exceedance(&planted, NULL_REPLICATES);
        // Every within-block pair must exceed the null strongly.
        for a in 0..block.len() {
            for b in (a + 1)..block.len() {
                let z = ex_p.excess_z(block[a], block[b]);
                assert!(
                    z >= 3.0,
                    "planted block pair ({},{}) must exceed the fixed-margin null; z={z}",
                    block[a],
                    block[b]
                );
            }
        }
    }
}

// --------------------------------------------------------------------------
// Sparse-route surrogate nulls (#2470 finding 7).
//
// Lifted verbatim out of `crates/gam-pyffi/src/latent/sae_spectral_ffi.rs`,
// where it was pure domain logic behind an FFI boundary: measured at zero
// PyO3 tokens and referencing nothing outside gam-sae's own modules. The
// pyfunctions there now import these instead of defining them, so the Rust
// API and the Python API run the same sampler rather than parallel-but-equal
// copies.
// --------------------------------------------------------------------------

#[derive(Clone)]
pub struct AuditSparseRoute {
    pub indices: ndarray::Array2<u32>,
    pub values: ndarray::Array3<f32>,
    pub n_units: usize,
    pub block_size: usize,
}

impl AuditSparseRoute {
    pub fn new(
        indices: ndarray::Array2<u32>,
        values: ndarray::Array3<f32>,
        n_units: usize,
        block_size: usize,
        label: &str,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err("audit_sae block_size must be >= 1".to_string());
        }
        if n_units == 0 {
            return Err(format!(
                "audit_sae {label} requires at least one routing unit"
            ));
        }
        let (n_rows, width) = indices.dim();
        if n_rows == 0 || width == 0 {
            return Err(format!(
                "audit_sae {label} must be a non-empty N×s route; got {:?}",
                indices.dim()
            ));
        }
        if values.shape() != [n_rows, width, block_size] {
            return Err(format!(
                "audit_sae {label} values shape {:?} does not match indices {:?} and block_size {block_size}",
                values.shape(),
                indices.dim()
            ));
        }
        for row in 0..n_rows {
            let mut live = std::collections::HashSet::with_capacity(width);
            for slot in 0..width {
                let unit = indices[[row, slot]] as usize;
                if unit >= n_units {
                    return Err(format!(
                        "audit_sae {label} index {unit} at row {row}, slot {slot} is outside 0..{n_units}"
                    ));
                }
                let mut norm2 = 0.0_f64;
                for offset in 0..block_size {
                    let value = values[[row, slot, offset]] as f64;
                    if !value.is_finite() {
                        return Err(format!(
                            "audit_sae {label} value at row {row}, slot {slot}, offset {offset} is not finite"
                        ));
                    }
                    norm2 += value * value;
                }
                if norm2 > 0.0 && !live.insert(unit) {
                    return Err(format!(
                        "audit_sae {label} repeats live unit {unit} in row {row}"
                    ));
                }
            }
        }
        Ok(Self {
            indices,
            values,
            n_units,
            block_size,
        })
    }

    pub fn nrows(&self) -> usize {
        self.indices.nrows()
    }

    pub fn width(&self) -> usize {
        self.indices.ncols()
    }

    pub fn gate(&self, row: usize, slot: usize) -> f64 {
        let mut norm2 = 0.0_f64;
        for offset in 0..self.block_size {
            let value = self.values[[row, slot, offset]] as f64;
            norm2 += value * value;
        }
        norm2.sqrt()
    }

    pub fn reconstruct(
        &self,
        decoder: ndarray::ArrayView2<'_, f32>,
    ) -> Result<ndarray::Array2<f32>, String> {
        if self.block_size == 1 {
            crate::sparse_dict::reconstruct_sparse_rows(
                decoder,
                self.indices.view(),
                self.values.index_axis(ndarray::Axis(2), 0),
            )
        } else {
            crate::sparse_dict::reconstruct_block_sparse_rows(
                decoder,
                self.indices.view(),
                self.values.view(),
                self.block_size,
            )
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct LiveAmplitudeMoments {
    pub count: usize,
    pub sum: f64,
    pub sum2: f64,
}

impl LiveAmplitudeMoments {
    pub fn mean(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn sd(self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let n = self.count as f64;
            ((self.sum2 - self.sum * self.sum / n) / (n - 1.0))
                .max(0.0)
                .sqrt()
        }
    }
}

pub fn live_amplitude_moments(route: &AuditSparseRoute) -> Vec<LiveAmplitudeMoments> {
    let mut moments = vec![LiveAmplitudeMoments::default(); route.n_units];
    for row in 0..route.nrows() {
        for slot in 0..route.width() {
            let gate = route.gate(row, slot);
            if gate > 0.0 {
                let unit = route.indices[[row, slot]] as usize;
                moments[unit].count += 1;
                moments[unit].sum += gate;
                moments[unit].sum2 += gate * gate;
            }
        }
    }
    moments
}

pub fn resample_sparse_architecture_null<R: rand::Rng + ?Sized>(
    observed: &AuditSparseRoute,
    donor: &AuditSparseRoute,
    rng: &mut R,
) -> Result<AuditSparseRoute, String> {
    use rand::RngExt;
    let observed_moments = live_amplitude_moments(observed);
    let donor_moments = live_amplitude_moments(donor);
    let mut indices = ndarray::Array2::<u32>::zeros((observed.nrows(), donor.width()));
    let mut values =
        ndarray::Array3::<f32>::zeros((observed.nrows(), donor.width(), donor.block_size));
    for row in 0..observed.nrows() {
        let source = rng.random_range(0..donor.nrows());
        for slot in 0..donor.width() {
            let unit = donor.indices[[source, slot]] as usize;
            indices[[row, slot]] = unit as u32;
            let gate = donor.gate(source, slot);
            if gate == 0.0 {
                continue;
            }
            let observed_moment = observed_moments[unit];
            let donor_moment = donor_moments[unit];
            if observed_moment.count == 0 {
                continue;
            }
            let donor_sd = donor_moment.sd();
            let target_gate = if donor_sd > 0.0 {
                (observed_moment.mean()
                    + (gate - donor_moment.mean()) * observed_moment.sd() / donor_sd)
                    .max(0.0)
            } else {
                observed_moment.mean()
            };
            if target_gate == 0.0 {
                continue;
            }
            let scale = target_gate / gate;
            for offset in 0..donor.block_size {
                values[[row, slot, offset]] =
                    (donor.values[[source, slot, offset]] as f64 * scale) as f32;
            }
        }
    }
    AuditSparseRoute::new(
        indices,
        values,
        observed.n_units,
        observed.block_size,
        "architecture-matched null route",
    )
}
