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
    n_atoms: usize,
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
            n_atoms,
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

    /// Accumulate the current matrix's pairwise joint counts into the flat upper-
    /// triangle buffer `joint[u*g + v]` (`u < v`), and the marginal counts into
    /// `marg`.
    fn accumulate(&self, joint: &mut [f64], marg: &mut [f64]) {
        let g = self.n_atoms;
        for row in &self.rows {
            for (idx, &u) in row.iter().enumerate() {
                marg[u] += 1.0;
                for &v in &row[idx + 1..] {
                    let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                    joint[lo * g + hi] += 1.0;
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
    g: usize,
    n_obs: usize,
    obs: Vec<f64>,      // upper-tri joint counts
    null_mean: Vec<f64>,
    z: Vec<f64>,
}

impl CoactivationExceedance {
    fn idx(&self, a: usize, b: usize) -> usize {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        lo * self.g + hi
    }

    /// Standardized excess `z = (observed − null_mean) / null_sd` of the pair's
    /// joint activation over the fixed-margin null. `0` on the diagonal.
    pub fn excess_z(&self, a: usize, b: usize) -> f64 {
        if a == b || a >= self.g || b >= self.g {
            return 0.0;
        }
        self.z[self.idx(a, b)]
    }

    /// Observed joint activation count of the pair.
    pub fn observed_joint(&self, a: usize, b: usize) -> f64 {
        if a == b || a >= self.g || b >= self.g {
            return 0.0;
        }
        self.obs[self.idx(a, b)]
    }

    /// Null-mean joint activation count of the pair under the fixed margins.
    pub fn null_mean_joint(&self, a: usize, b: usize) -> f64 {
        if a == b || a >= self.g || b >= self.g {
            return 0.0;
        }
        self.null_mean[self.idx(a, b)]
    }

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

/// Estimate the per-pair co-activation exceedance of `codes` over the fixed-margin
/// null, using `replicates` curveball replicates. The curveball mixing length is
/// derived from the matrix (one sweep ≈ its number of ones) so there are no tuned
/// constants; the sampler seed is a content hash of `codes` (deterministic).
///
/// Between replicates the chain is advanced by one mixing sweep and thinned, and a
/// burn-in sweep is run before the first sample, so the replicates are near-
/// independent draws from the fixed-margin class.
pub fn coactivation_exceedance(codes: &SparseAtomCodes, replicates: usize) -> CoactivationExceedance {
    let g = codes.k_atoms();
    let n_obs = codes.n_obs();
    let size = g * g;

    // Observed joint + marginal counts.
    let mut obs = vec![0.0_f64; size];
    let mut obs_marg = vec![0.0_f64; g];
    {
        let sampler = CurveballSampler::from_codes(codes);
        sampler.accumulate(&mut obs, &mut obs_marg);
    }

    let mut null_mean = vec![0.0_f64; size];
    let mut z = vec![0.0_f64; size];
    if g < 2 || n_obs < 2 || replicates == 0 {
        return CoactivationExceedance { g, n_obs, obs, null_mean, z };
    }

    let mut sampler = CurveballSampler::from_codes(codes);
    // Mixing sweep length: one chance per active entry to move (the canonical
    // curveball budget), at least the row count so even a very sparse matrix mixes.
    let sweep = sampler.n_ones().max(sampler.n_rows());
    sampler.mix(sweep); // burn-in

    // Welford accumulation of each pair's null joint count across replicates.
    let mut mean = vec![0.0_f64; size];
    let mut m2 = vec![0.0_f64; size];
    let mut scratch = vec![0.0_f64; size];
    let mut scratch_marg = vec![0.0_f64; g];
    for r in 0..replicates {
        sampler.mix(sweep); // thin between draws
        for v in scratch.iter_mut() {
            *v = 0.0;
        }
        for v in scratch_marg.iter_mut() {
            *v = 0.0;
        }
        sampler.accumulate(&mut scratch, &mut scratch_marg);
        let count = (r + 1) as f64;
        for u in 0..g {
            for w in (u + 1)..g {
                let idx = u * g + w;
                let x = scratch[idx];
                let delta = x - mean[idx];
                mean[idx] += delta / count;
                m2[idx] += delta * (x - mean[idx]);
            }
        }
    }

    let denom = (replicates.saturating_sub(1)).max(1) as f64;
    for u in 0..g {
        for w in (u + 1)..g {
            let idx = u * g + w;
            null_mean[idx] = mean[idx];
            let var = m2[idx] / denom;
            let sd = var.max(0.0).sqrt();
            z[idx] = if sd > NULL_SD_FLOOR {
                (obs[idx] - mean[idx]) / sd
            } else {
                // Deterministic (pinned) null: no resolvable spread ⇒ no exceedance.
                0.0
            };
        }
    }

    CoactivationExceedance { g, n_obs, obs, null_mean, z }
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
        // mechanical rate. On the even rows the block is forced on together and the
        // rest of k is filled from the NON-block atoms; the odd rows draw all k
        // from the non-block atoms only. So the block atoms fire exactly on the even
        // rows and always together — a genuine above-margin coupling the
        // fixed-margin null cannot reproduce (it can, and on average does, separate
        // two columns of equal margin), while the non-block atoms are top-k noise.
        let mut planted = SparseAtomCodes::empty(n, g);
        let block = [0usize, 1, 2, 3];
        let mut non_block: Vec<usize> = (block.len()..g).collect();
        for row in 0..n {
            if row % 2 == 0 {
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
