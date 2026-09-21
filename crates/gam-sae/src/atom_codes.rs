//! Per-point sparse atom codes for multi-manifold reconstruction.
//!
//! This module owns the storage of per-observation soft assignments over a
//! library of `K` candidate manifold-atoms (see [`crate::assignment::SaeAssignment`]
//! for the surrounding selection/gate layer). The two key types are:
//!
//! * [`BitVec`] — a minimal dependency-free bitset used to record the *active
//!   support* `S_n ⊆ {0, …, K−1}` of each observation. We avoid pulling in
//!   the external `bitvec` crate to keep this module aligned with the rest of
//!   `gam`'s "no extra deps for new primitives" policy.
//! * [`SparseAtomCode`] — the per-point pair `(active_mask, weights)` whose
//!   semantics are documented on the type. Reconstruction at point `n` is
//!
//!   ```text
//!   Ẑ_n  =  Σ_{k ∈ S_n}  w_{n,k}  ·  decoder_k(t_{n,k})
//!   ```
//!
//!   so `weights[k]` is meaningful only when `active_mask.get(k) == true`.
//!   We store `weights` densely (`Vec<f64>` of length `K`) rather than
//!   sparsely, so the codes of `N` rows hold `O(N·K)` words: the same order as
//!   the dense `N × K` assignment matrix the production producers read them
//!   from, so the container adds no new order of memory. What must not grow
//!   faster is what is computed from it: the support code
//!   ([`SparseAtomCodes::support_entropy`]) is never sized `K²` (#2933 F43).
//!   The mask carries the discrete active-set information; the weights carry
//!   the soft amplitudes.
//!
//! ## Per-point block locality (arrow structure)
//!
//! Each [`SparseAtomCode`] is the per-row ext-coordinate block for observation `n`
//! restricted to the `K` atoms. Combined with the per-atom on-manifold
//! coordinate `t_{n,k} ∈ ℝ^{d_k}` (held in the per-atom latent-coordinate
//! blocks of [`crate::assignment::SaeAssignment`]), the row-local
//! ext-coordinate vector is
//!
//! ```text
//!   ext_n  =  ( a_{n,1..K}  ;  t_{n,1,·}  ;  …  ;  t_{n,K,·} )
//! ```
//!
//! whose interaction graph with the shared decoder coefficients `B_1..B_K`
//! is exactly the arrow / bordered-Hessian pattern from `latent_coord.md`
//! §2.2. The Schur complement that Piece 1 uses to eliminate β before the
//! per-row solve generalises here with one change: the row-`n` block now
//! couples to *only the active subset* `S_n` of decoder borders, not to all
//! K of them. That is the structural fact this module records.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::f64::consts::{LN_2, PI};

use statrs::function::gamma::ln_gamma;

/// Minimal bit-vector. Backing storage is `Vec<u64>` words.
///
/// We expose only the operations the atom-selection layer needs: construction,
/// `get`, `set`, `count_ones`, and iteration of set indices. This is
/// deliberately tiny — adding the external `bitvec` crate would be overkill
/// for a few hundred bits per observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitVec {
    words: Vec<u64>,
    len: usize,
}

impl BitVec {
    /// All-zero bitset of length `len`.
    pub fn zeros(len: usize) -> Self {
        let words = vec![0u64; len.div_ceil(64)];
        Self { words, len }
    }

    /// All-ones bitset of length `len`.
    pub fn ones(len: usize) -> Self {
        let mut bv = Self::zeros(len);
        for i in 0..len {
            bv.set(i, true);
        }
        bv
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        assert!(
            i < self.len,
            "BitVec::get index {i} out of bounds {}",
            self.len
        );
        let (w, b) = (i / 64, i % 64);
        (self.words[w] >> b) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        assert!(
            i < self.len,
            "BitVec::set index {i} out of bounds {}",
            self.len
        );
        let (w, b) = (i / 64, i % 64);
        if v {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterator over set indices in ascending order. Whole zero words are
    /// skipped, so a support of `s` atoms out of `K` costs `O(K/64 + s)`.
    pub(crate) fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let mut remaining = word;
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        None
                    } else {
                        let bit = remaining.trailing_zeros() as usize;
                        remaining &= remaining - 1;
                        Some(word_index * 64 + bit)
                    }
                })
            })
    }

    /// Zero all bits in place.
    pub fn clear(&mut self) {
        for w in self.words.iter_mut() {
            *w = 0;
        }
    }
}

/// Per-point sparse code over `K` candidate atoms.
///
/// Invariants (checked in debug builds):
///
/// * `active_mask.len() == weights.len() == K`.
/// * For any `k` with `active_mask.get(k) == false`, the value `weights[k]`
///   is a nuisance — it must not influence reconstruction. Selection
///   strategies that lower a weight to zero (e.g. an L¹-relaxed gate after
///   thresholding) are responsible for clearing the
///   corresponding mask bit *and* zeroing `weights[k]`.
///
/// We do not require `weights[k] >= 0`; some strategies (entropic softmax,
/// TopK projection) keep the simplex, while others (L¹-relaxed) only enforce
/// non-negativity at the active-set step. The owning
/// gate/selection layer ([`crate::assignment::SaeAssignment`]) documents which
/// invariant it maintains.
#[derive(Debug, Clone)]
pub struct SparseAtomCode {
    /// Length-`K` bitmask of active atoms for this point.
    pub active_mask: BitVec,
    /// Length-`K` dense weight vector. Only entries at active indices are
    /// semantically meaningful.
    pub weights: Vec<f64>,
}

impl SparseAtomCode {
    /// Cold-start: no atoms active, all weights zero.
    pub fn empty(k_atoms: usize) -> Self {
        Self {
            active_mask: BitVec::zeros(k_atoms),
            weights: vec![0.0; k_atoms],
        }
    }

    /// Total number of candidate atoms `K` this code is sized for.
    pub fn k_atoms(&self) -> usize {
        self.weights.len()
    }

    /// Cardinality of the active support `|S_n|`.
    pub fn n_active(&self) -> usize {
        self.active_mask.count_ones()
    }

    /// Set the weight for atom `k` and mark it active.
    pub fn assign(&mut self, k: usize, w: f64) {
        assert!(k < self.k_atoms());
        self.active_mask.set(k, true);
        self.weights[k] = w;
    }

}

/// Storage for the per-row codes of all `N` observations.
///
/// Held column-of-structs rather than struct-of-columns: each row's
/// `(active_mask, weights)` lives together because the atom-selection
/// strategies all touch a single row at a time. Cross-row vectorization
/// happens through ndarray views built on demand.
#[derive(Debug, Clone)]
pub struct SparseAtomCodes {
    codes: Vec<SparseAtomCode>,
    k_atoms: usize,
}

impl SparseAtomCodes {
    /// Allocate `n_obs` empty codes, each sized for `k_atoms`.
    pub fn empty(n_obs: usize, k_atoms: usize) -> Self {
        let codes = (0..n_obs).map(|_| SparseAtomCode::empty(k_atoms)).collect();
        Self { codes, k_atoms }
    }

    pub fn n_obs(&self) -> usize {
        self.codes.len()
    }

    pub fn k_atoms(&self) -> usize {
        self.k_atoms
    }

    pub fn row(&self, n: usize) -> &SparseAtomCode {
        &self.codes[n]
    }

    pub fn row_mut(&mut self, n: usize) -> &mut SparseAtomCode {
        &mut self.codes[n]
    }

    pub fn iter(&self) -> impl Iterator<Item = &SparseAtomCode> {
        self.codes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SparseAtomCode> {
        self.codes.iter_mut()
    }

    /// All atom pairs that co-fire at least once, with their support and
    /// amplitude-code statistics, computed in one sparse pass over row supports.
    ///
    /// This is the structure-search candidate index: rows contribute only their
    /// active-set pairs, so the producer cost is `Σ_row |S_row|²` and the output
    /// is bounded by observed co-firings, not by `K²`.
    pub(crate) fn coactive_pair_stats(&self) -> Vec<(usize, usize, CoactivationStats)> {
        #[derive(Clone, Copy, Debug, Default)]
        struct PairAccum {
            n_joint: usize,
            sum_a: f64,
            sum_b: f64,
            sum_a2: f64,
            sum_b2: f64,
            sum_ab: f64,
        }

        let n_obs = self.n_obs();
        let mut marg = vec![0usize; self.k_atoms];
        let mut pairs: BTreeMap<(usize, usize), PairAccum> = BTreeMap::new();
        for code in &self.codes {
            let active: Vec<usize> = code.active_mask.iter_ones().collect();
            for &atom in &active {
                marg[atom] += 1;
            }
            for (idx, &u) in active.iter().enumerate() {
                for &v in &active[idx + 1..] {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    let wa = code.weights[a];
                    let wb = code.weights[b];
                    let acc = pairs.entry((a, b)).or_default();
                    acc.n_joint += 1;
                    acc.sum_a += wa;
                    acc.sum_b += wb;
                    acc.sum_a2 += wa * wa;
                    acc.sum_b2 += wb * wb;
                    acc.sum_ab += wa * wb;
                }
            }
        }

        pairs
            .into_iter()
            .map(|((a, b), acc)| {
                let weight_correlation = if acc.n_joint < 2 {
                    0.0
                } else {
                    let n = acc.n_joint as f64;
                    let cov = acc.sum_ab - acc.sum_a * acc.sum_b / n;
                    let var_a = acc.sum_a2 - acc.sum_a * acc.sum_a / n;
                    let var_b = acc.sum_b2 - acc.sum_b * acc.sum_b / n;
                    if var_a > 0.0 && var_b > 0.0 {
                        (cov / (var_a.sqrt() * var_b.sqrt())).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    }
                };
                let stats = CoactivationStats::from_counts(
                    n_obs,
                    marg[a],
                    marg[b],
                    acc.n_joint,
                    weight_correlation,
                );
                (a, b, stats)
            })
            .collect()
    }

    /// Universal per-token code lengths for the binary support process, in bits.
    ///
    /// The MDL *selection* price names, per token, WHICH atoms fired. The uniform
    /// (combinatorial) price `log₂ C(G, k)` assumes every `k`-subset is equally
    /// likely — it is the WORST case, and it grossly OVERPAYS a dictionary whose
    /// co-firing supports are predictable (a tiling SAE where adjacent atoms fire
    /// together): charging that worst case would let an MDL comparison argue with
    /// itself. The honest currency is the entropy of the EMPIRICAL support
    /// distribution `H(S) = −Σ_s p(s) log₂ p(s)` over the observed supports
    /// `s ⊆ {0,…,G−1}`. `H(S)` cannot be read off directly (the support space is
    /// exponential and each token is one sample), so it is priced by the
    /// achievable code length of a LOW-ORDER model of the binary support process
    /// fit to the data — no magic constants, only empirical marginals and
    /// pairwise co-occurrence counts:
    ///
    /// Both learned models use Krichevsky–Trofimov sequential probabilities, so
    /// no fitted Bernoulli parameter is transmitted for free. The Chow–Liu code
    /// additionally transmits its data-selected labeled tree using Cayley's
    /// `G^(G−2)` possibilities before conditionally KT-coding every child. The
    /// combinatorial reference transmits each row's cardinality and then its
    /// subset, so variable support sizes are charged exactly rather than at a
    /// rounded mean.
    ///
    /// # Scale (#2933 F43)
    ///
    /// Nothing here is sized `G²`. The sufficient statistics are every atom's
    /// firing count and the joint count of every pair that co-fires at least
    /// once, accumulated from the row supports in `O(Σ_i |S_i|²)` time; memory is
    /// `O(G + N + Σ_i |S_i| + |E|)` for `|E|` distinct co-firing pairs. A pair
    /// that never co-fires still carries mutual information (mutual exclusion),
    /// and that information increases in both firing counts, so the exact
    /// maximum spanning tree of the COMPLETE graph is found by Borůvka rounds
    /// that scan the explicit edges plus, per atom, one best never-co-firing
    /// partner (see `SupportCounts::chow_liu_tree`). The KT code lengths are
    /// closed forms in the counts. The code is the same at every scale: no
    /// memory or dictionary-size threshold switches the code family.
    pub fn support_entropy(&self) -> SupportEntropy {
        let n = self.n_obs();
        let g = self.k_atoms();
        if n == 0 || g == 0 {
            return SupportEntropy {
                tree_bits: 0.0,
                independent_bits: 0.0,
                combinatorial_bits: 0.0,
                mean_support: 0.0,
            };
        }

        let counts = SupportCounts::from_codes(self);
        let rows = counts.n_rows;
        let nn = n as f64;

        let independent_total: f64 = counts
            .marginal
            .iter()
            .map(|&ones| kt_code_bits(rows - ones, ones))
            .sum();

        let tree_structure_bits = if g <= 2 {
            0.0
        } else {
            (g as f64 - 2.0) * (g as f64).log2()
        };
        let root_ones = counts.marginal[0];
        let mut tree_total = tree_structure_bits + kt_code_bits(rows - root_ones, root_ones);
        for (child, parent, joint) in counts.chow_liu_arborescence() {
            let parent_ones = counts.marginal[parent];
            let child_ones = counts.marginal[child];
            // The child is KT-coded separately in each parent context: `joint`
            // ones among the `parent_ones` rows where the parent fires, and
            // `child_ones − joint` ones among the rest.
            tree_total += kt_code_bits(parent_ones - joint, joint)
                + kt_code_bits(rows + joint - parent_ones - child_ones, child_ones - joint);
        }

        let mean_support = counts.row_atoms.len() as f64 / nn;
        let mut cardinality_counts = Vec::new();
        for bounds in counts.row_ptr.windows(2) {
            let cardinality = bounds[1] - bounds[0];
            if cardinality >= cardinality_counts.len() {
                cardinality_counts.resize(cardinality + 1, 0);
            }
            cardinality_counts[cardinality] += 1;
        }
        let combinatorial_bits = combinatorial_support_bits(g, &cardinality_counts);

        SupportEntropy {
            tree_bits: tree_total / nn,
            independent_bits: independent_total / nn,
            combinatorial_bits,
            mean_support,
        }
    }
}

/// The empirical per-token support-entropy estimate produced by
/// [`SparseAtomCodes::support_entropy`], with the two references (independent
/// model and combinatorial worst case) it is read against. All fields are BITS
/// per token except [`Self::mean_support`]. See the method for the derivation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupportEntropy {
    /// Complete Chow–Liu universal code per token, including tree transmission
    /// and conditional KT parameter learning.
    pub tree_bits: f64,
    /// Independent-Bernoulli KT universal code per token.
    pub independent_bits: f64,
    /// `log₂(G+1) + mean_i log₂ C(G, |S_i|)`: a valid variable-cardinality
    /// fixed code, unlike `log₂ C(G, round(mean |S|))`.
    pub combinatorial_bits: f64,
    /// Mean support size `k̄` (mean number of active atoms per token).
    pub mean_support: f64,
}

/// Sufficient statistics of the binary support process for
/// [`SparseAtomCodes::support_entropy`], none of them sized `G²`: the
/// row-compressed supports, every atom's firing count, and the joint count and
/// mutual information of every pair that co-fires at least once.
struct SupportCounts {
    n_rows: u64,
    /// `row_atoms[row_ptr[i]..row_ptr[i + 1]]` is row `i`'s support, ascending.
    row_ptr: Vec<usize>,
    row_atoms: Vec<usize>,
    /// Rows on which each atom fires.
    marginal: Vec<u64>,
    /// Co-firing pairs `(u, v)` with `u < v`, grouped by `u` and ascending in
    /// `v`: `upper_atom[upper_ptr[u]..upper_ptr[u + 1]]`, with the pair's joint
    /// count and mutual information at the same index of the parallel arrays.
    upper_ptr: Vec<usize>,
    upper_atom: Vec<usize>,
    upper_joint: Vec<u64>,
    upper_information: Vec<f64>,
}

/// A candidate Chow–Liu edge between atoms `a` and `b`.
#[derive(Clone, Copy, Debug)]
struct ChowLiuEdge {
    a: usize,
    b: usize,
    /// Rows on which both atoms fire (zero for a never-co-firing pair).
    joint: u64,
    information: f64,
    /// The endpoints' ranks in the (firing count descending, atom index
    /// ascending) order, smaller first.
    rank_key: (usize, usize),
}

impl ChowLiuEdge {
    /// The strict total order the maximum spanning tree is taken in: larger
    /// mutual information first, ties broken toward the lexicographically
    /// smaller endpoint-rank pair. Distinct edges have distinct rank pairs, so
    /// the maximum spanning tree is unique.
    fn outranks(&self, other: &Self) -> bool {
        match self.information.total_cmp(&other.information) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => self.rank_key < other.rank_key,
        }
    }
}

/// Union–find over atoms with path halving and union by size.
struct DisjointSets {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSets {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            self.parent[node] = self.parent[self.parent[node]];
            node = self.parent[node];
        }
        node
    }

    /// Merge the sets of `a` and `b`; `false` when they were already one set.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (mut keep, mut merged) = (self.find(a), self.find(b));
        if keep == merged {
            return false;
        }
        if self.size[keep] < self.size[merged] {
            std::mem::swap(&mut keep, &mut merged);
        }
        self.parent[merged] = keep;
        self.size[keep] += self.size[merged];
        true
    }
}

impl SupportCounts {
    fn from_codes(codes: &SparseAtomCodes) -> Self {
        let g = codes.k_atoms();
        let n_rows = codes.n_obs() as u64;
        let mut row_ptr = Vec::with_capacity(codes.n_obs() + 1);
        let mut row_atoms = Vec::new();
        let mut marginal = vec![0_u64; g];
        row_ptr.push(0);
        for code in &codes.codes {
            for atom in code.active_mask.iter_ones() {
                marginal[atom] += 1;
                row_atoms.push(atom);
            }
            row_ptr.push(row_atoms.len());
        }

        // Posting lists: the rows each atom fires on, ascending.
        let mut posting_ptr = Vec::with_capacity(g + 1);
        let mut running = 0_usize;
        posting_ptr.push(running);
        for &ones in &marginal {
            running += ones as usize;
            posting_ptr.push(running);
        }
        let mut posting_fill = posting_ptr.clone();
        let mut posting_rows = vec![0_usize; row_atoms.len()];
        for (row, bounds) in row_ptr.windows(2).enumerate() {
            for &atom in &row_atoms[bounds[0]..bounds[1]] {
                posting_rows[posting_fill[atom]] = row;
                posting_fill[atom] += 1;
            }
        }

        // Joint counts, one atom `u` at a time: every row `u` fires on
        // contributes its later atoms `v > u`. A per-`u` counter over atoms plus
        // the list of partners it touched keeps this `O(Σ_i |S_i|²)` time with
        // `O(G)` scratch, and emits only pairs that actually co-fire.
        let mut joint = vec![0_u64; g];
        let mut partners = Vec::new();
        let mut upper_ptr = Vec::with_capacity(g + 1);
        let mut upper_atom = Vec::new();
        let mut upper_joint = Vec::new();
        upper_ptr.push(0);
        for u in 0..g {
            for &row in &posting_rows[posting_ptr[u]..posting_ptr[u + 1]] {
                let support = &row_atoms[row_ptr[row]..row_ptr[row + 1]];
                let later = support.partition_point(|&atom| atom <= u);
                for &v in &support[later..] {
                    if joint[v] == 0 {
                        partners.push(v);
                    }
                    joint[v] += 1;
                }
            }
            partners.sort_unstable();
            for &v in &partners {
                upper_atom.push(v);
                upper_joint.push(joint[v]);
                joint[v] = 0;
            }
            partners.clear();
            upper_ptr.push(upper_atom.len());
        }

        let mut upper_information = Vec::with_capacity(upper_atom.len());
        for u in 0..g {
            for index in upper_ptr[u]..upper_ptr[u + 1] {
                upper_information.push(mutual_information_bits(
                    n_rows,
                    marginal[u],
                    marginal[upper_atom[index]],
                    upper_joint[index],
                ));
            }
        }

        Self {
            n_rows,
            row_ptr,
            row_atoms,
            marginal,
            upper_ptr,
            upper_atom,
            upper_joint,
            upper_information,
        }
    }

    /// Whether distinct atoms `u` and `v` fire together on at least one row.
    fn co_fire(&self, u: usize, v: usize) -> bool {
        let (low, high) = if u < v { (u, v) } else { (v, u) };
        self.upper_atom[self.upper_ptr[low]..self.upper_ptr[low + 1]]
            .binary_search(&high)
            .is_ok()
    }

    /// The maximum-mutual-information spanning tree of the COMPLETE graph on
    /// the atoms, in the strict order of [`ChowLiuEdge::outranks`].
    ///
    /// Borůvka: each round, every component takes its best edge leaving it,
    /// and those edges are merged (the cut property; the strict order rules out
    /// cycles). A never-co-firing pair `(u, x)` is an edge too, with information
    /// `I₀(n_u, n_x)`. For firing rates `a, b` with `a + b ≤ 1`,
    /// `I₀ = φ(1−a−b) − φ(1−a) − φ(1−b)` with `φ(t) = t ln t`, so
    /// `∂I₀/∂b = ln((1−b)/(1−a−b)) ≥ 0`: the information does not decrease in the
    /// partner's firing count, and on equal information the rank tie-break
    /// prefers the higher-ranked partner. Hence the best never-co-firing edge
    /// from `u` out of its component goes to the FIRST atom in rank order that
    /// lies outside the component and does not co-fire with `u`. Finding it
    /// skips at most `deg(u)` co-firing partners and, by a precomputed
    /// next-different-component jump, at most `deg(u) + 1` runs of the own
    /// component, so a round costs `O(G + |E| log |E|)` and there are at most
    /// `log₂ G` rounds.
    ///
    /// The order argument needs the COMPUTED `I₀` to be monotone in the counts.
    /// [`mutual_information_bits`] evaluates it from the exact integer
    /// cross-product determinant through `ln_1p`, so its relative error is a
    /// few ulps, while one count changes `I₀` by a relative amount of order
    /// `1/N` or more: the computed order holds for corpora far below `10¹⁴`
    /// tokens.
    fn chow_liu_tree(&self) -> Vec<ChowLiuEdge> {
        let g = self.marginal.len();
        let mut order: Vec<usize> = (0..g).collect();
        order.sort_unstable_by(|&a, &b| {
            self.marginal[b]
                .cmp(&self.marginal[a])
                .then(a.cmp(&b))
        });
        let mut rank = vec![0_usize; g];
        for (position, &atom) in order.iter().enumerate() {
            rank[atom] = position;
        }
        let edge = |a: usize, b: usize, joint: u64, information: f64| ChowLiuEdge {
            a,
            b,
            joint,
            information,
            rank_key: (rank[a].min(rank[b]), rank[a].max(rank[b])),
        };
        let offer = |slot: &mut Option<ChowLiuEdge>, candidate: ChowLiuEdge| {
            if slot.is_none_or(|incumbent| candidate.outranks(&incumbent)) {
                *slot = Some(candidate);
            }
        };

        let mut sets = DisjointSets::new(g);
        let mut component = vec![0_usize; g];
        // `next_foreign[p]`: the first rank position after `p` whose atom lies in
        // a different component from the atom at `p` (or `g`).
        let mut next_foreign = vec![g; g];
        let mut best: Vec<Option<ChowLiuEdge>> = vec![None; g];
        let mut tree = Vec::with_capacity(g.saturating_sub(1));
        while tree.len() + 1 < g {
            for atom in 0..g {
                component[atom] = sets.find(atom);
            }
            for position in (0..g - 1).rev() {
                next_foreign[position] =
                    if component[order[position + 1]] != component[order[position]] {
                        position + 1
                    } else {
                        next_foreign[position + 1]
                    };
            }
            best.fill(None);

            // Explicit edges: pairs that co-fire.
            for u in 0..g {
                for index in self.upper_ptr[u]..self.upper_ptr[u + 1] {
                    let v = self.upper_atom[index];
                    if component[u] != component[v] {
                        let candidate =
                            edge(u, v, self.upper_joint[index], self.upper_information[index]);
                        offer(&mut best[component[u]], candidate);
                        offer(&mut best[component[v]], candidate);
                    }
                }
            }

            // Implicit edges: each atom's best never-co-firing partner outside
            // its component.
            for u in 0..g {
                let mut position = 0;
                while position < g {
                    let x = order[position];
                    if component[x] == component[u] {
                        position = next_foreign[position];
                    } else if self.co_fire(u, x) {
                        position += 1;
                    } else {
                        let information =
                            mutual_information_bits(self.n_rows, self.marginal[u], self.marginal[x], 0);
                        offer(&mut best[component[u]], edge(u, x, 0, information));
                        break;
                    }
                }
            }

            for root in 0..g {
                if let Some(candidate) = best[root]
                    && sets.union(candidate.a, candidate.b)
                {
                    tree.push(candidate);
                }
            }
        }
        tree
    }

    /// The Chow–Liu tree rooted at atom 0, as `(child, parent, joint count)` for
    /// every non-root atom.
    fn chow_liu_arborescence(&self) -> Vec<(usize, usize, u64)> {
        let g = self.marginal.len();
        let tree = self.chow_liu_tree();
        let mut tree_ptr = vec![0_usize; g + 1];
        for edge in &tree {
            tree_ptr[edge.a + 1] += 1;
            tree_ptr[edge.b + 1] += 1;
        }
        for atom in 0..g {
            tree_ptr[atom + 1] += tree_ptr[atom];
        }
        let mut fill = tree_ptr.clone();
        let mut neighbours = vec![(0_usize, 0_u64); 2 * tree.len()];
        for edge in &tree {
            neighbours[fill[edge.a]] = (edge.b, edge.joint);
            fill[edge.a] += 1;
            neighbours[fill[edge.b]] = (edge.a, edge.joint);
            fill[edge.b] += 1;
        }
        let mut visited = vec![false; g];
        visited[0] = true;
        let mut arborescence = Vec::with_capacity(tree.len());
        let mut frontier = vec![0_usize];
        while let Some(parent) = frontier.pop() {
            for &(child, joint) in &neighbours[tree_ptr[parent]..tree_ptr[parent + 1]] {
                if !visited[child] {
                    visited[child] = true;
                    arborescence.push((child, parent, joint));
                    frontier.push(child);
                }
            }
        }
        arborescence
    }
}

/// Krichevsky–Trofimov code length, in bits, of a binary sequence holding
/// `zeros` zeros and `ones` ones. The sequential KT predictor (Jeffreys
/// half-counts, which give every one-sample outcome one bit instead of the
/// zero-bit plug-in pathology) is exchangeable: the product of its predictive
/// probabilities depends only on the counts,
/// `P = Γ(zeros+½) Γ(ones+½) / (Γ(½)² Γ(zeros+ones+1))`, so the length is
/// evaluated in closed form rather than by replaying the sequence. Shared by
/// [`SparseAtomCodes::support_entropy`] and the Eq. 4 support charge.
pub(crate) fn kt_code_bits(zeros: u64, ones: u64) -> f64 {
    if zeros == 0 && ones == 0 {
        return 0.0;
    }
    let ln_gamma_half = 0.5 * PI.ln();
    (ln_gamma((zeros + ones) as f64 + 1.0) + 2.0 * ln_gamma_half
        - ln_gamma(zeros as f64 + 0.5)
        - ln_gamma(ones as f64 + 0.5))
        / LN_2
}

/// Per-token length, in bits, of the cardinality-then-subset support code:
/// each row names its cardinality `k ∈ {0,…,G}` uniformly (`log₂(G+1)` bits)
/// and then which `k`-subset of the `G` atoms fired (`log₂ C(G, k)` bits), so
/// variable support sizes are charged exactly rather than at a rounded mean.
/// `cardinality_counts[k]` is the number of rows of cardinality `k`; the slice
/// need only reach the largest cardinality present. Cardinalities with no rows
/// are skipped, so the cost is `O(len + Σ_i |S_i|)`. Zero when there are no
/// rows. Shared by [`SparseAtomCodes::support_entropy`] and the Eq. 4 support
/// charge.
pub(crate) fn combinatorial_support_bits(g: usize, cardinality_counts: &[usize]) -> f64 {
    let rows: usize = cardinality_counts.iter().sum();
    if rows == 0 {
        return 0.0;
    }
    let subset_bits: f64 = cardinality_counts
        .iter()
        .enumerate()
        .filter(|entry| *entry.1 > 0)
        .map(|(cardinality, &count)| count as f64 * log2_binom(g as i64, cardinality as i64))
        .sum();
    (g as f64 + 1.0).log2() + subset_bits / rows as f64
}

/// Plug-in mutual information `I(x_u; x_v)` in bits of two binary indicators
/// over `n` rows, from the firing counts `n_u`, `n_v` and the joint count
/// `n_uv`.
///
/// Each cell contributes `c_ij · ln(n·c_ij / (r_i·s_j))` with row and column
/// totals `r_i`, `s_j`, and `n·c_ij − r_i·s_j = ±D` for the cross-product
/// determinant `D = c₁₁c₀₀ − c₁₀c₀₁ = n·n_uv − n_u·n_v` (plus on the diagonal
/// cells), so the log-ratio is `ln_1p(±D / (r_i·s_j))` with `D` exact in
/// integers. The direct `p·log(p / (p_u·p_v))` form loses the `p₀₀` cell to
/// rounding once `n_u·n_v / n²` nears machine epsilon — rare atoms at corpus
/// scale — and can then invert the order of two never-co-firing pairs, which
/// `SupportCounts::chow_liu_tree` relies on. Exactly zero when `D = 0`, and
/// bit-for-bit symmetric in the two atoms.
fn mutual_information_bits(n: u64, n_u: u64, n_v: u64, n_uv: u64) -> f64 {
    let (n_u, n_v) = (n_u.min(n_v), n_u.max(n_v));
    let c11 = n_uv;
    let c10 = n_u - n_uv;
    let c01 = n_v - n_uv;
    let c00 = n + n_uv - n_u - n_v;
    let determinant = i128::from(c11) * i128::from(c00) - i128::from(c10) * i128::from(c01);
    if determinant == 0 {
        return 0.0;
    }
    let determinant = determinant as f64;
    let cell = |count: u64, signed_determinant: f64, row_total: u64, column_total: u64| {
        if count == 0 {
            0.0
        } else {
            count as f64 * (signed_determinant / (row_total as f64 * column_total as f64)).ln_1p()
        }
    };
    let nats = cell(c11, determinant, n_u, n_v)
        + cell(c10, -determinant, n_u, n - n_v)
        + cell(c01, -determinant, n - n_u, n_v)
        + cell(c00, determinant, n - n_u, n - n_v);
    let bits = nats / (n as f64 * LN_2);
    if bits > 0.0 { bits } else { 0.0 }
}

/// `log₂ C(g, k)`: bits to name which `k` of `g` atoms fired under the uniform
/// support prior. Computed as `Σ_{i=1..k} log₂((g−k+i)/i)` so it never overflows
/// a binomial. Zero when `g ≤ 0` or `k ≤ 0`; `k` is capped at `g`. (The same
/// combinatorial bound [`crate::description_length::selection_bits`] reports; kept
/// local so the support-entropy estimator stays self-contained.)
fn log2_binom(g: i64, k: i64) -> f64 {
    if g <= 0 || k <= 0 {
        return 0.0;
    }
    let k = k.min(g);
    let mut bits = 0.0;
    for i in 1..=k {
        bits += ((g - k + i) as f64 / i as f64).log2();
    }
    bits
}

/// Pairwise co-activation summary for two atoms (see
/// `SparseAtomCodes::coactivation`). All probabilities are empirical
/// popcount ratios over the active-support masks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoactivationStats {
    /// Total number of observations the codes cover.
    pub n_obs: usize,
    /// Rows where atom `a` is active.
    pub n_a: usize,
    /// Rows where atom `b` is active.
    pub n_b: usize,
    /// Rows where both are active.
    pub n_joint: usize,
    /// `P(a active | b active)`; `0` when `b` is never active.
    pub p_a_given_b: f64,
    /// `P(b active | a active)`; `0` when `a` is never active.
    pub p_b_given_a: f64,
    /// `P(a∧b) / (P(a)·P(b))`; `1` for independent atoms, `0` when either
    /// marginal is empty.
    pub lift: f64,
    /// Pearson correlation of the two atoms' activation WEIGHTS over the
    /// jointly-active rows (accumulated in `SparseAtomCodes::coactive_pair_stats`) — the
    /// amplitude/interaction half of the fusion criterion. `0` when the joint
    /// support is too small or a weight is constant there.
    pub weight_correlation: f64,
}

impl CoactivationStats {
    fn from_counts(
        n_obs: usize,
        n_a: usize,
        n_b: usize,
        n_joint: usize,
        weight_correlation: f64,
    ) -> Self {
        let cond = |joint: usize, marg: usize| {
            if marg == 0 {
                0.0
            } else {
                joint as f64 / marg as f64
            }
        };
        let lift = if n_a == 0 || n_b == 0 || n_obs == 0 {
            0.0
        } else {
            (n_joint as f64 * n_obs as f64) / (n_a as f64 * n_b as f64)
        };
        Self {
            n_obs,
            n_a,
            n_b,
            n_joint,
            p_a_given_b: cond(n_joint, n_b),
            p_b_given_a: cond(n_joint, n_a),
            lift,
            weight_correlation,
        }
    }

    /// Symmetric code dependence `min(P(a|b), P(b|a))` — the canonical-order
    /// trigger for FUSION proposals (descending). Near 0 for independent or
    /// disjoint atoms; near 1 only when the two supports essentially coincide,
    /// which is the shattering signature.
    pub fn dependence(&self) -> f64 {
        self.p_a_given_b.min(self.p_b_given_a)
    }

    /// Conditional asymmetry `|P(a|b) − P(b|a)|` — large when one atom's
    /// support nests inside the other's (the A⇒B absorption signature, where
    /// `P(parent|child) ≈ 1` but not conversely). Flags the pair for a
    /// targeted within-atom substructure audit; it is never itself an
    /// acceptance criterion.
    pub fn absorption_asymmetry(&self) -> f64 {
        (self.p_a_given_b - self.p_b_given_a).abs()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitvec_basic() {
        let mut bv = BitVec::zeros(70);
        assert_eq!(bv.len(), 70);
        assert!(!bv.get(5));
        bv.set(5, true);
        bv.set(64, true);
        assert!(bv.get(5));
        assert!(bv.get(64));
        assert_eq!(bv.count_ones(), 2);
        let ones: Vec<usize> = bv.iter_ones().collect();
        assert_eq!(ones, vec![5, 64]);
        bv.set(5, false);
        assert_eq!(bv.count_ones(), 1);
    }

    // ---- #2933 F43 references -------------------------------------------------

    /// The production mutual information, through one adapter so the reference
    /// code below reads the same edge weights the coder ranks.
    fn mi_bits(n: u64, n_u: u64, n_v: u64, n_uv: u64) -> f64 {
        mutual_information_bits(n, n_u, n_v, n_uv)
    }

    /// The direct cell formula over probabilities (the pre-F43 implementation).
    fn direct_cell_mutual_information_bits(n: f64, n_u: f64, n_v: f64, n_uv: f64) -> f64 {
        if n <= 0.0 {
            return 0.0;
        }
        let p1x = n_u / n;
        let px1 = n_v / n;
        let p11 = n_uv / n;
        let p10 = (p1x - p11).max(0.0);
        let p01 = (px1 - p11).max(0.0);
        let p00 = (1.0 - p11 - p10 - p01).max(0.0);
        let cell = |p: f64, pa: f64, pb: f64| -> f64 {
            if p > 0.0 && pa > 0.0 && pb > 0.0 {
                p * (p / (pa * pb)).log2()
            } else {
                0.0
            }
        };
        let mi = cell(p11, p1x, px1)
            + cell(p10, p1x, 1.0 - px1)
            + cell(p01, 1.0 - p1x, px1)
            + cell(p00, 1.0 - p1x, 1.0 - px1);
        mi.max(0.0)
    }

    /// Sequential KT code, replaying the decoder's predictive probabilities row
    /// by row.
    fn sequential_kt_bits(values: impl IntoIterator<Item = bool>) -> f64 {
        let mut counts = [0.5_f64, 0.5_f64];
        let mut bits = 0.0;
        for value in values {
            let index = usize::from(value);
            let probability = counts[index] / (counts[0] + counts[1]);
            bits -= probability.log2();
            counts[index] += 1.0;
        }
        bits
    }

    /// Sequential conditional KT code with one predictor per parent context.
    fn sequential_conditional_kt_bits(values: impl IntoIterator<Item = (bool, bool)>) -> f64 {
        let mut counts = [[0.5_f64, 0.5_f64], [0.5_f64, 0.5_f64]];
        let mut bits = 0.0;
        for (context, value) in values {
            let context_index = usize::from(context);
            let value_index = usize::from(value);
            let probability = counts[context_index][value_index]
                / (counts[context_index][0] + counts[context_index][1]);
            bits -= probability.log2();
            counts[context_index][value_index] += 1.0;
        }
        bits
    }

    fn codes_from_bits(bits: &[Vec<bool>]) -> SparseAtomCodes {
        let g = bits.first().map_or(0, Vec::len);
        let mut codes = SparseAtomCodes::empty(bits.len(), g);
        for (row, values) in bits.iter().enumerate() {
            for (atom, &on) in values.iter().enumerate() {
                if on {
                    codes.row_mut(row).assign(atom, 1.0);
                }
            }
        }
        codes
    }

    fn column(bits: &[Vec<bool>], atom: usize) -> impl Iterator<Item = bool> + '_ {
        bits.iter().map(move |row| row[atom])
    }

    /// Dense reference for the declared code: a `G × G` joint-count scan of the
    /// raw bits, every pair's mutual information, Kruskal in the declared order
    /// (information descending, then the sorted endpoint ranks in the firing
    /// count descending / index ascending order), and sequential KT codes over
    /// the rows. Returns `(tree_bits, independent_bits)` per token.
    fn dense_chow_liu_reference(bits: &[Vec<bool>]) -> (f64, f64) {
        let n = bits.len();
        let g = bits[0].len();
        let marginal: Vec<u64> = (0..g)
            .map(|atom| column(bits, atom).filter(|&on| on).count() as u64)
            .collect();
        let mut order: Vec<usize> = (0..g).collect();
        order.sort_by(|&a, &b| marginal[b].cmp(&marginal[a]).then(a.cmp(&b)));
        let mut rank = vec![0_usize; g];
        for (position, &atom) in order.iter().enumerate() {
            rank[atom] = position;
        }
        let mut edges: Vec<(f64, (usize, usize), usize, usize)> = Vec::new();
        for a in 0..g {
            for b in a + 1..g {
                let joint = bits.iter().filter(|row| row[a] && row[b]).count() as u64;
                let information = mi_bits(n as u64, marginal[a], marginal[b], joint);
                let key = (rank[a].min(rank[b]), rank[a].max(rank[b]));
                edges.push((information, key, a, b));
            }
        }
        edges.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        let mut label: Vec<usize> = (0..g).collect();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); g];
        for edge in &edges {
            let (a, b) = (edge.2, edge.3);
            if label[a] != label[b] {
                let (keep, merged) = (label[a], label[b]);
                for entry in label.iter_mut() {
                    if *entry == merged {
                        *entry = keep;
                    }
                }
                adjacency[a].push(b);
                adjacency[b].push(a);
            }
        }
        let mut parent = vec![usize::MAX; g];
        let mut visited = vec![false; g];
        visited[0] = true;
        let mut stack = vec![0_usize];
        while let Some(node) = stack.pop() {
            for &next in &adjacency[node] {
                if !visited[next] {
                    visited[next] = true;
                    parent[next] = node;
                    stack.push(next);
                }
            }
        }
        let independent: f64 = (0..g).map(|atom| sequential_kt_bits(column(bits, atom))).sum();
        let mut tree = sequential_kt_bits(column(bits, 0));
        if g > 2 {
            tree += (g as f64 - 2.0) * (g as f64).log2();
        }
        for child in 1..g {
            let p = parent[child];
            tree += sequential_conditional_kt_bits(bits.iter().map(|row| (row[p], row[child])));
        }
        (tree / n as f64, independent / n as f64)
    }

    /// The dense `G²` Prim coder this module shipped before #2933 F43, verbatim
    /// over its direct cell formula and sequential KT codes. Returns tree bits
    /// per token.
    fn dense_prim_reference_tree_bits(bits: &[Vec<bool>]) -> f64 {
        let n = bits.len();
        let g = bits[0].len();
        let mut marg = vec![0.0_f64; g];
        let mut co = vec![0.0_f64; g * g];
        for row in bits {
            let active: Vec<usize> = (0..g).filter(|&atom| row[atom]).collect();
            for (idx, &u) in active.iter().enumerate() {
                marg[u] += 1.0;
                for &v in &active[idx + 1..] {
                    co[u * g + v] += 1.0;
                }
            }
        }
        let nn = n as f64;
        let mi = |u: usize, v: usize| -> f64 {
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            direct_cell_mutual_information_bits(nn, marg[a], marg[b], co[a * g + b])
        };
        let mut in_tree = vec![false; g];
        let mut best_mi = vec![f64::NEG_INFINITY; g];
        let mut best_parent = vec![0usize; g];
        let mut parent = vec![0usize; g];
        in_tree[0] = true;
        for v in 1..g {
            best_mi[v] = mi(0, v);
            best_parent[v] = 0;
        }
        for step in 1..g {
            let mut pick = usize::MAX;
            let mut pick_w = f64::NEG_INFINITY;
            for v in 0..g {
                if !in_tree[v] && best_mi[v] > pick_w {
                    pick_w = best_mi[v];
                    pick = v;
                }
            }
            assert!(pick != usize::MAX, "Prim step {step} found no out-of-tree atom");
            in_tree[pick] = true;
            parent[pick] = best_parent[pick];
            for v in 0..g {
                if !in_tree[v] {
                    let w = mi(pick, v);
                    if w > best_mi[v] {
                        best_mi[v] = w;
                        best_parent[v] = pick;
                    }
                }
            }
        }
        let mut tree_total = sequential_kt_bits(column(bits, 0));
        if g > 2 {
            tree_total += (g as f64 - 2.0) * (g as f64).log2();
        }
        for child in 1..g {
            tree_total +=
                sequential_conditional_kt_bits(bits.iter().map(|row| (row[parent[child]], row[child])));
        }
        tree_total / nn
    }

    use gam_linalg::utils::splitmix64;

    fn draw(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Ten atoms carrying every tie the declared order has to break: a
    /// duplicated and a complementary column, two never-firing atoms, an
    /// always-firing atom, a rare atom, two atoms at the same rate, and a
    /// nested atom.
    fn tied_support_fixture(seed: u64, n: usize) -> Vec<Vec<bool>> {
        let mut state = seed;
        (0..n)
            .map(|row| {
                let base = draw(&mut state) < 0.5;
                let six = draw(&mut state) < 0.3;
                let seven = draw(&mut state) < 0.3;
                let nested = six && draw(&mut state) < 0.7;
                vec![
                    base,
                    base,
                    !base,
                    false,
                    true,
                    row % 17 == 3,
                    six,
                    seven,
                    nested,
                    false,
                ]
            })
            .collect()
    }

    fn sparse_random_fixture(seed: u64, n: usize, g: usize, rate: f64) -> Vec<Vec<bool>> {
        let mut state = seed;
        (0..n)
            .map(|_| (0..g).map(|_| draw(&mut state) < rate).collect())
            .collect()
    }

    #[test]
    fn sparse_chow_liu_support_code_matches_dense_reference_2933_f43() {
        let seeds = [11_u64, 29, 47, 83];
        let mut fixtures = Vec::new();
        for seed in seeds {
            fixtures.push(tied_support_fixture(seed, 40));
            fixtures.push(sparse_random_fixture(seed, 25, 12, 0.15));
        }
        for (index, bits) in fixtures.iter().enumerate() {
            let (tree, independent) = dense_chow_liu_reference(bits);
            let got = codes_from_bits(bits).support_entropy();
            assert!(
                (got.tree_bits - tree).abs() <= 1e-10 * tree.abs().max(1.0),
                "fixture {index}: sparse Chow-Liu tree bits {} != dense reference {tree}",
                got.tree_bits
            );
            assert!(
                (got.independent_bits - independent).abs() <= 1e-10 * independent.abs().max(1.0),
                "fixture {index}: closed-form KT bits {} != replayed reference {independent}",
                got.independent_bits
            );
        }
        // Positive control: the duplicated and complementary columns make the
        // tree code shorter than the independent code, so the agreement above is
        // on a tree that captures dependence.
        for seed in seeds {
            let got = codes_from_bits(&tied_support_fixture(seed, 40)).support_entropy();
            assert!(
                got.tree_bits < got.independent_bits,
                "seed {seed}: tree bits {} must undercut independent bits {}",
                got.tree_bits,
                got.independent_bits
            );
        }
    }

    #[test]
    fn sparse_chow_liu_reproduces_dense_prim_code_on_tie_free_support_2933_f43() {
        let mut state = 0x2933_u64;
        let n = 211_usize;
        let bits: Vec<Vec<bool>> = (0..n)
            .map(|_| {
                let x0 = draw(&mut state) < 0.5;
                let x1 = x0 && draw(&mut state) < 0.7;
                let x2 = draw(&mut state) < 0.2;
                let x3 = x2 || draw(&mut state) < 0.5;
                let x4 = draw(&mut state) < 0.45;
                let x5 = !x4 && draw(&mut state) < 0.5;
                vec![x0, x1, x2, x3, x4, x5]
            })
            .collect();
        let g = bits[0].len();
        // With `n` prime and every firing count in (0, n), `n·n_uv ≠ n_u·n_v`, so
        // no pair is exactly independent. The pairwise weights must also be
        // separated, so the maximum spanning tree is unique whatever the tie rule.
        let mut weights = Vec::new();
        for a in 0..g {
            let n_a = column(&bits, a).filter(|&on| on).count() as u64;
            assert!(n_a > 0 && (n_a as usize) < n, "atom {a} fires on {n_a} of {n} rows");
            for b in a + 1..g {
                let n_b = column(&bits, b).filter(|&on| on).count() as u64;
                let joint = bits.iter().filter(|row| row[a] && row[b]).count() as u64;
                weights.push(mi_bits(n as u64, n_a, n_b, joint));
            }
        }
        weights.sort_by(f64::total_cmp);
        for pair in weights.windows(2) {
            assert!(
                pair[1] - pair[0] > 1e-9 * pair[1],
                "fixture has near-tied edge weights {} and {}",
                pair[0],
                pair[1]
            );
        }
        let reference = dense_prim_reference_tree_bits(&bits);
        let got = codes_from_bits(&bits).support_entropy().tree_bits;
        assert!(
            (got - reference).abs() <= 1e-10 * reference.max(1.0),
            "sparse Chow-Liu tree bits {got} != dense Prim coder {reference}"
        );
    }

    #[test]
    fn support_codes_satisfy_kraft_over_every_support_matrix_2933_f43() {
        // Summing 2^(−N·bits) over every N×G binary support matrix: the
        // independent KT code and the cardinality-then-subset code are complete
        // (sum exactly one). The tree code spends (G−2)·log₂G bits naming one of
        // G^(G−2) labelled trees and then a KT code given that tree, so its sum
        // is at most one, and exactly one when G ≤ 2 leaves a single tree.
        for (n, g) in [(3_usize, 1_usize), (4, 2), (2, 3), (3, 3), (3, 4)] {
            let cells = n * g;
            let tokens = n as f64;
            let (mut tree, mut independent, mut combinatorial) = (0.0_f64, 0.0_f64, 0.0_f64);
            for mask in 0_u64..(1_u64 << cells) {
                let mut codes = SparseAtomCodes::empty(n, g);
                for cell in 0..cells {
                    if (mask >> cell) & 1 == 1 {
                        codes.row_mut(cell / g).assign(cell % g, 1.0);
                    }
                }
                let entropy = codes.support_entropy();
                tree += 2.0_f64.powf(-tokens * entropy.tree_bits);
                independent += 2.0_f64.powf(-tokens * entropy.independent_bits);
                combinatorial += 2.0_f64.powf(-tokens * entropy.combinatorial_bits);
            }
            assert!(
                (independent - 1.0).abs() <= 1e-12,
                "N={n}, G={g}: independent KT Kraft sum {independent} != 1"
            );
            assert!(
                (combinatorial - 1.0).abs() <= 1e-12,
                "N={n}, G={g}: combinatorial Kraft sum {combinatorial} != 1"
            );
            if g <= 2 {
                assert!(
                    (tree - 1.0).abs() <= 1e-12,
                    "N={n}, G={g}: single-tree Chow-Liu Kraft sum {tree} != 1"
                );
            } else {
                assert!(
                    tree <= 1.0 + 1e-12,
                    "N={n}, G={g}: Chow-Liu Kraft sum {tree} exceeds 1"
                );
            }
        }
    }

    #[test]
    fn mutual_information_of_rare_exclusive_atoms_is_accurate_and_monotone_at_corpus_scale_2933_f43()
    {
        // Two atoms that never co-fire, at rates a, b: I₀ = ψ(a+b) − ψ(a) − ψ(b)
        // with ψ(t) = Σ_{k≥2} t^k / (k(k−1)), i.e. ab + (a²b + ab²)/2 + O(t⁴) nats.
        // At N = 10¹⁰ the truncation is ~10⁻¹⁷ relative, while the direct cell
        // formula's p₀₀ log-ratio has absolute rounding ~10⁻¹⁶ against a value
        // ~10⁻²⁰.
        let n = 10_000_000_000_u64;
        for n_u in [1_u64, 2, 7] {
            let mut previous = 0.0_f64;
            for n_v in 1_u64..=48 {
                let got = mi_bits(n, n_u, n_v, 0);
                let a = n_u as f64 / n as f64;
                let b = n_v as f64 / n as f64;
                let expected = (a * b + 0.5 * (a * a * b + a * b * b)) / LN_2;
                assert!(
                    (got - expected).abs() <= 1e-9 * expected,
                    "I0(n_u={n_u}, n_v={n_v}, N={n}) = {got}, expected {expected}"
                );
                assert!(
                    got > previous,
                    "I0 must increase in the partner's count: n_u={n_u}, n_v={n_v}: {got} <= {previous}"
                );
                previous = got;
            }
        }
        // At moderate N the direct cell formula is accurate and is the reference.
        for (n, n_u, n_v, n_uv) in [
            (1000_u64, 500_u64, 500_u64, 500_u64),
            (1000, 500, 500, 0),
            (1000, 300, 200, 90),
            (1000, 300, 200, 10),
            (997, 1, 996, 0),
            (1000, 400, 250, 100),
        ] {
            let got = mi_bits(n, n_u, n_v, n_uv);
            let reference =
                direct_cell_mutual_information_bits(n as f64, n_u as f64, n_v as f64, n_uv as f64);
            assert!(
                (got - reference).abs() <= 1e-10 * reference.max(1e-3),
                "MI(N={n}, n_u={n_u}, n_v={n_v}, n_uv={n_uv}) = {got}, direct cells {reference}"
            );
        }
    }
}
