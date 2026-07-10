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
//!   sparsely; for the typical SAE workload `K` is small (tens to low
//!   hundreds), and the dense layout lets us reuse [`ndarray`] views and
//!   simple BLAS-shaped loops downstream. The mask carries the discrete
//!   active-set information; the weights carry the soft amplitudes.
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

use std::collections::BTreeMap;

use ndarray::Array1;

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

    /// Iterator over set indices in ascending order.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| self.get(i))
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

    /// Sum of active weights. For simplex-projected codes this should be ≈ 1.
    pub fn active_weight_sum(&self) -> f64 {
        self.active_mask.iter_ones().map(|k| self.weights[k]).sum()
    }

    /// Set the weight for atom `k` and mark it active.
    pub fn assign(&mut self, k: usize, w: f64) {
        assert!(k < self.k_atoms());
        self.active_mask.set(k, true);
        self.weights[k] = w;
    }

    /// Deactivate atom `k` and zero its stored weight.
    pub fn deactivate(&mut self, k: usize) {
        assert!(k < self.k_atoms());
        self.active_mask.set(k, false);
        self.weights[k] = 0.0;
    }

    /// Materialize the *effective* weight vector (zeros at inactive indices)
    /// as an owned `Array1`. Useful for matmul-shaped downstream code.
    pub fn effective_weights(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.k_atoms());
        for k in self.active_mask.iter_ones() {
            out[k] = self.weights[k];
        }
        out
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

    /// Flatten weights into a single `(N, K)` array, with zeros where the
    /// mask is unset. Allocates; intended for diagnostic / post-fit use.
    pub fn weights_matrix(&self) -> ndarray::Array2<f64> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = ndarray::Array2::<f64>::zeros((n, k));
        for n_idx in 0..n {
            let code = &self.codes[n_idx];
            for kk in code.active_mask.iter_ones() {
                out[[n_idx, kk]] = code.weights[kk];
            }
        }
        out
    }

    /// Co-activation statistics for one atom pair `(a, b)` — the #976
    /// code-dependence trigger. Pure popcount ratios over the active masks:
    /// `P(a|b) = #{rows: a∧b} / #{rows: b}` and symmetrically.
    ///
    /// Two derived readings drive the structure search:
    ///
    /// * [`CoactivationStats::dependence`] (symmetric, the FUSION trigger) —
    ///   independent atoms with marginal activation rates `π_a, π_b` co-activate
    ///   at rate `π_a·π_b`, so both conditionals stay near the marginals; a
    ///   shattered curved family re-encoded as several near-duplicate atoms
    ///   pushes *both* conditionals toward 1.
    /// * [`CoactivationStats::absorption_asymmetry`] (the ABSORPTION-audit
    ///   trigger) — an A⇒B hierarchy where sparsity folded B's content into A
    ///   shows `P(parent|child) ≈ 1` without the converse, so a large asymmetry
    ///   with one conditional near 1 flags the pair for the within-atom
    ///   substructure audit (#907 race on the atom's own code distribution).
    ///
    /// These are *triggers*, not decisions: they rank move proposals
    /// deterministically; acceptance is owned by the e-process gates in
    /// [`gam_solve::structure_search`].
    pub fn coactivation(&self, a: usize, b: usize) -> CoactivationStats {
        assert!(
            a < self.k_atoms && b < self.k_atoms,
            "SparseAtomCodes::coactivation: atoms ({a}, {b}) out of range K={}",
            self.k_atoms
        );
        let n_obs = self.n_obs();
        let mut n_a = 0usize;
        let mut n_b = 0usize;
        let mut n_joint = 0usize;
        for code in &self.codes {
            let on_a = code.active_mask.get(a);
            let on_b = code.active_mask.get(b);
            n_a += usize::from(on_a);
            n_b += usize::from(on_b);
            n_joint += usize::from(on_a && on_b);
        }
        CoactivationStats::from_counts(n_obs, n_a, n_b, n_joint, self.weight_codependence(a, b))
    }

    /// All atom pairs that co-fire at least once, with their support and
    /// amplitude-code statistics, computed in one sparse pass over row supports.
    ///
    /// This is the structure-search candidate index: rows contribute only their
    /// active-set pairs, so the producer cost is `Σ_row |S_row|²` and the output
    /// is bounded by observed co-firings, not by `K²`.
    pub fn coactive_pair_stats(&self) -> Vec<(usize, usize, CoactivationStats)> {
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

    /// #976 — the AMPLITUDE half of the fusion criterion: the Pearson
    /// correlation of the two atoms' activation WEIGHTS over the rows where both
    /// are active. Support co-activation ([`CoactivationStats::dependence`]) only
    /// says the two atoms fire together; it cannot distinguish a single curved
    /// family SHATTERED across two near-duplicate atoms (where moving along the
    /// family smoothly trades amplitude between the pair, so their weights are
    /// strongly — typically negatively — correlated on the joint support) from
    /// two GENUINELY INDEPENDENT atoms that merely happen to co-fire on the same
    /// input class (weights uncorrelated). The magnitude `|ρ|` of this
    /// correlation is the interaction-evidence the issue's fusion trigger pairs
    /// with code dependence: high support-overlap AND high `|weight_correlation|`
    /// is the shattering signature ("dependent codes + joint interaction
    /// evidence"), whereas high overlap with `|ρ|≈0` is two independent features
    /// that should NOT be fused.
    ///
    /// Returns `0.0` when fewer than two rows are jointly active or when either
    /// atom's weight is constant on the joint support (an undefined correlation
    /// is, for the trigger, "no amplitude dependence detected").
    pub fn weight_codependence(&self, a: usize, b: usize) -> f64 {
        assert!(
            a < self.k_atoms && b < self.k_atoms,
            "SparseAtomCodes::weight_codependence: atoms ({a}, {b}) out of range K={}",
            self.k_atoms
        );
        let mut wa = Vec::new();
        let mut wb = Vec::new();
        for code in &self.codes {
            if code.active_mask.get(a) && code.active_mask.get(b) {
                wa.push(code.weights[a]);
                wb.push(code.weights[b]);
            }
        }
        let m = wa.len();
        if m < 2 {
            return 0.0;
        }
        let inv = 1.0 / m as f64;
        let mean_a: f64 = wa.iter().sum::<f64>() * inv;
        let mean_b: f64 = wb.iter().sum::<f64>() * inv;
        let mut cov = 0.0_f64;
        let mut var_a = 0.0_f64;
        let mut var_b = 0.0_f64;
        for i in 0..m {
            let da = wa[i] - mean_a;
            let db = wb[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        if !(var_a > 0.0 && var_b > 0.0) {
            return 0.0;
        }
        let rho = cov / (var_a.sqrt() * var_b.sqrt());
        // Numerical clamp: accumulation can nudge a perfect ±1 a hair past the
        // bound.
        rho.clamp(-1.0, 1.0)
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

        // Marginal firing counts and pairwise co-occurrence counts, in ONE
        // streaming pass over the active-support masks. Only the (small) active
        // set of each row contributes to the pairwise counts, so the cost is
        // `Σ_row |S_row|²` — cheap for the sparse SAE codes this consumes.
        let mut marg = vec![0.0_f64; g];
        let mut co = vec![0.0_f64; g * g]; // symmetric; upper triangle used
        let mut total_active = 0.0_f64;
        for code in &self.codes {
            let active: Vec<usize> = code.active_mask.iter_ones().collect();
            total_active += active.len() as f64;
            for (idx, &u) in active.iter().enumerate() {
                marg[u] += 1.0;
                for &v in &active[idx + 1..] {
                    co[u * g + v] += 1.0;
                }
            }
        }

        let nn = n as f64;
        let independent_total: f64 = (0..g)
            .map(|atom| kt_bernoulli_bits(self.codes.iter().map(|code| code.active_mask.get(atom))))
            .sum();

        // Maximum-weight (mutual-information) spanning tree by Prim over the dense
        // pairwise-MI graph. `best_mi[v]` is the largest MI connecting an
        // out-of-tree node `v` to the current tree; grow the tree by the
        // out-of-tree node of largest such MI, accumulating its edge weight.
        let mi = |u: usize, v: usize| -> f64 {
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            mutual_information_bits(nn, marg[a], marg[b], co[a * g + b])
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
        for _ in 1..g {
            // Pick the out-of-tree node joined to the tree by the strongest edge.
            let mut pick = usize::MAX;
            let mut pick_w = f64::NEG_INFINITY;
            for v in 0..g {
                if !in_tree[v] && best_mi[v] > pick_w {
                    pick_w = best_mi[v];
                    pick = v;
                }
            }
            if pick == usize::MAX {
                break;
            }
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

        let tree_structure_bits = if g <= 2 {
            0.0
        } else {
            (g as f64 - 2.0) * (g as f64).log2()
        };
        let mut tree_total = tree_structure_bits
            + kt_bernoulli_bits(self.codes.iter().map(|code| code.active_mask.get(0)));
        for child in 1..g {
            tree_total += kt_conditional_bernoulli_bits(self.codes.iter().map(|code| {
                (
                    code.active_mask.get(parent[child]),
                    code.active_mask.get(child),
                )
            }));
        }

        let mean_support = total_active / nn;
        let cardinality_bits = (g as f64 + 1.0).log2();
        let combinatorial_bits = cardinality_bits
            + self
                .codes
                .iter()
                .map(|code| log2_binom(g as i64, code.active_mask.count_ones() as i64))
                .sum::<f64>()
                / nn;

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

/// Sequential Krichevsky–Trofimov code for a binary sequence. The Jeffreys
/// half-counts make the code universal and give every one-sample outcome one
/// bit instead of the zero-bit plug-in pathology.
fn kt_bernoulli_bits(values: impl IntoIterator<Item = bool>) -> f64 {
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

/// Conditional KT code with a separate Bernoulli predictor in each binary
/// parent context.
fn kt_conditional_bernoulli_bits(values: impl IntoIterator<Item = (bool, bool)>) -> f64 {
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

/// Pairwise mutual information `I(x_u; x_v)` in bits of two binary indicators,
/// from the `2×2` empirical joint implied by the counts `n` (rows), `n_u`, `n_v`
/// (marginals), and `n_uv` (joint). Non-negative up to floating-point noise; the
/// `0 · log 0` cells are dropped.
fn mutual_information_bits(n: f64, n_u: f64, n_v: f64, n_uv: f64) -> f64 {
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
/// [`SparseAtomCodes::coactivation`]). All probabilities are empirical
/// popcount ratios over the active-support masks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoactivationStats {
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
    /// jointly-active rows (see [`SparseAtomCodes::weight_codependence`]) — the
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

    /// #976 — the combined FUSION evidence: `dependence · |weight_correlation|`.
    /// A fusion proposal needs BOTH halves — the atoms must co-activate (support
    /// dependence) AND their amplitudes must be dependent on the joint support
    /// (the interaction evidence that a single curved family was shattered). Two
    /// independent atoms that happen to co-fire score near 0 on the second factor
    /// and so are NOT proposed for fusion even at high support overlap; a genuine
    /// shattered pair scores high on both. This is the scalar the canonical-order
    /// fusion ranking ("fusions by code dependence descending") should sort on,
    /// and the threshold the e-process acceptance gate guards.
    pub fn fusion_evidence(&self) -> f64 {
        self.dependence() * self.weight_correlation.abs()
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

    #[test]
    fn sparse_code_assign() {
        let mut c = SparseAtomCode::empty(8);
        c.assign(2, 0.7);
        c.assign(5, 0.3);
        assert_eq!(c.n_active(), 2);
        assert!((c.active_weight_sum() - 1.0).abs() < 1e-12);
        c.deactivate(2);
        assert_eq!(c.n_active(), 1);
        assert_eq!(c.weights[2], 0.0);
    }

    #[test]
    fn codes_matrix_roundtrip() {
        let mut codes = SparseAtomCodes::empty(3, 4);
        codes.row_mut(0).assign(1, 0.5);
        codes.row_mut(2).assign(3, 0.9);
        let m = codes.weights_matrix();
        assert_eq!(m[[0, 1]], 0.5);
        assert_eq!(m[[2, 3]], 0.9);
        assert_eq!(m[[1, 0]], 0.0);
    }

    /// Co-activation triggers separate the three planted regimes: independent
    /// atoms (low dependence), a shattered duplicate pair (dependence ≈ 1,
    /// symmetric), and an absorption hierarchy (high asymmetry, parent
    /// conditional ≈ 1).
    #[test]
    fn coactivation_separates_independent_shattered_and_absorbed() {
        let n = 100usize;
        let mut codes = SparseAtomCodes::empty(n, 4);
        for row in 0..n {
            // Atom 0: active on even rows; atom 1: active on rows ≡ 0 (mod 5)
            // — independent-ish supports (joint = rows ≡ 0 mod 10).
            if row % 2 == 0 {
                codes.row_mut(row).assign(0, 1.0);
            }
            if row % 5 == 0 {
                codes.row_mut(row).assign(1, 1.0);
            }
            // Atoms 2 and 3: a nested pair — 3 (child) active on rows ≡ 0
            // (mod 4), 2 (parent) active whenever 3 is plus half of the rest.
            if row % 4 == 0 || row % 2 == 1 {
                codes.row_mut(row).assign(2, 1.0);
            }
            if row % 4 == 0 {
                codes.row_mut(row).assign(3, 1.0);
            }
        }

        // Independent pair: P(0|1) = 0.5 (even rows among multiples of 5),
        // P(1|0) = 10/50 = 0.2 → low symmetric dependence, lift = 1.
        let indep = codes.coactivation(0, 1);
        assert_eq!(indep.n_joint, 10);
        assert!((indep.p_a_given_b - 0.5).abs() < 1e-12);
        assert!((indep.p_b_given_a - 0.2).abs() < 1e-12);
        assert!((indep.lift - 1.0).abs() < 1e-12);
        assert!(indep.dependence() < 0.25);

        // Nested (absorption-suspect) pair: P(parent|child) = 1, converse
        // small → near-maximal asymmetry.
        let nested = codes.coactivation(2, 3);
        assert!((nested.p_a_given_b - 1.0).abs() < 1e-12);
        assert!(nested.p_b_given_a < 0.5);
        assert!(nested.absorption_asymmetry() > 0.6);

        // Shattered pair: identical supports → dependence = 1, asymmetry = 0.
        let mut dup = SparseAtomCodes::empty(n, 2);
        for row in (0..n).step_by(3) {
            dup.row_mut(row).assign(0, 1.0);
            dup.row_mut(row).assign(1, 1.0);
        }
        let shat = dup.coactivation(0, 1);
        assert!((shat.dependence() - 1.0).abs() < 1e-12);
        assert!(shat.absorption_asymmetry() < 1e-12);

        // Empty marginals are total, not NaN.
        let empty = SparseAtomCodes::empty(4, 2).coactivation(0, 1);
        assert_eq!(empty.dependence(), 0.0);
        assert_eq!(empty.lift, 0.0);
    }

    /// #976 — the fusion criterion's discriminating power. Support co-activation
    /// alone cannot tell a SHATTERED curved family (one manifold smeared across
    /// two near-duplicate atoms) from two GENUINELY INDEPENDENT atoms that fire
    /// on the same input class: BOTH can have identical supports (dependence = 1).
    /// The amplitude half — [`SparseAtomCodes::weight_codependence`] — separates
    /// them: a shattered pair trades activation weight smoothly as it moves along
    /// the family, so the pair's weights are strongly correlated on the joint
    /// support, while independent co-active atoms have uncorrelated weights.
    /// `fusion_evidence` (dependence · |weight_correlation|) must therefore fire
    /// on the planted shatter and stay near zero for the independent pair.
    #[test]
    fn fusion_criterion_distinguishes_shattered_from_independent_coactive() {
        let n = 120usize;

        // Planted SHATTER: a 1-D family parametrised by t ∈ [0,1] re-encoded as
        // two near-duplicate atoms whose amplitudes are a smooth partition of
        // unity — atom 0 carries `t`, atom 1 carries `1 − t`. Moving along the
        // family trades weight between them, so on the (identical) joint support
        // their weights are PERFECTLY anti-correlated (|ρ| = 1).
        let mut shattered = SparseAtomCodes::empty(n, 2);
        for row in 0..n {
            let t = (row as f64 + 0.5) / n as f64;
            shattered.row_mut(row).assign(0, t);
            shattered.row_mut(row).assign(1, 1.0 - t);
        }
        let shat = shattered.coactivation(0, 1);
        assert!(
            (shat.dependence() - 1.0).abs() < 1e-12,
            "shattered pair shares support: dependence={}",
            shat.dependence()
        );
        assert!(
            shat.weight_correlation < -0.99,
            "shattered family's partition-of-unity weights are anti-correlated on \
             the joint support: weight_correlation={}",
            shat.weight_correlation
        );
        assert!(
            shat.fusion_evidence() > 0.95,
            "fusion evidence must FIRE on a planted shatter: {}",
            shat.fusion_evidence()
        );

        // INDEPENDENT co-active pair: same identical support (dependence = 1),
        // but the two atoms' weights are drawn independently — a deterministic,
        // mutually-uncorrelated pair of sequences (one from a low-frequency
        // sinusoid, one from a coprime-frequency sinusoid, phase-offset) so the
        // joint-support weight correlation is ≈ 0.
        let mut independent = SparseAtomCodes::empty(n, 2);
        for row in 0..n {
            let x = row as f64;
            let wa = 0.5 + 0.4 * (2.0 * std::f64::consts::PI * x / 7.0).sin();
            let wb = 0.5 + 0.4 * (2.0 * std::f64::consts::PI * x / 11.0 + 1.3).cos();
            independent.row_mut(row).assign(0, wa);
            independent.row_mut(row).assign(1, wb);
        }
        let indep = independent.coactivation(0, 1);
        assert!(
            (indep.dependence() - 1.0).abs() < 1e-12,
            "independent pair was constructed with identical support: dependence={}",
            indep.dependence()
        );
        assert!(
            indep.weight_correlation.abs() < 0.3,
            "independent co-active atoms have ~uncorrelated weights: \
             weight_correlation={}",
            indep.weight_correlation
        );

        // The criterion SEPARATES them despite identical support overlap.
        assert!(
            shat.fusion_evidence() > 3.0 * indep.fusion_evidence().max(1e-6),
            "fusion evidence must rank the shattered pair far above the independent \
             pair: shattered={}, independent={}",
            shat.fusion_evidence(),
            indep.fusion_evidence()
        );

        // Degenerate joint supports give a defined (zero) amplitude reading.
        let mut tiny = SparseAtomCodes::empty(3, 2);
        tiny.row_mut(0).assign(0, 1.0);
        tiny.row_mut(0).assign(1, 1.0);
        assert_eq!(
            tiny.weight_codependence(0, 1),
            0.0,
            "a single jointly-active row carries no amplitude correlation"
        );
    }
}
