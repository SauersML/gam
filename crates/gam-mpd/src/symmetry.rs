//! Symmetries a network's weights respect, and the bases representation theory forces from
//! them (#2951).
//!
//! # The principle
//!
//! If every operator `A_i` of a family commutes with a group action `ρ(g)`, Schur's lemma forces
//! the family to be block-diagonal in the isotypic decomposition of `ρ`, and within an isotypic
//! block of multiplicity `m` every `A_i` acts as `X_i ⊗ I_m` on its irreducible copies. The block
//! structure is a proven consequence of the symmetry, not a fit. On `Z_p` the isotypic components
//! of the regular action are the Fourier planes, which is why a network that learned modular
//! addition carries them. This module finds the symmetry from the weights and computes the forced
//! basis.
//!
//! # Groups and their forced bases (exact, weight-free)
//!
//! A [`PermutationGroup`] is given by generators on `n` points. Its orbits and its orbitals (the
//! orbits on ordered pairs) come from union–find over the generators, and its order and
//! membership from a deterministic Schreier–Sims chain. The orbital matrices `O_k` are an exact
//! integer basis of the centralizer algebra `C = End_G(ℝⁿ)`.
//!
//! [`isotypic_basis`] decomposes `ℝⁿ` under `G` with nothing but `C`:
//!
//! 1. A seeded symmetric element `H = Σ_k c_k (O_k + O_kᵀ)` of `C` (splitmix64 coefficients, so
//!    the result is reproducible) commutes with `G`, so each eigenspace is `G`-invariant. For a
//!    generic `H` each eigenspace is irreducible: in an isotypic block `W ⊗ ℝ^m`, `C` acts as
//!    `M_m(D)` on the multiplicity space, `D ∈ {ℝ, ℂ, ℍ}`, and a generic symmetric element there
//!    has eigenspaces `W ⊗ v`.
//! 2. Eigenvalues are grouped by the band `β = ‖ΔH‖₂ + n ε ‖H‖₂` (formation plus a backward-stable
//!    eigensolver, Weyl): computed eigenvalues further apart than `2β` are provably distinct. Each
//!    piece `V_a` carries the Davis–Kahan bar `η_a = β/(gap_a − β)`.
//! 3. A piece is certified irreducible when its endomorphism algebra, the span of the compressions
//!    `V_aᵀ O_k V_a` (which is all of `End_G(V_a)`, since an equivariant map on an invariant
//!    subspace extends by zero on its invariant complement), has dimension `e ∈ {1, 2, 4}` and is
//!    a normed algebra: `Y_iᵀY_j + Y_jᵀY_i = (2⟨Y_i, Y_j⟩/d) I` for an orthonormal basis `Y`.
//!    That separates `ℝ, ℂ, ℍ` from `M_2(ℝ)` (a repeated real irrep) and `ℝ ⊕ ℝ` (two irreps
//!    merged by an eigenvalue coincidence), which carry non-scalar `YᵀY`. A failure means the
//!    draw put two eigenvalues within the band; the next seeded element is tried.
//! 4. Two pieces are isomorphic iff some `V_bᵀ O_k V_a` is nonzero beyond its perturbation bound;
//!    isomorphic pieces form one isotypic block, whose characters on the generators,
//!    `χ(g) = tr(V_aᵀ P_g V_a)`, label the irrep (on `Z_p`, `2 cos(2πk/p)` names the frequency).
//! 5. Only the block projector is forced; the split of a multiplicity-`m` block into copies is a
//!    gauge. Each block's columns are the column-pivoted Gram–Schmidt of its projector, a function
//!    of the projector alone, and blocks are ordered by irrep dimension, multiplicity, characters
//!    and first pivot.
//!
//! # Discovering a permutation symmetry from weights
//!
//! A permutation `σ` of a token domain is a symmetry of a family of token-space operators
//! `A_i` (`n × n`: Grams `X Xᵀ` of token tables, bilinear forms such as `E W_QK Eᵀ`) when
//! `P_σ A_i P_σᵀ = A_i`. Its defect is `d(σ) = max_i ‖A_i[σ, σ] − A_i‖_F / ‖A_i‖_F`, and it is exact
//! when `d(σ)` is inside the family's formation band (a Gram `X Xᵀ` rounds each entry by at most
//! `γ_d (|X||X|ᵀ)_{xy}`, so an exactly invariant Gram is computed invariant up to twice that).
//!
//! [`discover_permutations`] searches by base-point levels, as a Schreier–Sims chain is built:
//! at level `ℓ` the earlier base points are held fixed and the level's base point `b` (the free
//! point whose permutation-invariant profile has the nearest twin) is sent to every free target
//! `t`. Each anchored correspondence is completed by linear assignment on the profile mismatch and
//! refined by iterated assignment on the full linearised quadratic mismatch (the operators'
//! rows and columns under the current `σ`), keeping the best defect. The level's candidates are
//! sorted by defect, identity first. A cut `k` is accepted when
//!
//! * it is gap-certified: `2 δ < γ`, `δ` the largest defect kept and `γ` the smallest dropped. By
//!   the triangle inequality `d(στ) ≤ d(σ) + d(τ)`, a product of two kept elements is never
//!   confused with a dropped candidate; keeping every candidate requires every one exact;
//! * and closed: the orbit of `b` under the kept elements' group is reached by group elements
//!   (Schreier transversal words) whose measured defect is below `γ`, so assignment failures on
//!   targets inside the orbit are replaced by products rather than breaking closure.
//!
//! The largest accepted cut gives the level's generators (those enlarging the orbit); a level with
//! no accepted cut beyond the identity ends the search. A gap certificate is a statement about the
//! candidates, not about exactness: on a small generic table the best anchored match can stand
//! alone (a Gaussian `15 × 10` table can give a lone involution at relative defect `0.37`
//! against `0.76` for the rest). Such a candidate is outside the exactness band, so
//! [`PermutationDiscovery::exact`] is false, and whether it pays is decided on the function
//! ([`certify_on_function`], in bits), never here. Nothing is declared: the candidates, the
//! cut and the stopping are read from the defects. Completeness is not claimed. The stabilizer is
//! probed only at the chosen base point, and a symmetry the assignment never proposes is not found.
//!
//! # Certifying on the network function
//!
//! A weight-level symmetry is a candidate; the network function decides. For a family of token
//! inputs and their logits, [`certify_on_function`] maps each input through `σ` on the acted
//! slots, finds the output permutation `ρ` by linear assignment on row-centred logits (not
//! declared), and bounds every row's `KL(softmax ℓ(σx)∘ρ ‖ softmax ℓ(x))` over the logit boxes with
//! [`kl_over_logit_boxes`]. A row is within band when its divergence does not exceed its numerical
//! error. The symmetry is exact when every row is; otherwise its defect is the summed divergence
//! in bits, which is what predicting each `σx` row from its `x` row costs.
//!
//! # The commutant of an operator family
//!
//! For square operators on `ℝⁿ` held by factors `A_i = L_i R_iᵀ` (normalised to unit Frobenius
//! norm), [`operator_commutant`] computes `C = {T : T A_i = A_i T}` for the `*`-closed family
//! `{A_i, A_iᵀ}` from the algebra side (Murota, Kanno, Kojima and Kojima, *Japan J. Indust.
//! Appl. Math.* 27, 2010), so nothing of size `n²` is formed:
//!
//! 1. A seeded symmetric element `H = Σ_i c_i (A_i + A_iᵀ)` lies in the algebra `T` the family
//!    generates, so each eigenspace is `C`-invariant and lies in one isotypic block of `T`.
//!    Eigenvalues are clustered by the band `β` as above.
//! 2. Every `A_i` is block-diagonal over the isotypic blocks, so two clusters whose coupling
//!    `(Σ_i ‖V_bᵀ A_i V_a‖²_F + ‖V_bᵀ A_iᵀ V_a‖²_F)^{1/2}` exceeds its bound (the clusters'
//!    Davis–Kahan bars and the compressions' rounding) are in one block. Within one block the
//!    family acts irreducibly on the irrep factor, so its clusters are connected: the blocks are
//!    the connected components. A coupling inside its bound is not proven zero, so the split is
//!    reported with its largest cross-block coupling.
//! 3. A block whose clusters are all one-dimensional has multiplicity one and real type (a
//!    one-dimensional `C`-invariant subspace of `W ⊗ ℝ^m` forces `m = 1`, `D = ℝ`), so its
//!    commutant is the scalars. A block of at most [`DENSE_COMMUTANT_MAX_N`] dimensions is read
//!    exactly instead, as the kernel of the positive semi-definite `L = Σ K_iᵀ K_i`,
//!    `K_i T = T A_i − A_i T`, on the compressed family, with the kernel's band and the smallest
//!    eigenvalues beyond it as the approximate commuting directions. Its irreducible pieces come
//!    from steps 1–5 above with the kernel in place of the orbitals. A large block with a
//!    repeated eigenvalue is retried with the next seeded element. `H` is generic in the span of
//!    the members, not in the whole algebra, so members whose symmetrised span has less than
//!    full rank on a large block leave a repeated zero eigenvalue there and the family is
//!    refused, not guessed.
//!
//! The commutant is the direct sum over blocks, and each `A_i` acts as one law on the `m` copies
//! of a multiplicity-`m` block. For orthogonal `T`, `T A = A T` is `Tᵀ A T = A`, so the
//! orthogonal part of `C` is the group of isometries that keeps every map and every bilinear
//! form of the family.
//!
//! # Candidate subdomains and twins
//!
//! A domain of many tokens (a vocabulary) is never searched whole. [`candidate_subdomains`]
//! reads, per token table `X` (one row per token), invariants of a point under every
//! permutation symmetry of the Gram `X Xᵀ`: its diagonal, its sum against the whole domain, and,
//! round by round, its sum `⟨X_x, Σ_{y∈C} X_y⟩` against every cell `C` of two or more points (a
//! symmetry maps each cell onto itself, since the cells are defined by invariants). Points whose
//! invariants are certified different (band intervals `γ` of the inner products that do not
//! chain together) are split; the refinement stops when a round splits nothing. It is `O(N d)`
//! per cell and round and never forms the `N × N` Gram. Only points in one cell can be exchanged
//! by an exact symmetry.
//!
//! Inside a cell, `x` and `y` are twins when the transposition `(x y)` is an exact symmetry:
//! `‖G[τ, τ] − G‖²_F = 4 (δᵀMδ − (δ·X_x)² − (δ·X_y)²) + 2 (G_xx − G_yy)²` with `δ = X_x − X_y` and
//! `M = XᵀX`, within the Gram's rounding band. A twin class of `k` points carries the whole
//! symmetric group `S_k`; its forced blocks are closed-form (the class mean, and the
//! `(k − 1)`-dimensional standard irrep, on which an invariant table is zero), so a
//! twin-invariant table needs one row per class. Cells with a point that is no twin are
//! reported as open: a symmetry beyond the twins could act there, and [`discover_permutations`]
//! is the search for it.
//!
//! # Code
//!
//! A basis forced by a group is described by the generators alone: [`generator_code_bits`] is the
//! prefix code of the generator count plus, per generator, each point's image as a fixed index
//! into the images still unused (`log₂ n!` bits). No basis real is sent; the decoder recomputes the
//! basis. Whether the basis pays is the engine's decoded-bits decision.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use faer::Side;
use gam_linalg::decision::projector_error_bar;
use gam_linalg::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd, fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth, symmetric_spectrum_rounding_band};
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use serde::Serialize;

use super::bounds::{BoundError, kl_over_logit_boxes};
use super::codec::{CodecError, fixed_index_len_bits, prefix_integer_len_bits};
use super::joint_operators::FactoredOperator;
use super::supports::EvidenceStatus;

/// Seeded central elements tried before a decomposition is refused as unresolved. A resource
/// budget, not a tolerance: in exact arithmetic the first draw succeeds with probability one.
const DECOMPOSITION_ATTEMPTS: u64 = 6;

/// Why a symmetry computation was refused.
#[derive(Debug)]
pub enum SymmetryError {
    /// Malformed input.
    Shape(String),
    Linalg(FaerLinalgError),
    Bound(BoundError),
    Codec(CodecError),
    /// No seeded element resolved the decomposition within its bands.
    Unresolved(String),
}

impl fmt::Display for SymmetryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(formatter, "symmetry: {reason}"),
            Self::Linalg(error) => write!(formatter, "symmetry: {error}"),
            Self::Bound(error) => write!(formatter, "symmetry: {error}"),
            Self::Codec(error) => write!(formatter, "symmetry: {error}"),
            Self::Unresolved(reason) => write!(formatter, "symmetry: unresolved: {reason}"),
        }
    }
}

impl std::error::Error for SymmetryError {}

impl From<FaerLinalgError> for SymmetryError {
    fn from(error: FaerLinalgError) -> Self {
        Self::Linalg(error)
    }
}

impl From<BoundError> for SymmetryError {
    fn from(error: BoundError) -> Self {
        Self::Bound(error)
    }
}

impl From<CodecError> for SymmetryError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

// ---------------------------------------------------------------------------------------------
// Permutations
// ---------------------------------------------------------------------------------------------

/// `σ` as images: `σ(x) = perm[x]`.
pub type Permutation = Vec<u32>;

fn identity(n: usize) -> Permutation {
    (0..n as u32).collect()
}

fn is_identity(perm: &[u32]) -> bool {
    perm.iter().enumerate().all(|(x, &y)| x as u32 == y)
}

/// `(a ∘ b)(x) = a(b(x))`.
fn compose(a: &[u32], b: &[u32]) -> Permutation {
    b.iter().map(|&y| a[y as usize]).collect()
}

fn inverse(perm: &[u32]) -> Permutation {
    let mut out = vec![0u32; perm.len()];
    for (x, &y) in perm.iter().enumerate() {
        out[y as usize] = x as u32;
    }
    out
}

fn validate_permutation(perm: &[u32], n: usize) -> Result<(), SymmetryError> {
    if perm.len() != n {
        return Err(SymmetryError::Shape(format!("a permutation of {n} points has {} images", perm.len())));
    }
    let mut seen = vec![false; n];
    for &y in perm {
        let y = y as usize;
        if y >= n || seen[y] {
            return Err(SymmetryError::Shape(format!("image {y} is out of range or repeated")));
        }
        seen[y] = true;
    }
    Ok(())
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect() }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut node = x;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (low, high) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[high] = low;
        }
    }
}

/// One level of a Schreier–Sims chain: a base point and a transversal of its orbit.
#[derive(Clone, Debug)]
struct ChainLevel {
    base: usize,
    /// `transversal[x] = u_x` with `u_x(base) = x`, for `x` in the orbit.
    transversal: Vec<Option<Permutation>>,
}

impl ChainLevel {
    fn orbit_len(&self) -> usize {
        self.transversal.iter().filter(|u| u.is_some()).count()
    }
}

/// A permutation group on `degree` points, given by generators.
#[derive(Clone, Debug)]
pub struct PermutationGroup {
    degree: usize,
    generators: Vec<Permutation>,
    chain: Vec<ChainLevel>,
    strong: Vec<Permutation>,
}

impl PartialEq for PermutationGroup {
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree && self.generators == other.generators
    }
}

impl PermutationGroup {
    /// The group generated by `generators` (identity generators are dropped).
    pub fn new(degree: usize, generators: Vec<Permutation>) -> Result<Self, SymmetryError> {
        for generator in &generators {
            validate_permutation(generator, degree)?;
        }
        let generators: Vec<Permutation> = generators.into_iter().filter(|g| !is_identity(g)).collect();
        let (chain, strong) = schreier_sims(degree, &generators);
        Ok(Self { degree, generators, chain, strong })
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn generators(&self) -> &[Permutation] {
        &self.generators
    }

    /// The group order, when it fits in `u128`.
    pub fn order(&self) -> Option<u128> {
        self.chain.iter().try_fold(1u128, |acc, level| acc.checked_mul(level.orbit_len() as u128))
    }

    /// `log₂` of the group order.
    pub fn log2_order(&self) -> f64 {
        self.chain.iter().map(|level| (level.orbit_len() as f64).log2()).sum()
    }

    /// Whether `perm` is an element, by sifting through the stabilizer chain.
    pub fn contains(&self, perm: &[u32]) -> bool {
        if validate_permutation(perm, self.degree).is_err() {
            return false;
        }
        let (residue, level) = sift(&self.chain, 0, perm.to_vec());
        level == self.chain.len() && is_identity(&residue)
    }

    /// Whether the generators commute pairwise.
    pub fn is_abelian(&self) -> bool {
        self.generators.iter().enumerate().all(|(i, a)| {
            self.generators[i + 1..].iter().all(|b| compose(a, b) == compose(b, a))
        })
    }

    /// The orbits, each sorted, in order of their smallest point.
    pub fn orbits(&self) -> Vec<Vec<usize>> {
        let mut uf = UnionFind::new(self.degree);
        for g in &self.generators {
            for (x, &y) in g.iter().enumerate() {
                uf.union(x, y as usize);
            }
        }
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for x in 0..self.degree {
            groups.entry(uf.find(x)).or_default().push(x);
        }
        let mut out: Vec<Vec<usize>> = groups.into_values().collect();
        out.sort_by_key(|orbit| orbit[0]);
        out
    }

    /// The orbitals: `index[x·n + y]` is the orbital of the ordered pair `(x, y)`, numbered in
    /// row-major order of first appearance, and the orbital count.
    pub fn orbitals(&self) -> (Vec<u32>, usize) {
        let n = self.degree;
        let mut uf = UnionFind::new(n * n);
        for g in &self.generators {
            for x in 0..n {
                for y in 0..n {
                    uf.union(x * n + y, g[x] as usize * n + g[y] as usize);
                }
            }
        }
        let mut label: HashMap<usize, u32> = HashMap::new();
        let mut index = vec![0u32; n * n];
        for (pair, slot) in index.iter_mut().enumerate() {
            let root = uf.find(pair);
            let next = label.len() as u32;
            *slot = *label.entry(root).or_insert(next);
        }
        let count = label.len();
        (index, count)
    }

    /// Every element, when the order is at most `limit`; `None` otherwise.
    pub fn elements(&self, limit: usize) -> Option<Vec<Permutation>> {
        let order = self.order()?;
        if order > limit as u128 {
            return None;
        }
        let mut elements = vec![identity(self.degree)];
        for level in self.chain.iter().rev() {
            let reps: Vec<&Permutation> = level.transversal.iter().flatten().collect();
            let mut next = Vec::with_capacity(elements.len() * reps.len());
            for u in reps {
                for h in &elements {
                    next.push(compose(u, h));
                }
            }
            elements = next;
        }
        Some(elements)
    }

    /// An element that is one odd cycle through every point of the largest orbit and fixes every
    /// other point, when the group has one: the cycle a character basis is written over. Among
    /// such elements the one sending the orbit's smallest point to the smallest image is chosen,
    /// a rule on the token indices, not a declaration. Enumerates the group, so `None` when the
    /// order exceeds `degree²`.
    pub fn regular_cycle(&self) -> Option<Permutation> {
        let orbit = self.orbits().into_iter().max_by_key(|o| (o.len(), std::cmp::Reverse(o[0])))?;
        let p = orbit.len();
        if p < 3 || p % 2 == 0 {
            return None;
        }
        let elements = self.elements(self.degree * self.degree)?;
        let start = orbit[0];
        elements
            .into_iter()
            .filter(|g| {
                (0..self.degree).all(|x| orbit.binary_search(&x).is_ok() || g[x] as usize == x)
                    && cycle_length(g, start) == p
            })
            .min_by_key(|g| g[start])
    }

    /// Cycle positions for a character basis over [`Self::regular_cycle`]: `positions[t]` is
    /// token `t`'s position on the cycle, starting at the orbit's smallest point.
    pub fn cycle_positions(&self) -> Option<Vec<Option<u32>>> {
        let cycle = self.regular_cycle()?;
        let start = (0..self.degree).find(|&x| cycle[x] as usize != x)?;
        let mut positions = vec![None; self.degree];
        let mut x = start;
        let mut position = 0u32;
        loop {
            positions[x] = Some(position);
            position += 1;
            x = cycle[x] as usize;
            if x == start {
                break;
            }
        }
        Some(positions)
    }

    /// A Schreier transversal word for each point of `base`'s orbit, from the generators.
    fn orbit_transversal(&self, base: usize) -> Vec<Option<Permutation>> {
        let mut transversal: Vec<Option<Permutation>> = vec![None; self.degree];
        transversal[base] = Some(identity(self.degree));
        let mut queue = vec![base];
        let mut head = 0;
        while head < queue.len() {
            let x = queue[head];
            head += 1;
            let ux = transversal[x].clone().unwrap_or_else(|| identity(self.degree));
            for g in &self.generators {
                let y = g[x] as usize;
                if transversal[y].is_none() {
                    transversal[y] = Some(compose(g, &ux));
                    queue.push(y);
                }
            }
        }
        transversal
    }

    /// The strong generating set of the chain.
    pub fn strong_generators(&self) -> &[Permutation] {
        &self.strong
    }
}

fn cycle_length(perm: &[u32], start: usize) -> usize {
    let mut x = perm[start] as usize;
    let mut length = 1;
    while x != start {
        x = perm[x] as usize;
        length += 1;
    }
    length
}

/// `(residue, level)`: `perm` stripped through `chain[from..]`; `level` is where the image of a
/// base point fell outside its orbit (`chain.len()` if it passed every level).
fn sift(chain: &[ChainLevel], from: usize, mut perm: Permutation) -> (Permutation, usize) {
    for (index, level) in chain.iter().enumerate().skip(from) {
        let image = perm[level.base] as usize;
        match &level.transversal[image] {
            Some(u) => perm = compose(&inverse(u), &perm),
            None => return (perm, index),
        }
    }
    (perm, chain.len())
}

fn transversal_of(n: usize, base: usize, generators: &[&Permutation]) -> Vec<Option<Permutation>> {
    let mut transversal: Vec<Option<Permutation>> = vec![None; n];
    transversal[base] = Some(identity(n));
    let mut queue = vec![base];
    let mut head = 0;
    while head < queue.len() {
        let x = queue[head];
        head += 1;
        let ux = transversal[x].clone().unwrap_or_else(|| identity(n));
        for g in generators {
            let y = g[x] as usize;
            if transversal[y].is_none() {
                transversal[y] = Some(compose(g, &ux));
                queue.push(y);
            }
        }
    }
    transversal
}

/// The deterministic Schreier–Sims algorithm (Holt, *Handbook of Computational Group Theory*,
/// §4.4.2): a base, a strong generating set, and each level's transversal.
fn schreier_sims(n: usize, generators: &[Permutation]) -> (Vec<ChainLevel>, Vec<Permutation>) {
    let mut strong: Vec<Permutation> = generators.to_vec();
    let mut bases: Vec<usize> = Vec::new();
    for g in &strong {
        if bases.iter().all(|&b| g[b] as usize == b) {
            if let Some(moved) = (0..n).find(|&x| g[x] as usize != x) {
                bases.push(moved);
            }
        }
    }
    let fixes_prefix = |g: &Permutation, bases: &[usize], level: usize| bases[..level].iter().all(|&b| g[b] as usize == b);
    let build_chain = |strong: &[Permutation], bases: &[usize]| -> Vec<ChainLevel> {
        (0..bases.len())
            .map(|level| {
                let gens: Vec<&Permutation> = strong.iter().filter(|g| fixes_prefix(g, bases, level)).collect();
                ChainLevel { base: bases[level], transversal: transversal_of(n, bases[level], &gens) }
            })
            .collect()
    };
    let mut chain = build_chain(&strong, &bases);
    let mut level = bases.len();
    while level > 0 {
        let i = level - 1;
        let gens: Vec<Permutation> = strong.iter().filter(|g| fixes_prefix(g, &bases, i)).cloned().collect();
        let mut added = None;
        'schreier: for x in 0..n {
            let Some(ux) = chain[i].transversal[x].clone() else { continue };
            for s in &gens {
                let sx = s[x] as usize;
                let Some(usx) = chain[i].transversal[sx].as_ref() else { continue };
                let schreier_generator = compose(&inverse(usx), &compose(s, &ux));
                let (residue, stop) = sift(&chain, i + 1, schreier_generator);
                if stop < chain.len() || !is_identity(&residue) {
                    added = Some((residue, stop));
                    break 'schreier;
                }
            }
        }
        match added {
            Some((residue, stop)) => {
                if stop == bases.len() {
                    if let Some(moved) = (0..n).find(|&x| residue[x] as usize != x) {
                        bases.push(moved);
                    }
                }
                strong.push(residue);
                chain = build_chain(&strong, &bases);
                level = stop + 1;
            }
            None => level -= 1,
        }
    }
    (chain, strong)
}

// ---------------------------------------------------------------------------------------------
// Isotypic decomposition from a *-algebra
// ---------------------------------------------------------------------------------------------

/// The type of a real irreducible representation: its endomorphism algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IrrepKind {
    Real,
    Complex,
    Quaternionic,
}

/// One isotypic component.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct IsotypicBlock {
    /// Real dimension of the irrep.
    pub irrep_dim: usize,
    pub multiplicity: usize,
    pub kind: IrrepKind,
    /// The block's columns of [`IsotypicBasis::basis`].
    pub columns: Range<usize>,
    /// `χ(g_j) = tr ρ(g_j)` of one irreducible copy on each generator (empty for an operator
    /// family, which has no generators).
    pub character: Vec<f64>,
    /// Davis–Kahan bar on the block projector's distance from the exact one (sum over its pieces).
    pub projector_error: f64,
}

/// The isotypic decomposition of `ℝⁿ` under an algebra.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct IsotypicBasis {
    /// `n × n` orthonormal, columns grouped by block.
    #[serde(skip)]
    pub basis: Array2<f64>,
    pub blocks: Vec<IsotypicBlock>,
    /// `β`: the eigenvalue band of the seeded central element.
    pub eigenvalue_band: f64,
    /// Smallest separation between distinct pieces, over `2β` (above 1 by construction).
    pub separation_over_band: f64,
    /// Dimension of the algebra the decomposition was read from.
    pub commutant_dim: usize,
    /// Seeded elements tried (1 when the first resolved).
    pub attempts: u64,
}

impl IsotypicBasis {
    /// The block's orthonormal columns.
    pub fn block_columns(&self, block: usize) -> ArrayView2<'_, f64> {
        self.basis.slice(s![.., self.blocks[block].columns.clone()])
    }

    /// `Σ_blocks ‖off-block part of Φᵀ A Φ‖²_F / ‖A‖²_F` for a square operator on the decomposed
    /// space: the energy Schur forces to zero for an operator in the commutant of the symmetry.
    pub fn off_block_fraction(&self, operator: ArrayView2<'_, f64>) -> Result<f64, SymmetryError> {
        let n = self.basis.nrows();
        if operator.dim() != (n, n) {
            return Err(SymmetryError::Shape(format!("operator is {:?}, the basis is {n}", operator.dim())));
        }
        let rotated = fast_atb(&self.basis, &fast_ab(&operator.to_owned(), &self.basis));
        let total: f64 = rotated.iter().map(|v| v * v).sum();
        if total == 0.0 {
            return Ok(0.0);
        }
        // Summed directly: `total − diagonal` would cancel to the rounding of `total`.
        let mut block_of = vec![0usize; n];
        for (index, block) in self.blocks.iter().enumerate() {
            for column in block.columns.clone() {
                block_of[column] = index;
            }
        }
        let mut off = 0.0;
        for ((i, j), v) in rotated.indexed_iter() {
            if block_of[i] != block_of[j] {
                off += v * v;
            }
        }
        Ok(off / total)
    }

    /// Share of a token table's (`n × d`) squared norm in each block, in block order.
    pub fn block_energy(&self, table: ArrayView2<'_, f64>) -> Result<Vec<f64>, SymmetryError> {
        if table.nrows() != self.basis.nrows() {
            return Err(SymmetryError::Shape(format!("table has {} rows, basis {}", table.nrows(), self.basis.nrows())));
        }
        let coefficients = fast_atb(&self.basis, &table.to_owned());
        let total: f64 = coefficients.iter().map(|v| v * v).sum();
        Ok(self
            .blocks
            .iter()
            .map(|b| {
                let part: f64 = coefficients.slice(s![b.columns.clone(), ..]).iter().map(|v| v * v).sum();
                if total > 0.0 { part / total } else { 0.0 }
            })
            .collect())
    }
}

/// A spanning set of a `*`-closed algebra of operators on `ℝⁿ`.
trait AlgebraSpan {
    fn n(&self) -> usize;
    fn dim(&self) -> usize;
    /// `Σ_k c_k (T_k + T_kᵀ)` and a bound on `‖computed − exact‖₂`.
    fn symmetric_element(&self, coefficients: &[f64]) -> (Array2<f64>, f64);
    /// `leftᵀ T_k right` for every `k`.
    fn compressions(&self, left: &Array2<f64>, right: &Array2<f64>) -> Vec<Array2<f64>>;
    /// Upper bounds on `‖T_k‖₂`.
    fn norm_bounds(&self) -> Vec<f64>;
}

struct OrbitalSpan {
    n: usize,
    index: Vec<u32>,
    count: usize,
    norms: Vec<f64>,
}

impl OrbitalSpan {
    fn new(group: &PermutationGroup) -> Self {
        let n = group.degree();
        let (index, count) = group.orbitals();
        let mut rows = vec![vec![0usize; n]; count];
        let mut cols = vec![vec![0usize; n]; count];
        for x in 0..n {
            for y in 0..n {
                let k = index[x * n + y] as usize;
                rows[k][x] += 1;
                cols[k][y] += 1;
            }
        }
        // A 0/1 matrix has ‖O‖₂ ≤ (max row count · max column count)^{1/2} (Schur's test).
        let norms = (0..count)
            .map(|k| {
                let r = *rows[k].iter().max().unwrap_or(&0) as f64;
                let c = *cols[k].iter().max().unwrap_or(&0) as f64;
                (r * c).sqrt()
            })
            .collect();
        Self { n, index, count, norms }
    }
}

impl AlgebraSpan for OrbitalSpan {
    fn n(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.count
    }

    fn symmetric_element(&self, coefficients: &[f64]) -> (Array2<f64>, f64) {
        let n = self.n;
        let mut h = Array2::<f64>::zeros((n, n));
        let mut largest = 0.0_f64;
        for x in 0..n {
            for y in 0..n {
                let value = coefficients[self.index[x * n + y] as usize] + coefficients[self.index[y * n + x] as usize];
                h[[x, y]] = value;
                largest = largest.max(value.abs());
            }
        }
        // One rounded addition per entry: |ΔH_xy| ≤ u |H_xy|, and ‖ΔH‖₂ ≤ n max|ΔH_xy|.
        (h, n as f64 * UNIT_ROUNDOFF * largest)
    }

    fn compressions(&self, left: &Array2<f64>, right: &Array2<f64>) -> Vec<Array2<f64>> {
        let n = self.n;
        let (db, da) = (left.ncols(), right.ncols());
        let mut out = vec![Array2::<f64>::zeros((db, da)); self.count];
        for x in 0..n {
            let lx = left.row(x);
            for y in 0..n {
                let k = self.index[x * n + y] as usize;
                let ry = right.row(y);
                let target = &mut out[k];
                for i in 0..db {
                    let li = lx[i];
                    if li != 0.0 {
                        for j in 0..da {
                            target[[i, j]] += li * ry[j];
                        }
                    }
                }
            }
        }
        out
    }

    fn norm_bounds(&self) -> Vec<f64> {
        self.norms.clone()
    }
}

struct DenseSpan {
    elements: Vec<Array2<f64>>,
}

impl AlgebraSpan for DenseSpan {
    fn n(&self) -> usize {
        self.elements.first().map_or(0, |m| m.nrows())
    }

    fn dim(&self) -> usize {
        self.elements.len()
    }

    fn symmetric_element(&self, coefficients: &[f64]) -> (Array2<f64>, f64) {
        let n = self.n();
        let mut h = Array2::<f64>::zeros((n, n));
        let mut absolute = Array2::<f64>::zeros((n, n));
        for (c, t) in coefficients.iter().zip(&self.elements) {
            let sym = t + &t.t();
            h.scaled_add(*c, &sym);
            absolute.scaled_add(c.abs(), &sym.mapv(f64::abs));
        }
        let largest = absolute.iter().copied().fold(0.0_f64, f64::max);
        let growth = accumulation_growth(2 * self.elements.len() + 1);
        (h, n as f64 * growth * largest)
    }

    fn compressions(&self, left: &Array2<f64>, right: &Array2<f64>) -> Vec<Array2<f64>> {
        self.elements.iter().map(|t| fast_atb(left, &fast_ab(t, right))).collect()
    }

    fn norm_bounds(&self) -> Vec<f64> {
        self.elements.iter().map(|t| t.iter().map(|v| v * v).sum::<f64>().sqrt()).collect()
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform on `[−1, 1)`, from the top 53 bits.
fn seeded_coefficients(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            let bits = splitmix64(&mut state) >> 11;
            2.0 * (bits as f64 / (1u64 << 53) as f64) - 1.0
        })
        .collect()
}

/// An irreducible piece: an eigenspace of the central element.
struct Piece {
    vectors: Array2<f64>,
    /// Davis–Kahan bar on its projector.
    eta: f64,
    kind: IrrepKind,
}

fn frobenius(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Certifies one piece irreducible from its endomorphism algebra, or says why not.
fn classify_piece(span: &dyn AlgebraSpan, vectors: &Array2<f64>, eta: f64) -> Result<Option<IrrepKind>, SymmetryError> {
    let d = vectors.ncols();
    let compressions = span.compressions(vectors, vectors);
    let norms = span.norm_bounds();
    let mut stacked = Array2::<f64>::zeros((compressions.len(), d * d));
    for (k, y) in compressions.iter().enumerate() {
        for (slot, value) in stacked.row_mut(k).iter_mut().zip(y.iter()) {
            *slot = *value;
        }
    }
    // Each compression moves by at most 2η‖T_k‖₂ (both sides' subspace error) plus its rounding.
    let perturbation = norms
        .iter()
        .map(|&norm| {
            let rounding = accumulation_growth(2 * span.n()) * norm * d as f64;
            let bound = 2.0 * eta * norm + rounding;
            bound * bound
        })
        .sum::<f64>()
        .sqrt();
    let (_, sigma, vt) = stacked.svd(false, true)?;
    let vt = vt.ok_or_else(|| SymmetryError::Unresolved("SVD returned no right vectors".into()))?;
    let e = sigma.iter().filter(|&&v| v > perturbation).count();
    let kind = match e {
        1 => IrrepKind::Real,
        2 if d % 2 == 0 => IrrepKind::Complex,
        4 if d % 4 == 0 => IrrepKind::Quaternionic,
        _ => return Ok(None),
    };
    if e == 1 {
        return Ok(Some(kind));
    }
    // Wedin: the top-e right singular subspace moves by ω = Δ/(σ_e − Δ); an orthonormal element
    // of a normed algebra has ‖Y‖₂ = d^{-1/2}.
    let sigma_e = sigma[e - 1];
    if sigma_e <= perturbation {
        return Ok(None);
    }
    let omega = perturbation / (sigma_e - perturbation);
    let ys: Vec<Array2<f64>> = (0..e)
        .map(|i| Array2::from_shape_vec((d, d), vt.row(i).to_vec()).expect("d² entries"))
        .collect();
    let allowed = 4.0 * std::f64::consts::SQRT_2 * omega / (d as f64).sqrt()
        + 4.0 * omega * omega
        + accumulation_growth(d) * 2.0 / d as f64;
    for i in 0..e {
        for j in i..e {
            let mut m = fast_atb(&ys[i], &ys[j]) + fast_atb(&ys[j], &ys[i]);
            let target = if i == j { 2.0 / d as f64 } else { 0.0 };
            for t in 0..d {
                m[[t, t]] -= target;
            }
            if frobenius(&m) > allowed {
                return Ok(None);
            }
        }
    }
    Ok(Some(kind))
}

/// Canonical orthonormal columns of a projector: column-pivoted Gram–Schmidt, ties to the lowest
/// index, `rank` steps. Returns the columns and the first pivot.
fn pivoted_columns(projector: &Array2<f64>, rank: usize) -> (Array2<f64>, usize) {
    let n = projector.nrows();
    let mut residual = projector.clone();
    let mut out = Array2::<f64>::zeros((n, rank));
    let mut first = 0;
    for step in 0..rank {
        let mut best = (0usize, -1.0_f64);
        for j in 0..n {
            let norm = residual.column(j).iter().map(|v| v * v).sum::<f64>();
            if norm > best.1 {
                best = (j, norm);
            }
        }
        if step == 0 {
            first = best.0;
        }
        let norm = best.1.sqrt();
        let q: Array1<f64> = residual.column(best.0).mapv(|v| v / norm);
        let projection = q.dot(&residual);
        for (i, qi) in q.iter().enumerate() {
            for j in 0..n {
                residual[[i, j]] -= qi * projection[j];
            }
        }
        out.column_mut(step).assign(&q);
    }
    (out, first)
}

/// One isotypic component before ordering.
struct Draft {
    irrep_dim: usize,
    multiplicity: usize,
    kind: IrrepKind,
    character: Vec<f64>,
    projector_error: f64,
    columns: Array2<f64>,
    first_pivot: usize,
}

/// Orders components by irrep dimension, multiplicity, characters (descending) and first pivot,
/// and lays their columns side by side.
fn assemble(n: usize, mut drafts: Vec<Draft>) -> (Array2<f64>, Vec<IsotypicBlock>) {
    drafts.sort_by(|a, b| {
        a.irrep_dim
            .cmp(&b.irrep_dim)
            .then(a.multiplicity.cmp(&b.multiplicity))
            .then_with(|| {
                b.character
                    .iter()
                    .zip(&a.character)
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then(a.first_pivot.cmp(&b.first_pivot))
    });
    let mut basis = Array2::<f64>::zeros((n, n));
    let mut blocks = Vec::with_capacity(drafts.len());
    let mut column = 0;
    for draft in drafts {
        let width = draft.columns.ncols();
        basis.slice_mut(s![.., column..column + width]).assign(&draft.columns);
        blocks.push(IsotypicBlock {
            irrep_dim: draft.irrep_dim,
            multiplicity: draft.multiplicity,
            kind: draft.kind,
            columns: column..column + width,
            character: draft.character,
            projector_error: draft.projector_error,
        });
        column += width;
    }
    (basis, blocks)
}

/// The isotypic decomposition read from a spanning set of a `*`-closed algebra (the module's
/// steps 1–5). `generators` (images) give the characters; empty for an operator family.
fn decompose(span: &dyn AlgebraSpan, generators: &[Permutation]) -> Result<IsotypicBasis, SymmetryError> {
    let n = span.n();
    if n == 0 {
        return Err(SymmetryError::Shape("empty space".into()));
    }
    let mut failure = String::new();
    for attempt in 0..DECOMPOSITION_ATTEMPTS {
        match decompose_once(span, generators, attempt)? {
            Ok(mut basis) => {
                basis.attempts = attempt + 1;
                return Ok(basis);
            }
            Err(reason) => failure = reason,
        }
    }
    Err(SymmetryError::Unresolved(format!("{DECOMPOSITION_ATTEMPTS} seeded elements: {failure}")))
}

fn decompose_once(
    span: &dyn AlgebraSpan,
    generators: &[Permutation],
    attempt: u64,
) -> Result<Result<IsotypicBasis, String>, SymmetryError> {
    let n = span.n();
    let coefficients = seeded_coefficients(span.dim(), 0x5EED_0000_0000_0000 ^ attempt);
    let (h, formation) = span.symmetric_element(&coefficients);
    let (values, vectors) = h.eigh(Side::Lower)?;
    let values: Vec<f64> = values.to_vec();
    let beta = formation + symmetric_spectrum_rounding_band(&values);
    // Clusters: consecutive (ascending) eigenvalues within 2β.
    let mut clusters: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || values[i] - values[i - 1] > 2.0 * beta {
            clusters.push(start..i);
            start = i;
        }
    }
    let mut separation = f64::INFINITY;
    let mut pieces: Vec<Piece> = Vec::with_capacity(clusters.len());
    for (c, range) in clusters.iter().enumerate() {
        let below = if c > 0 { values[range.start] - values[range.start - 1] } else { f64::INFINITY };
        let above = if c + 1 < clusters.len() { values[range.end] - values[range.end - 1] } else { f64::INFINITY };
        let gap = below.min(above);
        separation = separation.min(gap / (2.0 * beta));
        let eta = if gap.is_finite() { projector_error_bar(gap, beta) } else { 0.0 };
        let block = vectors.slice(s![.., range.clone()]).to_owned();
        match classify_piece(span, &block, eta)? {
            Some(kind) => pieces.push(Piece { vectors: block, eta, kind }),
            None => {
                return Ok(Err(format!(
                    "attempt {attempt}: eigenvalue cluster {range:?} is not certified irreducible"
                )));
            }
        }
    }
    // Isomorphism classes: a nonzero intertwiner compression beyond its perturbation.
    let norms = span.norm_bounds();
    let mut uf = UnionFind::new(pieces.len());
    for a in 0..pieces.len() {
        for b in (a + 1)..pieces.len() {
            if pieces[a].vectors.ncols() != pieces[b].vectors.ncols() || pieces[a].kind != pieces[b].kind {
                continue;
            }
            let hom = span.compressions(&pieces[b].vectors, &pieces[a].vectors);
            let size = hom.iter().map(|m| m.iter().map(|v| v * v).sum::<f64>()).sum::<f64>().sqrt();
            let d = pieces[a].vectors.ncols() as f64;
            let bound = norms
                .iter()
                .map(|&norm| {
                    let term = (pieces[a].eta + pieces[b].eta) * norm + accumulation_growth(2 * n) * norm * d;
                    term * term
                })
                .sum::<f64>()
                .sqrt();
            if size > bound {
                uf.union(a, b);
            }
        }
    }
    let mut classes: Vec<Vec<usize>> = Vec::new();
    let mut class_of: HashMap<usize, usize> = HashMap::new();
    for a in 0..pieces.len() {
        let root = uf.find(a);
        let index = *class_of.entry(root).or_insert_with(|| {
            classes.push(Vec::new());
            classes.len() - 1
        });
        classes[index].push(a);
    }
    let mut drafts: Vec<Draft> = Vec::with_capacity(classes.len());
    for class in &classes {
        let first = &pieces[class[0]];
        let d = first.vectors.ncols();
        let character: Vec<f64> = generators
            .iter()
            .map(|g| (0..n).map(|x| first.vectors.row(g[x] as usize).dot(&first.vectors.row(x))).sum())
            .collect();
        let mut projector = Array2::<f64>::zeros((n, n));
        let mut projector_error = 0.0;
        for &a in class {
            projector += &fast_abt(&pieces[a].vectors, &pieces[a].vectors);
            projector_error += pieces[a].eta;
        }
        let rank = d * class.len();
        let (columns, first_pivot) = pivoted_columns(&projector, rank);
        drafts.push(Draft {
            irrep_dim: d,
            multiplicity: class.len(),
            kind: first.kind,
            character,
            projector_error,
            columns,
            first_pivot,
        });
    }
    let (basis, blocks) = assemble(n, drafts);
    Ok(Ok(IsotypicBasis {
        basis,
        blocks,
        eigenvalue_band: beta,
        separation_over_band: separation,
        commutant_dim: span.dim(),
        attempts: 1,
    }))
}

/// The isotypic decomposition of the permutation representation of `group` on `ℝⁿ`: exact up to
/// the reported bands, and a function of the generators alone.
pub fn isotypic_basis(group: &PermutationGroup) -> Result<IsotypicBasis, SymmetryError> {
    let span = OrbitalSpan::new(group);
    decompose(&span, group.generators())
}

/// Bits to send a group's generators: the count in the prefix code, then per generator each
/// point's image as a fixed index into the images still unused (`log₂ n!` bits each).
pub fn generator_code_bits(group: &PermutationGroup) -> Result<u64, SymmetryError> {
    let n = group.degree();
    let mut per_generator = 0u64;
    for remaining in 2..=n {
        per_generator += fixed_index_len_bits(remaining)? as u64;
    }
    let count = group.generators().len() as u64;
    Ok(prefix_integer_len_bits(count + 1)? + count * per_generator)
}

// ---------------------------------------------------------------------------------------------
// Linear assignment
// ---------------------------------------------------------------------------------------------

/// Minimum-cost perfect matching of a square cost matrix (shortest augmenting paths with
/// potentials, `O(n³)`): `rows_to_cols[i]` is row `i`'s column. Ties go to the lowest column.
pub fn linear_assignment(cost: ArrayView2<'_, f64>) -> Result<Vec<usize>, SymmetryError> {
    let n = cost.nrows();
    if cost.ncols() != n {
        return Err(SymmetryError::Shape(format!("assignment cost is {:?}, not square", cost.dim())));
    }
    if cost.iter().any(|v| !v.is_finite()) {
        return Err(SymmetryError::Shape("assignment cost has a non-finite entry".into()));
    }
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let current = cost[[i0 - 1, j - 1]] - u[i0] - v[j];
                    if current < minv[j] {
                        minv[j] = current;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![0usize; n];
    for j in 1..=n {
        if p[j] != 0 {
            out[p[j] - 1] = j - 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Discovery on token-space operators
// ---------------------------------------------------------------------------------------------

/// Square operators on one token domain, each with the relative band inside which an exactly
/// invariant operator's computed defect can fall.
#[derive(Clone, Debug)]
pub struct TokenOperators {
    pub operators: Vec<Array2<f64>>,
    /// `‖computed − exact‖_F` bound per operator, relative to its Frobenius norm, doubled for the
    /// two operators a defect compares.
    pub exact_bands: Vec<f64>,
}

impl TokenOperators {
    /// Operators with their relative exactness bands.
    pub fn new(operators: Vec<Array2<f64>>, exact_bands: Vec<f64>) -> Result<Self, SymmetryError> {
        let n = operators.first().map(|m| m.nrows()).ok_or_else(|| SymmetryError::Shape("no operators".into()))?;
        if operators.len() != exact_bands.len() {
            return Err(SymmetryError::Shape("one band per operator".into()));
        }
        for m in &operators {
            if m.dim() != (n, n) || m.iter().any(|v| !v.is_finite()) || frobenius(m) == 0.0 {
                return Err(SymmetryError::Shape("operators must be finite, nonzero and share one square shape".into()));
            }
        }
        Ok(Self { operators, exact_bands })
    }

    /// The Grams `X Xᵀ` of token tables (`n × d_j`, one row per token). Each entry rounds by at
    /// most `γ_d (|X||X|ᵀ)_{xy}`, so an exactly invariant Gram's computed defect is at most twice
    /// that majorant's Frobenius norm, relative to the Gram's.
    pub fn from_tables(tables: &[ArrayView2<'_, f64>]) -> Result<Self, SymmetryError> {
        let mut operators = Vec::with_capacity(tables.len());
        let mut bands = Vec::with_capacity(tables.len());
        for table in tables {
            let x = table.to_owned();
            let gram = fast_abt(&x, &x);
            let absolute = x.mapv(f64::abs);
            let majorant = fast_abt(&absolute, &absolute);
            let band = 2.0 * accumulation_growth(x.ncols()) * frobenius(&majorant) / frobenius(&gram);
            operators.push(gram);
            bands.push(band);
        }
        Self::new(operators, bands)
    }

    /// The projectors onto every leading left singular subspace of a row-centred token table
    /// (`n × d`, read only through affine maps) that its spectrum resolves, each with its rank.
    /// An affine read absorbs the rows' mean and any invertible map of the features, so the table
    /// is known only up to its column span; a projector whitens its subspace, which makes a cycle
    /// drawn on an ellipse (unequal plane gains, non-orthogonal plane axes) as symmetric as one
    /// drawn on a circle. Rank `r` is offered when `σ_r − σ_{r+1} > 2β`, `β` the centring and SVD
    /// backward error (a subspace inside a tied cluster is not defined), except full centred rank
    /// `n − 1`, whose projector is the centring one that every permutation keeps. Which rank is
    /// used is not chosen here: each is a candidate, and the caller's code length decides. The band
    /// is Wedin's, `‖P̂ − P‖₂ ≤ η = β/(σ_r − σ_{r+1} − β)`, so an exactly invariant projector's
    /// computed defect is within `2√2 η`, relative. `input_radius` bounds `‖T − T₀‖_F` for the
    /// table `T₀` the reals stand for (a weight known to its lattice's half-step), and joins `β`:
    /// a symmetry of `T₀` is then exact within the band, though rounding broke it in `T`.
    pub fn leading_subspaces(table: ArrayView2<'_, f64>, input_radius: f64) -> Result<Vec<(usize, Self)>, SymmetryError> {
        let (n, d) = table.dim();
        if n < 3 || table.iter().any(|v| !v.is_finite()) {
            return Err(SymmetryError::Shape("a table needs three finite rows".into()));
        }
        let mean = table.mean_axis(Axis(0)).ok_or_else(|| SymmetryError::Shape("empty table".into()))?;
        let centred = &table.to_owned() - &mean.insert_axis(Axis(0));
        let (u, sigma, _) = centred.svd(true, false)?;
        let u = u.ok_or_else(|| SymmetryError::Unresolved("SVD returned no left vectors".into()))?;
        let mut order: Vec<usize> = (0..sigma.len()).collect();
        order.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
        let values: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
        let top = values.first().copied().unwrap_or(0.0);
        let beta = n.max(d) as f64 * UNIT_ROUNDOFF * top + accumulation_growth(n + d) * frobenius(&centred) + input_radius;
        let mut out = Vec::new();
        for rank in 1..=values.len().min(n - 2) {
            let gap = values[rank - 1] - values.get(rank).copied().unwrap_or(0.0);
            if values[rank - 1] <= beta || gap <= 2.0 * beta {
                continue;
            }
            let frame = u.select(Axis(1), &order[..rank]);
            let eta = projector_error_bar(gap, beta);
            out.push((rank, Self::new(vec![fast_abt(&frame, &frame)], vec![2.0 * std::f64::consts::SQRT_2 * eta])?));
        }
        Ok(out)
    }

    pub fn n(&self) -> usize {
        self.operators[0].nrows()
    }

    /// `max_i ‖A_i[σ, σ] − A_i‖_F / ‖A_i‖_F`.
    pub fn defect(&self, perm: &[u32]) -> f64 {
        let n = self.n();
        self.operators
            .iter()
            .map(|a| {
                let mut diff = 0.0;
                for x in 0..n {
                    let sx = perm[x] as usize;
                    for y in 0..n {
                        let delta = a[[sx, perm[y] as usize]] - a[[x, y]];
                        diff += delta * delta;
                    }
                }
                diff.sqrt() / frobenius(a)
            })
            .fold(0.0, f64::max)
    }

    /// The largest relative exactness band.
    pub fn exact_band(&self) -> f64 {
        self.exact_bands.iter().copied().fold(0.0, f64::max)
    }

    fn weights(&self) -> Vec<f64> {
        self.operators.iter().map(|a| 1.0 / frobenius(a).powi(2)).collect()
    }

    /// Each operator's rows and columns sorted: invariants of a point under any symmetry.
    fn sorted_profiles(&self) -> Vec<(Array2<f64>, Array2<f64>)> {
        self.operators
            .iter()
            .map(|a| {
                let sort = |m: ArrayView2<'_, f64>| {
                    let mut out = m.to_owned();
                    for mut row in out.outer_iter_mut() {
                        let mut values = row.to_vec();
                        values.sort_by(f64::total_cmp);
                        row.assign(&Array1::from(values));
                    }
                    out
                };
                (sort(a.view()), sort(a.t()))
            })
            .collect()
    }

    /// Weighted mismatch of two points' invariant profiles and of their entries against the
    /// fixed points.
    fn profile_distance(&self, profiles: &[(Array2<f64>, Array2<f64>)], weights: &[f64], fixed: &[usize], x: usize, y: usize) -> f64 {
        let mut total = 0.0;
        for ((a, (rows, cols)), w) in self.operators.iter().zip(profiles).zip(weights) {
            let mut sum: f64 = rows.row(x).iter().zip(rows.row(y)).map(|(p, q)| (p - q).powi(2)).sum::<f64>()
                + cols.row(x).iter().zip(cols.row(y)).map(|(p, q)| (p - q).powi(2)).sum::<f64>();
            for &f in fixed {
                sum += (a[[x, f]] - a[[y, f]]).powi(2) + (a[[f, x]] - a[[f, y]]).powi(2);
            }
            total += w * sum;
        }
        total
    }
}

/// The anchored completion: every free point assigned by the entries it shares with the anchors
/// (and its diagonal), its defect, and the completed pair the assignment is surest of (least cost),
/// absent when at most one point was free.
fn complete(
    ops: &TokenOperators,
    weights: &[f64],
    anchors: &[(usize, usize)],
) -> Result<(Permutation, f64, Option<(usize, usize)>), SymmetryError> {
    let n = ops.n();
    let mut source_fixed = vec![false; n];
    let mut target_fixed = vec![false; n];
    let mut perm = vec![u32::MAX; n];
    for &(x, t) in anchors {
        source_fixed[x] = true;
        target_fixed[t] = true;
        perm[x] = t as u32;
    }
    let sources: Vec<usize> = (0..n).filter(|&x| !source_fixed[x]).collect();
    let targets: Vec<usize> = (0..n).filter(|&y| !target_fixed[y]).collect();
    if sources.is_empty() {
        let defect = ops.defect(&perm);
        return Ok((perm, defect, None));
    }
    let m = sources.len();
    let mut cost = Array2::<f64>::zeros((m, m));
    for (i, &x) in sources.iter().enumerate() {
        for (j, &y) in targets.iter().enumerate() {
            let mut total = 0.0;
            for (a, w) in ops.operators.iter().zip(weights) {
                let mut sum = (a[[x, x]] - a[[y, y]]).powi(2);
                for &(s, t) in anchors {
                    sum += (a[[x, s]] - a[[y, t]]).powi(2) + (a[[s, x]] - a[[t, y]]).powi(2);
                }
                total += w * sum;
            }
            cost[[i, j]] = total;
        }
    }
    let matching = linear_assignment(cost.view())?;
    for (i, &j) in matching.iter().enumerate() {
        perm[sources[i]] = targets[j] as u32;
    }
    let defect = ops.defect(&perm);
    let surest = (m > 1)
        .then(|| matching.iter().enumerate().min_by(|a, b| cost[[a.0, *a.1]].total_cmp(&cost[[b.0, *b.1]])))
        .flatten()
        .map(|(i, &j)| (sources[i], targets[j]));
    Ok((perm, defect, surest))
}

/// Completes the anchored correspondence and refines it by iterated assignment on the
/// linearised quadratic mismatch; returns the best permutation found and its defect.
///
/// One anchor can leave the completion ambiguous: under a dihedral group, `base ↦ t` is met by a
/// rotation and by a reflection, which tie entry by entry, so one assignment may mix the two. The
/// completed pair the assignment is surest of (least cost) is then pinned as one more anchor and
/// the rest reassigned, for as long as the defect falls.
fn register(ops: &TokenOperators, anchors: &[(usize, usize)]) -> Result<(Permutation, f64), SymmetryError> {
    let n = ops.n();
    let weights = ops.weights();
    let (mut best, mut best_defect, mut pinned) = complete(ops, &weights, anchors)?;
    let mut pins = anchors.to_vec();
    while let Some(pin) = pinned {
        pins.push(pin);
        let (perm, defect, next) = complete(ops, &weights, &pins)?;
        if defect >= best_defect {
            break;
        }
        (best, best_defect, pinned) = (perm, defect, next);
    }
    let mut source_fixed = vec![false; n];
    let mut target_fixed = vec![false; n];
    for &(x, t) in anchors {
        source_fixed[x] = true;
        target_fixed[t] = true;
    }
    let sources: Vec<usize> = (0..n).filter(|&x| !source_fixed[x]).collect();
    let targets: Vec<usize> = (0..n).filter(|&y| !target_fixed[y]).collect();
    let m = sources.len();
    if m == 0 {
        return Ok((best, best_defect));
    }
    let assign = |cost: &Array2<f64>, perm: &mut Permutation| -> Result<(), SymmetryError> {
        let matching = linear_assignment(cost.view())?;
        for (i, &j) in matching.iter().enumerate() {
            perm[sources[i]] = targets[j] as u32;
        }
        Ok(())
    };
    let row_norms: Vec<Array1<f64>> = ops.operators.iter().map(|a| a.map_axis(Axis(1), |r| r.dot(&r))).collect();
    let col_norms: Vec<Array1<f64>> = ops.operators.iter().map(|a| a.map_axis(Axis(0), |c| c.dot(&c))).collect();
    loop {
        let sigma: Vec<usize> = best.iter().map(|&v| v as usize).collect();
        let mut cost = Array2::<f64>::zeros((m, m));
        for (k, (a, w)) in ops.operators.iter().zip(&weights).enumerate() {
            // cross_rows[x, y] = Σ_z A[x, z] A[y, σ(z)]; cross_cols[x, y] = Σ_z A[z, x] A[σ(z), y].
            let columns_permuted = a.select(Axis(1), &sigma);
            let rows_permuted = a.select(Axis(0), &sigma);
            let cross_rows = fast_abt(a, &columns_permuted);
            let cross_cols = fast_atb(a, &rows_permuted);
            for (i, &x) in sources.iter().enumerate() {
                for (j, &y) in targets.iter().enumerate() {
                    cost[[i, j]] += w
                        * (row_norms[k][x] + row_norms[k][y] - 2.0 * cross_rows[[x, y]] + col_norms[k][x] + col_norms[k][y]
                            - 2.0 * cross_cols[[x, y]]);
                }
            }
        }
        let mut next = best.clone();
        assign(&cost, &mut next)?;
        let defect = ops.defect(&next);
        if defect < best_defect {
            best_defect = defect;
            best = next;
        } else {
            break;
        }
    }
    Ok((best, best_defect))
}

/// The gap certificate of a level's accepted cut.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LevelGap {
    /// Largest defect kept.
    pub delta: f64,
    /// Smallest defect dropped (absent when every candidate is kept).
    pub gamma: Option<f64>,
}

/// One base-point level of the discovery search.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SearchLevel {
    pub base: usize,
    /// `(target, defect)` of every candidate, ascending by defect.
    pub candidates: Vec<(usize, f64)>,
    /// The orbit of `base` under the level's accepted elements (just `base` when none).
    pub orbit: Vec<usize>,
    pub gap: Option<LevelGap>,
    /// Defect of each orbit point's transversal element.
    pub orbit_defects: Vec<f64>,
}

/// The symmetry group discovered from token-space operators.
#[derive(Clone, Debug)]
pub struct PermutationDiscovery {
    pub levels: Vec<SearchLevel>,
    pub group: PermutationGroup,
    /// Defect of each generator.
    pub generator_defects: Vec<f64>,
    /// Largest defect over the group's elements, when it was enumerated (order at most `n²`).
    pub max_element_defect: Option<f64>,
    /// The operators' exactness band.
    pub exact_band: f64,
}

impl PermutationDiscovery {
    /// Whether every generator's defect is inside the exactness band.
    pub fn exact(&self) -> bool {
        self.generator_defects.iter().all(|&d| d <= self.exact_band)
    }
}

/// The accepted cut of one level, if any beyond the identity.
struct AcceptedLevel {
    generators: Vec<Permutation>,
    orbit: Vec<usize>,
    orbit_defects: Vec<f64>,
    gap: LevelGap,
}

fn accept_level(
    ops: &TokenOperators,
    base: usize,
    candidates: &[(usize, Permutation, f64)],
) -> Result<Option<AcceptedLevel>, SymmetryError> {
    let n = ops.n();
    let exact_band = ops.exact_band();
    let count = candidates.len();
    for k in (2..=count).rev() {
        let delta = candidates[k - 1].2;
        let gamma = if k < count { Some(candidates[k].2) } else { None };
        let certified = match gamma {
            Some(gamma) => 2.0 * delta < gamma,
            None => delta <= exact_band,
        };
        if !certified {
            continue;
        }
        let kept: Vec<Permutation> = candidates[1..k].iter().map(|c| c.1.clone()).collect();
        let group = PermutationGroup::new(n, kept.clone())?;
        let transversal = group.orbit_transversal(base);
        let orbit: Vec<usize> = (0..n).filter(|&x| transversal[x].is_some()).collect();
        let orbit_defects: Vec<f64> = orbit.iter().map(|&x| transversal[x].as_ref().map_or(0.0, |u| ops.defect(u))).collect();
        let limit = gamma.unwrap_or(f64::INFINITY);
        let closed = orbit_defects.iter().all(|&d| match gamma {
            Some(_) => d < limit,
            None => d <= exact_band,
        });
        if !closed {
            continue;
        }
        // Keep the elements that enlarge the orbit, in ascending defect order.
        let mut generators: Vec<Permutation> = Vec::new();
        let mut reached = vec![false; n];
        reached[base] = true;
        let mut reached_count = 1;
        for g in &kept {
            if reached_count == orbit.len() {
                break;
            }
            let mut trial = generators.clone();
            trial.push(g.clone());
            let trial_group = PermutationGroup::new(n, trial.clone())?;
            let t = trial_group.orbit_transversal(base);
            let size = t.iter().filter(|u| u.is_some()).count();
            if size > reached_count {
                generators = trial;
                reached_count = size;
                for (x, u) in t.iter().enumerate() {
                    reached[x] = u.is_some();
                }
            }
        }
        return Ok(Some(AcceptedLevel { generators, orbit, orbit_defects, gap: LevelGap { delta, gamma } }));
    }
    Ok(None)
}

/// Discovers the permutation symmetries of a family of token-space operators (module note).
pub fn discover_permutations(ops: &TokenOperators) -> Result<PermutationDiscovery, SymmetryError> {
    let n = ops.n();
    let mut fixed: Vec<usize> = Vec::new();
    let mut generators: Vec<Permutation> = Vec::new();
    let mut levels = Vec::new();
    let profiles = ops.sorted_profiles();
    let weights = ops.weights();
    while fixed.len() + 1 < n {
        let free: Vec<usize> = (0..n).filter(|x| !fixed.contains(x)).collect();
        // The base point: the free point whose invariant profile has the nearest twin.
        let mut base = free[0];
        let mut nearest = f64::INFINITY;
        for &x in &free {
            let twin = free.iter().filter(|&&y| y != x).map(|&y| ops.profile_distance(&profiles, &weights, &fixed, x, y)).fold(f64::INFINITY, f64::min);
            if twin < nearest {
                nearest = twin;
                base = x;
            }
        }
        let mut candidates: Vec<(usize, Permutation, f64)> = Vec::with_capacity(free.len());
        for &t in &free {
            if t == base {
                candidates.push((t, identity(n), 0.0));
                continue;
            }
            let mut anchors: Vec<(usize, usize)> = fixed.iter().map(|&f| (f, f)).collect();
            anchors.push((base, t));
            let (perm, defect) = register(ops, &anchors)?;
            candidates.push((t, perm, defect));
        }
        // The identity first, then ascending defect.
        candidates.sort_by(|a, b| (a.0 != base).cmp(&(b.0 != base)).then(a.2.total_cmp(&b.2)).then(a.0.cmp(&b.0)));
        let listing: Vec<(usize, f64)> = candidates.iter().map(|c| (c.0, c.2)).collect();
        match accept_level(ops, base, &candidates)? {
            Some(accepted) => {
                levels.push(SearchLevel {
                    base,
                    candidates: listing,
                    orbit: accepted.orbit,
                    gap: Some(accepted.gap),
                    orbit_defects: accepted.orbit_defects,
                });
                generators.extend(accepted.generators);
                fixed.push(base);
            }
            None => {
                levels.push(SearchLevel { base, candidates: listing, orbit: vec![base], gap: None, orbit_defects: vec![0.0] });
                break;
            }
        }
    }
    let generator_defects = generators.iter().map(|g| ops.defect(g)).collect();
    let group = PermutationGroup::new(n, generators)?;
    let max_element_defect = group
        .elements(n * n)
        .map(|elements| elements.iter().map(|g| ops.defect(g)).fold(0.0, f64::max));
    Ok(PermutationDiscovery { levels, group, generator_defects, max_element_defect, exact_band: ops.exact_band() })
}

// ---------------------------------------------------------------------------------------------
// Certificate on the network function
// ---------------------------------------------------------------------------------------------

/// A declared family of token inputs with the network's logits on them.
#[derive(Clone, Copy, Debug)]
pub struct TokenFunction<'a> {
    /// `N × slots` token ids.
    pub tokens: ArrayView2<'a, u32>,
    /// Which slots range over the symmetry's domain.
    pub acted: &'a [bool],
    /// `N × C` logits.
    pub logits: ArrayView2<'a, f64>,
    /// `N × C` forward-error radii (absent means computed values are taken as exact).
    pub radii: Option<ArrayView2<'a, f64>>,
}

/// How far `F(σx) = ρ(σ) F(x)` holds on the family.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionCertificate {
    /// `ρ`: `ℓ(σx)[ρ(c)]` is compared with `ℓ(x)[c]`.
    pub output_permutation: Vec<u32>,
    pub rows: usize,
    /// Rows whose divergence does not exceed its numerical error.
    pub rows_within_band: usize,
    /// `Σ_rows KL` at the centres, in bits.
    pub kl_bits: f64,
    /// `Σ_rows` numerical error, in bits.
    pub kl_error_bits: f64,
    pub max_row_kl_bits: f64,
    /// Rows where `argmax ℓ(σx) = ρ(argmax ℓ(x))`.
    pub argmax_agreement: usize,
}

impl FunctionCertificate {
    /// Every row within band.
    pub fn exact(&self) -> bool {
        self.rows_within_band == self.rows
    }
}

/// Certifies a token permutation on the network function (module note).
pub fn certify_on_function(perm: &[u32], function: &TokenFunction<'_>) -> Result<FunctionCertificate, SymmetryError> {
    let (rows, slots) = function.tokens.dim();
    let classes = function.logits.ncols();
    if function.logits.nrows() != rows || function.acted.len() != slots {
        return Err(SymmetryError::Shape("tokens, acted slots and logits disagree".into()));
    }
    if let Some(radii) = function.radii {
        if radii.dim() != function.logits.dim() {
            return Err(SymmetryError::Shape("radii and logits disagree".into()));
        }
    }
    let mut row_of: HashMap<Vec<u32>, usize> = HashMap::with_capacity(rows);
    for (r, tokens) in function.tokens.outer_iter().enumerate() {
        row_of.insert(tokens.to_vec(), r);
    }
    let mut image = vec![0usize; rows];
    for (r, tokens) in function.tokens.outer_iter().enumerate() {
        let moved: Vec<u32> = tokens
            .iter()
            .zip(function.acted)
            .map(|(&t, &acted)| {
                if acted {
                    perm.get(t as usize).copied().ok_or_else(|| SymmetryError::Shape(format!("token {t} outside the domain")))
                } else {
                    Ok(t)
                }
            })
            .collect::<Result<_, _>>()?;
        image[r] = *row_of
            .get(&moved)
            .ok_or_else(|| SymmetryError::Shape(format!("the family is not closed under σ: row {r} maps outside it")))?;
    }
    let mut centred = function.logits.to_owned();
    for mut row in centred.outer_iter_mut() {
        let mean = row.mean().unwrap_or(0.0);
        row.mapv_inplace(|v| v - mean);
    }
    let moved = centred.select(Axis(0), &image);
    let cross = fast_atb(&centred, &moved);
    let norm_ref = centred.map_axis(Axis(0), |c| c.dot(&c));
    let norm_moved = moved.map_axis(Axis(0), |c| c.dot(&c));
    let mut cost = Array2::<f64>::zeros((classes, classes));
    for c in 0..classes {
        for c2 in 0..classes {
            cost[[c, c2]] = norm_ref[c] + norm_moved[c2] - 2.0 * cross[[c, c2]];
        }
    }
    let rho = linear_assignment(cost.view())?;
    let zero = Array1::<f64>::zeros(classes);
    let mut certificate = FunctionCertificate {
        output_permutation: rho.iter().map(|&c| c as u32).collect(),
        rows,
        rows_within_band: 0,
        kl_bits: 0.0,
        kl_error_bits: 0.0,
        max_row_kl_bits: 0.0,
        argmax_agreement: 0,
    };
    let argmax = |row: ndarray::ArrayView1<'_, f64>| {
        row.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc }).0
    };
    for r in 0..rows {
        let reference: Array1<f64> = (0..classes).map(|c| function.logits[[image[r], rho[c]]]).collect();
        let reference_radius: Array1<f64> = match function.radii {
            Some(radii) => (0..classes).map(|c| radii[[image[r], rho[c]]]).collect(),
            None => zero.clone(),
        };
        let perturbed = function.logits.row(r);
        let perturbed_radius = match function.radii {
            Some(radii) => radii.row(r).to_owned(),
            None => zero.clone(),
        };
        let status = kl_over_logit_boxes(reference.view(), reference_radius.view(), perturbed, perturbed_radius.view())?;
        let (value, error) = match status {
            EvidenceStatus::Exact { value, numerical_error, .. } => (value, numerical_error),
            other => return Err(SymmetryError::Unresolved(format!("row {r}: the logit-box divergence is {other:?}"))),
        };
        let bits = value / std::f64::consts::LN_2;
        certificate.kl_bits += bits;
        certificate.kl_error_bits += error / std::f64::consts::LN_2;
        certificate.max_row_kl_bits = certificate.max_row_kl_bits.max(bits);
        if value <= error {
            certificate.rows_within_band += 1;
        }
        if argmax(function.logits.row(image[r])) == rho[argmax(perturbed)] {
            certificate.argmax_agreement += 1;
        }
    }
    Ok(certificate)
}


// ---------------------------------------------------------------------------------------------
// Commutant of an operator family
// ---------------------------------------------------------------------------------------------

/// Largest block whose commutant is read as the dense kernel on its `n_B²` coordinates.
pub const DENSE_COMMUTANT_MAX_N: usize = 48;

/// A normalised family member `s L Rᵀ`, `s = 1/‖L Rᵀ‖_F`.
struct Member<'a> {
    left: ArrayView2<'a, f64>,
    right: ArrayView2<'a, f64>,
    scale: f64,
    /// `s ‖L‖_F ‖R‖_F`: a product entry's rounding reach per unit of `γ`.
    reach: f64,
}

fn family_members<'a>(family: &[&'a FactoredOperator]) -> Result<(usize, Vec<Member<'a>>), SymmetryError> {
    let n = family.first().map(|m| m.width()).ok_or_else(|| SymmetryError::Shape("no members".into()))?;
    let mut members = Vec::with_capacity(family.len());
    for operator in family {
        if operator.width() != n {
            return Err(SymmetryError::Shape(format!("a member of width {} in a family of width {n}", operator.width())));
        }
        let (left, right) = (operator.left(), operator.right());
        // ‖L Rᵀ‖²_F = Σ (LᵀL) ∘ (RᵀR).
        let squared: f64 = (fast_atb(&left, &left) * fast_atb(&right, &right)).sum();
        if !(squared > 0.0) {
            continue;
        }
        let scale = 1.0 / squared.sqrt();
        let factor = |m: ArrayView2<'_, f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
        members.push(Member { left, right, scale, reach: scale * factor(left) * factor(right) });
    }
    if members.is_empty() {
        return Err(SymmetryError::Shape("every member is zero".into()));
    }
    Ok((n, members))
}

/// How a block's commutant was read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommutantRoute {
    /// The kernel of `L` on the block's `n_B²` coordinates.
    Dense,
    /// Every eigenvalue of the algebra element is simple on the block, so its commutant is the
    /// scalars; nothing of size `n_B²` is formed.
    Simple,
}

/// One isotypic block of the family's algebra and its share of the commutant.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CommutantBlock {
    pub dim: usize,
    /// Eigenvalue clusters of the algebra element the block joins.
    pub clusters: usize,
    /// Dimension of the commutant on the block.
    pub commutant_dim: usize,
    pub route: CommutantRoute,
    /// The band on `L`'s eigenvalues (dense route).
    pub kernel_band: Option<f64>,
    /// Relative defects `(λ/2N)^{1/2}` of `L`'s smallest eigenvalues beyond the kernel,
    /// ascending (dense route): the approximate commuting directions.
    pub approximate_defects: Vec<f64>,
    /// The smallest certified coupling in a maximum spanning tree of the block's clusters: the
    /// largest level at which the block is still one connected piece (absent for one cluster).
    pub bottleneck_coupling: Option<f64>,
    /// Davis–Kahan bar on the block projector (sum over its clusters).
    pub projector_error: f64,
}

/// The commutant of a `*`-closed operator family and its isotypic blocks.
#[derive(Clone, Debug, Serialize)]
pub struct OperatorCommutant {
    pub n: usize,
    /// Nonzero members (each enters with its transpose).
    pub members: usize,
    /// Dimension of the exact commutant: the sum over blocks.
    pub exact_dim: usize,
    pub blocks: Vec<CommutantBlock>,
    /// `β`: the eigenvalue band of the seeded algebra element.
    pub eigenvalue_band: f64,
    /// Smallest separation between clusters, over `2β`.
    pub separation_over_band: f64,
    /// Largest coupling between clusters of different blocks and the bound it was held under:
    /// the split is exact within that bound, not beyond it.
    pub split_coupling: f64,
    pub split_bound: f64,
    /// Seeded elements tried.
    pub attempts: u64,
    /// An orthonormal (Frobenius) basis of the exact commutant, `n × n` each.
    #[serde(skip)]
    pub basis: Vec<Array2<f64>>,
    pub isotypic: IsotypicBasis,
}

/// The commutant `{T : T A_i = A_i T}` of the `*`-closed family `{A_i, A_iᵀ}`, each member
/// normalised to unit Frobenius norm and held by its factors (module note).
pub fn operator_commutant(family: &[&FactoredOperator], reported: usize) -> Result<OperatorCommutant, SymmetryError> {
    let (n, members) = family_members(family)?;
    let mut failure = String::new();
    for attempt in 0..DECOMPOSITION_ATTEMPTS {
        match commutant_once(n, &members, reported, attempt)? {
            Ok(mut commutant) => {
                commutant.attempts = attempt + 1;
                return Ok(commutant);
            }
            Err(reason) => failure = reason,
        }
    }
    Err(SymmetryError::Unresolved(format!("{DECOMPOSITION_ATTEMPTS} seeded elements: {failure}")))
}

fn commutant_once(
    n: usize,
    members: &[Member<'_>],
    reported: usize,
    attempt: u64,
) -> Result<Result<OperatorCommutant, String>, SymmetryError> {
    // 1. A seeded symmetric element of the family's algebra and its clusters.
    let coefficients = seeded_coefficients(members.len(), 0xA15E_0000_0000_0000 ^ attempt);
    let rank = members.iter().map(|m| m.left.ncols()).max().unwrap_or(1);
    let mut h = Array2::<f64>::zeros((n, n));
    let mut reach = 0.0;
    for (c, m) in coefficients.iter().zip(members) {
        let product = fast_abt(&m.left, &m.right);
        h.scaled_add(c * m.scale, &product);
        h.scaled_add(c * m.scale, &product.t());
        reach += 2.0 * c.abs() * m.reach;
    }
    let formation = accumulation_growth(rank + 2 * members.len() + 2) * reach;
    let (values, vectors) = h.eigh(Side::Lower)?;
    drop(h);
    let values: Vec<f64> = values.to_vec();
    let beta = formation + symmetric_spectrum_rounding_band(&values);
    let mut clusters: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || values[i] - values[i - 1] > 2.0 * beta {
            clusters.push(start..i);
            start = i;
        }
    }
    let mut separation = f64::INFINITY;
    let mut etas = Vec::with_capacity(clusters.len());
    for (c, range) in clusters.iter().enumerate() {
        let below = if c > 0 { values[range.start] - values[range.start - 1] } else { f64::INFINITY };
        let above = if c + 1 < clusters.len() { values[range.end] - values[range.end - 1] } else { f64::INFINITY };
        let gap = below.min(above);
        separation = separation.min(gap / (2.0 * beta));
        etas.push(if gap.is_finite() { projector_error_bar(gap, beta) } else { 0.0 });
    }
    // 2. Couplings: S[x, y] = Σ_i s_i² ((VᵀA_iV)[x, y]² + (VᵀA_iV)[y, x]²), the member and its
    //    transpose; clusters a, b couple by (Σ_{x∈a, y∈b} S[x, y])^{1/2}.
    let mut squares = Array2::<f64>::zeros((n, n));
    let (mut reach_sum, mut reach_squares) = (0.0, 0.0);
    for m in members {
        let compressed = fast_abt(&fast_atb(&vectors, &m.left), &fast_atb(&vectors, &m.right));
        let weight = m.scale * m.scale;
        squares.zip_mut_with(&compressed, |s, &w| *s += weight * w * w);
        reach_sum += 2.0 * m.reach;
        reach_squares += 2.0 * m.reach * m.reach;
    }
    let squares = &squares + &squares.t();
    let count = clusters.len();
    let mut coupling = Array2::<f64>::zeros((count, count));
    for (a, ra) in clusters.iter().enumerate() {
        for (b, rb) in clusters.iter().enumerate() {
            if a != b {
                coupling[[a, b]] = squares.slice(s![rb.clone(), ra.clone()]).sum().sqrt();
            }
        }
    }
    drop(squares);
    // A pair in different exact blocks has zero exact coupling; the computed one is within
    // (Σ_i (e + g f_i)²)^{1/2}, e from the subspaces' errors (‖P̃ − P‖_F ≤ (2d)^{1/2} η on each
    // side, ‖A_i‖₂ ≤ 1), g f_i from rounding both compressions, over the 2N members.
    let growth = accumulation_growth(2 * n + rank);
    let family_size = 2.0 * members.len() as f64;
    let bound = |a: usize, b: usize| {
        let (da, db) = (clusters[a].len() as f64, clusters[b].len() as f64);
        let e = (etas[a] + etas[b]) * (2.0 * da.max(db)).sqrt();
        let g = growth * (da * db).sqrt();
        (family_size * e * e + 2.0 * e * g * reach_sum + g * g * reach_squares).sqrt()
    };
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    let mut uf = UnionFind::new(count);
    for a in 0..count {
        for b in (a + 1)..count {
            if coupling[[a, b]] > bound(a, b) {
                edges.push((coupling[[a, b]], a, b));
                uf.union(a, b);
            }
        }
    }
    // A coupling inside its bound between clusters another path connects is not a split.
    let mut split_coupling = (0.0_f64, 0.0_f64);
    for a in 0..count {
        for b in (a + 1)..count {
            if uf.find(a) != uf.find(b) && coupling[[a, b]] > split_coupling.0 {
                split_coupling = (coupling[[a, b]], bound(a, b));
            }
        }
    }
    // Maximum spanning forest (Kruskal, descending): each block's weakest tree edge.
    edges.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut forest = UnionFind::new(count);
    let mut weakest = vec![f64::INFINITY; count];
    for &(value, a, b) in &edges {
        let (ra, rb) = (forest.find(a), forest.find(b));
        if ra != rb {
            let low = weakest[ra].min(weakest[rb]).min(value);
            forest.union(a, b);
            let root = forest.find(a);
            weakest[root] = low;
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    for a in 0..count {
        let root = uf.find(a);
        let index = *group_of.entry(root).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[index].push(a);
    }
    // 3. Each block's commutant.
    let mut blocks = Vec::with_capacity(groups.len());
    let mut drafts = Vec::new();
    let mut basis = Vec::new();
    for group in &groups {
        let columns: Vec<usize> = group.iter().flat_map(|&a| clusters[a].clone()).collect();
        let dim = columns.len();
        let frame = vectors.select(Axis(1), &columns);
        let eta: f64 = group.iter().map(|&a| etas[a]).sum();
        let root = forest.find(group[0]);
        let bottleneck = (group.len() > 1).then_some(weakest[root]);
        let simple = group.iter().all(|&a| clusters[a].len() == 1);
        if dim <= DENSE_COMMUTANT_MAX_N {
            let compressed: Vec<Array2<f64>> =
                members.iter().map(|m| fast_abt(&fast_atb(&frame, &m.left), &fast_atb(&frame, &m.right)) * m.scale).collect();
            let kernel = dense_kernel(&compressed, eta, reported)?;
            let isotypic = decompose(&DenseSpan { elements: kernel.basis.clone() }, &[])?;
            for (index, block) in isotypic.blocks.iter().enumerate() {
                let full = fast_ab(&frame, &isotypic.block_columns(index).to_owned());
                let (canonical, first_pivot) = pivoted_columns(&fast_abt(&full, &full), full.ncols());
                drafts.push(Draft {
                    irrep_dim: block.irrep_dim,
                    multiplicity: block.multiplicity,
                    kind: block.kind,
                    character: Vec::new(),
                    projector_error: block.projector_error + eta,
                    columns: canonical,
                    first_pivot,
                });
            }
            for t in &kernel.basis {
                basis.push(fast_abt(&fast_ab(&frame, t), &frame));
            }
            blocks.push(CommutantBlock {
                dim,
                clusters: group.len(),
                commutant_dim: kernel.basis.len(),
                route: CommutantRoute::Dense,
                kernel_band: Some(kernel.band),
                approximate_defects: kernel.approximate_defects,
                bottleneck_coupling: bottleneck,
                projector_error: eta,
            });
        } else if simple {
            let projector = fast_abt(&frame, &frame);
            let (canonical, first_pivot) = pivoted_columns(&projector, dim);
            basis.push(projector / (dim as f64).sqrt());
            drafts.push(Draft {
                irrep_dim: dim,
                multiplicity: 1,
                kind: IrrepKind::Real,
                character: Vec::new(),
                projector_error: eta,
                columns: canonical,
                first_pivot,
            });
            blocks.push(CommutantBlock {
                dim,
                clusters: group.len(),
                commutant_dim: 1,
                route: CommutantRoute::Simple,
                kernel_band: None,
                approximate_defects: Vec::new(),
                bottleneck_coupling: bottleneck,
                projector_error: eta,
            });
        } else {
            return Ok(Err(format!(
                "attempt {attempt}: a block of dimension {dim} > {DENSE_COMMUTANT_MAX_N} has a repeated eigenvalue"
            )));
        }
    }
    let exact_dim = blocks.iter().map(|b| b.commutant_dim).sum();
    let (isotypic_basis, isotypic_blocks) = assemble(n, drafts);
    let isotypic = IsotypicBasis {
        basis: isotypic_basis,
        blocks: isotypic_blocks,
        eigenvalue_band: beta,
        separation_over_band: separation,
        commutant_dim: exact_dim,
        attempts: attempt + 1,
    };
    Ok(Ok(OperatorCommutant {
        n,
        members: members.len(),
        exact_dim,
        blocks,
        eigenvalue_band: beta,
        separation_over_band: separation,
        split_coupling: split_coupling.0,
        split_bound: split_coupling.1,
        attempts: 1,
        basis,
        isotypic,
    }))
}

/// The exact commutant of a small compressed family and its nearest approximate directions.
struct DenseKernel {
    basis: Vec<Array2<f64>>,
    band: f64,
    approximate_defects: Vec<f64>,
}

/// The kernel of `L = Σ K_iᵀK_i` over `{A_i, A_iᵀ}`, `K T = T A − A T`. With `vec` column-major,
/// `K = Aᵀ ⊗ I − I ⊗ A` and `KᵀK = AAᵀ ⊗ I − A ⊗ A − Aᵀ ⊗ Aᵀ + I ⊗ AᵀA`, where entry
/// `(i + n j, k + n l)` of `X ⊗ Y` is `X[j, l] Y[i, k]`. `leakage` bounds how far each compressed
/// member is from its exact block restriction (`‖ΔA‖₂ ≤ 2 leakage`), which moves `L` by at most
/// `Σ (2‖K‖‖ΔK‖ + ‖ΔK‖²)` with `‖K‖ ≤ 2`, `‖ΔK‖ ≤ 4 leakage`.
fn dense_kernel(compressed: &[Array2<f64>], leakage: f64, reported: usize) -> Result<DenseKernel, SymmetryError> {
    let n = compressed[0].nrows();
    let dim = n * n;
    let mut l = Array2::<f64>::zeros((dim, dim));
    let mut majorant = 0.0;
    let transposes: Vec<Array2<f64>> = compressed.iter().map(|a| a.t().to_owned()).collect();
    for a in compressed.iter().chain(&transposes) {
        let (aat, ata) = (fast_abt(a, a), fast_atb(a, a));
        let absolute = a.mapv(f64::abs);
        let largest = absolute.iter().copied().fold(0.0_f64, f64::max);
        majorant += fast_abt(&absolute, &absolute).iter().copied().fold(0.0_f64, f64::max)
            + fast_atb(&absolute, &absolute).iter().copied().fold(0.0_f64, f64::max)
            + 2.0 * largest * largest;
        for j in 0..n {
            for lcol in 0..n {
                for i in 0..n {
                    for k in 0..n {
                        let mut value = -a[[j, lcol]] * a[[i, k]] - a[[lcol, j]] * a[[k, i]];
                        if i == k {
                            value += aat[[j, lcol]];
                        }
                        if j == lcol {
                            value += ata[[i, k]];
                        }
                        l[[i + n * j, k + n * lcol]] += value;
                    }
                }
            }
        }
    }
    let (values, vectors) = l.eigh(Side::Lower)?;
    let values: Vec<f64> = values.to_vec();
    let family = 2.0 * compressed.len() as f64;
    let delta = 4.0 * leakage;
    let band = symmetric_spectrum_rounding_band(&values)
        + dim as f64 * accumulation_growth(2 * n + 4 * compressed.len()) * majorant
        + family * (4.0 * delta + delta * delta);
    let exact = values.iter().filter(|&&v| v <= band).count();
    let approximate_defects = values[exact..].iter().take(reported).map(|&v| (v.max(0.0) / family).sqrt()).collect();
    let basis = (0..exact)
        .map(|c| {
            let column = vectors.column(c);
            Array2::from_shape_fn((n, n), |(i, j)| column[i + n * j])
        })
        .collect();
    Ok(DenseKernel { basis, band, approximate_defects })
}

/// `‖T A − A T‖_F` for one pair.
pub fn commutator_norm(t: ArrayView2<'_, f64>, a: ArrayView2<'_, f64>) -> f64 {
    frobenius(&(fast_ab(&t, &a) - fast_ab(&a, &t)))
}

// ---------------------------------------------------------------------------------------------
// Candidate subdomains and twins
// ---------------------------------------------------------------------------------------------

/// The cells and twin classes of a token domain, read from its tables without forming a Gram.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateSubdomains {
    pub n: usize,
    /// Cells of two or more points, each sorted, in order of their smallest point.
    pub cells: Vec<Vec<usize>>,
    /// Refinement rounds, the last of which split nothing.
    pub rounds: usize,
    /// Twin classes of two or more points, each sorted, in order of their smallest point.
    pub twins: Vec<Vec<usize>>,
    /// Largest relative transposition defect of a twin against its class's first point.
    pub max_twin_defect: f64,
    /// The largest table's exactness band on a relative Gram defect.
    pub exact_band: f64,
}

impl CandidateSubdomains {
    /// `log₂ Π_c k_c!`: the order of the twin group.
    pub fn log2_twin_order(&self) -> f64 {
        self.twins.iter().map(|class| (2..=class.len()).map(|k| (k as f64).log2()).sum::<f64>()).sum()
    }

    /// Rows a twin-invariant table needs: one per class and one per other point.
    pub fn quotient_size(&self) -> usize {
        self.n - self.twins.iter().map(|class| class.len() - 1).sum::<usize>()
    }

    /// Cells holding a point that is no twin of the cell's other points: where a symmetry beyond
    /// the twin group could still act.
    pub fn open_cells(&self) -> Vec<&Vec<usize>> {
        self.cells
            .iter()
            .filter(|cell| {
                let twinned = self.twins.iter().filter(|class| cell.contains(&class[0])).map(|class| class.len()).max().unwrap_or(1);
                twinned < cell.len()
            })
            .collect()
    }
}

/// Splits `cell` by one invariant: points whose band intervals chain together (after sorting)
/// stay together, so every split is certified.
fn split_by_intervals(cell: &[usize], values: &[f64], bands: &[f64]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..cell.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then(cell[a].cmp(&cell[b])));
    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut reach = f64::NEG_INFINITY;
    for &i in &order {
        if parts.is_empty() || values[i] - bands[i] > reach {
            parts.push(Vec::new());
            reach = f64::NEG_INFINITY;
        }
        parts.last_mut().expect("a part").push(cell[i]);
        reach = reach.max(values[i] + bands[i]);
    }
    for part in &mut parts {
        part.sort_unstable();
    }
    parts
}

/// The coarsest partition of the domain by band-certified invariants of the Grams `X Xᵀ`, and
/// its twin classes (module note).
pub fn candidate_subdomains(tables: &[ArrayView2<'_, f64>]) -> Result<CandidateSubdomains, SymmetryError> {
    let n = tables.first().map(|t| t.nrows()).ok_or_else(|| SymmetryError::Shape("no tables".into()))?;
    for table in tables {
        if table.nrows() != n || table.iter().any(|v| !v.is_finite()) {
            return Err(SymmetryError::Shape("tables must be finite and share one row count".into()));
        }
    }
    let mut cells: Vec<Vec<usize>> = vec![(0..n).collect()];
    let mut rounds = 0;
    loop {
        rounds += 1;
        // Invariants of a point: its diagonal, its sum against the whole domain, and its sum
        // against every cell of two or more points (a symmetry maps each such cell onto itself).
        let sources: Vec<Vec<usize>> =
            if rounds == 1 { vec![(0..n).collect()] } else { cells.iter().filter(|c| c.len() > 1).cloned().collect() };
        let active: Vec<usize> = cells.iter().filter(|c| c.len() > 1).flatten().copied().collect();
        let mut position = vec![usize::MAX; n];
        for (i, &x) in active.iter().enumerate() {
            position[x] = i;
        }
        let mut invariants: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        for table in tables {
            let d = table.ncols();
            if rounds == 1 {
                let diagonal: Vec<f64> = active.iter().map(|&x| table.row(x).dot(&table.row(x))).collect();
                let bands = diagonal.iter().map(|v| accumulation_growth(d) * v).collect();
                invariants.push((diagonal, bands));
            }
            for source in &sources {
                let mut sum = Array1::<f64>::zeros(d);
                let mut magnitude = Array1::<f64>::zeros(d);
                for &y in source {
                    sum += &table.row(y);
                    magnitude.zip_mut_with(&table.row(y), |m, v| *m += v.abs());
                }
                let growth = accumulation_growth(d + source.len());
                let values = active.iter().map(|&x| table.row(x).dot(&sum)).collect();
                let bands = active
                    .iter()
                    .map(|&x| growth * table.row(x).iter().zip(&magnitude).map(|(v, m)| v.abs() * m).sum::<f64>())
                    .collect();
                invariants.push((values, bands));
            }
        }
        let mut next: Vec<Vec<usize>> = Vec::with_capacity(cells.len());
        let mut split = false;
        for cell in &cells {
            if cell.len() == 1 {
                next.push(cell.clone());
                continue;
            }
            let mut parts = vec![cell.clone()];
            for (values, bands) in &invariants {
                parts = parts
                    .into_iter()
                    .flat_map(|part| {
                        if part.len() == 1 {
                            return vec![part];
                        }
                        let v: Vec<f64> = part.iter().map(|&x| values[position[x]]).collect();
                        let b: Vec<f64> = part.iter().map(|&x| bands[position[x]]).collect();
                        split_by_intervals(&part, &v, &b)
                    })
                    .collect();
            }
            split |= parts.len() > 1;
            next.extend(parts);
        }
        next.sort_by_key(|c| c[0]);
        cells = next;
        if !split {
            break;
        }
    }
    let cells: Vec<Vec<usize>> = cells.into_iter().filter(|c| c.len() > 1).collect();
    // Twins: the transposition (x y) moves the Gram by
    // 4 (δᵀMδ − (δ·X_x)² − (δ·X_y)²) + 2 (G_xx − G_yy)², δ = X_x − X_y, M = XᵀX, ‖G‖_F = ‖M‖_F.
    let mut grams = Vec::with_capacity(tables.len());
    let mut exact_band = 0.0_f64;
    if !cells.is_empty() {
        for table in tables {
            // ‖|X||X|ᵀ‖_F ≤ ‖X‖²_F bounds the rounding majorant with no copy of the table.
            let m = fast_atb(table, table);
            let norm = frobenius(&m);
            let band = 2.0 * accumulation_growth(table.ncols()) * table.iter().map(|v| v * v).sum::<f64>() / norm;
            exact_band = exact_band.max(band);
            grams.push((m, norm, band));
        }
    }
    let transposition_defect = |x: usize, y: usize| -> f64 {
        tables
            .iter()
            .zip(&grams)
            .map(|(table, (m, norm, _))| {
                let (rx, ry) = (table.row(x), table.row(y));
                let delta = &rx - &ry;
                let quadratic = delta.dot(&m.dot(&delta));
                let (px, py) = (delta.dot(&rx), delta.dot(&ry));
                let diagonal = rx.dot(&rx) - ry.dot(&ry);
                (4.0 * (quadratic - px * px - py * py) + 2.0 * diagonal * diagonal).max(0.0).sqrt() / norm
            })
            .fold(0.0, f64::max)
    };
    let within = |defect: f64| grams.iter().all(|(_, _, band)| defect <= *band);
    let mut twins: Vec<Vec<usize>> = Vec::new();
    let mut max_twin_defect = 0.0_f64;
    for cell in &cells {
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for &x in cell {
            let mut joined = false;
            for class in &mut classes {
                let defect = transposition_defect(class[0], x);
                if within(defect) {
                    max_twin_defect = max_twin_defect.max(defect);
                    class.push(x);
                    joined = true;
                    break;
                }
            }
            if !joined {
                classes.push(vec![x]);
            }
        }
        twins.extend(classes.into_iter().filter(|c| c.len() > 1));
    }
    twins.sort_by_key(|c| c[0]);
    Ok(CandidateSubdomains { n, cells, rounds, twins, max_twin_defect, exact_band })
}
