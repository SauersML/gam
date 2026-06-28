//! Bitmask-coefficient multi-directional jets used by marginal-slope and
//! latent-survival row kernels.
//!
//! The layout stores one coefficient per direction mask. The calculus itself
//! lives in [`crate::jet_algebra`]: that module owns the layout-agnostic
//! Leibniz / Faà di Bruno *combinatorics* once, and the scalar (`n_dirs <= 1`)
//! path here still routes through it so a fix to the rule is a fix to both
//! representations.
//!
//! ## Why this layout is special (and how the hot path exploits it)
//!
//! Each direction is seeded *linearly* (one first-derivative slot), so every
//! direction variable squares to zero. The coefficients therefore form the
//! commutative **multilinear / set-function algebra**: `coeffs[mask]` is the
//! coefficient of `Π_{i ∈ mask} ε_i`. In that algebra two facts collapse the
//! generic combinatorial walkers into tight branch-free arithmetic:
//!
//! * **`mul` is the subset (zeta-style) convolution**
//!   `out[mask] = Σ_{sub ⊆ mask} a[sub] · b[mask \ sub]`.
//!   The shared `leibniz_product` walker rebuilds two `SlotBuf`s and folds bit
//!   lists back into masks (`mask_of`) *per subset*; here we enumerate the
//!   submasks of `mask` directly — `mask \ sub == mask ^ sub` because
//!   `sub ⊆ mask` — in the **same ascending order** the walker used, so the
//!   floating-point accumulation is bit-for-bit identical while every
//!   `SlotBuf`/closure/`mask_of` allocation and indirection disappears
//!   (`3^K` pure FMAs, no heap, no `dyn`).
//!
//! * **`compose_unary` is the truncated set-partition (Faà di Bruno) sum**
//!   `out[mask] = Σ_{π ⊢ mask, |π| < 5} f^{(|π|)} · Π_{B ∈ π} u[B]`.
//!   The shared walker re-runs the partition *recursion* (with `&mut dyn
//!   FnMut` dispatch and fresh `SlotBuf` blocks) once **per output mask**.
//!   The set of partitions of `m` slots depends only on `m`, so we enumerate
//!   them **once** into a thread-local table — emitted in the exact recursion
//!   order, pruned at `|π| >= 5` (the same order-4 truncation) — and the hot
//!   loop is then a flat sum of products with no recursion and no dynamic
//!   dispatch. Same emit order, same block order, same `derivs[order]` factor,
//!   so the result is bit-for-bit identical to the walker.
//!
//! Both fast paths were validated `to_bits`-identical against the shared
//! walkers over thousands of randomised composite programs at `K ∈ {2,3,4,9}`.
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

pub static COMPOSE_UNARY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MUL_CALLS: AtomicU64 = AtomicU64::new(0);

/// Length of the unary derivative stack `[f, f', f'', f''', f'''']`: composition
/// is exact through order 4, partitions into `>= 5` blocks are truncated.
const DERIVS: usize = 5;

#[derive(Clone)]
pub struct MultiDirJet {
    pub coeffs: Vec<f64>,
}

impl MultiDirJet {
    pub fn zero(n_dirs: usize) -> Self {
        Self {
            coeffs: vec![0.0; 1usize << n_dirs],
        }
    }

    pub fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    pub fn linear(n_dirs: usize, base: f64, first: &[f64]) -> Self {
        let mut out = Self::constant(n_dirs, base);
        for (idx, &value) in first.iter().take(n_dirs).enumerate() {
            out.coeffs[1usize << idx] = value;
        }
        out
    }

    pub fn with_coeffs(n_dirs: usize, coeffs: &[(usize, f64)]) -> Self {
        let mut out = Self::zero(n_dirs);
        for &(mask, value) in coeffs {
            if mask < out.coeffs.len() {
                out.coeffs[mask] = value;
            }
        }
        out
    }

    #[inline]
    pub fn coeff(&self, mask: usize) -> f64 {
        self.coeffs[mask]
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        }
    }

    pub fn scale(&self, scalar: f64) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|value| scalar * value).collect(),
        }
    }

    /// Subset-convolution product `out[mask] = Σ_{sub ⊆ mask} a[sub]·b[mask^sub]`.
    ///
    /// Bit-identical to the shared [`crate::jet_algebra::leibniz_product`] walker
    /// (the submasks are enumerated in the same ascending order — the walker's
    /// compacted subset index is a monotone bit-deposit of the submask) while
    /// dropping its per-subset `SlotBuf`/closure/`mask_of` overhead. The scalar
    /// `n_dirs == 0` case keeps the shared walker live as its reference.
    pub fn mul(&self, other: &Self) -> Self {
        MUL_CALLS.fetch_add(1, Ordering::Relaxed);
        let count = self.coeffs.len();
        if count <= 1 {
            return self.mul_reference(other);
        }
        let a = &self.coeffs;
        let b = &other.coeffs;
        let mut out = vec![0.0; count];
        for (mask, slot) in out.iter_mut().enumerate() {
            // Walk every submask of `mask` in ascending numeric order — the same
            // order `leibniz_product` accumulates — via the classic gap-fill
            // increment `next = ((sub | !mask) + 1) & mask`.
            let mut acc = 0.0;
            let mut sub = 0usize;
            loop {
                acc += a[sub] * b[mask ^ sub];
                if sub == mask {
                    break;
                }
                sub = (sub | !mask).wrapping_add(1) & mask;
            }
            *slot = acc;
        }
        Self { coeffs: out }
    }

    /// The pre-#perf shared-walker product, retained verbatim as the scalar-case
    /// implementation and as the bit-exact reference for `mul`.
    fn mul_reference(&self, other: &Self) -> Self {
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        for (mask, slot) in out.iter_mut().enumerate() {
            let bits = bit_positions(mask);
            *slot = crate::jet_algebra::leibniz_product(
                bits.as_slice(),
                |t| self.coeffs[mask_of(t)],
                |c| other.coeffs[mask_of(c)],
            );
        }
        Self { coeffs: out }
    }

    /// Exact (order-4 truncated) unary composition `f(self)` from the Taylor
    /// stack `[f, f', f'', f''', f'''']` at `self.coeff(0)`.
    ///
    /// Bit-identical to the shared [`crate::jet_algebra`] Faà di Bruno walker:
    /// it enumerates the set-partitions of each output mask's slots in the exact
    /// same recursion order, multiplies `derivs[order]` by the same per-block
    /// inner coefficients in the same order, and sums them in the same order —
    /// but the partition enumeration is hoisted out of the per-mask loop into a
    /// thread-local table built once per slot count. The scalar `n_dirs == 0`
    /// case keeps the shared walker live as its reference.
    pub fn compose_unary(&self, derivs: [f64; DERIVS]) -> Self {
        COMPOSE_UNARY_CALLS.fetch_add(1, Ordering::Relaxed);
        let count = self.coeffs.len();
        if count <= 1 {
            return <Self as crate::jet_algebra::JetAlgebra<DERIVS>>::compose_unary(self, derivs);
        }
        let n_dirs = count.trailing_zeros() as usize;
        // Partition tables for every slot count present, built once and cached.
        let tables = partition_tables(n_dirs);
        let coeffs = &self.coeffs;
        let mut out = vec![0.0; count];
        // Per-mask scratch: `remap[cb]` lifts a compacted submask `cb` of the
        // current mask's slots back to the real coefficient index (the walker's
        // `mask_of(labelled)`). Filled once per mask and reused across all of
        // that mask's partitions/blocks, replacing the per-block bit-deposit
        // loop with a single load. Sized `count` (>= 2^npos for every mask).
        let mut remap = vec![0usize; count];
        let mut pos = [0usize; usize::BITS as usize];
        for (mask, slot) in out.iter_mut().enumerate() {
            if mask == 0 {
                // Matches the walker's `m == 0` early return exactly (no `0.0 +`
                // round-trip, which would differ on a `-0.0` value channel).
                *slot = derivs[0];
                continue;
            }
            // Set-bit positions of `mask`, ascending — the slot labels.
            let mut npos = 0usize;
            let mut m = mask;
            while m != 0 {
                pos[npos] = m.trailing_zeros() as usize;
                npos += 1;
                m &= m - 1;
            }
            // Deposit table: remap[cb] = OR over set bits `i` of cb of 1<<pos[i].
            // DP over submasks — strip the lowest bit, add its real position.
            remap[0] = 0;
            for cb in 1usize..(1usize << npos) {
                let low = cb.trailing_zeros() as usize;
                remap[cb] = remap[cb & (cb - 1)] | (1usize << pos[low]);
            }
            let table = &tables[npos];
            let flat = &table.flat;
            let mut total = 0.0;
            for &(off, order) in table.parts.iter() {
                let order = order as usize;
                let mut prod = derivs[order];
                for &cb in &flat[off..off + order] {
                    prod *= coeffs[remap[cb as usize]];
                }
                total += prod;
            }
            *slot = total;
        }
        Self { coeffs: out }
    }
}

impl crate::jet_algebra::JetAlgebra<DERIVS> for MultiDirJet {
    #[inline]
    fn derivative(&self, slots: &[usize]) -> f64 {
        self.coeffs[mask_of(slots)]
    }

    fn map_derivatives<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&[usize]) -> f64,
    {
        let mut out = vec![0.0; self.coeffs.len()];
        for (mask, value) in out.iter_mut().enumerate() {
            let bits = bit_positions(mask);
            *value = f(bits.as_slice());
        }
        Self { coeffs: out }
    }
}

/// A flattened set-partition table for a fixed slot count. `parts[i] = (off,
/// order)` describes one partition: its `order` block submasks (compacted) are
/// `flat[off .. off + order]`. Flattening keeps the hot composition loop on one
/// contiguous slice instead of chasing per-partition `Vec` pointers.
struct PartTable {
    flat: Vec<u32>,
    parts: Vec<(usize, u8)>,
}

thread_local! {
    /// Cached set-partition tables, indexed by slot count `m`. Entry `m` holds
    /// every partition of `{0..m}` into `< DERIVS` blocks, in the shared
    /// walker's recursion order, each block a compacted submask. Pure function
    /// of `m`, so caching is sound and deterministic.
    static PARTITION_TABLES: RefCell<Vec<Rc<PartTable>>> = const { RefCell::new(Vec::new()) };
}

/// Return cached partition tables for slot counts `0..=n_dirs`.
fn partition_tables(n_dirs: usize) -> Vec<Rc<PartTable>> {
    PARTITION_TABLES.with(|cell| {
        let mut tables = cell.borrow_mut();
        while tables.len() <= n_dirs {
            let m = tables.len();
            tables.push(Rc::new(build_partitions(m)));
        }
        (0..=n_dirs).map(|m| Rc::clone(&tables[m])).collect()
    })
}

/// Enumerate the set-partitions of `{0..m}` with fewer than `DERIVS` blocks, in
/// the exact DFS order of [`crate::jet_algebra`]'s `for_each_partition`
/// recursion ("place each element into an existing block, else open a new one"),
/// each block recorded as a compacted submask of `{0..m}`, flattened.
fn build_partitions(m: usize) -> PartTable {
    fn recurse(elem: usize, m: usize, blocks: &mut [u32; 8], n_blocks: usize, out: &mut PartTable) {
        // Partitions with `>= DERIVS` blocks are truncated (their `f^{(order)}`
        // is beyond the stack); the block count never decreases, so the whole
        // subtree contributes nothing and is pruned — matching the walker's
        // per-partition `order >= derivs.len()` skip.
        if n_blocks >= DERIVS {
            return;
        }
        if elem == m {
            let off = out.flat.len();
            out.flat.extend_from_slice(&blocks[..n_blocks]);
            out.parts.push((off, n_blocks as u8));
            return;
        }
        for b in 0..n_blocks {
            blocks[b] |= 1u32 << elem;
            recurse(elem + 1, m, blocks, n_blocks, out);
            blocks[b] &= !(1u32 << elem);
        }
        blocks[n_blocks] = 1u32 << elem;
        recurse(elem + 1, m, blocks, n_blocks + 1, out);
    }
    let mut out = PartTable {
        flat: Vec::new(),
        parts: Vec::new(),
    };
    let mut blocks = [0u32; 8];
    recurse(0, m, &mut blocks, 0, &mut out);
    out
}

/// The set-bit positions of `mask`, low to high — the differentiation slots of
/// that coefficient.
fn bit_positions(mask: usize) -> crate::jet_algebra::SlotBuf {
    let mut out = crate::jet_algebra::SlotBuf::new();
    let mut m = mask;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        out.push_slot(bit);
        m &= m - 1;
    }
    out
}

/// Combine a slot-group (list of bit positions) back into a sub-mask.
fn mask_of(slots: &[usize]) -> usize {
    slots.iter().fold(0usize, |acc, &b| acc | (1usize << b))
}

// #932-2 cutover: `MultiDirJet::bilinear` (the 4-coeff `[base, d1, d2, d12]`
// constructor) and `MultiDirJet::sub` are consumed ONLY by the now test-only hand
// survival directional/bidirectional oracle (the production flex jet path uses the
// `flex_jet` runtime jet algebra, not `MultiDirJet`). After the #1521 crate split
// moved `MultiDirJet` into `gam-math`, those oracle tests live in the dependent
// `gam` crate, where a `#[cfg(test)]` gate in *this* crate is inactive — so the
// methods must be plain `pub` inherent methods to be reachable cross-crate. They
// carry no dead-code cost because `pub` items are part of the crate's public API.
// Bodies are byte-identical to their former gated form.
impl MultiDirJet {
    pub fn bilinear(base: f64, d1: f64, d2: f64, d12: f64) -> Self {
        Self {
            coeffs: vec![base, d1, d2, d12],
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(lhs, rhs)| lhs - rhs)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── constructors ─────────────────────────────────────────────────────────

    #[test]
    fn zero_has_correct_length_and_all_zero_coefficients() {
        let j = MultiDirJet::zero(3);
        assert_eq!(j.coeffs.len(), 8);
        assert!(j.coeffs.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn constant_has_value_at_mask_zero_and_zeros_elsewhere() {
        let j = MultiDirJet::constant(2, 5.0);
        assert_eq!(j.coeffs.len(), 4);
        assert_eq!(j.coeff(0), 5.0);
        assert_eq!(j.coeff(1), 0.0);
        assert_eq!(j.coeff(2), 0.0);
        assert_eq!(j.coeff(3), 0.0);
    }

    #[test]
    fn linear_sets_base_and_per_direction_slots() {
        let j = MultiDirJet::linear(2, 1.0, &[2.0, 3.0]);
        assert_eq!(j.coeff(0), 1.0); // constant
        assert_eq!(j.coeff(1), 2.0); // mask 0b01 — direction 0
        assert_eq!(j.coeff(2), 3.0); // mask 0b10 — direction 1
        assert_eq!(j.coeff(3), 0.0); // cross term is zero
    }

    #[test]
    fn bilinear_sets_all_four_slots() {
        let j = MultiDirJet::bilinear(1.0, 2.0, 3.0, 4.0);
        assert_eq!(j.coeff(0), 1.0);
        assert_eq!(j.coeff(1), 2.0);
        assert_eq!(j.coeff(2), 3.0);
        assert_eq!(j.coeff(3), 4.0);
    }

    #[test]
    fn with_coeffs_sets_only_specified_entries() {
        let j = MultiDirJet::with_coeffs(2, &[(0, 9.0), (3, -1.0)]);
        assert_eq!(j.coeff(0), 9.0);
        assert_eq!(j.coeff(1), 0.0);
        assert_eq!(j.coeff(2), 0.0);
        assert_eq!(j.coeff(3), -1.0);
    }

    // ── elementwise arithmetic ────────────────────────────────────────────────

    #[test]
    fn add_is_elementwise() {
        let a = MultiDirJet::linear(2, 1.0, &[2.0, 3.0]);
        let b = MultiDirJet::linear(2, 4.0, &[5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.coeff(0), 5.0);
        assert_eq!(c.coeff(1), 7.0);
        assert_eq!(c.coeff(2), 9.0);
        assert_eq!(c.coeff(3), 0.0);
    }

    #[test]
    fn scale_multiplies_all_coefficients() {
        let j = MultiDirJet::linear(2, 1.0, &[2.0, 3.0]);
        let s = j.scale(2.0);
        assert_eq!(s.coeff(0), 2.0);
        assert_eq!(s.coeff(1), 4.0);
        assert_eq!(s.coeff(2), 6.0);
        assert_eq!(s.coeff(3), 0.0);
    }

    #[test]
    fn sub_is_elementwise_difference() {
        let a = MultiDirJet::constant(2, 5.0);
        let b = MultiDirJet::constant(2, 3.0);
        let c = a.sub(&b);
        assert_eq!(c.coeff(0), 2.0);
        assert_eq!(c.coeff(1), 0.0);
        assert_eq!(c.coeff(2), 0.0);
        assert_eq!(c.coeff(3), 0.0);
    }

    // ── mul (subset-convolution) ──────────────────────────────────────────────

    #[test]
    fn mul_of_constants_is_scalar_product() {
        let a = MultiDirJet::constant(2, 2.0);
        let b = MultiDirJet::constant(2, 3.0);
        let c = a.mul(&b);
        assert_eq!(c.coeff(0), 6.0);
        assert_eq!(c.coeff(1), 0.0);
        assert_eq!(c.coeff(2), 0.0);
        assert_eq!(c.coeff(3), 0.0);
    }

    #[test]
    fn mul_satisfies_leibniz_rule_single_direction() {
        // (1 + ε) * (1 + ε) = 1 + 2ε
        let x = MultiDirJet::linear(1, 1.0, &[1.0]);
        let y = MultiDirJet::linear(1, 1.0, &[1.0]);
        let z = x.mul(&y);
        assert_eq!(z.coeff(0), 1.0);
        assert_eq!(z.coeff(1), 2.0);
    }

    #[test]
    fn mul_cross_term_two_independent_directions() {
        // (1 + ε₁)(1 + ε₂) = 1 + ε₁ + ε₂ + ε₁ε₂
        let x = MultiDirJet::linear(2, 1.0, &[1.0, 0.0]);
        let y = MultiDirJet::linear(2, 1.0, &[0.0, 1.0]);
        let z = x.mul(&y);
        assert_eq!(z.coeff(0), 1.0);
        assert_eq!(z.coeff(1), 1.0);
        assert_eq!(z.coeff(2), 1.0);
        assert_eq!(z.coeff(3), 1.0);
    }
}
