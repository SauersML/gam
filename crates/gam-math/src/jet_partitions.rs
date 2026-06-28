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
//! * **`compose_unary` is the truncated Faà di Bruno composition**, computed
//!   here by the exact **truncated-Taylor reassociation** rather than a direct
//!   set-partition sum. Let `v` be the non-constant part of `self`
//!   (`v[0] = 0`, `v[mask] = self[mask]`) and let `v^{⊛k}` be the `k`-fold
//!   *subset convolution* (the multilinear power). The ordered-tuple identity
//!   `v^{⊛k}[mask] = k! · Σ_{π ⊢ mask, |π| = k} Π_{B ∈ π} v[B]` turns the
//!   set-partition sum into a degree-4 polynomial in `v`:
//!
//!   ```text
//!   f(self)[mask] = Σ_{k=0}^{4} (f^{(k)} / k!) · v^{⊛k}[mask]      (mask ≠ 0)
//!   f(self)[0]    = f^{(0)}
//!   ```
//!
//!   so a composition is just **three subset convolutions** (`v²`, `v³=v²⊛v`,
//!   `v⁴=v²⊛v²` — the Motzkin floor for a quartic) plus a five-term combine.
//!   That is ~3× fewer FLOPs than the per-mask partition gather; each
//!   convolution is a four-lane compensated dot product (Ogita–Rump–Oishi
//!   Dot2, FMA-split products + TwoSum carry) so the result is computed in
//!   ~double the working precision and the rounding of `v²` cannot compound
//!   through `v³`/`v⁴`; the final per-mask combine is Neumaier-compensated and
//!   `wide::f64x4`-vectorised; and the whole call runs on reused thread-local
//!   scratch with no per-call heap traffic. The reassociation is algebraically
//!   exact; accuracy-vs-truth (a double-double oracle) is the test gate and is
//!   strictly ≤ the old partition sum's error (see `tests`).
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use wide::{CmpGe, f64x4};

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
    /// Computed by the truncated-Taylor reassociation (see the module note):
    /// `f(self) = Σ_{k=0}^{4} (f^{(k)}/k!)·v^{⊛k}` with `v` the non-constant
    /// part of `self`. The three subset-convolution powers `v²`, `v³`, `v⁴`
    /// are compensated (Dot2) and the per-mask combine is Neumaier-compensated
    /// and vectorised, so the result is *more* accurate vs. the true
    /// real-arithmetic value than the prior naive partition sum (proven against
    /// a double-double oracle in `tests`). The scalar `n_dirs == 0` case keeps
    /// the shared Faà di Bruno walker live as its reference.
    pub fn compose_unary(&self, derivs: [f64; DERIVS]) -> Self {
        COMPOSE_UNARY_CALLS.fetch_add(1, Ordering::Relaxed);
        let count = self.coeffs.len();
        if count <= 1 {
            return <Self as crate::jet_algebra::JetAlgebra<DERIVS>>::compose_unary(self, derivs);
        }
        // Per-block Taylor coefficients c_k = f^{(k)} / k!  (k = 1..=4): the
        // `1/k!` undoes the ordered-tuple overcount of the subset-convolution
        // power v^{⊛k} relative to the unordered set-partition sum.
        let c1 = derivs[1];
        let c2 = derivs[2] * 0.5;
        let c3 = derivs[3] * (1.0 / 6.0);
        let c4 = derivs[4] * (1.0 / 24.0);

        let mut out = vec![0.0; count];
        COMPOSE_SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            // Four contiguous scratch lanes: v, p2 = v², p3 = v³, p4 = v⁴.
            buf.clear();
            buf.resize(4 * count, 0.0);
            let (vbuf, rest) = buf.split_at_mut(count);
            let (p2, rest) = rest.split_at_mut(count);
            let (p3, p4) = rest.split_at_mut(count);

            // v = non-constant part of self (the constant channel squares to a
            // 0-block, which the k = 0 term carries separately).
            vbuf.copy_from_slice(&self.coeffs);
            vbuf[0] = 0.0;

            // Powers via compensated subset convolution, pruned by output
            // popcount: v^{⊛k}[mask] = 0 whenever popcount(mask) < k.
            subset_conv_into(vbuf, vbuf, p2, 2);
            subset_conv_into(p2, vbuf, p3, 3);
            subset_conv_into(p2, p2, p4, 4);

            // out[mask] = c1·v + c2·v² + c3·v³ + c4·v⁴ (mask ≠ 0), Neumaier-
            // compensated and f64x4-vectorised over masks. out[0] = f^{(0)}.
            combine_powers(vbuf, p2, p3, p4, [c1, c2, c3, c4], &mut out);
            out[0] = derivs[0];
        });
        Self { coeffs: out }
    }
}

thread_local! {
    /// Reused composition scratch (`4·count` f64s: v, v², v³, v⁴). Sized up on
    /// demand and never freed, so a steady-state `compose_unary` does zero heap
    /// work beyond the owned output `Vec`.
    static COMPOSE_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Branchless TwoSum: returns `(s, e)` with `s = fl(a+b)` and `a+b = s+e`
/// exactly (Knuth/Møller). Used by the compensated convolution and combine.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// Subset (zeta-style) convolution `out[mask] = Σ_{sub ⊆ mask} a[sub]·b[mask^sub]`,
/// evaluated as a **compensated dot product** (Ogita–Rump–Oishi Dot2): each
/// product is split into head + FMA error (`mul_add`) and the running sum
/// carries a TwoSum error term, so the result is accurate as if computed in
/// ~twice the working precision. This stops the rounding of `v²` from
/// compounding through `v³`/`v⁴`, which a single-rounding accumulation does
/// not. Output masks with `popcount < min_pop` are left at zero: the
/// multilinear power `v^{⊛k}` vanishes below popcount `k`, so the prune is exact
/// and skips the low-order masks entirely.
#[inline]
fn subset_conv_into(a: &[f64], b: &[f64], out: &mut [f64], min_pop: u32) {
    for (mask, slot) in out.iter_mut().enumerate() {
        if (mask as u64).count_ones() < min_pop {
            *slot = 0.0;
            continue;
        }
        // Descending submask enumeration `sub = (sub-1) & mask`, terminating
        // after `sub == 0` (the classic Gosper-style submask walk). The Dot2 is
        // spread across FOUR independent named accumulators (a 4-way unroll) so
        // the FMA/TwoSum latency chains overlap — the loop becomes throughput-
        // rather than latency-bound — then the lanes are merged with a final
        // compensated reduction. Every non-pruned mask has popcount ≥ 2, so its
        // `2^popcount` submask count is a multiple of 4 and the unroll is exact
        // (the all-zero submask always lands in the fourth lane). Reassociation
        // only; the value is the same real sum, in ~double the working precision.
        #[inline(always)]
        fn dot2_step(s: &mut f64, c: &mut f64, x: f64, y: f64) {
            let prod = x * y;
            let prod_err = x.mul_add(y, -prod); // exact: prod + prod_err == x*y
            let (t, sum_err) = two_sum(*s, prod);
            *s = t;
            *c += prod_err + sum_err;
        }
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut c0, mut c1, mut c2, mut c3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut sub = mask;
        loop {
            dot2_step(&mut s0, &mut c0, a[sub], b[mask ^ sub]);
            sub = (sub - 1) & mask;
            dot2_step(&mut s1, &mut c1, a[sub], b[mask ^ sub]);
            sub = (sub - 1) & mask;
            dot2_step(&mut s2, &mut c2, a[sub], b[mask ^ sub]);
            sub = (sub - 1) & mask;
            dot2_step(&mut s3, &mut c3, a[sub], b[mask ^ sub]);
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & mask;
        }
        // Merge the four lanes, compensated.
        let (s01, e01) = two_sum(s0, s1);
        let (s23, e23) = two_sum(s2, s3);
        let (total, etot) = two_sum(s01, s23);
        *slot = total + (etot + e01 + e23 + c0 + c1 + c2 + c3);
    }
}

/// `out[mask] = c[0]·p1 + c[1]·p2 + c[2]·p3 + c[3]·p4` for `mask ≥ 1`, with a
/// Neumaier-compensated four-term accumulation (the powers span growing
/// magnitudes, so the compensation recovers the bits a naive `+=` would drop)
/// and a `wide::f64x4` body over four masks at a time. `out[0]` is overwritten
/// by the caller with the value channel.
#[inline]
fn combine_powers(p1: &[f64], p2: &[f64], p3: &[f64], p4: &[f64], c: [f64; 4], out: &mut [f64]) {
    let n = out.len();
    let (c1, c2, c3, c4) = (c[0], c[1], c[2], c[3]);
    let (v1, v2, v3, v4) = (
        f64x4::splat(c1),
        f64x4::splat(c2),
        f64x4::splat(c3),
        f64x4::splat(c4),
    );
    let mut mask = 0usize;
    // Vector body: four contiguous masks per step. Neumaier compensation is
    // applied lane-wise; pick the larger magnitude to subtract first.
    while mask + 4 <= n {
        let load = |p: &[f64]| f64x4::new([p[mask], p[mask + 1], p[mask + 2], p[mask + 3]]);
        let mut s = v1 * load(p1);
        let mut comp = f64x4::splat(0.0);
        for (cv, pv) in [(v2, p2), (v3, p3), (v4, p4)] {
            let term = cv * load(pv);
            let t = s + term;
            let big_s = s.abs().cmp_ge(term.abs());
            let lost = big_s.blend((s - t) + term, (term - t) + s);
            comp += lost;
            s = t;
        }
        let res = s + comp;
        out[mask..mask + 4].copy_from_slice(&res.to_array());
        mask += 4;
    }
    // Scalar tail (and the small-K path where `n < 4`).
    while mask < n {
        let mut s = c1 * p1[mask];
        let mut comp = 0.0f64;
        for (cv, pv) in [(c2, p2), (c3, p3), (c4, p4)] {
            let term = cv * pv[mask];
            let (t, e) = two_sum(s, term);
            comp += e;
            s = t;
        }
        out[mask] = s + comp;
        mask += 1;
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

    /// A flattened set-partition table for a fixed slot count. `parts[i] = (off,
    /// order)` describes one partition: its `order` block submasks (compacted) are
    /// `flat[off .. off + order]`.
    ///
    /// This direct set-partition sum is the previous production `compose_unary`
    /// implementation, retained as the **accuracy reference** the new
    /// truncated-Taylor path is graded against: a double-double oracle is the
    /// truth, and the test asserts the new path's error-vs-truth is `≤` this naive
    /// partition sum's error-vs-truth on every randomised program.
    struct PartTable {
        flat: Vec<u32>,
        parts: Vec<(usize, u8)>,
    }

    thread_local! {
        /// Cached set-partition tables, indexed by slot count `m`. Entry `m` holds
        /// every partition of `{0..m}` into `< DERIVS` blocks, in the shared
        /// walker's recursion order, each block a compacted submask. Pure function
        /// of `m`, so caching is sound and deterministic.
        static PARTITION_TABLES: RefCell<Vec<std::rc::Rc<PartTable>>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Return cached partition tables for slot counts `0..=n_dirs`.
    fn partition_tables(n_dirs: usize) -> Vec<std::rc::Rc<PartTable>> {
        PARTITION_TABLES.with(|cell| {
            let mut tables = cell.borrow_mut();
            while tables.len() <= n_dirs {
                let m = tables.len();
                tables.push(std::rc::Rc::new(build_partitions(m)));
            }
            (0..=n_dirs).map(|m| std::rc::Rc::clone(&tables[m])).collect()
        })
    }

    /// The previous production `compose_unary`: a direct set-partition (Faà di
    /// Bruno) sum per output mask, retained as the accuracy reference.
    fn compose_unary_partition_reference(coeffs: &[f64], derivs: [f64; DERIVS]) -> Vec<f64> {
        let count = coeffs.len();
        let n_dirs = count.trailing_zeros() as usize;
        let tables = partition_tables(n_dirs);
        let mut out = vec![0.0; count];
        let mut remap = vec![0usize; count];
        let mut pos = [0usize; usize::BITS as usize];
        for (mask, slot) in out.iter_mut().enumerate() {
            if mask == 0 {
                *slot = derivs[0];
                continue;
            }
            let mut npos = 0usize;
            let mut m = mask;
            while m != 0 {
                pos[npos] = m.trailing_zeros() as usize;
                npos += 1;
                m &= m - 1;
            }
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
        out
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

    // ── compose_unary: truncated-Taylor reassociation ─────────────────────────
    //
    // The new `compose_unary` reassociates the per-mask Faà di Bruno set-partition
    // sum into a degree-4 polynomial in the subset-convolution power of the
    // non-constant part. These tests are the accuracy gate: a double-double
    // oracle is the truth, and the new path's error-vs-truth must be `≤` the old
    // naive partition sum's error-vs-truth on every randomised program.

    /// Deterministic xorshift64* — no `rand` dependency in the test.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        /// Uniform in `[-scale, scale]`.
        fn signed(&mut self, scale: f64) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            (2.0 * u - 1.0) * scale
        }
    }

    // ── A double-double oracle for the exact (order-4 truncated) composition ──

    #[inline]
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        (p, a.mul_add(b, -p))
    }
    #[inline]
    fn dd_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }
    #[derive(Clone, Copy)]
    struct Dd {
        hi: f64,
        lo: f64,
    }
    impl Dd {
        fn from(x: f64) -> Self {
            Self { hi: x, lo: 0.0 }
        }
        fn mul_f64(self, b: f64) -> Self {
            let (p, e) = two_prod(self.hi, b);
            let lo = self.lo.mul_add(b, e);
            let s = p + lo;
            Self {
                hi: s,
                lo: (p - s) + lo,
            }
        }
        fn add(self, o: Self) -> Self {
            let (s, e) = dd_two_sum(self.hi, o.hi);
            let (s2, e2) = dd_two_sum(self.lo, o.lo);
            let lo = e + s2;
            let h1 = s + lo;
            let l1 = (s - h1) + lo;
            let lo2 = l1 + e2;
            let h = h1 + lo2;
            Self {
                hi: h,
                lo: (h1 - h) + lo2,
            }
        }
        /// `|self - x|` to ~double precision in the residual (Sterbenz: `x` and
        /// `hi` agree to ~53 bits, so `x - hi` is essentially exact).
        fn abs_err_to(self, x: f64) -> f64 {
            ((x - self.hi) - self.lo).abs()
        }
    }

    /// High-precision truth for `compose_unary` via the set-partition reference,
    /// every product and sum carried in double-double.
    fn compose_truth(coeffs: &[f64], derivs: [f64; DERIVS]) -> Vec<Dd> {
        let count = coeffs.len();
        let n_dirs = count.trailing_zeros() as usize;
        let tables = partition_tables(n_dirs);
        let mut out = vec![Dd::from(0.0); count];
        let mut remap = vec![0usize; count];
        let mut pos = [0usize; 64];
        for (mask, slot) in out.iter_mut().enumerate() {
            if mask == 0 {
                *slot = Dd::from(derivs[0]);
                continue;
            }
            let mut npos = 0usize;
            let mut m = mask;
            while m != 0 {
                pos[npos] = m.trailing_zeros() as usize;
                npos += 1;
                m &= m - 1;
            }
            remap[0] = 0;
            for cb in 1usize..(1usize << npos) {
                let low = cb.trailing_zeros() as usize;
                remap[cb] = remap[cb & (cb - 1)] | (1usize << pos[low]);
            }
            let table = &tables[npos];
            let mut total = Dd::from(0.0);
            for &(off, order) in table.parts.iter() {
                let order = order as usize;
                let mut prod = Dd::from(derivs[order]);
                for &cb in &table.flat[off..off + order] {
                    prod = prod.mul_f64(coeffs[remap[cb as usize]]);
                }
                total = total.add(prod);
            }
            *slot = total;
        }
        out
    }

    /// Build a random composite jet so the composition input is a realistic
    /// non-trivial multilinear element (not just seeded directions).
    fn random_inner(n_dirs: usize, rng: &mut Rng) -> MultiDirJet {
        let base = rng.signed(0.8);
        let first: Vec<f64> = (0..n_dirs).map(|_| rng.signed(0.6)).collect();
        let a = MultiDirJet::linear(n_dirs, base, &first);
        let b = MultiDirJet::linear(
            n_dirs,
            rng.signed(0.7),
            &(0..n_dirs).map(|_| rng.signed(0.5)).collect::<Vec<_>>(),
        );
        // a*b + a populates the full cross-mask spectrum.
        a.mul(&b).add(&a)
    }

    #[test]
    fn compose_unary_matches_partition_reference_simple() {
        // exp-like stack on a 2-direction cross jet: every coeff agrees with the
        // direct set-partition reference to a tight tolerance.
        let j = MultiDirJet::linear(2, 0.3, &[0.5, -0.4])
            .mul(&MultiDirJet::linear(2, -0.2, &[0.1, 0.7]));
        let d = [0.9_f64, 1.1, -0.7, 0.4, -0.25];
        let got = j.compose_unary(d);
        let want = compose_unary_partition_reference(&j.coeffs, d);
        for (mask, (&g, &w)) in got.coeffs.iter().zip(want.iter()).enumerate() {
            let tol = 1e-13 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "mask {mask}: got={g:.17e} want={w:.17e}"
            );
        }
    }

    #[test]
    fn compose_unary_accuracy_beats_partition_sum_vs_double_double() {
        // The accuracy gate. Over many random programs at every K used in
        // production, the new path's error-vs-truth is never worse than the old
        // naive partition sum's, and is a strict improvement in aggregate.
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut sum_new = 0.0f64;
        let mut sum_old = 0.0f64;
        for &n_dirs in &[2usize, 3, 4, 6, 8] {
            for _ in 0..200 {
                let inner = random_inner(n_dirs, &mut rng);
                let d = [
                    rng.signed(1.5),
                    rng.signed(1.5),
                    rng.signed(2.0),
                    rng.signed(3.0),
                    rng.signed(4.0),
                ];
                let new = inner.compose_unary(d);
                let old = compose_unary_partition_reference(&inner.coeffs, d);
                let truth = compose_truth(&inner.coeffs, d);
                for mask in 0..inner.coeffs.len() {
                    let en = truth[mask].abs_err_to(new.coeffs[mask]);
                    let eo = truth[mask].abs_err_to(old[mask]);
                    sum_new += en;
                    sum_old += eo;
                    // Per-coefficient: new is never materially worse. The 4 ULP
                    // slack absorbs the rare tie where a differently-grouped but
                    // equally-valid rounding lands one ULP either way.
                    let scale = truth[mask].hi.abs().max(1.0);
                    assert!(
                        en <= eo + 4.0 * f64::EPSILON * scale,
                        "K={n_dirs} mask={mask}: new_err={en:.3e} old_err={eo:.3e}"
                    );
                }
            }
        }
        // Aggregate: the compensated reassociation is a real improvement.
        assert!(
            sum_new <= sum_old,
            "aggregate error regressed: new={sum_new:.6e} old={sum_old:.6e}"
        );
        eprintln!(
            "compose_unary accuracy: total |err| new={sum_new:.6e} old={sum_old:.6e} \
             (improvement {:.2}x)",
            sum_old / sum_new.max(f64::MIN_POSITIVE)
        );
    }

    #[test]
    fn compose_unary_speedup_over_partition_sum() {
        // Measure ns/call new vs. the previous partition-sum implementation
        // across the production K range. Prints the multiple; asserts a
        // conservative floor so CI noise can't make it flaky.
        use std::time::Instant;
        let mut rng = Rng(0xfeed_face_dead_beef);
        for &n_dirs in &[2usize, 4, 6, 8] {
            let n_inputs = 256usize;
            let inputs: Vec<(MultiDirJet, [f64; DERIVS])> = (0..n_inputs)
                .map(|_| {
                    (
                        random_inner(n_dirs, &mut rng),
                        [
                            rng.signed(1.5),
                            rng.signed(1.5),
                            rng.signed(2.0),
                            rng.signed(3.0),
                            rng.signed(4.0),
                        ],
                    )
                })
                .collect();
            let iters = 200usize;
            // Warm the scratch / partition tables.
            for (j, d) in &inputs {
                std::hint::black_box(j.compose_unary(*d));
                std::hint::black_box(compose_unary_partition_reference(&j.coeffs, *d));
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                for (j, d) in &inputs {
                    std::hint::black_box(j.compose_unary(*d));
                }
            }
            let new_ns = t0.elapsed().as_nanos() as f64 / (iters * inputs.len()) as f64;
            let t1 = Instant::now();
            for _ in 0..iters {
                for (j, d) in &inputs {
                    std::hint::black_box(compose_unary_partition_reference(&j.coeffs, *d));
                }
            }
            let old_ns = t1.elapsed().as_nanos() as f64 / (iters * inputs.len()) as f64;
            eprintln!(
                "compose_unary K={n_dirs}: new={new_ns:.1} ns/call  old={old_ns:.1} ns/call  \
                 speedup={:.2}x",
                old_ns / new_ns
            );
            // Guard only where the algorithmic win is robust: an optimised build
            // at the production-dominant K (the partition sum's `Σ_π |π|` work
            // grows steeply with K, while the new path is three convolutions).
            // Debug builds and tiny K are dominated by fixed per-call overhead
            // and the ratio there is not a meaningful guard, so it is printed
            // but not asserted (and timing asserts must not flake on CI).
            if !cfg!(debug_assertions) && n_dirs >= 6 {
                assert!(
                    new_ns < old_ns,
                    "K={n_dirs} new path slower: new={new_ns:.1}ns old={old_ns:.1}ns"
                );
            }
        }
    }
}
