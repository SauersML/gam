//! The single shared Faà di Bruno / Leibniz combinatorial kernel (#1151).
//!
//! Two jet representations live in this crate and historically each carried
//! its own hand-written copy of the same calculus:
//!
//! * [`crate::jet_tower::Tower4`] — full dense derivative tensors
//!   (`v`, `g`, `h`, `t3`, `t4`) in `K` primary variables, with the
//!   Leibniz product and Faà di Bruno composition written out term-by-term
//!   per derivative order.
//! * [`crate::jet_partitions::MultiDirJet`] — bitmask-coefficient jet over
//!   distinct seeded directions, with the same two rules written as general
//!   submask / set-partition loops.
//!
//! The data layouts are legitimately different (a complete small-`K` tower
//! vs. a handful of directions of a large-`K` expression) and stay separate.
//! What is identical is the *combinatorics*: for a group of differentiation
//! slots, the Leibniz rule sums over subsets of those slots and the Faà di
//! Bruno rule sums over their set-partitions. This module owns that
//! combinatorics once, as a layout-agnostic [`JetAlgebra`] trait plus walkers
//! parameterised by closures that read each representation's own derivative
//! for a slot-group. Both
//! `Tower4` and `MultiDirJet` route their `mul` / `compose_unary` through
//! these walkers, so a fix to the rule is a fix to both — and a bit-exact
//! equivalence test (see `tests`) proves the two layouts agree.
//!
//! A "slot-group" is a list of positions `0..m` (the differentiation
//! arguments of one output coefficient). Each representation maps a group to
//! a derivative:
//!
//! * For a tensor index tuple `(i, j, k, l)`, the positions `0..m` carry
//!   axis labels `[i, j, k, l]`; a sub-group of positions selects the
//!   corresponding lower-order tensor entry (e.g. positions `{0, 2}` →
//!   `h[i][k]`).
//! * For a bitmask coefficient `mask`, the set bits are the slots; a
//!   sub-group of bits is itself a sub-mask read straight out of `coeffs`.
//!
//! # Performance: the combinatorics is precomputed, not re-walked (#1151 perf)
//!
//! The subset enumeration (Leibniz) and the set-partition enumeration (Faà di
//! Bruno) over `m` slots are *fixed combinatorial objects*: they depend only
//! on the slot count `m`, never on the actual derivative values. The hot per-row
//! jet path nevertheless calls these walkers millions of times, and the original
//! kernel rebuilt that structure from scratch on every call — the Faà di Bruno
//! walk via a recursive `&mut dyn FnMut` "assign each element to a block"
//! enumeration whose leaf and every block read went through a vtable that never
//! inlined, plus a freshly cleared [`SlotBuf`] per block; the Leibniz walk via a
//! per-bit branch building two [`SlotBuf`]s for every one of the `2^m` subsets.
//!
//! Both structures are now built ONCE per slot-count `m` (lazily, into a
//! process-wide cache keyed by `m ∈ 0..=8`) and stored as flat, packed bitmask
//! tables. The walkers iterate those tables with straight-line loops — no
//! recursion, no `&mut dyn FnMut` dispatch, no per-call structure rebuild — so
//! the only work left on the hot path is the actual arithmetic (the closure
//! reads of derivatives and their products/sums). The emission order of the
//! cached tables is, by construction, the EXACT order the former recursive /
//! branch walkers produced (the table builder runs that same enumeration once),
//! so every product is left-associated identically and every channel's sum
//! accumulates in the same order: the result is `to_bits`-identical to the
//! former walkers, only with the combinatorial bookkeeping amortised away.

use std::sync::OnceLock;

/// The largest slot-count the packed-table caches cover, plus one. A slot list
/// is built in a [`SlotBuf`] (capacity 8), so `m ≤ 8` always holds on every
/// path that reaches these walkers; the caches are indexed directly by `m`.
const MAX_SLOTS: usize = 8;

/// Walk the Leibniz product rule for an output of `m` differentiation slots.
///
/// `D_S(ab) = Σ_{T ⊆ S} D_T(a) · D_{S∖T}(b)`, summed over every subset `T`
/// of the `m` positions. `left(t)` / `right(c)` receive the position lists of
/// the chosen subset and its complement and must return the corresponding
/// derivative of the two factors. Returns the summed output coefficient.
///
/// `m` is small (≤ 4 for the tower, ≤ 8 for the directional jet); the
/// `2^m` subset walk is the exact rule, not a truncation.
///
/// # Performance
///
/// The `(subset, complement)` index split for each of the `2^m` subsets depends
/// only on `m`, so it is computed once per `m` (see [`subset_split_table`]) and
/// cached as packed bit lists. Per call this loop only maps those cached indices
/// through `positions` and invokes the two closures — no per-bit branch, no
/// per-subset structure rebuild. BIT-IDENTICAL to the former branch walker:
/// subsets are enumerated in the same `sub = 0..2^m` order (subset bit `b` ↔
/// position `b`), the subset/complement position lists are in the same
/// increasing-bit order, and the running `total` starts at `0.0` so a
/// signed-zero leading product collapses to `+0.0` identically.
#[inline]
pub(crate) fn leibniz_product<L, R>(positions: &[usize], mut left: L, mut right: R) -> f64
where
    L: FnMut(&[usize]) -> f64,
    R: FnMut(&[usize]) -> f64,
{
    let m = positions.len();
    assert!(
        m <= MAX_SLOTS,
        "too many differentiation slots for subset enumeration"
    );
    let table = subset_split_table(m);
    let mut subset = SlotBuf::new();
    let mut complement = SlotBuf::new();
    let mut total = 0.0;
    for split in table {
        subset.len = 0;
        for &bit in split.subset.as_slice() {
            subset.push(positions[bit]);
        }
        complement.len = 0;
        for &bit in split.complement.as_slice() {
            complement.push(positions[bit]);
        }
        total += left(subset.as_slice()) * right(complement.as_slice());
    }
    total
}

/// Walk the multivariate Faà di Bruno rule for an output of `m` slots.
///
/// `D(f∘u) = Σ_{partitions π of the m slots} f^{(|π|)}(u) · Π_{B ∈ π} D_B(u)`.
/// `derivs[r]` is `f^{(r)}` at the inner value; `inner(block)` returns the
/// derivative of the inner expression for a block's position list. Returns the
/// summed output coefficient. Blocks of order ≥ `derivs.len()` are skipped
/// (their `f^{(r)}` is beyond the truncation), matching both legacy paths.
///
/// # Performance
///
/// The set partitions of `m` slots depend only on `m`, so the full partition
/// list is built once per `m` (see [`partition_table`]) and cached as packed
/// per-block bitmasks. This walk iterates that flat table directly — no
/// recursive enumeration, no `&mut dyn FnMut` leaf dispatch, no per-block
/// [`SlotBuf`] churn beyond translating a block's bitmask to labelled positions.
/// BIT-IDENTICAL to the former recursive walker: partitions are emitted in the
/// same order, each partition's blocks are in the same first-appearance order,
/// each block's positions are in the same increasing order, every block product
/// is left-associated from `derivs[order]`, and the channel `total` starts at
/// `0.0` (signed-zero products collapse to `+0.0` identically).
#[inline]
pub fn faa_di_bruno<F>(positions: &[usize], derivs: &[f64], mut inner: F) -> f64
where
    F: FnMut(&[usize]) -> f64,
{
    let m = positions.len();
    if m == 0 {
        return derivs[0];
    }
    let table = partition_table(m);
    let mut total = 0.0;
    let mut labelled = SlotBuf::new();
    for part in table {
        let order = part.n_blocks as usize;
        if order >= derivs.len() {
            continue;
        }
        let mut prod = derivs[order];
        for &block_mask in &part.blocks[..order] {
            // Translate the block's element-index bitmask to axis labels for
            // `inner`, in increasing element order (the walker's block order).
            labelled.len = 0;
            let mut bits = block_mask;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                labelled.push(positions[bit]);
                bits &= bits - 1;
            }
            prod *= inner(labelled.as_slice());
        }
        total += prod;
    }
    total
}

/// Layout hook for jets that share the Faà di Bruno unary-composition kernel.
///
/// `DERIVS` is the length of the unary derivative stack: `5` for fourth-order
/// jets (`[f, f′, f″, f‴, f⁗]`) and `3` for second-order jets. Implementors own
/// how slot lists map to their storage; the kernel owns the set-partition rule.
pub(crate) trait JetAlgebra<const DERIVS: usize>: Sized {
    /// Read the derivative for a slot list. An empty list is the value channel.
    fn derivative(&self, positions: &[usize]) -> f64;

    /// Build a jet with every stored derivative filled by `f(positions)`.
    fn map_derivatives<F>(&self, f: F) -> Self
    where
        F: FnMut(&[usize]) -> f64;

    /// Exact multivariate Faà di Bruno composition.
    fn compose_unary(&self, derivs: [f64; DERIVS]) -> Self {
        compose_unary_kernel(self, derivs)
    }
}

/// The single unary-composition kernel shared by tower and bitmask jets.
#[inline]
pub(crate) fn compose_unary_kernel<J, const DERIVS: usize>(inner: &J, derivs: [f64; DERIVS]) -> J
where
    J: JetAlgebra<DERIVS>,
{
    inner.map_derivatives(|positions| {
        faa_di_bruno(positions, &derivs, |block| inner.derivative(block))
    })
}

/// A tiny inline stack of slot indices — no heap traffic on the hot per-row
/// path. Capacity (8) covers the deepest tower (order 4) and the directional
/// jet's eight-bit masks.
#[derive(Clone, Copy)]
pub(crate) struct SlotBuf {
    data: [usize; 8],
    len: usize,
}

impl SlotBuf {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            data: [0; 8],
            len: 0,
        }
    }
    #[inline]
    fn push(&mut self, v: usize) {
        self.data[self.len] = v;
        self.len += 1;
    }
    /// Append a slot index. Public to the crate so other jet layouts (the
    /// bitmask [`crate::jet_partitions`]) can build a slot list to hand the
    /// shared walkers.
    #[inline]
    pub(crate) fn push_slot(&mut self, v: usize) {
        self.push(v);
    }
    #[inline]
    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.data[..self.len]
    }
}

// ───────────────────────── precomputed combinatorial tables ─────────────────
//
// Both tables are keyed by slot-count `m ∈ 0..=MAX_SLOTS` and built lazily on
// first use of that `m`. The build enumerations are the SAME recursions / loops
// the walkers formerly ran inline, so the cached emission order is identical and
// the walkers stay `to_bits`-exact. After the first call for a given `m` the hot
// path does zero structural work.

/// One subset of an `m`-slot Leibniz product: the bit indices `0..m` that fall
/// in the subset `T`, and those in its complement `S∖T`, each in increasing
/// order. Mirrors the former per-bit branch (`sub & (1<<bit) != 0`).
#[derive(Clone)]
struct SubsetSplit {
    subset: SlotBuf,
    complement: SlotBuf,
}

/// One set-partition of `m` slots: each block stored as a bitmask over the
/// element indices `0..m`, in the blocks' first-appearance order (the order the
/// former recursion appended them). `n_blocks` is the partition's order `|π|`.
#[derive(Clone, Copy)]
struct PackedPartition {
    blocks: [u8; MAX_SLOTS],
    n_blocks: u8,
}

static SUBSET_TABLES: [OnceLock<Vec<SubsetSplit>>; MAX_SLOTS + 1] =
    [const { OnceLock::new() }; MAX_SLOTS + 1];
static PARTITION_TABLES: [OnceLock<Vec<PackedPartition>>; MAX_SLOTS + 1] =
    [const { OnceLock::new() }; MAX_SLOTS + 1];

/// The cached `(subset, complement)` index splits for `m` slots, in the former
/// `sub = 0..2^m` enumeration order (subset bit `b` ↔ position `b`).
#[inline]
fn subset_split_table(m: usize) -> &'static [SubsetSplit] {
    SUBSET_TABLES[m].get_or_init(|| {
        let mut out = Vec::with_capacity(1usize << m);
        for sub in 0u32..(1u32 << m) {
            let mut subset = SlotBuf::new();
            let mut complement = SlotBuf::new();
            for bit in 0..m {
                if sub & (1u32 << bit) != 0 {
                    subset.push(bit);
                } else {
                    complement.push(bit);
                }
            }
            out.push(SubsetSplit { subset, complement });
        }
        out
    })
}

/// The cached set-partition list for `m` slots, in the former recursive
/// "assign each element to an existing or new block" emission order.
#[inline]
fn partition_table(m: usize) -> &'static [PackedPartition] {
    PARTITION_TABLES[m].get_or_init(|| {
        let mut out = Vec::new();
        let mut blocks = [0u8; MAX_SLOTS];
        build_partitions(0, m, &mut blocks, 0, &mut out);
        out
    })
}

/// Enumerate the set-partitions of `0..m` exactly as the former `recurse` did:
/// element `elem` is placed into each existing block (in block order) before a
/// fresh block is opened with it alone. Records each completed partition's
/// block bitmasks in first-appearance order. Runs once per `m`.
fn build_partitions(
    elem: usize,
    m: usize,
    blocks: &mut [u8; MAX_SLOTS],
    n_blocks: usize,
    out: &mut Vec<PackedPartition>,
) {
    if elem == m {
        let mut packed = PackedPartition {
            blocks: [0u8; MAX_SLOTS],
            n_blocks: n_blocks as u8,
        };
        packed.blocks[..n_blocks].copy_from_slice(&blocks[..n_blocks]);
        out.push(packed);
        return;
    }
    let bit = 1u8 << elem;
    // Place `elem` into each existing block.
    for b in 0..n_blocks {
        blocks[b] |= bit;
        build_partitions(elem + 1, m, blocks, n_blocks, out);
        blocks[b] &= !bit;
    }
    // Or open a new block with `elem` alone.
    blocks[n_blocks] = bit;
    build_partitions(elem + 1, m, blocks, n_blocks + 1, out);
}

#[cfg(test)]
mod tests {
    use crate::jet_partitions::MultiDirJet;
    use crate::jet_tower::Tower4;

    /// Bit-exact equivalence proof: evaluate the SAME polynomial-plus-unary
    /// composition on both jet layouts and assert every shared derivative
    /// coefficient is identical to the last bit. Because both layouts now
    /// route through this module's [`leibniz_product`] / [`faa_di_bruno`]
    /// walkers, the equality is a statement that the two *data structures*
    /// expose the same single arithmetic kernel — the #1151 guarantee.
    ///
    /// Program (K=2 directions / primaries, seeded as variables `x = p0`,
    /// `z = p1`):  `g = exp(x * z + x)` then `f = ln(g + 2) * g`.
    /// Both `mul` and `compose_unary` (exp, ln) are exercised.
    #[test]
    fn tower_and_dirjet_agree_bit_exact() {
        let x = 0.37_f64;
        let z = -0.81_f64;

        // ── Tower4<2> path ──
        let tx = Tower4::<2>::variable(x, 0);
        let tz = Tower4::<2>::variable(z, 1);
        let tg = (tx * tz + tx).exp();
        let tf = (tg + 2.0).ln() * tg;

        // ── MultiDirJet (2 directions) path ──
        let jx = MultiDirJet::linear(2, x, &[1.0, 0.0]);
        let jz = MultiDirJet::linear(2, z, &[0.0, 1.0]);
        let jg = exp_dirjet(&jx.mul(&jz).add(&jx));
        let jf = ln_dirjet(&jg.add(&MultiDirJet::constant(2, 2.0))).mul(&jg);

        // The directional jet carries coefficients for masks
        //   0b00=value, 0b01=∂x, 0b10=∂z, 0b11=∂x∂z.
        // The tower carries the same derivatives as tensor entries:
        //   v, g[0], g[1], h[0][1].
        assert_eq!(jf.coeff(0b00), tf.v, "value");
        assert_eq!(jf.coeff(0b01), tf.g[0], "∂x");
        assert_eq!(jf.coeff(0b10), tf.g[1], "∂z");
        assert_eq!(jf.coeff(0b11), tf.h[0][1], "∂x∂z");
        // Symmetry of the tower's mixed second partial is also bit-exact.
        assert_eq!(tf.h[0][1], tf.h[1][0], "tower mixed-partial symmetry");
    }

    #[test]
    fn tower_contractions_match_dirjet_directional_coefficients() {
        const K: usize = 3;
        let p = [0.37_f64, -0.42_f64, 0.19_f64];
        let q = [0.25_f64, -0.7_f64, 1.3_f64];
        let u = [-0.4_f64, 0.9_f64, 0.15_f64];
        let w = [1.1_f64, -0.2_f64, 0.6_f64];

        let tower = nonlinear_tower_program(p);
        let third = tower.third_contracted(&q);
        let fourth = tower.fourth_contracted(&u, &w);

        for a in 0..K {
            for b in 0..K {
                let mut dirs3 = [[0.0; K]; 3];
                dirs3[0][a] = 1.0;
                dirs3[1][b] = 1.0;
                dirs3[2] = q;
                let jet3 = nonlinear_dirjet_program(p, &dirs3);
                assert_close(
                    jet3.coeff(jet3.coeffs.len() - 1),
                    third[a][b],
                    &format!("third contraction ({a},{b})"),
                );

                let mut dirs4 = [[0.0; K]; 4];
                dirs4[0][a] = 1.0;
                dirs4[1][b] = 1.0;
                dirs4[2] = u;
                dirs4[3] = w;
                let jet4 = nonlinear_dirjet_program(p, &dirs4);
                assert_close(
                    jet4.coeff(jet4.coeffs.len() - 1),
                    fourth[a][b],
                    &format!("fourth contraction ({a},{b})"),
                );
            }
        }
    }

    fn nonlinear_tower_program(p: [f64; 3]) -> Tower4<3> {
        let x = Tower4::<3>::variable(p[0], 0);
        let y = Tower4::<3>::variable(p[1], 1);
        let z = Tower4::<3>::variable(p[2], 2);
        let eta = x * y + x * z + z * 0.7;
        let g = eta.exp();
        (g + 2.0).ln() * g
    }

    fn nonlinear_dirjet_program(p: [f64; 3], dirs: &[[f64; 3]]) -> MultiDirJet {
        let n_dirs = dirs.len();
        let x = MultiDirJet::linear(n_dirs, p[0], &direction_components(dirs, 0));
        let y = MultiDirJet::linear(n_dirs, p[1], &direction_components(dirs, 1));
        let z = MultiDirJet::linear(n_dirs, p[2], &direction_components(dirs, 2));
        let eta = x.mul(&y).add(&x.mul(&z)).add(&z.scale(0.7));
        let g = exp_dirjet(&eta);
        ln_dirjet(&g.add(&MultiDirJet::constant(n_dirs, 2.0))).mul(&g)
    }

    fn direction_components(dirs: &[[f64; 3]], axis: usize) -> Vec<f64> {
        dirs.iter().map(|dir| dir[axis]).collect()
    }

    fn assert_close(got: f64, want: f64, label: &str) {
        let tol = 1.0e-12 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{label}: got={got:.17e}, want={want:.17e}, diff={:.3e}, tol={tol:.3e}",
            (got - want).abs()
        );
    }

    fn exp_dirjet(j: &MultiDirJet) -> MultiDirJet {
        let e = j.coeff(0).exp();
        j.compose_unary([e, e, e, e, e])
    }

    fn ln_dirjet(j: &MultiDirJet) -> MultiDirJet {
        let u = j.coeff(0);
        let r = 1.0 / u;
        j.compose_unary([u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r])
    }
}
