//! The single shared Faà di Bruno / Leibniz combinatorial kernel (#1151).
//!
//! Two jet representations live in this crate and historically each carried
//! its own hand-written copy of the same calculus:
//!
//! * [`super::jet_tower::Tower4`] — full dense derivative tensors
//!   (`v`, `g`, `h`, `t3`, `t4`) in `K` primary variables, with the
//!   Leibniz product and Faà di Bruno composition written out term-by-term
//!   per derivative order.
//! * [`super::jet_partitions::MultiDirJet`] — bitmask-coefficient jet over
//!   distinct seeded directions, with the same two rules written as general
//!   submask / set-partition loops.
//!
//! The DATA layouts are legitimately different (a complete small-`K` tower
//! vs. a handful of directions of a large-`K` expression) and stay separate.
//! What is identical is the *combinatorics*: for a group of differentiation
//! slots, the Leibniz rule sums over subsets of those slots and the Faà di
//! Bruno rule sums over their set-partitions. This module owns that
//! combinatorics ONCE, as layout-agnostic walkers parameterised by closures
//! that read each representation's own derivative for a slot-group. Both
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

/// Walk the Leibniz product rule for an output of `m` differentiation slots.
///
/// `D_S(ab) = Σ_{T ⊆ S} D_T(a) · D_{S∖T}(b)`, summed over every subset `T`
/// of the `m` positions. `left(t)` / `right(c)` receive the position lists of
/// the chosen subset and its complement and must return the corresponding
/// derivative of the two factors. Returns the summed output coefficient.
///
/// `m` is small (≤ 4 for the tower, ≤ 8 for the directional jet); the
/// `2^m` subset walk is the exact rule, not a truncation.
#[inline]
pub(crate) fn leibniz_product<L, R>(positions: &[usize], mut left: L, mut right: R) -> f64
where
    L: FnMut(&[usize]) -> f64,
    R: FnMut(&[usize]) -> f64,
{
    let m = positions.len();
    debug_assert!(m <= usize::BITS as usize);
    let mut subset = SlotBuf::new();
    let mut complement = SlotBuf::new();
    let mut total = 0.0;
    for sub in 0u32..(1u32 << m) {
        subset.clear();
        complement.clear();
        for (bit, &pos) in positions.iter().enumerate() {
            if sub & (1u32 << bit) != 0 {
                subset.push(pos);
            } else {
                complement.push(pos);
            }
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
#[inline]
pub(crate) fn faa_di_bruno<F>(positions: &[usize], derivs: &[f64], mut inner: F) -> f64
where
    F: FnMut(&[usize]) -> f64,
{
    let m = positions.len();
    if m == 0 {
        return derivs[0];
    }
    let mut total = 0.0;
    for_each_partition(m, &mut |blocks: &[SlotBuf]| {
        let order = blocks.len();
        if order >= derivs.len() {
            return;
        }
        let mut prod = derivs[order];
        for block in blocks {
            // Translate block positions to their axis labels for `inner`.
            let mut labelled = SlotBuf::new();
            for &p in block.as_slice() {
                labelled.push(positions[p]);
            }
            prod *= inner(labelled.as_slice());
        }
        total += prod;
    });
    total
}

/// Number of differentiation orders the kernel supports (value + 4 derivs).
pub(crate) const MAX_ORDER: usize = 4;

/// A tiny inline stack of slot indices — no heap traffic on the hot per-row
/// path. Capacity is `MAX_ORDER` (the deepest tower) plus headroom for the
/// directional jet's eight-bit masks.
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
    fn clear(&mut self) {
        self.len = 0;
    }
    #[inline]
    fn push(&mut self, v: usize) {
        self.data[self.len] = v;
        self.len += 1;
    }
    #[inline]
    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.data[..self.len]
    }
}

/// Invoke `f` once per set-partition of positions `0..m`, passing the blocks
/// as slot lists. Recursive "assign each element to an existing or new block"
/// enumeration — allocation-free via the fixed-capacity [`SlotBuf`].
fn for_each_partition(m: usize, f: &mut dyn FnMut(&[SlotBuf])) {
    let mut blocks: [SlotBuf; 8] = [SlotBuf::new(); 8];
    recurse(0, m, &mut blocks, 0, f);
}

fn recurse(
    elem: usize,
    m: usize,
    blocks: &mut [SlotBuf; 8],
    n_blocks: usize,
    f: &mut dyn FnMut(&[SlotBuf]),
) {
    if elem == m {
        f(&blocks[..n_blocks]);
        return;
    }
    // Place `elem` into each existing block.
    for b in 0..n_blocks {
        blocks[b].push(elem);
        recurse(elem + 1, m, blocks, n_blocks, f);
        blocks[b].len -= 1;
    }
    // Or open a new block with `elem` alone.
    blocks[n_blocks].clear();
    blocks[n_blocks].push(elem);
    recurse(elem + 1, m, blocks, n_blocks + 1, f);
}

#[cfg(test)]
mod tests {
    use super::super::jet_partitions::MultiDirJet;
    use super::super::jet_tower::Tower4;

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
