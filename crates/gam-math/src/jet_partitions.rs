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
//!   here from the *multilinear powers* of the non-constant part rather than a
//!   direct set-partition sum. Let `v` be the non-constant part of `self`
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
//!   The powers themselves are built by the **pointed (lowest-set-bit)
//!   recurrence**, not by full subset convolutions. Write `ℓ` for the lowest
//!   set bit of `mask`. In any partition of `mask` exactly one block owns `ℓ`,
//!   and the `k` blocks of an ordered `k`-tuple are interchangeable, so pinning
//!   the outer block to the one containing `ℓ` counts each partition once
//!   instead of `k` times:
//!
//!   ```text
//!   v^{⊛k}[mask] = k · Σ_{B ⊆ mask, ℓ ∈ B} v[B] · v^{⊛(k-1)}[mask \ B]
//!   ```
//!
//!   This is an exact identity (`k = 2, 3, 4` reproduce `v²`, `v³`, `v⁴` to
//!   roundoff against a brute-force partition sum, gated in `tests`), and it is
//!   what makes the schedule cheap at small `K`: the complements `mask \ B`
//!   range over submasks of `mask ^ ℓ` rather than of `mask`, and the surviving
//!   term set shrinks with `k` because `v^{⊛(k-1)}` vanishes below popcount
//!   `k-1`. All three powers therefore share **one** descending submask walk
//!   whose `k = 3` and `k = 4` chains are popcount-suffixes of the `k = 2`
//!   chain — one walk, one `v[B]` load, and three independent Dot2 chains to
//!   interleave.
//!
//!   Each accumulation is a compensated dot product (Ogita–Rump–Oishi Dot2,
//!   FMA-split products + TwoSum carry) so the result is computed in ~double
//!   the working precision and the rounding of `v²` cannot compound through
//!   `v³`/`v⁴`; the integer multiplicity `k` is applied with its own FMA split
//!   so it costs one rounding rather than discarding the compensated tail; the
//!   final per-mask combine is Neumaier-compensated and `wide::f64x4`-vectorised;
//!   and the whole call runs on reused thread-local scratch with no per-call
//!   heap traffic.
//!
//! ### What this schedule costs
//!
//! Everything below is recomputed from the enumeration itself by
//! `compose_unary_work_model_matches_the_closed_form`, which replays the walk and
//! counts its steps. A counted model cannot drift the way a prose factor did:
//! this header once claimed "~3× fewer FLOPs than the partition gather" for a
//! schedule that in fact cost ~9× as much at the `K` production runs at.
//!
//! * the pointed recurrence walks `Σ_{p ≥ 2} C(K,p)·Σ_{k=2..4, k ≤ p} Σ_{j ≥ k-1}
//!   C(p-1,j)` terms, i.e. `Θ(3^K)`, each a compensated Dot2 (10 flops), plus a
//!   5-flop multiplicity epilogue per power per mask;
//! * the partition gather walks `Σ_p C(K,p)·B_{≤4}(p)` terms, i.e. `Θ(5^K/4!)`,
//!   each `|π|` plain multiplies and an add. (That count is a *lower bound* on
//!   the gather: it omits the `2^p` per-mask index remap the gather also needs,
//!   so every comparison below is stated against the gather at its best.)
//!
//! ```text
//!   K                        2     3     4     6      8      9     10     12
//!   pointed terms            1     7    34   534   6514  21589  69886  696810
//!   gather terms             5    15    52   855  18002  86472 422005 10306752
//!   pointed/gather flops  2.5x  3.5x  3.3x  2.0x  0.91x  0.59x  0.37x   0.15x
//! ```
//!
//! Two facts worth carrying:
//!
//! * The three-full-subset-convolution schedule this replaced walked **exactly
//!   4×** as many terms at every `K ≤ 4` (136 against 34 at `K = 4`) — one factor
//!   of 2 from pinning the block that owns `ℓ`, one from walking submasks of
//!   `mask ^ ℓ` instead of `mask`. Its flop crossover against the gather sat at
//!   `K = 10`; the pointed recurrence moves it to `K = 8`.
//! * At `K = 4` the schedule walks 34 terms against the gather's 52. It does
//!   strictly *less* combinatorial work than the partition sum it replaced —
//!   at every `K`, not just past a crossover — and the residual 3.3× flop ratio
//!   at `K = 4` is entirely the Dot2 compensation: 10 flops a term against ~3.
//!   That is the accuracy the double-double gate pins, and the only reason to
//!   prefer this schedule; it is not free, and a reader sizing a new call site
//!   should plan for it.
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use wide::f64x4;

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
    /// Bit-identical to the shared `crate::jet_algebra::leibniz_product` walker
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
        // Both operands carry the same direction set, so `b` is `count` long too.
        // With that established once, every `a[sub]`/`b[mask ^ sub]` below is
        // provably in bounds (`sub, mask ^ sub ⊆ mask < count`), so the inner
        // submask walk can drop its per-load bounds checks.
        assert_eq!(
            b.len(),
            count,
            "MultiDirJet::mul operands must share n_dirs"
        );
        let mut out = vec![0.0; count];
        for (mask, slot) in out.iter_mut().enumerate() {
            // Walk every submask of `mask` in ascending numeric order — the same
            // order `leibniz_product` accumulates — via the classic gap-fill
            // increment `next = ((sub | !mask) + 1) & mask`.
            let mut acc = 0.0;
            let mut sub = 0usize;
            // SAFETY: `sub ⊆ mask < count` and `mask ^ sub ⊆ mask < count`, and
            // both `a` and `b` are `count` long (asserted above).
            unsafe {
                loop {
                    acc += *a.get_unchecked(sub) * *b.get_unchecked(mask ^ sub);
                    if sub == mask {
                        break;
                    }
                    sub = (sub | !mask).wrapping_add(1) & mask;
                }
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
        let mut out = vec![0.0; count];
        COMPOSE_SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.resize(4 * count, 0.0);
            compose_unary_coefficients_into(&self.coeffs, derivs, buf.as_mut_slice(), &mut out);
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

#[inline]
fn compose_unary_coefficients_into(
    coefficients: &[f64],
    derivs: [f64; DERIVS],
    scratch: &mut [f64],
    out: &mut [f64],
) {
    let count = coefficients.len();
    assert!(count > 1 && count.is_power_of_two());
    assert!(scratch.len() == 4 * count && out.len() == count);
    let (vbuf, tail) = scratch.split_at_mut(count);
    let (p2, tail) = tail.split_at_mut(count);
    let (p3, p4) = tail.split_at_mut(count);

    // v is the non-constant part of the input. The k=0 Taylor term owns the
    // constant coefficient, so the zero mask must not enter any power.
    vbuf.copy_from_slice(coefficients);
    vbuf[0] = 0.0;

    // The three multilinear powers, by the pointed recurrence (module header).
    multilinear_powers_into(vbuf, p2, p3, p4);
    // `1/k!` undoes the ordered-tuple overcount of each k-fold subset power
    // relative to the unordered set-partition sum.
    let coefficients_by_order = [
        derivs[1],
        derivs[2] * 0.5,
        derivs[3] * (1.0 / 6.0),
        derivs[4] * (1.0 / 24.0),
    ];
    combine_powers(vbuf, p2, p3, p4, coefficients_by_order, out);
    out[0] = derivs[0];
}

/// Branchless TwoSum: returns `(s, e)` with `s = fl(a+b)` and `a+b = s+e`
/// exactly (Knuth/Møller). Used by the compensated power recurrence and combine.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// One step of an Ogita–Rump–Oishi Dot2: accumulate `x·y` into `(s, c)` so that
/// `s + c` carries the running sum in ~twice the working precision. The product
/// is split into head plus exact FMA error, and the addition's rounding error is
/// recovered by TwoSum, so neither the product nor the sum silently drops bits.
#[inline(always)]
fn dot2_step(s: &mut f64, c: &mut f64, x: f64, y: f64) {
    let prod = x * y;
    let prod_err = x.mul_add(y, -prod); // exact: prod + prod_err == x*y
    let (t, sum_err) = two_sum(*s, prod);
    *s = t;
    *c += prod_err + sum_err;
}

/// `k·(s + c)` for a small integer multiplicity `k`, with `k·s` split into head
/// plus exact FMA error so the multiplicity costs one final rounding rather than
/// discarding the compensated tail. For `k ∈ {2, 4}` the split is identically
/// zero (both are exact scalings); `k = 3` is the case that needs it.
#[inline(always)]
fn scaled_compensated(k: f64, s: f64, c: f64) -> f64 {
    let hi = k * s;
    let lo = k.mul_add(s, -hi); // exact: hi + lo == k*s
    hi + (lo + k * c)
}

/// The multilinear powers `v^{⊛2}`, `v^{⊛3}`, `v^{⊛4}` of the non-constant part
/// `v`, by the **pointed (lowest-set-bit) recurrence** derived in the module
/// header:
///
/// ```text
/// v^{⊛k}[mask] = k · Σ_{t ⊊ mask, ℓ ∉ t} v[mask \ t] · v^{⊛(k-1)}[t]
/// ```
///
/// with `ℓ` the lowest set bit of `mask`. Pinning the block that owns `ℓ` counts
/// each partition once instead of `k` times, and the surviving `t` range over
/// submasks of `mask ^ ℓ` rather than of `mask` — together an exactly 4× shorter
/// walk than the three full subset convolutions this replaced, at every
/// `K ≤ 4` (see `compose_unary_work_model_matches_the_closed_form`).
///
/// All three powers share **one** descending walk over the submasks of
/// `mask ^ ℓ`, because their term sets are nested: `v^{⊛(k-1)}[t]` vanishes
/// below `popcount(t) = k - 1`, so the `k = 3` and `k = 4` chains are the
/// `popcount ≥ 2` and `popcount ≥ 3` suffixes of the `k = 2` chain. Sharing the
/// walk also shares the `v[mask \ t]` load and gives three independent Dot2
/// dependency chains to interleave, which is what the old kernel's four-way
/// unroll was buying separately.
///
/// Every accumulation is a compensated Dot2, so the rounding of `v²` cannot
/// compound through `v³`/`v⁴`. Masks below popcount `k` are left at zero: the
/// `k`-fold multilinear power vanishes there, so the prune is exact.
#[inline]
fn multilinear_powers_into(v: &[f64], p2: &mut [f64], p3: &mut [f64], p4: &mut [f64]) {
    let count = v.len();
    // SAFETY precondition for the `get_unchecked` loads below, pinned once per
    // call (negligible next to the walk): all four buffers are `count` long.
    // Every index read is either `t` or `mask ^ t` for `t ⊆ mask < count`, and
    // both are submasks of `mask`, hence `< count`. The per-load bounds checks
    // LLVM cannot elide (the indices are data-dependent) are a real cost across
    // the exponential walk, and eliding them measured ~20% on the kernel this
    // replaced.
    assert!(p2.len() == count && p3.len() == count && p4.len() == count);
    if count > 0 {
        p2[0] = 0.0;
        p3[0] = 0.0;
        p4[0] = 0.0;
    }
    for mask in 1..count {
        // `v^{⊛k}` vanishes below popcount k, so a popcount-1 mask is all-zero
        // in every power and never enters a walk.
        let lowest = mask & mask.wrapping_neg();
        let rest = mask ^ lowest;
        if rest == 0 {
            p2[mask] = 0.0;
            p3[mask] = 0.0;
            p4[mask] = 0.0;
            continue;
        }
        let (mut s2, mut c2) = (0.0f64, 0.0f64);
        let (mut s3, mut c3) = (0.0f64, 0.0f64);
        let (mut s4, mut c4) = (0.0f64, 0.0f64);
        // Descending submask walk `t = (t - 1) & rest` over the NONZERO submasks
        // of `rest` (the classic Gosper-style enumeration). `t = 0` is skipped
        // because it is the one term whose complement is the whole mask, and
        // `v^{⊛(k-1)}[0] = 0` for every `k ≥ 2`.
        let mut t = rest;
        while t != 0 {
            // SAFETY: `t ⊆ rest ⊂ mask < count` and `mask ^ t ⊆ mask < count`,
            // and all four buffers are `count` long (asserted above).
            unsafe {
                let block = *v.get_unchecked(mask ^ t);
                dot2_step(&mut s2, &mut c2, block, *v.get_unchecked(t));
                let popcount = (t as u64).count_ones();
                if popcount >= 2 {
                    dot2_step(&mut s3, &mut c3, block, *p2.get_unchecked(t));
                    if popcount >= 3 {
                        dot2_step(&mut s4, &mut c4, block, *p3.get_unchecked(t));
                    }
                }
            }
            t = (t - 1) & rest;
        }
        // The pointed recurrence's multiplicity. `v^{⊛k}[mask]` must be written
        // before the `k+1` chain of any LATER mask reads it, and `t < mask`
        // strictly for every `t ⊆ mask ^ ℓ`, so writing all three here keeps the
        // recurrence's read-before-write order across the ascending mask loop.
        p2[mask] = scaled_compensated(2.0, s2, c2);
        p3[mask] = scaled_compensated(3.0, s3, c3);
        p4[mask] = scaled_compensated(4.0, s4, c4);
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
            let big_s = s.abs().simd_ge(term.abs());
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
            (0..=n_dirs)
                .map(|m| std::rc::Rc::clone(&tables[m]))
                .collect()
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
        fn recurse(
            elem: usize,
            m: usize,
            blocks: &mut [u32; 8],
            n_blocks: usize,
            out: &mut PartTable,
        ) {
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
        let j = MultiDirJet::linear(2, 0.3, &[0.5, -0.4]).mul(&MultiDirJet::linear(
            2,
            -0.2,
            &[0.1, 0.7],
        ));
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

    /// `v^{⊛k}[mask] = k! · Σ_{π ⊢ mask, |π| = k} Π_{B ∈ π} v[B]` by direct
    /// enumeration of the set partitions of `mask` — the *definition* the pointed
    /// recurrence claims to compute.
    ///
    /// Returns `[v², v³, v⁴]` and, per mask, the forward error the pair of
    /// evaluations is jointly entitled to: this reference accumulates `n` terms
    /// naively (Wilkinson: `n·u` times the sum of the term magnitudes) after up to
    /// three roundings per product, and the compensated walk is good to ~`u`, so
    /// `(n + 4)·EPSILON·Σ|term|` bounds their difference. It is derived from the
    /// enumeration, not fitted to an observed failure.
    fn brute_force_multilinear_powers(v: &[f64]) -> ([Vec<f64>; 3], [Vec<f64>; 3]) {
        fn recurse(
            elem: usize,
            bits: &[usize],
            blocks: &mut Vec<usize>,
            v: &[f64],
            acc: &mut [f64; 5],
            acc_abs: &mut [f64; 5],
            acc_count: &mut [u32; 5],
        ) {
            if blocks.len() > 4 {
                return;
            }
            if elem == bits.len() {
                let order = blocks.len();
                if order >= 2 {
                    let product: f64 = blocks.iter().map(|&b| v[b]).product();
                    acc[order] += product;
                    acc_abs[order] += product.abs();
                    acc_count[order] += 1;
                }
                return;
            }
            let bit = 1usize << bits[elem];
            for slot in 0..blocks.len() {
                blocks[slot] |= bit;
                recurse(elem + 1, bits, blocks, v, acc, acc_abs, acc_count);
                blocks[slot] &= !bit;
            }
            blocks.push(bit);
            recurse(elem + 1, bits, blocks, v, acc, acc_abs, acc_count);
            blocks.pop();
        }
        let count = v.len();
        let mut powers = [vec![0.0; count], vec![0.0; count], vec![0.0; count]];
        let mut entitled = [vec![0.0; count], vec![0.0; count], vec![0.0; count]];
        let factorial = [1.0, 1.0, 2.0, 6.0, 24.0];
        for mask in 1..count {
            let mut bits = Vec::new();
            let mut rest = mask;
            while rest != 0 {
                bits.push(rest.trailing_zeros() as usize);
                rest &= rest - 1;
            }
            let mut acc = [0.0f64; 5];
            let mut acc_abs = [0.0f64; 5];
            let mut acc_count = [0u32; 5];
            recurse(
                0,
                &bits,
                &mut Vec::new(),
                v,
                &mut acc,
                &mut acc_abs,
                &mut acc_count,
            );
            for order in 2..=4usize {
                powers[order - 2][mask] = factorial[order] * acc[order];
                entitled[order - 2][mask] = (f64::from(acc_count[order]) + 4.0)
                    * f64::EPSILON
                    * factorial[order]
                    * acc_abs[order];
            }
        }
        (powers, entitled)
    }

    /// The pointed (lowest-set-bit) recurrence is an *identity*, not an
    /// approximation: pinning the block that owns the lowest set bit and
    /// multiplying by `k` reproduces the `k`-fold multilinear power of the
    /// brute-force set-partition definition, to the forward error the two
    /// evaluations are jointly entitled to.
    ///
    /// This is the gate on the recurrence the module header derives. The header's
    /// cost claims are only worth anything if the cheaper walk computes the same
    /// quantity, and that is what this pins.
    #[test]
    fn pointed_recurrence_reproduces_the_brute_force_multilinear_powers() {
        let mut rng = Rng(0x0be1_1ab0_1a5e_c001);
        for n_dirs in 1usize..=6 {
            for _ in 0..24 {
                let count = 1usize << n_dirs;
                let mut v: Vec<f64> = (0..count).map(|_| rng.signed(1.0)).collect();
                v[0] = 0.0;
                let mut p2 = vec![f64::NAN; count];
                let mut p3 = vec![f64::NAN; count];
                let mut p4 = vec![f64::NAN; count];
                multilinear_powers_into(&v, &mut p2, &mut p3, &mut p4);
                let (want, entitled) = brute_force_multilinear_powers(&v);
                for (order, got) in [&p2, &p3, &p4].iter().enumerate() {
                    for mask in 0..count {
                        // Graded against the forward error the two evaluations are
                        // jointly entitled to (see the reference above), which is a
                        // derived bound rather than a fitted tolerance. Where the
                        // power vanishes identically the bound is zero and the
                        // agreement must be exact.
                        let tolerance = entitled[order][mask];
                        assert!(
                            (got[mask] - want[order][mask]).abs() <= tolerance,
                            "K={n_dirs} k={} mask={mask}: pointed={:.17e} partitions={:.17e} \
                             tol={tolerance:.3e}",
                            order + 2,
                            got[mask],
                            want[order][mask]
                        );
                    }
                }
            }
        }
    }

    /// The two schedules' operation counts, recomputed from the enumeration
    /// itself. This is the machine-independent statement of what each schedule
    /// costs and where the convolution path becomes the cheaper one; the
    /// wall-clock test below can only corroborate it.
    ///
    /// It exists because the header once claimed the convolution schedule was
    /// "~3× fewer FLOPs than the per-mask partition gather" full stop, and a
    /// wall-clock test asserted a speedup from `K = 6`. Both were false of the
    /// schedule then in the file: three full subset convolutions cost ~9× the
    /// gather at the `K = 4` the production entry point uses, and did not come
    /// out ahead until `K = 10`. A counted model cannot drift the way a prose
    /// factor did.
    #[test]
    fn compose_unary_work_model_matches_the_closed_form() {
        // Dot2: mul, FMA error, TwoSum (6), two carry adds.
        const DOT2_FLOPS: u64 = 10;
        // `scaled_compensated`: k·s, its FMA error, k·c, and two adds.
        const MULTIPLICITY_FLOPS: u64 = 5;

        let binom = |n: u32, r: u32| -> u64 {
            (0..r).fold(1u64, |acc, i| acc * u64::from(n - i) / (u64::from(i) + 1))
        };

        // Replay of `multilinear_powers_into`'s enumeration — same mask loop,
        // same lowest-bit pin, same descending walk over the submasks of
        // `mask ^ lowest`, same popcount gates — counting Dot2 steps instead of
        // performing them. This is what makes the closed form below a claim about
        // the kernel rather than about itself.
        fn walked_terms(n_dirs: u32) -> u64 {
            let count = 1usize << n_dirs;
            let mut steps = 0u64;
            for mask in 1..count {
                let lowest = mask & mask.wrapping_neg();
                let rest = mask ^ lowest;
                if rest == 0 {
                    continue;
                }
                let mut t = rest;
                while t != 0 {
                    steps += 1;
                    let popcount = (t as u64).count_ones();
                    if popcount >= 2 {
                        steps += 1;
                        if popcount >= 3 {
                            steps += 1;
                        }
                    }
                    t = (t - 1) & rest;
                }
            }
            steps
        }

        // Closed form: per mask of popcount p ≥ 2 and each power k ≤ p, the
        // submasks t of `mask ^ ℓ` (a (p-1)-set) with popcount(t) ≥ k-1.
        let pointed_terms = |n_dirs: u32| -> u64 {
            (2..=n_dirs)
                .map(|p| {
                    let per_mask: u64 = (2..=4u32)
                        .filter(|k| *k <= p)
                        .map(|k| (k - 1..=p - 1).map(|j| binom(p - 1, j)).sum::<u64>())
                        .sum();
                    binom(n_dirs, p) * per_mask
                })
                .sum()
        };
        let pointed_flops = |n_dirs: u32| -> u64 {
            (2..=n_dirs)
                .map(|p| {
                    let per_mask: u64 = (2..=4u32)
                        .filter(|k| *k <= p)
                        .map(|k| (k - 1..=p - 1).map(|j| binom(p - 1, j)).sum::<u64>())
                        .sum();
                    binom(n_dirs, p) * (per_mask * DOT2_FLOPS + 3 * MULTIPLICITY_FLOPS)
                })
                .sum()
        };

        // The schedule this replaced: three full subset convolutions v², v³=v²⊛v,
        // v⁴=v²⊛v², each pruned at popcount < k, each surviving mask walking all
        // 2^popcount of its submasks.
        let full_convolution_terms = |n_dirs: u32| -> u64 {
            (2..=4u32)
                .map(|k| {
                    (k..=n_dirs)
                        .map(|p| binom(n_dirs, p) * (1u64 << p))
                        .sum::<u64>()
                })
                .sum()
        };

        // Partition gather: per mask of popcount p, every set partition of a
        // p-set into 1..=4 blocks contributes |π| multiplies and one add. This
        // omits the gather's own `2^p` per-mask index remap, so it is a lower
        // bound on the gather and every comparison below is against its best case.
        fn stirling(n: u32, blocks: u32) -> u64 {
            let (n, blocks) = (n as usize, blocks as usize);
            let mut s = vec![vec![0u64; blocks + 1]; n + 1];
            s[0][0] = 1;
            for i in 1..=n {
                for j in 1..=blocks {
                    s[i][j] = (j as u64) * s[i - 1][j] + s[i - 1][j - 1];
                }
            }
            s[n][blocks]
        }
        let gather_terms = |n_dirs: u32| -> u64 {
            1 + (1..=n_dirs)
                .map(|p| binom(n_dirs, p) * (1..=4u32).map(|b| stirling(p, b)).sum::<u64>())
                .sum::<u64>()
        };
        let gather_flops = |n_dirs: u32| -> u64 {
            1 + (1..=n_dirs)
                .map(|p| {
                    binom(n_dirs, p)
                        * (1..=4u32)
                            .map(|b| stirling(p, b) * (u64::from(b) + 1))
                            .sum::<u64>()
                })
                .sum::<u64>()
        };

        // (a) The closed form describes the walk the kernel actually performs.
        for n_dirs in 2..=12u32 {
            assert_eq!(
                walked_terms(n_dirs),
                pointed_terms(n_dirs),
                "K={n_dirs}: closed form disagrees with the replayed enumeration"
            );
        }

        // (b) Against the three full subset convolutions this replaced: exactly
        // 4× fewer terms across the whole range production runs at — one factor
        // of 2 from pinning the block that owns the lowest set bit, one from
        // walking submasks of `mask ^ ℓ` rather than of `mask`.
        for n_dirs in 2..=4u32 {
            assert_eq!(
                full_convolution_terms(n_dirs),
                4 * pointed_terms(n_dirs),
                "K={n_dirs}: the replaced schedule should be exactly 4x this one"
            );
        }
        for n_dirs in 2..=14u32 {
            assert!(
                pointed_terms(n_dirs) < full_convolution_terms(n_dirs),
                "K={n_dirs}: the pointed recurrence must never walk more terms \
                 than the full convolutions it replaced ({} vs {})",
                pointed_terms(n_dirs),
                full_convolution_terms(n_dirs)
            );
        }

        // (c) Against the partition gather: strictly fewer terms at every K,
        // including the production K = 4 (34 against 52). The combinatorial
        // deficit that made the old schedule indefensible at small K is gone.
        for n_dirs in 2..=14u32 {
            assert!(
                pointed_terms(n_dirs) < gather_terms(n_dirs),
                "K={n_dirs}: pointed schedule should walk fewer terms than the \
                 partition gather ({} vs {})",
                pointed_terms(n_dirs),
                gather_terms(n_dirs)
            );
        }
        assert_eq!((pointed_terms(4), gather_terms(4)), (34, 52));

        // (d) What remains at the production K = 4 is the compensation premium
        // and nothing else: 10 flops a term against the gather's ~3, over fewer
        // terms, for a 3.3x flop ratio. That is the price of the ~double-precision
        // accumulation the accuracy gate pins — it is bought, not free.
        let production_ratio = pointed_flops(4) as f64 / gather_flops(4) as f64;
        assert!(
            (production_ratio - 3.34).abs() < 0.05,
            "at the production K=4 the pointed schedule should cost ~3.34x the \
             partition gather's flops, got {production_ratio:.2}x"
        );

        // (e) The flop crossover, which the pointed recurrence moves from K = 10
        // (three full convolutions) to K = 8.
        for n_dirs in 2..=7u32 {
            assert!(
                pointed_flops(n_dirs) > gather_flops(n_dirs),
                "K={n_dirs}: below the crossover the compensated schedule is still \
                 the more expensive one ({} vs {})",
                pointed_flops(n_dirs),
                gather_flops(n_dirs)
            );
        }
        for n_dirs in 8..=14u32 {
            assert!(
                pointed_flops(n_dirs) < gather_flops(n_dirs),
                "K={n_dirs}: at and above the K=8 crossover the compensated \
                 schedule should also be the cheaper one ({} vs {})",
                pointed_flops(n_dirs),
                gather_flops(n_dirs)
            );
        }
    }

    /// Wall-clock corroboration of the compensated `compose_unary` schedule
    /// against the previous partition-sum implementation, at every `K`.
    ///
    /// This is corroboration, not the contract. The contract is
    /// `compose_unary_work_model_matches_the_closed_form`, which counts
    /// operations: a count is the same on every machine, and a wall clock is
    /// not. The speed cell is asserted only at `K = 12`, far past the
    /// crossover the counted model reports (`K = 8` for the pointed
    /// recurrence), where the compensated schedule does ~7x less arithmetic;
    /// below that the multiple is printed for the record. The gate opens only
    /// in the release profile (`SpeedGate::open` documents why).
    #[test]
    fn compose_unary_speedup_over_partition_sum() {
        use crate::paired_timing::{SpeedGate, paired_interleaved};

        if cfg!(debug_assertions) {
            return;
        }
        let mut gate = SpeedGate::open("COMPOSE-UNARY-932");
        let mut rng = Rng(0xfeed_face_dead_beef);
        for &n_dirs in &[2usize, 4, 6, 8, 12] {
            // The per-call cost spans ~5 orders of magnitude across this K
            // range (both schedules are exponential in K), so the sample count
            // has to shrink with K or the K=12 arm alone would run for hours.
            let n_inputs = if n_dirs >= 12 { 4usize } else { 256 };
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
            let iterations = if n_dirs >= 12 { 3usize } else { 200 };
            // One arm call composes every input once; the nudge perturbs the
            // outer derivative stack so no composition is loop-invariant.
            let timing = paired_interleaved(
                15,
                iterations,
                0x9320_C0DE ^ n_dirs as u64,
                |nudge| {
                    let mut sink = 0.0f64;
                    for (jet, derivs) in &inputs {
                        let mut derivs = *derivs;
                        derivs[0] += nudge;
                        sink += jet.compose_unary(derivs).coeffs.iter().sum::<f64>();
                    }
                    sink
                },
                |nudge| {
                    let mut sink = 0.0f64;
                    for (jet, derivs) in &inputs {
                        let mut derivs = *derivs;
                        derivs[0] += nudge;
                        sink += compose_unary_partition_reference(&jet.coeffs, derivs)
                            .iter()
                            .sum::<f64>();
                    }
                    sink
                },
            );
            if n_dirs >= 12 {
                gate.faster(
                    &format!("K={n_dirs}"),
                    &timing,
                    "compensated",
                    "partition_sum",
                );
            } else {
                eprintln!(
                    "COMPOSE-UNARY-932 K={n_dirs} {} (below the counted crossover; not gated)",
                    timing.summary("compensated", "partition_sum"),
                );
            }
        }
        gate.finish();
    }
}
