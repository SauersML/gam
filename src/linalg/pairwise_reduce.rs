//! Deterministic, bit-reproducible pairwise-tree reduction.
//!
//! This module provides a fixed-shape chunked pairwise (a.k.a. cascade /
//! divide-and-conquer) reduction primitive. Unlike a naive sequential fold,
//! whose floating-point rounding error grows with the number of terms and
//! whose result depends on the *order* in which terms are visited, a pairwise
//! tree reduces error to `O(log n)` and — crucially for this codebase — has a
//! reduction structure that is a **pure function of the input length**, not of
//! the values, of any thread scheduling, or of the platform.
//!
//! It is the load-bearing accumulation primitive for streaming Fisher-mass
//! accumulation (#973): the streaming driver feeds successive fixed-size chunks
//! into a running pairwise tree and is guaranteed a bit-identical result no
//! matter how the stream is sliced into chunks, as long as the underlying
//! sequence of summands (in order) is the same.
//!
//! # Determinism contract
//!
//! For a fixed combine operation `combine` and identity `identity` (and for
//! [`pairwise_sum`], for ordinary IEEE-754 `f64` addition):
//!
//! 1. **Pure function of (ordered) input.** The result is a deterministic
//!    function of the ordered sequence of inputs alone. Reducing the same slice
//!    in the same order twice yields a **bit-identical** result
//!    (`to_bits()` equality). No clocks, randomness, thread counts, global
//!    state, or memory addresses participate.
//!
//! 2. **Association order is a pure function of length.** The shape of the
//!    reduction tree — i.e. *which* elements are combined with *which*, and in
//!    *what association order* — depends only on the number of elements
//!    `n`, never on the element values. Two inputs of equal length are
//!    associated identically.
//!
//! 3. **Chunking/streaming invariance.** Feeding a sequence to the streaming
//!    entry points ([`StreamingPairwise`], [`pairwise_reduce_chunked`]) in any
//!    chunking — including one element at a time, or all at once — produces a
//!    result bit-identical to reducing the whole concatenated sequence with
//!    [`pairwise_reduce`]. The tree shape is determined by the total element
//!    index, not by chunk boundaries.
//!
//! Note that the contract is about *reproducibility and order-independence of
//! the association tree*, not about commutativity of `combine`. The *order* of
//! the summands still matters (floating-point addition is not associative);
//! what is guaranteed is that a given ordered input always associates the same
//! way.
//!
//! # Tree shape
//!
//! The base case is a contiguous run of at most [`BASE_CHUNK`] elements, summed
//! left-to-right (sequential within the small block, which bounds per-block
//! error to `O(BASE_CHUNK)`). Above the base case the range `[lo, hi)` is split
//! at a deterministic midpoint and the two halves are reduced recursively, then
//! combined. The split point is chosen so that the *left* subtree always holds
//! a number of elements that is the largest power-of-two multiple of
//! [`BASE_CHUNK`] strictly less than the length. This makes the tree shape a
//! pure, stable function of length and keeps it balanced.

/// Base-case block size for the pairwise tree.
///
/// Runs of at most this many elements are summed sequentially (left to right);
/// larger ranges are split into a balanced binary tree of such blocks. The
/// value is a fixed compile-time constant so that the tree shape — and hence
/// the exact floating-point result — never depends on tuning, platform, or
/// runtime conditions. 128 keeps the base block in cache while bounding the
/// sequential portion of the error to a small constant.
pub const BASE_CHUNK: usize = 128;

/// Largest power-of-two multiple of [`BASE_CHUNK`] that is strictly less than
/// `len`, for `len > BASE_CHUNK`. This is the size of the left subtree.
///
/// The split is a pure function of `len`: it does not look at any value. By
/// pinning the left subtree to a power-of-two block count we obtain a stable,
/// balanced, length-only tree shape.
#[inline]
const fn left_split(len: usize) -> usize {
    assert!(
        len > BASE_CHUNK,
        "left_split: caller must guarantee len > BASE_CHUNK"
    );
    // Number of whole base blocks needed to cover `len`, rounded down to the
    // span that fits in the left subtree. We want the largest `k = BASE_CHUNK *
    // 2^p` with `k < len`.
    let mut k = BASE_CHUNK;
    // Double while the doubled span still leaves at least one element for the
    // right subtree (i.e. stays strictly below `len`).
    while k.saturating_mul(2) < len {
        k = k.saturating_mul(2);
    }
    k
}

/// Reduce a contiguous run sequentially, left to right, starting from `acc`.
#[inline]
fn reduce_block<T, F>(acc: T, items: &[T], combine: &F) -> T
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let mut out = acc;
    for &x in items {
        out = combine(out, x);
    }
    out
}

/// Recursively reduce `items` over a deterministic, length-only pairwise tree.
///
/// `combine` must be a deterministic binary operation; `identity` is its
/// neutral element, returned for an empty slice. The association order is fixed
/// by [`left_split`] / [`BASE_CHUNK`] and never depends on the values.
///
/// This is the generic, monoid-style core. See [`pairwise_sum`] for the `f64`
/// addition specialization.
pub fn pairwise_reduce<T, F>(items: &[T], combine: F, identity: T) -> T
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    reduce_range(items, &combine, identity)
}

/// Internal recursion: reduce `items` (a contiguous range) using `combine`.
fn reduce_range<T, F>(items: &[T], combine: &F, identity: T) -> T
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let len = items.len();
    if len == 0 {
        return identity;
    }
    if len <= BASE_CHUNK {
        // Sequential base block, seeded from the first element so that the
        // identity does not perturb the floating-point result.
        return reduce_block(items[0], &items[1..], combine);
    }
    let mid = left_split(len);
    let left = reduce_range(&items[..mid], combine, identity);
    let right = reduce_range(&items[mid..], combine, identity);
    combine(left, right)
}

/// Deterministic, bit-reproducible pairwise sum of `f64` values.
///
/// Equivalent to `pairwise_reduce(xs, |a, b| a + b, 0.0)` but specialized to
/// IEEE-754 addition. Compared with a naive sequential fold this both reduces
/// rounding error from `O(n)` to `O(log n)` *and* makes the result independent
/// of the association order (a pure function of the ordered input). An empty
/// slice sums to `+0.0`.
pub fn pairwise_sum(xs: &[f64]) -> f64 {
    pairwise_reduce(xs, |a, b| a + b, 0.0)
}

/// A running pairwise accumulator that consumes successive fixed-size chunks
/// while preserving the exact same tree shape — and hence the exact same
/// floating-point result — as a single whole-slice [`pairwise_reduce`] over the
/// concatenation of all chunks.
///
/// # How the streaming invariance is achieved
///
/// The naive approach of "reduce each chunk, then combine the partials" would
/// make the tree shape depend on chunk boundaries, breaking contract point (3).
/// Instead, the accumulator maintains a *forest of completed subtree partials*,
/// each tagged with the number of leaf elements it covers. Partials are merged
/// only when two adjacent subtrees have equal leaf-counts that are
/// power-of-two multiples of [`BASE_CHUNK`] — exactly the merges the recursive
/// [`reduce_range`] tree would perform. This makes the resulting association
/// order identical to the whole-slice tree, regardless of how the input was
/// sliced into chunks.
///
/// The implementation buffers incoming elements into base blocks of
/// [`BASE_CHUNK`]; each completed base block becomes a partial of weight
/// `BASE_CHUNK`, and equal-weight adjacent partials cascade-merge. A final
/// [`StreamingPairwise::finish`] folds the remaining (possibly unequal-weight)
/// forest — including a short trailing block — in the same right-leaning order
/// the recursive tree uses for a non-power-of-two tail.
pub struct StreamingPairwise<T, F>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    combine: F,
    identity: T,
    /// Buffer of leaf elements not yet sealed into a base block.
    buf: Vec<T>,
    /// Stack of completed subtree partials, each with its leaf-count weight.
    /// Invariant: weights are strictly decreasing from bottom to top, and each
    /// is a power-of-two multiple of `BASE_CHUNK` (except possibly after a
    /// short final block is pushed during `finish`).
    forest: Vec<(usize, T)>,
}

impl<T, F> StreamingPairwise<T, F>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    /// Create an empty streaming accumulator.
    pub fn new(combine: F, identity: T) -> Self {
        Self {
            combine,
            identity,
            buf: Vec::with_capacity(BASE_CHUNK),
            forest: Vec::new(),
        }
    }

    /// Push a single leaf element into the running tree.
    pub fn push(&mut self, x: T) {
        self.buf.push(x);
        if self.buf.len() == BASE_CHUNK {
            let block = reduce_block(self.buf[0], &self.buf[1..], &self.combine);
            self.buf.clear();
            self.absorb(BASE_CHUNK, block);
        }
    }

    /// Push a contiguous chunk of leaf elements, in order.
    pub fn extend_from_slice(&mut self, chunk: &[T]) {
        for &x in chunk {
            self.push(x);
        }
    }

    /// Merge a completed subtree partial of the given leaf-count `weight` into
    /// the forest, cascading equal-weight merges so the association order
    /// matches the recursive whole-slice tree.
    fn absorb(&mut self, weight: usize, value: T) {
        let mut w = weight;
        let mut v = value;
        // While the top of the forest has the same weight, it is the left
        // sibling of the subtree we are inserting; combine them into a parent
        // of double weight. This reproduces the balanced power-of-two left
        // subtrees that `left_split` builds.
        while let Some(&(top_w, top_v)) = self.forest.last() {
            if top_w == w {
                self.forest.pop();
                v = (self.combine)(top_v, v);
                w = w.saturating_mul(2);
            } else {
                break;
            }
        }
        self.forest.push((w, v));
    }

    /// Finish the stream, returning the bit-identical result of reducing the
    /// whole concatenated input with [`pairwise_reduce`].
    pub fn finish(mut self) -> T {
        // Seal any trailing partial base block (length in 1..BASE_CHUNK).
        if !self.buf.is_empty() {
            let tail = reduce_block(self.buf[0], &self.buf[1..], &self.combine);
            let tail_w = self.buf.len();
            self.buf.clear();
            self.forest.push((tail_w, tail));
        }
        // Fold the forest. The forest is laid out left-to-right as a sequence
        // of subtrees with non-increasing weights (the trailing short block may
        // tie or be smaller). The recursive `reduce_range` combines a balanced
        // left subtree with the (smaller, right-leaning) remainder; folding the
        // forest from the right reproduces exactly that association: each parent
        // is `combine(left_partial, accumulated_right)`.
        let mut iter = self.forest.into_iter().rev();
        match iter.next() {
            None => self.identity,
            Some((_, mut acc)) => {
                for (_, left) in iter {
                    acc = (self.combine)(left, acc);
                }
                acc
            }
        }
    }
}

/// Convenience: reduce an iterator of fixed-size chunks through a streaming
/// pairwise tree, returning the bit-identical whole-slice result.
///
/// `chunks` may be sliced arbitrarily — the result depends only on the ordered
/// concatenation of all elements, per the determinism contract.
pub fn pairwise_reduce_chunked<'a, T, F, I>(chunks: I, combine: F, identity: T) -> T
where
    T: Copy + 'a,
    F: Fn(T, T) -> T,
    I: IntoIterator<Item = &'a [T]>,
{
    let mut acc = StreamingPairwise::new(combine, identity);
    for chunk in chunks {
        acc.extend_from_slice(chunk);
    }
    acc.finish()
}

/// Streaming `f64` pairwise sum over an iterator of chunks. Bit-identical to
/// [`pairwise_sum`] over the concatenation of all chunks.
pub fn pairwise_sum_chunked<'a, I>(chunks: I) -> f64
where
    I: IntoIterator<Item = &'a [f64]>,
{
    pairwise_reduce_chunked(chunks, |a, b| a + b, 0.0)
}
